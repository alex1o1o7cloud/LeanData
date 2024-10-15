import Mathlib

namespace NUMINAMATH_GPT_polynomial_divisibility_l164_16442

theorem polynomial_divisibility (C D : ℝ) (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C * x + D = 0) :
  C + D = 2 := 
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l164_16442


namespace NUMINAMATH_GPT_no_integer_n_exists_l164_16411

theorem no_integer_n_exists : ∀ (n : ℤ), n ^ 2022 - 2 * n ^ 2021 + 3 * n ^ 2019 ≠ 2020 :=
by sorry

end NUMINAMATH_GPT_no_integer_n_exists_l164_16411


namespace NUMINAMATH_GPT_alice_bob_same_point_after_3_turns_l164_16449

noncomputable def alice_position (t : ℕ) : ℕ := (15 + 4 * t) % 15

noncomputable def bob_position (t : ℕ) : ℕ :=
  if t < 2 then 15
  else (15 - 11 * (t - 2)) % 15

theorem alice_bob_same_point_after_3_turns :
  ∃ t, t = 3 ∧ alice_position t = bob_position t :=
by
  exists 3
  simp only [alice_position, bob_position]
  norm_num
  -- Alice's position after 3 turns
  -- alice_position 3 = (15 + 4 * 3) % 15
  -- bob_position 3 = (15 - 11 * (3 - 2)) % 15
  -- Therefore,
  -- alice_position 3 = 12
  -- bob_position 3 = 12
  sorry

end NUMINAMATH_GPT_alice_bob_same_point_after_3_turns_l164_16449


namespace NUMINAMATH_GPT_length_of_train_l164_16419

-- Conditions
variable (L E T : ℝ)
axiom h1 : 300 * E = L + 300 * T
axiom h2 : 90 * E = L - 90 * T

-- The statement to be proved
theorem length_of_train : L = 200 * E :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l164_16419


namespace NUMINAMATH_GPT_rectangle_width_l164_16489

theorem rectangle_width
  (L W : ℝ)
  (h1 : W = L + 2)
  (h2 : 2 * L + 2 * W = 16) :
  W = 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l164_16489


namespace NUMINAMATH_GPT_negation_of_existence_statement_l164_16491

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 8 * x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8 * x + 18 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_statement_l164_16491


namespace NUMINAMATH_GPT_deer_families_stayed_l164_16458

-- Define the initial number of deer families
def initial_deer_families : ℕ := 79

-- Define the number of deer families that moved out
def moved_out_deer_families : ℕ := 34

-- The theorem stating how many deer families stayed
theorem deer_families_stayed : initial_deer_families - moved_out_deer_families = 45 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_deer_families_stayed_l164_16458


namespace NUMINAMATH_GPT_downstream_speed_is_45_l164_16457

-- Define the conditions
def upstream_speed := 35 -- The man can row upstream at 35 kmph
def still_water_speed := 40 -- The speed of the man in still water is 40 kmph

-- Define the speed of the stream based on the given conditions
def stream_speed := still_water_speed - upstream_speed 

-- Define the speed of the man rowing downstream
def downstream_speed := still_water_speed + stream_speed

-- The assertion to prove
theorem downstream_speed_is_45 : downstream_speed = 45 := by
  sorry

end NUMINAMATH_GPT_downstream_speed_is_45_l164_16457


namespace NUMINAMATH_GPT_find_third_side_l164_16460

theorem find_third_side
  (cubes : ℕ) (cube_volume : ℚ) (side1 side2 : ℚ)
  (fits : cubes = 24) (vol_cube : cube_volume = 27)
  (dim1 : side1 = 8) (dim2 : side2 = 9) :
  (side1 * side2 * (cube_volume * cubes) / (side1 * side2)) = 9 := by
  sorry

end NUMINAMATH_GPT_find_third_side_l164_16460


namespace NUMINAMATH_GPT_total_fires_l164_16415

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end NUMINAMATH_GPT_total_fires_l164_16415


namespace NUMINAMATH_GPT_exist_unique_rectangular_prism_Q_l164_16477

variable (a b c : ℝ) (h_lt : a < b ∧ b < c)
variable (x y z : ℝ) (hx_lt : x < y ∧ y < z ∧ z < a)

theorem exist_unique_rectangular_prism_Q :
  (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) ∧ (x < y ∧ y < z ∧ z < a) → 
  ∃! x y z, (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) :=
sorry

end NUMINAMATH_GPT_exist_unique_rectangular_prism_Q_l164_16477


namespace NUMINAMATH_GPT_find_value_of_expr_l164_16414

variables (a b : ℝ)

def condition1 : Prop := a^2 + a * b = -2
def condition2 : Prop := b^2 - 3 * a * b = -3

theorem find_value_of_expr (h1 : condition1 a b) (h2 : condition2 a b) : a^2 + 4 * a * b - b^2 = 1 :=
sorry

end NUMINAMATH_GPT_find_value_of_expr_l164_16414


namespace NUMINAMATH_GPT_correct_statement_l164_16475

noncomputable def f (x : ℝ) := Real.exp x - x
noncomputable def g (x : ℝ) := Real.log x + x + 1

def proposition_p := ∀ x : ℝ, f x > 0
def proposition_q := ∃ x0 : ℝ, 0 < x0 ∧ g x0 = 0

theorem correct_statement : (proposition_p ∧ proposition_q) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l164_16475


namespace NUMINAMATH_GPT_marcy_total_time_l164_16480

theorem marcy_total_time 
    (petting_time : ℝ)
    (fraction_combing : ℝ)
    (H1 : petting_time = 12)
    (H2 : fraction_combing = 1/3) :
    (petting_time + (fraction_combing * petting_time) = 16) :=
  sorry

end NUMINAMATH_GPT_marcy_total_time_l164_16480


namespace NUMINAMATH_GPT_find_original_speed_l164_16433

theorem find_original_speed :
  ∀ (v T : ℝ), 
    (300 = 212 + 88) →
    (T + 2/3 = 212 / v + 88 / (v - 50)) →
    v = 110 :=
by
  intro v T h_dist h_trip
  sorry

end NUMINAMATH_GPT_find_original_speed_l164_16433


namespace NUMINAMATH_GPT_gcd_g_x_1155_l164_16423

def g (x : ℕ) := (4 * x + 5) * (5 * x + 3) * (6 * x + 7) * (3 * x + 11)

theorem gcd_g_x_1155 (x : ℕ) (h : x % 18711 = 0) : Nat.gcd (g x) x = 1155 := by
  sorry

end NUMINAMATH_GPT_gcd_g_x_1155_l164_16423


namespace NUMINAMATH_GPT_max_value_proof_l164_16484

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_max_value_proof_l164_16484


namespace NUMINAMATH_GPT_joey_speed_return_l164_16490

/--
Joey the postman takes 1 hour to run a 5-mile-long route every day, delivering packages along the way.
On his return, he must climb a steep hill covering 3 miles and then navigate a rough, muddy terrain spanning 2 miles.
If the average speed of the entire round trip is 8 miles per hour, prove that the speed with which Joey returns along the path is 20 miles per hour.
-/
theorem joey_speed_return
  (dist_out : ℝ := 5)
  (time_out : ℝ := 1)
  (dist_hill : ℝ := 3)
  (dist_terrain : ℝ := 2)
  (avg_speed_round : ℝ := 8)
  (total_dist : ℝ := dist_out * 2)
  (total_time : ℝ := total_dist / avg_speed_round)
  (time_return : ℝ := total_time - time_out)
  (dist_return : ℝ := dist_hill + dist_terrain) :
  (dist_return / time_return = 20) := 
sorry

end NUMINAMATH_GPT_joey_speed_return_l164_16490


namespace NUMINAMATH_GPT_problem_I_problem_II_l164_16401

def f (x : ℝ) : ℝ := abs (x - 1)

theorem problem_I (x : ℝ) : f (2 * x) + f (x + 4) ≥ 8 ↔ x ≤ -10 / 3 ∨ x ≥ 2 := by
  sorry

variable {a b : ℝ}
theorem problem_II (ha : abs a < 1) (hb : abs b < 1) (h_neq : a ≠ 0) : 
  (abs (a * b - 1) / abs a) > abs ((b / a) - 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l164_16401


namespace NUMINAMATH_GPT_animal_count_l164_16424

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end NUMINAMATH_GPT_animal_count_l164_16424


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l164_16447

theorem arithmetic_sequence_common_difference (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 3 = 4) (h₂ : S 3 = 3)
  (h₃ : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h₄ : ∀ n, a n = a 1 + (n - 1) * d) :
  ∃ d, d = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l164_16447


namespace NUMINAMATH_GPT_rectangle_area_increase_l164_16440

theorem rectangle_area_increase :
  let l := 33.333333333333336
  let b := l / 2
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 4
  let A_new := l_new * b_new
  A_new - A_original = 30 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l164_16440


namespace NUMINAMATH_GPT_books_about_fish_l164_16487

theorem books_about_fish (F : ℕ) (spent : ℕ) (cost_whale_books : ℕ) (cost_magazines : ℕ) (cost_fish_books_per_unit : ℕ) (whale_books : ℕ) (magazines : ℕ) :
  whale_books = 9 →
  magazines = 3 →
  cost_whale_books = 11 →
  cost_magazines = 1 →
  spent = 179 →
  99 + 11 * F + 3 = spent → F = 7 :=
by
  sorry

end NUMINAMATH_GPT_books_about_fish_l164_16487


namespace NUMINAMATH_GPT_prove_M_l164_16478

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem prove_M :
  M = {1} :=
by
  sorry

end NUMINAMATH_GPT_prove_M_l164_16478


namespace NUMINAMATH_GPT_inequality_solution_set_l164_16473

theorem inequality_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l164_16473


namespace NUMINAMATH_GPT_remainder_of_pencils_l164_16459

def number_of_pencils : ℕ := 13254839
def packages : ℕ := 7

theorem remainder_of_pencils :
  number_of_pencils % packages = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_of_pencils_l164_16459


namespace NUMINAMATH_GPT_tan_ratio_l164_16405

theorem tan_ratio (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5 / 8)
  (h2 : Real.sin (a - b) = 1 / 4) : 
  Real.tan a / Real.tan b = 7 / 3 := 
sorry

end NUMINAMATH_GPT_tan_ratio_l164_16405


namespace NUMINAMATH_GPT_difference_of_scores_correct_l164_16454

-- Define the parameters
def num_innings : ℕ := 46
def batting_avg : ℕ := 63
def highest_score : ℕ := 248
def reduced_avg : ℕ := 58
def excluded_innings : ℕ := num_innings - 2

-- Necessary calculations
def total_runs := batting_avg * num_innings
def reduced_total_runs := reduced_avg * excluded_innings
def sum_highest_lowest := total_runs - reduced_total_runs
def lowest_score := sum_highest_lowest - highest_score

-- The correct answer to prove
def expected_difference := highest_score - lowest_score
def correct_answer := 150

-- Define the proof problem
theorem difference_of_scores_correct :
  expected_difference = correct_answer := by
  sorry

end NUMINAMATH_GPT_difference_of_scores_correct_l164_16454


namespace NUMINAMATH_GPT_solve_some_number_l164_16426

theorem solve_some_number (n : ℝ) (h : (n * 10) / 100 = 0.032420000000000004) : n = 0.32420000000000004 :=
by
  -- The proof steps are omitted with 'sorry' here.
  sorry

end NUMINAMATH_GPT_solve_some_number_l164_16426


namespace NUMINAMATH_GPT_cow_manure_plant_height_l164_16416

theorem cow_manure_plant_height
  (control_plant_height : ℝ)
  (bone_meal_ratio : ℝ)
  (cow_manure_ratio : ℝ)
  (h1 : control_plant_height = 36)
  (h2 : bone_meal_ratio = 1.25)
  (h3 : cow_manure_ratio = 2) :
  (control_plant_height * bone_meal_ratio * cow_manure_ratio) = 90 :=
sorry

end NUMINAMATH_GPT_cow_manure_plant_height_l164_16416


namespace NUMINAMATH_GPT_female_members_count_l164_16461

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_female_members_count_l164_16461


namespace NUMINAMATH_GPT_value_modulo_7_l164_16421

theorem value_modulo_7 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := 
  by 
  sorry

end NUMINAMATH_GPT_value_modulo_7_l164_16421


namespace NUMINAMATH_GPT_ages_of_three_persons_l164_16409

theorem ages_of_three_persons (y m e : ℕ) 
  (h1 : e = m + 16)
  (h2 : m = y + 8)
  (h3 : e - 6 = 3 * (y - 6))
  (h4 : e - 6 = 2 * (m - 6)) :
  y = 18 ∧ m = 26 ∧ e = 42 := 
by 
  sorry

end NUMINAMATH_GPT_ages_of_three_persons_l164_16409


namespace NUMINAMATH_GPT_part1_arithmetic_sequence_part2_general_term_part3_max_m_l164_16420

-- Part (1)
theorem part1_arithmetic_sequence (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : a 1 + a 2 = 2 * m) : 
  m = 9 / 8 := 
sorry

-- Part (2)
theorem part2_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2) : 
  ∀ n, a n = 8 ^ (1 - 2 ^ (n - 1)) := 
sorry

-- Part (3)
theorem part3_max_m (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : ∀ n, a n < 4) : 
  m ≤ 2 := 
sorry

end NUMINAMATH_GPT_part1_arithmetic_sequence_part2_general_term_part3_max_m_l164_16420


namespace NUMINAMATH_GPT_sequence_odd_l164_16428

theorem sequence_odd (a : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = 7)
  (hr : ∀ n ≥ 2, -1 < (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ∧ (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ≤ 1) :
  ∀ n > 1, Odd (a n) := 
  sorry

end NUMINAMATH_GPT_sequence_odd_l164_16428


namespace NUMINAMATH_GPT_median_to_longest_side_l164_16486

theorem median_to_longest_side
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26)
  (h4 : a^2 + b^2 = c^2) :
  ∃ m : ℕ, m = c / 2 ∧ m = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_median_to_longest_side_l164_16486


namespace NUMINAMATH_GPT_evaluate_expression_l164_16456

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l164_16456


namespace NUMINAMATH_GPT_intersection_area_correct_l164_16474

noncomputable def intersection_area (XY YE FX EX FY : ℕ) : ℚ :=
  if XY = 12 ∧ YE = FX ∧ YE = 15 ∧ EX = FY ∧ EX = 20 then
    18
  else
    0

theorem intersection_area_correct {XY YE FX EX FY : ℕ} (h1 : XY = 12) (h2 : YE = FX) (h3 : YE = 15) (h4 : EX = FY) (h5 : EX = 20) : 
  intersection_area XY YE FX EX FY = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_area_correct_l164_16474


namespace NUMINAMATH_GPT_max_value_y_l164_16434

noncomputable def y (x : ℝ) : ℝ := x * (3 - 2 * x)

theorem max_value_y : ∃ x, 0 < x ∧ x < (3:ℝ) / 2 ∧ y x = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_value_y_l164_16434


namespace NUMINAMATH_GPT_work_completion_l164_16476

theorem work_completion (A : ℝ) (B : ℝ) (work_duration : ℝ) (total_days : ℝ) (B_days : ℝ) :
  B_days = 28 ∧ total_days = 8 ∧ (A * 2 + (A * 6 + B * 6) = work_duration) →
  A = 84 / 11 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l164_16476


namespace NUMINAMATH_GPT_number_of_machines_sold_l164_16418

-- Define the parameters and conditions given in the problem
def commission_of_first_150 (sale_price : ℕ) : ℕ := 150 * (sale_price * 3 / 100)
def commission_of_next_100 (sale_price : ℕ) : ℕ := 100 * (sale_price * 4 / 100)
def commission_of_after_250 (sale_price : ℕ) (x : ℕ) : ℕ := x * (sale_price * 5 / 100)

-- Define the total commission using these commissions
def total_commission (x : ℕ) : ℕ :=
  commission_of_first_150 10000 + 
  commission_of_next_100 9500 + 
  commission_of_after_250 9000 x

-- The main statement we want to prove
theorem number_of_machines_sold (x : ℕ) (total_commission : ℕ) : x = 398 ↔ total_commission = 150000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_machines_sold_l164_16418


namespace NUMINAMATH_GPT_length_of_lawn_l164_16446

-- Definitions based on conditions
def area_per_bag : ℝ := 250
def width : ℝ := 36
def num_bags : ℝ := 4
def extra_area : ℝ := 208

-- Statement to prove
theorem length_of_lawn :
  (num_bags * area_per_bag + extra_area) / width = 33.56 := by
  sorry

end NUMINAMATH_GPT_length_of_lawn_l164_16446


namespace NUMINAMATH_GPT_students_playing_both_l164_16427

theorem students_playing_both (T F L N B : ℕ)
  (hT : T = 39)
  (hF : F = 26)
  (hL : L = 20)
  (hN : N = 10)
  (hTotal : (F + L - B) + N = T) :
  B = 17 :=
by
  sorry

end NUMINAMATH_GPT_students_playing_both_l164_16427


namespace NUMINAMATH_GPT_number_of_pounds_colombian_beans_l164_16455

def cost_per_pound_colombian : ℝ := 5.50
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def desired_cost_per_pound : ℝ := 4.60
noncomputable def amount_colombian_beans (C : ℝ) : Prop := 
  let P := total_weight - C
  cost_per_pound_colombian * C + cost_per_pound_peruvian * P = desired_cost_per_pound * total_weight

theorem number_of_pounds_colombian_beans : ∃ C, amount_colombian_beans C ∧ C = 11.2 :=
sorry

end NUMINAMATH_GPT_number_of_pounds_colombian_beans_l164_16455


namespace NUMINAMATH_GPT_ratio_of_democrats_l164_16436

theorem ratio_of_democrats (F M : ℕ) 
  (h1 : F + M = 990) 
  (h2 : (1 / 2 : ℚ) * F = 165) 
  (h3 : (1 / 4 : ℚ) * M = 165) : 
  (165 + 165) / 990 = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_democrats_l164_16436


namespace NUMINAMATH_GPT_eight_digit_number_min_max_l164_16443

theorem eight_digit_number_min_max (Amin Amax B : ℕ) 
  (hAmin: Amin = 14444446) 
  (hAmax: Amax = 99999998) 
  (hB_coprime: Nat.gcd B 12 = 1) 
  (hB_length: 44444444 < B) 
  (h_digits: ∀ (b : ℕ), b < 10 → ∃ (A : ℕ), A = 10^7 * b + (B - b) / 10 ∧ A < 100000000) :
  (∃ b, Amin = 10^7 * b + (44444461 - b) / 10 ∧ Nat.gcd 44444461 12 = 1 ∧ 44444444 < 44444461) ∧
  (∃ b, Amax = 10^7 * b + (999999989 - b) / 10 ∧ Nat.gcd 999999989 12 = 1 ∧ 44444444 < 999999989) :=
  sorry

end NUMINAMATH_GPT_eight_digit_number_min_max_l164_16443


namespace NUMINAMATH_GPT_geom_seq_sum_l164_16403

variable (a : ℕ → ℝ) (r : ℝ) (a1 a4 : ℝ)

theorem geom_seq_sum :
  (∀ n : ℕ, a (n + 1) = a n * r) → r = 2 → a 2 + a 3 = 4 → a 1 + a 4 = 6 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_l164_16403


namespace NUMINAMATH_GPT_vector_cross_product_coordinates_l164_16472

variables (a1 a2 a3 b1 b2 b3 : ℝ)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem vector_cross_product_coordinates :
  cross_product (a1, a2, a3) (b1, b2, b3) = 
    (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1) :=
by
sorry

end NUMINAMATH_GPT_vector_cross_product_coordinates_l164_16472


namespace NUMINAMATH_GPT_fair_prize_division_l164_16404

theorem fair_prize_division (eq_chance : ∀ (game : ℕ), 0.5 ≤ 1 ∧ 1 ≤ 0.5)
  (first_to_six : ∀ (p1_wins p2_wins : ℕ), (p1_wins = 6 ∨ p2_wins = 6) → (p1_wins + p2_wins) ≤ 11)
  (current_status : 5 + 3 = 8) :
  (7 : ℝ) / 8 = 7 / (8 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_fair_prize_division_l164_16404


namespace NUMINAMATH_GPT_boxes_produced_by_machine_A_in_10_minutes_l164_16453

-- Define the variables and constants involved
variables {A : ℕ} -- number of boxes machine A produces in 10 minutes

-- Define the condition that machine B produces 4*A boxes in 10 minutes
def boxes_produced_by_machine_B_in_10_minutes := 4 * A

-- Define the combined production working together for 20 minutes
def combined_production_in_20_minutes := 10 * A

-- Statement to prove that machine A produces A boxes in 10 minutes
theorem boxes_produced_by_machine_A_in_10_minutes :
  ∀ (boxes_produced_by_machine_B_in_10_minutes : ℕ) (combined_production_in_20_minutes : ℕ),
    boxes_produced_by_machine_B_in_10_minutes = 4 * A →
    combined_production_in_20_minutes = 10 * A →
    A = A :=
by
  intros _ _ hB hC
  sorry

end NUMINAMATH_GPT_boxes_produced_by_machine_A_in_10_minutes_l164_16453


namespace NUMINAMATH_GPT_students_taking_statistics_l164_16469

-- Definitions based on conditions
def total_students := 89
def history_students := 36
def history_or_statistics := 59
def history_not_statistics := 27

-- The proof problem
theorem students_taking_statistics : ∃ S : ℕ, S = 32 ∧
  ((history_students - history_not_statistics) + S - (history_students - history_not_statistics)) = history_or_statistics :=
by
  use 32
  sorry

end NUMINAMATH_GPT_students_taking_statistics_l164_16469


namespace NUMINAMATH_GPT_solve_inequality_l164_16488

theorem solve_inequality :
  {x : ℝ | -3 * x^2 + 5 * x + 4 < 0} = {x : ℝ | x < 3 / 4} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l164_16488


namespace NUMINAMATH_GPT_clerk_daily_salary_l164_16406

theorem clerk_daily_salary (manager_salary : ℝ) (num_managers num_clerks : ℕ) (total_salary : ℝ) (clerk_salary : ℝ)
  (h1 : manager_salary = 5)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16) :
  clerk_salary = 2 :=
by
  sorry

end NUMINAMATH_GPT_clerk_daily_salary_l164_16406


namespace NUMINAMATH_GPT_factor_expression_l164_16407

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end NUMINAMATH_GPT_factor_expression_l164_16407


namespace NUMINAMATH_GPT_cos_value_in_second_quadrant_l164_16448

theorem cos_value_in_second_quadrant {B : ℝ} (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = - (12 / 13) :=
sorry

end NUMINAMATH_GPT_cos_value_in_second_quadrant_l164_16448


namespace NUMINAMATH_GPT_distance_and_ratio_correct_l164_16493

noncomputable def distance_and_ratio (a : ℝ) : ℝ × ℝ :=
  let dist : ℝ := a / Real.sqrt 3
  let ratio : ℝ := 1 / 2
  ⟨dist, ratio⟩

theorem distance_and_ratio_correct (a : ℝ) :
  distance_and_ratio a = (a / Real.sqrt 3, 1 / 2) := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_distance_and_ratio_correct_l164_16493


namespace NUMINAMATH_GPT_perpendicular_lines_b_eq_neg9_l164_16413

-- Definitions for the conditions.
def eq1 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def eq2 (b x y : ℝ) : Prop := b * x + 3 * y + 4 = 0

-- The problem statement
theorem perpendicular_lines_b_eq_neg9 (b : ℝ) : 
  (∀ x y, eq1 x y → eq2 b x y) ∧ (∀ x y, eq2 b x y → eq1 x y) → b = -9 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_b_eq_neg9_l164_16413


namespace NUMINAMATH_GPT_parallelogram_isosceles_angles_l164_16429

def angle_sum_isosceles_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = b ∨ b = c ∨ a = c)

theorem parallelogram_isosceles_angles :
  ∀ (A B C D P : Type) (AB BC CD DA BD : ℝ)
    (angle_DAB angle_BCD angle_ABC angle_CDA angle_ABP angle_BAP angle_PBD angle_BDP angle_CBD angle_BCD : ℝ),
  AB ≠ BC →
  angle_DAB = 72 →
  angle_BCD = 72 →
  angle_ABC = 108 →
  angle_CDA = 108 →
  angle_sum_isosceles_triangle angle_ABP angle_BAP 108 →
  angle_sum_isosceles_triangle 72 72 angle_BDP →
  angle_sum_isosceles_triangle 108 36 36 →
  ∃! (ABP BPD BCD : Type),
   (angle_ABP = 36 ∧ angle_BAP = 36 ∧ angle_PBA = 108) ∧
   (angle_PBD = 72 ∧ angle_PDB = 72 ∧ angle_BPD = 36) ∧
   (angle_CBD = 108 ∧ angle_BCD = 36 ∧ angle_BDC = 36) :=
sorry

end NUMINAMATH_GPT_parallelogram_isosceles_angles_l164_16429


namespace NUMINAMATH_GPT_sum_octal_eq_1021_l164_16408

def octal_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let r1 := n / 10
  let d1 := r1 % 10
  let r2 := r1 / 10
  let d2 := r2 % 10
  (d2 * 64) + (d1 * 8) + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let d0 := n % 8
  let r1 := n / 8
  let d1 := r1 % 8
  let r2 := r1 / 8
  let d2 := r2 % 8
  d2 * 100 + d1 * 10 + d0

theorem sum_octal_eq_1021 :
  decimal_to_octal (octal_to_decimal 642 + octal_to_decimal 157) = 1021 := by
  sorry

end NUMINAMATH_GPT_sum_octal_eq_1021_l164_16408


namespace NUMINAMATH_GPT_union_M_N_l164_16482

def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_M_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} := 
sorry

end NUMINAMATH_GPT_union_M_N_l164_16482


namespace NUMINAMATH_GPT_binomial_sum_l164_16483

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum (n : ℤ) (h1 : binomial 25 n.natAbs + binomial 25 12 = binomial 26 13 ∧ n ≥ 0) : 
    (n = 12 ∨ n = 13) → n.succ + n = 25 := 
    sorry

end NUMINAMATH_GPT_binomial_sum_l164_16483


namespace NUMINAMATH_GPT_linear_regression_decrease_l164_16497

theorem linear_regression_decrease (x : ℝ) (y : ℝ) (h : y = 2 - 1.5 * x) : 
  y = 2 - 1.5 * (x + 1) -> (y - (2 - 1.5 * (x +1))) = -1.5 :=
by
  sorry

end NUMINAMATH_GPT_linear_regression_decrease_l164_16497


namespace NUMINAMATH_GPT_infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l164_16441

theorem infinite_n_square_plus_one_divides_factorial :
  ∃ (infinitely_many n : ℕ), (n^2 + 1) ∣ (n!) := sorry

theorem infinite_n_square_plus_one_not_divide_factorial :
  ∃ (infinitely_many n : ℕ), ¬((n^2 + 1) ∣ (n!)) := sorry

end NUMINAMATH_GPT_infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l164_16441


namespace NUMINAMATH_GPT_souvenir_prices_total_profit_l164_16498

variables (x y m n : ℝ)

-- Conditions for the first part
def conditions_part1 : Prop :=
  7 * x + 8 * y = 380 ∧
  10 * x + 6 * y = 380

-- Result for the first part
def result_part1 : Prop :=
  x = 20 ∧ y = 30

-- Conditions for the second part
def conditions_part2 : Prop :=
  m + n = 40 ∧
  20 * m + 30 * n = 900 

-- Result for the second part
def result_part2 : Prop :=
  30 * 5 + 10 * 7 = 220

theorem souvenir_prices (x y : ℝ) (h : conditions_part1 x y) : result_part1 x y :=
by { sorry }

theorem total_profit (m n : ℝ) (h : conditions_part2 m n) : result_part2 :=
by { sorry }

end NUMINAMATH_GPT_souvenir_prices_total_profit_l164_16498


namespace NUMINAMATH_GPT_smallest_n_for_isosceles_trapezoid_coloring_l164_16485

def isIsoscelesTrapezoid (a b c d : ℕ) : Prop :=
  -- conditions to check if vertices a, b, c, d form an isosceles trapezoid in a regular n-gon
  sorry  -- definition of an isosceles trapezoid

def vertexColors (n : ℕ) : Fin n → Fin 3 :=
  sorry  -- vertex coloring function

theorem smallest_n_for_isosceles_trapezoid_coloring :
  ∃ n : ℕ, (∀ (vertices : Fin n → Fin 3), ∃ (a b c d : Fin n),
    vertexColors n a = vertexColors n b ∧
    vertexColors n b = vertexColors n c ∧
    vertexColors n c = vertexColors n d ∧
    isIsoscelesTrapezoid a b c d) ∧ n = 17 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_isosceles_trapezoid_coloring_l164_16485


namespace NUMINAMATH_GPT_value_of_expression_l164_16495

theorem value_of_expression (r s : ℝ) (h₁ : 3 * r^2 - 5 * r - 7 = 0) (h₂ : 3 * s^2 - 5 * s - 7 = 0) : 
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l164_16495


namespace NUMINAMATH_GPT_magnitude_fourth_power_l164_16462

open Complex

noncomputable def complex_magnitude_example : ℂ := 4 + 3 * Real.sqrt 3 * Complex.I

theorem magnitude_fourth_power :
  ‖complex_magnitude_example ^ 4‖ = 1849 := by
  sorry

end NUMINAMATH_GPT_magnitude_fourth_power_l164_16462


namespace NUMINAMATH_GPT_value_of_f_log3_54_l164_16438

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem value_of_f_log3_54
  (h1 : is_odd f)
  (h2 : ∀ x, f (x + 2) = -1 / f x)
  (h3 : ∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) :
  f (Real.log 54 / Real.log 3) = -3 / 2 := sorry

end NUMINAMATH_GPT_value_of_f_log3_54_l164_16438


namespace NUMINAMATH_GPT_not_product_of_two_primes_l164_16492

theorem not_product_of_two_primes (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : ∃ n : ℕ, a^3 + b^3 = n^2) :
  ¬ (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ a + b = p * q) :=
by
  sorry

end NUMINAMATH_GPT_not_product_of_two_primes_l164_16492


namespace NUMINAMATH_GPT_oranges_taken_by_susan_l164_16437

-- Defining the conditions
def original_number_of_oranges_in_box : ℕ := 55
def oranges_left_in_box_after_susan_takes : ℕ := 20

-- Statement to prove:
theorem oranges_taken_by_susan :
  original_number_of_oranges_in_box - oranges_left_in_box_after_susan_takes = 35 :=
by
  sorry

end NUMINAMATH_GPT_oranges_taken_by_susan_l164_16437


namespace NUMINAMATH_GPT_union_of_sets_l164_16400

def setA := {x : ℝ | x^2 < 4}
def setB := {y : ℝ | ∃ x ∈ setA, y = x^2 - 2 * x - 1}

theorem union_of_sets : (setA ∪ setB) = {x : ℝ | -2 ≤ x ∧ x < 7} :=
by sorry

end NUMINAMATH_GPT_union_of_sets_l164_16400


namespace NUMINAMATH_GPT_right_triangle_side_length_l164_16479

theorem right_triangle_side_length (area : ℝ) (side1 : ℝ) (side2 : ℝ) (h_area : area = 8) (h_side1 : side1 = Real.sqrt 10) (h_area_eq : area = 0.5 * side1 * side2) :
  side2 = 1.6 * Real.sqrt 10 :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_side_length_l164_16479


namespace NUMINAMATH_GPT_lisa_interest_earned_l164_16463

/-- Lisa's interest earned after three years from Bank of Springfield's Super High Yield savings account -/
theorem lisa_interest_earned :
  let P := 2000
  let r := 0.02
  let n := 3
  let A := P * (1 + r)^n
  A - P = 122 := by
  sorry

end NUMINAMATH_GPT_lisa_interest_earned_l164_16463


namespace NUMINAMATH_GPT_monogram_count_is_correct_l164_16499

def count_possible_monograms : ℕ :=
  Nat.choose 23 2

theorem monogram_count_is_correct : 
  count_possible_monograms = 253 := 
by 
  -- The proof will show this matches the combination formula calculation
  -- The final proof is left incomplete as per the instructions
  sorry

end NUMINAMATH_GPT_monogram_count_is_correct_l164_16499


namespace NUMINAMATH_GPT_complement_intersection_eq_4_l164_16402

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection_eq_4 (hU : U = {0, 1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4}) :
  ((U \ A) ∩ B) = {4} :=
by {
  -- Proof goes here
  exact sorry
}

end NUMINAMATH_GPT_complement_intersection_eq_4_l164_16402


namespace NUMINAMATH_GPT_compute_fg_neg1_l164_16425

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem compute_fg_neg1 : f (g (-1)) = 3 := by
  sorry

end NUMINAMATH_GPT_compute_fg_neg1_l164_16425


namespace NUMINAMATH_GPT_tunnel_length_l164_16422

/-- A train travels at 80 kmph, enters a tunnel at 5:12 am, and leaves at 5:18 am.
    The length of the train is 1 km. Prove the length of the tunnel is 7 km. -/
theorem tunnel_length 
(speed : ℕ) (enter_time leave_time : ℕ) (train_length : ℕ) 
(h_enter : enter_time = 5 * 60 + 12) 
(h_leave : leave_time = 5 * 60 + 18) 
(h_speed : speed = 80) 
(h_train_length : train_length = 1) 
: ∃ tunnel_length : ℕ, tunnel_length = 7 :=
sorry

end NUMINAMATH_GPT_tunnel_length_l164_16422


namespace NUMINAMATH_GPT_maple_taller_than_pine_l164_16432

theorem maple_taller_than_pine :
  let pine_tree := 24 + 1/4
  let maple_tree := 31 + 2/3
  (maple_tree - pine_tree) = 7 + 5/12 :=
by
  sorry

end NUMINAMATH_GPT_maple_taller_than_pine_l164_16432


namespace NUMINAMATH_GPT_ishas_pencil_initial_length_l164_16430

theorem ishas_pencil_initial_length (l : ℝ) (h1 : l - 4 = 18) : l = 22 :=
by
  sorry

end NUMINAMATH_GPT_ishas_pencil_initial_length_l164_16430


namespace NUMINAMATH_GPT_area_of_right_triangle_l164_16417

theorem area_of_right_triangle (a b c : ℝ) 
  (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 30 :=
by sorry

end NUMINAMATH_GPT_area_of_right_triangle_l164_16417


namespace NUMINAMATH_GPT_find_x_l164_16451

noncomputable section

variable (x : ℝ)
def vector_v : ℝ × ℝ := (x, 4)
def vector_w : ℝ × ℝ := (5, 2)
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let num := (v.1 * w.1 + v.2 * w.2)
  let den := (w.1 * w.1 + w.2 * w.2)
  (num / den * w.1, num / den * w.2)

theorem find_x (h : projection (vector_v x) (vector_w) = (3, 1.2)) : 
  x = 47 / 25 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l164_16451


namespace NUMINAMATH_GPT_cube_edge_length_l164_16452

theorem cube_edge_length (a : ℝ) (base_length : ℝ) (base_width : ℝ) (rise_height : ℝ) 
  (h_conditions : base_length = 20 ∧ base_width = 15 ∧ rise_height = 11.25 ∧ 
                  (base_length * base_width * rise_height) = a^3) : 
  a = 15 := 
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l164_16452


namespace NUMINAMATH_GPT_simplify_fraction_l164_16464

theorem simplify_fraction : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l164_16464


namespace NUMINAMATH_GPT_perimeter_of_nonagon_l164_16439

-- Definitions based on the conditions
def sides := 9
def side_length : ℝ := 2

-- The problem statement in Lean
theorem perimeter_of_nonagon : sides * side_length = 18 := 
by sorry

end NUMINAMATH_GPT_perimeter_of_nonagon_l164_16439


namespace NUMINAMATH_GPT_valid_lineups_l164_16481

def total_players : ℕ := 15
def k : ℕ := 2  -- number of twins
def total_chosen : ℕ := 7
def remaining_players := total_players - k

def nCr (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

def total_choices : ℕ := nCr total_players total_chosen
def restricted_choices : ℕ := nCr remaining_players (total_chosen - k)

theorem valid_lineups : total_choices - restricted_choices = 5148 := by
  sorry

end NUMINAMATH_GPT_valid_lineups_l164_16481


namespace NUMINAMATH_GPT_balance_four_heartsuits_with_five_circles_l164_16435

variables (x y z : ℝ)

-- Given conditions
axiom condition1 : 4 * x + 3 * y = 12 * z
axiom condition2 : 2 * x = y + 3 * z

-- Statement to prove
theorem balance_four_heartsuits_with_five_circles : 4 * y = 5 * z :=
by sorry

end NUMINAMATH_GPT_balance_four_heartsuits_with_five_circles_l164_16435


namespace NUMINAMATH_GPT_circles_intersect_l164_16468

noncomputable def positional_relationship (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : String :=
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  if radius1 + radius2 > d ∧ d > abs (radius1 - radius2) then "Intersecting"
  else if radius1 + radius2 = d then "Externally tangent"
  else if abs (radius1 - radius2) = d then "Internally tangent"
  else "Separate"

theorem circles_intersect :
  positional_relationship (0, 1) (1, 2) 1 2 = "Intersecting" :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_l164_16468


namespace NUMINAMATH_GPT_bouquet_branches_l164_16494

variable (w : ℕ) (b : ℕ)

theorem bouquet_branches :
  (w + b = 7) → 
  (w ≥ 1) → 
  (∀ x y, x ≠ y → (x = w ∨ y = w) → (x = b ∨ y = b)) → 
  (w = 1 ∧ b = 6) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_bouquet_branches_l164_16494


namespace NUMINAMATH_GPT_prove_functions_same_l164_16431

theorem prove_functions_same (u v : ℝ) (huv : u = v) : 
  (u > 1) → (v > 1) → (Real.sqrt ((u + 1) / (u - 1)) = Real.sqrt ((v + 1) / (v - 1))) :=
by
  sorry

end NUMINAMATH_GPT_prove_functions_same_l164_16431


namespace NUMINAMATH_GPT_scooter_gain_percent_l164_16471

def initial_cost : ℝ := 900
def first_repair_cost : ℝ := 150
def second_repair_cost : ℝ := 75
def third_repair_cost : ℝ := 225
def selling_price : ℝ := 1800

theorem scooter_gain_percent :
  let total_cost := initial_cost + first_repair_cost + second_repair_cost + third_repair_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_scooter_gain_percent_l164_16471


namespace NUMINAMATH_GPT_packages_per_hour_A_B_max_A_robots_l164_16496

-- Define the number of packages sorted by each unit of type A and B robots
def packages_by_A_robot (x : ℕ) := x
def packages_by_B_robot (y : ℕ) := y

-- Problem conditions
def cond1 (x y : ℕ) : Prop := 80 * x + 100 * y = 8200
def cond2 (x y : ℕ) : Prop := 50 * x + 50 * y = 4500

-- Part 1: to prove type A and type B robot's packages per hour
theorem packages_per_hour_A_B (x y : ℕ) (h1 : cond1 x y) (h2 : cond2 x y) : x = 40 ∧ y = 50 :=
by sorry

-- Part 2: prove maximum units of type A robots when purchasing 200 robots ensuring not < 9000 packages/hour
def cond3 (m : ℕ) : Prop := 40 * m + 50 * (200 - m) ≥ 9000

theorem max_A_robots (m : ℕ) (h3 : cond3 m) : m ≤ 100 :=
by sorry

end NUMINAMATH_GPT_packages_per_hour_A_B_max_A_robots_l164_16496


namespace NUMINAMATH_GPT_train_speed_l164_16450

theorem train_speed
  (distance_meters : ℝ := 400)
  (time_seconds : ℝ := 12)
  (distance_kilometers : ℝ := distance_meters / 1000)
  (time_hours : ℝ := time_seconds / 3600) :
  distance_kilometers / time_hours = 120 := by
  sorry

end NUMINAMATH_GPT_train_speed_l164_16450


namespace NUMINAMATH_GPT_three_students_with_B_l164_16470

-- Define the students and their statements as propositions
variables (Eva B_Frank B_Gina B_Harry : Prop)

-- Condition 1: Eva said, "If I get a B, then Frank will get a B."
axiom Eva_statement : Eva → B_Frank

-- Condition 2: Frank said, "If I get a B, then Gina will get a B."
axiom Frank_statement : B_Frank → B_Gina

-- Condition 3: Gina said, "If I get a B, then Harry will get a B."
axiom Gina_statement : B_Gina → B_Harry

-- Condition 4: Only three students received a B.
axiom only_three_Bs : (Eva ∧ B_Frank ∧ B_Gina ∧ B_Harry) → False

-- The theorem we need to prove: The three students who received B's are Frank, Gina, and Harry.
theorem three_students_with_B (h_B_Frank : B_Frank) (h_B_Gina : B_Gina) (h_B_Harry : B_Harry) : ¬Eva :=
by
  sorry

end NUMINAMATH_GPT_three_students_with_B_l164_16470


namespace NUMINAMATH_GPT_y_share_l164_16444

theorem y_share (total_amount : ℝ) (x_share y_share z_share : ℝ)
  (hx : x_share = 1) (hy : y_share = 0.45) (hz : z_share = 0.30)
  (h_total : total_amount = 105) :
  (60 * y_share) = 27 :=
by
  have h_cycle : 1 + y_share + z_share = 1.75 := by sorry
  have h_num_cycles : total_amount / 1.75 = 60 := by sorry
  sorry

end NUMINAMATH_GPT_y_share_l164_16444


namespace NUMINAMATH_GPT_base5_division_l164_16465

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end NUMINAMATH_GPT_base5_division_l164_16465


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_sum_of_powers_l164_16467

theorem smallest_prime_divisor_of_sum_of_powers :
  let a := 5
  let b := 7
  let n := 23
  let m := 17
  Nat.minFac (a^n + b^m) = 2 := by
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_of_sum_of_powers_l164_16467


namespace NUMINAMATH_GPT_value_of_a_plus_b_l164_16445

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end NUMINAMATH_GPT_value_of_a_plus_b_l164_16445


namespace NUMINAMATH_GPT_range_a_two_zeros_l164_16412

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- The theorem statement about the range of a
theorem range_a_two_zeros (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 1 ≤ a ∧ a ≤ 5 := sorry

end NUMINAMATH_GPT_range_a_two_zeros_l164_16412


namespace NUMINAMATH_GPT_zero_in_tens_place_l164_16410

variable {A B : ℕ} {m : ℕ}

-- Define the conditions
def condition1 (A : ℕ) (B : ℕ) (m : ℕ) : Prop :=
  ∀ A B : ℕ, ∀ m : ℕ, A * 10^(m+1) + B = 9 * (A * 10^m + B)

theorem zero_in_tens_place (A B : ℕ) (m : ℕ) :
  condition1 A B m → m = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_zero_in_tens_place_l164_16410


namespace NUMINAMATH_GPT_alice_still_needs_to_fold_l164_16466

theorem alice_still_needs_to_fold (total_cranes alice_folds friend_folds remains: ℕ) 
  (h1 : total_cranes = 1000)
  (h2 : alice_folds = total_cranes / 2)
  (h3 : friend_folds = (total_cranes - alice_folds) / 5)
  (h4 : remains = total_cranes - alice_folds - friend_folds) :
  remains = 400 := 
  by
    sorry

end NUMINAMATH_GPT_alice_still_needs_to_fold_l164_16466
