import Mathlib

namespace red_crayons_count_l95_95045

variable (R : ℕ) -- Number of red crayons
variable (B : ℕ) -- Number of blue crayons
variable (Y : ℕ) -- Number of yellow crayons

-- Conditions
axiom h1 : B = R + 5
axiom h2 : Y = 2 * B - 6
axiom h3 : Y = 32

-- Statement to prove
theorem red_crayons_count : R = 14 :=
by
  sorry

end red_crayons_count_l95_95045


namespace petya_no_win_implies_draw_or_lost_l95_95039

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end petya_no_win_implies_draw_or_lost_l95_95039


namespace factorize_quadratic_l95_95506

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l95_95506


namespace number_exceeds_its_3_over_8_part_by_20_l95_95978

theorem number_exceeds_its_3_over_8_part_by_20 (x : ℝ) (h : x = (3 / 8) * x + 20) : x = 32 :=
by
  sorry

end number_exceeds_its_3_over_8_part_by_20_l95_95978


namespace seats_in_16th_row_l95_95653

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem seats_in_16th_row : arithmetic_sequence 5 2 16 = 35 := by
  sorry

end seats_in_16th_row_l95_95653


namespace matrix_solution_correct_l95_95871

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -7/3], ![4, -1/3]]

def v1 : Fin 2 → ℚ := ![4, 0]
def v2 : Fin 2 → ℚ := ![2, 3]

def result1 : Fin 2 → ℚ := ![12, 16]
def result2 : Fin 2 → ℚ := ![-1, 7]

theorem matrix_solution_correct :
  (mulVec N v1 = result1) ∧ 
  (mulVec N v2 = result2) := by
  sorry

end matrix_solution_correct_l95_95871


namespace complement_N_subset_M_l95_95641

-- Definitions for the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- Complement of N in ℝ
def complement_N : Set ℝ := {x | ¬(x < 1 ∨ x ≥ 3)}

-- The theorem stating that complement_N is a subset of M
theorem complement_N_subset_M : complement_N ⊆ M :=
by
  sorry

end complement_N_subset_M_l95_95641


namespace average_speed_last_segment_l95_95323

variable (total_distance : ℕ := 120)
variable (total_minutes : ℕ := 120)
variable (first_segment_minutes : ℕ := 40)
variable (first_segment_speed : ℕ := 50)
variable (second_segment_minutes : ℕ := 40)
variable (second_segment_speed : ℕ := 55)
variable (third_segment_speed : ℕ := 75)

theorem average_speed_last_segment :
  let total_hours := total_minutes / 60
  let average_speed := total_distance / total_hours
  let speed_first_segment := first_segment_speed * (first_segment_minutes / 60)
  let speed_second_segment := second_segment_speed * (second_segment_minutes / 60)
  let speed_third_segment := third_segment_speed * (third_segment_minutes / 60)
  average_speed = (speed_first_segment + speed_second_segment + speed_third_segment) / 3 →
  third_segment_speed = 75 :=
by
  sorry

end average_speed_last_segment_l95_95323


namespace calculation_is_correct_l95_95609

theorem calculation_is_correct : -1^6 + 8 / (-2)^2 - abs (-4 * 3) = -9 := by
  sorry

end calculation_is_correct_l95_95609


namespace bags_filled_on_saturday_l95_95017

-- Definitions of the conditions
def bags_sat (S : ℕ) := S
def bags_sun := 4
def cans_per_bag := 9
def total_cans := 63

-- The statement to prove
theorem bags_filled_on_saturday (S : ℕ) 
  (h : total_cans = (bags_sat S + bags_sun) * cans_per_bag) : 
  S = 3 :=
by sorry

end bags_filled_on_saturday_l95_95017


namespace petya_cannot_win_l95_95042

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end petya_cannot_win_l95_95042


namespace table_height_l95_95726

-- Definitions
def height_of_table (h l x: ℕ): ℕ := h 
def length_of_block (l: ℕ): ℕ := l 
def width_of_block (w x: ℕ): ℕ := x + 6
def overlap_in_first_arrangement (x : ℕ) : ℕ := x 

-- Conditions
axiom h_conditions (h l x: ℕ): 
  (l + h - x = 42) ∧ (x + 6 + h - l = 36)

-- Proof statement
theorem table_height (h l x : ℕ) (h_conditions : (l + h - x = 42) ∧ (x + 6 + h - l = 36)) :
  height_of_table h l x = 36 := sorry

end table_height_l95_95726


namespace complement_of_A_in_U_l95_95404

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 2, 4, 5}

-- Proof statement
theorem complement_of_A_in_U : (U \ A) = {3, 6, 7} := by
  sorry

end complement_of_A_in_U_l95_95404


namespace smallest_inverse_defined_l95_95055

theorem smallest_inverse_defined (n : ℤ) : n = 5 :=
by sorry

end smallest_inverse_defined_l95_95055


namespace horner_evaluation_at_3_l95_95524

def f (x : ℤ) : ℤ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem horner_evaluation_at_3 : f 3 = 328 := by
  sorry

end horner_evaluation_at_3_l95_95524


namespace area_of_circle_l95_95735

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l95_95735


namespace minimum_value_of_K_l95_95005

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x

noncomputable def f_K (K x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

theorem minimum_value_of_K :
  (∀ x > 0, f_K (1 / Real.exp 1) x = f x) → (∃ K : ℝ, K = 1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_K_l95_95005


namespace fraction_of_ripe_oranges_eaten_l95_95440

theorem fraction_of_ripe_oranges_eaten :
  ∀ (total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges : ℕ),
    total_oranges = 96 →
    ripe_oranges = total_oranges / 2 →
    unripe_oranges = total_oranges / 2 →
    eaten_unripe_oranges = unripe_oranges / 8 →
    uneaten_oranges = 78 →
    eaten_ripe_oranges = (total_oranges - uneaten_oranges) - eaten_unripe_oranges →
    (eaten_ripe_oranges : ℚ) / ripe_oranges = 1 / 4 :=
by
  intros total_oranges ripe_oranges unripe_oranges uneaten_oranges eaten_unripe_oranges eaten_ripe_oranges
  intros h_total h_ripe h_unripe h_eaten_unripe h_uneaten h_eaten_ripe
  sorry

end fraction_of_ripe_oranges_eaten_l95_95440


namespace harkamal_total_payment_l95_95349

def grapes_quantity : ℕ := 10
def grapes_rate : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_rate : ℕ := 55

def cost_of_grapes : ℕ := grapes_quantity * grapes_rate
def cost_of_mangoes : ℕ := mangoes_quantity * mangoes_rate

def total_amount_paid : ℕ := cost_of_grapes + cost_of_mangoes

theorem harkamal_total_payment : total_amount_paid = 1195 := by
  sorry

end harkamal_total_payment_l95_95349


namespace campers_afternoon_l95_95567

theorem campers_afternoon (total_campers morning_campers afternoon_campers : ℕ)
  (h1 : total_campers = 60)
  (h2 : morning_campers = 53)
  (h3 : afternoon_campers = total_campers - morning_campers) :
  afternoon_campers = 7 := by
  sorry

end campers_afternoon_l95_95567


namespace initial_music_files_eq_sixteen_l95_95973

theorem initial_music_files_eq_sixteen (M : ℕ) :
  (M + 48 - 30 = 34) → (M = 16) :=
by
  sorry

end initial_music_files_eq_sixteen_l95_95973


namespace no_three_nat_sum_pair_is_pow_of_three_l95_95084

theorem no_three_nat_sum_pair_is_pow_of_three :
  ¬ ∃ (a b c : ℕ) (m n p : ℕ), a + b = 3 ^ m ∧ b + c = 3 ^ n ∧ c + a = 3 ^ p := 
by 
  sorry

end no_three_nat_sum_pair_is_pow_of_three_l95_95084


namespace find_n_l95_95394

theorem find_n (n : ℕ) (h_pos : 0 < n) (h_prime : Nat.Prime (n^4 - 16 * n^2 + 100)) : n = 3 := 
sorry

end find_n_l95_95394


namespace complete_square_solution_l95_95741

-- Define the initial equation 
def equation_to_solve (x : ℝ) : Prop := x^2 - 4 * x = 6

-- Define the transformed equation after completing the square
def transformed_equation (x : ℝ) : Prop := (x - 2)^2 = 10

-- Prove that solving the initial equation using completing the square results in the transformed equation
theorem complete_square_solution : 
  ∀ x : ℝ, equation_to_solve x → transformed_equation x := 
by
  -- Proof will be provided here
  sorry

end complete_square_solution_l95_95741


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95855

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95855


namespace suitable_M_unique_l95_95499

noncomputable def is_suitable_M (M : ℝ) : Prop :=
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (1 + M ≤ a + M / (a * b)) ∨ 
  (1 + M ≤ b + M / (b * c)) ∨ 
  (1 + M ≤ c + M / (c * a))

theorem suitable_M_unique : is_suitable_M (1/2) ∧ 
  (∀ (M : ℝ), is_suitable_M M → M = 1/2) :=
by
  sorry

end suitable_M_unique_l95_95499


namespace inverse_of_5_mod_34_l95_95237

theorem inverse_of_5_mod_34 : ∃ x : ℕ, (5 * x) % 34 = 1 ∧ 0 ≤ x ∧ x < 34 :=
by
  use 7
  have h : (5 * 7) % 34 = 1 := by sorry
  exact ⟨h, by norm_num, by norm_num⟩

end inverse_of_5_mod_34_l95_95237


namespace find_x_l95_95296

theorem find_x (x : ℝ) 
  (h: 3 * x + 6 * x + 2 * x + x = 360) : 
  x = 30 := 
sorry

end find_x_l95_95296


namespace jane_mean_score_l95_95543

-- Define Jane's scores as a list
def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

-- Define the total number of quizzes
def total_quizzes : ℕ := 6

-- Define the sum of Jane's scores
def sum_scores : ℕ := 95 + 88 + 94 + 86 + 92 + 91

-- Define the mean score calculation
def mean_score : ℕ := sum_scores / total_quizzes

-- The theorem to state Jane's mean score
theorem jane_mean_score : mean_score = 91 := by
  -- This theorem statement correctly reflects the mathematical problem provided.
  sorry

end jane_mean_score_l95_95543


namespace adi_baller_prob_l95_95080

theorem adi_baller_prob (a b : ℕ) (p : ℝ) (h_prime: Nat.Prime a) (h_pos_b: 0 < b)
  (h_p: p = (1 / 2) ^ (1 / 35)) : a + b = 37 :=
sorry

end adi_baller_prob_l95_95080


namespace Payton_score_l95_95317

-- Conditions turned into definitions
def num_students := 15
def first_14_avg := 80
def full_class_avg := 81

-- Desired fact to prove
theorem Payton_score :
  ∃ score : ℕ, score = 95 ∧
  (let total_14 := 14 * first_14_avg in
   let total_15 := num_students * full_class_avg in
   score = total_15 - total_14) :=
begin
  have H14 : total_14 = 1120, from rfl,
  have H15 : total_15 = 1215, from rfl,
  use 95,
  split,
  {
    refl,  -- score = 95
  },
  {
    dsimp [total_14, total_15],
    rw [H14, H15],
    refl,
  },
end

end Payton_score_l95_95317


namespace polynomial_product_linear_term_zero_const_six_l95_95649

theorem polynomial_product_linear_term_zero_const_six (a b : ℝ)
  (h1 : (a + 2 * b = 0)) 
  (h2 : b = 6) : (a + b = -6) :=
by
  sorry

end polynomial_product_linear_term_zero_const_six_l95_95649


namespace cos_150_degree_l95_95827

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95827


namespace line_parabola_intersection_l95_95401

noncomputable def intersection_range (m : ℝ) : Prop :=
  ∀ (x : ℝ), x^2 + m * x - 1 = 2 * x - 2 * m → -1 ≤ x ∧ x ≤ 3

theorem line_parabola_intersection (m : ℝ) :
  intersection_range m ↔ -3 / 5 < m ∧ m < 5 :=
by
  sorry

end line_parabola_intersection_l95_95401


namespace problem1_problem2_l95_95533

-- Definitions based on the given conditions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := a^2 + a * b - 1

-- Statement for problem (1)
theorem problem1 (a b : ℝ) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 4 * a^2 + 5 * a * b - 2 * a - 3 :=
by sorry

-- Statement for problem (2)
theorem problem2 (a b : ℝ) (h : ∀ a, A a b - 2 * B a b = k) : 
  b = 2 :=
by sorry

end problem1_problem2_l95_95533


namespace gwen_science_problems_l95_95407

theorem gwen_science_problems (math_problems : ℕ) (finished_problems : ℕ) (remaining_problems : ℕ)
  (h1 : math_problems = 18) (h2 : finished_problems = 24) (h3 : remaining_problems = 5) :
  (finished_problems + remaining_problems - math_problems = 11) :=
by
  sorry

end gwen_science_problems_l95_95407


namespace cubes_with_all_three_faces_l95_95163

theorem cubes_with_all_three_faces (total_cubes red_cubes blue_cubes green_cubes: ℕ) 
  (h_total: total_cubes = 100)
  (h_red: red_cubes = 80)
  (h_blue: blue_cubes = 85)
  (h_green: green_cubes = 75) :
  40 ≤ total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes)) ∧ (total_cubes - ((total_cubes - red_cubes) + (total_cubes - blue_cubes) + (total_cubes - green_cubes))) ≤ 75 :=
by {
  sorry
}

end cubes_with_all_three_faces_l95_95163


namespace cos_150_eq_negative_cos_30_l95_95786

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95786


namespace more_ones_than_twos_in_digital_roots_l95_95452

/-- Define the digital root (i.e., repeated sum of digits until a single digit). -/
def digitalRoot (n : Nat) : Nat :=
  if n == 0 then 0 else 1 + (n - 1) % 9

/-- Statement of the problem: For numbers 1 to 1,000,000, the count of digital root 1 is higher than the count of digital root 2. -/
theorem more_ones_than_twos_in_digital_roots :
  (Finset.filter (fun n => digitalRoot n = 1) (Finset.range 1000000)).card >
  (Finset.filter (fun n => digitalRoot n = 2) (Finset.range 1000000)).card :=
by
  sorry

end more_ones_than_twos_in_digital_roots_l95_95452


namespace cos_150_eq_neg_half_l95_95807

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95807


namespace linear_equation_in_one_variable_proof_l95_95487

noncomputable def is_linear_equation_in_one_variable (eq : String) : Prop :=
  eq = "3x = 2x" ∨ eq = "ax + b = 0"

theorem linear_equation_in_one_variable_proof :
  is_linear_equation_in_one_variable "3x = 2x" ∧ ¬is_linear_equation_in_one_variable "3x - (4 + 3x) = 2"
  ∧ ¬is_linear_equation_in_one_variable "x + y = 1" ∧ ¬is_linear_equation_in_one_variable "x^2 + 1 = 5" :=
by
  sorry

end linear_equation_in_one_variable_proof_l95_95487


namespace cos_150_eq_neg_sqrt3_div_2_l95_95777

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95777


namespace minimum_A_l95_95639

noncomputable def minA : ℝ := (1 + Real.sqrt 2) / 2

theorem minimum_A (x y z w : ℝ) (A : ℝ) 
    (h : xy + 2 * yz + zw ≤ A * (x^2 + y^2 + z^2 + w^2)) :
    A ≥ minA := 
sorry

end minimum_A_l95_95639


namespace avg_daily_distance_third_dog_summer_l95_95369

theorem avg_daily_distance_third_dog_summer :
  ∀ (total_days weekends miles_walked_weekday : ℕ), 
    total_days = 30 → weekends = 8 → miles_walked_weekday = 3 →
    (66 / 30 : ℝ) = 2.2 :=
by
  intros total_days weekends miles_walked_weekday h_total h_weekends h_walked
  -- proof goes here
  sorry

end avg_daily_distance_third_dog_summer_l95_95369


namespace quadratic_function_has_specific_k_l95_95413

theorem quadratic_function_has_specific_k (k : ℤ) :
  (∀ x : ℝ, ∃ y : ℝ, y = (k-1)*x^(k^2-k+2) + k*x - 1) ↔ k = 0 :=
by
  sorry

end quadratic_function_has_specific_k_l95_95413


namespace solve_for_y_in_equation_l95_95325

theorem solve_for_y_in_equation : ∃ y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 :=
by
  use -4
  sorry

end solve_for_y_in_equation_l95_95325


namespace sub_one_inequality_l95_95899

theorem sub_one_inequality (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end sub_one_inequality_l95_95899


namespace identity_function_l95_95510

theorem identity_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ y : ℝ, f y = y :=
by 
  sorry

end identity_function_l95_95510


namespace complete_square_transform_l95_95343

theorem complete_square_transform (x : ℝ) :
  x^2 - 8 * x + 2 = 0 → (x - 4)^2 = 14 :=
by
  intro h
  sorry

end complete_square_transform_l95_95343


namespace recipe_calls_for_nine_cups_of_flour_l95_95682

def cups_of_flour (x : ℕ) := 
  ∃ cups_added_sugar : ℕ, 
    cups_added_sugar = (6 - 4) ∧ 
    x = cups_added_sugar + 7

theorem recipe_calls_for_nine_cups_of_flour : cups_of_flour 9 :=
by
  sorry

end recipe_calls_for_nine_cups_of_flour_l95_95682


namespace minimum_bailing_rate_l95_95945

theorem minimum_bailing_rate
  (distance_from_shore : Real := 1.5)
  (rowing_speed : Real := 3)
  (water_intake_rate : Real := 12)
  (max_water : Real := 45) :
  (distance_from_shore / rowing_speed) * 60 * water_intake_rate - max_water / ((distance_from_shore / rowing_speed) * 60) >= 10.5 :=
by
  -- Provide the units are consistent and the calculations agree with the given numerical data
  sorry

end minimum_bailing_rate_l95_95945


namespace output_correct_l95_95713

-- Define the initial values and assignments
def initial_a : ℕ := 1
def initial_b : ℕ := 2
def initial_c : ℕ := 3

-- Perform the assignments in sequence
def after_c_assignment : ℕ := initial_b
def after_b_assignment : ℕ := initial_a
def after_a_assignment : ℕ := after_c_assignment

-- Final values after all assignments
def final_a := after_a_assignment
def final_b := after_b_assignment
def final_c := after_c_assignment

-- Theorem statement
theorem output_correct :
  final_a = 2 ∧ final_b = 1 ∧ final_c = 2 :=
by {
  -- Proof is omitted
  sorry
}

end output_correct_l95_95713


namespace cos_150_eq_neg_sqrt3_div_2_l95_95804

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95804


namespace percentage_dogs_and_video_games_l95_95376

theorem percentage_dogs_and_video_games (total_students : ℕ)
  (students_dogs_movies : ℕ)
  (students_prefer_dogs : ℕ) :
  total_students = 30 →
  students_dogs_movies = 3 →
  students_prefer_dogs = 18 →
  (students_prefer_dogs - students_dogs_movies) * 100 / total_students = 50 :=
by
  intros h1 h2 h3
  sorry

end percentage_dogs_and_video_games_l95_95376


namespace square_area_in_right_triangle_l95_95450

theorem square_area_in_right_triangle (XY ZC : ℝ) (hXY : XY = 40) (hZC : ZC = 70) : 
  ∃ s : ℝ, s^2 = 2800 ∧ s = (40 * 70) / (XY + ZC) := 
by
  sorry

end square_area_in_right_triangle_l95_95450


namespace volume_is_correct_l95_95874

noncomputable def volume_of_solid : ℝ :=
  let solid := {p : ℝ × ℝ × ℝ | sqrt (p.1^2 + p.2^2) + abs p.3 ≤ 1} in
  volume solid

theorem volume_is_correct :
  volume_of_solid = (2 * real.pi) / 3 :=
sorry

end volume_is_correct_l95_95874


namespace transaction_loss_l95_95760

theorem transaction_loss :
  let house_sale_price := 10000
  let store_sale_price := 15000
  let house_loss_percentage := 0.25
  let store_gain_percentage := 0.25
  let h := house_sale_price / (1 - house_loss_percentage)
  let s := store_sale_price / (1 + store_gain_percentage)
  let total_cost_price := h + s
  let total_selling_price := house_sale_price + store_sale_price
  let difference := total_selling_price - total_cost_price
  difference = -1000 / 3 :=
by
  sorry

end transaction_loss_l95_95760


namespace valid_four_digit_numbers_count_l95_95265

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l95_95265


namespace find_remainder_l95_95518

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end find_remainder_l95_95518


namespace maximize_area_of_sector_l95_95361

noncomputable def area_of_sector (x y : ℝ) : ℝ := (1 / 2) * x * y

theorem maximize_area_of_sector : 
  ∃ x y : ℝ, 2 * x + y = 20 ∧ (∀ (x : ℝ), x > 0 → 
  (∀ (y : ℝ), y > 0 → 2 * x + y = 20 → area_of_sector x y ≤ area_of_sector 5 (20 - 2 * 5))) ∧ x = 5 :=
by
  sorry

end maximize_area_of_sector_l95_95361


namespace orange_profit_44_percent_l95_95212

theorem orange_profit_44_percent :
  (∀ CP SP : ℚ, 0.99 * CP = 1 ∧ SP = CP / 16 → 1 / 11 = CP * (1 + 44 / 100)) :=
by
  sorry

end orange_profit_44_percent_l95_95212


namespace sum_not_complete_residue_system_l95_95931

theorem sum_not_complete_residue_system {n : ℕ} (hn_even : Even n)
    (a b : Fin n → ℕ) (ha : ∀ k, a k < n) (hb : ∀ k, b k < n) 
    (h_complete_a : ∀ x : Fin n, ∃ k : Fin n, a k = x) 
    (h_complete_b : ∀ y : Fin n, ∃ k : Fin n, b k = y) :
    ¬ (∀ z : Fin n, ∃ k : Fin n, ∃ l : Fin n, z = (a k + b l) % n) :=
by
  sorry

end sum_not_complete_residue_system_l95_95931


namespace max_xy_l95_95749

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 2 * y = 12) : 
  xy ≤ 6 :=
sorry

end max_xy_l95_95749


namespace evaluate_g_x_plus_2_l95_95001

theorem evaluate_g_x_plus_2 (x : ℝ) (h₁ : x ≠ -3/2) (h₂ : x ≠ 2) : 
  (2 * (x + 2) + 3) / ((x + 2) - 2) = (2 * x + 7) / x :=
by 
  sorry

end evaluate_g_x_plus_2_l95_95001


namespace remainder_2456789_div_7_l95_95054

theorem remainder_2456789_div_7 :
  2456789 % 7 = 6 := 
by 
  sorry

end remainder_2456789_div_7_l95_95054


namespace cylinder_heights_relation_l95_95462

variables {r1 r2 h1 h2 : ℝ}

theorem cylinder_heights_relation 
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = (6 / 5) * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_heights_relation_l95_95462


namespace sum_and_product_of_roots_l95_95037

theorem sum_and_product_of_roots (m p : ℝ) 
    (h₁ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α + β = 9)
    (h₂ : ∀ α β : ℝ, (3 * α^2 - m * α + p = 0 ∧ 3 * β^2 - m * β + p = 0) → α * β = 14) :
    m + p = 69 := 
sorry

end sum_and_product_of_roots_l95_95037


namespace gcd_204_85_l95_95030

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l95_95030


namespace point_not_in_third_quadrant_l95_95904

theorem point_not_in_third_quadrant (A : ℝ × ℝ) (h : A.snd = -A.fst + 8) : ¬ (A.fst < 0 ∧ A.snd < 0) :=
sorry

end point_not_in_third_quadrant_l95_95904


namespace min_angle_B_l95_95126

-- Definitions using conditions from part a)
def triangle (A B C : ℝ) : Prop := A + B + C = Real.pi
def arithmetic_sequence_prop (A B C : ℝ) : Prop := 
  Real.tan A + Real.tan C = 2 * (1 + Real.sqrt 2) * Real.tan B

-- Main theorem to prove
theorem min_angle_B (A B C : ℝ) (h1 : triangle A B C) (h2 : arithmetic_sequence_prop A B C) :
  B ≥ Real.pi / 4 :=
sorry

end min_angle_B_l95_95126


namespace cone_cube_volume_ratio_l95_95484

theorem cone_cube_volume_ratio (s : ℝ) (h : ℝ) (r : ℝ) (π : ℝ) 
  (cone_inscribed_in_cube : r = s / 2 ∧ h = s ∧ π > 0) :
  ((1/3) * π * r^2 * h) / (s^3) = π / 12 :=
by
  sorry

end cone_cube_volume_ratio_l95_95484


namespace required_fabric_l95_95751

noncomputable def total_fabric :=
  let a := 2011 in
  let r := (4 : ℚ) / 5 in
  a / (1 - r)

theorem required_fabric:
  total_fabric = 10055 := by
  have a : ℚ := 2011
  have r : ℚ := 4 / 5
  have h1 : a / (1 - r) = 10055
  calc
    a / (1 - r)
    = 2011 * 5 : by rw [div_eq_mul_inv, inv_of_15, mul_comm, mul_assoc, div_self, mul_one]
  exact h1

end required_fabric_l95_95751


namespace cos_150_eq_neg_sqrt3_div_2_l95_95837

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95837


namespace problem_solution_l95_95977

theorem problem_solution : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 :=
by
  sorry

end problem_solution_l95_95977


namespace fraction_of_track_Scottsdale_to_Forest_Grove_l95_95948

def distance_between_Scottsdale_and_Sherbourne : ℝ := 200
def round_trip_duration : ℝ := 5
def time_Harsha_to_Sherbourne : ℝ := 2

theorem fraction_of_track_Scottsdale_to_Forest_Grove :
  ∃ f : ℝ, f = 1/5 ∧
    ∀ (d : ℝ) (t : ℝ) (h : ℝ),
    d = distance_between_Scottsdale_and_Sherbourne →
    t = round_trip_duration →
    h = time_Harsha_to_Sherbourne →
    (2.5 - h) / t = f :=
sorry

end fraction_of_track_Scottsdale_to_Forest_Grove_l95_95948


namespace square_roots_equal_49_l95_95908

theorem square_roots_equal_49 (x a : ℝ) (hx1 : (2 * x - 3)^2 = a) (hx2 : (5 - x)^2 = a) (ha_pos: a > 0) : a = 49 := 
by 
  sorry

end square_roots_equal_49_l95_95908


namespace enclosed_region_area_l95_95733

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l95_95733


namespace gcd_204_85_l95_95027

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l95_95027


namespace cos_150_degree_l95_95833

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95833


namespace find_ABC_sum_l95_95026

-- Conditions
def poly (A B C : ℤ) (x : ℤ) := x^3 + A * x^2 + B * x + C
def roots_condition (A B C : ℤ) := poly A B C (-1) = 0 ∧ poly A B C 3 = 0 ∧ poly A B C 4 = 0

-- Proof goal
theorem find_ABC_sum (A B C : ℤ) (h : roots_condition A B C) : A + B + C = 11 :=
sorry

end find_ABC_sum_l95_95026


namespace maximize_distance_l95_95386

theorem maximize_distance (D_F D_R : ℕ) (x y : ℕ) (h1 : D_F = 21000) (h2 : D_R = 28000)
  (h3 : x + y ≤ D_F) (h4 : x + y ≤ D_R) :
  x + y = 24000 :=
sorry

end maximize_distance_l95_95386


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95998

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95998


namespace find_value_l95_95889

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom explicit_form : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem statement
theorem find_value : f (-5/2) = -1/2 :=
by
  -- Here would be the place to start the proof based on the above axioms
  sorry

end find_value_l95_95889


namespace triangle_angle_and_area_l95_95299

section Geometry

variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def triangle_sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 0 < A ∧ A < Real.pi ∧
  0 < B ∧ B < Real.pi ∧
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

def vectors_parallel (a b : ℝ) (A B : ℝ) : Prop := 
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A

-- Problem statement
theorem triangle_angle_and_area (A B C a b c : ℝ) : 
  triangle_sides_opposite_angles a b c A B C ∧ vectors_parallel a b A B ∧ a = Real.sqrt 7 ∧ b = 2 ∧ A = Real.pi / 3
  → A = Real.pi / 3 ∧ (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end Geometry

end triangle_angle_and_area_l95_95299


namespace arithmetic_sequence_sum_l95_95309

/-!
    Let \( \{a_n\} \) be an arithmetic sequence with the sum of the first \( n \) terms denoted as \( S_n \).
    If \( S_{17} = \frac{17}{2} \), then \( a_3 + a_{15} = 1 \).
-/

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_sum : sum_of_first_n_terms a S) (h_S17 : S 17 = 17 / 2) : a 2 + a 14 = 1 :=
sorry

end arithmetic_sequence_sum_l95_95309


namespace quadratic_eq_real_roots_l95_95140

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end quadratic_eq_real_roots_l95_95140


namespace solve_system_of_equations_l95_95566

theorem solve_system_of_equations (u v w : ℝ) (h₀ : u ≠ 0) (h₁ : v ≠ 0) (h₂ : w ≠ 0) :
  (3 / (u * v) + 15 / (v * w) = 2) ∧
  (15 / (v * w) + 5 / (w * u) = 2) ∧
  (5 / (w * u) + 3 / (u * v) = 2) →
  (u = 1 ∧ v = 3 ∧ w = 5) ∨
  (u = -1 ∧ v = -3 ∧ w = -5) :=
by
  sorry

end solve_system_of_equations_l95_95566


namespace cos_150_eq_neg_sqrt3_over_2_l95_95797

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95797


namespace solve_equation_a_solve_equation_b_l95_95326

-- Problem a
theorem solve_equation_a (a b x : ℝ) (h₀ : x ≠ a) (h₁ : x ≠ b) (h₂ : a + b ≠ 0) (h₃ : a ≠ 0) (h₄ : b ≠ 0) (h₅ : a ≠ b):
  (x + a) / (x - a) + (x + b) / (x - b) = 2 ↔ x = (2 * a * b) / (a + b) :=
by
  sorry

-- Problem b
theorem solve_equation_b (a b c d x : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : x ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : ab + c ≠ 0):
  c * (d / (a * b) - (a * b) / x) + d = c^2 / x ↔ x = (a * b * c) / d :=
by
  sorry

end solve_equation_a_solve_equation_b_l95_95326


namespace highway_extension_completion_l95_95477

def current_length := 200
def final_length := 650
def built_first_day := 50
def built_second_day := 3 * built_first_day

theorem highway_extension_completion :
  (final_length - current_length - built_first_day - built_second_day) = 250 := by
  sorry

end highway_extension_completion_l95_95477


namespace cos_150_eq_negative_cos_30_l95_95791

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95791


namespace candy_bars_per_friend_l95_95438

-- Definitions based on conditions
def total_candy_bars : ℕ := 24
def spare_candy_bars : ℕ := 10
def number_of_friends : ℕ := 7

-- The problem statement as a Lean theorem
theorem candy_bars_per_friend :
  (total_candy_bars - spare_candy_bars) / number_of_friends = 2 := 
by
  sorry

end candy_bars_per_friend_l95_95438


namespace gcd_204_85_l95_95029

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l95_95029


namespace playground_perimeter_l95_95483

theorem playground_perimeter (x y : ℝ) 
  (h1 : x^2 + y^2 = 289) 
  (h2 : x * y = 120) : 
  2 * (x + y) = 46 :=
by 
  sorry

end playground_perimeter_l95_95483


namespace time_to_fill_tank_with_leak_l95_95443

theorem time_to_fill_tank_with_leak (A L : ℚ) (hA : A = 1/6) (hL : L = 1/24) :
  (1 / (A - L)) = 8 := 
by 
  sorry

end time_to_fill_tank_with_leak_l95_95443


namespace isabel_piggy_bank_l95_95918

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end isabel_piggy_bank_l95_95918


namespace bisection_interval_length_l95_95138

theorem bisection_interval_length (n : ℕ) : 
  (1 / (2:ℝ)^n) ≤ 0.01 → n ≥ 7 :=
by 
  sorry

end bisection_interval_length_l95_95138


namespace total_size_of_game_is_880_l95_95554

-- Define the initial amount already downloaded
def initialAmountDownloaded : ℕ := 310

-- Define the download speed after the connection slows (in MB per minute)
def downloadSpeed : ℕ := 3

-- Define the remaining download time (in minutes)
def remainingDownloadTime : ℕ := 190

-- Define the total additional data to be downloaded in the remaining time (speed * time)
def additionalDataDownloaded : ℕ := downloadSpeed * remainingDownloadTime

-- Define the total size of the game as the sum of initial and additional data downloaded
def totalSizeOfGame : ℕ := initialAmountDownloaded + additionalDataDownloaded

-- State the theorem to prove
theorem total_size_of_game_is_880 : totalSizeOfGame = 880 :=
by 
  -- We provide no proof here; 'sorry' indicates an unfinished proof.
  sorry

end total_size_of_game_is_880_l95_95554


namespace geometric_sequence_problem_l95_95145

variable {a : ℕ → ℝ} -- Considering the sequence is a real number sequence
variable {q : ℝ} -- Common ratio

-- Conditions
axiom a2a6_eq_16 : a 2 * a 6 = 16
axiom a4_plus_a8_eq_8 : a 4 + a 8 = 8

-- Geometric sequence definition
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_problem : a 20 / a 10 = 1 :=
  by
  sorry

end geometric_sequence_problem_l95_95145


namespace beach_ball_problem_l95_95077

noncomputable def change_in_radius (C₁ C₂ : ℝ) : ℝ := (C₂ - C₁) / (2 * Real.pi)

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

noncomputable def percentage_increase_in_volume (V₁ V₂ : ℝ) : ℝ := (V₂ - V₁) / V₁ * 100

theorem beach_ball_problem (C₁ C₂ : ℝ) (hC₁ : C₁ = 30) (hC₂ : C₂ = 36) :
  change_in_radius C₁ C₂ = 3 / Real.pi ∧
  percentage_increase_in_volume (volume (C₁ / (2 * Real.pi))) (volume (C₂ / (2 * Real.pi))) = 72.78 :=
by
  sorry

end beach_ball_problem_l95_95077


namespace brick_length_l95_95070

theorem brick_length (L : ℝ) :
  (∀ (V_wall V_brick : ℝ),
    V_wall = 29 * 100 * 2 * 100 * 0.75 * 100 ∧
    V_wall = 29000 * V_brick ∧
    V_brick = L * 10 * 7.5) →
  L = 20 :=
by
  intro h
  sorry

end brick_length_l95_95070


namespace number_of_ordered_pairs_l95_95512

theorem number_of_ordered_pairs : ∃ (s : Finset (ℂ × ℂ)), 
    (∀ (a b : ℂ), (a, b) ∈ s → a^5 * b^3 = 1 ∧ a^9 * b^2 = 1) ∧ 
    s.card = 17 := 
by
  sorry

end number_of_ordered_pairs_l95_95512


namespace initial_bacteria_count_l95_95569

theorem initial_bacteria_count (d: ℕ) (t_final: ℕ) (N_final: ℕ) 
    (h1: t_final = 4 * 60)  -- 4 minutes equals 240 seconds
    (h2: d = 15)            -- Doubling interval is 15 seconds
    (h3: N_final = 2097152) -- Final bacteria count is 2,097,152
    :
    ∃ n: ℕ, N_final = n * 2^((t_final / d)) ∧ n = 32 :=
by
  sorry

end initial_bacteria_count_l95_95569


namespace area_enclosed_by_region_l95_95731

theorem area_enclosed_by_region :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6*x + 8*y = -9) →
  let radius := 4 in
  let area := real.pi * radius^2 in
  area = 16 * real.pi :=
sorry

end area_enclosed_by_region_l95_95731


namespace problem_solution_l95_95069

def white_ball_condition (n : ℕ) : Prop :=
  ∀ (k : ℕ), let i := (k % 5) in
  2 ≤ (if i = 0 then 1 else if i = 1 then 1 else 0) +

       (if i = 1 then 1 else if i = 2 then 1 else 0) +

       (if i = 2 then 1 else if i = 3 then 1 else 0) +

       (if i = 3 then 1 else if i = 4 then 1 else 0) +

       (if i = 4 then 1 else if i = 0 then 1 else 0) ∧
  (if k < n then true else false) = 2

theorem problem_solution {n : ℕ} (h1 : n ≠ 0) (h2 : n ≠ 2021) (h3 : n ≠ 2022) (h4 : n ≠ 2023) (h5 : n ≠ 2024) :
  (¬ (n % 5 = 0)) ∧ white_ball_condition n → false :=
by {
  sorry
}

end problem_solution_l95_95069


namespace cos_150_eq_neg_sqrt3_over_2_l95_95795

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95795


namespace side_length_S2_l95_95756

theorem side_length_S2 (r s : ℕ) (h1 : 2 * r + s = 2260) (h2 : 2 * r + 3 * s = 3782) : s = 761 :=
by
  -- proof omitted
  sorry

end side_length_S2_l95_95756


namespace teams_same_matches_l95_95352

theorem teams_same_matches (n : ℕ) (h : n = 30) : ∃ (i j : ℕ), i ≠ j ∧ ∀ (m : ℕ), m ≤ n - 1 → (some_number : ℕ) = (some_number : ℕ) :=
by {
  sorry
}

end teams_same_matches_l95_95352


namespace three_letter_initials_count_l95_95131

theorem three_letter_initials_count :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  (finset.univ.filter (λ l : list Char, l.length = 3 ∧ l.nodup ∧ 
                                            ∀ x ∈ l, x ∈ letters)).card = 720 :=
by
  sorry

end three_letter_initials_count_l95_95131


namespace cos_150_eq_neg_sqrt3_over_2_l95_95798

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95798


namespace find_p_l95_95408

theorem find_p (p : ℚ) : (∀ x : ℚ, (3 * x + 4) = 0 → (4 * x ^ 3 + p * x ^ 2 + 17 * x + 24) = 0) → p = 13 / 4 :=
by
  sorry

end find_p_l95_95408


namespace uncover_area_is_64_l95_95475

-- Conditions as definitions
def length_of_floor := 10
def width_of_floor := 8
def side_of_carpet := 4

-- The statement of the problem
theorem uncover_area_is_64 :
  let area_of_floor := length_of_floor * width_of_floor
  let area_of_carpet := side_of_carpet * side_of_carpet
  let uncovered_area := area_of_floor - area_of_carpet
  uncovered_area = 64 :=
by
  sorry

end uncover_area_is_64_l95_95475


namespace Sarah_shampoo_conditioner_usage_l95_95695

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l95_95695


namespace multiply_polynomials_l95_95228

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end multiply_polynomials_l95_95228


namespace unique_four_digit_square_l95_95761

theorem unique_four_digit_square (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 10 = (n / 10) % 10) ∧ 
  ((n / 100) % 10 = (n / 1000) % 10) ∧ 
  (∃ k : ℕ, n = k^2) ↔ n = 7744 := 
by 
  sorry

end unique_four_digit_square_l95_95761


namespace jump_difference_l95_95710

-- Definitions based on conditions
def grasshopper_jump : ℕ := 13
def frog_jump : ℕ := 11

-- Proof statement
theorem jump_difference : grasshopper_jump - frog_jump = 2 := by
  sorry

end jump_difference_l95_95710


namespace identify_not_increasing_l95_95468

open Real

section

variable (a : ℝ)

def funcA (x : ℝ) : ℝ := x + a^2 * x - 3
def funcB (x : ℝ) : ℝ := 2^x
def funcC (x : ℝ) : ℝ := 2 * x^2 + x + 1
def funcD (x : ℝ) : ℝ := |3 - x|

theorem identify_not_increasing :
  ∃ x : ℝ, 0 ≤ x ∧ (derivative funcD x) < 0 ∨ 3 < x ∧ (derivative funcD x) > 0 :=
sorry

end

end identify_not_increasing_l95_95468


namespace largest_n_factors_l95_95240

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end largest_n_factors_l95_95240


namespace number_of_four_digit_numbers_l95_95269

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l95_95269


namespace bus_ride_duration_l95_95678

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end bus_ride_duration_l95_95678


namespace trigonometric_expression_eval_l95_95886

-- Conditions
variable (α : Real) (h1 : ∃ x : Real, 3 * x^2 - x - 2 = 0 ∧ x = Real.cos α) (h2 : α > π ∧ α < 3 * π / 2)

-- Question and expected answer
theorem trigonometric_expression_eval :
  (Real.sin (-α + 3 * π / 2) * Real.cos (3 * π / 2 + α) * Real.tan (π - α)^2) /
  (Real.cos (π / 2 + α) * Real.sin (π / 2 - α)) = 5 / 4 := sorry

end trigonometric_expression_eval_l95_95886


namespace insurance_percentage_l95_95724

noncomputable def total_pills_per_year : ℕ := 2 * 365

noncomputable def cost_per_pill : ℕ := 5

noncomputable def total_medication_cost_per_year : ℕ := total_pills_per_year * cost_per_pill

noncomputable def doctor_visits_per_year : ℕ := 2

noncomputable def cost_per_doctor_visit : ℕ := 400

noncomputable def total_doctor_cost_per_year : ℕ := doctor_visits_per_year * cost_per_doctor_visit

noncomputable def total_yearly_cost_without_insurance : ℕ := total_medication_cost_per_year + total_doctor_cost_per_year

noncomputable def total_payment_per_year : ℕ := 1530

noncomputable def insurance_coverage_per_year : ℕ := total_yearly_cost_without_insurance - total_payment_per_year

theorem insurance_percentage:
  (insurance_coverage_per_year * 100) / total_medication_cost_per_year = 80 :=
by sorry

end insurance_percentage_l95_95724


namespace harrys_fish_count_l95_95009

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l95_95009


namespace repeating_decimal_fraction_l95_95869

noncomputable def repeating_decimal := 7 + ((789 : ℚ) / (10^4 - 1))

theorem repeating_decimal_fraction :
  repeating_decimal = (365 : ℚ) / 85 :=
by
  sorry

end repeating_decimal_fraction_l95_95869


namespace dual_colored_numbers_l95_95291

theorem dual_colored_numbers (table : Matrix (Fin 10) (Fin 20) ℕ)
  (distinct_numbers : ∀ (i j k l : Fin 10) (m n : Fin 20), 
    (i ≠ k ∨ m ≠ n) → table i m ≠ table k n)
  (row_red : ∀ (i : Fin 10), ∃ r₁ r₂ : Fin 20, r₁ ≠ r₂ ∧ 
    (∀ (j : Fin 20), table i j ≤ table i r₁ ∨ table i j ≤ table i r₂))
  (col_blue : ∀ (j : Fin 20), ∃ b₁ b₂ : Fin 10, b₁ ≠ b₂ ∧ 
    (∀ (i : Fin 10), table i j ≤ table b₁ j ∨ table i j ≤ table b₂ j)) : 
  ∃ i₁ i₂ i₃ : Fin 10, ∃ j₁ j₂ j₃ : Fin 20, 
    i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₂ ≠ i₃ ∧ j₁ ≠ j₂ ∧ j₁ ≠ j₃ ∧ j₂ ≠ j₃ ∧ 
    ((table i₁ j₁ ≤ table i₁ j₂ ∨ table i₁ j₁ ≤ table i₃ j₂) ∧ 
     (table i₂ j₂ ≤ table i₂ j₁ ∨ table i₂ j₂ ≤ table i₃ j₁) ∧ 
     (table i₃ j₃ ≤ table i₃ j₁ ∨ table i₃ j₃ ≤ table i₂ j₁)) := 
  sorry

end dual_colored_numbers_l95_95291


namespace cos_150_eq_neg_sqrt3_div_2_l95_95836

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95836


namespace find_multiple_of_smaller_integer_l95_95716

theorem find_multiple_of_smaller_integer (L S k : ℕ) 
  (h1 : S = 10) 
  (h2 : L + S = 30) 
  (h3 : 2 * L = k * S - 10) 
  : k = 5 := 
by
  sorry

end find_multiple_of_smaller_integer_l95_95716


namespace amount_with_r_l95_95472

theorem amount_with_r (p q r T : ℝ) 
  (h1 : p + q + r = 4000)
  (h2 : r = (2/3) * T)
  (h3 : T = p + q) : 
  r = 1600 := by
  sorry

end amount_with_r_l95_95472


namespace part_a_prob_part_b_expected_time_l95_95990

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l95_95990


namespace harry_fish_count_l95_95013

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l95_95013


namespace ratio_of_ages_three_years_ago_l95_95185

theorem ratio_of_ages_three_years_ago (k Y_c : ℕ) (h1 : 45 - 3 = k * (Y_c - 3)) (h2 : (45 + 7) + (Y_c + 7) = 83) : (45 - 3) / (Y_c - 3) = 2 :=
by {
  sorry
}

end ratio_of_ages_three_years_ago_l95_95185


namespace triangle_angles_l95_95255

theorem triangle_angles (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : B = 120)
  (h3 : (∃D, A = D ∧ (A + A + C = 180 ∨ A + C + C = 180)) ∨ (∃E, C = E ∧ (B + 15 + 45 = 180 ∨ B + 15 + 15 = 180))) :
  (A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) :=
sorry

end triangle_angles_l95_95255


namespace intersecting_lines_a_b_sum_zero_l95_95625

theorem intersecting_lines_a_b_sum_zero
    (a b : ℝ)
    (h₁ : ∀ z : ℝ × ℝ, z = (3, -3) → z.1 = (1 / 3) * z.2 + a)
    (h₂ : ∀ z : ℝ × ℝ, z = (3, -3) → z.2 = (1 / 3) * z.1 + b)
    :
    a + b = 0 := by
  sorry

end intersecting_lines_a_b_sum_zero_l95_95625


namespace pages_read_on_Sunday_l95_95148

def total_pages : ℕ := 93
def pages_read_on_Saturday : ℕ := 30
def pages_remaining_after_Sunday : ℕ := 43

theorem pages_read_on_Sunday : total_pages - pages_read_on_Saturday - pages_remaining_after_Sunday = 20 := by
  sorry

end pages_read_on_Sunday_l95_95148


namespace point_in_first_quadrant_l95_95419

theorem point_in_first_quadrant (x y : ℝ) (h₁ : x = 3) (h₂ : y = 2) (hx : x > 0) (hy : y > 0) :
  ∃ q : ℕ, q = 1 := 
by
  sorry

end point_in_first_quadrant_l95_95419


namespace p_distance_300_l95_95473

-- Assume q's speed is v meters per second, and the race ends in a tie
variables (v : ℝ) (t : ℝ)
variable (d : ℝ)

-- Conditions
def q_speed : ℝ := v
def p_speed : ℝ := 1.25 * v
def q_distance : ℝ := d
def p_distance : ℝ := d + 60

-- Time equations
def q_time_eq : Prop := d = v * t
def p_time_eq : Prop := d + 60 = (1.25 * v) * t

-- Given the conditions, prove that p ran 300 meters in the race
theorem p_distance_300
  (v_pos : v > 0) 
  (t_pos : t > 0)
  (q_time : q_time_eq v d t)
  (p_time : p_time_eq v d t) :
  p_distance d = 300 :=
by
  sorry

end p_distance_300_l95_95473


namespace three_integers_desc_order_l95_95059

theorem three_integers_desc_order (a b c : ℤ) : ∃ a' b' c' : ℤ, 
  (a = a' ∨ a = b' ∨ a = c') ∧
  (b = a' ∨ b = b' ∨ b = c') ∧
  (c = a' ∨ c = b' ∨ c = c') ∧ 
  (a' ≠ b' ∨ a' ≠ c' ∨ b' ≠ c') ∧
  a' ≥ b' ∧ b' ≥ c' :=
sorry

end three_integers_desc_order_l95_95059


namespace unique_games_count_l95_95088

noncomputable def total_games_played (n : ℕ) (m : ℕ) : ℕ :=
  (n * m) / 2

theorem unique_games_count (students : ℕ) (games_per_student : ℕ) (h1 : students = 9) (h2 : games_per_student = 6) :
  total_games_played students games_per_student = 27 :=
by
  rw [h1, h2]
  -- This partially evaluates total_games_played using the values from h1 and h2.
  -- Performing actual proof steps is not necessary, so we'll use sorry.
  sorry

end unique_games_count_l95_95088


namespace eddie_games_l95_95102

-- Define the study block duration in minutes
def study_block_duration : ℕ := 60

-- Define the homework time in minutes
def homework_time : ℕ := 25

-- Define the time for one game in minutes
def game_time : ℕ := 5

-- Define the total time Eddie can spend playing games
noncomputable def time_for_games : ℕ := study_block_duration - homework_time

-- Define the number of games Eddie can play
noncomputable def number_of_games : ℕ := time_for_games / game_time

-- Theorem stating the number of games Eddie can play while completing his homework
theorem eddie_games : number_of_games = 7 := by
  sorry

end eddie_games_l95_95102


namespace amount_left_after_spending_l95_95921

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end amount_left_after_spending_l95_95921


namespace power_inequality_l95_95887

open Nat

theorem power_inequality (a b : ℝ) (n : ℕ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a) + (1 / b) = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
  sorry

end power_inequality_l95_95887


namespace expression_value_l95_95494

theorem expression_value : 2013 * (2015 / 2014) + 2014 * (2016 / 2015) + (4029 / (2014 * 2015)) = 4029 :=
by
  sorry

end expression_value_l95_95494


namespace max_value_expression_l95_95905

theorem max_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
by sorry

end max_value_expression_l95_95905


namespace inscribed_square_area_ratio_l95_95501

theorem inscribed_square_area_ratio (side_length : ℝ) (h_pos : side_length > 0) :
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  (inscribed_square_area / large_square_area) = (1 / 4) :=
by
  let large_square_area := side_length * side_length
  let inscribed_square_side_length := side_length / 2
  let inscribed_square_area := inscribed_square_side_length * inscribed_square_side_length
  sorry

end inscribed_square_area_ratio_l95_95501


namespace donovan_lap_time_l95_95866

-- Definitions based on problem conditions
def lap_time_michael := 40  -- Michael's lap time in seconds
def laps_michael := 9       -- Laps completed by Michael to pass Donovan
def laps_donovan := 8       -- Laps completed by Donovan in the same time

-- Condition based on the solution
def race_duration := laps_michael * lap_time_michael

-- define the conjecture
theorem donovan_lap_time : 
  (race_duration = laps_donovan * 45) := 
sorry

end donovan_lap_time_l95_95866


namespace cos_150_eq_neg_sqrt3_div_2_l95_95779

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95779


namespace train_length_l95_95216

-- Definitions based on conditions
def train_speed_kmh := 54 -- speed of the train in km/h
def time_to_cross_sec := 16 -- time to cross the telegraph post in seconds
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 5 / 18 -- conversion factor from km/h to m/s

-- Prove that the length of the train is 240 meters
theorem train_length (h1 : train_speed_kmh = 54) (h2 : time_to_cross_sec = 16) : 
  (kmh_to_ms train_speed_kmh * time_to_cross_sec) = 240 := by
  sorry

end train_length_l95_95216


namespace like_terms_correct_l95_95345

theorem like_terms_correct : 
  (¬(∀ x y z w : ℝ, (x * y^2 = z ∧ x^2 * y = w)) ∧ 
   ¬(∀ x y : ℝ, (x * y = -2 * y)) ∧ 
    (2^3 = 8 ∧ 3^2 = 9) ∧ 
   ¬(∀ x y z w : ℝ, (5 * x * y = z ∧ 6 * x * y^2 = w))) :=
by
  sorry

end like_terms_correct_l95_95345


namespace area_of_fourth_square_l95_95463

theorem area_of_fourth_square (AB BC AC CD AD : ℝ) (h_sum_ABC : AB^2 + 25 = 50)
  (h_sum_ACD : 50 + 49 = AD^2) : AD^2 = 99 :=
by
  sorry

end area_of_fourth_square_l95_95463


namespace weight_of_first_lift_l95_95540

-- Definitions as per conditions
variables (x y : ℝ)
def condition1 : Prop := x + y = 1800
def condition2 : Prop := 2 * x = y + 300

-- Prove that the weight of Joe's first lift is 700 pounds
theorem weight_of_first_lift (h1 : condition1 x y) (h2 : condition2 x y) : x = 700 :=
by
  sorry

end weight_of_first_lift_l95_95540


namespace hexagon_points_fourth_layer_l95_95604

theorem hexagon_points_fourth_layer :
  ∃ (h : ℕ → ℕ), h 1 = 1 ∧ (∀ n ≥ 2, h n = h (n - 1) + 6 * (n - 1)) ∧ h 4 = 37 :=
by
  sorry

end hexagon_points_fourth_layer_l95_95604


namespace exists_increasing_seq_with_sum_square_diff_l95_95380

/-- There exists an increasing sequence of natural numbers in which
  the sum of any two consecutive terms is equal to the square of their
  difference. -/
theorem exists_increasing_seq_with_sum_square_diff :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, a n + a (n + 1) = (a (n + 1) - a n) ^ 2) :=
sorry

end exists_increasing_seq_with_sum_square_diff_l95_95380


namespace alice_password_probability_l95_95218

section AlicePassword

def is_even_digit (n : ℕ) : Prop := n ∈ {0, 2, 4, 6, 8}
def is_non_zero_digit (n : ℕ) : Prop := n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
def is_vowel (c : Char) : Prop := c ∈ {'A', 'E', 'I', 'O', 'U'}
def is_letter (c : Char) : Prop := c.isAlpha

theorem alice_password_probability :
  let total_digits := 10 in
  let total_letters := 26 in
  let even_digits := 5 in
  let non_zero_digits := 9 in
  let vowel_letters := 5 in
  (even_digits/total_digits : ℚ) * (1 : ℚ) * (non_zero_digits/total_digits : ℚ) * (vowel_letters/total_letters : ℚ) = 9/104 :=
sorry

end AlicePassword

end alice_password_probability_l95_95218


namespace find_rate_of_interest_l95_95081

theorem find_rate_of_interest (P R : ℝ) (H1: 17640 = P * (1 + R / 100)^2) (H2: 21168 = P * (1 + R / 100)^3) : R = 6.27 :=
by 
  sorry

end find_rate_of_interest_l95_95081


namespace four_digit_numbers_l95_95274

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l95_95274


namespace other_number_in_product_l95_95647

theorem other_number_in_product (w : ℕ) (n : ℕ) (hw_pos : 0 < w) (n_factor : Nat.lcm (2^5) (Nat.gcd  864 w) = 2^5 * 3^3) (h_w : w = 144) : n = 6 :=
by
  -- proof would go here
  sorry

end other_number_in_product_l95_95647


namespace total_money_raised_l95_95183

-- Assume there are 30 students in total
def total_students := 30

-- Assume 10 students raised $20 each
def students_raising_20 := 10
def money_raised_per_20 := 20

-- The rest of the students raised $30 each
def students_raising_30 := total_students - students_raising_20
def money_raised_per_30 := 30

-- Prove that the total amount raised is $800
theorem total_money_raised :
  (students_raising_20 * money_raised_per_20) +
  (students_raising_30 * money_raised_per_30) = 800 :=
by
  sorry

end total_money_raised_l95_95183


namespace fraction_problem_l95_95608

theorem fraction_problem (a b c d e: ℚ) (val: ℚ) (h_a: a = 1/4) (h_b: b = 1/3) 
  (h_c: c = 1/6) (h_d: d = 1/8) (h_val: val = 72) :
  (a * b * c * val + d) = 9 / 8 :=
by {
  sorry
}

end fraction_problem_l95_95608


namespace nancy_pictures_l95_95058

theorem nancy_pictures (z m b d : ℕ) (hz : z = 120) (hm : m = 75) (hb : b = 45) (hd : d = 93) :
  (z + m + b) - d = 147 :=
by {
  -- Theorem definition capturing the problem statement
  sorry
}

end nancy_pictures_l95_95058


namespace total_pawns_left_l95_95308

  -- Definitions of initial conditions
  def initial_pawns_in_chess : Nat := 8
  def kennedy_pawns_lost : Nat := 4
  def riley_pawns_lost : Nat := 1

  -- Theorem statement to prove the total number of pawns left
  theorem total_pawns_left : (initial_pawns_in_chess - kennedy_pawns_lost) + (initial_pawns_in_chess - riley_pawns_lost) = 11 := by
    sorry
  
end total_pawns_left_l95_95308


namespace cos_150_eq_neg_sqrt3_div_2_l95_95776

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95776


namespace company_fund_initial_amount_l95_95711

-- Let n be the number of employees in the company.
variable (n : ℕ)

-- Conditions from the problem.
def initial_fund := 60 * n - 10
def adjusted_fund := 50 * n + 150
def employees_count := 16

-- Given the conditions, prove that the initial fund amount was $950.
theorem company_fund_initial_amount
    (h1 : adjusted_fund n = initial_fund n)
    (h2 : n = employees_count) : 
    initial_fund n = 950 := by
  sorry

end company_fund_initial_amount_l95_95711


namespace two_point_questions_count_l95_95196

theorem two_point_questions_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
sorry

end two_point_questions_count_l95_95196


namespace Katie_old_games_l95_95546

theorem Katie_old_games (O : ℕ) (hk1 : Katie_new_games = 57) (hf1 : Friends_new_games = 34) (hk2 : Katie_total_games = Friends_total_games + 62) : 
  O = 39 :=
by
  sorry

variables (Katie_new_games Friends_new_games Katie_total_games Friends_total_games : ℕ)

end Katie_old_games_l95_95546


namespace total_caps_produced_l95_95354

-- Define the production of each week as given in the conditions.
def week1_caps : ℕ := 320
def week2_caps : ℕ := 400
def week3_caps : ℕ := 300

-- Define the average of the first three weeks.
def average_caps : ℕ := (week1_caps + week2_caps + week3_caps) / 3

-- Define the production increase for the fourth week.
def increase_caps : ℕ := average_caps / 5  -- 20% is equivalent to dividing by 5

-- Calculate the total production for the fourth week (including the increase).
def week4_caps : ℕ := average_caps + increase_caps

-- Calculate the total number of caps produced in four weeks.
def total_caps : ℕ := week1_caps + week2_caps + week3_caps + week4_caps

-- Theorem stating the total production over the four weeks.
theorem total_caps_produced : total_caps = 1428 := by sorry

end total_caps_produced_l95_95354


namespace linear_function_k_range_l95_95287

theorem linear_function_k_range (k b : ℝ) (h1 : k ≠ 0) (h2 : ∃ x : ℝ, (x = 2) ∧ (-3 = k * x + b)) (h3 : 0 < b ∧ b < 1) : -2 < k ∧ k < -3 / 2 :=
by
  sorry

end linear_function_k_range_l95_95287


namespace simplify_expression_l95_95236

theorem simplify_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 :=
sorry

end simplify_expression_l95_95236


namespace largest_possible_value_n_l95_95570

theorem largest_possible_value_n (n : ℕ) (h : ∀ m : ℕ, m ≠ n → n % m = 0 → m ≤ 35) : n = 35 :=
sorry

end largest_possible_value_n_l95_95570


namespace triangle_median_length_l95_95659

variable (XY XZ XM YZ : ℝ)

theorem triangle_median_length :
  XY = 6 →
  XZ = 8 →
  XM = 5 →
  YZ = 10 := by
  sorry

end triangle_median_length_l95_95659


namespace distance_A_to_C_through_B_l95_95025

-- Define the distances on the map
def Distance_AB_map : ℝ := 20
def Distance_BC_map : ℝ := 10

-- Define the scale of the map
def scale : ℝ := 5

-- Define the actual distances
def Distance_AB := Distance_AB_map * scale
def Distance_BC := Distance_BC_map * scale

-- Define the total distance from A to C through B
def Distance_AC_through_B := Distance_AB + Distance_BC

-- Theorem to be proved
theorem distance_A_to_C_through_B : Distance_AC_through_B = 150 := by
  sorry

end distance_A_to_C_through_B_l95_95025


namespace concave_number_count_l95_95414

def is_concave_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  n >= 100 ∧ n < 1000 ∧ tens < hundreds ∧ tens < units

theorem concave_number_count : ∃ n : ℕ, 
  (∀ m < 1000, is_concave_number m → m = n) ∧ n = 240 :=
by
  sorry

end concave_number_count_l95_95414


namespace pen_cost_l95_95168

theorem pen_cost
  (p q : ℕ)
  (h1 : 6 * p + 5 * q = 380)
  (h2 : 3 * p + 8 * q = 298) :
  p = 47 :=
sorry

end pen_cost_l95_95168


namespace coefficient_a_neg_one_is_28_l95_95913

noncomputable def coefficient_of_a_neg_one_in_expansion : ℤ :=
  ∑ k in Finset.range (8 + 1), if (8 - k - k / 2 = -1) then (Nat.choose 8 k) * (-1)^k else 0

theorem coefficient_a_neg_one_is_28 : coefficient_of_a_neg_one_in_expansion = 28 :=
by
  sorry

end coefficient_a_neg_one_is_28_l95_95913


namespace find_number_of_rabbits_l95_95459

def total_heads (R P : ℕ) : ℕ := R + P
def total_legs (R P : ℕ) : ℕ := 4 * R + 2 * P

theorem find_number_of_rabbits (R P : ℕ)
  (h1 : total_heads R P = 60)
  (h2 : total_legs R P = 192) :
  R = 36 := by
  sorry

end find_number_of_rabbits_l95_95459


namespace intersection_eq_l95_95640

def M : Set Real := {x | x^2 < 3 * x}
def N : Set Real := {x | Real.log x < 0}

theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l95_95640


namespace cos_150_eq_neg_sqrt3_div_2_l95_95838

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95838


namespace gcd_204_85_l95_95033

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l95_95033


namespace cube_path_count_l95_95207

noncomputable def numberOfWaysToMoveOnCube : Nat :=
  20

theorem cube_path_count :
  ∀ (cube : Type) (top bottom side1 side2 side3 side4 : cube),
    (∀ (p : cube → cube → Prop), 
      (p top side1 ∨ p top side2 ∨ p top side3 ∨ p top side4) ∧ 
      (p side1 bottom ∨ p side2 bottom ∨ p side3 bottom ∨ p side4 bottom)) →
    numberOfWaysToMoveOnCube = 20 :=
by
  intros
  sorry

end cube_path_count_l95_95207


namespace symmetric_line_eq_l95_95329

theorem symmetric_line_eq : ∀ (x y : ℝ), (x - 2*y - 1 = 0) ↔ (2*x - y + 1 = 0) :=
by sorry

end symmetric_line_eq_l95_95329


namespace largest_in_eight_consecutive_integers_l95_95717

theorem largest_in_eight_consecutive_integers (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 4304) :
  n + 7 = 544 :=
by
  sorry

end largest_in_eight_consecutive_integers_l95_95717


namespace corrected_mean_l95_95063

theorem corrected_mean (incorrect_mean : ℕ) (num_observations : ℕ) (wrong_value actual_value : ℕ) : 
  (50 * 36 + (43 - 23)) / 50 = 36.4 :=
by
  sorry

end corrected_mean_l95_95063


namespace solution_exists_l95_95944

theorem solution_exists (x : ℝ) :
  (|2 * x - 3| ≤ 3 ∧ (1 / x) < 1 ∧ x ≠ 0) ↔ (1 < x ∧ x ≤ 3) :=
by
  sorry

end solution_exists_l95_95944


namespace construct_triangle_l95_95254

theorem construct_triangle
  (A B C A0 : Point)
  (s_a : ℝ)
  (α1 α2 : ℝ)
  (h_midpoint : midpoint A0 B C)
  (h_median : dist A A0 = s_a)
  (h_angle1 : angle A0 A B = α1)
  (h_angle2 : angle A0 A C = α2) :
  ∃ (Δ : Triangle), same_triangle Δ (Triangle.mk A B C) := by
  sorry

end construct_triangle_l95_95254


namespace probability_of_C_and_D_are_equal_l95_95758

theorem probability_of_C_and_D_are_equal (h1 : Prob_A = 1/4) (h2 : Prob_B = 1/3) (h3 : total_prob = 1) (h4 : Prob_C = Prob_D) : 
  Prob_C = 5/24 ∧ Prob_D = 5/24 := by
  sorry

end probability_of_C_and_D_are_equal_l95_95758


namespace ellipse_area_l95_95082

def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 9 * y^2 - 36 * y + 36 = 0

theorem ellipse_area :
  (∀ x y : ℝ, ellipse_equation x y → true) →
  (π * 1 * (4/3) = 4 * π / 3) :=
by
  intro h
  norm_num
  sorry

end ellipse_area_l95_95082


namespace cat_count_after_10_days_l95_95365

def initial_cats := 60 -- Shelter had 60 cats before the intake
def intake_cats := 30 -- Shelter took in 30 cats
def total_cats_at_start := initial_cats + intake_cats -- 90 cats after intake

def even_days_adoptions := 5 -- Cats adopted on even days
def odd_days_adoptions := 15 -- Cats adopted on odd days
def total_adoptions := even_days_adoptions + odd_days_adoptions -- Total adoptions over 10 days

def day4_births := 10 -- Kittens born on day 4
def day7_births := 5 -- Kittens born on day 7
def total_births := day4_births + day7_births -- Total births over 10 days

def claimed_pets := 2 -- Number of mothers claimed as missing pets

def final_cat_count := total_cats_at_start - total_adoptions + total_births - claimed_pets -- Final cat count

theorem cat_count_after_10_days : final_cat_count = 83 := by
  sorry

end cat_count_after_10_days_l95_95365


namespace cos_150_eq_neg_half_l95_95808

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95808


namespace Nina_has_16dollars65_l95_95318

-- Definitions based on given conditions
variables (W M : ℝ)

-- Condition 1: Nina has exactly enough money to purchase 5 widgets
def condition1 : Prop := 5 * W = M

-- Condition 2: If the cost of each widget were reduced by $1.25, Nina would have exactly enough money to purchase 8 widgets
def condition2 : Prop := 8 * (W - 1.25) = M

-- Statement: Proving the amount of money Nina has is $16.65
theorem Nina_has_16dollars65 (h1 : condition1 W M) (h2 : condition2 W M) : M = 16.65 :=
sorry

end Nina_has_16dollars65_l95_95318


namespace remainder_of_power_sums_modulo_seven_l95_95342

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l95_95342


namespace minimum_students_in_class_l95_95909

def min_number_of_students (b g : ℕ) : ℕ :=
  b + g

theorem minimum_students_in_class
  (b g : ℕ)
  (h1 : b = 2 * g / 3)
  (h2 : ∃ k : ℕ, g = 3 * k)
  (h3 : ∃ k : ℕ, 1 / 2 < (2 / 3) * g / b) :
  min_number_of_students b g = 5 :=
sorry

end minimum_students_in_class_l95_95909


namespace relationship_between_first_and_third_numbers_l95_95721

variable (A B C : ℕ)

theorem relationship_between_first_and_third_numbers
  (h1 : A + B + C = 660)
  (h2 : A = 2 * B)
  (h3 : B = 180) :
  C = A - 240 :=
by
  sorry

end relationship_between_first_and_third_numbers_l95_95721


namespace chocolate_mixture_l95_95281

theorem chocolate_mixture (x : ℝ) (h_initial : 110 / 220 = 0.5)
  (h_equation : (110 + x) / (220 + x) = 0.75) : x = 220 := by
  sorry

end chocolate_mixture_l95_95281


namespace line_k_x_intercept_l95_95676

theorem line_k_x_intercept :
  ∀ (x y : ℝ), 3 * x - 5 * y + 40 = 0 ∧ 
  ∃ m' b', (m' = 4) ∧ (b' = 20 - 4 * 20) ∧ 
  (y = m' * x + b') →
  ∃ x_inter, (y = 0) → (x_inter = 15) := 
by
  sorry

end line_k_x_intercept_l95_95676


namespace borrowed_movie_price_correct_l95_95557

def ticket_price : ℝ := 5.92
def number_of_tickets : ℕ := 2
def total_paid : ℝ := 20.00
def change_received : ℝ := 1.37
def tickets_cost : ℝ := number_of_tickets * ticket_price
def total_spent : ℝ := total_paid - change_received
def borrowed_movie_cost : ℝ := total_spent - tickets_cost

theorem borrowed_movie_price_correct : borrowed_movie_cost = 6.79 := by
  sorry

end borrowed_movie_price_correct_l95_95557


namespace lesser_solution_quadratic_l95_95736

theorem lesser_solution_quadratic (x : ℝ) :
  x^2 + 9 * x - 22 = 0 → x = -11 ∨ x = 2 :=
sorry

end lesser_solution_quadratic_l95_95736


namespace multiply_polynomials_l95_95226

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end multiply_polynomials_l95_95226


namespace consecutive_odd_integers_sum_l95_95044

theorem consecutive_odd_integers_sum (a b c : ℤ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (h3 : c % 2 = 1) (h4 : a < b) (h5 : b < c) (h6 : c = -47) : a + b + c = -141 := 
sorry

end consecutive_odd_integers_sum_l95_95044


namespace rational_coefficients_terms_count_l95_95192

theorem rational_coefficients_terms_count : 
  (∃ s : Finset ℕ, ∀ k ∈ s, k % 20 = 0 ∧ k ≤ 725 ∧ s.card = 37) :=
by
  -- Translates to finding the set of all k satisfying the condition and 
  -- ensuring it has a cardinality of 37.
  sorry

end rational_coefficients_terms_count_l95_95192


namespace cos_150_deg_l95_95820

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95820


namespace mul_65_35_eq_2275_l95_95495

theorem mul_65_35_eq_2275 : 65 * 35 = 2275 := by
  sorry

end mul_65_35_eq_2275_l95_95495


namespace enclosed_region_area_l95_95732

theorem enclosed_region_area :
  (∃ x y : ℝ, x ^ 2 + y ^ 2 - 6 * x + 8 * y = -9) →
  ∃ (r : ℝ), r ^ 2 = 16 ∧ ∀ (area : ℝ), area = π * 4 ^ 2 :=
by
  sorry

end enclosed_region_area_l95_95732


namespace four_digit_numbers_count_l95_95279

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l95_95279


namespace inverse_of_3_mod_199_l95_95107

theorem inverse_of_3_mod_199 : (3 * 133) % 199 = 1 :=
by
  sorry

end inverse_of_3_mod_199_l95_95107


namespace minimum_value_of_fraction_sum_l95_95932

open Real

theorem minimum_value_of_fraction_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 2) : 
    6 ≤ (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) := by 
  sorry

end minimum_value_of_fraction_sum_l95_95932


namespace inequality_neg_multiplication_l95_95900

theorem inequality_neg_multiplication (m n : ℝ) (h : m > n) : -2 * m < -2 * n :=
by {
  sorry
}

end inequality_neg_multiplication_l95_95900


namespace inequality_solution_set_l95_95383

theorem inequality_solution_set (x : ℝ) : (3 - 2 * x) * (x + 1) ≤ 0 ↔ (x < -1) ∨ (x ≥ 3 / 2) :=
  sorry

end inequality_solution_set_l95_95383


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95996

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95996


namespace difference_in_tiles_l95_95485

theorem difference_in_tiles (n : ℕ) (hn : n = 9) : (n + 1)^2 - n^2 = 19 :=
by sorry

end difference_in_tiles_l95_95485


namespace valid_four_digit_numbers_count_l95_95266

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l95_95266


namespace find_smaller_number_l95_95162

theorem find_smaller_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : x = 18 :=
by
  sorry

end find_smaller_number_l95_95162


namespace acute_triangle_contains_vertex_l95_95441

open Real EuclideanGeometry

-- We define the specific proof problem based on the conditions and required conclusion
theorem acute_triangle_contains_vertex (A B C : EuclideanGeometry.Point ℝ) :
  (acute_angle (∠ A B C) ∧ acute_angle (∠ B C A) ∧ acute_angle (∠ C A B)) →
  (exists (P : EuclideanGeometry.Point ℝ), is_cell_vertex P ∧ (point_in_triangle P A B C ∨ point_on_triangle_side P A B C)) := 
sorry

end acute_triangle_contains_vertex_l95_95441


namespace find_x_minus_y_l95_95120

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end find_x_minus_y_l95_95120


namespace part_a_prob_part_b_expected_time_l95_95991

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l95_95991


namespace find_sum_of_squares_l95_95384

theorem find_sum_of_squares (a b c m : ℤ) (h1 : a + b + c = 0) (h2 : a * b + b * c + a * c = -2023) (h3 : a * b * c = -m) : a^2 + b^2 + c^2 = 4046 := by
  sorry

end find_sum_of_squares_l95_95384


namespace trout_to_bass_ratio_l95_95356

theorem trout_to_bass_ratio 
  (bass : ℕ) 
  (trout : ℕ) 
  (blue_gill : ℕ)
  (h1 : bass = 32) 
  (h2 : blue_gill = 2 * bass) 
  (h3 : bass + trout + blue_gill = 104) 
  : (trout / bass) = 1 / 4 :=
by 
  -- intermediate steps can be included here
  sorry

end trout_to_bass_ratio_l95_95356


namespace number_of_gigs_played_l95_95353

/-- Given earnings per gig for each band member and the total earnings, prove the total number of gigs played -/

def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer1_earnings : ℕ := 15
def backup_singer2_earnings : ℕ := 18
def backup_singer3_earnings : ℕ := 12
def total_earnings : ℕ := 3465

def total_earnings_per_gig : ℕ :=
  lead_singer_earnings +
  guitarist_earnings +
  bassist_earnings +
  drummer_earnings +
  keyboardist_earnings +
  backup_singer1_earnings +
  backup_singer2_earnings +
  backup_singer3_earnings

theorem number_of_gigs_played : (total_earnings / total_earnings_per_gig) = 21 := by
  sorry

end number_of_gigs_played_l95_95353


namespace compute_xy_l95_95461

theorem compute_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) : 
  xy = -126 / 25 ∨ xy = -6 := 
sorry

end compute_xy_l95_95461


namespace ellipse_product_l95_95444

noncomputable def AB_CD_product (a b c : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) : ℝ :=
  2 * a * 2 * b

-- The main statement
theorem ellipse_product (c : ℝ) (h_c : c = 8) (h_diameter : 6 = 6)
  (a b : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) :
  AB_CD_product a b c h1 h2 = 175 := sorry

end ellipse_product_l95_95444


namespace cos_150_eq_neg_sqrt3_over_2_l95_95792

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95792


namespace no_solution_when_k_eq_7_l95_95097

theorem no_solution_when_k_eq_7 
  (x : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : 
  (∀ k : ℝ, (x - 3) / (x - 4) = (x - k) / (x - 8) → False) ↔ k = 7 :=
by
  sorry

end no_solution_when_k_eq_7_l95_95097


namespace abs_inequality_solution_set_l95_95961

theorem abs_inequality_solution_set {x : ℝ} : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end abs_inequality_solution_set_l95_95961


namespace cos_150_eq_neg_sqrt3_over_2_l95_95794

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95794


namespace total_birds_from_monday_to_wednesday_l95_95752

def birds_monday := 70
def birds_tuesday := birds_monday / 2
def birds_wednesday := birds_tuesday + 8
def total_birds := birds_monday + birds_tuesday + birds_wednesday

theorem total_birds_from_monday_to_wednesday : total_birds = 148 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end total_birds_from_monday_to_wednesday_l95_95752


namespace chicken_feathers_after_crossing_l95_95456

def cars_dodged : ℕ := 23
def initial_feathers : ℕ := 5263
def feathers_lost : ℕ := 2 * cars_dodged
def final_feathers : ℕ := initial_feathers - feathers_lost

theorem chicken_feathers_after_crossing :
  final_feathers = 5217 := by
sorry

end chicken_feathers_after_crossing_l95_95456


namespace correct_average_l95_95199

theorem correct_average (S' : ℝ) (a a' b b' c c' : ℝ) (n : ℕ) 
  (incorrect_avg : S' / n = 22) 
  (a_eq : a = 52) (a'_eq : a' = 32)
  (b_eq : b = 47) (b'_eq : b' = 27) 
  (c_eq : c = 68) (c'_eq : c' = 45)
  (n_eq : n = 12) 
  : ((S' - (a' + b' + c') + (a + b + c)) / 12 = 27.25) := 
by
  sorry

end correct_average_l95_95199


namespace simplify_and_evaluate_expression_l95_95699

variable (a : ℚ)

theorem simplify_and_evaluate_expression (h : a = -1/3) : 
  (a + 1) * (a - 1) - a * (a + 3) = 0 := 
by
  sorry

end simplify_and_evaluate_expression_l95_95699


namespace factorize_quadratic_l95_95507

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l95_95507


namespace product_power_conjecture_calculate_expression_l95_95446

-- Conjecture Proof
theorem product_power_conjecture (a b : ℂ) (n : ℕ) : (a * b)^n = (a^n) * (b^n) :=
sorry

-- Calculation Proof
theorem calculate_expression : 
  ((-0.125 : ℂ)^2022) * ((2 : ℂ)^2021) * ((4 : ℂ)^2020) = (1 / 32 : ℂ) :=
sorry

end product_power_conjecture_calculate_expression_l95_95446


namespace job_completion_time_l95_95481

def time (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

noncomputable def start_time : ℕ := time 9 45
noncomputable def half_completion_time : ℕ := time 13 0  -- 1:00 PM in 24-hour time format

theorem job_completion_time :
  ∃ finish_time, finish_time = time 16 15 ∧
  (half_completion_time - start_time) * 2 = finish_time - start_time :=
by
  sorry

end job_completion_time_l95_95481


namespace smallest_winning_N_and_digit_sum_l95_95086

-- Definitions of operations
def B (x : ℕ) : ℕ := 3 * x
def S (x : ℕ) : ℕ := x + 100

/-- The main theorem confirming the smallest winning number and sum of its digits -/
theorem smallest_winning_N_and_digit_sum :
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ (900 ≤ 9 * N + 400 ∧ 9 * N + 400 < 1000) ∧ (N = 56) ∧ (5 + 6 = 11) :=
by {
  -- Proof skipped
  sorry
}

end smallest_winning_N_and_digit_sum_l95_95086


namespace nine_div_repeating_decimal_l95_95050

noncomputable def repeating_decimal := 1 / 3

theorem nine_div_repeating_decimal : 9 / repeating_decimal = 27 := by
  sorry

end nine_div_repeating_decimal_l95_95050


namespace number_of_triangles_and_squares_l95_95043

theorem number_of_triangles_and_squares (x y : ℕ) (h1 : x + y = 13) (h2 : 3 * x + 4 * y = 47) : 
  x = 5 ∧ y = 8 :=
by
  sorry

end number_of_triangles_and_squares_l95_95043


namespace blue_balls_needed_l95_95685

-- Conditions
variables (R Y B W : ℝ)
axiom h1 : 2 * R = 5 * B
axiom h2 : 3 * Y = 7 * B
axiom h3 : 9 * B = 6 * W

-- Proof Problem
theorem blue_balls_needed : (3 * R + 4 * Y + 3 * W) = (64 / 3) * B := by
  sorry

end blue_balls_needed_l95_95685


namespace harry_fish_count_l95_95011

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l95_95011


namespace base_angle_of_isosceles_triangle_l95_95121

theorem base_angle_of_isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = 180) (h_isosceles : A = B ∨ B = C ∨ A = C) (h_angle : A = 42 ∨ B = 42 ∨ C = 42) :
  A = 42 ∨ A = 69 ∨ B = 42 ∨ B = 69 ∨ C = 42 ∨ C = 69 :=
by
  sorry

end base_angle_of_isosceles_triangle_l95_95121


namespace first_term_of_geometric_series_l95_95601

theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) :
  r = 1 / 4 → S = 20 → S = a / (1 - r) → a = 15 :=
by
  intro hr hS hsum
  sorry

end first_term_of_geometric_series_l95_95601


namespace binom_9_5_equals_126_l95_95089

theorem binom_9_5_equals_126 : Nat.binom 9 5 = 126 := 
by 
  sorry

end binom_9_5_equals_126_l95_95089


namespace distance_between_wheels_l95_95958

theorem distance_between_wheels 
  (D : ℕ) 
  (back_perimeter : ℕ) (front_perimeter : ℕ) 
  (more_revolutions : ℕ)
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : more_revolutions = 10)
  (h4 : D / front_perimeter = D / back_perimeter + more_revolutions) : 
  D = 315 :=
by
  sorry

end distance_between_wheels_l95_95958


namespace pieces_per_box_l95_95362

theorem pieces_per_box 
  (a : ℕ) -- Adam bought 13 boxes of chocolate candy 
  (g : ℕ) -- Adam gave 7 boxes to his little brother 
  (p : ℕ) -- Adam still has 36 pieces 
  (n : ℕ) (b : ℕ) 
  (h₁ : a = 13) 
  (h₂ : g = 7) 
  (h₃ : p = 36) 
  (h₄ : n = a - g) 
  (h₅ : p = n * b) 
  : b = 6 :=
by 
  sorry

end pieces_per_box_l95_95362


namespace product_in_M_l95_95633

def M : Set ℤ := {x | ∃ (a b : ℤ), x = a^2 - b^2}

theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M :=
by
  sorry

end product_in_M_l95_95633


namespace value_of_sum_ratio_l95_95652

theorem value_of_sum_ratio (w x y: ℝ) (hx: w / x = 1 / 3) (hy: w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end value_of_sum_ratio_l95_95652


namespace part_a_prob_part_b_expected_time_l95_95993

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l95_95993


namespace price_of_brand_y_pen_l95_95502

-- Definitions based on the conditions
def num_brand_x_pens : ℕ := 8
def price_per_brand_x_pen : ℝ := 4.0
def total_spent : ℝ := 40.0
def total_pens : ℕ := 12

-- price of brand Y that needs to be proven
def price_per_brand_y_pen : ℝ := 2.0

-- Proof statement
theorem price_of_brand_y_pen :
  let num_brand_y_pens := total_pens - num_brand_x_pens
  let spent_on_brand_x_pens := num_brand_x_pens * price_per_brand_x_pen
  let spent_on_brand_y_pens := total_spent - spent_on_brand_x_pens
  spent_on_brand_y_pens / num_brand_y_pens = price_per_brand_y_pen :=
by
  sorry

end price_of_brand_y_pen_l95_95502


namespace ann_has_30_more_cards_than_anton_l95_95366

theorem ann_has_30_more_cards_than_anton (heike_cards : ℕ) (anton_cards : ℕ) (ann_cards : ℕ) 
  (h1 : anton_cards = 3 * heike_cards)
  (h2 : ann_cards = 6 * heike_cards)
  (h3 : ann_cards = 60) : ann_cards - anton_cards = 30 :=
by
  sorry

end ann_has_30_more_cards_than_anton_l95_95366


namespace jayson_age_l95_95469

/-- When Jayson is a certain age J, his dad is four times his age,
    and his mom is 2 years younger than his dad. Jayson's mom was
    28 years old when he was born. Prove that Jayson is 10 years old
    when his dad is four times his age. -/
theorem jayson_age {J : ℕ} (h1 : ∀ J, J > 0 → J * 4 < J + 4) 
                   (h2 : ∀ J, (4 * J - 2) = J + 28) 
                   (h3 : J - (4 * J - 28) = 0): 
                   J = 10 :=
by 
  sorry

end jayson_age_l95_95469


namespace inequality_negative_solution_l95_95522

theorem inequality_negative_solution (a : ℝ) (h : a ≥ -17/4 ∧ a < 4) : 
  ∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a| :=
by
  sorry

end inequality_negative_solution_l95_95522


namespace math_proof_problems_l95_95116

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  (sin (π - α) - 2 * sin (π / 2 + α) = 0) → (sin α * cos α + sin α ^ 2 = 6 / 5)

noncomputable def problem2 (α β : ℝ) : Prop :=
  (tan (α + β) = -1) → (tan α = 2) → (tan β = 3)

-- Example of how to state these problems as a theorem
theorem math_proof_problems (α β : ℝ) : problem1 α ∧ problem2 α β := by
  sorry

end math_proof_problems_l95_95116


namespace solve_quadratic_l95_95702

   theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 5 * x^2 + 8 * x - 24 = 0) : x = 6 / 5 :=
   sorry
   
end solve_quadratic_l95_95702


namespace equation_of_line_l_l95_95591

theorem equation_of_line_l :
  (∃ l : ℝ → ℝ → Prop, 
     (∀ x y, l x y ↔ (x - y + 3) = 0)
     ∧ (∀ x y, l x y → x^2 + (y - 3)^2 = 4)
     ∧ (∀ x y, l x y → x + y + 1 = 0)) :=
sorry

end equation_of_line_l_l95_95591


namespace probability_of_product_divisible_by_3_l95_95111

noncomputable def probability_product_divisible_by_3 : ℚ :=
  let outcomes := 6 * 6
  let favorable_outcomes := outcomes - (4 * 4)
  favorable_outcomes / outcomes

theorem probability_of_product_divisible_by_3 (d1 d2 : Fin 6) :
  (d1 * d2) % 3 = 0 :=
sorry

end probability_of_product_divisible_by_3_l95_95111


namespace sugar_used_in_two_minutes_l95_95190

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l95_95190


namespace cos_150_eq_neg_sqrt3_div_2_l95_95839

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95839


namespace percentage_passed_in_all_three_subjects_l95_95146

-- Define the given failed percentages as real numbers
def A : ℝ := 0.25  -- 25%
def B : ℝ := 0.48  -- 48%
def C : ℝ := 0.35  -- 35%
def AB : ℝ := 0.27 -- 27%
def AC : ℝ := 0.20 -- 20%
def BC : ℝ := 0.15 -- 15%
def ABC : ℝ := 0.10 -- 10%

-- State the theorem to prove the percentage of students who passed in all three subjects
theorem percentage_passed_in_all_three_subjects : 
  1 - (A + B + C - AB - AC - BC + ABC) = 0.44 :=
by
  sorry

end percentage_passed_in_all_three_subjects_l95_95146


namespace radius_comparison_l95_95883

theorem radius_comparison 
  (a b c : ℝ)
  (da db dc r ρ : ℝ)
  (h₁ : da ≤ r)
  (h₂ : db ≤ r)
  (h₃ : dc ≤ r)
  (h₄ : 1 / 2 * (a * da + b * db + c * dc) = ρ * ((a + b + c) / 2)) :
  r ≥ ρ := 
sorry

end radius_comparison_l95_95883


namespace gcd_204_85_l95_95032

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l95_95032


namespace intersection_M_N_l95_95127

def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | |x| > 1}

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l95_95127


namespace factorize_expr1_factorize_expr2_l95_95377

-- Proof Problem 1
theorem factorize_expr1 (a : ℝ) : 
  (a^2 - 4 * a + 4 - 4 * (a - 2) + 4) = (a - 4)^2 :=
sorry

-- Proof Problem 2
theorem factorize_expr2 (x y : ℝ) : 
  16 * x^4 - 81 * y^4 = (4 * x^2 + 9 * y^2) * (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

end factorize_expr1_factorize_expr2_l95_95377


namespace evaluate_g_at_3_l95_95530

def g (x: ℝ) := 5 * x^3 - 4 * x^2 + 3 * x - 7

theorem evaluate_g_at_3 : g 3 = 101 :=
by 
  sorry

end evaluate_g_at_3_l95_95530


namespace stock_percentage_l95_95417

theorem stock_percentage (investment income : ℝ) (investment total : ℝ) (P : ℝ) : 
  (income = 3800) → (total = 15200) → (income = (total * P) / 100) → P = 25 :=
by
  intros h1 h2 h3
  sorry

end stock_percentage_l95_95417


namespace mean_median_mode_l95_95879

theorem mean_median_mode (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : m + 7 < n) 
  (h4 : (m + (m + 3) + (m + 7) + n + (n + 5) + (2 * n - 1)) / 6 = n)
  (h5 : ((m + 7) + n) / 2 = n)
  (h6 : (m+3 < m+7 ∧ m+7 = n ∧ n < n+5 ∧ n+5 < 2*n - 1 )) :
  m+n = 2*n := by
  sorry

end mean_median_mode_l95_95879


namespace cube_root_of_5_irrational_l95_95586

theorem cube_root_of_5_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ)^3 = 5 * (q : ℚ)^3 := by
  sorry

end cube_root_of_5_irrational_l95_95586


namespace paper_boat_travel_time_l95_95914

theorem paper_boat_travel_time :
  ∀ (length_of_embankment : ℝ) (length_of_motorboat : ℝ)
    (time_downstream : ℝ) (time_upstream : ℝ) (v_boat : ℝ) (v_current : ℝ),
  length_of_embankment = 50 →
  length_of_motorboat = 10 →
  time_downstream = 5 →
  time_upstream = 4 →
  v_boat + v_current = length_of_embankment / time_downstream →
  v_boat - v_current = length_of_embankment / time_upstream →
  let speed_paper_boat := v_current in
  let travel_time := length_of_embankment / speed_paper_boat in
  travel_time = 40 :=
by
  intros length_of_embankment length_of_motorboat time_downstream time_upstream v_boat v_current
  intros h_length_emb h_length_motor t_down t_up h_v_boat_plus_current h_v_boat_minus_current
  let speed_paper_boat := v_current
  let travel_time := length_of_embankment / speed_paper_boat
  sorry

end paper_boat_travel_time_l95_95914


namespace cos_150_eq_neg_sqrt3_div_2_l95_95800

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95800


namespace triangle_inequality_l95_95047
-- Import necessary libraries

-- Define the problem
theorem triangle_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (α β γ : ℝ) (h_alpha : α = 2 * Real.sqrt (b * c)) (h_beta : β = 2 * Real.sqrt (c * a)) (h_gamma : γ = 2 * Real.sqrt (a * b)) :
  (a / α) + (b / β) + (c / γ) ≥ (3 / 2) :=
by
  sorry

end triangle_inequality_l95_95047


namespace highest_lowest_difference_l95_95246

variable (x1 x2 x3 x4 x5 x_max x_min : ℝ)

theorem highest_lowest_difference (h1 : x1 + x2 + x3 + x4 + x5 - x_max = 37.84)
                                  (h2 : x1 + x2 + x3 + x4 + x5 - x_min = 38.64):
                                  x_max - x_min = 0.8 := 
by
  sorry

end highest_lowest_difference_l95_95246


namespace samantha_bedtime_l95_95562

-- Defining the conditions
def sleeps_for := 6 -- Samantha sleeps for 6 hours
def wakes_up_time : Time := Time.mk 11 0 -- Samantha woke up at 11:00 AM

-- Define the expected answer
def bedtime : Time := Time.mk 5 0 -- Should be 5:00 AM

-- The theorem we need to prove
theorem samantha_bedtime : 
  (wakes_up_time - Time.mk sleeps_for 0) = bedtime := sorry

end samantha_bedtime_l95_95562


namespace compare_abc_l95_95248

noncomputable def a : ℝ := (2 / 5) ^ (3 / 5)
noncomputable def b : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def c : ℝ := (3 / 5) ^ (2 / 5)

theorem compare_abc : a < b ∧ b < c := sorry

end compare_abc_l95_95248


namespace number_of_n_l95_95873

theorem number_of_n (n : ℕ) (h1 : n > 0) (h2 : n ≤ 1200) (h3 : ∃ k : ℕ, 12 * n = k^2) :
  ∃ m : ℕ, m = 10 :=
by { sorry }

end number_of_n_l95_95873


namespace calculation_correct_l95_95672

def f (x : ℚ) := (2 * x^2 + 6 * x + 9) / (x^2 + 3 * x + 5)
def g (x : ℚ) := 2 * x + 1

theorem calculation_correct : f (g 2) + g (f 2) = 308 / 45 := by
  sorry

end calculation_correct_l95_95672


namespace pipe_B_fills_6_times_faster_l95_95319

theorem pipe_B_fills_6_times_faster :
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  (R_B / R_A = 6) :=
by
  let R_A := 1 / 32
  let combined_rate := 7 / 32
  let R_B := combined_rate - R_A
  sorry

end pipe_B_fills_6_times_faster_l95_95319


namespace probability_businessmen_wait_two_minutes_l95_95987

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l95_95987


namespace some_number_is_l95_95286

theorem some_number_is (x some_number : ℤ) (h1 : x = 4) (h2 : 5 * x + 3 = 10 * x - some_number) : some_number = 17 := by
  sorry

end some_number_is_l95_95286


namespace sarah_shampoo_and_conditioner_usage_l95_95693

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l95_95693


namespace complement_union_M_N_eq_16_l95_95129

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subsets M and N
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

-- Define the union of M and N
def unionMN : Set ℕ := M ∪ N

-- Define the complement of M ∪ N in U
def complementUnionMN : Set ℕ := U \ unionMN

-- State the theorem that the complement is {1, 6}
theorem complement_union_M_N_eq_16 : complementUnionMN = {1, 6} := by
  sorry

end complement_union_M_N_eq_16_l95_95129


namespace certain_number_divisible_l95_95903

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end certain_number_divisible_l95_95903


namespace sin_2017pi_div_3_l95_95574

theorem sin_2017pi_div_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := 
  sorry

end sin_2017pi_div_3_l95_95574


namespace one_positive_real_solution_l95_95132

theorem one_positive_real_solution : 
    ∃! x : ℝ, 0 < x ∧ (x ^ 10 + 7 * x ^ 9 + 14 * x ^ 8 + 1729 * x ^ 7 - 1379 * x ^ 6 = 0) :=
sorry

end one_positive_real_solution_l95_95132


namespace find_a_l95_95651

noncomputable def f (a x : ℝ) : ℝ := Real.log (a * x + 1)

theorem find_a {a : ℝ} (h : (deriv (f a) 0 = 1)) : a = 1 :=
by
  -- Proof goes here
  sorry

end find_a_l95_95651


namespace base_6_four_digit_odd_final_digit_l95_95520

-- Definition of the conditions
def four_digit_number (n b : ℕ) : Prop :=
  b^3 ≤ n ∧ n < b^4

def odd_digit (n b : ℕ) : Prop :=
  (n % b) % 2 = 1

-- Problem statement
theorem base_6_four_digit_odd_final_digit :
  four_digit_number 350 6 ∧ odd_digit 350 6 := by
  sorry

end base_6_four_digit_odd_final_digit_l95_95520


namespace article_cost_price_l95_95061

theorem article_cost_price :
  ∃ C : ℝ, 
  (1.05 * C) - 2 = (1.045 * C) ∧ 
  ∃ C_new : ℝ, C_new = (0.95 * C) ∧ ((1.045 * C) = (C_new + 0.1 * C_new)) ∧ C = 400 := 
sorry

end article_cost_price_l95_95061


namespace bus_ride_time_l95_95679

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end bus_ride_time_l95_95679


namespace negation_of_forall_statement_l95_95956

theorem negation_of_forall_statement :
  ¬ (∀ x : ℝ, x^2 + 2 * x ≥ 0) ↔ ∃ x : ℝ, x^2 + 2 * x < 0 := 
by
  sorry

end negation_of_forall_statement_l95_95956


namespace cos_150_eq_neg_sqrt3_div_2_l95_95805

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95805


namespace expression_evaluation_l95_95230

theorem expression_evaluation : -20 + 8 * (5 ^ 2 - 3) = 156 := by
  sorry

end expression_evaluation_l95_95230


namespace axis_of_symmetry_parabola_l95_95453

theorem axis_of_symmetry_parabola : ∀ (x y : ℝ), y = 2 * x^2 → x = 0 :=
by
  sorry

end axis_of_symmetry_parabola_l95_95453


namespace evaluate_trig_expression_l95_95122

theorem evaluate_trig_expression (α : ℝ) (h : Real.tan α = -4/3) : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1 / 7 :=
by
  sorry

end evaluate_trig_expression_l95_95122


namespace common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l95_95755

-- a) Prove the statements about the number of weeks and extra days
theorem common_year_has_52_weeks_1_day: 
  ∀ (days_in_common_year : ℕ), 
  days_in_common_year = 365 → 
  (days_in_common_year / 7 = 52 ∧ days_in_common_year % 7 = 1)
:= by
  sorry

theorem leap_year_has_52_weeks_2_days: 
  ∀ (days_in_leap_year : ℕ), 
  days_in_leap_year = 366 → 
  (days_in_leap_year / 7 = 52 ∧ days_in_leap_year % 7 = 2)
:= by
  sorry

-- b) If a common year starts on a Tuesday, prove the following year starts on a Wednesday
theorem next_year_starts_on_wednesday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (365 % 7 = 1) → 
  ((start_day + 365 % 7) % 7 = 3)
:= by
  sorry

-- c) If a leap year starts on a Tuesday, prove the following year starts on a Thursday
theorem next_year_starts_on_thursday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (366 % 7 = 2) →
  ((start_day + 366 % 7) % 7 = 4)
:= by
  sorry

end common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l95_95755


namespace solution_set_of_inequality_l95_95332

theorem solution_set_of_inequality (x : ℝ) : |x^2 - 2| < 2 ↔ ((-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2)) :=
by sorry

end solution_set_of_inequality_l95_95332


namespace difference_between_first_and_third_l95_95981

variable (x : ℕ)

-- Condition 1: The first number is twice the second.
def first_number : ℕ := 2 * x

-- Condition 2: The first number is three times the third.
def third_number : ℕ := first_number x / 3

-- Condition 3: The average of the three numbers is 88.
def average_condition : Prop := (first_number x + x + third_number x) / 3 = 88

-- Prove that the difference between first and third number is 96.
theorem difference_between_first_and_third 
  (h : average_condition x) : first_number x - third_number x = 96 :=
by
  sorry -- Proof omitted

end difference_between_first_and_third_l95_95981


namespace max_pages_copied_l95_95922

-- Definitions based on conditions
def cents_per_page := 7 / 4
def budget_cents := 1500

-- The theorem to prove
theorem max_pages_copied (c : ℝ) (budget : ℝ) (h₁ : c = cents_per_page) (h₂ : budget = budget_cents) : 
  ⌊(budget / c)⌋ = 857 :=
sorry

end max_pages_copied_l95_95922


namespace stratified_sampling_number_of_products_drawn_l95_95208

theorem stratified_sampling_number_of_products_drawn (T S W X : ℕ) 
  (h1 : T = 1024) (h2 : S = 64) (h3 : W = 128) :
  X = S * (W / T) → X = 8 :=
by
  sorry

end stratified_sampling_number_of_products_drawn_l95_95208


namespace positive_number_l95_95718

theorem positive_number (n : ℕ) (h : n^2 + 2 * n = 170) : n = 12 :=
sorry

end positive_number_l95_95718


namespace area_enclosed_by_region_l95_95730

theorem area_enclosed_by_region :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6*x + 8*y = -9) →
  let radius := 4 in
  let area := real.pi * radius^2 in
  area = 16 * real.pi :=
sorry

end area_enclosed_by_region_l95_95730


namespace circle_length_l95_95169

theorem circle_length (n : ℕ) (arm_span : ℝ) (overlap : ℝ) (contribution : ℝ) (total_length : ℝ) :
  n = 16 ->
  arm_span = 10.4 ->
  overlap = 3.5 ->
  contribution = arm_span - overlap ->
  total_length = n * contribution ->
  total_length = 110.4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end circle_length_l95_95169


namespace Riley_fewer_pairs_l95_95235

-- Define the conditions
def Ellie_pairs : ℕ := 8
def Total_pairs : ℕ := 13

-- Prove the statement
theorem Riley_fewer_pairs : (Total_pairs - Ellie_pairs) - Ellie_pairs = 3 :=
by
  -- Skip the proof
  sorry

end Riley_fewer_pairs_l95_95235


namespace ann_has_30_more_cards_than_anton_l95_95367

theorem ann_has_30_more_cards_than_anton (heike_cards : ℕ) (anton_cards : ℕ) (ann_cards : ℕ) 
  (h1 : anton_cards = 3 * heike_cards)
  (h2 : ann_cards = 6 * heike_cards)
  (h3 : ann_cards = 60) : ann_cards - anton_cards = 30 :=
by
  sorry

end ann_has_30_more_cards_than_anton_l95_95367


namespace average_last_four_numbers_l95_95023

theorem average_last_four_numbers (numbers : List ℝ) 
  (h1 : numbers.length = 7)
  (h2 : (numbers.sum / 7) = 62)
  (h3 : (numbers.take 3).sum / 3 = 58) : 
  ((numbers.drop 3).sum / 4) = 65 :=
by
  sorry

end average_last_four_numbers_l95_95023


namespace LCM_30_45_l95_95765

theorem LCM_30_45 : Nat.lcm 30 45 = 90 := by
  sorry

end LCM_30_45_l95_95765


namespace circles_intersect_and_common_chord_l95_95893

theorem circles_intersect_and_common_chord :
  (∃ P : ℝ × ℝ, P.1 ^ 2 + P.2 ^ 2 - P.1 + P.2 - 2 = 0 ∧
                P.1 ^ 2 + P.2 ^ 2 = 5) ∧
  (∀ x y : ℝ, (x ^ 2 + y ^ 2 - x + y - 2 = 0 ∧ x ^ 2 + y ^ 2 = 5) →
              x - y - 3 = 0) ∧
  (∃ A B : ℝ × ℝ, A.1 ^ 2 + A.2 ^ 2 - A.1 + A.2 - 2 = 0 ∧
                   A.1 ^ 2 + A.2 ^ 2 = 5 ∧
                   B.1 ^ 2 + B.2 ^ 2 - B.1 + B.2 - 2 = 0 ∧
                   B.1 ^ 2 + B.2 ^ 2 = 5 ∧
                   (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 2) := sorry

end circles_intersect_and_common_chord_l95_95893


namespace triangular_prism_distance_sum_l95_95525

theorem triangular_prism_distance_sum (V K H1 H2 H3 H4 S1 S2 S3 S4 : ℝ)
  (h1 : S1 = K)
  (h2 : S2 = 2 * K)
  (h3 : S3 = 3 * K)
  (h4 : S4 = 4 * K)
  (hV : (S1 * H1 + S2 * H2 + S3 * H3 + S4 * H4) / 3 = V) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / K :=
by sorry

end triangular_prism_distance_sum_l95_95525


namespace optimal_floor_optimal_floor_achieved_at_three_l95_95597

theorem optimal_floor : ∀ (n : ℕ), n > 0 → (n + 9 / n : ℝ) ≥ 6 := sorry

theorem optimal_floor_achieved_at_three : ∃ n : ℕ, (n > 0 ∧ (n + 9 / n : ℝ) = 6) := sorry

end optimal_floor_optimal_floor_achieved_at_three_l95_95597


namespace range_of_a_l95_95142

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else 2 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x, 3 ≤ f x a) ∧ (0 < a) ∧ (a ≠ 1) → 1 < a ∧ a ≤ 2 :=
by
  intro h
  sorry

end range_of_a_l95_95142


namespace igor_min_score_needed_l95_95288

theorem igor_min_score_needed
  (scores : List ℕ)
  (goal : ℚ)
  (next_test_score : ℕ)
  (h_scores : scores = [88, 92, 75, 83, 90])
  (h_goal : goal = 87)
  (h_solution : next_test_score = 94)
  : 
  let current_sum := scores.sum
  let current_tests := scores.length
  let required_total := (goal * (current_tests + 1))
  let next_test_needed := required_total - current_sum
  next_test_needed ≤ next_test_score := 
by 
  sorry

end igor_min_score_needed_l95_95288


namespace two_digit_number_representation_l95_95600

def tens_digit := ℕ
def units_digit := ℕ

theorem two_digit_number_representation (b a : ℕ) : 
  (∀ (b a : ℕ), 10 * b + a = 10 * b + a) := sorry

end two_digit_number_representation_l95_95600


namespace part1_part2_part3_l95_95634

-- Conditions
def A : Set ℝ := { x : ℝ | 2 < x ∧ x < 6 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m }

-- Proof statements
theorem part1 : A ∪ B 2 = { x : ℝ | 2 < x ∧ x < 6 } := by
  sorry

theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) → m ≤ 3 := by
  sorry

theorem part3 (m : ℝ) : (∃ x, x ∈ B m) ∧ (∀ x, x ∉ A ∩ B m) → m ≥ 5 := by
  sorry

end part1_part2_part3_l95_95634


namespace smallest_CCD_value_l95_95360

theorem smallest_CCD_value :
  ∃ (C D : ℕ), (C ≠ 0) ∧ (D ≠ C) ∧ (C < 10) ∧ (D < 10) ∧ (110 * C + D = 227) ∧ (10 * C + D = (110 * C + D) / 7) :=
by
  sorry

end smallest_CCD_value_l95_95360


namespace quadratic_has_real_roots_l95_95110

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 := by
  sorry

end quadratic_has_real_roots_l95_95110


namespace common_ratio_of_series_l95_95219

-- Define the terms and conditions for the infinite geometric series problem.
def first_term : ℝ := 500
def series_sum : ℝ := 4000

-- State the theorem that needs to be proven: the common ratio of the series is 7/8.
theorem common_ratio_of_series (a S r : ℝ) (h_a : a = 500) (h_S : S = 4000) (h_eq : S = a / (1 - r)) :
  r = 7 / 8 :=
by
  sorry

end common_ratio_of_series_l95_95219


namespace domain_of_function_l95_95500

def valid_domain (x : ℝ) : Prop :=
  (2 - x ≥ 0) ∧ (x > 0) ∧ (x ≠ 2)

theorem domain_of_function :
  {x : ℝ | ∃ (y : ℝ), y = x ∧ valid_domain x} = {x | 0 < x ∧ x < 2} :=
by
  sorry

end domain_of_function_l95_95500


namespace determine_m_value_l95_95616

-- Define the condition that the roots of the quadratic are given
def quadratic_equation_has_given_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, (8 * x^2 + 4 * x + m = 0) → (x = (-2 + (Complex.I * Real.sqrt 88)) / 8) ∨ (x = (-2 - (Complex.I * Real.sqrt 88)) / 8)

-- The main statement to be proven
theorem determine_m_value (m : ℝ) (h : quadratic_equation_has_given_roots m) : m = 13 / 4 :=
sorry

end determine_m_value_l95_95616


namespace natives_cannot_obtain_910_rupees_with_50_coins_l95_95729

theorem natives_cannot_obtain_910_rupees_with_50_coins (x y z : ℤ) : 
  x + y + z = 50 → 
  10 * x + 34 * y + 62 * z = 910 → 
  false :=
by
  sorry

end natives_cannot_obtain_910_rupees_with_50_coins_l95_95729


namespace fish_caught_in_second_catch_l95_95910

theorem fish_caught_in_second_catch {N x : ℕ} (hN : N = 1750) (hx1 : 70 * x = 2 * N) : x = 50 :=
by
  sorry

end fish_caught_in_second_catch_l95_95910


namespace part_I_part_I_correct_interval_part_II_min_value_l95_95435

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem part_I : ∀ x : ℝ, (f x > 2) ↔ ( x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_I_correct_interval : ∀ x : ℝ, (f x > 2) → (x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_II_min_value : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ ∀ x : ℝ, f x ≥ y := 
sorry

end part_I_part_I_correct_interval_part_II_min_value_l95_95435


namespace zephyr_island_population_capacity_reach_l95_95147

-- Definitions for conditions
def acres := 30000
def acres_per_person := 2
def initial_year := 2023
def initial_population := 500
def population_growth_rate := 4
def growth_period := 20

-- Maximum population supported by the island
def max_population := acres / acres_per_person

-- Function to calculate population after a given number of years
def population (years : ℕ) : ℕ := initial_population * (population_growth_rate ^ (years / growth_period))

-- The Lean statement to prove that the population will reach or exceed max_capacity in 60 years
theorem zephyr_island_population_capacity_reach : ∃ t : ℕ, t ≤ 60 ∧ population t ≥ max_population :=
by
  sorry

end zephyr_island_population_capacity_reach_l95_95147


namespace largest_n_for_factorable_polynomial_l95_95242

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end largest_n_for_factorable_polynomial_l95_95242


namespace tony_comics_average_l95_95971

theorem tony_comics_average :
  let a1 := 10
  let d := 6
  let n := 8
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  (S_n n) / n = 31 := by
  sorry

end tony_comics_average_l95_95971


namespace cos_150_eq_neg_sqrt3_div_2_l95_95851

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95851


namespace paper_boat_travel_time_l95_95916

-- Given conditions
def embankment_length : ℝ := 50
def boat_length : ℝ := 10
def downstream_time : ℝ := 5
def upstream_time : ℝ := 4

-- Derived conditions from the given problem
def downstream_speed := embankment_length / downstream_time
def upstream_speed := embankment_length / upstream_time

-- Prove that the paper boat's travel time is 40 seconds
theorem paper_boat_travel_time :
  let v_boat := (downstream_speed + upstream_speed) / 2 in
  let v_current := (downstream_speed - upstream_speed) / 2 in
  let travel_time := embankment_length / v_current in
  travel_time = 40 := 
  sorry

end paper_boat_travel_time_l95_95916


namespace brownie_cost_l95_95606

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) 
    (total_money = 32) (num_pans = 2) (pieces_per_pan = 8) : 
    (total_money / (num_pans * pieces_per_pan) = 2) := by
  sorry

end brownie_cost_l95_95606


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95857

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95857


namespace parallel_lines_condition_l95_95067

variable {a : ℝ}

theorem parallel_lines_condition (a_is_2 : a = 2) :
  (∀ x y : ℝ, a * x + 2 * y = 0 → x + y = 1) ∧ (∀ x y : ℝ, x + y = 1 → a * x + 2 * y = 0) :=
by
  sorry

end parallel_lines_condition_l95_95067


namespace zebras_total_games_l95_95222

theorem zebras_total_games 
  (x y : ℝ)
  (h1 : x = 0.40 * y)
  (h2 : (x + 8) / (y + 11) = 0.55) 
  : y + 11 = 24 :=
sorry

end zebras_total_games_l95_95222


namespace ratio_of_men_to_women_l95_95165

-- Define constants
def total_people : ℕ := 60
def men_in_meeting : ℕ := 4
def women_in_meeting : ℕ := 6
def women_reduction_percentage : ℕ := 20

-- Statement of the problem
theorem ratio_of_men_to_women (total_people men_in_meeting women_in_meeting women_reduction_percentage: ℕ)
  (total_people_eq : total_people = 60)
  (men_in_meeting_eq : men_in_meeting = 4)
  (women_in_meeting_eq : women_in_meeting = 6)
  (women_reduction_percentage_eq : women_reduction_percentage = 20) :
  (men_in_meeting + ((total_people - men_in_meeting - women_in_meeting) * women_reduction_percentage / 100)) 
  = total_people / 2 :=
sorry

end ratio_of_men_to_women_l95_95165


namespace divisors_of_2700_l95_95623

def prime_factors_2700 : ℕ := 2^2 * 3^3 * 5^2

def number_of_positive_divisors (n : ℕ) (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem divisors_of_2700 : number_of_positive_divisors 2700 2 3 2 = 36 := by
  sorry

end divisors_of_2700_l95_95623


namespace paytons_score_l95_95316

theorem paytons_score (total_score_14_students : ℕ)
    (average_14_students : total_score_14_students / 14 = 80)
    (total_score_15_students : ℕ)
    (average_15_students : total_score_15_students / 15 = 81) :
  total_score_15_students - total_score_14_students = 95 :=
by
  sorry

end paytons_score_l95_95316


namespace jesse_started_with_l95_95544

-- Define the conditions
variables (g e : ℕ)

-- Theorem stating that given the conditions, Jesse started with 78 pencils
theorem jesse_started_with (g e : ℕ) (h1 : g = 44) (h2 : e = 34) : e + g = 78 :=
by sorry

end jesse_started_with_l95_95544


namespace sequence_not_distinct_l95_95927

-- We will state the theorem

theorem sequence_not_distinct (a0 : ℕ) (h : 0 < a0) : 
  ∃ (n m : ℕ), n ≠ m ∧ 
   (let a : ℕ → ℕ := λ b, 
      let ⟨digits, _⟩ := nat.digits 10 b in
      digits.sum (λ c, c ^ 2005)
    in a^[n] a0 = a^[m] a0) :=
begin
  -- The proof will be written here
  sorry
end

end sequence_not_distinct_l95_95927


namespace initial_number_is_nine_l95_95213

theorem initial_number_is_nine (x : ℕ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end initial_number_is_nine_l95_95213


namespace complement_of_A_in_I_is_246_l95_95313

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def complement_A_in_I : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_I_is_246 :
  (universal_set \ set_A) = complement_A_in_I :=
  by sorry

end complement_of_A_in_I_is_246_l95_95313


namespace derivative_at_neg_one_l95_95644

variable (a b : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Given condition f'(1) = 2
axiom h : f' a b 1 = 2

-- Statement to prove f'(-1) = -2
theorem derivative_at_neg_one : f' a b (-1) = -2 :=
by 
  sorry

end derivative_at_neg_one_l95_95644


namespace correct_calculation_l95_95584

theorem correct_calculation : -Real.sqrt ((-5)^2) = -5 := 
by 
  sorry

end correct_calculation_l95_95584


namespace prime_condition_l95_95940

theorem prime_condition (p : ℕ) (hp : Nat.Prime p) (h2p : Nat.Prime (p + 2)) : p = 3 ∨ 6 ∣ (p + 1) := 
sorry

end prime_condition_l95_95940


namespace excircles_touch_midline_properties_l95_95175

noncomputable def midpoint (P Q : Point) : Point := 
  sorry

theorem excircles_touch_midline_properties 
(A B C K L : Point)
(h1 : ExcircleTouches AC B K)
(h2 : ExcircleTouches BC A L) :
∃ M N : Point,
  (is_midpoint M K L) ∧ (is_midpoint N A B) ∧
  (divides_perimeter_half (segment M N) (triangle A B C)) ∧
  (parallel (segment M N) (angle_bisector ∠ACB)) :=
by
  sorry

end excircles_touch_midline_properties_l95_95175


namespace sugar_used_in_two_minutes_l95_95191

-- Definitions according to conditions
def sugar_per_bar : ℝ := 1.5
def bars_per_minute : ℝ := 36
def minutes : ℝ := 2

-- Theorem statement
theorem sugar_used_in_two_minutes : bars_per_minute * sugar_per_bar * minutes = 108 :=
by
  -- We add sorry here to complete the proof later.
  sorry

end sugar_used_in_two_minutes_l95_95191


namespace calculation_problem_l95_95224

theorem calculation_problem : (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
by sorry

end calculation_problem_l95_95224


namespace heavy_operators_earn_129_dollars_per_day_l95_95083

noncomputable def heavy_operator_daily_wage (H : ℕ) : Prop :=
  let laborer_wage := 82
  let total_people := 31
  let total_payroll := 3952
  let laborers_count := 1
  let heavy_operators_count := total_people - laborers_count
  let heavy_operators_payroll := total_payroll - (laborer_wage * laborers_count)
  H = heavy_operators_payroll / heavy_operators_count

theorem heavy_operators_earn_129_dollars_per_day : heavy_operator_daily_wage 129 :=
by
  unfold heavy_operator_daily_wage
  sorry

end heavy_operators_earn_129_dollars_per_day_l95_95083


namespace words_fully_lit_probability_l95_95605

/-- Definition stating the display states of the words "I", "love", "Gaoyou" --/
inductive LightState
  | allOff
  | loveOn
  | loveAndIOff
  | loveAndGaoyouOff
  | allLit

/-- Probability of the words being fully lit given the lighting conditions. --/
def words_lit_probability : ℚ :=
  1 / 3

theorem words_fully_lit_probability
  (initial_state : LightState)
  (∀s : LightState, s = LightState.loveOn → s = LightState.allLit ∨ s = LightState.loveAndIOff ∨ s = LightState.loveAndGaoyouOff) :
  initial_state = LightState.allLit → words_lit_probability = 1 / 3 :=
by
  sorry

end words_fully_lit_probability_l95_95605


namespace trigonometric_identity_proof_l95_95320

theorem trigonometric_identity_proof :
  3.438 * (Real.sin (84 * Real.pi / 180)) * (Real.sin (24 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * (Real.sin (12 * Real.pi / 180)) = 1 / 16 :=
  sorry

end trigonometric_identity_proof_l95_95320


namespace total_volume_calculation_l95_95976

noncomputable def total_volume_of_four_cubes (edge_length_in_feet : ℝ) (conversion_factor : ℝ) : ℝ :=
  let edge_length_in_meters := edge_length_in_feet * conversion_factor
  let volume_of_one_cube := edge_length_in_meters^3
  4 * volume_of_one_cube

theorem total_volume_calculation :
  total_volume_of_four_cubes 5 0.3048 = 14.144 :=
by
  -- Proof needs to be filled in.
  sorry

end total_volume_calculation_l95_95976


namespace total_cost_of_apples_l95_95963

theorem total_cost_of_apples (cost_per_kg : ℝ) (packaging_fee : ℝ) (weight : ℝ) :
  cost_per_kg = 15.3 →
  packaging_fee = 0.25 →
  weight = 2.5 →
  (weight * (cost_per_kg + packaging_fee) = 38.875) :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_apples_l95_95963


namespace number_of_squares_sharing_two_vertices_l95_95000

-- Given conditions
def right_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (AB BC AC : ℝ),
  AB = BC ∧
  angle A B C = π / 2 ∧
  AB ^ 2 + BC ^ 2 = AC ^ 2

-- Desired proof problem
theorem number_of_squares_sharing_two_vertices
  (A B C : ℝ × ℝ) (h : right_isosceles_triangle A B C) :
  ∃ n, n = 2 :=
begin
  sorry
end

end number_of_squares_sharing_two_vertices_l95_95000


namespace inequality_proof_l95_95930

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
  sorry

end inequality_proof_l95_95930


namespace value_of_s_in_base_b_l95_95552

noncomputable def b : ℕ :=
  10

def fourteen_in_b (b : ℕ) : ℕ :=
  b + 4

def seventeen_in_b (b : ℕ) : ℕ :=
  b + 7

def eighteen_in_b (b : ℕ) : ℕ :=
  b + 8

def five_thousand_four_and_four_in_b (b : ℕ) : ℕ :=
  5 * b ^ 3 + 4 * b ^ 2 + 4

def product_in_base_b_equals (b : ℕ) : Prop :=
  (fourteen_in_b b) * (seventeen_in_b b) * (eighteen_in_b b) = five_thousand_four_and_four_in_b b

def s_in_base_b (b : ℕ) : ℕ :=
  fourteen_in_b b + seventeen_in_b b + eighteen_in_b b

theorem value_of_s_in_base_b (b : ℕ) (h : product_in_base_b_equals b) : s_in_base_b b = 49 := by
  sorry

end value_of_s_in_base_b_l95_95552


namespace valid_four_digit_numbers_count_l95_95268

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l95_95268


namespace boat_travels_125_km_downstream_l95_95754

/-- The speed of the boat in still water is 20 km/hr -/
def boat_speed_still_water : ℝ := 20

/-- The speed of the stream is 5 km/hr -/
def stream_speed : ℝ := 5

/-- The total time taken downstream is 5 hours -/
def total_time_downstream : ℝ := 5

/-- The effective speed of the boat downstream -/
def effective_speed_downstream : ℝ := boat_speed_still_water + stream_speed

/-- The distance the boat travels downstream -/
def distance_downstream : ℝ := effective_speed_downstream * total_time_downstream

/-- The boat travels 125 km downstream -/
theorem boat_travels_125_km_downstream :
  distance_downstream = 125 := 
sorry

end boat_travels_125_km_downstream_l95_95754


namespace compare_exp_square_l95_95519

theorem compare_exp_square (n : ℕ) : 
  (n ≥ 3 → 2^(2 * n) > (2 * n + 1)^2) ∧ ((n = 1 ∨ n = 2) → 2^(2 * n) < (2 * n + 1)^2) :=
by
  sorry

end compare_exp_square_l95_95519


namespace max_travel_distance_l95_95385

theorem max_travel_distance (D_F D_R : ℕ) (hF : D_F = 21000) (hR : D_R = 28000) : ∃ D_max, D_max = 24000 :=
by
  let x := 10500
  let y := 10500
  have D_max := x + y
  have hD_max : D_max = 21000 := by sorry
  exact ⟨D_max, hD_max⟩

end max_travel_distance_l95_95385


namespace new_average_age_l95_95705

/--
The average age of 7 people in a room is 28 years.
A 22-year-old person leaves the room, and a 30-year-old person enters the room.
Prove that the new average age of the people in the room is \( 29 \frac{1}{7} \).
-/
theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (entering_age : ℕ)
  (H1 : avg_age = 28)
  (H2 : num_people = 7)
  (H3 : leaving_age = 22)
  (H4 : entering_age = 30) :
  (avg_age * num_people - leaving_age + entering_age) / num_people = 29 + 1 / 7 := 
by
  sorry

end new_average_age_l95_95705


namespace find_number_l95_95746

theorem find_number (x : ℤ) : (150 - x = x + 68) → x = 41 :=
by
  intro h
  sorry

end find_number_l95_95746


namespace sarah_shampoo_and_conditioner_usage_l95_95694

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end sarah_shampoo_and_conditioner_usage_l95_95694


namespace total_birds_seen_l95_95753

theorem total_birds_seen (M T W : ℕ) 
  (hM : M = 70)
  (hT : T = M / 2)
  (hW : W = T + 8) : 
  M + T + W = 148 :=
by
  subst hM
  subst hT
  subst hW
  have hT' : 35 = 70 / 2 := by norm_num
  have hW' : 43 = 35 + 8 := by norm_num
  rw [hT', hW']
  norm_num

end total_birds_seen_l95_95753


namespace cos_150_eq_neg_sqrt3_div_2_l95_95801

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95801


namespace Sara_quarters_after_borrowing_l95_95324

theorem Sara_quarters_after_borrowing (initial_quarters borrowed_quarters : ℕ) (h1 : initial_quarters = 783) (h2 : borrowed_quarters = 271) :
  initial_quarters - borrowed_quarters = 512 := by
  sorry

end Sara_quarters_after_borrowing_l95_95324


namespace trapezium_second_side_length_l95_95620

-- Define the problem in Lean
variables (a h A b : ℝ)

-- Define the conditions
def conditions : Prop :=
  a = 20 ∧ h = 25 ∧ A = 475

-- Prove the length of the second parallel side
theorem trapezium_second_side_length (h_cond : conditions a h A) : b = 18 :=
by
  sorry

end trapezium_second_side_length_l95_95620


namespace tyrone_gave_marbles_l95_95578

theorem tyrone_gave_marbles :
  ∃ x : ℝ, (120 - x = 3 * (30 + x)) ∧ x = 7.5 :=
by
  sorry

end tyrone_gave_marbles_l95_95578


namespace xy_sum_l95_95624

theorem xy_sum (x y : ℝ) (h1 : x^3 - 6 * x^2 + 15 * x = 12) (h2 : y^3 - 6 * y^2 + 15 * y = 16) : x + y = 4 := 
sorry

end xy_sum_l95_95624


namespace min_value_expression_l95_95622

theorem min_value_expression : ∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 = 2 := 
sorry

end min_value_expression_l95_95622


namespace minimum_value_fraction_l95_95894

theorem minimum_value_fraction (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_parallel : m / (4 - n) = 1 / 2) : 
  (1 / m + 8 / n) ≥ 9 / 2 :=
by
  sorry

end minimum_value_fraction_l95_95894


namespace jana_walking_distance_l95_95302

theorem jana_walking_distance (t_walk_mile : ℝ) (speed : ℝ) (time : ℝ) (distance : ℝ) :
  t_walk_mile = 24 → speed = 1 / t_walk_mile → time = 36 → distance = speed * time → distance = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end jana_walking_distance_l95_95302


namespace train_avg_speed_l95_95348

variable (x : ℝ)

def avg_speed_of_train (x : ℝ) : ℝ := 3

theorem train_avg_speed (h : x > 0) : avg_speed_of_train x / (x / 7.5) = 22.5 :=
  sorry

end train_avg_speed_l95_95348


namespace bus_ride_time_l95_95680

def walking_time : ℕ := 15
def waiting_time : ℕ := 2 * walking_time
def train_ride_time : ℕ := 360
def total_trip_time : ℕ := 8 * 60

theorem bus_ride_time : 
  (total_trip_time - (walking_time + waiting_time + train_ride_time)) = 75 := by
  sorry

end bus_ride_time_l95_95680


namespace find_f2_l95_95666

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (f x) = (x ^ 2 - x) / 2 * f x + 2 - x

theorem find_f2 : f 2 = 2 :=
by
  sorry

end find_f2_l95_95666


namespace problem_a_problem_b_problem_c_l95_95521

open Real

noncomputable def conditions (x : ℝ) := x >= 1 / 2

/-- 
a) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = \sqrt{2} \)
valid if and only if x in [1/2, 1].
-/
theorem problem_a (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = sqrt 2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) :=
  sorry

/-- 
b) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 1 \)
has no solution.
-/
theorem problem_b (x : ℝ) (h : conditions x) :
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 1 → False :=
  sorry

/-- 
c) Prove \( \sqrt{x + \sqrt{2x - 1}} + \sqrt{x - \sqrt{2x - 1}} = 2 \)
if and only if x = 3/2.
-/
theorem problem_c (x : ℝ) (h : conditions x) :
  (sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = 2) ↔ (x = 3 / 2) :=
  sorry

end problem_a_problem_b_problem_c_l95_95521


namespace star_value_example_l95_95631

def star (a b c : ℤ) : ℤ := (a + b + c) ^ 2

theorem star_value_example : star 3 (-5) 2 = 0 :=
by
  sorry

end star_value_example_l95_95631


namespace smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l95_95737

def degree_movement_per_minute_of_minute_hand : ℝ := 6
def degree_movement_per_hour_of_hour_hand : ℝ := 30
def degree_movement_per_minute_of_hour_hand : ℝ := 0.5

def minute_position_at_3_40_pm : ℝ := 40 * degree_movement_per_minute_of_minute_hand
def hour_position_at_3_40_pm : ℝ := 3 * degree_movement_per_hour_of_hour_hand + 40 * degree_movement_per_minute_of_hour_hand

def clockwise_angle_between_hands_at_3_40_pm : ℝ := minute_position_at_3_40_pm - hour_position_at_3_40_pm
def counterclockwise_angle_between_hands_at_3_40_pm : ℝ := 360 - clockwise_angle_between_hands_at_3_40_pm

theorem smaller_angle_between_hands_at_3_40_pm : clockwise_angle_between_hands_at_3_40_pm = 130.0 := 
by
  sorry

theorem larger_angle_between_hands_at_3_40_pm : counterclockwise_angle_between_hands_at_3_40_pm = 230.0 := 
by
  sorry

end smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l95_95737


namespace distance_from_focus_l95_95252

theorem distance_from_focus (x : ℝ) (A : ℝ × ℝ) (hA_on_parabola : A.1^2 = 4 * A.2) (hA_coord : A.2 = 4) : 
  dist A (0, 1) = 5 := 
by
  sorry

end distance_from_focus_l95_95252


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95994

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95994


namespace math_problem_l95_95155

theorem math_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^3 + y^3 = x - y) : x^2 + 4 * y^2 < 1 := 
sorry

end math_problem_l95_95155


namespace male_salmon_count_l95_95184

theorem male_salmon_count (total_salmon : ℕ) (female_salmon : ℕ) (male_salmon : ℕ) 
  (h1 : total_salmon = 971639) 
  (h2 : female_salmon = 259378) 
  (h3 : male_salmon = total_salmon - female_salmon) : 
  male_salmon = 712261 :=
by
  sorry

end male_salmon_count_l95_95184


namespace square_area_with_circles_l95_95078

theorem square_area_with_circles 
  (radius : ℝ) 
  (circle_count : ℕ) 
  (side_length : ℝ) 
  (total_area : ℝ)
  (h1 : radius = 7)
  (h2 : circle_count = 4)
  (h3 : side_length = 2 * (2 * radius))
  (h4 : total_area = side_length * side_length)
  : total_area = 784 :=
sorry

end square_area_with_circles_l95_95078


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95858

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95858


namespace circle_area_l95_95559

noncomputable def pointA : ℝ × ℝ := (2, 7)
noncomputable def pointB : ℝ × ℝ := (8, 5)

def is_tangent_with_intersection_on_x_axis (A B C : ℝ × ℝ) : Prop :=
  ∃ R : ℝ, ∃ r : ℝ, ∀ M : ℝ × ℝ, dist M C = R → dist A M = r ∧ dist B M = r

theorem circle_area (A B : ℝ × ℝ) (hA : A = (2, 7)) (hB : B = (8, 5))
    (h : ∃ C : ℝ × ℝ, is_tangent_with_intersection_on_x_axis A B C) 
    : ∃ R : ℝ, π * R^2 = 12.5 * π := 
sorry

end circle_area_l95_95559


namespace football_games_per_month_l95_95969

theorem football_games_per_month :
  let total_games := 5491
  let months := 17.0
  total_games / months = 323 := 
by
  let total_games := 5491
  let months := 17.0
  -- This is where the actual computation would happen if we were to provide a proof
  sorry

end football_games_per_month_l95_95969


namespace sally_book_pages_l95_95561

def pages_read_weekdays (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def pages_read_weekends (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def total_pages (weekdays: ℕ) (weekends: ℕ) (pages_weekdays: ℕ) (pages_weekends: ℕ): ℕ :=
  pages_read_weekdays weekdays pages_weekdays + pages_read_weekends weekends pages_weekends

theorem sally_book_pages :
  total_pages 10 4 10 20 = 180 :=
sorry

end sally_book_pages_l95_95561


namespace unique_root_when_abs_t_gt_2_l95_95066

theorem unique_root_when_abs_t_gt_2 (t : ℝ) (h : |t| > 2) :
  ∃! x : ℝ, x^3 - 3 * x = t ∧ |x| > 2 :=
sorry

end unique_root_when_abs_t_gt_2_l95_95066


namespace largest_triangle_perimeter_l95_95217

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem largest_triangle_perimeter : 
  ∃ (x : ℕ), x ≤ 14 ∧ 2 ≤ x ∧ is_valid_triangle 7 8 x ∧ (7 + 8 + x = 29) :=
sorry

end largest_triangle_perimeter_l95_95217


namespace solution_l95_95748

noncomputable def problem_statement (n : ℕ) : Prop :=
  ∀ P : Polynomial ℤ,
  P.degree = n →
  ∃ a b : ℕ, a ≠ b ∧ (P.eval a + P.eval b) % (a + b) = 0

theorem solution :
  ∀ n : ℕ, ( even n ) → problem_statement n :=
by
  intros n hn_even
  sorry

end solution_l95_95748


namespace probability_at_least_one_die_shows_three_l95_95725

theorem probability_at_least_one_die_shows_three : 
  let outcomes := 36
  let not_three_outcomes := 25
  (outcomes - not_three_outcomes) / outcomes = 11 / 36 := sorry

end probability_at_least_one_die_shows_three_l95_95725


namespace sqrt_inequality_sum_inverse_ge_9_l95_95201

-- (1) Prove that \(\sqrt{3} + \sqrt{8} < 2 + \sqrt{7}\)
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := sorry

-- (2) Prove that given \(a > 0, b > 0, c > 0\) and \(a + b + c = 1\), \(\frac{1}{a} + \frac{1}{b} + \frac{1}{c} \geq 9\)
theorem sum_inverse_ge_9 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) : 
    1 / a + 1 / b + 1 / c ≥ 9 := sorry

end sqrt_inequality_sum_inverse_ge_9_l95_95201


namespace max_points_on_four_coplanar_circles_l95_95628

noncomputable def max_points_on_circles (num_circles : ℕ) (max_intersections : ℕ) : ℕ :=
num_circles * max_intersections

theorem max_points_on_four_coplanar_circles :
  max_points_on_circles 4 2 = 8 := 
sorry

end max_points_on_four_coplanar_circles_l95_95628


namespace lucy_current_fish_l95_95933

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end lucy_current_fish_l95_95933


namespace cos_150_eq_neg_sqrt3_over_2_l95_95814

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95814


namespace sugar_used_in_two_minutes_l95_95187

-- Define constants for problem conditions
def sugar_per_bar : Float := 1.5
def bars_per_minute : Nat := 36
def minutes : Nat := 2

-- Define the total sugar used in two minutes
def total_sugar_used : Float := (bars_per_minute * sugar_per_bar) * minutes

-- State the theorem and its proof
theorem sugar_used_in_two_minutes : total_sugar_used = 108 := by
  sorry

end sugar_used_in_two_minutes_l95_95187


namespace binom_9_5_l95_95092

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l95_95092


namespace arithmetic_expression_evaluation_l95_95974

theorem arithmetic_expression_evaluation : (8 / 2 - 3 * 2 + 5^2 / 5) = 3 := by
  sorry

end arithmetic_expression_evaluation_l95_95974


namespace n_mod_9_eq_6_l95_95371

def n : ℕ := 2 + 333 + 44444 + 555555 + 6666666 + 77777777 + 888888888 + 9999999999

theorem n_mod_9_eq_6 : n % 9 = 6 :=
by
  sorry

end n_mod_9_eq_6_l95_95371


namespace area_triangle_FQH_l95_95656

open Set

structure Point where
  x : ℝ
  y : ℝ

def Rectangle (A B C D : Point) : Prop :=
  A.x = B.x ∧ C.x = D.x ∧ A.y = D.y ∧ B.y = C.y

def IsMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def AreaTrapezoid (A B C D : Point) : ℝ :=
  0.5 * (B.x - A.x + D.x - C.x) * (A.y - C.y)

def AreaTriangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

variables (E P R H F Q G : Point)

-- Conditions
axiom h1 : Rectangle E F G H
axiom h2 : E.y - P.y = 8
axiom h3 : R.y - H.y = 8
axiom h4 : F.x - E.x = 16
axiom h5 : AreaTrapezoid P R H G = 160

-- Target to prove
theorem area_triangle_FQH : AreaTriangle F Q H = 80 :=
sorry

end area_triangle_FQH_l95_95656


namespace jar_ratios_l95_95303

theorem jar_ratios (C_X C_Y : ℝ) 
  (h1 : 0 < C_X) 
  (h2 : 0 < C_Y)
  (h3 : (1/2) * C_X + (1/2) * C_Y = (3/4) * C_X) : 
  C_Y = (1/2) * C_X := 
sorry

end jar_ratios_l95_95303


namespace slope_of_line_l95_95387

theorem slope_of_line (P Q : ℝ × ℝ) (hP : P = (1, 2)) (hQ : Q = (4, 3)) :
  (Q.snd - P.snd) / (Q.fst - P.fst) = 1 / 3 := by
  sorry

end slope_of_line_l95_95387


namespace proof_l95_95683

-- Definition of the logical statements
def all_essays_correct (maria : Type) : Prop := sorry
def passed_course (maria : Type) : Prop := sorry

-- Condition provided in the problem
axiom condition : ∀ (maria : Type), all_essays_correct maria → passed_course maria

-- We need to prove this
theorem proof (maria : Type) : ¬ (passed_course maria) → ¬ (all_essays_correct maria) :=
by sorry

end proof_l95_95683


namespace bert_kangaroos_equal_to_kameron_in_40_days_l95_95428

theorem bert_kangaroos_equal_to_kameron_in_40_days
  (k_count : ℕ) (b_count : ℕ) (rate : ℕ) (days : ℕ)
  (h1 : k_count = 100)
  (h2 : b_count = 20)
  (h3 : rate = 2)
  (h4 : days = 40) :
  b_count + days * rate = k_count := 
by
  sorry

end bert_kangaroos_equal_to_kameron_in_40_days_l95_95428


namespace select_team_with_at_least_girls_l95_95556

-- Definitions based on the conditions
def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls
def team_size : ℕ := 7
def at_least_girls : ℕ := 3

-- The main theorem statement
theorem select_team_with_at_least_girls :
  (finset.card (finset.powerset_len 3 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 3) (finset.range boys)) + 
  (finset.card (finset.powerset_len 4 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 4) (finset.range boys)) + 
  (finset.card (finset.powerset_len 5 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 5) (finset.range boys)) + 
  (finset.card (finset.powerset_len 6 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 6) (finset.range boys)) + 
  (finset.card (finset.powerset_len 7 (finset.range girls))).card * finset.card (finset.powerset_len (team_size - 7) (finset.range boys)) =
  10620 :=
sorry

end select_team_with_at_least_girls_l95_95556


namespace range_of_m_l95_95312

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def B (m : ℝ) := { x : ℝ | x^2 - (2 * m + 1) * x + 2 * m < 0 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → (-1 / 2 ≤ m ∧ m ≤ 1) :=
by
  sorry

end range_of_m_l95_95312


namespace expression_evaluation_l95_95581

theorem expression_evaluation : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end expression_evaluation_l95_95581


namespace cos_150_eq_negative_cos_30_l95_95789

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95789


namespace gcd_204_85_l95_95035

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l95_95035


namespace family_reunion_people_l95_95613

theorem family_reunion_people (pasta_per_person : ℚ) (total_pasta : ℚ) (recipe_people : ℚ) : 
  pasta_per_person = 2 / 7 ∧ total_pasta = 10 -> recipe_people = 35 :=
by
  sorry

end family_reunion_people_l95_95613


namespace contrapositive_inequality_l95_95949

theorem contrapositive_inequality (a b : ℝ) :
  (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) := by
sorry

end contrapositive_inequality_l95_95949


namespace sugar_needed_in_two_minutes_l95_95189

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end sugar_needed_in_two_minutes_l95_95189


namespace tall_mirror_passes_l95_95867

theorem tall_mirror_passes (T : ℕ)
    (s_tall_ref : ℕ)
    (s_wide_ref : ℕ)
    (e_tall_ref : ℕ)
    (e_wide_ref : ℕ)
    (wide_passes : ℕ)
    (total_reflections : ℕ)
    (H1 : s_tall_ref = 10)
    (H2 : s_wide_ref = 5)
    (H3 : e_tall_ref = 6)
    (H4 : e_wide_ref = 3)
    (H5 : wide_passes = 5)
    (H6 : s_tall_ref * T + s_wide_ref * wide_passes + e_tall_ref * T + e_wide_ref * wide_passes = 88) : 
    T = 3 := 
by sorry

end tall_mirror_passes_l95_95867


namespace four_digit_numbers_l95_95275

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l95_95275


namespace cos_150_eq_neg_sqrt3_div_2_l95_95799

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95799


namespace relationship_of_variables_l95_95409

theorem relationship_of_variables
  (a b c d : ℚ)
  (h : (a + b) / (b + c) = (c + d) / (d + a)) :
  a = c ∨ a + b + c + d = 0 :=
by sorry

end relationship_of_variables_l95_95409


namespace budget_circle_salaries_degrees_l95_95060

theorem budget_circle_salaries_degrees :
  let transportation := 20
  let research_development := 9
  let utilities := 5
  let equipment := 4
  let supplies := 2
  let total_percent := 100
  let full_circle_degrees := 360
  let total_allocated_percent := transportation + research_development + utilities + equipment + supplies
  let salaries_percent := total_percent - total_allocated_percent
  let salaries_degrees := (salaries_percent * full_circle_degrees) / total_percent
  salaries_degrees = 216 :=
by
  sorry

end budget_circle_salaries_degrees_l95_95060


namespace sarah_total_volume_in_two_weeks_l95_95692

def shampoo_daily : ℝ := 1

def conditioner_daily : ℝ := 1 / 2 * shampoo_daily

def days : ℕ := 14

def total_volume : ℝ := (shampoo_daily * days) + (conditioner_daily * days)

theorem sarah_total_volume_in_two_weeks : total_volume = 21 := by
  sorry

end sarah_total_volume_in_two_weeks_l95_95692


namespace four_digit_numbers_count_l95_95278

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l95_95278


namespace disloyal_bound_l95_95449

variable {p n : ℕ}

/-- A number is disloyal if its GCD with n is not 1 -/
def isDisloyal (x : ℕ) (n : ℕ) := Nat.gcd x n ≠ 1

theorem disloyal_bound (p : ℕ) (n : ℕ) (hp : p.Prime) (hn : n % p^2 = 0) :
  (∃ D : Finset ℕ, (∀ x ∈ D, isDisloyal x n) ∧ D.card ≤ (n - 1) / p) :=
sorry

end disloyal_bound_l95_95449


namespace abs_ineq_l95_95451

open Real

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

theorem abs_ineq (a b c : ℝ) (h1 : a + b ≥ 0) (h2 : b + c ≥ 0) (h3 : c + a ≥ 0) :
  a + b + c ≥ (absolute_value a + absolute_value b + absolute_value c) / 3 := by
  sorry

end abs_ineq_l95_95451


namespace napkin_coloring_l95_95167

structure Napkin where
  top : ℝ
  bottom : ℝ
  left : ℝ
  right : ℝ

def intersects_vertically (n1 n2 : Napkin) : Prop :=
  n1.left ≤ n2.right ∧ n2.left ≤ n1.right

def intersects_horizontally (n1 n2 : Napkin) : Prop :=
  n1.bottom ≤ n2.top ∧ n2.bottom ≤ n1.top

def can_be_crossed_by_line (n1 n2 : Napkin) : Prop :=
  intersects_vertically n1 n2 ∨ intersects_horizontally n1 n2

theorem napkin_coloring
  (blue_napkins green_napkins : List Napkin)
  (h_cross : ∀ (b : Napkin) (g : Napkin), 
    b ∈ blue_napkins → g ∈ green_napkins → can_be_crossed_by_line b g) :
  ∃ (color : String) (h1 h2 : ℝ) (v : ℝ), 
    (color = "blue" ∧ ∀ b ∈ blue_napkins, (b.bottom ≤ h1 ∧ h1 ≤ b.top) ∨ (b.bottom ≤ h2 ∧ h2 ≤ b.top) ∨ (b.left ≤ v ∧ v ≤ b.right)) ∨
    (color = "green" ∧ ∀ g ∈ green_napkins, (g.bottom ≤ h1 ∧ h1 ≤ g.top) ∨ (g.bottom ≤ h2 ∧ h2 ≤ g.top) ∨ (g.left ≤ v ∧ v ≤ g.right)) :=
sorry

end napkin_coloring_l95_95167


namespace cos_150_eq_neg_sqrt3_div_2_l95_95840

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95840


namespace p_sufficient_but_not_necessary_for_q_l95_95573

def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x = 1 ∨ x = -2

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) : (p x → q x) ∧ ¬(q x → p x) := 
by {
  sorry
}

end p_sufficient_but_not_necessary_for_q_l95_95573


namespace f_of_3_l95_95118

theorem f_of_3 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := 
sorry

end f_of_3_l95_95118


namespace four_digit_numbers_l95_95273

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l95_95273


namespace smallest_consecutive_integer_sum_l95_95181

-- Definitions based on conditions
def consecutive_integer_sum (n : ℕ) := 20 * n + 190

-- Theorem statement
theorem smallest_consecutive_integer_sum : 
  ∃ (n k : ℕ), (consecutive_integer_sum n = k^3) ∧ (∀ m l : ℕ, (consecutive_integer_sum m = l^3) → k^3 ≤ l^3) :=
sorry

end smallest_consecutive_integer_sum_l95_95181


namespace cos_150_eq_neg_half_l95_95812

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95812


namespace general_term_of_sequence_l95_95253

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 3 * n^2 - n + 1) :
  (∀ n, a n = if n = 1 then 3 else 6 * n - 4) :=
by
  sorry

end general_term_of_sequence_l95_95253


namespace cube_root_of_5_irrational_l95_95587

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end cube_root_of_5_irrational_l95_95587


namespace arithmetic_sequence_fifth_term_l95_95709

theorem arithmetic_sequence_fifth_term (x y : ℝ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x^2 + y^2
    let a2 := x^2 - y^2
    let a3 := x^2 * y^2
    let a4 := x^2 / y^2
    let d := a2 - a1
    let a5 := a4 + d
    a5 = 2 := by
  sorry

end arithmetic_sequence_fifth_term_l95_95709


namespace polynomial_smallest_e_l95_95714

theorem polynomial_smallest_e :
  ∃ (a b c d e : ℤ), (a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ∧ a ≠ 0 ∧ e > 0 ∧ (x + 3) * (x - 6) * (x - 10) * (2 * x + 1) = 0) 
  ∧ e = 180 :=
by
  sorry

end polynomial_smallest_e_l95_95714


namespace river_width_l95_95204

theorem river_width (boat_max_speed : ℝ) (river_current_speed : ℝ) (time_to_cross : ℝ) (width : ℝ) :
  boat_max_speed = 4 ∧ river_current_speed = 3 ∧ time_to_cross = 2 ∧ width = 8 → 
  width = boat_max_speed * time_to_cross := by
  intros h
  cases h
  sorry

end river_width_l95_95204


namespace natasha_average_speed_l95_95439

theorem natasha_average_speed :
  (4 * 2.625 * 2) / (4 + 2) = 3.5 := 
by
  sorry

end natasha_average_speed_l95_95439


namespace power_24_eq_one_l95_95284

theorem power_24_eq_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 :=
by
  sorry

end power_24_eq_one_l95_95284


namespace find_remainder_l95_95517

theorem find_remainder (n : ℕ) 
  (h1 : n^2 % 7 = 3)
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := 
by sorry

end find_remainder_l95_95517


namespace probability_one_red_ball_distribution_of_X_l95_95476

-- Definitions of probabilities
def C (n k : ℕ) : ℕ := Nat.choose n k

def P_one_red_ball : ℚ := (C 2 1 * C 3 2 : ℚ) / C 5 3

#check (1 : ℚ)
#check (3 : ℚ)
#check (5 : ℚ)
def X_distribution (i : ℕ) : ℚ :=
  if i = 0 then (C 3 3 : ℚ) / C 5 3
  else if i = 1 then (C 2 1 * C 3 2 : ℚ) / C 5 3
  else if i = 2 then (C 2 2 * C 3 1 : ℚ) / C 5 3
  else 0

-- Statement to prove
theorem probability_one_red_ball : 
  P_one_red_ball = 3 / 5 := 
sorry

theorem distribution_of_X :
  Π i, (i = 0 → X_distribution i = 1 / 10) ∧
       (i = 1 → X_distribution i = 3 / 5) ∧
       (i = 2 → X_distribution i = 3 / 10) :=
sorry

end probability_one_red_ball_distribution_of_X_l95_95476


namespace laundry_loads_l95_95575

-- Conditions
def wash_time_per_load : ℕ := 45 -- in minutes
def dry_time_per_load : ℕ := 60 -- in minutes
def total_time : ℕ := 14 -- in hours

theorem laundry_loads (L : ℕ) 
  (h1 : total_time = 14)
  (h2 : total_time * 60 = L * (wash_time_per_load + dry_time_per_load)) :
  L = 8 :=
by
  sorry

end laundry_loads_l95_95575


namespace length_of_ln_l95_95703

theorem length_of_ln (sin_N_eq : Real.sin angle_N = 3 / 5) (LM_eq : length_LM = 15) :
  length_LN = 25 :=
sorry

end length_of_ln_l95_95703


namespace smallest_value_of_a_minus_b_l95_95064

theorem smallest_value_of_a_minus_b (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_value_of_a_minus_b_l95_95064


namespace solve_equation_l95_95907

theorem solve_equation (a : ℝ) (x : ℝ) : (2 * a * x + 3) / (a - x) = 3 / 4 → x = 1 → a = -3 :=
by
  intros h h1
  rw [h1] at h
  sorry

end solve_equation_l95_95907


namespace cube_root_of_5_irrational_l95_95585

theorem cube_root_of_5_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ)^3 = 5 * (q : ℚ)^3 := by
  sorry

end cube_root_of_5_irrational_l95_95585


namespace cos_150_eq_neg_half_l95_95843

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95843


namespace factor_expression_l95_95504

theorem factor_expression (x y : ℝ) : 5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) :=
by
  sorry

end factor_expression_l95_95504


namespace tan_product_pi_over_6_3_2_undefined_l95_95612

noncomputable def tan_pi_over_6 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_pi_over_3 : ℝ := Real.tan (Real.pi / 3)
noncomputable def tan_pi_over_2 : ℝ := Real.tan (Real.pi / 2)

theorem tan_product_pi_over_6_3_2_undefined :
  ∃ (x y : ℝ), Real.tan (Real.pi / 6) = x ∧ Real.tan (Real.pi / 3) = y ∧ Real.tan (Real.pi / 2) = 0 :=
by
  sorry

end tan_product_pi_over_6_3_2_undefined_l95_95612


namespace total_apples_picked_l95_95437

theorem total_apples_picked (Mike_apples Nancy_apples Keith_apples : ℕ)
  (hMike : Mike_apples = 7)
  (hNancy : Nancy_apples = 3)
  (hKeith : Keith_apples = 6) :
  Mike_apples + Nancy_apples + Keith_apples = 16 :=
by
  sorry

end total_apples_picked_l95_95437


namespace ensure_A_win_product_l95_95999

theorem ensure_A_win_product {s : Finset ℕ} (h1 : s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : 8 ∈ s) (h3 : 5 ∈ s) :
  (4 ∈ s ∧ 6 ∈ s ∧ 7 ∈ s) →
  4 * 6 * 7 = 168 := 
by 
  intro _ 
  exact Nat.mul_assoc 4 6 7

end ensure_A_win_product_l95_95999


namespace mr_smiths_sixth_child_not_represented_l95_95158

def car_plate_number := { n : ℕ // ∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 }
def mr_smith_is_45 (n : ℕ) := (n % 100) = 45
def divisible_by_children_ages (n : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → n % i = 0

theorem mr_smiths_sixth_child_not_represented :
    ∃ n : car_plate_number, mr_smith_is_45 n.val ∧ divisible_by_children_ages n.val → ¬ (6 ∣ n.val) :=
by
  sorry

end mr_smiths_sixth_child_not_represented_l95_95158


namespace sugar_needed_in_two_minutes_l95_95188

-- Let a be the amount of sugar needed per chocolate bar.
def sugar_per_chocolate_bar : ℝ := 1.5

-- Let b be the number of chocolate bars produced per minute.
def chocolate_bars_per_minute : ℕ := 36

-- Let t be the time in minutes.
def time_in_minutes : ℕ := 2

theorem sugar_needed_in_two_minutes : 
  let sugar_in_one_minute := chocolate_bars_per_minute * sugar_per_chocolate_bar
  let total_sugar := sugar_in_one_minute * time_in_minutes
  total_sugar = 108 := by
  sorry

end sugar_needed_in_two_minutes_l95_95188


namespace fraction_product_cube_l95_95049

theorem fraction_product_cube :
  ((5 : ℚ) / 8)^3 * ((4 : ℚ) / 9)^3 = (125 : ℚ) / 5832 :=
by
  sorry

end fraction_product_cube_l95_95049


namespace sally_book_pages_l95_95560

/-- 
  Sally reads 10 pages on weekdays and 20 pages on weekends. 
  It takes 2 weeks for Sally to finish her book. 
  We want to prove that the book has 180 pages.
-/
theorem sally_book_pages
  (weekday_pages : ℕ)
  (weekend_pages : ℕ)
  (num_weeks : ℕ)
  (total_pages : ℕ) :
  weekday_pages = 10 → 
  weekend_pages = 20 → 
  num_weeks = 2 → 
  total_pages = (5 * weekday_pages + 2 * weekend_pages) * num_weeks → 
  total_pages = 180 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw h4
  norm_num
  sorry

end sally_book_pages_l95_95560


namespace probability_businessmen_wait_two_minutes_l95_95988

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l95_95988


namespace quadratic_complex_roots_condition_l95_95474

theorem quadratic_complex_roots_condition (a : ℝ) :
  (∀ a, -2 ≤ a ∧ a ≤ 2 → (a^2 < 4)) ∧ 
  ¬(∀ a, (a^2 < 4) → -2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end quadratic_complex_roots_condition_l95_95474


namespace cos_150_deg_l95_95826

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95826


namespace find_f_2015_l95_95258

noncomputable def f : ℝ → ℝ :=
sorry

lemma f_period : ∀ x : ℝ, f (x + 8) = f x :=
sorry

axiom f_func_eq : ∀ x : ℝ, f (x + 2) = (1 + f x) / (1 - f x)

axiom f_initial : f 1 = 1 / 4

theorem find_f_2015 : f 2015 = -3 / 5 :=
sorry

end find_f_2015_l95_95258


namespace four_digit_numbers_count_l95_95280

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l95_95280


namespace sqrt_sum_bounds_l95_95151

theorem sqrt_sum_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
    4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2 - b)^2) + 
                   Real.sqrt (b^2 + (2 - c)^2) + 
                   Real.sqrt (c^2 + (2 - d)^2) + 
                   Real.sqrt (d^2 + (2 - a)^2) ∧
    Real.sqrt (a^2 + (2 - b)^2) + 
    Real.sqrt (b^2 + (2 - c)^2) + 
    Real.sqrt (c^2 + (2 - d)^2) + 
    Real.sqrt (d^2 + (2 - a)^2) ≤ 8 :=
sorry

end sqrt_sum_bounds_l95_95151


namespace super_domino_double_probability_l95_95079

theorem super_domino_double_probability :
  ∃ (dominos : set (ℕ × ℕ)),
    (∀ a b : ℕ, a ∈ finset.range 15 ∧ b ∈ finset.range 15 → (a, b) ∈ dominos) ∧
    (∀ d ∈ finset.range 15, (d, d) ∈ dominos) →
    (∃ n : ℚ, n = 15 / (Nat.choose 15 2 + 15) ∧ n = 1 / 8) :=
by
  -- Placeholder for the proof steps
  sorry

end super_domino_double_probability_l95_95079


namespace highway_extension_completion_l95_95478

def current_length := 200
def final_length := 650
def built_first_day := 50
def built_second_day := 3 * built_first_day

theorem highway_extension_completion :
  (final_length - current_length - built_first_day - built_second_day) = 250 := by
  sorry

end highway_extension_completion_l95_95478


namespace blueberries_in_blue_box_l95_95747

theorem blueberries_in_blue_box (B S : ℕ) (h1 : S - B = 12) (h2 : S + B = 76) : B = 32 :=
sorry

end blueberries_in_blue_box_l95_95747


namespace cost_of_fencing_l95_95982

open Real

theorem cost_of_fencing
  (ratio_length_width : ∃ x : ℝ, 3 * x * 2 * x = 3750)
  (cost_per_meter : ℝ := 0.50) :
  ∃ cost : ℝ, cost = 125 := by
  sorry

end cost_of_fencing_l95_95982


namespace cos_150_deg_l95_95821

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95821


namespace pond_ratios_l95_95967

theorem pond_ratios (T A : ℕ) (h1 : T = 48) (h2 : A = 32) : A / (T - A) = 2 :=
by
  sorry

end pond_ratios_l95_95967


namespace cos_150_eq_neg_sqrt3_over_2_l95_95818

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95818


namespace largest_n_factors_l95_95241

theorem largest_n_factors (n : ℤ) :
  (∃ A B : ℤ, 3 * B + A = n ∧ A * B = 72) → n ≤ 217 :=
by {
  sorry
}

end largest_n_factors_l95_95241


namespace simplify_fraction_l95_95698

theorem simplify_fraction (a b c : ℕ) (h1 : a = 222) (h2 : b = 8888) (h3 : c = 44) : 
  (a : ℚ) / b * c = 111 / 101 := 
by 
  sorry

end simplify_fraction_l95_95698


namespace liam_more_heads_than_mina_l95_95555

def toss_results (tosses : ℕ) : list (list bool) :=
  list.fin_cases (2 ^ tosses) (λ n, list.of_fn (λ i, test_bit n i < tosses))

def count_heads (results : list bool) : ℕ :=
  results.count id

def count_favorable (mina_tosses liam_tosses : list (list bool)) : ℕ :=
  (mina_tosses.product liam_tosses).count (λ p, count_heads p.2 = count_heads p.1 + 1)

def probability (mina_tosses liam_tosses : list (list bool)) : ℚ :=
  count_favorable mina_tosses liam_tosses / (mina_tosses.length * liam_tosses.length : ℚ)

theorem liam_more_heads_than_mina :
  probability (toss_results 2) (toss_results 3) = 5 / 32 :=
by
  sorry

end liam_more_heads_than_mina_l95_95555


namespace usual_time_56_l95_95200

theorem usual_time_56 (S : ℝ) (T : ℝ) (h : (T + 24) * S = T * (0.7 * S)) : T = 56 :=
by sorry

end usual_time_56_l95_95200


namespace next_bell_ringing_time_l95_95762

theorem next_bell_ringing_time (post_office_interval train_station_interval town_hall_interval start_time : ℕ)
  (h1 : post_office_interval = 18)
  (h2 : train_station_interval = 24)
  (h3 : town_hall_interval = 30)
  (h4 : start_time = 9) :
  let lcm := Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval)
  lcm + start_time = 15 := by
  sorry

end next_bell_ringing_time_l95_95762


namespace negation_equivalence_l95_95712

variable (x : ℝ)

def original_proposition := ∃ x : ℝ, x^2 - 3*x + 3 < 0

def negation_proposition := ∀ x : ℝ, x^2 - 3*x + 3 ≥ 0

theorem negation_equivalence : ¬ original_proposition ↔ negation_proposition :=
by 
  -- Lean doesn’t require the actual proof here
  sorry

end negation_equivalence_l95_95712


namespace part1_part2_part3_l95_95877

def is_beautiful_point (x y : ℝ) (a b : ℝ) : Prop :=
  a = -x ∧ b = x - y

def beautiful_points (x y : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let a := -x
  let b := x - y
  ((a, b), (b, a))

theorem part1 (x y : ℝ) (h : (x, y) = (4, 1)) :
  beautiful_points x y = ((-4, 3), (3, -4)) := by
  sorry

theorem part2 (x y : ℝ) (h : x = 2) (h' : (-x = 2 - y)) :
  y = 4 := by
  sorry

theorem part3 (x y : ℝ) (h : ((-x, x-y) = (-2, 7)) ∨ ((x-y, -x) = (-2, 7))) :
  (x = 2 ∧ y = -5) ∨ (x = -7 ∧ y = -5) := by
  sorry

end part1_part2_part3_l95_95877


namespace push_ups_total_l95_95497

theorem push_ups_total (d z : ℕ) (h1 : d = 51) (h2 : d = z + 49) : d + z = 53 := by
  sorry

end push_ups_total_l95_95497


namespace precise_approximate_classification_l95_95592

def data_points : List String := ["Xiao Ming bought 5 books today",
                                  "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                  "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                  "The human brain has 10,000,000,000 cells",
                                  "Xiao Hong scored 92 points on this test",
                                  "The Earth has more than 1.5 trillion tons of coal reserves"]

def is_precise (data : String) : Bool :=
  match data with
  | "Xiao Ming bought 5 books today" => true
  | "The war in Afghanistan cost the United States $1 billion per month in 2002" => true
  | "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion" => true
  | "Xiao Hong scored 92 points on this test" => true
  | _ => false

def is_approximate (data : String) : Bool :=
  match data with
  | "The human brain has 10,000,000,000 cells" => true
  | "The Earth has more than 1.5 trillion tons of coal reserves" => true
  | _ => false

theorem precise_approximate_classification :
  (data_points.filter is_precise = ["Xiao Ming bought 5 books today",
                                    "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                    "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                    "Xiao Hong scored 92 points on this test"]) ∧
  (data_points.filter is_approximate = ["The human brain has 10,000,000,000 cells",
                                        "The Earth has more than 1.5 trillion tons of coal reserves"]) :=
by sorry

end precise_approximate_classification_l95_95592


namespace no_solution_abs_eq_l95_95133

theorem no_solution_abs_eq : ∀ y : ℝ, |y - 2| ≠ |y - 1| + |y - 4| :=
by
  intros y
  sorry

end no_solution_abs_eq_l95_95133


namespace binom_9_5_l95_95093

theorem binom_9_5 : nat.binomial 9 5 = 126 := by
  sorry

end binom_9_5_l95_95093


namespace least_number_to_subtract_l95_95739

theorem least_number_to_subtract 
  (n : ℤ) 
  (h1 : 7 ∣ (90210 - n + 12)) 
  (h2 : 11 ∣ (90210 - n + 12)) 
  (h3 : 13 ∣ (90210 - n + 12)) 
  (h4 : 17 ∣ (90210 - n + 12)) 
  (h5 : 19 ∣ (90210 - n + 12)) : 
  n = 90198 :=
sorry

end least_number_to_subtract_l95_95739


namespace smallest_possible_degree_p_l95_95177

theorem smallest_possible_degree_p (p : Polynomial ℝ) :
  (∀ x, 0 < |x| → ∃ C, |((3 * x^7 + 2 * x^6 - 4 * x^3 + x - 5) / (p.eval x)) - C| < ε)
  → (Polynomial.degree p) ≥ 7 := by
  sorry

end smallest_possible_degree_p_l95_95177


namespace modified_monotonous_count_l95_95611

def is_modified_monotonous (n : ℕ) : Prop :=
  -- Definition that determines if a number is modified-monotonous
  -- Must include digit '5', and digits must form a strictly increasing or decreasing sequence
  sorry 

def count_modified_monotonous (n : ℕ) : ℕ :=
  2 * (8 * (2^8) + 2^8) + 1 -- Formula for counting modified-monotonous numbers including '5'

theorem modified_monotonous_count : count_modified_monotonous 5 = 4609 := 
  by 
    sorry

end modified_monotonous_count_l95_95611


namespace jihye_marbles_l95_95662

theorem jihye_marbles (Y : ℕ) (h1 : Y + (Y + 11) = 85) : Y + 11 = 48 := by
  sorry

end jihye_marbles_l95_95662


namespace bill_take_home_salary_l95_95491

-- Define the parameters
def property_taxes : ℝ := 2000
def sales_taxes : ℝ := 3000
def gross_salary : ℝ := 50000
def income_tax_rate : ℝ := 0.10

-- Define income tax calculation
def income_tax : ℝ := income_tax_rate * gross_salary

-- Define total taxes calculation
def total_taxes : ℝ := property_taxes + sales_taxes + income_tax

-- Define the take-home salary calculation
def take_home_salary : ℝ := gross_salary - total_taxes

-- Statement of the theorem
theorem bill_take_home_salary : take_home_salary = 40000 := by
  -- Sorry is used to skip the proof.
  sorry

end bill_take_home_salary_l95_95491


namespace pqrs_product_l95_95256

theorem pqrs_product :
  let P := (Real.sqrt 2010 + Real.sqrt 2009 + Real.sqrt 2008)
  let Q := (-Real.sqrt 2010 - Real.sqrt 2009 + Real.sqrt 2008)
  let R := (Real.sqrt 2010 - Real.sqrt 2009 - Real.sqrt 2008)
  let S := (-Real.sqrt 2010 + Real.sqrt 2009 - Real.sqrt 2008)
  P * Q * R * S = 1 := by
{
  sorry -- Proof is omitted as per the provided instructions.
}

end pqrs_product_l95_95256


namespace largest_divisor_360_450_l95_95337

theorem largest_divisor_360_450 : ∃ d, (d ∣ 360 ∧ d ∣ 450) ∧ (∀ e, (e ∣ 360 ∧ e ∣ 450) → e ≤ d) ∧ d = 90 :=
by
  sorry

end largest_divisor_360_450_l95_95337


namespace origin_not_in_A_point_M_in_A_l95_95180

def set_A : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ x + 2 * y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2 * x + y - 5 ≤ 0}

theorem origin_not_in_A : (0, 0) ∉ set_A := by
  sorry

theorem point_M_in_A : (1, 1) ∈ set_A := by
  sorry

end origin_not_in_A_point_M_in_A_l95_95180


namespace arithmetic_problem_l95_95526

noncomputable def arithmetic_progression (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d

noncomputable def sum_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_problem (a₁ d : ℝ)
  (h₁ : a₁ + (a₁ + 2 * d) = 5)
  (h₂ : 4 * (2 * a₁ + 3 * d) / 2 = 20) :
  (sum_terms a₁ d 8 - 2 * sum_terms a₁ d 4) / (sum_terms a₁ d 6 - sum_terms a₁ d 4 - sum_terms a₁ d 2) = 10 := by
  sorry

end arithmetic_problem_l95_95526


namespace find_x_l95_95539

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 6) : x = 14 :=
by
  sorry

end find_x_l95_95539


namespace binomial_square_constant_l95_95537

theorem binomial_square_constant :
  ∃ c : ℝ, (∀ x : ℝ, 9*x^2 - 21*x + c = (3*x + -3.5)^2) → c = 12.25 :=
by
  sorry

end binomial_square_constant_l95_95537


namespace max_xy_l95_95411

open Real

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : x + 4 * y = 4) :
  ∃ y : ℝ, (x = 4 - 4 * y) → y = 1 / 2 → x * y = 1 :=
by
  sorry

end max_xy_l95_95411


namespace max_blocks_fit_l95_95053

-- Defining the dimensions of the box and blocks
def box_length : ℝ := 4
def box_width : ℝ := 3
def box_height : ℝ := 2

def block_length : ℝ := 3
def block_width : ℝ := 1
def block_height : ℝ := 1

-- Theorem stating the maximum number of blocks that fit
theorem max_blocks_fit : (24 / 3 = 8) ∧ (1 * 3 * 2 = 6) → 6 = 6 := 
by
  sorry

end max_blocks_fit_l95_95053


namespace cos_150_eq_neg_half_l95_95811

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95811


namespace problem_proof_l95_95259

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- The conditions
def even_function_on_ℝ := ∀ x : ℝ, f x = f (-x)
def f_at_0_is_2 := f 0 = 2
def odd_after_translation := ∀ x : ℝ, f (x - 1) = -f (-x - 1)

-- Prove the required condition
theorem problem_proof (h1 : even_function_on_ℝ f) (h2 : f_at_0_is_2 f) (h3 : odd_after_translation f) :
    f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
by
  sorry

end problem_proof_l95_95259


namespace range_of_m_l95_95412

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (x^2 - 4*|x| + 5 - m = 0) → (∃ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)) → (1 < m ∧ m < 5) :=
by
  sorry

end range_of_m_l95_95412


namespace cos_150_deg_l95_95823

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95823


namespace largest_number_is_870_l95_95966

-- Define the set of digits {8, 7, 0}
def digits : Set ℕ := {8, 7, 0}

-- Define the largest number that can be made by arranging these digits
def largest_number (s : Set ℕ) : ℕ := 870

-- Statement to prove
theorem largest_number_is_870 : largest_number digits = 870 :=
by
  -- Proof is omitted
  sorry

end largest_number_is_870_l95_95966


namespace binom_9_5_equals_126_l95_95090

theorem binom_9_5_equals_126 : Nat.binom 9 5 = 126 := 
by 
  sorry

end binom_9_5_equals_126_l95_95090


namespace field_length_to_width_ratio_l95_95954

theorem field_length_to_width_ratio
(W L : ℕ) (P : ℕ) 
(hW : W = 80)
(hP : P = 384)
(hP_def : P = 2 * L + 2 * W) :
  (L : ℚ) / W = 7 / 5 :=
by {
  -- Definitions from problem conditions
  have hW_def : W = 80 := hW,
  have hP_def' : P = 384 := hP,
  have hP_eq : P = 2 * L + 2 * W := hP_def,
  
  -- Defining lengths and solving for ratio
  sorry
}

end field_length_to_width_ratio_l95_95954


namespace cos_150_degree_l95_95831

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95831


namespace largest_plot_area_l95_95359

def plotA_area : Real := 10
def plotB_area : Real := 10 + 1
def plotC_area : Real := 9 + 1.5
def plotD_area : Real := 12
def plotE_area : Real := 11 + 1

theorem largest_plot_area :
  max (max (max (max plotA_area plotB_area) plotC_area) plotD_area) plotE_area = 12 ∧ 
  (plotD_area = 12 ∧ plotE_area = 12) := by sorry

end largest_plot_area_l95_95359


namespace range_of_a_l95_95403

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x) → (-3^x ≤ a)) ↔ (a ≥ -1) :=
by
  sorry

end range_of_a_l95_95403


namespace marys_number_l95_95425

theorem marys_number (j m : ℕ) (h₁ : j * m = 2002)
  (h₂ : ∃ k, k * m = 2002 ∧ k ≠ j)
  (h₃ : ∃ l, j * l = 2002 ∧ l ≠ m) :
  m = 1001 :=
sorry

end marys_number_l95_95425


namespace sliced_meat_cost_per_type_with_rush_shipping_l95_95881

theorem sliced_meat_cost_per_type_with_rush_shipping:
  let original_cost := 40.0
  let rush_delivery_percentage := 0.3
  let num_types := 4
  let rush_delivery_cost := rush_delivery_percentage * original_cost
  let total_cost := original_cost + rush_delivery_cost
  let cost_per_type := total_cost / num_types
  cost_per_type = 13.0 :=
by
  sorry

end sliced_meat_cost_per_type_with_rush_shipping_l95_95881


namespace replace_digits_correct_l95_95938

def digits_eq (a b c d e : ℕ) : Prop :=
  5 * 10 + a + (b * 100) + (c * 10) + 3 = (d * 1000) + (e * 100) + 1

theorem replace_digits_correct :
  ∃ (a b c d e : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
    digits_eq a b c d e ∧ a = 1 ∧ b = 1 ∧ c = 4 ∧ d = 1 ∧ e = 4 :=
by
  sorry

end replace_digits_correct_l95_95938


namespace probability_exactly_one_red_ball_l95_95486

-- Define the given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 3
def children : ℕ := 10

-- Define the question and calculate the probability
theorem probability_exactly_one_red_ball : 
  (3 * (3 / 10) * ((7 / 10) * (7 / 10))) = 0.441 := 
by 
  sorry

end probability_exactly_one_red_ball_l95_95486


namespace cos_150_eq_neg_sqrt3_div_2_l95_95772

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95772


namespace hazel_walked_distance_l95_95895

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end hazel_walked_distance_l95_95895


namespace find_repeating_digits_l95_95215

-- Specify given conditions
def incorrect_result (a : ℚ) (b : ℚ) : ℚ := 54 * b - 1.8
noncomputable def correct_multiplication_value (d: ℚ) := 2 + d
noncomputable def repeating_decimal_value : ℚ := 2 + 35 / 99

-- Define what needs to be proved
theorem find_repeating_digits : ∃ (x : ℕ), x * 100 = 35 := by
  sorry

end find_repeating_digits_l95_95215


namespace minimum_a_plus_2c_l95_95421

theorem minimum_a_plus_2c (a c : ℝ) (h : (1 / a) + (1 / c) = 1) : a + 2 * c ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end minimum_a_plus_2c_l95_95421


namespace percentage_reduction_in_price_l95_95210

variable (R P : ℝ) (R_eq : R = 30) (H : 600 / R - 600 / P = 4)

theorem percentage_reduction_in_price (R_eq : R = 30) (H : 600 / R - 600 / P = 4) :
  ((P - R) / P) * 100 = 20 := sorry

end percentage_reduction_in_price_l95_95210


namespace correct_operation_l95_95344

variable (a : ℝ)

theorem correct_operation :
  (2 * a^2 * a = 2 * a^3) ∧
  ((a + 1)^2 ≠ a^2 + 1) ∧
  ((a^2 / (2 * a)) ≠ 2 * a) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) :=
by
  { sorry }

end correct_operation_l95_95344


namespace trig_inequality_sin_cos_l95_95690

theorem trig_inequality_sin_cos :
  Real.sin 2 + Real.cos 2 + 2 * (Real.sin 1 - Real.cos 1) ≥ 1 :=
by
  sorry

end trig_inequality_sin_cos_l95_95690


namespace distance_between_stations_l95_95464

-- distance calculation definitions
def distance (rate time : ℝ) := rate * time

-- problem conditions as definitions
def rate_slow := 20 -- km/hr
def rate_fast := 25 -- km/hr
def extra_distance := 50 -- km

-- final statement
theorem distance_between_stations :
  ∃ (D : ℝ) (T : ℝ),
    (distance rate_slow T = D) ∧
    (distance rate_fast T = D + extra_distance) ∧
    (D + (D + extra_distance) = 450) :=
by
  sorry

end distance_between_stations_l95_95464


namespace employed_males_percentage_l95_95298

variables {p : ℕ} -- total population
variables {employed_p : ℕ} {employed_females_p : ℕ}

-- 60 percent of the population is employed
def employed_population (p : ℕ) : ℕ := 60 * p / 100

-- 20 percent of the employed people are females
def employed_females (employed : ℕ) : ℕ := 20 * employed / 100

-- The question we're solving:
theorem employed_males_percentage (h1 : employed_p = employed_population p)
  (h2 : employed_females_p = employed_females employed_p)
  : (employed_p - employed_females_p) * 100 / p = 48 :=
by
  sorry

end employed_males_percentage_l95_95298


namespace remaining_miles_to_be_built_l95_95480

-- Definitions from problem conditions
def current_length : ℕ := 200
def target_length : ℕ := 650
def first_day_miles : ℕ := 50
def second_day_miles : ℕ := 3 * first_day_miles

-- Lean theorem statement
theorem remaining_miles_to_be_built : 
  (target_length - current_length) - (first_day_miles + second_day_miles) = 250 := 
by 
  sorry

end remaining_miles_to_be_built_l95_95480


namespace kelly_games_l95_95547

theorem kelly_games (initial_games give_away in_stock : ℕ) (h1 : initial_games = 50) (h2 : in_stock = 35) :
  give_away = initial_games - in_stock :=
by {
  -- initial_games = 50
  -- in_stock = 35
  -- Therefore, give_away = initial_games - in_stock
  sorry
}

end kelly_games_l95_95547


namespace jill_commute_time_l95_95862

theorem jill_commute_time :
  let dave_steps_per_min := 80
  let dave_cm_per_step := 70
  let dave_time_min := 20
  let dave_speed :=
    dave_steps_per_min * dave_cm_per_step
  let dave_distance :=
    dave_speed * dave_time_min
  let jill_steps_per_min := 120
  let jill_cm_per_step := 50
  let jill_speed :=
    jill_steps_per_min * jill_cm_per_step
  let jill_time :=
    dave_distance / jill_speed
  jill_time = 18 + 2 / 3 := by
  sorry

end jill_commute_time_l95_95862


namespace number_of_students_in_all_events_l95_95144

variable (T A B : ℕ)

-- Defining given conditions
-- Total number of students in the class
def total_students : ℕ := 45
-- Number of students participating in the Soccer event
def soccer_students : ℕ := 39
-- Number of students participating in the Basketball event
def basketball_students : ℕ := 28

-- Main theorem to prove
theorem number_of_students_in_all_events
  (h_total : T = total_students)
  (h_soccer : A = soccer_students)
  (h_basketball : B = basketball_students) :
  ∃ x : ℕ, x = A + B - T := sorry

end number_of_students_in_all_events_l95_95144


namespace min_crossing_time_proof_l95_95161

def min_crossing_time (times : List ℕ) : ℕ :=
  -- Function to compute the minimum crossing time. Note: Actual implementation skipped.
sorry

theorem min_crossing_time_proof
  (times : List ℕ)
  (h_times : times = [2, 4, 8, 16]) :
  min_crossing_time times = 30 :=
sorry

end min_crossing_time_proof_l95_95161


namespace cos_150_eq_neg_sqrt3_div_2_l95_95848

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95848


namespace find_alpha_l95_95260

theorem find_alpha
  (α : Real)
  (h1 : α > 0)
  (h2 : α < π)
  (h3 : 1 / Real.sin α + 1 / Real.cos α = 2) :
  α = π + 1 / 2 * Real.arcsin ((1 - Real.sqrt 5) / 2) :=
sorry

end find_alpha_l95_95260


namespace range_of_m_l95_95397

theorem range_of_m (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ 7 < m ∧ m < 24 :=
sorry

end range_of_m_l95_95397


namespace binom_9_5_l95_95094

theorem binom_9_5 : nat.binomial 9 5 = 126 := by
  sorry

end binom_9_5_l95_95094


namespace sum_of_distances_l95_95205

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_1 = 1 / 9 * d_2) (h2 : d_1 + d_2 = 6) : d_1 + d_2 + 6 = 20 :=
by
  sorry

end sum_of_distances_l95_95205


namespace josanna_next_test_score_l95_95663

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

end josanna_next_test_score_l95_95663


namespace cos_150_eq_neg_sqrt3_div_2_l95_95853

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95853


namespace intersection_minimum_distance_l95_95638

theorem intersection_minimum_distance (α t1 t2 : ℝ) (hα : 0 < α ∧ α < π)
  (l_eqns : ∀ t : ℝ, 1 + t * Real.cos α = 4 / (Real.sin² α) ∧ t * Real.sin α * t * Real.sin α = 4 * (1 + t * Real.cos α)) :
  (|t1 - t2| = 4 ↔ α = π / 2) :=
begin
  sorry
end

end intersection_minimum_distance_l95_95638


namespace cos_150_eq_neg_sqrt3_div_2_l95_95778

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95778


namespace john_weekly_allowance_l95_95534

noncomputable def weekly_allowance (A : ℝ) :=
  (3/5) * A + (1/3) * ((2/5) * A) + 0.60 <= A

theorem john_weekly_allowance : ∃ A : ℝ, (3/5) * A + (1/3) * ((2/5) * A) + 0.60 = A := by
  let A := 2.25
  sorry

end john_weekly_allowance_l95_95534


namespace find_number_of_women_in_first_group_l95_95596

variables (W : ℕ)

-- Conditions
def women_coloring_rate := 10
def total_cloth_colored_in_3_days := 180
def women_in_first_group := total_cloth_colored_in_3_days / 3

theorem find_number_of_women_in_first_group
  (h1 : 5 * women_coloring_rate * 4 = 200)
  (h2 : W * women_coloring_rate = women_in_first_group) :
  W = 6 :=
by
  sorry

end find_number_of_women_in_first_group_l95_95596


namespace cost_price_of_apple_l95_95076

theorem cost_price_of_apple (C : ℚ) (h1 : 19 = 5/6 * C) : C = 22.8 := by
  sorry

end cost_price_of_apple_l95_95076


namespace odd_function_condition_l95_95906

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x) / ((x - a) * (x + 1))

theorem odd_function_condition (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 := 
sorry

end odd_function_condition_l95_95906


namespace math_problem_l95_95671

variables {a b c d e : ℤ}

theorem math_problem 
(h1 : a - b + c - e = 7)
(h2 : b - c + d + e = 9)
(h3 : c - d + a - e = 5)
(h4 : d - a + b + e = 1)
: a + b + c + d + e = 11 := 
by 
  sorry

end math_problem_l95_95671


namespace women_lawyers_percentage_l95_95593

-- Define the conditions of the problem
variable {T : ℝ} (h1 : 0.80 * T = 0.80 * T)                          -- Placeholder for group size, not necessarily used directly
variable (h2 : 0.32 = 0.80 * L)                                       -- Given condition of the problem: probability of selecting a woman lawyer

-- Define the theorem to be proven
theorem women_lawyers_percentage (h2 : 0.32 = 0.80 * L) : L = 0.4 :=
by
  sorry

end women_lawyers_percentage_l95_95593


namespace ac_eq_af_l95_95669

theorem ac_eq_af
  (Γ₁ Γ₂ : Circle)
  (A D : Point)
  (h₁ : A ∈ Γ₁)
  (h₂ : D ∈ Γ₁)
  (h₃ : A ∈ Γ₂)
  (h₄ : D ∈ Γ₂)
  (B : Point)
  (h₅ : is_tangent Γ₁ (Line.through A B) A)
  (h₆ : B ∈ Γ₂)
  (C : Point)
  (h₇ : is_tangent Γ₂ (Line.through A C) A)
  (h₈ : C ∈ Γ₁)
  (E : Point)
  (h₉ : E ∈ Line.ray A B)
  (h₁₀ : distance B E = distance A B)
  (Ω : Circle)
  (h₁₁ : A ∈ Ω)
  (h₁₂ : D ∈ Ω)
  (h₁₃ : E ∈ Ω)
  (h₁₄ : F : Point)
  (h₁₅ : F ∈ Line.segment A C)
  (h₁₆ : second_inter (Line.segment A C) Ω F) :
  distance A C = distance A F :=
begin
  sorry
end

end ac_eq_af_l95_95669


namespace money_left_after_shopping_l95_95576

def initial_budget : ℝ := 999.00
def shoes_price : ℝ := 165.00
def yoga_mat_price : ℝ := 85.00
def sports_watch_price : ℝ := 215.00
def hand_weights_price : ℝ := 60.00
def sales_tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.10

def total_cost_before_discount : ℝ :=
  shoes_price + yoga_mat_price + sports_watch_price + hand_weights_price

def discount_on_watch : ℝ := sports_watch_price * discount_rate

def discounted_watch_price : ℝ := sports_watch_price - discount_on_watch

def total_cost_after_discount : ℝ :=
  shoes_price + yoga_mat_price + discounted_watch_price + hand_weights_price

def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

def total_cost_including_tax : ℝ := total_cost_after_discount + sales_tax

def money_left : ℝ := initial_budget - total_cost_including_tax

theorem money_left_after_shopping : 
  money_left = 460.25 :=
by
  sorry

end money_left_after_shopping_l95_95576


namespace cos_150_eq_neg_sqrt3_div_2_l95_95783

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95783


namespace range_of_x_l95_95004

theorem range_of_x (x : ℝ) (h1 : 1 / x < 4) (h2 : 1 / x > -2) : x < -1/2 ∨ x > 1/4 :=
by
  sorry

end range_of_x_l95_95004


namespace meaningful_fraction_range_l95_95723

theorem meaningful_fraction_range (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) := sorry

end meaningful_fraction_range_l95_95723


namespace mary_baseball_cards_l95_95157

theorem mary_baseball_cards :
  let initial_cards := 18
  let torn_cards := 8
  let fred_gifted_cards := 26
  let bought_cards := 40
  let exchanged_cards := 10
  let lost_cards := 5
  
  let remaining_cards := initial_cards - torn_cards
  let after_gift := remaining_cards + fred_gifted_cards
  let after_buy := after_gift + bought_cards
  let after_exchange := after_buy - exchanged_cards + exchanged_cards
  let final_count := after_exchange - lost_cards
  
  final_count = 71 :=
by
  sorry

end mary_baseball_cards_l95_95157


namespace count_numbers_with_digit_3_l95_95134

def contains_digit_3 (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.contains 3

theorem count_numbers_with_digit_3 :
  let nums := List.range' 300 300 in
  (nums.filter contains_digit_3).length = 138 :=
by
  sorry

end count_numbers_with_digit_3_l95_95134


namespace set_M_roster_method_l95_95006

open Set

theorem set_M_roster_method :
  {a : ℤ | ∃ (n : ℕ), 6 = n * (5 - a)} = {-1, 2, 3, 4} := by
  sorry

end set_M_roster_method_l95_95006


namespace outfit_choices_l95_95897

-- Define the numbers of shirts, pants, and hats.
def num_shirts : ℕ := 6
def num_pants : ℕ := 7
def num_hats : ℕ := 6

-- Define the number of colors and the constraints.
def num_colors : ℕ := 6

-- The total number of outfits without restrictions.
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- Number of outfits where all items are the same color.
def same_color_outfits : ℕ := num_colors

-- Number of outfits where the shirt and pants are the same color.
def same_shirt_pants_color_outfits : ℕ := num_colors + 1  -- accounting for the extra pair of pants

-- The total number of valid outfits calculated.
def valid_outfits : ℕ :=
  total_outfits - same_color_outfits - same_shirt_pants_color_outfits

-- The theorem statement asserting the correct answer.
theorem outfit_choices : valid_outfits = 239 := by
  sorry

end outfit_choices_l95_95897


namespace hazel_walked_distance_l95_95896

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end hazel_walked_distance_l95_95896


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95859

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95859


namespace certain_number_l95_95285

theorem certain_number (x y : ℕ) (h₁ : x = 14) (h₂ : 2^x - 2^(x - 2) = 3 * 2^y) : y = 12 :=
  by
  sorry

end certain_number_l95_95285


namespace find_number_of_adults_l95_95368

variable (A : ℕ) -- Variable representing the number of adults.
def C : ℕ := 5  -- Number of children.

def meal_cost : ℕ := 3  -- Cost per meal in dollars.
def total_cost (A : ℕ) : ℕ := (A + C) * meal_cost  -- Total cost formula.

theorem find_number_of_adults 
  (h1 : meal_cost = 3)
  (h2 : total_cost A = 21)
  (h3 : C = 5) :
  A = 2 :=
sorry

end find_number_of_adults_l95_95368


namespace quadratic_eq_real_roots_l95_95141

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end quadratic_eq_real_roots_l95_95141


namespace range_of_a_l95_95888

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 = 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + a > 0

theorem range_of_a (a : ℝ) (hp : proposition_p a) (hq : proposition_q a) : a ≥ 2 :=
sorry

end range_of_a_l95_95888


namespace iPhone_savings_l95_95314

theorem iPhone_savings
  (costX costY : ℕ)
  (discount_same_model discount_mixed : ℝ)
  (h1 : costX = 600)
  (h2 : costY = 800)
  (h3 : discount_same_model = 0.05)
  (h4 : discount_mixed = 0.03) :
  (costX + costX + costY) - ((costX * (1 - discount_same_model)) * 2 + costY * (1 - discount_mixed)) = 84 :=
by
  sorry

end iPhone_savings_l95_95314


namespace lines_intersect_at_same_point_l95_95178

theorem lines_intersect_at_same_point : 
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = -3 * x + 4 ∧ y = 4 * x + m) → m = -3 :=
by
  sorry

end lines_intersect_at_same_point_l95_95178


namespace remainder_of_power_sums_modulo_seven_l95_95341

theorem remainder_of_power_sums_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := 
by 
  sorry

end remainder_of_power_sums_modulo_seven_l95_95341


namespace rectangle_length_l95_95648

theorem rectangle_length (sq_side_len rect_width : ℕ) (sq_area : ℕ) (rect_len : ℕ) 
    (h1 : sq_side_len = 6) 
    (h2 : rect_width = 4) 
    (h3 : sq_area = sq_side_len * sq_side_len) 
    (h4 : sq_area = rect_width * rect_len) :
    rect_len = 9 := 
by 
  sorry

end rectangle_length_l95_95648


namespace Tim_cookie_packages_l95_95336

theorem Tim_cookie_packages 
    (cookies_in_package : ℕ)
    (packets_in_package : ℕ)
    (min_packet_count : ℕ)
    (h1 : cookies_in_package = 5)
    (h2 : packets_in_package = 7)
    (h3 : min_packet_count = 30) :
  ∃ (cookie_packages : ℕ) (packet_packages : ℕ),
    cookie_packages = 7 ∧ packet_packages = 5 ∧
    cookie_packages * cookies_in_package = packet_packages * packets_in_package ∧
    packet_packages * packets_in_package ≥ min_packet_count :=
by
  sorry

end Tim_cookie_packages_l95_95336


namespace junior_high_ten_total_games_l95_95327

theorem junior_high_ten_total_games :
  let teams := 10
  let conference_games_per_team := 3
  let non_conference_games_per_team := 5
  let pairs_of_teams := Nat.choose teams 2
  let total_conference_games := pairs_of_teams * conference_games_per_team
  let total_non_conference_games := teams * non_conference_games_per_team
  let total_games := total_conference_games + total_non_conference_games
  total_games = 185 :=
by
  sorry

end junior_high_ten_total_games_l95_95327


namespace bus_ride_duration_l95_95677

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end bus_ride_duration_l95_95677


namespace cos_150_eq_neg_half_l95_95809

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95809


namespace integer_solution_x_l95_95232

theorem integer_solution_x (x : ℤ) (h₁ : x + 8 > 10) (h₂ : -3 * x < -9) : x ≥ 4 ↔ x > 3 := by
  sorry

end integer_solution_x_l95_95232


namespace complement_set_A_is_04_l95_95532

theorem complement_set_A_is_04 :
  let U := {0, 1, 2, 4}
  let compA := {1, 2}
  ∃ (A : Set ℕ), A = {0, 4} ∧ U = {0, 1, 2, 4} ∧ (U \ A) = compA := 
by
  sorry

end complement_set_A_is_04_l95_95532


namespace find_positive_x_l95_95511

theorem find_positive_x :
  ∃ x : ℝ, x > 0 ∧ (1 / 2 * (4 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4))
  ∧ x = 21 + Real.sqrt 449 :=
by
  sorry

end find_positive_x_l95_95511


namespace harrys_fish_count_l95_95008

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l95_95008


namespace total_carriages_in_towns_l95_95742

noncomputable def total_carriages (euston norfolk norwich flyingScotsman victoria waterloo : ℕ) : ℕ :=
  euston + norfolk + norwich + flyingScotsman + victoria + waterloo

theorem total_carriages_in_towns :
  let euston := 130
  let norfolk := euston - (20 * euston / 100)
  let norwich := 100
  let flyingScotsman := 3 * norwich / 2
  let victoria := euston - (15 * euston / 100)
  let waterloo := 2 * norwich
  total_carriages euston norfolk norwich flyingScotsman victoria waterloo = 794 :=
by
  sorry

end total_carriages_in_towns_l95_95742


namespace cos_150_eq_neg_sqrt3_div_2_l95_95775

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95775


namespace find_p_q_l95_95410

theorem find_p_q (p q : ℤ) 
    (h1 : (3:ℤ)^5 - 2 * (3:ℤ)^4 + 3 * (3:ℤ)^3 - p * (3:ℤ)^2 + q * (3:ℤ) - 12 = 0)
    (h2 : (-1:ℤ)^5 - 2 * (-1:ℤ)^4 + 3 * (-1:ℤ)^3 - p * (-1:ℤ)^2 + q * (-1:ℤ) - 12 = 0) : 
    (p, q) = (-8, -10) :=
by
  sorry

end find_p_q_l95_95410


namespace a₀_value_sum_even_coeffs_l95_95136

open Polynomial

noncomputable def poly := (X - 3)^3 * (2 * X + 1)^5
noncomputable def expansion := C a₀ + C a₁ * X + C a₂ * X^2 + C a₃ * X^3 + C a₄ * X^4 + C a₅ * X^5 + C a₆ * X^6 + C a₇ * X^7 + C a₈ * X^8

-- Prove that the constant term is -27
theorem a₀_value : (poly.eval 0) = (-27 : ℤ) := sorry

-- Prove that the sum a₀ + a₂ + ... + a₈ is -940
theorem sum_even_coeffs : 
  a₀ + a₂ + a₄ + a₆ + a₈ = (-940 : ℤ) := sorry

end a₀_value_sum_even_coeffs_l95_95136


namespace power_fraction_example_l95_95108

theorem power_fraction_example : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := 
by
  sorry

end power_fraction_example_l95_95108


namespace fraction_to_decimal_l95_95503

theorem fraction_to_decimal : (59 / (2^2 * 5^7) : ℝ) = 0.0001888 := by
  sorry

end fraction_to_decimal_l95_95503


namespace cos_150_eq_neg_half_l95_95846

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95846


namespace arithmetic_sequence_sum_l95_95655

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 2 + a 12 = 32) : a 3 + a 11 = 32 :=
sorry

end arithmetic_sequence_sum_l95_95655


namespace inequality_proof_l95_95154

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + b^2 + c^2 + a * b * c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a * b * c ≤ 4 := by
  sorry

end inequality_proof_l95_95154


namespace probability_extremum_at_1_l95_95529

noncomputable def dice_rolls := {(a, b) | a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6}}

noncomputable def favorable_outcomes := {(a, b) | 2 * a = b}

theorem probability_extremum_at_1 : 
  probability (favorable_outcomes ∩ dice_rolls) = 1 / 12 :=
by sorry

end probability_extremum_at_1_l95_95529


namespace largest_square_test_plots_l95_95209

/-- 
  A fenced, rectangular field measures 30 meters by 45 meters. 
  An agricultural researcher has 1500 meters of fence that can be used for internal fencing to partition 
  the field into congruent, square test plots. 
  The entire field must be partitioned, and the sides of the squares must be parallel to the edges of the field. 
  What is the largest number of square test plots into which the field can be partitioned using all or some of the 1500 meters of fence?
 -/
theorem largest_square_test_plots
  (field_length : ℕ := 30)
  (field_width : ℕ := 45)
  (total_fence_length : ℕ := 1500):
  ∃ (n : ℕ), n = 576 := 
sorry

end largest_square_test_plots_l95_95209


namespace trig_simplification_l95_95700

theorem trig_simplification :
  (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
sorry

end trig_simplification_l95_95700


namespace monthly_income_A_l95_95062

theorem monthly_income_A (A B C : ℝ) :
  A + B = 10100 ∧ B + C = 12500 ∧ A + C = 10400 →
  A = 4000 :=
by
  intro h
  have h1 : A + B = 10100 := h.1
  have h2 : B + C = 12500 := h.2.1
  have h3 : A + C = 10400 := h.2.2
  sorry

end monthly_income_A_l95_95062


namespace possible_values_n_l95_95297

theorem possible_values_n (n : ℕ) (h_pos : 0 < n) (h1 : n > 9 / 4) (h2 : n < 14) :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ k ∈ S, k = n :=
by
  -- proof to be filled in
  sorry

end possible_values_n_l95_95297


namespace unique_solution_l95_95619

theorem unique_solution (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  n^2 = m^4 + m^3 + m^2 + m + 1 ↔ (n, m) = (11, 3) :=
by sorry

end unique_solution_l95_95619


namespace probability_in_smaller_spheres_l95_95358

theorem probability_in_smaller_spheres 
    (R r : ℝ)
    (h_eq : ∀ (R r : ℝ), R + r = 4 * r)
    (vol_eq : ∀ (R r : ℝ), (4/3) * π * r^3 * 5 = (4/3) * π * R^3 * (5/27)) :
    P = 0.2 := by
  sorry

end probability_in_smaller_spheres_l95_95358


namespace find_missing_digit_l95_95379

theorem find_missing_digit (B : ℕ) : 
  (B = 2 ∨ B = 4 ∨ B = 7 ∨ B = 8 ∨ B = 9) → 
  (2 * 1000 + B * 100 + 4 * 10 + 0) % 15 = 0 → 
  B = 7 :=
by 
  intro h1 h2
  sorry

end find_missing_digit_l95_95379


namespace cos_150_degree_l95_95832

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95832


namespace harrys_fish_count_l95_95010

theorem harrys_fish_count : 
  let sam_fish := 7
  let joe_fish := 8 * sam_fish
  let harry_fish := 4 * joe_fish
  harry_fish = 224 :=
by
  sorry

end harrys_fish_count_l95_95010


namespace gcd_assoc_gcd_three_eq_gcd_assoc_l95_95689

open Int

theorem gcd_assoc {a b c : ℕ} : Nat.gcd a (Nat.gcd b c) = Nat.gcd (Nat.gcd a b) c := by
  sorry

theorem gcd_three_eq_gcd_assoc {a b c : ℕ} : Nat.gcd3 a b c = Nat.gcd (Nat.gcd a b) c := by
  sorry

end gcd_assoc_gcd_three_eq_gcd_assoc_l95_95689


namespace place_mat_length_l95_95764

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ)
  (table_is_round : r = 3)
  (number_of_mats : n = 8)
  (mat_width : w = 1)
  (mat_length : ∀ (k: ℕ), 0 ≤ k ∧ k < n → (2 * r * Real.sin (Real.pi / n) = x)) :
  x = (3 * Real.sqrt 35) / 10 + 1 / 2 :=
sorry

end place_mat_length_l95_95764


namespace box_surface_area_l95_95962

variables (a b c : ℝ)

noncomputable def sum_edges : ℝ := 4 * (a + b + c)
noncomputable def diagonal_length : ℝ := Real.sqrt (a^2 + b^2 + c^2)
noncomputable def surface_area : ℝ := 2 * (a * b + b * c + c * a)

/- The problem states that the sum of the lengths of the edges and the diagonal length gives us these values. -/
theorem box_surface_area (h1 : sum_edges a b c = 168) (h2 : diagonal_length a b c = 25) : surface_area a b c = 1139 :=
sorry

end box_surface_area_l95_95962


namespace find_f_values_l95_95675

noncomputable def f : ℕ → ℕ := sorry

axiom condition1 : ∀ (a b : ℕ), a ≠ b → (a * f a + b * f b > a * f b + b * f a)
axiom condition2 : ∀ (n : ℕ), f (f n) = 3 * n

theorem find_f_values : f 1 + f 6 + f 28 = 66 := 
by
  sorry

end find_f_values_l95_95675


namespace basis_v_l95_95257

variable {V : Type*} [AddCommGroup V] [Module ℝ V]  -- specifying V as a real vector space
variables (a b c : V)

-- Assume a, b, and c are linearly independent, forming a basis
axiom linear_independent_a_b_c : LinearIndependent ℝ ![a, b, c]

-- The main theorem which we need to prove
theorem basis_v (h : LinearIndependent ℝ ![a, b, c]) :
  LinearIndependent ℝ ![c, a + b, a - b] :=
sorry

end basis_v_l95_95257


namespace roger_shelves_l95_95939

theorem roger_shelves (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : 
  total_books = 24 → 
  books_taken = 3 → 
  books_per_shelf = 4 → 
  Nat.ceil ((total_books - books_taken) / books_per_shelf) = 6 :=
by
  intros h_total h_taken h_per_shelf
  rw [h_total, h_taken, h_per_shelf]
  sorry

end roger_shelves_l95_95939


namespace find_number_l95_95244

noncomputable def solve_N (x : ℝ) (N : ℝ) : Prop :=
  ((N / x) / (3.6 * 0.2) = 2)

theorem find_number (x : ℝ) (N : ℝ) (h1 : x = 12) (h2 : solve_N x N) : N = 17.28 :=
  by
  sorry

end find_number_l95_95244


namespace cos_150_deg_l95_95824

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95824


namespace john_personal_payment_l95_95149

-- Definitions of the conditions
def cost_of_one_hearing_aid : ℕ := 2500
def number_of_hearing_aids : ℕ := 2
def insurance_coverage_percent : ℕ := 80

-- Derived definitions based on conditions
def total_cost : ℕ := cost_of_one_hearing_aid * number_of_hearing_aids
def insurance_coverage_amount : ℕ := total_cost * insurance_coverage_percent / 100
def johns_share : ℕ := total_cost - insurance_coverage_amount

-- Theorem statement (proof not included)
theorem john_personal_payment : johns_share = 1000 :=
sorry

end john_personal_payment_l95_95149


namespace intersection_of_M_and_N_is_1_l95_95531

open Nat

noncomputable def NatStar : set ℕ := {n | n > 0}

def M : set ℕ := {0, 1, 2}

def N : set ℕ := {x | ∃ a ∈ NatStar, x = 2 * a - 1}

theorem intersection_of_M_and_N_is_1 : M ∩ N = {1} :=
by
  sorry

end intersection_of_M_and_N_is_1_l95_95531


namespace find_remainder_l95_95960

def dividend : ℕ := 997
def divisor : ℕ := 23
def quotient : ℕ := 43

theorem find_remainder : ∃ r : ℕ, dividend = (divisor * quotient) + r ∧ r = 8 :=
by
  sorry

end find_remainder_l95_95960


namespace family_b_initial_members_l95_95347

variable (x : ℕ)

theorem family_b_initial_members (h : 6 + (x - 1) + 9 + 12 + 5 + 9 = 48) : x = 8 :=
by
  sorry

end family_b_initial_members_l95_95347


namespace find_x_l95_95767

def magic_constant (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧ d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧ b + e + h = c + f + i ∧
  a + e + i = c + e + g

def given_magic_square (x : ℤ) : Prop :=
  magic_constant (4017) (2012) (0) 
                 (4015) (x - 2003) (11) 
                 (2014) (9) (x)

theorem find_x (x : ℤ) (h : given_magic_square x) : x = 4003 :=
by {
  sorry
}

end find_x_l95_95767


namespace sandwiches_sold_out_l95_95722

-- Define the parameters as constant values
def original : ℕ := 9
def available : ℕ := 4

-- The theorem stating the problem and the expected result
theorem sandwiches_sold_out : (original - available) = 5 :=
by
  -- This is the placeholder for the proof
  sorry

end sandwiches_sold_out_l95_95722


namespace third_dimension_of_box_l95_95179

theorem third_dimension_of_box (h : ℕ) (H : (151^2 - 150^2) * h + 151^2 = 90000) : h = 223 :=
sorry

end third_dimension_of_box_l95_95179


namespace cos_150_eq_neg_sqrt3_over_2_l95_95816

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95816


namespace abc_sum_zero_l95_95418

variable (a b c : ℝ)

-- Conditions given in the original problem
axiom h1 : a + b / c = 1
axiom h2 : b + c / a = 1
axiom h3 : c + a / b = 1

theorem abc_sum_zero : a * b + b * c + c * a = 0 :=
by
  sorry

end abc_sum_zero_l95_95418


namespace base_conversion_l95_95301

theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 71) : x = 16 := 
by {
  sorry
}

end base_conversion_l95_95301


namespace Seokgi_candies_l95_95448

theorem Seokgi_candies (C : ℕ) 
  (h1 : C / 2 + (C - C / 2) / 3 + 12 = C)
  (h2 : ∃ x, x = 12) :
  C = 36 := 
by 
  sorry

end Seokgi_candies_l95_95448


namespace nate_distance_after_resting_l95_95937

variables (length_of_field total_distance : ℕ)

def distance_before_resting (length_of_field : ℕ) := 4 * length_of_field

def distance_after_resting (total_distance length_of_field : ℕ) : ℕ := 
  total_distance - distance_before_resting length_of_field

theorem nate_distance_after_resting
  (length_of_field_val : length_of_field = 168)
  (total_distance_val : total_distance = 1172) :
  distance_after_resting total_distance length_of_field = 500 :=
by
  -- Proof goes here
  sorry

end nate_distance_after_resting_l95_95937


namespace remainder_when_x_plus_4uy_div_y_l95_95946

theorem remainder_when_x_plus_4uy_div_y (x y u v : ℕ) (h₀: x = u * y + v) (h₁: 0 ≤ v) (h₂: v < y) : 
  ((x + 4 * u * y) % y) = v := 
by 
  sorry

end remainder_when_x_plus_4uy_div_y_l95_95946


namespace multiply_polynomials_l95_95227

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end multiply_polynomials_l95_95227


namespace sin_cos_monotonic_increasing_interval_l95_95176

theorem sin_cos_monotonic_increasing_interval : 
  ∃ (a b : ℝ), a = -π / 8 ∧ b = 3 * π / 8 ∧
  ∀ x y : ℝ, (a ≤ x ∧ x < y ∧ y ≤ b) → (sin (2 * x) - cos (2 * x) ≤ sin (2 * y) - cos (2 * y)) := by
  sorry

end sin_cos_monotonic_increasing_interval_l95_95176


namespace thirty_divides_p_squared_minus_one_iff_p_eq_five_l95_95283

theorem thirty_divides_p_squared_minus_one_iff_p_eq_five (p : ℕ) (hp : Nat.Prime p) (h_ge : p ≥ 5) : 30 ∣ (p^2 - 1) ↔ p = 5 :=
by
  sorry

end thirty_divides_p_squared_minus_one_iff_p_eq_five_l95_95283


namespace min_value_of_expression_l95_95674

noncomputable def minimum_value_expression : ℝ :=
  let f (a b : ℝ) := a^4 + b^4 + 16 / (a^2 + b^2)^2
  4

theorem min_value_of_expression (a b : ℝ) (h : 0 < a ∧ 0 < b) : 
  let f := a^4 + b^4 + 16 / (a^2 + b^2)^2
  ∃ c : ℝ, f = c ∧ c = 4 :=
sorry

end min_value_of_expression_l95_95674


namespace time_expression_l95_95203

theorem time_expression (h V₀ g S V t : ℝ) :
  (V = g * t + V₀) →
  (S = h + (1 / 2) * g * t^2 + V₀ * t) →
  t = (2 * (S - h)) / (V + V₀) :=
by
  intro h_eq v_eq
  sorry

end time_expression_l95_95203


namespace total_students_l95_95182

theorem total_students (groups students_per_group : ℕ) (h : groups = 6) (k : students_per_group = 5) :
  groups * students_per_group = 30 := 
by
  sorry

end total_students_l95_95182


namespace index_cards_per_student_l95_95087

theorem index_cards_per_student
    (periods_per_day : ℕ)
    (students_per_class : ℕ)
    (cost_per_pack : ℕ)
    (total_spent : ℕ)
    (cards_per_pack : ℕ)
    (total_packs : ℕ)
    (total_index_cards : ℕ)
    (total_students : ℕ)
    (index_cards_per_student : ℕ)
    (h1 : periods_per_day = 6)
    (h2 : students_per_class = 30)
    (h3 : cost_per_pack = 3)
    (h4 : total_spent = 108)
    (h5 : cards_per_pack = 50)
    (h6 : total_packs = total_spent / cost_per_pack)
    (h7 : total_index_cards = total_packs * cards_per_pack)
    (h8 : total_students = periods_per_day * students_per_class)
    (h9 : index_cards_per_student = total_index_cards / total_students) :
    index_cards_per_student = 10 := 
  by
    sorry

end index_cards_per_student_l95_95087


namespace selling_price_l95_95445

noncomputable def total_cost_first_mixture : ℝ := 27 * 150
noncomputable def total_cost_second_mixture : ℝ := 36 * 125
noncomputable def total_cost_third_mixture : ℝ := 18 * 175
noncomputable def total_cost_fourth_mixture : ℝ := 24 * 120

noncomputable def total_cost : ℝ := total_cost_first_mixture + total_cost_second_mixture + total_cost_third_mixture + total_cost_fourth_mixture

noncomputable def profit_first_mixture : ℝ := 0.4 * total_cost_first_mixture
noncomputable def profit_second_mixture : ℝ := 0.3 * total_cost_second_mixture
noncomputable def profit_third_mixture : ℝ := 0.2 * total_cost_third_mixture
noncomputable def profit_fourth_mixture : ℝ := 0.25 * total_cost_fourth_mixture

noncomputable def total_profit : ℝ := profit_first_mixture + profit_second_mixture + profit_third_mixture + profit_fourth_mixture

noncomputable def total_weight : ℝ := 27 + 36 + 18 + 24
noncomputable def total_selling_price : ℝ := total_cost + total_profit

noncomputable def selling_price_per_kg : ℝ := total_selling_price / total_weight

theorem selling_price : selling_price_per_kg = 180 := by
  sorry

end selling_price_l95_95445


namespace bridge_toll_fees_for_annie_are_5_l95_95315

-- Conditions
def start_fee : ℝ := 2.50
def cost_per_mile : ℝ := 0.25
def mike_miles : ℕ := 36
def annie_miles : ℕ := 16
def total_cost_mike : ℝ := start_fee + cost_per_mile * mike_miles

-- Hypothesis from conditions
axiom both_charged_same : ∀ (bridge_fees : ℝ), total_cost_mike = start_fee + cost_per_mile * annie_miles + bridge_fees

-- Proof problem
theorem bridge_toll_fees_for_annie_are_5 : ∃ (bridge_fees : ℝ), bridge_fees = 5 :=
by
  existsi 5
  sorry

end bridge_toll_fees_for_annie_are_5_l95_95315


namespace cara_pairs_between_l95_95231

theorem cara_pairs_between (friends : Fin 8) (emma : Fin 8) (cara : Fin 8) :
  (emma ≠ cara ∧ ∀ f : Fin 8, f ≠ emma ∧ f ≠ cara → true) →
  ∃ (n : Nat), n = 6 :=
by
  sorry

end cara_pairs_between_l95_95231


namespace ratio_second_shop_to_shirt_l95_95156

-- Define the initial conditions in Lean
def initial_amount : ℕ := 55
def spent_on_shirt : ℕ := 7
def final_amount : ℕ := 27

-- Define the amount spent in the second shop calculation
def spent_in_second_shop (i_amt s_shirt f_amt : ℕ) : ℕ :=
  (i_amt - s_shirt) - f_amt

-- Define the ratio calculation
def ratio (a b : ℕ) : ℕ := a / b

-- Lean 4 statement proving the ratio of amounts
theorem ratio_second_shop_to_shirt : 
  ratio (spent_in_second_shop initial_amount spent_on_shirt final_amount) spent_on_shirt = 3 := 
by
  sorry

end ratio_second_shop_to_shirt_l95_95156


namespace no_rational_points_on_sqrt3_circle_l95_95603

theorem no_rational_points_on_sqrt3_circle (x y : ℚ) : x^2 + y^2 ≠ 3 :=
sorry

end no_rational_points_on_sqrt3_circle_l95_95603


namespace total_miles_walked_by_group_in_6_days_l95_95423

-- Conditions translated to Lean definitions
def miles_per_day_group := 3
def additional_miles_per_day := 2
def days_in_week := 6
def total_ladies := 5

-- Question translated to a Lean theorem statement
theorem total_miles_walked_by_group_in_6_days : 
  ∀ (miles_per_day_group additional_miles_per_day days_in_week total_ladies : ℕ),
  (miles_per_day_group * total_ladies * days_in_week) + 
  ((miles_per_day_group * (total_ladies - 1) * days_in_week) + (additional_miles_per_day * days_in_week)) = 120 := 
by
  intros
  sorry

end total_miles_walked_by_group_in_6_days_l95_95423


namespace probability_businessmen_wait_two_minutes_l95_95984

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l95_95984


namespace solve_for_x_l95_95171

theorem solve_for_x (x : ℚ) :
  (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 73 ↔ x = -647 / 177 :=
by sorry

end solve_for_x_l95_95171


namespace position_of_21_over_19_in_sequence_l95_95890

def sequence_term (n : ℕ) : ℚ := (n + 3) / (n + 1)

theorem position_of_21_over_19_in_sequence :
  ∃ n : ℕ, sequence_term n = 21 / 19 ∧ n = 18 :=
by sorry

end position_of_21_over_19_in_sequence_l95_95890


namespace no_tetrahedron_with_given_heights_l95_95099

theorem no_tetrahedron_with_given_heights (h1 h2 h3 h4 : ℝ) (V : ℝ) (V_pos : V > 0)
    (S1 : ℝ := 3*V) (S2 : ℝ := (3/2)*V) (S3 : ℝ := V) (S4 : ℝ := V/2) :
    (h1 = 1) → (h2 = 2) → (h3 = 3) → (h4 = 6) → ¬ ∃ (S1 S2 S3 S4 : ℝ), S1 < S2 + S3 + S4 := by
  intros
  sorry

end no_tetrahedron_with_given_heights_l95_95099


namespace sum_even_if_product_odd_l95_95650

theorem sum_even_if_product_odd (a b : ℤ) (h : (a * b) % 2 = 1) : (a + b) % 2 = 0 := 
by
  sorry

end sum_even_if_product_odd_l95_95650


namespace area_of_triangle_l95_95022

theorem area_of_triangle (S_x S_y S_z S : ℝ)
  (hx : S_x = Real.sqrt 7) (hy : S_y = Real.sqrt 6)
  (hz : ∃ k : ℕ, S_z = k) (hs : ∃ n : ℕ, S = n)
  : S = 7 := by
  sorry

end area_of_triangle_l95_95022


namespace cos_150_eq_neg_sqrt3_div_2_l95_95802

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95802


namespace chicken_feathers_after_crossing_l95_95457

def cars_dodged : ℕ := 23
def initial_feathers : ℕ := 5263
def feathers_lost : ℕ := 2 * cars_dodged
def final_feathers : ℕ := initial_feathers - feathers_lost

theorem chicken_feathers_after_crossing :
  final_feathers = 5217 := by
sorry

end chicken_feathers_after_crossing_l95_95457


namespace probability_businessmen_wait_two_minutes_l95_95985

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l95_95985


namespace simplify_fraction_l95_95942

theorem simplify_fraction :
  (1 / (1 / (1 / 2) ^ 1 + 1 / (1 / 2) ^ 2 + 1 / (1 / 2) ^ 3)) = (1 / 14) :=
by 
  sorry

end simplify_fraction_l95_95942


namespace tomato_price_l95_95571

theorem tomato_price (P : ℝ) (W : ℝ) :
  (0.9956 * 0.9 * W = P * W + 0.12 * (P * W)) → P = 0.8 :=
by
  intro h
  sorry

end tomato_price_l95_95571


namespace divisible_by_n_sequence_l95_95016

theorem divisible_by_n_sequence (n : ℕ) (h1 : n > 1) (h2 : n % 2 = 1) : 
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ n ∣ (2^k - 1) :=
by {
  sorry
}

end divisible_by_n_sequence_l95_95016


namespace sarah_total_volume_in_two_weeks_l95_95691

def shampoo_daily : ℝ := 1

def conditioner_daily : ℝ := 1 / 2 * shampoo_daily

def days : ℕ := 14

def total_volume : ℝ := (shampoo_daily * days) + (conditioner_daily * days)

theorem sarah_total_volume_in_two_weeks : total_volume = 21 := by
  sorry

end sarah_total_volume_in_two_weeks_l95_95691


namespace determine_operation_l95_95233

theorem determine_operation (a b c d : Int) : ((a - b) + c - (3 * 1) = d) → ((a - b) + 2 = 6) → (a - b = 4) :=
by
  sorry

end determine_operation_l95_95233


namespace intersection_M_N_l95_95128

open Set Real

def M := {x : ℝ | x^2 + x - 6 < 0}
def N := {x : ℝ | abs (x - 1) ≤ 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l95_95128


namespace incorrect_option_D_l95_95589

-- Definitions based on the given conditions:
def contrapositive_correct : Prop :=
  ∀ x : ℝ, (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0) ↔ (x^2 - 3 * x + 2 = 0 → x = 1)

def sufficient_but_not_necessary : Prop :=
  ∀ x : ℝ, (x > 2 → x^2 - 3 * x + 2 > 0) ∧ (x^2 - 3 * x + 2 > 0 → x > 2 ∨ x < 1)

def negation_correct (p : Prop) (neg_p : Prop) : Prop :=
  p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0 ∧ neg_p ↔ ∃ x_0 : ℝ, x_0^2 + x_0 + 1 = 0

theorem incorrect_option_D (p q : Prop) (h : p ∨ q) :
  ¬ (p ∧ q) :=
sorry  -- Proof is to be done later

end incorrect_option_D_l95_95589


namespace vertices_of_regular_hexagonal_pyramid_l95_95513

-- Define a structure for a regular hexagonal pyramid
structure RegularHexagonalPyramid where
  baseVertices : Nat
  apexVertices : Nat

-- Define a specific regular hexagonal pyramid with given conditions
def regularHexagonalPyramid : RegularHexagonalPyramid :=
  { baseVertices := 6, apexVertices := 1 }

-- The theorem stating the number of vertices of the pyramid
theorem vertices_of_regular_hexagonal_pyramid : regularHexagonalPyramid.baseVertices + regularHexagonalPyramid.apexVertices = 7 := 
  by
  sorry

end vertices_of_regular_hexagonal_pyramid_l95_95513


namespace cos_150_deg_l95_95825

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95825


namespace arithmetic_sequence_ratio_l95_95172

theorem arithmetic_sequence_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = (1/2) * n * (2 * a 1 + (n-1) * d))
  (h2 : ∀ n, T n = (1/2) * n * (2 * b 1 + (n-1) * d'))
  (h3 : ∀ n, S n / T n = 7*n / (n + 3)): a 5 / b 5 = 21 / 4 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l95_95172


namespace cos_150_eq_neg_sqrt3_over_2_l95_95819

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95819


namespace max_volume_of_box_l95_95740

theorem max_volume_of_box (sheetside : ℝ) (cutside : ℝ) (volume : ℝ) 
  (h1 : sheetside = 6) 
  (h2 : ∀ (x : ℝ), 0 < x ∧ x < (sheetside / 2) → volume = x * (sheetside - 2 * x)^2) : 
  cutside = 1 :=
by
  sorry

end max_volume_of_box_l95_95740


namespace total_weight_puffy_muffy_l95_95223

def scruffy_weight : ℕ := 12
def muffy_weight : ℕ := scruffy_weight - 3
def puffy_weight : ℕ := muffy_weight + 5

theorem total_weight_puffy_muffy : puffy_weight + muffy_weight = 23 := 
by
  sorry

end total_weight_puffy_muffy_l95_95223


namespace copper_tin_ratio_l95_95300

theorem copper_tin_ratio 
    (w1 w2 w_new : ℝ) 
    (r1_copper r1_tin r2_copper r2_tin : ℝ) 
    (r_new_copper r_new_tin : ℝ)
    (pure_copper : ℝ)
    (h1 : w1 = 10)
    (h2 : w2 = 16)
    (h3 : r1_copper = 4 / 5 * w1)
    (h4 : r1_tin = 1 / 5 * w1)
    (h5 : r2_copper = 1 / 4 * w2)
    (h6 : r2_tin = 3 / 4 * w2)
    (h7 : r_new_copper = r1_copper + r2_copper + pure_copper)
    (h8 : r_new_tin = r1_tin + r2_tin)
    (h9 : w_new = 35)
    (h10 : r_new_copper + r_new_tin + pure_copper = w_new)
    (h11 : pure_copper = 9) :
    r_new_copper / r_new_tin = 3 / 2 :=
by
  sorry

end copper_tin_ratio_l95_95300


namespace sin_phi_value_l95_95400

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem sin_phi_value (φ : ℝ) (h_shift : ∀ x, g x = f (x - φ)) : Real.sin φ = 24 / 25 :=
by
  sorry

end sin_phi_value_l95_95400


namespace joy_sees_grandma_in_48_hours_l95_95924

def days_until_joy_sees_grandma : ℕ := 2
def hours_per_day : ℕ := 24

theorem joy_sees_grandma_in_48_hours :
  days_until_joy_sees_grandma * hours_per_day = 48 := 
by
  sorry

end joy_sees_grandma_in_48_hours_l95_95924


namespace best_choice_to_calculate_89_8_sq_l95_95706

theorem best_choice_to_calculate_89_8_sq 
  (a b c d : ℚ) 
  (h1 : (89 + 0.8)^2 = a) 
  (h2 : (80 + 9.8)^2 = b) 
  (h3 : (90 - 0.2)^2 = c) 
  (h4 : (100 - 10.2)^2 = d) : 
  c = 89.8^2 := by
  sorry

end best_choice_to_calculate_89_8_sq_l95_95706


namespace simplify_fraction_l95_95941

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) : 
  (b - 1) / (b + b / (b - 1)) = (b - 1) ^ 2 / b ^ 2 := 
by {
  sorry
}

end simplify_fraction_l95_95941


namespace differential_equation_for_lines_one_unit_from_origin_l95_95880

-- Define the problem conditions
theorem differential_equation_for_lines_one_unit_from_origin
  (α : ℝ) (x y : ℝ) (h : x * cos α + y * sin α = 1) :
  ∃ y', y = x * y' + sqrt (1 + y' ^ 2) :=
by
  sorry

end differential_equation_for_lines_one_unit_from_origin_l95_95880


namespace quadratic_function_properties_l95_95635

-- We define the primary conditions
def axis_of_symmetry (f : ℝ → ℝ) (x_sym : ℝ) : Prop := 
  ∀ x, f x = f (2 * x_sym - x)

def minimum_value (f : ℝ → ℝ) (y_min : ℝ) (x_min : ℝ) : Prop := 
  ∀ x, f x_min ≤ f x

def passes_through (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop := 
  f pt.1 = pt.2

-- We need to prove that a quadratic function exists with the given properties and find intersections
theorem quadratic_function_properties :
  ∃ f : ℝ → ℝ,
    axis_of_symmetry f (-1) ∧
    minimum_value f (-4) (-1) ∧
    passes_through f (-2, 5) ∧
    (∀ y : ℝ, f 0 = y → y = 5) ∧
    (∀ x : ℝ, f x = 0 → (x = -5/3 ∨ x = -1/3)) :=
sorry

end quadratic_function_properties_l95_95635


namespace cos_150_deg_l95_95822

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l95_95822


namespace kim_paints_fewer_tiles_than_laura_l95_95375

-- Given conditions and definitions
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def total_tiles_per_15_minutes : ℕ := 375
def total_rate_per_minute : ℕ := total_tiles_per_15_minutes / 15
def kim_rate : ℕ := total_rate_per_minute - (don_rate + ken_rate + laura_rate)

-- Proof goal
theorem kim_paints_fewer_tiles_than_laura :
  laura_rate - kim_rate = 3 :=
by
  sorry

end kim_paints_fewer_tiles_than_laura_l95_95375


namespace average_visitors_per_day_in_november_l95_95211
-- Import the entire Mathlib library for necessary definitions and operations.

-- Define the average visitors per different days of the week.
def sunday_visitors := 510
def monday_visitors := 240
def tuesday_visitors := 240
def wednesday_visitors := 300
def thursday_visitors := 300
def friday_visitors := 200
def saturday_visitors := 200

-- Define the counts of each type of day in November.
def sundays := 5
def mondays := 4
def tuesdays := 4
def wednesdays := 4
def thursdays := 4
def fridays := 4
def saturdays := 4

-- Define the number of days in November.
def days_in_november := 30

-- State the theorem to prove the average number of visitors per day.
theorem average_visitors_per_day_in_november : 
  (5 * sunday_visitors + 
   4 * monday_visitors + 
   4 * tuesday_visitors + 
   4 * wednesday_visitors + 
   4 * thursday_visitors + 
   4 * friday_visitors + 
   4 * saturday_visitors) / days_in_november = 282 :=
by
  sorry

end average_visitors_per_day_in_november_l95_95211


namespace vikki_hourly_pay_rate_l95_95465

-- Define the variables and conditions
def hours_worked : ℝ := 42
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5
def net_pay : ℝ := 310

-- Define Vikki's hourly pay rate (we will solve for this)
variable (hourly_pay : ℝ)

-- Define the gross earnings
def gross_earnings (hourly_pay : ℝ) : ℝ := hours_worked * hourly_pay

-- Define the total deductions
def total_deductions (hourly_pay : ℝ) : ℝ := (tax_rate * gross_earnings hourly_pay) + (insurance_rate * gross_earnings hourly_pay) + union_dues

-- Define the net pay
def calculate_net_pay (hourly_pay : ℝ) : ℝ := gross_earnings hourly_pay - total_deductions hourly_pay

-- Prove the solution
theorem vikki_hourly_pay_rate : calculate_net_pay hourly_pay = net_pay → hourly_pay = 10 := by
  sorry

end vikki_hourly_pay_rate_l95_95465


namespace factorize_quadratic_l95_95508

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l95_95508


namespace total_additions_in_2_hours_30_minutes_l95_95075

def additions_rate : ℕ := 15000

def time_in_seconds : ℕ := 2 * 3600 + 30 * 60

def total_additions : ℕ := additions_rate * time_in_seconds

theorem total_additions_in_2_hours_30_minutes :
  total_additions = 135000000 :=
by
  -- Non-trivial proof skipped
  sorry

end total_additions_in_2_hours_30_minutes_l95_95075


namespace solution_correct_l95_95615

-- Define the conditions
def condition1 (x : ℝ) : Prop := 2 ≤ |x - 3| ∧ |x - 3| ≤ 5
def condition2 (x : ℝ) : Prop := (x - 3) ^ 2 ≤ 16

-- Define the solution set
def solution_set (x : ℝ) : Prop := (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7)

-- Prove that the solution set is correct given the conditions
theorem solution_correct (x : ℝ) : condition1 x ∧ condition2 x ↔ solution_set x :=
by
  sorry

end solution_correct_l95_95615


namespace UN_anniversary_day_l95_95020

/--
The United Nations was founded on October 24, 1945, which was a Wednesday.
Prove that the 75th anniversary of this event occurred on a Friday in the year 2020.
-/
theorem UN_anniversary_day
  (start_day : ℕ := 3) -- Wednesday is represented by 3
  (years_diff : ℕ := 75)
  (leap_years : ℕ := 18)
  (regular_years : ℕ := 57)
  (days_in_week : ℕ := 7)
  (mod_days : ℕ := 2) : 
  (start_day + regular_years + 2 * leap_years) % days_in_week = 5 := 
by sorry

end UN_anniversary_day_l95_95020


namespace proof_l95_95496

variable {S : Type} 
variable (op : S → S → S)

-- Condition given in the problem
def condition (a b : S) : Prop :=
  op (op a b) a = b

-- Statement to be proven
theorem proof (h : ∀ a b : S, condition op a b) :
  ∀ a b : S, op a (op b a) = b :=
by
  intros a b
  sorry

end proof_l95_95496


namespace geometric_to_arithmetic_sequence_l95_95250

theorem geometric_to_arithmetic_sequence {a : ℕ → ℝ} (q : ℝ) 
    (h_gt0 : 0 < q) (h_pos : ∀ n, 0 < a n)
    (h_geom_seq : ∀ n, a (n + 1) = a n * q)
    (h_arith_seq : 2 * (1 / 2 * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end geometric_to_arithmetic_sequence_l95_95250


namespace calculate_ratio_l95_95898

variables (M Q P N R : ℝ)

-- Definitions of conditions
def M_def : M = 0.40 * Q := by sorry
def Q_def : Q = 0.30 * P := by sorry
def N_def : N = 0.60 * P := by sorry
def R_def : R = 0.20 * P := by sorry

-- Statement of the proof problem
theorem calculate_ratio (hM : M = 0.40 * Q) (hQ : Q = 0.30 * P)
  (hN : N = 0.60 * P) (hR : R = 0.20 * P) : 
  (M + R) / N = 8 / 15 := by
  sorry

end calculate_ratio_l95_95898


namespace option_d_always_holds_l95_95056

theorem option_d_always_holds (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b := by
  sorry

end option_d_always_holds_l95_95056


namespace cuboid_edge_length_l95_95328

theorem cuboid_edge_length (x : ℝ) (h1 : (2 * (x * 5 + x * 6 + 5 * 6)) = 148) : x = 4 :=
by 
  sorry

end cuboid_edge_length_l95_95328


namespace max_value_abs_x_sub_3y_l95_95436

theorem max_value_abs_x_sub_3y 
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + 3 * y ≤ 4)
  (h3 : x ≥ -2) : 
  ∃ z, z = |x - 3 * y| ∧ ∀ (x y : ℝ), (y ≥ x) → (x + 3 * y ≤ 4) → (x ≥ -2) → |x - 3 * y| ≤ 4 :=
sorry

end max_value_abs_x_sub_3y_l95_95436


namespace range_of_f_log_gt_zero_l95_95863

open Real

noncomputable def f (x : ℝ) : ℝ := -- Placeholder function definition
  sorry

theorem range_of_f_log_gt_zero :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) ∧
  (f (1 / 3) = 0) →
  {x : ℝ | f ((log x) / (log (1 / 8))) > 0} = 
    (Set.Ioo 0 (1 / 2) ∪ Set.Ioi 2) :=
  sorry

end range_of_f_log_gt_zero_l95_95863


namespace statement_c_false_l95_95057

theorem statement_c_false : ¬ ∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end statement_c_false_l95_95057


namespace multiply_polynomials_l95_95229

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end multiply_polynomials_l95_95229


namespace probability_businessmen_wait_two_minutes_l95_95986

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l95_95986


namespace chi_square_hypothesis_test_l95_95460

-- Definitions based on the conditions
def males_like_sports := "Males like to participate in sports activities"
def females_dislike_sports := "Females do not like to participate in sports activities"
def activities_related_to_gender := "Liking to participate in sports activities is related to gender"
def activities_not_related_to_gender := "Liking to participate in sports activities is not related to gender"

-- Statement to prove that D is the correct null hypothesis
theorem chi_square_hypothesis_test :
  activities_not_related_to_gender = "H₀: Liking to participate in sports activities is not related to gender" :=
sorry

end chi_square_hypothesis_test_l95_95460


namespace cos_150_eq_neg_half_l95_95842

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95842


namespace john_probability_l95_95426

/-- John arrives at a terminal which has sixteen gates arranged in a straight line with exactly 50 feet between adjacent gates. His departure gate is assigned randomly. After waiting at that gate, John is informed that the departure gate has been changed to another gate, chosen randomly again. Prove that the probability that John walks 200 feet or less to the new gate is \(\frac{4}{15}\), and find \(4 + 15 = 19\) -/
theorem john_probability :
  let n_gates := 16
  let dist_between_gates := 50
  let max_walk_dist := 200
  let total_possibilities := n_gates * (n_gates - 1)
  let valid_cases :=
    4 * (2 + 2 * (4 - 1))
  let probability_within_200_feet := valid_cases / total_possibilities
  let fraction := probability_within_200_feet * (15 / 4)
  fraction = 1 → 4 + 15 = 19 := by
  sorry -- Proof goes here 

end john_probability_l95_95426


namespace solve_for_a_l95_95261

-- Defining the equation and given solution
theorem solve_for_a (x a : ℝ) (h : 2 * x - 5 * a = 3 * a + 22) (hx : x = 3) : a = -2 := by
  sorry

end solve_for_a_l95_95261


namespace kindergarteners_line_up_probability_l95_95763

theorem kindergarteners_line_up_probability :
  let total_line_up := Nat.choose 20 9
  let first_scenario := Nat.choose 14 9
  let second_scenario_single := Nat.choose 13 8
  let second_scenario := 6 * second_scenario_single
  let valid_arrangements := first_scenario + second_scenario
  valid_arrangements / total_line_up = 9724 / 167960 := by
  sorry

end kindergarteners_line_up_probability_l95_95763


namespace lucy_current_fish_l95_95934

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end lucy_current_fish_l95_95934


namespace smarties_modulo_l95_95868

theorem smarties_modulo (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end smarties_modulo_l95_95868


namespace geometric_sequence_a5_value_l95_95541

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n m : ℕ, a n = a 0 * r ^ n)
  (h_condition : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_a5_value_l95_95541


namespace find_MN_l95_95577

theorem find_MN (d D : ℝ) (h_d_lt_D : d < D) :
  ∃ MN : ℝ, MN = (d * D) / (D - d) :=
by
  sorry

end find_MN_l95_95577


namespace sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l95_95968

theorem sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions :
  ∃ (a b : ℕ), (4 < a ∧ a < b ∧ b < 16) ∧
  (∃ r : ℚ, a = 4 * r ∧ b = 4 * r * r) ∧
  (a + b = 2 * b - a + 16) ∧
  a + b = 24 :=
by
  sorry

end sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l95_95968


namespace cos_150_eq_neg_half_l95_95845

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95845


namespace cos_150_eq_neg_sqrt3_div_2_l95_95780

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95780


namespace boys_trees_l95_95100

theorem boys_trees (avg_per_person trees_per_girl trees_per_boy : ℕ) :
  avg_per_person = 6 →
  trees_per_girl = 15 →
  (1 / trees_per_boy + 1 / trees_per_girl = 1 / avg_per_person) →
  trees_per_boy = 10 :=
by
  intros h_avg h_girl h_eq
  -- We will provide the proof here eventually
  sorry

end boys_trees_l95_95100


namespace cuboid_volume_l95_95072

theorem cuboid_volume (a b c : ℕ) (h_incr_by_2_becomes_cube : c + 2 = a)
  (surface_area_incr : 2*a*(a + a + c + 2) - 2*a*(c + a + b) = 56) : a * b * c = 245 :=
sorry

end cuboid_volume_l95_95072


namespace cos_150_eq_neg_sqrt3_div_2_l95_95852

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95852


namespace expression_even_l95_95393

theorem expression_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 1) :
  ∃ k : ℕ, 2^a * (b+1) ^ 2 * c = 2 * k :=
by
sorry

end expression_even_l95_95393


namespace how_many_lassis_l95_95770

def lassis_per_mango : ℕ := 15 / 3

def lassis15mangos : ℕ := 15

theorem how_many_lassis (H : lassis_per_mango = 5) : lassis15mangos * lassis_per_mango = 75 :=
by
  rw [H]
  sorry

end how_many_lassis_l95_95770


namespace colonization_combinations_l95_95282

noncomputable def number_of_combinations : ℕ :=
  let earth_like_bound := 7
  let mars_like_bound := 8
  let total_units := 21
  Finset.card ((Finset.range (earth_like_bound + 1)).filter (λ a, 
    (3 * a ≤ total_units ∧ (total_units - 3 * a) ≤ mars_like_bound)))

theorem colonization_combinations : number_of_combinations = 981 := by
  sorry

end colonization_combinations_l95_95282


namespace two_false_propositions_l95_95614

theorem two_false_propositions (a : ℝ) :
  (¬((a > -3) → (a > -6))) ∧ (¬((a > -6) → (a > -3))) → (¬(¬(a > -3) → ¬(a > -6))) :=
by
  sorry

end two_false_propositions_l95_95614


namespace cos_150_degree_l95_95829

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95829


namespace repeating_decimals_subtraction_l95_95738

theorem repeating_decimals_subtraction :
  let x := (246/999 : ℚ)
  let y := (135/999 : ℚ)
  let z := (9753/9999 : ℚ)
  (x - y - z) = (-8647897/9989001 : ℚ) := 
by
  sorry

end repeating_decimals_subtraction_l95_95738


namespace chess_tournament_num_players_l95_95292

theorem chess_tournament_num_players (n : ℕ) :
  (∀ k, k ≠ n → exists m, m ≠ n ∧ (k = m)) ∧ 
  ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))) = (1 / 13 * ((1 / 2 * n * (n - 1)) - ((1 / 2 * (n - 1)) + (1 / 4 * (n - 1))))) →
  n = 21 :=
by
  sorry

end chess_tournament_num_players_l95_95292


namespace quadratic_value_at_6_l95_95389

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 3

theorem quadratic_value_at_6 
  (a b : ℝ) (h : a ≠ 0) 
  (h_eq : f a b 2 = f a b 4) : 
  f a b 6 = -3 :=
by
  sorry

end quadratic_value_at_6_l95_95389


namespace problem_solution_l95_95392

theorem problem_solution (x m : ℝ) (h1 : x ≠ 0) (h2 : x / (x^2 - m*x + 1) = 1) :
  x^3 / (x^6 - m^3 * x^3 + 1) = 1 / (3 * m^2 - 2) :=
by
  sorry

end problem_solution_l95_95392


namespace find_number_l95_95370

theorem find_number (x : ℕ) (h : 24 * x = 2376) : x = 99 :=
by
  sorry

end find_number_l95_95370


namespace original_number_of_men_l95_95470

/-- 
Given:
1. A group of men decided to do a work in 20 days,
2. When 2 men became absent, the remaining men did the work in 22 days,

Prove:
The original number of men in the group was 22.
-/
theorem original_number_of_men (x : ℕ) (h : 20 * x = 22 * (x - 2)) : x = 22 :=
by
  sorry

end original_number_of_men_l95_95470


namespace cos_150_eq_negative_cos_30_l95_95787

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95787


namespace large_planks_need_15_nails_l95_95923

-- Definitions based on given conditions
def total_nails : ℕ := 20
def small_planks_nails : ℕ := 5

-- Question: How many nails do the large planks need together?
-- Prove that the large planks need 15 nails together given the conditions.
theorem large_planks_need_15_nails : total_nails - small_planks_nails = 15 :=
by
  sorry

end large_planks_need_15_nails_l95_95923


namespace factorize_m_square_minus_16_l95_95105

-- Define the expression
def expr (m : ℝ) : ℝ := m^2 - 16

-- Define the factorized form
def factorized_expr (m : ℝ) : ℝ := (m + 4) * (m - 4)

-- State the theorem
theorem factorize_m_square_minus_16 (m : ℝ) : expr m = factorized_expr m :=
by
  sorry

end factorize_m_square_minus_16_l95_95105


namespace ratio_of_x_intercepts_l95_95727

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v : ℝ)
  (hu : u = -b / 5) (hv : v = -b / 3) : u / v = 3 / 5 := by
  sorry

end ratio_of_x_intercepts_l95_95727


namespace soda_cost_l95_95558

-- Definitions of the given conditions
def initial_amount : ℝ := 40
def cost_pizza : ℝ := 2.75
def cost_jeans : ℝ := 11.50
def quarters_left : ℝ := 97
def value_per_quarter : ℝ := 0.25

-- Calculate amount left in dollars
def amount_left : ℝ := quarters_left * value_per_quarter

-- Statement we want to prove: the cost of the soda
theorem soda_cost :
  initial_amount - amount_left - (cost_pizza + cost_jeans) = 1.5 :=
by
  sorry

end soda_cost_l95_95558


namespace four_digit_numbers_count_l95_95277

theorem four_digit_numbers_count :
  ∃ n : ℕ, n = 4140 ∧
  (∀ d1 d2 d3 d4 : ℕ,
    (4 ≤ d1 ∧ d1 ≤ 9) ∧
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d2 * d3 > 8) →
    (∃ m : ℕ, m = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ m > 3999) →
    n = 4140) :=
sorry

end four_digit_numbers_count_l95_95277


namespace tangent_line_eq_l95_95310

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

theorem tangent_line_eq
  (a b : ℝ)
  (h1 : 3 + 2*a + b = 2*a)
  (h2 : 12 + 4*a + b = -b)
  : ∀ x y : ℝ , (f a b 1 = -5/2 ∧
  y - (f a b 1) = -3 * (x - 1))
  → (6*x + 2*y - 1 = 0) :=
by
  sorry

end tangent_line_eq_l95_95310


namespace cos_150_eq_neg_half_l95_95841

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95841


namespace defective_and_shipped_percent_l95_95542

def defective_percent : ℝ := 0.05
def shipped_percent : ℝ := 0.04

theorem defective_and_shipped_percent : (defective_percent * shipped_percent) * 100 = 0.2 :=
by
  sorry

end defective_and_shipped_percent_l95_95542


namespace least_n_div_mod_l95_95715

theorem least_n_div_mod (n : ℕ) (h_pos : n > 1) (h_mod25 : n % 25 = 1) (h_mod7 : n % 7 = 1) : n = 176 :=
by
  sorry

end least_n_div_mod_l95_95715


namespace solve_for_a_l95_95124

theorem solve_for_a (a x : ℝ) (h : x = 1 ∧ 2 * a * x - 2 = a + 3) : a = 5 :=
by
  sorry

end solve_for_a_l95_95124


namespace nine_digit_numbers_divisible_by_eleven_l95_95264

theorem nine_digit_numbers_divisible_by_eleven :
  ∃ (n : ℕ), n = 31680 ∧
    ∃ (num : ℕ), num < 10^9 ∧ num ≥ 10^8 ∧
      (∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ i : ℕ, i ≤ 8 ∧ (num / 10^i) % 10 = d) ∧
      (num % 11 = 0) := 
sorry

end nine_digit_numbers_divisible_by_eleven_l95_95264


namespace isabel_piggy_bank_l95_95919

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end isabel_piggy_bank_l95_95919


namespace f_2017_eq_one_l95_95632

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x - β)

-- Given conditions
variables {a b α β : ℝ}
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0)
variable (h_f2016 : f 2016 a α b β = -1)

-- The goal
theorem f_2017_eq_one : f 2017 a α b β = 1 :=
sorry

end f_2017_eq_one_l95_95632


namespace minimization_problem_l95_95551

theorem minimization_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) (h5 : x ≤ y) (h6 : y ≤ z) (h7 : z ≤ 3 * x) :
  x * y * z ≥ 1 / 18 := 
sorry

end minimization_problem_l95_95551


namespace sixty_five_inv_mod_sixty_six_l95_95509

theorem sixty_five_inv_mod_sixty_six : (65 : ℤ) * 65 ≡ 1 [ZMOD 66] → (65 : ℤ) ≡ 65⁻¹ [ZMOD 66] :=
by
  intro h
  -- Proof goes here
  sorry

end sixty_five_inv_mod_sixty_six_l95_95509


namespace problem1_problem2_l95_95610

-- Problem 1 Statement
theorem problem1 (a : ℝ) (h : a ≠ 1) : (a^2 / (a - 1) - a - 1) = (1 / (a - 1)) :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) : 
  (2 * x * y / (x^2 - y^2)) / ((1 / (x - y)) + (1 / (x + y))) = y :=
by
  sorry

end problem1_problem2_l95_95610


namespace sum_of_angles_l95_95657

theorem sum_of_angles (A B C x y : ℝ) 
  (hA : A = 34) 
  (hB : B = 80) 
  (hC : C = 30)
  (pentagon_angles_sum : A + B + (360 - x) + 90 + (120 - y) = 540) : 
  x + y = 144 :=
by
  sorry

end sum_of_angles_l95_95657


namespace probability_of_no_three_heads_consecutive_l95_95580

-- Definitions based on conditions
def total_sequences : ℕ := 2 ^ 12

def D : ℕ → ℕ
| 1     := 2
| 2     := 4
| 3     := 7
| (n+4) := D (n + 1) + D (n + 2) + D (n + 3)

-- The target probability calculation
def probability_no_three_heads_consecutive : ℚ := D 12 / total_sequences

-- The statement to be proven
theorem probability_of_no_three_heads_consecutive :
  probability_no_three_heads_consecutive = 1705 / 4096 := 
sorry

end probability_of_no_three_heads_consecutive_l95_95580


namespace cos_150_eq_neg_sqrt3_over_2_l95_95815

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95815


namespace variable_swap_l95_95065

theorem variable_swap (x y t : Nat) (h1 : x = 5) (h2 : y = 6) (h3 : t = x) (h4 : x = y) (h5 : y = t) : 
  x = 6 ∧ y = 5 := 
by
  sorry

end variable_swap_l95_95065


namespace interest_problem_l95_95330

theorem interest_problem
  (P : ℝ)
  (h : P * 0.04 * 5 = P * 0.05 * 4) : 
  (P * 0.04 * 5) = 20 := 
by 
  sorry

end interest_problem_l95_95330


namespace trig_expression_equality_l95_95630

theorem trig_expression_equality (α : ℝ) (h : Real.tan α = 1 / 2) : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 :=
by
  sorry

end trig_expression_equality_l95_95630


namespace factor_poly_l95_95708

theorem factor_poly (a b : ℤ) (h : 3*(y^2) - y - 24 = (3*y + a)*(y + b)) : a - b = 11 :=
sorry

end factor_poly_l95_95708


namespace sum_of_possible_values_l95_95572

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 10) = -7) :
  ∃ N1 N2 : ℝ, (N1 * (N1 - 10) = -7 ∧ N2 * (N2 - 10) = -7) ∧ (N1 + N2 = 10) :=
sorry

end sum_of_possible_values_l95_95572


namespace find_a_l95_95112

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end find_a_l95_95112


namespace petya_cannot_win_l95_95041

theorem petya_cannot_win (n : ℕ) (h : n ≥ 3) : ¬ ∃ strategy : ℕ → ℕ → Prop, 
  (∀ k, strategy k (k+1) ∧ strategy k (k-1))
  ∧ ∀ m, ¬ strategy n m :=
sorry

end petya_cannot_win_l95_95041


namespace find_a_l95_95114

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - y = 3) : a = 2 :=
by
  sorry

end find_a_l95_95114


namespace cos_double_angle_l95_95143

theorem cos_double_angle (α : ℝ) (h : Real.tan α = -3) : Real.cos (2 * α) = -4 / 5 := sorry

end cos_double_angle_l95_95143


namespace John_has_15_snakes_l95_95150

theorem John_has_15_snakes (S : ℕ)
  (H1 : ∀ M, M = 2 * S)
  (H2 : ∀ M L, L = M - 5)
  (H3 : ∀ L P, P = L + 8)
  (H4 : ∀ P D, D = P / 3)
  (H5 : S + (2 * S) + ((2 * S) - 5) + (((2 * S) - 5) + 8) + (((((2 * S) - 5) + 8) / 3)) = 114) :
  S = 15 :=
by sorry

end John_has_15_snakes_l95_95150


namespace average_price_per_book_l95_95198

theorem average_price_per_book
  (amount_spent_first_shop : ℕ)
  (amount_spent_second_shop : ℕ)
  (books_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_amount_spent : ℕ := amount_spent_first_shop + amount_spent_second_shop)
  (total_books_bought : ℕ := books_first_shop + books_second_shop)
  (average_price : ℕ := total_amount_spent / total_books_bought) :
  amount_spent_first_shop = 520 → amount_spent_second_shop = 248 →
  books_first_shop = 42 → books_second_shop = 22 →
  average_price = 12 :=
by
  intros
  sorry

end average_price_per_book_l95_95198


namespace length_of_third_side_l95_95015

theorem length_of_third_side (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 12) (h2 : c = 18) (h3 : B = 2 * C) :
  ∃ a, a = 15 :=
by {
  sorry
}

end length_of_third_side_l95_95015


namespace correct_formulas_l95_95130

noncomputable def S (a x : ℝ) := (a^x - a^(-x)) / 2
noncomputable def C (a x : ℝ) := (a^x + a^(-x)) / 2

variable {a x y : ℝ}

axiom h1 : a > 0
axiom h2 : a ≠ 1

theorem correct_formulas : S a (x + y) = S a x * C a y + C a x * S a y ∧ S a (x - y) = S a x * C a y - C a x * S a y :=
by 
  sorry

end correct_formulas_l95_95130


namespace additional_toothpicks_needed_l95_95306

theorem additional_toothpicks_needed 
  (t : ℕ → ℕ)
  (h1 : t 1 = 4)
  (h2 : t 2 = 10)
  (h3 : t 3 = 18)
  (h4 : t 4 = 28)
  (h5 : t 5 = 40)
  (h6 : t 6 = 54) :
  t 6 - t 4 = 26 :=
by
  sorry

end additional_toothpicks_needed_l95_95306


namespace steve_writes_24_pages_per_month_l95_95170

/-- Calculate the number of pages Steve writes in a month given the conditions. -/
theorem steve_writes_24_pages_per_month :
  (∃ (days_in_month : ℕ) (letter_interval : ℕ) (letter_minutes : ℕ) (page_minutes : ℕ) 
      (long_letter_factor : ℕ) (long_letter_minutes : ℕ) (total_pages : ℕ),
    days_in_month = 30 ∧ 
    letter_interval = 3 ∧ 
    letter_minutes = 20 ∧ 
    page_minutes = 10 ∧ 
    long_letter_factor = 2 ∧ 
    long_letter_minutes = 80 ∧ 
    total_pages = 24 ∧ 
    (days_in_month / letter_interval * (letter_minutes / page_minutes)
      + long_letter_minutes / (long_letter_factor * page_minutes) = total_pages)) :=
sorry

end steve_writes_24_pages_per_month_l95_95170


namespace correct_statement_about_K_l95_95707

-- Defining the possible statements about the chemical equilibrium constant K
def K (n : ℕ) : String :=
  match n with
  | 1 => "The larger the K, the smaller the conversion rate of the reactants."
  | 2 => "K is related to the concentration of the reactants."
  | 3 => "K is related to the concentration of the products."
  | 4 => "K is related to temperature."
  | _ => "Invalid statement"

-- Given that the correct answer is that K is related to temperature
theorem correct_statement_about_K : K 4 = "K is related to temperature." :=
by
  rfl

end correct_statement_about_K_l95_95707


namespace ball_arrangement_divisibility_l95_95068

theorem ball_arrangement_divisibility :
  ∀ (n : ℕ), (∀ (i : ℕ), i < n → (∃ j k l m : ℕ, j < k ∧ k < l ∧ l < m ∧ m < n ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m ∧ m ≠ j
    ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m)) →
  ¬((n = 2021) ∨ (n = 2022) ∨ (n = 2023) ∨ (n = 2024)) :=
sorry

end ball_arrangement_divisibility_l95_95068


namespace petya_no_win_implies_draw_or_lost_l95_95040

noncomputable def petya_cannot_win (n : ℕ) (h : n ≥ 3) : Prop :=
  ∀ (Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ),
    ∃ m : ℕ, Petya_strategy m ≠ Vasya_strategy m

theorem petya_no_win_implies_draw_or_lost (n : ℕ) (h : n ≥ 3) :
  ¬ ∃ Petya_strategy Vasya_strategy : ℕ → ℕ → ℕ, 
    (∀ m : ℕ, Petya_strategy m = Vasya_strategy m) :=
by {
  sorry
}

end petya_no_win_implies_draw_or_lost_l95_95040


namespace factorize_expression_l95_95104

theorem factorize_expression (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

end factorize_expression_l95_95104


namespace ceil_sqrt_product_l95_95618

noncomputable def ceil_sqrt_3 : ℕ := ⌈Real.sqrt 3⌉₊
noncomputable def ceil_sqrt_12 : ℕ := ⌈Real.sqrt 12⌉₊
noncomputable def ceil_sqrt_120 : ℕ := ⌈Real.sqrt 120⌉₊

theorem ceil_sqrt_product :
  ceil_sqrt_3 * ceil_sqrt_12 * ceil_sqrt_120 = 88 :=
by
  sorry

end ceil_sqrt_product_l95_95618


namespace range_of_x_l95_95658

theorem range_of_x (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
by {
  sorry
}

end range_of_x_l95_95658


namespace remaining_watermelons_l95_95661

-- Define the given conditions
def initial_watermelons : ℕ := 35
def watermelons_eaten : ℕ := 27

-- Define the question as a theorem
theorem remaining_watermelons : 
  initial_watermelons - watermelons_eaten = 8 :=
by
  sorry

end remaining_watermelons_l95_95661


namespace odd_three_digit_numbers_count_l95_95627

open Finset Nat

def digits := {1, 2, 3, 4, 5}

theorem odd_three_digit_numbers_count : 
  ∃ count : ℕ, 
    count = 36 ∧ 
    ∀ (n : ℕ), 
      (n ∈ digits ∧ n % 2 = 1) →
      (∃ (units : ℕ) (tens : ℕ) (hundreds : ℕ), 
        units ∈ digits ∧ tens ∈ digits ∧ hundreds ∈ digits ∧ 
        units ≠ tens ∧ tens ≠ hundreds ∧ hundreds ≠ units ∧ 
        n = units + 10 * tens + 100 * hundreds) -> 
      count = 36 :=
sorry

end odd_three_digit_numbers_count_l95_95627


namespace find_a_l95_95113

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end find_a_l95_95113


namespace sum_of_coefficients_l95_95135

theorem sum_of_coefficients :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ,
    (1 - 2 * x)^9 = a_9 * x^9 + a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
    a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -2 :=
by
  sorry

end sum_of_coefficients_l95_95135


namespace moles_of_CaCl2_l95_95872

theorem moles_of_CaCl2 (HCl moles_of_HCl : ℕ) (CaCO3 moles_of_CaCO3 : ℕ) 
  (reaction : (CaCO3 = 1) → (HCl = 2) → (moles_of_HCl = 6) → (moles_of_CaCO3 = 3)) :
  ∃ moles_of_CaCl2 : ℕ, moles_of_CaCl2 = 3 :=
by
  sorry

end moles_of_CaCl2_l95_95872


namespace cone_volume_proof_l95_95331

noncomputable def cone_volume (l h : ℕ) : ℝ :=
  let r := Real.sqrt (l^2 - h^2)
  1 / 3 * Real.pi * r^2 * h

theorem cone_volume_proof :
  cone_volume 13 12 = 100 * Real.pi :=
by
  sorry

end cone_volume_proof_l95_95331


namespace strawb_eaten_by_friends_l95_95363

theorem strawb_eaten_by_friends (initial_strawberries remaining_strawberries eaten_strawberries : ℕ) : 
  initial_strawberries = 35 → 
  remaining_strawberries = 33 → 
  eaten_strawberries = initial_strawberries - remaining_strawberries → 
  eaten_strawberries = 2 := 
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end strawb_eaten_by_friends_l95_95363


namespace operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l95_95865

-- Define what an even integer is
def is_even (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * k

-- Define the operations
def add_four (a : ℤ) := a + 4
def subtract_six (a : ℤ) := a - 6
def multiply_by_eight (a : ℤ) := a * 8
def divide_by_two_add_two (a : ℤ) := a / 2 + 2
def average_with_ten (a : ℤ) := (a + 10) / 2

-- The proof statements
theorem operation_1_even_if_input_even (a : ℤ) (h : is_even a) : is_even (add_four a) := sorry
theorem operation_2_even_if_input_even (a : ℤ) (h : is_even a) : is_even (subtract_six a) := sorry
theorem operation_3_even_if_input_even (a : ℤ) (h : is_even a) : is_even (multiply_by_eight a) := sorry
theorem operation_4_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (divide_by_two_add_two a) := sorry
theorem operation_5_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (average_with_ten a) := sorry

end operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l95_95865


namespace johns_umbrellas_in_house_l95_95545

-- Definitions based on the conditions
def umbrella_cost : Nat := 8
def total_amount_paid : Nat := 24
def umbrella_in_car : Nat := 1

-- The goal is to prove that the number of umbrellas in John's house is 2
theorem johns_umbrellas_in_house : 
  (total_amount_paid / umbrella_cost) - umbrella_in_car = 2 :=
by sorry

end johns_umbrellas_in_house_l95_95545


namespace part_a_prob_part_b_expected_time_l95_95989

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l95_95989


namespace jane_total_drawing_paper_l95_95660

theorem jane_total_drawing_paper (brown_sheets : ℕ) (yellow_sheets : ℕ) 
    (h1 : brown_sheets = 28) (h2 : yellow_sheets = 27) : 
    brown_sheets + yellow_sheets = 55 := 
by
    sorry

end jane_total_drawing_paper_l95_95660


namespace percentage_vanilla_orders_l95_95697

theorem percentage_vanilla_orders 
  (V C : ℕ) 
  (h1 : V = 2 * C) 
  (h2 : V + C = 220) 
  (h3 : C = 22) : 
  (V * 100) / 220 = 20 := 
by 
  sorry

end percentage_vanilla_orders_l95_95697


namespace cos_150_eq_neg_sqrt3_div_2_l95_95784

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95784


namespace find_x_minus_y_l95_95119

-- Variables and conditions
variables (x y : ℝ)
def abs_x_eq_3 := abs x = 3
def y_sq_eq_one_fourth := y^2 = 1 / 4
def x_plus_y_neg := x + y < 0

-- Proof problem stating that x - y must equal one of the two possible values
theorem find_x_minus_y (h1 : abs x = 3) (h2 : y^2 = 1 / 4) (h3 : x + y < 0) : 
  x - y = -7 / 2 ∨ x - y = -5 / 2 :=
  sorry

end find_x_minus_y_l95_95119


namespace circle_with_diameter_AB_equation_line_MN_fixed_point_l95_95262

noncomputable def ellipse_Equation (x y : ℝ) : Prop := 
  (y^2 / 12) + (x^2 / 4) = 1

noncomputable def point_A : CLLocationCoordinate2D :=
  ⟨2, 0⟩

noncomputable def point_P : CLLocationCoordinate2D :=
  ⟨0, -2⟩

theorem circle_with_diameter_AB_equation :
  ∃ x y, (point_P (0, -2)) ∧ (point_A (2,0)) ∧  
  (x - 2) * (x + 1) + (y - 0) * (y + 3) = 0 →
  (x^2 + y^2 - x + 3y - 2 = 0) := 
sorry

theorem line_MN_fixed_point :
  ∀ k (x_C y_C x_D y_D : ℝ),
  ellipse_Equation x_C y_C ∧ ellipse_Equation x_D y_D ∧
  (x - (4 * k)/(3 + k^2)) - (y) + 10 = 0 ∨ (x - 3 * y - 10 = 0) →
  (∃ m n, (m = 0) ∧ (n = -10/3)) → 
sorry

end circle_with_diameter_AB_equation_line_MN_fixed_point_l95_95262


namespace laptop_sticker_price_l95_95263

theorem laptop_sticker_price (x : ℝ) (h1 : 0.8 * x - 120 = y) (h2 : 0.7 * x = z) (h3 : y + 25 = z) : x = 950 :=
sorry

end laptop_sticker_price_l95_95263


namespace race_course_length_l95_95595

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : 4 * (d - 69) = d) : d = 92 :=
by
  sorry

end race_course_length_l95_95595


namespace product_sum_divisible_by_1987_l95_95687

theorem product_sum_divisible_by_1987 :
  let A : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 1) (List.range (1987 + 1)))
  let B : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 0) (List.range (1987 + 1)))
  A + B ≡ 0 [MOD 1987] := by
  -- The proof goes here
  sorry

end product_sum_divisible_by_1987_l95_95687


namespace power_calculation_l95_95493

theorem power_calculation : 8^6 * 27^6 * 8^18 * 27^18 = 216^24 := by
  sorry

end power_calculation_l95_95493


namespace quadratic_roots_identity_l95_95123

theorem quadratic_roots_identity (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) (hmn : m * n = -5) (hm_plus_n : m + n = -2) : m^2 + m * n + 2 * m = 0 :=
by {
    sorry
}

end quadratic_roots_identity_l95_95123


namespace min_value_is_neg2032188_l95_95153

noncomputable def min_expression_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) : ℝ :=
(x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016)

theorem min_value_is_neg2032188 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) :
  min_expression_value x y h_pos_x h_pos_y h_neq h_cond = -2032188 := 
sorry

end min_value_is_neg2032188_l95_95153


namespace sharona_bought_more_pencils_l95_95422

-- Define constants for the amounts paid
def amount_paid_jamar : ℚ := 1.43
def amount_paid_sharona : ℚ := 1.87

-- Define the function that computes the number of pencils given the price per pencil and total amount paid
def num_pencils (amount_paid : ℚ) (price_per_pencil : ℚ) : ℚ := amount_paid / price_per_pencil

-- Define the theorem stating that Sharona bought 4 more pencils than Jamar
theorem sharona_bought_more_pencils {price_per_pencil : ℚ} (h_price : price_per_pencil > 0) :
  num_pencils amount_paid_sharona price_per_pencil = num_pencils amount_paid_jamar price_per_pencil + 4 :=
sorry

end sharona_bought_more_pencils_l95_95422


namespace prime_has_property_p_l95_95876

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end prime_has_property_p_l95_95876


namespace square_side_length_l95_95489

theorem square_side_length 
  (AF DH BG AE : ℝ) 
  (AF_eq : AF = 7) 
  (DH_eq : DH = 4) 
  (BG_eq : BG = 5) 
  (AE_eq : AE = 1) 
  (area_EFGH : ℝ) 
  (area_EFGH_eq : area_EFGH = 78) : 
  (∃ s : ℝ, s^2 = 144) :=
by
  use 12
  sorry

end square_side_length_l95_95489


namespace johns_train_speed_l95_95307

noncomputable def average_speed_of_train (D : ℝ) (V_t : ℝ) : ℝ := D / (0.8 * D / V_t + 0.2 * D / 20)

theorem johns_train_speed (D : ℝ) (V_t : ℝ) (h1 : average_speed_of_train D V_t = 50) : V_t = 64 :=
by
  sorry

end johns_train_speed_l95_95307


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95995

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95995


namespace decimal_to_base5_equiv_l95_95096

def base5_representation (n : ℕ) : ℕ := -- Conversion function (implementation to be filled later)
  sorry

theorem decimal_to_base5_equiv : base5_representation 88 = 323 :=
by
  -- Proof steps go here.
  sorry

end decimal_to_base5_equiv_l95_95096


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95860

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95860


namespace nancy_yearly_payment_l95_95159

open Real

-- Define the monthly cost of the car insurance
def monthly_cost : ℝ := 80

-- Nancy's percentage contribution
def percentage : ℝ := 0.40

-- Calculate the monthly payment Nancy will make
def monthly_payment : ℝ := percentage * monthly_cost

-- Calculate the yearly payment Nancy will make
def yearly_payment : ℝ := 12 * monthly_payment

-- State the proof problem
theorem nancy_yearly_payment : yearly_payment = 384 :=
by
  -- Proof goes here
  sorry

end nancy_yearly_payment_l95_95159


namespace cos_150_eq_neg_half_l95_95844

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95844


namespace fraction_value_l95_95523

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end fraction_value_l95_95523


namespace sum_mean_median_mode_l95_95975

theorem sum_mean_median_mode : 
  let data := [2, 5, 1, 5, 2, 6, 1, 5, 0, 2]
  let ordered_data := [0, 1, 1, 2, 2, 2, 5, 5, 5, 6]
  let mean := (0 + 1 + 1 + 2 + 2 + 2 + 5 + 5 + 5 + 6) / 10
  let median := (2 + 2) / 2
  let mode := 5
  mean + median + mode = 9.9 := by
  sorry

end sum_mean_median_mode_l95_95975


namespace combined_profit_is_14000_l95_95629

-- Define constants and conditions
def center1_daily_packages : ℕ := 10000
def daily_profit_per_package : ℝ := 0.05
def center2_multiplier : ℕ := 3
def days_per_week : ℕ := 7

-- Define the profit for the first center
def center1_daily_profit : ℝ := center1_daily_packages * daily_profit_per_package

-- Define the packages processed by the second center
def center2_daily_packages : ℕ := center1_daily_packages * center2_multiplier

-- Define the profit for the second center
def center2_daily_profit : ℝ := center2_daily_packages * daily_profit_per_package

-- Define the combined daily profit
def combined_daily_profit : ℝ := center1_daily_profit + center2_daily_profit

-- Define the combined weekly profit
def combined_weekly_profit : ℝ := combined_daily_profit * days_per_week

-- Prove that the combined weekly profit is $14,000
theorem combined_profit_is_14000 : combined_weekly_profit = 14000 := by
  -- You can replace sorry with the steps to solve the proof.
  sorry

end combined_profit_is_14000_l95_95629


namespace ming_belief_contradiction_l95_95744

theorem ming_belief_contradiction
  (A B C : Type)
  [Plane A B C]
  (angle_B angle_C : Angle)
  (side_AC side_AB : Length) :
  (angle_B ≠ angle_C) → (AC = AB) → False :=
begin
  sorry
end

end ming_belief_contradiction_l95_95744


namespace factorize_quadratic_l95_95505

theorem factorize_quadratic (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 :=
by
  sorry

end factorize_quadratic_l95_95505


namespace other_person_time_to_complete_job_l95_95173

-- Define the conditions
def SureshTime : ℕ := 15
def SureshWorkHours : ℕ := 9
def OtherPersonWorkHours : ℕ := 4

-- The proof problem: Prove that the other person can complete the job in 10 hours.
theorem other_person_time_to_complete_job (x : ℕ) 
  (h1 : ∀ SureshWorkHours SureshTime, SureshWorkHours * (1 / SureshTime) = (SureshWorkHours / SureshTime) ∧ 
       4 * (SureshWorkHours / SureshTime / 4) = 1) : 
  (x = 10) :=
sorry

end other_person_time_to_complete_job_l95_95173


namespace cos_150_eq_neg_half_l95_95810

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95810


namespace option_c_is_incorrect_l95_95599

/-- Define the temperature data -/
def temps : List Int := [-20, -10, 0, 10, 20, 30]

/-- Define the speed of sound data corresponding to the temperatures -/
def speeds : List Int := [318, 324, 330, 336, 342, 348]

/-- The speed of sound at 10 degrees Celsius -/
def speed_at_10 : Int := 336

/-- The incorrect claim in option C -/
def incorrect_claim : Prop := (speed_at_10 * 4 ≠ 1334)

/-- Prove that the claim in option C is incorrect -/
theorem option_c_is_incorrect : incorrect_claim :=
by {
  sorry
}

end option_c_is_incorrect_l95_95599


namespace totalPeaches_l95_95333

-- Definitions based on the given conditions
def redPeaches : Nat := 13
def greenPeaches : Nat := 3

-- Problem statement
theorem totalPeaches : redPeaches + greenPeaches = 16 := by
  sorry

end totalPeaches_l95_95333


namespace min_xyz_value_l95_95673

theorem min_xyz_value (x y z : ℝ) (h1 : x + y + z = 1) (h2 : z = 2 * y) (h3 : y ≤ (1 / 3)) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (∀ a b c : ℝ, (a + b + c = 1) → (c = 2 * b) → (b ≤ (1 / 3)) → 0 < a → 0 < b → 0 < c → (a * b * c) ≥ (x * y * z) → (a * b * c) = (8 / 243)) :=
by sorry

end min_xyz_value_l95_95673


namespace number_of_four_digit_numbers_l95_95271

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l95_95271


namespace roots_equation_l95_95550

theorem roots_equation (m n : ℝ) (h1 : ∀ x, (x - m) * (x - n) = x^2 + 2 * x - 2025) : m^2 + 3 * m + n = 2023 :=
by
  sorry

end roots_equation_l95_95550


namespace first_proof_l95_95214

def triangular (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def covers_all_columns (k : ℕ) : Prop :=
  ∀ c : ℕ, (c < 10) → (∃ m : ℕ, m ≤ k ∧ (triangular m) % 10 = c)

theorem first_proof (k : ℕ) (h : covers_all_columns 28) : 
  triangular k = 435 :=
sorry

end first_proof_l95_95214


namespace sugar_used_in_two_minutes_l95_95186

-- Define constants for problem conditions
def sugar_per_bar : Float := 1.5
def bars_per_minute : Nat := 36
def minutes : Nat := 2

-- Define the total sugar used in two minutes
def total_sugar_used : Float := (bars_per_minute * sugar_per_bar) * minutes

-- State the theorem and its proof
theorem sugar_used_in_two_minutes : total_sugar_used = 108 := by
  sorry

end sugar_used_in_two_minutes_l95_95186


namespace tickets_difference_l95_95490

def tickets_used_for_clothes : ℝ := 85
def tickets_used_for_accessories : ℝ := 45.5
def tickets_used_for_food : ℝ := 51
def tickets_used_for_toys : ℝ := 58

theorem tickets_difference : 
  (tickets_used_for_clothes + tickets_used_for_food + tickets_used_for_accessories) - tickets_used_for_toys = 123.5 := 
by
  sorry

end tickets_difference_l95_95490


namespace all_have_perp_property_l95_95891

def M₁ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, x^3 - 2 * x^2 + 3)}
def M₂ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.log (2 - x) / Real.log 2)}
def M₃ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 2 - 2^x)}
def M₄ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 1 - Real.sin x)}

def perp_property (M : Set (ℝ × ℝ)) : Prop :=
∀ p ∈ M, ∃ q ∈ M, p.1 * q.1 + p.2 * q.2 = 0

-- Theorem statement
theorem all_have_perp_property :
  perp_property M₁ ∧ perp_property M₂ ∧ perp_property M₃ ∧ perp_property M₄ :=
sorry

end all_have_perp_property_l95_95891


namespace Kiera_envelopes_l95_95429

theorem Kiera_envelopes (blue yellow green : ℕ) (total_envelopes : ℕ) 
  (cond1 : blue = 14) 
  (cond2 : total_envelopes = 46) 
  (cond3 : green = 3 * yellow) 
  (cond4 : total_envelopes = blue + yellow + green) : yellow = 6 - 8 := 
by sorry

end Kiera_envelopes_l95_95429


namespace cos_150_eq_neg_sqrt3_div_2_l95_95803

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95803


namespace temperature_reaches_100_at_5_hours_past_noon_l95_95289

theorem temperature_reaches_100_at_5_hours_past_noon :
  ∃ t : ℝ, (-2 * t^2 + 16 * t + 40 = 100) ∧ ∀ t' : ℝ, (-2 * t'^2 + 16 * t' + 40 = 100) → 5 ≤ t' :=
by
  -- We skip the proof and assume the theorem is true.
  sorry

end temperature_reaches_100_at_5_hours_past_noon_l95_95289


namespace cube_root_of_5_irrational_l95_95588

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end cube_root_of_5_irrational_l95_95588


namespace merck_hourly_rate_l95_95664

-- Define the relevant data from the problem
def hours_donaldsons : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def total_earnings : ℕ := 273

-- Define the total hours based on the conditions
def total_hours : ℕ := hours_donaldsons + hours_merck + hours_hille

-- Define what we want to prove:
def hourly_rate := total_earnings / total_hours

theorem merck_hourly_rate : hourly_rate = 273 / (7 + 6 + 3) := by
  sorry

end merck_hourly_rate_l95_95664


namespace ten_term_sequence_l95_95750
open Real

theorem ten_term_sequence (a b : ℝ) 
    (h₁ : a + b = 1)
    (h₂ : a^2 + b^2 = 3)
    (h₃ : a^3 + b^3 = 4)
    (h₄ : a^4 + b^4 = 7)
    (h₅ : a^5 + b^5 = 11) :
    a^10 + b^10 = 123 :=
  sorry

end ten_term_sequence_l95_95750


namespace valid_four_digit_numbers_count_l95_95267

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l95_95267


namespace ratio_Florence_Rene_l95_95936

theorem ratio_Florence_Rene :
  ∀ (I F R : ℕ), R = 300 → F = k * R → I = 1/3 * (F + R + I) → F + R + I = 1650 → F / R = 3 / 2 := 
by 
  sorry

end ratio_Florence_Rene_l95_95936


namespace oxygen_atom_diameter_in_scientific_notation_l95_95950

theorem oxygen_atom_diameter_in_scientific_notation :
  0.000000000148 = 1.48 * 10^(-10) :=
sorry

end oxygen_atom_diameter_in_scientific_notation_l95_95950


namespace cos_150_eq_neg_sqrt3_div_2_l95_95771

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95771


namespace total_shirts_l95_95563

def initial_shirts : ℕ := 9
def new_shirts : ℕ := 8

theorem total_shirts : initial_shirts + new_shirts = 17 := by
  sorry

end total_shirts_l95_95563


namespace paper_boat_time_proof_l95_95917

/-- A 50-meter long embankment exists along a river.
 - A motorboat that passes this embankment in 5 seconds while moving downstream.
 - The same motorboat passes this embankment in 4 seconds while moving upstream.
 - Determine the time in seconds it takes for a paper boat, which moves with the current, to travel the length of this embankment.
 -/
noncomputable def paper_boat_travel_time 
  (embankment_length : ℝ)
  (motorboat_length : ℝ)
  (time_downstream : ℝ)
  (time_upstream : ℝ) : ℝ :=
  let v_eff_downstream := embankment_length / time_downstream,
      v_eff_upstream := embankment_length / time_upstream,
      v_boat := (v_eff_downstream + v_eff_upstream) / 2,
      v_current := (v_eff_downstream - v_eff_upstream) / 2 in
  embankment_length / v_current

theorem paper_boat_time_proof :
  paper_boat_travel_time 50 10 5 4 = 40 := 
begin
  sorry,
end

end paper_boat_time_proof_l95_95917


namespace probability_correct_l95_95197

noncomputable def probability_of_getting_number_greater_than_4 : ℚ :=
  let favorable_outcomes := 2
  let total_outcomes := 6
  favorable_outcomes / total_outcomes

theorem probability_correct :
  probability_of_getting_number_greater_than_4 = 1 / 3 := by sorry

end probability_correct_l95_95197


namespace my_op_example_l95_95374

def my_op (a b : Int) : Int := a^2 - abs b

theorem my_op_example : my_op (-2) (-1) = 3 := by
  sorry

end my_op_example_l95_95374


namespace walnut_trees_in_park_l95_95720

def num_current_walnut_trees (num_plant : ℕ) (num_total : ℕ) : ℕ :=
  num_total - num_plant

theorem walnut_trees_in_park :
  num_current_walnut_trees 6 10 = 4 :=
by
  -- By the definition of num_current_walnut_trees
  -- We have 10 (total) - 6 (to be planted) = 4 (current)
  sorry

end walnut_trees_in_park_l95_95720


namespace cos_150_degree_l95_95830

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95830


namespace factory_Y_bulbs_proportion_l95_95378

theorem factory_Y_bulbs_proportion :
  (0.60 * 0.59 + 0.40 * P_Y = 0.62) → (P_Y = 0.665) :=
by
  sorry

end factory_Y_bulbs_proportion_l95_95378


namespace circle_center_and_radius_l95_95951

theorem circle_center_and_radius (x y : ℝ) : 
  (x^2 + y^2 - 6 * x = 0) → ((x - 3)^2 + (y - 0)^2 = 9) :=
by
  intro h
  -- The proof is left as an exercise.
  sorry

end circle_center_and_radius_l95_95951


namespace solutions_diff_squared_l95_95928

theorem solutions_diff_squared (a b : ℝ) (h : 5 * a^2 - 6 * a - 55 = 0 ∧ 5 * b^2 - 6 * b - 55 = 0) :
  (a - b)^2 = 1296 / 25 := by
  sorry

end solutions_diff_squared_l95_95928


namespace construct_quad_root_of_sums_l95_95564

theorem construct_quad_root_of_sums (a b : ℝ) : ∃ c : ℝ, c = (a^4 + b^4)^(1/4) := 
by
  sorry

end construct_quad_root_of_sums_l95_95564


namespace cos_150_eq_neg_half_l95_95806

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l95_95806


namespace Thelma_cuts_each_tomato_into_8_slices_l95_95048

-- Conditions given in the problem
def slices_per_meal := 20
def family_size := 8
def tomatoes_needed := 20

-- The quantity we want to prove
def slices_per_tomato := 8

-- Statement to be proven: Thelma cuts each green tomato into the correct number of slices
theorem Thelma_cuts_each_tomato_into_8_slices :
  (slices_per_meal * family_size) = (tomatoes_needed * slices_per_tomato) :=
by 
  sorry

end Thelma_cuts_each_tomato_into_8_slices_l95_95048


namespace arith_seq_geom_seq_l95_95527

theorem arith_seq_geom_seq (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : (a 9)^2 = a 5 * a 15) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end arith_seq_geom_seq_l95_95527


namespace remainder_of_n_div_7_l95_95515

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end remainder_of_n_div_7_l95_95515


namespace gcd_204_85_l95_95031

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := 
by sorry

end gcd_204_85_l95_95031


namespace gcd_160_200_360_l95_95582

theorem gcd_160_200_360 : Nat.gcd (Nat.gcd 160 200) 360 = 40 := by
  sorry

end gcd_160_200_360_l95_95582


namespace area_of_triangle_ABC_l95_95290

theorem area_of_triangle_ABC
  {A B C : Type*} 
  (AC BC : ℝ)
  (B : ℝ)
  (h1 : AC = Real.sqrt (13))
  (h2 : BC = 1)
  (h3 : B = Real.sqrt 3 / 2): 
  ∃ area : ℝ, area = Real.sqrt 3 := 
sorry

end area_of_triangle_ABC_l95_95290


namespace ones_digit_of_sum_of_powers_l95_95193

theorem ones_digit_of_sum_of_powers :
  (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 :=
by
  sorry

end ones_digit_of_sum_of_powers_l95_95193


namespace chicken_feathers_after_crossing_l95_95455

def feathers_remaining_after_crossings (cars_dodged feathers_before pulling_factor : ℕ) : ℕ :=
  let feathers_lost := cars_dodged * pulling_factor
  feathers_before - feathers_lost

theorem chicken_feathers_after_crossing 
  (cars_dodged : ℕ := 23)
  (feathers_before : ℕ := 5263)
  (pulling_factor : ℕ := 2) :
  feathers_remaining_after_crossings cars_dodged feathers_before pulling_factor = 5217 :=
by
  sorry

end chicken_feathers_after_crossing_l95_95455


namespace cubic_as_diff_of_squares_l95_95249

theorem cubic_as_diff_of_squares (n : ℕ) (h : n > 1) :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := 
sorry

end cubic_as_diff_of_squares_l95_95249


namespace probability_four_distinct_numbers_l95_95335

theorem probability_four_distinct_numbers (prob : ℚ) (h : prob = 325 / 648) :
  let total_outcomes := 6 ^ 6 in
  let case1_outcomes := 6 * 20 * 10 * 6 in
  let case2_outcomes := 15 * 15 * 6 * 6 * 2 in
  prob = (case1_outcomes + case2_outcomes) / total_outcomes := by
  sorry

end probability_four_distinct_numbers_l95_95335


namespace probability_divisible_by_8_l95_95972

-- Define the problem conditions
def is_8_sided_die (n : ℕ) : Prop := n = 6
def roll_dice (m : ℕ) : Prop := m = 8

-- Define the main proof statement
theorem probability_divisible_by_8 (n m : ℕ) (hn : is_8_sided_die n) (hm : roll_dice m) :  
  (35 : ℚ) / 36 = 
  (1 - ((1/2) ^ m + 28 * ((1/n) ^ 2 * ((1/2) ^ 6))) : ℚ) :=
by
  sorry

end probability_divisible_by_8_l95_95972


namespace phone_number_fraction_l95_95768

theorem phone_number_fraction : 
  let total_valid_numbers := 6 * (10^6)
  let valid_numbers_with_conditions := 10^5
  valid_numbers_with_conditions / total_valid_numbers = 1 / 60 :=
by sorry

end phone_number_fraction_l95_95768


namespace sum_of_series_l95_95373

def series_sum : ℕ := 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))

theorem sum_of_series : series_sum = 2730 := by
  -- Expansion: 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))) = 2 + 2 * 4 + 2 * 4^2 + 2 * 4^3 + 2 * 4^4 + 2 * 4^5 
  -- Geometric series sum formula application: S = 2 + 2*4 + 2*4^2 + 2*4^3 + 2*4^4 + 2*4^5 = 2730
  sorry

end sum_of_series_l95_95373


namespace average_score_girls_l95_95654

theorem average_score_girls (num_boys num_girls : ℕ) (avg_boys avg_class : ℕ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  avg_boys = 84 → 
  avg_class = 86 → 
  ∃ avg_girls : ℕ, avg_girls = 92 :=
by
  intros h1 h2 h3 h4
  sorry

end average_score_girls_l95_95654


namespace max_value_t_min_value_y_l95_95637

-- 1. Prove that the maximum value of t given |2x+5| + |2x-1| - t ≥ 0 is s = 6.
theorem max_value_t (t : ℝ) (x : ℝ) :
  (abs (2*x + 5) + abs (2*x - 1) - t ≥ 0) → (t ≤ 6) :=
by sorry

-- 2. Given s = 6 and 4a + 5b = s, prove that the minimum value of y = 1/(a+2b) + 4/(3a+3b) is y = 3/2.
theorem min_value_y (a b : ℝ) (s : ℝ) :
  s = 6 → (4*a + 5*b = s) → (a > 0) → (b > 0) → 
  (1/(a + 2*b) + 4/(3*a + 3*b) ≥ 3/2) :=
by sorry

end max_value_t_min_value_y_l95_95637


namespace point_in_fourth_quadrant_l95_95912

-- Definitions of the quadrants as provided in the conditions
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Given point
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : fourth_quadrant point.fst point.snd :=
sorry

end point_in_fourth_quadrant_l95_95912


namespace value_of_x_squared_plus_9y_squared_l95_95901

theorem value_of_x_squared_plus_9y_squared (x y : ℝ)
  (h1 : x + 3 * y = 5)
  (h2 : x * y = -8) : x^2 + 9 * y^2 = 73 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l95_95901


namespace plane_equation_l95_95239

-- We will create a structure for 3D points to use in our problem
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the problem conditions and the equation we want to prove
def containsPoint (p: Point3D) : Prop := p.x = 1 ∧ p.y = 4 ∧ p.z = -8

def onLine (p: Point3D) : Prop := 
  ∃ t : ℝ, 
    (p.x = 4 * t + 2) ∧ 
    (p.y = - t - 1) ∧ 
    (p.z = 5 * t + 3)

def planeEq (p: Point3D) : Prop := 
  -4 * p.x + 2 * p.y - 5 * p.z + 3 = 0

-- Now state the theorem
theorem plane_equation (p: Point3D) : 
  containsPoint p ∨ onLine p → planeEq p := 
  sorry

end plane_equation_l95_95239


namespace georgie_initial_avocados_l95_95247

-- Define the conditions
def avocados_needed_per_serving := 3
def servings_made := 3
def avocados_bought_by_sister := 4
def total_avocados_needed := avocados_needed_per_serving * servings_made

-- The statement to prove
theorem georgie_initial_avocados : (total_avocados_needed - avocados_bought_by_sister) = 5 :=
sorry

end georgie_initial_avocados_l95_95247


namespace number_of_cds_l95_95926

-- Define the constants
def total_money : ℕ := 37
def cd_price : ℕ := 14
def cassette_price : ℕ := 9

theorem number_of_cds (total_money cd_price cassette_price : ℕ) (h_total_money : total_money = 37) (h_cd_price : cd_price = 14) (h_cassette_price : cassette_price = 9) :
  ∃ n : ℕ, n * cd_price + cassette_price = total_money ∧ n = 2 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_cds_l95_95926


namespace nine_digit_divisible_by_11_l95_95139

theorem nine_digit_divisible_by_11 (m : ℕ) (k : ℤ) (h1 : 8 + 4 + m + 6 + 8 = 26 + m)
(h2 : 5 + 2 + 7 + 1 = 15)
(h3 : 26 + m - 15 = 11 + m)
(h4 : 11 + m = 11 * k) :
m = 0 := by
  sorry

end nine_digit_divisible_by_11_l95_95139


namespace value_of_polynomial_at_2_l95_95882

def f (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3 * x^3 - 2 * x^2 - 2500 * x + 434

theorem value_of_polynomial_at_2 : f 2 = -3390 := by
  -- proof would go here
  sorry

end value_of_polynomial_at_2_l95_95882


namespace cos_150_eq_neg_sqrt3_div_2_l95_95854

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95854


namespace number_of_siblings_l95_95617

-- Definitions for the given conditions
def total_height : ℕ := 330
def sibling1_height : ℕ := 66
def sibling2_height : ℕ := 66
def sibling3_height : ℕ := 60
def last_sibling_height : ℕ := 70  -- Derived from the solution steps
def eliza_height : ℕ := last_sibling_height - 2

-- The final question to validate
theorem number_of_siblings (h : 2 * sibling1_height + sibling3_height + last_sibling_height + eliza_height = total_height) :
  4 = 4 :=
by {
  -- Condition h states that the total height is satisfied
  -- Therefore, it directly justifies our claim without further computation here.
  sorry
}

end number_of_siblings_l95_95617


namespace cos_150_eq_negative_cos_30_l95_95788

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95788


namespace triangle_inequality_lt_l95_95152

theorem triangle_inequality_lt {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a < b + c) (h2 : b < a + c) (h3 : c < a + b) : a^2 + b^2 + c^2 < 2 * (a*b + b*c + c*a) := 
sorry

end triangle_inequality_lt_l95_95152


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95997

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l95_95997


namespace sum_of_100th_row_l95_95095

def triangularArraySum (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2^(n+1) - 3*n

theorem sum_of_100th_row :
  triangularArraySum 100 = 2^100 - 297 :=
by
  sorry

end sum_of_100th_row_l95_95095


namespace jackson_miles_l95_95085

theorem jackson_miles (beka_miles jackson_miles : ℕ) (h1 : beka_miles = 873) (h2 : beka_miles = jackson_miles + 310) : jackson_miles = 563 := by
  sorry

end jackson_miles_l95_95085


namespace complementary_angle_l95_95391

-- Define the complementary angle condition
def complement (angle : ℚ) := 90 - angle

theorem complementary_angle : complement 30.467 = 59.533 :=
by
  -- Adding sorry to signify the missing proof to ensure Lean builds successfully
  sorry

end complementary_angle_l95_95391


namespace three_inv_mod_199_l95_95106

theorem three_inv_mod_199 : ∃ x : ℤ, 3 * x ≡ 1 [MOD 199] ∧ (0 ≤ x ∧ x < 199) :=
by
  use 133
  split
  · show 3 * 133 ≡ 1 [MOD 199]
    sorry
  · split
    · show 0 ≤ 133
      linarith
    · show 133 < 199
      linarith

end three_inv_mod_199_l95_95106


namespace mans_speed_against_current_l95_95745

/-- Given the man's speed with the current and the speed of the current, prove the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ) (speed_of_current : ℝ)
  (h1 : speed_with_current = 16)
  (h2 : speed_of_current = 3.2) :
  speed_with_current - 2 * speed_of_current = 9.6 :=
sorry

end mans_speed_against_current_l95_95745


namespace inequality_solution_l95_95238

theorem inequality_solution (x : ℝ) :
  (3 / 16) + abs (x - 17 / 64) < 7 / 32 ↔ (15 / 64) < x ∧ x < (19 / 64) :=
by
  sorry

end inequality_solution_l95_95238


namespace current_short_trees_l95_95334

-- Definitions of conditions in a)
def tall_trees : ℕ := 44
def short_trees_planted : ℕ := 57
def total_short_trees_after_planting : ℕ := 98

-- Statement to prove the question == answer given conditions
theorem current_short_trees (S : ℕ) (h : S + short_trees_planted = total_short_trees_after_planting) : S = 41 :=
by
  -- Proof would go here
  sorry

end current_short_trees_l95_95334


namespace amount_left_after_spending_l95_95920

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end amount_left_after_spending_l95_95920


namespace max_blocks_fit_l95_95052

def block_dimensions : ℝ × ℝ × ℝ := (3, 1, 1)
def box_dimensions : ℝ × ℝ × ℝ := (4, 3, 2)

theorem max_blocks_fit :
  let (block_l, block_w, block_h) := block_dimensions in
  let (box_l, box_w, box_h) := box_dimensions in
  (block_l ≤ box_l ∧ block_w ≤ box_w ∧ block_h ≤ box_h) →
  block_l * block_w * block_h > 0 →
  ∃ k : ℕ, k = 6 :=
by
  sorry

end max_blocks_fit_l95_95052


namespace limit_of_sequence_l95_95225

open Real

noncomputable def sequence (n : ℕ) : ℝ :=
  (2 * n - real.sin n) / (sqrt n - (n ^ 3 - 7) ^ (1 / 3))

theorem limit_of_sequence : 
  tendsto (λ n : ℕ, (2 * (n : ℝ) - sin (n : ℝ)) / (sqrt n - cbrt (↑n^3 - 7))) atTop (𝓝 (-2)) :=
by
  sorry

end limit_of_sequence_l95_95225


namespace twelfth_term_arithmetic_sequence_l95_95458

theorem twelfth_term_arithmetic_sequence (a d : ℤ) (h1 : a + 2 * d = 13) (h2 : a + 6 * d = 25) : a + 11 * d = 40 := 
sorry

end twelfth_term_arithmetic_sequence_l95_95458


namespace probability_no_more_than_10_seconds_l95_95221

noncomputable def total_cycle_time : ℕ := 80
noncomputable def green_time : ℕ := 30
noncomputable def yellow_time : ℕ := 10
noncomputable def red_time : ℕ := 40
noncomputable def can_proceed : ℕ := green_time + yellow_time + yellow_time

theorem probability_no_more_than_10_seconds : 
  can_proceed / total_cycle_time = 5 / 8 := 
  sorry

end probability_no_more_than_10_seconds_l95_95221


namespace prove_expression_value_l95_95415

theorem prove_expression_value (m n : ℝ) (h : m^2 + 3 * n - 1 = 2) : 2 * m^2 + 6 * n + 1 = 7 := by
  sorry

end prove_expression_value_l95_95415


namespace cos_150_eq_neg_sqrt3_over_2_l95_95793

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95793


namespace find_breadth_of_rectangle_l95_95036

theorem find_breadth_of_rectangle
  (L R S : ℝ)
  (h1 : L = (2 / 5) * R)
  (h2 : R = S)
  (h3 : S ^ 2 = 625)
  (A : ℝ := 100)
  (h4 : A = L * B) :
  B = 10 := sorry

end find_breadth_of_rectangle_l95_95036


namespace student_ticket_cost_l95_95970

theorem student_ticket_cost :
  ∀ (S : ℤ),
  (525 - 388) * S + 388 * 6 = 2876 → S = 4 :=
by
  sorry

end student_ticket_cost_l95_95970


namespace find_n_l95_95007

theorem find_n (n : ℕ) (M : ℕ) (A : ℕ) 
  (hM : M = n - 11) 
  (hA : A = n - 2) 
  (hM_ge_one : M ≥ 1) 
  (hA_ge_one : A ≥ 1) 
  (hM_plus_A_lt_n : M + A < n) : 
  n = 12 := 
by 
  sorry

end find_n_l95_95007


namespace remainder_of_powers_l95_95339

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l95_95339


namespace sum_ratio_15_l95_95642

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequences
def sum_a (n : ℕ) := S n
def sum_b (n : ℕ) := T n

-- The ratio condition
def ratio_condition := ∀ n, a n * (n + 1) = b n * (3 * n + 21)

theorem sum_ratio_15
  (ha : sum_a 15 = 15 * a 8)
  (hb : sum_b 15 = 15 * b 8)
  (h_ratio : ratio_condition a b) :
  sum_a 15 / sum_b 15 = 5 :=
sorry

end sum_ratio_15_l95_95642


namespace cos_150_eq_neg_sqrt3_div_2_l95_95782

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95782


namespace correct_statements_l95_95251

open Classical

variables {α l m n p : Type*}
variables (is_perpendicular_to : α → α → Prop) (is_parallel_to : α → α → Prop)
variables (is_in_plane : α → α → Prop)

noncomputable def problem_statement (l : α) (α : α) : Prop :=
  (∀ m, is_perpendicular_to m l → is_parallel_to m α) ∧
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α)

theorem correct_statements (l : α) (α : α) (h_l_α : is_perpendicular_to l α) :
  (∀ m, is_perpendicular_to m α → is_parallel_to m l) ∧
  (∀ m, is_parallel_to m α → is_perpendicular_to m l) ∧
  (∀ m, is_parallel_to m l → is_perpendicular_to m α) :=
sorry

end correct_statements_l95_95251


namespace matrix_vector_combination_l95_95549

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : α →ₗ[ℝ] ℝ × ℝ)
variables (u v w : α)
variables (h1 : M u = (-3, 4))
variables (h2 : M v = (2, -7))
variables (h3 : M w = (9, 0))

theorem matrix_vector_combination :
  M (3 • u - 4 • v + 2 • w) = (1, 40) :=
by sorry

end matrix_vector_combination_l95_95549


namespace largest_n_for_factorable_polynomial_l95_95243

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end largest_n_for_factorable_polynomial_l95_95243


namespace fx_root_and_decreasing_l95_95929

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - Real.log x / Real.log 2

theorem fx_root_and_decreasing (a x0 : ℝ) (h0 : 0 < a) (hx0 : 0 < x0) (h_cond : a < x0) (hf_root : f x0 = 0) 
  (hf_decreasing : ∀ x y : ℝ, x < y → f y < f x) : f a > 0 := 
sorry

end fx_root_and_decreasing_l95_95929


namespace range_of_a_l95_95548

-- Definition of sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B (a : ℝ) : Set ℝ := { x | x < a }

-- Condition of the union of A and B
theorem range_of_a (a : ℝ) : (A ∪ B a = { x | x < 1 }) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l95_95548


namespace number_of_people_speaking_both_languages_l95_95979

theorem number_of_people_speaking_both_languages
  (total : ℕ) (L : ℕ) (F : ℕ) (N : ℕ) (B : ℕ) :
  total = 25 → L = 13 → F = 15 → N = 6 → total = L + F - B + N → B = 9 :=
by
  intros h_total h_L h_F h_N h_inclusion_exclusion
  sorry

end number_of_people_speaking_both_languages_l95_95979


namespace inequality_for_distinct_integers_l95_95003

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

end inequality_for_distinct_integers_l95_95003


namespace dogs_in_kennel_l95_95980

variable (C D : ℕ)

-- definition of the ratio condition 
def ratio_condition : Prop :=
  C * 4 = 3 * D

-- definition of the difference condition
def difference_condition : Prop :=
  C = D - 8

theorem dogs_in_kennel (h1 : ratio_condition C D) (h2 : difference_condition C D) : D = 32 :=
by 
  -- proof steps go here
  sorry

end dogs_in_kennel_l95_95980


namespace distance_with_tide_60_min_l95_95482

variable (v_m v_t : ℝ)

axiom man_with_tide : (v_m + v_t) = 5
axiom man_against_tide : (v_m - v_t) = 4

theorem distance_with_tide_60_min : (v_m + v_t) = 5 := by
  sorry

end distance_with_tide_60_min_l95_95482


namespace minimum_value_ineq_l95_95434

noncomputable def minimum_value (x y z : ℝ) := x^2 + 4 * x * y + 4 * y^2 + 4 * z^2

theorem minimum_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 64) : minimum_value x y z ≥ 192 :=
by {
  sorry
}

end minimum_value_ineq_l95_95434


namespace cos_150_eq_neg_sqrt3_div_2_l95_95849

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95849


namespace partition_cities_l95_95719

-- Define the type for cities and airlines.
variable (City : Type) (Airline : Type)

-- Define the number of cities and airlines
variable (n k : ℕ)

-- Define a relation to represent bidirectional direct flights
variable (flight : Airline → City → City → Prop)

-- Define the condition: Some pairs of cities are connected by exactly one direct flight operated by one of the airline companies
-- or there are no such flights between them.
axiom unique_flight : ∀ (a : Airline) (c1 c2 : City), flight a c1 c2 → ¬ (∃ (a' : Airline), flight a' c1 c2 ∧ a' ≠ a)

-- Define the condition: Any two direct flights operated by the same company share a common endpoint
axiom shared_endpoint :
  ∀ (a : Airline) (c1 c2 c3 c4 : City), flight a c1 c2 → flight a c3 c4 → (c1 = c3 ∨ c1 = c4 ∨ c2 = c3 ∨ c2 = c4)

-- The main theorem to prove
theorem partition_cities :
  ∃ (partition : City → Fin (k + 2)), ∀ (c1 c2 : City) (a : Airline), flight a c1 c2 → partition c1 ≠ partition c2 :=
sorry

end partition_cities_l95_95719


namespace even_function_cos_sin_l95_95538

theorem even_function_cos_sin {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = Real.cos (3 * x) + Real.sin (2 * x)) :
  ∀ x, x > 0 → f x = Real.cos (3 * x) - Real.sin (2 * x) := by
  sorry

end even_function_cos_sin_l95_95538


namespace number_of_diet_soda_l95_95073

variable (d r : ℕ)

-- Define the conditions of the problem
def condition1 : Prop := r = d + 79
def condition2 : Prop := r = 83

-- State the theorem we want to prove
theorem number_of_diet_soda (h1 : condition1 d r) (h2 : condition2 r) : d = 4 :=
by
  sorry

end number_of_diet_soda_l95_95073


namespace fifth_equation_l95_95125

-- Define the conditions
def condition1 : Prop := 2^1 * 1 = 2
def condition2 : Prop := 2^2 * 1 * 3 = 3 * 4
def condition3 : Prop := 2^3 * 1 * 3 * 5 = 4 * 5 * 6

-- The statement to prove
theorem fifth_equation (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
sorry

end fifth_equation_l95_95125


namespace sum_single_digit_numbers_l95_95514

noncomputable def are_single_digit_distinct (a b c d : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_single_digit_numbers :
  ∀ (A B C D : ℕ),
  are_single_digit_distinct A B C D →
  1000 * A + B - (5000 + 10 * C + 9) = 1000 + 100 * D + 93 →
  A + B + C + D = 18 :=
by
  sorry

end sum_single_digit_numbers_l95_95514


namespace product_of_g_on_roots_l95_95311

-- Define the given polynomials f and g
def f (x : ℝ) : ℝ := x^5 + 3 * x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 5

-- Define the roots of the polynomial f
axiom roots : ∃ (x1 x2 x3 x4 x5 : ℝ), 
  f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0

theorem product_of_g_on_roots : 
  (∃ x1 x2 x3 x4 x5: ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0) 
  → g x1 * g x2 * g x3 * g x4 * g x5 = 131 := 
by
  sorry

end product_of_g_on_roots_l95_95311


namespace min_expression_value_l95_95002

def distinct_elements (s : Set ℤ) : Prop := s = {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_expression_value :
  ∃ (p q r s t u v w : ℤ),
    distinct_elements {p, q, r, s, t, u, v, w} ∧
    (p + q + r + s) ≥ 5 ∧
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
     q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
     r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
     s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
     t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
     u ≠ v ∧ u ≠ w ∧
     v ≠ w) →
    (p + q + r + s)^2 + (t + u + v + w)^2 = 26 :=
sorry

end min_expression_value_l95_95002


namespace remaining_miles_to_be_built_l95_95479

-- Definitions from problem conditions
def current_length : ℕ := 200
def target_length : ℕ := 650
def first_day_miles : ℕ := 50
def second_day_miles : ℕ := 3 * first_day_miles

-- Lean theorem statement
theorem remaining_miles_to_be_built : 
  (target_length - current_length) - (first_day_miles + second_day_miles) = 250 := 
by 
  sorry

end remaining_miles_to_be_built_l95_95479


namespace bert_reaches_kameron_l95_95427

theorem bert_reaches_kameron {days : ℕ} (Kameron_kangaroos Bert_kangaroos rate : ℕ) 
  (hK : Kameron_kangaroos = 100) (hB : Bert_kangaroos = 20) (hr : rate = 2) :
  days = (Kameron_kangaroos - Bert_kangaroos) / rate := 
by
  sorry

example : ∃ days, bert_reaches_kameron 100 20 2 days = 40 := 
by
  use 40
  apply bert_reaches_kameron
  repeat { sorry }

end bert_reaches_kameron_l95_95427


namespace _l95_95743

def triangle (A B C : Type) : Prop :=
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def angles_not_equal_sides_not_equal (A B C : Type) (angleB angleC : ℝ) (sideAC sideAB : ℝ) : Prop :=
  triangle A B C →
  (angleB ≠ angleC → sideAC ≠ sideAB)
  
lemma xiaoming_theorem {A B C : Type} 
  (hTriangle : triangle A B C)
  (angleB angleC : ℝ)
  (sideAC sideAB : ℝ) :
  angleB ≠ angleC → sideAC ≠ sideAB := 
sorry

end _l95_95743


namespace line_passes_through_fixed_point_l95_95583

theorem line_passes_through_fixed_point (m : ℝ) :
  (m-1) * 9 + (2*m-1) * (-4) = m - 5 :=
by
  sorry

end line_passes_through_fixed_point_l95_95583


namespace infinite_geometric_series_common_ratio_l95_95220

theorem infinite_geometric_series_common_ratio :
  ∀ (a r S : ℝ), a = 500 ∧ S = 4000 ∧ (a / (1 - r) = S) → r = 7 / 8 :=
by
  intros a r S h
  cases h with h_a h_S_eq
  cases h_S_eq with h_S h_sum_eq
  -- Now we have: a = 500, S = 4000, and a / (1 - r) = S
  sorry

end infinite_geometric_series_common_ratio_l95_95220


namespace oranges_for_price_of_apples_l95_95416

-- Given definitions based on the conditions provided
def cost_of_apples_same_as_bananas (a b : ℕ) : Prop := 12 * a = 6 * b
def cost_of_bananas_same_as_cucumbers (b c : ℕ) : Prop := 3 * b = 5 * c
def cost_of_cucumbers_same_as_oranges (c o : ℕ) : Prop := 2 * c = 1 * o

-- The theorem to prove
theorem oranges_for_price_of_apples (a b c o : ℕ) 
  (hab : cost_of_apples_same_as_bananas a b)
  (hbc : cost_of_bananas_same_as_cucumbers b c)
  (hco : cost_of_cucumbers_same_as_oranges c o) : 
  24 * a = 10 * o :=
sorry

end oranges_for_price_of_apples_l95_95416


namespace first_year_with_sum_of_digits_10_after_2020_l95_95381

theorem first_year_with_sum_of_digits_10_after_2020 :
  ∃ (y : ℕ), y > 2020 ∧ (y.digits 10).sum = 10 ∧ ∀ (z : ℕ), (z > 2020 ∧ (z.digits 10).sum = 10) → y ≤ z :=
sorry

end first_year_with_sum_of_digits_10_after_2020_l95_95381


namespace cos_150_eq_neg_sqrt3_div_2_l95_95781

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95781


namespace solve_f_eq_x_l95_95875

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_domain : ∀ (x : ℝ), 0 ≤ x ∧ x < 1 → 1 ≤ f_inv x ∧ f_inv x < 2
axiom f_inv_range : ∀ (x : ℝ), 2 < x ∧ x ≤ 4 → 0 ≤ f_inv x ∧ f_inv x < 1
-- Assumption that f is invertible on [0, 3]
axiom f_inv_exists : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, f y = x

theorem solve_f_eq_x : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = x → x = 2 :=
by
  sorry

end solve_f_eq_x_l95_95875


namespace no_four_digit_number_ending_in_47_is_divisible_by_5_l95_95535

theorem no_four_digit_number_ending_in_47_is_divisible_by_5 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n % 100 = 47 → n % 10 ≠ 0 ∧ n % 10 ≠ 5) := by
  intro n
  intro Hn
  intro H47
  sorry

end no_four_digit_number_ending_in_47_is_divisible_by_5_l95_95535


namespace inequality_reciprocal_of_negatives_l95_95488

theorem inequality_reciprocal_of_negatives (a b : ℝ) (ha : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_of_negatives_l95_95488


namespace onions_left_l95_95447

def sallyOnions : ℕ := 5
def fredOnions : ℕ := 9
def onionsGivenToSara : ℕ := 4

theorem onions_left : (sallyOnions + fredOnions) - onionsGivenToSara = 10 := by
  sorry

end onions_left_l95_95447


namespace range_of_a_l95_95645

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x < a → x ^ 2 > 1 ∧ ¬(x ^ 2 > 1 → x < a)) : a ≤ -1 :=
sorry

end range_of_a_l95_95645


namespace probability_of_prime_number_on_spinner_l95_95338

-- Definitions of conditions
def spinner_sections : List ℕ := [2, 3, 4, 5, 7, 9, 10, 11]
def total_sectors : ℕ := 8
def prime_count : ℕ := List.filter Nat.Prime spinner_sections |>.length

-- Statement of the theorem we want to prove
theorem probability_of_prime_number_on_spinner :
  (prime_count : ℚ) / total_sectors = 5 / 8 := by
  sorry

end probability_of_prime_number_on_spinner_l95_95338


namespace contradiction_in_triangle_l95_95321

theorem contradiction_in_triangle :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A < 60 ∧ B < 60 ∧ C < 60 → false) :=
by
  sorry

end contradiction_in_triangle_l95_95321


namespace combined_time_l95_95322

def time_pulsar : ℕ := 10
def time_polly : ℕ := 3 * time_pulsar
def time_petra : ℕ := time_polly / 6

theorem combined_time : time_pulsar + time_polly + time_petra = 45 := 
by 
  -- proof steps will go here
  sorry

end combined_time_l95_95322


namespace fractional_part_of_students_who_walk_home_l95_95769

def fraction_bus := 1 / 3
def fraction_automobile := 1 / 5
def fraction_bicycle := 1 / 8
def fraction_scooter := 1 / 10

theorem fractional_part_of_students_who_walk_home :
  (1 : ℚ) - (fraction_bus + fraction_automobile + fraction_bicycle + fraction_scooter) = 29 / 120 :=
by
  sorry

end fractional_part_of_students_who_walk_home_l95_95769


namespace walnut_trees_currently_in_park_l95_95046

-- Definitions from the conditions
def total_trees : ℕ := 77
def trees_to_be_planted : ℕ := 44

-- Statement to prove: number of current trees = 33
theorem walnut_trees_currently_in_park : total_trees - trees_to_be_planted = 33 :=
by
  sorry

end walnut_trees_currently_in_park_l95_95046


namespace tangent_line_at_1_range_a_II_range_a_III_l95_95399

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + log x

-- I. Prove the equation of the tangent line at the point (1, f(1)) when a = 1 is y = -2
theorem tangent_line_at_1 (x : ℝ) (hx : x = 1) : ∀ y, 
  let f1 := f 1 x,
  let f1' := (deriv (f 1)) 1 in
  f1 = -2 ∧ f1' = 0 → 
  y = -2 :=
  by
  sorry

-- II. Prove the range of a is a ≥ 1 given the minimum value of f(x) on [1, e] is -2.
theorem range_a_II (a : ℝ) (ha : a > 0) : 
  ∀ x ∈ set.Icc 1 Real.exp,
  let minf := -2
  in Inf (set.image (f a) (set.Icc 1 Real.exp)) = minf → 
  a ≥ 1 :=
  by
  sorry

-- III. Prove the range of a is 0 ≤ a ≤ 8 for the condition on the interval (0,+∞) 
theorem range_a_III (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ set.Ioi 0 ∧ x2 ∈ set.Ioi 0 → x1 < x2 → 
  f a x1 + 2 * x1 < f a x2 + 2 * x2) →
  0 ≤ a ∧ a ≤ 8 :=
  by
  sorry

end tangent_line_at_1_range_a_II_range_a_III_l95_95399


namespace inequality_holds_l95_95467

theorem inequality_holds (a b c : ℝ) (h1 : a > b) (h2 : b > c) : (a - b) * |c - b| > 0 :=
sorry

end inequality_holds_l95_95467


namespace cos_150_eq_neg_sqrt3_div_2_l95_95835

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95835


namespace percentage_stock_sold_l95_95174

/-!
# Problem Statement
Given:
1. The cash realized on selling a certain percentage stock is Rs. 109.25.
2. The brokerage is 1/4%.
3. The cash after deducting the brokerage is Rs. 109.

Prove:
The percentage of the stock sold is 100%.
-/

noncomputable def brokerage_fee (S : ℝ) : ℝ :=
  S * 0.0025

noncomputable def selling_price (realized_cash : ℝ) (fee : ℝ) : ℝ :=
  realized_cash + fee

theorem percentage_stock_sold (S : ℝ) (realized_cash : ℝ) (cash_after_brokerage : ℝ)
  (h1 : realized_cash = 109.25)
  (h2 : cash_after_brokerage = 109)
  (h3 : brokerage_fee S = S * 0.0025) :
  S = 109.25 :=
by
  sorry

end percentage_stock_sold_l95_95174


namespace remainder_of_powers_l95_95340

theorem remainder_of_powers (n1 n2 n3 : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 :=
by
  sorry

end remainder_of_powers_l95_95340


namespace sam_initial_balloons_l95_95166

theorem sam_initial_balloons:
  ∀ (S : ℕ), (S - 10 + 16 = 52) → S = 46 :=
by
  sorry

end sam_initial_balloons_l95_95166


namespace multiplication_addition_example_l95_95492

theorem multiplication_addition_example :
  469138 * 9999 + 876543 * 12345 = 15512230997 :=
by
  sorry

end multiplication_addition_example_l95_95492


namespace first_tap_time_l95_95355

-- Define the variables and conditions
variables (T : ℝ)
-- The cistern can be emptied by the second tap in 9 hours
-- Both taps together fill the cistern in 7.2 hours.
def first_tap_fills_cistern_in_time (T : ℝ) :=
  (1 / T) - (1 / 9) = 1 / 7.2

theorem first_tap_time :
  first_tap_fills_cistern_in_time 4 :=
by
  -- now we can use the definition to show the proof
  unfold first_tap_fills_cistern_in_time
  -- directly substitute and show
  sorry

end first_tap_time_l95_95355


namespace third_box_weight_l95_95014

def b1 : ℕ := 2
def difference := 11

def weight_third_box (b1 b3 difference : ℕ) : Prop :=
  b3 - b1 = difference

theorem third_box_weight : weight_third_box b1 13 difference :=
by
  simp [b1, difference]
  sorry

end third_box_weight_l95_95014


namespace rent_3600_rents_88_max_revenue_is_4050_l95_95351

def num_total_cars : ℕ := 100
def initial_rent : ℕ := 3000
def rent_increase_step : ℕ := 50
def maintenance_cost_rented : ℕ := 150
def maintenance_cost_unrented : ℕ := 50

def rented_cars (rent : ℕ) : ℕ :=
  if rent < initial_rent then num_total_cars
  else num_total_cars - ((rent - initial_rent) / rent_increase_step)

def monthly_revenue (rent : ℕ) : ℕ :=
  let rented := rented_cars rent
  rent * rented - (rented * maintenance_cost_rented + (num_total_cars - rented) * maintenance_cost_unrented)

theorem rent_3600_rents_88 :
  rented_cars 3600 = 88 := by 
  sorry

theorem max_revenue_is_4050 :
  ∃ (rent : ℕ), rent = 4050 ∧ monthly_revenue rent = 37050 := by
  sorry

end rent_3600_rents_88_max_revenue_is_4050_l95_95351


namespace martha_points_calculation_l95_95681

theorem martha_points_calculation :
  let beef_cost := 3 * 11
  let beef_discount := 0.10 * beef_cost
  let total_beef_cost := beef_cost - beef_discount

  let fv_cost := 8 * 4
  let fv_discount := 0.05 * fv_cost
  let total_fv_cost := fv_cost - fv_discount

  let spices_cost := 2 * 6

  let other_groceries_cost := 37 - 3

  let total_cost := total_beef_cost + total_fv_cost + spices_cost + other_groceries_cost

  let spending_points := (total_cost / 10).floor * 50

  let bonus_points_over_100 := if total_cost > 100 then 250 else 0

  let loyalty_points := 100
  
  spending_points + bonus_points_over_100 + loyalty_points = 850 := by
    sorry

end martha_points_calculation_l95_95681


namespace part_a_prob_part_b_expected_time_l95_95992

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l95_95992


namespace cos_150_eq_neg_sqrt3_div_2_l95_95773

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95773


namespace pyramid_base_edge_length_l95_95947

-- Prove that the edge-length of the base of the pyramid is as specified
theorem pyramid_base_edge_length
  (r h : ℝ)
  (hemisphere_radius : r = 3)
  (pyramid_height : h = 8)
  (tangency_condition : true) : true :=
by
  sorry

end pyramid_base_edge_length_l95_95947


namespace violet_prob_l95_95965

noncomputable def total_candies := 8 + 5 + 9 + 10 + 6

noncomputable def prob_green_first := (8 : ℚ) / total_candies
noncomputable def prob_yellow_second := (10 : ℚ) / (total_candies - 1)
noncomputable def prob_pink_third := (6 : ℚ) / (total_candies - 2)

noncomputable def combined_prob := prob_green_first * prob_yellow_second * prob_pink_third

theorem violet_prob :
  combined_prob = (20 : ℚ) / 2109 := by
    sorry

end violet_prob_l95_95965


namespace triangle_angle_sine_identity_l95_95406

theorem triangle_angle_sine_identity (A B C : ℝ) (n : ℤ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n + 1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) :=
by
  sorry

end triangle_angle_sine_identity_l95_95406


namespace area_above_line_is_zero_l95_95051

-- Define the circle: (x - 4)^2 + (y - 5)^2 = 12
def circle (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 12

-- Define the line: y = x - 2
def line (x y : ℝ) : Prop := y = x - 2

-- Prove that the area of the circle above this line is 0
theorem area_above_line_is_zero : 
  ∀ x y : ℝ, circle x y ∧ y > x - 2 → false := sorry

end area_above_line_is_zero_l95_95051


namespace smallest_n_value_l95_95466

-- Define the conditions as given in the problem
def num_birthdays := 365

-- Formulating the main statement
theorem smallest_n_value : ∃ (n : ℕ), (∀ (group_size : ℕ), group_size = 2 * n - 10 → group_size ≥ 3286) ∧ n = 1648 :=
by
  use 1648
  sorry

end smallest_n_value_l95_95466


namespace solve_for_x_l95_95701

theorem solve_for_x : ∃ x : ℚ, 24 - 4 = 3 * (1 + x) ∧ x = 17 / 3 :=
by
  sorry

end solve_for_x_l95_95701


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95856

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95856


namespace min_value_of_A_div_B_l95_95098

noncomputable def A (g1 : Finset ℕ) : ℕ :=
  g1.prod id

noncomputable def B (g2 : Finset ℕ) : ℕ :=
  g2.prod id

theorem min_value_of_A_div_B : ∃ (g1 g2 : Finset ℕ), 
  g1 ∪ g2 = (Finset.range 31).erase 0 ∧ g1 ∩ g2 = ∅ ∧ A g1 % B g2 = 0 ∧ A g1 / B g2 = 1077205 :=
by
  sorry

end min_value_of_A_div_B_l95_95098


namespace cos_150_degree_l95_95828

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l95_95828


namespace four_digit_numbers_l95_95276

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l95_95276


namespace find_k_l95_95643

-- Definition of vectors a and b
def vec_a (k : ℝ) : ℝ × ℝ := (-1, k)
def vec_b : ℝ × ℝ := (3, 1)

-- Definition of dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Property of perpendicular vectors (dot product is zero)
def is_perpendicular (u v : ℝ × ℝ) : Prop := dot_product u v = 0

-- Problem statement
theorem find_k (k : ℝ) :
  is_perpendicular (vec_a k) (vec_a k) →
  (k = -2 ∨ k = 1) :=
sorry

end find_k_l95_95643


namespace sum_digits_next_l95_95668

-- Given the sum of the digits function S(n)
def S (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Defining the properties based on conditions
theorem sum_digits_next (n : ℕ) (h : S n = 876) : S (n + 1) = 877 :=
begin
  sorry -- Proof goes here
end

end sum_digits_next_l95_95668


namespace seat_adjustment_schemes_l95_95293

theorem seat_adjustment_schemes {n k : ℕ} (h1 : n = 7) (h2 : k = 3) :
  (2 * Nat.choose n k) = 70 :=
by
  -- n is the number of people, k is the number chosen
  rw [h1, h2]
  -- the rest is skipped for the statement only
  sorry

end seat_adjustment_schemes_l95_95293


namespace isosceles_triangles_possible_l95_95295

theorem isosceles_triangles_possible :
  ∃ (sticks : List ℕ), (sticks = [1, 1, 2, 2, 3, 3] ∧ 
    ∀ (a b c : ℕ), a ∈ sticks → b ∈ sticks → c ∈ sticks → 
    ((a + b > c ∧ b + c > a ∧ c + a > b) → a = b ∨ b = c ∨ c = a)) :=
sorry

end isosceles_triangles_possible_l95_95295


namespace roots_ratio_sum_eq_six_l95_95646

theorem roots_ratio_sum_eq_six (x1 x2 : ℝ) (h1 : 2 * x1^2 - 4 * x1 + 1 = 0) (h2 : 2 * x2^2 - 4 * x2 + 1 = 0) :
  (x1 / x2) + (x2 / x1) = 6 :=
sorry

end roots_ratio_sum_eq_six_l95_95646


namespace remove_5_increases_probability_l95_95728

theorem remove_5_increases_probability :
  let T := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let remove_5 := T.erase 5
  let valid_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
    S.product S.filter (λ b, 10 - b ∈ S) -- pairs (a, b) with a + b = 10 and distinct

  (0 < (valid_pairs remove_5).card) ∧
  (valid_pairs remove_5).card ≤ 4 :=
begin
  let T := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ,
  let valid_pairs := λ S : Finset ℕ, S.product S.filter (λ b, 10 - b ∈ S),

  have hs: T.card = 10 := rfl,
  have h_remove_5: (T.erase 5).card = 9 := rfl,

  sorry
end

end remove_5_increases_probability_l95_95728


namespace binom_9_5_l95_95091

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end binom_9_5_l95_95091


namespace gcd_204_85_l95_95028

theorem gcd_204_85 : Nat.gcd 204 85 = 17 :=
sorry

end gcd_204_85_l95_95028


namespace cos_150_eq_neg_sqrt3_div_2_l95_95774

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95774


namespace range_of_x_l95_95964

theorem range_of_x (total_students math_club chemistry_club : ℕ) (h_total : total_students = 45) 
(h_math : math_club = 28) (h_chemistry : chemistry_club = 21) (x : ℕ) :
  4 ≤ x ∧ x ≤ 21 ↔ (28 + 21 - x ≤ 45) :=
by sorry

end range_of_x_l95_95964


namespace lemons_count_l95_95704

def total_fruits (num_baskets : ℕ) (total : ℕ) : Prop := num_baskets = 5 ∧ total = 58
def basket_contents (basket : ℕ → ℕ) : Prop := 
  basket 1 = 18 ∧ -- mangoes
  basket 2 = 10 ∧ -- pears
  basket 3 = 12 ∧ -- pawpaws
  (∀ i, (i = 4 ∨ i = 5) → basket i = (basket 4 + basket 5) / 2)

theorem lemons_count (num_baskets : ℕ) (total : ℕ) (basket : ℕ → ℕ) : 
  total_fruits num_baskets total ∧ basket_contents basket → basket 5 = 9 :=
by
  sorry

end lemons_count_l95_95704


namespace cos_150_eq_neg_sqrt3_div_2_l95_95850

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95850


namespace satisfying_n_l95_95498

theorem satisfying_n (n : ℕ) 
  (h : n ≥ 2)
  (cond : ∀ (a b : ℤ), Int.gcd a n = 1 → Int.gcd b n = 1 → (a % n = b % n ↔ a * b % n = 1 % n)) :
  n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 12 ∨ n = 24 := sorry

end satisfying_n_l95_95498


namespace pizza_area_percentage_increase_l95_95021

theorem pizza_area_percentage_increase :
  let r1 := 6
  let r2 := 4
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  let deltaA := A1 - A2
  let N := (deltaA / A2) * 100
  N = 125 := by
  sorry

end pizza_area_percentage_increase_l95_95021


namespace problem_l95_95117

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 2

-- We are given f(-2) = m
variables (a b m : ℝ)
theorem problem (h : f (-2) a b = m) : f 2 a b + f (-2) a b = -4 :=
by sorry

end problem_l95_95117


namespace sum_of_possible_values_of_g1_l95_95433

def g (x : ℝ) : ℝ := sorry

axiom g_prop : ∀ x y : ℝ, g (g (x - y)) = g x * g y - g x + g y - x^2 * y^2

theorem sum_of_possible_values_of_g1 : g 1 = -1 := by sorry

end sum_of_possible_values_of_g1_l95_95433


namespace range_of_m_l95_95390

theorem range_of_m (a m x : ℝ) (p q : Prop) :
  (p ↔ ∃ (a : ℝ) (m : ℝ), ∀ (x : ℝ), 4 * x^2 - 2 * a * x + 2 * a + 5 = 0) →
  (q ↔ 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0) →
  (¬ p → ¬ q) →
  (∀ a, -2 ≤ a ∧ a ≤ 10) →
  (1 - m ≤ -2) ∧ (1 + m ≥ 10) →
  m ≥ 9 :=
by
  intros hp hq npnq ha hm
  sorry  -- Proof omitted

end range_of_m_l95_95390


namespace negation_of_universal_l95_95955

variable {f g : ℝ → ℝ}

theorem negation_of_universal :
  ¬ (∀ x : ℝ, f x * g x ≠ 0) ↔ ∃ x₀ : ℝ, f x₀ = 0 ∨ g x₀ = 0 :=
by
  sorry

end negation_of_universal_l95_95955


namespace number_of_four_digit_numbers_l95_95270

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l95_95270


namespace cosine_150_eq_neg_sqrt3_div_2_l95_95861

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l95_95861


namespace probability_event_1_eq_half_probability_event_2_not_eq_half_probability_event_3_eq_half_probability_event_4_not_eq_half_l95_95164

-- Define the number of tosses for players A and B
def num_tosses_A : ℕ := 2017
def num_tosses_B : ℕ := 2016

-- Define the random variables for the number of heads for players A and B
def heads_A : ℕ := num_tosses_A // 2 -- Assuming a fair division of heads and tails
def heads_B : ℕ := num_tosses_B // 2

-- Define the events
def event_1 : Prop := heads_A > heads_B
def event_2 : Prop := (num_tosses_A - heads_A) < heads_B
def event_3 : Prop := (num_tosses_A - heads_A) > heads_A
def event_4 : Prop := heads_B = (num_tosses_B - heads_B)

-- The conjectures we need to prove
theorem probability_event_1_eq_half : probability event_1 = 0.5 := sorry
theorem probability_event_2_not_eq_half : probability event_2 ≠ 0.5 := sorry
theorem probability_event_3_eq_half : probability event_3 = 0.5 := sorry
theorem probability_event_4_not_eq_half : probability event_4 ≠ 0.5 := sorry

end probability_event_1_eq_half_probability_event_2_not_eq_half_probability_event_3_eq_half_probability_event_4_not_eq_half_l95_95164


namespace population_in_2001_l95_95957

-- Define the populations at specific years
def pop_2000 := 50
def pop_2002 := 146
def pop_2003 := 350

-- Define the population difference condition
def pop_condition (n : ℕ) (pop : ℕ → ℕ) :=
  pop (n + 3) - pop n = 3 * pop (n + 2)

-- Given that the population condition holds, and specific populations are known,
-- the population in the year 2001 is 100
theorem population_in_2001 :
  (∃ (pop : ℕ → ℕ), pop 2000 = pop_2000 ∧ pop 2002 = pop_2002 ∧ pop 2003 = pop_2003 ∧ 
    pop_condition 2000 pop) → ∃ (pop : ℕ → ℕ), pop 2001 = 100 :=
by
  -- Placeholder for the actual proof
  sorry

end population_in_2001_l95_95957


namespace ball_bounce_height_l95_95202

theorem ball_bounce_height (h : ℝ) (r : ℝ) (k : ℕ) (hk : h * r^k < 6) :
  h = 2000 ∧ r = 1/3 → k = 6 :=
by
  intros h_cond r_cond
  rw [h_cond, r_cond] at hk
  sorry

end ball_bounce_height_l95_95202


namespace area_of_circle_l95_95734

def circleEquation (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y = -9

theorem area_of_circle :
  (∃ (center : ℝ × ℝ) (radius : ℝ), radius = 4 ∧ ∀ (x y : ℝ), circleEquation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) →
  (16 * Real.pi) = 16 * Real.pi := 
by 
  intro h
  have := h
  sorry

end area_of_circle_l95_95734


namespace units_digit_2_pow_2015_l95_95160

theorem units_digit_2_pow_2015 : ∃ u : ℕ, (2 ^ 2015 % 10) = u ∧ u = 8 := 
by
  sorry

end units_digit_2_pow_2015_l95_95160


namespace helium_balloon_buoyancy_l95_95590

variable (m m₁ Mₐ M_b : ℝ)
variable (h₁ : m₁ = 10)
variable (h₂ : Mₐ = 4)
variable (h₃ : M_b = 29)

theorem helium_balloon_buoyancy :
  m = (m₁ * Mₐ) / (M_b - Mₐ) :=
by
  sorry

end helium_balloon_buoyancy_l95_95590


namespace choose_starters_with_twins_l95_95759

theorem choose_starters_with_twins :
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  total_ways - without_twins = 540 := 
by
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  exact Nat.sub_eq_of_eq_add sorry -- here we will need the exact proof steps which we skip

end choose_starters_with_twins_l95_95759


namespace log_2_bounds_l95_95579

theorem log_2_bounds:
  (2^9 = 512) → (2^8 = 256) → (10^2 = 100) → (10^3 = 1000) → 
  (2 / 9 < Real.log 2 / Real.log 10) ∧ (Real.log 2 / Real.log 10 < 3 / 8) :=
by
  intros h1 h2 h3 h4
  sorry

end log_2_bounds_l95_95579


namespace disproves_proposition_b_l95_95346

-- Definition and condition of complementary angles
def angles_complementary (angle1 angle2: ℝ) : Prop := angle1 + angle2 = 180

-- Proposition to disprove
def disprove (angle1 angle2: ℝ) : Prop := ¬ ((angle1 < 90 ∧ angle2 > 90 ∧ angle2 < 180) ∨ (angle2 < 90 ∧ angle1 > 90 ∧ angle1 < 180))

-- Definition of angles in sets
def set_a := (120, 60)
def set_b := (95.1, 84.9)
def set_c := (30, 60)
def set_d := (90, 90)

-- Statement to prove
theorem disproves_proposition_b : 
  (angles_complementary 95.1 84.9) ∧ (disprove 95.1 84.9) :=
by
  sorry

end disproves_proposition_b_l95_95346


namespace science_club_members_neither_l95_95684

theorem science_club_members_neither {S B C : ℕ} (total : S = 60) (bio : B = 40) (chem : C = 35) (both : ℕ := 25) :
    S - ((B - both) + (C - both) + both) = 10 :=
by
  sorry

end science_club_members_neither_l95_95684


namespace point_coordinates_in_second_quadrant_l95_95686

theorem point_coordinates_in_second_quadrant (P : ℝ × ℝ)
  (hx : P.1 ≤ 0)
  (hy : P.2 ≥ 0)
  (dist_x_axis : abs P.2 = 3)
  (dist_y_axis : abs P.1 = 10) :
  P = (-10, 3) :=
by
  sorry

end point_coordinates_in_second_quadrant_l95_95686


namespace quadratic_has_one_solution_l95_95019

theorem quadratic_has_one_solution (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) ∧ (∀ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ + m = 0) → (3 * x₂^2 - 6 * x₂ + m = 0) → x₁ = x₂) → m = 3 :=
by
  -- intricate steps would go here
  sorry

end quadratic_has_one_solution_l95_95019


namespace intersection_of_sets_l95_95892

def set_M : Set ℝ := { x : ℝ | (x + 2) * (x - 1) < 0 }
def set_N : Set ℝ := { x : ℝ | x + 1 < 0 }
def intersection (A B : Set ℝ) : Set ℝ := { x : ℝ | x ∈ A ∧ x ∈ B }

theorem intersection_of_sets :
  intersection set_M set_N = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry

end intersection_of_sets_l95_95892


namespace cos_150_eq_neg_sqrt3_over_2_l95_95817

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95817


namespace cylindrical_to_rectangular_point_l95_95024

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_point :
  cylindrical_to_rectangular (Real.sqrt 2) (Real.pi / 4) 1 = (1, 1, 1) :=
by
  sorry

end cylindrical_to_rectangular_point_l95_95024


namespace quadratic_value_at_6_l95_95388

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 3

theorem quadratic_value_at_6 
  (a b : ℝ) (h : a ≠ 0) 
  (h_eq : f a b 2 = f a b 4) : 
  f a b 6 = -3 :=
by
  sorry

end quadratic_value_at_6_l95_95388


namespace solve_positive_integer_x_l95_95943

theorem solve_positive_integer_x : ∃ (x : ℕ), 4 * x^2 - 16 * x - 60 = 0 ∧ x = 6 :=
by
  sorry

end solve_positive_integer_x_l95_95943


namespace rancher_problem_l95_95598

theorem rancher_problem (s c : ℕ) (h : 30 * s + 35 * c = 1500) : (s = 1 ∧ c = 42) ∨ (s = 36 ∧ c = 12) := 
by
  sorry

end rancher_problem_l95_95598


namespace number_of_girls_l95_95294

theorem number_of_girls
  (B G : ℕ)
  (h1 : B = (8 * G) / 5)
  (h2 : B + G = 351) :
  G = 135 :=
sorry

end number_of_girls_l95_95294


namespace totalKidsInLawrenceCounty_l95_95925

-- Constants representing the number of kids in each category
def kidsGoToCamp : ℕ := 629424
def kidsStayHome : ℕ := 268627

-- Statement of the total number of kids in Lawrence county
theorem totalKidsInLawrenceCounty : kidsGoToCamp + kidsStayHome = 898051 := by
  sorry

end totalKidsInLawrenceCounty_l95_95925


namespace find_positive_integer_l95_95870

def product_of_digits (n : Nat) : Nat :=
  -- Function to compute product of digits, assume it is defined correctly
  sorry

theorem find_positive_integer (x : Nat) (h : x > 0) :
  product_of_digits x = x * x - 10 * x - 22 ↔ x = 12 :=
by
  sorry

end find_positive_integer_l95_95870


namespace tangent_parallel_l95_95621

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel (P₀ : ℝ × ℝ) :
  (∃ x : ℝ, (P₀ = (x, f x) ∧ deriv f x = 4)) 
  ↔ (P₀ = (1, 0) ∨ P₀ = (-1, -4)) :=
by 
  sorry

end tangent_parallel_l95_95621


namespace D_72_l95_95667

def D (n : ℕ) : ℕ :=
  -- Definition of D(n) should be provided here
  sorry

theorem D_72 : D 72 = 121 :=
  sorry

end D_72_l95_95667


namespace luke_trays_l95_95935

theorem luke_trays 
  (carries_per_trip : ℕ)
  (trips : ℕ)
  (second_table_trays : ℕ)
  (total_trays : carries_per_trip * trips = 36)
  (second_table_value : second_table_trays = 16) : 
  carries_per_trip * trips - second_table_trays = 20 :=
by sorry

end luke_trays_l95_95935


namespace cos_150_eq_neg_sqrt3_over_2_l95_95796

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95796


namespace triangle_perimeter_l95_95952

-- Definitions for the conditions
def inscribed_circle_of_triangle_tangent_at (radius : ℝ) (DP : ℝ) (PE : ℝ) : Prop :=
  radius = 27 ∧ DP = 29 ∧ PE = 33

-- Perimeter calculation theorem
theorem triangle_perimeter (r DP PE : ℝ) (h : inscribed_circle_of_triangle_tangent_at r DP PE) : 
  ∃ perimeter : ℝ, perimeter = 774 :=
by
  sorry

end triangle_perimeter_l95_95952


namespace books_sold_l95_95305

theorem books_sold (initial_books remaining_books sold_books : ℕ):
  initial_books = 33 → 
  remaining_books = 7 → 
  sold_books = initial_books - remaining_books → 
  sold_books = 26 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end books_sold_l95_95305


namespace cone_height_l95_95071

theorem cone_height (r : ℝ) (n : ℕ) (sec_r : ℝ) (h : ℝ) (slant_height : ℝ) ( base_radius : ℝ):
  (r = 10) → (n = 4) → (sec_r = (2 * r * Real.pi) / n) → (slant_height = r) →
  (base_radius = sec_r / (2 * Real.pi)) → 
  (h = Real.sqrt (slant_height ^ 2 - base_radius ^ 2)) →
  h = Real.sqrt 93.75 := 
by {
  intros r_val n_val sec_r_val slant_height_val base_radius_val height_eq,
  exact height_eq,
  sorry
}

end cone_height_l95_95071


namespace sequence_divisible_by_three_l95_95432

-- Define the conditions
variable (k : ℕ) (h_pos_k : k > 0)
variable (a : ℕ → ℤ)
variable (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n)

-- Define the proof goal
theorem sequence_divisible_by_three (k : ℕ) (h_pos_k : k > 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n : ℕ, n ≥ 1 -> a n = (a (n-1) + n^k) / n) : (k - 2) % 3 = 0 :=
by
  sorry

end sequence_divisible_by_three_l95_95432


namespace solve_for_x_l95_95536

theorem solve_for_x : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) → x = 31.2 :=
by sorry

end solve_for_x_l95_95536


namespace find_a_l95_95396

theorem find_a : (∃ x, x^2 + x + 2 * a - 1 = 0) → a = 1 / 2 :=
by
  assume h
  sorry

end find_a_l95_95396


namespace day_of_week_dec_26_l95_95101

theorem day_of_week_dec_26 (nov_26_is_thu : true) : true :=
sorry

end day_of_week_dec_26_l95_95101


namespace cos_150_eq_neg_sqrt3_div_2_l95_95834

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l95_95834


namespace elise_hospital_distance_l95_95234

noncomputable def distance_to_hospital (total_fare: ℝ) (base_price: ℝ) (toll_price: ℝ) 
(tip_percent: ℝ) (cost_per_mile: ℝ) (increase_percent: ℝ) (toll_count: ℕ) : ℝ :=
let base_and_tolls := base_price + (toll_price * toll_count)
let fare_before_tip := total_fare / (1 + tip_percent)
let distance_fare := fare_before_tip - base_and_tolls
let original_travel_fare := distance_fare / (1 + increase_percent)
original_travel_fare / cost_per_mile

theorem elise_hospital_distance : distance_to_hospital 34.34 3 2 0.15 4 0.20 3 = 5 := 
sorry

end elise_hospital_distance_l95_95234


namespace filled_sacks_count_l95_95372

-- Definitions from the problem conditions
def pieces_per_sack := 20
def total_pieces := 80

theorem filled_sacks_count : total_pieces / pieces_per_sack = 4 := 
by sorry

end filled_sacks_count_l95_95372


namespace deg_q_l95_95018

-- Define the polynomials and their degrees
variables (p q : Polynomial ℝ) -- Generic polynomials p and q
variable (i : Polynomial ℝ := p.comp(q)^2 - q^3) -- Define i(x) = p(q(x))^2 - q(x)^3

-- State the conditions
axiom deg_p : p.degree = 4
axiom deg_i : i.degree = 12

-- Define the statement to prove
theorem deg_q : q.degree = 4 :=
by
  sorry -- Proof is omitted as per instructions

end deg_q_l95_95018


namespace planes_parallel_l95_95195

theorem planes_parallel (n1 n2 : ℝ × ℝ × ℝ)
  (h1 : n1 = (2, -1, 0)) 
  (h2 : n2 = (-4, 2, 0)) :
  ∃ k : ℝ, n2 = k • n1 := by
  -- Proof is beyond the scope of this exercise.
  sorry

end planes_parallel_l95_95195


namespace add_neg_two_eq_zero_l95_95350

theorem add_neg_two_eq_zero :
  (-2) + 2 = 0 :=
by
  sorry

end add_neg_two_eq_zero_l95_95350


namespace cubic_sum_l95_95568

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end cubic_sum_l95_95568


namespace simplify_expression_l95_95565

variable (x : ℝ)

theorem simplify_expression :
  3 * x^3 + 4 * x + 5 * x^2 + 2 - (7 - 3 * x^3 - 4 * x - 5 * x^2) =
  6 * x^3 + 10 * x^2 + 8 * x - 5 :=
by
  sorry

end simplify_expression_l95_95565


namespace career_preference_angles_l95_95959

theorem career_preference_angles (m f : ℕ) (total_degrees : ℕ) (one_fourth_males one_half_females : ℚ) (male_ratio female_ratio : ℚ) :
  total_degrees = 360 → male_ratio = 2/3 → female_ratio = 3/3 →
  m = 2 * f / 3 → one_fourth_males = 1/4 * m → one_half_females = 1/2 * f →
  (one_fourth_males + one_half_females) / (m + f) * total_degrees = 144 :=
by
  sorry

end career_preference_angles_l95_95959


namespace vector_calculation_l95_95405

namespace VectorProof

variables (a b : ℝ × ℝ) (m : ℝ)

def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k • v2)

theorem vector_calculation
  (h₁ : a = (1, -2))
  (h₂ : b = (m, 4))
  (h₃ : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end VectorProof

end vector_calculation_l95_95405


namespace parametric_to_cartesian_l95_95402

theorem parametric_to_cartesian (t : ℝ) (x y : ℝ) (h1 : x = 5 + 3 * t) (h2 : y = 10 - 4 * t) : 4 * x + 3 * y = 50 :=
by sorry

end parametric_to_cartesian_l95_95402


namespace cos_150_eq_negative_cos_30_l95_95785

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95785


namespace total_digits_in_numbering_pages_l95_95983

theorem total_digits_in_numbering_pages (n : ℕ) (h : n = 100000) : 
  let digits1 := 9 * 1
  let digits2 := (99 - 10 + 1) * 2
  let digits3 := (999 - 100 + 1) * 3
  let digits4 := (9999 - 1000 + 1) * 4
  let digits5 := (99999 - 10000 + 1) * 5
  let digits6 := 6
  (digits1 + digits2 + digits3 + digits4 + digits5 + digits6) = 488895 :=
by
  sorry

end total_digits_in_numbering_pages_l95_95983


namespace sum_of_fractions_l95_95194

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end sum_of_fractions_l95_95194


namespace gcd_204_85_l95_95034

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l95_95034


namespace football_cost_correct_l95_95364

variable (total_spent_on_toys : ℝ := 12.30)
variable (spent_on_marbles : ℝ := 6.59)

theorem football_cost_correct :
  (total_spent_on_toys - spent_on_marbles = 5.71) :=
by
  sorry

end football_cost_correct_l95_95364


namespace cubic_roots_expression_l95_95670

noncomputable def polynomial : Polynomial ℂ :=
  Polynomial.X^3 - 3 * Polynomial.X - 2

theorem cubic_roots_expression (α β γ : ℂ)
  (h1 : (Polynomial.X - Polynomial.C α) * 
        (Polynomial.X - Polynomial.C β) * 
        (Polynomial.X - Polynomial.C γ) = polynomial) :
  α * (β - γ)^2 + β * (γ - α)^2 + γ * (α - β)^2 = -18 :=
by
  sorry

end cubic_roots_expression_l95_95670


namespace kiran_has_105_l95_95553

theorem kiran_has_105 
  (R G K L : ℕ) 
  (ratio_rg : 6 * G = 7 * R)
  (ratio_gk : 6 * K = 15 * G)
  (R_value : R = 36) : 
  K = 105 :=
by
  sorry

end kiran_has_105_l95_95553


namespace inequality_has_no_solution_l95_95038

theorem inequality_has_no_solution (x : ℝ) : -x^2 + 2*x - 2 > 0 → false :=
by
  sorry

end inequality_has_no_solution_l95_95038


namespace abcd_solution_l95_95878

-- Define the problem statement
theorem abcd_solution (a b c d : ℤ) (h1 : a + c = -2) (h2 : a * c + b + d = 3) (h3 : a * d + b * c = 4) (h4 : b * d = -10) : 
  a + b + c + d = 1 := by 
  sorry

end abcd_solution_l95_95878


namespace polynomial_root_p_value_l95_95137

theorem polynomial_root_p_value (p : ℝ) : (3 : ℝ) ^ 3 + p * (3 : ℝ) - 18 = 0 → p = -3 :=
by
  intro h
  sorry

end polynomial_root_p_value_l95_95137


namespace harry_fish_count_l95_95012

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l95_95012


namespace number_of_real_roots_l95_95528

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then Real.exp x else -x^2 + 2.5 * x

theorem number_of_real_roots : ∃! x, f x = 0.5 * x + 1 :=
sorry

end number_of_real_roots_l95_95528


namespace sixth_term_geometric_mean_l95_95688

variable (a d : ℝ)

-- Define the arithmetic progression terms
def a_n (n : ℕ) := a + (n - 1) * d

-- Provided condition: second term is the geometric mean of the 1st and 4th terms
def condition (a d : ℝ) := a_n a d 2 = Real.sqrt (a_n a d 1 * a_n a d 4)

-- The goal to be proved: sixth term is the geometric mean of the 4th and 9th terms
theorem sixth_term_geometric_mean (a d : ℝ) (h : condition a d) : 
  a_n a d 6 = Real.sqrt (a_n a d 4 * a_n a d 9) :=
sorry

end sixth_term_geometric_mean_l95_95688


namespace measure_of_angle_A_l95_95398

noncomputable def angle_A (angle_B : ℝ) := 3 * angle_B - 40

theorem measure_of_angle_A (x : ℝ) (angle_A_parallel_B : true) (h : ∃ k : ℝ, (k = x ∧ (angle_A x = x ∨ angle_A x + x = 180))) :
  angle_A x = 20 ∨ angle_A x = 125 :=
by
  sorry

end measure_of_angle_A_l95_95398


namespace smallest_cost_l95_95602

def gift1_choc := 3
def gift1_caramel := 15
def price1 := 350

def gift2_choc := 20
def gift2_caramel := 5
def price2 := 500

def equal_candies (m n : ℕ) : Prop :=
  gift1_choc * m + gift2_choc * n = gift1_caramel * m + gift2_caramel * n

def total_cost (m n : ℕ) : ℕ :=
  price1 * m + price2 * n

theorem smallest_cost :
  ∃ m n : ℕ, equal_candies m n ∧ total_cost m n = 3750 :=
by {
  sorry
}

end smallest_cost_l95_95602


namespace find_x_floor_mult_eq_45_l95_95382

theorem find_x_floor_mult_eq_45 (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 45) : x = 7.5 :=
sorry

end find_x_floor_mult_eq_45_l95_95382


namespace sequence_a8_equals_neg2_l95_95420

theorem sequence_a8_equals_neg2 (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a n * a (n + 1) = -2) 
  : a 8 = -2 :=
sorry

end sequence_a8_equals_neg2_l95_95420


namespace cubic_expression_identity_l95_95902

theorem cubic_expression_identity (x : ℝ) (hx : x + 1/x = 8) : 
  x^3 + 1/x^3 = 332 :=
sorry

end cubic_expression_identity_l95_95902


namespace exponentiation_product_l95_95471

theorem exponentiation_product (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3 ^ a) ^ b = 3 ^ 3) : 3 ^ a * 3 ^ b = 3 ^ 4 :=
by
  sorry

end exponentiation_product_l95_95471


namespace larger_pie_crust_flour_l95_95304

theorem larger_pie_crust_flour
  (p1 p2 : ℕ)
  (f1 f2 c : ℚ)
  (h1 : p1 = 36)
  (h2 : p2 = 24)
  (h3 : f1 = 1 / 8)
  (h4 : p1 * f1 = c)
  (h5 : p2 * f2 = c)
  : f2 = 3 / 16 :=
sorry

end larger_pie_crust_flour_l95_95304


namespace paper_boat_travel_time_l95_95915

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l95_95915


namespace find_point_on_curve_l95_95864

theorem find_point_on_curve :
  ∃ P : ℝ × ℝ, (P.1^3 - P.1 + 3 = P.2) ∧ (3 * P.1^2 - 1 = 2) ∧ (P = (1, 3) ∨ P = (-1, 3)) :=
sorry

end find_point_on_curve_l95_95864


namespace krishna_fraction_wins_l95_95430

theorem krishna_fraction_wins (matches_total : ℕ) (callum_points : ℕ) (points_per_win : ℕ) (callum_wins : ℕ) :
  matches_total = 8 → callum_points = 20 → points_per_win = 10 → callum_wins = callum_points / points_per_win →
  (matches_total - callum_wins) / matches_total = 3 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end krishna_fraction_wins_l95_95430


namespace inverse_function_fixed_point_l95_95953

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 1

theorem inverse_function_fixed_point
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (f a y) = y) ∧ g 0 = 2 :=
sorry

end inverse_function_fixed_point_l95_95953


namespace pirates_gold_coins_l95_95626

theorem pirates_gold_coins (S a b c d e : ℕ) (h1 : a = S / 3) (h2 : b = S / 4) (h3 : c = S / 5) (h4 : d = S / 6) (h5 : e = 90) :
  S = 1800 :=
by
  -- Definitions and assumptions would go here
  sorry

end pirates_gold_coins_l95_95626


namespace find_a_l95_95115

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - y = 3) : a = 2 :=
by
  sorry

end find_a_l95_95115


namespace inequality_xyz_l95_95665

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * ((x ^ 3 + y ^ 3 + z ^ 3) ^ (1 / 3)) :=
by
  sorry

end inequality_xyz_l95_95665


namespace birds_in_house_l95_95911

theorem birds_in_house (B : ℕ) :
  let dogs := 3
  let cats := 18
  let humans := 7
  let total_heads := B + dogs + cats + humans
  let total_feet := 2 * B + 4 * dogs + 4 * cats + 2 * humans
  total_feet = total_heads + 74 → B = 4 :=
by
  intros dogs cats humans total_heads total_feet condition
  -- We assume the condition and work towards the proof.
  sorry

end birds_in_house_l95_95911


namespace probability_fourth_quadrant_l95_95885

open Set

def A : Set ℤ := {-2, 1, 2}
def B : Set ℤ := {-1, 1, 3}

def passes_fourth_quadrant (a b : ℤ) : Prop :=
  (a < 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0)

def successful_events : Finset (ℤ × ℤ) :=
  {(a, b) | a ∈ A ∧ b ∈ B ∧ passes_fourth_quadrant a b}.to_finset

def total_events : ℕ := (A.to_finset.card) * (B.to_finset.card)

def successful_event_count : ℕ := successful_events.card

theorem probability_fourth_quadrant :
  (successful_event_count : ℚ) / total_events = 5 / 9 := by
  sorry

end probability_fourth_quadrant_l95_95885


namespace cos_150_eq_neg_sqrt3_over_2_l95_95813

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l95_95813


namespace cos_150_eq_neg_half_l95_95847

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l95_95847


namespace root_of_quadratic_l95_95395

theorem root_of_quadratic (a : ℝ) (h : ∃ (x : ℝ), x = 0 ∧ x^2 + x + 2 * a - 1 = 0) : a = 1 / 2 := by
  sorry

end root_of_quadratic_l95_95395


namespace cos_150_eq_negative_cos_30_l95_95790

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l95_95790


namespace probability_top_card_is_joker_l95_95766

def deck_size : ℕ := 54
def joker_count : ℕ := 2

theorem probability_top_card_is_joker :
  (joker_count : ℝ) / (deck_size : ℝ) = 1 / 27 :=
by
  sorry

end probability_top_card_is_joker_l95_95766


namespace part_1_part_3_500_units_part_3_1000_units_l95_95757

/-- Define the pricing function P as per the given conditions -/
def P (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x <= 550 then 62 - 0.02 * x
  else 51

/-- Verify that ordering 550 units results in a per-unit price of 51 yuan -/
theorem part_1 : P 550 = 51 := sorry

/-- Compute profit for given order quantities -/
def profit (x : ℕ) : ℝ :=
  x * (P x - 40)

/-- Verify that an order of 500 units results in a profit of 6000 yuan -/
theorem part_3_500_units : profit 500 = 6000 := sorry

/-- Verify that an order of 1000 units results in a profit of 11000 yuan -/
theorem part_3_1000_units : profit 1000 = 11000 := sorry

end part_1_part_3_500_units_part_3_1000_units_l95_95757


namespace number_of_four_digit_numbers_l95_95272

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l95_95272


namespace value_of_expression_l95_95245

theorem value_of_expression (x y z : ℝ) (h : x * y * z = 1) :
  1 / (1 + x + x * y) + 1 / (1 + y + y * z) + 1 / (1 + z + z * x) = 1 :=
sorry

end value_of_expression_l95_95245


namespace jane_number_of_muffins_l95_95424

theorem jane_number_of_muffins 
    (m b c : ℕ) 
    (h1 : m + b + c = 6) 
    (h2 : b = 2) 
    (h3 : (50 * m + 75 * b + 65 * c) % 100 = 0) : 
    m = 4 := 
sorry

end jane_number_of_muffins_l95_95424


namespace speed_of_stream_l95_95074

theorem speed_of_stream (downstream_speed upstream_speed : ℕ) (h1 : downstream_speed = 12) (h2 : upstream_speed = 8) : 
  (downstream_speed - upstream_speed) / 2 = 2 :=
by
  sorry

end speed_of_stream_l95_95074


namespace percentage_return_is_25_l95_95206

noncomputable def percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ) : ℝ :=
  (dividend_rate / 100 * face_value / purchase_price) * 100

theorem percentage_return_is_25 :
  percentage_return_on_investment 18.5 50 37 = 25 := 
by
  sorry

end percentage_return_is_25_l95_95206


namespace potatoes_fraction_l95_95594

theorem potatoes_fraction (w : ℝ) (x : ℝ) (h_weight : w = 36) (h_fraction : w / x = 36) : x = 1 :=
by
  sorry

end potatoes_fraction_l95_95594


namespace prime_power_value_l95_95636

theorem prime_power_value (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h1 : Nat.Prime (7 * p + q)) (h2 : Nat.Prime (p * q + 11)) : 
  p ^ q = 8 ∨ p ^ q = 9 := 
sorry

end prime_power_value_l95_95636


namespace exists_solution_in_interal_l95_95109

theorem exists_solution_in_interal :
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ (t + ((t + 1 / Real.sqrt 3) / (1 - t / Real.sqrt 3)) + (3 * t - t^3) / (1 - 3 * (t^2)) = 0) :=
sorry

end exists_solution_in_interal_l95_95109


namespace fran_threw_away_80_pct_l95_95442

-- Definitions based on the conditions
def initial_votes_game_of_thrones := 10
def initial_votes_twilight := 12
def initial_votes_art_of_deal := 20
def altered_votes_twilight := initial_votes_twilight / 2
def new_total_votes := 2 * initial_votes_game_of_thrones

-- Theorem we are proving
theorem fran_threw_away_80_pct :
  ∃ x, x = 80 ∧
    new_total_votes = initial_votes_game_of_thrones + altered_votes_twilight + (initial_votes_art_of_deal * (1 - x / 100)) := by
  sorry

end fran_threw_away_80_pct_l95_95442


namespace remainder_of_n_div_7_l95_95516

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end remainder_of_n_div_7_l95_95516


namespace Cade_remaining_marbles_l95_95607

def initial_marbles := 87
def given_marbles := 8
def remaining_marbles := initial_marbles - given_marbles

theorem Cade_remaining_marbles : remaining_marbles = 79 := by
  sorry

end Cade_remaining_marbles_l95_95607


namespace Sarah_shampoo_conditioner_usage_l95_95696

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end Sarah_shampoo_conditioner_usage_l95_95696


namespace distance_to_place_l95_95357

theorem distance_to_place 
  (row_speed_still_water : ℝ) 
  (current_speed : ℝ) 
  (headwind_speed : ℝ) 
  (tailwind_speed : ℝ) 
  (total_trip_time : ℝ) 
  (htotal_trip_time : total_trip_time = 15) 
  (hrow_speed_still_water : row_speed_still_water = 10) 
  (hcurrent_speed : current_speed = 2) 
  (hheadwind_speed : headwind_speed = 4) 
  (htailwind_speed : tailwind_speed = 4) :
  ∃ (D : ℝ), D = 48 :=
by
  sorry

end distance_to_place_l95_95357


namespace range_of_a_l95_95884

def A : Set ℝ := { x | x^2 - x - 2 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - a) < 3 }

theorem range_of_a (a : ℝ) :
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by
  sorry

end range_of_a_l95_95884


namespace min_abs_val_sum_l95_95431

noncomputable def abs_val_sum_min : ℝ := (4:ℝ)^(1/3)

theorem min_abs_val_sum (a b c : ℝ) (h : |(a - b) * (b - c) * (c - a)| = 1) :
  |a| + |b| + |c| >= abs_val_sum_min :=
sorry

end min_abs_val_sum_l95_95431


namespace ratio_avg_speeds_l95_95103

-- Definitions based on the problem conditions
def distance_A_B := 600
def time_Eddy := 3
def distance_A_C := 460
def time_Freddy := 4

-- Definition of average speeds
def avg_speed_Eddy := distance_A_B / time_Eddy
def avg_speed_Freddy := distance_A_C / time_Freddy

-- Theorem statement
theorem ratio_avg_speeds : avg_speed_Eddy / avg_speed_Freddy = 40 / 23 := 
sorry

end ratio_avg_speeds_l95_95103


namespace chicken_feathers_after_crossing_l95_95454

def feathers_remaining_after_crossings (cars_dodged feathers_before pulling_factor : ℕ) : ℕ :=
  let feathers_lost := cars_dodged * pulling_factor
  feathers_before - feathers_lost

theorem chicken_feathers_after_crossing 
  (cars_dodged : ℕ := 23)
  (feathers_before : ℕ := 5263)
  (pulling_factor : ℕ := 2) :
  feathers_remaining_after_crossings cars_dodged feathers_before pulling_factor = 5217 :=
by
  sorry

end chicken_feathers_after_crossing_l95_95454
