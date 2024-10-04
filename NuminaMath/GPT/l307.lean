import Mathlib

namespace moles_of_H2O_formed_l307_307432

def NH4NO3 (n : ℕ) : Prop := n = 1
def NaOH (n : ℕ) : Prop := ∃ m : ℕ, m = n
def H2O (n : ℕ) : Prop := n = 1

theorem moles_of_H2O_formed :
  ∀ (n : ℕ), NH4NO3 1 → NaOH n → H2O 1 := 
by
  intros n hNH4NO3 hNaOH
  exact sorry

end moles_of_H2O_formed_l307_307432


namespace incorrect_statement_implies_m_eq_zero_l307_307776

theorem incorrect_statement_implies_m_eq_zero
  (m : ℝ)
  (y : ℝ → ℝ)
  (h : ∀ x, y x = m * x + 4 * m - 2)
  (intersects_y_axis_at : y 0 = -2) :
  m = 0 :=
sorry

end incorrect_statement_implies_m_eq_zero_l307_307776


namespace min_distance_to_line_l307_307062

theorem min_distance_to_line : 
  let A := 5
  let B := -3
  let C := 4
  let d (x₀ y₀ : ℤ) := (abs (A * x₀ + B * y₀ + C) : ℝ) / (Real.sqrt (A ^ 2 + B ^ 2))
  ∃ (x₀ y₀ : ℤ), d x₀ y₀ = Real.sqrt 34 / 85 := 
by 
  sorry

end min_distance_to_line_l307_307062


namespace max_weight_each_shipping_box_can_hold_l307_307147

noncomputable def max_shipping_box_weight_pounds 
  (total_plates : ℕ)
  (weight_per_plate_ounces : ℕ)
  (plates_removed : ℕ)
  (ounce_to_pound : ℕ) : ℕ :=
  (total_plates - plates_removed) * weight_per_plate_ounces / ounce_to_pound

theorem max_weight_each_shipping_box_can_hold :
  max_shipping_box_weight_pounds 38 10 6 16 = 20 :=
by
  sorry

end max_weight_each_shipping_box_can_hold_l307_307147


namespace discount_percentage_is_correct_l307_307019

noncomputable def cost_prices := [540, 660, 780]
noncomputable def markup_percentages := [0.15, 0.20, 0.25]
noncomputable def selling_prices := [496.80, 600, 750]

noncomputable def marked_price (cost : ℝ) (markup : ℝ) : ℝ := cost + (markup * cost)

noncomputable def total_marked_price : ℝ := 
  (marked_price 540 0.15) + (marked_price 660 0.20) + (marked_price 780 0.25)

noncomputable def total_selling_price : ℝ := 496.80 + 600 + 750

noncomputable def overall_discount_percentage : ℝ :=
  ((total_marked_price - total_selling_price) / total_marked_price) * 100

theorem discount_percentage_is_correct : overall_discount_percentage = 22.65 :=
by
  sorry

end discount_percentage_is_correct_l307_307019


namespace cassie_has_8_parrots_l307_307926

-- Define the conditions
def num_dogs : ℕ := 4
def nails_per_foot : ℕ := 4
def feet_per_dog : ℕ := 4
def nails_per_dog := nails_per_foot * feet_per_dog

def nails_total_dogs : ℕ := num_dogs * nails_per_dog

def claws_per_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def normal_claws_per_parrot := claws_per_leg * legs_per_parrot

def extra_toe_parrot_claws : ℕ := normal_claws_per_parrot + 1

def total_nails : ℕ := 113

-- Establishing the proof problem
theorem cassie_has_8_parrots : 
  ∃ (P : ℕ), (6 * (P - 1) + 7 = 49) ∧ P = 8 := by
  sorry

end cassie_has_8_parrots_l307_307926


namespace complex_solution_l307_307864

theorem complex_solution (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (Complex.mk a b)^2 = Complex.mk 3 4) :
  Complex.mk a b = Complex.mk 2 1 :=
sorry

end complex_solution_l307_307864


namespace totalShortBushes_l307_307504

namespace ProofProblem

def initialShortBushes : Nat := 37
def additionalShortBushes : Nat := 20

theorem totalShortBushes :
  initialShortBushes + additionalShortBushes = 57 := by
  sorry

end ProofProblem

end totalShortBushes_l307_307504


namespace kath_total_cost_l307_307671

def admission_cost : ℝ := 8
def discount_percentage_pre6pm : ℝ := 0.25
def discount_percentage_student : ℝ := 0.10
def time_of_movie : ℝ := 4
def num_people : ℕ := 6
def num_students : ℕ := 2

theorem kath_total_cost :
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1 -- remaining people (total - 2 students - Kath)
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  total_cost = 34.80 := by
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  sorry

end kath_total_cost_l307_307671


namespace pi_bounds_l307_307037

theorem pi_bounds :
  3 < Real.pi ∧ Real.pi < 4 :=
by
  sorry

end pi_bounds_l307_307037


namespace symmetry_axis_is_neg_pi_over_12_l307_307875

noncomputable def symmetry_axis_of_sine_function : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, (3 * x + 3 * Real.pi / 4 = Real.pi / 2 + k * Real.pi) ↔ (x = - Real.pi / 12 + k * Real.pi / 3)

theorem symmetry_axis_is_neg_pi_over_12 : symmetry_axis_of_sine_function := sorry

end symmetry_axis_is_neg_pi_over_12_l307_307875


namespace unfenced_side_length_l307_307940

-- Define the conditions
variables (L W : ℝ)
axiom area_condition : L * W = 480
axiom fence_condition : 2 * W + L = 64

-- Prove the unfenced side of the yard (L) is 40 feet
theorem unfenced_side_length : L = 40 :=
by
  -- Conditions, definitions, and properties go here.
  -- But we leave the proof as a placeholder since the statement is sufficient.
  sorry

end unfenced_side_length_l307_307940


namespace problem_statement_l307_307980

open Real

variables {f : ℝ → ℝ} {a b c : ℝ}

-- f is twice differentiable on ℝ
axiom hf : ∀ x : ℝ, Differentiable ℝ f
axiom hf' : ∀ x : ℝ, Differentiable ℝ (deriv f)

-- ∃ c ∈ ℝ, such that (f(b) - f(a)) / (b - a) ≠ f'(c) for all a ≠ b
axiom hc : ∃ c : ℝ, ∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c

-- Prove that f''(c) = 0
theorem problem_statement : ∃ c : ℝ, (∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c) → deriv (deriv f) c = 0 := sorry

end problem_statement_l307_307980


namespace original_cost_of_luxury_bag_l307_307906

theorem original_cost_of_luxury_bag (SP : ℝ) (profit_margin : ℝ) (original_cost : ℝ) 
  (h1 : SP = 3450) (h2 : profit_margin = 0.15) (h3 : SP = original_cost * (1 + profit_margin)) : 
  original_cost = 3000 :=
by
  sorry

end original_cost_of_luxury_bag_l307_307906


namespace number_of_valid_programs_l307_307914

-- Definitions based on the problem conditions
def courses := {'English, 'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Science}
def math_courses := {'Algebra, 'Geometry}
def remaining_courses := {'Algebra, 'Geometry, 'History, 'Art, 'Latin, 'Science}

-- Condition that the program includes English and at least two math courses.
def valid_program (program : Finset Char) : Prop := 
  'English ∈ program ∧ (math_courses ∩ program).card ≥ 2 ∧ program.card = 5

-- The main proof statement
theorem number_of_valid_programs : 
  (finset.univ.filter valid_program).card = 6 :=
sorry

end number_of_valid_programs_l307_307914


namespace right_triangle_largest_side_l307_307252

theorem right_triangle_largest_side (b d : ℕ) (h_triangle : (b - d)^2 + b^2 = (b + d)^2)
  (h_arith_seq : (b - d) < b ∧ b < (b + d))
  (h_perimeter : (b - d) + b + (b + d) = 840) :
  (b + d = 350) :=
by sorry

end right_triangle_largest_side_l307_307252


namespace probability_of_picking_same_color_shoes_l307_307189

theorem probability_of_picking_same_color_shoes
  (n_pairs_black : ℕ) (n_pairs_brown : ℕ) (n_pairs_gray : ℕ)
  (h_black_pairs : n_pairs_black = 8)
  (h_brown_pairs : n_pairs_brown = 4)
  (h_gray_pairs : n_pairs_gray = 3)
  (total_shoes : ℕ := 2 * (n_pairs_black + n_pairs_brown + n_pairs_gray)) :
  (16 / total_shoes * 8 / (total_shoes - 1) + 
   8 / total_shoes * 4 / (total_shoes - 1) + 
   6 / total_shoes * 3 / (total_shoes - 1)) = 89 / 435 :=
by
  sorry

end probability_of_picking_same_color_shoes_l307_307189


namespace variance_le_second_moment_l307_307023

noncomputable def variance (X : ℝ → ℝ) (MX : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - MX)^2]

noncomputable def second_moment (X : ℝ → ℝ) (C : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - C)^2]

theorem variance_le_second_moment (X : ℝ → ℝ) :
  ∀ C : ℝ, C ≠ MX → variance X MX ≤ second_moment X C := 
by
  sorry

end variance_le_second_moment_l307_307023


namespace three_digit_sum_seven_l307_307619

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l307_307619


namespace correct_options_l307_307141

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l307_307141


namespace equal_real_roots_of_quadratic_eq_l307_307966

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end equal_real_roots_of_quadratic_eq_l307_307966


namespace central_angle_of_sector_l307_307124

noncomputable def central_angle (l S r : ℝ) : ℝ :=
  2 * S / r^2

theorem central_angle_of_sector (r : ℝ) (h₁ : 4 * r / 2 = 4) (h₂ : r = 2) : central_angle 4 4 r = 2 :=
by
  sorry

end central_angle_of_sector_l307_307124


namespace complex_number_solution_l307_307648

open Complex

theorem complex_number_solution (z : ℂ) (h1 : ∥z∥ = Real.sqrt 2) (h2 : z + conj z = 2) :
  z = 1 + I ∨ z = 1 - I :=
  sorry

end complex_number_solution_l307_307648


namespace first_candidate_percentage_l307_307460

theorem first_candidate_percentage (total_votes : ℕ) (invalid_percentage : ℕ) (second_candidate_votes : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_invalid_percentage : invalid_percentage = 20) 
  (h_second_candidate_votes : second_candidate_votes = 2700) : 
  (100 * (total_votes * (1 - (invalid_percentage / 100)) - second_candidate_votes) / (total_votes * (1 - (invalid_percentage / 100)))) = 55 :=
by
  sorry

end first_candidate_percentage_l307_307460


namespace l_shaped_tile_rectangle_multiple_of_8_l307_307547

theorem l_shaped_tile_rectangle_multiple_of_8 (m n : ℕ) 
  (h : ∃ k : ℕ, 4 * k = m * n) : ∃ s : ℕ, m * n = 8 * s :=
by
  sorry

end l_shaped_tile_rectangle_multiple_of_8_l307_307547


namespace schedule_courses_l307_307256

-- Define the number of courses and periods
def num_courses : Nat := 4
def num_periods : Nat := 8

-- Define the total number of ways to schedule courses without restrictions
def unrestricted_schedules : Nat := Nat.choose num_periods num_courses * Nat.factorial num_courses

-- Define the number of invalid schedules using PIE (approximate value given in problem)
def invalid_schedules : Nat := 1008 + 180 + 120

-- Define the number of valid schedules
def valid_schedules : Nat := unrestricted_schedules - invalid_schedules

theorem schedule_courses : valid_schedules = 372 := sorry

end schedule_courses_l307_307256


namespace number_of_three_digit_numbers_with_sum_7_l307_307571

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l307_307571


namespace number_of_three_digit_numbers_with_sum_7_l307_307569

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l307_307569


namespace number_is_minus_72_l307_307897

noncomputable def find_number (x : ℝ) : Prop :=
  0.833 * x = -60

theorem number_is_minus_72 : ∃ x : ℝ, find_number x ∧ x = -72 :=
by
  sorry

end number_is_minus_72_l307_307897


namespace maria_remaining_towels_l307_307067

-- Define the number of green towels Maria bought
def greenTowels : ℕ := 58

-- Define the number of white towels Maria bought
def whiteTowels : ℕ := 43

-- Define the total number of towels Maria bought
def totalTowels : ℕ := greenTowels + whiteTowels

-- Define the number of towels Maria gave to her mother
def towelsGiven : ℕ := 87

-- Define the resulting number of towels Maria has
def remainingTowels : ℕ := totalTowels - towelsGiven

-- Prove that the remaining number of towels is 14
theorem maria_remaining_towels : remainingTowels = 14 :=
by
  sorry

end maria_remaining_towels_l307_307067


namespace train_passing_tree_time_l307_307258

theorem train_passing_tree_time
  (train_length : ℝ) (train_speed_kmhr : ℝ) (conversion_factor : ℝ)
  (train_speed_ms : train_speed_ms = train_speed_kmhr * conversion_factor) :
  train_length = 500 → train_speed_kmhr = 72 → conversion_factor = 5 / 18 →
  500 / (72 * (5 / 18)) = 25 := 
by
  intros h1 h2 h3
  sorry

end train_passing_tree_time_l307_307258


namespace half_abs_diff_of_squares_l307_307224

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l307_307224


namespace exists_graph_with_properties_l307_307484

theorem exists_graph_with_properties (n : ℕ) (N : ℕ)
  (h_n : n > 2) (h_N : N > 0) :
  ∃ (G : SimpleGraph ℕ), G.chromaticNumber = n ∧ G.vertexCount ≥ N ∧
  ∀ v, (G.removeVertex v).chromaticNumber = n - 1 :=
sorry

end exists_graph_with_properties_l307_307484


namespace find_x_of_floor_eq_72_l307_307563

theorem find_x_of_floor_eq_72 (x : ℝ) (hx_pos : 0 < x) (hx_eq : x * ⌊x⌋ = 72) : x = 9 :=
by 
  sorry

end find_x_of_floor_eq_72_l307_307563


namespace age_ratio_holds_l307_307711

variables (e s : ℕ)

-- Conditions based on the problem statement
def condition_1 : Prop := e - 3 = 2 * (s - 3)
def condition_2 : Prop := e - 5 = 3 * (s - 5)

-- Proposition to prove that in 1 year, the age ratio will be 3:2
def age_ratio_in_one_year : Prop := (e + 1) * 2 = (s + 1) * 3

theorem age_ratio_holds (h1 : condition_1 e s) (h2 : condition_2 e s) : age_ratio_in_one_year e s :=
by {
  sorry
}

end age_ratio_holds_l307_307711


namespace rationalize_sqrt_three_sub_one_l307_307028

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l307_307028


namespace solve_stream_speed_l307_307243

noncomputable def boat_travel (v : ℝ) : Prop :=
  let downstream_speed := 12 + v
  let upstream_speed := 12 - v
  let downstream_time := 60 / downstream_speed
  let upstream_time := 60 / upstream_speed
  upstream_time - downstream_time = 2

theorem solve_stream_speed : ∃ v : ℝ, boat_travel v ∧ v = 2.31 :=
by {
  sorry
}

end solve_stream_speed_l307_307243


namespace solve_for_k_l307_307963

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end solve_for_k_l307_307963


namespace half_abs_diff_squares_eq_40_l307_307229

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l307_307229


namespace ff1_is_1_l307_307838

noncomputable def f (x : ℝ) := Real.log x - 2 * x + 3

theorem ff1_is_1 : f (f 1) = 1 := by
  sorry

end ff1_is_1_l307_307838


namespace fill_pipe_fraction_l307_307734

theorem fill_pipe_fraction (x : ℝ) (h : x = 1 / 2) : x = 1 / 2 :=
by
  sorry

end fill_pipe_fraction_l307_307734


namespace todd_runs_faster_l307_307548

-- Define the times taken by Brian and Todd
def brian_time : ℕ := 96
def todd_time : ℕ := 88

-- The theorem stating the problem
theorem todd_runs_faster : brian_time - todd_time = 8 :=
by
  -- Solution here
  sorry

end todd_runs_faster_l307_307548


namespace paul_lost_crayons_l307_307478

theorem paul_lost_crayons :
  ∀ (initial_crayons given_crayons left_crayons lost_crayons : ℕ),
    initial_crayons = 1453 →
    given_crayons = 563 →
    left_crayons = 332 →
    lost_crayons = (initial_crayons - given_crayons) - left_crayons →
    lost_crayons = 558 :=
by
  intros initial_crayons given_crayons left_crayons lost_crayons
  intros h_initial h_given h_left h_lost
  sorry

end paul_lost_crayons_l307_307478


namespace center_of_symmetry_l307_307919

def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

/--
  Given the function f(x) := sin x - sqrt(3) * cos x,
  prove that (π/3, 0) is the center of symmetry for f.
-/
theorem center_of_symmetry : symmetry_center (fun x => Real.sin x - Real.sqrt 3 * Real.cos x) (Real.pi / 3, 0) :=
by
  sorry

end center_of_symmetry_l307_307919


namespace number_of_divisors_greater_than_22_l307_307207

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l307_307207


namespace vaccine_II_more_effective_l307_307355

noncomputable theory

structure VaccineResult where
  n : ℕ   -- number of people vaccinated
  infected : ℕ   -- number of people infected

def infection_rate : ℝ := 0.2

def vaccine_I_result : VaccineResult := { n := 8, infected := 0 }
def vaccine_II_result : VaccineResult := { n := 25, infected := 1 }

theorem vaccine_II_more_effective : 
  (vaccine_I_result.infected = 0 → P(vaccine_I_result) > P(vaccine_II_result)) → (vaccine_II_result.infected = 1 → P(vaccine_I_result) < P(vaccine_II_result)) → 
  (1 / (vaccine_I_result.n ^ infection_rate) < 1 / (vaccine_II_result.n ^ infection_rate)) :=
by
sorry

end vaccine_II_more_effective_l307_307355


namespace symmetric_line_eq_l307_307562

theorem symmetric_line_eq (x y : ℝ) (c : ℝ) (P : ℝ × ℝ)
  (h₁ : 3 * x - y - 4 = 0)
  (h₂ : P = (2, -1))
  (h₃ : 3 * x - y + c = 0)
  (h : 3 * 2 - (-1) + c = 0) : 
  c = -7 :=
by
  sorry

end symmetric_line_eq_l307_307562


namespace three_digit_numbers_sum_seven_l307_307635

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l307_307635


namespace half_abs_diff_of_squares_l307_307226

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l307_307226


namespace magician_earnings_l307_307737

theorem magician_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (decks_remaining : ℕ) (money_earned : ℕ) : 
    price_per_deck = 7 →
    initial_decks = 16 →
    decks_remaining = 8 →
    money_earned = (initial_decks - decks_remaining) * price_per_deck →
    money_earned = 56 :=
by
  intros hp hi hd he
  rw [hp, hi, hd] at he
  exact he

end magician_earnings_l307_307737


namespace inversely_proportional_x_y_l307_307041

-- Define the problem statement
theorem inversely_proportional_x_y (x y k : ℝ)
  (h1 : x * y = k)
  (h2 : x = 5)
  (h3 : y = 15) :
  let y' := -30 in
  let x' := -5/2 in
  (x' * y' = k) :=
by 
  sorry

#check inversely_proportional_x_y

end inversely_proportional_x_y_l307_307041


namespace towel_percentage_decrease_l307_307916

theorem towel_percentage_decrease (L B : ℝ) (hL: L > 0) (hB: B > 0) :
  let OriginalArea := L * B
  let NewLength := 0.8 * L
  let NewBreadth := 0.8 * B
  let NewArea := NewLength * NewBreadth
  let PercentageDecrease := ((OriginalArea - NewArea) / OriginalArea) * 100
  PercentageDecrease = 36 :=
by
  sorry

end towel_percentage_decrease_l307_307916


namespace equal_roots_polynomial_l307_307434

open ComplexConjugate

theorem equal_roots_polynomial (k : ℚ) :
  (3 : ℚ) * x^2 - k * x + 2 * x + (12 : ℚ) = 0 → 
  (b : ℚ) ^ 2 - 4 * (3 : ℚ) * (12 : ℚ) = 0 ↔ k = -10 ∨ k = 14 :=
by
  sorry

end equal_roots_polynomial_l307_307434


namespace division_of_polynomial_l307_307748

theorem division_of_polynomial (a : ℤ) : (-28 * a^3) / (7 * a) = -4 * a^2 := by
  sorry

end division_of_polynomial_l307_307748


namespace three_digit_sum_seven_l307_307601

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l307_307601


namespace book_cost_is_2_l307_307482

-- Define initial amount of money
def initial_amount : ℕ := 48

-- Define the number of books purchased
def num_books : ℕ := 5

-- Define the amount of money left after purchasing the books
def amount_left : ℕ := 38

-- Define the cost per book
def cost_per_book (initial amount_left : ℕ) (num_books : ℕ) : ℕ := (initial - amount_left) / num_books

-- The theorem to prove
theorem book_cost_is_2
    (initial_amount : ℕ := 48) 
    (amount_left : ℕ := 38) 
    (num_books : ℕ := 5) :
    cost_per_book initial_amount amount_left num_books = 2 :=
by
  sorry

end book_cost_is_2_l307_307482


namespace find_n_l307_307760

theorem find_n (n x y k : ℕ) (h_coprime : Nat.gcd x y = 1) (h_eq : 3^n = x^k + y^k) : n = 2 :=
sorry

end find_n_l307_307760


namespace circles_area_sum_l307_307366

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l307_307366


namespace cannot_be_sum_of_four_consecutive_even_integers_l307_307066

-- Define what it means to be the sum of four consecutive even integers
def sum_of_four_consecutive_even_integers (n : ℤ) : Prop :=
  ∃ m : ℤ, n = 4 * m + 12 ∧ m % 2 = 0

-- State the problem in Lean 4
theorem cannot_be_sum_of_four_consecutive_even_integers :
  ¬ sum_of_four_consecutive_even_integers 32 ∧
  ¬ sum_of_four_consecutive_even_integers 80 ∧
  ¬ sum_of_four_consecutive_even_integers 104 ∧
  ¬ sum_of_four_consecutive_even_integers 200 :=
by
  sorry

end cannot_be_sum_of_four_consecutive_even_integers_l307_307066


namespace range_of_a_l307_307874

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) := by
  sorry

end range_of_a_l307_307874


namespace tens_digit_23_1987_l307_307278

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l307_307278


namespace cost_price_of_table_l307_307393

theorem cost_price_of_table (CP : ℝ) (SP : ℝ) (h1 : SP = CP * 1.10) (h2 : SP = 8800) : CP = 8000 :=
by
  sorry

end cost_price_of_table_l307_307393


namespace distributor_cost_l307_307082

variable (C : ℝ) -- Cost of the item for the distributor
variable (P_observed : ℝ) -- Observed price
variable (commission_rate : ℝ) -- Commission rate
variable (profit_rate : ℝ) -- Desired profit rate

-- Conditions
def is_observed_price_correct (C : ℝ) (P_observed : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) : Prop :=
  let SP := C * (1 + profit_rate)
  let observed := SP * (1 - commission_rate)
  observed = P_observed

-- The proof goal
theorem distributor_cost (h : is_observed_price_correct C 30 0.20 0.20) : C = 31.25 := sorry

end distributor_cost_l307_307082


namespace ratio_EG_GD_l307_307004

theorem ratio_EG_GD (a EG GD : ℝ)
  (h1 : EG = 4 * GD)
  (gcd_1 : Int.gcd 4 1 = 1) :
  4 + 1 = 5 := by
  sorry

end ratio_EG_GD_l307_307004


namespace problem_statement_l307_307494

theorem problem_statement
  (c d : ℕ)
  (h_factorization : ∀ x, x^2 - 18 * x + 72 = (x - c) * (x - d))
  (h_c_nonnegative : c ≥ 0)
  (h_d_nonnegative : d ≥ 0)
  (h_c_greater_d : c > d) :
  4 * d - c = 12 :=
sorry

end problem_statement_l307_307494


namespace geometric_sequence_properties_l307_307789

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ) (h : r ≠ 0)
  (h1 : a = r * (-1))
  (h2 : b = r * a)
  (h3 : c = r * b)
  (h4 : -9 = r * c) :
  b = -3 ∧ a * c = 9 :=
by sorry

end geometric_sequence_properties_l307_307789


namespace sample_frequency_in_range_l307_307406

theorem sample_frequency_in_range :
  let total_capacity := 100
  let freq_0_10 := 12
  let freq_10_20 := 13
  let freq_20_30 := 24
  let freq_30_40 := 15
  (freq_0_10 + freq_10_20 + freq_20_30 + freq_30_40) / total_capacity = 0.64 :=
by
  sorry

end sample_frequency_in_range_l307_307406


namespace bob_mother_twice_age_2040_l307_307022

theorem bob_mother_twice_age_2040 :
  ∀ (bob_age_2010 mother_age_2010 : ℕ), 
  bob_age_2010 = 10 ∧ mother_age_2010 = 50 →
  ∃ (x : ℕ), (mother_age_2010 + x = 2 * (bob_age_2010 + x)) ∧ (2010 + x = 2040) :=
by
  sorry

end bob_mother_twice_age_2040_l307_307022


namespace find_expression_value_l307_307657

theorem find_expression_value (m: ℝ) (h: m^2 - 2 * m - 1 = 0) : 
  (m - 1)^2 - (m - 3) * (m + 3) - (m - 1) * (m - 3) = 6 := 
by 
  sorry

end find_expression_value_l307_307657


namespace pencils_per_pack_l307_307754

def packs := 28
def rows := 42
def pencils_per_row := 16

theorem pencils_per_pack (total_pencils : ℕ) : total_pencils = rows * pencils_per_row → total_pencils / packs = 24 :=
by
  sorry

end pencils_per_pack_l307_307754


namespace find_t_l307_307299

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def tangent_point (t : ℝ) : ℝ × ℝ := (t, 0)

theorem find_t :
  (∀ (A : ℝ × ℝ), ellipse_eq A.1 A.2 → 
    ∃ (C : ℝ × ℝ),
      tangent_point 2 = C ∧
      -- C is tangent to the extended line of F1A
      -- C is tangent to the extended line of F1F2
      -- C is tangent to segment AF2
      true
  ) :=
sorry

end find_t_l307_307299


namespace range_of_m_l307_307971

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m^2 > 0) ↔ -1 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l307_307971


namespace cody_needs_total_steps_l307_307927

theorem cody_needs_total_steps 
  (weekly_steps : ℕ → ℕ)
  (h1 : ∀ n, weekly_steps n = (n + 1) * 1000 * 7)
  (h2 : 4 * 7 * 1000 + 3 * 7 * 1000 + 2 * 7 * 1000 + 1 * 7 * 1000 = 70000) 
  (h3 : 70000 + 30000 = 100000) :
  ∃ total_steps, total_steps = 100000 := 
by
  sorry

end cody_needs_total_steps_l307_307927


namespace tens_digit_of_23_pow_1987_l307_307277

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l307_307277


namespace side_length_of_square_l307_307490

theorem side_length_of_square (d : ℝ) (h : d = 4) : ∃ (s : ℝ), s = 2 * Real.sqrt 2 :=
by
  sorry

end side_length_of_square_l307_307490


namespace geometric_sequence_values_l307_307956

theorem geometric_sequence_values (l a b c : ℝ) (h : ∃ r : ℝ, a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : b = -3 ∧ a * c = 9 :=
by
  sorry

end geometric_sequence_values_l307_307956


namespace three_digit_integers_sum_to_7_l307_307587

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l307_307587


namespace number_of_three_digit_numbers_with_digit_sum_seven_l307_307627

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l307_307627


namespace sum_ratio_l307_307469

variable {α : Type _} [LinearOrderedField α]

def geometric_sequence (a₁ q : α) : ℕ → α
| 0       => a₁
| (n + 1) => (geometric_sequence a₁ q n) * q

noncomputable def sum_geometric (a₁ q : α) (n : ℕ) : α :=
  if q = 1 then a₁ * (n + 1)
  else a₁ * (1 - q^(n + 1)) / (1 - q)

theorem sum_ratio {a₁ q : α} (h : 8 * (geometric_sequence a₁ q 1) + (geometric_sequence a₁ q 4) = 0) :
  (sum_geometric a₁ q 4) / (sum_geometric a₁ q 1) = -11 :=
sorry

end sum_ratio_l307_307469


namespace find_angle_l307_307921

-- Define the conditions
def circles_intersect (r : ℝ) : Prop :=
  r > 0

def area_shaded (α r : ℝ) : ℝ :=
  (α * r^2 / 2) - (1 / 2 * r^2 * (Real.sin α))

def angle_condition (α : ℝ) : Prop :=
  α - Real.sin(α) = (4 * Real.pi / 3)

-- Define the constant 2.6053
def alpha_value : ℝ := 2.6053

theorem find_angle (r : ℝ) (hr : circles_intersect r) :
  ∃ α, angle_condition α ∧ (α = alpha_value) :=
by
  sorry

end find_angle_l307_307921


namespace price_of_each_shirt_l307_307177

theorem price_of_each_shirt 
  (toys_cost : ℕ := 3 * 10)
  (cards_cost : ℕ := 2 * 5)
  (total_spent : ℕ := 70)
  (remaining_cost: ℕ := total_spent - (toys_cost + cards_cost))
  (num_shirts : ℕ := 3 + 2) :
  (remaining_cost / num_shirts) = 6 :=
by
  sorry

end price_of_each_shirt_l307_307177


namespace mutual_acquainted_or_unacquainted_l307_307185

theorem mutual_acquainted_or_unacquainted :
  ∀ (G : SimpleGraph (Fin 6)), 
  ∃ (V : Finset (Fin 6)), V.card = 3 ∧ ((∀ (u v : Fin 6), u ∈ V → v ∈ V → G.Adj u v) ∨ (∀ (u v : Fin 6), u ∈ V → v ∈ V → ¬G.Adj u v)) :=
by
  sorry

end mutual_acquainted_or_unacquainted_l307_307185


namespace angle_A_in_quadrilateral_l307_307759

noncomputable def degree_measure_A (A B C D : ℝ) := A

theorem angle_A_in_quadrilateral 
  (A B C D : ℝ)
  (hA : A = 3 * B)
  (hC : A = 4 * C)
  (hD : A = 6 * D)
  (sum_angles : A + B + C + D = 360) :
  degree_measure_A A B C D = 206 :=
by
  sorry

end angle_A_in_quadrilateral_l307_307759


namespace total_puff_pastries_l307_307245

theorem total_puff_pastries (batches trays puff_pastry volunteers : ℕ) 
  (h_batches : batches = 1) 
  (h_trays : trays = 8) 
  (h_puff_pastry : puff_pastry = 25) 
  (h_volunteers : volunteers = 1000) : 
  (volunteers * trays * puff_pastry) = 200000 := 
by 
  have h_total_trays : volunteers * trays = 1000 * 8 := by sorry
  have h_total_puff_pastries_per_volunteer : trays * puff_pastry = 8 * 25 := by sorry
  have h_total_puff_pastries : volunteers * trays * puff_pastry = 1000 * 8 * 25 := by sorry
  sorry

end total_puff_pastries_l307_307245


namespace equal_real_roots_of_quadratic_eq_l307_307969

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end equal_real_roots_of_quadratic_eq_l307_307969


namespace remainder_sum_l307_307797

theorem remainder_sum (n : ℤ) : ((7 - n) + (n + 3)) % 7 = 3 :=
sorry

end remainder_sum_l307_307797


namespace square_area_proof_l307_307743

   theorem square_area_proof (x : ℝ) (h1 : 4 * x - 15 = 20 - 3 * x) :
     (20 - 3 * x) * (4 * x - 15) = 25 :=
   by
     sorry
   
end square_area_proof_l307_307743


namespace train_distance_difference_l307_307059

theorem train_distance_difference (t : ℝ) (D₁ D₂ : ℝ)
(h_speed1 : D₁ = 20 * t)
(h_speed2 : D₂ = 25 * t)
(h_total_dist : D₁ + D₂ = 540) :
  D₂ - D₁ = 60 :=
by {
  -- These are the conditions as stated in step c)
  sorry
}

end train_distance_difference_l307_307059


namespace express_in_scientific_notation_l307_307192

theorem express_in_scientific_notation (x : ℝ) (h : x = 720000) : x = 7.2 * 10^5 :=
by sorry

end express_in_scientific_notation_l307_307192


namespace count_positive_integers_l307_307566

theorem count_positive_integers (x : ℤ) : 
  (25 < x^2 + 6 * x + 8) → (x^2 + 6 * x + 8 < 50) → (x > 0) → (x = 3 ∨ x = 4) :=
by sorry

end count_positive_integers_l307_307566


namespace solve_equation_l307_307487

theorem solve_equation (x : ℚ) :
  (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by sorry

end solve_equation_l307_307487


namespace girls_ratio_correct_l307_307506

-- Define the number of total attendees
def total_attendees : ℕ := 100

-- Define the percentage of faculty and staff
def faculty_staff_percentage : ℕ := 10

-- Define the number of boys among the students
def number_of_boys : ℕ := 30

-- Define the function to calculate the number of faculty and staff
def faculty_staff (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  (faculty_staff_percentage * total_attendees) / 100

-- Define the function to calculate the number of students
def number_of_students (total_attendees faculty_staff_percentage: ℕ) : ℕ :=
  total_attendees - faculty_staff total_attendees faculty_staff_percentage

-- Define the function to calculate the number of girls
def number_of_girls (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℕ :=
  number_of_students total_attendees faculty_staff_percentage - number_of_boys

-- Define the function to calculate the ratio of girls to the remaining attendees
def ratio_girls_to_attendees (total_attendees faculty_staff_percentage number_of_boys: ℕ) : ℚ :=
  (number_of_girls total_attendees faculty_staff_percentage number_of_boys) / 
  (number_of_students total_attendees faculty_staff_percentage)

-- The theorem statement that needs to be proven (no proof required)
theorem girls_ratio_correct : ratio_girls_to_attendees total_attendees faculty_staff_percentage number_of_boys = 2 / 3 := 
by 
  -- The proof is skipped.
  sorry

end girls_ratio_correct_l307_307506


namespace ab_value_l307_307688

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5 / 8) : ab = (Real.sqrt 3) / 4 :=
by
  sorry

end ab_value_l307_307688


namespace proposition_D_l307_307696

theorem proposition_D (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by {
    sorry
}

end proposition_D_l307_307696


namespace trajectory_eqn_l307_307943

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Conditions given in the problem
def PA_squared (P : ℝ × ℝ) : ℝ := (P.1 + 1)^2 + P.2^2
def PB_squared (P : ℝ × ℝ) : ℝ := (P.1 - 1)^2 + P.2^2

-- The main statement to prove
theorem trajectory_eqn (P : ℝ × ℝ) (h : PA_squared P = 3 * PB_squared P) : 
  P.1^2 + P.2^2 - 4 * P.1 + 1 = 0 :=
by 
  sorry

end trajectory_eqn_l307_307943


namespace rationalize_denominator_correct_l307_307027

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l307_307027


namespace pentagon_PT_value_l307_307674

-- Given conditions
def length_QR := 3
def length_RS := 3
def length_ST := 3
def angle_T := 90
def angle_P := 120
def angle_Q := 120
def angle_R := 120

-- The target statement to prove
theorem pentagon_PT_value (a b : ℝ) (h : PT = a + 3 * Real.sqrt b) : a + b = 6 :=
sorry

end pentagon_PT_value_l307_307674


namespace count_arithmetic_sequence_l307_307788

theorem count_arithmetic_sequence :
  ∃ n, 195 - (n - 1) * 3 = 12 ∧ n = 62 :=
by {
  sorry
}

end count_arithmetic_sequence_l307_307788


namespace projectile_max_height_l307_307089

theorem projectile_max_height :
  ∀ (t : ℝ), -12 * t^2 + 72 * t + 45 ≤ 153 :=
by
  sorry

end projectile_max_height_l307_307089


namespace problem_conditions_l307_307824

def sequence_s_n (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def sequence_b_n (s : ℕ → ℝ) (n : ℕ) : ℝ := ∏ i in finset.range n, s i

theorem problem_conditions (a : ℕ → ℝ) (s : ℕ → ℝ) (b : ℕ → ℝ) (hS : ∀ n, s n = sequence_s_n a n)
  (hb : ∀ n, b n = sequence_b_n s n) (h : ∀ n, 2 / s n + 1 / b n = 2) :
  (∃ a_0 : ℕ → ℝ, a_0 1 = 3/2 ∧ (∀ n, n ≥ 2 → a_0 n = -1 / ((n) * (n + 1)))) ∧ 
  (∃ b_0 : ℕ → ℝ, (∀ n, b_0 n = 3/2 + (n - 1) / 2) ∧ has_arithmetic_diff b_0 (1/2)) := by sorry

noncomputable def has_arithmetic_diff (b : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, b (n + 1) - b n = d

end problem_conditions_l307_307824


namespace petes_original_number_l307_307479

theorem petes_original_number (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y - 5) (h3 : 3 * z = 96) :
  x = 12.33 :=
by
  -- Proof goes here
  sorry

end petes_original_number_l307_307479


namespace find_expression_value_l307_307284

theorem find_expression_value : 1 + 2 * 3 - 4 + 5 = 8 :=
by
  sorry

end find_expression_value_l307_307284


namespace max_chocolates_l307_307880

theorem max_chocolates (b c k : ℕ) (h1 : b + c = 36) (h2 : c = k * b) (h3 : k > 0) : b ≤ 18 :=
sorry

end max_chocolates_l307_307880


namespace x_y_sum_l307_307306

theorem x_y_sum (x y : ℝ) (h1 : |x| - 2 * x + y = 1) (h2 : x - |y| + y = 8) :
  x + y = 17 ∨ x + y = 1 :=
by
  sorry

end x_y_sum_l307_307306


namespace avg_speed_additional_hours_l307_307900

/-- Definitions based on the problem conditions -/
def first_leg_speed : ℕ := 30 -- miles per hour
def first_leg_time : ℕ := 6 -- hours
def total_trip_time : ℕ := 8 -- hours
def total_avg_speed : ℕ := 34 -- miles per hour

/-- The theorem that ties everything together -/
theorem avg_speed_additional_hours : 
  ((total_avg_speed * total_trip_time) - (first_leg_speed * first_leg_time)) / (total_trip_time - first_leg_time) = 46 := 
sorry

end avg_speed_additional_hours_l307_307900


namespace log_base_9_of_729_l307_307556

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end log_base_9_of_729_l307_307556


namespace min_value_of_fraction_l307_307667

theorem min_value_of_fraction (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : (m * (-3) + n * (-1) + 2 = 0)) 
    (h4 : (m * (-2) + n * 0 + 2 = 0)) : 
    (1 / m + 3 / n) = 6 :=
by
  sorry

end min_value_of_fraction_l307_307667


namespace julia_download_songs_l307_307011

-- Basic definitions based on conditions
def internet_speed_MBps : ℕ := 20
def song_size_MB : ℕ := 5
def half_hour_seconds : ℕ := 30 * 60

-- Statement of the proof problem
theorem julia_download_songs : 
  (internet_speed_MBps * half_hour_seconds) / song_size_MB = 7200 :=
by
  sorry

end julia_download_songs_l307_307011


namespace sum_of_remainders_is_six_l307_307508

theorem sum_of_remainders_is_six (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
  (a + b + c) % 15 = 6 :=
by
  sorry

end sum_of_remainders_is_six_l307_307508


namespace lateral_surface_area_of_pyramid_inscribed_in_sphere_l307_307910
-- Importing the entire Mathlib library to ensure all necessary definitions and theorems are available.

-- Formulate the problem as a Lean statement.

theorem lateral_surface_area_of_pyramid_inscribed_in_sphere :
  let R := (1 : ℝ)
  let theta := (45 : ℝ) * Real.pi / 180 -- Convert degrees to radians.
  -- Assuming the pyramid is regular and quadrilateral, inscribed in a sphere of radius 1
  ∃ S : ℝ, S = 4 :=
  sorry

end lateral_surface_area_of_pyramid_inscribed_in_sphere_l307_307910


namespace negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l307_307497

-- Definition of a triangle with a property on the angles.
def triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c

-- Definition of an obtuse angle.
def obtuse (x : ℝ) : Prop := x > 90

-- Proposition: In a triangle, at most one angle is obtuse.
def at_most_one_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a → ¬ obtuse b ∧ ¬ obtuse c) ∧ (obtuse b → ¬ obtuse a ∧ ¬ obtuse c) ∧ (obtuse c → ¬ obtuse a ∧ ¬ obtuse b)

-- Negation: In a triangle, there are at least two obtuse angles.
def at_least_two_obtuse (a b c : ℝ) : Prop := 
  triangle a b c ∧ (obtuse a ∧ obtuse b) ∨ (obtuse a ∧ obtuse c) ∨ (obtuse b ∧ obtuse c)

-- Prove the negation equivalence
theorem negation_of_at_most_one_obtuse_is_at_least_two_obtuse (a b c : ℝ) :
  (¬ at_most_one_obtuse a b c) ↔ at_least_two_obtuse a b c :=
sorry

end negation_of_at_most_one_obtuse_is_at_least_two_obtuse_l307_307497


namespace count_good_divisors_l307_307217

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l307_307217


namespace greatest_int_with_gcd_of_24_eq_2_l307_307717

theorem greatest_int_with_gcd_of_24_eq_2 (n : ℕ) (h1 : n < 200) (h2 : Int.gcd n 24 = 2) : n = 194 := 
sorry

end greatest_int_with_gcd_of_24_eq_2_l307_307717


namespace partition_into_triples_l307_307138

open Finset

theorem partition_into_triples (M : Finset ℕ) (hM : M = {1, 2, 3, ..., 15}) :
  ∃ A B C : Finset ℕ,
      ∃ D E : Finset ℕ,
        M = A ∪ B ∪ C ∪ D ∪ E ∧
        A.card = 3 ∧ B.card = 3 ∧ C.card = 3 ∧ D.card = 3 ∧ E.card = 3 ∧
        A.sum id = 24 ∧ B.sum id = 24 ∧ C.sum id = 24 ∧ D.sum id = 24 ∧ E.sum id = 24 := 
begin
  sorry
end

end partition_into_triples_l307_307138


namespace max_min_x_sub_2y_l307_307944

theorem max_min_x_sub_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 0 ≤ x - 2*y ∧ x - 2*y ≤ 10 :=
sorry

end max_min_x_sub_2y_l307_307944


namespace det_min_matrix_l307_307468

open Matrix

def min (a b : ℕ) : ℕ := if a ≤ b then a else b

noncomputable def compute_det_matrix (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  λ i j, 1 / (min i j).val

theorem det_min_matrix (n : ℕ) : 
  (compute_det_matrix n).det = (-1)^(n - 1) / (n - 1)! / n! :=
sorry

end det_min_matrix_l307_307468


namespace distance_travelled_downstream_l307_307048

theorem distance_travelled_downstream :
  let speed_boat_still_water := 42 -- km/hr
  let rate_current := 7 -- km/hr
  let time_travelled_min := 44 -- minutes
  let time_travelled_hrs := time_travelled_min / 60.0 -- converting minutes to hours
  let effective_speed_downstream := speed_boat_still_water + rate_current -- km/hr
  let distance_downstream := effective_speed_downstream * time_travelled_hrs
  distance_downstream = 35.93 :=
by
  -- Proof will go here
  sorry

end distance_travelled_downstream_l307_307048


namespace max_value_k_l307_307240

noncomputable def max_k (S : Finset ℕ) (A : ℕ → Finset ℕ) (k : ℕ) :=
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2)

theorem max_value_k : ∀ (S : Finset ℕ) (A : ℕ → Finset ℕ), 
  S = Finset.range 14 \{0} → 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2) →
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) →
  ∃ k, max_k S A k ∧ k = 4 :=
sorry

end max_value_k_l307_307240


namespace valid_numbers_count_is_7_l307_307955

-- Definitions based on the problem statement
def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 0

def last_two_digits_div_by_4_eq_1 (n : ℕ) : Prop :=
  (n % 100) % 4 = 1

def nine_digit_number (n : ℕ) : Prop :=
  10^8 ≤ n ∧ n < 10^9 ∧ ∀ d ∈ digits 10 n, is_valid_digit d

def count_valid_numbers : ℕ :=
  (Finset.range (10^9)).filter (λ n, nine_digit_number n ∧ last_two_digits_div_by_4_eq_1 n).card

-- Statement to prove
theorem valid_numbers_count_is_7 : count_valid_numbers = 7 :=
by
  sorry

end valid_numbers_count_is_7_l307_307955


namespace tens_digit_of_23_pow_1987_l307_307276

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l307_307276


namespace div3_of_div9_l307_307689

theorem div3_of_div9 (u v : ℤ) (h : 9 ∣ (u^2 + u * v + v^2)) : 3 ∣ u ∧ 3 ∣ v :=
sorry

end div3_of_div9_l307_307689


namespace product_even_probability_l307_307865

theorem product_even_probability : 
  let S := {x // 4 ≤ x ∧ x ≤ 20 ∧ x ≠ 13}
  let total_choices := (S.card.choose 2)
  let even_product_choices := (S.filter (λ p, p.1 * p.2 % 2 = 0)).card
  total_choices = nat.choose 16 2 ∧ even_product_choices = 99 
  → even_product_choices / total_choices = 33 / 40 :=
by
  sorry

end product_even_probability_l307_307865


namespace pen_price_equation_l307_307913

theorem pen_price_equation
  (x y : ℤ)
  (h1 : 100 * x - y = 100)
  (h2 : 2 * y - 100 * x = 200) : x = 4 :=
by
  sorry

end pen_price_equation_l307_307913


namespace binomial_coeff_sum_l307_307946

theorem binomial_coeff_sum :
  ∀ a b : ℝ, 15 * a^4 * b^2 = 135 ∧ 6 * a^5 * b = -18 →
  (a + b) ^ 6 = 64 :=
by
  intros a b h
  sorry

end binomial_coeff_sum_l307_307946


namespace evaluate_expression_l307_307430

theorem evaluate_expression :
  (18 : ℝ) / (14 * 5.3) = (1.8 : ℝ) / 7.42 :=
by
  sorry

end evaluate_expression_l307_307430


namespace bananas_used_l307_307109

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l307_307109


namespace insufficient_pharmacies_l307_307532

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l307_307532


namespace no_value_of_b_valid_l307_307266

theorem no_value_of_b_valid (b n : ℤ) : b^2 + 3 * b + 1 ≠ n^2 := by
  sorry

end no_value_of_b_valid_l307_307266


namespace number_of_good_numbers_l307_307213

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l307_307213


namespace rectangle_ratio_l307_307871

noncomputable def ratio_of_length_to_width (w : ℝ) : ℝ :=
  40 / w

theorem rectangle_ratio (w : ℝ) 
  (hw1 : 35 * (w + 5) = 40 * w + 75) : 
  ratio_of_length_to_width w = 2 :=
by
  sorry

end rectangle_ratio_l307_307871


namespace bird_weights_l307_307260

variables (A B V G : ℕ)

theorem bird_weights : 
  A + B + V + G = 32 ∧ 
  V < G ∧ 
  V + G < B ∧ 
  A < V + B ∧ 
  G + B < A + V 
  → 
  (A = 13 ∧ V = 4 ∧ G = 5 ∧ B = 10) :=
sorry

end bird_weights_l307_307260


namespace rectangle_area_l307_307514

theorem rectangle_area (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 4) :
  l * w = 8 / 9 :=
by
  sorry

end rectangle_area_l307_307514


namespace set_theory_problem_l307_307894

def U : Set ℤ := {x ∈ Set.univ | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_theory_problem : 
  (A ∩ B = {4}) ∧ 
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧ 
  (U \ (A ∪ C) = {6, 8, 10}) ∧ 
  ((U \ A) ∩ (U \ B) = {3}) := 
by 
  sorry

end set_theory_problem_l307_307894


namespace prob_relatively_prime_42_l307_307882

noncomputable def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ i => Nat.gcd i n = 1).length

theorem prob_relatively_prime_42 : 
  (euler_totient 42 : ℚ) / 42 = 2 / 7 := 
by
  sorry

end prob_relatively_prime_42_l307_307882


namespace B_power_15_minus_3_B_power_14_l307_307681

def B : Matrix (Fin 2) (Fin 2) ℝ := !!
  [3, 4]
  [0, 2]

theorem B_power_15_minus_3_B_power_14 :
  B^15 - 3 • B^14 = !!
    [0, 4]
    [0, -1] := by
  sorry

end B_power_15_minus_3_B_power_14_l307_307681


namespace distinct_natural_numbers_circles_sum_equal_impossible_l307_307815

theorem distinct_natural_numbers_circles_sum_equal_impossible :
  ¬∃ (f : ℕ → ℕ) (distinct : ∀ i j, i ≠ j → f i ≠ f j) (equal_sum : ∀ i j k, (f i + f j + f k = f (i+1) + f (j+1) + f (k+1))),
  true :=
  sorry

end distinct_natural_numbers_circles_sum_equal_impossible_l307_307815


namespace part1_part2_l307_307171

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part (1)
theorem part1 (a : ℝ) : (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1 / 3) :=
sorry

-- Part (2)
theorem part2 (a x : ℝ) :
  (a ≠ 0 → ( a > 0 ↔ -1/a < x ∧ x < 1)
  ∧ (a = 0 ↔ x < 1)
  ∧ (-1 < a ∧ a < 0 ↔ x < 1 ∨ x > -1/a)
  ∧ (a = -1 ↔ x ≠ 1)
  ∧ (a < -1 ↔ x < -1/a ∨ x > 1)) :=
sorry

end part1_part2_l307_307171


namespace bottles_from_Shop_C_l307_307553

theorem bottles_from_Shop_C (TotalBottles ShopA ShopB ShopC : ℕ) 
  (h1 : TotalBottles = 550) 
  (h2 : ShopA = 150) 
  (h3 : ShopB = 180) 
  (h4 : TotalBottles = ShopA + ShopB + ShopC) : 
  ShopC = 220 := 
by
  sorry

end bottles_from_Shop_C_l307_307553


namespace interest_rate_10_percent_l307_307158

-- Definitions for the problem
variables (P : ℝ) (R : ℝ) (T : ℝ)

-- Condition that the money doubles in 10 years on simple interest
def money_doubles_in_10_years (P R : ℝ) : Prop :=
  P = (P * R * 10) / 100

-- Statement that R is 10% if the money doubles in 10 years
theorem interest_rate_10_percent {P : ℝ} (h : money_doubles_in_10_years P R) : R = 10 :=
by
  sorry

end interest_rate_10_percent_l307_307158


namespace largest_five_digit_number_with_product_l307_307060

theorem largest_five_digit_number_with_product :
  ∃ (x : ℕ), (x = 98752) ∧ (∀ (d : List ℕ), (x.digits 10 = d) → (d.prod = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧ (x < 100000) ∧ (x ≥ 10000) :=
by
  sorry

end largest_five_digit_number_with_product_l307_307060


namespace factor_polynomial_l307_307431

theorem factor_polynomial (x y z : ℝ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) :=
by
  sorry

end factor_polynomial_l307_307431


namespace tom_distance_before_karen_wins_l307_307324

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end tom_distance_before_karen_wins_l307_307324


namespace isosceles_triangle_roots_l307_307673

theorem isosceles_triangle_roots (k : ℝ) (a b : ℝ) 
  (h1 : a = 2 ∨ b = 2)
  (h2 : a^2 - 6 * a + k = 0)
  (h3 : b^2 - 6 * b + k = 0) :
  k = 9 :=
by
  sorry

end isosceles_triangle_roots_l307_307673


namespace maximum_sum_of_digits_difference_l307_307015

-- Definition of the sum of the digits of a number
-- For the purpose of this statement, we'll assume the existence of a function sum_of_digits

def sum_of_digits (n : ℕ) : ℕ :=
  sorry -- Assume the function is defined elsewhere

-- Statement of the problem
theorem maximum_sum_of_digits_difference :
  ∃ x : ℕ, sum_of_digits (x + 2019) - sum_of_digits x = 12 :=
sorry

end maximum_sum_of_digits_difference_l307_307015


namespace scott_awards_l307_307979

theorem scott_awards (S : ℕ) 
  (h1 : ∃ J, J = 3 * S)
  (h2 : ∃ B, B = 2 * (3 * S) ∧ B = 24) : S = 4 := 
by 
  sorry

end scott_awards_l307_307979


namespace sum_possible_values_of_x_l307_307720

open Real

noncomputable def mean (x : ℝ) : ℝ := (25 + x) / 7

def mode : ℝ := 2

def median (x : ℝ) : ℝ :=
  if x ≤ 2 then 2
  else if 4 ≤ x ∧ x ≤ 5 then 4
  else x

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem sum_possible_values_of_x 
  (values : set ℝ)
  (h : ∀ (x : ℝ), x ∈ values ↔ (is_arithmetic_progression 
                                    mode 
                                    (median x) 
                                    (mean x))): 
  ∑ x in values, x = 20 :=
by {
  have h1 : (is_arithmetic_progression 2 2 (mean 2)) → false :=
    -- Calculation here would show contradiction
    sorry,
  have h2 : (is_arithmetic_progression 2 4 (mean 17)) :=
    -- Arithmetic progression check here
    sorry,
  have h3 : (is_arithmetic_progression 2 3 (mean 3)) :=
    -- Arithmetic progression check here
    sorry,
  let values := { 17, 3 },
  have values_eq : values = {17, 3} := rfl,
  rw values_eq,
  exact sum_singleton 17 + sum_singleton 3 -- Sum of elements
}

end sum_possible_values_of_x_l307_307720


namespace garden_breadth_l307_307970

-- Problem statement conditions
def perimeter : ℝ := 600
def length : ℝ := 205

-- Translate the problem into Lean:
theorem garden_breadth (breadth : ℝ) (h1 : 2 * (length + breadth) = perimeter) : breadth = 95 := 
by sorry

end garden_breadth_l307_307970


namespace problem1_l307_307293

theorem problem1 :
  let total_products := 10
  let defective_products := 4
  let first_def_pos := 5
  let last_def_pos := 10
  ∃ (num_methods : Nat), num_methods = 103680 :=
by
  sorry

end problem1_l307_307293


namespace stratified_sampling_distribution_l307_307246

/-- A high school has a total of 2700 students, among which there are 900 freshmen, 
1200 sophomores, and 600 juniors. Using stratified sampling, a sample of 135 students 
is drawn. Prove that the sample contains 45 freshmen, 60 sophomores, and 30 juniors --/
theorem stratified_sampling_distribution :
  let total_students := 2700
  let freshmen := 900
  let sophomores := 1200
  let juniors := 600
  let sample_size := 135
  (sample_size * freshmen / total_students = 45) ∧ 
  (sample_size * sophomores / total_students = 60) ∧ 
  (sample_size * juniors / total_students = 30) :=
by
  sorry

end stratified_sampling_distribution_l307_307246


namespace monotonic_range_a_l307_307780

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (3 * x + a / x - 2) / Real.log 2

theorem monotonic_range_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ 1 ≤ y ∧ x ≤ y → f x a ≤ f y a) ↔ (-1 < a ∧ a ≤ 3) :=
sorry

end monotonic_range_a_l307_307780


namespace bus_driver_hours_worked_last_week_l307_307730

-- Definitions for given conditions
def regular_rate : ℝ := 12
def passenger_rate : ℝ := 0.50
def overtime_rate_1 : ℝ := 1.5 * regular_rate
def overtime_rate_2 : ℝ := 2 * regular_rate
def total_compensation : ℝ := 1280
def total_passengers : ℝ := 350
def earnings_from_passengers : ℝ := total_passengers * passenger_rate
def earnings_from_hourly_rate : ℝ := total_compensation - earnings_from_passengers
def regular_hours : ℝ := 40
def first_tier_overtime_hours : ℝ := 5

-- Theorem to prove the number of hours worked is 67
theorem bus_driver_hours_worked_last_week :
  ∃ (total_hours : ℝ),
    total_hours = 67 ∧
    earnings_from_passengers = total_passengers * passenger_rate ∧
    earnings_from_hourly_rate = total_compensation - earnings_from_passengers ∧
    (∃ (overtime_hours : ℝ),
      (overtime_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2) ∧
      total_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2 )
  :=
sorry

end bus_driver_hours_worked_last_week_l307_307730


namespace point_below_line_l307_307452

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end point_below_line_l307_307452


namespace probability_abs_xi_less_1_point_96_l307_307778

noncomputable def standard_normal (ξ : ℝ → ℝ) :=
  ∃ μ σ : ℝ, μ = 0 ∧ σ = 1 ∧ (∀ x, ξ x = Real.exp (- (x - μ)^2 / (2 * σ^2)) / (σ * Real.sqrt (2 * Real.pi)))

theorem probability_abs_xi_less_1_point_96 (ξ : ℝ → ℝ)
  (Hξ : standard_normal ξ)
  (H₁ : ∃ p, p = 0.025 ∧ P (ξ < -1.96) = p):
  P (|ξ| < 1.96) = 0.95 :=
sorry

end probability_abs_xi_less_1_point_96_l307_307778


namespace water_left_in_bucket_l307_307264

theorem water_left_in_bucket :
  ∀ (original_poured water_left : ℝ),
    original_poured = 0.8 →
    water_left = 0.6 →
    ∃ (poured : ℝ), poured = 0.2 ∧ original_poured - poured = water_left :=
by
  intros original_poured water_left ho hw
  apply Exists.intro 0.2
  simp [ho, hw]
  sorry

end water_left_in_bucket_l307_307264


namespace cut_grid_into_six_polygons_with_identical_pair_l307_307290

noncomputable def totalCells : Nat := 24
def polygonArea : Nat := 4

theorem cut_grid_into_six_polygons_with_identical_pair :
  ∃ (polygons : Fin 6 → Nat → Prop),
  (∀ i, (∃ (cells : Finset (Fin totalCells)), (cells.card = polygonArea ∧ ∀ c ∈ cells, polygons i c))) ∧
  (∃ i j, i ≠ j ∧ ∀ c, polygons i c ↔ polygons j c) :=
sorry

end cut_grid_into_six_polygons_with_identical_pair_l307_307290


namespace students_on_bus_l307_307253

theorem students_on_bus
    (initial_students : ℝ) (first_get_on : ℝ) (first_get_off : ℝ)
    (second_get_on : ℝ) (second_get_off : ℝ)
    (third_get_on : ℝ) (third_get_off : ℝ) :
  initial_students = 21 →
  first_get_on = 7.5 → first_get_off = 2 → 
  second_get_on = 1.2 → second_get_off = 5.3 →
  third_get_on = 11 → third_get_off = 4.8 →
  (initial_students + (first_get_on - first_get_off) +
   (second_get_on - second_get_off) +
   (third_get_on - third_get_off)) = 28.6 := by
  intros
  sorry

end students_on_bus_l307_307253


namespace polynomial_coefficient_sum_l307_307435

theorem polynomial_coefficient_sum
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + 2 * x^3 - 5 * x^2 + 8 * x - 12) :
  a + b + c + d = 6 := 
sorry

end polynomial_coefficient_sum_l307_307435


namespace geom_seq_sum_half_l307_307677

theorem geom_seq_sum_half (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∃ L, L = ∑' n, a n ∧ L = 1 / 2) (h_abs : |q| < 1) :
  a 0 ∈ (Set.Ioo 0 (1 / 2)) ∪ (Set.Ioo (1 / 2) 1) :=
sorry

end geom_seq_sum_half_l307_307677


namespace line_slope_l307_307152

theorem line_slope (t : ℝ) : 
  (∃ (t : ℝ), x = 1 + 2 * t ∧ y = 2 - 3 * t) → 
  (∃ (m : ℝ), m = -3 / 2) :=
sorry

end line_slope_l307_307152


namespace tony_initial_money_l307_307055

theorem tony_initial_money (ticket_cost hotdog_cost money_left initial_money : ℕ) 
  (h_ticket : ticket_cost = 8)
  (h_hotdog : hotdog_cost = 3) 
  (h_left : money_left = 9)
  (h_spent : initial_money = ticket_cost + hotdog_cost + money_left) :
  initial_money = 20 := 
by 
  sorry

end tony_initial_money_l307_307055


namespace fg_of_2_eq_15_l307_307800

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2_eq_15 : f (g 2) = 15 :=
by
  -- The detailed proof would go here
  sorry

end fg_of_2_eq_15_l307_307800


namespace lcm_is_perfect_square_l307_307843

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l307_307843


namespace rains_at_least_once_l307_307181

noncomputable def prob_rains_on_weekend : ℝ :=
  let prob_rain_saturday := 0.60
  let prob_rain_sunday := 0.70
  let prob_no_rain_saturday := 1 - prob_rain_saturday
  let prob_no_rain_sunday := 1 - prob_rain_sunday
  let independent_events := prob_no_rain_saturday * prob_no_rain_sunday
  1 - independent_events

theorem rains_at_least_once :
  prob_rains_on_weekend = 0.88 :=
by sorry

end rains_at_least_once_l307_307181


namespace digits_sum_eq_seven_l307_307626

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l307_307626


namespace matrix_power_problem_l307_307680

open Matrix
open_locale matrix big_operators

def B : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 4], ![0, 2]]

theorem matrix_power_problem :
  B^15 - 3 • (B^14) = ![![0, 4], ![0, -1]] :=
by
  sorry

end matrix_power_problem_l307_307680


namespace exists_arrangement_for_P_23_l307_307994

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l307_307994


namespace elois_banana_bread_l307_307112

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l307_307112


namespace three_digit_numbers_sum_seven_l307_307594

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l307_307594


namespace number_of_three_digit_numbers_with_digit_sum_seven_l307_307628

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l307_307628


namespace peter_ends_up_with_eleven_erasers_l307_307850

def eraser_problem : Nat :=
  let initial_erasers := 8
  let additional_erasers := 3
  let total_erasers := initial_erasers + additional_erasers
  total_erasers

theorem peter_ends_up_with_eleven_erasers :
  eraser_problem = 11 :=
by
  sorry

end peter_ends_up_with_eleven_erasers_l307_307850


namespace solve_for_y_l307_307858

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l307_307858


namespace units_digit_fraction_l307_307063

open Nat

theorem units_digit_fraction : 
  (15 * 16 * 17 * 18 * 19 * 20) % 500 % 10 = 2 := by
  sorry

end units_digit_fraction_l307_307063


namespace find_total_quantities_l307_307697

theorem find_total_quantities (n S S_3 S_2 : ℕ) (h1 : S = 8 * n) (h2 : S_3 = 4 * 3) (h3 : S_2 = 14 * 2) (h4 : S = S_3 + S_2) : n = 5 :=
by
  sorry

end find_total_quantities_l307_307697


namespace twelve_pharmacies_not_sufficient_l307_307526

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l307_307526


namespace find_q_l307_307868

noncomputable def expr (a b c : ℝ) := a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2

noncomputable def lhs (a b c : ℝ) := (a - b) * (b - c) * (c - a)

theorem find_q (a b c : ℝ) : expr a b c = lhs a b c * 1 := by
  sorry

end find_q_l307_307868


namespace sum_of_areas_of_circles_l307_307384

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l307_307384


namespace speed_boat_25_kmph_l307_307899

noncomputable def speed_of_boat_in_still_water (V_s : ℝ) (time : ℝ) (distance : ℝ) : ℝ :=
  let V_d := distance / time
  V_d - V_s

theorem speed_boat_25_kmph (h_vs : V_s = 5) (h_time : time = 4) (h_distance : distance = 120) :
  speed_of_boat_in_still_water V_s time distance = 25 :=
by
  rw [h_vs, h_time, h_distance]
  unfold speed_of_boat_in_still_water
  simp
  norm_num

end speed_boat_25_kmph_l307_307899


namespace ben_final_salary_is_2705_l307_307099

def initial_salary : ℕ := 3000

def salary_after_raise (salary : ℕ) : ℕ :=
  salary * 110 / 100

def salary_after_pay_cut (salary : ℕ) : ℕ :=
  salary * 85 / 100

def final_salary (initial : ℕ) : ℕ :=
  (salary_after_pay_cut (salary_after_raise initial)) - 100

theorem ben_final_salary_is_2705 : final_salary initial_salary = 2705 := 
by 
  sorry

end ben_final_salary_is_2705_l307_307099


namespace tulips_sum_l307_307095

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_l307_307095


namespace fraction_capacity_noah_ali_l307_307178

def capacity_Ali_closet : ℕ := 200
def total_capacity_Noah_closet : ℕ := 100
def each_capacity_Noah_closet : ℕ := total_capacity_Noah_closet / 2

theorem fraction_capacity_noah_ali : (each_capacity_Noah_closet : ℚ) / capacity_Ali_closet = 1 / 4 :=
by sorry

end fraction_capacity_noah_ali_l307_307178


namespace point_D_coordinates_l307_307774

noncomputable def point := ℝ × ℝ

def A : point := (2, 3)
def B : point := (-1, 5)

def vector_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2)
def scalar_mul (k : ℝ) (v : point) : point := (k * v.1, k * v.2)
def vector_add (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)

def D : point := vector_add A (scalar_mul 3 (vector_sub B A))

theorem point_D_coordinates : D = (-7, 9) :=
by
  -- Proof goes here
  sorry

end point_D_coordinates_l307_307774


namespace combined_total_score_is_correct_l307_307459

-- Definitions of point values
def touchdown_points := 6
def extra_point_points := 1
def field_goal_points := 3

-- Hawks' Scores
def hawks_touchdowns := 4
def hawks_successful_extra_points := 2
def hawks_field_goals := 2

-- Eagles' Scores
def eagles_touchdowns := 3
def eagles_successful_extra_points := 3
def eagles_field_goals := 3

-- Calculations
def hawks_total_points := hawks_touchdowns * touchdown_points +
                          hawks_successful_extra_points * extra_point_points +
                          hawks_field_goals * field_goal_points

def eagles_total_points := eagles_touchdowns * touchdown_points +
                           eagles_successful_extra_points * extra_point_points +
                           eagles_field_goals * field_goal_points

def combined_total_score := hawks_total_points + eagles_total_points

-- The theorem that needs to be proved
theorem combined_total_score_is_correct : combined_total_score = 62 :=
by
  -- proof would go here
  sorry

end combined_total_score_is_correct_l307_307459


namespace merchant_profit_percentage_l307_307806

noncomputable def cost_price_of_one_article (C : ℝ) : Prop := ∃ S : ℝ, 20 * C = 16 * S

theorem merchant_profit_percentage (C S : ℝ) (h : cost_price_of_one_article C) : 
  100 * ((S - C) / C) = 25 :=
by 
  sorry

end merchant_profit_percentage_l307_307806


namespace one_sixth_of_x_l307_307151

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 :=
sorry

end one_sixth_of_x_l307_307151


namespace largest_number_of_cakes_l307_307712

theorem largest_number_of_cakes : ∃ (c : ℕ), c = 65 :=
by
  sorry

end largest_number_of_cakes_l307_307712


namespace part_one_part_two_l307_307685

-- Definitions based on the conditions
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 - a}

-- Prove intersection A ∩ B = (0, 1)
theorem part_one : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

-- Prove range of a when A ∪ C = A
theorem part_two (a : ℝ) (h : A ∪ C a = A) : 1 < a := by
  sorry

end part_one_part_two_l307_307685


namespace greatest_possible_z_l307_307074

theorem greatest_possible_z (x y z : ℕ) (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hx_cond : 7 < x) (hy_cond : y < 15) (hx_lt_y : x < y) (hz_gt_zero : z > 0) 
  (hy_sub_x_div_z : (y - x) % z = 0) : z = 2 := 
sorry

end greatest_possible_z_l307_307074


namespace sequence_bn_arithmetic_and_an_formula_l307_307831

theorem sequence_bn_arithmetic_and_an_formula :
  (∀ n : ℕ, ∃ S_n b_n : ℚ, 
  (S_n = (finset.range n).sum (λ i, a_(i+1))) ∧ 
  (b_n = (finset.range n).prod (λ i, S_(i+1))) ∧ 
  ((2 / S_n) + (1 / b_n) = 2)) →
  (∃ d : ℚ, (∀ n : ℕ, n > 0 → b_n = 3 / 2 + (n - 1) * d) ∧ d = 1 / 2) ∧
  (∀ n : ℕ, a_(n+1) = 
    if n+1 = 1 then 3 / 2 
    else -1 / ((n+1) * (n+2))) :=
sorry

end sequence_bn_arithmetic_and_an_formula_l307_307831


namespace min_value_functions_l307_307723

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 1 / x^2
noncomputable def f_B (x : ℝ) : ℝ := 2 * x + 2 / x
noncomputable def f_C (x : ℝ) : ℝ := (x - 1) / (x + 1)
noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x + 1)

theorem min_value_functions :
  (∃ x : ℝ, ∀ y : ℝ, f_A x ≤ f_A y) ∧
  (∃ x : ℝ, ∀ y : ℝ, f_D x ≤ f_D y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_B x ≤ f_B y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_C x ≤ f_C y) :=
by
  sorry

end min_value_functions_l307_307723


namespace distance_from_apex_l307_307385

theorem distance_from_apex (A B : ℝ)
  (h_A : A = 216 * Real.sqrt 3)
  (h_B : B = 486 * Real.sqrt 3)
  (distance_planes : ℝ)
  (h_distance_planes : distance_planes = 8) :
  ∃ h : ℝ, h = 24 :=
by
  sorry

end distance_from_apex_l307_307385


namespace a_beats_b_by_10_seconds_l307_307974

theorem a_beats_b_by_10_seconds :
  ∀ (T_A T_B D_A D_B : ℕ),
    T_A = 615 →
    D_A = 1000 →
    D_A - D_B = 16 →
    T_B = (D_A * T_A) / D_B →
    T_B - T_A = 10 :=
by
  -- Placeholder to ensure the theorem compiles
  intros T_A T_B D_A D_B h1 h2 h3 h4
  sorry

end a_beats_b_by_10_seconds_l307_307974


namespace max_rooks_max_rooks_4x4_max_rooks_8x8_l307_307387

theorem max_rooks (n : ℕ) : ℕ :=
  2 * (2 * n / 3)

theorem max_rooks_4x4 :
  max_rooks 4 = 4 :=
  sorry

theorem max_rooks_8x8 :
  max_rooks 8 = 10 :=
  sorry

end max_rooks_max_rooks_4x4_max_rooks_8x8_l307_307387


namespace dividend_rate_of_stock_l307_307537

variable (MarketPrice : ℝ) (YieldPercent : ℝ) (DividendPercent : ℝ)
variable (NominalValue : ℝ) (AnnualDividend : ℝ)

def stock_dividend_rate_condition (YieldPercent MarketPrice NominalValue DividendPercent : ℝ) 
  (AnnualDividend : ℝ) : Prop :=
  YieldPercent = 20 ∧ MarketPrice = 125 ∧ DividendPercent = 0.25 ∧ NominalValue = 100 ∧
  AnnualDividend = (YieldPercent / 100) * MarketPrice

theorem dividend_rate_of_stock :
  stock_dividend_rate_condition 20 125 100 0.25 25 → (DividendPercent * NominalValue) = 25 :=
by 
  sorry

end dividend_rate_of_stock_l307_307537


namespace find_p_of_binomial_distribution_l307_307294

noncomputable def binomial_mean (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem find_p_of_binomial_distribution (p : ℝ) (h : binomial_mean 5 p = 2) : p = 0.4 :=
by
  sorry

end find_p_of_binomial_distribution_l307_307294


namespace remainder_when_divided_by_84_l307_307233

/-- 
  Given conditions:
  x ≡ 11 [MOD 14]
  Find the remainder when x is divided by 84, which equivalently means proving: 
  x ≡ 81 [MOD 84]
-/

theorem remainder_when_divided_by_84 (x : ℤ) (h1 : x % 14 = 11) : x % 84 = 81 :=
by
  sorry

end remainder_when_divided_by_84_l307_307233


namespace simplify_expression_zero_l307_307333

noncomputable def simplify_expression (a b c d : ℝ) : ℝ :=
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_expression_zero (a b c d : ℝ) (h : a + b + c = d)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  simplify_expression a b c d = 0 :=
by
  sorry

end simplify_expression_zero_l307_307333


namespace Jake_has_fewer_peaches_l307_307695

def Steven_peaches := 14
def Jill_peaches := 5
def Jake_peaches := Jill_peaches + 3

theorem Jake_has_fewer_peaches : Steven_peaches - Jake_peaches = 6 :=
by
  sorry

end Jake_has_fewer_peaches_l307_307695


namespace half_abs_diff_of_squares_l307_307221

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l307_307221


namespace chloe_total_books_l307_307101

noncomputable def total_books (average_books_per_shelf : ℕ) 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (science_fiction_shelves : ℕ) (history_shelves : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + science_fiction_shelves + history_shelves) * average_books_per_shelf

theorem chloe_total_books : 
  total_books 85 7 5 3 2 = 14500 / 100 :=
  by
  sorry

end chloe_total_books_l307_307101


namespace power_function_value_at_neg2_l307_307869

theorem power_function_value_at_neg2 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x : ℝ, f x = x^a)
  (h2 : f 2 = 1 / 4) 
  : f (-2) = 1 / 4 := by
  sorry

end power_function_value_at_neg2_l307_307869


namespace circles_area_sum_l307_307367

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l307_307367


namespace number_of_three_digit_numbers_with_sum_seven_l307_307573

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l307_307573


namespace percentage_change_area_right_triangle_l307_307308

theorem percentage_change_area_right_triangle
  (b h : ℝ)
  (hb : b = 0.5 * h)
  (A_original A_new : ℝ)
  (H_original : A_original = (1 / 2) * b * h)
  (H_new : A_new = (1 / 2) * (1.10 * b) * (1.10 * h)) :
  ((A_new - A_original) / A_original) * 100 = 21 := by
  sorry

end percentage_change_area_right_triangle_l307_307308


namespace find_A_when_A_clubsuit_7_equals_61_l307_307802

-- Define the operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- Define the main problem statement
theorem find_A_when_A_clubsuit_7_equals_61 : 
  ∃ A : ℝ, clubsuit A 7 = 61 ∧ A = (2 * Real.sqrt 30) / 3 :=
by
  sorry

end find_A_when_A_clubsuit_7_equals_61_l307_307802


namespace cotangent_distribution_equivalence_cosine_squared_distribution_equivalence_l307_307331

noncomputable section

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

def uniformDistribution : ProbabilityDistribution ℝ := sorry
def cauchyDistribution : ProbabilityDistribution ℝ := sorry

variable {θ : Ω → ℝ}
variable {C : Ω → ℝ}

axiom θ_uniform : (μ.with_density (λ _, (uniformDistribution.density (/2π)))).pdf θ = (μ.with_density (λ _, 1)).pdf θ
axiom C_cauchy : μ.with_density (cauchyDistribution.density C) = μ.with_density (λ _, 1 / (π * (1 + C^2)))

theorem cotangent_distribution_equivalence :
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cot (θ ω))) =
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cot ((θ ω) / 2))) ∧
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cot ((θ ω) / 2))) =
  (∀ ω, MeasureTheory.PDF (λ ω, C ω)) :=
sorry

theorem cosine_squared_distribution_equivalence :
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cos (θ ω) ^ 2)) =
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cos ((θ ω) / 2) ^ 2)) ∧
  (∀ ω, MeasureTheory.PDF (λ ω, Real.cos ((θ ω) / 2) ^ 2)) =
  (∀ ω, MeasureTheory.PDF (λ ω, (C ω ^ 2) / (1 + C ω ^ 2))) :=
sorry

end cotangent_distribution_equivalence_cosine_squared_distribution_equivalence_l307_307331


namespace sheela_overall_total_income_l307_307852

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end sheela_overall_total_income_l307_307852


namespace solve_for_a_l307_307645

theorem solve_for_a (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (x^3) = Real.log x / Real.log a)
  (h2 : f 8 = 1) :
  a = 2 :=
sorry

end solve_for_a_l307_307645


namespace exist_arrangement_for_P_23_l307_307990

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l307_307990


namespace simplify_sqrt_expression_l307_307929

theorem simplify_sqrt_expression (t : ℝ) : (Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1)) :=
by sorry

end simplify_sqrt_expression_l307_307929


namespace find_x_l307_307304

theorem find_x 
  (x y : ℤ) 
  (h1 : 2 * x - y = 5) 
  (h2 : x + 2 * y = 5) : 
  x = 3 := 
sorry

end find_x_l307_307304


namespace total_spent_on_computer_l307_307320

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end total_spent_on_computer_l307_307320


namespace proof_problem_l307_307047

def sequence : Nat → Rat
| 0 => 2000000
| (n + 1) => sequence n / 2

theorem proof_problem :
  (∀ n, ((sequence n).den = 1) → n < 7) ∧ 
  (sequence 7 = 15625) ∧ 
  (sequence 7 - 3 = 15622) :=
by
  sorry

end proof_problem_l307_307047


namespace Wayne_initially_collected_blocks_l307_307716

-- Let's denote the initial blocks collected by Wayne as 'w'.
-- According to the problem:
-- - Wayne's father gave him 6 more blocks.
-- - He now has 15 blocks in total.
--
-- We need to prove that the initial number of blocks Wayne collected (w) is 9.

theorem Wayne_initially_collected_blocks : 
  ∃ w : ℕ, (w + 6 = 15) ↔ (w = 9) := by
  sorry

end Wayne_initially_collected_blocks_l307_307716


namespace right_triangle_geo_seq_ratio_l307_307364

theorem right_triangle_geo_seq_ratio (l r : ℝ) (ht : 0 < l)
  (hr : 1 < r) (hgeo : l^2 + (l * r)^2 = (l * r^2)^2) :
  (l * r^2) / l = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end right_triangle_geo_seq_ratio_l307_307364


namespace linda_lines_through_O_l307_307416

theorem linda_lines_through_O (O : ℕ × ℕ) (n : ℕ) (grid_Size : ℕ)
  (h_O : O = (0, 0)) (h_n : n = 5) (h_grid : grid_Size = n * n) : 
  ∃ (count : ℕ), count = 8 :=
by
  let valid_points := {(x, y) | 1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4 ∧ Nat.gcd x y = 1}
  let lines_count := valid_points.card
  exact Exists.intro lines_count sorry -- The proof should eventually show lines_count = 8

end linda_lines_through_O_l307_307416


namespace gcd_a_b_l307_307332

noncomputable def a : ℕ := 3333333
noncomputable def b : ℕ := 666666666

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l307_307332


namespace tom_drives_distance_before_karen_wins_l307_307326

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end tom_drives_distance_before_karen_wins_l307_307326


namespace packs_needed_l307_307999

-- Define the problem conditions
def bulbs_bedroom : ℕ := 2
def bulbs_bathroom : ℕ := 1
def bulbs_kitchen : ℕ := 1
def bulbs_basement : ℕ := 4
def bulbs_pack : ℕ := 2

def total_bulbs_main_areas : ℕ := bulbs_bedroom + bulbs_bathroom + bulbs_kitchen + bulbs_basement
def bulbs_garage : ℕ := total_bulbs_main_areas / 2

def total_bulbs : ℕ := total_bulbs_main_areas + bulbs_garage

def total_packs : ℕ := total_bulbs / bulbs_pack

-- The proof statement
theorem packs_needed : total_packs = 6 :=
by
  sorry

end packs_needed_l307_307999


namespace highest_score_is_96_l307_307670

theorem highest_score_is_96 :
  let standard_score := 85
  let deviations := [-9, -4, 11, -7, 0]
  let actual_scores := deviations.map (λ x => standard_score + x)
  actual_scores.maximum = 96 :=
by
  sorry

end highest_score_is_96_l307_307670


namespace marble_ratio_l307_307798

-- Definitions and assumptions from the conditions
def my_marbles : ℕ := 16
def total_marbles : ℕ := 63
def transfer_amount : ℕ := 2

-- After transferring marbles to my brother
def my_marbles_after_transfer := my_marbles - transfer_amount
def brother_marbles (B : ℕ) := B + transfer_amount

-- Friend's marbles
def friend_marbles (F : ℕ) := F = 3 * my_marbles_after_transfer

-- Prove the ratio of marbles after transfer
theorem marble_ratio (B F : ℕ) (hf : F = 3 * my_marbles_after_transfer) (h_total : my_marbles + B + F = total_marbles)
  (h_multiple : ∃ M : ℕ, my_marbles_after_transfer = M * brother_marbles B) :
  (my_marbles_after_transfer : ℚ) / (brother_marbles B : ℚ) = 2 / 1 :=
by
  sorry

end marble_ratio_l307_307798


namespace range_of_function_l307_307934

theorem range_of_function : ∀ (y : ℝ), (0 < y ∧ y ≤ 1 / 2) ↔ ∃ (x : ℝ), y = 1 / (x^2 + 2) := 
by
  sorry

end range_of_function_l307_307934


namespace three_digit_sum_seven_l307_307597

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l307_307597


namespace bananas_used_l307_307107

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l307_307107


namespace three_digit_numbers_sum_seven_l307_307633

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l307_307633


namespace good_numbers_count_l307_307211

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l307_307211


namespace amy_soups_total_l307_307261

def total_soups (chicken_soups tomato_soups : ℕ) : ℕ :=
  chicken_soups + tomato_soups

theorem amy_soups_total : total_soups 6 3 = 9 :=
by
  -- insert the proof here
  sorry

end amy_soups_total_l307_307261


namespace balloon_ways_is_seven_times_pot_ways_l307_307399

open Set

def numberOfBalloonWays : ℕ :=
  7 * 6 ^ 24 - ∑ m in (finset.range 7), (-1 : ℤ) ^ m * finset.card (finset.image (λ s : finset ℕ, m) (finset.powerset (finset.range 7))) * (m * (m - 1) ^ 24)

def numberOfPotWays : ℕ :=
  6 ^ 24 - ∑ m in (finset.range 6), (-1 : ℤ) ^ m * finset.card (finset.image (λ s : finset ℕ, m) (finset.powerset (finset.range 6))) * m ^ 24

theorem balloon_ways_is_seven_times_pot_ways :
  numberOfBalloonWays = 7 * numberOfPotWays :=
by
  -- Proof steps are omitted


-- Placeholder to declare the statement without proof
sorry

end balloon_ways_is_seven_times_pot_ways_l307_307399


namespace find_m_l307_307303

theorem find_m (x y m : ℝ)
  (h1 : 2 * x + y = 6 * m)
  (h2 : 3 * x - 2 * y = 2 * m)
  (h3 : x / 3 - y / 5 = 4) :
  m = 15 :=
by
  sorry

end find_m_l307_307303


namespace expected_coincidences_l307_307396

-- Definitions for the given conditions
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8

def prob_correct (correct : ℕ) : ℚ := correct / num_questions
def prob_incorrect (correct : ℕ) : ℚ := 1 - prob_correct correct 

def prob_vasya_correct := prob_correct vasya_correct
def prob_vasya_incorrect := prob_incorrect vasya_correct
def prob_misha_correct := prob_correct misha_correct
def prob_misha_incorrect := prob_incorrect misha_correct

-- Probability that both guessed correctly or incorrectly
def prob_both_correct_or_incorrect : ℚ := 
  (prob_vasya_correct * prob_misha_correct) + (prob_vasya_incorrect * prob_misha_incorrect)

-- Expected value for one question being a coincidence
def expected_I_k : ℚ := prob_both_correct_or_incorrect

-- Definition of the total number of coincidences
def total_coincidences : ℚ := num_questions * expected_I_k

-- Proof statement
theorem expected_coincidences : 
  total_coincidences = 10.8 := by
  -- calculation for the expected number
  sorry

end expected_coincidences_l307_307396


namespace minimum_value_of_quadratic_function_l307_307302

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 8 * x + 15

theorem minimum_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 ∧ ∀ y : ℝ, quadratic_function y ≥ -1 :=
by
  sorry

end minimum_value_of_quadratic_function_l307_307302


namespace problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l307_307305

theorem problem_a_lt_b_lt_0_implies_ab_gt_b_sq (a b : ℝ) (h : a < b ∧ b < 0) : ab > b^2 := by
  sorry

end problem_a_lt_b_lt_0_implies_ab_gt_b_sq_l307_307305


namespace max_sum_a_b_l307_307168

theorem max_sum_a_b (a b : ℝ) (ha : 4 * a + 3 * b ≤ 10) (hb : 3 * a + 6 * b ≤ 12) : a + b ≤ 22 / 7 :=
sorry

end max_sum_a_b_l307_307168


namespace bianca_marathon_total_miles_l307_307100

theorem bianca_marathon_total_miles : 8 + 4 = 12 :=
by
  sorry

end bianca_marathon_total_miles_l307_307100


namespace inequality_proof_l307_307764

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : (3 / (a * b * c)) ≥ (a + b + c)) : 
    (1 / a + 1 / b + 1 / c) ≥ (a + b + c) :=
  sorry

end inequality_proof_l307_307764


namespace raj_is_older_than_ravi_l307_307474

theorem raj_is_older_than_ravi
  (R V H L x : ℕ)
  (h1 : R = V + x)
  (h2 : H = V - 2)
  (h3 : R = 3 * L)
  (h4 : H * 2 = 3 * L)
  (h5 : 20 = (4 * H) / 3) :
  x = 13 :=
by
  sorry

end raj_is_older_than_ravi_l307_307474


namespace number_of_three_digit_numbers_with_sum_seven_l307_307576

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l307_307576


namespace three_digit_numbers_sum_seven_l307_307596

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l307_307596


namespace absent_minded_scientist_l307_307191

theorem absent_minded_scientist :
  let P := 1/2 in
  let PA_R := 0.05 in
  let PB_R := 0.80 in
  let P_not_R := 1/2 in
  let PA_not_R := 0.90 in
  let PB_not_R := 0.02 in
  let P_R_AB := P * PA_R * PB_R in
  let P_not_R_AB := P_not_R * PA_not_R * PB_not_R in
  let P_A_and_B := P_R_AB + P_not_R_AB in
  (P_R_AB / P_A_and_B) ≈ 0.69 :=
by
  sorry

end absent_minded_scientist_l307_307191


namespace a6_add_b6_geq_ab_a4_add_b4_l307_307291

theorem a6_add_b6_geq_ab_a4_add_b4 (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end a6_add_b6_geq_ab_a4_add_b4_l307_307291


namespace luke_piles_of_quarters_l307_307477

theorem luke_piles_of_quarters (Q D : ℕ) 
  (h1 : Q = D) -- number of piles of quarters equals number of piles of dimes
  (h2 : 3 * Q + 3 * D = 30) -- total number of coins is 30
  : Q = 5 :=
by
  sorry

end luke_piles_of_quarters_l307_307477


namespace probability_multiple_of_200_l307_307153

open Finset

def S : Finset ℕ := {2, 4, 10, 12, 15, 20, 25, 50, 100}

def is_multiple_of_200 (x : ℕ) : Prop :=
  ∃ (a b : ℕ), x = 200 * a + b

theorem probability_multiple_of_200 :
  (∃ (count : ℕ),
    count = (S.filter (λ x, ∃ y ∈ S, y ≠ x ∧ is_multiple_of_200 (x * y))).card ∧
    (36 : ℕ) = (S.card.choose 2) ∧
    (count : ℚ) / 36 = 1 / 3) :=
begin
  sorry
end

end probability_multiple_of_200_l307_307153


namespace train_B_speed_l307_307394

-- Given conditions
def speed_train_A := 70 -- km/h
def time_after_meet_A := 9 -- hours
def time_after_meet_B := 4 -- hours

-- Proof statement
theorem train_B_speed : 
  ∃ (V_b : ℕ),
    V_b * time_after_meet_B + V_b * s = speed_train_A * time_after_meet_A + speed_train_A * s ∧
    V_b = speed_train_A := 
sorry

end train_B_speed_l307_307394


namespace minimal_bananas_l307_307121

noncomputable def total_min_bananas : ℕ :=
  let b1 := 72
  let b2 := 72
  let b3 := 216
  let b4 := 72
  b1 + b2 + b3 + b4

theorem minimal_bananas (total_bananas : ℕ) (ratio1 ratio2 ratio3 ratio4 : ℕ) 
  (b1 b2 b3 b4 : ℕ) 
  (h_ratio : ratio1 = 4 ∧ ratio2 = 3 ∧ ratio3 = 2 ∧ ratio4 = 1) 
  (h_div_constraints : ∀ n m : ℕ, (n % m = 0 ∨ m % n = 0) ∧ n ≥ ratio1 * ratio2 * ratio3 * ratio4) 
  (h_bananas : b1 = 72 ∧ b2 = 72 ∧ b3 = 216 ∧ b4 = 72 ∧ 
              4 * (b1 / 2 + b2 / 6 + b3 / 9 + 7 * b4 / 72) = 3 * (b1 / 6 + b2 / 3 + b3 / 9 + 7 * b4 / 72) ∧ 
              2 * (b1 / 6 + b2 / 6 + b3 / 6 + 7 * b4 / 72) = (b1 / 6 + b2 / 6 + b3 / 9 + b4 / 8)) : 
  total_bananas = 432 := by
  sorry

end minimal_bananas_l307_307121


namespace lowest_price_l307_307724

theorem lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components produced_cost total_variable_cost total_cost lowest_price : ℝ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 2)
  (h3 : fixed_costs = 16200)
  (h4 : number_of_components = 150)
  (h5 : total_variable_cost = cost_per_component + shipping_cost_per_unit)
  (h6 : produced_cost = total_variable_cost * number_of_components)
  (h7 : total_cost = produced_cost + fixed_costs)
  (h8 : lowest_price = total_cost / number_of_components) :
  lowest_price = 190 :=
  by
  sorry

end lowest_price_l307_307724


namespace amy_small_gardens_l307_307092

-- Define the initial number of seeds
def initial_seeds : ℕ := 101

-- Define the number of seeds planted in the big garden
def big_garden_seeds : ℕ := 47

-- Define the number of seeds planted in each small garden
def seeds_per_small_garden : ℕ := 6

-- Define the number of small gardens
def number_of_small_gardens : ℕ := (initial_seeds - big_garden_seeds) / seeds_per_small_garden

-- Prove that Amy has 9 small gardens
theorem amy_small_gardens : number_of_small_gardens = 9 := by
  sorry

end amy_small_gardens_l307_307092


namespace part1_arithmetic_sequence_part2_general_formula_l307_307832

variable (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i
axiom condition2 : ∀ n, b n = ∏ i in Finset.range (n + 1), S i
axiom condition3 : ∀ n, (2 / S n) + (1 / b n) = 2

-- Part (1) Proof Statement
theorem part1_arithmetic_sequence : ∃ c d, b 1 = c ∧ (∀ n, b n - b (n - 1) = d) := sorry

-- Part (2) General Formula
theorem part2_general_formula : 
  a 1 = 3 / 2 ∧ (∀ n, n ≥ 2 → a n = - (1 / (n * (n + 1)))) := sorry

end part1_arithmetic_sequence_part2_general_formula_l307_307832


namespace least_number_divisible_by_13_l307_307515

theorem least_number_divisible_by_13 (n : ℕ) :
  (∀ m : ℕ, 2 ≤ m ∧ m ≤ 7 → n % m = 2) ∧ (n % 13 = 0) → n = 1262 :=
by sorry

end least_number_divisible_by_13_l307_307515


namespace same_color_pick_probability_l307_307084

/-- 
  Given that a jar has 15 red candies and 5 blue candies,
  Terry picks two candies at random, then Mary picks one of 
  the remaining candies at random. Prove that the probability 
  that all picked candies are of the same color is 31/76.
-/
theorem same_color_pick_probability :
  let total_candies := 20
  let red_candies := 15
  let blue_candies := 5
  let terry_two_reds := (red_candies * (red_candies - 1)) / (total_candies * (total_candies - 1))
  let mary_one_red_given_two_reds := (red_candies - 2) / (total_candies - 2)
  let all_red_probability := terry_two_reds * mary_one_red_given_two_reds
  let terry_two_blues := (blue_candies * (blue_candies - 1)) / (total_candies * (total_candies - 1))
  let mary_one_blue_given_two_blues := (blue_candies - 2) / (total_candies - 2)
  let all_blue_probability := terry_two_blues * mary_one_blue_given_two_blues
  let total_probability := all_red_probability + all_blue_probability
  total_probability = (31 / 76) :=
by
  sorry

end same_color_pick_probability_l307_307084


namespace solve_for_y_l307_307856

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l307_307856


namespace length_of_hypotenuse_l307_307911

/-- Define the problem's parameters -/
def perimeter : ℝ := 34
def area : ℝ := 24
def length_hypotenuse (a b c : ℝ) : Prop := a + b + c = perimeter 
  ∧ (1/2) * a * b = area
  ∧ a^2 + b^2 = c^2

/- Lean statement for the proof problem -/
theorem length_of_hypotenuse (a b c : ℝ) 
  (h1: a + b + c = 34)
  (h2: (1/2) * a * b = 24)
  (h3: a^2 + b^2 = c^2)
  : c = 62 / 4 := sorry

end length_of_hypotenuse_l307_307911


namespace part_I_part_II_l307_307781

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem part_I (x : ℝ) : (f x > 5) ↔ (x < -3 ∨ x > 2) :=
  sorry

theorem part_II (a : ℝ) : (∀ x, f x < a ↔ false) ↔ (a ≤ 3) :=
  sorry

end part_I_part_II_l307_307781


namespace remainder_and_division_l307_307521

theorem remainder_and_division (x y : ℕ) (h1 : x % y = 8) (h2 : (x / y : ℝ) = 76.4) : y = 20 :=
sorry

end remainder_and_division_l307_307521


namespace solve_for_y_l307_307857

theorem solve_for_y (y : ℚ) (h : 1/3 + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solve_for_y_l307_307857


namespace probability_of_odd_divisor_l307_307872

noncomputable def prime_factorization_15! : ℕ :=
  (2 ^ 11) * (3 ^ 6) * (5 ^ 3) * (7 ^ 2) * 11 * 13

def total_factors_15! : ℕ :=
  (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def odd_factors_15! : ℕ :=
  (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def probability_odd_divisor_15! : ℚ :=
  odd_factors_15! / total_factors_15!

theorem probability_of_odd_divisor : probability_odd_divisor_15! = 1 / 12 :=
by
  sorry

end probability_of_odd_divisor_l307_307872


namespace sum_powers_seventh_l307_307021

/-- Given the sequence values for sums of powers of 'a' and 'b', prove the value of the sum of the 7th powers. -/
theorem sum_powers_seventh (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^7 + b^7 = 29 := 
  sorry

end sum_powers_seventh_l307_307021


namespace binom_factorial_eq_120_factorial_l307_307102

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem binom_factorial_eq_120_factorial : (factorial (binomial 10 3)) = factorial 120 := by
  sorry

end binom_factorial_eq_120_factorial_l307_307102


namespace outfit_count_l307_307353

theorem outfit_count (shirts pants ties belts : ℕ) (h_shirts : shirts = 8) (h_pants : pants = 5) (h_ties : ties = 4) (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end outfit_count_l307_307353


namespace expanded_form_correct_l307_307193

theorem expanded_form_correct :
  (∃ a b c : ℤ, (∀ x : ℚ, 2 * (x - 3)^2 - 12 = a * x^2 + b * x + c) ∧ (10 * a - b - 4 * c = 8)) :=
by
  sorry

end expanded_form_correct_l307_307193


namespace digits_sum_eq_seven_l307_307621

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l307_307621


namespace solve_for_y_l307_307860

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l307_307860


namespace time_to_cut_womans_hair_l307_307007

theorem time_to_cut_womans_hair 
  (WL : ℕ) (WM : ℕ) (WK : ℕ) (total_time : ℕ) 
  (num_women : ℕ) (num_men : ℕ) (num_kids : ℕ) 
  (men_haircut_time : ℕ) (kids_haircut_time : ℕ) 
  (overall_time : ℕ) :
  men_haircut_time = 15 →
  kids_haircut_time = 25 →
  num_women = 3 →
  num_men = 2 →
  num_kids = 3 →
  overall_time = 255 →
  overall_time = (num_women * WL + num_men * men_haircut_time + num_kids * kids_haircut_time) →
  WL = 50 :=
by
  sorry

end time_to_cut_womans_hair_l307_307007


namespace count_good_numbers_l307_307215

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l307_307215


namespace sum_of_values_of_z_l307_307682

def f (x : ℝ) := x^2 - 2*x + 3

theorem sum_of_values_of_z (z : ℝ) (h : f (5 * z) = 7) : z = 2 / 25 :=
sorry

end sum_of_values_of_z_l307_307682


namespace lcm_is_perfect_square_l307_307848

open Nat

theorem lcm_is_perfect_square (a b : ℕ) : 
  (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0 → ∃ k : ℕ, k^2 = lcm a b :=
by
  sorry

end lcm_is_perfect_square_l307_307848


namespace graph_passes_through_fixed_point_l307_307644

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x-1)

theorem graph_passes_through_fixed_point (a : ℝ) : f a 1 = 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end graph_passes_through_fixed_point_l307_307644


namespace inequality_abc_l307_307982

theorem inequality_abc (a b c : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) (h5 : 2 ≤ n) :
  (a / (b + c)^(1/(n:ℝ)) + b / (c + a)^(1/(n:ℝ)) + c / (a + b)^(1/(n:ℝ)) ≥ 3 / 2^(1/(n:ℝ))) :=
by sorry

end inequality_abc_l307_307982


namespace initial_orange_balloons_l307_307350

-- Definitions
variable (x : ℕ)
variable (h1 : x - 2 = 7)

-- Theorem to prove
theorem initial_orange_balloons (h1 : x - 2 = 7) : x = 9 :=
sorry

end initial_orange_balloons_l307_307350


namespace ticket_sales_total_cost_l307_307257

noncomputable def total_ticket_cost (O B : ℕ) : ℕ :=
  12 * O + 8 * B

theorem ticket_sales_total_cost (O B : ℕ) (h1 : O + B = 350) (h2 : B = O + 90) :
  total_ticket_cost O B = 3320 :=
by
  -- the proof steps calculating the total cost will go here
  sorry

end ticket_sales_total_cost_l307_307257


namespace suggestions_difference_l307_307180

def mashed_potatoes_suggestions : ℕ := 408
def pasta_suggestions : ℕ := 305
def bacon_suggestions : ℕ := 137
def grilled_vegetables_suggestions : ℕ := 213
def sushi_suggestions : ℕ := 137

theorem suggestions_difference :
  let highest := mashed_potatoes_suggestions
  let lowest := bacon_suggestions
  highest - lowest = 271 :=
by
  sorry

end suggestions_difference_l307_307180


namespace arriving_late_l307_307204

-- Definitions from conditions
def usual_time : ℕ := 24
def slower_factor : ℚ := 3 / 4

-- Derived from conditions
def slower_time : ℚ := usual_time * (4 / 3)

-- To be proven
theorem arriving_late : slower_time - usual_time = 8 := by
  sorry

end arriving_late_l307_307204


namespace rationalize_sqrt_three_sub_one_l307_307029

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l307_307029


namespace monotonic_increasing_interval_l307_307773

noncomputable def f (x : ℝ) : ℝ := sorry

theorem monotonic_increasing_interval :
  (∀ x Δx : ℝ, 0 < x → 0 < Δx → 
  (f (x + Δx) - f x) / Δx = (2 / (Real.sqrt (x + Δx) + Real.sqrt x)) - (1 / (x^2 + x * Δx))) →
  ∀ x : ℝ, 1 < x → (∃ ε > 0, ∀ y, x < y ∧ y < x + ε → f y > f x) :=
by
  intro hyp
  sorry

end monotonic_increasing_interval_l307_307773


namespace value_of_x3_plus_inv_x3_l307_307665

theorem value_of_x3_plus_inv_x3 (x : ℝ) (h : 728 = x^6 + 1 / x^6) : 
  x^3 + 1 / x^3 = Real.sqrt 730 :=
sorry

end value_of_x3_plus_inv_x3_l307_307665


namespace num_three_digit_sums7_l307_307608

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l307_307608


namespace equal_real_roots_of_quadratic_eq_l307_307967

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end equal_real_roots_of_quadratic_eq_l307_307967


namespace sufficient_but_not_necessary_condition_l307_307076

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 2 → x^2 + 2 * x - 8 > 0) ∧ (¬(x > 2) → ¬(x^2 + 2 * x - 8 > 0)) → false :=
by 
  sorry

end sufficient_but_not_necessary_condition_l307_307076


namespace determine_digits_l307_307935

theorem determine_digits :
  ∃ (A B C D : ℕ), 
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧ 
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧ 
    1000 ≤ 1000 * C + 100 * B + 10 * A + D ∧ 
    1000 * C + 100 * B + 10 * A + D ≤ 9999 ∧ 
    (1000 * A + 100 * B + 10 * C + D) * D = 1000 * C + 100 * B + 10 * A + D ∧ 
    A = 2 ∧ B = 1 ∧ C = 7 ∧ D = 8 :=
by
  sorry

end determine_digits_l307_307935


namespace monomial_k_add_n_l307_307298

variable (k n : ℤ)

-- Conditions
def is_monomial_coefficient (k : ℤ) : Prop := -k = 5
def is_monomial_degree (n : ℤ) : Prop := n + 1 = 7

-- Theorem to prove
theorem monomial_k_add_n (hk : is_monomial_coefficient k) (hn : is_monomial_degree n) : k + n = 1 :=
by
  sorry

end monomial_k_add_n_l307_307298


namespace sum_is_24_l307_307390

-- Define the conditions
def A := 3
def B := 7 * A

-- Define the theorem to prove that the sum is 24
theorem sum_is_24 : A + B = 24 :=
by
  -- Adding sorry here since we're not required to provide the proof
  sorry

end sum_is_24_l307_307390


namespace islander_real_name_l307_307072

-- Definition of types of people on the island
inductive IslanderType
| Knight   -- Always tells the truth
| Liar     -- Always lies
| Normal   -- Can lie or tell the truth

-- The possible names of the islander
inductive Name
| Edwin
| Edward

-- Condition: You met the islander who can be Edwin or Edward
def possible_names : List Name := [Name.Edwin, Name.Edward]

-- Condition: The islander said their name is Edward
def islander_statement : Name := Name.Edward

-- Condition: The islander is a Liar (as per the solution interpretation)
def islander_type : IslanderType := IslanderType.Liar

-- The proof problem: Prove the islander's real name is Edwin
theorem islander_real_name : islander_type = IslanderType.Liar ∧ islander_statement = Name.Edward → ∃ n : Name, n = Name.Edwin :=
by
  sorry

end islander_real_name_l307_307072


namespace three_digit_sum_7_l307_307612

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l307_307612


namespace lines_parallel_if_perpendicular_to_same_plane_l307_307784

-- Definitions and conditions
variables {Point : Type*} [MetricSpace Point]
variables {Line Plane : Type*}

def is_parallel (l₁ l₂ : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry

variables (m n : Line) (α : Plane)

-- Theorem statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  is_perpendicular m α → is_perpendicular n α → is_parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l307_307784


namespace weight_of_each_pack_l307_307198

-- Definitions based on conditions
def total_sugar : ℕ := 3020
def leftover_sugar : ℕ := 20
def number_of_packs : ℕ := 12

-- Definition of sugar used for packs
def sugar_used_for_packs : ℕ := total_sugar - leftover_sugar

-- Proof statement to be verified
theorem weight_of_each_pack : sugar_used_for_packs / number_of_packs = 250 := by
  sorry

end weight_of_each_pack_l307_307198


namespace binomial_coeff_x_squared_l307_307461

theorem binomial_coeff_x_squared (x : ℝ) :
  (∑ r in Finset.range 7, (Nat.choose 6 r) * ((sqrt x / 2) ^ (6 - r)) * ((- 2 / sqrt x) ^ r)) =
  ((-2) * ((1 / 2)^5) * (Nat.choose 6 1)) * x ^ 2 + 
  ∑ r in (Finset.range 7).filter (λ r, r ≠ 1), (Nat.choose 6 r) * ((sqrt x / 2) ^ (6 - r)) * ((- 2 / sqrt x) ^ r) :=
begin
  sorry
end

end binomial_coeff_x_squared_l307_307461


namespace major_axis_range_l307_307660

theorem major_axis_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ x M N : ℝ, (x + (1 - x)) = 1 → x * (1 - x) = 0) 
  (e : ℝ) (h4 : (Real.sqrt 3 / 3) ≤ e ∧ e ≤ (Real.sqrt 2 / 2)) :
  ∃ a : ℝ, 2 * (Real.sqrt 5) ≤ 2 * a ∧ 2 * a ≤ 2 * (Real.sqrt 6) := 
sorry

end major_axis_range_l307_307660


namespace sum_of_exponents_l307_307117

def power_sum_2021 (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ r → (a k = 1 ∨ a k = -1)) ∧
  (a 1 * 3 ^ n 1 + a 2 * 3 ^ n 2 + a 3 * 3 ^ n 3 + a 4 * 3 ^ n 4 + a 5 * 3 ^ n 5 + a 6 * 3 ^ n 6 = 2021) ∧
  (n 1 = 7 ∧ n 2 = 5 ∧ n 3 = 4 ∧ n 4 = 2 ∧ n 5 = 1 ∧ n 6 = 0) ∧
  (a 1 = 1 ∧ a 2 = -1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 1 ∧ a 6 = -1)

theorem sum_of_exponents : ∃ (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ), power_sum_2021 a n r ∧ (n 1 + n 2 + n 3 + n 4 + n 5 + n 6 = 19) :=
by {
  sorry
}

end sum_of_exponents_l307_307117


namespace rationalization_correct_l307_307033

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l307_307033


namespace triangular_partition_l307_307690

-- Define triangular number function
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_partition (s : ℕ) (h_pos : s > 0) (h_ne_2 : s ≠ 2) :
  ∃ (t : Finset ℕ), t.card = s ∧ (∀ (n ∈ t), ∃ k, triangular_number k = n) ∧ 
  (∑ n in t, (1 : ℚ) / n) = 1 :=
sorry

end triangular_partition_l307_307690


namespace num_solutions_abcd_eq_2020_l307_307938

theorem num_solutions_abcd_eq_2020 :
  ∃ S : Finset (ℕ × ℕ × ℕ × ℕ), 
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔ (a^2 + b^2) * (c^2 - d^2) = 2020 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧
    S.card = 6 :=
sorry

end num_solutions_abcd_eq_2020_l307_307938


namespace remainder_when_M_div_1000_l307_307270

open Function

def minimallyIntersectingTriple (A B C : Set ℤ) : Prop :=
  (|A ∩ B| = 1) ∧ (|B ∩ C| = 1) ∧ (|C ∩ A| = 1) ∧ (A ∩ B ∩ C = ∅)

def tripleSet  := {s : Set ℤ // s \in (powerset (Finset.range 8)).val}
noncomputable instance : Fintype tripleSet := by
  unfold tripleSet
  sorry

def M : ℕ := 
  Fintype.card  {t : tripleSet × tripleSet × tripleSet // minimallyIntersectingTriple t.1.val t.2.val t.2.snd.val }

theorem remainder_when_M_div_1000 : M % 1000 = 344 := by
  sorry

end remainder_when_M_div_1000_l307_307270


namespace num_pos_int_solutions_2a_plus_3b_eq_15_l307_307873

theorem num_pos_int_solutions_2a_plus_3b_eq_15 : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 2 * a + 3 * b = 15) ∧ 
  (∀ (a1 a2 b1 b2 : ℕ), 0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2 ∧ 
  (2 * a1 + 3 * b1 = 15) ∧ (2 * a2 + 3 * b2 = 15) → 
  ((a1 = 3 ∧ b1 = 3 ∨ a1 = 6 ∧ b1 = 1) ∧ (a2 = 3 ∧ b2 = 3 ∨ a2 = 6 ∧ b2 = 1))) := 
  sorry

end num_pos_int_solutions_2a_plus_3b_eq_15_l307_307873


namespace twelve_pharmacies_not_enough_l307_307531

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l307_307531


namespace solve_for_m_l307_307769

theorem solve_for_m (m α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
by
  sorry

end solve_for_m_l307_307769


namespace total_marbles_l307_307457

-- Definitions to state the problem
variables {r b g : ℕ}
axiom ratio_condition : r / b = 2 / 4 ∧ r / g = 2 / 6
axiom blue_marbles : b = 30

-- Theorem statement
theorem total_marbles : r + b + g = 90 :=
by sorry

end total_marbles_l307_307457


namespace correct_eq_count_l307_307889

-- Define the correctness of each expression
def eq1 := (∀ x : ℤ, (-2 * x)^3 = 2 * x^3 = false)
def eq2 := (∀ a : ℤ, a^2 * a^3 = a^3 = false)
def eq3 := (∀ x : ℤ, (-x)^9 / (-x)^3 = x^6 = true)
def eq4 := (∀ a : ℤ, (-3 * a^2)^3 = -9 * a^6 = false)

-- Define the condition that there are exactly one correct equation
def num_correct_eqs := (1 = 1)

-- The theorem statement, proving the count of correct equations is 1
theorem correct_eq_count : eq1 → eq2 → eq3 → eq4 → num_correct_eqs :=
  by intros; sorry

end correct_eq_count_l307_307889


namespace smaller_circle_circumference_l307_307649

-- Definitions based on the conditions given in the problem
def AB : ℝ := 24
def BC : ℝ := 45
def CD : ℝ := 28
def DA : ℝ := 53
def smaller_circle_diameter : ℝ := AB

-- Main statement to prove
theorem smaller_circle_circumference :
  let r : ℝ := smaller_circle_diameter / 2
  let circumference := 2 * Real.pi * r
  circumference = 24 * Real.pi := by
  sorry

end smaller_circle_circumference_l307_307649


namespace number_of_liars_l307_307313

/-- Definition of conditions -/
def total_islands : Nat := 17
def population_per_island : Nat := 119

-- Conditions based on the problem description
def islands_yes_first_question : Nat := 7
def islands_no_first_question : Nat := total_islands - islands_yes_first_question

def islands_no_second_question : Nat := 7
def islands_yes_second_question : Nat := total_islands - islands_no_second_question

def minimum_knights_for_no_second_question : Nat := 60  -- At least 60 knights

/-- Main theorem -/
theorem number_of_liars : 
  ∃ x y: Nat, 
    (x + (islands_no_first_question - y) = islands_yes_first_question ∧ 
     y - x = 3 ∧ 
     60 * x + 59 * y + 119 * (islands_no_first_question - y) = 1010 ∧
     (total_islands * population_per_island - 1010 = 1013)) := by
  sorry

end number_of_liars_l307_307313


namespace train_speed_is_72_kmph_l307_307741

-- Define the given conditions in Lean
def crossesMan (L V : ℝ) : Prop := L = 19 * V
def crossesPlatform (L V : ℝ) : Prop := L + 220 = 30 * V

-- The main theorem which states that the speed of the train is 72 km/h under given conditions
theorem train_speed_is_72_kmph (L V : ℝ) (h1 : crossesMan L V) (h2 : crossesPlatform L V) :
  (V * 3.6) = 72 := by
  -- We will provide a full proof here later
  sorry

end train_speed_is_72_kmph_l307_307741


namespace three_digit_numbers_sum_seven_l307_307593

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l307_307593


namespace find_N_l307_307666

theorem find_N (N : ℕ) : 
  981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N → N = 91 :=
by
  assume h : 981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N
  sorry

end find_N_l307_307666


namespace find_third_divisor_l307_307433

theorem find_third_divisor (n : ℕ) (d : ℕ) 
  (h1 : (n - 4) % 12 = 0)
  (h2 : (n - 4) % 16 = 0)
  (h3 : (n - 4) % d = 0)
  (h4 : (n - 4) % 21 = 0)
  (h5 : (n - 4) % 28 = 0)
  (h6 : n = 1012) :
  d = 3 :=
by
  sorry

end find_third_divisor_l307_307433


namespace b_arithmetic_sequence_a_general_formula_l307_307830

-- Definitions based on the problem conditions
def S (n : ℕ) : ℝ := -- Define the sum of the first n terms of the sequence a
sorry

def b (n : ℕ) : ℝ := -- Define the product of the first n terms of the sequence S
sorry

-- The given condition in the problem
axiom condition (n : ℕ) (n_pos : n > 0) : 
  (2 / S n) + (1 / b n) = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) (n_pos : n > 0) : 
  ∃ d : ℝ, ∀ m, b (m+1) - b m = d := 
sorry

-- Find the general formula for a_n
def a : ℕ → ℝ
| 1 => 3 / 2
| n+2 => -1 / (n+1) / (n+2)
| _ => 0

theorem a_general_formula : 
  ∀ n, (n = 1 → a n = 3 / 2) ∧ (n ≥ 2 → a n = -1 / (n * (n + 1))) :=
sorry

end b_arithmetic_sequence_a_general_formula_l307_307830


namespace john_piano_lessons_l307_307010

theorem john_piano_lessons (total_cost piano_cost original_price_per_lesson discount : ℕ) 
    (total_spent : ℕ) : 
    total_spent = piano_cost + ((total_cost - piano_cost) / (original_price_per_lesson - discount)) → 
    total_cost = 1100 ∧ piano_cost = 500 ∧ original_price_per_lesson = 40 ∧ discount = 10 → 
    (total_cost - piano_cost) / (original_price_per_lesson - discount) = 20 :=
by
  intros h1 h2
  sorry

end john_piano_lessons_l307_307010


namespace not_enough_pharmacies_l307_307528

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l307_307528


namespace fraction_of_tomatoes_eaten_l307_307201

theorem fraction_of_tomatoes_eaten (original : ℕ) (remaining : ℕ) (birds_ate : ℕ) (h1 : original = 21) (h2 : remaining = 14) (h3 : birds_ate = original - remaining) :
  (birds_ate : ℚ) / original = 1 / 3 :=
by
  sorry

end fraction_of_tomatoes_eaten_l307_307201


namespace find_m_l307_307656

open Set Real

noncomputable def setA : Set ℝ := {x | x < 2}
noncomputable def setB : Set ℝ := {x | x > 4}
noncomputable def setC (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m - 1}

theorem find_m (m : ℝ) : setC m ⊆ (setA ∪ setB) → m < 3 :=
by
  sorry

end find_m_l307_307656


namespace maria_average_speed_l307_307195

noncomputable def average_speed (total_distance : ℕ) (total_time : ℕ) : ℚ :=
  total_distance / total_time

theorem maria_average_speed :
  average_speed 200 7 = 28 + 4 / 7 :=
sorry

end maria_average_speed_l307_307195


namespace find_k_l307_307961

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l307_307961


namespace count_of_good_numbers_l307_307210

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l307_307210


namespace percent_greater_than_l307_307359

theorem percent_greater_than (M N : ℝ) (hN : N ≠ 0) : (M - N) / N * 100 = 100 * (M - N) / N :=
by sorry

end percent_greater_than_l307_307359


namespace expected_coincidence_proof_l307_307398

noncomputable def expected_coincidences (total_questions : ℕ) (vasya_correct : ℕ) (misha_correct : ℕ) : ℝ :=
  let vasya_probability := vasya_correct / total_questions
  let misha_probability := misha_correct / total_questions
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  total_questions * coincidence_probability

theorem expected_coincidence_proof : 
  expected_coincidences 20 6 8 = 10.8 :=
by {
  let total_questions := 20
  let vasya_correct := 6
  let misha_correct := 8
  
  let vasya_probability := 0.3
  let misha_probability := 0.4
  
  let both_correct_probability := vasya_probability * misha_probability
  let both_incorrect_probability := (1 - vasya_probability) * (1 - misha_probability)
  let coincidence_probability := both_correct_probability + both_incorrect_probability
  let expected := total_questions * coincidence_probability
  
  have h1 : vasya_probability = 6 / 20 := by sorry
  have h2 : misha_probability = 8 / 20 := by sorry
  have h3 : both_correct_probability = 0.3 * 0.4 := by sorry
  have h4 : both_incorrect_probability = 0.7 * 0.6 := by sorry
  have h5 : coincidence_probability = 0.54 := by sorry
  have h6 : total_questions * coincidence_probability = 20 * 0.54 := by sorry
  have h7 : 20 * 0.54 = 10.8 := by sorry

  sorry
}

end expected_coincidence_proof_l307_307398


namespace sum_of_areas_of_circles_l307_307369

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l307_307369


namespace tangent_addition_l307_307803

open Real

theorem tangent_addition (x : ℝ) (h : tan x = 3) :
  tan (x + π / 6) = - (5 * (sqrt 3 + 3)) / 3 := by
  -- Providing a brief outline of the proof steps is not necessary for the statement
  sorry

end tangent_addition_l307_307803


namespace savings_calculation_l307_307912

-- Definitions of the given conditions
def window_price : ℕ := 100
def free_window_offer (purchased : ℕ) : ℕ := purchased / 4

-- Number of windows needed
def dave_needs : ℕ := 7
def doug_needs : ℕ := 8

-- Calculations based on the conditions
def individual_costs : ℕ :=
  (dave_needs - free_window_offer dave_needs) * window_price +
  (doug_needs - free_window_offer doug_needs) * window_price

def together_costs : ℕ :=
  let total_needs := dave_needs + doug_needs
  (total_needs - free_window_offer total_needs) * window_price

def savings : ℕ := individual_costs - together_costs

-- Proof statement
theorem savings_calculation : savings = 100 := by
  sorry

end savings_calculation_l307_307912


namespace atomic_weight_Oxygen_l307_307286

theorem atomic_weight_Oxygen :
  ∀ (Ba_atomic_weight S_atomic_weight : ℝ),
    (Ba_atomic_weight = 137.33) →
    (S_atomic_weight = 32.07) →
    (Ba_atomic_weight + S_atomic_weight + 4 * 15.9 = 233) →
    15.9 = 233 - 137.33 - 32.07 / 4 := 
by
  intros Ba_atomic_weight S_atomic_weight hBa hS hm
  sorry

end atomic_weight_Oxygen_l307_307286


namespace pseudocode_output_l307_307186

theorem pseudocode_output :
  let s := 0
  let t := 1
  let (s, t) := (List.range 3).foldl (fun (s, t) i => (s + (i + 1), t * (i + 1))) (s, t)
  let r := s * t
  r = 36 :=
by
  sorry

end pseudocode_output_l307_307186


namespace good_numbers_count_l307_307212

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l307_307212


namespace good_numbers_2020_has_count_10_l307_307205

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l307_307205


namespace relationship_between_x_y_z_l307_307771

noncomputable def x := Real.sqrt 0.82
noncomputable def y := Real.sin 1
noncomputable def z := Real.log 7 / Real.log 3

theorem relationship_between_x_y_z : y < z ∧ z < x := 
by sorry

end relationship_between_x_y_z_l307_307771


namespace completing_the_square_l307_307065

theorem completing_the_square (x : ℝ) :
  x^2 - 6 * x + 2 = 0 →
  (x - 3)^2 = 7 :=
by sorry

end completing_the_square_l307_307065


namespace library_pupils_count_l307_307403

-- Definitions for the conditions provided in the problem
def num_rectangular_tables : Nat := 7
def num_pupils_per_rectangular_table : Nat := 10
def num_square_tables : Nat := 5
def num_pupils_per_square_table : Nat := 4

-- Theorem stating the problem's question and the required proof
theorem library_pupils_count :
  num_rectangular_tables * num_pupils_per_rectangular_table + 
  num_square_tables * num_pupils_per_square_table = 90 :=
sorry

end library_pupils_count_l307_307403


namespace stephanie_gas_payment_l307_307040

variables (electricity_bill : ℕ) (gas_bill : ℕ) (water_bill : ℕ) (internet_bill : ℕ)
variables (electricity_paid : ℕ) (gas_paid_fraction : ℚ) (water_paid_fraction : ℚ) (internet_paid : ℕ)
variables (additional_gas_payment : ℕ) (remaining_payment : ℕ) (expected_remaining : ℕ)

def stephanie_budget : Prop :=
  electricity_bill = 60 ∧
  electricity_paid = 60 ∧
  gas_bill = 40 ∧
  gas_paid_fraction = 3/4 ∧
  water_bill = 40 ∧
  water_paid_fraction = 1/2 ∧
  internet_bill = 25 ∧
  internet_paid = 4 * 5 ∧
  remaining_payment = 30 ∧
  expected_remaining = 
    (gas_bill - gas_paid_fraction * gas_bill) +
    (water_bill - water_paid_fraction * water_bill) + 
    (internet_bill - internet_paid) - 
    additional_gas_payment ∧
  expected_remaining = remaining_payment

theorem stephanie_gas_payment : additional_gas_payment = 5 :=
by sorry

end stephanie_gas_payment_l307_307040


namespace extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l307_307952

-- Define the function f(x) = 2*x^3 + 3*(a-2)*x^2 - 12*a*x
def f (x : ℝ) (a : ℝ) := 2*x^3 + 3*(a-2)*x^2 - 12*a*x

-- Define the function f(x) when a = 0
def f_a_zero (x : ℝ) := f x 0

-- Define the intervals and extreme values problem
theorem extreme_values_of_f_a_zero_on_interval :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 4, f_a_zero x ≤ max ∧ f_a_zero x ≥ min) ∧ max = 32 ∧ min = -40 :=
sorry

-- Define the function for the derivative of f(x)
def f_derivative (x : ℝ) (a : ℝ) := 6*x^2 + 6*(a-2)*x - 12*a

-- Prove the monotonicity based on the value of a
theorem monotonicity_of_f (a : ℝ) :
  (a > -2 → (∀ x, x < -a → f_derivative x a > 0) ∧ (∀ x, -a < x ∧ x < 2 → f_derivative x a < 0) ∧ (∀ x, x > 2 → f_derivative x a > 0)) ∧
  (a = -2 → ∀ x, f_derivative x a ≥ 0) ∧
  (a < -2 → (∀ x, x < 2 → f_derivative x a > 0) ∧ (∀ x, 2 < x ∧ x < -a → f_derivative x a < 0) ∧ (∀ x, x > -a → f_derivative x a > 0)) :=
sorry

end extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l307_307952


namespace range_of_a_l307_307947

theorem range_of_a (a : ℝ) : 
  (∀ x1 x2 : ℝ, (x1 + x2 = -2 * a) ∧ (x1 * x2 = 1) ∧ (x1 < 0) ∧ (x2 < 0)) ↔ (a ≥ 1) :=
by
  sorry

end range_of_a_l307_307947


namespace xy_value_l307_307150

theorem xy_value (x y : ℝ) (h : |x - 1| + (x + y)^2 = 0) : x * y = -1 := 
by
  sorry

end xy_value_l307_307150


namespace original_number_is_25_l307_307088

theorem original_number_is_25 (x : ℕ) (h : ∃ n : ℕ, (x^2 - 600)^n = x) : x = 25 :=
sorry

end original_number_is_25_l307_307088


namespace remainder_of_2345678_div_5_l307_307718

theorem remainder_of_2345678_div_5 : (2345678 % 5) = 3 :=
by
  sorry

end remainder_of_2345678_div_5_l307_307718


namespace smallest_next_divisor_221_l307_307821

structure Conditions (m : ℕ) :=
  (m_even : m % 2 = 0)
  (m_4digit : 1000 ≤ m ∧ m < 10000)
  (m_div_221 : 221 ∣ m)

theorem smallest_next_divisor_221 (m : ℕ) (h : Conditions m) : ∃ k, k > 221 ∧ k ∣ m ∧ k = 289 := by
  sorry

end smallest_next_divisor_221_l307_307821


namespace speed_in_first_hour_l307_307200

variable (x : ℕ)
-- Conditions: 
-- The speed of the car in the second hour:
def speed_in_second_hour : ℕ := 30
-- The average speed of the car:
def average_speed : ℕ := 60
-- The total time traveled:
def total_time : ℕ := 2

-- Proof problem: Prove that the speed of the car in the first hour is 90 km/h.
theorem speed_in_first_hour : x + speed_in_second_hour = average_speed * total_time → x = 90 := 
by 
  intro h
  sorry

end speed_in_first_hour_l307_307200


namespace range_of_a_l307_307782

theorem range_of_a (a : ℝ) : (∃ (x : ℤ), x > 1 ∧ x ≤ a) → ∃ (x : ℤ), (x = 2 ∨ x = 3 ∨ x = 4) ∧ 4 ≤ a ∧ a < 5 :=
by
  sorry

end range_of_a_l307_307782


namespace divergence_of_vector_field_l307_307714

noncomputable def vector_field (x : ℝ) : ℝ → ℝ := λ x, x

def sphere_radius (ε : ℝ) : ℝ := (4 / 3) * π * ε^3

def divergence_at_origin (ε : ℝ) (a : ℝ → ℝ) : ℝ :=
  let surface_integral := (∫ (σ : set ℝ), (a σ) * (σ / |σ|)) in
  (surface_integral / sphere_radius ε)

theorem divergence_of_vector_field : ∀ (a : ℝ → ℝ),
  (a = vector_field) →
  ∀ ε > 0,
  (divergence_at_origin ε a) = 1 := 
by
  intros a ha ε ε_pos
  have h_surface_integral : (∫ (σ : set ℝ), (a σ) * (σ / |σ|)) = (4 / 3) * π * ε^3,
  sorry
  rw [divergence_at_origin, h_surface_integral, sphere_radius]
  field_simp [ε_pos],
  rw [divergence_at_origin],
  field_simp [sphere_radius, ε_pos],
  sorry -- further simplifications here.

end divergence_of_vector_field_l307_307714


namespace sqrt_equality_l307_307799

theorem sqrt_equality (n : ℤ) (h : Real.sqrt (8 + n) = 9) : n = 73 :=
by
  sorry

end sqrt_equality_l307_307799


namespace bn_is_arithmetic_an_general_formula_l307_307834

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}

-- Conditions
def condition (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) :=
  ∀ n, (2 / (S n)) + (1 / (b n)) = 2

-- Theorem 1: The sequence {b_n} is an arithmetic sequence
theorem bn_is_arithmetic (h : condition n S b) :
  ∃ d : ℝ, ∀ n, b n = b 1 + (n - 1) * d :=
sorry

-- Theorem 2: General formula for {a_n}
theorem an_general_formula (h : condition n S b) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
sorry

end bn_is_arithmetic_an_general_formula_l307_307834


namespace largest_five_digit_integer_with_digit_product_proof_l307_307061

noncomputable def largest_five_digit_integer_with_digit_product : ℕ :=
  98752

theorem largest_five_digit_integer_with_digit_product_proof :
  ∃ n : ℕ, (n >= 10000) ∧ (n < 100000) ∧ 
           (∃ (digits : list ℕ), (n = digits.foldl (λ acc d, acc * 10 + d) 0) ∧
           (digits.prod = (8 * 7 * 6 * 5 * 4 * 3 * 2 * 1)) ∧
           (n = 98752)) :=
by {
    let answer := 98752,
    use answer,
    split, { norm_num },
    split, { norm_num },
    use [9, 8, 7, 5, 2],
    split, { refl },
    split, {
        rw [list.prod_cons, list.prod_cons, list.prod_cons, list.prod_cons, list.prod_nil],
        norm_num,
    },
    refl,
}

end largest_five_digit_integer_with_digit_product_proof_l307_307061


namespace xiao_gang_steps_l307_307008

theorem xiao_gang_steps (x : ℕ) (H1 : 9000 / x = 13500 / (x + 15)) : x = 30 :=
by
  sorry

end xiao_gang_steps_l307_307008


namespace garden_length_is_60_l307_307909

noncomputable def garden_length (w l : ℕ) : Prop :=
  l = 2 * w ∧ 2 * w + 2 * l = 180

theorem garden_length_is_60 (w l : ℕ) (h : garden_length w l) : l = 60 :=
by
  sorry

end garden_length_is_60_l307_307909


namespace sufficient_condition_l307_307450

theorem sufficient_condition (a b : ℝ) (h : |a + b| > 1) : |a| + |b| > 1 := 
by sorry

end sufficient_condition_l307_307450


namespace track_length_l307_307746

theorem track_length (x : ℝ) : 
  (∃ B S : ℝ, B + S = x ∧ S = (x / 2 - 75) ∧ B = 75 ∧ S + 100 = x / 2 + 25 ∧ B = x / 2 - 50 ∧ B / S = (x / 2 - 50) / 100) → 
  x = 220 :=
by
  sorry

end track_length_l307_307746


namespace longest_side_of_quadrilateral_l307_307267

theorem longest_side_of_quadrilateral :
  ∀ (x y : ℝ), 
    x + y ≤ 4 →
    3 * x + y ≥ 3 →
    0 ≤ x →
    0 ≤ y →
    ∃ (longest_side_length : ℝ), longest_side_length = 5 :=
by
  intros x y h1 h2 h3 h4
  let vertices := [(0, 0), (1, 0), (4, 0), (0, 3)]
  let distance (p1 p2 : ℝ × ℝ) : ℝ := 
    real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)
  let side_lengths := [distance (0, 0) (1, 0), distance (1, 0) (4, 0), distance (4, 0) (0, 3), distance (0, 3) (0, 0)]
  have max_side_length : list ℝ := list.maximum side_lengths
  exact ⟨max_side_length, sorry⟩

end longest_side_of_quadrilateral_l307_307267


namespace larry_substituted_value_l307_307174

theorem larry_substituted_value :
  ∀ (a b c d e : ℤ), a = 5 → b = 3 → c = 4 → d = 2 → e = 2 → 
  (a + b - c + d - e = a + (b - (c + (d - e)))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end larry_substituted_value_l307_307174


namespace half_abs_diff_squares_eq_40_l307_307228

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l307_307228


namespace three_digit_numbers_sum_seven_l307_307637

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l307_307637


namespace globe_surface_area_l307_307083

theorem globe_surface_area (d : ℚ) (h : d = 9) : 
  4 * Real.pi * (d / 2) ^ 2 = 81 * Real.pi := 
by 
  sorry

end globe_surface_area_l307_307083


namespace xiao_ming_kite_payment_l307_307070

/-- Xiao Ming has multiple 1 yuan, 2 yuan, and 5 yuan banknotes. 
    He wants to buy a kite priced at 18 yuan using no more than 10 of these banknotes
    and must use at least two different denominations.
    Prove that there are exactly 11 different ways he can pay. -/
theorem xiao_ming_kite_payment : 
  ∃ (combinations : Nat), 
    (∀ (c1 c2 c5 : Nat), (c1 * 1 + c2 * 2 + c5 * 5 = 18) → 
    (c1 + c2 + c5 ≤ 10) → 
    ((c1 > 0 ∧ c2 > 0) ∨ (c1 > 0 ∧ c5 > 0) ∨ (c2 > 0 ∧ c5 > 0)) →
    combinations = 11) :=
sorry

end xiao_ming_kite_payment_l307_307070


namespace arrangement_exists_for_P_eq_23_l307_307988

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l307_307988


namespace icosahedron_probability_div_by_three_at_least_one_fourth_l307_307513
open ProbabilityTheory

theorem icosahedron_probability_div_by_three_at_least_one_fourth (a b c : ℕ) (h : a + b + c = 20) :
  (a^3 + b^3 + c^3 + 6 * a * b * c : ℚ) / (a + b + c)^3 ≥ 1 / 4 :=
sorry

end icosahedron_probability_div_by_three_at_least_one_fourth_l307_307513


namespace banana_equivalence_l307_307489

theorem banana_equivalence :
  (3 / 4 : ℚ) * 12 = 9 → (1 / 3 : ℚ) * 6 = 2 :=
by
  intro h1
  linarith

end banana_equivalence_l307_307489


namespace b_arithmetic_a_general_formula_l307_307823

section sum_and_product_sequences

/- Definitions for sequences -/
def S (a : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).sum (λ i, a (i + 1))
def b (S : ℕ → ℚ) (n : ℕ) : ℚ := (List.range n).prod (λ i, S (i + 1))

/- Given condition -/
axiom condition (a : ℕ → ℚ) (n : ℕ) : (n > 0) → (2 / S a n) + (1 / b (S a) n) = 2

/- Proof statements -/
theorem b_arithmetic (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (∀ n ≥ 2, b (S a) n - b (S a) (n - 1) = 1 / 2) ∧ b (S a) 1 = 3 / 2 :=
  sorry

theorem a_general_formula (a : ℕ → ℚ) (h : ∀ n > 0, (2 / S a n) + (1 / b (S a) n) = 2) :
  (a 1 = 3 / 2) ∧ (∀ n ≥ 2, a n = -1 / (n * (n + 1))) :=
  sorry

end sum_and_product_sequences

end b_arithmetic_a_general_formula_l307_307823


namespace sum_n_binom_30_15_eq_31_16_l307_307517

open Nat

-- Given n = 30 and k = 15, we are given the components to test Pascal's identity
def PascalIdentity (n k : Nat) : Prop :=
  Nat.choose (n-1) (k-1) + Nat.choose (n-1) k = Nat.choose n k

theorem sum_n_binom_30_15_eq_31_16 : 
  (∑ n in { n : ℕ | Nat.choose 30 15 + Nat.choose 30 n = Nat.choose 31 16 }, n) = 30 := 
sorry

end sum_n_binom_30_15_eq_31_16_l307_307517


namespace smallest_class_number_selected_l307_307247

theorem smallest_class_number_selected
  {n k : ℕ} (hn : n = 30) (hk : k = 5) (h_sum : ∃ x : ℕ, x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 75) :
  ∃ x : ℕ, x = 3 := 
sorry

end smallest_class_number_selected_l307_307247


namespace sqrt_one_half_eq_sqrt_two_over_two_l307_307893

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end sqrt_one_half_eq_sqrt_two_over_two_l307_307893


namespace kerosene_cost_l307_307892

/-- A dozen eggs cost as much as a pound of rice, a half-liter of kerosene costs as much as 8 eggs,
and each pound of rice costs $0.33. Prove that a liter of kerosene costs 44 cents. -/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost := half_liter_kerosene_cost * 2
  liter_kerosene_cost * 100 = 44 := 
by
  sorry

end kerosene_cost_l307_307892


namespace sheela_overall_total_income_l307_307853

def monthly_income_in_rs (income: ℝ) (savings: ℝ) (percent: ℝ): Prop :=
  savings = percent * income

def overall_total_income_in_rs (monthly_income: ℝ) 
                              (savings_deposit: ℝ) (fd_deposit: ℝ) 
                              (savings_interest_rate_monthly: ℝ) 
                              (fd_interest_rate_annual: ℝ): ℝ :=
  let annual_income := monthly_income * 12
  let savings_interest := savings_deposit * (savings_interest_rate_monthly * 12)
  let fd_interest := fd_deposit * fd_interest_rate_annual
  annual_income + savings_interest + fd_interest

theorem sheela_overall_total_income:
  ∀ (monthly_income: ℝ)
    (savings_deposit: ℝ) (fd_deposit: ℝ)
    (savings_interest_rate_monthly: ℝ) (fd_interest_rate_annual: ℝ),
    (monthly_income_in_rs monthly_income savings_deposit 0.28)  →
    monthly_income = 16071.43 →
    savings_deposit = 4500 →
    fd_deposit = 3000 →
    savings_interest_rate_monthly = 0.02 →
    fd_interest_rate_annual = 0.06 →
    overall_total_income_in_rs monthly_income savings_deposit fd_deposit
                           savings_interest_rate_monthly fd_interest_rate_annual
    = 194117.16 := 
by
  intros
  sorry

end sheela_overall_total_income_l307_307853


namespace solve_for_y_l307_307859

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l307_307859


namespace expected_value_unfair_die_l307_307018

open ProbabilityTheory

noncomputable def expected_value (pmf : pmf ℕ) : ℝ :=
∑' i, (i:ℝ) * pmf i

theorem expected_value_unfair_die :
  let pmf := @pmf.of_finite_support _ _ _ _ ⟨
    { val := ![
      (1, 2/21:ℝ),
      (2, 2/21:ℝ),
      (3, 2/21:ℝ),
      (4, 2/21:ℝ),
      (5, 2/21:ℝ),
      (6, 2/21:ℝ),
      (7, 2/21:ℝ),
      (8, 1/3:ℝ)
    ], nodup := by simp [finset.nodup_map_on prod.fst_injective (finset.range 8).nodup] }⟩ in
  expected_value pmf = 16 / 3 :=
begin
  sorry
end

end expected_value_unfair_die_l307_307018


namespace tom_drives_distance_before_karen_wins_l307_307327

def karen_late_minutes := 4
def karen_speed_mph := 60
def tom_speed_mph := 45

theorem tom_drives_distance_before_karen_wins : 
  ∃ d : ℝ, d = 21 := by
  sorry

end tom_drives_distance_before_karen_wins_l307_307327


namespace distance_sum_is_ten_l307_307539

noncomputable def angle_sum_distance (C A B : ℝ) (d : ℝ) (k : ℝ) : ℝ := 
  let h_A : ℝ := sorry -- replace with expression for h_A based on conditions
  let h_B : ℝ := sorry -- replace with expression for h_B based on conditions
  h_A + h_B

theorem distance_sum_is_ten 
  (A B C : ℝ) 
  (h : ℝ) 
  (k : ℝ) 
  (h_pos : h = 4) 
  (ratio_condition : h_A = 4 * h_B)
  : angle_sum_distance C A B h k = 10 := 
  sorry

end distance_sum_is_ten_l307_307539


namespace sqrt_170569_sqrt_175561_l307_307756

theorem sqrt_170569 : Nat.sqrt 170569 = 413 := 
by 
  sorry 

theorem sqrt_175561 : Nat.sqrt 175561 = 419 := 
by 
  sorry

end sqrt_170569_sqrt_175561_l307_307756


namespace how_long_it_lasts_l307_307338

-- Define a structure to hold the conditions
structure MoneySpending where
  mowing_income : ℕ
  weeding_income : ℕ
  weekly_expense : ℕ

-- Example conditions given in the problem
def lukesEarnings : MoneySpending :=
{ mowing_income := 9,
  weeding_income := 18,
  weekly_expense := 3 }

-- Main theorem proving the number of weeks he can sustain his spending
theorem how_long_it_lasts (data : MoneySpending) : 
  (data.mowing_income + data.weeding_income) / data.weekly_expense = 9 := by
  sorry

end how_long_it_lasts_l307_307338


namespace senior_ticket_cost_l307_307512

theorem senior_ticket_cost (total_tickets : ℕ) (adult_ticket_cost : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) (senior_ticket_cost : ℕ) :
  total_tickets = 510 →
  adult_ticket_cost = 21 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  senior_ticket_cost = 15 :=
by
  sorry

end senior_ticket_cost_l307_307512


namespace eval_expression_l307_307232

theorem eval_expression (a b : ℤ) (h₁ : a = 4) (h₂ : b = -2) : -a - b^2 + a*b + a^2 = 0 := by
  sorry

end eval_expression_l307_307232


namespace problem_rational_sum_of_powers_l307_307448

theorem problem_rational_sum_of_powers :
  ∃ (a b : ℚ), (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 ∧ a + b = 70 :=
by
  sorry

end problem_rational_sum_of_powers_l307_307448


namespace total_guests_l307_307386

-- Define the conditions.
def number_of_tables := 252.0
def guests_per_table := 4.0

-- Define the statement to prove.
theorem total_guests : number_of_tables * guests_per_table = 1008.0 := by
  sorry

end total_guests_l307_307386


namespace aubrey_distance_from_school_l307_307098

-- Define average speed and travel time
def average_speed : ℝ := 22 -- in miles per hour
def travel_time : ℝ := 4 -- in hours

-- Define the distance function
def calc_distance (speed time : ℝ) : ℝ := speed * time

-- State the theorem
theorem aubrey_distance_from_school : calc_distance average_speed travel_time = 88 := 
by
  sorry

end aubrey_distance_from_school_l307_307098


namespace proof_correct_word_choice_l307_307239

def sentence_completion_correct (word : String) : Prop :=
  "Most of them are kind, but " ++ word ++ " is so good to me as Bruce" = "Most of them are kind, but none is so good to me as Bruce"

theorem proof_correct_word_choice : 
  (sentence_completion_correct "none") → 
  ("none" = "none") := 
by
  sorry

end proof_correct_word_choice_l307_307239


namespace nth_equation_l307_307345

theorem nth_equation (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end nth_equation_l307_307345


namespace sum_of_possible_x_values_l307_307719

theorem sum_of_possible_x_values : 
  let lst : List ℝ := [10, 2, 5, 2, 4, 2, x]
  let mean := (25 + x) / 7
  let mode := 2
  let median := if x ≤ 2 then 2 else if 2 < x ∧ x < 4 then x else 4
  mean, median, and mode form a non-constant arithmetic progression 
  -> ∃ x_values : List ℝ, sum x_values = 20 :=
by
  sorry

end sum_of_possible_x_values_l307_307719


namespace solutions_to_quadratic_l307_307358

noncomputable def a : ℝ := (6 + Real.sqrt 92) / 2
noncomputable def b : ℝ := (6 - Real.sqrt 92) / 2

theorem solutions_to_quadratic :
  a ≥ b ∧ ((∀ x : ℝ, x^2 - 6 * x + 11 = 25 → x = a ∨ x = b) → 3 * a + 2 * b = 15 + Real.sqrt 92 / 2) := by
  sorry

end solutions_to_quadratic_l307_307358


namespace tulips_sum_l307_307096

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum_l307_307096


namespace isosceles_triangle_perimeter_l307_307546

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4) (h2 : b = 6) : 
  ∃ p, (p = 14 ∨ p = 16) :=
by
  sorry

end isosceles_triangle_perimeter_l307_307546


namespace sum_of_coefficients_eq_10_l307_307473

theorem sum_of_coefficients_eq_10 
  (s : ℕ → ℝ) 
  (a b c : ℝ) 
  (h0 : s 0 = 3) 
  (h1 : s 1 = 5) 
  (h2 : s 2 = 9)
  (h : ∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) : 
  a + b + c = 10 :=
sorry

end sum_of_coefficients_eq_10_l307_307473


namespace complex_simplify_l307_307485

theorem complex_simplify :
  10.25 * Real.sqrt 6 * Complex.exp (Complex.I * 160 * Real.pi / 180)
  / (Real.sqrt 3 * Complex.exp (Complex.I * 40 * Real.pi / 180))
  = (-Real.sqrt 2 / 2) + Complex.I * (Real.sqrt 6 / 2) := by
  sorry

end complex_simplify_l307_307485


namespace KodyAgeIs32_l307_307641

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end KodyAgeIs32_l307_307641


namespace regular_polygon_interior_angle_of_108_has_5_sides_l307_307811

theorem regular_polygon_interior_angle_of_108_has_5_sides (interior_angle : ℝ) 
  (h : interior_angle = 108) : 
  let exterior_angle := 180 - interior_angle in
  let n := 360 / exterior_angle in
  n = 5 := 
by 
  unfold exterior_angle n
  rw [h, sub_eq_add_neg, add_neg_eq_sub]
  norm_num
sorry

end regular_polygon_interior_angle_of_108_has_5_sides_l307_307811


namespace sum_of_coefficients_eq_two_l307_307120

noncomputable def poly_eq (a b c d : ℤ) : Prop :=
  (X^2 + a * X + b) * (X^2 + c * X + d) = X^4 - 2 * X^3 + 3 * X^2 - 4 * X + 6

theorem sum_of_coefficients_eq_two (a b c d : ℤ) (h : poly_eq a b c d) : a + b + c + d = 2 :=
by sorry

end sum_of_coefficients_eq_two_l307_307120


namespace f_zero_derivative_not_extremum_l307_307495

noncomputable def f (x : ℝ) : ℝ := x ^ 3

theorem f_zero_derivative_not_extremum (x : ℝ) : 
  deriv f 0 = 0 ∧ ∀ (y : ℝ), y ≠ 0 → (∃ δ > 0, ∀ z, abs (z - 0) < δ → (f z / z : ℝ) ≠ 0) :=
by
  sorry

end f_zero_derivative_not_extremum_l307_307495


namespace no_integer_solutions_l307_307039

theorem no_integer_solutions : ¬∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := 
by
  sorry

end no_integer_solutions_l307_307039


namespace sequence_formula_l307_307663

theorem sequence_formula (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h : ∀ n : ℕ, S n = 3 * a n + (-1)^n) :
  ∀ n : ℕ, a n = (1/10) * (3/2)^(n-1) - (2/5) * (-1)^n :=
by sorry

end sequence_formula_l307_307663


namespace derivative_at_0_l307_307134

noncomputable def f (x : ℝ) := Real.exp x / (x + 2)

theorem derivative_at_0 : deriv f 0 = 1 / 4 := sorry

end derivative_at_0_l307_307134


namespace part1_arithmetic_sequence_part2_general_formula_l307_307829

variable {a : ℕ → ℝ} -- The sequence {a_n}
variable {S : ℕ → ℝ} -- The sequence {S_n}
variable {b : ℕ → ℝ} -- The sequence {b_n}

-- Conditions
def condition1 (n : ℕ) : Prop := S n = (∑ i in finset.range n, a i)
def condition2 (n : ℕ) : Prop := b n = (∏ i in finset.range n, S i)
def condition3 (n : ℕ) : Prop := 2 / S n + 1 / b n = 2

-- Part (1) Prove {b_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∃ d : ℝ, ∃ a₀ : ℝ, (∀ n : ℕ, b n = a₀ + n * d) := by 
  sorry

-- Part (2) Find the general formula for {a_n}
theorem part2_general_formula (h1 : ∀ n, condition1 n) (h2 : ∀ n, condition2 n) (h3 : ∀ n, condition3 n) : 
  ∀ n, a n = 
    if n = 1 then 3/2 
    else -1 / (n * (n + 1)) := by 
  sorry

end part1_arithmetic_sequence_part2_general_formula_l307_307829


namespace faye_age_l307_307552

variables (C D E F G : ℕ)
variables (h1 : D = E - 2)
variables (h2 : E = C + 6)
variables (h3 : F = C + 4)
variables (h4 : G = C - 5)
variables (h5 : D = 16)

theorem faye_age : F = 16 :=
by
  -- Proof will be placed here
  sorry

end faye_age_l307_307552


namespace difference_between_sums_l307_307932

open Nat

-- Sum of the first 'n' positive odd integers formula: n^2
def sum_of_first_odd (n : ℕ) : ℕ := n * n

-- Sum of the first 'n' positive even integers formula: n(n+1)
def sum_of_first_even (n : ℕ) : ℕ := n * (n + 1)

-- The main theorem stating the difference between the sums
theorem difference_between_sums (n : ℕ) (h : n = 3005) :
  sum_of_first_even n - sum_of_first_odd n = 3005 :=
by
  sorry

end difference_between_sums_l307_307932


namespace problem_statement_l307_307269

-- Define the operation * based on the given mathematical definition
def op (a b : ℕ) : ℤ := a * (a - b)

-- The core theorem to prove the expression in the problem
theorem problem_statement : op 2 3 + op (6 - 2) 4 = -2 :=
by
  -- This is where the proof would go, but it's omitted with sorry.
  sorry

end problem_statement_l307_307269


namespace sum_of_areas_of_circles_l307_307382

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l307_307382


namespace good_numbers_2020_has_count_10_l307_307206

theorem good_numbers_2020_has_count_10 :
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  set.count divisors = 10 :=
by
  let n := 2020
  let r := 22
  let k := 1998
  let divisors := {d : ℕ | d > 22 ∧ k % d = 0}
  sorry

end good_numbers_2020_has_count_10_l307_307206


namespace new_average_after_doubling_l307_307525

theorem new_average_after_doubling (n : ℕ) (avg : ℝ) (h_n : n = 12) (h_avg : avg = 50) :
  2 * avg = 100 :=
by
  sorry

end new_average_after_doubling_l307_307525


namespace find_a_from_derivative_l307_307135

-- Define the function f(x) = ax^3 + 3x^2 - 6
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

-- State the theorem to prove that a = 10/3 given f'(-1) = 4
theorem find_a_from_derivative (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 := 
  sorry

end find_a_from_derivative_l307_307135


namespace non_shaded_perimeter_6_l307_307042

theorem non_shaded_perimeter_6 
  (area_shaded : ℝ) (area_large_rect : ℝ) (area_extension : ℝ) (total_area : ℝ)
  (non_shaded_area : ℝ) (perimeter : ℝ) :
  area_shaded = 104 → 
  area_large_rect = 12 * 8 → 
  area_extension = 5 * 2 → 
  total_area = area_large_rect + area_extension → 
  non_shaded_area = total_area - area_shaded → 
  non_shaded_area = 2 → 
  perimeter = 2 * (2 + 1) → 
  perimeter = 6 := 
by 
  sorry

end non_shaded_perimeter_6_l307_307042


namespace relationship_among_a_b_c_l307_307292

theorem relationship_among_a_b_c :
  let a := (1/6) ^ (1/2)
  let b := Real.log (1/3) / Real.log 6
  let c := Real.log (1/7) / Real.log (1/6)
  c > a ∧ a > b :=
by
  sorry

end relationship_among_a_b_c_l307_307292


namespace vertex_of_parabola_l307_307699

theorem vertex_of_parabola : ∀ x y : ℝ, y = 2 * (x - 1) ^ 2 + 2 → (1, 2) = (1, 2) :=
by
  sorry

end vertex_of_parabola_l307_307699


namespace equilateral_triangle_sum_l307_307017

theorem equilateral_triangle_sum (a u v w : ℝ)
  (h1: u^2 + v^2 = w^2):
  w^2 + Real.sqrt 3 * u * v = a^2 := 
sorry

end equilateral_triangle_sum_l307_307017


namespace star_computation_l307_307565

def star (x y : ℝ) (h : x ≠ y) : ℝ := (x + y) / (x - y)

theorem star_computation :
  star (star (-1) 4 (by norm_num)) (star (-5) 2 (by norm_num)) (by norm_num) = 1 / 6 := 
sorry

end star_computation_l307_307565


namespace expand_polynomial_correct_l307_307559

open Polynomial

noncomputable def expand_polynomial : Polynomial ℤ :=
  (C 3 * X^3 - C 2 * X^2 + X - C 4) * (C 4 * X^2 - C 2 * X + C 5)

theorem expand_polynomial_correct :
  expand_polynomial = C 12 * X^5 - C 14 * X^4 + C 23 * X^3 - C 28 * X^2 + C 13 * X - C 20 :=
by sorry

end expand_polynomial_correct_l307_307559


namespace positive_solution_iff_abs_a_b_lt_one_l307_307480

theorem positive_solution_iff_abs_a_b_lt_one
  (a b : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 - x2 = a)
  (h2 : x3 - x4 = b)
  (h3 : x1 + x2 + x3 + x4 = 1)
  (h4 : x1 > 0)
  (h5 : x2 > 0)
  (h6 : x3 > 0)
  (h7 : x4 > 0) :
  |a| + |b| < 1 :=
sorry

end positive_solution_iff_abs_a_b_lt_one_l307_307480


namespace min_value_of_sum_of_sides_proof_l307_307006

noncomputable def min_value_of_sum_of_sides (a b c : ℝ) (angleC : ℝ) : ℝ :=
  if (angleC = 60 * (Real.pi / 180)) ∧ ((a + b)^2 - c^2 = 4) then 4 * Real.sqrt 3 / 3 
  else 0

theorem min_value_of_sum_of_sides_proof (a b c : ℝ) (angleC : ℝ) 
  (h1 : angleC = 60 * (Real.pi / 180)) 
  (h2 : (a + b)^2 - c^2 = 4) 
  : min_value_of_sum_of_sides a b c angleC = 4 * Real.sqrt 3 / 3 := 
by
  sorry

end min_value_of_sum_of_sides_proof_l307_307006


namespace fraction_playing_in_field_l307_307311

def class_size : ℕ := 50
def students_painting : ℚ := 3/5
def students_left_in_classroom : ℕ := 10

theorem fraction_playing_in_field :
  (class_size - students_left_in_classroom - students_painting * class_size) / class_size = 1/5 :=
by
  sorry

end fraction_playing_in_field_l307_307311


namespace probability_B_does_not_lose_l307_307058

def prob_A_wins : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Theorem: the probability that B does not lose is 70%.
theorem probability_B_does_not_lose : prob_A_wins + prob_draw ≤ 1 → 1 - prob_A_wins - (1 - prob_draw - prob_A_wins) = 0.7 := by
  sorry

end probability_B_does_not_lose_l307_307058


namespace circles_area_sum_l307_307368

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l307_307368


namespace exists_negative_fraction_lt_four_l307_307068

theorem exists_negative_fraction_lt_four : 
  ∃ (x : ℚ), x < 0 ∧ |x| < 4 := 
sorry

end exists_negative_fraction_lt_four_l307_307068


namespace vasya_misha_expected_coincidences_l307_307397

noncomputable def expected_coincidences (n : ℕ) (pA pB : ℝ) : ℝ :=
  n * ((pA * pB) + ((1 - pA) * (1 - pB)))

theorem vasya_misha_expected_coincidences :
  expected_coincidences 20 (6 / 20) (8 / 20) = 10.8 :=
by
  -- Test definition and expected output
  let n := 20
  let pA := 6 / 20
  let pB := 8 / 20
  have h : expected_coincidences n pA pB =  20 * ((pA * pB) + ((1 - pA) * (1 - pB))) := rfl
  rw h
  sorry

end vasya_misha_expected_coincidences_l307_307397


namespace number_of_pairs_l307_307149

theorem number_of_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m^2 + n < 50) : 
  ∃! p : ℕ, p = 203 := 
sorry

end number_of_pairs_l307_307149


namespace green_peppers_weight_l307_307426

theorem green_peppers_weight (total_weight : ℝ) (w : ℝ) (h1 : total_weight = 5.666666667)
  (h2 : 2 * w = total_weight) : w = 2.8333333335 :=
by
  sorry

end green_peppers_weight_l307_307426


namespace tangent_line_equation_at_point_l307_307044

theorem tangent_line_equation_at_point {x y : ℝ} (h_curve : y = x * (3 * Real.log x + 1))
  (h_point : (1, 1)) :
  ∃ m b, m = 4 ∧ b = -3 ∧ (∀ x y, y = m * x + b) := by
  sorry

end tangent_line_equation_at_point_l307_307044


namespace max_pages_within_budget_l307_307678

-- Definitions based on the problem conditions
def page_cost_in_cents : ℕ := 5
def total_budget_in_cents : ℕ := 5000
def max_expenditure_in_cents : ℕ := 4500

-- Proof problem statement
theorem max_pages_within_budget : 
  ∃ (pages : ℕ), pages = max_expenditure_in_cents / page_cost_in_cents ∧ 
                  pages * page_cost_in_cents ≤ total_budget_in_cents :=
by {
  sorry
}

end max_pages_within_budget_l307_307678


namespace solve_for_k_l307_307964

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end solve_for_k_l307_307964


namespace find_numbers_l307_307255

def is_7_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n < 10000000
def is_14_digit (n : ℕ) : Prop := n >= 10^13 ∧ n < 10^14

theorem find_numbers (x y z : ℕ) (hx7 : is_7_digit x) (hy7 : is_7_digit y) (hz14 : is_14_digit z) :
  3 * x * y = z ∧ z = 10^7 * x + y → 
  x = 1666667 ∧ y = 3333334 ∧ z = 16666673333334 := 
by
  sorry

end find_numbers_l307_307255


namespace derivative_at_pi_over_4_l307_307951

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 : 
  deriv f (Real.pi / 4) = Real.sqrt 2 / 2 + Real.sqrt 2 * Real.pi / 8 :=
by
  -- Proof goes here
  sorry

end derivative_at_pi_over_4_l307_307951


namespace anna_money_left_l307_307094

theorem anna_money_left : 
  let initial_money := 10.0
  let gum_cost := 3.0 -- 3 packs at $1.00 each
  let chocolate_cost := 5.0 -- 5 bars at $1.00 each
  let cane_cost := 1.0 -- 2 canes at $0.50 each
  let total_spent := gum_cost + chocolate_cost + cane_cost
  let money_left := initial_money - total_spent
  money_left = 1.0 := by
  sorry

end anna_money_left_l307_307094


namespace anicka_savings_l307_307745

theorem anicka_savings (x y : ℕ) (h1 : x + y = 290) (h2 : (1/4 : ℚ) * (2 * y) = (1/3 : ℚ) * x) : 2 * y + x = 406 :=
by
  sorry

end anicka_savings_l307_307745


namespace solve_x_value_l307_307790
-- Import the necessary libraries

-- Define the problem and the main theorem
theorem solve_x_value (x : ℝ) (h : 3 / x^2 = x / 27) : x = 3 * Real.sqrt 3 :=
by
  sorry

end solve_x_value_l307_307790


namespace Jillian_had_200_friends_l307_307318

def oranges : ℕ := 80
def pieces_per_orange : ℕ := 10
def pieces_per_friend : ℕ := 4
def number_of_friends : ℕ := oranges * pieces_per_orange / pieces_per_friend

theorem Jillian_had_200_friends :
  number_of_friends = 200 :=
sorry

end Jillian_had_200_friends_l307_307318


namespace exists_increasing_sequence_l307_307119

theorem exists_increasing_sequence (n : ℕ) : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → ∃ x : ℕ → ℕ, (∀ i : ℕ, 1 ≤ i → i ≤ n → x i < x (i + 1)) :=
by
  sorry

end exists_increasing_sequence_l307_307119


namespace option_a_option_b_option_d_l307_307144

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l307_307144


namespace chickens_and_rabbits_l307_307812

theorem chickens_and_rabbits (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chickens_and_rabbits_l307_307812


namespace Cindy_crayons_l307_307467

variable (K : ℕ) -- Karen's crayons
variable (C : ℕ) -- Cindy's crayons

-- Given conditions
def Karen_has_639_crayons : Prop := K = 639
def Karen_has_135_more_crayons_than_Cindy : Prop := K = C + 135

-- The proof problem: showing Cindy's crayons
theorem Cindy_crayons (h1 : Karen_has_639_crayons K) (h2 : Karen_has_135_more_crayons_than_Cindy K C) : C = 504 :=
by
  sorry

end Cindy_crayons_l307_307467


namespace find_number_l307_307755

theorem find_number (x : ℝ) (h : x / 4 + 15 = 4 * x - 15) : x = 8 :=
sorry

end find_number_l307_307755


namespace overlapped_squares_area_l307_307288

/-- 
Theorem: The area of the figure formed by overlapping four identical squares, 
each with an area of \(3 \, \text{cm}^2\), and with an overlapping region 
that double-counts 6 small squares is \(10.875 \, \text{cm}^2\).
-/
theorem overlapped_squares_area (area_of_square : ℝ) (num_squares : ℕ) (overlap_small_squares : ℕ) :
  area_of_square = 3 → 
  num_squares = 4 → 
  overlap_small_squares = 6 →
  ∃ total_area : ℝ, total_area = (num_squares * area_of_square) - (overlap_small_squares * (area_of_square / 16)) ∧
                         total_area = 10.875 :=
by
  sorry

end overlapped_squares_area_l307_307288


namespace sum_areas_of_circles_l307_307373

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l307_307373


namespace biology_vs_reading_diff_l307_307996

def math_hw_pages : ℕ := 2
def reading_hw_pages : ℕ := 3
def total_hw_pages : ℕ := 15

def biology_hw_pages : ℕ := total_hw_pages - (math_hw_pages + reading_hw_pages)

theorem biology_vs_reading_diff : (biology_hw_pages - reading_hw_pages) = 7 := by
  sorry

end biology_vs_reading_diff_l307_307996


namespace sum_radical_conjugates_l307_307551

theorem sum_radical_conjugates (n : ℝ) (m : ℝ) (h1 : n = 5) (h2 : m = (sqrt 500)) : 
  (n - m) + (n + m) = 10 :=
by 
  rw [h1, h2]
  sorry

end sum_radical_conjugates_l307_307551


namespace three_digit_sum_seven_l307_307598

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l307_307598


namespace sum_of_areas_of_circles_l307_307383

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l307_307383


namespace find_range_of_a_l307_307167

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}

noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem find_range_of_a : {a : ℝ | set_B a ⊆ set_A} = {a : ℝ | a < -1} ∪ {1} :=
by
  sorry

end find_range_of_a_l307_307167


namespace circle_tangent_to_x_axis_at_origin_l307_307453

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h1 : ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0 ∨ y = -D/E ∧ x = 0 ∧ F = 0):
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l307_307453


namespace car_stopping_probability_l307_307923

theorem car_stopping_probability :
  let pG_A := (1 : ℚ) / 3,
      pG_B := (1 : ℚ) / 2,
      pG_C := (2 : ℚ) / 3,
      pR_A := 1 - pG_A,
      pR_B := 1 - pG_B,
      pR_C := 1 - pG_C,
      p_stop_A := pR_A * pG_B * pG_C,
      p_stop_B := pG_A * pR_B * pG_C,
      p_stop_C := pG_A * pG_B * pR_C,
      p_stopping_once := p_stop_A + p_stop_B + p_stop_C
  in p_stopping_once = 7 / 18 := 
by 
  sorry

end car_stopping_probability_l307_307923


namespace alice_number_l307_307544

theorem alice_number (n : ℕ) 
  (h1 : 243 ∣ n) 
  (h2 : 36 ∣ n) 
  (h3 : 1000 < n) 
  (h4 : n < 3000) : 
  n = 1944 ∨ n = 2916 := 
sorry

end alice_number_l307_307544


namespace inequality_solution_l307_307501

theorem inequality_solution (x : ℝ) : 4 * x - 1 < 0 ↔ x < 1 / 4 := 
sorry

end inequality_solution_l307_307501


namespace three_digit_numbers_sum_seven_l307_307636

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l307_307636


namespace repeating_decimal_as_fraction_l307_307936

theorem repeating_decimal_as_fraction :
  ∃ x : ℝ, x = 7.45 ∧ (100 * x - x = 738) → x = 82 / 11 :=
by
  sorry

end repeating_decimal_as_fraction_l307_307936


namespace tens_digit_23_1987_l307_307280

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l307_307280


namespace height_of_smaller_cone_is_18_l307_307903

theorem height_of_smaller_cone_is_18
  (height_frustum : ℝ)
  (area_larger_base : ℝ)
  (area_smaller_base : ℝ) :
  let R := (area_larger_base / π).sqrt
  let r := (area_smaller_base / π).sqrt
  let ratio := r / R
  let H := height_frustum / (1 - ratio)
  let h := ratio * H
  height_frustum = 18 ∧ area_larger_base = 400 * π ∧ area_smaller_base = 100 * π
  → h = 18 := by
  sorry

end height_of_smaller_cone_is_18_l307_307903


namespace three_digit_sum_7_l307_307609

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l307_307609


namespace trigonometric_identity_l307_307118

theorem trigonometric_identity (α : Real) (h : Real.tan (α / 2) = 4) :
    (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85 / 44 := by
  sorry

end trigonometric_identity_l307_307118


namespace division_result_l307_307335

def n : ℕ := 16^1024

theorem division_result : n / 8 = 2^4093 :=
by sorry

end division_result_l307_307335


namespace exists_arrangement_for_P_23_l307_307993

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l307_307993


namespace luna_total_monthly_budget_l307_307339

theorem luna_total_monthly_budget
  (H F phone_bill : ℝ)
  (h1 : F = 0.60 * H)
  (h2 : H + F = 240)
  (h3 : phone_bill = 0.10 * F) :
  H + F + phone_bill = 249 :=
by sorry

end luna_total_monthly_budget_l307_307339


namespace cylinder_volume_l307_307405

theorem cylinder_volume (length width : ℝ) (h₁ h₂ : ℝ) (radius1 radius2 : ℝ) (V1 V2 : ℝ) (π : ℝ)
  (h_length : length = 12) (h_width : width = 8) 
  (circumference1 : circumference1 = length)
  (circumference2 : circumference2 = width)
  (h_radius1 : radius1 = 6 / π) (h_radius2 : radius2 = 4 / π)
  (h_height1 : h₁ = width) (h_height2 : h₂ = length)
  (h_V1 : V1 = π * radius1^2 * h₁) (h_V2 : V2 = π * radius2^2 * h₂) :
  V1 = 288 / π ∨ V2 = 192 / π :=
sorry


end cylinder_volume_l307_307405


namespace exists_similar_sizes_P_23_l307_307991

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l307_307991


namespace find_a_b_range_of_a_l307_307443

-- Define the function f(x)
def f (x : ℝ) (a b : ℝ) : ℝ := x - a * Real.log x + b

-- Define the derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / x

-- 1. Prove the values of a and b given tangent line condition
theorem find_a_b (a b : ℝ) (h1 : f' 1 a = 2) (h2 : f 1 a b = 5) : a = -1 ∧ b = 4 :=
by
  sorry

-- 2. Prove the range of values of a given the condition on f'(x)
theorem range_of_a (a : ℝ) (h : ∀ x ∈ Set.Icc 2 3, |f' x a| < 3 / x^2) : 2 ≤ a ∧ a ≤ 7 / 2 :=
by
  sorry

end find_a_b_range_of_a_l307_307443


namespace number_of_female_students_in_sample_l307_307314

theorem number_of_female_students_in_sample (male_students female_students sample_size : ℕ)
  (h1 : male_students = 560)
  (h2 : female_students = 420)
  (h3 : sample_size = 280) :
  (female_students * sample_size) / (male_students + female_students) = 120 := 
sorry

end number_of_female_students_in_sample_l307_307314


namespace remainder_when_112222333_divided_by_37_l307_307388

theorem remainder_when_112222333_divided_by_37 : 112222333 % 37 = 0 :=
by
  sorry

end remainder_when_112222333_divided_by_37_l307_307388


namespace power_mod_7_l307_307230

theorem power_mod_7 {a : ℤ} (h : a = 3) : (a ^ 123) % 7 = 6 := by
  sorry

end power_mod_7_l307_307230


namespace binomial_sum_eq_sum_valid_n_values_l307_307519

theorem binomial_sum_eq (n : ℕ) (h₁ : nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16) :
  n = 14 ∨ n = 16 :=
sorry

theorem sum_valid_n_values :
  let n1 := 16
  let n2 := 14
  n1 + n2 = 30 :=
by
  -- proof to be provided; this is to check if the theorem holds
  sorry

end binomial_sum_eq_sum_valid_n_values_l307_307519


namespace number_of_pupils_l307_307890

theorem number_of_pupils
  (pupil_mark_wrong : ℕ)
  (pupil_mark_correct : ℕ)
  (average_increase : ℚ)
  (n : ℕ)
  (h1 : pupil_mark_wrong = 73)
  (h2 : pupil_mark_correct = 45)
  (h3 : average_increase = 1/2)
  (h4 : 28 / n = average_increase) : n = 56 := 
sorry

end number_of_pupils_l307_307890


namespace expected_value_l307_307315

noncomputable def p : ℝ := 0.25
noncomputable def P_xi_1 : ℝ := 0.24
noncomputable def P_black_bag_b : ℝ := 0.8
noncomputable def P_xi_0 : ℝ := (1 - p) * (1 - P_black_bag_b) * (1 - P_black_bag_b)
noncomputable def P_xi_2 : ℝ := p * (1 - P_black_bag_b) * (1 - P_black_bag_b) + (1 - p) * P_black_bag_b * P_black_bag_b
noncomputable def P_xi_3 : ℝ := p * P_black_bag_b + p * (1 - P_black_bag_b) * P_black_bag_b
noncomputable def E_xi : ℝ := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2 + 3 * P_xi_3

theorem expected_value : E_xi = 1.94 := by
  sorry

end expected_value_l307_307315


namespace expression_value_l307_307870

theorem expression_value (a b c d : ℝ) 
  (intersect1 : 4 = a * (2:ℝ)^2 + b * 2 + 1) 
  (intersect2 : 4 = (2:ℝ)^2 + c * 2 + d) 
  (hc : b + c = 1) : 
  4 * a + d = 1 := 
sorry

end expression_value_l307_307870


namespace three_digit_sum_seven_l307_307600

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l307_307600


namespace smallest_possible_bob_number_l307_307742

theorem smallest_possible_bob_number : 
  let alices_number := 60
  let bobs_smallest_number := 30
  ∃ (bob_number : ℕ), (∀ p : ℕ, Prime p → p ∣ alices_number → p ∣ bob_number) ∧ bob_number = bobs_smallest_number :=
by
  sorry

end smallest_possible_bob_number_l307_307742


namespace rationalization_correct_l307_307032

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l307_307032


namespace part1_sequence_arithmetic_part2_general_formula_l307_307827

-- Given conditions
variables {α : Type*} [division_ring α] [char_zero α]

def S (a : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∑ i in finset.range n, a i else 0
def b (S : ℕ → α) (n : ℕ) : α := if h : n > 0 then ∏ i in finset.range n, S i else 1

-- The main theorem to prove
theorem part1_sequence_arithmetic (a : ℕ → α) (S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : ∃ a₁ d : α, ∀ n,  b n = a₁ + n * d := 
sorry

-- To verify the formula for a_n 
theorem part2_general_formula (a S b : ℕ → α) (hSn : ∀ n, S n = if h : n > 0 then ∑ i in finset.range n, a i else 0) (hb : ∀ n, b n = if h : n > 0 then ∏ i in finset.range n, S i else 1)
  (h : ∀ n, 2 / S n + 1 / b n = 2) : 
∃ (a₁ a₂ : ℕ → α), (a₁ 1 = 3/2 ∧ a₂ = λ n : ℕ, if n ≥ 2 then -1 / (n * (n + 1)) else 0) := 
sorry

end part1_sequence_arithmetic_part2_general_formula_l307_307827


namespace negation_of_exists_l307_307498

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_exists_l307_307498


namespace count_integers_with_block_178_l307_307954

theorem count_integers_with_block_178 (a b : ℕ) : 10000 ≤ a ∧ a < 100000 → 10000 ≤ b ∧ b < 100000 → a = b → b - a = 99999 → ∃ n, n = 280 ∧ (n = a + b) := sorry

end count_integers_with_block_178_l307_307954


namespace proof_problem_l307_307945

noncomputable def f (a x : ℝ) : ℝ := (x^2 + a*x - 2*a - 3) * Real.exp x

theorem proof_problem :
  (∃ a : ℝ, (a = -5) ∧
    let fval := f a in
    (∀ x : ℝ, fval 2 = Real.exp 2) ∧
    (∀ x ∈ Set.Icc (3/2 : ℝ) 3, (max (fval 3) (fval 2) = fval 3 ∧ min (fval 2) (fval 3) = fval 2))) :=
by
  sorry

end proof_problem_l307_307945


namespace children_got_off_bus_l307_307728

theorem children_got_off_bus :
  ∀ (initial_children final_children new_children off_children : ℕ),
    initial_children = 21 → final_children = 16 → new_children = 5 →
    initial_children - off_children + new_children = final_children →
    off_children = 10 :=
by
  intro initial_children final_children new_children off_children
  intros h_init h_final h_new h_eq
  sorry

end children_got_off_bus_l307_307728


namespace residue_calculation_l307_307422

theorem residue_calculation :
  (196 * 18 - 21 * 9 + 5) % 18 = 14 := 
by 
  sorry

end residue_calculation_l307_307422


namespace circles_area_sum_l307_307365

noncomputable def sum_of_areas_of_circles (r s t : ℝ) : ℝ :=
  π * (r^2 + s^2 + t^2)

theorem circles_area_sum (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  sum_of_areas_of_circles r s t = 56 * π :=
begin
  sorry
end

end circles_area_sum_l307_307365


namespace option_a_correct_option_c_correct_option_d_correct_l307_307957

theorem option_a_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (1 / a > 1 / b) :=
sorry

theorem option_c_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (Real.sqrt (-a) > Real.sqrt (-b)) :=
sorry

theorem option_d_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (|a| > -b) :=
sorry

end option_a_correct_option_c_correct_option_d_correct_l307_307957


namespace max_term_of_sequence_l307_307137

def a (n : ℕ) : ℚ := (n : ℚ) / (n^2 + 156)

theorem max_term_of_sequence : ∃ n, (n = 12 ∨ n = 13) ∧ (∀ m, a m ≤ a n) := by 
  sorry

end max_term_of_sequence_l307_307137


namespace total_voters_l307_307157

-- Definitions
def number_of_voters_first_hour (x : ℕ) := x
def percentage_october_22 (x : ℕ) := 35 * x / 100
def percentage_october_29 (x : ℕ) := 65 * x / 100
def additional_voters_october_22 := 80
def final_percentage_october_29 (total_votes : ℕ) := 45 * total_votes / 100

-- Statement
theorem total_voters (x : ℕ) (h1 : percentage_october_22 x + additional_voters_october_22 = 35 * (x + additional_voters_october_22) / 100)
                      (h2 : percentage_october_29 x = 65 * x / 100)
                      (h3 : final_percentage_october_29 (x + additional_voters_october_22) = 45 * (x + additional_voters_october_22) / 100):
  x + additional_voters_october_22 = 260 := 
sorry

end total_voters_l307_307157


namespace minimum_abs_sum_l307_307013

def matrix_squared_condition (p q r s : ℤ) : Prop :=
  (p * p + q * r = 9) ∧ 
  (q * r + s * s = 9) ∧ 
  (p * q + q * s = 0) ∧ 
  (r * p + r * s = 0)

def abs_sum (p q r s : ℤ) : ℤ :=
  |p| + |q| + |r| + |s|

theorem minimum_abs_sum (p q r s : ℤ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0) 
  (h5 : matrix_squared_condition p q r s) : abs_sum p q r s = 8 :=
by 
  sorry

end minimum_abs_sum_l307_307013


namespace Isabella_exchange_l307_307183

/-
Conditions:
1. Isabella exchanged d U.S. dollars to receive (8/5)d Canadian dollars.
2. After spending 80 Canadian dollars, she had d + 20 Canadian dollars left.
3. Sum of the digits of d is 14.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (.+.) 0

theorem Isabella_exchange (d : ℕ) (h : (8 * d / 5) - 80 = d + 20) : sum_of_digits d = 14 :=
by sorry

end Isabella_exchange_l307_307183


namespace value_of_five_l307_307296

variable (f : ℝ → ℝ)

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = f (x)

theorem value_of_five (hf_odd : odd_function f) (hf_periodic : periodic_function f) : f 5 = 0 :=
by 
  sorry

end value_of_five_l307_307296


namespace three_digit_sum_7_l307_307611

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l307_307611


namespace questions_answered_second_half_l307_307524

theorem questions_answered_second_half :
  ∀ (q1 q2 p s : ℕ), q1 = 3 → p = 3 → s = 15 → s = (q1 + q2) * p → q2 = 2 :=
by
  intros q1 q2 p s hq1 hp hs h_final_score
  -- proofs go here, but we skip them
  sorry

end questions_answered_second_half_l307_307524


namespace exists_infinite_subset_with_gcd_l307_307014

/-- A set of natural numbers where each number is a product of at most 1987 primes -/
def is_bounded_product_set (A : Set ℕ) (k : ℕ) : Prop :=
  ∀ a ∈ A, ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ a = S.prod id ∧ S.card ≤ k

/-- Prove the existence of an infinite subset and a common gcd for any pair of its elements -/
theorem exists_infinite_subset_with_gcd (A : Set ℕ) (k : ℕ) (hk : k = 1987)
  (hA : is_bounded_product_set A k) (h_inf : Set.Infinite A) :
  ∃ (B : Set ℕ) (b : ℕ), Set.Subset B A ∧ Set.Infinite B ∧ ∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → Nat.gcd x y = b := 
sorry

end exists_infinite_subset_with_gcd_l307_307014


namespace trig_expression_evaluation_l307_307122

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := 
  sorry

end trig_expression_evaluation_l307_307122


namespace matrix_power_identity_l307_307679

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end matrix_power_identity_l307_307679


namespace fraction_sum_eq_decimal_l307_307077

theorem fraction_sum_eq_decimal : (2 / 5) + (2 / 50) + (2 / 500) = 0.444 := by
  sorry

end fraction_sum_eq_decimal_l307_307077


namespace no_rectangular_prism_equal_measures_l307_307464

theorem no_rectangular_prism_equal_measures (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0): 
  ¬ (4 * (a + b + c) = 2 * (a * b + b * c + c * a) ∧ 2 * (a * b + b * c + c * a) = a * b * c) :=
by
  sorry

end no_rectangular_prism_equal_measures_l307_307464


namespace sufficient_condition_for_product_l307_307128

-- Given conditions
def intersects_parabola_at_two_points (x1 y1 x2 y2 : ℝ) : Prop :=
  y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧ x1 ≠ x2

def line_through_focus (x y : ℝ) (k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 1)

-- The theorem to prove
theorem sufficient_condition_for_product 
  (x1 y1 x2 y2 k : ℝ)
  (h1 : intersects_parabola_at_two_points x1 y1 x2 y2)
  (h2 : line_through_focus x1 y1 k)
  (h3 : line_through_focus x2 y2 k) :
  x1 * x2 = 1 :=
sorry

end sufficient_condition_for_product_l307_307128


namespace locus_of_points_where_tangents_are_adjoint_lines_l307_307445

theorem locus_of_points_where_tangents_are_adjoint_lines 
  (p : ℝ) (y x : ℝ)
  (h_parabola : y^2 = 2 * p * x) :
  y^2 = - (p / 2) * x :=
sorry

end locus_of_points_where_tangents_are_adjoint_lines_l307_307445


namespace final_bill_is_correct_l307_307817

def Alicia_order := [7.50, 4.00, 5.00]
def Brant_order := [10.00, 4.50, 6.00]
def Josh_order := [8.50, 4.00, 3.50]
def Yvette_order := [9.00, 4.50, 6.00]

def discount_rate := 0.10
def sales_tax_rate := 0.08
def tip_rate := 0.20

noncomputable def calculate_final_bill : Float :=
  let subtotal := (Alicia_order.sum + Brant_order.sum + Josh_order.sum + Yvette_order.sum)
  let discount := discount_rate * subtotal
  let discounted_total := subtotal - discount
  let sales_tax := sales_tax_rate * discounted_total
  let pre_tax_and_discount_total := subtotal
  let tip := tip_rate * pre_tax_and_discount_total
  discounted_total + sales_tax + tip

theorem final_bill_is_correct : calculate_final_bill = 84.97 := by
  sorry

end final_bill_is_correct_l307_307817


namespace log_base_9_of_729_l307_307557

theorem log_base_9_of_729 : ∃ (x : ℝ), (9 : ℝ)^x = (729 : ℝ) ∧ x = 3 := 
by {
  have h1 : (9 : ℝ) = (3 : ℝ)^2 := by norm_num,
  have h2 : (729 : ℝ) = (3 : ℝ)^6 := by norm_num,
  use 3,
  split,
  {
    calc (9 : ℝ) ^ 3
        = (3^2 : ℝ) ^ 3 : by rw h1
    ... = (3^6 : ℝ) : by rw pow_mul
    ... = (729 : ℝ) : by rw h2,
  },
  { 
    refl,
  }
}

end log_base_9_of_729_l307_307557


namespace apples_on_tree_l307_307997

-- Defining initial number of apples on the tree
def initial_apples : ℕ := 4

-- Defining apples picked from the tree
def apples_picked : ℕ := 2

-- Defining new apples grown on the tree
def new_apples : ℕ := 3

-- Prove the final number of apples on the tree is 5
theorem apples_on_tree : initial_apples - apples_picked + new_apples = 5 :=
by
  -- This is where the proof would go
  sorry

end apples_on_tree_l307_307997


namespace sum_radical_conjugates_l307_307550

theorem sum_radical_conjugates : (5 - Real.sqrt 500) + (5 + Real.sqrt 500) = 10 :=
by
  sorry

end sum_radical_conjugates_l307_307550


namespace solve_problem_l307_307438

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 2 x else 3^x

theorem solve_problem : f (f (1 / 2)) = 1 / 3 := by
  sorry

end solve_problem_l307_307438


namespace supplement_angle_greater_complement_angle_equal_unique_perpendicular_shortest_perpendicular_distance_incorrect_statement_l307_307523

theorem supplement_angle_greater (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 180) : ¬(180 - θ > θ) → (θ ≥ 90) :=
begin
  intro h,
  sorry,
end

theorem complement_angle_equal (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ 90) : (90 - θ = 90 - θ) :=
begin
  refl,
end

theorem unique_perpendicular {P : Type} [EuclideanSpace ℝ P]
  (p : P) (l : line P) (hp : p ∉ l) : ∃! m : line P, p ∈ m ∧ m ⊥ l :=
begin
  sorry,
end

theorem shortest_perpendicular_distance {P : Type} [EuclideanSpace ℝ P]
  (p : P) (l : line P) (hp : p ∉ l) : ∀ q ∈ l, dist p l = dist p (orth_proj l p) :=
begin
  intros q hq,
  sorry,
end

theorem incorrect_statement (θ : ℝ) : (¬ θ ≥ 90) :=
begin
  sorry,
end

end supplement_angle_greater_complement_angle_equal_unique_perpendicular_shortest_perpendicular_distance_incorrect_statement_l307_307523


namespace max_constant_C_all_real_numbers_l307_307933

theorem max_constant_C_all_real_numbers:
  ∀ (x1 x2 x3 x4 x5 x6 : ℝ), 
  (x1 + x2 + x3 + x4 + x5 + x6)^2 ≥ 
  3 * (x1 * (x2 + x3) + x2 * (x3 + x4) + x3 * (x4 + x5) + x4 * (x5 + x6) + x5 * (x6 + x1) + x6 * (x1 + x2)) := 
by 
  sorry

end max_constant_C_all_real_numbers_l307_307933


namespace bn_arithmetic_sequence_an_formula_l307_307833

variable (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ)

axiom h1 : ∀ n, 2 / S n + 1 / b n = 2
axiom S_def : ∀ n, S n = (∑ i in Finset.range n, a (i+1))
axiom b_def : ∀ n, b n = ∏ i in Finset.range n, S (i+1)

theorem bn_arithmetic_sequence : 
  ∀ n, n ≥ 2 → (b n - b (n - 1) = 1 / 2) → ∃ c, ∀ n, b n = b 1 + (n - 1) * c := sorry

theorem an_formula :
  (a 1 = 3 / 2) ∧ (∀ n, n ≥ 2 → a n = -1 / (n * (n + 1))) := sorry

end bn_arithmetic_sequence_an_formula_l307_307833


namespace f_neg_l307_307683

/-- Define f(x) as an odd function --/
def f : ℝ → ℝ := sorry

/-- The property of odd functions: f(-x) = -f(x) --/
axiom odd_fn_property (x : ℝ) : f (-x) = -f x

/-- Define the function for non-negative x --/
axiom f_nonneg (x : ℝ) (hx : 0 ≤ x) : f x = x + 1

/-- The goal is to determine f(x) when x < 0 --/
theorem f_neg (x : ℝ) (h : x < 0) : f x = x - 1 :=
by
  sorry

end f_neg_l307_307683


namespace boys_without_calculators_l307_307020

theorem boys_without_calculators 
  (total_students : ℕ)
  (total_boys : ℕ)
  (students_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (H_total_students : total_students = 30)
  (H_total_boys : total_boys = 20)
  (H_students_with_calculators : students_with_calculators = 25)
  (H_girls_with_calculators : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 13 :=
by
  sorry

end boys_without_calculators_l307_307020


namespace shortest_tree_height_l307_307051

theorem shortest_tree_height :
  (tallest_tree_height = 150) →
  (middle_tree_height = (2 / 3) * tallest_tree_height) →
  (shortest_tree_height = (1 / 2) * middle_tree_height) →
  shortest_tree_height = 50 :=
by
  intros h1 h2 h3
  sorry

end shortest_tree_height_l307_307051


namespace total_number_of_workers_l307_307698

theorem total_number_of_workers 
  (W : ℕ) 
  (h_all_avg : W * 8000 = 10 * 12000 + (W - 10) * 6000) : 
  W = 30 := 
by
  sorry

end total_number_of_workers_l307_307698


namespace correct_options_l307_307142

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l307_307142


namespace solve_quadratic_problem_l307_307486

theorem solve_quadratic_problem :
  ∀ x : ℝ, (x^2 + 6 * x + 8 = -(x + 4) * (x + 7)) ↔ (x = -4 ∨ x = -4.5) := by
  sorry

end solve_quadratic_problem_l307_307486


namespace sheela_total_income_l307_307855

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end sheela_total_income_l307_307855


namespace tens_digit_23_pow_1987_l307_307274

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l307_307274


namespace simplify_and_evaluate_fraction_l307_307352

theorem simplify_and_evaluate_fraction (x : ℤ) (hx : x = 5) :
  ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 4 :=
by
  rw [hx]
  sorry

end simplify_and_evaluate_fraction_l307_307352


namespace total_time_to_school_and_back_l307_307729

-- Definition of the conditions
def speed_to_school : ℝ := 3 -- in km/hr
def speed_back_home : ℝ := 2 -- in km/hr
def distance : ℝ := 6 -- in km

-- Proof statement
theorem total_time_to_school_and_back : 
  (distance / speed_to_school) + (distance / speed_back_home) = 5 := 
by
  sorry

end total_time_to_school_and_back_l307_307729


namespace number_of_three_digit_numbers_with_sum_7_l307_307570

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l307_307570


namespace mason_grandmother_age_l307_307840

theorem mason_grandmother_age (mason_age: ℕ) (sydney_age: ℕ) (father_age: ℕ) (grandmother_age: ℕ)
  (h1: mason_age = 20)
  (h2: mason_age * 3 = sydney_age)
  (h3: sydney_age + 6 = father_age)
  (h4: father_age * 2 = grandmother_age) : 
  grandmother_age = 132 :=
by
  sorry

end mason_grandmother_age_l307_307840


namespace three_digit_sum_7_l307_307614

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l307_307614


namespace fraction_of_sum_l307_307244

theorem fraction_of_sum (n S : ℕ) 
  (h1 : S = (n-1) * ((n:ℚ) / 3))
  (h2 : n > 0) : 
  (n:ℚ) / (S + n) = 3 / (n + 2) := 
by 
  sorry

end fraction_of_sum_l307_307244


namespace vacant_student_seats_given_to_parents_l307_307049

-- Definitions of the conditions
def total_seats : Nat := 150

def awardees_seats : Nat := 15
def admins_teachers_seats : Nat := 45
def students_seats : Nat := 60
def parents_seats : Nat := 30

def awardees_occupied_seats : Nat := 15
def admins_teachers_occupied_seats : Nat := 9 * admins_teachers_seats / 10
def students_occupied_seats : Nat := 4 * students_seats / 5
def parents_occupied_seats : Nat := 7 * parents_seats / 10

-- Vacant seats calculation
def awardees_vacant_seats : Nat := awardees_seats - awardees_occupied_seats
def admins_teachers_vacant_seats : Nat := admins_teachers_seats - admins_teachers_occupied_seats
def students_vacant_seats : Nat := students_seats - students_occupied_seats
def parents_vacant_seats : Nat := parents_seats - parents_occupied_seats

-- Theorem statement
theorem vacant_student_seats_given_to_parents :
  students_vacant_seats = 12 →
  parents_vacant_seats = 9 →
  9 ≤ students_vacant_seats ∧ 9 ≤ parents_vacant_seats :=
by
  sorry

end vacant_student_seats_given_to_parents_l307_307049


namespace intersection_of_P_and_Q_l307_307783

def P : Set ℝ := {x | 1 ≤ x}
def Q : Set ℝ := {x | x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l307_307783


namespace hexagon_perimeter_proof_l307_307462

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_proof :
  ∀ (hexagon : { h : ℕ → ℝ × ℝ // ∀ i, i < 6 → dist (h i) (h ((i+1) % 6)) = dist (h i) (h 0) ∧
                                      ∠ (h i, h ((i+2) % 6), h 0) = 30 })
    (area : ℝ),
  (area = 6 * real.sqrt 3) →
  (∃ s : ℝ, area = 0.5 * s * s * real.sin (real.pi / 6) * 3 + 0.5 * (s^2 - (0.5 * s^2 - 0.5 * s^2 * real.sqrt 3 / 2) / 2) * real.sqrt 3) →
  hexagon_perimeter (real.sqrt 12) = 12 * real.sqrt 3 :=
begin
  sorry
end

end hexagon_perimeter_proof_l307_307462


namespace digits_sum_eq_seven_l307_307625

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l307_307625


namespace log_base_9_of_729_l307_307558

-- Define the conditions
def nine_eq := 9 = 3^2
def seven_two_nine_eq := 729 = 3^6

-- State the goal to be proved
theorem log_base_9_of_729 (h1 : nine_eq) (h2 : seven_two_nine_eq) : log 9 729 = 3 :=
by
  sorry

end log_base_9_of_729_l307_307558


namespace symmetry_condition_l307_307456

-- Define grid and initial conditions
def grid : Type := ℕ × ℕ
def is_colored (pos : grid) : Prop := 
  pos = (1,4) ∨ pos = (2,1) ∨ pos = (4,2)

-- Conditions for symmetry: horizontal and vertical line symmetry and 180-degree rotational symmetry
def is_symmetric_line (grid_size : grid) (pos : grid) : Prop :=
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨ 
  (pos.1 > grid_size.1 / 2 ∧ pos.2 <= grid_size.2 / 2) ∨
  (pos.1 <= grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2) ∨
  (pos.1 > grid_size.1 / 2 ∧ pos.2 > grid_size.2 / 2)

def grid_size : grid := (4, 5)
def add_squares_needed (num : ℕ) : Prop :=
  ∀ (pos : grid), is_symmetric_line grid_size pos → is_colored pos

theorem symmetry_condition : 
  ∃ n, add_squares_needed n ∧ n = 9
  := sorry

end symmetry_condition_l307_307456


namespace number_of_correct_answers_l307_307672

-- We define variables C (number of correct answers) and W (number of wrong answers).
variables (C W : ℕ)

-- Define the conditions given in the problem.
def conditions :=
  C + W = 75 ∧ 4 * C - W = 125

-- Define the theorem which states that the number of correct answers is 40.
theorem number_of_correct_answers
  (h : conditions C W) :
  C = 40 :=
sorry

end number_of_correct_answers_l307_307672


namespace number_of_three_digit_numbers_with_digit_sum_seven_l307_307631

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l307_307631


namespace time_to_cross_is_30_seconds_l307_307407

variable (length_train : ℕ) (speed_km_per_hr : ℕ) (length_bridge : ℕ)

def total_distance := length_train + length_bridge

def speed_m_per_s := (speed_km_per_hr * 1000 : ℕ) / 3600

def time_to_cross_bridge := total_distance length_train length_bridge / speed_m_per_s speed_km_per_hr

theorem time_to_cross_is_30_seconds 
  (h_train_length : length_train = 140)
  (h_train_speed : speed_km_per_hr = 45)
  (h_bridge_length : length_bridge = 235) :
  time_to_cross_bridge length_train speed_km_per_hr length_bridge = 30 :=
by
  sorry

end time_to_cross_is_30_seconds_l307_307407


namespace john_spending_l307_307322

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end john_spending_l307_307322


namespace find_number_of_pourings_l307_307904

-- Define the sequence of remaining water after each pouring
def remaining_water (n : ℕ) : ℚ :=
  (2 : ℚ) / (n + 2)

-- The main theorem statement
theorem find_number_of_pourings :
  ∃ n : ℕ, remaining_water n = 1 / 8 :=
by
  sorry

end find_number_of_pourings_l307_307904


namespace range_of_b_l307_307132

-- Definitions
def polynomial_inequality (b : ℝ) (x : ℝ) : Prop := x^2 + b * x - b - 3/4 > 0

-- The main statement
theorem range_of_b (b : ℝ) : (∀ x : ℝ, polynomial_inequality b x) ↔ -3 < b ∧ b < -1 :=
by {
    sorry -- proof goes here
}

end range_of_b_l307_307132


namespace half_abs_diff_of_squares_l307_307225

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 21) (h2 : b = 19) :
  (abs (a^2 - b^2)) / 2 = 40 := by
  sorry

end half_abs_diff_of_squares_l307_307225


namespace polygon_sides_l307_307249

theorem polygon_sides {S n : ℕ} (h : S = 2160) (hs : S = 180 * (n - 2)) : n = 14 := 
by
  sorry

end polygon_sides_l307_307249


namespace find_number_l307_307307

def number_condition (N : ℝ) : Prop := 
  0.20 * 0.15 * 0.40 * 0.30 * 0.50 * N = 180

theorem find_number (N : ℝ) (h : number_condition N) : N = 1000000 :=
sorry

end find_number_l307_307307


namespace area_triangle_COD_l307_307749

noncomputable def area_of_triangle (t s : ℝ) : ℝ := 
  1 / 2 * abs (5 + 2 * s + 7 * t)

theorem area_triangle_COD (t s : ℝ) : 
  ∃ (C : ℝ × ℝ) (D : ℝ × ℝ), 
    C = (3 + 5 * t, 2 + 4 * t) ∧ 
    D = (2 + 5 * s, 3 + 4 * s) ∧ 
    area_of_triangle t s = 1 / 2 * abs (5 + 2 * s + 7 * t) :=
by
  sorry

end area_triangle_COD_l307_307749


namespace janet_hiking_distance_l307_307465

theorem janet_hiking_distance
  (A B C : EuclideanSpace ℝ)
  (hAB : dist A B = 3)
  (hAC : dist B C = 8)
  (angle_ABC : angle A B C = π / 6) : 
  dist A C = Real.sqrt 57 :=
by
  sorry

end janet_hiking_distance_l307_307465


namespace ducks_problem_l307_307543

theorem ducks_problem :
  ∃ (adelaide ephraim kolton : ℕ),
    adelaide = 30 ∧
    adelaide = 2 * ephraim ∧
    ephraim + 45 = kolton ∧
    (adelaide + ephraim + kolton) % 9 = 0 ∧
    1 ≤ adelaide ∧
    1 ≤ ephraim ∧
    1 ≤ kolton ∧
    adelaide + ephraim + kolton = 108 ∧
    (adelaide + ephraim + kolton) / 3 = 36 :=
by
  sorry

end ducks_problem_l307_307543


namespace jillian_oranges_l307_307317

theorem jillian_oranges:
  let oranges := 80 in
  let pieces_per_orange := 10 in
  let pieces_per_friend := 4 in
  (oranges * (pieces_per_orange / pieces_per_friend) = 200) :=
by sorry

end jillian_oranges_l307_307317


namespace white_pairs_coincide_l307_307751

def num_red : Nat := 4
def num_blue : Nat := 4
def num_green : Nat := 2
def num_white : Nat := 6
def red_pairs : Nat := 3
def blue_pairs : Nat := 2
def green_pairs : Nat := 1 
def red_white_pairs : Nat := 2
def green_blue_pairs : Nat := 1

theorem white_pairs_coincide :
  (num_red = 4) ∧ 
  (num_blue = 4) ∧ 
  (num_green = 2) ∧ 
  (num_white = 6) ∧ 
  (red_pairs = 3) ∧ 
  (blue_pairs = 2) ∧ 
  (green_pairs = 1) ∧ 
  (red_white_pairs = 2) ∧ 
  (green_blue_pairs = 1) → 
  4 = 4 :=
by
  sorry

end white_pairs_coincide_l307_307751


namespace hyperbola_eccentricity_l307_307005

theorem hyperbola_eccentricity 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (focus_on_x_axis : ∃ c : ℝ, c > 0)
  (asymptote_eq : ∀ x y : ℝ, (4 + 3 * y = 0) ∨ (4 - 3 * y = 0)) :
  ∃ e : ℝ, e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l307_307005


namespace fraction_of_airing_time_spent_on_commercials_l307_307081

theorem fraction_of_airing_time_spent_on_commercials 
  (num_programs : ℕ) (minutes_per_program : ℕ) (total_commercial_time : ℕ) 
  (h1 : num_programs = 6) (h2 : minutes_per_program = 30) (h3 : total_commercial_time = 45) : 
  (total_commercial_time : ℚ) / (num_programs * minutes_per_program : ℚ) = 1 / 4 :=
by {
  -- The proof is omitted here as only the statement is required according to the instruction.
  sorry
}

end fraction_of_airing_time_spent_on_commercials_l307_307081


namespace rainfall_on_tuesday_l307_307265

theorem rainfall_on_tuesday 
  (r_Mon r_Wed r_Total r_Tue : ℝ)
  (h_Mon : r_Mon = 0.16666666666666666)
  (h_Wed : r_Wed = 0.08333333333333333)
  (h_Total : r_Total = 0.6666666666666666)
  (h_Tue : r_Tue = r_Total - (r_Mon + r_Wed)) :
  r_Tue = 0.41666666666666663 := 
sorry

end rainfall_on_tuesday_l307_307265


namespace a_2_value_l307_307959

theorem a_2_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) (x : ℝ) :
  x^3 + x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 +
  a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^10 → 
  a2 = 42 :=
by
  sorry

end a_2_value_l307_307959


namespace percentage_increase_mario_salary_is_zero_l307_307687

variable (M : ℝ) -- Mario's salary last year
variable (P : ℝ) -- Percentage increase in Mario's salary

-- Condition 1: Mario's salary increased to $4000 this year
def mario_salary_increase (M P : ℝ) : Prop :=
  M * (1 + P / 100) = 4000 

-- Condition 2: Bob's salary last year was 3 times Mario's salary this year
def bob_salary_last_year (M : ℝ) : Prop :=
  3 * 4000 = 12000 

-- Condition 3: Bob's current salary is 20% more than his salary last year
def bob_current_salary : Prop :=
  12000 * 1.2 = 14400

-- Theorem : The percentage increase in Mario's salary is 0%
theorem percentage_increase_mario_salary_is_zero
  (h1 : mario_salary_increase M P)
  (h2 : bob_salary_last_year M)
  (h3 : bob_current_salary) : 
  P = 0 := 
sorry

end percentage_increase_mario_salary_is_zero_l307_307687


namespace solution_set_of_inequality_l307_307242

theorem solution_set_of_inequality (x : ℝ) : |2 * x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 :=
by 
  sorry

end solution_set_of_inequality_l307_307242


namespace expand_expression_l307_307113

variable (x : ℝ)

theorem expand_expression : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 :=
by
  sorry

end expand_expression_l307_307113


namespace calc_expression_l307_307925

theorem calc_expression : 
  abs (Real.sqrt 3 - 2) + (8:ℝ)^(1/3) - Real.sqrt 16 + (-1)^(2023:ℝ) = -(Real.sqrt 3) - 1 :=
by
  sorry

end calc_expression_l307_307925


namespace frances_towels_weight_in_ounces_l307_307341

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end frances_towels_weight_in_ounces_l307_307341


namespace plan_y_cheaper_than_plan_x_l307_307091

def cost_plan_x (z : ℕ) : ℕ := 15 * z

def cost_plan_y (z : ℕ) : ℕ :=
  if z > 500 then 3000 + 7 * z - 1000 else 3000 + 7 * z

theorem plan_y_cheaper_than_plan_x (z : ℕ) (h : z > 500) : cost_plan_y z < cost_plan_x z :=
by
  sorry

end plan_y_cheaper_than_plan_x_l307_307091


namespace geometric_sequence_properties_l307_307813

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ) :
  a 1 = 1 / 2 ∧ a 4 = -4 → q = -2 ∧ (∀ n, a n = 1 / 2 * q ^ (n - 1)) :=
by
  intro h
  sorry

end geometric_sequence_properties_l307_307813


namespace rationalize_denominator_correct_l307_307025

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l307_307025


namespace prisoners_freedom_guaranteed_l307_307400

-- Definition of the problem strategy
def prisoners_strategy (n : ℕ) : Prop :=
  ∃ counter regular : ℕ → ℕ,
    (∀ i, i < n - 1 → regular i < 2) ∧ -- Each regular prisoner turns on the light only once
    (∃ count : ℕ, 
      counter count = 99 ∧  -- The counter counts to 99 based on the strategy
      (∀ k, k < 99 → (counter (k + 1) = counter k + 1))) -- Each turn off increases the count by one

-- The main proof statement that there is a strategy ensuring the prisoners' release
theorem prisoners_freedom_guaranteed : ∀ (n : ℕ), n = 100 →
  prisoners_strategy n :=
by {
  sorry -- The actual proof is omitted
}

end prisoners_freedom_guaranteed_l307_307400


namespace fg_of_3_l307_307145

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 + 1
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem we want to prove
theorem fg_of_3 : f (g 3) = 344 := by
  sorry

end fg_of_3_l307_307145


namespace function_behaviour_l307_307691

noncomputable def dec_function (x : ℝ) : ℝ := 3 / x

theorem function_behaviour :
  ∀ x > 0, ∀ y > 0, (dec_function x = y) → (x₂ > x₁) → (dec_function x₂ < dec_function x₁) :=
by
  intros x y hxy hinc
  rw dec_function at hxy
  sorry

end function_behaviour_l307_307691


namespace parametric_to_standard_line_parametric_to_standard_ellipse_l307_307056

theorem parametric_to_standard_line (t : ℝ) (x y : ℝ) 
  (h₁ : x = 1 - 3 * t)
  (h₂ : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := by
sorry

theorem parametric_to_standard_ellipse (θ x y : ℝ) 
  (h₁ : x = 5 * Real.cos θ)
  (h₂ : y = 4 * Real.sin θ) :
  (x^2 / 25) + (y^2 / 16) = 1 := by
sorry

end parametric_to_standard_line_parametric_to_standard_ellipse_l307_307056


namespace square_area_l307_307744

theorem square_area :
  ∀ (x : ℝ), (4 * x - 15 = 20 - 3 * x) → (let edge := 4 * x - 15 in edge ^ 2 = 25) :=
by
  intros x h
  have h1 : 4 * x - 15 = 20 - 3 * x := h
  sorry

end square_area_l307_307744


namespace max_value_of_a_l307_307662

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - a * x + 1

theorem max_value_of_a :
  ∃ (a : ℝ), (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) → |f a x| ≤ 1) ∧ a = 8 := by
  sorry

end max_value_of_a_l307_307662


namespace men_women_arrangement_l307_307078

theorem men_women_arrangement :
  let men := 2
  let women := 4
  let slots := 5
  (Nat.choose slots women) * women.factorial * men.factorial = 240 :=
by
  sorry

end men_women_arrangement_l307_307078


namespace number_of_three_digit_numbers_with_sum_seven_l307_307574

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l307_307574


namespace rationalize_denominator_correct_l307_307026

noncomputable def rationalize_denominator : Prop :=
  1 / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l307_307026


namespace selling_price_correct_l307_307024

-- Define the conditions
def cost_per_cupcake : ℝ := 0.75
def total_cupcakes_burnt : ℕ := 24
def total_eaten_first : ℕ := 5
def total_eaten_later : ℕ := 4
def net_profit : ℝ := 24
def total_cupcakes_made : ℕ := 72
def total_cost : ℝ := total_cupcakes_made * cost_per_cupcake
def total_eaten : ℕ := total_eaten_first + total_eaten_later
def total_sold : ℕ := total_cupcakes_made - total_eaten
def revenue (P : ℝ) : ℝ := total_sold * P

-- Prove the correctness of the selling price P
theorem selling_price_correct : 
  ∃ P : ℝ, revenue P - total_cost = net_profit ∧ (P = 1.24) :=
by
  sorry

end selling_price_correct_l307_307024


namespace num_three_digit_sums7_l307_307604

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l307_307604


namespace trigonometric_identity_l307_307442

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3 / 2 :=
by
  sorry

end trigonometric_identity_l307_307442


namespace average_age_l307_307752

theorem average_age (Devin_age Eden_age mom_age : ℕ)
  (h1 : Devin_age = 12)
  (h2 : Eden_age = 2 * Devin_age)
  (h3 : mom_age = 2 * Eden_age) :
  (Devin_age + Eden_age + mom_age) / 3 = 28 := by
  sorry

end average_age_l307_307752


namespace avg_speed_of_car_l307_307238

noncomputable def average_speed (distance1 distance2 : ℕ) (time1 time2 : ℕ) : ℕ :=
  (distance1 + distance2) / (time1 + time2)

theorem avg_speed_of_car :
  average_speed 65 45 1 1 = 55 := by
  sorry

end avg_speed_of_car_l307_307238


namespace number_of_divisors_greater_than_22_l307_307208

theorem number_of_divisors_greater_than_22 :
  (∃ (S : Finset ℕ), S = (Finset.filter (λ n, 1998 % n = 0 ∧ n > 22) (Finset.range (1998 + 1))) ∧ S.card = 10) :=
sorry

end number_of_divisors_greater_than_22_l307_307208


namespace arithmetic_sequence_a1_a7_a3_a5_l307_307002

noncomputable def arithmetic_sequence_property (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_a1_a7_a3_a5 (a : ℕ → ℝ) (h_arith : arithmetic_sequence_property a)
  (h_cond : a 1 + a 7 = 10) : a 3 + a 5 = 10 :=
by
  sorry

end arithmetic_sequence_a1_a7_a3_a5_l307_307002


namespace evaluate_expression_l307_307283

variable (b x : ℝ)

theorem evaluate_expression (h : x = b + 9) : x - b + 4 = 13 := by
  sorry

end evaluate_expression_l307_307283


namespace count_good_numbers_l307_307216

-- Define the property of a good number
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- Count the number of good numbers
theorem count_good_numbers : (finset.filter is_good_number (finset.range 2021)).card = 10 :=
by sorry

end count_good_numbers_l307_307216


namespace twelve_pharmacies_not_enough_l307_307534

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l307_307534


namespace find_number_of_folders_l307_307319

theorem find_number_of_folders :
  let price_pen := 1
  let price_notebook := 3
  let price_folder := 5
  let pens_bought := 3
  let notebooks_bought := 4
  let bill := 50
  let change := 25
  let total_cost_pens_notebooks := pens_bought * price_pen + notebooks_bought * price_notebook
  let amount_spent := bill - change
  let amount_spent_on_folders := amount_spent - total_cost_pens_notebooks
  let number_of_folders := amount_spent_on_folders / price_folder
  number_of_folders = 2 :=
by
  sorry

end find_number_of_folders_l307_307319


namespace perpendicular_vector_l307_307446

-- Vectors a and b are given
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 3)

-- Defining the vector addition and scalar multiplication for our context
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (m : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (m * v.1, m * v.2)

-- The vector a + m * b
def a_plus_m_b (m : ℝ) : ℝ × ℝ := vector_add a (scalar_mul m b)

-- The dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The statement that a is perpendicular to (a + m * b) when m = 5
theorem perpendicular_vector : dot_product a (a_plus_m_b 5) = 0 :=
sorry

end perpendicular_vector_l307_307446


namespace parabola_inequality_l307_307653

theorem parabola_inequality {y1 y2 : ℝ} :
  (∀ x1 x2 : ℝ, x1 = -5 → x2 = 2 →
  y1 = x1^2 + 2 * x1 + 3 ∧ y2 = x2^2 + 2 * x2 + 3) → (y1 > y2) :=
by
  intros h
  sorry

end parabola_inequality_l307_307653


namespace solution_set_inequality_l307_307176

theorem solution_set_inequality (x : ℝ) : |5 - x| < |x - 2| + |7 - 2 * x| ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3.5 :=
by
  sorry

end solution_set_inequality_l307_307176


namespace school_total_students_l307_307503

theorem school_total_students (T G : ℕ) (h1 : 80 + G = T) (h2 : G = (80 * T) / 100) : T = 400 :=
by
  sorry

end school_total_students_l307_307503


namespace tom_distance_before_karen_wins_l307_307325

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end tom_distance_before_karen_wins_l307_307325


namespace KodyAgeIs32_l307_307640

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end KodyAgeIs32_l307_307640


namespace three_digit_numbers_sum_seven_l307_307591

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l307_307591


namespace total_campers_correct_l307_307536

-- Definitions for the conditions
def campers_morning : ℕ := 15
def campers_afternoon : ℕ := 17

-- Define total campers, question is to prove it is indeed 32
def total_campers : ℕ := campers_morning + campers_afternoon

theorem total_campers_correct : total_campers = 32 :=
by
  -- Proof omitted
  sorry

end total_campers_correct_l307_307536


namespace grid_divisibility_l307_307766

theorem grid_divisibility (n : ℕ) (h_div : n % 7 = 0) (h_gt : n > 7) :
  ∃ m : ℕ, n * n = 7 * m ∧ (∃ arrangement : bool, arrangement = true) := sorry

end grid_divisibility_l307_307766


namespace find_four_numbers_l307_307050

theorem find_four_numbers (a b c d : ℕ) : 
  a + b + c + d = 45 ∧ (∃ k : ℕ, a + 2 = k ∧ b - 2 = k ∧ 2 * c = k ∧ d / 2 = k) → (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
by
  sorry

end find_four_numbers_l307_307050


namespace arithmetic_sequence_difference_l307_307415

theorem arithmetic_sequence_difference :
  ∀ (a d : ℝ),
  a ≥ 15 ∧ a ≤ 120 ∧ 15 ≤ a + 299 * d ∧ a + 299 * d ≤ 120 ∧ 
  300 * (a + 149 * d) = 12000 →
  (G - L = 31320 / 299) :=
by
  intros a d h,
  let L := a + 148 * (-105 / 299),
  let G := a + 148 * (105 / 299),
  have h_1 : G = 120 + 15660 / 299,
  have h_2 : L = 15 - 15660 / 299,
  sorry

end arithmetic_sequence_difference_l307_307415


namespace ship_cargo_weight_l307_307090

theorem ship_cargo_weight (initial_cargo_tons additional_cargo_tons : ℝ) (unloaded_cargo_pounds : ℝ)
    (ton_to_kg pound_to_kg : ℝ) :
    initial_cargo_tons = 5973.42 →
    additional_cargo_tons = 8723.18 →
    unloaded_cargo_pounds = 2256719.55 →
    ton_to_kg = 907.18474 →
    pound_to_kg = 0.45359237 →
    (initial_cargo_tons * ton_to_kg + additional_cargo_tons * ton_to_kg - unloaded_cargo_pounds * pound_to_kg = 12302024.7688159) :=
by
  intros
  sorry

end ship_cargo_weight_l307_307090


namespace three_digit_numbers_sum_seven_l307_307638

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l307_307638


namespace greatest_divisible_by_13_l307_307250

def is_distinct_nonzero_digits (A B C : ℕ) : Prop :=
  0 < A ∧ A < 10 ∧ 0 < B ∧ B < 10 ∧ 0 < C ∧ C < 10 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def number (A B C : ℕ) : ℕ :=
  10000 * A + 1000 * B + 100 * C + 10 * B + A

theorem greatest_divisible_by_13 :
  ∃ (A B C : ℕ), is_distinct_nonzero_digits A B C ∧ number A B C % 13 = 0 ∧ number A B C = 96769 :=
sorry

end greatest_divisible_by_13_l307_307250


namespace value_of_expression_l307_307887

theorem value_of_expression (x : ℤ) (h : x = 5) : x^5 - 10 * x = 3075 := by
  sorry

end value_of_expression_l307_307887


namespace grandmother_ratio_l307_307786

noncomputable def Grace_Age := 60
noncomputable def Mother_Age := 80

theorem grandmother_ratio :
  ∃ GM, Grace_Age = (3 / 8 : Rat) * GM ∧ GM / Mother_Age = 2 :=
by
  sorry

end grandmother_ratio_l307_307786


namespace sqrt_meaningful_condition_l307_307976

theorem sqrt_meaningful_condition (x : ℝ) : (2 * x + 6 >= 0) ↔ (x >= -3) := by
  sorry

end sqrt_meaningful_condition_l307_307976


namespace vec_eqn_solution_l307_307785

theorem vec_eqn_solution :
  ∀ m : ℝ, let a : ℝ × ℝ := (1, -2) 
           let b : ℝ × ℝ := (m, 4) 
           (a.1 * b.2 = a.2 * b.1) → 2 • a - b = (4, -8) :=
by
  intro m a b h_parallel
  sorry

end vec_eqn_solution_l307_307785


namespace solution_positive_then_opposite_signs_l307_307310

theorem solution_positive_then_opposite_signs
  (a b : ℝ) (h : a ≠ 0) (x : ℝ) (hx : ax + b = 0) (x_pos : x > 0) :
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) :=
by
  sorry

end solution_positive_then_opposite_signs_l307_307310


namespace value_of_a_is_2_l307_307654

def point_symmetric_x_axis (a b : ℝ) : Prop :=
  (2 * a + b = 1 - 2 * b) ∧ (a - 2 * b = -(-2 * a - b - 1))

theorem value_of_a_is_2 (a b : ℝ) (h : point_symmetric_x_axis a b) : a = 2 :=
by sorry

end value_of_a_is_2_l307_307654


namespace percentage_error_in_side_l307_307417

theorem percentage_error_in_side
  (s s' : ℝ) -- the actual and measured side lengths
  (h : (s' * s' - s * s) / (s * s) * 100 = 41.61) : 
  ((s' - s) / s) * 100 = 19 :=
sorry

end percentage_error_in_side_l307_307417


namespace find_numbers_l307_307475

theorem find_numbers (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (sums : multiset ℕ) 
  (h_sums : sums = {21, 26, 35, 40, 49, 51, 54, 60, 65, 79}) :
  a = 6 ∧ b = 15 ∧ c = 20 ∧ d = 34 ∧ e = 45 :=
begin
  sorry
end

end find_numbers_l307_307475


namespace three_digit_sum_seven_l307_307599

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l307_307599


namespace second_wrongly_copied_number_l307_307356

theorem second_wrongly_copied_number 
  (avg_err : ℝ) 
  (total_nums : ℕ) 
  (sum_err : ℝ) 
  (first_err_corr : ℝ) 
  (correct_avg : ℝ) 
  (correct_num : ℝ) 
  (second_num_wrong : ℝ) :
  (avg_err = 40.2) → 
  (total_nums = 10) → 
  (sum_err = total_nums * avg_err) → 
  (first_err_corr = 16) → 
  (correct_avg = 40) → 
  (correct_num = 31) → 
  sum_err - first_err_corr + (correct_num - second_num_wrong) = total_nums * correct_avg → 
  second_num_wrong = 17 := 
by 
  intros h_avg h_total h_sum_err h_first_corr h_correct_avg h_correct_num h_corrected_sum 
  sorry

end second_wrongly_copied_number_l307_307356


namespace b_arithmetic_sequence_general_formula_a_l307_307828

-- Define the sequences S_n and b_n
def S (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), a i
def b (n : ℕ) : ℝ := ∏ i in finset.range (n + 1), S i

-- Given condition
axiom condition (n : ℕ) (hn : n > 0) : 2 / S n + 1 / b n = 2

-- Prove that b_n is an arithmetic sequence
theorem b_arithmetic_sequence (n : ℕ) : 
  ∃ (b1 d : ℝ), b 1 = b1 ∧ (∀ k ≥ 1, b (k + 1) = b1 + k * d) ∧ b1 = 3 / 2 ∧ d = 1 / 2 :=
  sorry

-- Define the sequence a_n based on the findings
def a (n : ℕ) : ℝ :=
  if n = 0 then 3 / 2 else -(1 / (n * (n + 1)))

-- Prove the general formula for a_n
theorem general_formula_a (n : ℕ) : 
  a n = if n = 0 then 3 / 2 else -(1 / (n * (n + 1))) :=
  sorry

end b_arithmetic_sequence_general_formula_a_l307_307828


namespace train_time_original_l307_307740

theorem train_time_original (D : ℝ) (T : ℝ) 
  (h1 : D = 48 * T) 
  (h2 : D = 60 * (2/3)) : T = 5 / 6 := 
by
  sorry

end train_time_original_l307_307740


namespace P_Q_sum_equals_44_l307_307837

theorem P_Q_sum_equals_44 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3))) :
  P + Q = 44 :=
sorry

end P_Q_sum_equals_44_l307_307837


namespace mask_production_decrease_l307_307808

theorem mask_production_decrease (x : ℝ) : 
  (1 : ℝ) * (1 - x)^2 = 0.64 → 100 * (1 - x)^2 = 64 :=
by
  intro h
  sorry

end mask_production_decrease_l307_307808


namespace prime_gt_three_square_minus_one_divisible_by_twentyfour_l307_307851

theorem prime_gt_three_square_minus_one_divisible_by_twentyfour (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 24 ∣ (p^2 - 1) :=
sorry

end prime_gt_three_square_minus_one_divisible_by_twentyfour_l307_307851


namespace line_perpendicular_intersection_l307_307129

noncomputable def line_equation (x y : ℝ) := 3 * x + y + 2 = 0

def is_perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

theorem line_perpendicular_intersection (x y : ℝ) :
  (x - y + 2 = 0) →
  (2 * x + y + 1 = 0) →
  is_perpendicular (1 / 3) (-3) →
  line_equation x y := 
sorry

end line_perpendicular_intersection_l307_307129


namespace last_term_of_sequence_l307_307948

theorem last_term_of_sequence (u₀ : ℤ) (diffs : List ℤ) (sum_diffs : ℤ) :
  u₀ = 0 → diffs = [2, 4, -1, 0, -5, -3, 3] → sum_diffs = diffs.sum → 
  u₀ + sum_diffs = 0 := by
  sorry

end last_term_of_sequence_l307_307948


namespace b_arithmetic_a_formula_l307_307825

-- Define sequences and conditions
def S : ℕ → ℚ := λ n, (∑ i in Finset.range n, (a i))
def b : ℕ → ℚ := λ n, (∏ i in Finset.range n, (S i))

-- Given condition
axiom cond (n : ℕ) : 2 / (S n) + 1 / (b n) = 2

-- Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n, b n = (3 / 2) + (n - 1) * (1 / 2) :=
  sorry

-- General formula for {a_n}
def a : ℕ → ℚ
| 0 => (3 / 2)
| (n + 1) => -(1 / ((n + 1) * (n + 2)))

theorem a_formula : ∀ n, a n = (if n = 0 then (3 / 2) else -(1 / (n * (n + 1)))) :=
  sorry

end b_arithmetic_a_formula_l307_307825


namespace problem_a2_sub_b2_problem_a_mul_b_l307_307804

theorem problem_a2_sub_b2 {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
sorry

theorem problem_a_mul_b {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a * b = 12 :=
sorry

end problem_a2_sub_b2_problem_a_mul_b_l307_307804


namespace problem_statement_l307_307555

theorem problem_statement (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  rw [h]
  sorry

end problem_statement_l307_307555


namespace solve_for_x_l307_307104

namespace RationalOps

-- Define the custom operation ※ on rational numbers
def star (a b : ℚ) : ℚ := a + b

-- Define the equation involving the custom operation
def equation (x : ℚ) : Prop := star 4 (star x 3) = 1

-- State the theorem to prove the solution
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -6 := by
  sorry

end solve_for_x_l307_307104


namespace number_of_three_digit_numbers_with_sum_seven_l307_307577

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l307_307577


namespace area_of_square_field_l307_307075

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area of the square based on the side length
def square_area (side : ℝ) : ℝ := side * side

-- The theorem stating the area of a square with side length 15 meters
theorem area_of_square_field : square_area side_length = 225 := 
by 
  sorry

end area_of_square_field_l307_307075


namespace units_digit_of_17_pow_549_l307_307064

theorem units_digit_of_17_pow_549 : (17 ^ 549) % 10 = 7 :=
by {
  -- Provide the necessary steps or strategies to prove the theorem
  sorry
}

end units_digit_of_17_pow_549_l307_307064


namespace lcm_is_perfect_square_l307_307844

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l307_307844


namespace problem_solution_l307_307389

theorem problem_solution (a b d : ℤ) (ha : a = 2500) (hb : b = 2409) (hd : d = 81) :
  (a - b) ^ 2 / d = 102 := by
  sorry

end problem_solution_l307_307389


namespace problem_proof_l307_307561

noncomputable def problem : Prop :=
  ∀ x : ℝ, (x ≠ 2 ∧ (x-2)/(x-4) ≤ 3) ↔ (4 < x ∧ x < 5)

theorem problem_proof : problem := sorry

end problem_proof_l307_307561


namespace total_cost_l307_307819

theorem total_cost
  (permits_cost : ℕ)
  (contractor_hourly_rate : ℕ)
  (contractor_days : ℕ)
  (contractor_hours_per_day : ℕ)
  (inspector_discount : ℕ)
  (h_pc : permits_cost = 250)
  (h_chr : contractor_hourly_rate = 150)
  (h_cd : contractor_days = 3)
  (h_chpd : contractor_hours_per_day = 5)
  (h_id : inspector_discount = 80)
  (contractor_total_hours : ℕ := contractor_days * contractor_hours_per_day)
  (contractor_total_cost : ℕ := contractor_total_hours * contractor_hourly_rate)
  (inspector_cost : ℕ := contractor_total_cost - (contractor_total_cost * inspector_discount / 100))
  (total_cost : ℕ := permits_cost + contractor_total_cost + inspector_cost) :
  total_cost = 2950 :=
by
  sorry

end total_cost_l307_307819


namespace exponent_properties_l307_307437

variables (a : ℝ) (m n : ℕ)
-- Conditions
axiom h1 : a^m = 3
axiom h2 : a^n = 2

-- Goal
theorem exponent_properties :
  a^(m + n) = 6 :=
by
  sorry

end exponent_properties_l307_307437


namespace symmetric_about_one_symmetric_about_two_l307_307481

-- Part 1
theorem symmetric_about_one (rational_num_x : ℚ) (rational_num_r : ℚ) 
(h1 : 3 - 1 = 1 - rational_num_x) (hr1 : r = 3 - 1): 
  rational_num_x = -1 ∧ rational_num_r = 2 := 
by
  sorry

-- Part 2
theorem symmetric_about_two (a b : ℚ) (symmetric_radius : ℚ) 
(h2 : (a + b) / 2 = 2) (condition : |a| = 2 * |b|) : 
  symmetric_radius = 2 / 3 ∨ symmetric_radius = 6 := 
by
  sorry

end symmetric_about_one_symmetric_about_two_l307_307481


namespace rational_b_if_rational_a_l307_307162

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end rational_b_if_rational_a_l307_307162


namespace egg_production_difference_l307_307165

-- Define the conditions
def last_year_production : ℕ := 1416
def this_year_production : ℕ := 4636

-- Define the theorem statement
theorem egg_production_difference :
  this_year_production - last_year_production = 3220 :=
by
  sorry

end egg_production_difference_l307_307165


namespace abc_not_less_than_two_l307_307984

theorem abc_not_less_than_two (a b c : ℝ) (h : a + b + c = 6) : a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2 :=
sorry

end abc_not_less_than_two_l307_307984


namespace five_letter_arrangements_count_l307_307148

theorem five_letter_arrangements_count :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
      arrangements := { l : List Char // ∃ h l', l = 'D' :: l' ++ [h] ∧
                       ¬(h = 'G') ∧
                       'B' ∈ l' ∧
                       l.length = 4 ∧
                       ∀ c ∈ l, c ∈ letters ∧ -- Validity check
                       ∀ c, In c l → ∀ c', In c' l → (c ≠ c')} -- Uniqueness constraint
  in arrangements.toFinset.card = 960 := sorry

end five_letter_arrangements_count_l307_307148


namespace starting_number_range_l307_307977

theorem starting_number_range (n : ℕ) (h₁: ∀ m : ℕ, (m > n) → (m ≤ 50) → (m = 55) → True) : n = 54 :=
sorry

end starting_number_range_l307_307977


namespace stratified_sampling_l307_307898

/-- Given a batch of 98 water heaters with 56 from Factory A and 42 from Factory B,
    and a stratified sample of 14 units is to be drawn, prove that the number 
    of water heaters sampled from Factory A is 8 and from Factory B is 6. --/

theorem stratified_sampling (batch_size A B sample_size : ℕ) 
  (h_batch : batch_size = 98) 
  (h_fact_a : A = 56) 
  (h_fact_b : B = 42) 
  (h_sample : sample_size = 14) : 
  (A * sample_size / batch_size = 8) ∧ (B * sample_size / batch_size = 6) := 
  by
    sorry

end stratified_sampling_l307_307898


namespace probability_exactly_one_win_l307_307045

theorem probability_exactly_one_win :
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  P_exactly_one_win = 8 / 15 :=
by
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  have h1 : P_exactly_one_win = 8 / 15 := sorry
  exact h1

end probability_exactly_one_win_l307_307045


namespace num_three_digit_sums7_l307_307607

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l307_307607


namespace additional_days_when_selling_5_goats_l307_307401

variables (G D F X : ℕ)

def total_feed (num_goats days : ℕ) := G * num_goats * days

theorem additional_days_when_selling_5_goats
  (h1 : total_feed G 20 D = F)
  (h2 : total_feed G 15 (D + X) = F)
  (h3 : total_feed G 30 (D - 3) = F):
  X = 9 :=
by
  -- the exact proof is omitted and presented as 'sorry'
  sorry

end additional_days_when_selling_5_goats_l307_307401


namespace distance_from_P_to_focus_l307_307439

-- Define the parabola equation and the definition of the point P
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the given condition that P's distance to the x-axis is 12
def point_P (x y : ℝ) : Prop := parabola x y ∧ |y| = 12

-- The Lean proof problem statement
theorem distance_from_P_to_focus :
  ∃ (x y : ℝ), point_P x y → dist (x, y) (4, 0) = 13 :=
by {
  sorry   -- proof to be completed
}

end distance_from_P_to_focus_l307_307439


namespace three_digit_integers_sum_to_7_l307_307585

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l307_307585


namespace polygon_properties_l307_307702

theorem polygon_properties
  (n : ℕ)
  (h_exterior_angle : 360 / 20 = n)
  (h_n_sides : n = 18) :
  (180 * (n - 2) = 2880) ∧ (n * (n - 3) / 2 = 135) :=
by
  sorry

end polygon_properties_l307_307702


namespace number_of_three_digit_numbers_with_sum_7_l307_307568

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l307_307568


namespace distance_between_first_and_last_is_140_l307_307429

-- Given conditions
def eightFlowers : ℕ := 8
def distanceFirstToFifth : ℕ := 80
def intervalsBetweenFirstAndFifth : ℕ := 4 -- 1 to 5 means 4 intervals
def intervalsBetweenFirstAndLast : ℕ := 7 -- 1 to 8 means 7 intervals
def distanceBetweenConsecutiveFlowers : ℕ := distanceFirstToFifth / intervalsBetweenFirstAndFifth
def totalDistanceFirstToLast : ℕ := distanceBetweenConsecutiveFlowers * intervalsBetweenFirstAndLast

-- Theorem to prove the question equals the correct answer
theorem distance_between_first_and_last_is_140 :
  totalDistanceFirstToLast = 140 := by
  sorry

end distance_between_first_and_last_is_140_l307_307429


namespace twelve_pharmacies_not_enough_l307_307530

-- Define the grid dimensions and necessary parameters
def grid_size := 9
def total_intersections := (grid_size + 1) * (grid_size + 1) -- 10 * 10 grid
def walking_distance := 3
def coverage_side := (walking_distance * 2 + 1)  -- 7x7 grid coverage
def max_covered_per_pharmacy := (coverage_side - 1) * (coverage_side - 1)  -- Coverage per direction

-- Define the main theorem
theorem twelve_pharmacies_not_enough (n m : ℕ): 
  n = grid_size + 1 -> m = grid_size + 1 -> total_intersections = n * m -> 
  (walking_distance < n) -> (walking_distance < m) -> (pharmacies : ℕ) -> pharmacies = 12 ->
  (coverage_side <= n) -> (coverage_side <= m) ->
  ¬ (∀ i j : ℕ, i < n -> j < m -> ∃ p : ℕ, p < pharmacies -> 
  abs (i - (p / (grid_size + 1))) + abs (j - (p % (grid_size + 1))) ≤ walking_distance) :=
begin
  intros,
  sorry -- Proof omitted
end

end twelve_pharmacies_not_enough_l307_307530


namespace area_ratio_of_squares_l307_307346

theorem area_ratio_of_squares (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a * a) = 16 * (b * b) :=
by
  sorry

end area_ratio_of_squares_l307_307346


namespace triangle_side_range_l307_307463

theorem triangle_side_range (AB AC x : ℝ) (hAB : AB = 16) (hAC : AC = 7) (hBC : BC = x) :
  9 < x ∧ x < 23 :=
by
  sorry

end triangle_side_range_l307_307463


namespace three_digit_integers_sum_to_7_l307_307589

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l307_307589


namespace select_3_people_in_5x5_matrix_l307_307455

theorem select_3_people_in_5x5_matrix :
  let n := 5
  let k := 3
  choose n k * k! == 600 :=
by
  let n := 5
  let k := 3
  have h1 : choose n k = Nat.choose n k := rfl
  have h2 : k! = Nat.factorial k := rfl
  have h3 : Nat.choose 5 3 = 10 := rfl
  have h4 : Nat.factorial 3 = 6 := rfl
  calc
    Nat.choose 5 3 * 3! = 10 * 6 : by rw [h3, h4]
                 ... = 60 := by rfl
                 ... = 600 := by sorry

end select_3_people_in_5x5_matrix_l307_307455


namespace shaded_areas_I_and_III_equal_l307_307053

def area_shaded_square_I : ℚ := 1 / 4
def area_shaded_square_II : ℚ := 1 / 2
def area_shaded_square_III : ℚ := 1 / 4

theorem shaded_areas_I_and_III_equal :
  area_shaded_square_I = area_shaded_square_III ∧
   area_shaded_square_I ≠ area_shaded_square_II ∧
   area_shaded_square_III ≠ area_shaded_square_II :=
by {
  sorry
}

end shaded_areas_I_and_III_equal_l307_307053


namespace alpha_squared_plus_3alpha_plus_beta_equals_2023_l307_307792

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end alpha_squared_plus_3alpha_plus_beta_equals_2023_l307_307792


namespace num_three_digit_integers_sum_to_seven_l307_307583

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l307_307583


namespace determine_a_b_l307_307114

-- Define the polynomial expression
def poly (x a b : ℝ) : ℝ := x^2 + a * x + b

-- Define the factored form
def factored_poly (x : ℝ) : ℝ := (x + 1) * (x - 3)

-- State the theorem
theorem determine_a_b (a b : ℝ) (h : ∀ x, poly x a b = factored_poly x) : a = -2 ∧ b = -3 :=
by 
  sorry

end determine_a_b_l307_307114


namespace caterer_min_people_l307_307987

theorem caterer_min_people (x : ℕ) : 150 + 18 * x > 250 + 15 * x → x ≥ 34 :=
by
  intro h
  sorry

end caterer_min_people_l307_307987


namespace boy_running_time_l307_307787

theorem boy_running_time :
  let side_length := 60
  let speed1 := 9 * 1000 / 3600       -- 9 km/h to m/s
  let speed2 := 6 * 1000 / 3600       -- 6 km/h to m/s
  let speed3 := 8 * 1000 / 3600       -- 8 km/h to m/s
  let speed4 := 7 * 1000 / 3600       -- 7 km/h to m/s
  let hurdle_time := 5 * 3 * 4        -- 3 hurdles per side, 4 sides
  let time1 := side_length / speed1
  let time2 := side_length / speed2
  let time3 := side_length / speed3
  let time4 := side_length / speed4
  let total_time := time1 + time2 + time3 + time4 + hurdle_time
  total_time = 177.86 := by
{
  -- actual proof would be provided here
  sorry
}

end boy_running_time_l307_307787


namespace sufficient_but_not_necessary_condition_l307_307127

open Real

theorem sufficient_but_not_necessary_condition :
  ∀ (m : ℝ),
  (∀ x, (x^2 - 3*x - 4 ≤ 0) → (x^2 - 6*x + 9 - m^2 ≤ 0)) ∧
  (∃ x, ¬(x^2 - 3*x - 4 ≤ 0) ∧ (x^2 - 6*x + 9 - m^2 ≤ 0)) ↔
  m ∈ Set.Iic (-4) ∪ Set.Ici 4 :=
by
  sorry

end sufficient_but_not_necessary_condition_l307_307127


namespace Sasha_added_digit_l307_307329

noncomputable def Kolya_number : Nat := 45 -- Sum of all digits 0 to 9

theorem Sasha_added_digit (d x : Nat) (h : 0 ≤ d ∧ d ≤ 9) (h1 : 0 ≤ x ∧ x ≤ 9) (condition : Kolya_number - d + x ≡ 0 [MOD 9]) : x = 0 ∨ x = 9 := 
sorry

end Sasha_added_digit_l307_307329


namespace calculate_x_l307_307703

theorem calculate_x (a b x : ℕ) (h1 : b = 9) (h2 : b - a = 5) (h3 : a * b = 2 * (a + b) + x) : x = 10 :=
by
  sorry

end calculate_x_l307_307703


namespace find_n_l307_307115

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end find_n_l307_307115


namespace jill_travel_time_to_school_is_20_minutes_l307_307750

variables (dave_rate : ℕ) (dave_step : ℕ) (dave_time : ℕ)
variables (jill_rate : ℕ) (jill_step : ℕ)

def dave_distance : ℕ := dave_rate * dave_step * dave_time
def jill_time_to_school : ℕ := dave_distance dave_rate dave_step dave_time / (jill_rate * jill_step)

theorem jill_travel_time_to_school_is_20_minutes : 
  dave_rate = 85 → dave_step = 80 → dave_time = 18 → 
  jill_rate = 120 → jill_step = 50 → jill_time_to_school 85 80 18 120 50 = 20 :=
by
  intros
  unfold jill_time_to_school
  unfold dave_distance
  sorry

end jill_travel_time_to_school_is_20_minutes_l307_307750


namespace travel_time_l307_307701

noncomputable def distance (time: ℝ) (rate: ℝ) : ℝ := time * rate

theorem travel_time
  (initial_time: ℝ)
  (initial_speed: ℝ)
  (reduced_speed: ℝ)
  (stopover: ℝ)
  (h1: initial_time = 4)
  (h2: initial_speed = 80)
  (h3: reduced_speed = 50)
  (h4: stopover = 0.5) :
  (distance initial_time initial_speed) / reduced_speed + stopover = 6.9 := 
by
  sorry

end travel_time_l307_307701


namespace triangles_side_product_relation_l307_307930

-- Define the two triangles with their respective angles and side lengths
variables (A B C A1 B1 C1 : Type) 
          (angle_A angle_A1 angle_B angle_B1 : ℝ) 
          (a b c a1 b1 c1 : ℝ)

-- Given conditions
def angles_sum_to_180 (angle_A angle_A1 : ℝ) : Prop :=
  angle_A + angle_A1 = 180

def angles_equal (angle_B angle_B1 : ℝ) : Prop :=
  angle_B = angle_B1

-- The main theorem to be proven
theorem triangles_side_product_relation 
  (h1 : angles_sum_to_180 angle_A angle_A1)
  (h2 : angles_equal angle_B angle_B1) :
  a * a1 = b * b1 + c * c1 :=
sorry

end triangles_side_product_relation_l307_307930


namespace incorrect_solution_among_four_l307_307488

theorem incorrect_solution_among_four 
  (x y : ℤ) 
  (h1 : 2 * x - 3 * y = 5) 
  (h2 : 3 * x - 2 * y = 7) : 
  ¬ ((2 * (2 * x - 3 * y) - ((-3) * (3 * x - 2 * y))) = (2 * 5 - (-3) * 7)) :=
sorry

end incorrect_solution_among_four_l307_307488


namespace lizz_team_loses_by_8_points_l307_307686

-- Definitions of the given conditions
def initial_deficit : ℕ := 20
def free_throw_points : ℕ := 5 * 1
def three_pointer_points : ℕ := 3 * 3
def jump_shot_points : ℕ := 4 * 2
def liz_points : ℕ := free_throw_points + three_pointer_points + jump_shot_points
def other_team_points : ℕ := 10
def points_caught_up : ℕ := liz_points - other_team_points
def final_deficit : ℕ := initial_deficit - points_caught_up

-- Theorem proving Liz's team loses by 8 points
theorem lizz_team_loses_by_8_points : final_deficit = 8 :=
  by
    -- Proof will be here
    sorry

end lizz_team_loses_by_8_points_l307_307686


namespace area_between_circles_l307_307203

noncomputable def k_value (θ : ℝ) : ℝ := Real.tan θ

theorem area_between_circles {θ k : ℝ} (h₁ : k = Real.tan θ) (h₂ : θ = 4/3) (h_area : (3 * θ / 2) = 2) :
  k = Real.tan (4/3) :=
sorry

end area_between_circles_l307_307203


namespace bill_profit_difference_l307_307263

theorem bill_profit_difference (P : ℝ) 
  (h1 : 1.10 * P = 549.9999999999995)
  (h2 : ∀ NP NSP, NP = 0.90 * P ∧ NSP = 1.30 * NP →
  NSP - 549.9999999999995 = 35) :
  true :=
by {
  sorry
}

end bill_profit_difference_l307_307263


namespace pairwise_sums_l307_307476

theorem pairwise_sums (
  a b c d e : ℕ
) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧
  (a + b = 21) ∧ (a + c = 26) ∧ (a + d = 35) ∧ (a + e = 40) ∧
  (b + c = 49) ∧ (b + d = 51) ∧ (b + e = 54) ∧ (c + d = 60) ∧
  (c + e = 65) ∧ (d + e = 79)
  ↔ 
  (a = 6) ∧ (b = 15) ∧ (c = 20) ∧ (d = 34) ∧ (e = 45) := 
by 
  sorry

end pairwise_sums_l307_307476


namespace mary_max_earnings_l307_307237

def regular_rate : ℝ := 8
def max_hours : ℝ := 60
def regular_hours : ℝ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate
def overtime_hours : ℝ := max_hours - regular_hours
def earnings_regular : ℝ := regular_hours * regular_rate
def earnings_overtime : ℝ := overtime_hours * overtime_rate
def total_earnings : ℝ := earnings_regular + earnings_overtime

theorem mary_max_earnings : total_earnings = 560 := by
  sorry

end mary_max_earnings_l307_307237


namespace victors_friend_decks_l307_307715

theorem victors_friend_decks:
  ∀ (deck_cost : ℕ) (victor_decks : ℕ) (total_spent : ℕ)
  (friend_decks : ℕ),
  deck_cost = 8 →
  victor_decks = 6 →
  total_spent = 64 →
  (victor_decks * deck_cost + friend_decks * deck_cost = total_spent) →
  friend_decks = 2 :=
by
  intros deck_cost victor_decks total_spent friend_decks hc hv ht heq
  sorry

end victors_friend_decks_l307_307715


namespace square_area_with_circles_l307_307254

theorem square_area_with_circles
  (radius : ℝ) 
  (side_length : ℝ)
  (area : ℝ)
  (h_radius : radius = 7) 
  (h_side_length : side_length = 2 * (2 * radius)) 
  (h_area : area = side_length ^ 2) : 
  area = 784 := by
  sorry

end square_area_with_circles_l307_307254


namespace num_three_digit_integers_sum_to_seven_l307_307579

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l307_307579


namespace power_function_passes_through_fixed_point_l307_307496

theorem power_function_passes_through_fixed_point 
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
  (P : ℝ × ℝ) (hP : P = (4, 2))
  (f : ℝ → ℝ) (hf : f x = x ^ a) : ∀ x, f x = x ^ (1 / 2) :=
by
  sorry

end power_function_passes_through_fixed_point_l307_307496


namespace find_x_l307_307170

theorem find_x (a b x : ℝ) (h_a : a > 0) (h_b : b > 0) (h_x : x > 0)
  (s : ℝ) (h_s1 : s = (a ^ 2) ^ (4 * b)) (h_s2 : s = a ^ (2 * b) * x ^ (3 * b)) :
  x = a ^ 2 :=
sorry

end find_x_l307_307170


namespace count_good_numbers_l307_307219

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l307_307219


namespace transformation_correct_l307_307949

theorem transformation_correct (a x y : ℝ) (h : a * x = a * y) : 3 - a * x = 3 - a * y :=
sorry

end transformation_correct_l307_307949


namespace problem_solution_l307_307136

theorem problem_solution (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 2) (h3 : a^3 + b^3 + c^3 = 3) :
  (a * b * c = 1 / 6) ∧ (a^4 + b^4 + c^4 = 25 / 6) :=
by {
  sorry
}

end problem_solution_l307_307136


namespace grims_groks_zeets_l307_307312

variable {T : Type}
variable (Groks Zeets Grims Snarks : Set T)

-- Given conditions as definitions in Lean 4
variable (h1 : Groks ⊆ Zeets)
variable (h2 : Grims ⊆ Zeets)
variable (h3 : Snarks ⊆ Groks)
variable (h4 : Grims ⊆ Snarks)

-- The statement to be proved
theorem grims_groks_zeets : Grims ⊆ Groks ∧ Grims ⊆ Zeets := by
  sorry

end grims_groks_zeets_l307_307312


namespace count_false_propositions_l307_307499

def prop (a : ℝ) := a > 1 → a > 2
def converse (a : ℝ) := a > 2 → a > 1
def inverse (a : ℝ) := a ≤ 1 → a ≤ 2
def contrapositive (a : ℝ) := a ≤ 2 → a ≤ 1

theorem count_false_propositions (a : ℝ) (h : ¬(prop a)) : 
  (¬(prop a) ∧ ¬(contrapositive a)) ∧ (converse a ∧ inverse a) ↔ 2 = 2 := 
  by
    sorry

end count_false_propositions_l307_307499


namespace tom_change_l307_307879

theorem tom_change :
  let SNES_value := 150
  let credit_percent := 0.80
  let amount_given := 80
  let game_value := 30
  let NES_sale_price := 160
  let credit_for_SNES := credit_percent * SNES_value
  let amount_to_pay_for_NES := NES_sale_price - credit_for_SNES
  let effective_amount_paid := amount_to_pay_for_NES - game_value
  let change_received := amount_given - effective_amount_paid
  change_received = 70 :=
by
  sorry

end tom_change_l307_307879


namespace sum_powers_l307_307330

theorem sum_powers :
  ∃ (α β γ : ℂ), α + β + γ = 2 ∧ α^2 + β^2 + γ^2 = 5 ∧ α^3 + β^3 + γ^3 = 8 ∧ α^5 + β^5 + γ^5 = 46.5 :=
by
  sorry

end sum_powers_l307_307330


namespace three_digit_numbers_sum_seven_l307_307595

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l307_307595


namespace sum_of_areas_of_circles_l307_307370

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l307_307370


namespace relatively_prime_probability_42_l307_307884

theorem relatively_prime_probability_42 : 
  (λ x, (x ≤ 42 ∧ x > 0 ∧ Nat.gcd x 42 = 1)) / (λ x, (x ≤ 42 ∧ x > 0)) = 2/7 :=
by 
  sorry

end relatively_prime_probability_42_l307_307884


namespace sum_areas_of_tangent_circles_l307_307705

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end sum_areas_of_tangent_circles_l307_307705


namespace union_complement_real_domain_l307_307950

noncomputable def M : Set ℝ := {x | -2 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -2 < x}

theorem union_complement_real_domain :
  M ∪ (Set.univ \ N) = {x : ℝ | x < 2} :=
by
  sorry

end union_complement_real_domain_l307_307950


namespace rationalize_denominator_l307_307036

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l307_307036


namespace total_balloons_l307_307483

theorem total_balloons:
  ∀ (R1 R2 G1 G2 B1 B2 Y1 Y2 O1 O2: ℕ),
    R1 = 31 →
    R2 = 24 →
    G1 = 15 →
    G2 = 7 →
    B1 = 12 →
    B2 = 14 →
    Y1 = 18 →
    Y2 = 20 →
    O1 = 10 →
    O2 = 16 →
    (R1 + R2 = 55) ∧
    (G1 + G2 = 22) ∧
    (B1 + B2 = 26) ∧
    (Y1 + Y2 = 38) ∧
    (O1 + O2 = 26) :=
by
  intros
  sorry

end total_balloons_l307_307483


namespace debate_team_selections_l307_307187

theorem debate_team_selections
  (A_selected C_selected B_selected E_selected : Prop)
  (h1: A_selected ∨ C_selected)
  (h2: B_selected ∨ E_selected)
  (h3: ¬ (B_selected ∧ E_selected) ∧ ¬ (C_selected ∧ E_selected))
  (not_B_selected : ¬ B_selected) :
  A_selected ∧ E_selected :=
by
  sorry

end debate_team_selections_l307_307187


namespace find_n_l307_307194

def f (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + n
def g (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + 5 * n

theorem find_n (n : ℝ) (h : 3 * f 3 n = 2 * g 3 n) : n = 9 / 7 := by
  sorry

end find_n_l307_307194


namespace number_of_three_digit_numbers_with_sum_seven_l307_307578

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l307_307578


namespace john_spending_l307_307323

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end john_spending_l307_307323


namespace find_k_l307_307962

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l307_307962


namespace half_abs_diff_squares_eq_40_l307_307227

theorem half_abs_diff_squares_eq_40 (x y : ℤ) (hx : x = 21) (hy : y = 19) :
  (|x^2 - y^2| / 2) = 40 :=
by
  sorry

end half_abs_diff_squares_eq_40_l307_307227


namespace geom_prog_all_integers_l307_307918

theorem geom_prog_all_integers (b : ℕ) (r : ℚ) (a c : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, b * r ^ n = a * n + c) ∧ ∃ b_1 : ℤ, b = b_1 →
  (∀ n : ℕ, ∃ b_n : ℤ, b * r ^ n = b_n) :=
by
  sorry

end geom_prog_all_integers_l307_307918


namespace probability_same_color_l307_307073

/-
Problem statement:
Given a bag contains 6 green balls and 7 white balls,
if two balls are drawn simultaneously, prove that the probability 
that both balls are the same color is 6/13.
-/

theorem probability_same_color
  (total_balls : ℕ := 6 + 7)
  (green_balls : ℕ := 6)
  (white_balls : ℕ := 7)
  (two_balls_drawn_simultaneously : Prop := true) :
  ((green_balls / total_balls) * ((green_balls - 1) / (total_balls - 1))) +
  ((white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))) = 6 / 13 :=
sorry

end probability_same_color_l307_307073


namespace option_a_option_b_option_d_l307_307143

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l307_307143


namespace sum_of_areas_of_circles_l307_307379

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l307_307379


namespace still_water_speed_l307_307085

-- The conditions as given in the problem
variables (V_m V_r V'_r : ℝ)
axiom upstream_speed : V_m - V_r = 20
axiom downstream_increased_speed : V_m + V_r = 30
axiom downstream_reduced_speed : V_m + V'_r = 26

-- Prove that the man's speed in still water is 25 km/h
theorem still_water_speed : V_m = 25 :=
by
  sorry

end still_water_speed_l307_307085


namespace central_angle_agree_l307_307202

theorem central_angle_agree (ratio_agree : ℕ) (ratio_disagree : ℕ) (ratio_no_preference : ℕ) (total_angle : ℝ) :
  ratio_agree = 7 → ratio_disagree = 2 → ratio_no_preference = 1 → total_angle = 360 →
  (ratio_agree / (ratio_agree + ratio_disagree + ratio_no_preference) * total_angle = 252) :=
by
  -- conditions and assumptions
  intros h_agree h_disagree h_no_preference h_total_angle
  -- simplified steps here
  sorry

end central_angle_agree_l307_307202


namespace three_digit_integers_sum_to_7_l307_307586

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l307_307586


namespace algebraic_expression_evaluation_l307_307795

theorem algebraic_expression_evaluation (a b : ℤ) (h : a - 3 * b = -3) : 5 - a + 3 * b = 8 :=
by 
  sorry

end algebraic_expression_evaluation_l307_307795


namespace male_students_count_l307_307867

variable (M F : ℕ)
variable (average_all average_male average_female : ℕ)
variable (total_male total_female total_all : ℕ)

noncomputable def male_students (M F : ℕ) : ℕ := 8

theorem male_students_count:
  F = 32 -> average_all = 90 -> average_male = 82 -> average_female = 92 ->
  total_male = average_male * M -> total_female = average_female * F -> 
  total_all = average_all * (M + F) -> total_male + total_female = total_all ->
  M = male_students M F := 
by
  intros hF hAvgAll hAvgMale hAvgFemale hTotalMale hTotalFemale hTotalAll hEqTotal
  sorry

end male_students_count_l307_307867


namespace mean_score_is_74_l307_307236

theorem mean_score_is_74 (M SD : ℝ) 
  (h1 : 58 = M - 2 * SD) 
  (h2 : 98 = M + 3 * SD) : 
  M = 74 := 
by 
  -- problem statement without solving steps
  sorry

end mean_score_is_74_l307_307236


namespace initial_candies_equal_twenty_l307_307347

-- Definitions based on conditions
def friends : ℕ := 6
def candies_per_friend : ℕ := 4
def total_needed_candies : ℕ := friends * candies_per_friend
def additional_candies : ℕ := 4

-- Main statement
theorem initial_candies_equal_twenty :
  (total_needed_candies - additional_candies) = 20 := by
  sorry

end initial_candies_equal_twenty_l307_307347


namespace find_x_l307_307972

-- Definitions based on the conditions
def remaining_scores_after_removal (s: List ℕ) : List ℕ :=
  s.erase 87 |>.erase 94

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Converting the given problem into a Lean 4 theorem statement
theorem find_x (x : ℕ) (s : List ℕ) :
  s = [94, 87, 89, 88, 92, 90, x, 93, 92, 91] →
  average (remaining_scores_after_removal s) = 91 →
  x = 2 :=
by
  intros h1 h2
  sorry

end find_x_l307_307972


namespace max_gcd_of_sequence_l307_307046

def term (n : ℕ) : ℕ := 100 + n^2

def gcd_terms (n : ℕ) : ℕ := Int.gcd (term n) (term (n + 1))

theorem max_gcd_of_sequence : ∃ n : ℕ, ∀ m : ℕ, gcd_terms m ≤ 401 ∧ gcd_terms 200 = 401 :=
by
  sorry

end max_gcd_of_sequence_l307_307046


namespace tens_digit_of_23_pow_1987_l307_307275

theorem tens_digit_of_23_pow_1987 : (23 ^ 1987 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_23_pow_1987_l307_307275


namespace remainder_of_3_pow_500_mod_17_l307_307721

theorem remainder_of_3_pow_500_mod_17 : (3 ^ 500) % 17 = 13 := 
by
  sorry

end remainder_of_3_pow_500_mod_17_l307_307721


namespace half_angle_in_first_quadrant_l307_307779

theorem half_angle_in_first_quadrant {α : ℝ} (h : 0 < α ∧ α < π / 2) : 
  0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l307_307779


namespace right_handed_total_l307_307179

theorem right_handed_total (total_players throwers : Nat) (h1 : total_players = 70) (h2 : throwers = 37) :
  let non_throwers := total_players - throwers
  let left_handed := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed
  let right_handed := right_handed_non_throwers + throwers
  right_handed = 59 :=
by
  sorry

end right_handed_total_l307_307179


namespace sid_initial_money_l307_307351

variable (M : ℝ)
variable (spent_on_accessories : ℝ := 12)
variable (spent_on_snacks : ℝ := 8)
variable (remaining_money_condition : ℝ := (M / 2) + 4)

theorem sid_initial_money : (M = 48) → (remaining_money_condition = M - (spent_on_accessories + spent_on_snacks)) :=
by
  sorry

end sid_initial_money_l307_307351


namespace relatively_prime_probability_l307_307886

theorem relatively_prime_probability (n : ℕ) (h : n = 42) :
  let phi := n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7) in
  (phi / n) = 2 / 7 :=
by
  sorry

end relatively_prime_probability_l307_307886


namespace number_of_teams_in_BIG_N_l307_307160

theorem number_of_teams_in_BIG_N (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end number_of_teams_in_BIG_N_l307_307160


namespace boat_distance_against_stream_l307_307159

-- Define the conditions
variable (v_s : ℝ)
variable (speed_still_water : ℝ := 9)
variable (distance_downstream : ℝ := 13)

-- Assert the given condition
axiom condition : speed_still_water + v_s = distance_downstream

-- Prove the required distance against the stream
theorem boat_distance_against_stream : (speed_still_water - (distance_downstream - speed_still_water)) = 5 :=
by
  sorry

end boat_distance_against_stream_l307_307159


namespace sum_of_areas_of_circles_l307_307371

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l307_307371


namespace exists_F_12_mod_23_zero_l307_307989

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l307_307989


namespace factor_expression_l307_307937

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) :=
sorry

end factor_expression_l307_307937


namespace f_nonneg_f_positive_f_zero_condition_l307_307440

noncomputable def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) +
  B * (a^2 * b + b^2 * c + c^2 * a + a * b^2 + b * c^2 + c * a^2) +
  C * a * b * c

theorem f_nonneg (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 ≥ 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

theorem f_positive (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 > 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c > 0 :=
by sorry

theorem f_zero_condition (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 = 0) 
  (h2 : f A B C 1 1 0 > 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

end f_nonneg_f_positive_f_zero_condition_l307_307440


namespace not_enough_pharmacies_l307_307529

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l307_307529


namespace proposition_A_proposition_B_proposition_C_proposition_D_l307_307522

theorem proposition_A (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a ≠ 1) :=
by {
  sorry
}

theorem proposition_B : (¬ ∀ x : ℝ, x^2 + x + 1 < 0) → (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by {
  sorry
}

theorem proposition_C : ¬ ∀ x ≠ 0, x + 1 / x ≥ 2 :=
by {
  sorry
}

theorem proposition_D (m : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 2) ∧ x^2 + m * x + 4 < 0) → m < -4 :=
by {
  sorry
}

end proposition_A_proposition_B_proposition_C_proposition_D_l307_307522


namespace correct_options_l307_307139

theorem correct_options (a b : ℝ) (h_a_pos : a > 0) (h_discriminant : a^2 = 4 * b):
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬(∃ x1 x2 : ℝ, (x1 * x2 > 0 ∧ a^2 - x1x2 ≠ 4b))) ∧ 
  (∀ c x1 x2 : ℝ, (x1 - x2 = 4) → (a^2 - 4 * (b - c) = 16) → (c = 4)) :=
by
  sorry

end correct_options_l307_307139


namespace evaluate_g_at_8_l307_307169

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 37 * x ^ 2 - 28 * x - 84

theorem evaluate_g_at_8 : g 8 = 1036 :=
by
  sorry

end evaluate_g_at_8_l307_307169


namespace sum_n_binom_30_15_eq_31_16_l307_307516

open Nat

-- Given n = 30 and k = 15, we are given the components to test Pascal's identity
def PascalIdentity (n k : Nat) : Prop :=
  Nat.choose (n-1) (k-1) + Nat.choose (n-1) k = Nat.choose n k

theorem sum_n_binom_30_15_eq_31_16 : 
  (∑ n in { n : ℕ | Nat.choose 30 15 + Nat.choose 30 n = Nat.choose 31 16 }, n) = 30 := 
sorry

end sum_n_binom_30_15_eq_31_16_l307_307516


namespace three_digit_sum_seven_l307_307620

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l307_307620


namespace smallest_a_for_polynomial_roots_l307_307361

theorem smallest_a_for_polynomial_roots :
  ∃ (a b c : ℕ), 
         (∃ (r s t u : ℕ), r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧ r * s * t * u = 5160 ∧ a = r + s + t + u) 
    ∧  (∀ (r' s' t' u' : ℕ), r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧ r' * s' * t' * u' = 5160 ∧ r' + s' + t' + u' < a → false) 
    := sorry

end smallest_a_for_polynomial_roots_l307_307361


namespace machine_working_time_l307_307507

theorem machine_working_time (y : ℝ) :
  (1 / (y + 4) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) → y = 2 :=
by
  sorry

end machine_working_time_l307_307507


namespace total_photos_l307_307839

-- Define the number of photos Claire has taken
def photos_by_Claire : ℕ := 8

-- Define the number of photos Lisa has taken
def photos_by_Lisa : ℕ := 3 * photos_by_Claire

-- Define the number of photos Robert has taken
def photos_by_Robert : ℕ := photos_by_Claire + 16

-- State the theorem we want to prove
theorem total_photos : photos_by_Lisa + photos_by_Robert = 48 :=
by
  sorry

end total_photos_l307_307839


namespace odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l307_307684

def is_in_A (a : ℤ) : Prop := ∃ (x y : ℤ), a = x^2 - y^2

theorem odd_numbers_in_A :
  ∀ (n : ℤ), n % 2 = 1 → is_in_A n :=
sorry

theorem even_4k_minus_2_not_in_A :
  ∀ (k : ℤ), ¬ is_in_A (4 * k - 2) :=
sorry

theorem product_in_A :
  ∀ (a b : ℤ), is_in_A a → is_in_A b → is_in_A (a * b) :=
sorry

end odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l307_307684


namespace number_of_three_digit_numbers_with_digit_sum_seven_l307_307629

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l307_307629


namespace set_A_enum_l307_307753

def A : Set ℤ := {z | ∃ x : ℕ, 6 / (x - 2) = z ∧ 6 % (x - 2) = 0}

theorem set_A_enum : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end set_A_enum_l307_307753


namespace expression_value_l307_307796

variable (m n : ℝ)

theorem expression_value (h : m - n = 1) : (m - n)^2 - 2 * m + 2 * n = -1 :=
by
  sorry

end expression_value_l307_307796


namespace sum_of_distances_l307_307540

noncomputable def sum_distances (d_AB : ℝ) (d_A : ℝ) (d_B : ℝ) : ℝ :=
d_A + d_B

theorem sum_of_distances
    (tangent_to_sides : Circle → Point → Point → Prop)
    (C_on_circle : Circle → Point → Prop)
    (A B C : Point)
    (γ : Circle)
    (h_dist_to_AB : ∃ C, C_on_circle γ C → distance_from_line C A B = 4)
    (h_ratio : ∃ hA hB, distance_from_side C A = hA ∧ distance_from_side C B = hB ∧ (hA = 4 * hB ∨ hB = 4 * hA)) :
    sum_distances (distance_from_line C A B) (distance_from_side C A) (distance_from_side C B) = 10 :=
by
  sorry

end sum_of_distances_l307_307540


namespace minimum_distance_midpoint_l307_307300

theorem minimum_distance_midpoint 
    (θ : ℝ)
    (P : ℝ × ℝ := (-4, 4))
    (C1_standard : ∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = 1)
    (C2_standard : ∀ (x y : ℝ), x^2 / 64 + y^2 / 9 = 1)
    (Q : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ))
    (M : ℝ × ℝ := (-2 + 4 * Real.cos θ, 2 + 3 / 2 * Real.sin θ))
    (C3_standard : ∀ (x y : ℝ), x - 2*y - 7 = 0) :
    ∃ (θ : ℝ), θ = Real.arcsin (-3/5) ∧ (θ = Real.arccos 4/5) ∧
    (∀ (d : ℝ), d = abs (5 * Real.sin (Real.arctan (4 / 3) - θ) - 13) / Real.sqrt 5 ∧ 
    d = 8 * Real.sqrt 5 / 5) :=
sorry

end minimum_distance_midpoint_l307_307300


namespace speed_of_man_in_still_water_l307_307086

-- Define the conditions as given in step (a)
axiom conditions :
  ∃ (v_m v_s : ℝ),
    (40 / 5 = v_m + v_s) ∧
    (30 / 5 = v_m - v_s)

-- State the theorem that proves the speed of the man in still water
theorem speed_of_man_in_still_water : ∃ v_m : ℝ, v_m = 7 :=
by
  obtain ⟨v_m, v_s, h1, h2⟩ := conditions
  have h3 : v_m + v_s = 8 := by sorry
  have h4 : v_m - v_s = 6 := by sorry
  have h5 : 2 * v_m = 14 := by sorry
  exact ⟨7, by linarith⟩

end speed_of_man_in_still_water_l307_307086


namespace edge_length_of_inscribed_cube_in_sphere_l307_307125

noncomputable def edge_length_of_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) : ℝ :=
  let x := 2 * Real.sqrt 3
  x

theorem edge_length_of_inscribed_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) :
  edge_length_of_cube_in_sphere surface_area_sphere π_cond = 2 * Real.sqrt 3 :=
by
  sorry

end edge_length_of_inscribed_cube_in_sphere_l307_307125


namespace f_monotonicity_l307_307130

noncomputable def f (x : ℝ) : ℝ := abs (x^2 - 1)

theorem f_monotonicity :
  (∀ x y : ℝ, (-1 < x ∧ x < 0 ∧ x < y ∧ y < 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < -1 ∧ y < -1 ∧ y < x) → f x < f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ y < x) → f x < f y) :=
by
  sorry

end f_monotonicity_l307_307130


namespace thermos_count_l307_307079

theorem thermos_count
  (total_gallons : ℝ)
  (pints_per_gallon : ℝ)
  (thermoses_drunk_by_genevieve : ℕ)
  (pints_drunk_by_genevieve : ℝ)
  (total_pints : ℝ) :
  total_gallons * pints_per_gallon = total_pints ∧
  pints_drunk_by_genevieve / thermoses_drunk_by_genevieve = 2 →
  total_pints / 2 = 18 :=
by
  intros h
  have := h.2
  sorry

end thermos_count_l307_307079


namespace num_three_digit_integers_sum_to_seven_l307_307581

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l307_307581


namespace insufficient_pharmacies_l307_307533

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l307_307533


namespace cyclist_arrives_first_l307_307732

-- Definitions based on given conditions
def speed_cyclist (v : ℕ) := v
def speed_motorist (v : ℕ) := 5 * v

def distance_total (d : ℕ) := d
def distance_half (d : ℕ) := d / 2

def time_motorist_first_half (d v : ℕ) : ℕ := distance_half d / speed_motorist v

def remaining_distance_cyclist (d v : ℕ) := d - v * time_motorist_first_half d v

def speed_motorist_walking (v : ℕ) := v / 2

def time_motorist_second_half (d v : ℕ) := distance_half d / speed_motorist_walking v
def time_cyclist_remaining (d v : ℕ) : ℕ := remaining_distance_cyclist d v / speed_cyclist v

-- Comparison to prove cyclist arrives first
theorem cyclist_arrives_first (d v : ℕ) (hv : 0 < v) (hd : 0 < d) :
  time_cyclist_remaining d v < time_motorist_second_half d v :=
by sorry

end cyclist_arrives_first_l307_307732


namespace sufficient_but_not_necessary_l307_307794

theorem sufficient_but_not_necessary (a b : ℝ) : (a > b ∧ b > 0) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > b ∧ b > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l307_307794


namespace larrys_correct_substitution_l307_307337

noncomputable def lucky_larry_expression (a b c d e f : ℤ) : ℤ :=
  a + (b - (c + (d - (e + f))))

noncomputable def larrys_substitution (a b c d e f : ℤ) : ℤ :=
  a + b - c + d - e + f

theorem larrys_correct_substitution : 
  (lucky_larry_expression 2 4 6 8 e 5 = larrys_substitution 2 4 6 8 e 5) ↔ (e = 8) :=
by
  sorry

end larrys_correct_substitution_l307_307337


namespace probability_of_region_C_zero_l307_307917

theorem probability_of_region_C_zero :
  let p_A := 1/5
      p_B := 1/3
      x := (7/30 : ℚ)
      p_C := (0 : ℚ)
  in p_A + p_B + p_C + x + x = 1 :=
by
  let p_A := (1/5 : ℚ)
  let p_B := (1/3 : ℚ)
  let x := (7/30 : ℚ)
  let p_C := (0 : ℚ)
  have h : p_A + p_B + p_C + x + x = 1 :=
    by sorry
  exact h

end probability_of_region_C_zero_l307_307917


namespace frances_towel_weight_in_ounces_l307_307343

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end frances_towel_weight_in_ounces_l307_307343


namespace total_cost_l307_307818

def permit_cost : Int := 250
def contractor_hourly_rate : Int := 150
def contractor_days : Int := 3
def contractor_hours_per_day : Int := 5
def inspector_discount_rate : Float := 0.80

theorem total_cost : Int :=
  let total_hours := contractor_days * contractor_hours_per_day
  let contractor_total_cost := total_hours * contractor_hourly_rate
  let inspector_hourly_rate := contractor_hourly_rate - (inspector_discount_rate * contractor_hourly_rate)
  let inspector_total_cost := total_hours * Int.ofFloat inspector_hourly_rate
  permit_cost + contractor_total_cost + inspector_total_cost

example : total_cost = 2950 := by
  sorry

end total_cost_l307_307818


namespace find_value_of_expression_l307_307805

theorem find_value_of_expression (m n : ℝ) (h : |m - n - 5| + (2 * m + n - 4)^2 = 0) : 3 * m + n = 7 := 
sorry

end find_value_of_expression_l307_307805


namespace calculate_expression_l307_307747

theorem calculate_expression : 
  (π - 3.14) ^ 0 - 8 ^ (2 / 3) + (1 / 5) ^ 2 * (Real.logb 2 32) + 5 ^ (Real.logb 5 3) = 1 / 5 :=
by
  sorry

end calculate_expression_l307_307747


namespace twice_not_square_l307_307502

theorem twice_not_square (m : ℝ) : 2 * m ≠ m * m := by
  sorry

end twice_not_square_l307_307502


namespace treaty_signed_on_saturday_l307_307866

-- Define the start day and the total days until the treaty.
def start_day_of_week : Nat := 4 -- Thursday is the 4th day (0 = Sunday, ..., 6 = Saturday)
def days_until_treaty : Nat := 919

-- Calculate the final day of the week after 919 days since start_day_of_week.
def treaty_day_of_week : Nat := (start_day_of_week + days_until_treaty) % 7

-- The goal is to prove that the treaty was signed on a Saturday.
theorem treaty_signed_on_saturday : treaty_day_of_week = 6 :=
by
  -- Implement the proof steps
  sorry

end treaty_signed_on_saturday_l307_307866


namespace smallest_y_not_defined_l307_307231

theorem smallest_y_not_defined : 
  ∃ y : ℝ, (6 * y^2 - 37 * y + 6 = 0) ∧ (∀ z : ℝ, (6 * z^2 - 37 * z + 6 = 0) → y ≤ z) ∧ y = 1 / 6 :=
by
  sorry

end smallest_y_not_defined_l307_307231


namespace johns_speed_l307_307164

def time1 : ℕ := 2
def time2 : ℕ := 3
def total_distance : ℕ := 225

def total_time : ℕ := time1 + time2

theorem johns_speed :
  (total_distance : ℝ) / (total_time : ℝ) = 45 :=
sorry

end johns_speed_l307_307164


namespace milk_left_after_third_operation_l307_307941

theorem milk_left_after_third_operation :
  ∀ (initial_milk : ℝ), initial_milk > 0 →
  (initial_milk * 0.8 * 0.8 * 0.8 / initial_milk) * 100 = 51.2 :=
by
  intros initial_milk h_initial_milk_pos
  sorry

end milk_left_after_third_operation_l307_307941


namespace find_acute_angle_l307_307126

theorem find_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < 90) (h2 : ∃ k : ℤ, 10 * α = α + k * 360) :
  α = 40 ∨ α = 80 :=
by
  sorry

end find_acute_angle_l307_307126


namespace total_players_on_ground_l307_307973

theorem total_players_on_ground :
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  cricket_players + hockey_players + football_players + softball_players +
  basketball_players + volleyball_players + netball_players + rugby_players = 263 := 
by 
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  sorry

end total_players_on_ground_l307_307973


namespace no_solution_inequalities_l307_307668

theorem no_solution_inequalities (m : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3) → (x > m) → false) ↔ (m ≥ 2) :=
by 
  sorry

end no_solution_inequalities_l307_307668


namespace alpha_squared_plus_3alpha_plus_beta_equals_2023_l307_307791

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end alpha_squared_plus_3alpha_plus_beta_equals_2023_l307_307791


namespace green_more_than_blue_l307_307891

variable (B Y G : ℕ)

theorem green_more_than_blue
  (h_sum : B + Y + G = 126)
  (h_ratio : ∃ k : ℕ, B = 3 * k ∧ Y = 7 * k ∧ G = 8 * k) :
  G - B = 35 := by
  sorry

end green_more_than_blue_l307_307891


namespace tens_digit_23_pow_1987_l307_307272

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l307_307272


namespace number_of_three_digit_numbers_with_sum_7_l307_307572

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l307_307572


namespace lenny_initial_money_l307_307166

-- Definitions based on the conditions
def spent_on_video_games : ℕ := 24
def spent_at_grocery_store : ℕ := 21
def amount_left : ℕ := 39

-- Statement of the problem
theorem lenny_initial_money : spent_on_video_games + spent_at_grocery_store + amount_left = 84 :=
by
  sorry

end lenny_initial_money_l307_307166


namespace overall_gain_is_10_percent_l307_307354

noncomputable def total_cost_price : ℝ := 700 + 500 + 300
noncomputable def total_gain : ℝ := 70 + 50 + 30
noncomputable def overall_gain_percentage : ℝ := (total_gain / total_cost_price) * 100

theorem overall_gain_is_10_percent :
  overall_gain_percentage = 10 :=
by
  sorry

end overall_gain_is_10_percent_l307_307354


namespace sum_areas_of_circles_l307_307376

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l307_307376


namespace rectangle_width_length_ratio_l307_307156

theorem rectangle_width_length_ratio (w l : ℕ) 
  (h1 : l = 12) 
  (h2 : 2 * w + 2 * l = 36) : 
  w / l = 1 / 2 := 
by 
  sorry

end rectangle_width_length_ratio_l307_307156


namespace isosceles_triangle_sides_l307_307920

theorem isosceles_triangle_sides (a b c : ℝ) (hb : b = 3) (hc : a = 3 ∨ c = 3) (hperim : a + b + c = 7) :
  a = 2 ∨ a = 3 ∨ c = 2 ∨ c = 3 :=
by
  sorry

end isosceles_triangle_sides_l307_307920


namespace identity_n1_n2_product_l307_307470

theorem identity_n1_n2_product :
  (∃ (N1 N2 : ℤ),
    (∀ x : ℚ, (35 * x - 29) / (x^2 - 3 * x + 2) = N1 / (x - 1) + N2 / (x - 2)) ∧
    N1 * N2 = -246) :=
sorry

end identity_n1_n2_product_l307_307470


namespace RelativelyPrimeProbability_l307_307883

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l307_307883


namespace percentage_of_profit_without_discount_l307_307235

-- Definitions for the conditions
def cost_price : ℝ := 100
def discount_rate : ℝ := 0.04
def profit_rate : ℝ := 0.32

-- The statement to prove
theorem percentage_of_profit_without_discount :
  let selling_price := cost_price + (profit_rate * cost_price)
  (selling_price - cost_price) / cost_price * 100 = 32 := by
  let selling_price := cost_price + (profit_rate * cost_price)
  sorry

end percentage_of_profit_without_discount_l307_307235


namespace num_three_digit_integers_sum_to_seven_l307_307584

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l307_307584


namespace extremum_of_f_l307_307285

def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

theorem extremum_of_f :
  ∃ (xmin xmax : ℝ) (x1 y1 : ℝ), f x1 y1 = xmin ∧ (x1 = 11 ∧ y1 = 2) ∧
  ∃ (xmax : ℝ) (x2 y2 : ℝ), f x2 y2 = xmax ∧ (x2 = 1 ∧ y2 = 4) ∧
  xmin = 10 ∧ xmax = 570 := 
by
  sorry

end extremum_of_f_l307_307285


namespace tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l307_307133

noncomputable def f (a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

noncomputable def f_prime (a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_eq_a1 (b : ℝ) (h : f_prime 1 b (-1) = 0) : 
  ∃ m q, m = 1 ∧ q = 1 ∧ ∀ y, y = f 1 b 0 + m * y := sorry

theorem max_value_f_a_gt_1_div_5 (a b : ℝ) 
  (h_gt : a > 1/5) 
  (h_fp_eq : f_prime a b (-1) = 0)
  (h_max : ∀ x, -1 ≤ x ∧ x ≤ 1 → f a b x ≤ 4 * Real.exp 1) : 
  a = (24 * Real.exp 2 - 9) / 15 ∧ b = (12 * Real.exp 2 - 2) / 5 := sorry

end tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l307_307133


namespace sum_of_three_consecutive_integers_l307_307707

theorem sum_of_three_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 :=
sorry

end sum_of_three_consecutive_integers_l307_307707


namespace solve_for_y_l307_307861

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end solve_for_y_l307_307861


namespace rationalization_correct_l307_307031

noncomputable def rationalize_denominator (a b : ℝ) : ℝ :=
  a / (b + 1)

theorem rationalization_correct :
  rationalize_denominator 1 (sqrt 3 - 1) = (sqrt 3 + 1) / 2 :=
by
  sorry

end rationalization_correct_l307_307031


namespace all_lucky_years_l307_307248

def is_lucky_year (y : ℕ) : Prop :=
  ∃ m d : ℕ, 1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 31 ∧ (m * d = y % 100)

theorem all_lucky_years :
  (is_lucky_year 2024) ∧ (is_lucky_year 2025) ∧ (is_lucky_year 2026) ∧ (is_lucky_year 2027) ∧ (is_lucky_year 2028) :=
sorry

end all_lucky_years_l307_307248


namespace find_number_l307_307731

theorem find_number (n x : ℝ) (hx : x = 0.8999999999999999) (h : n / x = 0.01) : n = 0.008999999999999999 := by
  sorry

end find_number_l307_307731


namespace student_correct_answers_l307_307001

noncomputable def correct_answers : ℕ := 58

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = correct_answers :=
by {
  -- placeholder for actual proof
  sorry
}

end student_correct_answers_l307_307001


namespace x_lt_1_iff_x_abs_x_lt_1_l307_307646

theorem x_lt_1_iff_x_abs_x_lt_1 (x : ℝ) : x < 1 ↔ x * |x| < 1 :=
sorry

end x_lt_1_iff_x_abs_x_lt_1_l307_307646


namespace pieces_per_pizza_is_five_l307_307985

-- Definitions based on the conditions
def cost_per_pizza (total_cost : ℕ) (number_of_pizzas : ℕ) : ℕ :=
  total_cost / number_of_pizzas

def number_of_pieces_per_pizza (cost_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  cost_per_pizza / cost_per_piece

-- Given conditions
def total_cost : ℕ := 80
def number_of_pizzas : ℕ := 4
def cost_per_piece : ℕ := 4

-- Prove
theorem pieces_per_pizza_is_five : number_of_pieces_per_pizza (cost_per_pizza total_cost number_of_pizzas) cost_per_piece = 5 :=
by sorry

end pieces_per_pizza_is_five_l307_307985


namespace P_iff_q_l307_307793

variables (a b c: ℝ)

def P : Prop := a * c < 0
def q : Prop := ∃ α β : ℝ, α * β < 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0

theorem P_iff_q : P a c ↔ q a b c := 
sorry

end P_iff_q_l307_307793


namespace curves_intersect_condition_l307_307888

noncomputable def curves_intersect_exactly_three_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (x^2 + y^2 = a^2) ∧ (y = x^2 + a) ∧ 
    (y = a → x = 0) ∧ 
    ((2 * a + 1 < 0) → y = -(2 * a + 1) - 1)

theorem curves_intersect_condition (a : ℝ) : 
  curves_intersect_exactly_three_points a ↔ a < -1/2 :=
sorry

end curves_intersect_condition_l307_307888


namespace sum_of_areas_of_circles_l307_307381

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l307_307381


namespace box_volume_l307_307097

theorem box_volume (initial_length initial_width cut_length : ℕ)
  (length_condition : initial_length = 13) (width_condition : initial_width = 9)
  (cut_condition : cut_length = 2) : 
  (initial_length - 2 * cut_length) * (initial_width - 2 * cut_length) * cut_length = 90 := 
by
  sorry

end box_volume_l307_307097


namespace sum_areas_of_circles_l307_307375

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l307_307375


namespace chi_squared_test_expected_value_correct_l307_307538
open ProbabilityTheory

section Part1

def n : ℕ := 400
def a : ℕ := 60
def b : ℕ := 20
def c : ℕ := 180
def d : ℕ := 140
def alpha : ℝ := 0.005
def chi_critical : ℝ := 7.879

noncomputable def chi_squared : ℝ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test : chi_squared > chi_critical :=
  sorry

end Part1

section Part2

def reward_med : ℝ := 6  -- 60,000 yuan in 10,000 yuan unit
def reward_small : ℝ := 2  -- 20,000 yuan in 10,000 yuan unit
def total_support : ℕ := 12
def total_rewards : ℕ := 9

noncomputable def dist_table : List (ℝ × ℝ) :=
  [(180, 1 / 220),
   (220, 27 / 220),
   (260, 27 / 55),
   (300, 21 / 55)]

noncomputable def expected_value : ℝ :=
  dist_table.foldr (fun (xi : ℝ × ℝ) acc => acc + xi.1 * xi.2) 0

theorem expected_value_correct : expected_value = 270 :=
  sorry

end Part2

end chi_squared_test_expected_value_correct_l307_307538


namespace digits_sum_eq_seven_l307_307624

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l307_307624


namespace sum_first_2017_terms_l307_307651

theorem sum_first_2017_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → S (n + 1) - S n = 3^n / a n) :
  S 2017 = 3^1009 - 2 := sorry

end sum_first_2017_terms_l307_307651


namespace yard_flower_beds_fraction_l307_307251

theorem yard_flower_beds_fraction :
  let yard_length := 30
  let yard_width := 10
  let pool_length := 10
  let pool_width := 4
  let trap_parallel_diff := 22 - 16
  let triangle_leg := trap_parallel_diff / 2
  let triangle_area := (1 / 2) * (triangle_leg ^ 2)
  let total_triangle_area := 2 * triangle_area
  let total_yard_area := yard_length * yard_width
  let pool_area := pool_length * pool_width
  let usable_yard_area := total_yard_area - pool_area
  (total_triangle_area / usable_yard_area) = 9 / 260 :=
by 
  sorry

end yard_flower_beds_fraction_l307_307251


namespace incorrect_reasoning_form_l307_307362

-- Define what it means to be a rational number
def is_rational (x : ℚ) : Prop := true

-- Define what it means to be a fraction
def is_fraction (x : ℚ) : Prop := true

-- Define what it means to be an integer
def is_integer (x : ℤ) : Prop := true

-- State the premises as hypotheses
theorem incorrect_reasoning_form (h1 : ∃ x : ℚ, is_rational x ∧ is_fraction x)
                                 (h2 : ∀ z : ℤ, is_rational z) :
  ¬ (∀ z : ℤ, is_fraction z) :=
by
  -- We are stating the conclusion as a hypothesis that needs to be proven incorrect
  sorry

end incorrect_reasoning_form_l307_307362


namespace schur_theorem_l307_307895

theorem schur_theorem {n : ℕ} (P : Fin n → Set ℕ) (h_partition : ∀ x : ℕ, ∃ i : Fin n, x ∈ P i) :
  ∃ (i : Fin n) (x y : ℕ), x ∈ P i ∧ y ∈ P i ∧ x + y ∈ P i :=
sorry

end schur_theorem_l307_307895


namespace right_triangle_legs_l307_307650

theorem right_triangle_legs (a b : ℕ) (h : a^2 + b^2 = 100) (h_r: a + b - 10 = 4) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
sorry

end right_triangle_legs_l307_307650


namespace rational_root_uniqueness_l307_307693

theorem rational_root_uniqueness (c : ℚ) :
  ∀ x1 x2 : ℚ, (x1 ≠ x2) →
  (x1^3 - 3 * c * x1^2 - 3 * x1 + c = 0) →
  (x2^3 - 3 * c * x2^2 - 3 * x2 + c = 0) →
  false := 
by
  intros x1 x2 h1 h2 h3
  sorry

end rational_root_uniqueness_l307_307693


namespace shortest_tree_height_is_correct_l307_307052

-- Definitions of the tree heights
def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

-- Theorem statement
theorem shortest_tree_height_is_correct :
  shortest_tree_height = 50 :=
by
  sorry

end shortest_tree_height_is_correct_l307_307052


namespace part_a_part_b_part_c_l307_307069

-- Part (a)
theorem part_a : (7 * (2 / 3) + 16 * (5 / 12)) = (34 / 3) :=
by
  sorry

-- Part (b)
theorem part_b : (5 - (2 / (5 / 3))) = (19 / 5) :=
by
  sorry

-- Part (c)
theorem part_c : (1 + (2 / (1 + (3 / (1 + 4))))) = (9 / 4) :=
by
  sorry

end part_a_part_b_part_c_l307_307069


namespace units_digit_2019_pow_2019_l307_307896

theorem units_digit_2019_pow_2019 : (2019^2019) % 10 = 9 := 
by {
  -- The statement of the problem is proved below
  sorry  -- Solution to be filled in
}

end units_digit_2019_pow_2019_l307_307896


namespace Y_tagged_value_l307_307262

variables (W X Y Z : ℕ)
variables (tag_W : W = 200)
variables (tag_X : X = W / 2)
variables (tag_Z : Z = 400)
variables (total : W + X + Y + Z = 1000)

theorem Y_tagged_value : Y = 300 :=
by sorry

end Y_tagged_value_l307_307262


namespace find_g_of_polynomial_l307_307758

variable (x : ℝ)

theorem find_g_of_polynomial :
  ∃ g : ℝ → ℝ, (4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) → (g x = -2 * x^4 - 13 * x^3 + 7 * x + 4) :=
sorry

end find_g_of_polynomial_l307_307758


namespace range_of_a_l307_307297

def satisfies_p (x : ℝ) : Prop := (2 * x - 1) / (x - 1) ≤ 0

def satisfies_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) < 0

def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  ∀ x, p x → q x ∧ ∃ x, q x ∧ ¬(p x)

theorem range_of_a :
  (∀ (x a : ℝ), satisfies_p x → satisfies_q x a → 0 ≤ a ∧ a < 1 / 2) ↔ (∀ a, 0 ≤ a ∧ a < 1 / 2) := by sorry

end range_of_a_l307_307297


namespace find_a_l307_307472

theorem find_a (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) 
    (h3 : a ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
    (h4 : a ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
    a = 253 :=
by 
  sorry

end find_a_l307_307472


namespace andrea_rhinestones_ratio_l307_307419

theorem andrea_rhinestones_ratio :
  (∃ (B : ℕ), B = 45 - (1 / 5 * 45) - 21) →
  (1/5 * 45 : ℕ) + B + 21 = 45 →
  (B : ℕ) / 45 = 1 / 3 := 
sorry

end andrea_rhinestones_ratio_l307_307419


namespace three_digit_sum_seven_l307_307602

theorem three_digit_sum_seven : ∃ (n : ℕ), n = 28 ∧ 
  ∃ (a b c : ℕ), 100 * a + 10 * b + c < 1000 ∧ a + b + c = 7 ∧ a ≥ 1 := 
by
  sorry

end three_digit_sum_seven_l307_307602


namespace carol_first_six_l307_307545

-- A formalization of the probabilities involved when Alice, Bob, Carol,
-- and Dave take turns rolling a die, and the process repeats.
def probability_carol_first_six (prob_rolling_six : ℚ) : ℚ := sorry

theorem carol_first_six (prob_rolling_six : ℚ) (h : prob_rolling_six = 1/6) :
  probability_carol_first_six prob_rolling_six = 25 / 91 :=
sorry

end carol_first_six_l307_307545


namespace max_value_of_y_l307_307761

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (π / 4 + x) - sin (π / 4 - x)) * sin (π / 3 + x)

theorem max_value_of_y : 
  ∃ x : ℝ, (∀ x, y x ≤ 3 * sqrt 2 / 4) ∧ (∀ k : ℤ, x = k * π + π / 3 → y x = 3 * sqrt 2 / 4) :=
sorry

end max_value_of_y_l307_307761


namespace three_digit_integers_sum_to_7_l307_307590

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l307_307590


namespace inverse_proportion_decreases_l307_307692

theorem inverse_proportion_decreases {x : ℝ} (h : x > 0 ∨ x < 0) : 
  y = 3 / x → ∀ (x1 x2 : ℝ), (x1 > 0 ∨ x1 < 0) → (x2 > 0 ∨ x2 < 0) → x1 < x2 → (3 / x1) > (3 / x2) := 
by
  sorry

end inverse_proportion_decreases_l307_307692


namespace frances_towel_weight_in_ounces_l307_307342

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end frances_towel_weight_in_ounces_l307_307342


namespace find_angle_ACB_l307_307772

theorem find_angle_ACB
    (convex_quadrilateral : Prop)
    (angle_BAC : ℝ)
    (angle_CAD : ℝ)
    (angle_ADB : ℝ)
    (angle_BDC : ℝ)
    (h1 : convex_quadrilateral)
    (h2 : angle_BAC = 20)
    (h3 : angle_CAD = 60)
    (h4 : angle_ADB = 50)
    (h5 : angle_BDC = 10)
    : ∃ angle_ACB : ℝ, angle_ACB = 80 :=
by
  -- Here use sorry to skip the proof.
  sorry

end find_angle_ACB_l307_307772


namespace can_cross_all_rivers_and_extra_material_l307_307087

-- Definitions for river widths, bridge length, and additional material.
def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def bridge_length : ℕ := 295
def additional_material : ℕ := 1020

-- Calculations for material needed for each river.
def material_needed_for_river1 : ℕ := river1_width - bridge_length
def material_needed_for_river2 : ℕ := river2_width - bridge_length
def material_needed_for_river3 : ℕ := river3_width - bridge_length

-- Total material needed to cross all three rivers.
def total_material_needed : ℕ := material_needed_for_river1 + material_needed_for_river2 + material_needed_for_river3

-- The main theorem statement to prove.
theorem can_cross_all_rivers_and_extra_material :
  total_material_needed <= additional_material ∧ (additional_material - total_material_needed = 421) := 
by 
  sorry

end can_cross_all_rivers_and_extra_material_l307_307087


namespace count_of_good_numbers_l307_307209

-- Define the conditions of the problem
def is_good_number (n : ℕ) : Prop :=
  2020 % n = 22

-- The main statement proving the number of 'good' numbers
theorem count_of_good_numbers :
  (finset.filter is_good_number (finset.Icc 1 2020)).card = 10 :=
by
  sorry

end count_of_good_numbers_l307_307209


namespace find_vertex_angle_of_cone_l307_307182

noncomputable def vertexAngleCone (r1 r2 : ℝ) (O1 O2 : ℝ) (touching : Prop) (Ctable : Prop) (equalAngles : Prop) : Prop :=
  -- The given conditions:
  -- r1, r2 are the radii of the spheres, where r1 = 4 and r2 = 1.
  -- O1, O2 are the centers of the spheres.
  -- touching indicates the spheres touch externally.
  -- Ctable indicates that vertex C of the cone is on the segment connecting the points where the spheres touch the table.
  -- equalAngles indicates that the rays CO1 and CO2 form equal angles with the table.
  touching → 
  Ctable → 
  equalAngles →
  -- The target to prove:
  ∃ α : ℝ, 2 * α = 2 * Real.arctan (2 / 5)

theorem find_vertex_angle_of_cone (r1 r2 : ℝ) (O1 O2 : ℝ) :
  let touching : Prop := (r1 = 4 ∧ r2 = 1 ∧ abs (O1 - O2) = r1 + r2)
  let Ctable : Prop := (True)  -- Provided by problem conditions, details can be expanded
  let equalAngles : Prop := (True)  
  vertexAngleCone r1 r2 O1 O2 touching Ctable equalAngles := 
by
  sorry

end find_vertex_angle_of_cone_l307_307182


namespace find_a9_l307_307659

variable {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a3_eq_1 (a : ℕ → ℝ) : Prop := 
  a 3 = 1

def a5_a6_a7_eq_8 (a : ℕ → ℝ) : Prop := 
  a 5 * a 6 * a 7 = 8

-- Theorem to prove
theorem find_a9 {a : ℕ → ℝ} {q : ℝ} 
  (geom : geom_seq a q)
  (ha3 : a3_eq_1 a)
  (ha5a6a7 : a5_a6_a7_eq_8 a) : a 9 = 4 := 
sorry

end find_a9_l307_307659


namespace minimum_value_of_expression_l307_307655

theorem minimum_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 := 
sorry

end minimum_value_of_expression_l307_307655


namespace probability_sum_odd_l307_307505

theorem probability_sum_odd (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) 
  (h_even : ∃ z : ℕ, z % 2 = 0 ∧ z > 0) 
  (h_odd : ∃ z : ℕ, z % 2 = 1 ∧ z > 0) : 
  (∃ p : ℝ, 0 < p ∧ p < 1 ∧ p = 0.5) :=
sorry

end probability_sum_odd_l307_307505


namespace Xiaoli_estimate_is_larger_l307_307190

variables {x y x' y' : ℝ}

theorem Xiaoli_estimate_is_larger (h1 : x > y) (h2 : y > 0) (h3 : x' = 1.01 * x) (h4 : y' = 0.99 * y) : x' - y' > x - y :=
by sorry

end Xiaoli_estimate_is_larger_l307_307190


namespace area_enclosed_l307_307057

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sin (x - Real.pi / 3)
noncomputable def area_between (a b : ℝ) (f g : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, (g x - f x)

theorem area_enclosed (h₀ : 0 ≤ 2 * Real.pi) (h₁ : 2 * Real.pi ≤ 2 * Real.pi) :
  area_between (2 * Real.pi / 3) (5 * Real.pi / 3) f g = 2 :=
by 
  sorry

end area_enclosed_l307_307057


namespace container_volume_ratio_l307_307411

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end container_volume_ratio_l307_307411


namespace mean_integer_board_l307_307942

theorem mean_integer_board (n : ℕ) (nums : Fin n → ℕ)
  (h : ∀ (i j : Fin n), i ≠ j →
    (∃ a ∈ ℤ, a = (nums i + nums j) / 2) ∨
    (∃ b ∈ ℤ, b = Real.sqrt ((nums i : ℝ) * (nums j : ℝ)))) :
  ∃ board : Fin n → ℤ, 
    (∀ i j : Fin n, i ≠ j → 
      (board = λ k, (nums k + nums j) / 2) ∨ 
      (board = λ k, Real.sqrt ((nums k : ℝ) * (nums j : ℝ)).toInt)) :=
sorry

end mean_integer_board_l307_307942


namespace numBills_is_9_l307_307344

-- Define the conditions: Mike has 45 dollars in 5-dollar bills
def totalDollars : ℕ := 45
def billValue : ℕ := 5
def numBills : ℕ := 9

-- Prove that the number of 5-dollar bills Mike has is 9
theorem numBills_is_9 : (totalDollars = billValue * numBills) → (numBills = 9) :=
by
  intro h
  sorry

end numBills_is_9_l307_307344


namespace find_a_l307_307471

theorem find_a (a b c : ℕ) (h1 : a ≥ b ∧ b ≥ c)  
  (h2 : (a:ℤ) ^ 2 - b ^ 2 - c ^ 2 + a * b = 2011) 
  (h3 : (a:ℤ) ^ 2 + 3 * b ^ 2 + 3 * c ^ 2 - 3 * a * b - 2 * a * c - 2 * b * c = -1997) : 
a = 253 := 
sorry

end find_a_l307_307471


namespace person_B_work_days_l307_307511

theorem person_B_work_days :
  ∃ x : ℝ, (1 / 30 + 1 / x) * 2 = 1 / 9 ∧ x = 45 := by
  -- Declare the conditions
  let workRateA := (1:ℝ) / 30
  let portionTogether := (1:ℝ) / 9
  -- Assertion
  use 45
  have workRateB := (1:ℝ) / 45
  -- Show the combined rate completes 1/9 of the work in 2 days
  have combined_rate := workRateA + workRateB
  have two_day_work := combined_rate * 2
  -- Conclude
  exact ⟨two_day_work = portionTogether, workRateB = (1:ℝ) / 45⟩

end person_B_work_days_l307_307511


namespace number_of_three_digit_numbers_with_digit_sum_seven_l307_307632

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l307_307632


namespace three_digit_sum_seven_l307_307616

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l307_307616


namespace sum_areas_of_circles_l307_307374

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end sum_areas_of_circles_l307_307374


namespace joe_eggs_club_house_l307_307163

theorem joe_eggs_club_house (C : ℕ) (h : C + 5 + 3 = 20) : C = 12 :=
by 
  sorry

end joe_eggs_club_house_l307_307163


namespace sum_areas_of_tangent_circles_l307_307706

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end sum_areas_of_tangent_circles_l307_307706


namespace farmer_revenue_correct_l307_307733

-- Define the conditions
def average_bacon : ℕ := 20
def price_per_pound : ℕ := 6
def size_factor : ℕ := 1 / 2

-- Calculate the bacon from the runt pig
def bacon_from_runt := average_bacon * size_factor

-- Calculate the revenue from selling the bacon
def revenue := bacon_from_runt * price_per_pound

-- Lean 4 Statement to prove
theorem farmer_revenue_correct :
  revenue = 60 :=
sorry

end farmer_revenue_correct_l307_307733


namespace sum_of_areas_of_circles_l307_307380

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l307_307380


namespace remainder_415_pow_420_div_16_l307_307762

theorem remainder_415_pow_420_div_16 : 415^420 % 16 = 1 := by
  sorry

end remainder_415_pow_420_div_16_l307_307762


namespace range_of_m_l307_307765

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 2) * x + m - 1 → (x ≥ 0 ∨ y ≥ 0))) ↔ (1 ≤ m ∧ m < 2) :=
by sorry

end range_of_m_l307_307765


namespace emma_age_proof_l307_307814

theorem emma_age_proof (Inez Zack Jose Emma : ℕ)
  (hJose : Jose = 20)
  (hZack : Zack = Jose + 4)
  (hInez : Inez = Zack - 12)
  (hEmma : Emma = Jose + 5) :
  Emma = 25 :=
by
  sorry

end emma_age_proof_l307_307814


namespace three_digit_sum_seven_l307_307618

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l307_307618


namespace correct_options_l307_307140

theorem correct_options (a b : ℝ) (h_a_pos : a > 0) (h_discriminant : a^2 = 4 * b):
  (a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬(∃ x1 x2 : ℝ, (x1 * x2 > 0 ∧ a^2 - x1x2 ≠ 4b))) ∧ 
  (∀ c x1 x2 : ℝ, (x1 - x2 = 4) → (a^2 - 4 * (b - c) = 16) → (c = 4)) :=
by
  sorry

end correct_options_l307_307140


namespace area_shaded_region_is_correct_l307_307003

noncomputable def radius_of_larger_circle : ℝ := 8
noncomputable def radius_of_smaller_circle := radius_of_larger_circle / 2

-- Define areas
noncomputable def area_of_larger_circle := Real.pi * radius_of_larger_circle ^ 2
noncomputable def area_of_smaller_circle := Real.pi * radius_of_smaller_circle ^ 2
noncomputable def total_area_of_smaller_circles := 2 * area_of_smaller_circle
noncomputable def area_of_shaded_region := area_of_larger_circle - total_area_of_smaller_circles

-- Prove that the area of the shaded region is 32π
theorem area_shaded_region_is_correct : area_of_shaded_region = 32 * Real.pi := by
  sorry

end area_shaded_region_is_correct_l307_307003


namespace three_digit_sum_7_l307_307613

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l307_307613


namespace daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l307_307901

-- Definitions based on given conditions
noncomputable def purchase_price : ℝ := 30
noncomputable def max_selling_price : ℝ := 55
noncomputable def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 140

-- Definition of daily profit based on selling price x
noncomputable def daily_profit (x : ℝ) : ℝ := (x - purchase_price) * daily_sales_volume x

-- Lean 4 statements for the proofs
theorem daily_profit_at_35_yuan : daily_profit 35 = 350 := sorry

theorem selling_price_for_600_profit : ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ daily_profit x = 600 ∧ x = 40 := sorry

theorem selling_price_impossible_for_900_profit :
  ∀ x, 30 ≤ x ∧ x ≤ 55 → daily_profit x ≠ 900 := sorry

end daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l307_307901


namespace worst_is_father_l307_307669

-- Definitions for players
inductive Player
| father
| sister
| daughter
| son
deriving DecidableEq

open Player

def opposite_sex (p1 p2 : Player) : Bool :=
match p1, p2 with
| father, sister => true
| father, daughter => true
| sister, father => true
| daughter, father => true
| son, sister => true
| son, daughter => true
| daughter, son => true
| sister, son => true
| _, _ => false 

-- Problem conditions
variables (worst best : Player)
variable (twins : Player → Player)
variable (worst_best_twins : twins worst = best)
variable (worst_twin_conditions : opposite_sex (twins worst) best)

-- Goal: Prove that the worst player is the father
theorem worst_is_father : worst = Player.father := by
  sorry

end worst_is_father_l307_307669


namespace sum_of_areas_of_circles_l307_307372

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l307_307372


namespace arithmetic_sequence_sum_six_l307_307676

open Nat

noncomputable def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  3 * (2 * a1 + 5 * d) / 3

theorem arithmetic_sequence_sum_six (a : ℕ → ℚ) (h : a 2 + a 5 = 2 / 3) : sum_first_six_terms a = 2 :=
by
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  have eq1 : a 5 = a1 + 4 * d := by sorry
  have eq2 : 3 * (2 * a1 + 5 * d) / 3 = (2 : ℚ) := by sorry
  sorry

end arithmetic_sequence_sum_six_l307_307676


namespace VasyaSlowerWalkingFullWayHome_l307_307289

namespace FishingTrip

-- Define the variables involved
variables (x v S : ℝ)   -- x is the speed of Vasya and Petya, v is the speed of Kolya on the bicycle, S is the distance from the house to the lake

-- Conditions derived from the problem statement:
-- Condition 1: When Kolya meets Vasya then Petya starts
-- Condition 2: Given: Petya’s travel time is \( \frac{5}{4} \times \) Vasya's travel time.

theorem VasyaSlowerWalkingFullWayHome (h1 : v = 3 * x) :
  2 * (S / x + v) = (5 / 2) * (S / x) :=
sorry

end FishingTrip

end VasyaSlowerWalkingFullWayHome_l307_307289


namespace rationalize_denominator_l307_307035

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l307_307035


namespace min_value_expression_l307_307801

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    (x + 1 / y) ^ 2 + (y + 1 / (2 * x)) ^ 2 ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end min_value_expression_l307_307801


namespace volume_of_pyramid_in_cube_l307_307726

structure Cube :=
(side_length : ℝ)

noncomputable def base_triangle_area (side_length : ℝ) : ℝ :=
(1/2) * side_length * side_length

noncomputable def pyramid_volume (triangle_area : ℝ) (height : ℝ) : ℝ :=
(1/3) * triangle_area * height

theorem volume_of_pyramid_in_cube (c : Cube) (h : c.side_length = 2) : 
  pyramid_volume (base_triangle_area c.side_length) c.side_length = 4/3 :=
by {
  sorry
}

end volume_of_pyramid_in_cube_l307_307726


namespace equal_popularity_l307_307922

theorem equal_popularity :
  let drama := (8 : ℚ) / 24
  let sports := (9 : ℚ) / 27
  let art := (10 : ℚ) / 30
  let music := (7 : ℚ) / 21
  drama = sports ∧ sports = art ∧ art = music :=
by
  let drama := (8 : ℚ) / 24
  let sports := (9 : ℚ) / 27
  let art := (10 : ℚ) / 30
  let music := (7 : ℚ) / 21
  -- simplify the fractions
  have h1 : drama = (90 : ℚ) / 270 := by sorry
  have h2 : sports = (90 : ℚ) / 270 := by sorry
  have h3 : art = (90 : ℚ) / 270 := by sorry
  have h4 : music = (90 : ℚ) / 270 := by sorry
  exact ⟨h1.trans h2, h2.trans h3, h3.trans h4⟩

end equal_popularity_l307_307922


namespace Frank_seeds_per_orange_l307_307768

noncomputable def Betty_oranges := 15
noncomputable def Bill_oranges := 12
noncomputable def total_oranges := Betty_oranges + Bill_oranges
noncomputable def Frank_oranges := 3 * total_oranges
noncomputable def oranges_per_tree := 5
noncomputable def Philip_oranges := 810
noncomputable def number_of_trees := Philip_oranges / oranges_per_tree
noncomputable def seeds_per_orange := number_of_trees / Frank_oranges

theorem Frank_seeds_per_orange :
  seeds_per_orange = 2 :=
by
  sorry

end Frank_seeds_per_orange_l307_307768


namespace ellipse_equation_l307_307658

-- Definitions from conditions
def ecc (e : ℝ) := e = Real.sqrt 3 / 2
def parabola_focus (c : ℝ) (a : ℝ) := c = Real.sqrt 3 ∧ a = 2
def b_val (b a c : ℝ) := b = Real.sqrt (a^2 - c^2)

-- Main problem statement
theorem ellipse_equation (e a b c : ℝ) (x y : ℝ) :
  ecc e → parabola_focus c a → b_val b a c → (x^2 + y^2 / 4 = 1) := 
by
  intros h1 h2 h3
  sorry

end ellipse_equation_l307_307658


namespace find_multiplier_l307_307738

theorem find_multiplier (x y : ℝ) (hx : x = 0.42857142857142855) (hx_nonzero : x ≠ 0) (h_eq : (x * y) / 7 = x^2) : y = 3 :=
sorry

end find_multiplier_l307_307738


namespace solve_x_for_equation_l307_307520

theorem solve_x_for_equation :
  ∃ (x : ℚ), 3 * x - 5 = abs (-20 + 6) ∧ x = 19 / 3 :=
by
  sorry

end solve_x_for_equation_l307_307520


namespace smaller_mold_radius_l307_307402

theorem smaller_mold_radius (R : ℝ) (third_volume_sharing : ℝ) (molds_count : ℝ) (r : ℝ) 
  (hR : R = 3) 
  (h_third_volume_sharing : third_volume_sharing = 1/3) 
  (h_molds_count : molds_count = 9) 
  (h_r : (2/3) * Real.pi * r^3 = (2/3) * Real.pi / molds_count) : 
  r = 1 := 
by
  sorry

end smaller_mold_radius_l307_307402


namespace equal_real_roots_of_quadratic_eq_l307_307968

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end equal_real_roots_of_quadratic_eq_l307_307968


namespace max_value_of_expression_l307_307981

theorem max_value_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) :
  (∃ (x : ℝ), x = (ab/(a + b)) + (ac/(a + c)) + (bc/(b + c)) ∧ x = 9/4) :=
  sorry

end max_value_of_expression_l307_307981


namespace num_classes_received_basketballs_l307_307500

theorem num_classes_received_basketballs (total_basketballs left_basketballs : ℕ) 
  (h : total_basketballs = 54) (h_left : left_basketballs = 5) : 
  (total_basketballs - left_basketballs) / 7 = 7 :=
by
  sorry

end num_classes_received_basketballs_l307_307500


namespace more_white_birds_than_grey_l307_307708

def num_grey_birds_in_cage : ℕ := 40
def num_remaining_birds : ℕ := 66

def num_grey_birds_freed : ℕ := num_grey_birds_in_cage / 2
def num_grey_birds_left_in_cage : ℕ := num_grey_birds_in_cage - num_grey_birds_freed
def num_white_birds : ℕ := num_remaining_birds - num_grey_birds_left_in_cage

theorem more_white_birds_than_grey : num_white_birds - num_grey_birds_in_cage = 6 := by
  sorry

end more_white_birds_than_grey_l307_307708


namespace projectile_first_reaches_28_l307_307493

theorem projectile_first_reaches_28 (t : ℝ) (h_eq : ∀ t, -4.9 * t^2 + 23.8 * t = 28) : 
    t = 2 :=
sorry

end projectile_first_reaches_28_l307_307493


namespace maximize_tables_eqn_l307_307878

theorem maximize_tables_eqn :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 12 → 400 * x = 20 * (12 - x) * 4 :=
by
  sorry

end maximize_tables_eqn_l307_307878


namespace color_cube_color_octahedron_l307_307770

theorem color_cube (colors : Fin 6) : ∃ (ways : Nat), ways = 30 :=
  sorry

theorem color_octahedron (colors : Fin 8) : ∃ (ways : Nat), ways = 1680 :=
  sorry

end color_cube_color_octahedron_l307_307770


namespace simplify_expression_l307_307924

theorem simplify_expression :
  (2 * 6 / (12 * 14)) * (3 * 12 * 14 / (2 * 6 * 3)) * 2 = 2 := 
  sorry

end simplify_expression_l307_307924


namespace days_to_cover_half_lake_l307_307155

-- Define the problem conditions in Lean
def doubles_every_day (size: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def takes_25_days_to_cover_lake (size: ℕ → ℝ) (lake_size: ℝ) : Prop :=
  size 25 = lake_size

-- Define the main theorem
theorem days_to_cover_half_lake (size: ℕ → ℝ) (lake_size: ℝ) 
  (h1: doubles_every_day size) (h2: takes_25_days_to_cover_lake size lake_size) : 
  size 24 = lake_size / 2 :=
sorry

end days_to_cover_half_lake_l307_307155


namespace solution_set_f_prime_pos_l307_307958

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

theorem solution_set_f_prime_pos : 
  {x : ℝ | 0 < x ∧ (deriv f x > 0)} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_f_prime_pos_l307_307958


namespace complement_union_l307_307173

namespace SetComplement

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union :
  U \ (A ∪ B) = {1, 2, 6} := by
  sorry

end SetComplement

end complement_union_l307_307173


namespace height_of_table_without_book_l307_307542

-- Define the variables and assumptions
variables (l h w : ℝ) (b : ℝ := 6)

-- State the conditions from the problem
-- Condition 1: l + h - w = 40
-- Condition 2: w + h - l + b = 34

theorem height_of_table_without_book (hlw : l + h - w = 40) (whlb : w + h - l + b = 34) : h = 34 :=
by
  -- Since we are skipping the proof, we put sorry here
  sorry

end height_of_table_without_book_l307_307542


namespace count_good_divisors_l307_307218

/-- We call a natural number \( n \) good if 2020 divided by \( n \) leaves a remainder of 22.
    This means \( n \) must be a divisor of 1998 and \( n > 22 \) -/
theorem count_good_divisors : finset.card (finset.filter (λ m, 22 < m ∧ 1998 % m = 0) (finset.range 2000)) = 10 :=
by
  -- This is where the proof would go
  sorry

end count_good_divisors_l307_307218


namespace perpendicular_lines_condition_l307_307196

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end perpendicular_lines_condition_l307_307196


namespace surface_area_of_figure_l307_307328

theorem surface_area_of_figure 
  (block_surface_area : ℕ) 
  (loss_per_block : ℕ) 
  (number_of_blocks : ℕ) 
  (effective_surface_area : ℕ)
  (total_surface_area : ℕ) 
  (h_block : block_surface_area = 18) 
  (h_loss : loss_per_block = 2) 
  (h_blocks : number_of_blocks = 4) 
  (h_effective : effective_surface_area = block_surface_area - loss_per_block) 
  (h_total : total_surface_area = number_of_blocks * effective_surface_area) : 
  total_surface_area = 64 :=
by
  sorry

end surface_area_of_figure_l307_307328


namespace half_abs_diff_of_squares_l307_307223

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l307_307223


namespace K9_le_89_K9_example_171_l307_307709

section weights_proof

def K (n : ℕ) (P : ℕ) : ℕ := sorry -- Assume the definition of K given by the problem

theorem K9_le_89 : ∀ P, K 9 P ≤ 89 := by
  sorry -- Proof to be filled

def example_weight : ℕ := 171

theorem K9_example_171 : K 9 example_weight = 89 := by
  sorry -- Proof to be filled

end weights_proof

end K9_le_89_K9_example_171_l307_307709


namespace tan_alpha_minus_beta_l307_307449

theorem tan_alpha_minus_beta (α β : ℝ) (hα : Real.tan α = 8) (hβ : Real.tan β = 7) :
  Real.tan (α - β) = 1 / 57 := 
sorry

end tan_alpha_minus_beta_l307_307449


namespace cuboid_volume_is_correct_l307_307700

-- Definition of cuboid edges and volume calculation
def cuboid_volume (a b c : ℕ) : ℕ := a * b * c

-- Given conditions
def edge1 : ℕ := 2
def edge2 : ℕ := 5
def edge3 : ℕ := 3

-- Theorem statement
theorem cuboid_volume_is_correct : cuboid_volume edge1 edge2 edge3 = 30 := 
by sorry

end cuboid_volume_is_correct_l307_307700


namespace frances_towels_weight_in_ounces_l307_307340

theorem frances_towels_weight_in_ounces (Mary_towels Frances_towels : ℕ) (Mary_weight Frances_weight : ℝ) (total_weight : ℝ) :
  Mary_towels = 24 ∧ Mary_towels = 4 * Frances_towels ∧ total_weight = Mary_weight + Frances_weight →
  Frances_weight * 16 = 240 :=
by
  sorry

end frances_towels_weight_in_ounces_l307_307340


namespace percentage_students_below_8_years_l307_307458

theorem percentage_students_below_8_years :
  ∀ (n8 : ℕ) (n_gt8 : ℕ) (n_total : ℕ),
  n8 = 24 →
  n_gt8 = 2 * n8 / 3 →
  n_total = 50 →
  (n_total - (n8 + n_gt8)) * 100 / n_total = 20 :=
by
  intros n8 n_gt8 n_total h1 h2 h3
  sorry

end percentage_students_below_8_years_l307_307458


namespace parabola_area_l307_307441

theorem parabola_area (m p : ℝ) (h1 : p > 0) (h2 : (1:ℝ)^2 = 2 * p * m)
    (h3 : (1/2) * (m + p / 2) = 1/2) : p = 1 :=
  by
    sorry

end parabola_area_l307_307441


namespace A_share_of_annual_gain_l307_307409

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end A_share_of_annual_gain_l307_307409


namespace num_correct_statements_l307_307939

def doubleAbsDiff (a b c d : ℝ) : ℝ :=
  |a - b| - |c - d|

theorem num_correct_statements : 
  (∀ a b c d : ℝ, (a, b, c, d) = (24, 25, 29, 30) → 
    (doubleAbsDiff a b c d = 0) ∨
    (doubleAbsDiff a c b d = 0) ∨
    (doubleAbsDiff a d b c = -0.5) ∨
    (doubleAbsDiff b c a d = 0.5)) → 
  (∀ x : ℝ, x ≥ 2 → 
    doubleAbsDiff (x^2) (2*x) 1 1 = 7 → 
    (x^4 + 2401 / x^4 = 226)) →
  (∀ x : ℝ, x ≥ -2 → 
    (doubleAbsDiff (2*x-5) (3*x-2) (4*x-1) (5*x+3)) ≠ 0) →
  (0 = 0)
:= by
  sorry

end num_correct_statements_l307_307939


namespace problem_statement_l307_307413

def A : Prop := (∀ (x : ℝ), x^2 - 3*x + 2 = 0 → x = 2)
def B : Prop := (∃ (x : ℝ), x^2 - x + 1 < 0)
def C : Prop := (¬(∀ (x : ℝ), x > 2 → x^2 - 3*x + 2 > 0))

theorem problem_statement :
  ¬ (A ∧ ∀ (x : ℝ), (B → (x^2 - x + 1) ≥ 0) ∧ (¬(A) ∧ C)) :=
sorry

end problem_statement_l307_307413


namespace valerie_light_bulbs_deficit_l307_307881

theorem valerie_light_bulbs_deficit :
  let small_price := 8.75
  let medium_price := 11.25
  let large_price := 15.50
  let xsmall_price := 6.10
  let budget := 120
  
  let lamp_A_cost := 2 * small_price
  let lamp_B_cost := 3 * medium_price
  let lamp_C_cost := large_price
  let lamp_D_cost := 4 * xsmall_price
  let lamp_E_cost := 2 * large_price
  let lamp_F_cost := small_price + medium_price

  let total_cost := lamp_A_cost + lamp_B_cost + lamp_C_cost + lamp_D_cost + lamp_E_cost + lamp_F_cost

  total_cost - budget = 22.15 :=
by
  sorry

end valerie_light_bulbs_deficit_l307_307881


namespace companion_sets_count_l307_307451

def companion_set (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (x ≠ 0) → (1 / x) ∈ A

def M : Set ℝ := { -1, 0, 1/2, 2, 3 }

theorem companion_sets_count : 
  ∃ S : Finset (Set ℝ), (∀ A ∈ S, companion_set A) ∧ (∀ A ∈ S, A ⊆ M) ∧ S.card = 3 := 
by
  sorry

end companion_sets_count_l307_307451


namespace find_range_of_a_l307_307172

-- Define the conditions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4 * x + a^2 > 0
def q (a : ℝ) : Prop := a^2 - 5 * a - 6 ≥ 0

-- Define the proposition that one of p or q is true and the other is false
def p_or_q (a : ℝ) : Prop := p a ∨ q a
def not_p_and_q (a : ℝ) : Prop := ¬(p a ∧ q a)

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (2 < a ∧ a < 6) ∨ (-2 ≤ a ∧ a ≤ -1)

-- Theorem statement
theorem find_range_of_a (a : ℝ) : p_or_q a ∧ not_p_and_q a → range_of_a a :=
by
  sorry

end find_range_of_a_l307_307172


namespace rectangle_vertex_area_y_value_l307_307739

theorem rectangle_vertex_area_y_value (y : ℕ) (hy : 0 ≤ y) :
  let A := (0, y)
  let B := (10, y)
  let C := (0, 4)
  let D := (10, 4)
  10 * (y - 4) = 90 → y = 13 :=
by
  sorry

end rectangle_vertex_area_y_value_l307_307739


namespace binomial_coeff_coprime_l307_307016

def binom (a b : ℕ) : ℕ := Nat.factorial a / (Nat.factorial b * Nat.factorial (a - b))

theorem binomial_coeff_coprime (p a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hp : Nat.Prime p) 
  (hbase_p_a : ∀ i, (a / p^i % p) ≥ (b / p^i % p)) 
  : Nat.gcd (binom a b) p = 1 :=
by sorry

end binomial_coeff_coprime_l307_307016


namespace count_good_numbers_l307_307220

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l307_307220


namespace sum_of_areas_of_circles_l307_307378

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l307_307378


namespace half_abs_diff_of_squares_l307_307222

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end half_abs_diff_of_squares_l307_307222


namespace volume_difference_is_867_25_l307_307549

noncomputable def charlie_volume : ℝ :=
  let h_C := 9
  let circumference_C := 7
  let r_C := circumference_C / (2 * Real.pi)
  let v_C := Real.pi * r_C^2 * h_C
  v_C

noncomputable def dana_volume : ℝ :=
  let h_D := 5
  let circumference_D := 10
  let r_D := circumference_D / (2 * Real.pi)
  let v_D := Real.pi * r_D^2 * h_D
  v_D

noncomputable def volume_difference : ℝ :=
  Real.pi * (abs (charlie_volume - dana_volume))

theorem volume_difference_is_867_25 : volume_difference = 867.25 := by
  sorry

end volume_difference_is_867_25_l307_307549


namespace probability_cube_selection_l307_307902

/-- Define the structure related to the problem conditions --/
structure UnitCubes :=
  (total : ℕ := 125)
  (three_painted_faces : ℕ := 1)
  (two_painted_faces : ℕ := 9)
  (one_painted_face : ℕ := 9)
  (no_painted_faces : ℕ := 106)

/-- Calculate the probability of selecting one cube with 3 painted faces and one with no painted faces from 125 total cubes. --/
theorem probability_cube_selection :
  let total_ways := Nat.choose 125 2,
      successful_outcomes := 1 * 106
  in (successful_outcomes : ℝ) / (total_ways : ℝ) = 53 / 3875 := 
by 
  -- Sorry is used to skip the proof
  sorry

end probability_cube_selection_l307_307902


namespace domain_of_function_l307_307492

theorem domain_of_function :
  {x : ℝ | ∀ k : ℤ, 2 * x + (π / 4) ≠ k * π + (π / 2)}
  = {x : ℝ | ∀ k : ℤ, x ≠ (k * π / 2) + (π / 8)} :=
sorry

end domain_of_function_l307_307492


namespace rationalize_sqrt_three_sub_one_l307_307030

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l307_307030


namespace stamps_per_book_type2_eq_15_l307_307998

-- Defining the conditions
def num_books_type1 : ℕ := 4
def stamps_per_book_type1 : ℕ := 10
def num_books_type2 : ℕ := 6
def total_stamps : ℕ := 130

-- Stating the theorem to prove the number of stamps in each book of the second type is 15
theorem stamps_per_book_type2_eq_15 : 
  ∀ (x : ℕ), 
    (num_books_type1 * stamps_per_book_type1 + num_books_type2 * x = total_stamps) → 
    x = 15 :=
by
  sorry

end stamps_per_book_type2_eq_15_l307_307998


namespace arrangement_for_P23_exists_l307_307992

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l307_307992


namespace number_of_good_numbers_l307_307214

theorem number_of_good_numbers :
  let n := 2020 in 
  let remainder := 22 in
  let number := 1998 in
  let divisors := {d ∣ number | d > remainder}.to_list.length = 10 := by sorry

end number_of_good_numbers_l307_307214


namespace three_digit_integers_sum_to_7_l307_307588

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end three_digit_integers_sum_to_7_l307_307588


namespace initial_percentage_of_grape_juice_l307_307960

theorem initial_percentage_of_grape_juice
  (P : ℝ)    -- P is the initial percentage in decimal
  (h₁ : 0 ≤ P ∧ P ≤ 1)    -- P is a valid probability
  (h₂ : 40 * P + 10 = 0.36 * 50):    -- Given condition from the problem
  P = 0.2 := 
sorry

end initial_percentage_of_grape_juice_l307_307960


namespace prob_B_hits_once_prob_hits_with_ABC_l307_307710

section
variable (P_A P_B P_C : ℝ)
variable (hA : P_A = 1 / 2)
variable (hB : P_B = 1 / 3)
variable (hC : P_C = 1 / 4)

-- Part (Ⅰ): Probability of hitting the target exactly once when B shoots twice
theorem prob_B_hits_once : 
  (P_B * (1 - P_B) + (1 - P_B) * P_B) = 4 / 9 := 
by
  rw [hB]
  sorry

-- Part (Ⅱ): Probability of hitting the target when A, B, and C each shoot once
theorem prob_hits_with_ABC :
  (1 - ((1 - P_A) * (1 - P_B) * (1 - P_C))) = 3 / 4 := 
by
  rw [hA, hB, hC]
  sorry

end

end prob_B_hits_once_prob_hits_with_ABC_l307_307710


namespace ratio_four_l307_307908

variable {x y : ℝ}

theorem ratio_four : y = 0.25 * x → x / y = 4 := by
  sorry

end ratio_four_l307_307908


namespace cone_radius_l307_307704

theorem cone_radius (h : ℝ) (V : ℝ) (π : ℝ) (r : ℝ)
    (h_def : h = 21)
    (V_def : V = 2199.114857512855)
    (volume_formula : V = (1/3) * π * r^2 * h) : r = 10 :=
by {
  sorry
}

end cone_radius_l307_307704


namespace value_of_b_l307_307965

variable (a b : ℤ)

theorem value_of_b : a = 105 ∧ a ^ 3 = 21 * 49 * 45 * b → b = 1 := by
  sorry

end value_of_b_l307_307965


namespace grid_divisible_by_rectangles_l307_307767

theorem grid_divisible_by_rectangles (n : ℕ) :
  (∃ m : ℕ, n * n = 7 * m) ↔ (∃ k : ℕ, n = 7 * k ∧ k > 1) :=
by
  sorry

end grid_divisible_by_rectangles_l307_307767


namespace xiaoming_grandfather_age_l307_307234

-- Define the conditions
def age_cond (x : ℕ) : Prop :=
  ((x - 15) / 4 - 6) * 10 = 100

-- State the problem
theorem xiaoming_grandfather_age (x : ℕ) (h : age_cond x) : x = 79 := 
sorry

end xiaoming_grandfather_age_l307_307234


namespace point_B_represent_l307_307184

-- Given conditions
def point_A := -2
def units_moved := 4

-- Lean statement to prove
theorem point_B_represent : 
  ∃ B : ℤ, (B = point_A - units_moved) ∨ (B = point_A + units_moved) := by
    sorry

end point_B_represent_l307_307184


namespace washer_cost_difference_l307_307259

theorem washer_cost_difference (W D : ℝ) 
  (h1 : W + D = 1200) (h2 : D = 490) : W - D = 220 :=
sorry

end washer_cost_difference_l307_307259


namespace sequence_general_term_l307_307316

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2^n) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end sequence_general_term_l307_307316


namespace arithmetic_sequence_a5_value_l307_307295

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) 
  (h1 : a 2 + a 4 = 16) 
  (h2 : a 1 = 1) : 
  a 5 = 15 := 
by 
  sorry

end arithmetic_sequence_a5_value_l307_307295


namespace value_of_a_plus_b_minus_c_l307_307012

def a : ℤ := 1 -- smallest positive integer
def b : ℤ := 0 -- number with the smallest absolute value
def c : ℤ := -1 -- largest negative integer

theorem value_of_a_plus_b_minus_c : a + b - c = 2 := by
  -- skipping the proof
  sorry

end value_of_a_plus_b_minus_c_l307_307012


namespace sin_alpha_eq_63_over_65_l307_307775

open Real

variables {α β : ℝ}

theorem sin_alpha_eq_63_over_65
  (h1 : tan β = 4 / 3)
  (h2 : sin (α + β) = 5 / 13)
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  sin α = 63 / 65 := 
by
  sorry

end sin_alpha_eq_63_over_65_l307_307775


namespace ratio_mets_redsox_l307_307154

theorem ratio_mets_redsox 
    (Y M R : ℕ) 
    (h1 : Y = 3 * (M / 2))
    (h2 : M = 88)
    (h3 : Y + M + R = 330) : 
    M / R = 4 / 5 := 
by 
    sorry

end ratio_mets_redsox_l307_307154


namespace hexagon_diagonals_l307_307454

theorem hexagon_diagonals (n : ℕ) (h : n = 6) : (n * (n - 3)) / 2 = 9 := by
  sorry

end hexagon_diagonals_l307_307454


namespace terminating_decimal_of_7_div_200_l307_307287

theorem terminating_decimal_of_7_div_200 : (7 / 200 : ℝ) = 0.028 := sorry

end terminating_decimal_of_7_div_200_l307_307287


namespace container_volume_ratio_l307_307410

theorem container_volume_ratio
  (A B : ℝ)
  (h : (5 / 6) * A = (3 / 4) * B) :
  (A / B = 9 / 10) :=
sorry

end container_volume_ratio_l307_307410


namespace number_of_valid_m_l307_307816

def is_right_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  (Qx - Px) * (Qx - Px) + (Qy - Py) * (Qy - Py) + (Rx - Qx) * (Rx - Qx) + (Ry - Qy) * (Ry - Qy) ==
  (Px - Rx) * (Px - Rx) + (Py - Ry) * (Py - Ry) + 2 * ((Qx - Px) * (Rx - Qx) + (Qy - Py) * (Ry - Qy))

def legs_parallel_to_axes (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  Px = Qx ∨ Px = Rx ∨ Qx = Rx ∧ Py = Qy ∨ Py = Ry ∨ Qy = Ry

def medians_condition (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  let M_PQ := ((Px + Qx) / 2, (Py + Qy) / 2);
  let M_PR := ((Px + Rx) / 2, (Py + Ry) / 2);
  (M_PQ.2 = 3 * M_PQ.1 + 1) ∧ (M_PR.2 = 2)

theorem number_of_valid_m (a b c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (P := (a, b)) (Q := (a, b+2*c)) (R := (a-2*d, b)) :
  is_right_triangle P Q R →
  legs_parallel_to_axes P Q R →
  medians_condition P Q R →
  ∃ m, m = 1 :=
sorry

end number_of_valid_m_l307_307816


namespace tens_digit_23_pow_1987_l307_307273

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end tens_digit_23_pow_1987_l307_307273


namespace twelve_pharmacies_not_sufficient_l307_307527

-- Define an intersection grid of size 10 x 10 (100 squares).
def city_grid : Type := Fin 10 × Fin 10 

-- Define the distance measure between intersections, assumed as L1 metric for grid paths.
def dist (p q : city_grid) : Nat := (abs (p.1.val - q.1.val) + abs (p.2.val - q.2.val))

-- Define a walking distance pharmacy 
def is_walking_distance (p q : city_grid) : Prop := dist p q ≤ 3

-- State that having 12 pharmacies is not sufficient
theorem twelve_pharmacies_not_sufficient (pharmacies : Fin 12 → city_grid) :
  ¬ (∀ intersection: city_grid, ∃ (p_index : Fin 12), is_walking_distance (pharmacies p_index) intersection) :=
sorry

end twelve_pharmacies_not_sufficient_l307_307527


namespace total_spent_on_computer_l307_307321

def initial_cost_of_pc : ℕ := 1200
def sale_price_old_card : ℕ := 300
def cost_new_card : ℕ := 500

theorem total_spent_on_computer : 
  (initial_cost_of_pc + (cost_new_card - sale_price_old_card)) = 1400 :=
by
  sorry

end total_spent_on_computer_l307_307321


namespace original_price_of_apples_l307_307447

-- Define the conditions and problem
theorem original_price_of_apples 
  (discounted_price : ℝ := 0.60 * original_price)
  (total_cost : ℝ := 30)
  (weight : ℝ := 10) :
  original_price = 5 :=
by
  -- This is the point where the proof steps would go.
  sorry

end original_price_of_apples_l307_307447


namespace number_of_three_digit_numbers_with_digit_sum_seven_l307_307630

theorem number_of_three_digit_numbers_with_digit_sum_seven : 
  ( ∑ a b c in finset.Icc 1 9, if a + b + c = 7 then 1 else 0 ) = 28 := sorry

end number_of_three_digit_numbers_with_digit_sum_seven_l307_307630


namespace frustum_smaller_cone_height_l307_307735

theorem frustum_smaller_cone_height (H frustum_height radius1 radius2 : ℝ) 
  (h : ℝ) (h_eq : h = 30 - 18) : 
  radius1 = 6 → radius2 = 10 → frustum_height = 18 → H = 30 → h = 12 := 
by
  intros
  sorry

end frustum_smaller_cone_height_l307_307735


namespace units_digit_Fermat_5_l307_307423

def Fermat_number (n: ℕ) : ℕ :=
  2 ^ (2 ^ n) + 1

theorem units_digit_Fermat_5 : (Fermat_number 5) % 10 = 7 := by
  sorry

end units_digit_Fermat_5_l307_307423


namespace intersection_of_intervals_l307_307664

theorem intersection_of_intervals (m n x : ℝ) (h1 : -1 < m) (h2 : m < 0) (h3 : 0 < n) :
  (m < x ∧ x < n) ∧ (-1 < x ∧ x < 0) ↔ -1 < x ∧ x < 0 :=
by sorry

end intersection_of_intervals_l307_307664


namespace optionD_is_not_linear_system_l307_307392

-- Define the equations for each option
def eqA1 (x y : ℝ) : Prop := 3 * x + 2 * y = 10
def eqA2 (x y : ℝ) : Prop := 2 * x - 3 * y = 5

def eqB1 (x y : ℝ) : Prop := 3 * x + 5 * y = 1
def eqB2 (x y : ℝ) : Prop := 2 * x - y = 4

def eqC1 (x y : ℝ) : Prop := x + 5 * y = 1
def eqC2 (x y : ℝ) : Prop := x - 5 * y = 2

def eqD1 (x y : ℝ) : Prop := x - y = 1
def eqD2 (x y : ℝ) : Prop := y + 1 / x = 3

-- Define the property of a linear equation
def is_linear (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, eq x y → a * x + b * y = c

-- State the theorem
theorem optionD_is_not_linear_system : ¬ (is_linear eqD1 ∧ is_linear eqD2) :=
by
  sorry

end optionD_is_not_linear_system_l307_307392


namespace mario_total_flowers_l307_307175

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

end mario_total_flowers_l307_307175


namespace find_c_l307_307560

noncomputable def func_condition (f : ℝ → ℝ) (c : ℝ) :=
  ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)

theorem find_c :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), func_condition f c → (c = 1 ∨ c = -1) :=
sorry

end find_c_l307_307560


namespace find_q_l307_307360

def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h1 : -p = 2 * (-r)) (h2 : -p = 1 + p + q + r) (hy_intercept : r = 5) : q = -24 :=
by
  sorry

end find_q_l307_307360


namespace prove_dollar_op_l307_307639

variable {a b x y : ℝ}

def dollar_op (a b : ℝ) : ℝ := (a - b) ^ 2

theorem prove_dollar_op :
  dollar_op (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) := by
  sorry

end prove_dollar_op_l307_307639


namespace Q_subset_P_l307_307336

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l307_307336


namespace problem_geometric_sequence_l307_307652

variable {α : Type*} [LinearOrderedField α]

noncomputable def geom_sequence_5_8 (a : α) (h : a + 8 * a = 2) : α :=
  (a * 2^4 + a * 2^7)

theorem problem_geometric_sequence : ∃ (a : α), (a + 8 * a = 2) ∧ geom_sequence_5_8 a (sorry) = 32 := 
by sorry

end problem_geometric_sequence_l307_307652


namespace elois_banana_bread_l307_307111

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l307_307111


namespace find_constant_term_l307_307363

-- Definitions based on conditions:
def sum_of_coeffs (n : ℕ) : ℕ := 4 ^ n
def sum_of_binom_coeffs (n : ℕ) : ℕ := 2 ^ n
def P_plus_Q_equals (n : ℕ) : Prop := sum_of_coeffs n + sum_of_binom_coeffs n = 272

-- Constant term in the binomial expansion:
def constant_term (n r : ℕ) : ℕ := Nat.choose n r * (3 ^ (n - r))

-- The proof statement
theorem find_constant_term : 
  ∃ n r : ℕ, P_plus_Q_equals n ∧ n = 4 ∧ r = 1 ∧ constant_term n r = 108 :=
by {
  sorry
}

end find_constant_term_l307_307363


namespace expected_coincidences_l307_307395

/-- Given conditions for the test -/
def num_questions : ℕ := 20
def vasya_correct : ℕ := 6
def misha_correct : ℕ := 8
def vasya_prob_correct : ℝ := 6 / 20
def misha_prob_correct : ℝ := 8 / 20
def coincidence_prob : ℝ :=
  (vasya_prob_correct * misha_prob_correct) + (1 - vasya_prob_correct) * (1 - misha_prob_correct)

/-- Expected number of coincidences -/
theorem expected_coincidences :
  20 * coincidence_prob = 10.8 :=
by {
  -- vasya_prob_correct = 0.3
  -- misha_prob_correct = 0.4
  -- probability of coincidence = 0.3 * 0.4 + 0.7 * 0.6 = 0.54
  -- expected number of coincidences = 20 * 0.54 = 10.8
  sorry
}

end expected_coincidences_l307_307395


namespace lcm_perfect_square_l307_307845

-- Define the conditions and the final statement in Lean 4
theorem lcm_perfect_square (a b : ℕ) 
  (h: (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : 
  ∃ k : ℕ, lcm a b = k^2 :=
sorry

end lcm_perfect_square_l307_307845


namespace parallel_perpendicular_implies_l307_307863

variables {Line : Type} {Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions
axiom distinct_lines : m ≠ n
axiom distinct_planes : α ≠ β

-- Parallel and Perpendicular relationships
axiom parallel : Line → Plane → Prop
axiom perpendicular : Line → Plane → Prop

-- Given conditions
axiom parallel_mn : parallel m n
axiom perpendicular_mα : perpendicular m α

-- Proof statement
theorem parallel_perpendicular_implies (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α :=
sorry

end parallel_perpendicular_implies_l307_307863


namespace largest_r_in_subset_l307_307116

theorem largest_r_in_subset (A : Finset ℕ) (hA : A.card = 500) : 
  ∃ (B C : Finset ℕ), B ⊆ A ∧ C ⊆ A ∧ (B ∩ C).card ≥ 100 := sorry

end largest_r_in_subset_l307_307116


namespace f_neg_2008_value_l307_307301

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem f_neg_2008_value (h : f a b 2008 = 10) : f a b (-2008) = -12 := by
  sorry

end f_neg_2008_value_l307_307301


namespace chandler_weeks_to_buy_bike_l307_307424

-- Define the given problem conditions as variables/constants
def bike_cost : ℕ := 650
def grandparents_gift : ℕ := 60
def aunt_gift : ℕ := 45
def cousin_gift : ℕ := 25
def weekly_earnings : ℕ := 20
def total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift

-- Define the total money Chandler will have after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_birthday_money + weekly_earnings * x

-- The main theorem states that Chandler needs 26 weeks to save enough money to buy the bike
theorem chandler_weeks_to_buy_bike : ∃ x : ℕ, total_money_after_weeks x = bike_cost :=
by
  -- Since we know x = 26 from the solution:
  use 26
  sorry

end chandler_weeks_to_buy_bike_l307_307424


namespace number_of_three_digit_numbers_with_sum_seven_l307_307575

theorem number_of_three_digit_numbers_with_sum_seven : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) →
  (card { n : ℕ | ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ a + b + c = 7 ∧ 100 ≤ n ∧ n < 1000 } = 28) :=
by
  sorry

end number_of_three_digit_numbers_with_sum_seven_l307_307575


namespace morgan_olivia_same_debt_l307_307841

theorem morgan_olivia_same_debt (t : ℝ) : 
  (200 * (1 + 0.12 * t) = 300 * (1 + 0.04 * t)) → 
  t = 25 / 3 :=
by
  sorry

end morgan_olivia_same_debt_l307_307841


namespace min_value_proof_l307_307777

noncomputable def min_value (x y : ℝ) : ℝ :=
x^3 + y^3 - x^2 - y^2

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) : 
min_value x y ≥ 1 := by
  sorry

end min_value_proof_l307_307777


namespace lcm_perfect_square_l307_307846

-- Define the conditions and the final statement in Lean 4
theorem lcm_perfect_square (a b : ℕ) 
  (h: (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : 
  ∃ k : ℕ, lcm a b = k^2 :=
sorry

end lcm_perfect_square_l307_307846


namespace area_spot_can_reach_l307_307694

noncomputable def area_reachable_by_spot (s : ℝ) (r : ℝ) : ℝ := 
  if s = 1 ∧ r = 3 then 6.5 * Real.pi else 0

theorem area_spot_can_reach : area_reachable_by_spot 1 3 = 6.5 * Real.pi :=
by
  -- The theorem proof should go here.
  sorry

end area_spot_can_reach_l307_307694


namespace lines_parallel_if_perpendicular_to_plane_l307_307334

variables (m n l : Line) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m n : Line) : Prop := sorry

theorem lines_parallel_if_perpendicular_to_plane
  (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l307_307334


namespace hyperbola_intersection_l307_307428

theorem hyperbola_intersection (b : ℝ) (h₁ : b > 0) :
  (b > 1) → (∀ x y : ℝ, ((x + 3 * y - 1 = 0) → ( ∃ x y : ℝ, (x^2 / 4 - y^2 / b^2 = 1) ∧ (x + 3 * y - 1 = 0))))
  :=
  sorry

end hyperbola_intersection_l307_307428


namespace fourth_boy_payment_l307_307436

theorem fourth_boy_payment (a b c d : ℝ) 
  (h₁ : a = (1 / 2) * (b + c + d)) 
  (h₂ : b = (1 / 3) * (a + c + d)) 
  (h₃ : c = (1 / 4) * (a + b + d)) 
  (h₄ : a + b + c + d = 60) : 
  d = 13 := 
sorry

end fourth_boy_payment_l307_307436


namespace fraction_decomposition_l307_307281

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ -8/3 → (7 * x - 19) / (3 * x^2 + 5 * x - 8) = A / (x - 1) + B / (3 * x + 8)) →
  A = -12 / 11 ∧ B = 113 / 11 :=
by
  sorry

end fraction_decomposition_l307_307281


namespace pencil_cost_is_11_l307_307842

-- Define the initial and remaining amounts
def initial_amount : ℤ := 15
def remaining_amount : ℤ := 4

-- Define the cost of the pencil
def cost_of_pencil : ℤ := initial_amount - remaining_amount

-- The statement we need to prove
theorem pencil_cost_is_11 : cost_of_pencil = 11 :=
by
  sorry

end pencil_cost_is_11_l307_307842


namespace elois_banana_bread_l307_307110

theorem elois_banana_bread :
  let bananas_per_loaf := 4
  let loaves_monday := 3
  let loaves_tuesday := 2 * loaves_monday
  let total_loaves := loaves_monday + loaves_tuesday
  let total_bananas := total_loaves * bananas_per_loaf
  total_bananas = 36 := sorry

end elois_banana_bread_l307_307110


namespace arithmetic_sequence_and_formula_l307_307826

noncomputable def S_n (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).sum a

noncomputable def b_n (S : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range (n + 1)).prod S

theorem arithmetic_sequence_and_formula (S : ℕ → ℚ) (b : ℕ → ℚ) (a : ℕ → ℚ) (h1 : ∀ (n : ℕ), S n = S_n a n)
  (h2 : ∀ (n : ℕ), b n = b_n S n) (h3 : ∀ (n : ℕ), 2 / S n + 1 / b n = 2) :
  (∀ (n : ℕ), b (n + 1) - b n = 1 / 2) ∧
  (a 1 = 3 / 2 ∧ ∀ (n : ℕ), n ≥ 1 → a (n + 1) = -(1 / ((n + 1) * (n + 2)))) :=
by
  sorry

end arithmetic_sequence_and_formula_l307_307826


namespace digits_sum_eq_seven_l307_307623

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l307_307623


namespace incenter_lines_pass_through_orthocenter_l307_307835

theorem incenter_lines_pass_through_orthocenter
  (A E F D B C : Point)
  (A1 B1 C1 : Point)
  (hA1 : is_incenter_of_triangle A1 A E F)
  (hB1 : is_incenter_of_triangle B1 B D F)
  (hC1 : is_incenter_of_triangle C1 C D E) :
  passes_through_orthocenter A1 D B1 E C1 F A1 B1 C1 := sorry

end incenter_lines_pass_through_orthocenter_l307_307835


namespace find_physics_marks_l307_307915

variable (P C M : ℕ)

theorem find_physics_marks
  (h1 : P + C + M = 225)
  (h2 : P + M = 180)
  (h3 : P + C = 140) : 
  P = 95 :=
by
  sorry

end find_physics_marks_l307_307915


namespace problem_l307_307418

noncomputable def octagon_chord_length : ℚ := sorry

theorem problem : ∃ (p q : ℕ), p + q = 5 ∧ (nat.gcd p q = 1) ∧ (octagon_chord_length = p / q) :=
begin
  sorry
end

end problem_l307_307418


namespace line_slope_intercept_l307_307905

theorem line_slope_intercept :
  (∀ (x y : ℝ), 3 * (x + 2) - 4 * (y - 8) = 0 → y = (3/4) * x + 9.5) :=
sorry

end line_slope_intercept_l307_307905


namespace num_three_digit_sums7_l307_307605

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l307_307605


namespace three_digit_sum_seven_l307_307615

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l307_307615


namespace allocation_count_l307_307509

def allocate_volunteers (num_service_points : Nat) (num_volunteers : Nat) : Nat :=
  -- Definition that captures the counting logic as per the problem statement
  if num_service_points = 4 ∧ num_volunteers = 6 then 660 else 0

theorem allocation_count :
  allocate_volunteers 4 6 = 660 :=
sorry

end allocation_count_l307_307509


namespace ursula_hourly_wage_l307_307713

def annual_salary : ℝ := 16320
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

theorem ursula_hourly_wage : 
  (annual_salary / months_per_year) / (hours_per_day * days_per_month) = 8.50 := by 
  sorry

end ursula_hourly_wage_l307_307713


namespace inverse_47_mod_48_l307_307757

theorem inverse_47_mod_48 : ∃ x, x < 48 ∧ x > 0 ∧ 47 * x % 48 = 1 :=
sorry

end inverse_47_mod_48_l307_307757


namespace num_three_digit_integers_sum_to_seven_l307_307580

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l307_307580


namespace horizontal_asymptote_condition_l307_307105

open Polynomial

def polynomial_deg_with_horiz_asymp (p : Polynomial ℝ) : Prop :=
  degree p ≤ 4

theorem horizontal_asymptote_condition (p : Polynomial ℝ) :
  polynomial_deg_with_horiz_asymp p :=
sorry

end horizontal_asymptote_condition_l307_307105


namespace bananas_used_l307_307108

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l307_307108


namespace probability_rel_prime_to_42_l307_307885

theorem probability_rel_prime_to_42 : 
  let n := 42 in
  let prime_factors := [2, 3, 7] in
  let relatively_prime_count := n * (1 - 1/prime_factors[0]) * (1 - 1/prime_factors[1]) * (1 - 1/prime_factors[2]) in
  let total_count := 42 in
  (relatively_prime_count / total_count) = 4 / 7 :=
by
  sorry

end probability_rel_prime_to_42_l307_307885


namespace no_divisor_neighbors_l307_307348

def is_divisor (a b : ℕ) : Prop := b % a = 0

def circle_arrangement (arr : Fin 8 → ℕ) : Prop :=
  arr 0 = 7 ∧ arr 1 = 9 ∧ arr 2 = 4 ∧ arr 3 = 5 ∧ arr 4 = 3 ∧ arr 5 = 6 ∧ arr 6 = 8 ∧ arr 7 = 2

def valid_neighbors (arr : Fin 8 → ℕ) : Prop :=
  ¬ is_divisor (arr 0) (arr 1) ∧ ¬ is_divisor (arr 0) (arr 3) ∧
  ¬ is_divisor (arr 1) (arr 2) ∧ ¬ is_divisor (arr 1) (arr 3) ∧ ¬ is_divisor (arr 1) (arr 5) ∧
  ¬ is_divisor (arr 2) (arr 1) ∧ ¬ is_divisor (arr 2) (arr 6) ∧ ¬ is_divisor (arr 2) (arr 3) ∧
  ¬ is_divisor (arr 3) (arr 1) ∧ ¬ is_divisor (arr 3) (arr 4) ∧ ¬ is_divisor (arr 3) (arr 2) ∧ ¬ is_divisor (arr 3) (arr 0) ∧
  ¬ is_divisor (arr 4) (arr 3) ∧ ¬ is_divisor (arr 4) (arr 5) ∧
  ¬ is_divisor (arr 5) (arr 1) ∧ ¬ is_divisor (arr 5) (arr 4) ∧ ¬ is_divisor (arr 5) (arr 6) ∧
  ¬ is_divisor (arr 6) (arr 2) ∧ ¬ is_divisor (arr 6) (arr 5) ∧ ¬ is_divisor (arr 6) (arr 7) ∧
  ¬ is_divisor (arr 7) (arr 6)

theorem no_divisor_neighbors :
  ∀ (arr : Fin 8 → ℕ), circle_arrangement arr → valid_neighbors arr :=
by
  intros arr h
  sorry

end no_divisor_neighbors_l307_307348


namespace correct_comparison_l307_307391

theorem correct_comparison :
  ( 
    (-1 > -0.1) = false ∧ 
    (-4 / 3 < -5 / 4) = true ∧ 
    (-1 / 2 > -(-1 / 3)) = false ∧ 
    (Real.pi = 3.14) = false 
  ) :=
by
  sorry

end correct_comparison_l307_307391


namespace small_load_clothing_count_l307_307510

def initial_clothes : ℕ := 36
def first_load_clothes : ℕ := 18
def remaining_clothes := initial_clothes - first_load_clothes
def small_load_clothes := remaining_clothes / 2

theorem small_load_clothing_count : 
  small_load_clothes = 9 :=
by
  sorry

end small_load_clothing_count_l307_307510


namespace quincy_monthly_payment_l307_307995

-- Definitions based on the conditions:
def car_price : ℕ := 20000
def down_payment : ℕ := 5000
def loan_years : ℕ := 5
def months_in_year : ℕ := 12

-- The mathematical problem to be proven:
theorem quincy_monthly_payment :
  let amount_to_finance := car_price - down_payment
  let total_months := loan_years * months_in_year
  amount_to_finance / total_months = 250 := by
  sorry

end quincy_monthly_payment_l307_307995


namespace f_2009_l307_307268

def f (x : ℝ) : ℝ := x^3 -- initial definition for x in [-1, 1]

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom symmetric_around_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_cubed : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3

theorem f_2009 : f 2009 = 1 := by {
  -- The body of the theorem will be filled with proof steps
  sorry
}

end f_2009_l307_307268


namespace simon_practice_hours_l307_307038

theorem simon_practice_hours (x : ℕ) (h : (12 + 16 + 14 + x) / 4 ≥ 15) : x = 18 := 
by {
  -- placeholder for the proof
  sorry
}

end simon_practice_hours_l307_307038


namespace volume_percentage_correct_l307_307541

-- Define the initial conditions
def box_length := 8
def box_width := 6
def box_height := 12
def cube_side := 3

-- Calculate the number of cubes along each dimension
def num_cubes_length := box_length / cube_side
def num_cubes_width := box_width / cube_side
def num_cubes_height := box_height / cube_side

-- Calculate volumes
def volume_cube := cube_side ^ 3
def volume_box := box_length * box_width * box_height
def volume_cubes := (num_cubes_length * num_cubes_width * num_cubes_height) * volume_cube

-- Prove the percentage calculation
theorem volume_percentage_correct : (volume_cubes.toFloat / volume_box.toFloat) * 100 = 75 := by
  sorry

end volume_percentage_correct_l307_307541


namespace vector_identity_l307_307241

namespace VectorAddition

variable {V : Type*} [AddCommGroup V]

theorem vector_identity
  (AD DC AB BC : V)
  (h1 : AD + DC = AC)
  (h2 : AC - AB = BC) :
  AD + DC - AB = BC :=
by
  sorry

end VectorAddition

end vector_identity_l307_307241


namespace number_of_yellow_balls_l307_307093

theorem number_of_yellow_balls (x : ℕ) :
  (4 : ℕ) / (4 + x) = 2 / 3 → x = 2 :=
by
  sorry

end number_of_yellow_balls_l307_307093


namespace tens_digit_23_1987_l307_307279

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l307_307279


namespace Tim_gave_kittens_to_Jessica_l307_307877

def Tim_original_kittens : ℕ := 6
def kittens_given_to_Jessica := 3
def kittens_given_by_Sara : ℕ := 9 
def Tim_final_kittens : ℕ := 12

theorem Tim_gave_kittens_to_Jessica :
  (Tim_original_kittens + kittens_given_by_Sara - kittens_given_to_Jessica = Tim_final_kittens) :=
by sorry

end Tim_gave_kittens_to_Jessica_l307_307877


namespace b_seq_arithmetic_a_seq_formula_l307_307822

-- Definitions and conditions
def a_seq (n : ℕ) : ℚ := sorry
def S_n (n : ℕ) : ℚ := (finset.range n).sum (λ k, a_seq (k + 1))
def b_n (n : ℕ) : ℚ := (finset.range n).prod (λ k, S_n (k + 1))

axiom given_condition : ∀ n : ℕ, 2 / S_n n + 1 / b_n n = 2

-- Theorems to be proved
theorem b_seq_arithmetic : ∃ d : ℚ, ∀ n ≥ 1, b_n n = b_n (n - 1) + d := sorry

theorem a_seq_formula : ∀ n : ℕ, 
  a_seq n = if n = 1 then 3 / 2 else -1 / (n * (n + 1)) := sorry

end b_seq_arithmetic_a_seq_formula_l307_307822


namespace geometric_series_sum_l307_307564

noncomputable def T (r : ℝ) := 15 / (1 - r)

theorem geometric_series_sum (b : ℝ) (hb1 : -1 < b) (hb2 : b < 1) (H : T b * T (-b) = 3240) : T b + T (-b) = 432 := 
by sorry

end geometric_series_sum_l307_307564


namespace problem_inequality_I_problem_inequality_II_l307_307643

theorem problem_inequality_I (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  1 / a + 1 / b ≥ 4 := sorry

theorem problem_inequality_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := sorry

end problem_inequality_I_problem_inequality_II_l307_307643


namespace problem_l307_307661

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x|

theorem problem :
  (∀ x, f x ≤ 1) ∧
  (∃ x, f x = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 
    ∃ x, (x = (a^2 / (b + 1) + b^2 / (a + 1)) ∧ x = 1 / 3)) :=
by {
  sorry
}

end problem_l307_307661


namespace line_always_intersects_circle_shortest_chord_line_equation_l307_307647

open Real

noncomputable def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 9 = 0

noncomputable def line_eqn (m x y : ℝ) : Prop := 2 * m * x - 3 * m * y + x - y - 1 = 0

theorem line_always_intersects_circle (m : ℝ) : 
  ∀ (x y : ℝ), circle_eqn x y → line_eqn m x y → True := 
by
  sorry

theorem shortest_chord_line_equation : 
  ∃ (m x y : ℝ), line_eqn m x y ∧ (∀ x y, line_eqn m x y → x - y - 1 = 0) :=
by
  sorry

end line_always_intersects_circle_shortest_chord_line_equation_l307_307647


namespace train_crossing_time_l307_307408

-- Define the length of the train, the speed of the train, and the length of the bridge
def train_length : ℕ := 140
def train_speed_kmh : ℕ := 45
def bridge_length : ℕ := 235

-- Define constants for unit conversions
def km_to_m : ℕ := 1000
def hr_to_s : ℕ := 3600

-- Calculate the speed in m/s
def train_speed_ms : ℝ :=
  (train_speed_kmh : ℝ) * (km_to_m : ℝ) / (hr_to_s : ℝ)

-- Calculate the total distance to cover (length of train + length of bridge)
def total_distance : ℕ := train_length + bridge_length

-- Calculate the time in seconds required for the train to cross the bridge
def crossing_time : ℝ :=
  (total_distance : ℝ) / train_speed_ms

-- Prove that the crossing time is 30 seconds
theorem train_crossing_time : crossing_time = 30 := by
  sorry

end train_crossing_time_l307_307408


namespace primes_in_sequence_are_12_l307_307103

-- Definition of Q
def Q : Nat := (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47)

-- Set of m values
def ms : List Nat := List.range' 3 101

-- Function to check if Q + m is prime
def is_prime_minus_Q (m : Nat) : Bool := Nat.Prime (Q + m)

-- Counting primes in the sequence
def count_primes_in_sequence : Nat := (ms.filter (λ m => is_prime_minus_Q m = true)).length

theorem primes_in_sequence_are_12 :
  count_primes_in_sequence = 12 := by 
  sorry

end primes_in_sequence_are_12_l307_307103


namespace number_of_three_digit_numbers_with_sum_7_l307_307567

noncomputable def count_three_digit_nums_with_digit_sum_7 : ℕ :=
  (Finset.Icc 1 9).sum (λ a =>
    (Finset.Icc 0 9).sum (λ b =>
      (Finset.Icc 0 9).count (λ c => a + b + c = 7)))

theorem number_of_three_digit_numbers_with_sum_7 : count_three_digit_nums_with_digit_sum_7 = 28 := sorry

end number_of_three_digit_numbers_with_sum_7_l307_307567


namespace paul_money_last_weeks_l307_307725

theorem paul_money_last_weeks (a b c: ℕ) (h1: a = 68) (h2: b = 13) (h3: c = 9) : 
  (a + b) / c = 9 := 
by 
  sorry

end paul_money_last_weeks_l307_307725


namespace find_scooters_l307_307736

variables (b t s : ℕ)

theorem find_scooters (h1 : b + t + s = 13) (h2 : 2 * b + 3 * t + 2 * s = 30) : s = 9 :=
sorry

end find_scooters_l307_307736


namespace minimum_value_of_a_l307_307309

theorem minimum_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y ≥ 9) : ∃ a > 0, a ≥ 4 :=
sorry

end minimum_value_of_a_l307_307309


namespace twelve_pharmacies_not_enough_l307_307535

def grid := ℕ × ℕ

def is_within_walking_distance (p1 p2 : grid) : Prop :=
  abs (p1.1 - p1.2) ≤ 3 ∧ abs (p2.1 - p2.2) ≤ 3

def walking_distance_coverage (pharmacies : set grid) (p : grid) : Prop :=
  ∃ pharmacy ∈ pharmacies, is_within_walking_distance pharmacy p

def sufficient_pharmacies (pharmacies : set grid) : Prop :=
  ∀ p : grid, walking_distance_coverage pharmacies p

theorem twelve_pharmacies_not_enough (pharmacies : set grid) (h : pharmacies.card = 12) : 
  ¬ sufficient_pharmacies pharmacies :=
sorry

end twelve_pharmacies_not_enough_l307_307535


namespace quadratic_polynomial_exists_l307_307425

theorem quadratic_polynomial_exists (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ p : ℝ → ℝ, (∀ x, p x = (a^2 + ab + b^2 + ac + bc + c^2) * x^2 
                   - (a + b) * (b + c) * (a + c) * x 
                   + abc * (a + b + c))
              ∧ p a = a^4 
              ∧ p b = b^4 
              ∧ p c = c^4 := 
sorry

end quadratic_polynomial_exists_l307_307425


namespace combined_girls_avg_l307_307412

noncomputable def centralHS_boys_avg := 68
noncomputable def deltaHS_boys_avg := 78
noncomputable def combined_boys_avg := 74
noncomputable def centralHS_girls_avg := 72
noncomputable def deltaHS_girls_avg := 85
noncomputable def centralHS_combined_avg := 70
noncomputable def deltaHS_combined_avg := 80

theorem combined_girls_avg (C c D d : ℝ) 
  (h1 : (68 * C + 72 * c) / (C + c) = 70)
  (h2 : (78 * D + 85 * d) / (D + d) = 80)
  (h3 : (68 * C + 78 * D) / (C + D) = 74) :
  (3/7 * 72 + 4/7 * 85) = 79 := 
by 
  sorry

end combined_girls_avg_l307_307412


namespace crayons_eaten_l307_307978

def initial_crayons : ℕ := 87
def remaining_crayons : ℕ := 80

theorem crayons_eaten : initial_crayons - remaining_crayons = 7 := by
  sorry

end crayons_eaten_l307_307978


namespace ratio_depends_on_S_and_r_l307_307931

theorem ratio_depends_on_S_and_r
    (S : ℝ) (r : ℝ) (P1 : ℝ) (C2 : ℝ)
    (h1 : P1 = 4 * S)
    (h2 : C2 = 2 * Real.pi * r) :
    (P1 / C2 = 4 * S / (2 * Real.pi * r)) := by
  sorry

end ratio_depends_on_S_and_r_l307_307931


namespace yoongi_class_combination_l307_307071

theorem yoongi_class_combination : (Nat.choose 10 3 = 120) := by
  sorry

end yoongi_class_combination_l307_307071


namespace three_digit_numbers_sum_seven_l307_307634

theorem three_digit_numbers_sum_seven : 
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a b c : ℕ), (10*a + b + c = n) ∧ (a + b + c = 7) ∧ (1 ≤ a)) ∧ 
  (nat.choose 8 2 = 28) := 
by
  sorry

end three_digit_numbers_sum_seven_l307_307634


namespace perpendicular_lines_condition_l307_307197

variable {A1 B1 C1 A2 B2 C2 : ℝ}

theorem perpendicular_lines_condition :
  (∀ x y : ℝ, A1 * x + B1 * y + C1 = 0) ∧ (∀ x y : ℝ, A2 * x + B2 * y + C2 = 0) → 
  (A1 * A2) / (B1 * B2) = -1 := 
sorry

end perpendicular_lines_condition_l307_307197


namespace minimum_production_quantity_l307_307876

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the revenue function given the selling price per unit
def revenue (x : ℝ) : ℝ := 25 * x

-- Define the interval for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 240

-- State the minimum production quantity required to avoid a loss
theorem minimum_production_quantity (x : ℝ) (h : x_range x) : 150 <= x :=
by
  -- Sorry replaces the detailed proof steps
  sorry

end minimum_production_quantity_l307_307876


namespace max_area_rectangle_l307_307404

theorem max_area_rectangle :
  ∃ (l w : ℕ), (2 * (l + w) = 40) ∧ (l ≥ w + 3) ∧ (l * w = 91) :=
by
  sorry

end max_area_rectangle_l307_307404


namespace solve_quadratic_substitution_l307_307349

theorem solve_quadratic_substitution (x : ℝ) : 
  (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 := 
by sorry

end solve_quadratic_substitution_l307_307349


namespace sum_of_areas_of_circles_l307_307377

-- Definitions of conditions
def r : ℝ := by sorry  -- radius of the circle at vertex A
def s : ℝ := by sorry  -- radius of the circle at vertex B
def t : ℝ := by sorry  -- radius of the circle at vertex C

axiom sum_radii_r_s : r + s = 6
axiom sum_radii_r_t : r + t = 8
axiom sum_radii_s_t : s + t = 10

-- The statement we want to prove
theorem sum_of_areas_of_circles : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by
  -- Use given axioms and properties of the triangle and circles
  sorry

end sum_of_areas_of_circles_l307_307377


namespace total_paintable_area_correct_l307_307080

namespace BarnPainting

-- Define the dimensions of the barn
def barn_width : ℕ := 12
def barn_length : ℕ := 15
def barn_height : ℕ := 6

-- Define the dimensions of the windows
def window_width : ℕ := 2
def window_height : ℕ := 3
def num_windows : ℕ := 2

-- Calculate the total number of square yards to be painted
def total_paintable_area : ℕ :=
  let wall1_area := barn_height * barn_width
  let wall2_area := barn_height * barn_length
  let wall_area := 2 * wall1_area + 2 * wall2_area
  let window_area := num_windows * (window_width * window_height)
  let painted_walls_area := wall_area - window_area
  let ceiling_area := barn_width * barn_length
  let total_area := 2 * painted_walls_area + ceiling_area
  total_area

theorem total_paintable_area_correct : total_paintable_area = 780 :=
  by sorry

end BarnPainting

end total_paintable_area_correct_l307_307080


namespace some_base_value_l307_307807

noncomputable def some_base (x y : ℝ) (h1 : x * y = 1) (h2 : (some_base : ℝ) → (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : ℝ :=
  7

theorem some_base_value (x y : ℝ) (h1 : x * y = 1) (h2 : ∀ some_base : ℝ, (some_base ^ (x + y))^2 / (some_base ^ (x - y))^2 = 2401) : some_base x y h1 h2 = 7 :=
by
  sorry

end some_base_value_l307_307807


namespace smallest_d_in_range_l307_307763

theorem smallest_d_in_range (d : ℝ) : (∃ x : ℝ, x^2 + 5 * x + d = 5) ↔ d ≤ 45 / 4 := 
sorry

end smallest_d_in_range_l307_307763


namespace special_burger_cost_l307_307421

/-
  Prices of individual items and meals:
  - Burger: $5
  - French Fries: $3
  - Soft Drink: $3
  - Kid’s Burger: $3
  - Kid’s French Fries: $2
  - Kid’s Juice Box: $2
  - Kids Meal: $5

  Mr. Parker purchases:
  - 2 special burger meals for adults
  - 2 special burger meals and 2 kids' meals for 4 children
  - Saving $10 by buying 6 meals instead of the individual items

  Goal: 
  - Prove that the cost of one special burger meal is $8.
-/

def price_burger : Nat := 5
def price_fries : Nat := 3
def price_drink : Nat := 3
def price_kid_burger : Nat := 3
def price_kid_fries : Nat := 2
def price_kid_juice : Nat := 2
def price_kids_meal : Nat := 5

def total_adults_cost : Nat :=
  2 * price_burger + 2 * price_fries + 2 * price_drink

def total_kids_cost : Nat :=
  2 * price_kid_burger + 2 * price_kid_fries + 2 * price_kid_juice

def total_individual_cost : Nat :=
  total_adults_cost + total_kids_cost

def total_meals_cost : Nat :=
  total_individual_cost - 10

def cost_kids_meals : Nat :=
  2 * price_kids_meal

def total_cost_4_meals : Nat :=
  total_meals_cost

def cost_special_burger_meal : Nat :=
  (total_cost_4_meals - cost_kids_meals) / 2

theorem special_burger_cost : cost_special_burger_meal = 8 := by
  sorry

end special_burger_cost_l307_307421


namespace problem_l307_307727

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x + c

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * b

theorem problem (a b c : ℝ) (h1 : f_prime a b 2 = 0) (h2 : f_prime a b 1 = -3) :
  a = -1 ∧ b = 0 ∧ (let f_min := f (-1) 0 c 2 
                   let f_max := 0 
                   f_max - f_min = 4) :=
by
  sorry

end problem_l307_307727


namespace grace_age_l307_307953

theorem grace_age 
  (H : ℕ) 
  (I : ℕ) 
  (J : ℕ) 
  (G : ℕ)
  (h1 : H = I - 5)
  (h2 : I = J + 7)
  (h3 : G = 2 * J)
  (h4 : H = 18) : 
  G = 32 := 
sorry

end grace_age_l307_307953


namespace three_digit_numbers_sum_seven_l307_307592

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l307_307592


namespace speed_with_stream_l307_307907

-- Definitions for the conditions in part a
def Vm : ℕ := 8  -- Speed of the man in still water (in km/h)
def Vs : ℕ := Vm - 4  -- Speed of the stream (in km/h), derived from man's speed against the stream

-- The statement to prove the man's speed with the stream
theorem speed_with_stream : Vm + Vs = 12 := by sorry

end speed_with_stream_l307_307907


namespace product_increase_l307_307420

variable (x : ℤ)

theorem product_increase (h : 53 * x = 1585) : 1585 - (35 * x) = 535 :=
by sorry

end product_increase_l307_307420


namespace inscribed_sphere_radius_l307_307199

theorem inscribed_sphere_radius {a : ℝ} :
  ∃ r : ℝ, (r = (a * (Real.sqrt 21 - 3)) / 4) :=
by
  sorry

end inscribed_sphere_radius_l307_307199


namespace tomato_land_correct_l307_307849

-- Define the conditions
def total_land : ℝ := 4999.999999999999
def cleared_fraction : ℝ := 0.9
def grapes_fraction : ℝ := 0.1
def potato_fraction : ℝ := 0.8

-- Define the calculated values based on conditions
def cleared_land : ℝ := cleared_fraction * total_land
def grapes_land : ℝ := grapes_fraction * cleared_land
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := cleared_land - (grapes_land + potato_land)

-- Prove the question using conditions, which should end up being 450 acres.
theorem tomato_land_correct : tomato_land = 450 :=
by sorry

end tomato_land_correct_l307_307849


namespace sheela_total_income_l307_307854

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end sheela_total_income_l307_307854


namespace solution_proof_l307_307862

variable (x y z : ℝ)

-- Given system of equations
def equation1 := 6 / (3 * x + 4 * y) + 4 / (5 * x - 4 * z) = 7 / 12
def equation2 := 9 / (4 * y + 3 * z) - 4 / (3 * x + 4 * y) = 1 / 3
def equation3 := 2 / (5 * x - 4 * z) + 6 / (4 * y + 3 * z) = 1 / 2

theorem solution_proof : 
  equation1 4 3 2 ∧ equation2 4 3 2 ∧ equation3 4 3 2 := by
  sorry

end solution_proof_l307_307862


namespace remainder_cd_42_l307_307820

theorem remainder_cd_42 (c d : ℕ) (p q : ℕ) (hc : c = 84 * p + 76) (hd : d = 126 * q + 117) : 
  (c + d) % 42 = 25 :=
by
  sorry

end remainder_cd_42_l307_307820


namespace union_A_B_l307_307146

noncomputable def U := Set.univ ℝ

def A : Set ℝ := {x | x^2 - x - 2 = 0}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x + 3}

theorem union_A_B : A ∪ B = { -1, 2, 5 } :=
by
  sorry

end union_A_B_l307_307146


namespace problem_statement_l307_307131

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def pow_log2 (x : ℝ) : ℝ := x ^ log2 x

theorem problem_statement (a b c : ℝ)
  (h0 : 1 ≤ a)
  (h1 : 1 ≤ b)
  (h2 : 1 ≤ c)
  (h3 : a * b * c = 10)
  (h4 : pow_log2 a * pow_log2 b * pow_log2 c ≥ 10) :
  a + b + c = 12 := by
  sorry

end problem_statement_l307_307131


namespace find_valid_pair_l307_307427

noncomputable def valid_angle (x : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ x = 180 * (n - 2) / n

noncomputable def valid_pair (x k : ℕ) : Prop :=
  valid_angle x ∧ valid_angle (k * x) ∧ 1 < k ∧ k < 5

theorem find_valid_pair : valid_pair 60 2 :=
by
  sorry

end find_valid_pair_l307_307427


namespace num_three_digit_sums7_l307_307606

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l307_307606


namespace digits_sum_eq_seven_l307_307622

theorem digits_sum_eq_seven : 
  (∃ n : Finset ℕ, n.card = 28 ∧ ∀ x : Finset ℕ, x.card = 3 → ∀ (a b c : ℕ), a + b + c = 7 → x = {a, b, c} → a >= 1 → n.card = 28) ∧
  ∃ (a b c : ℕ), a + b + c = 7 ∧ a >= 1 ∧ finset.card (finset.image (λ n, (a + b + c)) (finset.range 1000)) = 28 → finset.card (finset.range 1000) = 28 :=
by sorry

end digits_sum_eq_seven_l307_307622


namespace ratio_of_triangle_to_square_l307_307282

theorem ratio_of_triangle_to_square (s : ℝ) (hs : 0 < s) :
  let A_square := s^2
  let A_triangle := (1/2) * s * (s/2)
  A_triangle / A_square = 1/4 :=
by
  sorry

end ratio_of_triangle_to_square_l307_307282


namespace sum_real_imag_parts_l307_307642

noncomputable section

open Complex

theorem sum_real_imag_parts (z : ℂ) (h : z / (1 + 2 * i) = 2 + i) : 
  ((z + 5).re + (z + 5).im) = 0 :=
  by
  sorry

end sum_real_imag_parts_l307_307642


namespace factor_difference_of_squares_l307_307722

theorem factor_difference_of_squares (a b p q : ℝ) :
  (∃ c d : ℝ, -a ^ 2 + 9 = c ^ 2 - d ^ 2) ∧
  (¬(∃ c d : ℝ, -a ^ 2 - b ^ 2 = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, p ^ 2 - (-q ^ 2) = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, a ^ 2 - b ^ 3 = c ^ 2 - d ^ 2)) := 
  by 
  sorry

end factor_difference_of_squares_l307_307722


namespace lcm_is_perfect_square_l307_307847

open Nat

theorem lcm_is_perfect_square (a b : ℕ) : 
  (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0 → ∃ k : ℕ, k^2 = lcm a b :=
by
  sorry

end lcm_is_perfect_square_l307_307847


namespace xyz_inequality_l307_307983

theorem xyz_inequality (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + 
  (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
sorry

end xyz_inequality_l307_307983


namespace probability_greater_than_4_l307_307975

-- Given conditions
def die_faces : ℕ := 6
def favorable_outcomes : Finset ℕ := {5, 6}

-- Probability calculation
def probability (total : ℕ) (favorable : Finset ℕ) : ℚ :=
  favorable.card / total

theorem probability_greater_than_4 :
  probability die_faces favorable_outcomes = 1 / 3 :=
by
  sorry

end probability_greater_than_4_l307_307975


namespace num_three_digit_sums7_l307_307603

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l307_307603


namespace division_of_negatives_example_div_l307_307928

theorem division_of_negatives (a b : Int) (ha : a < 0) (hb : b < 0) (hb_neq : b ≠ 0) : 
  (-a) / (-b) = a / b :=
by sorry

theorem example_div : (-300) / (-50) = 6 :=
by
  apply division_of_negatives
  repeat { sorry }

end division_of_negatives_example_div_l307_307928


namespace num_three_digit_integers_sum_to_seven_l307_307582

open Nat

-- Define the three digits a, b, and c and their constraints
def digits_satisfy (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7

-- State the theorem we wish to prove
theorem num_three_digit_integers_sum_to_seven :
  (∃ n : ℕ, ∃ a b c : ℕ, digits_satisfy a b c ∧ a * 100 + b * 10 + c = n) →
  (∃ N : ℕ, N = 28) :=
sorry

end num_three_digit_integers_sum_to_seven_l307_307582


namespace quadratic_non_negative_iff_a_in_range_l307_307444

theorem quadratic_non_negative_iff_a_in_range :
  (∀ x : ℝ, x^2 + (a - 2) * x + 1/4 ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
sorry

end quadratic_non_negative_iff_a_in_range_l307_307444


namespace rhombus_diagonal_l307_307491

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 10) (h2 : area = 60) : 
  d1 = 12 :=
by 
  have : (d1 * d2) / 2 = area := sorry
  sorry

end rhombus_diagonal_l307_307491


namespace fraction_picked_l307_307414

/--
An apple tree has three times as many apples as the number of plums on a plum tree.
Damien picks a certain fraction of the fruits from the trees, and there are 96 plums
and apples remaining on the tree. There were 180 apples on the apple tree before 
Damien picked any of the fruits. Prove that Damien picked 3/5 of the fruits from the trees.
-/
theorem fraction_picked (P F : ℝ) (h1 : 3 * P = 180) (h2 : (1 - F) * (180 + P) = 96) :
  F = 3 / 5 :=
by
  sorry

end fraction_picked_l307_307414


namespace find_f_100_l307_307123

-- Define the function f such that it satisfies the condition f(10^x) = x
noncomputable def f : ℝ → ℝ := sorry

-- Define the main theorem to prove f(100) = 2 given the condition f(10^x) = x
theorem find_f_100 (h : ∀ x : ℝ, f (10^x) = x) : f 100 = 2 :=
by {
  sorry
}

end find_f_100_l307_307123


namespace area_of_CEF_l307_307675

-- Definitions of points and triangles based on given ratios
def is_right_triangle (A B C : Type) : Prop := sorry -- Placeholder for right triangle condition

def divides_ratio (A B : Type) (ratio : ℚ) : Prop := sorry -- Placeholder for ratio division condition

def area_of_triangle (A B C : Type) : ℚ := sorry -- Function to calculate area of triangle - placeholder

theorem area_of_CEF {A B C E F : Type} 
  (h1 : is_right_triangle A B C)
  (h2 : divides_ratio A C (1/4))
  (h3 : divides_ratio A B (2/3))
  (h4 : area_of_triangle A B C = 50) : 
  area_of_triangle C E F = 25 :=
sorry

end area_of_CEF_l307_307675


namespace three_digit_sum_seven_l307_307617

theorem three_digit_sum_seven : 
  (∃ (a b c : ℕ), a + b + c = 7 ∧ 1 ≤ a) → 28 :=
begin
  sorry
end

end three_digit_sum_seven_l307_307617


namespace smallest_constant_N_l307_307106

theorem smallest_constant_N (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) > 1 :=
by
  sorry

end smallest_constant_N_l307_307106


namespace find_softball_players_l307_307810

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def total_players : ℕ := 59

theorem find_softball_players :
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  S = T - (C + H + F) :=
by
  let C := cricket_players
  let H := hockey_players
  let F := football_players
  let T := total_players
  show S = T - (C + H + F)
  sorry

end find_softball_players_l307_307810


namespace bottles_from_shop_C_l307_307554

theorem bottles_from_shop_C (A B C T : ℕ) (hA : A = 150) (hB : B = 180) (hT : T = 550) (hSum : T = A + B + C) :
  C = 220 :=
by
  rw [hA, hB, hT] at hSum
  simpa using hSum

end bottles_from_shop_C_l307_307554


namespace num_common_points_l307_307271

-- Definitions of the given conditions:
def line1 (x y : ℝ) := x + 2 * y - 3 = 0
def line2 (x y : ℝ) := 4 * x - y + 1 = 0
def line3 (x y : ℝ) := 2 * x - y - 5 = 0
def line4 (x y : ℝ) := 3 * x + 4 * y - 8 = 0

-- The proof goal:
theorem num_common_points : 
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∨ line2 p.1 p.2) ∧ (line3 p.1 p.2 ∨ line4 p.1 p.2) :=
sorry

end num_common_points_l307_307271


namespace time_after_3108_hours_l307_307357

/-- The current time is 3 o'clock. On a 12-hour clock, 
 what time will it be 3108 hours from now? -/
theorem time_after_3108_hours : (3 + 3108) % 12 = 3 := 
by
  sorry

end time_after_3108_hours_l307_307357


namespace train_length_l307_307809

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h_speed : speed_kmph = 60) 
  (h_time : time_sec = 7.199424046076314) 
  (h_length : length_m = 120)
  : speed_kmph * (1000 / 3600) * time_sec = length_m :=
by 
  sorry

end train_length_l307_307809


namespace polynomials_equal_l307_307836

noncomputable def P : ℝ → ℝ := sorry -- assume P is a nonconstant polynomial
noncomputable def Q : ℝ → ℝ := sorry -- assume Q is a nonconstant polynomial

axiom floor_eq_for_all_y (y : ℝ) : ⌊P y⌋ = ⌊Q y⌋

theorem polynomials_equal (x : ℝ) : P x = Q x :=
by
  sorry

end polynomials_equal_l307_307836


namespace three_digit_sum_7_l307_307610

theorem three_digit_sum_7 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7}.card = 28 := 
by
  sorry

end three_digit_sum_7_l307_307610


namespace no_six_consecutive010101_l307_307188

def unit_digit (n: ℕ) : ℕ := n % 10

def sequence : ℕ → ℕ
| 0     => 1
| 1     => 0
| 2     => 1
| 3     => 0
| 4     => 1
| 5     => 0
| (n + 6) => unit_digit (sequence n + sequence (n + 1) + sequence (n + 2) + sequence (n + 3) + sequence (n + 4) + sequence (n + 5))

theorem no_six_consecutive010101 : ∀ n, ¬ (sequence n = 0 ∧ sequence (n + 1) = 1 ∧ sequence (n + 2) = 0 ∧ sequence (n + 3) = 1 ∧ sequence (n + 4) = 0 ∧ sequence (n + 5) = 1) :=
sorry

end no_six_consecutive010101_l307_307188


namespace age_difference_l307_307986

theorem age_difference (M T J X S : ℕ)
  (hM : M = 3)
  (hT : T = 4 * M)
  (hJ : J = T - 5)
  (hX : X = 2 * J)
  (hS : S = 3 * X - 1) :
  S - M = 38 :=
by
  sorry

end age_difference_l307_307986


namespace three_people_on_staircase_l307_307054

theorem three_people_on_staircase (A B C : Type) (steps : Finset ℕ) (h1 : steps.card = 7) 
  (h2 : ∀ step ∈ steps, step ≤ 2) : 
  ∃ (total_ways : ℕ), total_ways = 336 :=
by {
  sorry
}

end three_people_on_staircase_l307_307054


namespace jessica_balloons_l307_307009

-- Given conditions
def joan_balloons : Nat := 9
def sally_balloons : Nat := 5
def total_balloons : Nat := 16

-- The theorem to prove the number of balloons Jessica has
theorem jessica_balloons : (total_balloons - (joan_balloons + sally_balloons) = 2) :=
by
  -- Proof goes here
  sorry

end jessica_balloons_l307_307009


namespace symmetric_point_origin_l307_307043

-- Define the notion of symmetry with respect to the origin
def symmetric_with_origin (p : ℤ × ℤ) : ℤ × ℤ :=
  (-p.1, -p.2)

-- Define the given point
def given_point : ℤ × ℤ :=
  (-2, 5)

-- State the theorem to be proven
theorem symmetric_point_origin : 
  symmetric_with_origin given_point = (2, -5) :=
by 
  -- The proof will go here, use sorry for now
  sorry

end symmetric_point_origin_l307_307043


namespace isabella_hair_length_l307_307161

theorem isabella_hair_length (h : ℕ) (g : ℕ) (future_length : ℕ) (hg : g = 4) (future_length_eq : future_length = 22) :
  h = future_length - g :=
by
  rw [future_length_eq, hg]
  exact sorry

end isabella_hair_length_l307_307161


namespace tail_flip_probability_after_six_heads_l307_307466

-- Define the problem conditions
def john_flips_six_heads : Prop := true -- This just stands for the fact that John flipped six heads, which does not affect future flips.

-- Define the probability of flipping a tail on a fair coin
def prob_tail : ℚ := 1 / 2

-- State the theorem to be proven
theorem tail_flip_probability_after_six_heads (h : john_flips_six_heads) : 
  Prob (fair_coin = tail) = prob_tail := 
sorry

end tail_flip_probability_after_six_heads_l307_307466


namespace rationalize_denominator_l307_307034

theorem rationalize_denominator : (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_denominator_l307_307034


namespace probability_maria_given_win_l307_307000

-- Define the problem conditions in Lean
namespace Race

-- Each car's lap time uniformly distributed between 150 and 155 seconds
noncomputable def LapTimes := {i | 150 ≤ i ∧ i ≤ 155 ∧ ∃ n : ℕ, i = 150 + n ∧ n ≤ 5}.to_finset

-- Maria's lap time
def maria_lap_time : ℕ := 152
def maria_wins (maria_lap_time : ℕ) := ∀ (other_lap_time : ℕ), other_lap_time ∈ LapTimes → other_lap_time > maria_lap_time

-- Probability definition
noncomputable def uniform_prob (val : ℕ) (S : finset ℕ) : ℝ :=
  if val ∈ S then 1 / S.card else 0

#check (LapTimes.card : ℤ) -- ensure LapTimes is of size 6

lemma probability_maria_lap_time_152 : uniform_prob 152 LapTimes = 1 / 6 := by
  sorry

lemma probability_maria_wins : ∀ (t : ℕ), t ∈ LapTimes → t > 152 → (LapTimes) = 27 / 216 := by
  sorry

lemma probability_maria_wins_given_lap_time_152 : ∀ (t : ℕ), maria_wins 152 → (LapTimes) = 1 / 8 := by
  sorry

lemma probability_maria_wins_overall : (LapTimes) = 1 / 4 := by
  sorry

theorem probability_maria_given_win : 
  uniform_prob 152 LapTimes * probability_maria_wins_given_lap_time_152 / probability_maria_wins_overall = 1 / 3  
:= by lem probability_maria_lap_time_152; lem probability_maria_wins; sorry

end Race

end probability_maria_given_win_l307_307000


namespace binomial_sum_eq_sum_valid_n_values_l307_307518

theorem binomial_sum_eq (n : ℕ) (h₁ : nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16) :
  n = 14 ∨ n = 16 :=
sorry

theorem sum_valid_n_values :
  let n1 := 16
  let n2 := 14
  n1 + n2 = 30 :=
by
  -- proof to be provided; this is to check if the theorem holds
  sorry

end binomial_sum_eq_sum_valid_n_values_l307_307518
