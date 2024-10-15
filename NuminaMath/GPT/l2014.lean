import Mathlib

namespace NUMINAMATH_GPT_find_x_l2014_201480

def f (x: ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * (f x) - 10 = f (x - 2)) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2014_201480


namespace NUMINAMATH_GPT_factorize_polynomial_l2014_201482

theorem factorize_polynomial (m : ℤ) : 4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l2014_201482


namespace NUMINAMATH_GPT_radius_of_circle_l2014_201410

noncomputable def circle_radius {k : ℝ} (hk : k > -6) : ℝ := 6 * Real.sqrt 2 + 6

theorem radius_of_circle (k : ℝ) (hk : k > -6)
  (tangent_y_eq_x : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_negx : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -P.2) = 6 * Real.sqrt 2 + 6)
  (tangent_y_eq_neg6 : ∀ (P : ℝ × ℝ), P.1 = 0 → P.2 = k → dist P (0, -6) = 6 * Real.sqrt 2 + 6) :
  circle_radius hk = 6 * Real.sqrt 2 + 6 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l2014_201410


namespace NUMINAMATH_GPT_zookeeper_fish_total_l2014_201443

def fish_given : ℕ := 19
def fish_needed : ℕ := 17

theorem zookeeper_fish_total : fish_given + fish_needed = 36 :=
by
  sorry

end NUMINAMATH_GPT_zookeeper_fish_total_l2014_201443


namespace NUMINAMATH_GPT_min_correct_answers_l2014_201486

theorem min_correct_answers (x : ℕ) (hx : 10 * x - 5 * (30 - x) > 90) : x ≥ 17 :=
by {
  -- calculations and solution steps go here.
  sorry
}

end NUMINAMATH_GPT_min_correct_answers_l2014_201486


namespace NUMINAMATH_GPT_points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l2014_201405

-- Problem 1: Prove that if \(x^3 + y^3 + z^3 = (x + y + z)^3\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_cubic_eq (x y z : ℝ) (h : x^3 + y^3 + z^3 = (x + y + z)^3) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

-- Problem 2: Prove that if \(x^5 + y^5 + z^5 = (x + y + z)^5\), the points lie on the planes \(x + y = 0\), \(y + z = 0\), \(z + x = 0\).
theorem points_on_planes_of_quintic_eq (x y z : ℝ) (h : x^5 + y^5 + z^5 = (x + y + z)^5) :
  x + y = 0 ∨ y + z = 0 ∨ z + x = 0 :=
sorry

end NUMINAMATH_GPT_points_on_planes_of_cubic_eq_points_on_planes_of_quintic_eq_l2014_201405


namespace NUMINAMATH_GPT_diamond_and_face_card_probability_l2014_201478

noncomputable def probability_first_diamond_second_face_card : ℚ :=
  let total_cards := 52
  let total_faces := 12
  let diamond_faces := 3
  let diamond_non_faces := 10
  (9/52) * (12/51) + (3/52) * (11/51)

theorem diamond_and_face_card_probability :
  probability_first_diamond_second_face_card = 47 / 884 := 
by {
  sorry
}

end NUMINAMATH_GPT_diamond_and_face_card_probability_l2014_201478


namespace NUMINAMATH_GPT_sum_of_cubes_l2014_201463

theorem sum_of_cubes (p q r : ℝ) (h1 : p + q + r = 7) (h2 : p * q + p * r + q * r = 10) (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l2014_201463


namespace NUMINAMATH_GPT_number_of_eggplant_packets_l2014_201461

-- Defining the problem conditions in Lean 4
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def sunflower_packets := 6
def total_plants := 116

-- Our goal is to prove the number of eggplant seed packets Shyne bought
theorem number_of_eggplant_packets : ∃ E : ℕ, E * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants ∧ E = 4 :=
sorry

end NUMINAMATH_GPT_number_of_eggplant_packets_l2014_201461


namespace NUMINAMATH_GPT_justin_current_age_l2014_201458

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end NUMINAMATH_GPT_justin_current_age_l2014_201458


namespace NUMINAMATH_GPT_math_club_team_selection_l2014_201476

open scoped BigOperators

def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem math_club_team_selection :
  (comb 7 2 * comb 9 4) + 
  (comb 7 3 * comb 9 3) +
  (comb 7 4 * comb 9 2) +
  (comb 7 5 * comb 9 1) +
  (comb 7 6 * comb 9 0) = 7042 := 
sorry

end NUMINAMATH_GPT_math_club_team_selection_l2014_201476


namespace NUMINAMATH_GPT_number_of_boys_l2014_201475

theorem number_of_boys (n : ℕ) (handshakes : ℕ) (h_handshakes : handshakes = n * (n - 1) / 2) (h_total : handshakes = 55) : n = 11 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l2014_201475


namespace NUMINAMATH_GPT_movie_ticket_percentage_decrease_l2014_201450

theorem movie_ticket_percentage_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100) 
  (h2 : new_price = 80) :
  ((old_price - new_price) / old_price) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_movie_ticket_percentage_decrease_l2014_201450


namespace NUMINAMATH_GPT_symmetric_points_on_ellipse_are_m_in_range_l2014_201462

open Real

theorem symmetric_points_on_ellipse_are_m_in_range (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1 ∧ 
                   (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1 ∧ 
                   ∃ x0 y0 : ℝ, y0 = 4 * x0 + m ∧ x0 = (A.1 + B.1) / 2 ∧ y0 = (A.2 + B.2) / 2) 
  ↔ -2 * sqrt 13 / 13 < m ∧ m < 2 * sqrt 13 / 13 := 
 sorry

end NUMINAMATH_GPT_symmetric_points_on_ellipse_are_m_in_range_l2014_201462


namespace NUMINAMATH_GPT_probability_red_chips_drawn_first_l2014_201483

def probability_all_red_drawn (total_chips : Nat) (red_chips : Nat) (green_chips : Nat) : ℚ :=
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose (total_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem probability_red_chips_drawn_first :
  probability_all_red_drawn 9 5 4 = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_chips_drawn_first_l2014_201483


namespace NUMINAMATH_GPT_calculate_expression_l2014_201496

theorem calculate_expression : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2014_201496


namespace NUMINAMATH_GPT_spent_on_board_game_l2014_201487

theorem spent_on_board_game (b : ℕ)
  (h1 : 4 * 7 = 28)
  (h2 : b + 28 = 30) : 
  b = 2 := 
sorry

end NUMINAMATH_GPT_spent_on_board_game_l2014_201487


namespace NUMINAMATH_GPT_remainder_123456789012_div_252_l2014_201425

theorem remainder_123456789012_div_252 :
  (∃ x : ℕ, 123456789012 % 252 = x ∧ x = 204) :=
sorry

end NUMINAMATH_GPT_remainder_123456789012_div_252_l2014_201425


namespace NUMINAMATH_GPT_smallest_integer_in_ratio_l2014_201430

theorem smallest_integer_in_ratio (a b c : ℕ) 
    (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_sum : a + b + c = 100) 
    (h_ratio : c = 5 * a / 2 ∧ b = 3 * a / 2) : 
    a = 20 := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_ratio_l2014_201430


namespace NUMINAMATH_GPT_total_length_figure2_l2014_201423

-- Define the initial lengths of each segment in Figure 1.
def initial_length_horizontal1 := 5
def initial_length_vertical1 := 10
def initial_length_horizontal2 := 4
def initial_length_vertical2 := 3
def initial_length_horizontal3 := 3
def initial_length_vertical3 := 5
def initial_length_horizontal4 := 4
def initial_length_vertical_sum := 10 + 3 + 5

-- Define the transformations.
def bottom_length := initial_length_horizontal1
def rightmost_vertical_length := initial_length_vertical1 - 2
def top_horizontal_length := initial_length_horizontal2 - 3
def leftmost_vertical_length := initial_length_vertical1

-- Define the total length in Figure 2 as a theorem to be proved.
theorem total_length_figure2:
  bottom_length + rightmost_vertical_length + top_horizontal_length + leftmost_vertical_length = 24 := by
  sorry

end NUMINAMATH_GPT_total_length_figure2_l2014_201423


namespace NUMINAMATH_GPT_technician_round_trip_percentage_l2014_201481

theorem technician_round_trip_percentage
  (D : ℝ) 
  (H1 : D > 0) -- Assume D is positive
  (H2 : true) -- The technician completes the drive to the center
  (H3 : true) -- The technician completes 20% of the drive from the center
  : (1.20 * D / (2 * D)) * 100 = 60 := 
by
  simp [H1, H2, H3]
  sorry

end NUMINAMATH_GPT_technician_round_trip_percentage_l2014_201481


namespace NUMINAMATH_GPT_max_value_x_plus_y_l2014_201493

theorem max_value_x_plus_y :
  ∃ x y : ℝ, 5 * x + 3 * y ≤ 10 ∧ 3 * x + 5 * y = 15 ∧ x + y = 47 / 16 :=
by
  sorry

end NUMINAMATH_GPT_max_value_x_plus_y_l2014_201493


namespace NUMINAMATH_GPT_eddies_sister_pies_per_day_l2014_201426

theorem eddies_sister_pies_per_day 
  (Eddie_daily : ℕ := 3) 
  (Mother_daily : ℕ := 8) 
  (total_days : ℕ := 7)
  (total_pies : ℕ := 119) :
  ∃ (S : ℕ), S = 6 ∧ (Eddie_daily * total_days + Mother_daily * total_days + S * total_days = total_pies) :=
by
  sorry

end NUMINAMATH_GPT_eddies_sister_pies_per_day_l2014_201426


namespace NUMINAMATH_GPT_solve_equation_l2014_201459

theorem solve_equation :
  ∀ x : ℝ, (-x^2 = (2*x + 4) / (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intro x
  -- the proof steps would go here
  sorry

end NUMINAMATH_GPT_solve_equation_l2014_201459


namespace NUMINAMATH_GPT_linear_function_no_fourth_quadrant_l2014_201435

theorem linear_function_no_fourth_quadrant (k : ℝ) (hk : k > 2) : 
  ∀ x (hx : x > 0), (k-2) * x + k ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_no_fourth_quadrant_l2014_201435


namespace NUMINAMATH_GPT_weight_of_tin_of_cookies_l2014_201477

def weight_of_bag_of_chips := 20 -- in ounces
def weight_jasmine_carries := 336 -- converting 21 pounds to ounces
def bags_jasmine_buys := 6
def tins_multiplier := 4

theorem weight_of_tin_of_cookies 
  (weight_of_bag_of_chips : ℕ := weight_of_bag_of_chips)
  (weight_jasmine_carries : ℕ := weight_jasmine_carries)
  (bags_jasmine_buys : ℕ := bags_jasmine_buys)
  (tins_multiplier : ℕ := tins_multiplier) : 
  ℕ :=
  let total_weight_bags := bags_jasmine_buys * weight_of_bag_of_chips
  let total_weight_cookies := weight_jasmine_carries - total_weight_bags
  let num_of_tins := bags_jasmine_buys * tins_multiplier
  total_weight_cookies / num_of_tins

example : weight_of_tin_of_cookies weight_of_bag_of_chips weight_jasmine_carries bags_jasmine_buys tins_multiplier = 9 :=
by sorry

end NUMINAMATH_GPT_weight_of_tin_of_cookies_l2014_201477


namespace NUMINAMATH_GPT_find_q_l2014_201448

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_GPT_find_q_l2014_201448


namespace NUMINAMATH_GPT_solve_fractional_equation_l2014_201489

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2014_201489


namespace NUMINAMATH_GPT_smallest_X_l2014_201416

theorem smallest_X (T : ℕ) (hT_digits : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) (hX_int : ∃ (X : ℕ), T = 20 * X) : ∃ T, ∀ X, X = T / 20 → X = 55 :=
by
  sorry

end NUMINAMATH_GPT_smallest_X_l2014_201416


namespace NUMINAMATH_GPT_intersection_M_N_l2014_201402

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2014_201402


namespace NUMINAMATH_GPT_part1_part2_l2014_201470

noncomputable def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

theorem part1 (a : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → f x ≤ a) ↔ a ≥ 4 :=
sorry

theorem part2 : {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ -4 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_part1_part2_l2014_201470


namespace NUMINAMATH_GPT_student_marks_problem_l2014_201427

-- Define the variables
variables (M P C X : ℕ)

-- State the conditions
-- Condition 1: M + P = 70
def condition1 : Prop := M + P = 70

-- Condition 2: C = P + X
def condition2 : Prop := C = P + X

-- Condition 3: (M + C) / 2 = 45
def condition3 : Prop := (M + C) / 2 = 45

-- The theorem stating the problem
theorem student_marks_problem (h1 : condition1 M P) (h2 : condition2 C P X) (h3 : condition3 M C) : X = 20 :=
by sorry

end NUMINAMATH_GPT_student_marks_problem_l2014_201427


namespace NUMINAMATH_GPT_demokhar_lifespan_l2014_201495

-- Definitions based on the conditions
def boy_fraction := 1 / 4
def young_man_fraction := 1 / 5
def adult_man_fraction := 1 / 3
def old_man_years := 13

-- Statement without proof
theorem demokhar_lifespan :
  ∀ (x : ℕ), (boy_fraction * x) + (young_man_fraction * x) + (adult_man_fraction * x) + old_man_years = x → x = 60 :=
by
  sorry

end NUMINAMATH_GPT_demokhar_lifespan_l2014_201495


namespace NUMINAMATH_GPT_factor_expression_l2014_201453

variable (x : ℝ)

-- Mathematically define the expression e
def e : ℝ := 4 * x * (x + 2) + 10 * (x + 2) + 2 * (x + 2)

-- State that e is equivalent to the factored form
theorem factor_expression : e x = (x + 2) * (4 * x + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2014_201453


namespace NUMINAMATH_GPT_total_fence_poles_needed_l2014_201473

def number_of_poles_per_side := 27

theorem total_fence_poles_needed (n : ℕ) (h : n = number_of_poles_per_side) : 
  4 * n - 4 = 104 :=
by sorry

end NUMINAMATH_GPT_total_fence_poles_needed_l2014_201473


namespace NUMINAMATH_GPT_max_a_plus_b_l2014_201421

theorem max_a_plus_b (a b c d e : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : a + 2*b + 3*c + 4*d + 5*e = 300) : a + b ≤ 35 :=
sorry

end NUMINAMATH_GPT_max_a_plus_b_l2014_201421


namespace NUMINAMATH_GPT_largest_constant_l2014_201446

def equation_constant (c d : ℝ) : ℝ :=
  5 * c + (d - 12)^2

theorem largest_constant : ∃ constant : ℝ, (∀ c, c ≤ 47) → (∀ d, equation_constant 47 d = constant) → constant = 235 := 
by
  sorry

end NUMINAMATH_GPT_largest_constant_l2014_201446


namespace NUMINAMATH_GPT_sum_of_babies_ages_in_five_years_l2014_201491

-- Given Definitions
def lioness_age := 12
def hyena_age := lioness_age / 2
def lioness_baby_age := lioness_age / 2
def hyena_baby_age := hyena_age / 2

-- The declaration of the statement to be proven
theorem sum_of_babies_ages_in_five_years : (lioness_baby_age + 5) + (hyena_baby_age + 5) = 19 :=
by 
  sorry 

end NUMINAMATH_GPT_sum_of_babies_ages_in_five_years_l2014_201491


namespace NUMINAMATH_GPT_sum_of_coefficients_l2014_201409

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (2 * x + 1)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 →
  a₀ = 1 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 :=
by
  intros h_expand h_a₀
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2014_201409


namespace NUMINAMATH_GPT_Jim_paycheck_correct_l2014_201432

noncomputable def Jim_paycheck_after_deductions (gross_pay : ℝ) (retirement_percentage : ℝ) (tax_deduction : ℝ) : ℝ :=
  gross_pay - (gross_pay * retirement_percentage) - tax_deduction

theorem Jim_paycheck_correct :
  Jim_paycheck_after_deductions 1120 0.25 100 = 740 :=
by sorry

end NUMINAMATH_GPT_Jim_paycheck_correct_l2014_201432


namespace NUMINAMATH_GPT_little_john_spent_on_sweets_l2014_201420

theorem little_john_spent_on_sweets:
  let initial_amount := 10.10
  let amount_given_to_each_friend := 2.20
  let amount_left := 2.45
  let total_given_to_friends := 2 * amount_given_to_each_friend
  let amount_before_sweets := initial_amount - total_given_to_friends
  let amount_spent_on_sweets := amount_before_sweets - amount_left
  amount_spent_on_sweets = 3.25 :=
by
  sorry

end NUMINAMATH_GPT_little_john_spent_on_sweets_l2014_201420


namespace NUMINAMATH_GPT_ratio_Mandy_to_Pamela_l2014_201444

-- Definitions based on conditions in the problem
def exam_items : ℕ := 100
def Lowella_correct : ℕ := (35 * exam_items) / 100  -- 35% of 100
def Pamela_correct : ℕ := Lowella_correct + (20 * Lowella_correct) / 100 -- 20% more than Lowella
def Mandy_score : ℕ := 84

-- The proof problem statement
theorem ratio_Mandy_to_Pamela : Mandy_score / Pamela_correct = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_Mandy_to_Pamela_l2014_201444


namespace NUMINAMATH_GPT_proof_problem_l2014_201488

theorem proof_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l2014_201488


namespace NUMINAMATH_GPT_smallest_class_size_l2014_201467

theorem smallest_class_size (N : ℕ) (G : ℕ) (h1: 0.25 < (G : ℝ) / N) (h2: (G : ℝ) / N < 0.30) : N = 7 := 
sorry

end NUMINAMATH_GPT_smallest_class_size_l2014_201467


namespace NUMINAMATH_GPT_count_edge_cubes_l2014_201417

/-- 
A cube is painted red on all faces and then cut into 27 equal smaller cubes.
Prove that the number of smaller cubes that are painted on only 2 faces is 12. 
-/
theorem count_edge_cubes (c : ℕ) (inner : ℕ)  (edge : ℕ) (face : ℕ) :
  (c = 27 ∧ inner = 1 ∧ edge = 12 ∧ face = 6) → edge = 12 :=
by
  -- Given the conditions from the problem statement
  sorry

end NUMINAMATH_GPT_count_edge_cubes_l2014_201417


namespace NUMINAMATH_GPT_tan_double_angle_l2014_201415

theorem tan_double_angle (α : ℝ) (h : Real.tan α = 1 / 3) : Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l2014_201415


namespace NUMINAMATH_GPT_sub_numbers_correct_l2014_201455

theorem sub_numbers_correct : 
  (500.50 - 123.45 - 55 : ℝ) = 322.05 := by 
-- The proof can be filled in here
sorry

end NUMINAMATH_GPT_sub_numbers_correct_l2014_201455


namespace NUMINAMATH_GPT_proof_x_y_l2014_201436

noncomputable def x_y_problem (x y : ℝ) : Prop :=
  (x^2 = 9) ∧ (|y| = 4) ∧ (x < y) → (x - y = -1 ∨ x - y = -7)

theorem proof_x_y (x y : ℝ) : x_y_problem x y :=
by
  sorry

end NUMINAMATH_GPT_proof_x_y_l2014_201436


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2014_201447

theorem solution_set_of_inequality (a b x : ℝ) (h1 : 0 < a) (h2 : b = 2 * a) : ax > b ↔ x > -2 :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2014_201447


namespace NUMINAMATH_GPT_min_moves_to_reassemble_l2014_201465

theorem min_moves_to_reassemble (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, (∀ pieces, pieces = n - 1) ∧ pieces = 1 → move_count = n - 1 :=
by
  sorry

end NUMINAMATH_GPT_min_moves_to_reassemble_l2014_201465


namespace NUMINAMATH_GPT_rectangle_length_to_width_ratio_l2014_201445

variables (s : ℝ)

-- Given conditions
def small_square_side := s
def large_square_side := 3 * s
def rectangle_length := large_square_side
def rectangle_width := large_square_side - 2 * small_square_side

-- Theorem to prove the ratio of the length to the width of the rectangle
theorem rectangle_length_to_width_ratio : 
  ∃ (r : ℝ), r = rectangle_length s / rectangle_width s ∧ r = 3 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_length_to_width_ratio_l2014_201445


namespace NUMINAMATH_GPT_original_ribbon_length_l2014_201414

theorem original_ribbon_length :
  ∃ x : ℝ, 
    (∀ a b : ℝ, 
       a = x - 18 ∧ 
       b = x - 12 ∧ 
       b = 2 * a → x = 24) :=
by
  sorry

end NUMINAMATH_GPT_original_ribbon_length_l2014_201414


namespace NUMINAMATH_GPT_initial_money_amount_l2014_201498

theorem initial_money_amount (M : ℝ)
  (h_clothes : M * (1 / 3) = c)
  (h_food : (M - c) * (1 / 5) = f)
  (h_travel : (M - c - f) * (1 / 4) = t)
  (h_remaining : M - c - f - t = 600) : M = 1500 := by
  sorry

end NUMINAMATH_GPT_initial_money_amount_l2014_201498


namespace NUMINAMATH_GPT_find_number_l2014_201434

theorem find_number (N : ℝ) (h : 0.15 * 0.30 * 0.50 * N = 108) : N = 4800 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2014_201434


namespace NUMINAMATH_GPT_cost_price_one_meter_l2014_201466

theorem cost_price_one_meter (selling_price : ℤ) (total_meters : ℤ) (profit_per_meter : ℤ) 
  (h1 : selling_price = 6788) (h2 : total_meters = 78) (h3 : profit_per_meter = 29) : 
  (selling_price - (profit_per_meter * total_meters)) / total_meters = 58 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_one_meter_l2014_201466


namespace NUMINAMATH_GPT_fraction_cubed_equality_l2014_201440

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end NUMINAMATH_GPT_fraction_cubed_equality_l2014_201440


namespace NUMINAMATH_GPT_gcd_38_23_is_1_l2014_201451

theorem gcd_38_23_is_1 : Nat.gcd 38 23 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_38_23_is_1_l2014_201451


namespace NUMINAMATH_GPT_valid_three_digit_numbers_count_l2014_201494

def is_prime_or_even (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def count_valid_numbers : ℕ :=
  (4 * 4) -- number of valid combinations for hundreds and tens digits

theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by 
  -- outline the structure of the proof here, but we use sorry to indicate the proof is not complete
  sorry

end NUMINAMATH_GPT_valid_three_digit_numbers_count_l2014_201494


namespace NUMINAMATH_GPT_no_linear_factor_l2014_201429

theorem no_linear_factor : ∀ x y z : ℤ,
  ¬ ∃ a b c : ℤ, a*x + b*y + c*z + (x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z) = 0 :=
by sorry

end NUMINAMATH_GPT_no_linear_factor_l2014_201429


namespace NUMINAMATH_GPT_division_of_field_l2014_201472

theorem division_of_field :
  (∀ (hectares : ℕ) (parts : ℕ), hectares = 5 ∧ parts = 8 →
  (1 / parts = 1 / 8) ∧ (hectares / parts = 5 / 8)) :=
by
  sorry


end NUMINAMATH_GPT_division_of_field_l2014_201472


namespace NUMINAMATH_GPT_part_a_l2014_201439

theorem part_a (p : ℕ → ℕ → ℝ) (m : ℕ) (hm : m ≥ 1) : p m 0 = (3 / 4) * p (m-1) 0 + (1 / 2) * p (m-1) 2 + (1 / 8) * p (m-1) 4 :=
by
  sorry

end NUMINAMATH_GPT_part_a_l2014_201439


namespace NUMINAMATH_GPT_longest_tape_l2014_201474

theorem longest_tape (r b y : ℚ) (h₀ : r = 11 / 6) (h₁ : b = 7 / 4) (h₂ : y = 13 / 8) : r > b ∧ r > y :=
by 
  sorry

end NUMINAMATH_GPT_longest_tape_l2014_201474


namespace NUMINAMATH_GPT_line_quadrant_conditions_l2014_201441

theorem line_quadrant_conditions (k b : ℝ) 
  (H1 : ∃ x : ℝ, x > 0 ∧ k * x + b > 0)
  (H3 : ∃ x : ℝ, x < 0 ∧ k * x + b < 0)
  (H4 : ∃ x : ℝ, x > 0 ∧ k * x + b < 0) : k > 0 ∧ b < 0 :=
sorry

end NUMINAMATH_GPT_line_quadrant_conditions_l2014_201441


namespace NUMINAMATH_GPT_polynomial_divisor_l2014_201479

theorem polynomial_divisor (f : Polynomial ℂ) (n : ℕ) (h : (X - 1) ∣ (f.comp (X ^ n))) : (X ^ n - 1) ∣ (f.comp (X ^ n)) :=
sorry

end NUMINAMATH_GPT_polynomial_divisor_l2014_201479


namespace NUMINAMATH_GPT_union_of_A_and_B_l2014_201412

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 5} := 
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2014_201412


namespace NUMINAMATH_GPT_team_total_points_l2014_201497

theorem team_total_points : 
  ∀ (Tobee Jay Sean : ℕ),
  (Tobee = 4) →
  (Jay = Tobee + 6) →
  (Sean = Tobee + Jay - 2) →
  (Tobee + Jay + Sean = 26) :=
by
  intros Tobee Jay Sean h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_team_total_points_l2014_201497


namespace NUMINAMATH_GPT_correct_calculation_l2014_201437

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2014_201437


namespace NUMINAMATH_GPT_sum_two_numbers_in_AP_and_GP_equals_20_l2014_201433

theorem sum_two_numbers_in_AP_and_GP_equals_20 :
  ∃ a b : ℝ, 
    (a > 0) ∧ (b > 0) ∧ 
    (4 < a) ∧ (a < b) ∧ 
    (4 + (a - 4) = a) ∧ (4 + 2 * (a - 4) = b) ∧
    (a * (b / a) = b) ∧ (b * (b / a) = 16) ∧ 
    a + b = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_two_numbers_in_AP_and_GP_equals_20_l2014_201433


namespace NUMINAMATH_GPT_jimmy_paid_total_l2014_201442

-- Data for the problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100
def park_pizzas : ℕ := 3
def building_distance : ℕ := 2000
def building_pizzas : ℕ := 2
def house_distance : ℕ := 800
def house_pizzas : ℕ := 4
def community_center_distance : ℕ := 1500
def community_center_pizzas : ℕ := 5
def office_distance : ℕ := 300
def office_pizzas : ℕ := 1
def bus_stop_distance : ℕ := 1200
def bus_stop_pizzas : ℕ := 3

def cost (distance pizzas : ℕ) : ℕ := 
  let base_cost := pizzas * pizza_cost
  if distance > 1000 then base_cost + delivery_charge else base_cost

def total_cost : ℕ :=
  cost park_distance park_pizzas +
  cost building_distance building_pizzas +
  cost house_distance house_pizzas +
  cost community_center_distance community_center_pizzas +
  cost office_distance office_pizzas +
  cost bus_stop_distance bus_stop_pizzas

theorem jimmy_paid_total : total_cost = 222 :=
  by
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_jimmy_paid_total_l2014_201442


namespace NUMINAMATH_GPT_remaining_calories_proof_l2014_201418

def volume_of_rectangular_block (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_cube (side : ℝ) : ℝ :=
  side * side * side

def remaining_volume (initial_volume eaten_volume : ℝ) : ℝ :=
  initial_volume - eaten_volume

def remaining_calories (remaining_volume calorie_density : ℝ) : ℝ :=
  remaining_volume * calorie_density

theorem remaining_calories_proof :
  let calorie_density := 110
  let original_length := 4
  let original_width := 8
  let original_height := 2
  let cube_side := 2
  let original_volume := volume_of_rectangular_block original_length original_width original_height
  let eaten_volume := volume_of_cube cube_side
  let remaining_vol := remaining_volume original_volume eaten_volume
  let resulting_calories := remaining_calories remaining_vol calorie_density
  resulting_calories = 6160 := by
  repeat { sorry }

end NUMINAMATH_GPT_remaining_calories_proof_l2014_201418


namespace NUMINAMATH_GPT_complement_of_A_l2014_201471

def A : Set ℝ := { x | x^2 - x ≥ 0 }
def R_complement_A : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem complement_of_A :
  ∀ x : ℝ, x ∈ R_complement_A ↔ x ∉ A :=
sorry

end NUMINAMATH_GPT_complement_of_A_l2014_201471


namespace NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l2014_201424

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l2014_201424


namespace NUMINAMATH_GPT_solve_x_plus_y_l2014_201492

variable {x y : ℚ} -- Declare x and y as rational numbers

theorem solve_x_plus_y
  (h1: (1 / x) + (1 / y) = 1)
  (h2: (1 / x) - (1 / y) = 5) :
  x + y = -1 / 6 :=
sorry

end NUMINAMATH_GPT_solve_x_plus_y_l2014_201492


namespace NUMINAMATH_GPT_inequality_proof_l2014_201484

variable (a b c d : ℝ)

theorem inequality_proof
  (h_pos: 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1)
  (h_product: a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2014_201484


namespace NUMINAMATH_GPT_solve_equation_l2014_201401

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ -1) :
  -x^2 = (4 * x + 2) / (x^2 + 3 * x + 2) ↔ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2014_201401


namespace NUMINAMATH_GPT_solution1_solution2_l2014_201460

-- Define the first problem
def equation1 (x : ℝ) : Prop :=
  (x + 1) / 3 - 1 = (x - 1) / 2

-- Prove that x = -1 is the solution to the first problem
theorem solution1 : equation1 (-1) := 
by 
  sorry

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  x - y = 1 ∧ 3 * x + y = 7

-- Prove that x = 2 and y = 1 are the solutions to the system of equations
theorem solution2 : system_of_equations 2 1 :=
by 
  sorry

end NUMINAMATH_GPT_solution1_solution2_l2014_201460


namespace NUMINAMATH_GPT_ellipse_eccentricity_range_l2014_201449

theorem ellipse_eccentricity_range (a b : ℝ) (h : a > b) (h_b : b > 0) : 
  ∃ e : ℝ, (e = (Real.sqrt (a^2 - b^2)) / a) ∧ (e > 1/2 ∧ e < 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_range_l2014_201449


namespace NUMINAMATH_GPT_sector_angle_radian_measure_l2014_201411

theorem sector_angle_radian_measure (r l : ℝ) (h1 : r = 1) (h2 : l = 2) : l / r = 2 := by
  sorry

end NUMINAMATH_GPT_sector_angle_radian_measure_l2014_201411


namespace NUMINAMATH_GPT_triangle_side_length_l2014_201468

theorem triangle_side_length (a b c : ℝ) (B : ℝ) (ha : a = 2) (hB : B = 60) (hc : c = 3) :
  b = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l2014_201468


namespace NUMINAMATH_GPT_measure_of_angle_C_l2014_201456

theorem measure_of_angle_C
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := 
sorry

end NUMINAMATH_GPT_measure_of_angle_C_l2014_201456


namespace NUMINAMATH_GPT_smallest_prime_sum_l2014_201438

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_distinct_primes (n k : ℕ) (s : List ℕ) : Prop :=
  s.length = k ∧ (∀ x ∈ s, is_prime x) ∧ (∀ (x y : ℕ), x ≠ y → x ∈ s → y ∈ s → x ≠ y) ∧ s.sum = n

theorem smallest_prime_sum :
  (is_prime 61) ∧ 
  (∃ s2, is_sum_of_distinct_primes 61 2 s2) ∧ 
  (∃ s3, is_sum_of_distinct_primes 61 3 s3) ∧ 
  (∃ s4, is_sum_of_distinct_primes 61 4 s4) ∧ 
  (∃ s5, is_sum_of_distinct_primes 61 5 s5) ∧ 
  (∃ s6, is_sum_of_distinct_primes 61 6 s6) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_sum_l2014_201438


namespace NUMINAMATH_GPT_find_f_2015_l2014_201428

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_periodic_2 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom f_at_1 : f 1 = 2

theorem find_f_2015 : f 2015 = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2015_l2014_201428


namespace NUMINAMATH_GPT_kyle_money_left_l2014_201485

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end NUMINAMATH_GPT_kyle_money_left_l2014_201485


namespace NUMINAMATH_GPT_solve_polynomial_relation_l2014_201408

--Given Conditions
def polynomial_relation (x y : ℤ) : Prop := y^3 = x^3 + 8 * x^2 - 6 * x + 8 

--Proof Problem
theorem solve_polynomial_relation : ∃ (x y : ℤ), (polynomial_relation x y) ∧ 
  ((y = 11 ∧ x = 9) ∨ (y = 2 ∧ x = 0)) :=
by 
  sorry

end NUMINAMATH_GPT_solve_polynomial_relation_l2014_201408


namespace NUMINAMATH_GPT_four_digit_num_exists_l2014_201452

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem four_digit_num_exists :
  ∃ (n : ℕ), (is_two_digit (n / 100)) ∧ (is_two_digit (n % 100)) ∧
  ((n / 100) + (n % 100))^2 = 100 * (n / 100) + (n % 100) :=
by
  sorry

end NUMINAMATH_GPT_four_digit_num_exists_l2014_201452


namespace NUMINAMATH_GPT_quadratic_root_square_of_another_l2014_201490

theorem quadratic_root_square_of_another (a : ℚ) :
  (∃ x y : ℚ, x^2 - (15/4) * x + a^3 = 0 ∧ (x = y^2 ∨ y = x^2) ∧ (x*y = a^3)) →
  (a = 3/2 ∨ a = -5/2) :=
sorry

end NUMINAMATH_GPT_quadratic_root_square_of_another_l2014_201490


namespace NUMINAMATH_GPT_problem_l2014_201403

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1 else S n - S (n - 1)

def sum_abs_a_10 : ℤ :=
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|)

theorem problem : sum_abs_a_10 = 67 := by
  sorry

end NUMINAMATH_GPT_problem_l2014_201403


namespace NUMINAMATH_GPT_roger_earned_correct_amount_l2014_201454

def small_lawn_rate : ℕ := 9
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def initial_small_lawns : ℕ := 5
def initial_medium_lawns : ℕ := 4
def initial_large_lawns : ℕ := 5

def forgot_small_lawns : ℕ := 2
def forgot_medium_lawns : ℕ := 3
def forgot_large_lawns : ℕ := 3

def actual_small_lawns := initial_small_lawns - forgot_small_lawns
def actual_medium_lawns := initial_medium_lawns - forgot_medium_lawns
def actual_large_lawns := initial_large_lawns - forgot_large_lawns

def money_earned_small := actual_small_lawns * small_lawn_rate
def money_earned_medium := actual_medium_lawns * medium_lawn_rate
def money_earned_large := actual_large_lawns * large_lawn_rate

def total_money_earned := money_earned_small + money_earned_medium + money_earned_large

theorem roger_earned_correct_amount : total_money_earned = 69 := by
  sorry

end NUMINAMATH_GPT_roger_earned_correct_amount_l2014_201454


namespace NUMINAMATH_GPT_inequality_solution_l2014_201499

theorem inequality_solution (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_l2014_201499


namespace NUMINAMATH_GPT_basketball_problem_l2014_201404

theorem basketball_problem :
  ∃ x y : ℕ, (3 + x + y = 14) ∧ (3 * 3 + 2 * x + y = 28) ∧ (x = 8) ∧ (y = 3) :=
by
  sorry

end NUMINAMATH_GPT_basketball_problem_l2014_201404


namespace NUMINAMATH_GPT_evaluate_expression_l2014_201457

-- Define the given numbers as real numbers
def x : ℝ := 175.56
def y : ℝ := 54321
def z : ℝ := 36947
def w : ℝ := 1521

-- State the theorem to be proved
theorem evaluate_expression : (x / y) * (z / w) = 0.07845 :=
by 
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2014_201457


namespace NUMINAMATH_GPT_red_or_blue_probability_is_half_l2014_201464

-- Define the number of each type of marble
def num_red_marbles : ℕ := 3
def num_blue_marbles : ℕ := 2
def num_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ := num_red_marbles + num_blue_marbles + num_yellow_marbles

-- Define the number of marbles that are either red or blue
def num_red_or_blue_marbles : ℕ := num_red_marbles + num_blue_marbles

-- Define the probability of drawing a red or blue marble
def probability_red_or_blue : ℚ := num_red_or_blue_marbles / total_marbles

-- Theorem stating the probability is 0.5
theorem red_or_blue_probability_is_half : probability_red_or_blue = 0.5 := by
  sorry

end NUMINAMATH_GPT_red_or_blue_probability_is_half_l2014_201464


namespace NUMINAMATH_GPT_chess_game_problem_l2014_201422

-- Mathematical definitions based on the conditions
def petr_wins : ℕ := 6
def petr_draws : ℕ := 2
def karel_points : ℤ := 9
def points_for_win : ℕ := 3
def points_for_loss : ℕ := 2
def points_for_draw : ℕ := 0

-- Defining the final statement to prove
theorem chess_game_problem :
    ∃ (total_games : ℕ) (leader : String), total_games = 15 ∧ leader = "Karel" := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_chess_game_problem_l2014_201422


namespace NUMINAMATH_GPT_f_zero_eq_one_f_positive_f_increasing_f_range_x_l2014_201419

noncomputable def f : ℝ → ℝ := sorry
axiom f_condition1 : f 0 ≠ 0
axiom f_condition2 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_condition3 : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_positive : ∀ x : ℝ, f x > 0 :=
sorry

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
sorry

theorem f_range_x (x : ℝ) (h : f x * f (2 * x - x^2) > 1) : x ∈ { x : ℝ | f x > 1 ∧ f (2 * x - x^2) > 1 } :=
sorry

end NUMINAMATH_GPT_f_zero_eq_one_f_positive_f_increasing_f_range_x_l2014_201419


namespace NUMINAMATH_GPT_find_N_l2014_201469

theorem find_N (x y : ℕ) (N : ℕ) (h1 : N = x * (x + 9)) (h2 : N = y * (y + 6)) : 
  N = 112 :=
  sorry

end NUMINAMATH_GPT_find_N_l2014_201469


namespace NUMINAMATH_GPT_equilateral_triangle_ratio_l2014_201431

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let perimeter := 3 * s
  let area := (s * s * Real.sqrt 3) / 4
  perimeter / area = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_ratio_l2014_201431


namespace NUMINAMATH_GPT_find_z_l2014_201406

-- Definitions from the problem statement
variables {x y z : ℤ}
axiom consecutive (h1: x = z + 2) (h2: y = z + 1) : true
axiom ordered (h3: x > y) (h4: y > z) : true
axiom equation (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : true

-- The proof goal
theorem find_z (h1: x = z + 2) (h2: y = z + 1) (h3: x > y) (h4: y > z) (h5: 2 * x + 3 * y + 3 * z = 5 * y + 8) : z = 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_z_l2014_201406


namespace NUMINAMATH_GPT_sum_reciprocals_geom_seq_l2014_201407

theorem sum_reciprocals_geom_seq (a₁ q : ℝ) (h_pos_a₁ : 0 < a₁) (h_pos_q : 0 < q)
    (h_sum : a₁ + a₁ * q + a₁ * q^2 + a₁ * q^3 = 9)
    (h_prod : a₁^4 * q^6 = 81 / 4) :
    (1 / a₁) + (1 / (a₁ * q)) + (1 / (a₁ * q^2)) + (1 / (a₁ * q^3)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_geom_seq_l2014_201407


namespace NUMINAMATH_GPT_kim_money_l2014_201413

theorem kim_money (S P K A : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : A = 1.25 * (S + K)) (h4 : S + P + A = 3.60) : K = 0.96 :=
by
  sorry

end NUMINAMATH_GPT_kim_money_l2014_201413


namespace NUMINAMATH_GPT_units_digit_is_valid_l2014_201400

theorem units_digit_is_valid (n : ℕ) : 
  (∃ k : ℕ, (k^3 % 10 = n)) → 
  (n = 2 ∨ n = 3 ∨ n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end NUMINAMATH_GPT_units_digit_is_valid_l2014_201400
