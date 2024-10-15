import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1526_152688

def proposition_P (a : ℝ) := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_Q (a : ℝ) := 5 - 2*a > 1

theorem range_of_a :
  (∃! (p : Prop), (p = proposition_P a ∨ p = proposition_Q a) ∧ p) →
  a ∈ Set.Iic (-2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1526_152688


namespace NUMINAMATH_GPT_max_profit_l1526_152653

-- Define the conditions
def profit (m : ℝ) := (m - 8) * (900 - 15 * m)

-- State the theorem
theorem max_profit (m : ℝ) : 
  ∃ M, M = profit 34 ∧ ∀ x, profit x ≤ M :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_max_profit_l1526_152653


namespace NUMINAMATH_GPT_correct_transformation_l1526_152648

theorem correct_transformation (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
sorry

end NUMINAMATH_GPT_correct_transformation_l1526_152648


namespace NUMINAMATH_GPT_museum_ticket_cost_l1526_152681

theorem museum_ticket_cost 
  (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_cost : ℕ) (teacher_ticket_cost : ℕ)
  (h_students : num_students = 12) (h_teachers : num_teachers = 4)
  (h_student_cost : student_ticket_cost = 1) (h_teacher_cost : teacher_ticket_cost = 3) :
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end NUMINAMATH_GPT_museum_ticket_cost_l1526_152681


namespace NUMINAMATH_GPT_wendy_time_correct_l1526_152606

noncomputable section

def bonnie_time : ℝ := 7.80
def wendy_margin : ℝ := 0.25

theorem wendy_time_correct : (bonnie_time - wendy_margin) = 7.55 := by
  sorry

end NUMINAMATH_GPT_wendy_time_correct_l1526_152606


namespace NUMINAMATH_GPT_function_inequality_l1526_152617

variable {f : ℕ → ℝ}
variable {a : ℝ}

theorem function_inequality (h : ∀ n : ℕ, f (n + 1) ≥ a^n * f n) :
  ∀ n : ℕ, f n = a^((n * (n - 1)) / 2) * f 1 := 
sorry

end NUMINAMATH_GPT_function_inequality_l1526_152617


namespace NUMINAMATH_GPT_distance_between_points_A_and_B_is_240_l1526_152628

noncomputable def distance_between_A_and_B (x y : ℕ) : ℕ := 6 * x * 2

theorem distance_between_points_A_and_B_is_240 (x y : ℕ)
  (h1 : 6 * x = 6 * y)
  (h2 : 5 * (x + 4) = 6 * y) :
  distance_between_A_and_B x y = 240 := by
  sorry

end NUMINAMATH_GPT_distance_between_points_A_and_B_is_240_l1526_152628


namespace NUMINAMATH_GPT_pow_div_l1526_152655

theorem pow_div (a : ℝ) : (-a) ^ 6 / a ^ 3 = a ^ 3 := by
  sorry

end NUMINAMATH_GPT_pow_div_l1526_152655


namespace NUMINAMATH_GPT_find_k_l1526_152635

theorem find_k 
  (k : ℝ)
  (p_eq : ∀ x : ℝ, (4 * x + 3 = k * x - 9) → (x = -3 → (k = 0)))
: k = 0 :=
by sorry

end NUMINAMATH_GPT_find_k_l1526_152635


namespace NUMINAMATH_GPT_rachel_pizza_eaten_l1526_152636

theorem rachel_pizza_eaten (pizza_total : ℕ) (pizza_bella : ℕ) (pizza_rachel : ℕ) :
  pizza_total = pizza_bella + pizza_rachel → pizza_bella = 354 → pizza_total = 952 → pizza_rachel = 598 :=
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  sorry

end NUMINAMATH_GPT_rachel_pizza_eaten_l1526_152636


namespace NUMINAMATH_GPT_age_of_new_person_l1526_152669

theorem age_of_new_person (avg_age : ℝ) (x : ℝ) 
  (h1 : 10 * avg_age - (10 * (avg_age - 3)) = 42 - x) : 
  x = 12 := 
by
  sorry

end NUMINAMATH_GPT_age_of_new_person_l1526_152669


namespace NUMINAMATH_GPT_range_of_a_l1526_152619

noncomputable def proposition_p (a : ℝ) : Prop := 
  0 < a ∧ a < 1

noncomputable def proposition_q (a : ℝ) : Prop := 
  a > 1 / 4

theorem range_of_a (a : ℝ) : 
  (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) ↔
  (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1526_152619


namespace NUMINAMATH_GPT_principal_amount_l1526_152610

variable (P : ℝ)
variable (R : ℝ := 4)
variable (T : ℝ := 5)

theorem principal_amount :
  ((P * R * T) / 100 = P - 2000) → P = 2500 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l1526_152610


namespace NUMINAMATH_GPT_total_students_correct_l1526_152634

-- Define the number of students who play football, cricket, both and neither.
def play_football : ℕ := 325
def play_cricket : ℕ := 175
def play_both : ℕ := 90
def play_neither : ℕ := 50

-- Define the total number of students
def total_students : ℕ := play_football + play_cricket - play_both + play_neither

-- Prove that the total number of students is 460 given the conditions
theorem total_students_correct : total_students = 460 := by
  sorry

end NUMINAMATH_GPT_total_students_correct_l1526_152634


namespace NUMINAMATH_GPT_modular_inverse_addition_l1526_152607

theorem modular_inverse_addition :
  (3 * 9 + 9 * 37) % 63 = 45 :=
by
  sorry

end NUMINAMATH_GPT_modular_inverse_addition_l1526_152607


namespace NUMINAMATH_GPT_multiplicative_inverse_of_AB_l1526_152618

def A : ℕ := 222222
def B : ℕ := 476190
def N : ℕ := 189
def modulus : ℕ := 1000000

theorem multiplicative_inverse_of_AB :
  (A * B * N) % modulus = 1 % modulus :=
by
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_of_AB_l1526_152618


namespace NUMINAMATH_GPT_exterior_angle_of_triangle_cond_40_degree_l1526_152695

theorem exterior_angle_of_triangle_cond_40_degree (A B C : ℝ)
  (h1 : (A = 40 ∨ B = 40 ∨ C = 40))
  (h2 : A = B)
  (h3 : A + B + C = 180) :
  ((180 - C) = 80 ∨ (180 - C) = 140) :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_of_triangle_cond_40_degree_l1526_152695


namespace NUMINAMATH_GPT_sanctuary_feeding_ways_l1526_152658

/-- A sanctuary houses six different pairs of animals, each pair consisting of a male and female.
  The caretaker must feed the animals alternately by gender, meaning no two animals of the same gender 
  can be fed consecutively. Given the additional constraint that the male giraffe cannot be fed 
  immediately before the female giraffe and that the feeding starts with the male lion, 
  there are exactly 7200 valid ways to complete the feeding. -/
theorem sanctuary_feeding_ways : 
  ∃ ways : ℕ, ways = 7200 :=
by sorry

end NUMINAMATH_GPT_sanctuary_feeding_ways_l1526_152658


namespace NUMINAMATH_GPT_value_of_expression_l1526_152624

theorem value_of_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1526_152624


namespace NUMINAMATH_GPT_Abby_in_seat_3_l1526_152689

variables (P : Type) [Inhabited P]
variables (Abby Bret Carl Dana : P)
variables (seat : P → ℕ)

-- Conditions from the problem:
-- Bret is actually sitting in seat #2.
axiom Bret_in_seat_2 : seat Bret = 2

-- False statement 1: Dana is next to Bret.
axiom false_statement_1 : ¬ (seat Dana = 1 ∨ seat Dana = 3)

-- False statement 2: Carl is sitting between Dana and Bret.
axiom false_statement_2 : ¬ (seat Carl = 1)

-- The final translated proof problem:
theorem Abby_in_seat_3 : seat Abby = 3 :=
sorry

end NUMINAMATH_GPT_Abby_in_seat_3_l1526_152689


namespace NUMINAMATH_GPT_common_tangent_line_range_a_l1526_152614

open Real

theorem common_tangent_line_range_a (a : ℝ) (h_pos : 0 < a) :
  (∃ x₁ x₂ : ℝ, 2 * a * x₁ = exp x₂ ∧ (exp x₂ - a * x₁^2) / (x₂ - x₁) = 2 * a * x₁) →
  a ≥ exp 2 / 4 := 
sorry

end NUMINAMATH_GPT_common_tangent_line_range_a_l1526_152614


namespace NUMINAMATH_GPT_trigonometric_identity_l1526_152630

variable (α : ℝ)

theorem trigonometric_identity :
  4.9 * (Real.sin (7 * Real.pi / 8 - 2 * α))^2 - (Real.sin (9 * Real.pi / 8 - 2 * α))^2 = 
  Real.sin (4 * α) / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1526_152630


namespace NUMINAMATH_GPT_max_boxes_fit_l1526_152661

theorem max_boxes_fit 
  (L_large W_large H_large : ℕ) 
  (L_small W_small H_small : ℕ) 
  (h1 : L_large = 12) 
  (h2 : W_large = 14) 
  (h3 : H_large = 16) 
  (h4 : L_small = 3) 
  (h5 : W_small = 7) 
  (h6 : H_small = 2) 
  : ((L_large * W_large * H_large) / (L_small * W_small * H_small) = 64) :=
by
  sorry

end NUMINAMATH_GPT_max_boxes_fit_l1526_152661


namespace NUMINAMATH_GPT_find_a_plus_b_l1526_152693

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := -1

noncomputable def l : ℝ := -1 -- Slope of line l (since angle is 3π/4)

noncomputable def l1_slope : ℝ := 1 -- Slope of line l1 which is perpendicular to l

noncomputable def a : ℝ := 0 -- Calculated from k_{AB} = 1

noncomputable def b : ℝ := -2 -- Calculated from line parallel condition

theorem find_a_plus_b : a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1526_152693


namespace NUMINAMATH_GPT_sequence_inequality_l1526_152604

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
  (h_condition : ∀ n m, a (n + m) ≤ a n + a m) :
  ∀ n m, n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := by
  sorry

end NUMINAMATH_GPT_sequence_inequality_l1526_152604


namespace NUMINAMATH_GPT_basketball_team_win_rate_l1526_152647

theorem basketball_team_win_rate (won_first : ℕ) (total : ℕ) (remaining : ℕ)
    (desired_rate : ℚ) (x : ℕ) (H_won : won_first = 30) (H_total : total = 100)
    (H_remaining : remaining = 55) (H_desired : desired_rate = 13/20) :
    (30 + x) / 100 = 13 / 20 ↔ x = 35 := by
    sorry

end NUMINAMATH_GPT_basketball_team_win_rate_l1526_152647


namespace NUMINAMATH_GPT_sum_of_numbers_l1526_152668

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a - 2)^2 - (b - 2)^2 = 18): 
  a + b = -2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1526_152668


namespace NUMINAMATH_GPT_parallelogram_angle_B_l1526_152690

theorem parallelogram_angle_B (A C B D : ℝ) (h₁ : A + C = 110) (h₂ : A = C) : B = 125 :=
by sorry

end NUMINAMATH_GPT_parallelogram_angle_B_l1526_152690


namespace NUMINAMATH_GPT_total_earthworms_in_box_l1526_152671

-- Definitions of the conditions
def applesPaidByOkeydokey := 5
def applesPaidByArtichokey := 7
def earthwormsReceivedByOkeydokey := 25
def ratio := earthwormsReceivedByOkeydokey / applesPaidByOkeydokey -- which should be 5

-- Theorem statement proving the total number of earthworms in the box
theorem total_earthworms_in_box :
  (applesPaidByOkeydokey + applesPaidByArtichokey) * ratio = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_earthworms_in_box_l1526_152671


namespace NUMINAMATH_GPT_intersection_M_N_l1526_152637

def M : Set ℝ := {x | x / (x - 1) > 0}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1526_152637


namespace NUMINAMATH_GPT_cube_volume_of_surface_area_l1526_152616

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end NUMINAMATH_GPT_cube_volume_of_surface_area_l1526_152616


namespace NUMINAMATH_GPT_probability_of_rain_l1526_152657

-- Define the conditions in Lean
variables (x : ℝ) -- probability of rain

-- Known condition: taking an umbrella 20% of the time
def takes_umbrella : Prop := 0.2 = x + ((1 - x) * x)

-- The desired problem statement
theorem probability_of_rain : takes_umbrella x → x = 1 / 9 :=
by
  -- placeholder for the proof
  intro h
  sorry

end NUMINAMATH_GPT_probability_of_rain_l1526_152657


namespace NUMINAMATH_GPT_total_donation_correct_l1526_152694

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end NUMINAMATH_GPT_total_donation_correct_l1526_152694


namespace NUMINAMATH_GPT_cost_of_iced_coffee_for_2_weeks_l1526_152682

def cost_to_last_for_2_weeks (servings_per_bottle servings_per_day price_per_bottle duration_in_days : ℕ) : ℕ :=
  let total_servings_needed := servings_per_day * duration_in_days
  let bottles_needed := total_servings_needed / servings_per_bottle
  bottles_needed * price_per_bottle

theorem cost_of_iced_coffee_for_2_weeks :
  cost_to_last_for_2_weeks 6 3 3 14 = 21 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_iced_coffee_for_2_weeks_l1526_152682


namespace NUMINAMATH_GPT_carl_max_value_carry_l1526_152691

variables (rock_weight_3_pound : ℕ := 3) (rock_value_3_pound : ℕ := 9)
          (rock_weight_6_pound : ℕ := 6) (rock_value_6_pound : ℕ := 20)
          (rock_weight_2_pound : ℕ := 2) (rock_value_2_pound : ℕ := 5)
          (weight_limit : ℕ := 20)
          (max_six_pound_rocks : ℕ := 2)

noncomputable def max_value_carry : ℕ :=
  max (2 * rock_value_6_pound + 2 * rock_value_3_pound) 
      (4 * rock_value_3_pound + 4 * rock_value_2_pound)

theorem carl_max_value_carry : max_value_carry = 58 :=
by sorry

end NUMINAMATH_GPT_carl_max_value_carry_l1526_152691


namespace NUMINAMATH_GPT_average_weight_ten_students_l1526_152601

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end NUMINAMATH_GPT_average_weight_ten_students_l1526_152601


namespace NUMINAMATH_GPT_sequence_1001st_term_l1526_152679

theorem sequence_1001st_term (a b : ℤ) (h1 : b = 2 * a - 3) : 
  ∃ n : ℤ, n = 1001 → (a + 1000 * (20 * a - 30)) = 30003 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_1001st_term_l1526_152679


namespace NUMINAMATH_GPT_chessboard_L_T_equivalence_l1526_152620

theorem chessboard_L_T_equivalence (n : ℕ) :
  ∃ L_count T_count : ℕ, 
  (L_count = T_count) ∧ -- number of L-shaped pieces is equal to number of T-shaped pieces
  (L_count + T_count = n * (n + 1)) := 
sorry

end NUMINAMATH_GPT_chessboard_L_T_equivalence_l1526_152620


namespace NUMINAMATH_GPT_circle_diameter_l1526_152675

open Real

theorem circle_diameter (r_D : ℝ) (r_C : ℝ) (h_D : r_D = 10) (h_ratio: (π * (r_D ^ 2 - r_C ^ 2)) / (π * r_C ^ 2) = 4) : 2 * r_C = 4 * sqrt 5 :=
by sorry

end NUMINAMATH_GPT_circle_diameter_l1526_152675


namespace NUMINAMATH_GPT_age_difference_28_l1526_152685

variable (li_lin_age_father_sum li_lin_age_future father_age_future : ℕ)

theorem age_difference_28 
    (h1 : li_lin_age_father_sum = 50)
    (h2 : ∀ x, li_lin_age_future = x → father_age_future = 3 * x - 2)
    (h3 : li_lin_age_future + 4 = li_lin_age_father_sum + 8 - (father_age_future + 4))
    : li_lin_age_father_sum - li_lin_age_future = 28 :=
sorry

end NUMINAMATH_GPT_age_difference_28_l1526_152685


namespace NUMINAMATH_GPT_min_value_of_a_l1526_152659

-- Defining the properties of the function f
variable {f : ℝ → ℝ}
variable (even_f : ∀ x, f x = f (-x))
variable (mono_f : ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f x ≤ f y)

-- Necessary condition involving f and a
variable {a : ℝ}
variable (a_condition : f (Real.log a / Real.log 2) + f (Real.log a / Real.log (1/2)) ≤ 2 * f 1)

-- Main statement proving that the minimum value of a is 1/2
theorem min_value_of_a : a = 1/2 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l1526_152659


namespace NUMINAMATH_GPT_hall_length_width_difference_l1526_152696

variable (L W : ℕ)

theorem hall_length_width_difference (h₁ : W = 1 / 2 * L) (h₂ : L * W = 800) :
  L - W = 20 :=
sorry

end NUMINAMATH_GPT_hall_length_width_difference_l1526_152696


namespace NUMINAMATH_GPT_expected_pairs_of_adjacent_face_cards_is_44_over_17_l1526_152609
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end NUMINAMATH_GPT_expected_pairs_of_adjacent_face_cards_is_44_over_17_l1526_152609


namespace NUMINAMATH_GPT_prove_ordered_pair_l1526_152651

-- Definition of the problem
def satisfies_equation1 (x y : ℚ) : Prop :=
  3 * x - 4 * y = -7

def satisfies_equation2 (x y : ℚ) : Prop :=
  7 * x - 3 * y = 5

-- Definition of the correct answer
def correct_answer (x y : ℚ) : Prop :=
  x = -133 / 57 ∧ y = 64 / 19

-- Main theorem to prove
theorem prove_ordered_pair :
  correct_answer (-133 / 57) (64 / 19) :=
by
  unfold correct_answer
  constructor
  { sorry }
  { sorry }

end NUMINAMATH_GPT_prove_ordered_pair_l1526_152651


namespace NUMINAMATH_GPT_total_baskets_l1526_152613

theorem total_baskets (Alex_baskets Sandra_baskets Hector_baskets Jordan_baskets total_baskets : ℕ)
  (h1 : Alex_baskets = 8)
  (h2 : Sandra_baskets = 3 * Alex_baskets)
  (h3 : Hector_baskets = 2 * Sandra_baskets)
  (total_combined_baskets := Alex_baskets + Sandra_baskets + Hector_baskets)
  (h4 : Jordan_baskets = total_combined_baskets / 5)
  (h5 : total_baskets = Alex_baskets + Sandra_baskets + Hector_baskets + Jordan_baskets) :
  total_baskets = 96 := by
  sorry

end NUMINAMATH_GPT_total_baskets_l1526_152613


namespace NUMINAMATH_GPT_find_values_of_real_numbers_l1526_152650

theorem find_values_of_real_numbers (x y : ℝ)
  (h : 2 * x - 1 + (y + 1) * Complex.I = x - y - (x + y) * Complex.I) :
  x = 3 ∧ y = -2 :=
sorry

end NUMINAMATH_GPT_find_values_of_real_numbers_l1526_152650


namespace NUMINAMATH_GPT_length_of_BC_l1526_152640

theorem length_of_BC (b : ℝ) (h : b ^ 4 = 125) : 2 * b = 10 :=
sorry

end NUMINAMATH_GPT_length_of_BC_l1526_152640


namespace NUMINAMATH_GPT_theater_tickets_l1526_152629

theorem theater_tickets (O B P : ℕ) (h1 : O + B + P = 550) 
  (h2 : 15 * O + 10 * B + 25 * P = 9750) (h3: P = 5 * O) (h4 : O ≥ 50) : 
  B - O = 179 :=
by
  sorry

end NUMINAMATH_GPT_theater_tickets_l1526_152629


namespace NUMINAMATH_GPT_more_students_than_rabbits_l1526_152680

theorem more_students_than_rabbits :
  let number_of_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let total_students := students_per_classroom * number_of_classrooms
  let total_rabbits := rabbits_per_classroom * number_of_classrooms
  total_students - total_rabbits = 95 := by
  sorry

end NUMINAMATH_GPT_more_students_than_rabbits_l1526_152680


namespace NUMINAMATH_GPT_no_infinite_harmonic_mean_sequence_l1526_152625

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), (∀ n, a n = a 0 → False) ∧
                   (∀ i, 1 ≤ i → a i = (2 * a (i - 1) * a (i + 1)) / (a (i - 1) + a (i + 1))) :=
sorry

end NUMINAMATH_GPT_no_infinite_harmonic_mean_sequence_l1526_152625


namespace NUMINAMATH_GPT_product_probability_correct_l1526_152633

/-- Define probabilities for spins of Paco and Dani --/
def prob_paco := 1 / 5
def prob_dani := 1 / 15

/-- Define the probability that the product of spins is less than 30 --/
def prob_product_less_than_30 : ℚ :=
  (2 / 5) + (1 / 5) * (9 / 15) + (1 / 5) * (7 / 15) + (1 / 5) * (5 / 15)

theorem product_probability_correct : prob_product_less_than_30 = 17 / 25 :=
by sorry

end NUMINAMATH_GPT_product_probability_correct_l1526_152633


namespace NUMINAMATH_GPT_total_capacity_both_dressers_l1526_152603

/-- Definition of drawers and capacity -/
def first_dresser_drawers : ℕ := 12
def first_dresser_capacity_per_drawer : ℕ := 8
def second_dresser_drawers : ℕ := 6
def second_dresser_capacity_per_drawer : ℕ := 10

/-- Theorem stating the total capacity of both dressers -/
theorem total_capacity_both_dressers :
  (first_dresser_drawers * first_dresser_capacity_per_drawer) +
  (second_dresser_drawers * second_dresser_capacity_per_drawer) = 156 :=
by sorry

end NUMINAMATH_GPT_total_capacity_both_dressers_l1526_152603


namespace NUMINAMATH_GPT_fourth_person_height_l1526_152692

theorem fourth_person_height 
  (H : ℕ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) : 
  H + 10 = 85 :=
by
  sorry

end NUMINAMATH_GPT_fourth_person_height_l1526_152692


namespace NUMINAMATH_GPT_floor_sqrt_225_l1526_152641

theorem floor_sqrt_225 : Int.floor (Real.sqrt 225) = 15 := by
  sorry

end NUMINAMATH_GPT_floor_sqrt_225_l1526_152641


namespace NUMINAMATH_GPT_range_of_mn_l1526_152698

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x

theorem range_of_mn (m n : ℝ)
  (h₁ : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4)
  (h₂ : ∀ z, -5 ≤ z ∧ z ≤ 4 → ∃ x, f x = z ∧ m ≤ x ∧ x ≤ n) :
  1 ≤ m + n ∧ m + n ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_range_of_mn_l1526_152698


namespace NUMINAMATH_GPT_area_of_curvilinear_trapezoid_steps_l1526_152632

theorem area_of_curvilinear_trapezoid_steps (steps : List String) :
  (steps = ["division", "approximation", "summation", "taking the limit"]) :=
sorry

end NUMINAMATH_GPT_area_of_curvilinear_trapezoid_steps_l1526_152632


namespace NUMINAMATH_GPT_total_people_hired_l1526_152663

theorem total_people_hired (H L : ℕ) (hL : L = 1) (payroll : ℕ) (hPayroll : 129 * H + 82 * L = 3952) : H + L = 31 := by
  sorry

end NUMINAMATH_GPT_total_people_hired_l1526_152663


namespace NUMINAMATH_GPT_tony_rollercoasters_l1526_152646

theorem tony_rollercoasters :
  let s1 := 50 -- speed of the first rollercoaster
  let s2 := 62 -- speed of the second rollercoaster
  let s3 := 73 -- speed of the third rollercoaster
  let s4 := 70 -- speed of the fourth rollercoaster
  let s5 := 40 -- speed of the fifth rollercoaster
  let avg_speed := 59 -- Tony's average speed during the day
  let total_speed := s1 + s2 + s3 + s4 + s5
  total_speed / avg_speed = 5 := sorry

end NUMINAMATH_GPT_tony_rollercoasters_l1526_152646


namespace NUMINAMATH_GPT_triangular_array_sum_digits_l1526_152697

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 3780) : (N / 10 + N % 10) = 15 :=
sorry

end NUMINAMATH_GPT_triangular_array_sum_digits_l1526_152697


namespace NUMINAMATH_GPT_sin_four_alpha_l1526_152652

theorem sin_four_alpha (α : ℝ) (h1 : Real.sin (2 * α) = -4 / 5) (h2 : -Real.pi / 4 < α ∧ α < Real.pi / 4) :
  Real.sin (4 * α) = -24 / 25 :=
sorry

end NUMINAMATH_GPT_sin_four_alpha_l1526_152652


namespace NUMINAMATH_GPT_tan_to_trig_identity_l1526_152612

theorem tan_to_trig_identity (α : ℝ) (h : Real.tan α = 3) : (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_to_trig_identity_l1526_152612


namespace NUMINAMATH_GPT_sum_of_possible_values_of_z_l1526_152684

theorem sum_of_possible_values_of_z (x y z : ℂ) 
  (h₁ : z^2 + 5 * x = 10 * z)
  (h₂ : y^2 + 5 * z = 10 * y)
  (h₃ : x^2 + 5 * y = 10 * x) :
  z = 0 ∨ z = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_z_l1526_152684


namespace NUMINAMATH_GPT_average_speed_l1526_152662

theorem average_speed
  (distance1 : ℝ)
  (time1 : ℝ)
  (distance2 : ℝ)
  (time2 : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (average_speed : ℝ)
  (h1 : distance1 = 90)
  (h2 : time1 = 1)
  (h3 : distance2 = 50)
  (h4 : time2 = 1)
  (h5 : total_distance = distance1 + distance2)
  (h6 : total_time = time1 + time2)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 70 := 
sorry

end NUMINAMATH_GPT_average_speed_l1526_152662


namespace NUMINAMATH_GPT_num_three_digit_numbers_no_repeat_l1526_152673

theorem num_three_digit_numbers_no_repeat (digits : Finset ℕ) (h : digits = {1, 2, 3, 4}) :
  (digits.card = 4) →
  ∀ d1 d2 d3, d1 ∈ digits → d2 ∈ digits → d3 ∈ digits →
  d1 ≠ d2 → d1 ≠ d3 → d2 ≠ d3 → 
  3 * 2 * 1 * digits.card = 24 :=
by
  sorry

end NUMINAMATH_GPT_num_three_digit_numbers_no_repeat_l1526_152673


namespace NUMINAMATH_GPT_num_solutions_l1526_152676

theorem num_solutions (k : ℤ) :
  (∃ a b c : ℝ, (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
    (a^2 + b^2 = k * c * (a + b)) ∧
    (b^2 + c^2 = k * a * (b + c)) ∧
    (c^2 + a^2 = k * b * (c + a))) ↔ k = 1 ∨ k = -2 :=
sorry

end NUMINAMATH_GPT_num_solutions_l1526_152676


namespace NUMINAMATH_GPT_carl_personal_owe_l1526_152639

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end NUMINAMATH_GPT_carl_personal_owe_l1526_152639


namespace NUMINAMATH_GPT_next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l1526_152631

-- Problem: Next number after 48 in the sequence
theorem next_number_after_48 (x : ℕ) (h₁ : x % 3 = 0) (h₂ : (x + 1) = 64) : x = 63 := sorry

-- Problem: Eighth number in the sequence
theorem eighth_number_in_sequence (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 8) : n = 168 := sorry

-- Problem: 2013th number in the sequence
theorem two_thousand_thirteenth_number (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 2013) : n = 9120399 := sorry

end NUMINAMATH_GPT_next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l1526_152631


namespace NUMINAMATH_GPT_find_the_number_l1526_152602

theorem find_the_number (x : ℕ) (h : x * 9999 = 4691110842) : x = 469211 := by
    sorry

end NUMINAMATH_GPT_find_the_number_l1526_152602


namespace NUMINAMATH_GPT_football_team_total_players_l1526_152699

variable (P : ℕ)
variable (throwers : ℕ := 52)
variable (total_right_handed : ℕ := 64)
variable (remaining := P - throwers)
variable (left_handed := remaining / 3)
variable (right_handed_non_throwers := 2 * remaining / 3)

theorem football_team_total_players:
  right_handed_non_throwers + throwers = total_right_handed →
  P = 70 :=
by
  sorry

end NUMINAMATH_GPT_football_team_total_players_l1526_152699


namespace NUMINAMATH_GPT_find_smallest_n_l1526_152654

theorem find_smallest_n 
    (a_n : ℕ → ℝ)
    (S_n : ℕ → ℝ)
    (h1 : a_n 1 + a_n 2 = 9 / 2)
    (h2 : S_n 4 = 45 / 8)
    (h3 : ∀ n, S_n n = (1 / 2) * n * (a_n 1 + a_n n)) :
    ∃ n : ℕ, a_n n < 1 / 10 ∧ ∀ m : ℕ, m < n → a_n m ≥ 1 / 10 := 
sorry

end NUMINAMATH_GPT_find_smallest_n_l1526_152654


namespace NUMINAMATH_GPT_calculate_expression_l1526_152665

theorem calculate_expression : 2^4 * 5^2 * 3^3 * 7 = 75600 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1526_152665


namespace NUMINAMATH_GPT_find_f_x_l1526_152678

def tan : ℝ → ℝ := sorry  -- tan function placeholder
def cos : ℝ → ℝ := sorry  -- cos function placeholder
def sin : ℝ → ℝ := sorry  -- sin function placeholder

axiom conditions : 
  tan 45 = 1 ∧
  cos 60 = 2 ∧
  sin 90 = 3 ∧
  cos 180 = 4 ∧
  sin 270 = 5

theorem find_f_x :
  ∃ f x, (f x = 6) ∧ 
  (f = tan ∧ x = 360) := 
sorry

end NUMINAMATH_GPT_find_f_x_l1526_152678


namespace NUMINAMATH_GPT_roxy_total_plants_remaining_l1526_152622

def initial_flowering_plants : Nat := 7
def initial_fruiting_plants : Nat := 2 * initial_flowering_plants
def flowering_plants_bought : Nat := 3
def fruiting_plants_bought : Nat := 2
def flowering_plants_given_away : Nat := 1
def fruiting_plants_given_away : Nat := 4

def total_remaining_plants : Nat :=
  let flowering_plants_now := initial_flowering_plants + flowering_plants_bought - flowering_plants_given_away
  let fruiting_plants_now := initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given_away
  flowering_plants_now + fruiting_plants_now

theorem roxy_total_plants_remaining
  : total_remaining_plants = 21 := by
  sorry

end NUMINAMATH_GPT_roxy_total_plants_remaining_l1526_152622


namespace NUMINAMATH_GPT_angle_complementary_supplementary_l1526_152627

theorem angle_complementary_supplementary (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle1 + angle3 = 180)
  (h3 : angle3 = 125) :
  angle2 = 35 :=
by 
  sorry

end NUMINAMATH_GPT_angle_complementary_supplementary_l1526_152627


namespace NUMINAMATH_GPT_andrew_apples_l1526_152600

theorem andrew_apples : ∃ (A n : ℕ), (6 * n = A) ∧ (5 * (n + 2) = A) ∧ (A = 60) :=
by 
  sorry

end NUMINAMATH_GPT_andrew_apples_l1526_152600


namespace NUMINAMATH_GPT_smallest_n_l1526_152621

theorem smallest_n (n : ℕ) : 
  (n % 6 = 4) ∧ (n % 7 = 2) ∧ (n > 20) → n = 58 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1526_152621


namespace NUMINAMATH_GPT_find_second_number_l1526_152608

theorem find_second_number 
    (lcm : ℕ) (gcf : ℕ) (num1 : ℕ) (num2 : ℕ)
    (h_lcm : lcm = 56) (h_gcf : gcf = 10) (h_num1 : num1 = 14) 
    (h_product : lcm * gcf = num1 * num2) : 
    num2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l1526_152608


namespace NUMINAMATH_GPT_find_number_l1526_152674

theorem find_number (x : ℕ) (h : x / 46 - 27 = 46) : x = 3358 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1526_152674


namespace NUMINAMATH_GPT_union_condition_implies_l1526_152638

-- Define set A as per the given condition
def setA : Set ℝ := { x | x * (x - 1) ≤ 0 }

-- Define set B as per the given condition with parameter a
def setB (a : ℝ) : Set ℝ := { x | Real.log x ≤ a }

-- Given condition A ∪ B = A, we need to prove that a ≤ 0
theorem union_condition_implies (a : ℝ) (h : setA ∪ setB a = setA) : a ≤ 0 := 
by
  sorry

end NUMINAMATH_GPT_union_condition_implies_l1526_152638


namespace NUMINAMATH_GPT_exists_congruent_triangle_covering_with_parallel_side_l1526_152605

variable {Point : Type}
variable [MetricSpace Point]
variable {Triangle : Type}
variable {Polygon : Type}

-- Definitions of triangle and polygon covering relationships.
def covers (T : Triangle) (P : Polygon) : Prop := sorry 
def congruent (T1 T2 : Triangle) : Prop := sorry
def side_parallel_or_coincident (T : Triangle) (P : Polygon) : Prop := sorry

-- Statement: Given a triangle covering a polygon, there exists a congruent triangle which covers the polygon 
-- and has one side parallel to or coincident with a side of the polygon.
theorem exists_congruent_triangle_covering_with_parallel_side 
  (ABC : Triangle) (M : Polygon) 
  (h_cover : covers ABC M) : 
  ∃ Δ : Triangle, congruent Δ ABC ∧ covers Δ M ∧ side_parallel_or_coincident Δ M := 
sorry

end NUMINAMATH_GPT_exists_congruent_triangle_covering_with_parallel_side_l1526_152605


namespace NUMINAMATH_GPT_intersection_eq_T_l1526_152649

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end NUMINAMATH_GPT_intersection_eq_T_l1526_152649


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l1526_152664

theorem solve_quadratic_inequality (x : ℝ) : 3 * x^2 - 5 * x - 2 < 0 → (-1 / 3 < x ∧ x < 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l1526_152664


namespace NUMINAMATH_GPT_red_apples_count_l1526_152611

theorem red_apples_count
  (r y g : ℕ)
  (h1 : r = y)
  (h2 : g = 2 * r)
  (h3 : r + y + g = 28) : r = 7 :=
sorry

end NUMINAMATH_GPT_red_apples_count_l1526_152611


namespace NUMINAMATH_GPT_derivative_evaluation_at_pi_over_3_l1526_152643

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2 * x) + Real.tan x

theorem derivative_evaluation_at_pi_over_3 :
  deriv f (Real.pi / 3) = 3 :=
sorry

end NUMINAMATH_GPT_derivative_evaluation_at_pi_over_3_l1526_152643


namespace NUMINAMATH_GPT_smallest_factor_of_36_l1526_152626

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_factor_of_36_l1526_152626


namespace NUMINAMATH_GPT_sum_of_roots_l1526_152670

-- sum of roots of first polynomial
def S1 : ℚ := -(-6 / 3)

-- sum of roots of second polynomial
def S2 : ℚ := -(8 / 4)

-- proof statement
theorem sum_of_roots : S1 + S2 = 0 :=
by
  -- placeholders
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1526_152670


namespace NUMINAMATH_GPT_find_n_l1526_152623

theorem find_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (S_odd : ℝ) (S_even : ℝ)
  (h1 : ∀ k, a (2 * k - 1) = a 0 + (2 * k - 2) * d)
  (h2 : ∀ k, a (2 * k) = a 1 + (2 * k - 1) * d)
  (h3 : 2 * n + 1 = n + (n + 1))
  (h4 : S_odd = (n + 1) * (a 0 + n * d))
  (h5 : S_even = n * (a 1 + (n - 1) * d))
  (h6 : S_odd = 4)
  (h7 : S_even = 3) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1526_152623


namespace NUMINAMATH_GPT_problem_a_b_c_d_l1526_152666

open Real

/-- The main theorem to be proved -/
theorem problem_a_b_c_d
  (a b c d : ℝ)
  (hab : 0 < a) (hcd : 0 < c) (hab' : 0 < b) (hcd' : 0 < d)
  (h1 : a > c) (h2 : b < d)
  (h3 : a + sqrt b ≥ c + sqrt d)
  (h4 : sqrt a + b ≤ sqrt c + d) :
  a + b + c + d > 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_b_c_d_l1526_152666


namespace NUMINAMATH_GPT_radius_increase_l1526_152656

theorem radius_increase (C1 C2 : ℝ) (π : ℝ) (hC1 : C1 = 40) (hC2 : C2 = 50) (hπ : π > 0) : 
  (C2 - C1) / (2 * π) = 5 / π := 
sorry

end NUMINAMATH_GPT_radius_increase_l1526_152656


namespace NUMINAMATH_GPT_goods_train_speed_l1526_152687

def speed_of_goods_train (length_in_meters : ℕ) (time_in_seconds : ℕ) (speed_of_man_train_kmph : ℕ) : ℕ :=
  let length_in_km := length_in_meters / 1000
  let time_in_hours := time_in_seconds / 3600
  let relative_speed_kmph := (length_in_km * 3600) / time_in_hours
  relative_speed_kmph - speed_of_man_train_kmph

theorem goods_train_speed :
  speed_of_goods_train 280 9 50 = 62 := by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l1526_152687


namespace NUMINAMATH_GPT_parabola_triangle_areas_l1526_152686

-- Define necessary points and expressions
variables (x1 y1 x2 y2 x3 y3 : ℝ)
variables (m n : ℝ)
def parabola_eq (x y : ℝ) := y ^ 2 = 4 * x
def median_line (m n x y : ℝ) := m * x + n * y - m = 0
def areas_sum_sq (S1 S2 S3 : ℝ) := S1 ^ 2 + S2 ^ 2 + S3 ^ 2 = 3

-- Main statement
theorem parabola_triangle_areas :
  (parabola_eq x1 y1 ∧ parabola_eq x2 y2 ∧ parabola_eq x3 y3) →
  (m ≠ 0) →
  (median_line m n 1 0) →
  (x1 + x2 + x3 = 3) →
  ∃ S1 S2 S3 : ℝ, areas_sum_sq S1 S2 S3 :=
by sorry

end NUMINAMATH_GPT_parabola_triangle_areas_l1526_152686


namespace NUMINAMATH_GPT_baseball_weight_l1526_152667

theorem baseball_weight
  (weight_total : ℝ)
  (weight_soccer_ball : ℝ)
  (n_soccer_balls : ℕ)
  (n_baseballs : ℕ)
  (total_weight : ℝ)
  (B : ℝ) :
  n_soccer_balls * weight_soccer_ball + n_baseballs * B = total_weight →
  n_soccer_balls = 9 →
  weight_soccer_ball = 0.8 →
  n_baseballs = 7 →
  total_weight = 10.98 →
  B = 0.54 := sorry

end NUMINAMATH_GPT_baseball_weight_l1526_152667


namespace NUMINAMATH_GPT_second_number_is_915_l1526_152672

theorem second_number_is_915 :
  ∃ (n1 n2 n3 n4 n5 n6 : ℤ), 
    n1 = 3 ∧ 
    n2 = 915 ∧ 
    n3 = 138 ∧ 
    n4 = 1917 ∧ 
    n5 = 2114 ∧ 
    ∃ x: ℤ, 
      (n1 + n2 + n3 + n4 + n5 + x) / 6 = 12 ∧ 
      n2 = 915 :=
by 
  sorry

end NUMINAMATH_GPT_second_number_is_915_l1526_152672


namespace NUMINAMATH_GPT_set_union_complement_l1526_152645

-- Definitions based on provided problem statement
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}
def CRQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- The theorem to prove
theorem set_union_complement : P ∪ CRQ = {x | -2 < x ∧ x ≤ 3} :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_set_union_complement_l1526_152645


namespace NUMINAMATH_GPT_solvability_condition_l1526_152683

def is_solvable (p : ℕ) [Fact (Nat.Prime p)] :=
  ∃ α : ℤ, α * (α - 1) + 3 ≡ 0 [ZMOD p] ↔ ∃ β : ℤ, β * (β - 1) + 25 ≡ 0 [ZMOD p]

theorem solvability_condition (p : ℕ) [Fact (Nat.Prime p)] : 
  is_solvable p :=
sorry

end NUMINAMATH_GPT_solvability_condition_l1526_152683


namespace NUMINAMATH_GPT_Carly_applications_l1526_152615

theorem Carly_applications (x : ℕ) (h1 : ∀ y, y = 2 * x) (h2 : x + 2 * x = 600) : x = 200 :=
sorry

end NUMINAMATH_GPT_Carly_applications_l1526_152615


namespace NUMINAMATH_GPT_trees_in_yard_l1526_152677

theorem trees_in_yard (L d : ℕ) (hL : L = 250) (hd : d = 5) : 
  (L / d + 1) = 51 := by
  sorry

end NUMINAMATH_GPT_trees_in_yard_l1526_152677


namespace NUMINAMATH_GPT_equilibrium_possible_l1526_152660

theorem equilibrium_possible (n : ℕ) : (∃ k : ℕ, 4 * k = n) ∨ (∃ k : ℕ, 4 * k + 3 = n) ↔
  (∃ S1 S2 : Finset ℕ, S1 ∪ S2 = Finset.range (n+1) ∧
                     S1 ∩ S2 = ∅ ∧
                     S1.sum id = S2.sum id) := 
sorry

end NUMINAMATH_GPT_equilibrium_possible_l1526_152660


namespace NUMINAMATH_GPT_four_digit_numbers_count_l1526_152644

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

end NUMINAMATH_GPT_four_digit_numbers_count_l1526_152644


namespace NUMINAMATH_GPT_quadratic_eq_with_roots_l1526_152642

theorem quadratic_eq_with_roots (x y : ℝ) (h : (x^2 - 6 * x + 9) = -|y - 1|) : 
  ∃ a : ℝ, (a^2 - 4 * a + 3 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_eq_with_roots_l1526_152642
