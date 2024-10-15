import Mathlib

namespace NUMINAMATH_GPT_fisherman_gets_8_red_snappers_l2306_230674

noncomputable def num_red_snappers (R : ℕ) : Prop :=
  let cost_red_snapper := 3
  let cost_tuna := 2
  let num_tunas := 14
  let total_earnings := 52
  (R * cost_red_snapper) + (num_tunas * cost_tuna) = total_earnings

theorem fisherman_gets_8_red_snappers : num_red_snappers 8 :=
by
  sorry

end NUMINAMATH_GPT_fisherman_gets_8_red_snappers_l2306_230674


namespace NUMINAMATH_GPT_find_N_l2306_230685

noncomputable def N : ℕ := 1156

-- Condition 1: N is a perfect square
axiom N_perfect_square : ∃ n : ℕ, N = n^2

-- Condition 2: All digits of N are less than 7
axiom N_digits_less_than_7 : ∀ d, d ∈ [1, 1, 5, 6] → d < 7

-- Condition 3: Adding 3 to each digit yields another perfect square
axiom N_plus_3_perfect_square : ∃ m : ℕ, (m^2 = 1156 + 3333)

theorem find_N : N = 1156 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_N_l2306_230685


namespace NUMINAMATH_GPT_y_minus_x_value_l2306_230604

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end NUMINAMATH_GPT_y_minus_x_value_l2306_230604


namespace NUMINAMATH_GPT_no_rectangle_from_five_distinct_squares_l2306_230640

theorem no_rectangle_from_five_distinct_squares (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : q1 < q2) 
  (h2 : q2 < q3) 
  (h3 : q3 < q4) 
  (h4 : q4 < q5) : 
  ¬∃(a b: ℝ), a * b = 5 ∧ a = q1 + q2 + q3 + q4 + q5 := sorry

end NUMINAMATH_GPT_no_rectangle_from_five_distinct_squares_l2306_230640


namespace NUMINAMATH_GPT_solve_x_in_equation_l2306_230646

theorem solve_x_in_equation : ∃ (x : ℤ), 24 - 4 * 2 = 3 + x ∧ x = 13 :=
by
  use 13
  sorry

end NUMINAMATH_GPT_solve_x_in_equation_l2306_230646


namespace NUMINAMATH_GPT_number_of_sides_l2306_230679

theorem number_of_sides (P s : ℝ) (hP : P = 108) (hs : s = 12) : P / s = 9 :=
by sorry

end NUMINAMATH_GPT_number_of_sides_l2306_230679


namespace NUMINAMATH_GPT_probability_of_lamps_arrangement_l2306_230683

noncomputable def probability_lava_lamps : ℚ :=
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_turn_on := 4
  let left_red_on := 1
  let right_blue_off := 1
  let ways_to_choose_positions := Nat.choose total_lamps red_lamps
  let ways_to_choose_turn_on := Nat.choose total_lamps total_turn_on
  let remaining_positions := total_lamps - left_red_on - right_blue_off
  let remaining_red_lamps := red_lamps - left_red_on
  let remaining_turn_on := total_turn_on - left_red_on
  let arrangements_of_remaining_red := Nat.choose remaining_positions remaining_red_lamps
  let arrangements_of_turn_on :=
    Nat.choose (remaining_positions - right_blue_off) remaining_turn_on
  -- The probability calculation
  (arrangements_of_remaining_red * arrangements_of_turn_on : ℚ) / 
    (ways_to_choose_positions * ways_to_choose_turn_on)

theorem probability_of_lamps_arrangement :
    probability_lava_lamps = 4 / 49 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_lamps_arrangement_l2306_230683


namespace NUMINAMATH_GPT_problem_statement_l2306_230641

theorem problem_statement
  (a b : ℝ)
  (ha : a = Real.sqrt 2 + 1)
  (hb : b = Real.sqrt 2 - 1) :
  a^2 - a * b + b^2 = 5 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2306_230641


namespace NUMINAMATH_GPT_problem_solution_l2306_230659

theorem problem_solution (a b c d : ℝ) (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : c = 6 * d) :
  (a + b * c) / (c + d * b) = (3 * (5 + 6 * d)) / (1 + 3 * d) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2306_230659


namespace NUMINAMATH_GPT_total_apples_picked_l2306_230603

def apples_picked : ℕ :=
  let mike := 7
  let nancy := 3
  let keith := 6
  let olivia := 12
  let thomas := 8
  mike + nancy + keith + olivia + thomas

theorem total_apples_picked :
  apples_picked = 36 :=
by
  -- Proof would go here; 'sorry' is used to skip the proof.
  sorry

end NUMINAMATH_GPT_total_apples_picked_l2306_230603


namespace NUMINAMATH_GPT_mike_baseball_cards_l2306_230643

theorem mike_baseball_cards :
  let InitialCards : ℕ := 87
  let BoughtCards : ℕ := 13
  (InitialCards - BoughtCards = 74)
:= by
  sorry

end NUMINAMATH_GPT_mike_baseball_cards_l2306_230643


namespace NUMINAMATH_GPT_arithmetic_mean_l2306_230633

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 5/9) :
  (a + b) / 2 = 31/63 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_mean_l2306_230633


namespace NUMINAMATH_GPT_range_m_plus_2n_l2306_230613

noncomputable def f (x : ℝ) : ℝ := Real.log x - 1 / x
noncomputable def m_value (t : ℝ) : ℝ := 1 / t + 1 / (t ^ 2)

noncomputable def n_value (t : ℝ) : ℝ := Real.log t - 2 / t - 1

noncomputable def g (x : ℝ) : ℝ := (1 / (x ^ 2)) + 2 * Real.log x - (3 / x) - 2

theorem range_m_plus_2n :
  ∀ m n : ℝ, (∃ t > 0, m = m_value t ∧ n = n_value t) →
  (m + 2 * n) ∈ Set.Ici (-2 * Real.log 2 - 4) := by
  sorry

end NUMINAMATH_GPT_range_m_plus_2n_l2306_230613


namespace NUMINAMATH_GPT_min_girls_in_class_l2306_230655

theorem min_girls_in_class : ∃ d : ℕ, 20 - d ≤ 2 * (d + 1) ∧ d ≥ 6 := by
  sorry

end NUMINAMATH_GPT_min_girls_in_class_l2306_230655


namespace NUMINAMATH_GPT_triangle_inequality_condition_l2306_230645

variable (a b c : ℝ)
variable (α : ℝ) -- angle in radians

-- Define the condition where c must be less than a + b
theorem triangle_inequality_condition : c < a + b := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_condition_l2306_230645


namespace NUMINAMATH_GPT_kat_boxing_trainings_per_week_l2306_230653

noncomputable def strength_training_hours_per_week : ℕ := 3
noncomputable def boxing_training_hours (x : ℕ) : ℚ := 1.5 * x
noncomputable def total_training_hours : ℕ := 9

theorem kat_boxing_trainings_per_week (x : ℕ) (h : total_training_hours = strength_training_hours_per_week + boxing_training_hours x) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_kat_boxing_trainings_per_week_l2306_230653


namespace NUMINAMATH_GPT_domain_of_reciprocal_shifted_function_l2306_230688

def domain_of_function (x : ℝ) : Prop :=
  x ≠ 1

theorem domain_of_reciprocal_shifted_function : 
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 1)) ↔ domain_of_function x :=
by 
  sorry

end NUMINAMATH_GPT_domain_of_reciprocal_shifted_function_l2306_230688


namespace NUMINAMATH_GPT_hair_growth_l2306_230677

-- Define the length of Isabella's hair initially and the growth
def initial_length : ℕ := 18
def growth : ℕ := 4

-- Define the final length of the hair after growth
def final_length (initial_length : ℕ) (growth : ℕ) : ℕ := initial_length + growth

-- State the theorem that the final length is 22 inches
theorem hair_growth : final_length initial_length growth = 22 := 
by
  sorry

end NUMINAMATH_GPT_hair_growth_l2306_230677


namespace NUMINAMATH_GPT_find_k_l2306_230666

theorem find_k (k : ℕ) : (1 / 2) ^ 18 * (1 / 81) ^ k = 1 / 18 ^ 18 → k = 0 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l2306_230666


namespace NUMINAMATH_GPT_square_root_unique_l2306_230632

theorem square_root_unique (x : ℝ) (h1 : x + 3 ≥ 0) (h2 : 2 * x - 6 ≥ 0)
  (h : (x + 3)^2 = (2 * x - 6)^2) :
  x = 1 ∧ (x + 3)^2 = 16 := 
by
  sorry

end NUMINAMATH_GPT_square_root_unique_l2306_230632


namespace NUMINAMATH_GPT_factor_27x6_minus_512y6_sum_coeffs_is_152_l2306_230654

variable {x y : ℤ}

theorem factor_27x6_minus_512y6_sum_coeffs_is_152 :
  ∃ a b c d e f g h j k : ℤ, 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) ∧ 
    (a + b + c + d + e + f + g + h + j + k = 152) := 
sorry

end NUMINAMATH_GPT_factor_27x6_minus_512y6_sum_coeffs_is_152_l2306_230654


namespace NUMINAMATH_GPT_students_exceed_goldfish_l2306_230601

theorem students_exceed_goldfish 
    (num_classrooms : ℕ) 
    (students_per_classroom : ℕ) 
    (goldfish_per_classroom : ℕ) 
    (h1 : num_classrooms = 5) 
    (h2 : students_per_classroom = 20) 
    (h3 : goldfish_per_classroom = 3) 
    : (students_per_classroom * num_classrooms) - (goldfish_per_classroom * num_classrooms) = 85 := by
  sorry

end NUMINAMATH_GPT_students_exceed_goldfish_l2306_230601


namespace NUMINAMATH_GPT_total_crayons_l2306_230634
-- Import the whole Mathlib to ensure all necessary components are available

-- Definitions of the number of crayons each person has
def Billy_crayons : ℕ := 62
def Jane_crayons : ℕ := 52
def Mike_crayons : ℕ := 78
def Sue_crayons : ℕ := 97

-- Theorem stating the total number of crayons is 289
theorem total_crayons : (Billy_crayons + Jane_crayons + Mike_crayons + Sue_crayons) = 289 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l2306_230634


namespace NUMINAMATH_GPT_increase_by_thirteen_possible_l2306_230651

-- Define the main condition which states the reduction of the original product
def product_increase_by_thirteen (a : Fin 7 → ℕ) : Prop :=
  let P := (List.range 7).map (fun i => a ⟨i, sorry⟩) |>.prod
  let Q := (List.range 7).map (fun i => a ⟨i, sorry⟩ - 3) |>.prod
  Q = 13 * P

-- State the theorem to be proved
theorem increase_by_thirteen_possible : ∃ (a : Fin 7 → ℕ), product_increase_by_thirteen a :=
sorry

end NUMINAMATH_GPT_increase_by_thirteen_possible_l2306_230651


namespace NUMINAMATH_GPT_sum_series_evaluation_l2306_230675

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (if k = 0 then 0 else (2 * k) / (4 : ℝ) ^ k)

theorem sum_series_evaluation : sum_series = 8 / 9 := by
  sorry

end NUMINAMATH_GPT_sum_series_evaluation_l2306_230675


namespace NUMINAMATH_GPT_inscribed_quadrilateral_circle_eq_radius_l2306_230693

noncomputable def inscribed_circle_condition (AB CD AD BC : ℝ) : Prop :=
  AB + CD = AD + BC

noncomputable def equal_radius_condition (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  r₁ = r₃ ∨ r₄ = r₂

theorem inscribed_quadrilateral_circle_eq_radius 
  (AB CD AD BC r₁ r₂ r₃ r₄ : ℝ)
  (h_inscribed_circle: inscribed_circle_condition AB CD AD BC)
  (h_four_circles: ∀ i, (i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4) → ∃ (r : ℝ), r = rᵢ): 
  equal_radius_condition r₁ r₂ r₃ r₄ :=
by {
  sorry
}

end NUMINAMATH_GPT_inscribed_quadrilateral_circle_eq_radius_l2306_230693


namespace NUMINAMATH_GPT_proof_problem_l2306_230656

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2306_230656


namespace NUMINAMATH_GPT_tournament_trio_l2306_230687

theorem tournament_trio
  (n : ℕ)
  (h_n : n ≥ 3)
  (match_result : Fin n → Fin n → Prop)
  (h1 : ∀ i j : Fin n, i ≠ j → (match_result i j ∨ match_result j i))
  (h2 : ∀ i : Fin n, ∃ j : Fin n, match_result i j)
:
  ∃ (A B C : Fin n), match_result A B ∧ match_result B C ∧ match_result C A :=
by
  sorry

end NUMINAMATH_GPT_tournament_trio_l2306_230687


namespace NUMINAMATH_GPT_person_speed_l2306_230607

noncomputable def distance_meters : ℝ := 1080
noncomputable def time_minutes : ℝ := 14
noncomputable def distance_kilometers : ℝ := distance_meters / 1000
noncomputable def time_hours : ℝ := time_minutes / 60
noncomputable def speed_km_per_hour : ℝ := distance_kilometers / time_hours

theorem person_speed :
  abs (speed_km_per_hour - 4.63) < 0.01 :=
by
  -- conditions extracted
  let distance_in_km := distance_meters / 1000
  let time_in_hours := time_minutes / 60
  let speed := distance_in_km / time_in_hours
  -- We expect speed to be approximately 4.63
  sorry 

end NUMINAMATH_GPT_person_speed_l2306_230607


namespace NUMINAMATH_GPT_number_of_uncool_parents_l2306_230657

variable (total_students cool_dads cool_moms cool_both : ℕ)

theorem number_of_uncool_parents (h1 : total_students = 40)
                                  (h2 : cool_dads = 18)
                                  (h3 : cool_moms = 22)
                                  (h4 : cool_both = 10) :
    total_students - (cool_dads + cool_moms - cool_both) = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_uncool_parents_l2306_230657


namespace NUMINAMATH_GPT_jogger_ahead_engine_l2306_230627

-- Define the given constants for speed and length
def jogger_speed : ℝ := 2.5 -- in m/s
def train_speed : ℝ := 12.5 -- in m/s
def train_length : ℝ := 120 -- in meters
def passing_time : ℝ := 40 -- in seconds

-- Define the target distance
def jogger_ahead : ℝ := 280 -- in meters

-- Lean 4 statement to prove the jogger is 280 meters ahead of the train's engine
theorem jogger_ahead_engine :
  passing_time * (train_speed - jogger_speed) - train_length = jogger_ahead :=
by
  sorry

end NUMINAMATH_GPT_jogger_ahead_engine_l2306_230627


namespace NUMINAMATH_GPT_inequality_for_positive_real_numbers_l2306_230618

theorem inequality_for_positive_real_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^4 + b^4 + c^4) / (a + b + c) ≥ a * b * c :=
  sorry

end NUMINAMATH_GPT_inequality_for_positive_real_numbers_l2306_230618


namespace NUMINAMATH_GPT_inequality_solution_l2306_230669

theorem inequality_solution (x : ℝ) :
  (∀ y : ℝ, (0 < y) → (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y + y)) ↔ (1 < x) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2306_230669


namespace NUMINAMATH_GPT_number_of_purchasing_schemes_l2306_230676

def total_cost (a : Nat) (b : Nat) : Nat := 8 * a + 10 * b

def valid_schemes : List (Nat × Nat) :=
  [(4, 4), (4, 5), (4, 6), (4, 7),
   (5, 4), (5, 5), (5, 6),
   (6, 4), (6, 5),
   (7, 4)]

theorem number_of_purchasing_schemes : valid_schemes.length = 9 := sorry

end NUMINAMATH_GPT_number_of_purchasing_schemes_l2306_230676


namespace NUMINAMATH_GPT_area_of_roof_l2306_230664

-- Definitions and conditions
def length (w : ℝ) := 4 * w
def difference_eq (l w : ℝ) := l - w = 39
def area (l w : ℝ) := l * w

-- Theorem statement
theorem area_of_roof (w l : ℝ) (h_length : l = length w) (h_diff : difference_eq l w) : area l w = 676 :=
by
  sorry

end NUMINAMATH_GPT_area_of_roof_l2306_230664


namespace NUMINAMATH_GPT_segments_count_l2306_230691

/--
Given two concentric circles, with chords of the larger circle that are tangent to the smaller circle,
if each chord subtends an angle of 80 degrees at the center, then the number of such segments 
drawn before returning to the starting point is 18.
-/
theorem segments_count (angle_ABC : ℝ) (circumference_angle_sum : ℝ → ℝ) (n m : ℕ) :
  angle_ABC = 80 → 
  circumference_angle_sum angle_ABC = 360 → 
  100 * n = 360 * m → 
  5 * n = 18 * m →
  n = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_segments_count_l2306_230691


namespace NUMINAMATH_GPT_even_function_f_l2306_230689

-- Problem statement:
-- Given that f is an even function and that for x < 0, f(x) = x^2 - 1/x,
-- prove that f(1) = 2.

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - 1/x else 0

theorem even_function_f {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x))
  (h_neg_def : ∀ x, x < 0 → f x = x^2 - 1/x) : f 1 = 2 :=
by
  -- Proof body (to be completed)
  sorry

end NUMINAMATH_GPT_even_function_f_l2306_230689


namespace NUMINAMATH_GPT_some_number_value_l2306_230606

theorem some_number_value (some_number : ℝ): 
  (∀ n : ℝ, (n / some_number) * (n / 80) = 1 → n = 40) → some_number = 80 :=
by
  sorry

end NUMINAMATH_GPT_some_number_value_l2306_230606


namespace NUMINAMATH_GPT_interest_rate_is_six_percent_l2306_230616

noncomputable def amount : ℝ := 1120
noncomputable def principal : ℝ := 979.0209790209791
noncomputable def time_years : ℝ := 2 + 2 / 5

noncomputable def total_interest (A P: ℝ) : ℝ := A - P

noncomputable def interest_rate_per_annum (I P T: ℝ) : ℝ := I / (P * T) * 100

theorem interest_rate_is_six_percent :
  interest_rate_per_annum (total_interest amount principal) principal time_years = 6 := 
by
  sorry

end NUMINAMATH_GPT_interest_rate_is_six_percent_l2306_230616


namespace NUMINAMATH_GPT_gigi_remaining_batches_l2306_230690

variable (f b1 tf remaining_batches : ℕ)
variable (f_pos : 0 < f)
variable (batches_nonneg : 0 ≤ b1)
variable (t_f_pos : 0 < tf)
variable (h_f : f = 2)
variable (h_b1 : b1 = 3)
variable (h_tf : tf = 20)

theorem gigi_remaining_batches (h : remaining_batches = (tf - (f * b1)) / f) : remaining_batches = 7 := by
  sorry

end NUMINAMATH_GPT_gigi_remaining_batches_l2306_230690


namespace NUMINAMATH_GPT_proposition_p_proposition_q_l2306_230644

theorem proposition_p : ∅ ≠ ({∅} : Set (Set Empty)) := by
  sorry

theorem proposition_q (A : Set ℕ) (B : Set (Set ℕ)) (hA : A = {1, 2})
    (hB : B = {x | x ⊆ A}) : A ∈ B := by
  sorry

end NUMINAMATH_GPT_proposition_p_proposition_q_l2306_230644


namespace NUMINAMATH_GPT_fraction_diff_l2306_230652

open Real

theorem fraction_diff (x y : ℝ) (hx : x = sqrt 5 - 1) (hy : y = sqrt 5 + 1) :
  (1 / x - 1 / y) = 1 / 2 := sorry

end NUMINAMATH_GPT_fraction_diff_l2306_230652


namespace NUMINAMATH_GPT_third_vertex_l2306_230630

/-- Two vertices of a right triangle are located at (4, 3) and (0, 0).
The third vertex of the triangle lies on the positive branch of the x-axis.
Determine the coordinates of the third vertex if the area of the triangle is 24 square units. -/
theorem third_vertex (x : ℝ) (h : x > 0) : 
  (1 / 2 * |x| * 3 = 24) → (x, 0) = (16, 0) :=
by
  intro h_area
  sorry

end NUMINAMATH_GPT_third_vertex_l2306_230630


namespace NUMINAMATH_GPT_tiffany_reading_homework_pages_l2306_230637

theorem tiffany_reading_homework_pages 
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (reading_pages : ℕ)
  (H1 : math_pages = 6)
  (H2 : problems_per_page = 3)
  (H3 : total_problems = 30)
  (H4 : reading_pages = (total_problems - math_pages * problems_per_page) / problems_per_page) 
  : reading_pages = 4 := 
sorry

end NUMINAMATH_GPT_tiffany_reading_homework_pages_l2306_230637


namespace NUMINAMATH_GPT_false_converse_of_vertical_angles_l2306_230671

theorem false_converse_of_vertical_angles
  (P : Prop) (Q : Prop) (V : ∀ {A B C D : Type}, (A = B ∧ C = D) → P) (C1 : P → Q) :
  ¬ (Q → P) :=
sorry

end NUMINAMATH_GPT_false_converse_of_vertical_angles_l2306_230671


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l2306_230628

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r = 18) 
  (h2 : a * r^4 = 1458) : 
  a = 6 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l2306_230628


namespace NUMINAMATH_GPT_anticipated_margin_l2306_230609

noncomputable def anticipated_profit_margin (original_purchase_price : ℝ) (decrease_percentage : ℝ) (profit_margin_increase : ℝ) (selling_price : ℝ) : ℝ :=
original_purchase_price * (1 + profit_margin_increase / 100)

theorem anticipated_margin (x : ℝ) (original_purchase_price_decrease : ℝ := 0.064) (profit_margin_increase : ℝ := 8) (selling_price : ℝ) :
  selling_price = original_purchase_price * (1 + x / 100) ∧ selling_price = (1 - original_purchase_price_decrease) * (1 + (x + profit_margin_increase) / 100) →
  true :=
by
  sorry

end NUMINAMATH_GPT_anticipated_margin_l2306_230609


namespace NUMINAMATH_GPT_log_a_less_than_neg_b_minus_one_l2306_230622

variable {x : ℝ} (a b : ℝ) (f : ℝ → ℝ)

theorem log_a_less_than_neg_b_minus_one
  (h1 : 0 < a)
  (h2 : ∀ x > 0, f x ≥ f 3)
  (h3 : ∀ x > 0, f x = -3 * Real.log x + a * x^2 + b * x) :
  Real.log a < -b - 1 :=
  sorry

end NUMINAMATH_GPT_log_a_less_than_neg_b_minus_one_l2306_230622


namespace NUMINAMATH_GPT_find_lengths_of_DE_and_HJ_l2306_230631

noncomputable def lengths_consecutive_segments (BD DE EF FG GH HJ : ℝ) (BC : ℝ) : Prop :=
  BD = 5 ∧ EF = 11 ∧ FG = 7 ∧ GH = 3 ∧ BC = 29 ∧ BD + DE + EF + FG + GH + HJ = BC ∧ DE = HJ

theorem find_lengths_of_DE_and_HJ (x : ℝ) : lengths_consecutive_segments 5 x 11 7 3 x 29 → x = 1.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_lengths_of_DE_and_HJ_l2306_230631


namespace NUMINAMATH_GPT_cost_per_page_first_time_l2306_230684

-- Definitions based on conditions
variables (num_pages : ℕ) (rev_once_pages : ℕ) (rev_twice_pages : ℕ)
variables (rev_cost : ℕ) (total_cost : ℕ)
variables (first_time_cost : ℕ)

-- Conditions
axiom h1 : num_pages = 100
axiom h2 : rev_once_pages = 35
axiom h3 : rev_twice_pages = 15
axiom h4 : rev_cost = 4
axiom h5 : total_cost = 860

-- Proof statement: Prove that the cost per page for the first time a page is typed is $6
theorem cost_per_page_first_time : first_time_cost = 6 :=
sorry

end NUMINAMATH_GPT_cost_per_page_first_time_l2306_230684


namespace NUMINAMATH_GPT_problem_solution_l2306_230602

-- Declare the proof problem in Lean 4

theorem problem_solution (x y : ℝ) 
  (h1 : (y + 1) ^ 2 + (x - 2) ^ (1/2) = 0) : 
  y ^ x = 1 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2306_230602


namespace NUMINAMATH_GPT_squares_with_equal_black_and_white_cells_l2306_230681

open Nat

/-- Given a specific coloring of cells in a 5x5 grid, prove that there are
exactly 16 squares that have an equal number of black and white cells. --/
theorem squares_with_equal_black_and_white_cells :
  let gridSize := 5
  let number_of_squares_with_equal_black_and_white_cells := 16
  true := sorry

end NUMINAMATH_GPT_squares_with_equal_black_and_white_cells_l2306_230681


namespace NUMINAMATH_GPT_max_n_value_l2306_230647

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem max_n_value (m : ℝ) (x_i : ℕ → ℝ) (n : ℕ) (h1 : ∀ i, i < n → f (x_i i) / (x_i i) = m)
  (h2 : ∀ i, i < n → -2 * Real.pi ≤ x_i i ∧ x_i i ≤ 2 * Real.pi) :
  n ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_n_value_l2306_230647


namespace NUMINAMATH_GPT_temperature_decrease_l2306_230642

theorem temperature_decrease (initial : ℤ) (decrease : ℤ) : initial = -3 → decrease = 6 → initial - decrease = -9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_temperature_decrease_l2306_230642


namespace NUMINAMATH_GPT_median_of_consecutive_integers_l2306_230626

theorem median_of_consecutive_integers (a b : ℤ) (h : a + b = 50) : 
  (a + b) / 2 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_median_of_consecutive_integers_l2306_230626


namespace NUMINAMATH_GPT_searchlight_revolutions_l2306_230699

theorem searchlight_revolutions (p : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : p = 0.6666666666666667) 
  (h2 : t = 10) 
  (h3 : p = (60 / r - t) / (60 / r)) : 
  r = 2 :=
by sorry

end NUMINAMATH_GPT_searchlight_revolutions_l2306_230699


namespace NUMINAMATH_GPT_road_repair_equation_l2306_230662

variable (x : ℝ) 

-- Original problem conditions
def total_road_length := 150
def extra_repair_per_day := 5
def days_ahead := 5

-- The proof problem to show that the schedule differential equals 5 days ahead
theorem road_repair_equation :
  (total_road_length / x) - (total_road_length / (x + extra_repair_per_day)) = days_ahead :=
sorry

end NUMINAMATH_GPT_road_repair_equation_l2306_230662


namespace NUMINAMATH_GPT_ratio_Nicolai_to_Charliz_l2306_230611

-- Definitions based on conditions
def Haylee_guppies := 3 * 12
def Jose_guppies := Haylee_guppies / 2
def Charliz_guppies := Jose_guppies / 3
def Total_guppies := 84
def Nicolai_guppies := Total_guppies - (Haylee_guppies + Jose_guppies + Charliz_guppies)

-- Proof statement
theorem ratio_Nicolai_to_Charliz : Nicolai_guppies / Charliz_guppies = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_Nicolai_to_Charliz_l2306_230611


namespace NUMINAMATH_GPT_find_units_min_selling_price_l2306_230615

-- Definitions for the given conditions
def total_units : ℕ := 160
def cost_A : ℕ := 150
def cost_B : ℕ := 350
def total_cost : ℕ := 36000
def min_profit : ℕ := 11000

-- Part 1: Proving number of units purchased
theorem find_units :
  ∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x :=
by
  sorry

-- Part 2: Finding the minimum selling price per unit of model A for the profit condition
theorem min_selling_price (t : ℕ) :
  (∃ x y : ℕ,
    x + y = total_units ∧
    cost_A * x + cost_B * y = total_cost ∧
    y = total_units - x) →
  100 * (t - cost_A) + 60 * 2 * (t - cost_A) ≥ min_profit →
  t ≥ 200 :=
by
  sorry

end NUMINAMATH_GPT_find_units_min_selling_price_l2306_230615


namespace NUMINAMATH_GPT_music_stand_cost_proof_l2306_230661

-- Definitions of the constants involved
def flute_cost : ℝ := 142.46
def song_book_cost : ℝ := 7.00
def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := total_spent - (flute_cost + song_book_cost)

-- The statement we need to prove
theorem music_stand_cost_proof : music_stand_cost = 8.89 := 
by
  sorry

end NUMINAMATH_GPT_music_stand_cost_proof_l2306_230661


namespace NUMINAMATH_GPT_factorization_of_x_squared_minus_64_l2306_230621

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end NUMINAMATH_GPT_factorization_of_x_squared_minus_64_l2306_230621


namespace NUMINAMATH_GPT_algebraic_expression_value_l2306_230635

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2306_230635


namespace NUMINAMATH_GPT_hotel_accommodation_l2306_230695

theorem hotel_accommodation :
  ∃ (arrangements : ℕ), arrangements = 27 :=
by
  -- problem statement
  let triple_room := 1
  let double_room := 1
  let single_room := 1
  let adults := 3
  let children := 2
  
  -- use the given conditions and properties of combinations to calculate arrangements
  sorry

end NUMINAMATH_GPT_hotel_accommodation_l2306_230695


namespace NUMINAMATH_GPT_johns_overall_average_speed_l2306_230620

open Real

noncomputable def johns_average_speed (scooter_time_min : ℝ) (scooter_speed_mph : ℝ) 
    (jogging_time_min : ℝ) (jogging_speed_mph : ℝ) : ℝ :=
  let scooter_time_hr := scooter_time_min / 60
  let jogging_time_hr := jogging_time_min / 60
  let distance_scooter := scooter_speed_mph * scooter_time_hr
  let distance_jogging := jogging_speed_mph * jogging_time_hr
  let total_distance := distance_scooter + distance_jogging
  let total_time := scooter_time_hr + jogging_time_hr
  total_distance / total_time

theorem johns_overall_average_speed :
  johns_average_speed 40 20 60 6 = 11.6 :=
by
  sorry

end NUMINAMATH_GPT_johns_overall_average_speed_l2306_230620


namespace NUMINAMATH_GPT_full_seasons_already_aired_l2306_230668

variable (days_until_premiere : ℕ)
variable (episodes_per_day : ℕ)
variable (episodes_per_season : ℕ)

theorem full_seasons_already_aired (h_days : days_until_premiere = 10)
                                  (h_episodes_day : episodes_per_day = 6)
                                  (h_episodes_season : episodes_per_season = 15) :
  (days_until_premiere * episodes_per_day) / episodes_per_season = 4 := by
  sorry

end NUMINAMATH_GPT_full_seasons_already_aired_l2306_230668


namespace NUMINAMATH_GPT_algebraic_expression_value_l2306_230694

theorem algebraic_expression_value (p q : ℝ) 
  (h : p * 3^3 + q * 3 + 1 = 2015) : 
  p * (-3)^3 + q * (-3) + 1 = -2013 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2306_230694


namespace NUMINAMATH_GPT_dante_initially_has_8_jelly_beans_l2306_230636

-- Conditions
def aaron_jelly_beans : ℕ := 5
def bianca_jelly_beans : ℕ := 7
def callie_jelly_beans : ℕ := 8
def dante_jelly_beans_initially (D : ℕ) : Prop := 
  ∀ (D : ℕ), (6 ≤ D - 1 ∧ D - 1 ≤ callie_jelly_beans - 1)

-- Theorem
theorem dante_initially_has_8_jelly_beans :
  ∃ (D : ℕ), (aaron_jelly_beans + 1 = 6) →
             (callie_jelly_beans = 8) →
             dante_jelly_beans_initially D →
             D = 8 := 
by
  sorry

end NUMINAMATH_GPT_dante_initially_has_8_jelly_beans_l2306_230636


namespace NUMINAMATH_GPT_apple_tree_distribution_l2306_230608

-- Definition of the problem
noncomputable def paths := 4

-- Definition of the apple tree positions
structure Position where
  x : ℕ -- Coordinate x
  y : ℕ -- Coordinate y

-- Definition of the initial condition: one existing apple tree
def existing_apple_tree : Position := {x := 0, y := 0}

-- Problem: proving the existence of a configuration with three new apple trees
theorem apple_tree_distribution :
  ∃ (p1 p2 p3 : Position),
    (p1 ≠ existing_apple_tree) ∧ (p2 ≠ existing_apple_tree) ∧ (p3 ≠ existing_apple_tree) ∧
    -- Ensure each path has equal number of trees on both sides
    (∃ (path1 path2 : ℕ), 
      -- Horizontal path balance
      path1 = (if p1.x > 0 then 1 else 0) + (if p2.x > 0 then 1 else 0) + (if p3.x > 0 then 1 else 0) + 1 ∧
      path2 = (if p1.x < 0 then 1 else 0) + (if p2.x < 0 then 1 else 0) + (if p3.x < 0 then 1 else 0) ∧
      path1 = path2) ∧
    (∃ (path3 path4 : ℕ), 
      -- Vertical path balance
      path3 = (if p1.y > 0 then 1 else 0) + (if p2.y > 0 then 1 else 0) + (if p3.y > 0 then 1 else 0) + 1 ∧
      path4 = (if p1.y < 0 then 1 else 0) + (if p2.y < 0 then 1 else 0) + (if p3.y < 0 then 1 else 0) ∧
      path3 = path4)
  := by sorry

end NUMINAMATH_GPT_apple_tree_distribution_l2306_230608


namespace NUMINAMATH_GPT_reduce_to_original_l2306_230638

theorem reduce_to_original (x : ℝ) (factor : ℝ) (original : ℝ) :
  original = x → factor = 1/1000 → x * factor = 0.0169 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_reduce_to_original_l2306_230638


namespace NUMINAMATH_GPT_B_grazed_months_l2306_230663

-- Define the conditions
variables (A_cows B_cows C_cows D_cows : ℕ)
variables (A_months B_months C_months D_months : ℕ)
variables (A_rent total_rent : ℕ)

-- Given conditions
def A_condition := (A_cows = 24 ∧ A_months = 3)
def B_condition := (B_cows = 10)
def C_condition := (C_cows = 35 ∧ C_months = 4)
def D_condition := (D_cows = 21 ∧ D_months = 3)
def A_rent_condition := (A_rent = 720)
def total_rent_condition := (total_rent = 3250)

-- Define cow-months calculation
def cow_months (cows months : ℕ) : ℕ := cows * months

-- Define cost per cow-month
def cost_per_cow_month (rent cow_months : ℕ) : ℕ := rent / cow_months

-- Define B's months of grazing proof problem
theorem B_grazed_months
  (A_cows_months : cow_months 24 3 = 72)
  (B_cows := 10)
  (C_cows_months : cow_months 35 4 = 140)
  (D_cows_months : cow_months 21 3 = 63)
  (A_rent_condition : A_rent = 720)
  (total_rent_condition : total_rent = 3250) :
  ∃ (B_months : ℕ), 10 * B_months = 50 ∧ B_months = 5 := sorry

end NUMINAMATH_GPT_B_grazed_months_l2306_230663


namespace NUMINAMATH_GPT_intersection_point_exists_correct_line_l2306_230670

noncomputable def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
noncomputable def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 2 = 0
noncomputable def parallel_line (x y : ℝ) : Prop := 4 * x - 2 * y + 7 = 0
noncomputable def target_line (x y : ℝ) : Prop := 2 * x - y - 18 = 0

theorem intersection_point_exists (x y : ℝ) : line1 x y ∧ line2 x y → (x = 14 ∧ y = 10) := 
by sorry

theorem correct_line (x y : ℝ) : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ parallel_line x y 
  → target_line x y :=
by sorry

end NUMINAMATH_GPT_intersection_point_exists_correct_line_l2306_230670


namespace NUMINAMATH_GPT_max_abs_sum_eq_two_l2306_230672

theorem max_abs_sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 2) : |x| + |y| ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_abs_sum_eq_two_l2306_230672


namespace NUMINAMATH_GPT_solve_quadratic_l2306_230617

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 + 5 * x = 0 ↔ x = 0 ∨ x = -5/2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2306_230617


namespace NUMINAMATH_GPT_cost_price_computer_table_l2306_230680

theorem cost_price_computer_table (C S : ℝ) (hS1 : S = 1.25 * C) (hS2 : S = 1000) : C = 800 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l2306_230680


namespace NUMINAMATH_GPT_mass_percentages_correct_l2306_230619

noncomputable def mass_percentage_of_Ba (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * 137.327 + (y / 153.326) * 137.327) / (x + y) ) * 100

noncomputable def mass_percentage_of_F (x y : ℝ) : ℝ :=
  ( ((x / 175.323) * (2 * 18.998)) / (x + y) ) * 100

noncomputable def mass_percentage_of_O (x y : ℝ) : ℝ :=
  ( ((y / 153.326) * 15.999) / (x + y) ) * 100

theorem mass_percentages_correct (x y : ℝ) :
  ∃ (Ba F O : ℝ), 
    Ba = mass_percentage_of_Ba x y ∧
    F = mass_percentage_of_F x y ∧
    O = mass_percentage_of_O x y :=
sorry

end NUMINAMATH_GPT_mass_percentages_correct_l2306_230619


namespace NUMINAMATH_GPT_sequence_count_l2306_230650

theorem sequence_count :
  ∃ f : ℕ → ℕ,
    (f 3 = 1) ∧ (f 4 = 1) ∧ (f 5 = 1) ∧ (f 6 = 2) ∧ (f 7 = 2) ∧
    (∀ n, n ≥ 8 → f n = f (n-4) + 2 * f (n-5) + f (n-6)) ∧
    f 15 = 21 :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_count_l2306_230650


namespace NUMINAMATH_GPT_solution_set_l2306_230649

def f (x m : ℝ) : ℝ := |x - 1| - |x - m|

theorem solution_set (x : ℝ) :
  (f x 2) ≥ 1 ↔ x ≥ 2 :=
sorry

end NUMINAMATH_GPT_solution_set_l2306_230649


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l2306_230605

-- Definitions based on given conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
axiom a_4 : a 4 = 6
axiom a_6 : a 6 = 10
axiom all_positive_b : ∀ n, 0 < b n
axiom b_3 : b 3 = a 3
axiom T_2 : T 2 = 3

-- Required to prove
theorem arithmetic_sequence_general_formula : ∀ n, a n = 2 * n - 2 :=
sorry

theorem geometric_sequence_sum_formula : ∀ n, T n = 2^n - 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l2306_230605


namespace NUMINAMATH_GPT_problem_l2306_230625

def vec_a : ℝ × ℝ := (5, 3)
def vec_b : ℝ × ℝ := (1, -2)
def two_vec_b : ℝ × ℝ := (2 * 1, 2 * -2)
def expected_result : ℝ × ℝ := (3, 7)

theorem problem : (vec_a.1 - two_vec_b.1, vec_a.2 - two_vec_b.2) = expected_result :=
by
  sorry

end NUMINAMATH_GPT_problem_l2306_230625


namespace NUMINAMATH_GPT_total_treats_value_l2306_230648

noncomputable def hotel_per_night := 4000
noncomputable def nights := 2
noncomputable def car_value := 30000
noncomputable def house_value := 4 * car_value
noncomputable def total_value := hotel_per_night * nights + car_value + house_value

theorem total_treats_value : total_value = 158000 :=
by
  sorry

end NUMINAMATH_GPT_total_treats_value_l2306_230648


namespace NUMINAMATH_GPT_fixed_monthly_fee_l2306_230623

theorem fixed_monthly_fee (x y : ℝ)
  (h₁ : x + y = 18.70)
  (h₂ : x + 3 * y = 34.10) : x = 11.00 :=
by sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l2306_230623


namespace NUMINAMATH_GPT_solution_correct_l2306_230600

noncomputable def a := 3 + 3 * Real.sqrt 2
noncomputable def b := 3 - 3 * Real.sqrt 2

theorem solution_correct (h : a ≥ b) : 3 * a + 2 * b = 15 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_solution_correct_l2306_230600


namespace NUMINAMATH_GPT_total_annual_salary_excluding_turban_l2306_230697

-- Let X be the total amount of money Gopi gives as salary for one year, excluding the turban.
variable (X : ℝ)

-- Condition: The servant leaves after 9 months and receives Rs. 60 plus the turban.
variable (received_money : ℝ)
variable (turban_price : ℝ)

-- Condition values:
axiom received_money_condition : received_money = 60
axiom turban_price_condition : turban_price = 30

-- Question: Prove that X equals 90.
theorem total_annual_salary_excluding_turban :
  3/4 * (X + turban_price) = 90 :=
sorry

end NUMINAMATH_GPT_total_annual_salary_excluding_turban_l2306_230697


namespace NUMINAMATH_GPT_passing_grade_fraction_l2306_230698

theorem passing_grade_fraction (A B C D F : ℚ) (hA : A = 1/4) (hB : B = 1/2) (hC : C = 1/8) (hD : D = 1/12) (hF : F = 1/24) : 
  A + B + C = 7/8 :=
by
  sorry

end NUMINAMATH_GPT_passing_grade_fraction_l2306_230698


namespace NUMINAMATH_GPT_highland_high_students_highland_high_num_both_clubs_l2306_230629

theorem highland_high_students (total_students drama_club science_club either_both both_clubs : ℕ)
  (h1 : total_students = 320)
  (h2 : drama_club = 90)
  (h3 : science_club = 140)
  (h4 : either_both = 200) : 
  both_clubs = drama_club + science_club - either_both :=
by
  sorry

noncomputable def num_both_clubs : ℕ :=
if h : 320 = 320 ∧ 90 = 90 ∧ 140 = 140 ∧ 200 = 200
then 90 + 140 - 200
else 0

theorem highland_high_num_both_clubs : num_both_clubs = 30 :=
by
  sorry

end NUMINAMATH_GPT_highland_high_students_highland_high_num_both_clubs_l2306_230629


namespace NUMINAMATH_GPT_meeting_time_l2306_230658

/--
The Racing Magic takes 150 seconds to circle the racing track once.
The Charging Bull makes 40 rounds of the track in an hour.
Prove that Racing Magic and Charging Bull meet at the starting point for the second time 
after 300 minutes.
-/
theorem meeting_time (rac_magic_time : ℕ) (chrg_bull_rounds_hour : ℕ)
  (h1 : rac_magic_time = 150) (h2 : chrg_bull_rounds_hour = 40) : 
  ∃ t: ℕ, t = 300 := 
by
  sorry

end NUMINAMATH_GPT_meeting_time_l2306_230658


namespace NUMINAMATH_GPT_max_min_sum_difference_l2306_230686

-- The statement that we need to prove
theorem max_min_sum_difference : 
  ∃ (max_sum min_sum: ℕ), (∀ (RST UVW XYZ : ℕ),
   -- Constraints for Max's and Minnie's sums respectively
   (RST = 100 * 9 + 10 * 6 + 3 ∧ UVW = 100 * 8 + 10 * 5 + 2 ∧ XYZ = 100 * 7 + 10 * 4 + 1 → max_sum = 2556) ∧ 
   (RST = 100 * 1 + 10 * 0 + 6 ∧ UVW = 100 * 2 + 10 * 4 + 7 ∧ XYZ = 100 * 3 + 10 * 5 + 8 → min_sum = 711)) → 
    max_sum - min_sum = 1845 :=
by
  sorry

end NUMINAMATH_GPT_max_min_sum_difference_l2306_230686


namespace NUMINAMATH_GPT_original_number_eq_0_000032_l2306_230660

theorem original_number_eq_0_000032 (x : ℝ) (hx : 0 < x) 
  (h : 10^8 * x = 8 * (1 / x)) : x = 0.000032 :=
sorry

end NUMINAMATH_GPT_original_number_eq_0_000032_l2306_230660


namespace NUMINAMATH_GPT_remainder_7_pow_2010_l2306_230612

theorem remainder_7_pow_2010 :
  (7 ^ 2010) % 100 = 49 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_7_pow_2010_l2306_230612


namespace NUMINAMATH_GPT_nickels_eq_100_l2306_230624

variables (P D N Q H DollarCoins : ℕ)

def conditions :=
  D = P + 10 ∧
  N = 2 * D ∧
  Q = 4 ∧
  P = 10 * Q ∧
  H = Q + 5 ∧
  DollarCoins = 3 * H ∧
  (P + 10 * D + 5 * N + 25 * Q + 50 * H + 100 * DollarCoins = 2000)

theorem nickels_eq_100 (h : conditions P D N Q H DollarCoins) : N = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_nickels_eq_100_l2306_230624


namespace NUMINAMATH_GPT_height_percentage_difference_l2306_230665

theorem height_percentage_difference
  (h_B h_A : ℝ)
  (hA_def : h_A = h_B * 0.55) :
  ((h_B - h_A) / h_A) * 100 = 81.82 := by 
  sorry

end NUMINAMATH_GPT_height_percentage_difference_l2306_230665


namespace NUMINAMATH_GPT_tetris_blocks_form_square_l2306_230696

-- Definitions of Tetris blocks types
inductive TetrisBlock
| A | B | C | D | E | F | G

open TetrisBlock

-- Definition of a block's ability to form a square
def canFormSquare (block: TetrisBlock) : Prop :=
  block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G

-- The main theorem statement
theorem tetris_blocks_form_square : ∀ (block : TetrisBlock), canFormSquare block → block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G := 
by
  intros block h
  exact h

end NUMINAMATH_GPT_tetris_blocks_form_square_l2306_230696


namespace NUMINAMATH_GPT_multiplication_factor_correct_l2306_230610

theorem multiplication_factor_correct (N X : ℝ) (h1 : 98 = abs ((N * X - N / 10) / (N * X)) * 100) : X = 5 := by
  sorry

end NUMINAMATH_GPT_multiplication_factor_correct_l2306_230610


namespace NUMINAMATH_GPT_probability_of_two_red_balls_l2306_230673

-- Definitions of quantities
def total_balls := 11
def red_balls := 3
def blue_balls := 4 
def green_balls := 4 
def balls_picked := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  ((red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1) / balls_picked)) = 3 / 55 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_red_balls_l2306_230673


namespace NUMINAMATH_GPT_two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l2306_230614

def is_singular_number (n : ℕ) (num : ℕ) : Prop :=
  let first_n_digits := num / 10^n;
  let last_n_digits := num % 10^n;
  (num > 0) ∧
  (first_n_digits > 0) ∧
  (last_n_digits > 0) ∧
  (first_n_digits < 10^n) ∧
  (last_n_digits < 10^n) ∧
  (num = first_n_digits * 10^n + last_n_digits) ∧
  (∃ k, num = k^2) ∧
  (∃ k, first_n_digits = k^2) ∧
  (∃ k, last_n_digits = k^2)

-- (1) Prove that 49 is a two-digit singular number and 1681 is a four-digit singular number
theorem two_and_four_digit_singular_numbers :
  is_singular_number 1 49 ∧ is_singular_number 2 1681 :=
sorry

-- (2) Prove that 256036 is a six-digit singular number
theorem six_digit_singular_number :
  is_singular_number 3 256036 :=
sorry

-- (3) Prove the existence of a 20-digit singular number
theorem exists_twenty_digit_singular_number :
  ∃ num, is_singular_number 10 num :=
sorry

-- (4) Prove that there are at most 10 singular numbers with 100 digits
theorem at_most_ten_singular_numbers_with_100_digits :
  ∃! n, n <= 10 ∧ ∀ num, num < 10^100 → is_singular_number 50 num → num < 10 ∧ num > 0 :=
sorry

-- (5) Prove the existence of a 30-digit singular number
theorem exists_thirty_digit_singular_number :
  ∃ num, is_singular_number 15 num :=
sorry

end NUMINAMATH_GPT_two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l2306_230614


namespace NUMINAMATH_GPT_total_number_of_coins_l2306_230692

variable (nickels dimes total_value : ℝ)
variable (total_nickels : ℕ)

def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10

theorem total_number_of_coins :
  total_value = 3.50 → total_nickels = 30 → total_value = total_nickels * value_of_nickel + dimes * value_of_dime → 
  total_nickels + dimes = 50 :=
by
  intros h_total_value h_total_nickels h_value_equation
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l2306_230692


namespace NUMINAMATH_GPT_water_current_speed_l2306_230639

theorem water_current_speed (v : ℝ) (swimmer_speed : ℝ := 4) (time : ℝ := 3.5) (distance : ℝ := 7) :
  (4 - v) = distance / time → v = 2 := 
by
  sorry

end NUMINAMATH_GPT_water_current_speed_l2306_230639


namespace NUMINAMATH_GPT_trig_identity_proof_l2306_230678

theorem trig_identity_proof 
  (α : ℝ) 
  (h1 : Real.sin (4 * α) = 2 * Real.sin (2 * α) * Real.cos (2 * α))
  (h2 : Real.cos (4 * α) = Real.cos (2 * α) ^ 2 - Real.sin (2 * α) ^ 2) : 
  (1 - 2 * Real.sin (2 * α) ^ 2) / (1 - Real.sin (4 * α)) = 
  (1 + Real.tan (2 * α)) / (1 - Real.tan (2 * α)) := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l2306_230678


namespace NUMINAMATH_GPT_find_A_l2306_230682

def clubsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem find_A (A : ℝ) : clubsuit A 6 = 31 → A = 10.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_A_l2306_230682


namespace NUMINAMATH_GPT_cupcakes_per_package_l2306_230667

theorem cupcakes_per_package
  (packages : ℕ) (total_left : ℕ) (cupcakes_eaten : ℕ) (initial_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_packages = 3)
  (h2 : cupcakes_eaten = 5)
  (h3 : total_left = 7)
  (h4 : packages = initial_packages * cupcakes_per_package - cupcakes_eaten)
  (h5 : packages = total_left) : 
  cupcakes_per_package = 4 := 
by
  sorry

end NUMINAMATH_GPT_cupcakes_per_package_l2306_230667
