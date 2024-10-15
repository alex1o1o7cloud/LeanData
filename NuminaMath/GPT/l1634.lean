import Mathlib

namespace NUMINAMATH_GPT_total_coins_received_l1634_163462

theorem total_coins_received (coins_first_day coins_second_day : ℕ) 
  (h_first_day : coins_first_day = 22) 
  (h_second_day : coins_second_day = 12) : 
  coins_first_day + coins_second_day = 34 := 
by 
  sorry

end NUMINAMATH_GPT_total_coins_received_l1634_163462


namespace NUMINAMATH_GPT_age_ratio_in_future_l1634_163465

variables (t j x : ℕ)

theorem age_ratio_in_future:
  (t - 4 = 5 * (j - 4)) → 
  (t - 10 = 6 * (j - 10)) →
  (t + x = 3 * (j + x)) →
  x = 26 := 
by {
  sorry
}

end NUMINAMATH_GPT_age_ratio_in_future_l1634_163465


namespace NUMINAMATH_GPT_angle_equivalence_modulo_l1634_163443

-- Defining the given angles
def theta1 : ℤ := -510
def theta2 : ℤ := 210

-- Proving that the angles are equivalent modulo 360
theorem angle_equivalence_modulo : theta1 % 360 = theta2 % 360 :=
by sorry

end NUMINAMATH_GPT_angle_equivalence_modulo_l1634_163443


namespace NUMINAMATH_GPT_continuous_function_solution_l1634_163478

theorem continuous_function_solution (f : ℝ → ℝ) (a : ℝ) (h_continuous : Continuous f) (h_pos : 0 < a)
    (h_equation : ∀ x, f x = a^x * f (x / 2)) :
    ∃ C : ℝ, ∀ x, f x = C * a^(2 * x) := 
sorry

end NUMINAMATH_GPT_continuous_function_solution_l1634_163478


namespace NUMINAMATH_GPT_avg_of_8_numbers_l1634_163460

theorem avg_of_8_numbers
  (n : ℕ)
  (h₁ : n = 8)
  (sum_first_half : ℝ)
  (h₂ : sum_first_half = 158.4)
  (avg_second_half : ℝ)
  (h₃ : avg_second_half = 46.6) :
  ((sum_first_half + avg_second_half * (n / 2)) / n) = 43.1 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_8_numbers_l1634_163460


namespace NUMINAMATH_GPT_find_x_l1634_163489

theorem find_x (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := 
sorry

end NUMINAMATH_GPT_find_x_l1634_163489


namespace NUMINAMATH_GPT_average_star_rating_is_four_l1634_163466

-- Define the conditions
def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

-- Define total star points as per the conditions
def total_star_points : ℕ := (5 * five_star_reviews) + (4 * four_star_reviews) + (3 * three_star_reviews) + (2 * two_star_reviews)

-- Define the average rating calculation
def average_rating : ℚ := total_star_points / total_reviews

theorem average_star_rating_is_four : average_rating = 4 := 
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_average_star_rating_is_four_l1634_163466


namespace NUMINAMATH_GPT_roberto_starting_salary_l1634_163420

-- Given conditions as Lean definitions
def current_salary : ℝ := 134400
def previous_salary (S : ℝ) : ℝ := 1.40 * S

-- The proof problem statement
theorem roberto_starting_salary (S : ℝ) 
    (h1 : current_salary = 1.20 * previous_salary S) : 
    S = 80000 :=
by
  -- We will insert the proof here
  sorry

end NUMINAMATH_GPT_roberto_starting_salary_l1634_163420


namespace NUMINAMATH_GPT_average_speed_l1634_163453

theorem average_speed (d1 d2 d3 v1 v2 v3 total_distance total_time avg_speed : ℝ)
    (h1 : d1 = 40) (h2 : d2 = 20) (h3 : d3 = 10) 
    (h4 : v1 = 8) (h5 : v2 = 40) (h6 : v3 = 20) 
    (h7 : total_distance = d1 + d2 + d3)
    (h8 : total_time = d1 / v1 + d2 / v2 + d3 / v3) 
    (h9 : avg_speed = total_distance / total_time) : avg_speed = 11.67 :=
by 
  sorry

end NUMINAMATH_GPT_average_speed_l1634_163453


namespace NUMINAMATH_GPT_student_percentage_l1634_163438

theorem student_percentage (s1 s3 overall : ℕ) (percentage_second_subject : ℕ) :
  s1 = 60 →
  s3 = 85 →
  overall = 75 →
  (s1 + percentage_second_subject + s3) / 3 = overall →
  percentage_second_subject = 80 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_student_percentage_l1634_163438


namespace NUMINAMATH_GPT_parabola_intersects_line_exactly_once_l1634_163485

theorem parabola_intersects_line_exactly_once (p q : ℚ) : 
  (∀ x : ℝ, 2 * (x - p) ^ 2 = x - 4 ↔ p = 31 / 8) ∧ 
  (∀ x : ℝ, 2 * x ^ 2 - q = x - 4 ↔ q = 31 / 8) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_intersects_line_exactly_once_l1634_163485


namespace NUMINAMATH_GPT_zhang_qiu_jian_problem_l1634_163411

-- Define the arithmetic sequence
def arithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

-- Sum of first n terms of an arithmetic sequence
def sumArithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem zhang_qiu_jian_problem :
  sumArithmeticSequence 5 (16 / 29) 30 = 390 := 
by 
  sorry

end NUMINAMATH_GPT_zhang_qiu_jian_problem_l1634_163411


namespace NUMINAMATH_GPT_houses_with_neither_l1634_163449

theorem houses_with_neither (T G P GP N : ℕ) (hT : T = 65) (hG : G = 50) (hP : P = 40) (hGP : GP = 35) (hN : N = T - (G + P - GP)) :
  N = 10 :=
by
  rw [hT, hG, hP, hGP] at hN
  exact hN

-- Proof is not required, just the statement is enough.

end NUMINAMATH_GPT_houses_with_neither_l1634_163449


namespace NUMINAMATH_GPT_initial_puppies_l1634_163492

-- Define the conditions
variable (a : ℕ) (t : ℕ) (p_added : ℕ) (p_total_adopted : ℕ)

-- State the theorem with the conditions and the target proof
theorem initial_puppies
  (h₁ : a = 3) 
  (h₂ : t = 2)
  (h₃ : p_added = 3)
  (h₄ : p_total_adopted = a * t) :
  (p_total_adopted - p_added) = 3 :=
sorry

end NUMINAMATH_GPT_initial_puppies_l1634_163492


namespace NUMINAMATH_GPT_joan_gave_sam_43_seashells_l1634_163432

def joan_original_seashells : ℕ := 70
def joan_seashells_left : ℕ := 27
def seashells_given_to_sam : ℕ := 43

theorem joan_gave_sam_43_seashells :
  joan_original_seashells - joan_seashells_left = seashells_given_to_sam :=
by
  sorry

end NUMINAMATH_GPT_joan_gave_sam_43_seashells_l1634_163432


namespace NUMINAMATH_GPT_area_of_shaded_region_l1634_163480

-- Definitions of given conditions
def octagon_side_length : ℝ := 5
def arc_radius : ℝ := 4

-- Theorem statement
theorem area_of_shaded_region : 
  let octagon_area := 50
  let sectors_area := 16 * Real.pi
  octagon_area - sectors_area = 50 - 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1634_163480


namespace NUMINAMATH_GPT_cake_eating_contest_l1634_163498

-- Define the fractions representing the amounts of cake eaten by the two students.
def first_student : ℚ := 7 / 8
def second_student : ℚ := 5 / 6

-- The statement of our proof problem
theorem cake_eating_contest : first_student - second_student = 1 / 24 := by
  sorry

end NUMINAMATH_GPT_cake_eating_contest_l1634_163498


namespace NUMINAMATH_GPT_monotonic_increasing_range_l1634_163499

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, (3*x^2 + 2*x - a) ≥ 0) ↔ (a ≤ -1/3) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_range_l1634_163499


namespace NUMINAMATH_GPT_minimize_sum_AP_BP_l1634_163488

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (1, 0)
def center : point := (3, 4)
def radius : ℝ := 2

def on_circle (P : point) : Prop := (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

def AP_squared (P : point) : ℝ := (P.1 - A.1)^2 + (P.2 - A.2)^2
def BP_squared (P : point) : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
def sum_AP_BP_squared (P : point) : ℝ := AP_squared P + BP_squared P

theorem minimize_sum_AP_BP :
  ∀ P : point, on_circle P → sum_AP_BP_squared P = AP_squared (9/5, 12/5) + BP_squared (9/5, 12/5) → 
  P = (9/5, 12/5) :=
sorry

end NUMINAMATH_GPT_minimize_sum_AP_BP_l1634_163488


namespace NUMINAMATH_GPT_city_A_fare_higher_than_city_B_l1634_163439

def fare_in_city_A (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

def fare_in_city_B (x : ℝ) : ℝ :=
  8 + 2.5 * (x - 3)

theorem city_A_fare_higher_than_city_B (x : ℝ) (h : x > 3) :
  fare_in_city_A x > fare_in_city_B x → 3 < x ∧ x < 7 :=
by
  sorry

end NUMINAMATH_GPT_city_A_fare_higher_than_city_B_l1634_163439


namespace NUMINAMATH_GPT_negation_of_forall_inequality_l1634_163405

theorem negation_of_forall_inequality:
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 - x > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_inequality_l1634_163405


namespace NUMINAMATH_GPT_sunzi_wood_problem_l1634_163422

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end NUMINAMATH_GPT_sunzi_wood_problem_l1634_163422


namespace NUMINAMATH_GPT_lcm_prime_factors_l1634_163477

-- Conditions
def n1 := 48
def n2 := 180
def n3 := 250

-- The equivalent proof problem
theorem lcm_prime_factors (l : ℕ) (h1: l = Nat.lcm n1 (Nat.lcm n2 n3)) :
  l = 18000 ∧ (∀ a : ℕ, a ∣ l ↔ a ∣ 2^4 * 3^2 * 5^3) :=
by
  sorry

end NUMINAMATH_GPT_lcm_prime_factors_l1634_163477


namespace NUMINAMATH_GPT_find_base_l1634_163483

def distinct_three_digit_numbers (b : ℕ) : ℕ :=
    (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)

theorem find_base (b : ℕ) (h : distinct_three_digit_numbers b = 144) : b = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_base_l1634_163483


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l1634_163427

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l1634_163427


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1634_163469

theorem solve_quadratic_1 (x : ℝ) : x^2 - 4 * x + 1 = 0 → x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by sorry

theorem solve_quadratic_2 (x : ℝ) : x^2 - 5 * x + 6 = 0 → x = 2 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1634_163469


namespace NUMINAMATH_GPT_train_speed_correct_l1634_163476

-- Definitions based on the conditions in a)
def train_length_meters : ℝ := 160
def time_seconds : ℝ := 4

-- Correct answer identified in b)
def expected_speed_kmh : ℝ := 144

-- Proof statement verifying that speed computed from the conditions equals the expected speed
theorem train_speed_correct :
  train_length_meters / 1000 / (time_seconds / 3600) = expected_speed_kmh :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l1634_163476


namespace NUMINAMATH_GPT_people_got_off_train_l1634_163467

theorem people_got_off_train (initial_people : ℕ) (people_left : ℕ) (people_got_off : ℕ) 
  (h1 : initial_people = 48) 
  (h2 : people_left = 31) 
  : people_got_off = 17 := by
  sorry

end NUMINAMATH_GPT_people_got_off_train_l1634_163467


namespace NUMINAMATH_GPT_intersecting_lines_implies_a_eq_c_l1634_163487

theorem intersecting_lines_implies_a_eq_c
  (k b a c : ℝ)
  (h_kb : k ≠ b)
  (exists_point : ∃ (x y : ℝ), (y = k * x + k) ∧ (y = b * x + b) ∧ (y = a * x + c)) :
  a = c := 
sorry

end NUMINAMATH_GPT_intersecting_lines_implies_a_eq_c_l1634_163487


namespace NUMINAMATH_GPT_woman_year_of_birth_l1634_163442

def year_of_birth (x : ℕ) : ℕ := x^2 - x

theorem woman_year_of_birth : ∃ (x : ℕ), 1850 ≤ year_of_birth x ∧ year_of_birth x < 1900 ∧ year_of_birth x = 1892 :=
by
  sorry

end NUMINAMATH_GPT_woman_year_of_birth_l1634_163442


namespace NUMINAMATH_GPT_standard_deviation_is_two_l1634_163434

def weights : List ℝ := [125, 124, 121, 123, 127]

noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum / l.length)

noncomputable def variance (l : List ℝ) : ℝ :=
  ((l.map (λ x => (x - mean l)^2)).sum / l.length)

noncomputable def standard_deviation (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_two : standard_deviation weights = 2 := 
by
  sorry

end NUMINAMATH_GPT_standard_deviation_is_two_l1634_163434


namespace NUMINAMATH_GPT_preimage_of_mapping_l1634_163402

def f (a b : ℝ) : ℝ × ℝ := (a + 2 * b, 2 * a - b)

theorem preimage_of_mapping : ∃ (a b : ℝ), f a b = (3, 1) ∧ (a, b) = (1, 1) :=
by
  sorry

end NUMINAMATH_GPT_preimage_of_mapping_l1634_163402


namespace NUMINAMATH_GPT_MrKozelGarden_l1634_163456

theorem MrKozelGarden :
  ∀ (x y : ℕ), 
  (y = 3 * x + 1) ∧ (y = 4 * (x - 1)) → (x = 5 ∧ y = 16) := 
by
  intros x y h
  sorry

end NUMINAMATH_GPT_MrKozelGarden_l1634_163456


namespace NUMINAMATH_GPT_heesu_received_most_sweets_l1634_163451

theorem heesu_received_most_sweets
  (total_sweets : ℕ)
  (minsus_sweets : ℕ)
  (jaeyoungs_sweets : ℕ)
  (heesus_sweets : ℕ)
  (h_total : total_sweets = 30)
  (h_minsu : minsus_sweets = 12)
  (h_jaeyoung : jaeyoungs_sweets = 3)
  (h_heesu : heesus_sweets = 15) :
  heesus_sweets = max minsus_sweets (max jaeyoungs_sweets heesus_sweets) :=
by sorry

end NUMINAMATH_GPT_heesu_received_most_sweets_l1634_163451


namespace NUMINAMATH_GPT_vector_coordinates_l1634_163475

theorem vector_coordinates (A B : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (-1, 2)) :
  B - A = (-1, 1) :=
sorry

end NUMINAMATH_GPT_vector_coordinates_l1634_163475


namespace NUMINAMATH_GPT_city_map_scale_l1634_163400

theorem city_map_scale 
  (map_length : ℝ) (actual_length_km : ℝ) (actual_length_cm : ℝ) (conversion_factor : ℝ)
  (h1 : map_length = 240) 
  (h2 : actual_length_km = 18)
  (h3 : actual_length_cm = actual_length_km * conversion_factor)
  (h4 : conversion_factor = 100000) :
  map_length / actual_length_cm = 1 / 7500 :=
by
  sorry

end NUMINAMATH_GPT_city_map_scale_l1634_163400


namespace NUMINAMATH_GPT_mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l1634_163472

-- Definition of Mr. Wang's movements
def movements : List Int := [6, -3, 10, -8, 12, -7, -10]

-- Definitions of given conditions
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.3

theorem mr_wang_returned_to_1st_floor :
  (List.sum movements = 0) :=
by
  sorry

theorem mr_wang_electricity_consumption :
  (List.sum (movements.map Int.natAbs) * floor_height * electricity_per_meter = 50.4) :=
by
  sorry

end NUMINAMATH_GPT_mr_wang_returned_to_1st_floor_mr_wang_electricity_consumption_l1634_163472


namespace NUMINAMATH_GPT_inflated_cost_per_person_l1634_163445

def estimated_cost : ℝ := 30e9
def people_sharing : ℝ := 200e6
def inflation_rate : ℝ := 0.05

theorem inflated_cost_per_person :
  (estimated_cost * (1 + inflation_rate)) / people_sharing = 157.5 := by
  sorry

end NUMINAMATH_GPT_inflated_cost_per_person_l1634_163445


namespace NUMINAMATH_GPT_flour_cups_l1634_163413

theorem flour_cups (f : ℚ) (h : f = 4 + 3/4) : (1/3) * f = 1 + 7/12 := by
  sorry

end NUMINAMATH_GPT_flour_cups_l1634_163413


namespace NUMINAMATH_GPT_probability_correct_l1634_163448

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_probability_correct_l1634_163448


namespace NUMINAMATH_GPT_calculate_paving_cost_l1634_163437

theorem calculate_paving_cost
  (length : ℝ) (width : ℝ) (rate_per_sq_meter : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_rate : rate_per_sq_meter = 1200) :
  (length * width * rate_per_sq_meter = 24750) :=
by
  sorry

end NUMINAMATH_GPT_calculate_paving_cost_l1634_163437


namespace NUMINAMATH_GPT_limiting_reactant_and_products_l1634_163429

def balanced_reaction 
  (al_moles : ℕ) (h2so4_moles : ℕ) 
  (al2_so4_3_moles : ℕ) (h2_moles : ℕ) : Prop :=
  2 * al_moles >= 0 ∧ 3 * h2so4_moles >= 0 ∧ 
  al_moles = 2 ∧ h2so4_moles = 3 ∧ 
  al2_so4_3_moles = 1 ∧ h2_moles = 3 ∧ 
  (2 : ℕ) * al_moles + (3 : ℕ) * h2so4_moles = 2 * 2 + 3 * 3

theorem limiting_reactant_and_products :
  balanced_reaction 2 3 1 3 :=
by {
  -- Here we would provide the proof based on the conditions and balances provided in the problem statement.
  sorry
}

end NUMINAMATH_GPT_limiting_reactant_and_products_l1634_163429


namespace NUMINAMATH_GPT_minimum_value_inequality_l1634_163454

theorem minimum_value_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 = 1) :
  (x / (1 - x^2)) + (y / (1 - y^2)) + (z / (1 - z^2)) ≥ (3 * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1634_163454


namespace NUMINAMATH_GPT_sum_of_values_for_one_solution_l1634_163446

noncomputable def sum_of_a_values (a1 a2 : ℝ) : ℝ :=
  a1 + a2

theorem sum_of_values_for_one_solution :
  ∃ a1 a2 : ℝ, 
  (∀ x : ℝ, 4 * x^2 + (a1 + 8) * x + 9 = 0 ∨ 4 * x^2 + (a2 + 8) * x + 9 = 0) ∧
  ((a1 + 8)^2 - 144 = 0) ∧ ((a2 + 8)^2 - 144 = 0) ∧
  sum_of_a_values a1 a2 = -16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_values_for_one_solution_l1634_163446


namespace NUMINAMATH_GPT_wrong_mark_is_43_l1634_163428

theorem wrong_mark_is_43
  (correct_mark : ℕ)
  (wrong_mark : ℕ)
  (num_students : ℕ)
  (avg_increase : ℕ)
  (h_correct : correct_mark = 63)
  (h_num_students : num_students = 40)
  (h_avg_increase : avg_increase = 40 / 2) 
  (h_wrong_avg : (num_students - 1) * (correct_mark + avg_increase) / num_students = (num_students - 1) * (wrong_mark + avg_increase + correct_mark) / num_students) :
  wrong_mark = 43 :=
sorry

end NUMINAMATH_GPT_wrong_mark_is_43_l1634_163428


namespace NUMINAMATH_GPT_Adeline_hourly_wage_l1634_163464

theorem Adeline_hourly_wage
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hours_per_day = 9) 
  (h2 : days_per_week = 5) 
  (h3 : weeks = 7) 
  (h4 : total_earnings = 3780) :
  total_earnings = 12 * (hours_per_day * days_per_week * weeks) :=
by
  sorry

end NUMINAMATH_GPT_Adeline_hourly_wage_l1634_163464


namespace NUMINAMATH_GPT_combined_supply_duration_l1634_163486

variable (third_of_pill_per_third_day : ℕ → Prop)
variable (alternate_days : ℕ → ℕ → Prop)
variable (supply : ℕ)
variable (days_in_month : ℕ)

-- Conditions:
def one_third_per_third_day (p: ℕ) (d: ℕ) : Prop := 
  third_of_pill_per_third_day d ∧ alternate_days d (d + 3)
def total_supply (s: ℕ) := s = 60
def duration_per_pill (d: ℕ) := d = 9
def month_days (m: ℕ) := m = 30

-- Proof Problem Statement:
theorem combined_supply_duration :
  ∀ (s t: ℕ), total_supply s ∧ duration_per_pill t ∧ month_days 30 → 
  (s * t / 30) = 18 :=
by
  intros s t h
  sorry

end NUMINAMATH_GPT_combined_supply_duration_l1634_163486


namespace NUMINAMATH_GPT_cyclist_trip_time_l1634_163414

variable (a v : ℝ)
variable (h1 : a / v = 5)

theorem cyclist_trip_time
  (increase_factor : ℝ := 1.25) :
  (a / (2 * v) + a / (2 * (increase_factor * v)) = 4.5) :=
sorry

end NUMINAMATH_GPT_cyclist_trip_time_l1634_163414


namespace NUMINAMATH_GPT_solve_for_a_and_b_l1634_163415

theorem solve_for_a_and_b (a b : ℤ) :
  (∀ x : ℤ, (x + a) * (x - 2) = x^2 + b * x - 6) →
  a = 3 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_and_b_l1634_163415


namespace NUMINAMATH_GPT_mark_first_vaccine_wait_time_l1634_163433

-- Define the variables and conditions
variable (x : ℕ)
variable (total_wait_time : ℕ)
variable (second_appointment_wait : ℕ)
variable (effectiveness_wait : ℕ)

-- Given conditions
axiom h1 : second_appointment_wait = 20
axiom h2 : effectiveness_wait = 14
axiom h3 : total_wait_time = 38

-- The statement to be proven
theorem mark_first_vaccine_wait_time
  (h4 : x + second_appointment_wait + effectiveness_wait = total_wait_time) :
  x = 4 := by
  sorry

end NUMINAMATH_GPT_mark_first_vaccine_wait_time_l1634_163433


namespace NUMINAMATH_GPT_girl_speed_l1634_163416

theorem girl_speed (distance time : ℝ) (h₁ : distance = 128) (h₂ : time = 32) : distance / time = 4 := 
by 
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_girl_speed_l1634_163416


namespace NUMINAMATH_GPT_shaded_region_area_and_circle_centers_l1634_163426

theorem shaded_region_area_and_circle_centers :
  ∃ (R : ℝ) (center_big center_small1 center_small2 : ℝ × ℝ),
    R = 10 ∧ 
    center_small1 = (4, 0) ∧
    center_small2 = (10, 0) ∧
    center_big = (7, 0) ∧
    (π * R^2) - (π * 4^2 + π * 6^2) = 48 * π :=
by 
  sorry

end NUMINAMATH_GPT_shaded_region_area_and_circle_centers_l1634_163426


namespace NUMINAMATH_GPT_fair_eight_sided_die_probability_l1634_163441

def prob_at_least_seven_at_least_four_times (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem fair_eight_sided_die_probability : prob_at_least_seven_at_least_four_times 5 4 (1 / 4) + (1 / 4) ^ 5 = 1 / 64 :=
by
  sorry

end NUMINAMATH_GPT_fair_eight_sided_die_probability_l1634_163441


namespace NUMINAMATH_GPT_transformed_curve_l1634_163406

def curve_C (x y : ℝ) := (x - y)^2 + y^2 = 1

theorem transformed_curve (x y : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
    A = ![![2, -2], ![0, 1]] →
    (∃ (x0 y0 : ℝ), curve_C x0 y0 ∧ x = 2 * x0 - 2 * y0 ∧ y = y0) →
    (∃ (x y : ℝ), (x^2 / 4 + y^2 = 1)) :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_transformed_curve_l1634_163406


namespace NUMINAMATH_GPT_Bill_donut_combinations_correct_l1634_163421

/-- Bill has to purchase exactly six donuts from a shop with four kinds of donuts, ensuring he gets at least one of each kind. -/
def Bill_donut_combinations : ℕ :=
  let k := 4  -- number of kinds of donuts
  let n := 6  -- total number of donuts Bill needs to buy
  let m := 2  -- remaining donuts after buying one of each kind
  let same_kind := k          -- ways to choose 2 donuts of the same kind
  let different_kind := (k * (k - 1)) / 2  -- ways to choose 2 donuts of different kinds
  same_kind + different_kind

theorem Bill_donut_combinations_correct : Bill_donut_combinations = 10 :=
  by
    sorry  -- Proof is omitted; we assert this statement is true

end NUMINAMATH_GPT_Bill_donut_combinations_correct_l1634_163421


namespace NUMINAMATH_GPT_book_cost_price_l1634_163474

theorem book_cost_price 
  (M : ℝ) (hM : M = 64.54) 
  (h1 : ∃ L : ℝ, 0.92 * L = M ∧ L = 1.25 * 56.12) :
  ∃ C : ℝ, C = 56.12 :=
by
  sorry

end NUMINAMATH_GPT_book_cost_price_l1634_163474


namespace NUMINAMATH_GPT_fg_neg_one_eq_neg_eight_l1634_163440

def f (x : ℤ) : ℤ := x - 4
def g (x : ℤ) : ℤ := x^2 + 2*x - 3

theorem fg_neg_one_eq_neg_eight : f (g (-1)) = -8 := by
  sorry

end NUMINAMATH_GPT_fg_neg_one_eq_neg_eight_l1634_163440


namespace NUMINAMATH_GPT_sum_of_digits_l1634_163419

theorem sum_of_digits (P Q R : ℕ) (hP : P < 10) (hQ : Q < 10) (hR : R < 10)
 (h_sum : P * 1000 + Q * 100 + Q * 10 + R = 2009) : P + Q + R = 10 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_sum_of_digits_l1634_163419


namespace NUMINAMATH_GPT_no_b_satisfies_condition_l1634_163401

noncomputable def f (b x : ℝ) : ℝ :=
  x^2 + 3 * b * x + 5 * b

theorem no_b_satisfies_condition :
  ∀ b : ℝ, ¬ (∃ x : ℝ, ∀ y : ℝ, |f b y| ≤ 5 → y = x) :=
by
  sorry

end NUMINAMATH_GPT_no_b_satisfies_condition_l1634_163401


namespace NUMINAMATH_GPT_hcf_of_two_numbers_900_l1634_163447

theorem hcf_of_two_numbers_900 (A B H : ℕ) (h_lcm : lcm A B = H * 11 * 15) (h_A : A = 900) : gcd A B = 165 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_900_l1634_163447


namespace NUMINAMATH_GPT_math_olympiad_proof_l1634_163444

theorem math_olympiad_proof (scores : Fin 20 → ℕ) 
  (h_diff : ∀ i j, i ≠ j → scores i ≠ scores j) 
  (h_sum : ∀ i j k, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) : 
  ∀ i, scores i > 18 :=
by
  sorry

end NUMINAMATH_GPT_math_olympiad_proof_l1634_163444


namespace NUMINAMATH_GPT_find_a_l1634_163491

noncomputable def binomial_expansion_term_coefficient
  (n : ℕ) (r : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  (2^(n-r)) * ((-a)^r) * (Nat.choose n r) * (x^(n - 2*r))

theorem find_a 
  (a : ℝ)
  (h : binomial_expansion_term_coefficient 7 5 a 1 = 84) 
  : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l1634_163491


namespace NUMINAMATH_GPT_quadratic_vertex_properties_l1634_163418

theorem quadratic_vertex_properties (a : ℝ) (x1 x2 y1 y2 : ℝ) (h_ax : a ≠ 0) (h_sum : x1 + x2 = 2) (h_order : x1 < x2) (h_value : y1 > y2) :
  a < -2 / 5 :=
sorry

end NUMINAMATH_GPT_quadratic_vertex_properties_l1634_163418


namespace NUMINAMATH_GPT_exponent_properties_l1634_163490

variables (a : ℝ) (m n : ℕ)
-- Conditions
axiom h1 : a^m = 3
axiom h2 : a^n = 2

-- Goal
theorem exponent_properties :
  a^(m + n) = 6 :=
by
  sorry

end NUMINAMATH_GPT_exponent_properties_l1634_163490


namespace NUMINAMATH_GPT_triangle_third_side_l1634_163493

open Nat

theorem triangle_third_side (a b c : ℝ) (h1 : a = 4) (h2 : b = 9) (h3 : c > 0) :
  (5 < c ∧ c < 13) ↔ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_l1634_163493


namespace NUMINAMATH_GPT_shirt_price_percentage_l1634_163425

variable (original_price : ℝ) (final_price : ℝ)

def calculate_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_new_sale_price (p : ℝ) : ℝ := 0.80 * p

def calculate_final_price (p : ℝ) : ℝ := 0.85 * p

theorem shirt_price_percentage :
  (original_price = 60) →
  (final_price = calculate_final_price (calculate_new_sale_price (calculate_sale_price original_price))) →
  (final_price / original_price) * 100 = 54.4 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_shirt_price_percentage_l1634_163425


namespace NUMINAMATH_GPT_xyz_squared_sum_l1634_163479

theorem xyz_squared_sum (x y z : ℝ) 
  (h1 : x^2 + 4 * y^2 + 16 * z^2 = 48)
  (h2 : x * y + 4 * y * z + 2 * z * x = 24) :
  x^2 + y^2 + z^2 = 21 :=
sorry

end NUMINAMATH_GPT_xyz_squared_sum_l1634_163479


namespace NUMINAMATH_GPT_incorrect_inequality_l1634_163407

theorem incorrect_inequality (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) : ¬ (a^2 < a * b) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_inequality_l1634_163407


namespace NUMINAMATH_GPT_eq_zero_or_one_if_square_eq_self_l1634_163436

theorem eq_zero_or_one_if_square_eq_self (a : ℝ) (h : a^2 = a) : a = 0 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_eq_zero_or_one_if_square_eq_self_l1634_163436


namespace NUMINAMATH_GPT_find_number_l1634_163404

theorem find_number (x : ℝ) (n : ℝ) (h1 : x = 12) (h2 : (27 / n) * x - 18 = 3 * x + 27) : n = 4 :=
sorry

end NUMINAMATH_GPT_find_number_l1634_163404


namespace NUMINAMATH_GPT_Michael_points_l1634_163410

theorem Michael_points (total_points : ℕ) (num_other_players : ℕ) (avg_points : ℕ) (Michael_points : ℕ) 
  (h1 : total_points = 75)
  (h2 : num_other_players = 5)
  (h3 : avg_points = 6)
  (h4 : Michael_points = total_points - num_other_players * avg_points) :
  Michael_points = 45 := by
  sorry

end NUMINAMATH_GPT_Michael_points_l1634_163410


namespace NUMINAMATH_GPT_monotonicity_of_g_l1634_163455

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb a (|x + 1|)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.logb a (- (3 / 2) * x^2 + a * x)

theorem monotonicity_of_g (a : ℝ) (h : 0 < a ∧ a ≠ 1) (h0 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x a < 0) :
  ∀ x : ℝ, 0 < x ∧ x ≤ a / 3 → (g x a) < (g (x + ε) a) := 
sorry


end NUMINAMATH_GPT_monotonicity_of_g_l1634_163455


namespace NUMINAMATH_GPT_rainfall_ratio_l1634_163468

theorem rainfall_ratio (r_wed tuesday_rate : ℝ)
    (h_monday : 7 * 1 = 7)
    (h_tuesday : 4 * 2 = 8)
    (h_total : 7 + 8 + 2 * r_wed = 23)
    (h_wed_eq: r_wed = 8 / 2)
    (h_tuesday_rate: tuesday_rate = 2) 
    : r_wed / tuesday_rate = 2 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l1634_163468


namespace NUMINAMATH_GPT_two_people_same_birthday_l1634_163450

noncomputable def population : ℕ := 6000000000

noncomputable def max_age_seconds : ℕ := 150 * 366 * 24 * 60 * 60

theorem two_people_same_birthday :
  ∃ (a b : ℕ) (ha : a < population) (hb : b < population) (hab : a ≠ b),
  (∃ (t : ℕ) (ht_a : t < max_age_seconds) (ht_b : t < max_age_seconds), true) :=
by
  sorry

end NUMINAMATH_GPT_two_people_same_birthday_l1634_163450


namespace NUMINAMATH_GPT_total_balloons_is_18_l1634_163473

-- Define the number of balloons each person has
def Fred_balloons : Nat := 5
def Sam_balloons : Nat := 6
def Mary_balloons : Nat := 7

-- Define the total number of balloons
def total_balloons : Nat := Fred_balloons + Sam_balloons + Mary_balloons

-- The theorem statement to prove
theorem total_balloons_is_18 : total_balloons = 18 := sorry

end NUMINAMATH_GPT_total_balloons_is_18_l1634_163473


namespace NUMINAMATH_GPT_min_g_l1634_163424

noncomputable def f (a m x : ℝ) := m + Real.log x / Real.log a -- definition of f(x) = m + logₐ(x)

-- Given conditions
variables (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
variables (m : ℝ)
axiom h_f8 : f a m 8 = 2
axiom h_f1 : f a m 1 = -1

-- Derived expressions
noncomputable def g (x : ℝ) := 2 * f a m x - f a m (x - 1)

-- Theorem statement
theorem min_g : ∃ (x : ℝ), x > 1 ∧ g a m x = 1 ∧ ∀ x' > 1, g a m x' ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_g_l1634_163424


namespace NUMINAMATH_GPT_garden_area_is_correct_l1634_163409

def width_of_property : ℕ := 1000
def length_of_property : ℕ := 2250

def width_of_garden : ℕ := width_of_property / 8
def length_of_garden : ℕ := length_of_property / 10

def area_of_garden : ℕ := width_of_garden * length_of_garden

theorem garden_area_is_correct : area_of_garden = 28125 := by
  -- Skipping proof for the purpose of this example
  sorry

end NUMINAMATH_GPT_garden_area_is_correct_l1634_163409


namespace NUMINAMATH_GPT_range_of_a_l1634_163484

noncomputable def f (a : ℝ) (x : ℝ) := x * Real.log x - a * x^2

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 
0 < a ∧ a < 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1634_163484


namespace NUMINAMATH_GPT_ratio_suspension_to_fingers_toes_l1634_163481

-- Definition of conditions
def suspension_days_per_instance : Nat := 3
def bullying_instances : Nat := 20
def fingers_and_toes : Nat := 20

-- Theorem statement
theorem ratio_suspension_to_fingers_toes :
  (suspension_days_per_instance * bullying_instances) / fingers_and_toes = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_suspension_to_fingers_toes_l1634_163481


namespace NUMINAMATH_GPT_power_of_negative_fraction_l1634_163482

theorem power_of_negative_fraction :
  (- (1/3))^2 = 1/9 := 
by 
  sorry

end NUMINAMATH_GPT_power_of_negative_fraction_l1634_163482


namespace NUMINAMATH_GPT_translate_parabola_l1634_163452

theorem translate_parabola (x y : ℝ) :
  (y = 2 * x^2 + 3) →
  (∃ x y, y = 2 * (x - 3)^2 + 5) :=
sorry

end NUMINAMATH_GPT_translate_parabola_l1634_163452


namespace NUMINAMATH_GPT_initial_money_l1634_163403

theorem initial_money (B S G M : ℕ) 
  (hB : B = 8) 
  (hS : S = 2 * B) 
  (hG : G = 3 * S) 
  (change : ℕ) 
  (h_change : change = 28)
  (h_total : B + S + G + change = M) : 
  M = 100 := 
by 
  sorry

end NUMINAMATH_GPT_initial_money_l1634_163403


namespace NUMINAMATH_GPT_value_of_m_l1634_163459

theorem value_of_m : ∃ (m : ℕ), (3 * 4 * 5 * m = Nat.factorial 8) ∧ m = 672 := by
  sorry

end NUMINAMATH_GPT_value_of_m_l1634_163459


namespace NUMINAMATH_GPT_remainder_2_power_404_l1634_163494

theorem remainder_2_power_404 (y : ℕ) (h_y : y = 2^101) :
  (2^404 + 404) % (2^203 + 2^101 + 1) = 403 := by
sorry

end NUMINAMATH_GPT_remainder_2_power_404_l1634_163494


namespace NUMINAMATH_GPT_michael_total_time_l1634_163457

def time_for_200_meters (distance speed : ℕ) : ℚ :=
  distance / speed

def total_time_per_lap : ℚ :=
  (time_for_200_meters 200 6) + (time_for_200_meters 200 3)

def total_time_8_laps : ℚ :=
  8 * total_time_per_lap

theorem michael_total_time : total_time_8_laps = 800 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_michael_total_time_l1634_163457


namespace NUMINAMATH_GPT_total_combinations_meals_l1634_163461

-- Define the total number of menu items
def menu_items : ℕ := 12

-- Define the function for computing the number of combinations of meals ordered by three people
def combinations_of_meals (n : ℕ) : ℕ := n * n * n

-- Theorem stating the total number of different combinations of meals is 1728
theorem total_combinations_meals : combinations_of_meals menu_items = 1728 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_total_combinations_meals_l1634_163461


namespace NUMINAMATH_GPT_problem_statement_l1634_163470

theorem problem_statement : ∃ n : ℤ, 0 < n ∧ (1 / 3 + 1 / 4 + 1 / 8 + 1 / n : ℚ).den = 1 ∧ ¬ n > 96 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1634_163470


namespace NUMINAMATH_GPT_cylinder_base_ratio_l1634_163497

variable (O : Point) -- origin
variable (a b c : ℝ) -- fixed point
variable (p q : ℝ) -- center of circular base
variable (α β : ℝ) -- intersection points with axis

-- Let O be the origin
-- Let (a, b, c) be the fixed point through which the cylinder passes
-- The cylinder's axis is parallel to the z-axis and the center of its base is (p, q)
-- The cylinder intersects the x-axis at (α, 0, 0) and the y-axis at (0, β, 0)
-- Let α = 2p and β = 2q

theorem cylinder_base_ratio : 
  α = 2 * p ∧ β = 2 * q → (a / p + b / q = 4) := by
  sorry

end NUMINAMATH_GPT_cylinder_base_ratio_l1634_163497


namespace NUMINAMATH_GPT_equidistant_trajectory_l1634_163412

theorem equidistant_trajectory {x y z : ℝ} :
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3 * x - 2 * y - z = 2 :=
sorry

end NUMINAMATH_GPT_equidistant_trajectory_l1634_163412


namespace NUMINAMATH_GPT_stanleyRanMore_l1634_163463

def distanceStanleyRan : ℝ := 0.4
def distanceStanleyWalked : ℝ := 0.2

theorem stanleyRanMore : distanceStanleyRan - distanceStanleyWalked = 0.2 := by
  sorry

end NUMINAMATH_GPT_stanleyRanMore_l1634_163463


namespace NUMINAMATH_GPT_c_share_of_rent_l1634_163417

/-- 
Given the conditions:
- a puts 10 oxen for 7 months,
- b puts 12 oxen for 5 months,
- c puts 15 oxen for 3 months,
- The rent of the pasture is Rs. 210,
Prove that C should pay Rs. 54 as his share of rent.
-/
noncomputable def total_rent : ℝ := 210
noncomputable def oxen_months_a : ℝ := 10 * 7
noncomputable def oxen_months_b : ℝ := 12 * 5
noncomputable def oxen_months_c : ℝ := 15 * 3
noncomputable def total_oxen_months : ℝ := oxen_months_a + oxen_months_b + oxen_months_c

theorem c_share_of_rent : (total_rent / total_oxen_months) * oxen_months_c = 54 :=
by
  sorry

end NUMINAMATH_GPT_c_share_of_rent_l1634_163417


namespace NUMINAMATH_GPT_proof_a6_bounds_l1634_163495

theorem proof_a6_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 :=
by
  sorry

end NUMINAMATH_GPT_proof_a6_bounds_l1634_163495


namespace NUMINAMATH_GPT_zero_is_multiple_of_every_integer_l1634_163423

theorem zero_is_multiple_of_every_integer (x : ℤ) : ∃ n : ℤ, 0 = n * x := by
  use 0
  exact (zero_mul x).symm

end NUMINAMATH_GPT_zero_is_multiple_of_every_integer_l1634_163423


namespace NUMINAMATH_GPT_Kates_hair_length_l1634_163435

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end NUMINAMATH_GPT_Kates_hair_length_l1634_163435


namespace NUMINAMATH_GPT_total_blue_balloons_l1634_163431

def Joan_balloons : Nat := 9
def Sally_balloons : Nat := 5
def Jessica_balloons : Nat := 2

theorem total_blue_balloons : Joan_balloons + Sally_balloons + Jessica_balloons = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_blue_balloons_l1634_163431


namespace NUMINAMATH_GPT_ned_long_sleeve_shirts_l1634_163471

-- Define the conditions
def total_shirts_washed_before_school : ℕ := 29
def short_sleeve_shirts : ℕ := 9
def unwashed_shirts : ℕ := 1

-- Define the proof problem
theorem ned_long_sleeve_shirts (total_shirts_washed_before_school short_sleeve_shirts unwashed_shirts: ℕ) : 
(total_shirts_washed_before_school - unwashed_shirts - short_sleeve_shirts) = 19 :=
by
  -- It is given: 29 total shirts - 1 unwashed shirt = 28 washed shirts
  -- Out of the 28 washed shirts, 9 are short sleeve shirts
  -- Therefore, Ned washed 28 - 9 = 19 long sleeve shirts
  sorry

end NUMINAMATH_GPT_ned_long_sleeve_shirts_l1634_163471


namespace NUMINAMATH_GPT_max_profit_l1634_163408

noncomputable def fixed_cost : ℝ := 2.5

noncomputable def cost (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def revenue (x : ℝ) : ℝ := 0.05 * 1000 * x

noncomputable def profit (x : ℝ) : ℝ :=
  revenue x - cost x - fixed_cost * 10

theorem max_profit : ∃ x_opt : ℝ, ∀ x : ℝ, 0 < x → 
  profit x ≤ profit 100 ∧ x_opt = 100 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l1634_163408


namespace NUMINAMATH_GPT_minimum_sum_of_squares_l1634_163458

theorem minimum_sum_of_squares (α p q : ℝ) 
  (h1: p + q = α - 2) (h2: p * q = - (α + 1)) :
  p^2 + q^2 ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_sum_of_squares_l1634_163458


namespace NUMINAMATH_GPT_basketball_club_boys_l1634_163430

theorem basketball_club_boys (B G : ℕ)
  (h1 : B + G = 30)
  (h2 : B + (1 / 3) * G = 18) : B = 12 := 
by
  sorry

end NUMINAMATH_GPT_basketball_club_boys_l1634_163430


namespace NUMINAMATH_GPT_complex_quadrant_l1634_163496

open Complex

noncomputable def z : ℂ := (2 * I) / (1 - I)

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 2 * I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1634_163496
