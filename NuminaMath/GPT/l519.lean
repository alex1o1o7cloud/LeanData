import Mathlib

namespace NUMINAMATH_GPT_Jasmine_total_weight_in_pounds_l519_51936

-- Definitions for the conditions provided
def weight_chips_ounces : ℕ := 20
def weight_cookies_ounces : ℕ := 9
def bags_chips : ℕ := 6
def tins_cookies : ℕ := 4 * bags_chips
def total_weight_ounces : ℕ := (weight_chips_ounces * bags_chips) + (weight_cookies_ounces * tins_cookies)
def total_weight_pounds : ℕ := total_weight_ounces / 16

-- The proof problem statement
theorem Jasmine_total_weight_in_pounds : total_weight_pounds = 21 := 
by
  sorry

end NUMINAMATH_GPT_Jasmine_total_weight_in_pounds_l519_51936


namespace NUMINAMATH_GPT_smallest_triangle_perimeter_l519_51911

theorem smallest_triangle_perimeter : ∃ (a b c : ℕ), a = 3 ∧ b = a + 1 ∧ c = b + 1 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 12 := by
  sorry

end NUMINAMATH_GPT_smallest_triangle_perimeter_l519_51911


namespace NUMINAMATH_GPT_find_a_l519_51972

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a : 
  ( ∀ a : ℝ, 
    (∀ x : ℝ,  0 ≤ x ∧ x ≤ 1 → f a 0 + f a 1 = a) → a = 1/2 ) :=
sorry

end NUMINAMATH_GPT_find_a_l519_51972


namespace NUMINAMATH_GPT_median_squared_formula_l519_51950

theorem median_squared_formula (a b c m : ℝ) (AC_is_median : 2 * m^2 + c^2 = a^2 + b^2) : 
  m^2 = (1/4) * (2 * a^2 + 2 * b^2 - c^2) := 
by
  sorry

end NUMINAMATH_GPT_median_squared_formula_l519_51950


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l519_51903

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = 2):
  ( ( (2 * m + 1) / m - 1 ) / ( (m^2 - 1) / m ) ) = 1 :=
by
  rw [h] -- Replace m by 2
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l519_51903


namespace NUMINAMATH_GPT_interior_triangles_from_chords_l519_51985

theorem interior_triangles_from_chords (h₁ : ∀ p₁ p₂ p₃ : Prop, ¬(p₁ ∧ p₂ ∧ p₃)) : 
  ∀ (nine_points_on_circle : Finset ℝ) (h₂ : nine_points_on_circle.card = 9), 
    ∃ (triangles : ℕ), triangles = 210 := 
by 
  sorry

end NUMINAMATH_GPT_interior_triangles_from_chords_l519_51985


namespace NUMINAMATH_GPT_other_factor_computation_l519_51914

theorem other_factor_computation (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
  a = 11 → b = 43 → c = 2 → d = 31 → e = 1311 → 33 ∣ 363 →
  a * b * c * d * e = 38428986 :=
by
  intros ha hb hc hd he hdiv
  rw [ha, hb, hc, hd, he]
  -- proof steps go here if required
  sorry

end NUMINAMATH_GPT_other_factor_computation_l519_51914


namespace NUMINAMATH_GPT_exists_multiple_of_10_of_three_distinct_integers_l519_51981

theorem exists_multiple_of_10_of_three_distinct_integers
    (a b c : ℤ) 
    (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    ∃ x y : ℤ, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ (10 ∣ (x^5 * y^3 - x^3 * y^5)) :=
by
  sorry

end NUMINAMATH_GPT_exists_multiple_of_10_of_three_distinct_integers_l519_51981


namespace NUMINAMATH_GPT_find_y_values_l519_51938

theorem find_y_values (x : ℝ) (y : ℝ) 
  (h : x^2 + 4 * ((x / (x + 3))^2) = 64) : 
  y = (x + 3)^2 * (x - 2) / (2 * x + 3) → 
  y = 250 / 3 :=
sorry

end NUMINAMATH_GPT_find_y_values_l519_51938


namespace NUMINAMATH_GPT_sin_subtract_pi_over_6_l519_51901

theorem sin_subtract_pi_over_6 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hcos : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_subtract_pi_over_6_l519_51901


namespace NUMINAMATH_GPT_amphibians_count_l519_51956

-- Define the conditions
def frogs : Nat := 7
def salamanders : Nat := 4
def tadpoles : Nat := 30
def newt : Nat := 1

-- Define the total number of amphibians observed by Hunter
def total_amphibians : Nat := frogs + salamanders + tadpoles + newt

-- State the theorem
theorem amphibians_count : total_amphibians = 42 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_amphibians_count_l519_51956


namespace NUMINAMATH_GPT_speed_of_faster_train_l519_51924

noncomputable def speed_of_slower_train_kmph := 36
def time_to_cross_seconds := 12
def length_of_faster_train_meters := 120

-- Speed of train V_f in kmph 
theorem speed_of_faster_train 
  (relative_speed_mps : ℝ := length_of_faster_train_meters / time_to_cross_seconds)
  (speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * (1000 / 3600))
  (speed_of_faster_train_mps : ℝ := relative_speed_mps + speed_of_slower_train_mps)
  (speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * (3600 / 1000) )
  : speed_of_faster_train_kmph = 72 := 
sorry

end NUMINAMATH_GPT_speed_of_faster_train_l519_51924


namespace NUMINAMATH_GPT_square_integer_2209_implies_value_l519_51996

theorem square_integer_2209_implies_value (x : ℤ) (h : x^2 = 2209) : (2*x + 1)*(2*x - 1) = 8835 :=
by sorry

end NUMINAMATH_GPT_square_integer_2209_implies_value_l519_51996


namespace NUMINAMATH_GPT_ratio_of_ages_l519_51998

-- Definitions of the conditions
def son_current_age : ℕ := 28
def man_current_age : ℕ := son_current_age + 30
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem
theorem ratio_of_ages : (man_age_in_two_years / son_age_in_two_years) = 2 :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l519_51998


namespace NUMINAMATH_GPT_find_sum_of_money_invested_l519_51970

theorem find_sum_of_money_invested (P : ℝ) (h1 : SI_15 = P * (15 / 100) * 2)
                                    (h2 : SI_12 = P * (12 / 100) * 2)
                                    (h3 : SI_15 - SI_12 = 720) : 
                                    P = 12000 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_find_sum_of_money_invested_l519_51970


namespace NUMINAMATH_GPT_find_a5_l519_51910

noncomputable def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
a₁ + (n - 1) * d

theorem find_a5 (a₁ d : ℚ) (h₁ : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 5 - arithmetic_sequence a₁ d 8 = 1)
(h₂ : arithmetic_sequence a₁ d 9 - arithmetic_sequence a₁ d 2 = 5) :
arithmetic_sequence a₁ d 5 = 6 :=
sorry

end NUMINAMATH_GPT_find_a5_l519_51910


namespace NUMINAMATH_GPT_four_digit_number_l519_51954

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_l519_51954


namespace NUMINAMATH_GPT_find_first_half_speed_l519_51963

theorem find_first_half_speed (distance time total_time : ℝ) (v2 : ℝ)
    (h_distance : distance = 300) 
    (h_time : total_time = 11) 
    (h_v2 : v2 = 25) 
    (half_distance : distance / 2 = 150) :
    (150 / (total_time - (150 / v2)) = 30) :=
by
  sorry

end NUMINAMATH_GPT_find_first_half_speed_l519_51963


namespace NUMINAMATH_GPT_sqrt_fraction_arith_sqrt_16_l519_51983

-- Prove that the square root of 4/9 is ±2/3
theorem sqrt_fraction (a b : ℕ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (h_a : a = 4) (h_b : b = 9) : 
    (Real.sqrt (a / (b : ℝ)) = abs (Real.sqrt a / Real.sqrt b)) :=
by
    rw [h_a, h_b]
    sorry

-- Prove that the arithmetic square root of √16 is 4.
theorem arith_sqrt_16 : Real.sqrt (Real.sqrt 16) = 4 :=
by
    sorry

end NUMINAMATH_GPT_sqrt_fraction_arith_sqrt_16_l519_51983


namespace NUMINAMATH_GPT_volume_of_four_cubes_l519_51905

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end NUMINAMATH_GPT_volume_of_four_cubes_l519_51905


namespace NUMINAMATH_GPT_school_students_l519_51909

theorem school_students
  (total_students : ℕ)
  (students_in_both : ℕ)
  (students_chemistry : ℕ)
  (students_biology : ℕ)
  (students_only_chemistry : ℕ)
  (students_only_biology : ℕ)
  (h1 : total_students = students_only_chemistry + students_only_biology + students_in_both)
  (h2 : students_chemistry = 3 * students_biology)
  (students_in_both_eq : students_in_both = 5)
  (total_students_eq : total_students = 43) :
  students_only_chemistry + students_in_both = 36 :=
by
  sorry

end NUMINAMATH_GPT_school_students_l519_51909


namespace NUMINAMATH_GPT_square_perimeter_l519_51916

theorem square_perimeter (s : ℝ) (h1 : (2 * (s + s / 4)) = 40) :
  4 * s = 64 :=
by
  sorry

end NUMINAMATH_GPT_square_perimeter_l519_51916


namespace NUMINAMATH_GPT_solution_exists_l519_51926

theorem solution_exists (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_l519_51926


namespace NUMINAMATH_GPT_largest_divisor_if_n_sq_div_72_l519_51948

theorem largest_divisor_if_n_sq_div_72 (n : ℕ) (h : n > 0) (h72 : 72 ∣ n^2) : ∃ m, m = 12 ∧ m ∣ n :=
by { sorry }

end NUMINAMATH_GPT_largest_divisor_if_n_sq_div_72_l519_51948


namespace NUMINAMATH_GPT_find_investment_sum_l519_51934

theorem find_investment_sum (P : ℝ)
  (h1 : SI_15 = P * (15 / 100) * 2)
  (h2 : SI_12 = P * (12 / 100) * 2)
  (h3 : SI_15 - SI_12 = 420) :
  P = 7000 :=
by
  sorry

end NUMINAMATH_GPT_find_investment_sum_l519_51934


namespace NUMINAMATH_GPT_closed_chain_possible_l519_51971

-- Define the angle constraint
def angle_constraint (θ : ℝ) : Prop :=
  θ ≥ 150

-- Define meshing condition between two gears
def meshed_gears (θ : ℝ) : Prop :=
  angle_constraint θ

-- Define the general condition for a closed chain of gears
def closed_chain (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → meshed_gears 150

theorem closed_chain_possible : closed_chain 61 :=
by sorry

end NUMINAMATH_GPT_closed_chain_possible_l519_51971


namespace NUMINAMATH_GPT_value_of_expression_l519_51953

variable {a b m n x : ℝ}

def opposite (a b : ℝ) : Prop := a = -b
def reciprocal (m n : ℝ) : Prop := m * n = 1
def distance_to_2 (x : ℝ) : Prop := abs (x - 2) = 3

theorem value_of_expression (h1 : opposite a b) (h2 : reciprocal m n) (h3 : distance_to_2 x) :
  (a + b - m * n) * x + (a + b)^2022 + (- m * n)^2023 = 
  if x = 5 then -6 else if x = -1 then 0 else sorry :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l519_51953


namespace NUMINAMATH_GPT_powerjet_30_minutes_500_gallons_per_hour_l519_51945

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end NUMINAMATH_GPT_powerjet_30_minutes_500_gallons_per_hour_l519_51945


namespace NUMINAMATH_GPT_determine_common_ratio_l519_51927

variable (a : ℕ → ℝ) (q : ℝ)

-- Given conditions
axiom a2 : a 2 = 1 / 2
axiom a5 : a 5 = 4
axiom geom_seq_def : ∀ n, a n = a 1 * q ^ (n - 1)

-- Prove the common ratio q == 2
theorem determine_common_ratio : q = 2 :=
by
  -- here we should unfold the proof steps given in the solution
  sorry

end NUMINAMATH_GPT_determine_common_ratio_l519_51927


namespace NUMINAMATH_GPT_abs_gt_two_nec_but_not_suff_l519_51931

theorem abs_gt_two_nec_but_not_suff (x : ℝ) : (|x| > 2 → x < -2) ∧ (¬ (|x| > 2 ↔ x < -2)) := 
sorry

end NUMINAMATH_GPT_abs_gt_two_nec_but_not_suff_l519_51931


namespace NUMINAMATH_GPT_ABCD_area_is_correct_l519_51940

-- Define rectangle ABCD with the given conditions
def ABCD_perimeter (x : ℝ) : Prop :=
  2 * (4 * x + x) = 160

-- Define the area to be proved
def ABCD_area (x : ℝ) : ℝ :=
  4 * (x ^ 2)

-- The proof problem: given the conditions, the area should be 1024 square centimeters
theorem ABCD_area_is_correct (x : ℝ) (h : ABCD_perimeter x) : 
  ABCD_area x = 1024 := 
by {
  sorry
}

end NUMINAMATH_GPT_ABCD_area_is_correct_l519_51940


namespace NUMINAMATH_GPT_range_of_a_l519_51912

theorem range_of_a (a : ℝ) :
  (0 + 0 + a) * (2 - 1 + a) < 0 ↔ (-1 < a ∧ a < 0) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l519_51912


namespace NUMINAMATH_GPT_line_intersects_hyperbola_l519_51949

variables (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0)

def line (x y : ℝ) := a * x - y + b = 0

def hyperbola (x y : ℝ) := x^2 / (|a| / |b|) - y^2 / (|b| / |a|) = 1

theorem line_intersects_hyperbola :
  ∃ x y : ℝ, line a b x y ∧ hyperbola a b x y := 
sorry

end NUMINAMATH_GPT_line_intersects_hyperbola_l519_51949


namespace NUMINAMATH_GPT_initial_players_round_robin_l519_51937

-- Definitions of conditions
def num_matches_round_robin (x : ℕ) : ℕ := x * (x - 1) / 2
def num_matches_after_drop_out (x : ℕ) : ℕ := num_matches_round_robin x - 2 * (x - 4) + 1

-- The theorem statement
theorem initial_players_round_robin (x : ℕ) 
  (two_players_dropped : num_matches_after_drop_out x = 84) 
  (round_robin_condition : num_matches_round_robin x - 2 * (x - 4) + 1 = 84 ∨ num_matches_round_robin x - 2 * (x - 4) = 84) :
  x = 15 :=
sorry

end NUMINAMATH_GPT_initial_players_round_robin_l519_51937


namespace NUMINAMATH_GPT_christine_sales_value_l519_51921

variable {X : ℝ}

def commission_rate : ℝ := 0.12
def personal_needs_percent : ℝ := 0.60
def savings_amount : ℝ := 1152
def savings_percent : ℝ := 0.40

theorem christine_sales_value:
  (savings_percent * (commission_rate * X) = savings_amount) → 
  (X = 24000) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_christine_sales_value_l519_51921


namespace NUMINAMATH_GPT_correct_system_of_equations_l519_51974

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end NUMINAMATH_GPT_correct_system_of_equations_l519_51974


namespace NUMINAMATH_GPT_water_usage_eq_13_l519_51929

theorem water_usage_eq_13 (m x : ℝ) (h : 16 * m = 10 * m + (x - 10) * 2 * m) : x = 13 :=
by sorry

end NUMINAMATH_GPT_water_usage_eq_13_l519_51929


namespace NUMINAMATH_GPT_onions_total_l519_51992

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ) 
  (h1: Sara_onions = 4) (h2: Sally_onions = 5) (h3: Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 :=
by
  sorry

end NUMINAMATH_GPT_onions_total_l519_51992


namespace NUMINAMATH_GPT_height_percentage_difference_l519_51995

theorem height_percentage_difference 
  (r1 h1 r2 h2 : ℝ) 
  (V1_eq_V2 : π * r1^2 * h1 = π * r2^2 * h2)
  (r2_eq_1_2_r1 : r2 = (6 / 5) * r1) :
  h1 = (36 / 25) * h2 :=
by
  sorry

end NUMINAMATH_GPT_height_percentage_difference_l519_51995


namespace NUMINAMATH_GPT_range_of_a_l519_51977

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + Real.exp x - Real.exp (-x)

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l519_51977


namespace NUMINAMATH_GPT_three_digit_number_count_l519_51918

def total_three_digit_numbers : ℕ := 900

def count_ABA : ℕ := 9 * 9  -- 81

def count_ABC : ℕ := 9 * 9 * 8  -- 648

def valid_three_digit_numbers : ℕ := total_three_digit_numbers - (count_ABA + count_ABC)

theorem three_digit_number_count :
  valid_three_digit_numbers = 171 := by
  sorry

end NUMINAMATH_GPT_three_digit_number_count_l519_51918


namespace NUMINAMATH_GPT_stratified_sampling_third_year_students_l519_51967

theorem stratified_sampling_third_year_students 
  (N : ℕ) (N_1 : ℕ) (P_sophomore : ℝ) (n : ℕ) (N_2 : ℕ) :
  N = 2000 →
  N_1 = 760 →
  P_sophomore = 0.37 →
  n = 20 →
  N_2 = Nat.ceil (N - N_1 - P_sophomore * N) →
  Nat.floor ((n : ℝ) / (N : ℝ) * (N_2 : ℝ)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_year_students_l519_51967


namespace NUMINAMATH_GPT_smallest_natural_number_l519_51955

theorem smallest_natural_number :
  ∃ n : ℕ, (n > 0) ∧ (7 * n % 10000 = 2012) ∧ ∀ m : ℕ, (7 * m % 10000 = 2012) → (n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_natural_number_l519_51955


namespace NUMINAMATH_GPT_john_bought_3_croissants_l519_51947

variable (c k : ℕ)

theorem john_bought_3_croissants
  (h1 : c + k = 5)
  (h2 : ∃ n : ℕ, 88 * c + 44 * k = 100 * n) :
  c = 3 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_john_bought_3_croissants_l519_51947


namespace NUMINAMATH_GPT_handshake_problem_l519_51989

-- Define the remainder operation
def r_mod (n : ℕ) (k : ℕ) : ℕ := n % k

-- Define the function F
def F (t : ℕ) : ℕ := r_mod (t^3) 5251

-- The lean theorem statement with the given conditions and expected results
theorem handshake_problem :
  ∃ (x y : ℕ),
    F x = 506 ∧
    F (x + 1) = 519 ∧
    F y = 229 ∧
    F (y + 1) = 231 ∧
    x = 102 ∧
    y = 72 :=
by
  sorry

end NUMINAMATH_GPT_handshake_problem_l519_51989


namespace NUMINAMATH_GPT_arithmetic_sequence_find_side_length_l519_51922

variable (A B C a b c : ℝ)

-- Condition: Given that b(1 + cos(C)) = c(2 - cos(B))
variable (h : b * (1 + Real.cos C) = c * (2 - Real.cos B))

-- Question I: Prove that a + b = 2 * c
theorem arithmetic_sequence (h : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : a + b = 2 * c :=
sorry

-- Additional conditions for Question II
variable (C_eq : C = Real.pi / 3)
variable (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3)

-- Question II: Find c
theorem find_side_length (C_eq : C = Real.pi / 3) (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) : c = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_find_side_length_l519_51922


namespace NUMINAMATH_GPT_probability_of_specific_individual_drawn_on_third_attempt_l519_51930

theorem probability_of_specific_individual_drawn_on_third_attempt :
  let population_size := 6
  let sample_size := 3
  let prob_not_drawn_first_attempt := 5 / 6
  let prob_not_drawn_second_attempt := 4 / 5
  let prob_drawn_third_attempt := 1 / 4
  (prob_not_drawn_first_attempt * prob_not_drawn_second_attempt * prob_drawn_third_attempt) = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_probability_of_specific_individual_drawn_on_third_attempt_l519_51930


namespace NUMINAMATH_GPT_ink_cartridge_15th_month_l519_51964

def months_in_year : ℕ := 12
def first_change_month : ℕ := 1   -- January is the first month

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + (3 * (n - 1))) % months_in_year

theorem ink_cartridge_15th_month : nth_change_month 15 = 7 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_ink_cartridge_15th_month_l519_51964


namespace NUMINAMATH_GPT_larger_number_is_1590_l519_51968

theorem larger_number_is_1590 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_1590_l519_51968


namespace NUMINAMATH_GPT_gcd_45123_32768_l519_51976

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_45123_32768_l519_51976


namespace NUMINAMATH_GPT_arccos_one_half_eq_pi_div_three_l519_51928

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_arccos_one_half_eq_pi_div_three_l519_51928


namespace NUMINAMATH_GPT_students_who_did_not_receive_an_A_l519_51925

def total_students : ℕ := 40
def a_in_literature : ℕ := 10
def a_in_science : ℕ := 18
def a_in_both : ℕ := 6

theorem students_who_did_not_receive_an_A :
  total_students - ((a_in_literature + a_in_science) - a_in_both) = 18 :=
by
  sorry

end NUMINAMATH_GPT_students_who_did_not_receive_an_A_l519_51925


namespace NUMINAMATH_GPT_nat_pairs_solution_l519_51932

theorem nat_pairs_solution (x y : ℕ) :
  2^(2*x+1) + 2^x + 1 = y^2 → (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by
  sorry

end NUMINAMATH_GPT_nat_pairs_solution_l519_51932


namespace NUMINAMATH_GPT_find_b_l519_51969

theorem find_b (a b c d : ℝ) (h : ∃ k : ℝ, 2 * k = π ∧ k * (b / 2) = π) : b = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l519_51969


namespace NUMINAMATH_GPT_west_of_1km_l519_51979

def east_direction (d : Int) : Int :=
  d

def west_direction (d : Int) : Int :=
  -d

theorem west_of_1km :
  east_direction (2) = 2 →
  west_direction (1) = -1 := by
  sorry

end NUMINAMATH_GPT_west_of_1km_l519_51979


namespace NUMINAMATH_GPT_gain_percent_l519_51946

-- Definitions for the problem
variables (MP CP SP : ℝ)
def cost_price := CP = 0.64 * MP
def selling_price := SP = 0.88 * MP

-- The statement to prove
theorem gain_percent (h1 : cost_price MP CP) (h2 : selling_price MP SP) :
  (SP - CP) / CP * 100 = 37.5 := 
sorry

end NUMINAMATH_GPT_gain_percent_l519_51946


namespace NUMINAMATH_GPT_cube_mono_increasing_l519_51920

theorem cube_mono_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

end NUMINAMATH_GPT_cube_mono_increasing_l519_51920


namespace NUMINAMATH_GPT_find_m_l519_51978

noncomputable def hex_to_dec (m : ℕ) : ℕ :=
  3 * 6^4 + m * 6^3 + 5 * 6^2 + 2

theorem find_m (m : ℕ) : hex_to_dec m = 4934 ↔ m = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l519_51978


namespace NUMINAMATH_GPT_maria_total_distance_in_miles_l519_51939

theorem maria_total_distance_in_miles :
  ∀ (steps_per_mile : ℕ) (full_cycles : ℕ) (remaining_steps : ℕ),
    steps_per_mile = 1500 →
    full_cycles = 50 →
    remaining_steps = 25000 →
    (100000 * full_cycles + remaining_steps) / steps_per_mile = 3350 := by
  intros
  sorry

end NUMINAMATH_GPT_maria_total_distance_in_miles_l519_51939


namespace NUMINAMATH_GPT_function_d_is_odd_l519_51973

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Given function
def f (x : ℝ) : ℝ := x^3

-- Proof statement
theorem function_d_is_odd : is_odd_function f := 
by sorry

end NUMINAMATH_GPT_function_d_is_odd_l519_51973


namespace NUMINAMATH_GPT_smallest_w_l519_51917

theorem smallest_w (w : ℕ) (h : 2^5 ∣ 936 * w ∧ 3^3 ∣ 936 * w ∧ 11^2 ∣ 936 * w) : w = 4356 :=
sorry

end NUMINAMATH_GPT_smallest_w_l519_51917


namespace NUMINAMATH_GPT_arrange_athletes_l519_51923

theorem arrange_athletes :
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  (Nat.choose athletes country_athletes) *
  (Nat.choose (athletes - country_athletes) country_athletes) *
  (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
  (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520 :=
by
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  show (Nat.choose athletes country_athletes) *
       (Nat.choose (athletes - country_athletes) country_athletes) *
       (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
       (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520
  sorry

end NUMINAMATH_GPT_arrange_athletes_l519_51923


namespace NUMINAMATH_GPT_simplify_expression_l519_51941

theorem simplify_expression (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 5) * (2 * x - 1) - 
  (2 * x - 1) * (x^2 + 2 * x - 8) + 
  (x^2 - 2 * x + 3) * (2 * x - 1) * (x - 2) = 
  8 * x^4 - 2 * x^3 - 5 * x^2 + 32 * x - 15 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l519_51941


namespace NUMINAMATH_GPT_probability_of_selecting_girl_l519_51975

theorem probability_of_selecting_girl (boys girls : ℕ) (total_students : ℕ) (prob : ℚ) 
  (h1 : boys = 3) 
  (h2 : girls = 2) 
  (h3 : total_students = boys + girls) 
  (h4 : prob = girls / total_students) : 
  prob = 2 / 5 := 
sorry

end NUMINAMATH_GPT_probability_of_selecting_girl_l519_51975


namespace NUMINAMATH_GPT_min_value_product_expression_l519_51902

theorem min_value_product_expression (x : ℝ) : ∃ m, m = -2746.25 ∧ (∀ y : ℝ, (13 - y) * (8 - y) * (13 + y) * (8 + y) ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_product_expression_l519_51902


namespace NUMINAMATH_GPT_moving_circle_passes_through_focus_l519_51944

-- Given conditions
def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def is_tangent_to_line (circle_center_x : ℝ) : Prop :=
  circle_center_x + 2 = 0

-- Prove that the point (2,0) lies on the moving circle
theorem moving_circle_passes_through_focus (circle_center_x circle_center_y : ℝ) :
  is_on_parabola circle_center_x circle_center_y →
  is_tangent_to_line circle_center_x →
  (circle_center_x - 2)^2 + circle_center_y^2 = (circle_center_x + 2)^2 :=
by
  -- Proof skipped with sorry.
  sorry

end NUMINAMATH_GPT_moving_circle_passes_through_focus_l519_51944


namespace NUMINAMATH_GPT_smallest_positive_n_l519_51919

theorem smallest_positive_n (x y z : ℕ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^13) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_n_l519_51919


namespace NUMINAMATH_GPT_find_c_and_d_l519_51908

theorem find_c_and_d :
  ∀ (y c d : ℝ), (y^2 - 5 * y + 5 / y + 1 / (y^2) = 17) ∧ (y = c - Real.sqrt d) ∧ (0 < c) ∧ (0 < d) → (c + d = 106) :=
by
  intros y c d h
  sorry

end NUMINAMATH_GPT_find_c_and_d_l519_51908


namespace NUMINAMATH_GPT_sum_of_squares_l519_51980

variable {x y z a b c : Real}
variable (h₁ : x * y = a) (h₂ : x * z = b) (h₃ : y * z = c)
variable (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem sum_of_squares : x^2 + y^2 + z^2 = (a * b)^2 / (a * b * c) + (a * c)^2 / (a * b * c) + (b * c)^2 / (a * b * c) := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l519_51980


namespace NUMINAMATH_GPT_max_value_of_f_l519_51951

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem: the maximum value of f(x) is 4
theorem max_value_of_f : ∃ x : ℝ, f x = 4 := sorry

end NUMINAMATH_GPT_max_value_of_f_l519_51951


namespace NUMINAMATH_GPT_eqn_abs_3x_minus_2_solution_l519_51990

theorem eqn_abs_3x_minus_2_solution (x : ℝ) :
  (|x + 5| = 3 * x - 2) ↔ x = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eqn_abs_3x_minus_2_solution_l519_51990


namespace NUMINAMATH_GPT_equivalent_shaded_areas_l519_51906

/- 
  Definitions and parameters:
  - l_sq: the side length of the larger square.
  - s_sq: the side length of the smaller square.
-/
variables (l_sq s_sq : ℝ)
  
-- The area of the larger square
def area_larger_square : ℝ := l_sq * l_sq
  
-- The area of the smaller square
def area_smaller_square : ℝ := s_sq * s_sq
  
-- The shaded area in diagram i
def shaded_area_diagram_i : ℝ := area_larger_square l_sq - area_smaller_square s_sq

-- The polygonal areas in diagrams ii and iii
variables (polygon_area_ii polygon_area_iii : ℝ)

-- The theorem to prove the equivalence of the areas
theorem equivalent_shaded_areas :
  polygon_area_ii = shaded_area_diagram_i l_sq s_sq ∧ polygon_area_iii = shaded_area_diagram_i l_sq s_sq :=
sorry

end NUMINAMATH_GPT_equivalent_shaded_areas_l519_51906


namespace NUMINAMATH_GPT_chocolate_game_winner_l519_51993

theorem chocolate_game_winner (m n : ℕ) (h_m : m = 6) (h_n : n = 8) :
  (∃ k : ℕ, (48 - 1) - 2 * k = 0) ↔ true :=
by
  sorry

end NUMINAMATH_GPT_chocolate_game_winner_l519_51993


namespace NUMINAMATH_GPT_f_is_even_l519_51959

noncomputable def f (x : ℝ) : ℝ := x ^ 2

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := 
by
  intros x
  sorry

end NUMINAMATH_GPT_f_is_even_l519_51959


namespace NUMINAMATH_GPT_meeting_time_l519_51952

noncomputable def start_time : ℕ := 13 -- 1 pm in 24-hour format
noncomputable def speed_A : ℕ := 5 -- in kmph
noncomputable def speed_B : ℕ := 7 -- in kmph
noncomputable def initial_distance : ℕ := 24 -- in km

theorem meeting_time : start_time + (initial_distance / (speed_A + speed_B)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_meeting_time_l519_51952


namespace NUMINAMATH_GPT_chicks_increased_l519_51943

theorem chicks_increased (chicks_day1 chicks_day2: ℕ) (H1 : chicks_day1 = 23) (H2 : chicks_day2 = 12) : 
  chicks_day1 + chicks_day2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_chicks_increased_l519_51943


namespace NUMINAMATH_GPT_hexagon_largest_angle_l519_51994

theorem hexagon_largest_angle (x : ℝ) 
  (h_angles_sum : 80 + 100 + x + x + x + (2 * x + 20) = 720) : 
  (2 * x + 20) = 228 :=
by 
  sorry

end NUMINAMATH_GPT_hexagon_largest_angle_l519_51994


namespace NUMINAMATH_GPT_permutations_of_BANANA_l519_51942

theorem permutations_of_BANANA : 
  let word := ["B", "A", "N", "A", "N", "A"]
  let total_letters := 6
  let repeated_A := 3
  (total_letters.factorial / repeated_A.factorial) = 120 :=
by
  sorry

end NUMINAMATH_GPT_permutations_of_BANANA_l519_51942


namespace NUMINAMATH_GPT_binary_to_octal_of_101101110_l519_51958

def binaryToDecimal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 2 + b) 0 (Nat.digits 2 n)

def decimalToOctal (n : Nat) : Nat :=
  List.foldl (fun acc b => acc * 10 + b) 0 (Nat.digits 8 n)

theorem binary_to_octal_of_101101110 :
  decimalToOctal (binaryToDecimal 0b101101110) = 556 :=
by sorry

end NUMINAMATH_GPT_binary_to_octal_of_101101110_l519_51958


namespace NUMINAMATH_GPT_abc_def_ratio_l519_51904

theorem abc_def_ratio (a b c d e f : ℝ)
    (h1 : a / b = 1 / 3)
    (h2 : b / c = 2)
    (h3 : c / d = 1 / 2)
    (h4 : d / e = 3)
    (h5 : e / f = 1 / 8) :
    (a * b * c) / (d * e * f) = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_abc_def_ratio_l519_51904


namespace NUMINAMATH_GPT_fermats_little_theorem_for_q_plus_1_l519_51988

theorem fermats_little_theorem_for_q_plus_1 (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  (q + 1)^(q - 1) % q = 1 := by
  sorry

end NUMINAMATH_GPT_fermats_little_theorem_for_q_plus_1_l519_51988


namespace NUMINAMATH_GPT_perpendicular_line_plane_l519_51915

variables {m : ℝ}

theorem perpendicular_line_plane (h : (4 / 2) = (2 / 1) ∧ (2 / 1) = (m / -1)) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_plane_l519_51915


namespace NUMINAMATH_GPT_total_time_over_weekend_l519_51935

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end NUMINAMATH_GPT_total_time_over_weekend_l519_51935


namespace NUMINAMATH_GPT_average_speed_return_trip_l519_51913

def speed1 : ℝ := 12 -- Speed for the first part of the trip in miles per hour
def distance1 : ℝ := 18 -- Distance for the first part of the trip in miles
def speed2 : ℝ := 10 -- Speed for the second part of the trip in miles per hour
def distance2 : ℝ := 18 -- Distance for the second part of the trip in miles
def total_round_trip_time : ℝ := 7.3 -- Total time for the round trip in hours

theorem average_speed_return_trip :
  let time1 := distance1 / speed1 -- Time taken for the first part of the trip
  let time2 := distance2 / speed2 -- Time taken for the second part of the trip
  let total_time_to_destination := time1 + time2 -- Total time for the trip to the destination
  let time_return_trip := total_round_trip_time - total_time_to_destination -- Time for the return trip
  let return_trip_distance := distance1 + distance2 -- Distance for the return trip (same as to the destination)
  let avg_speed_return_trip := return_trip_distance / time_return_trip -- Average speed for the return trip
  avg_speed_return_trip = 9 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_return_trip_l519_51913


namespace NUMINAMATH_GPT_final_output_M_l519_51987

-- Definitions of the steps in the conditions
def initial_M : ℕ := 1
def increment_M1 (M : ℕ) : ℕ := M + 1
def increment_M2 (M : ℕ) : ℕ := M + 2

-- Define the final value of M after performing the operations
def final_M : ℕ := increment_M2 (increment_M1 initial_M)

-- The statement to prove
theorem final_output_M : final_M = 4 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_final_output_M_l519_51987


namespace NUMINAMATH_GPT_cups_remaining_l519_51957

-- Each definition only directly appears in the conditions problem
def required_cups : ℕ := 7
def added_cups : ℕ := 3

-- The proof problem capturing Joan needs to add 4 more cups of flour.
theorem cups_remaining : required_cups - added_cups = 4 := 
by
  -- The proof is skipped using sorry.
  sorry

end NUMINAMATH_GPT_cups_remaining_l519_51957


namespace NUMINAMATH_GPT_whiteboards_per_class_is_10_l519_51982

-- Definitions from conditions
def classes : ℕ := 5
def ink_per_whiteboard_ml : ℕ := 20
def cost_per_ml_cents : ℕ := 50
def total_cost_cents : ℕ := 100 * 100  -- converting $100 to cents

-- Following the solution, define other useful constants
def cost_per_whiteboard_cents : ℕ := ink_per_whiteboard_ml * cost_per_ml_cents
def total_cost_all_classes_cents : ℕ := classes * total_cost_cents
def total_whiteboards : ℕ := total_cost_all_classes_cents / cost_per_whiteboard_cents
def whiteboards_per_class : ℕ := total_whiteboards / classes

-- We want to prove that each class uses 10 whiteboards.
theorem whiteboards_per_class_is_10 : whiteboards_per_class = 10 :=
  sorry

end NUMINAMATH_GPT_whiteboards_per_class_is_10_l519_51982


namespace NUMINAMATH_GPT_find_sum_of_abs_roots_l519_51961

variable {p q r n : ℤ}

theorem find_sum_of_abs_roots (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2024) (h3 : p * q * r = -n) :
  |p| + |q| + |r| = 100 :=
  sorry

end NUMINAMATH_GPT_find_sum_of_abs_roots_l519_51961


namespace NUMINAMATH_GPT_exists_unique_representation_l519_51966

theorem exists_unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3 * x + y) / 2 :=
sorry

end NUMINAMATH_GPT_exists_unique_representation_l519_51966


namespace NUMINAMATH_GPT_gcd_1734_816_l519_51984

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end NUMINAMATH_GPT_gcd_1734_816_l519_51984


namespace NUMINAMATH_GPT_find_n_l519_51900

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem find_n (n : ℤ) (h : ∃ x, n < x ∧ x < n+1 ∧ f x = 0) : n = 2 :=
sorry

end NUMINAMATH_GPT_find_n_l519_51900


namespace NUMINAMATH_GPT_greatest_possible_z_l519_51999

theorem greatest_possible_z (x y z : ℕ) (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z)
  (hx_cond : 7 < x) (hy_cond : y < 15) (hx_lt_y : x < y) (hz_gt_zero : z > 0) 
  (hy_sub_x_div_z : (y - x) % z = 0) : z = 2 := 
sorry

end NUMINAMATH_GPT_greatest_possible_z_l519_51999


namespace NUMINAMATH_GPT_max_value_of_S_l519_51907

-- Define the sequence sum function
def S (n : ℕ) : ℤ :=
  -2 * (n : ℤ) ^ 3 + 21 * (n : ℤ) ^ 2 + 23 * (n : ℤ)

theorem max_value_of_S :
  ∃ (n : ℕ), S n = 504 ∧ 
             (∀ k : ℕ, S k ≤ 504) :=
sorry

end NUMINAMATH_GPT_max_value_of_S_l519_51907


namespace NUMINAMATH_GPT_bisections_needed_l519_51986

theorem bisections_needed (ε : ℝ) (ε_pos : ε = 0.01) (h : 0 < ε) : 
  ∃ n : ℕ, n ≤ 7 ∧ 1 / (2^n) < ε :=
by
  sorry

end NUMINAMATH_GPT_bisections_needed_l519_51986


namespace NUMINAMATH_GPT_division_subtraction_l519_51960

theorem division_subtraction : 144 / (12 / 3) - 5 = 31 := by
  sorry

end NUMINAMATH_GPT_division_subtraction_l519_51960


namespace NUMINAMATH_GPT_find_x_for_which_ffx_eq_fx_l519_51997

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_for_which_ffx_eq_fx :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_which_ffx_eq_fx_l519_51997


namespace NUMINAMATH_GPT_max_of_four_expressions_l519_51962

theorem max_of_four_expressions :
  996 * 996 > 995 * 997 ∧ 996 * 996 > 994 * 998 ∧ 996 * 996 > 993 * 999 :=
by
  sorry

end NUMINAMATH_GPT_max_of_four_expressions_l519_51962


namespace NUMINAMATH_GPT_find_ratio_of_hyperbola_l519_51991

noncomputable def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem find_ratio_of_hyperbola (a b : ℝ) (h : a > b) 
  (h_asymptote_angle : ∀ α : ℝ, (y = ↑(b / a) * x -> α = 45)) :
  a / b = 1 :=
sorry

end NUMINAMATH_GPT_find_ratio_of_hyperbola_l519_51991


namespace NUMINAMATH_GPT_percentage_of_girls_who_like_basketball_l519_51933

theorem percentage_of_girls_who_like_basketball 
  (total_students : ℕ)
  (percentage_girls : ℝ)
  (percentage_boys_basketball : ℝ)
  (factor_girls_to_boys_not_basketball : ℝ)
  (total_students_eq : total_students = 25)
  (percentage_girls_eq : percentage_girls = 0.60)
  (percentage_boys_basketball_eq : percentage_boys_basketball = 0.40)
  (factor_girls_to_boys_not_basketball_eq : factor_girls_to_boys_not_basketball = 2) 
  : 
  ((factor_girls_to_boys_not_basketball * (total_students * (1 - percentage_girls) * (1 - percentage_boys_basketball))) / 
  (total_students * percentage_girls)) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_girls_who_like_basketball_l519_51933


namespace NUMINAMATH_GPT_kevin_started_with_cards_l519_51965

-- The definitions corresponding to the conditions in the problem
def ended_with : Nat := 54
def found_cards : Nat := 47
def started_with (ended_with found_cards : Nat) : Nat := ended_with - found_cards

-- The Lean statement for the proof problem itself
theorem kevin_started_with_cards : started_with ended_with found_cards = 7 := by
  sorry

end NUMINAMATH_GPT_kevin_started_with_cards_l519_51965
