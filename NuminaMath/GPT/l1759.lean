import Mathlib

namespace NUMINAMATH_GPT_skyler_total_songs_skyler_success_breakdown_l1759_175927

noncomputable def skyler_songs : ℕ :=
  let hit_songs := 25
  let top_100_songs := hit_songs + 10
  let unreleased_songs := hit_songs - 5
  let duets_total := 12
  let duets_top_20 := duets_total / 2
  let duets_not_top_200 := duets_total / 2
  let soundtracks_total := 18
  let soundtracks_extremely := 3
  let soundtracks_moderate := 8
  let soundtracks_lukewarm := 7
  let projects_total := 22
  let projects_global := 1
  let projects_regional := 7
  let projects_overlooked := 14
  hit_songs + top_100_songs + unreleased_songs + duets_total + soundtracks_total + projects_total

theorem skyler_total_songs : skyler_songs = 132 := by
  sorry

theorem skyler_success_breakdown :
  let extremely_successful := 25 + 1
  let successful := 35 + 6 + 3
  let moderately_successful := 8 + 7
  let less_successful := 7 + 14 + 6
  let unreleased := 20
  (extremely_successful, successful, moderately_successful, less_successful, unreleased) =
  (26, 44, 15, 27, 20) := by
  sorry

end NUMINAMATH_GPT_skyler_total_songs_skyler_success_breakdown_l1759_175927


namespace NUMINAMATH_GPT_simplify_expression_l1759_175931

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ((x + y) ^ 2 - (x - y) ^ 2) / (4 * x * y) = 1 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1759_175931


namespace NUMINAMATH_GPT_solve_system_of_equations_l1759_175958

theorem solve_system_of_equations (x y : ℝ) :
  (3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6) ∧
  (x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7) →
  (x = 1 / 2) ∧ (y = -3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1759_175958


namespace NUMINAMATH_GPT_average_score_is_8_9_l1759_175966

-- Define the scores and their frequencies
def scores : List ℝ := [7.5, 8.5, 9, 10]
def frequencies : List ℕ := [2, 2, 3, 3]

-- Express the condition that the total number of shots is 10
def total_shots : ℕ := frequencies.sum

-- Calculate the weighted sum of the scores
def weighted_sum (scores : List ℝ) (frequencies : List ℕ) : ℝ :=
  (List.zip scores frequencies).foldl (λ acc (sc, freq) => acc + (sc * freq)) 0

-- Prove that the average score is 8.9
theorem average_score_is_8_9 :
  total_shots = 10 →
  weighted_sum scores frequencies / total_shots = 8.9 :=
by
  intros h_total_shots
  sorry

end NUMINAMATH_GPT_average_score_is_8_9_l1759_175966


namespace NUMINAMATH_GPT_rectangle_area_proof_l1759_175908

def rectangle_width : ℕ := 5

def rectangle_length (width : ℕ) : ℕ := 3 * width

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem rectangle_area_proof : rectangle_area (rectangle_length rectangle_width) rectangle_width = 75 := by
  sorry -- Proof can be added later

end NUMINAMATH_GPT_rectangle_area_proof_l1759_175908


namespace NUMINAMATH_GPT_exists_constant_not_geometric_l1759_175953

-- Definitions for constant and geometric sequences
def is_constant_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, seq n = c

def is_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, seq (n + 1) = r * seq n

-- The negation problem statement
theorem exists_constant_not_geometric :
  ∃ seq : ℕ → ℝ, is_constant_sequence seq ∧ ¬is_geometric_sequence seq :=
sorry

end NUMINAMATH_GPT_exists_constant_not_geometric_l1759_175953


namespace NUMINAMATH_GPT_hours_worked_l1759_175952

theorem hours_worked (w e : ℝ) (hw : w = 6.75) (he : e = 67.5) 
  : e / w = 10 := by
  sorry

end NUMINAMATH_GPT_hours_worked_l1759_175952


namespace NUMINAMATH_GPT_solution_set_of_bx2_ax_c_lt_zero_l1759_175983

theorem solution_set_of_bx2_ax_c_lt_zero (a b c : ℝ) (h1 : a > 0) (h2 : b = a) (h3 : c = -6 * a) (h4 : ∀ x, ax^2 - bx + c < 0 ↔ -2 < x ∧ x < 3) :
  ∀ x, bx^2 + ax + c < 0 ↔ -3 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_bx2_ax_c_lt_zero_l1759_175983


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1759_175993

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : S 2 = 2 * a 2 + 3)
  (h2 : S 3 = 2 * a 3 + 3)
  (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) : q = 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1759_175993


namespace NUMINAMATH_GPT_anya_age_l1759_175960

theorem anya_age (n : ℕ) (h : 110 ≤ (n * (n + 1)) / 2 ∧ (n * (n + 1)) / 2 ≤ 130) : n = 15 :=
sorry

end NUMINAMATH_GPT_anya_age_l1759_175960


namespace NUMINAMATH_GPT_distance_min_value_l1759_175928

theorem distance_min_value (a b c d : ℝ) 
  (h₁ : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  (a - c)^2 + (b - d)^2 = 9 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_min_value_l1759_175928


namespace NUMINAMATH_GPT_sum_of_ages_of_alex_and_allison_is_47_l1759_175912

theorem sum_of_ages_of_alex_and_allison_is_47 (diane_age_now : ℕ)
  (diane_age_at_30_alex_relation : diane_age_now + 14 = 30 ∧ diane_age_now + 14 = 60 / 2)
  (diane_age_at_30_allison_relation : diane_age_now + 14 = 30 ∧ 30 = 2 * (diane_age_now + 14 - (30 - 15)))
  : (60 - (30 - 16)) + (15 - (30 - 16)) = 47 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_of_alex_and_allison_is_47_l1759_175912


namespace NUMINAMATH_GPT_f_is_odd_l1759_175921

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

end NUMINAMATH_GPT_f_is_odd_l1759_175921


namespace NUMINAMATH_GPT_distance_between_intersections_l1759_175978

theorem distance_between_intersections (a : ℝ) (a_pos : 0 < a) : 
  |(Real.log a / Real.log 2) - (Real.log (a / 3) / Real.log 2)| = Real.log 3 / Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_intersections_l1759_175978


namespace NUMINAMATH_GPT_distance_between_cars_after_third_checkpoint_l1759_175941

theorem distance_between_cars_after_third_checkpoint
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (speed_after_first : ℝ)
  (speed_after_second : ℝ)
  (speed_after_third : ℝ)
  (distance_travelled : ℝ) :
  initial_distance = 100 →
  initial_speed = 60 →
  speed_after_first = 80 →
  speed_after_second = 100 →
  speed_after_third = 120 →
  distance_travelled = 200 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_cars_after_third_checkpoint_l1759_175941


namespace NUMINAMATH_GPT_price_of_pants_l1759_175929

theorem price_of_pants (P : ℝ) (h1 : 4 * 33 = 132) (h2 : 2 * P + 132 = 240) : P = 54 :=
sorry

end NUMINAMATH_GPT_price_of_pants_l1759_175929


namespace NUMINAMATH_GPT_range_of_k_l1759_175998

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → k * (Real.exp (k * x) + 1) - ((1 / x) + 1) * Real.log x > 0) ↔ k > 1 / Real.exp 1 := 
  sorry

end NUMINAMATH_GPT_range_of_k_l1759_175998


namespace NUMINAMATH_GPT_smallest_number_divisible_l1759_175955

theorem smallest_number_divisible (x : ℕ) :
  (∃ n : ℕ, x = n * 5 + 24) ∧
  (∃ n : ℕ, x = n * 10 + 24) ∧
  (∃ n : ℕ, x = n * 15 + 24) ∧
  (∃ n : ℕ, x = n * 20 + 24) →
  x = 84 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_l1759_175955


namespace NUMINAMATH_GPT_first_percentage_reduction_l1759_175987

theorem first_percentage_reduction (P : ℝ) (x : ℝ) :
  (P - (x / 100) * P) * 0.4 = P * 0.3 → x = 25 := by
  sorry

end NUMINAMATH_GPT_first_percentage_reduction_l1759_175987


namespace NUMINAMATH_GPT_pupils_who_like_both_l1759_175988

theorem pupils_who_like_both (total_pupils pizza_lovers burger_lovers : ℕ) (h1 : total_pupils = 200) (h2 : pizza_lovers = 125) (h3 : burger_lovers = 115) :
  (pizza_lovers + burger_lovers - total_pupils = 40) :=
by
  sorry

end NUMINAMATH_GPT_pupils_who_like_both_l1759_175988


namespace NUMINAMATH_GPT_shaded_fraction_is_one_fourth_l1759_175937

def quilt_block_shaded_fraction : ℚ :=
  let total_unit_squares := 16
  let triangles_per_unit_square := 2
  let shaded_triangles := 8
  let shaded_unit_squares := shaded_triangles / triangles_per_unit_square
  shaded_unit_squares / total_unit_squares

theorem shaded_fraction_is_one_fourth :
  quilt_block_shaded_fraction = 1 / 4 :=
sorry

end NUMINAMATH_GPT_shaded_fraction_is_one_fourth_l1759_175937


namespace NUMINAMATH_GPT_rachel_reading_pages_l1759_175985

theorem rachel_reading_pages (M T : ℕ) (hM : M = 10) (hT : T = 23) : T - M = 3 := 
by
  rw [hM, hT]
  norm_num
  sorry

end NUMINAMATH_GPT_rachel_reading_pages_l1759_175985


namespace NUMINAMATH_GPT_verify_chebyshev_polynomials_l1759_175999

-- Define the Chebyshev polynomials of the first kind Tₙ(x)
def T : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n+1), x => 2 * x * T n x - T (n-1) x

-- Define the Chebyshev polynomials of the second kind Uₙ(x)
def U : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => 2 * x
| (n+1), x => 2 * x * U n x - U (n-1) x

-- State the theorem to verify the Chebyshev polynomials initial conditions and recurrence relations
theorem verify_chebyshev_polynomials (n : ℕ) (x : ℝ) :
  T 0 x = 1 ∧ T 1 x = x ∧
  U 0 x = 1 ∧ U 1 x = 2 * x ∧
  (T (n+1) x = 2 * x * T n x - T (n-1) x) ∧
  (U (n+1) x = 2 * x * U n x - U (n-1) x) := sorry

end NUMINAMATH_GPT_verify_chebyshev_polynomials_l1759_175999


namespace NUMINAMATH_GPT_coronavirus_diameter_in_meters_l1759_175976

theorem coronavirus_diameter_in_meters (n : ℕ) (h₁ : 1 = (10 : ℤ) ^ 9) (h₂ : n = 125) :
  (n * 10 ^ (-9 : ℤ) : ℝ) = 1.25 * 10 ^ (-7 : ℤ) :=
by
  sorry

end NUMINAMATH_GPT_coronavirus_diameter_in_meters_l1759_175976


namespace NUMINAMATH_GPT_statement_A_statement_C_statement_D_statement_B_l1759_175907

variable (a b : ℝ)

theorem statement_A :
  4 * a^2 - a * b + b^2 = 1 → |a| ≤ 2 * Real.sqrt 15 / 15 :=
sorry

theorem statement_C :
  (4 * a^2 - a * b + b^2 = 1) → 4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3 :=
sorry

theorem statement_D :
  4 * a^2 - a * b + b^2 = 1 → |2 * a - b| ≤ 2 * Real.sqrt 10 / 5 :=
sorry

theorem statement_B :
  4 * a^2 - a * b + b^2 = 1 → ¬(|a + b| < 1) :=
sorry

end NUMINAMATH_GPT_statement_A_statement_C_statement_D_statement_B_l1759_175907


namespace NUMINAMATH_GPT_remainder_5310_mod8_l1759_175946

theorem remainder_5310_mod8 : (53 ^ 10) % 8 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_5310_mod8_l1759_175946


namespace NUMINAMATH_GPT_quadratic_min_value_l1759_175949

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end NUMINAMATH_GPT_quadratic_min_value_l1759_175949


namespace NUMINAMATH_GPT_coefficients_equality_l1759_175913

theorem coefficients_equality (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : a_1 * (x-1)^4 + a_2 * (x-1)^3 + a_3 * (x-1)^2 + a_4 * (x-1) + a_5 = x^4)
  (h1 : a_1 = 1)
  (h2 : a_5 = 1)
  (h3 : 1 - a_2 + a_3 - a_4 + 1 = 0) :
  a_2 - a_3 + a_4 = 2 :=
sorry

end NUMINAMATH_GPT_coefficients_equality_l1759_175913


namespace NUMINAMATH_GPT_vector_calculation_l1759_175959

def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (-1, 6)
def v3 : ℝ × ℝ := (2, -1)

theorem vector_calculation :
  (5:ℝ) • v1 - (3:ℝ) • v2 + v3 = (20, -44) :=
by
  sorry

end NUMINAMATH_GPT_vector_calculation_l1759_175959


namespace NUMINAMATH_GPT_john_took_11_more_l1759_175902

/-- 
If Ray took 10 chickens, Ray took 6 chickens less than Mary, and 
John took 5 more chickens than Mary, then John took 11 more 
chickens than Ray. 
-/
theorem john_took_11_more (R M J : ℕ) (h1 : R = 10) 
  (h2 : R + 6 = M) (h3 : M + 5 = J) : J - R = 11 :=
by
  sorry

end NUMINAMATH_GPT_john_took_11_more_l1759_175902


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l1759_175956

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)  
  (h : 19 / x + 98 / y = 1) : x + y ≥ 203 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l1759_175956


namespace NUMINAMATH_GPT_person_income_l1759_175950

theorem person_income 
    (income expenditure savings : ℕ) 
    (h1 : income = 3 * (income / 3)) 
    (h2 : expenditure = 2 * (income / 3)) 
    (h3 : savings = 7000) 
    (h4 : income = expenditure + savings) : 
    income = 21000 := 
by 
  sorry

end NUMINAMATH_GPT_person_income_l1759_175950


namespace NUMINAMATH_GPT_geometric_sequence_a1_l1759_175982

theorem geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) 
  (hq : 0 < q)
  (h1 : a 4 * a 8 = 2 * (a 5) ^ 2)
  (h2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a1_l1759_175982


namespace NUMINAMATH_GPT_total_coins_is_16_l1759_175911

theorem total_coins_is_16 (x y : ℕ) (h₁ : x ≠ y) (h₂ : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := 
sorry

end NUMINAMATH_GPT_total_coins_is_16_l1759_175911


namespace NUMINAMATH_GPT_solve_for_y_l1759_175964

theorem solve_for_y (y : ℕ) (h : 5 * (2 ^ y) = 320) : y = 6 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l1759_175964


namespace NUMINAMATH_GPT_problem_b_c_constants_l1759_175972

theorem problem_b_c_constants (b c : ℝ) (h : ∀ x : ℝ, (x + 2) * (x + b) = x^2 + c * x + 6) : c = 5 := 
by sorry

end NUMINAMATH_GPT_problem_b_c_constants_l1759_175972


namespace NUMINAMATH_GPT_polygon_sides_l1759_175932

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1759_175932


namespace NUMINAMATH_GPT_correct_statement_l1759_175940

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 2))
noncomputable def g (x : ℝ) : ℝ := Real.cos (x + (3 * Real.pi / 2))

theorem correct_statement (x : ℝ) : f (x - (Real.pi / 2)) = g x :=
by sorry

end NUMINAMATH_GPT_correct_statement_l1759_175940


namespace NUMINAMATH_GPT_perfect_number_divisibility_l1759_175943

theorem perfect_number_divisibility (P : ℕ) (h1 : P > 28) (h2 : Nat.Perfect P) (h3 : 7 ∣ P) : 49 ∣ P := 
sorry

end NUMINAMATH_GPT_perfect_number_divisibility_l1759_175943


namespace NUMINAMATH_GPT_problem_1_problem_2_l1759_175915

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x - 3|

theorem problem_1 (a x : ℝ) (h1 : a < 3) (h2 : (∀ x, f x a >= 4 ↔ x ≤ 1/2 ∨ x ≥ 9/2)) : 
  a = 2 :=
sorry

theorem problem_2 (a : ℝ) (h1 : ∀ x : ℝ, f x a + |x - 3| ≥ 1) : 
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1759_175915


namespace NUMINAMATH_GPT_sequence_periodic_of_period_9_l1759_175979

theorem sequence_periodic_of_period_9 (a : ℕ → ℤ) (h : ∀ n, a (n + 2) = |a (n + 1)| - a n) (h_nonzero : ∃ n, a n ≠ 0) :
  ∃ m, ∃ k, m > 0 ∧ k > 0 ∧ (∀ n, a (n + m + k) = a (n + m)) ∧ k = 9 :=
by
  sorry

end NUMINAMATH_GPT_sequence_periodic_of_period_9_l1759_175979


namespace NUMINAMATH_GPT_least_distance_between_ticks_l1759_175925

theorem least_distance_between_ticks (x : ℚ) :
  (∀ n : ℕ, ∃ k : ℤ, k = n * 11 ∨ k = n * 13) →
  x = 1 / 143 :=
by
  sorry

end NUMINAMATH_GPT_least_distance_between_ticks_l1759_175925


namespace NUMINAMATH_GPT_find_a2_l1759_175905

noncomputable def a_sequence (k : ℕ+) (n : ℕ) : ℚ :=
  -(1 / 2 : ℚ) * n^2 + k * n

theorem find_a2
  (k : ℕ+)
  (max_S : ∀ n : ℕ, a_sequence k n ≤ 8)
  (max_reached : ∃ n : ℕ, a_sequence k n = 8) :
  a_sequence 4 2 - a_sequence 4 1 = 5 / 2 :=
by
  -- To be proved, insert appropriate steps here
  sorry

end NUMINAMATH_GPT_find_a2_l1759_175905


namespace NUMINAMATH_GPT_find_wrong_observation_value_l1759_175922

-- Defining the given conditions
def original_mean : ℝ := 36
def corrected_mean : ℝ := 36.5
def num_observations : ℕ := 50
def correct_value : ℝ := 30

-- Defining the given sums based on means
def original_sum : ℝ := num_observations * original_mean
def corrected_sum : ℝ := num_observations * corrected_mean

-- The wrong value can be calculated based on the difference
def wrong_value : ℝ := correct_value + (corrected_sum - original_sum)

-- The theorem to prove
theorem find_wrong_observation_value (h : original_sum = 1800) (h' : corrected_sum = 1825) :
  wrong_value = 55 :=
sorry

end NUMINAMATH_GPT_find_wrong_observation_value_l1759_175922


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1759_175995

theorem hyperbola_asymptotes
    (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (1 + (b^2) / (a^2))) :
    (∀ x y : ℝ, (y = x * Real.sqrt 3) ∨ (y = -x * Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1759_175995


namespace NUMINAMATH_GPT_solve_system_equations_l1759_175930

theorem solve_system_equations (x y : ℝ) :
  x + y = 0 ∧ 2 * x + 3 * y = 3 → x = -3 ∧ y = 3 :=
by {
  -- Leave the proof as a placeholder with "sorry".
  sorry
}

end NUMINAMATH_GPT_solve_system_equations_l1759_175930


namespace NUMINAMATH_GPT_find_x_l1759_175986

-- Define the conditions
def condition (x : ℕ) := (4 * x)^2 - 2 * x = 8062

-- State the theorem
theorem find_x : ∃ x : ℕ, condition x ∧ x = 134 := sorry

end NUMINAMATH_GPT_find_x_l1759_175986


namespace NUMINAMATH_GPT_smallest_number_of_students_l1759_175934

theorem smallest_number_of_students 
  (n : ℕ) 
  (h1 : 4 * 80 + (n - 4) * 50 ≤ 65 * n) :
  n = 8 :=
by sorry

end NUMINAMATH_GPT_smallest_number_of_students_l1759_175934


namespace NUMINAMATH_GPT_no_integral_points_on_AB_l1759_175942

theorem no_integral_points_on_AB (k m n : ℤ) (h1: ((m^3 - m)^2 + (n^3 - n)^2 > (3*k + 1)^2)) :
  ¬ ∃ (x y : ℤ), (m^3 - m) * x + (n^3 - n) * y = (3*k + 1)^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_integral_points_on_AB_l1759_175942


namespace NUMINAMATH_GPT_find_years_ago_twice_age_l1759_175970

-- Definitions of given conditions
def age_sum (H J : ℕ) : Prop := H + J = 43
def henry_age : ℕ := 27
def jill_age : ℕ := 16

-- Definition of the problem to be proved
theorem find_years_ago_twice_age (X : ℕ) 
  (h1 : age_sum henry_age jill_age) 
  (h2 : henry_age = 27) 
  (h3 : jill_age = 16) : (27 - X = 2 * (16 - X)) → X = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_years_ago_twice_age_l1759_175970


namespace NUMINAMATH_GPT_range_of_a_l1759_175990

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - (x^2 / 2) - a * x - 1

theorem range_of_a (x : ℝ) (a : ℝ) (h : 1 ≤ x) : (0 ≤ f a x) → (a ≤ Real.exp 1 - 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1759_175990


namespace NUMINAMATH_GPT_pie_eating_contest_l1759_175996

def a : ℚ := 7 / 8
def b : ℚ := 5 / 6
def difference : ℚ := 1 / 24

theorem pie_eating_contest : a - b = difference := 
sorry

end NUMINAMATH_GPT_pie_eating_contest_l1759_175996


namespace NUMINAMATH_GPT_circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l1759_175961

theorem circumscribe_quadrilateral_a : 
  ∃ (x : ℝ), 2 * x + 4 * x + 5 * x + 3 * x = 360 
          ∧ (2 * x + 5 * x = 180) 
          ∧ (4 * x + 3 * x = 180) := sorry

theorem circumscribe_quadrilateral_b : 
  ∃ (x : ℝ), 5 * x + 7 * x + 8 * x + 9 * x = 360 
          ∧ (5 * x + 8 * x ≠ 180) 
          ∧ (7 * x + 9 * x ≠ 180) := sorry

end NUMINAMATH_GPT_circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l1759_175961


namespace NUMINAMATH_GPT_find_angle_four_l1759_175935

theorem find_angle_four (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle1 + angle3 + 60 = 180)
  (h3 : angle3 = angle4) :
  angle4 = 60 :=
by sorry

end NUMINAMATH_GPT_find_angle_four_l1759_175935


namespace NUMINAMATH_GPT_xy_value_l1759_175980

theorem xy_value (x y : ℝ) (h : x ≠ y) (h_eq : x^2 + 2 / x^2 = y^2 + 2 / y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l1759_175980


namespace NUMINAMATH_GPT_tom_dimes_count_l1759_175916

def originalDimes := 15
def dimesFromDad := 33
def dimesSpent := 11

theorem tom_dimes_count : originalDimes + dimesFromDad - dimesSpent = 37 := by
  sorry

end NUMINAMATH_GPT_tom_dimes_count_l1759_175916


namespace NUMINAMATH_GPT_max_net_income_is_50000_l1759_175962

def tax_rate (y : ℝ) : ℝ :=
  10 * y ^ 2

def net_income (y : ℝ) : ℝ :=
  1000 * y - tax_rate y

theorem max_net_income_is_50000 :
  ∃ y : ℝ, (net_income y = 25000 ∧ 1000 * y = 50000) :=
by
  use 50
  sorry

end NUMINAMATH_GPT_max_net_income_is_50000_l1759_175962


namespace NUMINAMATH_GPT_binomial_coefficient_30_3_l1759_175906

theorem binomial_coefficient_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_30_3_l1759_175906


namespace NUMINAMATH_GPT_twenty_five_question_test_l1759_175948

def not_possible_score (score total_questions correct_points unanswered_points incorrect_points : ℕ) : Prop :=
  ∀ correct unanswered incorrect : ℕ,
    correct + unanswered + incorrect = total_questions →
    correct * correct_points + unanswered * unanswered_points + incorrect * incorrect_points ≠ score

theorem twenty_five_question_test :
  not_possible_score 96 25 4 2 0 :=
by
  sorry

end NUMINAMATH_GPT_twenty_five_question_test_l1759_175948


namespace NUMINAMATH_GPT_total_amount_in_bank_l1759_175974

-- Definition of the checks and their values
def checks_1mil : Nat := 25
def checks_100k : Nat := 8
def value_1mil : Nat := 1000000
def value_100k : Nat := 100000

-- The proof statement
theorem total_amount_in_bank 
  (total : Nat) 
  (h1 : checks_1mil * value_1mil = 25000000)
  (h2 : checks_100k * value_100k = 800000):
  total = 25000000 + 800000 :=
sorry

end NUMINAMATH_GPT_total_amount_in_bank_l1759_175974


namespace NUMINAMATH_GPT_probability_blue_then_red_l1759_175967

/--
A box contains 15 balls, of which 5 are blue and 10 are red.
Two balls are drawn sequentially from the box without returning the first ball to the box.
Prove that the probability that the first ball drawn is blue and the second ball is red is 5 / 21.
-/
theorem probability_blue_then_red :
  let total_balls := 15
  let blue_balls := 5
  let red_balls := 10
  let first_is_blue := (blue_balls : ℚ) / total_balls
  let second_is_red_given_blue := (red_balls : ℚ) / (total_balls - 1)
  first_is_blue * second_is_red_given_blue = 5 / 21 := by
  sorry

end NUMINAMATH_GPT_probability_blue_then_red_l1759_175967


namespace NUMINAMATH_GPT_total_tissues_used_l1759_175981

-- Definitions based on the conditions
def initial_tissues := 97
def remaining_tissues := 47
def alice_tissues := 12
def bob_tissues := 2 * alice_tissues
def eve_tissues := alice_tissues - 3
def carol_tissues := initial_tissues - remaining_tissues
def friends_tissues := alice_tissues + bob_tissues + eve_tissues

-- The theorem to prove
theorem total_tissues_used : carol_tissues + friends_tissues = 95 := sorry

end NUMINAMATH_GPT_total_tissues_used_l1759_175981


namespace NUMINAMATH_GPT_degree_measure_of_supplement_of_complement_of_35_degree_angle_l1759_175917

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end NUMINAMATH_GPT_degree_measure_of_supplement_of_complement_of_35_degree_angle_l1759_175917


namespace NUMINAMATH_GPT_minimum_value_x_plus_3y_plus_6z_l1759_175919

theorem minimum_value_x_plus_3y_plus_6z 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : 0 < y) 
  (h3 : 0 < z) 
  (h4 : x * y * z = 18) : 
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end NUMINAMATH_GPT_minimum_value_x_plus_3y_plus_6z_l1759_175919


namespace NUMINAMATH_GPT_rajeev_share_of_profit_l1759_175945

open Nat

theorem rajeev_share_of_profit (profit : ℕ) (ramesh_xyz_ratio1 ramesh_xyz_ratio2 xyz_rajeev_ratio1 xyz_rajeev_ratio2 : ℕ) (rajeev_ratio_part : ℕ) (total_parts : ℕ) (individual_part_value : ℕ) :
  profit = 36000 →
  ramesh_xyz_ratio1 = 5 →
  ramesh_xyz_ratio2 = 4 →
  xyz_rajeev_ratio1 = 8 →
  xyz_rajeev_ratio2 = 9 →
  rajeev_ratio_part = 9 →
  total_parts = ramesh_xyz_ratio1 * (xyz_rajeev_ratio1 / ramesh_xyz_ratio2) + xyz_rajeev_ratio1 + xyz_rajeev_ratio2 →
  individual_part_value = profit / total_parts →
  rajeev_ratio_part * individual_part_value = 12000 := 
sorry

end NUMINAMATH_GPT_rajeev_share_of_profit_l1759_175945


namespace NUMINAMATH_GPT_students_drawn_in_sample_l1759_175936

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_students_drawn_in_sample_l1759_175936


namespace NUMINAMATH_GPT_exchange_5_dollars_to_francs_l1759_175991

-- Define the exchange rates
def dollar_to_lire (d : ℕ) : ℕ := d * 5000
def lire_to_francs (l : ℕ) : ℕ := (l / 1000) * 3

-- Define the main theorem
theorem exchange_5_dollars_to_francs : lire_to_francs (dollar_to_lire 5) = 75 :=
by
  sorry

end NUMINAMATH_GPT_exchange_5_dollars_to_francs_l1759_175991


namespace NUMINAMATH_GPT_average_roots_of_quadratic_l1759_175918

open Real

theorem average_roots_of_quadratic (a b : ℝ) (h_eq : ∃ x1 x2 : ℝ, a * x1^2 - 2 * a * x1 + b = 0 ∧ a * x2^2 - 2 * a * x2 + b = 0):
  (b = b) → (a ≠ 0) → (h_discriminant : (2 * a)^2 - 4 * a * b ≥ 0) → (x1 + x2) / 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_average_roots_of_quadratic_l1759_175918


namespace NUMINAMATH_GPT_janet_initial_action_figures_l1759_175947

theorem janet_initial_action_figures (x : ℕ) :
  (x - 2 + 2 * (x - 2) = 24) -> x = 10 := 
by
  sorry

end NUMINAMATH_GPT_janet_initial_action_figures_l1759_175947


namespace NUMINAMATH_GPT_simplify_fraction_expression_l1759_175926

theorem simplify_fraction_expression : 
  (18 / 42 - 3 / 8 - 1 / 12 : ℚ) = -5 / 168 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_expression_l1759_175926


namespace NUMINAMATH_GPT_simplify_expression_l1759_175944

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  (3 * x - 1 - 5 * x) / 3 = -(2 / 3) * x - (1 / 3) := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1759_175944


namespace NUMINAMATH_GPT_find_m_range_a_l1759_175951

noncomputable def f (x m : ℝ) : ℝ :=
  m - |x - 3|

theorem find_m (m : ℝ) (h : ∀ x, 2 < f x m ↔ 2 < x ∧ x < 4) : m = 3 :=
  sorry

theorem range_a (a : ℝ) (h : ∀ x, |x - a| ≥ f x 3) : a ≤ 0 ∨ 6 ≤ a :=
  sorry

end NUMINAMATH_GPT_find_m_range_a_l1759_175951


namespace NUMINAMATH_GPT_distance_problem_l1759_175965

theorem distance_problem (x y n : ℝ) (h1 : y = 15) (h2 : Real.sqrt ((x - 2) ^ 2 + (15 - 7) ^ 2) = 13) (h3 : x > 2) :
  n = Real.sqrt ((2 + Real.sqrt 105) ^ 2 + 15 ^ 2) := by
  sorry

end NUMINAMATH_GPT_distance_problem_l1759_175965


namespace NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1759_175957

theorem fifth_term_arithmetic_sequence (a d : ℤ) 
  (h_twentieth : a + 19 * d = 12) 
  (h_twenty_first : a + 20 * d = 16) : 
  a + 4 * d = -48 := 
by sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_sequence_l1759_175957


namespace NUMINAMATH_GPT_unique_sequence_l1759_175984

/-- Define an infinite sequence of positive real numbers -/
def infinite_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n, 0 < X n

/-- Define the recurrence relation for the sequence -/
def recurrence_relation (X : ℕ → ℝ) : Prop :=
  ∀ n, X (n + 2) = (1 / 2) * (1 / X (n + 1) + X n)

/-- Prove that the only infinite sequence satisfying the recurrence relation is the constant sequence 1 -/
theorem unique_sequence (X : ℕ → ℝ) (h_seq : infinite_sequence X) (h_recur : recurrence_relation X) :
  ∀ n, X n = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_sequence_l1759_175984


namespace NUMINAMATH_GPT_base6_sub_base9_to_base10_l1759_175933

theorem base6_sub_base9_to_base10 :
  (3 * 6^2 + 2 * 6^1 + 5 * 6^0) - (2 * 9^2 + 1 * 9^1 + 5 * 9^0) = -51 :=
by
  sorry

end NUMINAMATH_GPT_base6_sub_base9_to_base10_l1759_175933


namespace NUMINAMATH_GPT_smallest_interesting_number_l1759_175975

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end NUMINAMATH_GPT_smallest_interesting_number_l1759_175975


namespace NUMINAMATH_GPT_complement_A_correct_l1759_175973

def A : Set ℝ := {x | 1 - (8 / (x - 2)) < 0}

def complement_A : Set ℝ := {x | x ≤ 2 ∨ x ≥ 10}

theorem complement_A_correct : (Aᶜ = complement_A) :=
by {
  -- Placeholder for the necessary proof
  sorry
}

end NUMINAMATH_GPT_complement_A_correct_l1759_175973


namespace NUMINAMATH_GPT_lloyd_hourly_rate_l1759_175914

variable (R : ℝ)  -- Lloyd's regular hourly rate

-- Conditions
def lloyd_works_regular_hours_per_day : Prop := R > 0
def lloyd_earns_excess_rate : Prop := 1.5 * R > 0
def lloyd_worked_hours : Prop := 10.5 > 7.5
def lloyd_earned_amount : Prop := 7.5 * R + 3 * 1.5 * R = 66

-- Theorem statement
theorem lloyd_hourly_rate (hr_pos : lloyd_works_regular_hours_per_day R)
                           (excess_rate : lloyd_earns_excess_rate R)
                           (worked_hours : lloyd_worked_hours)
                           (earned_amount : lloyd_earned_amount R) : 
    R = 5.5 :=
by sorry

end NUMINAMATH_GPT_lloyd_hourly_rate_l1759_175914


namespace NUMINAMATH_GPT_total_cards_square_l1759_175954

theorem total_cards_square (s : ℕ) (h_perim : 4 * s - 4 = 240) : s * s = 3721 := by
  sorry

end NUMINAMATH_GPT_total_cards_square_l1759_175954


namespace NUMINAMATH_GPT_perceived_temperature_difference_l1759_175977

theorem perceived_temperature_difference (N : ℤ) (M L : ℤ)
  (h1 : M = L + N)
  (h2 : M - 11 - (L + 5) = 6 ∨ M - 11 - (L + 5) = -6) :
  N = 22 ∨ N = 10 := by
  sorry

end NUMINAMATH_GPT_perceived_temperature_difference_l1759_175977


namespace NUMINAMATH_GPT_range_of_a_l1759_175920

variable (a : ℝ)

-- Definitions of propositions p and q
def p := ∀ x : ℝ, x^2 - 2*x - a ≥ 0
def q := ∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0

-- Lean 4 statement of the proof problem
theorem range_of_a : ¬ p a ∧ q a → -1 < a ∧ a ≤ 5/8 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1759_175920


namespace NUMINAMATH_GPT_max_liters_of_water_heated_l1759_175904

theorem max_liters_of_water_heated
  (heat_initial : ℕ := 480) 
  (heat_drop : ℝ := 0.25)
  (temp_initial : ℝ := 20)
  (temp_boiling : ℝ := 100)
  (specific_heat_capacity : ℝ := 4.2)
  (kJ_to_liters_conversion : ℝ := 336) :
  (∀ m : ℕ, (m * kJ_to_liters_conversion > ((heat_initial : ℝ) / (1 - heat_drop)) → m ≤ 5)) :=
by
  sorry

end NUMINAMATH_GPT_max_liters_of_water_heated_l1759_175904


namespace NUMINAMATH_GPT_portrait_in_silver_box_l1759_175909

-- Definitions for the first trial
def gold_box_1 : Prop := false
def gold_box_2 : Prop := true
def silver_box_1 : Prop := true
def silver_box_2 : Prop := false
def lead_box_1 : Prop := false
def lead_box_2 : Prop := true

-- Definitions for the second trial
def gold_box_3 : Prop := false
def gold_box_4 : Prop := true
def silver_box_3 : Prop := true
def silver_box_4 : Prop := false
def lead_box_3 : Prop := false
def lead_box_4 : Prop := true

-- The main theorem statement
theorem portrait_in_silver_box
  (gold_b1 : gold_box_1 = false)
  (gold_b2 : gold_box_2 = true)
  (silver_b1 : silver_box_1 = true)
  (silver_b2 : silver_box_2 = false)
  (lead_b1 : lead_box_1 = false)
  (lead_b2 : lead_box_2 = true)
  (gold_b3 : gold_box_3 = false)
  (gold_b4 : gold_box_4 = true)
  (silver_b3 : silver_box_3 = true)
  (silver_b4 : silver_box_4 = false)
  (lead_b3 : lead_box_3 = false)
  (lead_b4 : lead_box_4 = true) : 
  (silver_box_1 ∧ ¬lead_box_2) ∧ (silver_box_3 ∧ ¬lead_box_4) :=
sorry

end NUMINAMATH_GPT_portrait_in_silver_box_l1759_175909


namespace NUMINAMATH_GPT_visitor_increase_l1759_175901

variable (x : ℝ) -- The percentage increase each day

theorem visitor_increase (h1 : 1.2 * (1 + x)^2 = 2.5) : 1.2 * (1 + x)^2 = 2.5 :=
by exact h1

end NUMINAMATH_GPT_visitor_increase_l1759_175901


namespace NUMINAMATH_GPT_garage_sale_items_l1759_175900

theorem garage_sale_items (h : 34 = 13 + n + 1 + 14 - 14) : n = 22 := by
  sorry

end NUMINAMATH_GPT_garage_sale_items_l1759_175900


namespace NUMINAMATH_GPT_total_weight_of_fish_l1759_175997

-- Define the weights of fish caught by Peter, Ali, and Joey.
variables (P A J : ℕ)

-- Ali caught twice as much fish as Peter.
def condition1 := A = 2 * P

-- Joey caught 1 kg more fish than Peter.
def condition2 := J = P + 1

-- Ali caught 12 kg of fish.
def condition3 := A = 12

-- Prove the total weight of the fish caught by all three is 25 kg.
theorem total_weight_of_fish :
  condition1 P A → condition2 P J → condition3 A → P + A + J = 25 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_weight_of_fish_l1759_175997


namespace NUMINAMATH_GPT_roses_problem_l1759_175968

variable (R B C : ℕ)

theorem roses_problem
    (h1 : R = B + 10)
    (h2 : C = 10)
    (h3 : 16 - 6 = C)
    (h4 : B = R - C):
  R = B + 10 ∧ R - C = B := 
by 
  have hC: C = 10 := by linarith
  have hR: R = B + 10 := by linarith
  have hRC: R - C = B := by linarith
  exact ⟨hR, hRC⟩

end NUMINAMATH_GPT_roses_problem_l1759_175968


namespace NUMINAMATH_GPT_stock_values_l1759_175994

theorem stock_values (AA_invest : ℕ) (BB_invest : ℕ) (CC_invest : ℕ)
  (AA_first_year_increase : ℝ) (BB_first_year_decrease : ℝ) (CC_first_year_change : ℝ)
  (AA_second_year_decrease : ℝ) (BB_second_year_increase : ℝ) (CC_second_year_increase : ℝ)
  (A_final : ℝ) (B_final : ℝ) (C_final : ℝ) :
  AA_invest = 150 → BB_invest = 100 → CC_invest = 50 →
  AA_first_year_increase = 1.10 → BB_first_year_decrease = 0.70 → CC_first_year_change = 1 →
  AA_second_year_decrease = 0.95 → BB_second_year_increase = 1.10 → CC_second_year_increase = 1.08 →
  A_final = (AA_invest * AA_first_year_increase) * AA_second_year_decrease →
  B_final = (BB_invest * BB_first_year_decrease) * BB_second_year_increase →
  C_final = (CC_invest * CC_first_year_change) * CC_second_year_increase →
  C_final < B_final ∧ B_final < A_final :=
by
  intros
  sorry

end NUMINAMATH_GPT_stock_values_l1759_175994


namespace NUMINAMATH_GPT_boat_capacity_problem_l1759_175903

variables (L S : ℕ)

theorem boat_capacity_problem
  (h1 : L + 4 * S = 46)
  (h2 : 2 * L + 3 * S = 57) :
  3 * L + 6 * S = 96 :=
sorry

end NUMINAMATH_GPT_boat_capacity_problem_l1759_175903


namespace NUMINAMATH_GPT_alex_buys_17_1_pounds_of_corn_l1759_175924

-- Definitions based on conditions
def corn_cost_per_pound : ℝ := 1.20
def bean_cost_per_pound : ℝ := 0.50
def total_pounds : ℝ := 30
def total_cost : ℝ := 27.00

-- Define the variables
variables (c b : ℝ)

-- Theorem statement to prove the number of pounds of corn Alex buys
theorem alex_buys_17_1_pounds_of_corn (h1 : b + c = total_pounds) (h2 : bean_cost_per_pound * b + corn_cost_per_pound * c = total_cost) :
  c = 17.1 :=
sorry

end NUMINAMATH_GPT_alex_buys_17_1_pounds_of_corn_l1759_175924


namespace NUMINAMATH_GPT_line_intersects_circle_l1759_175910

theorem line_intersects_circle (k : ℝ) : ∀ (x y : ℝ),
  (x + y) ^ 2 = x ^ 2 + y ^ 2 →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2)) ∧ 
  ((-1/2)^2 + (0)^2 < 1) →
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2) := 
by
  intro x y h₁ h₂
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l1759_175910


namespace NUMINAMATH_GPT_polynomial_evaluation_l1759_175969

theorem polynomial_evaluation :
  101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1759_175969


namespace NUMINAMATH_GPT_correct_quotient_l1759_175989

theorem correct_quotient (D : ℕ) (Q : ℕ) (h1 : D = 21 * Q) (h2 : D = 12 * 49) : Q = 28 := 
by
  sorry

end NUMINAMATH_GPT_correct_quotient_l1759_175989


namespace NUMINAMATH_GPT_arithmetic_sequence_20th_term_l1759_175971

-- Definitions for the first term and common difference
def first_term : ℤ := 8
def common_difference : ℤ := -3

-- Define the general term for an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- The specific property we seek to prove: the 20th term is -49
theorem arithmetic_sequence_20th_term : arithmetic_sequence 20 = -49 := by
  -- Proof is omitted, filled with sorry
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_20th_term_l1759_175971


namespace NUMINAMATH_GPT_tuesday_snow_correct_l1759_175963

-- Define the snowfall amounts as given in the conditions
def monday_snow : ℝ := 0.32
def total_snow : ℝ := 0.53

-- Define the amount of snow on Tuesday as per the question to be proved
def tuesday_snow : ℝ := total_snow - monday_snow

-- State the theorem to prove that the snowfall on Tuesday is 0.21 inches
theorem tuesday_snow_correct : tuesday_snow = 0.21 := by
  -- Proof skipped with sorry
  sorry

end NUMINAMATH_GPT_tuesday_snow_correct_l1759_175963


namespace NUMINAMATH_GPT_only_solutions_l1759_175923

theorem only_solutions (m n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (condition : (Nat.choose m 2) - 1 = p^n) :
  (m = 5 ∧ n = 2 ∧ p = 3) ∨ (m = 8 ∧ n = 3 ∧ p = 3) :=
by
  sorry

end NUMINAMATH_GPT_only_solutions_l1759_175923


namespace NUMINAMATH_GPT_intersection_correct_l1759_175992

variable (x : ℝ)

def M : Set ℝ := { x | x^2 > 4 }
def N : Set ℝ := { x | x^2 - 3 * x ≤ 0 }
def NM_intersection : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem intersection_correct :
  {x | (M x) ∧ (N x)} = NM_intersection :=
sorry

end NUMINAMATH_GPT_intersection_correct_l1759_175992


namespace NUMINAMATH_GPT_calculate_expression_l1759_175938

theorem calculate_expression : 1^345 + 5^10 / 5^7 = 126 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1759_175938


namespace NUMINAMATH_GPT_range_a_ineq_value_of_a_plus_b_l1759_175939

open Real

def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)
def g (a x : ℝ) : ℝ := a - abs (x - 2)

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < g a x

theorem range_a_ineq (a : ℝ) : range_a a ↔ 4 < a := sorry

def solution_set (b : ℝ) : Prop :=
  ∀ x : ℝ, f x < g ((13/2) : ℝ) x ↔ (b < x ∧ x < 7/2)

theorem value_of_a_plus_b (b : ℝ) (h : solution_set b) : (13/2) + b = 6 := sorry

end NUMINAMATH_GPT_range_a_ineq_value_of_a_plus_b_l1759_175939
