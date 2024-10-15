import Mathlib

namespace NUMINAMATH_GPT_gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l1215_121567

theorem gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1 :
  Int.gcd (79^7 + 1) (79^7 + 79^3 + 1) = 1 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_gcd_79_pow_7_plus_1_79_pow_7_plus_79_pow_3_plus_1_l1215_121567


namespace NUMINAMATH_GPT_power_function_m_l1215_121506

theorem power_function_m (m : ℝ) 
  (h_even : ∀ x : ℝ, x^m = (-x)^m) 
  (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → x^m > y^m) : m = -2 :=
sorry

end NUMINAMATH_GPT_power_function_m_l1215_121506


namespace NUMINAMATH_GPT_matroskin_milk_amount_l1215_121570

theorem matroskin_milk_amount :
  ∃ S M x : ℝ, S + M = 10 ∧ (S - x) = (1 / 3) * S ∧ (M + x) = 3 * M ∧ (M + x) = 7.5 := 
sorry

end NUMINAMATH_GPT_matroskin_milk_amount_l1215_121570


namespace NUMINAMATH_GPT_f_g_evaluation_l1215_121582

-- Definitions of the functions g and f
def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3 * x - 2

-- Goal: Prove that f(g(2)) = 22
theorem f_g_evaluation : f (g 2) = 22 :=
by
  sorry

end NUMINAMATH_GPT_f_g_evaluation_l1215_121582


namespace NUMINAMATH_GPT_find_x_l1215_121560

theorem find_x 
  (AB AC BC : ℝ) 
  (x : ℝ)
  (hO : π * (AB / 2)^2 = 12 + 2 * x)
  (hP : π * (AC / 2)^2 = 24 + x)
  (hQ : π * (BC / 2)^2 = 108 - x)
  : AC^2 + BC^2 = AB^2 → x = 60 :=
by {
   sorry
}

end NUMINAMATH_GPT_find_x_l1215_121560


namespace NUMINAMATH_GPT_identity_is_only_sum_free_preserving_surjection_l1215_121521

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, f m = n

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ x y : ℕ, x ∈ A → y ∈ A → x + y ∉ A

noncomputable def identity_function_property : Prop :=
  ∀ f : ℕ → ℕ, is_surjective f →
  (∀ A : Set ℕ, is_sum_free A → is_sum_free (Set.image f A)) →
  ∀ n : ℕ, f n = n

theorem identity_is_only_sum_free_preserving_surjection : identity_function_property := sorry

end NUMINAMATH_GPT_identity_is_only_sum_free_preserving_surjection_l1215_121521


namespace NUMINAMATH_GPT_find_side_b_l1215_121591

theorem find_side_b
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 2 * Real.sin B = Real.sin A + Real.sin C)
  (h2 : Real.cos B = 3 / 5)
  (h3 : (1 / 2) * a * c * Real.sin B = 4) :
  b = 4 * Real.sqrt 6 / 3 := 
sorry

end NUMINAMATH_GPT_find_side_b_l1215_121591


namespace NUMINAMATH_GPT_trapezoidal_field_perimeter_l1215_121501

-- Definitions derived from the conditions
def length_of_longer_parallel_side : ℕ := 15
def length_of_shorter_parallel_side : ℕ := 9
def total_perimeter_of_rectangle : ℕ := 52

-- Correct Answer
def correct_perimeter_of_trapezoidal_field : ℕ := 46

-- Theorem statement
theorem trapezoidal_field_perimeter 
  (a b w : ℕ)
  (h1 : a = length_of_longer_parallel_side)
  (h2 : b = length_of_shorter_parallel_side)
  (h3 : 2 * (a + w) = total_perimeter_of_rectangle)
  (h4 : w = 11) -- from the solution calculation
  : a + b + 2 * w = correct_perimeter_of_trapezoidal_field :=
by
  sorry

end NUMINAMATH_GPT_trapezoidal_field_perimeter_l1215_121501


namespace NUMINAMATH_GPT_smallest_positive_integer_congruence_l1215_121590

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 18 [MOD 31] ∧ 0 < x ∧ x < 31 ∧ x = 16 := 
by sorry

end NUMINAMATH_GPT_smallest_positive_integer_congruence_l1215_121590


namespace NUMINAMATH_GPT_find_integer_pairs_l1215_121579

theorem find_integer_pairs (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1215_121579


namespace NUMINAMATH_GPT_y_sum_equals_three_l1215_121512

noncomputable def sum_of_y_values (solutions : List (ℝ × ℝ × ℝ)) : ℝ :=
  solutions.foldl (fun acc (_, y, _) => acc + y) 0

theorem y_sum_equals_three (solutions : List (ℝ × ℝ × ℝ))
  (h1 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → x + y * z = 5)
  (h2 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → y + x * z = 8)
  (h3 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → z + x * y = 12) :
  sum_of_y_values solutions = 3 := sorry

end NUMINAMATH_GPT_y_sum_equals_three_l1215_121512


namespace NUMINAMATH_GPT_neg_p_equiv_l1215_121513

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x / (x - 1) > 0

theorem neg_p_equiv :
  ¬p I ↔ ∃ x ∈ I, x / (x - 1) ≤ 0 ∨ x - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_p_equiv_l1215_121513


namespace NUMINAMATH_GPT_fractional_inequality_solution_set_l1215_121581

theorem fractional_inequality_solution_set (x : ℝ) :
  (x / (x + 1) < 0) ↔ (-1 < x) ∧ (x < 0) :=
sorry

end NUMINAMATH_GPT_fractional_inequality_solution_set_l1215_121581


namespace NUMINAMATH_GPT_appleJuicePercentageIsCorrect_l1215_121534

-- Define the initial conditions
def MikiHas : ℕ × ℕ := (15, 10) -- Miki has 15 apples and 10 bananas

-- Define the juice extraction rates
def appleJuicePerApple : ℚ := 9 / 3 -- 9 ounces from 3 apples
def bananaJuicePerBanana : ℚ := 10 / 2 -- 10 ounces from 2 bananas

-- Define the number of apples and bananas used for the blend
def applesUsed : ℕ := 5
def bananasUsed : ℕ := 4

-- Calculate the total juice extracted
def appleJuice : ℚ := applesUsed * appleJuicePerApple
def bananaJuice : ℚ := bananasUsed * bananaJuicePerBanana

-- Calculate the total juice and percentage of apple juice
def totalJuice : ℚ := appleJuice + bananaJuice
def percentageAppleJuice : ℚ := (appleJuice / totalJuice) * 100

theorem appleJuicePercentageIsCorrect : percentageAppleJuice = 42.86 := by
  sorry

end NUMINAMATH_GPT_appleJuicePercentageIsCorrect_l1215_121534


namespace NUMINAMATH_GPT_least_sum_exponents_of_520_l1215_121537

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_least_sum_exponents_of_520_l1215_121537


namespace NUMINAMATH_GPT_population_increase_l1215_121542

theorem population_increase (birth_rate : ℝ) (death_rate : ℝ) (initial_population : ℝ) :
  initial_population = 1000 →
  birth_rate = 32 / 1000 →
  death_rate = 11 / 1000 →
  ((birth_rate - death_rate) / initial_population) * 100 = 2.1 :=
by
  sorry

end NUMINAMATH_GPT_population_increase_l1215_121542


namespace NUMINAMATH_GPT_circle_convex_polygons_count_l1215_121529

theorem circle_convex_polygons_count : 
  let total_subsets := (2^15 - 1) - (15 + 105 + 455 + 255)
  let final_count := total_subsets - 500
  final_count = 31437 :=
by
  sorry

end NUMINAMATH_GPT_circle_convex_polygons_count_l1215_121529


namespace NUMINAMATH_GPT_arithmetic_geometric_sequences_l1215_121574

noncomputable def geometric_sequence_sum (a q n : ℝ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem arithmetic_geometric_sequences (a : ℝ) (q : ℝ) (S : ℕ → ℝ) :
  q = 2 →
  S 5 = geometric_sequence_sum a q 5 →
  2 * a * q = 6 + a * q^4 →
  S 5 = -31 / 2 :=
by
  intros hq1 hS5 hAR
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequences_l1215_121574


namespace NUMINAMATH_GPT_find_equation_of_line_l1215_121528

theorem find_equation_of_line
  (m b : ℝ) 
  (h1 : ∃ k : ℝ, (k^2 - 2*k + 3 = k*m + b ∧ ∃ d : ℝ, d = 4) 
        ∧ (4*m - k^2 + 2*m*k - 3 + b = 0)) 
  (h2 : 8 = 2*m + b)
  (h3 : b ≠ 0) 
  : y = 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_equation_of_line_l1215_121528


namespace NUMINAMATH_GPT_number_of_elements_l1215_121548

def average_incorrect (N : ℕ) := 21
def correction (incorrect : ℕ) (correct : ℕ) := correct - incorrect
def average_correct (N : ℕ) := 22

theorem number_of_elements (N : ℕ) (incorrect : ℕ) (correct : ℕ) :
  average_incorrect N = 21 ∧ incorrect = 26 ∧ correct = 36 ∧ average_correct N = 22 →
  N = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_elements_l1215_121548


namespace NUMINAMATH_GPT_displacement_during_interval_l1215_121504

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem displacement_during_interval :
  (∫ t in (0 : ℝ)..3, velocity t) = 36 :=
by
  sorry

end NUMINAMATH_GPT_displacement_during_interval_l1215_121504


namespace NUMINAMATH_GPT_only_solution_l1215_121531

theorem only_solution (x : ℝ) : (3 / (x - 3) = 5 / (x - 5)) ↔ (x = 0) := 
sorry

end NUMINAMATH_GPT_only_solution_l1215_121531


namespace NUMINAMATH_GPT_maximum_value_of_f_l1215_121511

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem maximum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = Real.exp 1 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_f_l1215_121511


namespace NUMINAMATH_GPT_no_positive_integer_solution_l1215_121573

theorem no_positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ¬ (x^2 * y^4 - x^4 * y^2 + 4 * x^2 * y^2 * z^2 + x^2 * z^4 - y^2 * z^4 = 0) :=
sorry

end NUMINAMATH_GPT_no_positive_integer_solution_l1215_121573


namespace NUMINAMATH_GPT_overall_percentage_change_is_113_point_4_l1215_121553

-- Define the conditions
def total_customers_survey_1 := 100
def male_percentage_survey_1 := 60
def respondents_survey_1 := 10
def male_respondents_survey_1 := 5

def total_customers_survey_2 := 80
def male_percentage_survey_2 := 70
def respondents_survey_2 := 16
def male_respondents_survey_2 := 12

def total_customers_survey_3 := 70
def male_percentage_survey_3 := 40
def respondents_survey_3 := 21
def male_respondents_survey_3 := 13

def total_customers_survey_4 := 90
def male_percentage_survey_4 := 50
def respondents_survey_4 := 27
def male_respondents_survey_4 := 8

-- Define the calculated response rates
def original_male_response_rate := (male_respondents_survey_1.toFloat / (total_customers_survey_1 * male_percentage_survey_1 / 100).toFloat) * 100
def final_male_response_rate := (male_respondents_survey_4.toFloat / (total_customers_survey_4 * male_percentage_survey_4 / 100).toFloat) * 100

-- Calculate the percentage change in response rate
def percentage_change := ((final_male_response_rate - original_male_response_rate) / original_male_response_rate) * 100

-- The target theorem 
theorem overall_percentage_change_is_113_point_4 : percentage_change = 113.4 := sorry

end NUMINAMATH_GPT_overall_percentage_change_is_113_point_4_l1215_121553


namespace NUMINAMATH_GPT_fraction_eq_l1215_121532

def f(x : ℤ) : ℤ := 3 * x + 2
def g(x : ℤ) : ℤ := 2 * x - 3

theorem fraction_eq : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by 
  sorry

end NUMINAMATH_GPT_fraction_eq_l1215_121532


namespace NUMINAMATH_GPT_xiaoming_statement_incorrect_l1215_121596

theorem xiaoming_statement_incorrect (s : ℕ) : 
    let x_h := 3
    let x_m := 6
    let steps_xh := (x_h - 1) * s
    let steps_xm := (x_m - 1) * s
    (steps_xm ≠ 2 * steps_xh) :=
by
  let x_h := 3
  let x_m := 6
  let steps_xh := (x_h - 1) * s
  let steps_xm := (x_m - 1) * s
  sorry

end NUMINAMATH_GPT_xiaoming_statement_incorrect_l1215_121596


namespace NUMINAMATH_GPT_find_min_value_l1215_121540

-- Define a structure to represent vectors in 2D space
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

-- Define the dot product of two vectors
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the condition for perpendicular vectors (dot product is zero)
def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

-- Define the problem: given vectors a = (m, 1) and b = (1, n - 2)
-- with conditions m > 0, n > 0, and a ⊥ b, then prove the minimum value of 1/m + 2/n
theorem find_min_value (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0)
  (h₂ : perpendicular ⟨m, 1⟩ ⟨1, n - 2⟩) :
  (1 / m + 2 / n) = (3 + 2 * Real.sqrt 2) / 2 :=
  sorry

end NUMINAMATH_GPT_find_min_value_l1215_121540


namespace NUMINAMATH_GPT_find_sum_invested_l1215_121519

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end NUMINAMATH_GPT_find_sum_invested_l1215_121519


namespace NUMINAMATH_GPT_player2_wins_l1215_121536

-- Definitions for the initial conditions and game rules
def initial_piles := [10, 15, 20]
def split_rule (piles : List ℕ) (move : ℕ → ℕ × ℕ) : List ℕ :=
  let (pile1, pile2) := move (piles.head!)
  (pile1 :: pile2 :: piles.tail!)

-- Winning condition proof
theorem player2_wins :
  ∀ piles : List ℕ, piles = [10, 15, 20] →
  (∀ move_count : ℕ, move_count = 42 →
  (move_count > 0 ∧ ¬ ∃ split : ℕ → ℕ × ℕ, move_count % 2 = 1)) :=
by
  intro piles hpiles
  intro move_count hmove_count
  sorry

end NUMINAMATH_GPT_player2_wins_l1215_121536


namespace NUMINAMATH_GPT_div_product_four_consecutive_integers_l1215_121558

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end NUMINAMATH_GPT_div_product_four_consecutive_integers_l1215_121558


namespace NUMINAMATH_GPT_num_ordered_triples_l1215_121597

theorem num_ordered_triples 
  (a b c : ℕ)
  (h_cond1 : 1 ≤ a ∧ a ≤ b ∧ b ≤ c)
  (h_cond2 : a * b * c = 4 * (a * b + b * c + c * a)) : 
  ∃ (n : ℕ), n = 5 :=
sorry

end NUMINAMATH_GPT_num_ordered_triples_l1215_121597


namespace NUMINAMATH_GPT_gallons_per_hour_l1215_121577

-- Define conditions
def total_runoff : ℕ := 240000
def days : ℕ := 10
def hours_per_day : ℕ := 24

-- Define the goal: proving the sewers handle 1000 gallons of run-off per hour
theorem gallons_per_hour : (total_runoff / (days * hours_per_day)) = 1000 :=
by
  -- Proof can be inserted here
  sorry

end NUMINAMATH_GPT_gallons_per_hour_l1215_121577


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1215_121508

variable (x y : ℝ)

theorem simplify_expr1 : 
  3 * x^2 - 2 * x * y + y^2 - 3 * x^2 + 3 * x * y = x * y + y^2 :=
by
  sorry

theorem simplify_expr2 : 
  (7 * x^2 - 3 * x * y) - 6 * (x^2 - 1/3 * x * y) = x^2 - x * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1215_121508


namespace NUMINAMATH_GPT_boat_speed_still_water_l1215_121588

theorem boat_speed_still_water (v c : ℝ) (h1 : v + c = 13) (h2 : v - c = 4) : v = 8.5 :=
by sorry

end NUMINAMATH_GPT_boat_speed_still_water_l1215_121588


namespace NUMINAMATH_GPT_ratio_of_speeds_l1215_121598

theorem ratio_of_speeds (v_A v_B : ℝ) (t : ℝ) (hA : v_A = 120 / t) (hB : v_B = 60 / t) : v_A / v_B = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_speeds_l1215_121598


namespace NUMINAMATH_GPT_regina_total_cost_l1215_121520

-- Definitions
def daily_cost : ℝ := 30
def mileage_cost : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 450
def fixed_fee : ℝ := 15

-- Proposition for total cost
noncomputable def total_cost : ℝ := daily_cost * days_rented + mileage_cost * miles_driven + fixed_fee

-- Theorem statement
theorem regina_total_cost : total_cost = 217.5 := by
  sorry

end NUMINAMATH_GPT_regina_total_cost_l1215_121520


namespace NUMINAMATH_GPT_test_group_type_A_probability_atleast_one_type_A_group_probability_l1215_121594

noncomputable def probability_type_A_group : ℝ :=
  let pA := 2 / 3
  let pB := 1 / 2
  let P_A1 := 2 * (1 - pA) * pA
  let P_A2 := pA * pA
  let P_B0 := (1 - pB) * (1 - pB)
  let P_B1 := 2 * (1 - pB) * pB
  P_B0 * P_A1 + P_B0 * P_A2 + P_B1 * P_A2

theorem test_group_type_A_probability :
  probability_type_A_group = 4 / 9 :=
by
  sorry

noncomputable def at_least_one_type_A_in_3_groups : ℝ :=
  let P_type_A_group := 4 / 9
  1 - (1 - P_type_A_group) ^ 3

theorem atleast_one_type_A_group_probability :
  at_least_one_type_A_in_3_groups = 604 / 729 :=
by
  sorry

end NUMINAMATH_GPT_test_group_type_A_probability_atleast_one_type_A_group_probability_l1215_121594


namespace NUMINAMATH_GPT_find_a_and_b_l1215_121589

theorem find_a_and_b (a b : ℝ) :
  (∀ x, y = a + b / x) →
  (y = 3 → x = 2) →
  (y = -1 → x = -4) →
  a + b = 4 :=
by sorry

end NUMINAMATH_GPT_find_a_and_b_l1215_121589


namespace NUMINAMATH_GPT_quadratic_has_one_real_root_l1215_121545

theorem quadratic_has_one_real_root (k : ℝ) : 
  (∃ (x : ℝ), -2 * x^2 + 8 * x + k = 0 ∧ ∀ y, -2 * y^2 + 8 * y + k = 0 → y = x) ↔ k = -8 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_real_root_l1215_121545


namespace NUMINAMATH_GPT_quadrilateral_area_proof_l1215_121555

-- Definitions of points
def A : (ℝ × ℝ) := (1, 3)
def B : (ℝ × ℝ) := (1, 1)
def C : (ℝ × ℝ) := (3, 1)
def D : (ℝ × ℝ) := (2010, 2011)

-- Function to calculate the area of the quadrilateral
def area_of_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ := 
  let area_triangle (P Q R : (ℝ × ℝ)) : ℝ := 
    0.5 * (P.1 * Q.2 + Q.1 * R.2 + R.1 * P.2 - P.2 * Q.1 - Q.2 * R.1 - R.2 * P.1)
  area_triangle A B C + area_triangle A C D

-- Lean statement to prove the desired area
theorem quadrilateral_area_proof : area_of_quadrilateral A B C D = 7 := 
  sorry

end NUMINAMATH_GPT_quadrilateral_area_proof_l1215_121555


namespace NUMINAMATH_GPT_triangle_area_l1215_121575

/-- 
In a triangle ABC, given that ∠B=30°, AB=2√3, and AC=2, 
prove that the area of the triangle ABC is either √3 or 2√3.
 -/
theorem triangle_area (B : Real) (AB AC : Real) 
  (h_B : B = 30) (h_AB : AB = 2 * Real.sqrt 3) (h_AC : AC = 2) :
  ∃ S : Real, (S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3) := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_l1215_121575


namespace NUMINAMATH_GPT_solve_fraction_l1215_121562

open Real

theorem solve_fraction (x : ℝ) (hx : 1 - 4 / x + 4 / x^2 = 0) : 2 / x = 1 :=
by
  -- We'll include the necessary steps of the proof here, but for now we leave it as sorry.
  sorry

end NUMINAMATH_GPT_solve_fraction_l1215_121562


namespace NUMINAMATH_GPT_max_radius_of_circle_l1215_121524

theorem max_radius_of_circle (c : ℝ × ℝ → Prop) (h1 : c (16, 0)) (h2 : c (-16, 0)) :
  ∃ r : ℝ, r = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_radius_of_circle_l1215_121524


namespace NUMINAMATH_GPT_mathematicians_correctness_l1215_121583

theorem mathematicians_correctness:
  ∀ (w1₁ w1₂ t1₁ t1₂ w2₁ w2₂ t2₁ t2₂: ℕ),
  (w1₁ = 4) ∧ (t1₁ = 7) ∧ (w1₂ = 3) ∧ (t1₂ = 5) →
  (w2₁ = 8) ∧ (t2₁ = 14) ∧ (w2₂ = 3) ∧ (t2₂ = 5) →
  (4/7 : ℚ) < 3/5 →
  (w1₁ + w1₂) / (t1₁ + t1₂) = 7 / 12 ∨ (w2₁ + w2₂) / (t2₁ + t2₂) = 11 / 19 →
  ¬ (w1₁ + w1₂) / (t1₁ + t1₂) < 19 / 35 ∧ (w2₁ + w2₂) / (t2₁ + t2₂) < 19 / 35 :=
by
  sorry

end NUMINAMATH_GPT_mathematicians_correctness_l1215_121583


namespace NUMINAMATH_GPT_min_students_wearing_both_glasses_and_watches_l1215_121535

theorem min_students_wearing_both_glasses_and_watches
  (n : ℕ)
  (H_glasses : n * 3 / 5 = 18)
  (H_watches : n * 5 / 6 = 25)
  (H_neither : n * 1 / 10 = 3):
  ∃ (x : ℕ), x = 16 := 
by
  sorry

end NUMINAMATH_GPT_min_students_wearing_both_glasses_and_watches_l1215_121535


namespace NUMINAMATH_GPT_tan_alpha_expression_value_l1215_121556

-- (I) Prove that tan(α) = 4/3 under the given conditions
theorem tan_alpha (O A B C P : ℝ × ℝ) (α : ℝ)
  (hO : O = (0, 0))
  (hA : A = (Real.sin α, 1))
  (hB : B = (Real.cos α, 0))
  (hC : C = (-Real.sin α, 2))
  (hP : P = (2 * Real.cos α - Real.sin α, 1))
  (h_collinear : ∃ t : ℝ, C = t • (P.1, P.2)) :
  Real.tan α = 4 / 3 := sorry

-- (II) Prove the given expression under the condition tan(α) = 4/3
theorem expression_value (α : ℝ)
  (h_tan : Real.tan α = 4 / 3) :
  (Real.sin (2 * α) + Real.sin α) / (2 * Real.cos (2 * α) + 2 * Real.sin α * Real.sin α + Real.cos α) + Real.sin (2 * α) = 
  172 / 75 := sorry

end NUMINAMATH_GPT_tan_alpha_expression_value_l1215_121556


namespace NUMINAMATH_GPT_binomial_square_value_l1215_121526

theorem binomial_square_value (c : ℝ) : (∃ d : ℝ, 16 * x^2 + 40 * x + c = (4 * x + d) ^ 2) → c = 25 :=
by
  sorry

end NUMINAMATH_GPT_binomial_square_value_l1215_121526


namespace NUMINAMATH_GPT_first_machine_rate_l1215_121518

theorem first_machine_rate (x : ℕ) (h1 : 30 * x + 30 * 65 = 3000) : x = 35 := sorry

end NUMINAMATH_GPT_first_machine_rate_l1215_121518


namespace NUMINAMATH_GPT_customers_per_table_l1215_121592

theorem customers_per_table (total_tables : ℝ) (left_tables : ℝ) (total_customers : ℕ)
  (h1 : total_tables = 44.0)
  (h2 : left_tables = 12.0)
  (h3 : total_customers = 256) :
  total_customers / (total_tables - left_tables) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_customers_per_table_l1215_121592


namespace NUMINAMATH_GPT_height_relationship_l1215_121527

theorem height_relationship (B V G : ℝ) (h1 : B = 2 * V) (h2 : V = (2 / 3) * G) : B = (4 / 3) * G :=
sorry

end NUMINAMATH_GPT_height_relationship_l1215_121527


namespace NUMINAMATH_GPT_multiple_of_10_and_12_within_100_l1215_121586

theorem multiple_of_10_and_12_within_100 :
  ∀ (n : ℕ), n ≤ 100 → (∃ k₁ k₂ : ℕ, n = 10 * k₁ ∧ n = 12 * k₂) ↔ n = 60 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_10_and_12_within_100_l1215_121586


namespace NUMINAMATH_GPT_horner_polynomial_rewrite_polynomial_value_at_5_l1215_121578

def polynomial (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 6 * x^3 - 2 * x^2 - 5 * x - 2

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_polynomial_rewrite :
  polynomial = horner_polynomial := 
sorry

theorem polynomial_value_at_5 :
  polynomial 5 = 7548 := 
sorry

end NUMINAMATH_GPT_horner_polynomial_rewrite_polynomial_value_at_5_l1215_121578


namespace NUMINAMATH_GPT_product_of_two_numbers_l1215_121557

theorem product_of_two_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 560) (h_hcf : Nat.gcd a b = 75) :
  a * b = 42000 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1215_121557


namespace NUMINAMATH_GPT_angle_405_eq_45_l1215_121585

def same_terminal_side (angle1 angle2 : ℝ) : Prop :=
  ∃ k : ℤ, angle1 = angle2 + k * 360

theorem angle_405_eq_45 (k : ℤ) : same_terminal_side 405 45 := 
sorry

end NUMINAMATH_GPT_angle_405_eq_45_l1215_121585


namespace NUMINAMATH_GPT_square_D_perimeter_l1215_121572

theorem square_D_perimeter 
(C_perimeter: Real) 
(D_area_ratio : Real) 
(hC : C_perimeter = 32) 
(hD : D_area_ratio = 1/3) : 
    ∃ D_perimeter, D_perimeter = (32 * Real.sqrt 3) / 3 := 
by 
    sorry

end NUMINAMATH_GPT_square_D_perimeter_l1215_121572


namespace NUMINAMATH_GPT_common_roots_of_cubic_polynomials_l1215_121541

/-- The polynomials \( x^3 + 6x^2 + 11x + 6 \) and \( x^3 + 7x^2 + 14x + 8 \) have two distinct roots in common. -/
theorem common_roots_of_cubic_polynomials :
  ∃ r s : ℝ, r ≠ s ∧ (r^3 + 6 * r^2 + 11 * r + 6 = 0) ∧ (s^3 + 6 * s^2 + 11 * s + 6 = 0)
  ∧ (r^3 + 7 * r^2 + 14 * r + 8 = 0) ∧ (s^3 + 7 * s^2 + 14 * s + 8 = 0) :=
sorry

end NUMINAMATH_GPT_common_roots_of_cubic_polynomials_l1215_121541


namespace NUMINAMATH_GPT_achieve_target_ratio_l1215_121549

-- Initial volume and ratio
def initial_volume : ℕ := 20
def initial_milk_ratio : ℕ := 3
def initial_water_ratio : ℕ := 2

-- Mixture removal and addition
def removal_volume : ℕ := 10
def added_milk : ℕ := 10

-- Target ratio of milk to water
def target_milk_ratio : ℕ := 9
def target_water_ratio : ℕ := 1

-- Number of operations required
def operations_needed: ℕ := 2

-- Statement of proof problem
theorem achieve_target_ratio :
  (initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio) + added_milk * operations_needed) / 
  (initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio) - removal_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) = target_milk_ratio :=
sorry

end NUMINAMATH_GPT_achieve_target_ratio_l1215_121549


namespace NUMINAMATH_GPT_bus_distance_time_relation_l1215_121576

theorem bus_distance_time_relation (t : ℝ) :
    (0 ≤ t ∧ t ≤ 1 → s = 60 * t) ∧
    (1 < t ∧ t ≤ 1.5 → s = 60) ∧
    (1.5 < t ∧ t ≤ 2.5 → s = 80 * (t - 1.5) + 60) :=
sorry

end NUMINAMATH_GPT_bus_distance_time_relation_l1215_121576


namespace NUMINAMATH_GPT_smallest_sum_l1215_121543

theorem smallest_sum (x y : ℕ) (h : (2010 / 2011 : ℚ) < x / y ∧ x / y < (2011 / 2012 : ℚ)) : x + y = 8044 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l1215_121543


namespace NUMINAMATH_GPT_velocity_zero_times_l1215_121544

noncomputable def s (t : ℝ) : ℝ := (1 / 4) * t^4 - (5 / 3) * t^3 + 2 * t^2

theorem velocity_zero_times :
  {t : ℝ | deriv s t = 0} = {0, 1, 4} :=
by 
  sorry

end NUMINAMATH_GPT_velocity_zero_times_l1215_121544


namespace NUMINAMATH_GPT_fraction_mango_sold_l1215_121515

theorem fraction_mango_sold :
  ∀ (choco_total mango_total choco_sold unsold: ℕ) (x : ℚ),
    choco_total = 50 →
    mango_total = 54 →
    choco_sold = (3 * 50) / 5 →
    unsold = 38 →
    (choco_total + mango_total) - (choco_sold + x * mango_total) = unsold →
    x = 4 / 27 :=
by
  intros choco_total mango_total choco_sold unsold x
  sorry

end NUMINAMATH_GPT_fraction_mango_sold_l1215_121515


namespace NUMINAMATH_GPT_complement_union_complement_intersection_l1215_121546

open Set

noncomputable def universal_set : Set ℝ := univ

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 6}

theorem complement_union :
  compl (A ∪ B) = {x : ℝ | x ≤ 2 ∨ 7 ≤ x} := by
  sorry

theorem complement_intersection :
  (compl A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_complement_union_complement_intersection_l1215_121546


namespace NUMINAMATH_GPT_plates_arrangement_l1215_121563

theorem plates_arrangement : 
  let blue := 6
  let red := 3
  let green := 2
  let yellow := 1
  let total_ways_without_rest := Nat.factorial (blue + red + green + yellow - 1) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial yellow)
  let green_adj_ways := Nat.factorial (blue + red + green + yellow - 2) / (Nat.factorial blue * Nat.factorial red * Nat.factorial 1 * Nat.factorial yellow)
  total_ways_without_rest - green_adj_ways = 22680 
:= sorry

end NUMINAMATH_GPT_plates_arrangement_l1215_121563


namespace NUMINAMATH_GPT_part_a_part_b_l1215_121584

namespace ProofProblem

def number_set := {n : ℕ | ∃ k : ℕ, n = (10^k - 1)}

noncomputable def special_structure (n : ℕ) : Prop :=
  ∃ m : ℕ, n = 2 * m + 1 ∨ n = 2 * m + 2

theorem part_a :
  ∃ (a b c : ℕ) (ha : a ∈ number_set) (hb : b ∈ number_set) (hc : c ∈ number_set),
    special_structure (a + b + c) :=
by
  sorry

theorem part_b (cards : List ℕ) (h : ∀ x ∈ cards, x ∈ number_set)
    (hs : special_structure (cards.sum)) :
  ∃ (d : ℕ), d ≠ 2 ∧ (d = 0 ∨ d = 1) :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_part_a_part_b_l1215_121584


namespace NUMINAMATH_GPT_empty_subset_of_A_l1215_121571

def A : Set ℤ := {x | 0 < x ∧ x < 3}

theorem empty_subset_of_A : ∅ ⊆ A :=
by
  sorry

end NUMINAMATH_GPT_empty_subset_of_A_l1215_121571


namespace NUMINAMATH_GPT_john_change_proof_l1215_121559

def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5

def cost_of_candy_bar : ℕ := 131
def quarters_paid : ℕ := 4
def dimes_paid : ℕ := 3
def nickels_paid : ℕ := 1

def total_payment : ℕ := (quarters_paid * quarter_value) + (dimes_paid * dime_value) + (nickels_paid * nickel_value)
def change_received : ℕ := total_payment - cost_of_candy_bar

theorem john_change_proof : change_received = 4 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_john_change_proof_l1215_121559


namespace NUMINAMATH_GPT_problem_l1215_121510

def f (x a b : ℝ) : ℝ := a * x ^ 3 - b * x + 1

theorem problem (a b : ℝ) (h : f 2 a b = -1) : f (-2) a b = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l1215_121510


namespace NUMINAMATH_GPT_set_A_range_l1215_121525

def A := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ (-1 ≤ x ∧ x ≤ 2)}

theorem set_A_range :
  A = {y : ℝ | -4 ≤ y ∧ y ≤ 0} :=
sorry

end NUMINAMATH_GPT_set_A_range_l1215_121525


namespace NUMINAMATH_GPT_complementary_angles_difference_l1215_121500

theorem complementary_angles_difference :
  ∃ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 90 ∧ 5 * θ₁ = 3 * θ₂ ∧ abs (θ₁ - θ₂) = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_difference_l1215_121500


namespace NUMINAMATH_GPT_multiples_of_3_or_4_probability_l1215_121599

theorem multiples_of_3_or_4_probability :
  let total_cards := 36
  let multiples_of_3 := 12
  let multiples_of_4 := 9
  let multiples_of_both := 3
  let favorable_outcomes := multiples_of_3 + multiples_of_4 - multiples_of_both
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_3_or_4_probability_l1215_121599


namespace NUMINAMATH_GPT_bridge_length_is_correct_l1215_121587

noncomputable def length_of_bridge (train_length : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let distance_covered := speed_mps * time
  distance_covered - train_length

theorem bridge_length_is_correct :
  length_of_bridge 100 16.665333439991468 54 = 149.97999909987152 :=
by sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l1215_121587


namespace NUMINAMATH_GPT_smartphone_price_l1215_121595

theorem smartphone_price (S : ℝ) (pc_price : ℝ) (tablet_price : ℝ) 
  (total_cost : ℝ) (h1 : pc_price = S + 500) 
  (h2 : tablet_price = 2 * S + 500) 
  (h3 : S + pc_price + tablet_price = 2200) : 
  S = 300 :=
by
  sorry

end NUMINAMATH_GPT_smartphone_price_l1215_121595


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1215_121564

noncomputable def distance_between_points (v_A v_B : ℝ) (t_meet t_A_to_B_after_meet : ℝ) : ℝ :=
  let t_total_A := t_meet + t_A_to_B_after_meet
  let t_total_B := t_meet + (t_meet - t_A_to_B_after_meet)
  let D := v_A * t_total_A + v_B * t_total_B
  D

-- Given conditions
def t_meet : ℝ := 4
def t_A_to_B_after_meet : ℝ := 3
def speed_difference : ℝ := 20

-- Function to calculate speeds based on given conditions
noncomputable def calculate_speeds (v_B : ℝ) : ℝ × ℝ :=
  let v_A := v_B + speed_difference
  (v_A, v_B)

-- Statement of the problem in Lean 4
theorem distance_between_A_and_B : ∃ (v_B v_A : ℝ), 
  v_A = v_B + speed_difference ∧
  distance_between_points v_A v_B t_meet t_A_to_B_after_meet = 240 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1215_121564


namespace NUMINAMATH_GPT_solve_fractional_equation_l1215_121514

theorem solve_fractional_equation (x : ℝ) (h₀ : 2 = 3 * (x + 1) / (4 - x)) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1215_121514


namespace NUMINAMATH_GPT_b_alone_completion_days_l1215_121593

theorem b_alone_completion_days (Rab : ℝ) (w_12_days : (1 / (Rab + 4 * Rab)) = 12⁻¹) : 
    (1 / Rab = 60) :=
sorry

end NUMINAMATH_GPT_b_alone_completion_days_l1215_121593


namespace NUMINAMATH_GPT_solve_cubic_eq_with_geo_prog_coeff_l1215_121503

variables {a q x : ℝ}

theorem solve_cubic_eq_with_geo_prog_coeff (h_a_nonzero : a ≠ 0) 
    (h_b : b = a * q) (h_c : c = a * q^2) (h_d : d = a * q^3) :
    (a * x^3 + b * x^2 + c * x + d = 0) → (x = -q) :=
by
  intros h_cubic_eq
  have h_b' : b = a * q := h_b
  have h_c' : c = a * q^2 := h_c
  have h_d' : d = a * q^3 := h_d
  sorry

end NUMINAMATH_GPT_solve_cubic_eq_with_geo_prog_coeff_l1215_121503


namespace NUMINAMATH_GPT_find_z_solutions_l1215_121517

open Real

noncomputable def is_solution (z : ℝ) : Prop :=
  sin z + sin (2 * z) + sin (3 * z) = cos z + cos (2 * z) + cos (3 * z)

theorem find_z_solutions (z : ℝ) : 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k - 1)) ∨ 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k + 1)) ∨ 
  (∃ k : ℤ, z = π / 8 * (4 * k + 1)) ↔
  is_solution z :=
by
  sorry

end NUMINAMATH_GPT_find_z_solutions_l1215_121517


namespace NUMINAMATH_GPT_trajectory_of_P_eqn_l1215_121523

noncomputable def point_A : ℝ × ℝ := (1, 0)

def curve_C (x : ℝ) : ℝ := x^2 - 2

def symmetric_point (Qx Qy Px Py : ℝ) : Prop :=
  Qx = 2 - Px ∧ Qy = -Py

theorem trajectory_of_P_eqn (Qx Qy Px Py : ℝ) (hQ_on_C : Qy = curve_C Qx)
  (h_symm : symmetric_point Qx Qy Px Py) :
  Py = -Px^2 + 4 * Px - 2 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_P_eqn_l1215_121523


namespace NUMINAMATH_GPT_john_quiz_goal_l1215_121565

theorem john_quiz_goal
  (total_quizzes : ℕ)
  (goal_percentage : ℕ)
  (quizzes_completed : ℕ)
  (quizzes_remaining : ℕ)
  (quizzes_with_A_completed : ℕ)
  (total_quizzes_with_A_needed : ℕ)
  (additional_A_needed : ℕ)
  (quizzes_below_A_allowed : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 75)
  (h3 : quizzes_completed = 40)
  (h4 : quizzes_remaining = total_quizzes - quizzes_completed)
  (h5 : quizzes_with_A_completed = 27)
  (h6 : total_quizzes_with_A_needed = total_quizzes * goal_percentage / 100)
  (h7 : additional_A_needed = total_quizzes_with_A_needed - quizzes_with_A_completed)
  (h8 : quizzes_below_A_allowed = quizzes_remaining - additional_A_needed)
  (h_goal : quizzes_below_A_allowed ≤ 2) : quizzes_below_A_allowed = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_quiz_goal_l1215_121565


namespace NUMINAMATH_GPT_mr_klinker_twice_as_old_l1215_121580

theorem mr_klinker_twice_as_old (x : ℕ) (current_age_klinker : ℕ) (current_age_daughter : ℕ)
  (h1 : current_age_klinker = 35) (h2 : current_age_daughter = 10) 
  (h3 : current_age_klinker + x = 2 * (current_age_daughter + x)) : 
  x = 15 :=
by 
  -- We include sorry to indicate where the proof should be
  sorry

end NUMINAMATH_GPT_mr_klinker_twice_as_old_l1215_121580


namespace NUMINAMATH_GPT_margaret_speed_on_time_l1215_121566
-- Import the necessary libraries from Mathlib

-- Define the problem conditions and state the theorem
theorem margaret_speed_on_time :
  ∃ r : ℝ, (∀ d t : ℝ,
    d = 50 * (t - 1/12) ∧
    d = 30 * (t + 1/12) →
    r = d / t) ∧
  r = 37.5 := 
sorry

end NUMINAMATH_GPT_margaret_speed_on_time_l1215_121566


namespace NUMINAMATH_GPT_maximum_value_of_function_l1215_121552

theorem maximum_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ M, (∀ y, y = x * (1 - 2 * x) → y ≤ M) ∧ M = 1/8 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_function_l1215_121552


namespace NUMINAMATH_GPT_solve_fraction_eq_l1215_121554

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x^2 - x - 2) / (x + 2) = x + 3 ↔ x = -4 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l1215_121554


namespace NUMINAMATH_GPT_total_selling_price_l1215_121530

def original_price : ℝ := 120
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.15

def sale_price (original_price discount_percent : ℝ) : ℝ :=
  original_price * (1 - discount_percent)

def final_price (sale_price tax_percent : ℝ) : ℝ :=
  sale_price * (1 + tax_percent)

theorem total_selling_price :
  final_price (sale_price original_price discount_percent) tax_percent = 96.6 :=
sorry

end NUMINAMATH_GPT_total_selling_price_l1215_121530


namespace NUMINAMATH_GPT_find_a_equidistant_l1215_121505

theorem find_a_equidistant :
  ∀ a : ℝ, (abs (a - 2) = abs (6 - 2 * a)) →
    (a = 8 / 3 ∨ a = 4) :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_find_a_equidistant_l1215_121505


namespace NUMINAMATH_GPT_Neil_candy_collected_l1215_121509

variable (M H N : ℕ)

-- Conditions
def Maggie_collected := M = 50
def Harper_collected := H = M + (30 * M) / 100
def Neil_collected := N = H + (40 * H) / 100

-- Theorem statement 
theorem Neil_candy_collected
  (hM : Maggie_collected M)
  (hH : Harper_collected M H)
  (hN : Neil_collected H N) :
  N = 91 := by
  sorry

end NUMINAMATH_GPT_Neil_candy_collected_l1215_121509


namespace NUMINAMATH_GPT_minFuseLength_l1215_121561

namespace EarthquakeRelief

def fuseLengthRequired (distanceToSafety : ℕ) (speedOperator : ℕ) (burningSpeed : ℕ) (lengthFuse : ℕ) : Prop :=
  (lengthFuse : ℝ) / (burningSpeed : ℝ) > (distanceToSafety : ℝ) / (speedOperator : ℝ)

theorem minFuseLength 
  (distanceToSafety : ℕ := 400) 
  (speedOperator : ℕ := 5) 
  (burningSpeed : ℕ := 12) : 
  ∀ lengthFuse: ℕ, 
  fuseLengthRequired distanceToSafety speedOperator burningSpeed lengthFuse → lengthFuse > 96 := 
by
  sorry

end EarthquakeRelief

end NUMINAMATH_GPT_minFuseLength_l1215_121561


namespace NUMINAMATH_GPT_incorrect_table_value_l1215_121538

theorem incorrect_table_value (a b c : ℕ) (values : List ℕ) (correct : values = [2051, 2197, 2401, 2601, 2809, 3025, 3249, 3481]) : 
  (2401 ∉ [2051, 2197, 2399, 2601, 2809, 3025, 3249, 3481]) :=
sorry

end NUMINAMATH_GPT_incorrect_table_value_l1215_121538


namespace NUMINAMATH_GPT_find_B_l1215_121507

noncomputable def A : ℝ := 1 / 49
noncomputable def C : ℝ := -(1 / 7)

theorem find_B :
  (∀ x : ℝ, 1 / (x^3 + 2 * x^2 - 25 * x - 50) 
            = (A / (x - 2)) + (B / (x + 5)) + (C / ((x + 5)^2))) 
    → B = - (11 / 490) :=
sorry

end NUMINAMATH_GPT_find_B_l1215_121507


namespace NUMINAMATH_GPT_johns_earnings_without_bonus_l1215_121533
-- Import the Mathlib library to access all necessary functions and definitions

-- Define the conditions of the problem
def hours_without_bonus : ℕ := 8
def bonus_amount : ℕ := 20
def extra_hours_for_bonus : ℕ := 2
def hours_with_bonus : ℕ := hours_without_bonus + extra_hours_for_bonus
def hourly_wage_with_bonus : ℕ := 10

-- Define the total earnings with the performance bonus
def total_earnings_with_bonus : ℕ := hours_with_bonus * hourly_wage_with_bonus

-- Statement to prove the earnings without the bonus
theorem johns_earnings_without_bonus :
  total_earnings_with_bonus - bonus_amount = 80 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_johns_earnings_without_bonus_l1215_121533


namespace NUMINAMATH_GPT_find_principal_sum_l1215_121569

theorem find_principal_sum (R P : ℝ) 
  (h1 : (3 * P * (R + 1) / 100 - 3 * P * R / 100) = 72) : 
  P = 2400 := 
by 
  sorry

end NUMINAMATH_GPT_find_principal_sum_l1215_121569


namespace NUMINAMATH_GPT_g_at_9_l1215_121516

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end NUMINAMATH_GPT_g_at_9_l1215_121516


namespace NUMINAMATH_GPT_right_triangle_inequality_l1215_121539

-- Definition of a right-angled triangle with given legs a, b, hypotenuse c, and altitude h_c to the hypotenuse
variables {a b c h_c : ℝ}

-- Right-angled triangle condition definition with angle at C is right
def right_angled_triangle (a b c : ℝ) : Prop :=
  ∃ (a b c : ℝ), c^2 = a^2 + b^2

-- Definition of the altitude to the hypotenuse
def altitude_to_hypotenuse (a b c h_c : ℝ) : Prop :=
  h_c = (a * b) / c

-- Theorem statement to prove the inequality for any right-angled triangle
theorem right_triangle_inequality (a b c h_c : ℝ) (h1 : right_angled_triangle a b c) (h2 : altitude_to_hypotenuse a b c h_c) : 
  a + b < c + h_c :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_inequality_l1215_121539


namespace NUMINAMATH_GPT_weight_of_3_moles_of_BaF2_is_correct_l1215_121547

-- Definitions for the conditions
def atomic_weight_Ba : ℝ := 137.33 -- g/mol
def atomic_weight_F : ℝ := 19.00 -- g/mol

-- Definition of the molecular weight of BaF2
def molecular_weight_BaF2 : ℝ := (1 * atomic_weight_Ba) + (2 * atomic_weight_F)

-- The statement to prove
theorem weight_of_3_moles_of_BaF2_is_correct : (3 * molecular_weight_BaF2) = 525.99 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_weight_of_3_moles_of_BaF2_is_correct_l1215_121547


namespace NUMINAMATH_GPT_winter_spending_l1215_121522

-- Define the total spending by the end of November
def total_spending_end_november : ℝ := 3.3

-- Define the total spending by the end of February
def total_spending_end_february : ℝ := 7.0

-- Formalize the problem: prove that the spending during December, January, and February is 3.7 million dollars
theorem winter_spending : total_spending_end_february - total_spending_end_november = 3.7 := by
  sorry

end NUMINAMATH_GPT_winter_spending_l1215_121522


namespace NUMINAMATH_GPT_inequality1_solution_inequality2_solution_l1215_121568

-- Definitions for the conditions
def cond1 (x : ℝ) : Prop := abs (1 - (2 * x - 1) / 3) ≤ 2
def cond2 (x : ℝ) : Prop := (2 - x) * (x + 3) < 2 - x

-- Lean 4 statement for the proof problem
theorem inequality1_solution (x : ℝ) : cond1 x → -1 ≤ x ∧ x ≤ 5 := by
  sorry

theorem inequality2_solution (x : ℝ) : cond2 x → x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_GPT_inequality1_solution_inequality2_solution_l1215_121568


namespace NUMINAMATH_GPT_problem_1_problem_2_l1215_121550

def p (x : ℝ) : Prop := -x^2 + 6*x + 16 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0 ∧ m > 0

theorem problem_1 (x : ℝ) : p x → -2 ≤ x ∧ x ≤ 8 :=
by
  -- Proof goes here
  sorry

theorem problem_2 (m : ℝ) : (∀ x, p x → q x m) ∧ (∃ x, ¬ p x ∧ q x m) → m ≥ 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1215_121550


namespace NUMINAMATH_GPT_solution_set_a_eq_half_l1215_121551

theorem solution_set_a_eq_half (a : ℝ) : (∀ x : ℝ, (ax / (x - 1) < 1 ↔ (x < 1 ∨ x > 2))) → a = 1 / 2 :=
by
sorry

end NUMINAMATH_GPT_solution_set_a_eq_half_l1215_121551


namespace NUMINAMATH_GPT_sum_g_eq_half_l1215_121502

noncomputable def g (n : ℕ) : ℝ := ∑' k, if h : k ≥ 3 then (1 / (k : ℝ) ^ n) else 0

theorem sum_g_eq_half : (∑' n, if h : n ≥ 3 then g n else 0) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sum_g_eq_half_l1215_121502
