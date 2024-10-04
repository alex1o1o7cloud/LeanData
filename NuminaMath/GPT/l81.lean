import Mathlib

namespace existence_not_implied_by_validity_l81_81113

-- Let us formalize the theorem and then show that its validity does not imply the existence of such a function.

-- Definitions for condition (A) and the theorem statement
axiom condition_A (f : ℝ → ℝ) : Prop
axiom theorem_239 : ∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x

-- Translation of the problem statement into Lean
theorem existence_not_implied_by_validity :
  (∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x) → 
  ¬ (∃ f, condition_A f) :=
sorry

end existence_not_implied_by_validity_l81_81113


namespace housewife_more_kgs_l81_81723

theorem housewife_more_kgs (P R money more_kgs : ℝ)
  (hR: R = 40)
  (hReduction: R = P - 0.25 * P)
  (hMoney: money = 800)
  (hMoreKgs: more_kgs = (money / R) - (money / P)) :
  more_kgs = 5 :=
  by
    sorry

end housewife_more_kgs_l81_81723


namespace abs_eq_5_iff_l81_81300

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l81_81300


namespace simplify_expr1_simplify_expr2_l81_81184

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l81_81184


namespace graph_passes_quadrants_l81_81906

theorem graph_passes_quadrants {x y : ℝ} (h : y = -x - 2) :
  -- Statement that the graph passes through the second, third, and fourth quadrants.
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x < 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y > 0 ∧ y = -x - 2)) :=
by
  sorry

end graph_passes_quadrants_l81_81906


namespace simplify_expression_l81_81672

theorem simplify_expression : ((3 * 2 + 4 + 6) / 3 - 2 / 3) = 14 / 3 := by
  sorry

end simplify_expression_l81_81672


namespace least_positive_number_divisible_by_five_smallest_primes_l81_81042

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l81_81042


namespace tangent_position_is_six_oclock_l81_81019

-- Define constants and initial conditions
def bigRadius : ℝ := 30
def smallRadius : ℝ := 15
def initialPosition := 12 -- 12 o'clock represented as initial tangent position
def initialArrowDirection := 0 -- upwards direction

-- Define that the small disk rolls counterclockwise around the clock face.
def rollsCCW := true

-- Define the destination position when the arrow next points upward.
def diskTangencyPosition (bR sR : ℝ) (initPos initDir : ℕ) (rolls : Bool) : ℕ :=
  if rolls then 6 else 12

theorem tangent_position_is_six_oclock :
  diskTangencyPosition bigRadius smallRadius initialPosition initialArrowDirection rollsCCW = 6 :=
sorry  -- the proof is omitted

end tangent_position_is_six_oclock_l81_81019


namespace laptop_repair_cost_l81_81711

theorem laptop_repair_cost
  (price_phone_repair : ℝ)
  (price_computer_repair : ℝ)
  (price_laptop_repair : ℝ)
  (condition1 : price_phone_repair = 11)
  (condition2 : price_computer_repair = 18)
  (condition3 : 5 * price_phone_repair + 2 * price_laptop_repair + 2 * price_computer_repair = 121) :
  price_laptop_repair = 15 :=
by
  sorry

end laptop_repair_cost_l81_81711


namespace multiply_103_97_l81_81260

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end multiply_103_97_l81_81260


namespace greatest_n_and_k_l81_81629

-- (condition): k is a positive integer
def isPositive (k : Nat) : Prop :=
  k > 0

-- (condition): k < n
def lessThan (k n : Nat) : Prop :=
  k < n

/-- Let m = 3^n and k be a positive integer such that k < n.
     Determine the greatest value of n for which 3^n divides 25!,
     and the greatest value of k such that 3^k divides (25! - 3^n). -/
theorem greatest_n_and_k :
  ∃ (n k : Nat), (3^n ∣ Nat.factorial 25) ∧ (isPositive k) ∧ (lessThan k n) ∧ (3^k ∣ (Nat.factorial 25 - 3^n)) ∧ n = 10 ∧ k = 9 := by
    sorry

end greatest_n_and_k_l81_81629


namespace calculate_sqrt_expression_l81_81848

theorem calculate_sqrt_expression :
  (2 * Real.sqrt 24 + 3 * Real.sqrt 6) / Real.sqrt 3 = 7 * Real.sqrt 2 :=
by
  sorry

end calculate_sqrt_expression_l81_81848


namespace linda_savings_l81_81062

theorem linda_savings :
  ∀ (S : ℝ), (5 / 6 * S + 500 = S) → S = 3000 :=
by
  intros S h
  sorry

end linda_savings_l81_81062


namespace apples_and_oranges_l81_81369

theorem apples_and_oranges :
  ∃ x y : ℝ, 2 * x + 3 * y = 6 ∧ 4 * x + 7 * y = 13 ∧ (16 * x + 23 * y = 47) :=
by
  sorry

end apples_and_oranges_l81_81369


namespace squares_ratio_l81_81843

noncomputable def inscribed_squares_ratio :=
  let x := 60 / 17
  let y := 780 / 169
  (x / y : ℚ)

theorem squares_ratio (x y : ℚ) (h₁ : x = 60 / 17) (h₂ : y = 780 / 169) :
  x / y = 169 / 220 := by
  rw [h₁, h₂]
  -- Here we would perform calculations to show equality, omitted for brevity.
  sorry

end squares_ratio_l81_81843


namespace digit_D_is_five_l81_81021

variable (A B C D : Nat)
variable (h1 : (B * A) % 10 = A % 10)
variable (h2 : ∀ (C : Nat), B - A = B % 10 ∧ C ≤ A)

theorem digit_D_is_five : D = 5 :=
by
  sorry

end digit_D_is_five_l81_81021


namespace solution_set_condition_l81_81428

theorem solution_set_condition {a : ℝ} : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ (x ≥ 3)) → a < 3 := 
by 
  intros h
  sorry

end solution_set_condition_l81_81428


namespace triangle_tan_inequality_l81_81457

theorem triangle_tan_inequality 
  {A B C : ℝ} 
  (h1 : π / 2 ≠ A) 
  (h2 : A ≥ B) 
  (h3 : B ≥ C) : 
  |Real.tan A| ≥ Real.tan B ∧ Real.tan B ≥ Real.tan C := 
  by
    sorry

end triangle_tan_inequality_l81_81457


namespace min_value_fraction_l81_81589

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end min_value_fraction_l81_81589


namespace stock_investment_net_increase_l81_81527

theorem stock_investment_net_increase :
  ∀ (initial_investment : ℝ)
    (increase_first_year : ℝ)
    (decrease_second_year : ℝ)
    (increase_third_year : ℝ),
  initial_investment = 100 → 
  increase_first_year = 0.60 → 
  decrease_second_year = 0.30 → 
  increase_third_year = 0.20 → 
  ((initial_investment * (1 + increase_first_year)) * (1 - decrease_second_year)) * (1 + increase_third_year) - initial_investment = 34.40 :=
by 
  intros initial_investment increase_first_year decrease_second_year increase_third_year 
  intros h_initial_investment h_increase_first_year h_decrease_second_year h_increase_third_year 
  rw [h_initial_investment, h_increase_first_year, h_decrease_second_year, h_increase_third_year]
  sorry

end stock_investment_net_increase_l81_81527


namespace calculate_expression_l81_81095

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l81_81095


namespace Keith_picked_6_apples_l81_81630

def m : ℝ := 7.0
def n : ℝ := 3.0
def t : ℝ := 10.0

noncomputable def r_m := m - n
noncomputable def k := t - r_m

-- Theorem Statement confirming Keith picked 6.0 apples
theorem Keith_picked_6_apples : k = 6.0 := by
  sorry

end Keith_picked_6_apples_l81_81630


namespace problem_solution_l81_81986

theorem problem_solution : (3127 - 2972) ^ 3 / 343 = 125 := by
  sorry

end problem_solution_l81_81986


namespace largest_positive_integer_divisible_l81_81030

theorem largest_positive_integer_divisible (n : ℕ) :
  (n + 20 ∣ n^3 - 100) ↔ n = 2080 :=
sorry

end largest_positive_integer_divisible_l81_81030


namespace matrix_determinant_6_l81_81879

theorem matrix_determinant_6 (x y z w : ℝ)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 2 * w) - z * (5 * x + 2 * y)) = 6 :=
by
  sorry

end matrix_determinant_6_l81_81879


namespace number_of_containers_needed_l81_81015

/-
  Define the parameters for the given problem
-/
def bags_suki : ℝ := 6.75
def weight_per_bag_suki : ℝ := 27

def bags_jimmy : ℝ := 4.25
def weight_per_bag_jimmy : ℝ := 23

def bags_natasha : ℝ := 3.80
def weight_per_bag_natasha : ℝ := 31

def container_capacity : ℝ := 17

/-
  The total weight bought by each person and the total combined weight
-/
def total_weight_suki : ℝ := bags_suki * weight_per_bag_suki
def total_weight_jimmy : ℝ := bags_jimmy * weight_per_bag_jimmy
def total_weight_natasha : ℝ := bags_natasha * weight_per_bag_natasha

def total_weight_combined : ℝ := total_weight_suki + total_weight_jimmy + total_weight_natasha

/-
  Prove that number of containers needed is 24
-/
theorem number_of_containers_needed : 
  Nat.ceil (total_weight_combined / container_capacity) = 24 := 
by
  sorry

end number_of_containers_needed_l81_81015


namespace product_of_four_consecutive_is_perfect_square_l81_81929

theorem product_of_four_consecutive_is_perfect_square (n : ℕ) :
  ∃ k : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
by
  sorry

end product_of_four_consecutive_is_perfect_square_l81_81929


namespace proof_of_problem_l81_81628

noncomputable def problem_statement : Prop :=
  ∃ (x y z m : ℝ), (x > 0 ∧ y > 0 ∧ z > 0 ∧ x^3 * y^2 * z = 1 ∧ m = x + 2*y + 3*z ∧ m^3 = 72)

theorem proof_of_problem : problem_statement :=
sorry

end proof_of_problem_l81_81628


namespace distance_between_hyperbola_vertices_l81_81732

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l81_81732


namespace social_survey_arrangements_l81_81812

theorem social_survey_arrangements :
  let total_ways := Nat.choose 9 3
  let all_male_ways := Nat.choose 5 3
  let all_female_ways := Nat.choose 4 3
  total_ways - all_male_ways - all_female_ways = 70 :=
by
  sorry

end social_survey_arrangements_l81_81812


namespace expression_evaluation_l81_81102

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l81_81102


namespace faye_rows_l81_81114

theorem faye_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h_total_pencils : total_pencils = 720)
  (h_pencils_per_row : pencils_per_row = 24) : 
  total_pencils / pencils_per_row = 30 := by 
  sorry

end faye_rows_l81_81114


namespace initial_average_l81_81647

variable (A : ℝ)
variables (nums : Fin 5 → ℝ)
variables (h_sum : 5 * A = nums 0 + nums 1 + nums 2 + nums 3 + nums 4)
variables (h_num : nums 0 = 12)
variables (h_new_avg : (5 * A + 12) / 5 = 9.2)

theorem initial_average :
  A = 6.8 :=
sorry

end initial_average_l81_81647


namespace distinct_socks_pairs_l81_81662

theorem distinct_socks_pairs (n : ℕ) (h : n = 9) : (Nat.choose n 2) = 36 := by
  rw [h]
  norm_num
  sorry

end distinct_socks_pairs_l81_81662


namespace fraction_multiplication_l81_81364

theorem fraction_multiplication :
  (3 / 4 : ℚ) * (1 / 2) * (2 / 5) * 5000 = 750 :=
by
  norm_num
  done

end fraction_multiplication_l81_81364


namespace chosen_number_is_129_l81_81976

theorem chosen_number_is_129 (x : ℕ) (h : 2 * x - 148 = 110) : x = 129 :=
by
  sorry

end chosen_number_is_129_l81_81976


namespace expression_not_defined_at_x_eq_5_l81_81739

theorem expression_not_defined_at_x_eq_5 :
  ∃ x : ℝ, x^3 - 15 * x^2 + 75 * x - 125 = 0 ↔ x = 5 :=
by
  sorry

end expression_not_defined_at_x_eq_5_l81_81739


namespace expression_not_computable_by_square_difference_l81_81680

theorem expression_not_computable_by_square_difference (x : ℝ) :
  ¬ ((x + 1) * (1 + x) = (x + 1) * (x - 1) ∨
     (x + 1) * (1 + x) = (-x + 1) * (-x - 1) ∨
     (x + 1) * (1 + x) = (x + 1) * (-x + 1)) :=
by
  sorry

end expression_not_computable_by_square_difference_l81_81680


namespace max_intersections_between_quadrilateral_and_pentagon_l81_81841

-- Definitions based on the conditions
def quadrilateral_sides : ℕ := 4
def pentagon_sides : ℕ := 5

-- Theorem statement based on the problem
theorem max_intersections_between_quadrilateral_and_pentagon 
  (qm_sides : ℕ := quadrilateral_sides) 
  (pm_sides : ℕ := pentagon_sides) : 
  (∀ (n : ℕ), n = qm_sides →
    ∀ (m : ℕ), m = pm_sides →
      ∀ (intersection_points : ℕ), 
        intersection_points = (n * m) →
        intersection_points = 20) :=
sorry

end max_intersections_between_quadrilateral_and_pentagon_l81_81841


namespace roots_reciprocal_sum_eq_25_l81_81206

theorem roots_reciprocal_sum_eq_25 (p q r : ℝ) (hpq : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) (hroot : ∀ x, x^3 - 9*x^2 + 8*x + 2 = 0 → (x = p ∨ x = q ∨ x = r)) :
  1/p^2 + 1/q^2 + 1/r^2 = 25 :=
by sorry

end roots_reciprocal_sum_eq_25_l81_81206


namespace tim_prank_combinations_l81_81229

def number_of_combinations (monday_choices : ℕ) (tuesday_choices : ℕ) (wednesday_choices : ℕ) (thursday_choices : ℕ) (friday_choices : ℕ) : ℕ :=
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations 2 3 0 6 1 = 0 :=
by
  -- Calculation yields 2 * 3 * 0 * 6 * 1 = 0
  sorry

end tim_prank_combinations_l81_81229


namespace inequality_correct_l81_81889

variable (m n c : ℝ)

theorem inequality_correct (h : m > n) : m + c > n + c := 
by sorry

end inequality_correct_l81_81889


namespace ratio_transformation_l81_81807

theorem ratio_transformation (x1 y1 x2 y2 : ℚ) (h₁ : x1 / y1 = 7 / 5) (h₂ : x2 = x1 * y1) (h₃ : y2 = y1 * x1) : x2 / y2 = 1 := by
  sorry

end ratio_transformation_l81_81807


namespace jordan_weight_after_exercise_l81_81914

theorem jordan_weight_after_exercise :
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  current_weight = 222 :=
by
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  show current_weight = 222
  sorry

end jordan_weight_after_exercise_l81_81914


namespace find_other_endpoint_l81_81474

theorem find_other_endpoint (x_m y_m : ℤ) (x1 y1 : ℤ) 
(m_cond : x_m = (x1 + (-1)) / 2) (m_cond' : y_m = (y1 + (-4)) / 2) : 
(x_m, y_m) = (3, -1) ∧ (x1, y1) = (7, 2) → (-1, -4) = (-1, -4) :=
by
  sorry

end find_other_endpoint_l81_81474


namespace a9_value_l81_81583

-- Define the sequence
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n+1) = 1 - (1 / a n)

-- State the theorem
theorem a9_value : ∃ a : ℕ → ℚ, seq a ∧ a 9 = -1/2 :=
by
  sorry

end a9_value_l81_81583


namespace eccentricity_of_hyperbola_l81_81744

variables (a b c e : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c = Real.sqrt (a^2 + b^2))
variable (h4 : 3 * -(a^2 / c) + c = a^2 * c / (b^2 - a^2) + c)
variable (h5 : e = c / a)

theorem eccentricity_of_hyperbola : e = Real.sqrt 3 :=
by {
  sorry
}

end eccentricity_of_hyperbola_l81_81744


namespace determine_p_q_l81_81997

theorem determine_p_q (r1 r2 p q : ℝ) (h1 : r1 + r2 = 5) (h2 : r1 * r2 = 6) (h3 : r1^2 + r2^2 = -p) (h4 : r1^2 * r2^2 = q) : p = -13 ∧ q = 36 :=
by
  sorry

end determine_p_q_l81_81997


namespace least_positive_number_divisible_by_five_smallest_primes_l81_81041

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l81_81041


namespace abs_eq_five_l81_81303

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l81_81303


namespace max_consecutive_sum_l81_81495

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l81_81495


namespace maximum_value_l81_81170

noncomputable def p : ℝ := 1 + 1/2 + 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5

theorem maximum_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h_constraint : (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 27) : 
  x^p + y^p + z^p ≤ 40.4 :=
sorry

end maximum_value_l81_81170


namespace number_of_dogs_l81_81214

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l81_81214


namespace find_k_l81_81208

theorem find_k {k : ℚ} :
    (∃ x y : ℚ, y = 3 * x + 6 ∧ y = -4 * x - 20 ∧ y = 2 * x + k) →
    k = 16 / 7 := 
  sorry

end find_k_l81_81208


namespace max_imag_part_of_roots_l81_81254

noncomputable def polynomial (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

theorem max_imag_part_of_roots :
  ∃ (z : ℂ), polynomial z = 0 ∧ ∀ w, polynomial w = 0 → (z.im ≤ w.im) := sorry

end max_imag_part_of_roots_l81_81254


namespace minimum_value_expr_l81_81055

theorem minimum_value_expr (x y : ℝ) : 
  (xy - 2)^2 + (x^2 + y^2)^2 ≥ 4 :=
sorry

end minimum_value_expr_l81_81055


namespace range_of_a_l81_81996

open Real

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 + a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l81_81996


namespace arithmetic_sequence_sum_l81_81317

open Real

noncomputable def a_n : ℕ → ℝ := sorry -- to represent the arithmetic sequence

theorem arithmetic_sequence_sum :
  (∃ d : ℝ, ∀ (n : ℕ), a_n n = a_n 1 + (n - 1) * d) ∧
  (∃ a1 a2011 : ℝ, (a_n 1 = a1) ∧ (a_n 2011 = a2011) ∧ (a1 ^ 2 - 10 * a1 + 16 = 0) ∧ (a2011 ^ 2 - 10 * a2011 + 16 = 0)) →
  a_n 2 + a_n 1006 + a_n 2010 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l81_81317


namespace log_base_27_of_3_l81_81562

theorem log_base_27_of_3 : log 27 3 = 1 / 3 :=
by
  have h : 27 = 3 ^ 3 := by norm_num
  rw [←h, log_rpow_self]
  norm_num
  sorry

end log_base_27_of_3_l81_81562


namespace functionMachine_output_l81_81264

-- Define the function machine according to the specified conditions
def functionMachine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  let step3 := if step2 <= 20 then step2 + 8 else step2 - 5
  step3

-- Statement: Prove that the functionMachine applied to 10 yields 25
theorem functionMachine_output : functionMachine 10 = 25 :=
  by
    sorry

end functionMachine_output_l81_81264


namespace find_initial_maple_trees_l81_81663

def initial_maple_trees (final_maple_trees planted_maple_trees : ℕ) : ℕ :=
  final_maple_trees - planted_maple_trees

theorem find_initial_maple_trees : initial_maple_trees 11 9 = 2 := by
  sorry

end find_initial_maple_trees_l81_81663


namespace range_f_does_not_include_zero_l81_81624

noncomputable def f (x : ℝ) : ℤ :=
if x > 0 then ⌈1 / (x + 1)⌉ else if x < 0 then ⌈1 / (x - 1)⌉ else 0 -- this will be used only as a formal definition

theorem range_f_does_not_include_zero : ¬ (0 ∈ {y : ℤ | ∃ x : ℝ, x ≠ 0 ∧ y = f x}) :=
by sorry

end range_f_does_not_include_zero_l81_81624


namespace max_consecutive_integers_sum_l81_81503

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l81_81503


namespace mul_103_97_l81_81262

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end mul_103_97_l81_81262


namespace intersection_M_N_l81_81268

def M : Set ℝ := { x | x^2 ≤ 4 }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≥ 1 }

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l81_81268


namespace eval_x2_sub_y2_l81_81139

theorem eval_x2_sub_y2 (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x + y = 13) : x^2 - y^2 = -40 := by
  sorry

end eval_x2_sub_y2_l81_81139


namespace find_number_l81_81523

variable (x : ℝ)

theorem find_number (h : 0.46 * x = 165.6) : x = 360 :=
sorry

end find_number_l81_81523


namespace min_speed_x_l81_81670

theorem min_speed_x (V_X : ℝ) : 
  let relative_speed_xy := V_X + 40;
  let relative_speed_xz := V_X - 30;
  (500 / relative_speed_xy) > (300 / relative_speed_xz) → 
  V_X ≥ 136 :=
by
  intros;
  sorry

end min_speed_x_l81_81670


namespace identity_holds_l81_81005

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l81_81005


namespace jordan_final_weight_l81_81911

-- Defining the initial weight and the weight losses over the specified weeks
def initial_weight := 250
def loss_first_4_weeks := 4 * 3
def loss_next_8_weeks := 8 * 2
def total_loss := loss_first_4_weeks + loss_next_8_weeks

-- Theorem stating the final weight of Jordan
theorem jordan_final_weight : initial_weight - total_loss = 222 := by
  -- We skip the proof as requested
  sorry

end jordan_final_weight_l81_81911


namespace least_number_divisible_by_five_smallest_primes_l81_81053

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l81_81053


namespace prove_value_of_custom_ops_l81_81279

-- Define custom operations to match problem statement
def custom_op1 (x : ℤ) : ℤ := 7 - x
def custom_op2 (x : ℤ) : ℤ := x - 10

-- The main proof statement
theorem prove_value_of_custom_ops : custom_op2 (custom_op1 12) = -15 :=
by sorry

end prove_value_of_custom_ops_l81_81279


namespace area_when_other_side_shortened_l81_81436

def original_width := 5
def original_length := 8
def target_area := 24
def shortened_amount := 2

theorem area_when_other_side_shortened :
  (original_width - shortened_amount) * original_length = target_area →
  original_width * (original_length - shortened_amount) = 30 :=
by
  intros h
  sorry

end area_when_other_side_shortened_l81_81436


namespace sqrt_expr_evaluation_l81_81392

theorem sqrt_expr_evaluation :
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3)) = 2 * Real.sqrt 2 :=
  sorry

end sqrt_expr_evaluation_l81_81392


namespace number_of_students_passed_both_tests_l81_81314

theorem number_of_students_passed_both_tests 
  (total_students : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both_tests : ℕ) 
  (students_with_union : ℕ := total_students) :
  (students_with_union = passed_long_jump + passed_shot_put - passed_both_tests + failed_both_tests) 
  → passed_both_tests = 25 :=
by sorry

end number_of_students_passed_both_tests_l81_81314


namespace A_n_is_integer_l81_81171

open Real

noncomputable def A_n (a b : ℕ) (θ : ℝ) (n : ℕ) : ℝ :=
  (a^2 + b^2)^n * sin (n * θ)

theorem A_n_is_integer (a b : ℕ) (h : a > b) (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < pi/2) (h_sin : sin θ = 2 * a * b / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, A_n a b θ n = k :=
by
  sorry

end A_n_is_integer_l81_81171


namespace number_of_girls_in_class_l81_81366

variable (B S G : ℕ)

theorem number_of_girls_in_class
  (h1 : (3 / 4 : ℚ) * B = 18)
  (h2 : B = (2 / 3 : ℚ) * S) :
  G = S - B → G = 12 := by
  intro hg
  sorry

end number_of_girls_in_class_l81_81366


namespace least_positive_number_divisible_by_five_primes_l81_81034

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l81_81034


namespace ivan_prob_more_than_5_points_l81_81608

open ProbabilityTheory Finset

/-- Conditions -/
def prob_correct_A : ℝ := 1 / 4
def prob_correct_B : ℝ := 1 / 3
def prob_A (k : ℕ) : ℝ := 
(C(10, k) * (prob_correct_A ^ k) * ((1 - prob_correct_A) ^ (10 - k)))

/-- Probabilities for type A problems -/
def prob_A_4 := ∑ i in (range 7).filter (λ i, i ≥ 4), prob_A i
def prob_A_6 := ∑ i in (range 7).filter (λ i, i ≥ 6), prob_A i

/-- Final combined probability -/
def final_prob : ℝ := 
(prob_A_4 * prob_correct_B) + (prob_A_6 * (1 - prob_correct_B))

/-- Proof -/
theorem ivan_prob_more_than_5_points : 
  final_prob = 0.088 := by
    sorry

end ivan_prob_more_than_5_points_l81_81608


namespace base_conversion_positive_b_l81_81349

theorem base_conversion_positive_b :
  (∃ (b : ℝ), 3 * 5^1 + 2 * 5^0 = 17 ∧ 1 * b^2 + 2 * b^1 + 0 * b^0 = 17 ∧ b = -1 + 3 * Real.sqrt 2) :=
by
  sorry

end base_conversion_positive_b_l81_81349


namespace max_consecutive_integers_sum_lt_500_l81_81492

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l81_81492


namespace probability_die_greater_than_4_after_3_tails_l81_81532

noncomputable def probability_question : ℚ :=
  let p_tails := (1/2)^3  -- Probability of getting three tails
  let p_heads_heads := (1/2) * (1/2)  -- Probability of getting two consecutive heads
  let p_die_greater_than_4 := (1/6) + (1/6)  -- Probability of the die showing a number greater than 4
  (p_heads_heads * p_die_greater_than_4)

theorem probability_die_greater_than_4_after_3_tails :
  probability_question = 1/12 := by
  sorry

end probability_die_greater_than_4_after_3_tails_l81_81532


namespace jane_total_score_l81_81601

theorem jane_total_score :
  let correct_answers := 17
  let incorrect_answers := 12
  let unanswered_questions := 6
  let total_questions := 35
  let points_per_correct := 1
  let points_per_incorrect := -0.25
  let correct_points := correct_answers * points_per_correct
  let incorrect_points := incorrect_answers * points_per_incorrect
  let total_score := correct_points + incorrect_points
  total_score = 14 :=
by
  sorry

end jane_total_score_l81_81601


namespace max_consecutive_sum_less_500_l81_81487

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l81_81487


namespace each_person_pays_l81_81345

def numPeople : ℕ := 6
def rentalDays : ℕ := 4
def weekdayRate : ℕ := 420
def weekendRate : ℕ := 540
def numWeekdays : ℕ := 2
def numWeekends : ℕ := 2

theorem each_person_pays : 
  (numWeekdays * weekdayRate + numWeekends * weekendRate) / numPeople = 320 :=
by
  sorry

end each_person_pays_l81_81345


namespace sum_of_coeffs_binomial_eq_32_l81_81226

noncomputable def sum_of_coeffs_binomial (x : ℝ) : ℝ :=
  (3 * x - 1 / Real.sqrt x)^5

theorem sum_of_coeffs_binomial_eq_32 :
  sum_of_coeffs_binomial 1 = 32 :=
by
  sorry

end sum_of_coeffs_binomial_eq_32_l81_81226


namespace maximum_consecutive_positive_integers_sum_500_l81_81498

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l81_81498


namespace probability_three_correct_letters_l81_81946

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))

noncomputable def derangement_count (n : ℕ) : ℕ :=
match n with
| 0 => 1
| 1 => 0
| n + 1 =>
    let d_n := derangement_count n
    let d_n1 := derangement_count (n - 1)
    n * (d_n + d_n1)

theorem probability_three_correct_letters :
  let total_distributions := factorial 7 in
  let favorable_distributions := choose 7 3 * derangement_count 4 in
  favorable_distributions / total_distributions = 1 / 16 :=
by
  let total_distributions := factorial 7
  let favorable_distributions := choose 7 3 * derangement_count 4
  have prob := (favorable_distributions : ℝ) / (total_distributions : ℝ)
  show prob = 1 / 16
  sorry

end probability_three_correct_letters_l81_81946


namespace smallest_number_of_students_l81_81971

theorem smallest_number_of_students 
  (A6 A7 A8 : Nat)
  (h1 : A8 * 3 = A6 * 5)
  (h2 : A8 * 5 = A7 * 8) :
  A6 + A7 + A8 = 89 :=
sorry

end smallest_number_of_students_l81_81971


namespace maximum_consecutive_positive_integers_sum_500_l81_81500

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l81_81500


namespace circle_equation_tangent_line_l81_81865

theorem circle_equation_tangent_line :
  ∃ r : ℝ, ∀ x y : ℝ, (x - 3)^2 + (y + 5)^2 = r^2 ↔ x - 7 * y + 2 = 0 :=
sorry

end circle_equation_tangent_line_l81_81865


namespace eden_stuffed_bears_l81_81717

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end eden_stuffed_bears_l81_81717


namespace hyperbola_vertex_distance_l81_81733

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l81_81733


namespace determine_y_value_l81_81162

theorem determine_y_value {k y : ℕ} (h1 : k > 0) (h2 : y > 0) (hk : k < 10) (hy : y < 10) :
  (8 * 100 + k * 10 + 8) + (k * 100 + 8 * 10 + 8) - (1 * 100 + 6 * 10 + y * 1) = 8 * 100 + k * 10 + 8 → 
  y = 9 :=
by
  sorry

end determine_y_value_l81_81162


namespace find_quadratic_function_l81_81277

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

theorem find_quadratic_function (a b : ℝ) :
  (∀ x, (quadratic_function a b (quadratic_function a b x - x)) / (quadratic_function a b x) = x^2 + 2023 * x + 1777) →
  a = 2025 ∧ b = 249 :=
by
  intro h
  sorry

end find_quadratic_function_l81_81277


namespace dandelion_seeds_percentage_approx_29_27_l81_81987

/-
Mathematical conditions:
- Carla has the following set of plants and seeds per plant:
  - 6 sunflowers with 9 seeds each
  - 8 dandelions with 12 seeds each
  - 4 roses with 7 seeds each
  - 10 tulips with 15 seeds each.
- Calculate:
  - total seeds
  - percentage of seeds from dandelions
-/ 

def num_sunflowers : ℕ := 6
def num_dandelions : ℕ := 8
def num_roses : ℕ := 4
def num_tulips : ℕ := 10

def seeds_per_sunflower : ℕ := 9
def seeds_per_dandelion : ℕ := 12
def seeds_per_rose : ℕ := 7
def seeds_per_tulip : ℕ := 15

def total_sunflower_seeds : ℕ := num_sunflowers * seeds_per_sunflower
def total_dandelion_seeds : ℕ := num_dandelions * seeds_per_dandelion
def total_rose_seeds : ℕ := num_roses * seeds_per_rose
def total_tulip_seeds : ℕ := num_tulips * seeds_per_tulip

def total_seeds : ℕ := total_sunflower_seeds + total_dandelion_seeds + total_rose_seeds + total_tulip_seeds

def percentage_dandelion_seeds : ℚ := (total_dandelion_seeds : ℚ) / total_seeds * 100

theorem dandelion_seeds_percentage_approx_29_27 : abs (percentage_dandelion_seeds - 29.27) < 0.01 :=
sorry

end dandelion_seeds_percentage_approx_29_27_l81_81987


namespace probability_Ivan_more_than_5_points_l81_81611

noncomputable def prob_type_A_correct := 1 / 4
noncomputable def total_type_A := 10
noncomputable def prob_type_B_correct := 1 / 3

def binomial (n k : ℕ) : ℚ :=
  (Nat.choose n k) * (prob_type_A_correct ^ k) * ((1 - prob_type_A_correct) ^ (n - k))

def prob_A_4 := ∑ k in finset.range (total_type_A + 1), if k ≥ 4 then binomial total_type_A k else 0
def prob_A_6 := ∑ k in finset.range (total_type_A + 1), if k ≥ 6 then binomial total_type_A k else 0

def prob_B := prob_type_B_correct
def prob_not_B := 1 - prob_type_B_correct

noncomputable def prob_more_than_5_points :=
  prob_A_4 * prob_B + prob_A_6 * prob_not_B

theorem probability_Ivan_more_than_5_points :
  prob_more_than_5_points = 0.088 := by
  sorry

end probability_Ivan_more_than_5_points_l81_81611


namespace math_problem_l81_81785

variable {a b c d e f : ℕ}
variable (h1 : f < a)
variable (h2 : (a * b * d + 1) % c = 0)
variable (h3 : (a * c * e + 1) % b = 0)
variable (h4 : (b * c * f + 1) % a = 0)

theorem math_problem
  (h5 : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by {
  skip -- Adding "by" ... "sorry" to make the statement complete since no proof is required.
  sorry
}

end math_problem_l81_81785


namespace express_in_scientific_notation_l81_81646

theorem express_in_scientific_notation 
  (A : 149000000 = 149 * 10^6)
  (B : 149000000 = 1.49 * 10^8)
  (C : 149000000 = 14.9 * 10^7)
  (D : 149000000 = 1.5 * 10^8) :
  149000000 = 1.49 * 10^8 := 
by
  sorry

end express_in_scientific_notation_l81_81646


namespace expression_comparison_l81_81954

theorem expression_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) :
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  (exprI = exprII ∨ exprI = exprIII ∨ exprII = exprIII ∨ 
   (exprI > exprII ∧ exprI > exprIII) ∨
   (exprII > exprI ∧ exprII > exprIII) ∨
   (exprIII > exprI ∧ exprIII > exprII)) ∧
  ¬((exprI > exprII ∧ exprI > exprIII) ∨
    (exprII > exprI ∧ exprII > exprIII) ∨
    (exprIII > exprI ∧ exprIII > exprII)) :=
by
  let exprI := (a + (1 / a)) * (b + (1 / b))
  let exprII := (Real.sqrt (a * b) + (1 / Real.sqrt (a * b))) ^ 2
  let exprIII := (((a + b) / 2) + (2 / (a + b))) ^ 2
  sorry

end expression_comparison_l81_81954


namespace net_price_change_l81_81517

theorem net_price_change (P : ℝ) : 
  let decreased_price := P * (1 - 0.30)
  let increased_price := decreased_price * (1 + 0.20)
  increased_price - P = -0.16 * P :=
by
  -- The proof would go here. We just need the statement as per the prompt.
  sorry

end net_price_change_l81_81517


namespace unique_t_digit_l81_81815

theorem unique_t_digit (t : ℕ) (ht : t < 100) (ht2 : 10 ≤ t) (h : 13 * t ≡ 42 [MOD 100]) : t = 34 := 
by
-- Proof is omitted
sorry

end unique_t_digit_l81_81815


namespace parallelogram_sides_eq_l81_81357

theorem parallelogram_sides_eq (x y : ℚ) :
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -1.6 := by
  sorry

end parallelogram_sides_eq_l81_81357


namespace min_value_y1_y2_sq_l81_81294

theorem min_value_y1_y2_sq (k : ℝ) (y1 y2 : ℝ) :
  ∃ y1 y2, y1 + y2 = 4 / k ∧ y1 * y2 = -4 ∧ y1^2 + y2^2 = 8 :=
sorry

end min_value_y1_y2_sq_l81_81294


namespace max_value_function_l81_81762

theorem max_value_function (x : ℝ) (h : x > 4) : -x + (1 / (4 - x)) ≤ -6 :=
sorry

end max_value_function_l81_81762


namespace fraction_irreducible_l81_81637

open Nat

theorem fraction_irreducible (m n : ℕ) : Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 :=
  sorry

end fraction_irreducible_l81_81637


namespace equation_of_ellipse_AN_BM_constant_l81_81406

noncomputable def a := 2
noncomputable def b := 1
noncomputable def e := (Real.sqrt 3) / 2
noncomputable def c := Real.sqrt 3

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem equation_of_ellipse :
  ellipse a b
:=
by
  sorry

theorem AN_BM_constant (x0 y0 : ℝ) (hx : x0^2 + 4 * y0^2 = 4) :
  let AN := 2 + x0 / (y0 - 1)
  let BM := 1 + 2 * y0 / (x0 - 2)
  abs (AN * BM) = 4
:=
by
  sorry

end equation_of_ellipse_AN_BM_constant_l81_81406


namespace solve_equation_l81_81461

variable (x : ℝ)

def equation := (x / (2 * x - 3)) + (5 / (3 - 2 * x)) = 4
def condition := x ≠ 3 / 2

theorem solve_equation : equation x ∧ condition x → x = 1 :=
by
  sorry

end solve_equation_l81_81461


namespace find_d_l81_81737

theorem find_d : ∃ d : ℝ, (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) ∧ d = 8 :=
by
  sorry

end find_d_l81_81737


namespace james_total_cost_is_100_l81_81618

def cost_of_shirts (number_of_shirts : Nat) (cost_per_shirt : Nat) : Nat :=
  number_of_shirts * cost_per_shirt

def cost_of_pants (number_of_pants : Nat) (cost_per_pants : Nat) : Nat :=
  number_of_pants * cost_per_pants

def total_cost (number_of_shirts : Nat) (number_of_pants : Nat) (cost_per_shirt : Nat) (cost_per_pants : Nat) : Nat :=
  cost_of_shirts number_of_shirts cost_per_shirt + cost_of_pants number_of_pants cost_per_pants

theorem james_total_cost_is_100 : 
  total_cost 10 (10 / 2) 6 8 = 100 :=
by
  sorry

end james_total_cost_is_100_l81_81618


namespace cone_sector_volume_ratio_l81_81537

theorem cone_sector_volume_ratio 
  (H R : ℝ) 
  (nonneg_H : 0 ≤ H) 
  (nonneg_R : 0 ≤ R) :
  let volume_original := (1/3) * π * R^2 * H
  let volume_sector   := (1/12) * π * R^2 * H
  volume_sector / volume_sector = 1 :=
  by
    sorry

end cone_sector_volume_ratio_l81_81537


namespace simplify_expr1_simplify_expr2_l81_81185

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l81_81185


namespace rent_cost_l81_81437

-- Definitions based on conditions
def daily_supplies_cost : ℕ := 12
def price_per_pancake : ℕ := 2
def pancakes_sold_per_day : ℕ := 21

-- Proving the daily rent cost
theorem rent_cost (total_sales : ℕ) (rent : ℕ) :
  total_sales = pancakes_sold_per_day * price_per_pancake →
  rent = total_sales - daily_supplies_cost →
  rent = 30 :=
by
  intro h_total_sales h_rent
  sorry

end rent_cost_l81_81437


namespace geometric_sum_l81_81896

theorem geometric_sum 
  (a : ℕ → ℝ) (q : ℝ) (h1 : a 2 + a 4 = 32) (h2 : a 6 + a 8 = 16) 
  (h_seq : ∀ n, a (n+2) = a n * q ^ 2):
  a 10 + a 12 + a 14 + a 16 = 12 :=
by
  -- Proof needs to be written here
  sorry

end geometric_sum_l81_81896


namespace monotonic_intervals_logarithmic_inequality_l81_81753

noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

theorem monotonic_intervals :
  (∀ x ∈ Set.Ioo 0 1, f x > f (x + 1E-9) ∧ f x < f (x - 1E-9)) ∧ 
  (∀ y ∈ Set.Ioi 1, f y < f (y + 1E-9) ∧ f y > f (y - 1E-9)) := sorry

theorem logarithmic_inequality (a : ℝ) (ha : a > 0) (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hneq : x1 ≠ x2)
  (h_eq1 : a * x1 + f x1 = x1^2 - x1) (h_eq2 : a * x2 + f x2 = x2^2 - x2) :
  Real.log x1 + Real.log x2 + 2 * Real.log a < 0 := sorry

end monotonic_intervals_logarithmic_inequality_l81_81753


namespace sum_of_consecutive_even_integers_l81_81825

theorem sum_of_consecutive_even_integers
  (a1 a2 a3 a4 : ℤ)
  (h1 : a2 = a1 + 2)
  (h2 : a3 = a1 + 4)
  (h3 : a4 = a1 + 6)
  (h_sum : a1 + a3 = 146) :
  a1 + a2 + a3 + a4 = 296 :=
by sorry

end sum_of_consecutive_even_integers_l81_81825


namespace lines_intersect_l81_81355

theorem lines_intersect (a b : ℝ) (h1 : 2 = (1/3) * 1 + a) (h2 : 1 = (1/2) * 2 + b) : a + b = 5 / 3 := 
by {
  -- Skipping the proof itself
  sorry
}

end lines_intersect_l81_81355


namespace smallest_x_value_l81_81187

theorem smallest_x_value (x : ℝ) (h : 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36)) : x = -3 :=
sorry

end smallest_x_value_l81_81187


namespace positive_difference_is_329_l81_81674

-- Definitions of the fractions involved
def fraction1 : ℚ := (7^2 + 7^2) / 7
def fraction2 : ℚ := (7^2 * 7^2) / 7

-- Statement of the positive difference proof
theorem positive_difference_is_329 : abs (fraction2 - fraction1) = 329 := by
  -- Skipping the proof here
  sorry

end positive_difference_is_329_l81_81674


namespace gcd_80_180_450_l81_81824

theorem gcd_80_180_450 : Int.gcd (Int.gcd 80 180) 450 = 10 := by
  sorry

end gcd_80_180_450_l81_81824


namespace cannot_be_computed_using_square_diff_l81_81685

-- Define the conditions
def A := (x + 1) * (x - 1)
def B := (-x + 1) * (-x - 1)
def C := (x + 1) * (-x + 1)
def D := (x + 1) * (1 + x)

-- Proposition: Prove that D cannot be computed using the square difference formula
theorem cannot_be_computed_using_square_diff (x : ℝ) : 
  ¬ (D = x^2 - y^2) := 
sorry

end cannot_be_computed_using_square_diff_l81_81685


namespace fraction_checked_by_worker_y_l81_81861

-- Definitions of conditions given in the problem
variable (P Px Py : ℝ)
variable (h1 : Px + Py = P)
variable (h2 : 0.005 * Px = defective_x)
variable (h3 : 0.008 * Py = defective_y)
variable (defective_x defective_y : ℝ)
variable (total_defective : ℝ)
variable (h4 : defective_x + defective_y = total_defective)
variable (h5 : total_defective = 0.0065 * P)

-- The fraction of products checked by worker y
theorem fraction_checked_by_worker_y (h : Px + Py = P) (h2 : 0.005 * Px = 0.0065 * P) (h3 : 0.008 * Py = 0.0065 * P) :
  Py / P = 1 / 2 := 
  sorry

end fraction_checked_by_worker_y_l81_81861


namespace altitude_division_l81_81821

variables {A B C D E : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E]

theorem altitude_division 
  (AD DC CE EB y : ℝ)
  (hAD : AD = 6)
  (hDC : DC = 4)
  (hCE : CE = 3)
  (hEB : EB = y)
  (h_similarity : CE / DC = (AD + DC) / (y + CE)) : 
  y = 31 / 3 :=
by
  sorry

end altitude_division_l81_81821


namespace greatest_term_in_expansion_l81_81953

theorem greatest_term_in_expansion :
  ∃ k : ℕ, k = 63 ∧
  (∀ n : ℕ, n ∈ (Finset.range 101) → n ≠ k → 
    (Nat.choose 100 n * (Real.sqrt 3)^n) < 
    (Nat.choose 100 k * (Real.sqrt 3)^k)) :=
by
  sorry

end greatest_term_in_expansion_l81_81953


namespace probability_nina_taller_than_lena_is_zero_l81_81125

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l81_81125


namespace car_interval_length_l81_81468

theorem car_interval_length (S1 T : ℝ) (interval_length : ℝ) 
  (h1 : S1 = 39) 
  (h2 : (fun (n : ℕ) => S1 - 3 * n) 4 = 27)
  (h3 : 3.6 = 27 * T) 
  (h4 : interval_length = T * 60) :
  interval_length = 8 :=
by
  sorry

end car_interval_length_l81_81468


namespace find_number_l81_81635

-- Define the condition that one-third of a certain number is 300% of 134
def one_third_eq_300percent_number (n : ℕ) : Prop :=
  n / 3 = 3 * 134

-- State the theorem that the number is 1206 given the above condition
theorem find_number (n : ℕ) (h : one_third_eq_300percent_number n) : n = 1206 :=
  by sorry

end find_number_l81_81635


namespace max_consecutive_sum_less_500_l81_81486

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l81_81486


namespace card_draw_suit_probability_l81_81305

noncomputable def probability_at_least_one_card_each_suit : ℚ :=
  3 / 32

theorem card_draw_suit_probability : 
  (∃ deck : set ℕ, deck = set.univ ∧ ∀ suit : ℕ in deck, suit < 4) →
  (∃ draws : list ℕ, length draws = 5) →
  ∃ prob : ℚ, prob = probability_at_least_one_card_each_suit :=
sorry

end card_draw_suit_probability_l81_81305


namespace area_bounded_by_curves_eq_l81_81064

open Real

noncomputable def area_bounded_by_curves : ℝ :=
  1 / 2 * (∫ (φ : ℝ) in (π/4)..(π/2), (sqrt 2 * cos (φ - π / 4))^2) +
  1 / 2 * (∫ (φ : ℝ) in (π/2)..(3 * π / 4), (sqrt 2 * sin (φ - π / 4))^2)

theorem area_bounded_by_curves_eq : area_bounded_by_curves = (π + 2) / 4 :=
  sorry

end area_bounded_by_curves_eq_l81_81064


namespace total_pieces_of_art_l81_81061

variable (A : ℕ) (displayed : ℕ) (sculptures_on_display : ℕ) (not_on_display : ℕ) (paintings_not_on_display : ℕ) (sculptures_not_on_display : ℕ)

-- Constants and conditions from the problem
axiom H1 : displayed = 1 / 3 * A
axiom H2 : sculptures_on_display = 1 / 6 * displayed
axiom H3 : not_on_display = 2 / 3 * A
axiom H4 : paintings_not_on_display = 1 / 3 * not_on_display
axiom H5 : sculptures_not_on_display = 800
axiom H6 : sculptures_not_on_display = 2 / 3 * not_on_display

-- Prove that the total number of pieces of art is 1800
theorem total_pieces_of_art : A = 1800 :=
by
  sorry

end total_pieces_of_art_l81_81061


namespace least_number_subtracted_l81_81826

/--
  What least number must be subtracted from 9671 so that the remaining number is divisible by 5, 7, and 11?
-/
theorem least_number_subtracted
  (x : ℕ) :
  (9671 - x) % 5 = 0 ∧ (9671 - x) % 7 = 0 ∧ (9671 - x) % 11 = 0 ↔ x = 46 :=
sorry

end least_number_subtracted_l81_81826


namespace boat_downstream_distance_l81_81693

-- Given conditions
def speed_boat_still_water : ℕ := 25
def speed_stream : ℕ := 5
def travel_time_downstream : ℕ := 3

-- Proof statement: The distance travelled downstream is 90 km
theorem boat_downstream_distance :
  speed_boat_still_water + speed_stream * travel_time_downstream = 90 :=
by
  -- omitting the actual proof steps
  sorry

end boat_downstream_distance_l81_81693


namespace nat_games_volunteer_allocation_l81_81932

theorem nat_games_volunteer_allocation 
  (volunteers : Fin 6 → Type) 
  (venues : Fin 3 → Type)
  (A B : volunteers 0)
  (remaining : Fin 4 → Type) 
  (assigned_pairings : Π (v : Fin 3), Fin 2 → volunteers 0) :
  (∀ v, assigned_pairings v 0 = A ∨ assigned_pairings v 1 = B) →
  (3 * 6 = 18) := 
by
  sorry

end nat_games_volunteer_allocation_l81_81932


namespace least_positive_whole_number_divisible_by_five_primes_l81_81044

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l81_81044


namespace scientific_notation_819000_l81_81903

theorem scientific_notation_819000 :
  ∃ (a : ℝ) (n : ℤ), 819000 = a * 10 ^ n ∧ a = 8.19 ∧ n = 5 :=
by
  -- Proof goes here
  sorry

end scientific_notation_819000_l81_81903


namespace width_of_grass_field_l81_81536

-- Define the conditions
def length_of_grass_field : ℝ := 75
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2
def total_cost : ℝ := 1200

-- Define the width of the grass field as a variable
variable (w : ℝ)

-- Define the total length and width including the path
def total_length : ℝ := length_of_grass_field + 2 * path_width
def total_width (w : ℝ) : ℝ := w + 2 * path_width

-- Define the area of the path
def area_of_path (w : ℝ) : ℝ := (total_length * total_width w) - (length_of_grass_field * w)

-- Define the cost equation
def cost_eq (w : ℝ) : Prop := cost_per_sq_m * area_of_path w = total_cost

-- The theorem to prove
theorem width_of_grass_field : cost_eq 40 :=
by
  -- To be proved
  sorry

end width_of_grass_field_l81_81536


namespace natural_number_sets_solution_l81_81572

theorem natural_number_sets_solution (x y n : ℕ) (h : (x! + y!) / n! = 3^n) : (x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
by
  sorry

end natural_number_sets_solution_l81_81572


namespace solve_for_x_l81_81789

variable (x : ℝ)

theorem solve_for_x (h : (4 * x + 2) / (5 * x - 5) = 3 / 4) : x = -23 := 
by
  sorry

end solve_for_x_l81_81789


namespace number_of_distinct_lines_l81_81887

noncomputable def count_lines_through_five_points :
  ℕ :=
  let points := { (i, j, k) // 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 },
      lines := { l : fin 27 → points // ∀ n m, n ≠ m → l n ≠ l m },
  finset.card lines

theorem number_of_distinct_lines :
  count_lines_through_five_points = 15 :=
sorry

end number_of_distinct_lines_l81_81887


namespace fourth_leg_length_l81_81333

theorem fourth_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) :
  (∃ x : ℕ, x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ (a + x = b + c ∨ b + x = a + c ∨ c + x = a + b) ∧ (x = 7 ∨ x = 11)) :=
by sorry

end fourth_leg_length_l81_81333


namespace ratio_students_above_8_to_8_years_l81_81160

-- Definitions of the problem's known conditions
def total_students : ℕ := 125
def students_below_8_years : ℕ := 25
def students_of_8_years : ℕ := 60

-- Main proof inquiry
theorem ratio_students_above_8_to_8_years :
  ∃ (A : ℕ), students_below_8_years + students_of_8_years + A = total_students ∧
             A * 3 = students_of_8_years * 2 := 
sorry

end ratio_students_above_8_to_8_years_l81_81160


namespace least_positive_divisible_by_five_primes_l81_81051

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l81_81051


namespace power_function_properties_l81_81764

theorem power_function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ a) (h2 : f 2 = Real.sqrt 2) : 
  a = 1 / 2 ∧ ∀ x, 0 ≤ x → f x ≤ f (x + 1) :=
by
  sorry

end power_function_properties_l81_81764


namespace molecular_weight_compound_l81_81232

theorem molecular_weight_compound :
  let weight_H := 1.008
  let weight_Cr := 51.996
  let weight_O := 15.999
  let n_H := 2
  let n_Cr := 1
  let n_O := 4
  (n_H * weight_H) + (n_Cr * weight_Cr) + (n_O * weight_O) = 118.008 :=
by
  sorry

end molecular_weight_compound_l81_81232


namespace sum_of_fractions_eq_five_fourteen_l81_81571

theorem sum_of_fractions_eq_five_fourteen :
  (1 : ℚ) / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) = 5 / 14 := 
by
  sorry

end sum_of_fractions_eq_five_fourteen_l81_81571


namespace rect_area_correct_l81_81831

-- Defining the function to calculate the area of a rectangle given the coordinates of its vertices
noncomputable def rect_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ) : ℤ :=
  let length := abs (x2 - x1)
  let width := abs (y1 - y3)
  length * width

-- The vertices of the rectangle
def x1 : ℤ := -8
def y1 : ℤ := 1
def x2 : ℤ := 1
def y2 : ℤ := 1
def x3 : ℤ := 1
def y3 : ℤ := -7
def x4 : ℤ := -8
def y4 : ℤ := -7

-- Proving that the area of the rectangle is 72 square units
theorem rect_area_correct : rect_area x1 y1 x2 y2 x3 y3 x4 y4 = 72 := by
  sorry

end rect_area_correct_l81_81831


namespace volume_of_rectangular_prism_l81_81944

theorem volume_of_rectangular_prism
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : z * x = 12) :
  x * y * z = 60 :=
sorry

end volume_of_rectangular_prism_l81_81944


namespace complex_root_product_value_l81_81779

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end complex_root_product_value_l81_81779


namespace ratio_of_bronze_to_silver_l81_81545

def total_gold_coins := 3500
def num_chests := 5
def total_silver_coins := 500
def coins_per_chest := 1000

-- Definitions based on the conditions to be used in the proof
def gold_coins_per_chest := total_gold_coins / num_chests
def silver_coins_per_chest := total_silver_coins / num_chests
def bronze_coins_per_chest := coins_per_chest - gold_coins_per_chest - silver_coins_per_chest
def bronze_to_silver_ratio := bronze_coins_per_chest / silver_coins_per_chest

theorem ratio_of_bronze_to_silver : bronze_to_silver_ratio = 2 := 
by
  sorry

end ratio_of_bronze_to_silver_l81_81545


namespace quadratic_distinct_real_roots_l81_81738

-- Defining the main hypothesis
theorem quadratic_distinct_real_roots (k : ℝ) :
  (k < 4 / 3) ∧ (k ≠ 1) ↔ (∀ x : ℂ, ((k-1) * x^2 - 2 * x + 3 = 0) → ∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ ((k-1) * x₁ ^ 2 - 2 * x₁ + 3 = 0) ∧ ((k-1) * x₂ ^ 2 - 2 * x₂ + 3 = 0)) := by
sorry

end quadratic_distinct_real_roots_l81_81738


namespace largest_of_six_consecutive_sum_2070_is_347_l81_81478

theorem largest_of_six_consecutive_sum_2070_is_347 (n : ℕ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070 → n + 5 = 347 :=
by
  intro h
  sorry

end largest_of_six_consecutive_sum_2070_is_347_l81_81478


namespace find_angle_beta_l81_81872

open Real

theorem find_angle_beta
  (α β : ℝ)
  (h1 : sin α = (sqrt 5) / 5)
  (h2 : sin (α - β) = - (sqrt 10) / 10)
  (hα_range : 0 < α ∧ α < π / 2)
  (hβ_range : 0 < β ∧ β < π / 2) :
  β = π / 4 :=
sorry

end find_angle_beta_l81_81872


namespace function_decomposition_l81_81786

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (a : ℝ) (f₁ f₂ : ℝ → ℝ), a > 0 ∧ (∀ x, f₁ x = f₁ (-x)) ∧ (∀ x, f₂ x = f₂ (2 * a - x)) ∧ (∀ x, f x = f₁ x + f₂ x) :=
sorry

end function_decomposition_l81_81786


namespace smallest_four_digit_integer_l81_81677

theorem smallest_four_digit_integer (n : ℕ) :
  (75 * n ≡ 225 [MOD 450]) ∧ (1000 ≤ n ∧ n < 10000) → n = 1005 :=
sorry

end smallest_four_digit_integer_l81_81677


namespace add_to_fraction_eq_l81_81234

theorem add_to_fraction_eq (n : ℕ) : (4 + n) / (7 + n) = 6 / 7 → n = 14 :=
by sorry

end add_to_fraction_eq_l81_81234


namespace ten_person_round_robin_l81_81590

def number_of_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem ten_person_round_robin : number_of_matches 10 = 45 :=
by
  -- Proof steps would go here, but are omitted for this task
  sorry

end ten_person_round_robin_l81_81590


namespace employees_excluding_manager_l81_81467

theorem employees_excluding_manager (E : ℕ) (avg_salary_employee : ℕ) (manager_salary : ℕ) (new_avg_salary : ℕ) (total_employees_with_manager : ℕ) :
  avg_salary_employee = 1800 →
  manager_salary = 4200 →
  new_avg_salary = avg_salary_employee + 150 →
  total_employees_with_manager = E + 1 →
  (1800 * E + 4200) / total_employees_with_manager = new_avg_salary →
  E = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end employees_excluding_manager_l81_81467


namespace min_value_geom_seq_l81_81750

theorem min_value_geom_seq (a : ℕ → ℝ) (r m n : ℕ) (h_geom : ∃ r, ∀ i, a (i + 1) = a i * r)
  (h_ratio : r = 2) (h_a_m : 4 * a 1 = a m) :
  ∃ (m n : ℕ), (m + n = 6) → (1 / m + 4 / n) = 3 / 2 :=
by 
  sorry

end min_value_geom_seq_l81_81750


namespace equilateral_triangle_side_length_l81_81933

theorem equilateral_triangle_side_length (c : ℕ) (h : c = 4 * 21) : c / 3 = 28 := by
  sorry

end equilateral_triangle_side_length_l81_81933


namespace fraction_evaluation_l81_81727

theorem fraction_evaluation :
  ( (1 / 2 * 1 / 3 * 1 / 4 * 1 / 5 + 3 / 2 * 3 / 4 * 3 / 5) / 
    (1 / 2 * 2 / 3 * 2 / 5) ) = 41 / 8 :=
by
  sorry

end fraction_evaluation_l81_81727


namespace eventually_non_multiples_of_5_l81_81250

def sequence_condition (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if a n % 5 = 0 then a n / 5 else Nat.floor (Real.sqrt 5 * (a n))

theorem eventually_non_multiples_of_5 (a : ℕ → ℕ) (h0 : 0 < a 0)
  (h1 : ∀ n, a (n + 1) = sequence_condition a n) :
  ∃ N, ∀ n, N ≤ n → a n % 5 ≠ 0 :=
by sorry

end eventually_non_multiples_of_5_l81_81250


namespace fewest_four_dollar_frisbees_l81_81060

-- Definitions based on the conditions
variables (x y : ℕ) -- The numbers of $3 and $4 frisbees, respectively.
def total_frisbees (x y : ℕ) : Prop := x + y = 60
def total_receipts (x y : ℕ) : Prop := 3 * x + 4 * y = 204

-- The statement to prove
theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : total_frisbees x y) (h2 : total_receipts x y) : y = 24 :=
sorry

end fewest_four_dollar_frisbees_l81_81060


namespace area_of_enclosed_figure_l81_81796

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), ((x)^(1/2) - x^2)

theorem area_of_enclosed_figure :
  area_enclosed_by_curves = (1 / 3) :=
by
  sorry

end area_of_enclosed_figure_l81_81796


namespace fewest_erasers_l81_81169

theorem fewest_erasers :
  ∀ (JK JM SJ : ℕ), 
  (JK = 6) →
  (JM = JK + 4) →
  (SJ = JM - 3) →
  (JK ≤ JM ∧ JK ≤ SJ) :=
by
  intros JK JM SJ hJK hJM hSJ
  sorry

end fewest_erasers_l81_81169


namespace bobby_has_candy_left_l81_81090

def initial_candy := 36
def candy_eaten_first := 17
def candy_eaten_second := 15

theorem bobby_has_candy_left : 
  initial_candy - (candy_eaten_first + candy_eaten_second) = 4 := 
by
  sorry


end bobby_has_candy_left_l81_81090


namespace abs_eq_five_l81_81301

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l81_81301


namespace natural_number_1981_l81_81248

theorem natural_number_1981 (x : ℕ) 
  (h1 : ∃ a : ℕ, x - 45 = a^2)
  (h2 : ∃ b : ℕ, x + 44 = b^2) :
  x = 1981 :=
sorry

end natural_number_1981_l81_81248


namespace senior_year_allowance_more_than_twice_l81_81915

noncomputable def middle_school_allowance : ℝ :=
  8 + 2

noncomputable def twice_middle_school_allowance : ℝ :=
  2 * middle_school_allowance

noncomputable def senior_year_increase : ℝ :=
  1.5 * middle_school_allowance

noncomputable def senior_year_allowance : ℝ :=
  middle_school_allowance + senior_year_increase

theorem senior_year_allowance_more_than_twice : 
  senior_year_allowance = twice_middle_school_allowance + 5 :=
by
  sorry

end senior_year_allowance_more_than_twice_l81_81915


namespace max_value_expression_l81_81401

theorem max_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 5) : 
  (∀ x y : ℝ, x = 2 * a + 2 → y = 3 * b + 1 → x * y ≤ 16) := by
  sorry

end max_value_expression_l81_81401


namespace f_zero_one_and_odd_l81_81585

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (a b : ℝ) : f (a * b) = a * f b + b * f a
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

theorem f_zero_one_and_odd :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end f_zero_one_and_odd_l81_81585


namespace friends_carrying_bananas_l81_81922

theorem friends_carrying_bananas :
  let total_friends := 35
  let friends_with_pears := 14
  let friends_with_oranges := 8
  let friends_with_apples := 5
  total_friends - (friends_with_pears + friends_with_oranges + friends_with_apples) = 8 := 
by
  sorry

end friends_carrying_bananas_l81_81922


namespace clock_four_different_digits_l81_81205

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l81_81205


namespace express_vector_c_as_linear_combination_l81_81740

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := (1, -1)
noncomputable def c : ℝ × ℝ := (2, 3)

theorem express_vector_c_as_linear_combination :
  ∃ x y : ℝ, c = (x * (1, 1).1 + y * (1, -1).1, x * (1, 1).2 + y * (1, -1).2) ∧
             x = 5 / 2 ∧ y = -1 / 2 :=
by
  sorry

end express_vector_c_as_linear_combination_l81_81740


namespace other_root_of_quadratic_eq_l81_81308

theorem other_root_of_quadratic_eq (m : ℝ) (t : ℝ) (h1 : (polynomial.X ^ 2 + polynomial.C m * polynomial.X + polynomial.C (-6)).roots = {2, t}) : t = -3 :=
sorry

end other_root_of_quadratic_eq_l81_81308


namespace clock_shows_four_different_digits_for_588_minutes_l81_81191

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l81_81191


namespace max_value_f_at_a1_f_div_x_condition_l81_81146

noncomputable def f (a x : ℝ) : ℝ := (a - x) * Real.exp x - 1

theorem max_value_f_at_a1 :
  ∀ x : ℝ, (f 1 0) = 0 ∧ ( ∀ y : ℝ, y ≠ 0 → f 1 y < f 1 0) := 
sorry

theorem f_div_x_condition :
  ∀ x : ℝ, x ≠ 0 → (((f 1 x) / x) < 1) :=
sorry

end max_value_f_at_a1_f_div_x_condition_l81_81146


namespace clock_displays_unique_digits_minutes_l81_81196

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l81_81196


namespace jasons_shelves_l81_81438

theorem jasons_shelves (total_books : ℕ) (number_of_shelves : ℕ) (h_total_books : total_books = 315) (h_number_of_shelves : number_of_shelves = 7) : (total_books / number_of_shelves) = 45 := 
by
  sorry

end jasons_shelves_l81_81438


namespace bus_speed_excluding_stoppages_l81_81862

theorem bus_speed_excluding_stoppages (v : Real) 
  (h1 : ∀ x, x = 41) 
  (h2 : ∀ y, y = 14.444444444444443 / 60) : 
  v = 54 := 
by
  -- Proving the statement. Proof steps are skipped.
  sorry

end bus_speed_excluding_stoppages_l81_81862


namespace xiao_hua_correct_answers_l81_81897

theorem xiao_hua_correct_answers :
  ∃ (correct_answers wrong_answers : ℕ), 
    correct_answers + wrong_answers = 15 ∧
    8 * correct_answers - 4 * wrong_answers = 72 ∧
    correct_answers = 11 :=
by
  sorry

end xiao_hua_correct_answers_l81_81897


namespace count_valid_x_satisfying_heartsuit_condition_l81_81773

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem count_valid_x_satisfying_heartsuit_condition :
  (∃ n, ∀ x, 1 ≤ x ∧ x < 1000 → digit_sum (digit_sum x) = 4 → n = 36) :=
by
  sorry

end count_valid_x_satisfying_heartsuit_condition_l81_81773


namespace probability_N_lt_L_is_zero_l81_81129

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l81_81129


namespace arithmetic_sum_first_11_terms_l81_81358

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

variable (a : ℕ → ℝ)

theorem arithmetic_sum_first_11_terms (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_sum_condition : a 2 + a 6 + a 10 = 6) :
  sum_first_n_terms a 11 = 22 :=
sorry

end arithmetic_sum_first_11_terms_l81_81358


namespace replace_stars_identity_l81_81000

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l81_81000


namespace speed_of_water_l81_81972

variable (v : ℝ)
variable (swimming_speed_still_water : ℝ := 10)
variable (time_against_current : ℝ := 8)
variable (distance_against_current : ℝ := 16)

theorem speed_of_water :
  distance_against_current = (swimming_speed_still_water - v) * time_against_current ↔ v = 8 := by
  sorry

end speed_of_water_l81_81972


namespace segment_length_XZ_l81_81850

noncomputable def circle_radius_from_circumference (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

theorem segment_length_XZ (C : ℝ) (angle_TXZ : ℝ) (r : ℝ) (XZ : ℝ) :
  C = 18 * Real.pi → angle_TXZ = Real.pi / 6 →
  r = circle_radius_from_circumference C →
  XZ = r * Real.sqrt (2 - Real.sqrt 3) :=
by
  intros hC hAngle hr
  sorry

-- Given a circle T with circumference 18π, angle TXZ = 30 degrees (π/6 radians),
-- we need to show the length of segment XZ is 9√(2 - √3) inches.

end segment_length_XZ_l81_81850


namespace log_27_3_eq_one_third_l81_81557

theorem log_27_3_eq_one_third :
  log 27 3 = 1 / 3 :=
by
  -- Given conditions
  have h1 : 27 = 3 ^ 3 := by norm_num
  -- Using logarithmic identity and the conditions
  have h2 : (27 : ℝ) ^ (1 / 3 : ℝ) = 3 := by
    rw [h1, ←rpow_mul, div_mul_cancel 1 3]
    norm_num
  sorry

end log_27_3_eq_one_third_l81_81557


namespace customers_tried_sample_l81_81082

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l81_81082


namespace solve_fraction_l81_81625

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : x / y + y / x = 8)

theorem solve_fraction : (x + y) / (x - y) = Real.sqrt (5 / 3) :=
by
  sorry

end solve_fraction_l81_81625


namespace solve_equation_l81_81223

theorem solve_equation : ∃ x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := 
by
  -- Introducing x as a real number and stating the goal
  use 4
  -- Show that 2 * 4 - 3 = 5
  simp
  -- Adding the sorry to skip the proof step
  sorry

end solve_equation_l81_81223


namespace minimum_erasures_correct_l81_81458

open Nat List

-- define a function that checks if a number represented as a list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- the given problem statement
def given_number := [1, 2, 3, 2, 3, 3, 1, 4]

-- function to find the minimum erasures to make a list a palindrome
noncomputable def min_erasures_to_palindrome (l : List ℕ) : ℕ :=
  sorry -- function implementation skipped

-- the main theorem statement
theorem minimum_erasures_correct : min_erasures_to_palindrome given_number = 3 :=
  sorry

end minimum_erasures_correct_l81_81458


namespace foci_distance_l81_81470

def hyperbola (x y : ℝ) := x * y = 4

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem foci_distance :
  distance (2, 2) (-2, -2) = 4 * real.sqrt 2 :=
by
  sorry

end foci_distance_l81_81470


namespace actual_price_of_good_l81_81514

theorem actual_price_of_good (P : ℝ) (h : 0.684 * P = 6600) : P = 9649.12 :=
sorry

end actual_price_of_good_l81_81514


namespace find_f_of_2_l81_81882

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x) = (1 + x) / x) : f 2 = 3 :=
sorry

end find_f_of_2_l81_81882


namespace infinite_geometric_series_sum_l81_81564

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 2
  let r := (1 : ℝ) / 2
  (a + a * r + a * r^2 + a * r^3 + ∑' n : ℕ, a * r^n) = 1 :=
by
  sorry

end infinite_geometric_series_sum_l81_81564


namespace cannot_be_square_difference_l81_81678

def square_difference_formula (a b : ℝ) : ℝ := a^2 - b^2

def expression_A (x : ℝ) : ℝ := (x + 1) * (x - 1)
def expression_B (x : ℝ) : ℝ := (-x + 1) * (-x - 1)
def expression_C (x : ℝ) : ℝ := (x + 1) * (-x + 1)
def expression_D (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference (x : ℝ) : 
  ¬ (∃ a b, (x + 1) * (1 + x) = square_difference_formula a b) := 
sorry

end cannot_be_square_difference_l81_81678


namespace tangent_line_l81_81575

open Real

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 + x + 0.5

-- Define the point of tangency
def point : ℝ × ℝ := (0, 0.5)

-- Define the derivative of the curve
def curve_derivative := deriv (λ x : ℝ, x^2 + x + 0.5)

-- Statement to prove
theorem tangent_line :
  ∃ m b, m = (deriv curve 0) ∧ b = (0.5 - m * 0) ∧ (λ x, m * x + b) = (λ x, x + 0.5) :=
by
  sorry

end tangent_line_l81_81575


namespace width_of_rectangular_field_l81_81354

theorem width_of_rectangular_field
  (L W : ℝ)
  (h1 : L = (7/5) * W)
  (h2 : 2 * L + 2 * W = 384) :
  W = 80 :=
by
  sorry

end width_of_rectangular_field_l81_81354


namespace tan_diff_l81_81399

variables {α β : ℝ}

theorem tan_diff (h1 : Real.tan α = -3/4) (h2 : Real.tan (Real.pi - β) = 1/2) :
  Real.tan (α - β) = -2/11 :=
by
  sorry

end tan_diff_l81_81399


namespace pet_store_dogs_l81_81217

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l81_81217


namespace max_consecutive_sum_leq_500_l81_81489

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l81_81489


namespace monotonic_intervals_l81_81266

noncomputable def y : ℝ → ℝ := λ x => x * Real.log x

theorem monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < (1 / Real.exp 1) → y x < -1) ∧ 
  (∀ x : ℝ, (1 / Real.exp 1) < x → x < 5 → y x > 1) := 
by
  sorry -- Proof goes here.

end monotonic_intervals_l81_81266


namespace sample_size_calculation_l81_81242

theorem sample_size_calculation : 
  ∀ (high_school_students junior_high_school_students sampled_high_school_students n : ℕ), 
  high_school_students = 3500 →
  junior_high_school_students = 1500 →
  sampled_high_school_students = 70 →
  n = (3500 + 1500) * 70 / 3500 →
  n = 100 :=
by
  intros high_school_students junior_high_school_students sampled_high_school_students n
  intros h1 h2 h3 h4
  sorry

end sample_size_calculation_l81_81242


namespace rectangular_solid_sum_of_edges_l81_81661

noncomputable def sum_of_edges (x y z : ℝ) := 4 * (x + y + z)

theorem rectangular_solid_sum_of_edges :
  ∃ (x y z : ℝ), (x * y * z = 512) ∧ (2 * (x * y + y * z + z * x) = 384) ∧
  (∃ (r a : ℝ), x = a / r ∧ y = a ∧ z = a * r) ∧ sum_of_edges x y z = 96 :=
by
  sorry

end rectangular_solid_sum_of_edges_l81_81661


namespace selling_price_of_mixture_per_litre_l81_81930

def cost_per_litre : ℝ := 3.60
def litres_of_pure_milk : ℝ := 25
def litres_of_water : ℝ := 5
def total_volume_of_mixture : ℝ := litres_of_pure_milk + litres_of_water
def total_cost_of_pure_milk : ℝ := cost_per_litre * litres_of_pure_milk

theorem selling_price_of_mixture_per_litre :
  total_cost_of_pure_milk / total_volume_of_mixture = 3 := by
  sorry

end selling_price_of_mixture_per_litre_l81_81930


namespace find_common_ratio_l81_81916

variable (a₃ a₂ : ℝ)
variable (S₁ S₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * S₂ = a₃ - 2
def condition2 : Prop := 3 * S₁ = a₂ - 2

-- Theorem statement
theorem find_common_ratio (h1 : condition1 a₃ S₂)
                          (h2 : condition2 a₂ S₁) : 
                          (a₃ / a₂ = 4) :=
by 
  sorry

end find_common_ratio_l81_81916


namespace instantaneous_velocity_at_1_2_l81_81597

def equation_of_motion (t : ℝ) : ℝ := 2 * (1 - t^2)

def velocity_function (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 :
  velocity_function 1.2 = -4.8 :=
by sorry

end instantaneous_velocity_at_1_2_l81_81597


namespace exponent_division_l81_81742

theorem exponent_division (m n : ℕ) (h : m - n = 1) : 5 ^ m / 5 ^ n = 5 :=
by {
  sorry
}

end exponent_division_l81_81742


namespace circumcircle_radius_proof_l81_81765

noncomputable def circumcircle_radius (AB A S : ℝ) : ℝ :=
  if AB = 3 ∧ A = 120 ∧ S = 9 * Real.sqrt 3 / 4 then 3 else 0

theorem circumcircle_radius_proof :
  circumcircle_radius 3 120 (9 * Real.sqrt 3 / 4) = 3 := by
  sorry

end circumcircle_radius_proof_l81_81765


namespace remainder_division_l81_81073

theorem remainder_division :
  ∃ N R1 Q2, N = 44 * 432 + R1 ∧ N = 30 * Q2 + 18 ∧ R1 < 44 ∧ 18 = R1 :=
by
  sorry

end remainder_division_l81_81073


namespace inequality_relationship_l81_81408

variable (a b : ℝ)

theorem inequality_relationship
  (h1 : a < 0)
  (h2 : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end inequality_relationship_l81_81408


namespace cannot_be_square_difference_l81_81679

def square_difference_formula (a b : ℝ) : ℝ := a^2 - b^2

def expression_A (x : ℝ) : ℝ := (x + 1) * (x - 1)
def expression_B (x : ℝ) : ℝ := (-x + 1) * (-x - 1)
def expression_C (x : ℝ) : ℝ := (x + 1) * (-x + 1)
def expression_D (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference (x : ℝ) : 
  ¬ (∃ a b, (x + 1) * (1 + x) = square_difference_formula a b) := 
sorry

end cannot_be_square_difference_l81_81679


namespace angle_division_l81_81876

theorem angle_division (α : ℝ) (n : ℕ) (θ : ℝ) (h : α = 78) (hn : n = 26) (ht : θ = 3) :
  α / n = θ :=
by
  sorry

end angle_division_l81_81876


namespace ivan_scores_more_than_5_points_l81_81603

-- Definitions based on problem conditions
def typeA_problem_probability (correct_guesses : ℕ) (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  (Nat.choose total_tasks correct_guesses : ℚ) * (success_prob ^ correct_guesses) * (failure_prob ^ (total_tasks - correct_guesses))

def probability_A4 (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  ∑ i in Finset.range (total_tasks + 1), if i ≥ 4 then typeA_problem_probability i total_tasks success_prob failure_prob else 0

def probability_A6 (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  ∑ i in Finset.range (total_tasks + 1), if i ≥ 6 then typeA_problem_probability i total_tasks success_prob failure_prob else 0

def final_probability (p_A4 : ℚ) (p_A6 : ℚ) (p_B : ℚ) : ℚ :=
  (p_A4 * p_B) + (p_A6 * (1 - p_B))

noncomputable def probability_ivan_scores_more_than_5 : ℚ :=
  let total_tasks := 10
  let success_prob := 1 / 4
  let failure_prob := 3 / 4
  let p_B := 1 / 3
  let p_A4 := probability_A4 total_tasks success_prob failure_prob
  let p_A6 := probability_A6 total_tasks success_prob failure_prob
  final_probability p_A4 p_A6 p_B

theorem ivan_scores_more_than_5_points : probability_ivan_scores_more_than_5 = 0.088 := 
  sorry

end ivan_scores_more_than_5_points_l81_81603


namespace probability_of_rain_l81_81473

theorem probability_of_rain {p : ℝ} (h : p = 0.95) :
  ∃ (q : ℝ), q = (1 - p) ∧ q < p :=
by
  sorry

end probability_of_rain_l81_81473


namespace domain_ln_x_minus_x_sq_l81_81934

noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

theorem domain_ln_x_minus_x_sq : { x : ℝ | x - x^2 > 0 } = { x : ℝ | 0 < x ∧ x < 1 } :=
by {
  -- These are placeholders for conditions needed in the proof
  sorry
}

end domain_ln_x_minus_x_sq_l81_81934


namespace math_problem_l81_81699

theorem math_problem : 
  ∀ n : ℕ, 
  n = 5 * 96 → 
  ((n + 17) * 69) = 34293 := 
by
  intros n h
  sorry

end math_problem_l81_81699


namespace remainder_1534_base12_div_by_9_l81_81828

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem remainder_1534_base12_div_by_9 :
  (base12_to_base10 1534) % 9 = 4 :=
by
  sorry

end remainder_1534_base12_div_by_9_l81_81828


namespace probability_two_consecutive_pairs_of_four_dice_correct_l81_81871

open Classical

noncomputable def probability_two_consecutive_pairs_of_four_dice : ℚ :=
  let total_outcomes := 6^4
  let favorable_outcomes := 48
  favorable_outcomes / total_outcomes

theorem probability_two_consecutive_pairs_of_four_dice_correct :
  probability_two_consecutive_pairs_of_four_dice = 1 / 27 := 
by
  sorry

end probability_two_consecutive_pairs_of_four_dice_correct_l81_81871


namespace least_number_divisible_by_five_primes_l81_81037

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l81_81037


namespace fence_remaining_l81_81816

noncomputable def totalFence : Float := 150.0
noncomputable def ben_whitewashed : Float := 20.0

-- Remaining fence after Ben's contribution
noncomputable def remaining_after_ben : Float := totalFence - ben_whitewashed

noncomputable def billy_fraction : Float := 1.0 / 5.0
noncomputable def billy_whitewashed : Float := billy_fraction * remaining_after_ben

-- Remaining fence after Billy's contribution
noncomputable def remaining_after_billy : Float := remaining_after_ben - billy_whitewashed

noncomputable def johnny_fraction : Float := 1.0 / 3.0
noncomputable def johnny_whitewashed : Float := johnny_fraction * remaining_after_billy

-- Remaining fence after Johnny's contribution
noncomputable def remaining_after_johnny : Float := remaining_after_billy - johnny_whitewashed

noncomputable def timmy_percentage : Float := 15.0 / 100.0
noncomputable def timmy_whitewashed : Float := timmy_percentage * remaining_after_johnny

-- Remaining fence after Timmy's contribution
noncomputable def remaining_after_timmy : Float := remaining_after_johnny - timmy_whitewashed

noncomputable def alice_fraction : Float := 1.0 / 8.0
noncomputable def alice_whitewashed : Float := alice_fraction * remaining_after_timmy

-- Remaining fence after Alice's contribution
noncomputable def remaining_fence : Float := remaining_after_timmy - alice_whitewashed

theorem fence_remaining : remaining_fence = 51.56 :=
by
    -- Placeholder for actual proof
    sorry

end fence_remaining_l81_81816


namespace company_food_purchase_1_l81_81964

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end company_food_purchase_1_l81_81964


namespace total_cost_is_100_l81_81616

def shirts : ℕ := 10
def pants : ℕ := shirts / 2
def cost_shirt : ℕ := 6
def cost_pant : ℕ := 8

theorem total_cost_is_100 :
  shirts * cost_shirt + pants * cost_pant = 100 := by
  sorry

end total_cost_is_100_l81_81616


namespace wine_cost_is_3_60_l81_81983

noncomputable def appetizer_cost : ℕ := 8
noncomputable def steak_cost : ℕ := 20
noncomputable def dessert_cost : ℕ := 6
noncomputable def total_spent : ℝ := 38
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def number_of_wines : ℕ := 2

noncomputable def discounted_steak_cost : ℝ := steak_cost / 2
noncomputable def full_meal_cost : ℝ := appetizer_cost + steak_cost + dessert_cost
noncomputable def meal_cost_after_discount : ℝ := appetizer_cost + discounted_steak_cost + dessert_cost
noncomputable def full_meal_tip := tip_percentage * full_meal_cost
noncomputable def meal_cost_with_tip := meal_cost_after_discount + full_meal_tip
noncomputable def total_wine_cost := total_spent - meal_cost_with_tip
noncomputable def cost_per_wine := total_wine_cost / number_of_wines

theorem wine_cost_is_3_60 : cost_per_wine = 3.60 := by
  sorry

end wine_cost_is_3_60_l81_81983


namespace parallel_lines_a_eq_3_div_2_l81_81586

theorem parallel_lines_a_eq_3_div_2 (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by sorry

end parallel_lines_a_eq_3_div_2_l81_81586


namespace least_positive_whole_number_divisible_by_five_primes_l81_81043

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l81_81043


namespace principal_amount_l81_81020

theorem principal_amount (P : ℝ) (CI SI : ℝ) 
  (H1 : CI = P * 0.44) 
  (H2 : SI = P * 0.4) 
  (H3 : CI - SI = 216) : 
  P = 5400 :=
by {
  sorry
}

end principal_amount_l81_81020


namespace factor_polynomial_sum_l81_81594

theorem factor_polynomial_sum (P Q : ℤ) :
  (∀ x : ℂ, (x^2 + 4*x + 5) ∣ (x^4 + P*x^2 + Q)) → P + Q = 19 :=
by
  intro h
  sorry

end factor_polynomial_sum_l81_81594


namespace correct_operation_l81_81058

variable (a b : ℝ)

theorem correct_operation : 2 * (a - 1) = 2 * a - 2 :=
sorry

end correct_operation_l81_81058


namespace base_angle_isosceles_triangle_l81_81901

theorem base_angle_isosceles_triangle
  (sum_angles : ∀ (α β γ : ℝ), α + β + γ = 180)
  (isosceles : ∀ (α β : ℝ), α = β)
  (one_angle_forty : ∃ α : ℝ, α = 40) :
  ∃ β : ℝ, β = 70 ∨ β = 40 :=
by
  sorry

end base_angle_isosceles_triangle_l81_81901


namespace bears_on_each_shelf_l81_81084

theorem bears_on_each_shelf 
    (initial_bears : ℕ) (shipment_bears : ℕ) (shelves : ℕ)
    (h1 : initial_bears = 4) (h2 : shipment_bears = 10) (h3 : shelves = 2) :
    (initial_bears + shipment_bears) / shelves = 7 := by
  sorry

end bears_on_each_shelf_l81_81084


namespace car_average_speed_l81_81957

theorem car_average_speed :
  let distance_uphill := 100
  let distance_downhill := 50
  let speed_uphill := 30
  let speed_downhill := 80
  let total_distance := distance_uphill + distance_downhill
  let time_uphill := distance_uphill / speed_uphill
  let time_downhill := distance_downhill / speed_downhill
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 37.92 := by
  sorry

end car_average_speed_l81_81957


namespace solve_trig_eq_l81_81270

noncomputable def arccos (x : ℝ) : ℝ := sorry

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  -3 * (Real.cos x) ^ 2 + 5 * (Real.sin x) + 1 = 0 ↔
  (x = Real.arcsin (1 / 3) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (1 / 3) + 2 * k * Real.pi) :=
sorry

end solve_trig_eq_l81_81270


namespace height_percentage_difference_l81_81595

theorem height_percentage_difference (H : ℝ) (p r q : ℝ) 
  (hp : p = 0.60 * H) 
  (hr : r = 1.30 * H) : 
  (r - p) / p * 100 = 116.67 :=
by
  sorry

end height_percentage_difference_l81_81595


namespace part1_part2_l81_81447

def f (x a : ℝ) : ℝ := |x + a| + |x - a^2|

theorem part1 (x : ℝ) : f x 1 ≥ 4 ↔ x ≤ -2 ∨ x ≥ 2 := sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ a : ℝ, -1 < a ∧ a < 3 ∧ m < f x a) ↔ m < 12 := sorry

end part1_part2_l81_81447


namespace minimum_value_expression_l81_81446

open Real

theorem minimum_value_expression : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 := 
sorry

end minimum_value_expression_l81_81446


namespace problem_statement_l81_81096

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l81_81096


namespace find_Y_value_l81_81760

-- Define the conditions
def P : ℕ := 4020 / 4
def Q : ℕ := P * 2
def Y : ℤ := P - Q

-- State the theorem
theorem find_Y_value : Y = -1005 := by
  -- Proof goes here
  sorry

end find_Y_value_l81_81760


namespace fold_string_twice_l81_81888

theorem fold_string_twice (initial_length : ℕ) (half_folds : ℕ) (result_length : ℕ) 
  (h1 : initial_length = 12)
  (h2 : half_folds = 2)
  (h3 : result_length = initial_length / (2 ^ half_folds)) :
  result_length = 3 := 
by
  -- This is where the proof would go
  sorry

end fold_string_twice_l81_81888


namespace calculate_expression_l81_81106

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l81_81106


namespace eval_expression_l81_81565

theorem eval_expression : (256 : ℝ) ^ ((-2 : ℝ) ^ (-3 : ℝ)) = 1 / 2 := by
  sorry

end eval_expression_l81_81565


namespace probability_Ivan_more_than_5_points_l81_81613

noncomputable def prob_type_A_correct := 1 / 4
noncomputable def total_type_A := 10
noncomputable def prob_type_B_correct := 1 / 3

def binomial (n k : ℕ) : ℚ :=
  (Nat.choose n k) * (prob_type_A_correct ^ k) * ((1 - prob_type_A_correct) ^ (n - k))

def prob_A_4 := ∑ k in finset.range (total_type_A + 1), if k ≥ 4 then binomial total_type_A k else 0
def prob_A_6 := ∑ k in finset.range (total_type_A + 1), if k ≥ 6 then binomial total_type_A k else 0

def prob_B := prob_type_B_correct
def prob_not_B := 1 - prob_type_B_correct

noncomputable def prob_more_than_5_points :=
  prob_A_4 * prob_B + prob_A_6 * prob_not_B

theorem probability_Ivan_more_than_5_points :
  prob_more_than_5_points = 0.088 := by
  sorry

end probability_Ivan_more_than_5_points_l81_81613


namespace log_domain_l81_81276

theorem log_domain (x : ℝ) : (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) :=
by
  sorry

end log_domain_l81_81276


namespace part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l81_81068

def cost_option1 (x : ℕ) : ℕ :=
  20 * x + 1200

def cost_option2 (x : ℕ) : ℕ :=
  18 * x + 1440

theorem part1_option1_payment (x : ℕ) (h : x > 20) : cost_option1 x = 20 * x + 1200 :=
  by sorry

theorem part1_option2_payment (x : ℕ) (h : x > 20) : cost_option2 x = 18 * x + 1440 :=
  by sorry

theorem part2_cost_effective (x : ℕ) (h : x = 30) : cost_option1 x < cost_option2 x :=
  by sorry

theorem part3_more_cost_effective (x : ℕ) (h : x = 30) : 20 * 80 + 20 * 10 * 9 / 10 = 1780 :=
  by sorry

end part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l81_81068


namespace divisible_by_1995_l81_81918

theorem divisible_by_1995 (n : ℕ) : 
  1995 ∣ (256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n)) := 
sorry

end divisible_by_1995_l81_81918


namespace ivan_prob_more_than_5_points_l81_81609

open ProbabilityTheory Finset

/-- Conditions -/
def prob_correct_A : ℝ := 1 / 4
def prob_correct_B : ℝ := 1 / 3
def prob_A (k : ℕ) : ℝ := 
(C(10, k) * (prob_correct_A ^ k) * ((1 - prob_correct_A) ^ (10 - k)))

/-- Probabilities for type A problems -/
def prob_A_4 := ∑ i in (range 7).filter (λ i, i ≥ 4), prob_A i
def prob_A_6 := ∑ i in (range 7).filter (λ i, i ≥ 6), prob_A i

/-- Final combined probability -/
def final_prob : ℝ := 
(prob_A_4 * prob_correct_B) + (prob_A_6 * (1 - prob_correct_B))

/-- Proof -/
theorem ivan_prob_more_than_5_points : 
  final_prob = 0.088 := by
    sorry

end ivan_prob_more_than_5_points_l81_81609


namespace players_scores_l81_81599

/-- Lean code to verify the scores of three players in a guessing game -/
theorem players_scores (H F S : ℕ) (h1 : H = 42) (h2 : F - H = 24) (h3 : S - F = 18) (h4 : H < F) (h5 : H < S) : 
  F = 66 ∧ S = 84 :=
by
  sorry

end players_scores_l81_81599


namespace tickets_used_correct_l81_81709

def ferris_wheel_rides : ℕ := 7
def bumper_car_rides : ℕ := 3
def cost_per_ride : ℕ := 5

def total_rides : ℕ := ferris_wheel_rides + bumper_car_rides
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_correct : total_tickets_used = 50 := by
  sorry

end tickets_used_correct_l81_81709


namespace determine_n_for_11111_base_n_is_perfect_square_l81_81719

theorem determine_n_for_11111_base_n_is_perfect_square:
  ∃ m : ℤ, m^2 = 3^4 + 3^3 + 3^2 + 3 + 1 :=
by
  sorry

end determine_n_for_11111_base_n_is_perfect_square_l81_81719


namespace abs_eq_5_iff_l81_81295

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l81_81295


namespace abs_diff_eq_10_l81_81658

variable {x y : ℝ}

-- Given conditions as definitions.
def condition1 : Prop := x + y = 30
def condition2 : Prop := x * y = 200

-- The theorem statement to prove the given question equals the correct answer.
theorem abs_diff_eq_10 (h1 : condition1) (h2 : condition2) : |x - y| = 10 :=
by
  sorry

end abs_diff_eq_10_l81_81658


namespace base_angle_of_isosceles_triangle_l81_81846

-- Definitions corresponding to the conditions
def isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a = b ∧ A + B + C = 180) ∧ A = 40 -- Isosceles and sum of angles is 180° with apex angle A = 40°

-- The theorem to be proven
theorem base_angle_of_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  isosceles_triangle a b c A B C → B = 70 :=
by
  intros h
  sorry

end base_angle_of_isosceles_triangle_l81_81846


namespace abs_eq_5_iff_l81_81296

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l81_81296


namespace odd_function_def_l81_81749

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * (x - 1)
else -x * (x + 1)

theorem odd_function_def {x : ℝ} (h : x > 0) :
  f x = -x * (x + 1) :=
by
  sorry

end odd_function_def_l81_81749


namespace horner_method_poly_at_neg2_l81_81823

-- Define the polynomial using the given conditions and Horner's method transformation
def polynomial : ℤ → ℤ := fun x => (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

-- State the theorem
theorem horner_method_poly_at_neg2 : polynomial (-2) = -40 := by
  sorry

end horner_method_poly_at_neg2_l81_81823


namespace onion_harvest_scientific_notation_l81_81664

theorem onion_harvest_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 325000000 = a * 10^n ∧ a = 3.25 ∧ n = 8 := 
by
  sorry

end onion_harvest_scientific_notation_l81_81664


namespace length_of_bridge_l81_81253

/-- A train that is 357 meters long is running at a speed of 42 km/hour. 
    It takes 42.34285714285714 seconds to pass a bridge. 
    Prove that the length of the bridge is 136.7142857142857 meters. -/
theorem length_of_bridge : 
  let train_length := 357 -- meters
  let speed_kmh := 42 -- km/hour
  let passing_time := 42.34285714285714 -- seconds
  let speed_mps := 42 * (1000 / 3600) -- meters/second
  let total_distance := speed_mps * passing_time -- meters
  let bridge_length := total_distance - train_length -- meters
  bridge_length = 136.7142857142857 :=
by
  sorry

end length_of_bridge_l81_81253


namespace undefined_sum_slope_y_intercept_of_vertical_line_l81_81177

theorem undefined_sum_slope_y_intercept_of_vertical_line :
  ∀ (C D : ℝ × ℝ), C.1 = 8 → D.1 = 8 → C.2 ≠ D.2 →
  ∃ (m b : ℝ), false :=
by
  intros
  sorry

end undefined_sum_slope_y_intercept_of_vertical_line_l81_81177


namespace xy_range_l81_81623

theorem xy_range (x y : ℝ) (h1 : y = 3 * (⌊x⌋) + 2) (h2 : y = 4 * (⌊x - 3⌋) + 6) (h3 : (⌊x⌋ : ℝ) ≠ x) :
  34 < x + y ∧ x + y < 35 := 
by 
  sorry

end xy_range_l81_81623


namespace five_times_x_plus_four_l81_81154

theorem five_times_x_plus_four (x : ℝ) (h : 4 * x - 3 = 13 * x + 12) : 5 * (x + 4) = 35 / 3 := 
by
  sorry

end five_times_x_plus_four_l81_81154


namespace fuel_at_40_min_fuel_l81_81705

section FuelConsumption

noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

theorem fuel_at_40 : total_fuel 40 = 17.5 :=
by sorry

theorem min_fuel : total_fuel 80 = 11.25 ∧ ∀ x, (0 < x ∧ x ≤ 120) → total_fuel x ≥ total_fuel 80 :=
by sorry

end FuelConsumption

end fuel_at_40_min_fuel_l81_81705


namespace a_minus_b_eq_one_l81_81746

variable (a b : ℕ)

theorem a_minus_b_eq_one
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.sqrt 18 = a * Real.sqrt 2) 
  (h4 : Real.sqrt 8 = 2 * Real.sqrt b) : 
  a - b = 1 := 
sorry

end a_minus_b_eq_one_l81_81746


namespace intersection_distance_l81_81163

noncomputable def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
  (2 + 2 * Real.cos α, 2 * Real.sin α)

noncomputable def polar_eq_C2 (θ : ℝ) : ℝ :=
  4 * Real.cos θ / (1 - Real.cos (2 * θ))

theorem intersection_distance (θ : ℝ) (h1 : -Real.pi / 2 ≤ θ) (h2 : θ ≤ Real.pi / 2) (h3 : θ ≠ 0) :
  let ρ := 4 * Real.cos θ
  in ρ = polar_eq_C2 θ → |ρ| = 2 * Real.sqrt 2 :=
by sorry

end intersection_distance_l81_81163


namespace root_in_interval_l81_81650

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 / x

variable (h_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
variable (h_f_half : f (1 / 2) < 0)
variable (h_f_one : f 1 < 0)
variable (h_f_three_half : f (3 / 2) < 0)
variable (h_f_two : f 2 > 0)

theorem root_in_interval : ∃ c : ℝ, c ∈ Set.Ioo (3 / 2) 2 ∧ f c = 0 :=
sorry

end root_in_interval_l81_81650


namespace find_number_of_tails_l81_81522

-- Definitions based on conditions
variables (T H : ℕ)
axiom total_coins : T + H = 1250
axiom heads_more_than_tails : H = T + 124

-- The goal is to prove T = 563
theorem find_number_of_tails : T = 563 :=
sorry

end find_number_of_tails_l81_81522


namespace probability_of_each_suit_in_five_draws_with_replacement_l81_81306

theorem probability_of_each_suit_in_five_draws_with_replacement :
  let deck_size := 52
  let num_cards := 5
  let num_suits := 4
  let prob_each_suit := 1/4
  let target_probability := 9/16
  prob_each_suit * (3/4) * (1/2) * (1/4) * 24 = target_probability :=
by sorry

end probability_of_each_suit_in_five_draws_with_replacement_l81_81306


namespace sqrt_sum_eq_nine_point_six_l81_81792

variable (y : ℝ)

/- Given conditions -/
def condition (y : ℝ) : Prop :=
  sqrt (64 - y^2) - sqrt (16 - y^2) = 5

/- Statement to prove -/
theorem sqrt_sum_eq_nine_point_six (hy : condition y) : 
  sqrt (64 - y^2) + sqrt (16 - y^2) = 9.6 :=
sorry

end sqrt_sum_eq_nine_point_six_l81_81792


namespace radius_formula_l81_81653

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  let angle := 42 * Real.pi / 180 -- converting 42 degrees to radians
  let R := a / (Real.sqrt 3)
  let h := R * Real.tan angle
  Real.sqrt ((R * R) + (h * h))

theorem radius_formula (a : ℝ) : radius_of_circumscribed_sphere a = (a * Real.sqrt 3) / 3 :=
by
  sorry

end radius_formula_l81_81653


namespace best_representation_is_B_l81_81334

-- Define the conditions
structure Trip :=
  (home_to_diner : ℝ)
  (diner_stop : ℝ)
  (diner_to_highway : ℝ)
  (highway_to_mall : ℝ)
  (mall_stop : ℝ)
  (highway_return : ℝ)
  (construction_zone : ℝ)
  (return_city_traffic : ℝ)

-- Graph description
inductive Graph
| plateau : Graph
| increasing : Graph → Graph
| decreasing : Graph → Graph

-- Condition that describes the pattern of the graph
def correct_graph (trip : Trip) : Prop :=
  let d1 := trip.home_to_diner
  let d2 := trip.diner_stop
  let d3 := trip.diner_to_highway
  let d4 := trip.highway_to_mall
  let d5 := trip.mall_stop
  let d6 := trip.highway_return
  let d7 := trip.construction_zone
  let d8 := trip.return_city_traffic
  d1 > 0 ∧ d2 = 0 ∧ d3 > 0 ∧ d4 > 0 ∧ d5 = 0 ∧ d6 < 0 ∧ d7 < 0 ∧ d8 < 0

-- Theorem statement
theorem best_representation_is_B (trip : Trip) : correct_graph trip :=
by sorry

end best_representation_is_B_l81_81334


namespace fixed_point_l81_81409

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  2 + a^(1-1) = 3 :=
by
  sorry

end fixed_point_l81_81409


namespace james_total_cost_is_100_l81_81619

def cost_of_shirts (number_of_shirts : Nat) (cost_per_shirt : Nat) : Nat :=
  number_of_shirts * cost_per_shirt

def cost_of_pants (number_of_pants : Nat) (cost_per_pants : Nat) : Nat :=
  number_of_pants * cost_per_pants

def total_cost (number_of_shirts : Nat) (number_of_pants : Nat) (cost_per_shirt : Nat) (cost_per_pants : Nat) : Nat :=
  cost_of_shirts number_of_shirts cost_per_shirt + cost_of_pants number_of_pants cost_per_pants

theorem james_total_cost_is_100 : 
  total_cost 10 (10 / 2) 6 8 = 100 :=
by
  sorry

end james_total_cost_is_100_l81_81619


namespace pencils_brought_l81_81708

-- Given conditions
variables (A B : ℕ)

-- There are 7 people in total
def total_people : Prop := A + B = 7

-- 11 charts in total
def total_charts : Prop := A + 2 * B = 11

-- Question: Total pencils
def total_pencils : ℕ := 2 * A + B

-- Statement to be proved
theorem pencils_brought
  (h1 : total_people A B)
  (h2 : total_charts A B) :
  total_pencils A B = 10 := by
  sorry

end pencils_brought_l81_81708


namespace money_distribution_l81_81235

theorem money_distribution (a : ℕ) (h1 : 5 * a = 1500) : 7 * a - 3 * a = 1200 := by
  sorry

end money_distribution_l81_81235


namespace right_obtuse_triangle_impossible_l81_81059

def triangle_interior_angles_sum (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def is_right_angle (α : ℝ) : Prop :=
  α = 90

def is_obtuse_angle (α : ℝ) : Prop :=
  α > 90

theorem right_obtuse_triangle_impossible (α β γ : ℝ) (h1 : triangle_interior_angles_sum α β γ) (h2 : is_right_angle α) (h3 : is_obtuse_angle β) : false :=
  sorry

end right_obtuse_triangle_impossible_l81_81059


namespace clock_displays_unique_digits_minutes_l81_81195

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l81_81195


namespace joyce_apples_l81_81442

/-- Joyce starts with some apples. She gives 52 apples to Larry and ends up with 23 apples. 
    Prove that Joyce initially had 75 apples. -/
theorem joyce_apples (initial_apples given_apples final_apples : ℕ) 
  (h1 : given_apples = 52) 
  (h2 : final_apples = 23) 
  (h3 : initial_apples = given_apples + final_apples) : 
  initial_apples = 75 := 
by 
  sorry

end joyce_apples_l81_81442


namespace mitchell_total_pages_read_l81_81344

def pages_per_chapter : ℕ := 40
def chapters_read_before : ℕ := 10
def pages_read_11th_before : ℕ := 20
def chapters_read_after : ℕ := 2

def total_pages_read := 
  pages_per_chapter * chapters_read_before + pages_read_11th_before + pages_per_chapter * chapters_read_after

theorem mitchell_total_pages_read : total_pages_read = 500 := by
  sorry

end mitchell_total_pages_read_l81_81344


namespace product_of_roots_l81_81395

theorem product_of_roots :
  ∀ a b c : ℚ, (a ≠ 0) → a = 24 → b = 60 → c = -600 → (c / a) = -25 :=
sorry

end product_of_roots_l81_81395


namespace bookstore_earnings_difference_l81_81088

def base_price_TOP := 8.0
def base_price_ABC := 23.0
def discount_TOP := 0.10
def discount_ABC := 0.05
def sales_tax := 0.07
def num_TOP_sold := 13
def num_ABC_sold := 4

def discounted_price (base_price discount : Float) : Float :=
  base_price * (1.0 - discount)

def final_price (discounted_price tax : Float) : Float :=
  discounted_price * (1.0 + tax)

def total_earnings (final_price : Float) (quantity : Nat) : Float :=
  final_price * (quantity.toFloat)

theorem bookstore_earnings_difference :
  let discounted_price_TOP := discounted_price base_price_TOP discount_TOP
  let discounted_price_ABC := discounted_price base_price_ABC discount_ABC
  let final_price_TOP := final_price discounted_price_TOP sales_tax
  let final_price_ABC := final_price discounted_price_ABC sales_tax
  let total_earnings_TOP := total_earnings final_price_TOP num_TOP_sold
  let total_earnings_ABC := total_earnings final_price_ABC num_ABC_sold
  total_earnings_TOP - total_earnings_ABC = 6.634 :=
by
  sorry

end bookstore_earnings_difference_l81_81088


namespace cos_neg_300_eq_positive_half_l81_81837

theorem cos_neg_300_eq_positive_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_neg_300_eq_positive_half_l81_81837


namespace sum_final_numbers_l81_81227

theorem sum_final_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_final_numbers_l81_81227


namespace calculate_expression_l81_81107

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l81_81107


namespace vacuum_cleaner_cost_l81_81109

-- Define initial amount collected
def initial_amount : ℕ := 20

-- Define amount added each week
def weekly_addition : ℕ := 10

-- Define number of weeks
def number_of_weeks : ℕ := 10

-- Define the total amount after 10 weeks
def total_amount : ℕ := initial_amount + (weekly_addition * number_of_weeks)

-- Prove that the total amount is equal to the cost of the vacuum cleaner
theorem vacuum_cleaner_cost : total_amount = 120 := by
  sorry

end vacuum_cleaner_cost_l81_81109


namespace quadratic_condition_not_necessary_and_sufficient_l81_81137

theorem quadratic_condition_not_necessary_and_sufficient (a b c : ℝ) :
  ¬((∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (b^2 - 4 * a * c < 0)) :=
sorry

end quadratic_condition_not_necessary_and_sufficient_l81_81137


namespace find_BD_l81_81165

theorem find_BD 
  (A B C D : Type)
  (AC BC : ℝ) (h₁ : AC = 10) (h₂ : BC = 10)
  (AD CD : ℝ) (h₃ : AD = 12) (h₄ : CD = 5) :
  ∃ (BD : ℝ), BD = 152 / 24 := 
sorry

end find_BD_l81_81165


namespace select_k_numbers_l81_81853

theorem select_k_numbers (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, 0 < a n) 
  (h2 : ∀ n m, n < m → a n ≥ a m) (h3 : a 1 = 1 / (2 * k)) 
  (h4 : ∑' n, a n = 1) :
  ∃ (f : ℕ → ℕ) (hf : ∀ i j, i ≠ j → f i ≠ f j), 
    (∀ i, i < k → a (f i) > 1/2 * a (f 0)) :=
by
  sorry

end select_k_numbers_l81_81853


namespace ratio_price_16_to_8_l81_81011

def price_8_inch := 5
def P : ℝ := sorry
def price_16_inch := 5 * P
def daily_earnings := 3 * price_8_inch + 5 * price_16_inch
def three_day_earnings := 3 * daily_earnings
def total_earnings := 195

theorem ratio_price_16_to_8 : total_earnings = three_day_earnings → P = 2 :=
by
  sorry

end ratio_price_16_to_8_l81_81011


namespace line_passes_through_point_l81_81581

-- We declare the variables for the real numbers a, b, and c
variables (a b c : ℝ)

-- We state the condition that a + b - c = 0
def condition1 : Prop := a + b - c = 0

-- We state the condition that not all of a, b, c are zero
def condition2 : Prop := ¬ (a = 0 ∧ b = 0 ∧ c = 0)

-- We state the theorem: the line ax + by + c = 0 passes through the point (-1, -1)
theorem line_passes_through_point (h1 : condition1 a b c) (h2 : condition2 a b c) :
  a * (-1) + b * (-1) + c = 0 := sorry

end line_passes_through_point_l81_81581


namespace Piper_gym_sessions_l81_81927

theorem Piper_gym_sessions
  (start_on_monday : Bool)
  (alternate_except_sunday : (∀ (n : ℕ), n % 2 = 1 → n % 7 ≠ 0 → Bool))
  (sessions_over_on_wednesday : Bool)
  : ∃ (n : ℕ), n = 5 :=
by 
  sorry

end Piper_gym_sessions_l81_81927


namespace tweets_when_hungry_l81_81456

theorem tweets_when_hungry (H : ℕ) : 
  (18 * 20) + (H * 20) + (45 * 20) = 1340 → H = 4 := by
  sorry

end tweets_when_hungry_l81_81456


namespace find_c_k_l81_81476

theorem find_c_k (a b : ℕ → ℕ) (c : ℕ → ℕ) (k : ℕ) (d r : ℕ) 
  (h1 : ∀ n, a n = 1 + (n-1)*d)
  (h2 : ∀ n, b n = r^(n-1))
  (h3 : ∀ n, c n = a n + b n)
  (h4 : c (k-1) = 80)
  (h5 : c (k+1) = 500) :
  c k = 167 := sorry

end find_c_k_l81_81476


namespace total_cost_is_100_l81_81617

def shirts : ℕ := 10
def pants : ℕ := shirts / 2
def cost_shirt : ℕ := 6
def cost_pant : ℕ := 8

theorem total_cost_is_100 :
  shirts * cost_shirt + pants * cost_pant = 100 := by
  sorry

end total_cost_is_100_l81_81617


namespace molecular_weight_calc_l81_81673

theorem molecular_weight_calc (total_weight : ℕ) (num_moles : ℕ) (one_mole_weight : ℕ) :
  total_weight = 1170 → num_moles = 5 → one_mole_weight = total_weight / num_moles → one_mole_weight = 234 :=
by
  intros h1 h2 h3
  sorry

end molecular_weight_calc_l81_81673


namespace race_result_l81_81124

-- Define the contestants
inductive Contestants
| Alyosha
| Borya
| Vanya
| Grisha

open Contestants

-- Define their statements
def Alyosha_statement (place : Contestants → ℕ) : Prop :=
  place Alyosha ≠ 1 ∧ place Alyosha ≠ 4

def Borya_statement (place : Contestants → ℕ) : Prop :=
  place Borya ≠ 4

def Vanya_statement (place : Contestants → ℕ) : Prop :=
  place Vanya = 1

def Grisha_statement (place : Contestants → ℕ) : Prop :=
  place Grisha = 4

-- Define that exactly one statement is false and the rest are true
def three_true_one_false (place : Contestants → ℕ) : Prop :=
  (Alyosha_statement place ∧ ¬ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (¬ Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ ¬ Borya_statement place ∧ Grisha_statement place) ∨
  (Alyosha_statement place ∧ Vanya_statement place ∧ Borya_statement place ∧ ¬ Grisha_statement place)

-- Define the conclusion: Vanya lied and Borya was first
theorem race_result (place : Contestants → ℕ) : 
  three_true_one_false place → 
  (¬ Vanya_statement place ∧ place Borya = 1) :=
sorry

end race_result_l81_81124


namespace find_tangent_equal_l81_81394

theorem find_tangent_equal (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180)) : n = 75 :=
sorry

end find_tangent_equal_l81_81394


namespace least_multiple_of_five_primes_l81_81047

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l81_81047


namespace inequality_of_pos_reals_l81_81329

open Real

theorem inequality_of_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤
  (1 / 4) * (a + b + c) :=
by
  sorry

end inequality_of_pos_reals_l81_81329


namespace david_biology_marks_l81_81387

theorem david_biology_marks
  (english math physics chemistry avg_marks num_subjects : ℕ)
  (h_english : english = 86)
  (h_math : math = 85)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 87)
  (h_avg_marks : avg_marks = 85)
  (h_num_subjects : num_subjects = 5) :
  ∃ (biology : ℕ), biology = 85 :=
by
  -- Total marks for all subjects
  let total_marks_for_all_subjects := avg_marks * num_subjects
  -- Total marks in English, Mathematics, Physics, and Chemistry
  let total_marks_in_other_subjects := english + math + physics + chemistry
  -- Marks in Biology
  let biology := total_marks_for_all_subjects - total_marks_in_other_subjects
  existsi biology
  sorry

end david_biology_marks_l81_81387


namespace part1_part2_l81_81741

variable (m : ℝ)

def p (m : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x - 2 ≥ m^2 - 3 * m
def q (m : ℝ) : Prop := ∃ x0 : ℝ, -1 ≤ x0 ∧ x0 ≤ 1 ∧ m ≤ x0

theorem part1 (h : p m) : 1 ≤ m ∧ m ≤ 2 := sorry

theorem part2 (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m < 1) ∨ (1 < m ∧ m ≤ 2) := sorry

end part1_part2_l81_81741


namespace total_students_left_l81_81810

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end total_students_left_l81_81810


namespace tank_full_capacity_l81_81955

theorem tank_full_capacity (C : ℝ) (H1 : 0.4 * C + 36 = 0.7 * C) : C = 120 :=
by
  sorry

end tank_full_capacity_l81_81955


namespace triangle_area_l81_81231

theorem triangle_area (base height : ℝ) (h_base : base = 8.4) (h_height : height = 5.8) :
  0.5 * base * height = 24.36 := by
  sorry

end triangle_area_l81_81231


namespace system_inconsistent_l81_81065

theorem system_inconsistent :
  ¬(∃ (x1 x2 x3 x4 : ℝ), 
    (5 * x1 + 12 * x2 + 19 * x3 + 25 * x4 = 25) ∧
    (10 * x1 + 22 * x2 + 16 * x3 + 39 * x4 = 25) ∧
    (5 * x1 + 12 * x2 + 9 * x3 + 25 * x4 = 30) ∧
    (20 * x1 + 46 * x2 + 34 * x3 + 89 * x4 = 70)) := 
by
  sorry

end system_inconsistent_l81_81065


namespace g_expression_l81_81239

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := sorry

theorem g_expression :
  (∀ x : ℝ, g (x + 2) = f x) → ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end g_expression_l81_81239


namespace sum_of_consecutive_integers_l81_81506

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l81_81506


namespace initial_apples_l81_81525

theorem initial_apples (A : ℕ) 
  (H1 : A - 2 + 4 + 5 = 14) : 
  A = 7 := 
by 
  sorry

end initial_apples_l81_81525


namespace days_to_empty_tube_l81_81666

-- Define the conditions
def gelInTube : ℕ := 128
def dailyUsage : ℕ := 4

-- Define the proof statement
theorem days_to_empty_tube : gelInTube / dailyUsage = 32 := 
by 
  sorry

end days_to_empty_tube_l81_81666


namespace evaluate_expression_l81_81273

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l81_81273


namespace square_properties_l81_81275

theorem square_properties (perimeter : ℝ) (h1 : perimeter = 40) :
  ∃ (side length area diagonal : ℝ), side = 10 ∧ length = 10 ∧ area = 100 ∧ diagonal = 10 * Real.sqrt 2 :=
by
  sorry

end square_properties_l81_81275


namespace beth_finishes_first_l81_81379

open Real

noncomputable def andy_lawn_area : ℝ := sorry
noncomputable def beth_lawn_area : ℝ := andy_lawn_area / 3
noncomputable def carlos_lawn_area : ℝ := andy_lawn_area / 4

noncomputable def andy_mowing_rate : ℝ := sorry
noncomputable def beth_mowing_rate : ℝ := andy_mowing_rate
noncomputable def carlos_mowing_rate : ℝ := andy_mowing_rate / 2

noncomputable def carlos_break : ℝ := 10

noncomputable def andy_mowing_time := andy_lawn_area / andy_mowing_rate
noncomputable def beth_mowing_time := beth_lawn_area / beth_mowing_rate
noncomputable def carlos_mowing_time := (carlos_lawn_area / carlos_mowing_rate) + carlos_break

theorem beth_finishes_first :
  beth_mowing_time < andy_mowing_time ∧ beth_mowing_time < carlos_mowing_time := by
  sorry

end beth_finishes_first_l81_81379


namespace ratio_of_socks_l81_81440

variable (b : ℕ)            -- the number of pairs of blue socks
variable (x : ℝ)            -- the price of blue socks per pair

def original_cost : ℝ := 5 * 3 * x + b * x
def interchanged_cost : ℝ := b * 3 * x + 5 * x

theorem ratio_of_socks :
  (5 : ℝ) / b = 5 / 14 :=
by
  sorry

end ratio_of_socks_l81_81440


namespace factor_expression_l81_81274

noncomputable def numerator (a b c : ℝ) : ℝ := 
(|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3)

noncomputable def denominator (a b c : ℝ) : ℝ := 
(|a + b|^3 + |b + c|^3 + |c + a|^3)

theorem factor_expression (a b c : ℝ) : 
  (denominator a b c) ≠ 0 → 
  (numerator a b c) / (denominator a b c) = 1 :=
by
  sorry

end factor_expression_l81_81274


namespace car_distance_l81_81370

noncomputable def distance_covered (S : ℝ) (T : ℝ) (new_speed : ℝ) : ℝ :=
  S * T

theorem car_distance (S : ℝ) (T : ℝ) (new_time : ℝ) (new_speed : ℝ)
  (h1 : T = 12)
  (h2 : new_time = (3/4) * T)
  (h3 : new_speed = 60)
  (h4 : distance_covered new_speed new_time = 540) :
    distance_covered S T = 540 :=
by
  sorry

end car_distance_l81_81370


namespace min_value_func_y_l81_81291

noncomputable def geometric_sum (t : ℝ) (n : ℕ) : ℝ :=
  t * 3^(n-1) - (1 / 3)

noncomputable def func_y (x t : ℝ) : ℝ :=
  (x + 2) * (x + 10) / (x + t)

theorem min_value_func_y :
  ∀ (t : ℝ), (∀ n : ℕ, geometric_sum t n = (1) → (∀ x > 0, func_y x t ≥ 16)) :=
  sorry

end min_value_func_y_l81_81291


namespace train_speed_correct_l81_81541

def train_length : ℝ := 2500  -- Length of the train in meters.
def crossing_time : ℝ := 100  -- Time to cross the electric pole in seconds.
def expected_speed : ℝ := 25  -- Expected speed of the train in meters/second.

theorem train_speed_correct :
  (train_length / crossing_time) = expected_speed :=
by
  sorry

end train_speed_correct_l81_81541


namespace find_larger_number_l81_81515

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 8 * S + 15) : L = 1557 := 
sorry

end find_larger_number_l81_81515


namespace percentage_error_in_calculated_area_l81_81830

theorem percentage_error_in_calculated_area 
  (s : ℝ) 
  (measured_side : ℝ) 
  (h : measured_side = s * 1.04) :
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 8.16 :=
by
  sorry

end percentage_error_in_calculated_area_l81_81830


namespace fixed_monthly_charge_l81_81389

-- Given conditions
variable (F C_J : ℕ)
axiom january_bill : F + C_J = 46
axiom february_bill : F + 2 * C_J = 76

-- Proof problem
theorem fixed_monthly_charge : F = 16 :=
by
  sorry

end fixed_monthly_charge_l81_81389


namespace grocery_store_more_expensive_l81_81839

def bulk_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def grocery_price_per_can (total_price : ℚ) (num_cans : ℕ) : ℚ := total_price / num_cans

def price_difference_in_cents (price1 : ℚ) (price2 : ℚ) : ℚ := (price2 - price1) * 100

theorem grocery_store_more_expensive
  (bulk_total_price : ℚ)
  (bulk_cans : ℕ)
  (grocery_total_price : ℚ)
  (grocery_cans : ℕ)
  (difference_in_cents : ℚ) :
  bulk_total_price = 12.00 →
  bulk_cans = 48 →
  grocery_total_price = 6.00 →
  grocery_cans = 12 →
  difference_in_cents = 25 →
  price_difference_in_cents (bulk_price_per_can bulk_total_price bulk_cans) 
                            (grocery_price_per_can grocery_total_price grocery_cans) = difference_in_cents := by
  sorry

end grocery_store_more_expensive_l81_81839


namespace percentage_cleared_all_sections_l81_81433

def total_candidates : ℝ := 1200
def cleared_none : ℝ := 0.05 * total_candidates
def cleared_one_section : ℝ := 0.25 * total_candidates
def cleared_four_sections : ℝ := 0.20 * total_candidates
def cleared_two_sections : ℝ := 0.245 * total_candidates
def cleared_three_sections : ℝ := 300

-- Let x be the percentage of candidates who cleared all sections
def cleared_all_sections (x: ℝ) : Prop :=
  let total_cleared := (cleared_none + 
                        cleared_one_section + 
                        cleared_four_sections + 
                        cleared_two_sections + 
                        cleared_three_sections + 
                        x * total_candidates / 100)
  total_cleared = total_candidates

theorem percentage_cleared_all_sections :
  ∃ x, cleared_all_sections x ∧ x = 0.5 :=
by
  sorry

end percentage_cleared_all_sections_l81_81433


namespace least_positive_divisible_by_five_primes_l81_81050

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l81_81050


namespace positive_difference_of_squares_l81_81659

theorem positive_difference_of_squares (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 12) : x^2 - y^2 = 600 := by
  sorry

end positive_difference_of_squares_l81_81659


namespace system_a_l81_81014

theorem system_a (x y z : ℝ) (h1 : x + y + z = 6) (h2 : 1/x + 1/y + 1/z = 11/6) (h3 : x*y + y*z + z*x = 11) :
  x = 1 ∧ y = 2 ∧ z = 3 ∨ x = 1 ∧ y = 3 ∧ z = 2 ∨ x = 2 ∧ y = 1 ∧ z = 3 ∨ x = 2 ∧ y = 3 ∧ z = 1 ∨ x = 3 ∧ y = 1 ∧ z = 2 ∨ x = 3 ∧ y = 2 ∧ z = 1 :=
sorry

end system_a_l81_81014


namespace baker_earnings_l81_81554

-- Define the number of cakes and pies sold
def cakes_sold := 453
def pies_sold := 126

-- Define the prices per cake and pie
def price_per_cake := 12
def price_per_pie := 7

-- Calculate the total earnings
def total_earnings : ℕ := (cakes_sold * price_per_cake) + (pies_sold * price_per_pie)

-- Theorem stating the baker's earnings
theorem baker_earnings : total_earnings = 6318 := by
  unfold total_earnings cakes_sold pies_sold price_per_cake price_per_pie
  sorry

end baker_earnings_l81_81554


namespace SamBalloonsCount_l81_81579

-- Define the conditions
def FredBalloons : ℕ := 10
def DanBalloons : ℕ := 16
def TotalBalloons : ℕ := 72

-- Define the function to calculate Sam's balloons and the main theorem to prove
def SamBalloons := TotalBalloons - (FredBalloons + DanBalloons)

theorem SamBalloonsCount : SamBalloons = 46 := by
  -- The proof is omitted here
  sorry

end SamBalloonsCount_l81_81579


namespace ivan_score_more_than_5_points_l81_81607

/-- Definitions of the given conditions --/
def type_A_tasks : ℕ := 10
def type_B_probability : ℝ := 1/3
def type_B_points : ℕ := 2
def task_A_probability : ℝ := 1/4
def task_A_points : ℕ := 1
def more_than_5_points_probability : ℝ := 0.088

/-- Lean 4 statement equivalent to the math proof problem --/
theorem ivan_score_more_than_5_points:
  let P_A4 := ∑ i in finset.range (7 + 1), nat.choose type_A_tasks i * (task_A_probability ^ i) * ((1 - task_A_probability) ^ (type_A_tasks - i)) in
  let P_A6 := ∑ i in finset.range (11 - 6), nat.choose type_A_tasks (i + 6) * (task_A_probability ^ (i + 6)) * ((1 - task_A_probability) ^ (type_A_tasks - (i + 6))) in
  (P_A4 * type_B_probability + P_A6 * (1 - type_B_probability) = more_than_5_points_probability) := sorry

end ivan_score_more_than_5_points_l81_81607


namespace net_change_in_collection_is_94_l81_81854

-- Definitions for the given conditions
def thrown_away_caps : Nat := 6
def initially_found_caps : Nat := 50
def additionally_found_caps : Nat := 44 + thrown_away_caps

-- Definition of the total found bottle caps
def total_found_caps : Nat := initially_found_caps + additionally_found_caps

-- Net change in Bottle Cap collection
def net_change_in_collection : Nat := total_found_caps - thrown_away_caps

-- Proof statement
theorem net_change_in_collection_is_94 : net_change_in_collection = 94 :=
by
  -- skipped proof
  sorry

end net_change_in_collection_is_94_l81_81854


namespace dogs_count_l81_81221

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l81_81221


namespace quadratic_roots_one_is_twice_l81_81856

theorem quadratic_roots_one_is_twice (a b c : ℝ) (m : ℝ) :
  (∃ x1 x2 : ℝ, 2 * x1^2 - (2 * m + 1) * x1 + m^2 - 9 * m + 39 = 0 ∧ x2 = 2 * x1) ↔ m = 10 ∨ m = 7 :=
by 
  sorry

end quadratic_roots_one_is_twice_l81_81856


namespace find_a_for_perpendicular_tangent_line_l81_81143

theorem find_a_for_perpendicular_tangent_line :
  ∃ a : ℝ, (∀ x y : ℝ, x = 3 → y = (x+1)/(x-1) →
    ∂ (λ x, (x+1)/(x-1)) x = -1/2 →
    ∃ t (h : t = -1 / (-a)), t = 1) ∧ a = -2 :=
by
  sorry

end find_a_for_perpendicular_tangent_line_l81_81143


namespace part_a_part_b_l81_81845

-- Conditions
def has_three_classmates_in_any_group_of_ten (students : Fin 60 → Type) : Prop :=
  ∀ (g : Finset (Fin 60)), g.card = 10 → ∃ (a b c : Fin 60), a ∈ g ∧ b ∈ g ∧ c ∈ g ∧ students a = students b ∧ students b = students c

-- Part (a)
theorem part_a (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ∃ g : Finset (Fin 60), g.card ≥ 15 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

-- Part (b)
theorem part_b (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ¬ ∃ g : Finset (Fin 60), g.card ≥ 16 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

end part_a_part_b_l81_81845


namespace jordan_weight_after_exercise_l81_81913

theorem jordan_weight_after_exercise :
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  current_weight = 222 :=
by
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  show current_weight = 222
  sorry

end jordan_weight_after_exercise_l81_81913


namespace yule_log_surface_area_increase_l81_81241

theorem yule_log_surface_area_increase :
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initial_surface_area := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let slice_height := h / n
  let slice_surface_area := 2 * Real.pi * r * slice_height + 2 * Real.pi * r^2
  let total_surface_area_slices := n * slice_surface_area
  let delta_surface_area := total_surface_area_slices - initial_surface_area
  delta_surface_area = 100 * Real.pi :=
by
  sorry

end yule_log_surface_area_increase_l81_81241


namespace percentage_of_total_spent_on_other_items_l81_81455

-- Definitions for the given problem conditions

def total_amount (a : ℝ) := a
def spent_on_clothing (a : ℝ) := 0.50 * a
def spent_on_food (a : ℝ) := 0.10 * a
def spent_on_other_items (a x_clothing x_food : ℝ) := a - x_clothing - x_food
def tax_on_clothing (x_clothing : ℝ) := 0.04 * x_clothing
def tax_on_food := 0
def tax_on_other_items (x_other_items : ℝ) := 0.08 * x_other_items
def total_tax (a : ℝ) := 0.052 * a

-- The theorem we need to prove
theorem percentage_of_total_spent_on_other_items (a x_clothing x_food x_other_items : ℝ)
    (h1 : x_clothing = spent_on_clothing a)
    (h2 : x_food = spent_on_food a)
    (h3 : x_other_items = spent_on_other_items a x_clothing x_food)
    (h4 : tax_on_clothing x_clothing + tax_on_food + tax_on_other_items x_other_items = total_tax a) :
    0.40 * a = x_other_items :=
sorry

end percentage_of_total_spent_on_other_items_l81_81455


namespace find_a_l81_81351

-- Definitions from conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def directrix : ℝ := 1

-- Statement to prove
theorem find_a (a : ℝ) (h : directrix = 1) : a = -1/4 :=
sorry

end find_a_l81_81351


namespace find_number_l81_81057

theorem find_number (x : ℕ) (h : x * 625 = 584638125) : x = 935420 :=
sorry

end find_number_l81_81057


namespace statement_B_not_true_l81_81553

def diamondsuit (x y : ℝ) : ℝ := 2 * |(x - y)| + 1

theorem statement_B_not_true : ¬ (∀ x y : ℝ, 3 * diamondsuit x y = 3 * diamondsuit (2 * x) (2 * y)) :=
sorry

end statement_B_not_true_l81_81553


namespace lower_bound_expression_l81_81123

theorem lower_bound_expression (n : ℤ) (L : ℤ) :
  (∃ k : ℕ, k = 20 ∧
          ∀ n, (L < 4 * n + 7 ∧ 4 * n + 7 < 80)) →
  L = 3 :=
by
  sorry

end lower_bound_expression_l81_81123


namespace LCM_of_two_numbers_l81_81361

theorem LCM_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 11) (h_product : a * b = 1991) : Nat.lcm a b = 181 :=
by
  sorry

end LCM_of_two_numbers_l81_81361


namespace system_of_linear_equations_l81_81886

-- Define the system of linear equations and a lemma stating the given conditions and the proof goals.
theorem system_of_linear_equations (x y m : ℚ) :
  (x + 3 * y = 7) ∧ (2 * x - 3 * y = 2) ∧ (x - 3 * y + m * x + 3 = 0) ↔ 
  (x = 4 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ m = -2 / 3 :=
by
  sorry

end system_of_linear_equations_l81_81886


namespace probability_Ivan_more_than_5_points_l81_81612

noncomputable def prob_type_A_correct := 1 / 4
noncomputable def total_type_A := 10
noncomputable def prob_type_B_correct := 1 / 3

def binomial (n k : ℕ) : ℚ :=
  (Nat.choose n k) * (prob_type_A_correct ^ k) * ((1 - prob_type_A_correct) ^ (n - k))

def prob_A_4 := ∑ k in finset.range (total_type_A + 1), if k ≥ 4 then binomial total_type_A k else 0
def prob_A_6 := ∑ k in finset.range (total_type_A + 1), if k ≥ 6 then binomial total_type_A k else 0

def prob_B := prob_type_B_correct
def prob_not_B := 1 - prob_type_B_correct

noncomputable def prob_more_than_5_points :=
  prob_A_4 * prob_B + prob_A_6 * prob_not_B

theorem probability_Ivan_more_than_5_points :
  prob_more_than_5_points = 0.088 := by
  sorry

end probability_Ivan_more_than_5_points_l81_81612


namespace john_and_mike_safe_weight_l81_81322

def weight_bench_max_support : ℕ := 1000
def safety_margin_percentage : ℕ := 20
def john_weight : ℕ := 250
def mike_weight : ℕ := 180

def safety_margin : ℕ := (safety_margin_percentage * weight_bench_max_support) / 100
def max_safe_weight : ℕ := weight_bench_max_support - safety_margin
def combined_weight : ℕ := john_weight + mike_weight
def weight_on_bar_together : ℕ := max_safe_weight - combined_weight

theorem john_and_mike_safe_weight :
  weight_on_bar_together = 370 := by
  sorry

end john_and_mike_safe_weight_l81_81322


namespace sum_of_interior_diagonals_l81_81249

theorem sum_of_interior_diagonals (a b c : ℝ)
  (h₁ : 2 * (a * b + b * c + c * a) = 166)
  (h₂ : a + b + c = 16) :
  4 * Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) = 12 * Real.sqrt 10 :=
by
  sorry

end sum_of_interior_diagonals_l81_81249


namespace interest_rate_difference_l81_81978

-- Definitions for given conditions
def principal : ℝ := 3000
def time : ℝ := 9
def additional_interest : ℝ := 1350

-- The Lean 4 statement for the equivalence
theorem interest_rate_difference 
  (R H : ℝ) 
  (h_interest_formula_original : principal * R * time / 100 = principal * R * time / 100) 
  (h_interest_formula_higher : principal * H * time / 100 = principal * R * time / 100 + additional_interest) 
  : (H - R) = 5 :=
sorry

end interest_rate_difference_l81_81978


namespace tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l81_81293

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x) - x * Real.exp x

theorem tangent_line_a_zero (x : ℝ) (y : ℝ) : 
  a = 0 ∧ x = 1 → (2 * Real.exp 1) * x + y - Real.exp 1 = 0 :=
sorry

theorem range_a_if_fx_neg (a : ℝ) : 
  (∀ x ≥ 1, f a x < 0) → a < Real.exp 1 :=
sorry

theorem max_value_a_one (x : ℝ) : 
  a = 1 → x = (Real.exp 1)⁻¹ → f 1 x = -1 :=
sorry

end tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l81_81293


namespace football_practice_hours_l81_81373

theorem football_practice_hours (practice_hours_per_day : ℕ) (days_per_week : ℕ) (missed_days_due_to_rain : ℕ) 
  (practice_hours_per_day_eq_six : practice_hours_per_day = 6)
  (days_per_week_eq_seven : days_per_week = 7)
  (missed_days_due_to_rain_eq_one : missed_days_due_to_rain = 1) : 
  practice_hours_per_day * (days_per_week - missed_days_due_to_rain) = 36 := 
by
  -- proof goes here
  sorry

end football_practice_hours_l81_81373


namespace algae_cells_count_10_days_l81_81087

-- Define the initial condition where the pond starts with one algae cell.
def initial_algae_cells : ℕ := 1

-- Define the daily splitting of each cell into 3 new cells.
def daily_split (cells : ℕ) : ℕ := cells * 3

-- Define the function to compute the number of algae cells after n days.
def algae_cells_after_days (n : ℕ) : ℕ :=
  initial_algae_cells * (3 ^ n)

-- State the theorem to be proved.
theorem algae_cells_count_10_days : algae_cells_after_days 10 = 59049 :=
by {
  sorry
}

end algae_cells_count_10_days_l81_81087


namespace distinct_integers_sum_l81_81398

theorem distinct_integers_sum {a b c d : ℤ} (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_product : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end distinct_integers_sum_l81_81398


namespace find_triples_l81_81236

theorem find_triples (x y z : ℝ) :
  (x + 1)^2 = x + y + 2 ∧
  (y + 1)^2 = y + z + 2 ∧
  (z + 1)^2 = z + x + 2 ↔ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end find_triples_l81_81236


namespace walter_bus_time_l81_81484

noncomputable def walter_schedule : Prop :=
  let wake_up_time := 6  -- Walter gets up at 6:00 a.m.
  let leave_home_time := 7  -- Walter catches the school bus at 7:00 a.m.
  let arrival_home_time := 17  -- Walter arrives home at 5:00 p.m.
  let num_classes := 8  -- Walter has 8 classes
  let class_duration := 45  -- Each class lasts 45 minutes
  let lunch_duration := 40  -- Walter has 40 minutes for lunch
  let additional_activities_hours := 2.5  -- Walter has 2.5 hours of additional activities

  -- Total time calculation
  let total_away_hours := arrival_home_time - leave_home_time
  let total_away_minutes := total_away_hours * 60

  -- School-related activities calculation
  let total_class_minutes := num_classes * class_duration
  let total_additional_activities_minutes := additional_activities_hours * 60
  let total_school_activity_minutes := total_class_minutes + lunch_duration + total_additional_activities_minutes

  -- Time spent on the bus
  let bus_time := total_away_minutes - total_school_activity_minutes
  bus_time = 50

-- Statement to prove
theorem walter_bus_time : walter_schedule :=
  sorry

end walter_bus_time_l81_81484


namespace sale_price_l81_81910

def original_price : ℝ := 100
def discount_rate : ℝ := 0.80

theorem sale_price (original_price discount_rate : ℝ) : original_price * (1 - discount_rate) = 20 := by
  sorry

end sale_price_l81_81910


namespace degrees_to_radians_l81_81713

theorem degrees_to_radians : (800 : ℝ) * (Real.pi / 180) = (40 / 9) * Real.pi :=
by
  sorry

end degrees_to_radians_l81_81713


namespace unique_sequence_exists_and_bounded_l81_81404

theorem unique_sequence_exists_and_bounded (a : ℝ) (n : ℕ) :
  ∃! (x : ℕ → ℝ), -- There exists a unique sequence x : ℕ → ℝ
    (x 1 = x (n - 1)) ∧ -- x_1 = x_{n-1}
    (∀ i, 1 ≤ i ∧ i ≤ n → (1 / 2) * (x (i - 1) + x i) = x i + x i ^ 3 - a ^ 3) ∧ -- Condition for all 1 ≤ i ≤ n
    (∀ i, 0 ≤ i ∧ i ≤ n + 1 → |x i| ≤ |a|) -- Bounding condition for all 0 ≤ i ≤ n + 1
:= sorry

end unique_sequence_exists_and_bounded_l81_81404


namespace multiplication_verification_l81_81935

-- Define the variables
variables (P Q R S T U : ℕ)

-- Define the known digits in the numbers
def multiplicand := 60000 + 1000 * P + 100 * Q + 10 * R
def multiplier := 5000000 + 10000 * S + 1000 * T + 100 * U + 5

-- Define the proof statement
theorem multiplication_verification : 
  (multiplicand P Q R) * (multiplier S T U) = 20213 * 732575 :=
  sorry

end multiplication_verification_l81_81935


namespace sum_of_fractions_eq_five_fourteen_l81_81570

theorem sum_of_fractions_eq_five_fourteen :
  (1 : ℚ) / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) = 5 / 14 := 
by
  sorry

end sum_of_fractions_eq_five_fourteen_l81_81570


namespace ivan_scores_more_than_5_points_l81_81602

-- Definitions based on problem conditions
def typeA_problem_probability (correct_guesses : ℕ) (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  (Nat.choose total_tasks correct_guesses : ℚ) * (success_prob ^ correct_guesses) * (failure_prob ^ (total_tasks - correct_guesses))

def probability_A4 (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  ∑ i in Finset.range (total_tasks + 1), if i ≥ 4 then typeA_problem_probability i total_tasks success_prob failure_prob else 0

def probability_A6 (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  ∑ i in Finset.range (total_tasks + 1), if i ≥ 6 then typeA_problem_probability i total_tasks success_prob failure_prob else 0

def final_probability (p_A4 : ℚ) (p_A6 : ℚ) (p_B : ℚ) : ℚ :=
  (p_A4 * p_B) + (p_A6 * (1 - p_B))

noncomputable def probability_ivan_scores_more_than_5 : ℚ :=
  let total_tasks := 10
  let success_prob := 1 / 4
  let failure_prob := 3 / 4
  let p_B := 1 / 3
  let p_A4 := probability_A4 total_tasks success_prob failure_prob
  let p_A6 := probability_A6 total_tasks success_prob failure_prob
  final_probability p_A4 p_A6 p_B

theorem ivan_scores_more_than_5_points : probability_ivan_scores_more_than_5 = 0.088 := 
  sorry

end ivan_scores_more_than_5_points_l81_81602


namespace radius_moon_scientific_notation_l81_81211

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end radius_moon_scientific_notation_l81_81211


namespace min_k_value_l81_81870

-- Definition of the problem's conditions
def remainder_condition (n k : ℕ) : Prop :=
  ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1

def in_range (x a b : ℕ) : Prop :=
  a < x ∧ x < b

-- The statement of the proof problem in Lean 4
theorem min_k_value (n k : ℕ) (h1 : remainder_condition n k) (hn_range : in_range n 2000 3000) :
  k = 9 :=
sorry

end min_k_value_l81_81870


namespace sqrt_expression_value_l81_81328

variable (a b : ℝ) 

theorem sqrt_expression_value (ha : a ≠ 0) (hb : b ≠ 0) (ha_neg : a < 0) :
  Real.sqrt (-a^3) * Real.sqrt ((-b)^4) = -a * |b| * Real.sqrt (-a) := by
  sorry

end sqrt_expression_value_l81_81328


namespace base3_sum_example_l81_81981

noncomputable def base3_add (a b : ℕ) : ℕ := sorry  -- Function to perform base-3 addition

theorem base3_sum_example : 
  base3_add (base3_add (base3_add (base3_add 2 120) 221) 1112) 1022 = 21201 := sorry

end base3_sum_example_l81_81981


namespace find_factor_l81_81072

theorem find_factor {n f : ℝ} (h1 : n = 10) (h2 : f * (2 * n + 8) = 84) : f = 3 :=
by
  sorry

end find_factor_l81_81072


namespace same_function_representation_l81_81829

theorem same_function_representation : 
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = x^2 - 2*x - 1) ∧ (∀ m, g m = m^2 - 2*m - 1) →
    (f = g) :=
by
  sorry

end same_function_representation_l81_81829


namespace part1_part2_l81_81577

noncomputable def A_m (m : ℕ) (k : ℕ) : ℕ := (2 * k - 1) * m + k

theorem part1 (m : ℕ) (hm : m ≥ 2) :
  ∃ a : ℕ, 1 ≤ a ∧ a < m ∧ (∃ k : ℕ, 2^a = A_m m k) ∨ (∃ k : ℕ, 2^a + 1 = A_m m k) :=
sorry

theorem part2 {m : ℕ} (hm : m ≥ 2) 
  (a b : ℕ) (ha : ∃ k, 2^a = A_m m k) (hb : ∃ k, 2^b + 1 = A_m m k)
  (hmin_a : ∀ x, (∃ k, 2^x = A_m m k) → a ≤ x) 
  (hmin_b : ∀ y, (∃ k, 2^y + 1 = A_m m k) → b ≤ y) :
  a = 2 * b + 1 :=
sorry

end part1_part2_l81_81577


namespace clock_shows_four_different_digits_for_588_minutes_l81_81192

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l81_81192


namespace work_completion_time_of_x_l81_81961

def totalWork := 1  -- We can normalize W to 1 unit to simplify the problem

theorem work_completion_time_of_x (W : ℝ) (Wx Wy : ℝ) 
  (hx : 8 * Wx + 16 * Wy = W)
  (hy : Wy = W / 20) :
  Wx = W / 40 :=
by
  -- The proof goes here, but we just put sorry for now.
  sorry

end work_completion_time_of_x_l81_81961


namespace remainder_777_777_mod_13_l81_81233

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end remainder_777_777_mod_13_l81_81233


namespace math_expression_equivalent_l81_81099

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l81_81099


namespace fraction_of_girls_on_trip_l81_81984

theorem fraction_of_girls_on_trip (b g : ℕ) (h : b = g) :
  ((2 / 3 * g) / (5 / 6 * b + 2 / 3 * g)) = 4 / 9 :=
by
  sorry

end fraction_of_girls_on_trip_l81_81984


namespace proof_value_g_expression_l81_81188

noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom g_invertible : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x
axiom g_table : ∀ x, (x = 1 → g x = 4) ∧ (x = 2 → g x = 5) ∧ (x = 3 → g x = 7) ∧ (x = 4 → g x = 9) ∧ (x = 5 → g x = 10)

theorem proof_value_g_expression :
  g (g 2) + g (g_inv 9) + g_inv (g_inv 7) = 21 :=
by
  sorry

end proof_value_g_expression_l81_81188


namespace eden_bears_count_l81_81714

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end eden_bears_count_l81_81714


namespace maximum_consecutive_positive_integers_sum_500_l81_81499

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l81_81499


namespace asymptotes_of_hyperbola_l81_81584

theorem asymptotes_of_hyperbola (a b : ℝ) (h_cond1 : a > b) (h_cond2 : b > 0) 
  (h_eq_ell : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h_eq_hyp : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h_product : ∀ e1 e2 : ℝ, (e1 = Real.sqrt (1 - (b^2 / a^2))) → 
                (e2 = Real.sqrt (1 + (b^2 / a^2))) → 
                (e1 * e2 = Real.sqrt 3 / 2)) :
  ∀ x y : ℝ, x + Real.sqrt 2 * y = 0 ∨ x - Real.sqrt 2 * y = 0 :=
sorry

end asymptotes_of_hyperbola_l81_81584


namespace sport_flavoring_to_water_ratio_l81_81435

/-- The ratio by volume of flavoring to corn syrup to water in the 
standard formulation is 1:12:30. The sport formulation has a ratio 
of flavoring to corn syrup three times as great as in the standard formulation. 
A large bottle of the sport formulation contains 4 ounces of corn syrup and 
60 ounces of water. Prove that the ratio of the amount of flavoring to water 
in the sport formulation compared to the standard formulation is 1:2. -/
theorem sport_flavoring_to_water_ratio 
    (standard_flavoring : ℝ) 
    (standard_corn_syrup : ℝ) 
    (standard_water : ℝ) : 
  standard_flavoring = 1 → standard_corn_syrup = 12 → 
  standard_water = 30 → 
  ∃ sport_flavoring : ℝ, 
  ∃ sport_corn_syrup : ℝ, 
  ∃ sport_water : ℝ, 
  sport_corn_syrup = 4 ∧ 
  sport_water = 60 ∧ 
  (sport_flavoring / sport_water) = (standard_flavoring / standard_water) / 2 :=
by
  sorry

end sport_flavoring_to_water_ratio_l81_81435


namespace tangent_line_at_x_5_l81_81648

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_x_5 :
  (∀ x, f x = -x + 8 → f 5 + deriv f 5 = 2) := sorry

end tangent_line_at_x_5_l81_81648


namespace seventh_grade_problem_l81_81766

theorem seventh_grade_problem (x y : ℕ) (h1 : x + y = 12) (h2 : 6 * x = 3 * 4 * y) :
  (x + y = 12 ∧ 6 * x = 3 * 4 * y) :=
by
  apply And.intro
  . exact h1
  . exact h2

end seventh_grade_problem_l81_81766


namespace final_value_of_x_l81_81429

noncomputable def initial_x : ℝ := 52 * 1.2
noncomputable def decreased_x : ℝ := initial_x * 0.9
noncomputable def final_x : ℝ := decreased_x * 1.15

theorem final_value_of_x : final_x = 64.584 := by
  sorry

end final_value_of_x_l81_81429


namespace symmetric_function_value_l81_81892

theorem symmetric_function_value (f : ℝ → ℝ)
  (h : ∀ x, f (2^(x-2)) = x) : f 8 = 5 :=
sorry

end symmetric_function_value_l81_81892


namespace remainder_of_3_pow_244_mod_5_l81_81507

theorem remainder_of_3_pow_244_mod_5 : (3^244) % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l81_81507


namespace space_station_arrangement_l81_81805

theorem space_station_arrangement :
  ∃ (n : ℕ), (n = 6) ∧ ∃ (modules : ℕ), (modules = 3) ∧
  (∀ (a b c : ℕ), a + b + c = n → (1 ≤ a ∧ a ≤ 3) ∧ (1 ≤ b ∧ b ≤ 3) ∧ (1 ≤ c ∧ c ≤ 3) →
  (module_arrangements (a, b, c).fst (a, b, c).snd (a, b, c).trd = 450)) :=
begin
  sorry
end

end space_station_arrangement_l81_81805


namespace envelope_of_family_of_lines_l81_81864

theorem envelope_of_family_of_lines (a α : ℝ) (hα : α > 0) :
    ∀ (x y : ℝ), (∃ α > 0,
    (x = a * α / 2 ∧ y = a / (2 * α))) ↔ (x * y = a^2 / 4) := by
  sorry

end envelope_of_family_of_lines_l81_81864


namespace triangles_in_plane_l81_81627

-- Definitions for the problem conditions
def set_of_n_points (n : ℕ) (h : n ≥ 3) : Set (ℝ × ℝ) := sorry
def no_three_collinear (P : Set (ℝ × ℝ)) : Prop := sorry

-- Claim and proof statement
theorem triangles_in_plane (n : ℕ) (h_pos : n ≥ 3) (P : Set (ℝ × ℝ)) 
  (h_points : P = set_of_n_points n h_pos) (h_collinear : no_three_collinear P) :
  (∃ T : Finset (Finset (ℝ × ℝ)), 
    T.card = (n - 1).choose 2 ∧
    (∀ t ∈ T, ∀ s ∈ T, t ≠ s → t ∩ s = ∅)) →
  (if n = 3 then T.card = 1 
   else T.card = n) :=
sorry

end triangles_in_plane_l81_81627


namespace find_larger_number_l81_81820

theorem find_larger_number (x y : ℤ) (h1 : x - y = 7) (h2 : x + y = 41) : x = 24 :=
by sorry

end find_larger_number_l81_81820


namespace conference_games_scheduled_l81_81464

theorem conference_games_scheduled
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_games_per_pair : ℕ)
  (inter_games_per_pair : ℕ)
  (h_div : divisions = 3)
  (h_teams : teams_per_division = 4)
  (h_intra : intra_games_per_pair = 3)
  (h_inter : inter_games_per_pair = 2) :
  let intra_division_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_games_per_pair
  let intra_division_total := intra_division_games * divisions
  let inter_division_games := teams_per_division * (teams_per_division * (divisions - 1)) * inter_games_per_pair
  let inter_division_total := inter_division_games * divisions / 2
  let total_games := intra_division_total + inter_division_total
  total_games = 150 :=
by
  sorry

end conference_games_scheduled_l81_81464


namespace algebraic_expression_value_l81_81881

theorem algebraic_expression_value (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
  (1 / (a^2 + 1) + 1 / (b^2 + 1)) = 1 :=
sorry

end algebraic_expression_value_l81_81881


namespace factor_in_range_l81_81475

-- Define the given constants
def a : ℕ := 201212200619
def lower_bound : ℕ := 6000000000
def upper_bound : ℕ := 6500000000
def m : ℕ := 6490716149

-- The Lean proof statement
theorem factor_in_range :
  m ∣ a ∧ lower_bound < m ∧ m < upper_bound :=
by
  exact ⟨sorry, sorry, sorry⟩

end factor_in_range_l81_81475


namespace tan_sum_simplification_l81_81943

open Real

theorem tan_sum_simplification :
  tan 70 + tan 50 - sqrt 3 * tan 70 * tan 50 = -sqrt 3 := by
  sorry

end tan_sum_simplification_l81_81943


namespace child_ticket_cost_l81_81652

theorem child_ticket_cost 
    (x : ℝ)
    (adult_ticket_cost : ℝ := 5)
    (total_sales : ℝ := 178)
    (total_tickets_sold : ℝ := 42)
    (child_tickets_sold : ℝ := 16) 
    (adult_tickets_sold : ℝ := total_tickets_sold - child_tickets_sold)
    (total_adult_sales : ℝ := adult_tickets_sold * adult_ticket_cost)
    (sales_equation : total_adult_sales + child_tickets_sold * x = total_sales) : 
    x = 3 :=
by
  sorry

end child_ticket_cost_l81_81652


namespace trailing_zeroes_500_l81_81257

open Nat

theorem trailing_zeroes_500! (a b : ℕ) (h₀ : 500! = a) (h₁ : trailing_zeroes a = 124) (h₂ : 200! = b) (h₃ : trailing_zeroes b = 49) :
  trailing_zeroes (500! + 200!) = 124 :=
by
  -- We can use Lean's inbuilt functions for calculations related to factorial and trailing zeroes
  sorry

end trailing_zeroes_500_l81_81257


namespace equal_probability_after_adding_balls_l81_81161

theorem equal_probability_after_adding_balls :
  let initial_white := 2
  let initial_yellow := 3
  let added_white := 4
  let added_yellow := 3
  let total_white := initial_white + added_white
  let total_yellow := initial_yellow + added_yellow
  let total_balls := total_white + total_yellow
  (total_white / total_balls) = (total_yellow / total_balls) := by
  sorry

end equal_probability_after_adding_balls_l81_81161


namespace slices_with_both_toppings_l81_81524

theorem slices_with_both_toppings :
  ∀ (h p b : ℕ),
  (h + b = 9) ∧ (p + b = 12) ∧ (h + p + b = 15) → b = 6 :=
by
  sorry

end slices_with_both_toppings_l81_81524


namespace engineer_formula_updated_l81_81999

theorem engineer_formula_updated (T H : ℕ) (hT : T = 5) (hH : H = 10) :
  (30 * T^5) / (H^3 : ℚ) = 375 / 4 := by
  sorry

end engineer_formula_updated_l81_81999


namespace increase_percent_exceeds_l81_81150

theorem increase_percent_exceeds (p q M : ℝ) (M_positive : 0 < M) (p_positive : 0 < p) (q_positive : 0 < q) (q_less_p : q < p) :
  (M * (1 + p / 100) * (1 + q / 100) > M) ↔ (0 < p ∧ 0 < q) :=
by
  sorry

end increase_percent_exceeds_l81_81150


namespace least_positive_number_divisible_by_five_smallest_primes_l81_81040

theorem least_positive_number_divisible_by_five_smallest_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  Nat.min (p1 * p2 * p3 * p4 * p5) := 
  2310 :=
by
  sorry

end least_positive_number_divisible_by_five_smallest_primes_l81_81040


namespace sufficient_but_not_necessary_condition_l81_81880

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a < -1) → (|a| > 1) ∧ ¬((|a| > 1) → (a < -1)) :=
by
-- This statement represents the required proof.
sorry

end sufficient_but_not_necessary_condition_l81_81880


namespace percentage_runs_by_running_l81_81067

theorem percentage_runs_by_running 
  (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) 
  (runs_per_boundary : ℕ) (runs_per_six : ℕ)
  (H_total_runs : total_runs = 120)
  (H_boundaries : boundaries = 3)
  (H_sixes : sixes = 8)
  (H_runs_per_boundary : runs_per_boundary = 4)
  (H_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs : ℚ) * 100 = 50 := 
by
  sorry

end percentage_runs_by_running_l81_81067


namespace part1_part2_l81_81402

def p (x a : ℝ) : Prop := x - a < 0
def q (x : ℝ) : Prop := x * x - 4 * x + 3 ≤ 0

theorem part1 (a : ℝ) (h : a = 2) (hpq : ∀ x : ℝ, p x a ∧ q x) :
  Set.Ico 1 (2 : ℝ) = {x : ℝ | p x a ∧ q x} :=
by {
  sorry
}

theorem part2 (hp : ∀ (x a : ℝ), p x a → ¬ q x) : {a : ℝ | ∀ x : ℝ, q x → p x a} = Set.Ioi 3 :=
by {
  sorry
}

end part1_part2_l81_81402


namespace negation_of_exists_l81_81209

theorem negation_of_exists (x : ℝ) :
  ¬ (∃ x > 0, 2 * x + 3 ≤ 0) ↔ ∀ x > 0, 2 * x + 3 > 0 :=
by
  sorry

end negation_of_exists_l81_81209


namespace smallest_number_increased_by_3_divisible_by_divisors_l81_81056

theorem smallest_number_increased_by_3_divisible_by_divisors
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : d1 = 27)
  (h2 : d2 = 35)
  (h3 : d3 = 25)
  (h4 : d4 = 21) :
  (n + 3) % d1 = 0 →
  (n + 3) % d2 = 0 →
  (n + 3) % d3 = 0 →
  (n + 3) % d4 = 0 →
  n = 4722 :=
by
  sorry

end smallest_number_increased_by_3_divisible_by_divisors_l81_81056


namespace find_range_g_l81_81265

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

theorem find_range_g :
  {x : ℝ | g (2 * x - 1) < g 3} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end find_range_g_l81_81265


namespace speed_ratio_l81_81431

-- Define the speeds of A and B
variables (v_A v_B : ℝ)

-- Assume the conditions of the problem
axiom h1 : 200 / v_A = 400 / v_B

-- Prove the ratio of the speeds
theorem speed_ratio : v_A / v_B = 1 / 2 :=
by
  sorry

end speed_ratio_l81_81431


namespace clock_display_four_different_digits_l81_81201

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l81_81201


namespace prob_of_target_hit_l81_81417

noncomputable def probability_target_hit : ℚ :=
  let pA := (1 : ℚ) / 2
  let pB := (1 : ℚ) / 3
  let pC := (1 : ℚ) / 4
  let pA' := 1 - pA
  let pB' := 1 - pB
  let pC' := 1 - pC
  let pNoneHit := pA' * pB' * pC'
  1 - pNoneHit

-- Statement to be proved
theorem prob_of_target_hit : probability_target_hit = 3 / 4 :=
  sorry

end prob_of_target_hit_l81_81417


namespace enclosed_area_correct_l81_81795

noncomputable def enclosed_area : ℝ :=
  ∫ x in (1/2)..2, (-x + 5/2 - 1/x)

theorem enclosed_area_correct :
  enclosed_area = (15/8) - 2 * Real.log 2 :=
by
  sorry

end enclosed_area_correct_l81_81795


namespace ratio_perimeter_pentagon_to_square_l81_81190

theorem ratio_perimeter_pentagon_to_square
  (a : ℝ) -- Let a be the length of each side of the square
  (T_perimeter S_perimeter : ℝ) 
  (h1 : T_perimeter = S_perimeter) -- Given the perimeter of the triangle equals the perimeter of the square
  (h2 : S_perimeter = 4 * a) -- Given the perimeter of the square is 4 times the length of its side
  (P_perimeter : ℝ)
  (h3 : P_perimeter = (T_perimeter + S_perimeter) - 2 * a) -- Perimeter of the pentagon considering shared edge
  :
  P_perimeter / S_perimeter = 3 / 2 := 
sorry

end ratio_perimeter_pentagon_to_square_l81_81190


namespace number_of_positive_integer_values_l81_81720

theorem number_of_positive_integer_values (N : ℕ) :
  ∃ (card_positive_N : ℕ), 
  (∀ (N : ℕ), (N > 0) → ∃ (d : ℕ), d ∣ 49 ∧ d > 3 ∧ N = d - 3)
  ∧ card_positive_N = 2 := 
sorry

end number_of_positive_integer_values_l81_81720


namespace hotel_cost_l81_81968

theorem hotel_cost (x y : ℕ) (h1 : 3 * x + 6 * y = 1020) (h2 : x + 5 * y = 700) :
  5 * (x + y) = 1100 :=
sorry

end hotel_cost_l81_81968


namespace cannot_be_computed_using_square_difference_l81_81683

theorem cannot_be_computed_using_square_difference (x : ℝ) :
  (x+1)*(1+x) ≠ (a + b)*(a - b) :=
by
  intro a b
  have h : (x + 1) * (1 + x) = (a + b) * (a - b) → false := sorry
  exact h

#align $

end cannot_be_computed_using_square_difference_l81_81683


namespace find_pairs_1984_l81_81573

theorem find_pairs_1984 (m n : ℕ) :
  19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by
  sorry

end find_pairs_1984_l81_81573


namespace max_consecutive_sum_l81_81496

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l81_81496


namespace abs_eq_five_l81_81302

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l81_81302


namespace square_integer_2209_implies_value_l81_81885

theorem square_integer_2209_implies_value (x : ℤ) (h : x^2 = 2209) : (2*x + 1)*(2*x - 1) = 8835 :=
by sorry

end square_integer_2209_implies_value_l81_81885


namespace find_a_l81_81992

def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_a : ∃ a : ℝ, f a + g a = 0 ∧ a = 5 / 7 :=
by
  sorry

end find_a_l81_81992


namespace pipes_fill_cistern_time_l81_81668

noncomputable def pipe_fill_time : ℝ :=
  let rateA := 1 / 80
  let rateC := 1 / 60
  let combined_rateAB := 1 / 20
  let rateB := combined_rateAB - rateA
  let combined_rateABC := rateA + rateB - rateC
  1 / combined_rateABC

theorem pipes_fill_cistern_time :
  pipe_fill_time = 30 := by
  sorry

end pipes_fill_cistern_time_l81_81668


namespace problem1_problem2_l81_81745

section
variable {α : Real}
variable (tan_α : Real)
variable (sin_α cos_α : Real)

def trigonometric_identities (tan_α sin_α cos_α : Real) : Prop :=
  tan_α = 2 ∧ sin_α = tan_α * cos_α

theorem problem1 (h : trigonometric_identities tan_α sin_α cos_α) :
  (4 * sin_α - 2 * cos_α) / (5 * cos_α + 3 * sin_α) = 6 / 11 := by
  sorry

theorem problem2 (h : trigonometric_identities tan_α sin_α cos_α) :
  (1 / 4 * sin_α^2 + 1 / 3 * sin_α * cos_α + 1 / 2 * cos_α^2) = 13 / 30 := by
  sorry
end

end problem1_problem2_l81_81745


namespace distance_between_foci_l81_81469

-- Define the equation of the hyperbola.
def hyperbola_eq (x y : ℝ) : Prop := x * y = 4

-- The coordinates of foci for hyperbola of the form x*y = 4
def foci_1 : (ℝ × ℝ) := (2, 2)
def foci_2 : (ℝ × ℝ) := (-2, -2)

-- Define the Euclidean distance function.
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that the distance between the foci is 4√2.
theorem distance_between_foci : euclidean_distance foci_1 foci_2 = 4 * real.sqrt 2 := sorry

end distance_between_foci_l81_81469


namespace max_a4_l81_81284

variable (a1 d : ℝ)

theorem max_a4 (h1 : 2 * a1 + 6 * d ≥ 10) (h2 : 2.5 * a1 + 10 * d ≤ 15) :
  ∃ max_a4, max_a4 = 4 ∧ a1 + 3 * d ≤ max_a4 :=
by
  sorry

end max_a4_l81_81284


namespace find_vasya_floor_l81_81551

theorem find_vasya_floor (steps_petya: ℕ) (steps_vasya: ℕ) (petya_floors: ℕ) (steps_per_floor: ℝ):
  steps_petya = 36 → petya_floors = 2 → steps_vasya = 72 → 
  steps_per_floor = steps_petya / petya_floors → 
  (1 + (steps_vasya / steps_per_floor)) = 5 := by 
  intros h1 h2 h3 h4 
  sorry

end find_vasya_floor_l81_81551


namespace tenth_student_solved_six_l81_81531

theorem tenth_student_solved_six : 
  ∀ (n : ℕ), 
    (∀ (i : ℕ) (j : ℕ), 1 ≤ i ∧ i ≤ 10 → 1 ≤ j ∧ j ≤ n → (∀ k : ℕ, k ≤ n → ∃ s : ℕ, s = 7)) → 
    (∀ (i : ℕ), 1 ≤ i ∧ i ≤ 9 → ∃ p : ℕ, p = 4) → ∃ m : ℕ, m = 6 := 
by
  sorry

end tenth_student_solved_six_l81_81531


namespace knights_on_red_chairs_l81_81814

variable (K L Kr Lb : ℕ)
variable (h1 : K + L = 20)
variable (h2 : Kr + Lb = 10)
variable (h3 : Kr = L - Lb)

/-- Given the conditions:
1. There are 20 seats with knights and liars such that K + L = 20.
2. Half of the individuals claim to be sitting on blue chairs, and half on red chairs such that Kr + Lb = 10.
3. Knights on red chairs (Kr) must be equal to liars minus liars on blue chairs (Lb).
Prove that the number of knights now sitting on red chairs is 5. -/
theorem knights_on_red_chairs : Kr = 5 :=
by
  sorry

end knights_on_red_chairs_l81_81814


namespace cucumbers_count_l81_81847

theorem cucumbers_count:
  ∀ (C T : ℕ), C + T = 420 ∧ T = 4 * C → C = 84 :=
by
  intros C T h
  sorry

end cucumbers_count_l81_81847


namespace largest_num_pencils_in_package_l81_81452

theorem largest_num_pencils_in_package (Ming_pencils Catherine_pencils : ℕ) 
  (Ming_pencils := 40) 
  (Catherine_pencils := 24) 
  (H : ∃ k, Ming_pencils = k * a ∧ Catherine_pencils = k * b) :
  gcd Ming_pencils Catherine_pencils = 8 :=
by
  sorry

end largest_num_pencils_in_package_l81_81452


namespace sum_of_cubics_l81_81782

noncomputable def root_polynomial (x : ℝ) := 5 * x^3 + 2003 * x + 3005

theorem sum_of_cubics (a b c : ℝ)
  (h1 : root_polynomial a = 0)
  (h2 : root_polynomial b = 0)
  (h3 : root_polynomial c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
sorry

end sum_of_cubics_l81_81782


namespace transport_cost_correct_l81_81794

-- Defining the weights of the sensor unit and communication module in grams
def weight_sensor_grams : ℕ := 500
def weight_comm_module_grams : ℕ := 1500

-- Defining the transport cost per kilogram
def cost_per_kg_sensor : ℕ := 25000
def cost_per_kg_comm_module : ℕ := 20000

-- Converting weights to kilograms
def weight_sensor_kg : ℚ := weight_sensor_grams / 1000
def weight_comm_module_kg : ℚ := weight_comm_module_grams / 1000

-- Calculating the transport costs
def cost_sensor : ℚ := weight_sensor_kg * cost_per_kg_sensor
def cost_comm_module : ℚ := weight_comm_module_kg * cost_per_kg_comm_module

-- Total cost of transporting both units
def total_cost : ℚ := cost_sensor + cost_comm_module

-- Proving that the total cost is $42500
theorem transport_cost_correct : total_cost = 42500 := by
  sorry

end transport_cost_correct_l81_81794


namespace number_of_customers_l81_81080

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l81_81080


namespace ruth_train_track_length_l81_81012

theorem ruth_train_track_length (n : ℕ) (R : ℕ)
  (h_sean : 72 = 8 * n)
  (h_ruth : 72 = R * n) : 
  R = 8 :=
by
  sorry

end ruth_train_track_length_l81_81012


namespace num_isosceles_right_triangles_in_ellipse_l81_81877

theorem num_isosceles_right_triangles_in_ellipse
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ ∃ t : ℝ, (x, y) = (a * Real.cos t, b * Real.sin t))
  :
  (∃ n : ℕ,
    (n = 3 ∧ a > Real.sqrt 3 * b) ∨
    (n = 1 ∧ (b < a ∧ a ≤ Real.sqrt 3 * b))
  ) :=
sorry

end num_isosceles_right_triangles_in_ellipse_l81_81877


namespace max_x_minus_y_isosceles_l81_81378

theorem max_x_minus_y_isosceles (x y : ℝ) (hx : x ≠ 50) (hy : y ≠ 50) 
  (h_iso1 : x = y ∨ 50 = y) (h_iso2 : x = y ∨ 50 = x)
  (h_triangle : 50 + x + y = 180) : 
  max (x - y) (y - x) = 30 :=
sorry

end max_x_minus_y_isosceles_l81_81378


namespace slope_of_line_l81_81755

theorem slope_of_line 
  (p l t : ℝ) (p_pos : p > 0)
  (h_parabola : (2:ℝ)*p = 4) -- Since the parabola passes through M(l,2)
  (h_incircle_center : ∃ (k m : ℝ), (k + 1 = 0) ∧ (k^2 - k - 2 = 0)) :
  ∃ (k : ℝ), k = -1 :=
by {
  sorry
}

end slope_of_line_l81_81755


namespace part1_part2_distribution_part2_expected_value_l81_81822

noncomputable def prob_A_win : ℝ := 2 / 3
noncomputable def prob_B_win : ℝ := 1 / 3

def prob_B_one_A_win_match : ℝ :=
  (prob_B_win * prob_A_win * prob_A_win) +
  (prob_A_win * prob_B_win * prob_A_win * prob_A_win)

theorem part1 : prob_B_one_A_win_match = 20 / 81 := 
  sorry

def prob_X (n : ℕ) : ℝ :=
  match n with
  | 2 => (prob_A_win * prob_A_win) + (prob_B_win * prob_B_win)
  | 3 => 2 * (prob_B_win * prob_A_win * prob_A_win)
  | 4 => 2 * (prob_A_win * prob_B_win * prob_A_win * prob_A_win)
  | 5 => 2 * (prob_B_win * prob_A_win * prob_B_win * prob_A_win)
  | _ => 0

theorem part2_distribution :
  prob_X 2 = 5 / 9 ∧ 
  prob_X 3 = 2 / 9 ∧ 
  prob_X 4 = 10 / 81 ∧ 
  prob_X 5 = 8 / 81 := 
  sorry

def expected_value_X : ℝ :=
  2 * prob_X 2 + 3 * prob_X 3 + 4 * prob_X 4 + 5 * prob_X 5

theorem part2_expected_value :
  expected_value_X = 224 / 81 :=
  sorry

end part1_part2_distribution_part2_expected_value_l81_81822


namespace people_in_room_l81_81951

open Nat

theorem people_in_room (C : ℕ) (P : ℕ) (h1 : 1 / 4 * C = 6) (h2 : 3 / 4 * C = 2 / 3 * P) : P = 27 := by
  sorry

end people_in_room_l81_81951


namespace range_of_k_l81_81596

noncomputable def function_defined_for_all_x (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 + 4 * k * x + 3 ≠ 0

theorem range_of_k :
  {k : ℝ | function_defined_for_all_x k} = {k : ℝ | 0 ≤ k ∧ k < 3 / 4} :=
by
  sorry

end range_of_k_l81_81596


namespace initial_fee_is_correct_l81_81321

noncomputable def initial_fee (total_charge : ℝ) (charge_per_segment : ℝ) (segment_length : ℝ) (distance : ℝ) : ℝ :=
  total_charge - (⌊distance / segment_length⌋ * charge_per_segment)

theorem initial_fee_is_correct :
  initial_fee 4.5 0.25 (2/5) 3.6 = 2.25 :=
by 
  sorry

end initial_fee_is_correct_l81_81321


namespace polynomial_expansion_identity_l81_81397

theorem polynomial_expansion_identity
  (a a1 a3 a4 a5 : ℝ)
  (h : (a - x)^5 = a + a1 * x + 80 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) :
  a + a1 + 80 + a3 + a4 + a5 = 1 := 
sorry

end polynomial_expansion_identity_l81_81397


namespace set_intersection_nonempty_implies_m_le_neg1_l81_81884

theorem set_intersection_nonempty_implies_m_le_neg1
  (m : ℝ)
  (A : Set ℝ := {x | x^2 - 4 * m * x + 2 * m + 6 = 0})
  (B : Set ℝ := {x | x < 0}) :
  (A ∩ B).Nonempty → m ≤ -1 := 
sorry

end set_intersection_nonempty_implies_m_le_neg1_l81_81884


namespace inequality_constant_l81_81721

noncomputable def smallest_possible_real_constant : ℝ :=
  1.0625

theorem inequality_constant (C : ℝ) : 
  (∀ x y z : ℝ, (x + y + z = -1) → 
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1| ) ↔ C ≥ smallest_possible_real_constant :=
sorry

end inequality_constant_l81_81721


namespace dawn_hourly_income_l81_81110

theorem dawn_hourly_income 
  (n : ℕ) (t_s t_p t_f I_p I_s I_f : ℝ)
  (h_n : n = 12)
  (h_t_s : t_s = 1.5)
  (h_t_p : t_p = 2)
  (h_t_f : t_f = 0.5)
  (h_I_p : I_p = 3600)
  (h_I_s : I_s = 1200)
  (h_I_f : I_f = 300) :
  (I_p + I_s + I_f) / (n * (t_s + t_p + t_f)) = 106.25 := 
  by
  sorry

end dawn_hourly_income_l81_81110


namespace eval_expression_l81_81521

theorem eval_expression : 3 * (3 + 3) / 3 = 6 := by
  sorry

end eval_expression_l81_81521


namespace horner_example_l81_81092

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end horner_example_l81_81092


namespace smallest_divisible_fraction_l81_81396

theorem smallest_divisible_fraction :
  let nums := [8, 7, 15]
  let dens := [33, 22, 26]
  let lcm_nums := 120 -- LCM of the numerators
  let gcd_dens := 1   -- GCD of the denominators
  (forall f ∈ nums, lcm_nums % f = 0) ∧ (forall d ∈ dens, d % gcd_dens = 0) ->
  (lcm_nums : ℚ) / gcd_dens = 120 :=
by
  sorry

end smallest_divisible_fraction_l81_81396


namespace biology_books_needed_l81_81600

-- Define the problem in Lean
theorem biology_books_needed
  (B P Q R F Z₁ Z₂ : ℕ)
  (b p : ℝ)
  (H1 : B ≠ P)
  (H2 : B ≠ Q)
  (H3 : B ≠ R)
  (H4 : B ≠ F)
  (H5 : P ≠ Q)
  (H6 : P ≠ R)
  (H7 : P ≠ F)
  (H8 : Q ≠ R)
  (H9 : Q ≠ F)
  (H10 : R ≠ F)
  (H11 : 0 < B ∧ 0 < P ∧ 0 < Q ∧ 0 < R ∧ 0 < F)
  (H12 : Bb + Pp = Z₁)
  (H13 : Qb + Rp = Z₂)
  (H14 : Fb = Z₁)
  (H15 : Z₂ < Z₁) :
  F = (Q - B) / (P - R) :=
by
  sorry  -- Proof to be provided

end biology_books_needed_l81_81600


namespace hyperbola_vertex_distance_l81_81730

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l81_81730


namespace angle_of_inclination_45_l81_81267

def plane (x y z : ℝ) : Prop := (x = y) ∧ (y = z)
def image_planes (x y : ℝ) : Prop := (x = 45 ∧ y = 45)

theorem angle_of_inclination_45 (t₁₂ : ℝ) :
  ∃ θ: ℝ, (plane t₁₂ t₁₂ t₁₂ → image_planes 45 45 → θ = 45) :=
sorry

end angle_of_inclination_45_l81_81267


namespace binom_expansion_properties_l81_81770

-- Definitions and conditions definition as per part a):
def binom_expansion (a b : ℤ) (n : ℕ) : ℤ := (a - b) ^ n

-- Translating the proof problem into Lean 4:
theorem binom_expansion_properties (x : ℤ) (h : x ≠ 0) :
  let expansion := binom_expansion (1 / x) x 6 in
  ( ∃ k: ℕ, k = 4 ∧ nat.choose 6 k = list.max (list.map (λ i, nat.choose 6 i) (finset.range 7).toList) ) ∧
  ( expansion.eval (1) = 0 ) :=
by
  sorry

end binom_expansion_properties_l81_81770


namespace eden_bears_count_l81_81715

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end eden_bears_count_l81_81715


namespace length_of_each_side_is_25_nails_l81_81077

-- Definitions based on the conditions
def nails_per_side := 25
def total_nails := 96

-- The theorem stating the equivalent mathematical problem
theorem length_of_each_side_is_25_nails
  (n : ℕ) (h1 : n = nails_per_side * 4 - 4)
  (h2 : total_nails = 96):
  n = nails_per_side :=
by
  sorry

end length_of_each_side_is_25_nails_l81_81077


namespace sum_odd_even_integers_l81_81508

theorem sum_odd_even_integers :
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  odd_terms_sum + even_terms_sum = 335 :=
by
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  show odd_terms_sum + even_terms_sum = 335
  sorry

end sum_odd_even_integers_l81_81508


namespace machine_value_after_two_years_l81_81237

def machine_value (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - rate)^years

theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 :=
by
  sorry

end machine_value_after_two_years_l81_81237


namespace distribute_candies_l81_81859

theorem distribute_candies (candies kids : ℕ) (h1 : candies = 5) (h2 : kids = 3) :
  ∃ ways, ways = 6 ∧ (∀ k1 k2 k3, k1 + k2 + k3 = candies → 1 ≤ k1 ∧ 1 ≤ k2 ∧ 1 ≤ k3 → ways) :=
by
  use 6
  sorry

end distribute_candies_l81_81859


namespace difference_between_c_and_a_l81_81424

variables (a b c : ℝ)

theorem difference_between_c_and_a
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end difference_between_c_and_a_l81_81424


namespace express_a_in_terms_of_b_l81_81420

noncomputable def a : ℝ := Real.log 1250 / Real.log 6
noncomputable def b : ℝ := Real.log 50 / Real.log 3

theorem express_a_in_terms_of_b : a = (b + 0.6826) / 1.2619 :=
by
  sorry

end express_a_in_terms_of_b_l81_81420


namespace katie_initial_candies_l81_81323

theorem katie_initial_candies (K : ℕ) (h1 : K + 23 - 8 = 23) : K = 8 :=
sorry

end katie_initial_candies_l81_81323


namespace total_age_is_47_l81_81245

-- Define the ages of B and conditions
def B : ℕ := 18
def A : ℕ := B + 2
def C : ℕ := B / 2

-- Prove the total age of A, B, and C
theorem total_age_is_47 : A + B + C = 47 :=
by
  sorry

end total_age_is_47_l81_81245


namespace milk_butterfat_mixture_l81_81418

theorem milk_butterfat_mixture (x gallons_50 gall_10_perc final_gall mixture_perc: ℝ)
    (H1 : gall_10_perc = 24) 
    (H2 : mixture_perc = 0.20 * (x + gall_10_perc))
    (H3 : 0.50 * x + 0.10 * gall_10_perc = 0.20 * (x + gall_10_perc)) 
    (H4 : final_gall = 20) :
    x = 8 :=
sorry

end milk_butterfat_mixture_l81_81418


namespace roberts_monthly_expenses_l81_81010

-- Conditions
def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.1
def total_sales : ℝ := 23600
def savings_rate : ℝ := 0.2

-- Definitions derived from the conditions
noncomputable def commission : ℝ := commission_rate * total_sales
noncomputable def total_earnings : ℝ := basic_salary + commission
noncomputable def savings : ℝ := savings_rate * total_earnings
noncomputable def monthly_expenses : ℝ := total_earnings - savings

-- The statement to be proved
theorem roberts_monthly_expenses : monthly_expenses = 2888 := by
  sorry

end roberts_monthly_expenses_l81_81010


namespace cost_of_each_box_is_8_33_l81_81340

noncomputable def cost_per_box (boxes pens_per_box pens_packaged price_per_packaged price_per_set profit_total : ℕ) : ℝ :=
  let total_pens := boxes * pens_per_box
  let packaged_pens := pens_packaged * pens_per_box
  let packages := packaged_pens / 6
  let revenue_packages := packages * price_per_packaged
  let remaining_pens := total_pens - packaged_pens
  let sets := remaining_pens / 3
  let revenue_sets := sets * price_per_set
  let total_revenue := revenue_packages + revenue_sets
  let cost_total := total_revenue - profit_total
  cost_total / boxes

theorem cost_of_each_box_is_8_33 :
  cost_per_box 12 30 5 3 2 115 = 100 / 12 :=
by
  unfold cost_per_box
  sorry

end cost_of_each_box_is_8_33_l81_81340


namespace integer_solutions_count_l81_81280

theorem integer_solutions_count :
  ∃ (count : ℤ), (∀ (a : ℤ), 
  (∃ x : ℤ, x^2 + a * x + 8 * a = 0) ↔ count = 8) :=
sorry

end integer_solutions_count_l81_81280


namespace Louisa_average_speed_l81_81925

theorem Louisa_average_speed : 
  ∀ (v : ℝ), (∀ v, (160 / v) + 3 = (280 / v)) → v = 40 :=
by
  intros v h
  sorry

end Louisa_average_speed_l81_81925


namespace total_cost_is_100_l81_81615

-- Define the conditions as constants
constant shirt_count : ℕ := 10
constant pant_count : ℕ := shirt_count / 2
constant shirt_cost : ℕ := 6
constant pant_cost : ℕ := 8

-- Define the cost calculations
def total_shirt_cost : ℕ := shirt_count * shirt_cost
def total_pant_cost : ℕ := pant_count * pant_cost

-- Define the total cost calculation
def total_cost : ℕ := total_shirt_cost + total_pant_cost

-- Prove that the total cost is 100
theorem total_cost_is_100 : total_cost = 100 :=
by
  sorry

end total_cost_is_100_l81_81615


namespace units_digit_of_square_l81_81702

theorem units_digit_of_square (a b : ℕ) (h₁ : (10 * a + b) ^ 2 % 100 / 10 = 7) : b = 6 :=
sorry

end units_digit_of_square_l81_81702


namespace complex_seventh_root_of_unity_l81_81776

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end complex_seventh_root_of_unity_l81_81776


namespace mass_of_15_moles_is_9996_9_l81_81381

/-- Calculation of the molar mass of potassium aluminum sulfate dodecahydrate -/
def KAl_SO4_2_12H2O_molar_mass : ℝ :=
  let K := 39.10
  let Al := 26.98
  let S := 32.07
  let O := 16.00
  let H := 1.01
  K + Al + 2 * S + (8 + 24) * O + 24 * H

/-- Mass calculation for 15 moles of potassium aluminum sulfate dodecahydrate -/
def mass_of_15_moles_KAl_SO4_2_12H2O : ℝ :=
  15 * KAl_SO4_2_12H2O_molar_mass

/-- Proof statement that the mass of 15 moles of potassium aluminum sulfate dodecahydrate is 9996.9 grams -/
theorem mass_of_15_moles_is_9996_9 : mass_of_15_moles_KAl_SO4_2_12H2O = 9996.9 := by
  -- assume KAl_SO4_2_12H2O_molar_mass = 666.46 (from the problem solution steps)
  sorry

end mass_of_15_moles_is_9996_9_l81_81381


namespace range_of_m_l81_81416

open Real Set

def P (m : ℝ) := |m + 1| ≤ 2
def Q (m : ℝ) := ∃ x : ℝ, x^2 - m*x + 1 = 0 ∧ (m^2 - 4 ≥ 0)

theorem range_of_m (m : ℝ) :
  (¬¬ P m ∧ ¬ (P m ∧ Q m)) → -2 < m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l81_81416


namespace hens_count_l81_81958

theorem hens_count (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 140) : H = 22 :=
by
  sorry

end hens_count_l81_81958


namespace earliest_time_100_degrees_l81_81430

def temperature (t : ℝ) : ℝ := -t^2 + 15 * t + 40

theorem earliest_time_100_degrees :
  ∃ t : ℝ, temperature t = 100 ∧ (∀ t' : ℝ, temperature t' = 100 → t' ≥ t) :=
by
  sorry

end earliest_time_100_degrees_l81_81430


namespace evaluate_expression_l81_81391

theorem evaluate_expression : (7 - 3) ^ 2 + (7 ^ 2 - 3 ^ 2) = 56 := by
  sorry

end evaluate_expression_l81_81391


namespace dogs_count_l81_81219

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l81_81219


namespace min_value_of_bS_l81_81135

variable (n : ℕ)

noncomputable def a_n : ℝ := ∫ x in 0..n, (2 * x + 1)

noncomputable def S_n (a : ℕ → ℝ) : ℝ := ∑ i in Finset.range n,  1 / a (i + 1)

noncomputable def b_n (n : ℕ) : ℤ := n - 8

noncomputable def bS (b : ℕ → ℤ) (S : ℕ → ℝ) (n : ℕ) : ℝ := b n * S n

theorem min_value_of_bS :
    ∃ n : ℕ, bS b_n (S_n a_n) n = -4 :=
sorry

end min_value_of_bS_l81_81135


namespace pet_store_dogs_l81_81218

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l81_81218


namespace part1_beef_noodles_mix_sauce_purchased_l81_81967

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end part1_beef_noodles_mix_sauce_purchased_l81_81967


namespace fraction_sum_equals_l81_81549

theorem fraction_sum_equals : 
    (4 / 2) + (7 / 4) + (11 / 8) + (21 / 16) + (41 / 32) + (81 / 64) - 8 = 63 / 64 :=
by 
    sorry

end fraction_sum_equals_l81_81549


namespace PeytonManning_total_distance_l81_81338

noncomputable def PeytonManning_threw_distance : Prop :=
  let throw_distance_50 := 20
  let throw_times_sat := 20
  let throw_times_sun := 30
  let total_distance := 1600
  ∃ R : ℚ, 
    let throw_distance_80 := R * throw_distance_50
    let distance_sat := throw_distance_50 * throw_times_sat
    let distance_sun := throw_distance_80 * throw_times_sun
    distance_sat + distance_sun = total_distance

theorem PeytonManning_total_distance :
  PeytonManning_threw_distance := by
  sorry

end PeytonManning_total_distance_l81_81338


namespace rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l81_81405

-- Case 1
theorem rt_triangle_case1
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 30) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = 4) (hb : b = 4 * Real.sqrt 3) (hc : c = 8)
  : b = 4 * Real.sqrt 3 ∧ c = 8 := by
  sorry

-- Case 2
theorem rt_triangle_case2
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : B = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (ha : a = Real.sqrt 3 - 1) (hb : b = 3 - Real.sqrt 3) 
  (ha_b: A = 30)
  (h_c: c = 2 * Real.sqrt 3 - 2)
  : B = 60 ∧ A = 30 ∧ c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem rt_triangle_case3
  (a : ℝ) (b : ℝ) (c : ℝ) (A B C : ℝ)
  (h : A = 60) (h_bc : B + C = 90) (h_ac : A + C = 90)
  (hc : c = 2 + Real.sqrt 3)
  (ha : a = Real.sqrt 3 + 3/2) 
  (hb: b = (2 + Real.sqrt 3) / 2)
  : a = Real.sqrt 3 + 3/2 ∧ b = (2 + Real.sqrt 3) / 2 := by
  sorry

end rt_triangle_case1_rt_triangle_case2_rt_triangle_case3_l81_81405


namespace odd_function_increasing_ln_x_condition_l81_81285

theorem odd_function_increasing_ln_x_condition 
  {f : ℝ → ℝ} 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) 
  {x : ℝ} 
  (h_f_ln_x : f (Real.log x) < 0) : 
  0 < x ∧ x < 1 := 
sorry

end odd_function_increasing_ln_x_condition_l81_81285


namespace number_of_dogs_l81_81215

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l81_81215


namespace combined_votes_l81_81724

theorem combined_votes {A B : ℕ} (h1 : A = 14) (h2 : 2 * B = A) : A + B = 21 := 
by 
sorry

end combined_votes_l81_81724


namespace senya_mistakes_in_OCTAHEDRON_l81_81181

noncomputable def mistakes_in_word (word : String) : Nat :=
  if word = "TETRAHEDRON" then 5
  else if word = "DODECAHEDRON" then 6
  else if word = "ICOSAHEDRON" then 7
  else if word = "OCTAHEDRON" then 5 
  else 0

theorem senya_mistakes_in_OCTAHEDRON : mistakes_in_word "OCTAHEDRON" = 5 := by
  sorry

end senya_mistakes_in_OCTAHEDRON_l81_81181


namespace ratio_of_fifteenth_terms_l81_81325

def S (n : ℕ) : ℝ := sorry
def T (n : ℕ) : ℝ := sorry
def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

theorem ratio_of_fifteenth_terms 
  (h1: ∀ n, S n / T n = (5 * n + 3) / (3 * n + 35))
  (h2: ∀ n, a n = S n) -- Example condition
  (h3: ∀ n, b n = T n) -- Example condition
  : (a 15 / b 15) = 59 / 57 := 
  by 
  -- Placeholder proof
  sorry

end ratio_of_fifteenth_terms_l81_81325


namespace quadratic_inequality_solution_l81_81895

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : ∀ x, -3 < x ∧ x < 1/2 ↔ cx^2 + bx + a < 0) :
  ∀ x, -1/3 ≤ x ∧ x ≤ 2 ↔ ax^2 + bx + c ≥ 0 :=
sorry

end quadratic_inequality_solution_l81_81895


namespace ivan_score_more_than_5_points_l81_81605

/-- Definitions of the given conditions --/
def type_A_tasks : ℕ := 10
def type_B_probability : ℝ := 1/3
def type_B_points : ℕ := 2
def task_A_probability : ℝ := 1/4
def task_A_points : ℕ := 1
def more_than_5_points_probability : ℝ := 0.088

/-- Lean 4 statement equivalent to the math proof problem --/
theorem ivan_score_more_than_5_points:
  let P_A4 := ∑ i in finset.range (7 + 1), nat.choose type_A_tasks i * (task_A_probability ^ i) * ((1 - task_A_probability) ^ (type_A_tasks - i)) in
  let P_A6 := ∑ i in finset.range (11 - 6), nat.choose type_A_tasks (i + 6) * (task_A_probability ^ (i + 6)) * ((1 - task_A_probability) ^ (type_A_tasks - (i + 6))) in
  (P_A4 * type_B_probability + P_A6 * (1 - type_B_probability) = more_than_5_points_probability) := sorry

end ivan_score_more_than_5_points_l81_81605


namespace total_games_High_School_Nine_l81_81315

-- Define the constants and assumptions.
def num_teams := 9
def games_against_non_league := 6

-- Calculation of the number of games within the league.
def games_within_league := (num_teams * (num_teams - 1) / 2) * 2

-- Calculation of the number of games against non-league teams.
def games_non_league := num_teams * games_against_non_league

-- The total number of games.
def total_games := games_within_league + games_non_league

-- The statement to prove.
theorem total_games_High_School_Nine : total_games = 126 := 
by
  -- You do not need to provide the proof.
  sorry

end total_games_High_School_Nine_l81_81315


namespace correct_method_eliminates_y_l81_81149

def eliminate_y_condition1 (x y : ℝ) : Prop :=
  5 * x + 2 * y = 20

def eliminate_y_condition2 (x y : ℝ) : Prop :=
  4 * x - y = 8

theorem correct_method_eliminates_y (x y : ℝ) :
  eliminate_y_condition1 x y ∧ eliminate_y_condition2 x y →
  5 * x + 2 * y + 2 * (4 * x - y) = 36 :=
by
  sorry

end correct_method_eliminates_y_l81_81149


namespace max_consecutive_sum_leq_500_l81_81491

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l81_81491


namespace sphere_volume_l81_81842

/-- A sphere is perfectly inscribed in a cube. 
If the edge of the cube measures 10 inches, the volume of the sphere in cubic inches is \(\frac{500}{3}\pi\). -/
theorem sphere_volume (a : ℝ) (h : a = 10) : 
  ∃ V : ℝ, V = (4 / 3) * Real.pi * (a / 2)^3 ∧ V = (500 / 3) * Real.pi :=
by
  use (4 / 3) * Real.pi * (a / 2)^3
  sorry

end sphere_volume_l81_81842


namespace least_number_divisible_by_five_primes_l81_81038

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l81_81038


namespace brothers_complete_task_in_3_days_l81_81166

theorem brothers_complete_task_in_3_days :
  (1 / 4 + 1 / 12) * 3 = 1 :=
by
  sorry

end brothers_complete_task_in_3_days_l81_81166


namespace compute_tensor_operation_l81_81588

def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

theorem compute_tensor_operation :
  tensor (tensor 8 4) 2 = 202 / 9 :=
by
  sorry

end compute_tensor_operation_l81_81588


namespace find_k_l81_81692

def green_balls : ℕ := 7

noncomputable def probability_green (k : ℕ) : ℚ := green_balls / (green_balls + k)
noncomputable def probability_purple (k : ℕ) : ℚ := k / (green_balls + k)

noncomputable def winning_for_green : ℤ := 3
noncomputable def losing_for_purple : ℤ := -1

noncomputable def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * (winning_for_green : ℚ) + (probability_purple k) * (losing_for_purple : ℚ)

theorem find_k (k : ℕ) (h : expected_value k = 1) : k = 7 :=
  sorry

end find_k_l81_81692


namespace spinner_final_direction_l81_81026

-- Define the directions as an enumeration
inductive Direction
| north
| east
| south
| west

-- Convert between revolution fractions to direction
def direction_after_revolutions (initial : Direction) (revolutions : ℚ) : Direction :=
  let quarters := (revolutions * 4) % 4
  match initial with
  | Direction.south => if quarters == 0 then Direction.south
                       else if quarters == 1 then Direction.west
                       else if quarters == 2 then Direction.north
                       else Direction.east
  | Direction.east  => if quarters == 0 then Direction.east
                       else if quarters == 1 then Direction.south
                       else if quarters == 2 then Direction.west
                       else Direction.north
  | Direction.north => if quarters == 0 then Direction.north
                       else if quarters == 1 then Direction.east
                       else if quarters == 2 then Direction.south
                       else Direction.west
  | Direction.west  => if quarters == 0 then Direction.west
                       else if quarters == 1 then Direction.north
                       else if quarters == 2 then Direction.east
                       else Direction.south

-- Final proof statement
theorem spinner_final_direction : direction_after_revolutions Direction.south (4 + 3/4 - (6 + 1/2)) = Direction.east := 
by 
  sorry

end spinner_final_direction_l81_81026


namespace joe_lift_ratio_l81_81769

theorem joe_lift_ratio (F S : ℕ) 
  (h1 : F + S = 1800) 
  (h2 : F = 700) 
  (h3 : 2 * F = S + 300) : F / S = 7 / 11 :=
by
  sorry

end joe_lift_ratio_l81_81769


namespace initial_liquid_X_percentage_is_30_l81_81076

variable (initial_liquid_X_percentage : ℝ)

theorem initial_liquid_X_percentage_is_30
  (solution_total_weight : ℝ := 8)
  (initial_water_percentage : ℝ := 70)
  (evaporated_water_weight : ℝ := 3)
  (added_solution_weight : ℝ := 3)
  (new_liquid_X_percentage : ℝ := 41.25)
  (total_new_solution_weight : ℝ := 8)
  :
  initial_liquid_X_percentage = 30 :=
sorry

end initial_liquid_X_percentage_is_30_l81_81076


namespace vertical_distance_to_Felix_l81_81974

/--
  Dora is at point (8, -15).
  Eli is at point (2, 18).
  Felix is at point (5, 7).
  Calculate the vertical distance they need to walk to reach Felix.
-/
theorem vertical_distance_to_Felix :
  let Dora := (8, -15)
  let Eli := (2, 18)
  let Felix := (5, 7)
  let midpoint := ((Dora.1 + Eli.1) / 2, (Dora.2 + Eli.2) / 2)
  let vertical_distance := Felix.2 - midpoint.2
  vertical_distance = 5.5 :=
by
  sorry

end vertical_distance_to_Felix_l81_81974


namespace eggs_leftover_l81_81995

theorem eggs_leftover (d e f : ℕ) (total_eggs_per_carton : ℕ) 
  (h_d : d = 53) (h_e : e = 65) (h_f : f = 26) (h_carton : total_eggs_per_carton = 15) : (d + e + f) % total_eggs_per_carton = 9 :=
by {
  sorry
}

end eggs_leftover_l81_81995


namespace area_of_each_triangle_is_half_l81_81622

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def area (t : Triangle) : ℝ :=
  0.5 * |t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y)|

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 0 }
def C : Point := { x := 1, y := 1 }
def D : Point := { x := 0, y := 1 }
def K : Point := { x := 0.5, y := 1 }
def L : Point := { x := 0, y := 0.5 }
def M : Point := { x := 0.5, y := 0 }
def N : Point := { x := 1, y := 0.5 }

def AKB : Triangle := { p1 := A, p2 := K, p3 := B }
def BLC : Triangle := { p1 := B, p2 := L, p3 := C }
def CMD : Triangle := { p1 := C, p2 := M, p3 := D }
def DNA : Triangle := { p1 := D, p2 := N, p3 := A }

theorem area_of_each_triangle_is_half :
  area AKB = 0.5 ∧ area BLC = 0.5 ∧ area CMD = 0.5 ∧ area DNA = 0.5 := by sorry

end area_of_each_triangle_is_half_l81_81622


namespace gcd_irreducible_fraction_l81_81919

theorem gcd_irreducible_fraction (n : ℕ) (hn: 0 < n) : gcd (3*n + 1) (5*n + 2) = 1 :=
  sorry

end gcd_irreducible_fraction_l81_81919


namespace least_number_divisible_by_five_primes_l81_81039

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, (n > 0) ∧ (∀ p : ℕ, p ∈ {2, 3, 5, 7, 11} → p ∣ n) ∧ n = 2310 :=
begin
  sorry
end

end least_number_divisible_by_five_primes_l81_81039


namespace smallest_numbers_l81_81271

-- Define the problem statement
theorem smallest_numbers (m n : ℕ) :
  (∃ (m1 n1 m2 n2 : ℕ), 7 * m1^2 - 11 * n1^2 = 1 ∧ 7 * m2^2 - 11 * n2^2 = 5) ↔
  (7 * m^2 - 11 * n^2 = 1) ∨ (7 * m^2 - 11 * n^2 = 5) :=
by
  sorry

end smallest_numbers_l81_81271


namespace least_positive_number_divisible_by_five_primes_l81_81035

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l81_81035


namespace find_m_plus_n_l81_81029

def probability_no_exact_k_pairs (k n : ℕ) : ℚ :=
  -- A function to calculate the probability
  -- Placeholder definition (details omitted for brevity)
  sorry

theorem find_m_plus_n : ∃ m n : ℕ,
  gcd m n = 1 ∧ 
  (probability_no_exact_k_pairs k n = (97 / 1000) → m + n = 1097) :=
sorry

end find_m_plus_n_l81_81029


namespace least_multiple_of_five_primes_l81_81048

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l81_81048


namespace each_person_gets_equal_share_l81_81244

-- Definitions based on the conditions
def number_of_friends: Nat := 4
def initial_chicken_wings: Nat := 9
def additional_chicken_wings: Nat := 7

-- The proof statement
theorem each_person_gets_equal_share (total_chicken_wings := initial_chicken_wings + additional_chicken_wings) : 
       total_chicken_wings / number_of_friends = 4 := 
by 
  sorry

end each_person_gets_equal_share_l81_81244


namespace perimeter_difference_zero_l81_81993

theorem perimeter_difference_zero :
  let shape1_length := 4
  let shape1_width := 3
  let shape2_length := 6
  let shape2_width := 1
  let perimeter (l w : ℕ) := 2 * (l + w)
  perimeter shape1_length shape1_width = perimeter shape2_length shape2_width :=
by
  sorry

end perimeter_difference_zero_l81_81993


namespace game_returns_to_A_after_three_rolls_l81_81790

theorem game_returns_to_A_after_three_rolls :
  (∃ i j k : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ 1 ≤ k ∧ k ≤ 6 ∧ (i + j + k) % 12 = 0) → 
  true :=
by
  sorry

end game_returns_to_A_after_three_rolls_l81_81790


namespace probability_nina_taller_than_lena_is_zero_l81_81126

-- Definition of participants and conditions
variable (M N L O : ℝ)

-- Conditions
def condition1 := N < M
def condition2 := L > O

-- Statement: Given conditions, the probability that N > L is 0
theorem probability_nina_taller_than_lena_is_zero
  (h1 : condition1)
  (h2 : condition2) :
  (P : ℝ) = 0 :=
by
  sorry

end probability_nina_taller_than_lena_is_zero_l81_81126


namespace Hari_investment_contribution_l81_81339

noncomputable def Praveen_investment : ℕ := 3780
noncomputable def Praveen_time : ℕ := 12
noncomputable def Hari_time : ℕ := 7
noncomputable def profit_ratio : ℚ := 2 / 3

theorem Hari_investment_contribution :
  ∃ H : ℕ, (Praveen_investment * Praveen_time) / (H * Hari_time) = (2 : ℕ) / 3 ∧ H = 9720 :=
by
  sorry

end Hari_investment_contribution_l81_81339


namespace cost_of_pet_snake_l81_81283

theorem cost_of_pet_snake (original_amount : ℕ) (amount_left : ℕ) (cost : ℕ) 
  (h1 : original_amount = 73) (h2 : amount_left = 18) : cost = 55 :=
by
  sorry

end cost_of_pet_snake_l81_81283


namespace incorrect_operation_D_l81_81686

theorem incorrect_operation_D (x y: ℝ) : ¬ (-2 * x * (x - y) = -2 * x^2 - 2 * x * y) :=
by sorry

end incorrect_operation_D_l81_81686


namespace foci_ellipsoid_hyperboloid_l81_81804

theorem foci_ellipsoid_hyperboloid (a b : ℝ) 
(h1 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → dist (0,y) (0, 5) = 5)
(h2 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → dist (x,0) (7, 0) = 7) :
  |a * b| = Real.sqrt 444 := sorry

end foci_ellipsoid_hyperboloid_l81_81804


namespace radius_moon_scientific_notation_l81_81212

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end radius_moon_scientific_notation_l81_81212


namespace set_in_quadrant_I_l81_81025

theorem set_in_quadrant_I (x y : ℝ) (h1 : y ≥ 3 * x) (h2 : y ≥ 5 - x) (h3 : y < 7) : 
  x > 0 ∧ y > 0 :=
sorry

end set_in_quadrant_I_l81_81025


namespace max_consecutive_integers_sum_lt_500_l81_81493

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l81_81493


namespace square_roots_N_l81_81759

theorem square_roots_N (m N : ℤ) (h1 : (3 * m - 4) ^ 2 = N) (h2 : (7 - 4 * m) ^ 2 = N) : N = 25 := 
by
  sorry

end square_roots_N_l81_81759


namespace min_value_expr_l81_81875

theorem min_value_expr (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) : 
  (∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 = 2 ∧ |x| ≠ |y| → m ≤ (1 / (x + y)^2 + 1 / (x - y)^2)) ∧ m = 1) :=
by
  sorry

end min_value_expr_l81_81875


namespace original_denominator_l81_81703

theorem original_denominator (d : ℕ) (h : 3 * (d : ℚ) = 2) : d = 3 := 
by
  sorry

end original_denominator_l81_81703


namespace leak_drain_time_l81_81687

theorem leak_drain_time (P L : ℝ) (hP : P = 1/2) (h_combined : P - L = 3/7) : 1 / L = 14 :=
by
  -- Definitions of the conditions
  -- The rate of the pump filling the tank
  have hP : P = 1 / 2 := hP
  -- The combined rate of the pump (filling) and leak (draining)
  have h_combined : P - L = 3 / 7 := h_combined
  -- From these definitions, continue the proof
  sorry

end leak_drain_time_l81_81687


namespace evaluate_expression_l81_81726

theorem evaluate_expression (a b c : ℤ) 
  (h1 : c = b - 12) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2) * (b + 1) / (b - 3) * (c + 10) / (c + 7) = 10 / 3) :=
by
  sorry

end evaluate_expression_l81_81726


namespace find_k_l81_81327

-- Define point type and distances
structure Point :=
(x : ℝ)
(y : ℝ)

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition: H is the orthocenter of triangle ABC
variable (A B C H Q : Point)
variable (H_is_orthocenter : ∀ P : Point, dist P H = dist P A + dist P B - dist A B)

-- Prove the given equation
theorem find_k :
  dist Q A + dist Q B + dist Q C = 3 * dist Q H + dist H A + dist H B + dist H C :=
sorry

end find_k_l81_81327


namespace intersection_y_sum_zero_l81_81318

theorem intersection_y_sum_zero :
  ∀ (x1 y1 x2 y2 : ℝ), (y1 = 2 * x1) ∧ (y1 = 2 / x1) ∧ (y2 = 2 * x2) ∧ (y2 = 2 / x2) →
  (x2 = -x1) ∧ (y2 = -y1) →
  y1 + y2 = 0 :=
by
  sorry

end intersection_y_sum_zero_l81_81318


namespace only_integer_solution_is_zero_l81_81835

theorem only_integer_solution_is_zero (x y : ℤ) (h : x^4 + y^4 = 3 * x^3 * y) : x = 0 ∧ y = 0 :=
by {
  -- Here we would provide the proof steps.
  sorry
}

end only_integer_solution_is_zero_l81_81835


namespace topless_cubical_box_l81_81016

def squares : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

def valid_placement (s : Char) : Bool :=
  match s with
  | 'A' => true
  | 'B' => true
  | 'C' => true
  | 'D' => false
  | 'E' => false
  | 'F' => true
  | 'G' => true
  | 'H' => false
  | _ => false

def valid_configurations : List Char := squares.filter valid_placement

theorem topless_cubical_box:
  valid_configurations.length = 5 := by
  sorry

end topless_cubical_box_l81_81016


namespace time_to_cross_platform_l81_81085

-- Definitions from conditions
def train_speed_kmph : ℕ := 72
def speed_conversion_factor : ℕ := 1000 / 3600
def train_speed_mps : ℤ := train_speed_kmph * speed_conversion_factor
def time_cross_man_sec : ℕ := 16
def platform_length_meters : ℕ := 280

-- Proving the total time to cross platform
theorem time_to_cross_platform : ∃ t : ℕ, t = (platform_length_meters + (train_speed_mps * time_cross_man_sec)) / train_speed_mps ∧ t = 30 := 
by
  -- Since the proof isn't required, we add "sorry" to act as a placeholder.
  sorry

end time_to_cross_platform_l81_81085


namespace problem1_problem2_l81_81931

theorem problem1 (x y : ℝ) (h1 : x - y = 4) (h2 : x > 3) (h3 : y < 1) : 
  2 < x + y ∧ x + y < 6 :=
sorry

theorem problem2 (x y m : ℝ) (h1 : y > 1) (h2 : x < -1) (h3 : x - y = m) : 
  m + 2 < x + y ∧ x + y < -m - 2 :=
sorry

end problem1_problem2_l81_81931


namespace integer_solutions_of_equation_l81_81833

theorem integer_solutions_of_equation :
  ∀ (x y : ℤ), (x^4 + y^4 = 3 * x^3 * y) → (x = 0 ∧ y = 0) := by
  intros x y h
  sorry

end integer_solutions_of_equation_l81_81833


namespace math_expression_equivalent_l81_81100

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l81_81100


namespace calculate_fraction_l81_81477

theorem calculate_fraction : 
  ∃ f : ℝ, (14.500000000000002 ^ 2) * f = 126.15 ∧ f = 0.6 :=
by
  sorry

end calculate_fraction_l81_81477


namespace sum_of_fractions_is_514_l81_81569

theorem sum_of_fractions_is_514 : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 5 / 14 := 
by
  sorry

end sum_of_fractions_is_514_l81_81569


namespace fraction_shaded_area_l81_81282

theorem fraction_shaded_area (PX XQ : ℝ) (PA PR PQ : ℝ) (h1 : PX = 1) (h2 : 3 * XQ = PX) (h3 : PQ = PR) (h4 : PA = 1) (h5 : PA + AR = PR) (h6 : PR = 4):
  (3 / 16 : ℝ) = 0.375 :=
by
  -- proof here
  sorry

end fraction_shaded_area_l81_81282


namespace calculate_negative_subtraction_l81_81258

theorem calculate_negative_subtraction : -2 - (-3) = 1 :=
by sorry

end calculate_negative_subtraction_l81_81258


namespace prime_sum_divisors_l81_81122

theorem prime_sum_divisors (p : ℕ) (s : ℕ) : 
  (2 ≤ s ∧ s ≤ 10) → 
  (p = 2^s - 1) → 
  (p = 3 ∨ p = 7 ∨ p = 31 ∨ p = 127) :=
by
  intros h1 h2
  sorry

end prime_sum_divisors_l81_81122


namespace daily_construction_areas_minimum_area_A_must_build_l81_81243

-- Definitions based on conditions and questions
variable {area : ℕ}
variable {daily_A : ℕ}
variable {daily_B : ℕ}
variable (h_area : area = 5100)
variable (h_A_B_diff : daily_A = daily_B + 2)
variable (h_A_days : 900 / daily_A = 720 / daily_B)

-- Proof statements for the questions in the problem
theorem daily_construction_areas (daily_B : ℕ) (daily_A : ℕ) :
  daily_B = 8 ∧ daily_A = 10 :=
by sorry

theorem minimum_area_A_must_build (daily_A : ℕ) (daily_B : ℕ) (area_A : ℕ) :
  (area_A ≥ 2 * (5100 - area_A)) → (area_A ≥ 3400) :=
by sorry

end daily_construction_areas_minimum_area_A_must_build_l81_81243


namespace identity_solution_l81_81008

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l81_81008


namespace price_after_reductions_l81_81513

theorem price_after_reductions (P : ℝ) : ((P * 0.85) * 0.90) = P * 0.765 :=
by sorry

end price_after_reductions_l81_81513


namespace nine_tuples_satisfy_condition_l81_81772

noncomputable def num_satisfying_tuples : ℕ :=
  1 + (24 * 9 * 8) + (24 * Nat.choose 9 2 * Nat.choose 8 2) + 1

theorem nine_tuples_satisfy_condition :
  ∀ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ),
  (∀ (i j k : Fin 9),
    ∃ (l : Fin 9),
    l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ i < j ∧ j < k ∧ a_1 + a_2 + a_3 + a_4 = 100) →
  num_satisfying_tuples = 1 + (24 * 9 * 8) + (24 * Nat.choose 9 2 * Nat.choose 8 2) + 1 :=
sorry

end nine_tuples_satisfy_condition_l81_81772


namespace customers_tried_sample_l81_81083

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end customers_tried_sample_l81_81083


namespace ordered_pair_of_positive_integers_l81_81385

theorem ordered_pair_of_positive_integers :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^y + 4 = y^x) ∧ (3 * x^y = y^x + 10) ∧ (x = 7 ∧ y = 1) :=
by
  sorry

end ordered_pair_of_positive_integers_l81_81385


namespace least_positive_number_divisible_by_primes_l81_81033

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l81_81033


namespace sum_of_consecutive_integers_l81_81505

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l81_81505


namespace circle_point_outside_range_l81_81142

theorem circle_point_outside_range (m : ℝ) :
  ¬ (1 + 1 + 4 * m - 2 * 1 + 5 * m = 0) → 
  (m > 1 ∨ (0 < m ∧ m < 1 / 4)) := 
sorry

end circle_point_outside_range_l81_81142


namespace unique_solution_l81_81281

noncomputable def uniquely_solvable (a : ℝ) : Prop :=
  ∀ x : ℝ, a > 0 ∧ a ≠ 1 → ∃! x, a^x = (Real.log x / Real.log (1/4))

theorem unique_solution (a : ℝ) : a > 0 ∧ a ≠ 1 → uniquely_solvable a :=
by sorry

end unique_solution_l81_81281


namespace multiply_103_97_l81_81259

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end multiply_103_97_l81_81259


namespace dog_rabbit_age_ratio_l81_81028

-- Definitions based on conditions
def cat_age := 8
def rabbit_age := cat_age / 2
def dog_age := 12
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

-- Theorem statement
theorem dog_rabbit_age_ratio : is_multiple dog_age rabbit_age ∧ dog_age / rabbit_age = 3 :=
by
  sorry

end dog_rabbit_age_ratio_l81_81028


namespace system_solution_l81_81224

theorem system_solution (x y : ℤ) (h1 : x + y = 1) (h2 : 2*x + y = 5) : x = 4 ∧ y = -3 :=
by {
  sorry
}

end system_solution_l81_81224


namespace tim_words_per_day_l81_81949

variable (original_words : ℕ)
variable (years : ℕ)
variable (increase_percent : ℚ)

noncomputable def words_per_day (original_words : ℕ) (years : ℕ) (increase_percent : ℚ) : ℚ :=
  let increase_words := original_words * increase_percent
  let total_days := years * 365
  increase_words / total_days

theorem tim_words_per_day :
    words_per_day 14600 2 (50 / 100) = 10 := by
  sorry

end tim_words_per_day_l81_81949


namespace problem_statement_l81_81097

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l81_81097


namespace distance_between_foci_of_hyperbola_l81_81471

theorem distance_between_foci_of_hyperbola (x y : ℝ) (h : x * y = 4) : 
  distance (2, 2) (-2, -2) = 8 :=
sorry

end distance_between_foci_of_hyperbola_l81_81471


namespace area_of_rectangle_l81_81832

theorem area_of_rectangle
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h_a : a = 16)
  (h_c : c = 17)
  (h_diag : a^2 + b^2 = c^2) :
  abs (a * b - 91.9136) < 0.0001 :=
by
  sorry

end area_of_rectangle_l81_81832


namespace solve_for_y_l81_81642

theorem solve_for_y (y : ℝ) (h : 5^(3 * y) = Real.sqrt 125) : y = 1 / 2 :=
by sorry

end solve_for_y_l81_81642


namespace cube_surface_area_unchanged_l81_81691

def cubeSurfaceAreaAfterCornersRemoved
  (original_side : ℕ)
  (corner_side : ℕ)
  (original_surface_area : ℕ)
  (number_of_corners : ℕ)
  (surface_reduction_per_corner : ℕ)
  (new_surface_addition_per_corner : ℕ) : Prop :=
  (original_side * original_side * 6 = original_surface_area) →
  (corner_side * corner_side * 3 = surface_reduction_per_corner) →
  (corner_side * corner_side * 3 = new_surface_addition_per_corner) →
  original_surface_area - (number_of_corners * surface_reduction_per_corner) + (number_of_corners * new_surface_addition_per_corner) = original_surface_area
  
theorem cube_surface_area_unchanged :
  cubeSurfaceAreaAfterCornersRemoved 4 1 96 8 3 3 :=
by
  intro h1 h2 h3
  sorry

end cube_surface_area_unchanged_l81_81691


namespace probability_freedom_l81_81908

open Classical
open BigOperators

theorem probability_freedom :
  let DREAM := finset.range 5 -- letters in the word DREAM
  let FLIGHTS := finset.range 8 -- letters in the word FLIGHTS
  let DOOR := finset.range 4 -- letters in the word DOOR
  let p1 := 1 / (DREAM.choose 3).card
  let p2 := 1 / (FLIGHTS.choose 5).card
  let p3 := (DOOR.subset {0, 1, 2, 3}).card / (DOOR.choose 2).card
  p1 * p2 * p3 = 1 / 840 :=
by
  sorry

end probability_freedom_l81_81908


namespace identity_holds_l81_81002

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l81_81002


namespace unique_solution_fraction_l81_81269

theorem unique_solution_fraction (x : ℝ) :
  (2 * x^2 - 10 * x + 8 ≠ 0) → 
  (∃! (x : ℝ), (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4) :=
by
  sorry

end unique_solution_fraction_l81_81269


namespace jordan_final_weight_l81_81912

-- Defining the initial weight and the weight losses over the specified weeks
def initial_weight := 250
def loss_first_4_weeks := 4 * 3
def loss_next_8_weeks := 8 * 2
def total_loss := loss_first_4_weeks + loss_next_8_weeks

-- Theorem stating the final weight of Jordan
theorem jordan_final_weight : initial_weight - total_loss = 222 := by
  -- We skip the proof as requested
  sorry

end jordan_final_weight_l81_81912


namespace num_triangles_with_area_2_l81_81802

-- Define the grid and points
def is_grid_point (x y : ℕ) : Prop := x ≤ 3 ∧ y ≤ 3

-- Function to calculate the area of a triangle using vertices (x1, y1), (x2, y2), and (x3, y3)
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℕ) : ℤ := 
  (x1 * y2 + x2 * y3 + x3 * y1) 
  - (y1 * x2 + y2 * x3 + y3 * x1)

-- Check if the area is 2 (since we are dealing with a lattice grid, 
-- we can consider non-fractional form by multiplying by 2 to avoid half-area)
def has_area_2 (x1 y1 x2 y2 x3 y3 : ℕ) : Prop :=
  abs (area_of_triangle x1 y1 x2 y2 x3 y3) = 4

-- Define the main theorem that needs to be proved
theorem num_triangles_with_area_2 : 
  ∃ (n : ℕ), n = 64 ∧
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ), 
  is_grid_point x1 y1 ∧ is_grid_point x2 y2 ∧ is_grid_point x3 y3 ∧ 
  has_area_2 x1 y1 x2 y2 x3 y3 → n = 64 :=
sorry

end num_triangles_with_area_2_l81_81802


namespace megan_earnings_l81_81175

-- Define the given conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- Define the total number of necklaces
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

-- Define the total earnings
def total_earnings : ℕ := total_necklaces * cost_per_necklace

-- Prove that the total earnings are 90 dollars
theorem megan_earnings : total_earnings = 90 := by
  sorry

end megan_earnings_l81_81175


namespace number_of_blue_pens_minus_red_pens_is_seven_l81_81228

-- Define the problem conditions in Lean
variable (R B K T : ℕ) -- where R is red pens, B is black pens, K is blue pens, T is total pens

-- Define the hypotheses from the problem conditions
def hypotheses :=
  (R = 8) ∧ 
  (B = R + 10) ∧ 
  (T = 41) ∧ 
  (T = R + B + K)

-- Define the theorem we need to prove based on the question and the correct answer
theorem number_of_blue_pens_minus_red_pens_is_seven : 
  hypotheses R B K T → K - R = 7 :=
by 
  intro h
  sorry

end number_of_blue_pens_minus_red_pens_is_seven_l81_81228


namespace max_sides_of_convex_polygon_with_arithmetic_angles_l81_81952

theorem max_sides_of_convex_polygon_with_arithmetic_angles :
  ∀ (n : ℕ), (∃ α : ℝ, α > 0 ∧ α + (n - 1) * 1 < 180) → 
  n * (2 * α + (n - 1)) / 2 = (n - 2) * 180 → n ≤ 27 :=
by
  sorry

end max_sides_of_convex_polygon_with_arithmetic_angles_l81_81952


namespace maximize_profit_l81_81827

noncomputable def selling_price_to_maximize_profit (original_price selling_price : ℝ) (units units_sold_decrease : ℝ) : ℝ :=
  let x := 5
  let optimal_selling_price := selling_price + x
  optimal_selling_price

theorem maximize_profit :
  selling_price_to_maximize_profit 80 90 400 20 = 95 :=
by
  sorry

end maximize_profit_l81_81827


namespace tank_capacity_l81_81704

theorem tank_capacity (C : ℕ) 
  (h : 0.9 * (C : ℝ) - 0.4 * (C : ℝ) = 63) : C = 126 := 
by
  sorry

end tank_capacity_l81_81704


namespace number_of_photographs_is_twice_the_number_of_paintings_l81_81899

theorem number_of_photographs_is_twice_the_number_of_paintings (P Q : ℕ) :
  (Q * (Q - 1) * P) = 2 * (P * (Q * (Q - 1)) / 2) := by
  sorry

end number_of_photographs_is_twice_the_number_of_paintings_l81_81899


namespace sum_of_digits_l81_81434

theorem sum_of_digits (P Q R : ℕ) (hP : P < 10) (hQ : Q < 10) (hR : R < 10)
 (h_sum : P * 1000 + Q * 100 + Q * 10 + R = 2009) : P + Q + R = 10 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_l81_81434


namespace fraction_exponentiation_l81_81547

theorem fraction_exponentiation : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end fraction_exponentiation_l81_81547


namespace arithmetic_progression_15th_term_l81_81117

theorem arithmetic_progression_15th_term :
  let a := 2
  let d := 3
  let n := 15
  a + (n - 1) * d = 44 :=
by
  let a := 2
  let d := 3
  let n := 15
  sorry

end arithmetic_progression_15th_term_l81_81117


namespace subset_r_elements_max_m_l81_81780

open Finset Nat

theorem subset_r_elements_max_m (n r m : ℕ) (S : Finset ℕ) 
    (A : Finset (Finset ℕ)) (hS : S = (range n).filter (λ x, 1 ≤ x)) 
    (hm : m ≥ 2) 
    (hAi : ∀ Ai ∈ A, card Ai = r) 
    (hAj : ∀ Ai ∈ A, ∃ Aj ∈ A, Aj ≠ Ai ∧ ((Finset.max' Ai (nonempty_of_card_eq_succ (hAi Ai (by_ho_ad_filter Ai))).elim_left =  
                                                Finset.min' Aj (nonempty_of_card_eq_succ (hAi Aj (by_ho_ad_filter Aj))).elim_left) ∨ 
                                               (Finset.min' Ai (nonempty_of_card_eq_succ (hAi Ai (by_ho_ad_filter Ai))).elim_left = 
                                                Finset.max' Aj (nonempty_of_card_eq_succ (hAi Aj (by_ho_ad_filter Aj))).elim_left))) :
  2 ≤ r ∧ r ≤ (n + 1) / 2 ∧ m ≤ 2 * choose (n + 1) r - choose (n + 2 - 2 * r) r := 
sorry

end subset_r_elements_max_m_l81_81780


namespace hiker_speed_third_day_l81_81374

-- Define the conditions
def first_day_distance : ℕ := 18
def first_day_speed : ℕ := 3
def second_day_distance : ℕ :=
  let first_day_hours := first_day_distance / first_day_speed
  let second_day_hours := first_day_hours - 1
  let second_day_speed := first_day_speed + 1
  second_day_hours * second_day_speed
def total_distance : ℕ := 53
def third_day_hours : ℕ := 3

-- Define the speed on the third day based on given conditions
def speed_on_third_day : ℕ :=
  let third_day_distance := total_distance - first_day_distance - second_day_distance
  third_day_distance / third_day_hours

-- The theorem we need to prove
theorem hiker_speed_third_day : speed_on_third_day = 5 := by
  sorry

end hiker_speed_third_day_l81_81374


namespace even_number_divisible_by_8_l81_81421

theorem even_number_divisible_by_8 {n : ℤ} (h : ∃ k : ℤ, n = 2 * k) : 
  (n * (n^2 + 20)) % 8 = 0 ∧ 
  (n * (n^2 - 20)) % 8 = 0 ∧ 
  (n * (n^2 + 4)) % 8 = 0 ∧ 
  (n * (n^2 - 4)) % 8 = 0 :=
by
  sorry

end even_number_divisible_by_8_l81_81421


namespace product_of_numbers_l81_81482

variable {x y : ℝ}

theorem product_of_numbers (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 40 * k) : 
  x * y = 6400 / 63 := by
  sorry

end product_of_numbers_l81_81482


namespace complex_seventh_root_of_unity_l81_81777

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end complex_seventh_root_of_unity_l81_81777


namespace positive_reals_inequality_l81_81138

variable {a b c : ℝ}

theorem positive_reals_inequality (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a * b)^(1/4) + (b * c)^(1/4) + (c * a)^(1/4) < 1/4 := 
sorry

end positive_reals_inequality_l81_81138


namespace cannot_tile_with_sphinxes_l81_81362

def triangle_side_length : ℕ := 6
def small_triangles_count : ℕ := 36
def upward_triangles_count : ℕ := 21
def downward_triangles_count : ℕ := 15

theorem cannot_tile_with_sphinxes (n : ℕ) (small_triangles : ℕ) (upward : ℕ) (downward : ℕ) :
  n = triangle_side_length →
  small_triangles = small_triangles_count →
  upward = upward_triangles_count →
  downward = downward_triangles_count →
  (upward % 2 ≠ 0) ∨ (downward % 2 ≠ 0) →
  ¬ (upward + downward = small_triangles ∧
     ∀ k, (k * 6) ≤ small_triangles →
     ∃ u d, u + d = k * 6 ∧ u % 2 = 0 ∧ d % 2 = 0) := 
by
  intros
  sorry

end cannot_tile_with_sphinxes_l81_81362


namespace waiter_date_trick_l81_81980

theorem waiter_date_trick :
  ∃ d₂ : ℕ, ∃ x : ℝ, 
  (∀ d₁ : ℕ, ∀ x : ℝ, x + d₁ = 168) ∧
  3 * x + d₂ = 486 ∧
  3 * (x + d₂) = 516 ∧
  d₂ = 15 :=
by
  sorry

end waiter_date_trick_l81_81980


namespace difference_of_x_values_l81_81155

theorem difference_of_x_values : 
  ∀ x y : ℝ, ( (x + 3) ^ 2 / (3 * x + 29) = 2 ∧ (y + 3) ^ 2 / (3 * y + 29) = 2 ) → |x - y| = 14 := 
sorry

end difference_of_x_values_l81_81155


namespace graduate_degree_ratio_l81_81158

theorem graduate_degree_ratio (G C N : ℕ) (h1 : C = (2 / 3 : ℚ) * N)
  (h2 : (G : ℚ) / (G + C) = 0.15789473684210525) :
  (G : ℚ) / N = 1 / 8 :=
  sorry

end graduate_degree_ratio_l81_81158


namespace speed_of_stream_l81_81942

def upstream_speed (v : ℝ) := 72 - v
def downstream_speed (v : ℝ) := 72 + v

theorem speed_of_stream (v : ℝ) (h : 1 / upstream_speed v = 2 * (1 / downstream_speed v)) : v = 24 :=
by 
  sorry

end speed_of_stream_l81_81942


namespace least_positive_divisible_by_five_primes_l81_81049

-- Define the smallest 5 primes
def smallest_five_primes : List ℕ := [2, 3, 5, 7, 11]

-- Define the least positive whole number divisible by these primes
def least_positive_divisible_by_primes (primes : List ℕ) : ℕ :=
  primes.foldl (· * ·) 1

-- State the theorem
theorem least_positive_divisible_by_five_primes :
  least_positive_divisible_by_primes smallest_five_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_five_primes_l81_81049


namespace kickball_students_l81_81632

theorem kickball_students (w t : ℕ) (hw : w = 37) (ht : t = w - 9) : w + t = 65 :=
by
  sorry

end kickball_students_l81_81632


namespace average_and_difference_l81_81798

theorem average_and_difference
  (x y : ℚ) 
  (h1 : (15 + 24 + x + y) / 4 = 20)
  (h2 : x - y = 6) :
  x = 23.5 ∧ y = 17.5 := by
  sorry

end average_and_difference_l81_81798


namespace difference_in_pups_l81_81320

theorem difference_in_pups :
  let huskies := 5
  let pitbulls := 2
  let golden_retrievers := 4
  let pups_per_husky := 3
  let pups_per_pitbull := 3
  let total_adults := huskies + pitbulls + golden_retrievers
  let total_pups := total_adults + 30
  let total_husky_pups := huskies * pups_per_husky
  let total_pitbull_pups := pitbulls * pups_per_pitbull
  let H := pups_per_husky
  let D := (total_pups - total_husky_pups - total_pitbull_pups - 3 * golden_retrievers) / golden_retrievers
  D = 2 := sorry

end difference_in_pups_l81_81320


namespace students_on_playground_l81_81343

theorem students_on_playground (rows_left : ℕ) (rows_right : ℕ) (rows_front : ℕ) (rows_back : ℕ) (h1 : rows_left = 12) (h2 : rows_right = 11) (h3 : rows_front = 18) (h4 : rows_back = 8) :
    (rows_left + rows_right - 1) * (rows_front + rows_back - 1) = 550 := 
by
  sorry

end students_on_playground_l81_81343


namespace first_number_in_list_is_55_l81_81018

theorem first_number_in_list_is_55 : 
  ∀ (x : ℕ), (55 + 57 + 58 + 59 + 62 + 62 + 63 + 65 + x) / 9 = 60 → x = 65 → 55 = 55 :=
by
  intros x avg_cond x_is_65
  rfl

end first_number_in_list_is_55_l81_81018


namespace range_of_m_l81_81312

theorem range_of_m (m : ℝ) : (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x < 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y < 0) ↔ 0 < m ∧ m < 2 :=
by sorry

end range_of_m_l81_81312


namespace solve_for_y_l81_81643

theorem solve_for_y (y : ℝ) : 5^(3 * y) = real.sqrt 125 → y = 1 / 2 := by
  intro h
  apply eq_of_pow_eq_pow _ _
  exact h
sorry

end solve_for_y_l81_81643


namespace Jill_earnings_l81_81620

theorem Jill_earnings :
  let earnings_first_month := 10 * 30
  let earnings_second_month := 20 * 30
  let earnings_third_month := 20 * 15
  earnings_first_month + earnings_second_month + earnings_third_month = 1200 :=
by
  sorry

end Jill_earnings_l81_81620


namespace profits_equal_l81_81851

-- Define the profit variables
variables (profitA profitB profitC profitD : ℝ)

-- The conditions
def storeA_profit : profitA = 1.2 * profitB := sorry
def storeB_profit : profitB = 1.2 * profitC := sorry
def storeD_profit : profitD = profitA * 0.6 := sorry

-- The statement to be proven
theorem profits_equal : profitC = profitD :=
by sorry

end profits_equal_l81_81851


namespace log_base_27_of_3_l81_81561

theorem log_base_27_of_3 : log 27 3 = 1 / 3 :=
by
  have h1 : 27 = 3 ^ 3 := by norm_num
  have h2 : log 27 3 = (log 3 3) / 3 := by rw [log_pow, h1, log_div_log]
  have h3 : log 3 3 = 1 := by norm_num
  rw [h3, one_div, mul_one] at h2
  exact h2

end log_base_27_of_3_l81_81561


namespace janet_total_lives_l81_81689

/-
  Janet's initial lives: 38
  Lives lost: 16
  Lives gained: 32
  Prove that total lives == 54 after the changes
-/

theorem janet_total_lives (initial_lives lost_lives gained_lives : ℕ) 
(h1 : initial_lives = 38)
(h2 : lost_lives = 16)
(h3 : gained_lives = 32):
  initial_lives - lost_lives + gained_lives = 54 := by
  sorry

end janet_total_lives_l81_81689


namespace smallest_total_cashews_l81_81027

noncomputable def first_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  (2 * c1) / 3 + c2 / 6 + (4 * c3) / 18

noncomputable def second_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + (4 * c3) / 18

noncomputable def third_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + c3 / 9

theorem smallest_total_cashews : ∃ (c1 c2 c3 : ℕ), ∃ y : ℕ,
  3 * y = first_monkey_final c1 c2 c3 ∧
  2 * y = second_monkey_final c1 c2 c3 ∧
  y = third_monkey_final c1 c2 c3 ∧
  c1 + c2 + c3 = 630 :=
sorry

end smallest_total_cashews_l81_81027


namespace least_positive_number_divisible_by_five_primes_l81_81036

theorem least_positive_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧ 
    (∀ m : ℕ, (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m) → n ≤ m) :=
by
  use 2310
  split
  { exact by norm_num }
  split
  { intros p hp, fin_cases hp; norm_num }
  { intros m hm, apply nat.le_of_dvd; norm_num }
  sorry

end least_positive_number_divisible_by_five_primes_l81_81036


namespace find_center_of_circle_l81_81729

noncomputable def center_of_circle (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 + 4 * y = 16

theorem find_center_of_circle (x y : ℝ) (h : center_of_circle x y) : (x, y) = (4, -2) :=
by 
  sorry

end find_center_of_circle_l81_81729


namespace find_length_DE_l81_81669

theorem find_length_DE (AB AC BC : ℝ) (angleA : ℝ) 
                         (DE DF EF : ℝ) (angleD : ℝ) :
  AB = 9 → AC = 11 → BC = 7 →
  angleA = 60 → DE = 3 → DF = 5.5 → EF = 2.5 →
  angleD = 60 →
  DE = 9 * 2.5 / 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end find_length_DE_l81_81669


namespace find_all_quartets_l81_81917

def is_valid_quartet (a b c d : ℕ) : Prop :=
  a + b = c * d ∧
  a * b = c + d ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d

theorem find_all_quartets :
  ∀ (a b c d : ℕ),
  is_valid_quartet a b c d ↔
  (a, b, c, d) = (1, 5, 3, 2) ∨ 
  (a, b, c, d) = (1, 5, 2, 3) ∨ 
  (a, b, c, d) = (5, 1, 3, 2) ∨
  (a, b, c, d) = (5, 1, 2, 3) ∨ 
  (a, b, c, d) = (2, 3, 1, 5) ∨ 
  (a, b, c, d) = (3, 2, 1, 5) ∨ 
  (a, b, c, d) = (2, 3, 5, 1) ∨ 
  (a, b, c, d) = (3, 2, 5, 1) := by
  sorry

end find_all_quartets_l81_81917


namespace hyperbola_asymptotes_l81_81207

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
by
  sorry

end hyperbola_asymptotes_l81_81207


namespace scientific_notation_819000_l81_81902

theorem scientific_notation_819000 :
  ∃ (a : ℝ) (n : ℤ), 819000 = a * 10 ^ n ∧ a = 8.19 ∧ n = 5 :=
by
  -- Proof goes here
  sorry

end scientific_notation_819000_l81_81902


namespace identity_solution_l81_81006

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l81_81006


namespace simplify_expression_l81_81331

theorem simplify_expression (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x^3 + y^3 = 3 * (x + y)) :
  (x / y) + (y / x) - (3 / (x * y)) = 1 :=
by
  sorry

end simplify_expression_l81_81331


namespace identity_holds_l81_81004

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l81_81004


namespace remainder_of_pencils_l81_81890

def number_of_pencils : ℕ := 13254839
def packages : ℕ := 7

theorem remainder_of_pencils :
  number_of_pencils % packages = 3 := by
  sorry

end remainder_of_pencils_l81_81890


namespace geometric_series_sum_l81_81985

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h_a : a = 1) (h_r : r = 1 / 2) (h_n : n = 5) :
  ((a * (1 - r^n)) / (1 - r)) = 31 / 16 := 
by
  sorry

end geometric_series_sum_l81_81985


namespace max_consecutive_integers_sum_l81_81501

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l81_81501


namespace sum_infinite_series_l81_81712

noncomputable def series_term (n : ℕ) : ℚ := 
  (2 * n + 3) / (n * (n + 1) * (n + 2))

noncomputable def partial_fractions (n : ℕ) : ℚ := 
  (3 / 2) / n - 1 / (n + 1) - (1 / 2) / (n + 2)

theorem sum_infinite_series : 
  (∑' n : ℕ, series_term (n + 1)) = 5 / 4 := 
by
  sorry

end sum_infinite_series_l81_81712


namespace calc_molecular_weight_l81_81382

/-- Atomic weights in g/mol -/
def atomic_weight (e : String) : Float :=
  match e with
  | "Ca"   => 40.08
  | "O"    => 16.00
  | "H"    => 1.01
  | "Al"   => 26.98
  | "S"    => 32.07
  | "K"    => 39.10
  | "N"    => 14.01
  | _      => 0.0

/-- Molecular weight calculation for specific compounds -/
def molecular_weight (compound : String) : Float :=
  match compound with
  | "Ca(OH)2"     => atomic_weight "Ca" + 2 * atomic_weight "O" + 2 * atomic_weight "H"
  | "Al2(SO4)3"   => 2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")
  | "KNO3"        => atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"
  | _             => 0.0

/-- Given moles of different compounds, calculate the total molecular weight -/
def total_molecular_weight (moles : List (String × Float)) : Float :=
  moles.foldl (fun acc (compound, n) => acc + n * molecular_weight compound) 0.0

/-- The given problem -/
theorem calc_molecular_weight :
  total_molecular_weight [("Ca(OH)2", 4), ("Al2(SO4)3", 2), ("KNO3", 3)] = 1284.07 :=
by
  sorry

end calc_molecular_weight_l81_81382


namespace distance_between_foci_l81_81472

-- Let the hyperbola be defined by the equation xy = 4.
def hyperbola (x y : ℝ) : Prop := x * y = 4

-- Prove that the distance between the foci of this hyperbola is 8.
theorem distance_between_foci : ∀ (x y : ℝ), hyperbola x y → ∃ d, d = 8 :=
by {
    sorry
}

end distance_between_foci_l81_81472


namespace roof_problem_l81_81941

theorem roof_problem (w l : ℝ) (h1 : l = 4 * w) (h2 : l * w = 900) : l - w = 45 := 
by
  sorry

end roof_problem_l81_81941


namespace julia_total_watches_l81_81621

namespace JuliaWatches

-- Given conditions
def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def platinum_watches : ℕ := 2 * bronze_watches
def gold_watches : ℕ := (20 * (silver_watches + platinum_watches)) / 100  -- 20 is 20% and division by 100 to get the percentage

-- Proving the total watches Julia owns after the purchase
theorem julia_total_watches : silver_watches + bronze_watches + platinum_watches + gold_watches = 228 := by
  sorry

end JuliaWatches

end julia_total_watches_l81_81621


namespace probability_N_taller_than_L_l81_81134

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l81_81134


namespace quadratic_vertex_a_l81_81353

theorem quadratic_vertex_a
  (a b c : ℝ)
  (h1 : ∀ x, (a * x^2 + b * x + c = a * (x - 2)^2 + 5))
  (h2 : a * 0^2 + b * 0 + c = 0) :
  a = -5/4 :=
by
  -- Use the given conditions to outline the proof (proof not provided here as per instruction)
  sorry

end quadratic_vertex_a_l81_81353


namespace solve_pounds_l81_81994

def price_per_pound_corn : ℝ := 1.20
def price_per_pound_beans : ℝ := 0.60
def price_per_pound_rice : ℝ := 0.80
def total_weight : ℕ := 30
def total_cost : ℝ := 24.00
def equal_beans_rice (b r : ℕ) : Prop := b = r

theorem solve_pounds (c b r : ℕ) (h1 : price_per_pound_corn * ↑c + price_per_pound_beans * ↑b + price_per_pound_rice * ↑r = total_cost)
    (h2 : c + b + r = total_weight) (h3 : equal_beans_rice b r) : c = 6 ∧ b = 12 ∧ r = 12 := by
  sorry

end solve_pounds_l81_81994


namespace complex_projective_form_and_fixed_points_l81_81787

noncomputable def complex_projective_transformation (a b c d : ℂ) (z : ℂ) : ℂ :=
  (a * z + b) / (c * z + d)

theorem complex_projective_form_and_fixed_points (a b c d : ℂ) (h : d ≠ 0) :
  (∃ (f : ℂ → ℂ), ∀ z, f z = complex_projective_transformation a b c d z)
  ∧ ∃ (z₁ z₂ : ℂ), complex_projective_transformation a b c d z₁ = z₁ ∧ complex_projective_transformation a b c d z₂ = z₂ :=
by
  -- omitted proof, this is just the statement
  sorry

end complex_projective_form_and_fixed_points_l81_81787


namespace range_of_a_l81_81286

def p (a : ℝ) : Prop := a ≤ -4 ∨ a ≥ 4
def q (a : ℝ) : Prop := a ≥ -12
def either_p_or_q_but_not_both (a : ℝ) : Prop := (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a)

theorem range_of_a :
  {a : ℝ | either_p_or_q_but_not_both a} = {a : ℝ | (-4 < a ∧ a < 4) ∨ a < -12} :=
sorry

end range_of_a_l81_81286


namespace ratio_perimeters_of_squares_l81_81797

theorem ratio_perimeters_of_squares 
  (s₁ s₂ : ℝ)
  (h : (s₁ ^ 2) / (s₂ ^ 2) = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 :=
by
  sorry

end ratio_perimeters_of_squares_l81_81797


namespace locus_of_circle_center_l81_81574

theorem locus_of_circle_center (x y : ℝ) : 
    (exists C : ℝ × ℝ, (C.1, C.2) = (x,y)) ∧ 
    ((x - 0)^2 + (y - 3)^2 = r^2) ∧ 
    (y + 3 = 0) → x^2 = 12 * y :=
sorry

end locus_of_circle_center_l81_81574


namespace remainder_3_mod_6_l81_81598

theorem remainder_3_mod_6 (n : ℕ) (h : n % 18 = 3) : n % 6 = 3 :=
by
    sorry

end remainder_3_mod_6_l81_81598


namespace negation_of_proposition_l81_81415

theorem negation_of_proposition :
  (∀ x : ℝ, 2^x + x^2 > 0) → (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
sorry

end negation_of_proposition_l81_81415


namespace positive_difference_l81_81676

def a : ℝ := (7^2 + 7^2) / 7
def b : ℝ := (7^2 * 7^2) / 7

theorem positive_difference : |b - a| = 329 := by
  sorry

end positive_difference_l81_81676


namespace dogs_count_l81_81220

namespace PetStore

-- Definitions derived from the conditions
def ratio_cats_dogs := 3 / 4
def num_cats := 18
def num_groups := num_cats / 3
def num_dogs := 4 * num_groups

-- The statement to prove
theorem dogs_count : num_dogs = 24 :=
by
  sorry

end PetStore

end dogs_count_l81_81220


namespace part_one_part_two_l81_81292

noncomputable def f (x : ℝ) : ℝ := (3 * x) / (x + 1)

-- First part: Prove that f(x) is increasing on [2, 5]
theorem part_one (x₁ x₂ : ℝ) (hx₁ : 2 ≤ x₁) (hx₂ : x₂ ≤ 5) (h : x₁ < x₂) : f x₁ < f x₂ :=
by {
  -- Proof is to be filled in
  sorry
}

-- Second part: Find maximum and minimum of f(x) on [2, 5]
theorem part_two :
  f 2 = 2 ∧ f 5 = 5 / 2 :=
by {
  -- Proof is to be filled in
  sorry
}

end part_one_part_two_l81_81292


namespace abs_eq_5_iff_l81_81298

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l81_81298


namespace original_price_calculation_l81_81982

-- Definitions directly from problem conditions
def price_after_decrease (original_price : ℝ) : ℝ := 0.76 * original_price
def new_price : ℝ := 988

-- Statement embedding our problem
theorem original_price_calculation (x : ℝ) (hx : price_after_decrease x = new_price) : x = 1300 :=
by
  sorry

end original_price_calculation_l81_81982


namespace dima_walking_speed_l81_81722

def Dima_station_time := 18 * 60 -- in minutes
def Dima_actual_arrival := 17 * 60 + 5 -- in minutes
def car_speed := 60 -- in km/h
def early_arrival := 10 -- in minutes

def walking_speed (arrival_time actual_arrival car_speed early_arrival : ℕ) : ℕ :=
(car_speed * early_arrival / 60) * (60 / (arrival_time - actual_arrival - early_arrival))

theorem dima_walking_speed :
  walking_speed Dima_station_time Dima_actual_arrival car_speed early_arrival = 6 :=
sorry

end dima_walking_speed_l81_81722


namespace barnyard_owl_hoots_per_minute_l81_81634

theorem barnyard_owl_hoots_per_minute :
  (20 - 5) / 3 = 5 := 
by
  sorry

end barnyard_owl_hoots_per_minute_l81_81634


namespace multiple_of_people_l81_81342

-- Define the conditions
variable (P : ℕ) -- number of people who can do the work in 8 days

-- define a function that represents the work capacity of M * P people in days, 
-- we abstract away the solving steps into one declaration.

noncomputable def work_capacity (M P : ℕ) (days : ℕ) : ℚ :=
  M * (1/8) * days

-- Set up the problem to prove that the multiple of people is 2
theorem multiple_of_people (P : ℕ) : ∃ M : ℕ, work_capacity M P 2 = 1/2 :=
by
  use 2
  unfold work_capacity
  sorry

end multiple_of_people_l81_81342


namespace find_cost_price_l81_81700

variable (CP : ℝ)

def selling_price (CP : ℝ) := CP * 1.40

theorem find_cost_price (h : selling_price CP = 1680) : CP = 1200 :=
by
  sorry

end find_cost_price_l81_81700


namespace solution_set_of_equation_l81_81120

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solution_set_of_equation (x : ℝ) (h : x > 0): (x^(log_base 10 x) = x^3 / 100) ↔ (x = 10 ∨ x = 100) := 
by sorry

end solution_set_of_equation_l81_81120


namespace paths_from_A_to_B_l81_81695

-- Definitions based on the conditions given in part a)
def paths_from_red_to_blue : ℕ := 2
def paths_from_blue_to_green : ℕ := 3
def paths_from_green_to_purple : ℕ := 2
def paths_from_purple_to_orange : ℕ := 1
def num_red_arrows : ℕ := 2
def num_blue_arrows : ℕ := 2
def num_green_arrows : ℕ := 4
def num_purple_arrows : ℕ := 4
def num_orange_arrows : ℕ := 4

-- Prove the total number of distinct paths from A to B
theorem paths_from_A_to_B : 
  (paths_from_red_to_blue * num_red_arrows) * 
  (paths_from_blue_to_green * num_blue_arrows) * 
  (paths_from_green_to_purple * num_green_arrows) * 
  (paths_from_purple_to_orange * num_purple_arrows) = 16 := 
by sorry

end paths_from_A_to_B_l81_81695


namespace quadratic_equation_roots_sum_and_difference_l81_81867

theorem quadratic_equation_roots_sum_and_difference :
  ∃ (p q : ℝ), 
    p + q = 7 ∧ 
    |p - q| = 9 ∧ 
    (∀ x, (x - p) * (x - q) = x^2 - 7 * x - 8) :=
sorry

end quadratic_equation_roots_sum_and_difference_l81_81867


namespace mul_103_97_l81_81261

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end mul_103_97_l81_81261


namespace geometric_sequence_solve_a1_l81_81141

noncomputable def geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
    (h2 : a 2 = 1) (h3 : a 3 * a 9 = 2 * (a 5 ^ 2)) :=
  a 1 = (Real.sqrt 2) / 2

-- Define the main statement
theorem geometric_sequence_solve_a1 (a : ℕ → ℝ) (q : ℝ)
    (hq : 0 < q) (ha2 : a 2 = 1) (ha3_ha9 : a 3 * a 9 = 2 * (a 5 ^ 2)) :
    a 1 = (Real.sqrt 2) / 2 :=
sorry  -- The proof will be written here

end geometric_sequence_solve_a1_l81_81141


namespace find_a_value_l81_81371

theorem find_a_value :
  let center := (0.5, Real.sqrt 2)
  let line_dist (a : ℝ) := (abs (0.5 * a + Real.sqrt 2 - Real.sqrt 2)) / Real.sqrt (a^2 + 1)
  line_dist a = Real.sqrt 2 / 4 ↔ (a = 1 ∨ a = -1) :=
by
  sorry

end find_a_value_l81_81371


namespace shifted_parabola_eq_l81_81210

theorem shifted_parabola_eq :
  ∀ x, (∃ y, y = 2 * (x - 3)^2 + 2) →
       (∃ y, y = 2 * (x + 0)^2 + 4) :=
by sorry

end shifted_parabola_eq_l81_81210


namespace intersection_point_exists_l81_81819

noncomputable def line1 (t : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + 6 * t)
noncomputable def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3 * u)

theorem intersection_point_exists :
  ∃ t u : ℝ, line1 t = (1, 2) ∧ line2 u = (1, 2) := 
by
  sorry

end intersection_point_exists_l81_81819


namespace neg_distance_represents_west_l81_81156

def represents_east (distance : Int) : Prop :=
  distance > 0

def represents_west (distance : Int) : Prop :=
  distance < 0

theorem neg_distance_represents_west (pos_neg : represents_east 30) :
  represents_west (-50) :=
by
  sorry

end neg_distance_represents_west_l81_81156


namespace total_height_of_buildings_l81_81660

-- Definitions based on the conditions
def tallest_building : ℤ := 100
def second_tallest_building : ℤ := tallest_building / 2
def third_tallest_building : ℤ := second_tallest_building / 2
def fourth_tallest_building : ℤ := third_tallest_building / 5

-- Use the definitions to state the theorem
theorem total_height_of_buildings : 
  tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building = 180 := by
  sorry

end total_height_of_buildings_l81_81660


namespace sum_of_roots_is_three_l81_81763

theorem sum_of_roots_is_three :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) → x1 + x2 = 3 :=
by sorry

end sum_of_roots_is_three_l81_81763


namespace max_consecutive_sum_l81_81497

theorem max_consecutive_sum : ∃ (n : ℕ), (∀ m : ℕ, (m < n → (m * (m + 1)) / 2 < 500)) ∧ ¬((n * (n + 1)) / 2 < 500) :=
by {
  sorry
}

end max_consecutive_sum_l81_81497


namespace tangent_line_values_l81_81157

theorem tangent_line_values (m : ℝ) :
  (∃ s : ℝ, 3 * s^2 = 12 ∧ 12 * s + m = s^3 - 2) ↔ (m = -18 ∨ m = 14) :=
by
  sorry

end tangent_line_values_l81_81157


namespace six_n_digit_remains_divisible_by_7_l81_81182

-- Given the conditions
def is_6n_digit_number (N : ℕ) (n : ℕ) : Prop :=
  N < 10^(6*n) ∧ N ≥ 10^(6*(n-1))

def is_divisible_by_7 (N : ℕ) : Prop :=
  N % 7 = 0

-- Define new number M formed by moving the unit digit to the beginning
def new_number (N : ℕ) (n : ℕ) : ℕ :=
  let a_0 := N % 10
  let rest := N / 10
  a_0 * 10^(6*n - 1) + rest

-- The theorem statement
theorem six_n_digit_remains_divisible_by_7 (N : ℕ) (n : ℕ)
  (hN : is_6n_digit_number N n)
  (hDiv7 : is_divisible_by_7 N) : is_divisible_by_7 (new_number N n) :=
sorry

end six_n_digit_remains_divisible_by_7_l81_81182


namespace comparison1_comparison2_comparison3_l81_81108

theorem comparison1 : -3.2 > -4.3 :=
by sorry

theorem comparison2 : (1 : ℚ) / 2 > -(1 / 3) :=
by sorry

theorem comparison3 : (1 : ℚ) / 4 > 0 :=
by sorry

end comparison1_comparison2_comparison3_l81_81108


namespace sum_of_all_numbers_after_n_steps_l81_81089

def initial_sum : ℕ := 2

def sum_after_step (n : ℕ) : ℕ :=
  2 * 3^n

theorem sum_of_all_numbers_after_n_steps (n : ℕ) : 
  sum_after_step n = 2 * 3^n :=
by sorry

end sum_of_all_numbers_after_n_steps_l81_81089


namespace solve_for_x_l81_81641

theorem solve_for_x : ∀ (x : ℂ) (i : ℂ), i^2 = -1 → 3 - 2 * i * x = 6 + i * x → x = i :=
by
  intros x i hI2 hEq
  sorry

end solve_for_x_l81_81641


namespace science_and_technology_group_total_count_l81_81222

theorem science_and_technology_group_total_count 
  (number_of_girls : ℕ)
  (number_of_boys : ℕ)
  (h1 : number_of_girls = 18)
  (h2 : number_of_girls = 2 * number_of_boys - 2)
  : number_of_girls + number_of_boys = 28 := 
by
  sorry

end science_and_technology_group_total_count_l81_81222


namespace nina_not_taller_than_lena_l81_81132

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l81_81132


namespace binomial_expansion_calculation_l81_81710

theorem binomial_expansion_calculation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end binomial_expansion_calculation_l81_81710


namespace incircle_intersections_equation_l81_81907

-- Assume a triangle ABC with the given configuration
variables {A B C D E F M N : Type}

-- Incircle touches sides CA, AB at points E, F respectively
-- Lines BE and CF intersect the incircle again at points M and N respectively

theorem incircle_intersections_equation
  (triangle_ABC : Type)
  (incircle_I : Type)
  (touch_CA : Type)
  (touch_AB : Type)
  (intersect_BE : Type)
  (intersect_CF : Type)
  (E F : triangle_ABC → incircle_I)
  (M N : intersect_BE → intersect_CF)
  : 
  MN * EF = 3 * MF * NE :=
by 
  -- Sorry as the proof is omitted
  sorry

end incircle_intersections_equation_l81_81907


namespace probability_N_lt_L_is_zero_l81_81130

variable (M N L O : ℝ)
variable (hNM : N < M)
variable (hLO : L > O)

theorem probability_N_lt_L_is_zero : 
  (∃ (permutations : List (ℝ → ℝ)), 
  (∀ perm : ℝ → ℝ, perm ∈ permutations → N < M ∧ L > O) ∧ 
  ∀ perm : ℝ → ℝ, N > L) → false :=
by {
  sorry
}

end probability_N_lt_L_is_zero_l81_81130


namespace least_positive_x_l81_81783

variable (a b : ℝ)

noncomputable def tan_inv (x : ℝ) : ℝ := Real.arctan x

theorem least_positive_x (x k : ℝ) 
  (h1 : Real.tan x = a / b)
  (h2 : Real.tan (2 * x) = b / (a + b))
  (h3 : Real.tan (3 * x) = (a - b) / (a + b))
  (h4 : x = tan_inv k)
  : k = 13 / 9 := sorry

end least_positive_x_l81_81783


namespace trig_identity_proof_l81_81962

theorem trig_identity_proof : 
  (Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end trig_identity_proof_l81_81962


namespace total_eyes_in_extended_family_l81_81167

def mom_eyes := 1
def dad_eyes := 3
def kids_eyes := 3 * 4
def moms_previous_child_eyes := 5
def dads_previous_children_eyes := 6 + 2
def dads_ex_wife_eyes := 1
def dads_ex_wifes_new_partner_eyes := 7
def child_of_ex_wife_and_partner_eyes := 8

theorem total_eyes_in_extended_family :
  mom_eyes + dad_eyes + kids_eyes + moms_previous_child_eyes + dads_previous_children_eyes +
  dads_ex_wife_eyes + dads_ex_wifes_new_partner_eyes + child_of_ex_wife_and_partner_eyes = 45 :=
by
  -- add proof here
  sorry

end total_eyes_in_extended_family_l81_81167


namespace gcd_consecutive_term_max_l81_81991

def b (n : ℕ) : ℕ := n.factorial + 2^n + n 

theorem gcd_consecutive_term_max (n : ℕ) (hn : n ≥ 0) :
  ∃ m ≤ (n : ℕ), (m = 2) := sorry

end gcd_consecutive_term_max_l81_81991


namespace solve_r_l81_81341

variable (r : ℝ)

theorem solve_r : (r + 3) / (r - 2) = (r - 1) / (r + 1) → r = -1/7 := by
  sorry

end solve_r_l81_81341


namespace identity_holds_l81_81003

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l81_81003


namespace positive_difference_l81_81675

theorem positive_difference :
    let a := (7^2 + 7^2) / 7
    let b := (7^2 * 7^2) / 7
    abs (a - b) = 329 :=
by
  let a := (7^2 + 7^2) / 7
  let b := (7^2 * 7^2) / 7
  have ha : a = 14 := by sorry
  have hb : b = 343 := by sorry
  show abs (a - b) = 329
  from by
    rw [ha, hb]
    show abs (14 - 343) = 329 by norm_num
  

end positive_difference_l81_81675


namespace cos_theta_value_l81_81375

open Real

-- Define vectors v and w
def v : Fin 2 → ℝ := ![4, 5]
def w : Fin 2 → ℝ := ![2, 3]

-- Define the dot product function for 2D vectors
def dot_product (a b : Fin 2 → ℝ) : ℝ :=
  a 0 * b 0 + a 1 * b 1

-- Define the magnitude function for 2D vectors
def magnitude (a : Fin 2 → ℝ) : ℝ :=
  sqrt (a 0 * a 0 + a 1 * a 1)

-- Define θ as the acute angle between vectors v and w
def cos_theta : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

-- The theorem stating the result
theorem cos_theta_value : cos_theta = 23 / sqrt 533 := by
  sorry

end cos_theta_value_l81_81375


namespace sector_radius_l81_81465

theorem sector_radius (l : ℝ) (a : ℝ) (r : ℝ) (h1 : l = 2) (h2 : a = 4) (h3 : a = (1 / 2) * l * r) : r = 4 := by
  sorry

end sector_radius_l81_81465


namespace total_marble_weight_l81_81636

theorem total_marble_weight (w1 w2 w3 : ℝ) (h_w1 : w1 = 0.33) (h_w2 : w2 = 0.33) (h_w3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 :=
by {
  sorry
}

end total_marble_weight_l81_81636


namespace problem_l81_81781

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry
noncomputable def c : ℂ := sorry

def condition1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 := sorry
def condition2 : a + b + c = 30 := sorry
def condition3 : (a - b) ^ 2 + (a - c) ^ 2 + (b - c) ^ 2 = 2 * a * b * c := sorry

theorem problem : condition1 ∧ condition2 ∧ condition3 → (a^3 + b^3 + c^3) / (a * b * c) = 33 := 
by 
  sorry

end problem_l81_81781


namespace number_of_dogs_l81_81213

-- Conditions
def ratio_cats_dogs : ℚ := 3 / 4
def number_cats : ℕ := 18

-- Define the theorem to prove
theorem number_of_dogs : ∃ (dogs : ℕ), dogs = 24 :=
by
  -- Proof steps will go here, but we can use sorry for now to skip actual proving.
  sorry

end number_of_dogs_l81_81213


namespace compare_decimal_fraction_l81_81959

theorem compare_decimal_fraction : 0.8 - (1 / 2) = 0.3 := by
  sorry

end compare_decimal_fraction_l81_81959


namespace relay_race_total_time_correct_l81_81556

-- Conditions as definitions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25
def athlete5_time : ℕ := 80
def athlete6_time : ℕ := athlete5_time - 20
def athlete7_time : ℕ := 70
def athlete8_time : ℕ := athlete7_time - 5

-- Sum of all athletes' times
def total_time : ℕ :=
  athlete1_time + athlete2_time + athlete3_time + athlete4_time + athlete5_time +
  athlete6_time + athlete7_time + athlete8_time

-- Statement to prove
theorem relay_race_total_time_correct : total_time = 475 :=
  by
  sorry

end relay_race_total_time_correct_l81_81556


namespace real_solutions_l81_81863

theorem real_solutions (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) (h3 : x ≠ 5) :
  ( (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 3) * (x - 2) * (x - 1) ) / 
  ( (x - 2) * (x - 4) * (x - 5) * (x - 2) ) = 1 
  ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by sorry

end real_solutions_l81_81863


namespace simplify_expr1_simplify_expr2_l81_81186

variable (a b t : ℝ)

theorem simplify_expr1 : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l81_81186


namespace min_value_expression_l81_81403

theorem min_value_expression (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  ∃ (z : ℝ), z = (1 / (2 * x) + x / (y + 1)) ∧ z = 5 / 4 :=
sorry

end min_value_expression_l81_81403


namespace total_students_left_l81_81809

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end total_students_left_l81_81809


namespace find_a_l81_81172

-- Definitions based on conditions
def xi_distribution := NormalDist 3 2

-- Statement of the theorem
theorem find_a (a : ℝ) : 
  (μ (set_of (λ ω, xi_distribution.pdf ω < 2 * a - 3)) = μ (set_of (λ ω, xi_distribution.pdf ω > a + 2))) → 
  a = 7 / 3 :=
sorry

end find_a_l81_81172


namespace max_consecutive_sum_leq_500_l81_81490

theorem max_consecutive_sum_leq_500 : ∃ n : ℕ, (∑ i in finset.range (n+1), i) ≤ 500 ∧ (∀ m : ℕ, m > n → (∑ i in finset.range (m+1), i) > 500) :=
sorry

end max_consecutive_sum_leq_500_l81_81490


namespace minimum_n_is_835_l81_81866

def problem_statement : Prop :=
  ∀ (S : Finset ℕ), S.card = 835 → (∀ (T : Finset ℕ), T ⊆ S → T.card = 4 →
    ∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + 2 * b + 3 * c = d)

theorem minimum_n_is_835 : problem_statement :=
sorry

end minimum_n_is_835_l81_81866


namespace cost_price_of_ball_l81_81176

variable (C : ℝ)

theorem cost_price_of_ball (h : 15 * C - 720 = 5 * C) : C = 72 :=
by
  sorry

end cost_price_of_ball_l81_81176


namespace smallest_prime_perimeter_l81_81539

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

def is_prime_perimeter_scalene_triangle (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_scalene_triangle a b c ∧ is_prime (a + b + c)

theorem smallest_prime_perimeter (a b c : ℕ) :
  (a = 5 ∧ a < b ∧ a < c ∧ is_prime_perimeter_scalene_triangle a b c) →
  (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l81_81539


namespace pipe_empty_cistern_l81_81535

theorem pipe_empty_cistern (h : 1 / 3 * t = 6) : 2 / 3 * t = 12 :=
sorry

end pipe_empty_cistern_l81_81535


namespace eden_stuffed_bears_l81_81716

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end eden_stuffed_bears_l81_81716


namespace proof_problem_l81_81288

-- Conditions
def a : ℤ := 1
def b : ℤ := 0
def c : ℤ := -1 + 3

-- Proof Statement
theorem proof_problem : (2 * a + 3 * c) * b = 0 := by
  sorry

end proof_problem_l81_81288


namespace odd_function_b_value_f_monotonically_increasing_l81_81752

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x)

-- part (1): Prove that if y = f(x) is an odd function, then b = 1
theorem odd_function_b_value :
  (∀ x : ℝ, f x b + f (-x) b = 0) → b = 1 := sorry

-- part (2): Prove that y = f(x) is monotonically increasing for all x in ℝ given b = 1
theorem f_monotonically_increasing (b : ℝ) :
  b = 1 → ∀ x1 x2 : ℝ, x1 < x2 → f x1 b < f x2 b := sorry

end odd_function_b_value_f_monotonically_increasing_l81_81752


namespace greatest_integer_b_l81_81485

theorem greatest_integer_b (b : ℤ) :
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 ≠ -25) → b ≤ 10 :=
by
  intro
  sorry

end greatest_integer_b_l81_81485


namespace rowing_speed_in_still_water_l81_81376

noncomputable def speedInStillWater (distance_m : ℝ) (time_s : ℝ) (speed_current : ℝ) : ℝ :=
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let speed_downstream := distance_km / time_h
  speed_downstream - speed_current

theorem rowing_speed_in_still_water :
  speedInStillWater 45.5 9.099272058235341 8.5 = 9.5 :=
by
  sorry

end rowing_speed_in_still_water_l81_81376


namespace abs_diff_commute_times_l81_81534

theorem abs_diff_commute_times
  (x y : ℝ)
  (h_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (h_var : ((x - 10)^2 + (y - 10)^2 + (0)^2 + (1)^2 + (-1)^2) / 5 = 2) :
  |x - y| = 4 :=
by
  sorry

end abs_diff_commute_times_l81_81534


namespace find_ab_from_conditions_l81_81874

theorem find_ab_from_conditions (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := 
by
  sorry

end find_ab_from_conditions_l81_81874


namespace sine_thirteen_pi_over_six_l81_81550

theorem sine_thirteen_pi_over_six : Real.sin ((13 * Real.pi) / 6) = 1 / 2 := by
  sorry

end sine_thirteen_pi_over_six_l81_81550


namespace scientific_notation_of_819000_l81_81905

theorem scientific_notation_of_819000 : (819000 : ℝ) = 8.19 * 10^5 :=
by
  sorry

end scientific_notation_of_819000_l81_81905


namespace sum_of_volumes_of_two_cubes_l81_81869

-- Definitions for edge length and volume formula
def edge_length : ℕ := 5

def volume (s : ℕ) : ℕ := s ^ 3

-- Statement to prove the sum of volumes of two cubes with edge length 5 cm
theorem sum_of_volumes_of_two_cubes : volume edge_length + volume edge_length = 250 :=
by
  sorry

end sum_of_volumes_of_two_cubes_l81_81869


namespace distance_between_vertices_l81_81731

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l81_81731


namespace jellybean_probability_l81_81529

theorem jellybean_probability :
  ∀ (C : Fin 5 → ℕ) (sample : Fin 5 → ℕ), 
    (∀ i, C i = 1) -- each color is equally proportioned with an arbitrary unit
    → (sample 0 + sample 1 + sample 2 + sample 3 + sample 4 = 5) -- total of exactly 5 jellybeans in the sample
    → (∑ i, (sample i > 0).toNat = 2) -- exactly two distinct colors in the sample
    → (sample 0 ≠ 0 ∨ sample 1 ≠ 0 ∨ sample 2 ≠ 0 ∨ sample 3 ≠ 0 ∨ sample 4 ≠ 0) -- non-empty sample
    → ∃ P : ℚ, P = 12 / 125 := 
begin
  sorry
end

end jellybean_probability_l81_81529


namespace columbia_distinct_arrangements_l81_81857

theorem columbia_distinct_arrangements : 
  let total_letters := 8
  let repeat_I := 2
  let repeat_U := 2
  Nat.factorial total_letters / (Nat.factorial repeat_I * Nat.factorial repeat_U) = 90720 := by
  sorry

end columbia_distinct_arrangements_l81_81857


namespace buffalo_theft_l81_81728

theorem buffalo_theft (initial_apples falling_apples remaining_apples stolen_apples : ℕ)
  (h1 : initial_apples = 79)
  (h2 : falling_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - falling_apples - stolen_apples = remaining_apples ↔ stolen_apples = 45 :=
by sorry

end buffalo_theft_l81_81728


namespace sum_of_fractions_l81_81566

theorem sum_of_fractions :
  ∑ n in Finset.range 5, (1 : ℚ) / (n+2) / (n+3) = 5 / 14 := by
  sorry

end sum_of_fractions_l81_81566


namespace aquarium_water_ratio_l81_81923

theorem aquarium_water_ratio :
  let length := 4
  let width := 6
  let height := 3
  let volume := length * width * height
  let halfway_volume := volume / 2
  let water_after_cat := halfway_volume / 2
  let final_water := 54
  (final_water / water_after_cat) = 3 := by
  sorry

end aquarium_water_ratio_l81_81923


namespace mb_range_l81_81246
-- Define the slope m and y-intercept b
def m : ℚ := 2 / 3
def b : ℚ := -1 / 2

-- Define the product mb
def mb : ℚ := m * b

-- Prove the range of mb
theorem mb_range : -1 < mb ∧ mb < 0 := by
  unfold mb
  sorry

end mb_range_l81_81246


namespace total_area_of_WIN_sectors_l81_81372

theorem total_area_of_WIN_sectors (r : ℝ) (A_total : ℝ) (Prob_WIN : ℝ) (A_WIN : ℝ) : 
  r = 15 → 
  A_total = π * r^2 → 
  Prob_WIN = 3/7 → 
  A_WIN = Prob_WIN * A_total → 
  A_WIN = 3/7 * 225 * π :=
by {
  intros;
  sorry
}

end total_area_of_WIN_sectors_l81_81372


namespace octopus_leg_count_l81_81384

theorem octopus_leg_count :
  let num_initial_octopuses := 5
  let legs_per_normal_octopus := 8
  let num_removed_octopuses := 2
  let legs_first_mutant := 10
  let legs_second_mutant := 6
  let legs_third_mutant := 2 * legs_per_normal_octopus
  let num_initial_legs := num_initial_octopuses * legs_per_normal_octopus
  let num_removed_legs := num_removed_octopuses * legs_per_normal_octopus
  let num_mutant_legs := legs_first_mutant + legs_second_mutant + legs_third_mutant
  num_initial_legs - num_removed_legs + num_mutant_legs = 56 :=
by
  -- proof to be filled in later
  sorry

end octopus_leg_count_l81_81384


namespace least_number_divisible_by_five_smallest_primes_l81_81052

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l81_81052


namespace joshua_crates_l81_81441

def joshua_packs (b : ℕ) (not_packed : ℕ) (b_per_crate : ℕ) : ℕ :=
  (b - not_packed) / b_per_crate

theorem joshua_crates : joshua_packs 130 10 12 = 10 := by
  sorry

end joshua_crates_l81_81441


namespace log_27_3_l81_81558

noncomputable def log_base (a b : ℝ) : ℝ := Real.log a / Real.log b

theorem log_27_3 :
  log_base 3 27 = 1 / 3 := by
  sorry

end log_27_3_l81_81558


namespace abs_x_gt_1_iff_x_sq_minus1_gt_0_l81_81520

theorem abs_x_gt_1_iff_x_sq_minus1_gt_0 (x : ℝ) : (|x| > 1) ↔ (x^2 - 1 > 0) := by
  sorry

end abs_x_gt_1_iff_x_sq_minus1_gt_0_l81_81520


namespace ratio_of_graduate_to_non_graduate_l81_81159

variable (G C N : ℕ)

theorem ratio_of_graduate_to_non_graduate (h1 : C = (2:ℤ)*N/(3:ℤ))
                                         (h2 : G.toRat / (G + C) = 0.15789473684210525) :
  G.toRat / N.toRat = 1 / 8 :=
sorry

end ratio_of_graduate_to_non_graduate_l81_81159


namespace license_plate_palindrome_l81_81451

-- Define the components of the problem

def prob_digit_palindrome : ℚ := 1 / 100
def prob_letter_palindrome : ℚ := 13 / 8884

def prob_no_digit_palindrome : ℚ := 1 - prob_digit_palindrome
def prob_no_letter_palindrome : ℚ := 1 - prob_letter_palindrome
def prob_no_palindrome : ℚ := prob_no_digit_palindrome * prob_no_letter_palindrome

def prob_at_least_one_palindrome : ℚ := 1 - prob_no_palindrome

def reduced_fraction (pq : ℚ) : ℚ :=
  let num := pq.num.natAbs
  let den := pq.denom
  let g := Nat.gcd num den
  ⟨num / g, den / g⟩

def m : ℕ := (reduced_fraction prob_at_least_one_palindrome).num.natAbs
def n : ℕ := (reduced_fraction prob_at_least_one_palindrome).denom

def m_plus_n : ℕ := m + n

theorem license_plate_palindrome : m_plus_n = 897071 :=
by
  sorry

end license_plate_palindrome_l81_81451


namespace distance_travelled_l81_81891

def speed : ℕ := 3 -- speed in feet per second
def time : ℕ := 3600 -- time in seconds (1 hour)

theorem distance_travelled : speed * time = 10800 := by
  sorry

end distance_travelled_l81_81891


namespace equal_x_l81_81422

theorem equal_x (x y : ℝ) (h : x / (x + 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) :
  x = (2 * y^2 + 6 * y - 4) / 3 :=
sorry

end equal_x_l81_81422


namespace tiles_on_square_area_l81_81844

theorem tiles_on_square_area (n : ℕ) (h1 : 2 * n - 1 = 25) : n ^ 2 = 169 :=
by
  sorry

end tiles_on_square_area_l81_81844


namespace acute_angles_in_triangle_l81_81768

theorem acute_angles_in_triangle (α β γ : ℝ) (A_ext B_ext C_ext : ℝ) 
  (h_sum : α + β + γ = 180) 
  (h_ext1 : A_ext = 180 - β) 
  (h_ext2 : B_ext = 180 - γ) 
  (h_ext3 : C_ext = 180 - α) 
  (h_ext_acute1 : A_ext < 90 → β > 90) 
  (h_ext_acute2 : B_ext < 90 → γ > 90) 
  (h_ext_acute3 : C_ext < 90 → α > 90) : 
  ((α < 90 ∧ β < 90) ∨ (α < 90 ∧ γ < 90) ∨ (β < 90 ∧ γ < 90)) ∧ 
  ((A_ext < 90 → ¬ (B_ext < 90 ∨ C_ext < 90)) ∧ 
   (B_ext < 90 → ¬ (A_ext < 90 ∨ C_ext < 90)) ∧ 
   (C_ext < 90 → ¬ (A_ext < 90 ∨ B_ext < 90))) :=
sorry

end acute_angles_in_triangle_l81_81768


namespace sum_of_coefficients_eq_39_l81_81278

theorem sum_of_coefficients_eq_39 :
  5 * (2 * 1^8 - 3 * 1^3 + 4) - 6 * (1^6 + 4 * 1^3 - 9) = 39 :=
by
  sorry

end sum_of_coefficients_eq_39_l81_81278


namespace total_students_left_l81_81811

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end total_students_left_l81_81811


namespace part1_beef_noodles_mix_sauce_purchased_l81_81966

theorem part1_beef_noodles_mix_sauce_purchased (x y : ℕ) (h1 : x + y = 170) (h2 : 15 * x + 20 * y = 3000) :
  x = 80 ∧ y = 90 :=
sorry

end part1_beef_noodles_mix_sauce_purchased_l81_81966


namespace eight_digit_number_div_by_9_l81_81426

theorem eight_digit_number_div_by_9 (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 9)
  (h : (8 + 5 + 4 + n + 5 + 2 + 6 + 8) % 9 = 0) : n = 7 :=
by
  sorry

end eight_digit_number_div_by_9_l81_81426


namespace complex_root_product_value_l81_81778

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end complex_root_product_value_l81_81778


namespace area_to_paint_l81_81180

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def bookshelf_height : ℕ := 3
def bookshelf_length : ℕ := 5

theorem area_to_paint : (wall_height * wall_length) - (bookshelf_height * bookshelf_length) = 135 :=
by 
  sorry

end area_to_paint_l81_81180


namespace quadratic_solution_l81_81654

theorem quadratic_solution (x : ℝ) : 2 * x * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 3 / 2) := by
  sorry

end quadratic_solution_l81_81654


namespace a1d1_a2d2_a3d3_eq_neg1_l81_81326

theorem a1d1_a2d2_a3d3_eq_neg1 (a1 a2 a3 d1 d2 d3 : ℝ) (h : ∀ x : ℝ, 
  x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 + 1)) : 
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end a1d1_a2d2_a3d3_eq_neg1_l81_81326


namespace amoeba_reproduction_time_l81_81543

/--
An amoeba reproduces by fission, splitting itself into two separate amoebae. 
It takes 8 days for one amoeba to divide into 16 amoebae. 

Prove that it takes 2 days for an amoeba to reproduce.
-/
theorem amoeba_reproduction_time (day_per_cycle : ℕ) (n_cycles : ℕ) 
  (h1 : n_cycles * day_per_cycle = 8)
  (h2 : 2^n_cycles = 16) : 
  day_per_cycle = 2 :=
by
  sorry

end amoeba_reproduction_time_l81_81543


namespace sum_of_remainders_l81_81956

theorem sum_of_remainders (d e f g : ℕ)
  (hd : d % 30 = 15)
  (he : e % 30 = 5)
  (hf : f % 30 = 10)
  (hg : g % 30 = 20) :
  (d + e + f + g) % 30 = 20 :=
by
  sorry

end sum_of_remainders_l81_81956


namespace clock_displays_unique_digits_minutes_l81_81194

def minutes_with_unique_digits (h1 h2 m1 m2 : ℕ) : Prop :=
  h1 ≠ h2 ∧ h1 ≠ m1 ∧ h1 ≠ m2 ∧ h2 ≠ m1 ∧ h2 ≠ m2 ∧ m1 ≠ m2

def count_unique_digit_minutes (total_minutes : ℕ) :=
  let range0_19 := 1200
  let valid_0_19 := 504
  let range20_23 := 240
  let valid_20_23 := 84
  valid_0_19 + valid_20_23 = total_minutes

theorem clock_displays_unique_digits_minutes :
  count_unique_digit_minutes 588 :=
  by
    sorry

end clock_displays_unique_digits_minutes_l81_81194


namespace rachel_study_time_l81_81638

-- Define the conditions
def pages_math := 2
def pages_reading := 3
def pages_biology := 10
def pages_history := 4
def pages_physics := 5
def pages_chemistry := 8

def total_pages := pages_math + pages_reading + pages_biology + pages_history + pages_physics + pages_chemistry

def percent_study_time_biology := 30
def percent_study_time_reading := 30

-- State the theorem
theorem rachel_study_time :
  percent_study_time_biology = 30 ∧ 
  percent_study_time_reading = 30 →
  (100 - (percent_study_time_biology + percent_study_time_reading)) = 40 :=
by
  sorry

end rachel_study_time_l81_81638


namespace parabola_ratio_l81_81247

noncomputable def AF_over_BF (p : ℝ) (h_p : p > 0) : ℝ :=
  let AF := 4 * p
  let x := (4 / 7) * p -- derived from solving the equation in the solution
  AF / x

theorem parabola_ratio (p : ℝ) (h_p : p > 0) : AF_over_BF p h_p = 7 :=
  sorry

end parabola_ratio_l81_81247


namespace probability_3_queens_or_at_least_2_aces_l81_81304

-- Definitions of drawing from a standard deck and probabilities involved
def num_cards : ℕ := 52
def num_queens : ℕ := 4
def num_aces : ℕ := 4

def probability_all_queens : ℚ := (4/52) * (3/51) * (2/50)
def probability_2_aces_1_non_ace : ℚ := (4/52) * (3/51) * (48/50)
def probability_3_aces : ℚ := (4/52) * (3/51) * (2/50)
def probability_at_least_2_aces : ℚ := (probability_2_aces_1_non_ace) + (probability_3_aces)

def total_probability : ℚ := probability_all_queens + probability_at_least_2_aces

-- Statement to be proved
theorem probability_3_queens_or_at_least_2_aces :
  total_probability = 220 / 581747 :=
sorry

end probability_3_queens_or_at_least_2_aces_l81_81304


namespace range_of_a_l81_81148

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * Real.cos (Real.pi / 2 - x)

theorem range_of_a (a : ℝ) (h_condition : f (2 * a ^ 2) + f (a - 3) + f 0 < 0) : -3/2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l81_81148


namespace cannot_be_computed_using_square_difference_l81_81682

theorem cannot_be_computed_using_square_difference (x : ℝ) :
  (x+1)*(1+x) ≠ (a + b)*(a - b) :=
by
  intro a b
  have h : (x + 1) * (1 + x) = (a + b) * (a - b) → false := sorry
  exact h

#align $

end cannot_be_computed_using_square_difference_l81_81682


namespace paco_ate_more_cookies_l81_81926

-- Define the number of cookies Paco originally had
def original_cookies : ℕ := 25

-- Define the number of cookies Paco ate
def eaten_cookies : ℕ := 5

-- Define the number of cookies Paco bought
def bought_cookies : ℕ := 3

-- Define the number of more cookies Paco ate than bought
def more_cookies_eaten_than_bought : ℕ := eaten_cookies - bought_cookies

-- Prove that Paco ate 2 more cookies than he bought
theorem paco_ate_more_cookies : more_cookies_eaten_than_bought = 2 := by
  sorry

end paco_ate_more_cookies_l81_81926


namespace range_of_f_x1_x2_l81_81412

theorem range_of_f_x1_x2 (a x1 x2 : ℝ) (h1 : 2 * (Real.exp 1 + Real.exp (-1)) < a)
  (h2 : a < 20 / 3) (hx_ext : f '' Set.Ioc (0:ℝ) a ∶ Set.Pair x1 x2)
  (hx_le : x1 < x2) :
  e^2 - (1/e^2) - 4 < f(x1) - f(x2) < 80/9 - 4* log 3 :=
by
  sorry

noncomputable def f (x : ℝ) := x^2 - a*x + 2 * log x

end range_of_f_x1_x2_l81_81412


namespace log5_square_simplification_l81_81509

noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

theorem log5_square_simplification : (log5 (7 * log5 25))^2 = (log5 14)^2 :=
by
  sorry

end log5_square_simplification_l81_81509


namespace four_diff_digits_per_day_l81_81199

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l81_81199


namespace sum_of_consecutive_integers_l81_81504

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) / 2 ≤ 500) :
  n = 31 ∨ ((32 * (32 + 1) / 2) ≤ 500) := 
sorry

end sum_of_consecutive_integers_l81_81504


namespace probability_same_color_is_117_200_l81_81313

/-- There are eight green balls, five red balls, and seven blue balls in a bag. 
    A ball is taken from the bag, its color recorded, then placed back in the bag.
    A second ball is taken and its color recorded. -/
def probability_two_balls_same_color : ℚ :=
  let pGreen := (8 : ℚ) / 20
  let pRed := (5 : ℚ) / 20
  let pBlue := (7 : ℚ) / 20
  pGreen^2 + pRed^2 + pBlue^2

theorem probability_same_color_is_117_200 : probability_two_balls_same_color = 117 / 200 := by
  sorry

end probability_same_color_is_117_200_l81_81313


namespace solution_set_g_lt_6_range_of_a_given_opposite_values_l81_81413

open Real

def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

theorem solution_set_g_lt_6 : {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
by
  sorry

theorem range_of_a_given_opposite_values :
  (∃ (x1 x2 : ℝ), f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
by
  sorry

end solution_set_g_lt_6_range_of_a_given_opposite_values_l81_81413


namespace jose_julia_completion_time_l81_81510

variable (J N L : ℝ)

theorem jose_julia_completion_time :
  J + N + L = 1/4 ∧
  J * (1/3) = 1/18 ∧
  N = 1/9 ∧
  L * (1/3) = 1/18 →
  1/J = 6 ∧ 1/L = 6 := sorry

end jose_julia_completion_time_l81_81510


namespace find_t_and_m_l81_81754

theorem find_t_and_m 
  (t m : ℝ) 
  (ineq : ∀ x : ℝ, x^2 - 3 * x + t < 0 ↔ 1 < x ∧ x < m) : 
  t = 2 ∧ m = 2 :=
sorry

end find_t_and_m_l81_81754


namespace doughnut_completion_time_l81_81066

noncomputable def time_completion : Prop :=
  let start_time : ℕ := 7 * 60 -- 7:00 AM in minutes
  let quarter_complete_time : ℕ := 10 * 60 + 20 -- 10:20 AM in minutes
  let efficiency_decrease_time : ℕ := 12 * 60 -- 12:00 PM in minutes
  let one_quarter_duration : ℕ := quarter_complete_time - start_time
  let total_time_before_efficiency_decrease : ℕ := 5 * 60 -- from 7:00 AM to 12:00 PM is 5 hours
  let remaining_time_without_efficiency : ℕ := 4 * one_quarter_duration - total_time_before_efficiency_decrease
  let adjusted_remaining_time : ℕ := remaining_time_without_efficiency * 10 / 9 -- decrease by 10% efficiency
  let total_job_duration : ℕ := total_time_before_efficiency_decrease + adjusted_remaining_time
  let completion_time := efficiency_decrease_time + adjusted_remaining_time
  completion_time = 21 * 60 + 15 -- 9:15 PM in minutes

theorem doughnut_completion_time : time_completion :=
  by 
    sorry

end doughnut_completion_time_l81_81066


namespace price_per_piece_l81_81924

variable (y : ℝ)

theorem price_per_piece (h : (20 + y - 12) * (240 - 40 * y) = 1980) :
  20 + y = 21 ∨ 20 + y = 23 :=
sorry

end price_per_piece_l81_81924


namespace total_distance_walked_l81_81757

noncomputable def hazel_total_distance : ℕ := 3

def distance_first_hour := 2  -- The distance traveled in the first hour (in kilometers)
def distance_second_hour := distance_first_hour * 2  -- The distance traveled in the second hour
def distance_third_hour := distance_second_hour / 2  -- The distance traveled in the third hour, with a 50% speed decrease

theorem total_distance_walked :
  distance_first_hour + distance_second_hour + distance_third_hour = 8 :=
  by
    sorry

end total_distance_walked_l81_81757


namespace polynomial_divisible_by_x_sub_a_squared_l81_81178

theorem polynomial_divisible_by_x_sub_a_squared (a x : ℕ) (n : ℕ) 
    (h : a ≠ 0) : ∃ q : ℕ → ℕ, x ^ n - n * a ^ (n - 1) * x + (n - 1) * a ^ n = (x - a) ^ 2 * q x := 
by 
  sorry

end polynomial_divisible_by_x_sub_a_squared_l81_81178


namespace compute_100p_plus_q_l81_81445

theorem compute_100p_plus_q
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 → 
                  x ≠ -4 → x ≠ -15 → x ≠ -p → x ≠ -q)
  (h2 : ∀ x : ℝ, (x + 2 * p) * (x + 4) * (x + 9) = 0 → 
                  x ≠ -q → x ≠ -15 → (x = -4 ∨ x = -9))
  : 100 * p + q = -191 := 
sorry

end compute_100p_plus_q_l81_81445


namespace general_term_seq_l81_81164

universe u

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

-- State the theorem
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, a n = 2^n - 1 :=
by
  sorry

end general_term_seq_l81_81164


namespace total_cost_is_100_l81_81614

-- Define the conditions as constants
constant shirt_count : ℕ := 10
constant pant_count : ℕ := shirt_count / 2
constant shirt_cost : ℕ := 6
constant pant_cost : ℕ := 8

-- Define the cost calculations
def total_shirt_cost : ℕ := shirt_count * shirt_cost
def total_pant_cost : ℕ := pant_count * pant_cost

-- Define the total cost calculation
def total_cost : ℕ := total_shirt_cost + total_pant_cost

-- Prove that the total cost is 100
theorem total_cost_is_100 : total_cost = 100 :=
by
  sorry

end total_cost_is_100_l81_81614


namespace distance_from_edge_l81_81701

theorem distance_from_edge (wall_width picture_width x : ℕ) (h_wall : wall_width = 24) (h_picture : picture_width = 4) (h_centered : x + picture_width + x = wall_width) : x = 10 := by
  -- Proof is omitted
  sorry

end distance_from_edge_l81_81701


namespace value_of_m_l81_81136

theorem value_of_m
  (m : ℝ)
  (a : ℝ × ℝ := (-1, 3))
  (b : ℝ × ℝ := (m, m - 2))
  (collinear : a.1 * b.2 = a.2 * b.1) :
  m = 1 / 2 :=
sorry

end value_of_m_l81_81136


namespace video_game_map_width_l81_81377

theorem video_game_map_width (volume length height : ℝ) (h1 : volume = 50)
                            (h2 : length = 5) (h3 : height = 2) :
  ∃ width : ℝ, volume = length * width * height ∧ width = 5 :=
by
  sorry

end video_game_map_width_l81_81377


namespace boys_and_girls_arrangement_l81_81945

/--
Given 3 boys and 3 girls arranged in a line such that students 
of the same gender are adjacent, the number of distinct arrangements is 72.
-/
theorem boys_and_girls_arrangement : 
  let boys := 3
  let girls := 3
  (boys * girls) * 2 * (∏ i in finset.range boys, i + 1) * (∏ i in finset.range girls, i + 1) = 72 := by
  sorry

end boys_and_girls_arrangement_l81_81945


namespace num_customers_who_tried_sample_l81_81079

theorem num_customers_who_tried_sample :
  ∀ (samples_per_box boxes_opened samples_left : ℕ), 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  let total_samples := samples_per_box * boxes_opened in
  let samples_used := total_samples - samples_left in
  samples_used = 235 :=
by 
  intros samples_per_box boxes_opened samples_left h_samples_per_box h_boxes_opened h_samples_left total_samples samples_used
  simp [h_samples_per_box, h_boxes_opened, h_samples_left]
  sorry

end num_customers_who_tried_sample_l81_81079


namespace percentage_decrease_to_gain_30_percent_profit_l81_81698

theorem percentage_decrease_to_gain_30_percent_profit
  (C : ℝ) (P : ℝ) (S : ℝ) (S_new : ℝ) 
  (C_eq : C = 60)
  (S_eq : S = 1.25 * C)
  (S_new_eq1 : S_new = S - 12.60)
  (S_new_eq2 : S_new = 1.30 * (C - P * C)) : 
  P = 0.20 := by
  sorry

end percentage_decrease_to_gain_30_percent_profit_l81_81698


namespace find_hourly_rate_l81_81174

-- Defining the conditions
def hours_worked : ℝ := 7.5
def overtime_factor : ℝ := 1.5
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 48

-- Proving the hourly rate
theorem find_hourly_rate (R : ℝ) (h : 7.5 * R + (10.5 - 7.5) * 1.5 * R = 48) : R = 4 := by
  sorry

end find_hourly_rate_l81_81174


namespace sum_digits_least_time_l81_81813

/-- Given 12 horses each running a lap in the k-th prime minute,
  return the sum of the digits of the least time (in minutes) at which at least 5 horses meet 
  simultaneously at the starting point -/

theorem sum_digits_least_time (T : ℕ) (first_12_primes : ∀ k : ℕ, k < 12 → Nat.Prime (nth_prime k)) :
  (∀ i ∈ [2, 3, 5, 7, 11], i ∣ T) ∧ ∀ d, d < T → ((∀ i ∈ [2, 3, 5, 7, 11], ¬ (i ∣ d)))
  → (T.digits 10).sum = 6 := sorry

end sum_digits_least_time_l81_81813


namespace abs_eq_5_iff_l81_81297

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l81_81297


namespace range_of_a_l81_81140

theorem range_of_a (f : ℝ → ℝ) (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2) (h_ineq : f (1 - a) < f (2 * a - 1)) : a < 2 / 3 :=
sorry

end range_of_a_l81_81140


namespace range_h_l81_81939

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem range_h (a b : ℝ) (h_range : Set.Ioo a b = Set.Icc 0 1) : a + b = 1 := by
  sorry

end range_h_l81_81939


namespace simplify_expr1_simplify_expr2_l81_81183

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l81_81183


namespace geometric_series_sum_l81_81563

noncomputable def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ :=
  a / (1 - r)

theorem geometric_series_sum : infinite_geometric_series_sum (1/2) (1/2) (by norm_num : |(1/2 : ℝ)| < 1) = 1 :=
  sorry

end geometric_series_sum_l81_81563


namespace max_consecutive_integers_sum_l81_81502

theorem max_consecutive_integers_sum (n : ℕ) : 
  (∃ n, (n * (n + 1) / 2 < 500) ∧ (∀ m > n, m * (m + 1) / 2 ≥ 500)) :=
by {
  use 31,
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    have h : m = 32 := by linarith,
    rw h,
    norm_num,
  },
  sorry
}

end max_consecutive_integers_sum_l81_81502


namespace actual_revenue_percent_of_projected_l81_81920

noncomputable def projected_revenue (R : ℝ) : ℝ := 1.2 * R
noncomputable def actual_revenue (R : ℝ) : ℝ := 0.75 * R

theorem actual_revenue_percent_of_projected (R : ℝ) :
  (actual_revenue R / projected_revenue R) * 100 = 62.5 :=
  sorry

end actual_revenue_percent_of_projected_l81_81920


namespace value_of_a_l81_81894

theorem value_of_a
  (a b : ℚ)
  (h1 : b / a = 4)
  (h2 : b = 18 - 6 * a) :
  a = 9 / 5 := by
  sorry

end value_of_a_l81_81894


namespace remainder_of_72nd_integers_div_by_8_is_5_l81_81332

theorem remainder_of_72nd_integers_div_by_8_is_5 (s : Set ℤ) (h₁ : ∀ x ∈ s, ∃ k : ℤ, x = 8 * k + r) 
  (h₂ : 573 ∈ (s : Set ℤ)) : 
  ∃ (r : ℤ), r = 5 :=
by
  sorry

end remainder_of_72nd_integers_div_by_8_is_5_l81_81332


namespace moles_of_HCl_used_l81_81576

theorem moles_of_HCl_used (moles_amyl_alcohol : ℕ) (moles_product : ℕ) : 
  moles_amyl_alcohol = 2 ∧ moles_product = 2 → moles_amyl_alcohol = 2 :=
by
  sorry

end moles_of_HCl_used_l81_81576


namespace baker_earnings_l81_81555

theorem baker_earnings:
  ∀ (cakes_sold pies_sold cake_price pie_price : ℕ),
  cakes_sold = 453 →
  pies_sold = 126 →
  cake_price = 12 →
  pie_price = 7 →
  cakes_sold * cake_price + pies_sold * pie_price = 6318 := 
by
  intros cakes_sold pies_sold cake_price pie_price h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end baker_earnings_l81_81555


namespace initial_capacity_of_drum_x_l81_81388

theorem initial_capacity_of_drum_x (C x : ℝ) (h_capacity_y : 2 * x = 2 * 0.75 * C) :
  x = 0.75 * C :=
sorry

end initial_capacity_of_drum_x_l81_81388


namespace distance_from_origin_to_point_l81_81767

def point : ℝ × ℝ := (12, -16)
def origin : ℝ × ℝ := (0, 0)

theorem distance_from_origin_to_point : 
  let (x1, y1) := origin
  let (x2, y2) := point 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 20 :=
by
  sorry

end distance_from_origin_to_point_l81_81767


namespace evaluate_expression_l81_81390

theorem evaluate_expression (x b : ℝ) (h : x = b + 4) : 2 * x - b + 5 = b + 13 := by
  sorry

end evaluate_expression_l81_81390


namespace negation_universal_proposition_l81_81023

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) → ∃ x : ℝ, x^2 - 2 * x + 1 < 0 :=
by sorry

end negation_universal_proposition_l81_81023


namespace ivan_scores_more_than_5_points_l81_81604

-- Definitions based on problem conditions
def typeA_problem_probability (correct_guesses : ℕ) (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  (Nat.choose total_tasks correct_guesses : ℚ) * (success_prob ^ correct_guesses) * (failure_prob ^ (total_tasks - correct_guesses))

def probability_A4 (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  ∑ i in Finset.range (total_tasks + 1), if i ≥ 4 then typeA_problem_probability i total_tasks success_prob failure_prob else 0

def probability_A6 (total_tasks : ℕ) (success_prob : ℚ) (failure_prob : ℚ) : ℚ :=
  ∑ i in Finset.range (total_tasks + 1), if i ≥ 6 then typeA_problem_probability i total_tasks success_prob failure_prob else 0

def final_probability (p_A4 : ℚ) (p_A6 : ℚ) (p_B : ℚ) : ℚ :=
  (p_A4 * p_B) + (p_A6 * (1 - p_B))

noncomputable def probability_ivan_scores_more_than_5 : ℚ :=
  let total_tasks := 10
  let success_prob := 1 / 4
  let failure_prob := 3 / 4
  let p_B := 1 / 3
  let p_A4 := probability_A4 total_tasks success_prob failure_prob
  let p_A6 := probability_A6 total_tasks success_prob failure_prob
  final_probability p_A4 p_A6 p_B

theorem ivan_scores_more_than_5_points : probability_ivan_scores_more_than_5 = 0.088 := 
  sorry

end ivan_scores_more_than_5_points_l81_81604


namespace ken_height_l81_81017

theorem ken_height 
  (height_ivan : ℝ) (height_jackie : ℝ) (height_ken : ℝ)
  (h1 : height_ivan = 175) (h2 : height_jackie = 175)
  (h_avg : (height_ivan + height_jackie + height_ken) / 3 = (height_ivan + height_jackie) / 2 * 1.04) :
  height_ken = 196 := 
sorry

end ken_height_l81_81017


namespace compare_neg_fractions_l81_81989

theorem compare_neg_fractions : (- (2 / 3) < - (1 / 2)) :=
sorry

end compare_neg_fractions_l81_81989


namespace flat_terrain_length_l81_81940

noncomputable def terrain_distance_equation (x y z : ℝ) : Prop :=
  (x + y + z = 11.5) ∧
  (x / 3 + y / 4 + z / 5 = 2.9) ∧
  (z / 3 + y / 4 + x / 5 = 3.1)

theorem flat_terrain_length (x y z : ℝ) 
  (h : terrain_distance_equation x y z) :
  y = 4 :=
sorry

end flat_terrain_length_l81_81940


namespace total_cost_of_books_l81_81758

-- Conditions from the problem
def C1 : ℝ := 350
def loss_percent : ℝ := 0.15
def gain_percent : ℝ := 0.19
def SP1 : ℝ := C1 - (loss_percent * C1) -- Selling price of the book sold at a loss
def SP2 : ℝ := SP1 -- Selling price of the book sold at a gain

-- Statement to prove the total cost
theorem total_cost_of_books : C1 + (SP2 / (1 + gain_percent)) = 600 := by
  sorry

end total_cost_of_books_l81_81758


namespace range_omega_for_three_zeros_l81_81410

theorem range_omega_for_three_zeros (ω : ℝ) (h : ω > 0)
  (h_three_zeros : ∃ a b c ∈ (Set.Icc (0 : ℝ) (2 * Real.pi)), a ≠ b ∧ b ≠ c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x ∈ (Set.Icc (0 : ℝ) (2 * Real.pi)), f x = 0 → x ∈ {a, b, c}) :
  2 ≤ ω ∧ ω < 3 :=
begin
  let f := λ x, Real.cos (ω * x) - 1,
  sorry
end

end range_omega_for_three_zeros_l81_81410


namespace number_of_players_in_tournament_l81_81645

theorem number_of_players_in_tournament (n : ℕ) (h : 2 * 30 = n * (n - 1)) : n = 10 :=
sorry

end number_of_players_in_tournament_l81_81645


namespace gcd_eight_digit_repeating_four_digit_l81_81706

theorem gcd_eight_digit_repeating_four_digit :
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) →
  Nat.gcd (10001 * n) (10001 * m) = 10001) :=
by
  intros n hn m hm
  sorry

end gcd_eight_digit_repeating_four_digit_l81_81706


namespace distance_swam_against_current_l81_81533

def swimming_speed_in_still_water : ℝ := 4
def speed_of_current : ℝ := 2
def time_taken_against_current : ℝ := 5

theorem distance_swam_against_current : ∀ distance : ℝ,
  (distance = (swimming_speed_in_still_water - speed_of_current) * time_taken_against_current) → distance = 10 :=
by
  intros distance h
  sorry

end distance_swam_against_current_l81_81533


namespace remainder_5_pow_100_mod_18_l81_81240

theorem remainder_5_pow_100_mod_18 : (5 ^ 100) % 18 = 13 := 
by
  -- We will skip the proof since only the statement is required.
  sorry

end remainder_5_pow_100_mod_18_l81_81240


namespace replace_stars_identity_l81_81001

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l81_81001


namespace log_a_plus_b_eq_zero_l81_81775

open Complex

noncomputable def a_b_expression : ℂ := (⟨2, 1⟩ / ⟨1, 1⟩ : ℂ)

noncomputable def a : ℝ := a_b_expression.re

noncomputable def b : ℝ := a_b_expression.im

theorem log_a_plus_b_eq_zero : log (a + b) = 0 := by
  sorry

end log_a_plus_b_eq_zero_l81_81775


namespace transformations_map_figure_l81_81263

noncomputable def count_transformations : ℕ := sorry

theorem transformations_map_figure :
  count_transformations = 3 :=
sorry

end transformations_map_figure_l81_81263


namespace lisa_interest_after_10_years_l81_81793

noncomputable def compounded_amount (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r) ^ n

theorem lisa_interest_after_10_years :
  let P := 2000
  let r := (2 : ℚ) / 100
  let n := 10
  let A := compounded_amount P r n
  A - P = 438 := by
    let P := 2000
    let r := (2 : ℚ) / 100
    let n := 10
    let A := compounded_amount P r n
    have : A - P = 438 := sorry
    exact this

end lisa_interest_after_10_years_l81_81793


namespace clock_display_four_different_digits_l81_81202

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l81_81202


namespace max_value_expr_l81_81937

theorem max_value_expr (a b c d : ℝ) (ha : -4 ≤ a ∧ a ≤ 4) (hb : -4 ≤ b ∧ b ≤ 4) (hc : -4 ≤ c ∧ c ≤ 4) (hd : -4 ≤ d ∧ d ≤ 4) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 72 :=
sorry

end max_value_expr_l81_81937


namespace ellipse_standard_equation_l81_81748

-- Define the conditions
def equation1 (x y : ℝ) : Prop := x^2 + (y^2 / 2) = 1
def equation2 (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def equation3 (x y : ℝ) : Prop := x^2 + (y^2 / 4) = 1
def equation4 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the points
def point1 (x y : ℝ) : Prop := (x = 1 ∧ y = 0)
def point2 (x y : ℝ) : Prop := (x = 0 ∧ y = 2)

-- Define the main theorem
theorem ellipse_standard_equation :
  (equation4 1 0 ∧ equation4 0 2) ↔
  ((equation1 1 0 ∧ equation1 0 2) ∨
   (equation2 1 0 ∧ equation2 0 2) ∨
   (equation3 1 0 ∧ equation3 0 2) ∨
   (equation4 1 0 ∧ equation4 0 2)) :=
by
  sorry

end ellipse_standard_equation_l81_81748


namespace payment_to_y_l81_81518

theorem payment_to_y (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 580) : Y = 263.64 :=
by
  sorry

end payment_to_y_l81_81518


namespace find_possible_values_for_P_l81_81448

theorem find_possible_values_for_P (x y P : ℕ) (h1 : x < y) :
  P = (x^3 - y) / (1 + x * y) → (P = 0 ∨ P ≥ 2) :=
by
  sorry

end find_possible_values_for_P_l81_81448


namespace quadratic_no_real_roots_l81_81800

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 :=
by
  sorry

end quadratic_no_real_roots_l81_81800


namespace total_formula_portions_l81_81459

def puppies : ℕ := 7
def feedings_per_day : ℕ := 3
def days : ℕ := 5

theorem total_formula_portions : 
  (feedings_per_day * days * puppies = 105) := 
by
  sorry

end total_formula_portions_l81_81459


namespace cos_omega_x_3_zeros_interval_l81_81411

theorem cos_omega_x_3_zeros_interval (ω : ℝ) (hω : ω > 0)
  (h3_zeros : ∃ a b c : ℝ, (0 ≤ a ∧ a ≤ 2 * Real.pi) ∧
    (0 ≤ b ∧ b ≤ 2 * Real.pi ∧ b ≠ a) ∧
    (0 ≤ c ∧ c ≤ 2 * Real.pi ∧ c ≠ a ∧ c ≠ b) ∧
    (∀ x : ℝ, (0 ≤ x ∧ x ≤ 2 * Real.pi) →
      (Real.cos (ω * x) - 1 = 0 ↔ x = a ∨ x = b ∨ x = c))) :
  2 ≤ ω ∧ ω < 3 :=
sorry

end cos_omega_x_3_zeros_interval_l81_81411


namespace exists_station_to_complete_loop_l81_81671

structure CircularHighway where
  fuel_at_stations : List ℝ -- List of fuel amounts at each station
  travel_cost : List ℝ -- List of travel costs between consecutive stations

def total_fuel (hw : CircularHighway) : ℝ :=
  hw.fuel_at_stations.sum

def total_travel_cost (hw : CircularHighway) : ℝ :=
  hw.travel_cost.sum

def sufficient_fuel (hw : CircularHighway) : Prop :=
  total_fuel hw ≥ 2 * total_travel_cost hw

noncomputable def can_return_to_start (hw : CircularHighway) (start_station : ℕ) : Prop :=
  -- Function that checks if starting from a specific station allows for a return
  sorry

theorem exists_station_to_complete_loop (hw : CircularHighway) (h : sufficient_fuel hw) : ∃ start_station, can_return_to_start hw start_station :=
  sorry

end exists_station_to_complete_loop_l81_81671


namespace rectangle_ratio_l81_81013

theorem rectangle_ratio (s : ℝ) (h : s > 0) :
    let large_square_side := 3 * s
    let rectangle_length := 3 * s
    let rectangle_width := 2 * s
    rectangle_length / rectangle_width = 3 / 2 := by
  sorry

end rectangle_ratio_l81_81013


namespace alex_basketball_points_l81_81316

theorem alex_basketball_points (f t s : ℕ) 
  (h : f + t + s = 40) 
  (points_scored : ℝ := 0.8 * f + 0.3 * t + s) :
  points_scored = 28 :=
sorry

end alex_basketball_points_l81_81316


namespace least_positive_number_divisible_by_primes_l81_81031

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l81_81031


namespace sequences_power_of_two_l81_81878

open scoped Classical

theorem sequences_power_of_two (n : ℕ) (a b : Fin n → ℚ)
  (h1 : (∃ i j, i < j ∧ a i = a j) → ∀ i, a i = b i)
  (h2 : {p | ∃ (i j : Fin n), i < j ∧ (a i + a j = p)} = {q | ∃ (i j : Fin n), i < j ∧ (b i + b j = q)})
  (h3 : ∃ i j, i < j ∧ a i ≠ b i) :
  ∃ k : ℕ, n = 2 ^ k := 
sorry

end sequences_power_of_two_l81_81878


namespace right_angled_triangle_count_in_pyramid_l81_81319

-- Define the cuboid and the triangular pyramid within it
variables (A B C D A₁ B₁ C₁ D₁ : Type)

-- Assume there exists a cuboid ABCD-A₁B₁C₁D₁
axiom cuboid : Prop

-- Define the triangular pyramid A₁-ABC
structure triangular_pyramid (A₁ A B C : Type) : Type :=
  (vertex₁ : A₁)
  (vertex₂ : A)
  (vertex₃ : B)
  (vertex4 : C)
  
-- The mathematical statement to prove: the number of right-angled triangles in A₁-ABC is 4
theorem right_angled_triangle_count_in_pyramid (A : Type) (B : Type) (C : Type) (A₁ : Type)
  (h_pyramid : triangular_pyramid A₁ A B C) (h_cuboid : cuboid) :
  ∃ n : ℕ, n = 4 :=
by
  sorry

end right_angled_triangle_count_in_pyramid_l81_81319


namespace total_votes_cast_l81_81528

theorem total_votes_cast (V : ℝ) (h1 : ∃ x : ℝ, x = 0.31 * V) (h2 : ∃ y : ℝ, y = x + 2451) :
  V = 6450 :=
by
  sorry

end total_votes_cast_l81_81528


namespace max_xy_l81_81582

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 16) : 
  xy ≤ 32 :=
sorry

end max_xy_l81_81582


namespace clock_four_different_digits_l81_81203

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l81_81203


namespace clock_display_four_different_digits_l81_81200

theorem clock_display_four_different_digits :
  (∑ t in finset.range (24*60), if (((t / 60).div1000 ≠ (t / 60).mod1000) ∧ 
    ((t / 60).div1000 ≠ (t % 60).div1000) ∧ ((t / 60).div1000 ≠ (t % 60).mod1000) ∧ 
    ((t / 60).mod1000 ≠ (t % 60).div1000) ∧ ((t / 60).mod1000 ≠ (t % 60).mod1000) ∧ 
    ((t % 60).div1000 ≠ (t % 60).mod1000)) then 1 else 0) = 588 :=
by
  sorry

end clock_display_four_different_digits_l81_81200


namespace abs_diff_two_numbers_l81_81656

theorem abs_diff_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 := by
  sorry

end abs_diff_two_numbers_l81_81656


namespace find_base_k_representation_l81_81578

theorem find_base_k_representation :
  ∃ (k : ℕ), k > 0 ∧ (4 * k + 5) * 143 = (k^2 - 1) * 11 := 
begin
  sorry
end

end find_base_k_representation_l81_81578


namespace abigail_savings_l81_81542

-- Define the parameters for monthly savings and number of months in a year.
def monthlySavings : ℕ := 4000
def numberOfMonthsInYear : ℕ := 12

-- Define the total savings calculation.
def totalSavings (monthlySavings : ℕ) (numberOfMonths : ℕ) : ℕ :=
  monthlySavings * numberOfMonths

-- State the theorem that we need to prove.
theorem abigail_savings : totalSavings monthlySavings numberOfMonthsInYear = 48000 := by
  sorry

end abigail_savings_l81_81542


namespace intersection_line_l81_81988

-- Define the equations of the circles in Cartesian coordinates.
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y = 0

-- The theorem to prove.
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → y + 4 * x = 0 :=
by
  sorry

end intersection_line_l81_81988


namespace find_n_l81_81483

theorem find_n (n : ℕ) :
  Int.lcm n 16 = 52 ∧ Nat.gcd n 16 = 8 → n = 26 :=
by
  sorry

end find_n_l81_81483


namespace expression_not_computable_by_square_difference_l81_81681

theorem expression_not_computable_by_square_difference (x : ℝ) :
  ¬ ((x + 1) * (1 + x) = (x + 1) * (x - 1) ∨
     (x + 1) * (1 + x) = (-x + 1) * (-x - 1) ∨
     (x + 1) * (1 + x) = (x + 1) * (-x + 1)) :=
by
  sorry

end expression_not_computable_by_square_difference_l81_81681


namespace ivan_prob_more_than_5_points_l81_81610

open ProbabilityTheory Finset

/-- Conditions -/
def prob_correct_A : ℝ := 1 / 4
def prob_correct_B : ℝ := 1 / 3
def prob_A (k : ℕ) : ℝ := 
(C(10, k) * (prob_correct_A ^ k) * ((1 - prob_correct_A) ^ (10 - k)))

/-- Probabilities for type A problems -/
def prob_A_4 := ∑ i in (range 7).filter (λ i, i ≥ 4), prob_A i
def prob_A_6 := ∑ i in (range 7).filter (λ i, i ≥ 6), prob_A i

/-- Final combined probability -/
def final_prob : ℝ := 
(prob_A_4 * prob_correct_B) + (prob_A_6 * (1 - prob_correct_B))

/-- Proof -/
theorem ivan_prob_more_than_5_points : 
  final_prob = 0.088 := by
    sorry

end ivan_prob_more_than_5_points_l81_81610


namespace probability_of_odd_number_l81_81251

theorem probability_of_odd_number (wedge1 wedge2 wedge3 wedge4 wedge5 : ℝ)
  (h_wedge1_split : wedge1/3 = wedge2) 
  (h_wedge2_twice_wedge1 : wedge2 = 2 * (wedge1/3))
  (h_wedge3 : wedge3 = 1/4)
  (h_wedge5 : wedge5 = 1/4)
  (h_total : wedge1/3 + wedge2 + wedge3 + wedge4 + wedge5 = 1) :
  wedge1/3 + wedge3 + wedge5 = 7 / 12 :=
by
  sorry

end probability_of_odd_number_l81_81251


namespace system_of_equations_solution_l81_81115

theorem system_of_equations_solution (b : ℝ) :
  (∀ (a : ℝ), ∃ (x y : ℝ), (x - 1)^2 + y^2 = 1 ∧ a * x + y = a * b) ↔ 0 ≤ b ∧ b ≤ 2 :=
by
  sorry

end system_of_equations_solution_l81_81115


namespace pet_store_dogs_l81_81216

theorem pet_store_dogs (cats dogs : ℕ) (h1 : 18 = cats) (h2 : 3 * dogs = 4 * cats) : dogs = 24 :=
by
  sorry

end pet_store_dogs_l81_81216


namespace identity_solution_l81_81007

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l81_81007


namespace time_to_drain_l81_81688

theorem time_to_drain (V R C : ℝ) (hV : V = 75000) (hR : R = 60) (hC : C = 0.80) : 
  (V * C) / R = 1000 := by
  sorry

end time_to_drain_l81_81688


namespace line_always_passes_through_fixed_point_l81_81631

theorem line_always_passes_through_fixed_point :
  ∀ m : ℝ, (m-1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  
  -- Proof would go here
  sorry

end line_always_passes_through_fixed_point_l81_81631


namespace min_value_a_l81_81414

theorem min_value_a (a : ℝ) : (∀ x : ℝ, a < x → 2 * x + 2 / (x - a) ≥ 7) → a ≥ 3 / 2 :=
by
  sorry

end min_value_a_l81_81414


namespace expression_divisible_by_41_l81_81179

theorem expression_divisible_by_41 (n : ℕ) : 41 ∣ (5 * 7^(2*(n+1)) + 2^(3*n)) :=
  sorry

end expression_divisible_by_41_l81_81179


namespace find_y_l81_81356

theorem find_y : 
  (6 + 10 + 14 + 22) / 4 = (15 + y) / 2 → y = 11 :=
by
  intros h
  sorry

end find_y_l81_81356


namespace single_digit_solution_l81_81697

theorem single_digit_solution :
  ∃ A : ℕ, A < 10 ∧ A^3 = 210 + A ∧ A = 6 :=
by
  existsi 6
  sorry

end single_digit_solution_l81_81697


namespace inequality_squares_l81_81400

theorem inequality_squares (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end inequality_squares_l81_81400


namespace identity_solution_l81_81009

theorem identity_solution (x : ℝ) :
  ∃ a b : ℝ, (2 * x + a) ^ 3 = 5 * x ^ 3 + (3 * x + b) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x ∧
             a = -1 ∧ b = 1 :=
by
  -- we can skip the proof as this is just a statement
  sorry

end identity_solution_l81_81009


namespace sufficient_but_not_necessary_condition_l81_81289

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x >= 3) → (x^2 - 2*x - 3 >= 0) ∧ ¬((x^2 - 2*x - 3 >= 0) → (x >= 3)) := by
  sorry

end sufficient_but_not_necessary_condition_l81_81289


namespace find_x_l81_81538

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) : 
  let l5 := log_base 5 x
  let l6 := log_base 6 x
  let l7 := log_base 7 x
  let surface_area := 2 * (l5 * l6 + l5 * l7 + l6 * l7)
  let volume := l5 * l6 * l7 
  (surface_area = 2 * volume) → x = 210 :=
by 
  sorry

end find_x_l81_81538


namespace total_students_left_l81_81808

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end total_students_left_l81_81808


namespace original_ratio_l81_81651

theorem original_ratio (F J : ℚ) (hJ : J = 180) (h_ratio : (F + 45) / J = 3 / 2) : F / J = 5 / 4 :=
by
  sorry

end original_ratio_l81_81651


namespace profit_percentage_calculation_l81_81544

def selling_price : ℝ := 120
def cost_price : ℝ := 96

theorem profit_percentage_calculation (sp cp : ℝ) (hsp : sp = selling_price) (hcp : cp = cost_price) : 
  ((sp - cp) / cp) * 100 = 25 := 
 by
  sorry

end profit_percentage_calculation_l81_81544


namespace winner_percentage_of_votes_l81_81900

theorem winner_percentage_of_votes (V W O : ℕ) (W_votes : W = 720) (won_by : W - O = 240) (total_votes : V = W + O) :
  (W * 100) / V = 60 :=
by
  sorry

end winner_percentage_of_votes_l81_81900


namespace ivan_score_more_than_5_points_l81_81606

/-- Definitions of the given conditions --/
def type_A_tasks : ℕ := 10
def type_B_probability : ℝ := 1/3
def type_B_points : ℕ := 2
def task_A_probability : ℝ := 1/4
def task_A_points : ℕ := 1
def more_than_5_points_probability : ℝ := 0.088

/-- Lean 4 statement equivalent to the math proof problem --/
theorem ivan_score_more_than_5_points:
  let P_A4 := ∑ i in finset.range (7 + 1), nat.choose type_A_tasks i * (task_A_probability ^ i) * ((1 - task_A_probability) ^ (type_A_tasks - i)) in
  let P_A6 := ∑ i in finset.range (11 - 6), nat.choose type_A_tasks (i + 6) * (task_A_probability ^ (i + 6)) * ((1 - task_A_probability) ^ (type_A_tasks - (i + 6))) in
  (P_A4 * type_B_probability + P_A6 * (1 - type_B_probability) = more_than_5_points_probability) := sorry

end ivan_score_more_than_5_points_l81_81606


namespace sum_of_fractions_l81_81567

theorem sum_of_fractions :
  ∑ n in Finset.range 5, (1 : ℚ) / (n+2) / (n+3) = 5 / 14 := by
  sorry

end sum_of_fractions_l81_81567


namespace math_problem_l81_81774

theorem math_problem 
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : ∀ x, (x < -2 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 )) :
  a + 2 * b + 3 * c = 86 :=
sorry

end math_problem_l81_81774


namespace integer_solutions_of_equation_l81_81834

theorem integer_solutions_of_equation :
  ∀ (x y : ℤ), (x^4 + y^4 = 3 * x^3 * y) → (x = 0 ∧ y = 0) := by
  intros x y h
  sorry

end integer_solutions_of_equation_l81_81834


namespace power_equation_l81_81419

theorem power_equation (p : ℕ) : 81^6 = 3^p → p = 24 :=
by
  intro h
  have h1 : 81 = 3^4 := by norm_num
  rw [h1] at h
  rw [pow_mul] at h
  norm_num at h
  exact eq_of_pow_eq_pow _ h

end power_equation_l81_81419


namespace john_bought_soap_l81_81771

theorem john_bought_soap (weight_per_bar : ℝ) (cost_per_pound : ℝ) (total_spent : ℝ) (h1 : weight_per_bar = 1.5) (h2 : cost_per_pound = 0.5) (h3 : total_spent = 15) : 
  total_spent / (weight_per_bar * cost_per_pound) = 20 :=
by
  -- The proof would go here
  sorry

end john_bought_soap_l81_81771


namespace expression_evaluation_l81_81104

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l81_81104


namespace congruent_rectangle_perimeter_l81_81649

theorem congruent_rectangle_perimeter (x y w l P : ℝ) 
  (h1 : x + 2 * w = 2 * y) 
  (h2 : x + 2 * l = y) 
  (hP : P = 2 * l + 2 * w) : 
  P = 3 * y - 2 * x :=
by sorry

end congruent_rectangle_perimeter_l81_81649


namespace smallest_n_exists_square_smallest_n_exists_cube_l81_81519

open Nat

-- Statement for part (a)
theorem smallest_n_exists_square (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^2) → (∃ (x y : ℕ), n = 3 ∧ (x * (x + 3) = y^2))) := sorry

-- Statement for part (b)
theorem smallest_n_exists_cube (n x y : ℕ) : (∀ n x y, (x * (x + n) = y^3) → (∃ (x y : ℕ), n = 2 ∧ (x * (x + 2) = y^3))) := sorry

end smallest_n_exists_square_smallest_n_exists_cube_l81_81519


namespace raspberry_package_cost_l81_81921

noncomputable def cost_of_raspberries (cost_strawberries : ℕ) (cost_heavy_cream : ℕ) (total_cost : ℕ) : ℕ := 
  total_cost - (cost_strawberries + 2 * cost_heavy_cream)

theorem raspberry_package_cost :
  let cost_of_one_2cup_package (S : ℕ) := 3
      cost_of_heavy_cream (H : ℕ) := 4
      cost_S := 2 * cost_of_one_2cup_package 3
      cost_H1 := cost_of_heavy_cream 4 / 2
      cost_H2 := cost_of_heavy_cream 4 / 2
      total_known_cost := cost_S + cost_H1 + cost_H2
      total_budget := 20 in
  cost_of_raspberries cost_S cost_H1 total_budget / 2 = 5 := by
  unfold cost_of_raspberries
  have h1 : cost_of_raspberries cost_S cost_H1 total_budget = total_budget - (cost_S + 2 * cost_H1),
    { unfold cost_of_raspberries },
  calc
    cost_of_raspberries cost_S cost_H1 total_budget
    = total_budget - (cost_S + 2 * cost_H1) : by rw [h1]
    ... = 20 - (6 + 4) : by norm_num
    ... = 10 : by norm_num
    ... / 2 = 5 : by norm_num

end raspberry_package_cost_l81_81921


namespace company_food_purchase_1_l81_81965

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end company_food_purchase_1_l81_81965


namespace speed_of_second_car_l81_81481

theorem speed_of_second_car
  (t : ℝ)
  (distance_apart : ℝ)
  (speed_first_car : ℝ)
  (speed_second_car : ℝ)
  (h_total_distance : distance_apart = t * speed_first_car + t * speed_second_car)
  (h_time : t = 2.5)
  (h_distance_apart : distance_apart = 310)
  (h_speed_first_car : speed_first_car = 60) :
  speed_second_car = 64 := by
  sorry

end speed_of_second_car_l81_81481


namespace vasya_floor_l81_81552

variable (first third vasyaFloor : ℕ)
variable (steps_petya steps_vasya steps_per_floor : ℕ)

-- Conditions
def petya_climbs : Prop := steps_petya = 36 ∧ third - first = 2
def vasya_climbs : Prop := steps_vasya = 72
def steps_per_floor_def : Prop := steps_per_floor = steps_petya / (third - first)

-- Prove Vasya lives on the 5th floor
theorem vasya_floor : petya_climbs ∧ vasya_climbs ∧ steps_per_floor_def → vasyaFloor = first + steps_vasya / steps_per_floor :=
by 
  -- Proof omitted
  sorry

end vasya_floor_l81_81552


namespace cement_mixture_weight_l81_81963

theorem cement_mixture_weight 
  (W : ℝ)
  (h1 : W = (2/5) * W + (1/6) * W + (1/10) * W + (1/8) * W + 12) :
  W = 57.6 := by
  sorry

end cement_mixture_weight_l81_81963


namespace sum_of_squares_iff_double_sum_of_squares_l81_81928

theorem sum_of_squares_iff_double_sum_of_squares (n : ℕ) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ a b : ℤ, 2 * n = a^2 + b^2) :=
sorry

end sum_of_squares_iff_double_sum_of_squares_l81_81928


namespace percentage_in_biology_is_correct_l81_81530

/-- 
There are 840 students at a college.
546 students are not enrolled in a biology class.
We need to show what percentage of students are enrolled in biology classes.
--/

def num_students := 840
def not_in_biology := 546

def percentage_in_biology : ℕ := 
  ((num_students - not_in_biology) * 100) / num_students

theorem percentage_in_biology_is_correct : percentage_in_biology = 35 := 
  by
    -- proof is skipped
    sorry

end percentage_in_biology_is_correct_l81_81530


namespace like_terms_powers_eq_l81_81591

theorem like_terms_powers_eq (m n : ℕ) :
  (-2 : ℝ) * (x : ℝ) * (y : ℝ) ^ m = (1 / 3 : ℝ) * (x : ℝ) ^ n * (y : ℝ) ^ 3 → m = 3 ∧ n = 1 :=
by
  sorry

end like_terms_powers_eq_l81_81591


namespace trajectory_of_M_constant_NA_dot_NB_l81_81973

theorem trajectory_of_M (x y : ℝ) (hx : x^2 + 2*y^2 = 4) : 
  (x^2 / 4 + y^2 / 2 = 1) := 
sorry

theorem constant_NA_dot_NB (x1 x2 y1 y2 n : ℝ) 
  (hx1x2_sum : x1 + x2 = -4 * (y1^2 / (2 * y1^2 + 1)))
  (hx1x2_prod : x1 * x2 = (2 * y1^2 - 4) / (2 * y1^2 + 1))
  (hN : n = -7/4) :
  (1 + y1^2) * (x1 * x2) + (y1^2 - n) * (x1 + x2) + y1^2 + n^2 = -15/16 :=
sorry

end trajectory_of_M_constant_NA_dot_NB_l81_81973


namespace degrees_to_radians_300_l81_81386

theorem degrees_to_radians_300:
  (300 * (Real.pi / 180) = 5 * Real.pi / 3) := 
by
  repeat { sorry }

end degrees_to_radians_300_l81_81386


namespace point_quadrant_l81_81593

theorem point_quadrant (a b : ℝ) (h : |a - 4| + (b + 3)^2 = 0) : b < 0 ∧ a > 0 := 
by {
  sorry
}

end point_quadrant_l81_81593


namespace jeans_cost_proof_l81_81256

def cheaper_jeans_cost (coat_price: Float) (backpack_price: Float) (shoes_price: Float) (subtotal: Float) (difference: Float): Float :=
  let known_items_cost := coat_price + backpack_price + shoes_price
  let jeans_total_cost := subtotal - known_items_cost
  let x := (jeans_total_cost - difference) / 2
  x

def more_expensive_jeans_cost (cheaper_price : Float) (difference: Float): Float :=
  cheaper_price + difference

theorem jeans_cost_proof : ∀ (coat_price backpack_price shoes_price subtotal difference : Float),
  coat_price = 45 →
  backpack_price = 25 →
  shoes_price = 30 →
  subtotal = 139 →
  difference = 15 →
  cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference = 12 ∧
  more_expensive_jeans_cost (cheaper_jeans_cost coat_price backpack_price shoes_price subtotal difference) difference = 27 :=
by
  intros coat_price backpack_price shoes_price subtotal difference
  intros h1 h2 h3 h4 h5
  sorry

end jeans_cost_proof_l81_81256


namespace sum_max_min_values_of_g_l81_81330

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_max_min_values_of_g : (∀ x, 1 ≤ x ∧ x ≤ 7 → g x = 15 - 2 * x ∨ g x = 5) ∧ 
      (g 1 = 13 ∧ g 5 = 5)
      → (13 + 5 = 18) :=
by
  sorry

end sum_max_min_values_of_g_l81_81330


namespace abs_eq_5_iff_l81_81299

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l81_81299


namespace root_ratio_equiv_l81_81091

theorem root_ratio_equiv :
  (81 ^ (1 / 3)) / (81 ^ (1 / 4)) = 81 ^ (1 / 12) :=
by
  sorry

end root_ratio_equiv_l81_81091


namespace part1_part2_l81_81883

noncomputable def f (a x : ℝ) : ℝ := a * x - a * Real.log x - Real.exp x / x

theorem part1 (a : ℝ) :
  (∀ x > 0, f a x < 0) → a < Real.exp 1 :=
sorry

theorem part2 (a : ℝ) (x1 x2 x3 : ℝ) :
  (∀ x, f a x = 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧
  f a x1 + f a x2 + f a x3 ≤ 3 * Real.exp 2 - Real.exp 1 →
  Real.exp 1 < a ∧ a ≤ Real.exp 2 :=
sorry

end part1_part2_l81_81883


namespace sum_of_fractions_is_514_l81_81568

theorem sum_of_fractions_is_514 : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7)) = 5 / 14 := 
by
  sorry

end sum_of_fractions_is_514_l81_81568


namespace other_root_of_quadratic_l81_81310

theorem other_root_of_quadratic (m : ℝ) (root1 : ℝ) (h_roots : root1 = 2)
  (h_quadratic : ∀ x, x^2 + m * x - 6 = 0 ↔ x = root1 ∨ x = -3) : 
  ∃ root2 : ℝ, root2 = -3 :=
by
  use -3
  sorry

end other_root_of_quadratic_l81_81310


namespace cumulative_percentage_decrease_l81_81252

theorem cumulative_percentage_decrease :
  let original_price := 100
  let first_reduction := original_price * 0.85
  let second_reduction := first_reduction * 0.90
  let third_reduction := second_reduction * 0.95
  let fourth_reduction := third_reduction * 0.80
  let final_price := fourth_reduction
  (original_price - final_price) / original_price * 100 = 41.86 := by
  sorry

end cumulative_percentage_decrease_l81_81252


namespace fraction_exponentiation_l81_81548

theorem fraction_exponentiation : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end fraction_exponentiation_l81_81548


namespace zoe_earnings_from_zachary_l81_81365

noncomputable def babysitting_earnings 
  (total_earnings : ℕ) (pool_cleaning_earnings : ℕ) (earnings_julie_ratio : ℕ) 
  (earnings_chloe_ratio : ℕ) 
  (earnings_zachary : ℕ) : Prop := 
total_earnings = 8000 ∧ 
pool_cleaning_earnings = 2600 ∧ 
earnings_julie_ratio = 3 ∧ 
earnings_chloe_ratio = 5 ∧ 
9 * earnings_zachary = 5400

theorem zoe_earnings_from_zachary : babysitting_earnings 8000 2600 3 5 600 :=
by 
  unfold babysitting_earnings
  sorry

end zoe_earnings_from_zachary_l81_81365


namespace trapezoid_median_properties_l81_81119

-- Define the variables
variables (a b x : ℝ)

-- State the conditions and the theorem
theorem trapezoid_median_properties (h1 : x = (2 * a) / 3) (h2 : x = b + 3) (h3 : x = (a + b) / 2) : x = 6 :=
by
  sorry

end trapezoid_median_properties_l81_81119


namespace complex_sum_is_2_l81_81947

theorem complex_sum_is_2 
  (a b c d e f : ℂ) 
  (hb : b = 4) 
  (he : e = 2 * (-a - c)) 
  (hr : a + c + e = 0) 
  (hi : b + d + f = 6) 
  : d + f = 2 := 
  by
  sorry

end complex_sum_is_2_l81_81947


namespace largest_x_undefined_largest_solution_l81_81736

theorem largest_x_undefined (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0) → x = 10 ∨ x = 1 / 10 :=
by
  sorry

theorem largest_solution (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end largest_x_undefined_largest_solution_l81_81736


namespace inequality_system_solution_l81_81644

theorem inequality_system_solution {x : ℝ} (h1 : 2 * x - 1 < x + 5) (h2 : (x + 1)/3 < x - 1) : 2 < x ∧ x < 6 :=
by
  sorry

end inequality_system_solution_l81_81644


namespace line_equation_l81_81393

theorem line_equation (x y : ℝ) : 
  (3 * x + y = 0) ∧ (x + y - 2 = 0) ∧ 
  ∃ m : ℝ, -2 = -(1 / m) ∧ 
  (∃ b : ℝ, (y = m * x + b) ∧ (3 = m * (-1) + b)) ∧ 
  x - 2 * y + 7 = 0 :=
sorry

end line_equation_l81_81393


namespace calculate_expression_l81_81093

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l81_81093


namespace boy_overall_average_speed_l81_81526

noncomputable def total_distance : ℝ := 100
noncomputable def distance1 : ℝ := 15
noncomputable def speed1 : ℝ := 12

noncomputable def distance2 : ℝ := 20
noncomputable def speed2 : ℝ := 8

noncomputable def distance3 : ℝ := 10
noncomputable def speed3 : ℝ := 25

noncomputable def distance4 : ℝ := 15
noncomputable def speed4 : ℝ := 18

noncomputable def distance5 : ℝ := 20
noncomputable def speed5 : ℝ := 10

noncomputable def distance6 : ℝ := 20
noncomputable def speed6 : ℝ := 22

noncomputable def time1 : ℝ := distance1 / speed1
noncomputable def time2 : ℝ := distance2 / speed2
noncomputable def time3 : ℝ := distance3 / speed3
noncomputable def time4 : ℝ := distance4 / speed4
noncomputable def time5 : ℝ := distance5 / speed5
noncomputable def time6 : ℝ := distance6 / speed6

noncomputable def total_time : ℝ := time1 + time2 + time3 + time4 + time5 + time6

noncomputable def overall_average_speed : ℝ := total_distance / total_time

theorem boy_overall_average_speed : overall_average_speed = 100 / (15 / 12 + 20 / 8 + 10 / 25 + 15 / 18 + 20 / 10 + 20 / 22) :=
by
  sorry

end boy_overall_average_speed_l81_81526


namespace area_of_shaded_region_l81_81540

theorem area_of_shaded_region : 
  let side_length := 4
  let radius := side_length / 2 
  let area_of_square := side_length * side_length 
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle 
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles 
  area_of_shaded_region = 16 - 4 * pi :=
by
  let side_length := 4
  let radius := side_length / 2
  let area_of_square := side_length * side_length
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles
  sorry

end area_of_shaded_region_l81_81540


namespace thirty_percent_more_than_80_is_one_fourth_less_l81_81948

-- Translating the mathematical equivalency conditions into Lean definitions and theorems

def thirty_percent_more (n : ℕ) : ℕ :=
  n + (n * 30 / 100)

def one_fourth_less (x : ℕ) : ℕ :=
  x - (x / 4)

theorem thirty_percent_more_than_80_is_one_fourth_less (x : ℕ) :
  thirty_percent_more 80 = one_fourth_less x → x = 139 :=
by
  sorry

end thirty_percent_more_than_80_is_one_fourth_less_l81_81948


namespace parabola_equation_l81_81074

theorem parabola_equation (h1: ∃ k, ∀ x y : ℝ, (x, y) = (4, -2) → y^2 = k * x) 
                          (h2: ∃ m, ∀ x y : ℝ, (x, y) = (4, -2) → x^2 = -2 * m * y) :
                          (y : ℝ)^2 = x ∨ (x : ℝ)^2 = -8 * y :=
by 
  sorry

end parabola_equation_l81_81074


namespace calculate_expression_l81_81849

theorem calculate_expression : 
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := 
by
  sorry

end calculate_expression_l81_81849


namespace original_number_is_16_l81_81071

theorem original_number_is_16 (x : ℤ) (h1 : 3 * (2 * x + 5) = 111) : x = 16 :=
by
  sorry

end original_number_is_16_l81_81071


namespace trick_deck_cost_l81_81725

theorem trick_deck_cost :
  (∃ x : ℝ, 4 * x + 4 * x = 72) → ∃ x : ℝ, x = 9 := sorry

end trick_deck_cost_l81_81725


namespace meet_at_starting_point_second_time_in_minutes_l81_81806

theorem meet_at_starting_point_second_time_in_minutes :
  let racing_magic_time := 60 -- in seconds
  let charging_bull_time := 3600 / 40 -- in seconds
  let lcm_time := Nat.lcm racing_magic_time charging_bull_time -- LCM of the round times in seconds
  let answer := lcm_time / 60 -- convert seconds to minutes
  answer = 3 :=
by
  sorry

end meet_at_starting_point_second_time_in_minutes_l81_81806


namespace expression_evaluation_l81_81103

theorem expression_evaluation :
  (π - 1)^0 + 4 * real.sin (real.pi / 4) - real.sqrt 8 + abs (-3) = 4 := 
sorry

end expression_evaluation_l81_81103


namespace abs_diff_two_numbers_l81_81655

theorem abs_diff_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 := by
  sorry

end abs_diff_two_numbers_l81_81655


namespace octadecagon_identity_l81_81784

theorem octadecagon_identity (a r : ℝ) (h : a = 2 * r * Real.sin (π / 18)) :
  a^3 + r^3 = 3 * r^2 * a :=
sorry

end octadecagon_identity_l81_81784


namespace minimum_value_f_l81_81803

noncomputable def f (x : ℝ) : ℝ := (x^2 / 8) + x * (Real.cos x) + (Real.cos (2 * x))

theorem minimum_value_f : ∃ x : ℝ, f x = -1 :=
by {
  sorry
}

end minimum_value_f_l81_81803


namespace problem_l81_81152

noncomputable def vector_a (ω φ x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x + φ), 1)
noncomputable def vector_b (ω φ x : ℝ) : ℝ × ℝ := (1, Real.cos (ω / 2 * x + φ))
noncomputable def f (ω φ x : ℝ) : ℝ := 
  let a := vector_a ω φ x
  let b := vector_b ω φ x
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)

theorem problem 
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 4)
  (h_period : Function.Periodic (f ω φ) 4)
  (h_point1 : f ω φ 1 = 1 / 2) : 
  ω = π / 2 ∧ ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f (π / 2) (π / 12) x ∧ f (π / 2) (π / 12) x ≤ 1 / 2 := 
by
  sorry

end problem_l81_81152


namespace find_x_l81_81690

theorem find_x (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 := 
by 
  sorry

end find_x_l81_81690


namespace haley_initial_trees_l81_81086

theorem haley_initial_trees (dead_trees trees_left initial_trees : ℕ) 
    (h_dead: dead_trees = 2)
    (h_left: trees_left = 10)
    (h_initial: initial_trees = trees_left + dead_trees) : 
    initial_trees = 12 := 
by sorry

end haley_initial_trees_l81_81086


namespace nonnegative_integers_existence_l81_81307

open Classical

theorem nonnegative_integers_existence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d) ↔ (5 * x ≥ 7 * y) :=
by
  sorry

end nonnegative_integers_existence_l81_81307


namespace video_time_per_week_l81_81909

-- Define the basic conditions
def short_video_length : ℕ := 2
def multiplier : ℕ := 6
def long_video_length : ℕ := multiplier * short_video_length
def short_videos_per_day : ℕ := 2
def long_videos_per_day : ℕ := 1
def days_in_week : ℕ := 7

-- Calculate daily and weekly video release time
def daily_video_time : ℕ := (short_videos_per_day * short_video_length) + (long_videos_per_day * long_video_length)
def weekly_video_time : ℕ := daily_video_time * days_in_week

-- Main theorem to prove
theorem video_time_per_week : weekly_video_time = 112 := by
    sorry

end video_time_per_week_l81_81909


namespace selection_methods_count_l81_81460

-- Define a function to compute combinations (n choose r)
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement
theorem selection_methods_count :
  combination 5 2 * combination 3 1 * combination 2 1 = 60 :=
by
  sorry

end selection_methods_count_l81_81460


namespace distance_between_hyperbola_vertices_l81_81735

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l81_81735


namespace teacher_student_arrangements_boy_girl_selection_program_arrangements_l81_81838

-- Question 1
theorem teacher_student_arrangements : 
  let positions := 5
  let student_arrangements := 720
  positions * student_arrangements = 3600 :=
by
  sorry

-- Question 2
theorem boy_girl_selection :
  let total_selections := 330
  let opposite_selections := 20
  total_selections - opposite_selections = 310 :=
by
  sorry

-- Question 3
theorem program_arrangements :
  let total_permutations := 120
  let relative_order_permutations := 6
  total_permutations / relative_order_permutations = 20 :=
by
  sorry

end teacher_student_arrangements_boy_girl_selection_program_arrangements_l81_81838


namespace Liza_initial_balance_l81_81454

theorem Liza_initial_balance
  (W: Nat)   -- Liza's initial balance on Tuesday
  (rent: Nat := 450)
  (deposit: Nat := 1500)
  (electricity: Nat := 117)
  (internet: Nat := 100)
  (phone: Nat := 70)
  (final_balance: Nat := 1563) 
  (balance_eq: W - rent + deposit - electricity - internet - phone = final_balance) 
  : W = 800 :=
sorry

end Liza_initial_balance_l81_81454


namespace trig_expression_value_l81_81287

variable (θ : ℝ)

theorem trig_expression_value (h : Real.cos (π + θ) = 1 / 3) : 
  (Real.cos (2 * π - θ)) / 
  (Real.sin (π / 2 + θ) * Real.cos (π - θ) + Real.cos (-θ)) = 3 / 4 :=
by
  sorry

end trig_expression_value_l81_81287


namespace range_of_H_l81_81858

def H (x : ℝ) : ℝ := 2 * |2 * x + 2| - 3 * |2 * x - 2|

theorem range_of_H : Set.range H = Set.Ici 8 := 
by 
  sorry

end range_of_H_l81_81858


namespace mod_remainder_w_l81_81868

theorem mod_remainder_w (w : ℕ) (h : w = 3^39) : w % 13 = 1 :=
by
  sorry

end mod_remainder_w_l81_81868


namespace rectangular_prism_width_l81_81463

variables (w : ℝ)

theorem rectangular_prism_width (h : ℝ) (l : ℝ) (d : ℝ) (hyp_l : l = 5) (hyp_h : h = 7) (hyp_d : d = 15) :
  w = Real.sqrt 151 :=
by 
  -- Proof goes here
  sorry

end rectangular_prism_width_l81_81463


namespace melted_ice_cream_depth_l81_81975

noncomputable def ice_cream_depth : ℝ :=
  let r1 := 3 -- radius of the sphere
  let r2 := 10 -- radius of the cylinder
  let V_sphere := (4/3) * Real.pi * r1^3 -- volume of the sphere
  let V_cylinder h := Real.pi * r2^2 * h -- volume of the cylinder
  V_sphere / (Real.pi * r2^2)

theorem melted_ice_cream_depth :
  ice_cream_depth = 9 / 25 :=
by
  sorry

end melted_ice_cream_depth_l81_81975


namespace max_consecutive_sum_less_500_l81_81488

theorem max_consecutive_sum_less_500 : 
  (argmax (λ n : ℕ, n * (n + 1) / 2 < 500) = 31) :=
by sorry

end max_consecutive_sum_less_500_l81_81488


namespace rajan_income_l81_81063

theorem rajan_income : 
  ∀ (x y : ℕ), 
  7 * x - 6 * y = 1000 → 
  6 * x - 5 * y = 1000 → 
  7 * x = 7000 := 
by 
  intros x y h1 h2
  sorry

end rajan_income_l81_81063


namespace quadratic_vertex_coordinates_l81_81350

theorem quadratic_vertex_coordinates (x y : ℝ) (h : y = 2 * x^2 - 4 * x + 5) : (x, y) = (1, 3) :=
sorry

end quadratic_vertex_coordinates_l81_81350


namespace average_weight_20_boys_l81_81479

theorem average_weight_20_boys 
  (A : Real)
  (numBoys₁ numBoys₂ : ℕ)
  (weight₂ : Real)
  (avg_weight_class : Real)
  (h_numBoys₁ : numBoys₁ = 20)
  (h_numBoys₂ : numBoys₂ = 8)
  (h_weight₂ : weight₂ = 45.15)
  (h_avg_weight_class : avg_weight_class = 48.792857142857144)
  (h_total_boys : numBoys₁ + numBoys₂ = 28)
  (h_eq_weight : numBoys₁ * A + numBoys₂ * weight₂ = 28 * avg_weight_class) :
  A = 50.25 :=
  sorry

end average_weight_20_boys_l81_81479


namespace simplified_expression_at_minus_one_is_negative_two_l81_81639

-- Define the problem: simplifying the given expression
def simplify_expression (x : ℝ) : ℝ := (2 / (x^2 - 4)) * ((x^2 - 2 * x) / 1)

-- Prove that when x = -1, the simplified expression equals -2
theorem simplified_expression_at_minus_one_is_negative_two : simplify_expression (-1) = -2 := 
by 
  sorry

end simplified_expression_at_minus_one_is_negative_two_l81_81639


namespace rug_length_l81_81450

theorem rug_length (d : ℕ) (x y : ℕ) (h1 : x * x + y * y = d * d) (h2 : y / x = 2) (h3 : (x = 25 ∧ y = 50)) : 
  x = 25 := 
sorry

end rug_length_l81_81450


namespace class_size_l81_81511

theorem class_size :
  ∃ (N : ℕ), (20 ≤ N) ∧ (N ≤ 30) ∧ (∃ (x : ℕ), N = 3 * x + 1) ∧ (∃ (y : ℕ), N = 4 * y + 1) ∧ (N = 25) :=
by { sorry }

end class_size_l81_81511


namespace number_of_customers_l81_81081

theorem number_of_customers 
    (boxes_opened : ℕ) 
    (samples_per_box : ℕ) 
    (samples_left_over : ℕ) 
    (samples_limit_per_person : ℕ)
    (h1 : boxes_opened = 12)
    (h2 : samples_per_box = 20)
    (h3 : samples_left_over = 5)
    (h4 : samples_limit_per_person = 1) : 
    ∃ customers : ℕ, customers = (boxes_opened * samples_per_box) - samples_left_over ∧ customers = 235 :=
by {
  sorry
}

end number_of_customers_l81_81081


namespace exam_questions_count_l81_81346

theorem exam_questions_count (Q S : ℕ) 
    (hS : S = (4 * Q) / 5)
    (sergio_correct : Q - 4 = S + 6) : 
    Q = 50 :=
by 
  sorry

end exam_questions_count_l81_81346


namespace certain_percentage_l81_81368

theorem certain_percentage (P : ℝ) : 
  0.15 * P * 0.50 * 4000 = 90 → P = 0.3 :=
by
  sorry

end certain_percentage_l81_81368


namespace perpendicular_lines_condition_l81_81022

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → m * x + y + 1 = 0 → False) ↔ m = 1 / 2 :=
by sorry

end perpendicular_lines_condition_l81_81022


namespace least_number_divisible_by_five_smallest_primes_l81_81054

theorem least_number_divisible_by_five_smallest_primes : 
  ∃ n ∈ ℕ+, n = 2 * 3 * 5 * 7 * 11 ∧ n = 2310 :=
by
  sorry

end least_number_divisible_by_five_smallest_primes_l81_81054


namespace cannot_be_computed_using_square_diff_l81_81684

-- Define the conditions
def A := (x + 1) * (x - 1)
def B := (-x + 1) * (-x - 1)
def C := (x + 1) * (-x + 1)
def D := (x + 1) * (1 + x)

-- Proposition: Prove that D cannot be computed using the square difference formula
theorem cannot_be_computed_using_square_diff (x : ℝ) : 
  ¬ (D = x^2 - y^2) := 
sorry

end cannot_be_computed_using_square_diff_l81_81684


namespace math_expression_equivalent_l81_81101

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l81_81101


namespace calculate_expression_l81_81094

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l81_81094


namespace inequality_proof_l81_81743

variable (a b c d e p q : ℝ)

theorem inequality_proof
  (h₀ : 0 < p)
  (h₁ : p ≤ a) (h₂ : a ≤ q)
  (h₃ : p ≤ b) (h₄ : b ≤ q)
  (h₅ : p ≤ c) (h₆ : c ≤ q)
  (h₇ : p ≤ d) (h₈ : d ≤ q)
  (h₉ : p ≤ e) (h₁₀ : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 := 
by
  sorry -- The actual proof will be filled here

end inequality_proof_l81_81743


namespace problem_statement_l81_81098

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l81_81098


namespace max_value_of_k_l81_81592

noncomputable def max_possible_k (x y : ℝ) (k : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ 0 < k ∧
  (3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x))

theorem max_value_of_k (x y : ℝ) (k : ℝ) :
  max_possible_k x y k → k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end max_value_of_k_l81_81592


namespace east_bound_cyclist_speed_l81_81667

-- Define the speeds of the cyclists and the relationship between them
def east_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * x
def west_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * (x + 4)

-- Condition: After 5 hours, they are 200 miles apart
def total_distance (t : ℕ) (x : ℕ) : ℕ := east_bound_speed t x + west_bound_speed t x

theorem east_bound_cyclist_speed :
  ∃ x : ℕ, total_distance 5 x = 200 ∧ x = 18 :=
by
  sorry

end east_bound_cyclist_speed_l81_81667


namespace no_solution_inequality_l81_81427

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, |x - 5| + |x + 3| < a) ↔ a ≤ 8 := 
sorry

end no_solution_inequality_l81_81427


namespace find_x_values_l81_81116

theorem find_x_values (x : ℝ) (h : x + 60 / (x - 3) = -12) : x = -3 ∨ x = -6 :=
sorry

end find_x_values_l81_81116


namespace tom_total_amount_after_saving_l81_81817

theorem tom_total_amount_after_saving :
  let hourly_rate := 6.50
  let work_hours := 31
  let saving_rate := 0.10
  let total_earnings := hourly_rate * work_hours
  let amount_set_aside := total_earnings * saving_rate
  let amount_for_purchases := total_earnings - amount_set_aside
  amount_for_purchases = 181.35 :=
by
  sorry

end tom_total_amount_after_saving_l81_81817


namespace Morley_l81_81707

-- define the vertices and trisectors
variable (A B C : Point)
variable (α1 α1' α2 α2' α3 α3' : Ray)
variable (β1 β1' β2 β2' β3 β3' : Ray)
variable (γ1 γ1' γ2 γ2' γ3 γ3' : Ray)

-- define the angles between the trisectors
axiom trisector_condition1 :
  (angle AB α1 = π / 3) ∧ (angle α1 α1' = π / 3) ∧ (angle α1' AC = π / 3)
axiom trisector_condition2 :
  (angle AB α2 = π / 3) ∧ (angle α2 α2' = π / 3) ∧ (angle α2' AC = π / 3)
axiom trisector_condition3 :
  (angle AB α3 = π / 3) ∧ (angle α3 α3' = π / 3) ∧ (angle α3' AC = π / 3)
-- similarly for β and γ conditions

-- define the triangles
noncomputable def triangle_formed (i j k : ℕ) : Triangle := {
  p1 := intersection_line (α_i i) (β_j' j)
  p2 := intersection_line (β_j j) (γ_k' k)
  p3 := intersection_line (γ_k k) (α_i' i)
}

-- main theorem
theorem Morley's_theorem_equilateral (i j k : ℕ) (h : (i + j + k - 1) % 3 ≠ 0) :
  is_equilateral (triangle_formed i j k) ∧
  sides_parallel (triangle_formed i j k) ∧
  vertices_on_lines (triangle_formed i j k) :=
sorry

end Morley_l81_81707


namespace SusanBooks_l81_81449

-- Definitions based on the conditions of the problem
def Lidia (S : ℕ) : ℕ := 4 * S
def TotalBooks (S : ℕ) : ℕ := S + Lidia S

-- The proof statement
theorem SusanBooks (S : ℕ) (h : TotalBooks S = 3000) : S = 600 :=
by
  sorry

end SusanBooks_l81_81449


namespace find_sticker_price_l81_81153

variable (x : ℝ)

def price_at_store_A (x : ℝ) : ℝ := 0.80 * x - 120
def price_at_store_B (x : ℝ) : ℝ := 0.70 * x
def savings (x : ℝ) : ℝ := price_at_store_B x - price_at_store_A x

theorem find_sticker_price (h : savings x = 30) : x = 900 :=
by
  -- proof can be filled in here
  sorry

end find_sticker_price_l81_81153


namespace original_ratio_l81_81818

theorem original_ratio (x y : ℤ)
  (h1 : y = 48)
  (h2 : (x + 12) * 2 = y) :
  x * 4 = y := sorry

end original_ratio_l81_81818


namespace namjoonKoreanScore_l81_81336

variables (mathScore englishScore : ℝ) (averageScore : ℝ := 95) (koreanScore : ℝ)

def namjoonMathScore : Prop := mathScore = 100
def namjoonEnglishScore : Prop := englishScore = 95
def namjoonAverage : Prop := (koreanScore + mathScore + englishScore) / 3 = averageScore

theorem namjoonKoreanScore
  (H1 : namjoonMathScore 100)
  (H2 : namjoonEnglishScore 95)
  (H3 : namjoonAverage koreanScore 100 95 95) :
  koreanScore = 90 :=
by
  sorry

end namjoonKoreanScore_l81_81336


namespace solve_for_y_l81_81791

theorem solve_for_y (y : ℚ) : 
  3 + 1 / (1 + 1 / (3 + 3 / (4 + y))) = 169 / 53 → y = -605 / 119 :=
by
  intro h
  sorry

end solve_for_y_l81_81791


namespace largest_possible_perimeter_l81_81360

theorem largest_possible_perimeter
  (a b c : ℕ)
  (h1 : a > 2 ∧ b > 2 ∧ c > 2)  -- sides are greater than 2
  (h2 : a = c ∨ b = c ∨ a = b)  -- at least two polygons are congruent
  (h3 : (a - 2) * (b - 2) = 8 ∨ (a - 2) * (c - 2) = 8 ∨ (b - 2) * (c - 2) = 8)  -- possible factorizations
  (h4 : (a - 2) + (b - 2) + (c - 2) = 12)  -- sum of interior angles at A is 360 degrees
  : 2 * a + 2 * b + 2 * c - 6 ≤ 21 :=
sorry

end largest_possible_perimeter_l81_81360


namespace find_interval_n_l81_81121

theorem find_interval_n 
  (n : ℕ) 
  (h1 : n < 500)
  (h2 : (∃ abcde : ℕ, 0 < abcde ∧ abcde < 99999 ∧ n * abcde = 99999))
  (h3 : (∃ uvw : ℕ, 0 < uvw ∧ uvw < 999 ∧ (n + 3) * uvw = 999)) 
  : 201 ≤ n ∧ n ≤ 300 := 
sorry

end find_interval_n_l81_81121


namespace distance_to_other_focus_of_ellipse_l81_81423

noncomputable def ellipse_param (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def is_focus_distance (a distF1 distF2 : ℝ) : Prop :=
  ∀ P₁ P₂ : ℝ, distF1 + distF2 = 2 * a

theorem distance_to_other_focus_of_ellipse (x y : ℝ) (distF1 : ℝ) :
  ellipse_param 4 5 x y ∧ distF1 = 6 → is_focus_distance 5 distF1 4 :=
by
  simp [ellipse_param, is_focus_distance]
  sorry

end distance_to_other_focus_of_ellipse_l81_81423


namespace product_of_two_numbers_l81_81347

theorem product_of_two_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 560) (h_hcf : Nat.gcd a b = 75) :
  a * b = 42000 :=
by
  sorry

end product_of_two_numbers_l81_81347


namespace katelyn_sandwiches_difference_l81_81546

theorem katelyn_sandwiches_difference :
  ∃ (K : ℕ), K - 49 = 47 ∧ (49 + K + K / 4 = 169) := 
sorry

end katelyn_sandwiches_difference_l81_81546


namespace tree_planting_growth_rate_l81_81024

theorem tree_planting_growth_rate {x : ℝ} :
  400 * (1 + x) ^ 2 = 625 :=
sorry

end tree_planting_growth_rate_l81_81024


namespace distance_between_foci_of_ellipse_l81_81480

-- Define the three given points
structure Point where
  x : ℝ
  y : ℝ

def p1 : Point := ⟨1, 3⟩
def p2 : Point := ⟨5, -1⟩
def p3 : Point := ⟨10, 3⟩

-- Define the statement that the distance between the foci of the ellipse they define is 2 * sqrt(4.25)
theorem distance_between_foci_of_ellipse : 
  ∃ (c : ℝ) (f : ℝ), f = 2 * Real.sqrt 4.25 ∧ 
  (∃ (ellipse : Point → Prop), ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
sorry

end distance_between_foci_of_ellipse_l81_81480


namespace total_students_in_school_l81_81432

theorem total_students_in_school : 
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  C1 + C2 + C3 + C4 + C5 = 140 :=
by
  let C1 := 32
  let C2 := C1 - 2
  let C3 := C2 - 2
  let C4 := C3 - 2
  let C5 := C4 - 2
  sorry

end total_students_in_school_l81_81432


namespace num_customers_who_tried_sample_l81_81078

theorem num_customers_who_tried_sample :
  ∀ (samples_per_box boxes_opened samples_left : ℕ), 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  let total_samples := samples_per_box * boxes_opened in
  let samples_used := total_samples - samples_left in
  samples_used = 235 :=
by 
  intros samples_per_box boxes_opened samples_left h_samples_per_box h_boxes_opened h_samples_left total_samples samples_used
  simp [h_samples_per_box, h_boxes_opened, h_samples_left]
  sorry

end num_customers_who_tried_sample_l81_81078


namespace scientific_notation_of_819000_l81_81904

theorem scientific_notation_of_819000 : (819000 : ℝ) = 8.19 * 10^5 :=
by
  sorry

end scientific_notation_of_819000_l81_81904


namespace tan_75_eq_2_plus_sqrt_3_l81_81990

theorem tan_75_eq_2_plus_sqrt_3 :
  Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_eq_2_plus_sqrt_3_l81_81990


namespace probability_N_taller_than_L_l81_81133

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l81_81133


namespace log_base_27_of_3_l81_81559

theorem log_base_27_of_3 : log 27 3 = 1 / 3 :=
by
  -- Define the conditions
  have h1 : 27 = 3 ^ 3 := by norm_num
  have h2 : ∀ (a b n : ℝ), a ^ n = b → log b a = 1 / n,
    from λ a b n h, by rw [←h, log_pow]; norm_num,
  -- Use the conditions to prove the theorem
  exact h2 3 27 3 h1

end log_base_27_of_3_l81_81559


namespace moli_bought_7_clips_l81_81335

theorem moli_bought_7_clips (R C S x : ℝ) 
  (h1 : 3*R + x*C + S = 120) 
  (h2 : 4*R + 10*C + S = 164) 
  (h3 : R + C + S = 32) : 
  x = 7 := 
by
  sorry

end moli_bought_7_clips_l81_81335


namespace abs_diff_eq_10_l81_81657

variable {x y : ℝ}

-- Given conditions as definitions.
def condition1 : Prop := x + y = 30
def condition2 : Prop := x * y = 200

-- The theorem statement to prove the given question equals the correct answer.
theorem abs_diff_eq_10 (h1 : condition1) (h2 : condition2) : |x - y| = 10 :=
by
  sorry

end abs_diff_eq_10_l81_81657


namespace distance_between_A_and_B_l81_81075

theorem distance_between_A_and_B 
  (v1 v2: ℝ) (s: ℝ)
  (h1 : (s - 8) / v1 = s / v2)
  (h2 : s / (2 * v1) = (s - 15) / v2)
  (h3: s = 40) : 
  s = 40 := 
sorry

end distance_between_A_and_B_l81_81075


namespace distance_between_circle_center_and_point_l81_81363

theorem distance_between_circle_center_and_point (x y : ℝ) (h : x^2 + y^2 = 8*x - 12*y + 40) : 
  dist (4, -6) (4, -2) = 4 := 
by
  sorry

end distance_between_circle_center_and_point_l81_81363


namespace four_diff_digits_per_day_l81_81197

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l81_81197


namespace relay_race_team_members_l81_81238

theorem relay_race_team_members (n : ℕ) (d : ℕ) (h1 : n = 5) (h2 : d = 150) : d / n = 30 := 
by {
  -- Place the conditions here as hypotheses
  sorry
}

end relay_race_team_members_l81_81238


namespace nina_not_taller_than_lena_l81_81131

noncomputable def friends_heights := ℝ 
variables (M N L O : friends_heights)

def nina_shorter_than_masha (N M : friends_heights) : Prop := N < M
def lena_taller_than_olya (L O : friends_heights) : Prop := L > O
def nina_taller_than_lena (N L : friends_heights) : Prop := N > L

theorem nina_not_taller_than_lena (N M L O : friends_heights) 
  (h₁ : nina_shorter_than_masha N M) 
  (h₂ : lena_taller_than_olya L O) : 
  (0 : ℝ) = 0 :=
sorry

end nina_not_taller_than_lena_l81_81131


namespace debby_vacation_pictures_l81_81860

theorem debby_vacation_pictures :
  let zoo_initial := 150
  let aquarium_initial := 210
  let museum_initial := 90
  let amusement_park_initial := 120
  let zoo_deleted := (25 * zoo_initial) / 100  -- 25% of zoo pictures deleted
  let aquarium_deleted := (15 * aquarium_initial) / 100  -- 15% of aquarium pictures deleted
  let museum_added := 30  -- 30 additional pictures at the museum
  let amusement_park_deleted := 20  -- 20 pictures deleted at the amusement park
  let zoo_kept := zoo_initial - zoo_deleted
  let aquarium_kept := aquarium_initial - aquarium_deleted
  let museum_kept := museum_initial + museum_added
  let amusement_park_kept := amusement_park_initial - amusement_park_deleted
  let total_pictures := zoo_kept + aquarium_kept + museum_kept + amusement_park_kept
  total_pictures = 512 :=
by
  sorry

end debby_vacation_pictures_l81_81860


namespace middle_number_is_correct_l81_81348

theorem middle_number_is_correct (numbers : List ℝ) (h_length : numbers.length = 11)
  (h_avg11 : numbers.sum / 11 = 9.9)
  (first_6 : List ℝ) (h_first6_length : first_6.length = 6)
  (h_avg6_1 : first_6.sum / 6 = 10.5)
  (last_6 : List ℝ) (h_last6_length : last_6.length = 6)
  (h_avg6_2 : last_6.sum / 6 = 11.4) :
  (∃ m : ℝ, m ∈ first_6 ∧ m ∈ last_6 ∧ m = 22.5) :=
by
  sorry

end middle_number_is_correct_l81_81348


namespace area_of_union_of_triangle_and_reflection_l81_81979

-- Define points in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the original triangle
def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -1⟩
def C : Point := ⟨7, 0⟩

-- Define the vertices of the reflected triangle
def A' : Point := ⟨-2, 3⟩
def B' : Point := ⟨-4, -1⟩
def C' : Point := ⟨-7, 0⟩

-- Calculate the area of a triangle given three points
def triangleArea (P Q R : Point) : ℝ :=
  0.5 * |P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)|

-- Statement to prove: the area of the union of the original and reflected triangles
theorem area_of_union_of_triangle_and_reflection :
  triangleArea A B C + triangleArea A' B' C' = 14 := 
sorry

end area_of_union_of_triangle_and_reflection_l81_81979


namespace determine_functions_l81_81718

noncomputable def functional_eq_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x) ^ 2

theorem determine_functions (f : ℝ → ℝ) (h : functional_eq_condition f) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
sorry

end determine_functions_l81_81718


namespace initial_average_is_100_l81_81466

-- Definitions based on the conditions from step a)
def students : ℕ := 10
def wrong_mark : ℕ := 90
def correct_mark : ℕ := 10
def correct_average : ℝ := 92

-- Initial average marks before correcting the error
def initial_average_marks (A : ℝ) : Prop :=
  10 * A = (students * correct_average) + (wrong_mark - correct_mark)

theorem initial_average_is_100 :
  ∃ A : ℝ, initial_average_marks A ∧ A = 100 :=
by {
  -- We are defining the placeholder for the actual proof.
  sorry
}

end initial_average_is_100_l81_81466


namespace numberOfBooks_correct_l81_81694

variable (totalWeight : ℕ) (weightPerBook : ℕ)

def numberOfBooks (totalWeight weightPerBook : ℕ) : ℕ :=
  totalWeight / weightPerBook

theorem numberOfBooks_correct (h1 : totalWeight = 42) (h2 : weightPerBook = 3) :
  numberOfBooks totalWeight weightPerBook = 14 := by
  sorry

end numberOfBooks_correct_l81_81694


namespace range_of_a_l81_81147

noncomputable def f (x : ℝ) : ℝ := 4 * x + 3 * Real.sin x

theorem range_of_a (a : ℝ) (h : f (1 - a) + f (1 - a^2) < 0) : 1 < a ∧ a < Real.sqrt 2 := sorry

end range_of_a_l81_81147


namespace clock_four_different_digits_l81_81204

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l81_81204


namespace sum_lent_is_3000_l81_81969

noncomputable def principal_sum (P : ℕ) : Prop :=
  let R := 5
  let T := 5
  let SI := (P * R * T) / 100
  SI = P - 2250

theorem sum_lent_is_3000 : ∃ (P : ℕ), principal_sum P ∧ P = 3000 :=
by
  use 3000
  unfold principal_sum
  -- The following are the essential parts
  sorry

end sum_lent_is_3000_l81_81969


namespace lenny_has_39_left_l81_81443

/-- Define the initial amount Lenny has -/
def initial_amount : ℕ := 84

/-- Define the amount Lenny spent on video games -/
def spent_on_video_games : ℕ := 24

/-- Define the amount Lenny spent at the grocery store -/
def spent_on_groceries : ℕ := 21

/-- Define the total amount Lenny spent -/
def total_spent : ℕ := spent_on_video_games + spent_on_groceries

/-- Calculate the amount Lenny has left -/
def amount_left (initial amount_spent : ℕ) : ℕ :=
  initial - amount_spent

/-- The statement of our mathematical equivalent proof problem
  Prove that Lenny has $39 left given the initial amount,
  and the amounts spent on video games and groceries.
-/
theorem lenny_has_39_left :
  amount_left initial_amount total_spent = 39 :=
by
  -- Leave the proof as 'sorry' for now
  sorry

end lenny_has_39_left_l81_81443


namespace least_positive_number_divisible_by_primes_l81_81032

theorem least_positive_number_divisible_by_primes :
  ∃ n : ℕ, n > 0 ∧
    (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ n) ∧
    (∀ m : ℕ, (m > 0 ∧ (∀ p ∈ {2, 3, 5, 7, 11}, p ∣ m)) → n ≤ m) ∧
    n = 2310 :=
by
  sorry

end least_positive_number_divisible_by_primes_l81_81032


namespace new_roots_quadratic_l81_81462

variable {p q : ℝ}

theorem new_roots_quadratic :
  (∀ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = q → 
  (x : ℝ) → x^2 + ((p^2 - 2 * q)^2 - 2 * q^2) * x + q^4 = 0) :=
by 
  intros r₁ r₂ h x
  have : r₁ + r₂ = -p := h.1
  have : r₁ * r₂ = q := h.2
  sorry

end new_roots_quadratic_l81_81462


namespace cannot_finish_third_l81_81788

-- Define the racers
inductive Racer
| P | Q | R | S | T | U
open Racer

-- Define the conditions
def beats (a b : Racer) : Prop := sorry  -- placeholder for strict order
def ties (a b : Racer) : Prop := sorry   -- placeholder for tie condition
def position (r : Racer) (p : Fin (6)) : Prop := sorry  -- placeholder for position in the race

theorem cannot_finish_third :
  (beats P Q) ∧
  (ties P R) ∧
  (beats Q S) ∧
  ∃ p₁ p₂ p₃, position P p₁ ∧ position T p₂ ∧ position Q p₃ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧
  ∃ p₄ p₅, position U p₄ ∧ position S p₅ ∧ p₄ < p₅ →
  ¬ position P (3 : Fin (6)) ∧ ¬ position U (3 : Fin (6)) ∧ ¬ position S (3 : Fin (6)) :=
by sorry   -- Proof is omitted

end cannot_finish_third_l81_81788


namespace range_of_a_l81_81893

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * x + a - 3 > 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l81_81893


namespace least_multiple_of_five_primes_l81_81046

noncomputable def smallest_multiple_of_five_primes : ℕ :=
  let primes := [2, 3, 5, 7, 11] in
  primes.foldl (· * ·) 1

theorem least_multiple_of_five_primes : smallest_multiple_of_five_primes = 2310 := by
  sorry

end least_multiple_of_five_primes_l81_81046


namespace percent_round_trip_tickets_l81_81337

-- Define the main variables
variables (P R : ℝ)

-- Define the conditions based on the problem statement
def condition1 : Prop := 0.3 * P = 0.3 * R
 
-- State the theorem to prove
theorem percent_round_trip_tickets (h1 : condition1 P R) : R / P * 100 = 30 := by sorry

end percent_round_trip_tickets_l81_81337


namespace inequality_proof_l81_81290

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) :
  (x + y + z) / 3 ≥ (2 * x * y * z)^(1/3 : ℝ) :=
by
  sorry

end inequality_proof_l81_81290


namespace set_inter_complement_eq_l81_81173

-- Given conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 < 1}
def B : Set ℝ := {x | x^2 - 2 * x > 0}

-- Question translated to proof problem statement
theorem set_inter_complement_eq :
  A ∩ (U \ B) = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_inter_complement_eq_l81_81173


namespace carB_speed_l81_81799

variable (distance : ℝ) (time : ℝ) (ratio : ℝ) (speedB : ℝ)

theorem carB_speed (h1 : distance = 240) (h2 : time = 1.5) (h3 : ratio = 3 / 5) 
(h4 : (speedB + ratio * speedB) * time = distance) : speedB = 100 := 
by 
  sorry

end carB_speed_l81_81799


namespace general_term_formula_for_sequence_l81_81151

theorem general_term_formula_for_sequence (a b : ℕ → ℝ) 
  (h1 : ∀ n, 2 * b n = a n + a (n + 1)) 
  (h2 : ∀ n, (a (n + 1))^2 = b n * b (n + 1)) 
  (h3 : a 1 = 1) 
  (h4 : a 2 = 3) :
  ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_formula_for_sequence_l81_81151


namespace hyperbola_focal_distance_distance_focus_to_asymptote_l81_81118

theorem hyperbola_focal_distance :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  (2 * c = 4) :=
by sorry

theorem distance_focus_to_asymptote :
  let a := 1
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let focus := (c, 0)
  let A := -Real.sqrt 3
  let B := 1
  let C := 0
  let distance := (|A * focus.fst + B * focus.snd + C|) / Real.sqrt (A ^ 2 + B ^ 2)
  (distance = Real.sqrt 3) :=
by sorry

end hyperbola_focal_distance_distance_focus_to_asymptote_l81_81118


namespace intersection_of_A_and_B_l81_81311

def setA : Set ℝ := {x : ℝ | |x| > 1}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : setA ∩ setB = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_A_and_B_l81_81311


namespace union_A_B_comp_U_A_inter_B_range_of_a_l81_81407

namespace ProofProblem

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := Set.univ

theorem union_A_B : A ∪ B = { x | 1 < x ∧ x ≤ 8 } := by
  sorry

theorem comp_U_A_inter_B : (U \ A) ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by
  sorry

end ProofProblem

end union_A_B_comp_U_A_inter_B_range_of_a_l81_81407


namespace hyperbola_foci_y_axis_condition_l81_81367

theorem hyperbola_foci_y_axis_condition (m n : ℝ) (h : m * n < 0) : 
  (mx^2 + ny^2 = 1) →
  (m < 0 ∧ n > 0) :=
sorry

end hyperbola_foci_y_axis_condition_l81_81367


namespace other_root_of_quadratic_l81_81309

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end other_root_of_quadratic_l81_81309


namespace probability_nina_taller_than_lena_l81_81128

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l81_81128


namespace find_circle_center_l81_81840

noncomputable def circle_center_lemma (a b : ℝ) : Prop :=
  -- Condition: Circle passes through (1, 0)
  (a - 1)^2 + b^2 = (a - 1)^2 + (b - 0)^2 ∧
  -- Condition: Circle is tangent to the parabola y = x^2 at (1, 1)
  (a - 1)^2 + (b - 1)^2 = 0

theorem find_circle_center : ∃ a b : ℝ, circle_center_lemma a b ∧ a = 1 ∧ b = 1 :=
by
  sorry

end find_circle_center_l81_81840


namespace intersection_sums_l81_81936

theorem intersection_sums :
  (∀ (x y : ℝ), (y = x^3 - 3 * x - 4) → (x + 3 * y = 3) → (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
  (y1 = x1^3 - 3 * x1 - 4) ∧ (x1 + 3 * y1 = 3) ∧
  (y2 = x2^3 - 3 * x2 - 4) ∧ (x2 + 3 * y2 = 3) ∧
  (y3 = x3^3 - 3 * x3 - 4) ∧ (x3 + 3 * y3 = 3) ∧
  x1 + x2 + x3 = 8 / 3 ∧ y1 + y2 + y3 = 19 / 9)) :=
sorry

end intersection_sums_l81_81936


namespace clock_shows_four_different_digits_for_588_minutes_l81_81193

-- Definition of the problem
def isFourDifferentDigits (h1 h2 m1 m2 : Nat) : Bool :=
  (h1 ≠ h2) && (h1 ≠ m1) && (h1 ≠ m2) && (h2 ≠ m1) && (h2 ≠ m2) && (m1 ≠ m2)

noncomputable def countFourDifferentDigitsMinutes : Nat :=
  let validMinutes := List.filter (λ (t : Nat × Nat),
    let (h, m) := t
    let h1 := h / 10
    let h2 := h % 10
    let m1 := m / 10
    let m2 := m % 10
    isFourDifferentDigits h1 h2 m1 m2
  ) (List.product (List.range 24) (List.range 60))
  validMinutes.length

theorem clock_shows_four_different_digits_for_588_minutes :
  countFourDifferentDigitsMinutes = 588 := sorry

end clock_shows_four_different_digits_for_588_minutes_l81_81193


namespace MrsHilt_money_left_l81_81453

theorem MrsHilt_money_left (initial_amount pencil_cost remaining_amount : ℕ) 
  (h_initial : initial_amount = 15) 
  (h_cost : pencil_cost = 11) 
  (h_remaining : remaining_amount = 4) : 
  initial_amount - pencil_cost = remaining_amount := 
by 
  sorry

end MrsHilt_money_left_l81_81453


namespace smallest_d_l81_81516

theorem smallest_d (d : ℕ) (h_pos : 0 < d) (h_square : ∃ k : ℕ, 3150 * d = k^2) : d = 14 :=
sorry

end smallest_d_l81_81516


namespace log_27_3_l81_81560

noncomputable def log_base : ℝ → ℝ → ℝ
| b, x := Real.log x / Real.log b

theorem log_27_3 :
  log_base 27 3 = 1 / 3 :=
by
  have h1 : 27 = 3 ^ 3 := by norm_num
  have h2 : ∀ a k b, log_base (a ^ k) b = (1 / k) * log_base a b := by
    intros a k b
    rw [log_base, log_base, Real.log_pow, mul_inv_cancel]
    norm_num
  have h3 : log_base 3 3 = 1 := by
    rw [log_base, Real.log_self]
  rw [h2 3 3 3, h3, mul_one, one_div]
  norm_num

end log_27_3_l81_81560


namespace find_product_abcd_l81_81751

def prod_abcd (a b c d : ℚ) :=
  4 * a - 2 * b + 3 * c + 5 * d = 22 ∧
  2 * (d + c) = b - 2 ∧
  4 * b - c = a + 1 ∧
  c + 1 = 2 * d

theorem find_product_abcd (a b c d : ℚ) (h : prod_abcd a b c d) :
  a * b * c * d = -30751860 / 11338912 :=
sorry

end find_product_abcd_l81_81751


namespace find_f_80_l81_81352

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_relation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  f (x * y) = f x / y^2

axiom f_40 : f 40 = 50

-- Proof that f 80 = 12.5
theorem find_f_80 : f 80 = 12.5 := 
by
  sorry

end find_f_80_l81_81352


namespace max_consecutive_integers_sum_lt_500_l81_81494

theorem max_consecutive_integers_sum_lt_500 
  (n : ℕ)
  (h₁ : ∑ k in Finset.range (n+1), k = n * (n + 1) / 2)
  (h₂ : n * (n + 1) / 2 < 500) 
  (h₃ : (n + 1) * (n + 2) / 2 > 500)
  : n = 31 := 
by sorry

end max_consecutive_integers_sum_lt_500_l81_81494


namespace quadratic_roots_condition_l81_81938

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1*x1 + m*x1 + 4 = 0 ∧ x2*x2 + m*x2 + 4 = 0) →
  m ≤ -4 :=
by
  sorry

end quadratic_roots_condition_l81_81938


namespace other_root_l81_81761

theorem other_root (m : ℝ) (h : 1^2 + m*1 + 3 = 0) : 
  ∃ α : ℝ, (1 + α = -m ∧ 1 * α = 3) ∧ α = 3 := 
by 
  sorry

end other_root_l81_81761


namespace math_proof_problem_l81_81383

theorem math_proof_problem : 
  (325 - Real.sqrt 125) / 425 = 65 - 5 := 
by sorry

end math_proof_problem_l81_81383


namespace radius_of_circle_zero_l81_81111

theorem radius_of_circle_zero (x y : ℝ) :
    (x^2 + 4*x + y^2 - 2*y + 5 = 0) → 0 = 0 :=
by
  sorry

end radius_of_circle_zero_l81_81111


namespace rectangle_coloring_l81_81756

theorem rectangle_coloring (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  ∃ (num_colorings : ℕ), num_colorings = 18 * 2^(m*n - 1) * 3^(m+n-2) :=
by
  use 18 * 2^(m*n - 1) * 3^(m+n-2)
  sorry

end rectangle_coloring_l81_81756


namespace bob_weight_l81_81225

theorem bob_weight (j b : ℝ) (h1 : j + b = 200) (h2 : b - j = b / 3) : b = 120 :=
sorry

end bob_weight_l81_81225


namespace initial_machines_l81_81696

theorem initial_machines (n x : ℕ) (hx : x > 0) (h : x / (4 * n) = x / 20) : n = 5 :=
by sorry

end initial_machines_l81_81696


namespace solve_quadratic_l81_81144

theorem solve_quadratic (x : ℝ) (h : (9 / x^2) - (6 / x) + 1 = 0) : 2 / x = 2 / 3 :=
by
  sorry

end solve_quadratic_l81_81144


namespace Andrew_is_19_l81_81255

-- Define individuals and their relationships
def Andrew_age (Bella_age : ℕ) : ℕ := Bella_age - 5
def Bella_age (Carlos_age : ℕ) : ℕ := Carlos_age + 4
def Carlos_age : ℕ := 20

-- Formulate the problem statement
theorem Andrew_is_19 : Andrew_age (Bella_age Carlos_age) = 19 :=
by
  sorry

end Andrew_is_19_l81_81255


namespace downstream_speed_is_40_l81_81070

variable (Vu : ℝ) (Vs : ℝ) (Vd : ℝ)

theorem downstream_speed_is_40 (h1 : Vu = 26) (h2 : Vs = 33) :
  Vd = 40 :=
by
  sorry

end downstream_speed_is_40_l81_81070


namespace number_of_pupils_l81_81512

theorem number_of_pupils (n : ℕ) (h1 : 79 - 45 = 34)
  (h2 : 34 = 1 / 2 * n) : n = 68 :=
by
  sorry

end number_of_pupils_l81_81512


namespace number_of_adult_female_alligators_l81_81324

-- Define the conditions
def total_alligators (females males: ℕ) : ℕ := females + males

def male_alligators : ℕ := 25
def female_alligators : ℕ := 25
def juvenile_percentage : ℕ := 40

-- Calculate the number of juveniles
def juvenile_count : ℕ := (juvenile_percentage * female_alligators) / 100

-- Calculate the number of adults
def adult_female_alligators : ℕ := female_alligators - juvenile_count

-- The main theorem statement
theorem number_of_adult_female_alligators : adult_female_alligators = 15 :=
by
    sorry

end number_of_adult_female_alligators_l81_81324


namespace four_diff_digits_per_day_l81_81198

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l81_81198


namespace polygon_area_is_1008_l81_81272

variables (vertices : List (ℕ × ℕ)) (units : ℕ)

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
sorry -- The function would compute the area based on vertices.

theorem polygon_area_is_1008 :
  vertices = [(0, 0), (12, 0), (24, 12), (24, 0), (36, 0), (36, 24), (24, 36), (12, 36), (0, 36), (0, 24), (0, 0)] →
  units = 1 →
  polygon_area vertices = 1008 :=
sorry

end polygon_area_is_1008_l81_81272


namespace relay_team_member_distance_l81_81359

theorem relay_team_member_distance (n_people : ℕ) (total_distance : ℕ)
  (h1 : n_people = 5) (h2 : total_distance = 150) : total_distance / n_people = 30 :=
by 
  sorry

end relay_team_member_distance_l81_81359


namespace least_positive_whole_number_divisible_by_five_primes_l81_81045

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l81_81045


namespace calculate_expression_l81_81105

noncomputable def solve_expression : ℝ :=
  let term1 := (real.pi - 1) ^ 0
  let term2 := 4 * real.sin (real.pi / 4) -- sin 45° = sin (π/4)
  let term3 := real.sqrt 8
  let term4 := real.abs (-3)
  term1 + term2 - term3 + term4

theorem calculate_expression : solve_expression = 4 := by
  sorry

end calculate_expression_l81_81105


namespace distance_between_vertices_l81_81734

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l81_81734


namespace largest_trailing_zeros_l81_81380

def count_trailing_zeros (n : Nat) : Nat :=
  if n = 0 then 0
  else Nat.min (Nat.factorial (n / 10)) (Nat.factorial (n / 5))

theorem largest_trailing_zeros :
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^5 * 3^4 * 5^6)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^4 * 3^4 * 5^5)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (4^2 * 5^4 * 6^3)) :=
  sorry

end largest_trailing_zeros_l81_81380


namespace max_n_no_constant_term_l81_81145

theorem max_n_no_constant_term (n : ℕ) (h : n < 10 ∧ n ≠ 3 ∧ n ≠ 6 ∧ n ≠ 9 ∧ n ≠ 2 ∧ n ≠ 5 ∧ n ≠ 8): n ≤ 7 :=
by {
  sorry
}

end max_n_no_constant_term_l81_81145


namespace triangle_altitude_l81_81189

variable (Area : ℝ) (base : ℝ) (altitude : ℝ)

theorem triangle_altitude (hArea : Area = 1250) (hbase : base = 50) :
  2 * Area / base = altitude :=
by
  sorry

end triangle_altitude_l81_81189


namespace ratio_of_a_b_l81_81873

theorem ratio_of_a_b (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : a / b = 3 / 2 :=
by sorry

end ratio_of_a_b_l81_81873


namespace estimate_total_children_l81_81580

variables (k m n T : ℕ)

/-- There are k children initially given red ribbons. 
    Then m children are randomly selected, 
    and n of them have red ribbons. -/

theorem estimate_total_children (h : n * T = k * m) : T = k * m / n :=
by sorry

end estimate_total_children_l81_81580


namespace problem_statement_l81_81747

theorem problem_statement {x₁ x₂ : ℝ} (h1 : 3 * x₁^2 - 9 * x₁ - 21 = 0) (h2 : 3 * x₂^2 - 9 * x₂ - 21 = 0) :
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := sorry

end problem_statement_l81_81747


namespace projectile_height_30_in_2_seconds_l81_81801

theorem projectile_height_30_in_2_seconds (t y : ℝ) : 
  (y = -5 * t^2 + 25 * t ∧ y = 30) → t = 2 :=
by
  sorry

end projectile_height_30_in_2_seconds_l81_81801


namespace no_four_distinct_numbers_l81_81998

theorem no_four_distinct_numbers (x y : ℝ) (h : x ≠ y ∧ 
    (x^(10:ℕ) + (x^(9:ℕ)) * y + (x^(8:ℕ)) * (y^(2:ℕ)) + 
    (x^(7:ℕ)) * (y^(3:ℕ)) + (x^(6:ℕ)) * (y^(4:ℕ)) + 
    (x^(5:ℕ)) * (y^(5:ℕ)) + (x^(4:ℕ)) * (y^(6:ℕ)) + 
    (x^(3:ℕ)) * (y^(7:ℕ)) + (x^(2:ℕ)) * (y^(8:ℕ)) + 
    (x^(1:ℕ)) * (y^(9:ℕ)) + (y^(10:ℕ)) = 1)) : False :=
by
  sorry

end no_four_distinct_numbers_l81_81998


namespace egg_sales_l81_81230

/-- Two vendors together sell 110 eggs and both have equal revenues.
    Given the conditions about changing the number of eggs and corresponding revenues,
    the first vendor sells 60 eggs and the second vendor sells 50 eggs. -/
theorem egg_sales (x y : ℝ) (h1 : x + (110 - x) = 110) (h2 : 110 * (y / x) = 5) (h3 : 110 * (y / (110 - x)) = 7.2) :
  x = 60 ∧ (110 - x) = 50 :=
by sorry

end egg_sales_l81_81230


namespace kickball_students_total_l81_81633

theorem kickball_students_total :
  let students_wednesday := 37
  let students_thursday := students_wednesday - 9
  students_wednesday + students_thursday = 65 :=
by 
  let students_wednesday := 37
  let students_thursday := students_wednesday - 9
  have h1 : students_thursday = 28 := 
    by rw [students_thursday, students_wednesday]; norm_num
  have h2 : students_wednesday + students_thursday = 65 := 
    by rw [h1]; norm_num
  exact h2

end kickball_students_total_l81_81633


namespace blue_pens_removed_l81_81970

def initial_blue_pens := 9
def initial_black_pens := 21
def initial_red_pens := 6
def removed_black_pens := 7
def pens_left := 25

theorem blue_pens_removed (x : ℕ) :
  initial_blue_pens - x + (initial_black_pens - removed_black_pens) + initial_red_pens = pens_left ↔ x = 4 := 
by 
  sorry

end blue_pens_removed_l81_81970


namespace time_to_fill_cistern_l81_81950

-- Define the rates at which each pipe fills or empties the cistern
def rate_A := 1 / 10
def rate_B := 1 / 12
def rate_C := -1 / 40

-- Define the combined rate when all pipes are opened simultaneously
def combined_rate := rate_A + rate_B + rate_C

-- Define the expected time to fill the cistern
noncomputable def expected_time_to_fill := 120 / 19

-- The main theorem to prove that the expected time to fill the cistern is correct.
theorem time_to_fill_cistern :
  (1 : ℚ) / combined_rate = expected_time_to_fill :=
sorry

end time_to_fill_cistern_l81_81950


namespace height_of_david_l81_81855

theorem height_of_david
  (building_height : ℕ)
  (building_shadow : ℕ)
  (david_shadow : ℕ)
  (ratio : ℕ)
  (h1 : building_height = 50)
  (h2 : building_shadow = 25)
  (h3 : david_shadow = 18)
  (h4 : ratio = building_height / building_shadow) :
  david_shadow * ratio = 36 := sorry

end height_of_david_l81_81855


namespace jiahao_estimate_larger_l81_81665

variable (x y : ℝ)
variable (hxy : x > y)
variable (hy0 : y > 0)

theorem jiahao_estimate_larger (x y : ℝ) (hxy : x > y) (hy0 : y > 0) :
  (x + 2) - (y - 1) > x - y :=
by
  sorry

end jiahao_estimate_larger_l81_81665


namespace probability_nina_taller_than_lena_l81_81127

variables {M N L O : ℝ}

theorem probability_nina_taller_than_lena (h₁ : N < M) (h₂ : L > O) : 
  ∃ P : ℝ, P = 0 ∧ ∀ M N L O, M ≠ N ∧ M ≠ L ∧ M ≠ O ∧ N ≠ L ∧ N ≠ O ∧ L ≠ O → 
  (M > N → O < L → P = 0) :=
by sorry

end probability_nina_taller_than_lena_l81_81127


namespace aquariums_have_13_saltwater_animals_l81_81439

theorem aquariums_have_13_saltwater_animals:
  ∀ x : ℕ, 26 * x = 52 → (∀ n : ℕ, n = 26 → (n * x = 52 ∧ x % 2 = 1 ∧ x > 1)) → x = 13 :=
by
  sorry

end aquariums_have_13_saltwater_animals_l81_81439


namespace girl_travel_distance_l81_81069

def speed : ℝ := 6 -- meters per second
def time : ℕ := 16 -- seconds

def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem girl_travel_distance : distance speed time = 96 :=
by 
  unfold distance
  sorry

end girl_travel_distance_l81_81069


namespace books_of_jason_l81_81168

theorem books_of_jason (M J : ℕ) (hM : M = 42) (hTotal : M + J = 60) : J = 18 :=
by
  sorry

end books_of_jason_l81_81168


namespace only_integer_solution_is_zero_l81_81836

theorem only_integer_solution_is_zero (x y : ℤ) (h : x^4 + y^4 = 3 * x^3 * y) : x = 0 ∧ y = 0 :=
by {
  -- Here we would provide the proof steps.
  sorry
}

end only_integer_solution_is_zero_l81_81836


namespace circle_bisect_line_l81_81425

theorem circle_bisect_line (a : ℝ) :
  (∃ x y, (x - a) ^ 2 + (y + 1) ^ 2 = 3 ∧ 5 * x + 4 * y - a = 0) →
  a = 1 :=
by
  sorry

end circle_bisect_line_l81_81425


namespace area_of_triangle_arithmetic_sides_l81_81587

theorem area_of_triangle_arithmetic_sides 
  (a : ℝ) (h : a > 0) (h_sin : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2) :
  let s₁ := a - 2
  let s₂ := a
  let s₃ := a + 2
  ∃ (a b c : ℝ), 
    a = s₁ ∧ b = s₂ ∧ c = s₃ ∧ 
    Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 → 
    (1/2 * s₁ * s₂ * Real.sin (2 * Real.pi / 3) = 15 * Real.sqrt 3 / 4) :=
by
  sorry

end area_of_triangle_arithmetic_sides_l81_81587


namespace real_and_equal_roots_condition_l81_81977

theorem real_and_equal_roots_condition (k : ℝ) : 
  ∀ k : ℝ, (∃ (x : ℝ), 3 * x^2 + 6 * k * x + 9 = 0) ↔ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end real_and_equal_roots_condition_l81_81977


namespace problem_statements_correctness_l81_81112

theorem problem_statements_correctness :
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (12 ∣ 72 ∧ 12 ∣ 120) ∧ (7 ∣ 49 ∧ 7 ∣ 84) ∧ (7 ∣ 63) → 
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (7 ∣ 63) :=
by
  intro h
  sorry

end problem_statements_correctness_l81_81112


namespace bounded_g_of_f_l81_81444

theorem bounded_g_of_f
  (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := 
sorry

end bounded_g_of_f_l81_81444


namespace geometric_sequence_product_l81_81626

variable (a : ℕ → ℝ)

def is_geometric_seq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (h_geom : is_geometric_seq a) (h_a6 : a 6 = 3) :
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end geometric_sequence_product_l81_81626


namespace simplify_and_evaluate_l81_81640

theorem simplify_and_evaluate (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hx3 : x ≠ -2) (hx4 : x = -1) :
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = -2 :=
by
  sorry

end simplify_and_evaluate_l81_81640


namespace min_cost_for_boxes_l81_81960

theorem min_cost_for_boxes
  (box_length: ℕ) (box_width: ℕ) (box_height: ℕ)
  (cost_per_box: ℝ) (total_volume: ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 15)
  (h4 : cost_per_box = 1.30)
  (h5 : total_volume = 3060000) :
  ∃ cost: ℝ, cost = 663 :=
by
  sorry

end min_cost_for_boxes_l81_81960


namespace product_of_roots_l81_81852

theorem product_of_roots : 
  ∀ (r1 r2 r3 : ℝ), (2 * r1 * r2 * r3 - 3 * (r1 * r2 + r2 * r3 + r3 * r1) - 15 * (r1 + r2 + r3) + 35 = 0) → 
  (r1 * r2 * r3 = -35 / 2) :=
by
  sorry

end product_of_roots_l81_81852


namespace students_passing_in_sixth_year_l81_81898

def numStudentsPassed (year : ℕ) : ℕ :=
 if year = 1 then 200 else 
 if year = 2 then 300 else 
 if year = 3 then 390 else 
 if year = 4 then 565 else 
 if year = 5 then 643 else 
 if year = 6 then 780 else 0

theorem students_passing_in_sixth_year : numStudentsPassed 6 = 780 := by
  sorry

end students_passing_in_sixth_year_l81_81898
