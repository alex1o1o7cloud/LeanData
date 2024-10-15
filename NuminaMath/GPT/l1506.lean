import Mathlib

namespace NUMINAMATH_GPT_factorize_polynomial_l1506_150644
   
   -- Define the polynomial
   def polynomial (x : ℝ) : ℝ :=
     x^3 + 3 * x^2 - 4
   
   -- Define the factorized form
   def factorized_form (x : ℝ) : ℝ :=
     (x - 1) * (x + 2)^2
   
   -- The theorem statement
   theorem factorize_polynomial (x : ℝ) : polynomial x = factorized_form x := 
   by
     sorry
   
end NUMINAMATH_GPT_factorize_polynomial_l1506_150644


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1506_150697

noncomputable def has_solution_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ (x^2 + a*x - 2 < 0)

theorem range_of_a_for_inequality : ∀ a : ℝ, has_solution_in_interval a ↔ a < 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l1506_150697


namespace NUMINAMATH_GPT_smallest_integer_remainder_l1506_150633

theorem smallest_integer_remainder :
  ∃ n : ℕ, n > 1 ∧
           (n % 3 = 2) ∧
           (n % 4 = 2) ∧
           (n % 5 = 2) ∧
           (n % 7 = 2) ∧
           n = 422 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_remainder_l1506_150633


namespace NUMINAMATH_GPT_minimum_value_l1506_150636

-- Define geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * ((a 2 / a 1) ^ n)

-- Define the condition for positive geometric sequence
def positive_geometric_sequence (a : ℕ → ℝ) :=
  is_geometric_sequence a ∧ ∀ n : ℕ, a n > 0

-- Condition given in the problem
def condition (a : ℕ → ℝ) :=
  2 * a 4 + a 3 = 2 * a 2 + a 1 + 8

-- Define the problem statement to be proved
theorem minimum_value (a : ℕ → ℝ) (h1 : positive_geometric_sequence a) (h2 : condition a) :
  2 * a 6 + a 5 = 32 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1506_150636


namespace NUMINAMATH_GPT_problem_l1506_150692

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (x - Real.pi / 2)

theorem problem 
: (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ c, c = (Real.pi / 2) ∧ f c = 0) → (T = Real.pi ∧ c = (Real.pi / 2)) :=
sorry

end NUMINAMATH_GPT_problem_l1506_150692


namespace NUMINAMATH_GPT_dependence_of_Q_l1506_150606

theorem dependence_of_Q (a d k : ℕ) :
    ∃ (Q : ℕ), Q = (2 * k * (2 * a + 4 * k * d - d)) 
                - (k * (2 * a + (2 * k - 1) * d)) 
                - (k / 2 * (2 * a + (k - 1) * d)) 
                → Q = k * a + 13 * k^2 * d := 
sorry

end NUMINAMATH_GPT_dependence_of_Q_l1506_150606


namespace NUMINAMATH_GPT_sum_fractions_geq_six_l1506_150611

variable (x y z : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

theorem sum_fractions_geq_six : 
  (x / y + y / z + z / x + x / z + z / y + y / x) ≥ 6 := 
by
  sorry

end NUMINAMATH_GPT_sum_fractions_geq_six_l1506_150611


namespace NUMINAMATH_GPT_salt_mixture_problem_l1506_150670

theorem salt_mixture_problem :
  ∃ (m : ℝ), 0.20 = (150 + 0.05 * m) / (600 + m) :=
by
  sorry

end NUMINAMATH_GPT_salt_mixture_problem_l1506_150670


namespace NUMINAMATH_GPT_distinct_values_of_b_l1506_150607

theorem distinct_values_of_b : ∃ b_list : List ℝ, b_list.length = 8 ∧ ∀ b ∈ b_list, ∃ p q : ℤ, p + q = b ∧ p * q = 8 * b :=
by
  sorry

end NUMINAMATH_GPT_distinct_values_of_b_l1506_150607


namespace NUMINAMATH_GPT_no_int_representation_l1506_150616

theorem no_int_representation (A B : ℤ) : (99999 + 111111 * Real.sqrt 3) ≠ (A + B * Real.sqrt 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_no_int_representation_l1506_150616


namespace NUMINAMATH_GPT_bill_weight_training_l1506_150653

theorem bill_weight_training (jugs : ℕ) (gallons_per_jug : ℝ) (percent_filled : ℝ) (density : ℝ) 
  (h_jugs : jugs = 2)
  (h_gallons_per_jug : gallons_per_jug = 2)
  (h_percent_filled : percent_filled = 0.70)
  (h_density : density = 5) :
  jugs * gallons_per_jug * percent_filled * density = 14 := 
by
  subst h_jugs
  subst h_gallons_per_jug
  subst h_percent_filled
  subst h_density
  norm_num
  done

end NUMINAMATH_GPT_bill_weight_training_l1506_150653


namespace NUMINAMATH_GPT_sum_of_coefficients_l1506_150637

def P (x : ℝ) : ℝ :=
  -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : P 1 = 48 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1506_150637


namespace NUMINAMATH_GPT_gumball_cost_l1506_150622

theorem gumball_cost (n : ℕ) (T : ℕ) (h₁ : n = 4) (h₂ : T = 32) : T / n = 8 := by
  sorry

end NUMINAMATH_GPT_gumball_cost_l1506_150622


namespace NUMINAMATH_GPT_wine_price_increase_l1506_150662

-- Definitions translating the conditions
def wine_cost_today : ℝ := 20.0
def bottles_count : ℕ := 5
def tariff_rate : ℝ := 0.25

-- Statement to prove
theorem wine_price_increase (wine_cost_today : ℝ) (bottles_count : ℕ) (tariff_rate : ℝ) : 
  bottles_count * wine_cost_today * tariff_rate = 25.0 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_wine_price_increase_l1506_150662


namespace NUMINAMATH_GPT_Vikas_submitted_6_questions_l1506_150602

theorem Vikas_submitted_6_questions (R V A : ℕ) (h1 : 7 * V = 3 * R) (h2 : 2 * V = 3 * A) (h3 : R + V + A = 24) : V = 6 :=
by
  sorry

end NUMINAMATH_GPT_Vikas_submitted_6_questions_l1506_150602


namespace NUMINAMATH_GPT_smallest_perimeter_scalene_triangle_l1506_150642

theorem smallest_perimeter_scalene_triangle (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a) :
  a + b + c = 9 := 
sorry

end NUMINAMATH_GPT_smallest_perimeter_scalene_triangle_l1506_150642


namespace NUMINAMATH_GPT_find_v2002_l1506_150661

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 6
  | 4 => 2
  | 5 => 1
  | 6 => 7
  | 7 => 4
  | _ => 0

def seq_v : ℕ → ℕ
| 0       => 5
| (n + 1) => g (seq_v n)

theorem find_v2002 : seq_v 2002 = 5 :=
  sorry

end NUMINAMATH_GPT_find_v2002_l1506_150661


namespace NUMINAMATH_GPT_remainder_div_by_7_l1506_150634

theorem remainder_div_by_7 (n : ℤ) (k m : ℤ) (r : ℤ) (h₀ : n = 7 * k + r) (h₁ : 3 * n = 7 * m + 3) (hrange : 0 ≤ r ∧ r < 7) : r = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_by_7_l1506_150634


namespace NUMINAMATH_GPT_least_multiple_x_correct_l1506_150600

noncomputable def least_multiple_x : ℕ :=
  let x := 20
  let y := 8
  let z := 5
  5 * y

theorem least_multiple_x_correct (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 33) (h5 : 5 * y = 8 * z) : least_multiple_x = 40 :=
by
  sorry

end NUMINAMATH_GPT_least_multiple_x_correct_l1506_150600


namespace NUMINAMATH_GPT_product_div_by_six_l1506_150632

theorem product_div_by_six (A B C : ℤ) (h1 : A^2 + B^2 = C^2) 
  (h2 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 4 * k + 2) 
  (h3 : ∀ n : ℤ, ¬ ∃ k : ℤ, n^2 = 3 * k + 2) : 
  6 ∣ (A * B) :=
sorry

end NUMINAMATH_GPT_product_div_by_six_l1506_150632


namespace NUMINAMATH_GPT_gcd_150_450_l1506_150612

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end NUMINAMATH_GPT_gcd_150_450_l1506_150612


namespace NUMINAMATH_GPT_questionnaires_drawn_from_unit_D_l1506_150605

theorem questionnaires_drawn_from_unit_D 
  (total_sample: ℕ) 
  (sample_from_B: ℕ) 
  (d: ℕ) 
  (h_total_sample: total_sample = 150) 
  (h_sample_from_B: sample_from_B = 30) 
  (h_arithmetic_sequence: (30 - d) + 30 + (30 + d) + (30 + 2 * d) = total_sample) 
  : 30 + 2 * d = 60 :=
by 
  sorry

end NUMINAMATH_GPT_questionnaires_drawn_from_unit_D_l1506_150605


namespace NUMINAMATH_GPT_evaluation_of_expression_l1506_150663

theorem evaluation_of_expression
  (a b x y m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * (|m|) - 2 * (x * y) = 1 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_evaluation_of_expression_l1506_150663


namespace NUMINAMATH_GPT_ways_to_distribute_balls_l1506_150654

theorem ways_to_distribute_balls :
  let balls : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
  let boxes : Finset ℕ := {0, 1, 2, 3}
  let choose_distinct (n k : ℕ) : ℕ := Nat.choose n k
  let distribution_patterns : List (ℕ × ℕ × ℕ × ℕ) := 
    [(6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0), 
     (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)]
  let ways_to_pattern (pattern : ℕ × ℕ × ℕ × ℕ) : ℕ :=
    match pattern with
    | (6,0,0,0) => 1
    | (5,1,0,0) => choose_distinct 6 5
    | (4,2,0,0) => choose_distinct 6 4 * choose_distinct 2 2
    | (4,1,1,0) => choose_distinct 6 4
    | (3,3,0,0) => choose_distinct 6 3 * choose_distinct 3 3 / 2
    | (3,2,1,0) => choose_distinct 6 3 * choose_distinct 3 2 * choose_distinct 1 1
    | (3,1,1,1) => choose_distinct 6 3
    | (2,2,2,0) => choose_distinct 6 2 * choose_distinct 4 2 * choose_distinct 2 2 / 6
    | (2,2,1,1) => choose_distinct 6 2 * choose_distinct 4 2 / 2
    | _ => 0
  let total_ways : ℕ := distribution_patterns.foldl (λ acc x => acc + ways_to_pattern x) 0
  total_ways = 182 := by
  sorry

end NUMINAMATH_GPT_ways_to_distribute_balls_l1506_150654


namespace NUMINAMATH_GPT_value_divided_by_l1506_150696

theorem value_divided_by {x : ℝ} : (5 / x) * 12 = 10 → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_divided_by_l1506_150696


namespace NUMINAMATH_GPT_lawn_length_l1506_150679

-- Defining the main conditions
def area : ℕ := 20
def width : ℕ := 5

-- The proof statement (goal)
theorem lawn_length : (area / width) = 4 := by
  sorry

end NUMINAMATH_GPT_lawn_length_l1506_150679


namespace NUMINAMATH_GPT_least_number_subtracted_divisible_by_six_l1506_150699

theorem least_number_subtracted_divisible_by_six :
  ∃ d : ℕ, d = 6 ∧ (427398 - 6) % d = 0 := by
sorry

end NUMINAMATH_GPT_least_number_subtracted_divisible_by_six_l1506_150699


namespace NUMINAMATH_GPT_solve_inequality_l1506_150650

theorem solve_inequality (x : ℝ) : 
  1 / (x^2 + 2) > 4 / x + 21 / 10 ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1506_150650


namespace NUMINAMATH_GPT_percentage_paid_l1506_150671

/-- 
Given the marked price is 80% of the suggested retail price,
and Alice paid 60% of the marked price,
prove that the percentage of the suggested retail price Alice paid is 48%.
-/
theorem percentage_paid (P : ℝ) (MP : ℝ) (price_paid : ℝ)
  (h1 : MP = 0.80 * P)
  (h2 : price_paid = 0.60 * MP) :
  (price_paid / P) * 100 = 48 := 
sorry

end NUMINAMATH_GPT_percentage_paid_l1506_150671


namespace NUMINAMATH_GPT_Carl_chops_more_onions_than_Brittney_l1506_150640

theorem Carl_chops_more_onions_than_Brittney :
  let Brittney_rate := 15 / 5
  let Carl_rate := 20 / 5
  let Brittney_onions := Brittney_rate * 30
  let Carl_onions := Carl_rate * 30
  Carl_onions = Brittney_onions + 30 :=
by
  sorry

end NUMINAMATH_GPT_Carl_chops_more_onions_than_Brittney_l1506_150640


namespace NUMINAMATH_GPT_sum_of_coordinates_D_l1506_150695

theorem sum_of_coordinates_D (M C D : ℝ × ℝ)
  (h1 : M = (5, 5))
  (h2 : C = (10, 10))
  (h3 : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 0 := 
sorry

end NUMINAMATH_GPT_sum_of_coordinates_D_l1506_150695


namespace NUMINAMATH_GPT_find_t1_t2_l1506_150629

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (1, 2)

-- Define the conditions for t1 and t2
def t1_condition (t1 : ℝ) : Prop := (2 / 1) = (t1 / 2)
def t2_condition (t2 : ℝ) : Prop := (2 * 1 + t2 * 2 = 0)

-- The statement to prove
theorem find_t1_t2 (t1 t2 : ℝ) (h1 : t1_condition t1) (h2 : t2_condition t2) : (t1 = 4) ∧ (t2 = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_t1_t2_l1506_150629


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t2_l1506_150688

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t2_l1506_150688


namespace NUMINAMATH_GPT_complement_union_A_B_l1506_150615

-- Define the sets U, A, and B as per the conditions
def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Specify the statement to prove the complement of A ∪ B with respect to U
theorem complement_union_A_B : (U \ (A ∪ B)) = {2, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_l1506_150615


namespace NUMINAMATH_GPT_square_side_length_l1506_150627

theorem square_side_length (s : ℝ) (h : s^2 = 1 / 9) : s = 1 / 3 :=
sorry

end NUMINAMATH_GPT_square_side_length_l1506_150627


namespace NUMINAMATH_GPT_amount_spent_on_marbles_l1506_150680

-- Definitions of conditions
def cost_of_football : ℝ := 5.71
def total_spent_on_toys : ℝ := 12.30

-- Theorem statement
theorem amount_spent_on_marbles : (total_spent_on_toys - cost_of_football) = 6.59 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_on_marbles_l1506_150680


namespace NUMINAMATH_GPT_derivative_f_l1506_150691

noncomputable def f (x : ℝ) : ℝ := 1 + Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = -Real.sin x := 
by 
  sorry

end NUMINAMATH_GPT_derivative_f_l1506_150691


namespace NUMINAMATH_GPT_find_f_neg_3_l1506_150694

theorem find_f_neg_3
    (a : ℝ)
    (f : ℝ → ℝ)
    (h : ∀ x, f x = a^2 * x^3 + a * Real.sin x + abs x + 1)
    (h_f3 : f 3 = 5) :
    f (-3) = 3 :=
by
    sorry

end NUMINAMATH_GPT_find_f_neg_3_l1506_150694


namespace NUMINAMATH_GPT_solve_inequality_l1506_150659

theorem solve_inequality (x : ℝ) : 
  let quad := (x - 2)^2 + 9
  let numerator := x - 3
  quad > 0 ∧ numerator ≥ 0 ↔ x ≥ 3 :=
by
    sorry

end NUMINAMATH_GPT_solve_inequality_l1506_150659


namespace NUMINAMATH_GPT_am_gm_iq_l1506_150631

theorem am_gm_iq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (a + 1/a) * (b + 1/b) ≥ 25/4 := sorry

end NUMINAMATH_GPT_am_gm_iq_l1506_150631


namespace NUMINAMATH_GPT_original_square_area_is_correct_l1506_150625

noncomputable def original_square_side_length (s : ℝ) :=
  let original_area := s^2
  let new_width := 0.8 * s
  let new_length := 5 * s
  let new_area := new_width * new_length
  let increased_area := new_area - original_area
  increased_area = 15.18

theorem original_square_area_is_correct (s : ℝ) (h : original_square_side_length s) : s^2 = 5.06 := by
  sorry

end NUMINAMATH_GPT_original_square_area_is_correct_l1506_150625


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1506_150620

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > b + 1) → (a > b) ∧ ¬(a > b → a > b + 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1506_150620


namespace NUMINAMATH_GPT_neg_sin_leq_one_l1506_150655

theorem neg_sin_leq_one (p : Prop) :
  (∀ x : ℝ, Real.sin x ≤ 1) → (¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end NUMINAMATH_GPT_neg_sin_leq_one_l1506_150655


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_not_neccessary_condition_l1506_150693

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  ((x + 3)^2 + (y - 4)^2 = 0) → ((x + 3) * (y - 4) = 0) :=
by { sorry }

theorem not_neccessary_condition (x y : ℝ) :
  ((x + 3) * (y - 4) = 0) ↔ ((x + 3)^2 + (y - 4)^2 = 0) :=
by { sorry }

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_not_neccessary_condition_l1506_150693


namespace NUMINAMATH_GPT_necklaces_made_l1506_150608

theorem necklaces_made (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 18) (h2 : beads_per_necklace = 3) : total_beads / beads_per_necklace = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_necklaces_made_l1506_150608


namespace NUMINAMATH_GPT_total_number_of_students_l1506_150681

theorem total_number_of_students 
  (b g : ℕ) 
  (ratio_condition : 5 * g = 8 * b) 
  (girls_count : g = 160) : 
  b + g = 260 := by
  sorry

end NUMINAMATH_GPT_total_number_of_students_l1506_150681


namespace NUMINAMATH_GPT_sum_mod_11_l1506_150626

theorem sum_mod_11 (h1 : 8735 % 11 = 1) (h2 : 8736 % 11 = 2) (h3 : 8737 % 11 = 3) (h4 : 8738 % 11 = 4) :
  (8735 + 8736 + 8737 + 8738) % 11 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_11_l1506_150626


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1506_150673

-- Definitions for the coordinates of the points
def A : ℤ × ℤ := (3, 2)
def B : ℤ × ℤ := (-3, -2)
def C : ℤ × ℤ := (3, -2)
def D : ℤ × ℤ := (-3, 2)

-- Definition for the second quadrant condition
def isSecondQuadrant (p : ℤ × ℤ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- The theorem we need to prove
theorem point_in_second_quadrant : isSecondQuadrant D :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1506_150673


namespace NUMINAMATH_GPT_third_stack_shorter_by_five_l1506_150686

theorem third_stack_shorter_by_five
    (first_stack second_stack third_stack fourth_stack : ℕ)
    (h1 : first_stack = 5)
    (h2 : second_stack = first_stack + 2)
    (h3 : fourth_stack = third_stack + 5)
    (h4 : first_stack + second_stack + third_stack + fourth_stack = 21) :
    second_stack - third_stack = 5 :=
by
  sorry

end NUMINAMATH_GPT_third_stack_shorter_by_five_l1506_150686


namespace NUMINAMATH_GPT_KHSO4_formed_l1506_150656

-- Define the reaction condition and result using moles
def KOH_moles : ℕ := 2
def H2SO4_moles : ℕ := 2

-- The balanced chemical reaction in terms of moles
-- 1 mole of KOH reacts with 1 mole of H2SO4 to produce 
-- 1 mole of KHSO4
def react (koh : ℕ) (h2so4 : ℕ) : ℕ := 
  -- stoichiometry 1:1 ratio of KOH and H2SO4 to KHSO4
  if koh ≤ h2so4 then koh else h2so4

-- The proof statement that verifies the expected number of moles of KHSO4
theorem KHSO4_formed (koh : ℕ) (h2so4 : ℕ) (hrs : react koh h2so4 = koh) : 
  koh = KOH_moles → h2so4 = H2SO4_moles → react koh h2so4 = 2 := 
by
  intros 
  sorry

end NUMINAMATH_GPT_KHSO4_formed_l1506_150656


namespace NUMINAMATH_GPT_find_pairs_l1506_150648

theorem find_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m^2 + n^2) ∣ (3 * m * n + 3 * m) ↔ (m, n) = (1, 1) ∨ (m, n) = (4, 2) ∨ (m, n) = (4, 10) :=
sorry

end NUMINAMATH_GPT_find_pairs_l1506_150648


namespace NUMINAMATH_GPT_original_number_is_80_l1506_150646

-- Define the existence of the numbers A and B
variable (A B : ℕ)

-- Define the conditions from the problem
def conditions :=
  A = 35 ∧ A / 7 = B / 9

-- Define the statement to prove
theorem original_number_is_80 (h : conditions A B) : A + B = 80 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_original_number_is_80_l1506_150646


namespace NUMINAMATH_GPT_average_of_numbers_l1506_150623

theorem average_of_numbers (x : ℝ) (h : (5 + -1 + -2 + x) / 4 = 1) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_of_numbers_l1506_150623


namespace NUMINAMATH_GPT_football_hits_ground_l1506_150669

theorem football_hits_ground :
  ∃ t : ℚ, -16 * t^2 + 18 * t + 60 = 0 ∧ 0 < t ∧ t = 41 / 16 :=
by
  sorry

end NUMINAMATH_GPT_football_hits_ground_l1506_150669


namespace NUMINAMATH_GPT_factorization_correct_l1506_150601

theorem factorization_correct (x : ℝ) : 
  98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1506_150601


namespace NUMINAMATH_GPT_find_x_in_terms_of_z_l1506_150652

variable (z : ℝ)
variable (x y : ℝ)

theorem find_x_in_terms_of_z (h1 : 0.35 * (400 + y) = 0.20 * x) 
                             (h2 : x = 2 * z^2) 
                             (h3 : y = 3 * z - 5) : 
  x = 2 * z^2 :=
by
  exact h2

end NUMINAMATH_GPT_find_x_in_terms_of_z_l1506_150652


namespace NUMINAMATH_GPT_magic_square_sum_l1506_150649

theorem magic_square_sum (v w x y z : ℤ)
    (h1 : 25 + z + 23 = 25 + x + w)
    (h2 : 18 + x + y = 25 + x + w)
    (h3 : v + 22 + w = 25 + x + w)
    (h4 : 25 + 18 + v = 25 + x + w)
    (h5 : z + x + 22 = 25 + x + w)
    (h6 : 23 + y + w = 25 + x + w)
    (h7 : 25 + x + w = 25 + x + w)
    (h8 : v + x + 23 = 25 + x + w) 
:
    y + z = 45 :=
by
  sorry

end NUMINAMATH_GPT_magic_square_sum_l1506_150649


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l1506_150690

theorem sum_of_consecutive_integers (a : ℤ) (h₁ : a = 18) (h₂ : a + 1 = 19) (h₃ : a + 2 = 20) : a + (a + 1) + (a + 2) = 57 :=
by
  -- Add a sorry to focus on creating the statement successfully
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l1506_150690


namespace NUMINAMATH_GPT_intersection_eq_l1506_150617

variable {x : ℝ}

def set_A := {x : ℝ | x^2 - 4 * x < 0}
def set_B := {x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5}
def set_intersection := {x : ℝ | 1 / 3 ≤ x ∧ x < 4}

theorem intersection_eq : (set_A ∩ set_B) = set_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1506_150617


namespace NUMINAMATH_GPT_total_spots_l1506_150604

variable (P : ℕ)
variable (Bill_spots : ℕ := 2 * P - 1)

-- Given conditions
variable (h1 : Bill_spots = 39)

-- Theorem we need to prove
theorem total_spots (P : ℕ) (Bill_spots : ℕ := 2 * P - 1) (h1 : Bill_spots = 39) : 
  Bill_spots + P = 59 := 
by
  sorry

end NUMINAMATH_GPT_total_spots_l1506_150604


namespace NUMINAMATH_GPT_find_k_l1506_150618

theorem find_k (k x : ℝ) (h1 : x + k - 4 = 0) (h2 : x = 2) : k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1506_150618


namespace NUMINAMATH_GPT_total_people_attended_l1506_150682

theorem total_people_attended (A C : ℕ) (ticket_price_adult ticket_price_child : ℕ) (total_receipts : ℕ) 
  (number_of_children : ℕ) (h_ticket_prices : ticket_price_adult = 60 ∧ ticket_price_child = 25)
  (h_total_receipts : total_receipts = 140 * 100) (h_children : C = 80) 
  (h_equation : ticket_price_adult * A + ticket_price_child * C = total_receipts) : 
  A + C = 280 :=
by
  sorry

end NUMINAMATH_GPT_total_people_attended_l1506_150682


namespace NUMINAMATH_GPT_gummy_bear_production_time_l1506_150603

theorem gummy_bear_production_time 
  (gummy_bears_per_minute : ℕ)
  (gummy_bears_per_packet : ℕ)
  (total_packets : ℕ)
  (h1 : gummy_bears_per_minute = 300)
  (h2 : gummy_bears_per_packet = 50)
  (h3 : total_packets = 240) :
  (total_packets / (gummy_bears_per_minute / gummy_bears_per_packet) = 40) :=
sorry

end NUMINAMATH_GPT_gummy_bear_production_time_l1506_150603


namespace NUMINAMATH_GPT_problem_statement_l1506_150684

def f (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n - 3

theorem problem_statement : f (f (f 3)) = 31 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1506_150684


namespace NUMINAMATH_GPT_dot_product_AB_BC_l1506_150672

variable (AB AC : ℝ × ℝ)

def BC (AB AC : ℝ × ℝ) : ℝ × ℝ := (AC.1 - AB.1, AC.2 - AB.2)

def dot_product (u v : ℝ × ℝ) : ℝ := (u.1 * v.1) + (u.2 * v.2)

theorem dot_product_AB_BC :
  ∀ (AB AC : ℝ × ℝ), AB = (2, 3) → AC = (3, 4) →
  dot_product AB (BC AB AC) = 5 :=
by
  intros
  unfold BC
  unfold dot_product
  sorry

end NUMINAMATH_GPT_dot_product_AB_BC_l1506_150672


namespace NUMINAMATH_GPT_empty_set_iff_k_single_element_set_iff_k_l1506_150641

noncomputable def quadratic_set (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

theorem empty_set_iff_k (k : ℝ) : 
  quadratic_set k = ∅ ↔ k > 9/8 := by
  sorry

theorem single_element_set_iff_k (k : ℝ) : 
  (∃ x : ℝ, quadratic_set k = {x}) ↔ (k = 0 ∧ quadratic_set k = {2 / 3}) ∨ (k = 9 / 8 ∧ quadratic_set k = {4 / 3}) := by
  sorry

end NUMINAMATH_GPT_empty_set_iff_k_single_element_set_iff_k_l1506_150641


namespace NUMINAMATH_GPT_heartsuit_calc_l1506_150674

-- Define the operation x ♡ y = 4x + 6y
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_calc : heartsuit 5 3 = 38 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_heartsuit_calc_l1506_150674


namespace NUMINAMATH_GPT_triangle_area_with_median_l1506_150609

theorem triangle_area_with_median (a b m : ℝ) (area : ℝ) 
  (h_a : a = 6) (h_b : b = 8) (h_m : m = 5) : 
  area = 24 :=
sorry

end NUMINAMATH_GPT_triangle_area_with_median_l1506_150609


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1506_150628

noncomputable def p (x : ℝ) : Prop := (1 - x^2 < 0 ∧ |x| - 2 > 0) ∨ (1 - x^2 > 0 ∧ |x| - 2 < 0)
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1506_150628


namespace NUMINAMATH_GPT_find_y_l1506_150668

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : x / y = 86 ∧ ((x % y : ℚ) / y = 0.12)) : y = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1506_150668


namespace NUMINAMATH_GPT_permutation_by_transpositions_l1506_150619

-- Formalizing the conditions in Lean
section permutations
  variable {n : ℕ}

  -- Define permutations
  def is_permutation (σ : Fin n → Fin n) : Prop :=
    ∃ σ_inv : Fin n → Fin n, 
      (∀ i, σ (σ_inv i) = i) ∧ 
      (∀ i, σ_inv (σ i) = i)

  -- Define transposition
  def transposition (σ : Fin n → Fin n) (i j : Fin n) : Fin n → Fin n :=
    fun x => if x = i then j else if x = j then i else σ x

  -- Main theorem stating that any permutation can be obtained through a series of transpositions
  theorem permutation_by_transpositions (σ : Fin n → Fin n) (h : is_permutation σ) :
    ∃ τ : ℕ → (Fin n → Fin n),
      (∀ i, is_permutation (τ i)) ∧
      (∀ m, ∃ k, τ m = transposition (τ (m - 1)) (⟨ k, sorry ⟩) (σ (⟨ k, sorry⟩))) ∧
      (∃ m, τ m = σ) :=
  sorry
end permutations

end NUMINAMATH_GPT_permutation_by_transpositions_l1506_150619


namespace NUMINAMATH_GPT_cat_moves_on_circular_arc_l1506_150685

theorem cat_moves_on_circular_arc (L : ℝ) (x y : ℝ)
  (h : x^2 + y^2 = L^2) :
  (x / 2)^2 + (y / 2)^2 = (L / 2)^2 :=
  by sorry

end NUMINAMATH_GPT_cat_moves_on_circular_arc_l1506_150685


namespace NUMINAMATH_GPT_rectangle_area_l1506_150624

-- Definitions of the conditions
variables (Length Width Area : ℕ)
variable (h1 : Length = 4 * Width)
variable (h2 : Length = 20)

-- Statement to prove
theorem rectangle_area : Area = Length * Width → Area = 100 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1506_150624


namespace NUMINAMATH_GPT_four_digit_number_perfect_square_l1506_150638

theorem four_digit_number_perfect_square (abcd : ℕ) (h1 : abcd ≥ 1000 ∧ abcd < 10000) (h2 : ∃ k : ℕ, k^2 = 4000000 + abcd) :
  abcd = 4001 ∨ abcd = 8004 :=
sorry

end NUMINAMATH_GPT_four_digit_number_perfect_square_l1506_150638


namespace NUMINAMATH_GPT_salt_mixture_l1506_150666

theorem salt_mixture (x y : ℝ) (p c z : ℝ) (hx : x = 50) (hp : p = 0.60) (hc : c = 0.40) (hy_eq : y = 50) :
  (50 * z) + (50 * 0.60) = 0.40 * (50 + 50) → (50 * z) + (50 * p) = c * (x + y) → y = 50 :=
by sorry

end NUMINAMATH_GPT_salt_mixture_l1506_150666


namespace NUMINAMATH_GPT_correct_total_cost_l1506_150675

noncomputable def total_cost_after_discount : ℝ :=
  let sandwich_cost := 4
  let soda_cost := 3
  let sandwich_count := 7
  let soda_count := 5
  let total_items := sandwich_count + soda_count
  let total_cost := sandwich_count * sandwich_cost + soda_count * soda_cost
  let discount := if total_items ≥ 10 then 0.1 * total_cost else 0
  total_cost - discount

theorem correct_total_cost :
  total_cost_after_discount = 38.7 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_correct_total_cost_l1506_150675


namespace NUMINAMATH_GPT_least_number_to_subtract_l1506_150658

theorem least_number_to_subtract (n : ℕ) (h : n = 652543) : 
  ∃ x : ℕ, x = 7 ∧ (n - x) % 12 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1506_150658


namespace NUMINAMATH_GPT_roberta_listen_days_l1506_150635

-- Define the initial number of records
def initial_records : ℕ := 8

-- Define the number of records received as gifts
def gift_records : ℕ := 12

-- Define the number of records bought
def bought_records : ℕ := 30

-- Define the number of days to listen to 1 record
def days_per_record : ℕ := 2

-- Define the total number of records
def total_records : ℕ := initial_records + gift_records + bought_records

-- Define the total number of days required to listen to all records
def total_days : ℕ := total_records * days_per_record

-- Theorem to prove the total days needed to listen to all records is 100
theorem roberta_listen_days : total_days = 100 := by
  sorry

end NUMINAMATH_GPT_roberta_listen_days_l1506_150635


namespace NUMINAMATH_GPT_find_integer_l1506_150677

theorem find_integer (n : ℕ) (h1 : 0 < n) (h2 : 200 % n = 2) (h3 : 398 % n = 2) : n = 6 :=
sorry

end NUMINAMATH_GPT_find_integer_l1506_150677


namespace NUMINAMATH_GPT_find_total_amount_before_brokerage_l1506_150621

noncomputable def total_amount_before_brokerage (realized_amount : ℝ) (brokerage_rate : ℝ) : ℝ :=
  realized_amount / (1 - brokerage_rate / 100)

theorem find_total_amount_before_brokerage :
  total_amount_before_brokerage 107.25 (1 / 4) = 107.25 * 400 / 399 := by
sorry

end NUMINAMATH_GPT_find_total_amount_before_brokerage_l1506_150621


namespace NUMINAMATH_GPT_devin_basketball_chances_l1506_150676

theorem devin_basketball_chances 
  (initial_chances : ℝ := 0.1) 
  (base_height : ℕ := 66) 
  (chance_increase_per_inch : ℝ := 0.1)
  (initial_height : ℕ := 65) 
  (growth : ℕ := 3) :
  initial_chances + (growth + initial_height - base_height) * chance_increase_per_inch = 0.3 := 
by 
  sorry

end NUMINAMATH_GPT_devin_basketball_chances_l1506_150676


namespace NUMINAMATH_GPT_jori_water_left_l1506_150614

theorem jori_water_left (initial used : ℚ) (h1 : initial = 3) (h2 : used = 4 / 3) :
  initial - used = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_jori_water_left_l1506_150614


namespace NUMINAMATH_GPT_number_equals_fifty_l1506_150665

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90

theorem number_equals_fifty (x : ℝ) (h : (5 / 4) * x = thirty_percent_less_than_ninety) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_equals_fifty_l1506_150665


namespace NUMINAMATH_GPT_solve_sum_of_squares_l1506_150651

theorem solve_sum_of_squares
  (k l m n a b c : ℕ)
  (h_cond1 : k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n)
  (h_cond2 : a * k^2 - b * k + c = 0)
  (h_cond3 : a * l^2 - b * l + c = 0)
  (h_cond4 : c * m^2 - 16 * b * m + 256 * a = 0)
  (h_cond5 : c * n^2 - 16 * b * n + 256 * a = 0) :
  k^2 + l^2 + m^2 + n^2 = 325 :=
by
  sorry

end NUMINAMATH_GPT_solve_sum_of_squares_l1506_150651


namespace NUMINAMATH_GPT_trapezoid_perimeter_l1506_150639

-- Define the problem conditions
variables (A B C D : Point) (BC AD : Line) (AB CD : Segment)

-- Conditions
def is_parallel (L1 L2 : Line) : Prop := sorry
def is_right_angle (A B C : Point) : Prop := sorry
def is_angle_150 (A B C : Point) : Prop := sorry

noncomputable def length (s : Segment) : ℝ := sorry

def trapezoid_conditions (A B C D : Point) (BC AD : Line) (AB CD : Segment) : Prop :=
  is_parallel BC AD ∧ is_angle_150 A B C ∧ is_right_angle C D B ∧
  length AB = 4 ∧ length BC = 3 - Real.sqrt 3

-- Perimeter calculation
noncomputable def perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) : ℝ :=
  length AB + length BC + length CD + length AD

-- Lean statement for the math proof problem
theorem trapezoid_perimeter (A B C D : Point) (BC AD : Line) (AB CD : Segment) :
  trapezoid_conditions A B C D BC AD AB CD → perimeter A B C D BC AD AB CD = 12 :=
sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l1506_150639


namespace NUMINAMATH_GPT_problem_statement_l1506_150664

variable (a b : ℝ)

-- Conditions
variable (h1 : a > 0) (h2 : b > 0) (h3 : ∃ x, x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a)))

-- The Lean theorem statement for the problem
theorem problem_statement : 
  ∀ x, (x = (1 / 2) * (Real.sqrt (a / b) - Real.sqrt (b / a))) →
  (2 * a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := 
sorry


end NUMINAMATH_GPT_problem_statement_l1506_150664


namespace NUMINAMATH_GPT_smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l1506_150683

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

end NUMINAMATH_GPT_smaller_angle_between_hands_at_3_40_pm_larger_angle_between_hands_at_3_40_pm_l1506_150683


namespace NUMINAMATH_GPT_find_last_four_digits_of_N_l1506_150630

def P (n : Nat) : Nat :=
  match n with
  | 0     => 1 -- usually not needed but for completeness
  | 1     => 2
  | _     => 2 + (n - 1) * n

theorem find_last_four_digits_of_N : (P 2011) % 10000 = 2112 := by
  -- we define P(2011) as per the general formula derived and then verify the modulo operation
  sorry

end NUMINAMATH_GPT_find_last_four_digits_of_N_l1506_150630


namespace NUMINAMATH_GPT_average_distance_is_600_l1506_150689

-- Definitions based on the conditions
def one_lap_distance : ℕ := 200
def johnny_lap_count : ℕ := 4
def mickey_lap_count : ℕ := johnny_lap_count / 2

-- Total distances run by Johnny and Mickey
def johnny_distance : ℕ := johnny_lap_count * one_lap_distance
def mickey_distance : ℕ := mickey_lap_count * one_lap_distance

-- Sum of distances
def total_distance : ℕ := johnny_distance + mickey_distance

-- Number of people
def number_of_people : ℕ := 2

-- Average distance calculation
def average_distance : ℕ := total_distance / number_of_people

-- The theorem to prove: the average distance run by Johnny and Mickey
theorem average_distance_is_600 : average_distance = 600 := by
  sorry

end NUMINAMATH_GPT_average_distance_is_600_l1506_150689


namespace NUMINAMATH_GPT_mark_has_24_dollars_l1506_150698

theorem mark_has_24_dollars
  (small_bag_cost : ℕ := 4)
  (small_bag_balloons : ℕ := 50)
  (medium_bag_cost : ℕ := 6)
  (medium_bag_balloons : ℕ := 75)
  (large_bag_cost : ℕ := 12)
  (large_bag_balloons : ℕ := 200)
  (total_balloons : ℕ := 400) :
  total_balloons / large_bag_balloons = 2 ∧ 2 * large_bag_cost = 24 := by
  sorry

end NUMINAMATH_GPT_mark_has_24_dollars_l1506_150698


namespace NUMINAMATH_GPT_focus_of_parabola_l1506_150647

theorem focus_of_parabola : (∃ p : ℝ × ℝ, p = (-1, 35/12)) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l1506_150647


namespace NUMINAMATH_GPT_eggs_per_hen_l1506_150657

theorem eggs_per_hen (total_eggs : Float) (num_hens : Float) (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) : 
  total_eggs / num_hens = 10.821428571428571 :=
by 
  sorry

end NUMINAMATH_GPT_eggs_per_hen_l1506_150657


namespace NUMINAMATH_GPT_compare_sqrt_differences_l1506_150613

theorem compare_sqrt_differences :
  let a := (Real.sqrt 7) - (Real.sqrt 6)
  let b := (Real.sqrt 3) - (Real.sqrt 2)
  a < b :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_compare_sqrt_differences_l1506_150613


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1506_150678

theorem geometric_sequence_a4 (a : ℕ → ℝ) (q : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * q)
    (h_a2 : a 2 = 1)
    (h_q : q = 2) : 
    a 4 = 4 :=
by
  -- Skip the proof as instructed
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1506_150678


namespace NUMINAMATH_GPT_part_I_part_II_l1506_150660

open Real

noncomputable def alpha₁ : Real := sorry -- Placeholder for the angle α in part I
noncomputable def alpha₂ : Real := sorry -- Placeholder for the angle α in part II

-- Given a point P(-4, 3) and a point on the terminal side of angle α₁ such that tan(α₁) = -3/4
theorem part_I :
  tan α₁ = - (3 / 4) → 
  (cos (π / 2 + α₁) * sin (-π - α₁)) / (cos (11 * π / 2 - α₁) * sin (9 * π / 2 + α₁)) = - (3 / 4) :=
by 
  intro h
  sorry

-- Given vector a = (3,1) and b = (sin α, cos α) where a is parallel to b such that tan(α₂) = 3
theorem part_II :
  tan α₂ = 3 → 
  (4 * sin α₂ - 2 * cos α₂) / (5 * cos α₂ + 3 * sin α₂) = 5 / 7 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1506_150660


namespace NUMINAMATH_GPT_smallest_side_for_table_rotation_l1506_150667

theorem smallest_side_for_table_rotation (S : ℕ) : (S ≥ Int.ofNat (Nat.sqrt (8^2 + 12^2) + 1)) → S = 15 := 
by
  sorry

end NUMINAMATH_GPT_smallest_side_for_table_rotation_l1506_150667


namespace NUMINAMATH_GPT_find_A2_A7_l1506_150643

theorem find_A2_A7 (A : ℕ → ℝ) (hA1A11 : A 11 - A 1 = 56)
  (hAiAi2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → A (i+2) - A i ≤ 12)
  (hAjAj3 : ∀ j, 1 ≤ j ∧ j ≤ 8 → A (j+3) - A j ≥ 17) : 
  A 7 - A 2 = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_A2_A7_l1506_150643


namespace NUMINAMATH_GPT_bunchkin_total_distance_l1506_150687

theorem bunchkin_total_distance
  (a b c d e : ℕ)
  (ha : a = 17)
  (hb : b = 43)
  (hc : c = 56)
  (hd : d = 66)
  (he : e = 76) :
  (a + b + c + d + e) / 2 = 129 :=
by
  sorry

end NUMINAMATH_GPT_bunchkin_total_distance_l1506_150687


namespace NUMINAMATH_GPT_Brenda_weight_correct_l1506_150645

-- Conditions
def MelWeight : ℕ := 70
def BrendaWeight : ℕ := 3 * MelWeight + 10

-- Proof problem
theorem Brenda_weight_correct : BrendaWeight = 220 := by
  sorry

end NUMINAMATH_GPT_Brenda_weight_correct_l1506_150645


namespace NUMINAMATH_GPT_average_cost_per_individual_before_gratuity_l1506_150610

theorem average_cost_per_individual_before_gratuity
  (total_bill : ℝ)
  (num_people : ℕ)
  (gratuity_percentage : ℝ)
  (bill_including_gratuity : total_bill = 840)
  (group_size : num_people = 7)
  (gratuity : gratuity_percentage = 0.20) :
  (total_bill / (1 + gratuity_percentage)) / num_people = 100 :=
by
  sorry

end NUMINAMATH_GPT_average_cost_per_individual_before_gratuity_l1506_150610
