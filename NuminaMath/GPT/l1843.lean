import Mathlib

namespace NUMINAMATH_GPT_field_day_difference_l1843_184370

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end NUMINAMATH_GPT_field_day_difference_l1843_184370


namespace NUMINAMATH_GPT_exists_divisor_c_of_f_l1843_184334

theorem exists_divisor_c_of_f (f : ℕ → ℕ) 
  (h₁ : ∀ n, f n ≥ 2)
  (h₂ : ∀ m n, f (m + n) ∣ (f m + f n)) :
  ∃ c > 1, ∀ n, c ∣ f n :=
sorry

end NUMINAMATH_GPT_exists_divisor_c_of_f_l1843_184334


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1843_184393

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- The statement to prove: The equation of the asymptotes of the hyperbola is as follows
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x)) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1843_184393


namespace NUMINAMATH_GPT_expression_value_l1843_184302

def a : ℕ := 1000
def b1 : ℕ := 15
def b2 : ℕ := 314
def c1 : ℕ := 201
def c2 : ℕ := 360
def c3 : ℕ := 110
def d1 : ℕ := 201
def d2 : ℕ := 360
def d3 : ℕ := 110
def e1 : ℕ := 15
def e2 : ℕ := 314

theorem expression_value :
  (a + b1 + b2) * (c1 + c2 + c3) + (a - d1 - d2 - d3) * (e1 + e2) = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1843_184302


namespace NUMINAMATH_GPT_book_arrangement_ways_l1843_184306

open Nat

theorem book_arrangement_ways : 
  let m := 4  -- Number of math books
  let h := 6  -- Number of history books
  -- Number of ways to place a math book on both ends:
  let ways_ends := m * (m - 1)  -- Choices for the left end and right end
  -- Number of ways to arrange the remaining books:
  let ways_entities := 2!  -- Arrangements of the remaining entities
  -- Number of ways to arrange history books within the block:
  let arrange_history := factorial h
  -- Total arrangements
  let total_ways := ways_ends * ways_entities * arrange_history
  total_ways = 17280 := sorry

end NUMINAMATH_GPT_book_arrangement_ways_l1843_184306


namespace NUMINAMATH_GPT_price_of_other_frisbees_l1843_184398

theorem price_of_other_frisbees 
  (P : ℝ) 
  (x : ℝ)
  (h1 : x + (64 - x) = 64)
  (h2 : P * x + 4 * (64 - x) = 196)
  (h3 : 64 - x ≥ 4) 
  : P = 3 :=
sorry

end NUMINAMATH_GPT_price_of_other_frisbees_l1843_184398


namespace NUMINAMATH_GPT_choir_blonde_black_ratio_l1843_184385

theorem choir_blonde_black_ratio 
  (b x : ℕ) 
  (h1 : ∀ (b x : ℕ), b / ((5 / 3 : ℚ) * b) = (3 / 5 : ℚ)) 
  (h2 : ∀ (b x : ℕ), (b + x) / ((5 / 3 : ℚ) * b) = (3 / 2 : ℚ)) :
  x = (3 / 2 : ℚ) * b ∧ 
  ∃ k : ℚ, k = (5 / 3 : ℚ) * b :=
by {
  sorry
}

end NUMINAMATH_GPT_choir_blonde_black_ratio_l1843_184385


namespace NUMINAMATH_GPT_exists_monotonicity_b_range_l1843_184361

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - 2 * a * x + Real.log x

theorem exists_monotonicity_b_range :
  ∀ (a : ℝ) (b : ℝ), 1 < a ∧ a < 2 →
  (∀ (x0 : ℝ), x0 ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2 →
   f a x0 + Real.log (a + 1) > b * (a^2 - 1) - (a + 1) + 2 * Real.log 2) →
   b ∈ Set.Iic (-1/4) :=
sorry

end NUMINAMATH_GPT_exists_monotonicity_b_range_l1843_184361


namespace NUMINAMATH_GPT_height_of_rectangular_block_l1843_184311

variable (V A h : ℕ)

theorem height_of_rectangular_block :
  V = 120 ∧ A = 24 ∧ V = A * h → h = 5 :=
by
  sorry

end NUMINAMATH_GPT_height_of_rectangular_block_l1843_184311


namespace NUMINAMATH_GPT_infinite_positive_sequence_geometric_l1843_184327

theorem infinite_positive_sequence_geometric {a : ℕ → ℝ} (h : ∀ n ≥ 1, a (n + 2) = a n - a (n + 1)) 
  (h_pos : ∀ n, a n > 0) :
  ∃ (a1 : ℝ) (q : ℝ), q = (Real.sqrt 5 - 1) / 2 ∧ (∀ n, a n = a1 * q^(n - 1)) := by
  sorry

end NUMINAMATH_GPT_infinite_positive_sequence_geometric_l1843_184327


namespace NUMINAMATH_GPT_girls_at_start_l1843_184378

theorem girls_at_start (B G : ℕ) (h1 : B + G = 600) (h2 : 6 * B + 7 * G = 3840) : G = 240 :=
by
  -- actual proof is omitted
  sorry

end NUMINAMATH_GPT_girls_at_start_l1843_184378


namespace NUMINAMATH_GPT_ratio_of_A_to_B_l1843_184396

theorem ratio_of_A_to_B (A B C : ℝ) (h1 : A + B + C = 544) (h2 : B = (1/4) * C) (hA : A = 64) (hB : B = 96) (hC : C = 384) : A / B = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_A_to_B_l1843_184396


namespace NUMINAMATH_GPT_intersection_M_N_l1843_184307

def M := {m : ℤ | -3 < m ∧ m < 2}
def N := {x : ℤ | x * (x - 1) = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := sorry

end NUMINAMATH_GPT_intersection_M_N_l1843_184307


namespace NUMINAMATH_GPT_max_numbers_with_240_product_square_l1843_184376

theorem max_numbers_with_240_product_square :
  ∃ (S : Finset ℕ), S.card = 11 ∧ ∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ n m, 240 * k = (n * m) ^ 2 :=
sorry

end NUMINAMATH_GPT_max_numbers_with_240_product_square_l1843_184376


namespace NUMINAMATH_GPT_f_at_3_l1843_184373

-- Define the function f and its conditions
variable (f : ℝ → ℝ)

-- The domain of the function f is ℝ, hence f : ℝ → ℝ
-- Also given:
axiom f_symm : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom f_add : f (-1) + f (3) = 12

-- Final proof statement
theorem f_at_3 : f 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_f_at_3_l1843_184373


namespace NUMINAMATH_GPT_monomial_exponent_match_l1843_184303

theorem monomial_exponent_match (m : ℤ) (x y : ℂ) : (-x^(2*m) * y^3 = 2 * x^6 * y^3) → m = 3 := 
by 
  sorry

end NUMINAMATH_GPT_monomial_exponent_match_l1843_184303


namespace NUMINAMATH_GPT_fifth_equation_in_pattern_l1843_184342

theorem fifth_equation_in_pattern :
  (1 - 4 + 9 - 16 + 25) = (1 + 2 + 3 + 4 + 5) :=
sorry

end NUMINAMATH_GPT_fifth_equation_in_pattern_l1843_184342


namespace NUMINAMATH_GPT_translate_line_up_l1843_184375

theorem translate_line_up (x y : ℝ) (h : y = 2 * x - 3) : y + 6 = 2 * x + 3 :=
by sorry

end NUMINAMATH_GPT_translate_line_up_l1843_184375


namespace NUMINAMATH_GPT_triangle_problem_l1843_184372

-- Define a triangle with given parameters and properties
variables {A B C : ℝ}
variables {a b c : ℝ} (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) 
variables (h_b2a : b = 2 * a)
variables (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3)

-- Prove the required angles and side length
theorem triangle_problem 
    (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
    (h_b2a : b = 2 * a)
    (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) :

    Real.cos C = -1/2 ∧ C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_problem_l1843_184372


namespace NUMINAMATH_GPT_segment_length_calc_l1843_184374

noncomputable def segment_length_parallel_to_side
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : ℝ :=
  a * (b + c) / (a + b + c)

theorem segment_length_calc
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  segment_length_parallel_to_side a b c a_pos b_pos c_pos = a * (b + c) / (a + b + c) :=
sorry

end NUMINAMATH_GPT_segment_length_calc_l1843_184374


namespace NUMINAMATH_GPT_calculation_correct_l1843_184380

theorem calculation_correct : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1843_184380


namespace NUMINAMATH_GPT_balance_difference_l1843_184383

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  (round diff = 3553) :=
by 
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  have h : round diff = 3553 := sorry
  assumption

end NUMINAMATH_GPT_balance_difference_l1843_184383


namespace NUMINAMATH_GPT_problem_statement_l1843_184335

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2002 + a ^ 2001 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1843_184335


namespace NUMINAMATH_GPT_calculate_y_when_x_is_neg2_l1843_184387

def conditional_program (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 3
  else if x > 0 then
    -2 * x + 5
  else
    0

theorem calculate_y_when_x_is_neg2 : conditional_program (-2) = -1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_y_when_x_is_neg2_l1843_184387


namespace NUMINAMATH_GPT_toby_candies_left_l1843_184300

def total_candies : ℕ := 56 + 132 + 8 + 300
def num_cousins : ℕ := 13

theorem toby_candies_left : total_candies % num_cousins = 2 :=
by sorry

end NUMINAMATH_GPT_toby_candies_left_l1843_184300


namespace NUMINAMATH_GPT_train_speed_late_l1843_184312

theorem train_speed_late (v : ℝ) 
  (h1 : ∀ (d : ℝ) (s : ℝ), d = 15 ∧ s = 100 → d / s = 0.15) 
  (h2 : ∀ (t1 t2 : ℝ), t1 = 0.15 ∧ t2 = 0.4 → t2 = t1 + 0.25)
  (h3 : ∀ (d : ℝ) (t : ℝ), d = 15 ∧ t = 0.4 → v = d / t) : 
  v = 37.5 := sorry

end NUMINAMATH_GPT_train_speed_late_l1843_184312


namespace NUMINAMATH_GPT_solution_set_inequality_l1843_184367

noncomputable def f (x : ℝ) := Real.exp (2 * x) - 1
noncomputable def g (x : ℝ) := Real.log (x + 1)

theorem solution_set_inequality :
  {x : ℝ | f (g x) - g (f x) ≤ 1} = Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1843_184367


namespace NUMINAMATH_GPT_katya_age_l1843_184348

theorem katya_age (A K V : ℕ) (h1 : A + K = 19) (h2 : A + V = 14) (h3 : K + V = 7) : K = 6 := by
  sorry

end NUMINAMATH_GPT_katya_age_l1843_184348


namespace NUMINAMATH_GPT_bird_problem_l1843_184320

theorem bird_problem (B : ℕ) (h : (2 / 15) * B = 60) : B = 450 ∧ (2 / 15) * B = 60 :=
by
  sorry

end NUMINAMATH_GPT_bird_problem_l1843_184320


namespace NUMINAMATH_GPT_fraction_sent_afternoon_l1843_184391

theorem fraction_sent_afternoon :
  ∀ (total_fliers morning_fraction fliers_left_next_day : ℕ),
  total_fliers = 3000 →
  morning_fraction = 1/5 →
  fliers_left_next_day = 1800 →
  ((total_fliers - total_fliers * morning_fraction) - fliers_left_next_day) / (total_fliers - total_fliers * morning_fraction) = 1/4 :=
by
  intros total_fliers morning_fraction fliers_left_next_day h1 h2 h3
  sorry

end NUMINAMATH_GPT_fraction_sent_afternoon_l1843_184391


namespace NUMINAMATH_GPT_base_conversion_subtraction_l1843_184310

def base8_to_base10 : Nat := 5 * 8^5 + 4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0
def base9_to_base10 : Nat := 6 * 9^4 + 5 * 9^3 + 4 * 9^2 + 3 * 9^1 + 2 * 9^0

theorem base_conversion_subtraction :
  base8_to_base10 - base9_to_base10 = 136532 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l1843_184310


namespace NUMINAMATH_GPT_usual_time_to_reach_school_l1843_184390

theorem usual_time_to_reach_school
  (R T : ℝ)
  (h1 : (7 / 6) * R = R / (T - 3) * T) : T = 21 :=
sorry

end NUMINAMATH_GPT_usual_time_to_reach_school_l1843_184390


namespace NUMINAMATH_GPT_constant_c_square_of_binomial_l1843_184352

theorem constant_c_square_of_binomial (c : ℝ) (h : ∃ d : ℝ, (3*x + d)^2 = 9*x^2 - 18*x + c) : c = 9 :=
sorry

end NUMINAMATH_GPT_constant_c_square_of_binomial_l1843_184352


namespace NUMINAMATH_GPT_find_k_l1843_184355

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end NUMINAMATH_GPT_find_k_l1843_184355


namespace NUMINAMATH_GPT_hypotenuse_length_l1843_184316

-- Definitions and conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Hypotheses
def leg1 := 8
def leg2 := 15

-- The theorem to be proven
theorem hypotenuse_length : ∃ c : ℕ, is_right_triangle leg1 leg2 c ∧ c = 17 :=
by { sorry }

end NUMINAMATH_GPT_hypotenuse_length_l1843_184316


namespace NUMINAMATH_GPT_junior_score_proof_l1843_184360

noncomputable def class_total_score (total_students : ℕ) (average_class_score : ℕ) : ℕ :=
total_students * average_class_score

noncomputable def number_of_juniors (total_students : ℕ) (percent_juniors : ℕ) : ℕ :=
percent_juniors * total_students / 100

noncomputable def number_of_seniors (total_students juniors : ℕ) : ℕ :=
total_students - juniors

noncomputable def total_senior_score (seniors average_senior_score : ℕ) : ℕ :=
seniors * average_senior_score

noncomputable def total_junior_score (total_score senior_score : ℕ) : ℕ :=
total_score - senior_score

noncomputable def junior_score (junior_total_score juniors : ℕ) : ℕ :=
junior_total_score / juniors

theorem junior_score_proof :
  ∀ (total_students: ℕ) (percent_juniors average_class_score average_senior_score : ℕ),
  total_students = 20 →
  percent_juniors = 15 →
  average_class_score = 85 →
  average_senior_score = 84 →
  (junior_score (total_junior_score (class_total_score total_students average_class_score)
                                    (total_senior_score (number_of_seniors total_students (number_of_juniors total_students percent_juniors))
                                                        average_senior_score))
                (number_of_juniors total_students percent_juniors)) = 91 :=
by
  intros
  sorry

end NUMINAMATH_GPT_junior_score_proof_l1843_184360


namespace NUMINAMATH_GPT_jasmine_first_exceed_500_l1843_184349

theorem jasmine_first_exceed_500 {k : ℕ} (initial : ℕ) (factor : ℕ) :
  initial = 5 → factor = 4 → (5 * 4^k > 500) → k = 4 :=
by
  sorry

end NUMINAMATH_GPT_jasmine_first_exceed_500_l1843_184349


namespace NUMINAMATH_GPT_initial_deadline_is_75_days_l1843_184333

-- Define constants for the problem
def initial_men : ℕ := 100
def initial_hours_per_day : ℕ := 8
def days_worked_initial : ℕ := 25
def fraction_work_completed : ℚ := 1 / 3
def additional_men : ℕ := 60
def new_hours_per_day : ℕ := 10
def total_man_hours : ℕ := 60000

-- Prove that the initial deadline for the project is 75 days
theorem initial_deadline_is_75_days : 
  ∃ (D : ℕ), (D * initial_men * initial_hours_per_day = total_man_hours) ∧ D = 75 := 
by {
  sorry
}

end NUMINAMATH_GPT_initial_deadline_is_75_days_l1843_184333


namespace NUMINAMATH_GPT_sum_of_smallest_x_and_y_l1843_184362

theorem sum_of_smallest_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (hx : ∃ k : ℕ, (480 * x) = k * k ∧ ∀ z : ℕ, 0 < z → (480 * z) = k * k → x ≤ z)
  (hy : ∃ n : ℕ, (480 * y) = n * n * n ∧ ∀ z : ℕ, 0 < z → (480 * z) = n * n * n → y ≤ z) :
  x + y = 480 := sorry

end NUMINAMATH_GPT_sum_of_smallest_x_and_y_l1843_184362


namespace NUMINAMATH_GPT_paco_salty_cookies_left_l1843_184350

theorem paco_salty_cookies_left (S₁ S₂ : ℕ) (h₁ : S₁ = 6) (e1_eaten : ℕ) (a₁ : e1_eaten = 3)
(h₂ : S₂ = 24) (r1_ratio : ℚ) (a_ratio : r1_ratio = (2/3)) :
  S₁ - e1_eaten + r1_ratio * S₂ = 19 :=
by
  sorry

end NUMINAMATH_GPT_paco_salty_cookies_left_l1843_184350


namespace NUMINAMATH_GPT_geometric_sequence_min_value_l1843_184351

theorem geometric_sequence_min_value 
  (a b c : ℝ)
  (h1 : b^2 = ac)
  (h2 : b = -Real.exp 1) :
  ac = Real.exp 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_min_value_l1843_184351


namespace NUMINAMATH_GPT_problems_per_page_l1843_184304

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def remaining_pages : ℕ := 5
def remaining_problems : ℕ := total_problems - finished_problems

theorem problems_per_page : remaining_problems / remaining_pages = 8 := 
by
  sorry

end NUMINAMATH_GPT_problems_per_page_l1843_184304


namespace NUMINAMATH_GPT_gcd_of_B_is_2_l1843_184330

-- Condition: B is the set of all numbers which can be represented as the sum of four consecutive positive integers
def B := { n : ℕ | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) }

-- Question: What is the greatest common divisor of all numbers in \( B \)
-- Mathematical equivalent proof problem: Prove gcd of all elements in set \( B \) is 2

theorem gcd_of_B_is_2 : ∀ n ∈ B, ∃ y : ℕ, n = 2 * (2 * y + 1) → ∀ m ∈ B, n.gcd m = 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_B_is_2_l1843_184330


namespace NUMINAMATH_GPT_ivan_chess_false_l1843_184382

theorem ivan_chess_false (n : ℕ) :
  ∃ n, n + 3 * n + 6 * n = 64 → False :=
by
  use 6
  sorry

end NUMINAMATH_GPT_ivan_chess_false_l1843_184382


namespace NUMINAMATH_GPT_nonnegative_solution_positive_solution_l1843_184345

/-- For k > 7, there exist non-negative integers x and y such that 5*x + 3*y = k. -/
theorem nonnegative_solution (k : ℤ) (hk : k > 7) : ∃ x y : ℕ, 5 * x + 3 * y = k :=
sorry

/-- For k > 15, there exist positive integers x and y such that 5*x + 3*y = k. -/
theorem positive_solution (k : ℤ) (hk : k > 15) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y = k :=
sorry

end NUMINAMATH_GPT_nonnegative_solution_positive_solution_l1843_184345


namespace NUMINAMATH_GPT_grace_dimes_count_l1843_184319

-- Defining the conditions
def dimes_to_pennies (d : ℕ) : ℕ := 10 * d
def nickels_to_pennies : ℕ := 10 * 5
def total_pennies (d : ℕ) : ℕ := dimes_to_pennies d + nickels_to_pennies

-- The statement of the theorem
theorem grace_dimes_count (d : ℕ) (h : total_pennies d = 150) : d = 10 := 
sorry

end NUMINAMATH_GPT_grace_dimes_count_l1843_184319


namespace NUMINAMATH_GPT_find_coefficient_b_l1843_184359

noncomputable def polynomial_f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_coefficient_b 
  (a b c d : ℝ)
  (h1 : polynomial_f a b c d (-2) = 0)
  (h2 : polynomial_f a b c d 0 = 0)
  (h3 : polynomial_f a b c d 2 = 0)
  (h4 : polynomial_f a b c d (-1) = 3) :
  b = 0 :=
sorry

end NUMINAMATH_GPT_find_coefficient_b_l1843_184359


namespace NUMINAMATH_GPT_farmer_kent_income_l1843_184331

-- Define the constants and conditions
def watermelon_weight : ℕ := 23
def price_per_pound : ℕ := 2
def number_of_watermelons : ℕ := 18

-- Construct the proof statement
theorem farmer_kent_income : 
  price_per_pound * watermelon_weight * number_of_watermelons = 828 := 
by
  -- Skipping the proof here, just stating the theorem.
  sorry

end NUMINAMATH_GPT_farmer_kent_income_l1843_184331


namespace NUMINAMATH_GPT_molecular_weight_AlOH3_l1843_184371

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_AlOH3 :
  (atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H) = 78.01 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_AlOH3_l1843_184371


namespace NUMINAMATH_GPT_problem_l1843_184377

def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x^7 + b * x^5 - c * x^3 + d * x + 3

theorem problem (a b c d : ℝ) (h : f 92 a b c d = 2) : f 92 a b c d + f (-92) a b c d = 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1843_184377


namespace NUMINAMATH_GPT_domain_of_k_l1843_184313

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 6)) + (1 / (x^2 + 2*x + 9)) + (1 / (x^3 - 27))

theorem domain_of_k : {x : ℝ | k x ≠ 0} = {x : ℝ | x ≠ -6 ∧ x ≠ 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_k_l1843_184313


namespace NUMINAMATH_GPT_rosy_current_age_l1843_184301

theorem rosy_current_age 
  (R : ℕ) 
  (h1 : ∀ (david_age rosy_age : ℕ), david_age = rosy_age + 12) 
  (h2 : ∀ (david_age_plus_4 rosy_age_plus_4 : ℕ), david_age_plus_4 = 2 * rosy_age_plus_4) : 
  R = 8 := 
sorry

end NUMINAMATH_GPT_rosy_current_age_l1843_184301


namespace NUMINAMATH_GPT_log_comparison_l1843_184314

/-- Assuming a = log base 3 of 2, b = natural log of 3, and c = log base 2 of 3,
    prove that c > b > a. -/
theorem log_comparison (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3)
                                (h2 : b = Real.log 3)
                                (h3 : c = Real.log 3 / Real.log 2) :
  c > b ∧ b > a :=
by {
  sorry
}

end NUMINAMATH_GPT_log_comparison_l1843_184314


namespace NUMINAMATH_GPT_max_value_of_f_l1843_184323

noncomputable def f (x : ℝ) : ℝ :=
  (2 * x + 1) / (4 * x ^ 2 + 1)

theorem max_value_of_f : ∃ (M : ℝ), ∀ (x : ℝ), x > 0 → f x ≤ M ∧ M = (Real.sqrt 2 + 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1843_184323


namespace NUMINAMATH_GPT_sock_ratio_l1843_184347

theorem sock_ratio (b : ℕ) (x : ℕ) (hx_pos : 0 < x)
  (h1 : 5 * x + 3 * b * x = k) -- Original cost is 5x + 3bx
  (h2 : b * x + 15 * x = 2 * k) -- Interchanged cost is doubled
  : b = 1 :=
by sorry

end NUMINAMATH_GPT_sock_ratio_l1843_184347


namespace NUMINAMATH_GPT_value_of_f_at_2_l1843_184392

-- Given the conditions
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)
variable (h_cond : ∀ x : ℝ, f (f x - 3^x) = 4)

-- Define the proof goal
theorem value_of_f_at_2 : f 2 = 10 := 
sorry

end NUMINAMATH_GPT_value_of_f_at_2_l1843_184392


namespace NUMINAMATH_GPT_greatest_of_3_consecutive_integers_l1843_184363

theorem greatest_of_3_consecutive_integers (x : ℤ) (h : x + (x + 1) + (x + 2) = 24) : (x + 2) = 9 :=
by
-- Proof would go here.
sorry

end NUMINAMATH_GPT_greatest_of_3_consecutive_integers_l1843_184363


namespace NUMINAMATH_GPT_esteban_exercise_days_l1843_184395

theorem esteban_exercise_days
  (natasha_exercise_per_day : ℕ)
  (natasha_days : ℕ)
  (esteban_exercise_per_day : ℕ)
  (total_exercise_hours : ℕ)
  (hours_to_minutes : ℕ)
  (natasha_exercise_total : ℕ)
  (total_exercise_minutes : ℕ)
  (esteban_exercise_total : ℕ)
  (esteban_days : ℕ) :
  natasha_exercise_per_day = 30 →
  natasha_days = 7 →
  esteban_exercise_per_day = 10 →
  total_exercise_hours = 5 →
  hours_to_minutes = 60 →
  natasha_exercise_total = natasha_exercise_per_day * natasha_days →
  total_exercise_minutes = total_exercise_hours * hours_to_minutes →
  esteban_exercise_total = total_exercise_minutes - natasha_exercise_total →
  esteban_days = esteban_exercise_total / esteban_exercise_per_day →
  esteban_days = 9 :=
by
  sorry

end NUMINAMATH_GPT_esteban_exercise_days_l1843_184395


namespace NUMINAMATH_GPT_rectangle_decomposition_l1843_184388

theorem rectangle_decomposition (m n k : ℕ) : ((k ∣ m) ∨ (k ∣ n)) ↔ (∃ P : ℕ, m * n = P * k) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_decomposition_l1843_184388


namespace NUMINAMATH_GPT_find_David_marks_in_Physics_l1843_184346

theorem find_David_marks_in_Physics
  (english_marks : ℕ) (math_marks : ℕ) (chem_marks : ℕ) (biology_marks : ℕ)
  (avg_marks : ℕ) (num_subjects : ℕ)
  (h_english : english_marks = 76)
  (h_math : math_marks = 65)
  (h_chem : chem_marks = 67)
  (h_bio : biology_marks = 85)
  (h_avg : avg_marks = 75) 
  (h_num_subjects : num_subjects = 5) :
  english_marks + math_marks + chem_marks + biology_marks + physics_marks = avg_marks * num_subjects → physics_marks = 82 := 
  sorry

end NUMINAMATH_GPT_find_David_marks_in_Physics_l1843_184346


namespace NUMINAMATH_GPT_surface_area_of_cube_l1843_184368

-- Definition of the problem in Lean 4
theorem surface_area_of_cube (a : ℝ) (s : ℝ) (h : s * Real.sqrt 3 = a) : 6 * (s^2) = 2 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_l1843_184368


namespace NUMINAMATH_GPT_smallest_k_for_perfect_cube_l1843_184386

noncomputable def isPerfectCube (m : ℕ) : Prop :=
  ∃ n : ℤ, n^3 = m

theorem smallest_k_for_perfect_cube :
  ∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, ((2^4) * (3^2) * (5^5) * k = m) → isPerfectCube m) ∧ k = 60 :=
sorry

end NUMINAMATH_GPT_smallest_k_for_perfect_cube_l1843_184386


namespace NUMINAMATH_GPT_find_g2_l1843_184343

variable (g : ℝ → ℝ)

def condition (x : ℝ) : Prop :=
  g x - 2 * g (1 / x) = 3^x

theorem find_g2 (h : ∀ x ≠ 0, condition g x) : g 2 = -3 - (4 * Real.sqrt 3) / 9 :=
  sorry

end NUMINAMATH_GPT_find_g2_l1843_184343


namespace NUMINAMATH_GPT_money_r_gets_l1843_184353

def total_amount : ℕ := 1210
def p_to_q := 5 / 4
def q_to_r := 9 / 10

theorem money_r_gets :
  let P := (total_amount * 45) / 121
  let Q := (total_amount * 36) / 121
  let R := (total_amount * 40) / 121
  R = 400 := by
  sorry

end NUMINAMATH_GPT_money_r_gets_l1843_184353


namespace NUMINAMATH_GPT_sequence_inequality_l1843_184305

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (h_non_decreasing : ∀ i j : ℕ, i ≤ j → a i ≤ a j)
  (h_range : ∀ i, 1 ≤ i ∧ i ≤ 10 → a i = a (i - 1)) :
  (1 / 6) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) ≤ (1 / 10) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) :=
by
  sorry

end NUMINAMATH_GPT_sequence_inequality_l1843_184305


namespace NUMINAMATH_GPT_Jeremy_strolled_20_kilometers_l1843_184344

def speed : ℕ := 2 -- Jeremy's speed in kilometers per hour
def time : ℕ := 10 -- Time Jeremy strolled in hours

noncomputable def distance : ℕ := speed * time -- The computed distance

theorem Jeremy_strolled_20_kilometers : distance = 20 := by
  sorry

end NUMINAMATH_GPT_Jeremy_strolled_20_kilometers_l1843_184344


namespace NUMINAMATH_GPT_max_a_2017_2018_ge_2017_l1843_184324

def seq_a (a : ℕ → ℤ) (b : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ (∀ n, n ≥ 1 → 
  (b (n-1) = 1 → a (n+1) = a n * b n + a (n-1)) ∧ 
  (b (n-1) > 1 → a (n+1) = a n * b n - a (n-1)))

theorem max_a_2017_2018_ge_2017 (a : ℕ → ℤ) (b : ℕ → ℕ) (h : seq_a a b) :
  max (a 2017) (a 2018) ≥ 2017 :=
sorry

end NUMINAMATH_GPT_max_a_2017_2018_ge_2017_l1843_184324


namespace NUMINAMATH_GPT_value_of_expression_l1843_184358

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1843_184358


namespace NUMINAMATH_GPT_linear_function_quadrants_l1843_184329

theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) : 
  (∀ x : ℝ, (k < 0 ∧ b > 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) ∧ 
  (∀ x : ℝ, (k > 0 ∧ b < 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) :=
sorry

end NUMINAMATH_GPT_linear_function_quadrants_l1843_184329


namespace NUMINAMATH_GPT_compound_interest_difference_l1843_184364

variable (P r : ℝ)

theorem compound_interest_difference :
  (P * 9 * r^2 = 360) → (P * r^2 = 40) :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_difference_l1843_184364


namespace NUMINAMATH_GPT_triangle_base_is_8_l1843_184308

/- Problem Statement:
We have a square with a perimeter of 48 and a triangle with a height of 36.
We need to prove that if both the square and the triangle have the same area, then the base of the triangle (x) is 8.
-/

theorem triangle_base_is_8
  (square_perimeter : ℝ)
  (triangle_height : ℝ)
  (same_area : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  same_area = (square_perimeter / 4) ^ 2 →
  same_area = (1 / 2) * x * triangle_height →
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_is_8_l1843_184308


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1843_184366

/-- 
In a geometric sequence of real numbers, the sum of the first 2 terms is 15,
and the sum of the first 6 terms is 195. Prove that the sum of the first 4 terms is 82.
-/
theorem geometric_sequence_sum :
  ∃ (a r : ℝ), (a + a * r = 15) ∧ (a * (1 - r^6) / (1 - r) = 195) ∧ (a * (1 + r + r^2 + r^3) = 82) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1843_184366


namespace NUMINAMATH_GPT_tan_theta_3_l1843_184340

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_3_l1843_184340


namespace NUMINAMATH_GPT_find_x_angle_l1843_184336

theorem find_x_angle (ABC ACB CDE : ℝ) (h1 : ABC = 70) (h2 : ACB = 90) (h3 : CDE = 42) : 
  ∃ x : ℝ, x = 158 :=
by
  sorry

end NUMINAMATH_GPT_find_x_angle_l1843_184336


namespace NUMINAMATH_GPT_correct_subtraction_l1843_184325

theorem correct_subtraction (x : ℕ) (h : x - 63 = 8) : x - 36 = 35 :=
by sorry

end NUMINAMATH_GPT_correct_subtraction_l1843_184325


namespace NUMINAMATH_GPT_percent_increase_visual_range_l1843_184397

theorem percent_increase_visual_range (original new : ℝ) (h_original : original = 60) (h_new : new = 150) : 
  ((new - original) / original) * 100 = 150 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_visual_range_l1843_184397


namespace NUMINAMATH_GPT_Bill_trips_l1843_184384

theorem Bill_trips (total_trips : ℕ) (Jean_trips : ℕ) (Bill_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : Jean_trips = 23) 
  (h3 : Bill_trips + Jean_trips = total_trips) : 
  Bill_trips = 17 := 
by
  sorry

end NUMINAMATH_GPT_Bill_trips_l1843_184384


namespace NUMINAMATH_GPT_second_bag_roger_is_3_l1843_184357

def total_candy_sandra := 2 * 6
def total_candy_roger := total_candy_sandra + 2
def first_bag_roger := 11
def second_bag_roger := total_candy_roger - first_bag_roger

theorem second_bag_roger_is_3 : second_bag_roger = 3 :=
by
  sorry

end NUMINAMATH_GPT_second_bag_roger_is_3_l1843_184357


namespace NUMINAMATH_GPT_sum_of_roots_of_cubic_l1843_184381

noncomputable def P (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_cubic (a b c d : ℝ) (h : ∀ x : ℝ, P a b c d (x^2 + x) ≥ P a b c d (x + 1)) :
  (-b / a) = (P a b c d 0) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_cubic_l1843_184381


namespace NUMINAMATH_GPT_smallest_bob_number_l1843_184399

theorem smallest_bob_number (b : ℕ) (h : ∀ p : ℕ, Prime p → p ∣ 30 → p ∣ b) : 30 ≤ b :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_bob_number_l1843_184399


namespace NUMINAMATH_GPT_asparagus_spears_needed_l1843_184328

def BridgetteGuests : Nat := 84
def AlexGuests : Nat := (2 * BridgetteGuests) / 3
def TotalGuests : Nat := BridgetteGuests + AlexGuests
def ExtraPlates : Nat := 10
def TotalPlates : Nat := TotalGuests + ExtraPlates
def VegetarianPercent : Nat := 20
def LargePortionPercent : Nat := 10
def VegetarianMeals : Nat := (VegetarianPercent * TotalGuests) / 100
def LargePortionMeals : Nat := (LargePortionPercent * TotalGuests) / 100
def RegularMeals : Nat := TotalGuests - (VegetarianMeals + LargePortionMeals)
def AsparagusPerRegularMeal : Nat := 8
def AsparagusPerVegetarianMeal : Nat := 6
def AsparagusPerLargePortionMeal : Nat := 12

theorem asparagus_spears_needed : 
  RegularMeals * AsparagusPerRegularMeal + 
  VegetarianMeals * AsparagusPerVegetarianMeal + 
  LargePortionMeals * AsparagusPerLargePortionMeal = 1120 := by
  sorry

end NUMINAMATH_GPT_asparagus_spears_needed_l1843_184328


namespace NUMINAMATH_GPT_ellipse_equation_max_area_abcd_l1843_184394

open Real

theorem ellipse_equation (x y : ℝ) (a b c : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (x^2 / 2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem max_area_abcd (a b c t : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (∀ (t : ℝ), 4 * sqrt 2 * sqrt (1 + t^2) / (t^2 + 2) ≤ 2 * sqrt 2) := by
  sorry

end NUMINAMATH_GPT_ellipse_equation_max_area_abcd_l1843_184394


namespace NUMINAMATH_GPT_age_relation_l1843_184341

theorem age_relation (S M D Y : ℝ)
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2))
  (h3 : D = S - 4)
  (h4 : M + Y = 3 * (D + Y))
  : Y = -10.5 :=
by
  sorry

end NUMINAMATH_GPT_age_relation_l1843_184341


namespace NUMINAMATH_GPT_mail_in_six_months_l1843_184321

/-- The post office receives 60 letters and 20 packages per day. Each month has 30 days. -/
def daily_letters := 60
def daily_packages := 20
def days_per_month := 30
def months := 6

/-- Prove that the post office handles 14400 pieces of mail in six months. -/
theorem mail_in_six_months : (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  sorry

end NUMINAMATH_GPT_mail_in_six_months_l1843_184321


namespace NUMINAMATH_GPT_trapezoid_area_l1843_184326

theorem trapezoid_area 
  (a b c : ℝ)
  (h_a : a = 5)
  (h_b : b = 15)
  (h_c : c = 13)
  : (1 / 2) * (a + b) * (Real.sqrt (c ^ 2 - ((b - a) / 2) ^ 2)) = 120 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1843_184326


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1843_184322

theorem quadratic_no_real_roots (m : ℝ) : ¬ ∃ x : ℝ, x^2 + 2 * x - m = 0 → m < -1 := 
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_no_real_roots_l1843_184322


namespace NUMINAMATH_GPT_rohit_distance_from_start_l1843_184332

noncomputable def rohit_final_position : ℕ × ℕ :=
  let start := (0, 0)
  let p1 := (start.1, start.2 - 25)       -- Moves 25 meters south.
  let p2 := (p1.1 + 20, p1.2)           -- Turns left (east) and moves 20 meters.
  let p3 := (p2.1, p2.2 + 25)           -- Turns left (north) and moves 25 meters.
  let result := (p3.1 + 15, p3.2)       -- Turns right (east) and moves 15 meters.
  result

theorem rohit_distance_from_start :
  rohit_final_position = (35, 0) :=
sorry

end NUMINAMATH_GPT_rohit_distance_from_start_l1843_184332


namespace NUMINAMATH_GPT_inscribed_square_ratio_l1843_184369

-- Define the problem context:
variables {x y : ℝ}

-- Conditions on the triangles and squares:
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

def inscribed_square_first_triangle (a b c x : ℝ) : Prop :=
  is_right_triangle a b c ∧ a = 5 ∧ b = 12 ∧ c = 13 ∧
  x = 60 / 17

def inscribed_square_second_triangle (d e f y : ℝ) : Prop :=
  is_right_triangle d e f ∧ d = 6 ∧ e = 8 ∧ f = 10 ∧
  y = 25 / 8

-- Lean theorem to be proven with given conditions:
theorem inscribed_square_ratio :
  inscribed_square_first_triangle 5 12 13 x →
  inscribed_square_second_triangle 6 8 10 y →
  x / y = 96 / 85 := by
  sorry

end NUMINAMATH_GPT_inscribed_square_ratio_l1843_184369


namespace NUMINAMATH_GPT_runners_meet_opposite_dir_l1843_184365

theorem runners_meet_opposite_dir 
  {S x y : ℝ}
  (h1 : S / x + 5 = S / y)
  (h2 : S / (x - y) = 30) :
  S / (x + y) = 6 := 
sorry

end NUMINAMATH_GPT_runners_meet_opposite_dir_l1843_184365


namespace NUMINAMATH_GPT_two_digit_numbers_of_form_3_pow_n_l1843_184318

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_of_form_3_pow_n_l1843_184318


namespace NUMINAMATH_GPT_sqrt_meaningful_iff_l1843_184356

theorem sqrt_meaningful_iff (x: ℝ) : (6 - 2 * x ≥ 0) ↔ (x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_iff_l1843_184356


namespace NUMINAMATH_GPT_factorize_expression_simplify_fraction_expr_l1843_184338

-- (1) Prove the factorization of m^3 - 4m^2 + 4m
theorem factorize_expression (m : ℝ) : 
  m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

-- (2) Simplify the fraction operation correctly
theorem simplify_fraction_expr (x : ℝ) (h : x ≠ 1) : 
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_simplify_fraction_expr_l1843_184338


namespace NUMINAMATH_GPT_solve_fraction_equation_l1843_184339

theorem solve_fraction_equation :
  {x : ℝ | (1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 15 * x - 12) = 0)} =
  {1, -12, 12, -1} :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1843_184339


namespace NUMINAMATH_GPT_find_t_over_q_l1843_184379

theorem find_t_over_q
  (q r s v t : ℝ)
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := 
sorry

end NUMINAMATH_GPT_find_t_over_q_l1843_184379


namespace NUMINAMATH_GPT_calculate_star_difference_l1843_184389

def star (a b : ℕ) : ℕ := a^2 + 2 * a * b + b^2

theorem calculate_star_difference : (star 3 5) - (star 2 4) = 28 := by
  sorry

end NUMINAMATH_GPT_calculate_star_difference_l1843_184389


namespace NUMINAMATH_GPT_icosahedron_minimal_rotation_l1843_184354

structure Icosahedron :=
  (faces : ℕ)
  (is_regular : Prop)
  (face_shape : Prop)

def icosahedron := Icosahedron.mk 20 (by sorry) (by sorry)

def theta (θ : ℝ) : Prop :=
  ∃ θ > 0, ∀ h : Icosahedron, 
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72

theorem icosahedron_minimal_rotation :
  ∃ θ > 0, ∀ h : Icosahedron,
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72 :=
by sorry

end NUMINAMATH_GPT_icosahedron_minimal_rotation_l1843_184354


namespace NUMINAMATH_GPT_product_area_perimeter_eq_104sqrt26_l1843_184317

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2).sqrt

noncomputable def side_length := distance (5, 5) (0, 4)

noncomputable def area_of_square := side_length ^ 2

noncomputable def perimeter_of_square := 4 * side_length

noncomputable def product_area_perimeter := area_of_square * perimeter_of_square

theorem product_area_perimeter_eq_104sqrt26 :
  product_area_perimeter = 104 * Real.sqrt 26 :=
by 
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_product_area_perimeter_eq_104sqrt26_l1843_184317


namespace NUMINAMATH_GPT_moles_of_AgOH_formed_l1843_184315

theorem moles_of_AgOH_formed (moles_AgNO3 : ℕ) (moles_NaOH : ℕ) 
  (reaction : moles_AgNO3 + moles_NaOH = 2) : moles_AgNO3 + 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_AgOH_formed_l1843_184315


namespace NUMINAMATH_GPT_inequality_holds_for_k_2_l1843_184337

theorem inequality_holds_for_k_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_holds_for_k_2_l1843_184337


namespace NUMINAMATH_GPT_common_difference_is_1_l1843_184309

variable (a_2 a_5 : ℕ) (d : ℤ)

def arithmetic_sequence (n a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

theorem common_difference_is_1 
  (h1 : arithmetic_sequence 2 a_1 d = 3) 
  (h2 : arithmetic_sequence 5 a_1 d = 6) : 
  d = 1 := 
sorry

end NUMINAMATH_GPT_common_difference_is_1_l1843_184309
