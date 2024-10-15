import Mathlib

namespace NUMINAMATH_GPT_part1_part2_l889_88934

variable (a m : ℝ)

def f (x : ℝ) : ℝ := 2 * |x - 1| - a

theorem part1 (h : ∃ x, f a x - 2 * |x - 7| ≤ 0) : a ≥ -12 :=
sorry

theorem part2 (h : ∀ x, f 1 x + |x + 7| ≥ m) : m ≤ 7 :=
sorry

end NUMINAMATH_GPT_part1_part2_l889_88934


namespace NUMINAMATH_GPT_count_solutions_l889_88953

theorem count_solutions :
  ∃ (n : ℕ), (∀ (x y z : ℕ), x * y * z + x * y + y * z + z * x + x + y + z = 2012 ↔ n = 27) :=
sorry

end NUMINAMATH_GPT_count_solutions_l889_88953


namespace NUMINAMATH_GPT_area_of_triangle_is_3_l889_88958

noncomputable def area_of_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_is_3 : 
  ∀ (A B C : ℝ × ℝ), 
  A = (-5, -2) → 
  B = (0, 0) → 
  C = (7, -4) →
  area_of_triangle_ABC A B C = 3 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_3_l889_88958


namespace NUMINAMATH_GPT_final_price_relative_l889_88987

-- Definitions of the conditions
variable (x : ℝ)
#check x * 1.30  -- original price increased by 30%
#check x * 1.30 * 0.85  -- after 15% discount on increased price
#check x * 1.30 * 0.85 * 1.05  -- after applying 5% tax on discounted price

-- Theorem to prove the final price relative to the original price
theorem final_price_relative (x : ℝ) : 
  (x * 1.30 * 0.85 * 1.05) = (1.16025 * x) :=
by
  sorry

end NUMINAMATH_GPT_final_price_relative_l889_88987


namespace NUMINAMATH_GPT_value_of_a_plus_b_2023_l889_88985

theorem value_of_a_plus_b_2023 
    (x y a b : ℤ)
    (h1 : 4*x + 3*y = 11)
    (h2 : 2*x - y = 3)
    (h3 : a*x + b*y = -2)
    (h4 : b*x - a*y = 6)
    (hx : x = 2)
    (hy : y = 1) :
    (a + b) ^ 2023 = 0 := 
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_2023_l889_88985


namespace NUMINAMATH_GPT_rectangle_horizontal_length_l889_88998

variable (squareside rectheight : ℕ)

-- Condition: side of the square is 80 cm, vertical side length of the rectangle is 100 cm
def square_side_length := 80
def rect_vertical_length := 100

-- Question: Calculate the horizontal length of the rectangle
theorem rectangle_horizontal_length :
  (4 * square_side_length) = (2 * rect_vertical_length + 2 * rect_horizontal_length) -> rect_horizontal_length = 60 := by
  sorry

end NUMINAMATH_GPT_rectangle_horizontal_length_l889_88998


namespace NUMINAMATH_GPT_total_hexagons_calculation_l889_88911

-- Define the conditions
-- Regular hexagon side length
def hexagon_side_length : ℕ := 3

-- Number of smaller triangles
def small_triangle_count : ℕ := 54

-- Small triangle side length
def small_triangle_side_length : ℕ := 1

-- Define the total number of hexagons calculated
def total_hexagons : ℕ := 36

-- Theorem stating that given the conditions, the total number of hexagons is 36
theorem total_hexagons_calculation :
    (hexagon_side_length = 3) →
    (small_triangle_count = 54) →
    (small_triangle_side_length = 1) →
    total_hexagons = 36 :=
    by
    intros
    sorry

end NUMINAMATH_GPT_total_hexagons_calculation_l889_88911


namespace NUMINAMATH_GPT_trig_identity_proof_l889_88910

theorem trig_identity_proof :
  let sin := Real.sin
  let cos := Real.cos
  let deg_to_rad := fun θ : ℝ => θ * Real.pi / 180
  sin (deg_to_rad 30) * sin (deg_to_rad 75) - sin (deg_to_rad 60) * cos (deg_to_rad 105) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l889_88910


namespace NUMINAMATH_GPT_totalFourOfAKindCombinations_l889_88933

noncomputable def numberOfFourOfAKindCombinations : Nat :=
  13 * 48

theorem totalFourOfAKindCombinations : numberOfFourOfAKindCombinations = 624 := by
  sorry

end NUMINAMATH_GPT_totalFourOfAKindCombinations_l889_88933


namespace NUMINAMATH_GPT_find_circle_center_l889_88942

-- Definition of the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 6*x + y^2 + 10*y - 7 = 0

-- The main statement to prove
theorem find_circle_center :
  (∃ center : ℝ × ℝ, center = (3, -5) ∧ ∀ x y : ℝ, circle_eq x y ↔ (x - 3)^2 + (y + 5)^2 = 41) :=
sorry

end NUMINAMATH_GPT_find_circle_center_l889_88942


namespace NUMINAMATH_GPT_gear_ratio_proportion_l889_88901

variables {x y z w : ℕ} {ω_A ω_B ω_C ω_D : ℝ}

theorem gear_ratio_proportion 
  (h1: x * ω_A = y * ω_B) 
  (h2: y * ω_B = z * ω_C) 
  (h3: z * ω_C = w * ω_D):
  ω_A / ω_B = y * z * w / (x * z * w) ∧ 
  ω_B / ω_C = x * z * w / (y * x * w) ∧ 
  ω_C / ω_D = x * y * w / (z * y * w) ∧ 
  ω_D / ω_A = x * y * z / (w * z * y) :=
sorry  -- Proof is not included

end NUMINAMATH_GPT_gear_ratio_proportion_l889_88901


namespace NUMINAMATH_GPT_number_of_different_pairs_l889_88915

theorem number_of_different_pairs :
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48 :=
by
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  show (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48
  sorry

end NUMINAMATH_GPT_number_of_different_pairs_l889_88915


namespace NUMINAMATH_GPT_no_integer_solution_l889_88967

theorem no_integer_solution (a b : ℤ) : ¬ (4 ∣ a^2 + b^2 + 1) :=
by
  -- Prevent use of the solution steps and add proof obligations
  sorry

end NUMINAMATH_GPT_no_integer_solution_l889_88967


namespace NUMINAMATH_GPT_ratio_w_y_l889_88945

theorem ratio_w_y (w x y z : ℚ) 
  (h1 : w / x = 5 / 2) 
  (h2 : y / z = 5 / 3) 
  (h3 : z / x = 1 / 6) : 
  w / y = 9 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_w_y_l889_88945


namespace NUMINAMATH_GPT_original_price_per_tire_l889_88903

-- Definitions derived from the problem
def number_of_tires : ℕ := 4
def sale_price_per_tire : ℝ := 75
def total_savings : ℝ := 36

-- Goal to prove the original price of each tire
theorem original_price_per_tire :
  (sale_price_per_tire + total_savings / number_of_tires) = 84 :=
by sorry

end NUMINAMATH_GPT_original_price_per_tire_l889_88903


namespace NUMINAMATH_GPT_log35_28_l889_88908

variable (a b : ℝ)
variable (log : ℝ → ℝ → ℝ)

-- Conditions
axiom log14_7_eq_a : log 14 7 = a
axiom log14_5_eq_b : log 14 5 = b

-- Theorem to prove
theorem log35_28 (h1 : log 14 7 = a) (h2 : log 14 5 = b) : log 35 28 = (2 - a) / (a + b) :=
sorry

end NUMINAMATH_GPT_log35_28_l889_88908


namespace NUMINAMATH_GPT_real_y_iff_x_interval_l889_88937

theorem real_y_iff_x_interval (x : ℝ) :
  (∃ y : ℝ, 3*y^2 + 2*x*y + x + 5 = 0) ↔ (x ≤ -3 ∨ x ≥ 5) :=
by
  sorry

end NUMINAMATH_GPT_real_y_iff_x_interval_l889_88937


namespace NUMINAMATH_GPT_smallest_square_side_length_l889_88922

theorem smallest_square_side_length (s : ℕ) :
  (∃ s, s > 3 ∧ s ≤ 4 ∧ (s - 1) * (s - 1) = 5) ↔ s = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_square_side_length_l889_88922


namespace NUMINAMATH_GPT_compare_fractions_l889_88959

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_fractions_l889_88959


namespace NUMINAMATH_GPT_unique_solution_l889_88977

noncomputable def uniquely_solvable (a : ℝ) : Prop :=
  ∀ x : ℝ, a > 0 ∧ a ≠ 1 → ∃! x, a^x = (Real.log x / Real.log (1/4))

theorem unique_solution (a : ℝ) : a > 0 ∧ a ≠ 1 → uniquely_solvable a :=
by sorry

end NUMINAMATH_GPT_unique_solution_l889_88977


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l889_88979

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set (h : ∀ x, x > -1 ∧ x < 2 → ax^2 - bx + c > 0) :
  a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l889_88979


namespace NUMINAMATH_GPT_problem_b_50_l889_88930

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n ≥ 1, b (n + 1) = b n + 3 * n

theorem problem_b_50 (b : ℕ → ℕ) (h : seq b) : b 50 = 3678 := 
sorry

end NUMINAMATH_GPT_problem_b_50_l889_88930


namespace NUMINAMATH_GPT_scatter_plot_exists_l889_88940

theorem scatter_plot_exists (sample_data : List (ℝ × ℝ)) :
  ∃ plot : List (ℝ × ℝ), plot = sample_data :=
by
  sorry

end NUMINAMATH_GPT_scatter_plot_exists_l889_88940


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l889_88923

theorem angle_in_second_quadrant (n : ℤ) : (460 : ℝ) = 360 * n + 100 := by
  sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l889_88923


namespace NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l889_88952

theorem ninth_term_arithmetic_sequence :
  ∃ (a d : ℤ), (a + 2 * d = 5 ∧ a + 5 * d = 17) ∧ (a + 8 * d = 29) := 
by
  sorry

end NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l889_88952


namespace NUMINAMATH_GPT_exponentiation_rule_l889_88944

theorem exponentiation_rule (m n : ℤ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_rule_l889_88944


namespace NUMINAMATH_GPT_range_of_expression_l889_88969

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end NUMINAMATH_GPT_range_of_expression_l889_88969


namespace NUMINAMATH_GPT_total_amount_collected_l889_88919

theorem total_amount_collected (h1 : ∀ (P_I P_II : ℕ), P_I * 50 = P_II) 
                               (h2 : ∀ (F_I F_II : ℕ), F_I = 3 * F_II) 
                               (h3 : ∀ (P_II F_II : ℕ), P_II * F_II = 1250) : 
                               ∃ (Total : ℕ), Total = 1325 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_collected_l889_88919


namespace NUMINAMATH_GPT_art_collection_area_l889_88956

theorem art_collection_area :
  let square_paintings := 3 * (6 * 6)
  let small_paintings := 4 * (2 * 3)
  let large_painting := 1 * (10 * 15)
  square_paintings + small_paintings + large_painting = 282 := by
  sorry

end NUMINAMATH_GPT_art_collection_area_l889_88956


namespace NUMINAMATH_GPT_solve_system_of_equations_l889_88964

-- Conditions from the problem
variables (x y : ℚ)

-- Definitions (the original equations)
def equation1 := x + 2 * y = 3
def equation2 := 9 * x - 8 * y = 5

-- Correct answer
def solution_x := 17 / 13
def solution_y := 11 / 13

-- The final proof statement
theorem solve_system_of_equations (h1 : equation1 solution_x solution_y) (h2 : equation2 solution_x solution_y) :
  x = solution_x ∧ y = solution_y := sorry

end NUMINAMATH_GPT_solve_system_of_equations_l889_88964


namespace NUMINAMATH_GPT_crayons_birthday_l889_88980

theorem crayons_birthday (C E : ℕ) (hC : C = 523) (hE : E = 457) (hDiff : C = E + 66) : C = 523 := 
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_crayons_birthday_l889_88980


namespace NUMINAMATH_GPT_team_C_games_played_l889_88954

variable (x : ℕ)
variable (winC : ℕ := 5 * x / 7)
variable (loseC : ℕ := 2 * x / 7)
variable (winD : ℕ := 2 * x / 3)
variable (loseD : ℕ := x / 3)

theorem team_C_games_played :
  winD = winC - 5 →
  loseD = loseC - 5 →
  x = 105 := by
  sorry

end NUMINAMATH_GPT_team_C_games_played_l889_88954


namespace NUMINAMATH_GPT_no_solutions_cryptarithm_l889_88949

theorem no_solutions_cryptarithm : 
  ∀ (K O P H A B U y C : ℕ), 
  K ≠ O ∧ K ≠ P ∧ K ≠ H ∧ K ≠ A ∧ K ≠ B ∧ K ≠ U ∧ K ≠ y ∧ K ≠ C ∧ 
  O ≠ P ∧ O ≠ H ∧ O ≠ A ∧ O ≠ B ∧ O ≠ U ∧ O ≠ y ∧ O ≠ C ∧ 
  P ≠ H ∧ P ≠ A ∧ P ≠ B ∧ P ≠ U ∧ P ≠ y ∧ P ≠ C ∧ 
  H ≠ A ∧ H ≠ B ∧ H ≠ U ∧ H ≠ y ∧ H ≠ C ∧ 
  A ≠ B ∧ A ≠ U ∧ A ≠ y ∧ A ≠ C ∧ 
  B ≠ U ∧ B ≠ y ∧ B ≠ C ∧ 
  U ≠ y ∧ U ≠ C ∧ 
  y ≠ C ∧
  K < O ∧ O < P ∧ P > O ∧ O > H ∧ H > A ∧ A > B ∧ B > U ∧ U > P ∧ P > y ∧ y > C → 
  false :=
sorry

end NUMINAMATH_GPT_no_solutions_cryptarithm_l889_88949


namespace NUMINAMATH_GPT_initial_customers_count_l889_88924

theorem initial_customers_count (left_count remaining_people_per_table tables remaining_customers : ℕ) 
  (h1 : left_count = 14) 
  (h2 : remaining_people_per_table = 4) 
  (h3 : tables = 2) 
  (h4 : remaining_customers = tables * remaining_people_per_table) 
  : n = 22 :=
  sorry

end NUMINAMATH_GPT_initial_customers_count_l889_88924


namespace NUMINAMATH_GPT_stacy_history_paper_pages_l889_88997

def stacy_paper := 1 -- Number of pages Stacy writes per day
def days_to_finish := 12 -- Number of days Stacy has to finish the paper

theorem stacy_history_paper_pages : stacy_paper * days_to_finish = 12 := by
  sorry

end NUMINAMATH_GPT_stacy_history_paper_pages_l889_88997


namespace NUMINAMATH_GPT_triangle_angle_C_l889_88943

open Real

theorem triangle_angle_C (b c : ℝ) (B C : ℝ) (hb : b = sqrt 2) (hc : c = 1) (hB : B = 45) : C = 30 :=
sorry

end NUMINAMATH_GPT_triangle_angle_C_l889_88943


namespace NUMINAMATH_GPT_sum_of_digits_of_N_is_19_l889_88961

-- Given facts about N
variables (N : ℕ) (h1 : 100 ≤ N ∧ N < 1000) 
           (h2 : N % 10 = 7) 
           (h3 : N % 11 = 7) 
           (h4 : N % 12 = 7)

-- Main theorem statement
theorem sum_of_digits_of_N_is_19 : 
  ((N / 100) + ((N % 100) / 10) + (N % 10) = 19) := sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_is_19_l889_88961


namespace NUMINAMATH_GPT_rectangle_properties_l889_88950

theorem rectangle_properties (w l : ℝ) (h₁ : l = 4 * w) (h₂ : 2 * l + 2 * w = 200) :
  ∃ A d, A = 1600 ∧ d = 82.46 := 
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_properties_l889_88950


namespace NUMINAMATH_GPT_diagonal_of_rectangle_l889_88907

noncomputable def L : ℝ := 40 * Real.sqrt 3
noncomputable def W : ℝ := 30 * Real.sqrt 3
noncomputable def d : ℝ := Real.sqrt (L^2 + W^2)

theorem diagonal_of_rectangle :
  d = 50 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_diagonal_of_rectangle_l889_88907


namespace NUMINAMATH_GPT_value_of_a_sq_sub_b_sq_l889_88921

theorem value_of_a_sq_sub_b_sq (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_sq_sub_b_sq_l889_88921


namespace NUMINAMATH_GPT_debate_students_handshake_l889_88975

theorem debate_students_handshake 
    (S1 S2 S3 : ℕ)
    (h1 : S1 = 2 * S2)
    (h2 : S2 = S3 + 40)
    (h3 : S3 = 200) :
    S1 + S2 + S3 = 920 :=
by
  sorry

end NUMINAMATH_GPT_debate_students_handshake_l889_88975


namespace NUMINAMATH_GPT_prob_both_primes_l889_88978

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end NUMINAMATH_GPT_prob_both_primes_l889_88978


namespace NUMINAMATH_GPT_solution_set_l889_88995

noncomputable def f : ℝ → ℝ := sorry
def dom := {x : ℝ | x < 0 ∨ x > 0 } -- Definition of the function domain

-- Assumptions and conditions as definitions in Lean
axiom f_odd : ∀ x ∈ dom, f (-x) = -f x
axiom f_at_1 : f 1 = 1
axiom symmetric_f : ∀ x ∈ dom, (f (x + 1)) = -f (-x + 1)
axiom inequality_condition : ∀ (x1 x2 : ℝ), x1 ∈ dom → x2 ∈ dom → x1 ≠ x2 → (x1^3 * f x1 - x2^3 * f x2) / (x1 - x2) > 0

-- The main statement to be proved
theorem solution_set :
  {x ∈ dom | f x ≤ 1 / x^3} = {x ∈ dom | x ≤ -1} ∪ {x ∈ dom | 0 < x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_solution_set_l889_88995


namespace NUMINAMATH_GPT_blue_to_red_ratio_l889_88909

-- Define the conditions as given in the problem
def initial_red_balls : ℕ := 16
def lost_red_balls : ℕ := 6
def bought_yellow_balls : ℕ := 32
def total_balls_after_events : ℕ := 74

-- Based on the conditions, we define the remaining red balls and the total balls equation
def remaining_red_balls := initial_red_balls - lost_red_balls

-- Suppose B is the number of blue balls
def blue_balls (B : ℕ) : Prop :=
  remaining_red_balls + B + bought_yellow_balls = total_balls_after_events

-- Now, state the theorem to prove the ratio of blue balls to red balls is 16:5
theorem blue_to_red_ratio (B : ℕ) (h : blue_balls B) : B = 32 → B / remaining_red_balls = 16 / 5 :=
by
  intro B_eq
  subst B_eq
  have h1 : remaining_red_balls = 10 := rfl
  have h2 : 32 / 10  = 16 / 5 := by rfl
  exact h2

-- Note: The proof itself is skipped, so the statement is left with sorry.

end NUMINAMATH_GPT_blue_to_red_ratio_l889_88909


namespace NUMINAMATH_GPT_find_c_l889_88929

theorem find_c (a c : ℤ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end NUMINAMATH_GPT_find_c_l889_88929


namespace NUMINAMATH_GPT_intersection_complement_l889_88931

open Set

def U : Set ℤ := univ
def M : Set ℤ := {1, 2}
def P : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_complement :
  P ∩ (U \ M) = {-2, -1, 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l889_88931


namespace NUMINAMATH_GPT_fractions_problem_l889_88970

theorem fractions_problem (x y : ℚ) (hx : x = 2 / 3) (hy : y = 3 / 2) :
  (1 / 3) * x^5 * y^6 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_fractions_problem_l889_88970


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l889_88982

theorem largest_divisor_of_expression :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l889_88982


namespace NUMINAMATH_GPT_carX_travel_distance_after_carY_started_l889_88941

-- Define the conditions
def carX_speed : ℝ := 35
def carY_speed : ℝ := 40
def delay_time : ℝ := 1.2

-- Define the problem to prove the question is equal to the correct answer given the conditions
theorem carX_travel_distance_after_carY_started : 
  ∃ t : ℝ, carY_speed * t = carX_speed * t + carX_speed * delay_time ∧ 
           carX_speed * t = 294 :=
by
  sorry

end NUMINAMATH_GPT_carX_travel_distance_after_carY_started_l889_88941


namespace NUMINAMATH_GPT_least_n_for_distance_l889_88962

theorem least_n_for_distance (n : ℕ) : n = 17 ↔ (100 ≤ n * (n + 1) / 3) := sorry

end NUMINAMATH_GPT_least_n_for_distance_l889_88962


namespace NUMINAMATH_GPT_remainder_div_by_13_l889_88999

-- Define conditions
variable (N : ℕ)
variable (k : ℕ)

-- Given condition
def condition := N = 39 * k + 19

-- Goal statement
theorem remainder_div_by_13 (h : condition N k) : N % 13 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_div_by_13_l889_88999


namespace NUMINAMATH_GPT_cost_of_3600_pens_l889_88946

-- Define the conditions
def cost_per_200_pens : ℕ := 50
def pens_bought : ℕ := 3600

-- Define a theorem to encapsulate our question and provide the necessary definitions
theorem cost_of_3600_pens : cost_per_200_pens / 200 * pens_bought = 900 := by sorry

end NUMINAMATH_GPT_cost_of_3600_pens_l889_88946


namespace NUMINAMATH_GPT_other_root_and_m_l889_88928

-- Definitions for the conditions
def quadratic_eq (m : ℝ) := ∀ x : ℝ, x^2 + 2 * x + m = 0
def root (x : ℝ) (m : ℝ) := x^2 + 2 * x + m = 0

-- Theorem statement
theorem other_root_and_m (m : ℝ) (h : root 2 m) : ∃ t : ℝ, (2 + t = -2) ∧ (2 * t = m) ∧ t = -4 ∧ m = -8 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_other_root_and_m_l889_88928


namespace NUMINAMATH_GPT_conversion_base_10_to_5_l889_88947

theorem conversion_base_10_to_5 : 
  (425 : ℕ) = 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 0 * 5^0 :=
by sorry

end NUMINAMATH_GPT_conversion_base_10_to_5_l889_88947


namespace NUMINAMATH_GPT_approximate_value_correct_l889_88986

noncomputable def P1 : ℝ := (47 / 100) * 1442
noncomputable def P2 : ℝ := (36 / 100) * 1412
noncomputable def result : ℝ := (P1 - P2) + 63

theorem approximate_value_correct : abs (result - 232.42) < 0.01 := 
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_approximate_value_correct_l889_88986


namespace NUMINAMATH_GPT_largest_n_value_l889_88996

theorem largest_n_value (n : ℕ) (h1: n < 100000) (h2: (9 * (n - 3)^6 - n^3 + 16 * n - 27) % 7 = 0) : n = 99996 := 
sorry

end NUMINAMATH_GPT_largest_n_value_l889_88996


namespace NUMINAMATH_GPT_find_digit_A_l889_88927

theorem find_digit_A (A M C : ℕ) (h1 : A < 10) (h2 : M < 10) (h3 : C < 10) (h4 : (100 * A + 10 * M + C) * (A + M + C) = 2008) : 
  A = 2 :=
sorry

end NUMINAMATH_GPT_find_digit_A_l889_88927


namespace NUMINAMATH_GPT_Total_marbles_equal_231_l889_88974

def Connie_marbles : Nat := 39
def Juan_marbles : Nat := Connie_marbles + 25
def Maria_marbles : Nat := 2 * Juan_marbles
def Total_marbles : Nat := Connie_marbles + Juan_marbles + Maria_marbles

theorem Total_marbles_equal_231 : Total_marbles = 231 := sorry

end NUMINAMATH_GPT_Total_marbles_equal_231_l889_88974


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l889_88920

theorem arithmetic_geometric_sequence (a b c : ℝ) 
  (a_ne_b : a ≠ b) (b_ne_c : b ≠ c) (a_ne_c : a ≠ c)
  (h1 : 2 * b = a + c)
  (h2 : (a * b)^2 = a * b * c^2)
  (h3 : a + b + c = 15) : a = 20 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l889_88920


namespace NUMINAMATH_GPT_largest_trifecta_sum_l889_88981

def trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a ∣ b ∧ b ∣ c ∧ c ∣ (a * b) ∧ (100 ≤ a) ∧ (a < 1000) ∧ (100 ≤ b) ∧ (b < 1000) ∧ (100 ≤ c) ∧ (c < 1000)

theorem largest_trifecta_sum : ∃ (a b c : ℕ), trifecta a b c ∧ a + b + c = 700 :=
sorry

end NUMINAMATH_GPT_largest_trifecta_sum_l889_88981


namespace NUMINAMATH_GPT_valid_relationship_l889_88914

noncomputable def proof_statement (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : Prop :=
  b > a ∧ a > c

theorem valid_relationship (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : proof_statement a b c h_distinct h_pos h_eq :=
  sorry

end NUMINAMATH_GPT_valid_relationship_l889_88914


namespace NUMINAMATH_GPT_determine_constants_l889_88971

theorem determine_constants (k a b : ℝ) :
  (3*x^2 - 4*x + 5)*(5*x^2 + k*x + 8) = 15*x^4 - 47*x^3 + a*x^2 - b*x + 40 →
  k = -9 ∧ a = 15 ∧ b = 72 :=
by
  sorry

end NUMINAMATH_GPT_determine_constants_l889_88971


namespace NUMINAMATH_GPT_fraction_of_constants_l889_88918

theorem fraction_of_constants :
  ∃ a b c : ℤ, (4 : ℤ) * a * (k + b)^2 + c = 4 * k^2 - 8 * k + 16 ∧
             4 * -1 * (k + (-1))^2 + 12 = 4 * k^2 - 8 * k + 16 ∧
             a = 4 ∧ b = -1 ∧ c = 12 ∧ c / b = -12 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_constants_l889_88918


namespace NUMINAMATH_GPT_strokes_over_par_l889_88972

theorem strokes_over_par (n s p : ℕ) (t : ℕ) (par : ℕ )
  (h1 : n = 9)
  (h2 : s = 4)
  (h3 : p = 3)
  (h4: t = n * s)
  (h5: par = n * p) :
  t - par = 9 :=
by 
  sorry

end NUMINAMATH_GPT_strokes_over_par_l889_88972


namespace NUMINAMATH_GPT_range_of_f_l889_88955

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l889_88955


namespace NUMINAMATH_GPT_upgraded_fraction_l889_88965

theorem upgraded_fraction (N U : ℕ) (h1 : ∀ (k : ℕ), k = 24)
  (h2 : ∀ (n : ℕ), N = n) (h3 : ∀ (u : ℕ), U = u)
  (h4 : N = U / 8) : U / (24 * N + U) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_upgraded_fraction_l889_88965


namespace NUMINAMATH_GPT_selling_price_of_cricket_bat_l889_88902

variable (profit : ℝ) (profit_percentage : ℝ)
variable (selling_price : ℝ)

theorem selling_price_of_cricket_bat 
  (h1 : profit = 215)
  (h2 : profit_percentage = 33.85826771653544) : 
  selling_price = 849.70 :=
sorry

end NUMINAMATH_GPT_selling_price_of_cricket_bat_l889_88902


namespace NUMINAMATH_GPT_average_earnings_per_minute_l889_88948

theorem average_earnings_per_minute 
  (laps : ℕ) (meters_per_lap : ℕ) (dollars_per_100_meters : ℝ) (total_minutes : ℕ) (total_laps : ℕ)
  (h_laps : total_laps = 24)
  (h_meters_per_lap : meters_per_lap = 100)
  (h_dollars_per_100_meters : dollars_per_100_meters = 3.5)
  (h_total_minutes : total_minutes = 12)
  : (total_laps * meters_per_lap / 100 * dollars_per_100_meters / total_minutes) = 7 := 
by
  sorry

end NUMINAMATH_GPT_average_earnings_per_minute_l889_88948


namespace NUMINAMATH_GPT_max_profit_l889_88984

-- Define the given conditions
def cost_price : ℝ := 80
def sales_relationship (x : ℝ) : ℝ := -0.5 * x + 160
def selling_price_range (x : ℝ) : Prop := 120 ≤ x ∧ x ≤ 180

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_relationship x

-- The goal: prove the maximum profit and the selling price that achieves it
theorem max_profit : ∃ (x : ℝ), selling_price_range x ∧ profit x = 7000 := 
  sorry

end NUMINAMATH_GPT_max_profit_l889_88984


namespace NUMINAMATH_GPT_number_of_balls_condition_l889_88900

theorem number_of_balls_condition (X : ℕ) (h1 : 25 - 20 = X - 25) : X = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_balls_condition_l889_88900


namespace NUMINAMATH_GPT_ab_value_l889_88960

theorem ab_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b - 2| = 9) (h3 : a + b > 0) :
  ab = 33 ∨ ab = -33 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l889_88960


namespace NUMINAMATH_GPT_tom_used_10_plates_l889_88913

theorem tom_used_10_plates
  (weight_per_plate : ℕ := 30)
  (felt_weight : ℕ := 360)
  (heavier_factor : ℚ := 1.20) :
  (felt_weight / heavier_factor / weight_per_plate : ℚ) = 10 := by
  sorry

end NUMINAMATH_GPT_tom_used_10_plates_l889_88913


namespace NUMINAMATH_GPT_area_of_common_region_l889_88963

theorem area_of_common_region (β : ℝ) (h1 : 0 < β ∧ β < π / 2) (h2 : Real.cos β = 3 / 5) :
  ∃ (area : ℝ), area = 4 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_common_region_l889_88963


namespace NUMINAMATH_GPT_rower_rate_in_still_water_l889_88966

theorem rower_rate_in_still_water (V_m V_s : ℝ) (h1 : V_m + V_s = 16) (h2 : V_m - V_s = 12) : V_m = 14 := 
sorry

end NUMINAMATH_GPT_rower_rate_in_still_water_l889_88966


namespace NUMINAMATH_GPT_painted_rooms_l889_88912

/-- Given that there are a total of 11 rooms to paint, each room takes 7 hours to paint,
and the painter has 63 hours of work left to paint the remaining rooms,
prove that the painter has already painted 2 rooms. -/
theorem painted_rooms (total_rooms : ℕ) (hours_per_room : ℕ) (hours_left : ℕ) 
  (h_total_rooms : total_rooms = 11) (h_hours_per_room : hours_per_room = 7) 
  (h_hours_left : hours_left = 63) : 
  (total_rooms - hours_left / hours_per_room) = 2 := 
by
  sorry

end NUMINAMATH_GPT_painted_rooms_l889_88912


namespace NUMINAMATH_GPT_segment_measure_l889_88989

theorem segment_measure (a b : ℝ) (m : ℝ) (h : a = m * b) : (1 / m) * a = b :=
by sorry

end NUMINAMATH_GPT_segment_measure_l889_88989


namespace NUMINAMATH_GPT_olafs_dad_points_l889_88951

-- Let D be the number of points Olaf's dad scored.
def dad_points : ℕ := sorry

-- Olaf scored three times more points than his dad.
def olaf_points (dad_points : ℕ) : ℕ := 3 * dad_points

-- Total points scored is 28.
def total_points (dad_points olaf_points : ℕ) : Prop := dad_points + olaf_points = 28

theorem olafs_dad_points (D : ℕ) :
  (D + olaf_points D = 28) → (D = 7) :=
by
  sorry

end NUMINAMATH_GPT_olafs_dad_points_l889_88951


namespace NUMINAMATH_GPT_spencer_total_distance_l889_88973

def distances : ℝ := 0.3 + 0.1 + 0.4

theorem spencer_total_distance :
  distances = 0.8 :=
sorry

end NUMINAMATH_GPT_spencer_total_distance_l889_88973


namespace NUMINAMATH_GPT_probability_calculation_l889_88936

noncomputable def probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 : ℝ :=
  let total_interval_length := 100
  let intersection_interval_length := 324 - 312.5
  intersection_interval_length / total_interval_length

theorem probability_calculation : probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 = 23 / 200 := by
  sorry

end NUMINAMATH_GPT_probability_calculation_l889_88936


namespace NUMINAMATH_GPT_total_students_in_class_l889_88932

def students_chorus := 18
def students_band := 26
def students_both := 2
def students_neither := 8

theorem total_students_in_class : 
  (students_chorus + students_band - students_both) + students_neither = 50 := by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l889_88932


namespace NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_5_l889_88988

theorem largest_integer_less_than_100_with_remainder_5 (n : ℤ) (h₁ : n < 100) (h₂ : n % 8 = 5) : n ≤ 99 :=
sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_5_l889_88988


namespace NUMINAMATH_GPT_intersection_point_of_curve_and_line_l889_88957

theorem intersection_point_of_curve_and_line : 
  ∃ (e : ℝ), (0 < e) ∧ (e = Real.exp 1) ∧ ((e, e) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), x ^ y = y ^ x ∧ 0 ≤ x ∧ 0 ≤ y}) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_point_of_curve_and_line_l889_88957


namespace NUMINAMATH_GPT_fraction_book_read_l889_88938

theorem fraction_book_read (read_pages : ℚ) (h : read_pages = 3/7) :
  (1 - read_pages = 4/7) ∧ (read_pages / (1 - read_pages) = 3/4) :=
by
  sorry

end NUMINAMATH_GPT_fraction_book_read_l889_88938


namespace NUMINAMATH_GPT_mabel_age_l889_88976

theorem mabel_age (n : ℕ) (h : n * (n + 1) / 2 = 28) : n = 7 :=
sorry

end NUMINAMATH_GPT_mabel_age_l889_88976


namespace NUMINAMATH_GPT_students_with_all_three_pets_l889_88904

theorem students_with_all_three_pets :
  ∀ (total_students : ℕ)
    (dog_fraction cat_fraction : ℚ)
    (other_pets students_no_pets dogs_only cats_only other_pets_only x y z w : ℕ),
    total_students = 40 →
    dog_fraction = 5 / 8 →
    cat_fraction = 1 / 4 →
    other_pets = 8 →
    students_no_pets = 4 →
    dogs_only = 15 →
    cats_only = 3 →
    other_pets_only = 2 →
    dogs_only + x + z + w = total_students * dog_fraction →
    cats_only + x + y + w = total_students * cat_fraction →
    other_pets_only + y + z + w = other_pets →
    dogs_only + cats_only + other_pets_only + x + y + z + w = total_students - students_no_pets →
    w = 4  := 
by
  sorry

end NUMINAMATH_GPT_students_with_all_three_pets_l889_88904


namespace NUMINAMATH_GPT_sequence_of_arrows_l889_88983

theorem sequence_of_arrows (n : ℕ) (h : n % 5 = 0) : 
  (n < 570 ∧ n % 5 = 0) → 
  (n + 1 < 573 ∧ (n + 1) % 5 = 1) → 
  (n + 2 < 573 ∧ (n + 2) % 5 = 2) → 
  (n + 3 < 573 ∧ (n + 3) % 5 = 3) →
    true :=
by
  sorry

end NUMINAMATH_GPT_sequence_of_arrows_l889_88983


namespace NUMINAMATH_GPT_poles_intersection_l889_88905

-- Define the known heights and distances
def heightOfIntersection (d h1 h2 x : ℝ) : ℝ := sorry

theorem poles_intersection :
  heightOfIntersection 120 30 60 40 = 20 := by
  sorry

end NUMINAMATH_GPT_poles_intersection_l889_88905


namespace NUMINAMATH_GPT_cos_alpha_neg_3_5_l889_88917

open Real

variables {α : ℝ} (h_alpha : sin α = 4 / 5) (h_quadrant : π / 2 < α ∧ α < π)

theorem cos_alpha_neg_3_5 : cos α = -3 / 5 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cos_alpha_neg_3_5_l889_88917


namespace NUMINAMATH_GPT_prob_two_white_balls_l889_88992

open Nat

def total_balls : ℕ := 8 + 10

def prob_first_white : ℚ := 8 / total_balls

def prob_second_white (total_balls_minus_one : ℕ) : ℚ := 7 / total_balls_minus_one

theorem prob_two_white_balls : 
  ∃ (total_balls_minus_one : ℕ) (p_first p_second : ℚ), 
    total_balls_minus_one = total_balls - 1 ∧
    p_first = prob_first_white ∧
    p_second = prob_second_white total_balls_minus_one ∧
    p_first * p_second = 28 / 153 := 
by
  sorry

end NUMINAMATH_GPT_prob_two_white_balls_l889_88992


namespace NUMINAMATH_GPT_AngeliCandies_l889_88993

def CandyProblem : Prop :=
  ∃ (C B G : ℕ), 
    (1/3 : ℝ) * C = 3 * (B : ℝ) ∧
    (2/3 : ℝ) * C = 2 * (G : ℝ) ∧
    (B + G = 40) ∧ 
    C = 144

theorem AngeliCandies :
  CandyProblem :=
sorry

end NUMINAMATH_GPT_AngeliCandies_l889_88993


namespace NUMINAMATH_GPT_largest_divisor_of_m_l889_88991

theorem largest_divisor_of_m (m : ℤ) (hm_pos : 0 < m) (h : 33 ∣ m^2) : 33 ∣ m :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_m_l889_88991


namespace NUMINAMATH_GPT_average_of_integers_is_ten_l889_88994

theorem average_of_integers_is_ten (k m r s t : ℕ) 
  (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : k > 0) (h6 : m > 0)
  (h7 : t = 20) (h8 : r = 13)
  (h9 : k = 1) (h10 : m = 2) (h11 : s = 14) :
  (k + m + r + s + t) / 5 = 10 := by
  sorry

end NUMINAMATH_GPT_average_of_integers_is_ten_l889_88994


namespace NUMINAMATH_GPT_expression_value_l889_88926

theorem expression_value :
  (100 - (3000 - 300) + (3000 - (300 - 100)) = 200) := by
  sorry

end NUMINAMATH_GPT_expression_value_l889_88926


namespace NUMINAMATH_GPT_calculate_f_g_l889_88906

noncomputable def f (x : ℕ) : ℕ := 4 * x + 3
noncomputable def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem calculate_f_g : f (g 3) = 103 :=
by 
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_calculate_f_g_l889_88906


namespace NUMINAMATH_GPT_heart_beats_during_marathon_l889_88990

theorem heart_beats_during_marathon :
  (∃ h_per_min t1 t2 total_time,
    h_per_min = 140 ∧
    t1 = 15 * 6 ∧
    t2 = 15 * 5 ∧
    total_time = t1 + t2 ∧
    23100 = h_per_min * total_time) :=
  sorry

end NUMINAMATH_GPT_heart_beats_during_marathon_l889_88990


namespace NUMINAMATH_GPT_paul_crayons_l889_88939

def initial_crayons : ℝ := 479.0
def additional_crayons : ℝ := 134.0
def total_crayons : ℝ := initial_crayons + additional_crayons

theorem paul_crayons : total_crayons = 613.0 :=
by
  sorry

end NUMINAMATH_GPT_paul_crayons_l889_88939


namespace NUMINAMATH_GPT_negation_proposition_l889_88916

theorem negation_proposition {x : ℝ} : ¬ (x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := sorry

end NUMINAMATH_GPT_negation_proposition_l889_88916


namespace NUMINAMATH_GPT_negation_of_universal_sin_l889_88968

theorem negation_of_universal_sin (h : ∀ x : ℝ, Real.sin x > 0) : ∃ x : ℝ, Real.sin x ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_universal_sin_l889_88968


namespace NUMINAMATH_GPT_simplify_expression_l889_88925

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l889_88925


namespace NUMINAMATH_GPT_problem_A_plus_B_l889_88935

variable {A B : ℝ} (h1 : A ≠ B) (h2 : ∀ x : ℝ, (A * (B * x + A) + B) - (B * (A * x + B) + A) = 2 * (B - A))

theorem problem_A_plus_B : A + B = -2 :=
by
  sorry

end NUMINAMATH_GPT_problem_A_plus_B_l889_88935
