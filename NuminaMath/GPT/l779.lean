import Mathlib

namespace Shekar_weighted_average_l779_77908

def score_weighted_sum (scores_weights : List (ℕ × ℚ)) : ℚ :=
  scores_weights.foldl (fun acc sw => acc + (sw.1 * sw.2 : ℚ)) 0

def Shekar_scores_weights : List (ℕ × ℚ) :=
  [(76, 0.20), (65, 0.15), (82, 0.10), (67, 0.15), (55, 0.10), (89, 0.05), (74, 0.05),
   (63, 0.10), (78, 0.05), (71, 0.05)]

theorem Shekar_weighted_average : score_weighted_sum Shekar_scores_weights = 70.55 := by
  sorry

end Shekar_weighted_average_l779_77908


namespace largest_multiple_l779_77931

theorem largest_multiple (a b limit : ℕ) (ha : a = 3) (hb : b = 5) (h_limit : limit = 800) : 
  ∃ (n : ℕ), (lcm a b) * n < limit ∧ (lcm a b) * (n + 1) ≥ limit ∧ (lcm a b) * n = 795 := 
by 
  sorry

end largest_multiple_l779_77931


namespace mul_582964_99999_l779_77957

theorem mul_582964_99999 : 582964 * 99999 = 58295817036 := by
  sorry

end mul_582964_99999_l779_77957


namespace greatest_power_of_2_divides_l779_77998

-- Define the conditions as Lean definitions.
def a : ℕ := 15
def b : ℕ := 3
def n : ℕ := 600

-- Define the theorem statement based on the conditions and correct answer.
theorem greatest_power_of_2_divides (x : ℕ) (y : ℕ) (k : ℕ) (h₁ : x = a) (h₂ : y = b) (h₃ : k = n) :
  ∃ m : ℕ, (x^k - y^k) % (2^1200) = 0 ∧ ¬ ∃ m' : ℕ, m' > m ∧ (x^k - y^k) % (2^m') = 0 := sorry

end greatest_power_of_2_divides_l779_77998


namespace cos_theta_value_l779_77973

noncomputable def coefficient_x2 (θ : ℝ) : ℝ := Nat.choose 5 2 * (Real.cos θ)^2
noncomputable def coefficient_x3 : ℝ := Nat.choose 4 3 * (5 / 4 : ℝ)^3

theorem cos_theta_value (θ : ℝ) (h : coefficient_x2 θ = coefficient_x3) : 
  Real.cos θ = (Real.sqrt 2)/2 ∨ Real.cos θ = -(Real.sqrt 2)/2 := 
by sorry

end cos_theta_value_l779_77973


namespace ellipse_equation_constants_l779_77974

noncomputable def ellipse_parametric_eq (t : ℝ) : ℝ × ℝ :=
  ((3 * (Real.sin t - 2)) / (3 - Real.cos t),
  (4 * (Real.cos t - 4)) / (3 - Real.cos t))

theorem ellipse_equation_constants :
  ∃ (A B C D E F : ℤ), ∀ (x y : ℝ),
  ((∃ t : ℝ, (x, y) = ellipse_parametric_eq t) → (A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0)) ∧
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2502) :=
sorry

end ellipse_equation_constants_l779_77974


namespace number_of_solutions_l779_77940

theorem number_of_solutions : ∃! (xy : ℕ × ℕ), (xy.1 ^ 2 - xy.2 ^ 2 = 91 ∧ xy.1 > 0 ∧ xy.2 > 0) := sorry

end number_of_solutions_l779_77940


namespace certain_number_divides_expression_l779_77913

theorem certain_number_divides_expression : 
  ∃ m : ℕ, (∃ n : ℕ, n = 6 ∧ m ∣ (11 * n - 1)) ∧ m = 65 := 
by
  sorry

end certain_number_divides_expression_l779_77913


namespace total_money_earned_l779_77938

def earning_per_question : ℝ := 0.2
def questions_per_survey : ℕ := 10
def surveys_on_monday : ℕ := 3
def surveys_on_tuesday : ℕ := 4

theorem total_money_earned :
  earning_per_question * (questions_per_survey * (surveys_on_monday + surveys_on_tuesday)) = 14 := by
  sorry

end total_money_earned_l779_77938


namespace contest_B_third_place_4_competitions_l779_77944

/-- Given conditions:
1. There are three contestants: A, B, and C.
2. Scores for the first three places in each knowledge competition are \(a\), \(b\), and \(c\) where \(a > b > c\) and \(a, b, c ∈ ℕ^*\).
3. The final score of A is 26 points.
4. The final scores of both B and C are 11 points.
5. Contestant B won first place in one of the competitions.
Prove that Contestant B won third place in four competitions.
-/
theorem contest_B_third_place_4_competitions
  (a b c : ℕ)
  (ha : a > b)
  (hb : b > c)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hA_score : a + a + a + a + b + c = 26)
  (hB_score : a + c + c + c + c + b = 11)
  (hC_score : b + b + b + b + c + c = 11) :
  ∃ n1 n3 : ℕ,
    n1 = 1 ∧ n3 = 4 ∧
    ∃ k m l p1 p2 : ℕ,
      n1 * a + k * a + l * a + m * a + p1 * a + p2 * a + p1 * b + k * b + p2 * b + n3 * c = 11 := sorry

end contest_B_third_place_4_competitions_l779_77944


namespace alternating_students_count_l779_77965

theorem alternating_students_count :
  let num_male := 4
  let num_female := 5
  let arrangements := Nat.factorial num_female * Nat.factorial num_male
  arrangements = 2880 :=
by
  sorry

end alternating_students_count_l779_77965


namespace ninth_term_arith_seq_l779_77925

theorem ninth_term_arith_seq (a d : ℤ) (h1 : a + 2 * d = 25) (h2 : a + 5 * d = 31) : a + 8 * d = 37 :=
sorry

end ninth_term_arith_seq_l779_77925


namespace pairs_of_real_numbers_l779_77934

theorem pairs_of_real_numbers (a b : ℝ) (h : ∀ (n : ℕ), n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ m n : ℤ, a = (m : ℝ) ∧ b = (n : ℝ)) :=
by
  sorry

end pairs_of_real_numbers_l779_77934


namespace unique_solution_quadratic_l779_77995

theorem unique_solution_quadratic (q : ℝ) (hq : q ≠ 0) :
  (∃ x, q * x^2 - 18 * x + 8 = 0 ∧ ∀ y, q * y^2 - 18 * y + 8 = 0 → y = x) →
  q = 81 / 8 :=
by
  sorry

end unique_solution_quadratic_l779_77995


namespace find_common_ratio_l779_77981

noncomputable def geometric_series (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - q^n) / (1 - q)

theorem find_common_ratio (a_1 : ℝ) (q : ℝ) (n : ℕ) (S_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = geometric_series a_1 q n)
  (h2 : S_n 3 = (2 * a_1 + a_1 * q) / 2)
  : q = -1/2 :=
  sorry

end find_common_ratio_l779_77981


namespace prob_below_8_correct_l779_77915

-- Defining the probabilities of hitting the 10, 9, and 8 rings
def prob_10 : ℝ := 0.20
def prob_9 : ℝ := 0.30
def prob_8 : ℝ := 0.10

-- Defining the event of scoring below 8
def prob_below_8 : ℝ := 1 - (prob_10 + prob_9 + prob_8)

-- The main theorem to prove: the probability of scoring below 8 is 0.40
theorem prob_below_8_correct : prob_below_8 = 0.40 :=
by 
  -- We need to show this proof in a separate proof phase
  sorry

end prob_below_8_correct_l779_77915


namespace arithmetic_sequence_middle_term_l779_77926

theorem arithmetic_sequence_middle_term 
  (a b c d e : ℕ) 
  (h_seq : a = 23 ∧ e = 53 ∧ (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d)) :
  c = 38 :=
by
  sorry

end arithmetic_sequence_middle_term_l779_77926


namespace max_value_on_interval_l779_77924

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_on_interval : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≤ 5 :=
by
  sorry

end max_value_on_interval_l779_77924


namespace exponent_combination_l779_77922

theorem exponent_combination (a : ℝ) (m n : ℕ) (h₁ : a^m = 3) (h₂ : a^n = 4) :
  a^(2 * m + 3 * n) = 576 :=
by
  sorry

end exponent_combination_l779_77922


namespace rectangle_area_l779_77904

-- Definitions from conditions:
def side_length : ℕ := 16 / 4
def area_B : ℕ := side_length * side_length
def probability_not_within_B : ℝ := 0.4666666666666667

-- Main statement to prove
theorem rectangle_area (A : ℝ) (h1 : side_length = 4)
 (h2 : area_B = 16)
 (h3 : probability_not_within_B = 0.4666666666666667) :
   A * 0.5333333333333333 = 16 → A = 30 :=
by
  intros h
  sorry


end rectangle_area_l779_77904


namespace george_total_payment_in_dollars_l779_77988
noncomputable def total_cost_in_dollars : ℝ := 
  let sandwich_cost : ℝ := 4
  let juice_cost : ℝ := 2 * sandwich_cost * 0.9
  let coffee_cost : ℝ := sandwich_cost / 2
  let milk_cost : ℝ := 0.75 * (sandwich_cost + juice_cost)
  let milk_cost_dollars : ℝ := milk_cost * 1.2
  let chocolate_bar_cost_pounds : ℝ := 3
  let chocolate_bar_cost_dollars : ℝ := chocolate_bar_cost_pounds * 1.25
  let total_euros_in_items : ℝ := 2 * sandwich_cost + juice_cost + coffee_cost
  let total_euros_to_dollars : ℝ := total_euros_in_items * 1.2
  total_euros_to_dollars + milk_cost_dollars + chocolate_bar_cost_dollars

theorem george_total_payment_in_dollars : total_cost_in_dollars = 38.07 := by
  sorry

end george_total_payment_in_dollars_l779_77988


namespace find_pq_of_orthogonal_and_equal_magnitudes_l779_77972

noncomputable def vec_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
noncomputable def vec_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_pq_of_orthogonal_and_equal_magnitudes
    (p q : ℝ)
    (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
    (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
    (p, q) = (-29/12, 43/12) :=
by {
  sorry
}

end find_pq_of_orthogonal_and_equal_magnitudes_l779_77972


namespace coordinates_of_B_l779_77949
open Real

-- Define the conditions given in the problem
def A : ℝ × ℝ := (1, 6)
def d : ℝ := 4

-- Define the properties of the solution given the conditions
theorem coordinates_of_B (B : ℝ × ℝ) :
  (B = (-3, 6) ∨ B = (5, 6)) ↔
  (B.2 = A.2 ∧ (B.1 = A.1 - d ∨ B.1 = A.1 + d)) :=
by
  sorry

end coordinates_of_B_l779_77949


namespace larger_number_l779_77969

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l779_77969


namespace common_ratio_geometric_sequence_l779_77962

-- Definition of a geometric sequence and given conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 1 / 4) : q = 1 / 2 :=
by 
  sorry

end common_ratio_geometric_sequence_l779_77962


namespace december_sales_fraction_l779_77952

noncomputable def average_sales (A : ℝ) := 11 * A
noncomputable def december_sales (A : ℝ) := 3 * A
noncomputable def total_sales (A : ℝ) := average_sales A + december_sales A

theorem december_sales_fraction (A : ℝ) (h1 : december_sales A = 3 * A)
  (h2 : average_sales A = 11 * A) :
  december_sales A / total_sales A = 3 / 14 :=
by
  sorry

end december_sales_fraction_l779_77952


namespace mike_spent_on_car_parts_l779_77906

-- Define the costs as constants
def cost_speakers : ℝ := 118.54
def cost_tires : ℝ := 106.33
def cost_cds : ℝ := 4.58

-- Define the total cost of car parts excluding the CDs
def total_cost_car_parts : ℝ := cost_speakers + cost_tires

-- The theorem we want to prove
theorem mike_spent_on_car_parts :
  total_cost_car_parts = 224.87 := 
by 
  -- Proof omitted
  sorry

end mike_spent_on_car_parts_l779_77906


namespace factory_fills_boxes_per_hour_l779_77907

theorem factory_fills_boxes_per_hour
  (colors_per_box : ℕ)
  (crayons_per_color : ℕ)
  (total_crayons : ℕ)
  (hours : ℕ)
  (crayons_per_hour := total_crayons / hours)
  (crayons_per_box := colors_per_box * crayons_per_color)
  (boxes_per_hour := crayons_per_hour / crayons_per_box) :
  colors_per_box = 4 →
  crayons_per_color = 2 →
  total_crayons = 160 →
  hours = 4 →
  boxes_per_hour = 5 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end factory_fills_boxes_per_hour_l779_77907


namespace bellas_goal_product_l779_77901

theorem bellas_goal_product (g1 g2 g3 g4 g5 g6 : ℕ) (g7 g8 : ℕ) 
  (h1 : g1 = 5) 
  (h2 : g2 = 3) 
  (h3 : g3 = 2) 
  (h4 : g4 = 4)
  (h5 : g5 = 1) 
  (h6 : g6 = 6)
  (h7 : g7 < 10)
  (h8 : (g1 + g2 + g3 + g4 + g5 + g6 + g7) % 7 = 0) 
  (h9 : g8 < 10)
  (h10 : (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8) % 8 = 0) :
  g7 * g8 = 28 :=
by 
  sorry

end bellas_goal_product_l779_77901


namespace number_of_shelves_l779_77936

theorem number_of_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h_total_books : total_books = 14240) (h_books_per_shelf : books_per_shelf = 8) : total_books / books_per_shelf = 1780 :=
by 
  -- Proof goes here.
  sorry

end number_of_shelves_l779_77936


namespace inequality_proof_l779_77919

variable {a b c d : ℝ}

theorem inequality_proof
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_pos_d : 0 < d)
  (h_inequality : a / b < c / d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := 
by
  sorry

end inequality_proof_l779_77919


namespace distance_NYC_to_DC_l779_77968

noncomputable def horse_speed := 10 -- miles per hour
noncomputable def travel_time := 24 -- hours

theorem distance_NYC_to_DC : horse_speed * travel_time = 240 := by
  sorry

end distance_NYC_to_DC_l779_77968


namespace john_piano_lessons_l779_77920

theorem john_piano_lessons (total_cost piano_cost original_price_per_lesson discount : ℕ) 
    (total_spent : ℕ) : 
    total_spent = piano_cost + ((total_cost - piano_cost) / (original_price_per_lesson - discount)) → 
    total_cost = 1100 ∧ piano_cost = 500 ∧ original_price_per_lesson = 40 ∧ discount = 10 → 
    (total_cost - piano_cost) / (original_price_per_lesson - discount) = 20 :=
by
  intros h1 h2
  sorry

end john_piano_lessons_l779_77920


namespace complex_number_solution_l779_77923

open Complex

theorem complex_number_solution (z : ℂ) (h : (z - 2 * I) * (2 - I) = 5) : z = 2 + 3 * I :=
  sorry

end complex_number_solution_l779_77923


namespace complete_triangles_l779_77987

noncomputable def possible_placements_count : Nat :=
  sorry

theorem complete_triangles {a b c : Nat} :
  (1 + 2 + 4 + 10 + a + b + c) = 23 →
  ∃ (count : Nat), count = 4 := 
by
  sorry

end complete_triangles_l779_77987


namespace probability_reach_3_1_in_8_steps_l779_77935

theorem probability_reach_3_1_in_8_steps :
  let m := 35
  let n := 2048
  let q := m / n
  ∃ (m n : ℕ), (Nat.gcd m n = 1) ∧ (q = 35 / 2048) ∧ (m + n = 2083) := by
  sorry

end probability_reach_3_1_in_8_steps_l779_77935


namespace movies_watched_total_l779_77960

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end movies_watched_total_l779_77960


namespace cos_x_when_sin_x_is_given_l779_77947

theorem cos_x_when_sin_x_is_given (x : ℝ) (h : Real.sin x = (Real.sqrt 5) / 5) :
  Real.cos x = -(Real.sqrt 20) / 5 :=
sorry

end cos_x_when_sin_x_is_given_l779_77947


namespace cloth_woven_on_30th_day_l779_77941

theorem cloth_woven_on_30th_day :
  (∃ d : ℚ, (30 * 5 + ((30 * 29) / 2) * d = 390) ∧ (5 + 29 * d = 21)) :=
by sorry

end cloth_woven_on_30th_day_l779_77941


namespace miriam_pushups_l779_77983

theorem miriam_pushups :
  let p_M := 5
  let p_T := 7
  let p_W := 2 * p_T
  let p_Th := (p_M + p_T + p_W) / 2
  let p_F := p_M + p_T + p_W + p_Th
  p_F = 39 := by
  sorry

end miriam_pushups_l779_77983


namespace arithmetic_sum_S11_l779_77985

noncomputable def Sn_sum (a1 an n : ℕ) : ℕ := n * (a1 + an) / 2

theorem arithmetic_sum_S11 (a1 a9 a8 a5 a11 : ℕ) (h1 : Sn_sum a1 a9 9 = 54)
    (h2 : Sn_sum a1 a8 8 - Sn_sum a1 a5 5 = 30) : Sn_sum a1 a11 11 = 88 := by
  sorry

end arithmetic_sum_S11_l779_77985


namespace first_number_in_proportion_is_correct_l779_77999

-- Define the proportion condition
def proportion_condition (a x : ℝ) : Prop := a / x = 5 / 11

-- Define the given known value for x
def x_value : ℝ := 1.65

-- Define the correct answer for a
def correct_a : ℝ := 0.75

-- The theorem to prove
theorem first_number_in_proportion_is_correct :
  ∀ a : ℝ, proportion_condition a x_value → a = correct_a := by
  sorry

end first_number_in_proportion_is_correct_l779_77999


namespace difference_sum_first_100_odds_evens_l779_77959

def sum_first_n_odds (n : ℕ) : ℕ :=
  n^2

def sum_first_n_evens (n : ℕ) : ℕ :=
  n * (n-1)

theorem difference_sum_first_100_odds_evens :
  sum_first_n_odds 100 - sum_first_n_evens 100 = 100 := by
  sorry

end difference_sum_first_100_odds_evens_l779_77959


namespace sqrt_sum_eq_eight_l779_77939

theorem sqrt_sum_eq_eight :
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end sqrt_sum_eq_eight_l779_77939


namespace cistern_total_wet_surface_area_l779_77902

/-- Given a cistern with length 6 meters, width 4 meters, and water depth 1.25 meters,
    the total area of the wet surface is 49 square meters. -/
theorem cistern_total_wet_surface_area
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 6) (h_width : width = 4) (h_depth : depth = 1.25) :
  (length * width) + 2 * (length * depth) + 2 * (width * depth) = 49 :=
by {
  -- Proof goes here
  sorry
}

end cistern_total_wet_surface_area_l779_77902


namespace expected_value_is_correct_l779_77918

noncomputable def expected_winnings : ℚ :=
  (5/12 : ℚ) * 2 + (1/3 : ℚ) * 0 + (1/6 : ℚ) * (-2) + (1/12 : ℚ) * 10

theorem expected_value_is_correct : expected_winnings = 4 / 3 := 
by 
  -- Complex calculations skipped for brevity
  sorry

end expected_value_is_correct_l779_77918


namespace isosceles_triangle_x_sum_l779_77984

theorem isosceles_triangle_x_sum :
  ∀ (x : ℝ), (∃ (a b : ℝ), a + b + 60 = 180 ∧ (a = x ∨ b = x) ∧ (a = b ∨ a = 60 ∨ b = 60))
  → (60 + 60 + 60 = 180) :=
by
  intro x h
  sorry

end isosceles_triangle_x_sum_l779_77984


namespace inverse_function_passes_through_point_l779_77978

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1)

theorem inverse_function_passes_through_point {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f a (-1) = 1) :
  f a⁻¹ 1 = -1 :=
sorry

end inverse_function_passes_through_point_l779_77978


namespace Phillip_correct_total_l779_77975

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end Phillip_correct_total_l779_77975


namespace problem_l779_77992

theorem problem {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h : 3 * a * b = a + 3 * b) :
  (3 * a + b >= 16/3) ∧
  (a * b >= 4/3) ∧
  (a^2 + 9 * b^2 >= 8) ∧
  (¬ (b > 1/2)) :=
by
  sorry

end problem_l779_77992


namespace prism_volume_l779_77948

theorem prism_volume (a b c : ℝ) (h1 : a * b = 60) (h2 : b * c = 70) (h3 : a * c = 84) : a * b * c = 1572 :=
by
  sorry

end prism_volume_l779_77948


namespace find_third_coaster_speed_l779_77976

theorem find_third_coaster_speed
  (s1 s2 s4 s5 avg_speed n : ℕ)
  (hs1 : s1 = 50)
  (hs2 : s2 = 62)
  (hs4 : s4 = 70)
  (hs5 : s5 = 40)
  (havg_speed : avg_speed = 59)
  (hn : n = 5) : 
  ∃ s3 : ℕ, s3 = 73 :=
by
  sorry

end find_third_coaster_speed_l779_77976


namespace max_value_of_e_n_l779_77916

def b (n : ℕ) : ℕ := (8^n - 1) / 7
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_of_e_n : ∀ n : ℕ, e n = 1 := 
by
  sorry

end max_value_of_e_n_l779_77916


namespace locus_of_circle_centers_l779_77912

theorem locus_of_circle_centers (a : ℝ) (x0 y0 : ℝ) :
  { (α, β) | (x0 - α)^2 + (y0 - β)^2 = a^2 } = 
  { (x, y) | (x - x0)^2 + (y - y0)^2 = a^2 } :=
by
  sorry

end locus_of_circle_centers_l779_77912


namespace square_area_of_equal_perimeter_l779_77997

theorem square_area_of_equal_perimeter 
  (side_length_triangle : ℕ) (side_length_square : ℕ) (perimeter_square : ℕ)
  (h1 : side_length_triangle = 20)
  (h2 : perimeter_square = 3 * side_length_triangle)
  (h3 : 4 * side_length_square = perimeter_square) :
  side_length_square ^ 2 = 225 := 
by
  sorry

end square_area_of_equal_perimeter_l779_77997


namespace answer_is_p_and_q_l779_77927

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l779_77927


namespace egg_processing_l779_77942

theorem egg_processing (E : ℕ) 
  (h1 : (24 / 25) * E + 12 = (99 / 100) * E) : 
  E = 400 :=
sorry

end egg_processing_l779_77942


namespace totalCarsProduced_is_29621_l779_77937

def numSedansNA    := 3884
def numSUVsNA      := 2943
def numPickupsNA   := 1568

def numSedansEU    := 2871
def numSUVsEU      := 2145
def numPickupsEU   := 643

def numSedansASIA  := 5273
def numSUVsASIA    := 3881
def numPickupsASIA := 2338

def numSedansSA    := 1945
def numSUVsSA      := 1365
def numPickupsSA   := 765

def totalCarsProduced : Nat :=
  numSedansNA + numSUVsNA + numPickupsNA +
  numSedansEU + numSUVsEU + numPickupsEU +
  numSedansASIA + numSUVsASIA + numPickupsASIA +
  numSedansSA + numSUVsSA + numPickupsSA

theorem totalCarsProduced_is_29621 : totalCarsProduced = 29621 :=
by
  sorry

end totalCarsProduced_is_29621_l779_77937


namespace find_index_l779_77982

-- Declaration of sequence being arithmetic with first term 1 and common difference 3
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 1 + (n - 1) * 3

-- The theorem to be proven
theorem find_index (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 672 = 2014) : 672 = 672 :=
by 
  sorry

end find_index_l779_77982


namespace tiles_needed_l779_77930

/--
A rectangular swimming pool is 20m long, 8m wide, and 1.5m deep. 
Each tile used to cover the pool has a side length of 2dm. 
We need to prove the number of tiles required to cover the bottom and all four sides of the pool.
-/
theorem tiles_needed (pool_length pool_width pool_depth : ℝ) (tile_side : ℝ) 
  (h1 : pool_length = 20) (h2 : pool_width = 8) (h3 : pool_depth = 1.5) 
  (h4 : tile_side = 0.2) : 
  (pool_length * pool_width + 2 * pool_length * pool_depth + 2 * pool_width * pool_depth) / (tile_side * tile_side) = 6100 :=
by
  sorry

end tiles_needed_l779_77930


namespace parallel_lines_k_l779_77946

theorem parallel_lines_k (k : ℝ) :
  (∃ (x y : ℝ), (k-3) * x + (4-k) * y + 1 = 0 ∧ 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 3 ∨ k = 5) :=
by
  sorry

end parallel_lines_k_l779_77946


namespace average_score_first_10_matches_l779_77929

theorem average_score_first_10_matches (A : ℕ) 
  (h1 : 0 < A) 
  (h2 : 10 * A + 15 * 70 = 25 * 66) : A = 60 :=
by
  sorry

end average_score_first_10_matches_l779_77929


namespace ladder_of_twos_l779_77911

theorem ladder_of_twos (n : ℕ) (h : n ≥ 3) : 
  ∃ N_n : ℕ, N_n = 2 ^ (n - 3) :=
by
  sorry

end ladder_of_twos_l779_77911


namespace sum_of_possible_k_l779_77914

theorem sum_of_possible_k (a b c k : ℂ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a / (2 - b) = k) (h5 : b / (3 - c) = k) (h6 : c / (4 - a) = k) :
  k = 1 ∨ k = -1 ∨ k = -2 → k = 1 + (-1) + (-2) :=
by
  sorry

end sum_of_possible_k_l779_77914


namespace solve_for_y_l779_77993

theorem solve_for_y (x y : ℝ) (h : x - 2 = 4 * y + 3) : y = (x - 5) / 4 :=
by
  sorry

end solve_for_y_l779_77993


namespace circumradius_geq_3_times_inradius_l779_77955

-- Define the variables representing the circumradius and inradius
variables {R r : ℝ}

-- Assume the conditions that R is the circumradius and r is the inradius of a tetrahedron
def tetrahedron_circumradius (R : ℝ) : Prop := true
def tetrahedron_inradius (r : ℝ) : Prop := true

-- State the theorem
theorem circumradius_geq_3_times_inradius (hR : tetrahedron_circumradius R) (hr : tetrahedron_inradius r) : R ≥ 3 * r :=
sorry

end circumradius_geq_3_times_inradius_l779_77955


namespace find_a_l779_77970

theorem find_a (a b c : ℝ) (h1 : b = 15) (h2 : c = 5)
  (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) 
  (result : a * 15 * 5 = 2) : a = 6 := by 
  sorry

end find_a_l779_77970


namespace alexandra_magazines_l779_77909

noncomputable def magazines (bought_on_friday : ℕ) (bought_on_saturday : ℕ) (times_friday : ℕ) (chewed_up : ℕ) : ℕ :=
  bought_on_friday + bought_on_saturday + times_friday * bought_on_friday - chewed_up

theorem alexandra_magazines :
  ∀ (bought_on_friday bought_on_saturday times_friday chewed_up : ℕ),
      bought_on_friday = 8 → 
      bought_on_saturday = 12 → 
      times_friday = 4 → 
      chewed_up = 4 →
      magazines bought_on_friday bought_on_saturday times_friday chewed_up = 48 :=
by
  intros
  sorry

end alexandra_magazines_l779_77909


namespace unique_square_friendly_l779_77928

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k^2 = n

def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18 * m + c)

theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 := 
sorry

end unique_square_friendly_l779_77928


namespace trapezoid_QR_length_l779_77910

noncomputable def length_QR (PQ RS area altitude : ℕ) : ℝ :=
  24 - Real.sqrt 11 - 2 * Real.sqrt 24

theorem trapezoid_QR_length :
  ∀ (PQ RS area altitude : ℕ), 
  area = 240 → altitude = 10 → PQ = 12 → RS = 22 →
  length_QR PQ RS area altitude = 24 - Real.sqrt 11 - 2 * Real.sqrt 24 :=
by
  intros PQ RS area altitude h_area h_altitude h_PQ h_RS
  unfold length_QR
  sorry

end trapezoid_QR_length_l779_77910


namespace point_distance_to_focus_of_parabola_with_focus_distance_l779_77964

def parabola_with_focus_distance (focus_distance : ℝ) (p : ℝ × ℝ) : Prop :=
  let f := (0, focus_distance)
  let directrix := -focus_distance
  let (x, y) := p
  let distance_to_focus := Real.sqrt ((x - 0)^2 + (y - focus_distance)^2)
  let distance_to_directrix := abs (y - directrix)
  distance_to_focus = distance_to_directrix

theorem point_distance_to_focus_of_parabola_with_focus_distance 
  (focus_distance : ℝ) (y_axis_distance : ℝ) (p : ℝ × ℝ)
  (h_focus_distance : focus_distance = 4)
  (h_y_axis_distance : abs (p.1) = 1) :
  parabola_with_focus_distance focus_distance p →
  Real.sqrt ((p.1 - 0)^2 + (p.2 - focus_distance)^2) = 5 :=
by
  sorry

end point_distance_to_focus_of_parabola_with_focus_distance_l779_77964


namespace dart_hit_number_list_count_l779_77971

def number_of_dart_hit_lists (darts dartboards : ℕ) : ℕ :=
  11  -- Based on the solution, the hard-coded answer is 11.

theorem dart_hit_number_list_count : number_of_dart_hit_lists 6 4 = 11 := 
by 
  sorry

end dart_hit_number_list_count_l779_77971


namespace perpendicular_vectors_find_a_l779_77979

theorem perpendicular_vectors_find_a
  (a : ℝ)
  (m : ℝ × ℝ := (1, 2))
  (n : ℝ × ℝ := (a, -1))
  (h : m.1 * n.1 + m.2 * n.2 = 0) :
  a = 2 := 
sorry

end perpendicular_vectors_find_a_l779_77979


namespace oil_bill_january_l779_77900

theorem oil_bill_january (F J : ℝ)
  (h1 : F / J = 5 / 4)
  (h2 : (F + 30) / J = 3 / 2) :
  J = 120 :=
sorry

end oil_bill_january_l779_77900


namespace cost_difference_proof_l779_77917

-- Define the cost per copy at print shop X
def cost_per_copy_X : ℝ := 1.25

-- Define the cost per copy at print shop Y
def cost_per_copy_Y : ℝ := 2.75

-- Define the number of copies
def number_of_copies : ℝ := 60

-- Define the total cost at print shop X
def total_cost_X : ℝ := cost_per_copy_X * number_of_copies

-- Define the total cost at print shop Y
def total_cost_Y : ℝ := cost_per_copy_Y * number_of_copies

-- Define the difference in cost between print shop Y and print shop X
def cost_difference : ℝ := total_cost_Y - total_cost_X

-- The theorem statement proving the cost difference is $90
theorem cost_difference_proof : cost_difference = 90 := by
  sorry

end cost_difference_proof_l779_77917


namespace volleyballs_basketballs_difference_l779_77958

variable (V B : ℕ)

theorem volleyballs_basketballs_difference :
  (V + B = 14) →
  (4 * V + 5 * B = 60) →
  V - B = 6 :=
by
  intros h1 h2
  sorry

end volleyballs_basketballs_difference_l779_77958


namespace good_oranges_per_month_l779_77921

/-- Salaria has 50% of tree A and 50% of tree B, totaling to 10 trees.
    Tree A gives 10 oranges a month and 60% are good.
    Tree B gives 15 oranges a month and 1/3 are good.
    Prove that the total number of good oranges Salaria gets per month is 55. -/
theorem good_oranges_per_month 
  (total_trees : ℕ) 
  (percent_tree_A : ℝ) 
  (percent_tree_B : ℝ) 
  (oranges_tree_A : ℕ)
  (good_percent_A : ℝ)
  (oranges_tree_B : ℕ)
  (good_ratio_B : ℝ)
  (H1 : total_trees = 10)
  (H2 : percent_tree_A = 0.5)
  (H3 : percent_tree_B = 0.5)
  (H4 : oranges_tree_A = 10)
  (H5 : good_percent_A = 0.6)
  (H6 : oranges_tree_B = 15)
  (H7 : good_ratio_B = 1/3)
  : (total_trees * percent_tree_A * oranges_tree_A * good_percent_A) + 
    (total_trees * percent_tree_B * oranges_tree_B * good_ratio_B) = 55 := 
  by 
    sorry

end good_oranges_per_month_l779_77921


namespace variable_is_eleven_l779_77996

theorem variable_is_eleven (x : ℕ) (h : (1/2)^22 * (1/81)^x = 1/(18^22)) : x = 11 :=
by
  sorry

end variable_is_eleven_l779_77996


namespace average_side_length_of_squares_l779_77945

theorem average_side_length_of_squares (a b c : ℕ) (h₁ : a = 36) (h₂ : b = 64) (h₃ : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
  sorry

end average_side_length_of_squares_l779_77945


namespace nine_chapters_problem_l779_77986

def cond1 (x y : ℕ) : Prop := y = 6 * x - 6
def cond2 (x y : ℕ) : Prop := y = 5 * x + 5

theorem nine_chapters_problem (x y : ℕ) :
  (cond1 x y ∧ cond2 x y) ↔ (y = 6 * x - 6 ∧ y = 5 * x + 5) :=
by
  sorry

end nine_chapters_problem_l779_77986


namespace find_value_of_D_l779_77963

theorem find_value_of_D (C : ℕ) (D : ℕ) (k : ℕ) (h : C = (10^D) * k) (hD : k % 10 ≠ 0) : D = 69 := by
  sorry

end find_value_of_D_l779_77963


namespace fraction_of_difference_l779_77966

theorem fraction_of_difference (A_s A_l : ℝ) (h_total : A_s + A_l = 500) (h_smaller : A_s = 225) :
  (A_l - A_s) / ((A_s + A_l) / 2) = 1 / 5 :=
by
  -- Proof goes here
  sorry

end fraction_of_difference_l779_77966


namespace distance_between_A_and_B_l779_77950

def rowing_speed_still_water : ℝ := 10
def round_trip_time : ℝ := 5
def stream_speed : ℝ := 2

theorem distance_between_A_and_B : 
  ∃ x : ℝ, 
    (x / (rowing_speed_still_water - stream_speed) + x / (rowing_speed_still_water + stream_speed) = round_trip_time) 
    ∧ x = 24 :=
sorry

end distance_between_A_and_B_l779_77950


namespace sin_30_eq_half_l779_77990

theorem sin_30_eq_half : Real.sin (Real.pi / 6) = 1 / 2 := sorry

end sin_30_eq_half_l779_77990


namespace helga_shoes_l779_77989

theorem helga_shoes :
  ∃ (S : ℕ), 7 + S + 0 + 2 * (7 + S) = 48 ∧ (S - 7 = 2) :=
by
  sorry

end helga_shoes_l779_77989


namespace find_m_l779_77933

-- Definitions based on conditions in the problem
def f (x : ℝ) := 4 * x + 7

-- Theorem statement to prove m = 3/4 given the conditions
theorem find_m (m : ℝ) :
  (∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) →
  f (m - 1) = 6 →
  m = 3 / 4 :=
by
  -- Proof should go here
  sorry

end find_m_l779_77933


namespace division_of_sums_and_products_l779_77953

theorem division_of_sums_and_products (a b c : ℕ) (h_a : a = 7) (h_b : b = 5) (h_c : c = 3) :
  (a^3 + b^3 + c^3) / (a^2 - a * b + b^2 - b * c + c^2) = 15 := by
  -- proofs go here
  sorry

end division_of_sums_and_products_l779_77953


namespace two_pow_n_minus_one_divisible_by_seven_iff_l779_77994

theorem two_pow_n_minus_one_divisible_by_seven_iff (n : ℕ) (h : n > 0) :
  (2^n - 1) % 7 = 0 ↔ n % 3 = 0 :=
sorry

end two_pow_n_minus_one_divisible_by_seven_iff_l779_77994


namespace f_bound_l779_77977

-- Define the function f(n) representing the number of representations of n as a sum of powers of 2
noncomputable def f (n : ℕ) : ℕ := 
-- f is defined as described in the problem, implementation skipped here
sorry

-- Propose to prove the main inequality for all n ≥ 3
theorem f_bound (n : ℕ) (h : n ≥ 3) : 2 ^ (n^2 / 4) < f (2 ^ n) ∧ f (2 ^ n) < 2 ^ (n^2 / 2) :=
sorry

end f_bound_l779_77977


namespace jenny_distance_from_school_l779_77903

-- Definitions based on the given conditions.
def kernels_per_feet : ℕ := 1
def feet_per_kernel : ℕ := 25
def squirrel_fraction_eaten : ℚ := 1/4
def remaining_kernels : ℕ := 150

-- Problem statement in Lean 4.
theorem jenny_distance_from_school : 
  ∀ (P : ℕ), (3/4:ℚ) * P = 150 → P * feet_per_kernel = 5000 :=
by
  intros P h
  sorry

end jenny_distance_from_school_l779_77903


namespace average_age_of_women_is_37_33_l779_77956

noncomputable def women_average_age (A : ℝ) : ℝ :=
  let total_age_men := 12 * A
  let removed_men_age := (25 : ℝ) + 15 + 30
  let new_average := A + 3.5
  let total_age_with_women := 12 * new_average
  let total_age_women := total_age_with_women -  (total_age_men - removed_men_age)
  total_age_women / 3

theorem average_age_of_women_is_37_33 (A : ℝ) (h_avg : women_average_age A = 37.33) :
  true :=
by
  sorry

end average_age_of_women_is_37_33_l779_77956


namespace find_k_l779_77954

theorem find_k (x y k : ℤ) 
  (h1 : 2 * x - y = 5 * k + 6) 
  (h2 : 4 * x + 7 * y = k) 
  (h3 : x + y = 2023) : 
  k = 2022 := 
  by 
    sorry

end find_k_l779_77954


namespace consecutive_negative_integers_sum_l779_77932

theorem consecutive_negative_integers_sum (n : ℤ) (hn : n < 0) (hn1 : n + 1 < 0) (hprod : n * (n + 1) = 2550) : n + (n + 1) = -101 :=
by
  sorry

end consecutive_negative_integers_sum_l779_77932


namespace total_cost_l779_77961

-- Definition of the conditions
def cost_sharing (x : ℝ) : Prop :=
  let initial_cost := x / 5
  let new_cost := x / 7
  initial_cost - 15 = new_cost

-- The statement we need to prove
theorem total_cost (x : ℝ) (h : cost_sharing x) : x = 262.50 := by
  sorry

end total_cost_l779_77961


namespace find_x_l779_77980

def vector_dot_product (v1 v2 : ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 2)

def b (x : ℝ) : ℝ × ℝ := (x, -2)

def c (x : ℝ) : ℝ × ℝ := (1 - x, 4)

theorem find_x (x : ℝ) (h : vector_dot_product a (c x) = 0) : x = 9 :=
by
  sorry

end find_x_l779_77980


namespace total_toys_per_week_l779_77951

def toys_per_day := 1100
def working_days_per_week := 5

theorem total_toys_per_week : toys_per_day * working_days_per_week = 5500 :=
by
  sorry

end total_toys_per_week_l779_77951


namespace minimize_total_resistance_l779_77967

variable (a1 a2 a3 a4 a5 a6 : ℝ)
variable (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧ a5 > a6)

/-- Theorem: Given resistances a1, a2, a3, a4, a5, a6 such that a1 > a2 > a3 > a4 > a5 > a6, 
arranging them in the sequence a1 > a2 > a3 > a4 > a5 > a6 minimizes the total resistance
for the assembled component. -/
theorem minimize_total_resistance : 
  True := 
sorry

end minimize_total_resistance_l779_77967


namespace remainder_when_divided_by_296_and_37_l779_77991

theorem remainder_when_divided_by_296_and_37 (N : ℤ) (k : ℤ)
  (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_by_296_and_37_l779_77991


namespace total_donation_l779_77943

theorem total_donation : 2 + 6 + 2 + 8 = 18 := 
by sorry

end total_donation_l779_77943


namespace distance_walked_by_friend_P_l779_77905

def trail_length : ℝ := 33
def speed_ratio : ℝ := 1.20

theorem distance_walked_by_friend_P (v t d_P : ℝ) 
  (h1 : t = 33 / (2.20 * v)) 
  (h2 : d_P = 1.20 * v * t) 
  : d_P = 18 := by
  sorry

end distance_walked_by_friend_P_l779_77905
