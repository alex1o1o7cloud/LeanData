import Mathlib

namespace fill_pipe_half_cistern_time_l2356_235645

theorem fill_pipe_half_cistern_time (time_to_fill_half : ℕ) 
  (H : time_to_fill_half = 10) : 
  time_to_fill_half = 10 := 
by
  -- Proof is omitted
  sorry

end fill_pipe_half_cistern_time_l2356_235645


namespace frosting_cupcakes_l2356_235675

theorem frosting_cupcakes :
  let r1 := 1 / 15
  let r2 := 1 / 25
  let r3 := 1 / 40
  let t := 600
  t * (r1 + r2 + r3) = 79 :=
by
  sorry

end frosting_cupcakes_l2356_235675


namespace union_A_B_m_eq_3_range_of_m_l2356_235642

def A (x : ℝ) : Prop := x^2 - x - 12 ≤ 0
def B (x m : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem union_A_B_m_eq_3 :
  A x ∨ B x 3 ↔ (-3 : ℝ) ≤ x ∧ x ≤ 5 := sorry

theorem range_of_m (h : ∀ x, A x ∨ B x m ↔ A x) : m ≤ (5 / 2) := sorry

end union_A_B_m_eq_3_range_of_m_l2356_235642


namespace solve_system_nat_l2356_235639

theorem solve_system_nat (a b c d : ℕ) :
  (a * b = c + d ∧ c * d = a + b) →
  (a = 1 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 2) ∨
  (a = 5 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 5 ∧ b = 1 ∧ c = 3 ∧ d = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5) ∨
  (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
  (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) :=
sorry

end solve_system_nat_l2356_235639


namespace proof_product_eq_l2356_235624

theorem proof_product_eq (a b c d : ℚ) (h1 : 2 * a + 3 * b + 5 * c + 7 * d = 42)
    (h2 : 4 * (d + c) = b) (h3 : 2 * b + 2 * c = a) (h4 : c - 2 = d) :
    a * b * c * d = -26880 / 729 := by
  sorry

end proof_product_eq_l2356_235624


namespace candy_bar_calories_l2356_235666

theorem candy_bar_calories
  (miles_walked : ℕ)
  (calories_per_mile : ℕ)
  (net_calorie_deficit : ℕ)
  (total_calories_burned : ℕ)
  (candy_bar_calories : ℕ)
  (h1 : miles_walked = 3)
  (h2 : calories_per_mile = 150)
  (h3 : net_calorie_deficit = 250)
  (h4 : total_calories_burned = miles_walked * calories_per_mile)
  (h5 : candy_bar_calories = total_calories_burned - net_calorie_deficit) :
  candy_bar_calories = 200 := 
by
  sorry

end candy_bar_calories_l2356_235666


namespace sum_of_m_and_n_l2356_235685

theorem sum_of_m_and_n (m n : ℚ) (h : (m - 3) * (Real.sqrt 5) + 2 - n = 0) : m + n = 5 :=
sorry

end sum_of_m_and_n_l2356_235685


namespace move_3m_left_is_neg_3m_l2356_235647

-- Define the notation for movements
def move_right (distance : Int) : Int := distance
def move_left (distance : Int) : Int := -distance

-- Define the specific condition
def move_1m_right : Int := move_right 1

-- Define the assertion for moving 3m to the left
def move_3m_left : Int := move_left 3

-- State the proof problem
theorem move_3m_left_is_neg_3m : move_3m_left = -3 := by
  unfold move_3m_left
  unfold move_left
  rfl

end move_3m_left_is_neg_3m_l2356_235647


namespace tori_needs_more_correct_answers_l2356_235616

theorem tori_needs_more_correct_answers :
  let total_questions := 80
  let arithmetic_questions := 20
  let algebra_questions := 25
  let geometry_questions := 35
  let arithmetic_correct := 0.60 * arithmetic_questions
  let algebra_correct := Float.round (0.50 * algebra_questions)
  let geometry_correct := Float.round (0.70 * geometry_questions)
  let correct_answers := arithmetic_correct + algebra_correct + geometry_correct
  let passing_percentage := 0.65
  let required_correct := passing_percentage * total_questions
-- assertion
  required_correct - correct_answers = 2 := 
by 
  sorry

end tori_needs_more_correct_answers_l2356_235616


namespace necessary_and_sufficient_condition_l2356_235676

theorem necessary_and_sufficient_condition 
  (a b c : ℝ) :
  (a^2 = b^2 + c^2) ↔
  (∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0) := 
sorry

end necessary_and_sufficient_condition_l2356_235676


namespace system_sum_of_squares_l2356_235664

theorem system_sum_of_squares :
  (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    9*y1^2 - 4*x1^2 = 144 - 48*x1 ∧ 9*y1^2 + 4*x1^2 = 144 + 18*x1*y1 ∧
    9*y2^2 - 4*x2^2 = 144 - 48*x2 ∧ 9*y2^2 + 4*x2^2 = 144 + 18*x2*y2 ∧
    9*y3^2 - 4*x3^2 = 144 - 48*x3 ∧ 9*y3^2 + 4*x3^2 = 144 + 18*x3*y3 ∧
    (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2 = 68)) :=
by sorry

end system_sum_of_squares_l2356_235664


namespace number_of_students_l2356_235650

/-- 
We are given that 36 students are selected from three grades: 
15 from the first grade, 12 from the second grade, and the rest from the third grade. 
Additionally, there are 900 students in the third grade.
We need to prove: the total number of students in the high school is 3600
-/
theorem number_of_students (x y z : ℕ) (s_total : ℕ) (x_sel : ℕ) (y_sel : ℕ) (z_students : ℕ) 
  (h1 : x_sel = 15) 
  (h2 : y_sel = 12) 
  (h3 : x_sel + y_sel + (s_total - (x_sel + y_sel)) = s_total) 
  (h4 : s_total = 36) 
  (h5 : z_students = 900) 
  (h6 : (s_total - (x_sel + y_sel)) = 9) 
  (h7 : 9 / 900 = 1 / 100) : 
  (36 * 100 = 3600) :=
by sorry

end number_of_students_l2356_235650


namespace division_correct_l2356_235694

theorem division_correct (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 :=
by
  sorry

end division_correct_l2356_235694


namespace sum_in_range_l2356_235687

theorem sum_in_range :
  let a := (27 : ℚ) / 8
  let b := (22 : ℚ) / 5
  let c := (67 : ℚ) / 11
  13 < a + b + c ∧ a + b + c < 14 :=
by
  sorry

end sum_in_range_l2356_235687


namespace intersecting_lines_l2356_235658

-- Definitions based on conditions
def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 4
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- Lean 4 Statement of the problem
theorem intersecting_lines (m b : ℝ) (h1 : line1 m 6 = 10) (h2 : line2 b 6 = 10) : b + m = -7 :=
by
  sorry

end intersecting_lines_l2356_235658


namespace find_x_l2356_235646

def star (p q : Int × Int) : Int × Int :=
  (p.1 + q.2, p.2 - q.1)

theorem find_x : ∀ (x y : Int), star (x, y) (4, 2) = (5, 4) → x = 3 :=
by
  intros x y h
  -- The statement is correct, just add a placeholder for the proof
  sorry

end find_x_l2356_235646


namespace technician_round_trip_percentage_l2356_235656

theorem technician_round_trip_percentage (D: ℝ) (hD: D ≠ 0): 
  let round_trip_distance := 2 * D
  let distance_to_center := D
  let distance_back_10_percent := 0.10 * D
  let total_distance_completed := distance_to_center + distance_back_10_percent
  let percentage_completed := (total_distance_completed / round_trip_distance) * 100
  percentage_completed = 55 := 
by
  simp
  sorry -- Proof is not required per instructions

end technician_round_trip_percentage_l2356_235656


namespace pencils_per_row_l2356_235693

-- Define the conditions
def total_pencils := 25
def number_of_rows := 5

-- Theorem statement: The number of pencils per row is 5 given the conditions
theorem pencils_per_row : total_pencils / number_of_rows = 5 :=
by
  -- The proof should go here
  sorry

end pencils_per_row_l2356_235693


namespace k_ge_1_l2356_235641

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end k_ge_1_l2356_235641


namespace add_fraction_l2356_235674

theorem add_fraction (x : ℚ) (h : x - 7/3 = 3/2) : x + 7/3 = 37/6 :=
by
  sorry

end add_fraction_l2356_235674


namespace correct_exponentiation_l2356_235663

theorem correct_exponentiation : ∀ (x : ℝ), (x^(4/5))^(5/4) = x :=
by
  intro x
  sorry

end correct_exponentiation_l2356_235663


namespace range_of_f_l2356_235626

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) + Real.arctan (2 * x)

theorem range_of_f : Set.Ioo (-(Real.pi / 2)) (Real.pi / 2) = Set.range f :=
  sorry

end range_of_f_l2356_235626


namespace calculate_expression_l2356_235684

theorem calculate_expression :
  (2^3 * 3 * 5) + (18 / 2) = 129 := by
  -- Proof skipped
  sorry

end calculate_expression_l2356_235684


namespace remainder_sum_div_40_l2356_235673

variable (k m n : ℤ)
variables (a b c : ℤ)
variable (h1 : a % 80 = 75)
variable (h2 : b % 120 = 115)
variable (h3 : c % 160 = 155)

theorem remainder_sum_div_40 : (a + b + c) % 40 = 25 :=
by
  -- Use sorry as we are not required to fill in the proof
  sorry

end remainder_sum_div_40_l2356_235673


namespace find_m_l2356_235657

noncomputable def g (d e f x : ℤ) : ℤ := d * x * x + e * x + f

theorem find_m (d e f m : ℤ) (h₁ : g d e f 2 = 0)
    (h₂ : 60 < g d e f 6 ∧ g d e f 6 < 70) 
    (h₃ : 80 < g d e f 9 ∧ g d e f 9 < 90)
    (h₄ : 10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1)) :
  m = -1 :=
sorry

end find_m_l2356_235657


namespace simplify_fraction_l2356_235629

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 :=
by sorry

end simplify_fraction_l2356_235629


namespace fifteenth_term_is_correct_l2356_235611

-- Define the initial conditions of the arithmetic sequence
def firstTerm : ℕ := 4
def secondTerm : ℕ := 9

-- Calculate the common difference
def commonDifference : ℕ := secondTerm - firstTerm

-- Define the nth term formula of the arithmetic sequence
def nthTerm (a d n : ℕ) : ℕ := a + (n - 1) * d

-- The main statement: proving that the 15th term of the given sequence is 74
theorem fifteenth_term_is_correct : nthTerm firstTerm commonDifference 15 = 74 :=
by
  sorry

end fifteenth_term_is_correct_l2356_235611


namespace menu_choices_l2356_235665

theorem menu_choices :
  let lunchChinese := 5 
  let lunchJapanese := 4 
  let dinnerChinese := 3 
  let dinnerJapanese := 5 
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  lunchOptions * dinnerOptions = 72 :=
by
  let lunchChinese := 5
  let lunchJapanese := 4
  let dinnerChinese := 3
  let dinnerJapanese := 5
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  have h : lunchOptions * dinnerOptions = 72 :=
    by 
      sorry
  exact h

end menu_choices_l2356_235665


namespace max_rectangle_area_max_rectangle_area_exists_l2356_235669

theorem max_rectangle_area (l w : ℕ) (h : l + w = 20) : l * w ≤ 100 :=
by sorry

-- Alternatively, to also show the existence of the maximum value.
theorem max_rectangle_area_exists : ∃ l w : ℕ, l + w = 20 ∧ l * w = 100 :=
by sorry

end max_rectangle_area_max_rectangle_area_exists_l2356_235669


namespace alpha_nonneg_integer_l2356_235627

theorem alpha_nonneg_integer (α : ℝ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, n = k * α) : α ≥ 0 ∧ ∃ k : ℤ, α = k := 
sorry

end alpha_nonneg_integer_l2356_235627


namespace percentage_decrease_l2356_235688

theorem percentage_decrease (x : ℝ) 
  (h1 : 400 * (1 - x / 100) * 1.40 = 476) : 
  x = 15 := 
by 
  sorry

end percentage_decrease_l2356_235688


namespace number_of_strawberries_in_each_basket_l2356_235630

variable (x : ℕ) (Lilibeth_picks : 6 * x)
variable (total_strawberries : 4 * 6 * x = 1200)

theorem number_of_strawberries_in_each_basket : x = 50 := by
  sorry

end number_of_strawberries_in_each_basket_l2356_235630


namespace radius_of_inner_circle_l2356_235690

def right_triangle_legs (AC BC : ℝ) : Prop :=
  AC = 3 ∧ BC = 4

theorem radius_of_inner_circle (AC BC : ℝ) (h : right_triangle_legs AC BC) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_inner_circle_l2356_235690


namespace sum_reciprocals_factors_12_l2356_235698

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l2356_235698


namespace smallest_n_l2356_235672

-- Define the costs of candies
def cost_purple := 24
def cost_yellow := 30

-- Define the number of candies Lara can buy
def pieces_red := 10
def pieces_green := 16
def pieces_blue := 18
def pieces_yellow := 22

-- Define the total money Lara has equivalently expressed by buying candies
def lara_total_money (n : ℕ) := n * cost_purple

-- Prove the smallest value of n that satisfies the conditions stated
theorem smallest_n : ∀ n : ℕ, 
  (lara_total_money n = 10 * pieces_red * cost_purple) ∧
  (lara_total_money n = 16 * pieces_green * cost_purple) ∧
  (lara_total_money n = 18 * pieces_blue * cost_purple) ∧
  (lara_total_money n = pieces_yellow * cost_yellow) → 
  n = 30 :=
by
  intro
  sorry

end smallest_n_l2356_235672


namespace orlie_age_l2356_235609

theorem orlie_age (O R : ℕ) (h1 : R = 9) (h2 : R = (3 * O) / 4)
  (h3 : R - 4 = ((O - 4) / 2) + 1) : O = 12 :=
by
  sorry

end orlie_age_l2356_235609


namespace ticket_difference_l2356_235651

theorem ticket_difference (V G : ℕ) (h1 : V + G = 320) (h2 : 45 * V + 20 * G = 7500) :
  G - V = 232 :=
by
  sorry

end ticket_difference_l2356_235651


namespace Adam_teaches_students_l2356_235655

-- Define the conditions
def students_first_year : ℕ := 40
def students_per_year : ℕ := 50
def total_years : ℕ := 10
def remaining_years : ℕ := total_years - 1

-- Define the statement we are proving
theorem Adam_teaches_students (total_students : ℕ) :
  total_students = students_first_year + (students_per_year * remaining_years) :=
sorry

end Adam_teaches_students_l2356_235655


namespace initial_apples_l2356_235601

-- Defining the conditions
def apples_handed_out := 8
def pies_made := 6
def apples_per_pie := 9
def apples_for_pies := pies_made * apples_per_pie

-- Prove the initial number of apples
theorem initial_apples : apples_handed_out + apples_for_pies = 62 :=
by
  sorry

end initial_apples_l2356_235601


namespace difference_in_amount_paid_l2356_235636

variable (P Q : ℝ)

def original_price := P
def intended_quantity := Q

def new_price := P * 1.10
def new_quantity := Q * 0.80

theorem difference_in_amount_paid :
  ((new_price P * new_quantity Q) - (original_price P * intended_quantity Q)) = -0.12 * (original_price P * intended_quantity Q) :=
by
  sorry

end difference_in_amount_paid_l2356_235636


namespace fisherman_sale_l2356_235631

/-- 
If the price of the radio is both the 4th highest price and the 13th lowest price 
among the prices of the fishes sold at a sale, then the total number of fishes 
sold at the fisherman sale is 16. 
-/
theorem fisherman_sale (h4_highest : ∃ price : ℕ, ∀ p : ℕ, p > price → p ∈ {a | a ≠ price} ∧ p > 3)
                       (h13_lowest : ∃ price : ℕ, ∀ p : ℕ, p < price → p ∈ {a | a ≠ price} ∧ p < 13) :
  ∃ n : ℕ, n = 16 :=
sorry

end fisherman_sale_l2356_235631


namespace absolute_sum_l2356_235689

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem absolute_sum : 
    (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end absolute_sum_l2356_235689


namespace quadratic_increasing_implies_m_gt_1_l2356_235610

theorem quadratic_increasing_implies_m_gt_1 (m : ℝ) (x : ℝ) 
(h1 : x > 1) 
(h2 : ∀ x, (y = x^2 + (m-3) * x + m + 1) → (∀ z > x, y < z^2 + (m-3) * z + m + 1)) 
: m > 1 := 
sorry

end quadratic_increasing_implies_m_gt_1_l2356_235610


namespace probability_of_defective_l2356_235632

theorem probability_of_defective (p_first_grade p_second_grade : ℝ) (h_fg : p_first_grade = 0.65) (h_sg : p_second_grade = 0.3) : (1 - (p_first_grade + p_second_grade) = 0.05) :=
by
  sorry

end probability_of_defective_l2356_235632


namespace james_meditation_time_is_30_l2356_235606

noncomputable def james_meditation_time_per_session 
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) : ℕ :=
  (hours_per_week * minutes_per_hour) / (sessions_per_day * days_per_week)

theorem james_meditation_time_is_30
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) 
  (h_sessions : sessions_per_day = 2) 
  (h_days : days_per_week = 7) 
  (h_hours : hours_per_week = 7) 
  (h_minutes : minutes_per_hour = 60) : 
  james_meditation_time_per_session sessions_per_day days_per_week hours_per_week minutes_per_hour = 30 := by
  sorry

end james_meditation_time_is_30_l2356_235606


namespace find_a_l2356_235613

noncomputable def l1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 1) * y + 1
noncomputable def l2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 2

def perp_lines (a : ℝ) : Prop :=
  let m1 := -a
  let m2 := -1 / a
  m1 * m2 = -1

theorem find_a (a : ℝ) : (perp_lines a) ↔ (a = 0 ∨ a = -2) := 
sorry

end find_a_l2356_235613


namespace find_de_l2356_235644

namespace MagicSquare

variables (a b c d e : ℕ)

-- Hypotheses based on the conditions provided.
axiom H1 : 20 + 15 + a = 57
axiom H2 : 25 + b + a = 57
axiom H3 : 18 + c + a = 57
axiom H4 : 20 + c + b = 57
axiom H5 : d + c + a = 57
axiom H6 : d + e + 18 = 57
axiom H7 : e + 25 + 15 = 57

def magicSum := 57

theorem find_de :
  ∃ d e, d + e = 42 :=
by sorry

end MagicSquare

end find_de_l2356_235644


namespace tina_husband_brownies_days_l2356_235679

variable (d : Nat)

theorem tina_husband_brownies_days : 
  (exists (d : Nat), 
    let total_brownies := 24
    let tina_daily := 2
    let husband_daily := 1
    let total_daily := tina_daily + husband_daily
    let shared_with_guests := 4
    let remaining_brownies := total_brownies - shared_with_guests
    let final_leftover := 5
    let brownies_eaten := remaining_brownies - final_leftover
    brownies_eaten = d * total_daily) → d = 5 := 
by
  sorry

end tina_husband_brownies_days_l2356_235679


namespace min_expr_l2356_235692

theorem min_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = 1) :
  ∃ s : ℝ, (s = a + b) ∧ (s ≥ 2) ∧ (a^2 + b^2 + 4/(s^2) = 3) :=
by sorry

end min_expr_l2356_235692


namespace max_c_friendly_value_l2356_235625

def is_c_friendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → |f x - f y| ≤ c * |x - y|

theorem max_c_friendly_value (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  c > 1 → is_c_friendly c f → |f x - f y| ≤ (c + 1) / 2 :=
sorry

end max_c_friendly_value_l2356_235625


namespace find_number_l2356_235648

theorem find_number (n x : ℕ) (h1 : n * (x - 1) = 21) (h2 : x = 4) : n = 7 :=
by
  sorry

end find_number_l2356_235648


namespace scaling_matrix_unique_l2356_235600

variable {α : Type*} [AddCommGroup α] [Module ℝ α]

noncomputable def matrix_N : Matrix (Fin 4) (Fin 4) ℝ := ![![3, 0, 0, 0], ![0, 3, 0, 0], ![0, 0, 3, 0], ![0, 0, 0, 3]]

theorem scaling_matrix_unique (N : Matrix (Fin 4) (Fin 4) ℝ) :
  (∀ (w : Fin 4 → ℝ), N.mulVec w = 3 • w) → N = matrix_N :=
by
  intros h
  sorry

end scaling_matrix_unique_l2356_235600


namespace alice_sales_goal_l2356_235696

def price_adidas := 45
def price_nike := 60
def price_reeboks := 35
def price_puma := 50
def price_converse := 40

def num_adidas := 10
def num_nike := 12
def num_reeboks := 15
def num_puma := 8
def num_converse := 14

def quota := 2000

def total_sales :=
  (num_adidas * price_adidas) +
  (num_nike * price_nike) +
  (num_reeboks * price_reeboks) +
  (num_puma * price_puma) +
  (num_converse * price_converse)

def exceed_amount := total_sales - quota

theorem alice_sales_goal : exceed_amount = 655 := by
  -- calculation steps would go here
  sorry

end alice_sales_goal_l2356_235696


namespace remainder_of_polynomial_l2356_235699

-- Define the polynomial and the divisor
def f (x : ℝ) := x^3 - 4 * x + 6
def a := -3

-- State the theorem
theorem remainder_of_polynomial :
  f a = -9 := by
  sorry

end remainder_of_polynomial_l2356_235699


namespace area_of_triangle_AEB_l2356_235623

noncomputable def rectangle_area_AEB : ℝ :=
  let AB := 8
  let BC := 4
  let DF := 2
  let GC := 2
  let FG := 8 - DF - GC -- DC (8 units) minus DF and GC.
  let ratio := AB / FG
  let altitude_AEB := BC * ratio
  let area_AEB := 0.5 * AB * altitude_AEB
  area_AEB

theorem area_of_triangle_AEB : rectangle_area_AEB = 32 :=
by
  -- placeholder for detailed proof
  sorry

end area_of_triangle_AEB_l2356_235623


namespace star_value_l2356_235612

def star (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem star_value : star 3 2 = 125 :=
by
  sorry

end star_value_l2356_235612


namespace difference_of_values_l2356_235681

theorem difference_of_values (num : Nat) : 
  (num = 96348621) →
  let face_value := 8
  let local_value := 8 * 10000
  local_value - face_value = 79992 := 
by
  intros h_eq
  have face_value := 8
  have local_value := 8 * 10000
  sorry

end difference_of_values_l2356_235681


namespace distinct_ordered_pairs_count_l2356_235670

theorem distinct_ordered_pairs_count :
  ∃ (n : ℕ), n = 29 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b → a + b = 30 → ∃! p : ℕ × ℕ, p = (a, b)) :=
sorry

end distinct_ordered_pairs_count_l2356_235670


namespace circle_circumference_l2356_235682

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l2356_235682


namespace mr_bhaskar_tour_duration_l2356_235607

theorem mr_bhaskar_tour_duration :
  ∃ d : Nat, 
    (d > 0) ∧ 
    (∃ original_daily_expense new_daily_expense : ℕ,
      original_daily_expense = 360 / d ∧
      new_daily_expense = original_daily_expense - 3 ∧
      360 = new_daily_expense * (d + 4)) ∧
      d = 20 :=
by
  use 20
  -- Here would come the proof steps to verify the conditions and reach the conclusion.
  sorry

end mr_bhaskar_tour_duration_l2356_235607


namespace positive_difference_complementary_angles_l2356_235615

theorem positive_difference_complementary_angles (a b : ℝ) 
  (h1 : a + b = 90) 
  (h2 : 3 * b = a) :
  |a - b| = 45 :=
by
  sorry

end positive_difference_complementary_angles_l2356_235615


namespace domain_of_log_base_half_l2356_235652

noncomputable def domain_log_base_half : Set ℝ := { x : ℝ | x > 5 }

theorem domain_of_log_base_half :
  (∀ x : ℝ, x > 5 ↔ x - 5 > 0) →
  (domain_log_base_half = { x : ℝ | x - 5 > 0 }) :=
by
  sorry

end domain_of_log_base_half_l2356_235652


namespace percent_decrease_correct_l2356_235654

def original_price_per_pack : ℚ := 7 / 3
def promotional_price_per_pack : ℚ := 8 / 4
def percent_decrease_in_price (old_price new_price : ℚ) : ℚ := 
  ((old_price - new_price) / old_price) * 100

theorem percent_decrease_correct :
  percent_decrease_in_price original_price_per_pack promotional_price_per_pack = 14 := by
  sorry

end percent_decrease_correct_l2356_235654


namespace binom_np_p_div_p4_l2356_235619

theorem binom_np_p_div_p4 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (h3 : 3 < p) (hn : n % p = 1) : p^4 ∣ Nat.choose (n * p) p - n := 
sorry

end binom_np_p_div_p4_l2356_235619


namespace simplify_fraction_l2356_235667

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1/y ≠ 0) (h2 : y + 1/x ≠ 0) : 
  (x + 1/y) / (y + 1/x) = x / y :=
sorry

end simplify_fraction_l2356_235667


namespace remainder_of_7_9_power_2008_mod_64_l2356_235683

theorem remainder_of_7_9_power_2008_mod_64 :
  (7^2008 + 9^2008) % 64 = 2 := 
sorry

end remainder_of_7_9_power_2008_mod_64_l2356_235683


namespace center_of_image_circle_l2356_235643

def point := ℝ × ℝ

def reflect_about_y_eq_neg_x (p : point) : point :=
  let (a, b) := p
  (-b, -a)

theorem center_of_image_circle :
  reflect_about_y_eq_neg_x (8, -3) = (3, -8) :=
by
  sorry

end center_of_image_circle_l2356_235643


namespace probability_king_then_ten_l2356_235618

-- Define the conditions
def standard_deck_size : ℕ := 52
def num_kings : ℕ := 4
def num_tens : ℕ := 4

-- Define the event probabilities
def prob_first_card_king : ℚ := num_kings / standard_deck_size
def prob_second_card_ten (remaining_deck_size : ℕ) : ℚ := num_tens / remaining_deck_size

-- The theorem statement to be proved
theorem probability_king_then_ten : 
  prob_first_card_king * prob_second_card_ten (standard_deck_size - 1) = 4 / 663 :=
by
  sorry

end probability_king_then_ten_l2356_235618


namespace prove_moles_of_C2H6_l2356_235635

def moles_of_CCl4 := 4
def moles_of_Cl2 := 14
def moles_of_C2H6 := 2

theorem prove_moles_of_C2H6
  (h1 : moles_of_Cl2 = 14)
  (h2 : moles_of_CCl4 = 4)
  : moles_of_C2H6 = 2 := 
sorry

end prove_moles_of_C2H6_l2356_235635


namespace landmark_distance_l2356_235680

theorem landmark_distance (d : ℝ) : 
  (d >= 7 → d < 7) ∨ (d <= 8 → d > 8) ∨ (d <= 10 → d > 10) → d > 10 :=
by
  sorry

end landmark_distance_l2356_235680


namespace total_acorns_l2356_235677

theorem total_acorns (x y : ℝ) :
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  x + sheila_acorns + danny_acorns = 11.6 * x + y :=
by
  sorry

end total_acorns_l2356_235677


namespace find_integer_n_l2356_235604

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n) % 151 = 93 :=
by
  sorry

end find_integer_n_l2356_235604


namespace missing_number_l2356_235668

theorem missing_number (x : ℝ) : (306 / x) * 15 + 270 = 405 ↔ x = 34 := 
by
  sorry

end missing_number_l2356_235668


namespace area_of_YZW_l2356_235620

-- Definitions from conditions
def area_of_triangle_XYZ := 36
def base_XY := 8
def base_YW := 32

-- The theorem to prove
theorem area_of_YZW : 1/2 * base_YW * (2 * area_of_triangle_XYZ / base_XY) = 144 := 
by
  -- Placeholder for the proof  
  sorry

end area_of_YZW_l2356_235620


namespace find_m_l2356_235608

theorem find_m {m : ℝ} :
  (∃ x y : ℝ, y = x + 1 ∧ y = -x ∧ y = mx + 3) → m = 5 :=
by
  sorry

end find_m_l2356_235608


namespace union_dues_proof_l2356_235659

noncomputable def h : ℕ := 42
noncomputable def r : ℕ := 10
noncomputable def tax_rate : ℝ := 0.20
noncomputable def insurance_rate : ℝ := 0.05
noncomputable def take_home_pay : ℝ := 310

noncomputable def gross_earnings : ℝ := h * r
noncomputable def tax_deduction : ℝ := tax_rate * gross_earnings
noncomputable def insurance_deduction : ℝ := insurance_rate * gross_earnings
noncomputable def total_deductions : ℝ := tax_deduction + insurance_deduction
noncomputable def net_earnings_before_union_dues : ℝ := gross_earnings - total_deductions
noncomputable def union_dues_deduction : ℝ := net_earnings_before_union_dues - take_home_pay

theorem union_dues_proof : union_dues_deduction = 5 := 
by sorry

end union_dues_proof_l2356_235659


namespace part1_part2_l2356_235634

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 2) - abs (x + 1)

theorem part1 (x : ℝ) : f x ≤ 3 ↔ -2/3 ≤ x ∧ x ≤ 6 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, f x ≤ abs (x + 1) + a^2) ↔ a ≤ -2 ∨ 2 ≤ a :=
by
  sorry

end part1_part2_l2356_235634


namespace quad_form_unique_solution_l2356_235661

theorem quad_form_unique_solution (d e f : ℤ) (h1 : d * d = 16) (h2 : 2 * d * e = -40) (h3 : e * e + f = -56) : d * e = -20 :=
by sorry

end quad_form_unique_solution_l2356_235661


namespace evaluate_expression_l2356_235621

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + 1) = 107 :=
by
  -- The proof will go here.
  sorry

end evaluate_expression_l2356_235621


namespace jonathans_and_sisters_total_letters_l2356_235614

theorem jonathans_and_sisters_total_letters:
  (jonathan_first: Nat) = 8 ∧
  (jonathan_surname: Nat) = 10 ∧
  (sister_first: Nat) = 5 ∧
  (sister_surname: Nat) = 10 →
  jonathan_first + jonathan_surname + sister_first + sister_surname = 33 := by
  intros
  sorry

end jonathans_and_sisters_total_letters_l2356_235614


namespace locus_midpoint_l2356_235649

-- Conditions
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1

def perpendicular_rays (OA OB : ℝ × ℝ) : Prop := (OA.1 * OB.1 + OA.2 * OB.2) = 0 -- Dot product zero for perpendicularity

-- Given the hyperbola and perpendicularity conditions, prove the locus equation
theorem locus_midpoint (x y : ℝ) :
  (∃ A B : ℝ × ℝ, hyperbola_eq A.1 A.2 ∧ hyperbola_eq B.1 B.2 ∧ perpendicular_rays A B ∧
  x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → 3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2) :=
sorry

end locus_midpoint_l2356_235649


namespace product_of_roots_l2356_235671

theorem product_of_roots :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) ∧ (x₁ ^ 2 + 2 * x₁ - 4 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 4 = 0) := by
  sorry

end product_of_roots_l2356_235671


namespace g_ln_1_over_2017_l2356_235662

theorem g_ln_1_over_2017 (a : ℝ) (h_a_pos : 0 < a) (h_a_neq_1 : a ≠ 1) (f g : ℝ → ℝ)
  (h_f_add : ∀ m n : ℝ, f (m + n) = f m + f n - 1)
  (h_g : ∀ x : ℝ, g x = f x + a^x / (a^x + 1))
  (h_g_ln_2017 : g (Real.log 2017) = 2018) :
  g (Real.log (1 / 2017)) = -2015 :=
sorry

end g_ln_1_over_2017_l2356_235662


namespace triangular_number_30_sum_of_first_30_triangular_numbers_l2356_235678

theorem triangular_number_30 
  (T : ℕ → ℕ)
  (hT : ∀ n : ℕ, T n = n * (n + 1) / 2) : 
  T 30 = 465 :=
by
  -- Skipping proof with sorry
  sorry

theorem sum_of_first_30_triangular_numbers 
  (S : ℕ → ℕ)
  (hS : ∀ n : ℕ, S n = n * (n + 1) * (n + 2) / 6) : 
  S 30 = 4960 :=
by
  -- Skipping proof with sorry
  sorry

end triangular_number_30_sum_of_first_30_triangular_numbers_l2356_235678


namespace product_not_perfect_power_l2356_235697

theorem product_not_perfect_power (n : ℕ) : ¬∃ (k : ℕ) (a : ℤ), k > 1 ∧ n * (n + 1) = a^k := by
  sorry

end product_not_perfect_power_l2356_235697


namespace compound_interest_comparison_l2356_235695

theorem compound_interest_comparison :
  (1 + 0.04) < (1 + 0.04 / 12) ^ 12 := sorry

end compound_interest_comparison_l2356_235695


namespace total_tank_capacity_l2356_235605

-- Definitions based on conditions
def initial_condition (w c : ℝ) : Prop := w / c = 1 / 3
def after_adding_five (w c : ℝ) : Prop := (w + 5) / c = 1 / 2

-- The problem statement
theorem total_tank_capacity (w c : ℝ) (h1 : initial_condition w c) (h2 : after_adding_five w c) : c = 30 :=
sorry

end total_tank_capacity_l2356_235605


namespace fourth_power_of_cube_third_smallest_prime_l2356_235637

-- Define the third smallest prime number
def third_smallest_prime : Nat := 5

-- Define a function that calculates the fourth power of a number
def fourth_power (x : Nat) : Nat := x * x * x * x

-- Define a function that calculates the cube of a number
def cube (x : Nat) : Nat := x * x * x

-- The proposition stating the fourth power of the cube of the third smallest prime number is 244140625
theorem fourth_power_of_cube_third_smallest_prime : 
  fourth_power (cube third_smallest_prime) = 244140625 :=
by
  -- skip the proof
  sorry

end fourth_power_of_cube_third_smallest_prime_l2356_235637


namespace households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l2356_235691

namespace VehicleHouseholds

-- Definitions for the conditions
def totalHouseholds : ℕ := 250
def householdsNoVehicles : ℕ := 25
def householdsAllVehicles : ℕ := 36
def householdsCarOnly : ℕ := 62
def householdsBikeOnly : ℕ := 45
def householdsScooterOnly : ℕ := 30

-- Proof Statements
theorem households_with_two_types_of_vehicles :
  (totalHouseholds - householdsNoVehicles - householdsAllVehicles - 
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly)) = 52 := by
  sorry

theorem households_with_exactly_one_type_of_vehicle :
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly) = 137 := by
  sorry

theorem households_with_at_least_one_type_of_vehicle :
  (totalHouseholds - householdsNoVehicles) = 225 := by
  sorry

end VehicleHouseholds

end households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l2356_235691


namespace algebraic_expression_value_l2356_235640

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x + 3 = 7) : 3 * x^2 + 3 * x + 7 = 19 :=
sorry

end algebraic_expression_value_l2356_235640


namespace even_increasing_ordering_l2356_235653

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove
theorem even_increasing_ordering (h_even : is_even_function f) (h_increasing : is_increasing_on_pos f) : 
  f 1 < f (-2) ∧ f (-2) < f 3 :=
by
  sorry

end even_increasing_ordering_l2356_235653


namespace math_problem_l2356_235603

theorem math_problem : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end math_problem_l2356_235603


namespace mod_equiv_solution_l2356_235633

theorem mod_equiv_solution (a b : ℤ) (n : ℤ) 
  (h₁ : a ≡ 22 [ZMOD 50])
  (h₂ : b ≡ 78 [ZMOD 50])
  (h₃ : 150 ≤ n ∧ n ≤ 201)
  (h₄ : n = 194) :
  a - b ≡ n [ZMOD 50] :=
by
  sorry

end mod_equiv_solution_l2356_235633


namespace mother_to_grandfather_age_ratio_l2356_235660

theorem mother_to_grandfather_age_ratio
  (rachel_age : ℕ)
  (grandfather_ratio : ℕ)
  (father_mother_gap : ℕ) 
  (future_rachel_age: ℕ) 
  (future_father_age : ℕ)
  (current_father_age current_mother_age current_grandfather_age : ℕ) 
  (h1 : rachel_age = 12)
  (h2 : grandfather_ratio = 7)
  (h3 : father_mother_gap = 5)
  (h4 : future_rachel_age = 25)
  (h5 : future_father_age = 60)
  (h6 : current_father_age = future_father_age - (future_rachel_age - rachel_age))
  (h7 : current_mother_age = current_father_age - father_mother_gap)
  (h8 : current_grandfather_age = grandfather_ratio * rachel_age) :
  current_mother_age = current_grandfather_age / 2 :=
by
  sorry

end mother_to_grandfather_age_ratio_l2356_235660


namespace maximize_x_minus_y_plus_z_l2356_235628

-- Define the given condition as a predicate
def given_condition (x y z : ℝ) : Prop :=
  2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5

-- Define the statement we want to prove
theorem maximize_x_minus_y_plus_z :
  ∃ x y z : ℝ, given_condition x y z ∧ (x - y + z = 4) :=
by
  sorry

end maximize_x_minus_y_plus_z_l2356_235628


namespace smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l2356_235617

theorem smallest_number_after_operations_n_111 :
  ∀ (n : ℕ), n = 111 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 111 →
       (f l) = 0)) :=
by 
  sorry

theorem smallest_number_after_operations_n_110 :
  ∀ (n : ℕ), n = 110 → 
  (∃ (f : List ℕ → ℕ), -- The function f represents the sequence of operations
     (∀ (l : List ℕ), l = List.range 110 →
       (f l) = 1)) :=
by 
  sorry

end smallest_number_after_operations_n_111_smallest_number_after_operations_n_110_l2356_235617


namespace missing_score_and_variance_l2356_235622

theorem missing_score_and_variance (score_A score_B score_D score_E : ℕ) (avg_score : ℕ)
  (h_scores : score_A = 81 ∧ score_B = 79 ∧ score_D = 80 ∧ score_E = 82)
  (h_avg : avg_score = 80):
  ∃ (score_C variance : ℕ), score_C = 78 ∧ variance = 2 := by
  sorry

end missing_score_and_variance_l2356_235622


namespace speed_of_current_l2356_235686

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end speed_of_current_l2356_235686


namespace Julia_played_kids_on_Monday_l2356_235638

theorem Julia_played_kids_on_Monday
  (t : ℕ) (w : ℕ) (h1 : t = 18) (h2 : w = 97) (h3 : t + m = 33) :
  ∃ m : ℕ, m = 15 :=
by
  sorry

end Julia_played_kids_on_Monday_l2356_235638


namespace Rachel_spent_on_lunch_fraction_l2356_235602

variable {MoneyEarned MoneySpentOnDVD MoneyLeft MoneySpentOnLunch : ℝ}

-- Given conditions
axiom Rachel_earnings : MoneyEarned = 200
axiom Rachel_spent_on_DVD : MoneySpentOnDVD = MoneyEarned / 2
axiom Rachel_leftover : MoneyLeft = 50
axiom Rachel_total_spent : MoneyEarned - MoneyLeft = MoneySpentOnLunch + MoneySpentOnDVD

-- Prove that Rachel spent 1/4 of her money on lunch
theorem Rachel_spent_on_lunch_fraction :
  MoneySpentOnLunch / MoneyEarned = 1 / 4 :=
sorry

end Rachel_spent_on_lunch_fraction_l2356_235602
