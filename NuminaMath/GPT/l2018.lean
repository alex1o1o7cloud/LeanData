import Mathlib

namespace NUMINAMATH_GPT_sam_bought_nine_books_l2018_201890

-- Definitions based on the conditions
def initial_money : ℕ := 79
def cost_per_book : ℕ := 7
def money_left : ℕ := 16

-- The amount spent on books
def money_spent_on_books : ℕ := initial_money - money_left

-- The number of books bought
def number_of_books (spent : ℕ) (cost : ℕ) : ℕ := spent / cost

-- Let x be the number of books bought and prove x = 9
theorem sam_bought_nine_books : number_of_books money_spent_on_books cost_per_book = 9 :=
by
  sorry

end NUMINAMATH_GPT_sam_bought_nine_books_l2018_201890


namespace NUMINAMATH_GPT_total_students_l2018_201826

theorem total_students (x : ℕ) (h1 : (x + 6) / (2*x + 6) = 2 / 3) : 2 * x + 6 = 18 :=
sorry

end NUMINAMATH_GPT_total_students_l2018_201826


namespace NUMINAMATH_GPT_work_ratio_l2018_201845

theorem work_ratio (M B : ℝ) 
  (h1 : 5 * (12 * M + 16 * B) = 1)
  (h2 : 4 * (13 * M + 24 * B) = 1) : 
  M / B = 2 := 
  sorry

end NUMINAMATH_GPT_work_ratio_l2018_201845


namespace NUMINAMATH_GPT_sum_of_abc_is_40_l2018_201876

theorem sum_of_abc_is_40 (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * b + c = 55) (h2 : b * c + a = 55) (h3 : c * a + b = 55) :
    a + b + c = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abc_is_40_l2018_201876


namespace NUMINAMATH_GPT_prime_solution_l2018_201868

theorem prime_solution (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 :=
sorry

end NUMINAMATH_GPT_prime_solution_l2018_201868


namespace NUMINAMATH_GPT_polynomial_expansion_l2018_201818

variable (x : ℝ)

theorem polynomial_expansion :
  (7*x^2 + 3)*(5*x^3 + 4*x + 1) = 35*x^5 + 43*x^3 + 7*x^2 + 12*x + 3 := by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l2018_201818


namespace NUMINAMATH_GPT_prisoners_can_be_freed_l2018_201873

-- Condition: We have 100 prisoners and 100 drawers.
def prisoners : Nat := 100
def drawers : Nat := 100

-- Predicate to represent the strategy
def successful_strategy (strategy: (Fin prisoners) → (Fin drawers) → Bool) : Bool :=
  -- We use a hypothetical strategy function to model this
  (true) -- Placeholder for the actual strategy computation

-- Statement: Prove that there exists a strategy where all prisoners finding their names has a probability greater than 30%.
theorem prisoners_can_be_freed :
  ∃ strategy: (Fin prisoners) → (Fin drawers) → Bool, 
    (successful_strategy strategy) ∧ (0.3118 > 0.3) :=
sorry

end NUMINAMATH_GPT_prisoners_can_be_freed_l2018_201873


namespace NUMINAMATH_GPT_missing_angle_correct_l2018_201881

theorem missing_angle_correct (n : ℕ) (h1 : n ≥ 3) (angles_sum : ℕ) (h2 : angles_sum = 2017) 
    (sum_interior_angles : ℕ) (h3 : sum_interior_angles = 180 * (n - 2)) :
    (sum_interior_angles - angles_sum) = 143 :=
by
  sorry

end NUMINAMATH_GPT_missing_angle_correct_l2018_201881


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l2018_201808

-- To express that a real number x is in the interval (0, 2)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 2

theorem solve_quadratic_inequality :
  { x : ℝ | x^2 < 2 * x } = { x : ℝ | in_interval x } :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l2018_201808


namespace NUMINAMATH_GPT_Freddie_ratio_l2018_201850

noncomputable def Veronica_distance : ℕ := 1000

noncomputable def Freddie_distance (F : ℕ) : Prop :=
  1000 + 12000 = 5 * F - 2000

theorem Freddie_ratio (F : ℕ) (h : Freddie_distance F) :
  F / Veronica_distance = 3 := by
  sorry

end NUMINAMATH_GPT_Freddie_ratio_l2018_201850


namespace NUMINAMATH_GPT_joe_lowest_score_dropped_l2018_201863

theorem joe_lowest_score_dropped (A B C D : ℕ) 
  (h1 : A + B + C + D = 160)
  (h2 : A + B + C = 135) 
  (h3 : D ≤ A ∧ D ≤ B ∧ D ≤ C) :
  D = 25 :=
sorry

end NUMINAMATH_GPT_joe_lowest_score_dropped_l2018_201863


namespace NUMINAMATH_GPT_total_selling_price_l2018_201853

theorem total_selling_price (cost_per_meter profit_per_meter : ℕ) (total_meters : ℕ) :
  cost_per_meter = 90 → 
  profit_per_meter = 15 → 
  total_meters = 85 → 
  (cost_per_meter + profit_per_meter) * total_meters = 8925 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_selling_price_l2018_201853


namespace NUMINAMATH_GPT_game_cost_l2018_201871

theorem game_cost
    (total_earnings : ℕ)
    (expenses : ℕ)
    (games_bought : ℕ)
    (remaining_money := total_earnings - expenses)
    (cost_per_game := remaining_money / games_bought)
    (h1 : total_earnings = 104)
    (h2 : expenses = 41)
    (h3 : games_bought = 7) :
    cost_per_game = 9 := by
  sorry

end NUMINAMATH_GPT_game_cost_l2018_201871


namespace NUMINAMATH_GPT_minimum_distance_from_lattice_point_to_line_l2018_201809

theorem minimum_distance_from_lattice_point_to_line :
  let distance (x y : ℤ) := |25 * x - 15 * y + 12| / (5 * Real.sqrt 34)
  ∃ (x y : ℤ), distance x y = Real.sqrt 34 / 85 :=
sorry

end NUMINAMATH_GPT_minimum_distance_from_lattice_point_to_line_l2018_201809


namespace NUMINAMATH_GPT_females_with_advanced_degrees_l2018_201896

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (total_advanced_degrees : ℕ) 
  (total_college_degrees : ℕ) 
  (males_with_college_degree : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : total_advanced_degrees = 90) 
  (h4 : total_college_degrees = 90) 
  (h5 : males_with_college_degree = 35) : 
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 55 := 
by {
  sorry
}

end NUMINAMATH_GPT_females_with_advanced_degrees_l2018_201896


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2018_201869

-- Definition of the parabola C1: y^2 = 2px with p > 0.
def parabola (p : ℝ) (p_pos : 0 < p) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Definition of the hyperbola C2: x^2 / a^2 - y^2 / b^2 = 1 with a > 0 and b > 0.
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

-- Definition of having a common focus F at (p / 2, 0).
def common_focus (p a b c : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) : Prop := 
  c = p / 2 ∧ c^2 = a^2 + b^2

-- Definition for points A and B on parabola C1 and point M on hyperbola C2.
def points_A_B_M (c a b : ℝ) (x1 y1 x2 y2 yM : ℝ) : Prop := 
  x1 = c ∧ y1 = 2 * c ∧ x2 = c ∧ y2 = -2 * c ∧ yM = b^2 / a

-- Condition for OM, OA, and OB relation and mn = 1/8.
def OM_OA_OB_relation (m n : ℝ) : Prop := 
  m * n = 1 / 8

-- Theorem statement: Given the conditions, the eccentricity of hyperbola C2 is √6 + √2 / 2.
theorem hyperbola_eccentricity (p a b c m n : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) :
  parabola p p_pos c (2 * c) → 
  hyperbola a b a_pos b_pos c (b^2 / a) → 
  common_focus p a b c p_pos a_pos b_pos →
  points_A_B_M c a b c (2 * c) c (-2 * c) (b^2 / a) →
  OM_OA_OB_relation m n → 
  m * n = 1 / 8 →
  ∃ e : ℝ, e = (Real.sqrt 6 + Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2018_201869


namespace NUMINAMATH_GPT_determinant_zero_l2018_201886

theorem determinant_zero (α β : ℝ) :
  Matrix.det ![
    ![0, Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, Real.sin β],
    ![Real.cos α, -Real.sin β, 0]
  ] = 0 :=
by sorry

end NUMINAMATH_GPT_determinant_zero_l2018_201886


namespace NUMINAMATH_GPT_min_x_squared_plus_y_squared_l2018_201827

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  x^2 + y^2 ≥ 50 :=
by
  sorry

end NUMINAMATH_GPT_min_x_squared_plus_y_squared_l2018_201827


namespace NUMINAMATH_GPT_Anil_profit_in_rupees_l2018_201835

def cost_scooter (C : ℝ) : Prop := 0.10 * C = 500
def profit (C P : ℝ) : Prop := P = 0.20 * C

theorem Anil_profit_in_rupees (C P : ℝ) (h1 : cost_scooter C) (h2 : profit C P) : P = 1000 :=
by
  sorry

end NUMINAMATH_GPT_Anil_profit_in_rupees_l2018_201835


namespace NUMINAMATH_GPT_speed_of_stream_l2018_201825

theorem speed_of_stream (b s : ℝ) (h1 : 75 = 5 * (b + s)) (h2 : 45 = 5 * (b - s)) : s = 3 :=
by
  have eq1 : b + s = 15 := by linarith [h1]
  have eq2 : b - s = 9 := by linarith [h2]
  have b_val : b = 12 := by linarith [eq1, eq2]
  linarith 

end NUMINAMATH_GPT_speed_of_stream_l2018_201825


namespace NUMINAMATH_GPT_total_selling_amount_l2018_201899

-- Defining the given conditions
def total_metres_of_cloth := 200
def loss_per_metre := 6
def cost_price_per_metre := 66

-- Theorem statement to prove the total selling amount
theorem total_selling_amount : 
    (cost_price_per_metre - loss_per_metre) * total_metres_of_cloth = 12000 := 
by 
    sorry

end NUMINAMATH_GPT_total_selling_amount_l2018_201899


namespace NUMINAMATH_GPT_keith_total_cost_correct_l2018_201855

noncomputable def total_cost_keith_purchases : Real :=
  let discount_toy := 6.51
  let price_toy := discount_toy / 0.90
  let pet_food := 5.79
  let cage_price := 12.51
  let tax_rate := 0.08
  let cage_tax := cage_price * tax_rate
  let price_with_tax := cage_price + cage_tax
  let water_bottle := 4.99
  let bedding := 7.65
  let discovered_money := 1.0
  let total_cost := discount_toy + pet_food + price_with_tax + water_bottle + bedding
  total_cost - discovered_money

theorem keith_total_cost_correct :
  total_cost_keith_purchases = 37.454 :=
by
  sorry -- Proof of the theorem will go here

end NUMINAMATH_GPT_keith_total_cost_correct_l2018_201855


namespace NUMINAMATH_GPT_sum_first_two_integers_l2018_201819

/-- Prove that the sum of the first two integers n > 1 such that 3^n is divisible by n 
and 3^n - 1 is divisible by n - 1 is equal to 30. -/
theorem sum_first_two_integers (n : ℕ) (h1 : n > 1) (h2 : 3 ^ n % n = 0) (h3 : (3 ^ n - 1) % (n - 1) = 0) : 
  n = 3 ∨ n = 27 → n + 3 + 27 = 30 :=
sorry

end NUMINAMATH_GPT_sum_first_two_integers_l2018_201819


namespace NUMINAMATH_GPT_find_socks_cost_l2018_201865

variable (S : ℝ)
variable (socks_cost : ℝ := 9.5)
variable (shoe_cost : ℝ := 92)
variable (jack_has : ℝ := 40)
variable (needs_more : ℝ := 71)
variable (total_funds : ℝ := jack_has + needs_more)

theorem find_socks_cost (h : 2 * S + shoe_cost = total_funds) : S = socks_cost :=
by 
  sorry

end NUMINAMATH_GPT_find_socks_cost_l2018_201865


namespace NUMINAMATH_GPT_minimize_expression_10_l2018_201828

theorem minimize_expression_10 (n : ℕ) (h : 0 < n) : 
  (∃ m : ℕ, 0 < m ∧ (∀ k : ℕ, 0 < k → (n = k) → (n = 10))) :=
by
  sorry

end NUMINAMATH_GPT_minimize_expression_10_l2018_201828


namespace NUMINAMATH_GPT_weight_of_daughter_l2018_201811

def mother_daughter_grandchild_weight (M D C : ℝ) :=
  M + D + C = 130 ∧
  D + C = 60 ∧
  C = 1/5 * M

theorem weight_of_daughter (M D C : ℝ) 
  (h : mother_daughter_grandchild_weight M D C) : D = 46 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end NUMINAMATH_GPT_weight_of_daughter_l2018_201811


namespace NUMINAMATH_GPT_walking_speed_l2018_201824

theorem walking_speed (W : ℝ) : (1 / (1 / W + 1 / 8)) * 6 = 2.25 * (12 / 2) -> W = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_walking_speed_l2018_201824


namespace NUMINAMATH_GPT_max_subjects_per_teacher_l2018_201878

theorem max_subjects_per_teacher
  (math_teachers : ℕ := 7)
  (physics_teachers : ℕ := 6)
  (chemistry_teachers : ℕ := 5)
  (min_teachers_required : ℕ := 6)
  (total_subjects : ℕ := 18) :
  ∀ (x : ℕ), x ≥ 3 ↔ 6 * x ≥ total_subjects := by
  sorry

end NUMINAMATH_GPT_max_subjects_per_teacher_l2018_201878


namespace NUMINAMATH_GPT_journey_time_l2018_201815

theorem journey_time 
  (d1 d2 T : ℝ)
  (h1 : d1 / 30 + (150 - d1) / 10 = T)
  (h2 : d1 / 30 + d2 / 30 + (150 - (d1 - d2)) / 30 = T)
  (h3 : (d1 - d2) / 10 + (150 - (d1 - d2)) / 30 = T) :
  T = 5 := 
sorry

end NUMINAMATH_GPT_journey_time_l2018_201815


namespace NUMINAMATH_GPT_m_equals_p_of_odd_prime_and_integers_l2018_201812

theorem m_equals_p_of_odd_prime_and_integers (p m : ℕ) (x y : ℕ) (hp : p > 1 ∧ ¬ (p % 2 = 0)) 
    (hx : x > 1) (hy : y > 1) 
    (h : (x ^ p + y ^ p) / 2 = ((x + y) / 2) ^ m): 
    m = p := 
by 
  sorry

end NUMINAMATH_GPT_m_equals_p_of_odd_prime_and_integers_l2018_201812


namespace NUMINAMATH_GPT_probability_of_same_color_is_correct_l2018_201889

-- Definitions from the problem conditions
def red_marbles := 6
def white_marbles := 7
def blue_marbles := 8
def total_marbles := red_marbles + white_marbles + blue_marbles -- 21

-- Calculate the probability of drawing 4 red marbles
def P_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 white marbles
def P_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 blue marbles
def P_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles of the same color
def P_all_same_color := P_all_red + P_all_white + P_all_blue

-- Proof that the total probability is equal to the given correct answer
theorem probability_of_same_color_is_correct : P_all_same_color = 240 / 11970 := by
  sorry

end NUMINAMATH_GPT_probability_of_same_color_is_correct_l2018_201889


namespace NUMINAMATH_GPT_adult_ticket_cost_given_conditions_l2018_201849

variables (C A S : ℕ)

def cost_relationships : Prop :=
  A = C + 10 ∧ S = A - 5 ∧ (5 * C + 2 * A + 2 * S + (S - 3) = 212)

theorem adult_ticket_cost_given_conditions :
  cost_relationships C A S → A = 28 :=
by
  intros h
  have h1 : A = C + 10 := h.left
  have h2 : S = A - 5 := h.right.left
  have h3 : (5 * C + 2 * A + 2 * S + (S - 3) = 212) := h.right.right
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_given_conditions_l2018_201849


namespace NUMINAMATH_GPT_boys_down_slide_l2018_201821

theorem boys_down_slide (boys_1 boys_2 : ℕ) (h : boys_1 = 22) (h' : boys_2 = 13) : boys_1 + boys_2 = 35 := by
  sorry

end NUMINAMATH_GPT_boys_down_slide_l2018_201821


namespace NUMINAMATH_GPT_part_I_part_II_l2018_201806

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part_I (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≥ (1 : ℝ) / Real.exp 1 :=
sorry

theorem part_II (a x1 x2 x : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) (hx : x1 < x ∧ x < x2) :
  (f x a - f x1 a) / (x - x1) < (f x a - f x2 a) / (x - x2) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2018_201806


namespace NUMINAMATH_GPT_positive_number_property_l2018_201834

-- Define the problem conditions and the goal
theorem positive_number_property (y : ℝ) (hy : y > 0) (h : y^2 / 100 = 9) : y = 30 := by
  sorry

end NUMINAMATH_GPT_positive_number_property_l2018_201834


namespace NUMINAMATH_GPT_inheritance_amount_l2018_201894

def federalTaxRate : ℝ := 0.25
def stateTaxRate : ℝ := 0.15
def totalTaxPaid : ℝ := 16500

theorem inheritance_amount :
  ∃ x : ℝ, (federalTaxRate * x + stateTaxRate * (1 - federalTaxRate) * x = totalTaxPaid) → x = 45500 := by
  sorry

end NUMINAMATH_GPT_inheritance_amount_l2018_201894


namespace NUMINAMATH_GPT_smallest_solution_is_39_over_8_l2018_201844

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_is_39_over_8_l2018_201844


namespace NUMINAMATH_GPT_coordinate_difference_l2018_201892

theorem coordinate_difference (m n : ℝ) (h : m = 4 * n + 5) :
  (4 * (n + 0.5) + 5) - m = 2 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_coordinate_difference_l2018_201892


namespace NUMINAMATH_GPT_train_speed_equivalent_l2018_201856

def length_train1 : ℝ := 180
def length_train2 : ℝ := 160
def speed_train1 : ℝ := 60 
def crossing_time_sec : ℝ := 12.239020878329734

noncomputable def speed_train2 (length1 length2 speed1 time : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hr := time / 3600
  let relative_speed := total_length_km / time_hr
  relative_speed - speed1

theorem train_speed_equivalent :
  speed_train2 length_train1 length_train2 speed_train1 crossing_time_sec = 40 :=
by
  simp [length_train1, length_train2, speed_train1, crossing_time_sec, speed_train2]
  sorry

end NUMINAMATH_GPT_train_speed_equivalent_l2018_201856


namespace NUMINAMATH_GPT_sum_op_two_triangles_l2018_201882

def op (a b c : ℕ) : ℕ := 2 * a - b + c

theorem sum_op_two_triangles : op 3 7 5 + op 6 2 8 = 22 := by
  sorry

end NUMINAMATH_GPT_sum_op_two_triangles_l2018_201882


namespace NUMINAMATH_GPT_prove_a3_l2018_201829

variable (a1 a2 a3 a4 : ℕ)
variable (q : ℕ)

-- Definition of the geometric sequence
def geom_seq (n : ℕ) : ℕ :=
  a1 * q^(n-1)

-- Given conditions
def cond1 := geom_seq 4 = 8
def cond2 := (geom_seq 2 + geom_seq 3) / (geom_seq 1 + geom_seq 2) = 2

-- Proving the required condition
theorem prove_a3 : cond1 ∧ cond2 → geom_seq 3 = 4 :=
by
sorry

end NUMINAMATH_GPT_prove_a3_l2018_201829


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2018_201879

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, 0 < a n)
  (h3 : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2) : q = 3 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2018_201879


namespace NUMINAMATH_GPT_male_students_count_l2018_201804

variable (M F : ℕ)
variable (average_all average_male average_female : ℕ)
variable (total_male total_female total_all : ℕ)

noncomputable def male_students (M F : ℕ) : ℕ := 8

theorem male_students_count:
  F = 32 -> average_all = 90 -> average_male = 82 -> average_female = 92 ->
  total_male = average_male * M -> total_female = average_female * F -> 
  total_all = average_all * (M + F) -> total_male + total_female = total_all ->
  M = male_students M F := 
by
  intros hF hAvgAll hAvgMale hAvgFemale hTotalMale hTotalFemale hTotalAll hEqTotal
  sorry

end NUMINAMATH_GPT_male_students_count_l2018_201804


namespace NUMINAMATH_GPT_margo_total_distance_l2018_201817

theorem margo_total_distance
  (t1 t2 : ℚ) (rate1 rate2 : ℚ)
  (h1 : t1 = 15 / 60)
  (h2 : t2 = 25 / 60)
  (r1 : rate1 = 5)
  (r2 : rate2 = 3) :
  (t1 * rate1 + t2 * rate2 = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_margo_total_distance_l2018_201817


namespace NUMINAMATH_GPT_find_value_in_box_l2018_201842

theorem find_value_in_box (x : ℕ) :
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * x ↔ x = 50 := by
  sorry

end NUMINAMATH_GPT_find_value_in_box_l2018_201842


namespace NUMINAMATH_GPT_compare_sums_l2018_201833

open Classical

-- Define the necessary sequences and their properties
variable {α : Type*} [LinearOrderedField α]

-- Arithmetic Sequence {a_n}
noncomputable def arith_seq (a_1 d : α) : ℕ → α
| 0     => a_1
| (n+1) => (arith_seq a_1 d n) + d

-- Geometric Sequence {b_n}
noncomputable def geom_seq (b_1 q : α) : ℕ → α
| 0     => b_1
| (n+1) => (geom_seq b_1 q n) * q

-- Sum of the first n terms of an arithmetic sequence
noncomputable def arith_sum (a_1 d : α) (n : ℕ) : α :=
(n + 1) * (a_1 + arith_seq a_1 d n) / 2

-- Sum of the first n terms of a geometric sequence
noncomputable def geom_sum (b_1 q : α) (n : ℕ) : α :=
if q = 1 then (n + 1) * b_1
else b_1 * (1 - q^(n + 1)) / (1 - q)

theorem compare_sums
  (a_1 b_1 : α) (d q : α)
  (hd : d ≠ 0) (hq : q > 0) (hq1 : q ≠ 1)
  (h_eq1 : a_1 = b_1)
  (h_eq2 : arith_seq a_1 d 1011 = geom_seq b_1 q 1011) :
  arith_sum a_1 d 2022 < geom_sum b_1 q 2022 :=
sorry

end NUMINAMATH_GPT_compare_sums_l2018_201833


namespace NUMINAMATH_GPT_train_length_l2018_201870

theorem train_length (speed_kmph : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmph = 60 →
  time_s = 3 →
  length_m = 50.01 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l2018_201870


namespace NUMINAMATH_GPT_sequences_equal_l2018_201813

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2018 / (n + 1)) * a (n + 1) + a n

noncomputable def b : ℕ → ℚ
| 0 => 0
| 1 => 1
| (n+2) => (2020 / (n + 1)) * b (n + 1) + b n

theorem sequences_equal :
  (a 1010) / 1010 = (b 1009) / 1009 :=
sorry

end NUMINAMATH_GPT_sequences_equal_l2018_201813


namespace NUMINAMATH_GPT_sampled_students_within_interval_l2018_201820

/-- Define the conditions for the student's problem --/
def student_count : ℕ := 1221
def sampled_students : ℕ := 37
def sampling_interval : ℕ := student_count / sampled_students
def interval_lower_bound : ℕ := 496
def interval_upper_bound : ℕ := 825
def interval_range : ℕ := interval_upper_bound - interval_lower_bound + 1

/-- State the goal within the above conditions --/
theorem sampled_students_within_interval :
  interval_range / sampling_interval = 10 :=
sorry

end NUMINAMATH_GPT_sampled_students_within_interval_l2018_201820


namespace NUMINAMATH_GPT_sum_of_sides_l2018_201877

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosB cosC : ℝ)
variable (sinB : ℝ)
variable (area : ℝ)

-- Given conditions
axiom h1 : b = 2
axiom h2 : b * cosC + c * cosB = 3 * a * cosB
axiom h3 : area = 3 * Real.sqrt 2 / 2
axiom h4 : sinB = Real.sqrt (1 - cosB ^ 2)

-- Prove the desired result
theorem sum_of_sides (A B C a b c cosB cosC sinB : ℝ) (area : ℝ)
  (h1 : b = 2)
  (h2 : b * cosC + c * cosB = 3 * a * cosB)
  (h3 : area = 3 * Real.sqrt 2 / 2)
  (h4 : sinB = Real.sqrt (1 - cosB ^ 2)) :
  a + c = 4 := 
sorry

end NUMINAMATH_GPT_sum_of_sides_l2018_201877


namespace NUMINAMATH_GPT_solve_for_x_l2018_201888

theorem solve_for_x (x y : ℝ) (h₁ : y = 1 / (4 * x + 2)) (h₂ : y = 1 / 2) : x = 0 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_solve_for_x_l2018_201888


namespace NUMINAMATH_GPT_amoeba_population_after_5_days_l2018_201857

theorem amoeba_population_after_5_days 
  (initial : ℕ)
  (split_factor : ℕ)
  (days : ℕ)
  (h_initial : initial = 2)
  (h_split : split_factor = 3)
  (h_days : days = 5) :
  (initial * split_factor ^ days) = 486 :=
by sorry

end NUMINAMATH_GPT_amoeba_population_after_5_days_l2018_201857


namespace NUMINAMATH_GPT_product_consecutive_natural_not_equal_even_l2018_201866

theorem product_consecutive_natural_not_equal_even (n m : ℕ) (h : m % 2 = 0 ∧ m > 0) : n * (n + 1) ≠ m * (m + 2) :=
sorry

end NUMINAMATH_GPT_product_consecutive_natural_not_equal_even_l2018_201866


namespace NUMINAMATH_GPT_find_d_l2018_201891

-- Define the polynomial g(x)
def g (d : ℚ) (x : ℚ) : ℚ := d * x^4 + 11 * x^3 + 5 * d * x^2 - 28 * x + 72

-- The main proof statement
theorem find_d (hd : g d 4 = 0) : d = -83 / 42 := by
  sorry -- proof not needed as per prompt

end NUMINAMATH_GPT_find_d_l2018_201891


namespace NUMINAMATH_GPT_min_balls_to_guarantee_18_l2018_201836

noncomputable def min_balls_needed {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) : ℕ :=
  95

theorem min_balls_to_guarantee_18 {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) :
  min_balls_needed h_red h_green h_yellow h_blue h_white h_black = 95 :=
  by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_min_balls_to_guarantee_18_l2018_201836


namespace NUMINAMATH_GPT_domain_f_l2018_201874

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x-1) / Real.log 2) + 1

theorem domain_f : domain f = {x | 1 < x} :=
by {
  sorry
}

end NUMINAMATH_GPT_domain_f_l2018_201874


namespace NUMINAMATH_GPT_proof_problem_l2018_201884

noncomputable def p (a : ℝ) : Prop :=
∀ x : ℝ, x^2 + a * x + a^2 ≥ 0

noncomputable def q : Prop :=
∃ x₀ : ℕ, 0 < x₀ ∧ 2 * x₀^2 - 1 ≤ 0

theorem proof_problem (a : ℝ) (hp : p a) (hq : q) : p a ∨ q :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2018_201884


namespace NUMINAMATH_GPT_ellipse_focal_distance_l2018_201802

theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 9 = 9) → (∃ c : ℝ, c = 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focal_distance_l2018_201802


namespace NUMINAMATH_GPT_max_total_toads_l2018_201885

variable (x y : Nat)
variable (frogs total_frogs : Nat)
variable (total_toads : Nat)

def pond1_frogs := 3 * x
def pond1_toads := 4 * x
def pond2_frogs := 5 * y
def pond2_toads := 6 * y

def all_frogs := pond1_frogs x + pond2_frogs y
def all_toads := pond1_toads x + pond2_toads y

theorem max_total_toads (h_frogs : all_frogs x y = 36) : all_toads x y = 46 := 
sorry

end NUMINAMATH_GPT_max_total_toads_l2018_201885


namespace NUMINAMATH_GPT_carpool_commute_distance_l2018_201807

theorem carpool_commute_distance :
  (∀ (D : ℕ),
    4 * 5 * ((2 * D : ℝ) / 30) * 2.50 = 5 * 14 →
    D = 21) :=
by
  intro D
  intro h
  sorry

end NUMINAMATH_GPT_carpool_commute_distance_l2018_201807


namespace NUMINAMATH_GPT_butter_remaining_correct_l2018_201838

-- Definitions of the conditions
def cupsOfBakingMix : ℕ := 6
def butterPerCup : ℕ := 2
def substituteRatio : ℕ := 1
def coconutOilUsed : ℕ := 8

-- Calculation based on the conditions
def butterNeeded : ℕ := butterPerCup * cupsOfBakingMix
def butterReplaced : ℕ := coconutOilUsed * substituteRatio
def butterRemaining : ℕ := butterNeeded - butterReplaced

-- The theorem to prove the chef has 4 ounces of butter remaining
theorem butter_remaining_correct : butterRemaining = 4 := 
by
  -- Note: We insert 'sorry' since the proof itself is not required.
  sorry

end NUMINAMATH_GPT_butter_remaining_correct_l2018_201838


namespace NUMINAMATH_GPT_gcd_765432_654321_l2018_201880

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end NUMINAMATH_GPT_gcd_765432_654321_l2018_201880


namespace NUMINAMATH_GPT_find_original_price_l2018_201875

-- Definitions based on the conditions
def original_price_increased (x : ℝ) : ℝ := 1.25 * x
def loan_payment (total_cost : ℝ) : ℝ := 0.75 * total_cost
def own_funds (total_cost : ℝ) : ℝ := 0.25 * total_cost

-- Condition values
def new_home_cost : ℝ := 500000
def loan_amount := loan_payment new_home_cost
def funds_paid := own_funds new_home_cost

-- Proof statement
theorem find_original_price : 
  ∃ x : ℝ, original_price_increased x = funds_paid ↔ x = 100000 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_find_original_price_l2018_201875


namespace NUMINAMATH_GPT_journey_time_l2018_201895

noncomputable def velocity_of_stream : ℝ := 4
noncomputable def speed_of_boat_in_still_water : ℝ := 14
noncomputable def distance_A_to_B : ℝ := 180
noncomputable def distance_B_to_C : ℝ := distance_A_to_B / 2
noncomputable def downstream_speed : ℝ := speed_of_boat_in_still_water + velocity_of_stream
noncomputable def upstream_speed : ℝ := speed_of_boat_in_still_water - velocity_of_stream

theorem journey_time : (distance_A_to_B / downstream_speed) + (distance_B_to_C / upstream_speed) = 19 := by
  sorry

end NUMINAMATH_GPT_journey_time_l2018_201895


namespace NUMINAMATH_GPT_ratio_of_width_to_length_l2018_201851

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_width_to_length_l2018_201851


namespace NUMINAMATH_GPT_value_of_a_l2018_201822

theorem value_of_a (a : ℝ) (h : (a, 0) ∈ {p : ℝ × ℝ | p.2 = p.1 + 8}) : a = -8 :=
sorry

end NUMINAMATH_GPT_value_of_a_l2018_201822


namespace NUMINAMATH_GPT_zero_interval_of_f_l2018_201839

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_interval_of_f :
    ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_interval_of_f_l2018_201839


namespace NUMINAMATH_GPT_quadratic_range_l2018_201893

open Real

theorem quadratic_range (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 4 * x1 - 2 = 0 ∧ a * x2^2 + 4 * x2 - 2 = 0) : 
  a > -2 ∧ a ≠ 0 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_range_l2018_201893


namespace NUMINAMATH_GPT_cos_sum_is_one_or_cos_2a_l2018_201800

open Real

theorem cos_sum_is_one_or_cos_2a (a b : ℝ) (h : ∫ x in a..b, sin x = 0) : cos (a + b) = 1 ∨ cos (a + b) = cos (2 * a) :=
  sorry

end NUMINAMATH_GPT_cos_sum_is_one_or_cos_2a_l2018_201800


namespace NUMINAMATH_GPT_find_monthly_income_l2018_201859

-- Define the percentages spent on various categories
def household_items_percentage : ℝ := 0.35
def clothing_percentage : ℝ := 0.18
def medicines_percentage : ℝ := 0.06
def entertainment_percentage : ℝ := 0.11
def transportation_percentage : ℝ := 0.12
def mutual_fund_percentage : ℝ := 0.05
def taxes_percentage : ℝ := 0.07

-- Define the savings amount
def savings_amount : ℝ := 12500

-- Total spent percentage
def total_spent_percentage := household_items_percentage + clothing_percentage + medicines_percentage + entertainment_percentage + transportation_percentage + mutual_fund_percentage + taxes_percentage

-- Percentage saved
def savings_percentage := 1 - total_spent_percentage

-- Prove that Ajay's monthly income is Rs. 208,333.33
theorem find_monthly_income (I : ℝ) (h : I * savings_percentage = savings_amount) : I = 208333.33 := by
  sorry

end NUMINAMATH_GPT_find_monthly_income_l2018_201859


namespace NUMINAMATH_GPT_total_playtime_l2018_201837

-- Conditions
def lena_playtime_hours : ℝ := 3.5
def minutes_per_hour : ℝ := 60
def lena_playtime_minutes : ℝ := lena_playtime_hours * minutes_per_hour
def brother_playtime_extra_minutes : ℝ := 17
def brother_playtime_minutes : ℝ := lena_playtime_minutes + brother_playtime_extra_minutes

-- Proof problem
theorem total_playtime : lena_playtime_minutes + brother_playtime_minutes = 437 := by
  sorry

end NUMINAMATH_GPT_total_playtime_l2018_201837


namespace NUMINAMATH_GPT_exists_m_divisible_by_2k_l2018_201847

theorem exists_m_divisible_by_2k {k : ℕ} (h_k : 0 < k) {a : ℤ} (h_a : a % 8 = 3) :
  ∃ m : ℕ, 0 < m ∧ 2^k ∣ (a^m + a + 2) :=
sorry

end NUMINAMATH_GPT_exists_m_divisible_by_2k_l2018_201847


namespace NUMINAMATH_GPT_log_expression_simplifies_to_zero_l2018_201854

theorem log_expression_simplifies_to_zero : 
  (1/2 : ℝ) * (Real.log 4) + Real.log 5 - Real.exp (0 * Real.log (Real.pi + 1)) = 0 := 
by
  sorry

end NUMINAMATH_GPT_log_expression_simplifies_to_zero_l2018_201854


namespace NUMINAMATH_GPT_basketball_game_proof_l2018_201810

-- Definition of the conditions
def num_teams (x : ℕ) : Prop := ∃ n : ℕ, n = x

def games_played (x : ℕ) (total_games : ℕ) : Prop := total_games = 28

def game_combinations (x : ℕ) : ℕ := (x * (x - 1)) / 2

-- Proof statement using the conditions
theorem basketball_game_proof (x : ℕ) (h1 : num_teams x) (h2 : games_played x 28) : 
  game_combinations x = 28 := by
  sorry

end NUMINAMATH_GPT_basketball_game_proof_l2018_201810


namespace NUMINAMATH_GPT_motel_percentage_reduction_l2018_201898

theorem motel_percentage_reduction
  (x y : ℕ) 
  (h : 40 * x + 60 * y = 1000) :
  ((1000 - (40 * (x + 10) + 60 * (y - 10))) / 1000) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_motel_percentage_reduction_l2018_201898


namespace NUMINAMATH_GPT_avg_of_first_three_groups_prob_of_inspection_l2018_201883
  
-- Define the given frequency distribution as constants
def freq_40_50 : ℝ := 0.04
def freq_50_60 : ℝ := 0.06
def freq_60_70 : ℝ := 0.22
def freq_70_80 : ℝ := 0.28
def freq_80_90 : ℝ := 0.22
def freq_90_100 : ℝ := 0.18

-- Calculate the midpoint values for the first three groups
def mid_40_50 : ℝ := 45
def mid_50_60 : ℝ := 55
def mid_60_70 : ℝ := 65

-- Define the probabilities interpreted from the distributions
def prob_poor : ℝ := freq_40_50 + freq_50_60
def prob_avg : ℝ := freq_60_70 + freq_70_80
def prob_good : ℝ := freq_80_90 + freq_90_100

-- Define the main theorem for the average score of the first three groups
theorem avg_of_first_three_groups :
  (mid_40_50 * freq_40_50 + mid_50_60 * freq_50_60 + mid_60_70 * freq_60_70) /
  (freq_40_50 + freq_50_60 + freq_60_70) = 60.625 := 
by { sorry }

-- Define the theorem for the probability of inspection
theorem prob_of_inspection :
  1 - (3 * (prob_good * prob_avg * prob_avg) + 3 * (prob_avg * prob_avg * prob_good) + (prob_good * prob_good * prob_good)) = 0.396 :=
by { sorry }

end NUMINAMATH_GPT_avg_of_first_three_groups_prob_of_inspection_l2018_201883


namespace NUMINAMATH_GPT_vincent_books_cost_l2018_201872

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end NUMINAMATH_GPT_vincent_books_cost_l2018_201872


namespace NUMINAMATH_GPT_gum_pieces_per_package_l2018_201897

theorem gum_pieces_per_package :
  (∀ (packages pieces each_package : ℕ), packages = 9 ∧ pieces = 135 → each_package = pieces / packages → each_package = 15) := 
by
  intros packages pieces each_package
  sorry

end NUMINAMATH_GPT_gum_pieces_per_package_l2018_201897


namespace NUMINAMATH_GPT_correct_quotient_of_original_division_operation_l2018_201805

theorem correct_quotient_of_original_division_operation 
  (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 102)
  (h2 : correct_divisor = 201)
  (h3 : incorrect_quotient = 753)
  (h4 : ∃ k, k = incorrect_quotient * 3) :
  ∃ q, q = 1146 ∧ (correct_divisor * q = incorrect_divisor * (incorrect_quotient * 3)) :=
by
  sorry

end NUMINAMATH_GPT_correct_quotient_of_original_division_operation_l2018_201805


namespace NUMINAMATH_GPT_apple_and_cherry_pies_total_l2018_201852

-- Given conditions state that:
def apple_pies : ℕ := 6
def cherry_pies : ℕ := 5

-- We aim to prove that the total number of apple and cherry pies is 11.
theorem apple_and_cherry_pies_total : apple_pies + cherry_pies = 11 := by
  sorry

end NUMINAMATH_GPT_apple_and_cherry_pies_total_l2018_201852


namespace NUMINAMATH_GPT_sum_of_powers_of_i_l2018_201848

open Complex

def i := Complex.I

theorem sum_of_powers_of_i : (i + i^2 + i^3 + i^4) = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_i_l2018_201848


namespace NUMINAMATH_GPT_group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l2018_201840

-- Question 1
theorem group_photo_arrangements {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ arrangements : ℕ, arrangements = 14400 := 
sorry

-- Question 2
theorem grouping_methods {N : ℕ} (hN : N = 8) :
  ∃ methods : ℕ, methods = 2520 := 
sorry

-- Question 3
theorem selection_methods_with_at_least_one_male {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ methods : ℕ, methods = 1560 := 
sorry

end NUMINAMATH_GPT_group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l2018_201840


namespace NUMINAMATH_GPT_original_population_has_factor_three_l2018_201803

theorem original_population_has_factor_three (x y z : ℕ) 
  (hx : ∃ n : ℕ, x = n ^ 2) -- original population is a perfect square
  (h1 : x + 150 = y^2 - 1)  -- after increase of 150, population is one less than a perfect square
  (h2 : y^2 - 1 + 150 = z^2) -- after another increase of 150, population is a perfect square again
  : 3 ∣ x :=
sorry

end NUMINAMATH_GPT_original_population_has_factor_three_l2018_201803


namespace NUMINAMATH_GPT_andrena_has_more_dolls_than_debelyn_l2018_201860

-- Define the initial number of dolls
def initial_dolls_Debelyn : ℕ := 20
def initial_dolls_Christel : ℕ := 24

-- Define the number of dolls given to Andrena
def dolls_given_by_Debelyn : ℕ := 2
def dolls_given_by_Christel : ℕ := 5

-- Define the condition that Andrena has 2 more dolls than Christel after receiving the dolls
def andrena_more_than_christel : ℕ := 2

-- Define the dolls count after gift exchange
def dolls_Debelyn_after : ℕ := initial_dolls_Debelyn - dolls_given_by_Debelyn
def dolls_Christel_after : ℕ := initial_dolls_Christel - dolls_given_by_Christel
def dolls_Andrena_after : ℕ := dolls_Christel_after + andrena_more_than_christel

-- Define the proof problem
theorem andrena_has_more_dolls_than_debelyn : dolls_Andrena_after - dolls_Debelyn_after = 3 := by
  sorry

end NUMINAMATH_GPT_andrena_has_more_dolls_than_debelyn_l2018_201860


namespace NUMINAMATH_GPT_discount_percentage_for_two_pairs_of_jeans_l2018_201830

theorem discount_percentage_for_two_pairs_of_jeans
  (price_per_pair : ℕ := 40)
  (price_for_three_pairs : ℕ := 112)
  (discount : ℕ := 8)
  (original_price_for_two_pairs : ℕ := price_per_pair * 2)
  (discount_percentage : ℕ := (discount * 100) / original_price_for_two_pairs) :
  discount_percentage = 10 := 
by
  sorry

end NUMINAMATH_GPT_discount_percentage_for_two_pairs_of_jeans_l2018_201830


namespace NUMINAMATH_GPT_min_value_l2018_201831

theorem min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 3 * y + 3 * x * y = 6) : 2 * x + 3 * y ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_l2018_201831


namespace NUMINAMATH_GPT_correct_option_D_l2018_201823

theorem correct_option_D (x y : ℝ) : (x - y) ^ 2 = (y - x) ^ 2 := by
  sorry

end NUMINAMATH_GPT_correct_option_D_l2018_201823


namespace NUMINAMATH_GPT_quadratic_one_solution_l2018_201867

theorem quadratic_one_solution (p : ℝ) : (3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0) 
  → ((-6) ^ 2 - 4 * 3 * p = 0) 
  → p = 3 :=
by
  intro h1 h2
  have h1' : 3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0 := h1
  have h2' : (-6) ^ 2 - 4 * 3 * p = 0 := h2
  sorry

end NUMINAMATH_GPT_quadratic_one_solution_l2018_201867


namespace NUMINAMATH_GPT_David_marks_in_Chemistry_l2018_201862

theorem David_marks_in_Chemistry (e m p b avg c : ℕ) 
  (h1 : e = 91) 
  (h2 : m = 65) 
  (h3 : p = 82) 
  (h4 : b = 85) 
  (h5 : avg = 78) 
  (h6 : avg * 5 = e + m + p + b + c) :
  c = 67 := 
sorry

end NUMINAMATH_GPT_David_marks_in_Chemistry_l2018_201862


namespace NUMINAMATH_GPT_no_prime_p_satisfies_l2018_201801

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_prime_p_satisfies (p : ℕ) (hp : Nat.Prime p) (hp1 : is_perfect_square (7 * p + 3 ^ p - 4)) : False :=
by
  sorry

end NUMINAMATH_GPT_no_prime_p_satisfies_l2018_201801


namespace NUMINAMATH_GPT_mary_total_cards_l2018_201887

def mary_initial_cards := 33
def torn_cards := 6
def cards_given_by_sam := 23

theorem mary_total_cards : mary_initial_cards - torn_cards + cards_given_by_sam = 50 :=
  by
    sorry

end NUMINAMATH_GPT_mary_total_cards_l2018_201887


namespace NUMINAMATH_GPT_circle_center_radius_l2018_201814

theorem circle_center_radius
    (x y : ℝ)
    (eq_circle : (x - 2)^2 + y^2 = 4) :
    (2, 0) = (2, 0) ∧ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l2018_201814


namespace NUMINAMATH_GPT_jackson_holidays_l2018_201843

theorem jackson_holidays (holidays_per_month : ℕ) (months_per_year : ℕ) (total_holidays : ℕ) :
  holidays_per_month = 3 → months_per_year = 12 → total_holidays = holidays_per_month * months_per_year →
  total_holidays = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jackson_holidays_l2018_201843


namespace NUMINAMATH_GPT_parabola_c_value_l2018_201858

theorem parabola_c_value (b c : ℝ)
  (h1 : 3 = 2^2 + b * 2 + c)
  (h2 : 6 = 5^2 + b * 5 + c) :
  c = -13 :=
by
  -- Proof would follow here
  sorry

end NUMINAMATH_GPT_parabola_c_value_l2018_201858


namespace NUMINAMATH_GPT_tile_ratio_l2018_201861

-- Definitions corresponding to the conditions in the problem
def orig_grid_size : ℕ := 6
def orig_black_tiles : ℕ := 12
def orig_white_tiles : ℕ := 24
def border_size : ℕ := 1

-- The combined problem statement
theorem tile_ratio (orig_grid_size orig_black_tiles orig_white_tiles border_size : ℕ) :
  let new_grid_size := orig_grid_size + 2 * border_size
  let new_tiles := new_grid_size^2
  let added_tiles := new_tiles - orig_grid_size^2
  let total_white_tiles := orig_white_tiles + added_tiles
  let black_to_white_ratio := orig_black_tiles / total_white_tiles
  black_to_white_ratio = (3 : ℕ) / 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_tile_ratio_l2018_201861


namespace NUMINAMATH_GPT_quadratic_inequality_l2018_201816

theorem quadratic_inequality (c : ℝ) (h₁ : 0 < c) (h₂ : c < 16): ∃ x : ℝ, x^2 - 8 * x + c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l2018_201816


namespace NUMINAMATH_GPT_socks_selection_l2018_201832

theorem socks_selection :
  ∀ (R Y G B O : ℕ), 
    R = 80 → Y = 70 → G = 50 → B = 60 → O = 40 →
    (∃ k, k = 38 ∧ ∀ (N : ℕ → ℕ), (N R + N Y + N G + N B + N O ≥ k)
          → (exists (pairs : ℕ), pairs ≥ 15 ∧ pairs = (N R / 2) + (N Y / 2) + (N G / 2) + (N B / 2) + (N O / 2) )) :=
by
  sorry

end NUMINAMATH_GPT_socks_selection_l2018_201832


namespace NUMINAMATH_GPT_sum_of_faces_of_rectangular_prism_l2018_201864

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_faces_of_rectangular_prism_l2018_201864


namespace NUMINAMATH_GPT_set_intersection_l2018_201846

   -- Define set A
   def A : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≥ 0 }
   
   -- Define set B
   def B : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}

   -- Define the relative complement of A in the real numbers
   def complement_R (A : Set ℝ) : Set ℝ := {x : ℝ | ¬ (A x)}

   -- The main statement that needs to be proven
   theorem set_intersection :
     (complement_R A) ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
     sorry
   
end NUMINAMATH_GPT_set_intersection_l2018_201846


namespace NUMINAMATH_GPT_sum_of_variables_l2018_201841

theorem sum_of_variables (a b c d : ℤ)
  (h1 : a - b + 2 * c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_variables_l2018_201841
