import Mathlib

namespace NUMINAMATH_GPT_rational_solution_l1481_148144

theorem rational_solution (a b c : ℚ) 
  (h : (3 * a - 2 * b + c - 4)^2 + (a + 2 * b - 3 * c + 6)^2 + (2 * a - b + 2 * c - 2)^2 ≤ 0) : 
  2 * a + b - 4 * c = -4 := 
by
  sorry

end NUMINAMATH_GPT_rational_solution_l1481_148144


namespace NUMINAMATH_GPT_perimeter_triangle_formed_by_parallel_lines_l1481_148165

-- Defining the side lengths of the triangle ABC
def AB := 150
def BC := 270
def AC := 210

-- Defining the lengths of the segments formed by intersections with lines parallel to the sides of ABC
def length_lA := 65
def length_lB := 60
def length_lC := 20

-- The perimeter of the triangle formed by the intersection of the lines
theorem perimeter_triangle_formed_by_parallel_lines :
  let perimeter : ℝ := 5.71 + 20 + 83.33 + 65 + 91 + 60 + 5.71
  perimeter = 330.75 := by
  sorry

end NUMINAMATH_GPT_perimeter_triangle_formed_by_parallel_lines_l1481_148165


namespace NUMINAMATH_GPT_f_inequality_l1481_148136

variables {n1 n2 d : ℕ} (f : ℕ → ℕ → ℕ)

theorem f_inequality (hn1 : n1 > 0) (hn2 : n2 > 0) (hd : d > 0) :
  f (n1 * n2) d ≤ f n1 d + n1 * (f n2 d - 1) :=
sorry

end NUMINAMATH_GPT_f_inequality_l1481_148136


namespace NUMINAMATH_GPT_algebra_books_cannot_be_determined_uniquely_l1481_148161

theorem algebra_books_cannot_be_determined_uniquely (A H S M E : ℕ) (pos_A : A > 0) (pos_H : H > 0) (pos_S : S > 0) 
  (pos_M : M > 0) (pos_E : E > 0) (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ S ≠ M ∧ S ≠ E ∧ M ≠ E) 
  (cond1: S < A) (cond2: M > H) (cond3: A + 2 * H = S + 2 * M) : 
  E = 0 :=
sorry

end NUMINAMATH_GPT_algebra_books_cannot_be_determined_uniquely_l1481_148161


namespace NUMINAMATH_GPT_average_cost_is_2_l1481_148173

noncomputable def total_amount_spent (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℕ :=
  apples_quantity * apples_cost + bananas_quantity * bananas_cost + oranges_quantity * oranges_cost

noncomputable def total_number_of_fruits (apples_quantity bananas_quantity oranges_quantity : ℕ) : ℕ :=
  apples_quantity + bananas_quantity + oranges_quantity

noncomputable def average_cost (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℚ :=
  (total_amount_spent apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℚ) /
  (total_number_of_fruits apples_quantity bananas_quantity oranges_quantity : ℚ)

theorem average_cost_is_2 :
  average_cost 12 4 4 2 1 3 = 2 := 
by
  sorry

end NUMINAMATH_GPT_average_cost_is_2_l1481_148173


namespace NUMINAMATH_GPT_quadratic_properties_l1481_148177

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_properties_l1481_148177


namespace NUMINAMATH_GPT_revenue_for_recent_quarter_l1481_148185

noncomputable def previous_year_revenue : ℝ := 85.0
noncomputable def percentage_fall : ℝ := 43.529411764705884
noncomputable def recent_quarter_revenue : ℝ := previous_year_revenue - (previous_year_revenue * (percentage_fall / 100))

theorem revenue_for_recent_quarter : recent_quarter_revenue = 48.0 := 
by 
  sorry -- Proof is skipped

end NUMINAMATH_GPT_revenue_for_recent_quarter_l1481_148185


namespace NUMINAMATH_GPT_root_interval_range_l1481_148132

theorem root_interval_range (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^3 - 3*x + m = 0) → (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_root_interval_range_l1481_148132


namespace NUMINAMATH_GPT_men_per_table_l1481_148129

theorem men_per_table (total_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) (total_women : ℕ)
    (h1 : total_tables = 9)
    (h2 : women_per_table = 7)
    (h3 : total_customers = 90)
    (h4 : total_women = women_per_table * total_tables)
    (h5 : total_women + total_men = total_customers) :
  total_men / total_tables = 3 :=
by
  have total_women := 7 * 9
  have total_men := 90 - total_women
  exact sorry

end NUMINAMATH_GPT_men_per_table_l1481_148129


namespace NUMINAMATH_GPT_Kenny_jumping_jacks_wednesday_l1481_148157

variable (Sunday Monday Tuesday Wednesday Thursday Friday Saturday : ℕ)
variable (LastWeekTotal : ℕ := 324)
variable (SundayJumpingJacks : ℕ := 34)
variable (MondayJumpingJacks : ℕ := 20)
variable (TuesdayJumpingJacks : ℕ := 0)
variable (SomeDayJumpingJacks : ℕ := 64)
variable (FridayJumpingJacks : ℕ := 23)
variable (SaturdayJumpingJacks : ℕ := 61)

def Kenny_jumping_jacks_this_week (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ) : ℕ :=
  SundayJumpingJacks + MondayJumpingJacks + TuesdayJumpingJacks + WednesdayJumpingJacks + ThursdayJumpingJacks + FridayJumpingJacks + SaturdayJumpingJacks

def Kenny_jumping_jacks_to_beat (weekTotal : ℕ) : ℕ :=
  LastWeekTotal + 1

theorem Kenny_jumping_jacks_wednesday : 
  ∃ (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ), 
  Kenny_jumping_jacks_this_week WednesdayJumpingJacks ThursdayJumpingJacks = LastWeekTotal + 1 ∧ 
  (WednesdayJumpingJacks = 59 ∧ ThursdayJumpingJacks = 64) ∨ (WednesdayJumpingJacks = 64 ∧ ThursdayJumpingJacks = 59) :=
by
  sorry

end NUMINAMATH_GPT_Kenny_jumping_jacks_wednesday_l1481_148157


namespace NUMINAMATH_GPT_find_a_8_l1481_148169

noncomputable def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∃ b : ℕ → ℤ, (∀ n : ℕ, 0 < n → b n = a (n + 1) - a n) ∧
  b 3 = -2 ∧ b 10 = 12

theorem find_a_8 (a : ℕ → ℤ) (h : sequence_a a) : a 8 = 3 :=
sorry

end NUMINAMATH_GPT_find_a_8_l1481_148169


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l1481_148156

-- Define the sides of the triangle
def side1 : ℕ := 7
def side2 : ℕ := 23

-- State the conditions using the triangle inequality theorem
def triangle_inequality_satisfied (s : ℕ) : Prop :=
  (side1 + side2 > s) ∧ (side1 + s > side2) ∧ (side2 + s > side1)

-- The proof statement
theorem smallest_whole_number_larger_than_perimeter
  (s : ℕ) (h : triangle_inequality_satisfied s) : 
  ∃ n : ℕ, n = 60 ∧ ∀ p : ℕ, (p > side1 + side2 + s) → (p ≥ n) :=
sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_perimeter_l1481_148156


namespace NUMINAMATH_GPT_find_original_denominator_l1481_148114

variable (d : ℕ)

theorem find_original_denominator
  (h1 : ∀ n : ℕ, n = 3)
  (h2 : 3 + 7 = 10)
  (h3 : (10 : ℕ) = 1 * (d + 7) / 3) :
  d = 23 := by
  sorry

end NUMINAMATH_GPT_find_original_denominator_l1481_148114


namespace NUMINAMATH_GPT_marked_price_correct_l1481_148181

noncomputable def marked_price (original_price discount_percent purchase_price profit_percent final_price_percent : ℝ) := 
  (purchase_price * (1 + profit_percent)) / final_price_percent

theorem marked_price_correct
  (original_price : ℝ)
  (discount_percent : ℝ)
  (profit_percent : ℝ)
  (final_price_percent : ℝ)
  (purchase_price : ℝ := original_price * (1 - discount_percent))
  (expected_marked_price : ℝ) :
  original_price = 40 →
  discount_percent = 0.15 →
  profit_percent = 0.25 →
  final_price_percent = 0.90 →
  expected_marked_price = 47.20 →
  marked_price original_price discount_percent purchase_price profit_percent final_price_percent = expected_marked_price := 
by
  intros
  sorry

end NUMINAMATH_GPT_marked_price_correct_l1481_148181


namespace NUMINAMATH_GPT_kira_travel_time_l1481_148146

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end NUMINAMATH_GPT_kira_travel_time_l1481_148146


namespace NUMINAMATH_GPT_exists_rhombus_with_given_side_and_diag_sum_l1481_148175

-- Define the context of the problem
variables (a s : ℝ)

-- Necessary definitions for a rhombus
structure Rhombus (side diag_sum : ℝ) :=
  (side_length : ℝ)
  (diag_sum : ℝ)
  (d1 d2 : ℝ)
  (side_length_eq : side_length = side)
  (diag_sum_eq : d1 + d2 = diag_sum)
  (a_squared : 2 * (side_length)^2 = d1^2 + d2^2)

-- The proof problem
theorem exists_rhombus_with_given_side_and_diag_sum (a s : ℝ) : 
  ∃ (r : Rhombus a (2*s)), r.side_length = a ∧ r.diag_sum = 2 * s :=
by
  sorry

end NUMINAMATH_GPT_exists_rhombus_with_given_side_and_diag_sum_l1481_148175


namespace NUMINAMATH_GPT_no_always_1x3_rectangle_l1481_148137

/-- From a sheet of graph paper measuring 8 x 8 cells, 12 rectangles of size 1 x 2 were cut out along the grid lines. 
Prove that it is not necessarily possible to always find a 1 x 3 checkered rectangle in the remaining part. -/
theorem no_always_1x3_rectangle (grid_size : ℕ) (rectangles_removed : ℕ) (rect_size : ℕ) :
  grid_size = 64 → rectangles_removed * rect_size = 24 → ¬ (∀ remaining_cells, remaining_cells ≥ 0 → remaining_cells ≤ 64 → ∃ (x y : ℕ), remaining_cells = x * y ∧ x = 1 ∧ y = 3) :=
  by
  intro h1 h2 h3
  /- Exact proof omitted for brevity -/
  sorry

end NUMINAMATH_GPT_no_always_1x3_rectangle_l1481_148137


namespace NUMINAMATH_GPT_Jane_remaining_time_l1481_148100

noncomputable def JaneRate : ℚ := 1 / 4
noncomputable def RoyRate : ℚ := 1 / 5
noncomputable def workingTime : ℚ := 2
noncomputable def cakeFractionCompletedTogether : ℚ := (JaneRate + RoyRate) * workingTime
noncomputable def remainingCakeFraction : ℚ := 1 - cakeFractionCompletedTogether
noncomputable def timeForJaneToCompleteRemainingCake : ℚ := remainingCakeFraction / JaneRate

theorem Jane_remaining_time :
  timeForJaneToCompleteRemainingCake = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_Jane_remaining_time_l1481_148100


namespace NUMINAMATH_GPT_remainder_of_expression_mod7_l1481_148135

theorem remainder_of_expression_mod7 :
  (7^6 + 8^7 + 9^8) % 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_expression_mod7_l1481_148135


namespace NUMINAMATH_GPT_perimeter_square_C_l1481_148140

theorem perimeter_square_C (pA pB pC : ℕ) (hA : pA = 16) (hB : pB = 32) (hC : pC = (pA + pB) / 2) : pC = 24 := by
  sorry

end NUMINAMATH_GPT_perimeter_square_C_l1481_148140


namespace NUMINAMATH_GPT_rectangle_area_l1481_148195

theorem rectangle_area {H W : ℝ} (h_height : H = 24) (ratio : W / H = 0.875) :
  H * W = 504 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l1481_148195


namespace NUMINAMATH_GPT_speed_of_faster_train_approx_l1481_148153

noncomputable def speed_of_slower_train_kmph : ℝ := 40
noncomputable def speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * 1000 / 3600
noncomputable def distance_train1 : ℝ := 250
noncomputable def distance_train2 : ℝ := 500
noncomputable def total_distance : ℝ := distance_train1 + distance_train2
noncomputable def crossing_time : ℝ := 26.99784017278618
noncomputable def relative_speed_train_crossing : ℝ := total_distance / crossing_time
noncomputable def speed_of_faster_train_mps : ℝ := relative_speed_train_crossing - speed_of_slower_train_mps
noncomputable def speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * 3600 / 1000

theorem speed_of_faster_train_approx : abs (speed_of_faster_train_kmph - 60.0152) < 0.001 :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_faster_train_approx_l1481_148153


namespace NUMINAMATH_GPT_ap_minus_aq_eq_8_l1481_148170

theorem ap_minus_aq_eq_8 (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) (p q : ℕ) 
  (h1 : ∀ n, S_n n = n^2 - 5 * n) 
  (h2 : ∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) 
  (h3 : p - q = 4) :
  a_n p - a_n q = 8 := sorry

end NUMINAMATH_GPT_ap_minus_aq_eq_8_l1481_148170


namespace NUMINAMATH_GPT_bicycle_cost_calculation_l1481_148122

theorem bicycle_cost_calculation 
  (CP_A CP_B CP_C : ℝ)
  (h1 : CP_B = 1.20 * CP_A)
  (h2 : CP_C = 1.25 * CP_B)
  (h3 : CP_C = 225) :
  CP_A = 150 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_cost_calculation_l1481_148122


namespace NUMINAMATH_GPT_price_of_second_oil_l1481_148126

open Real

-- Define conditions
def litres_of_first_oil : ℝ := 10
def price_per_litre_first_oil : ℝ := 50
def litres_of_second_oil : ℝ := 5
def total_volume_of_mixture : ℝ := 15
def rate_of_mixture : ℝ := 55.67
def total_cost_of_mixture : ℝ := total_volume_of_mixture * rate_of_mixture

-- Define total cost of the first oil
def total_cost_first_oil : ℝ := litres_of_first_oil * price_per_litre_first_oil

-- Define total cost of the second oil in terms of unknown price P
def total_cost_second_oil (P : ℝ) : ℝ := litres_of_second_oil * P

-- Theorem to prove price per litre of the second oil
theorem price_of_second_oil : ∃ P : ℝ, total_cost_first_oil + (total_cost_second_oil P) = total_cost_of_mixture ∧ P = 67.01 :=
by
  sorry

end NUMINAMATH_GPT_price_of_second_oil_l1481_148126


namespace NUMINAMATH_GPT_compute_expression_l1481_148133

-- The definition and conditions
def is_nonreal_root_of_unity (ω : ℂ) : Prop := ω ^ 3 = 1 ∧ ω ≠ 1

-- The statement
theorem compute_expression (ω : ℂ) (hω : is_nonreal_root_of_unity ω) : 
  (1 - 2 * ω + 2 * ω ^ 2) ^ 6 + (1 + 2 * ω - 2 * ω ^ 2) ^ 6 = 0 :=
sorry

end NUMINAMATH_GPT_compute_expression_l1481_148133


namespace NUMINAMATH_GPT_find_dinner_bill_l1481_148124

noncomputable def total_dinner_bill (B : ℝ) (silas_share : ℝ) (remaining_friends_pay : ℝ) (each_friend_pays : ℝ) :=
  silas_share = (1/2) * B ∧
  remaining_friends_pay = (1/2) * B + 0.10 * B ∧
  each_friend_pays = remaining_friends_pay / 5 ∧
  each_friend_pays = 18

theorem find_dinner_bill : ∃ B : ℝ, total_dinner_bill B ((1/2) * B) ((1/2) * B + 0.10 * B) (18) → B = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_dinner_bill_l1481_148124


namespace NUMINAMATH_GPT_compare_real_numbers_l1481_148127

theorem compare_real_numbers (a b : ℝ) : (a > b) ∨ (a = b) ∨ (a < b) :=
sorry

end NUMINAMATH_GPT_compare_real_numbers_l1481_148127


namespace NUMINAMATH_GPT_derivative_not_in_second_quadrant_l1481_148117

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b * x + c
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x - 4

-- Given condition: Axis of symmetry is x = 2
def axis_of_symmetry (b : ℝ) : Prop := b = -4

-- Additional condition: behavior of the derivative and quadrant check
def not_in_second_quadrant (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f' x < 0

-- The main theorem to be proved
theorem derivative_not_in_second_quadrant (b c : ℝ) (h : axis_of_symmetry b) :
  not_in_second_quadrant f_derivative :=
by {
  sorry
}

end NUMINAMATH_GPT_derivative_not_in_second_quadrant_l1481_148117


namespace NUMINAMATH_GPT_distance_between_parabola_vertices_l1481_148174

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_parabola_vertices :
  distance (0, 3) (0, -1) = 4 := 
by {
  -- Proof omitted here
  sorry
}

end NUMINAMATH_GPT_distance_between_parabola_vertices_l1481_148174


namespace NUMINAMATH_GPT_answer_l1481_148120

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, Real.exp x > 1

theorem answer (hp : p) (hq : ¬ q) : p ∧ ¬ q :=
  by
    exact ⟨hp, hq⟩

end NUMINAMATH_GPT_answer_l1481_148120


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1481_148113

theorem quadratic_inequality_solution_set
  (a b : ℝ)
  (h1 : 2 + 3 = -a)
  (h2 : 2 * 3 = b) :
  ∀ x : ℝ, 6 * x^2 - 5 * x + 1 > 0 ↔ x < (1 / 3) ∨ x > (1 / 2) := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1481_148113


namespace NUMINAMATH_GPT_initial_percentage_increase_l1481_148102

theorem initial_percentage_increase 
  (W R : ℝ) 
  (P : ℝ)
  (h1 : R = W * (1 + P/100)) 
  (h2 : R * 0.70 = W * 1.18999999999999993) :
  P = 70 :=
by sorry

end NUMINAMATH_GPT_initial_percentage_increase_l1481_148102


namespace NUMINAMATH_GPT_exponent_relation_l1481_148152

theorem exponent_relation (a : ℝ) (m n : ℕ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m - n) = 3 := 
sorry

end NUMINAMATH_GPT_exponent_relation_l1481_148152


namespace NUMINAMATH_GPT_most_consistent_player_l1481_148145

section ConsistentPerformance

variables (σA σB σC σD : ℝ)
variables (σA_eq : σA = 0.023)
variables (σB_eq : σB = 0.018)
variables (σC_eq : σC = 0.020)
variables (σD_eq : σD = 0.021)

theorem most_consistent_player : σB < σC ∧ σB < σD ∧ σB < σA :=
by 
  rw [σA_eq, σB_eq, σC_eq, σD_eq]
  sorry

end ConsistentPerformance

end NUMINAMATH_GPT_most_consistent_player_l1481_148145


namespace NUMINAMATH_GPT_lisa_flight_time_l1481_148119

theorem lisa_flight_time
  (distance : ℕ) (speed : ℕ) (time : ℕ)
  (h_distance : distance = 256)
  (h_speed : speed = 32)
  (h_time : time = distance / speed) :
  time = 8 :=
by sorry

end NUMINAMATH_GPT_lisa_flight_time_l1481_148119


namespace NUMINAMATH_GPT_set_complement_union_l1481_148150

-- Definitions of the sets
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

-- The statement to prove
theorem set_complement_union : (U \ A) ∪ (U \ B) = {1, 4, 5} :=
by sorry

end NUMINAMATH_GPT_set_complement_union_l1481_148150


namespace NUMINAMATH_GPT_problem_f8_minus_f4_l1481_148128

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 2

theorem problem_f8_minus_f4 : f 8 - f 4 = -1 :=
by sorry

end NUMINAMATH_GPT_problem_f8_minus_f4_l1481_148128


namespace NUMINAMATH_GPT_all_points_lie_on_circle_l1481_148189

theorem all_points_lie_on_circle {s : ℝ} :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := (2 * s) / (s^2 + 1)
  x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_all_points_lie_on_circle_l1481_148189


namespace NUMINAMATH_GPT_no_valid_arrangement_l1481_148121

open Nat

theorem no_valid_arrangement :
  ¬ ∃ (f : Fin 30 → ℕ), 
    (∀ (i : Fin 30), 1 ≤ f i ∧ f i ≤ 30) ∧ 
    (∀ (i : Fin 30), ∃ n : ℕ, (f i + f (i + 1) % 30) = n^2) ∧ 
    (∀ i1 i2, i1 ≠ i2 → f i1 ≠ f i2) :=
  sorry

end NUMINAMATH_GPT_no_valid_arrangement_l1481_148121


namespace NUMINAMATH_GPT_negation_proposition_l1481_148186

theorem negation_proposition (x : ℝ) : ¬ (x ≥ 1 → x^2 - 4 * x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4 * x + 2 < -1) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1481_148186


namespace NUMINAMATH_GPT_number_of_intersections_l1481_148167

noncomputable def y1 (x: ℝ) : ℝ := (x - 1) ^ 4
noncomputable def y2 (x: ℝ) : ℝ := 2 ^ (abs x) - 2

theorem number_of_intersections : (∃ x₁ x₂ x₃ x₄ : ℝ, y1 x₁ = y2 x₁ ∧ y1 x₂ = y2 x₂ ∧ y1 x₃ = y2 x₃ ∧ y1 x₄ = y2 x₄ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
sorry

end NUMINAMATH_GPT_number_of_intersections_l1481_148167


namespace NUMINAMATH_GPT_Jeongyeon_record_is_1_44_m_l1481_148141

def Eunseol_record_in_cm : ℕ := 100 + 35
def Jeongyeon_record_in_cm : ℕ := Eunseol_record_in_cm + 9
def Jeongyeon_record_in_m : ℚ := Jeongyeon_record_in_cm / 100

theorem Jeongyeon_record_is_1_44_m : Jeongyeon_record_in_m = 1.44 := by
  sorry

end NUMINAMATH_GPT_Jeongyeon_record_is_1_44_m_l1481_148141


namespace NUMINAMATH_GPT_contradiction_example_l1481_148176

theorem contradiction_example (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
by
  sorry

end NUMINAMATH_GPT_contradiction_example_l1481_148176


namespace NUMINAMATH_GPT_joseph_cards_l1481_148138

theorem joseph_cards (cards_per_student : ℕ) (students : ℕ) (cards_left : ℕ) 
    (H1 : cards_per_student = 23)
    (H2 : students = 15)
    (H3 : cards_left = 12) 
    : (cards_per_student * students + cards_left = 357) := 
  by
  sorry

end NUMINAMATH_GPT_joseph_cards_l1481_148138


namespace NUMINAMATH_GPT_decreasing_function_range_l1481_148143

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else -a * x

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1 / 8 ≤ a ∧ a < 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l1481_148143


namespace NUMINAMATH_GPT_simplify_expression_l1481_148109

theorem simplify_expression (x y z : ℝ) (h1 : x ≠ 2) (h2 : y ≠ 3) (h3 : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by 
sorry

end NUMINAMATH_GPT_simplify_expression_l1481_148109


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1481_148191

/-- In an arithmetic sequence {a_n}, let S_n represent the sum of the first n terms, 
and it is given that S_6 < S_7 and S_7 > S_8. 
Prove that the correct statements among the given options are: 
1. The common difference d < 0 
2. S_9 < S_6 
3. S_7 is definitively the maximum value among all sums S_n. -/
theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_S6_lt_S7 : S 6 < S 7)
  (h_S7_gt_S8 : S 7 > S 8) :
  (a 7 > 0 ∧ a 8 < 0 ∧ ∃ d, ∀ n, a (n + 1) = a n + d ∧ d < 0 ∧ S 9 < S 6 ∧ ∀ n, S n ≤ S 7) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1481_148191


namespace NUMINAMATH_GPT_janet_time_to_home_l1481_148103

-- Janet's initial and final positions
def initial_position : ℕ × ℕ := (0, 0) -- (x, y)
def north_blocks : ℕ := 3
def west_multiplier : ℕ := 7
def south_blocks : ℕ := 8
def east_multiplier : ℕ := 2
def speed_blocks_per_minute : ℕ := 2

def west_blocks : ℕ := west_multiplier * north_blocks
def east_blocks : ℕ := east_multiplier * south_blocks

-- Net movement calculations
def net_south_blocks : ℕ := south_blocks - north_blocks
def net_west_blocks : ℕ := west_blocks - east_blocks

-- Time calculation
def total_blocks_to_home : ℕ := net_south_blocks + net_west_blocks
def time_to_home : ℕ := total_blocks_to_home / speed_blocks_per_minute

theorem janet_time_to_home : time_to_home = 5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_janet_time_to_home_l1481_148103


namespace NUMINAMATH_GPT_smallest_b_not_divisible_by_5_l1481_148172

theorem smallest_b_not_divisible_by_5 :
  ∃ b : ℕ, b > 2 ∧ ¬ (5 ∣ (2 * b^3 - 1)) ∧ ∀ b' > 2, ¬ (5 ∣ (2 * (b'^3) - 1)) → b = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_not_divisible_by_5_l1481_148172


namespace NUMINAMATH_GPT_range_of_k_if_f_monotonically_increasing_l1481_148199

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k_if_f_monotonically_increasing :
  (∀ (x : ℝ), 1 < x → 0 ≤ (k - 1 / x)) → k ∈ Set.Ici (1: ℝ) :=
by
  intro hyp
  have : ∀ (x : ℝ), 1 < x → 0 ≤ k - 1 / x := hyp
  sorry

end NUMINAMATH_GPT_range_of_k_if_f_monotonically_increasing_l1481_148199


namespace NUMINAMATH_GPT_min_value_y_of_parabola_l1481_148151

theorem min_value_y_of_parabola :
  ∃ y : ℝ, ∃ x : ℝ, (∀ y' x', (y' + x') = (y' - x')^2 + 3 * (y' - x') + 3 → y' ≥ y) ∧
            y = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_y_of_parabola_l1481_148151


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1481_148112

/-- Define the quadrants in the Cartesian coordinate system -/
def quadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

theorem point_in_second_quadrant :
  quadrant (-3) 2005 = "Second quadrant" :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1481_148112


namespace NUMINAMATH_GPT_inequality_proof_l1481_148182

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) : 
  (x*y^2/z + y*z^2/x + z*x^2/y) ≥ (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1481_148182


namespace NUMINAMATH_GPT_positive_sum_inequality_l1481_148164

theorem positive_sum_inequality 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) ≥ (ab + bc + ca)^3 := 
by 
  sorry

end NUMINAMATH_GPT_positive_sum_inequality_l1481_148164


namespace NUMINAMATH_GPT_circumcircle_radius_of_triangle_l1481_148158

theorem circumcircle_radius_of_triangle
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (AB BC : ℝ)
  (angle_ABC : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 4)
  (h_angle_ABC : angle_ABC = 120) :
  ∃ (R : ℝ), R = 4 := by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_of_triangle_l1481_148158


namespace NUMINAMATH_GPT_inequality_a4b_to_abcd_l1481_148168

theorem inequality_a4b_to_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end NUMINAMATH_GPT_inequality_a4b_to_abcd_l1481_148168


namespace NUMINAMATH_GPT_sum_of_reciprocals_eq_two_l1481_148154

theorem sum_of_reciprocals_eq_two (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : 1 / x + 1 / y = 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_eq_two_l1481_148154


namespace NUMINAMATH_GPT_initial_number_of_rabbits_is_50_l1481_148190

-- Initial number of weasels
def initial_weasels := 100

-- Each fox catches 4 weasels and 2 rabbits per week
def weasels_caught_per_fox_per_week := 4
def rabbits_caught_per_fox_per_week := 2

-- There are 3 foxes
def num_foxes := 3

-- After 3 weeks, 96 weasels and rabbits are left
def weasels_and_rabbits_left := 96
def weeks := 3

theorem initial_number_of_rabbits_is_50 :
  (initial_weasels + (initial_weasels + weasels_and_rabbits_left)) - initial_weasels = 50 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_rabbits_is_50_l1481_148190


namespace NUMINAMATH_GPT_sin_minus_cos_eq_pm_sqrt_b_l1481_148162

open Real

/-- If θ is an acute angle such that cos(2θ) = b, then sin(θ) - cos(θ) = ±√b. -/
theorem sin_minus_cos_eq_pm_sqrt_b (θ b : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hcos2θ : cos (2 * θ) = b) :
  sin θ - cos θ = sqrt b ∨ sin θ - cos θ = -sqrt b :=
sorry

end NUMINAMATH_GPT_sin_minus_cos_eq_pm_sqrt_b_l1481_148162


namespace NUMINAMATH_GPT_find_c_l1481_148105

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem find_c (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 :=
sorry

end NUMINAMATH_GPT_find_c_l1481_148105


namespace NUMINAMATH_GPT_factoring_sum_of_coefficients_l1481_148194

theorem factoring_sum_of_coefficients 
  (a b c d e f g h j k : ℤ)
  (h1 : 64 * x^6 - 729 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) :
  a + b + c + d + e + f + g + h + j + k = 30 :=
sorry

end NUMINAMATH_GPT_factoring_sum_of_coefficients_l1481_148194


namespace NUMINAMATH_GPT_simplify_root_product_l1481_148197

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end NUMINAMATH_GPT_simplify_root_product_l1481_148197


namespace NUMINAMATH_GPT_compute_combination_product_l1481_148139

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem compute_combination_product :
  combination 10 3 * combination 8 3 = 6720 :=
by
  sorry

end NUMINAMATH_GPT_compute_combination_product_l1481_148139


namespace NUMINAMATH_GPT_negation_equiv_l1481_148179

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end NUMINAMATH_GPT_negation_equiv_l1481_148179


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_zero_l1481_148187

variable {a : ℕ → ℤ}
variable {d : ℤ}

theorem arithmetic_sequence_a5_zero 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : d ≠ 0)
  (h3 : a 3 + a 9 = a 10 - a 8) : 
  a 5 = 0 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_zero_l1481_148187


namespace NUMINAMATH_GPT_transformed_parabola_correct_l1481_148166

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end NUMINAMATH_GPT_transformed_parabola_correct_l1481_148166


namespace NUMINAMATH_GPT_jerry_can_escape_l1481_148171

theorem jerry_can_escape (d : ℝ) (V_J V_T : ℝ) (h1 : (1 / 5) < d) (h2 : d < (1 / 4)) (h3 : V_T = 4 * V_J) :
  (4 * d) / V_J < 1 / (2 * V_J) :=
by
  sorry

end NUMINAMATH_GPT_jerry_can_escape_l1481_148171


namespace NUMINAMATH_GPT_can_cut_into_equal_parts_l1481_148118

-- We assume the existence of a shape S and some grid G along with a function cut
-- that cuts the shape S along grid G lines and returns two parts.
noncomputable def Shape := Type
noncomputable def Grid := Type
noncomputable def cut (S : Shape) (G : Grid) : Shape × Shape := sorry

-- We assume a function superimpose that checks whether two shapes can be superimposed
noncomputable def superimpose (S1 S2 : Shape) : Prop := sorry

-- Assume the given shape S and grid G
variable (S : Shape) (G : Grid)

-- The question rewritten as a Lean statement
theorem can_cut_into_equal_parts : ∃ (S₁ S₂ : Shape), cut S G = (S₁, S₂) ∧ superimpose S₁ S₂ := sorry

end NUMINAMATH_GPT_can_cut_into_equal_parts_l1481_148118


namespace NUMINAMATH_GPT_sally_has_18_nickels_and_total_value_98_cents_l1481_148159

-- Define the initial conditions
def pennies_initial := 8
def nickels_initial := 7
def nickels_from_dad := 9
def nickels_from_mom := 2

-- Define calculations based on the initial conditions
def total_nickels := nickels_initial + nickels_from_dad + nickels_from_mom
def value_pennies := pennies_initial
def value_nickels := total_nickels * 5
def total_value := value_pennies + value_nickels

-- State the theorem to prove the correct answers
theorem sally_has_18_nickels_and_total_value_98_cents :
  total_nickels = 18 ∧ total_value = 98 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_sally_has_18_nickels_and_total_value_98_cents_l1481_148159


namespace NUMINAMATH_GPT_shots_and_hits_l1481_148111

theorem shots_and_hits (n k : ℕ) (h₀ : 10 < n) (h₁ : n < 20) (h₂ : 5 * k = 3 * (n - k)) : (n = 16) ∧ (k = 6) :=
by {
  -- We state the result that we wish to prove
  sorry
}

end NUMINAMATH_GPT_shots_and_hits_l1481_148111


namespace NUMINAMATH_GPT_parabola_and_line_solutions_l1481_148116

-- Definition of the parabola with its focus
def parabola_with_focus (p : ℝ) : Prop :=
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (x : ℝ), x = 1 / 2)

-- Definitions of conditions for intersection and orthogonal vectors
def line_intersecting_parabola (slope t : ℝ) (p : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), 
  (y1 = 2 * x1 + t) ∧ (y2 = 2 * x2 + t) ∧
  (y1^2 = 2 * x1) ∧ (y2^2 = 2 * x2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧
  (x1 * x2 = (t^2) / 4) ∧ (x1 * x2 + y1 * y2 = 0)

-- Lean statement for the proof problem
theorem parabola_and_line_solutions :
  ∀ p t : ℝ, 
  parabola_with_focus p → 
  (line_intersecting_parabola 2 t p → t = -4)
  → p = 1 :=
by
  intros p t h_parabola h_line
  sorry

end NUMINAMATH_GPT_parabola_and_line_solutions_l1481_148116


namespace NUMINAMATH_GPT_steps_in_staircase_l1481_148134

theorem steps_in_staircase :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n = 19 :=
by
  sorry

end NUMINAMATH_GPT_steps_in_staircase_l1481_148134


namespace NUMINAMATH_GPT_find_u5_l1481_148184

theorem find_u5 
  (u : ℕ → ℝ)
  (h_rec : ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n)
  (h_u3 : u 3 = 9)
  (h_u6 : u 6 = 243) : 
  u 5 = 69 :=
sorry

end NUMINAMATH_GPT_find_u5_l1481_148184


namespace NUMINAMATH_GPT_inequality_problem_l1481_148148

theorem inequality_problem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1481_148148


namespace NUMINAMATH_GPT_solve_equation_l1481_148108

theorem solve_equation :
  ∃ x : ℝ, (3 * x^2 / (x - 2)) - (4 * x + 11) / 5 + (7 - 9 * x) / (x - 2) + 2 = 0 :=
sorry

end NUMINAMATH_GPT_solve_equation_l1481_148108


namespace NUMINAMATH_GPT_distinct_real_roots_l1481_148147

theorem distinct_real_roots :
  ∀ x : ℝ, (x^3 - 3*x^2 + x - 2) * (x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18 = 0 ↔
  x = 1 ∨ x = -2 ∨ x = 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_distinct_real_roots_l1481_148147


namespace NUMINAMATH_GPT_function_satisfies_conditions_l1481_148131

-- Define the conditions
def f (n : ℕ) : ℕ := n + 1

-- Prove that the function f satisfies the given conditions
theorem function_satisfies_conditions : 
  (f 0 = 1) ∧ (f 2012 = 2013) :=
by
  sorry

end NUMINAMATH_GPT_function_satisfies_conditions_l1481_148131


namespace NUMINAMATH_GPT_color_of_last_bead_l1481_148101

-- Define the sequence and length of repeated pattern
def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "green", "blue"]
def pattern_length : Nat := bead_pattern.length

-- Define the total number of beads in the bracelet
def total_beads : Nat := 85

-- State the theorem to prove the color of the last bead
theorem color_of_last_bead : bead_pattern.get? ((total_beads - 1) % pattern_length) = some "yellow" :=
by
  sorry

end NUMINAMATH_GPT_color_of_last_bead_l1481_148101


namespace NUMINAMATH_GPT_slope_of_line_through_PQ_is_4_l1481_148123

theorem slope_of_line_through_PQ_is_4
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a4 : a 4 = 15)
  (h_a9 : a 9 = 55) :
  let a3 := a 3
  let a8 := a 8
  (a 9 - a 4) / (9 - 4) = 8 → (a 8 - a 3) / (13 - 3) = 4 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_through_PQ_is_4_l1481_148123


namespace NUMINAMATH_GPT_point_not_on_graph_l1481_148125

theorem point_not_on_graph : ∀ (x y : ℝ), (x, y) = (-1, 1) → ¬ (∃ z : ℝ, z ≠ -1 ∧ y = z / (z + 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_point_not_on_graph_l1481_148125


namespace NUMINAMATH_GPT_equations_solutions_l1481_148196

-- Definition and statement for Equation 1
noncomputable def equation1_solution1 : ℝ :=
  (-3 + Real.sqrt 17) / 4

noncomputable def equation1_solution2 : ℝ :=
  (-3 - Real.sqrt 17) / 4

-- Definition and statement for Equation 2
def equation2_solution : ℝ :=
  -6

-- Theorem proving the solutions to the given equations
theorem equations_solutions :
  (∃ x : ℝ, 2 * x^2 + 3 * x = 1 ∧ (x = equation1_solution1 ∨ x = equation1_solution2)) ∧
  (∃ x : ℝ, 3 / (x - 2) = 5 / (2 - x) - 1 ∧ x = equation2_solution) :=
by
  sorry

end NUMINAMATH_GPT_equations_solutions_l1481_148196


namespace NUMINAMATH_GPT_count_inverses_modulo_11_l1481_148192

theorem count_inverses_modulo_11 : (Finset.filter (λ a => Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
  by
  sorry

end NUMINAMATH_GPT_count_inverses_modulo_11_l1481_148192


namespace NUMINAMATH_GPT_annuity_payment_l1481_148155

variable (P : ℝ) (A : ℝ) (i : ℝ) (n1 n2 : ℕ)

-- Condition: Principal amount
axiom principal_amount : P = 24000

-- Condition: Annual installment for the first 5 years
axiom annual_installment : A = 1500 

-- Condition: Annual interest rate
axiom interest_rate : i = 0.045 

-- Condition: Years before equal annual installments
axiom years_before_installment : n1 = 5 

-- Condition: Years for repayment after the first 5 years
axiom repayment_years : n2 = 7 

-- Remaining debt after n1 years
noncomputable def remaining_debt_after_n1 : ℝ :=
  P * (1 + i) ^ n1 - A * ((1 + i) ^ n1 - 1) / i

-- Annual payment for n2 years to repay the remaining debt
noncomputable def annual_payment (D : ℝ) : ℝ :=
  D * (1 + i) ^ n2 / (((1 + i) ^ n2 - 1) / i)

axiom remaining_debt_amount : remaining_debt_after_n1 P A i n1 = 21698.685 

theorem annuity_payment : annual_payment (remaining_debt_after_n1 P A i n1) = 3582 := by
  sorry

end NUMINAMATH_GPT_annuity_payment_l1481_148155


namespace NUMINAMATH_GPT_inscribed_square_length_l1481_148198

-- Define the right triangle PQR with given sides
variables (PQ QR PR : ℕ)
variables (h s : ℚ)

-- Given conditions
def right_triangle_PQR : Prop := PQ = 5 ∧ QR = 12 ∧ PR = 13
def altitude_Q_to_PR : Prop := h = (PQ * QR) / PR
def side_length_of_square : Prop := s = h * (1 - h / PR)

theorem inscribed_square_length (PQ QR PR h s : ℚ) 
    (right_triangle_PQR : PQ = 5 ∧ QR = 12 ∧ PR = 13)
    (altitude_Q_to_PR : h = (PQ * QR) / PR) 
    (side_length_of_square : s = h * (1 - h / PR)) 
    : s = 6540 / 2207 := by
  -- we skip the proof here as requested
  sorry

end NUMINAMATH_GPT_inscribed_square_length_l1481_148198


namespace NUMINAMATH_GPT_find_solutions_l1481_148110

theorem find_solutions (x y : ℕ) : 33 ^ x + 31 = 2 ^ y → (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := 
by
  sorry

end NUMINAMATH_GPT_find_solutions_l1481_148110


namespace NUMINAMATH_GPT_volume_of_remaining_sphere_after_hole_l1481_148178

noncomputable def volume_of_remaining_sphere (R : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * R^3
  let volume_cylinder := (4 / 3) * Real.pi * (R / 2)^3
  volume_sphere - volume_cylinder

theorem volume_of_remaining_sphere_after_hole : 
  volume_of_remaining_sphere 5 = (500 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_remaining_sphere_after_hole_l1481_148178


namespace NUMINAMATH_GPT_shortest_distance_between_stations_l1481_148193

/-- 
Given two vehicles A and B shuttling between two locations,
with Vehicle A stopping every 0.5 kilometers and Vehicle B stopping every 0.8 kilometers,
prove that the shortest distance between two stations where Vehicles A and B do not stop at the same place is 0.1 kilometers.
-/
theorem shortest_distance_between_stations :
  ∀ (dA dB : ℝ), (dA = 0.5) → (dB = 0.8) → ∃ δ : ℝ, (δ = 0.1) ∧ (∀ n m : ℕ, dA * n ≠ dB * m → abs ((dA * n) - (dB * m)) = δ) :=
by
  intros dA dB hA hB
  use 0.1
  sorry

end NUMINAMATH_GPT_shortest_distance_between_stations_l1481_148193


namespace NUMINAMATH_GPT_fraction_identity_l1481_148107

theorem fraction_identity (a b : ℝ) (h : a / b = 5 / 2) : (a + 2 * b) / (a - b) = 3 :=
by sorry

end NUMINAMATH_GPT_fraction_identity_l1481_148107


namespace NUMINAMATH_GPT_chair_cost_l1481_148130

theorem chair_cost :
  (∃ (C : ℝ), 3 * C + 50 + 40 = 130 - 4) → 
  (∃ (C : ℝ), C = 12) :=
by
  sorry

end NUMINAMATH_GPT_chair_cost_l1481_148130


namespace NUMINAMATH_GPT_fettuccine_to_tortellini_ratio_l1481_148180

-- Definitions based on the problem conditions
def total_students := 800
def preferred_spaghetti := 320
def preferred_fettuccine := 200
def preferred_tortellini := 160
def preferred_penne := 120

-- Theorem to prove that the ratio is 5/4
theorem fettuccine_to_tortellini_ratio :
  (preferred_fettuccine : ℚ) / (preferred_tortellini : ℚ) = 5 / 4 :=
sorry

end NUMINAMATH_GPT_fettuccine_to_tortellini_ratio_l1481_148180


namespace NUMINAMATH_GPT_actual_distance_in_km_l1481_148104

-- Given conditions
def scale_factor : ℕ := 200000
def map_distance_cm : ℚ := 3.5

-- Proof goal: the actual distance in kilometers
theorem actual_distance_in_km : (map_distance_cm * scale_factor) / 100000 = 7 := 
by
  sorry

end NUMINAMATH_GPT_actual_distance_in_km_l1481_148104


namespace NUMINAMATH_GPT_calculate_expression_l1481_148149

theorem calculate_expression : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 :=
sorry

end NUMINAMATH_GPT_calculate_expression_l1481_148149


namespace NUMINAMATH_GPT_polynomial_expansion_coefficient_a8_l1481_148183

theorem polynomial_expansion_coefficient_a8 :
  let a := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  a_8 = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_expansion_coefficient_a8_l1481_148183


namespace NUMINAMATH_GPT_rainy_days_l1481_148142

theorem rainy_days
  (rain_on_first_day : ℕ) (rain_on_second_day : ℕ) (rain_on_third_day : ℕ) (sum_of_first_two_days : ℕ)
  (h1 : rain_on_first_day = 4)
  (h2 : rain_on_second_day = 5 * rain_on_first_day)
  (h3 : sum_of_first_two_days = rain_on_first_day + rain_on_second_day)
  (h4 : rain_on_third_day = sum_of_first_two_days - 6) :
  rain_on_third_day = 18 :=
by
  sorry

end NUMINAMATH_GPT_rainy_days_l1481_148142


namespace NUMINAMATH_GPT_tetrahedron_volume_l1481_148163

noncomputable def volume_tetrahedron (A₁ A₂ : ℝ) (θ : ℝ) (d : ℝ) : ℝ :=
  (A₁ * A₂ * Real.sin θ) / (3 * d)

theorem tetrahedron_volume:
  ∀ (PQ PQR PQS : ℝ) (θ : ℝ),
  PQ = 5 → PQR = 20 → PQS = 18 → θ = Real.pi / 4 → volume_tetrahedron PQR PQS θ PQ = 24 * Real.sqrt 2 :=
by
  intros
  unfold volume_tetrahedron
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1481_148163


namespace NUMINAMATH_GPT_probability_of_all_heads_or_tails_l1481_148106

def num_favorable_outcomes : ℕ := 2

def total_outcomes : ℕ := 2 ^ 5

def probability_all_heads_or_tails : ℚ := num_favorable_outcomes / total_outcomes

theorem probability_of_all_heads_or_tails :
  probability_all_heads_or_tails = 1 / 16 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_all_heads_or_tails_l1481_148106


namespace NUMINAMATH_GPT_hiker_distance_l1481_148115

variable (s t d : ℝ)
variable (h₁ : (s + 1) * (2 / 3 * t) = d)
variable (h₂ : (s - 1) * (t + 3) = d)

theorem hiker_distance  : d = 6 :=
by
  sorry

end NUMINAMATH_GPT_hiker_distance_l1481_148115


namespace NUMINAMATH_GPT_find_base_of_triangle_l1481_148160

-- Given data
def perimeter : ℝ := 20 -- The perimeter of the triangle
def tangent_segment : ℝ := 2.4 -- The segment of the tangent to the inscribed circle contained between the sides

-- Define the problem and expected result
theorem find_base_of_triangle (a b c : ℝ) (P : a + b + c = perimeter)
  (tangent_parallel_base : ℝ := tangent_segment):
  a = 4 ∨ a = 6 :=
sorry

end NUMINAMATH_GPT_find_base_of_triangle_l1481_148160


namespace NUMINAMATH_GPT_correct_answer_is_B_l1481_148188

-- Define what it means to be a quadratic equation in one variable
def is_quadratic_in_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x ^ 2 + b * x + c = 0)

-- Conditions:
def eqA (x : ℝ) : Prop := 2 * x + 1 = 0
def eqB (x : ℝ) : Prop := x ^ 2 + 1 = 0
def eqC (x y : ℝ) : Prop := y ^ 2 + x = 1
def eqD (x : ℝ) : Prop := 1 / x + x ^ 2 = 1

-- Theorem statement: Prove which equation is a quadratic equation in one variable
theorem correct_answer_is_B : is_quadratic_in_one_variable eqB :=
sorry  -- Proof is not required as per the instructions

end NUMINAMATH_GPT_correct_answer_is_B_l1481_148188
