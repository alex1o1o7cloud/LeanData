import Mathlib

namespace evaluate_expression_l380_380144

def g (x : ℝ) : ℝ := x^2 + 3 * real.sqrt x

theorem evaluate_expression : 3 * g 3 - g 9 = -63 + 9 * real.sqrt 3 := by
  sorry

end evaluate_expression_l380_380144


namespace door_opening_probability_l380_380806

theorem door_opening_probability:
  (∀ (k : ℕ), k = 4 → (∃ (open_keys close_keys: ℕ), open_keys = 2 ∧ close_keys = 2 ∧ 
    ((2/4) * (2/3) = 1/3))) :=
begin
  intro k,
  assume h_k,
  use [2, 2],
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end door_opening_probability_l380_380806


namespace combined_stock_cost_l380_380767

def stock_A_face_value := 100
def stock_A_discount := 2
def stock_A_brokerage := 1 / 5

def stock_B_face_value := 150
def stock_B_premium := 1.5
def stock_B_brokerage := 1 / 6

def stock_C_face_value := 200
def stock_C_discount := 3
def stock_C_brokerage := 0.5

theorem combined_stock_cost :
  let stock_A_cost := stock_A_face_value - (stock_A_discount / 100 * stock_A_face_value)
  let stock_A_brokerage_cost := stock_A_brokerage / 100 * stock_A_cost
  let total_stock_A_cost := stock_A_cost + stock_A_brokerage_cost
  
  let stock_B_cost := stock_B_face_value + (stock_B_premium / 100 * stock_B_face_value)
  let stock_B_brokerage_cost := stock_B_brokerage / 100 * stock_B_cost
  let total_stock_B_cost := stock_B_cost + stock_B_brokerage_cost
  
  let stock_C_cost := stock_C_face_value - (stock_C_discount / 100 * stock_C_face_value)
  let stock_C_brokerage_cost := stock_C_brokerage / 100 * stock_C_cost
  let total_stock_C_cost := stock_C_cost + stock_C_brokerage_cost
  
  total_stock_A_cost + total_stock_B_cost + total_stock_C_cost = 445.66975 :=
by 
  compute; sorry

end combined_stock_cost_l380_380767


namespace tan_4050_deg_undefined_l380_380139

theorem tan_4050_deg_undefined :
  let theta := 4050 * (π / 180) in -- Convert degrees to radians
  4050 % 360 = 90 ∧
  tan 90 = Real.tan (π / 2) ∧
  cos (π / 2) = 0
  → tan theta = 0 := by
  sorry

end tan_4050_deg_undefined_l380_380139


namespace minimum_reciprocal_sum_l380_380211

theorem minimum_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  4 ≤ (1 / a) + (1 / b) :=
sorry

end minimum_reciprocal_sum_l380_380211


namespace interpret_billion_correctly_l380_380155

theorem interpret_billion_correctly:
  let billion := 10^9 in
  let interpretation := 2.74 * 10^8 in
  (2.74 * billion = interpretation * 10) :=
by sorry

end interpret_billion_correctly_l380_380155


namespace sphere_to_hemisphere_volume_ratio_l380_380745

theorem sphere_to_hemisphere_volume_ratio (π : ℝ) (q : ℝ) (hq : 0 < q) : 
  let V_sphere := (4 / 3) * π * (3 * q)^3 in
  let V_hemisphere := (1 / 2) * (4 / 3) * π * q^3 in
  V_sphere / V_hemisphere = 54 :=
by
  let V_sphere := (4 / 3) * π * (3 * q)^3
  let V_hemisphere := (1 / 2) * (4 / 3) * π * q^3
  have h1 : V_sphere = 36 * π * q^3 := sorry
  have h2 : V_hemisphere = (2 / 3) * π * q^3 := sorry
  calc
    V_sphere / V_hemisphere
      = (36 * π * q^3) / ((2 / 3) * π * q^3) : by rw [h1, h2]
  ... = 54 : sorry

end sphere_to_hemisphere_volume_ratio_l380_380745


namespace train_crossing_time_l380_380480

-- Define the conditions
def train_length : ℕ := 170
def train_speed_kmh : ℝ := 45
def bridge_length : ℕ := 205
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- Convert speed to m/s
def total_distance : ℕ := train_length + bridge_length

-- State the theorem
theorem train_crossing_time : (total_distance / train_speed_ms) = 30 := by 
  sorry

end train_crossing_time_l380_380480


namespace find_n_for_max_triangles_l380_380412

-- Let's define the problem in Lean state.
def max_triangles (n : ℕ) : ℕ :=
  2 + 2 * n

theorem find_n_for_max_triangles : ∃ n : ℕ, max_triangles n = 1992 :=
by
  use 995
  dsimp [max_triangles]
  rfl

end find_n_for_max_triangles_l380_380412


namespace jenny_eighth_time_l380_380279

def jennys_times : List ℕ := [102, 108, 110, 99, 104, 107, 113]

def median (l : List ℕ) : Option ℕ :=
  let sorted := l.qsort (· < ·)
  if sorted.length % 2 = 0 then
    let mid := sorted.length / 2
    if h : mid < sorted.length then
      some ((sorted.get ⟨mid - 1, by linarith⟩ + sorted.get ⟨mid, h⟩) / 2)
    else
      none
  else
    some (sorted.get ⟨sorted.length / 2, by linarith⟩)

theorem jenny_eighth_time (x : ℕ) :
  (median (x :: jennys_times) = some 106) ↔ x = 105 :=
by sorry

end jenny_eighth_time_l380_380279


namespace a_9_equals_18_l380_380201

def is_sequence_of_positive_integers (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → 0 < a n

def satisfies_recursive_relation (a : ℕ → ℕ) : Prop :=
∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem a_9_equals_18 (a : ℕ → ℕ)
  (H1 : is_sequence_of_positive_integers a)
  (H2 : satisfies_recursive_relation a)
  (H3 : a 2 = 4) : a 9 = 18 :=
sorry

end a_9_equals_18_l380_380201


namespace min_balls_to_guarantee_18_single_color_l380_380446

theorem min_balls_to_guarantee_18_single_color :
  ∀ (red green yellow blue white black : ℕ),
  red = 35 → green = 22 → yellow = 18 → blue = 15 → white = 12 → black = 8 →
  (∀ (n : ℕ), (n = red - 17 + green - 17 + yellow - 17 + blue + white + black) →
    n + 1 = 87) :=
by
  intros red green yellow blue white black h_red h_green h_yellow h_blue h_white h_black n h_n
  have h_sum : 17 + 17 + 17 + 15 + 12 + 8 = 86 := by norm_num
  have h_drawing : red - 17 = 18 := by rw [h_red, nat.sub_eq_or_eq_add, eq_add_iff_add_eq.mpr] at h_sum; symm; rw ← nat.add_sub_assoc_left h_sum; norm_num
  have h_drawing_proves : n + 1 = 87 := by rw [h_n, h_sum]; norm_num
  exact h_drawing_proves

end min_balls_to_guarantee_18_single_color_l380_380446


namespace larger_solution_of_quadratic_equation_l380_380544

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l380_380544


namespace probability_is_correct_l380_380788

noncomputable def probability_hare_given_claims : ℝ :=
  let P_A := 1/2 -- Probability the individual is a hare
  let P_notA := 1/2 -- Probability the individual is not a hare (rabbit)
  let P_B_given_A := 1/4 -- Probability a hare claims not to be a hare
  let P_C_given_A := 3/4 -- Probability a hare claims not to be a rabbit
  let P_B_given_notA := 2/3 -- Probability a rabbit claims not to be a hare
  let P_C_given_notA := 1/3 -- Probability a rabbit claims not to be a rabbit
  let P_A_and_B_and_C := P_A * P_B_given_A * P_C_given_A -- Joint probability A ∩ B ∩ C
  let P_notA_and_B_and_C := P_notA * P_B_given_notA * P_C_given_notA -- Joint probability ¬A ∩ B ∩ C
  let P_B_and_C := P_A_and_B_and_C + P_notA_and_B_and_C -- Total probability of B ∩ C
  P_A_and_B_and_C / P_B_and_C -- Conditional probability A | (B ∩ C)

theorem probability_is_correct : probability_hare_given_claims = 27 / 59 := 
  by 
    -- Establish the values directly as per the conditions
    let P_A : ℝ := 1/2
    let P_B_given_A : ℝ := 1/4
    let P_C_given_A : ℝ := 3/4
    let P_notA : ℝ := 1/2
    let P_B_given_notA : ℝ := 2/3
    let P_C_given_notA : ℝ := 1/3
    let P_A_and_B_and_C : ℝ := P_A * P_B_given_A * P_C_given_A
    let P_notA_and_B_and_C : ℝ := P_notA * P_B_given_notA * P_C_given_notA
    let P_B_and_C : ℝ := P_A_and_B_and_C + P_notA_and_B_and_C
    have P_B_and_C_value : P_B_and_C = 59 / 288 := by sorry
    have P_A_and_B_and_C_value : P_A_and_B_and_C = 3 / 32 := by sorry
    have prob_value : (3 / 32) * (288 / 59) = 27 / 59 :=
      by sorry
    exact prob_value

end probability_is_correct_l380_380788


namespace candles_left_l380_380820

theorem candles_left (total_candles : ℕ) (alyssa_fraction_used : ℚ) (chelsea_fraction_used : ℚ) 
  (h_total : total_candles = 40) 
  (h_alyssa : alyssa_fraction_used = (1 / 2)) 
  (h_chelsea : chelsea_fraction_used = (70 / 100)) : 
  total_candles - (alyssa_fraction_used * total_candles).toNat - (chelsea_fraction_used * (total_candles - (alyssa_fraction_used * total_candles).toNat)).toNat = 6 :=
by 
  sorry

end candles_left_l380_380820


namespace avg_students_teacher_minus_student_l380_380101

theorem avg_students_teacher_minus_student 
  (students teachers : ℕ) 
  (class_enrollments : List ℕ)
  (h1 : students = 120)
  (h2 : teachers = 4)
  (h3 : class_enrollments = [60, 30, 20, 10]) :
  let t := (List.sum class_enrollments) / teachers
  let s := (60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 10 * (10 / 120))
  t - s = -11.66 :=
by
  sorry

end avg_students_teacher_minus_student_l380_380101


namespace three_digit_integers_with_at_least_two_identical_digits_l380_380952

/-- Prove that the number of positive three-digit integers less than 700 that have at least two identical digits is 162. -/
theorem three_digit_integers_with_at_least_two_identical_digits : 
  ∃ n : ℕ, (n = 162) ∧ (count_three_digit_integers_with_at_least_two_identical_digits n) :=
by
  sorry

/-- Define a function to count the number of three-digit integers less than 700 with at least two identical digits -/
noncomputable def count_three_digit_integers_with_at_least_two_identical_digits (n : ℕ) : Prop :=
  n = 162

end three_digit_integers_with_at_least_two_identical_digits_l380_380952


namespace ln_inequality_f_g_inequality_l380_380577

theorem ln_inequality (x : ℝ) (h : x > 1) : 
  log x < (x / 2) - (1 / (2 * x)) :=
sorry

theorem f_g_inequality (a : ℝ) (h : ∀ x > 0, a * exp (x - 2) > (x + (1 / x) + 2) * log x) : 
  a >= 4 :=
sorry

end ln_inequality_f_g_inequality_l380_380577


namespace digit_150_in_17_div_70_l380_380003

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380003


namespace problem_equiv_l380_380171

theorem problem_equiv {a : ℤ} : (a^2 ≡ 9 [ZMOD 10]) ↔ (a ≡ 3 [ZMOD 10] ∨ a ≡ -3 [ZMOD 10] ∨ a ≡ 7 [ZMOD 10] ∨ a ≡ -7 [ZMOD 10]) :=
sorry

end problem_equiv_l380_380171


namespace find_analytical_expression_of_f_find_max_value_of_f_on_interval_l380_380230

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1 / 3) * x^3 + a * x + b

theorem find_analytical_expression_of_f :
  ∃ a b, (∀ x, f x a b = (1/3) * x^3 - 4 * x + 4) ∧
         (f (-2) a b = 28 / 3) ∧
         (f (4) a b = 28 / 3) :=
sorry

theorem find_max_value_of_f_on_interval (m : ℝ) :
  let fx := f in
  (∀ x, fx x (-4) 4 = (1/3) * x^3 - 4 * x + 4) →
  if m < -4 then true else
  if m > 4 then true else
  if -4 < m ∧ m < -2 then
    (max (fx (-4) (-4) 4) (fx m (-4) 4) = fx m (-4) 4) ∧ (fx m (-4) 4 = (1/3) * m^3 - 4 * m + 4)
  else if -2 ≤ m ∧ m ≤ 4 then
    (max (fx (-4) (-4) 4) (max (fx (-2) (-4) 4) (fx 4 (-4) 4)) = fx (-2) (-4) 4) ∧ (fx (-2) (-4) 4 = 28 / 3)
  else
    (max (fx (-4) (-4) 4) (fx m (-4) 4) = fx m (-4) 4) ∧ (fx m (-4) 4 = (1/3) * m^3 - 4 * m + 4) :=
sorry

end find_analytical_expression_of_f_find_max_value_of_f_on_interval_l380_380230


namespace bc_length_l380_380100

theorem bc_length
  (A D E F B C : Type)
  [pointOnCircle A]
  (BC : B × C)
  (D_on_BC : D ∈ BC)
  (E_on_BC : E ∈ BC)
  (F_on_extension_BC_beyond_B : ∃ F', F = F' ∧ F' ∈ BC ∧ B ∈ BC ∧ B ≠ F')
  (angle_BAD_eq_ACD : ∀ (α : ℝ), angle_BAD = α ∧ angle_ACD = α)
  (angle_BAF_eq_CAE : ∀ (β : ℝ), angle_BAF = β ∧ angle_CAE = β)
  (BD_eq_2 : DynkinTopUnitary ≅ 2)
  (BE_eq_5 : DynkinTopUnitary ≅ 5)
  (BF_eq_4 : DynkinTopUnitary ≅ 4) :
  BC = 11 := sorry

end bc_length_l380_380100


namespace part1_part2_l380_380688

variables {x a : ℝ}

noncomputable def vec_a := (Real.cos x, Real.sin x)
noncomputable def vec_b := (Real.cos x + 2 * Real.sqrt 3, Real.sin x)
noncomputable def vec_c := (Real.sin a, Real.cos a)
noncomputable def f (x : ℝ) := vec_a.1 * (vec_b.1 - 2 * vec_c.1) + vec_a.2 * (vec_b.2 - 2 * vec_c.2)

theorem part1 (h : vec_a.1 * vec_c.1 + vec_a.2 * vec_c.2 = 0) : Real.cos (2 * x + 2 * a) = 1 := sorry

theorem part2 (ha : a = 0) : 
  ∃ k : ℤ, ∀ x ∈ {x | f x = 5}, x = 2 * k * Real.pi - Real.pi / 6 := sorry

end part1_part2_l380_380688


namespace cost_formula_correct_l380_380455

def total_cost (P : ℕ) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem cost_formula_correct (P : ℕ) : 
  total_cost P = (if P ≤ 2 then 15 else 15 + 5 * (P - 2)) :=
by 
  exact rfl

end cost_formula_correct_l380_380455


namespace inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l380_380220

-- Definitions for the conditions
def inside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 < r^2 ∧ (M.1 ≠ 0 ∨ M.2 ≠ 0)

def on_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 = r^2

def outside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 > r^2

def line_l_intersects_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 < r^2 ∨ M.1 * M.1 + M.2 * M.2 = r^2

def line_l_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 = r^2

def line_l_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 > r^2

-- Propositions
theorem inside_circle_implies_line_intersects_circle (M : ℝ × ℝ) (r : ℝ) : 
  inside_circle M r → line_l_intersects_circle M r := 
sorry

theorem on_circle_implies_line_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) :
  on_circle M r → line_l_tangent_to_circle M r :=
sorry

theorem outside_circle_implies_line_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) :
  outside_circle M r → line_l_does_not_intersect_circle M r :=
sorry

end inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l380_380220


namespace broken_line_coverable_l380_380417

noncomputable def cover_broken_line (length_of_line : ℝ) (radius_of_circle : ℝ) : Prop :=
  length_of_line = 5 ∧ radius_of_circle > 1.25

theorem broken_line_coverable :
  ∃ radius_of_circle, cover_broken_line 5 radius_of_circle :=
by sorry

end broken_line_coverable_l380_380417


namespace sky_colors_l380_380523

theorem sky_colors (h1 : ∀ t : ℕ, t = 2) (h2 : ∀ m : ℕ, m = 60) (h3 : ∀ c : ℕ, c = 10) : 
  ∃ n : ℕ, n = 12 :=
by
  let total_duration := (2 * 60 : ℕ)
  let num_colors := total_duration / 10
  have : num_colors = 12 := by decide
  use num_colors
  assumption_needed

end sky_colors_l380_380523


namespace vector_norm_squared_sum_l380_380302

variables {a b : EuclideanSpace ℝ (Fin 2)}
noncomputable def m : EuclideanSpace ℝ (Fin 2) := ![4, 5]

theorem vector_norm_squared_sum :
  (m = (a + b) / 2) ∧
  (inner a b = 10) →
  ‖a‖^2 + ‖b‖^2 = 144 :=
by {
  sorry
}

end vector_norm_squared_sum_l380_380302


namespace complex_number_quad_l380_380655

def is_in_second_quadrant(z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quad :
  is_in_second_quadrant (2 * complex.I - 1) :=
sorry

end complex_number_quad_l380_380655


namespace chameleon_increase_l380_380349

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380349


namespace chameleon_problem_l380_380361

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380361


namespace triangle_area_l380_380418

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (0, 0)) (hB : B = (0, 8)) (hC : C = (10, 15)) : 
  let base := 8
  let height := 10
  let area := 1 / 2 * base * height
  area = 40.0 :=
by
  sorry

end triangle_area_l380_380418


namespace chameleon_increase_l380_380351

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380351


namespace candidate_got_30_99_percent_l380_380795

noncomputable def candidate_percentage (total_votes lost_by : ℕ) : ℚ := 
  let P := (((total_votes - lost_by : ℕ) * 100 / total_votes : ℚ) / 2)
  in P

theorem candidate_got_30_99_percent :
  candidate_percentage 6450 2451 = 30.99 :=
by
  sorry

end candidate_got_30_99_percent_l380_380795


namespace distinct_lines_intersect_iff_cond_l380_380672

noncomputable def line_intersection (L1 L2 : Set (ℝ × ℝ)) : Prop :=
  ∃ x y, (x, y) ∈ L1 ∧ (x, y) ∈ L2

def exists_points (L1 L2 : Set (ℝ × ℝ)) (λ : ℝ) (P : ℝ × ℝ) : Prop :=
  ∀ (λ : ℝ) (hλ : λ ≠ 0) (P : ℝ × ℝ) (hP : (P ∉ L1) ∧ (P ∉ L2)),
  ∃ (A1 ∈ L1) (A2 ∈ L2), (∃ x1 y1 x2 y2, P = (x1, y1) ∧ A1 = (x2, y2) ∧ 
                          vector_subtraction A2 P = vector_scalar_mul λ (vector_subtraction A1 P))

theorem distinct_lines_intersect_iff_cond (L1 L2 : Set (ℝ × ℝ)) :
  distinct L1 L2 → (line_intersection L1 L2 ↔ exists_points L1 L2) :=
sorry

end distinct_lines_intersect_iff_cond_l380_380672


namespace interest_equality_l380_380476

theorem interest_equality (total_sum : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) (n : ℝ) :
  total_sum = 2730 ∧ part1 = 1050 ∧ part2 = 1680 ∧
  rate1 = 3 ∧ time1 = 8 ∧ rate2 = 5 ∧ part1 * rate1 * time1 = part2 * rate2 * n →
  n = 3 :=
by
  sorry

end interest_equality_l380_380476


namespace second_player_wins_l380_380698

theorem second_player_wins :
  ∀ (n : ℕ), n ∈ {1, 2, ..., 1000} →
  ∃ (m : ℕ), m ∈ {1, 2, ..., 1000} ∧ (n + m = 1001) →
  (n^2 - m^2) % 13 = 0 :=
by {
  intros n hn,
  use 1001 - n,
  split,
  {
    sorry -- prove that 1001 - n is in the set
  },
  {
    intro h,
    have h1: n + (1001 - n) = 1001 := by linarith,
    rw h1,
    sorry -- prove divisibility by 13
  }
}

end second_player_wins_l380_380698


namespace problem1_problem2_l380_380561

noncomputable def z : ℂ := ((1 + complex.i)^2 + 3 * (1 - complex.i)) / (2 + complex.i)

theorem problem1 : complex.abs z = real.sqrt 2 := by
  sorry

theorem problem2 : ∃ (a b : ℝ), (z^2 + a*z + b = 1 + complex.i) ∧ (a = -3) ∧ (b = 4) := by
  sorry

end problem1_problem2_l380_380561


namespace find_x_for_mean_l380_380429

theorem find_x_for_mean 
(x : ℝ) 
(h_mean : (3 + 11 + 7 + 9 + 15 + 13 + 8 + 19 + 17 + 21 + 14 + x) / 12 = 12) : 
x = 7 :=
sorry

end find_x_for_mean_l380_380429


namespace meal_combinations_l380_380966

theorem meal_combinations : 
  let menu_items := 15 in
  let yann_choices := menu_items in
  let camille_choices := menu_items in
  yann_choices * camille_choices = 225 :=
by {
  let menu_items := 15,
  let yann_choices := menu_items,
  let camille_choices := menu_items,
  show yann_choices * camille_choices = 225,
  sorry
}

end meal_combinations_l380_380966


namespace range_of_m_for_p_and_not_q_l380_380206

def proposition_p (m : ℝ) := ∀ x y : ℝ, 0 < x → 0 < y → x < y → (m / x) < (m / y)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

def proposition_q (m : ℝ) := let d := discriminant 3 (2 * m) (m - 6) in d > 0

theorem range_of_m_for_p_and_not_q :
  {m : ℝ | (proposition_p m) ∧ ¬(proposition_q m)} = {m : ℝ | -3 ≤ m ∧ m < 0} :=
by
  sorry

end range_of_m_for_p_and_not_q_l380_380206


namespace find_f_of_4_l380_380202

variable {f : ℝ → ℝ}

-- Define the property of the odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

-- Define the specific function property for x < 0
def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f(x) = x * (2 - x)

-- The proof statement
theorem find_f_of_4 (hf : is_odd_function f) (hf_neg : function_property f) : f 4 = 24 := 
by
  sorry

end find_f_of_4_l380_380202


namespace problem_statement_l380_380566

def f (x : ℝ) : ℝ := sorry

theorem problem_statement
  (cond1 : ∀ {x y w : ℝ}, x > y → f x + x ≥ w → w ≥ f y + y → ∃ (z : ℝ), z ∈ Set.Icc y x ∧ f z = w - z)
  (cond2 : ∃ (u : ℝ), 0 ∈ Set.range f ∧ ∀ a ∈ Set.range f, u ≤ a)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 := sorry

end problem_statement_l380_380566


namespace incorrect_statement_l380_380784

theorem incorrect_statement :
  ¬ (∀ x, (x ≠ 1 ∧ x ≠ -1) ↔ (x ≠ 1 ∨ x ≠ -1)) ∧
  (∀ x y, (2 * (x + y) / 2 * x) = (x + y) / x) ∧
  (∀ x, (x + 2) / (|x| - 2) ≠ 0) ∧
  (∃ xs : List Int, xs.length = 4 ∧ ∀ x, 3 / (x + 1) ∈ xs) :=
sorry

end incorrect_statement_l380_380784


namespace hyperbola_equation_l380_380399

-- Definition of a hyperbola with given properties
def is_hyperbola_with_properties (C : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a = 1 ∧ b = 1 ∧ ∀ x y : ℝ, C x y ↔ y^2 - x^2 = 1

-- Lean 4 theorem statement
theorem hyperbola_equation :
  is_hyperbola_with_properties (λ x y, y^2 - x^2 = 1) :=
begin
  unfold is_hyperbola_with_properties,
  use [1, 1],
  split, { refl },
  split, { refl },
  intro x,
  intro y,
  split; intro h; exact h,
end

end hyperbola_equation_l380_380399


namespace benches_required_l380_380805

theorem benches_required (path_length : ℕ) (placement_interval : ℕ) (start_end_benches : ℕ) :
  path_length = 120 →
  placement_interval = 10 →
  start_end_benches = 2 →
  (path_length / placement_interval) + start_end_benches = 13 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end benches_required_l380_380805


namespace minimum_r_l380_380303

theorem minimum_r (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) (x : ℝ) (r : ℕ):
  (∀ n : ℕ+, a n > 0) →
  (∀ n : ℕ+, S n = ∑ i in finset.range n, a i) →
  (∀ n : ℕ+, 2 * S n = a n + a n ^ 2) →
  (∀ n : ℕ+, a n - a (n - 1) = 1) →
  (∀ n : ℕ+, b n = (log x)^n / (a n)^2) →
  (∀ n : ℕ+, T n = ∑ i in finset.range n, b i) →
  (1 < x ∧ x ≤ Real.exp 1) →
  (∀ n : ℕ+, T n < r) →
  r = 2 :=
by
  sorry

end minimum_r_l380_380303


namespace find_three_leaf_clovers_l380_380816

-- Define the conditions
def total_leaves : Nat := 1000

-- Define the statement
theorem find_three_leaf_clovers (n : Nat) (h : 3 * n + 4 = total_leaves) : n = 332 :=
  sorry

end find_three_leaf_clovers_l380_380816


namespace sum_of_largest_and_smallest_angles_l380_380749

theorem sum_of_largest_and_smallest_angles (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 8)
  (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ∃ θ α β : ℝ, 
  (cos θ = (b^2 + c^2 - a^2) / (2 * b * c)) ∧ 
  θ = real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) ∧ 
  (θ = 60) ∧ 
  (α + β = 120) :=
by sorry

end sum_of_largest_and_smallest_angles_l380_380749


namespace initial_observations_count_l380_380722

theorem initial_observations_count (S x n : ℕ) (h1 : S = 12 * n) (h2 : S + x = 11 * (n + 1)) (h3 : x = 5) : n = 6 :=
sorry

end initial_observations_count_l380_380722


namespace only_correct_option_is_C_l380_380069

-- Definitions of the conditions as per the given problem
def option_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def option_B (a : ℝ) : Prop := (a^2)^3 = a^5
def option_C (a b : ℝ) : Prop := (a * b)^3 = a^3 * b^3
def option_D (a : ℝ) : Prop := a^8 / a^2 = a^4

-- The theorem stating that only option C is correct
theorem only_correct_option_is_C (a b : ℝ) : 
  ¬(option_A a) ∧ ¬(option_B a) ∧ option_C a b ∧ ¬(option_D a) :=
by sorry

end only_correct_option_is_C_l380_380069


namespace emily_final_grade_min_l380_380485

theorem emily_final_grade_min (n : ℕ) (ahmed_grade : ℕ) (emily_grade : ℕ) 
  (final_assignment_same_weight : ∀ i, i ∈ finset.range (n + 1) → i = 100 → i > 0):
  (ahmed_grade = 91) →
  (emily_grade = 92) →
  let ahmed_total := ahmed_grade * n in
  let emily_total := emily_grade * n in
  ahmed_total + 100 ≤ emily_total + x →
  n = 9 →
  ∀ x : ℕ, x ≥ 92 :=
by
  assume h1 h2 h3,
  let ahmed_total := 91 * 9,
  let emily_total := 92 * 9,
  let total_points := 100 * n,
  let minimum_emily_grade := ahmed_total + 100 - 828,
  sorry

end emily_final_grade_min_l380_380485


namespace find_point_C_l380_380092

-- Define points A, B, and line c
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (6, 2)
def line_c (x : ℝ) : (ℝ × ℝ) := (x, 2 * x)

-- Define the point C we are trying to prove
def C : (ℝ × ℝ) := (2, 4)

-- The theorem stating the result
theorem find_point_C : 
  ∃ (C : ℝ × ℝ), C = (2, 4) ∧ 
   ∀ C' : ℝ × ℝ, C' ∈ (set_of (λ p : ℝ × ℝ, p.2 = 2 * p.1)) → 
   ¬((angle A C' B) > (angle A C B)) :=
begin
  use C,
  split,
  { refl, },
  {
    intros C' hC',
    have h1 : C' ∈ (set_of (λ p : ℝ × ℝ, p.2 = 2 * p.1)) := hC',
    sorry,
  }
end

end find_point_C_l380_380092


namespace find_m_l380_380935

def vector (α : Type) := α × α

def add_vectors : (vector ℝ) → (vector ℝ) → (vector ℝ)
| (x1, y1) (x2, y2) := (x1 + x2, y1 + y2)

def are_parallel (u v : vector ℝ) : Prop :=
u.1 * v.2 = u.2 * v.1

theorem find_m (m : ℝ):
  let a : vector ℝ := (2, -1)
  let b : vector ℝ := (-1, m)
  let c : vector ℝ := (-1, 2)
  (are_parallel (add_vectors a b) c) → m = -1 :=
sorry

end find_m_l380_380935


namespace log_addition_log_power_log_50_plus_log_8sq_l380_380129

theorem log_addition :
  ∀ (a x y : ℝ), a > 0 → a ≠ 1 →
  log a (x * y) = log a x + log a y :=
begin
  sorry
end

theorem log_power :
  ∀ (a x : ℝ) (b : ℕ), a > 0 → a ≠ 1 →
  log a (x ^ b) = b * log a x :=
begin
  sorry
end

theorem log_50_plus_log_8sq :
  log 10 50 + log 10 (8^2) = 5 * log 10 2 + 2 :=
begin
  have h1 : log 10 (50 * 8^2) = log 10 50 + log 10 (8^2),
    from log_addition 10 50 (8^2) (by norm_num) (by norm_num),
  have h2 : log 10 (8^2) = 2 * log 10 8,
    from log_power 10 8 2 (by norm_num) (by norm_num),
  have h3 : 50 * 8^2 = 3200,
    norm_num,
  have h4 : log 10 3200 = log 10 (32 * 10^2),
    norm_num,
  have h5 : log 10 (32 * 10^2) = log 10 32 + log 10 (10^2),
    from log_addition 10 32 (10^2) (by norm_num) (by norm_num),
  have h6 : log 10 (10^2) = 2,
    from log_power 10 10 2 (by norm_num) (by norm_num),
  have h7 : log 10 32 = 5 * log 10 2,
    from log_power 10 2 5 (by norm_num) (by norm_num),
  linarith,
end

end log_addition_log_power_log_50_plus_log_8sq_l380_380129


namespace find_n_tangent_eq_1234_l380_380865

theorem find_n_tangent_eq_1234 (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : Real.tan (n * Real.pi / 180) = Real.tan (1234 * Real.pi / 180)) : n = -26 := 
by 
  sorry

end find_n_tangent_eq_1234_l380_380865


namespace count_j_with_gj_eq_l380_380964

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).sum id

def integer_square_root (n : ℕ) : ℕ :=
  Nat.sqrt n

theorem count_j_with_gj_eq (g : ℕ → ℕ) (h : ∀ n, g n = sum_of_divisors n) :
  (Finset.range 4097).filter (λ j, g j = 1 + j + integer_square_root j).card = 18 :=
  by
    sorry

end count_j_with_gj_eq_l380_380964


namespace parallelogram_height_same_area_l380_380467

noncomputable def rectangle_area (length width : ℕ) : ℕ := length * width

theorem parallelogram_height_same_area (length width base height : ℕ) 
  (h₁ : rectangle_area length width = base * height) 
  (h₂ : length = 12) 
  (h₃ : width = 6) 
  (h₄ : base = 12) : 
  height = 6 := 
sorry

end parallelogram_height_same_area_l380_380467


namespace problem1_solution_set_problem2_range_l380_380918

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (x^2 - 4 * x + 3 + a) / (x - 1)

-- Problem 1: When a = 2, the solution set of the inequality f(x) ≥ 1 is (1, 2] ∪ [3, +∞)
theorem problem1_solution_set (x : ℝ) : (f x 2 ≥ 1) ↔ ((1 < x ∧ x ≤ 2) ∨ (3 ≤ x)) ∧ x ≠ 1 := sorry

-- Problem 2: When a < 0, the range of the function f(x) for x in (1, 3] is (-∞, a / 2]
theorem problem2_range (a : ℝ) (h : a < 0) : (∀ x, 1 < x ∧ x ≤ 3 → (f x a) ∈ (-∞, a / 2]) := sorry

end problem1_solution_set_problem2_range_l380_380918


namespace inequality_solution_set_l380_380178

theorem inequality_solution_set (x : ℝ) : 4 * x^2 - 4 * x + 1 ≥ 0 := 
by
  sorry

end inequality_solution_set_l380_380178


namespace sum_of_length_differences_l380_380719

theorem sum_of_length_differences 
  (area_A : ℕ) (area_B : ℕ) (area_C : ℕ)
  (hA : area_A = 25) (hB : area_B = 81) (hC : area_C = 64) :
  ((nat.sqrt area_B - nat.sqrt area_A) + (nat.sqrt area_B - nat.sqrt area_C) = 5) :=
by
  sorry

end sum_of_length_differences_l380_380719


namespace increase_in_green_chameleons_is_11_l380_380316

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380316


namespace number_of_students_playing_both_l380_380637

-- Definitions based on conditions
def total_students : ℕ := 30

def plays_neither(n : ℕ) : ℕ := n

def plays_football(n : ℕ) : ℕ := 2 * n

def plays_basketball(n : ℕ) : ℕ := 4 * n

def plays_both(x : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 5 ∧ n ≤ 6 ∧ (7 * n - 30 = x) ∧ (2 * n - x ≥ 0) ∧ (x = 5)

theorem number_of_students_playing_both (x : ℕ) : plays_both(x) → x = 5 := 
  sorry

end number_of_students_playing_both_l380_380637


namespace parallel_vectors_lambda_l380_380613

def vector_a : (ℝ × ℝ × ℝ) := (2, -3, 5)
def vector_b (λ : ℝ) : (ℝ × ℝ × ℝ) := (3, λ, 15 / 2)

theorem parallel_vectors_lambda (λ : ℝ) (h : ∃ k : ℝ, vector_b λ = (k • vector_a)) : λ = -9 / 2 :=
by
  sorry

end parallel_vectors_lambda_l380_380613


namespace meeting_point_l380_380294

open_locale classical
noncomputable theory

-- Definitions of speeds of Jane and Hector
variables (s : ℝ) -- Hector's speed
noncomputable def jane_speed := 3 * s -- Jane's speed

-- Time taken until they meet
noncomputable def time_to_meet : ℝ := 24 / (4 * s)

-- Distances walked
noncomputable def hector_distance : ℝ := s * time_to_meet
noncomputable def jane_distance : ℝ := 3 * s * time_to_meet

-- Position after walking the distances
def hector_position := hector_distance % 24
def jane_position := (24 - jane_distance % 24) % 24

-- Given point when they meet
constant point_A point_E : ℝ

-- Specification that determines their respective distances from point A and E
axiom initial_positions : hector_position = 0 ∧ jane_position = 24

-- Meeting point condition
theorem meeting_point : hector_position % 24 = point_E % 24 :=
by sorry

end meeting_point_l380_380294


namespace no_positive_x_for_volume_l380_380075

noncomputable def volume (x : ℤ) : ℤ :=
  (x + 5) * (x - 7) * (x^2 + x + 30)

theorem no_positive_x_for_volume : ¬ ∃ x : ℕ, 0 < x ∧ volume x < 800 := by
  sorry

end no_positive_x_for_volume_l380_380075


namespace length_of_goods_train_l380_380799

theorem length_of_goods_train 
  (train_speed_kmph : ℕ)
  (platform_length_m : ℕ)
  (time_to_cross_sec : ℕ) 
  (conversion_factor : ℚ)
  (train_speed_m_per_s : ℚ)
  (distance_covered_m : ℚ)
  (train_length_m : ℚ) :
  train_speed_kmph = 72 →
  platform_length_m = 250 →
  time_to_cross_sec = 26 →
  conversion_factor = 5 / 18 →
  train_speed_m_per_s = train_speed_kmph * conversion_factor →
  distance_covered_m = train_speed_m_per_s * time_to_cross_sec →
  train_length_m = distance_covered_m - platform_length_m →
  train_length_m = 270 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  have h8 : train_speed_m_per_s = 72 * (5 / 18), 
  { rw h1, rw h4 },
  have h9 : distance_covered_m = (72 * (5 / 18)) * 26,
  { rw h6, rw h8 },
  have h10 : distance_covered_m = 20 * 26,
  { norm_num at h9, exact_mod_cast h8, rw ←h9 },  
  have h11 : distance_covered_m = 520,
  { norm_num at h10 },
  have h12 : train_length_m = 520 - 250,
  { rw h7, rw h11, exact_mod_cast h2 },
  norm_num at h12,
  exact h12,
end

end length_of_goods_train_l380_380799


namespace find_circle_equation_l380_380198

noncomputable def center_of_circle_on_line (x y : ℝ) : Prop := x - y + 1 = 0

noncomputable def point_on_circle (x_c y_c r x y : ℝ) : Prop := (x - x_c)^2 + (y - y_c)^2 = r^2

theorem find_circle_equation 
  (x_A y_A x_B y_B : ℝ)
  (hA : x_A = 1 ∧ y_A = 1)
  (hB : x_B = 2 ∧ y_B = -2)
  (h_center_on_line : ∃ x_c y_c, center_of_circle_on_line x_c y_c ∧ point_on_circle x_c y_c r 1 1 ∧ point_on_circle x_c y_c r 2 (-2))
  : ∃ (x_c y_c r : ℝ), (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end find_circle_equation_l380_380198


namespace digit_150_is_7_l380_380041

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380041


namespace hyperbola_eccentricity_proof_l380_380890

noncomputable def hyperbola_eccentricity {a b : ℝ} (x y : ℝ) (P F₁ F₂ : ℝ × ℝ) : ℝ :=
  if a > 0 ∧ b > 0 ∧ ( (frac (x^2) (a^2)) - (frac (y^2) (b^2)) = 1 ) ∧ ( |P - F₁| = 3 * |P - F₂| ) 
  ∧ ( (P - F₁) • (P - F₂) = -a^2 )
  then sqrt 3 else 0

-- Test to check if the property holds under the given conditions
theorem hyperbola_eccentricity_proof {a b : ℝ} {x y : ℝ} {P F₁ F₂ : ℝ × ℝ}
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : ( (frac (x^2) (a^2)) - (frac (y^2) (b^2)) = 1 ))
  (h4 : ( |P - F₁| = 3 * |P - F₂| ))
  (h5 : ( (P - F₁) • (P - F₂) = -a^2 )) :
  hyperbola_eccentricity x y P F₁ F₂ = sqrt 3 :=
begin
  sorry
end

end hyperbola_eccentricity_proof_l380_380890


namespace incorrect_option_C_l380_380661

def angles_of_triangle_sum_to_pi (A B C : ℝ) (h : A + B + C = Real.pi) : Prop :=
  A + B + C = Real.pi

def triangle_properties (A B C : ℝ) : Prop :=
  angles_of_triangle_sum_to_pi A B C (by sorry)

theorem incorrect_option_C (A B C : ℝ) (h : triangle_properties A B C) : 
  ¬ (cos (B + C) = cos A) :=
by sorry

end incorrect_option_C_l380_380661


namespace no_digit_satisfies_equations_l380_380074

-- Define the conditions as predicates.
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x < 10

-- Formulate the proof problem based on the given problem conditions and conclusion
theorem no_digit_satisfies_equations : 
  ¬ (∃ x : ℤ, is_digit x ∧ (x - (10 * x + x) = 801 ∨ x - (10 * x + x) = 812)) :=
by
  sorry

end no_digit_satisfies_equations_l380_380074


namespace total_insect_legs_l380_380389

/--
This Lean statement defines the conditions and question,
proving that given 5 insects in the laboratory and each insect
having 6 legs, the total number of insect legs is 30.
-/
theorem total_insect_legs (n_insects : Nat) (legs_per_insect : Nat) (h1 : n_insects = 5) (h2 : legs_per_insect = 6) : (n_insects * legs_per_insect) = 30 :=
by
  sorry

end total_insect_legs_l380_380389


namespace gcd_sum_91_l380_380553

theorem gcd_sum_91 :
  (∑ n in Finset.range 92, Int.gcd n 91) = 325 := by
sorry

end gcd_sum_91_l380_380553


namespace evaluate_complex_magnitude_l380_380856

def magnitude (z : Complex) : ℝ := Complex.abs z

theorem evaluate_complex_magnitude :
  magnitude (Complex.mk 3 (-5)) - magnitude (Complex.mk 3 5) = 0 := by
  sorry

end evaluate_complex_magnitude_l380_380856


namespace slope_range_l380_380932

def line_l_eq (k : ℝ) : (ℝ × ℝ) → Prop := 
  λ p, p.2 = k * p.1 - k - 1

def line_m_eq (k : ℝ) : (ℝ × ℝ) → Prop := 
  λ p, p.2 = k * p.1 - 4 * k + 1

def line_m_in_fourth_quadrant (k : ℝ) : Prop := 
  ∃ x y : ℝ, line_m_eq k (x, y) ∧ x > 0 ∧ y < 0

theorem slope_range (k : ℝ) (h : ∀ x y, line_l_eq k (x, y) ↔ y = k * x - k - 1) :
  ¬ (line_m_in_fourth_quadrant k) → 0 ≤ k ∧ k ≤ (1 / 4) :=
by
  sorry

end slope_range_l380_380932


namespace Olya_school_journey_time_l380_380314

theorem Olya_school_journey_time:
  ∀ (t : ℕ),
  let d := (1 / 5 : ℝ) ∧ 
  let x := (4 / 5 : ℝ) in
  ((t - 6) = t - 6) ∧ 
  ((t + 2) = t + 2) ∧ 
  ((9 / 5) * t = (8 + t - t + 2 - 6)) →
  t = 20 :=
begin
  sorry
end

end Olya_school_journey_time_l380_380314


namespace correct_statement_is_C_l380_380826

-- Definitions of the conditions
def mode_data_claim : Prop := 
    mode [5, 4, 4, 3, 5, 2] = 4

def std_dev_square_of_variance_claim : Prop := 
    ∀ (s : List ℝ), std_dev s = (variance s) ^ 2

def std_dev_half_claim : Prop :=
    std_dev [2, 3, 4, 5] = 1 / 2 * std_dev [4, 6, 8, 10]

def freq_distribution_histogram_claim : Prop :=
    ∀ (histogram : List (ℝ × ℝ)), 
    ∀ (rectangle : ℝ × ℝ), 
    rectangle ∈ histogram →
    area_of_rectangle rectangle = frequency_of_group rectangle

-- Problem statement to be proved
theorem correct_statement_is_C : std_dev_half_claim :=
by
  sorry

end correct_statement_is_C_l380_380826


namespace vector_forms_new_basis_l380_380902

open RealEuclideanSpace

noncomputable def a : RealEuclideanSpace := sorry  
noncomputable def b : RealEuclideanSpace := sorry
noncomputable def c : RealEuclideanSpace := sorry

def p := a + b
def q := a - b

theorem vector_forms_new_basis :
  LinearlyIndependent ℝ ![a, b, c] →
  LinearlyIndependent ℝ ![p, q, a + 2 * c] :=
sorry

end vector_forms_new_basis_l380_380902


namespace train_crossing_time_l380_380478

variable (L_train : ℝ) (L_bridge : ℝ) (Speed_train_kmh : ℝ)

noncomputable def Speed_train_ms := Speed_train_kmh * (1000 / 3600)
noncomputable def Total_distance := L_train + L_bridge
noncomputable def Time_to_cross := Total_distance / Speed_train_ms

theorem train_crossing_time 
  (h_train_length : L_train = 170)
  (h_bridge_length : L_bridge = 205)
  (h_train_speed : Speed_train_kmh = 45) :
  Time_to_cross L_train L_bridge Speed_train_kmh = 30 :=
by
  rw [h_train_length, h_bridge_length, h_train_speed]
  simp [Speed_train_ms, Total_distance, Time_to_cross]
  sorry

end train_crossing_time_l380_380478


namespace opposite_angles_equal_l380_380404

theorem opposite_angles_equal {a b : ℝ} (h : ∠a + ∠b = 180) : ∠a = ∠b :=
sorry

end opposite_angles_equal_l380_380404


namespace exists_constant_C_l380_380893

noncomputable def sequence_property (x : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, (∑ i in finset.range (n + 1), x i) ≥ 2 * (∑ i in finset.range n, x i)

theorem exists_constant_C (x : ℕ → ℝ) (hpos : ∀ n, x n > 0) (hs : sequence_property x) :
  ∃ C > 0, ∀ n, x n ≥ C * 2^n :=
sorry

end exists_constant_C_l380_380893


namespace simultaneous_equations_solution_l380_380147

theorem simultaneous_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 20) ∧ (9 * x - 8 * y = 36) ∧ (x = 76 / 15) ∧ (y = 18 / 15) :=
by
  sorry

end simultaneous_equations_solution_l380_380147


namespace arithmetic_sequence_seventh_term_l380_380846

noncomputable def a3 := (2 : ℚ) / 11
noncomputable def a11 := (5 : ℚ) / 6

noncomputable def a7 := (a3 + a11) / 2

theorem arithmetic_sequence_seventh_term :
  a7 = 67 / 132 := by
  sorry

end arithmetic_sequence_seventh_term_l380_380846


namespace digit_150th_of_17_div_70_is_7_l380_380020

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380020


namespace sum_f_1_to_100_l380_380917

noncomputable def f (n : ℕ) : ℝ := Real.tan (n * Real.pi / 3)

theorem sum_f_1_to_100 : (∑ n in Finset.range 100, f (n + 1)) = Real.sqrt 3 := by
  sorry

end sum_f_1_to_100_l380_380917


namespace partnership_total_gain_l380_380110

theorem partnership_total_gain (x : ℝ) (A_share : ℝ) :
  A_share = 6100 →
  let B_share := 2 * x * 6 in
  let C_share := 3 * x * 4 in
  let total_gain := A_share + B_share + C_share in
  total_gain = 18300 := by
sorry

end partnership_total_gain_l380_380110


namespace directrix_of_parabola_l380_380174

-- Define the given conditions
def parabola_eqn (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 5

-- The problem is to show that the directrix of this parabola has the equation y = 23/12
theorem directrix_of_parabola : 
  (∃ y : ℝ, ∀ x : ℝ, parabola_eqn x = y) →

  ∃ y : ℝ, y = 23 / 12 :=
sorry

end directrix_of_parabola_l380_380174


namespace sin_theta_eq_half_l380_380883

noncomputable def z (u v : ℝ) : ℂ := u + complex.I * v

theorem sin_theta_eq_half (u v θ : ℝ) (h : u^2 * (1 / real.cos θ)^2 - v^2 * (1 / real.sin θ)^2 = 1) :
  real.sin θ = (1 / 2) * (1 - complex.abs (z u v)^2 + complex.abs (z u v)^2 - 1) :=
sorry

end sin_theta_eq_half_l380_380883


namespace KM_perp_KD_l380_380683

variables {Point : Type} [MetricSpace Point] 

-- Points and Line Segments Definitions
variable (L M K B D C : Point)

-- Conditions
variable (h1 : is_midpoint L C D)
variable (h2 : is_median M L C D)
variable (h3 : is_perpendicular M L C D)
variable (h4 : is_parallel B K C D) 
variable (h5 : is_perpendicular L B K)
variable (h6 : is_parallel K L A D)
variable (h7 : is_perpendicular B M K L)
variable (h8 : is_orthocenter M K L B)
variable (h9 : is_parallel B L K D)

-- Final Proof Target
theorem KM_perp_KD : is_perpendicular K M K D := 
by
  sorry

end KM_perp_KD_l380_380683


namespace digit_150_of_17_div_70_l380_380027

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380027


namespace count_integers_divisible_by_13_and_12_l380_380252

theorem count_integers_divisible_by_13_and_12 (a b m n : ℤ) (h1 : a = 200) (h2 : b = 500) (h3 : m = 13) (h4 : n = 12) :
  (set.count (λ x, a ≤ x ∧ x ≤ b ∧ x % (nat.lcm m n) = 0) (set.Icc a b) = 2) :=
by
  rw [h1, h2, h3, h4]
  sorry

end count_integers_divisible_by_13_and_12_l380_380252


namespace find_a_l380_380584

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  x * (x + 1)
else
  -((-x) * ((-x) + 1))

theorem find_a (a : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, x >= 0 → f x = x * (x + 1)) (h_a: f a = -2) : a = -1 :=
sorry

end find_a_l380_380584


namespace partial_fraction_decomposition_l380_380172

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 ∧ x^2 ≠ 1 →
  (x^2 + 5 * x - 2) / (x^3 - x) = (2 / x) + ((-x + 5) / (x^2 - 1)) :=
by
  intro x hx,
  have h₁ : x ≠ 0 := hx.1,
  have h₂ : x^2 - 1 ≠ 0 := hx.2,
  sorry

end partial_fraction_decomposition_l380_380172


namespace h2o_production_l380_380849

def balanced_reaction := 
    ∀ (NH4Cl NaOH NH3 H2O NaCl : ℕ), 
        NH4Cl = NaOH ∧ NH4Cl = NH3 ∧ 
        NH4Cl = H2O ∧ NH4Cl = NaCl

theorem h2o_production (NH4Cl NaOH HCl NH3 H2O NaCl : ℕ) 
    (h₁ : NH4Cl = 2) 
    (h₂ : NaOH = 3) 
    (h₃ : HCl = 1)
    (h_balanced : balanced_reaction NH4Cl NaOH NH3 H2O NaCl) :
    H2O = 2 := 
by 
    --Proof goes here
    sorry

end h2o_production_l380_380849


namespace sum_of_roots_eq_five_l380_380908

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_roots_eq_five 
  (h1 : ∀ x : ℝ, f(x) = f(2 - x))
  (h2 : ∃ roots : finset ℝ, roots.card = 5 ∧ ∀ r ∈ roots, f r = 0 ∧ ∀ i j ∈ roots, i ≠ j → i ≠ j) : 
  (roots : finset ℝ) -> ∑ r in roots, r = 5 := 
sorry

end sum_of_roots_eq_five_l380_380908


namespace green_chameleon_increase_l380_380328

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380328


namespace noStraightFlush_waysToDraw13Cards_l380_380797

-- Definitions from the conditions
def numCards := 52
def suits := 4
def ranks := 13

def consecutive (a b : Nat) : Prop :=
(a = 1 ∧ b = 13) ∨ (b = 1 ∧ a = 13) ∨ (a + 1 = b) ∨ (b + 1 = a)

def waysToDraw13Cards : Nat :=
3^13 - 3

-- Proof statement
theorem noStraightFlush_waysToDraw13Cards :
  ∀ n ≥ 3, (∑ (k : ℕ) in Finset.range n, x (n - k.succ)) + x (n - 1) = 4 * 3^(n - 1)
    → x 13 = 3^13 - 3
    → waysToDraw13Cards = 3^13 - 3 :=
by
  sorry

end noStraightFlush_waysToDraw13Cards_l380_380797


namespace rectangle_problem_l380_380391

def rectangle_perimeter (L B : ℕ) : ℕ :=
  2 * (L + B)

theorem rectangle_problem (L B : ℕ) (h1 : L - B = 23) (h2 : L * B = 2520) : rectangle_perimeter L B = 206 := by
  sorry

end rectangle_problem_l380_380391


namespace martha_no_daughters_count_l380_380691

-- Definitions based on conditions
def total_people : ℕ := 40
def martha_daughters : ℕ := 8
def granddaughters_per_child (x : ℕ) : ℕ := if x = 1 then 8 else 0

-- Statement of the problem
theorem martha_no_daughters_count : 
  (total_people - martha_daughters) +
  (martha_daughters - (total_people - martha_daughters) / 8) = 36 := 
  by
    sorry

end martha_no_daughters_count_l380_380691


namespace digit_150_is_7_l380_380037

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380037


namespace traveled_distance_l380_380832

def distance_first_day : ℕ := 5 * 7
def distance_second_day_part1 : ℕ := 6 * 6
def distance_second_day_part2 : ℕ := (6 / 2) * 3
def distance_third_day : ℕ := 7 * 5

def total_distance : ℕ := distance_first_day + distance_second_day_part1 + distance_second_day_part2 + distance_third_day

theorem traveled_distance : total_distance = 115 := by
  unfold total_distance
  unfold distance_first_day distance_second_day_part1 distance_second_day_part2 distance_third_day
  norm_num
  rfl

end traveled_distance_l380_380832


namespace polar_eq_E3_at_pi_over_6_common_point_E3_E2_l380_380604

-- Condition: Polar equations of curves E1 and E2
def polar_eq_E1 (θ : ℝ) : ℝ := 4 * Real.cos θ
def polar_eq_E2 (ρ θ : ℝ) : Prop := ρ * Real.cos (θ - (Real.pi / 4)) = 4

-- Function to get the polar equation of rotated curve E1, creating curve E3
def polar_eq_E3_rotated (α ρ θ : ℝ) : Prop := 
  ρ = 4 * Real.cos (θ - α)

-- Assertion 1: When α = π/6, polar equation of E3 is ρ = 4cos(θ - π/6)
theorem polar_eq_E3_at_pi_over_6 : 
  polar_eq_E3_rotated (Real.pi / 6) = λ ρ θ, ρ = 4 * Real.cos (θ - Real.pi / 6) := 
by
  sorry

-- Function to check if E3 and E2 have exactly one common point given α
def E3_E2_tangent (α : ℝ) : ℝ := 
  2 * (Real.cos α + Real.sin α) - 4 * Real.sqrt 2

-- Assertion 2: When E3 and E2 have exactly one common point, α = π/4
theorem common_point_E3_E2 : 
  E3_E2_tangent (Real.pi / 4) = 0 :=
by
  sorry

end polar_eq_E3_at_pi_over_6_common_point_E3_E2_l380_380604


namespace count_pos_three_digit_ints_with_same_digits_l380_380949

-- Define a structure to encapsulate the conditions for a three-digit number less than 700 with at least two digits the same.
structure valid_int (n : ℕ) : Prop :=
  (three_digit : 100 ≤ n ∧ n < 700)
  (same_digits : ∃ d₁ d₂ d₃ : ℕ, ((100 * d₁ + 10 * d₂ + d₃ = n) ∧ (d₁ = d₂ ∨ d₂ = d₃ ∨ d₁ = d₃)))

-- The number of integers satisfying the conditions
def count_valid_ints : ℕ :=
  168

-- The theorem to prove
theorem count_pos_three_digit_ints_with_same_digits : 
  (∃ n, valid_int n) → 168 :=
by
  -- Since the proof is not required, we add sorry here.
  sorry

end count_pos_three_digit_ints_with_same_digits_l380_380949


namespace sky_color_changes_l380_380519

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l380_380519


namespace num_divisors_of_factorial_9_multiple_3_l380_380848

-- Define the prime factorization of 9!
def factorial_9 := 2^7 * 3^4 * 5 * 7

-- Define the conditions for the exponents a, b, c, d
def valid_exponents (a b c d : ℕ) : Prop :=
  (0 ≤ a ∧ a ≤ 7) ∧ (1 ≤ b ∧ b ≤ 4) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1)

-- Define the number of valid exponent combinations
def num_valid_combinations : ℕ :=
  8 * 4 * 2 * 2

-- Theorem stating that the number of divisors of 9! that are multiples of 3 is 128
theorem num_divisors_of_factorial_9_multiple_3 : num_valid_combinations = 128 := by
  sorry

end num_divisors_of_factorial_9_multiple_3_l380_380848


namespace unique_function_l380_380860

theorem unique_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, f(2 * m + 2 * n) = f(m) * f(n)) : 
  ∀ x : ℕ, f(x) = 1 := by
  sorry

end unique_function_l380_380860


namespace chameleon_problem_l380_380343

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380343


namespace movie_preferences_related_to_gender_expected_value_X_equals_12_7_l380_380828

def contingency_table_data : Type := {m_dom : ℕ, m_for : ℕ, f_dom : ℕ, f_for : ℕ, total : ℕ}

def survey_data : contingency_table_data := {m_dom := 60, m_for := 40, f_dom := 80, f_for := 20, total := 200}

def chi_square_test (data : contingency_table_data) (alpha : ℝ) : Prop := 
  let n := data.total
  let a := data.m_dom
  let b := data.f_dom
  let c := data.m_for
  let d := data.f_for
  let chi_square := (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))
  chi_square > 7.879

theorem movie_preferences_related_to_gender : chi_square_test survey_data 0.005 := 
by {
  -- We need to compute chi_square_test to see the result. We will skip the proof here.
  sorry
}

def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ 
| k => (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def expected_value_binomial (n : ℕ) (p : ℝ) : ℝ :=
  ∑ i in finset.range (n+1), (i * (binomial_distribution n p i))

def random_variable_X_distribution : finset (ℕ × ℝ) := 
  finset.range 4 |>.map (λ i, (i, binomial_distribution 3 (4/7) i))

def expected_value_X : ℝ := expected_value_binomial 3 (4 / 7)

theorem expected_value_X_equals_12_7 : expected_value_X = 12 / 7 := 
by {
  -- Calculations should be shown here to prove that expected value of X equals 12/7. We will skip the proof here.
  sorry
}

end movie_preferences_related_to_gender_expected_value_X_equals_12_7_l380_380828


namespace arithmetic_sequence_common_difference_l380_380653

variable (a : ℕ → ℝ)

-- Conditions
axiom a6_eq_5 : a 6 = 5
axiom a10_eq_6 : a 10 = 6

-- Common difference definition in terms of arithmetic sequence
def common_difference (a : ℕ → ℝ) : ℝ :=
  (a 10 - a 6) / (10 - 6)

-- Main theorem
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) :
  a 6 = 5 → a 10 = 6 → d = common_difference a → d = (1 / 4) :=
by
  intro h1 h2 h3
  -- Proof omitted
  sorry

end arithmetic_sequence_common_difference_l380_380653


namespace function_properties_l380_380225

def f (x : ℝ) : ℝ := x * (Real.exp x + Real.exp (-x))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) := sorry

end function_properties_l380_380225


namespace digit_150th_in_decimal_of_fraction_l380_380047

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380047


namespace bisector_construction_l380_380431

theorem bisector_construction (A O B C M : Point) (a : ℝ) 
  (hAcute : acute_angle A O B)
  (hParallel1 : parallel (line_through B C) (line_through O M))
  (hDistEqual : distance_axiom a B C O M) :
  angle_bisector A O C = ray_through O A :=
sorry

end bisector_construction_l380_380431


namespace max_value_x_plus_y_max_value_x_plus_y_achieved_l380_380771

theorem max_value_x_plus_y (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : x + y ≤ 6 * Real.sqrt 5 :=
by
  sorry

theorem max_value_x_plus_y_achieved (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : ∃ x y, x + y = 6 * Real.sqrt 5 :=
by
  sorry

end max_value_x_plus_y_max_value_x_plus_y_achieved_l380_380771


namespace james_tax_deduction_l380_380986

theorem james_tax_deduction :
  let w := 25
  let r := 0.024
  let wage_in_cents := w * 100
  let tax_deduction_in_cents := r * wage_in_cents
  tax_deduction_in_cents = 60 :=
by
  let w := 25
  let r := 0.024
  let wage_in_cents := w * 100
  let tax_deduction_in_cents := r * wage_in_cents
  show tax_deduction_in_cents = 60
  sorry

end james_tax_deduction_l380_380986


namespace parallelogram_perimeter_l380_380635

-- Conditions for the problem
variables (P Q R S T U : Type)
variables [Segment PQ PR QR PQ_PR PQ_QR PR_PQ : MeasureSpace P Q R S T U]
variables (h1 : PQ = 17) (h2 : PR = 17) (h3 : QR = 16)
variables (ST PR U S T U : SegmentChoice P Q R S T U GRP UNIV_SYM UNIV.Aux)
variables (ST_parallel_PR : parallel ST PR)
variables (TU_parallel_PQ : parallel TU PQ)

-- Theorem to prove the perimeter of parallelogram PSTU
theorem parallelogram_perimeter (P Q R S T U : Type) :
  perimeter (parallelogram P S T U) = 34 :=
sorry

end parallelogram_perimeter_l380_380635


namespace count_divisible_2_3_or_5_lt_100_l380_380253
-- We need the Mathlib library for general mathematical functions

-- The main theorem statement
theorem count_divisible_2_3_or_5_lt_100 : 
  let A2 := Nat.floor (100 / 2)
  let A3 := Nat.floor (100 / 3)
  let A5 := Nat.floor (100 / 5)
  let A23 := Nat.floor (100 / 6)
  let A25 := Nat.floor (100 / 10)
  let A35 := Nat.floor (100 / 15)
  let A235 := Nat.floor (100 / 30)
  (A2 + A3 + A5 - A23 - A25 - A35 + A235) = 74 :=
by
  sorry

end count_divisible_2_3_or_5_lt_100_l380_380253


namespace number_of_valid_pairs_l380_380145

theorem number_of_valid_pairs :
  (∃ (count : ℕ), count = 280 ∧
    (∃ (m n : ℕ),
      1 ≤ m ∧ m ≤ 2899 ∧
      5^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 5^(n+1))) :=
sorry

end number_of_valid_pairs_l380_380145


namespace larger_of_two_solutions_l380_380538

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l380_380538


namespace exists_fixed_point_A4_l380_380278

-- Definitions for Part 1
def A1 := (1 : ℝ, 0 : ℝ)
def A2 := (-2 : ℝ, 0 : ℝ)

lemma locus_of_M (M : ℝ × ℝ) :
  (real.sqrt ((M.1 - A1.1) ^ 2 + M.2 ^ 2) / real.sqrt ((M.1 + 2) ^ 2 + M.2 ^ 2) = real.sqrt 2 / 2)
  ↔ (M.1^2 + M.2^2 - 8 * M.1 - 2 = 0) :=
sorry

-- Definitions for Part 2
def circle_N (x y : ℝ) := (x-3)^2 + y^2 = 4
def A3 := (-1 : ℝ, 0 : ℝ)
def ratio_condition (N A4 : ℝ × ℝ) := real.sqrt ((N.1 + 1) ^ 2 + N.2 ^ 2) / real.sqrt ((N.1 - A4.1) ^ 2 + (N.2 - A4.2) ^ 2) = 2

theorem exists_fixed_point_A4 (N : ℝ × ℝ) (hN : circle_N N.1 N.2) :
  ∃ A4 : ℝ × ℝ, ratio_condition N A4 ∧ A4 = (2, 0) :=
sorry

end exists_fixed_point_A4_l380_380278


namespace smallest_k_for_64k_greater_than_6_l380_380420

theorem smallest_k_for_64k_greater_than_6 : ∃ (k : ℕ), 64 ^ k > 6 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 6 :=
by
  use 1
  sorry

end smallest_k_for_64k_greater_than_6_l380_380420


namespace find_x_l380_380260

theorem find_x (x y : ℝ) (h₁ : x - y = 10) (h₂ : x + y = 14) : x = 12 :=
by
  sorry

end find_x_l380_380260


namespace maximum_value_of_y_over_x_minimum_value_of_y_minus_x_maximum_value_of_x2_plus_y2_l380_380579

noncomputable def problem_statement (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + 1 = 0

theorem maximum_value_of_y_over_x (x y : ℝ) (h : problem_statement x y) :
  ∃ θ : ℝ, θ ∈ set.Icc 0 (2 * π) ∧ 
           (x = 2 + real.sqrt 3 * real.cos θ) ∧ 
           (y = real.sqrt 3 * real.sin θ) ∧ 
           real.sqrt 3 = y / x :=
begin
  sorry
end

theorem minimum_value_of_y_minus_x (x y : ℝ) (h : problem_statement x y) :
  ∃ θ : ℝ, θ ∈ set.Icc 0 (2 * π) ∧ 
           (x = 2 + real.sqrt 3 * real.cos θ) ∧ 
           (y = real.sqrt 3 * real.sin θ) ∧
           y - x = -real.sqrt 6 - 2 :=
begin
  sorry
end

theorem maximum_value_of_x2_plus_y2 (x y : ℝ) (h : problem_statement x y) :
  ∃ θ : ℝ, θ ∈ set.Icc 0 (2 * π) ∧ 
           (x = 2 + real.sqrt 3 * real.cos θ) ∧ 
           (y = real.sqrt 3 * real.sin θ) ∧
           x^2 + y^2 = 7 + 4 * real.sqrt 3 :=
begin
  sorry
end

end maximum_value_of_y_over_x_minimum_value_of_y_minus_x_maximum_value_of_x2_plus_y2_l380_380579


namespace maximize_profit_l380_380812

def P (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 30 then 1800
  else if 30 < x ∧ x ≤ 75 then -20 * x + 2400
  else 0

def y (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 30 then 1800 * x - 30000
  else if 30 < x ∧ x ≤ 75 then -20 * x * x + 2400 * x - 30000
  else 0

theorem maximize_profit :
  (∃ x : ℕ, 1 ≤ x ∧ x ≤ 75 ∧ ∀ y : ℕ, 1 ≤ y ∧ y ≤ 75 → y y ≤ y x) ∧
  y 60 = 42000 :=
sorry

end maximize_profit_l380_380812


namespace chameleon_problem_l380_380344

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380344


namespace sufficient_condition_l380_380881

variable (a x : ℝ)

def p : Prop := x^2 - 8 * x - 20 < 0
def q : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0

theorem sufficient_condition (h : p → q) (h_not_necessary : q → p → False) (ha : a > 0) : a ≥ 9 :=
  sorry

end sufficient_condition_l380_380881


namespace mp_dot_op_range_l380_380565

theorem mp_dot_op_range :
  ∀ (P : ℝ × ℝ), (P.1^2 + P.2^2 = 4) → 
    let M := (0 : ℝ, 4 : ℝ)
    let MP := (P.1, P.2 - 4)
    let OP := (P.1, P.2)
    ∃ y, y = MP.1 * OP.1 + MP.2 * OP.2 ∧ y ∈ Icc (-4) 12 :=
by {
  sorry
}

end mp_dot_op_range_l380_380565


namespace jane_reading_days_l380_380666

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end jane_reading_days_l380_380666


namespace malcolm_red_lights_bought_l380_380690

-- Define the problem's parameters and conditions
variable (R : ℕ) (B : ℕ := 3 * R) (G : ℕ := 6)
variable (initial_white_lights : ℕ := 59) (remaining_colored_lights : ℕ := 5)

-- The total number of colored lights that he still needs to replace the white lights
def total_colored_lights_needed : ℕ := initial_white_lights - remaining_colored_lights

-- Total colored lights bought so far
def total_colored_lights_bought : ℕ := R + B + G

-- The main theorem to prove that Malcolm bought 12 red lights
theorem malcolm_red_lights_bought (h : total_colored_lights_bought = total_colored_lights_needed) :
  R = 12 := by
  sorry

end malcolm_red_lights_bought_l380_380690


namespace sum_of_x_coordinates_of_P_l380_380758

-- Definitions and problem statement
def area_triangle (A B C : Point) : ℝ := 
  ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)) / 2

structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def Q : Point := ⟨0, 0⟩
def R : Point := ⟨368, 0⟩
def S1 : Point := ⟨901, 501⟩
def S2 : Point := ⟨912, 514⟩

def area_PQR (P : Point) : ℝ := area_triangle P Q R
def area_PRS1 (P : Point) : ℝ := area_triangle P R S1
def area_PRS2 (P : Point) : ℝ := area_triangle P R S2

axiom P_position (P : Point) :
  area_PQR P = 4128 ∧ (area_PRS1 P = 12384 ∨ area_PRS2 P = 12384) 

-- Theorem statement
theorem sum_of_x_coordinates_of_P : 
  ∃ (px_sum : ℝ), (∀ P, P_position P → true) ∧ px_sum = 4000 := 
sorry

end sum_of_x_coordinates_of_P_l380_380758


namespace nonneg_int_solutions_eq_count_l380_380941

theorem nonneg_int_solutions_eq_count (x : ℕ) : (x^3 = -6 * x) ↔ (x = 0) := sorry

end nonneg_int_solutions_eq_count_l380_380941


namespace number_of_true_propositions_is_2_l380_380825

theorem number_of_true_propositions_is_2 :
  let P1 := ∀ (a b c x : ℝ), a ≠ 0 → (a * x^2 + b * x + c = 0)
  let P2 := ∀ (S : Set ℝ), (∅ ⊂ S)
  let P3 := ∀ (a : ℝ), 0 ≤ a^2
  let P4 := ∀ (a b : ℝ), 0 < a * b → (0 < a ∧ 0 < b)
  true :=
  (P1 ∧ P3) ∧ ¬P2 ∧ ¬P4 := by
  sorry

end number_of_true_propositions_is_2_l380_380825


namespace handshake_count_l380_380492

theorem handshake_count (n_total n_group1 n_group2 : ℕ) 
  (h_total : n_total = 40) (h_group1 : n_group1 = 25) (h_group2 : n_group2 = 15) 
  (h_sum : n_group1 + n_group2 = n_total) : 
  (15 * 39) / 2 = 292 := 
by sorry

end handshake_count_l380_380492


namespace relationship_among_abc_l380_380904

noncomputable def a : ℝ := Real.log (1 / 3)
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := (1 / 3) ^ 2

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l380_380904


namespace cricket_problem_solved_l380_380657

noncomputable def cricket_problem : Prop :=
  let run_rate_10 := 3.2
  let target := 252
  let required_rate := 5.5
  let overs_played := 10
  let total_overs := 50
  let runs_scored := run_rate_10 * overs_played
  let runs_remaining := target - runs_scored
  let overs_remaining := total_overs - overs_played
  (runs_remaining / overs_remaining = required_rate)

theorem cricket_problem_solved : cricket_problem :=
by
  sorry

end cricket_problem_solved_l380_380657


namespace minimal_questions_l380_380415

theorem minimal_questions (n : ℕ) (x : Fin n → ℤ) :
  ∃ S : ℤ, ∀ a : Fin n → ℝ, (S = ∑ j in Finset.range n, (a j) * (x j)) ∧ 
  (∀ j, a j = (100 : ℤ)^(j : ℕ)) → 
  ∃! x' : Fin n → ℤ, (∀ i, x i = x' i) := sorry

end minimal_questions_l380_380415


namespace green_chameleon_increase_l380_380326

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380326


namespace find_b_range_f2_l380_380922

-- Definitions of the function and derivative
def f (a b c x : ℝ) : ℝ := -x^3 + a * x^2 + b * x + c
def f' (a b x : ℝ) : ℝ := -3 * x^2 + 2 * a * x + b

-- Problem statement
theorem find_b (a c : ℝ) (H1 : ∀ x ∈ Icc (-∞:ℝ) 0, f a 0 c x ≤ f a 0 c 0)
               (H2 : ∀ x ∈ Ioc 0 1, f a 0 c x > f a 0 c 0) :
  b = 0 := sorry

theorem range_f2 (a : ℝ) (c : ℝ) (H3 : 1 ∈ set_of (λ x, f a 0 c x = 0))
                (H4 : a > 3 / 2) :
  ∃ r, f a 0 (1 - a) 2 = r ∧ r > -5 / 2 := sorry

end find_b_range_f2_l380_380922


namespace triangle_area_proof_l380_380632

noncomputable def triangle_area {A B C : ℝ} (angle_B : ℝ) (AC AB : ℝ) : ℝ :=
if angle_B = 30 * Real.pi / 180 ∧ AC = 1 ∧ AB = Real.sqrt 3 then
  let sin_30 := Real.sin (30 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  let area1 := 1/2 * AB * AC * sin_90
  let area2 := 1/2 * AB * AC * sin_30
  if area1 = Real.sqrt 3 / 2 then area1 else area2
else
  0

theorem triangle_area_proof : triangle_area 30 1 (Real.sqrt 3) = Real.sqrt 3 / 2 ∨ triangle_area 30 1 (Real.sqrt 3) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_proof_l380_380632


namespace gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l380_380905

theorem gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4 (k : Int) :
  Int.gcd ((360 * k)^2 + 6 * (360 * k) + 8) (360 * k + 4) = 4 := 
sorry

end gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l380_380905


namespace log_subtraction_example_l380_380707

noncomputable def log {a N : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) : ℝ := 
  if h : (∃ b : ℝ, N = a ^ b) then Classical.choose h else 0

theorem log_subtraction_example (h1 : 0 < (3 : ℝ)) (h2 : (3 : ℝ) ≠ 1)
  (h3 : 0 < (5 : ℝ)) (h4 : (5 : ℝ) ≠ 1) :
  (log h1 h2 9) - (log h3 h4 125) = -1 := by
  sorry

end log_subtraction_example_l380_380707


namespace Carla_retains_42_8_percent_l380_380131

-- Initial amount of marbles
variable (N : ℝ)

-- Conditions
def marbles_given_Lucia := 0.30 * N
def marbles_after_Lucia := N - marbles_given_Lucia
def marbles_given_Julius := 0.15 * marbles_after_Lucia
def marbles_after_Julius := marbles_after_Lucia - marbles_given_Julius
def marbles_given_Minh := 0.20 * marbles_after_Julius
def marbles_after_Minh := marbles_after_Julius - marbles_given_Minh
def marbles_given_Noah := 0.10 * marbles_after_Minh
def marbles_after_Noah := marbles_after_Minh - marbles_given_Noah

-- Final statement to prove
theorem Carla_retains_42_8_percent : marbles_after_Noah = 0.428 * N := 
sorry

end Carla_retains_42_8_percent_l380_380131


namespace hot_drinks_sales_l380_380811

theorem hot_drinks_sales (x: ℝ) (h: x = 4) : abs ((-2.35 * x + 155.47) - 146) < 1 :=
by sorry

end hot_drinks_sales_l380_380811


namespace increase_in_green_chameleons_is_11_l380_380319

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380319


namespace inequality_proof_l380_380188

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) +
    (b / Real.sqrt (b^2 + 8 * a * c)) +
    (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_proof_l380_380188


namespace value_of_m_l380_380960

theorem value_of_m (m : ℤ) (h : (-2)^(2*m) = 2^(24 - m)) : m = 8 :=
by
  sorry

end value_of_m_l380_380960


namespace digit_150th_in_decimal_of_fraction_l380_380044

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380044


namespace sequence_probability_correct_l380_380473

noncomputable def m : ℕ := 377
noncomputable def n : ℕ := 4096

theorem sequence_probability_correct :
  let m := 377
  let n := 4096
  (m.gcd n = 1) ∧ (m + n = 4473) := 
by
  -- Proof requires the given equivalent statement in Lean, so include here
  sorry

end sequence_probability_correct_l380_380473


namespace part_a_part_b_part_c_l380_380787

-- Part (a)
theorem part_a : ∀ (points : Finset ℝ × ℝ), 
  (∀ P in points, (dist P (0,0) ≤ 1)) ∧ (∀ P Q in points, P ≠ Q → dist P Q > 1) → points.card ≤ 5 :=
begin
  sorry,
end

-- Part (b)
theorem part_b : ∀ (points : Finset ℝ × ℝ), 
  (∀ P in points, (dist P (0,0) ≤ 10)) ∧ (∀ P Q in points, P ≠ Q → dist P Q > 1) → points.card ≤ 449 :=
begin
  sorry,
end

-- Part (c)
theorem part_c : ∃ (points : Finset ℝ × ℝ), 
  (∀ P in points, (dist P (0,0) ≤ 10)) ∧ (∀ P Q in points, P ≠ Q → dist P Q > 1) ∧ points.card = 400 :=
begin
  sorry,
end

end part_a_part_b_part_c_l380_380787


namespace fraction_identity_l380_380617

theorem fraction_identity (m n r t : ℚ) 
  (h₁ : m / n = 3 / 5) 
  (h₂ : r / t = 8 / 9) :
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := 
by
  sorry

end fraction_identity_l380_380617


namespace original_three_digit_number_a_original_three_digit_number_b_l380_380432

section ProblemA

variables {x y z : ℕ}

/-- In a three-digit number, the first digit on the left was erased. Then, the resulting
  two-digit number was multiplied by 7, and the original three-digit number was obtained. -/
theorem original_three_digit_number_a (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  N = 7 * (10 * y + z)) : ∃ (N : ℕ), N = 350 :=
sorry

end ProblemA

section ProblemB

variables {x y z : ℕ}

/-- In a three-digit number, the middle digit was erased, and the resulting number 
  is 6 times smaller than the original. --/
theorem original_three_digit_number_b (h : ∃ (N : ℕ), N = 100 * x + 10 * y + z ∧ 
  6 * (10 * x + z) = N) : ∃ (N : ℕ), N = 108 :=
sorry

end ProblemB

end original_three_digit_number_a_original_three_digit_number_b_l380_380432


namespace van_helsing_strategy_l380_380765

theorem van_helsing_strategy (V W : ℕ) (hW : W = 4 * V):
  (V / 2) * 5 + 80 = 105 → 
  let vampires_removed := V / 2 in
  let werewolves_removed := 8 in
  V / 2 * 2 + 8 = 7 ∧
  (8 / 40) * 100 = 20 :=
by
  intros
  sorry

end van_helsing_strategy_l380_380765


namespace trapezoid_angles_l380_380762

-- Definition of the problem statement in Lean 4
theorem trapezoid_angles (A B C D : ℝ) (h1 : A = 60) (h2 : B = 130)
  (h3 : A + D = 180) (h4 : B + C = 180) (h_sum : A + B + C + D = 360) :
  C = 50 ∧ D = 120 :=
by
  sorry

end trapezoid_angles_l380_380762


namespace origin_outside_circle_l380_380651

noncomputable def point : Type := ℝ × ℝ

def dist (p q : point) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def A : point := (-4, -3)
def O : point := (0, 0)
def r : ℝ := 4

theorem origin_outside_circle : dist O A > r := 
by
  calc
    dist O A = Real.sqrt ((-4 - 0)^2 + (-3 - 0)^2) : by rfl
    ... = Real.sqrt (16 + 9) : by rfl
    ... = 5 : by simp [Real.sqrt]
    ... > 4 : by linarith

end origin_outside_circle_l380_380651


namespace larger_of_two_solutions_l380_380540

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l380_380540


namespace stratified_sampling_third_year_students_l380_380270

theorem stratified_sampling_third_year_students (total_students : ℕ) (sample_size : ℕ) 
  (first_year_students : ℕ) (second_year_students : ℕ) :
  total_students = 2400 →
  sample_size = 120 →
  first_year_students = 760 →
  second_year_students = 840 →
  let third_year_students := total_students - first_year_students - second_year_students in
  let probability := (sample_size : ℚ) / total_students in
  let third_year_samples := (third_year_students : ℚ) * probability in
  third_year_samples = 40 :=
by
  intros ht hs hf hs2
  let third_year_students := total_students - first_year_students - second_year_students
  let probability := (sample_size : ℚ) / total_students
  let third_year_samples := (third_year_students : ℚ) * probability
  rw [ht, hs, hf, hs2] at *
  have ht3 : third_year_students = 2400 - 760 - 840 := rfl
  rw ht3
  have third_year_students_value : third_year_students = 800 := by decide
  rw third_year_students_value
  unfold probability
  have probability_value : probability = (120 : ℚ) / 2400 := by decide
  rw probability_value
  have probability_value_simplified : probability = 1 / 20 := by decide
  rw probability_value_simplified
  simp only [mul_one, div_eq_mul_one_div]
  have result : (800 : ℚ) * (1 / 20) = 40 := by decide
  exact result

end stratified_sampling_third_year_students_l380_380270


namespace dot_product_l380_380610

def vec (α : Type*) := α × α

noncomputable def a : vec ℝ := (1/2, real.sqrt 3 / 2)
noncomputable def b : vec ℝ := (-real.sqrt 3 / 2, 1/2)

def dot (u v : vec ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product (a b : vec ℝ) : 
  dot (a.1 + b.1, a.2 + b.2) a = 1 := by
  sorry

end dot_product_l380_380610


namespace paper_fold_ratio_l380_380475

theorem paper_fold_ratio (paper_side : ℕ) (fold_fraction : ℚ) (cut_fraction : ℚ)
  (thin_section_width thick_section_width : ℕ) (small_width large_width : ℚ)
  (P_small P_large : ℚ) (ratio : ℚ) :
  paper_side = 6 →
  fold_fraction = 1 / 3 →
  cut_fraction = 2 / 3 →
  thin_section_width = 2 →
  thick_section_width = 4 →
  small_width = 2 →
  large_width = 16 / 3 →
  P_small = 2 * (6 + small_width) →
  P_large = 2 * (6 + large_width) →
  ratio = P_small / P_large →
  ratio = 12 / 17 :=
by
  sorry

end paper_fold_ratio_l380_380475


namespace midpoint_y_coordinate_of_MN_l380_380236

theorem midpoint_y_coordinate_of_MN (a : ℝ) (ha : 0 < a ∧ a < π / 2) (hMN : abs (sin a - cos a) = sqrt 2) :
  (sin a + cos a) / 2 = sqrt 2 / 2 := by
  sorry

end midpoint_y_coordinate_of_MN_l380_380236


namespace number_of_dragon_numbers_l380_380626

def digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

def isDragonNumber (a b c : digit) : Prop :=
  (a.val * 10 + b.val > b.val * 10 + c.val) ∧ 
  (b.val * 10 + c.val > c.val * 10 + a.val)

def countDragonNumbers : ℕ := by
  let possible_digits : list digit := [1,2,3,4,5,6,7,8,9].map (λ n, ⟨n, ⟨by decide, by decide⟩⟩)
  let triples := possible_digits.product (possible_digits.product possible_digits)
  let is_dragon := triples.filter (λ ⟨a, ⟨b, c⟩⟩, isDragonNumber a b c)
  exact is_dragon.length
  sorry

theorem number_of_dragon_numbers : countDragonNumbers = 120 := by
  sorry

end number_of_dragon_numbers_l380_380626


namespace F_even_or_odd_l380_380552

-- Define F(n) for number of ways n can be expressed as sum of three distinct positive integers
def F (n : ℕ) : ℕ := 
  {count // ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = n}.val

-- Define main theorem statement
theorem F_even_or_odd (n : ℕ) : 
  (n % 6 = 2 ∨ n % 6 = 4 → (F n) % 2 = 0) ∧ 
  (n % 6 = 0 → (F n) % 2 = 1) :=
sorry

end F_even_or_odd_l380_380552


namespace radius_of_two_equal_circles_eq_16_l380_380756

noncomputable def radius_of_congruent_circles : ℝ := 16

theorem radius_of_two_equal_circles_eq_16 :
  ∃ x : ℝ, 
    (∀ r1 r2 r3 : ℝ, r1 = 4 ∧ r2 = r3 ∧ r2 = x ∧ 
    ∃ line : ℝ → ℝ → Prop, 
    (line 0 r1) ∧ (line 0 r2)  ∧ 
    (line 0 r3) ∧ 
    (line r2 r3) ∧
    (line r1 r2)  ∧ (line r1 r3) ∧ (line (r1 + r2) r2) ) 
    → x = 16 := sorry

end radius_of_two_equal_circles_eq_16_l380_380756


namespace complex_inequality_l380_380563

theorem complex_inequality (z1 z2 z3 z4 : ℂ) :
  abs(z1 - z3)^2 + abs(z2 - z4)^2 ≤ abs(z1 - z2)^2 + abs(z2 - z3)^2 + abs(z3 - z4)^2 + abs(z4 - z1)^2 ∧
  (abs(z1 - z3)^2 + abs(z2 - z4)^2 = abs(z1 - z2)^2 + abs(z2 - z3)^2 + abs(z3 - z4)^2 + abs(z4 - z1)^2 ↔ z1 + z3 = z2 + z4) :=
  by
  sorry

end complex_inequality_l380_380563


namespace smallest_distance_l380_380300

-- Definition of a point on the circle
def is_on_circle (A : ℝ × ℝ) : Prop :=
  let (x, y) := A in
  x^2 + y^2 - 16 * x - 6 * y + 89 = 0

-- Definition of a point on the parabola
def is_on_parabola (B : ℝ × ℝ) : Prop :=
  let (x, y) := B in
  y^2 = 8 * x

-- Hypothesize points A and B
variables (A B : ℝ × ℝ)

-- Lean statement for the problem
theorem smallest_distance (hA : is_on_circle A) (hB : is_on_parabola B) : ℝ :=
  have center := (8, 3)
  have AC := 2
  let BC := ((fst B - 8)^2 + (snd B - 3)^2)^(1 / 2)
  6 * real.sqrt 2 - 2

-- Here, we do not perform the actual proof steps, so we omit the proof body with sorry.

end smallest_distance_l380_380300


namespace g_of_sum_eq_zero_l380_380730

theorem g_of_sum_eq_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 :=
by
  let g := λ x : ℝ, x^2 - 2013 * x
  sorry

end g_of_sum_eq_zero_l380_380730


namespace induction_step_addition_l380_380703

theorem induction_step_addition (k : ℕ) :
  let lhs_k := ∑ i in finset.range (k+1), i^2 + ∑ i in finset.range k, i^2
  let lhs_k1 := ∑ i in finset.range (k+2), i^2 + ∑ i in finset.range (k+1), i^2
  (lhs_k1 = lhs_k + (k+1)^2 + k^2) :=
by
  let lhs_k := ∑ i in finset.range (k+1), i^2 + ∑ i in finset.range k, i^2
  let lhs_k1 := ∑ i in finset.range (k+2), i^2 + ∑ i in finset.range (k+1), i^2
  sorry

end induction_step_addition_l380_380703


namespace cylindrical_to_rectangular_conversion_l380_380149

theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (h1 : r = 10) 
  (h2 : θ = Real.pi / 3) 
  (h3 : z = -2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5, 5 * Real.sqrt 3, -2) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l380_380149


namespace measles_cases_in_1990_l380_380965

noncomputable def measles_cases_1970 := 480000
noncomputable def measles_cases_2000 := 600
noncomputable def years_between := 2000 - 1970
noncomputable def total_decrease := measles_cases_1970 - measles_cases_2000
noncomputable def decrease_per_year := total_decrease / years_between
noncomputable def years_from_1970_to_1990 := 1990 - 1970
noncomputable def decrease_to_1990 := years_from_1970_to_1990 * decrease_per_year
noncomputable def measles_cases_1990 := measles_cases_1970 - decrease_to_1990

theorem measles_cases_in_1990 : measles_cases_1990 = 160400 := by
  sorry

end measles_cases_in_1990_l380_380965


namespace three_digit_integers_with_two_identical_digits_less_than_700_l380_380943

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def less_than_700 (n : ℕ) : Prop :=
  n < 700

def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.nodup = false

theorem three_digit_integers_with_two_identical_digits_less_than_700 : 
  ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_three_digit n ∧ less_than_700 n ∧ has_at_least_two_identical_digits n) ∧
  s.card = 156 := by
  sorry

end three_digit_integers_with_two_identical_digits_less_than_700_l380_380943


namespace find_f_log_value_l380_380910

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then 2^x + 1 else sorry

theorem find_f_log_value (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_spec : ∀ x, 0 < x → x < 1 → f x = 2^x + 1) :
  f (Real.logb (1/2) (1/15)) = -31/15 :=
sorry

end find_f_log_value_l380_380910


namespace females_over_30_prefer_webstream_l380_380477

-- Define the total number of survey participants
def total_participants : ℕ := 420

-- Define the number of participants who prefer WebStream
def prefer_webstream : ℕ := 200

-- Define the number of participants who do not prefer WebStream
def not_prefer_webstream : ℕ := 220

-- Define the number of males who prefer WebStream
def males_prefer : ℕ := 80

-- Define the number of females under 30 who do not prefer WebStream
def females_under_30_not_prefer : ℕ := 90

-- Define the number of females over 30 who do not prefer WebStream
def females_over_30_not_prefer : ℕ := 70

-- Define the total number of females under 30 who do not prefer WebStream
def females_not_prefer : ℕ := females_under_30_not_prefer + females_over_30_not_prefer

-- Define the total number of participants who do not prefer WebStream
def total_not_prefer : ℕ := 220

-- Define the number of males who do not prefer WebStream
def males_not_prefer : ℕ := total_not_prefer - females_not_prefer

-- Define the number of females who prefer WebStream
def females_prefer : ℕ := prefer_webstream - males_prefer

-- Define the total number of females under 30 who prefer WebStream
def females_under_30_prefer : ℕ := total_participants - prefer_webstream - females_under_30_not_prefer

-- Define the remaining females over 30 who prefer WebStream
def females_over_30_prefer : ℕ := females_prefer - females_under_30_prefer

-- The Lean statement to prove
theorem females_over_30_prefer_webstream : females_over_30_prefer = 110 := by
  sorry

end females_over_30_prefer_webstream_l380_380477


namespace hyperbola_asymptote_equations_l380_380599

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : Prop :=
  ∀ E : ℝ × ℝ, (∃ x y, E = (x, y) ∧ (y^2 / a^2 - x^2 / b^2 = 1)) →
  (forall x y, let D := (x, y) in 
    let F1 := (0, b / 2) in 
    D = (b * sqrt (4 * (b / 2)^2 - b^2) / (4 * (b / 2)), - b^2 / (4 * (b / 2))) ∧
    ( E = (b * sqrt (4 * (b / 2)^2 - b^2) / (2 * (b / 2)), (2 * (b / 2)^2 - b^2) / (2 * (b / 2))) ) →
    ( (y / a = 1 / 2) → (y = ±(1 / 2) * x) )) sorry

theorem hyperbola_asymptote_equations (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  hyperbola_asymptotes a b h_a_pos h_b_pos :=
  sorry

end hyperbola_asymptote_equations_l380_380599


namespace speed_of_first_part_l380_380457

variable (v : ℝ) -- Speed of the first part of the trip
variable (d_total d_part d_remaining : ℝ) -- Total distance, first part distance, remaining part distance
variable (s_remaining s_avg : ℝ) -- Remaining part speed, average speed
variable (time_total : ℝ) -- Total time of the trip

-- Define constants based on the problem's conditions
def d_total := 70
def d_part := 35
def d_remaining := 35
def s_remaining := 24
def s_avg := 32

-- Define the time taken for each part and the total time
def time_part1 := d_part / v
def time_part2 := d_remaining / s_remaining
def time_total := time_part1 + time_part2

-- Define the average speed equation as given in the problem
def avg_speed_eq := s_avg = d_total / time_total

theorem speed_of_first_part :
  avg_speed_eq → v = 48 :=
by sorry

end speed_of_first_part_l380_380457


namespace least_product_of_distinct_primes_greater_than_50_l380_380176

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  p ≠ q ∧ is_prime p ∧ is_prime q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 : 
  ∃ p q, distinct_primes_greater_than_50 p q ∧ p * q = 3127 := 
sorry

end least_product_of_distinct_primes_greater_than_50_l380_380176


namespace repeating_decimal_to_fraction_l380_380168

theorem repeating_decimal_to_fraction : (2.353535... : Rational) = 233/99 :=
by
  sorry

end repeating_decimal_to_fraction_l380_380168


namespace sum_of_remainders_l380_380423

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 11) : (n % 4) + (n % 5) = 4 :=
by
  sorry

end sum_of_remainders_l380_380423


namespace chameleon_problem_l380_380363

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380363


namespace number_of_quadruples_l380_380674

open Nat

theorem number_of_quadruples :
  let quadruples := { (a, b, c, d) ∈ Finset.product (Finset.product (Finset.range 4) (Finset.range 4)) (Finset.product (Finset.range 4) (Finset.range 4)) | 
                       (2 * a * d - b * c) % 2 = 1 }
  quadruples.card = 64 :=
by
  let quadruples := { (a, b, c, d) ∈ Finset.product (Finset.product (Finset.range 4) (Finset.range 4)) (Finset.product (Finset.range 4) (Finset.range 4)) | 
                       (2 * a * d - b * c) % 2 = 1 }
  show quadruples.card = 64
  sorry

end number_of_quadruples_l380_380674


namespace all_points_D_on_single_circle_l380_380608

noncomputable theory

open_locale classical

variables {K : Type*} [field K]

def intersecting_circles (P Q : K) (circle1 circle2 : set K) : Prop :=
  P ∈ circle1 ∧ P ∈ circle2 ∧ Q ∈ circle1 ∧ Q ∈ circle2

def line_through_Q (Q : K) (l : set K) : Prop :=
  Q ∈ l

def line_intersects_circles_at (l : set K) (A B : K) (circle1 circle2 : set K) : Prop :=
  (A ∈ circle1 ∧ A ∈ l) ∧ (B ∈ circle2 ∧ B ∈ l)

def tangents_intersect_at (A B C : K) (circle1 circle2 : set K) : Prop :=
  ∃ (tangentA tangentB : set K), 
    (A ∈ tangentA ∧ ∀ P ∈ circle1, P ≠ A → ¬P ∈ tangentA) ∧
    (B ∈ tangentB ∧ ∀ P ∈ circle2, P ≠ B → ¬P ∈ tangentB) ∧
    ∃ I ∈ tangentA, I ∈ tangentB ∧ I = C

def bisector_intersects (C P Q D : K) (l AB : set K) : Prop :=
  ∃ (bisector : set K), 
    C ∈ bisector ∧ P ∈ bisector ∧ Q ∈ bisector ∧ D ∈ AB ∧
    ∀ θ₁ θ₂, (θ₁, θ₂) ∈ line_intersects_circles_at l A B → 
      (∠ (C, P, Q) = ∠ (θ₁, θ₂))

theorem all_points_D_on_single_circle 
  (P Q : K) (circle1 circle2 : set K)
  (h1 : intersecting_circles P Q circle1 circle2)
  (h2 : ∀ (l : set K) (hl : line_through_Q Q l), 
        ∃ A B, line_intersects_circles_at l A B circle1 circle2)
  (h3 : ∀ (A B l : K) (hl : line_through_Q Q l), ∃ C, tangents_intersect_at A B C circle1 circle2)
  (h4 : ∀ (l : set K) (A B C : K), line_through_Q Q l → 
        tangents_intersect_at A B C circle1 circle2 →
        ∃ D, bisector_intersects C P Q D l l):
  ∃ Ω : set K, ∀ (D : K) (l : set K)
    (hl₁: line_through_Q Q l)
    (hl₂ : ∃ A B, line_intersects_circles_at l A B circle1 circle2)
    (hl₃ : ∃ C, tangents_intersect_at A B C circle1 circle2)
    (hl₄ : ∃ (D : K), bisector_intersects C P Q D l l),
    D ∈ Ω := sorry

end all_points_D_on_single_circle_l380_380608


namespace centroid_APQ_on_BD_l380_380299

-- Define rhombus and related notions
variables (A B C D P Q: Type) [EuclideanGeometry]

-- Conditions for the problem
axiom rhombus_ABCD : is_rhombus A B C D
axiom P_on_BC : on_segment P B C
axiom Q_on_CD : on_segment Q C D
axiom BP_eq_CQ : dist B P = dist C Q

-- Goal: prove that the centroid of triangle APQ is on segment BD
theorem centroid_APQ_on_BD : centroid (triangle A P Q) ∈ segment B D :=
  sorry

end centroid_APQ_on_BD_l380_380299


namespace total_prime_factors_sum_l380_380868

theorem total_prime_factors_sum :
  let exponents := [17, 13, 11, 9, 7, 5, 3, 2] in
  List.sum exponents = 67 :=
sorry

end total_prime_factors_sum_l380_380868


namespace discount_percentage_theorem_l380_380810

variable (cost_price : ℝ) 
variable (marked_price_percentage : ℝ) 
variable (profit_percentage : ℝ)
variable (marked_price : ℝ)
variable (selling_price : ℝ)
variable (discount_amount : ℝ)
variable (discount_percentage : ℝ)

def cost_price := 50
def marked_price_percentage := 0.30
def profit_percentage := 0.17

-- Calculate marked price
def marked_price := cost_price + (marked_price_percentage * cost_price)

-- Calculate selling price for 17% profit
def profit_amount := profit_percentage * cost_price
def selling_price := cost_price + profit_amount

-- Calculate discount amount
def discount_amount := marked_price - selling_price

-- Calculate discount percentage
def discount_percentage := (discount_amount / marked_price) * 100

theorem discount_percentage_theorem : discount_percentage = 10 := by
  sorry

end discount_percentage_theorem_l380_380810


namespace exists_non_cubic_integer_with_cubic_term_at_most_one_cubic_in_sequence_l380_380684

def sequence (a : ℕ) : ℕ → ℕ
| 0          := a
| (n + 1)    := (sequence n)^2 + 20

theorem exists_non_cubic_integer_with_cubic_term :
  ∃ (a : ℕ), a ≠ (b : ℕ) ^ 3 ∧ ∃ (n : ℕ), ∃ m, sequence a n = m ^ 3 :=
sorry

theorem at_most_one_cubic_in_sequence (a : ℕ) :
  ∀ m1 m2, m1 ≠ 0 → m2 ≠ 0 → sequence a m1 = m2 ^ 3 → (∀ n, n ≠ m1 → ¬ (∃ m, sequence a n = m ^ 3)) :=
sorry

end exists_non_cubic_integer_with_cubic_term_at_most_one_cubic_in_sequence_l380_380684


namespace no_point_with_given_projection_property_l380_380291

variable {α : Type*} [linear_ordered_field α]

structure point (α : Type*) :=
(x y : α)

structure convex_polygon (α : Type*) :=
(vertices : list (point α))
(is_convex : ∀ (A B : point α), A ≠ B → ∃ (C : point α), C ∈ vertices → dist C A < dist C B)

noncomputable def point_projections_outside_convex (P : convex_polygon α) (X : point α) : Prop :=
∀ (line : set (point α)), line ∈ (λ (vertices : list (point α)), list.zip vertices (vertices.tail ++ [vertices.head])).to_finset → 
  let proj := projection X line in ¬(proj ∈ (line.segment_of (list.to_finset P.vertices)))

theorem no_point_with_given_projection_property (P : convex_polygon α) (X : point α) :
  ¬ point_projections_outside_convex P X :=
sorry

end no_point_with_given_projection_property_l380_380291


namespace equation_solution_count_l380_380955

def is_valid_solution (x : ℕ) : Prop :=
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 50 ∧ 
    ¬(∃ m : ℕ, x = m^2 ∧ odd m) ∧ 
    ¬(∃ k : ℕ, x = k^3 ∧ even k)

theorem equation_solution_count :
  {x : ℕ | is_valid_solution x}.to_finset.card = 2 :=
by
  sorry

end equation_solution_count_l380_380955


namespace total_people_waiting_at_SFL_l380_380777

open Nat

theorem total_people_waiting_at_SFL 
  (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ) 
  (left_A : ℕ) (joined_C : ℕ) (left_E : ℕ)
  (hA : initial_A = 283) (hB : initial_B = 356) (hC : initial_C = 412) (hD : initial_D = 179) (hE : initial_E = 389)
  (hleft_A : left_A = 15) (hjoined_C : joined_C = 10) (hleft_E : left_E = 20) :
  initial_A - left_A + initial_B + initial_C + joined_C + initial_D + (initial_E - left_E) = 1594 :=
by
  rw [hA, hB, hC, hD, hE, hleft_A, hjoined_C, hleft_E]
  norm_num
  exact rfl

end total_people_waiting_at_SFL_l380_380777


namespace increase_in_green_chameleons_is_11_l380_380321

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380321


namespace monotonicity_F_range_c_l380_380222

section

variable (a c : ℝ)

-- Function f(x)
def f (x : ℝ) : ℝ := 1 / (exp x) - 1

-- Function F(x) for monotonicity analysis
def F (x : ℝ) : ℝ := f x + a * x

-- Ensuring there are no issues with non-computable terms
noncomputable def F' (x : ℝ) : ℝ := -1 / (exp x) + a

-- Condition for proving (I)
theorem monotonicity_F : 
  (∀ x : ℝ, a ≤ 0 → F' a ≤ 0) ∧
  (∀ x : ℝ, a > 0 → (F' x = 0 → x = -ln a) ∧ 
               (∀ x : ℝ, x < -ln a → F' x < 0) ∧ 
               (∀ x : ℝ, x > -ln a → F' x > 0)) :=
sorry

-- Inequality for proving (II)
def inequality (x : ℝ) : Prop := exp x * f x ≤ c * (x - 1) + 1

-- Function g(x) in part (II)
noncomputable def g (x : ℝ) : ℝ := exp x + c * x - c
noncomputable def g' (x : ℝ) : ℝ := exp x + c

-- Condition for the range of c in (II)
theorem range_c : 
  (forall x : ℝ, inequality x) ↔ -exp 2 ≤ c ∧ c < 0 :=
sorry

end

end monotonicity_F_range_c_l380_380222


namespace tangent_locus_l380_380689

theorem tangent_locus (l₁ l₂ : Line) (s : Sphere) (M : Point) (N : Point) (MN : Line) :
  is_tangent l₁ s ∧ is_tangent l₂ s ∧ on_line M l₁ ∧ on_line N l₂ ∧ is_tangent MN s → 
  ∃ c₁ c₂ : Circle, locus_of_tangency MN = { c₁, c₂ } :=
sorry

end tangent_locus_l380_380689


namespace investment_C_l380_380111

def investment_A : ℝ := 8000
def investment_B : ℝ := 10000
def profit_B : ℝ := 1700
def profit_difference_AC : ℝ := 680

theorem investment_C : ∃ (investment_C : ℝ), 
  let profit_A := (investment_A / investment_B) * profit_B in
  let profit_C := profit_A + profit_difference_AC in
  profit_C / investment_C = profit_B / investment_B :=
by
  let profit_A := (investment_A / investment_B) * profit_B
  let profit_C := profit_A + profit_difference_AC
  let investment_C := (profit_C * investment_B) / profit_B
  use investment_C
  field_simp [investment_C, profit_C, profit_B, investment_B, investment_A, profit_A]
  sorry

end investment_C_l380_380111


namespace sum_of_two_integers_l380_380744

open Nat

def relatively_prime (a b : ℕ) : Prop :=
  gcd a b = 1

theorem sum_of_two_integers :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a < 25 ∧ b < 25 ∧ relatively_prime a b ∧ (a + 1) * (b + 1) = 132 ∧ a + b = 21 := 
by
  sorry

end sum_of_two_integers_l380_380744


namespace tangency_points_l380_380547

variable (A B C D : Type*)
variable [MetricSpace D]

-- Given an angle with apex C and sides tangent to points A and B
axiom given_angle (α γ : Line D) (C A B : D) (h1 : is_tangent_to α A C) (h2 : is_tangent_to γ B C)

-- Two circles that are tangent to the sides of this angle at points A and B
axiom tangent_circles (S1 S2 : Circle D) (h3 : is_tangent_to S1 α A) (h4 : is_tangent_to S2 γ B)

-- The set of tangency points is the arc AB of the circle passing through A, B, and D*
theorem tangency_points :
  Σ (D_star : D), is_arc (Circle.mk A B D_star) A B :=
sorry

end tangency_points_l380_380547


namespace sum_of_first_n_terms_l380_380571

def sequence (a : ℕ → ℕ) :=
  a 1 = 2 ∧ a 1 + a 2 = 7 ∧ a 1 + a 2 + a 3 = 16

def sum_of_sequence (S n : ℕ) :=
  S = n * (n + 1)

theorem sum_of_first_n_terms {a : ℕ → ℕ} {S : ℕ} (h : sequence a) : 
  sum_of_sequence S := sorry

end sum_of_first_n_terms_l380_380571


namespace chameleon_problem_l380_380340

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380340


namespace terms_are_not_like_l380_380071

-- Define the terms
def term1 := 2 * a^2 * b
def term2 := -b^2 * a

-- Define like terms condition: same variables with same exponents
def like_terms (t1 t2 : ℕ → ℕ → ℕ → ℕ) := ∀ (a b : ℕ), 
    (∀ x y z, t1 x y z = a * x^2 * y ↔ t2 x y x = b * y^2 * x)

-- The theorem stating they are not like terms
theorem terms_are_not_like : ¬ like_terms term1 term2 := by
    -- Proof goes here
    sorry

end terms_are_not_like_l380_380071


namespace sum_m_n_eq_345_l380_380680

theorem sum_m_n_eq_345 
  (m : ℕ) 
  (h1 : (∀ k : ℕ, k < m → ¬ (k > 0 ∧ (∀ d, d ∣ k → d = 1 ∨ d = k))))
  (n : ℕ) 
  (h2 : n < 500) 
  (h3 : (∀ k : ℕ, k > n → (k < 500 ∧ (∃ p : ℕ, p.prime ∧ (k = p * p * p))) → false))
  (h4 : ∃ p : ℕ, p.prime ∧ (n = p * p * p)) 
  : m + n = 345 :=
sorry

end sum_m_n_eq_345_l380_380680


namespace negation_of_existence_implies_universal_l380_380741

theorem negation_of_existence_implies_universal (x : ℝ) :
  (∀ x : ℝ, ¬(x^2 ≤ |x|)) ↔ (∀ x : ℝ, x^2 > |x|) :=
by 
  sorry

end negation_of_existence_implies_universal_l380_380741


namespace digit_150th_l380_380054

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380054


namespace integral_calculation_l380_380500

noncomputable def definite_integral_value : ℝ :=
  ∫ x in 0..π, 16 * (sin (x / 2))^2 * (cos (x / 2))^6

theorem integral_calculation :
  definite_integral_value = (5 * π) / 8 :=
by
  sorry

end integral_calculation_l380_380500


namespace champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l380_380969

-- Define the structure and relationship between teams in the tournament
structure Tournament (Team : Type) :=
  (competes : Team → Team → Prop) -- teams play against each other
  (no_ties : ∀ A B : Team, (competes A B ∧ ¬competes B A) ∨ (competes B A ∧ ¬competes A B)) -- no ties
  (superior : Team → Team → Prop) -- superiority relationship
  (superior_def : ∀ A B : Team, superior A B ↔ (competes A B ∧ ¬competes B A) ∨ (∃ C : Team, superior A C ∧ superior C B))

-- The main theorem based on the given questions
theorem champion_team_exists {Team : Type} (tournament : Tournament Team) :
  ∃ champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B) :=
  sorry

theorem unique_champion_wins_all {Team : Type} (tournament : Tournament Team)
  (h : ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B)) :
  ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B ∧ tournament.competes champion B ∧ ¬tournament.competes B champion) :=
  sorry

theorem not_exactly_two_champions {Team : Type} (tournament : Tournament Team) :
  ¬∃ A B : Team, A ≠ B ∧ (∀ C : Team, C ≠ A → tournament.superior A C) ∧ (∀ C : Team, C ≠ B → tournament.superior B C) :=
  sorry

end champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l380_380969


namespace sum_a_n_eq_1992_l380_380551

def a_n (n : ℕ) : ℕ :=
  if n % 105 = 0 then 15
  else if n % 42 = 0 then 18
  else if n % 210 = 0 then 21
  else 0

theorem sum_a_n_eq_1992 : (∑ n in Finset.range 3000, a_n (n + 1)) = 1992 :=
  sorry

end sum_a_n_eq_1992_l380_380551


namespace incorrect_statements_l380_380264

def f (α : ℝ) (x : ℝ) := x^α

theorem incorrect_statements (α : ℝ) :
  (¬∀ (α : ℝ), f α (-1) = 1) ∧
  (f α 0 = 1 ∧ (f α (-1) = (-1)) → ∀ x, f α (-x) = - f α x) ∧
  (f α 0 = 1 ∧ (f α 1 = 1) → ∀ x, f α (-x) = f α x) ∧
  (¬∃ (α > 0), f α (√3) < f α (√2)) :=
by
  split
  { intro h,
    sorry },
  split
  { split
    { sorry },
    intro h,
    sorry },
  split
  { intro h,
    sorry },
  { intro h,
    sorry }

end incorrect_statements_l380_380264


namespace probability_heads_after_biased_tails_l380_380985

theorem probability_heads_after_biased_tails (p_tails : ℚ) (h_prob_tails : p_tails = 3/4) :
  let p_heads := 1 - p_tails in
  p_heads = 1/4 :=
by
  sorry

end probability_heads_after_biased_tails_l380_380985


namespace six_points_on_common_sphere_l380_380873

theorem six_points_on_common_sphere
    (s1 s2 s3 s4 : Sphere)
    (p12 : Point) (p13 : Point) (p14 : Point)
    (p23 : Point) (p24 : Point) (p34 : Point)
    (h1 : s1.touches s2 at p12)
    (h2 : s1.touches s3 at p13)
    (h3 : s1.touches s4 at p14)
    (h4 : s2.touches s3 at p23)
    (h5 : s2.touches s4 at p24)
    (h6 : s3.touches s4 at p34) :
    ∃ (sphere : Sphere), 
        lies_on_sphere p12 sphere ∧ 
        lies_on_sphere p13 sphere ∧ 
        lies_on_sphere p14 sphere ∧ 
        lies_on_sphere p23 sphere ∧ 
        lies_on_sphere p24 sphere ∧ 
        lies_on_sphere p34 sphere := 
by
  sorry

end six_points_on_common_sphere_l380_380873


namespace sum_reciprocal_sequence_l380_380892

theorem sum_reciprocal_sequence (a : ℕ → ℝ) (n : ℕ) (h0 : a 0 = 3)
  (hn : ∀ k, (3 - a (k + 1)) * (6 + a k) = 18) :
  ∑ i in Finset.range (n + 1), 1 / (a i) = (1 / 3) * (2^(n + 2) - n - 3) :=
by
  sorry

end sum_reciprocal_sequence_l380_380892


namespace length_of_segment_in_trapezoid_l380_380981

theorem length_of_segment_in_trapezoid
  (A B C D : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_base1 : dist B C = 3)
  (h_base2 : dist A D = 9)
  (h_angle_BAD : angle A B D = 30) 
  (h_angle_ADC : angle A D C = 60):
  ∃ (DK_length : ℝ), DK_length = sqrt 39 :=
  sorry

end length_of_segment_in_trapezoid_l380_380981


namespace part1_when_m_eq_2_part2_singleton_B_inter_Z_part3_A_inter_B_inter_Z_eq_n_l380_380207

open Set Real

def A : Set ℝ := { x | x^2 - x - 2 ≥ 0 }
def B (m : ℝ) : Set ℝ := { x | (1 - m^2) * x^2 + 2 * m * x - 1 < 0 }

-- 1. When m = 2, find set ∁ᵣA and set B
def complement_R_A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B_m2 : Set ℝ := { x | -1 < x ∧ x < (1 / 3) }

theorem part1_when_m_eq_2 :
  (complement_R A = complement_R_A) ∧ (B 2 = B_m2) :=
sorry

-- 2. If set B ∩ ℤ is a singleton set, find the set of possible values for real number m
def singleton_B_nZ (B : Set ℝ) : ∀{a : ℤ}, B ∩ (Set.Icc (a : ℝ) (a : ℝ)) = {↑a}

theorem part2_singleton_B_inter_Z : 
  {m : ℝ | ∃ (a : ℤ), B m ∩ (Set.Icc (a : ℝ) (a : ℝ)) = {↑a}} = {0} :=
sorry

-- 3. If the number of elements in set (A ∩ B) ∩ ℤ is n (n ∈ ℕ*), find the set of possible values for real number m
theorem part3_A_inter_B_inter_Z_eq_n (n : ℕ) (hn : 0 < n) : 
  {m : ℝ | ((A ∩ B m) ∩ (Set.Icc (0 : ℤ) (0 : ℤ))) = {(0 : ℤ)} ∧ n = 1} ∨
  ({m : ℝ | ((A ∩ B m) ∩ (Set.Icc (1 : ℤ) (1 : ℤ))) = {(0 : ℤ), (1 : ℤ)} ∧ n = 2}) = 
  if n = 1 then {0} else if n = 2 then {m : ℝ | -1/2 ≤ m ∧ m < 0} else ∅ :=
sorry

end part1_when_m_eq_2_part2_singleton_B_inter_Z_part3_A_inter_B_inter_Z_eq_n_l380_380207


namespace average_payment_debt_l380_380089

theorem average_payment_debt :
  let total_payments := 65
  let first_20_payment := 410
  let increment := 65
  let remaining_payment := first_20_payment + increment
  let first_20_total := 20 * first_20_payment
  let remaining_total := 45 * remaining_payment
  let total_paid := first_20_total + remaining_total
  let average_payment := total_paid / total_payments
  average_payment = 455 := by sorry

end average_payment_debt_l380_380089


namespace percentage_reduced_price_l380_380311

-- Definitions of prices
def reduced_price : ℝ := 6
def original_price : ℝ := 24

-- The theorem statement
theorem percentage_reduced_price :
  (reduced_price / original_price) * 100 = 25 := 
by
sory

end percentage_reduced_price_l380_380311


namespace number_of_3_digit_numbers_divisible_by_5_l380_380247

theorem number_of_3_digit_numbers_divisible_by_5 (digits : Finset ℕ) (h_digits : digits = {2, 3, 5, 6, 7, 9}) : 
  ∃ n : ℕ, n = 20 ∧ ∀ a b c, a ≠ b → b ≠ c → a ≠ c →
  a ∈ digits → b ∈ digits → (c ∈ digits ∧ c = 5) →
  (a * 100 + b * 10 + c) % 5 = 0 → (a * 100 + b * 10 + c) < 1000 :=
by
  use 20
  split
  . refl
  . sorry

end number_of_3_digit_numbers_divisible_by_5_l380_380247


namespace chameleon_problem_l380_380360

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380360


namespace combined_stock_cost_l380_380766

def stock_A_face_value := 100
def stock_A_discount := 2
def stock_A_brokerage := 1 / 5

def stock_B_face_value := 150
def stock_B_premium := 1.5
def stock_B_brokerage := 1 / 6

def stock_C_face_value := 200
def stock_C_discount := 3
def stock_C_brokerage := 0.5

theorem combined_stock_cost :
  let stock_A_cost := stock_A_face_value - (stock_A_discount / 100 * stock_A_face_value)
  let stock_A_brokerage_cost := stock_A_brokerage / 100 * stock_A_cost
  let total_stock_A_cost := stock_A_cost + stock_A_brokerage_cost
  
  let stock_B_cost := stock_B_face_value + (stock_B_premium / 100 * stock_B_face_value)
  let stock_B_brokerage_cost := stock_B_brokerage / 100 * stock_B_cost
  let total_stock_B_cost := stock_B_cost + stock_B_brokerage_cost
  
  let stock_C_cost := stock_C_face_value - (stock_C_discount / 100 * stock_C_face_value)
  let stock_C_brokerage_cost := stock_C_brokerage / 100 * stock_C_cost
  let total_stock_C_cost := stock_C_cost + stock_C_brokerage_cost
  
  total_stock_A_cost + total_stock_B_cost + total_stock_C_cost = 445.66975 :=
by 
  compute; sorry

end combined_stock_cost_l380_380766


namespace nolan_saves_l380_380312

theorem nolan_saves (monthly_savings : ℕ) (months : ℕ) (h1 : monthly_savings = 3 * 1000) (h2 : months = 12) :
  monthly_savings * months = 36 * 1000 :=
by
  rw [h1, h2]
  sorry

end nolan_saves_l380_380312


namespace chameleon_increase_l380_380352

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380352


namespace larger_root_of_quadratic_eq_l380_380533

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l380_380533


namespace triangle_expression_simplification_l380_380559

variable (a b c : ℝ)

theorem triangle_expression_simplification (h1 : a + b > c) 
                                           (h2 : a + c > b) 
                                           (h3 : b + c > a) :
  |a - b - c| + |b - a - c| - |c - a + b| = a - b + c :=
sorry

end triangle_expression_simplification_l380_380559


namespace roof_area_difference_l380_380963

variable (W : ℝ)
variable (A1 : ℝ) (L : ℝ) (W2 : ℝ) (L2 : ℝ) (A2 : ℝ)

theorem roof_area_difference (h1 : L = 4 * W) 
                              (h2 : A1 = 784) 
                              (h3 : A1 = L * W) 
                              (h4 : L2 = 5 * W2) 
                              (h5 : W2 = W) 
                              (h6 : A2 = L2 * W2) 
                              (h7 : L = 4 * W) 
                              (h8 : W = Real.sqrt (784 / 4)) 
                              (h9 : L2 = 5 * W) 
                              (h10 : L2 = 5 * 14) 
                              (h11 : W2 * 14) 
                              (h12 : A2 = 980):
  A2 - A1 = 196 := 
by 
  sorry

end roof_area_difference_l380_380963


namespace house_painting_l380_380641

theorem house_painting (n : ℕ) (h1 : n = 1000)
  (occupants : Fin n → Fin n) (perm : ∀ i, occupants i ≠ i) :
  ∃ (coloring : Fin n → Fin 3), ∀ i, coloring i ≠ coloring (occupants i) :=
by
  sorry

end house_painting_l380_380641


namespace statement_A_statement_B_statement_D_l380_380887

noncomputable def f : ℝ → ℝ := sorry -- Definition of function f is given, assumption made by 'sorry'

-- Conditions
axiom f_deriv_cos_gt_f_sin {x : ℝ} (hx : 0 < x ∧ x < π / 2) : derivative f x * cos x > f x * sin x

-- Proof of Statements
theorem statement_A : f (π / 3) > sqrt 2 * f (π / 4) :=
by {
  -- Proof goes here, omitted using sorry
  sorry
}

theorem statement_B : 2 * f (π / 4) > sqrt 6 * f (π / 6) :=
by {
  -- Proof goes here, omitted using sorry
  sorry
}

theorem statement_D : f (π / 3) > 2 * cos 1 * f 1 :=
by {
  -- Proof goes here, omitted using sorry
  sorry
}

end statement_A_statement_B_statement_D_l380_380887


namespace supplement_of_angle_l380_380961

-- Condition: The complement of angle α is 54 degrees 32 minutes
theorem supplement_of_angle (α : ℝ) (h : α = 90 - (54 + 32 / 60)) :
  180 - α = 144 + 32 / 60 := by
sorry

end supplement_of_angle_l380_380961


namespace complete_square_transform_l380_380778

theorem complete_square_transform (x : ℝ) (h : x^2 + 8*x + 7 = 0) : (x + 4)^2 = 9 :=
by sorry

end complete_square_transform_l380_380778


namespace magician_marbles_l380_380462

theorem magician_marbles (red_init blue_init red_taken : ℕ) (h1 : red_init = 20) (h2 : blue_init = 30) (h3 : red_taken = 3) :
  let blue_taken := 4 * red_taken in
  let red_left := red_init - red_taken in
  let blue_left := blue_init - blue_taken in
  red_left + blue_left = 35 := by
  sorry

end magician_marbles_l380_380462


namespace digit_150th_in_decimal_of_fraction_l380_380049

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380049


namespace cube_side_length_l380_380813

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = 1 / 4 * 6 * n^3) : n = 4 := 
by 
  sorry

end cube_side_length_l380_380813


namespace digit_150_of_17_div_70_l380_380031

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380031


namespace percentage_problem_l380_380620

variable (y x z : ℝ)

def A := y * x^2 + 3 * z - 6

theorem percentage_problem (h : A y x z > 0) :
  (2 * A y x z / 5) + (3 * A y x z / 10) = (70 / 100) * A y x z :=
by
  sorry

end percentage_problem_l380_380620


namespace classify_event_as_random_l380_380096

def num_pages : ℕ := 200
def opened_randomly (n : ℕ) : Prop := n ≤ num_pages
def lands_on_page_8 (n : ℕ) : Prop := n = 8

theorem classify_event_as_random :
  ∃ n, (opened_randomly n ∧ lands_on_page_8 n) → (random_event (opened_randomly n ∧ lands_on_page_8 n)) :=
sorry

end classify_event_as_random_l380_380096


namespace isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l380_380289

open Real

theorem isosceles_triangle_of_sine_ratio (a b c : ℝ) (A B C : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h1 : a = b * sin C + c * cos B) :
  C = π / 4 :=
sorry

theorem obtuse_triangle_of_tan_sum_neg (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π)
  (h_tan_sum : tan A + tan B + tan C < 0) :
  ∃ (E : ℝ), (A = E ∨ B = E ∨ C = E) ∧ π / 2 < E :=
sorry

end isosceles_triangle_of_sine_ratio_obtuse_triangle_of_tan_sum_neg_l380_380289


namespace find_q_l380_380405

def polynomial_q (x p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem find_q (p q r : ℝ) (h₀ : r = 3)
  (h₁ : (-p / 3) = -r)
  (h₂ : (-r) = 1 + p + q + r) :
  q = -16 :=
by
  -- h₀ implies r = 3
  -- h₁ becomes (-p / 3) = -3
  -- which results in p = 9
  -- h₂ becomes -3 = 1 + 9 + q + 3
  -- leading to q = -16
  sorry

end find_q_l380_380405


namespace digit_150th_in_decimal_of_fraction_l380_380046

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380046


namespace exists_unique_zero_point_a_value_l380_380224

theorem exists_unique_zero_point_a_value (a : ℝ) :
  (∃! x ∈ Icc 0.real.pi, a * exp x - 2 * sin x = 0) →
  a = sqrt 2 * exp (- real.pi / 4) :=
by
  sorry

end exists_unique_zero_point_a_value_l380_380224


namespace largest_c_for_4_in_range_of_quadratic_l380_380175

theorem largest_c_for_4_in_range_of_quadratic :
  (∃ x : ℝ, x^2 + 5 * x + (frac 41 4) = 4) ∧ 
  (∀ c x : ℝ, x^2 + 5 * x + c = 4 → c ≤ (frac 41 4)) :=
by
  -- There exists some x for which the given equation is satisfied
  existsi ...  -- The correct term will be filled in the actual proof
  apply ...   -- The correct justification to be supplied in the proof
  
  -- Show the maximum value of c for which the given quadratic has 4 in its range
  intro c x h
  calc
    c = ... : by ... 
    ... ≤ (41 / 4) : by ...

end largest_c_for_4_in_range_of_quadratic_l380_380175


namespace digit_150th_l380_380056

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380056


namespace log_tan_inequality_l380_380628

theorem log_tan_inequality (a : ℝ) (x : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (hx : 0 < x ∧ x < Real.pi / 4) :
  log a x > Real.tan x ↔ a ∈ Set.Icc (Real.pi / 4) 1 :=
sorry

end log_tan_inequality_l380_380628


namespace number_of_beavers_in_second_group_l380_380791

-- Define the number of beavers and the time for the first group
def numBeavers1 := 20
def time1 := 3

-- Define the time for the second group
def time2 := 5

-- Define the total work done (which is constant)
def work := numBeavers1 * time1

-- Define the number of beavers in the second group
def numBeavers2 := 12

-- Theorem stating the mathematical equivalence
theorem number_of_beavers_in_second_group : numBeavers2 * time2 = work :=
by
  -- remaining proof steps would go here
  sorry

end number_of_beavers_in_second_group_l380_380791


namespace wallis_formula_l380_380407

theorem wallis_formula : 
  (tendsto (λ m, ∏ k in finset.range m, ((2 * k + 1) / (2 * k)).to_real * ((2 * k + 2) / (2 * k + 1)).to_real) at_top (𝓝 (π / 2))) := sorry

end wallis_formula_l380_380407


namespace sequence_value_at_20_l380_380978

open Nat

def arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 4

theorem sequence_value_at_20 (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 20 = 77 :=
sorry

end sequence_value_at_20_l380_380978


namespace evaluate_expression_l380_380160

theorem evaluate_expression :
  (-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4 = 9 :=
by
  sorry

end evaluate_expression_l380_380160


namespace g_at_one_eq_l380_380305

noncomputable def g : ℝ → ℝ := sorry  -- Define 'g' as a non-constant polynomial

-- Condition: g(x - 1) + g(x) + g(x + 1) = (g(x))^2 / (4026 * x) for all nonzero x
axiom g_condition (x : ℝ) (hx : x ≠ 0) : g(x - 1) + g(x) + g(x + 1) = (g(x))^2 / (4026 * x)

-- Theorem: g(1) = 12078
theorem g_at_one_eq : g(1) = 12078 :=
sorry

end g_at_one_eq_l380_380305


namespace number_of_members_l380_380800

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l380_380800


namespace find_f_32_l380_380218

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def f : ℝ → ℝ
| x < 0     := -Real.logb 2 (-2 * x)
| _         := sorry  -- placeholder for the even function property

lemma f_is_even : is_even_function f := sorry

theorem find_f_32 : f 32 = -6 :=
by
  have f_even := f_is_even
  -- using symmetries and given conditions
  sorry

end find_f_32_l380_380218


namespace greatest_integer_of_2_7_l380_380128

theorem greatest_integer_of_2_7 : int.floor 2.7 = 2 :=
sorry

end greatest_integer_of_2_7_l380_380128


namespace digit_150_is_7_l380_380042

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380042


namespace equation1_solution_equation2_solution_l380_380380

theorem equation1_solution (x : ℝ) (h : 5 / (x + 1) = 1 / (x - 3)) : x = 4 :=
sorry

theorem equation2_solution (x : ℝ) (h : (2 - x) / (x - 3) + 2 = 1 / (3 - x)) : x = 7 / 3 :=
sorry

end equation1_solution_equation2_solution_l380_380380


namespace rate_of_current_l380_380411

theorem rate_of_current (c : ℝ) : 
(exists Vdownstream : ℝ, (Vdownstream = 12 + c) ∧ (4.8 = Vdownstream * (18 / 60))) → 
c = 4 :=
begin
  sorry
end

end rate_of_current_l380_380411


namespace length_F_to_F_l380_380414

-- Definitions and conditions based on the problem statement
def point_F : ℝ × ℝ := (-2, 3)
def point_F' : ℝ × ℝ := (-2, -3)

-- Lean statement to prove the length from F to F' is 6
theorem length_F_to_F'_is_6 : 
  let d := Real.sqrt ((point_F'.1 - point_F.1)^2 + (point_F'.2 - point_F.2)^2) in
  d = 6 := 
by 
  sorry

end length_F_to_F_l380_380414


namespace min_apples_l380_380413

theorem min_apples (N : ℕ) : 
  (N ≡ 1 [MOD 3]) ∧ 
  (N ≡ 3 [MOD 4]) ∧ 
  (N ≡ 2 [MOD 5]) 
  → N = 67 := 
by
  sorry

end min_apples_l380_380413


namespace digit_150_in_17_div_70_l380_380008

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380008


namespace sum_binom_eq_one_l380_380704

theorem sum_binom_eq_one (n : ℕ) (h : n ≥ 1) :
  ∑ k in Finset.range (⟨n, Nat.div2_le_self _⟩.val + 1),
    (-1 : ℤ)^k * (Nat.choose n k) * (Nat.choose (2 * n - 2 * k - 1) (n - 1)) = 1 :=
sorry

end sum_binom_eq_one_l380_380704


namespace heptagon_angle_sum_lt_450_l380_380801

theorem heptagon_angle_sum_lt_450 
{A1 A2 A3 A4 A5 A6 A7 : Point} {O : Point} :
  InscribedHeptagon A1 A2 A3 A4 A5 A6 A7 O → 
  CenterInsideHeptagon A1 A2 A3 A4 A5 A6 A7 O →
  ∠ A1 + ∠ A3 + ∠ A5 < 450 :=
by
  sorry

end heptagon_angle_sum_lt_450_l380_380801


namespace max_value_f_extreme_range_b_for_g_sum_inequality_l380_380598

-- Part 1
theorem max_value_f_extreme (f : ℝ → ℝ) (h : ∀ x, f x = (x / (1 + x)) - log (1 + x))
  (extreme_at_zero : ∃ x, x = 0 ∧ is_extreme f x) :
  ∃ M, M = 0 ∧ ∀ x, f x ≤ M :=
sorry

-- Part 2.①
theorem range_b_for_g (g : ℝ → ℝ) (h : ∀ x, g x = log (1 + x) - b * x) :
  ∃ b, b ∈ set.Ici 1 ∧ ∀ x, x > 0 → g x < 0 :=
sorry

-- Part 2.②
theorem sum_inequality (n : ℕ) (h_pos : n > 0) :
  -1 < (∑ k in finset.range n, k / (k^2 + 1) - log n) ∧ (∑ k in finset.range n, k / (k^2 + 1) - log n) ≤ 1/2 :=
sorry

end max_value_f_extreme_range_b_for_g_sum_inequality_l380_380598


namespace green_chameleon_increase_l380_380324

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380324


namespace problem1_problem2_l380_380603

-- Problem 1: Prove the equation of the circle Q with the segment MN as its diameter
theorem problem1 (P : ℝ × ℝ) (C : ℝ → ℝ → Prop)
  (hP : P = (2,0)) (hC : ∀ x y, C x y ↔ x^2 + y^2 - 6*x + 4*y + 4 = 0)
  (MN_length : 4) :
  ∃ Q : ℝ → ℝ → Prop, (∀ x y, Q x y ↔ (x - 2)^2 + y^2 = 4) :=
sorry

-- Problem 2: Prove there is no real number a such that the line l_2 passing through point P(2,0)
-- perpendicularly bisects the chord AB, where A and B are the intersection points of the line ax - y + 1 = 0
-- and the circle C.
theorem problem2 (P : ℝ × ℝ) (C : ℝ → ℝ → Prop) (l_2 : ℝ → ℝ → Prop)
  (hP : P = (2,0)) (hC : ∀ x y, C x y ↔ x^2 + y^2 - 6*x + 4*y + 4 = 0)
  (hL2 : ∀ x y, l_2 x y ↔ x = 2 ∧ y = 0) :
  ¬ ∃ (a : ℝ), ∀ x y, (ax - y + 1 = 0) ∧ C x y ∧ P y = l_2 P x :=
sorry

end problem1_problem2_l380_380603


namespace z_conjugate_in_fourth_quadrant_l380_380681

-- Define the complex number z
def z : ℂ := (Complex.I) / (1 + Complex.I)

-- Define the conjugate of z
def z_conjugate : ℂ := Complex.conj z

-- Define the coordinates of the conjugate of z
def z_conjugate_coord : ℝ × ℝ := (z_conjugate.re, z_conjugate.im)

-- Define a proof to show that the point lies in the fourth quadrant
theorem z_conjugate_in_fourth_quadrant :
  z_conjugate_coord.1 > 0 ∧ z_conjugate_coord.2 < 0 :=
by
  sorry

end z_conjugate_in_fourth_quadrant_l380_380681


namespace part1_part2_l380_380194

noncomputable def quadratic_roots (b c : ℝ) : ℝ × ℝ :=
  let Δ := b^2 - 4 * c in
  if h : Δ >= 0 then (1 - b + Real.sqrt Δ) / 2, (1 - b - Real.sqrt Δ) / 2
  else (0, 0)

theorem part1 (b c : ℝ) (h_eqn : c + b = 5) :
  ((x1 - 1) * (x2 - 1) = 5) ∧ (x1 = x2 + 4) → b = 5 ∨ b = -7 := sorry

theorem part2 (b : ℝ) (h_ineq : b < -1 - 2 * Real.sqrt 5 ∨ b > -1 + 2 * Real.sqrt 5) :
  ∃ (x1 x2 : ℝ), (x1 + x2 = 1 - b) ∧ (x1 * x2 = 5 - b) ∧ (x1 = x2 + 4) ∧ 
  (x1^2 + x2^2 > 12 - 4 * Real.sqrt 5) := sorry

end part1_part2_l380_380194


namespace solution_set_l380_380921

def f (x : ℝ) : ℝ := 
if x < 0 then -x + 1 else x - 1

theorem solution_set (x : ℝ) : x + (x + 1) * f (x + 1) ≤ 1 ↔ x ≤ real.sqrt 2 - 1 := by sorry

end solution_set_l380_380921


namespace approximate_roots_l380_380001

noncomputable def tg (x : ℝ) : ℝ := Real.tan x

def f (x : ℝ) : ℝ := tg x + tg (2 * x) + tg (3 * x)

theorem approximate_roots :
  (∃ x1 x2 x3 x4 x5 : ℝ, 
   f x1 = 0.1 ∧ f x2 = 0.1 ∧ f x3 = 0.1 ∧ f x4 = 0.1 ∧ f x5 = 0.1
   ∧ -90 < x1 ∧ x1 < 90 
   ∧ -90 < x2 ∧ x2 < 90 
   ∧ -90 < x3 ∧ x3 < 90 
   ∧ -90 < x4 ∧ x4 < 90 
   ∧ -90 < x5 ∧ x5 < 90 
   ∧ x1 ≈ -59.62 ∧ x2 ≈ -35.17 ∧ x3 ≈ 0.956 ∧ x4 ≈ 35.36 ∧ x5 ≈ 60.39) :=
sorry

end approximate_roots_l380_380001


namespace length_of_plot_is_55_l380_380735

noncomputable def length_of_plot (breadth : ℕ) : ℕ :=
  let length := breadth + 10 in
  let perimeter := 2 * (length + breadth) in
  let total_cost := 5300 in
  let cost_per_meter := 26.50 in
  let expected_perimeter := total_cost / cost_per_meter in
  if H : expected_perimeter = perimeter then length
  else 0 -- provide a non-valid length if condition fails

theorem length_of_plot_is_55 : length_of_plot 45 = 55 :=
by
  dsimp [length_of_plot]
  norm_num
  sorry

end length_of_plot_is_55_l380_380735


namespace city_a_received_l380_380398

theorem city_a_received (cityB_sand : ℝ) (cityC_sand : ℝ) (cityD_sand : ℝ) (total_sand : ℝ) :
  cityB_sand = 26 →
  cityC_sand = 24.5 →
  cityD_sand = 28 →
  total_sand = 95 →
  (total_sand - (cityB_sand + cityC_sand + cityD_sand) = 16.5) :=
by
  intros hB hC hD hTotal
  rw [hB, hC, hD, hTotal]
  norm_num
  exact (show 95 - (26 + 24.5 + 28) = 16.5 by norm_num)

end city_a_received_l380_380398


namespace sky_colors_l380_380522

theorem sky_colors (h1 : ∀ t : ℕ, t = 2) (h2 : ∀ m : ℕ, m = 60) (h3 : ∀ c : ℕ, c = 10) : 
  ∃ n : ℕ, n = 12 :=
by
  let total_duration := (2 * 60 : ℕ)
  let num_colors := total_duration / 10
  have : num_colors = 12 := by decide
  use num_colors
  assumption_needed

end sky_colors_l380_380522


namespace part1_part2_l380_380595

noncomputable def f (a x : ℝ) : ℝ := x - 1 - a * Real.log x

theorem part1 (h : ∀ x > 0, f a x ≥ 0) : a = 1 :=
  sorry

theorem part2 (h : ∀ n : ℕ, n > 0 → ∏ i in Finset.range n, (1 + 1 / 2 ^ (i + 1)) < m) : m = 3 :=
  sorry

end part1_part2_l380_380595


namespace num_possible_radii_l380_380134

theorem num_possible_radii:
  ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 60 ∧ (2 * r * π ∣ 120 * π)) ∧ 
  S.card = 11 := 
sorry

end num_possible_radii_l380_380134


namespace find_line_equation_l380_380898

theorem find_line_equation
  (P : Point := ⟨1, 2⟩)
  (length_AB : ℝ := Real.sqrt 2)
  (l1 : Line := {a := 4, b := 3, c := 1})
  (l2 : Line := {a := 4, b := 3, c := 6})
  (l_intersections : ∃ l : Line, 
    ∀ A B : Point, line_intersects_at l l1 A ∧ line_intersects_at l l2 B → 
      length (A - B) = length_AB) :
  (l = {a := 7, b := -1, c := -5}) ∨ (l = {a := 1, b := 7, c := -15}) := 
sorry

end find_line_equation_l380_380898


namespace find_real_solutions_l380_380861

theorem find_real_solutions (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  ( (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) ) / ( (x - 4) * (x - 5) ) = -1 ↔ x = 10 / 3 ∨ x = 2 / 3 :=
by sorry

end find_real_solutions_l380_380861


namespace find_train_speed_l380_380095

variable (bridge_length train_length train_crossing_time : ℕ)

def speed_of_train (bridge_length train_length train_crossing_time : ℕ) : ℕ :=
  (bridge_length + train_length) / train_crossing_time

theorem find_train_speed
  (bridge_length : ℕ) (train_length : ℕ) (train_crossing_time : ℕ)
  (h_bridge_length : bridge_length = 180)
  (h_train_length : train_length = 120)
  (h_train_crossing_time : train_crossing_time = 20) :
  speed_of_train bridge_length train_length train_crossing_time = 15 := by
  sorry

end find_train_speed_l380_380095


namespace pink_highlighters_count_l380_380638

-- Define the necessary constants and types
def total_highlighters : ℕ := 12
def yellow_highlighters : ℕ := 2
def blue_highlighters : ℕ := 4

-- We aim to prove that the number of pink highlighters is 6
theorem pink_highlighters_count : ∃ (pink_highlighters : ℕ), 
  pink_highlighters = total_highlighters - (yellow_highlighters + blue_highlighters) ∧
  pink_highlighters = 6 :=
by
  sorry

end pink_highlighters_count_l380_380638


namespace power_division_simplify_l380_380838

theorem power_division_simplify :
  ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end power_division_simplify_l380_380838


namespace green_chameleon_increase_l380_380333

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380333


namespace digits_11120_four_digit_multiple_of_5_l380_380972

theorem digits_11120_four_digit_multiple_of_5 : 
  ∀ (digits : list ℕ), digits = [1, 1, 1, 2, 0] → 
  (∀ n : ℕ, (n ∈ digits → n = 0 ∨ n = 1 ∨ n = 2) →
    let arrangements := list.permutations digits in 
    let valid_numb := arrangements.filter (λ x, list.length x = 4 ∧ list.last' x = some 0) in
    list.length valid_numb = 4) :=
by
  intros digits hdigits hdigits_cond arrangements valid_numb
  sorry

end digits_11120_four_digit_multiple_of_5_l380_380972


namespace min_value_part1_l380_380083

open Real

theorem min_value_part1 (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by {
  sorry
}

end min_value_part1_l380_380083


namespace heather_average_balance_l380_380484

theorem heather_average_balance :
  let balance_J := 150
  let balance_F := 250
  let balance_M := 100
  let balance_A := 200
  let balance_May := 300
  let total_balance := balance_J + balance_F + balance_M + balance_A + balance_May
  let avg_balance := total_balance / 5
  avg_balance = 200 :=
by
  sorry

end heather_average_balance_l380_380484


namespace similar_rect_tiling_l380_380710

-- Define the dimensions of rectangles A and B
variables {a1 a2 b1 b2 : ℝ}

-- Define the tiling condition
def similar_tiled (a1 a2 b1 b2 : ℝ) : Prop := 
  -- A placeholder for the actual definition of similar tiling
  sorry

-- The main theorem to prove
theorem similar_rect_tiling (h : similar_tiled a1 a2 b1 b2) : similar_tiled b1 b2 a1 a2 :=
sorry

end similar_rect_tiling_l380_380710


namespace line_intersects_at_fixed_point_l380_380575

def eq_of_ellipse (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) : Prop :=
  (eq_of_ellipse0 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1)

theorem line_intersects_at_fixed_point (m : ℝ) (C : ℝ → ℝ → Prop)
  (ha : ∀ x y : ℝ, C x y → x^2 + 4 * y^2 = 4)
  (m_ne_0 : m ≠ 0)
  (P Q P1 : ℝ × ℝ)
  (P_line : P.fst = m * [2].snd + 1)
  (Q_line : Q.fst = m * Q.snd + 1)
  (P1_line : P1.fst = P.fst)
  (P1_is_reflect : P1.snd = -P.snd)
  (P_ne_Q : P1 ≠ Q)
  : ∃ x : ℝ, ∀ y : ℝ, y = 0 → (∃ P1 Q : ℝ × ℝ, P1 = (x, y)) :=
begin
  sorry
end

end line_intersects_at_fixed_point_l380_380575


namespace digit_150_is_7_l380_380040

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380040


namespace chameleon_problem_l380_380347

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380347


namespace point_coordinates_respect_to_origin_l380_380652

-- Define the coordinates of point P
def pointP : ℝ × ℝ := (-2, -4)

-- The theorem statement
theorem point_coordinates_respect_to_origin :
  pointP = (-2, -4) :=
by
sorr

end point_coordinates_respect_to_origin_l380_380652


namespace traveled_distance_l380_380833

def distance_first_day : ℕ := 5 * 7
def distance_second_day_part1 : ℕ := 6 * 6
def distance_second_day_part2 : ℕ := (6 / 2) * 3
def distance_third_day : ℕ := 7 * 5

def total_distance : ℕ := distance_first_day + distance_second_day_part1 + distance_second_day_part2 + distance_third_day

theorem traveled_distance : total_distance = 115 := by
  unfold total_distance
  unfold distance_first_day distance_second_day_part1 distance_second_day_part2 distance_third_day
  norm_num
  rfl

end traveled_distance_l380_380833


namespace paul_initial_books_l380_380364

theorem paul_initial_books (sold_percentage : ℝ) (remaining_books : ℕ) 
  (h_sold : sold_percentage = 0.60) (h_remaining : remaining_books = 27) : 
  (initial_books : ℕ), 
  initial_books = 68 :=
by 
  sorry

end paul_initial_books_l380_380364


namespace volume_of_pure_water_added_l380_380793

theorem volume_of_pure_water_added 
  (V0 : ℝ) (P0 : ℝ) (Pf : ℝ) 
  (V0_eq : V0 = 50) 
  (P0_eq : P0 = 0.30) 
  (Pf_eq : Pf = 0.1875) : 
  ∃ V : ℝ, V = 30 ∧ (15 / (V0 + V)) = Pf := 
by
  sorry

end volume_of_pure_water_added_l380_380793


namespace find_fraction_l380_380384

theorem find_fraction (c d : ℕ) (h1 : 435 = 2 * 100 + c * 10 + d) :
  (c + d) / 12 = 5 / 6 :=
by sorry

end find_fraction_l380_380384


namespace expansion_correct_l380_380858

noncomputable def P (x y : ℝ) : ℝ := 2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9

noncomputable def M (x : ℝ) : ℝ := 3 * x^7

theorem expansion_correct (x y : ℝ) :
  (P x y) * (M x) = 6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 :=
by
  sorry

end expansion_correct_l380_380858


namespace jordan_reads_more_l380_380999

-- Definitions of Jordan and Alexandre's readings
def jordan_read : ℕ := 120
def proportion_read_by_alexandre : ℚ := 1 / 10
def alexandre_read : ℕ := jordan_read * proportion_read_by_alexandre

-- Theorem statement
theorem jordan_reads_more : jordan_read - alexandre_read = 108 := by
  sorry

end jordan_reads_more_l380_380999


namespace log2_eq_3_solutions_l380_380550

theorem log2_eq_3_solutions (x : ℝ) : 
  log 2 (x^3 - 15*x^2 + 60*x) = 3 ↔ (x = 2 ∨ x = (13 + real.sqrt 153) / 2 ∨ x = (13 - real.sqrt 153) / 2) := 
sorry

end log2_eq_3_solutions_l380_380550


namespace angle_BD1_AM_l380_380438

def angle_between_lines {α : Type*} [inner_product_space ℝ α] (x y z : α) (a : ℝ) (H : (2 * a) = (norm z)) : ℝ :=
arc_cos ((5:ℝ) / (3 * real.sqrt 6))

theorem angle_BD1_AM {α : Type*} [inner_product_space ℝ α]
  (x y z : α) (a : ℝ) (H : (2 * a) = (norm z)) :
  ∃ M : α, angle_between_lines x y z a H = arc_cos ((5:ℝ) / (3 * real.sqrt 6)) := 
sorry

end angle_BD1_AM_l380_380438


namespace red_balls_in_bag_l380_380079

theorem red_balls_in_bag (total_balls : ℕ) (red_ratio : ℕ) (green_ratio : ℕ) (blue_ratio : ℕ) 
  (H1 : total_balls = 2400) 
  (H2 : red_ratio = 15) 
  (H3 : green_ratio = 13) 
  (H4 : blue_ratio = 17) : 
  let total_ratio := red_ratio + green_ratio + blue_ratio in 
  let balls_per_part := total_balls / total_ratio in 
  red_ratio * balls_per_part = 795 := by 
{
  let total_ratio := red_ratio + green_ratio + blue_ratio,
  let balls_per_part := total_balls / total_ratio,
  have h1 : total_ratio = 45 := by { rw [H2, H3, H4] ; refl },
  have h2 : balls_per_part = 53 := by { rw [H1, h1] ; exact Nat.div_eq_of_lt_of_le_of_nonneg _ _ (Nat.zero_le 2400) (Nat.succ_le_of_lt Nat.prime_lt_prime_of_ne Nat.even_prime Nat.odd_neven) },
  show red_ratio * balls_per_part = 795,
  rw [H2, h2],
  exact rfl,
}

end red_balls_in_bag_l380_380079


namespace problem1_problem2_problem3_problem4_l380_380725

-- Definitions for the function and conditions
variable (f : ℝ → ℝ)
variable (D : Set ℝ) (hD : ∀ x, x ∈ D ↔ x ≠ 0)
variable (h_func : ∀ x1 x2, x1 ∈ D → x2 ∈ D → f(x1) + f(x2) = f(x1 * x2))

-- Problem 1: Prove f(-1) = 0 and f is an even function
theorem problem1 :
  f(-1) = 0 ∧ ∀ x ∈ D, f(-x) = f(x) :=
sorry

-- Problem 2: Given f(-4) = 4, prove sum of first 2015 terms of a_n is -4062240
variable (f_neg4 : f(-4) = 4)
variable (a_n : ℕ → ℝ) (ha_n : ∀ n, n ≥ 1 → a_n n = (-1)^n * f(2^n))

theorem problem2 :
  ∑ k in Finset.range 2015, a_n (k + 1) = -4062240 :=
sorry

-- Problem 3 (Science track): For x > 1, f(x) < 0
variable (h_fx_neg : ∀ x, 1 < x → f(x) < 0)
variable (ineq : ∀ x y, 0 < x → 0 < y → f(Real.sqrt(x^2 + y^2)) ≤ f(Real.sqrt(xy)) + f(a))

theorem problem3 :
  0 < |a| ∧ |a| ≤ Real.sqrt(2) :=
sorry

-- Problem 4 (Arts track): Solve inequality with respect to x
variable (h_fx_pos : ∀ x, x > 1 → f(x) < 0)

theorem problem4 :
  ∀ x, 1 < x → f(x - 3) ≥ 0 → (2 ≤ x ∧ x < 3) ∨ (3 ≤ x ∧ x ≤ 4) :=
sorry

end problem1_problem2_problem3_problem4_l380_380725


namespace sky_color_changes_l380_380520

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l380_380520


namespace range_of_a_l380_380748

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (9 - 5 * x) / 4 > 1 → x < 1) → x < a → a ≥ 1 :=
by
    assume h1 : ∀ x : ℝ, (9 - 5 * x) / 4 > 1 → x < 1,
    assume h2 : x < a,
    sorry

end range_of_a_l380_380748


namespace calculate_length_X_l380_380271

theorem calculate_length_X 
  (X : ℝ)
  (h1 : 3 + X + 4 = 5 + 7 + X)
  : X = 5 :=
sorry

end calculate_length_X_l380_380271


namespace intersecting_functions_k_range_l380_380216

theorem intersecting_functions_k_range 
  (k : ℝ) (h : 0 < k) : 
    ∃ x : ℝ, -2 * x + 3 = k / x ↔ k ≤ 9 / 8 :=
by 
  sorry

end intersecting_functions_k_range_l380_380216


namespace original_data_properties_l380_380187

variables (X : Type*)
variables [Nonempty X] [Fintype X] [DecidableEq X] (f : X → ℝ)

-- Define the transformation
def g (x : X) : ℝ := f x - 80

-- Given conditions:
#check (finset.univ.sum (λ x, g x) / finset.univ.card X = 1.2)
#check (finset.univ.sum (λ x, (g x - 1.2) ^ 2) / finset.univ.card X = 4.4)

-- Prove the statement:
theorem original_data_properties :
  (finset.univ.sum (λ x, f x) / finset.univ.card X = 81.2) ∧ 
  (finset.univ.sum (λ x, (f x - 81.2) ^ 2) / finset.univ.card X = 4.4) :=
sorry

end original_data_properties_l380_380187


namespace find_f_log4_inv9_l380_380879

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem find_f_log4_inv9 : 
  (∀ x : ℝ, x > 0 → f(x) = 2^x) → 
  is_even f → 
  f (Real.log (1 / 9) / Real.log 4) = 3 := 
by
  sorry

end find_f_log4_inv9_l380_380879


namespace line_through_P_with_direction_AB_is_standard_form_l380_380899

-- Define points A, B, and P
structure Point (ℝ : Type _) := 
  (x : ℝ)
  (y : ℝ)

-- Define the points A, B, P
def A := Point.mk 2 3
def B := Point.mk 4 (-5)
def P := Point.mk 1 2

-- Define the direction vector AB
def direction_vector (A B : Point ℝ) : Point ℝ := 
  Point.mk (B.x - A.x) (B.y - A.y)

-- Define the slope of the line
def slope (d : Point ℝ) : ℝ := d.y / d.x

-- The equation for a line in point-slope form
def line_eq (P : Point ℝ) (k : ℝ) : ℝ → ℝ := 
  λ x, k * (x - P.x) + P.y

-- Hypothesis definitions, which are derived from points and slope
def line_in_standard_form : ℝ → ℝ → ℝ := 
  λ x y, 4 * x + y - 6

-- Lean theorem statement
theorem line_through_P_with_direction_AB_is_standard_form : 
  line_eq P (slope (direction_vector A B)) = line_in_standard_form :=
by
  sorry

end line_through_P_with_direction_AB_is_standard_form_l380_380899


namespace carla_must_clean_34_pieces_per_hour_l380_380842

def pieces_of_laundry_per_hour (total_pieces hours : ℝ) : ℝ :=
  total_pieces / hours

theorem carla_must_clean_34_pieces_per_hour : 
  pieces_of_laundry_per_hour 150 4.5 >= 34 :=
by
  sorry

end carla_must_clean_34_pieces_per_hour_l380_380842


namespace length_of_AC_l380_380135

noncomputable def circle_radius (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

theorem length_of_AC
  (circumference_u : ℝ) (h1 : circumference_u = 18 * Real.pi)
  (angle_UAC : ℝ) (h2 : angle_UAC = 30 * Real.pi / 180) :
  let r := circle_radius circumference_u in
  let AC := Real.sqrt (r^2 + r^2 - 2 * r * r * Real.cos angle_UAC) in
  AC = 9 * Real.sqrt (2 - Real.sqrt 3) :=
by
  sorry

end length_of_AC_l380_380135


namespace find_a_l380_380623

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l380_380623


namespace inverse_of_A_l380_380532

def mat := Matrix (Fin 2) (Fin 2) Int

def A : mat := ![![7, -5], ![-4, 3]]
def A_inv : mat := ![![3, 5], ![4, 7]]

theorem inverse_of_A : (A * A_inv = 1) ∧ (A_inv * A = 1) :=
sorry

end inverse_of_A_l380_380532


namespace fraction_identity_l380_380503

variable (a b : ℝ)

theorem fraction_identity (h : a ≠ 0) : 
  (2 * b + a) / a + (a - 2 * b) / a = 2 := 
by
  sorry

end fraction_identity_l380_380503


namespace value_of_a_set_of_x_l380_380591

open Real

noncomputable def f (x a : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6) + cos x + a

theorem value_of_a : ∀ a, (∀ x, f x a ≤ 1) → a = -1 :=
sorry

theorem set_of_x (a : ℝ) (k : ℤ) : a = -1 →
  {x : ℝ | f x a = 0} = {x | ∃ k : ℤ, x = 2 * k * π ∨ x = 2 * k * π + 2 * π / 3} :=
sorry

end value_of_a_set_of_x_l380_380591


namespace overall_gain_percent_l380_380447

-- Definitions based on the conditions
def CP_A : ℝ := 100
def SP_A : ℝ := 120
def CP_B : ℝ := 150
def SP_B : ℝ := 180
def CP_C : ℝ := 200
def SP_C : ℝ := 210

-- Compute overall gain percent
theorem overall_gain_percent : 
  let TCP := CP_A + CP_B + CP_C in
  let TSP := SP_A + SP_B + SP_C in
  let Gain := TSP - TCP in
  let GainPercent := (Gain / TCP) * 100 in
  GainPercent = 13.33 :=
by
  -- Skipping the proof steps
  sorry

end overall_gain_percent_l380_380447


namespace second_player_win_strategy_l380_380696

theorem second_player_win_strategy:
  ∃ strategy : (ℕ → ℕ) → ℕ, 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
    (strategy n + n = 1001) ∧
    (strategy n - n) % 13 = 0) :=
sorry

end second_player_win_strategy_l380_380696


namespace Saheed_earnings_l380_380376

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end Saheed_earnings_l380_380376


namespace sum_of_numbers_1_to_100_l380_380549

theorem sum_of_numbers_1_to_100 : (∑ k in finset.range 101, k) = 5050 := by
  sorry

end sum_of_numbers_1_to_100_l380_380549


namespace complement_union_l380_380243

theorem complement_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  U \ (M ∪ N) = {1, 6} :=
by
  sorry

end complement_union_l380_380243


namespace increasing_function_in_interval_l380_380487

theorem increasing_function_in_interval :
  (∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = x * exp(x) ∧ (∀ y: ℝ, 0 < y → y * exp(y) > 0))) ∧
  (¬(∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = sin x ∧ (∀ y: ℝ, 0 < y → sin y > 0)))) ∧
  (¬(∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = x^3 - x ∧ (∀ y: ℝ, 0 < y → y^3 - y > 0)))) ∧
  (¬(∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = log x - x ∧ (∀ y: ℝ, 0 < y → log y - y > 0)))) := by
  sorry

end increasing_function_in_interval_l380_380487


namespace problem1_interval_monotonically_increasing_problem2_range_of_a_problem3_f_x1_f_x2_l380_380594

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - b * x + Real.log x

theorem problem1_interval_monotonically_increasing :
  ∃ (i1 i2 : Set ℝ), f 1 3 '' (i1 ∪ i2) = {x | x > 0 ∧ f 1 3 '' i1 ⊆ {y | ∀ x, ∃ ε > 0, ∀ δ ∈ (-ε, ε), f 1 3 (x + δ) > f 1 3 x ∧ f 1 3 '' i2 ⊆ {y | ∀ x, ∃ ε > 0, ∀ δ ∈ (-ε, ε), f 1 3 (x + δ) > f 1 3 x }} :=
sorry

theorem problem2_range_of_a (a : ℝ) :
  (∀ x ≥ 1, f a 0 x ≤ 0) ↔ a ≤ -1 / (2 * Real.exp 1) :=
sorry

theorem problem3_f_x1_f_x2 (b : ℝ) (hb : b > 9/2) (x1 x2 : ℝ) (hx : x1 < x2) (H : (2* x1^2 - b * x1 + 1 = 0) ∧ (2 * x2^2 - b * x2 + 1 = 0)) :
  f 1 b x1 - f 1 b x2 > 63/16 - 3 * Real.log 2 :=
sorry

end problem1_interval_monotonically_increasing_problem2_range_of_a_problem3_f_x1_f_x2_l380_380594


namespace min_value_tan_cot_expression_l380_380850

variable {x : ℝ}

-- Conditions
def tan_cot_identity (x : ℝ) : Prop := Real.tan x * Real.cot x = 1
def sec_squared (x : ℝ) : Prop := Real.sec x^2 = 1 + Real.tan x^2
def csc_squared (x : ℝ) : Prop := Real.csc x^2 = 1 + Real.cot x^2
def sine_cosine_identity (x : ℝ) : Prop := Real.sin x^2 + Real.cos x^2 = 1
def sin_cos_double_angle (x : ℝ) : Prop := Real.sin x * Real.cos x = 1/2 * Real.sin (2 * x)

-- Theorem statement
theorem min_value_tan_cot_expression : 
  (∀ x, 0 < x ∧ x < π/2 → 
    tan_cot_identity x ∧
    sec_squared x ∧
    csc_squared x ∧
    sine_cosine_identity x ∧
    sin_cos_double_angle x →
    (Real.tan x + Real.cot x)^2 + 4 = 8) := sorry

end min_value_tan_cot_expression_l380_380850


namespace invalid_votes_l380_380493

theorem invalid_votes (W L total_polls : ℕ) 
  (h1 : total_polls = 90830) 
  (h2 : L = 9 * W / 11) 
  (h3 : W = L + 9000)
  (h4 : 100 * (W + L) = 90000) : 
  total_polls - (W + L) = 830 := 
sorry

end invalid_votes_l380_380493


namespace integral_represents_area_of_shape_l380_380891

variable {a : ℝ}
variable (ha : a > 0)

def integral_expression : ℝ :=
  ∫ x in 0..a, (a^2 - x^2)

theorem integral_represents_area_of_shape :
  integral_expression a ha = ∫ x in 0..a, (a^2 - x^2) := sorry

end integral_represents_area_of_shape_l380_380891


namespace general_term_arithmetic_general_term_geometric_cn_arithmetic_sequence_tn_value_l380_380912

open Real

-- Conditions and preliminary definitions
def a (n : ℕ) : ℝ := 3 * n - 1
def b (n : ℕ) : ℝ := 2 * 3^(n - 1)
def c (n : ℕ) : ℝ := log 3 (b n)
def T (n : ℕ) : ℝ := (27^n - 1) / 13

-- Mathematical problem statements

-- Prove the general term of the arithmetic sequence a_n
theorem general_term_arithmetic :
  a 3 = 8 ∧ a 6 = 17 → ∀ n, a n = 3 * n - 1 := by sorry

-- Prove the general term of the geometric sequence b_n
theorem general_term_geometric :
  b 1 = 2 ∧ b 1 * b 2 * b 3 = 9 * (a 2 + a 3 + a 4) → ∀ n, b n = 2 * 3^(n - 1) := by sorry

-- Prove that c_n = log_3 b_n forms an arithmetic sequence
theorem cn_arithmetic_sequence :
  ∀ n, c n = log 3 (b n) →
  ∃ d, c 1 = log 3 2 ∧ d = 1 ∧ ∀ n, c (n + 1) - c n = d := by sorry

-- Prove the value of T_n
theorem tn_value (n : ℕ) (h : n > 0) :
  T n = b 1 + b 4 + b 7 + ⋯ + b (3 * n - 2) → T n = (27^n - 1) / 13 := by sorry

end general_term_arithmetic_general_term_geometric_cn_arithmetic_sequence_tn_value_l380_380912


namespace digit_150th_l380_380052

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380052


namespace option_C_correct_l380_380781

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct_l380_380781


namespace eccentricity_of_hyperbola_l380_380208

open Real

variables {a b c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b)

def hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1
def circle (x y : ℝ) := x^2 + y^2 = a^2
def foci {a b : ℝ} (x : ℝ) := sqrt (a^2 + b^2)

theorem eccentricity_of_hyperbola 
  {x y : ℝ}
  (h1 : hyperbola a b x y)
  (h2 : ∃ P, (P.1^2 + P.2^2 = a^2) ∧ (P.1 = -c) ∧ (P.2 = y))
  (h3 : abs |P.1 - F₁.1| = 3 * abs |P.1 - F₁.2|) :
  let e := (sqrt (c^2 + a^2)) / a in e = sqrt 6 / 2 :=
by
  sorry

end eccentricity_of_hyperbola_l380_380208


namespace cost_price_600_l380_380099

variable (CP SP : ℝ)

theorem cost_price_600 
  (h1 : SP = 1.08 * CP) 
  (h2 : SP = 648) : 
  CP = 600 := 
by
  sorry

end cost_price_600_l380_380099


namespace repeating_decimal_as_fraction_l380_380164

theorem repeating_decimal_as_fraction : 
  let x := 2.353535... in
  x = 233 / 99 ∧ Nat.gcd 233 99 = 1 :=
by
  sorry

end repeating_decimal_as_fraction_l380_380164


namespace find_angle_x_eq_38_l380_380282

theorem find_angle_x_eq_38
  (angle_ACD angle_ECB angle_DCE : ℝ)
  (h1 : angle_ACD = 90)
  (h2 : angle_ECB = 52)
  (h3 : angle_ACD + angle_ECB + angle_DCE = 180) :
  angle_DCE = 38 :=
by
  sorry

end find_angle_x_eq_38_l380_380282


namespace max_integer_solutions_l380_380807

noncomputable def p (x : ℤ) : ℤ := sorry  -- Since p is a polynomial function with integer coefficients.

-- Conditions:
axiom int_coeffs : (∀ n : ℤ, ∃ m : ℤ, p(n) = m)  -- p has integer coefficients.
axiom p_at_100 : p(100) = 100  -- Given condition p(100) = 100.

-- Definition of q(x):
noncomputable def q (x : ℤ) : ℤ := p(x) - (x * x)

-- Main theorem stating the problem
theorem max_integer_solutions :
  ∃ k_max : ℕ, (∀ k, (p(k) = k * k → k ≤ k_max)) ∧ k_max = 6 :=
begin
  sorry  -- Proof omitted
end

end max_integer_solutions_l380_380807


namespace distance_T1_T2_distance_T2_T1_l380_380433

-- Define the triangles with side lengths 1 and 2
structure EquilateralTriangle :=
  (centroid : ℝ × ℝ)
  (side_length : ℝ)

noncomputable def T1 : EquilateralTriangle := { centroid := (0, 0), side_length := 1 }
noncomputable def T2 : EquilateralTriangle := { centroid := (0, 0), side_length := 2 }

def distance_EquilateralTriangles (T1 T2 : EquilateralTriangle) : ℝ :=
  let s1 := T1.side_length
  let s2 := T2.side_length
  (s2 * real.sqrt 3 / 2) - (s1 * real.sqrt 3 / 6)

def distance_EquilateralTriangles_reverse (T2 T1 : EquilateralTriangle) : ℝ :=
  distance_EquilateralTriangles T1 T2 * 2

theorem distance_T1_T2 : distance_EquilateralTriangles T1 T2 = real.sqrt 3 / 6 := by
  sorry

theorem distance_T2_T1 : distance_EquilateralTriangles_reverse T2 T1 = real.sqrt 3 / 3 := by
  sorry

end distance_T1_T2_distance_T2_T1_l380_380433


namespace solve_for_x_l380_380528

theorem solve_for_x (x : ℝ) (h : 3^(3 * x - 2) = (1 : ℝ) / 27) : x = -(1 : ℝ) / 3 :=
sorry

end solve_for_x_l380_380528


namespace length_LD_l380_380572

-- Definitions of points and the square
variables (a : ℝ)  -- side length of the square
def D := (0 : ℝ, 0)  -- origin
def A := (a, 0)
def B := (a, a)
def C := (0, a)
def L := (6, 0)
def K := (a + 19, 0)

-- Main hypothesis in the problem
axiom angle_KBL_90 : ∠(K, B, L) = 90

-- Given conditions
axiom KD_19 : dist K D = 19
axiom CL_6 : dist C L = 6

-- The proof goal is to show the length of LD equals 7
theorem length_LD (h1 : (A, B, C, D).is_square)
    (h2 : KD_19)
    (h3 : CL_6)
    (h4 : angle_KBL_90) : dist L D = 7 :=
begin
    sorry
end

end length_LD_l380_380572


namespace range_of_x_l380_380614

variables (x m : ℝ)

def a : ℝ × ℝ := (1, x)
def b : ℝ × ℝ := (x^2 + x, -x)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def cond1 : Prop := m ≤ -2
def inequality : Prop := dot_product a b + 2 > m * (2 / dot_product a b + 1)

theorem range_of_x (h1 : cond1) : 
  (m < -2 → (m < x ∧ x < -2) ∨ 0 < x) ∧ (m = -2 → 0 < x) := 
sorry

end range_of_x_l380_380614


namespace points_per_touchdown_l380_380987

theorem points_per_touchdown (P : ℕ) (games : ℕ) (touchdowns_per_game : ℕ) (two_point_conversions : ℕ) (two_point_conversion_value : ℕ) (total_points : ℕ) :
  touchdowns_per_game = 4 →
  games = 15 →
  two_point_conversions = 6 →
  two_point_conversion_value = 2 →
  total_points = (4 * P * 15 + 6 * two_point_conversion_value) →
  total_points = 372 →
  P = 6 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end points_per_touchdown_l380_380987


namespace max_min_sum_zero_l380_380402

def cubic_function (x : ℝ) : ℝ :=
  x^3 - 3 * x

def first_derivative (x : ℝ) : ℝ :=
  3 * x^2 - 3

theorem max_min_sum_zero :
  let m := cubic_function (-1);
  let n := cubic_function 1;
  m + n = 0 :=
by
  sorry

end max_min_sum_zero_l380_380402


namespace ratio_of_intersection_l380_380285

def TrianglePrism (A B C A1 B1 C1 : Type) := 
  ∃ AB CC1 A1B1 CB: Type, 
    (midpoint C C1 = M) ∧ 
    (midpoint C B = N) 

theorem ratio_of_intersection
  (A B C A1 B1 C1 : Type)
  (M N : Type)
  (h1 : midpoint C C1 = M)
  (h2 : midpoint C B = N)
  (L : ℝ) :
  (length_of_intersection_line_segment_within_prism A B C A1 B1 C1 M N) / (length_of_edge AB) = 3 / 5 :=
sorry

end ratio_of_intersection_l380_380285


namespace larger_solution_of_quadratic_equation_l380_380542

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l380_380542


namespace digit_150th_l380_380057

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380057


namespace problem_l380_380513

-- Define the operation table as a function
def op : ℕ → ℕ → ℕ
| 1, 1 := 1
| 1, 2 := 2
| 1, 3 := 3
| 1, 4 := 4
| 2, 1 := 2
| 2, 2 := 4
| 2, 3 := 1
| 2, 4 := 3
| 3, 1 := 3
| 3, 2 := 1
| 3, 3 := 4
| 3, 4 := 2
| 4, 1 := 4
| 4, 2 := 3
| 4, 3 := 2
| 4, 4 := 1
| _, _ := 0  -- handle cases outside 1 to 4

theorem problem : op (op 2 3) (op 4 2) = 3 := by
  -- Provided conditions
  have h1 : op 2 3 = 1 := rfl
  have h2 : op 4 2 = 3 := rfl
  have h3 : op 1 3 = 3 := rfl
  -- Conclusion
  show op (op 2 3) (op 4 2) = 3, from (rfl : op (op 2 3) (op 4 2) = 3)


end problem_l380_380513


namespace find_x_l380_380064

theorem find_x (
  w : ℕ,
  z : ℕ,
  y : ℕ,
  x : ℕ
)
  (h_w : w = 50)
  (h_z : z = w + 25)
  (h_y : y = z + 15)
  (h_x : x = y + 7) :
  x = 97 :=
sorry

end find_x_l380_380064


namespace has_four_digits_l380_380734

def least_number_divisible (n: ℕ) : Prop := 
  n = 9600 ∧ 
  (∃ k1 k2 k3 k4: ℕ, n = 15 * k1 ∧ n = 25 * k2 ∧ n = 40 * k3 ∧ n = 75 * k4)

theorem has_four_digits : ∀ n: ℕ, least_number_divisible n → (Nat.digits 10 n).length = 4 :=
by
  intros n h
  sorry

end has_four_digits_l380_380734


namespace train_crossing_man_time_l380_380106

/-- Given the length of the train, speed of the man, and speed of the train, find the time it takes for the train to cross the man. -/
theorem train_crossing_man_time
  (S_man : ℝ) -- speed of the man in kmph
  (S_train : ℝ) -- speed of the train in kmph
  (L_train : ℝ) -- length of the train in meters
  (h_S_man : S_man = 5) -- man speed is 5 kmph
  (h_S_train : S_train = 24.997600191984645) -- train speed is 24.997600191984645 kmph
  (h_L_train : L_train = 50) -- length of the train is 50 meters
  : approx ((L_train) / ((S_train + S_man) * (5 / 18))) 6 := 
sorry

end train_crossing_man_time_l380_380106


namespace third_part_is_correct_l380_380959

def total_amount : Real := 1250
def ratio_A : Real := 3 / 5
def ratio_B : Real := 2 / 7
def ratio_C : Real := 4 / 9
def ratio_D : Real := 3 / 8
def ratio_E : Real := 5 / 7

def total_parts : Real :=
  (3 * 5) + (2 * 7) + (4 * 9) + (3 * 8) + (5 * 7)

def third_part_value : Real :=
  (total_amount / total_parts) * (4 * 9)

theorem third_part_is_correct :
  third_part_value ≈ 362.90 := sorry

end third_part_is_correct_l380_380959


namespace cars_in_fourth_store_l380_380073

theorem cars_in_fourth_store
  (mean : ℝ) 
  (a1 a2 a3 a5 : ℝ) 
  (num_stores : ℝ) 
  (mean_value : mean = 20.8) 
  (a1_value : a1 = 30) 
  (a2_value : a2 = 14) 
  (a3_value : a3 = 14) 
  (a5_value : a5 = 25) 
  (num_stores_value : num_stores = 5) :
  ∃ x : ℝ, (a1 + a2 + a3 + x + a5) / num_stores = mean ∧ x = 21 :=
by
  sorry

end cars_in_fourth_store_l380_380073


namespace longest_gap_ultra_even_years_l380_380790

def ultra_even (y : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ (Nat.digits 10 y) → d ∈ {0, 2, 4, 6, 8}

def in_year_range (y : ℕ) : Prop :=
  1 ≤ y ∧ y ≤ 10000

noncomputable def longest_gap (gap : ℕ) : Prop :=
  ∀ (y1 y2 : ℕ), in_year_range y1 → in_year_range y2 → ultra_even y1 → ultra_even y2 → y2 - y1 = gap → 
  ∀ (y3 : ℕ), in_year_range y3 → ultra_even y3 → (y1 < y3 ∧ y3 < y2) → False

theorem longest_gap_ultra_even_years : longest_gap 1112 :=
  sorry

end longest_gap_ultra_even_years_l380_380790


namespace proof_it_eq_ia_l380_380581

variables {A B C D E I A1 T : Type}
-- Assume the necessary geometric entities and relations
variables [geometry.points A B C D E I A1 T]
variables [geometry.bisector A D A (angle A)]
variables [geometry.touches_circle incircle (triangle A B C) (angle bisector A D) BC E]
variables [geometry.on_circumcircle A1 (triangle A B C)]
variables [geometry.parallel AA1 BC]
variables [geometry.intersection T (circumcircle A E D) (line EA1) ≠ E]

theorem proof_it_eq_ia : geometry.equivalent_distance I T I A := by
  sorry

end proof_it_eq_ia_l380_380581


namespace sum_b_ge_sum_a_sum_b_eq_sum_a_iff_l380_380436

theorem sum_b_ge_sum_a {n : ℕ} (a b : Fin n → ℝ) 
(h1 : ∀ i : Fin (n-1), a i ≥ a (i+1)) 
(h2 : ∀ i : Fin n, a i > 0) 
(h3 : b 0 ≥ a 0) 
(h4 : ∀ i : Fin (n-1), (finset.range ((i:ℕ)+2)).prod b ≥ (finset.range ((i:ℕ)+2)).prod a) 
  : (finset.univ.sum b) ≥ (finset.univ.sum a) :=
sorry 

theorem sum_b_eq_sum_a_iff {n : ℕ} (a b : Fin n → ℝ) 
(h1 : ∀ i : Fin (n-1), a i ≥ a (i+1)) 
(h2 : ∀ i : Fin n, a i > 0) 
(h3 : b 0 ≥ a 0) 
(h4 : ∀ i : Fin (n-1), (finset.range ((i:ℕ)+2)).prod b ≥ (finset.range ((i:ℕ)+2)).prod a) 
  : (finset.univ.sum b) = (finset.univ.sum a) ↔ ∀ i : Fin n, a i = b i :=
sorry

end sum_b_ge_sum_a_sum_b_eq_sum_a_iff_l380_380436


namespace debby_soda_bottles_l380_380515

noncomputable def total_bottles (d t : ℕ) : ℕ := d * t

theorem debby_soda_bottles :
  ∀ (d t: ℕ), d = 9 → t = 40 → total_bottles d t = 360 :=
by
  intros d t h1 h2
  sorry

end debby_soda_bottles_l380_380515


namespace correct_product_l380_380642

def reverse_digits (n: ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d2 * 10 + d1

theorem correct_product (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : b > 0) (h3 : reverse_digits a * b = 221) :
  a * b = 527 ∨ a * b = 923 :=
sorry

end correct_product_l380_380642


namespace product_of_a_values_l380_380737

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem product_of_a_values :
  (distance (3 * a, 2 * a - 5) (5, 0) = 5 * real.sqrt 5) →
  (∏ x in {x : ℝ | (distance (3 * a, 2 * a - 5) (5, 0) = 5 * real.sqrt 5)}, x) = -749 / 676 :=
by
  sorry

end product_of_a_values_l380_380737


namespace remaining_balance_is_235_l380_380416

/--  Define initial amount and item costs --/
def initial_amount : ℕ := 500
def cost_of_rice_packet : ℕ := 20
def number_of_rice_packets : ℕ := 2
def cost_of_wheat_packet : ℕ := 25
def number_of_wheat_packets : ℕ := 3
def cost_of_soda : ℕ := 150

/-- Calculate the total spending --/
def total_spent : ℕ :=
  (number_of_rice_packets * cost_of_rice_packet) +
  (number_of_wheat_packets * cost_of_wheat_packet) +
  cost_of_soda

/-- Calculate the remaining balance --/
def remaining_balance : ℕ :=
  initial_amount - total_spent

/-- The theorem stating that the remaining balance equals 235 --/
theorem remaining_balance_is_235 : remaining_balance = 235 :=
  by simp [remaining_balance, initial_amount, total_spent, number_of_rice_packets, cost_of_rice_packet, number_of_wheat_packets, cost_of_wheat_packet, cost_of_soda]
  sorry

end remaining_balance_is_235_l380_380416


namespace sequence_comparison_l380_380424

theorem sequence_comparison :
  let seq1 := (List.range' 1 100).map (λ n, if n % 2 = 1 then n else -n).sum
  let seq2 := (List.range' 1 101).enumerate.map (λ ⟨i, n⟩, if i % 2 = 0 then n else -n).reverse.sum
  seq1 = -50 ∧ seq2 = 52 ∧ seq2 > seq1 := 
by {
  let seq1 := (List.range' 1 100).map (λ n, if n % 2 = 1 then n else -n).sum,
  let seq2 := (List.range' 1 101).enumerate.map (λ ⟨i, n⟩, if i % 2 = 0 then n else -n).reverse.sum,
  have h1 : seq1 = -50 := sorry,
  have h2 : seq2 = 52 := sorry,
  exact ⟨h1, h2, by linarith⟩
}

end sequence_comparison_l380_380424


namespace integer_solution_l380_380770

theorem integer_solution (x : ℤ) (h : x^2 < 3 * x) : x = 1 ∨ x = 2 :=
sorry

end integer_solution_l380_380770


namespace grid_coloring_exists_l380_380156

variable (f : Fin 50 → Fin 50 → ℝ)

axiom cell_cond_1 (i j : Fin 50) :
  (3 : ℝ) * f i j = (if i > 0 then f (i-1) j else 0) + (if i < 49 then f (i+1) j else 0) + 
                     (if j > 0 then f i (j-1) else 0) + (if j < 49 then f i (j+1) else 0)

axiom cell_cond_2 (i j : Fin 50) :
  (2 : ℝ) * f i j = (if i > 0 ∧ j > 0 then f (i-1) (j-1) else 0) + 
                     (if i > 0 ∧ j < 49 then f (i-1) (j+1) else 0) + 
                     (if i < 49 ∧ j > 0 then f (i+1) (j-1) else 0) + 
                     (if i < 49 ∧ j < 49 then f (i+1) (j+1) else 0)

theorem grid_coloring_exists :
  ∃ (color : Fin 50 → Fin 50 → Bool),
    (let red_sum := ∑ i j, if color i j then f i j else 0,
         blue_sum := ∑ i j, if ¬ color i j then f i j else 0
     in red_sum = blue_sum) :=
sorry

end grid_coloring_exists_l380_380156


namespace determine_f_l380_380309

noncomputable def f : ℝ → ℝ := sorry

theorem determine_f : (f(0) = 1) 
  ∧ (∀ x y : ℝ, f(x * y + 1) = f(x) * f(y) - f(y) - x + 2)
  → ∀ x : ℝ, f(x) = x + 1 :=
by
  intro h
  sorry

end determine_f_l380_380309


namespace range_of_f_less_than_zero_l380_380265

noncomputable def f (x : ℝ) : ℝ := sorry -- placeholder for the even function

theorem range_of_f_less_than_zero
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_decreasing : ∀ x y : ℝ, x < y → y < 0 → f y ≤ f x)
  (h_f2_zero : f 2 = 0) :
  { x : ℝ | f x < 0 } = set.Ioo (-2) 2 :=
sorry

end range_of_f_less_than_zero_l380_380265


namespace range_f_when_b_eq_1_find_b_range_l380_380193

-- Define function f for any b
def f (b : ℝ) (x : ℝ) : ℝ := x + b / x - 3

-- Prove the range of f(x) when b = 1 is [-1, -1/2]
theorem range_f_when_b_eq_1 : 
  (set.Icc (-1 : ℝ) (-1/2) = 
  {y | ∃ x ∈ set.Icc 1 2, f 1 x = y}) :=
by
  sorry


-- If b ≥ 2, given M - m ≥ 4, prove b ∈ [10, ∞)
theorem find_b_range (b : ℝ) (hb : b ≥ 2) (hM : ∀ x ∈ set.Icc 1 2, ∃ m M, m = (⨅ x, f b x) ∧ M = (⨆ x, f b x) ∧ M - m ≥ 4) :
  b ∈ set.Ici 10 :=
by
  sorry

end range_f_when_b_eq_1_find_b_range_l380_380193


namespace nat_exponent_sum_eq_l380_380862

theorem nat_exponent_sum_eq (n p q : ℕ) : n^p + n^q = n^2010 ↔ (n = 2 ∧ p = 2009 ∧ q = 2009) :=
by
  sorry

end nat_exponent_sum_eq_l380_380862


namespace set_intersection_complement_l380_380242

/-- Definition of the universal set U. -/
def U := ({1, 2, 3, 4, 5} : Set ℕ)

/-- Definition of the set M. -/
def M := ({3, 4, 5} : Set ℕ)

/-- Definition of the set N. -/
def N := ({2, 3} : Set ℕ)

/-- Statement of the problem to be proven. -/
theorem set_intersection_complement :
  ((U \ N) ∩ M) = ({4, 5} : Set ℕ) :=
by
  sorry

end set_intersection_complement_l380_380242


namespace mean_siblings_l380_380488
open Real

/-- The set of numbers representing the number of siblings -/
def siblings : List ℝ := [1, 6, 10, 4, 3, 3, 11, 3, 10]

/-- The mean of the list of numbers -/
noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

/-- The expected mean value -/
def expected_mean : ℝ := 5.67

/-- Proof that the mean of the siblings list is approximately 5.67 -/
theorem mean_siblings : abs (mean siblings - expected_mean) < 0.01 := by
  sorry

end mean_siblings_l380_380488


namespace cube_edges_count_l380_380248

theorem cube_edges_count : (∀ (A : Type), A ≈ cube -> 12 edges = true :=
by
  sorry

end cube_edges_count_l380_380248


namespace ratio_of_triangle_side_to_rectangle_width_l380_380124

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l380_380124


namespace chameleon_problem_l380_380342

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380342


namespace translation_coordinates_l380_380365

theorem translation_coordinates (x y : ℤ) (a b : ℤ) 
                                (init_point : x = 2 ∧ y = 3) 
                                (translate_left : a = -3) 
                                (translate_up : b = 2) 
                                (new_coords : (x + a, y + b) = (-1, 5)) : 
  (2 + -3 = -1) ∧ (3 + 2 = 5) :=
by {
  intros,
  rw [init_point.1, init_point.2],
  rw [translate_left, translate_up],
  exact new_coords, 
  sorry
}

end translation_coordinates_l380_380365


namespace johns_weekly_earnings_increase_l380_380297

-- Define original and new weekly earnings
def original_earnings : ℝ := 50
def new_earnings : ℝ := 80

-- Define the function to calculate percentage increase
def percentage_increase (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

-- Theorem stating the percentage increase in earnings
theorem johns_weekly_earnings_increase : percentage_increase original_earnings new_earnings = 60 :=
by 
    sorry

end johns_weekly_earnings_increase_l380_380297


namespace circumscribed_circle_diameter_l380_380152

def diameter_of_circumscribed_circle (a : ℝ) (angle_A : ℝ) : ℝ :=
  a / Real.sin angle_A

theorem circumscribed_circle_diameter
  (a : ℝ) (angle_A : ℝ)
  (h1 : a = 15)
  (h2 : angle_A = Real.pi / 4) :
  diameter_of_circumscribed_circle a angle_A = 15 * Real.sqrt 2 :=
by
  sorry

end circumscribed_circle_diameter_l380_380152


namespace primes_and_one_l380_380576

-- Given conditions:
variables {a n : ℕ}
variable (ha : a > 100 ∧ a % 2 = 1)  -- a is an odd natural number greater than 100
variable (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime (a - n^2) / 4)  -- for all n ≤ √(a / 5), (a - n^2) / 4 is prime

-- Theorem: For all n > √(a / 5), (a - n^2) / 4 is either prime or 1
theorem primes_and_one {a : ℕ} (ha : a > 100 ∧ a % 2 = 1)
  (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime ((a - n^2) / 4)) :
  ∀ n > Nat.sqrt (a / 5), Prime ((a - n^2) / 4) ∨ ((a - n^2) / 4) = 1 :=
sorry

end primes_and_one_l380_380576


namespace number_increases_by_one_or_prime_l380_380290

theorem number_increases_by_one_or_prime :
  ∀ n : ℕ, ∀ k : ℕ, (k ≥ 6) ∧ (k = 6 → ∀ m, (m > 0 → k + gcd k m = k + m)) ∧
  (∀ k' : ℕ, k' = k + gcd k n → k' = k + 1 ∨ (∃ p : ℕ, nat.prime p ∧ k' = k + p)) :=
by
  sorry

end number_increases_by_one_or_prime_l380_380290


namespace digit_150th_l380_380051

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380051


namespace value_of_k_l380_380629

theorem value_of_k (k : ℤ) : 
  (∀ x : ℤ, (x + k) * (x - 4) = x^2 - 4 * x + k * x - 4 * k ∧ 
  (k - 4) * x = 0) → k = 4 := 
by 
  sorry

end value_of_k_l380_380629


namespace rachel_math_homework_l380_380373

def rachel_homework (M : ℕ) (reading : ℕ) (biology : ℕ) (total : ℕ) : Prop :=
reading = 3 ∧ biology = 10 ∧ total = 15 ∧ reading + biology + M = total

theorem rachel_math_homework: ∃ M : ℕ, rachel_homework M 3 10 15 ∧ M = 2 := 
by 
  sorry

end rachel_math_homework_l380_380373


namespace cheryl_distance_walked_l380_380505

theorem cheryl_distance_walked :
  let s1 := 2  -- speed during the first segment in miles per hour
  let t1 := 3  -- time during the first segment in hours
  let s2 := 4  -- speed during the second segment in miles per hour
  let t2 := 2  -- time during the second segment in hours
  let s3 := 1  -- speed during the third segment in miles per hour
  let t3 := 3  -- time during the third segment in hours
  let s4 := 3  -- speed during the fourth segment in miles per hour
  let t4 := 5  -- time during the fourth segment in hours
  let d1 := s1 * t1  -- distance for the first segment
  let d2 := s2 * t2  -- distance for the second segment
  let d3 := s3 * t3  -- distance for the third segment
  let d4 := s4 * t4  -- distance for the fourth segment
  d1 + d2 + d3 + d4 = 32 :=
by
  sorry

end cheryl_distance_walked_l380_380505


namespace product_of_all_possible_K_l380_380870

theorem product_of_all_possible_K (x y : ℝ) (h₁ : log 2 (2 * x + y) = log 4 (x^2 + x * y + 7 * y^2)) :
  let K₁ := -6;
      K₂ := -3 in
  K₁ * K₂ = 18 := sorry

end product_of_all_possible_K_l380_380870


namespace digit_150_of_17_div_70_is_2_l380_380015

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380015


namespace product_of_units_digits_of_sophie_germain_primes_l380_380430

noncomputable def is_sophie_germain_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (2 * p + 1)

theorem product_of_units_digits_of_sophie_germain_primes :
  let units_digits := {d | ∃ p, p > 6 ∧ is_sophie_germain_prime p ∧ (p % 10 = d)} in
  ∏ d in units_digits, d = 3 :=
by
  sorry

end product_of_units_digits_of_sophie_germain_primes_l380_380430


namespace Jenna_total_cost_l380_380993

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end Jenna_total_cost_l380_380993


namespace part1_part2_part3_l380_380221

def f (x : Real) : Real := 2 * cos x * (sin x + cos x)

theorem part1 : f (5 * Real.pi / 4) = 2 :=
by {
  -- Here we will prove that f(5π/4) = 2
  sorry
}

theorem part2 : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
by {
  -- Here we will prove that the smallest positive period is π
  sorry
}

theorem part3 : ∀ k : Int, ∀ x : Real, k * Real.pi - (3 * Real.pi / 8) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 8) →
  (∃ k : Int, k * Real.pi - (3 * Real.pi / 8) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 8)) :=
by {
  -- Here we will prove the monotonically increasing interval
  sorry
}

end part1_part2_part3_l380_380221


namespace minimum_omega_l380_380880

def f (x ω : ℝ) : ℝ := (sqrt 3) * (cos (ω * x))^2 + (sin (ω * x)) * (cos (ω * x))
noncomputable def min_omega : ℝ := 1 / 4044

theorem minimum_omega (ω : ℝ) (hω : ω > 0) (h : ∃ x0, ∀ x, f x0 ω ≤ f x ω ∧ f x ω ≤ f (x0 + 2022 * Real.pi) ω) :
  ω = min_omega :=
sorry

end minimum_omega_l380_380880


namespace baker_sold_cakes_l380_380495

def cakes_sold (initial_cakes bought_cakes sold_more_than bought_sold) : ℕ :=
sold_more_than + bought_cakes

theorem baker_sold_cakes :
  let initial_cakes := 170 
  let bought_cakes := 31 
  let sold_more_than := 47 
  cakes_sold initial_cakes bought_cakes sold_more_than = 78 := 
by
  intros
  dsimp [cakes_sold]
  rfl

end baker_sold_cakes_l380_380495


namespace population_increase_time_l380_380406

theorem population_increase_time (persons_added : ℕ) (time_minutes : ℕ) (seconds_per_minute : ℕ) (total_seconds : ℕ) (time_for_one_person : ℕ) :
  persons_added = 160 →
  time_minutes = 40 →
  seconds_per_minute = 60 →
  total_seconds = time_minutes * seconds_per_minute →
  time_for_one_person = total_seconds / persons_added →
  time_for_one_person = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end population_increase_time_l380_380406


namespace quadratic_root_zero_l380_380555

theorem quadratic_root_zero (k : ℝ) :
    (∃ x : ℝ, x = 0 ∧ (k - 1) * x ^ 2 + 6 * x + k ^ 2 - k = 0) → k = 0 :=
by
  sorry

end quadratic_root_zero_l380_380555


namespace three_digit_integers_with_at_least_two_identical_digits_l380_380953

/-- Prove that the number of positive three-digit integers less than 700 that have at least two identical digits is 162. -/
theorem three_digit_integers_with_at_least_two_identical_digits : 
  ∃ n : ℕ, (n = 162) ∧ (count_three_digit_integers_with_at_least_two_identical_digits n) :=
by
  sorry

/-- Define a function to count the number of three-digit integers less than 700 with at least two identical digits -/
noncomputable def count_three_digit_integers_with_at_least_two_identical_digits (n : ℕ) : Prop :=
  n = 162

end three_digit_integers_with_at_least_two_identical_digits_l380_380953


namespace polynomial_roots_parallelogram_l380_380151

theorem polynomial_roots_parallelogram (b : ℝ) :
  (∃ (z : ℂ → ℂ), 
    (z^4 - 8*z^3 + 15*b*z^2 - 5*(3*b^2 + 4*b - 4)*z + 9 = 0) ∧
    ∃ w1 w2 : ℂ, z = w1 + 2 ∧ z = -w1 + 2 ∧ z = w2 + 2 ∧ z = -w2 + 2 ) ↔ 
  (b = 2/3 ∨ b = -2) :=
sorry

end polynomial_roots_parallelogram_l380_380151


namespace fraction_of_remaining_birds_left_l380_380753

theorem fraction_of_remaining_birds_left (B : ℕ) (F : ℚ) (hB : B = 60)
  (H : (1/3) * (2/3 : ℚ) * B * (1 - F) = 8) :
  F = 4/5 := 
sorry

end fraction_of_remaining_birds_left_l380_380753


namespace angle_and_ratio_between_lines_l380_380739

noncomputable def common_midpoint {A B C A1 B1 C1 : Type} [AffineSpace A B C] [AffineSpace A1 B1 C1] (M : A) : Prop :=
  midpoint B C M ∧ midpoint B1 C1 M

noncomputable def equilateral_triangle {A B C : Type} [AffineSpace A B C] : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

noncomputable def clockwise_order {A B C A1 B1 C1 : Type} : Prop :=
  -- This definition is often specific to the geometric interpretation and might need more context
  ∀ P₀ P₁ P₂ Q₀ Q₁ Q₂, polygon_orientation P₀ P₁ P₂ = .clockwise 
  ∧ polygon_orientation Q₀ Q₁ Q₂ = .clockwise

theorem angle_and_ratio_between_lines 
  {A B C A1 B1 C1 : Type} 
  [AffineSpace A B C] 
  [AffineSpace A1 B1 C1]
  (M : A)
  (h1 : common_midpoint M)
  (h2 : equilateral_triangle A B C)
  (h3 : equilateral_triangle A1 B1 C1)
  (h4 : clockwise_order A B C A1 B1 C1) :
  ∠(A, A1) (B, B1) = 90 ° ∧ dist A A1 / dist B B1 = √3 :=
  sorry

end angle_and_ratio_between_lines_l380_380739


namespace partition_two_houses_l380_380975

-- Declare types and assumptions
variable {V : Type*} [Fintype V]

-- Assume a graph G representing the enemy relationships
variable (G : SimpleGraph V)

-- Each member has at most 3 enemies (degree constraint)
variable (h_max_enemies : ∀ v : V, G.degree v ≤ 3)

-- Statement of the proof problem
theorem partition_two_houses (G : SimpleGraph V) (h : ∀ v : V, G.degree v ≤ 3) : 
  ∃ (A B : Finset V), A ∩ B = ∅ ∧ A ∪ B = Finset.univ ∧ 
  (∀ v ∈ A, (Finset.filter (λ w, G.adj v w) A).card ≤ 1) ∧ 
  (∀ v ∈ B, (Finset.filter (λ w, G.adj v w) B).card ≤ 1) :=
by
  sorry

end partition_two_houses_l380_380975


namespace area_of_square_l380_380785

theorem area_of_square (a : ℕ) (h : a = 20) : a * a = 400 :=
by
  -- Given
  have ha : a = 20 := h,
  -- to prove a * a = 400
  sorry

end area_of_square_l380_380785


namespace digit_150th_of_17_div_70_is_7_l380_380026

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380026


namespace Saheed_earnings_l380_380377

theorem Saheed_earnings (Vika_earnings : ℕ) (Kayla_earnings : ℕ) (Saheed_earnings : ℕ)
  (h1 : Vika_earnings = 84) (h2 : Kayla_earnings = Vika_earnings - 30) (h3 : Saheed_earnings = 4 * Kayla_earnings) :
  Saheed_earnings = 216 := 
by
  sorry

end Saheed_earnings_l380_380377


namespace salary_increment_l380_380814

-- Definitions based on conditions in the problem
def originalSalary : ℝ := 48000
def monthlySalary : ℝ := originalSalary / 12
def reducedSalary : ℝ := monthlySalary * 0.9
def incrementPercent : ℝ := (1 / 9) * 100

-- Statement
theorem salary_increment (originalSalary : ℝ) (reducedSalary : ℝ) :
    let monthlySalary := originalSalary / 12 in
    let reducedSalary := monthlySalary * 0.9 in
    let incrementPercent := (1 / 9) * 100 in
    incrementPercent ≈ 11.11 ∧
    let sixMonthReduced := 6 * reducedSalary in
    let sixMonthOriginal := 6 * monthlySalary in
    let yearlyReduced := sixMonthReduced + sixMonthOriginal in
    originalSalary - yearlyReduced = 2400 :=
begin
    sorry
end

end salary_increment_l380_380814


namespace hydrogen_atoms_count_l380_380453

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Given conditions
def total_molecular_weight : ℝ := 88
def number_of_C_atoms : ℕ := 4
def number_of_O_atoms : ℕ := 2

theorem hydrogen_atoms_count (nh : ℕ) 
  (h_molecular_weight : total_molecular_weight = 88) 
  (h_C_atoms : number_of_C_atoms = 4) 
  (h_O_atoms : number_of_O_atoms = 2) :
  nh = 8 :=
by
  -- skipping proof
  sorry

end hydrogen_atoms_count_l380_380453


namespace prob1_prob2_l380_380926

noncomputable def hyperbola_equation (x y λ : ℝ) : Prop :=
  x^2 / (1 - λ) - y^2 / λ = 1

noncomputable def λ_values: set ℝ := {λ | 0 < λ ∧ λ < 1}

theorem prob1 (λ : ℝ) (hk : λ ∈ λ_values):
  (hyperbola_equation 1 1 λ)
  ∧ (abs (1 - (-1)) = 2)
  → λ = (Real.sqrt 5 - 1) / 2 := by
  sorry

theorem prob2 (x y : ℝ):
  ∀ (λ : ℝ) (ho : λ = 3/4)
  (hk : hyperbola_equation x y λ)
  (line_eq : y = Real.sqrt 3 * (x - 1)),
  Real.sqrt (x^2 + (y)^2) = Real.sqrt 13 / 4 := by
  sorry

end prob1_prob2_l380_380926


namespace Jenna_total_cost_l380_380992

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end Jenna_total_cost_l380_380992


namespace total_votes_election_l380_380647

theorem total_votes_election 
  (votes_A : ℝ) 
  (valid_votes_percentage : ℝ) 
  (invalid_votes_percentage : ℝ)
  (votes_candidate_A : ℝ) 
  (total_votes : ℝ) 
  (h1 : votes_A = 0.60) 
  (h2 : invalid_votes_percentage = 0.15) 
  (h3 : votes_candidate_A = 285600) 
  (h4 : valid_votes_percentage = 0.85) 
  (h5 : total_votes = 560000) 
  : 
  ((votes_A * valid_votes_percentage * total_votes) = votes_candidate_A) 
  := 
  by sorry

end total_votes_election_l380_380647


namespace green_chameleon_increase_l380_380334

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380334


namespace pizza_slices_leftover_l380_380383

theorem pizza_slices_leftover (initial_pizzas : ℕ) (slices_per_pizza : ℕ) (stephen_fraction : ℚ) (pete_fraction : ℚ) : 
  initial_pizzas = 2 → slices_per_pizza = 12 → stephen_fraction = 0.25 → pete_fraction = 0.5 →
  let total_slices := initial_pizzas * slices_per_pizza in
  let stephen_ate := (stephen_fraction * total_slices).natAbs in
  let slices_after_stephen := total_slices - stephen_ate in
  let pete_ate := (pete_fraction * (slices_after_stephen : ℚ)).natAbs in
  let slices_left := slices_after_stephen - pete_ate in
  slices_left = 9 := 
by
  intros h1 h2 h3 h4
  have total_slices := initial_pizzas * slices_per_pizza
  have stephen_ate := (stephen_fraction * total_slices).natAbs
  have slices_after_stephen := total_slices - stephen_ate
  have pete_ate := (pete_fraction * (slices_after_stephen : ℚ)).natAbs
  have slices_left := slices_after_stephen - pete_ate
  exact sorry

end pizza_slices_leftover_l380_380383


namespace seq_sum_mod_1500_l380_380097

open Nat

def a : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| (n+3) := a (n+2) + a (n+1) + a n

noncomputable def sum_a_to (n : ℕ) : ℕ :=
(0 to n).sum a

theorem seq_sum_mod_1500 : 
  a 15 = 3136 → 
  a 16 = 5768 → 
  a 17 = 10609 → 
  (sum_a_to 14) % 1500 = 646 :=
by
  intros a15 a16 a17
  sorry

end seq_sum_mod_1500_l380_380097


namespace increase_in_green_chameleons_is_11_l380_380323

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380323


namespace find_radius_of_circle_l380_380659

variable (AB BC AC R : ℝ)

-- Conditions
def is_right_triangle (ABC : Type) (AB BC : ℝ) (AC : outParam ℝ) : Prop :=
  AC = Real.sqrt (AB^2 + BC^2)

def is_tangent (O : Type) (AB BC AC R : ℝ) : Prop :=
  ∃ (P Q : ℝ), P = R ∧ Q = R ∧ P < AC ∧ Q < AC

theorem find_radius_of_circle (h1 : is_right_triangle ABC 21 28 AC) (h2 : is_tangent O 21 28 AC R) : R = 12 :=
sorry

end find_radius_of_circle_l380_380659


namespace chameleon_increase_l380_380353

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380353


namespace group_interval_eq_l380_380977

noncomputable def group_interval (a b m h : ℝ) : ℝ := abs (a - b)

theorem group_interval_eq (a b m h : ℝ) 
  (h1 : h = m / abs (a - b)) :
  abs (a - b) = m / h := 
by 
  sorry

end group_interval_eq_l380_380977


namespace solve_equation_l380_380711

theorem solve_equation (x : ℝ) (h : (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1) : x = -1/2 :=
sorry

end solve_equation_l380_380711


namespace min_n_for_positive_sum_l380_380654

theorem min_n_for_positive_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : ∀ n, S n = n * (a 0 + a (n - 1)) / 2)
  (h_cond : (a 8 : ℚ) ≠ 0 ∧ ((a 9 : ℚ) / (a 8 : ℚ) < -1))
  (h_min_val : ∃ n, ∀ m, S m ≥ S n) :
  ∃ n, S n > 0 ∧ ∀ m, m < n → S m ≤ 0 :=
sorry

end min_n_for_positive_sum_l380_380654


namespace eccentricity_of_ellipse_l380_380393

/-- Define the parameters a and b for the ellipse -/
def a : ℝ := 3
def b : ℝ := 2

/-- Define the semi-major axis and semi-minor axis lengths -/
def c : ℝ := real.sqrt (a^2 - b^2)

/-- Define the ellipse equation -/
def ellipse_eq (x y : ℝ) : Prop := (x^2 / (a^2)) + (y^2 / (b^2)) = 1

/-- Theorem stating the eccentricity for the given ellipse equation -/
theorem eccentricity_of_ellipse : 
  ∀ x y, ellipse_eq x y → (c / a = real.sqrt 5 / 3) :=
by 
  intros x y h
  sorry

end eccentricity_of_ellipse_l380_380393


namespace sufficient_but_not_necessary_condition_l380_380854

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a - Real.sin x

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, f' a x > 0 → (a > 1)) ∧ (¬∀ x, f' a x ≥ 0 → (a > 1)) := sorry

end sufficient_but_not_necessary_condition_l380_380854


namespace chameleon_problem_l380_380356

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380356


namespace area_of_quadrilateral_EFCD_l380_380286

noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  h * (b1 + b2) / 2

theorem area_of_quadrilateral_EFCD :
  ∀ (AB CD AD BC : ℝ),
    AB = 10 →
    CD = 26 →
    AD = 15 →
    let E := AD / 3 in
    let F := BC / 3 in
    trapezoid_area ((1/3) * AB + (2/3) * CD) CD (2/3 * AD) = 700 / 3 :=
by {
  intros AB CD AD BC h1 h2 h3 E F,
  simp only [E, F],
  sorry
}

end area_of_quadrilateral_EFCD_l380_380286


namespace x_intersection_difference_l380_380180

-- Define the conditions
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem x_intersection_difference :
  let x₁ := (1 + Real.sqrt 6) / 5
  let x₂ := (1 - Real.sqrt 6) / 5
  (parabola1 x₁ = parabola2 x₁) → (parabola1 x₂ = parabola2 x₂) →
  (x₁ - x₂) = (2 * Real.sqrt 6) / 5 := 
by
  sorry

end x_intersection_difference_l380_380180


namespace jane_spent_75_days_reading_l380_380668

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end jane_spent_75_days_reading_l380_380668


namespace incorrect_line_pass_through_Q_l380_380486

theorem incorrect_line_pass_through_Q (a b : ℝ) : 
  (∀ (k : ℝ), ∃ (Q : ℝ × ℝ), Q = (0, b) ∧ y = k * x + b) →
  (¬ ∃ k : ℝ, ∀ y x, y = k * x + b ∧ x = 0)
:= 
sorry

end incorrect_line_pass_through_Q_l380_380486


namespace total_distance_traveled_l380_380834

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end total_distance_traveled_l380_380834


namespace mass_and_center_of_mass_l380_380546

-- Define the parametric equations for x and y
def x (t : ℝ) : ℝ := 10 * (Real.cos t) ^ 3
def y (t : ℝ) : ℝ := 10 * (Real.sin t) ^ 3

-- Define the density
def density : ℝ := 1

-- Define the derivatives
def x' (t : ℝ) : ℝ := -30 * (Real.cos t) ^ 2 * Real.sin t
def y' (t : ℝ) : ℝ := 30 * (Real.sin t) ^ 2 * Real.cos t

-- Define the differential element of arc length
def dl (t : ℝ) : ℝ := 30 * (Real.sin t) * (Real.cos t)

-- Define the mass integral
def mass : ℝ := ∫ t in 0..Real.pi/2, density * dl t

-- Define the static moments integrals
def M_x : ℝ := ∫ t in 0..Real.pi/2, density * x t * dl t
def M_y : ℝ := ∫ t in 0..Real.pi/2, density * y t * dl t

-- Define the coordinates of the center of mass
def x_c : ℝ := M_y / mass
def y_c : ℝ := M_x / mass

-- Prove the mass and the center of mass coordinates
theorem mass_and_center_of_mass :
  mass = 15 ∧ (x_c, y_c) = (4, 4) := by
  sorry

end mass_and_center_of_mass_l380_380546


namespace tan_angle_PAB_l380_380640

-- Define the sides of triangle ABC
def AB : ℝ := 13
def BC : ℝ := 14
def CA : ℝ := 15

-- Define the angle variable x
def x : ℝ := sorry  -- we define x but without computing it here

-- Define a point P such that the given angle conditions hold
def P_in_triangle (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  sorry -- formal definition of P being a point inside the triangle and the angle conditions

-- Theorem stating the final proof problem
theorem tan_angle_PAB (P : ℝ × ℝ) (A B C : ℝ × ℝ) (hP : P_in_triangle P A B C) :
  tan x = 168 / 295 :=
sorry

end tan_angle_PAB_l380_380640


namespace bobs_fruit_drink_cost_l380_380125

theorem bobs_fruit_drink_cost
  (cost_soda : ℕ)
  (cost_hamburger : ℕ)
  (cost_sandwiches : ℕ)
  (bob_total_spent same_amount : ℕ)
  (andy_spent_eq : same_amount = cost_soda + 2 * cost_hamburger)
  (andy_bob_spent_eq : same_amount = bob_total_spent)
  (bob_sandwich_cost_eq : cost_sandwiches = 3)
  (andy_spent_eq_total : cost_soda = 1)
  (andy_burger_cost : cost_hamburger = 2)
  : bob_total_spent - cost_sandwiches = 2 :=
by
  sorry

end bobs_fruit_drink_cost_l380_380125


namespace green_chameleon_increase_l380_380337

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380337


namespace employee_salary_l380_380760

theorem employee_salary (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 528) : Y = 240 :=
by
  sorry

end employee_salary_l380_380760


namespace digit_150_in_17_div_70_l380_380004

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380004


namespace ratio_of_second_to_third_l380_380757

theorem ratio_of_second_to_third (A B C : ℕ) (h1 : A + B + C = 98) (h2 : A * 3 = B * 2) (h3 : B = 30) :
  B * 8 = C * 5 :=
by
  sorry

end ratio_of_second_to_third_l380_380757


namespace decimal_to_fraction_equiv_l380_380065

theorem decimal_to_fraction_equiv : (0.38 : ℝ) = 19 / 50 :=
by
  sorry

end decimal_to_fraction_equiv_l380_380065


namespace find_eccentricity_l380_380400

noncomputable theory

-- Definitions
def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

def focus_right (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

def line_through_focus (c t : ℝ) := (c + t / 2, real.sqrt(3) * t / 2)

def midpoint (p1 p2 : ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Theorem statement
theorem find_eccentricity (a b : ℝ) :
  let c := focus_right a b,
      F := (c, 0),
      line_at_focus := λ t, line_through_focus c t in
  (∀ t₁ t₂, let A := line_at_focus t₁,
                 B := line_at_focus t₂,
                 M := midpoint A B in
             hyperbola A.1 A.2 a b ∧ hyperbola B.1 B.2 a b ∧ 
             |real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)| = c / 2) →
  let e := real.sqrt (1 + b^2 / a^2) in
  e = real.sqrt 2 :=
by
  intros a b c F line_at_focus hyp e 
  -- skipping the proof
  exact sorry

end find_eccentricity_l380_380400


namespace green_chameleon_increase_l380_380325

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380325


namespace count_pos_three_digit_ints_with_same_digits_l380_380947

-- Define a structure to encapsulate the conditions for a three-digit number less than 700 with at least two digits the same.
structure valid_int (n : ℕ) : Prop :=
  (three_digit : 100 ≤ n ∧ n < 700)
  (same_digits : ∃ d₁ d₂ d₃ : ℕ, ((100 * d₁ + 10 * d₂ + d₃ = n) ∧ (d₁ = d₂ ∨ d₂ = d₃ ∨ d₁ = d₃)))

-- The number of integers satisfying the conditions
def count_valid_ints : ℕ :=
  168

-- The theorem to prove
theorem count_pos_three_digit_ints_with_same_digits : 
  (∃ n, valid_int n) → 168 :=
by
  -- Since the proof is not required, we add sorry here.
  sorry

end count_pos_three_digit_ints_with_same_digits_l380_380947


namespace digit_150th_of_17_div_70_is_7_l380_380019

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380019


namespace second_player_win_strategy_l380_380695

theorem second_player_win_strategy:
  ∃ strategy : (ℕ → ℕ) → ℕ, 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
    (strategy n + n = 1001) ∧
    (strategy n - n) % 13 = 0) :=
sorry

end second_player_win_strategy_l380_380695


namespace project_assignment_l380_380844

open Nat

def binom (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem project_assignment :
  let A := 3
  let B := 1
  let C := 2
  let D := 2
  let total_projects := 8
  A + B + C + D = total_projects →
  (binom 8 3) * (binom 5 1) * (binom 4 2) * (binom 2 2) = 1680 :=
by
  intros
  sorry

end project_assignment_l380_380844


namespace statement_A_statement_B_statement_D_l380_380886

noncomputable def f : ℝ → ℝ := sorry -- Definition of function f is given, assumption made by 'sorry'

-- Conditions
axiom f_deriv_cos_gt_f_sin {x : ℝ} (hx : 0 < x ∧ x < π / 2) : derivative f x * cos x > f x * sin x

-- Proof of Statements
theorem statement_A : f (π / 3) > sqrt 2 * f (π / 4) :=
by {
  -- Proof goes here, omitted using sorry
  sorry
}

theorem statement_B : 2 * f (π / 4) > sqrt 6 * f (π / 6) :=
by {
  -- Proof goes here, omitted using sorry
  sorry
}

theorem statement_D : f (π / 3) > 2 * cos 1 * f 1 :=
by {
  -- Proof goes here, omitted using sorry
  sorry
}

end statement_A_statement_B_statement_D_l380_380886


namespace ranges_and_variances_same_l380_380885
open Complex

theorem ranges_and_variances_same {x1 x2 x3 x4 x5 : ℝ}
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h5 : x5 > 0) :
  let 
    z := λ x : ℝ, 2 + (Real.sqrt x) * Complex.i
    y := λ x : ℝ, (z x) * (conj (z x))
    x_set := {x1, x2, x3, x4, x5}
    y_set := {y x1, y x2, y x3, y x4, y x5}
  in
    Set.range ((λ x, 4 + x) <$> x_set) = Set.range (fun x => x) x_set ∧
    ∑ x in ((λ x, 4 + x) '' x_set), (x - (∑ y in ((λ x, 4 + x) '' x_set), y) / 5) ^ 2 / 5 = 
    ∑ x in x_set, (x - (∑ y in x_set, y) / 5) ^ 2 / 5 :=
begin
  sorry
end

end ranges_and_variances_same_l380_380885


namespace emma_withdrew_amount_l380_380518

variable (W : ℝ) -- Variable representing the amount Emma withdrew

theorem emma_withdrew_amount:
  (230 - W + 2 * W = 290) →
  W = 60 :=
by
  sorry

end emma_withdrew_amount_l380_380518


namespace sum_seven_l380_380281

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

axiom a2 : a 2 = 3
axiom a6 : a 6 = 11
axiom arithmetic_seq : arithmetic_sequence a
axiom sum_of_sequence : ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sum_seven : S 7 = 49 :=
sorry

end sum_seven_l380_380281


namespace combined_cost_of_stocks_l380_380769

variable (face_value_A : ℝ) (discount_A : ℝ) (brokerage_A : ℝ)
variable (face_value_B : ℝ) (premium_B : ℝ) (brokerage_B : ℝ)
variable (face_value_C : ℝ) (discount_C : ℝ) (brokerage_C : ℝ)

def cost_after_discount (face_value : ℝ) (discount : ℝ) : ℝ :=
  face_value - (discount / 100 * face_value)

def cost_after_premium (face_value : ℝ) (premium : ℝ) : ℝ :=
  face_value + (premium / 100 * face_value)

def brokerage_cost (face_value_after_adjustment : ℝ) (brokerage : ℝ) : ℝ :=
  brokerage / 100 * face_value_after_adjustment

def total_cost (face_value : ℝ) (adjustment : ℝ) (brokerage : ℝ) (is_discount : Bool) : ℝ :=
  let adjusted_cost := if is_discount then cost_after_discount face_value adjustment
                       else cost_after_premium face_value adjustment
  adjusted_cost + brokerage_cost adjusted_cost brokerage

theorem combined_cost_of_stocks : 
  total_cost 100 2 0.2 true
  + total_cost 150 1.5 0.1667 false
  + total_cost 200 3 0.5 true
  = 445.67 := by
  sorry

end combined_cost_of_stocks_l380_380769


namespace increase_in_green_chameleons_is_11_l380_380322

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380322


namespace probability_x_equals_1_l380_380212

noncomputable def P (n p : ℚ) (k : ℕ) : ℚ :=
(nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem probability_x_equals_1 (n : ℕ) (p : ℚ) (x : ℕ → ℕ) 
  (h1 : x ~ binomial n p) (h2 : E(x) = 6) (h3 : D(x) = 3) :
  P n p 1 = 3 / 1024 :=
sorry

end probability_x_equals_1_l380_380212


namespace two_x_three_sin_relationship_l380_380957

open Real

theorem two_x_three_sin_relationship (x : ℝ) :
  0 < x ∧ x < π / 2 →
  (2 * x < 3 * sin x ∧ x < arccos (2 / 3)) ∨
  (2 * x = 3 * sin x ∧ ∃ θ, θ = arc ⟨2 / 3⟩ ∧ x = θ) ∨
  (2 * x > 3 * sin x ∧ ∃ θ, θ = arc ⟨2 / 3⟩ ∧ θ < x ∧ x < π / 2) :=
by
  intro h
  sorry

end two_x_three_sin_relationship_l380_380957


namespace find_a_of_parabola_l380_380732

theorem find_a_of_parabola (a b c : ℝ) 
(vertex : (2 : ℝ), 5 : ℝ) 
(point : (1 : ℝ), 2 : ℝ) 
(h_eqn : ∀ x y, y = a * x^2 + b * x + c) : 
a = -3 :=
by
  sorry

end find_a_of_parabola_l380_380732


namespace vector_magnitude_l380_380246

open Real

variables (e1 e2 : ℝ^3) -- Due to lack of context, assuming 3D vectors.
variable (θ : ℝ)

-- Conditions
def unit_vector_1 : Prop := ∥e1∥ = 1
def unit_vector_2 : Prop := ∥e2∥ = 1
def angle_between : Prop := θ = π/3
def dot_product : Prop := e1 • e2 = cos θ

theorem vector_magnitude :
  unit_vector_1 e1 →
  unit_vector_2 e2 →
  angle_between θ →
  dot_product e1 e2 θ →
  ∥e1 - (2 • e2)∥ = sqrt 3 :=
by
  intros u1 u2 ang dp
  sorry

end vector_magnitude_l380_380246


namespace trail_length_is_48_meters_l380_380378

noncomputable def length_of_trail (d: ℝ) : Prop :=
  let normal_speed := 8 -- normal speed in m/s
  let mud_speed := normal_speed / 4 -- speed in mud in m/s

  let time_mud := (1 / 3 * d) / mud_speed -- time through the mud in seconds
  let time_normal := (2 / 3 * d) / normal_speed -- time through the normal trail in seconds

  let total_time := 12 -- total time in seconds

  total_time = time_mud + time_normal

theorem trail_length_is_48_meters : ∃ d: ℝ, length_of_trail d ∧ d = 48 :=
sorry

end trail_length_is_48_meters_l380_380378


namespace tan_half_sum_l380_380588

-- Definitions and conditions
variables (a : ℝ) (α β : ℝ)
hypothesis h₁ : a > 1
hypothesis h₂ : α ∈ Ioo (-(π/2)) (π/2)
hypothesis h₃ : β ∈ Ioo (-(π/2)) (π/2)
hypothesis h₄ : tan(α) * tan(β) = 3 * a + 1
hypothesis h₅ : tan(α) + tan(β) = -4 * a

-- Statement of the theorem
theorem tan_half_sum : tan ((α + β) / 2) = -2 :=
sorry

end tan_half_sum_l380_380588


namespace pyramid_layers_l380_380449

theorem pyramid_layers (n : ℕ) (hn : ∑ i in range n, 3^i = 40) : n = 4 :=
sorry

end pyramid_layers_l380_380449


namespace acute_angle_iff_median_greater_l380_380369

noncomputable def median (a b c : ℝ) : ℝ :=
  sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4)

theorem acute_angle_iff_median_greater (a b c : ℝ) (A B C : ℝ) (m_a : ℝ) :
  (A < 90) ↔ m_a > a / 2 :=
by
  let A1 := (b + c) / 2
  have A1_eq : A1 = (b + c) / 2 := rfl
  have ma_eq : m_a = median a b c := by sorry
  exact sorry

end acute_angle_iff_median_greater_l380_380369


namespace chameleon_problem_l380_380346

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380346


namespace cos_double_angle_from_sin_shift_l380_380618

theorem cos_double_angle_from_sin_shift (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := 
by 
  sorry

end cos_double_angle_from_sin_shift_l380_380618


namespace hexagon_side_length_l380_380564

theorem hexagon_side_length 
  (rectangle_length rectangle_width : ℕ)
  (h_div : rectangle_length = 12)
  (h_width : rectangle_width = 12)
  (h_area : rectangle_length * rectangle_width = 144)
  (h_hex_to_square : ∃ s, 2 * (rectangle_length * rectangle_width / 2) = s^2) :
  let y := 12 in y = 12 :=
by
  sorry

end hexagon_side_length_l380_380564


namespace time_for_a_to_complete_one_round_l380_380427

theorem time_for_a_to_complete_one_round (T_a T_b : ℝ) 
  (h1 : 4 * T_a = 3 * T_b)
  (h2 : T_b = T_a + 10) : 
  T_a = 30 := by
  sorry

end time_for_a_to_complete_one_round_l380_380427


namespace increase_by_thirteen_possible_l380_380284

-- Define the main condition which states the reduction of the original product
def product_increase_by_thirteen (a : Fin 7 → ℕ) : Prop :=
  let P := (List.range 7).map (fun i => a ⟨i, sorry⟩) |>.prod
  let Q := (List.range 7).map (fun i => a ⟨i, sorry⟩ - 3) |>.prod
  Q = 13 * P

-- State the theorem to be proved
theorem increase_by_thirteen_possible : ∃ (a : Fin 7 → ℕ), product_increase_by_thirteen a :=
sorry

end increase_by_thirteen_possible_l380_380284


namespace james_beats_old_record_l380_380664

def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def two_point_conversions : ℕ := 6
def points_per_two_point_conversion : ℕ := 2
def field_goals : ℕ := 8
def points_per_field_goal : ℕ := 3
def extra_points : ℕ := 20
def points_per_extra_point : ℕ := 1
def old_record : ℕ := 300

theorem james_beats_old_record :
  touchdowns_per_game * points_per_touchdown * games_in_season +
  two_point_conversions * points_per_two_point_conversion +
  field_goals * points_per_field_goal +
  extra_points * points_per_extra_point - old_record = 116 := by
  sorry -- Proof is omitted.

end james_beats_old_record_l380_380664


namespace green_chameleon_increase_l380_380329

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380329


namespace complex_relationship_l380_380307

-- Define the conditions
variables {a b : ℝ}
variable (h_b_ne_0 : b ≠ 0)
def z : ℂ := a + b * Complex.I

-- The statement to prove
theorem complex_relationship {z : ℂ} (h : z = a + b * Complex.I) :
  Complex.abs (z^2) = (Complex.abs z)^2 ∧ Complex.abs (z^2) ≠ z^2 := 
by
  sorry  -- The proof is not required; we skip it with 'sorry'

end complex_relationship_l380_380307


namespace cube_has_12_edges_l380_380250

-- Definition of the number of edges in a cube
def number_of_edges_of_cube : Nat := 12

-- The theorem that asserts the cube has 12 edges
theorem cube_has_12_edges : number_of_edges_of_cube = 12 := by
  -- proof to be filled later
  sorry

end cube_has_12_edges_l380_380250


namespace three_digit_integers_with_at_least_two_identical_digits_l380_380951

/-- Prove that the number of positive three-digit integers less than 700 that have at least two identical digits is 162. -/
theorem three_digit_integers_with_at_least_two_identical_digits : 
  ∃ n : ℕ, (n = 162) ∧ (count_three_digit_integers_with_at_least_two_identical_digits n) :=
by
  sorry

/-- Define a function to count the number of three-digit integers less than 700 with at least two identical digits -/
noncomputable def count_three_digit_integers_with_at_least_two_identical_digits (n : ℕ) : Prop :=
  n = 162

end three_digit_integers_with_at_least_two_identical_digits_l380_380951


namespace digit_150th_l380_380055

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380055


namespace solve_inequality_l380_380712

theorem solve_inequality (x : ℝ) (h : 2 - x ≥ 0) (h0 : x ≠ 0) :
  (2 - x ≥ 0 → (x < 0 ∨ (1 ≤ x ∧ x ≤ 2)) ↔ 
  (\frac{\sqrt 2 - x + 4 * x - 3}{x} ≥ 2)) :=
sorry

end solve_inequality_l380_380712


namespace program_produces_8_l380_380706

noncomputable def program_result : ℕ :=
by
  let S := 0
  let i := 0
  let rec loop S i :=
    if i > 10 then S
    else loop (S + i) (i * i + 1)
  pure $ loop S i

theorem program_produces_8 : program_result = 8 := 
sorry

end program_produces_8_l380_380706


namespace cab_speed_fraction_l380_380448

def usual_time := 30 -- The usual time of the journey in minutes
def delay_time := 6   -- The delay time in minutes
def usual_speed : ℝ := sorry -- Placeholder for the usual speed
def reduced_speed : ℝ := sorry -- Placeholder for the reduced speed

-- Given the conditions:
-- 1. The usual time for the cab to cover the journey is 30 minutes.
-- 2. The cab is 6 minutes late when walking at a reduced speed.
-- Prove that the fraction of the cab's usual speed it is walking at is 5/6

theorem cab_speed_fraction : (reduced_speed / usual_speed) = (5 / 6) :=
sorry

end cab_speed_fraction_l380_380448


namespace min_sum_characteristic_value_l380_380200

-- Definition of nonempty set of numbers and characteristic value
def is_nonempty_set_of_numbers (A : set ℕ) (n : ℕ) := (A ≠ ∅ ∧ card A = n)
def characteristic_value (A : set ℕ) : ℕ := set.max' A sorry + set.min' A sorry

-- Definitions based on conditions
def sets : list (set ℕ) := [A₁, A₂, A₃, A₄, A₅]
def union_sets : set ℕ := {x : ℕ | x > 0 ∧ x ≤ 100}

-- Auxiliary definitions
axiom A₁ A₂ A₃ A₄ A₅ : set ℕ
axiom h₁ : ∀ i ∈ sets, is_nonempty_set_of_numbers i 20
axiom h₂ : (⋃ i ∈ sets, i) = union_sets

-- Statement of the problem
theorem min_sum_characteristic_value : (∑ A in sets, characteristic_value A) = 325 := sorry

end min_sum_characteristic_value_l380_380200


namespace percentage_of_residents_watching_exactly_two_shows_l380_380694

noncomputable def percentage_exactly_two_shows : ℝ :=
  let total := 600 in
  let A := 0.35 * total in
  let B := 0.40 * total in
  let C := 0.50 * total in
  let all_three := 21 in
  let exactly_two := (A + B + C + all_three) - total in
  (exactly_two / total) * 100

theorem percentage_of_residents_watching_exactly_two_shows :
  percentage_exactly_two_shows = 28.5 :=
by
  have total : ℝ := 600
  have A := 0.35 * total
  have B := 0.40 * total
  have C := 0.50 * total
  have all_three := 21
  let exactly_two := (A + B + C + all_three) - total
  let percentage := (exactly_two / total) * 100
  show percentage = 28.5
  sorry  -- proof here

end percentage_of_residents_watching_exactly_two_shows_l380_380694


namespace coin_toss_sequences_l380_380644

theorem coin_toss_sequences :
  ∃ (s : list (list char)),
    (∀ (subseq : list char), subseq ∈ s → 18 = list.length (list.join s)) ∧
    3 = list.count (['H', 'H']) s ∧
    4 = list.count (['H', 'T']) s ∧
    5 = list.count (['T', 'H']) s ∧
    6 = list.count (['T', 'T']) s ∧
    4200 = list.permutations_count s := 
sorry

end coin_toss_sequences_l380_380644


namespace infinitely_many_squares_as_difference_of_cubes_l380_380370

theorem infinitely_many_squares_as_difference_of_cubes :
  ∃^∞ A : ℤ, ∃ k : ℤ, A^2 = (k + 1)^3 - k^3 := sorry

end infinitely_many_squares_as_difference_of_cubes_l380_380370


namespace distance_and_midpoint_l380_380173

def dist (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
def midpoint (x1 y1 x2 y2 : ℝ) := ((x1 + x2) / 2, (y1 + y2) / 2)

theorem distance_and_midpoint:
  let x1 := 4
  let y1 := -3
  let x2 := -6
  let y2 := 5
  dist x1 y1 x2 y2 = 2 * Real.sqrt 41 ∧ midpoint x1 y1 x2 y2 = (-1, 1) :=
by
  sorry

end distance_and_midpoint_l380_380173


namespace energy_drinks_bought_l380_380388

def price_per_cupcake : ℝ := 2
def number_of_cupcakes : ℤ := 50
def price_per_cookie : ℝ := 0.5
def number_of_cookies : ℤ := 40
def price_per_basketball : ℝ := 40
def number_of_basketballs : ℤ := 2
def price_per_energy_drink : ℝ := 2

theorem energy_drinks_bought :
  let total_money_earned := number_of_cupcakes * price_per_cupcake + number_of_cookies * price_per_cookie in
  let cost_of_basketballs := number_of_basketballs * price_per_basketball in
  let money_left := total_money_earned - cost_of_basketballs in
  let number_of_energy_drinks := money_left / price_per_energy_drink in
  number_of_energy_drinks = 20 := 
begin
  sorry
end

end energy_drinks_bought_l380_380388


namespace average_difference_l380_380721

theorem average_difference :
  let avg1 := (24 + 35 + 58) / 3
  let avg2 := (19 + 51 + 29) / 3
  avg1 - avg2 = 6 := by
sorry

end average_difference_l380_380721


namespace sum_n_eq_26_l380_380061

open_locale big_operators

-- Define the condition as a Lean function
def condition (n : ℕ) : Prop :=
  (nat.choose 26 13) + (nat.choose 26 n) = nat.choose 27 14

-- Define the proof problem in Lean
theorem sum_n_eq_26 : (finset.univ.filter condition).sum = 26 := sorry

end sum_n_eq_26_l380_380061


namespace Lindys_speed_l380_380293

theorem Lindys_speed (d_jc : ℕ) (v_j : ℕ) (v_c : ℕ) (d_lindy : ℕ) (h_gap : d_jc = 270) (h_j_speed : v_j = 4) (h_c_speed : v_c = 5) (h_lindy_distance : d_lindy = 240) : 
  (d_lindy / (d_jc / (v_j + v_c))) = 8 :=
by
  rw [h_gap, h_j_speed, h_c_speed, h_lindy_distance]
  simp
  sorry

end Lindys_speed_l380_380293


namespace larger_root_of_quadratic_eq_l380_380536

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l380_380536


namespace cylinder_volume_transformation_l380_380408

variable (r h : ℝ)
variable (V_original : ℝ)
variable (V_new : ℝ)

noncomputable def original_volume : ℝ := Real.pi * r^2 * h

noncomputable def new_volume : ℝ := Real.pi * (3 * r)^2 * (2 * h)

theorem cylinder_volume_transformation 
  (h_original : original_volume r h = 15) :
  new_volume r h = 270 :=
by
  unfold original_volume at h_original
  unfold new_volume
  sorry

end cylinder_volume_transformation_l380_380408


namespace find_remainder_l380_380724

-- Given conditions
def L := 1631
def S := 266
def quotient := 6
def diff := 1365

-- Mathematical relations and constants
axiom condition1 : L - S = diff
axiom condition2 : L = 1631
axiom condition3 : L = quotient * S + R

-- The theorem stating the proof problem
theorem find_remainder : R = 35 :=
by
  sorry

end find_remainder_l380_380724


namespace impossible_to_get_target_l380_380081

-- Define the initial polynomial
def p0 := (fun x : ℝ => x^2 + 4 * x + 3)

-- Define the target polynomial
def p_target := (fun x : ℝ => x^2 + 10 * x + 9)

-- Define the first allowed operation
def transform1 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => x^2 * f (1/x + 1)

-- Define the second allowed operation
def transform2 (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => (x-1)^2 * f (1/(x-1))

-- Prove that it's impossible to obtain p_target starting from p0 using the given operations
theorem impossible_to_get_target :
  ¬ ∃ (ops : List (ℝ → (ℝ → ℝ) → (ℝ → ℝ))), List.foldr (λ op acc f, op (acc f)) id ops = p_target p0 :=
sorry

end impossible_to_get_target_l380_380081


namespace length_of_AX_l380_380967

-- Definitions of the sides of the squares
def AB : ℝ := 13
def BC : ℝ := 13
def PQ : ℝ := 5
def QR : ℝ := 5

-- Definition of the lengths of the diagonals of the squares
def AC := Real.sqrt (AB^2 + BC^2)
def PS := Real.sqrt (PQ^2 + QR^2)

-- Definition of X being the midpoint of AC and PS
def X : ℝ := AC / 2

-- Expected result of the length of AX
def AX_expected : ℝ := Real.sqrt 338 / 2

-- The statement we need to prove
theorem length_of_AX : X = AX_expected :=
by 
  -- Sorry is used to skip the detailed proof steps
  sorry

end length_of_AX_l380_380967


namespace probability_same_gate_l380_380786

-- Conditions translated into Lean definitions
def gates : Finset ℤ := Finset.range 3

-- The mathematically equivalent proof problem Lean statement
theorem probability_same_gate :
  let total_combinations := (gates.card * gates.card)
  let same_gate_combinations := gates.card
  let probability := (same_gate_combinations : ℚ) / total_combinations
  probability = 1 / 3 :=
by
  sorry

end probability_same_gate_l380_380786


namespace limit_a_n_l380_380239

noncomputable def a_n (n : ℕ) : ℝ :=
  if n ≤ 4 then -n else (sqrt (n^2 - 4 * n) - n)

theorem limit_a_n : 
  tendsto (λ n, a_n n) at_top (𝓝 (-2)) :=
sorry

end limit_a_n_l380_380239


namespace analytical_expression_of_f_range_of_area_of_ABC_l380_380228

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem analytical_expression_of_f :
  (∀ x, f(x) = 2 * Real.sin (2 * x + Real.pi / 6)) :=
sorry

theorem range_of_area_of_ABC (A B C a b c : ℝ)
  (h1 : 0 < A) (h2 : A = Real.pi / 3)
  (h3 : b = f(A)) (h4 : b > 0) (h5 : c > 0)
  (h6 : C = Real.pi - A - B) (h7 : B > 0) (h8 : B < Real.pi / 2) :
  let S := (1 / 2) * b * c * Real.sin (A)
  in (sqrt 3 / 8 < S) ∧ (S < sqrt 3 / 2) :=
sorry

end analytical_expression_of_f_range_of_area_of_ABC_l380_380228


namespace diet_soda_bottles_l380_380458

theorem diet_soda_bottles (r d l t : Nat) (h1 : r = 49) (h2 : l = 6) (h3 : t = 89) (h4 : t = r + d) : d = 40 :=
by
  sorry

end diet_soda_bottles_l380_380458


namespace find_DA_find_sum_m_n_l380_380277

noncomputable def scaled_AU := 2
noncomputable def scaled_AN := 3
noncomputable def scaled_UB := 4
noncomputable def scaled_AB := 6
noncomputable def radius_OA_OB := 3
noncomputable def ratio_regions := (2:ℝ)/(3:ℝ)

def calculate_DA (AU AN UB : ℝ) : ℝ :=
  let scale_factor := 63
  let scaled_factor := sqrt 6 / 2
  scaled_factor * scale_factor * 3

theorem find_DA :
  let AU := 126
  let AN := 189
  let UB := 252
  calculate_DA AU AN UB = 94.5 * sqrt 6 := by
  sorry

theorem find_sum_m_n :
  let DA := calculate_DA 126 189 252
  let m := 94
  let n := 6
  m + n = 100 := by
  sorry

end find_DA_find_sum_m_n_l380_380277


namespace second_number_is_correct_l380_380085

theorem second_number_is_correct (x : Real) (h : 108^2 + x^2 = 19928) : x = Real.sqrt 8264 :=
by
  sorry

end second_number_is_correct_l380_380085


namespace general_formula_a_n_range_m_l380_380894

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

axiom S4_S2 : S_n 4 - S_n 2 = 7 * a_n 1
axiom S5_30 : S_n 5 = 30

theorem general_formula_a_n : a_n = λ n, 2 * n :=
by
  sorry

noncomputable def b_n (n : ℕ) := 1 / S_n n

noncomputable def T_n : ℕ → ℝ :=
∑ i in finset.range n, b_n (i + 1)

theorem range_m (m : ℝ) (h : ∀ n : ℕ, T_n n < real.log (m ^ 2 - m) / real.log 2) : 
  m ∈ set.Icc (-∞) (-1) ∪ set.Icc 2 ∞ :=
by
  sorry

end general_formula_a_n_range_m_l380_380894


namespace original_cost_price_l380_380463

theorem original_cost_price (P : ℝ) 
  (h1 : P - 0.07 * P = 0.93 * P)
  (h2 : 0.93 * P + 0.02 * 0.93 * P = 0.9486 * P)
  (h3 : 0.9486 * P * 1.05 = 0.99603 * P)
  (h4 : 0.93 * P * 0.95 = 0.8835 * P)
  (h5 : 0.8835 * P + 0.02 * 0.8835 * P = 0.90117 * P)
  (h6 : 0.99603 * P - 5 = (0.90117 * P) * 1.10)
: P = 5 / 0.004743 :=
by
  sorry

end original_cost_price_l380_380463


namespace coordinates_of_terminal_side_l380_380903

theorem coordinates_of_terminal_side (α : ℝ) (x y r : ℝ) :
  sin α = 3 / 5 ∧ cos α = -4 / 5 ∧ r = 5 → (x, y) = (-4, 3) :=
by 
  sorry

end coordinates_of_terminal_side_l380_380903


namespace quadratic_has_real_roots_l380_380872

theorem quadratic_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, K^2 * x^2 - (4 * K^2 + 1) * x + 3 * K^2 = 0 :=
begin
  -- Proof required here, using discriminant analysis.
  sorry
end

end quadratic_has_real_roots_l380_380872


namespace sale_in_fifth_month_correct_l380_380091

-- Definitions for given conditions
def sales_month_1 : ℕ := 6435
def sales_month_2 : ℕ := 6927
def sales_month_3 : ℕ := 6855
def sales_month_4 : ℕ := 7230
def sales_month_6 : ℕ := 7991
def average_sales : ℕ := 7000
def total_sales : ℕ := 42000

-- To compute the sale in the fifth month
def sale_in_fifth_month (total_sales : ℕ) (sum_first_four_months : ℕ) (sales_month_6 : ℕ) : ℕ :=
  total_sales - (sum_first_four_months + sales_month_6)

-- The proof problem to be stated
theorem sale_in_fifth_month_correct (sales_month_1 sales_month_2 sales_month_3 sales_month_4 sales_month_6 average_sales total_sales : ℕ) :
  total_sales = 42000 →
  sales_month_1 = 6435 →
  sales_month_2 = 6927 →
  sales_month_3 = 6855 →
  sales_month_4 = 7230 →
  sales_month_6 = 7991 →
  average_sales = 7000 →
  42000 - (sales_month_1 + sales_month_2 + sales_month_3 + sales_month_4 + sales_month_6) = 6562 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end sale_in_fifth_month_correct_l380_380091


namespace cos_theta_value_sin_theta_plus_pi_over_3_value_l380_380191

variable (θ : ℝ)
variable (H1 : 0 < θ ∧ θ < π / 2)
variable (H2 : Real.sin θ = 4 / 5)

theorem cos_theta_value : Real.cos θ = 3 / 5 := sorry

theorem sin_theta_plus_pi_over_3_value : 
    Real.sin (θ + π / 3) = (4 + 3 * Real.sqrt 3) / 10 := sorry

end cos_theta_value_sin_theta_plus_pi_over_3_value_l380_380191


namespace sum_of_exponents_of_1985_eq_40_l380_380525

theorem sum_of_exponents_of_1985_eq_40 :
  ∃ (e₀ e₁ e₂ e₃ e₄ e₅ : ℕ), 1985 = 2^e₀ + 2^e₁ + 2^e₂ + 2^e₃ + 2^e₄ + 2^e₅ 
  ∧ e₀ ≠ e₁ ∧ e₀ ≠ e₂ ∧ e₀ ≠ e₃ ∧ e₀ ≠ e₄ ∧ e₀ ≠ e₅
  ∧ e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₁ ≠ e₅
  ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₂ ≠ e₅
  ∧ e₃ ≠ e₄ ∧ e₃ ≠ e₅
  ∧ e₄ ≠ e₅
  ∧ e₀ + e₁ + e₂ + e₃ + e₄ + e₅ = 40 := 
by
  sorry

end sum_of_exponents_of_1985_eq_40_l380_380525


namespace num_even_4digit_numbers_l380_380425

def digits : List ℕ := [5, 6, 4, 7]

theorem num_even_4digit_numbers (d : List ℕ) (h : d = digits) : 
  ∃ n : ℕ, n = 12 ∧ ∀ num ∈ d.permutations, (num.length = 4 ∧ num.last % 2 = 0) → count_even_4digit_numbers d = n :=
by
  sorry

noncomputable def count_even_4digit_numbers (d : List ℕ) : ℕ := 
  (if 6 ∈ d then 6 else 0) + (if 4 ∈ d then 6 else 0)

end num_even_4digit_numbers_l380_380425


namespace a_put_10_oxen_l380_380109

noncomputable def number_of_oxen_A_put (A_months : ℕ) (B_oxen : ℕ) (B_months : ℕ) 
(C_oxen : ℕ) (C_months : ℕ) (total_rent : ℝ) (C_rent : ℝ) : ℕ :=
let total_B := B_oxen * B_months
let total_C := C_oxen * C_months
let cost_per_ox_per_month := C_rent / total_C
let total_ox_months := total_rent / cost_per_ox_per_month
(X_ox_month := total_ox_months - total_B - total_C) / A_months

theorem a_put_10_oxen : number_of_oxen_A_put 7 12 5 15 3 175 45 = 10 :=
by sorry

end a_put_10_oxen_l380_380109


namespace digit_150th_in_decimal_of_fraction_l380_380048

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380048


namespace diagonal_difference_l380_380678

noncomputable def f : ℕ → ℕ
| n := (n * (n - 3)) / 2

theorem diagonal_difference {n : ℕ} (h : n ≥ 4): f(n + 1) - f(n) = n - 1 := 
by 
  sorry

end diagonal_difference_l380_380678


namespace digit_150_in_17_div_70_l380_380005

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380005


namespace checkerboard_pattern_is_joyous_l380_380517

def CellColor : Type := Bool
def is_blue (cell : CellColor) : Prop := cell = true
def is_white (cell : CellColor) : Prop := cell = false
def neighbors (board : ℕ × ℕ → CellColor) (i j : ℕ) : list (ℕ × ℕ) :=
  [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

def is_joyous (board : ℕ × ℕ → CellColor) (i j : ℕ) : Prop :=
  countp is_blue (neighbors board i j) = 2

def checkerboard (i j : ℕ) : CellColor := 
  if (i + j) % 2 = 0 then true else false

theorem checkerboard_pattern_is_joyous :
  ∀ i j, i < 10 → j < 10 → is_joyous checkerboard i j :=
by sorry

end checkerboard_pattern_is_joyous_l380_380517


namespace function_B_increasing_on_interval_l380_380115

noncomputable def function_B (x : ℝ) := x^2 + 1

theorem function_B_increasing_on_interval : 
  ∀ x ∈ set.Ioo 0 2, deriv function_B x > 0 :=
by
  sorry

end function_B_increasing_on_interval_l380_380115


namespace digit_150_of_17_div_70_is_2_l380_380012

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380012


namespace digit_150_in_17_div_70_l380_380007

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380007


namespace triangle_area_l380_380729

theorem triangle_area
  (a b c : ℝ)
  (ha : a = Real.sqrt 2)
  (hb : b = Real.sqrt 3)
  (hc : c = 2) :
  let S := Real.sqrt ((1 / 4) * (c^2 * a^2 - ((c^2 + a^2 - b^2) / 2)^2))
  in S = Real.sqrt 23 / 4 :=
by
  sorry

end triangle_area_l380_380729


namespace rosy_current_age_l380_380434

theorem rosy_current_age 
  (R : ℕ) 
  (h1 : ∀ (david_age rosy_age : ℕ), david_age = rosy_age + 12) 
  (h2 : ∀ (david_age_plus_4 rosy_age_plus_4 : ℕ), david_age_plus_4 = 2 * rosy_age_plus_4) : 
  R = 8 := 
sorry

end rosy_current_age_l380_380434


namespace repeating_decimal_to_fraction_l380_380169

theorem repeating_decimal_to_fraction : (2.353535... : Rational) = 233/99 :=
by
  sorry

end repeating_decimal_to_fraction_l380_380169


namespace arithmetic_mean_twice_y_l380_380907

theorem arithmetic_mean_twice_y (y x : ℝ) (h1 : (8 + y + 24 + 6 + x) / 5 = 12) (h2 : x = 2 * y) :
  y = 22 / 3 ∧ x = 44 / 3 :=
by
  sorry

end arithmetic_mean_twice_y_l380_380907


namespace digit_150_in_17_div_70_l380_380006

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380006


namespace fill_time_l380_380489

def length := 1.5 -- meters
def width := 0.4 -- meters
def height := 0.8 -- meters
def volume := length * width * height -- cubic meters
def rate := 0.00003333 -- cubic meters per second

theorem fill_time : volume / rate = 14400 :=
by sorry

end fill_time_l380_380489


namespace opposite_of_8_is_neg_8_l380_380116

theorem opposite_of_8_is_neg_8 : - (8 : ℤ) = -8 :=
by
  sorry

end opposite_of_8_is_neg_8_l380_380116


namespace reconstruct_parallelogram_l380_380437

variables {Point : Type} [euclidean_space Point]

-- Definitions for the points
variables (A M N : Point)
-- Definitions for the segments
variables (C D B : Point)

-- Conditions
variables (magical_ruler : (Point → Point → Point) → Prop)
variables (is_midpoint : magical_ruler → Prop)
variables (equal_segments : magical_ruler → Prop)
variables (extend_segment : magical_ruler → Prop)

-- Midpoint condition
def midpoint_condition (M C D : Point) : Prop :=
  is_midpoint M C D

-- Equal segment condition
def equal_segment_condition (CB BN : Point) : Prop :=
  equal_segments CB BN

-- The main theorem
theorem reconstruct_parallelogram
  (midpoint_cond : midpoint_condition M C D)
  (equal_segment_cond : equal_segment_condition C B N)
  (magic_ruler : magical_ruler Point)
  : ∃ (A B C D : Point), 
    parallelogram A B C D ∧ 
    midpoint_condition M C D ∧ 
    equal_segment_condition C B N :=
sorry

end reconstruct_parallelogram_l380_380437


namespace cost_price_A_l380_380077

-- Establishing the definitions based on the conditions from a)

def profit_A_to_B (CP_A : ℝ) : ℝ := 1.20 * CP_A
def profit_B_to_C (CP_B : ℝ) : ℝ := 1.25 * CP_B
def price_paid_by_C : ℝ := 222

-- Stating the theorem to be proven:
theorem cost_price_A (CP_A : ℝ) (H : profit_B_to_C (profit_A_to_B CP_A) = price_paid_by_C) : CP_A = 148 :=
by 
  sorry

end cost_price_A_l380_380077


namespace trapezoid_perimeter_44_l380_380980

variables (A B C D : ℝ) (AB CD AC BD : ℝ)

def is_trapezoid (A B C D : ℝ) : Prop :=
  -- The vertical legs are each 6 units
  A = 6 ∧ C = 6 ∧
  -- The horizontal difference is 8 units each
  B = 8 ∧ D = 8 ∧
  -- AB and CD are hypotenuses of right triangles with legs 6 and 8
  AB = Real.sqrt (A^2 + B^2) ∧ CD = Real.sqrt (C^2 + D^2)

theorem trapezoid_perimeter_44 (h : is_trapezoid A B C D)
  (H_AB : AB = Real.sqrt (6^2 + 8^2))
  (H_CD : CD = Real.sqrt (6^2 + 8^2))
  (H_AC : AC = 8)
  (H_BD : BD = 16) :
  AB + AC + CD + BD = 44 :=
by sorry

end trapezoid_perimeter_44_l380_380980


namespace total_payment_correct_l380_380759

def payment_X (payment_Y : ℝ) : ℝ := 1.2 * payment_Y
def payment_Y : ℝ := 254.55
def total_payment (payment_X payment_Y : ℝ) : ℝ := payment_X + payment_Y

theorem total_payment_correct :
  total_payment (payment_X payment_Y) payment_Y = 560.01 :=
by
  sorry

end total_payment_correct_l380_380759


namespace total_value_of_button_collection_l380_380988

theorem total_value_of_button_collection:
  (∀ (n : ℕ) (v : ℕ), n = 2 → v = 8 → has_same_value → total_value = 10 * (v / n)) →
  has_same_value :=
  sorry

end total_value_of_button_collection_l380_380988


namespace digit_150_of_17_div_70_is_2_l380_380013

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380013


namespace smallest_k_for_marked_cells_in_9x9_board_l380_380060

theorem smallest_k_for_marked_cells_in_9x9_board :
  ∃ (k : ℕ), (∀ (board : Fin 9 × Fin 9 → Bool), (∀ (c : Fin 3 × Fin 3), ∃ (i j : Fin 3), board (i, j) = true) → (∀ (c : Fin 3 × Fin 3), (board (c.fst, c.snd) = false → c ≠ (2, 2))) → k = 56 ∧
    (∀ (placement : Fin 2 × Fin 2 → Bool), (placement (Fin.mk 0 (by norm_num), Fin.mk 0 (by norm_num)) = true ∧
                                            placement (Fin.mk 0 (by norm_num), Fin.mk 1 (by norm_num)) = true ∧
                                            placement (Fin.mk 1 (by norm_num), Fin.mk 0 (by norm_num)) = true) →
     ∃ (cells_marked : Fin 9 × Fin 9 → Bool), (∀ (i j : Fin 9), cells_marked (i, j) = true → board (i, j) = true) → ¬ cells_marked (Fin.mk 8 (by norm_num), Fin.mk 8 (by norm_num)))) :=
begin
  sorry
end

end smallest_k_for_marked_cells_in_9x9_board_l380_380060


namespace sum_arithmetic_sequence_l380_380574

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n + 1) / 2 * (2 * a 0 + n * (a 1 - a 0))

theorem sum_arithmetic_sequence (h_arith : arithmetic_sequence a) (h_condition : a 3 + a 4 + a 5 + a 6 = 18) :
  S a 9 = 45 :=
sorry

end sum_arithmetic_sequence_l380_380574


namespace card_pair_probability_l380_380798

theorem card_pair_probability :
  let total_cards := 52
  let pair_removed_cards := total_cards - 2
  let remaining_cards := pair_removed_cards
  let choose_two : ℕ := remaining_cards.choose 2
  let total_ways := 12 * (4.choose 2) + 1 * (2.choose 2)
  let pair_probability := (total_ways : ℚ) / choose_two
  let m := 73
  let n := 1225
  m.gcd n = 1 ∧ pair_probability = (m : ℚ) / n ∧ m + n = 1298 := by
  sorry

end card_pair_probability_l380_380798


namespace union_of_M_and_N_complement_of_intersection_l380_380573

variable (U : Set ℝ) (M : Set ℝ) (N : Set ℝ)

noncomputable def U := { x : ℝ | -6 ≤ x ∧ x ≤ 5 }
noncomputable def M := { x : ℝ | -3 ≤ x ∧ x ≤ 2 }
noncomputable def N := { x : ℝ | 0 < x ∧ x < 2 }

theorem union_of_M_and_N : M ∪ N = { x : ℝ | -3 ≤ x ∧ x ≤ 2 } := by
  sorry

theorem complement_of_intersection :
  U \ (M ∩ N) = { x : ℝ | (-6 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 5) } := by
  sorry

end union_of_M_and_N_complement_of_intersection_l380_380573


namespace find_integer_k_l380_380624

theorem find_integer_k : ∃ k : ℤ, 3 * k ^ 2 - 14 * k + 8 = 0 ∧ k = 4 :=
by
  use 4
  split
  sorry
  rfl

end find_integer_k_l380_380624


namespace find_dot_product_l380_380934

noncomputable def A : ℝ × ℝ := (0, -2)
noncomputable def B : ℝ × ℝ := (0, 2)

def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2) / 12 + (P.2^2) / 16 = 1

def condition (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let BP := (P.1 - B.1)^2 + (P.2 - B.2)^2
  real.sqrt AP - real.sqrt BP = 2

def dot_product (P : ℝ × ℝ) : ℝ :=
  (P.1, P.2 + 2) • (P.1, P.2 - 2)

theorem find_dot_product (P : ℝ × ℝ) (h1 : ellipse P) (h2 : condition P) : dot_product P = 9 := 
  sorry

end find_dot_product_l380_380934


namespace number_of_alternating_numbers_l380_380506

/-- 
  We want to count the number of five-digit numbers that can be formed from 
  three odd digits chosen from {1, 3, 5, 7, 9} and two even digits chosen from {2, 4, 6, 8},
  such that the digits alternate between odd and even, with no repeating digits.
 -/
theorem number_of_alternating_numbers : 
  let odds := {1, 3, 5, 7, 9}
  let evens := {2, 4, 6, 8}
  (∃ f : Fin 5 → Nat, 
    (∀ i, f i ∈ odds ∧ ∀ j, f j ∈ evens ∧ ∀ i, j, i ≠ j → f i ≠ f j ∧ 
    ((even i ∧ f i ∈ odds) ∨ (odd i ∧ f i ∈ evens))) 
  ) :=
  720 := sorry

end number_of_alternating_numbers_l380_380506


namespace digit_150th_of_17_div_70_is_7_l380_380023

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380023


namespace skeleton_ratio_l380_380272

theorem skeleton_ratio (W M C : ℕ) 
  (h1 : W + M + C = 20)
  (h2 : M = C)
  (h3 : 20 * W + 25 * M + 10 * C = 375) :
  (W : ℚ) / (W + M + C) = 1 / 2 :=
by
  sorry

end skeleton_ratio_l380_380272


namespace trig_identity_proof_l380_380397

theorem trig_identity_proof :
  (2 * real.sin (real.pi * 47 / 180) - real.sqrt 3 * real.sin (real.pi * 17 / 180)) / real.cos (real.pi * 17 / 180) = 1 := 
by
  sorry

end trig_identity_proof_l380_380397


namespace green_chameleon_increase_l380_380330

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380330


namespace digit_150_in_17_div_70_l380_380009

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380009


namespace greatest_perimeter_of_integer_sided_triangle_is_37_l380_380646

noncomputable def greatest_perimeter_of_integer_sided_triangle : ℕ :=
  let x_min := (10 : ℕ) / 3 + 1 in
  let x_max := 10 in
  let perimeters := List.map (λ x, x + 2 * x + 10) (List.range' x_min (x_max - x_min + 1)) in
  List.maximum' perimeters

theorem greatest_perimeter_of_integer_sided_triangle_is_37 :
  greatest_perimeter_of_integer_sided_triangle = 37 :=
by
  sorry

end greatest_perimeter_of_integer_sided_triangle_is_37_l380_380646


namespace three_digit_integers_with_at_least_two_identical_digits_l380_380954

/-- Prove that the number of positive three-digit integers less than 700 that have at least two identical digits is 162. -/
theorem three_digit_integers_with_at_least_two_identical_digits : 
  ∃ n : ℕ, (n = 162) ∧ (count_three_digit_integers_with_at_least_two_identical_digits n) :=
by
  sorry

/-- Define a function to count the number of three-digit integers less than 700 with at least two identical digits -/
noncomputable def count_three_digit_integers_with_at_least_two_identical_digits (n : ℕ) : Prop :=
  n = 162

end three_digit_integers_with_at_least_two_identical_digits_l380_380954


namespace relationship_of_AT_l380_380827

def S : ℝ := 300
def PC : ℝ := S + 500
def total_cost : ℝ := 2200

theorem relationship_of_AT (AT : ℝ) 
  (h1: S + PC + AT = total_cost) : 
  AT = S + PC - 400 :=
by
  sorry

end relationship_of_AT_l380_380827


namespace trig_identity_cos2theta_tan_minus_pi_over_4_l380_380558

variable (θ : ℝ)

-- Given condition
def tan_theta_is_2 : Prop := Real.tan θ = 2

-- Proof problem 1: Prove that cos(2θ) = -3/5
def cos2theta (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.cos (2 * θ) = -3 / 5

-- Proof problem 2: Prove that tan(θ - π/4) = 1/3
def tan_theta_minus_pi_over_4 (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.tan (θ - Real.pi / 4) = 1 / 3

-- Main theorem statement
theorem trig_identity_cos2theta_tan_minus_pi_over_4 
  (θ : ℝ) (h : tan_theta_is_2 θ) :
  cos2theta θ h ∧ tan_theta_minus_pi_over_4 θ h :=
sorry

end trig_identity_cos2theta_tan_minus_pi_over_4_l380_380558


namespace description_of_T_l380_380675

def set_T (x y : ℝ) : Prop :=
  (4 ≤ x - 1 ∧ 4 ≤ y + 3 ∧ y + 3 < 4) ∨
  (4 ≤ y + 3 ∧ 4 ≤ x - 1 ∧ x - 1 < 4) ∨
  (x - 1 ≤ y + 3 ∧ y + 3 ≤ 4 ∧ 4 < x - 1)

theorem description_of_T :
  ∃ T : ℝ × ℝ → Prop, T = set_T ∧ 
  (∃ x₀ y₀, T (x₀, y₀) ∧
           (∀ x y, (T (x, y) → ((x = 5 ∧ y < 1) ∨ (y = 1 ∧ x < 5) ∨ (y = x - 4 ∧ x > 5))) ∧
           (∀ x y, ((x = 5 ∧ y < 1) ∨ (y = 1 ∧ x < 5) ∨ (y = x - 4 ∧ x > 5)) → T (x, y)))) :=
by
    sorry

end description_of_T_l380_380675


namespace trailing_zeros_base_4_of_12_factorial_l380_380254

theorem trailing_zeros_base_4_of_12_factorial : (nat.trailing_zeros 12!) 4 = 5 := by
  sorry

end trailing_zeros_base_4_of_12_factorial_l380_380254


namespace value_of_series_l380_380931

noncomputable def solve_problem : ℤ :=
  let M := {x, x * y, log (x * y)} in
  let N := {0, abs x, y} in

  have h1 : M = N := sorry,
  have h2 : x * y = 1 := sorry,
  have h3 : x = -1 := sorry,
  have h4 : y = -1 := sorry,

  let s := finset.range 2001 in
  let series := s.sum (λ k, (-1)^(k + 1) + (-1)^(k + 1)) in
  series + (-1)^(2001 + 1) + (-1)^(2001 + 1)

theorem value_of_series : solve_problem = -2002 :=
  sorry

end value_of_series_l380_380931


namespace repeating_decimal_as_fraction_l380_380166

theorem repeating_decimal_as_fraction : 
  let x := 2.353535... in
  x = 233 / 99 ∧ Nat.gcd 233 99 = 1 :=
by
  sorry

end repeating_decimal_as_fraction_l380_380166


namespace geom_seq_specified_values_l380_380567

noncomputable def geom_seq_general_formula (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = (7 / 32) * 2^n

theorem geom_seq_specified_values (a : ℕ → ℝ) (q : ℝ) :
  (a 5 = 7) → (a 8 = 56) → ∃ q : ℝ, ∀ n : ℕ, a n = (7 / 32) * 2^n :=
begin
  intros h5 h8,
  use 2,
  sorry
end

end geom_seq_specified_values_l380_380567


namespace traveled_distance_l380_380831

def distance_first_day : ℕ := 5 * 7
def distance_second_day_part1 : ℕ := 6 * 6
def distance_second_day_part2 : ℕ := (6 / 2) * 3
def distance_third_day : ℕ := 7 * 5

def total_distance : ℕ := distance_first_day + distance_second_day_part1 + distance_second_day_part2 + distance_third_day

theorem traveled_distance : total_distance = 115 := by
  unfold total_distance
  unfold distance_first_day distance_second_day_part1 distance_second_day_part2 distance_third_day
  norm_num
  rfl

end traveled_distance_l380_380831


namespace shaded_fraction_is_half_l380_380103

/-- If a square is divided into nine smaller squares of equal area, then the center square is 
divided into four smaller squares of equal area, and this pattern continues indefinitely, the 
fractional part of the figure that is shaded is 1/2. -/
theorem shaded_fraction_is_half : 
  (∃ (S : ℕ → ℝ), S 0 = 4/9 ∧ ∀ n, S (n + 1) = (S n / 36) ∧ has_sum S (1/2)) :=
by
  sorry

end shaded_fraction_is_half_l380_380103


namespace inscribed_circle_max_radius_0_0_inscribed_circle_max_radius_non_0_0_l380_380882

variables (a : ℝ) (r : ℝ)

def curve (x y : ℝ) (a : ℝ) := abs y = 1 - a * x^2 ∧ abs x ≤ 1 / sqrt a

def f_a (a : ℝ) : ℝ :=
  if 0 < a ∧ a ≤ 1/2 then 1 else if a > 1/2 then sqrt (4 * a - 1) / (2 * a) else 0

theorem inscribed_circle_max_radius_0_0 (h : 0 < a) :
  ∃ r, ∀ x y, curve x y a → r ≤ f_a a :=
sorry

theorem inscribed_circle_max_radius_non_0_0 (h : 0 < a) (p q : ℝ) (hpq : (p, q) ≠ (0, 0)) :
  ∃ r, ∀ x y, curve x y a → r ≤ f_a a :=
sorry

end inscribed_circle_max_radius_0_0_inscribed_circle_max_radius_non_0_0_l380_380882


namespace triangle_rectangle_ratio_l380_380119

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l380_380119


namespace smallest_positive_period_of_f_range_of_f_in_interval_l380_380223

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem smallest_positive_period_of_f (a : ℝ) (h : f a (π / 3) = 0) :
  ∃ T : ℝ, T = 2 * π ∧ (∀ x, f a (x + T) = f a x) :=
sorry

theorem range_of_f_in_interval (a : ℝ) (h : f a (π / 3) = 0) :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a x ∧ f a x ≤ 2 :=
sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l380_380223


namespace average_percentage_decrease_selling_price_l380_380796

theorem average_percentage_decrease (p2019 p2021 : ℝ) (p2019_eq : p2019 = 144) (p2021_eq : p2021 = 100) :
  ∃ x : ℝ, (1 - x / 100) ^ 2 * p2019 = p2021 ∧ x = 16.67 :=
  by
  sorry
 
theorem selling_price (y : ℝ) (profit eq_price : ℝ) (sold_units : ℕ → ℕ) (profit_eq : profit = 1250) 
  (price_eq : eq_price = 140) (current_units : sold_units 140 = 20)
  (increase_units : ∀ p, p > 0 → sold_units (eq_price - p) = current_units + 2 *⌊p / 5⌋) :
  ∃ s : ℝ, s = 125 ∧ (s - 100) * (sold_units s) = profit :=
  by
  sorry

end average_percentage_decrease_selling_price_l380_380796


namespace range_of_a_real_root_l380_380590

theorem range_of_a_real_root :
  (∀ x : ℝ, x^2 - a * x + 4 = 0 → ∃ x : ℝ, (x^2 - a * x + 4 = 0 ∧ (a ≥ 4 ∨ a ≤ -4))) ∨
  (∀ x : ℝ, x^2 + (a-2) * x + 4 = 0 → ∃ x : ℝ, (x^2 + (a-2) * x + 4 = 0 ∧ (a ≥ 6 ∨ a ≤ -2))) ∨
  (∀ x : ℝ, x^2 + 2 * a * x + a^2 + 1 = 0 → False) →
  (a ≥ 4 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_real_root_l380_380590


namespace planes_intersection_parallel_to_line_l380_380245

open Plane Geometry

theorem planes_intersection_parallel_to_line
  (α β : Plane)
  (m l : Line)
  (h1 : α ∩ β = m)
  (h2 : l ∥ α)
  (h3 : l ∥ β) :
  m ∥ l :=
by
  sorry

end planes_intersection_parallel_to_line_l380_380245


namespace second_player_wins_l380_380697

theorem second_player_wins :
  ∀ (n : ℕ), n ∈ {1, 2, ..., 1000} →
  ∃ (m : ℕ), m ∈ {1, 2, ..., 1000} ∧ (n + m = 1001) →
  (n^2 - m^2) % 13 = 0 :=
by {
  intros n hn,
  use 1001 - n,
  split,
  {
    sorry -- prove that 1001 - n is in the set
  },
  {
    intro h,
    have h1: n + (1001 - n) = 1001 := by linarith,
    rw h1,
    sorry -- prove divisibility by 13
  }
}

end second_player_wins_l380_380697


namespace correct_option_exponent_equality_l380_380780

theorem correct_option_exponent_equality (a b : ℕ) : 
  (\left(2 * a * b^2\right)^2 = 4 * a^2 * b^4) :=
by
  sorry

end correct_option_exponent_equality_l380_380780


namespace cosine_angle_AB_AC_quadrilateral_is_trapezoid_l380_380217

-- Definitions of points A, B, C, and D
def A : (ℝ × ℝ) := (1, 0)
def B : (ℝ × ℝ) := (7, 3)
def C : (ℝ × ℝ) := (4, 4)
def D : (ℝ × ℝ) := (2, 3)

-- Vectors AB and AC
def vecAB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)
def vecAC : (ℝ × ℝ) := (C.1 - A.1, C.2 - A.2)

-- Proof that cosine of the angle between vectors AB and AC is 2√5 / 5
theorem cosine_angle_AB_AC : 
  (vecAB.1 * vecAC.1 + vecAB.2 * vecAC.2) / ((Real.sqrt (vecAB.1^2 + vecAB.2^2)) * (Real.sqrt (vecAC.1^2 + vecAC.2^2))) 
  = 2 * Real.sqrt 5 / 5 := 
sorry

-- Vectors AD and BC
def vecAD : (ℝ × ℝ) := (D.1 - A.1, D.2 - A.2)
def vecBC : (ℝ × ℝ) := (C.1 - B.1, C.2 - B.2)

-- Proof that quadrilateral ABCD is a trapezoid
theorem quadrilateral_is_trapezoid :
  (vecAB.1 / vecAB.2 = vecDC.1 / vecDC.2) ∧
  (Real.sqrt (vecAD.1^2 + vecAD.2^2) = Real.sqrt (vecBC.1^2 + vecBC.2^2)) :=
sorry

end cosine_angle_AB_AC_quadrilateral_is_trapezoid_l380_380217


namespace sum_divisible_by_12_l380_380502

theorem sum_divisible_by_12 :
  ((2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12) = 3 := by
  sorry

end sum_divisible_by_12_l380_380502


namespace evaluate_expression_l380_380143

def g (x : ℝ) : ℝ := x^2 + 3 * real.sqrt x

theorem evaluate_expression : 3 * g 3 - g 9 = -63 + 9 * real.sqrt 3 := by
  sorry

end evaluate_expression_l380_380143


namespace num_points_common_to_graphs_l380_380153

theorem num_points_common_to_graphs :
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  ∀ (x y : ℝ), ((2 * x - y + 3 = 0 ∨ 4 * x + y - 5 = 0) ∧ (x + y - 3 = 0 ∨ 3 * x - 4 * y + 8 = 0)) →
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 :=
sorry

end num_points_common_to_graphs_l380_380153


namespace cone_tangent_planes_angle_l380_380761

theorem cone_tangent_planes_angle 
  (cone1 cone2 : Cone) (V : Point)
  (h1 : cone1.vertex = V)
  (h2 : cone2.vertex = V)
  (h3 : cone1.generator = cone2.generator)
  (h4 : cone1.axial_section_angle = 60)
  (h5 : cone2.axial_section_angle = 60) : 
  ∃ γ, γ = 2 * Real.arccos (Real.sqrt (2 / 3)) ∧ γ = Real.arccos (1 / 3) :=
  sorry

end cone_tangent_planes_angle_l380_380761


namespace chameleon_increase_l380_380355

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380355


namespace unique_m_exists_l380_380863

theorem unique_m_exists (P Q : ℝ → ℝ) (R : ℝ × ℝ → ℝ) :
  (∀ a b : ℝ, a^m = b^2 → (P(R(a, b)) = a ∧ Q(R(a, b)) = b)) ↔ m = 1 :=
by sorry

end unique_m_exists_l380_380863


namespace part1_monotonicity_part2_inequality_l380_380592

-- Definitions and conditions for part 1
def f1 (x : ℝ) : ℝ := Real.exp x - 2 * x
def f1_deriv (x : ℝ) : ℝ := Real.exp x - 2

theorem part1_monotonicity :
  let x := Real.log 2
  ∀ x : ℝ, (x < Real.log 2 ∧ f1_deriv x < 0) ∨ (x > Real.log 2 ∧ f1_deriv x > 0) :=
by
  sorry

-- Definitions and conditions for part 2
def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x
noncomputable def f_deriv (x : ℝ) (m : ℝ) : ℝ := Real.exp x - 2 * m * x - 2
noncomputable def f'' (x : ℝ) (m : ℝ) : ℝ := Real.exp x - 2 * m

theorem part2_inequality (m : ℝ) (h : m < (Real.exp 1) / 2 - 1) :
  ∀ x : ℝ, 0 ≤ x → f x m > (Real.exp 1) / 2 - 1 :=
by
  sorry

end part1_monotonicity_part2_inequality_l380_380592


namespace cube_edges_count_l380_380249

theorem cube_edges_count : (∀ (A : Type), A ≈ cube -> 12 edges = true :=
by
  sorry

end cube_edges_count_l380_380249


namespace sum_first_5_terms_l380_380310

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

theorem sum_first_5_terms {a : ℕ → ℝ}
  (h_seq : is_arithmetic_sequence a)
  (h_cond : a 1 + a 3 = 6) :
  sum_of_first_n_terms a 5 = 15 :=
by sorry

end sum_first_5_terms_l380_380310


namespace range_of_a_l380_380609

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a (a : ℝ) (h : set_A ∪ set_B a = set_A) : 0 ≤ a ∧ a < 4 := 
sorry

end range_of_a_l380_380609


namespace xiao_ming_subject_selection_l380_380746

theorem xiao_ming_subject_selection :
  let subjects := ["Physics", "Chemistry", "Biology", "Politics", "History", "Geography"]
  let remaining_subjects := ["Chemistry", "Biology", "Politics", "History", "Geography"]
  let num_remaining_subjects := List.length remaining_subjects
  num_remaining_subjects.choose 2 = 10 :=
by
  let subjects := ["Physics", "Chemistry", "Biology", "Politics", "History", "Geography"]
  let remaining_subjects := ["Chemistry", "Biology", "Politics", "History", "Geography"]
  have h1 : num_remaining_subjects = 5, by rfl
  have h2 : num_remaining_subjects.choose 2 = 10, by sorry
  exact h2

end xiao_ming_subject_selection_l380_380746


namespace max_value_expression_l380_380911

theorem max_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 2 * x + y + z = 4) : 
  {e : ℝ // e = x^2 + x * (y + z) + y * z} ≤ 4 :=
sorry

end max_value_expression_l380_380911


namespace trig_func_evaluation_l380_380258

theorem trig_func_evaluation :
  (∀ x : ℝ, f (cos x) = cos (2 * x)) → f (sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  intro h
  sorry

end trig_func_evaluation_l380_380258


namespace chameleon_problem_l380_380359

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380359


namespace geometric_sequence_problem_l380_380587

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n - 1) * a (n + 1) = a n * a n)
  (h_int : a 2013 + a 2015 = ∫ x in 0 .. 2, real.sqrt (4 - x^2)) :
  a 2014 * (a 2012 + 2 * a 2014 + a 2016) = real.pi ^ 2 :=
by 
  sorry

end geometric_sequence_problem_l380_380587


namespace enclosed_area_is_half_pi_cubed_l380_380925

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (Real.pi^2 - x^2)
def lower_bound : ℝ := -Real.pi
def upper_bound : ℝ := Real.pi

theorem enclosed_area_is_half_pi_cubed :
  ∫ x in lower_bound..upper_bound, g x - f x = (Real.pi^3) / 2 :=
by
  sorry

end enclosed_area_is_half_pi_cubed_l380_380925


namespace circle_and_line_properties_l380_380197

-- Define the circle C with center on the positive x-axis and passing through the origin
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l: y = kx + 2
def line_l (k x y : ℝ) : Prop := y = k * x + 2

-- Statement: the circle and line setup
theorem circle_and_line_properties (k : ℝ) : 
  ∀ (x y : ℝ), 
  circle_C x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
  line_l k x1 y1 ∧ 
  line_l k x2 y2 ∧ 
  circle_C x1 y1 ∧ 
  circle_C x2 y2 ∧ 
  (x1 ≠ x2 ∧ y1 ≠ y2) → 
  k < -3/4 ∧
  ( (y1 / x1) + (y2 / x2) = 1 ) :=
by
  sorry

end circle_and_line_properties_l380_380197


namespace students_representing_x_percent_l380_380382

def total_students : ℝ := 122.47448713915891
def percentage_of_boys : ℝ := 0.60
def number_of_boys : ℝ := total_students * percentage_of_boys
def x_percentage_of_boys (x : ℝ) : ℝ := (x / 100) * number_of_boys

theorem students_representing_x_percent (x : ℝ) :
  x_percentage_of_boys x = (x / 100) * 73.48469228349535 := 
by 
  unfold x_percentage_of_boys number_of_boys
  rw [number_of_boys]
  simp
  sorry

end students_representing_x_percent_l380_380382


namespace total_distance_traveled_l380_380835

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end total_distance_traveled_l380_380835


namespace task_inequality_l380_380671

-- Define the conditions from the problem
noncomputable def polynomial {α : Type*} [Nonnegative α] (a : list α) (x : α) : α :=
  x ^ a.length - list.sum (list.map_with_index (λ i, λ ai, ai * (x ^ (a.length - 1 - i))) a)

-- Define the sums A and B
def sum_a {α : Type*} [Add α] (a : list α) : α :=
  list.sum a

def sum_ia {α : Type*} [HasScalar ℕ α] [Add α] (a : list α) : α :=
  list.sum (list.map_with_index (λ i ai, (i + 1) • ai) a)

-- Define the unique positive real root R
axiom unique_positive_root (a : list ℝ) (h : ∃ i, a.nth i ≠ 0) :
  ∃! R > 0, polynomial a R = 0

-- Prove the desired inequality A^A <= R^B
theorem task_inequality (a : list ℝ) (h : ∃ i, a.nth i ≠ 0) :
  let R := classical.some (unique_positive_root a h),
      A := sum_a a,
      B := sum_ia a in
  A ^ A ≤ R ^ B := by
  sorry

end task_inequality_l380_380671


namespace find_x_squared_exists_sin_arctan_eq_inv_l380_380755

theorem find_x_squared_exists_sin_arctan_eq_inv (x : ℝ) (h : x > 0) : sin (arctan x) = 1 / x → x^2 = 1 :=
by
  intro h_sin_arctan
  sorry

end find_x_squared_exists_sin_arctan_eq_inv_l380_380755


namespace speed_of_man_proof_l380_380088

noncomputable def speed_of_man (train_length : ℝ) (crossing_time : ℝ) (train_speed_kph : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kph * 1000) / 3600
  let relative_speed := train_length / crossing_time
  train_speed_mps - relative_speed

theorem speed_of_man_proof 
  (train_length : ℝ := 600) 
  (crossing_time : ℝ := 35.99712023038157) 
  (train_speed_kph : ℝ := 64) :
  speed_of_man train_length crossing_time train_speed_kph = 1.10977777777778 :=
by
  -- Proof goes here
  sorry

end speed_of_man_proof_l380_380088


namespace average_weight_of_boys_l380_380649

theorem average_weight_of_boys
  (average_weight_girls : ℕ) 
  (average_weight_students : ℕ) 
  (h_girls : average_weight_girls = 45)
  (h_students : average_weight_students = 50) : 
  ∃ average_weight_boys : ℕ, average_weight_boys = 55 :=
by
  sorry

end average_weight_of_boys_l380_380649


namespace find_y_l380_380877

noncomputable def log_x (a b c p q r x : ℝ) := log a / p = log b / q ∧ log b / q = log c / r ∧ log c / r = log x

theorem find_y (a b c p q r x y : ℝ) (h1 : log_x a b c p q r x) (h2 : x ≠ 1) (h3 : b^3 / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := by
  sorry

end find_y_l380_380877


namespace distribution_1_distribution_2_distribution_3_l380_380752

-- 1. Defining the problem statement for the first condition.
theorem distribution_1 : nat.choose 6 2 * nat.choose 4 2 * 1 = 90 := 
sorry

-- 2. Defining the problem statement for the second condition.
theorem distribution_2 : nat.choose 6 1 * nat.choose 5 2 * 1 = 60 := 
sorry

-- 3. Defining the problem statement for the third condition.
theorem distribution_3 : nat.choose 6 1 * nat.choose 5 2 * 1 * nat.factorial 3 = 360 := 
sorry

end distribution_1_distribution_2_distribution_3_l380_380752


namespace trader_gain_percentage_l380_380837

theorem trader_gain_percentage (C : ℝ) (h1 : 95 * C = (95 * C - cost_of_95_pens) + (19 * C)) :
  100 * (19 * C / (95 * C)) = 20 := 
by {
  sorry
}

end trader_gain_percentage_l380_380837


namespace general_pattern_l380_380313

theorem general_pattern (n : ℕ) : 
  ∑ i in Finset.range (n+1), (-1)^i * (2*i - 1) = (-1)^n * n := 
by
  induction n with k ih
  case zero => 
    -- Base case
    sorry
  case succ =>
    -- Inductive step
    sorry

end general_pattern_l380_380313


namespace jogging_lcm_l380_380514

theorem jogging_lcm (David_time Maria_time Leo_time : ℕ) 
  (hDavid : David_time = 5) 
  (hMaria : Maria_time = 8) 
  (hLeo : Leo_time = 10) : 
  let lcm := Nat.lcm (Nat.lcm David_time Maria_time) Leo_time in
  lcm = 40 ∧ 9 * 60 + 40 = 580 :=
by
  have h1 : David_time = 5 := by rw [hDavid]
  have h2 : Maria_time = 8 := by rw [hMaria]
  have h3 : Leo_time = 10 := by rw [hLeo]
  have h_lcm_D_M : Nat.lcm David_time Maria_time = 40 :=
      by
        rw [h1, h2]
        exact Nat.lcm_eq 5 8 2 sorry sorry sorry -- Prime factorizations and lcm steps assertion
  have h_lcm : lcm = 40 :=
      by
        rw [h3]
        exact nat_lcm_eq 40 10 2 sorry sorry sorry -- Prime factorizations and lcm steps assertion
  have h_time : 9 * 60 + 40 = 580 := rfl
  exact ⟨h_lcm, h_time⟩
  
#eval jogging_lcm 5 8 10 rfl rfl rfl

end jogging_lcm_l380_380514


namespace digit_150th_l380_380058

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380058


namespace length_of_ellipse_l380_380177

noncomputable def parametric_curve_length : ℝ :=
  ∫ t in 0..(2 * Real.pi), sqrt ((-3 * Real.sin t) ^ 2 + (Real.cos t) ^ 2)

theorem length_of_ellipse :
  abs (parametric_curve_length - 16.2) < 0.1 :=
sorry

end length_of_ellipse_l380_380177


namespace chord_length_and_lambda_l380_380215

noncomputable def vector_magnitude_min (λ : ℝ) (A B : ℝ × ℝ) : ℝ :=
  let magnitude := λ (A B : ℝ × ℝ), real.sqrt ((A.1 - λ * B.1)^2 + (A.2 - λ * B.2)^2)
  real.min (magnitude (1, 0) (real.cos θ, real.sin θ)) (magnitude (1, 0) (real.cos $ 2 * real.pi - θ, real.sin $ 2 * real.pi - θ))

theorem chord_length_and_lambda
  (A B : ℝ × ℝ)
  (θ : ℝ)
  (hθ : 0 ≤ θ ∧ θ < 2 * real.pi)
  (h₁ : |(A.1, A.2) - λ ⬝ (B.1, B.2)| = real.sqrt 3 / 2)
  (h₂ : A = (1, 0)) :
  ∃ AB_len ∈ {1, real.sqrt 3}, λ = ± (1 / 2) :=
by
  sorry

end chord_length_and_lambda_l380_380215


namespace proportional_function_y_decreases_l380_380627

theorem proportional_function_y_decreases (k : ℝ) (h₀ : k ≠ 0) (h₁ : (4 : ℝ) * k = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ :=
by 
  sorry

end proportional_function_y_decreases_l380_380627


namespace num_divisors_perfect_squares_or_cubes_l380_380942

theorem num_divisors_perfect_squares_or_cubes (n : ℕ) (h : n = 180 ^ 8) :
  number_of_divisors_perfect_squares_or_cubes n = 495 :=
by
  -- Assuming the definition and conditions from the problem
  let a := 2
  let b := 2
  let c := 1
  let m := 8
  have h1: 180 = 2^a * 3^b * 5^c := by norm_num
  have h2: 180^m = 2^(m * a) * 3^(m * b) * 5^(m * c) := by simp [pow_mul]
  -- Another step to ensure Lean can understand the simplified variables and proof steps
  sorry

end num_divisors_perfect_squares_or_cubes_l380_380942


namespace ratio_AB_PQ_f_half_func_f_l380_380656

-- Define given conditions
variables {m n : ℝ} -- Lengths of AB and PQ
variables {h : ℝ} -- Height of triangle and rectangle (both are 1)
variables {x : ℝ} -- Variable in the range [0, 1]

-- Same area and height conditions
axiom areas_equal : m / 2 = n
axiom height_equal : h = 1

-- Given the areas are equal and height is 1
theorem ratio_AB_PQ : m / n = 2 :=
by sorry -- Proof of the ratio 

-- Given the specific calculation for x = 1/2
theorem f_half (hx : x = 1 / 2) (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  f (1 / 2) = 3 / 4 :=
by sorry -- Proof of function value at 1/2

-- Prove the expression of the function f(x)
theorem func_f (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  ∀ x, 0 ≤ x → x ≤ 1 → f x = 2 * x - x^2 :=
by sorry -- Proof of the function expression


end ratio_AB_PQ_f_half_func_f_l380_380656


namespace polynomial_divisibility_p_q_l380_380605

theorem polynomial_divisibility_p_q (p' q' : ℝ) :
  (∀ x, x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0 → (x = -1 ∨ x = 2)) →
  p' = 0 ∧ q' = -9 :=
by sorry

end polynomial_divisibility_p_q_l380_380605


namespace correct_statements_l380_380589

def reasoning_statement_1 := "Inductive reasoning is reasoning from the specific to the general."
def reasoning_statement_2 := "Inductive reasoning is reasoning from the general to the general."
def reasoning_statement_3 := "Deductive reasoning is reasoning from the general to the specific."
def reasoning_statement_4 := "Analogical reasoning is reasoning from the specific to the general."
def reasoning_statement_5 := "Analogical reasoning is reasoning from the specific to specific."

def inductive_reasoning := "Inductive reasoning involves deriving general conclusions from specific observations."
def deductive_reasoning := "Deductive reasoning moves from general principles to deduce specific cases."
def analogical_reasoning := "Analogical reasoning involves inferring that if two or more objects or classes of objects share some attributes, they likely share other attributes as well."

theorem correct_statements :
  (reasoning_statement_1 = true ∧
   reasoning_statement_3 = true ∧
   reasoning_statement_5 = true) ∧
  (reasoning_statement_2 = false ∧
   reasoning_statement_4 = false) := by
  sorry

end correct_statements_l380_380589


namespace right_triangle_angles_l380_380643

theorem right_triangle_angles (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_order : a ≤ b) (h_geom_mean : a^2 = (c - a) * (b - a)) :
  let α := real.arccos (a / c),
      β := real.arcsin (b / c)
  in α * (180 / real.pi) = 27.96 ∧ β * (180 / real.pi) = 62.04 ∧ α + β = real.pi / 2 :=
by
  sorry

end right_triangle_angles_l380_380643


namespace cubes_with_even_red_faces_l380_380108

theorem cubes_with_even_red_faces :
  let length := 6
  let width := 4
  let height := 2
  let painted_faces : Finset (ℕ × ℕ × ℕ) :=
    {(i, j, 0) | i in {0, 1, 2, 3, 4, 5}, j in {0, 1, 2, 3}} ∪
    {(0, j, k) | j in {0, 1, 2, 3}, k in {0, 1}}
  let cubes : Finset (ℕ × ℕ × ℕ) :=
    { (i, j, k) | i in {0, 1, 2, 3, 4, 5}, j in {0, 1, 2, 3}, k in {0, 1} }
  (∃ (even_red_faces_cubes : Finset (ℕ × ℕ × ℕ)),
  even_red_faces_cubes = {c∈ cubes | (c.1, c.2, c.3) ∈ painted_faces ∧ (c.1, c.2, c.3) else 0}
  |
  even_red_faces_cubes.card = 16) :=
sorry

end cubes_with_even_red_faces_l380_380108


namespace digit_150_of_17_div_70_is_2_l380_380014

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380014


namespace monotonicity_of_f_inequality_f_h_l380_380597

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * log x + 11

-- Define the function h(x)
def h (x : ℝ) : ℝ := -x^3 + 3 * x^2 + (3 - x) * exp x

-- Lean statement for monotonicity of f(x)
theorem monotonicity_of_f : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → ∀ y : ℝ, 0 < y ∧ y < 1 → x < y → f x > f y) ∧
  (∀ x : ℝ, x > 1 → ∀ y : ℝ, y > 1 → x < y → f x < f y) :=
sorry

-- Lean statement for proving the inequality for x > 0
theorem inequality_f_h (x : ℝ) (h1 : x > 0) : f x > h x :=
sorry

end monotonicity_of_f_inequality_f_h_l380_380597


namespace sum_of_first_four_super_nice_eq_45_l380_380804

def is_super_nice (n : ℕ) : Prop :=
  (∀ d ∈ (nat.divisors n).filter (λ x, x ≠ n),
    ∃ p q : ℕ, p ≠ q ∧ n = p * q ∧ (d ∈ {p, q})) ∧
  (∀ d ∈ (nat.divisors n).filter (λ x, x ≠ n),
    ∃ p q : ℕ, p ≠ q ∧ n = p * q ∧ (nat.divisors_antidiagonal n).filter (λ (x, y), x + y = n).card = 2) 

def sum_of_first_n_super_nice (n : ℕ) : ℕ :=
  ((nat.filter (λ x, is_super_nice x) (finset.range (nat.succ 100))).take n).sum

theorem sum_of_first_four_super_nice_eq_45 : sum_of_first_n_super_nice 4 = 45 := 
  by 
    -- The proof steps would be added here, but are omitted in this statement.
    sorry

end sum_of_first_four_super_nice_eq_45_l380_380804


namespace find_x_y_l380_380557

theorem find_x_y (x y : ℝ) : 
  let a := (1, 2, -y)
  let b := (x, 1, 2)
  2 • b = ((2 : ℝ) * x, 2, 4) →
  (a - b) = (1 - x, 1, -y - 2) →
  ∃ x y : ℝ, 
    x = 1 / 2 ∧ y = -4 :=
begin
  -- variables definition
  intros a_def b_def,
  use [1 / 2, -4],
  split,
  { sorry },
  { sorry }
end

end find_x_y_l380_380557


namespace rahul_share_l380_380435

theorem rahul_share (work_rate_rahul: ℝ) (work_rate_rajesh: ℝ) (total_payment: ℝ) : ℝ :=
  let combined_work_rate := work_rate_rahul + work_rate_rajesh
  let rahul_share_ratio := work_rate_rahul / combined_work_rate
  total_payment * rahul_share_ratio

def rahul_work_rate := 1 / 3
def rajesh_work_rate := 1 / 2
def total_payment := 170
def expected_rahul_share := 68

example : rahul_share rahul_work_rate rajesh_work_rate total_payment = expected_rahul_share := 
by
  unfold rahul_share
  simp [rahul_work_rate, rajesh_work_rate, total_payment, expected_rahul_share]
  rw [add_div, one_div_mul, mul_div_cancel_left, div_div]
  norm_num
  -- The correct answer is $68
  sorry

end rahul_share_l380_380435


namespace line_intercepts_l380_380738

theorem line_intercepts :
  (exists a b : ℝ, (forall x y : ℝ, x - 2*y - 2 = 0 ↔ (x = 2 ∨ y = -1)) ∧ a = 2 ∧ b = -1) :=
by
  sorry

end line_intercepts_l380_380738


namespace range_of_a_l380_380229

def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + x else x - x^2

theorem range_of_a (a : ℝ) (h : f a > f (2 - a)) : a > 1 :=
by sorry

end range_of_a_l380_380229


namespace volume_and_surface_area_of_prism_l380_380422

theorem volume_and_surface_area_of_prism 
  (a b c : ℝ)
  (h1 : a * b = 24)
  (h2 : b * c = 18)
  (h3 : c * a = 12) :
  (a * b * c = 72) ∧ (2 * (a * b + b * c + c * a) = 108) := by
  sorry

end volume_and_surface_area_of_prism_l380_380422


namespace sum_equation_l380_380508

noncomputable def compute_sum : ℝ :=
  ∑' (a : ℕ) in {a | 1 ≤ a}.to_finset, ∑' (b : ℕ) in {b | a < b}.to_finset, 
  ∑' (c : ℕ) in {c | b < c}.to_finset, ∑' (d : ℕ) in {d | c < d}.to_finset, 
  (1 / (3 ^ a * 4 ^ b * 6 ^ c * 8 ^ d : ℝ))

theorem sum_equation : compute_sum = 1 / 210 := by sorry

end sum_equation_l380_380508


namespace appropriate_sampling_method_l380_380452

noncomputable def companies_outlets := {A := 150, B := 120, C := 180, D := 150}

noncomputable def large_outlets_in_C := 20

def total_survey_outlets := 7

theorem appropriate_sampling_method :
  is_appropriate_sampling companies_outlets large_outlets_in_C total_survey_outlets "Simple random sampling" := 
sorry

end appropriate_sampling_method_l380_380452


namespace andy_starting_problem_l380_380490

theorem andy_starting_problem (end_num problems_solved : ℕ) 
  (h_end : end_num = 125) (h_solved : problems_solved = 46) : 
  end_num - problems_solved + 1 = 80 := 
by
  sorry

end andy_starting_problem_l380_380490


namespace geometric_description_of_S_l380_380616

open Complex

noncomputable def S : Set ℂ := {z : ℂ | ∃ x y : ℝ, z = x + y * I ∧ (2 + 5 * I) * z ∈ ℝ}

theorem geometric_description_of_S :
  ∃ m : ℝ, S = {z : ℂ | ∃ y : ℝ, z = -((2:ℝ)/(5:ℝ)) * y + y * I} :=
by
  sorry

end geometric_description_of_S_l380_380616


namespace repeating_decimal_sum_l380_380859

theorem repeating_decimal_sum :
  let x := (0.3333333333333333 : ℚ) -- 0.\overline{3}
  let y := (0.0707070707070707 : ℚ) -- 0.\overline{07}
  let z := (0.008008008008008 : ℚ)  -- 0.\overline{008}
  x + y + z = 418 / 999 := by
sorry

end repeating_decimal_sum_l380_380859


namespace product_of_numerator_and_denominator_l380_380773

-- Defining the repeating decimal as a fraction in lowest terms
def repeating_decimal_as_fraction_in_lowest_terms : ℚ :=
  1 / 37

-- Theorem to prove the product of the numerator and the denominator
theorem product_of_numerator_and_denominator :
  (repeating_decimal_as_fraction_in_lowest_terms.num.natAbs *
   repeating_decimal_as_fraction_in_lowest_terms.den) = 37 :=
by
  -- declaration of the needed fact and its direct consequence
  sorry

end product_of_numerator_and_denominator_l380_380773


namespace count_prime_sum_of_divisors_l380_380306

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n > 1 ∧ n < p → p % n ≠ 0

def sum_of_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).sum id

theorem count_prime_sum_of_divisors :
  (finset.filter
    (λ n, is_prime (sum_of_divisors n))
    (finset.range 31)).card = 8 :=
by sorry

end count_prime_sum_of_divisors_l380_380306


namespace parity_of_f_max_value_of_f_l380_380916

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log2 (2^x + 1)

-- Statement 1: If a = -1/2, then f(x) is an even function
theorem parity_of_f (a : ℝ) : a = -1/2 → ∀ x : ℝ, f a x = f a (-x) := sorry

-- Statement 2: Given a > 0 and the minimum value condition of y = f(x) + f⁻¹(x),
-- show that a = 1 and the maximum value of f(x) on [1,2] is 2 + log_2 5
theorem max_value_of_f (a : ℝ) (h : a > 0)
  (h_min : ∀ x ∈ Icc 1 2, f a x + invFun (f a) x = 1 + Real.log2 3) :
  a = 1 ∧ ∀ x ∈ Icc 1 2, f a 2 = 2 + Real.log2 5 := sorry

end parity_of_f_max_value_of_f_l380_380916


namespace concave_function_m_range_l380_380687

-- Definitions
def f (x : ℝ) : ℝ := (1/20) * x^5 - (1/12) * m * x^4 - 2 * x^2
def f' (x : ℝ) : ℝ := (1/4) * x^4 - (1/3) * m * x^3 - 4 * x
def f'' (x : ℝ) : ℝ := x^3 - m * x^2 - 4

-- Conditions
variables {a b : ℝ} (h1 : a = 1) (h2 : b = 3) (m : ℝ)
  (h3 : ∀ x, 1 < x → x < 3 → f'' x > 0)

-- Theorem statement
theorem concave_function_m_range : m ≤ -3 :=
by
  sorry

end concave_function_m_range_l380_380687


namespace monthly_rent_per_resident_is_correct_l380_380692

-- Defining the total number of units and the occupancy rate.
def total_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4

-- Given the total rent collected in a year.
def annual_rent_collected : ℚ := 360000

-- We need to prove that the monthly rent per resident is $400.
theorem monthly_rent_per_resident_is_correct :
  let occupied_units := total_units * occupancy_rate in
  let annual_rent_per_resident := annual_rent_collected / occupied_units in
  let monthly_rent_per_resident := annual_rent_per_resident / 12 in
  monthly_rent_per_resident = 400 :=
by
  sorry

end monthly_rent_per_resident_is_correct_l380_380692


namespace pages_for_thirty_dollars_l380_380663

-- Problem Statement Definitions
def costPerCopy := 4 -- cents
def pagesPerCopy := 2 -- pages
def totalCents := 3000 -- cents
def totalPages := 1500 -- pages

-- Theorem: Calculating the number of pages for a given cost.
theorem pages_for_thirty_dollars (c_per_copy : ℕ) (p_per_copy : ℕ) (t_cents : ℕ) (t_pages : ℕ) : 
  c_per_copy = 4 → p_per_copy = 2 → t_cents = 3000 → t_pages = 1500 := by
  intros h_cpc h_ppc h_tc
  sorry

end pages_for_thirty_dollars_l380_380663


namespace chameleon_increase_l380_380354

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380354


namespace cyrus_pages_proof_l380_380150

def pages_remaining (total_pages: ℝ) (day1: ℝ) (day2: ℝ) (day3: ℝ) (day4: ℝ) (day5: ℝ) : ℝ :=
  total_pages - (day1 + day2 + day3 + day4 + day5)

theorem cyrus_pages_proof :
  let total_pages := 750
  let day1 := 30
  let day2 := 1.5 * day1
  let day3 := day2 / 2
  let day4 := 2.5 * day3
  let day5 := 15
  pages_remaining total_pages day1 day2 day3 day4 day5 = 581.25 :=
by 
  sorry

end cyrus_pages_proof_l380_380150


namespace combined_cost_of_stocks_l380_380768

variable (face_value_A : ℝ) (discount_A : ℝ) (brokerage_A : ℝ)
variable (face_value_B : ℝ) (premium_B : ℝ) (brokerage_B : ℝ)
variable (face_value_C : ℝ) (discount_C : ℝ) (brokerage_C : ℝ)

def cost_after_discount (face_value : ℝ) (discount : ℝ) : ℝ :=
  face_value - (discount / 100 * face_value)

def cost_after_premium (face_value : ℝ) (premium : ℝ) : ℝ :=
  face_value + (premium / 100 * face_value)

def brokerage_cost (face_value_after_adjustment : ℝ) (brokerage : ℝ) : ℝ :=
  brokerage / 100 * face_value_after_adjustment

def total_cost (face_value : ℝ) (adjustment : ℝ) (brokerage : ℝ) (is_discount : Bool) : ℝ :=
  let adjusted_cost := if is_discount then cost_after_discount face_value adjustment
                       else cost_after_premium face_value adjustment
  adjusted_cost + brokerage_cost adjusted_cost brokerage

theorem combined_cost_of_stocks : 
  total_cost 100 2 0.2 true
  + total_cost 150 1.5 0.1667 false
  + total_cost 200 3 0.5 true
  = 445.67 := by
  sorry

end combined_cost_of_stocks_l380_380768


namespace range_of_m_l380_380596

theorem range_of_m (m : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
  (m < 1) ∧ (0 < m → m < 1) :=
begin
  have h1 : ∀ x : ℝ, f x = x^3 + x, 
  from λ x, rfl,
  -- Assuming f(x) is odd and increasing function
  have h2 : ∀ x y : ℝ, (x < y) → (f x < f y), 
  -- by odd and increasing nature of f
  from sorry, 
  have h3 : 0 ≤ sin θ ∧ sin θ ≤ 1,
  from sorry,
  -- given the inequality
  assume h4 : f (m * sin θ) + f (1 - m) > 0,
  have h5 : f (m * sin θ) > f (m - 1),
  from sorry,
  -- deduce m * sin θ > m - 1
  have h6 : 0 ≤ sin θ,
  from sorry,
  have h7 : sin θ ≤ 1,
  from sorry,
  -- solve to find m < 1
  sorry
end

end range_of_m_l380_380596


namespace consecutive_odd_integers_l380_380754

theorem consecutive_odd_integers (n : ℕ) (h1 : ∑ i in finset.range(n-1).succ, (407 + 2 * (i : ℕ)) = 414 * n) : n = 8 := 
sorry

end consecutive_odd_integers_l380_380754


namespace digit_150th_in_decimal_of_fraction_l380_380043

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380043


namespace addition_of_two_negatives_l380_380504

theorem addition_of_two_negatives (a b : ℤ) (ha : a < 0) (hb : b < 0) : a + b < a ∧ a + b < b :=
by
  sorry

end addition_of_two_negatives_l380_380504


namespace num_4digit_numbers_divisible_by_5_last_digits_45_l380_380615

theorem num_4digit_numbers_divisible_by_5_last_digits_45 : 
  ∃ (n : ℕ), n = 90 ∧ 
  (∀ (a b : ℕ), 
    (a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
    (∃ (digit_number : ℕ), digit_number = a * 1000 + b * 100 + 45) ∧ 
    1000 ≤ a * 1000 + b * 100 + 45 ∧ a * 1000 + b * 100 + 45 < 10000) :=
by
  use 90
  split
  { sorry }
  { intros a b hab
    cases hab with ha hb
    use a * 1000 + b * 100 + 45
    split
    { rfl }
    split
    { sorry }
    { sorry }
  }

end num_4digit_numbers_divisible_by_5_last_digits_45_l380_380615


namespace speed_difference_is_3_l380_380445

-- Definitions from conditions
def bike_distance : ℝ := 136
def bike_time : ℝ := 8
def truck_distance : ℝ := 112
def truck_time : ℝ := 8

-- Definitions of speed
def bike_speed : ℝ := bike_distance / bike_time
def truck_speed : ℝ := truck_distance / truck_time

-- The theorem to prove the difference in speed
theorem speed_difference_is_3 : bike_speed - truck_speed = 3 := by
  sorry

end speed_difference_is_3_l380_380445


namespace chameleon_problem_l380_380362

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380362


namespace amplitude_of_sinusoidal_function_l380_380496

theorem amplitude_of_sinusoidal_function 
  (a b c d : ℝ) 
  (h1 : 0 < a)
  (h2 : ∀ x, -3 ≤ a * real.sin (b * x + c) + d ∧ a * real.sin (b * x + c) + d ≤ 5) :
  a = 4 :=
sorry

end amplitude_of_sinusoidal_function_l380_380496


namespace larger_solution_of_quadratic_equation_l380_380543

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l380_380543


namespace students_per_class_l380_380841

theorem students_per_class
  (cards_per_student : Nat)
  (periods_per_day : Nat)
  (cost_per_pack : Nat)
  (total_spent : Nat)
  (cards_per_pack : Nat)
  (students_per_class : Nat)
  (H1 : cards_per_student = 10)
  (H2 : periods_per_day = 6)
  (H3 : cost_per_pack = 3)
  (H4 : total_spent = 108)
  (H5 : cards_per_pack = 50)
  (H6 : students_per_class = 30)
  :
  students_per_class = (total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day) :=
sorry

end students_per_class_l380_380841


namespace alpha_less_than_60_degrees_l380_380442

theorem alpha_less_than_60_degrees
  (R r : ℝ)
  (b c : ℝ)
  (α : ℝ)
  (h1 : b * c = 8 * R * r) :
  α < 60 := sorry

end alpha_less_than_60_degrees_l380_380442


namespace fabric_per_pair_of_pants_l380_380670

theorem fabric_per_pair_of_pants 
  (jenson_shirts_per_day : ℕ)
  (kingsley_pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric_needed : ℕ)
  (days : ℕ)
  (fabric_per_pant : ℕ) :
  jenson_shirts_per_day = 3 →
  kingsley_pants_per_day = 5 →
  fabric_per_shirt = 2 →
  total_fabric_needed = 93 →
  days = 3 →
  fabric_per_pant = 5 :=
by sorry

end fabric_per_pair_of_pants_l380_380670


namespace exists_rectangle_enclosing_convex_polygon_l380_380366

theorem exists_rectangle_enclosing_convex_polygon (P : Set (ℝ × ℝ)) (hP : convex P) (h_area_P : area P = 1) :
  ∃ R : Set (ℝ × ℝ), is_rectangle R ∧ encloses R P ∧ area R ≤ 2 :=
sorry

end exists_rectangle_enclosing_convex_polygon_l380_380366


namespace candles_left_in_room_l380_380823

-- Define the variables and conditions
def total_candles : ℕ := 40
def alyssa_used : ℕ := total_candles / 2
def remaining_candles_after_alyssa : ℕ := total_candles - alyssa_used
def chelsea_used : ℕ := (7 * remaining_candles_after_alyssa) / 10
def final_remaining_candles : ℕ := remaining_candles_after_alyssa - chelsea_used

-- The theorem we need to prove
theorem candles_left_in_room : final_remaining_candles = 6 := by
  sorry

end candles_left_in_room_l380_380823


namespace find_c_l380_380232

theorem find_c (a c : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + c) 
  (h2 : f' = λ x, 3 * a * x^2) 
  (h3 : f' 1 = 6) 
  (h4 : ∀ x ∈ Set.Icc 1 2, f x ≤ 20) 
  (h5 : ∃ x ∈ Set.Icc 1 2, f x = 20) : 
  c = 4 := 
by {
  sorry
}

end find_c_l380_380232


namespace digit_150th_l380_380053

-- Define the fraction and its repeating decimal properties
def fraction := 17 / 70
def repeating_block := "242857"
def block_length : ℕ := 6

-- The 150th digit calculation
def digit_position : ℕ := 150 % block_length

theorem digit_150th : "150th digit of decimal representation" = 7 :=
by {
  have h1 : fraction = 0.242857142857, sorry,
  have h2 : "252857".cycle = repeating_block, sorry,
  have h3 : digit_position = 0, sorry,
  have h4 : repeating_block.get_last = '7', sorry,
  show "150th digit of decimal representation"= 7, sorry,
}

end digit_150th_l380_380053


namespace repeating_decimal_to_fraction_l380_380161

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end repeating_decimal_to_fraction_l380_380161


namespace bricks_required_l380_380076

noncomputable def courtyard_length : ℕ := 18
noncomputable def courtyard_width : ℕ := 12
noncomputable def brick_length_cm : ℕ := 12
noncomputable def brick_width_cm : ℕ := 6

theorem bricks_required 
  (courtyard_length : ℕ)
  (courtyard_width : ℕ)
  (brick_length_cm : ℕ)
  (brick_width_cm : ℕ) : 
  (courtyard_length * courtyard_width * 10000) / (brick_length_cm * brick_width_cm) = 30000 := 
by 
  -- definitions
  have area_courtyard := courtyard_length * courtyard_width * 10000
  have area_brick := brick_length_cm * brick_width_cm
  
  -- result
  have total_bricks := area_courtyard / area_brick
  exact total_bricks = 30000

end bricks_required_l380_380076


namespace centroid_lines_perpendicular_l380_380090

variables {A B C D W X Y Z S1 S2 S3 S4 : Type}
variables [ConvexQuadrilateral A B C D] [EquilateralTriangle W A B] [EquilateralTriangle X B C] [EquilateralTriangle Y C D] [EquilateralTriangle Z D A]
  (AC_eq_BD : dist (A C) = dist (B D))
  (S1_centroid : Centroid W = S1) (S2_centroid : Centroid X = S2) (S3_centroid : Centroid Y = S3) (S4_centroid : Centroid Z = S4)

theorem centroid_lines_perpendicular :
  dist S1 S3 ⟂ dist S2 S4 :=
sorry

end centroid_lines_perpendicular_l380_380090


namespace parabola_ellipse_focus_l380_380238
noncomputable def p_value : ℝ :=
  let e := ellipse 6 2 in
  let p_pos := parabola_focus 2 in
  if p_pos.1 = e.right_focus.1 then p_pos.2 else 0

theorem parabola_ellipse_focus (p : ℝ) (hp : p > 0) 
  (he : (ellipse_right_focus 6 2).1 = (parabola_focus p).1) : 
  p = 4 := by
  sorry

end parabola_ellipse_focus_l380_380238


namespace digit_150_is_7_l380_380035

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380035


namespace digit_150_of_17_div_70_is_2_l380_380017

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380017


namespace not_unique_right_angle_triangle_unique_right_angle_triangle_hypotenuse_angle_unique_right_angle_triangle_leg_angle_unique_right_angle_triangle_hypotenuse_leg_l380_380764

theorem not_unique_right_angle_triangle (α β : ℝ) (C : Type) [euclidean_space C] 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : α + β < π / 2) :
  ∃ (Δ1 Δ2 : triangle C), (Δ1 ≠ Δ2 ∧ acute_angle Δ1 = α ∧ acute_angle Δ2 = β) := 
sorry

theorem unique_right_angle_triangle_hypotenuse_angle {C : Type} [euclidean_space C]
  (a b : ℝ) (Δ : triangle C) 
  (h : Δ.hypotenuse = a ∧ Δ.angle = b ∧ 0 < a ∧ 0 < b ∧ b < π / 2) :
  unique_right_angled_triangle a b :=
sorry

theorem unique_right_angle_triangle_leg_angle {C : Type} [euclidean_space C]
  (a b : ℝ) (Δ : triangle C)
  (h : Δ.leg = a ∧ Δ.angle = b ∧ 0 < a ∧ 0 < b ∧ b < π / 2) :
  unique_right_angled_triangle a b :=
sorry

theorem unique_right_angle_triangle_hypotenuse_leg {C : Type} [euclidean_space C]
  (a b : ℝ) (Δ : triangle C)
  (h : Δ.hypotenuse = a ∧ Δ.leg = b ∧ 0 < a ∧ 0 < b) :
  unique_right_angled_triangle a b :=
sorry

end not_unique_right_angle_triangle_unique_right_angle_triangle_hypotenuse_angle_unique_right_angle_triangle_leg_angle_unique_right_angle_triangle_hypotenuse_leg_l380_380764


namespace opposite_direction_l380_380723

variable (movement_east : Int)

/-- 
Moving 5 meters to the east is denoted as +5 meters.
Negative numbers have the same quantity as their positive counterparts but opposite meanings.
-/
theorem opposite_direction (h : movement_east = 5) : -movement_east = -5 := 
by 
  -- translating the given steps
  rw [h]
  refl

end opposite_direction_l380_380723


namespace equation_of_perpendicular_line_l380_380728

theorem equation_of_perpendicular_line (x y c : ℝ) (h₁ : x = -1) (h₂ : y = 2)
  (h₃ : 2 * x - 3 * y = -c) (h₄ : 3 * x + 2 * y - 7 = 0) :
  2 * x - 3 * y + 8 = 0 :=
sorry

end equation_of_perpendicular_line_l380_380728


namespace max_experiments_fibonacci_search_l380_380976

-- Define the conditions and the theorem
def is_unimodal (f : ℕ → ℕ) : Prop :=
  ∃ k, ∀ n m, (n < k ∧ k ≤ m) → f n < f k ∧ f k > f m

def fibonacci_search_experiments (n : ℕ) : ℕ :=
  -- Placeholder function representing the steps of Fibonacci search
  if n <= 1 then n else fibonacci_search_experiments (n - 1) + fibonacci_search_experiments (n - 2)

theorem max_experiments_fibonacci_search (f : ℕ → ℕ) (n : ℕ) (hn : n = 33) (hf : is_unimodal f) : fibonacci_search_experiments n ≤ 7 :=
  sorry

end max_experiments_fibonacci_search_l380_380976


namespace quadratic_behavior_l380_380190

theorem quadratic_behavior (x : ℝ) : x < 3 → ∃ y : ℝ, y = 5 * (x - 3) ^ 2 + 2 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 3 ∧ x2 < 3 → (5 * (x1 - 3) ^ 2 + 2) > (5 * (x2 - 3) ^ 2 + 2) := 
by
  sorry

end quadratic_behavior_l380_380190


namespace chameleon_problem_l380_380358

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380358


namespace only_correct_option_is_C_l380_380070

-- Definitions of the conditions as per the given problem
def option_A (a : ℝ) : Prop := a^2 * a^3 = a^6
def option_B (a : ℝ) : Prop := (a^2)^3 = a^5
def option_C (a b : ℝ) : Prop := (a * b)^3 = a^3 * b^3
def option_D (a : ℝ) : Prop := a^8 / a^2 = a^4

-- The theorem stating that only option C is correct
theorem only_correct_option_is_C (a b : ℝ) : 
  ¬(option_A a) ∧ ¬(option_B a) ∧ option_C a b ∧ ¬(option_D a) :=
by sorry

end only_correct_option_is_C_l380_380070


namespace factor_difference_of_squares_l380_380170

theorem factor_difference_of_squares (y : ℝ) : 81 - 16 * y^2 = (9 - 4 * y) * (9 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l380_380170


namespace probability_non_perfect_power_equals_frac_l380_380403

noncomputable def non_perfect_power_probability : ℚ :=
  let is_perfect_power (n : ℕ) : Prop :=
    ∃ x y : ℕ, x > 0 ∧ y > 1 ∧ n = x ^ y
  let count_non_perfect_powers :=
    (Finset.range 201).filter (λ n, ¬ is_perfect_power n).card
  count_non_perfect_powers / 200

theorem probability_non_perfect_power_equals_frac :
  non_perfect_power_probability = 181 / 200 := 
  sorry

end probability_non_perfect_power_equals_frac_l380_380403


namespace problem_solution_l380_380702

open Real

variable (a : ℝ)

def p (x : ℝ) : Prop := x^2 + a*x + a^2 ≥ 0

def q : Prop := ∃ x : ℝ, sin x + cos x = 2

theorem problem_solution : (∀ x : ℝ, p a x) ∨ q → (∀ x : ℝ, p a x) := 
begin
  intro h,
  cases h,
  { apply h },
  { exfalso,
    obtain ⟨x, hx⟩ := h,
    have h1 := abs_sin_le_one x,
    have h2 := abs_cos_le_one x,
    linarith [(hx.symm ▸ (sin_add_cos_le_sqrt_two x))],
  }
end

end problem_solution_l380_380702


namespace larger_of_two_solutions_l380_380537

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l380_380537


namespace tan_4050_undefined_l380_380137

noncomputable def tan_is_undefined (θ : ℝ) : Prop := Real.cos θ = 0

theorem tan_4050_undefined : tan_is_undefined (4050 * Real.pi / 180) :=
by
  -- Convert degrees to radians
  have : (4050 : ℝ) * Real.pi / 180 = 90 * Real.pi / 180 + 11 * 360 * Real.pi / 180 := by
    norm_num
  rw [this]
  -- Simplify the trigonometric identity
  have : 90 * Real.pi / 180 + 11 * 360 * Real.pi / 180 = Real.pi / 2 + 11 * 2 * Real.pi := by
    norm_num
    ring
  rw [this, Real.cos_add, Real.cos_pi_div_two]
  -- Cos pi/2 is zero, hence tan is undefined
  norm_num
  show 0 = 0
  rfl

end tan_4050_undefined_l380_380137


namespace Carol_cleaning_time_l380_380292

/-
Problem:
It takes Alice 30 minutes to clean her room. 
It takes Bob 1/3 of the time it takes Alice to clean his room.
Carol cleans her room in 3/4 of the time it takes Bob.
Prove that it takes Carol 7.5 minutes to clean her room.
-/

-- Definitions based on conditions
def Alice_time : ℝ := 30
def Bob_time : ℝ := (1/3) * Alice_time
def Carol_time : ℝ := (3/4) * Bob_time

-- Theorem to prove
theorem Carol_cleaning_time : Carol_time = 7.5 := by
  sorry

end Carol_cleaning_time_l380_380292


namespace distinct_triangle_not_isosceles_l380_380630

theorem distinct_triangle_not_isosceles (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  ¬(a = b ∨ b = c ∨ c = a) :=
by {
  sorry
}

end distinct_triangle_not_isosceles_l380_380630


namespace four_radian_in_third_quadrant_l380_380444

theorem four_radian_in_third_quadrant : ∀ θ, π < θ ∧ θ < 3 * π / 2 → (θ = 4 → π ≤ 4 ∧ 4 < 3 * π / 2) := 
by
  intro θ
  assume h1 h2
  rw [h2] at h1
  exact h1

end four_radian_in_third_quadrant_l380_380444


namespace log_3x_256_real_l380_380257

/-- Problem statement:
If \(\log_{3x} 256 = x\) and \(x\) is a real number, then \(x = 1\).
-/
theorem log_3x_256_real (x : ℝ) (h : log (3 * x) 256 = x) : x = 1 :=
by 
sorry

end log_3x_256_real_l380_380257


namespace Denis_rocks_left_l380_380516

-- Define the constants and conditions for the problem
def initial_rocks : ℕ := 50
def fraction_eaten : ℚ := 2 / 3
def removed_rocks : ℕ := 5
def spat_out_rocks : ℕ := 7

-- Define the final number of rocks left in the aquarium
def final_rocks := initial_rocks - (fraction_eaten * initial_rocks).toNat - removed_rocks + spat_out_rocks

-- The theorem stating that the final number of rocks is 19
theorem Denis_rocks_left : final_rocks = 19 := by
  sorry

end Denis_rocks_left_l380_380516


namespace new_car_distance_l380_380466

theorem new_car_distance (old_car_distance : ℝ) (increase_percentage : ℝ) (new_car_distance : ℝ) :
  old_car_distance = 150 → increase_percentage = 0.30 → new_car_distance = old_car_distance * (1 + increase_percentage) → new_car_distance = 195 :=
by
  intros h_old h_increase h_new
  rw [h_old, h_increase] at h_new
  linarith

end new_car_distance_l380_380466


namespace product_geq_n_pow_n_minus_one_l380_380205

theorem product_geq_n_pow_n_minus_one
  {n : ℕ} (hn : 0 < n)
  (x : Fin (n+1) → ℝ)
  (hx : ∀ i, 0 < x i)
  (h : ∑ i, 1 / (1 + x i) = 1) :
  (∏ i, x i) ≥ n ^ (n - 1) :=
by sorry

end product_geq_n_pow_n_minus_one_l380_380205


namespace distance_ratio_sum_is_one_l380_380789

variables (A B C D P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (a' b' c' d' a b c d : ℝ)

-- Define the distances based on the given conditions
def is_tetrahedron (A B C D : Type) : Prop := sorry
def is_inside_tetrahedron (P A B C D : Type) : Prop := sorry
def distance_to_face (P : Type) (abcd_face : Type) : ℝ := sorry
def distance_from_vertex (vertex : Type) (abcd_face : Type) : ℝ := sorry

-- The actual theorem we want to prove
theorem distance_ratio_sum_is_one
  (hT : is_tetrahedron A B C D)
  (hP : is_inside_tetrahedron P A B C D)
  (ha' : distance_to_face P (A, B, C) = a')
  (hb' : distance_to_face P (A, D, C) = b')
  (hc' : distance_to_face P (A, B, D) = c')
  (hd' : distance_to_face P (B, C, D) = d')
  (ha : distance_from_vertex A (B, C, D) = a)
  (hb : distance_from_vertex B (A, D, C) = b)
  (hc : distance_from_vertex C (A, B, D) = c)
  (hd : distance_from_vertex D (B, C, A) = d) :
  a' / a + b' / b + c' / c + d' / d = 1 := sorry

end distance_ratio_sum_is_one_l380_380789


namespace limit_sequence_l380_380569

noncomputable def a : ℕ → ℝ
| 0     := 1
| (n+1) := a n / ((n + 1) * (a n + 1))

theorem limit_sequence (a : ℕ → ℝ) (h0 : a 0 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = a n / ((n + 1) * (a n + 1))) :
  tendsto (λ n, n! * a n) at_top (𝓝 (1 / Real.exp 1)) :=
by
  sorry

end limit_sequence_l380_380569


namespace graph_reflection_l380_380235

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ -3 ∧ x ≤ 0 then -2 - x
  else if h : x > 0 ∧ x ≤ 2 then sqrt (4 - (x - 2)^2) - 2
  else if h : x > 2 ∧ x ≤ 3 then 2 * (x - 2)
  else 0

theorem graph_reflection (x : ℝ) (y : ℝ) : y = f x → y = -f x ↔ (x, -y) lies in graph A :=
sorry

end graph_reflection_l380_380235


namespace evaluate_expression_l380_380857

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = -3) :
  (2 * x)^2 * (y^2)^3 * z^2 = 1 / 81 :=
by
  -- Proof omitted
  sorry

end evaluate_expression_l380_380857


namespace sky_colors_l380_380524

theorem sky_colors (h1 : ∀ t : ℕ, t = 2) (h2 : ∀ m : ℕ, m = 60) (h3 : ∀ c : ℕ, c = 10) : 
  ∃ n : ℕ, n = 12 :=
by
  let total_duration := (2 * 60 : ℕ)
  let num_colors := total_duration / 10
  have : num_colors = 12 := by decide
  use num_colors
  assumption_needed

end sky_colors_l380_380524


namespace expression_divisible_by_1961_l380_380705

theorem expression_divisible_by_1961 (n : ℕ) : 
  (5^(2*n) * 3^(4*n) - 2^(6*n)) % 1961 = 0 := by
  sorry

end expression_divisible_by_1961_l380_380705


namespace range_of_m_if_real_roots_specific_m_given_conditions_l380_380602

open Real

-- Define the quadratic equation and its conditions
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x ^ 2 - x + 2 * m - 4 = 0
def has_real_roots (m : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2

-- Proof that m ≤ 17/8 if the quadratic equation has real roots
theorem range_of_m_if_real_roots (m : ℝ) : has_real_roots m → m ≤ 17 / 8 := 
sorry

-- Define a condition on the roots
def roots_condition (x1 x2 m : ℝ) : Prop := (x1 - 3) * (x2 - 3) = m ^ 2 - 1

-- Proof of specific m when roots condition is given
theorem specific_m_given_conditions (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ roots_condition x1 x2 m) → m = -1 :=
sorry

end range_of_m_if_real_roots_specific_m_given_conditions_l380_380602


namespace sum_squares_binomial_coeff_l380_380379

theorem sum_squares_binomial_coeff {n : ℕ} :
  (Finset.range (n + 1)).sum (λ k, (Nat.choose n k)^2) = Nat.choose (2 * n) n :=
by
  -- The proof goes here, which can include the necessary combinatorial identity arguments.
  sorry

end sum_squares_binomial_coeff_l380_380379


namespace monotonic_intervals_minimum_integer_a_l380_380231

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - (1 / 2) * a * x^2

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x ∈ set.Ioo 0 Real.infinity, f' x a > 0) ∧
  (a > 0 → 
     (∀ x ∈ set.Ioo 0 (Real.sqrt (1 / a)), f' x a > 0) ∧
     (∀ x ∈ set.Ioo (Real.sqrt (1 / a)) Real.infinity, f' x a < 0)) :=
sorry

theorem minimum_integer_a (a : ℤ) :
  (∀ x ∈ set.Ioo 0 Real.infinity, f x a ≤ (a - 1) * x - 1) → a ≥ 2 :=
sorry

end monotonic_intervals_minimum_integer_a_l380_380231


namespace area_AMDN_eq_area_ABC_l380_380491

theorem area_AMDN_eq_area_ABC
  (A B C E F D M N : Point)
  (h1 : Triangle A B C)
  (h2 : AcuteTriangle A B C)
  (h3 : On BC E)
  (h4 : On BC F)
  (h5 : ∠ BAE = ∠ CAF)
  (h6 : LineExtendsToCircle (AE) (D) (circumcircle (Triangle A B C)))
  (h7 : PerpAt FM AB M)
  (h8 : PerpAt FN AC N) :
  area (Quadrilateral A M D N) = area (Triangle A B C) :=
sorry

end area_AMDN_eq_area_ABC_l380_380491


namespace largest_d_satisfying_inequality_l380_380545

-- Define the statement of the problem
theorem largest_d_satisfying_inequality :
  ∃ d : ℝ, (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y → 
    sqrt (x^2 + y^2) + d * abs (x - y) ≤ sqrt (2 * (x + y))) ∧
    (∀ d' : ℝ, (∀ (x y : ℝ), 0 ≤ x ∧ 0 ≤ y → 
      sqrt (x^2 + y^2) + d' * abs (x - y) ≤ sqrt (2 * (x + y))) → d' ≤ 1 / sqrt 2) :=
begin
  existsi 1 / sqrt 2,
  split,
  {
    intros x y hx hy,
    sorry, -- Proof part skipped
  },
  {
    intros d' H,
    sorry, -- Proof part skipped
  }
end

end largest_d_satisfying_inequality_l380_380545


namespace sum_gcd_lcm_l380_380774

theorem sum_gcd_lcm : 
  ∀ (a b c d : ℕ), a = 24 → b = 54 → c = 48 → d = 18 →
  Nat.gcd a b + Nat.lcm c d = 150 :=
begin
  intros a b c d ha hb hc hd,
  rw [ha, hb, hc, hd],
  sorry
end

end sum_gcd_lcm_l380_380774


namespace arithmetic_sequence_properties_l380_380210

-- Definitions based on the given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in range n, a (i + 1)

-- The main theorem that combines both parts (1) and (2) of the problem
theorem arithmetic_sequence_properties :
  ∃ (a : ℕ → ℤ), is_arithmetic_sequence a ∧ a 1 = 21 ∧ a 1 + a 2 + a 3 = 57 ∧
  (∀ n, a n = -2 * n + 23) ∧ ∃ n, sum_first_n_terms a n = 121 :=
by
  sorry

end arithmetic_sequence_properties_l380_380210


namespace general_formula_a_sum_first_n_b_l380_380973

-- Definitions based on given conditions
def arithmetic_seq (a d : ℕ → ℝ) (n : ℕ) := a 1 + (n - 1) * d
def a (n : ℕ) := (1 + n) / 2
def b (n : ℕ) := 1 / (n * a n)
def S (n : ℕ) := ∑ i in Finset.range n, b (i + 1)

-- Conditions
axiom a7_eq_4 : arithmetic_seq a (1/2) 7 = 4
axiom a19_eq_2a9 : arithmetic_seq a (1/2) 19 = 2 * arithmetic_seq a (1/2) 9

-- Proof statements
theorem general_formula_a (n : ℕ) : a n = (1 + n) / 2 := sorry

theorem sum_first_n_b (n : ℕ) : S n = 2 * (1 - 1 / (n + 1)) := sorry

end general_formula_a_sum_first_n_b_l380_380973


namespace sum_difference_999_99_l380_380419

theorem sum_difference_999_99 :
  let sum := λ n : ℕ, n * (n + 1) / 2 in
  sum 999 - sum 99 = 494550 := by
    sorry

end sum_difference_999_99_l380_380419


namespace total_distance_traveled_l380_380836

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end total_distance_traveled_l380_380836


namespace number_of_true_propositions_is_one_l380_380117

theorem number_of_true_propositions_is_one :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (¬ (∀ (p q : Prop), ¬ (p ∧ q) → (¬ p ∧ ¬ q))) ∧
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) →
  1 = 1 :=
by
  sorry

end number_of_true_propositions_is_one_l380_380117


namespace problem_statement_l380_380133

-- Define the days of the week
inductive Day : Type
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

open Day

-- Define the lying and truth-telling behavior of Chris
def Chris_tells_truth (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday ∨ d = Thursday

def Chris_lies (d : Day) : Prop :=
  d = Friday ∨ d = Saturday ∨ d = Sunday

-- Define the lying and truth-telling behavior of Mark
def Mark_tells_truth (d : Day) : Prop :=
  d = Friday ∨ d = Saturday ∨ d = Sunday ∨ d = Monday

def Mark_lies (d : Day) : Prop :=
  d = Tuesday ∨ d = Wednesday ∨ d = Thursday

-- Define the statement "Tomorrow, I will lie"
def Chris_says_tomorrow_I_will_lie (d : Day) : Prop :=
  (Chris_tells_truth d ∧ (∃ t, t = match d with
                                    | Monday => Tuesday
                                    | Tuesday => Wednesday
                                    | Wednesday => Thursday
                                    | Thursday => Friday
                                    | Friday => Saturday
                                    | Saturday => Sunday
                                    | Sunday => Monday
                                    end ∧ Chris_lies t)) ∨
  (Chris_lies d ∧ (∃ t, t = match d with
                             | Monday => Tuesday
                             | Tuesday => Wednesday
                             | Wednesday => Thursday
                             | Thursday => Friday
                             | Friday => Saturday
                             | Saturday => Sunday
                             | Sunday => Monday
                             end ∧ Chris_tells_truth t))

def Mark_says_tomorrow_I_will_lie (d : Day) : Prop :=
  (Mark_tells_truth d ∧ (∃ t, t = match d with
                                   | Monday => Tuesday
                                   | Tuesday => Wednesday
                                   | Wednesday => Thursday
                                   | Thursday => Friday
                                   | Friday => Saturday
                                   | Saturday => Sunday
                                   | Sunday => Monday
                                   end ∧ Mark_lies t)) ∨
  (Mark_lies d ∧ (∃ t, t = match d with
                            | Monday => Tuesday
                            | Tuesday => Wednesday
                            | Wednesday => Thursday
                            | Thursday => Friday
                            | Friday => Saturday
                            | Saturday => Sunday
                            | Sunday => Monday
                            end ∧ Mark_tells_truth t))

-- The theorem to be proved
theorem problem_statement : ∃ d : Day, Chris_says_tomorrow_I_will_lie d ∧ Mark_says_tomorrow_I_will_lie d ∧ d = Thursday :=
by
  sorry

end problem_statement_l380_380133


namespace length_imaginary_axis_hyperbola_l380_380736

theorem length_imaginary_axis_hyperbola : 
  ∀ (a b : ℝ), (a = 2) → (b = 1) → 
  (∀ x y : ℝ, (y^2 / a^2 - x^2 = 1) → 2 * b = 2) :=
by intros a b ha hb x y h; sorry

end length_imaginary_axis_hyperbola_l380_380736


namespace find_theta_l380_380619

theorem find_theta (θ : ℝ) (h1 : 0 < θ ∧ θ < 90) (h2 : real.sqrt 3 * real.sin (5 * real.pi / 180) = real.cos (θ * real.pi / 180) - real.sin (θ * real.pi / 180)) :
  θ = 40 :=
by
  sorry

end find_theta_l380_380619


namespace jumping_frog_hexagon_l380_380685

/-- 
A regular hexagon with vertices labeled A, B, C, D, E, F.
A frog starts at vertex A and can jump to one of the two adjacent vertices.
If the frog reaches vertex D within 5 jumps, it stops. If not, it stops after 5 jumps.
Prove that the number of distinct jump patterns such that 
the frog ends its jumping sequence is 26.
-/
theorem jumping_frog_hexagon : 
  let vertices := ['A', 'B', 'C', 'D', 'E', 'F'] in
  let start := 'A' in
  let jumps := 5 in
  let adj (v : Char) : List Char := 
    match v with
    | 'A' => ['B', 'F']
    | 'B' => ['A', 'C']
    | 'C' => ['B', 'D']
    | 'D' => ['C', 'E']
    | 'E' => ['D', 'F']
    | 'F' => ['A', 'E']
    | _ => [] in
  ∃ patterns : List (List Char),
  ∀ p ∈ patterns, 
    p.length = jumps ∧ p.head = start ∧ 
    (p.last = 'D' ∨ p.length = jumps) ∧ 
    ( ∀i, i < jumps → p.nth i ≠ p.nth (i + 1) → p.nth (i + 1) ∈ adj (p.nth i).val ) ∧
  patterns.length = 26 :=
sorry

end jumping_frog_hexagon_l380_380685


namespace arrangement_plans_correct_l380_380472

variables (X : Type) [Fintype X] (n : ℕ) 

noncomputable def arrangement_plans : ℕ :=
  4 * (Finset.card (Finset.univ : Finset (Finset (Fintype.of_range 1)))) +
  4 * (Finset.card (Finset.univ : Finset (Finset (Fintype.of_range 2)))) +
  6 * (Finset.card (Finset.univ : Finset (Finset (Fintype.of_range 1))))

theorem arrangement_plans_correct : arrangement_plans X n = 50 :=
begin
  sorry
end

end arrangement_plans_correct_l380_380472


namespace digit_150_of_17_div_70_l380_380030

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380030


namespace complex_fraction_sum_real_parts_l380_380914

theorem complex_fraction_sum_real_parts (a b : ℝ) (h : (⟨0, 1⟩ / ⟨1, 1⟩ : ℂ) = a + b * ⟨0, 1⟩) : a + b = 1 := by
  sorry

end complex_fraction_sum_real_parts_l380_380914


namespace jenna_costume_l380_380991

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end jenna_costume_l380_380991


namespace chameleon_problem_l380_380341

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380341


namespace tree_break_height_l380_380829

-- Define the problem conditions and prove the required height h
theorem tree_break_height (height_tree : ℝ) (distance_shore : ℝ) (height_break : ℝ) : 
  height_tree = 20 → distance_shore = 6 → 
  (distance_shore ^ 2 + height_break ^ 2 = (height_tree - height_break) ^ 2) →
  height_break = 9.1 :=
by
  intros h_tree_eq h_shore_eq hyp_eq
  have h_tree_20 := h_tree_eq
  have h_shore_6 := h_shore_eq
  have hyp := hyp_eq
  sorry -- Proof of the theorem is omitted

end tree_break_height_l380_380829


namespace sum_fractional_part_ξ_l380_380186

noncomputable def ξ (x : ℝ) := ∑ n in (Set.Ioi 0).toNat, (2 * n) ^ (-x)

theorem sum_fractional_part_ξ :
  (∑ k in (Set.Ioi 2).toNat, (ξ (2 * k + 1)) % 1) = 1 := 
sorry

end sum_fractional_part_ξ_l380_380186


namespace area_enclosed_by_curve_and_line_l380_380386

theorem area_enclosed_by_curve_and_line :
  let f := fun x : ℝ => x^2 + 2
  let g := fun x : ℝ => 3 * x
  let A := ∫ x in (0 : ℝ)..1, (f x - g x) + ∫ x in (1 : ℝ)..2, (g x - f x)
  A = 1 := by
    sorry

end area_enclosed_by_curve_and_line_l380_380386


namespace triangle_ABC_construction_l380_380606

-- Definitions given in the problem statement
variables {A B C O F D : ℝ}
variables (mid_AC : D = (A + C) / 2)
variables (mid_parallel_AC : F = (D + O) / 2)

-- The theorem to be proved
theorem triangle_ABC_construction :
  ∃ (B C : ℝ), 
    let D := (A + C) / 2 in
    let F := (D + O) / 2 in
    -- D is the reflection of B on F implies their relation
    D = 2 * F - B ∧
    -- Thales' theorem condition for the right triangle OAD
    D^2 + A^2 = O^2 ∧
    -- The conditions ensuring the placement of B and C
    F = (D + O) / 2 ∧
    -- construct the triangle, where D lies between A and C
    -- establishing the points C, B ensuring valid geometric construction
    B ∈ { x | (A - x)^2 + (C - x)^2 = (D - x)^2 } :=
sorry

end triangle_ABC_construction_l380_380606


namespace problem_solution_l380_380385

theorem problem_solution (y : Real)
  (h1 : (Real.sec y - Real.tan y = 15 / 8))
  (h2 : ∃ (p q : ℤ), Real.csc y - Real.cot y = p / q ∧ Int.gcd p q = 1) :
  ∃ (p q : ℕ), p + q = 30 ∧ Real.csc y - Real.cot y = p / (q : ℤ) := by
  sorry

end problem_solution_l380_380385


namespace projection_of_a_onto_b_l380_380876

/-
  Define the vectors a and b as given in the conditions
-/
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (3, 4)

/-
  Compute the dot product of vec_a and vec_b
-/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-
  Compute the magnitude (norm) of vec_b
-/
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

/-
  Compute the projection of vec_a in the direction of vec_b.
-/
def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / magnitude v2

/-
  State the theorem
-/
theorem projection_of_a_onto_b :
  projection vec_a vec_b = 2 :=
  by sorry

end projection_of_a_onto_b_l380_380876


namespace possible_values_of_n_l380_380275

theorem possible_values_of_n (n : ℕ) (h : ∀ (l : fin n) (s : fin n → Prop), s l → ∃ m ≥ 1999, IntersectingLines m) : n = 2000 ∨ n = 3998 :=
sorry

end possible_values_of_n_l380_380275


namespace train_crossing_time_l380_380479

variable (L_train : ℝ) (L_bridge : ℝ) (Speed_train_kmh : ℝ)

noncomputable def Speed_train_ms := Speed_train_kmh * (1000 / 3600)
noncomputable def Total_distance := L_train + L_bridge
noncomputable def Time_to_cross := Total_distance / Speed_train_ms

theorem train_crossing_time 
  (h_train_length : L_train = 170)
  (h_bridge_length : L_bridge = 205)
  (h_train_speed : Speed_train_kmh = 45) :
  Time_to_cross L_train L_bridge Speed_train_kmh = 30 :=
by
  rw [h_train_length, h_bridge_length, h_train_speed]
  simp [Speed_train_ms, Total_distance, Time_to_cross]
  sorry

end train_crossing_time_l380_380479


namespace ratio_xyz_l380_380154

theorem ratio_xyz (a x y z : ℝ) : 
  5 * x + 4 * y - 6 * z = a ∧
  4 * x - 5 * y + 7 * z = 27 * a ∧
  6 * x + 5 * y - 4 * z = 18 * a →
  (x :ℝ) / (y :ℝ) = 3 / 4 ∧
  (y :ℝ) / (z :ℝ) = 4 / 5 :=
by
  sorry

end ratio_xyz_l380_380154


namespace mean_points_scored_is_48_l380_380830

def class_points : List ℤ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_points_scored_is_48 : (class_points.sum / class_points.length) = 48 := by
  sorry

end mean_points_scored_is_48_l380_380830


namespace problem_1_problem_2_l380_380192

-- Problem I
theorem problem_1 (x : ℝ) (h : |x - 2| + |x - 1| < 4) : (-1/2 : ℝ) < x ∧ x < 7/2 :=
sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 1| ≥ 2) : a ≤ -1 ∨ a ≥ 3 :=
sorry

end problem_1_problem_2_l380_380192


namespace no_max_min_value_l380_380920

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem no_max_min_value {x : ℝ} (h : |x| < 1) :
  ¬ (∃ y ∈ set.Ioo (-1) 1, ∀ x ∈ set.Ioo (-1) 1, f y ≥ f x)
  ∧ ¬ (∃ y ∈ set.Ioo (-1) 1, ∀ x ∈ set.Ioo (-1) 1, f y ≤ f x) :=
sorry

end no_max_min_value_l380_380920


namespace count_pos_three_digit_ints_with_same_digits_l380_380950

-- Define a structure to encapsulate the conditions for a three-digit number less than 700 with at least two digits the same.
structure valid_int (n : ℕ) : Prop :=
  (three_digit : 100 ≤ n ∧ n < 700)
  (same_digits : ∃ d₁ d₂ d₃ : ℕ, ((100 * d₁ + 10 * d₂ + d₃ = n) ∧ (d₁ = d₂ ∨ d₂ = d₃ ∨ d₁ = d₃)))

-- The number of integers satisfying the conditions
def count_valid_ints : ℕ :=
  168

-- The theorem to prove
theorem count_pos_three_digit_ints_with_same_digits : 
  (∃ n, valid_int n) → 168 :=
by
  -- Since the proof is not required, we add sorry here.
  sorry

end count_pos_three_digit_ints_with_same_digits_l380_380950


namespace largest_value_l380_380072

def expr_A : ℕ := 3 + 1 + 0 + 5
def expr_B : ℕ := 3 * 1 + 0 + 5
def expr_C : ℕ := 3 + 1 * 0 + 5
def expr_D : ℕ := 3 * 1 + 0 * 5
def expr_E : ℕ := 3 * 1 + 0 * 5 * 3

theorem largest_value :
  expr_A > expr_B ∧
  expr_A > expr_C ∧
  expr_A > expr_D ∧
  expr_A > expr_E :=
by
  sorry

end largest_value_l380_380072


namespace alice_favorite_number_l380_380819

namespace AliceNumber

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m
def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem alice_favorite_number : 
  ∃ (x : ℕ), 
    100 < x ∧ 
    x < 150 ∧ 
    is_multiple_of x 13 ∧ 
    ¬ is_multiple_of x 2 ∧ 
    is_multiple_of (sum_of_digits x) 4 :=
by
  use 143
  have h1 : 100 < 143 := by norm_num
  have h2 : 143 < 150 := by norm_num
  have h3 : is_multiple_of 143 13 := by 
    use 11
    norm_num
  have h4 : ¬ is_multiple_of 143 2 := by
    intro ⟨m, hm⟩
    exact (nat.mod_eq_of_lt (by norm_num) (by norm_num)).symm ▸ hm.split_left
  have h5 : is_multiple_of (sum_of_digits 143) 4 := by
    calc
      sum_of_digits 143 = (1 + 4 + 3) := by norm_num 
                 ... = 8 := by norm_num
        show is_multiple_of 8 4
        use 2
        norm_num
  exact ⟨h1, h2, h3, h4, h5⟩

end alice_favorite_number_l380_380819


namespace digit_150th_of_17_div_70_is_7_l380_380021

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380021


namespace least_squares_representation_l380_380974

-- Define the conditions/options
def optionA (y : Fin n → ℝ) (y_bar : ℝ) : Prop := 
  ∑ i, (y i - y_bar) = 0

def optionB (y : Fin n → ℝ) (y_hat : ℝ) : Prop :=
  n * ∑ i, (y i - y_hat) = 0

def optionC (y : Fin n → ℝ) (y_hat : ℝ) : Prop :=
  (∑ i, (y i - y_hat))

def optionD (y : Fin n → ℝ) (y_hat : ℝ) : Prop :=
  (∑ i, (y i - y_hat) ^ 2)

-- The theorem stating Option D represents the method of least squares
theorem least_squares_representation (y : Fin n → ℝ) (y_hat : ℝ) :
  optionD y y_hat := 
sorry

end least_squares_representation_l380_380974


namespace max_t_is_3_l380_380896

noncomputable def max_t_when_APB_right_angle (t : ℝ) (h : t > 0) : Prop :=
  ∃ P : ℝ × ℝ, 
    (x - sqrt 3)^2 + (y - 1)^2 = 1 ∧ -t = x ∧ 0 = y ∧ t = x ∧ 0 = y ∧ ∠APB = 90° 

theorem max_t_is_3 : ∀ (t : ℝ) (h : t > 0), max_t_when_APB_right_angle t h → t ≤ 3 :=
begin
  intros t h ht,
  sorry
end

end max_t_is_3_l380_380896


namespace domain_range_same_eq_l380_380919

noncomputable def f (a b x : ℝ) : ℝ := real.sqrt (a * x ^ 2 + b * x)

theorem domain_range_same_eq {a b : ℝ} (hb : b > 0) :
  (∀ x : ℝ, (a * x ^ 2 + b * x ≥ 0) ∧ f a b x ∈ (range (f a b))) ↔ (a = -4 ∨ a = 0) := 
sorry

end domain_range_same_eq_l380_380919


namespace height_percentage_increase_l380_380259

/-- 
If A's height is 45% less than that of B, 
then B's height is approximately 81.82% more than that of A.
-/
theorem height_percentage_increase (B A : ℝ) (h : A = B * 0.55) : 
    (B - A) / A * 100 ≈ 81.82 := 
by 
  sorry

end height_percentage_increase_l380_380259


namespace candles_left_l380_380821

theorem candles_left (total_candles : ℕ) (alyssa_fraction_used : ℚ) (chelsea_fraction_used : ℚ) 
  (h_total : total_candles = 40) 
  (h_alyssa : alyssa_fraction_used = (1 / 2)) 
  (h_chelsea : chelsea_fraction_used = (70 / 100)) : 
  total_candles - (alyssa_fraction_used * total_candles).toNat - (chelsea_fraction_used * (total_candles - (alyssa_fraction_used * total_candles).toNat)).toNat = 6 :=
by 
  sorry

end candles_left_l380_380821


namespace hamiltonian_cycle_existence_l380_380639

theorem hamiltonian_cycle_existence (n m : ℕ) (G : SimpleGraph (Fin n)) 
  (hne : n = 2009)
  (h1004 : ∀ v : Fin n, G.degree v = 1004) :
  G.is_hamiltonian :=
sorry

end hamiltonian_cycle_existence_l380_380639


namespace units_digit_of_47_pow_25_l380_380063

theorem units_digit_of_47_pow_25 : 
  let cycle := [7, 9, 3, 1] in
  let n := 25 in
  let units_digit_47 := cycle[(n % 4)] in
  units_digit_47 = 7 :=
by
  let cycle := [7, 9, 3, 1]
  let n := 25
  let units_digit_47 := cycle[(n % 4)]
  have h : n % 4 = 1 := sorry
  have h1 : units_digit_47 = 7 := sorry
  exact h1

end units_digit_of_47_pow_25_l380_380063


namespace eval_g_at_3_l380_380396

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem eval_g_at_3 : g 3 = 10 := by
  -- Proof goes here
  sorry

end eval_g_at_3_l380_380396


namespace parallelogram_area_l380_380499

noncomputable def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.inner v)
noncomputable def cross_product (u v : ℝ^3) : ℝ^3 := sorry

theorem parallelogram_area 
  (p q : ℝ^3)
  (a := 4 • p - q)
  (b := p + 2 • q)
  (hp : magnitude p = 5)
  (hq : magnitude q = 4)
  (angle_pq : real.angle p q = real.pi / 4) : 
  magnitude (cross_product a b) = 90 * real.sqrt 2 := 
sorry

end parallelogram_area_l380_380499


namespace trig_expression_value_l380_380510

noncomputable def problem_expr : ℝ :=
  (√3 * sin (-20/3 * π) / tan (11/3 * π)) - (cos (13/4 * π) * tan (-35/4 * π))

theorem trig_expression_value : problem_expr = (√2 + √3) / 2 := by
  sorry

end trig_expression_value_l380_380510


namespace jane_spent_75_days_reading_l380_380667

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end jane_spent_75_days_reading_l380_380667


namespace no_valid_formation_l380_380803

-- Define the conditions related to the formation:
-- s : number of rows
-- t : number of musicians per row
-- Total musicians = s * t = 400
-- t is divisible by 4
-- 10 ≤ t ≤ 50
-- Additionally, the brass section needs to form a triangle in the first three rows
-- while maintaining equal distribution of musicians from each section in every row.

theorem no_valid_formation (s t : ℕ) (h_mul : s * t = 400) 
  (h_div : t % 4 = 0) 
  (h_range : 10 ≤ t ∧ t ≤ 50) 
  (h_triangle : ∀ (r1 r2 r3 : ℕ), r1 < r2 ∧ r2 < r3 → r1 + r2 + r3 = 100 → false) : 
  x = 0 := by
  sorry

end no_valid_formation_l380_380803


namespace number_of_different_keys_l380_380401

theorem number_of_different_keys (n : ℕ) (h : Even n) : 
  ∃ k : ℕ, k = 4 ^ (n ^ 2 / 4) := 
sorry

end number_of_different_keys_l380_380401


namespace smallest_number_of_pets_l380_380875

noncomputable def smallest_common_multiple (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

theorem smallest_number_of_pets : smallest_common_multiple 3 15 9 = 45 :=
by
  sorry

end smallest_number_of_pets_l380_380875


namespace max_tan_B_l380_380268

-- Declare the context of triangle ABC with sides a, b, c opposite angles A, B, C respectively
variables {A B C : ℝ}
variables {a b c : ℝ}

-- Declare the given condition
def condition : Prop := 3 * a * Real.cos C + b = 0

-- State the theorem
theorem max_tan_B (h : condition) : 
  ∃ M : ℝ, M = 3/4 ∧ ∀ (x : ℝ), (x = Real.tan B) → x ≤ M :=
by 
  sorry

end max_tan_B_l380_380268


namespace three_digit_integers_with_two_identical_digits_less_than_700_l380_380946

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def less_than_700 (n : ℕ) : Prop :=
  n < 700

def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.nodup = false

theorem three_digit_integers_with_two_identical_digits_less_than_700 : 
  ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_three_digit n ∧ less_than_700 n ∧ has_at_least_two_identical_digits n) ∧
  s.card = 156 := by
  sorry

end three_digit_integers_with_two_identical_digits_less_than_700_l380_380946


namespace find_nine_boxes_of_same_variety_l380_380105

theorem find_nine_boxes_of_same_variety (boxes : ℕ) (A B C : ℕ) (h_total : boxes = 25) (h_one_variety : boxes = A + B + C) 
  (hA : A ≤ 25) (hB : B ≤ 25) (hC : C ≤ 25) :
  (A ≥ 9) ∨ (B ≥ 9) ∨ (C ≥ 9) :=
sorry

end find_nine_boxes_of_same_variety_l380_380105


namespace midpoint_locus_distance_to_origin_l380_380280

-- Given conditions
variables (a b : ℝ) (h : a > b ∧ b > 0)

-- Statement 1: Prove the ordinary equation of the locus of the midpoint M
theorem midpoint_locus (α : ℝ) (x y : ℝ) (h_eq_x : x = (a * cos α - a * sin α) / 2)
    (h_eq_y : y = (b * sin α + b * cos α) / 2):
    (2 * x^2 / a^2 + 2 * y^2 / b^2 = 1) :=
sorry

-- Statement 2: Prove the distance from the origin O to the line AB is a fixed value
theorem distance_to_origin: 
    (ab / sqrt (a^2 + b^2)) :=
sorry

end midpoint_locus_distance_to_origin_l380_380280


namespace hyperbola_equation_l380_380531

theorem hyperbola_equation :
  ∀ (x y : ℝ),
  (let A := (sqrt 3, 2 * sqrt 5) in
   ∃ (λ : ℝ), λ ≠ 0 ∧ ∀ x y, (x, y) = A → (x^2 / 3 - y^2 / 2 = λ) → (y^2 / 18 - x^2 / 27 = 1)) :=
by sorry

end hyperbola_equation_l380_380531


namespace range_of_a_l380_380241

noncomputable def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 3}

theorem range_of_a (a : ℝ) :
  ((setA a ∩ setB) = setA a) ∧ (∃ x, x ∈ (setA a ∩ setB)) →
  (a < -3 ∨ a > 3) ∧ (a < -1 ∨ a > 1) :=
by sorry

end range_of_a_l380_380241


namespace additional_track_length_needed_l380_380465

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end additional_track_length_needed_l380_380465


namespace sphere_volume_equals_surface_area_l380_380474

theorem sphere_volume_equals_surface_area (r : ℝ) (hr : r = 3) :
  (4 / 3) * π * r^3 = 4 * π * r^2 := by
  sorry

end sphere_volume_equals_surface_area_l380_380474


namespace increase_in_green_chameleons_is_11_l380_380317

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380317


namespace possible_values_z_plus_i_conj_l380_380213

noncomputable def z : ℂ := complex.of_real a + complex.i * b
def z_conj : ℂ := complex.of_real a - complex.i * b

theorem possible_values_z_plus_i_conj (a b : ℝ) :
  ((z + complex.i) * (z_conj + complex.i) = 1 - 2 * complex.i) ∨
  ((z + complex.i) * (z_conj + complex.i) = 2 * complex.i) ∨
  ((z + complex.i) * (z_conj + complex.i) = -2 * complex.i) :=
sorry

end possible_values_z_plus_i_conj_l380_380213


namespace proof_problem_l380_380715

def g : ℕ → ℕ := sorry
def g_inv : ℕ → ℕ := sorry

axiom g_inv_is_inverse : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y
axiom g_4_eq_6 : g 4 = 6
axiom g_6_eq_2 : g 6 = 2
axiom g_3_eq_7 : g 3 = 7

theorem proof_problem :
  g_inv (g_inv 7 + g_inv 6) = 3 :=
by
  sorry

end proof_problem_l380_380715


namespace calculate_radius_l380_380107

noncomputable def radius_of_wheel (D : ℝ) (N : ℕ) (π : ℝ) : ℝ :=
  D / (2 * π * N)

theorem calculate_radius : 
  radius_of_wheel 4224 3000 Real.pi = 0.224 :=
by
  sorry

end calculate_radius_l380_380107


namespace part1_intersection_length_part2_minimum_distance_l380_380601

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + 1/2 * t, sqrt 3 / 2 * t)

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (1/2 * cos θ, sqrt 3 / 2 * sin θ)

theorem part1_intersection_length:
  let A := (1 - 1/2, -sqrt 3 / 2)  -- Point corresponding to t = -1
  let B := (1, 0)                   -- Point corresponding to t = 0
  abs (1 - 0) = 1
  := sorry

theorem part2_minimum_distance:
  ∀ (P : ℝ × ℝ), P ∈ set_of (λ θ : ℝ, curve_C2 θ) →
  let d (P : ℝ × ℝ) := abs (sqrt 3 / 2 * cos (θ + π / 4) - sqrt 3) / 2
  min (λ P, d P) = (sqrt 3 / 2 - sqrt 6 / 4)
  := sorry

end part1_intersection_length_part2_minimum_distance_l380_380601


namespace gasoline_added_l380_380262

noncomputable def initial_amount (capacity: ℕ) : ℝ :=
  (3 / 4) * capacity

noncomputable def final_amount (capacity: ℕ) : ℝ :=
  (9 / 10) * capacity

theorem gasoline_added (capacity: ℕ) (initial_fraction final_fraction: ℝ) (initial_amount final_amount: ℝ) : 
  capacity = 54 ∧ initial_fraction = 3/4 ∧ final_fraction = 9/10 ∧ 
  initial_amount = initial_fraction * capacity ∧ 
  final_amount = final_fraction * capacity →
  final_amount - initial_amount = 8.1 :=
sorry

end gasoline_added_l380_380262


namespace cube_root_of_456533_is_77_l380_380582

theorem cube_root_of_456533_is_77 (z : ℤ) (h1 : z = Int.root (456533 : ℤ) 3) (h2 : z * z * z = 456533) : z = 77 :=
sorry

end cube_root_of_456533_is_77_l380_380582


namespace find_common_ratio_l380_380283

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m, ∃ q, a (n + 1) = a n * q ∧ a (m + 1) = a m * q

theorem find_common_ratio 
  (a : ℕ → α) 
  (h : is_geometric_sequence a) 
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1 / 4) : 
  ∃ q, q = 1 / 2 :=
by
  sorry

end find_common_ratio_l380_380283


namespace quadratic_root_zero_l380_380554

theorem quadratic_root_zero (k : ℝ) :
    (∃ x : ℝ, x = 0 ∧ (k - 1) * x ^ 2 + 6 * x + k ^ 2 - k = 0) → k = 0 :=
by
  sorry

end quadratic_root_zero_l380_380554


namespace tan_alpha_eq_two_cos_plus_sin_over_cos_minus_sin_simplify_f_alpha_l380_380501

-- Proof Problem 1 definition
theorem tan_alpha_eq_two_cos_plus_sin_over_cos_minus_sin (α : ℝ) (h : tan α = 2) : 
  (cos α + sin α) / (cos α - sin α) = -3 := 
sorry

-- Proof Problem 2 definition
theorem simplify_f_alpha (α : ℝ) : 
  (sin (α - π / 2) * cos (π / 2 - α) * tan (π - α)) / (tan (π + α) * sin (π + α)) = -cos α := 
sorry

end tan_alpha_eq_two_cos_plus_sin_over_cos_minus_sin_simplify_f_alpha_l380_380501


namespace pairs_count_l380_380625

def nat_pairs_count_satisfying (a_max b_max : ℕ) (u v : ℝ) :=
  (finite (set.filter (λ (p : ℕ × ℕ), let (a, b) := p in 1 ≤ a ∧ a ≤ a_max ∧ 1 ≤ b ∧ b ≤ b_max ∧ u < (b : ℝ) / (a : ℝ) ∧ (b : ℝ) / (a : ℝ) < v) (finset.product (finset.range (a_max + 1)) (finset.range (b_max + 1))))).to_finset.card

theorem pairs_count :
  nat_pairs_count_satisfying 10 10 (1.0/3.0) (1.0/2.0) = 5 :=
by
  sorry

end pairs_count_l380_380625


namespace rachel_age_is_24_5_l380_380374

/-- Rachel is 4 years older than Leah -/
def rachel_age_eq_leah_plus_4 (R L : ℝ) : Prop := R = L + 4

/-- Together, Rachel and Leah are twice as old as Sam -/
def rachel_and_leah_eq_twice_sam (R L S : ℝ) : Prop := R + L = 2 * S

/-- Alex is twice as old as Rachel -/
def alex_eq_twice_rachel (A R : ℝ) : Prop := A = 2 * R

/-- The sum of all four friends' ages is 92 -/
def sum_ages_eq_92 (R L S A : ℝ) : Prop := R + L + S + A = 92

theorem rachel_age_is_24_5 (R L S A : ℝ) :
  rachel_age_eq_leah_plus_4 R L →
  rachel_and_leah_eq_twice_sam R L S →
  alex_eq_twice_rachel A R →
  sum_ages_eq_92 R L S A →
  R = 24.5 := 
by 
  sorry

end rachel_age_is_24_5_l380_380374


namespace part1_part2_l380_380878

noncomputable def f (x : ℝ) : ℝ := x / (3 * x + 1)

def a : ℕ → ℝ
| 0     := 1
| (n+1) := f (a n)

def b : ℕ → ℝ
| 0     := 1
| n     := 2^(n-1)

def S (n : ℕ) : ℝ := 2^n - 1

def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i / a i

theorem part1 : ∃ d : ℝ, ∀ n : ℕ, (1 / a (n + 1)) - (1 / a n) = d := 
sorry

theorem part2 (n : ℕ) : T n = (3 * n - 5) * 2^n + 5 :=
sorry

end part1_part2_l380_380878


namespace smallest_n_for_terminating_decimal_l380_380059

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), 0 < n ∧ ∀ m : ℕ, (0 < m ∧ m < n+53) → (∃ a b : ℕ, n + 53 = 2^a * 5^b) → n = 11 :=
by
  sorry

end smallest_n_for_terminating_decimal_l380_380059


namespace winning_strategy_for_first_player_l380_380699

theorem winning_strategy_for_first_player (cards : Fin 2006 → ℕ) :
  ∃ strategy, ∀ strategy', 
  let (first_player_sum, second_player_sum) := play_game cards strategy strategy' 
  in first_player_sum ≥ second_player_sum :=
sorry

-- Helper function template for game play, assuming a certain strategy implementation exists.
def play_game (cards : Fin 2006 → ℕ) (strategy : ℕ → bool) (strategy' : ℕ → bool) : ℕ × ℕ :=
sorry -- Implementation of the game's mechanics.

end winning_strategy_for_first_player_l380_380699


namespace geometric_sequence_term_formula_minimum_length_of_range_interval_l380_380583

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n-1)

noncomputable def S_n (a₁ : ℝ) (q : ℝ) (n : ℕ) :=
  (geometric_sequence a₁ q n).sum

theorem geometric_sequence_term_formula 
  (a₁ : ℝ) (q : ℝ) (S_2 S_3 S_4 : ℝ) 
  (h1 : -2 * S_2 + 4 * S_4 = 2 * S_3)
  (h2 : a₁ = 3 / 2) 
  (h3 : q = -1 / 2) :
  (geometric_sequence a₁ q n) = (3 / 2) * (-1 / 2)^(n-1) :=
sorry

theorem minimum_length_of_range_interval
  (a₁ : ℝ) (q : ℝ) (S_n : ℕ → ℝ)
  (b_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = 1 - (-1/2)^n ∨ S_n n = 1 + (1/2)^n)
  (h2 : ∀ n, b_n n = S_n n + 1 / (S_n n)) :
  ∃ M : set ℝ, (∀ n, b_n n ∈ M) ∧
  (∀ i j, i ≠ j → M i ≠ M j ) ∧ 
  (∀ i, ∃ I : ℝ, M = i ∧ M = [2, 13/6]) ∧ 
  (min_length M = 1/6) :=
sorry

end geometric_sequence_term_formula_minimum_length_of_range_interval_l380_380583


namespace suzanna_total_distance_l380_380733

def speed_at_interval (n : ℕ) : ℝ :=
  12 - n

def distance_per_interval (speed : ℝ) : ℝ :=
  speed * (5 / 60) -- 5 minutes is 5/60 hours

def total_distance : ℝ :=
  distance_per_interval 12 + distance_per_interval 11 +
  distance_per_interval 10 + distance_per_interval 9 +
  distance_per_interval 8 + distance_per_interval 7

theorem suzanna_total_distance : total_distance = 4.75 := 
by sorry

end suzanna_total_distance_l380_380733


namespace maximal_inradius_of_tetrahedron_l380_380179

-- Define the properties and variables
variables (A B C D : ℝ) (h_A h_B h_C h_D : ℝ) (V r : ℝ)

-- Assumptions
variable (h_A_ge_1 : h_A ≥ 1)
variable (h_B_ge_1 : h_B ≥ 1)
variable (h_C_ge_1 : h_C ≥ 1)
variable (h_D_ge_1 : h_D ≥ 1)

-- Volume expressed in terms of altitudes and face areas
axiom vol_eq_Ah : V = (1 / 3) * A * h_A
axiom vol_eq_Bh : V = (1 / 3) * B * h_B
axiom vol_eq_Ch : V = (1 / 3) * C * h_C
axiom vol_eq_Dh : V = (1 / 3) * D * h_D

-- Volume expressed in terms of inradius and sum of face areas
axiom vol_eq_inradius : V = (1 / 3) * (A + B + C + D) * r

-- The theorem to prove
theorem maximal_inradius_of_tetrahedron : r = 1 / 4 :=
sorry

end maximal_inradius_of_tetrahedron_l380_380179


namespace worker_new_wage_after_increase_l380_380483

theorem worker_new_wage_after_increase (initial_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) 
  (h1 : initial_wage = 34) (h2 : increase_percentage = 0.50) 
  (h3 : new_wage = initial_wage + (increase_percentage * initial_wage)) : new_wage = 51 := 
by
  sorry

end worker_new_wage_after_increase_l380_380483


namespace chameleon_problem_l380_380357

/-- There are red, yellow, green, and blue chameleons on an island. -/
variable (num_red num_yellow num_green num_blue : ℕ)

/-- On a cloudy day, either one red chameleon changes its color to yellow, 
    or one green chameleon changes its color to blue. -/
def cloudy_day_effect (cloudy_days : ℕ) (changes : ℕ) : ℕ :=
  changes - cloudy_days

/-- On a sunny day, either one red chameleon changes its color to green,
    or one yellow chameleon changes its color to blue. -/
def sunny_day_effect (sunny_days : ℕ) (changes : ℕ) : ℕ :=
  changes + sunny_days

/-- In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5. -/
theorem chameleon_problem (sunny_days cloudy_days yellow_increase : ℕ) (h_sunny : sunny_days = 18) (h_cloudy : cloudy_days = 12) (h_yellow : yellow_increase = 5) :
  ∀ changes : ℕ, sunny_day_effect sunny_days (cloudy_day_effect cloudy_days changes) - yellow_increase = 6 →
  changes + 5 = 11 :=
by
  intros
  subst_vars
  sorry

end chameleon_problem_l380_380357


namespace units_digit_17_pow_310_is_9_l380_380062

-- Definition of the recurring units digit sequence for powers of 7
def units_digit_sequence_7 : List ℕ := [7, 9, 3, 1]

-- Definition for extracting the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The theorem stating the core result
theorem units_digit_17_pow_310_is_9 :
  let seq := units_digit_sequence_7 in
  units_digit (17 ^ 310) = seq[(310 % 4) % seq.length] :=
by
  have h310_mod_4 : 310 % 4 = 2 := by norm_num
  have h7_pow2_units : units_digit (7 ^ 2) = 9 := by norm_num
  have h17_pow_units : units_digit (17 ^ 310) = units_digit (7 ^ 310) := by simp [units_digit]
  simp [units_digit] at h17_pow_units
  rw [h310_mod_4] at h17_pow_units
  simp [h7_pow2_units] at h17_pow_units
  exact h17_pow_units

end units_digit_17_pow_310_is_9_l380_380062


namespace shaded_area_is_correct_l380_380104

noncomputable def square_shaded_area (side : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
  if (0 < beta) ∧ (beta < 90) ∧ (cos_beta = 3 / 5) ∧ (side = 2) then 3 / 10 
  else 0

theorem shaded_area_is_correct :
  square_shaded_area 2 beta (3 / 5) = 3 / 10 :=
by
  sorry

end shaded_area_is_correct_l380_380104


namespace equilateral_triangle_grid_l380_380855

noncomputable def number_of_triangles (n : ℕ) : ℕ :=
1 + 3 + 5 + 7 + 9 + 1 + 2 + 3 + 4 + 3 + 1 + 2 + 3 + 1 + 2 + 1

theorem equilateral_triangle_grid (n : ℕ) (h : n = 5) : number_of_triangles n = 48 := by
  sorry

end equilateral_triangle_grid_l380_380855


namespace lines_perpendicular_to_same_plane_are_parallel_l380_380824

variables {Point Line Plane : Type}
variables (a b c : Line) (α β γ : Plane)
variables (perp_line_to_plane : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)
variables (subset_line_in_plane : Line → Plane → Prop)

-- The conditions
axiom a_perp_alpha : perp_line_to_plane a α
axiom b_perp_alpha : perp_line_to_plane b α

-- The statement to prove
theorem lines_perpendicular_to_same_plane_are_parallel :
  parallel_lines a b :=
by sorry

end lines_perpendicular_to_same_plane_are_parallel_l380_380824


namespace ratio_charliz_jose_guppies_l380_380938

theorem ratio_charliz_jose_guppies :
  (∀ (Haylee Jose Charliz Nicolai : ℕ),
    Haylee = 3 * 12 →
    Jose = Haylee / 2 →
    Nicolai = 4 * Charliz →
    Haylee + Jose + Charliz + Nicolai = 84 →
    Charliz / Jose = 1 / 3) :=
begin
  sorry,
end

end ratio_charliz_jose_guppies_l380_380938


namespace slope_equal_angles_l380_380237

-- Define the problem
theorem slope_equal_angles (k : ℝ) :
  (∀ (l1 l2 : ℝ), l1 = 1 ∧ l2 = 2 → (abs ((k - l1) / (1 + k * l1)) = abs ((l2 - k) / (1 + l2 * k)))) →
  (k = (1 + Real.sqrt 10) / 3 ∨ k = (1 - Real.sqrt 10) / 3) :=
by
  intros
  sorry

end slope_equal_angles_l380_380237


namespace complement_of_intersection_l380_380266

theorem complement_of_intersection (U M N : Set ℤ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 4}) (hN : N = {3, 4, 5}) :
   U \ (M ∩ N) = {1, 2, 3, 5} := by
   sorry

end complement_of_intersection_l380_380266


namespace chameleon_problem_l380_380345

-- Define the conditions
variables {cloudy_days sunny_days ΔB ΔA init_A init_B : ℕ}
variable increase_A_minus_B : ΔA - ΔB = 6
variable increase_B : ΔB = 5
variable cloudy_count : cloudy_days = 12
variable sunny_count : sunny_days = 18

-- Define the desired result
theorem chameleon_problem :
  ΔA = 11 := 
by 
  -- Proof omitted
  sorry

end chameleon_problem_l380_380345


namespace problem1_problem2_problem3_problem4_l380_380130

section
  variable (a b c d : Int)

  theorem problem1 : -27 + (-32) + (-8) + 72 = 5 := by
    sorry

  theorem problem2 : -4 - 2 * 32 + (-2 * 32) = -132 := by
    sorry

  theorem problem3 : (-48 : Int) / (-2 : Int)^3 - (-25 : Int) * (-4 : Int) + (-2 : Int)^3 = -102 := by
    sorry

  theorem problem4 : (-3 : Int)^2 - (3 / 2)^3 * (2 / 9) - 6 / (-(2 / 3))^3 = -12 := by
    sorry
end

end problem1_problem2_problem3_problem4_l380_380130


namespace count_pos_three_digit_ints_with_same_digits_l380_380948

-- Define a structure to encapsulate the conditions for a three-digit number less than 700 with at least two digits the same.
structure valid_int (n : ℕ) : Prop :=
  (three_digit : 100 ≤ n ∧ n < 700)
  (same_digits : ∃ d₁ d₂ d₃ : ℕ, ((100 * d₁ + 10 * d₂ + d₃ = n) ∧ (d₁ = d₂ ∨ d₂ = d₃ ∨ d₁ = d₃)))

-- The number of integers satisfying the conditions
def count_valid_ints : ℕ :=
  168

-- The theorem to prove
theorem count_pos_three_digit_ints_with_same_digits : 
  (∃ n, valid_int n) → 168 :=
by
  -- Since the proof is not required, we add sorry here.
  sorry

end count_pos_three_digit_ints_with_same_digits_l380_380948


namespace digit_150_of_17_div_70_l380_380032

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380032


namespace sum_of_digits_of_d_problem_correct_l380_380662

noncomputable def exchange_rate : ℚ := 13 / 9

def total_pesos_received(d : ℚ) : ℚ := exchange_rate * d

def remaining_pesos(d : ℚ) : ℚ := total_pesos_received(d) - 117

theorem sum_of_digits_of_d (d : ℕ) (hd : total_pesos_received (d:ℚ) = d + 117) :
  nat.digits 10 d = [2, 6, 4] :=
by 
  have h264 : d = 264 := sorry
  rw h264
  dsimp
  simp

theorem problem_correct (d : ℕ) (hd : total_pesos_received (d:ℚ) = d + 117) :
  nat.sum_digits 10 d = 12 :=
by 
  have h := sum_of_digits_of_d d hd
  rw [h, nat.sum_digits]
  dsimp
  simp

end sum_of_digits_of_d_problem_correct_l380_380662


namespace zoe_total_songs_l380_380439

def initial_songs : ℕ := 15
def deleted_songs : ℕ := 8
def added_songs : ℕ := 50

theorem zoe_total_songs : initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end zoe_total_songs_l380_380439


namespace sum_interior_edge_lengths_l380_380809

def frame_width := 1
def frame_area := 24
def outer_edge_length := 7

theorem sum_interior_edge_lengths :
  ∃ y : ℝ, (2 * y + 10 = frame_area) → (y = outer_edge_length) → (2 * (outer_edge_length - 2) = 20) :=
by
  exists outer_edge_length
  intros h1 h2
  linarith

end sum_interior_edge_lengths_l380_380809


namespace find_angle_B_l380_380634

noncomputable def angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∀ (h_tri : a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π),
  (cos B = √2 / 2) → B = π / 4

-- Defining the main theorem
theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A + B + C = π)
  (h5 : (c - b) / (√2 * c - b) = (sin A) / (sin B + sin C))
  : B = π / 4 :=
sorry

end find_angle_B_l380_380634


namespace intersection_points_in_circle_l380_380708

theorem intersection_points_in_circle :
  let polygons := [4, 5, 6, 9]
  let intersections (p q : ℕ) := 2 * min p q
  (list.sum (list.map (λ pair, intersections pair.fst pair.snd) 
    [(4, 5), (4, 6), (4, 9), (5, 6), (5, 9), (6, 9)])) = 56 :=
by
  sorry

end intersection_points_in_circle_l380_380708


namespace perpendicular_concur_l380_380673

open EuclideanGeometry

variables {A B C G D X Y Z : Point} (hABCacte: acute_triangle A B C)
(centroid_G: is_centroid G A B C)
(altitude_D: is_foot_of_altitude D A B C)
(gd_intersects_X: ray_intersects_circle GD X (circumcircle A B C))
(ag_intersects_Y: ag_intersects_nine_point_circle AG Y (nine_point_circle A B C))
(Z_is_tangent_midline_intersection: is_A_tangent_midline_intersection Z A (circumcircle A B C) (midline A B A C))
(perpendicular_from_Z: perpendicular_from Z euler_line)
(AX: passes_through A X)
(DY: passes_through D Y)

theorem perpendicular_concur :
  concurrent perpendicular_from_Z AX DY :=
sorry

end perpendicular_concur_l380_380673


namespace integer_count_abs_lt_4pi_l380_380940

theorem integer_count_abs_lt_4pi : 
  (Finset.card (Finset.filter (λ x : ℤ, |x| < 4 * Real.pi) (Finset.Icc (-100) 100))) = 25 :=
sorry

end integer_count_abs_lt_4pi_l380_380940


namespace total_seashells_found_l380_380996

-- Defining the conditions
def joan_daily_seashells : ℕ := 6
def jessica_daily_seashells : ℕ := 8
def length_of_vacation : ℕ := 7

-- Stating the theorem
theorem total_seashells_found : 
  (joan_daily_seashells + jessica_daily_seashells) * length_of_vacation = 98 :=
by
  sorry

end total_seashells_found_l380_380996


namespace sequence_transform_impossible_l380_380189

def det (a b c d : ℤ) : ℤ := a * d - b * c

theorem sequence_transform_impossible :
  let initial_seq := (1, 2, 3, 4)
  let target_seq := (3, 4, 5, 7)
  ∀ (n : ℤ), 
  ∀ (a b c d : ℤ), 
  (a, b, c, d) ∈ { 
    (c, d, a, b), 
    (b, a, d, c), 
    (a+n*c, b+n*d, c, d),
    (a+n*b, b, c+n*d, d)
  } → 
  ¬ (det a b c d = det 1 2 3 4 = 2) :=
  sorry

end sequence_transform_impossible_l380_380189


namespace min_value_expr_l380_380866

theorem min_value_expr : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 := 
sorry

end min_value_expr_l380_380866


namespace closest_integer_to_percentage_increase_l380_380718

def area (s : ℕ) : ℕ := s * s

def percentage_increase (A1 A2 : ℕ) : ℤ :=
  (((A1 - A2) : ℤ) * 100) / A2

theorem closest_integer_to_percentage_increase :
  let A1 := area 5
  let A2 := area 4
  let increase := percentage_increase A1 A2
  abs (increase - 56) < abs (increase - 55) ∧ abs (increase - 56) < abs (increase - 57) :=
by
  let A1 := area 5
  let A2 := area 4
  let increase := percentage_increase A1 A2
  have : abs (increase - 56) = abs (900 / 16 - 56), from sorry
  have : abs (900 / 16 - 56) < abs (900 / 16 - 55) ∧ abs (900 / 16 - 56) < abs (900 / 16 - 57), from sorry
  exact this

end closest_integer_to_percentage_increase_l380_380718


namespace tan_4050_undefined_l380_380136

noncomputable def tan_is_undefined (θ : ℝ) : Prop := Real.cos θ = 0

theorem tan_4050_undefined : tan_is_undefined (4050 * Real.pi / 180) :=
by
  -- Convert degrees to radians
  have : (4050 : ℝ) * Real.pi / 180 = 90 * Real.pi / 180 + 11 * 360 * Real.pi / 180 := by
    norm_num
  rw [this]
  -- Simplify the trigonometric identity
  have : 90 * Real.pi / 180 + 11 * 360 * Real.pi / 180 = Real.pi / 2 + 11 * 2 * Real.pi := by
    norm_num
    ring
  rw [this, Real.cos_add, Real.cos_pi_div_two]
  -- Cos pi/2 is zero, hence tan is undefined
  norm_num
  show 0 = 0
  rfl

end tan_4050_undefined_l380_380136


namespace compute_cube_sum_l380_380845

noncomputable def roots_are (a b c : ℝ) : Prop :=
    3 * a^3 - 2 * a^2 + 5 * a - 7 = 0 ∧
    3 * b^3 - 2 * b^2 + 5 * b - 7 = 0 ∧
    3 * c^3 - 2 * c^2 + 5 * c - 7 = 0

theorem compute_cube_sum (a b c : ℝ) (h1 : a + b + c = ⅔) (h2 : ab + bc + ca = ⅔) (h3 : abc = ⅔) (h : roots_are a b c) :
    a^3 + b^3 + c^3 = 137 / 27 :=
begin
    sorry,
end

end compute_cube_sum_l380_380845


namespace smaller_triangle_area_l380_380743

-- Define the original triangle's area
variable (T : ℝ)

-- Define the conditions
def original_triangle_area : ℝ := T
def segment_factor : ℝ := 1 / 3
def inside_triangle_scale_factor : ℝ := (2 / 3) * (2 / 3)

-- Assert the theorem statement
theorem smaller_triangle_area : 
  ∀ T : ℝ, 
  (∃ original_triangle_area : ℝ, original_triangle_area = T) → 
  (segment_factor = 1 / 3) → 
  (inside_triangle_scale_factor = (2 / 3) * (2 / 3)) →
  ∃ smaller_triangle_area : ℝ, smaller_triangle_area = (4 / 9) * T :=
by {
  intro T,
  intros,
  use (4 / 9) * T,
  sorry
}

end smaller_triangle_area_l380_380743


namespace number_of_possible_values_for_second_largest_element_l380_380093

theorem number_of_possible_values_for_second_largest_element (A : List ℕ) :
  A.length = 5 →
  (A.sum / 5 = 14) →
  (A.max' (by decide) - A.min' (by decide) = 16) →
  (A.mode = [8]) →
  A.nth_le 2 (by decide) = 8 →
  A.nth_le 3 (by decide) = 8 →
  (Finset.image (λ (A : List ℕ), A.nth_le 3 (by decide)) (generate_lists 5)).card = 6 :=
sorry

end number_of_possible_values_for_second_largest_element_l380_380093


namespace product_of_distances_eq_n_l380_380184

noncomputable def omega_k (n k : ℕ) (h : n > 1) : ℂ :=
  Complex.exp (2 * Real.pi * Complex.I * k / n)

noncomputable def product_of_distances (n : ℕ) (h : n > 1) : ℝ :=
  ∏ k in Finset.range (n - 1), Complex.abs (1 - omega_k n (k + 1) h)

theorem product_of_distances_eq_n (n : ℕ) (h : n > 1) : product_of_distances n h = n := by
  sorry

end product_of_distances_eq_n_l380_380184


namespace det_C_n_l380_380869

/--
  For \(n \geq 3\), let \((b₀, b₁, \ldots, b_{n-1}) = (1, 1, 1, 0, \ldots, 0)\). 
  Let \(C_n = (c_{i,j})_{i,j}\) be the \(n \times n\) matrix defined by \(c_{i,j} = b_{(j-i) \mod n}\).
  The determinant of \(C_n\) is 3 if \(n \mod 3 \neq 0\) and 0 if \(n \mod 3 = 0\).
-/
theorem det_C_n (n : ℕ) (h : n ≥ 3) :
  let b : Fin n → ℕ := λ i, if i < 3 then 1 else 0,
      C : Matrix (Fin n) (Fin n) ℕ := λ i j, b ((j - i) % n)
  in if n % 3 = 0 then det C = 0 else det C = 3 :=
by
  sorry

end det_C_n_l380_380869


namespace digit_150_in_17_div_70_l380_380010

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l380_380010


namespace ellipse_equation_length_of_MN_range_of_y0_l380_380586

-- Condition: The ellipse C has the right focus F(1, 0)
def focus_F : ℝ × ℝ := (1, 0)

-- Condition: The point A(2, 0) is on the ellipse C
def point_A : ℝ × ℝ := (2, 0)

-- Line l passing through point F intersects the ellipse C at two distinct points M and N
-- We will define the slope of the line l for part (II)
def slope_l : ℝ := 1

-- Define the equation of the ellipse C
def equation_of_ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

-- Part (I) Prove the equation of the ellipse (C)
theorem ellipse_equation :
  equation_of_ellipse (point_A.1) (point_A.2) :=
sorry

-- Part (II) Prove the length of the line segment MN
noncomputable def length_MN :=
  (24 : ℝ) / 7

theorem length_of_MN :
  length_MN = 24 / 7 :=
sorry

-- Part (III) Prove the range of y0 where the perpendicular bisector intersects the y-axis
def range_y0 (y0 : ℝ) := 
  -real.sqrt(3) / 12 ≤ y0 ∧ y0 ≤ real.sqrt(3) / 12

theorem range_of_y0 (y0 : ℝ) :
  range_y0 y0 :=
sorry

end ellipse_equation_length_of_MN_range_of_y0_l380_380586


namespace complex_multiplication_l380_380084

theorem complex_multiplication : (1 - (complex.I))^2 * (complex.I) = 2 := 
by 
  sorry

end complex_multiplication_l380_380084


namespace ms_warren_walking_time_l380_380693

/-- 
Ms. Warren ran at 6 mph for 20 minutes. After the run, 
she walked at 2 mph for a certain amount of time. 
She ran and walked a total of 3 miles.
-/
def time_spent_walking (running_speed walking_speed : ℕ) (running_time_minutes : ℕ) (total_distance : ℕ) : ℕ := 
  let running_time_hours := running_time_minutes / 60;
  let distance_ran := running_speed * running_time_hours;
  let distance_walked := total_distance - distance_ran;
  let time_walked_hours := distance_walked / walking_speed;
  time_walked_hours * 60

theorem ms_warren_walking_time :
  time_spent_walking 6 2 20 3 = 30 :=
by
  sorry

end ms_warren_walking_time_l380_380693


namespace match_S_U_round_l380_380968

-- Define the rounds in which matches are played
def Round := ℕ

-- Define teams as a type
inductive Team
| P | Q | R | S | T | U

open Team

-- Match schedule for team P
def match_schedule_P (r : Round) : Prop :=
  (r = 1 → ∃ t : Team, t = Q) ∧
  (r = 3 → ∃ t : Team, t = T) ∧
  (r = 5 → ∃ t : Team, t = R)

-- Match schedule for team U
def match_schedule_U (r : Round) : Prop :=
  r = 4 → ∃ t : Team, t = T

-- Prove that the match between S and U is in round 1
theorem match_S_U_round :
  ∀ (r : Round), (match_schedule_P r ∧ match_schedule_U r) → r = 1 :=
sorry

end match_S_U_round_l380_380968


namespace minimize_B_finite_l380_380600

noncomputable def solution_set_A (k : ℝ) : set ℝ :=
if k < 0 then {x : ℝ | (k / 4 + 9 / (4 * k) + 3 < x) ∧ (x < 11 / 2)}
else if k = 0 then {x : ℝ | x < 11 / 2}
else if (0 < k ∧ k < 1) ∨ k > 9 then {x : ℝ | (x < 11 / 2) ∨ (x > k / 4 + 9 / (4 * k) + 3)}
else {x : ℝ | (x < k / 4 + 9 / (4 * k) + 3) ∨ (x > 11 / 2)}

def solution_set_B (k : ℝ) : set ℤ :=
{n : ℤ | n ∈ (solution_set_A k) ∧ n ∈ set.univ}

theorem minimize_B_finite (k : ℝ) (finite_B : set.finite (solution_set_B k)) :
  k < 0 ∧ solution_set_B k = {2, 3, 4, 5} :=
sorry

end minimize_B_finite_l380_380600


namespace λ_is_4_div_9_l380_380936

theorem λ_is_4_div_9
  (a b : ℝ × ℝ)
  (λ : ℝ)
  (h1 : a = (2, 0))
  (h2 : b = (0, 3))
  (h_orth : ((λ • b) - a) ⊥ (a + b)) :
  λ = (4 / 9) :=
by sorry

end λ_is_4_div_9_l380_380936


namespace jogger_ahead_distance_l380_380802

-- Definitions of conditions
def jogger_speed : ℝ := 9  -- km/hr
def train_speed : ℝ := 45  -- km/hr
def train_length : ℝ := 150  -- meters
def passing_time : ℝ := 39  -- seconds

-- The main statement that we want to prove
theorem jogger_ahead_distance : 
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)  -- conversion to m/s
  let distance_covered := relative_speed * passing_time
  let jogger_ahead := distance_covered - train_length
  jogger_ahead = 240 :=
by
  sorry

end jogger_ahead_distance_l380_380802


namespace ladybugs_count_l380_380274

-- Definitions for the conditions
def ladybug_spots (spots : Nat) (truth : Bool) : Prop :=
  (truth ↔ spots = 6) ∧ (¬truth ↔ spots = 4)

def ladybug_statements (spots : Nat → Prop) (statements : List Prop) : Prop :=
  ∀ i j k : Nat, i < j ∧ j < k ∧ k < statements.length →
    (statements.get i = ∀ x, spots x → x = i) ∧
    (statements.get j = ∑ i in List.range statements.length, i = 30) ∧
    (statements.get k = ∑ i in List.range statements.length, i = 26)

def remaining_ladybugs_check (statements : List Prop) : Prop :=
  ∀ l, l ∈ statements.drop 3 → l = (statements[0] ⊕ statements[1] ⊕ statements[2])

-- Main theorem
theorem ladybugs_count (n : Nat) 
  (spots : Nat → Prop)
  (statements : List Prop) :
  (ladybug_spots spots true ∨ ladybug_spots spots false) ∧
  ladybug_statements spots statements ∧
  remaining_ladybugs_check statements →
  n = 5 := sorry

end ladybugs_count_l380_380274


namespace integral_sin_minus_cos_l380_380159

open Real

-- Define the integrand.
def f (x : ℝ) : ℝ := sin x - cos x

-- State the theorem.
theorem integral_sin_minus_cos : ∫ x in 0..π, f x = 2 := by
  sorry

end integral_sin_minus_cos_l380_380159


namespace payment_per_mile_l380_380994

theorem payment_per_mile (miles_one_way : ℝ) (total_payment : ℝ) (total_miles_round_trip : ℝ) (payment_per_mile : ℝ) :
  miles_one_way = 400 → total_payment = 320 → total_miles_round_trip = miles_one_way * 2 →
  payment_per_mile = total_payment / total_miles_round_trip → payment_per_mile = 0.4 :=
by
  intros h_miles_one_way h_total_payment h_total_miles_round_trip h_payment_per_mile
  sorry

end payment_per_mile_l380_380994


namespace C1_C2_intersections_OAB_area_l380_380897

-- Definition of the parametric equation of curve C1
def C1 (θ : ℝ) : ℝ × ℝ :=
  (-1 + Real.cos θ, Real.sin θ)

-- Definition of the polar equation of curve C2
def C2 (θ : ℝ) : ℝ :=
  2 * Real.sin θ

-- Provide a lemma to check if a point is on curve C2 in Cartesian coordinates
lemma C2_cartesian (x y : ℝ) : x^2 + y^2 = 2 * y ↔ ∃ θ : ℝ, (x = 2 * Real.sin θ * Real.cos θ) ∧ (y = 2 * Real.sin θ * Real.sin θ) :=
sorry

-- Proof of intersection points of C1 and C2
theorem C1_C2_intersections : 
  ∀ θ₁ θ₂ : ℝ, C1 θ₁ = (0,0) → C1 θ₁ = C1 θ₂ → C2 θ₂ = (0,0) →
  C1 θ₁ = (-1,1) → C2_xy = (-1,1) → 
  (((-1 + Real.cos θ₁ = 0) ∧ (Real.sin θ₁ = 0)) ∨ ((-1 + Real.cos θ₁ = -1) ∧ (Real.sin θ₁ = 1))) :=
sorry

-- Proof for the area of triangle OAB where O is the origin, and A and B are on curves C1 and C2 respectively
theorem OAB_area (A B : ℝ × ℝ) (AB_max : |A.1 - B.1| = 2 + Real.sqrt 2) : 
  A ∈ {p:ℝ×ℝ | ∃ θ, p = C1 θ} → B ∈ {p:ℝ×ℝ | ∃ θ, p = C2 θ} → 
  ∃ (S : ℝ), S = (Real.sqrt 2 + 1)/2 :=
sorry

end C1_C2_intersections_OAB_area_l380_380897


namespace arith_sign_change_geo_sign_change_l380_380648

-- Definitions for sequences
def arith_sequence (a₁ d : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => arith_sequence a₁ d n + d

def geo_sequence (a₁ r : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => geo_sequence a₁ r n * r

-- Problem statement
theorem arith_sign_change :
  ∀ (a₁ d : ℝ), (∃ N : ℕ, arith_sequence a₁ d N = 0) ∨ (∀ n m : ℕ, (arith_sequence a₁ d n) * (arith_sequence a₁ d m) ≥ 0) :=
sorry

theorem geo_sign_change :
  ∀ (a₁ r : ℝ), r < 0 → ∀ n : ℕ, (geo_sequence a₁ r n) * (geo_sequence a₁ r (n + 1)) < 0 :=
sorry

end arith_sign_change_geo_sign_change_l380_380648


namespace extreme_values_and_min_value_l380_380227

-- Function definition
def f (a b c x : ℝ) : ℝ := 2*x^3 + 3*a*x^2 + 3*b*x + c

-- Conditions and problem statement
theorem extreme_values_and_min_value
  (c : ℝ)
  (H1 : ∀ x, (f (-3) 4 c x).deriv = 6*x^2 - 18*x + 12)
  (H2 : (f (-3) 4 c 1 = 9))
  (H3 : ∀ x, x ∈ set.Icc (-1 : ℝ) 2 → 
    (f (-3) 4 4 x ≤ f (-3) 4 4 1)) :
  ∃ c, c = 4 →
  ∀ x, f (-3) 4 4 x = -19 :=
begin
  sorry
end

end extreme_values_and_min_value_l380_380227


namespace new_equation_incorrect_l380_380498

-- Definition of a function to change each digit of a number by +1 or -1 randomly.
noncomputable def modify_digit (num : ℕ) : ℕ := sorry

-- Proposition stating the original problem's condition and conclusion.
theorem new_equation_incorrect (a b : ℕ) (c := a + b) (a' b' c' : ℕ)
    (h1 : a' = modify_digit a)
    (h2 : b' = modify_digit b)
    (h3 : c' = modify_digit c) :
    a' + b' ≠ c' :=
sorry

end new_equation_incorrect_l380_380498


namespace three_digit_integers_with_two_identical_digits_less_than_700_l380_380945

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def less_than_700 (n : ℕ) : Prop :=
  n < 700

def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.nodup = false

theorem three_digit_integers_with_two_identical_digits_less_than_700 : 
  ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_three_digit n ∧ less_than_700 n ∧ has_at_least_two_identical_digits n) ∧
  s.card = 156 := by
  sorry

end three_digit_integers_with_two_identical_digits_less_than_700_l380_380945


namespace lim_of_geom_series_l380_380895

open Complex

def first_term_in_geom_seq : ℕ := Nat.C(7, 3)

noncomputable def common_ratio_modulus : ℂ := (1 / (1 + Complex.I * Real.sqrt 3)).abs

noncomputable def infinite_geom_series_sum (a r : ℝ) : ℝ :=
  if abs r < 1 then a / (1 - r) else 0

theorem lim_of_geom_series :
  ∃ S : ℝ, (∀ n : ℕ, S = infinite_geom_series_sum 35 (1 / 2)) ∧ S = 70 :=
sorry

end lim_of_geom_series_l380_380895


namespace digit_150_of_17_div_70_l380_380029

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380029


namespace scientific_notation_of_190_million_l380_380716

theorem scientific_notation_of_190_million : (190000000 : ℝ) = 1.9 * 10^8 :=
sorry

end scientific_notation_of_190_million_l380_380716


namespace value_of_a_l380_380884

theorem value_of_a (a : ℝ) (f : ℝ → ℝ) (h₁ : f = λ x, a * x^3 + 3 * x^2 + 2) (h₂ : deriv f (-1) = 4) : a = 10 / 3 := by
  sorry

end value_of_a_l380_380884


namespace area_of_S_l380_380102

noncomputable def square := {z : ℂ | (Re z ≤ 1 ∧ Re z ≥ -1) ∧ (Im z ≤ 1 ∧ Im z ≥ -1)}
def region_outside_square := {z : ℂ | z ∉ square}
def S := {w : ℂ | ∃ z ∈ region_outside_square, w = z⁻¹}

theorem area_of_S : real.pi / 4 = ∫∫ (x y : ℝ) in {p : ℝ × ℝ | let z : ℂ := ⟨x, y⟩ in z ∈ S}, 1 :=
sorry

end area_of_S_l380_380102


namespace angle_BAC_is_120_degrees_l380_380288

theorem angle_BAC_is_120_degrees
 (A B C D : Type)
 (B_neq_C : B ≠ C)
 (D_on_BC : D ∈ line_through B C)
 (AD_bisects_BAC : angle_bisector (angle B A C) (line_through A D))
 (BD_eq_DC : segment_len B D = segment_len D C)
 (AD_eq_BC : segment_len A D = segment_len B C)
 : angle_measure (angle B A C) = 120 :=
begin
  sorry,
end

end angle_BAC_is_120_degrees_l380_380288


namespace find_integer_l380_380775

theorem find_integer (n : ℕ) (h1 : 0 < n) (h2 : 200 % n = 2) (h3 : 398 % n = 2) : n = 6 :=
sorry

end find_integer_l380_380775


namespace cream_cheese_cost_l380_380182

theorem cream_cheese_cost
  (B C : ℝ)
  (h1 : 2 * B + 3 * C = 12)
  (h2 : 4 * B + 2 * C = 14) :
  C = 2.5 :=
by
  sorry

end cream_cheese_cost_l380_380182


namespace tangent_value_of_angle_l380_380219

-- Given the unit circle equation
theorem tangent_value_of_angle (α : ℝ) (x : ℝ) (h1 : x^2 + (sqrt(3)/2)^2 = 1) :
  tan α = sqrt(3) ∨ tan α = -sqrt(3) :=
by
  sorry

end tangent_value_of_angle_l380_380219


namespace joan_spent_amount_l380_380997

theorem joan_spent_amount (trumpet_cost music_tool_cost song_book_cost : ℝ)
  (h_trumpet : trumpet_cost = 149.16)
  (h_music_tool : music_tool_cost = 9.98)
  (h_song_book : song_book_cost = 4.14) :
  trumpet_cost + music_tool_cost + song_book_cost = 163.28 :=
by
  rw [h_trumpet, h_music_tool, h_song_book]
  norm_num
  sorry

end joan_spent_amount_l380_380997


namespace hypotenuse_length_l380_380650

variable (u v : ℝ) (AX XB AY YC BY CX BC : ℝ)

-- Define conditions
def condition1 : Prop := 2 * u = AX ∧ u = XB
def condition2 : Prop := 2 * v = AY ∧ v = YC
def condition3 : Prop := BY = 25
def condition4 : Prop := CX = 35

-- Pythagorean theorem equations for sub-triangles
def equation1 : Prop := u^2 + (3 * v)^2 = 25^2
def equation2 : Prop := (3 * u)^2 + v^2 = 35^2

-- Final goal
def hypotenuse_def : Prop := BC = 3 * Real.sqrt (185)

theorem hypotenuse_length
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (eq1 : equation1)
  (eq2 : equation2)
  : hypotenuse_def :=
  sorry

end hypotenuse_length_l380_380650


namespace non_resident_ticket_price_l380_380087

theorem non_resident_ticket_price 
  (total_attendees : ℕ) (resident_price : ℝ) 
  (total_revenue : ℝ) (num_residents : ℕ) 
  (non_resident_price : ℝ) : 
  total_attendees = 586 ∧ 
  resident_price = 12.95 ∧ 
  total_revenue = 9423.70 ∧ 
  num_residents = 219 ∧ 
  non_resident_price = (total_revenue - (num_residents * resident_price)) / (total_attendees - num_residents) 
  → non_resident_price ≈ 17.95 := 
by {
  sorry
}

end non_resident_ticket_price_l380_380087


namespace determine_xyz_l380_380851

theorem determine_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h : x * y * z ≤ min (4 * (x - 1 / y)) (min (4 * (y - 1 / z)) (4 * (z - 1 / x)))) :
  x = y ∧ y = z ∧ z = sqrt 2 :=
begin
  sorry
end

end determine_xyz_l380_380851


namespace number_of_bricks_l380_380454

noncomputable def bricklayer_one_hours : ℝ := 8
noncomputable def bricklayer_two_hours : ℝ := 12
noncomputable def reduction_rate : ℝ := 12
noncomputable def combined_hours : ℝ := 6

theorem number_of_bricks (y : ℝ) :
  ((combined_hours * ((y / bricklayer_one_hours) + (y / bricklayer_two_hours) - reduction_rate)) = y) →
  y = 288 :=
by sorry

end number_of_bricks_l380_380454


namespace magician_marbles_l380_380461

theorem magician_marbles (red_init blue_init red_taken : ℕ) (h1 : red_init = 20) (h2 : blue_init = 30) (h3 : red_taken = 3) :
  let blue_taken := 4 * red_taken in
  let red_left := red_init - red_taken in
  let blue_left := blue_init - blue_taken in
  red_left + blue_left = 35 := by
  sorry

end magician_marbles_l380_380461


namespace spider_perspective_angle_l380_380740

/-- Define the degrees covered in one minute by the minute hand of a clock -/
def angle_per_minute : ℝ := 6

/-- Define the inscribed angle theorem -/
def inscribed_angle (central_angle: ℝ) : ℝ := central_angle / 2

/-- Prove the angle subtended from the spider's perspective per minute is 3 degrees -/
theorem spider_perspective_angle :
  inscribed_angle angle_per_minute = 3 := 
by 
  -- The proof is omitted as per the instructions.
  sorry

end spider_perspective_angle_l380_380740


namespace num_points_in_quadrants_l380_380930

-- Define sets M and N
def M : Set ℤ := {1, -2, 3}
def N : Set ℤ := {-4, 5, 6, -7}

-- Define a function to determine if a point (x, y) lies in the first or second quadrant
def inFirstOrSecondQuadrant (x y : ℤ) : Bool := (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)

-- Define a function to count the total number of points in the first and second quadrants
def countPointsInFirstOrSecondQuadrant (M N : Set ℤ) : Nat :=
  (∑ x in M, ∑ y in N, if inFirstOrSecondQuadrant x y then 1 else 0) +
  (∑ x in N, ∑ y in M, if inFirstOrSecondQuadrant x y then 1 else 0)

-- Main theorem
theorem num_points_in_quadrants : countPointsInFirstOrSecondQuadrant M N = 14 := by
  sorry

end num_points_in_quadrants_l380_380930


namespace find_smallest_value_wz_l380_380195

open Complex

noncomputable def w : ℂ := 0
noncomputable def z : ℂ := 0

theorem find_smallest_value_wz
  (hwz : |w + z| = 2)
  (hw2z2 : |w^2 + z^2| = 18) :
  ∃ K, K = |w^3 + z^3| ∧ K ≥ 50 :=
sorry

end find_smallest_value_wz_l380_380195


namespace green_chameleon_increase_l380_380338

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380338


namespace TriangleCountInRectangle_l380_380511

-- Define the structure of the problem
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)
  (vertical_lines_step : ℕ)
  (horizontal_lines_step : ℕ)

-- Define the specific rectangle for our problem
def givenRectangle : Rectangle := {
  width := 40,
  height := 10,
  vertical_lines_step := 10,
  horizontal_lines_step := 5
}

-- Statement of the problem in Lean
theorem TriangleCountInRectangle (R : Rectangle) (D : (nat * nat) → (nat * nat) → Prop) :
  R = givenRectangle →
  (∀ p1 p2, D p1 p2 → (p1 = (0, 0) ∧ p2 = (R.width, R.height)) ∨ (p1 = (R.width, 0) ∧ p2 = (0, R.height))) →
  ∃ n : ℕ, n = 74 :=
by sorry

end TriangleCountInRectangle_l380_380511


namespace digit_150_of_17_div_70_l380_380034

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380034


namespace composite_seq_minus_22_l380_380409

noncomputable theory

def is_composite (n : ℕ) := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ p * q = n

def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) * a n + 1

theorem composite_seq_minus_22 (a : ℕ → ℕ) (h : sequence a) :
  ∀ n > 10, is_composite (a n - 22) :=
by
  sorry

end composite_seq_minus_22_l380_380409


namespace geometric_conclusions_l380_380113

theorem geometric_conclusions :
    (¬ (∀ a b, (internal_angle_bisectors_parallel a b))) ∧
    (∀ p l, is_perpendicular_segment_shortest p l) ∧
    (∀ a b c d, a ⊥ b ∧ b ∥ c ∧ c ⊥ d → a ∥ d) ∧
    (¬ (∀ t1 t2, have_equal_midsegments_on_two_sides_and_third_side t1 t2 → congruent t1 t2))
    :=
by sorry

end geometric_conclusions_l380_380113


namespace repeating_decimal_to_fraction_l380_380162

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end repeating_decimal_to_fraction_l380_380162


namespace maximum_area_ratio_of_ellipse_l380_380915

theorem maximum_area_ratio_of_ellipse (a b : ℝ) (h_ab : a > b) (h_b : b > 0) 
    (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) (h_P : (P.1^2 / a^2 + P.2^2 / b^2 = 1)) :
    ∃ (S₁ S₂ : ℝ), S₁ = |PD| * |DN| ∧ |DM| = |DN| = |PD| * sin (θ / 2) ∧
    |F₁ - F₂|^2 = (|PF₁| + |PF₂|)^2 - 2 * |PF₁| * |PF₂| (1 + cos θ) ∧
    maximum_value (S_triangle_DMN / S_triangle_PF₁F₂) = b^2 * c^2 / a^4 :=
sorry

end maximum_area_ratio_of_ellipse_l380_380915


namespace digit_150th_in_decimal_of_fraction_l380_380050

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380050


namespace domain_range_equal_l380_380853

noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

theorem domain_range_equal {a b : ℝ} (hb : b > 0) :
  (∀ y, ∃ x, f a b x = y) ↔ (a = -4 ∨ a = 0) :=
sorry

end domain_range_equal_l380_380853


namespace square_area_of_adjacent_vertices_l380_380204

open Real EuclideanSpace

noncomputable def distance (p q : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  eudist p q

def is_square_area_correct (p1 p2 : EuclideanSpace ℝ (Fin 2)) (d : ℝ) : Prop :=
  let side_length := distance p1 p2
  side_length = d ∧ d * d = 36

theorem square_area_of_adjacent_vertices :
  ∀ (p1 p2 : EuclideanSpace ℝ (Fin 2)), 
  p1 = ![-2,3] → p2 = ![4,3] → is_square_area_correct p1 p2 6 :=
by
  intros
  rw [dist_eq]
  sorry

end square_area_of_adjacent_vertices_l380_380204


namespace find_a_b_find_extreme_values_l380_380233

-- Definitions based on the conditions in the problem
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 2 * b

-- The function f attains a maximum value of 2 at x = -1
def f_max_at_neg_1 (a b : ℝ) : Prop :=
  (∃ x : ℝ, x = -1 ∧ 
  (∀ y : ℝ, f x a b ≤ f y a b)) ∧ f (-1) a b = 2

-- Statement (1): Finding the values of a and b
theorem find_a_b : ∃ a b : ℝ, f_max_at_neg_1 a b ∧ a = 2 ∧ b = 1 :=
sorry

-- The function f with a=2 and b=1
def f_specific (x : ℝ) : ℝ := f x 2 1

-- Statement (2): Finding the extreme values of f(x) on the interval [-1, 1]
def extreme_values_on_interval : Prop :=
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_specific x ≤ 6 ∧ f_specific x ≥ 50/27) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 6) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 50/27)

theorem find_extreme_values : extreme_values_on_interval :=
sorry

end find_a_b_find_extreme_values_l380_380233


namespace digit_150_of_17_div_70_is_2_l380_380011

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380011


namespace clock_angle_15_15_l380_380127

theorem clock_angle_15_15 :
  let hour_angle := 90 + 15 * 0.5,
      minute_angle := 15 * 6 in
  |hour_angle - minute_angle| = 7.5 :=
by
  let hour_angle := 90 + 15 * 0.5;
  let minute_angle := 15 * 6;
  sorry

end clock_angle_15_15_l380_380127


namespace max_height_reached_l380_380794

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_reached : ∃ t : ℝ, h t = 161 :=
by
  sorry

end max_height_reached_l380_380794


namespace percentage_problem_l380_380808

theorem percentage_problem
    (x : ℕ) (h1 : (x:ℝ) / 100 * 20 = 8) :
    x = 40 :=
by
    sorry

end percentage_problem_l380_380808


namespace green_chameleon_increase_l380_380327

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380327


namespace find_a_b_and_prove_f_minus_g_gt_2_l380_380226

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a - b * x^3) * Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem find_a_b_and_prove_f_minus_g_gt_2 :
  (a b : ℝ) (h1 : f a b 1 = Real.exp 1) 
           (h2 : (f a b)' 1 = -2 * Real.exp 1) :
  -- Part (1): Finding a and b
  a = 2 ∧ b = 1 ∧ 
  -- Part (2): Proving inequality
  (∀ x : ℝ, 0 < x ∧ x < 1 → f 2 1 x - g x > 2) :=
by
  sorry

end find_a_b_and_prove_f_minus_g_gt_2_l380_380226


namespace hyperbola_eccentricity_sqrt6_l380_380847

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : 4 * b * b + a * a = 6 * a * a) : ℝ :=
  c / a

theorem hyperbola_eccentricity_sqrt6 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : 4 * b * b + a * a = 6 * a * a) (f1 f2 : ℝ) :
    hyperbola_eccentricity a b c ha hb hc = sqrt (6) :=
sorry

end hyperbola_eccentricity_sqrt6_l380_380847


namespace cube_has_12_edges_l380_380251

-- Definition of the number of edges in a cube
def number_of_edges_of_cube : Nat := 12

-- The theorem that asserts the cube has 12 edges
theorem cube_has_12_edges : number_of_edges_of_cube = 12 := by
  -- proof to be filled later
  sorry

end cube_has_12_edges_l380_380251


namespace digit_150_of_17_div_70_is_2_l380_380016

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380016


namespace a_seq_increasing_and_bounded_l380_380308

def f (x : ℝ) := x - x * real.log x

lemma f_increasing_on_0_1 : ∀ x, 0 < x ∧ x < 1 → 0 < (derev.f x) :=
begin
  intro x,
  intro hx,
  have h_deriv : real.diff (λ x, x - x * real.log x) x = 1 - real.log x - 1 := sorry,
  rw h_deriv,
  exact sorry,
end

sequence {a_n} : ℕ → ℝ
def a_1 := sorry,
def a_satisfies (n: ℕ) : evaluating exact sequence {a_}
{
  sorry
}

theorem a_seq_increasing_and_bounded : ∀ n, a_n < a_{n+1} ∧ a_{n+1} < 1 :=
begin
  intros n,
  induction n with k ih,
  {
    sorry -- Base case
  },
  {
    sorry -- Inductive step
  }
end

end a_seq_increasing_and_bounded_l380_380308


namespace digit_150_of_17_div_70_is_2_l380_380018

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l380_380018


namespace digit_150_of_17_div_70_l380_380028

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380028


namespace fraction_simplification_l380_380066

theorem fraction_simplification (a b x : ℕ) (hx1 : x ≠ 1) (ha : 2 * a + 1 ≠ 2) :
  (∃ a b, ∀ x, (4 / (6 * a * b)) = (2 / (3 * a * b)) ∧ 
    (x / (x - 1)) = (x / (x - 1)) ∧ 
    (2 / (2 * a + 1)) = (2 / (2 * a + 1)) ∧ 
    ((x^2 + 1) / (2 * x^2)) = ((x^2 + 1) / (2 * x^2))) :=
begin
  sorry
end

end fraction_simplification_l380_380066


namespace part_a_part_b_l380_380078

theorem part_a (S : Circle) (A B O P Q : Point) :
  O = center S → P ∈ S → Q ∈ S → ∃ P Q, P ≠ Q ∧ (line_through A B).intersect_circle S = {P, Q} :=
by sorry

theorem part_b (A1 B1 A2 B2 : Point) :
  ∃ X, X = intersection_point (line_through A1 B1) (line_through A2 B2) :=
by sorry

end part_a_part_b_l380_380078


namespace multiply_and_simplify_l380_380840

variable (a b : ℝ)

theorem multiply_and_simplify :
  (3 * a + 2 * b) * (a - 2 * b) = 3 * a^2 - 4 * a * b - 4 * b^2 :=
by
  sorry

end multiply_and_simplify_l380_380840


namespace odd_square_sum_of_consec_b_l380_380874

-- Define sequences a_n and b_n
def a_sequence : ℕ → ℕ := λ n, if n % 4 = 0 ∨ n % 4 = 1 then 0 else n

def a_pos_sequence : ℕ → ℕ
| 0 := a_sequence 0
| (n+1) := let m := (a_pos_sequence n) + 1 in if a_sequence m = 0 then a_pos_sequence (n + 1) else m

def b_sequence : ℕ → ℕ
| n := let k := a_pos_sequence n in if k % 4 = 0 ∨ k % 4 = 1 then 0 else a_pos_sequence n

def b_pos_sequence : ℕ → ℕ
| 0 := b_sequence 0
| (n+1) := let m := (b_pos_sequence n) + 1 in if b_sequence m = 0 then b_pos_sequence (n + 1) else m

-- Prove that every odd square greater than 1 is the sum of two consecutive terms in the sequence b_n
theorem odd_square_sum_of_consec_b (k : ℕ) (h : k ≥ 1) : ∃ n : ℕ, (2 * k + 1)^2 = b_pos_sequence n + b_pos_sequence (n + 1) := by
  sorry

end odd_square_sum_of_consec_b_l380_380874


namespace circular_garden_remaining_grass_area_l380_380451

noncomputable def remaining_grass_area (diameter : ℝ) (path_width: ℝ) : ℝ :=
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let path_area := path_width * diameter
  circle_area - path_area

theorem circular_garden_remaining_grass_area :
  remaining_grass_area 10 2 = 25 * Real.pi - 20 := sorry

end circular_garden_remaining_grass_area_l380_380451


namespace orthocenter_ratio_tan_l380_380660

/-- In triangle ABC, H is the orthocenter. Prove that a / AH = tan(A). -/
theorem orthocenter_ratio_tan {A B C H : Type} [Real : Type] 
  (a AH : Real) (tan A : Real)
  (h1 : H is the orthocenter of the triangle ABC)
  (h2 : a = 2 * R * sin A)
  (h3 : AH = 2 * R * cos A) :
  a / AH = tan A :=
sorry

end orthocenter_ratio_tan_l380_380660


namespace solution_A_solution_B_solution_D_l380_380888

namespace Proof

variable {f : ℝ → ℝ} 
variable {f' : ℝ → ℝ}
variable h_deriv : ∀ x ∈ Ioo 0 (π / 2), f' x * cos x > f x * sin x

theorem solution_A (x y : ℝ) (hx : x = π / 3) (hy : y = π / 4) :
  f x > sqrt 2 * f y :=
sorry

theorem solution_B (x y : ℝ) (hx : x = π / 4) (hy : y = π / 6) :
  2 * f x > sqrt 6 * f y :=
sorry

theorem solution_D (x : ℝ) (h : x = π / 3) :
  f x > 2 * cos 1 * f 1 :=
sorry

end Proof

end solution_A_solution_B_solution_D_l380_380888


namespace ratio_of_sums_of_geometric_sequence_l380_380958

theorem ratio_of_sums_of_geometric_sequence (a : ℤ) :
  let seq := [a, -2*a, 4*a, -8*a, 16*a, -32*a, 64*a] in
  (seq[1] + seq[3] + seq[5]) / (seq[0] + seq[2] + seq[4] + seq[6]) = (-42 : ℚ) / 85 :=
by
  -- Definitions
  let seq := [a, -2*a, 4*a, -8*a, 16*a, -32*a, 64*a]
  -- Proof
  sorry

end ratio_of_sums_of_geometric_sequence_l380_380958


namespace correct_option_exponent_equality_l380_380779

theorem correct_option_exponent_equality (a b : ℕ) : 
  (\left(2 * a * b^2\right)^2 = 4 * a^2 * b^4) :=
by
  sorry

end correct_option_exponent_equality_l380_380779


namespace rate_of_mixed_oil_l380_380621

theorem rate_of_mixed_oil (V1 V2 : ℝ) (P1 P2 : ℝ) : 
  (V1 = 10) → 
  (P1 = 50) → 
  (V2 = 5) → 
  (P2 = 67) → 
  ((V1 * P1 + V2 * P2) / (V1 + V2) = 55.67) :=
by
  intros V1_eq P1_eq V2_eq P2_eq
  rw [V1_eq, P1_eq, V2_eq, P2_eq]
  norm_num
  sorry

end rate_of_mixed_oil_l380_380621


namespace max_projection_area_tetrahedron_l380_380157

-- Define the side length of the tetrahedron
variable (a : ℝ)

-- Define a theorem stating the maximum projection area of a tetrahedron
theorem max_projection_area_tetrahedron (h : a > 0) : 
  ∃ A, A = (a^2 / 2) :=
by
  -- Proof is omitted
  sorry

end max_projection_area_tetrahedron_l380_380157


namespace triangular_region_area_l380_380482

def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def y (x : ℝ) := x

theorem triangular_region_area : 
  ∀ (x y: ℝ),
  (y = line 1 2 x ∧ y = 3) ∨ 
  (y = line (-1) 8 x ∧ y = 3) ∨ 
  (y = line 1 2 x ∧ y = line (-1) 8 x)
  →
  ∃ (area: ℝ), area = 4.00 := 
by
  sorry

end triangular_region_area_l380_380482


namespace square_area_l380_380747

theorem square_area (x : ℝ) (s1 s2 area : ℝ) 
  (h1 : s1 = 5 * x - 21) 
  (h2 : s2 = 36 - 4 * x) 
  (hs : s1 = s2)
  (ha : area = s1 * s1) : 
  area = 113.4225 := 
by
  -- Proof goes here
  sorry

end square_area_l380_380747


namespace range_of_a_l380_380578

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x - 3 > 0 → x > a) ↔ a ≥ 1 :=
by
  sorry

end range_of_a_l380_380578


namespace cosine_angle_between_skew_lines_l380_380263

theorem cosine_angle_between_skew_lines :
  let a := (0, -2, -1) : ℝ × ℝ × ℝ
  let b := (2, 0, 4) : ℝ × ℝ × ℝ
  real.abs (real.cosine (a, b)) = 2 / 5 := 
by
  sorry

end cosine_angle_between_skew_lines_l380_380263


namespace indigo_restaurant_total_reviews_l380_380717

-- Define the number of 5-star reviews
def five_star_reviews : Nat := 6

-- Define the number of 4-star reviews
def four_star_reviews : Nat := 7

-- Define the number of 3-star reviews
def three_star_reviews : Nat := 4

-- Define the number of 2-star reviews
def two_star_reviews : Nat := 1

-- Define the total number of reviews
def total_reviews : Nat := five_star_reviews + four_star_reviews + three_star_reviews + two_star_reviews

-- Proof that the total number of customer reviews is 18
theorem indigo_restaurant_total_reviews : total_reviews = 18 :=
by
  -- Direct calculation
  sorry

end indigo_restaurant_total_reviews_l380_380717


namespace shirt_and_tie_outfits_l380_380713

theorem shirt_and_tie_outfits (shirts ties : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 6):
  shirts * ties = 48 :=
by
  rw [h_shirts, h_ties]
  simp
  exact rfl

end shirt_and_tie_outfits_l380_380713


namespace smallest_digit_sum_l380_380295

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

def is_valid_pair (m n : ℕ) : Prop :=
  m < 50 ∧ n < 50 ∧ m > 9 ∧ n > 9 ∧ (m * n / 1000) > 0 ∧ (m * n / 10000) = 0 ∧ 
  (function.injective (λ x : ℕ, (m / 10, m % 10, n / 10, n % 10)) : Prop)

theorem smallest_digit_sum : 
  ∃ m n : ℕ, is_valid_pair m n ∧ sum_of_digits (m * n) = 20 :=
by
  sorry

end smallest_digit_sum_l380_380295


namespace car_catches_up_in_4_hours_l380_380098

noncomputable def motorcycle_speed (x : ℕ) := 2 * x
noncomputable def car_speed (x : ℕ) := 3 * x
noncomputable def motorcycle_distance_in_7_hours (x : ℕ) := 7 * motorcycle_speed x
noncomputable def car_distance_in_6_hours (x : ℕ) := 6 * car_speed x

theorem car_catches_up_in_4_hours (x : ℕ) 
  (h1 : 120 = motorcycle_speed x * t) 
  (h2 : 180 = car_speed x * t)
  (h3 : motorcycle_distance_in_7_hours x = car_distance_in_6_hours x - 80) :
  let motorcycle_start_time := 0
      car_start_time := 2
      motorcycle_lead_distance := 80
      relative_speed := car_speed x - motorcycle_speed x in
  car_start_time + motorcycle_lead_distance / relative_speed = 4 :=
by
  sorry

end car_catches_up_in_4_hours_l380_380098


namespace six_nonzero_digits_l380_380368

theorem six_nonzero_digits (k : ℕ) (h : 10101010101 ∣ k) : 
  (k.to_string.filter (≠ '0')).length ≥ 6 := 
sorry

end six_nonzero_digits_l380_380368


namespace algebraic_expression_value_set_l380_380580

theorem algebraic_expression_value_set (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  { (frac x (abs x) + frac y (abs y) + frac z (abs z) + abs (x * y * z) / (x * y * z)) } = {0, 4, -4} :=
sorry

end algebraic_expression_value_set_l380_380580


namespace affine_transform_equal_length_iff_triangle_ineq_l380_380700

-- Define type for Vectors
variables {V : Type*} [add_comm_group V] [module ℝ V]

-- Definition of conditions
variables (a b c : V) (α β γ : ℝ)
hypothesis h : α • a + β • b + γ • c = 0

-- Statement of the Theorem
theorem affine_transform_equal_length_iff_triangle_ineq :
  (∃ f : V → V, ∀ (a' b' c' : V), f a = a' ∧ f b = b' ∧ f c = c' ∧ (∥a'∥ = ∥b'∥ ∧ ∥b'∥ = ∥c'∥)) ↔
  (|α| + |β| > |γ| ∧ |α| + |γ| > |β| ∧ |β| + |γ| > |α|) :=
sorry

end affine_transform_equal_length_iff_triangle_ineq_l380_380700


namespace curve_hyperbola_focal_length_l380_380726

theorem curve_hyperbola_focal_length (λ : ℝ) (h : λ < -1) :
  let C := { p : ℝ × ℝ | p.1^2 + p.2^2 / λ = 4 }
  in ∃ f : ℝ, f = 4 * Real.sqrt (1 - λ) ∧
     ∀ p ∈ C, 
       (C (abs (p.1))) = true ∧
       (C (abs (p.2))) = true :=
by sorry

end curve_hyperbola_focal_length_l380_380726


namespace digit_150_of_17_div_70_l380_380033

theorem digit_150_of_17_div_70 : (decimal_digit 150 (17 / 70)) = 7 :=
sorry

end digit_150_of_17_div_70_l380_380033


namespace find_total_students_surveyed_l380_380372

noncomputable def total_students_surveyed (x y : ℕ) : Prop :=
  (0.75 * y = x) ∧ (0.523 * x = 49)

theorem find_total_students_surveyed (x y : ℕ) (h : total_students_surveyed x y) : y = 125 :=
by
  sorry

end find_total_students_surveyed_l380_380372


namespace solution_A_solution_B_solution_D_l380_380889

namespace Proof

variable {f : ℝ → ℝ} 
variable {f' : ℝ → ℝ}
variable h_deriv : ∀ x ∈ Ioo 0 (π / 2), f' x * cos x > f x * sin x

theorem solution_A (x y : ℝ) (hx : x = π / 3) (hy : y = π / 4) :
  f x > sqrt 2 * f y :=
sorry

theorem solution_B (x y : ℝ) (hx : x = π / 4) (hy : y = π / 6) :
  2 * f x > sqrt 6 * f y :=
sorry

theorem solution_D (x : ℝ) (h : x = π / 3) :
  f x > 2 * cos 1 * f 1 :=
sorry

end Proof

end solution_A_solution_B_solution_D_l380_380889


namespace ratio_albert_betty_l380_380817

noncomputable def albert_age (M : ℕ) : ℕ := 2 * M
noncomputable def mary_age (A : ℕ) : ℕ := A - 8
def betty_age : ℕ := 4

theorem ratio_albert_betty (A M : ℕ) (h1 : A = 2 * M) (h2 : M = A - 8) (h3 : betty_age = 4) : A / betty_age = 4 :=
by
    have hA : A = 16 := by sorry
    rw [hA, betty_age]
    norm_num

end ratio_albert_betty_l380_380817


namespace value_of_f_2012_1_l380_380560

noncomputable def f : ℝ → ℝ :=
sorry

-- Condition 1: f is even
axiom even_f : ∀ x : ℝ, f x = f (-x)

-- Condition 2: f(x + 3) = -f(x)
axiom periodicity_f : ∀ x : ℝ, f (x + 3) = -f x

-- Condition 3: f(x) = 2x + 3 for -3 ≤ x ≤ 0
axiom defined_f_on_interval : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x = 2 * x + 3

-- Assertion to prove
theorem value_of_f_2012_1 : f 2012.1 = -1.2 :=
by sorry

end value_of_f_2012_1_l380_380560


namespace ratio_area_A_to_C_l380_380375

noncomputable def side_length (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area (side : ℕ) : ℕ :=
  side * side

theorem ratio_area_A_to_C : 
  let A_perimeter := 16
  let B_perimeter := 40
  let C_perimeter := 2 * A_perimeter
  let side_A := side_length A_perimeter
  let side_C := side_length C_perimeter
  let area_A := area side_A
  let area_C := area side_C
  (area_A : ℚ) / area_C = 1 / 4 :=
by
  sorry

end ratio_area_A_to_C_l380_380375


namespace hexagon_area_ratio_l380_380298

theorem hexagon_area_ratio (m n : ℕ) (hp : Nat.Coprime m n) :
    ∃ (s : ℝ), (∃ (a b c d e f g h i j k l : Point),
    is_regular_hexagon s a b c d e f ∧
    are_midpoints a b c d e f g h i j k l ∧
    form_smaller_hexagon g h i j k l) →
    (m = 3 ∧ n = 4 ∧ m + n = 7) :=
begin
  sorry
end

end hexagon_area_ratio_l380_380298


namespace digit_150_is_7_l380_380038

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380038


namespace missing_jar_size_l380_380296

theorem missing_jar_size (total_ounces jars_16 jars_28 jars_unknown m n p: ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : p = 3)
    (total_jars : m + n + p = 9)
    (total_peanut_butter : 16 * m + 28 * n + jars_unknown * p = 252)
    : jars_unknown = 40 := by
  sorry

end missing_jar_size_l380_380296


namespace is_largest_prime_divisor_needed_l380_380267

noncomputable def largest_prime_divisor_needed (N : ℕ) (h1 : 1000 ≤ N) (h2 : N ≤ 1100) : ℕ :=
  let sqrt_N := real.sqrt N
  let primes_up_to_sqrt := Nat.list_primes (⌊sqrt_N⌋ + 1)
  primes_up_to_sqrt.last'

theorem is_largest_prime_divisor_needed : largest_prime_divisor_needed 1100 (by norm_num) (by norm_num) = 31 :=
by
  sorry

end is_largest_prime_divisor_needed_l380_380267


namespace overall_average_output_correct_l380_380118

def order1 := 60
def rate1 := 15
def time1 := order1 / rate1

def order2 := 60
def rate2 := 60
def time2 := order2 / rate2

def order3 := 180
def rate3 := 90
def time3 := order3 / rate3

def order4 := 90
def rate4 := 45
def time4 := order4 / rate4

def total_cogs := order1 + order2 + order3 + order4
def total_time := time1 + time2 + time3 + time4
def average_output := total_cogs / total_time

theorem overall_average_output_correct : average_output = 43.33 :=
by
  sorry

end overall_average_output_correct_l380_380118


namespace smallest_n_satisfying_conditions_l380_380548

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, (n > 0) ∧ (∃ m : ℤ, n^2 = 3 * m^2 - 3 * m + 6) ∧ (∃ a : ℤ, 2 * n + 117 = a^2) ∧ 
  (∀ k : ℕ, (0 < k) ∧ (∃ m' : ℤ, k^2 = 3 * m'2 - 3 * m' + 6) ∧ (∃ a' : ℤ, 2 * k + 117 = a'^2) → n ≤ k) :=
begin
  use 145,
  split,
  { norm_num }, -- verifies that 145 > 0
  split,
  { use 26, norm_num }, -- verifies the first condition with m = 26
  split,
  { use 16, norm_num }, -- verifies the second condition with a = 16
  { intro k,
    assume h,
    cases h with hpos hk,
    cases hk with m_cond k_eq_sq,
    cases k_eq_sq with a_cond,
    sorry
  },
end

end smallest_n_satisfying_conditions_l380_380548


namespace green_chameleon_increase_l380_380332

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380332


namespace charles_travel_time_l380_380622

theorem charles_travel_time (D S T : ℕ) (hD : D = 6) (hS : S = 3) : T = D / S → T = 2 :=
by
  intros h
  rw [hD, hS] at h
  simp at h
  exact h

end charles_travel_time_l380_380622


namespace suzhou_tourism_revenue_scientific_notation_l380_380112

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
let a := x / (10 ^ (Int.natAbs (Int.ofNat (Math.log10 (Real.abs x) |>.toNat)))) in
let n := Int.ofNat (Math.log10 (Real.abs x) |>.toNat - 1) in
(a, n)

theorem suzhou_tourism_revenue_scientific_notation :
  scientific_notation (998.64 * 10^8) = (9.9864, 11) :=
by
  sorry

end suzhou_tourism_revenue_scientific_notation_l380_380112


namespace larger_root_of_quadratic_eq_l380_380535

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l380_380535


namespace repeating_decimal_to_fraction_l380_380163

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end repeating_decimal_to_fraction_l380_380163


namespace digit_150th_of_17_div_70_is_7_l380_380024

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380024


namespace linear_regression_equation_l380_380607

-- Define the conditions as Lean definitions
variable (x y : ℝ)
variable (mean_x mean_y : ℝ) (h_mean_x : mean_x = 3) (h_mean_y : mean_y = 4.5)
variable (negatively_correlated : ∀ x₁ x₂ y₁ y₂, (x₁ - x₂) * (y₁ - y₂) ≤ 0)

-- Define the statement that the correct regression line is y = -2x + 10.5
theorem linear_regression_equation : 
  negatively_correlated x mean_x mean_y mean_y →
  (mean_x = 3) →
  (mean_y = 4.5) →
  (∀ (x : ℝ), y = -2 * x + 10.5) :=
by
  intros h_corr h_x h_y
  sorry

end linear_regression_equation_l380_380607


namespace betsy_quilt_percentage_l380_380497

theorem betsy_quilt_percentage (total_squares_one_side : ℕ) (total_squares_other_side : ℕ) (squares_left : ℕ) :
  total_squares_one_side = 16 →
  total_squares_other_side = 16 →
  squares_left = 24 →
  (32 - 24) / 32 * 100 = 25 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end betsy_quilt_percentage_l380_380497


namespace evaluate_expression_l380_380158

theorem evaluate_expression : (125^(1/3 : ℝ)) * (64^(-1/2 : ℝ)) * (81^(1/4 : ℝ)) = (15 / 8) :=
by
  sorry

end evaluate_expression_l380_380158


namespace purely_imaginary_implication_l380_380962

noncomputable def complex_number (a b : ℝ) : ℂ := (a + b * complex.i) / (1 + complex.i)

theorem purely_imaginary_implication (a b : ℝ) (h : b ≠ 0) :
  (complex_number a b).re = 0 → a / b = -1 :=
by
  sorry

end purely_imaginary_implication_l380_380962


namespace total_copies_l380_380428

theorem total_copies (rate1 : ℕ) (rate2 : ℕ) (time : ℕ) (total : ℕ) 
  (h1 : rate1 = 25) (h2 : rate2 = 55) (h3 : time = 30) : 
  total = rate1 * time + rate2 * time := 
  sorry

end total_copies_l380_380428


namespace correct_judgments_l380_380395

-- Define the functions as per the problem statement.
def f1 (x : ℝ) : ℝ := abs x / x
def g1 (x : ℝ) : ℝ := if x ≥ 0 then 1 else -1

def f2 (x : ℝ) : ℝ := x^2 - 2*x + 1
def g2 (t : ℝ) : ℝ := t^2 - 2*t + 1

def f3 (x : ℝ) : ℝ := abs (x - 1) - abs x

theorem correct_judgments : 
  (f1 ≠ g1) ∧
  (∃ y, f1 1 = y) ∧ 
  (∀ y, (∃ z, f1 z = y ∧ z ≠ 1) → false) ∧
  (f2 = g2) ∧
  (f3 (f3 (1/2)) ≠ 0) :=
by
  sorry

end correct_judgments_l380_380395


namespace value_of_a_2011_l380_380593

def f (x : ℝ) := (3 * x) / (x + 3)

noncomputable def seq (a : ℕ → ℝ) : Prop := 
  a 1 = 1/2 ∧ ∀ n ≥ 2, a n = f (a (n - 1))

theorem value_of_a_2011 :
  ∃ a : ℕ → ℝ, seq a ∧ a 2011 = 1 / 672 := 
sorry

end value_of_a_2011_l380_380593


namespace middle_number_is_40_l380_380086

theorem middle_number_is_40 (A B C : ℕ) (h1 : C = 56) (h2 : C - A = 32) (h3 : B / C = 5 / 7) : B = 40 :=
  sorry

end middle_number_is_40_l380_380086


namespace increase_in_green_chameleons_is_11_l380_380320

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380320


namespace green_chameleon_increase_l380_380331

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end green_chameleon_increase_l380_380331


namespace two_pairs_of_dice_probability_l380_380772

noncomputable def two_pairs_probability : ℚ :=
  5 / 36

theorem two_pairs_of_dice_probability :
  ∃ p : ℚ, p = two_pairs_probability := 
by 
  use 5 / 36
  sorry

end two_pairs_of_dice_probability_l380_380772


namespace field_fence_length_l380_380469

theorem field_fence_length (L : ℝ) (A : ℝ) (W : ℝ) (fencing : ℝ) (hL : L = 20) (hA : A = 210) (hW : A = L * W) : 
  fencing = 2 * W + L → fencing = 41 :=
by
  rw [hL, hA] at hW
  sorry

end field_fence_length_l380_380469


namespace find_a_l380_380714

theorem find_a (b c : ℕ) (h₁ : b = 6) (h₂ : c = 8)
    (h₃ : ∃ k : ℕ, ∀ a b c : ℕ, a * b^2 = k * c)
    (h₄ : ∃ a : ℕ, a = 12 ∧ (λ k : ℕ, ∀ a b c : ℕ, a * b^2 = k * c) 3 4) :
    ∃ a : ℕ, a = 6 :=
by
  sorry

end find_a_l380_380714


namespace hyperbola_eccentricity_proof_l380_380460

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b = a * Real.sqrt 3) : ℝ :=
(let c := Real.sqrt (a^2 + b^2) in c / a)

theorem hyperbola_eccentricity_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : b = a * Real.sqrt 3) : hyperbola_eccentricity a b h1 h2 h3 = 2 := 
by
  -- Proof goes here, but is omitted for now
  sorry

end hyperbola_eccentricity_proof_l380_380460


namespace sum_first_42_odd_numbers_eq_l380_380421

theorem sum_first_42_odd_numbers_eq :
  (∑ i in Finset.range 42, (2 * i + 1)) = 1764 := 
by sorry

end sum_first_42_odd_numbers_eq_l380_380421


namespace sequence_a5_l380_380979

def sequence (n : ℕ) : ℕ → ℝ
| 0       := 1
| (n + 1) := (2 * sequence n) / (2 + sequence n)

theorem sequence_a5 : sequence 4 = 1 / 3 := 
sorry

end sequence_a5_l380_380979


namespace chameleon_increase_l380_380350

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380350


namespace cosine_between_vectors_l380_380612

noncomputable def vector_cos_angle (a b : ℝ × ℝ) := 
  let dot_product := (a.1 * b.1) + (a.2 * b.2)
  let norm_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / (norm_a * norm_b)

theorem cosine_between_vectors (t : ℝ) 
  (ht : let a := (1, t); let b := (-1, 2 * t);
        (3 * a.1 - b.1) * b.1 + (3 * a.2 - b.2) * b.2 = 0) :
  vector_cos_angle (1, t) (-1, 2 * t) = Real.sqrt 3 / 3 := 
by
  sorry

end cosine_between_vectors_l380_380612


namespace triangle_rectangle_ratio_l380_380121

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l380_380121


namespace ball_hits_ground_time_approx_l380_380394

variable (t : ℝ)

def height (t : ℝ) : ℝ := -16 * t^2 - 8 * t + 100

theorem ball_hits_ground_time_approx :
  height t = 0 → t ≈ 2.26 :=
by
  -- proof will be filled in here
  sorry

end ball_hits_ground_time_approx_l380_380394


namespace general_term_a_sum_sequence_b_l380_380240

noncomputable def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 3 else 1 + 2/(n : ℚ)

def sequence_b (n : ℕ) : ℚ :=
  (∏ i in Finset.range (n+1), sequence_a i) / ((n + 1) * 2^n)

def partial_sum (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n+1), sequence_b i

theorem general_term_a (n : ℕ) (h : n ≥ 1) :
  sequence_a n = 1 + 2/(n : ℚ) :=
sorry

theorem sum_sequence_b (n : ℕ) :
  partial_sum n = 2 - (n + 4)/(2^(n + 1)) :=
sorry

end general_term_a_sum_sequence_b_l380_380240


namespace repeating_decimal_as_fraction_l380_380165

theorem repeating_decimal_as_fraction : 
  let x := 2.353535... in
  x = 233 / 99 ∧ Nat.gcd 233 99 = 1 :=
by
  sorry

end repeating_decimal_as_fraction_l380_380165


namespace max_mass_of_sand_l380_380464

def platform_length : ℝ := 10
def platform_width : ℝ := 4
def sand_density : ℝ := 1500
def max_angle : ℝ := real.pi / 4 -- 45 degrees in radians

noncomputable def max_sand_mass : ℝ :=
  50667

theorem max_mass_of_sand (l w d θ : ℝ)
  (h_l : l = platform_length)
  (h_w : w = platform_width)
  (h_d : d = sand_density)
  (h_θ : θ = max_angle) :
  ∃ m, m = max_sand_mass :=
begin
  use 50667,
  sorry
end

end max_mass_of_sand_l380_380464


namespace students_in_both_clubs_l380_380751

theorem students_in_both_clubs (total_students drama_club science_club : ℕ) 
  (students_either_or_both both_clubs : ℕ) 
  (h_total_students : total_students = 250)
  (h_drama_club : drama_club = 80)
  (h_science_club : science_club = 120)
  (h_students_either_or_both : students_either_or_both = 180)
  (h_inclusion_exclusion : students_either_or_both = drama_club + science_club - both_clubs) :
  both_clubs = 20 :=
  by sorry

end students_in_both_clubs_l380_380751


namespace surveyed_individuals_not_working_percentage_l380_380645

theorem surveyed_individuals_not_working_percentage :
  (55 / 100 * 0 + 35 / 100 * (1 / 8) + 10 / 100 * (1 / 4)) = 6.875 / 100 :=
by
  sorry

end surveyed_individuals_not_working_percentage_l380_380645


namespace sin_over_sin_l380_380209

theorem sin_over_sin (a : Real) (h_cos : Real.cos (Real.pi / 4 - a) = 12 / 13)
  (h_quadrant : 0 < Real.pi / 4 - a ∧ Real.pi / 4 - a < Real.pi / 2) :
  Real.sin (Real.pi / 2 - 2 * a) / Real.sin (Real.pi / 4 + a) = 119 / 144 := by
sorry

end sin_over_sin_l380_380209


namespace num_pairs_satisfying_condition_l380_380527

/-- Prove that the number of pairs (x, y) in ℕ^2 such that 1/√x - 1/√y = 1/2016 is 165 --/
theorem num_pairs_satisfying_condition : 
  (Set.toFinset { p : ℕ × ℕ | (1 / Real.sqrt p.1 - 1 / Real.sqrt p.2) = (1 / 2016) }).card = 165 :=
sorry

end num_pairs_satisfying_condition_l380_380527


namespace correct_operation_l380_380068

theorem correct_operation : 
  (∀ (a b : ℝ), ¬(a^2 * a^3 = a^6) ∧ ¬((a^2)^3 = a^5) ∧ (∀ (a b : ℝ), (a * b)^3 = a^3 * b^3) ∧ ¬(a^8 / a^2 = a^4)) :=
by
  intros a b
  split
  -- proof for ¬(a^2 * a^3 = a^6)
  sorry
  split
  -- proof for ¬((a^2)^3 = a^5)
  sorry
  split
  -- proof for (a * b)^3 = a^3 * b^3
  sorry
  -- proof for ¬(a^8 / a^2 = a^4)
  sorry

end correct_operation_l380_380068


namespace number_of_correct_answers_is_95_l380_380970

variable (x y : ℕ) -- Define x as the number of correct answers and y as the number of wrong answers

-- Define the conditions
axiom h1 : x + y = 150
axiom h2 : 5 * x - 2 * y = 370

-- State the goal we want to prove
theorem number_of_correct_answers_is_95 : x = 95 :=
by
  sorry

end number_of_correct_answers_is_95_l380_380970


namespace find_b_l380_380924

theorem find_b (a b : ℝ) (h1 : f' 2 = 3 * (2 : ℝ)^2 + 4 * a * 2 + b = 0) (h2 : f 2 = (2 : ℝ)^3 + 2 * a * (2 : ℝ)^2 + b * 2 + a^2 = 17) : b = -100 :=
by
  sorry

end find_b_l380_380924


namespace convert_binary_1010_to_decimal_l380_380148

def binary_to_decimal (b : ℕ) : ℕ :=
  match b with
  | 0     => 0
  | _     => (b % 10) + 2 * binary_to_decimal (b / 10)

theorem convert_binary_1010_to_decimal : binary_to_decimal 1010 = 10 := by
  sorry

end convert_binary_1010_to_decimal_l380_380148


namespace proof_problem_l380_380199

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) + f x = 0

def decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

def satisfies_neq_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Main problem statement to prove (with conditions)
theorem proof_problem (f : ℝ → ℝ)
  (Hodd : odd_function f)
  (Hdec : decreasing_on f {y | 0 < y})
  (Hpt : satisfies_neq_point f (-2)) :
  {x : ℝ | (x - 1) * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end proof_problem_l380_380199


namespace eldest_boy_age_l380_380387

theorem eldest_boy_age
    (average_age : ℝ)
    (ratio_1 ratio_2 ratio_3 ratio_4 ratio_5 : ℝ)
    (num_boys : ℝ)
    (expected_age : ℝ)
    (h1 : average_age = 16.2)
    (h2 : ratio_1 = 2.5)
    (h3 : ratio_2 = 3.5)
    (h4 : ratio_3 = 5)
    (h5 : ratio_4 = 6.5)
    (h6 : ratio_5 = 9)
    (h7 : num_boys = 5)
    (h8 : expected_age = 27.51) :
    let total_age := average_age * num_boys
    let sum_ratios := ratio_1 + ratio_2 + ratio_3 + ratio_4 + ratio_5
    let x := total_age / sum_ratios
    let eldest_boy := ratio_5 * x in
    abs (eldest_boy - expected_age) < 0.01 := 
by 
  sorry

end eldest_boy_age_l380_380387


namespace unique_solution_l380_380530

theorem unique_solution :
  ∀ (x r p n : ℕ), prime p ∧ r ≥ 2 ∧ n ≥ 2 ∧ (x ^ r - 1 = p ^ n)
  → (x, r, p, n) = (3, 2, 2, 3) := by
  -- Proof omitted
  sorry

end unique_solution_l380_380530


namespace Priyanka_chocolates_l380_380141

variable (N S So P Sa T : ℕ)

theorem Priyanka_chocolates :
  (N + S = 10) →
  (So + P = 15) →
  (Sa + T = 10) →
  (N = 4) →
  ((S = 2 * y) ∨ (P = 2 * So)) →
  P = 10 :=
by
  sorry

end Priyanka_chocolates_l380_380141


namespace ratio_of_triangle_side_to_rectangle_width_l380_380122

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l380_380122


namespace base_n_215216_multiple_of_5_count_l380_380871

def g (n : ℕ) : ℕ := 6 + n + 2 * n^2 + 5 * n^3 + n^4 + 2 * n^5

theorem base_n_215216_multiple_of_5_count :
  (finset.filter (λ n, g n % 5 = 0) (finset.range (101+1)).filter (λ n, 2 ≤ n)).card = 20 :=
by
  sorry

end base_n_215216_multiple_of_5_count_l380_380871


namespace jason_remaining_pokemon_cards_l380_380989

theorem jason_remaining_pokemon_cards :
  (3 - 2) = 1 :=
by 
  sorry

end jason_remaining_pokemon_cards_l380_380989


namespace percent_monkeys_proof_l380_380132

def initial_counts : ℕ × ℕ × ℕ × ℕ := (6, 9, 3, 5) -- (monkeys, birds, squirrels, cats)
def events (initial : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
let (monkeys, birds, squirrels, cats) := initial in
(monkeys, birds - 4, squirrels - 1, cats) -- After 2 monkeys eat 2 birds, 2 cats chase 2 birds and 1 squirrel

def total_animals (counts : ℕ × ℕ × ℕ × ℕ) : ℕ :=
counts.1 + counts.2 + counts.3 + counts.4

def percent_monkeys (initial : ℕ × ℕ × ℕ × ℕ) (final : ℕ × ℕ × ℕ × ℕ) : ℕ :=
let total := total_animals final in
initial.1 * 100 / total

theorem percent_monkeys_proof :
  percent_monkeys initial_counts (events initial_counts) = 33 := by
  sorry

end percent_monkeys_proof_l380_380132


namespace larger_of_two_solutions_l380_380539

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l380_380539


namespace swimming_pool_length_l380_380470

noncomputable def solveSwimmingPoolLength : ℕ :=
  let w_pool := 22
  let w_deck := 3
  let total_area := 728
  let total_width := w_pool + 2 * w_deck
  let L := (total_area / total_width) - 2 * w_deck
  L

theorem swimming_pool_length : solveSwimmingPoolLength = 20 := 
  by
  -- Proof goes here
  sorry

end swimming_pool_length_l380_380470


namespace find_x_in_set_with_arithmetic_mean_l380_380214

theorem find_x_in_set_with_arithmetic_mean :
  let s := [12, 18, 24, 36, 6, x] in
  (∑ i in s, i) / s.length = 16 → x = 0 :=
sorry

end find_x_in_set_with_arithmetic_mean_l380_380214


namespace exists_question_l380_380443

-- Define the input conditions
def student := Fin 610
def question := Fin 10

-- Define a function that maps each student to the set of questions they answered correctly
variable (answers : student → Finset question)

-- Define unique correct set of answers for each student
def unique_answers : Prop :=
  ∀ (s1 s2 : student), s1 ≠ s2 → answers s1 ≠ answers s2

-- The statement we want to prove
theorem exists_question (answers : student → Finset question) (h_unique : unique_answers answers) :
  ∃ q : question, ∀ (s1 s2 : student), s1 ≠ s2 → (answers s1).erase q ≠ (answers s2).erase q :=
sorry

end exists_question_l380_380443


namespace total_gallons_needed_l380_380956

def gas_can_capacity : ℝ := 5.0
def number_of_cans : ℝ := 4.0
def total_gallons_of_gas : ℝ := gas_can_capacity * number_of_cans

theorem total_gallons_needed : total_gallons_of_gas = 20.0 := by
  -- proof goes here
  sorry

end total_gallons_needed_l380_380956


namespace max_difference_second_largest_second_smallest_l380_380792

theorem max_difference_second_largest_second_smallest :
  ∀ (a b c d e f g h : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h ∧
  a + b + c = 27 ∧
  a + b + c + d + e + f + g + h = 152 ∧
  f + g + h = 87 →
  g - b = 26 :=
by
  intros;
  sorry

end max_difference_second_largest_second_smallest_l380_380792


namespace cylinder_cone_surface_area_l380_380456

theorem cylinder_cone_surface_area (r h : ℝ) (π : ℝ) (l : ℝ)
    (h_relation : h = Real.sqrt 3 * r)
    (l_relation : l = 2 * r)
    (cone_lateral_surface_area : π * r * l = 2 * π * r ^ 2) :
    (2 * π * r * h) / (π * r ^ 2) = 2 * Real.sqrt 3 :=
by
    sorry

end cylinder_cone_surface_area_l380_380456


namespace jane_reading_days_l380_380665

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end jane_reading_days_l380_380665


namespace discount_on_second_toy_in_each_pair_l380_380392

-- Define the initial conditions
variable (num_toys : ℕ)
variable (price_per_toy : ℝ)
variable (total_spent : ℝ)

-- Given conditions in the problem
def initial_total : ℝ := num_toys * price_per_toy
def total_discount : ℝ := initial_total - total_spent
def num_discounted_toys : ℕ := num_toys / 2
def discount_per_toy : ℝ := total_discount / num_discounted_toys

-- The theorem to prove
theorem discount_on_second_toy_in_each_pair
  (h1 : num_toys = 4)
  (h2 : price_per_toy = 12)
  (h3 : total_spent = 36) :
  discount_per_toy num_toys price_per_toy total_spent = 6 :=
by
  sorry

end discount_on_second_toy_in_each_pair_l380_380392


namespace percentage_more_research_l380_380998

-- Defining the various times spent
def acclimation_period : ℝ := 1
def learning_basics_period : ℝ := 2
def dissertation_fraction : ℝ := 0.5
def total_time : ℝ := 7

-- Defining the time spent on each activity
def dissertation_period := dissertation_fraction * acclimation_period
def research_period := total_time - acclimation_period - learning_basics_period - dissertation_period

-- The main theorem to prove
theorem percentage_more_research : 
  ((research_period - learning_basics_period) / learning_basics_period) * 100 = 75 :=
by
  -- Placeholder for the proof
  sorry

end percentage_more_research_l380_380998


namespace magnitude_of_vector_sum_l380_380244

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vector_sum :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (0, 1)
  vector_magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = real.sqrt 5 :=
by 
  sorry

end magnitude_of_vector_sum_l380_380244


namespace digit_150_is_7_l380_380039

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380039


namespace broccoli_difference_l380_380459

theorem broccoli_difference (A : ℕ) (s : ℕ) (s' : ℕ)
  (h1 : A = 1600)
  (h2 : s = Nat.sqrt A)
  (h3 : s' < s)
  (h4 : (s')^2 < A)
  (h5 : A - (s')^2 = 79) :
  (1600 - (s')^2) = 79 :=
by
  sorry

end broccoli_difference_l380_380459


namespace find_a_and_b_find_intersection_l380_380933

-- Conditions from the problem
def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a-1)*x - a > 0}

def B (a b : ℝ) : Set ℝ := {x : ℝ | (x + a)*(x + b) > 0}

def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

-- Question 1
theorem find_a_and_b (a b : ℝ) (h : a ≠ b) (h1 : B a bᶜ = M) : 
  (a = -3 ∧ b = 1) ∨ (a = 1 ∧ b = -3) :=
sorry

-- Question 2
theorem find_intersection (a b : ℝ) (h1 : -1 < b) (h2 : b < a) (h3 : a < 1) : 
  A a ∩ B a b = {x : ℝ | x < -a ∨ x > 1} :=
sorry

end find_a_and_b_find_intersection_l380_380933


namespace OH_perp_MN_l380_380633

noncomputable def triangle_ABC (A B C D E F H O M N: Type) := Prop

-- Define the orthocenter and circumcenter properties
axiom orthocenter (A B C D E F H O: Type): Prop :=
is_orthocenter D E F

axiom circumcenter (A B C H E F O: Type): Prop :=
is_circumcenter A B C O ⟷ ⟨circumcenter_of_triangle ⟩

def altitudes (A B C D E F: Type) : Prop :=
is_altitude A D ∧ is_altitude B E ∧ is_altitude C F

def intersects_ab (ED AB M : Type) : Prop :=
intersects_at ED AB M

def intersects_ac (FD AC N : Type) : Prop :=
intersects_at FD AC N

theorem OH_perp_MN 
  (A B C D E F H O M N : Type)
  (h_triangle : triangle_ABC A B C D E F H O M N)
  (h_altitudes : altitudes A B C D E F)
  (h_orthocenter : orthocenter A B C D E F H O)
  (h_circumcenter : circumcenter A B C H E F O)
  (h_intersects_ab : intersects_ab ED AB M)
  (h_intersects_ac : intersects_ac FD AC N) : 
  _root_.is_perpendicular OH MN :=
sorry

end OH_perp_MN_l380_380633


namespace front_view_correct_l380_380526

-- Define the number of blocks in each column
def Blocks_Column_A : Nat := 3
def Blocks_Column_B : Nat := 5
def Blocks_Column_C : Nat := 2
def Blocks_Column_D : Nat := 4

-- Define the front view representation
def front_view : List Nat := [3, 5, 2, 4]

-- Statement to be proved
theorem front_view_correct :
  [Blocks_Column_A, Blocks_Column_B, Blocks_Column_C, Blocks_Column_D] = front_view :=
by
  sorry

end front_view_correct_l380_380526


namespace equation_of_line_AB_proof_l380_380203

noncomputable def equation_of_line_AB : Prop :=
  ∃ A B : ℝ × ℝ,
  (∃ x y : ℝ, (x - 4)^2 + (y - 2)^2 = 9 ∧ (A = (x, y) ∨ A = (x, y))) ∧
  (∃ x y : ℝ, (x - 4)^2 + (y - 2)^2 = 9 ∧ (B = (x, y) ∨ B = (x, y))) ∧
  ((x, y : ℝ) → (x = -2 ∧ y = -3 ∧
    ∃ Q : ℝ × ℝ, (Q = (4, 2)) ∧
    angle (P A Q) = π / 2 ∧ angle (P B Q) = π / 2 ) ∧
  (∃ l : Πx y : ℝ, 6 * x + 5 *  y - 25 = 0 )

theorem equation_of_line_AB_proof :
  equation_of_line_AB :=
sorry

end equation_of_line_AB_proof_l380_380203


namespace next_in_sequence_is_137_l380_380410

-- Define the sequence based on given conditions.
def initial_sequence := [12, 13, 15, 17, 111, 113, 117, 119, 123, 129]

-- Define a function to determine if a number contains digits '4' or '7'.
def contains_4_or_7 (n : Nat) : Bool :=
  n.digits 10 |>.any (λ d => d = 4 ∨ d = 7)

-- Define the function to generate the next number in the sequence
def next_number_in_sequence (prev : Nat) : Nat :=
  let candidates := List.range prev.succ (prev+20)
  candidates.find? (λ n => !contains_4_or_7 n && n > prev)

-- The main theorem: the next number in the sequence is 137
theorem next_in_sequence_is_137 : next_number_in_sequence 129 = some 137 := 
by sorry

end next_in_sequence_is_137_l380_380410


namespace find_m_l380_380929

theorem find_m {m : ℝ} : 
  let A := {m + 2, 2 * m ^ 2 + m}
  in 3 ∈ A ∧ (∀ x y ∈ A, x = y → x ≠ y) → m = -3 / 2 :=
by
  let A := {m + 2, 2 * m ^ 2 + m}
  intro h
  cases h with h1 h2
  sorry

end find_m_l380_380929


namespace frequency_of_3rd_group_l380_380658

theorem frequency_of_3rd_group (m : ℕ) (h_m : m ≥ 3) (x : ℝ) (h_area_relation : ∀ k, k ≠ 3 → 4 * x = k):
  100 * x = 20 :=
by
  sorry

end frequency_of_3rd_group_l380_380658


namespace option_C_correct_l380_380782

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct_l380_380782


namespace circumscribed_sphere_surface_area_l380_380568

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  a * sqrt 3 / 2

theorem circumscribed_sphere_surface_area
  (a : ℝ) (h : a = sqrt 3) :
  4 * π * (radius_of_circumscribed_sphere a)^2 = 9 * π :=
by
  sorry

end circumscribed_sphere_surface_area_l380_380568


namespace y_ordering_l380_380913

-- Definitions for the points on the parabola
def A : ℝ × ℝ := (-2, y1)
def B : ℝ × ℝ := (1, y2)
def C : ℝ × ℝ := (3, y3)

-- Conditions about the parabola's behavior
axiom parabola_behavior : ∀ x y, y > 0 → (x < 1 → y decreases as x increases) ∧ (x > 1 → y increases as x increases)

-- Axis of symmetry
def axis_of_symmetry : ℝ := 1

-- Symmetric point of A with respect to the axis of symmetry
def symmetric_A : ℝ × ℝ := (4, y1)

-- Proof statement to demonstrate the ordering of y-values
theorem y_ordering : y2 < y3 ∧ y3 < y1 :=
by 
  -- Parabola's behavior and ordering analysis goes here (proof omitted)
  sorry

end y_ordering_l380_380913


namespace fixed_point_log_function_l380_380731

theorem fixed_point_log_function (a : ℝ) (x : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (∃ y : ℝ, y = log a (x - 3) + 1) ∧ (x = 4 → y = 1) :=
by
  sorry

end fixed_point_log_function_l380_380731


namespace necessary_condition_l380_380440

-- Define sets
def R : Set ℤ := {x | x ∈ ℤ}
def N_star : Set ℤ := {x | x > 0}

-- Given condition
axiom subset_relation : N_star ⊆ Z

-- Proof problem statement
theorem necessary_condition (a : ℤ) (h : a ∈ R ∩ N_star) :
  a ∈ R ∩ Z :=
sorry

end necessary_condition_l380_380440


namespace find_y_l380_380776

theorem find_y (y : ℝ) (h : (sqrt y)^4 = 256) : y = 16 :=
sorry

end find_y_l380_380776


namespace identify_correct_propositions_l380_380906

variable (m n : Line) (α β : Plane)

def non_coincident_lines (m n : Line) : Prop := m ≠ n
def non_coincident_planes (α β : Plane) : Prop := α ≠ β
def line_in_plane (m : Line) (β : Plane) : Prop := m ⊆ β
def line_parallel_plane (m : Line) (α : Plane) : Prop := m ∥ α
def plane_parallel_plane (α β : Plane) : Prop := α ∥ β
def line_perpendicular_plane (m : Line) (α : Plane) : Prop := m ⟂ α
def lines_parallel (m n : Line) : Prop := m ∥ n

theorem identify_correct_propositions
  (hmn : non_coincident_lines m n)
  (hαβ : non_coincident_planes α β)
  (prop1 : line_in_plane m β ∧ plane_parallel_plane α β → line_parallel_plane m α)
  (prop2 : line_parallel_plane m β ∧ plane_parallel_plane α β → line_parallel_plane m α)
  (prop3 : line_perpendicular_plane m α ∧ plane_perpendicular_plane β α ∧ lines_parallel m n → lines_parallel n β)
  (prop4 : line_perpendicular_plane m α ∧ line_perpendicular_plane n β ∧ plane_parallel_plane α β → lines_parallel m n)
: prop1 ∧ prop4 :=
begin
  sorry,
end

end identify_correct_propositions_l380_380906


namespace time_on_DE_l380_380000

variables (A B C D E F : Type) -- Points in the park
variables (AB AC BC AD DE EF FC : ℝ) -- Lengths of the paths
variables (worker1_speed worker2_speed : ℝ) -- Speeds of workers
variables (worker1_time worker2_time : ℝ) -- Time taken by workers

-- Defining the conditions
def conditions :=
  worker1_speed > 0 ∧ worker2_speed = 1.2 * worker1_speed ∧ 
  worker1_time = 9 ∧ worker2_time = 9 ∧
  AB + BC = x ∧ AD + DE + EF + FC = 1.2 * x ∧
  DE = (worker2_speed * 1.2 - worker1_speed) / 2

theorem time_on_DE (h : conditions) : DE = 45 :=
by
  sorry

end time_on_DE_l380_380000


namespace tan_4050_deg_undefined_l380_380138

theorem tan_4050_deg_undefined :
  let theta := 4050 * (π / 180) in -- Convert degrees to radians
  4050 % 360 = 90 ∧
  tan 90 = Real.tan (π / 2) ∧
  cos (π / 2) = 0
  → tan theta = 0 := by
  sorry

end tan_4050_deg_undefined_l380_380138


namespace problem_statement_l380_380556

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem problem_statement :
  (M ∩ N) = N :=
by
  sorry

end problem_statement_l380_380556


namespace two_identical_squares_in_right_triangle_l380_380709

noncomputable def side_length_of_squares := 
  let AB := 6
  let BC := 8
  let AC := 10
  let h := (AB * BC) / AC
  let s := (h / 2)
  s

theorem two_identical_squares_in_right_triangle 
  (AB BC AC : ℝ) (h : ℝ) (s : ℝ) (h_calc : h = (AB * BC) / AC) (s_calc : s = h / 2) : 
  AB = 6 → BC = 8 → AC = 10 → s = 2.4 :=
by {
  intros,
  rw [←h_calc, ←s_calc],
  rw [*, real.mul_div_cancel_left, add_comm],
  field_simp,
  exact add_right_cancel (2 * s = 4.8),
  have temp: (6 * 8 : ℝ) / 10 = 4.8 := rfl,
  convert temp using 1,
  exact (h / 2 : ℝ)
}

#eval two_identical_squares_in_right_triangle

end two_identical_squares_in_right_triangle_l380_380709


namespace circle_radius_from_secants_l380_380468

theorem circle_radius_from_secants (P : ℝ) (PQ PR PS PT r : ℝ) 
  (hP : P = 15) 
  (hPQ : PQ = 11) (hQR : PR = PQ + 8) 
  (hPR : PR = 19) 
  (hPS : PS = 9) (hST : PT = PS + 6) 
  (hPT : PT = 15) 
  (h1 : 11 * 19 = (15 - r) * (15 + r)) 
  (h2 : 9 * 15 = (15 - r) * (15 + r)) :
  r = 4 :=
by
  have eq1 : 11 * 19 = 225 - r^2 := h1
  have eq2 : 9 * 15 = 225 - r^2 := h2
  have contr : 11 * 19 ≠ 9 * 15 := by norm_num
  contradiction
  -- Additional steps to handle the contradiction and conclude r = 4
  sorry

end circle_radius_from_secants_l380_380468


namespace solve_system_of_inequalities_l380_380381

theorem solve_system_of_inequalities {x : ℝ} :
  (|x^2 + 5 * x| < 6) ∧ (|x + 1| ≤ 1) ↔ (0 ≤ x ∧ x < 2) ∨ (4 < x ∧ x ≤ 6) :=
by
  sorry

end solve_system_of_inequalities_l380_380381


namespace repeating_decimal_to_fraction_l380_380167

theorem repeating_decimal_to_fraction : (2.353535... : Rational) = 233/99 :=
by
  sorry

end repeating_decimal_to_fraction_l380_380167


namespace parity_of_f_l380_380742

def f (x : ℝ) : ℝ := (sqrt (1 - x^2)) / (|x + 3| - 3)

theorem parity_of_f :
  ∀ x, (-1 ≤ x ∧ x ≤ 1) ∧ (|x + 3| ≠ 3) → f x = -f (-x) :=
by 
  intros x hx 
  sorry

end parity_of_f_l380_380742


namespace greatest_take_home_pay_l380_380276

-- Defining the tax system and take-home pay
def tax (x : ℝ) := (x / 100) * (500 * x - 2000)
def take_home_pay (x : ℝ) := 500 * x - tax x

-- Define the income that yields the greatest take-home pay
def max_income := 26040

-- Proof statement: For x = 52.08, the corresponding income yields the greatest take-home pay.
theorem greatest_take_home_pay : 
  let x := 52.08 in 
  take_home_pay x = 500 * x - tax x ∧ (500 * x - tax x = take_home_pay (max_income / 100)) :=
begin
  sorry
end

end greatest_take_home_pay_l380_380276


namespace vector_subtraction_proof_l380_380509

def vec1 := (3, -5)
def vec2 := (2, -9)
def scalar1 := 4.5
def scalar2 := 3
def result := (7.5, 4.5)

theorem vector_subtraction_proof :
  scalar1 • vec1 - scalar2 • vec2 = result :=
by
  sorry

end vector_subtraction_proof_l380_380509


namespace jennifer_remaining_amount_l380_380669

-- Define the initial amount Jennifer had
def initial_amount : ℝ := 150.75

-- Define the expenses
def gourmet_sandwich : ℝ := (3/10) * initial_amount
def museum_ticket : ℝ := (1/4) * initial_amount
def book : ℝ := (1/8) * initial_amount
def coffee : ℝ := (2.5/100) * initial_amount

-- Prove the remaining amount
theorem jennifer_remaining_amount :
  initial_amount - (gourmet_sandwich + museum_ticket + book + coffee) = 45.225 :=
by
  sorry

end jennifer_remaining_amount_l380_380669


namespace f_is_not_T_function_g_is_T_function_h_is_T_function_l380_380183

/-- Define what it means to be a T function -/
def is_T_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x : ℝ, f(x + T) = T * f(x)

/-- Assertion that f(x) = x is not a T function for any T ∈ ℝ -/
theorem f_is_not_T_function (T : ℝ) (hT_nonzero : T ≠ 0) : ¬ is_T_function (λ x, x) T :=
by sorry

/-- Assertion that g(x) = a^x is a T function if the function intersects y = x -/
theorem g_is_T_function (a T : ℝ) (h_a_pos : 0 < a) (h_a_neq_1 : a ≠ 1) (h_intersects : ∃ x, a^x = x) :
  is_T_function (λ x, a^x) T :=
by sorry

/-- Assertion that h(x) = cos(mx) is a T function if m = kπ for k ∈ ℤ -/
theorem h_is_T_function (m T : ℝ) (T_eq_1_or_neg1 : T = 1 ∨ T = -1) (m_ran : ∃ k : ℤ, m = k * π) :
  is_T_function (λ x, Real.cos (m * x)) T :=
by sorry

end f_is_not_T_function_g_is_T_function_h_is_T_function_l380_380183


namespace BrianchonsTheorem_l380_380682

-- Definitions related to geometric constructs might go here

theorem BrianchonsTheorem 
  (A B C D E F O : Point) 
  (hexagon_circumscribed : is_circumscribed_hexagon A B C D E F O)
  : ∃ P : Point, 
    intersects_at_single_point (diagonal A D) (diagonal B E) (diagonal C F) P :=
by sorry
-- "intersects_at_single_point" is assumed to denote intersection of three lines at point P.
-- "diagonal X Y" is representation of diagonals in the hexagon.

end BrianchonsTheorem_l380_380682


namespace Tanya_pays_face_moisturizer_l380_380507

variable (F : ℝ) -- Tanya pays F dollars for each face moisturizer.

variable (Tanya_total_spent : ℝ)
variable (Christy_total_spent : ℝ)
variable (total_spent : ℝ)

-- Conditions
def Tanya_spending (F : ℝ) : Tanya_total_spent = 2 * F + 4 * 60 := by sorry
def Christy_spending (Tanya_total_spent : ℝ) : Christy_total_spent = 2 * Tanya_total_spent := by sorry
def total_spending (Tanya_total_spent Christy_total_spent : ℝ) : total_spent = Tanya_total_spent + Christy_total_spent := by sorry
def total_spent_value : total_spent = 1020 := by sorry

theorem Tanya_pays_face_moisturizer : F = 50 :=
by
  have h1 : Tanya_total_spent = 2 * F + 4 * 60 := Tanya_spending F
  have h2 : Christy_total_spent = 2 * Tanya_total_spent := Christy_spending Tanya_total_spent
  have h3 : total_spent = Tanya_total_spent + Christy_total_spent := total_spending Tanya_total_spent Christy_total_spent
  have h4 : total_spent = 1020 := total_spent_value
  sorry

end Tanya_pays_face_moisturizer_l380_380507


namespace probability_units_digit_is_one_l380_380562

theorem probability_units_digit_is_one :
  (let valid_m := {11, 13, 15, 17, 19} : Finset ℕ;
       valid_n := (finset.range 20) + 1999; -- valid_n = {1999, 2000, ..., 2018}
       count_ways (m : ℕ) : ℕ :=
         if m = 11 then 20
         else if m = 13 then 5
         else if m = 15 then 0
         else if m = 17 then 5
         else if m = 19 then 10
         else 0 in
   let total_successful := ∑ m in valid_m, count_ways m;
       total_possible := valid_m.card * valid_n.card in
   total_successful / total_possible = 2 / 5) :=
sorry

end probability_units_digit_is_one_l380_380562


namespace equal_angles_l380_380287

variables (A B C D K P : Type) [trapezoid A B C D]
variables (AD BC AB KP : ℝ)

-- The conditions
noncomputable def condition1 (h_trap : is_trapezoid A B C D) : AD / BC = 3 / 2 := sorry
noncomputable def condition2 (h_trap : is_trapezoid A B C D) (h_perp1 : perpendicular A B AD) (h_perp2 : perpendicular A B BC) : true := sorry
noncomputable def condition3 (h_point : on_segment K A B) (ratio_AK_AB : KA / AB = 3 / 5) : true := sorry
noncomputable def condition4 (h_point : on_segment P K CD) (h_perp3 : perpendicular KP CD) : true := sorry

-- The theorem to prove
theorem equal_angles (h_trap : is_trapezoid A B C D) (h_cond1 : AD / BC = 3 / 2)
                     (h_perp1 : perpendicular A B AD) (h_perp2 : perpendicular A B BC)
                     (h_point : on_segment K A B) (ratio_AK_AB : KA / AB = 3 / 5)
                     (h_point2 : on_segment P K CD) (h_perp3 : perpendicular KP CD) :
                     angle K P A = angle K P B :=
begin
  -- Given conditions
  apply condition1 h_trap,
  apply condition2 h_trap h_perp1 h_perp2,
  apply condition3 h_point ratio_AK_AB,
  apply condition4 h_point2 h_perp3,
  -- Prove the theorem
  sorry -- skipping the detailed proof steps
end

end equal_angles_l380_380287


namespace range_of_largest_root_l380_380146

theorem range_of_largest_root :
  ∀ (a_2 a_1 a_0 : ℝ), 
  (|a_2| ≤ 1 ∧ |a_1| ≤ 1 ∧ |a_0| ≤ 1) ∧ (a_2 + a_1 + a_0 = 0) →
  (∃ s > 1, ∀ x > 0, x^3 + 3*a_2*x^2 + 5*a_1*x + a_0 = 0 → x ≤ s) ∧
  (s < 2) :=
by sorry

end range_of_largest_root_l380_380146


namespace triangle_rectangle_ratio_l380_380120

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l380_380120


namespace minimum_overlap_l380_380273

variable (U : Finset ℕ) -- This is the set of all people surveyed
variable (B V : Finset ℕ) -- These are the sets of people who like Beethoven and Vivaldi respectively.

-- Given conditions:
axiom h_total : U.card = 120
axiom h_B : B.card = 95
axiom h_V : V.card = 80
axiom h_subset_B : B ⊆ U
axiom h_subset_V : V ⊆ U

-- Question to prove:
theorem minimum_overlap : (B ∩ V).card = 95 + 80 - 120 := by
  sorry

end minimum_overlap_l380_380273


namespace twenty_five_th_number_l380_380995

theorem twenty_five_th_number (n : ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1)
(h2 : ∀ i, 1 ≤ i < 5 → a (i + 1) = 2 * a i)
(h3 : ∀ i, (i + 1) % 5 = 0 → a (i + 1) = a i + 1)
(h4 : ∀ i, (i + 1) % 5 ≠ 0 → a (i + 1) = 2 * a i): 
a 25 = 69956 := by
  sorry

end twenty_five_th_number_l380_380995


namespace train_crossing_time_l380_380481

-- Define the conditions
def train_length : ℕ := 170
def train_speed_kmh : ℝ := 45
def bridge_length : ℕ := 205
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600) -- Convert speed to m/s
def total_distance : ℕ := train_length + bridge_length

-- State the theorem
theorem train_crossing_time : (total_distance / train_speed_ms) = 30 := by 
  sorry

end train_crossing_time_l380_380481


namespace constant_term_binomial_expansion_l380_380390

theorem constant_term_binomial_expansion :
  let x := Real
  let f := λ (x : Real), (sqrt x - 1 / x)^9
  is_constant_term (binomial_expansion f 9) = -84 :=
sorry

end constant_term_binomial_expansion_l380_380390


namespace find_c_l380_380982

noncomputable def c_squared (a b : ℝ) (C : ℝ) := a^2 + b^2 - 2 * a * b * real.cos C

theorem find_c (a b : ℝ) (C : ℝ) (ha : a = 2) (hb : b = 1) (hC : C = real.pi / 3) :
  sqrt (c_squared a b C) = real.sqrt 3 :=
by {
  rw [ha, hb, hC],
  norm_num,
  sorry
}

end find_c_l380_380982


namespace supplement_of_angle_l380_380909

theorem supplement_of_angle (A : ℝ) (h : 90 - A = A - 18) : 180 - A = 126 := by
    sorry

end supplement_of_angle_l380_380909


namespace green_chameleon_increase_l380_380335

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380335


namespace angle_A_is_60_sequence_formula_minimum_value_f_correct_statements_about_prisms_l380_380441

-- Problem 1: Proof regarding angle A given a condition in a triangle
theorem angle_A_is_60 (a b c : ℝ) (h : b^2 + c^2 = b * c + a^2) : angle A = 60 :=
sorry

-- Problem 2: General formula for the sequence
theorem sequence_formula (n : ℕ) : (∃ a_n : ℝ, a_n = 7/9 * (10^n - 1)) :=
sorry

-- Problem 3: Minimum value of the function
theorem minimum_value_f (x : ℝ) (h₁ : 0 < x) (h₂ : x < π/2) : 
    ∃ m : ℝ, is_minimum (λ x, (1 + cos (2 * x) + 8 * sin^2 x) / sin (2 * x)) 4 :=
sorry

-- Problem 4: Correct statements about prisms
theorem correct_statements_about_prisms : 
    (∀ (P : Prism), 
        (all_faces_flat P ∧ number_of_lateral_faces P = number_of_sides_on_base P ∧ top_and_bottom_faces_congruent P)) :=
sorry

end angle_A_is_60_sequence_formula_minimum_value_f_correct_statements_about_prisms_l380_380441


namespace tenth_term_is_513_l380_380570

def nth_term (n : ℕ) : ℕ :=
  2^(n-1) + 1

theorem tenth_term_is_513 : nth_term 10 = 513 := 
by 
  sorry

end tenth_term_is_513_l380_380570


namespace B_subset_A_l380_380301

def A (x : ℝ) : Prop := abs (2 * x - 3) > 1
def B (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem B_subset_A : ∀ x, B x → A x := sorry

end B_subset_A_l380_380301


namespace sum_of_binary_digits_equals_eight_l380_380852

def dec_to_bin (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else list.reverse (nat.digits 2 n)

def sum_of_digits (l : list ℕ) : ℕ :=
  l.sum

def decimal_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (nat.digits 10 n)

def binary_sum_of_digits (n : ℕ) : ℕ :=
  sum_of_digits (dec_to_bin n)

theorem sum_of_binary_digits_equals_eight :
  binary_sum_of_digits 157 + binary_sum_of_digits (decimal_sum_of_digits 157) = 8 :=
by
  sorry

end sum_of_binary_digits_equals_eight_l380_380852


namespace initial_balloons_correct_l380_380494

-- Define the variables corresponding to the conditions given in the problem
def boy_balloon_count := 3
def girl_balloon_count := 12
def balloons_sold := boy_balloon_count + girl_balloon_count
def balloons_remaining := 21

-- State the theorem asserting the initial number of balloons
theorem initial_balloons_correct :
  balloons_sold + balloons_remaining = 36 := sorry

end initial_balloons_correct_l380_380494


namespace selection_count_l380_380971

theorem selection_count (word : String) (vowels : Finset Char) (consonants : Finset Char)
  (hword : word = "УЧЕБНИК")
  (hvowels : vowels = {'У', 'Е', 'И'})
  (hconsonants : consonants = {'Ч', 'Б', 'Н', 'К'})
  :
  vowels.card * consonants.card = 12 :=
by {
  sorry
}

end selection_count_l380_380971


namespace monotonic_intervals_f_range_f_diff_l380_380234

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * x + 2 * Real.log x

theorem monotonic_intervals_f :
  (∀ x, 0 < x → x < 1/2 → deriv (f x 5) x > 0) ∧
  (∀ x, 2 < x → deriv (f x 5) x > 0) ∧
  (∀ x, 1/2 < x → x < 2 → deriv (f x 5) x < 0) :=
sorry

theorem range_f_diff (x₁ x₂ : ℝ) (h₀ : 1/3 < x₁) (h₁ : x₁ < 1/e) (h₂ : 1/e < x₂) :
  (e ^ 2 - 1 / e ^ 2 - 4 < f x₁ 5 - f x₂ 5) ∧
  (f x₁ 5 - f x₂ 5 < 80 / 9 - 4 * Real.log 3) :=
sorry

end monotonic_intervals_f_range_f_diff_l380_380234


namespace fraction_subtraction_equals_one_l380_380839

theorem fraction_subtraction_equals_one (x : ℝ) (h : x ≠ 1) : (x / (x - 1)) - (1 / (x - 1)) = 1 := 
by sorry

end fraction_subtraction_equals_one_l380_380839


namespace digit_150th_of_17_div_70_is_7_l380_380022

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380022


namespace problem1_problem2_l380_380901

-- Definitions
def A := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) := {x : ℝ | x^2 - (2*m - 3)*x + m^2 - 3*m ≤ 0}

-- Problem 1
theorem problem1 {m : ℝ} (h : A ∩ B m = set.Icc 2 4) : m = 5 :=
sorry

-- Problem 2
theorem problem2 {m : ℝ} (h : A ⊆ (set.univ \ B m)) : m ∈ set.Icc (-∞) (-2) ∪ set.Icc 7 ∞ :=
sorry

end problem1_problem2_l380_380901


namespace smallest_square_area_l380_380185

theorem smallest_square_area :
  (∀ (x y : ℝ), (∃ (x1 x2 y1 y2 : ℝ), y1 = 3 * x1 - 4 ∧ y2 = 3 * x2 - 4 ∧ y = x^2 + 5 ∧ 
  ∀ (k : ℝ), x1 + x2 = 3 ∧ x1 * x2 = 5 - k ∧ 16 * k^2 - 332 * k + 396 = 0 ∧ 
  ((k = 1.5 ∧ 10 * (4 * k - 11) = 50) ∨ 
  (k = 16.5 ∧ 10 * (4 * k - 11) ≠ 50))) → 
  ∃ (A: Real), A = 50) :=
sorry

end smallest_square_area_l380_380185


namespace net_percentage_change_investment_l380_380636

theorem net_percentage_change_investment :
  ∀ (initial : ℝ) (first_loss_pct : ℝ) (second_gain_pct : ℝ) (third_loss_pct : ℝ),
    initial = 200 → 
    first_loss_pct = 0.10 → 
    second_gain_pct = 0.15 → 
    third_loss_pct = 0.10 →
    let first_year_remaining := initial * (1 - first_loss_pct) in
    let second_year_remaining := first_year_remaining * (1 + second_gain_pct) in
    let final_remaining := second_year_remaining * (1 - third_loss_pct) in
    let net_percentage_change := ((final_remaining - initial) / initial) * 100 in
    net_percentage_change = -6.85 :=
by
  intros
  sorry

end net_percentage_change_investment_l380_380636


namespace green_chameleon_increase_l380_380336

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380336


namespace two_point_questions_l380_380426

theorem two_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
by
  sorry

end two_point_questions_l380_380426


namespace green_chameleon_increase_l380_380339

variables (initial_green initial_yellow initial_red initial_blue: ℕ)
variables (sunny_days cloudy_days : ℕ)

-- Define number of yellow chameleons increased
def yellow_increase : ℕ := 5
-- Define number of sunny and cloudy days in September
def sunny_days_in_september : ℕ := 18
def cloudy_days_in_september : ℕ := 12

theorem green_chameleon_increase :
  let initial_diff := initial_green - initial_yellow in
  yellow_increase = 5 →
  sunny_days_in_september = 18 →
  cloudy_days_in_september = 12 →
  let final_diff := initial_diff + (sunny_days_in_september - cloudy_days_in_september) in
  final_diff - initial_diff = 6 →
  (initial_yellow + yellow_increase) + (final_diff - initial_diff) - initial_yellow = 11 :=
by
  intros initial_diff h1 h2 h3 final_diff h4 h5
  sorry

end green_chameleon_increase_l380_380339


namespace integral_evaluation_l380_380082

noncomputable def integral_solution : ℝ :=
  ∫ x in 0..Real.arcsin (Real.sqrt (7 / 8)), (6 * (Real.sin x)^2) / (4 + 3 * Real.cos (2 * x))

theorem integral_evaluation :
  integral_solution = (Real.sqrt 7) * Real.pi / 4 - Real.arctan (Real.sqrt 7) :=
by
  sorry

end integral_evaluation_l380_380082


namespace g_neg_two_is_zero_l380_380585

theorem g_neg_two_is_zero {f g : ℤ → ℤ} 
  (h_odd: ∀ x: ℤ, f (-x) + (-x) = -(f x + x)) 
  (hf_two: f 2 = 1) 
  (hg_def: ∀ x: ℤ, g x = f x + 1):
  g (-2) = 0 := 
sorry

end g_neg_two_is_zero_l380_380585


namespace not_satisfiable_conditions_l380_380631

theorem not_satisfiable_conditions (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) 
    (h3 : 10 * x + y % 80 = 0) (h4 : x + y = 2) : false := 
by 
  -- The proof is omitted because we are only asked for the statement.
  sorry

end not_satisfiable_conditions_l380_380631


namespace jenna_costume_l380_380990

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end jenna_costume_l380_380990


namespace sequence_sum_S15_S22_S31_l380_380928

def sequence_sum (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | m + 1 => sequence_sum m + (-1)^m * (3 * (m + 1) - 1)

theorem sequence_sum_S15_S22_S31 :
  sequence_sum 15 + sequence_sum 22 - sequence_sum 31 = -57 := 
sorry

end sequence_sum_S15_S22_S31_l380_380928


namespace cos_angle_intersection_of_diagonals_l380_380140

theorem cos_angle_intersection_of_diagonals (a : ℝ) (h : a = 16) :
  ∃ (θ : ℝ), θ = ∠PXS ∧ cos θ = 1 / Real.sqrt 2 :=
by
  sorry

end cos_angle_intersection_of_diagonals_l380_380140


namespace larger_solution_of_quadratic_equation_l380_380541

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l380_380541


namespace meaningful_log_condition_l380_380002

theorem meaningful_log_condition (θ : ℝ) :
  (cos θ * tan θ > 0 ∧ sin θ ≠ 1) →
  (0 < θ ∧ θ < π ∨ π < θ ∧ θ < 2 * π) :=
by
  sorry

end meaningful_log_condition_l380_380002


namespace simson_line_theorem_l380_380371

noncomputable theory
open_locale real

def power_of_point (A B C : Point) : ℝ := 
  A.dist_to B * A.dist_to C

def on_circle (A : Point) (Γ : Circle) : Prop := 
  A.dist_to Γ.center = Γ.radius

theorem simson_line_theorem (A B C B' C' : Point) (Γ : Circle) 
(h1 : A, B, C are aligned)
(h2 : A, B', C' are aligned)
(h3 : ∀ (A : Point), power_of_point A B C = power_of_point A B' C') :
cocyclic_points B B' C C' :=
begin
  sorry
end

end simson_line_theorem_l380_380371


namespace axis_of_symmetry_l380_380727

noncomputable def function_f : ℝ → ℝ :=
  λ x, sin (3 * x + π / 3) * cos (x - π / 6) - cos (3 * x + π / 3) * sin (x + π / 3)

theorem axis_of_symmetry :
  ∃ k : ℤ, (0:ℝ) = sin (4 * (π/12 + k * π/4) + π / 6) :=
sorry

end axis_of_symmetry_l380_380727


namespace no_ultra_prime_numbers_l380_380679

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def g (n : ℕ) : ℕ := (Finset.filter (λ d, d < n) (Finset.range (n + 1))).sum

def ultra_prime (n : ℕ) : Prop :=
  is_prime n ∧ g (g n) = n + 3

theorem no_ultra_prime_numbers : ∀ n : ℕ, ¬ ultra_prime n :=
by
  intro n
  sorry

end no_ultra_prime_numbers_l380_380679


namespace reporter_earns_per_article_l380_380471

noncomputable def pay_per_word := 0.1
noncomputable def articles := 3
noncomputable def hours := 4
noncomputable def words_per_minute := 10
noncomputable def expected_hourly_earning := 105

def earnings_per_article (pay_per_word : ℝ) (articles : ℕ) (hours : ℕ) 
    (words_per_minute : ℕ) (expected_hourly_earning : ℝ) :=
  let words_per_hour := words_per_minute * 60
  let earnings_from_words_per_hour := words_per_hour * pay_per_word
  let earnings_from_articles_per_hour := expected_hourly_earning - earnings_from_words_per_hour
  let total_earnings_from_articles := earnings_from_articles_per_hour * hours
  total_earnings_from_articles / articles

theorem reporter_earns_per_article (h : earnings_per_article pay_per_word articles hours words_per_minute expected_hourly_earning = 60) :
  True :=
sorry

end reporter_earns_per_article_l380_380471


namespace intersection_A_B_l380_380686

noncomputable def domain_ln_1_minus_x : Set ℝ := {x : ℝ | x < 1}
def range_x_squared : Set ℝ := {y : ℝ | 0 ≤ y}
def intersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

theorem intersection_A_B :
  (domain_ln_1_minus_x ∩ range_x_squared) = intersection :=
by sorry

end intersection_A_B_l380_380686


namespace scientific_notation_12e6_l380_380315

theorem scientific_notation_12e6 :
  ∃ m k, (1 ≤ m ∧ m < 10) ∧ k = 7 ∧ 12_000_000 = m * 10^k :=
by
  sorry

end scientific_notation_12e6_l380_380315


namespace inequality_f_l380_380367

-- Define the function f
def f (x : ℝ) : ℝ := 1 - Real.exp (-x)

-- Define the inequality to be proven
theorem inequality_f (x : ℝ) (h : x > -1) : f x ≥ x / (x + 1) := 
by
  sorry

end inequality_f_l380_380367


namespace digit_150th_in_decimal_of_fraction_l380_380045

theorem digit_150th_in_decimal_of_fraction : 
  (∀ n : ℕ, n > 0 → (let seq := [2, 4, 2, 8, 5, 7] in seq[((n - 1) % 6)] = 
  if n == 150 then 7 else seq[((n - 1) % 6)])) :=
by
  sorry

end digit_150th_in_decimal_of_fraction_l380_380045


namespace probability_allison_rolls_greater_l380_380114

theorem probability_allison_rolls_greater :
  let brian_faces := {1, 2, 3, 4, 5, 6}.to_finset
  let noah_faces := {1, 1, 1, 5, 5, 5}.to_multiset
  (∥noah_faces.filter (λ x, x < 4)∥ : ℝ) / ∥noah_faces∥ * (∥brian_faces.filter (λ x, x < 4)∥ : ℝ) / ∥brian_faces∥ = 1/4 := by
  sorry

end probability_allison_rolls_greater_l380_380114


namespace find_integer_pairs_l380_380529

theorem find_integer_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (int.sqrt ((a * b) / (2 * b^2 - a)) = (a + 2 * b) / (4 * b)) ↔ 
  (a = 72 ∧ b = 18) ∨ (a = 72 ∧ b = 12) :=
sorry

end find_integer_pairs_l380_380529


namespace diamondsuit_result_l380_380677

def diam (a b : ℕ) : ℕ := a

theorem diamondsuit_result : (diam 7 (diam 4 8)) = 7 :=
by sorry

end diamondsuit_result_l380_380677


namespace three_digit_integers_with_two_identical_digits_less_than_700_l380_380944

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def less_than_700 (n : ℕ) : Prop :=
  n < 700

def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.nodup = false

theorem three_digit_integers_with_two_identical_digits_less_than_700 : 
  ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_three_digit n ∧ less_than_700 n ∧ has_at_least_two_identical_digits n) ∧
  s.card = 156 := by
  sorry

end three_digit_integers_with_two_identical_digits_less_than_700_l380_380944


namespace pregnant_dogs_count_l380_380843

-- Definitions as conditions stated in the problem
def total_puppies (P : ℕ) : ℕ := 4 * P
def total_shots (P : ℕ) : ℕ := 2 * total_puppies P
def total_cost (P : ℕ) : ℕ := total_shots P * 5

-- Proof statement without proof
theorem pregnant_dogs_count : ∃ P : ℕ, total_cost P = 120 → P = 3 :=
by sorry

end pregnant_dogs_count_l380_380843


namespace correct_order_option_C_l380_380750

def length_unit_ordered (order : List String) : Prop :=
  order = ["kilometer", "meter", "centimeter", "millimeter"]

def option_A := ["kilometer", "meter", "millimeter", "centimeter"]
def option_B := ["meter", "kilometer", "centimeter", "millimeter"]
def option_C := ["kilometer", "meter", "centimeter", "millimeter"]

theorem correct_order_option_C : length_unit_ordered option_C := by
  sorry

end correct_order_option_C_l380_380750


namespace triangle_inequality_with_angle_bisectors_l380_380815

theorem triangle_inequality_with_angle_bisectors
  (ABC : Triangle)
  (A B C : Point)
  (hABC : IsTriangle A B C)
  (X Y : Point)
  (hX : IsAngleBisector A B C X)
  (hY : IsMeetingCircumcircleOf A B C X Y)
  (r_A r_B r_C : ℝ)
  (h_rA : r_A = AX / AY)
  (h_rB : r_B = BX / BY)
  (h_rC : r_C = CX / CY)
  (sin_A sin_B sin_C : ℝ)
  (h_sinA : sin_A = sin (angle A B C))
  (h_sinB : sin_B = sin (angle B C A))
  (h_sinC : sin_C = sin (angle C A B)) :
  (r_A / sin_A^2) + (r_B / sin_B^2) + (r_C / sin_C^2) ≥ 3 ↔ IsEquilateral A B C := sorry

end triangle_inequality_with_angle_bisectors_l380_380815


namespace age_twice_in_years_l380_380094

theorem age_twice_in_years : ∃ Y : ℕ, let S := 32, let M := S + 34 in M + Y = 2 * (S + Y) ↔ Y = 2 :=
by
  let S := 32
  let M := S + 34
  use 2
  sorry

end age_twice_in_years_l380_380094


namespace find_y_l380_380181

def vector (R : Type) := matrix (fin 2) (fin 1) R

def dot_product {R : Type*} [has_mul R] [has_add R] [has_zero R] 
  (v w : vector R) : R :=
  v 0 0 * w 0 0 + v 1 0 * w 1 0

def proj (v w : vector ℝ) : vector ℝ :=
  (dot_product v w / dot_product w w) • w

theorem find_y :
  let v := ![![2], ![y]],
      w := ![![8], ![4]],
      proj_w_v := ![![-4], ![-2]]
  in
  proj v w = proj_w_v → y = (-12 : ℝ) :=
by simp; sorry

end find_y_l380_380181


namespace sin_cos_theorem_l380_380255

variable {θ a b : ℝ}

theorem sin_cos_theorem
  (h : (sin θ)^6 / a + (cos θ)^6 / b = 1 / (a + b)) :
  (sin θ)^12 / a^2 + (cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 :=
sorry

end sin_cos_theorem_l380_380255


namespace find_ab_sum_eq_42_l380_380900

noncomputable def find_value (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem find_ab_sum_eq_42 (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) : find_value a b = 42 := by
  sorry

end find_ab_sum_eq_42_l380_380900


namespace triangle_area_correct_l380_380983

noncomputable def focus_of_parabola : Point := ⟨1, 0⟩

-- midpoint (M) of segment AB is (2,2)
def midpoint : Point := ⟨2, 2⟩

-- Triangle area calculation
def area_of_triangle_ABF (A B : Point) (F : Point) : ℝ :=
  -- Use half the determinant to compute the area of the triangle with vertices A, B and F
  1 / 2 * ((A.x - F.x) * (B.y - F.y) - (A.y - F.y) * (B.x - F.x)).abs

-- Main theorem statement
theorem triangle_area_correct {A B : Point} :
  parabola_contains A ∧ parabola_contains B ∧ midpoint (A, B) = midpoint → 
    area_of_triangle_ABF A B focus_of_parabola = 2 :=
by
  sorry

end triangle_area_correct_l380_380983


namespace φ_range_l380_380923

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) + 1

theorem φ_range (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < π / 2)
(h_period : (2 * π) / ω = 2 * π / 3)
(h_f_pos : ∀ x, x ∈ Ioo (-π / 12 : ℝ) (π / 6 : ℝ) → f ω φ x > 1) :
φ ∈ Icc (-π / 4) 0 := 
sorry

end φ_range_l380_380923


namespace sky_color_changes_l380_380521

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l380_380521


namespace circle_line_intersection_l380_380927

theorem circle_line_intersection (x y a : ℝ) (A B C O : ℝ × ℝ) :
  (x + y = 1) ∧ ((x^2 + y^2) = a) ∧ 
  (O = (0, 0)) ∧ 
  (x^2 + y^2 = a ∧ (A.1^2 + A.2^2 = a) ∧ (B.1^2 + B.2^2 = a) ∧ (C.1^2 + C.2^2 = a) ∧ 
  (A.1 + B.1 = C.1) ∧ (A.2 + B.2 = C.2)) -> 
  a = 2 := 
sorry

end circle_line_intersection_l380_380927


namespace area_under_parabola_l380_380720

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- State the theorem about the area under the curve
theorem area_under_parabola : (∫ x in (1 : ℝ)..3, parabola x) = 4 / 3 :=
by
  -- Proof goes here
  sorry

end area_under_parabola_l380_380720


namespace ratio_of_triangle_side_to_rectangle_width_l380_380123

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l380_380123


namespace chameleon_increase_l380_380348

/-- On an island, there are red, yellow, green, and blue chameleons.
- On a cloudy day, either one red chameleon changes its color to yellow, or one green chameleon changes its color to blue.
- On a sunny day, either one red chameleon changes its color to green, or one yellow chameleon changes its color to blue.
In September, there were 18 sunny days and 12 cloudy days. The number of yellow chameleons increased by 5.
Prove that the number of green chameleons increased by 11. 
-/
theorem chameleon_increase 
  (R Y G B : ℕ) -- numbers of red, yellow, green, and blue chameleons
  (cloudy_sunny : 18 * (R → G) - 12 * (G → B))
  (increase_yellow : Y + 5 = Y') 
  (sunny_days : 18)
  (cloudy_days : 12) : -- Since we are given 18 sunny days and 12 cloudy days,
  G' = G + 11 :=
by sorry

end chameleon_increase_l380_380348


namespace proof_correct_statements_l380_380783

theorem proof_correct_statements :
  (∀ (ξ : ℝ) (σ : ℝ), ξ ~ Normal 1 σ^2 → (P(ξ ≤ 4) = 0.79 → P(ξ < -2) = 0.21)) ∧
  (∀ (X : ℕ → ℝ) (n : ℕ) (p : ℝ), (X ~ Binomial n p) → (E(X) = 30 ∧ D(X) = 20 → n = 90)) ∧
  ¬(∀ (a b : ℝ) (data : Set ℝ), D({a * x + b | x ∈ data}) = a * D(data)) ∧
  (∀ (r : ℝ), |r| < 1 → strong_linear_correlation(r)) →
  (ABD is_correct).
sorry

end proof_correct_statements_l380_380783


namespace mouse_wins_l380_380763

def adjacent (p q : ℕ × ℕ) : Prop :=
(p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 = q.2 - 1)) ∨ 
(p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 = q.1 - 1))

noncomputable def winning_strategy_for_mouse := ∃ m : ℕ → ℕ × ℕ,
  (∀ k < 2013, adjacent (m k) (m (k + 1))) ∧
  (∀ k < 2013, let p := m k in
  let q := m (k + 1) in p ≠ q)

theorem mouse_wins : winning_strategy_for_mouse :=
sorry

end mouse_wins_l380_380763


namespace max_magnitude_l380_380611

def vector (α : Type*) := (α × α × α)

variables {V : Type*} [inner_product_space ℝ V] 

-- Given conditions
variables (a b : V)
variable (norm_a : ∥a∥ = 1)
variable (norm_b : ∥b∥ = 1)

theorem max_magnitude : ∃ (k : ℝ), k = 3 ∧ ∀ u v : V, ∥u∥ = 1 → ∥v∥ = 1 → ∥u + 2•v∥ ≤ k :=
sorry

end max_magnitude_l380_380611


namespace fg_minus_gf_l380_380304

def f(x : ℕ) : ℕ := x + 3
def g(x : ℕ) : ℕ := 3 * x + 5

theorem fg_minus_gf : f(g(4)) - g(f(4)) = -6 := by
  sorry

end fg_minus_gf_l380_380304


namespace complex_conjugate_of_z_l380_380864

def z : ℂ := (5 * complex.I) / ((2 - complex.I) * (2 + complex.I))

theorem complex_conjugate_of_z : complex.conj z = -complex.I := by
  -- Proof will be provided.
  sorry

end complex_conjugate_of_z_l380_380864


namespace no_product_equal_remainder_l380_380126

theorem no_product_equal_remainder (n : ℤ) : 
  ¬ (n = (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 1) = n * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 2) = n * (n + 1) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 3) = n * (n + 1) * (n + 2) * (n + 4) * (n + 5) ∨
     (n + 4) = n * (n + 1) * (n + 2) * (n + 3) * (n + 5) ∨
     (n + 5) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end no_product_equal_remainder_l380_380126


namespace roofs_have_equal_surface_area_l380_380984

def slant_height (a h : ℝ) : ℝ :=
  real.sqrt ((a / 2) ^ 2 + h ^ 2)

def area_gable_roof (a h : ℝ) : ℝ :=
  2 * a * slant_height a h

def area_hip_roof (a h : ℝ) : ℝ :=
  2 * a * slant_height a h

theorem roofs_have_equal_surface_area (a h : ℝ) :
  area_gable_roof a h = area_hip_roof a h :=
by
  sorry -- Proof needs to be constructed

end roofs_have_equal_surface_area_l380_380984


namespace increase_in_green_chameleons_is_11_l380_380318

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l380_380318


namespace side_length_of_square_l380_380450

theorem side_length_of_square (r : ℝ) (A : ℝ) (s : ℝ) 
  (h1 : π * r^2 = 36 * π) 
  (h2 : s = 2 * r) : 
  s = 12 :=
by 
  sorry

end side_length_of_square_l380_380450


namespace largest_invertible_interval_l380_380142

noncomputable def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

theorem largest_invertible_interval : 
  ∃ I : set ℝ, (∀ x ∈ I, g x = g x) ∧ (-1 : ℝ) ∈ I ∧ (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → ∀ y, g x₁ = y → g x₂ = y → x₁ = x₂) ∧ I = set.Iic (-1) := 
begin
  sorry
end

end largest_invertible_interval_l380_380142


namespace digit_150th_of_17_div_70_is_7_l380_380025

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end digit_150th_of_17_div_70_is_7_l380_380025


namespace at_least_one_negative_after_10_steps_l380_380512

noncomputable def transformation (x y z : ℤ) : ℤ × ℤ × ℤ :=
  (y + z - x, z + x - y, x + y - z)

theorem at_least_one_negative_after_10_steps
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_sum : a + b + c = 2013) :
  ∃ (i : ℕ), i ≤ 10 → let (x, y, z) := Nat.iterate (λ t => transformation t.1 t.2 t.3) i (a, b, c) in x < 0 ∨ y < 0 ∨ z < 0 :=
begin
  sorry
end

end at_least_one_negative_after_10_steps_l380_380512


namespace correct_operation_l380_380067

theorem correct_operation : 
  (∀ (a b : ℝ), ¬(a^2 * a^3 = a^6) ∧ ¬((a^2)^3 = a^5) ∧ (∀ (a b : ℝ), (a * b)^3 = a^3 * b^3) ∧ ¬(a^8 / a^2 = a^4)) :=
by
  intros a b
  split
  -- proof for ¬(a^2 * a^3 = a^6)
  sorry
  split
  -- proof for ¬((a^2)^3 = a^5)
  sorry
  split
  -- proof for (a * b)^3 = a^3 * b^3
  sorry
  -- proof for ¬(a^8 / a^2 = a^4)
  sorry

end correct_operation_l380_380067


namespace expected_deliveries_2017_l380_380818

-- Definitions based on the problem statement
noncomputable def expected_deliveries : ℕ → ℚ
| 2     := 1
| (n+2) := 1 + (n / (↑n + 1)) * expected_deliveries (n + 1)

-- The main theorem proving the expected number of deliveries for 2017 friends
theorem expected_deliveries_2017 : expected_deliveries 2017 = (∑ k in finset.range 2016, 1 / (k + 1)) :=
by
  sorry

end expected_deliveries_2017_l380_380818


namespace trigonometric_identities_l380_380937

theorem trigonometric_identities 
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : sin β = -4 / 5)
  (h4 : ∥(cos α - cos β, sin α - sin β)∥ = 4 * real.sqrt 13 / 13) :
  cos (α - β) = 5 / 13 ∧ sin α = 16 / 65 :=
by sorry

end trigonometric_identities_l380_380937


namespace circle_area_l380_380701

theorem circle_area (A B : ℝ × ℝ) (a b : ℝ)
  (h1 : A = (8, 17)) (h2 : B = (14, 15))
  (h3 : a = 10) (h4 : b = 5)
  (h5 : (A.1 - 8)^2 / a^2 + (A.2 - 17)^2 / b^2 = 1)
  (h6 : (B.1 - 14)^2 / a^2 + (B.2 - 15)^2 / b^2 = 1)
  (ω : ℝ × ℝ → Prop)
  (h7 : ∀P, ω P ↔ (P.1 - ((A.1 + B.1) / 2))^2 + (P.2 - ((A.2 + B.2) / 2))^2 = (A.1 - ((A.1 + B.1) / 2))^2 + (A.2 - ((A.2 + B.2) / 2))^2)
  (h8 : ∃ C : ℝ × ℝ, C.2 = 0 ∧ ∀ (P : ℝ × ℝ), ω P → (P = A ∨ P = B ∨ tangent_at C P ω))
  : (∀ r, r = ((A.1 - ((A.1 + B.1) / 2))^2 + (A.2 - ((A.2 + B.2) / 2))^2).sqrt → (area ω = π * r^2)) :=
sorry

end circle_area_l380_380701


namespace find_n_l380_380256

def satisfiesEquation(n : ℤ) : Prop :=
  (int.floor (n^2 / 4 : ℚ) - int.floor (n / 2 : ℚ)^2) = 5

theorem find_n : satisfiesEquation 11 :=
by sorry

end find_n_l380_380256


namespace candles_left_in_room_l380_380822

-- Define the variables and conditions
def total_candles : ℕ := 40
def alyssa_used : ℕ := total_candles / 2
def remaining_candles_after_alyssa : ℕ := total_candles - alyssa_used
def chelsea_used : ℕ := (7 * remaining_candles_after_alyssa) / 10
def final_remaining_candles : ℕ := remaining_candles_after_alyssa - chelsea_used

-- The theorem we need to prove
theorem candles_left_in_room : final_remaining_candles = 6 := by
  sorry

end candles_left_in_room_l380_380822


namespace three_digit_integers_count_l380_380939

def digit_set := {2, 3, 5, 5, 5, 6, 6}

theorem three_digit_integers_count : Finset.card (Finset.filter (λ n, 
    ∃ d1 d2 d3 : ℕ, d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ 
    (n = 100 * d1 + 10 * d2 + d3) ∧ 
    (d1 ∈ digit_set ∧ d2 ∈ digit_set ∧ d3 ∈ digit_set ∧ 
     (Finset.card (Finset.filter (λ x, x = 2) digit_set) ≥ 
     Finset.card (Finset.filter (λ x, x = d1) digit_set) ∧
     Finset.card (Finset.filter (λ x, x = 3) digit_set) ≥ 
     Finset.card (Finset.filter (λ x, x = d2) digit_set) ∧
     Finset.card (Finset.filter (λ x, x = 5) digit_set) ≥ 
     Finset.card (Finset.filter (λ x, x = d3) digit_set))) (Finset.range 1000)) = 43 :=
by sorry

end three_digit_integers_count_l380_380939


namespace new_team_average_l380_380080

theorem new_team_average (h7 : ∀ (weights : Fin 7 → ℝ), (∑ i, weights i) / 7 = 103)
  (w1 : ℝ) (w2 : ℝ) (hw1 : w1 = 110) (hw2 : w2 = 60) : 
  ∃ (new_avg : ℝ), new_avg = 99 :=
by
  sorry

end new_team_average_l380_380080


namespace num_real_solutions_eq_4_l380_380676

/-- Let ⌊x⌋ be the greatest integer less than or equal to x. Prove that the number of real solutions 
to the equation 3x² - 49⌊x⌋ + 100 = 0 is 4. -/
theorem num_real_solutions_eq_4 : 
  (∃ n : ℕ, 49 * n - 100 = 3 * x^2) -> ∀ x : ℝ, 3 * x^2 - 49 * floor(x) + 100 = 0 -> 
  ∃ c : ℕ, c = 4 := sorry

end num_real_solutions_eq_4_l380_380676


namespace larger_root_of_quadratic_eq_l380_380534

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l380_380534


namespace fraction_red_marbles_l380_380269

theorem fraction_red_marbles (x : ℕ) (h₁ : 2 / 3 ≤ 1) :
  let blue_marbles := (2 / 3) * x in
  let red_marbles := x - blue_marbles in
  let new_red_marbles := 2 * red_marbles in
  let total_marbles := blue_marbles + new_red_marbles in
  new_red_marbles / total_marbles = 1 / 2 :=
sorry

end fraction_red_marbles_l380_380269


namespace range_of_x_l380_380196

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) :
  (2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
   abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2)
  ↔ (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
by
  sorry

end range_of_x_l380_380196


namespace digit_150_is_7_l380_380036

theorem digit_150_is_7 : (0.242857242857242857 : ℚ) = (17/70 : ℚ) ∧ ( (17/70).decimal 150 = 7) := by
  sorry

end digit_150_is_7_l380_380036


namespace sum_of_solutions_l380_380867

theorem sum_of_solutions :
  let a := -48
  let b := 110
  let c := 165
  ( ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) → x1 ≠ x2 → (x1 + x2) = 55 / 24 ) :=
by
  let a := -48
  let b := 110
  let c := 165
  sorry

end sum_of_solutions_l380_380867


namespace two_presses_printing_time_l380_380261

def printing_time (presses newspapers hours : ℕ) : ℕ := sorry

theorem two_presses_printing_time :
  ∀ (presses newspapers hours : ℕ),
    (presses = 4) →
    (newspapers = 8000) →
    (hours = 6) →
    printing_time 2 6000 hours = 9 := sorry

end two_presses_printing_time_l380_380261
