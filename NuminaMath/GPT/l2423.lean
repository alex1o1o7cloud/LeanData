import Mathlib

namespace b_amount_l2423_242307

-- Define the conditions
def total_amount (a b : ℝ) : Prop := a + b = 1210
def fraction_condition (a b : ℝ) : Prop := (1/3) * a = (1/4) * b

-- Define the main theorem to prove B's amount
theorem b_amount (a b : ℝ) (h1 : total_amount a b) (h2 : fraction_condition a b) : b = 691.43 :=
sorry

end b_amount_l2423_242307


namespace sugar_initial_weight_l2423_242320

theorem sugar_initial_weight (packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) (used_percentage : ℝ)
  (h1 : packs = 30)
  (h2 : pack_weight = 350)
  (h3 : leftover = 50)
  (h4 : used_percentage = 0.60) : 
  (packs * pack_weight + leftover) = 10550 :=
by 
  sorry

end sugar_initial_weight_l2423_242320


namespace bridget_block_collection_l2423_242385

-- Defining the number of groups and blocks per group.
def num_groups : ℕ := 82
def blocks_per_group : ℕ := 10

-- Defining the total number of blocks calculation.
def total_blocks : ℕ := num_groups * blocks_per_group

-- Theorem stating the total number of blocks is 820.
theorem bridget_block_collection : total_blocks = 820 :=
  by
  sorry

end bridget_block_collection_l2423_242385


namespace JoggerDifference_l2423_242319

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end JoggerDifference_l2423_242319


namespace time_for_b_alone_l2423_242314

theorem time_for_b_alone (A B : ℝ) (h1 : A + B = 1 / 16) (h2 : A = 1 / 24) : B = 1 / 48 :=
by
  sorry

end time_for_b_alone_l2423_242314


namespace right_triangle_perimeter_l2423_242313

theorem right_triangle_perimeter (area : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (h_area : area = 120)
  (h_a : a = 24)
  (h_area_eq : area = (1/2) * a * b)
  (h_c : c^2 = a^2 + b^2) :
  a + b + c = 60 :=
by
  sorry

end right_triangle_perimeter_l2423_242313


namespace correct_arrangements_count_l2423_242388

def valid_arrangements_count : Nat :=
  let houses := ['O', 'R', 'B', 'Y', 'G']
  let arrangements := houses.permutations
  let valid_arr := arrangements.filter (fun a =>
    let o_idx := a.indexOf 'O'
    let r_idx := a.indexOf 'R'
    let b_idx := a.indexOf 'B'
    let y_idx := a.indexOf 'Y'
    let constraints_met :=
      o_idx < r_idx ∧       -- O before R
      b_idx < y_idx ∧       -- B before Y
      (b_idx + 1 != y_idx) ∧ -- B not next to Y
      (r_idx + 1 != b_idx) ∧ -- R not next to B
      (b_idx + 1 != r_idx)   -- symmetrical R not next to B

    constraints_met)
  valid_arr.length

theorem correct_arrangements_count : valid_arrangements_count = 5 :=
  by
    -- To be filled with proof steps.
    sorry

end correct_arrangements_count_l2423_242388


namespace stratified_sampling_example_l2423_242366

theorem stratified_sampling_example 
  (N : ℕ) (S : ℕ) (D : ℕ) 
  (hN : N = 1000) (hS : S = 50) (hD : D = 200) : 
  D * (S : ℝ) / (N : ℝ) = 10 := 
by
  sorry

end stratified_sampling_example_l2423_242366


namespace container_volume_ratio_l2423_242312

theorem container_volume_ratio
  (A B : ℚ)
  (H1 : 3/5 * A + 1/4 * B = 4/5 * B)
  (H2 : 3/5 * A = (4/5 * B - 1/4 * B)) :
  A / B = 11 / 12 :=
by
  sorry

end container_volume_ratio_l2423_242312


namespace negation_exists_equation_l2423_242355

theorem negation_exists_equation (P : ℝ → Prop) :
  (∃ x > 0, x^2 + 3 * x - 5 = 0) → ¬ (∃ x > 0, x^2 + 3 * x - 5 = 0) = ∀ x > 0, x^2 + 3 * x - 5 ≠ 0 :=
by sorry

end negation_exists_equation_l2423_242355


namespace problem_difference_l2423_242391

-- Define the sum of first n natural numbers
def sumFirstN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Define the rounding rule to the nearest multiple of 5
def roundToNearest5 (x : ℕ) : ℕ :=
  match x % 5 with
  | 0 => x
  | 1 => x - 1
  | 2 => x - 2
  | 3 => x + 2
  | 4 => x + 1
  | _ => x  -- This case is theoretically unreachable

-- Define the sum of the first n natural numbers after rounding to nearest 5
def sumRoundedFirstN (n : ℕ) : ℕ :=
  (List.range (n + 1)).map roundToNearest5 |>.sum

theorem problem_difference : sumFirstN 120 - sumRoundedFirstN 120 = 6900 := by
  sorry

end problem_difference_l2423_242391


namespace ineq_medians_triangle_l2423_242331

theorem ineq_medians_triangle (a b c s_a s_b s_c : ℝ)
  (h_mediana : s_a = 1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2))
  (h_medianb : s_b = 1 / 2 * Real.sqrt (2 * a^2 + 2 * c^2 - b^2))
  (h_medianc : s_c = 1 / 2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2))
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3 / 4) * (a + b + c) := 
sorry

end ineq_medians_triangle_l2423_242331


namespace simplify_fraction_l2423_242368

theorem simplify_fraction (h1 : 90 = 2 * 3^2 * 5) (h2 : 150 = 2 * 3 * 5^2) : (90 / 150 : ℚ) = 3 / 5 := by
  sorry

end simplify_fraction_l2423_242368


namespace angle_sum_x_y_l2423_242322

def angle_A := 36
def angle_B := 80
def angle_C := 24

def target_sum : ℕ := 140

theorem angle_sum_x_y (angle_A angle_B angle_C : ℕ) (x y : ℕ) : 
  angle_A = 36 → angle_B = 80 → angle_C = 24 → x + y = 140 := by 
  intros _ _ _
  sorry

end angle_sum_x_y_l2423_242322


namespace inverse_function_value_l2423_242378

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ y : ℝ, f (3^y) = y) : f 3 = 1 :=
sorry

end inverse_function_value_l2423_242378


namespace watch_current_price_l2423_242397

-- Definitions based on conditions
def original_price : ℝ := 15
def first_reduction_rate : ℝ := 0.25
def second_reduction_rate : ℝ := 0.40

-- The price after the first reduction
def first_reduced_price : ℝ := original_price * (1 - first_reduction_rate)

-- The price after the second reduction
def final_price : ℝ := first_reduced_price * (1 - second_reduction_rate)

-- The theorem that needs to be proved
theorem watch_current_price : final_price = 6.75 :=
by
  -- Proof goes here
  sorry

end watch_current_price_l2423_242397


namespace bottle_caps_left_l2423_242316

theorem bottle_caps_left {init_caps given_away_rebecca given_away_michael left_caps : ℝ} 
  (h1 : init_caps = 143.6)
  (h2 : given_away_rebecca = 89.2)
  (h3 : given_away_michael = 16.7)
  (h4 : left_caps = init_caps - (given_away_rebecca + given_away_michael)) :
  left_caps = 37.7 := by
  sorry

end bottle_caps_left_l2423_242316


namespace smallest_three_digit_number_with_property_l2423_242396

theorem smallest_three_digit_number_with_property :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 999 ∧ (∃ (n : ℕ), 317 ≤ n ∧ n ≤ 999 ∧ 1001 * a + 1 = n^2) ∧ a = 183 :=
by
  sorry

end smallest_three_digit_number_with_property_l2423_242396


namespace best_model_is_model1_l2423_242382

noncomputable def model_best_fitting (R1 R2 R3 R4 : ℝ) :=
  R1 = 0.975 ∧ R2 = 0.79 ∧ R3 = 0.55 ∧ R4 = 0.25

theorem best_model_is_model1 (R1 R2 R3 R4 : ℝ) (h : model_best_fitting R1 R2 R3 R4) :
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  cases h with
  | intro h1 h_rest =>
    cases h_rest with
    | intro h2 h_rest2 =>
      cases h_rest2 with
      | intro h3 h4 =>
        sorry

end best_model_is_model1_l2423_242382


namespace inequality_proof_l2423_242362

variable (m n : ℝ)

theorem inequality_proof (hm : m < 0) (hn : n > 0) (h_sum : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by
  -- introduction and proof commands would go here, but we use sorry to indicate the proof is omitted
  sorry

end inequality_proof_l2423_242362


namespace T_perimeter_is_20_l2423_242321

-- Define the perimeter of a rectangle given its length and width
def perimeter_rectangle (length width : ℝ) : ℝ :=
  2 * length + 2 * width

-- Given conditions
def rect1_length : ℝ := 1
def rect1_width : ℝ := 4
def rect2_length : ℝ := 2
def rect2_width : ℝ := 5
def overlap_height : ℝ := 1

-- Calculate the perimeter of each rectangle
def perimeter_rect1 : ℝ := perimeter_rectangle rect1_length rect1_width
def perimeter_rect2 : ℝ := perimeter_rectangle rect2_length rect2_width

-- Calculate the overlap adjustment
def overlap_adjustment : ℝ := 2 * overlap_height

-- The total perimeter of the T shape
def perimeter_T : ℝ := perimeter_rect1 + perimeter_rect2 - overlap_adjustment

-- The proof statement that we need to show
theorem T_perimeter_is_20 : perimeter_T = 20 := by
  sorry

end T_perimeter_is_20_l2423_242321


namespace div_ad_bc_by_k_l2423_242364

theorem div_ad_bc_by_k 
  (a b c d l k m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n) : 
  k ∣ (a * d - b * c) :=
sorry

end div_ad_bc_by_k_l2423_242364


namespace arianna_sleep_hours_l2423_242338

-- Defining the given conditions
def total_hours_in_a_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_in_class : ℕ := 3
def hours_at_gym : ℕ := 2
def hours_on_chores : ℕ := 5

-- Formulating the total hours spent on activities
def total_hours_on_activities := hours_at_work + hours_in_class + hours_at_gym + hours_on_chores

-- Proving Arianna's sleep hours
theorem arianna_sleep_hours : total_hours_in_a_day - total_hours_on_activities = 8 :=
by
  -- Direct proof placeholder, to be filled in with actual proof steps or tactic
  sorry

end arianna_sleep_hours_l2423_242338


namespace boat_problem_l2423_242324

theorem boat_problem (x n : ℕ) (h1 : n = 7 * x + 5) (h2 : n = 8 * x - 2) :
  n = 54 ∧ x = 7 := by
sorry

end boat_problem_l2423_242324


namespace a_not_multiple_of_5_l2423_242337

theorem a_not_multiple_of_5 (a : ℤ) (h : a % 5 ≠ 0) : (a^4 + 4) % 5 = 0 :=
sorry

end a_not_multiple_of_5_l2423_242337


namespace find_second_equation_value_l2423_242350

theorem find_second_equation_value:
  (∃ x y : ℝ, 2 * x + y = 26 ∧ (x + y) / 3 = 4) →
  (∃ x y : ℝ, 2 * x + y = 26 ∧ x + 2 * y = 10) :=
by
  sorry

end find_second_equation_value_l2423_242350


namespace triangle_equilateral_l2423_242325

noncomputable def point := (ℝ × ℝ)

noncomputable def D : point := (0, 0)
noncomputable def E : point := (2, 0)
noncomputable def F : point := (1, Real.sqrt 3)

noncomputable def dist (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def D' (l : ℝ) : point :=
  let ED := dist E D
  (D.1 + l * ED * (Real.sqrt 3), D.2 + l * ED)

noncomputable def E' (l : ℝ) : point :=
  let DF := dist D F
  (E.1 + l * DF * (Real.sqrt 3), E.2 + l * DF)

noncomputable def F' (l : ℝ) : point :=
  let DE := dist D E
  (F.1 - 2 * l * DE, F.2 + (Real.sqrt 3 - l * DE))

theorem triangle_equilateral (l : ℝ) (h : l = 1 / Real.sqrt 3) :
  let DD' := dist D (D' l)
  let EE' := dist E (E' l)
  let FF' := dist F (F' l)
  dist (D' l) (E' l) = dist (E' l) (F' l) ∧ dist (E' l) (F' l) = dist (F' l) (D' l) ∧ dist (F' l) (D' l) = dist (D' l) (E' l) := sorry

end triangle_equilateral_l2423_242325


namespace multiplier_for_difference_l2423_242349

variable (x y k : ℕ)
variable (h1 : x + y = 81)
variable (h2 : x^2 - y^2 = k * (x - y))
variable (h3 : x ≠ y)

theorem multiplier_for_difference : k = 81 := 
by
  sorry

end multiplier_for_difference_l2423_242349


namespace stratified_sampling_distribution_l2423_242301

/-- A high school has a total of 2700 students, among which there are 900 freshmen, 
1200 sophomores, and 600 juniors. Using stratified sampling, a sample of 135 students 
is drawn. Prove that the sample contains 45 freshmen, 60 sophomores, and 30 juniors --/
theorem stratified_sampling_distribution :
  let total_students := 2700
  let freshmen := 900
  let sophomores := 1200
  let juniors := 600
  let sample_size := 135
  (sample_size * freshmen / total_students = 45) ∧ 
  (sample_size * sophomores / total_students = 60) ∧ 
  (sample_size * juniors / total_students = 30) :=
by
  sorry

end stratified_sampling_distribution_l2423_242301


namespace cost_of_bananas_and_cantaloupe_l2423_242323

-- Define variables representing the prices
variables (a b c d : ℝ)

-- Define the given conditions as hypotheses
def conditions : Prop :=
  a + b + c + d = 33 ∧
  d = 3 * a ∧
  c = a + 2 * b

-- State the main theorem
theorem cost_of_bananas_and_cantaloupe (h : conditions a b c d) : b + c = 13 :=
by {
  sorry
}

end cost_of_bananas_and_cantaloupe_l2423_242323


namespace simplify_expression_l2423_242339

theorem simplify_expression : (Real.cos (18 * Real.pi / 180) * Real.cos (42 * Real.pi / 180) - 
                              Real.cos (72 * Real.pi / 180) * Real.sin (42 * Real.pi / 180) = 1 / 2) :=
by
  sorry

end simplify_expression_l2423_242339


namespace find_greatest_K_l2423_242302

theorem find_greatest_K {u v w K : ℝ} (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu2_gt_4vw : u^2 > 4 * v * w) :
  (u^2 - 4 * v * w)^2 > K * (2 * v^2 - u * w) * (2 * w^2 - u * v) ↔ K ≤ 16 := 
sorry

end find_greatest_K_l2423_242302


namespace find_first_number_l2423_242375

variable {A B C D : ℕ}

theorem find_first_number (h1 : A + B + C = 60) (h2 : B + C + D = 45) (h3 : D = 18) : A = 33 := 
  sorry

end find_first_number_l2423_242375


namespace profit_percent_approx_l2423_242374

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 30
noncomputable def selling_price : ℝ := 300

noncomputable def cost_price : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit / cost_price) * 100

theorem profit_percent_approx :
  purchase_price = 225 ∧ 
  overhead_expenses = 30 ∧ 
  selling_price = 300 → 
  abs (profit_percent - 17.65) < 0.01 := 
by 
  -- Proof omitted
  sorry

end profit_percent_approx_l2423_242374


namespace molecular_weight_compound_l2423_242305

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_Cl : ℝ := 35.453

def molecular_weight (nH nC nO nN nCl : ℕ) : ℝ :=
  nH * atomic_weight_H + nC * atomic_weight_C + nO * atomic_weight_O + nN * atomic_weight_N + nCl * atomic_weight_Cl

theorem molecular_weight_compound :
  molecular_weight 4 2 3 1 2 = 160.964 := by
  sorry

end molecular_weight_compound_l2423_242305


namespace min_xy_min_x_plus_y_l2423_242328

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 := 
sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 := 
sorry

end min_xy_min_x_plus_y_l2423_242328


namespace square_side_length_l2423_242381

theorem square_side_length (x : ℝ) (h : x ^ 2 = 4 * 3) : x = 2 * Real.sqrt 3 :=
by sorry

end square_side_length_l2423_242381


namespace total_students_in_high_school_l2423_242330

-- Definitions based on the problem conditions
def freshman_students : ℕ := 400
def sample_students : ℕ := 45
def sophomore_sample_students : ℕ := 15
def senior_sample_students : ℕ := 10

-- The theorem to be proved
theorem total_students_in_high_school : (sample_students = 45) → (freshman_students = 400) → (sophomore_sample_students = 15) → (senior_sample_students = 10) → ∃ total_students : ℕ, total_students = 900 :=
by
  sorry

end total_students_in_high_school_l2423_242330


namespace scientific_notation_coronavirus_diameter_l2423_242398

theorem scientific_notation_coronavirus_diameter : 0.00000011 = 1.1 * 10^(-7) :=
by {
  sorry
}

end scientific_notation_coronavirus_diameter_l2423_242398


namespace calculate_two_times_square_root_squared_l2423_242315

theorem calculate_two_times_square_root_squared : 2 * (Real.sqrt 50625) ^ 2 = 101250 := by
  sorry

end calculate_two_times_square_root_squared_l2423_242315


namespace perpendicular_line_equation_l2423_242310

theorem perpendicular_line_equation :
  (∀ (x y : ℝ), 2 * x + 3 * y + 1 = 0 → x - 3 * y + 4 = 0 →
  ∃ (l : ℝ) (m : ℝ), m = 4 / 3 ∧ y = m * x + l → y = 4 / 3 * x + 1 / 9) 
  ∧ (∀ (x y : ℝ), 3 * x + 4 * y - 7 = 0 → -3 / 4 * 4 / 3 = -1) :=
by 
  sorry

end perpendicular_line_equation_l2423_242310


namespace trigonometric_identity_simplification_l2423_242395

open Real

theorem trigonometric_identity_simplification (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  (sqrt (1 - 2 * sin (3 * π - θ) * sin (π / 2 + θ)) = cos θ - sin θ) :=
sorry

end trigonometric_identity_simplification_l2423_242395


namespace intersection_complement_l2423_242326

def A : Set ℝ := { x | x^2 ≤ 4 * x }
def B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 3) }

theorem intersection_complement (x : ℝ) : 
  x ∈ A ∩ (Set.univ \ B) ↔ x ∈ Set.Ico 0 3 := 
sorry

end intersection_complement_l2423_242326


namespace function_single_intersection_l2423_242306

theorem function_single_intersection (a : ℝ) : 
  (∃ x : ℝ, ax^2 - x + 1 = 0 ∧ ∀ y : ℝ, (ax^2 - x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1/4) :=
sorry

end function_single_intersection_l2423_242306


namespace max_value_of_f_f_is_increasing_on_intervals_l2423_242300

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem max_value_of_f :
  ∃ (k : ℤ), ∀ (x : ℝ), x = k * Real.pi + Real.pi / 6 → f x = 3 :=
sorry

theorem f_is_increasing_on_intervals :
  ∀ (k : ℤ), ∀ (x y : ℝ), k * Real.pi - Real.pi / 3 ≤ x →
                x ≤ y → y ≤ k * Real.pi + Real.pi / 6 →
                f x ≤ f y :=
sorry

end max_value_of_f_f_is_increasing_on_intervals_l2423_242300


namespace trapezoid_PQRS_PQ_squared_l2423_242353

theorem trapezoid_PQRS_PQ_squared
  (PR PS PQ : ℝ)
  (cond1 : PR = 13)
  (cond2 : PS = 17)
  (h : PQ^2 + PR^2 = PS^2) :
  PQ^2 = 120 :=
by
  rw [cond1, cond2] at h
  sorry

end trapezoid_PQRS_PQ_squared_l2423_242353


namespace irreducible_fraction_eq_l2423_242341

theorem irreducible_fraction_eq (p q : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : Nat.gcd p q = 1) (h4 : q % 2 = 1) :
  ∃ n k : ℕ, n > 0 ∧ k > 0 ∧ (p : ℚ) / q = (n : ℚ) / (2 ^ k - 1) :=
by
  sorry

end irreducible_fraction_eq_l2423_242341


namespace profit_percentage_l2423_242379

theorem profit_percentage (CP SP : ℝ) (h₁ : CP = 400) (h₂ : SP = 560) : 
  ((SP - CP) / CP) * 100 = 40 := by 
  sorry

end profit_percentage_l2423_242379


namespace transform_equation_l2423_242318

theorem transform_equation (x : ℝ) :
  x^2 + 4 * x + 1 = 0 → (x + 2)^2 = 3 :=
by
  intro h
  sorry

end transform_equation_l2423_242318


namespace intersection_sets_l2423_242376

theorem intersection_sets :
  let M := {x : ℝ | 0 < x} 
  let N := {y : ℝ | 1 ≤ y}
  M ∩ N = {z : ℝ | 1 ≤ z} :=
by
  -- Proof goes here
  sorry

end intersection_sets_l2423_242376


namespace polynomial_solution_l2423_242346

theorem polynomial_solution (f : ℝ → ℝ) (x : ℝ) (h : f (x^2 + 2) = x^4 + 6 * x^2 + 4) : 
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  sorry

end polynomial_solution_l2423_242346


namespace negation_of_universal_proposition_l2423_242377

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5 * x = 4) ↔ (∃ x : ℝ, x^2 + 5 * x ≠ 4) :=
by
  sorry

end negation_of_universal_proposition_l2423_242377


namespace magnitude_vec_sum_l2423_242372

open Real

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

theorem magnitude_vec_sum
    (a b : ℝ × ℝ)
    (h_angle : ∃ θ, θ = 150 * (π / 180) ∧ cos θ = cos (5 * π / 6))
    (h_norm_a : ‖a‖ = sqrt 3)
    (h_norm_b : ‖b‖ = 2) :
  ‖(2 * a.1 + b.1, 2 * a.2 + b.2)‖ = 2 :=
  by
  sorry

end magnitude_vec_sum_l2423_242372


namespace Phil_earns_per_hour_l2423_242348

-- Definitions based on the conditions in the problem
def Mike_hourly_rate : ℝ := 14
def Phil_hourly_rate : ℝ := Mike_hourly_rate - (0.5 * Mike_hourly_rate)

-- Mathematical assertion to prove
theorem Phil_earns_per_hour : Phil_hourly_rate = 7 :=
by 
  sorry

end Phil_earns_per_hour_l2423_242348


namespace intersection_point_l2423_242360

theorem intersection_point : ∃ (x y : ℝ), y = 3 - x ∧ y = 3 * x - 5 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l2423_242360


namespace distance_inequality_l2423_242342

theorem distance_inequality (a : ℝ) (h : |a - 1| < 3) : -2 < a ∧ a < 4 :=
sorry

end distance_inequality_l2423_242342


namespace arithmetic_sequence_sum_l2423_242394

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h_arithmetic : ∀ n, a (n+1) = a n + d)
  (h_pos_diff : d > 0)
  (h_sum_3 : a 0 + a 1 + a 2 = 15)
  (h_prod_3 : a 0 * a 1 * a 2 = 80) :
  a 10 + a 11 + a 12 = 105 :=
sorry

end arithmetic_sequence_sum_l2423_242394


namespace sum_remainder_div_9_l2423_242352

theorem sum_remainder_div_9 : 
  let S := (20 / 2) * (1 + 20)
  S % 9 = 3 := 
by
  -- use let S to simplify the proof
  let S := (20 / 2) * (1 + 20)
  -- sum of first 20 natural numbers
  have H1 : S = 210 := by sorry
  -- division and remainder result
  have H2 : 210 % 9 = 3 := by sorry
  -- combine both results to conclude 
  exact H2

end sum_remainder_div_9_l2423_242352


namespace avg_xy_l2423_242393

theorem avg_xy (x y : ℝ) (h : (4 + 6.5 + 8 + x + y) / 5 = 18) : (x + y) / 2 = 35.75 :=
by
  sorry

end avg_xy_l2423_242393


namespace subset_123_12_false_l2423_242303

-- Definitions derived from conditions
def is_int (x : ℤ) := true
def subset_123_12 (A B : Set ℕ) := A = {1, 2, 3} ∧ B = {1, 2}
def intersection_empty {A B : Set ℕ} (hA : A = {1, 2}) (hB : B = ∅) := (A ∩ B = ∅)
def union_nat_real {A B : Set ℝ} (hA : Set.univ ⊆ A) (hB : Set.univ ⊆ B) := (A ∪ B)

-- The mathematically equivalent proof problem
theorem subset_123_12_false (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {1, 2}):
  ¬ (A ⊆ B) :=
by
  sorry

end subset_123_12_false_l2423_242303


namespace evaluate_expression_l2423_242345

-- Define the terms a and b
def a : ℕ := 2023
def b : ℕ := 2024

-- The given expression
def expression : ℤ := (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b)

-- The theorem to prove
theorem evaluate_expression : expression = ↑a := 
by sorry

end evaluate_expression_l2423_242345


namespace total_volume_of_all_cubes_l2423_242370

def volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume_of_cubes (num_cubes : ℕ) (side_length : ℕ) : ℕ :=
  num_cubes * volume side_length

theorem total_volume_of_all_cubes :
  total_volume_of_cubes 3 3 + total_volume_of_cubes 4 4 = 337 :=
by
  sorry

end total_volume_of_all_cubes_l2423_242370


namespace bob_distance_when_meet_l2423_242347

def distance_xy : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def time_start_diff : ℝ := 1

theorem bob_distance_when_meet : ∃ t : ℝ, yolanda_rate * t + bob_rate * (t - time_start_diff) = distance_xy ∧ bob_rate * (t - time_start_diff) = 4 :=
by
  sorry

end bob_distance_when_meet_l2423_242347


namespace find_working_hours_for_y_l2423_242335

theorem find_working_hours_for_y (Wx Wy Wz Ww : ℝ) (h1 : Wx = 1/8)
  (h2 : Wy + Wz = 1/6) (h3 : Wx + Wz = 1/4) (h4 : Wx + Wy + Ww = 1/5)
  (h5 : Wx + Ww + Wz = 1/3) : 1 / Wy = 24 :=
by
  -- Given the conditions
  -- Wx = 1/8
  -- Wy + Wz = 1/6
  -- Wx + Wz = 1/4
  -- Wx + Wy + Ww = 1/5
  -- Wx + Ww + Wz = 1/3
  -- We need to prove that 1 / Wy = 24
  sorry

end find_working_hours_for_y_l2423_242335


namespace point_in_fourth_quadrant_l2423_242304

def point : ℝ × ℝ := (1, -2)

def is_fourth_quadrant (p: ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l2423_242304


namespace problem_proof_l2423_242389

-- Define the geometric sequence and vectors conditions
variables (a : ℕ → ℝ) (q : ℝ)
variables (h1 : ∀ n, a (n + 1) = q * a n)
variables (h2 : a 2 = a 2)
variables (h3 : a 3 = q * a 2)
variables (h4 : 3 * a 2 = 2 * a 3)

-- Statement to prove
theorem problem_proof:
  (a 2 + a 4) / (a 3 + a 5) = 2 / 3 :=
  sorry

end problem_proof_l2423_242389


namespace geo_seq_product_l2423_242317

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geo : ∀ n, a (n + 1) = a n * r) 
  (h_roots : a 1 ^ 2 - 10 * a 1 + 16 = 0) 
  (h_root19 : a 19 ^ 2 - 10 * a 19 + 16 = 0) : 
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end geo_seq_product_l2423_242317


namespace prescription_duration_l2423_242380

theorem prescription_duration (D : ℕ) (h1 : (2 * D) * (1 / 5) = 12) : D = 30 :=
by
  sorry

end prescription_duration_l2423_242380


namespace solve_inequality_l2423_242343

theorem solve_inequality (x : ℝ) :
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 :=
  sorry

end solve_inequality_l2423_242343


namespace part1_part2_l2423_242387

theorem part1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (2 / (1 + a)) + (2 / (1 + b)) + (2 / (1 + c)) :=
sorry

end part1_part2_l2423_242387


namespace different_signs_abs_value_larger_l2423_242356

variable {a b : ℝ}

theorem different_signs_abs_value_larger (h1 : a + b < 0) (h2 : ab < 0) : 
  (a > 0 ∧ b < 0 ∧ |a| < |b|) ∨ (a < 0 ∧ b > 0 ∧ |b| < |a|) :=
sorry

end different_signs_abs_value_larger_l2423_242356


namespace trigonometric_identity_l2423_242351

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (π / 4 + θ) = 3) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
sorry

end trigonometric_identity_l2423_242351


namespace sqrt_expression_evaluation_l2423_242392

theorem sqrt_expression_evaluation : 
  (Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2) := 
by
  sorry

end sqrt_expression_evaluation_l2423_242392


namespace smallest_positive_integer_n_l2423_242332

theorem smallest_positive_integer_n :
  ∃ n: ℕ, (n > 0) ∧ (∀ k: ℕ, 1 ≤ k ∧ k ≤ n → (∃ d: ℕ, d ∣ (n^2 - 2 * n) ∧ d ∣ k) ∧ (k ∣ (n^2 - 2 * n) → k = d)) ∧ n = 5 :=
by
  sorry

end smallest_positive_integer_n_l2423_242332


namespace first_tap_time_l2423_242334

-- Define the variables and conditions
variables (T : ℝ)
-- The cistern can be emptied by the second tap in 9 hours
-- Both taps together fill the cistern in 7.2 hours.
def first_tap_fills_cistern_in_time (T : ℝ) :=
  (1 / T) - (1 / 9) = 1 / 7.2

theorem first_tap_time :
  first_tap_fills_cistern_in_time 4 :=
by
  -- now we can use the definition to show the proof
  unfold first_tap_fills_cistern_in_time
  -- directly substitute and show
  sorry

end first_tap_time_l2423_242334


namespace train_a_distance_traveled_l2423_242336

variable (distance : ℝ) (speedA : ℝ) (speedB : ℝ) (relative_speed : ℝ) (time_to_meet : ℝ) 

axiom condition1 : distance = 450
axiom condition2 : speedA = 50
axiom condition3 : speedB = 50
axiom condition4 : relative_speed = speedA + speedB
axiom condition5 : time_to_meet = distance / relative_speed

theorem train_a_distance_traveled : (50 * time_to_meet) = 225 := by
  sorry

end train_a_distance_traveled_l2423_242336


namespace brick_laying_days_l2423_242369

theorem brick_laying_days (a m n d : ℕ) (hm : 0 < m) (hn : 0 < n) (hd : 0 < d) :
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  (a * rate_M * (d * total_days) + 2 * a * rate_N * (d * total_days)) = (a + 2 * a) :=
by
  -- Definitions from the problem conditions
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  have h0 : a * rate_M * (d * total_days) = a := sorry
  have h1 : 2 * a * rate_N * (d * total_days) = 2 * a := sorry
  exact sorry

end brick_laying_days_l2423_242369


namespace combined_weight_of_Meg_and_Chris_cats_l2423_242311

-- Definitions based on the conditions
def ratio (M A C : ℕ) : Prop := 13 * A = 21 * M ∧ 13 * C = 28 * M 
def half_anne (M A : ℕ) : Prop := M = 20 + A / 2
def total_weight (M A C T : ℕ) : Prop := T = M + A + C

-- Theorem statement
theorem combined_weight_of_Meg_and_Chris_cats (M A C T : ℕ) 
  (h1 : ratio M A C) 
  (h2 : half_anne M A) 
  (h3 : total_weight M A C T) : 
  M + C = 328 := 
sorry

end combined_weight_of_Meg_and_Chris_cats_l2423_242311


namespace mark_asphalt_total_cost_l2423_242384

noncomputable def total_cost (road_length : ℕ) (road_width : ℕ) (area_per_truckload : ℕ) (cost_per_truckload : ℕ) (sales_tax_rate : ℚ) : ℚ :=
  let total_area := road_length * road_width
  let num_truckloads := total_area / area_per_truckload
  let cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := cost_before_tax * sales_tax_rate
  let total_cost := cost_before_tax + sales_tax
  total_cost

theorem mark_asphalt_total_cost :
  total_cost 2000 20 800 75 0.2 = 4500 := 
by sorry

end mark_asphalt_total_cost_l2423_242384


namespace cube_volume_l2423_242333

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 864) : s^3 = 1728 := 
by
  sorry

end cube_volume_l2423_242333


namespace hyperbola_eccentricity_l2423_242340

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  let e := (1 + (b^2) / (a^2)).sqrt
  e

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a + b = 5)
  (h2 : a * b = 6)
  (h3 : a > b) :
  eccentricity a b = Real.sqrt 13 / 3 :=
sorry

end hyperbola_eccentricity_l2423_242340


namespace students_divided_into_groups_l2423_242309

theorem students_divided_into_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) (n_groups : ℕ) 
  (h1 : total_students = 64) 
  (h2 : not_picked = 36) 
  (h3 : students_per_group = 7) 
  (h4 : total_students - not_picked = 28) 
  (h5 : 28 / students_per_group = 4) :
  n_groups = 4 :=
by
  sorry

end students_divided_into_groups_l2423_242309


namespace base_three_to_base_ten_l2423_242373

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end base_three_to_base_ten_l2423_242373


namespace find_common_ratio_sum_arithmetic_sequence_l2423_242354

-- Conditions
variable {a : ℕ → ℝ}   -- a_n is a numeric sequence
variable (S : ℕ → ℝ)   -- S_n is the sum of the first n terms
variable {q : ℝ}       -- q is the common ratio
variable (k : ℕ)

-- Given: a_n is a geometric sequence with common ratio q, q ≠ 1, q ≠ 0
variable (h_geometric : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (h_q_ne_zero : q ≠ 0)

-- Given: S_n = a_1 * (1 - q^n) / (1 - q) when q ≠ 1 and q ≠ 0
variable (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))

-- Given: a_5, a_3, a_4 form an arithmetic sequence, so 2a_3 = a_5 + a_4
variable (h_arithmetic : 2 * a 3 = a 5 + a 4)

-- Prove part 1: common ratio q is -2
theorem find_common_ratio (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : 2 * a 3 = a 5 + a 4) 
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0) : q = -2 :=
sorry

-- Prove part 2: S_(k+2), S_k, S_(k+1) form an arithmetic sequence
theorem sum_arithmetic_sequence (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0)
  (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))
  (k : ℕ) : S (k + 2) + S k = 2 * S (k + 1) :=
sorry

end find_common_ratio_sum_arithmetic_sequence_l2423_242354


namespace token_exchange_l2423_242399

def booth1 (r : ℕ) (x : ℕ) : ℕ × ℕ × ℕ := (r - 3 * x, 2 * x, x)
def booth2 (b : ℕ) (y : ℕ) : ℕ × ℕ × ℕ := (y, b - 4 * y, y)

theorem token_exchange (x y : ℕ) (h1 : 100 - 3 * x + y = 2) (h2 : 50 + x - 4 * y = 3) :
  x + y = 58 :=
sorry

end token_exchange_l2423_242399


namespace total_area_of_sheet_l2423_242363

theorem total_area_of_sheet (x : ℕ) (h1 : 4 * x - x = 2208) : x + 4 * x = 3680 := 
sorry

end total_area_of_sheet_l2423_242363


namespace initial_number_306_l2423_242344

theorem initial_number_306 (x : ℝ) : 
  (x / 34) * 15 + 270 = 405 → x = 306 :=
by
  intro h
  sorry

end initial_number_306_l2423_242344


namespace Victor_bought_6_decks_l2423_242371

theorem Victor_bought_6_decks (V : ℕ) (h1 : 2 * 8 + 8 * V = 64) : V = 6 := by
  sorry

end Victor_bought_6_decks_l2423_242371


namespace bonnie_roark_wire_ratio_l2423_242357

theorem bonnie_roark_wire_ratio :
  let bonnie_wire_length := 12 * 8
  let bonnie_cube_volume := 8 ^ 3
  let roark_cube_volume := 2
  let roark_edge_length := 1.5
  let roark_cube_edge_count := 12
  let num_roark_cubes := bonnie_cube_volume / roark_cube_volume
  let roark_wire_per_cube := roark_cube_edge_count * roark_edge_length
  let roark_total_wire := num_roark_cubes * roark_wire_per_cube
  bonnie_wire_length / roark_total_wire = 1 / 48 :=
  by
  sorry

end bonnie_roark_wire_ratio_l2423_242357


namespace hours_worked_on_saturday_l2423_242327

-- Definitions from the problem conditions
def hourly_wage : ℝ := 15
def hours_friday : ℝ := 10
def hours_sunday : ℝ := 14
def total_earnings : ℝ := 450

-- Define number of hours worked on Saturday as a variable
variable (hours_saturday : ℝ)

-- Total earnings can be expressed as the sum of individual day earnings
def total_earnings_eq : Prop := 
  total_earnings = (hours_friday * hourly_wage) + (hours_sunday * hourly_wage) + (hours_saturday * hourly_wage)

-- Prove that the hours worked on Saturday is 6
theorem hours_worked_on_saturday :
  total_earnings_eq hours_saturday →
  hours_saturday = 6 := by
  sorry

end hours_worked_on_saturday_l2423_242327


namespace bananas_more_than_pears_l2423_242365

theorem bananas_more_than_pears (A P B : ℕ) (h1 : P = A + 2) (h2 : A + P + B = 19) (h3 : B = 9) : B - P = 3 :=
  by
  sorry

end bananas_more_than_pears_l2423_242365


namespace cube_surface_area_correct_l2423_242390

def edge_length : ℝ := 11

def cube_surface_area (e : ℝ) : ℝ := 6 * e^2

theorem cube_surface_area_correct : cube_surface_area edge_length = 726 := by
  sorry

end cube_surface_area_correct_l2423_242390


namespace zoe_calories_l2423_242361

theorem zoe_calories 
  (s : ℕ) (y : ℕ) (c_s : ℕ) (c_y : ℕ)
  (s_eq : s = 12) (y_eq : y = 6) (cs_eq : c_s = 4) (cy_eq : c_y = 17) :
  s * c_s + y * c_y = 150 :=
by
  sorry

end zoe_calories_l2423_242361


namespace range_of_m_l2423_242367

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + m > 0) ↔ 0 < m ∧ m < 4 :=
by sorry

end range_of_m_l2423_242367


namespace point_A_inside_circle_O_l2423_242358

-- Definitions based on conditions in the problem
def radius := 5 -- in cm
def distance_to_center := 4 -- in cm

-- The theorem to be proven
theorem point_A_inside_circle_O (r d : ℝ) (hr : r = 5) (hd : d = 4) (h : r > d) : true :=
by {
  sorry
}

end point_A_inside_circle_O_l2423_242358


namespace percentage_differences_equal_l2423_242359

noncomputable def calculation1 : ℝ := 0.60 * 50
noncomputable def calculation2 : ℝ := 0.30 * 30
noncomputable def calculation3 : ℝ := 0.45 * 90
noncomputable def calculation4 : ℝ := 0.20 * 40

noncomputable def diff1 : ℝ := abs (calculation1 - calculation2)
noncomputable def diff2 : ℝ := abs (calculation2 - calculation3)
noncomputable def diff3 : ℝ := abs (calculation3 - calculation4)
noncomputable def largest_diff1 : ℝ := max diff1 (max diff2 diff3)

noncomputable def calculation5 : ℝ := 0.40 * 120
noncomputable def calculation6 : ℝ := 0.25 * 80
noncomputable def calculation7 : ℝ := 0.35 * 150
noncomputable def calculation8 : ℝ := 0.55 * 60

noncomputable def diff4 : ℝ := abs (calculation5 - calculation6)
noncomputable def diff5 : ℝ := abs (calculation6 - calculation7)
noncomputable def diff6 : ℝ := abs (calculation7 - calculation8)
noncomputable def largest_diff2 : ℝ := max diff4 (max diff5 diff6)

theorem percentage_differences_equal :
  largest_diff1 = largest_diff2 :=
sorry

end percentage_differences_equal_l2423_242359


namespace correct_calculation_l2423_242386

theorem correct_calculation (a b : ℝ) : (3 * a * b) ^ 2 = 9 * a ^ 2 * b ^ 2 :=
by
  sorry

end correct_calculation_l2423_242386


namespace ratio_of_areas_l2423_242329

theorem ratio_of_areas (T A B : ℝ) (hT : T = 900) (hB : B = 405) (hSum : A + B = T) :
  (A - B) / ((A + B) / 2) = 1 / 5 :=
by
  sorry

end ratio_of_areas_l2423_242329


namespace num_and_sum_of_divisors_of_36_l2423_242308

noncomputable def num_divisors_and_sum (n : ℕ) : ℕ × ℕ :=
  let divisors := (List.range (n + 1)).filter (λ x => n % x = 0)
  (divisors.length, divisors.sum)

theorem num_and_sum_of_divisors_of_36 : num_divisors_and_sum 36 = (9, 91) := by
  sorry

end num_and_sum_of_divisors_of_36_l2423_242308


namespace min_value_of_quadratic_l2423_242383

theorem min_value_of_quadratic : ∀ x : ℝ, (x^2 + 6*x + 5) ≥ -4 :=
by 
  sorry

end min_value_of_quadratic_l2423_242383
