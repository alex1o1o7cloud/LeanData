import Mathlib

namespace NUMINAMATH_GPT_mean_of_remaining_students_l1400_140060

variable (k : ℕ) (h1 : k > 20)

def mean_of_class (mean : ℝ := 10) := mean
def mean_of_20_students (mean : ℝ := 16) := mean

theorem mean_of_remaining_students 
  (h2 : mean_of_class = 10)
  (h3 : mean_of_20_students = 16) :
  let remaining_students := (k - 20)
  let total_score_20 := 20 * mean_of_20_students
  let total_score_class := k * mean_of_class
  let total_score_remaining := total_score_class - total_score_20
  let mean_remaining := total_score_remaining / remaining_students
  mean_remaining = (10 * k - 320) / (k - 20) :=
sorry

end NUMINAMATH_GPT_mean_of_remaining_students_l1400_140060


namespace NUMINAMATH_GPT_function_D_is_odd_function_D_is_decreasing_l1400_140063

def f_D (x : ℝ) : ℝ := -x * |x|

theorem function_D_is_odd (x : ℝ) : f_D (-x) = -f_D x := by
  sorry

theorem function_D_is_decreasing (x y : ℝ) (h : x < y) : f_D x > f_D y := by
  sorry

end NUMINAMATH_GPT_function_D_is_odd_function_D_is_decreasing_l1400_140063


namespace NUMINAMATH_GPT_fraction_equals_decimal_l1400_140074

theorem fraction_equals_decimal : (3 : ℝ) / 2 = 1.5 := 
sorry

end NUMINAMATH_GPT_fraction_equals_decimal_l1400_140074


namespace NUMINAMATH_GPT_tree_sidewalk_space_l1400_140014

theorem tree_sidewalk_space
  (num_trees : ℕ)
  (distance_between_trees : ℝ)
  (total_road_length : ℝ)
  (total_gaps : ℝ)
  (space_each_tree : ℝ)
  (H1 : num_trees = 11)
  (H2 : distance_between_trees = 14)
  (H3 : total_road_length = 151)
  (H4 : total_gaps = (num_trees - 1) * distance_between_trees)
  (H5 : space_each_tree = (total_road_length - total_gaps) / num_trees)
  : space_each_tree = 1 := 
by
  sorry

end NUMINAMATH_GPT_tree_sidewalk_space_l1400_140014


namespace NUMINAMATH_GPT_parallel_lines_l1400_140007

theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (3 * a - 1) * x - 4 * a * y - 1 = 0 → False) → 
  (a = 0 ∨ a = -1/3) :=
sorry

end NUMINAMATH_GPT_parallel_lines_l1400_140007


namespace NUMINAMATH_GPT_mat_length_is_correct_l1400_140021

noncomputable def mat_length (r : ℝ) (w : ℝ) : ℝ :=
  let θ := 2 * Real.pi / 5
  let side := 2 * r * Real.sin (θ / 2)
  let D := r * Real.cos (Real.pi / 5)
  let x := ((Real.sqrt (r^2 - ((w / 2) ^ 2))) - D + (w / 2))
  x

theorem mat_length_is_correct :
  mat_length 5 1 = 1.4 :=
by
  sorry

end NUMINAMATH_GPT_mat_length_is_correct_l1400_140021


namespace NUMINAMATH_GPT_ellipse_equation_l1400_140085

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1400_140085


namespace NUMINAMATH_GPT_marked_price_correct_l1400_140057

theorem marked_price_correct
    (initial_price : ℝ)
    (initial_discount_rate : ℝ)
    (profit_margin_rate : ℝ)
    (final_discount_rate : ℝ)
    (purchase_price : ℝ)
    (final_selling_price : ℝ)
    (marked_price : ℝ)
    (h_initial_price : initial_price = 30)
    (h_initial_discount_rate : initial_discount_rate = 0.15)
    (h_profit_margin_rate : profit_margin_rate = 0.20)
    (h_final_discount_rate : final_discount_rate = 0.25)
    (h_purchase_price : purchase_price = initial_price * (1 - initial_discount_rate))
    (h_final_selling_price : final_selling_price = purchase_price * (1 + profit_margin_rate))
    (h_marked_price : marked_price * (1 - final_discount_rate) = final_selling_price) : 
    marked_price = 40.80 :=
by
  sorry

end NUMINAMATH_GPT_marked_price_correct_l1400_140057


namespace NUMINAMATH_GPT_speed_of_stream_l1400_140008

theorem speed_of_stream (v_s : ℝ) (D : ℝ) (h1 : D / (78 - v_s) = 2 * (D / (78 + v_s))) : v_s = 26 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1400_140008


namespace NUMINAMATH_GPT_plastic_bag_co2_release_l1400_140087

def total_co2_canvas_bag_lb : ℕ := 600
def total_co2_canvas_bag_oz : ℕ := 9600
def plastic_bags_per_trip : ℕ := 8
def shopping_trips : ℕ := 300

theorem plastic_bag_co2_release :
  total_co2_canvas_bag_oz = 2400 * 4 :=
by
  sorry

end NUMINAMATH_GPT_plastic_bag_co2_release_l1400_140087


namespace NUMINAMATH_GPT_ways_from_A_to_C_l1400_140094

theorem ways_from_A_to_C (ways_A_to_B : ℕ) (ways_B_to_C : ℕ) (hA_to_B : ways_A_to_B = 3) (hB_to_C : ways_B_to_C = 4) : ways_A_to_B * ways_B_to_C = 12 :=
by
  sorry

end NUMINAMATH_GPT_ways_from_A_to_C_l1400_140094


namespace NUMINAMATH_GPT_fish_count_l1400_140075

theorem fish_count (initial_fish : ℝ) (bought_fish : ℝ) (total_fish : ℝ) 
  (h1 : initial_fish = 212.0) 
  (h2 : bought_fish = 280.0) 
  (h3 : total_fish = initial_fish + bought_fish) : 
  total_fish = 492.0 := 
by 
  sorry

end NUMINAMATH_GPT_fish_count_l1400_140075


namespace NUMINAMATH_GPT_fifth_student_gold_stickers_l1400_140072

theorem fifth_student_gold_stickers :
  ∀ s1 s2 s3 s4 s5 s6 : ℕ,
  s1 = 29 →
  s2 = 35 →
  s3 = 41 →
  s4 = 47 →
  s6 = 59 →
  (s2 - s1 = 6) →
  (s3 - s2 = 6) →
  (s4 - s3 = 6) →
  (s6 - s4 = 12) →
  s5 = s4 + (s2 - s1) →
  s5 = 53 := by
  intros s1 s2 s3 s4 s5 s6 hs1 hs2 hs3 hs4 hs6 hd1 hd2 hd3 hd6 heq
  subst_vars
  sorry

end NUMINAMATH_GPT_fifth_student_gold_stickers_l1400_140072


namespace NUMINAMATH_GPT_sum_of_possible_values_l1400_140037

theorem sum_of_possible_values (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l1400_140037


namespace NUMINAMATH_GPT_solution_set_empty_range_a_l1400_140071

theorem solution_set_empty_range_a (a : ℝ) :
  (∀ x : ℝ, ¬((a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0)) ↔ -3 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_empty_range_a_l1400_140071


namespace NUMINAMATH_GPT_jake_not_drop_coffee_l1400_140050

theorem jake_not_drop_coffee :
  let p_trip := 0.40
  let p_drop_trip := 0.25
  let p_step := 0.30
  let p_drop_step := 0.20
  let p_no_drop_trip := 1 - (p_trip * p_drop_trip)
  let p_no_drop_step := 1 - (p_step * p_drop_step)
  (p_no_drop_trip * p_no_drop_step) = 0.846 :=
by
  sorry

end NUMINAMATH_GPT_jake_not_drop_coffee_l1400_140050


namespace NUMINAMATH_GPT_probability_all_quitters_same_tribe_l1400_140069

theorem probability_all_quitters_same_tribe :
  ∀ (people : Finset ℕ) (tribe1 tribe2 : Finset ℕ) (choose : ℕ → ℕ → ℕ) (prob : ℚ),
  people.card = 20 →
  tribe1.card = 10 →
  tribe2.card = 10 →
  tribe1 ∪ tribe2 = people →
  tribe1 ∩ tribe2 = ∅ →
  choose 20 3 = 1140 →
  choose 10 3 = 120 →
  prob = (2 * choose 10 3) / choose 20 3 →
  prob = 20 / 95 :=
by
  intro people tribe1 tribe2 choose prob
  intros hp20 ht1 ht2 hu hi hchoose20 hchoose10 hprob
  sorry

end NUMINAMATH_GPT_probability_all_quitters_same_tribe_l1400_140069


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1400_140080

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}
def C_UA : Set ℝ := U \ A

theorem complement_of_A_in_U :
  C_UA = {x | 0 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1400_140080


namespace NUMINAMATH_GPT_find_k_l1400_140053

-- Define the problem statement
theorem find_k (d : ℝ) (x : ℝ)
  (h_ratio : 3 * x / (5 * x) = 3 / 5)
  (h_diag : (10 * d)^2 = (3 * x)^2 + (5 * x)^2) :
  ∃ k : ℝ, (3 * x) * (5 * x) = k * d^2 ∧ k = 750 / 17 := by
  sorry

end NUMINAMATH_GPT_find_k_l1400_140053


namespace NUMINAMATH_GPT_andy_time_correct_l1400_140031

-- Define the conditions
def time_dawn_wash_dishes : ℕ := 20
def time_andy_put_laundry : ℕ := 2 * time_dawn_wash_dishes + 6

-- The theorem to prove
theorem andy_time_correct : time_andy_put_laundry = 46 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_andy_time_correct_l1400_140031


namespace NUMINAMATH_GPT_math_problem_proof_l1400_140025

def eight_to_zero : ℝ := 1
def log_base_10_of_100 : ℝ := 2

theorem math_problem_proof : eight_to_zero - log_base_10_of_100 = -1 :=
by sorry

end NUMINAMATH_GPT_math_problem_proof_l1400_140025


namespace NUMINAMATH_GPT_problem_1_problem_2_l1400_140066

-- Definitions and conditions for the problems
def A : Set ℝ := { x | abs (x - 2) < 3 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2 * x - m < 0 }

-- Problem (I)
theorem problem_1 : (A ∩ (Set.univ \ B 3)) = { x | 3 ≤ x ∧ x < 5 } :=
sorry

-- Problem (II)
theorem problem_2 (m : ℝ) : (A ∩ B m = { x | -1 < x ∧ x < 4 }) → m = 8 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1400_140066


namespace NUMINAMATH_GPT_abs_lt_one_suff_but_not_necc_l1400_140012

theorem abs_lt_one_suff_but_not_necc (x : ℝ) : (|x| < 1 → x^2 + x - 2 < 0) ∧ ¬(x^2 + x - 2 < 0 → |x| < 1) :=
by
  sorry

end NUMINAMATH_GPT_abs_lt_one_suff_but_not_necc_l1400_140012


namespace NUMINAMATH_GPT_concyclic_iff_ratio_real_l1400_140017

noncomputable def concyclic_condition (z1 z2 z3 z4 : ℂ) : Prop :=
  (∃ c : ℂ, c ≠ 0 ∧ ∀ (w : ℂ), (w - z1) * (w - z3) / ((w - z2) * (w - z4)) = c)

noncomputable def ratio_real (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ r : ℝ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3)) = r

theorem concyclic_iff_ratio_real (z1 z2 z3 z4 : ℂ) :
  concyclic_condition z1 z2 z3 z4 ↔ ratio_real z1 z2 z3 z4 :=
sorry

end NUMINAMATH_GPT_concyclic_iff_ratio_real_l1400_140017


namespace NUMINAMATH_GPT_charlie_rope_first_post_l1400_140056

theorem charlie_rope_first_post (X : ℕ) (h : X + 20 + 14 + 12 = 70) : X = 24 :=
sorry

end NUMINAMATH_GPT_charlie_rope_first_post_l1400_140056


namespace NUMINAMATH_GPT_tickets_distribution_l1400_140062

theorem tickets_distribution (people tickets : ℕ) (h_people : people = 9) (h_tickets : tickets = 24)
  (h_each_gets_at_least_one : ∀ (i : ℕ), i < people → (1 : ℕ) ≤ 1) :
  ∃ (count : ℕ), count ≥ 4 ∧ ∃ (f : ℕ → ℕ), (∀ i, i < people → 1 ≤ f i ∧ f i ≤ tickets) ∧ (∀ i < people, ∃ j < people, f i = f j) :=
  sorry

end NUMINAMATH_GPT_tickets_distribution_l1400_140062


namespace NUMINAMATH_GPT_simplify_expression_l1400_140043

theorem simplify_expression (x y m : ℤ) 
  (h1 : (x-5)^2 = -|m-1|)
  (h2 : y + 1 = 5) :
  (2 * x^2 - 3 * x * y - 4 * y^2) - m * (3 * x^2 - x * y + 9 * y^2) = -273 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1400_140043


namespace NUMINAMATH_GPT_complex_square_identity_l1400_140055

theorem complex_square_identity (i : ℂ) (h_i_squared : i^2 = -1) :
  i * (1 + i)^2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_complex_square_identity_l1400_140055


namespace NUMINAMATH_GPT_correct_answer_B_l1400_140005

def point_slope_form (k : ℝ) (x y : ℝ) : Prop := y + 1 = k * (x - 2)

def proposition_2 (k : ℝ) (x y : ℝ) : Prop :=
  ∃ k : ℝ, @point_slope_form k x y

def proposition_3 (k : ℝ) : Prop := point_slope_form k 2 (-1)

def proposition_4 (k : ℝ) : Prop := k ≠ 0

theorem correct_answer_B : 
  (∃ k : ℝ, @point_slope_form k 2 (-1)) ∧ 
  (∀ k : ℝ, @point_slope_form k 2 (-1)) ∧
  (∀ k : ℝ, k ≠ 0) → true := 
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_answer_B_l1400_140005


namespace NUMINAMATH_GPT_number_of_men_in_engineering_department_l1400_140054

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end NUMINAMATH_GPT_number_of_men_in_engineering_department_l1400_140054


namespace NUMINAMATH_GPT_contradiction_proof_example_l1400_140018

theorem contradiction_proof_example (a b : ℝ) (h: a ≤ b → False) : a > b :=
by sorry

end NUMINAMATH_GPT_contradiction_proof_example_l1400_140018


namespace NUMINAMATH_GPT_min_value_le_one_l1400_140006

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x
noncomputable def g (a : ℝ) : ℝ := a - a * Real.log a

theorem min_value_le_one (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, f x a ≥ g a) ∧ g a ≤ 1 := sorry

end NUMINAMATH_GPT_min_value_le_one_l1400_140006


namespace NUMINAMATH_GPT_multiplication_of_monomials_l1400_140020

-- Define the constants and assumptions
def a : ℝ := -2
def b : ℝ := 4
def e1 : ℤ := 4
def e2 : ℤ := 5
def result : ℝ := -8
def result_exp : ℤ := 9

-- State the theorem to be proven
theorem multiplication_of_monomials :
  (a * 10^e1) * (b * 10^e2) = result * 10^result_exp := 
by
  sorry

end NUMINAMATH_GPT_multiplication_of_monomials_l1400_140020


namespace NUMINAMATH_GPT_third_quadrant_angles_l1400_140002

theorem third_quadrant_angles :
  {α : ℝ | ∃ k : ℤ, π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π} =
  {α | π < α ∧ α < 3 * π / 2} :=
sorry

end NUMINAMATH_GPT_third_quadrant_angles_l1400_140002


namespace NUMINAMATH_GPT_graph_f_intersects_x_eq_1_at_most_once_l1400_140086

-- Define a function f from ℝ to ℝ
def f : ℝ → ℝ := sorry  -- Placeholder for the actual function

-- Define the domain of the function f (it's a generic function on ℝ for simplicity)
axiom f_unique : ∀ x y : ℝ, f x = f y → x = y  -- If f(x) = f(y), then x must equal y

-- Prove that the graph of y = f(x) intersects the line x = 1 at most once
theorem graph_f_intersects_x_eq_1_at_most_once : ∃ y : ℝ, (f 1 = y) ∨ (¬∃ y : ℝ, f 1 = y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_graph_f_intersects_x_eq_1_at_most_once_l1400_140086


namespace NUMINAMATH_GPT_kindergarten_children_count_l1400_140049

theorem kindergarten_children_count (D B C : ℕ) (hD : D = 18) (hB : B = 6) (hC : C + B = 12) : D + C + B = 30 :=
by
  sorry

end NUMINAMATH_GPT_kindergarten_children_count_l1400_140049


namespace NUMINAMATH_GPT_project_completion_time_l1400_140067

theorem project_completion_time (initial_workers : ℕ) (initial_days : ℕ) (extra_workers : ℕ) (extra_days : ℕ) : 
  initial_workers = 10 →
  initial_days = 15 →
  extra_workers = 5 →
  extra_days = 5 →
  total_days = 6 := by
  sorry

end NUMINAMATH_GPT_project_completion_time_l1400_140067


namespace NUMINAMATH_GPT_fabric_cut_l1400_140064

/-- Given a piece of fabric that is 2/3 meter long,
we can cut a piece measuring 1/2 meter
by folding the original piece into four equal parts and removing one part. -/
theorem fabric_cut :
  ∃ (f : ℚ), f = (2/3 : ℚ) → ∃ (half : ℚ), half = (1/2 : ℚ) ∧ half = f * (3/4 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_fabric_cut_l1400_140064


namespace NUMINAMATH_GPT_total_pikes_l1400_140044

theorem total_pikes (x : ℝ) (h : x = 4 + (1/2) * x) : x = 8 :=
sorry

end NUMINAMATH_GPT_total_pikes_l1400_140044


namespace NUMINAMATH_GPT_max_marked_cells_no_shared_vertices_l1400_140038

theorem max_marked_cells_no_shared_vertices (N : ℕ) (cube_side : ℕ) (total_cells : ℕ) (total_vertices : ℕ) :
  cube_side = 3 →
  total_cells = cube_side ^ 3 →
  total_vertices = 8 + 12 * 2 + 6 * 4 →
  ∀ (max_cells : ℕ), (4 * max_cells ≤ total_vertices) → (max_cells ≤ 14) :=
by
  sorry

end NUMINAMATH_GPT_max_marked_cells_no_shared_vertices_l1400_140038


namespace NUMINAMATH_GPT_unique_solution_otimes_l1400_140027

def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution_otimes : 
  (∃! y : ℝ, otimes 2 y = 20) := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_otimes_l1400_140027


namespace NUMINAMATH_GPT_equal_poly_terms_l1400_140042

theorem equal_poly_terms (p q : ℝ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : 
  (7 * p^6 * q = 21 * p^5 * q^2) -> p = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_equal_poly_terms_l1400_140042


namespace NUMINAMATH_GPT_coefficient_of_ab_is_correct_l1400_140083

noncomputable def a : ℝ := 15 / 7
noncomputable def b : ℝ := 15 / 2
noncomputable def ab : ℝ := 674.9999999999999
noncomputable def coeff_ab := ab / (a * b)

theorem coefficient_of_ab_is_correct :
  coeff_ab = 674.9999999999999 / ((15 * 15) / (7 * 2)) := sorry

end NUMINAMATH_GPT_coefficient_of_ab_is_correct_l1400_140083


namespace NUMINAMATH_GPT_quotient_of_division_l1400_140084

theorem quotient_of_division:
  ∀ (n d r q : ℕ), n = 165 → d = 18 → r = 3 → q = (n - r) / d → q = 9 :=
by sorry

end NUMINAMATH_GPT_quotient_of_division_l1400_140084


namespace NUMINAMATH_GPT_value_of_f5_f_neg5_l1400_140039

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 - a * x^3 + b * x + 2

-- Given conditions
variable (a b : ℝ)
axiom h1 : f (-5) a b = 3

-- The proposition to prove
theorem value_of_f5_f_neg5 : f 5 a b + f (-5) a b = 4 :=
by
  -- Include the result of the proof
  sorry

end NUMINAMATH_GPT_value_of_f5_f_neg5_l1400_140039


namespace NUMINAMATH_GPT_roots_squared_sum_l1400_140093

theorem roots_squared_sum (p q r : ℂ) (h : ∀ x : ℂ, 3 * x ^ 3 - 3 * x ^ 2 + 6 * x - 9 = 0 → x = p ∨ x = q ∨ x = r) :
  p^2 + q^2 + r^2 = -3 :=
by
  sorry

end NUMINAMATH_GPT_roots_squared_sum_l1400_140093


namespace NUMINAMATH_GPT_distinguishable_squares_count_is_70_l1400_140036

def count_distinguishable_squares : ℕ :=
  let total_colorings : ℕ := 2^9
  let rotation_90_270_fixed : ℕ := 2^3
  let rotation_180_fixed : ℕ := 2^5
  let average_fixed_colorings : ℕ :=
    (total_colorings + rotation_90_270_fixed + rotation_90_270_fixed + rotation_180_fixed) / 4
  let distinguishable_squares : ℕ := average_fixed_colorings / 2
  distinguishable_squares

theorem distinguishable_squares_count_is_70 :
  count_distinguishable_squares = 70 := by
  sorry

end NUMINAMATH_GPT_distinguishable_squares_count_is_70_l1400_140036


namespace NUMINAMATH_GPT_div_mul_fraction_eq_neg_81_over_4_l1400_140035

theorem div_mul_fraction_eq_neg_81_over_4 : 
  -4 / (4 / 9) * (9 / 4) = - (81 / 4) := 
by
  sorry

end NUMINAMATH_GPT_div_mul_fraction_eq_neg_81_over_4_l1400_140035


namespace NUMINAMATH_GPT_original_price_of_candy_box_is_8_l1400_140059

-- Define the given conditions
def candy_box_price_after_increase : ℝ := 10
def candy_box_increase_rate : ℝ := 1.25

-- Define the original price of the candy box
noncomputable def original_candy_box_price : ℝ := candy_box_price_after_increase / candy_box_increase_rate

-- The theorem to prove
theorem original_price_of_candy_box_is_8 :
  original_candy_box_price = 8 := by
  sorry

end NUMINAMATH_GPT_original_price_of_candy_box_is_8_l1400_140059


namespace NUMINAMATH_GPT_germs_per_dish_calc_l1400_140052

theorem germs_per_dish_calc :
    let total_germs := 0.036 * 10^5
    let total_dishes := 36000 * 10^(-3)
    (total_germs / total_dishes) = 100 := by
    sorry

end NUMINAMATH_GPT_germs_per_dish_calc_l1400_140052


namespace NUMINAMATH_GPT_MishaTotalMoney_l1400_140001

-- Define Misha's initial amount of money
def initialMoney : ℕ := 34

-- Define the amount of money Misha earns
def earnedMoney : ℕ := 13

-- Define the total amount of money Misha will have
def totalMoney : ℕ := initialMoney + earnedMoney

-- Statement to prove
theorem MishaTotalMoney : totalMoney = 47 := by
  sorry

end NUMINAMATH_GPT_MishaTotalMoney_l1400_140001


namespace NUMINAMATH_GPT_line_equation_l1400_140045

theorem line_equation (x y : ℝ) (c : ℝ)
  (h1 : 2 * x - y + 3 = 0)
  (h2 : 4 * x + 3 * y + 1 = 0)
  (h3 : 3 * x + 2 * y + c = 0) :
  c = 1 := sorry

end NUMINAMATH_GPT_line_equation_l1400_140045


namespace NUMINAMATH_GPT_product_prs_l1400_140040

open Real

theorem product_prs (p r s : ℕ) 
  (h1 : 4 ^ p + 64 = 272) 
  (h2 : 3 ^ r = 81)
  (h3 : 6 ^ s = 478) : 
  p * r * s = 64 :=
by
  sorry

end NUMINAMATH_GPT_product_prs_l1400_140040


namespace NUMINAMATH_GPT_integer_modulo_solution_l1400_140024

theorem integer_modulo_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] ∧ n = 15 :=
sorry

end NUMINAMATH_GPT_integer_modulo_solution_l1400_140024


namespace NUMINAMATH_GPT_excluded_twins_lineup_l1400_140051

/-- 
  Prove that the number of ways to choose 5 starters from 15 players,
  such that both Alice and Bob (twins) are not included together in the lineup, is 2717.
-/
theorem excluded_twins_lineup (n : ℕ) (k : ℕ) (t : ℕ) (u : ℕ) (h_n : n = 15) (h_k : k = 5) (h_t : t = 2) (h_u : u = 3) :
  ((n.choose k) - ((n - t).choose u)) = 2717 :=
by {
  sorry
}

end NUMINAMATH_GPT_excluded_twins_lineup_l1400_140051


namespace NUMINAMATH_GPT_find_m_of_hyperbola_l1400_140091

noncomputable def eccen_of_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2) / (a^2))

theorem find_m_of_hyperbola :
  ∃ (m : ℝ), (m > 0) ∧ (eccen_of_hyperbola 2 m = Real.sqrt 3) ∧ (m = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_hyperbola_l1400_140091


namespace NUMINAMATH_GPT_fraction_of_butterflies_flew_away_l1400_140058

theorem fraction_of_butterflies_flew_away (original_butterflies : ℕ) (left_butterflies : ℕ) (h1 : original_butterflies = 9) (h2 : left_butterflies = 6) : (original_butterflies - left_butterflies) / original_butterflies = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_butterflies_flew_away_l1400_140058


namespace NUMINAMATH_GPT_sister_age_is_one_l1400_140089

variable (B S : ℕ)

theorem sister_age_is_one (h : B = B * S) : S = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_sister_age_is_one_l1400_140089


namespace NUMINAMATH_GPT_simplify_fractions_l1400_140079

theorem simplify_fractions:
  (3 / 462 : ℚ) + (28 / 42 : ℚ) = 311 / 462 := sorry

end NUMINAMATH_GPT_simplify_fractions_l1400_140079


namespace NUMINAMATH_GPT_curve_passes_through_fixed_point_l1400_140016

theorem curve_passes_through_fixed_point (m n : ℝ) :
  (2:ℝ)^2 + (-2:ℝ)^2 - 2 * m * (2:ℝ) - 2 * n * (-2:ℝ) + 4 * (m - n - 2) = 0 :=
by sorry

end NUMINAMATH_GPT_curve_passes_through_fixed_point_l1400_140016


namespace NUMINAMATH_GPT_expected_heads_l1400_140004

def coin_flips : Nat := 64

def prob_heads (tosses : ℕ) : ℚ :=
  1 / 2^(tosses + 1)

def total_prob_heads : ℚ :=
  prob_heads 0 + prob_heads 1 + prob_heads 2 + prob_heads 3

theorem expected_heads : (coin_flips : ℚ) * total_prob_heads = 60 := by
  sorry

end NUMINAMATH_GPT_expected_heads_l1400_140004


namespace NUMINAMATH_GPT_sue_nuts_count_l1400_140010

theorem sue_nuts_count (B H S : ℕ) 
  (h1 : B = 6 * H) 
  (h2 : H = 2 * S) 
  (h3 : B + H = 672) : S = 48 := 
by
  sorry

end NUMINAMATH_GPT_sue_nuts_count_l1400_140010


namespace NUMINAMATH_GPT_find_tan_of_cos_in_4th_quadrant_l1400_140032

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end NUMINAMATH_GPT_find_tan_of_cos_in_4th_quadrant_l1400_140032


namespace NUMINAMATH_GPT_power_identity_l1400_140011

theorem power_identity {a n m k : ℝ} (h1: a^n = 2) (h2: a^m = 3) (h3: a^k = 4) :
  a^(2 * n + m - 2 * k) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_power_identity_l1400_140011


namespace NUMINAMATH_GPT_number_of_solutions_l1400_140099

-- Define the equation and the constraints
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + z = 800

def positive_integer (n : ℕ) : Prop := n > 0

-- The main theorem statement
theorem number_of_solutions : ∃ s, s = 127 ∧ ∀ (x y z : ℕ), positive_integer x → positive_integer y → positive_integer z → equation x y z → s = 127 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1400_140099


namespace NUMINAMATH_GPT_total_noodles_and_pirates_l1400_140070

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end NUMINAMATH_GPT_total_noodles_and_pirates_l1400_140070


namespace NUMINAMATH_GPT_toms_crab_buckets_l1400_140026

def crabs_per_bucket := 12
def price_per_crab := 5
def weekly_earnings := 3360

theorem toms_crab_buckets : (weekly_earnings / (crabs_per_bucket * price_per_crab)) = 56 := by
  sorry

end NUMINAMATH_GPT_toms_crab_buckets_l1400_140026


namespace NUMINAMATH_GPT_largest_consecutive_multiple_of_3_l1400_140097

theorem largest_consecutive_multiple_of_3 (n : ℕ) 
  (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 72) : 3 * (n + 2) = 27 :=
by 
  sorry

end NUMINAMATH_GPT_largest_consecutive_multiple_of_3_l1400_140097


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1400_140088

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) 
  (h_inc : ∀ n, b (n + 1) - b n = d)
  (h_pos : d > 0)
  (h_prod : b 5 * b 6 = 21) 
  : b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1400_140088


namespace NUMINAMATH_GPT_gina_tom_goals_l1400_140028

theorem gina_tom_goals :
  let g_day1 := 2
  let t_day1 := g_day1 + 3
  let t_day2 := 6
  let g_day2 := t_day2 - 2
  let g_total := g_day1 + g_day2
  let t_total := t_day1 + t_day2
  g_total + t_total = 17 := by
  sorry

end NUMINAMATH_GPT_gina_tom_goals_l1400_140028


namespace NUMINAMATH_GPT_henrikh_commute_distance_l1400_140000

theorem henrikh_commute_distance (x : ℕ)
    (h1 : ∀ y : ℕ, y = x → y = x)
    (h2 : 1 * x = x)
    (h3 : 20 * x = (x : ℕ))
    (h4 : x = (x / 3) + 8) :
    x = 12 := sorry

end NUMINAMATH_GPT_henrikh_commute_distance_l1400_140000


namespace NUMINAMATH_GPT_problem1_problem2_l1400_140047

theorem problem1 : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := 
by 
  sorry
  
theorem problem2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1) ^ 2 = 14 + 4 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1400_140047


namespace NUMINAMATH_GPT_percent_x_of_w_l1400_140090

theorem percent_x_of_w (x y z w : ℝ)
  (h1 : x = 1.2 * y)
  (h2 : y = 0.7 * z)
  (h3 : w = 1.5 * z) : (x / w) * 100 = 56 :=
by
  sorry

end NUMINAMATH_GPT_percent_x_of_w_l1400_140090


namespace NUMINAMATH_GPT_lily_pad_growth_rate_l1400_140046

theorem lily_pad_growth_rate 
  (day_37_covers_full : ℕ → ℝ)
  (day_36_covers_half : ℕ → ℝ)
  (exponential_growth : day_37_covers_full = 2 * day_36_covers_half) :
  (2 - 1) / 1 * 100 = 100 :=
by sorry

end NUMINAMATH_GPT_lily_pad_growth_rate_l1400_140046


namespace NUMINAMATH_GPT_cost_for_flour_for_two_cakes_l1400_140081

theorem cost_for_flour_for_two_cakes 
    (packages_per_cake : ℕ)
    (cost_per_package : ℕ)
    (cakes : ℕ) 
    (total_cost : ℕ)
    (H1 : packages_per_cake = 2)
    (H2 : cost_per_package = 3)
    (H3 : cakes = 2)
    (H4 : total_cost = 12) :
    total_cost = cakes * packages_per_cake * cost_per_package := 
by 
    rw [H1, H2, H3]
    sorry

end NUMINAMATH_GPT_cost_for_flour_for_two_cakes_l1400_140081


namespace NUMINAMATH_GPT_length_of_goods_train_l1400_140041

-- Define the given data
def speed_kmph := 72
def platform_length_m := 250
def crossing_time_s := 36

-- Convert speed from kmph to m/s
def speed_mps := speed_kmph * (5 / 18)

-- Define the total distance covered while crossing the platform
def distance_covered_m := speed_mps * crossing_time_s

-- Define the length of the train
def train_length_m := distance_covered_m - platform_length_m

-- The theorem to be proven
theorem length_of_goods_train : train_length_m = 470 := by
  sorry

end NUMINAMATH_GPT_length_of_goods_train_l1400_140041


namespace NUMINAMATH_GPT_total_sum_is_750_l1400_140048

-- Define the individual numbers
def joyce_number : ℕ := 30

def xavier_number (joyce : ℕ) : ℕ :=
  4 * joyce

def coraline_number (xavier : ℕ) : ℕ :=
  xavier + 50

def jayden_number (coraline : ℕ) : ℕ :=
  coraline - 40

def mickey_number (jayden : ℕ) : ℕ :=
  jayden + 20

def yvonne_number (xavier joyce : ℕ) : ℕ :=
  xavier + joyce

-- Prove the total sum is 750
theorem total_sum_is_750 :
  joyce_number + xavier_number joyce_number + coraline_number (xavier_number joyce_number) +
  jayden_number (coraline_number (xavier_number joyce_number)) +
  mickey_number (jayden_number (coraline_number (xavier_number joyce_number))) +
  yvonne_number (xavier_number joyce_number) joyce_number = 750 :=
by {
  -- Proof omitted for brevity
  sorry
}

end NUMINAMATH_GPT_total_sum_is_750_l1400_140048


namespace NUMINAMATH_GPT_age_ratio_l1400_140023

variable (p q : ℕ)

-- Conditions
def condition1 := p - 6 = (q - 6) / 2
def condition2 := p + q = 21

-- Theorem stating the desired ratio
theorem age_ratio (h1 : condition1 p q) (h2 : condition2 p q) : p / Nat.gcd p q = 3 ∧ q / Nat.gcd p q = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1400_140023


namespace NUMINAMATH_GPT_fraction_is_two_thirds_l1400_140082

noncomputable def fraction_of_price_of_ballet_slippers (f : ℚ) : Prop :=
  let price_high_heels := 60
  let num_ballet_slippers := 5
  let total_cost := 260
  price_high_heels + num_ballet_slippers * f * price_high_heels = total_cost

theorem fraction_is_two_thirds : fraction_of_price_of_ballet_slippers (2 / 3) := by
  sorry

end NUMINAMATH_GPT_fraction_is_two_thirds_l1400_140082


namespace NUMINAMATH_GPT_cupcakes_per_package_calculation_l1400_140098

noncomputable def sarah_total_cupcakes := 38
noncomputable def cupcakes_eaten_by_todd := 14
noncomputable def number_of_packages := 3
noncomputable def remaining_cupcakes := sarah_total_cupcakes - cupcakes_eaten_by_todd
noncomputable def cupcakes_per_package := remaining_cupcakes / number_of_packages

theorem cupcakes_per_package_calculation : cupcakes_per_package = 8 := by
  sorry

end NUMINAMATH_GPT_cupcakes_per_package_calculation_l1400_140098


namespace NUMINAMATH_GPT_seating_arrangement_l1400_140013

theorem seating_arrangement (x y z : ℕ) (h1 : z = x + y) (h2 : x*10 + y*9 = 67) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1400_140013


namespace NUMINAMATH_GPT_parabola_vertex_intercept_l1400_140022

variable (a b c p : ℝ)

theorem parabola_vertex_intercept (h_vertex : ∀ x : ℝ, (a * (x - p) ^ 2 + p) = a * x^2 + b * x + c)
                                  (h_intercept : a * p^2 + p = 2 * p)
                                  (hp : p ≠ 0) : b = -2 :=
sorry

end NUMINAMATH_GPT_parabola_vertex_intercept_l1400_140022


namespace NUMINAMATH_GPT_arithmetic_sequence_z_l1400_140061

-- Define the arithmetic sequence and value of z
theorem arithmetic_sequence_z (z : ℤ) (arith_seq : 9 + 27 = 2 * z) : z = 18 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_z_l1400_140061


namespace NUMINAMATH_GPT_even_function_m_eq_neg_one_l1400_140015

theorem even_function_m_eq_neg_one (m : ℝ) :
  (∀ x : ℝ, (m - 1)*x^2 - (m^2 - 1)*x + (m + 2) = (m - 1)*(-x)^2 - (m^2 - 1)*(-x) + (m + 2)) →
  m = -1 :=
  sorry

end NUMINAMATH_GPT_even_function_m_eq_neg_one_l1400_140015


namespace NUMINAMATH_GPT_find_cd_minus_dd_base_d_l1400_140030

namespace MathProof

variables (d C D : ℤ)

def digit_sum (C D : ℤ) (d : ℤ) : ℤ := d * C + D
def digit_sum_same (C : ℤ) (d : ℤ) : ℤ := d * C + C

theorem find_cd_minus_dd_base_d (h_d : d > 8) (h_eq : digit_sum C D d + digit_sum_same C d = d^2 + 8 * d + 4) :
  C - D = 1 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_find_cd_minus_dd_base_d_l1400_140030


namespace NUMINAMATH_GPT_true_statement_given_conditions_l1400_140019

theorem true_statement_given_conditions (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a < b) :
  |1| / |a| > |1| / |b| := 
by
  sorry

end NUMINAMATH_GPT_true_statement_given_conditions_l1400_140019


namespace NUMINAMATH_GPT_solve_for_y_l1400_140034

-- Define the conditions and the goal to prove in Lean 4
theorem solve_for_y
  (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1400_140034


namespace NUMINAMATH_GPT_fg_neg_two_l1400_140065

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x + 3

theorem fg_neg_two : f (g (-2)) = 2 := by
  sorry

end NUMINAMATH_GPT_fg_neg_two_l1400_140065


namespace NUMINAMATH_GPT_smallest_positive_integer_cube_ends_368_l1400_140092

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_cube_ends_368_l1400_140092


namespace NUMINAMATH_GPT_total_games_won_l1400_140096

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end NUMINAMATH_GPT_total_games_won_l1400_140096


namespace NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l1400_140009

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = (Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l1400_140009


namespace NUMINAMATH_GPT_math_club_members_count_l1400_140073

theorem math_club_members_count 
    (n_books : ℕ) 
    (n_borrow_each_member : ℕ) 
    (n_borrow_each_book : ℕ) 
    (total_borrow_count_books : n_books * n_borrow_each_book = 36) 
    (total_borrow_count_members : 2 * x = 36) 
    : x = 18 := 
by
  sorry

end NUMINAMATH_GPT_math_club_members_count_l1400_140073


namespace NUMINAMATH_GPT_total_dots_correct_l1400_140077

/-- Define the initial conditions -/
def monday_ladybugs : ℕ := 8
def monday_dots_per_ladybug : ℕ := 6
def tuesday_ladybugs : ℕ := 5
def wednesday_ladybugs : ℕ := 4

/-- Define the derived conditions -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- Calculate the total number of dots -/
def monday_total_dots : ℕ := monday_ladybugs * monday_dots_per_ladybug
def tuesday_total_dots : ℕ := tuesday_ladybugs * tuesday_dots_per_ladybug
def wednesday_total_dots : ℕ := wednesday_ladybugs * wednesday_dots_per_ladybug
def total_dots : ℕ := monday_total_dots + tuesday_total_dots + wednesday_total_dots

/-- Prove the total dots equal to 89 -/
theorem total_dots_correct : total_dots = 89 := by
  sorry

end NUMINAMATH_GPT_total_dots_correct_l1400_140077


namespace NUMINAMATH_GPT_average_weight_l1400_140068

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end NUMINAMATH_GPT_average_weight_l1400_140068


namespace NUMINAMATH_GPT_chessboard_edge_count_l1400_140095

theorem chessboard_edge_count (n : ℕ) 
  (border_white : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ w : ℕ, w ≥ n)) 
  (border_black : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ b : ℕ, b ≥ n)) :
  ∃ e : ℕ, e ≥ n :=
sorry

end NUMINAMATH_GPT_chessboard_edge_count_l1400_140095


namespace NUMINAMATH_GPT_compute_expr_l1400_140078

-- Definitions
def a := 150 / 5
def b := 40 / 8
def c := 16 / 32
def d := 3

def expr := 20 * (a - b + c + d)

-- Theorem
theorem compute_expr : expr = 570 :=
by
  sorry

end NUMINAMATH_GPT_compute_expr_l1400_140078


namespace NUMINAMATH_GPT_time_on_sideline_l1400_140076

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end NUMINAMATH_GPT_time_on_sideline_l1400_140076


namespace NUMINAMATH_GPT_find_functions_l1400_140033

theorem find_functions (M N : ℝ × ℝ)
  (hM : M.fst = -4) (hM_quad2 : 0 < M.snd)
  (hN : N = (-6, 0))
  (h_area : 1 / 2 * 6 * M.snd = 15) :
  (∃ k, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * k = -5 / 4 * x)) ∧ 
  (∃ a b, ∀ x, (M = (-4, 5) → N = (-6, 0) → x * a + b = 5 / 2 * x + 15)) := 
sorry

end NUMINAMATH_GPT_find_functions_l1400_140033


namespace NUMINAMATH_GPT_inequality_condition_l1400_140003

-- Define the inequality (x - 2) * (x + 2) > 0
def inequality_holds (x : ℝ) : Prop := (x - 2) * (x + 2) > 0

-- The sufficient and necessary condition for the inequality to hold is x > 2 or x < -2
theorem inequality_condition (x : ℝ) : inequality_holds x ↔ (x > 2 ∨ x < -2) :=
  sorry

end NUMINAMATH_GPT_inequality_condition_l1400_140003


namespace NUMINAMATH_GPT_remaining_shoes_to_sell_l1400_140029

def shoes_goal : Nat := 80
def shoes_sold_last_week : Nat := 27
def shoes_sold_this_week : Nat := 12

theorem remaining_shoes_to_sell : shoes_goal - (shoes_sold_last_week + shoes_sold_this_week) = 41 :=
by
  sorry

end NUMINAMATH_GPT_remaining_shoes_to_sell_l1400_140029
