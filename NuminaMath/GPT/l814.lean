import Mathlib

namespace NUMINAMATH_GPT_z_investment_correct_l814_81470

noncomputable def z_investment 
    (x_investment : ℕ) 
    (y_investment : ℕ) 
    (z_profit : ℕ) 
    (total_profit : ℕ)
    (profit_z : ℕ) : ℕ := 
  let x_time := 12
  let y_time := 12
  let z_time := 8
  let x_share := x_investment * x_time
  let y_share := y_investment * y_time
  let profit_ratio := total_profit - profit_z
  (x_share + y_share) * z_time / profit_ratio

theorem z_investment_correct : 
  z_investment 36000 42000 4032 13860 4032 = 52000 :=
by sorry

end NUMINAMATH_GPT_z_investment_correct_l814_81470


namespace NUMINAMATH_GPT_complex_number_imaginary_axis_l814_81438

theorem complex_number_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) → (a = 0 ∨ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_complex_number_imaginary_axis_l814_81438


namespace NUMINAMATH_GPT_remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l814_81450

theorem remainder_of_x_pow_105_div_x_sq_sub_4x_add_3 :
  ∀ (x : ℤ), (x^105) % (x^2 - 4*x + 3) = (3^105 * (x-1) - (x-2)) / 2 :=
by sorry

end NUMINAMATH_GPT_remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l814_81450


namespace NUMINAMATH_GPT_question_implies_answer_l814_81420

theorem question_implies_answer (x y : ℝ) (h : y^2 - x^2 < x) :
  (x ≥ 0 ∨ x ≤ -1) ∧ (-Real.sqrt (x^2 + x) < y ∧ y < Real.sqrt (x^2 + x)) :=
sorry

end NUMINAMATH_GPT_question_implies_answer_l814_81420


namespace NUMINAMATH_GPT_find_a_b_and_tangent_line_l814_81441

noncomputable def f (a b x : ℝ) := x^3 + 2 * a * x^2 + b * x + a
noncomputable def g (x : ℝ) := x^2 - 3 * x + 2
noncomputable def f' (a b x : ℝ) := 3 * x^2 + 4 * a * x + b
noncomputable def g' (x : ℝ) := 2 * x - 3

theorem find_a_b_and_tangent_line (a b : ℝ) :
  f a b 2 = 0 ∧ g 2 = 0 ∧ f' a b 2 = 1 ∧ g' 2 = 1 → (a = -2 ∧ b = 5 ∧ ∀ x y : ℝ, y = x - 2 ↔ x - y - 2 = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_b_and_tangent_line_l814_81441


namespace NUMINAMATH_GPT_find_k_l814_81497

theorem find_k
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : ∃ (x y : ℝ), (x - k * y - 5 = 0 ∧ x^2 + y^2 = 10 ∧ (A = (x, y) ∨ B = (x, y))))
  (h2 : (A.fst^2 + A.snd^2 = 10) ∧ (B.fst^2 + B.snd^2 = 10))
  (h3 : (A.fst - k * A.snd - 5 = 0) ∧ (B.fst - k * B.snd - 5 = 0))
  (h4 : A.fst * B.fst + A.snd * B.snd = 0) :
  k = 2 ∨ k = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l814_81497


namespace NUMINAMATH_GPT_total_cookies_prepared_l814_81421

-- Definition of conditions
def cookies_per_guest : ℕ := 19
def number_of_guests : ℕ := 2

-- Theorem statement
theorem total_cookies_prepared : (cookies_per_guest * number_of_guests) = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_prepared_l814_81421


namespace NUMINAMATH_GPT_probability_divisible_by_3_l814_81468

-- Define the set of numbers
def S : Set ℕ := {2, 3, 5, 6}

-- Define the pairs of numbers whose product is divisible by 3
def valid_pairs : Set (ℕ × ℕ) := {(2, 3), (2, 6), (3, 5), (3, 6), (5, 6)}

-- Define the total number of pairs
def total_pairs := 6

-- Define the number of valid pairs
def valid_pairs_count := 5

-- Prove that the probability of choosing two numbers whose product is divisible by 3 is 5/6
theorem probability_divisible_by_3 : (valid_pairs_count / total_pairs : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_divisible_by_3_l814_81468


namespace NUMINAMATH_GPT_amount_of_second_alloy_used_l814_81417

variable (x : ℝ)

-- Conditions
def chromium_in_first_alloy : ℝ := 0.10 * 15
def chromium_in_second_alloy (x : ℝ) : ℝ := 0.06 * x
def total_weight (x : ℝ) : ℝ := 15 + x
def chromium_in_third_alloy (x : ℝ) : ℝ := 0.072 * (15 + x)

-- Proof statement
theorem amount_of_second_alloy_used :
  1.5 + 0.06 * x = 0.072 * (15 + x) → x = 35 := by
  sorry

end NUMINAMATH_GPT_amount_of_second_alloy_used_l814_81417


namespace NUMINAMATH_GPT_max_profit_l814_81473

-- Definition of the conditions
def production_requirements (tonAprodA tonAprodB tonBprodA tonBprodB: ℕ )
  := tonAprodA = 3 ∧ tonAprodB = 1 ∧ tonBprodA = 2 ∧ tonBprodB = 3

def profit_per_ton ( profitA profitB: ℕ )
  := profitA = 50000 ∧ profitB = 30000

def raw_material_limits ( rawA rawB: ℕ)
  := rawA = 13 ∧ rawB = 18

theorem max_profit 
  (production_requirements: production_requirements 3 1 2 3)
  (profit_per_ton: profit_per_ton 50000 30000)
  (raw_material_limits: raw_material_limits 13 18)
: ∃ (maxProfit: ℕ), maxProfit = 270000 := 
by 
  sorry

end NUMINAMATH_GPT_max_profit_l814_81473


namespace NUMINAMATH_GPT_number_of_children_is_five_l814_81499

/-- The sum of the ages of children born at intervals of 2 years each is 50 years, 
    and the age of the youngest child is 6 years.
    Prove that the number of children is 5. -/
theorem number_of_children_is_five (n : ℕ) (h1 : (0 < n ∧ n / 2 * (8 + 2 * n) = 50)): n = 5 :=
sorry

end NUMINAMATH_GPT_number_of_children_is_five_l814_81499


namespace NUMINAMATH_GPT_min_sum_complementary_events_l814_81463

theorem min_sum_complementary_events (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hP : (1 / y) + (4 / x) = 1) : x + y ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_sum_complementary_events_l814_81463


namespace NUMINAMATH_GPT_carla_highest_final_number_l814_81493

def alice_final_number (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 - 3
  let step3 := step2 / 3
  step3 + 4

def bob_final_number (initial : ℕ) : ℕ :=
  let step1 := initial + 5
  let step2 := step1 * 2
  let step3 := step2 - 4
  step3 / 2

def carla_final_number (initial : ℕ) : ℕ :=
  let step1 := initial - 2
  let step2 := step1 * 2
  let step3 := step2 + 3
  step3 * 2

theorem carla_highest_final_number : carla_final_number 12 > bob_final_number 12 ∧ carla_final_number 12 > alice_final_number 12 :=
  by
  have h_alice : alice_final_number 12 = 11 := by rfl
  have h_bob : bob_final_number 12 = 15 := by rfl
  have h_carla : carla_final_number 12 = 46 := by rfl
  sorry

end NUMINAMATH_GPT_carla_highest_final_number_l814_81493


namespace NUMINAMATH_GPT_factor_expression_l814_81407

theorem factor_expression (a : ℝ) : 
  49 * a ^ 3 + 245 * a ^ 2 + 588 * a = 49 * a * (a ^ 2 + 5 * a + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l814_81407


namespace NUMINAMATH_GPT_find_larger_number_l814_81431

theorem find_larger_number (x y : ℕ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : x + y = 54) : 
  y = 30 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l814_81431


namespace NUMINAMATH_GPT_bees_count_l814_81442

-- Definitions of the conditions
def day1_bees (x : ℕ) := x  -- Number of bees on the first day
def day2_bees (x : ℕ) := 3 * day1_bees x  -- Number of bees on the second day is 3 times that on the first day

theorem bees_count (x : ℕ) (h : day2_bees x = 432) : day1_bees x = 144 :=
by
  dsimp [day1_bees, day2_bees] at h
  have h1 : 3 * x = 432 := h
  sorry

end NUMINAMATH_GPT_bees_count_l814_81442


namespace NUMINAMATH_GPT_initial_number_of_men_l814_81451

theorem initial_number_of_men (M A : ℕ) 
  (h1 : ((M * A) - 22 + 42 = M * (A + 2))) : M = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l814_81451


namespace NUMINAMATH_GPT_max_lateral_surface_area_of_tetrahedron_l814_81490

open Real

theorem max_lateral_surface_area_of_tetrahedron :
  ∀ (PA PB PC : ℝ), (PA^2 + PB^2 + PC^2 = 36) → (PA * PB + PB * PC + PA * PC ≤ 36) →
  (1/2 * (PA * PB + PB * PC + PA * PC) ≤ 18) :=
by
  intro PA PB PC hsum hineq
  sorry

end NUMINAMATH_GPT_max_lateral_surface_area_of_tetrahedron_l814_81490


namespace NUMINAMATH_GPT_probability_multiple_of_4_l814_81460

def prob_at_least_one_multiple_of_4 : ℚ :=
  1 - (38/50)^3

theorem probability_multiple_of_4 (n : ℕ) (h : n = 3) : 
  prob_at_least_one_multiple_of_4 = 28051 / 50000 :=
by
  rw [prob_at_least_one_multiple_of_4, ← h]
  sorry

end NUMINAMATH_GPT_probability_multiple_of_4_l814_81460


namespace NUMINAMATH_GPT_ab_equals_one_l814_81498

theorem ab_equals_one (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (f : ℝ → ℝ) (h3 : f = abs ∘ log) (h4 : f a = f b) : a * b = 1 :=
by
  sorry

end NUMINAMATH_GPT_ab_equals_one_l814_81498


namespace NUMINAMATH_GPT_profit_without_discount_l814_81414

theorem profit_without_discount (CP SP_original SP_discount : ℝ) (h1 : CP > 0) (h2 : SP_discount = CP * 1.14) (h3 : SP_discount = SP_original * 0.95) :
  (SP_original - CP) / CP * 100 = 20 :=
by
  have h4 : SP_original = SP_discount / 0.95 := by sorry
  have h5 : SP_original = CP * 1.2 := by sorry
  have h6 : (SP_original - CP) / CP * 100 = (CP * 1.2 - CP) / CP * 100 := by sorry
  have h7 : (SP_original - CP) / CP * 100 = 20 := by sorry
  exact h7

end NUMINAMATH_GPT_profit_without_discount_l814_81414


namespace NUMINAMATH_GPT_angle_bisector_slope_l814_81462

theorem angle_bisector_slope :
  let m₁ := 2
  let m₂ := 5
  let k := (7 - 2 * Real.sqrt 5) / 11
  True :=
by admit

end NUMINAMATH_GPT_angle_bisector_slope_l814_81462


namespace NUMINAMATH_GPT_max_point_diff_l814_81484

theorem max_point_diff (n : ℕ) : ∃ max_diff, max_diff = 2 :=
by
  -- Conditions from (a)
  -- - \( n \) teams participate in a football tournament.
  -- - Each team plays against every other team exactly once.
  -- - The winning team is awarded 2 points.
  -- - A draw gives -1 point to each team.
  -- - The losing team gets 0 points.
  -- Correct Answer from (b)
  -- - The maximum point difference between teams that are next to each other in the ranking is 2.
  sorry

end NUMINAMATH_GPT_max_point_diff_l814_81484


namespace NUMINAMATH_GPT_value_range_l814_81479

-- Step to ensure proofs about sine and real numbers are within scope
open Real

noncomputable def y (x : ℝ) : ℝ := 2 * sin x * cos x - 1

theorem value_range (x : ℝ) : -2 ≤ y x ∧ y x ≤ 0 :=
by sorry

end NUMINAMATH_GPT_value_range_l814_81479


namespace NUMINAMATH_GPT_real_part_of_i_squared_times_1_plus_i_l814_81429

noncomputable def imaginary_unit : ℂ := Complex.I

theorem real_part_of_i_squared_times_1_plus_i :
  (Complex.re (imaginary_unit^2 * (1 + imaginary_unit))) = -1 :=
by
  sorry

end NUMINAMATH_GPT_real_part_of_i_squared_times_1_plus_i_l814_81429


namespace NUMINAMATH_GPT_problem_solution_l814_81448

theorem problem_solution (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 14) (h2 : a = b + c) : ab - bc + ac = 7 :=
  sorry

end NUMINAMATH_GPT_problem_solution_l814_81448


namespace NUMINAMATH_GPT_length_of_PQ_l814_81459

theorem length_of_PQ (p : ℝ) (h : p > 0) (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hx1x2 : x1 + x2 = 3 * p) (hy1 : y1^2 = 2 * p * x1) (hy2 : y2^2 = 2 * p * x2) 
  (focus : ¬ (y1 = 0)) : (abs (x1 - x2 + 2 * p) = 4 * p) := 
sorry

end NUMINAMATH_GPT_length_of_PQ_l814_81459


namespace NUMINAMATH_GPT_solve_system_eqns_l814_81439

noncomputable def eq1 (x y z : ℚ) : Prop := x^2 + 2 * y * z = x
noncomputable def eq2 (x y z : ℚ) : Prop := y^2 + 2 * z * x = y
noncomputable def eq3 (x y z : ℚ) : Prop := z^2 + 2 * x * y = z

theorem solve_system_eqns (x y z : ℚ) :
  (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ↔
  ((x, y, z) = (0, 0, 0) ∨
   (x, y, z) = (1/3, 1/3, 1/3) ∨
   (x, y, z) = (1, 0, 0) ∨
   (x, y, z) = (0, 1, 0) ∨
   (x, y, z) = (0, 0, 1) ∨
   (x, y, z) = (2/3, -1/3, -1/3) ∨
   (x, y, z) = (-1/3, 2/3, -1/3) ∨
   (x, y, z) = (-1/3, -1/3, 2/3)) :=
by sorry

end NUMINAMATH_GPT_solve_system_eqns_l814_81439


namespace NUMINAMATH_GPT_min_value_2x_plus_y_l814_81467

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/(y + 1) = 2) : 2 * x + y = 3 :=
sorry

end NUMINAMATH_GPT_min_value_2x_plus_y_l814_81467


namespace NUMINAMATH_GPT_squirrel_nuts_collection_l814_81491

theorem squirrel_nuts_collection (n : ℕ) (e u : ℕ → ℕ) :
  (∀ k, 1 ≤ k ∧ k ≤ n → e k = u k + k) ∧
  (∀ k, 1 ≤ k ∧ k ≤ n → u k = e (k + 1) + u k / 100) ∧
  e n = n →
  n = 99 → 
  (∃ S : ℕ, (∀ k, 1 ≤ k ∧ k ≤ n → e k = S)) ∧ 
  S = 9801 :=
sorry

end NUMINAMATH_GPT_squirrel_nuts_collection_l814_81491


namespace NUMINAMATH_GPT_excluded_number_is_35_l814_81494

theorem excluded_number_is_35 (numbers : List ℝ) 
  (h_len : numbers.length = 5)
  (h_avg1 : (numbers.sum / 5) = 27)
  (h_len_excl : (numbers.length - 1) = 4)
  (avg_remaining : ℝ)
  (remaining_numbers : List ℝ)
  (remaining_condition : remaining_numbers.length = 4)
  (h_avg2 : (remaining_numbers.sum / 4) = 25) :
  numbers.sum - remaining_numbers.sum = 35 :=
by sorry

end NUMINAMATH_GPT_excluded_number_is_35_l814_81494


namespace NUMINAMATH_GPT_smallest_of_seven_consecutive_even_numbers_l814_81427

theorem smallest_of_seven_consecutive_even_numbers (n : ℤ) :
  (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 448 → 
  (n - 6) = 58 :=
by
  sorry

end NUMINAMATH_GPT_smallest_of_seven_consecutive_even_numbers_l814_81427


namespace NUMINAMATH_GPT_joan_initial_books_l814_81457

variable (books_sold : ℕ)
variable (books_left : ℕ)

theorem joan_initial_books (h1 : books_sold = 26) (h2 : books_left = 7) : books_sold + books_left = 33 := by
  sorry

end NUMINAMATH_GPT_joan_initial_books_l814_81457


namespace NUMINAMATH_GPT_cost_difference_is_360_l814_81411

def sailboat_cost_per_day : ℕ := 60
def ski_boat_cost_per_hour : ℕ := 80
def ken_days : ℕ := 2
def aldrich_hours_per_day : ℕ := 3
def aldrich_days : ℕ := 2

theorem cost_difference_is_360 :
  let ken_total_cost := sailboat_cost_per_day * ken_days
  let aldrich_total_cost_per_day := ski_boat_cost_per_hour * aldrich_hours_per_day
  let aldrich_total_cost := aldrich_total_cost_per_day * aldrich_days
  let cost_diff := aldrich_total_cost - ken_total_cost
  cost_diff = 360 :=
by
  sorry

end NUMINAMATH_GPT_cost_difference_is_360_l814_81411


namespace NUMINAMATH_GPT_merry_boxes_on_sunday_l814_81454

theorem merry_boxes_on_sunday
  (num_boxes_saturday : ℕ := 50)
  (apples_per_box : ℕ := 10)
  (total_apples_sold : ℕ := 720)
  (remaining_boxes : ℕ := 3) :
  num_boxes_saturday * apples_per_box ≤ total_apples_sold →
  (total_apples_sold - num_boxes_saturday * apples_per_box) / apples_per_box + remaining_boxes = 25 := by
  intros
  sorry

end NUMINAMATH_GPT_merry_boxes_on_sunday_l814_81454


namespace NUMINAMATH_GPT_solve_equation_l814_81428

theorem solve_equation : ∀ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) → x = 2 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_solve_equation_l814_81428


namespace NUMINAMATH_GPT_ages_correct_l814_81446

-- Definitions of the given conditions
def john_age : ℕ := 42
def tim_age : ℕ := 79
def james_age : ℕ := 30
def lisa_age : ℚ := 54.5
def kate_age : ℕ := 34
def michael_age : ℚ := 61.5
def anna_age : ℚ := 54.5

-- Mathematically equivalent proof problem
theorem ages_correct :
  (james_age = 30) ∧
  (lisa_age = 54.5) ∧
  (kate_age = 34) ∧
  (michael_age = 61.5) ∧
  (anna_age = 54.5) :=
by {
  sorry  -- Proof to be filled in
}

end NUMINAMATH_GPT_ages_correct_l814_81446


namespace NUMINAMATH_GPT_quadratic_has_real_root_l814_81409

theorem quadratic_has_real_root (a b : ℝ) : 
  (¬(∀ x : ℝ, x^2 + a * x + b ≠ 0)) → (∃ x : ℝ, x^2 + a * x + b = 0) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l814_81409


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_sqrt2_l814_81474

noncomputable def eccentricity_of_hyperbola {a b : ℝ} (h : a ≠ 0) (hb : b = a) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (c / a)

theorem hyperbola_eccentricity_is_sqrt2 {a : ℝ} (h : a ≠ 0) :
  eccentricity_of_hyperbola h (rfl) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_sqrt2_l814_81474


namespace NUMINAMATH_GPT_fractional_part_painted_correct_l814_81483

noncomputable def fractional_part_painted (time_fence : ℕ) (time_hole : ℕ) : ℚ :=
  (time_hole : ℚ) / time_fence

theorem fractional_part_painted_correct : fractional_part_painted 60 40 = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fractional_part_painted_correct_l814_81483


namespace NUMINAMATH_GPT_train_length_l814_81408

theorem train_length (v_kmh : ℝ) (p_len : ℝ) (t_sec : ℝ) (l_train : ℝ) 
  (h_v : v_kmh = 72) (h_p : p_len = 250) (h_t : t_sec = 26) :
  l_train = 270 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l814_81408


namespace NUMINAMATH_GPT_original_number_of_employees_l814_81425

theorem original_number_of_employees (x : ℕ) 
  (h1 : 0.77 * (x : ℝ) = 328) : x = 427 :=
sorry

end NUMINAMATH_GPT_original_number_of_employees_l814_81425


namespace NUMINAMATH_GPT_max_chickens_ducks_l814_81416

theorem max_chickens_ducks (x y : ℕ) 
  (h1 : ∀ (k : ℕ), k = 6 → x + y - 6 ≥ 2) 
  (h2 : ∀ (k : ℕ), k = 9 → y ≥ 1) : 
  x + y ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_chickens_ducks_l814_81416


namespace NUMINAMATH_GPT_kiwi_count_l814_81401

theorem kiwi_count (o a b k : ℕ) (h1 : o + a + b + k = 540) (h2 : a = 3 * o) (h3 : b = 4 * a) (h4 : k = 5 * b) : k = 420 :=
sorry

end NUMINAMATH_GPT_kiwi_count_l814_81401


namespace NUMINAMATH_GPT_set_union_example_l814_81433

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem set_union_example : M ∪ N = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_set_union_example_l814_81433


namespace NUMINAMATH_GPT_ellipse_equation_fixed_point_l814_81403

/-- Given an ellipse with equation x^2 / a^2 + y^2 / b^2 = 1 where a > b > 0 and eccentricity e = 1/2,
    prove that the equation of the ellipse is x^2 / 4 + y^2 / 3 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = b^2 + (a / 2)^2) :
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by sorry

/-- Given an ellipse with equation x^2 / 4 + y^2 / 3 = 1,
    if a line l: y = kx + m intersects the ellipse at two points A and B (which are not the left and right vertices),
    and a circle passing through the right vertex of the ellipse has AB as its diameter,
    prove that the line passes through a fixed point and find its coordinates -/
theorem fixed_point (k m : ℝ) :
  (∃ x y, (x = 2 / 7 ∧ y = 0)) :=
by sorry

end NUMINAMATH_GPT_ellipse_equation_fixed_point_l814_81403


namespace NUMINAMATH_GPT_rayden_has_more_birds_l814_81405

-- Definitions based on given conditions
def ducks_lily := 20
def geese_lily := 10
def chickens_lily := 5
def pigeons_lily := 30

def ducks_rayden := 3 * ducks_lily
def geese_rayden := 4 * geese_lily
def chickens_rayden := 5 * chickens_lily
def pigeons_rayden := pigeons_lily / 2

def more_ducks := ducks_rayden - ducks_lily
def more_geese := geese_rayden - geese_lily
def more_chickens := chickens_rayden - chickens_lily
def fewer_pigeons := pigeons_rayden - pigeons_lily

def total_more_birds := more_ducks + more_geese + more_chickens - fewer_pigeons

-- Statement to prove that Rayden has 75 more birds in total than Lily
theorem rayden_has_more_birds : total_more_birds = 75 := by
    sorry

end NUMINAMATH_GPT_rayden_has_more_birds_l814_81405


namespace NUMINAMATH_GPT_num_divisors_1215_l814_81415

theorem num_divisors_1215 : (Finset.filter (λ d => 1215 % d = 0) (Finset.range (1215 + 1))).card = 12 :=
by
  sorry

end NUMINAMATH_GPT_num_divisors_1215_l814_81415


namespace NUMINAMATH_GPT_max_min_values_of_f_l814_81456

noncomputable def f (x : ℝ) : ℝ := x^2

theorem max_min_values_of_f : 
  (∀ x, -3 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 9) ∧ (∃ x, -3 ≤ x ∧ x ≤ 1 ∧ f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_max_min_values_of_f_l814_81456


namespace NUMINAMATH_GPT_inequality_proof_l814_81472

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l814_81472


namespace NUMINAMATH_GPT_sally_quarters_l814_81489

noncomputable def initial_quarters : ℕ := 760
noncomputable def spent_quarters : ℕ := 418
noncomputable def remaining_quarters : ℕ := 342

theorem sally_quarters : initial_quarters - spent_quarters = remaining_quarters :=
by sorry

end NUMINAMATH_GPT_sally_quarters_l814_81489


namespace NUMINAMATH_GPT_sqrt_of_square_is_identity_l814_81480

variable {a : ℝ} (h : a > 0)

theorem sqrt_of_square_is_identity (h : a > 0) : Real.sqrt (a^2) = a := 
  sorry

end NUMINAMATH_GPT_sqrt_of_square_is_identity_l814_81480


namespace NUMINAMATH_GPT_find_pots_l814_81423

def num_pots := 46
def cost_green_lily := 9
def cost_spider_plant := 6
def total_cost := 390

theorem find_pots (x y : ℕ) (h1 : x + y = num_pots) (h2 : cost_green_lily * x + cost_spider_plant * y = total_cost) :
  x = 38 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_pots_l814_81423


namespace NUMINAMATH_GPT_solid_id_views_not_cylinder_l814_81402

theorem solid_id_views_not_cylinder :
  ∀ (solid : Type),
  (∃ (shape1 shape2 shape3 : solid),
    shape1 = shape2 ∧ shape2 = shape3) →
  solid ≠ cylinder :=
by 
  sorry

end NUMINAMATH_GPT_solid_id_views_not_cylinder_l814_81402


namespace NUMINAMATH_GPT_factorial_ends_with_base_8_zeroes_l814_81419

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end NUMINAMATH_GPT_factorial_ends_with_base_8_zeroes_l814_81419


namespace NUMINAMATH_GPT_sum_first_10_terms_arithmetic_sequence_l814_81466

-- Define the first term and the sum of the second and sixth terms as given conditions
def a1 : ℤ := -2
def condition_a2_a6 (a2 a6 : ℤ) : Prop := a2 + a6 = 2

-- Define the general term 'a_n' of the arithmetic sequence
def a_n (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum 'S_n' of the first 'n' terms of the arithmetic sequence
def S_n (a1 d : ℤ) (n : ℤ) : ℤ := n * (a1 + ((n - 1) * d) / 2)

-- The theorem statement to prove that S_10 = 25 given the conditions
theorem sum_first_10_terms_arithmetic_sequence 
  (d a2 a6 : ℤ) 
  (h1 : a2 = a_n a1 d 2) 
  (h2 : a6 = a_n a1 d 6)
  (h3 : condition_a2_a6 a2 a6) : 
  S_n a1 d 10 = 25 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_10_terms_arithmetic_sequence_l814_81466


namespace NUMINAMATH_GPT_spinning_class_frequency_l814_81461

/--
We define the conditions given in the problem:
- duration of each class in hours,
- calorie burn rate per minute,
- total calories burned per week.
We then state that the number of classes James attends per week is equal to 3.
-/
def class_duration_hours : ℝ := 1.5
def calories_per_minute : ℝ := 7
def total_calories_per_week : ℝ := 1890

theorem spinning_class_frequency :
  total_calories_per_week / (class_duration_hours * 60 * calories_per_minute) = 3 :=
by
  sorry

end NUMINAMATH_GPT_spinning_class_frequency_l814_81461


namespace NUMINAMATH_GPT_simpl_eval_l814_81482

variable (a b : ℚ)

theorem simpl_eval (h_a : a = 1/2) (h_b : b = -1/3) :
    5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (- a * b ^ 2 + 3 * a ^ 2 * b) = -11 / 36 := by
  sorry

end NUMINAMATH_GPT_simpl_eval_l814_81482


namespace NUMINAMATH_GPT_average_of_eight_twelve_and_N_is_12_l814_81418

theorem average_of_eight_twelve_and_N_is_12 (N : ℝ) (hN : 11 < N ∧ N < 19) : (8 + 12 + N) / 3 = 12 :=
by
  -- Place the complete proof step here
  sorry

end NUMINAMATH_GPT_average_of_eight_twelve_and_N_is_12_l814_81418


namespace NUMINAMATH_GPT_cos_squared_value_l814_81465

theorem cos_squared_value (α : ℝ) (h : Real.tan (α + π/4) = 3/4) : Real.cos (π/4 - α) ^ 2 = 9 / 25 :=
sorry

end NUMINAMATH_GPT_cos_squared_value_l814_81465


namespace NUMINAMATH_GPT_number_of_technicians_l814_81434

theorem number_of_technicians
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (avg_salary_techs : ℝ)
  (avg_salary_rest : ℝ)
  (num_techs num_rest : ℕ)
  (h_total_workers : total_workers = 56)
  (h_avg_salary_all : avg_salary_all = 6750)
  (h_avg_salary_techs : avg_salary_techs = 12000)
  (h_avg_salary_rest : avg_salary_rest = 6000)
  (h_eq_workers : num_techs + num_rest = total_workers)
  (h_eq_salaries : (num_techs * avg_salary_techs + num_rest * avg_salary_rest) = total_workers * avg_salary_all) :
  num_techs = 7 := sorry

end NUMINAMATH_GPT_number_of_technicians_l814_81434


namespace NUMINAMATH_GPT_solve_ticket_problem_l814_81435

def ticket_problem : Prop :=
  ∃ S N : ℕ, S + N = 2000 ∧ 9 * S + 11 * N = 20960 ∧ S = 520

theorem solve_ticket_problem : ticket_problem :=
sorry

end NUMINAMATH_GPT_solve_ticket_problem_l814_81435


namespace NUMINAMATH_GPT_Oliver_total_workout_hours_l814_81410

-- Define the working hours for each day
def Monday_hours : ℕ := 4
def Tuesday_hours : ℕ := Monday_hours - 2
def Wednesday_hours : ℕ := 2 * Monday_hours
def Thursday_hours : ℕ := 2 * Tuesday_hours

-- Prove that the total hours Oliver worked out adds up to 18
theorem Oliver_total_workout_hours : Monday_hours + Tuesday_hours + Wednesday_hours + Thursday_hours = 18 := by
  sorry

end NUMINAMATH_GPT_Oliver_total_workout_hours_l814_81410


namespace NUMINAMATH_GPT_interval_second_bell_l814_81452

theorem interval_second_bell 
  (T : ℕ)
  (h1 : ∀ n : ℕ, n ≠ 0 → 630 % n = 0)
  (h2 : gcd T 630 = T)
  (h3 : lcm 9 (lcm 14 18) = lcm 9 (lcm 14 18))
  (h4 : 630 % lcm 9 (lcm 14 18) = 0) : 
  T = 5 :=
sorry

end NUMINAMATH_GPT_interval_second_bell_l814_81452


namespace NUMINAMATH_GPT_gcd_1729_78945_is_1_l814_81443

theorem gcd_1729_78945_is_1 :
  ∃ m n : ℤ, 1729 * m + 78945 * n = 1 := sorry

end NUMINAMATH_GPT_gcd_1729_78945_is_1_l814_81443


namespace NUMINAMATH_GPT_find_13x2_22xy_13y2_l814_81432

variable (x y : ℝ)

theorem find_13x2_22xy_13y2 
  (h1 : 3 * x + 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) 
: 13 * x^2 + 22 * x * y + 13 * y^2 = 184 := 
sorry

end NUMINAMATH_GPT_find_13x2_22xy_13y2_l814_81432


namespace NUMINAMATH_GPT_impossible_to_reduce_time_l814_81495

def current_speed := 60 -- speed in km/h
def time_per_km (v : ℕ) : ℕ := 60 / v -- 60 minutes divided by speed in km/h gives time per km in minutes

theorem impossible_to_reduce_time (v : ℕ) (h : v = current_speed) : time_per_km v = 1 → ¬(time_per_km v - 1 = 0) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_impossible_to_reduce_time_l814_81495


namespace NUMINAMATH_GPT_paper_cranes_l814_81436

theorem paper_cranes (B C A : ℕ) (h1 : A + B + C = 1000)
  (h2 : A = 3 * B - 100)
  (h3 : C = A - 67) : A = 443 := by
  sorry

end NUMINAMATH_GPT_paper_cranes_l814_81436


namespace NUMINAMATH_GPT_fraction_of_pizza_peter_ate_l814_81478

theorem fraction_of_pizza_peter_ate (total_slices : ℕ) (peter_slices : ℕ) (shared_slices : ℚ) 
  (pizza_fraction : ℚ) : 
  total_slices = 16 → 
  peter_slices = 2 → 
  shared_slices = 1/3 → 
  pizza_fraction = peter_slices / total_slices + (1 / 2) * shared_slices / total_slices → 
  pizza_fraction = 13 / 96 :=
by 
  intros h1 h2 h3 h4
  -- to be proved later
  sorry

end NUMINAMATH_GPT_fraction_of_pizza_peter_ate_l814_81478


namespace NUMINAMATH_GPT_geometric_progression_exists_l814_81430

theorem geometric_progression_exists :
  ∃ (b_1 b_2 b_3 b_4 q : ℚ), 
    b_1 - b_2 = 35 ∧ 
    b_3 - b_4 = 560 ∧ 
    b_2 = b_1 * q ∧ 
    b_3 = b_1 * q^2 ∧ 
    b_4 = b_1 * q^3 ∧ 
    ((b_1 = 7 ∧ q = -4 ∧ b_2 = -28 ∧ b_3 = 112 ∧ b_4 = -448) ∨ 
    (b_1 = -35/3 ∧ q = 4 ∧ b_2 = -140/3 ∧ b_3 = -560/3 ∧ b_4 = -2240/3)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_exists_l814_81430


namespace NUMINAMATH_GPT_paul_bought_150_books_l814_81426

theorem paul_bought_150_books (initial_books sold_books books_now : ℤ)
  (h1 : initial_books = 2)
  (h2 : sold_books = 94)
  (h3 : books_now = 58) :
  initial_books - sold_books + books_now = 150 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_paul_bought_150_books_l814_81426


namespace NUMINAMATH_GPT_average_weasels_caught_per_week_l814_81469

-- Definitions based on the conditions
def initial_weasels : ℕ := 100
def initial_rabbits : ℕ := 50
def foxes : ℕ := 3
def rabbits_caught_per_week_per_fox : ℕ := 2
def weeks : ℕ := 3
def remaining_animals : ℕ := 96

-- Main theorem statement
theorem average_weasels_caught_per_week :
  (foxes * weeks * rabbits_caught_per_week_per_fox +
   foxes * weeks * W = initial_weasels + initial_rabbits - remaining_animals) →
  W = 4 :=
sorry

end NUMINAMATH_GPT_average_weasels_caught_per_week_l814_81469


namespace NUMINAMATH_GPT_cube_sum_decomposition_l814_81444

theorem cube_sum_decomposition : 
  (∃ (a b c d e : ℤ), (1000 * x^3 + 27) = (a * x + b) * (c * x^2 + d * x + e) ∧ (a + b + c + d + e = 92)) :=
by
  sorry

end NUMINAMATH_GPT_cube_sum_decomposition_l814_81444


namespace NUMINAMATH_GPT_sin_double_angle_l814_81447

theorem sin_double_angle (x : ℝ) (h : Real.sin (x - π / 4) = 3 / 5) : Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l814_81447


namespace NUMINAMATH_GPT_part1_part2_l814_81413

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1 (m : ℝ) : (∃ x, deriv f x = 2 ∧ f x = 2 * x + m) → m = -Real.exp 1 :=
sorry

theorem part2 : ∀ x > 0, -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x) :=
sorry

end NUMINAMATH_GPT_part1_part2_l814_81413


namespace NUMINAMATH_GPT_Reema_loan_problem_l814_81487

-- Define problem parameters
def Principal : ℝ := 150000
def Interest : ℝ := 42000
def ProfitRate : ℝ := 0.1
def Profit : ℝ := 25000

-- State the problem as a Lean 4 theorem
theorem Reema_loan_problem (R : ℝ) (Investment : ℝ) : 
  Principal * (R / 100) * R = Interest ∧ 
  Profit = Investment * ProfitRate * R ∧ 
  R = 5 ∧ 
  Investment = 50000 :=
by
  sorry

end NUMINAMATH_GPT_Reema_loan_problem_l814_81487


namespace NUMINAMATH_GPT_wrapping_paper_area_correct_l814_81492

-- Conditions as given in the problem
variables (l w h : ℝ)
variable (hlw : l > w)

-- Definition of the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ :=
  (l + 2 * h) * (w + 2 * h)

-- Proof statement
theorem wrapping_paper_area_correct (hlw : l > w) : 
  wrapping_paper_area l w h = l * w + 2 * l * h + 2 * w * h + 4 * h^2 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_correct_l814_81492


namespace NUMINAMATH_GPT_machine_working_days_l814_81455

variable {V a b c x y z : ℝ} 

noncomputable def machine_individual_times_condition (a b c : ℝ) : Prop :=
  ∀ (x y z : ℝ), (x = a + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (y = b + (-(a + b) + Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (z = (-(c * (a + b)) + c * Real.sqrt ((a - b) ^ 2 + 4 * a * b * c ^ 2)) / (2 * (c + 1))) ∧
                 (c > 1)

theorem machine_working_days (h1 : x = (z / c) + a) (h2 : y = (z / c) + b) (h3 : z = c * (z / c)) :
  machine_individual_times_condition a b c :=
by
  sorry

end NUMINAMATH_GPT_machine_working_days_l814_81455


namespace NUMINAMATH_GPT_three_digit_number_l814_81481

theorem three_digit_number (a b c : ℕ) (h1 : a + b + c = 10) (h2 : b = a + c) (h3 : 100 * c + 10 * b + a = 100 * a + 10 * b + c + 99) : (100 * a + 10 * b + c) = 253 := 
by
  sorry

end NUMINAMATH_GPT_three_digit_number_l814_81481


namespace NUMINAMATH_GPT_find_f3_l814_81404

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f3 (h1 : ∀ x : ℝ, f (x + 1) = f (-x - 1))
                (h2 : ∀ x : ℝ, f (2 - x) = -f x) :
  f 3 = 0 := 
sorry

end NUMINAMATH_GPT_find_f3_l814_81404


namespace NUMINAMATH_GPT_simplify_and_evaluate_l814_81496

-- Define the expression
def expr (x : ℝ) : ℝ := x^2 * (x + 1) - x * (x^2 - x + 1)

-- The main theorem stating the equivalence
theorem simplify_and_evaluate (x : ℝ) (h : x = 5) : expr x = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_and_evaluate_l814_81496


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l814_81475

theorem right_triangle_hypotenuse (a h : ℝ) (r : ℝ) (h1 : r = 8) (h2 : h = a * Real.sqrt 2)
  (h3 : r = (a - h) / 2) : h = 16 * (Real.sqrt 2 + 1) := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l814_81475


namespace NUMINAMATH_GPT_apples_to_grapes_proof_l814_81486

theorem apples_to_grapes_proof :
  (3 / 4 * 12 = 9) → (1 / 3 * 9 = 3) :=
by
  sorry

end NUMINAMATH_GPT_apples_to_grapes_proof_l814_81486


namespace NUMINAMATH_GPT_number_of_levels_l814_81424

theorem number_of_levels (total_capacity : ℕ) (additional_cars : ℕ) (already_parked_cars : ℕ) (n : ℕ) :
  total_capacity = 425 →
  additional_cars = 62 →
  already_parked_cars = 23 →
  n = total_capacity / (already_parked_cars + additional_cars) →
  n = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_levels_l814_81424


namespace NUMINAMATH_GPT_side_length_of_square_l814_81476

theorem side_length_of_square (r : ℝ) (A : ℝ) (s : ℝ) 
  (h1 : π * r^2 = 36 * π) 
  (h2 : s = 2 * r) : 
  s = 12 :=
by 
  sorry

end NUMINAMATH_GPT_side_length_of_square_l814_81476


namespace NUMINAMATH_GPT_mark_savings_l814_81477

-- Given conditions
def original_price : ℝ := 300
def discount_rate : ℝ := 0.20
def cheaper_lens_price : ℝ := 220

-- Definitions derived from conditions
def discount_amount : ℝ := original_price * discount_rate
def discounted_price : ℝ := original_price - discount_amount
def savings : ℝ := discounted_price - cheaper_lens_price

-- Statement to prove
theorem mark_savings : savings = 20 :=
by
  -- Definitions incorporated
  have h1 : discount_amount = 300 * 0.20 := rfl
  have h2 : discounted_price = 300 - discount_amount := rfl
  have h3 : cheaper_lens_price = 220 := rfl
  have h4 : savings = discounted_price - cheaper_lens_price := rfl
  sorry

end NUMINAMATH_GPT_mark_savings_l814_81477


namespace NUMINAMATH_GPT_box_volume_l814_81445

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end NUMINAMATH_GPT_box_volume_l814_81445


namespace NUMINAMATH_GPT_part1_l814_81488

theorem part1 (a b c : ℚ) (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : c^3 = 27) (h4 : a * b < 0) (h5 : b * c > 0) : 
  a * b - b * c + c * a = -33 := by
  sorry

end NUMINAMATH_GPT_part1_l814_81488


namespace NUMINAMATH_GPT_book_arrangement_count_l814_81458

theorem book_arrangement_count :
  let total_books := 6
  let identical_science_books := 3
  let unique_other_books := total_books - identical_science_books
  (total_books! / (identical_science_books! * unique_other_books!)) = 120 := by
  sorry

end NUMINAMATH_GPT_book_arrangement_count_l814_81458


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l814_81412

theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, (∀ (a_n : ℕ → ℝ), a_n 1 = 3 ∧ a_n 3 = 7 ∧ (∀ n, a_n n = 3 + (n - 1) * d)) → d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l814_81412


namespace NUMINAMATH_GPT_isosceles_triangle_area_l814_81464

theorem isosceles_triangle_area (p x : ℝ) 
  (h1 : 2 * p = 6 * x) 
  (h2 : 0 < p) 
  (h3 : 0 < x) :
  (1 / 2) * (2 * x) * (Real.sqrt (8 * p^2 / 9)) = (Real.sqrt 8 * p^2) / 3 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l814_81464


namespace NUMINAMATH_GPT_total_money_spent_l814_81437

def total_cost (blades_cost : Nat) (string_cost : Nat) : Nat :=
  blades_cost + string_cost

theorem total_money_spent 
  (num_blades : Nat)
  (cost_per_blade : Nat)
  (string_cost : Nat)
  (h1 : num_blades = 4)
  (h2 : cost_per_blade = 8)
  (h3 : string_cost = 7) :
  total_cost (num_blades * cost_per_blade) string_cost = 39 :=
by
  sorry

end NUMINAMATH_GPT_total_money_spent_l814_81437


namespace NUMINAMATH_GPT_matilda_jellybeans_l814_81471

theorem matilda_jellybeans (steve_jellybeans : ℕ) (h_steve : steve_jellybeans = 84)
  (h_matt : ℕ) (h_matt_calc : h_matt = 10 * steve_jellybeans)
  (h_matilda : ℕ) (h_matilda_calc : h_matilda = h_matt / 2) :
  h_matilda = 420 := by
  sorry

end NUMINAMATH_GPT_matilda_jellybeans_l814_81471


namespace NUMINAMATH_GPT_find_dividend_l814_81422

-- Define the given conditions
def quotient : ℝ := 0.0012000000000000001
def divisor : ℝ := 17

-- State the problem: Prove that the dividend is the product of the quotient and the divisor
theorem find_dividend (q : ℝ) (d : ℝ) (hq : q = 0.0012000000000000001) (hd : d = 17) : 
  q * d = 0.0204000000000000027 :=
sorry

end NUMINAMATH_GPT_find_dividend_l814_81422


namespace NUMINAMATH_GPT_f_monotonically_increasing_on_1_to_infinity_l814_81400

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_monotonically_increasing_on_1_to_infinity :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := 
sorry

end NUMINAMATH_GPT_f_monotonically_increasing_on_1_to_infinity_l814_81400


namespace NUMINAMATH_GPT_hypotenuse_of_right_triangle_l814_81406

theorem hypotenuse_of_right_triangle (a b : ℕ) (h : ℕ)
  (h1 : a = 15) (h2 : b = 36) (right_triangle : a^2 + b^2 = h^2) : h = 39 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_of_right_triangle_l814_81406


namespace NUMINAMATH_GPT_triangle_area_is_correct_l814_81449

noncomputable def triangle_area (a b c B : ℝ) : ℝ := 
  0.5 * a * c * Real.sin B

theorem triangle_area_is_correct :
  let a := Real.sqrt 2
  let c := Real.sqrt 2
  let b := Real.sqrt 6
  let B := 2 * Real.pi / 3 -- 120 degrees in radians
  triangle_area a b c B = Real.sqrt 3 / 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l814_81449


namespace NUMINAMATH_GPT_distance_between_Jay_and_Paul_l814_81453

theorem distance_between_Jay_and_Paul
  (initial_distance : ℕ)
  (jay_speed : ℕ)
  (paul_speed : ℕ)
  (time : ℕ)
  (jay_distance_walked : ℕ)
  (paul_distance_walked : ℕ) :
  initial_distance = 3 →
  jay_speed = 1 / 20 →
  paul_speed = 3 / 40 →
  time = 120 →
  jay_distance_walked = jay_speed * time →
  paul_distance_walked = paul_speed * time →
  initial_distance + jay_distance_walked + paul_distance_walked = 18 := by
  sorry

end NUMINAMATH_GPT_distance_between_Jay_and_Paul_l814_81453


namespace NUMINAMATH_GPT_value_of_w_l814_81440

theorem value_of_w (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := 
sorry

end NUMINAMATH_GPT_value_of_w_l814_81440


namespace NUMINAMATH_GPT_problem_dorlir_ahmeti_equality_case_l814_81485

theorem problem_dorlir_ahmeti (x y z : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h : x^2 + y^2 + z^2 = x + y + z) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 :=
sorry
  
theorem equality_case (x y z : ℝ)
  (hx : x = 0) (hy : y = 0) (hz : z = 0) : 
  (x + 1) / Real.sqrt (x^5 + x + 1) + 
  (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) = 3 :=
sorry

end NUMINAMATH_GPT_problem_dorlir_ahmeti_equality_case_l814_81485
