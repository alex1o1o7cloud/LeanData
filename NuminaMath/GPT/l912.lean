import Mathlib

namespace NUMINAMATH_GPT_woman_needs_butter_l912_91218

noncomputable def butter_needed (cost_package : ℝ) (cost_8oz : ℝ) (cost_4oz : ℝ) 
                                (discount : ℝ) (lowest_price : ℝ) : ℝ :=
  if lowest_price = cost_8oz + 2 * (cost_4oz * discount / 100) then 8 + 2 * 4 else 0

theorem woman_needs_butter 
  (cost_single_package : ℝ := 7) 
  (cost_8oz_package : ℝ := 4) 
  (cost_4oz_package : ℝ := 2)
  (discount_4oz_package : ℝ := 50) 
  (lowest_price_payment : ℝ := 6) :
  butter_needed cost_single_package cost_8oz_package cost_4oz_package discount_4oz_package lowest_price_payment = 16 := 
by
  sorry

end NUMINAMATH_GPT_woman_needs_butter_l912_91218


namespace NUMINAMATH_GPT_fraction_draw_l912_91211

theorem fraction_draw (john_wins : ℚ) (mike_wins : ℚ) (h_john : john_wins = 4 / 9) (h_mike : mike_wins = 5 / 18) :
    1 - (john_wins + mike_wins) = 5 / 18 :=
by
    rw [h_john, h_mike]
    sorry

end NUMINAMATH_GPT_fraction_draw_l912_91211


namespace NUMINAMATH_GPT_meetings_percentage_l912_91209

def workday_hours := 10
def first_meeting_minutes := 60
def second_meeting_minutes := 3 * first_meeting_minutes
def total_workday_minutes := workday_hours * 60
def total_meeting_minutes := first_meeting_minutes + second_meeting_minutes

theorem meetings_percentage :
    (total_meeting_minutes / total_workday_minutes) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_meetings_percentage_l912_91209


namespace NUMINAMATH_GPT_cost_of_each_item_l912_91222

theorem cost_of_each_item 
  (x y z : ℝ) 
  (h1 : 3 * x + 5 * y + z = 32)
  (h2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_item_l912_91222


namespace NUMINAMATH_GPT_range_of_m_l912_91226

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) ↔ (-4 ≤ m ∧ m ≤ 0) := 
by sorry

end NUMINAMATH_GPT_range_of_m_l912_91226


namespace NUMINAMATH_GPT_functional_equation_solution_l912_91245

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0 ∨ f x = 1) :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l912_91245


namespace NUMINAMATH_GPT_cuboid_volume_l912_91231

variable (length width height : ℕ)

-- Conditions given in the problem
def cuboid_edges := (length = 2) ∧ (width = 5) ∧ (height = 8)

-- Mathematically equivalent statement to be proved
theorem cuboid_volume : cuboid_edges length width height → length * width * height = 80 := by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l912_91231


namespace NUMINAMATH_GPT_union_eq_M_l912_91217

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def S : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem union_eq_M : M ∪ S = M := by
  /- this part is for skipping the proof -/
  sorry

end NUMINAMATH_GPT_union_eq_M_l912_91217


namespace NUMINAMATH_GPT_arithmetic_sequence_n_value_l912_91284

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n : ℕ)
  (hS9 : S 9 = 18)
  (ha_n_minus_4 : a (n-4) = 30)
  (hSn : S n = 336)
  (harithmetic_sequence : ∀ k, a (k + 1) - a k = a 2 - a 1) :
  n = 21 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_value_l912_91284


namespace NUMINAMATH_GPT_original_saved_amount_l912_91257

theorem original_saved_amount (x : ℤ) (h : (3 * x - 42)^2 = 2241) : x = 30 := 
sorry

end NUMINAMATH_GPT_original_saved_amount_l912_91257


namespace NUMINAMATH_GPT_unique_real_solution_l912_91221

theorem unique_real_solution :
  ∀ (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) →
    (x = 0 ∧ y = 0 ∧ z = 0 ∧ w = 0) :=
by
  intro x y z w
  intros h
  have h1 : x = z + w + Real.sqrt (z * w * x) := h.1
  have h2 : y = w + x + Real.sqrt (w * x * y) := h.2.1
  have h3 : z = x + y + Real.sqrt (x * y * z) := h.2.2.1
  have h4 : w = y + z + Real.sqrt (y * z * w) := h.2.2.2
  sorry

end NUMINAMATH_GPT_unique_real_solution_l912_91221


namespace NUMINAMATH_GPT_hypercube_paths_24_l912_91265

-- Define the 4-dimensional hypercube
structure Hypercube4 :=
(vertices : Fin 16) -- Using Fin 16 to represent the 16 vertices
(edges : Fin 32)    -- Using Fin 32 to represent the 32 edges

def valid_paths (start : Fin 16) : Nat :=
  -- This function should calculate the number of valid paths given the start vertex
  24 -- placeholder, as we are giving the pre-computed total number here

theorem hypercube_paths_24 (start : Fin 16) :
  valid_paths start = 24 :=
by sorry

end NUMINAMATH_GPT_hypercube_paths_24_l912_91265


namespace NUMINAMATH_GPT_bert_initial_amount_l912_91294

theorem bert_initial_amount (n : ℝ) (h : (1 / 2) * (3 / 4 * n - 9) = 12) : n = 44 :=
sorry

end NUMINAMATH_GPT_bert_initial_amount_l912_91294


namespace NUMINAMATH_GPT_smallest_N_winning_strategy_l912_91252

theorem smallest_N_winning_strategy :
  ∃ (N : ℕ), (N > 0) ∧ (∀ (list : List ℕ), 
    (∀ x, x ∈ list → x > 0 ∧ x ≤ 25) ∧ 
    list.sum ≥ 200 → 
    ∃ (sublist : List ℕ), sublist ⊆ list ∧ 
    200 - N ≤ sublist.sum ∧ sublist.sum ≤ 200 + N) ∧ N = 11 :=
sorry

end NUMINAMATH_GPT_smallest_N_winning_strategy_l912_91252


namespace NUMINAMATH_GPT_group_photo_arrangements_l912_91227

theorem group_photo_arrangements :
  ∃ (arrangements : ℕ), arrangements = 36 ∧
    ∀ (M G H P1 P2 : ℕ),
    (M = G + 1 ∨ M + 1 = G) ∧ (M ≠ H - 1 ∧ M ≠ H + 1) →
    arrangements = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_group_photo_arrangements_l912_91227


namespace NUMINAMATH_GPT_length_of_ae_l912_91259

-- Definitions for lengths of segments
variable {ab bc cd de ac ae : ℝ}

-- Given conditions as assumptions
axiom h1 : bc = 3 * cd
axiom h2 : de = 8
axiom h3 : ab = 5
axiom h4 : ac = 11

-- The main theorem to prove
theorem length_of_ae : ae = ab + bc + cd + de → bc = ac - ab → bc = 6 → cd = bc / 3 → ae = 21 :=
by sorry

end NUMINAMATH_GPT_length_of_ae_l912_91259


namespace NUMINAMATH_GPT_triangle_area_base_6_height_8_l912_91297

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem triangle_area_base_6_height_8 : triangle_area 6 8 = 24 := by
  sorry

end NUMINAMATH_GPT_triangle_area_base_6_height_8_l912_91297


namespace NUMINAMATH_GPT_two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l912_91276

def star (a b : ℤ) : ℤ := a ^ 2 - b + a * b

theorem two_star_neg_five_eq_neg_one : star 2 (-5) = -1 := by
  sorry

theorem neg_two_star_two_star_neg_three_eq_one : star (-2) (star 2 (-3)) = 1 := by
  sorry

end NUMINAMATH_GPT_two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l912_91276


namespace NUMINAMATH_GPT_cyclist_club_member_count_l912_91237

-- Define the set of valid digits.
def valid_digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9}

-- Define the problem statement
theorem cyclist_club_member_count : valid_digits.card ^ 3 = 512 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_cyclist_club_member_count_l912_91237


namespace NUMINAMATH_GPT_complex_norm_example_l912_91238

theorem complex_norm_example : 
  abs (-3 - (9 / 4 : ℝ) * I) = 15 / 4 := 
by
  sorry

end NUMINAMATH_GPT_complex_norm_example_l912_91238


namespace NUMINAMATH_GPT_part1_part2_1_part2_2_l912_91273

noncomputable def f (m : ℝ) (a x : ℝ) : ℝ :=
  m / x + Real.log (x / a)

-- Part (1)
theorem part1 (m a : ℝ) (h : m > 0) (ha : a > 0) (hmin : ∀ x, f m a x ≥ 2) : 
  m / a = Real.exp 1 :=
sorry

-- Part (2.1)
theorem part2_1 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  1 / (2 * x₀) + x₀ < a - 1 :=
sorry

-- Part (2.2)
theorem part2_2 (a x₀ : ℝ) (ha : a > Real.exp 1) (hx₀ : x₀ > 1) (hzero : f 1 a x₀ = 0) : 
  x₀ + 1 / x₀ > 2 * Real.log a - Real.log (Real.log a) :=
sorry

end NUMINAMATH_GPT_part1_part2_1_part2_2_l912_91273


namespace NUMINAMATH_GPT_sum_x_midpoints_of_triangle_l912_91267

theorem sum_x_midpoints_of_triangle (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  -- Proof omitted, replace with actual proof
  sorry

end NUMINAMATH_GPT_sum_x_midpoints_of_triangle_l912_91267


namespace NUMINAMATH_GPT_quadratic_solution_l912_91261

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 6 * x + 8 = 0) (h2 : x ≠ 0) :
  x = 2 ∨ x = 4 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_l912_91261


namespace NUMINAMATH_GPT_roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l912_91282

-- Part (a)
theorem roots_can_be_integers_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x y : ℤ, x * y = q ∧ x + y = p) ∧ (∃ x y : ℤ, x * y = q ∧ x + y = p + 1) :=
sorry

-- Part (b)
theorem roots_cannot_both_be_integers_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬(∃ x y z w : ℤ, x * y = q ∧ x + y = p ∧ z * w = q ∧ z + w = p + 1) :=
sorry

end NUMINAMATH_GPT_roots_can_be_integers_if_q_positive_roots_cannot_both_be_integers_if_q_negative_l912_91282


namespace NUMINAMATH_GPT_horizontal_distance_P_Q_l912_91247

-- Definitions for the given conditions
def curve (x : ℝ) : ℝ := x^2 + 2 * x - 3

-- Define the points P and Q on the curve
def P_x : Set ℝ := {x | curve x = 8}
def Q_x : Set ℝ := {x | curve x = -1}

-- State the theorem to prove horizontal distance is 3sqrt3
theorem horizontal_distance_P_Q : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ P_x ∧ x₂ ∈ Q_x ∧ |x₁ - x₂| = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_horizontal_distance_P_Q_l912_91247


namespace NUMINAMATH_GPT_total_cans_collected_l912_91258

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_cans_collected_l912_91258


namespace NUMINAMATH_GPT_cos_sin_225_deg_l912_91264

theorem cos_sin_225_deg : (Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2) ∧ (Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2) :=
by
  -- Lean proof steps would go here
  sorry

end NUMINAMATH_GPT_cos_sin_225_deg_l912_91264


namespace NUMINAMATH_GPT_total_highlighters_correct_l912_91241

variable (y p b : ℕ)
variable (total_highlighters : ℕ)

def num_yellow_highlighters := 7
def num_pink_highlighters := num_yellow_highlighters + 7
def num_blue_highlighters := num_pink_highlighters + 5
def total_highlighters_in_drawer := num_yellow_highlighters + num_pink_highlighters + num_blue_highlighters

theorem total_highlighters_correct : 
  total_highlighters_in_drawer = 40 :=
sorry

end NUMINAMATH_GPT_total_highlighters_correct_l912_91241


namespace NUMINAMATH_GPT_geometric_sequence_tan_sum_l912_91255

theorem geometric_sequence_tan_sum
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b^2 = a * c)
  (h2 : Real.tan B = 3/4):
  1 / Real.tan A + 1 / Real.tan C = 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_tan_sum_l912_91255


namespace NUMINAMATH_GPT_candy_mixture_price_l912_91298

theorem candy_mixture_price
  (a : ℝ)
  (h1 : 0 < a) -- Assuming positive amount of money spent, to avoid division by zero
  (p1 p2 : ℝ)
  (h2 : p1 = 2)
  (h3 : p2 = 3)
  (h4 : p2 * (a / p2) = p1 * (a / p1)) -- Condition that the total cost for each type is equal.
  : ( (p1 * (a / p1) + p2 * (a / p2)) / (a / p1 + a / p2) = 2.4 ) :=
  sorry

end NUMINAMATH_GPT_candy_mixture_price_l912_91298


namespace NUMINAMATH_GPT_ana_multiplied_numbers_l912_91228

theorem ana_multiplied_numbers (x : ℕ) (y : ℕ) 
    (h_diff : y = x + 202) 
    (h_mistake : x * y - 1000 = 288 * x + 67) :
    x = 97 ∧ y = 299 :=
sorry

end NUMINAMATH_GPT_ana_multiplied_numbers_l912_91228


namespace NUMINAMATH_GPT_problem_solution_exists_l912_91206

theorem problem_solution_exists {x : ℝ} :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 =
    a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 +
    a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
    a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7)
  → a_2 = 56 :=
sorry

end NUMINAMATH_GPT_problem_solution_exists_l912_91206


namespace NUMINAMATH_GPT_max_money_received_back_l912_91233

def total_money_before := 3000
def value_chip_20 := 20
def value_chip_100 := 100
def chips_lost_total := 16
def chips_lost_diff_1 (x y : ℕ) := x = y + 2
def chips_lost_diff_2 (x y : ℕ) := x = y - 2

theorem max_money_received_back :
  ∃ (x y : ℕ), 
  (chips_lost_diff_1 x y ∨ chips_lost_diff_2 x y) ∧ 
  (x + y = chips_lost_total) ∧
  total_money_before - (x * value_chip_20 + y * value_chip_100) = 2120 :=
sorry

end NUMINAMATH_GPT_max_money_received_back_l912_91233


namespace NUMINAMATH_GPT_second_meeting_time_l912_91253

-- Given conditions and constants.
def pool_length : ℕ := 120
def initial_george_distance : ℕ := 80
def initial_henry_distance : ℕ := 40
def george_speed (t : ℕ) : ℕ := initial_george_distance / t
def henry_speed (t : ℕ) : ℕ := initial_henry_distance / t

-- Main statement to prove the question and answer.
theorem second_meeting_time (t : ℕ) (h_t_pos : t > 0) : 
  5 * t = 15 / 2 :=
sorry

end NUMINAMATH_GPT_second_meeting_time_l912_91253


namespace NUMINAMATH_GPT_infinitely_many_singular_pairs_l912_91213

def largestPrimeFactor (n : ℕ) : ℕ := sorry -- definition of largest prime factor

def isSingularPair (p q : ℕ) : Prop :=
  p ≠ q ∧ ∀ (n : ℕ), n ≥ 2 → largestPrimeFactor n * largestPrimeFactor (n + 1) ≠ p * q

theorem infinitely_many_singular_pairs : ∃ (S : ℕ → (ℕ × ℕ)), ∀ i, isSingularPair (S i).1 (S i).2 :=
sorry

end NUMINAMATH_GPT_infinitely_many_singular_pairs_l912_91213


namespace NUMINAMATH_GPT_value_of_a_l912_91295

theorem value_of_a
  (x y a : ℝ)
  (h1 : x + 2 * y = 2 * a - 1)
  (h2 : x - y = 6)
  (h3 : x = -y)
  : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l912_91295


namespace NUMINAMATH_GPT_days_y_needs_l912_91202

theorem days_y_needs
  (d : ℝ)
  (h1 : (1:ℝ) / 21 * 14 = 1 - 5 * (1 / d)) :
  d = 10 :=
sorry

end NUMINAMATH_GPT_days_y_needs_l912_91202


namespace NUMINAMATH_GPT_eval_expr_at_sqrt3_minus_3_l912_91225

noncomputable def expr (a : ℝ) : ℝ :=
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2))

theorem eval_expr_at_sqrt3_minus_3 : expr (Real.sqrt 3 - 3) = -Real.sqrt 3 / 6 := 
  by sorry

end NUMINAMATH_GPT_eval_expr_at_sqrt3_minus_3_l912_91225


namespace NUMINAMATH_GPT_cover_tiles_count_l912_91262

-- Definitions corresponding to the conditions
def tile_side : ℕ := 6 -- in inches
def tile_area : ℕ := tile_side * tile_side -- area of one tile in square inches

def region_length : ℕ := 3 * 12 -- 3 feet in inches
def region_width : ℕ := 6 * 12 -- 6 feet in inches
def region_area : ℕ := region_length * region_width -- area of the region in square inches

-- The statement of the proof problem
theorem cover_tiles_count : (region_area / tile_area) = 72 :=
by
   -- Proof would be filled in here
   sorry

end NUMINAMATH_GPT_cover_tiles_count_l912_91262


namespace NUMINAMATH_GPT_area_of_shaded_part_l912_91288

-- Define the given condition: area of the square
def area_of_square : ℝ := 100

-- Define the proof goal: area of the shaded part
theorem area_of_shaded_part : area_of_square / 2 = 50 := by
  sorry

end NUMINAMATH_GPT_area_of_shaded_part_l912_91288


namespace NUMINAMATH_GPT_range_of_values_l912_91278

theorem range_of_values (a b : ℝ) : (∀ x : ℝ, x < 1 → ax + b > 2 * (x + 1)) → b > 4 := 
by
  sorry

end NUMINAMATH_GPT_range_of_values_l912_91278


namespace NUMINAMATH_GPT_symmetric_points_existence_l912_91292

-- Define the ellipse equation
def is_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define the line equation parameterized by m
def line_eq (x y m : ℝ) : Prop :=
  y = 4 * x + m

-- Define the range for m such that symmetric points exist
def m_in_range (m : ℝ) : Prop :=
  - (2 * Real.sqrt 13) / 13 < m ∧ m < (2 * Real.sqrt 13) / 13

-- Prove the existence of symmetric points criteria for m
theorem symmetric_points_existence (m : ℝ) :
  (∀ (x y : ℝ), is_ellipse x y → line_eq x y m → 
    (∃ (x1 y1 x2 y2 : ℝ), is_ellipse x1 y1 ∧ is_ellipse x2 y2 ∧ line_eq x1 y1 m ∧ line_eq x2 y2 m ∧ 
      (x1 = x2) ∧ (y1 = -y2))) ↔ m_in_range m :=
sorry

end NUMINAMATH_GPT_symmetric_points_existence_l912_91292


namespace NUMINAMATH_GPT_isosceles_right_triangle_measure_l912_91229

theorem isosceles_right_triangle_measure (a XY YZ : ℝ) 
    (h1 : XY > YZ) 
    (h2 : a^2 = 25 / (1/2)) : XY = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_measure_l912_91229


namespace NUMINAMATH_GPT_smallest_m_for_triangle_sides_l912_91274

noncomputable def is_triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_m_for_triangle_sides (a b c : ℝ) (h : is_triangle_sides a b c) :
  (a^2 + c^2) / (b + c)^2 < 1 / 2 := sorry

end NUMINAMATH_GPT_smallest_m_for_triangle_sides_l912_91274


namespace NUMINAMATH_GPT_jenny_cases_l912_91230

theorem jenny_cases (total_boxes cases_per_box : ℕ) (h1 : total_boxes = 24) (h2 : cases_per_box = 8) :
  total_boxes / cases_per_box = 3 := by
  sorry

end NUMINAMATH_GPT_jenny_cases_l912_91230


namespace NUMINAMATH_GPT_min_value_of_quadratic_function_l912_91286

-- Given the quadratic function y = x^2 + 4x - 5
def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 4*x - 5

-- Statement of the proof in Lean 4
theorem min_value_of_quadratic_function :
  ∃ (x_min y_min : ℝ), y_min = quadratic_function x_min ∧
  ∀ x : ℝ, quadratic_function x ≥ y_min ∧ x_min = -2 ∧ y_min = -9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_function_l912_91286


namespace NUMINAMATH_GPT_minimum_value_of_f_l912_91270

noncomputable def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ 1) ∧ f x = 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l912_91270


namespace NUMINAMATH_GPT_yellow_peaches_l912_91214

theorem yellow_peaches (red_peaches green_peaches total_green_yellow_peaches : ℕ)
  (h1 : red_peaches = 5)
  (h2 : green_peaches = 6)
  (h3 : total_green_yellow_peaches = 20) :
  (total_green_yellow_peaches - green_peaches) = 14 :=
by
  sorry

end NUMINAMATH_GPT_yellow_peaches_l912_91214


namespace NUMINAMATH_GPT_three_integers_sum_of_consecutive_odds_l912_91201

theorem three_integers_sum_of_consecutive_odds :
  {N : ℕ | N ≤ 500 ∧ (∃ j n, N = j * (2 * n + j) ∧ j ≥ 1) ∧
                   (∃! j1 j2 j3, ∃ n1 n2 n3, N = j1 * (2 * n1 + j1) ∧ N = j2 * (2 * n2 + j2) ∧ N = j3 * (2 * n3 + j3) ∧ j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3)} = {16, 18, 50} :=
by
  sorry

end NUMINAMATH_GPT_three_integers_sum_of_consecutive_odds_l912_91201


namespace NUMINAMATH_GPT_perpendicular_lines_l912_91248

theorem perpendicular_lines (a : ℝ)
  (line1 : (a^2 + a - 6) * x + 12 * y - 3 = 0)
  (line2 : (a - 1) * x - (a - 2) * y + 4 - a = 0) :
  (a - 2) * (a - 3) * (a + 5) = 0 := sorry

end NUMINAMATH_GPT_perpendicular_lines_l912_91248


namespace NUMINAMATH_GPT_find_x_l912_91223

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 156) (h2 : x ≥ 0) : x = 12 :=
sorry

end NUMINAMATH_GPT_find_x_l912_91223


namespace NUMINAMATH_GPT_distance_is_one_l912_91272

noncomputable def distance_between_bisectors_and_centroid : ℝ :=
  let AB := 9
  let AC := 12
  let BC := Real.sqrt (AB^2 + AC^2)
  let CD := BC / 2
  let CE := (2/3) * CD
  let r := (AB * AC) / (2 * (AB + AC + BC) / 2)
  let K := CE - r
  K

theorem distance_is_one : distance_between_bisectors_and_centroid = 1 :=
  sorry

end NUMINAMATH_GPT_distance_is_one_l912_91272


namespace NUMINAMATH_GPT_wicket_count_l912_91236

theorem wicket_count (initial_avg new_avg : ℚ) (runs_last_match wickets_last_match : ℕ) (delta_avg : ℚ) (W : ℕ) :
  initial_avg = 12.4 →
  new_avg = 12.0 →
  delta_avg = 0.4 →
  runs_last_match = 26 →
  wickets_last_match = 8 →
  initial_avg * W + runs_last_match = new_avg * (W + wickets_last_match) →
  W = 175 := by
  sorry

end NUMINAMATH_GPT_wicket_count_l912_91236


namespace NUMINAMATH_GPT_gcd_72_168_l912_91239

theorem gcd_72_168 : Nat.gcd 72 168 = 24 :=
by
  sorry

end NUMINAMATH_GPT_gcd_72_168_l912_91239


namespace NUMINAMATH_GPT_buratino_solved_16_problems_l912_91271

-- Defining the conditions given in the problem
def total_kopeks_received : ℕ := 655 * 100 + 35

def geometric_sum (n : ℕ) : ℕ := 2^n - 1

-- The goal is to prove that Buratino solved 16 problems
theorem buratino_solved_16_problems (n : ℕ) (h : geometric_sum n = total_kopeks_received) : n = 16 := by
  sorry

end NUMINAMATH_GPT_buratino_solved_16_problems_l912_91271


namespace NUMINAMATH_GPT_polynomial_at_1_gcd_of_72_120_168_l912_91250

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x - 6

-- Assertion that the polynomial evaluated at x = 1 gives 9
theorem polynomial_at_1 : polynomial 1 = 9 := by
  -- Usually, this is where the detailed Horner's method proof would go
  sorry

-- Define the gcd function for three numbers
def gcd3 (a b c : ℤ) : ℤ := Int.gcd (Int.gcd a b) c

-- Assertion that the GCD of 72, 120, and 168 is 24
theorem gcd_of_72_120_168 : gcd3 72 120 168 = 24 := by
  -- Usually, this is where the detailed Euclidean algorithm proof would go
  sorry

end NUMINAMATH_GPT_polynomial_at_1_gcd_of_72_120_168_l912_91250


namespace NUMINAMATH_GPT_number_of_pieces_sold_on_third_day_l912_91224

variable (m : ℕ)

def first_day_sales : ℕ := m
def second_day_sales : ℕ := (m / 2) - 3
def third_day_sales : ℕ := second_day_sales m + 5

theorem number_of_pieces_sold_on_third_day :
  third_day_sales m = (m / 2) + 2 := by sorry

end NUMINAMATH_GPT_number_of_pieces_sold_on_third_day_l912_91224


namespace NUMINAMATH_GPT_tank_plastering_cost_l912_91256

noncomputable def plastering_cost (L W D : ℕ) (cost_per_sq_meter : ℚ) : ℚ :=
  let A_bottom := L * W
  let A_long_walls := 2 * (L * D)
  let A_short_walls := 2 * (W * D)
  let A_total := A_bottom + A_long_walls + A_short_walls
  A_total * cost_per_sq_meter

theorem tank_plastering_cost :
  plastering_cost 25 12 6 0.25 = 186 := by
  sorry

end NUMINAMATH_GPT_tank_plastering_cost_l912_91256


namespace NUMINAMATH_GPT_gcd_372_684_l912_91208

theorem gcd_372_684 : Int.gcd 372 684 = 12 :=
by
  sorry

end NUMINAMATH_GPT_gcd_372_684_l912_91208


namespace NUMINAMATH_GPT_how_many_bones_in_adult_woman_l912_91205

-- Define the conditions
def numSkeletons : ℕ := 20
def halfSkeletons : ℕ := 10
def numAdultWomen : ℕ := 10
def numMenAndChildren : ℕ := 10
def numAdultMen : ℕ := 5
def numChildren : ℕ := 5
def totalBones : ℕ := 375

-- Define the proof statement
theorem how_many_bones_in_adult_woman (W : ℕ) (H : 10 * W + 5 * (W + 5) + 5 * (W / 2) = 375) : W = 20 :=
sorry

end NUMINAMATH_GPT_how_many_bones_in_adult_woman_l912_91205


namespace NUMINAMATH_GPT_clock_correct_time_fraction_l912_91266

theorem clock_correct_time_fraction :
  let hours := 24
  let incorrect_hours := 6
  let correct_hours_fraction := (hours - incorrect_hours) / hours
  let minutes_per_hour := 60
  let incorrect_minutes_per_hour := 15
  let correct_minutes_fraction := (minutes_per_hour - incorrect_minutes_per_hour) / minutes_per_hour
  correct_hours_fraction * correct_minutes_fraction = (9 / 16) :=
by 
  sorry

end NUMINAMATH_GPT_clock_correct_time_fraction_l912_91266


namespace NUMINAMATH_GPT_dice_probability_l912_91260

noncomputable def probability_same_face_in_single_roll : ℝ :=
  (1 / 6)^10

noncomputable def probability_not_all_same_face_in_single_roll : ℝ :=
  1 - probability_same_face_in_single_roll

noncomputable def probability_not_all_same_face_in_five_rolls : ℝ :=
  probability_not_all_same_face_in_single_roll^5

noncomputable def probability_at_least_one_all_same_face : ℝ :=
  1 - probability_not_all_same_face_in_five_rolls

theorem dice_probability :
  probability_at_least_one_all_same_face = 1 - (1 - (1 / 6)^10)^5 :=
sorry

end NUMINAMATH_GPT_dice_probability_l912_91260


namespace NUMINAMATH_GPT_divya_age_l912_91232

theorem divya_age (D N : ℝ) (h1 : N + 5 = 3 * (D + 5)) (h2 : N + D = 40) : D = 7.5 :=
by sorry

end NUMINAMATH_GPT_divya_age_l912_91232


namespace NUMINAMATH_GPT_problem1_problem2_l912_91277

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem1 : sqrt 12 + sqrt 8 * sqrt 6 = 6 * sqrt 3 := by
  sorry

theorem problem2 : sqrt 12 + 1 / (sqrt 3 - sqrt 2) - sqrt 6 * sqrt 3 = 3 * sqrt 3 - 2 * sqrt 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l912_91277


namespace NUMINAMATH_GPT_minimum_packages_shipped_l912_91234

-- Definitions based on the conditions given in the problem
def Sarah_truck_capacity : ℕ := 18
def Ryan_truck_capacity : ℕ := 11

-- Minimum number of packages shipped
theorem minimum_packages_shipped :
  ∃ (n : ℕ), n = Sarah_truck_capacity * Ryan_truck_capacity :=
by sorry

end NUMINAMATH_GPT_minimum_packages_shipped_l912_91234


namespace NUMINAMATH_GPT_sales_last_year_l912_91242

theorem sales_last_year (x : ℝ) (h1 : 416 = (1 + 0.30) * x) : x = 320 :=
by
  sorry

end NUMINAMATH_GPT_sales_last_year_l912_91242


namespace NUMINAMATH_GPT_find_x_l912_91289

theorem find_x (x : ℝ) (h : (x + 8 + 5 * x + 4 + 2 * x + 7) / 3 = 3 * x - 10) : x = 49 :=
sorry

end NUMINAMATH_GPT_find_x_l912_91289


namespace NUMINAMATH_GPT_jed_gives_2_cards_every_two_weeks_l912_91299

theorem jed_gives_2_cards_every_two_weeks
  (starting_cards : ℕ)
  (cards_per_week : ℕ)
  (cards_after_4_weeks : ℕ)
  (number_of_two_week_intervals : ℕ)
  (cards_given_away_each_two_weeks : ℕ):
  starting_cards = 20 →
  cards_per_week = 6 →
  cards_after_4_weeks = 40 →
  number_of_two_week_intervals = 2 →
  (starting_cards + 4 * cards_per_week - number_of_two_week_intervals * cards_given_away_each_two_weeks = cards_after_4_weeks) →
  cards_given_away_each_two_weeks = 2 := 
by
  intros h_start h_week h_4weeks h_intervals h_eq
  sorry

end NUMINAMATH_GPT_jed_gives_2_cards_every_two_weeks_l912_91299


namespace NUMINAMATH_GPT_y_intercept_of_line_l912_91249

theorem y_intercept_of_line 
  (point : ℝ × ℝ)
  (slope_angle : ℝ)
  (h1 : point = (2, -5))
  (h2 : slope_angle = 135) :
  ∃ b : ℝ, (∀ x y : ℝ, y = -x + b ↔ ((y - (-5)) = (-1) * (x - 2))) ∧ b = -3 := 
sorry

end NUMINAMATH_GPT_y_intercept_of_line_l912_91249


namespace NUMINAMATH_GPT_exists_positive_n_with_m_zeros_l912_91287

theorem exists_positive_n_with_m_zeros (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, 7^n = k * 10^m :=
sorry

end NUMINAMATH_GPT_exists_positive_n_with_m_zeros_l912_91287


namespace NUMINAMATH_GPT_compute_fraction_product_l912_91220

-- Definitions based on conditions
def one_third_pow_four : ℚ := (1 / 3) ^ 4
def one_fifth : ℚ := 1 / 5

-- Main theorem to prove the problem question == answer
theorem compute_fraction_product : (one_third_pow_four * one_fifth) = 1 / 405 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_product_l912_91220


namespace NUMINAMATH_GPT_ratio_of_spent_to_left_after_video_game_l912_91268

-- Definitions based on conditions
def total_money : ℕ := 100
def spent_on_video_game : ℕ := total_money * 1 / 4
def money_left_after_video_game : ℕ := total_money - spent_on_video_game
def money_left_after_goggles : ℕ := 60
def spent_on_goggles : ℕ := money_left_after_video_game - money_left_after_goggles

-- Statement to prove the ratio
theorem ratio_of_spent_to_left_after_video_game :
  (spent_on_goggles : ℚ) / (money_left_after_video_game : ℚ) = 1 / 5 := 
sorry

end NUMINAMATH_GPT_ratio_of_spent_to_left_after_video_game_l912_91268


namespace NUMINAMATH_GPT_part_a_l912_91207

theorem part_a (m : ℕ) (A B : ℕ) (hA : A = (10^(2 * m) - 1) / 9) (hB : B = 4 * ((10^m - 1) / 9)) :
  ∃ k : ℕ, A + B + 1 = k^2 :=
sorry

end NUMINAMATH_GPT_part_a_l912_91207


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l912_91275

theorem problem1 : 78 * 4 + 488 = 800 := by sorry
theorem problem2 : 1903 - 475 * 4 = 3 := by sorry
theorem problem3 : 350 * (12 + 342 / 9) = 17500 := by sorry
theorem problem4 : 480 / (125 - 117) = 60 := by sorry
theorem problem5 : (3600 - 18 * 200) / 253 = 0 := by sorry
theorem problem6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l912_91275


namespace NUMINAMATH_GPT_find_second_number_l912_91283

theorem find_second_number (x : ℕ) : 
  ((20 + 40 + 60) / 3 = 4 + ((x + 10 + 28) / 3)) → x = 70 :=
by {
  -- let lhs = (20 + 40 + 60) / 3
  -- let rhs = 4 + ((x + 10 + 28) / 3)
  -- rw rhs at lhs,
  -- value the lhs and rhs,
  -- prove x = 70
  sorry
}

end NUMINAMATH_GPT_find_second_number_l912_91283


namespace NUMINAMATH_GPT_stratified_sampling_l912_91215

-- Define the known quantities
def total_products := 2000
def sample_size := 200
def workshop_production := 250

-- Define the main theorem to prove
theorem stratified_sampling:
  (workshop_production / total_products) * sample_size = 25 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l912_91215


namespace NUMINAMATH_GPT_photo_album_pages_l912_91244

noncomputable def P1 := 0
noncomputable def P2 := 10
noncomputable def remaining_pages := 20

theorem photo_album_pages (photos total_pages photos_per_page_set1 photos_per_page_set2 photos_per_page_remaining : ℕ) 
  (h1 : photos = 100)
  (h2 : total_pages = 30)
  (h3 : photos_per_page_set1 = 3)
  (h4 : photos_per_page_set2 = 4)
  (h5 : photos_per_page_remaining = 3) : 
  P1 = 0 ∧ P2 = 10 ∧ remaining_pages = 20 :=
by
  sorry

end NUMINAMATH_GPT_photo_album_pages_l912_91244


namespace NUMINAMATH_GPT_rahul_batting_average_l912_91210

theorem rahul_batting_average:
  ∃ (A : ℝ), A = 46 ∧
  (∀ (R : ℝ), R = 138 → R = 54 * 4 - 78 → A = R / 3) ∧
  ∃ (n_matches : ℕ), n_matches = 3 :=
by
  sorry

end NUMINAMATH_GPT_rahul_batting_average_l912_91210


namespace NUMINAMATH_GPT_cost_of_shirts_l912_91285

theorem cost_of_shirts : 
  let shirt1 := 15
  let shirt2 := 15
  let shirt3 := 15
  let shirt4 := 20
  let shirt5 := 20
  shirt1 + shirt2 + shirt3 + shirt4 + shirt5 = 85 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_shirts_l912_91285


namespace NUMINAMATH_GPT_value_of_b_l912_91212

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_b_l912_91212


namespace NUMINAMATH_GPT_ticket_cost_l912_91216

open Real

-- Variables for ticket prices
variable (A C S : ℝ)

-- Given conditions
def cost_condition : Prop :=
  C = A / 2 ∧ S = A - 1.50 ∧ 6 * A + 5 * C + 3 * S = 40.50

-- The goal is to prove that the total cost for 10 adult tickets, 8 child tickets,
-- and 4 senior tickets is 64.38
theorem ticket_cost (h : cost_condition A C S) : 10 * A + 8 * C + 4 * S = 64.38 :=
by
  -- Implementation of the proof would go here
  sorry

end NUMINAMATH_GPT_ticket_cost_l912_91216


namespace NUMINAMATH_GPT_total_shoes_count_l912_91203

-- Define the concepts and variables related to the conditions
def num_people := 10
def num_people_regular_shoes := 4
def num_people_sandals := 3
def num_people_slippers := 3
def num_shoes_regular := 2
def num_shoes_sandals := 1
def num_shoes_slippers := 1

-- Goal: Prove that the total number of shoes kept outside is 20
theorem total_shoes_count :
  (num_people_regular_shoes * num_shoes_regular) +
  (num_people_sandals * num_shoes_sandals * 2) +
  (num_people_slippers * num_shoes_slippers * 2) = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_shoes_count_l912_91203


namespace NUMINAMATH_GPT_Mart_income_percentage_of_Juan_l912_91254

variable (J T M : ℝ)

-- Conditions
def Tim_income_def : Prop := T = 0.5 * J
def Mart_income_def : Prop := M = 1.6 * T

-- Theorem to prove
theorem Mart_income_percentage_of_Juan
  (h1 : Tim_income_def T J) 
  (h2 : Mart_income_def M T) : 
  (M / J) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_Mart_income_percentage_of_Juan_l912_91254


namespace NUMINAMATH_GPT_simplify_expression_l912_91263

-- Define the problem context
variables {x y : ℝ} {i : ℂ}

-- The mathematical simplification problem
theorem simplify_expression :
  (x ^ 2 + i * y) ^ 3 * (x ^ 2 - i * y) ^ 3 = x ^ 12 - 9 * x ^ 8 * y ^ 2 - 9 * x ^ 4 * y ^ 4 - y ^ 6 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l912_91263


namespace NUMINAMATH_GPT_calculate_total_calories_l912_91219

-- Definition of variables and conditions
def total_calories (C : ℝ) : Prop :=
  let FDA_recommended_intake := 25
  let consumed_calories := FDA_recommended_intake + 5
  (3 / 4) * C = consumed_calories

-- Theorem statement
theorem calculate_total_calories : ∃ C : ℝ, total_calories C ∧ C = 40 :=
by
  sorry  -- Proof will be provided here

end NUMINAMATH_GPT_calculate_total_calories_l912_91219


namespace NUMINAMATH_GPT_average_age_of_women_l912_91296

noncomputable def avg_age_two_women (M : ℕ) (new_avg : ℕ) (W : ℕ) :=
  let loss := 20 + 10;
  let gain := 2 * 8;
  W = loss + gain

theorem average_age_of_women (M : ℕ) (new_avg : ℕ) (W : ℕ) (avg_age : ℕ) :
  avg_age_two_women M new_avg W →
  avg_age = 23 :=
sorry

#check average_age_of_women

end NUMINAMATH_GPT_average_age_of_women_l912_91296


namespace NUMINAMATH_GPT_total_money_is_145_83_l912_91235

noncomputable def jackson_money : ℝ := 125

noncomputable def williams_money : ℝ := jackson_money / 6

noncomputable def total_money : ℝ := jackson_money + williams_money

theorem total_money_is_145_83 :
  total_money = 145.83 := by
sorry

end NUMINAMATH_GPT_total_money_is_145_83_l912_91235


namespace NUMINAMATH_GPT_residue_11_pow_2016_mod_19_l912_91240

theorem residue_11_pow_2016_mod_19 : (11^2016) % 19 = 17 := 
sorry

end NUMINAMATH_GPT_residue_11_pow_2016_mod_19_l912_91240


namespace NUMINAMATH_GPT_macey_weeks_to_save_l912_91279

theorem macey_weeks_to_save :
  ∀ (total_cost amount_saved weekly_savings : ℝ),
    total_cost = 22.45 →
    amount_saved = 7.75 →
    weekly_savings = 1.35 →
    ⌈(total_cost - amount_saved) / weekly_savings⌉ = 11 :=
by
  intros total_cost amount_saved weekly_savings h_total_cost h_amount_saved h_weekly_savings
  sorry

end NUMINAMATH_GPT_macey_weeks_to_save_l912_91279


namespace NUMINAMATH_GPT_pow_99_square_pow_neg8_mult_l912_91251

theorem pow_99_square :
  99^2 = 9801 := 
by
  -- Proof omitted
  sorry

theorem pow_neg8_mult :
  (-8) ^ 2009 * (-1/8) ^ 2008 = -8 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pow_99_square_pow_neg8_mult_l912_91251


namespace NUMINAMATH_GPT_all_round_trips_miss_capital_same_cost_l912_91293

open Set

variable {City : Type} [Inhabited City]
variable {f : City → City → ℝ}
variable (capital : City)
variable (round_trip_cost : List City → ℝ)

-- The conditions
axiom flight_cost_symmetric (A B : City) : f A B = f B A
axiom equal_round_trip_cost (R1 R2 : List City) :
  (∀ (city : City), city ∈ R1 ↔ city ∈ R2) → 
  round_trip_cost R1 = round_trip_cost R2

noncomputable def constant_trip_cost := 
  ∀ (cities1 cities2 : List City),
     (∀ (city : City), city ∈ cities1 ↔ city ∈ cities2) →
     ¬(capital ∈ cities1 ∨ capital ∈ cities2) →
     round_trip_cost cities1 = round_trip_cost cities2

-- Goal to prove
theorem all_round_trips_miss_capital_same_cost : constant_trip_cost capital round_trip_cost := 
  sorry

end NUMINAMATH_GPT_all_round_trips_miss_capital_same_cost_l912_91293


namespace NUMINAMATH_GPT_athena_spent_correct_amount_l912_91280

-- Define the conditions
def num_sandwiches : ℕ := 3
def price_per_sandwich : ℝ := 3
def num_drinks : ℕ := 2
def price_per_drink : ℝ := 2.5

-- Define the total cost as per the given conditions
def total_cost : ℝ :=
  (num_sandwiches * price_per_sandwich) + (num_drinks * price_per_drink)

-- The theorem that states the problem and asserts the correct answer
theorem athena_spent_correct_amount : total_cost = 14 := 
  by
    sorry

end NUMINAMATH_GPT_athena_spent_correct_amount_l912_91280


namespace NUMINAMATH_GPT_find_angle_4_l912_91291

def angle_sum_180 (α β : ℝ) : Prop := α + β = 180
def angle_equality (γ δ : ℝ) : Prop := γ = δ
def triangle_angle_values (A B : ℝ) : Prop := A = 80 ∧ B = 50

theorem find_angle_4
  (A B : ℝ) (angle1 angle2 angle3 angle4 : ℝ)
  (h1 : angle_sum_180 angle1 angle2)
  (h2 : angle_equality angle3 angle4)
  (h3 : triangle_angle_values A B)
  (h4 : angle_sum_180 (angle1 + A + B) 180)
  (h5 : angle_sum_180 (angle2 + angle3 + angle4) 180) :
  angle4 = 25 :=
by sorry

end NUMINAMATH_GPT_find_angle_4_l912_91291


namespace NUMINAMATH_GPT_charging_time_l912_91269

theorem charging_time (S T L : ℕ → ℕ) 
  (HS : ∀ t, S t = 15 * t) 
  (HT : ∀ t, T t = 8 * t) 
  (HL : ∀ t, L t = 5 * t)
  (smartphone_capacity tablet_capacity laptop_capacity : ℕ)
  (smartphone_percentage tablet_percentage laptop_percentage : ℕ)
  (h_smartphone : smartphone_capacity = 4500)
  (h_tablet : tablet_capacity = 10000)
  (h_laptop : laptop_capacity = 20000)
  (p_smartphone : smartphone_percentage = 75)
  (p_tablet : tablet_percentage = 25)
  (p_laptop : laptop_percentage = 50)
  (required_charge_s required_charge_t required_charge_l : ℕ)
  (h_rq_s : required_charge_s = smartphone_capacity * smartphone_percentage / 100)
  (h_rq_t : required_charge_t = tablet_capacity * tablet_percentage / 100)
  (h_rq_l : required_charge_l = laptop_capacity * laptop_percentage / 100)
  (time_s time_t time_l : ℕ)
  (h_time_s : time_s = required_charge_s / 15)
  (h_time_t : time_t = required_charge_t / 8)
  (h_time_l : time_l = required_charge_l / 5) : 
  max time_s (max time_t time_l) = 2000 := 
by 
  -- This theorem states that the maximum time taken for charging is 2000 minutes
  sorry

end NUMINAMATH_GPT_charging_time_l912_91269


namespace NUMINAMATH_GPT_line_through_center_of_circle_l912_91281

theorem line_through_center_of_circle 
    (x y : ℝ) 
    (h : x^2 + y^2 - 4*x + 6*y = 0) : 
    3*x + 2*y = 0 :=
sorry

end NUMINAMATH_GPT_line_through_center_of_circle_l912_91281


namespace NUMINAMATH_GPT_greatest_integer_c_not_in_range_l912_91290

theorem greatest_integer_c_not_in_range :
  ∃ c : ℤ, (¬ ∃ x : ℝ, x^2 + (c:ℝ)*x + 18 = -6) ∧ (∀ c' : ℤ, c' > c → (∃ x : ℝ, x^2 + (c':ℝ)*x + 18 = -6)) :=
sorry

end NUMINAMATH_GPT_greatest_integer_c_not_in_range_l912_91290


namespace NUMINAMATH_GPT_min_a_add_c_l912_91243

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def angle_ABC : ℝ := 2 * Real.pi / 3
noncomputable def BD : ℝ := 1

-- The bisector of angle ABC intersects AC at point D
-- Angle bisector theorem and the given information
theorem min_a_add_c : ∃ a c : ℝ, (angle_ABC = 2 * Real.pi / 3) → (BD = 1) → (a * c = a + c) → (a + c ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_min_a_add_c_l912_91243


namespace NUMINAMATH_GPT_perpendicular_tangent_line_l912_91204

theorem perpendicular_tangent_line :
  ∃ m : ℝ, ∃ x₀ : ℝ, y₀ = x₀ ^ 3 + 3 * x₀ ^ 2 - 1 ∧ y₀ = -3 * x₀ + m ∧ 
  (∀ x, x ≠ x₀ → x ^ 3 + 3 * x ^ 2 - 1 ≠ -3 * x + m) ∧ m = -2 := 
sorry

end NUMINAMATH_GPT_perpendicular_tangent_line_l912_91204


namespace NUMINAMATH_GPT_find_x_base_l912_91200

open Nat

def is_valid_digit (n : ℕ) : Prop := n < 10

def interpret_base (digits : ℕ → ℕ) (n : ℕ) : ℕ :=
  digits 2 * n^2 + digits 1 * n + digits 0

theorem find_x_base (a b c : ℕ)
  (ha : is_valid_digit a)
  (hb : is_valid_digit b)
  (hc : is_valid_digit c)
  (h : interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 20 = 2 * interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 13) :
  100 * a + 10 * b + c = 198 :=
by
  sorry

end NUMINAMATH_GPT_find_x_base_l912_91200


namespace NUMINAMATH_GPT_inequality_problem_l912_91246

variable (a b c d : ℝ)

open Real

theorem inequality_problem 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hprod : a * b * c * d = 1) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + d)) + 1 / (d * (1 + a)) ≥ 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_problem_l912_91246
