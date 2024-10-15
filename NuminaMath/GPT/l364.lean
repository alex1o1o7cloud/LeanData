import Mathlib

namespace NUMINAMATH_GPT_inverse_proportion_value_of_m_l364_36483

theorem inverse_proportion_value_of_m (m : ℤ) (x : ℝ) (y : ℝ) : 
  y = (m - 2) * x ^ (m^2 - 5) → (m = -2) := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_value_of_m_l364_36483


namespace NUMINAMATH_GPT_max_bag_weight_is_50_l364_36497

noncomputable def max_weight_allowed (people bags_per_person more_bags_allowed total_weight : ℕ) : ℝ := 
  total_weight / ((people * bags_per_person) + more_bags_allowed)

theorem max_bag_weight_is_50 : ∀ (people bags_per_person more_bags_allowed total_weight : ℕ), 
  people = 6 → 
  bags_per_person = 5 → 
  more_bags_allowed = 90 → 
  total_weight = 6000 →
  max_weight_allowed people bags_per_person more_bags_allowed total_weight = 50 := 
by 
  sorry

end NUMINAMATH_GPT_max_bag_weight_is_50_l364_36497


namespace NUMINAMATH_GPT_inverse_variation_example_l364_36449

theorem inverse_variation_example
  (k : ℝ)
  (h1 : ∀ (c d : ℝ), (c^2) * (d^4) = k)
  (h2 : ∃ (c : ℝ), c = 8 ∧ (∀ (d : ℝ), d = 2 → (c^2) * (d^4) = k)) : 
  (∀ (d : ℝ), d = 4 → (∃ (c : ℝ), (c^2) = 4)) := 
by 
  sorry

end NUMINAMATH_GPT_inverse_variation_example_l364_36449


namespace NUMINAMATH_GPT_length_of_c_l364_36464

theorem length_of_c (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h_triangle : 0 < c) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) → c = 3 :=
by
  intros h_ineq
  sorry

end NUMINAMATH_GPT_length_of_c_l364_36464


namespace NUMINAMATH_GPT_lcm_153_180_560_l364_36486

theorem lcm_153_180_560 : Nat.lcm (Nat.lcm 153 180) 560 = 85680 :=
by
  sorry

end NUMINAMATH_GPT_lcm_153_180_560_l364_36486


namespace NUMINAMATH_GPT_sum_of_cubes_l364_36499

theorem sum_of_cubes (a b : ℕ) (h1 : 2 * x = a) (h2 : 3 * x = b) (h3 : b - a = 3) : a^3 + b^3 = 945 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l364_36499


namespace NUMINAMATH_GPT_kids_went_home_l364_36463

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℝ) (went_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : remaining_kids = 8.0) : went_home = 14.0 :=
by 
  sorry

end NUMINAMATH_GPT_kids_went_home_l364_36463


namespace NUMINAMATH_GPT_gasoline_price_percentage_increase_l364_36406

theorem gasoline_price_percentage_increase 
  (price_month1_euros : ℝ) (price_month3_dollars : ℝ) (exchange_rate : ℝ) 
  (price_month1 : ℝ) (percent_increase : ℝ):
  price_month1_euros = 20 →
  price_month3_dollars = 15 →
  exchange_rate = 1.2 →
  price_month1 = price_month1_euros * exchange_rate →
  percent_increase = ((price_month1 - price_month3_dollars) / price_month3_dollars) * 100 →
  percent_increase = 60 :=
by intros; sorry

end NUMINAMATH_GPT_gasoline_price_percentage_increase_l364_36406


namespace NUMINAMATH_GPT_firecracker_confiscation_l364_36428

variables
  (F : ℕ)   -- Total number of firecrackers bought
  (R : ℕ)   -- Number of firecrackers remaining after confiscation
  (D : ℕ)   -- Number of defective firecrackers
  (G : ℕ)   -- Number of good firecrackers before setting off half
  (C : ℕ)   -- Number of firecrackers confiscated

-- Define the conditions:
def conditions := 
  F = 48 ∧
  D = R / 6 ∧
  G = 2 * 15 ∧
  R - D = G ∧
  F - R = C

-- The theorem to prove:
theorem firecracker_confiscation (h : conditions F R D G C) : C = 12 := 
  sorry

end NUMINAMATH_GPT_firecracker_confiscation_l364_36428


namespace NUMINAMATH_GPT_equation_solution_system_of_inequalities_solution_l364_36421

theorem equation_solution (x : ℝ) : (3 / (x - 1) = 1 / (2 * x + 3)) ↔ (x = -2) :=
by
  sorry

theorem system_of_inequalities_solution (x : ℝ) : ((3 * x - 1 ≥ x + 1) ∧ (x + 3 > 4 * x - 2)) ↔ (1 ≤ x ∧ x < 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_system_of_inequalities_solution_l364_36421


namespace NUMINAMATH_GPT_proof_area_of_squares_l364_36404

noncomputable def area_of_squares : Prop :=
  let side_C := 48
  let side_D := 60
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  (area_C / area_D = (16 / 25)) ∧ 
  ((area_D - area_C) / area_C = (36 / 100))

theorem proof_area_of_squares : area_of_squares := sorry

end NUMINAMATH_GPT_proof_area_of_squares_l364_36404


namespace NUMINAMATH_GPT_total_pages_in_book_l364_36417

/-- Bill started reading a book on the first day of April. 
    He read 8 pages every day and by the 12th of April, he 
    had covered two-thirds of the book. Prove that the 
    total number of pages in the book is 144. --/
theorem total_pages_in_book 
  (pages_per_day : ℕ)
  (days_till_april_12 : ℕ)
  (total_pages_read : ℕ)
  (fraction_of_book_read : ℚ)
  (total_pages : ℕ)
  (h1 : pages_per_day = 8)
  (h2 : days_till_april_12 = 12)
  (h3 : total_pages_read = pages_per_day * days_till_april_12)
  (h4 : fraction_of_book_read = 2/3)
  (h5 : total_pages_read = (fraction_of_book_read * total_pages)) :
  total_pages = 144 := by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l364_36417


namespace NUMINAMATH_GPT_length_of_square_side_l364_36475

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem length_of_square_side
  (time_seconds : ℝ)
  (speed_km_per_hr : ℝ)
  (distance_m : ℝ)
  (side_length : ℝ)
  (h1 : time_seconds = 72)
  (h2 : speed_km_per_hr = 10)
  (h3 : distance_m = speed_km_per_hr_to_m_per_s speed_km_per_hr * time_seconds)
  (h4 : distance_m = perimeter_of_square side_length) :
  side_length = 50 :=
sorry

end NUMINAMATH_GPT_length_of_square_side_l364_36475


namespace NUMINAMATH_GPT_range_of_a_l364_36401

theorem range_of_a 
  (a : ℕ) 
  (an : ℕ → ℕ)
  (Sn : ℕ → ℕ)
  (h1 : a_1 = a)
  (h2 : ∀ n : ℕ, n ≥ 2 → Sn n + Sn (n - 1) = 4 * n^2)
  (h3 : ∀ n : ℕ, an n < an (n + 1)) : 
  3 < a ∧ a < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l364_36401


namespace NUMINAMATH_GPT_number_of_workers_in_second_group_l364_36471

theorem number_of_workers_in_second_group (w₁ w₂ d₁ d₂ : ℕ) (total_wages₁ total_wages₂ : ℝ) (daily_wage : ℝ) :
  w₁ = 15 ∧ d₁ = 6 ∧ total_wages₁ = 9450 ∧ 
  w₂ * d₂ * daily_wage = total_wages₂ ∧ d₂ = 5 ∧ total_wages₂ = 9975 ∧ 
  daily_wage = 105 
  → w₂ = 19 :=
by
  sorry

end NUMINAMATH_GPT_number_of_workers_in_second_group_l364_36471


namespace NUMINAMATH_GPT_distinct_xy_values_l364_36470

theorem distinct_xy_values : ∃ (xy_values : Finset ℕ), 
  (∀ (x y : ℕ), (0 < x ∧ 0 < y) → (1 / Real.sqrt x + 1 / Real.sqrt y = 1 / Real.sqrt 20) → (xy_values = {8100, 6400})) ∧
  (xy_values.card = 2) :=
by
  sorry

end NUMINAMATH_GPT_distinct_xy_values_l364_36470


namespace NUMINAMATH_GPT_more_people_joined_l364_36451

def initial_people : Nat := 61
def final_people : Nat := 83

theorem more_people_joined :
  final_people - initial_people = 22 := by
  sorry

end NUMINAMATH_GPT_more_people_joined_l364_36451


namespace NUMINAMATH_GPT_number_of_shirts_is_20_l364_36480

/-- Given the conditions:
1. The total price for some shirts is 360,
2. The total price for 45 sweaters is 900,
3. The average price of a sweater exceeds that of a shirt by 2,
prove that the number of shirts is 20. -/

theorem number_of_shirts_is_20
  (S : ℕ) (P_shirt P_sweater : ℝ)
  (h1 : S * P_shirt = 360)
  (h2 : 45 * P_sweater = 900)
  (h3 : P_sweater = P_shirt + 2) :
  S = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_shirts_is_20_l364_36480


namespace NUMINAMATH_GPT_equal_real_roots_of_quadratic_eq_l364_36459

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end NUMINAMATH_GPT_equal_real_roots_of_quadratic_eq_l364_36459


namespace NUMINAMATH_GPT_mathematics_equivalent_proof_l364_36446

noncomputable def distinctRealNumbers (a b c d : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d

theorem mathematics_equivalent_proof (a b c d : ℝ)
  (H₀ : distinctRealNumbers a b c d)
  (H₁ : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 :=
sorry

end NUMINAMATH_GPT_mathematics_equivalent_proof_l364_36446


namespace NUMINAMATH_GPT_seventh_term_of_geometric_sequence_l364_36424

theorem seventh_term_of_geometric_sequence (r : ℝ) 
  (h1 : 3 * r^5 = 729) : 3 * r^6 = 2187 :=
sorry

end NUMINAMATH_GPT_seventh_term_of_geometric_sequence_l364_36424


namespace NUMINAMATH_GPT_value_of_expression_l364_36405

theorem value_of_expression (x y : ℕ) (h1 : x = 4) (h2 : y = 3) : x + 2 * y = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_expression_l364_36405


namespace NUMINAMATH_GPT_max_point_f_l364_36472

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Maximum point of the function f is -2
theorem max_point_f : ∃ m, m = -2 ∧ ∀ x, f x ≤ f (-2) :=
by
  sorry

end NUMINAMATH_GPT_max_point_f_l364_36472


namespace NUMINAMATH_GPT_triangle_area_correct_l364_36482

def line1 (x : ℝ) : ℝ := 8
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the intersection points
def intersection1 : ℝ × ℝ := (6, line1 6)
def intersection2 : ℝ × ℝ := (-6, line1 (-6))
def intersection3 : ℝ × ℝ := (0, line2 0)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_correct :
  triangle_area intersection1 intersection2 intersection3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l364_36482


namespace NUMINAMATH_GPT_simplify_expression_l364_36400

theorem simplify_expression (x y : ℝ) : 
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l364_36400


namespace NUMINAMATH_GPT_polynomial_root_condition_l364_36468

theorem polynomial_root_condition (a : ℝ) :
  (∃ x1 x2 x3 : ℝ,
    (x1^3 - 6 * x1^2 + a * x1 + a = 0) ∧
    (x2^3 - 6 * x2^2 + a * x2 + a = 0) ∧
    (x3^3 - 6 * x3^2 + a * x3 + a = 0) ∧
    ((x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0)) →
  a = -9 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_root_condition_l364_36468


namespace NUMINAMATH_GPT_line_through_origin_tangent_lines_line_through_tangents_l364_36445

section GeomProblem

variables {A : ℝ × ℝ} {C : ℝ × ℝ → Prop}

def is_circle (C : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
∀ (P : ℝ × ℝ), C P ↔ (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

theorem line_through_origin (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ m : ℝ, ∀ P : ℝ × ℝ, C P → abs ((m * P.1 - P.2) / Real.sqrt (m ^ 2 + 1)) = 1)
    ↔ m = 0 :=
sorry

theorem tangent_lines (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P : ℝ × ℝ, C P → (P.2 - 2 * Real.sqrt 3) = k * (P.1 - 1))
    ↔ (∀ P : ℝ × ℝ, C P → (Real.sqrt 3 * P.1 - 3 * P.2 + 5 * Real.sqrt 3 = 0 ∨ P.1 = 1)) :=
sorry

theorem line_through_tangents (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P D E : ℝ × ℝ, C P → (Real.sqrt 3 * D.1 - 3 * D.2 + 5 * Real.sqrt 3 = 0 ∧
                                      (E.1 - 1 = 0 ∨ Real.sqrt 3 * E.1 - 3 * E.2 + 5 * Real.sqrt 3 = 0)) →
    (D.1 + Real.sqrt 3 * D.2 - 1 = 0 ∧ E.1 + Real.sqrt 3 * E.2 - 1 = 0)) :=
sorry

end GeomProblem

end NUMINAMATH_GPT_line_through_origin_tangent_lines_line_through_tangents_l364_36445


namespace NUMINAMATH_GPT_projection_of_a_onto_b_l364_36413

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / magnitude v2

theorem projection_of_a_onto_b : projection vec_a vec_b = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_projection_of_a_onto_b_l364_36413


namespace NUMINAMATH_GPT_votes_cast_46800_l364_36474

-- Define the election context
noncomputable def total_votes (v : ℕ) : Prop :=
  let percentage_a := 0.35
  let percentage_b := 0.40
  let vote_diff := 2340
  (percentage_b - percentage_a) * (v : ℝ) = (vote_diff : ℝ)

-- Theorem stating the total number of votes cast in the election
theorem votes_cast_46800 : total_votes 46800 :=
by
  sorry

end NUMINAMATH_GPT_votes_cast_46800_l364_36474


namespace NUMINAMATH_GPT_natural_numbers_equal_power_l364_36496

theorem natural_numbers_equal_power
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n :=
by
  sorry

end NUMINAMATH_GPT_natural_numbers_equal_power_l364_36496


namespace NUMINAMATH_GPT_express_B_using_roster_l364_36434

open Set

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem express_B_using_roster :
  B = {4, 9, 16} := by
  sorry

end NUMINAMATH_GPT_express_B_using_roster_l364_36434


namespace NUMINAMATH_GPT_slope_of_line_l364_36422

theorem slope_of_line (θ : ℝ) (h : θ = 30) :
  ∃ k, k = Real.tan (60 * (π / 180)) ∨ k = Real.tan (120 * (π / 180)) := by
    sorry

end NUMINAMATH_GPT_slope_of_line_l364_36422


namespace NUMINAMATH_GPT_matrix_solution_l364_36411

open Matrix

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, -3], ![4, -1]]
noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := ![![ -8,  5], ![ 11, -7]]

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![ -1.2, -1.4], ![1.7, 1.9]]

theorem matrix_solution : M * A = B :=
by sorry

end NUMINAMATH_GPT_matrix_solution_l364_36411


namespace NUMINAMATH_GPT_work_day_meeting_percent_l364_36435

open Nat

theorem work_day_meeting_percent :
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 35 := 
by
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  sorry

end NUMINAMATH_GPT_work_day_meeting_percent_l364_36435


namespace NUMINAMATH_GPT_original_number_exists_l364_36423

theorem original_number_exists (x : ℤ) (h1 : x * 16 = 3408) (h2 : 0.016 * 2.13 = 0.03408) : x = 213 := 
by 
  sorry

end NUMINAMATH_GPT_original_number_exists_l364_36423


namespace NUMINAMATH_GPT_min_sum_of_gcd_and_lcm_eq_three_times_sum_l364_36403

theorem min_sum_of_gcd_and_lcm_eq_three_times_sum (a b d : ℕ) (h1 : d = Nat.gcd a b)
  (h2 : Nat.gcd a b + Nat.lcm a b = 3 * (a + b)) :
  a + b = 12 :=
by
sorry

end NUMINAMATH_GPT_min_sum_of_gcd_and_lcm_eq_three_times_sum_l364_36403


namespace NUMINAMATH_GPT_original_price_of_coat_l364_36443

theorem original_price_of_coat (P : ℝ) (h : 0.40 * P = 200) : P = 500 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_price_of_coat_l364_36443


namespace NUMINAMATH_GPT_find_y_l364_36476

-- Declare the variables and conditions
variable (x y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 1.5 * x = 0.3 * y
def condition2 : Prop := x = 20

-- State the theorem that given these conditions, y must be 100
theorem find_y (h1 : condition1 x y) (h2 : condition2 x) : y = 100 :=
by sorry

end NUMINAMATH_GPT_find_y_l364_36476


namespace NUMINAMATH_GPT_radius_increase_of_pizza_l364_36427

/-- 
Prove that the percent increase in radius from a medium pizza to a large pizza is 20% 
given the following conditions:
1. The radius of the large pizza is some percent larger than that of a medium pizza.
2. The percent increase in area between a medium and a large pizza is approximately 44%.
3. The area of a circle is given by the formula A = π * r^2.
--/
theorem radius_increase_of_pizza
  (r R : ℝ) -- r and R are the radii of the medium and large pizza respectively
  (h1 : R = (1 + k) * r) -- The radius of the large pizza is some percent larger than that of a medium pizza
  (h2 : π * R^2 = 1.44 * π * r^2) -- The percent increase in area between a medium and a large pizza is approximately 44%
  : k = 0.2 := 
sorry

end NUMINAMATH_GPT_radius_increase_of_pizza_l364_36427


namespace NUMINAMATH_GPT_percent_shaded_of_square_l364_36455

theorem percent_shaded_of_square (side_len : ℤ) (first_layer_side : ℤ) 
(second_layer_outer_side : ℤ) (second_layer_inner_side : ℤ)
(third_layer_outer_side : ℤ) (third_layer_inner_side : ℤ)
(h_side : side_len = 7) (h_first : first_layer_side = 2) 
(h_second_outer : second_layer_outer_side = 5) (h_second_inner : second_layer_inner_side = 3) 
(h_third_outer : third_layer_outer_side = 7) (h_third_inner : third_layer_inner_side = 6) : 
  (4 + (25 - 9) + (49 - 36)) / (side_len * side_len : ℝ) = 33 / 49 :=
by
  -- Sorry is used as we are only required to construct the statement, not the proof.
  sorry

end NUMINAMATH_GPT_percent_shaded_of_square_l364_36455


namespace NUMINAMATH_GPT_sasha_sequence_eventually_five_to_100_l364_36431

theorem sasha_sequence_eventually_five_to_100 :
  ∃ (n : ℕ), 
  (5 ^ 100) = initial_value + n * (3 ^ 100) - m * (2 ^ 100) ∧ 
  (initial_value + n * (3 ^ 100) - m * (2 ^ 100) > 0) :=
by
  let initial_value := 1
  let threshold := 2 ^ 100
  let increment := 3 ^ 100
  let decrement := 2 ^ 100
  sorry

end NUMINAMATH_GPT_sasha_sequence_eventually_five_to_100_l364_36431


namespace NUMINAMATH_GPT_polar_to_cartesian_l364_36456

theorem polar_to_cartesian :
  ∃ (x y : ℝ), x = 2 * Real.cos (Real.pi / 6) ∧ y = 2 * Real.sin (Real.pi / 6) ∧ 
  (x, y) = (Real.sqrt 3, 1) :=
by
  use (2 * Real.cos (Real.pi / 6)), (2 * Real.sin (Real.pi / 6))
  -- The proof will show the necessary steps
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l364_36456


namespace NUMINAMATH_GPT_inequality_solution_l364_36412

theorem inequality_solution (x : ℤ) : (1 + x) / 2 - (2 * x + 1) / 3 ≤ 1 → x ≥ -5 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l364_36412


namespace NUMINAMATH_GPT_find_hourly_wage_l364_36492

noncomputable def hourly_wage_inexperienced (x : ℝ) : Prop :=
  let sailors_total := 17
  let inexperienced_sailors := 5
  let experienced_sailors := sailors_total - inexperienced_sailors
  let wage_experienced := (6 / 5) * x
  let total_hours_month := 240
  let total_monthly_earnings_experienced := 34560
  (experienced_sailors * wage_experienced * total_hours_month) = total_monthly_earnings_experienced

theorem find_hourly_wage (x : ℝ) : hourly_wage_inexperienced x → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_hourly_wage_l364_36492


namespace NUMINAMATH_GPT_max_sum_of_digits_l364_36462

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_sum_of_digits : ∃ h m : ℕ, h < 24 ∧ m < 60 ∧
  sum_of_digits h + sum_of_digits m = 24 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_digits_l364_36462


namespace NUMINAMATH_GPT_move_line_down_l364_36457

theorem move_line_down (x y : ℝ) : (y = -3 * x + 5) → (y = -3 * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_move_line_down_l364_36457


namespace NUMINAMATH_GPT_systematic_sampling_second_invoice_l364_36426

theorem systematic_sampling_second_invoice 
  (N : ℕ) 
  (valid_invoice : N ≥ 10)
  (first_invoice : Fin 10) :
  ¬ (∃ k : ℕ, k ≥ 1 ∧ first_invoice.1 + k * 10 = 23) := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_systematic_sampling_second_invoice_l364_36426


namespace NUMINAMATH_GPT_rabbit_shape_area_l364_36433

theorem rabbit_shape_area (A_ear : ℝ) (h1 : A_ear = 10) (h2 : A_ear = (1/8) * A_total) :
  A_total = 80 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_shape_area_l364_36433


namespace NUMINAMATH_GPT_integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l364_36461

theorem integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5 :
  ∀ n : ℤ, ∃ k : ℕ, (n^3 - 3 * n^2 + n + 2 = 5^k) ↔ n = 3 :=
by
  intro n
  exists sorry
  sorry

end NUMINAMATH_GPT_integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l364_36461


namespace NUMINAMATH_GPT_area_of_triangle_l364_36490

theorem area_of_triangle :
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  triangle_area = 34 :=
by {
  -- Definitions
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  -- Proof (normally written here, but omitted with 'sorry')
  sorry
}

end NUMINAMATH_GPT_area_of_triangle_l364_36490


namespace NUMINAMATH_GPT_molecular_weight_of_NH4I_l364_36436

-- Define the conditions in Lean
def molecular_weight (moles grams: ℕ) : Prop :=
  grams / moles = 145

-- Statement of the proof problem
theorem molecular_weight_of_NH4I :
  molecular_weight 9 1305 :=
by
  -- Proof is omitted 
  sorry

end NUMINAMATH_GPT_molecular_weight_of_NH4I_l364_36436


namespace NUMINAMATH_GPT_equivalence_of_equation_and_conditions_l364_36484

open Real
open Set

-- Definitions for conditions
def condition1 (t : ℝ) : Prop := cos t ≠ 0
def condition2 (t : ℝ) : Prop := sin t ≠ 0
def condition3 (t : ℝ) : Prop := cos (2 * t) ≠ 0

-- The main statement to be proved
theorem equivalence_of_equation_and_conditions (t : ℝ) :
  ((sin t / cos t - cos t / sin t + 2 * (sin (2 * t) / cos (2 * t))) * (1 + cos (3 * t))) = 4 * sin (3 * t) ↔
  ((∃ k l : ℤ, t = (π / 5) * (2 * k + 1) ∧ k ≠ 5 * l + 2) ∨ (∃ n l : ℤ, t = (π / 3) * (2 * n + 1) ∧ n ≠ 3 * l + 1))
    ∧ condition1 t
    ∧ condition2 t
    ∧ condition3 t :=
by
  sorry

end NUMINAMATH_GPT_equivalence_of_equation_and_conditions_l364_36484


namespace NUMINAMATH_GPT_correct_expression_l364_36410

-- Definitions based on given conditions
def expr1 (a b : ℝ) := 3 * a + 2 * b = 5 * a * b
def expr2 (a : ℝ) := 2 * a^3 - a^3 = a^3
def expr3 (a b : ℝ) := a^2 * b - a * b = a
def expr4 (a : ℝ) := a^2 + a^2 = 2 * a^4

-- Statement to prove that expr2 is the only correct expression
theorem correct_expression (a b : ℝ) : 
  expr2 a := by
  sorry

end NUMINAMATH_GPT_correct_expression_l364_36410


namespace NUMINAMATH_GPT_solve_range_m_l364_36466

variable (m : ℝ)
def p := m < 0
def q := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem solve_range_m (hpq : p m ∧ q m) : -2 < m ∧ m < 0 := 
  sorry

end NUMINAMATH_GPT_solve_range_m_l364_36466


namespace NUMINAMATH_GPT_product_of_first_three_terms_is_960_l364_36493

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end NUMINAMATH_GPT_product_of_first_three_terms_is_960_l364_36493


namespace NUMINAMATH_GPT_no_such_primes_l364_36416

theorem no_such_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_three : p > 3) (hq_gt_three : q > 3) (hq_div_p2_minus_1 : q ∣ (p^2 - 1)) 
  (hp_div_q2_minus_1 : p ∣ (q^2 - 1)) : false := 
sorry

end NUMINAMATH_GPT_no_such_primes_l364_36416


namespace NUMINAMATH_GPT_increased_percentage_l364_36414

theorem increased_percentage (x : ℝ) (p : ℝ) (h : x = 75) (h₁ : p = 1.5) : x + (p * x) = 187.5 :=
by
  sorry

end NUMINAMATH_GPT_increased_percentage_l364_36414


namespace NUMINAMATH_GPT_area_under_pressure_l364_36429

theorem area_under_pressure (F : ℝ) (S : ℝ) (p : ℝ) (hF : F = 100) (hp : p > 1000) (hpressure : p = F / S) :
  S < 0.1 :=
by
  sorry

end NUMINAMATH_GPT_area_under_pressure_l364_36429


namespace NUMINAMATH_GPT_distinct_pos_real_ints_l364_36489

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem distinct_pos_real_ints (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : ∀ n : ℕ, (floor (n * a)) ∣ (floor (n * b))) : ∃ k l : ℤ, a = k ∧ b = l :=
by
  sorry

end NUMINAMATH_GPT_distinct_pos_real_ints_l364_36489


namespace NUMINAMATH_GPT_range_of_x_l364_36440

noncomputable def f (x : ℝ) : ℝ := (5 / (x^2)) - (3 * (x^2)) + 2

theorem range_of_x :
  { x : ℝ | f 1 < f (Real.log x / Real.log 3) } = { x : ℝ | (1 / 3) < x ∧ x < 1 ∨ 1 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l364_36440


namespace NUMINAMATH_GPT_cos_300_eq_one_half_l364_36467

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_300_eq_one_half_l364_36467


namespace NUMINAMATH_GPT_hosting_schedules_count_l364_36420

theorem hosting_schedules_count :
  let n_universities := 6
  let n_years := 8
  let total_ways := 6 * 5 * 4^6
  let excluding_one := 6 * 5 * 4 * 3^6
  let excluding_two := 15 * 4 * 3 * 2^6
  let excluding_three := 20 * 3 * 2 * 1^6
  total_ways - excluding_one + excluding_two - excluding_three = 46080 := 
by
  sorry

end NUMINAMATH_GPT_hosting_schedules_count_l364_36420


namespace NUMINAMATH_GPT_price_difference_l364_36479

theorem price_difference (P F : ℝ) (h1 : 0.85 * P = 78.2) (h2 : F = 78.2 * 1.25) : F - P = 5.75 :=
by
  sorry

end NUMINAMATH_GPT_price_difference_l364_36479


namespace NUMINAMATH_GPT_derivative_of_f_is_l364_36425

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2

theorem derivative_of_f_is (x : ℝ) : deriv f x = 2 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_f_is_l364_36425


namespace NUMINAMATH_GPT_proportion_of_salt_correct_l364_36478

def grams_of_salt := 50
def grams_of_water := 1000
def total_solution := grams_of_salt + grams_of_water
def proportion_of_salt : ℚ := grams_of_salt / total_solution

theorem proportion_of_salt_correct :
  proportion_of_salt = 1 / 21 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_proportion_of_salt_correct_l364_36478


namespace NUMINAMATH_GPT_unique_integer_solution_l364_36419

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2 * n^2 + m^2 + n^2 + 6 * m * n ↔ m = 0 ∧ n = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_integer_solution_l364_36419


namespace NUMINAMATH_GPT_gcd_lcm_of_45_and_150_l364_36487

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem gcd_lcm_of_45_and_150 :
  GCD 45 150 = 15 ∧ LCM 45 150 = 450 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_of_45_and_150_l364_36487


namespace NUMINAMATH_GPT_total_amount_received_correct_l364_36465

variable (total_won : ℝ) (fraction : ℝ) (students : ℕ)
variable (portion_per_student : ℝ := total_won * fraction)
variable (total_given : ℝ := portion_per_student * students)

theorem total_amount_received_correct :
  total_won = 555850 →
  fraction = 3 / 10000 →
  students = 500 →
  total_given = 833775 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_total_amount_received_correct_l364_36465


namespace NUMINAMATH_GPT_domain_of_func_l364_36402

noncomputable def func (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem domain_of_func :
  ∀ x : ℝ, x ≠ 1 / 2 ↔ ∃ y : ℝ, y = func x := sorry

end NUMINAMATH_GPT_domain_of_func_l364_36402


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l364_36432

-- Problem 1
theorem problem1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) (h : m - n = 2) : 2 * (n - m) - 4 * m + 4 * n - 3 = -15 :=
by sorry

-- Problem 3
theorem problem3 (m n : ℝ) (h1 : m^2 + 2 * m * n = -2) (h2 : m * n - n^2 = -4) : 
  3 * m^2 + (9 / 2) * m * n + (3 / 2) * n^2 = 0 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l364_36432


namespace NUMINAMATH_GPT_find_least_number_subtracted_l364_36418

theorem find_least_number_subtracted (n m : ℕ) (h : n = 78721) (h1 : m = 23) : (n % m) = 15 := by
  sorry

end NUMINAMATH_GPT_find_least_number_subtracted_l364_36418


namespace NUMINAMATH_GPT_door_height_is_eight_l364_36495

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_door_height_is_eight_l364_36495


namespace NUMINAMATH_GPT_rectangle_area_change_l364_36498

theorem rectangle_area_change
  (L B : ℝ)
  (hL : L > 0)
  (hB : B > 0)
  (new_L : ℝ := 1.25 * L)
  (new_B : ℝ := 0.85 * B):
  (new_L * new_B = 1.0625 * (L * B)) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_change_l364_36498


namespace NUMINAMATH_GPT_both_boys_and_girls_selected_probability_l364_36444

theorem both_boys_and_girls_selected_probability :
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) :=
by
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  have h : (only_girls_ways / total_ways : ℚ) = (1 / 10 : ℚ) := sorry
  have h1 : (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) := by rw [h]; norm_num
  exact h1

end NUMINAMATH_GPT_both_boys_and_girls_selected_probability_l364_36444


namespace NUMINAMATH_GPT_day_of_50th_in_year_N_minus_1_l364_36407

theorem day_of_50th_in_year_N_minus_1
  (N : ℕ)
  (day250_in_year_N_is_sunday : (250 % 7 = 0))
  (day150_in_year_N_plus_1_is_sunday : (150 % 7 = 0))
  : 
  (50 % 7 = 1) := 
sorry

end NUMINAMATH_GPT_day_of_50th_in_year_N_minus_1_l364_36407


namespace NUMINAMATH_GPT_sheepdog_rounded_up_percentage_l364_36473

/-- Carla's sheepdog rounded up a certain percentage of her sheep. We know the remaining 10% of the sheep  wandered off into the hills, which is 9 sheep out in the wilderness. There are 81 sheep in the pen. We need to prove that the sheepdog rounded up 90% of the total number of sheep. -/
theorem sheepdog_rounded_up_percentage (total_sheep pen_sheep wilderness_sheep : ℕ) 
  (h1 : wilderness_sheep = 9) 
  (h2 : pen_sheep = 81) 
  (h3 : wilderness_sheep = total_sheep / 10) :
  (pen_sheep * 100 / total_sheep) = 90 :=
sorry

end NUMINAMATH_GPT_sheepdog_rounded_up_percentage_l364_36473


namespace NUMINAMATH_GPT_incorrect_divisor_l364_36477

theorem incorrect_divisor (D x : ℕ) (h1 : D = 24 * x) (h2 : D = 48 * 36) : x = 72 := by
  sorry

end NUMINAMATH_GPT_incorrect_divisor_l364_36477


namespace NUMINAMATH_GPT_acute_angles_sine_relation_l364_36408

theorem acute_angles_sine_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : 2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β) : α < β :=
by
  sorry

end NUMINAMATH_GPT_acute_angles_sine_relation_l364_36408


namespace NUMINAMATH_GPT_problem_l364_36447

theorem problem (a b c d : ℝ) (h1 : b + c = 7) (h2 : c + d = 5) (h3 : a + d = 2) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_problem_l364_36447


namespace NUMINAMATH_GPT_find_2a_plus_b_l364_36494

open Real

variables {a b : ℝ}

-- Conditions
def angles_in_first_quadrant (a b : ℝ) : Prop := 
  0 < a ∧ a < π / 2 ∧ 0 < b ∧ b < π / 2

def cos_condition (a b : ℝ) : Prop :=
  5 * cos a ^ 2 + 3 * cos b ^ 2 = 2

def sin_condition (a b : ℝ) : Prop :=
  5 * sin (2 * a) + 3 * sin (2 * b) = 0

-- Problem statement
theorem find_2a_plus_b (a b : ℝ) 
  (h1 : angles_in_first_quadrant a b)
  (h2 : cos_condition a b)
  (h3 : sin_condition a b) :
  2 * a + b = π / 2 := 
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l364_36494


namespace NUMINAMATH_GPT_fish_to_rice_equivalence_l364_36437

variable (f : ℚ) (l : ℚ)

theorem fish_to_rice_equivalence (h1 : 5 * f = 3 * l) (h2 : l = 6) : f = 18 / 5 := by
  sorry

end NUMINAMATH_GPT_fish_to_rice_equivalence_l364_36437


namespace NUMINAMATH_GPT_smallest_root_of_g_l364_36458

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- The main statement: proving the smallest root of g(x) is -sqrt(7/5)
theorem smallest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → x ≤ y := 
sorry

end NUMINAMATH_GPT_smallest_root_of_g_l364_36458


namespace NUMINAMATH_GPT_calculate_expression_l364_36481

theorem calculate_expression :
  (16^16 * 8^8) / 4^32 = 16777216 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l364_36481


namespace NUMINAMATH_GPT_sqrt_meaningful_condition_l364_36430

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end NUMINAMATH_GPT_sqrt_meaningful_condition_l364_36430


namespace NUMINAMATH_GPT_mr_johnson_total_volunteers_l364_36439

theorem mr_johnson_total_volunteers (students_per_class : ℕ) (classes : ℕ) (teachers : ℕ) (additional_volunteers : ℕ) :
  students_per_class = 5 → classes = 6 → teachers = 13 → additional_volunteers = 7 →
  (students_per_class * classes + teachers + additional_volunteers) = 50 :=
by intros; simp [*]

end NUMINAMATH_GPT_mr_johnson_total_volunteers_l364_36439


namespace NUMINAMATH_GPT_time_per_flash_l364_36438

def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60
def light_flashes_in_three_fourths_hour : ℕ := 180

-- Converting ¾ of an hour to minutes and then to seconds
def seconds_in_three_fourths_hour : ℕ := (3 * minutes_per_hour / 4) * seconds_per_minute

-- Proving that the time taken for one flash is 15 seconds
theorem time_per_flash : (seconds_in_three_fourths_hour / light_flashes_in_three_fourths_hour) = 15 :=
by
  sorry

end NUMINAMATH_GPT_time_per_flash_l364_36438


namespace NUMINAMATH_GPT_find_value_of_f_neg_3_over_2_l364_36452

noncomputable def f : ℝ → ℝ := sorry

theorem find_value_of_f_neg_3_over_2 (h1 : ∀ x : ℝ, f (-x) = -f x) 
    (h2 : ∀ x : ℝ, f (x + 3/2) = -f x) : 
    f (- 3 / 2) = 0 := 
sorry

end NUMINAMATH_GPT_find_value_of_f_neg_3_over_2_l364_36452


namespace NUMINAMATH_GPT_amount_of_first_alloy_used_is_15_l364_36460

-- Definitions of percentages and weights
def chromium_percentage_first_alloy : ℝ := 0.12
def chromium_percentage_second_alloy : ℝ := 0.08
def weight_second_alloy : ℝ := 40
def chromium_percentage_new_alloy : ℝ := 0.0909090909090909
def total_weight_new_alloy (x : ℝ) : ℝ := x + weight_second_alloy
def chromium_content_first_alloy (x : ℝ) : ℝ := chromium_percentage_first_alloy * x
def chromium_content_second_alloy : ℝ := chromium_percentage_second_alloy * weight_second_alloy
def total_chromium_content (x : ℝ) : ℝ := chromium_content_first_alloy x + chromium_content_second_alloy

-- The proof problem
theorem amount_of_first_alloy_used_is_15 :
  ∃ x : ℝ, total_chromium_content x = chromium_percentage_new_alloy * total_weight_new_alloy x ∧ x = 15 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_first_alloy_used_is_15_l364_36460


namespace NUMINAMATH_GPT_max_constant_C_all_real_numbers_l364_36469

theorem max_constant_C_all_real_numbers:
  ∀ (x1 x2 x3 x4 x5 x6 : ℝ), 
  (x1 + x2 + x3 + x4 + x5 + x6)^2 ≥ 
  3 * (x1 * (x2 + x3) + x2 * (x3 + x4) + x3 * (x4 + x5) + x4 * (x5 + x6) + x5 * (x6 + x1) + x6 * (x1 + x2)) := 
by 
  sorry

end NUMINAMATH_GPT_max_constant_C_all_real_numbers_l364_36469


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l364_36488

variables (p q : Prop)

theorem necessary_but_not_sufficient_condition
  (h : ¬p → q) (hn : ¬q → p) : 
  (p → ¬q) ∧ ¬(¬q → p) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l364_36488


namespace NUMINAMATH_GPT_solve_equation_l364_36415

theorem solve_equation (x y : ℕ) (h_xy : x ≠ y) : x = 2 ∧ y = 4 ∨ x = 4 ∧ y = 2 :=
by {
  sorry -- Proof skipped
}

end NUMINAMATH_GPT_solve_equation_l364_36415


namespace NUMINAMATH_GPT_sequence_general_formula_l364_36442

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 12)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) :
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 12 :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l364_36442


namespace NUMINAMATH_GPT_triangle_height_l364_36453

theorem triangle_height (base height : ℝ) (area : ℝ) (h_base : base = 4) (h_area : area = 12) (h_area_eq : area = (base * height) / 2) :
  height = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l364_36453


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l364_36454

structure Point : Type where
  x : ℝ
  y : ℝ

def symmetric_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def P : Point := { x := -10, y := -1 }

def P1 : Point := symmetric_y P

def P2 : Point := symmetric_x P1

theorem symmetric_point_coordinates :
  P2 = { x := 10, y := 1 } := by
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l364_36454


namespace NUMINAMATH_GPT_quiz_passing_condition_l364_36450

theorem quiz_passing_condition (P Q : Prop) :
  (Q → P) → 
    (¬P → ¬Q) ∧ 
    (¬Q → ¬P) ∧ 
    (P → Q) :=
by sorry

end NUMINAMATH_GPT_quiz_passing_condition_l364_36450


namespace NUMINAMATH_GPT_scientific_notation_l364_36409

theorem scientific_notation : 350000000 = 3.5 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l364_36409


namespace NUMINAMATH_GPT_integer_solutions_set_l364_36441

theorem integer_solutions_set :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_integer_solutions_set_l364_36441


namespace NUMINAMATH_GPT_opposite_of_2023_l364_36485

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l364_36485


namespace NUMINAMATH_GPT_solution_set_to_coeff_properties_l364_36448

theorem solution_set_to_coeff_properties 
  (a b c : ℝ) 
  (h : ∀ x, (2 < x ∧ x < 3) → ax^2 + bx + c > 0) 
  : 
  (a < 0) 
  ∧ (b * c < 0) 
  ∧ (b + c = a) :=
sorry

end NUMINAMATH_GPT_solution_set_to_coeff_properties_l364_36448


namespace NUMINAMATH_GPT_dreamy_bookstore_sales_l364_36491

theorem dreamy_bookstore_sales :
  let total_sales_percent := 100
  let notebooks_percent := 45
  let bookmarks_percent := 25
  let neither_notebooks_nor_bookmarks_percent := total_sales_percent - (notebooks_percent + bookmarks_percent)
  neither_notebooks_nor_bookmarks_percent = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_dreamy_bookstore_sales_l364_36491
