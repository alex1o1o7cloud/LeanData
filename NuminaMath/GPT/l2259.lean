import Mathlib

namespace NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l2259_225946

theorem largest_multiple_of_8_less_than_100 : ∃ (n : ℕ), n < 100 ∧ 8 ∣ n ∧ ∀ (m : ℕ), m < 100 ∧ 8 ∣ m → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l2259_225946


namespace NUMINAMATH_GPT_largest_polygon_area_l2259_225923

variable (area : ℕ → ℝ)

def polygon_A_area : ℝ := 6
def polygon_B_area : ℝ := 3 + 4 * 0.5
def polygon_C_area : ℝ := 4 + 5 * 0.5
def polygon_D_area : ℝ := 7
def polygon_E_area : ℝ := 2 + 6 * 0.5

theorem largest_polygon_area : polygon_D_area = max (max (max polygon_A_area polygon_B_area) polygon_C_area) polygon_E_area :=
by
  sorry

end NUMINAMATH_GPT_largest_polygon_area_l2259_225923


namespace NUMINAMATH_GPT_smallest_possible_value_of_d_l2259_225989

theorem smallest_possible_value_of_d (c d : ℝ) (hc : 1 < c) (hd : c < d)
  (h_triangle1 : ¬(1 + c > d ∧ c + d > 1 ∧ 1 + d > c))
  (h_triangle2 : ¬(1 / c + 1 / d > 1 ∧ 1 / d + 1 > 1 / c ∧ 1 / c + 1 > 1 / d)) :
  d = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_d_l2259_225989


namespace NUMINAMATH_GPT_values_of_y_satisfy_quadratic_l2259_225903

theorem values_of_y_satisfy_quadratic :
  (∃ (x y : ℝ), 3 * x^2 + 4 * x + 7 * y + 2 = 0 ∧ 3 * x + 2 * y + 4 = 0) →
  (∃ (y : ℝ), 4 * y^2 + 29 * y + 6 = 0) :=
by sorry

end NUMINAMATH_GPT_values_of_y_satisfy_quadratic_l2259_225903


namespace NUMINAMATH_GPT_all_numbers_positive_l2259_225940

theorem all_numbers_positive (n : ℕ) (a : Fin (2 * n + 1) → ℝ) 
  (h : ∀ S : Finset (Fin (2 * n + 1)), 
        S.card = n + 1 → 
        S.sum a > (Finset.univ \ S).sum a) : 
  ∀ i, 0 < a i :=
by
  sorry

end NUMINAMATH_GPT_all_numbers_positive_l2259_225940


namespace NUMINAMATH_GPT_tan_value_l2259_225993

open Real

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

noncomputable def arithmetic_seq (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem tan_value
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : geometric_seq a)
  (hb : arithmetic_seq b)
  (h_geom : a 0 * a 5 * a 10 = -3 * sqrt 3)
  (h_arith : b 0 + b 5 + b 10 = 7 * π) :
  tan ((b 2 + b 8) / (1 - a 3 * a 7)) = -sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_value_l2259_225993


namespace NUMINAMATH_GPT_max_side_of_triangle_l2259_225919

theorem max_side_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) 
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : c ≤ 11 := 
sorry

end NUMINAMATH_GPT_max_side_of_triangle_l2259_225919


namespace NUMINAMATH_GPT_find_a_from_roots_l2259_225949

theorem find_a_from_roots (θ : ℝ) (a : ℝ) (h1 : ∀ x : ℝ, 4 * x^2 + 2 * a * x + a = 0 → (x = Real.sin θ ∨ x = Real.cos θ)) :
  a = 1 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_roots_l2259_225949


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2259_225991

theorem boat_speed_in_still_water  (b s : ℝ) (h1 : b + s = 13) (h2 : b - s = 9) : b = 11 :=
sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2259_225991


namespace NUMINAMATH_GPT_profit_percent_l2259_225926

-- Definitions based on the conditions in the problem
def marked_price_per_pen := ℝ
def total_pens := 52
def cost_equivalent_pens := 46
def discount_percentage := 1 / 100

-- Values calculated from conditions
def cost_price (P : ℝ) := cost_equivalent_pens * P
def selling_price_per_pen (P : ℝ) := P * (1 - discount_percentage)
def total_selling_price (P : ℝ) := total_pens * selling_price_per_pen P

-- The proof statement
theorem profit_percent (P : ℝ) (hP : P > 0) :
  ((total_selling_price P - cost_price P) / (cost_price P)) * 100 = 11.91 := by
    sorry

end NUMINAMATH_GPT_profit_percent_l2259_225926


namespace NUMINAMATH_GPT_speed_of_goods_train_l2259_225992

theorem speed_of_goods_train
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_crossing : ℕ)
  (h_length_train : length_train = 240)
  (h_length_platform : length_platform = 280)
  (h_time_crossing : time_crossing = 26)
  : (length_train + length_platform) / time_crossing * (3600 / 1000) = 72 := 
by sorry

end NUMINAMATH_GPT_speed_of_goods_train_l2259_225992


namespace NUMINAMATH_GPT_equation_has_real_root_l2259_225959

theorem equation_has_real_root (x : ℝ) : (x^3 + 3 = 0) ↔ (x = -((3:ℝ)^(1/3))) :=
sorry

end NUMINAMATH_GPT_equation_has_real_root_l2259_225959


namespace NUMINAMATH_GPT_magazines_in_third_pile_l2259_225904

-- Define the number of magazines in each pile.
def pile1 := 3
def pile2 := 4
def pile4 := 9
def pile5 := 13

-- Define the differences between the piles.
def diff2_1 := pile2 - pile1  -- Difference between second and first pile
def diff4_2 := pile4 - pile2  -- Difference between fourth and second pile

-- Assume the pattern continues with differences increasing by 4.
def diff3_2 := diff2_1 + 4    -- Difference between third and second pile

-- Define the number of magazines in the third pile.
def pile3 := pile2 + diff3_2

-- Theorem stating the number of magazines in the third pile.
theorem magazines_in_third_pile : pile3 = 9 := by sorry

end NUMINAMATH_GPT_magazines_in_third_pile_l2259_225904


namespace NUMINAMATH_GPT_systematic_sampling_missiles_l2259_225985

theorem systematic_sampling_missiles (S : Set ℕ) (hS : S = {n | 1 ≤ n ∧ n ≤ 50}) :
  (∃ seq : Fin 5 → ℕ, (∀ i : Fin 4, seq (Fin.succ i) - seq i = 10) ∧ seq 0 = 3)
  → (∃ seq : Fin 5 → ℕ, seq = ![3, 13, 23, 33, 43]) :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_missiles_l2259_225985


namespace NUMINAMATH_GPT_part1_x_values_part2_m_value_l2259_225957

/-- 
Part 1: Given \(2x^2 + 3x - 5\) and \(-2x + 2\) are opposite numbers, 
prove that \(x = -\frac{3}{2}\) or \(x = 1\).
-/
theorem part1_x_values (x : ℝ)
  (hyp : 2 * x ^ 2 + 3 * x - 5 = -(-2 * x + 2)) :
  2 * x ^ 2 + 5 * x - 7 = 0 → (x = -3 / 2 ∨ x = 1) :=
by
  sorry

/-- 
Part 2: If \(\sqrt{m^2 - 6}\) and \(\sqrt{6m + 1}\) are of the same type, 
prove that \(m = 7\).
-/
theorem part2_m_value (m : ℝ)
  (hyp : m ^ 2 - 6 = 6 * m + 1) :
  7 ^ 2 - 6 = 6 * 7 + 1 → m = 7 :=
by
  sorry

end NUMINAMATH_GPT_part1_x_values_part2_m_value_l2259_225957


namespace NUMINAMATH_GPT_distribute_items_among_people_l2259_225952

theorem distribute_items_among_people :
  (Nat.choose (10 + 3 - 1) 3) = 220 := 
by sorry

end NUMINAMATH_GPT_distribute_items_among_people_l2259_225952


namespace NUMINAMATH_GPT_total_animals_is_63_l2259_225938

def zoo_animals (penguins polar_bears total : ℕ) : Prop :=
  (penguins = 21) ∧
  (polar_bears = 2 * penguins) ∧
  (total = penguins + polar_bears)

theorem total_animals_is_63 :
  ∃ (penguins polar_bears total : ℕ), zoo_animals penguins polar_bears total ∧ total = 63 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_animals_is_63_l2259_225938


namespace NUMINAMATH_GPT_quotient_of_division_l2259_225914

theorem quotient_of_division (dividend divisor remainder quotient : ℕ)
  (h_dividend : dividend = 15)
  (h_divisor : divisor = 3)
  (h_remainder : remainder = 3)
  (h_relation : dividend = divisor * quotient + remainder) :
  quotient = 4 :=
by sorry

end NUMINAMATH_GPT_quotient_of_division_l2259_225914


namespace NUMINAMATH_GPT_same_color_pair_exists_l2259_225954

-- Define the coloring of a point on a plane
def is_colored (x y : ℝ) : Type := ℕ  -- Assume ℕ represents two colors 0 and 1

-- Prove there exists two points of the same color such that the distance between them is 2006 meters
theorem same_color_pair_exists (colored : ℝ → ℝ → ℕ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ colored x1 y1 = colored x2 y2 ∧ (x2 - x1)^2 + (y2 - y1)^2 = 2006^2) :=
sorry

end NUMINAMATH_GPT_same_color_pair_exists_l2259_225954


namespace NUMINAMATH_GPT_joe_probability_select_counsel_l2259_225965

theorem joe_probability_select_counsel :
  let CANOE := ['C', 'A', 'N', 'O', 'E']
  let SHRUB := ['S', 'H', 'R', 'U', 'B']
  let FLOW := ['F', 'L', 'O', 'W']
  let COUNSEL := ['C', 'O', 'U', 'N', 'S', 'E', 'L']
  -- Probability of selecting C and O from CANOE
  let p_CANOE := 1 / (Nat.choose 5 2)
  -- Probability of selecting U, S, and E from SHRUB
  let comb_SHRUB := Nat.choose 5 3
  let count_USE := 3  -- Determined from the solution
  let p_SHRUB := count_USE / comb_SHRUB
  -- Probability of selecting L, O, W, F from FLOW
  let p_FLOW := 1 / 1
  -- Total probability
  let total_prob := p_CANOE * p_SHRUB * p_FLOW
  total_prob = 3 / 100 := by
    sorry

end NUMINAMATH_GPT_joe_probability_select_counsel_l2259_225965


namespace NUMINAMATH_GPT_Angela_is_295_cm_l2259_225982

noncomputable def Angela_height (Carl_height : ℕ) : ℕ :=
  let Becky_height := 2 * Carl_height
  let Amy_height := Becky_height + Becky_height / 5  -- 20% taller than Becky
  let Helen_height := Amy_height + 3
  let Angela_height := Helen_height + 4
  Angela_height

theorem Angela_is_295_cm : Angela_height 120 = 295 := 
by 
  sorry

end NUMINAMATH_GPT_Angela_is_295_cm_l2259_225982


namespace NUMINAMATH_GPT_profit_percent_l2259_225958

theorem profit_percent (CP SP : ℕ) (h : CP * 5 = SP * 4) : 100 * (SP - CP) = 25 * CP :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_l2259_225958


namespace NUMINAMATH_GPT_angie_age_problem_l2259_225944

theorem angie_age_problem (a certain_number : ℕ) 
  (h1 : 2 * 8 + certain_number = 20) : 
  certain_number = 4 :=
by 
  sorry

end NUMINAMATH_GPT_angie_age_problem_l2259_225944


namespace NUMINAMATH_GPT_problem1_problem2_l2259_225960

theorem problem1 :
  0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.01 ^ (1 / 2) = 48 / 5 :=
by sorry

theorem problem2 :
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 
  - 25 ^ (Real.log 3 / Real.log 5) = -7 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2259_225960


namespace NUMINAMATH_GPT_unique_rectangles_perimeter_sum_correct_l2259_225907

def unique_rectangle_sum_of_perimeters : ℕ :=
  let possible_pairs := [(4, 12), (6, 6)]
  let perimeters := possible_pairs.map (λ (p : ℕ × ℕ) => 2 * (p.1 + p.2))
  perimeters.sum

theorem unique_rectangles_perimeter_sum_correct : unique_rectangle_sum_of_perimeters = 56 :=
  by 
  -- skipping actual proof
  sorry

end NUMINAMATH_GPT_unique_rectangles_perimeter_sum_correct_l2259_225907


namespace NUMINAMATH_GPT_salary_of_b_l2259_225948

theorem salary_of_b (S_A S_B : ℝ)
  (h1 : S_A + S_B = 14000)
  (h2 : 0.20 * S_A = 0.15 * S_B) :
  S_B = 8000 :=
by
  sorry

end NUMINAMATH_GPT_salary_of_b_l2259_225948


namespace NUMINAMATH_GPT_length_CF_is_7_l2259_225996

noncomputable def CF_length
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  ℝ :=
7

theorem length_CF_is_7
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  CF_length ABCD_rectangle triangle_ABE_right triangle_CDF_right area_triangle_ABE length_AE length_DF h1 h2 h3 h4 h5 h6 = 7 :=
by
  sorry

end NUMINAMATH_GPT_length_CF_is_7_l2259_225996


namespace NUMINAMATH_GPT_each_friend_pays_18_l2259_225963

theorem each_friend_pays_18 (total_bill : ℝ) (silas_share : ℝ) (tip_fraction : ℝ) (num_friends : ℕ) (silas : ℕ) (remaining_friends : ℕ) :
  total_bill = 150 →
  silas_share = total_bill / 2 →
  tip_fraction = 0.1 →
  num_friends = 6 →
  remaining_friends = num_friends - 1 →
  silas = 1 →
  (total_bill - silas_share + tip_fraction * total_bill) / remaining_friends = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_each_friend_pays_18_l2259_225963


namespace NUMINAMATH_GPT_evaluate_expression_l2259_225950

theorem evaluate_expression : 
  (10^8 / (2.5 * 10^5) * 3) = 1200 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2259_225950


namespace NUMINAMATH_GPT_Thabo_owns_25_hardcover_nonfiction_books_l2259_225973

variable (H P F : ℕ)

-- Conditions
def condition1 := P = H + 20
def condition2 := F = 2 * P
def condition3 := H + P + F = 160

-- Goal
theorem Thabo_owns_25_hardcover_nonfiction_books (H P F : ℕ) (h1 : condition1 H P) (h2 : condition2 P F) (h3 : condition3 H P F) : H = 25 :=
by
  sorry

end NUMINAMATH_GPT_Thabo_owns_25_hardcover_nonfiction_books_l2259_225973


namespace NUMINAMATH_GPT_inequality_transformation_l2259_225935

theorem inequality_transformation (x y a : ℝ) (hxy : x < y) (ha : a < 1) : x + a < y + 1 := by
  sorry

end NUMINAMATH_GPT_inequality_transformation_l2259_225935


namespace NUMINAMATH_GPT_number_is_10_l2259_225980

theorem number_is_10 (x : ℕ) (h : x * 15 = 150) : x = 10 :=
sorry

end NUMINAMATH_GPT_number_is_10_l2259_225980


namespace NUMINAMATH_GPT_min_cost_yogurt_l2259_225969

theorem min_cost_yogurt (cost_per_box : ℕ) (boxes : ℕ) (promotion : ℕ → ℕ) (cost : ℕ) :
  cost_per_box = 4 → 
  boxes = 10 → 
  promotion 3 = 2 → 
  cost = 28 := 
by {
  -- The proof will go here
  sorry
}

end NUMINAMATH_GPT_min_cost_yogurt_l2259_225969


namespace NUMINAMATH_GPT_correctOptionOnlyC_l2259_225931

-- Definitions for the transformations
def isTransformA (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b^2) / (a^2)) 
def isTransformB (a b : ℝ) : Prop := (a ≠ 0) → (b ≠ 0) → (b / a = (b + 1) / (a + 1))
def isTransformC (a b : ℝ) : Prop := (a ≠ 0) → (b / a = (a * b) / (a^2))
def isTransformD (a b : ℝ) : Prop := (a ≠ 0) → ((-b + 1) / a = -(b + 1) / a)

-- Main theorem to assert the correctness of the transformations
theorem correctOptionOnlyC (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  ¬isTransformA a b ∧ ¬isTransformB a b ∧ isTransformC a b ∧ ¬isTransformD a b :=
by
  sorry

end NUMINAMATH_GPT_correctOptionOnlyC_l2259_225931


namespace NUMINAMATH_GPT_night_crew_worker_fraction_l2259_225971

noncomputable def box_fraction_day : ℝ := 5/7

theorem night_crew_worker_fraction
  (D N : ℝ) -- Number of workers in day and night crew
  (B : ℝ)  -- Number of boxes each worker in the day crew loads
  (H1 : ∀ day_boxes_loaded : ℝ, day_boxes_loaded = D * B)
  (H2 : ∀ night_boxes_loaded : ℝ, night_boxes_loaded = N * (B / 2))
  (H3 : (D * B) / ((D * B) + (N * (B / 2))) = box_fraction_day) :
  N / D = 4/5 := 
sorry

end NUMINAMATH_GPT_night_crew_worker_fraction_l2259_225971


namespace NUMINAMATH_GPT_coplanar_iff_m_eq_neg_8_l2259_225910

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)
variable (m : ℝ)

theorem coplanar_iff_m_eq_neg_8 
  (h : 4 • A - 3 • B + 7 • C + m • D = 0) : m = -8 ↔ ∃ a b c d : ℝ, a + b + c + d = 0 ∧ a • A + b • B + c • C + d • D = 0 :=
by
  sorry

end NUMINAMATH_GPT_coplanar_iff_m_eq_neg_8_l2259_225910


namespace NUMINAMATH_GPT_length_of_AB_l2259_225922

-- Define the distances given as conditions
def AC : ℝ := 5
def BD : ℝ := 6
def CD : ℝ := 3

-- Define the linear relationship of points A, B, C, D on the line
def points_on_line_in_order := true -- This is just a placeholder

-- Main theorem to prove
theorem length_of_AB : AB = 2 :=
by
  -- Apply the conditions and the linear relationships
  have BC : ℝ := BD - CD
  have AB : ℝ := AC - BC
  -- This would contain the actual proof using steps, but we skip it here
  sorry

end NUMINAMATH_GPT_length_of_AB_l2259_225922


namespace NUMINAMATH_GPT_second_layer_ratio_l2259_225998

theorem second_layer_ratio
  (first_layer_sugar third_layer_sugar : ℕ)
  (third_layer_factor : ℕ)
  (h1 : first_layer_sugar = 2)
  (h2 : third_layer_sugar = 12)
  (h3 : third_layer_factor = 3) :
  third_layer_sugar = third_layer_factor * (2 * first_layer_sugar) →
  second_layer_factor = 2 :=
by
  sorry

end NUMINAMATH_GPT_second_layer_ratio_l2259_225998


namespace NUMINAMATH_GPT_right_triangle_of_ratio_and_right_angle_l2259_225902

-- Define the sides and the right angle condition based on the problem conditions
variable (x : ℝ) (hx : 0 < x)

-- Variables for the sides in the given ratio
def a := 3 * x
def b := 4 * x
def c := 5 * x

-- The proposition we need to prove
theorem right_triangle_of_ratio_and_right_angle (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by sorry  -- Proof not required as per instructions

end NUMINAMATH_GPT_right_triangle_of_ratio_and_right_angle_l2259_225902


namespace NUMINAMATH_GPT_rectangular_park_length_l2259_225994

noncomputable def length_of_rectangular_park
  (P : ℕ) (B : ℕ) (L : ℕ) : Prop :=
  (P = 1000) ∧ (B = 200) ∧ (P = 2 * (L + B)) → (L = 300)

theorem rectangular_park_length : length_of_rectangular_park 1000 200 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangular_park_length_l2259_225994


namespace NUMINAMATH_GPT_book_arrangement_l2259_225915

theorem book_arrangement : (Nat.choose 7 3 = 35) :=
by
  sorry

end NUMINAMATH_GPT_book_arrangement_l2259_225915


namespace NUMINAMATH_GPT_John_pushup_count_l2259_225909

-- Definitions arising from conditions
def Zachary_pushups : ℕ := 51
def David_pushups : ℕ := Zachary_pushups + 22
def John_pushups : ℕ := David_pushups - 4

-- Theorem statement
theorem John_pushup_count : John_pushups = 69 := 
by 
  sorry

end NUMINAMATH_GPT_John_pushup_count_l2259_225909


namespace NUMINAMATH_GPT_digit_ends_with_l2259_225906

theorem digit_ends_with (z : ℕ) (h : z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) :
  ∀ (k : ℕ), k ≥ 1 → ∃ (n : ℕ), n ≥ 1 ∧ (∃ m : ℕ, (n ^ 9) % (10 ^ k) = z * (10 ^ m)) :=
by
  sorry

end NUMINAMATH_GPT_digit_ends_with_l2259_225906


namespace NUMINAMATH_GPT_revenue_times_l2259_225908

noncomputable def revenue_ratio (D : ℝ) : ℝ :=
  let revenue_Nov := (2 / 5) * D
  let revenue_Jan := (1 / 3) * revenue_Nov
  let average := (revenue_Nov + revenue_Jan) / 2
  D / average

theorem revenue_times (D : ℝ) (hD : D ≠ 0) : revenue_ratio D = 3.75 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_revenue_times_l2259_225908


namespace NUMINAMATH_GPT_subtract_value_l2259_225978

theorem subtract_value (N x : ℤ) (h1 : (N - x) / 7 = 7) (h2 : (N - 6) / 8 = 6) : x = 5 := 
by 
  sorry

end NUMINAMATH_GPT_subtract_value_l2259_225978


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l2259_225956

theorem neither_necessary_nor_sufficient (x : ℝ) : 
  ¬ ((x = 0) ↔ (x^2 - 2 * x = 0) ∧ (x ≠ 0 → x^2 - 2 * x ≠ 0) ∧ (x = 0 → x^2 - 2 * x = 0)) := 
sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l2259_225956


namespace NUMINAMATH_GPT_max_surface_area_of_cut_l2259_225911

noncomputable def max_sum_surface_areas (l w h : ℝ) : ℝ :=
  if l = 5 ∧ w = 4 ∧ h = 3 then 144 else 0

theorem max_surface_area_of_cut (l w h : ℝ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) : 
  max_sum_surface_areas l w h = 144 :=
by 
  rw [max_sum_surface_areas, if_pos]
  exact ⟨h_l, h_w, h_h⟩

end NUMINAMATH_GPT_max_surface_area_of_cut_l2259_225911


namespace NUMINAMATH_GPT_smaller_cubes_total_l2259_225953

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end NUMINAMATH_GPT_smaller_cubes_total_l2259_225953


namespace NUMINAMATH_GPT_minimum_cards_to_draw_to_ensure_2_of_each_suit_l2259_225917

noncomputable def min_cards_to_draw {total_cards : ℕ} {suit_count : ℕ} {cards_per_suit : ℕ} {joker_count : ℕ}
  (h_total : total_cards = 54)
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : ℕ :=
  43

theorem minimum_cards_to_draw_to_ensure_2_of_each_suit 
  (total_cards suit_count cards_per_suit joker_count : ℕ)
  (h_total : total_cards = 54) 
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : 
  min_cards_to_draw h_total h_suits h_cards_per_suit h_jokers = 43 :=
  by
  sorry

end NUMINAMATH_GPT_minimum_cards_to_draw_to_ensure_2_of_each_suit_l2259_225917


namespace NUMINAMATH_GPT_bagels_count_l2259_225932

def total_items : ℕ := 90
def bread_rolls : ℕ := 49
def croissants : ℕ := 19

def bagels : ℕ := total_items - (bread_rolls + croissants)

theorem bagels_count : bagels = 22 :=
by
  sorry

end NUMINAMATH_GPT_bagels_count_l2259_225932


namespace NUMINAMATH_GPT_functional_eq_implies_odd_l2259_225979

variable (f : ℝ → ℝ)

def condition (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x * f y) = y * f x

theorem functional_eq_implies_odd (h : condition f) : ∀ x : ℝ, f (-x) = -f x :=
sorry

end NUMINAMATH_GPT_functional_eq_implies_odd_l2259_225979


namespace NUMINAMATH_GPT_solve_equation_l2259_225997

theorem solve_equation (x : ℝ) : 
  (x ^ (Real.log x / Real.log 2) = x^5 / 32) ↔ (x = 2^((5 + Real.sqrt 5) / 2) ∨ x = 2^((5 - Real.sqrt 5) / 2)) := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l2259_225997


namespace NUMINAMATH_GPT_abc_divisibility_l2259_225981

theorem abc_divisibility (a b c : Nat) (h1 : a^3 ∣ b) (h2 : b^3 ∣ c) (h3 : c^3 ∣ a) :
  ∃ k : Nat, (a + b + c)^13 = k * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_abc_divisibility_l2259_225981


namespace NUMINAMATH_GPT_x_intercept_of_line_l2259_225918

theorem x_intercept_of_line : ∀ x y : ℝ, 2 * x + 3 * y = 6 → y = 0 → x = 3 :=
by
  intros x y h_line h_y_zero
  sorry

end NUMINAMATH_GPT_x_intercept_of_line_l2259_225918


namespace NUMINAMATH_GPT_carpet_area_l2259_225936

-- Definitions
def Rectangle1 (length1 width1 : ℕ) : Prop :=
  length1 = 12 ∧ width1 = 9

def Rectangle2 (length2 width2 : ℕ) : Prop :=
  length2 = 6 ∧ width2 = 9

def feet_to_yards (feet : ℕ) : ℕ :=
  feet / 3

-- Statement to prove
theorem carpet_area (length1 width1 length2 width2 : ℕ) (h1 : Rectangle1 length1 width1) (h2 : Rectangle2 length2 width2) :
  feet_to_yards (length1 * width1) / 3 + feet_to_yards (length2 * width2) / 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_carpet_area_l2259_225936


namespace NUMINAMATH_GPT_only_setB_is_proportional_l2259_225921

-- Definitions for the line segments
def setA := (3, 4, 5, 6)
def setB := (5, 15, 2, 6)
def setC := (4, 8, 3, 5)
def setD := (8, 4, 1, 3)

-- Definition to check if a set of line segments is proportional
def is_proportional (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c, d) := s
  a * d = b * c

-- Theorem proving that the only proportional set is set B
theorem only_setB_is_proportional :
  is_proportional setA = false ∧
  is_proportional setB = true ∧
  is_proportional setC = false ∧
  is_proportional setD = false :=
by
  sorry

end NUMINAMATH_GPT_only_setB_is_proportional_l2259_225921


namespace NUMINAMATH_GPT_true_proposition_p_and_q_l2259_225968

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Define the proposition q
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- Statement to prove the conjunction p ∧ q
theorem true_proposition_p_and_q : p ∧ q := 
by 
    sorry

end NUMINAMATH_GPT_true_proposition_p_and_q_l2259_225968


namespace NUMINAMATH_GPT_number_of_routes_from_P_to_Q_is_3_l2259_225927

-- Definitions of the nodes and paths
inductive Node
| P | Q | R | S | T | U | V
deriving DecidableEq, Repr

-- Definition of paths between nodes based on given conditions
def leads_to : Node → Node → Prop
| Node.P, Node.R => True
| Node.P, Node.S => True
| Node.R, Node.T => True
| Node.R, Node.U => True
| Node.S, Node.Q => True
| Node.T, Node.Q => True
| Node.U, Node.V => True
| Node.V, Node.Q => True
| _, _ => False

-- Proof statement: the number of different routes from P to Q
theorem number_of_routes_from_P_to_Q_is_3 : 
  ∃ (n : ℕ), n = 3 ∧ (∀ (route_count : ℕ), route_count = n → 
  ((leads_to Node.P Node.R ∧ leads_to Node.R Node.T ∧ leads_to Node.T Node.Q) ∨ 
   (leads_to Node.P Node.R ∧ leads_to Node.R Node.U ∧ leads_to Node.U Node.V ∧ leads_to Node.V Node.Q) ∨
   (leads_to Node.P Node.S ∧ leads_to Node.S Node.Q))) :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_number_of_routes_from_P_to_Q_is_3_l2259_225927


namespace NUMINAMATH_GPT_pizza_boxes_sold_l2259_225986

variables (P : ℕ) -- Representing the number of pizza boxes sold

def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2

def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def goal_amount : ℝ := 500
def more_needed : ℝ := 258
def current_amount : ℝ := goal_amount - more_needed

-- Total earnings calculation
def total_earnings : ℝ := (P : ℝ) * pizza_price + fries_sold * fries_price + soda_sold * soda_price

theorem pizza_boxes_sold (h : total_earnings P = current_amount) : P = 15 := 
by
  sorry

end NUMINAMATH_GPT_pizza_boxes_sold_l2259_225986


namespace NUMINAMATH_GPT__l2259_225937

-- Define the notion of opposite (additive inverse) of a number
def opposite (n : Int) : Int :=
  -n

-- State the theorem that the opposite of -5 is 5
example : opposite (-5) = 5 := by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT__l2259_225937


namespace NUMINAMATH_GPT_taxi_ride_cost_l2259_225967

namespace TaxiFare

def baseFare : ℝ := 2.00
def costPerMile : ℝ := 0.30
def taxRate : ℝ := 0.10
def distance : ℝ := 8.0

theorem taxi_ride_cost :
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  total_fare = 4.84 := by
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  sorry

end TaxiFare

end NUMINAMATH_GPT_taxi_ride_cost_l2259_225967


namespace NUMINAMATH_GPT_find_other_man_age_l2259_225995

variable (avg_age_men inc_age_man other_man_age avg_age_women total_age_increase : ℕ)

theorem find_other_man_age 
    (h1 : inc_age_man = 2) 
    (h2 : ∀ m, m = 8 * (avg_age_men + inc_age_man))
    (h3 : ∃ y, y = 22) 
    (h4 : ∀ w, w = 29) 
    (h5 : total_age_increase = 2 * avg_age_women - (22 + other_man_age)) :
  total_age_increase = 16 → other_man_age = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_other_man_age_l2259_225995


namespace NUMINAMATH_GPT_units_digit_sum_l2259_225925

def base8_to_base10 (n : Nat) : Nat :=
  let units := n % 10
  let tens := (n / 10) % 10
  tens * 8 + units

theorem units_digit_sum (n1 n2 : Nat) (h1 : n1 = 45) (h2 : n2 = 67) : ((base8_to_base10 n1) + (base8_to_base10 n2)) % 8 = 4 := by
  sorry

end NUMINAMATH_GPT_units_digit_sum_l2259_225925


namespace NUMINAMATH_GPT_product_of_b_values_is_neg_12_l2259_225974

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end NUMINAMATH_GPT_product_of_b_values_is_neg_12_l2259_225974


namespace NUMINAMATH_GPT_quadrilateral_angles_arith_prog_l2259_225988

theorem quadrilateral_angles_arith_prog {x a b c : ℕ} (d : ℝ):
  (x^2 = 8^2 + 7^2 + 2 * 8 * 7 * Real.sin (3 * d)) →
  x = a + Real.sqrt b + Real.sqrt c →
  x = Real.sqrt 113 →
  a + b + c = 113 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_angles_arith_prog_l2259_225988


namespace NUMINAMATH_GPT_average_mileage_first_car_l2259_225987

theorem average_mileage_first_car (X Y : ℝ) 
  (h1 : X + Y = 75) 
  (h2 : 25 * X + 35 * Y = 2275) : 
  X = 35 :=
by 
  sorry

end NUMINAMATH_GPT_average_mileage_first_car_l2259_225987


namespace NUMINAMATH_GPT_min_value_a_sq_plus_b_sq_l2259_225930

theorem min_value_a_sq_plus_b_sq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x > 0 → y > 0 → (x - 1)^3 + (y - 1)^3 ≥ 3 * (2 - x - y) → x^2 + y^2 ≥ m) :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_sq_plus_b_sq_l2259_225930


namespace NUMINAMATH_GPT_solution_l2259_225977

theorem solution (t : ℝ) :
  let x := 3 * t
  let y := t
  let z := 0
  x^2 - 9 * y^2 = z^2 :=
by
  sorry

end NUMINAMATH_GPT_solution_l2259_225977


namespace NUMINAMATH_GPT_farmer_cows_more_than_goats_l2259_225939

-- Definitions of the variables
variables (C P G x : ℕ)

-- Conditions given in the problem
def twice_as_many_pigs_as_cows : Prop := P = 2 * C
def more_cows_than_goats : Prop := C = G + x
def goats_count : Prop := G = 11
def total_animals : Prop := C + P + G = 56

-- The theorem to prove
theorem farmer_cows_more_than_goats
  (h1 : twice_as_many_pigs_as_cows C P)
  (h2 : more_cows_than_goats C G x)
  (h3 : goats_count G)
  (h4 : total_animals C P G) :
  C - G = 4 :=
sorry

end NUMINAMATH_GPT_farmer_cows_more_than_goats_l2259_225939


namespace NUMINAMATH_GPT_value_of_a7_l2259_225913

-- Define the geometric sequence and its properties
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the conditions of the problem
variables (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n : ℕ, a n > 0) (h_product : a 3 * a 11 = 16)

-- Conjecture that we aim to prove
theorem value_of_a7 : a 7 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a7_l2259_225913


namespace NUMINAMATH_GPT_find_x_collinear_l2259_225964

-- Given vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -3)
def vec_c (x : ℝ) : ℝ × ℝ := (-2, x)

-- Definition of vectors being collinear
def collinear (v₁ v₂ : ℝ × ℝ) : Prop :=
∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Question: What is the value of x such that vec_a + vec_b is collinear with vec_c(x)?
theorem find_x_collinear : ∃ x : ℝ, collinear (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_c x) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_collinear_l2259_225964


namespace NUMINAMATH_GPT_S4k_eq_32_l2259_225976

-- Definition of the problem conditions
variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (k : ℕ)

-- Conditions: Arithmetic sequence sum properties
axiom sum_arithmetic_sequence : ∀ {n : ℕ}, S n = n * (a 1 + a n) / 2

-- Given conditions
axiom Sk_eq_2 : S k = 2
axiom S3k_eq_18 : S (3 * k) = 18

-- Prove the required statement
theorem S4k_eq_32 : S (4 * k) = 32 :=
by
  sorry

end NUMINAMATH_GPT_S4k_eq_32_l2259_225976


namespace NUMINAMATH_GPT_tank_filling_time_l2259_225916

theorem tank_filling_time
  (T : ℕ) (Rₐ R_b R_c : ℕ) (C : ℕ)
  (hRₐ : Rₐ = 40) (hR_b : R_b = 30) (hR_c : R_c = 20) (hC : C = 950)
  (h_cycle : T = 1 + 1 + 1) : 
  T * (C / (Rₐ + R_b - R_c)) - 1 = 56 :=
by
  sorry

end NUMINAMATH_GPT_tank_filling_time_l2259_225916


namespace NUMINAMATH_GPT_calculate_share_A_l2259_225951

-- Defining the investments
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def investment_D : ℕ := 13000
def investment_E : ℕ := 21000
def investment_F : ℕ := 15000
def investment_G : ℕ := 9000

-- Defining B's share
def share_B : ℚ := 3600

-- Function to calculate total investment
def total_investment : ℕ :=
  investment_A + investment_B + investment_C + investment_D + investment_E + investment_F + investment_G

-- Ratio of B's investment to total investment
def ratio_B : ℚ :=
  investment_B / total_investment

-- Calculate total profit using B's share and ratio
def total_profit : ℚ :=
  share_B / ratio_B

-- Ratio of A's investment to total investment
def ratio_A : ℚ :=
  investment_A / total_investment

-- Calculate A's share based on the total profit
def share_A : ℚ :=
  total_profit * ratio_A

-- The theorem to prove the share of A is approximately $2292.34
theorem calculate_share_A : 
  abs (share_A - 2292.34) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_calculate_share_A_l2259_225951


namespace NUMINAMATH_GPT_f_diff_l2259_225912

def f (n : ℕ) : ℚ := (1 / 3 : ℚ) * n * (n + 1) * (n + 2)

theorem f_diff (r : ℕ) : f r - f (r - 1) = r * (r + 1) := 
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_f_diff_l2259_225912


namespace NUMINAMATH_GPT_consecutive_integers_eq_l2259_225975

theorem consecutive_integers_eq (a b c d e: ℕ) (h1: b = a + 1) (h2: c = a + 2) (h3: d = a + 3) (h4: e = a + 4) (h5: a^2 + b^2 + c^2 = d^2 + e^2) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_eq_l2259_225975


namespace NUMINAMATH_GPT_sandy_nickels_remaining_l2259_225928

def original_nickels : ℕ := 31
def nickels_borrowed : ℕ := 20

theorem sandy_nickels_remaining : (original_nickels - nickels_borrowed) = 11 :=
by
  sorry

end NUMINAMATH_GPT_sandy_nickels_remaining_l2259_225928


namespace NUMINAMATH_GPT_douglas_percent_votes_l2259_225955

def percentageOfTotalVotesWon (votes_X votes_Y: ℕ) (percent_X percent_Y: ℕ) : ℕ :=
  let total_votes_Douglas : ℕ := (percent_X * 2 * votes_X + percent_Y * votes_Y)
  let total_votes_cast : ℕ := 3 * votes_Y
  (total_votes_Douglas * 100 / total_votes_cast)

theorem douglas_percent_votes (votes_X votes_Y : ℕ) (h_ratio : 2 * votes_X = votes_Y)
  (h_perc_X : percent_X = 64)
  (h_perc_Y : percent_Y = 46) :
  percentageOfTotalVotesWon votes_X votes_Y 64 46 = 58 := by
    sorry

end NUMINAMATH_GPT_douglas_percent_votes_l2259_225955


namespace NUMINAMATH_GPT_numberOfWaysToChooseLeadershipStructure_correct_l2259_225900

noncomputable def numberOfWaysToChooseLeadershipStructure : ℕ :=
  12 * 11 * 10 * Nat.choose 9 3 * Nat.choose 6 3

theorem numberOfWaysToChooseLeadershipStructure_correct :
  numberOfWaysToChooseLeadershipStructure = 221760 :=
by
  simp [numberOfWaysToChooseLeadershipStructure]
  -- Add detailed simplification/proof steps here if required
  sorry

end NUMINAMATH_GPT_numberOfWaysToChooseLeadershipStructure_correct_l2259_225900


namespace NUMINAMATH_GPT_pots_on_each_shelf_l2259_225945

variable (x : ℕ)
variable (h1 : 4 * 3 * x = 60)

theorem pots_on_each_shelf : x = 5 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_pots_on_each_shelf_l2259_225945


namespace NUMINAMATH_GPT_total_daisies_l2259_225905

-- Define the initial conditions
def white_daisies : Nat := 6
def pink_daisies : Nat := 9 * white_daisies
def red_daisies : Nat := 4 * pink_daisies - 3

-- The main theorem stating that the total number of daisies is 273
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  -- The proof is left as an exercise
  sorry

end NUMINAMATH_GPT_total_daisies_l2259_225905


namespace NUMINAMATH_GPT_find_a_minus_b_l2259_225943

theorem find_a_minus_b (a b x y : ℤ)
  (h_x : x = 1)
  (h_y : y = 1)
  (h1 : a * x + b * y = 2)
  (h2 : x - b * y = 3) :
  a - b = 6 := by
  subst h_x
  subst h_y
  simp at h1 h2
  have h_b: b = -2 := by linarith
  have h_a: a = 4 := by linarith
  rw [h_a, h_b]
  norm_num

end NUMINAMATH_GPT_find_a_minus_b_l2259_225943


namespace NUMINAMATH_GPT_hazel_sold_18_cups_to_kids_l2259_225942

theorem hazel_sold_18_cups_to_kids:
  ∀ (total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup: ℕ),
     total_cups = 56 →
     cups_sold_construction = 28 →
     crew_remaining = total_cups - cups_sold_construction →
     last_cup = 1 →
     crew_remaining = cups_sold_kids + (cups_sold_kids / 2) + last_cup →
     cups_sold_kids = 18 :=
by
  intros total_cups cups_sold_construction crew_remaining cups_sold_kids cups_given_away last_cup h_total h_construction h_remaining h_last h_equation
  sorry

end NUMINAMATH_GPT_hazel_sold_18_cups_to_kids_l2259_225942


namespace NUMINAMATH_GPT_annie_milkshakes_l2259_225901

theorem annie_milkshakes
  (A : ℕ) (C_hamburger : ℕ) (C_milkshake : ℕ) (H : ℕ) (L : ℕ)
  (initial_money : A = 120)
  (hamburger_cost : C_hamburger = 4)
  (milkshake_cost : C_milkshake = 3)
  (hamburgers_bought : H = 8)
  (money_left : L = 70) :
  ∃ (M : ℕ), A - H * C_hamburger - M * C_milkshake = L ∧ M = 6 :=
by
  sorry

end NUMINAMATH_GPT_annie_milkshakes_l2259_225901


namespace NUMINAMATH_GPT_ratio_c_div_d_l2259_225962

theorem ratio_c_div_d (a b d : ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : d = 0.05 * a) (c : ℝ) (h4 : c = b / a) : c / d = 1 / 320 := 
sorry

end NUMINAMATH_GPT_ratio_c_div_d_l2259_225962


namespace NUMINAMATH_GPT_solve_trig_equation_l2259_225947

open Real

theorem solve_trig_equation (k : ℕ) :
    (∀ x, 8.459 * cos x^2 * cos (x^2) * (tan (x^2) + 2 * tan x) + tan x^3 * (1 - sin (x^2)^2) * (2 - tan x * tan (x^2)) = 0) ↔
    (∃ k : ℕ, x = -1 + sqrt (π * k + 1) ∨ x = -1 - sqrt (π * k + 1)) :=
sorry

end NUMINAMATH_GPT_solve_trig_equation_l2259_225947


namespace NUMINAMATH_GPT_convex_polygon_sides_l2259_225929

theorem convex_polygon_sides (S : ℝ) (n : ℕ) (a₁ a₂ a₃ a₄ : ℝ) 
    (h₁ : S = 4320) 
    (h₂ : a₁ = 120) 
    (h₃ : a₂ = 120) 
    (h₄ : a₃ = 120) 
    (h₅ : a₄ = 120) 
    (h_sum : S = 180 * (n - 2)) :
    n = 26 :=
by
  sorry

end NUMINAMATH_GPT_convex_polygon_sides_l2259_225929


namespace NUMINAMATH_GPT_james_units_per_semester_l2259_225990

theorem james_units_per_semester
  (cost_per_unit : ℕ)
  (total_cost : ℕ)
  (num_semesters : ℕ)
  (payment_per_semester : ℕ)
  (units_per_semester : ℕ)
  (H1 : cost_per_unit = 50)
  (H2 : total_cost = 2000)
  (H3 : num_semesters = 2)
  (H4 : payment_per_semester = total_cost / num_semesters)
  (H5 : units_per_semester = payment_per_semester / cost_per_unit) :
  units_per_semester = 20 :=
sorry

end NUMINAMATH_GPT_james_units_per_semester_l2259_225990


namespace NUMINAMATH_GPT_population_correct_individual_correct_sample_correct_sample_size_correct_l2259_225983

-- Definitions based on the problem conditions
def Population : Type := {s : String // s = "all seventh-grade students in the city"}
def Individual : Type := {s : String // s = "each seventh-grade student in the city"}
def Sample : Type := {s : String // s = "the 500 students that were drawn"}
def SampleSize : ℕ := 500

-- Prove given conditions
theorem population_correct (p : Population) : p.1 = "all seventh-grade students in the city" :=
by sorry

theorem individual_correct (i : Individual) : i.1 = "each seventh-grade student in the city" :=
by sorry

theorem sample_correct (s : Sample) : s.1 = "the 500 students that were drawn" :=
by sorry

theorem sample_size_correct : SampleSize = 500 :=
by sorry

end NUMINAMATH_GPT_population_correct_individual_correct_sample_correct_sample_size_correct_l2259_225983


namespace NUMINAMATH_GPT_sequence_bounds_l2259_225924

theorem sequence_bounds :
    ∀ (a : ℕ → ℝ), a 0 = 5 → (∀ n : ℕ, a (n + 1) = a n + 1 / a n) → 45 < a 1000 ∧ a 1000 < 45.1 :=
by
  intros a h0 h_rec
  sorry

end NUMINAMATH_GPT_sequence_bounds_l2259_225924


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2259_225970

theorem distance_between_foci_of_hyperbola :
  ∀ x y : ℝ, (x^2 - 8 * x - 16 * y^2 - 16 * y = 48) → (∃ c : ℝ, 2 * c = 2 * Real.sqrt 63.75) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2259_225970


namespace NUMINAMATH_GPT_energy_stick_difference_l2259_225933

variable (B D : ℕ)

theorem energy_stick_difference (h1 : B = D + 17) : 
  let B' := B - 3
  let D' := D + 3
  D' < B' →
  (B' - D') = 11 :=
by
  sorry

end NUMINAMATH_GPT_energy_stick_difference_l2259_225933


namespace NUMINAMATH_GPT_final_bug_population_is_zero_l2259_225999

def initial_population := 400
def spiders := 12
def spider_consumption := 7
def ladybugs := 5
def ladybug_consumption := 6
def mantises := 8
def mantis_consumption := 4

def day1_population := initial_population * 80 / 100

def predators_consumption_day := (spiders * spider_consumption) +
                                 (ladybugs * ladybug_consumption) +
                                 (mantises * mantis_consumption)

def day2_population := day1_population - predators_consumption_day
def day3_population := day2_population - predators_consumption_day
def day4_population := max 0 (day3_population - predators_consumption_day)
def day5_population := max 0 (day4_population - predators_consumption_day)
def day6_population := max 0 (day5_population - predators_consumption_day)

def day7_population := day6_population * 70 / 100

theorem final_bug_population_is_zero: 
  day7_population = 0 :=
  by
  sorry

end NUMINAMATH_GPT_final_bug_population_is_zero_l2259_225999


namespace NUMINAMATH_GPT_zero_product_property_l2259_225984

theorem zero_product_property {a b : ℝ} (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end NUMINAMATH_GPT_zero_product_property_l2259_225984


namespace NUMINAMATH_GPT_smallest_area_of_square_containing_rectangles_l2259_225972

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end NUMINAMATH_GPT_smallest_area_of_square_containing_rectangles_l2259_225972


namespace NUMINAMATH_GPT_find_bicycle_speed_l2259_225941

def distanceAB := 40 -- Distance from A to B in km
def speed_walk := 6 -- Speed of the walking tourist in km/h
def distance_ahead := 5 -- Distance by which the second tourist is ahead initially in km
def speed_car := 24 -- Speed of the car in km/h
def meeting_time := 2 -- Time after departure when they meet in hours

theorem find_bicycle_speed (v : ℝ) : 
  (distanceAB = 40 ∧ speed_walk = 6 ∧ distance_ahead = 5 ∧ speed_car = 24 ∧ meeting_time = 2) →
  (v = 9) :=
by 
sorry

end NUMINAMATH_GPT_find_bicycle_speed_l2259_225941


namespace NUMINAMATH_GPT_angle_bisector_inequality_l2259_225966

theorem angle_bisector_inequality
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_perimeter : (x + y + z) = 6) :
  (1 / x^2) + (1 / y^2) + (1 / z^2) ≥ 1 := by
  sorry

end NUMINAMATH_GPT_angle_bisector_inequality_l2259_225966


namespace NUMINAMATH_GPT_problem_statement_l2259_225934

def Delta (a b : ℝ) : ℝ := a^2 - b

theorem problem_statement : Delta (2 ^ (Delta 5 8)) (4 ^ (Delta 2 7)) = 17179869183.984375 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2259_225934


namespace NUMINAMATH_GPT_haley_cider_pints_l2259_225920

noncomputable def apples_per_farmhand_per_hour := 240
noncomputable def working_hours := 5
noncomputable def total_farmhands := 6

noncomputable def golden_delicious_per_pint := 20
noncomputable def pink_lady_per_pint := 40
noncomputable def golden_delicious_ratio := 1
noncomputable def pink_lady_ratio := 2

noncomputable def total_apples := total_farmhands * apples_per_farmhand_per_hour * working_hours
noncomputable def total_parts := golden_delicious_ratio + pink_lady_ratio

noncomputable def golden_delicious_apples := total_apples / total_parts
noncomputable def pink_lady_apples := golden_delicious_apples * pink_lady_ratio

noncomputable def pints_golden_delicious := golden_delicious_apples / golden_delicious_per_pint
noncomputable def pints_pink_lady := pink_lady_apples / pink_lady_per_pint

theorem haley_cider_pints : 
  total_apples = 7200 → 
  golden_delicious_apples = 2400 → 
  pink_lady_apples = 4800 → 
  pints_golden_delicious = 120 → 
  pints_pink_lady = 120 → 
  pints_golden_delicious = pints_pink_lady →
  pints_golden_delicious = 120 :=
by
  sorry

end NUMINAMATH_GPT_haley_cider_pints_l2259_225920


namespace NUMINAMATH_GPT_max_AMC_expression_l2259_225961

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 15) : A * M * C + A * M + M * C + C * A ≤ 200 :=
by
  sorry

end NUMINAMATH_GPT_max_AMC_expression_l2259_225961
