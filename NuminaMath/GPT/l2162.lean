import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_properties_l2162_216217

-- Define the first term and common ratio
def first_term : ℕ := 12
def common_ratio : ℚ := 1/2

-- Define the formula for the n-th term of the geometric sequence
def nth_term (a : ℕ) (r : ℚ) (n : ℕ) := a * r^(n-1)

-- The 8th term in the sequence
def term_8 := nth_term first_term common_ratio 8

-- Half of the 8th term
def half_term_8 := (1/2) * term_8

-- Prove that the 8th term is 3/32 and half of the 8th term is 3/64
theorem geometric_sequence_properties : 
  (term_8 = (3/32)) ∧ (half_term_8 = (3/64)) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l2162_216217


namespace NUMINAMATH_GPT_water_cost_is_1_l2162_216279

-- Define the conditions
def cost_cola : ℝ := 3
def cost_juice : ℝ := 1.5
def bottles_sold_cola : ℝ := 15
def bottles_sold_juice : ℝ := 12
def bottles_sold_water : ℝ := 25
def total_revenue : ℝ := 88

-- Compute the revenue from cola and juice
def revenue_cola : ℝ := bottles_sold_cola * cost_cola
def revenue_juice : ℝ := bottles_sold_juice * cost_juice

-- Define a proof that the cost of a bottle of water is $1
theorem water_cost_is_1 : (total_revenue - revenue_cola - revenue_juice) / bottles_sold_water = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_water_cost_is_1_l2162_216279


namespace NUMINAMATH_GPT_present_age_of_son_l2162_216274

variable (S M : ℕ)

-- Conditions
def age_difference : Prop := M = S + 40
def age_relation_in_seven_years : Prop := M + 7 = 3 * (S + 7)

-- Theorem to prove
theorem present_age_of_son : age_difference S M → age_relation_in_seven_years S M → S = 13 := by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l2162_216274


namespace NUMINAMATH_GPT_limonia_largest_unachievable_l2162_216204

noncomputable def largest_unachievable_amount (n : ℕ) : ℕ :=
  12 * n^2 + 14 * n - 1

theorem limonia_largest_unachievable (n : ℕ) :
  ∀ k, ¬ ∃ a b c d : ℕ, 
    k = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10) 
    → k = largest_unachievable_amount n :=
sorry

end NUMINAMATH_GPT_limonia_largest_unachievable_l2162_216204


namespace NUMINAMATH_GPT_problem_inequality_l2162_216208

theorem problem_inequality (n a b : ℕ) (h₁ : n ≥ 2) 
  (h₂ : ∀ m, 2^m ∣ 5^n - 3^n → m ≤ a) 
  (h₃ : ∀ m, 2^m ≤ n → m ≤ b) : a ≤ b + 3 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l2162_216208


namespace NUMINAMATH_GPT_simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l2162_216228

def A (a b : ℝ) := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) := 4 * a^2 + 6 * a * b + 8 * a

theorem simplify_2A_minus_B {a b : ℝ} :
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a :=
by
  sorry

theorem twoA_minusB_value_when_a_neg2_b_1 :
  2 * A (-2) (1) - B (-2) (1) = 54 :=
by
  sorry

theorem twoA_minusB_independent_of_a {b : ℝ} :
  (∀ a : ℝ, 2 * A a b - B a b = 6 * b - 8 * a) → b = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_2A_minus_B_twoA_minusB_value_when_a_neg2_b_1_twoA_minusB_independent_of_a_l2162_216228


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_is_three_fourths_l2162_216272

noncomputable def circle_diameter : ℝ := Real.sqrt 12

noncomputable def radius_of_new_inscribed_circle : ℝ :=
  let R := circle_diameter / 2
  let s := R * Real.sqrt 3
  let h := s * Real.sqrt 3 / 2
  let a := Real.sqrt (h^2 - (h/2)^2)
  a * Real.sqrt 3 / 6

theorem radius_of_inscribed_circle_is_three_fourths :
  radius_of_new_inscribed_circle = 3 / 4 := sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_is_three_fourths_l2162_216272


namespace NUMINAMATH_GPT_purely_imaginary_m_no_m_in_fourth_quadrant_l2162_216242

def z (m : ℝ) : ℂ := ⟨m^2 - 8 * m + 15, m^2 - 5 * m⟩

theorem purely_imaginary_m :
  (∀ m : ℝ, z m = ⟨0, m^2 - 5 * m⟩ ↔ m = 3) :=
by
  sorry

theorem no_m_in_fourth_quadrant :
  ¬ ∃ m : ℝ, (m^2 - 8 * m + 15 > 0) ∧ (m^2 - 5 * m < 0) :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_m_no_m_in_fourth_quadrant_l2162_216242


namespace NUMINAMATH_GPT_car_catch_up_distance_l2162_216213

-- Define the problem: Prove that Car A catches Car B at the specified location, given the conditions

theorem car_catch_up_distance
  (distance : ℝ := 300)
  (v_A v_B t_A : ℝ)
  (start_A_late : ℝ := 1) -- Car A leaves 1 hour later
  (arrive_A_early : ℝ := 1) -- Car A arrives 1 hour earlier
  (t : ℝ := 2) -- Time when Car A catches up with Car B
  (dist_eq_A : distance = v_A * t_A)
  (dist_eq_B : distance = v_B * (t_A + 2))
  (catch_up_time_eq : v_A * t = v_B * (t + 1)):
  (distance - v_A * t) = 150 := sorry

end NUMINAMATH_GPT_car_catch_up_distance_l2162_216213


namespace NUMINAMATH_GPT_shaded_area_correct_l2162_216296

-- Define the side lengths of the squares
def side_length_large_square : ℕ := 14
def side_length_small_square : ℕ := 10

-- Define the areas of the squares
def area_large_square : ℕ := side_length_large_square * side_length_large_square
def area_small_square : ℕ := side_length_small_square * side_length_small_square

-- Define the area of the shaded regions
def area_shaded_regions : ℕ := area_large_square - area_small_square

-- State the theorem
theorem shaded_area_correct : area_shaded_regions = 49 := by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l2162_216296


namespace NUMINAMATH_GPT_a5_gt_b5_l2162_216264

variables {a_n b_n : ℕ → ℝ}
variables {a1 b1 a3 b3 : ℝ}
variables {q : ℝ} {d : ℝ}

/-- Given conditions -/
axiom h1 : a1 = b1
axiom h2 : a1 > 0
axiom h3 : a3 = b3
axiom h4 : a3 = a1 * q^2
axiom h5 : b3 = a1 + 2 * d
axiom h6 : a1 ≠ a3

/-- Prove that a_5 is greater than b_5 -/
theorem a5_gt_b5 : a1 * q^4 > a1 + 4 * d :=
by sorry

end NUMINAMATH_GPT_a5_gt_b5_l2162_216264


namespace NUMINAMATH_GPT_orchard_trees_l2162_216290

theorem orchard_trees (n : ℕ) (hn : n^2 + 146 = 7890) : 
    n^2 + 146 + 31 = 89^2 := by
  sorry

end NUMINAMATH_GPT_orchard_trees_l2162_216290


namespace NUMINAMATH_GPT_find_y_l2162_216215

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end NUMINAMATH_GPT_find_y_l2162_216215


namespace NUMINAMATH_GPT_triangle_area_l2162_216233

-- Define a triangle as a structure with vertices A, B, and C, where the lengths AB, AC, and BC are provided
structure Triangle :=
  (A B C : ℝ)
  (AB AC BC : ℝ)
  (is_isosceles : AB = AC)
  (BC_length : BC = 20)
  (AB_length : AB = 26)

-- Define the length bisector and Pythagorean properties
def bisects_base (t : Triangle) : Prop :=
  ∃ D : ℝ, (t.B - D) = (D - t.C) ∧ 2 * D = t.B + t.C

def pythagorean_theorem_AD (t : Triangle) (D : ℝ) (AD : ℝ) : Prop :=
  t.AB^2 = AD^2 + (t.B - D)^2

-- State the problem as a theorem
theorem triangle_area (t : Triangle) (D : ℝ) (AD : ℝ) (h1 : bisects_base t) (h2 : pythagorean_theorem_AD t D AD) :
  AD = 24 ∧ (1 / 2) * t.BC * AD = 240 :=
sorry

end NUMINAMATH_GPT_triangle_area_l2162_216233


namespace NUMINAMATH_GPT_sandwiches_bought_l2162_216201

theorem sandwiches_bought (sandwich_cost soda_cost total_cost_sodas total_cost : ℝ)
  (h1 : sandwich_cost = 2.45)
  (h2 : soda_cost = 0.87)
  (h3 : total_cost_sodas = 4 * soda_cost)
  (h4 : total_cost = 8.38) :
  ∃ (S : ℕ), sandwich_cost * S + total_cost_sodas = total_cost ∧ S = 2 :=
by
  use 2
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_sandwiches_bought_l2162_216201


namespace NUMINAMATH_GPT_bear_population_l2162_216231

theorem bear_population (black_bears white_bears brown_bears total_bears : ℕ) 
(h1 : black_bears = 60)
(h2 : white_bears = black_bears / 2)
(h3 : brown_bears = black_bears + 40) :
total_bears = black_bears + white_bears + brown_bears :=
sorry

end NUMINAMATH_GPT_bear_population_l2162_216231


namespace NUMINAMATH_GPT_penny_identified_whales_l2162_216265

theorem penny_identified_whales (sharks eels total : ℕ)
  (h_sharks : sharks = 35)
  (h_eels   : eels = 15)
  (h_total  : total = 55) :
  total - (sharks + eels) = 5 :=
by
  sorry

end NUMINAMATH_GPT_penny_identified_whales_l2162_216265


namespace NUMINAMATH_GPT_max_value_of_expression_l2162_216289

theorem max_value_of_expression (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  2 * Real.sqrt (abc / 8) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l2162_216289


namespace NUMINAMATH_GPT_scientific_notation_216000_l2162_216211

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_scientific_notation_216000_l2162_216211


namespace NUMINAMATH_GPT_flowers_to_embroider_l2162_216216

-- Defining constants based on the problem conditions
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1
def total_minutes : ℕ := 1085

-- Theorem statement to prove the number of flowers Carolyn wants to embroider
theorem flowers_to_embroider : 
  (total_minutes * stitches_per_minute - (num_godzillas * stitches_per_godzilla + num_unicorns * stitches_per_unicorn)) / stitches_per_flower = 50 :=
by
  sorry

end NUMINAMATH_GPT_flowers_to_embroider_l2162_216216


namespace NUMINAMATH_GPT_math_problem_l2162_216270

theorem math_problem (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y + x * y = 1) :
  x * y + 1 / (x * y) - y / x - x / y = 4 :=
sorry

end NUMINAMATH_GPT_math_problem_l2162_216270


namespace NUMINAMATH_GPT_perfect_square_iff_n_eq_5_l2162_216243

theorem perfect_square_iff_n_eq_5 (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, n * 2^(n-1) + 1 = m^2 ↔ n = 5 := by
  sorry

end NUMINAMATH_GPT_perfect_square_iff_n_eq_5_l2162_216243


namespace NUMINAMATH_GPT_total_cost_of_mangoes_l2162_216297

-- Definition of prices per dozen in one box
def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Number of dozens per box (constant for all boxes)
def dozens_per_box : ℕ := 10

-- Number of boxes
def number_of_boxes : ℕ := 36

-- Calculate the total cost of mangoes in all boxes
theorem total_cost_of_mangoes :
  (prices_per_dozen.sum * number_of_boxes = 3060) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_cost_of_mangoes_l2162_216297


namespace NUMINAMATH_GPT_part1_part2_l2162_216218

open Real

-- Definitions used in the proof
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0 ∧ a > 0
def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

theorem part1 (x : ℝ) : (p 1 x ∧ q x) → 2 < x ∧ x ≤ 3 := by
  sorry

theorem part2 (a : ℝ) : (¬ (∃ x, p a x) → ¬ (∃ x, q x)) → a > 3 / 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2162_216218


namespace NUMINAMATH_GPT_volume_first_cube_l2162_216277

theorem volume_first_cube (a b : ℝ) (h_ratio : a = 3 * b) (h_volume : b^3 = 8) : a^3 = 216 :=
by
  sorry

end NUMINAMATH_GPT_volume_first_cube_l2162_216277


namespace NUMINAMATH_GPT_range_of_b_l2162_216249

theorem range_of_b (b : ℝ) (h : Real.sqrt ((b-2)^2) = 2 - b) : b ≤ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_b_l2162_216249


namespace NUMINAMATH_GPT_inverse_f_486_l2162_216222

-- Define the function f with given properties.
def f : ℝ → ℝ := sorry

-- Condition 1: f(5) = 2
axiom f_at_5 : f 5 = 2

-- Condition 2: f(3x) = 3f(x) for all x
axiom f_scale : ∀ x, f (3 * x) = 3 * f x

-- Proposition: f⁻¹(486) = 1215
theorem inverse_f_486 : (∃ x, f x = 486) → ∀ x, f x = 486 → x = 1215 :=
by sorry

end NUMINAMATH_GPT_inverse_f_486_l2162_216222


namespace NUMINAMATH_GPT_new_average_age_l2162_216241

theorem new_average_age (avg_age : ℕ) (num_students : ℕ) (teacher_age : ℕ) (new_num_individuals : ℕ) (new_avg_age : ℕ) :
  avg_age = 15 ∧ num_students = 20 ∧ teacher_age = 36 ∧ new_num_individuals = 21 →
  new_avg_age = (num_students * avg_age + teacher_age) / new_num_individuals → new_avg_age = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_new_average_age_l2162_216241


namespace NUMINAMATH_GPT_number_of_players_l2162_216219

/-- Jane bought 600 minnows, each prize has 3 minnows, 15% of the players win a prize, 
and 240 minnows are left over. To find the total number of players -/
theorem number_of_players (total_minnows left_over_minnows minnows_per_prize prizes_win_percent : ℕ) 
(h1 : total_minnows = 600) 
(h2 : minnows_per_prize = 3)
(h3 : prizes_win_percent * 100 = 15)
(h4 : left_over_minnows = 240) : 
total_minnows - left_over_minnows = 360 → 
  360 / minnows_per_prize = 120 → 
  (prizes_win_percent * 100 / 100) * P = 120 → 
  P = 800 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_players_l2162_216219


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2162_216212

/--
The speed of the stream is 6 kmph.
The boat can cover 48 km downstream or 32 km upstream in the same time.
We want to prove that the speed of the boat in still water is 30 kmph.
-/
theorem boat_speed_in_still_water (x : ℝ)
  (h1 : ∃ t : ℝ, t = 48 / (x + 6) ∧ t = 32 / (x - 6)) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2162_216212


namespace NUMINAMATH_GPT_min_value_fraction_l2162_216251

theorem min_value_fraction (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃ m, (∀ z, z = (1 / (x + 1) + 1 / y) → z ≥ m) ∧ m = (3 + 2 * Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l2162_216251


namespace NUMINAMATH_GPT_compare_nsquare_pow2_pos_int_l2162_216239

-- Proposition that captures the given properties of comparing n^2 and 2^n
theorem compare_nsquare_pow2_pos_int (n : ℕ) (hn : n > 0) : 
  (n = 1 → n^2 < 2^n) ∧
  (n = 2 → n^2 = 2^n) ∧
  (n = 3 → n^2 > 2^n) ∧
  (n = 4 → n^2 = 2^n) ∧
  (n ≥ 5 → n^2 < 2^n) :=
by
  sorry

end NUMINAMATH_GPT_compare_nsquare_pow2_pos_int_l2162_216239


namespace NUMINAMATH_GPT_arithmetic_sequence_equal_sum_l2162_216259

variable (a d : ℕ) -- defining first term and common difference as natural numbers
variable (n : ℕ) -- defining n as a natural number

noncomputable def sum_arithmetic_sequence (n: ℕ) (a d: ℕ): ℕ := (n * (2 * a + (n - 1) * d) ) / 2

theorem arithmetic_sequence_equal_sum (a d n : ℕ) :
  sum_arithmetic_sequence (10 * n) a d = sum_arithmetic_sequence (15 * n) a d - sum_arithmetic_sequence (10 * n) a d :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_equal_sum_l2162_216259


namespace NUMINAMATH_GPT_find_a_l2162_216223

theorem find_a (a : ℝ) (h₁ : ¬ (a = 0)) (h_perp : (∀ x y : ℝ, (a * x + 1 = 0) 
  -> (a - 2) * x + y + a = 0 -> ∀ x₁ y₁, (a * x₁ + 1 = 0) -> y = y₁)) : a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l2162_216223


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l2162_216253

theorem sum_of_solutions_eq_zero :
  let f (x : ℝ) := 2^|x| + 4 * |x|
  (∀ x : ℝ, f x = 20) →
  (∃ x₁ x₂ : ℝ, f x₁ = 20 ∧ f x₂ = 20 ∧ x₁ + x₂ = 0) :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l2162_216253


namespace NUMINAMATH_GPT_cubic_root_relation_l2162_216250

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem cubic_root_relation
  (x1 x2 x3 : ℝ)
  (hx1x2 : x1 < x2)
  (hx2x3 : x2 < 0)
  (hx3pos : 0 < x3)
  (hfx1 : f x1 = 0)
  (hfx2 : f x2 = 0)
  (hfx3 : f x3 = 0) :
  x3^2 - x2^2 = x3 - x1 :=
sorry

end NUMINAMATH_GPT_cubic_root_relation_l2162_216250


namespace NUMINAMATH_GPT_find_c_l2162_216261

-- Define the problem conditions and statement

variables (a b c : ℝ) (A B C : ℝ)
variable (cos_C : ℝ)
variable (sin_A sin_B : ℝ)

-- Given conditions
axiom h1 : a = 2
axiom h2 : cos_C = -1/4
axiom h3 : 3 * sin_A = 2 * sin_B
axiom sine_rule : sin_A / a = sin_B / b

-- Using sine rule to derive relation between a and b
axiom h4 : 3 * a = 2 * b

-- Cosine rule axiom
axiom cosine_rule : c^2 = a^2 + b^2 - 2 * a * b * cos_C

-- Prove c = 4
theorem find_c : c = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2162_216261


namespace NUMINAMATH_GPT_determinant_of_matrixA_l2162_216269

variable (x : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x], ![x, x + 2, x], ![x, x, x + 2]]

theorem determinant_of_matrixA : Matrix.det (matrixA x) = 8 * x + 8 := by
  sorry

end NUMINAMATH_GPT_determinant_of_matrixA_l2162_216269


namespace NUMINAMATH_GPT_find_a_parallel_lines_l2162_216234

theorem find_a_parallel_lines (a : ℝ) (l1_parallel_l2 : x + a * y + 6 = 0 → (a - 1) * x + 2 * y + 3 * a = 0 → Parallel) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_parallel_lines_l2162_216234


namespace NUMINAMATH_GPT_right_triangle_ratio_l2162_216294

theorem right_triangle_ratio (a b c r s : ℝ) (h_right_angle : (a:ℝ)^2 + (b:ℝ)^2 = c^2)
  (h_perpendicular : ∀ h : ℝ, c = r + s)
  (h_ratio_ab : a / b = 2 / 5)
  (h_geometry_r : r = a^2 / c)
  (h_geometry_s : s = b^2 / c) :
  r / s = 4 / 25 :=
sorry

end NUMINAMATH_GPT_right_triangle_ratio_l2162_216294


namespace NUMINAMATH_GPT_tom_paid_1145_l2162_216238

-- Define the quantities
def quantity_apples : ℕ := 8
def rate_apples : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 65

-- Calculate costs
def cost_apples : ℕ := quantity_apples * rate_apples
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Calculate the total amount paid
def total_amount_paid : ℕ := cost_apples + cost_mangoes

-- The theorem to prove
theorem tom_paid_1145 :
  total_amount_paid = 1145 :=
by sorry

end NUMINAMATH_GPT_tom_paid_1145_l2162_216238


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2162_216278

def point (x y : ℝ) := (x, y)
def x_positive (p : ℝ × ℝ) : Prop := p.1 > 0
def y_negative (p : ℝ × ℝ) : Prop := p.2 < 0
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := x_positive p ∧ y_negative p

theorem point_in_fourth_quadrant : in_fourth_quadrant (2, -4) :=
by
  -- The proof states that (2, -4) is in the fourth quadrant.
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2162_216278


namespace NUMINAMATH_GPT_ott_fraction_is_3_over_13_l2162_216229

-- Defining the types and quantities involved
noncomputable def moes_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def lokis_original_money (amount_given: ℚ) := amount_given * 5
noncomputable def nicks_original_money (amount_given: ℚ) := amount_given * 3

-- Total original money of the group (excluding Ott)
noncomputable def total_original_money (amount_given: ℚ) :=
  moes_original_money amount_given + lokis_original_money amount_given + nicks_original_money amount_given

-- Total money received by Ott
noncomputable def otts_received_money (amount_given: ℚ) := 3 * amount_given

-- Fraction of the group's total money Ott now has
noncomputable def otts_fraction_of_total_money (amount_given: ℚ) : ℚ :=
  otts_received_money amount_given / total_original_money amount_given

-- The theorem to be proved
theorem ott_fraction_is_3_over_13 :
  otts_fraction_of_total_money 1 = 3 / 13 :=
by
  -- The body of the proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_ott_fraction_is_3_over_13_l2162_216229


namespace NUMINAMATH_GPT_value_of_k_l2162_216287

noncomputable def roots_in_ratio_equation {k : ℝ} (h : k ≠ 0) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ 
  (r₁ / r₂ = 3) ∧ 
  (r₁ + r₂ = -8) ∧ 
  (r₁ * r₂ = k)

theorem value_of_k (k : ℝ) (h : k ≠ 0) (hr : roots_in_ratio_equation h) : k = 12 :=
sorry

end NUMINAMATH_GPT_value_of_k_l2162_216287


namespace NUMINAMATH_GPT_problem_statement_l2162_216207

theorem problem_statement {f : ℝ → ℝ}
  (Hodd : ∀ x, f (-x) = -f x)
  (Hdecreasing : ∀ x y, x < y → f x > f y)
  (a b : ℝ) (H : f a + f b > 0) : a + b < 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2162_216207


namespace NUMINAMATH_GPT_largest_sum_of_ABC_l2162_216230

-- Define the variables and the conditions
def A := 533
def B := 5
def C := 1

-- Define the product condition
def product_condition : Prop := (A * B * C = 2665)

-- Define the distinct positive integers condition
def distinct_positive_integers_condition : Prop := (A > 0 ∧ B > 0 ∧ C > 0 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- State the theorem
theorem largest_sum_of_ABC : product_condition → distinct_positive_integers_condition → A + B + C = 539 := by
  intros _ _
  sorry

end NUMINAMATH_GPT_largest_sum_of_ABC_l2162_216230


namespace NUMINAMATH_GPT_rectangle_perimeter_l2162_216271

theorem rectangle_perimeter (z w : ℝ) (h : z > w) :
  (2 * ((z - w) + w)) = 2 * z := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2162_216271


namespace NUMINAMATH_GPT_original_selling_price_l2162_216284

-- Definitions and conditions
def cost_price (CP : ℝ) := CP
def profit (CP : ℝ) := 1.25 * CP
def loss (CP : ℝ) := 0.75 * CP
def loss_price (CP : ℝ) := 600

-- Main theorem statement
theorem original_selling_price (CP : ℝ) (h1 : loss CP = loss_price CP) : profit CP = 1000 :=
by
  -- Note: adding the proof that CP = 800 and then profit CP = 1000 would be here.
  sorry

end NUMINAMATH_GPT_original_selling_price_l2162_216284


namespace NUMINAMATH_GPT_S_10_minus_S_7_l2162_216210

-- Define the first term and common difference of the arithmetic sequence
variables (a₁ d : ℕ)

-- Define the arithmetic sequence based on the first term and common difference
def arithmetic_sequence (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions given in the problem
axiom a_5_eq : a₁ + 4 * d = 8
axiom S_3_eq : sum_arithmetic_sequence a₁ 3 = 6

-- The goal: prove that S_10 - S_7 = 48
theorem S_10_minus_S_7 : sum_arithmetic_sequence a₁ 10 - sum_arithmetic_sequence a₁ 7 = 48 :=
sorry

end NUMINAMATH_GPT_S_10_minus_S_7_l2162_216210


namespace NUMINAMATH_GPT_max_n_possible_l2162_216247

theorem max_n_possible (k : ℕ) (h_k : k > 1) : ∃ n : ℕ, n = k - 1 :=
by
  sorry

end NUMINAMATH_GPT_max_n_possible_l2162_216247


namespace NUMINAMATH_GPT_triangular_weight_60_l2162_216262

def round_weight := ℝ
def triangular_weight := ℝ
def rectangular_weight := 90

variables (c t : ℝ)

-- Conditions
axiom condition1 : c + t = 3 * c
axiom condition2 : 4 * c + t = t + c + rectangular_weight

theorem triangular_weight_60 : t = 60 :=
  sorry

end NUMINAMATH_GPT_triangular_weight_60_l2162_216262


namespace NUMINAMATH_GPT_find_set_B_l2162_216267

set_option pp.all true

variable (A : Set ℤ) (B : Set ℤ)

theorem find_set_B (hA : A = {-2, 0, 1, 3})
                    (hB : B = {x | -x ∈ A ∧ 1 - x ∉ A}) :
  B = {-3, -1, 2} :=
by
  sorry

end NUMINAMATH_GPT_find_set_B_l2162_216267


namespace NUMINAMATH_GPT_closest_perfect_square_to_314_l2162_216236

theorem closest_perfect_square_to_314 :
  ∃ n : ℤ, n^2 = 324 ∧ ∀ m : ℤ, m^2 ≠ 324 → |m^2 - 314| > |324 - 314| :=
by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_to_314_l2162_216236


namespace NUMINAMATH_GPT_find_chosen_number_l2162_216232

theorem find_chosen_number (x : ℤ) (h : 2 * x - 138 = 106) : x = 122 :=
by
  sorry

end NUMINAMATH_GPT_find_chosen_number_l2162_216232


namespace NUMINAMATH_GPT_compute_pqr_l2162_216257

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 26) (h_eq : (1 : ℚ) / ↑p + (1 : ℚ) / ↑q + (1 : ℚ) / ↑r + 360 / (p * q * r) = 1) : 
  p * q * r = 576 := 
sorry

end NUMINAMATH_GPT_compute_pqr_l2162_216257


namespace NUMINAMATH_GPT_isosceles_triangle_if_perpendiculars_intersect_at_single_point_l2162_216220

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_if_perpendiculars_intersect_at_single_point
  (a b c : ℝ)
  (D E F P Q R H : Type)
  (intersection_point: P = Q ∧ Q = R ∧ P = R ∧ P = H) :
  is_isosceles_triangle a b c := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_if_perpendiculars_intersect_at_single_point_l2162_216220


namespace NUMINAMATH_GPT_fangfang_travel_time_l2162_216299

theorem fangfang_travel_time (time_1_to_5 : ℕ) (start_floor end_floor : ℕ) (floors_1_to_5 : ℕ) (floors_2_to_7 : ℕ) :
  time_1_to_5 = 40 →
  floors_1_to_5 = 5 - 1 →
  floors_2_to_7 = 7 - 2 →
  end_floor = 7 →
  start_floor = 2 →
  (end_floor - start_floor) * (time_1_to_5 / floors_1_to_5) = 50 :=
by 
  sorry

end NUMINAMATH_GPT_fangfang_travel_time_l2162_216299


namespace NUMINAMATH_GPT_goods_train_length_l2162_216282

noncomputable def length_of_goods_train (speed_first_train_kmph speed_goods_train_kmph time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := speed_first_train_kmph + speed_goods_train_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5.0 / 18.0)
  relative_speed_mps * (time_seconds : ℝ)

theorem goods_train_length
  (speed_first_train_kmph : ℕ) (speed_goods_train_kmph : ℕ) (time_seconds : ℕ) 
  (h1 : speed_first_train_kmph = 50)
  (h2 : speed_goods_train_kmph = 62)
  (h3 : time_seconds = 9) :
  length_of_goods_train speed_first_train_kmph speed_goods_train_kmph time_seconds = 280 :=
  sorry

end NUMINAMATH_GPT_goods_train_length_l2162_216282


namespace NUMINAMATH_GPT_sum_squares_not_perfect_square_l2162_216291

theorem sum_squares_not_perfect_square (x y z : ℤ) (h : x^2 + y^2 + z^2 = 1993) : ¬ ∃ a : ℤ, x + y + z = a^2 :=
sorry

end NUMINAMATH_GPT_sum_squares_not_perfect_square_l2162_216291


namespace NUMINAMATH_GPT_find_y_l2162_216283

theorem find_y (x y : ℝ) (h1 : 2 * x - 3 * y = 24) (h2 : x + 2 * y = 15) : y = 6 / 7 :=
by sorry

end NUMINAMATH_GPT_find_y_l2162_216283


namespace NUMINAMATH_GPT_sum_series_eq_one_l2162_216295

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, (2^n + 1) / (3^(2^n) + 1)

theorem sum_series_eq_one : sum_series = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sum_series_eq_one_l2162_216295


namespace NUMINAMATH_GPT_total_amount_spent_correct_l2162_216224

noncomputable def total_amount_spent (mango_cost pineapple_cost cost_pineapple total_people : ℕ) : ℕ :=
  let pineapple_people := cost_pineapple / pineapple_cost
  let mango_people := total_people - pineapple_people
  let mango_cost_total := mango_people * mango_cost
  cost_pineapple + mango_cost_total

theorem total_amount_spent_correct :
  total_amount_spent 5 6 54 17 = 94 := by
  -- This is where the proof would go, but it's omitted per instructions
  sorry

end NUMINAMATH_GPT_total_amount_spent_correct_l2162_216224


namespace NUMINAMATH_GPT_final_pens_count_l2162_216286

-- Define the initial number of pens and subsequent operations
def initial_pens : ℕ := 7
def pens_after_mike (initial : ℕ) : ℕ := initial + 22
def pens_after_cindy (pens : ℕ) : ℕ := pens * 2
def pens_after_sharon (pens : ℕ) : ℕ := pens - 19

-- Prove that the final number of pens is 39
theorem final_pens_count : pens_after_sharon (pens_after_cindy (pens_after_mike initial_pens)) = 39 := 
sorry

end NUMINAMATH_GPT_final_pens_count_l2162_216286


namespace NUMINAMATH_GPT_shoe_store_sale_l2162_216292

theorem shoe_store_sale (total_sneakers : ℕ) (total_sandals : ℕ) (total_shoes : ℕ) (total_boots : ℕ) 
  (h1 : total_sneakers = 2) 
  (h2 : total_sandals = 4) 
  (h3 : total_shoes = 17) 
  (h4 : total_boots = total_shoes - (total_sneakers + total_sandals)) : 
  total_boots = 11 :=
by
  rw [h1, h2, h3] at h4
  exact h4
-- sorry

end NUMINAMATH_GPT_shoe_store_sale_l2162_216292


namespace NUMINAMATH_GPT_trig_proof_l2162_216240

variable {α a : ℝ}

theorem trig_proof (h₁ : (∃ a : ℝ, a < 0 ∧ (4 * a, -3 * a) = (4 * a, -3 * a)))
                    (h₂ : a < 0) :
  2 * Real.sin α + Real.cos α = 2 / 5 := 
sorry

end NUMINAMATH_GPT_trig_proof_l2162_216240


namespace NUMINAMATH_GPT_repeating_decimal_exceeds_finite_decimal_by_l2162_216203

-- Definitions based on the problem conditions
def repeating_decimal := 8 / 11
def finite_decimal := 18 / 25

-- Statement to be proved
theorem repeating_decimal_exceeds_finite_decimal_by : 
  repeating_decimal - finite_decimal = 2 / 275 :=
by
  -- Skipping the proof itself with 'sorry'
  sorry

end NUMINAMATH_GPT_repeating_decimal_exceeds_finite_decimal_by_l2162_216203


namespace NUMINAMATH_GPT_harriet_trip_time_l2162_216288

theorem harriet_trip_time
  (speed_AB : ℕ := 100)
  (speed_BA : ℕ := 150)
  (total_trip_time : ℕ := 5)
  (time_threshold : ℕ := 180) :
  let D := (speed_AB * speed_BA * total_trip_time) / (speed_AB + speed_BA)
  let time_AB := D / speed_AB
  let time_AB_min := time_AB * 60
  time_AB_min = time_threshold :=
by
  sorry

end NUMINAMATH_GPT_harriet_trip_time_l2162_216288


namespace NUMINAMATH_GPT_race_winner_l2162_216252

-- Definitions and conditions based on the problem statement
def tortoise_speed : ℕ := 5  -- Tortoise speed in meters per minute
def hare_speed_1 : ℕ := 20  -- Hare initial speed in meters per minute
def hare_time_1 : ℕ := 3  -- Hare initial running time in minutes
def hare_speed_2 : ℕ := 10  -- Hare speed when going back in meters per minute
def hare_time_2 : ℕ := 2  -- Hare back running time in minutes
def hare_sleep_time : ℕ := 5  -- Hare sleeping time in minutes
def hare_speed_3 : ℕ := 25  -- Hare final speed in meters per minute
def track_length : ℕ := 130  -- Total length of the race track in meters

-- The problem statement
theorem race_winner :
  track_length / tortoise_speed > hare_time_1 + hare_time_2 + hare_sleep_time + (track_length - (hare_speed_1 * hare_time_1 - hare_speed_2 * hare_time_2)) / hare_speed_3 :=
sorry

end NUMINAMATH_GPT_race_winner_l2162_216252


namespace NUMINAMATH_GPT_problem_statement_l2162_216254

noncomputable def omega : ℂ := sorry -- Definition placeholder for a specific nonreal root of x^4 = 1. 

theorem problem_statement (h1 : omega ^ 4 = 1) (h2 : omega ^ 2 = -1) : 
  (1 - omega + omega ^ 3) ^ 4 + (1 + omega - omega ^ 3) ^ 4 = -14 := 
sorry

end NUMINAMATH_GPT_problem_statement_l2162_216254


namespace NUMINAMATH_GPT_ratio_of_fruit_salads_l2162_216227

theorem ratio_of_fruit_salads 
  (salads_Alaya : ℕ) 
  (total_salads : ℕ) 
  (h1 : salads_Alaya = 200) 
  (h2 : total_salads = 600) : 
  (total_salads - salads_Alaya) / salads_Alaya = 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_fruit_salads_l2162_216227


namespace NUMINAMATH_GPT_sector_area_l2162_216268

theorem sector_area (r α l S : ℝ) (h1 : l + 2 * r = 8) (h2 : α = 2) (h3 : l = α * r) :
  S = 4 :=
by
  -- Let the radius be 2 as a condition derived from h1 and h2
  have r := 2
  -- Substitute and compute to find S
  have S_calculated := (1 / 2 * α * r * r)
  sorry

end NUMINAMATH_GPT_sector_area_l2162_216268


namespace NUMINAMATH_GPT_clara_climbs_stone_blocks_l2162_216258

-- Define the number of steps per level
def steps_per_level : Nat := 8

-- Define the number of blocks per step
def blocks_per_step : Nat := 3

-- Define the number of levels in the tower
def levels : Nat := 4

-- Define a function to compute the total number of blocks given the constants
def total_blocks (steps_per_level blocks_per_step levels : Nat) : Nat :=
  steps_per_level * blocks_per_step * levels

-- Statement of the theorem
theorem clara_climbs_stone_blocks :
  total_blocks steps_per_level blocks_per_step levels = 96 :=
by
  -- Lean requires 'sorry' as a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_clara_climbs_stone_blocks_l2162_216258


namespace NUMINAMATH_GPT_solution_fraction_l2162_216237

-- Conditions and definition of x
def initial_quantity : ℝ := 1
def concentration_70 : ℝ := 0.70
def concentration_25 : ℝ := 0.25
def concentration_new : ℝ := 0.35

-- Definition of the fraction of the solution replaced
def x (fraction : ℝ) : Prop :=
  concentration_70 * initial_quantity - concentration_70 * fraction + concentration_25 * fraction = concentration_new * initial_quantity

-- The theorem we need to prove
theorem solution_fraction : ∃ (fraction : ℝ), x fraction ∧ fraction = 7 / 9 :=
by
  use 7 / 9
  simp [x]
  sorry  -- Proof steps would be filled here

end NUMINAMATH_GPT_solution_fraction_l2162_216237


namespace NUMINAMATH_GPT_cube_surface_area_l2162_216281

-- Define the edge length of the cube.
def edge_length (a : ℝ) : ℝ := 6 * a

-- Define the surface area of a cube given the edge length.
def surface_area (e : ℝ) : ℝ := 6 * (e * e)

-- The theorem to prove.
theorem cube_surface_area (a : ℝ) : surface_area (edge_length a) = 216 * (a * a) := 
  sorry

end NUMINAMATH_GPT_cube_surface_area_l2162_216281


namespace NUMINAMATH_GPT_melanie_picked_plums_l2162_216263

variable (picked_plums : ℕ)
variable (given_plums : ℕ := 3)
variable (total_plums : ℕ := 10)

theorem melanie_picked_plums :
  picked_plums + given_plums = total_plums → picked_plums = 7 := by
  sorry

end NUMINAMATH_GPT_melanie_picked_plums_l2162_216263


namespace NUMINAMATH_GPT_trig_expression_value_l2162_216200

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l2162_216200


namespace NUMINAMATH_GPT_function_positive_on_interval_l2162_216255

theorem function_positive_on_interval (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (2 - a^2) * x + a > 0) ↔ 0 < a ∧ a < 2 :=
by
  sorry

end NUMINAMATH_GPT_function_positive_on_interval_l2162_216255


namespace NUMINAMATH_GPT_perimeter_ratio_l2162_216273

variables (K T k R r : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (r = 2 * T / K)
def condition2 : Prop := (2 * T = R * k)

-- The statement we want to prove
theorem perimeter_ratio :
  condition1 K T r →
  condition2 T R k →
  K / k = R / r :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_perimeter_ratio_l2162_216273


namespace NUMINAMATH_GPT_minimum_n_for_i_pow_n_eq_neg_i_l2162_216214

open Complex

theorem minimum_n_for_i_pow_n_eq_neg_i : ∃ (n : ℕ), 0 < n ∧ (i^n = -i) ∧ ∀ (m : ℕ), 0 < m ∧ (i^m = -i) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_minimum_n_for_i_pow_n_eq_neg_i_l2162_216214


namespace NUMINAMATH_GPT_total_area_of_field_l2162_216226

theorem total_area_of_field 
  (A_s : ℕ) 
  (h₁ : A_s = 315)
  (A_l : ℕ) 
  (h₂ : A_l - A_s = (1/5) * ((A_s + A_l) / 2)) : 
  A_s + A_l = 700 := 
  by 
    sorry

end NUMINAMATH_GPT_total_area_of_field_l2162_216226


namespace NUMINAMATH_GPT_ellipse_find_m_l2162_216256

theorem ellipse_find_m (a b m e : ℝ) 
  (h1 : a^2 = 4) 
  (h2 : b^2 = m)
  (h3 : e = 1/2) :
  m = 3 := 
by
  sorry

end NUMINAMATH_GPT_ellipse_find_m_l2162_216256


namespace NUMINAMATH_GPT_households_without_car_or_bike_l2162_216248

/--
In a neighborhood having 90 households, some did not have either a car or a bike.
If 16 households had both a car and a bike and 44 had a car, and
there were 35 households with a bike only.
Prove that there are 11 households that did not have either a car or a bike.
-/
theorem households_without_car_or_bike
  (total_households : ℕ)
  (both_car_and_bike : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : both_car_and_bike = 16)
  (H3 : car = 44)
  (H4 : bike_only = 35) :
  ∃ N : ℕ, N = total_households - (car - both_car_and_bike + bike_only + both_car_and_bike) ∧ N = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_households_without_car_or_bike_l2162_216248


namespace NUMINAMATH_GPT_expected_expenditure_l2162_216244

-- Define the parameters and conditions
def b : ℝ := 0.8
def a : ℝ := 2
def e_condition (e : ℝ) : Prop := |e| < 0.5
def revenue : ℝ := 10

-- Define the expenditure function based on the conditions
def expenditure (x e : ℝ) : ℝ := b * x + a + e

-- The expected expenditure should not exceed 10.5
theorem expected_expenditure (e : ℝ) (h : e_condition e) : expenditure revenue e ≤ 10.5 :=
sorry

end NUMINAMATH_GPT_expected_expenditure_l2162_216244


namespace NUMINAMATH_GPT_find_n_l2162_216245

noncomputable def objects_per_hour (n : ℕ) : ℕ := n

theorem find_n (n : ℕ) (h₁ : 1 + (2 / 3) + (1 / 3) + (1 / 3) = 7 / 3) 
  (h₂ : objects_per_hour n * 7 / 3 = 28) : n = 12 :=
by
  have total_hours := h₁ 
  have total_objects := h₂
  sorry

end NUMINAMATH_GPT_find_n_l2162_216245


namespace NUMINAMATH_GPT_emily_stickers_l2162_216266

theorem emily_stickers:
  ∃ S : ℕ, (S % 4 = 2) ∧
           (S % 6 = 2) ∧
           (S % 9 = 2) ∧
           (S % 10 = 2) ∧
           (S > 2) ∧
           (S = 182) :=
  sorry

end NUMINAMATH_GPT_emily_stickers_l2162_216266


namespace NUMINAMATH_GPT_original_number_l2162_216209

theorem original_number (x : ℝ) (h : 1.47 * x = 1214.33) : x = 826.14 :=
sorry

end NUMINAMATH_GPT_original_number_l2162_216209


namespace NUMINAMATH_GPT_parsley_rows_l2162_216221

-- Define the conditions laid out in the problem
def garden_rows : ℕ := 20
def plants_per_row : ℕ := 10
def rosemary_rows : ℕ := 2
def chives_planted : ℕ := 150

-- Define the target statement to prove
theorem parsley_rows :
  let total_plants := garden_rows * plants_per_row
  let remaining_rows := garden_rows - rosemary_rows
  let chives_rows := chives_planted / plants_per_row
  let parsley_rows := remaining_rows - chives_rows
  parsley_rows = 3 :=
by
  sorry

end NUMINAMATH_GPT_parsley_rows_l2162_216221


namespace NUMINAMATH_GPT_linear_equation_variables_l2162_216260

theorem linear_equation_variables (m n : ℤ) (h1 : 3 * m - 2 * n = 1) (h2 : n - m = 1) : m = 0 ∧ n = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_linear_equation_variables_l2162_216260


namespace NUMINAMATH_GPT_q_is_20_percent_less_than_p_l2162_216205

theorem q_is_20_percent_less_than_p (p q : ℝ) (h : p = 1.25 * q) : (q - p) / p * 100 = -20 := by
  sorry

end NUMINAMATH_GPT_q_is_20_percent_less_than_p_l2162_216205


namespace NUMINAMATH_GPT_max_ab_min_inv_a_plus_4_div_b_l2162_216246

theorem max_ab (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) : 
  ab ≤ 1 :=
by
  sorry

theorem min_inv_a_plus_4_div_b (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_sum : a + 4 * b = 4) :
  1 / a + 4 / b ≥ 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_min_inv_a_plus_4_div_b_l2162_216246


namespace NUMINAMATH_GPT_area_of_square_field_l2162_216276

-- Define the side length of the square
def side_length : ℝ := 15

-- Define the area of the square based on the side length
def square_area (side : ℝ) : ℝ := side * side

-- The theorem stating the area of a square with side length 15 meters
theorem area_of_square_field : square_area side_length = 225 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_square_field_l2162_216276


namespace NUMINAMATH_GPT_octahedron_tetrahedron_volume_ratio_l2162_216275

theorem octahedron_tetrahedron_volume_ratio (a : ℝ) :
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3
  V_o / V_t = 1 :=
by 
  -- Definitions from conditions
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3

  -- Proof omitted
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_octahedron_tetrahedron_volume_ratio_l2162_216275


namespace NUMINAMATH_GPT_intersection_M_N_l2162_216235

open Set

noncomputable def M : Set ℝ := {x | x ≥ 2}

noncomputable def N : Set ℝ := {x | x^2 - 6*x + 5 < 0}

theorem intersection_M_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2162_216235


namespace NUMINAMATH_GPT_find_k_l2162_216285

def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 6
def g (k x : ℝ) : ℝ := 2 * x^2 - k * x + 2

theorem find_k (k : ℝ) : 
  f 5 - g k 5 = 15 -> k = -15.8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l2162_216285


namespace NUMINAMATH_GPT_corridor_perimeter_l2162_216293

theorem corridor_perimeter
  (P1 P2 : ℕ)
  (h₁ : P1 = 16)
  (h₂ : P2 = 24) : 
  2 * ((P2 / 4 + (P1 + P2) / 4) + (P2 / 4) - (P1 / 4)) = 40 :=
by {
  -- The proof can be filled here
  sorry
}

end NUMINAMATH_GPT_corridor_perimeter_l2162_216293


namespace NUMINAMATH_GPT_domain_condition_l2162_216280

variable (k : ℝ)
def quadratic_expression (x : ℝ) : ℝ := k * x^2 - 4 * k * x + k + 8

theorem domain_condition (k : ℝ) : (∀ x : ℝ, quadratic_expression k x > 0) ↔ (0 ≤ k ∧ k < 8/3) :=
sorry

end NUMINAMATH_GPT_domain_condition_l2162_216280


namespace NUMINAMATH_GPT_quinn_free_donuts_l2162_216202

-- Definitions based on conditions
def books_per_week : ℕ := 2
def weeks : ℕ := 10
def books_needed_for_donut : ℕ := 5

-- Calculation based on conditions
def total_books_read : ℕ := books_per_week * weeks
def free_donuts (total_books : ℕ) : ℕ := total_books / books_needed_for_donut

-- Proof statement
theorem quinn_free_donuts : free_donuts total_books_read = 4 := by
  sorry

end NUMINAMATH_GPT_quinn_free_donuts_l2162_216202


namespace NUMINAMATH_GPT_mutually_exclusive_not_necessarily_complementary_l2162_216298

-- Define what it means for events to be mutually exclusive
def mutually_exclusive (E1 E2 : Prop) : Prop :=
  ¬ (E1 ∧ E2)

-- Define what it means for events to be complementary
def complementary (E1 E2 : Prop) : Prop :=
  (E1 ∨ E2) ∧ ¬ (E1 ∧ E2) ∧ (¬ E1 ∨ ¬ E2)

theorem mutually_exclusive_not_necessarily_complementary :
  ∀ E1 E2 : Prop, mutually_exclusive E1 E2 → ¬ complementary E1 E2 :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_not_necessarily_complementary_l2162_216298


namespace NUMINAMATH_GPT_rotted_tomatoes_is_correct_l2162_216225

noncomputable def shipment_1 : ℕ := 1000
noncomputable def sold_Saturday : ℕ := 300
noncomputable def shipment_2 : ℕ := 2 * shipment_1
noncomputable def tomatoes_Tuesday : ℕ := 2500

-- Define remaining tomatoes after the first shipment accounting for Saturday's sales
noncomputable def remaining_tomatoes_1 : ℕ := shipment_1 - sold_Saturday

-- Define total tomatoes after second shipment arrives
noncomputable def total_tomatoes_after_second_shipment : ℕ := remaining_tomatoes_1 + shipment_2

-- Define the amount of tomatoes that rotted
noncomputable def rotted_tomatoes : ℕ :=
  total_tomatoes_after_second_shipment - tomatoes_Tuesday

theorem rotted_tomatoes_is_correct :
  rotted_tomatoes = 200 := by
  sorry

end NUMINAMATH_GPT_rotted_tomatoes_is_correct_l2162_216225


namespace NUMINAMATH_GPT_mean_variance_transformation_l2162_216206

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (mean_original variance_original : ℝ)
variable (meam_new variance_new : ℝ)
variable (offset : ℝ)

theorem mean_variance_transformation (hmean : mean_original = 2.8) (hvariance : variance_original = 3.6) 
  (hoffset : offset = 60) : 
  (mean_new = mean_original + offset) ∧ (variance_new = variance_original) :=
  sorry

end NUMINAMATH_GPT_mean_variance_transformation_l2162_216206
