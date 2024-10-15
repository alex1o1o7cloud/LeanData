import Mathlib

namespace NUMINAMATH_GPT_prob_second_shot_l752_75215

theorem prob_second_shot (P_A : ℝ) (P_AB : ℝ) (p : ℝ) : 
  P_A = 0.75 → 
  P_AB = 0.6 → 
  P_A * p = P_AB → 
  p = 0.8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_prob_second_shot_l752_75215


namespace NUMINAMATH_GPT_linear_eq_zero_l752_75208

variables {a b c d x y : ℝ}

theorem linear_eq_zero (h1 : a * x + b * y = 0) (h2 : c * x + d * y = 0) (h3 : a * d - c * b ≠ 0) :
  x = 0 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_zero_l752_75208


namespace NUMINAMATH_GPT_find_m_l752_75205

open Real

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_m :
  let a := (-sqrt 3, m)
  let b := (2, 1)
  (dot_product a b = 0) → m = 2 * sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l752_75205


namespace NUMINAMATH_GPT_find_p_l752_75250

theorem find_p (p q : ℝ) (h1 : p + 2 * q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : 10 * p^9 * q = 45 * p^8 * q^2): 
  p = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l752_75250


namespace NUMINAMATH_GPT_find_two_digit_number_l752_75242

-- A type synonym for digit
def Digit := {n : ℕ // n < 10}

-- Define the conditions
variable (X Y : Digit)
-- The product of the digits is 8
def product_of_digits : Prop := X.val * Y.val = 8

-- When 18 is added, digits are reversed
def digits_reversed : Prop := 10 * X.val + Y.val + 18 = 10 * Y.val + X.val

-- The question translated to Lean: Prove that the two-digit number is 24
theorem find_two_digit_number (h1 : product_of_digits X Y) (h2 : digits_reversed X Y) : 10 * X.val + Y.val = 24 :=
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l752_75242


namespace NUMINAMATH_GPT_sequence_terms_l752_75239

/-- Given the sequence {a_n} with the sum of the first n terms S_n = n^2 - 3, 
    prove that a_1 = -2 and a_n = 2n - 1 for n ≥ 2. --/
theorem sequence_terms (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n : ℕ, S n = n^2 - 3)
  (h1 : ∀ n : ℕ, a n = S n - S (n - 1)) :
  a 1 = -2 ∧ (∀ n : ℕ, n ≥ 2 → a n = 2 * n - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_terms_l752_75239


namespace NUMINAMATH_GPT_man_buys_article_for_20_l752_75214

variable (SP : ℝ) (G : ℝ) (CP : ℝ)

theorem man_buys_article_for_20 (hSP : SP = 25) (hG : G = 0.25) (hEquation : SP = CP * (1 + G)) : CP = 20 :=
by
  sorry

end NUMINAMATH_GPT_man_buys_article_for_20_l752_75214


namespace NUMINAMATH_GPT_zhang_bing_age_18_l752_75203

theorem zhang_bing_age_18 {x a : ℕ} (h1 : x < 2023) 
  (h2 : a = x - 1953)
  (h3 : a % 9 = 0)
  (h4 : a = (x % 10) + ((x / 10) % 10) + ((x / 100) % 10) + ((x / 1000) % 10)) :
  a = 18 :=
sorry

end NUMINAMATH_GPT_zhang_bing_age_18_l752_75203


namespace NUMINAMATH_GPT_total_cost_l752_75267

theorem total_cost
  (permits_cost : ℕ)
  (contractor_hourly_rate : ℕ)
  (contractor_days : ℕ)
  (contractor_hours_per_day : ℕ)
  (inspector_discount : ℕ)
  (h_pc : permits_cost = 250)
  (h_chr : contractor_hourly_rate = 150)
  (h_cd : contractor_days = 3)
  (h_chpd : contractor_hours_per_day = 5)
  (h_id : inspector_discount = 80)
  (contractor_total_hours : ℕ := contractor_days * contractor_hours_per_day)
  (contractor_total_cost : ℕ := contractor_total_hours * contractor_hourly_rate)
  (inspector_cost : ℕ := contractor_total_cost - (contractor_total_cost * inspector_discount / 100))
  (total_cost : ℕ := permits_cost + contractor_total_cost + inspector_cost) :
  total_cost = 2950 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l752_75267


namespace NUMINAMATH_GPT_percentage_of_red_non_honda_cars_l752_75262

-- Define the conditions
def total_cars : ℕ := 900
def honda_cars : ℕ := 500
def red_per_100_honda_cars : ℕ := 90
def red_percent_total := 60

-- Define the question we want to answer
theorem percentage_of_red_non_honda_cars : 
  let red_honda_cars := (red_per_100_honda_cars / 100 : ℚ) * honda_cars
  let total_red_cars := (red_percent_total / 100 : ℚ) * total_cars
  let red_non_honda_cars := total_red_cars - red_honda_cars
  let non_honda_cars := total_cars - honda_cars
  (red_non_honda_cars / non_honda_cars) * 100 = (22.5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_red_non_honda_cars_l752_75262


namespace NUMINAMATH_GPT_ms_perez_class_total_students_l752_75271

/-- Half the students in Ms. Perez's class collected 12 cans each, two students didn't collect any cans,
    and the remaining 13 students collected 4 cans each. The total number of cans collected is 232. 
    Prove that the total number of students in Ms. Perez's class is 30. -/
theorem ms_perez_class_total_students (S : ℕ) :
  (S / 2) * 12 + 13 * 4 + 2 * 0 = 232 →
  S = S / 2 + 13 + 2 →
  S = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_ms_perez_class_total_students_l752_75271


namespace NUMINAMATH_GPT_product_of_roots_quadratic_l752_75264

noncomputable def product_of_roots (a b c : ℚ) : ℚ :=
  c / a

theorem product_of_roots_quadratic : product_of_roots 14 21 (-250) = -125 / 7 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_quadratic_l752_75264


namespace NUMINAMATH_GPT_problem_l752_75226

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem problem (a : ℝ) (h : f a = 2) : f (-a) = 0 := 
  sorry

end NUMINAMATH_GPT_problem_l752_75226


namespace NUMINAMATH_GPT_max_xy_value_l752_75211

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) : xy ≤ 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_max_xy_value_l752_75211


namespace NUMINAMATH_GPT_sum_of_center_coordinates_eq_neg2_l752_75220

theorem sum_of_center_coordinates_eq_neg2 
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 7)
  (h2 : y1 = -8)
  (h3 : x2 = -5)
  (h4 : y2 = 2) 
  : (x1 + x2) / 2 + (y1 + y2) / 2 = -2 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_sum_of_center_coordinates_eq_neg2_l752_75220


namespace NUMINAMATH_GPT_rectangle_sides_l752_75251

theorem rectangle_sides (n : ℕ) (hpos : n > 0)
  (h1 : (∃ (a : ℕ), (a^2 * n = n)))
  (h2 : (∃ (b : ℕ), (b^2 * (n + 98) = n))) :
  (∃ (l w : ℕ), l * w = n ∧ 
  ((n = 126 ∧ (l = 3 ∧ w = 42 ∨ l = 6 ∧ w = 21)) ∨
  (n = 1152 ∧ l = 24 ∧ w = 48))) :=
sorry

end NUMINAMATH_GPT_rectangle_sides_l752_75251


namespace NUMINAMATH_GPT_expression_varies_l752_75248

variables {x : ℝ}

noncomputable def expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x - 5) / ((x + 1) * (x - 3)) - (8 + x) / ((x + 1) * (x - 3))

theorem expression_varies (h1 : x ≠ -1) (h2 : x ≠ 3) : 
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ 
  expression x₀ ≠ expression x₁ :=
by
  sorry

end NUMINAMATH_GPT_expression_varies_l752_75248


namespace NUMINAMATH_GPT_solve_integers_l752_75200

theorem solve_integers (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^(2 * y) + (x + 1)^(2 * y) = (x + 2)^(2 * y) → (x = 3 ∧ y = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_integers_l752_75200


namespace NUMINAMATH_GPT_squares_arrangement_l752_75224

noncomputable def arrangement_possible (n : ℕ) (cond : n ≥ 5) : Prop :=
  ∃ (position : ℕ → ℕ × ℕ),
    (∀ i, 1 ≤ i ∧ i ≤ n → 
        ∃ j k, j ≠ k ∧ 
             dist (position i) (position j) = 1 ∧
             dist (position i) (position k) = 1)

theorem squares_arrangement (n : ℕ) (hn : n ≥ 5) :
  arrangement_possible n hn :=
  sorry

end NUMINAMATH_GPT_squares_arrangement_l752_75224


namespace NUMINAMATH_GPT_negation_of_proposition_l752_75234

variable (l : ℝ)

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l752_75234


namespace NUMINAMATH_GPT_intersection_of_sets_l752_75244

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3 * x - 2 ≥ 1}

-- Prove that A ∩ B = {x | 1 ≤ x ∧ x ≤ 2}
theorem intersection_of_sets : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_GPT_intersection_of_sets_l752_75244


namespace NUMINAMATH_GPT_num_supervisors_correct_l752_75269

theorem num_supervisors_correct (S : ℕ) 
  (avg_sal_total : ℕ) (avg_sal_supervisor : ℕ) (avg_sal_laborer : ℕ) (num_laborers : ℕ)
  (h1 : avg_sal_total = 1250) 
  (h2 : avg_sal_supervisor = 2450) 
  (h3 : avg_sal_laborer = 950) 
  (h4 : num_laborers = 42) 
  (h5 : avg_sal_total = (39900 + S * avg_sal_supervisor) / (num_laborers + S)) : 
  S = 10 := by sorry

end NUMINAMATH_GPT_num_supervisors_correct_l752_75269


namespace NUMINAMATH_GPT_unique_triple_gcd_square_l752_75213

theorem unique_triple_gcd_square (m n l : ℕ) (H1 : m + n = Nat.gcd m n ^ 2)
                                  (H2 : m + l = Nat.gcd m l ^ 2)
                                  (H3 : n + l = Nat.gcd n l ^ 2) : (m, n, l) = (2, 2, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_triple_gcd_square_l752_75213


namespace NUMINAMATH_GPT_goldfinch_percentage_l752_75245

noncomputable def percentage_of_goldfinches 
  (goldfinches : ℕ) (sparrows : ℕ) (grackles : ℕ) : ℚ :=
  (goldfinches : ℚ) / (goldfinches + sparrows + grackles) * 100

theorem goldfinch_percentage (goldfinches sparrows grackles : ℕ)
  (h_goldfinches : goldfinches = 6)
  (h_sparrows : sparrows = 9)
  (h_grackles : grackles = 5) :
  percentage_of_goldfinches goldfinches sparrows grackles = 30 :=
by
  rw [h_goldfinches, h_sparrows, h_grackles]
  show percentage_of_goldfinches 6 9 5 = 30
  sorry

end NUMINAMATH_GPT_goldfinch_percentage_l752_75245


namespace NUMINAMATH_GPT_equations_neither_directly_nor_inversely_proportional_l752_75289

-- Definitions for equations
def equation1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def equation2 (x y : ℝ) : Prop := 4 * x * y = 12
def equation3 (x y : ℝ) : Prop := y = 1/2 * x
def equation4 (x y : ℝ) : Prop := 5 * x - 2 * y = 20
def equation5 (x y : ℝ) : Prop := x / y = 5

-- Theorem stating that y is neither directly nor inversely proportional to x for the given equations
theorem equations_neither_directly_nor_inversely_proportional (x y : ℝ) :
  (¬∃ k : ℝ, x = k * y) ∧ (¬∃ k : ℝ, x * y = k) ↔ 
  (equation1 x y ∨ equation4 x y) :=
sorry

end NUMINAMATH_GPT_equations_neither_directly_nor_inversely_proportional_l752_75289


namespace NUMINAMATH_GPT_num_occupied_third_floor_rooms_l752_75270

-- Definitions based on conditions
def first_floor_rent : Int := 15
def second_floor_rent : Int := 20
def third_floor_rent : Int := 2 * first_floor_rent
def rooms_per_floor : Int := 3
def monthly_earnings : Int := 165

-- The proof statement
theorem num_occupied_third_floor_rooms : 
  let total_full_occupancy_cost := rooms_per_floor * first_floor_rent + rooms_per_floor * second_floor_rent + rooms_per_floor * third_floor_rent
  let revenue_difference := total_full_occupancy_cost - monthly_earnings
  revenue_difference / third_floor_rent = 1 → rooms_per_floor - revenue_difference / third_floor_rent = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_occupied_third_floor_rooms_l752_75270


namespace NUMINAMATH_GPT_value_two_stds_less_than_mean_l752_75282

theorem value_two_stds_less_than_mean (μ σ : ℝ) (hμ : μ = 16.5) (hσ : σ = 1.5) : (μ - 2 * σ) = 13.5 :=
by
  rw [hμ, hσ]
  norm_num

end NUMINAMATH_GPT_value_two_stds_less_than_mean_l752_75282


namespace NUMINAMATH_GPT_cloth_cost_l752_75260

theorem cloth_cost
  (L : ℕ)
  (C : ℚ)
  (hL : L = 10)
  (h_condition : L * C = (L + 4) * (C - 1)) :
  10 * C = 35 := by
  sorry

end NUMINAMATH_GPT_cloth_cost_l752_75260


namespace NUMINAMATH_GPT_equation_represents_two_intersecting_lines_l752_75223

theorem equation_represents_two_intersecting_lines :
  (∀ x y : ℝ, x^3 * (x + y - 2) = y^3 * (x + y - 2) ↔
    (x = y ∨ y = 2 - x)) :=
by sorry

end NUMINAMATH_GPT_equation_represents_two_intersecting_lines_l752_75223


namespace NUMINAMATH_GPT_find_even_increasing_l752_75275

theorem find_even_increasing (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 < x → x < y → 0 < y → f x < f y) ↔
  f = (fun x => 3 * x^2 - 1) ∨ f = (fun x => 2^|x|) :=
by
  sorry

end NUMINAMATH_GPT_find_even_increasing_l752_75275


namespace NUMINAMATH_GPT_sum_of_legs_of_larger_triangle_l752_75288

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def similar_triangles {a1 b1 c1 a2 b2 c2 : ℝ} (h1 : right_triangle a1 b1 c1) (h2 : right_triangle a2 b2 c2) :=
  ∃ k : ℝ, k > 0 ∧ (a2 = k * a1 ∧ b2 = k * b1)

theorem sum_of_legs_of_larger_triangle 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : right_triangle a1 b1 c1)
  (h2 : right_triangle a2 b2 c2)
  (h_sim : similar_triangles h1 h2)
  (area1 : ℝ) (area2 : ℝ)
  (hyp1 : c1 = 6) 
  (area_cond1 : (a1 * b1) / 2 = 8)
  (area_cond2 : (a2 * b2) / 2 = 200) :
  a2 + b2 = 40 := by
  sorry

end NUMINAMATH_GPT_sum_of_legs_of_larger_triangle_l752_75288


namespace NUMINAMATH_GPT_perimeter_after_adding_tiles_l752_75247

-- Definition of the initial configuration
def initial_perimeter := 16

-- Definition of the number of additional tiles
def additional_tiles := 3

-- Statement of the problem: to prove that the new perimeter is 22
theorem perimeter_after_adding_tiles : initial_perimeter + 2 * additional_tiles = 22 := 
by 
  -- The number initially added each side exposed would increase the perimeter incremented by 6
  -- You can also assume the boundary conditions for the shared sides reducing.
  sorry

end NUMINAMATH_GPT_perimeter_after_adding_tiles_l752_75247


namespace NUMINAMATH_GPT_line_always_intersects_circle_shortest_chord_line_equation_l752_75292

open Real

noncomputable def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 9 = 0

noncomputable def line_eqn (m x y : ℝ) : Prop := 2 * m * x - 3 * m * y + x - y - 1 = 0

theorem line_always_intersects_circle (m : ℝ) : 
  ∀ (x y : ℝ), circle_eqn x y → line_eqn m x y → True := 
by
  sorry

theorem shortest_chord_line_equation : 
  ∃ (m x y : ℝ), line_eqn m x y ∧ (∀ x y, line_eqn m x y → x - y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_always_intersects_circle_shortest_chord_line_equation_l752_75292


namespace NUMINAMATH_GPT_units_digit_7_power_2023_l752_75255

theorem units_digit_7_power_2023 : ∃ d, d = 7^2023 % 10 ∧ d = 3 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_7_power_2023_l752_75255


namespace NUMINAMATH_GPT_total_acorns_l752_75258

theorem total_acorns (s_a : ℕ) (s_b : ℕ) (d : ℕ)
  (h1 : s_a = 7)
  (h2 : s_b = 5 * s_a)
  (h3 : s_b + 3 = d) :
  s_a + s_b + d = 80 :=
by
  sorry

end NUMINAMATH_GPT_total_acorns_l752_75258


namespace NUMINAMATH_GPT_ratio_of_volumes_l752_75287

theorem ratio_of_volumes (r : ℝ) (π : ℝ) (V1 V2 : ℝ) 
  (h1 : V2 = (4 / 3) * π * r^3) 
  (h2 : V1 = 2 * π * r^3) : 
  V1 / V2 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_l752_75287


namespace NUMINAMATH_GPT_triangle_centroid_l752_75283

theorem triangle_centroid :
  let (x1, y1) := (2, 6)
  let (x2, y2) := (6, 2)
  let (x3, y3) := (4, 8)
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  (centroid_x, centroid_y) = (4, 16 / 3) :=
by
  let x1 := 2
  let y1 := 6
  let x2 := 6
  let y2 := 2
  let x3 := 4
  let y3 := 8
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  show (centroid_x, centroid_y) = (4, 16 / 3)
  sorry

end NUMINAMATH_GPT_triangle_centroid_l752_75283


namespace NUMINAMATH_GPT_total_cost_l752_75221

variable (E P M : ℝ)

axiom condition1 : E + 3 * P + 2 * M = 240
axiom condition2 : 2 * E + 5 * P + 4 * M = 440

theorem total_cost : 3 * E + 4 * P + 6 * M = 520 := 
sorry

end NUMINAMATH_GPT_total_cost_l752_75221


namespace NUMINAMATH_GPT_CombinedHeightOfTowersIsCorrect_l752_75210

-- Define the heights as non-negative reals for clarity.
noncomputable def ClydeTowerHeight : ℝ := 5.0625
noncomputable def GraceTowerHeight : ℝ := 40.5
noncomputable def SarahTowerHeight : ℝ := 2 * ClydeTowerHeight
noncomputable def LindaTowerHeight : ℝ := (ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight) / 3
noncomputable def CombinedHeight : ℝ := ClydeTowerHeight + GraceTowerHeight + SarahTowerHeight + LindaTowerHeight

-- State the theorem to be proven
theorem CombinedHeightOfTowersIsCorrect : CombinedHeight = 74.25 := 
by
  sorry

end NUMINAMATH_GPT_CombinedHeightOfTowersIsCorrect_l752_75210


namespace NUMINAMATH_GPT_difference_between_max_and_min_area_l752_75235

noncomputable def max_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

noncomputable def min_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 60 then l * w else 0

theorem difference_between_max_and_min_area :
  ∃ (l_max l_min w_max w_min : ℕ),
    2 * l_max + 2 * w_max = 60 ∧
    2 * l_min + 2 * w_min = 60 ∧
    (l_max * w_max - l_min * w_min = 196) :=
by
  sorry

end NUMINAMATH_GPT_difference_between_max_and_min_area_l752_75235


namespace NUMINAMATH_GPT_natural_number_divisor_problem_l752_75284

theorem natural_number_divisor_problem (x y z : ℕ) (h1 : (y+1)*(z+1) = 30) 
    (h2 : (x+1)*(z+1) = 42) (h3 : (x+1)*(y+1) = 35) :
    (2^x * 3^y * 5^z = 2^6 * 3^5 * 5^4) :=
sorry

end NUMINAMATH_GPT_natural_number_divisor_problem_l752_75284


namespace NUMINAMATH_GPT_minimum_voters_for_tall_win_l752_75293

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end NUMINAMATH_GPT_minimum_voters_for_tall_win_l752_75293


namespace NUMINAMATH_GPT_amusement_park_line_l752_75298

theorem amusement_park_line (h1 : Eunji_position = 6) (h2 : people_behind_Eunji = 7) : total_people_in_line = 13 :=
by
  sorry

end NUMINAMATH_GPT_amusement_park_line_l752_75298


namespace NUMINAMATH_GPT_chord_length_of_intersection_l752_75233

theorem chord_length_of_intersection 
  (x y : ℝ) (h_line : 2 * x - y - 1 = 0) (h_circle : (x - 2)^2 + (y + 2)^2 = 9) : 
  ∃ l, l = 4 := 
sorry

end NUMINAMATH_GPT_chord_length_of_intersection_l752_75233


namespace NUMINAMATH_GPT_MrSmithEnglishProof_l752_75265

def MrSmithLearningEnglish : Prop :=
  (∃ (decade: String) (age: String), 
    (decade = "1950's" ∧ age = "in his sixties") ∨ 
    (decade = "1950" ∧ age = "in the sixties") ∨ 
    (decade = "1950's" ∧ age = "over sixty"))
  
def correctAnswer : Prop :=
  MrSmithLearningEnglish →
  (∃ answer, answer = "D")

theorem MrSmithEnglishProof : correctAnswer :=
  sorry

end NUMINAMATH_GPT_MrSmithEnglishProof_l752_75265


namespace NUMINAMATH_GPT_min_sum_squares_l752_75252

variable {a b c t : ℝ}

def min_value_of_sum_squares (a b c : ℝ) (t : ℝ) : ℝ :=
  a^2 + b^2 + c^2

theorem min_sum_squares (h : a + b + c = t) : min_value_of_sum_squares a b c t ≥ t^2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_squares_l752_75252


namespace NUMINAMATH_GPT_original_average_score_of_class_l752_75254

theorem original_average_score_of_class {A : ℝ} 
  (num_students : ℝ) 
  (grace_marks : ℝ) 
  (new_average : ℝ) 
  (h1 : num_students = 35) 
  (h2 : grace_marks = 3) 
  (h3 : new_average = 40)
  (h_total_new : 35 * new_average = 35 * A + 35 * grace_marks) :
  A = 37 :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_original_average_score_of_class_l752_75254


namespace NUMINAMATH_GPT_convex_pentagon_angle_greater_than_36_l752_75297

theorem convex_pentagon_angle_greater_than_36
  (α γ : ℝ)
  (h_sum : 5 * α + 10 * γ = 3 * Real.pi)
  (h_convex : ∀ i : Fin 5, (α + i.val * γ < Real.pi)) :
  α > Real.pi / 5 :=
sorry

end NUMINAMATH_GPT_convex_pentagon_angle_greater_than_36_l752_75297


namespace NUMINAMATH_GPT_angle_conversion_l752_75225

-- Define the known conditions
def full_circle_vens : ℕ := 800
def full_circle_degrees : ℕ := 360
def given_angle_degrees : ℕ := 135
def expected_vens : ℕ := 300

-- Prove that an angle of 135 degrees corresponds to 300 vens.
theorem angle_conversion :
  (given_angle_degrees * full_circle_vens) / full_circle_degrees = expected_vens := by
  sorry

end NUMINAMATH_GPT_angle_conversion_l752_75225


namespace NUMINAMATH_GPT_price_of_pen_l752_75277

theorem price_of_pen (price_pen : ℚ) (price_notebook : ℚ) :
  (price_pen + 3 * price_notebook = 36.45) →
  (price_notebook = 15 / 4 * price_pen) →
  price_pen = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_price_of_pen_l752_75277


namespace NUMINAMATH_GPT_lines_intersect_l752_75256

variables {s v : ℝ}

def line1 (s : ℝ) : ℝ × ℝ :=
  (3 - 2 * s, 4 + 3 * s)

def line2 (v : ℝ) : ℝ × ℝ :=
  (1 - 3 * v, 5 + 2 * v)

theorem lines_intersect :
  ∃ s v : ℝ, line1 s = line2 v ∧ line1 s = (25 / 13, 73 / 13) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_l752_75256


namespace NUMINAMATH_GPT_intersection_result_l752_75299

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def M_compl : Set ℝ := { x | x < 1 }

theorem intersection_result : N ∩ M_compl = { x | 0 ≤ x ∧ x < 1 } :=
by sorry

end NUMINAMATH_GPT_intersection_result_l752_75299


namespace NUMINAMATH_GPT_max_stamps_l752_75202

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h1 : price_per_stamp = 45) (h2 : total_money = 5000) : ∃ n : ℕ, n = 111 ∧ 45 * n ≤ 5000 ∧ ∀ m : ℕ, (45 * m ≤ 5000) → m ≤ n := 
by
  sorry

end NUMINAMATH_GPT_max_stamps_l752_75202


namespace NUMINAMATH_GPT_linear_equation_value_m_l752_75204

theorem linear_equation_value_m (m : ℝ) (h : ∀ x : ℝ, 2 * x^(m - 1) + 3 = 0 → x ≠ 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_linear_equation_value_m_l752_75204


namespace NUMINAMATH_GPT_sqrt_expression_meaningful_l752_75274

theorem sqrt_expression_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) :=
by
  -- Proof will be skipped
  sorry

end NUMINAMATH_GPT_sqrt_expression_meaningful_l752_75274


namespace NUMINAMATH_GPT_range_of_x_satisfying_inequality_l752_75286

theorem range_of_x_satisfying_inequality (x : ℝ) : 
  (|x+1| + |x| < 2) ↔ (-3/2 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_GPT_range_of_x_satisfying_inequality_l752_75286


namespace NUMINAMATH_GPT_determine_g_10_l752_75296

noncomputable def g : ℝ → ℝ := sorry

-- Given condition
axiom g_condition : ∀ x y : ℝ, g x + g (2 * x + y) + 7 * x * y = g (3 * x - y) + 3 * x ^ 2 + 4

-- Theorem to prove
theorem determine_g_10 : g 10 = -46 := 
by
  -- skipping the proof here
  sorry

end NUMINAMATH_GPT_determine_g_10_l752_75296


namespace NUMINAMATH_GPT_certain_number_proof_l752_75227

noncomputable def certain_number : ℝ := 30

theorem certain_number_proof (h1: 0.60 * 50 = 30) (h2: 30 = 0.40 * certain_number + 18) : 
  certain_number = 30 := 
sorry

end NUMINAMATH_GPT_certain_number_proof_l752_75227


namespace NUMINAMATH_GPT_tan_theta_minus_pi_over4_l752_75290

theorem tan_theta_minus_pi_over4 (θ : Real) (h : Real.cos θ - 3 * Real.sin θ = 0) : 
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_tan_theta_minus_pi_over4_l752_75290


namespace NUMINAMATH_GPT_plates_usage_when_parents_join_l752_75278

theorem plates_usage_when_parents_join
  (total_plates : ℕ)
  (plates_per_day_matt_and_son : ℕ)
  (days_matt_and_son : ℕ)
  (days_with_parents : ℕ)
  (total_days_in_week : ℕ)
  (total_plates_needed : total_plates = 38)
  (plates_used_matt_and_son : plates_per_day_matt_and_son = 2)
  (days_matt_and_son_eq : days_matt_and_son = 3)
  (days_with_parents_eq : days_with_parents = 4)
  (total_days_in_week_eq : total_days_in_week = 7)
  (plates_used_when_parents_join : total_plates - plates_per_day_matt_and_son * days_matt_and_son = days_with_parents * 8) :
  true :=
sorry

end NUMINAMATH_GPT_plates_usage_when_parents_join_l752_75278


namespace NUMINAMATH_GPT_compute_ns_l752_75238

noncomputable def f : ℝ → ℝ :=
sorry

-- Defining the functional equation as a condition
def functional_equation (f : ℝ → ℝ) :=
∀ x y z : ℝ, f (x^2 + y^2 * f z) = x * f x + z * f (y^2)

-- Proving that the number of possible values of f(5) is 2
-- and their sum is 5, thus n * s = 10
theorem compute_ns (f : ℝ → ℝ) (hf : functional_equation f) : 2 * 5 = 10 :=
sorry

end NUMINAMATH_GPT_compute_ns_l752_75238


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_range_k_quadratic_real_roots_specific_value_k_l752_75212

theorem quadratic_real_roots_iff_range_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + k + 1 = 0 ∧ x2^2 - 4 * x2 + k + 1 = 0 ∧ x1 ≠ x2) ↔ k ≤ 3 :=
by
  sorry

theorem quadratic_real_roots_specific_value_k (k : ℝ) (x1 x2 : ℝ) :
  x1^2 - 4 * x1 + k + 1 = 0 →
  x2^2 - 4 * x2 + k + 1 = 0 →
  x1 ≠ x2 →
  (3 / x1 + 3 / x2 = x1 * x2 - 4) →
  k = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_range_k_quadratic_real_roots_specific_value_k_l752_75212


namespace NUMINAMATH_GPT_roger_daily_goal_l752_75232

-- Conditions
def steps_in_30_minutes : ℕ := 2000
def time_to_reach_goal_min : ℕ := 150
def time_interval_min : ℕ := 30

-- Theorem to prove
theorem roger_daily_goal : steps_in_30_minutes * (time_to_reach_goal_min / time_interval_min) = 10000 := by
  sorry

end NUMINAMATH_GPT_roger_daily_goal_l752_75232


namespace NUMINAMATH_GPT_all_options_valid_l752_75246

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Definitions of parameterizations for each option
def option_A (t : ℝ) : ℝ × ℝ := ⟨2 + (-1) * t, 0 + (-2) * t⟩
def option_B (t : ℝ) : ℝ × ℝ := ⟨6 + 4 * t, 8 + 8 * t⟩
def option_C (t : ℝ) : ℝ × ℝ := ⟨1 + 1 * t, -2 + 2 * t⟩
def option_D (t : ℝ) : ℝ × ℝ := ⟨0 + 0.5 * t, -4 + 1 * t⟩
def option_E (t : ℝ) : ℝ × ℝ := ⟨-2 + (-2) * t, -8 + (-4) * t⟩

-- The main statement to prove
theorem all_options_valid :
  (∀ t, line_eq (option_A t).1 (option_A t).2) ∧
  (∀ t, line_eq (option_B t).1 (option_B t).2) ∧
  (∀ t, line_eq (option_C t).1 (option_C t).2) ∧
  (∀ t, line_eq (option_D t).1 (option_D t).2) ∧
  (∀ t, line_eq (option_E t).1 (option_E t).2) :=
by sorry -- proof omitted

end NUMINAMATH_GPT_all_options_valid_l752_75246


namespace NUMINAMATH_GPT_smallest_number_of_students_l752_75231

theorem smallest_number_of_students (n : ℕ) (x : ℕ) 
  (h_total : n = 5 * x + 3) 
  (h_more_than_50 : n > 50) : 
  n = 53 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_of_students_l752_75231


namespace NUMINAMATH_GPT_part1_part2_part3_l752_75228

noncomputable def f (x m : ℝ) : ℝ :=
  -x^2 + m*x - m

-- Part (1)
theorem part1 (m : ℝ) : (∀ x, f x m ≤ 0) → (m = 0 ∨ m = 4) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x, -1 ≤ x ∧ x ≤ 0 → f x m ≤ f (-1) m) → (m ≤ -2) :=
sorry

-- Part (3)
theorem part3 : ∃ (m : ℝ), (∀ x, 2 ≤ x ∧ x ≤ 3 → (2 ≤ f x m ∧ f x m ≤ 3)) → m = 6 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l752_75228


namespace NUMINAMATH_GPT_xt_inequality_least_constant_l752_75218

theorem xt_inequality (x y z t : ℝ) (h : x ≤ y ∧ y ≤ z ∧ z ≤ t) (h_sum : x * y + x * z + x * t + y * z + y * t + z * t = 1) :
  x * t < 1 / 3 := sorry

theorem least_constant (x y z t : ℝ) (h : x ≤ y ∧ y ≤ z ∧ z ≤ t) (h_sum : x * y + x * z + x * t + y * z + y * t + z * t = 1) :
  ∃ C, ∀ (x t : ℝ), xt < C ∧ C = 1 / 3 := sorry

end NUMINAMATH_GPT_xt_inequality_least_constant_l752_75218


namespace NUMINAMATH_GPT_binomial_seven_four_l752_75294

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end NUMINAMATH_GPT_binomial_seven_four_l752_75294


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l752_75281

theorem perfect_square_trinomial_k (k : ℤ) :
  (∀ x : ℝ, 9 * x^2 + 6 * x + k = (3 * x + 1) ^ 2) → (k = 1) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l752_75281


namespace NUMINAMATH_GPT_minimum_moves_black_white_swap_l752_75285

-- Define an initial setup of the chessboard
def initial_positions_black := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8)]
def initial_positions_white := [(8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8)]

-- Define chess rules, positions, and switching places
def black_to_white_target := initial_positions_white
def white_to_black_target := initial_positions_black

-- Define a function to count minimal moves (trivial here just for the purpose of this statement)
def min_moves_to_switch_positions := 23

-- The main theorem statement proving necessity of at least 23 moves
theorem minimum_moves_black_white_swap :
  ∀ (black_positions white_positions : List (ℕ × ℕ)),
  black_positions = initial_positions_black →
  white_positions = initial_positions_white →
  min_moves_to_switch_positions ≥ 23 :=
by
  sorry

end NUMINAMATH_GPT_minimum_moves_black_white_swap_l752_75285


namespace NUMINAMATH_GPT_common_tangent_x_eq_neg1_l752_75222
open Real

-- Definitions of circles C₁ and C₂
def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def circle2 := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Statement of the problem
theorem common_tangent_x_eq_neg1 :
  ∀ (x : ℝ) (y : ℝ),
    (x, y) ∈ circle1 ∧ (x, y) ∈ circle2 → x = -1 :=
sorry

end NUMINAMATH_GPT_common_tangent_x_eq_neg1_l752_75222


namespace NUMINAMATH_GPT_john_and_mike_safe_weight_l752_75241

def weight_bench_max_support : ℕ := 1000
def safety_margin_percentage : ℕ := 20
def john_weight : ℕ := 250
def mike_weight : ℕ := 180

def safety_margin : ℕ := (safety_margin_percentage * weight_bench_max_support) / 100
def max_safe_weight : ℕ := weight_bench_max_support - safety_margin
def combined_weight : ℕ := john_weight + mike_weight
def weight_on_bar_together : ℕ := max_safe_weight - combined_weight

theorem john_and_mike_safe_weight :
  weight_on_bar_together = 370 := by
  sorry

end NUMINAMATH_GPT_john_and_mike_safe_weight_l752_75241


namespace NUMINAMATH_GPT_min_soda_packs_90_l752_75219

def soda_packs (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), 6 * x + 12 * y + 24 * z = n

theorem min_soda_packs_90 : (x y z : ℕ) → soda_packs 90 → x + y + z = 5 := by
  sorry

end NUMINAMATH_GPT_min_soda_packs_90_l752_75219


namespace NUMINAMATH_GPT_julie_hourly_rate_l752_75217

variable (daily_hours : ℕ) (weekly_days : ℕ) (monthly_weeks : ℕ) (missed_days : ℕ) (monthly_salary : ℝ)

def total_monthly_hours : ℕ := daily_hours * weekly_days * monthly_weeks - daily_hours * missed_days

theorem julie_hourly_rate : 
    daily_hours = 8 → 
    weekly_days = 6 → 
    monthly_weeks = 4 → 
    missed_days = 1 → 
    monthly_salary = 920 → 
    (monthly_salary / total_monthly_hours daily_hours weekly_days monthly_weeks missed_days) = 5 := by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end NUMINAMATH_GPT_julie_hourly_rate_l752_75217


namespace NUMINAMATH_GPT_find_number_l752_75266

theorem find_number (x n : ℝ) (h1 : (3 / 2) * x - n = 15) (h2 : x = 12) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l752_75266


namespace NUMINAMATH_GPT_algebraic_expression_value_l752_75240

theorem algebraic_expression_value (x y : ℕ) (h : 3 * x - y = 1) : (8^x : ℝ) / (2^y) / 2 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l752_75240


namespace NUMINAMATH_GPT_Juanico_age_30_years_from_now_l752_75209

-- Definitions and hypothesis
def currentAgeGladys : ℕ := 30 -- Gladys's current age, since she will be 40 in 10 years
def currentAgeJuanico : ℕ := (1 / 2) * currentAgeGladys - 4 -- Juanico's current age based on Gladys's current age

theorem Juanico_age_30_years_from_now :
  currentAgeJuanico + 30 = 41 :=
by
  -- You would normally fill out the proof here, but we use 'sorry' to skip it.
  sorry

end NUMINAMATH_GPT_Juanico_age_30_years_from_now_l752_75209


namespace NUMINAMATH_GPT_codys_grandmother_age_l752_75243

theorem codys_grandmother_age (cody_age : ℕ) (grandmother_factor : ℕ) (h1 : cody_age = 14) (h2 : grandmother_factor = 6) :
  grandmother_factor * cody_age = 84 :=
by
  sorry

end NUMINAMATH_GPT_codys_grandmother_age_l752_75243


namespace NUMINAMATH_GPT_number_of_students_in_class_l752_75291

theorem number_of_students_in_class
  (G : ℕ) (E_and_G : ℕ) (E_only: ℕ)
  (h1 : G = 22)
  (h2 : E_and_G = 12)
  (h3 : E_only = 23) :
  ∃ S : ℕ, S = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_in_class_l752_75291


namespace NUMINAMATH_GPT_find_alpha_l752_75249

theorem find_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2))
  (h2 : ∃ k : ℝ, (Real.cos α, Real.sin α) = k • (-3, -3)) :
  α = 3 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_l752_75249


namespace NUMINAMATH_GPT_divisible_by_five_l752_75201

theorem divisible_by_five (a : ℤ) (h : ¬ (5 ∣ a)) :
    (5 ∣ (a^2 - 1)) ↔ ¬ (5 ∣ (a^2 + 1)) :=
by
  -- Begin the proof here (proof not required according to instructions)
  sorry

end NUMINAMATH_GPT_divisible_by_five_l752_75201


namespace NUMINAMATH_GPT_gf_3_eq_495_l752_75279

def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := 3 * x^2 - x + 1

theorem gf_3_eq_495 : g (f 3) = 495 := by
  sorry

end NUMINAMATH_GPT_gf_3_eq_495_l752_75279


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l752_75206

theorem problem1 : -20 + (-14) - (-18) - 13 = -29 := by
  sorry

theorem problem2 : (-2) * 3 + (-5) - 4 / (-1/2) = -3 := by
  sorry

theorem problem3 : (-3/8 - 1/6 + 3/4) * (-24) = -5 := by
  sorry

theorem problem4 : -81 / (9/4) * abs (-4/9) - (-3)^3 / 27 = -15 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l752_75206


namespace NUMINAMATH_GPT_num_unique_triangle_areas_correct_l752_75259

noncomputable def num_unique_triangle_areas : ℕ :=
  let A := 0
  let B := 1
  let C := 3
  let D := 6
  let E := 0
  let F := 2
  let base_lengths := [1, 2, 3, 5, 6]
  (base_lengths.eraseDups).length

theorem num_unique_triangle_areas_correct : num_unique_triangle_areas = 5 :=
  by sorry

end NUMINAMATH_GPT_num_unique_triangle_areas_correct_l752_75259


namespace NUMINAMATH_GPT_squirrels_more_than_nuts_l752_75229

theorem squirrels_more_than_nuts 
  (squirrels : ℕ) 
  (nuts : ℕ) 
  (h_squirrels : squirrels = 4) 
  (h_nuts : nuts = 2) 
  : squirrels - nuts = 2 :=
by
  sorry

end NUMINAMATH_GPT_squirrels_more_than_nuts_l752_75229


namespace NUMINAMATH_GPT_modulus_of_z_l752_75230

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 + 2 * I) : abs z = 2 := 
sorry

end NUMINAMATH_GPT_modulus_of_z_l752_75230


namespace NUMINAMATH_GPT_XiaoZhang_four_vcd_probability_l752_75263

noncomputable def probability_four_vcd (zhang_vcd zhang_dvd wang_vcd wang_dvd : ℕ) : ℚ :=
  (4 * 2 / (7 * 3)) + (3 * 1 / (7 * 3))

theorem XiaoZhang_four_vcd_probability :
  probability_four_vcd 4 3 2 1 = 11 / 21 :=
by
  sorry

end NUMINAMATH_GPT_XiaoZhang_four_vcd_probability_l752_75263


namespace NUMINAMATH_GPT_quadratic_roots_two_l752_75273

theorem quadratic_roots_two (m : ℝ) :
  let a := 1
  let b := -m
  let c := m - 2
  let Δ := b^2 - 4 * a * c
  Δ > 0 :=
by
  let a := 1
  let b := -m
  let c := m - 2
  let Δ := b^2 - 4 * a * c
  sorry

end NUMINAMATH_GPT_quadratic_roots_two_l752_75273


namespace NUMINAMATH_GPT_g_of_2_l752_75276

noncomputable def g : ℝ → ℝ := sorry

axiom cond1 (x y : ℝ) : x * g y = y * g x
axiom cond2 : g 10 = 30

theorem g_of_2 : g 2 = 6 := by
  sorry

end NUMINAMATH_GPT_g_of_2_l752_75276


namespace NUMINAMATH_GPT_greatest_two_digit_prod_12_l752_75261

theorem greatest_two_digit_prod_12 : ∃(n : ℕ), n < 100 ∧ n ≥ 10 ∧
  (∃(d1 d2 : ℕ), n = 10 * d1 + d2 ∧ d1 * d2 = 12) ∧ ∀(k : ℕ), k < 100 ∧ k ≥ 10 ∧ (∃(d1 d2 : ℕ), k = 10 * d1 + d2 ∧ d1 * d2 = 12) → k ≤ 62 :=
by
  sorry

end NUMINAMATH_GPT_greatest_two_digit_prod_12_l752_75261


namespace NUMINAMATH_GPT_min_value_F_l752_75268

theorem min_value_F :
  ∀ (x y : ℝ), (x^2 + y^2 - 2*x - 2*y + 1 = 0) → (x + 1) / y ≥ 3 / 4 :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_min_value_F_l752_75268


namespace NUMINAMATH_GPT_max_rooks_max_rooks_4x4_max_rooks_8x8_l752_75295

theorem max_rooks (n : ℕ) : ℕ :=
  2 * (2 * n / 3)

theorem max_rooks_4x4 :
  max_rooks 4 = 4 :=
  sorry

theorem max_rooks_8x8 :
  max_rooks 8 = 10 :=
  sorry

end NUMINAMATH_GPT_max_rooks_max_rooks_4x4_max_rooks_8x8_l752_75295


namespace NUMINAMATH_GPT_sample_freq_0_40_l752_75280

def total_sample_size : ℕ := 100
def freq_group_0_10 : ℕ := 12
def freq_group_10_20 : ℕ := 13
def freq_group_20_30 : ℕ := 24
def freq_group_30_40 : ℕ := 15
def freq_group_40_50 : ℕ := 16
def freq_group_50_60 : ℕ := 13
def freq_group_60_70 : ℕ := 7

theorem sample_freq_0_40 : (freq_group_0_10 + freq_group_10_20 + freq_group_20_30 + freq_group_30_40) / (total_sample_size : ℝ) = 0.64 := by
  sorry

end NUMINAMATH_GPT_sample_freq_0_40_l752_75280


namespace NUMINAMATH_GPT_axis_of_symmetry_l752_75272

theorem axis_of_symmetry (x : ℝ) : 
  ∀ y, y = x^2 - 2 * x - 3 → (∃ k : ℝ, k = 1 ∧ ∀ x₀ : ℝ, y = (x₀ - k)^2 + C) := 
sorry

end NUMINAMATH_GPT_axis_of_symmetry_l752_75272


namespace NUMINAMATH_GPT_part1_part2_l752_75253

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (x - Real.pi / 3))

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 :
  {x : ℝ | f x < 1 / 4} = {x : ℝ | ∃ k : ℤ, x ∈ Set.Ioo (k * Real.pi - 7 * Real.pi / 12) (k * Real.pi - Real.pi / 12)} :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l752_75253


namespace NUMINAMATH_GPT_gingerbreads_per_tray_l752_75257

theorem gingerbreads_per_tray (x : ℕ) (h : 4 * x + 3 * 20 = 160) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_gingerbreads_per_tray_l752_75257


namespace NUMINAMATH_GPT_count_valid_m_values_l752_75207

theorem count_valid_m_values : ∃ (count : ℕ), count = 72 ∧
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5000 →
     (⌊Real.sqrt m⌋ = ⌊Real.sqrt (m+125)⌋)) ↔ count = 72 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_m_values_l752_75207


namespace NUMINAMATH_GPT_combined_transform_is_correct_l752_75216

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def reflection_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

def combined_transform (dilation_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  dilation_matrix dilation_factor * reflection_x_matrix

theorem combined_transform_is_correct :
  combined_transform 5 = !![5, 0; 0, -5] :=
by
  sorry

end NUMINAMATH_GPT_combined_transform_is_correct_l752_75216


namespace NUMINAMATH_GPT_unique_solution_l752_75236

def satisfies_equation (m n : ℕ) : Prop :=
  15 * m * n = 75 - 5 * m - 3 * n

theorem unique_solution : satisfies_equation 1 6 ∧ ∀ (m n : ℕ), m > 0 → n > 0 → satisfies_equation m n → (m, n) = (1, 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_l752_75236


namespace NUMINAMATH_GPT_complex_number_solution_l752_75237

open Complex

noncomputable def findComplexNumber (z : ℂ) : Prop :=
  abs (z - 2) = abs (z + 4) ∧ abs (z - 2) = abs (z - Complex.I * 2)

theorem complex_number_solution :
  ∃ z : ℂ, findComplexNumber z ∧ z = -1 + Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l752_75237
