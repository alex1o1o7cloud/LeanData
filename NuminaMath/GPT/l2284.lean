import Mathlib

namespace NUMINAMATH_GPT_solve_for_buttons_l2284_228442

def number_of_buttons_on_second_shirt (x : ℕ) : Prop :=
  200 * 3 + 200 * x = 1600

theorem solve_for_buttons : ∃ x : ℕ, number_of_buttons_on_second_shirt x ∧ x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_for_buttons_l2284_228442


namespace NUMINAMATH_GPT_number_of_puppies_l2284_228487

def total_portions : Nat := 105
def feeding_days : Nat := 5
def feedings_per_day : Nat := 3

theorem number_of_puppies (total_portions feeding_days feedings_per_day : Nat) : 
  (total_portions / feeding_days / feedings_per_day = 7) := 
by 
  sorry

end NUMINAMATH_GPT_number_of_puppies_l2284_228487


namespace NUMINAMATH_GPT_Darius_scored_10_points_l2284_228474

theorem Darius_scored_10_points
  (D Marius Matt : ℕ)
  (h1 : Marius = D + 3)
  (h2 : Matt = D + 5)
  (h3 : D + Marius + Matt = 38) : 
  D = 10 :=
by
  sorry

end NUMINAMATH_GPT_Darius_scored_10_points_l2284_228474


namespace NUMINAMATH_GPT_price_of_adult_ticket_l2284_228491

/--
Given:
1. The price of a child's ticket is half the price of an adult's ticket.
2. Janet buys tickets for 10 people, 4 of whom are children.
3. Janet buys a soda for $5.
4. With the soda, Janet gets a 20% discount on the total admission price.
5. Janet paid $197 in total for everything.

Prove that the price of an adult admission ticket is $30.
-/
theorem price_of_adult_ticket : 
  ∃ (A : ℝ), 
  (∀ (childPrice adultPrice total : ℝ),
    adultPrice = A →
    childPrice = A / 2 →
    total = adultPrice * 6 + childPrice * 4 →
    totalPriceWithDiscount = 192 →
    total / 0.8 = total + 5 →
    A = 30) :=
sorry

end NUMINAMATH_GPT_price_of_adult_ticket_l2284_228491


namespace NUMINAMATH_GPT_football_cost_is_correct_l2284_228439

def total_spent_on_toys : ℝ := 12.30
def spent_on_marbles : ℝ := 6.59
def spent_on_football := total_spent_on_toys - spent_on_marbles

theorem football_cost_is_correct : spent_on_football = 5.71 :=
by
  sorry

end NUMINAMATH_GPT_football_cost_is_correct_l2284_228439


namespace NUMINAMATH_GPT_product_largest_smallest_using_digits_l2284_228430

theorem product_largest_smallest_using_digits (a b : ℕ) (h1 : 100 * 6 + 10 * 2 + 0 = a) (h2 : 100 * 2 + 10 * 0 + 6 = b) : a * b = 127720 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_product_largest_smallest_using_digits_l2284_228430


namespace NUMINAMATH_GPT_certain_percentage_of_1600_l2284_228418

theorem certain_percentage_of_1600 (P : ℝ) 
  (h : 0.05 * (P / 100 * 1600) = 20) : 
  P = 25 :=
by 
  sorry

end NUMINAMATH_GPT_certain_percentage_of_1600_l2284_228418


namespace NUMINAMATH_GPT_correct_factorization_l2284_228479

theorem correct_factorization :
  (∀ (x y : ℝ), x^2 + y^2 ≠ (x + y)^2) ∧
  (∀ (x y : ℝ), x^2 + 2*x*y + y^2 ≠ (x - y)^2) ∧
  (∀ (x : ℝ), x^2 + x ≠ x * (x - 1)) ∧
  (∀ (x y : ℝ), x^2 - y^2 = (x + y) * (x - y)) :=
by 
  sorry

end NUMINAMATH_GPT_correct_factorization_l2284_228479


namespace NUMINAMATH_GPT_max_value_of_M_l2284_228427

noncomputable def M (x y z : ℝ) := min (min x y) z

theorem max_value_of_M
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_zero : b^2 - 4 * a * c ≥ 0) :
  M ((b + c) / a) ((c + a) / b) ((a + b) / c) ≤ 5 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_M_l2284_228427


namespace NUMINAMATH_GPT_triangle_area_inscribed_in_circle_l2284_228431

theorem triangle_area_inscribed_in_circle :
  ∀ (x : ℝ), (2 * x)^2 + (3 * x)^2 = (4 * x)^2 → (5 = (4 * x) / 2) → (1/2 * (2 * x) * (3 * x) = 18.75) :=
by
  -- Assume all necessary conditions
  intros x h_ratio h_radius
  -- Skip the proof part using sorry
  sorry

end NUMINAMATH_GPT_triangle_area_inscribed_in_circle_l2284_228431


namespace NUMINAMATH_GPT_solve_inequality_solve_system_of_inequalities_l2284_228459

-- Inequality proof problem
theorem solve_inequality (x : ℝ) (h : (2*x - 3)/3 > (3*x + 1)/6 - 1) : x > 1 := by
  sorry

-- System of inequalities proof problem
theorem solve_system_of_inequalities (x : ℝ) (h1 : x ≤ 3*x - 6) (h2 : 3*x + 1 > 2*(x - 1)) : x ≥ 3 := by
  sorry

end NUMINAMATH_GPT_solve_inequality_solve_system_of_inequalities_l2284_228459


namespace NUMINAMATH_GPT_find_DY_length_l2284_228419

noncomputable def angle_bisector_theorem (DE DY EF FY : ℝ) : ℝ :=
  (DE * FY) / EF

theorem find_DY_length :
  ∀ (DE EF FY : ℝ), DE = 26 → EF = 34 → FY = 30 →
  angle_bisector_theorem DE DY EF FY = 22.94 := 
by
  intros
  sorry

end NUMINAMATH_GPT_find_DY_length_l2284_228419


namespace NUMINAMATH_GPT_find_z_l2284_228451

open Complex

theorem find_z (z : ℂ) (h : ((1 - I) ^ 2) / z = 1 + I) : z = -1 - I :=
sorry

end NUMINAMATH_GPT_find_z_l2284_228451


namespace NUMINAMATH_GPT_greater_number_l2284_228445

theorem greater_number (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 2) (h3 : a > b) : a = 21 := by
  sorry

end NUMINAMATH_GPT_greater_number_l2284_228445


namespace NUMINAMATH_GPT_new_person_weight_l2284_228404

theorem new_person_weight (W x : ℝ) (h1 : (W - 55 + x) / 8 = (W / 8) + 2.5) : x = 75 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_new_person_weight_l2284_228404


namespace NUMINAMATH_GPT_probability_of_picking_letter_from_MATHEMATICS_l2284_228488

theorem probability_of_picking_letter_from_MATHEMATICS : 
  (8 : ℤ) / 26 = (4 : ℤ) / 13 :=
by
  norm_num

end NUMINAMATH_GPT_probability_of_picking_letter_from_MATHEMATICS_l2284_228488


namespace NUMINAMATH_GPT_total_cost_jello_l2284_228470

def total_cost_james_spent : Real := 259.20

theorem total_cost_jello 
  (pounds_per_cubic_foot : ℝ := 8)
  (gallons_per_cubic_foot : ℝ := 7.5)
  (tablespoons_per_pound : ℝ := 1.5)
  (cost_red_jello : ℝ := 0.50)
  (cost_blue_jello : ℝ := 0.40)
  (cost_green_jello : ℝ := 0.60)
  (percentage_red_jello : ℝ := 0.60)
  (percentage_blue_jello : ℝ := 0.30)
  (percentage_green_jello : ℝ := 0.10)
  (volume_cubic_feet : ℝ := 6) :
  (volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_red_jello * cost_red_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_blue_jello * cost_blue_jello
   + volume_cubic_feet * gallons_per_cubic_foot * pounds_per_cubic_foot * tablespoons_per_pound * percentage_green_jello * cost_green_jello) = total_cost_james_spent :=
by
  sorry

end NUMINAMATH_GPT_total_cost_jello_l2284_228470


namespace NUMINAMATH_GPT_flower_problem_solution_l2284_228468

/-
Given the problem conditions:
1. There are 88 flowers.
2. Each flower was visited by at least one bee.
3. Each bee visited exactly 54 flowers.

Prove that bitter flowers exceed sweet flowers by 14.
-/

noncomputable def flower_problem : Prop :=
  ∃ (s g : ℕ), 
    -- Condition: The total number of flowers
    s + g + (88 - s - g) = 88 ∧ 
    -- Condition: Total number of visits by bees
    3 * 54 = 162 ∧ 
    -- Proof goal: Bitter flowers exceed sweet flowers by 14
    g - s = 14

theorem flower_problem_solution : flower_problem :=
by
  sorry

end NUMINAMATH_GPT_flower_problem_solution_l2284_228468


namespace NUMINAMATH_GPT_four_nonzero_complex_numbers_form_square_l2284_228478

open Complex

theorem four_nonzero_complex_numbers_form_square :
  ∃ (S : Finset ℂ), S.card = 4 ∧ (∀ z ∈ S, z ≠ 0) ∧ (∀ z ∈ S, ∃ (θ : ℝ), z = exp (θ * I) ∧ (exp (4 * θ * I) - z).re = 0 ∧ (exp (4 * θ * I) - z).im = cos (π / 2)) := 
sorry

end NUMINAMATH_GPT_four_nonzero_complex_numbers_form_square_l2284_228478


namespace NUMINAMATH_GPT_sufficient_not_necessary_ellipse_l2284_228407

theorem sufficient_not_necessary_ellipse (m n : ℝ) (h : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m ≠ n) ∧
  ¬(∀ x y : ℝ, mx^2 + ny^2 = 1 → m > 0 ∧ n > 0 ∧ m > n ∧ n > 0) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_ellipse_l2284_228407


namespace NUMINAMATH_GPT_tracy_michelle_distance_ratio_l2284_228481

theorem tracy_michelle_distance_ratio :
  ∀ (T M K : ℕ), 
  (M = 294) → 
  (M = 3 * K) → 
  (T + M + K = 1000) →
  ∃ x : ℕ, (T = x * M + 20) ∧ x = 2 :=
by
  intro T M K
  intro hM hMK hDistance
  use 2
  sorry

end NUMINAMATH_GPT_tracy_michelle_distance_ratio_l2284_228481


namespace NUMINAMATH_GPT_total_tin_in_new_alloy_l2284_228420

-- Define the weights of alloy A and alloy B
def weightAlloyA : Float := 135
def weightAlloyB : Float := 145

-- Define the ratio of lead to tin in alloy A
def ratioLeadToTinA : Float := 3 / 5

-- Define the ratio of tin to copper in alloy B
def ratioTinToCopperB : Float := 2 / 3

-- Define the total parts for alloy A and alloy B
def totalPartsA : Float := 3 + 5
def totalPartsB : Float := 2 + 3

-- Define the fraction of tin in alloy A and alloy B
def fractionTinA : Float := 5 / totalPartsA
def fractionTinB : Float := 2 / totalPartsB

-- Calculate the amount of tin in alloy A and alloy B
def tinInAlloyA : Float := fractionTinA * weightAlloyA
def tinInAlloyB : Float := fractionTinB * weightAlloyB

-- Calculate the total amount of tin in the new alloy
def totalTinInNewAlloy : Float := tinInAlloyA + tinInAlloyB

-- The theorem to be proven
theorem total_tin_in_new_alloy : totalTinInNewAlloy = 142.375 := by
  sorry

end NUMINAMATH_GPT_total_tin_in_new_alloy_l2284_228420


namespace NUMINAMATH_GPT_calculate_f_of_g_l2284_228466

def g (x : ℝ) := 4 * x + 6
def f (x : ℝ) := 6 * x - 10

theorem calculate_f_of_g :
  f (g 10) = 266 := by
  sorry

end NUMINAMATH_GPT_calculate_f_of_g_l2284_228466


namespace NUMINAMATH_GPT_linear_function_not_passing_through_third_quadrant_l2284_228436

theorem linear_function_not_passing_through_third_quadrant
  (m : ℝ)
  (h : 4 + 4 * m < 0) : 
  ∀ x y : ℝ, (y = m * x - m) → ¬ (x < 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_not_passing_through_third_quadrant_l2284_228436


namespace NUMINAMATH_GPT_min_number_of_candy_kinds_l2284_228484

theorem min_number_of_candy_kinds (n : ℕ) (h : n = 91)
  (even_distance_condition : ∀ i j : ℕ, i ≠ j → n > i ∧ n > j → 
    (∃k : ℕ, i - j = 2 * k) ∨ (∃k : ℕ, j - i = 2 * k) ) :
  91 / 2 =  46 := by
  sorry

end NUMINAMATH_GPT_min_number_of_candy_kinds_l2284_228484


namespace NUMINAMATH_GPT_total_earnings_proof_l2284_228447

-- Definitions of the given conditions
def monthly_earning : ℕ := 4000
def monthly_saving : ℕ := 500
def total_savings_needed : ℕ := 45000

-- Lean statement for the proof problem
theorem total_earnings_proof : 
  (total_savings_needed / monthly_saving) * monthly_earning = 360000 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_proof_l2284_228447


namespace NUMINAMATH_GPT_total_kids_at_camp_l2284_228463

-- Definition of the conditions
def kids_from_lawrence_camp : ℕ := 34044
def kids_from_outside_camp : ℕ := 424944

-- The proof statement
theorem total_kids_at_camp : kids_from_lawrence_camp + kids_from_outside_camp = 459988 := by
  sorry

end NUMINAMATH_GPT_total_kids_at_camp_l2284_228463


namespace NUMINAMATH_GPT_suitable_chart_for_air_composition_l2284_228469

/-- Given that air is a mixture of various gases, prove that the most suitable
    type of statistical chart to depict this data, while introducing it
    succinctly and effectively, is a pie chart. -/
theorem suitable_chart_for_air_composition :
  ∀ (air_composition : String) (suitable_for_introduction : String → Prop),
  (air_composition = "mixture of various gases") →
  (suitable_for_introduction "pie chart") →
  suitable_for_introduction "pie chart" :=
by
  intros air_composition suitable_for_introduction h_air_composition h_pie_chart
  sorry

end NUMINAMATH_GPT_suitable_chart_for_air_composition_l2284_228469


namespace NUMINAMATH_GPT_max_value_of_k_l2284_228435

theorem max_value_of_k:
  ∃ (k : ℕ), 
  (∀ (a b : ℕ → ℕ) (h : ∀ i, a i < b i) (no_share : ∀ i j, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)) (distinct_sums : ∀ i j, i ≠ j → a i + b i ≠ a j + b j) (sum_limit : ∀ i, a i + b i ≤ 3011), 
    k ≤ 3011 ∧ k = 1204) := sorry

end NUMINAMATH_GPT_max_value_of_k_l2284_228435


namespace NUMINAMATH_GPT_triangle_perimeter_is_720_l2284_228490

-- Definitions corresponding to conditions
variables (x : ℕ)
noncomputable def shortest_side := 5 * x
noncomputable def middle_side := 6 * x
noncomputable def longest_side := 7 * x

-- Given the length of the longest side is 280 cm
axiom longest_side_eq : longest_side x = 280

-- Prove that the perimeter of the triangle is 720 cm
theorem triangle_perimeter_is_720 : 
  shortest_side x + middle_side x + longest_side x = 720 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_is_720_l2284_228490


namespace NUMINAMATH_GPT_f_inv_f_inv_14_l2284_228403

noncomputable def f (x : ℝ) : ℝ := 3 * x + 7

noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 3

theorem f_inv_f_inv_14 : f_inv (f_inv 14) = -14 / 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_f_inv_f_inv_14_l2284_228403


namespace NUMINAMATH_GPT_regular_polygon_sides_l2284_228475

theorem regular_polygon_sides (n : ℕ) (h₁ : n ≥ 3) (h₂ : 120 = 180 * (n - 2) / n) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2284_228475


namespace NUMINAMATH_GPT_calculate_value_l2284_228494

theorem calculate_value (x y d : ℕ) (hx : x = 2024) (hy : y = 1935) (hd : d = 225) : 
  (x - y)^2 / d = 35 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l2284_228494


namespace NUMINAMATH_GPT_Q_share_of_profit_l2284_228453

def P_investment : ℕ := 54000
def Q_investment : ℕ := 36000
def total_profit : ℕ := 18000

theorem Q_share_of_profit : Q_investment * total_profit / (P_investment + Q_investment) = 7200 := by
  sorry

end NUMINAMATH_GPT_Q_share_of_profit_l2284_228453


namespace NUMINAMATH_GPT_king_then_ten_prob_l2284_228465

def num_kings : ℕ := 4
def num_tens : ℕ := 4
def deck_size : ℕ := 52
def first_card_draw_prob := (num_kings : ℚ) / (deck_size : ℚ)
def second_card_draw_prob := (num_tens : ℚ) / (deck_size - 1 : ℚ)

theorem king_then_ten_prob : 
  first_card_draw_prob * second_card_draw_prob = 4 / 663 := by
  sorry

end NUMINAMATH_GPT_king_then_ten_prob_l2284_228465


namespace NUMINAMATH_GPT_no_7_edges_edges_greater_than_5_l2284_228440

-- Define the concept of a convex polyhedron in terms of its edges and faces.
structure ConvexPolyhedron where
  V : ℕ    -- Number of vertices
  E : ℕ    -- Number of edges
  F : ℕ    -- Number of faces
  Euler : V - E + F = 2   -- Euler's characteristic

-- Define properties of convex polyhedron

-- Part (a) statement: A convex polyhedron cannot have exactly 7 edges.
theorem no_7_edges (P : ConvexPolyhedron) : P.E ≠ 7 :=
sorry

-- Part (b) statement: A convex polyhedron can have any number of edges greater than 5 and different from 7.
theorem edges_greater_than_5 (n : ℕ) (h : n > 5) (h2 : n ≠ 7) : ∃ P : ConvexPolyhedron, P.E = n :=
sorry

end NUMINAMATH_GPT_no_7_edges_edges_greater_than_5_l2284_228440


namespace NUMINAMATH_GPT_sequence_formula_correct_l2284_228472

noncomputable def S (n : ℕ) : ℕ := 2^n - 3

def a (n : ℕ) : ℤ :=
  if n = 1 then -1
  else 2^(n-1)

theorem sequence_formula_correct (n : ℕ) :
  a n = (if n = 1 then -1 else 2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_correct_l2284_228472


namespace NUMINAMATH_GPT_other_root_of_quadratic_l2284_228417

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, x^2 + a * x - 2 = 0 → x = -1) → ∃ m, x = m ∧ m = 2 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l2284_228417


namespace NUMINAMATH_GPT_wendy_washing_loads_l2284_228408

theorem wendy_washing_loads (shirts sweaters machine_capacity : ℕ) (total_clothes := shirts + sweaters) 
  (loads := total_clothes / machine_capacity) 
  (remainder := total_clothes % machine_capacity) 
  (h_shirts : shirts = 39) 
  (h_sweaters : sweaters = 33) 
  (h_machine_capacity : machine_capacity = 8) : loads = 9 ∧ remainder = 0 := 
by 
  sorry

end NUMINAMATH_GPT_wendy_washing_loads_l2284_228408


namespace NUMINAMATH_GPT_regression_estimate_l2284_228433

theorem regression_estimate:
  ∀ (x : ℝ), (1.43 * x + 257 = 400) → x = 100 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_regression_estimate_l2284_228433


namespace NUMINAMATH_GPT_remainder_7623_div_11_l2284_228424

theorem remainder_7623_div_11 : 7623 % 11 = 0 := 
by sorry

end NUMINAMATH_GPT_remainder_7623_div_11_l2284_228424


namespace NUMINAMATH_GPT_least_subtr_from_12702_to_div_by_99_l2284_228449

theorem least_subtr_from_12702_to_div_by_99 : ∃ k : ℕ, 12702 - k = 99 * (12702 / 99) ∧ 0 ≤ k ∧ k < 99 :=
by
  sorry

end NUMINAMATH_GPT_least_subtr_from_12702_to_div_by_99_l2284_228449


namespace NUMINAMATH_GPT_keith_turnips_l2284_228413

theorem keith_turnips (a t k : ℕ) (h1 : a = 9) (h2 : t = 15) : k = t - a := by
  sorry

end NUMINAMATH_GPT_keith_turnips_l2284_228413


namespace NUMINAMATH_GPT_find_ordered_triple_l2284_228460

theorem find_ordered_triple (a b c : ℝ) (h1 : a > 2) (h2 : b > 2) (h3 : c > 2)
  (h4 : (a + 1)^2 / (b + c - 1) + (b + 3)^2 / (c + a - 3) + (c + 5)^2 / (a + b - 5) = 27) :
  (a, b, c) = (9, 7, 2) :=
by sorry

end NUMINAMATH_GPT_find_ordered_triple_l2284_228460


namespace NUMINAMATH_GPT_hexagon_side_lengths_l2284_228412

theorem hexagon_side_lengths (n : ℕ) (h1 : n ≥ 0) (h2 : n ≤ 6) (h3 : 10 * n + 8 * (6 - n) = 56) : n = 4 :=
sorry

end NUMINAMATH_GPT_hexagon_side_lengths_l2284_228412


namespace NUMINAMATH_GPT_cost_of_450_chocolates_l2284_228425

theorem cost_of_450_chocolates :
  ∀ (cost_per_box : ℝ) (candies_per_box total_candies : ℕ),
  cost_per_box = 7.50 →
  candies_per_box = 30 →
  total_candies = 450 →
  (total_candies / candies_per_box : ℝ) * cost_per_box = 112.50 :=
by
  intros cost_per_box candies_per_box total_candies h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_of_450_chocolates_l2284_228425


namespace NUMINAMATH_GPT_range_of_x_l2284_228493

noncomputable def a (x : ℝ) : ℝ := x
def b : ℝ := 2
def B : ℝ := 60

-- State the problem: Prove the range of x given the conditions
theorem range_of_x (x : ℝ) (A : ℝ) (C : ℝ) (h1 : a x = b / (Real.sin (B * Real.pi / 180)) * (Real.sin (A * Real.pi / 180)))
  (h2 : A + C = 180 - 60) (two_solutions : (60 < A ∧ A < 120)) :
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_x_l2284_228493


namespace NUMINAMATH_GPT_locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l2284_228496

noncomputable def locus_of_C (a x0 y0 ξ η : ℝ) : Prop :=
  (x0 - ξ) * η^2 - 2 * ξ * y0 * η + ξ^3 - 3 * x0 * ξ^2 - a^2 * ξ + 3 * a^2 * x0 = 0

noncomputable def special_case (a ξ η : ℝ) : Prop :=
  ξ = 0 ∨ ξ^2 + η^2 = a^2

theorem locus_of_C_general_case_eq_cubic (a x0 y0 ξ η : ℝ) (hs: locus_of_C a x0 y0 ξ η) : 
  locus_of_C a x0 y0 ξ η := 
  sorry

theorem locus_of_C_special_case_eq_y_axis_or_circle (a ξ η : ℝ) : 
  special_case a ξ η := 
  sorry

end NUMINAMATH_GPT_locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l2284_228496


namespace NUMINAMATH_GPT_multiples_6_8_not_both_l2284_228405

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end NUMINAMATH_GPT_multiples_6_8_not_both_l2284_228405


namespace NUMINAMATH_GPT_city_mpg_l2284_228400

-- Define the conditions
variables {T H C : ℝ}
axiom cond1 : H * T = 560
axiom cond2 : (H - 6) * T = 336

-- The formal proof goal
theorem city_mpg : C = 9 :=
by
  have h1 : H = 560 / T := by sorry
  have h2 : (560 / T - 6) * T = 336 := by sorry
  have h3 : C = H - 6 := by sorry
  have h4 :  C = 9 := by sorry
  exact h4

end NUMINAMATH_GPT_city_mpg_l2284_228400


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l2284_228483

theorem arithmetic_sequence_15th_term :
  ∀ (a d n : ℕ), a = 3 → d = 13 - a → n = 15 → 
  a + (n - 1) * d = 143 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l2284_228483


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2284_228492

-- Problem (1)
theorem problem1 : -36 * (5 / 4 - 5 / 6 - 11 / 12) = 18 := by
  sorry

-- Problem (2)
theorem problem2 : (-2) ^ 2 - 3 * (-1) ^ 3 + 0 * (-2) ^ 3 = 7 := by
  sorry

-- Problem (3)
theorem problem3 (x : ℚ) (y : ℚ) (h1 : x = -2) (h2 : y = 1 / 2) : 
    (3 / 2) * x^2 * y + x * y^2 = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2284_228492


namespace NUMINAMATH_GPT_odd_function_f_neg_9_l2284_228473

noncomputable def f (x : ℝ) : ℝ := 
if x > 0 then x^(1/2) 
else -((-x)^(1/2))

theorem odd_function_f_neg_9 : f (-9) = -3 := by
  sorry

end NUMINAMATH_GPT_odd_function_f_neg_9_l2284_228473


namespace NUMINAMATH_GPT_no_stew_left_l2284_228464

theorem no_stew_left (company : Type) (stew : ℝ)
    (one_third_stayed : ℝ)
    (two_thirds_went : ℝ)
    (camp_consumption : ℝ)
    (range_consumption_per_portion : ℝ)
    (range_portion_multiplier : ℝ)
    (total_stew : ℝ) : 
    one_third_stayed = 1 / 3 →
    two_thirds_went = 2 / 3 →
    camp_consumption = 1 / 4 →
    range_portion_multiplier = 1.5 →
    total_stew = camp_consumption + (range_portion_multiplier * (two_thirds_went * (camp_consumption / one_third_stayed))) →
    total_stew = 1 →
    stew = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- here would be the proof steps
  sorry

end NUMINAMATH_GPT_no_stew_left_l2284_228464


namespace NUMINAMATH_GPT_avg_age_of_team_is_23_l2284_228456

-- Conditions
def captain_age := 24
def wicket_keeper_age := captain_age + 7

def remaining_players_avg_age (team_avg_age : ℝ) := team_avg_age - 1
def total_team_age (team_avg_age : ℝ) := 11 * team_avg_age
def total_remaining_players_age (team_avg_age : ℝ) := 9 * remaining_players_avg_age team_avg_age

-- Proof statement
theorem avg_age_of_team_is_23 (team_avg_age : ℝ) :
  total_team_age team_avg_age = captain_age + wicket_keeper_age + total_remaining_players_age team_avg_age → 
  team_avg_age = 23 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_of_team_is_23_l2284_228456


namespace NUMINAMATH_GPT_average_rounds_rounded_eq_4_l2284_228498

def rounds_distribution : List (Nat × Nat) := [(1, 4), (2, 3), (4, 4), (5, 2), (6, 6)]

def total_rounds : Nat := rounds_distribution.foldl (λ acc (rounds, golfers) => acc + rounds * golfers) 0

def total_golfers : Nat := rounds_distribution.foldl (λ acc (_, golfers) => acc + golfers) 0

def average_rounds : Float := total_rounds.toFloat / total_golfers.toFloat

theorem average_rounds_rounded_eq_4 : Float.round average_rounds = 4 := by
  sorry

end NUMINAMATH_GPT_average_rounds_rounded_eq_4_l2284_228498


namespace NUMINAMATH_GPT_daily_sale_correct_l2284_228471

-- Define the original and additional amounts in kilograms
def original_rice := 4 * 1000 -- 4 tons converted to kilograms
def additional_rice := 4000 -- kilograms
def total_rice := original_rice + additional_rice -- total amount of rice in kilograms
def days := 4 -- days to sell all the rice

-- Statement to prove: The amount to be sold each day
def daily_sale_amount := 2000 -- kilograms per day

theorem daily_sale_correct : total_rice / days = daily_sale_amount :=
by 
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_daily_sale_correct_l2284_228471


namespace NUMINAMATH_GPT_LineDoesNotIntersectParabola_sum_r_s_l2284_228486

noncomputable def r : ℝ := -0.6
noncomputable def s : ℝ := 40.6
def Q : ℝ × ℝ := (10, -6)
def line_through_Q_with_slope (m : ℝ) (p : ℝ × ℝ) : ℝ := m * p.1 - 10 * m - 6
def parabola (x : ℝ) : ℝ := 2 * x^2

theorem LineDoesNotIntersectParabola (m : ℝ) :
  r < m ∧ m < s ↔ (m^2 - 4 * 2 * (10 * m + 6) < 0) :=
by sorry

theorem sum_r_s : r + s = 40 :=
by sorry

end NUMINAMATH_GPT_LineDoesNotIntersectParabola_sum_r_s_l2284_228486


namespace NUMINAMATH_GPT_centipede_shoes_and_socks_l2284_228415

-- Define number of legs
def num_legs : ℕ := 10

-- Define the total number of items
def total_items : ℕ := 2 * num_legs

-- Define the total permutations without constraints
def total_permutations : ℕ := Nat.factorial total_items

-- Define the probability constraint for each leg
def single_leg_probability : ℚ := 1 / 2

-- Define the combined probability constraint for all legs
def all_legs_probability : ℚ := single_leg_probability ^ num_legs

-- Define the number of valid permutations (the answer to prove)
def valid_permutations : ℚ := total_permutations / all_legs_probability

theorem centipede_shoes_and_socks : valid_permutations = (Nat.factorial 20 : ℚ) / 2^10 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_centipede_shoes_and_socks_l2284_228415


namespace NUMINAMATH_GPT_unique_students_total_l2284_228454

variables (euclid_students raman_students pythagoras_students overlap_3 : ℕ)

def total_students (E R P O : ℕ) : ℕ := E + R + P - O

theorem unique_students_total (hE : euclid_students = 12) 
                              (hR : raman_students = 10) 
                              (hP : pythagoras_students = 15) 
                              (hO : overlap_3 = 3) : 
    total_students euclid_students raman_students pythagoras_students overlap_3 = 34 :=
by
    sorry

end NUMINAMATH_GPT_unique_students_total_l2284_228454


namespace NUMINAMATH_GPT_persons_in_office_l2284_228476

theorem persons_in_office
  (P : ℕ)
  (h1 : (P - (1/7 : ℚ)*P) = (6/7 : ℚ)*P)
  (h2 : (16.66666666666667/100 : ℚ) = 1/6) :
  P = 35 :=
sorry

end NUMINAMATH_GPT_persons_in_office_l2284_228476


namespace NUMINAMATH_GPT_last_digit_3_pow_1991_plus_1991_pow_3_l2284_228457

theorem last_digit_3_pow_1991_plus_1991_pow_3 :
  (3 ^ 1991 + 1991 ^ 3) % 10 = 8 :=
  sorry

end NUMINAMATH_GPT_last_digit_3_pow_1991_plus_1991_pow_3_l2284_228457


namespace NUMINAMATH_GPT_binomial_coefficient_fourth_term_l2284_228411

theorem binomial_coefficient_fourth_term (n k : ℕ) (hn : n = 5) (hk : k = 3) : Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_fourth_term_l2284_228411


namespace NUMINAMATH_GPT_woman_finishes_work_in_225_days_l2284_228452

theorem woman_finishes_work_in_225_days
  (M W : ℝ)
  (h1 : (10 * M + 15 * W) * 6 = 1)
  (h2 : M * 100 = 1) :
  1 / W = 225 :=
by
  sorry

end NUMINAMATH_GPT_woman_finishes_work_in_225_days_l2284_228452


namespace NUMINAMATH_GPT_zoo_pandas_l2284_228485

-- Defining the conditions
variable (total_couples : ℕ)
variable (pregnant_couples : ℕ)
variable (baby_pandas : ℕ)
variable (total_pandas : ℕ)

-- Given conditions
def paired_mates : Prop := ∃ c : ℕ, c = total_couples

def pregnant_condition : Prop := pregnant_couples = (total_couples * 25) / 100

def babies_condition : Prop := baby_pandas = 2

def total_condition : Prop := total_pandas = total_couples * 2 + baby_pandas

-- The theorem to be proven
theorem zoo_pandas (h1 : paired_mates total_couples)
                   (h2 : pregnant_condition total_couples pregnant_couples)
                   (h3 : babies_condition baby_pandas)
                   (h4 : pregnant_couples = 2) :
                   total_condition total_couples baby_pandas total_pandas :=
by sorry

end NUMINAMATH_GPT_zoo_pandas_l2284_228485


namespace NUMINAMATH_GPT_probability_at_least_one_deciphers_l2284_228443

theorem probability_at_least_one_deciphers (P_A P_B : ℚ) (hA : P_A = 1/2) (hB : P_B = 1/3) :
    P_A + P_B - P_A * P_B = 2/3 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_deciphers_l2284_228443


namespace NUMINAMATH_GPT_chord_line_equation_l2284_228458

theorem chord_line_equation (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), y1^2 = -8 * x1 ∧ y2^2 = -8 * x2 ∧ (x1 + x2) / 2 = -1 ∧ (y1 + y2) / 2 = 1 ∧ y - 1 = -4 * (x + 1)) →
  4 * x + y + 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_chord_line_equation_l2284_228458


namespace NUMINAMATH_GPT_number_of_methods_l2284_228437

def doctors : ℕ := 6
def days : ℕ := 3

theorem number_of_methods : (days^doctors) = 729 := 
by sorry

end NUMINAMATH_GPT_number_of_methods_l2284_228437


namespace NUMINAMATH_GPT_simplify_polynomial_expression_l2284_228441

noncomputable def polynomial_expression (x : ℝ) := 
  (3 * x^3 + x^2 - 5 * x + 9) * (x + 2) - (x + 2) * (2 * x^3 - 4 * x + 8) + (x^2 - 6 * x + 13) * (x + 2) * (x - 3)

theorem simplify_polynomial_expression (x : ℝ) :
  polynomial_expression x = 2 * x^4 + x^3 + 9 * x^2 + 23 * x + 2 :=
sorry

end NUMINAMATH_GPT_simplify_polynomial_expression_l2284_228441


namespace NUMINAMATH_GPT_board_cut_ratio_l2284_228495

theorem board_cut_ratio (L S : ℝ) (h1 : S + L = 20) (h2 : S = L + 4) (h3 : S = 8.0) : S / L = 1 := by
  sorry

end NUMINAMATH_GPT_board_cut_ratio_l2284_228495


namespace NUMINAMATH_GPT_wholesale_price_l2284_228446

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end NUMINAMATH_GPT_wholesale_price_l2284_228446


namespace NUMINAMATH_GPT_product_of_third_side_l2284_228410

/-- Two sides of a right triangle have lengths 5 and 7. The product of the possible lengths of 
the third side is exactly √1776. -/
theorem product_of_third_side :
  let a := 5
  let b := 7
  (Real.sqrt (a^2 + b^2) * Real.sqrt (b^2 - a^2)) = Real.sqrt 1776 := 
by 
  let a := 5
  let b := 7
  sorry

end NUMINAMATH_GPT_product_of_third_side_l2284_228410


namespace NUMINAMATH_GPT_amount_paid_is_correct_l2284_228401

-- Conditions given in the problem
def jimmy_shorts_count : ℕ := 3
def jimmy_short_price : ℝ := 15.0
def irene_shirts_count : ℕ := 5
def irene_shirt_price : ℝ := 17.0
def discount_rate : ℝ := 0.10

-- Define the total cost for jimmy
def jimmy_total_cost : ℝ := jimmy_shorts_count * jimmy_short_price

-- Define the total cost for irene
def irene_total_cost : ℝ := irene_shirts_count * irene_shirt_price

-- Define the total cost before discount
def total_cost_before_discount : ℝ := jimmy_total_cost + irene_total_cost

-- Define the discount amount
def discount_amount : ℝ := total_cost_before_discount * discount_rate

-- Define the total amount to pay
def total_amount_to_pay : ℝ := total_cost_before_discount - discount_amount

-- The proposition we need to prove
theorem amount_paid_is_correct : total_amount_to_pay = 117 := by
  sorry

end NUMINAMATH_GPT_amount_paid_is_correct_l2284_228401


namespace NUMINAMATH_GPT_general_formula_l2284_228480

-- Define the sequence term a_n
def sequence_term (n : ℕ) : ℚ :=
  if h : n = 0 then 1
  else (2 * n - 1 : ℚ) / (n * n)

-- State the theorem for the general formula of the nth term
theorem general_formula (n : ℕ) (hn : n ≠ 0) : 
  sequence_term n = (2 * n - 1 : ℚ) / (n * n) :=
by sorry

end NUMINAMATH_GPT_general_formula_l2284_228480


namespace NUMINAMATH_GPT_exists_negative_root_of_P_l2284_228444

def P(x : ℝ) : ℝ := x^7 - 2 * x^6 - 7 * x^4 - x^2 + 10

theorem exists_negative_root_of_P : ∃ x : ℝ, x < 0 ∧ P x = 0 :=
sorry

end NUMINAMATH_GPT_exists_negative_root_of_P_l2284_228444


namespace NUMINAMATH_GPT_wrapping_paper_fraction_used_l2284_228426

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_fraction_used_l2284_228426


namespace NUMINAMATH_GPT_macy_miles_left_to_run_l2284_228455

-- Define the given conditions
def goal : ℕ := 24
def miles_per_day : ℕ := 3
def days : ℕ := 6

-- Define the statement to be proven
theorem macy_miles_left_to_run :
  goal - (miles_per_day * days) = 6 :=
by
  sorry

end NUMINAMATH_GPT_macy_miles_left_to_run_l2284_228455


namespace NUMINAMATH_GPT_find_number_l2284_228450

theorem find_number (n : ℕ) (h₁ : ∀ x : ℕ, 21 + 7 * x = n ↔ 3 + x = 47):
  n = 329 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_find_number_l2284_228450


namespace NUMINAMATH_GPT_pudding_cups_initial_l2284_228438

theorem pudding_cups_initial (P : ℕ) (students : ℕ) (extra_cups : ℕ) 
  (h1 : students = 218) (h2 : extra_cups = 121) (h3 : P + extra_cups = students) : P = 97 := 
by
  sorry

end NUMINAMATH_GPT_pudding_cups_initial_l2284_228438


namespace NUMINAMATH_GPT_total_amount_l2284_228434

noncomputable def A : ℝ := 360.00000000000006
noncomputable def B : ℝ := (3/2) * A
noncomputable def C : ℝ := 4 * B

theorem total_amount (A B C : ℝ)
  (hA : A = 360.00000000000006)
  (hA_B : A = (2/3) * B)
  (hB_C : B = (1/4) * C) :
  A + B + C = 3060.0000000000007 := by
  sorry

end NUMINAMATH_GPT_total_amount_l2284_228434


namespace NUMINAMATH_GPT_fixed_point_inequality_l2284_228461

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * a^((x + 1) / 2) - 4

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-1) = -1 :=
sorry

theorem inequality (a : ℝ) (x : ℝ) (h : a > 1) :
  f a (x - 3 / 4) ≥ 3 / (a^(x^2 / 2)) - 4 :=
sorry

end NUMINAMATH_GPT_fixed_point_inequality_l2284_228461


namespace NUMINAMATH_GPT_add_num_denom_fraction_l2284_228423

theorem add_num_denom_fraction (n : ℚ) : (2 + n) / (7 + n) = 3 / 5 ↔ n = 11 / 2 := 
by
  sorry

end NUMINAMATH_GPT_add_num_denom_fraction_l2284_228423


namespace NUMINAMATH_GPT_racing_championship_guarantee_l2284_228467

/-- 
In a racing championship consisting of five races, the points awarded are as follows: 
6 points for first place, 4 points for second place, and 2 points for third place, with no ties possible. 
What is the smallest number of points a racer must accumulate in these five races to be guaranteed of having more points than any other racer? 
-/
theorem racing_championship_guarantee :
  ∀ (points_1st : ℕ) (points_2nd : ℕ) (points_3rd : ℕ) (races : ℕ),
  points_1st = 6 → points_2nd = 4 → points_3rd = 2 → 
  races = 5 →
  (∃ min_points : ℕ, min_points = 26 ∧ 
    ∀ (possible_points : ℕ), possible_points ≠ min_points → 
    (possible_points < min_points)) :=
by
  sorry

end NUMINAMATH_GPT_racing_championship_guarantee_l2284_228467


namespace NUMINAMATH_GPT_bobby_candy_total_l2284_228482

-- Definitions for the conditions
def initial_candy : Nat := 20
def first_candy_eaten : Nat := 34
def second_candy_eaten : Nat := 18

-- Theorem to prove the total pieces of candy Bobby ate
theorem bobby_candy_total : first_candy_eaten + second_candy_eaten = 52 := by
  sorry

end NUMINAMATH_GPT_bobby_candy_total_l2284_228482


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2284_228432

-- Given conditions: x = 1/3 and y = -1/2
def x : ℚ := 1 / 3
def y : ℚ := -1 / 2

-- Problem statement: 
-- Prove that (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2
theorem simplify_and_evaluate :
  (2 * x + 3 * y)^2 - (2 * x + y) * (2 * x - y) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2284_228432


namespace NUMINAMATH_GPT_area_of_rectangle_l2284_228406

theorem area_of_rectangle (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 90) : w * l = 379.6875 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l2284_228406


namespace NUMINAMATH_GPT_min_marks_required_l2284_228416

-- Definitions and conditions
def grid_size := 7
def strip_size := 4

-- Question and answer as a proof statement
theorem min_marks_required (n : ℕ) (h : grid_size = 2 * n - 1) : 
  (∃ marks : ℕ, 
    (∀ row col : ℕ, 
      row < grid_size → col < grid_size → 
      (∃ i j : ℕ, 
        i < strip_size → j < strip_size → 
        (marks ≥ 12)))) :=
sorry

end NUMINAMATH_GPT_min_marks_required_l2284_228416


namespace NUMINAMATH_GPT_largest_non_formable_amount_l2284_228448

-- Definitions and conditions from the problem
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def cannot_be_formed (n a b : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ a * x + b * y

-- The statement to prove
theorem largest_non_formable_amount :
  is_coprime 8 15 ∧ cannot_be_formed 97 8 15 :=
by
  sorry

end NUMINAMATH_GPT_largest_non_formable_amount_l2284_228448


namespace NUMINAMATH_GPT_factorization_of_x_squared_minus_nine_l2284_228462

theorem factorization_of_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_x_squared_minus_nine_l2284_228462


namespace NUMINAMATH_GPT_melinda_probability_correct_l2284_228489

def probability_two_digit_between_20_and_30 : ℚ :=
  11 / 36

theorem melinda_probability_correct :
  probability_two_digit_between_20_and_30 = 11 / 36 :=
by
  sorry

end NUMINAMATH_GPT_melinda_probability_correct_l2284_228489


namespace NUMINAMATH_GPT_min_value_f_l2284_228421

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_f_l2284_228421


namespace NUMINAMATH_GPT_problem_solution_l2284_228497

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f x + 2 * Real.cos x ^ 2

theorem problem_solution :
  (∀ x, (∃ ω > 0, ∃ φ, |φ| < Real.pi / 2 ∧ Real.sin (ω * x - φ) = 0 ∧ 2 * ω = Real.pi)) →
  (∀ x, f x = Real.sin (2 * x - Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), (g x ≤ 2 ∧ g x ≥ 1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2284_228497


namespace NUMINAMATH_GPT_problem_3_div_27_l2284_228414

theorem problem_3_div_27 (a b : ℕ) (h : 2^a = 8^(b + 1)) : 3^a / 27^b = 27 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_3_div_27_l2284_228414


namespace NUMINAMATH_GPT_find_g_seven_l2284_228477

noncomputable def g : ℝ → ℝ :=
  sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x + g y
axiom g_six : g 6 = 7

theorem find_g_seven : g 7 = 49 / 6 :=
by
  -- Proof omitted here
  sorry

end NUMINAMATH_GPT_find_g_seven_l2284_228477


namespace NUMINAMATH_GPT_theater_revenue_l2284_228409

theorem theater_revenue
  (total_seats : ℕ)
  (adult_price : ℕ)
  (child_price : ℕ)
  (child_tickets_sold : ℕ)
  (total_sold_out : total_seats = 80)
  (child_tickets_sold_cond : child_tickets_sold = 63)
  (adult_ticket_price_cond : adult_price = 12)
  (child_ticket_price_cond : child_price = 5)
  : total_seats * adult_price + child_tickets_sold * child_price = 519 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_theater_revenue_l2284_228409


namespace NUMINAMATH_GPT_average_age_of_guardians_and_fourth_graders_l2284_228402

theorem average_age_of_guardians_and_fourth_graders (num_fourth_graders num_guardians : ℕ)
  (avg_age_fourth_graders avg_age_guardians : ℕ)
  (h1 : num_fourth_graders = 40)
  (h2 : avg_age_fourth_graders = 10)
  (h3 : num_guardians = 60)
  (h4 : avg_age_guardians = 35)
  : (num_fourth_graders * avg_age_fourth_graders + num_guardians * avg_age_guardians) / (num_fourth_graders + num_guardians) = 25 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_guardians_and_fourth_graders_l2284_228402


namespace NUMINAMATH_GPT_find_c_l2284_228422

-- Defining the given condition
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 + c

theorem find_c : (∃ c : ℝ, ∀ x : ℝ, parabola x c = 2 * x^2 + 1) :=
by 
  sorry

end NUMINAMATH_GPT_find_c_l2284_228422


namespace NUMINAMATH_GPT_positive_difference_two_numbers_l2284_228429

theorem positive_difference_two_numbers (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 2 * y - 3 * x = 5) : abs (y - x) = 8 := 
sorry

end NUMINAMATH_GPT_positive_difference_two_numbers_l2284_228429


namespace NUMINAMATH_GPT_proof_subset_l2284_228499

def set_A := {x : ℝ | x ≥ 0}

theorem proof_subset (B : Set ℝ) (h : set_A ∪ B = B) : set_A ⊆ B := 
by
  sorry

end NUMINAMATH_GPT_proof_subset_l2284_228499


namespace NUMINAMATH_GPT_max_students_per_class_l2284_228428

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end NUMINAMATH_GPT_max_students_per_class_l2284_228428
