import Mathlib

namespace NUMINAMATH_GPT_mark_jump_rope_hours_l1848_184880

theorem mark_jump_rope_hours 
    (record : ℕ := 54000)
    (jump_per_second : ℕ := 3)
    (seconds_per_hour : ℕ := 3600)
    (total_jumps_to_break_record : ℕ := 54001)
    (jumps_per_hour : ℕ := jump_per_second * seconds_per_hour) 
    (hours_needed : ℕ := total_jumps_to_break_record / jumps_per_hour) 
    (round_up : ℕ := if total_jumps_to_break_record % jumps_per_hour = 0 then hours_needed else hours_needed + 1) :
    round_up = 5 :=
sorry

end NUMINAMATH_GPT_mark_jump_rope_hours_l1848_184880


namespace NUMINAMATH_GPT_remainder_x_squared_mod_25_l1848_184882

theorem remainder_x_squared_mod_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 4 [ZMOD 25] :=
sorry

end NUMINAMATH_GPT_remainder_x_squared_mod_25_l1848_184882


namespace NUMINAMATH_GPT_union_with_complement_l1848_184843

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end NUMINAMATH_GPT_union_with_complement_l1848_184843


namespace NUMINAMATH_GPT_greatest_possible_large_chips_l1848_184899

theorem greatest_possible_large_chips : 
  ∃ s l p: ℕ, s + l = 60 ∧ s = l + 2 * p ∧ Prime p ∧ l = 28 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_large_chips_l1848_184899


namespace NUMINAMATH_GPT_has_exactly_two_solutions_iff_l1848_184809

theorem has_exactly_two_solutions_iff (a : ℝ) :
  (∃! x : ℝ, x^2 + 2 * x + 2 * (|x + 1|) = a) ↔ a > -1 :=
sorry

end NUMINAMATH_GPT_has_exactly_two_solutions_iff_l1848_184809


namespace NUMINAMATH_GPT_possible_measures_for_angle_A_l1848_184869

-- Definition of angles A and B, and their relationship
def is_supplementary_angles (A B : ℕ) : Prop := A + B = 180

def is_multiple_of (A B : ℕ) : Prop := ∃ k : ℕ, k ≥ 1 ∧ A = k * B

-- Prove there are 17 possible measures for angle A.
theorem possible_measures_for_angle_A : 
  (∀ (A B : ℕ), (A > 0) ∧ (B > 0) ∧ is_multiple_of A B ∧ is_supplementary_angles A B → 
  A = B * 17) := 
sorry

end NUMINAMATH_GPT_possible_measures_for_angle_A_l1848_184869


namespace NUMINAMATH_GPT_remainder_of_2_pow_2018_plus_1_mod_2018_l1848_184889

theorem remainder_of_2_pow_2018_plus_1_mod_2018 : (2 ^ 2018 + 1) % 2018 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_2_pow_2018_plus_1_mod_2018_l1848_184889


namespace NUMINAMATH_GPT_train_length_l1848_184857

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_sec : ℝ := 3
noncomputable def speed_m_s := speed_km_hr * 1000 / 3600
noncomputable def length_of_train := speed_m_s * time_sec

theorem train_length :
  length_of_train = 50.01 := by
  sorry

end NUMINAMATH_GPT_train_length_l1848_184857


namespace NUMINAMATH_GPT_trays_from_first_table_is_23_l1848_184873

-- Definitions of conditions
def trays_per_trip : ℕ := 7
def trips_made : ℕ := 4
def trays_from_second_table : ℕ := 5

-- Total trays carried
def total_trays_carried : ℕ := trays_per_trip * trips_made

-- Number of trays picked from first table
def trays_from_first_table : ℕ :=
  total_trays_carried - trays_from_second_table

-- Theorem stating that the number of trays picked up from the first table is 23
theorem trays_from_first_table_is_23 : trays_from_first_table = 23 := by
  sorry

end NUMINAMATH_GPT_trays_from_first_table_is_23_l1848_184873


namespace NUMINAMATH_GPT_complement_union_l1848_184813

open Set

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3}
noncomputable def C_UA : Set ℕ := U \ A

-- Statement to prove
theorem complement_union (U A B C_UA : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 5})
  (hB : B = {2, 3}) 
  (hCUA : C_UA = U \ A) : 
  (C_UA ∪ B) = {2, 3, 4} := 
sorry

end NUMINAMATH_GPT_complement_union_l1848_184813


namespace NUMINAMATH_GPT_composite_number_l1848_184896

theorem composite_number (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (10 ^ n + 1) * (10 ^ (n + 1) - 1) / 9 :=
by sorry

end NUMINAMATH_GPT_composite_number_l1848_184896


namespace NUMINAMATH_GPT_no_positive_integer_k_for_rational_solutions_l1848_184831

theorem no_positive_integer_k_for_rational_solutions :
  ∀ k : ℕ, k > 0 → ¬ ∃ m : ℤ, 12 * (27 - k ^ 2) = m ^ 2 := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_k_for_rational_solutions_l1848_184831


namespace NUMINAMATH_GPT_solve_nine_sections_bamboo_problem_l1848_184862

-- Define the bamboo stick problem
noncomputable def nine_sections_bamboo_problem : Prop :=
∃ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) ∧ -- Arithmetic sequence
  (a 1 + a 2 + a 3 + a 4 = 3) ∧ -- Top 4 sections' total volume
  (a 7 + a 8 + a 9 = 4) ∧ -- Bottom 3 sections' total volume
  (a 5 = 67 / 66) -- Volume of the 5th section

theorem solve_nine_sections_bamboo_problem : nine_sections_bamboo_problem :=
sorry

end NUMINAMATH_GPT_solve_nine_sections_bamboo_problem_l1848_184862


namespace NUMINAMATH_GPT_average_greater_than_median_by_22_l1848_184874

/-- Define the weights of the siblings -/
def hammie_weight : ℕ := 120
def triplet1_weight : ℕ := 4
def triplet2_weight : ℕ := 4
def triplet3_weight : ℕ := 7
def brother_weight : ℕ := 10

/-- Define the list of weights -/
def weights : List ℕ := [hammie_weight, triplet1_weight, triplet2_weight, triplet3_weight, brother_weight]

/-- Define the median and average weight -/
def median_weight : ℕ := 7
def average_weight : ℕ := 29

theorem average_greater_than_median_by_22 : average_weight - median_weight = 22 := by
  sorry

end NUMINAMATH_GPT_average_greater_than_median_by_22_l1848_184874


namespace NUMINAMATH_GPT_find_number_l1848_184867

theorem find_number (x y : ℝ) (h1 : x = y + 0.25 * y) (h2 : x = 110) : y = 88 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1848_184867


namespace NUMINAMATH_GPT_number_of_rows_l1848_184804

-- Definitions of conditions
def tomatoes : ℕ := 3 * 5
def cucumbers : ℕ := 5 * 4
def potatoes : ℕ := 30
def additional_vegetables : ℕ := 85
def spaces_per_row : ℕ := 15

-- Total number of vegetables already planted
def planted_vegetables : ℕ := tomatoes + cucumbers + potatoes

-- Total capacity of the garden
def garden_capacity : ℕ := planted_vegetables + additional_vegetables

-- Number of rows in the garden
def rows_in_garden : ℕ := garden_capacity / spaces_per_row

theorem number_of_rows : rows_in_garden = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_rows_l1848_184804


namespace NUMINAMATH_GPT_rosy_has_14_fish_l1848_184823

-- Define the number of Lilly's fish
def lilly_fish : ℕ := 10

-- Define the total number of fish
def total_fish : ℕ := 24

-- Define the number of Rosy's fish, which we need to prove equals 14
def rosy_fish : ℕ := total_fish - lilly_fish

-- Prove that Rosy has 14 fish
theorem rosy_has_14_fish : rosy_fish = 14 := by
  sorry

end NUMINAMATH_GPT_rosy_has_14_fish_l1848_184823


namespace NUMINAMATH_GPT_selling_price_eq_l1848_184827

noncomputable def cost_price : ℝ := 1300
noncomputable def selling_price_loss : ℝ := 1280
noncomputable def selling_price_profit_25_percent : ℝ := 1625

theorem selling_price_eq (cp sp_loss sp_profit sp: ℝ) 
  (h1 : sp_profit = 1.25 * cp)
  (h2 : sp_loss = cp - 20)
  (h3 : sp = cp + 20) :
  sp = 1320 :=
sorry

end NUMINAMATH_GPT_selling_price_eq_l1848_184827


namespace NUMINAMATH_GPT_polynomial_divisibility_l1848_184859

theorem polynomial_divisibility (a b : ℤ) :
  (∀ x : ℤ, x^2 - 1 ∣ x^5 - 3 * x^4 + a * x^3 + b * x^2 - 5 * x - 5) ↔ (a = 4 ∧ b = 8) :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1848_184859


namespace NUMINAMATH_GPT_intersected_squares_and_circles_l1848_184883

def is_intersected_by_line (p q : ℕ) : Prop :=
  p = q

def total_intersections : ℕ := 504 * 2

theorem intersected_squares_and_circles :
  total_intersections = 1008 :=
by
  sorry

end NUMINAMATH_GPT_intersected_squares_and_circles_l1848_184883


namespace NUMINAMATH_GPT_excircle_side_formula_l1848_184833

theorem excircle_side_formula 
  (a b c r_a r_b r_c : ℝ)
  (h1 : r_c = Real.sqrt (r_a * r_b)) :
  c = (a^2 + b^2) / (a + b) :=
sorry

end NUMINAMATH_GPT_excircle_side_formula_l1848_184833


namespace NUMINAMATH_GPT_quadrilateral_rectangle_ratio_l1848_184856

theorem quadrilateral_rectangle_ratio
  (s x y : ℝ)
  (h_area : (s + 2 * x) ^ 2 = 4 * s ^ 2)
  (h_y : 2 * y = s) :
  y / x = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_rectangle_ratio_l1848_184856


namespace NUMINAMATH_GPT_find_a_b_value_l1848_184838

-- Define the variables
variables {a b : ℤ}

-- Define the conditions for the monomials to be like terms
def exponents_match_x (a : ℤ) : Prop := a + 2 = 1
def exponents_match_y (b : ℤ) : Prop := b + 1 = 3

-- Main statement
theorem find_a_b_value (ha : exponents_match_x a) (hb : exponents_match_y b) : a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_value_l1848_184838


namespace NUMINAMATH_GPT_alice_net_amount_spent_l1848_184872

noncomputable def net_amount_spent : ℝ :=
  let price_per_pint := 4
  let sunday_pints := 4
  let sunday_cost := sunday_pints * price_per_pint

  let monday_discount := 0.1
  let monday_pints := 3 * sunday_pints
  let monday_price_per_pint := price_per_pint * (1 - monday_discount)
  let monday_cost := monday_pints * monday_price_per_pint

  let tuesday_discount := 0.2
  let tuesday_pints := monday_pints / 3
  let tuesday_price_per_pint := price_per_pint * (1 - tuesday_discount)
  let tuesday_cost := tuesday_pints * tuesday_price_per_pint

  let wednesday_returned_pints := tuesday_pints / 2
  let wednesday_refund := wednesday_returned_pints * tuesday_price_per_pint

  sunday_cost + monday_cost + tuesday_cost - wednesday_refund

theorem alice_net_amount_spent : net_amount_spent = 65.60 := by
  sorry

end NUMINAMATH_GPT_alice_net_amount_spent_l1848_184872


namespace NUMINAMATH_GPT_solve_phi_l1848_184820

-- Define the problem
noncomputable def f (phi x : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + phi)
noncomputable def f' (phi x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + phi)
noncomputable def g (phi x : ℝ) : ℝ := f phi x + f' phi x

-- Define the main theorem
theorem solve_phi (phi : ℝ) (h : -Real.pi < phi ∧ phi < 0) 
  (even_g : ∀ x, g phi x = g phi (-x)) : phi = -Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_solve_phi_l1848_184820


namespace NUMINAMATH_GPT_fibonacci_rabbits_l1848_184886

theorem fibonacci_rabbits : 
  ∀ (F : ℕ → ℕ), 
    (F 0 = 1) ∧ 
    (F 1 = 1) ∧ 
    (∀ n, F (n + 2) = F n + F (n + 1)) → 
    F 12 = 233 := 
by 
  intro F h; sorry

end NUMINAMATH_GPT_fibonacci_rabbits_l1848_184886


namespace NUMINAMATH_GPT_april_roses_l1848_184890

theorem april_roses (R : ℕ) (h1 : 7 * (R - 4) = 35) : R = 9 :=
sorry

end NUMINAMATH_GPT_april_roses_l1848_184890


namespace NUMINAMATH_GPT_paula_remaining_money_l1848_184887

-- Definitions based on the conditions
def initialMoney : ℕ := 1000
def shirtCost : ℕ := 45
def pantsCost : ℕ := 85
def jacketCost : ℕ := 120
def shoeCost : ℕ := 95
def jeansOriginalPrice : ℕ := 140
def jeansDiscount : ℕ := 30 / 100  -- 30%

-- Using definitions to compute the spending and remaining money
def totalShirtCost : ℕ := 6 * shirtCost
def totalPantsCost : ℕ := 2 * pantsCost
def totalShoeCost : ℕ := 3 * shoeCost
def jeansDiscountValue : ℕ := jeansDiscount * jeansOriginalPrice
def jeansDiscountedPrice : ℕ := jeansOriginalPrice - jeansDiscountValue
def totalSpent : ℕ := totalShirtCost + totalPantsCost + jacketCost + totalShoeCost
def remainingMoney : ℕ := initialMoney - totalSpent - jeansDiscountedPrice

-- Proof problem statement
theorem paula_remaining_money : remainingMoney = 57 := by
  sorry

end NUMINAMATH_GPT_paula_remaining_money_l1848_184887


namespace NUMINAMATH_GPT_f_2019_equals_neg2_l1848_184806

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 4) = f x)
variable (h_defined : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2)

theorem f_2019_equals_neg2 : f 2019 = -2 :=
by 
  sorry

end NUMINAMATH_GPT_f_2019_equals_neg2_l1848_184806


namespace NUMINAMATH_GPT_intersection_of_lines_l1848_184892

theorem intersection_of_lines :
  ∃ (x y : ℝ), (8 * x + 5 * y = 40) ∧ (3 * x - 10 * y = 15) ∧ (x = 5) ∧ (y = 0) := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1848_184892


namespace NUMINAMATH_GPT_vertex_y_coordinate_l1848_184868

theorem vertex_y_coordinate (x : ℝ) : 
    let a := -6
    let b := 24
    let c := -7
    ∃ k : ℝ, k = 17 ∧ ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x - 2)^2 + k) := 
by 
  sorry

end NUMINAMATH_GPT_vertex_y_coordinate_l1848_184868


namespace NUMINAMATH_GPT_distinguishable_arrangements_l1848_184832

-- Define the conditions: number of tiles of each color
def num_brown_tiles := 2
def num_purple_tile := 1
def num_green_tiles := 3
def num_yellow_tiles := 4

-- Total number of tiles
def total_tiles := num_brown_tiles + num_purple_tile + num_green_tiles + num_yellow_tiles

-- Factorials (using Lean's built-in factorial function)
def brown_factorial := Nat.factorial num_brown_tiles
def purple_factorial := Nat.factorial num_purple_tile
def green_factorial := Nat.factorial num_green_tiles
def yellow_factorial := Nat.factorial num_yellow_tiles
def total_factorial := Nat.factorial total_tiles

-- The result of the permutation calculation
def number_of_arrangements := total_factorial / (brown_factorial * purple_factorial * green_factorial * yellow_factorial)

-- The theorem stating the expected correct answer
theorem distinguishable_arrangements : number_of_arrangements = 12600 := 
by
    simp [number_of_arrangements, total_tiles, brown_factorial, purple_factorial, green_factorial, yellow_factorial, total_factorial]
    sorry

end NUMINAMATH_GPT_distinguishable_arrangements_l1848_184832


namespace NUMINAMATH_GPT_positive_numbers_with_cube_root_lt_10_l1848_184881

def cube_root_lt_10 (n : ℕ) : Prop :=
  (↑n : ℝ)^(1 / 3 : ℝ) < 10

theorem positive_numbers_with_cube_root_lt_10 : 
  ∃ (count : ℕ), (count = 999) ∧ ∀ n : ℕ, (1 ≤ n ∧ n ≤ 999) → cube_root_lt_10 n :=
by
  sorry

end NUMINAMATH_GPT_positive_numbers_with_cube_root_lt_10_l1848_184881


namespace NUMINAMATH_GPT_jogger_ahead_of_train_l1848_184800

noncomputable def distance_ahead_of_train (v_j v_t : ℕ) (L_t t : ℕ) : ℕ :=
  let relative_speed_kmh := v_t - v_j
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  let total_distance := relative_speed_ms * t
  total_distance - L_t

theorem jogger_ahead_of_train :
  distance_ahead_of_train 10 46 120 46 = 340 :=
by
  sorry

end NUMINAMATH_GPT_jogger_ahead_of_train_l1848_184800


namespace NUMINAMATH_GPT_problem_statement_l1848_184834

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a₁ d : ℤ, ∀ n : ℕ, a n = a₁ + n * d

theorem problem_statement :
  ∃ a : ℕ → ℤ, is_arithmetic_sequence a ∧
  a 2 = 7 ∧
  a 4 + a 6 = 26 ∧
  (∀ n : ℕ, a (n + 1) = 2 * n + 1) ∧
  ∃ S : ℕ → ℤ, (S n = n^2 + 2 * n) ∧
  ∃ b : ℕ → ℚ, (∀ n : ℕ, b n = 1 / (a n ^ 2 - 1)) ∧
  ∃ T : ℕ → ℚ, (T n = (n / 4) * (1 / (n + 1))) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1848_184834


namespace NUMINAMATH_GPT_initial_mixture_amount_l1848_184891

/-- A solution initially contains an unknown amount of a mixture consisting of 15% sodium chloride
(NaCl), 30% potassium chloride (KCl), 35% sugar, and 20% water. To this mixture, 50 grams of sodium chloride
and 80 grams of potassium chloride are added. If the new salt content of the solution (NaCl and KCl combined)
is 47.5%, how many grams of the mixture were present initially?

Given:
  * The initial mixture consists of 15% NaCl and 30% KCl.
  * 50 grams of NaCl and 80 grams of KCl are added.
  * The new mixture has 47.5% NaCl and KCl combined.
  
Prove that the initial amount of the mixture was 2730 grams. -/
theorem initial_mixture_amount
    (x : ℝ)
    (h_initial_mixture : 0.15 * x + 50 + 0.30 * x + 80 = 0.475 * (x + 130)) :
    x = 2730 := by
  sorry

end NUMINAMATH_GPT_initial_mixture_amount_l1848_184891


namespace NUMINAMATH_GPT_division_theorem_l1848_184815

noncomputable def p (z : ℝ) : ℝ := 4 * z ^ 3 - 8 * z ^ 2 + 9 * z - 7
noncomputable def d (z : ℝ) : ℝ := 4 * z + 2
noncomputable def q (z : ℝ) : ℝ := z ^ 2 - 2.5 * z + 3.5
def r : ℝ := -14

theorem division_theorem (z : ℝ) : p z = d z * q z + r := 
by
  sorry

end NUMINAMATH_GPT_division_theorem_l1848_184815


namespace NUMINAMATH_GPT_min_value_expr_l1848_184849

theorem min_value_expr (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + ((y / x) - 1)^2 + ((z / y) - 1)^2 + ((5 / z) - 1)^2 = 9 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l1848_184849


namespace NUMINAMATH_GPT_eval_expression_l1848_184801

theorem eval_expression : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1848_184801


namespace NUMINAMATH_GPT_unique_ordered_triple_l1848_184812

theorem unique_ordered_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 ∧ x = 2 ∧ y = 2 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_unique_ordered_triple_l1848_184812


namespace NUMINAMATH_GPT_part1_part2_l1848_184876

-- Definitions for the sides and the target equations
def triangleSides (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.sin (C / 2) ^ 2 + c * Real.sin (B / 2) ^ 2 = a / 2

-- The first part of the problem
theorem part1 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  b + c = 2 * a :=
  sorry

-- The second part of the problem
theorem part2 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  A ≤ π / 3 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1848_184876


namespace NUMINAMATH_GPT_max_area_of_triangle_l1848_184854

-- Defining the side lengths and constraints
def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main statement of the area maximization problem
theorem max_area_of_triangle (x : ℝ) (h1 : 2 < x) (h2 : x < 6) :
  triangle_sides 6 x (2 * x) →
  ∃ (S : ℝ), S = 12 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_l1848_184854


namespace NUMINAMATH_GPT_tan_pi_minus_alpha_l1848_184894

theorem tan_pi_minus_alpha (α : ℝ) (h : Real.tan (Real.pi - α) = -2) : 
  (1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_minus_alpha_l1848_184894


namespace NUMINAMATH_GPT_asha_borrowed_from_mother_l1848_184895

def total_money (M : ℕ) : ℕ := 20 + 40 + 70 + 100 + M

def remaining_money_after_spending_3_4 (total : ℕ) : ℕ := total * 1 / 4

theorem asha_borrowed_from_mother : ∃ M : ℕ, total_money M = 260 ∧ remaining_money_after_spending_3_4 (total_money M) = 65 :=
by
  sorry

end NUMINAMATH_GPT_asha_borrowed_from_mother_l1848_184895


namespace NUMINAMATH_GPT_science_fair_unique_students_l1848_184877

/-!
# Problem statement:
At Euclid Middle School, there are three clubs participating in the Science Fair: the Robotics Club, the Astronomy Club, and the Chemistry Club.
There are 15 students in the Robotics Club, 10 students in the Astronomy Club, and 12 students in the Chemistry Club.
Assuming 2 students are members of all three clubs, prove that the total number of unique students participating in the Science Fair is 33.
-/

theorem science_fair_unique_students (R A C : ℕ) (all_three : ℕ) (hR : R = 15) (hA : A = 10) (hC : C = 12) (h_all_three : all_three = 2) :
    R + A + C - 2 * all_three = 33 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_science_fair_unique_students_l1848_184877


namespace NUMINAMATH_GPT_greatest_possible_remainder_l1848_184853

theorem greatest_possible_remainder (x : ℕ) : ∃ r : ℕ, r < 12 ∧ r ≠ 0 ∧ x % 12 = r ∧ r = 11 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_possible_remainder_l1848_184853


namespace NUMINAMATH_GPT_inverse_function_domain_l1848_184898

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem inverse_function_domain :
  ∃ (g : ℝ → ℝ), (∀ x, 0 ≤ x → f (g x) = x) ∧ (∀ y, 0 ≤ y → g (f y) = y) ∧ (∀ x, 0 ≤ x ↔ 0 ≤ g x) :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_domain_l1848_184898


namespace NUMINAMATH_GPT_igor_number_proof_l1848_184824

noncomputable def igor_number (init_lineup : List ℕ) (igor_num : ℕ) : Prop :=
  let after_first_command := [9, 11, 10, 6, 8, 7] -- Results after first command 
  let after_second_command := [9, 11, 10, 8] -- Results after second command
  let after_third_command := [11, 10, 8] -- Results after third command
  ∃ (idx : ℕ), init_lineup.get? idx = some igor_num ∧
    (∀ new_lineup, 
       (new_lineup = after_first_command ∨ new_lineup = after_second_command ∨ new_lineup = after_third_command) →
       igor_num ∉ new_lineup) ∧ 
    after_third_command.length = 3

theorem igor_number_proof : igor_number [9, 1, 11, 2, 10, 3, 6, 4, 8, 5, 7] 5 :=
  sorry 

end NUMINAMATH_GPT_igor_number_proof_l1848_184824


namespace NUMINAMATH_GPT_gcd_polynomial_l1848_184871

theorem gcd_polynomial (b : ℤ) (h : b % 2 = 0 ∧ 1171 ∣ b) : 
  Int.gcd (3 * b^2 + 17 * b + 47) (b + 5) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_polynomial_l1848_184871


namespace NUMINAMATH_GPT_complex_multiplication_l1848_184845

theorem complex_multiplication :
  ∀ (i : ℂ), i * i = -1 → i * (1 + i) = -1 + i :=
by
  intros i hi
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1848_184845


namespace NUMINAMATH_GPT_no_solution_for_t_and_s_l1848_184807

theorem no_solution_for_t_and_s (m : ℝ) :
  (¬∃ t s : ℝ, (1 + 7 * t = -3 + 2 * s) ∧ (3 - 5 * t = 4 + m * s)) ↔ m = -10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_t_and_s_l1848_184807


namespace NUMINAMATH_GPT_algebraic_expression_value_l1848_184808

theorem algebraic_expression_value 
  (x y : ℝ) 
  (h : 2 * x + y = 1) : 
  (y + 1) ^ 2 - (y ^ 2 - 4 * x + 4) = -1 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1848_184808


namespace NUMINAMATH_GPT_domain_and_range_of_g_l1848_184852

noncomputable def f : ℝ → ℝ := sorry-- Given: a function f with domain [0,2] and range [0,1]
noncomputable def g (x : ℝ) := 1 - f (x / 2 + 1)

theorem domain_and_range_of_g :
  let dom_g := { x | -2 ≤ x ∧ x ≤ 2 }
  let range_g := { y | 0 ≤ y ∧ y ≤ 1 }
  ∀ (x : ℝ), (x ∈ dom_g → (g x) ∈ range_g) := 
sorry

end NUMINAMATH_GPT_domain_and_range_of_g_l1848_184852


namespace NUMINAMATH_GPT_num_dinosaur_dolls_l1848_184805

-- Define the number of dinosaur dolls
def dinosaur_dolls : Nat := 3

-- Define the theorem to prove the number of dinosaur dolls
theorem num_dinosaur_dolls : dinosaur_dolls = 3 := by
  -- Add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_num_dinosaur_dolls_l1848_184805


namespace NUMINAMATH_GPT_smallest_k_exists_l1848_184842

theorem smallest_k_exists : ∃ (k : ℕ) (n : ℕ), k = 53 ∧ k^2 + 49 = 180 * n :=
sorry

end NUMINAMATH_GPT_smallest_k_exists_l1848_184842


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1848_184850

theorem quadratic_inequality_solution (b c : ℝ) 
    (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + b * x + c < 0) :
    b + c = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1848_184850


namespace NUMINAMATH_GPT_sprinter_time_no_wind_l1848_184810

theorem sprinter_time_no_wind :
  ∀ (x y : ℝ), (90 / (x + y) = 10) → (70 / (x - y) = 10) → x = 8 * y → 100 / x = 12.5 :=
by
  intros x y h1 h2 h3
  sorry

end NUMINAMATH_GPT_sprinter_time_no_wind_l1848_184810


namespace NUMINAMATH_GPT_log_equation_solution_l1848_184811

theorem log_equation_solution (x : ℝ) (hx_pos : 0 < x) : 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_log_equation_solution_l1848_184811


namespace NUMINAMATH_GPT_sum_of_interior_angles_eq_1440_l1848_184888

theorem sum_of_interior_angles_eq_1440 (h : ∀ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ)) : 
    (∃ (n : ℕ), (360 : ℝ) / 36 = (n : ℝ) ∧ (n - 2) * 180 = 1440) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_eq_1440_l1848_184888


namespace NUMINAMATH_GPT_find_c_value_l1848_184858

def finds_c (c : ℝ) : Prop :=
  6 * (-(c / 6)) + 9 * (-(c / 9)) + c = 0 ∧ (-(c / 6) + -(c / 9) = 30)

theorem find_c_value : ∃ c : ℝ, finds_c c ∧ c = -108 :=
by
  use -108
  sorry

end NUMINAMATH_GPT_find_c_value_l1848_184858


namespace NUMINAMATH_GPT_Kyle_rose_cost_l1848_184865

/-- Given the number of roses Kyle picked last year, the number of roses he picked this year, 
and the cost of one rose, prove that the total cost he has to spend to buy the remaining roses 
is correct. -/
theorem Kyle_rose_cost (last_year_roses this_year_roses total_roses_needed cost_per_rose : ℕ)
    (h_last_year_roses : last_year_roses = 12) 
    (h_this_year_roses : this_year_roses = last_year_roses / 2) 
    (h_total_roses_needed : total_roses_needed = 2 * last_year_roses) 
    (h_cost_per_rose : cost_per_rose = 3) : 
    (total_roses_needed - this_year_roses) * cost_per_rose = 54 := 
by
sorry

end NUMINAMATH_GPT_Kyle_rose_cost_l1848_184865


namespace NUMINAMATH_GPT_correct_final_positions_l1848_184841

noncomputable def shapes_after_rotation (initial_positions : (String × String) × (String × String) × (String × String)) : (String × String) × (String × String) × (String × String) :=
  match initial_positions with
  | (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) =>
    (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left"))
  | _ => initial_positions

theorem correct_final_positions :
  shapes_after_rotation (("Triangle", "Top"), ("Circle", "Lower Left"), ("Pentagon", "Lower Right")) = (("Triangle", "Lower Right"), ("Circle", "Top"), ("Pentagon", "Lower Left")) :=
by
  unfold shapes_after_rotation
  rfl

end NUMINAMATH_GPT_correct_final_positions_l1848_184841


namespace NUMINAMATH_GPT_problem1_problem2_l1848_184837

-- Define the triangle and the condition a + 2a * cos B = c
variable {A B C : ℝ} (a b c : ℝ)
variable (cos_B : ℝ) -- cosine of angle B

-- Condition: a + 2a * cos B = c
variable (h1 : a + 2 * a * cos_B = c)

-- (I) Prove B = 2A
theorem problem1 (h1 : a + 2 * a * cos_B = c) : B = 2 * A :=
sorry

-- Define the acute triangle condition
variable (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

-- Given: c = 2
variable (h2 : c = 2)

-- (II) Determine the range for a if the triangle is acute and c = 2
theorem problem2 (h1 : a + 2 * a * cos_B = 2) (Acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : 1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1848_184837


namespace NUMINAMATH_GPT_simplify_radical_expr_l1848_184840

-- Define the variables and expressions
variables {x : ℝ} (hx : 0 ≤ x) 

-- State the problem
theorem simplify_radical_expr (hx : 0 ≤ x) :
  (Real.sqrt (100 * x)) * (Real.sqrt (3 * x)) * (Real.sqrt (18 * x)) = 30 * x * Real.sqrt (6 * x) :=
sorry

end NUMINAMATH_GPT_simplify_radical_expr_l1848_184840


namespace NUMINAMATH_GPT_find_a5_l1848_184878

noncomputable def arithmetic_sequence (n : ℕ) (a d : ℤ) : ℤ :=
a + n * d

theorem find_a5 (a d : ℤ) (a_2_a_4_sum : arithmetic_sequence 1 a d + arithmetic_sequence 3 a d = 16)
  (a1 : arithmetic_sequence 0 a d = 1) :
  arithmetic_sequence 4 a d = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_a5_l1848_184878


namespace NUMINAMATH_GPT_inequality_generalization_l1848_184875

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) :
  x + n^n / x^n ≥ n + 1 :=
sorry

end NUMINAMATH_GPT_inequality_generalization_l1848_184875


namespace NUMINAMATH_GPT_problem1_problem2_l1848_184836

def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

theorem problem1 (x : ℝ) : f x (-1) ≤ 2 ↔ -1 / 2 ≤ x ∧ x ≤ 1 / 2 :=
by sorry

theorem problem2 (a : ℝ) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 1, f x a ≤ |2 * x + 1|) → (0 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1848_184836


namespace NUMINAMATH_GPT_skipping_times_eq_l1848_184822

theorem skipping_times_eq (x : ℝ) (h : x > 0) :
  180 / x = 240 / (x + 5) :=
sorry

end NUMINAMATH_GPT_skipping_times_eq_l1848_184822


namespace NUMINAMATH_GPT_polar_distance_to_axis_l1848_184893

theorem polar_distance_to_axis (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = Real.pi / 6) : 
  ρ * Real.sin θ = 1 := 
by
  rw [hρ, hθ]
  -- The remaining proof steps would go here
  sorry

end NUMINAMATH_GPT_polar_distance_to_axis_l1848_184893


namespace NUMINAMATH_GPT_divisor_is_four_l1848_184821

theorem divisor_is_four (n d k l : ℤ) (hn : n % d = 3) (h2n : (2 * n) % d = 2) (hd : d > 3) : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_four_l1848_184821


namespace NUMINAMATH_GPT_circle_area_from_circumference_l1848_184863

theorem circle_area_from_circumference (C : ℝ) (hC : C = 48 * Real.pi) : 
  ∃ m : ℝ, (∀ r : ℝ, C = 2 * Real.pi * r → (Real.pi * r^2 = m * Real.pi)) ∧ m = 576 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_from_circumference_l1848_184863


namespace NUMINAMATH_GPT_sixty_fifth_term_is_sixteen_l1848_184839

def apply_rule (n : ℕ) : ℕ :=
  if n <= 12 then
    7 * n
  else if n % 2 = 0 then
    n - 7
  else
    n / 3

def sequence_term (a_0 : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate apply_rule n a_0

theorem sixty_fifth_term_is_sixteen : sequence_term 65 64 = 16 := by
  sorry

end NUMINAMATH_GPT_sixty_fifth_term_is_sixteen_l1848_184839


namespace NUMINAMATH_GPT_question_b_l1848_184818

theorem question_b (a b c : ℝ) (h : c ≠ 0) (h_eq : a / c = b / c) : a = b := 
by
  sorry

end NUMINAMATH_GPT_question_b_l1848_184818


namespace NUMINAMATH_GPT_min_dot_product_PA_PB_l1848_184860

noncomputable def point_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def point_on_ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem min_dot_product_PA_PB (A B P : ℝ × ℝ)
  (hA : point_on_circle A.1 A.2)
  (hB : point_on_circle B.1 B.2)
  (hAB : A ≠ B ∧ (B.1 = -A.1) ∧ (B.2 = -A.2))
  (hP : point_on_ellipse P.1 P.2) :
  ∃ PA PB : ℝ × ℝ, 
    PA = (P.1 - A.1, P.2 - A.2) ∧ PB = (P.1 - B.1, P.2 - B.2) ∧
    (PA.1 * PB.1 + PA.2 * PB.2) = 2 :=
by sorry

end NUMINAMATH_GPT_min_dot_product_PA_PB_l1848_184860


namespace NUMINAMATH_GPT_exponential_comparison_l1848_184848

theorem exponential_comparison (x y a b : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (hb : a < b) (hb' : b < 1) : 
  a^x < b^y :=
sorry

end NUMINAMATH_GPT_exponential_comparison_l1848_184848


namespace NUMINAMATH_GPT_original_difference_of_weights_l1848_184870

variable (F S T : ℝ)

theorem original_difference_of_weights :
  (F + S + T = 75) →
  (F - 2 = 0.7 * (S + 2)) →
  (S + 1 = 0.8 * (T + 1)) →
  T - F = 10.16 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_original_difference_of_weights_l1848_184870


namespace NUMINAMATH_GPT_yogurt_combinations_l1848_184828

-- Definitions: Given conditions from the problem
def num_flavors : ℕ := 5
def num_toppings : ℕ := 7

-- Function to calculate binomial coefficient
def nCr (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: The problem translated into Lean
theorem yogurt_combinations : 
  (num_flavors * nCr num_toppings 2) = 105 := by
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l1848_184828


namespace NUMINAMATH_GPT_cricket_players_count_l1848_184846

-- Define the conditions
def total_players_present : ℕ := 50
def hockey_players : ℕ := 17
def football_players : ℕ := 11
def softball_players : ℕ := 10

-- Define the result to prove
def cricket_players : ℕ := total_players_present - (hockey_players + football_players + softball_players)

-- The theorem stating the equivalence of cricket_players and the correct answer
theorem cricket_players_count : cricket_players = 12 := by
  -- A placeholder for the proof
  sorry

end NUMINAMATH_GPT_cricket_players_count_l1848_184846


namespace NUMINAMATH_GPT_theater_ticket_sales_l1848_184802

theorem theater_ticket_sales (x y : ℕ) (h1 : x + y = 175) (h2 : 6 * x + 2 * y = 750) : y = 75 :=
sorry

end NUMINAMATH_GPT_theater_ticket_sales_l1848_184802


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_any_triangle_perimeter_l1848_184855

def is_valid_triangle (a b c : ℕ) : Prop := 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem smallest_whole_number_larger_than_any_triangle_perimeter : 
  ∀ (s : ℕ), 16 < s ∧ s < 30 → is_valid_triangle 7 23 s → 
    60 = (Nat.succ (7 + 23 + s - 1)) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_any_triangle_perimeter_l1848_184855


namespace NUMINAMATH_GPT_Frank_read_books_l1848_184844

noncomputable def books_read (total_days : ℕ) (days_per_book : ℕ) : ℕ :=
total_days / days_per_book

theorem Frank_read_books : books_read 492 12 = 41 := by
  sorry

end NUMINAMATH_GPT_Frank_read_books_l1848_184844


namespace NUMINAMATH_GPT_num_trucks_l1848_184829

variables (T : ℕ) (num_cars : ℕ := 13) (total_wheels : ℕ := 100) (wheels_per_vehicle : ℕ := 4)

theorem num_trucks :
  (num_cars * wheels_per_vehicle + T * wheels_per_vehicle = total_wheels) -> T = 12 :=
by
  intro h
  -- skipping the proof implementation
  sorry

end NUMINAMATH_GPT_num_trucks_l1848_184829


namespace NUMINAMATH_GPT_complement_P_subset_PQ_intersection_PQ_eq_Q_l1848_184884

open Set

variable {R : Type*} [OrderedCommRing R]

def P (x : R) : Prop := -2 ≤ x ∧ x ≤ 10
def Q (m x : R) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem complement_P : (compl (setOf P)) = {x | x < -2} ∪ {x | x > 10} :=
by {
  sorry
}

theorem subset_PQ (m : R) : (∀ x, P x → Q m x) ↔ m ≥ 9 :=
by {
  sorry
}

theorem intersection_PQ_eq_Q (m : R) : (∀ x, Q m x → P x) ↔ m ≤ 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_P_subset_PQ_intersection_PQ_eq_Q_l1848_184884


namespace NUMINAMATH_GPT_inequality_solution_set_l1848_184879

theorem inequality_solution_set
  (a b c m n : ℝ) (h : a ≠ 0) 
  (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ m < x ∧ x < n)
  (h2 : 0 < m)
  (h3 : ∀ x : ℝ, cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) :
  (cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1848_184879


namespace NUMINAMATH_GPT_right_angle_triangle_l1848_184803

theorem right_angle_triangle (a b c : ℝ) (h : (a + b) ^ 2 - c ^ 2 = 2 * a * b) : a ^ 2 + b ^ 2 = c ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_right_angle_triangle_l1848_184803


namespace NUMINAMATH_GPT_solve_for_x_l1848_184897

-- Assumptions and conditions of the problem
def a : ℚ := 4 / 7
def b : ℚ := 1 / 5
def c : ℚ := 12
def d : ℚ := 105

-- The statement of the problem
theorem solve_for_x (x : ℚ) (h : a * b * x = c) : x = d :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1848_184897


namespace NUMINAMATH_GPT_circle_x_intersect_l1848_184819

theorem circle_x_intersect (x y : ℝ) : 
  (x, y) = (0, 0) ∨ (x, y) = (10, 0) → (x = 10) :=
by
  -- conditions:
  -- The endpoints of the diameter are (0,0) and (10,10)
  -- (proving that the second intersect on x-axis has x-coordinate 10)
  sorry

end NUMINAMATH_GPT_circle_x_intersect_l1848_184819


namespace NUMINAMATH_GPT_train_passes_jogger_in_time_l1848_184814

def jogger_speed_kmh : ℝ := 8
def train_speed_kmh : ℝ := 60
def initial_distance_m : ℝ := 360
def train_length_m : ℝ := 200

noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m
noncomputable def passing_time_s : ℝ := total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_time :
  passing_time_s = 38.75 := by
  sorry

end NUMINAMATH_GPT_train_passes_jogger_in_time_l1848_184814


namespace NUMINAMATH_GPT_quarterback_sacked_times_l1848_184885

theorem quarterback_sacked_times
    (total_throws : ℕ)
    (no_pass_percentage : ℚ)
    (half_sacked : ℚ)
    (no_passes : ℕ)
    (sacks : ℕ) :
    total_throws = 80 →
    no_pass_percentage = 0.30 →
    half_sacked = 0.50 →
    no_passes = total_throws * no_pass_percentage →
    sacks = no_passes / 2 →
    sacks = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_quarterback_sacked_times_l1848_184885


namespace NUMINAMATH_GPT_total_payment_correct_l1848_184866

def payment_X (payment_Y : ℝ) : ℝ := 1.2 * payment_Y
def payment_Y : ℝ := 254.55
def total_payment (payment_X payment_Y : ℝ) : ℝ := payment_X + payment_Y

theorem total_payment_correct :
  total_payment (payment_X payment_Y) payment_Y = 560.01 :=
by
  sorry

end NUMINAMATH_GPT_total_payment_correct_l1848_184866


namespace NUMINAMATH_GPT_original_price_l1848_184826

theorem original_price (P: ℝ) (h: 0.80 * 1.15 * P = 46) : P = 50 :=
by sorry

end NUMINAMATH_GPT_original_price_l1848_184826


namespace NUMINAMATH_GPT_complex_real_imag_eq_l1848_184861

theorem complex_real_imag_eq (b : ℝ) (h : (2 + b) / 5 = (2 * b - 1) / 5) : b = 3 :=
  sorry

end NUMINAMATH_GPT_complex_real_imag_eq_l1848_184861


namespace NUMINAMATH_GPT_cannot_be_square_of_binomial_B_l1848_184847

theorem cannot_be_square_of_binomial_B (x y m n : ℝ) :
  (∃ (a b : ℝ), (3*x + 7*y) * (3*x - 7*y) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -0.2*x - 0.3) * ( -0.2*x + 0.3) = a^2 - b^2) ∧
  (∃ (a b : ℝ), ( -3*n - m*n) * ( 3*n - m*n) = a^2 - b^2) ∧
  ¬(∃ (a b : ℝ), ( 5*m - n) * ( n - 5*m) = a^2 - b^2) :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_square_of_binomial_B_l1848_184847


namespace NUMINAMATH_GPT_integer_count_n_l1848_184830

theorem integer_count_n (n : ℤ) (H1 : n % 3 = 0) (H2 : 3 * n ≥ 1) (H3 : 3 * n ≤ 1000) : 
  ∃ k : ℕ, k = 111 := by
  sorry

end NUMINAMATH_GPT_integer_count_n_l1848_184830


namespace NUMINAMATH_GPT_apples_per_pie_l1848_184825

-- Conditions
def initial_apples : ℕ := 50
def apples_per_teacher_per_child : ℕ := 3
def number_of_teachers : ℕ := 2
def number_of_children : ℕ := 2
def remaining_apples : ℕ := 24

-- Proof goal: the number of apples Jill uses per pie
theorem apples_per_pie : 
  initial_apples 
  - (apples_per_teacher_per_child * number_of_teachers * number_of_children)  - remaining_apples = 14 -> 14 / 2 = 7 := 
by
  sorry

end NUMINAMATH_GPT_apples_per_pie_l1848_184825


namespace NUMINAMATH_GPT_determine_range_a_l1848_184864

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, 1 ≤ x → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

theorem determine_range_a (a : ℝ) (h : ¬ prop_p a ∧ (prop_p a ∨ prop_q a)) : 
  a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8) :=
sorry

end NUMINAMATH_GPT_determine_range_a_l1848_184864


namespace NUMINAMATH_GPT_range_of_a_l1848_184816

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else 2 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x, 3 ≤ f x a) ∧ (0 < a) ∧ (a ≠ 1) → 1 < a ∧ a ≤ 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1848_184816


namespace NUMINAMATH_GPT_hendecagon_diagonals_l1848_184851

-- Define the number of sides n of the hendecagon
def n : ℕ := 11

-- Define the formula for calculating the number of diagonals in an n-sided polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that there are 44 diagonals in a hendecagon
theorem hendecagon_diagonals : diagonals n = 44 :=
by
  -- Proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_hendecagon_diagonals_l1848_184851


namespace NUMINAMATH_GPT_describe_cylinder_l1848_184835

noncomputable def cylinder_geometric_shape (c : ℝ) (r θ z : ℝ) : Prop :=
  r = c

theorem describe_cylinder (c : ℝ) (hc : 0 < c) :
  ∀ r θ z : ℝ, cylinder_geometric_shape c r θ z ↔ (r = c) :=
by
  sorry

end NUMINAMATH_GPT_describe_cylinder_l1848_184835


namespace NUMINAMATH_GPT_remainder_17_pow_63_mod_7_l1848_184817

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_17_pow_63_mod_7_l1848_184817
