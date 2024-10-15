import Mathlib

namespace NUMINAMATH_GPT_regular_polygon_sides_l979_97932

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_regular_polygon_sides_l979_97932


namespace NUMINAMATH_GPT_cards_relationship_l979_97957

-- Definitions from the conditions given in the problem
variables (x y : ℕ)

-- Theorem statement proving the relationship
theorem cards_relationship (h : x + y = 8 * x) : y = 7 * x :=
sorry

end NUMINAMATH_GPT_cards_relationship_l979_97957


namespace NUMINAMATH_GPT_blue_line_length_l979_97999

theorem blue_line_length (w b : ℝ) (h1 : w = 7.666666666666667) (h2 : w = b + 4.333333333333333) :
  b = 3.333333333333334 :=
by sorry

end NUMINAMATH_GPT_blue_line_length_l979_97999


namespace NUMINAMATH_GPT_max_cells_primitive_dinosaur_l979_97959

section Dinosaur

universe u

-- Define a dinosaur as a structure with at least 2007 cells
structure Dinosaur (α : Type u) :=
(cells : ℕ) (connected : α → α → Prop)
(h_cells : cells ≥ 2007)
(h_connected : ∀ (x y : α), connected x y → connected y x)

-- Define a primitive dinosaur where the cells cannot be partitioned into two or more dinosaurs
structure PrimitiveDinosaur (α : Type u) extends Dinosaur α :=
(h_partition : ∀ (x : α), ¬∃ (d1 d2 : Dinosaur α), (d1.cells + d2.cells = cells) ∧ 
  (d1 ≠ d2 ∧ d1.cells ≥ 2007 ∧ d2.cells ≥ 2007))

-- Prove that the maximum number of cells in a Primitive Dinosaur is 8025
theorem max_cells_primitive_dinosaur : ∀ (α : Type u), ∃ (d : PrimitiveDinosaur α), d.cells = 8025 :=
sorry

end Dinosaur

end NUMINAMATH_GPT_max_cells_primitive_dinosaur_l979_97959


namespace NUMINAMATH_GPT_hyperbola_equation_l979_97912

variable (a b c : ℝ)

def system_eq1 := (4 / (-3 - c)) = (- a / b)
def system_eq2 := ((c - 3) / 2) * (b / a) = 2
def system_eq3 := a ^ 2 + b ^ 2 = c ^ 2

theorem hyperbola_equation (h1 : system_eq1 a b c) (h2 : system_eq2 a b c) (h3 : system_eq3 a b c) :
  ∃ a b : ℝ, c = 5 ∧ b^2 = 20 ∧ a^2 = 5 ∧ (∀ x y : ℝ, (x ^ 2 / 5) - (y ^ 2 / 20) = 1) :=
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l979_97912


namespace NUMINAMATH_GPT_gcd_lcm_product_eq_l979_97986

-- Define the numbers
def a : ℕ := 10
def b : ℕ := 15

-- Define the GCD and LCM
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Proposition that needs to be proved
theorem gcd_lcm_product_eq : gcd_ab * lcm_ab = 150 :=
  by
    -- Proof would go here
    sorry

end NUMINAMATH_GPT_gcd_lcm_product_eq_l979_97986


namespace NUMINAMATH_GPT_exists_y_equals_7_l979_97947

theorem exists_y_equals_7 : ∃ (x y z t : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ y = 7 ∧ x + y + z + t = 10 :=
by {
  sorry -- This is where the actual proof would go.
}

end NUMINAMATH_GPT_exists_y_equals_7_l979_97947


namespace NUMINAMATH_GPT_tangent_line_equation_l979_97981

theorem tangent_line_equation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ∃ (m b : ℝ), y = m * x + b ∧ y = 4 * x - 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l979_97981


namespace NUMINAMATH_GPT_problem_statement_l979_97906

variable (x y : ℝ)
variable (h_cond1 : 1 / x + 1 / y = 4)
variable (h_cond2 : x * y - x - y = -7)

theorem problem_statement (h_cond1 : 1 / x + 1 / y = 4) (h_cond2 : x * y - x - y = -7) : 
  x^2 * y + x * y^2 = 196 / 9 := 
sorry

end NUMINAMATH_GPT_problem_statement_l979_97906


namespace NUMINAMATH_GPT_sin_neg_five_sixths_pi_l979_97943

theorem sin_neg_five_sixths_pi : Real.sin (- 5 / 6 * Real.pi) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_sin_neg_five_sixths_pi_l979_97943


namespace NUMINAMATH_GPT_geometric_sequence_sum_reciprocal_ratio_l979_97983

theorem geometric_sequence_sum_reciprocal_ratio
  (a : ℚ) (r : ℚ) (n : ℕ) (S S' : ℚ)
  (h1 : a = 1/4)
  (h2 : r = 2)
  (h3 : S = a * (1 - r^n) / (1 - r))
  (h4 : S' = (1/a) * (1 - (1/r)^n) / (1 - 1/r)) :
  S / S' = 32 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_reciprocal_ratio_l979_97983


namespace NUMINAMATH_GPT_quotient_when_divided_by_8_l979_97968

theorem quotient_when_divided_by_8
  (n : ℕ)
  (h1 : n = 12 * 7 + 5)
  : (n / 8) = 11 :=
by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_quotient_when_divided_by_8_l979_97968


namespace NUMINAMATH_GPT_collinear_points_solves_a_l979_97927

theorem collinear_points_solves_a : 
  ∀ (a : ℝ),
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  (8 - 3) / (5 - 1) = (a - 8) / (29 - 5) → a = 38 :=
by 
  intro a
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  intro h
  sorry

end NUMINAMATH_GPT_collinear_points_solves_a_l979_97927


namespace NUMINAMATH_GPT_a_1995_is_squared_l979_97952

variable (a : ℕ → ℕ)

-- Conditions on the sequence 
axiom seq_condition  {m n : ℕ} (h : m ≥ n) : 
  a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

axiom initial_value : a 1 = 1

-- Goal to prove
theorem a_1995_is_squared : a 1995 = 1995^2 :=
sorry

end NUMINAMATH_GPT_a_1995_is_squared_l979_97952


namespace NUMINAMATH_GPT_fraction_uncovered_l979_97988

def area_rug (length width : ℕ) : ℕ := length * width
def area_square (side : ℕ) : ℕ := side * side

theorem fraction_uncovered 
  (rug_length rug_width floor_area : ℕ)
  (h_rug_length : rug_length = 2)
  (h_rug_width : rug_width = 7)
  (h_floor_area : floor_area = 64)
  : (floor_area - area_rug rug_length rug_width) / floor_area = 25 / 32 := 
sorry

end NUMINAMATH_GPT_fraction_uncovered_l979_97988


namespace NUMINAMATH_GPT_playground_ball_cost_l979_97917

-- Define the given conditions
def cost_jump_rope : ℕ := 7
def cost_board_game : ℕ := 12
def saved_by_dalton : ℕ := 6
def given_by_uncle : ℕ := 13
def additional_needed : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_by_dalton + given_by_uncle

-- Total cost needed to buy all three items
def total_cost_needed : ℕ := total_money + additional_needed

-- Combined cost of the jump rope and the board game
def combined_cost : ℕ := cost_jump_rope + cost_board_game

-- Prove the cost of the playground ball
theorem playground_ball_cost : ℕ := total_cost_needed - combined_cost

-- Expected result
example : playground_ball_cost = 4 := by
  sorry

end NUMINAMATH_GPT_playground_ball_cost_l979_97917


namespace NUMINAMATH_GPT_num_terminating_decimals_l979_97904

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 518) :
  (∃ k, (1 ≤ k ∧ k ≤ 518) ∧ n = k * 21) ↔ n = 24 :=
sorry

end NUMINAMATH_GPT_num_terminating_decimals_l979_97904


namespace NUMINAMATH_GPT_simplify_fraction_l979_97985

theorem simplify_fraction (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c ≠ 0) :
  (a^2 + a * b - b^2 + a * c) / (b^2 + b * c - c^2 + b * a) = (a - b) / (b - c) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l979_97985


namespace NUMINAMATH_GPT_smallest_square_perimeter_l979_97971

theorem smallest_square_perimeter (P_largest : ℕ) (units_apart : ℕ) (num_squares : ℕ) (H1 : P_largest = 96) (H2 : units_apart = 1) (H3 : num_squares = 8) : 
  ∃ P_smallest : ℕ, P_smallest = 40 := by
  sorry

end NUMINAMATH_GPT_smallest_square_perimeter_l979_97971


namespace NUMINAMATH_GPT_sunscreen_cost_l979_97936

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end NUMINAMATH_GPT_sunscreen_cost_l979_97936


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l979_97989

-- Define the arithmetic sequence properties
variable (S : ℕ → ℕ) -- S represents the sum of the first n terms
variable (a : ℕ → ℕ) -- a represents the terms in the arithmetic sequence
variable (d : ℤ) -- common difference

-- Define the conditions
axiom S2_eq_6 : S 2 = 6
axiom a1_eq_4 : a 1 = 4

-- The problem: show that d = -2
theorem common_difference_arithmetic_sequence :
  (a 2 - a 1 = d) → d = -2 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l979_97989


namespace NUMINAMATH_GPT_solution_set_condition_l979_97974

theorem solution_set_condition {a : ℝ} : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ (x ≥ 3)) → a < 3 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_solution_set_condition_l979_97974


namespace NUMINAMATH_GPT_inequality_proof_l979_97982

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_mul : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l979_97982


namespace NUMINAMATH_GPT_jessica_purchase_cost_l979_97997

noncomputable def c_toy : Real := 10.22
noncomputable def c_cage : Real := 11.73
noncomputable def c_total : Real := c_toy + c_cage

theorem jessica_purchase_cost : c_total = 21.95 :=
by
  sorry

end NUMINAMATH_GPT_jessica_purchase_cost_l979_97997


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l979_97964

theorem problem_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = 14 * Real.sqrt 5 / 5 :=
by sorry

theorem problem_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 :=
by sorry

theorem problem_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 :=
by sorry

theorem problem_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3) ^ 2 = 2 * Real.sqrt 15 - 6 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_problem_4_l979_97964


namespace NUMINAMATH_GPT_geometric_series_ratio_l979_97918

theorem geometric_series_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q))
  (h2 : a 3 + 2 * a 6 = 0)
  (h3 : a 6 = a 3 * q^3)
  (h4 : q^3 = -1 / 2) :
  S 3 / S 6 = 2 := 
sorry

end NUMINAMATH_GPT_geometric_series_ratio_l979_97918


namespace NUMINAMATH_GPT_grocery_store_spending_l979_97901

/-- Lenny has $84 initially. He spent $24 on video games and has $39 left.
We need to prove that he spent $21 at the grocery store. --/
theorem grocery_store_spending (initial_amount spent_on_video_games amount_left after_games_left : ℕ) 
    (h1 : initial_amount = 84)
    (h2 : spent_on_video_games = 24)
    (h3 : amount_left = 39)
    (h4 : after_games_left = initial_amount - spent_on_video_games) 
    : after_games_left - amount_left = 21 := 
sorry

end NUMINAMATH_GPT_grocery_store_spending_l979_97901


namespace NUMINAMATH_GPT_number_of_values_l979_97961

/-- Given:
  - The mean of some values was 190.
  - One value 165 was wrongly copied as 130 for the computation of the mean.
  - The correct mean is 191.4.
  Prove: the total number of values is 25. --/
theorem number_of_values (n : ℕ) (h₁ : (190 : ℝ) = ((190 * n) - (165 - 130)) / n) (h₂ : (191.4 : ℝ) = ((190 * n + 35) / n)) : n = 25 :=
sorry

end NUMINAMATH_GPT_number_of_values_l979_97961


namespace NUMINAMATH_GPT_number_of_sections_l979_97923

noncomputable def initial_rope : ℕ := 50
noncomputable def rope_for_art := initial_rope / 5
noncomputable def remaining_rope_after_art := initial_rope - rope_for_art
noncomputable def rope_given_to_friend := remaining_rope_after_art / 2
noncomputable def remaining_rope := remaining_rope_after_art - rope_given_to_friend
noncomputable def section_size : ℕ := 2
noncomputable def sections := remaining_rope / section_size

theorem number_of_sections : sections = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sections_l979_97923


namespace NUMINAMATH_GPT_max_students_before_new_year_l979_97975

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end NUMINAMATH_GPT_max_students_before_new_year_l979_97975


namespace NUMINAMATH_GPT_number_of_street_trees_l979_97946

-- Definitions from conditions
def road_length : ℕ := 1500
def interval_distance : ℕ := 25

-- The statement to prove
theorem number_of_street_trees : (road_length / interval_distance) + 1 = 61 := 
by
  unfold road_length
  unfold interval_distance
  sorry

end NUMINAMATH_GPT_number_of_street_trees_l979_97946


namespace NUMINAMATH_GPT_oak_trees_cut_down_l979_97979

   def number_of_cuts (initial: ℕ) (remaining: ℕ) : ℕ :=
     initial - remaining

   theorem oak_trees_cut_down : number_of_cuts 9 7 = 2 :=
   by
     -- Based on the conditions, we start with 9 and after workers finished, there are 7 oak trees.
     -- We calculate the number of trees cut down:
     -- 9 - 7 = 2
     sorry
   
end NUMINAMATH_GPT_oak_trees_cut_down_l979_97979


namespace NUMINAMATH_GPT_apples_remaining_l979_97938

variable (initial_apples : ℕ)
variable (picked_day1 : ℕ)
variable (picked_day2 : ℕ)
variable (picked_day3 : ℕ)

-- Given conditions
def condition1 : initial_apples = 200 := sorry
def condition2 : picked_day1 = initial_apples / 5 := sorry
def condition3 : picked_day2 = 2 * picked_day1 := sorry
def condition4 : picked_day3 = picked_day1 + 20 := sorry

-- Prove the total number of apples remaining is 20
theorem apples_remaining (H1 : initial_apples = 200) 
  (H2 : picked_day1 = initial_apples / 5) 
  (H3 : picked_day2 = 2 * picked_day1)
  (H4 : picked_day3 = picked_day1 + 20) : 
  initial_apples - (picked_day1 + picked_day2 + picked_day3) = 20 := 
by
  sorry

end NUMINAMATH_GPT_apples_remaining_l979_97938


namespace NUMINAMATH_GPT_cylinder_section_volume_l979_97911

theorem cylinder_section_volume (a : ℝ) :
  let volume := (π * a^3 / 4)
  let section1_volume := volume * (1 / 3)
  let section2_volume := volume * (1 / 4)
  let enclosed_volume := (section1_volume - section2_volume) / 2
  enclosed_volume = π * a^3 / 24 := by
  sorry

end NUMINAMATH_GPT_cylinder_section_volume_l979_97911


namespace NUMINAMATH_GPT_baba_yaga_powder_problem_l979_97907

theorem baba_yaga_powder_problem (A B d : ℤ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end NUMINAMATH_GPT_baba_yaga_powder_problem_l979_97907


namespace NUMINAMATH_GPT_total_trees_after_planting_l979_97967

def current_trees : ℕ := 7
def trees_planted_today : ℕ := 5
def trees_planted_tomorrow : ℕ := 4

theorem total_trees_after_planting : 
  current_trees + trees_planted_today + trees_planted_tomorrow = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_trees_after_planting_l979_97967


namespace NUMINAMATH_GPT_cylinder_volume_eq_pi_over_4_l979_97969

theorem cylinder_volume_eq_pi_over_4
  (r : ℝ)
  (h₀ : r > 0)
  (h₁ : 2 * r = r * 2)
  (h₂ : 4 * π * r^2 = π) : 
  (π * r^2 * (2 * r) = π / 4) :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_eq_pi_over_4_l979_97969


namespace NUMINAMATH_GPT_inverse_of_square_l979_97953

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_square (h : A⁻¹ = ![
  ![3, -2],
  ![1, 1]
]) : 
  (A^2)⁻¹ = ![
  ![7, -8],
  ![4, -1]
] :=
sorry

end NUMINAMATH_GPT_inverse_of_square_l979_97953


namespace NUMINAMATH_GPT_jessica_saves_l979_97980

-- Define the costs based on the conditions given
def basic_cost : ℕ := 15
def movie_cost : ℕ := 12
def sports_cost : ℕ := movie_cost - 3
def bundle_cost : ℕ := 25

-- Define the total cost when the packages are purchased separately
def separate_cost : ℕ := basic_cost + movie_cost + sports_cost

-- Define the savings when opting for the bundle
def savings : ℕ := separate_cost - bundle_cost

-- The theorem that states the savings are 11 dollars
theorem jessica_saves : savings = 11 :=
by
  sorry

end NUMINAMATH_GPT_jessica_saves_l979_97980


namespace NUMINAMATH_GPT_find_purchase_price_minimum_number_of_speed_skating_shoes_l979_97920

/-
A certain school in Zhangjiakou City is preparing to purchase speed skating shoes and figure skating shoes to promote ice and snow activities on campus.

If they buy 30 pairs of speed skating shoes and 20 pairs of figure skating shoes, the total cost is $8500.
If they buy 40 pairs of speed skating shoes and 10 pairs of figure skating shoes, the total cost is $8000.
The school purchases a total of 50 pairs of both types of ice skates, and the total cost does not exceed $8900.
-/

def price_system (x y : ℝ) : Prop :=
  30 * x + 20 * y = 8500 ∧ 40 * x + 10 * y = 8000

def minimum_speed_skating_shoes (x y m : ℕ) : Prop :=
  150 * m + 200 * (50 - m) ≤ 8900

theorem find_purchase_price :
  ∃ x y : ℝ, price_system x y ∧ x = 150 ∧ y = 200 :=
by
  /- Proof goes here -/
  sorry

theorem minimum_number_of_speed_skating_shoes :
  ∃ m, minimum_speed_skating_shoes 150 200 m ∧ m = 22 :=
by
  /- Proof goes here -/
  sorry

end NUMINAMATH_GPT_find_purchase_price_minimum_number_of_speed_skating_shoes_l979_97920


namespace NUMINAMATH_GPT_periodic_odd_function_value_l979_97903

theorem periodic_odd_function_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x)
    (h_periodic : ∀ x : ℝ, f (x + 2) = f x) (h_value : f 0.5 = -1) : f 7.5 = 1 :=
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_periodic_odd_function_value_l979_97903


namespace NUMINAMATH_GPT_mod_x_squared_l979_97972

theorem mod_x_squared :
  (∃ x : ℤ, 5 * x ≡ 9 [ZMOD 26] ∧ 4 * x ≡ 15 [ZMOD 26]) →
  ∃ y : ℤ, y ≡ 10 [ZMOD 26] :=
by
  intro h
  rcases h with ⟨x, h₁, h₂⟩
  exists x^2
  sorry

end NUMINAMATH_GPT_mod_x_squared_l979_97972


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l979_97929

theorem sum_of_squares_of_roots (x1 x2 : ℝ) 
    (h1 : 2 * x1^2 + 3 * x1 - 5 = 0) 
    (h2 : 2 * x2^2 + 3 * x2 - 5 = 0)
    (h3 : x1 + x2 = -3 / 2)
    (h4 : x1 * x2 = -5 / 2) : 
    x1^2 + x2^2 = 29 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l979_97929


namespace NUMINAMATH_GPT_compare_P_Q_l979_97951

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 - x)^(2*n - 1)
noncomputable def Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n - 1)*x + (n - 1)*(2*n - 1)*x^2

theorem compare_P_Q :
  ∀ (n : ℕ) (x : ℝ), n > 0 →
  ((n = 1 → P n x = Q n x) ∧
   (n = 2 → ((x = 0 → P n x = Q n x) ∧ (x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x))) ∧
   (n ≥ 3 → ((x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x)))) :=
by
  intros
  sorry

end NUMINAMATH_GPT_compare_P_Q_l979_97951


namespace NUMINAMATH_GPT_new_rectangle_area_l979_97996

theorem new_rectangle_area (L W : ℝ) (h : L * W = 300) :
  let L_new := 2 * L
  let W_new := 3 * W
  L_new * W_new = 1800 :=
by
  let L_new := 2 * L
  let W_new := 3 * W
  sorry

end NUMINAMATH_GPT_new_rectangle_area_l979_97996


namespace NUMINAMATH_GPT_skate_time_correct_l979_97954

noncomputable def skate_time (path_length miles_length : ℝ) (skating_speed : ℝ) : ℝ :=
  let time_taken := (1.58 * Real.pi) / skating_speed
  time_taken

theorem skate_time_correct :
  skate_time 1 1 4 = 1.58 * Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_skate_time_correct_l979_97954


namespace NUMINAMATH_GPT_triangle_trig_identity_l979_97941

open Real

theorem triangle_trig_identity (A B C : ℝ) (h_triangle : A + B + C = 180) (h_A : A = 15) :
  sqrt 3 * sin A - cos (B + C) = sqrt 2 := by
  sorry

end NUMINAMATH_GPT_triangle_trig_identity_l979_97941


namespace NUMINAMATH_GPT_jerome_bought_last_month_l979_97910

-- Definitions representing the conditions in the problem
def total_toy_cars_now := 40
def original_toy_cars := 25
def bought_this_month (bought_last_month : ℕ) := 2 * bought_last_month

-- The main statement to prove
theorem jerome_bought_last_month : ∃ x : ℕ, original_toy_cars + x + bought_this_month x = total_toy_cars_now ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_jerome_bought_last_month_l979_97910


namespace NUMINAMATH_GPT_expansion_eq_l979_97926

variable (x y : ℝ) -- x and y are real numbers
def a := 5
def b := 3
def c := 15

theorem expansion_eq : (x + a) * (b * y + c) = 3 * x * y + 15 * x + 15 * y + 75 := by 
  sorry

end NUMINAMATH_GPT_expansion_eq_l979_97926


namespace NUMINAMATH_GPT_isabella_hair_length_l979_97934

theorem isabella_hair_length (final_length growth_length initial_length : ℕ) 
  (h1 : final_length = 24) 
  (h2 : growth_length = 6) 
  (h3 : final_length = initial_length + growth_length) : 
  initial_length = 18 :=
by
  sorry

end NUMINAMATH_GPT_isabella_hair_length_l979_97934


namespace NUMINAMATH_GPT_katelyn_sandwiches_difference_l979_97973

theorem katelyn_sandwiches_difference :
  ∃ (K : ℕ), K - 49 = 47 ∧ (49 + K + K / 4 = 169) := 
sorry

end NUMINAMATH_GPT_katelyn_sandwiches_difference_l979_97973


namespace NUMINAMATH_GPT_range_a_condition_l979_97900

theorem range_a_condition (a : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ a → x^2 ≤ 2 * x + 3) ↔ (1 / 2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_a_condition_l979_97900


namespace NUMINAMATH_GPT_number_of_real_roots_l979_97993

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 19) ^ x + (5 / 19) ^ x + (11 / 19) ^ x

noncomputable def g (x : ℝ) : ℝ := sqrt (x - 1)

theorem number_of_real_roots : ∃! x : ℝ, 1 ≤ x ∧ f x = g x :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_roots_l979_97993


namespace NUMINAMATH_GPT_part1_part2_l979_97937

noncomputable section
open Real

section
variables {x A a b c : ℝ}
variables {k : ℤ}

def f (x : ℝ) : ℝ := sin (2 * x - (π / 6)) + 2 * cos x ^ 2 - 1

theorem part1 (k : ℤ) : 
  ∀ x : ℝ, 
  k * π - (π / 3) ≤ x ∧ x ≤ k * π + (π / 6) → 
    ∀ x₁ x₂, 
      k * π - (π / 3) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + (π / 6) → 
        f x₁ < f x₂ := sorry

theorem part2 {A a b c : ℝ} 
  (h_a_seq : 2 * a = b + c) 
  (h_dot : b * c * cos A = 9) 
  (h_A_fA : f A = 1 / 2) 
  : 
  a = 3 * sqrt 2 := sorry

end

end NUMINAMATH_GPT_part1_part2_l979_97937


namespace NUMINAMATH_GPT_solve_for_q_l979_97902

theorem solve_for_q (t h q : ℝ) (h_eq : h = -14 * (t - 3)^2 + q) (h_5_eq : h = 94) (t_5_eq : t = 3 + 2) : q = 150 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l979_97902


namespace NUMINAMATH_GPT_coefficients_divisible_by_7_l979_97960

theorem coefficients_divisible_by_7 
  {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  ∃ k l m n o : ℤ, a = 7*k ∧ b = 7*l ∧ c = 7*m ∧ d = 7*n ∧ e = 7*o :=
by
  sorry

end NUMINAMATH_GPT_coefficients_divisible_by_7_l979_97960


namespace NUMINAMATH_GPT_evaluate_powers_of_i_mod_4_l979_97949

theorem evaluate_powers_of_i_mod_4 :
  (Complex.I ^ 48 + Complex.I ^ 96 + Complex.I ^ 144) = 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_powers_of_i_mod_4_l979_97949


namespace NUMINAMATH_GPT_child_running_speed_on_still_sidewalk_l979_97933

theorem child_running_speed_on_still_sidewalk (c s : ℕ) 
  (h1 : c + s = 93) 
  (h2 : c - s = 55) : c = 74 :=
sorry

end NUMINAMATH_GPT_child_running_speed_on_still_sidewalk_l979_97933


namespace NUMINAMATH_GPT_nat_lemma_l979_97955

theorem nat_lemma (a b : ℕ) : (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) → (a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_GPT_nat_lemma_l979_97955


namespace NUMINAMATH_GPT_soup_can_pyramid_rows_l979_97928

theorem soup_can_pyramid_rows (n : ℕ) :
  (∃ (n : ℕ), (2 * n^2 - n = 225)) → n = 11 :=
by
  sorry

end NUMINAMATH_GPT_soup_can_pyramid_rows_l979_97928


namespace NUMINAMATH_GPT_nine_point_five_minutes_in_seconds_l979_97924

-- Define the number of seconds in one minute
def seconds_per_minute : ℝ := 60

-- Define the function to convert minutes to seconds
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * seconds_per_minute

-- Define the theorem to prove
theorem nine_point_five_minutes_in_seconds : minutes_to_seconds 9.5 = 570 :=
by
  sorry

end NUMINAMATH_GPT_nine_point_five_minutes_in_seconds_l979_97924


namespace NUMINAMATH_GPT_mart_income_percentage_juan_l979_97915

-- Define the conditions
def TimIncomeLessJuan (J T : ℝ) : Prop := T = 0.40 * J
def MartIncomeMoreTim (T M : ℝ) : Prop := M = 1.60 * T

-- Define the proof problem
theorem mart_income_percentage_juan (J T M : ℝ) 
  (h1 : TimIncomeLessJuan J T) 
  (h2 : MartIncomeMoreTim T M) :
  M = 0.64 * J := 
  sorry

end NUMINAMATH_GPT_mart_income_percentage_juan_l979_97915


namespace NUMINAMATH_GPT_number_of_poles_needed_l979_97905

def length := 90
def width := 40
def distance_between_poles := 5

noncomputable def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem number_of_poles_needed (l w d : ℕ) : perimeter l w / d = 52 :=
by
  rw [perimeter]
  sorry

end NUMINAMATH_GPT_number_of_poles_needed_l979_97905


namespace NUMINAMATH_GPT_cost_of_three_pencils_and_two_pens_l979_97958

theorem cost_of_three_pencils_and_two_pens 
  (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.15) 
  (h2 : 2 * p + 3 * q = 3.70) : 
  3 * p + 2 * q = 4.15 := 
by 
  exact h1

end NUMINAMATH_GPT_cost_of_three_pencils_and_two_pens_l979_97958


namespace NUMINAMATH_GPT_remainder_of_3_pow_17_mod_7_l979_97921

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_17_mod_7_l979_97921


namespace NUMINAMATH_GPT_sum_of_first_9_terms_zero_l979_97919

variable (a_n : ℕ → ℝ) (d a₁ : ℝ)
def arithmetic_seq := ∀ n, a_n n = a₁ + (n - 1) * d

def condition (a_n : ℕ → ℝ) := (a_n 2 + a_n 9 = a_n 6)

theorem sum_of_first_9_terms_zero 
  (h_arith : arithmetic_seq a_n d a₁) 
  (h_cond : condition a_n) : 
  (9 * a₁ + (9 * 8 / 2) * d) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_zero_l979_97919


namespace NUMINAMATH_GPT_solve_inequality_x_squared_minus_6x_gt_15_l979_97940

theorem solve_inequality_x_squared_minus_6x_gt_15 :
  { x : ℝ | x^2 - 6 * x > 15 } = { x : ℝ | x < -1.5 } ∪ { x : ℝ | x > 7.5 } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_x_squared_minus_6x_gt_15_l979_97940


namespace NUMINAMATH_GPT_odd_function_f_value_l979_97913

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x + 1 else x^3 + x - 1

theorem odd_function_f_value : 
  f 2 = 9 := by
  sorry

end NUMINAMATH_GPT_odd_function_f_value_l979_97913


namespace NUMINAMATH_GPT_lcm_12_18_24_l979_97956

theorem lcm_12_18_24 : Nat.lcm (Nat.lcm 12 18) 24 = 72 := by
  -- Given conditions (prime factorizations)
  have h1 : 12 = 2^2 * 3 := by norm_num
  have h2 : 18 = 2 * 3^2 := by norm_num
  have h3 : 24 = 2^3 * 3 := by norm_num
  -- Prove the LCM
  sorry

end NUMINAMATH_GPT_lcm_12_18_24_l979_97956


namespace NUMINAMATH_GPT_rate_of_current_l979_97987

def downstream_eq (b c : ℝ) : Prop := (b + c) * 4 = 24
def upstream_eq (b c : ℝ) : Prop := (b - c) * 6 = 24

theorem rate_of_current (b c : ℝ) (h1 : downstream_eq b c) (h2 : upstream_eq b c) : c = 1 :=
by sorry

end NUMINAMATH_GPT_rate_of_current_l979_97987


namespace NUMINAMATH_GPT_minimum_value_of_w_l979_97994

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end NUMINAMATH_GPT_minimum_value_of_w_l979_97994


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l979_97909

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : a 5 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l979_97909


namespace NUMINAMATH_GPT_option_C_equals_a5_l979_97995

theorem option_C_equals_a5 (a : ℕ) : (a^4 * a = a^5) :=
by sorry

end NUMINAMATH_GPT_option_C_equals_a5_l979_97995


namespace NUMINAMATH_GPT_retailer_initial_thought_profit_percentage_l979_97916

/-
  An uneducated retailer marks all his goods at 60% above the cost price and thinking that he will still make some profit, 
  offers a discount of 25% on the marked price. 
  His actual profit on the sales is 20.000000000000018%. 
  Prove that the profit percentage the retailer initially thought he would make is 60%.
-/

theorem retailer_initial_thought_profit_percentage
  (cost_price marked_price selling_price : ℝ)
  (h1 : marked_price = cost_price + 0.6 * cost_price)
  (h2 : selling_price = marked_price - 0.25 * marked_price)
  (h3 : selling_price - cost_price = 0.20000000000000018 * cost_price) :
  0.6 * 100 = 60 := by
  sorry

end NUMINAMATH_GPT_retailer_initial_thought_profit_percentage_l979_97916


namespace NUMINAMATH_GPT_train_speed_proof_l979_97922

noncomputable def train_length : ℝ := 620
noncomputable def crossing_time : ℝ := 30.99752019838413
noncomputable def man_speed_kmh : ℝ := 8

noncomputable def man_speed_ms : ℝ := man_speed_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := train_length / crossing_time
noncomputable def train_speed_ms : ℝ := relative_speed + man_speed_ms
noncomputable def train_speed_kmh : ℝ := train_speed_ms * (3600 / 1000)

theorem train_speed_proof : abs (train_speed_kmh - 80) < 0.0001 := by
  sorry

end NUMINAMATH_GPT_train_speed_proof_l979_97922


namespace NUMINAMATH_GPT_incorrect_proposition_example_l979_97944

theorem incorrect_proposition_example (p q : Prop) (h : ¬ (p ∧ q)) : ¬ (¬p ∧ ¬q) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_proposition_example_l979_97944


namespace NUMINAMATH_GPT_day_after_1999_cubed_days_is_tuesday_l979_97963

theorem day_after_1999_cubed_days_is_tuesday : 
    let today := "Monday"
    let days_in_week := 7
    let target_days := 1999 ^ 3
    ∃ remaining_days, remaining_days = (target_days % days_in_week) ∧ today = "Monday" ∧ remaining_days = 1 → 
    "Tuesday" = "Tuesday" := 
by
  sorry

end NUMINAMATH_GPT_day_after_1999_cubed_days_is_tuesday_l979_97963


namespace NUMINAMATH_GPT_tan_beta_value_l979_97978

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan (α + β) = -1) : Real.tan β = 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_beta_value_l979_97978


namespace NUMINAMATH_GPT_travel_times_l979_97990

variable (t v1 v2 : ℝ)

def conditions := 
  (v1 * 2 = v2 * t) ∧ 
  (v2 * 4.5 = v1 * t)

theorem travel_times (h : conditions t v1 v2) : 
  t = 3 ∧ 
  (t + 2 = 5) ∧ 
  (t + 4.5 = 7.5) := by
  sorry

end NUMINAMATH_GPT_travel_times_l979_97990


namespace NUMINAMATH_GPT_triangle_isosceles_or_right_angled_l979_97908

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ) ∨ (β + γ = π / 2) :=
sorry

end NUMINAMATH_GPT_triangle_isosceles_or_right_angled_l979_97908


namespace NUMINAMATH_GPT_inverse_of_g_at_1_over_32_l979_97950

noncomputable def g (x : ℝ) : ℝ := (x^5 + 2) / 4

theorem inverse_of_g_at_1_over_32 :
  g⁻¹ (1/32) = (-15 / 8)^(1/5) :=
sorry

end NUMINAMATH_GPT_inverse_of_g_at_1_over_32_l979_97950


namespace NUMINAMATH_GPT_find_m_for_root_l979_97945

-- Define the fractional equation to find m
def fractional_equation (x m : ℝ) : Prop :=
  (x + 2) / (x - 1) = m / (1 - x)

-- State the theorem that we need to prove
theorem find_m_for_root : ∃ m : ℝ, (∃ x : ℝ, fractional_equation x m) ∧ m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_root_l979_97945


namespace NUMINAMATH_GPT_carol_maximizes_at_0_75_l979_97930

def winning_probability (a b c : ℝ) : Prop :=
(0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (a < c ∧ c < b ∨ b < c ∧ c < a)

theorem carol_maximizes_at_0_75 :
  ∀ (a b : ℝ), (0 ≤ a ∧ a ≤ 1) → (0.25 ≤ b ∧ b ≤ 0.75) → (∃ c : ℝ, 0 ≤ c ∧ c ≤ 1 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → winning_probability a b x ≤ winning_probability a b 0.75)) :=
sorry

end NUMINAMATH_GPT_carol_maximizes_at_0_75_l979_97930


namespace NUMINAMATH_GPT_find_larger_number_l979_97984

theorem find_larger_number :
  ∃ x y : ℤ, x + y = 30 ∧ 2 * y - x = 6 ∧ x > y ∧ x = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l979_97984


namespace NUMINAMATH_GPT_part1_part2_l979_97939

-- Define Set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}

-- Define Set B, parameterized by m
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m + 1}

-- Proof Problem (1): When m = 1, A ∩ B = {x | 0 < x ∧ x ≤ 3/2}
theorem part1 (x : ℝ) : (x ∈ A ∩ B 1) ↔ (0 < x ∧ x ≤ 3/2) := by
  sorry

-- Proof Problem (2): If ∀ x, x ∈ A → x ∈ B m, then m ∈ (-∞, 1/6]
theorem part2 (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) → m ≤ 1/6 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l979_97939


namespace NUMINAMATH_GPT_carlos_biked_more_than_daniel_l979_97942

-- Definitions modeled from conditions
def distance_carlos : ℕ := 108
def distance_daniel : ℕ := 90
def time_hours : ℕ := 6

-- Lean statement to prove the difference in distance
theorem carlos_biked_more_than_daniel : distance_carlos - distance_daniel = 18 := 
  by 
    sorry

end NUMINAMATH_GPT_carlos_biked_more_than_daniel_l979_97942


namespace NUMINAMATH_GPT_point_on_graph_l979_97998

theorem point_on_graph (g : ℝ → ℝ) (h : g 8 = 10) :
  ∃ x y : ℝ, 3 * y = g (3 * x - 1) + 3 ∧ x = 3 ∧ y = 13 / 3 ∧ x + y = 22 / 3 :=
by
  sorry

end NUMINAMATH_GPT_point_on_graph_l979_97998


namespace NUMINAMATH_GPT_max_digit_d_for_number_divisible_by_33_l979_97970

theorem max_digit_d_for_number_divisible_by_33 : ∃ d e : ℕ, d ≤ 9 ∧ e ≤ 9 ∧ 8 * 100000 + d * 10000 + 8 * 1000 + 3 * 100 + 3 * 10 + e % 33 = 0 ∧  d = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_digit_d_for_number_divisible_by_33_l979_97970


namespace NUMINAMATH_GPT_find_p_l979_97966

def delta (a b : ℝ) : ℝ := a * b + a + b

theorem find_p (p : ℝ) (h : delta p 3 = 39) : p = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l979_97966


namespace NUMINAMATH_GPT_compare_08_and_one_eighth_l979_97976

theorem compare_08_and_one_eighth :
  0.8 - (1 / 8 : ℝ) = 0.675 := 
sorry

end NUMINAMATH_GPT_compare_08_and_one_eighth_l979_97976


namespace NUMINAMATH_GPT_ellipse_range_m_l979_97962

theorem ellipse_range_m (m : ℝ) :
    (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2 → 
    ∃ (c : ℝ), c = x^2 + (y + 1)^2 ∧ m > 5) :=
sorry

end NUMINAMATH_GPT_ellipse_range_m_l979_97962


namespace NUMINAMATH_GPT_total_chocolate_bars_in_colossal_box_l979_97914

theorem total_chocolate_bars_in_colossal_box :
  let colossal_boxes := 350
  let sizable_boxes := 49
  let small_boxes := 75
  colossal_boxes * sizable_boxes * small_boxes = 1287750 :=
by
  sorry

end NUMINAMATH_GPT_total_chocolate_bars_in_colossal_box_l979_97914


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l979_97965

variables (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 5 = 36 ∧ (v_m - v_s) * 7 = 22 → v_m = 5.17 :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l979_97965


namespace NUMINAMATH_GPT_factor_of_60n_l979_97925

theorem factor_of_60n
  (n : ℕ)
  (x : ℕ)
  (h_condition1 : ∃ k : ℕ, 60 * n = x * k)
  (h_condition2 : ∃ m : ℕ, 60 * n = 8 * m)
  (h_condition3 : n >= 8) :
  x = 60 :=
sorry

end NUMINAMATH_GPT_factor_of_60n_l979_97925


namespace NUMINAMATH_GPT_museum_pictures_l979_97992

theorem museum_pictures (P : ℕ) (h1 : ¬ (∃ k, P = 2 * k)) (h2 : ∃ k, P + 1 = 2 * k) : P = 3 := 
by 
  sorry

end NUMINAMATH_GPT_museum_pictures_l979_97992


namespace NUMINAMATH_GPT_chris_average_price_l979_97935

noncomputable def total_cost_dvd (price_per_dvd : ℝ) (num_dvds : ℕ) (discount : ℝ) : ℝ :=
  (price_per_dvd * (1 - discount)) * num_dvds

noncomputable def total_cost_bluray (price_per_bluray : ℝ) (num_blurays : ℕ) : ℝ :=
  price_per_bluray * num_blurays

noncomputable def total_cost_ultra_hd (price_per_ultra_hd : ℝ) (num_ultra_hds : ℕ) : ℝ :=
  price_per_ultra_hd * num_ultra_hds

noncomputable def total_cost (cost_dvd cost_bluray cost_ultra_hd : ℝ) : ℝ :=
  cost_dvd + cost_bluray + cost_ultra_hd

noncomputable def total_with_tax (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  total_cost * (1 + tax_rate)

noncomputable def average_price (total_with_tax : ℝ) (total_movies : ℕ) : ℝ :=
  total_with_tax / total_movies

theorem chris_average_price :
  let price_per_dvd := 15
  let num_dvds := 5
  let discount := 0.20
  let price_per_bluray := 20
  let num_blurays := 8
  let price_per_ultra_hd := 25
  let num_ultra_hds := 3
  let tax_rate := 0.10
  let total_movies := num_dvds + num_blurays + num_ultra_hds
  let cost_dvd := total_cost_dvd price_per_dvd num_dvds discount
  let cost_bluray := total_cost_bluray price_per_bluray num_blurays
  let cost_ultra_hd := total_cost_ultra_hd price_per_ultra_hd num_ultra_hds
  let pre_tax_total := total_cost cost_dvd cost_bluray cost_ultra_hd
  let total := total_with_tax pre_tax_total tax_rate
  average_price total total_movies = 20.28 :=
by
  -- substitute each definition one step at a time
  -- to show the average price exactly matches 20.28
  sorry

end NUMINAMATH_GPT_chris_average_price_l979_97935


namespace NUMINAMATH_GPT_pond_volume_l979_97931

theorem pond_volume {L W H : ℝ} (hL : L = 20) (hW : W = 12) (hH : H = 5) : L * W * H = 1200 := by
  sorry

end NUMINAMATH_GPT_pond_volume_l979_97931


namespace NUMINAMATH_GPT_cuboid_volume_l979_97991

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 18) (h_height : height = 8) : 
  base_area * height = 144 :=
by
  rw [h_base_area, h_height]
  norm_num

end NUMINAMATH_GPT_cuboid_volume_l979_97991


namespace NUMINAMATH_GPT_inequality_proof_l979_97948

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l979_97948


namespace NUMINAMATH_GPT_find_number_l979_97977

def is_three_digit_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
  n = 100 * x + 10 * y + z ∧ (100 * x + 10 * y + z) / 11 = x^2 + y^2 + z^2

theorem find_number : ∃ n : ℕ, is_three_digit_number n ∧ n = 550 :=
sorry

end NUMINAMATH_GPT_find_number_l979_97977
