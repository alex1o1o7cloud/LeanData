import Mathlib

namespace parabola_equation_l27_27437

-- Define the given conditions
def vertex : ℝ × ℝ := (3, 5)
def point_on_parabola : ℝ × ℝ := (4, 2)

-- Prove that the equation is as specified
theorem parabola_equation :
  ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x y : ℝ, (y = a * x^2 + b * x + c) ↔
     (y = -3 * x^2 + 18 * x - 22) ∧ (vertex.snd = -3 * (vertex.fst - 3)^2 + 5) ∧
     (point_on_parabola.snd = a * point_on_parabola.fst^2 + b * point_on_parabola.fst + c)) := 
sorry

end parabola_equation_l27_27437


namespace problem_statement_l27_27795

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

def n : ℕ := sorry  -- n is the number of possible values of g(3)
def s : ℝ := sorry  -- s is the sum of all possible values of g(3)

theorem problem_statement : n * s = 0 := sorry

end problem_statement_l27_27795


namespace parallel_and_equidistant_line_l27_27110

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 6 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 3 = 0

-- Define the desired property: a line parallel to line1 and line2, and equidistant from both
theorem parallel_and_equidistant_line :
  ∃ b : ℝ, ∀ x y : ℝ, (3 * x + 2 * y + b = 0) ∧
  (|-6 - b| / Real.sqrt (9 + 4) = |-3/2 - b| / Real.sqrt (9 + 4)) →
  (12 * x + 8 * y - 15 = 0) :=
by
  sorry

end parallel_and_equidistant_line_l27_27110


namespace sum_of_squares_99_in_distinct_ways_l27_27861

theorem sum_of_squares_99_in_distinct_ways : 
  ∃ a b c d e f g h i j k l : ℕ, 
    (a^2 + b^2 + c^2 + d^2 = 99) ∧ (e^2 + f^2 + g^2 + h^2 = 99) ∧ (i^2 + j^2 + k^2 + l^2 = 99) ∧ 
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) ∧ 
    (a ≠ i ∨ b ≠ j ∨ c ≠ k ∨ d ≠ l) ∧ 
    (i ≠ e ∨ j ≠ f ∨ k ≠ g ∨ l ≠ h) 
    :=
sorry

end sum_of_squares_99_in_distinct_ways_l27_27861


namespace parallel_lines_m_eq_l27_27246

theorem parallel_lines_m_eq (m : ℝ) : 
  (∃ k : ℝ, (x y : ℝ) → 2 * x + (m + 1) * y + 4 = k * (m * x + 3 * y - 2)) → 
  (m = 2 ∨ m = -3) :=
by
  intro h
  sorry

end parallel_lines_m_eq_l27_27246


namespace find_slope_l27_27779

theorem find_slope (k : ℝ) : (∃ x : ℝ, (y = k * x + 2) ∧ (y = 0) ∧ (abs x = 4)) ↔ (k = 1/2 ∨ k = -1/2) := by
  sorry

end find_slope_l27_27779


namespace jeff_total_cabinets_l27_27618

theorem jeff_total_cabinets : 
  let initial_cabinets := 3
  let additional_per_counter := 3 * 2
  let num_counters := 3
  let additional_total := additional_per_counter * num_counters
  let final_cabinets := additional_total + 5
in initial_cabinets + final_cabinets = 26 :=
by
  -- Proof omitted
  sorry

end jeff_total_cabinets_l27_27618


namespace series_satisfies_l27_27741

noncomputable def series (x : ℝ) : ℝ :=
  let S₁ := 1 / (1 + x^2)
  let S₂ := x / (1 + x^2)
  (S₁ - S₂)

theorem series_satisfies (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  x = series x ↔ x^3 + 2 * x - 1 = 0 :=
by 
  -- Proof outline:
  -- 1. Calculate the series S as a function of x
  -- 2. Equate series x to x and simplify to derive the polynomial equation
  sorry

end series_satisfies_l27_27741


namespace scale_down_multiplication_l27_27746

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end scale_down_multiplication_l27_27746


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27261

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27261


namespace original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l27_27363

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l27_27363


namespace beetle_distance_l27_27058

theorem beetle_distance :
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  dist1 + dist2 = 20 :=
by
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  show dist1 + dist2 = 20
  sorry

end beetle_distance_l27_27058


namespace least_possible_length_of_third_side_l27_27894

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l27_27894


namespace abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l27_27642

theorem abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one 
  (a b c : ℝ) 
  (h1 : a * b * c = 1) 
  (h2 : a + b + c = 1 / a + 1 / b + 1 / c) : 
  (a = 1) ∨ (b = 1) ∨ (c = 1) :=
by
  sorry

end abc_eq_one_and_sum_eq_reciprocal_implies_one_is_one_l27_27642


namespace t_shirt_sale_revenue_per_minute_l27_27514

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end t_shirt_sale_revenue_per_minute_l27_27514


namespace point_above_line_l27_27877

-- Define the point P with coordinates (-2, t)
variable (t : ℝ)

-- Define the line equation
def line_eq (x y : ℝ) : ℝ := 2 * x - 3 * y + 6

-- Proving that t must be greater than 2/3 for the point P to be above the line
theorem point_above_line : (line_eq (-2) t < 0) -> t > 2 / 3 :=
by
  sorry

end point_above_line_l27_27877


namespace xyz_positive_and_distinct_l27_27949

theorem xyz_positive_and_distinct (a b x y z : ℝ)
  (h₁ : x + y + z = a)
  (h₂ : x^2 + y^2 + z^2 = b^2)
  (h₃ : x * y = z^2)
  (ha_pos : a > 0)
  (hb_condition : b^2 < a^2 ∧ a^2 < 3*b^2) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end xyz_positive_and_distinct_l27_27949


namespace tina_wins_before_first_loss_l27_27539

-- Definitions based on conditions
variable (W : ℕ) -- The number of wins before Tina's first loss

-- Conditions
def win_before_first_loss : W = 10 := by sorry

def total_wins (W : ℕ) := W + 2 * W -- After her first loss, she doubles her wins and loses again
def total_losses : ℕ := 2 -- She loses twice

def career_record_condition (W : ℕ) : Prop := total_wins W - total_losses = 28

-- Proof Problem (Statement)
theorem tina_wins_before_first_loss : career_record_condition W → W = 10 :=
by sorry

end tina_wins_before_first_loss_l27_27539


namespace exists_1998_distinct_natural_numbers_l27_27569

noncomputable def exists_1998_distinct_numbers : Prop :=
  ∃ (s : Finset ℕ), s.card = 1998 ∧
    (∀ {x y : ℕ}, x ∈ s → y ∈ s → x ≠ y → (x * y) % ((x - y) ^ 2) = 0)

theorem exists_1998_distinct_natural_numbers : exists_1998_distinct_numbers :=
by
  sorry

end exists_1998_distinct_natural_numbers_l27_27569


namespace least_third_side_of_right_triangle_l27_27900

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l27_27900


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l27_27691

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l27_27691


namespace cross_sectional_area_volume_of_R_l27_27584

open Real

def semicircle (x : ℝ) := sqrt (1 - x^2)

def S (x : ℝ) : ℝ := (1 + 2 * x) / (1 + x) * sqrt (1 - x^2)

theorem cross_sectional_area (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  S(x) = (1 + 2 * x) / (1 + x) * sqrt (1 - x^2) := 
sorry

theorem volume_of_R :
  (∫ x in 0..1, S(x)) = 1 :=
sorry

end cross_sectional_area_volume_of_R_l27_27584


namespace smallest_integer_with_remainders_l27_27038

theorem smallest_integer_with_remainders :
  ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m) := 
sorry

end smallest_integer_with_remainders_l27_27038


namespace positive_solutions_l27_27436

theorem positive_solutions (x : ℝ) (hx : x > 0) :
  x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 ↔
  x = 1 ∨ x = 3 :=
by
  sorry

end positive_solutions_l27_27436


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27259

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27259


namespace diameter_increase_l27_27063

theorem diameter_increase (D D' : ℝ) (h : π * (D' / 2) ^ 2 = 2.4336 * π * (D / 2) ^ 2) : D' / D = 1.56 :=
by
  -- Statement only, proof is omitted
  sorry

end diameter_increase_l27_27063


namespace max_flags_l27_27910

theorem max_flags (n : ℕ) (h1 : ∀ k, n = 9 * k) (h2 : n ≤ 200)
  (h3 : ∃ m, n = 9 * m + k ∧ k ≤ 2 ∧ k + 1 ≠ 0 ∧ k - 2 ≠ 0) : n = 198 :=
by {
  sorry
}

end max_flags_l27_27910


namespace product_remainder_mod_7_l27_27677

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l27_27677


namespace quadratic_one_solution_l27_27767

theorem quadratic_one_solution (b d : ℝ) (h1 : b + d = 35) (h2 : b < d) (h3 : (24 : ℝ)^2 - 4 * b * d = 0) :
  (b, d) = (35 - Real.sqrt 649 / 2, 35 + Real.sqrt 649 / 2) := 
sorry

end quadratic_one_solution_l27_27767


namespace range_of_a_l27_27465

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := 
sorry

end range_of_a_l27_27465


namespace uncle_bob_can_park_l27_27717

-- Define the conditions
def total_spaces : Nat := 18
def cars : Nat := 15
def rv_spaces : Nat := 3

-- Define a function to calculate the probability (without implementation)
noncomputable def probability_RV_can_park (total_spaces cars rv_spaces : Nat) : Rat :=
  if h : rv_spaces <= total_spaces - cars then
    -- The probability calculation logic would go here
    16 / 51
  else
    0

-- The theorem stating the desired result
theorem uncle_bob_can_park : probability_RV_can_park total_spaces cars rv_spaces = 16 / 51 :=
  sorry

end uncle_bob_can_park_l27_27717


namespace find_other_integer_l27_27799

theorem find_other_integer (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : y = 14 ∨ x = 14 :=
  sorry

end find_other_integer_l27_27799


namespace erin_trolls_count_l27_27432

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l27_27432


namespace original_laborers_l27_27387

theorem original_laborers (x : ℕ) : (x * 8 = (x - 3) * 14) → x = 7 :=
by
  intro h
  sorry

end original_laborers_l27_27387


namespace total_trolls_l27_27428

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l27_27428


namespace union_of_M_and_N_is_correct_l27_27108

def M : Set ℤ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 ≤ n ∧ n ≤ 3 }

theorem union_of_M_and_N_is_correct : M ∪ N = { -2, -1, 0, 1, 2, 3 } := 
by
  sorry

end union_of_M_and_N_is_correct_l27_27108


namespace medium_kite_area_l27_27742

-- Define the points and the spacing on the grid
structure Point :=
mk :: (x : ℕ) (y : ℕ)

def medium_kite_vertices : List Point :=
[Point.mk 0 4, Point.mk 4 10, Point.mk 12 4, Point.mk 4 0]

def grid_spacing : ℕ := 2

-- Function to calculate the area of a kite given list of vertices and spacing
noncomputable def area_medium_kite (vertices : List Point) (spacing : ℕ) : ℕ := sorry

-- The theorem to be proved
theorem medium_kite_area (vertices : List Point) (spacing : ℕ) :
  vertices = medium_kite_vertices ∧ spacing = grid_spacing → area_medium_kite vertices spacing = 288 := 
by {
  -- The detailed proof would go here
  sorry
}

end medium_kite_area_l27_27742


namespace product_remainder_mod_7_l27_27662

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l27_27662


namespace min_value_expression_l27_27097

theorem min_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ x : ℝ, x = (1 / (a - 1)) + (1 / (2 * b)) ∧ x ≥ (3 / 2 + Real.sqrt 2)) :=
sorry

end min_value_expression_l27_27097


namespace min_distance_l27_27874

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - (1/2) * Real.log x
noncomputable def line (x : ℝ) : ℝ := (3/4) * x - 1

theorem min_distance :
  ∀ P Q : ℝ × ℝ, 
  P.2 = curve P.1 → 
  Q.2 = line Q.1 → 
  ∃ min_dist : ℝ, 
  min_dist = (2 - 2 * Real.log 2) / 5 := 
sorry

end min_distance_l27_27874


namespace branches_count_eq_6_l27_27735

theorem branches_count_eq_6 (x : ℕ) (h : 1 + x + x^2 = 43) : x = 6 :=
sorry

end branches_count_eq_6_l27_27735


namespace mia_money_l27_27502

def darwin_has := 45
def mia_has (d : ℕ) := 2 * d + 20

theorem mia_money : mia_has darwin_has = 110 :=
by
  unfold mia_has darwin_has
  rw [←nat.mul_assoc]
  rw [nat.mul_comm 2 45]
  sorry

end mia_money_l27_27502


namespace sum_of_roots_of_cis_equation_l27_27529

theorem sum_of_roots_of_cis_equation 
  (cis : ℝ → ℂ)
  (phi : ℕ → ℝ)
  (h_conditions : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 0 ≤ phi k ∧ phi k < 360)
  (h_equation : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → (cis (phi k)) ^ 5 = (1 / Real.sqrt 2) + (Complex.I / Real.sqrt 2))
  : (phi 1 + phi 2 + phi 3 + phi 4 + phi 5) = 450 :=
by
  sorry

end sum_of_roots_of_cis_equation_l27_27529


namespace product_mod_7_l27_27659

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l27_27659


namespace anand_income_l27_27389

theorem anand_income (x y : ℕ)
  (income_A : ℕ := 5 * x)
  (income_B : ℕ := 4 * x)
  (expenditure_A : ℕ := 3 * y)
  (expenditure_B : ℕ := 2 * y)
  (savings_A : ℕ := 800)
  (savings_B : ℕ := 800)
  (hA : income_A - expenditure_A = savings_A)
  (hB : income_B - expenditure_B = savings_B) :
  income_A = 2000 := by
  sorry

end anand_income_l27_27389


namespace product_mod_7_l27_27658

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l27_27658


namespace solve_a_for_pure_imaginary_l27_27303

theorem solve_a_for_pure_imaginary (a : ℝ) : (1 - a^2 = 0) ∧ (2 * a ≠ 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end solve_a_for_pure_imaginary_l27_27303


namespace smallest_four_digit_multiple_of_53_l27_27992

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27992


namespace find_F_58_59_60_l27_27850

def F : ℤ → ℤ → ℤ → ℝ := sorry

axiom F_scaling (a b c n : ℤ) : F (n * a) (n * b) (n * c) = n * F a b c
axiom F_shift (a b c n : ℤ) : F (a + n) (b + n) (c + n) = F a b c + n
axiom F_symmetry (a b c : ℤ) : F a b c = F c b a

theorem find_F_58_59_60 : F 58 59 60 = 59 :=
sorry

end find_F_58_59_60_l27_27850


namespace product_negative_probability_l27_27028

theorem product_negative_probability:
  let S := ({-5, -8, 7, 4, -2, 0, 6} : Finset ℤ),
      neg_ints := ({-5, -8, -2} : Finset ℤ),
      pos_ints := ({7, 4, 6} : Finset ℤ) in
  (S.card = 7) ∧
  (neg_ints.card = 3) ∧
  (pos_ints.card = 3) ∧
  (0 ∈ S) →
  let total_ways := (S.card.choose 3),
      neg_product_ways := ((neg_ints.card.choose 2) * (pos_ints.card.choose 1) +
                            neg_ints.card.choose 3) in
  (neg_product_ways : ℚ) / total_ways = 2 / 7 :=
by
  sorry

end product_negative_probability_l27_27028


namespace innings_played_l27_27359

noncomputable def cricket_player_innings : Nat :=
  let average_runs := 32
  let increase_in_average := 6
  let next_innings_runs := 158
  let new_average := average_runs + increase_in_average
  let runs_before_next_innings (n : Nat) := average_runs * n
  let total_runs_after_next_innings (n : Nat) := runs_before_next_innings n + next_innings_runs
  let total_runs_with_new_average (n : Nat) := new_average * (n + 1)

  let n := (total_runs_after_next_innings 20) - (total_runs_with_new_average 20)
  
  n
     
theorem innings_played : cricket_player_innings = 20 := by
  sorry

end innings_played_l27_27359


namespace sum_of_square_roots_l27_27700

theorem sum_of_square_roots : 
  (Real.sqrt 1) + (Real.sqrt (1 + 3)) + (Real.sqrt (1 + 3 + 5)) + (Real.sqrt (1 + 3 + 5 + 7)) = 10 := 
by 
  sorry

end sum_of_square_roots_l27_27700


namespace not_possible_2018_people_in_2019_minutes_l27_27615

-- Definitions based on conditions
def initial_people (t : ℕ) : ℕ := 0
def changed_people (x y : ℕ) : ℕ := 2 * x - y

theorem not_possible_2018_people_in_2019_minutes :
  ¬ ∃ (x y : ℕ), (x + y = 2019) ∧ (2 * x - y = 2018) :=
by
  sorry

end not_possible_2018_people_in_2019_minutes_l27_27615


namespace least_possible_length_of_third_side_l27_27893

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l27_27893


namespace scaled_multiplication_l27_27748

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end scaled_multiplication_l27_27748


namespace toothpaste_amount_in_tube_l27_27819

def dad_usage_per_brush : ℕ := 3
def mom_usage_per_brush : ℕ := 2
def kid_usage_per_brush : ℕ := 1
def brushes_per_day : ℕ := 3
def days : ℕ := 5

theorem toothpaste_amount_in_tube (dad_usage_per_brush mom_usage_per_brush kid_usage_per_brush brushes_per_day days : ℕ) : 
  dad_usage_per_brush * brushes_per_day * days + 
  mom_usage_per_brush * brushes_per_day * days + 
  (kid_usage_per_brush * brushes_per_day * days * 2) = 105 := 
  by sorry

end toothpaste_amount_in_tube_l27_27819


namespace least_possible_length_of_third_side_l27_27892

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l27_27892


namespace quadratic_has_distinct_real_roots_expression_value_l27_27254

variable (x m : ℝ)

-- Condition: Quadratic equation
def quadratic_eq := (x^2 - 2 * (m - 1) * x - m * (m + 2) = 0)

-- Prove that the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (m : ℝ) : 
  ∃ a b : ℝ, a ≠ b ∧ quadratic_eq a m ∧ quadratic_eq b m :=
by
  sorry

-- Given that x = -2 is a root, prove that 2018 - 3(m-1)^2 = 2015
theorem expression_value (m : ℝ) (h : quadratic_eq (-2) m) : 
  2018 - 3 * (m - 1)^2 = 2015 :=
by
  sorry

end quadratic_has_distinct_real_roots_expression_value_l27_27254


namespace product_mod_7_l27_27660

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l27_27660


namespace product_remainder_mod_7_l27_27678

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l27_27678


namespace simplest_quadratic_radical_l27_27839

theorem simplest_quadratic_radical :
  ∀ (a b c d : ℝ),
    a = Real.sqrt 12 →
    b = Real.sqrt (2 / 3) →
    c = Real.sqrt 0.3 →
    d = Real.sqrt 7 →
    d = Real.sqrt 7 :=
by
  intros a b c d ha hb hc hd
  rw [hd]
  sorry

end simplest_quadratic_radical_l27_27839


namespace solution_set_of_f_inequality_l27_27098

variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_deriv : ∀ x, f' x < f x)
variable (h_even : ∀ x, f (x + 2) = f (-x + 2))
variable (h_initial : f 0 = Real.exp 4)

theorem solution_set_of_f_inequality :
  {x : ℝ | f x < Real.exp x} = {x : ℝ | x > 4} := 
sorry

end solution_set_of_f_inequality_l27_27098


namespace factor_is_two_l27_27555

theorem factor_is_two (n f : ℤ) (h1 : n = 121) (h2 : n * f - 140 = 102) : f = 2 :=
by
  sorry

end factor_is_two_l27_27555


namespace competition_end_time_l27_27221

def time_in_minutes := 24 * 60  -- Total minutes in 24 hours

def competition_start_time := 14 * 60 + 30  -- 2:30 p.m. in minutes from midnight

theorem competition_end_time :
  competition_start_time + 1440 = competition_start_time :=
by 
  sorry

end competition_end_time_l27_27221


namespace range_of_sqrt_x_plus_3_l27_27144

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l27_27144


namespace function_range_l27_27590

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sin x

theorem function_range : 
  ∀ x : ℝ, (0 < x ∧ x < Real.pi) → 1 ≤ f x ∧ f x ≤ 3 / 2 :=
by
  intro x
  sorry

end function_range_l27_27590


namespace seven_digit_palindromes_l27_27882

def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

theorem seven_digit_palindromes : 
  (∃ l : List ℕ, l = [1, 1, 4, 4, 4, 6, 6] ∧ 
  ∃ pl : List ℕ, pl.length = 7 ∧ is_palindrome pl ∧ 
  ∀ d, d ∈ pl → d ∈ l) →
  ∃! n, n = 12 :=
by
  sorry

end seven_digit_palindromes_l27_27882


namespace remainder_of_N_mod_37_l27_27226

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_of_N_mod_37_l27_27226


namespace find_R_value_l27_27497

noncomputable def x (Q : ℝ) : ℝ := Real.sqrt (Q / 2 + Real.sqrt (Q / 2))
noncomputable def y (Q : ℝ) : ℝ := Real.sqrt (Q / 2 - Real.sqrt (Q / 2))
noncomputable def R (Q : ℝ) : ℝ := (x Q)^6 + (y Q)^6 / 40

theorem find_R_value (Q : ℝ) : R Q = 10 :=
sorry

end find_R_value_l27_27497


namespace oranges_savings_l27_27630

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l27_27630


namespace total_trolls_l27_27430

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l27_27430


namespace describe_difference_of_squares_l27_27083

def description_of_a_squared_minus_b_squared : Prop :=
  ∃ (a b : ℝ), (a^2 - b^2) = (a^2 - b^2)

theorem describe_difference_of_squares :
  description_of_a_squared_minus_b_squared :=
by sorry

end describe_difference_of_squares_l27_27083


namespace sum_rows_7_8_pascal_triangle_l27_27383

theorem sum_rows_7_8_pascal_triangle : (2^7 + 2^8 = 384) :=
by
  sorry

end sum_rows_7_8_pascal_triangle_l27_27383


namespace calculate_expression_l27_27860

variable (x : ℝ)

theorem calculate_expression : ((3 * x)^2) * (x^2) = 9 * (x^4) := 
sorry

end calculate_expression_l27_27860


namespace number_as_A_times_10_pow_N_integer_l27_27716

theorem number_as_A_times_10_pow_N_integer (A : ℝ) (N : ℝ) (hA1 : 1 ≤ A) (hA2 : A < 10) (hN : A * 10^N > 10) : ∃ (n : ℤ), N = n := 
sorry

end number_as_A_times_10_pow_N_integer_l27_27716


namespace least_third_side_of_right_triangle_l27_27899

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l27_27899


namespace smallest_four_digit_divisible_by_53_l27_27989

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27989


namespace sum_of_numbers_l27_27703

theorem sum_of_numbers (a b c : ℝ) :
  a^2 + b^2 + c^2 = 138 → ab + bc + ca = 131 → a + b + c = 20 :=
by
  sorry

end sum_of_numbers_l27_27703


namespace find_intersection_line_of_planes_l27_27764

-- Definitions needed for the problem
variables {P : Type} [plane P]
variables {A B : P} -- planes A and B
variables {t₁ t₂ : line P} -- first traces of the planes
variables {α β : ℝ} -- first angles of inclination of planes

-- The theorem statement we want to prove
theorem find_intersection_line_of_planes 
  (hA1 : A.first_trace = t₁) (hA2 : A.angle_of_inclination = α)
  (hB1 : B.first_trace = t₂) (hB2 : B.angle_of_inclination = β) :
  ∃ L : line P, (L.intersects A) ∧ (L.intersects B) := 
sorry

end find_intersection_line_of_planes_l27_27764


namespace percentage_error_l27_27719

-- Define the conditions
def actual_side (a : ℝ) := a
def measured_side (a : ℝ) := 1.05 * a
def actual_area (a : ℝ) := a^2
def calculated_area (a : ℝ) := (1.05 * a)^2

-- Define the statement that we need to prove
theorem percentage_error (a : ℝ) (h : a > 0) :
  (calculated_area a - actual_area a) / actual_area a * 100 = 10.25 :=
by
  -- Proof goes here
  sorry

end percentage_error_l27_27719


namespace cos_sum_identity_l27_27968

theorem cos_sum_identity :
  (Real.cos (75 * Real.pi / 180)) ^ 2 + (Real.cos (15 * Real.pi / 180)) ^ 2 + 
  (Real.cos (75 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 5 / 4 := 
by
  sorry

end cos_sum_identity_l27_27968


namespace rich_walks_ratio_is_2_l27_27940

-- Define the conditions in the problem
def house_to_sidewalk : ℕ := 20
def sidewalk_to_end : ℕ := 200
def total_distance_walked : ℕ := 1980
def ratio_after_left_to_so_far (x : ℕ) : ℕ := (house_to_sidewalk + sidewalk_to_end) * x / (house_to_sidewalk + sidewalk_to_end)

-- Main theorem to prove the ratio is 2:1
theorem rich_walks_ratio_is_2 (x : ℕ) (h : 2 * ((house_to_sidewalk + sidewalk_to_end) * 2 + house_to_sidewalk + sidewalk_to_end / 2 * 3 ) = total_distance_walked) :
  ratio_after_left_to_so_far x = 2 :=
by
  sorry

end rich_walks_ratio_is_2_l27_27940


namespace not_rectangle_determined_by_angle_and_side_l27_27040

axiom parallelogram_determined_by_two_sides_and_angle : Prop
axiom equilateral_triangle_determined_by_area : Prop
axiom square_determined_by_perimeter_and_side : Prop
axiom rectangle_determined_by_two_diagonals : Prop
axiom rectangle_determined_by_angle_and_side : Prop

theorem not_rectangle_determined_by_angle_and_side : ¬rectangle_determined_by_angle_and_side := 
sorry

end not_rectangle_determined_by_angle_and_side_l27_27040


namespace values_of_x0_l27_27096

noncomputable def x_seq (x_0 : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x_0
  | n + 1 => if 3 * (x_seq x_0 n) < 1 then 3 * (x_seq x_0 n)
             else if 3 * (x_seq x_0 n) < 2 then 3 * (x_seq x_0 n) - 1
             else 3 * (x_seq x_0 n) - 2

theorem values_of_x0 (x_0 : ℝ) (h : 0 ≤ x_0 ∧ x_0 < 1) :
  (∃! x_0, x_0 = x_seq x_0 6) → (x_seq x_0 6 = x_0) :=
  sorry

end values_of_x0_l27_27096


namespace simplify_expression_l27_27342

variable (x : ℝ)

theorem simplify_expression : 2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := 
  sorry

end simplify_expression_l27_27342


namespace find_g_9_l27_27319

-- Define the function g
def g (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 7

-- Given conditions
variables (a b c : ℝ)

-- g(-9) = 9
axiom h : g a b c (-9) = 9

-- Prove g(9) = -23
theorem find_g_9 : g a b c 9 = -23 :=
by
  sorry

end find_g_9_l27_27319


namespace smallest_four_digit_multiple_of_53_l27_27997

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27997


namespace stella_profit_l27_27511

def price_of_doll := 5
def price_of_clock := 15
def price_of_glass := 4

def number_of_dolls := 3
def number_of_clocks := 2
def number_of_glasses := 5

def cost := 40

def dolls_sales := number_of_dolls * price_of_doll
def clocks_sales := number_of_clocks * price_of_clock
def glasses_sales := number_of_glasses * price_of_glass

def total_sales := dolls_sales + clocks_sales + glasses_sales

def profit := total_sales - cost

theorem stella_profit : profit = 25 :=
by 
  sorry

end stella_profit_l27_27511


namespace range_of_independent_variable_l27_27138

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l27_27138


namespace total_savings_l27_27633

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l27_27633


namespace simplify_fraction_l27_27945

theorem simplify_fraction : 
  (16777216 = 16 ^ 6) → (Real.sqrt (Real.cbrt (Real.sqrt (1 / 16777216))) = 1 / 4) := 
by
  intro h
  sorry

end simplify_fraction_l27_27945


namespace work_done_by_gas_l27_27337

theorem work_done_by_gas (n : ℕ) (R T0 Pa : ℝ) (V0 : ℝ) (W : ℝ) :
  -- Conditions
  n = 1 ∧
  R = 8.314 ∧
  T0 = 320 ∧
  Pa * V0 = n * R * T0 ∧
  -- Question Statement and Correct Answer
  W = Pa * V0 / 2 →
  W = 665 :=
by sorry

end work_done_by_gas_l27_27337


namespace find_sum_l27_27559

theorem find_sum (A B : ℕ) (h1 : B = 278 + 365 * 3) (h2 : A = 20 * 100 + 87 * 10) : A + B = 4243 := by
    sorry

end find_sum_l27_27559


namespace no_constant_term_in_expansion_l27_27979

theorem no_constant_term_in_expansion : 
  ∀ (x : ℂ), ¬ ∃ (k : ℕ), ∃ (c : ℂ), c * x ^ (k / 3 - 2 * (12 - k)) = 0 :=
by sorry

end no_constant_term_in_expansion_l27_27979


namespace smallest_angle_in_triangle_l27_27188

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end smallest_angle_in_triangle_l27_27188


namespace find_number_l27_27027

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
by
  sorry

end find_number_l27_27027


namespace maximum_real_roots_maximum_total_real_roots_l27_27495

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

def quadratic_discriminant (p q r : ℝ) : ℝ := q^2 - 4 * p * r

theorem maximum_real_roots (h1 : quadratic_discriminant a b c < 0)
  (h2 : quadratic_discriminant b c a < 0)
  (h3 : quadratic_discriminant c a b < 0) :
  ∀ (x : ℝ), (a * x^2 + b * x + c ≠ 0) ∧ 
             (b * x^2 + c * x + a ≠ 0) ∧ 
             (c * x^2 + a * x + b ≠ 0) :=
sorry

theorem maximum_total_real_roots :
    ∃ x : ℝ, ∃ y : ℝ, ∃ z : ℝ,
    (a * x^2 + b * x + c = 0) ∧
    (b * y^2 + c * y + a = 0) ∧
    (a * y ≠ x) ∧
    (c * z^2 + a * z + b = 0) ∧
    (b * z ≠ x) ∧
    (c * z ≠ y) :=
sorry

end maximum_real_roots_maximum_total_real_roots_l27_27495


namespace turtles_still_on_sand_l27_27408

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l27_27408


namespace simple_interest_calculation_l27_27781

-- Define the known quantities
def principal : ℕ := 400
def rate_of_interest : ℕ := 15
def time : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Statement to be proved
theorem simple_interest_calculation :
  simple_interest principal rate_of_interest time = 60 :=
by
  -- This space is used for the proof, We assume the user will complete it
  sorry

end simple_interest_calculation_l27_27781


namespace sin_pi_plus_alpha_l27_27102

theorem sin_pi_plus_alpha {α : ℝ} (h1 : cos α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) = -12 / 13 :=
sorry

end sin_pi_plus_alpha_l27_27102


namespace square_side_length_l27_27853

variable (s : ℝ)
variable (k : ℝ := 6)

theorem square_side_length :
  s^2 = k * 4 * s → s = 24 :=
by
  intro h
  sorry

end square_side_length_l27_27853


namespace find_a_c_l27_27656

theorem find_a_c (a c : ℝ) (h_discriminant : ∀ x : ℝ, a * x^2 + 10 * x + c = 0 → ∃ k : ℝ, a * k^2 + 10 * k + c = 0 ∧ (a * x^2 + 10 * k + c = 0 → x = k))
  (h_sum : a + c = 12) (h_lt : a < c) : (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end find_a_c_l27_27656


namespace total_savings_l27_27632

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l27_27632


namespace survey_pop_and_coke_l27_27306

theorem survey_pop_and_coke (total_people : ℕ) (angle_pop angle_coke : ℕ) 
  (h_total : total_people = 500) (h_angle_pop : angle_pop = 240) (h_angle_coke : angle_coke = 90) :
  ∃ (pop_people coke_people : ℕ), pop_people = 333 ∧ coke_people = 125 :=
by 
  sorry

end survey_pop_and_coke_l27_27306


namespace find_k_l27_27766

noncomputable def vector_a : ℝ × ℝ := (-1, 1)
noncomputable def vector_b : ℝ × ℝ := (2, 3)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (-2, k)

def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem find_k (k : ℝ) (h : perp (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) (vector_c k)) : k = 1 / 2 :=
by
  sorry

end find_k_l27_27766


namespace total_wheels_in_garage_l27_27191

theorem total_wheels_in_garage (num_bicycles : ℕ) (wheels_per_bicycle : ℕ) 
                               (num_cars : ℕ) (wheels_per_car : ℕ) :
  num_bicycles = 9 → wheels_per_bicycle = 2 → 
  num_cars = 16 → wheels_per_car = 4 → 
  (num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car) = 82 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_wheels_in_garage_l27_27191


namespace inequality_sqrt_ab_l27_27252

theorem inequality_sqrt_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 / (1 / a + 1 / b) ≤ Real.sqrt (a * b) :=
sorry

end inequality_sqrt_ab_l27_27252


namespace difference_of_squares_36_l27_27531

theorem difference_of_squares_36 {x y : ℕ} (h₁ : x + y = 18) (h₂ : x * y = 80) (h₃ : x > y) : x^2 - y^2 = 36 :=
by
  sorry

end difference_of_squares_36_l27_27531


namespace rectangle_breadth_l27_27184

theorem rectangle_breadth (length radius side breadth: ℝ)
  (h1: length = (2/5) * radius)
  (h2: radius = side)
  (h3: side ^ 2 = 1600)
  (h4: length * breadth = 160) :
  breadth = 10 := 
by
  sorry

end rectangle_breadth_l27_27184


namespace no_real_roots_of_quadratic_l27_27112

theorem no_real_roots_of_quadratic :
  ∀ (a b c : ℝ), a = 1 → b = -Real.sqrt 5 → c = Real.sqrt 2 →
  (b^2 - 4 * a * c < 0) → ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros a b c ha hb hc hD
  rw [ha, hb, hc] at hD
  sorry

end no_real_roots_of_quadratic_l27_27112


namespace share_of_B_l27_27388

theorem share_of_B (x : ℕ) (A B C : ℕ) (h1 : A = 3 * B) (h2 : B = C + 25)
  (h3 : A + B + C = 645) : B = 134 :=
by
  sorry

end share_of_B_l27_27388


namespace number_of_2_dollar_socks_l27_27881

theorem number_of_2_dollar_socks :
  ∃ (a b c : ℕ), (a + b + c = 15) ∧ (2 * a + 3 * b + 5 * c = 40) ∧ (a ≥ 1) ∧ (b ≥ 1) ∧ (c ≥ 1) ∧ (a = 7 ∨ a = 9 ∨ a = 11) :=
by {
  -- The details of the proof will go here, but we skip it for our requirements
  sorry
}

end number_of_2_dollar_socks_l27_27881


namespace total_trolls_l27_27429

theorem total_trolls (P B T : ℕ) (hP : P = 6) (hB : B = 4 * P - 6) (hT : T = B / 2) : P + B + T = 33 := by
  sorry

end total_trolls_l27_27429


namespace value_of_a_l27_27422

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem value_of_a : ∃ a : ℤ, star a 3 = 63 ∧ a = 30 := by
  sorry

end value_of_a_l27_27422


namespace polynomial_coeff_sum_l27_27120

theorem polynomial_coeff_sum :
  let p1 : Polynomial ℝ := Polynomial.C 4 * Polynomial.X ^ 2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 5
  let p2 : Polynomial ℝ := Polynomial.C 8 - Polynomial.C 3 * Polynomial.X
  let product : Polynomial ℝ := p1 * p2
  let a : ℝ := - (product.coeff 3)
  let b : ℝ := (product.coeff 2)
  let c : ℝ := - (product.coeff 1)
  let d : ℝ := (product.coeff 0)
  8 * a + 4 * b + 2 * c + d = 18 := sorry

end polynomial_coeff_sum_l27_27120


namespace mrs_doe_inheritance_l27_27919

noncomputable def calculateInheritance (totalTaxes : ℝ) : ℝ :=
  totalTaxes / 0.3625

theorem mrs_doe_inheritance (h : 0.3625 * calculateInheritance 15000 = 15000) :
  calculateInheritance 15000 = 41379 :=
by
  unfold calculateInheritance
  field_simp
  norm_cast
  sorry

end mrs_doe_inheritance_l27_27919


namespace probability_of_double_tile_is_one_fourth_l27_27402

noncomputable def probability_double_tile : ℚ :=
  let total_pairs := (7 * 7) / 2
  let double_pairs := 7
  double_pairs / total_pairs

theorem probability_of_double_tile_is_one_fourth :
  probability_double_tile = 1 / 4 :=
by
  sorry

end probability_of_double_tile_is_one_fourth_l27_27402


namespace ratio_x_to_y_l27_27204

theorem ratio_x_to_y (x y : ℤ) (h : (10*x - 3*y) / (13*x - 2*y) = 3 / 5) : x / y = 9 / 11 := 
by sorry

end ratio_x_to_y_l27_27204


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27281

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27281


namespace min_value_f_prime_at_2_l27_27289

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1/a) * x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 4*a*x + (1/a)

theorem min_value_f_prime_at_2 (a : ℝ) (h : a > 0) : 
  f_prime a 2 >= 12 + 4 * Real.sqrt 2 := 
by
  -- proof will be written here
  sorry

end min_value_f_prime_at_2_l27_27289


namespace distance_ratio_l27_27228

variables (dw dr : ℝ)

theorem distance_ratio (h1 : 4 * (dw / 4) + 8 * (dr / 8) = 8)
  (h2 : dw + dr = 8)
  (h3 : (dw / 4) + (dr / 8) = 1.5) :
  dw / dr = 1 :=
by
  sorry

end distance_ratio_l27_27228


namespace diameter_large_circle_correct_l27_27510

noncomputable def diameter_of_large_circle : ℝ :=
  2 * (Real.sqrt 17 + 4)

theorem diameter_large_circle_correct :
  ∃ (d : ℝ), (∀ (r : ℝ), r = Real.sqrt 17 + 4 → d = 2 * r) ∧ d = diameter_of_large_circle := by
    sorry

end diameter_large_circle_correct_l27_27510


namespace operation_result_l27_27785

-- Define x and the operations
def x : ℕ := 40

-- Define the operation sequence
def operation (y : ℕ) : ℕ :=
  let step1 := y / 4
  let step2 := step1 * 5
  let step3 := step2 + 10
  let step4 := step3 - 12
  step4

-- The statement we need to prove
theorem operation_result : operation x = 48 := by
  sorry

end operation_result_l27_27785


namespace min_people_for_no_empty_triplet_60_l27_27372

noncomputable def min_people_for_no_empty_triplet (total_chairs : ℕ) : ℕ :=
  if h : total_chairs % 3 = 0 then total_chairs / 3 else sorry

theorem min_people_for_no_empty_triplet_60 :
  min_people_for_no_empty_triplet 60 = 20 :=
by
  sorry

end min_people_for_no_empty_triplet_60_l27_27372


namespace algebraic_expression_value_l27_27751

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) : -2 * x + 4 * y^2 + 1 = -1 :=
by
  sorry

end algebraic_expression_value_l27_27751


namespace all_real_K_have_real_roots_l27_27243

noncomputable def quadratic_discriminant (K : ℝ) : ℝ :=
  let a := K ^ 3
  let b := -(4 * K ^ 3 + 1)
  let c := 3 * K ^ 3
  b ^ 2 - 4 * a * c

theorem all_real_K_have_real_roots : ∀ K : ℝ, quadratic_discriminant K ≥ 0 :=
by
  sorry

end all_real_K_have_real_roots_l27_27243


namespace number_of_triangles_l27_27376

-- Definition of given conditions
def original_wire_length : ℝ := 84
def remaining_wire_length : ℝ := 12
def wire_per_triangle : ℝ := 3

-- The goal is to prove that the number of triangles that can be made is 24
theorem number_of_triangles : (original_wire_length - remaining_wire_length) / wire_per_triangle = 24 := by
  sorry

end number_of_triangles_l27_27376


namespace exists_special_cubic_polynomial_l27_27488

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end exists_special_cubic_polynomial_l27_27488


namespace num_unique_seven_digit_integers_l27_27299

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def unique_seven_digit_integers : ℕ :=
  factorial 7 / (factorial 2 * factorial 2 * factorial 2)

theorem num_unique_seven_digit_integers : unique_seven_digit_integers = 630 := by
  sorry

end num_unique_seven_digit_integers_l27_27299


namespace find_radius_of_semicircle_l27_27070

-- Definitions for the rectangle and semi-circle
variable (L W : ℝ) -- Length and width of the rectangle
variable (r : ℝ) -- Radius of the semi-circle

-- Conditions given in the problem
def rectangle_perimeter : Prop := 2 * L + 2 * W = 216
def semicircle_diameter_eq_length : Prop := L = 2 * r 
def width_eq_twice_radius : Prop := W = 2 * r

-- Proof statement
theorem find_radius_of_semicircle
  (h_perimeter : rectangle_perimeter L W)
  (h_diameter : semicircle_diameter_eq_length L r)
  (h_width : width_eq_twice_radius W r) :
  r = 27 := by
  sorry

end find_radius_of_semicircle_l27_27070


namespace parabola_equation_l27_27521

theorem parabola_equation (p : ℝ) (h1 : 0 < p) (h2 : p / 2 = 2) : ∀ y x : ℝ, y^2 = -8 * x :=
by
  sorry

end parabola_equation_l27_27521


namespace exists_special_cubic_polynomial_l27_27487

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end exists_special_cubic_polynomial_l27_27487


namespace inequality_proving_l27_27321

theorem inequality_proving (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x^2 + y^2 + z^2 = 1) :
  (1 / x + 1 / y + 1 / z) - (x + y + z) ≥ 2 * Real.sqrt 3 :=
by
  sorry

end inequality_proving_l27_27321


namespace minimum_value_of_expression_l27_27322

theorem minimum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (sum_eq : x + y + z = 5) :
  (9 / x + 4 / y + 25 / z) ≥ 20 :=
sorry

end minimum_value_of_expression_l27_27322


namespace number_of_solutions_l27_27524

-- Given conditions
def positiveIntSolution (x y : ℤ) : Prop := x > 0 ∧ y > 0 ∧ 4 * x + 7 * y = 2001

-- Theorem statement
theorem number_of_solutions : ∃ (count : ℕ), 
  count = 71 ∧ ∃ f : Fin count → ℤ × ℤ,
    (∀ i, positiveIntSolution (f i).1 (f i).2) :=
by
  sorry

end number_of_solutions_l27_27524


namespace umbrella_cost_l27_27621

theorem umbrella_cost (house_umbrellas car_umbrellas cost_per_umbrella : ℕ) (h1 : house_umbrellas = 2) (h2 : car_umbrellas = 1) (h3 : cost_per_umbrella = 8) : 
  (house_umbrellas + car_umbrellas) * cost_per_umbrella = 24 := 
by
  sorry

end umbrella_cost_l27_27621


namespace intersection_A_B_l27_27284

-- Define the sets A and B
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

-- Prove that A ∩ B = {0, 1}
theorem intersection_A_B :
  A ∩ B = {0, 1} :=
by
  -- Proof goes here
  sorry

end intersection_A_B_l27_27284


namespace number_of_right_triangles_with_hypotenuse_is_12_l27_27300

theorem number_of_right_triangles_with_hypotenuse_is_12 : 
  ∃ (n : ℕ), n = 12 ∧ 
    (∀ (a b : ℕ), 
     (b < 150) →
     (a^2 + b^2 = (b + 2)^2) →
     ∃ (k : ℕ), a = 2 * k ∧ k^2 = b + 1) := 
  sorry

end number_of_right_triangles_with_hypotenuse_is_12_l27_27300


namespace rain_third_day_l27_27031

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end rain_third_day_l27_27031


namespace choose_9_4_l27_27156

theorem choose_9_4 : (Nat.choose 9 4) = 126 :=
  by
  sorry

end choose_9_4_l27_27156


namespace quadratic_root_l27_27458

theorem quadratic_root (m : ℝ) (h : m^2 + 2 * m - 1 = 0) : 2 * m^2 + 4 * m = 2 := by
  sorry

end quadratic_root_l27_27458


namespace least_side_is_8_l27_27886

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l27_27886


namespace smallest_four_digit_divisible_by_53_l27_27988

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27988


namespace f_injective_on_restricted_domain_l27_27793

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- Define the restricted domain
def f_restricted (x : ℝ) (h : -2 <= x) : ℝ := f x

-- The main statement to be proved
theorem f_injective_on_restricted_domain : 
  (∀ x1 x2 : {x // -2 <= x}, f_restricted x1.val x1.property = f_restricted x2.val x2.property → x1 = x2) := 
sorry

end f_injective_on_restricted_domain_l27_27793


namespace subtraction_of_decimals_l27_27357

theorem subtraction_of_decimals :
  888.8888 - 444.4444 = 444.4444 := 
sorry

end subtraction_of_decimals_l27_27357


namespace two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l27_27378

def star (a b : ℤ) : ℤ := a ^ 2 - b + a * b

theorem two_star_neg_five_eq_neg_one : star 2 (-5) = -1 := by
  sorry

theorem neg_two_star_two_star_neg_three_eq_one : star (-2) (star 2 (-3)) = 1 := by
  sorry

end two_star_neg_five_eq_neg_one_neg_two_star_two_star_neg_three_eq_one_l27_27378


namespace simplify_expression_l27_27644

theorem simplify_expression (a : ℤ) :
  ((36 * a^9)^4 * (63 * a^9)^4) = a^4 :=
sorry

end simplify_expression_l27_27644


namespace arctan_tan_75_sub_2_tan_30_eq_l27_27421

theorem arctan_tan_75_sub_2_tan_30_eq :
  arctan (tan (75 * real.pi / 180) - 2 * tan (30 * real.pi / 180)) * 180 / real.pi = 75 :=
sorry

end arctan_tan_75_sub_2_tan_30_eq_l27_27421


namespace ArletteAge_l27_27504

/-- Define the ages of Omi, Kimiko, and Arlette -/
def OmiAge (K : ℕ) : ℕ := 2 * K
def KimikoAge : ℕ := 28   /- K = 28 -/
def averageAge (O K A : ℕ) : Prop := (O + K + A) / 3 = 35

/-- Prove Arlette's age given the conditions -/
theorem ArletteAge (A : ℕ) (h1 : A + OmiAge KimikoAge + KimikoAge = 3 * 35) : A = 21 := by
  /- Hypothesis h1 unpacks the third condition into equality involving O, K, and A -/
  sorry

end ArletteAge_l27_27504


namespace simplify_expr_l27_27348

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l27_27348


namespace number_of_fish_bought_each_year_l27_27201

-- Define the conditions
def initial_fish : ℕ := 2
def net_gain_each_year (x : ℕ) : ℕ := x - 1
def years : ℕ := 5
def final_fish : ℕ := 7

-- Define the problem statement as a Lean theorem
theorem number_of_fish_bought_each_year (x : ℕ) : 
  initial_fish + years * net_gain_each_year x = final_fish → x = 2 := 
sorry

end number_of_fish_bought_each_year_l27_27201


namespace systematic_sampling_probability_l27_27911

/-- Given a population of 1002 individuals, if we remove 2 randomly and then pick 50 out of the remaining 1000, then the probability of picking each individual is 50/1002. 
This is because the process involves two independent steps: not being removed initially and then being chosen in the sample of size 50. --/
theorem systematic_sampling_probability :
  let population_size := 1002
  let removal_count := 2
  let sample_size := 50
  ∀ p : ℕ, p = 50 / (1002 : ℚ) := sorry

end systematic_sampling_probability_l27_27911


namespace total_savings_l27_27631

def liam_oranges : ℕ := 40
def liam_price_per_two : ℝ := 2.50
def claire_oranges : ℕ := 30
def claire_price_per_one : ℝ := 1.20

theorem total_savings : (liam_oranges / 2 * liam_price_per_two) + (claire_oranges * claire_price_per_one) = 86 := by
  sorry

end total_savings_l27_27631


namespace evaluate_f_at_7_l27_27885

theorem evaluate_f_at_7 :
  (∃ f : ℕ → ℕ, (∀ x, f (2 * x + 1) = x ^ 2 - 2 * x) ∧ f 7 = 3) :=
by 
  sorry

end evaluate_f_at_7_l27_27885


namespace scale_down_multiplication_l27_27745

theorem scale_down_multiplication (h : 14.97 * 46 = 688.62) : 1.497 * 4.6 = 6.8862 :=
by
  -- here we assume the necessary steps to justify the statement.
  sorry

end scale_down_multiplication_l27_27745


namespace toes_on_bus_is_164_l27_27330

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l27_27330


namespace find_h_neg_one_l27_27082

theorem find_h_neg_one (h : ℝ → ℝ) (H : ∀ x, (x^7 - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) + 1) : 
  h (-1) = 1 := 
by 
  sorry

end find_h_neg_one_l27_27082


namespace find_m_of_quadratic_root_l27_27754

theorem find_m_of_quadratic_root
  (m : ℤ) 
  (h : ∃ x : ℤ, x^2 - (m+3)*x + m + 2 = 0 ∧ x = 81) : 
  m = 79 :=
by
  sorry

end find_m_of_quadratic_root_l27_27754


namespace tina_sells_more_than_katya_l27_27490

noncomputable def katya_rev : ℝ := 8 * 1.5
noncomputable def ricky_rev : ℝ := 9 * 2.0
noncomputable def combined_rev : ℝ := katya_rev + ricky_rev
noncomputable def tina_target : ℝ := 2 * combined_rev
noncomputable def tina_glasses : ℝ := tina_target / 3.0
noncomputable def difference_glasses : ℝ := tina_glasses - 8

theorem tina_sells_more_than_katya :
  difference_glasses = 12 := by
  sorry

end tina_sells_more_than_katya_l27_27490


namespace cistern_empty_time_l27_27842

theorem cistern_empty_time
  (fill_time_without_leak : ℝ := 4)
  (additional_time_due_to_leak : ℝ := 2) :
  (1 / (fill_time_without_leak + additional_time_due_to_leak - fill_time_without_leak / fill_time_without_leak)) = 12 :=
by
  sorry

end cistern_empty_time_l27_27842


namespace milk_production_l27_27603

theorem milk_production (y : ℕ) (hcows : y > 0) (hcans : y + 2 > 0) (hdays : y + 3 > 0) :
  let daily_production_per_cow := (y + 2 : ℕ) / (y * (y + 3) : ℕ)
  let total_daily_production := (y + 4 : ℕ) * daily_production_per_cow
  let required_days := (y + 6 : ℕ) / total_daily_production
  required_days = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) :=
by
  sorry

end milk_production_l27_27603


namespace taylor_scores_l27_27197

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end taylor_scores_l27_27197


namespace min_value_of_expression_l27_27463

theorem min_value_of_expression (m n : ℝ) (h1 : m + 2 * n = 2) (h2 : m > 0) (h3 : n > 0) : 
  (1 / (m + 1) + 1 / (2 * n)) ≥ 4 / 3 :=
sorry

end min_value_of_expression_l27_27463


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27260

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27260


namespace commute_times_absolute_difference_l27_27371

theorem commute_times_absolute_difference
  (x y : ℝ)
  (H_avg : (x + y + 10 + 11 + 9) / 5 = 10)
  (H_var : ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) / 5 = 2) :
  abs (x - y) = 4 :=
by
  -- proof steps are omitted
  sorry

end commute_times_absolute_difference_l27_27371


namespace product_remainder_mod_7_l27_27667

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l27_27667


namespace range_of_a_l27_27471

theorem range_of_a (a : ℝ) : ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l27_27471


namespace value_of_expression_l27_27776

theorem value_of_expression (x : ℕ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  simp
  rfl

end value_of_expression_l27_27776


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27258

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27258


namespace simplify_rationalize_l27_27351

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l27_27351


namespace smallest_four_digit_multiple_of_53_l27_27999

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27999


namespace smallest_angle_in_triangle_l27_27187

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end smallest_angle_in_triangle_l27_27187


namespace total_trolls_l27_27427

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l27_27427


namespace committee_meeting_people_l27_27214

theorem committee_meeting_people (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 11) : 
  A + B = 7 :=
sorry

end committee_meeting_people_l27_27214


namespace log2_a_div_b_squared_l27_27625

variable (a b : ℝ)
variable (ha_ne_1 : a ≠ 1) (hb_ne_1 : b ≠ 1)
variable (ha_pos : 0 < a) (hb_pos : 0 < b)
variable (h1 : 2 ^ (Real.log 32 / Real.log b) = a)
variable (h2 : a * b = 128)

theorem log2_a_div_b_squared :
  (Real.log ((a / b) : ℝ) / Real.log 2) ^ 2 = 29 + (49 / 4) :=
sorry

end log2_a_div_b_squared_l27_27625


namespace simplify_rationalize_l27_27353

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l27_27353


namespace rectangle_dimensions_l27_27216

theorem rectangle_dimensions (w l : ℕ) (h : l = w + 5) (hp : 2 * l + 2 * w = 34) : w = 6 ∧ l = 11 := 
by 
  sorry

end rectangle_dimensions_l27_27216


namespace expected_heads_three_coins_is_three_halves_l27_27375

noncomputable def expected_heads_three_coins : ℚ :=
  ((0 * (1 / 8)) + (1 * (3 / 8)) + (2 * (3 / 8)) + (3 * (1 / 8)))

theorem expected_heads_three_coins_is_three_halves :
  expected_heads_three_coins = 3 / 2 :=
by
  sorry

end expected_heads_three_coins_is_three_halves_l27_27375


namespace inequality_pqr_l27_27924

theorem inequality_pqr (p q r : ℝ) (n : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p * q * r = 1) :
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 :=
sorry

end inequality_pqr_l27_27924


namespace total_profit_correct_l27_27117

noncomputable def total_profit (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : ℝ := Tp

theorem total_profit_correct (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : 
  total_profit Cp Cq Cr Tp h1 h2 hR = 4650 :=
sorry

end total_profit_correct_l27_27117


namespace find_a_l27_27592

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, x - 2 * a * y - 3 = 0 ∧ x^2 + y^2 - 2 * x + 2 * y - 3 = 0) → a = 1 :=
by
  sorry

end find_a_l27_27592


namespace combine_octahedrons_tetrahedrons_to_larger_octahedron_l27_27169

theorem combine_octahedrons_tetrahedrons_to_larger_octahedron (edge : ℝ) :
  ∃ (octahedrons : ℕ) (tetrahedrons : ℕ),
    octahedrons = 6 ∧ tetrahedrons = 8 ∧
    (∃ (new_octahedron_edge : ℝ), new_octahedron_edge = 2 * edge) :=
by {
  -- The proof will construct the larger octahedron
  sorry
}

end combine_octahedrons_tetrahedrons_to_larger_octahedron_l27_27169


namespace yellow_balls_in_bag_l27_27124

theorem yellow_balls_in_bag (x : ℕ) (prob : 1 / (1 + x) = 1 / 4) :
  x = 3 :=
sorry

end yellow_balls_in_bag_l27_27124


namespace sequence_is_periodic_l27_27682

open Nat

def is_periodic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ p > 0, ∀ i, a (i + p) = a i

theorem sequence_is_periodic (a : ℕ → ℕ)
  (h1 : ∀ n, a n < 1988)
  (h2 : ∀ m n, a m + a n ∣ a (m + n)) : is_periodic_sequence a :=
by
  sorry

end sequence_is_periodic_l27_27682


namespace probability_no_two_adjacent_stand_l27_27011

-- Definitions and Conditions
def fair_coin := { outcome : Bool // outcome = true ∨ outcome = false }
def flip_coin := (flip : fin 10 → fair_coin)

/-- Predicate to check if two adjacent people stand given a table arrangement -/
def no_two_adjacent (arr : flip_coin) : Prop :=
  ∀ i : fin 10, ¬ (arr.flip i).outcome = true ∧ (arr.flip (i + 1) % 10).outcome = true

-- The problem statement
theorem probability_no_two_adjacent_stand :
  (∑ (arr : flip_coin) in finset.univ, if no_two_adjacent arr then 1 else 0) / 1024 = (123 : ℚ) / 1024 :=
begin
  sorry
end

end probability_no_two_adjacent_stand_l27_27011


namespace total_trolls_l27_27425

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l27_27425


namespace remainder_when_divided_l27_27224

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_l27_27224


namespace rectangular_box_proof_l27_27533

noncomputable def rectangular_box_surface_area
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) : ℝ :=
2 * (a * b + b * c + c * a)

theorem rectangular_box_proof
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) :
  rectangular_box_surface_area a b c h1 h2 = 784 :=
by
  sorry

end rectangular_box_proof_l27_27533


namespace percent_of_decimal_l27_27976

theorem percent_of_decimal : (3 / 8 / 100) * 240 = 0.9 :=
by
  sorry

end percent_of_decimal_l27_27976


namespace simplify_rationalize_l27_27354

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l27_27354


namespace least_third_side_of_right_triangle_l27_27897

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l27_27897


namespace weight_of_b_l27_27954

variable (A B C : ℕ)

theorem weight_of_b 
  (h1 : A + B + C = 180) 
  (h2 : A + B = 140) 
  (h3 : B + C = 100) :
  B = 60 :=
sorry

end weight_of_b_l27_27954


namespace no_integer_triple_exists_for_10_l27_27249

theorem no_integer_triple_exists_for_10 :
  ¬ ∃ (x y z : ℤ), 3 * x^2 + 4 * y^2 - 5 * z^2 = 10 :=
sorry

end no_integer_triple_exists_for_10_l27_27249


namespace difference_of_digits_l27_27821

theorem difference_of_digits (A B : ℕ) (h1 : 6 * 10 + A - (B * 10 + 2) = 36) (h2 : A ≠ B) : A - B = 5 :=
sorry

end difference_of_digits_l27_27821


namespace trig_log_exp_identity_l27_27053

theorem trig_log_exp_identity : 
  (Real.sin (330 * Real.pi / 180) + 
   (Real.sqrt 2 - 1)^0 + 
   3^(Real.log 2 / Real.log 3)) = 5 / 2 :=
by
  -- Proof omitted
  sorry

end trig_log_exp_identity_l27_27053


namespace total_toes_on_bus_l27_27334

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l27_27334


namespace M_intersection_N_equals_M_l27_27626

variable (x a : ℝ)

def M : Set ℝ := { y | ∃ x, y = x^2 + 1 }
def N : Set ℝ := { y | ∃ a, y = 2 * a^2 - 4 * a + 1 }

theorem M_intersection_N_equals_M : M ∩ N = M := by
  sorry

end M_intersection_N_equals_M_l27_27626


namespace time_per_harvest_is_three_months_l27_27173

variable (area : ℕ) (trees_per_m2 : ℕ) (coconuts_per_tree : ℕ) 
variable (price_per_coconut : ℚ) (total_earning_6_months : ℚ)

theorem time_per_harvest_is_three_months 
  (h1 : area = 20) 
  (h2 : trees_per_m2 = 2) 
  (h3 : coconuts_per_tree = 6) 
  (h4 : price_per_coconut = 0.50) 
  (h5 : total_earning_6_months = 240) :
    (6 / (total_earning_6_months / (area * trees_per_m2 * coconuts_per_tree * price_per_coconut)) = 3) := 
  by 
    sorry

end time_per_harvest_is_three_months_l27_27173


namespace range_of_m_l27_27778

noncomputable def equation_has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, 25^(-|x+1|) - 4 * 5^(-|x+1|) - m = 0

theorem range_of_m : ∀ m : ℝ, equation_has_real_roots m ↔ (-3 ≤ m ∧ m < 0) :=
by
  -- Proof omitted
  sorry

end range_of_m_l27_27778


namespace no_polyhedron_with_area_ratio_ge_two_l27_27153

theorem no_polyhedron_with_area_ratio_ge_two (n : ℕ) (areas : Fin n → ℝ)
  (h : ∀ (i j : Fin n), i < j → (areas j) / (areas i) ≥ 2) : False := by
  sorry

end no_polyhedron_with_area_ratio_ge_two_l27_27153


namespace parabola_focus_l27_27015

theorem parabola_focus (x y p : ℝ) (h_eq : y = 2 * x^2) (h_standard_form : x^2 = (1 / 2) * y) (h_p : p = 1 / 4) : 
    (0, p / 2) = (0, 1 / 8) := by
    sorry

end parabola_focus_l27_27015


namespace functional_relationship_max_annual_profit_l27_27712

namespace FactoryProfit

-- Definitions of conditions
def fixed_annual_investment : ℕ := 100
def unit_investment : ℕ := 1
def sales_revenue (x : ℕ) : ℕ :=
  if x > 20 then 260 
  else 33 * x - x^2

def annual_profit (x : ℕ) : ℤ :=
  let revenue := sales_revenue x
  let total_investment := fixed_annual_investment + x
  revenue - total_investment

-- Statements to prove
theorem functional_relationship (x : ℕ) (hx : x > 0) :
  annual_profit x =
  if x ≤ 20 then
    (-x^2 : ℤ) + 32 * x - 100
  else
    160 - x :=
by sorry

theorem max_annual_profit : 
  ∃ x, annual_profit x = 144 ∧
  ∀ y, annual_profit y ≤ 144 :=
by sorry

end FactoryProfit

end functional_relationship_max_annual_profit_l27_27712


namespace sock_problem_l27_27601

def sock_pair_count (total_socks : Nat) (socks_distribution : List (String × Nat)) (target_color : String) (different_color : String) : Nat :=
  if target_color = different_color then 0
  else match socks_distribution with
    | [] => 0
    | (color, count) :: tail =>
        if color = target_color then count * socks_distribution.foldl (λ acc (col_count : String × Nat) =>
          if col_count.fst ≠ target_color then acc + col_count.snd else acc) 0
        else sock_pair_count total_socks tail target_color different_color

theorem sock_problem : sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "white" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "brown" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "blue" =
                        48 :=
by sorry

end sock_problem_l27_27601


namespace triangle_inequality_equality_condition_l27_27167

variables {A B C a b c : ℝ}

theorem triangle_inequality (A a B b C c : ℝ) :
  A * a + B * b + C * c ≥ 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b) :=
sorry

theorem equality_condition (A B C a b c : ℝ) :
  (A * a + B * b + C * c = 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b)) ↔ (a = b ∧ b = c ∧ A = B ∧ B = C) :=
sorry

end triangle_inequality_equality_condition_l27_27167


namespace guo_can_pay_exactly_l27_27880

theorem guo_can_pay_exactly (
  x y z : ℕ
) (h : 10 * x + 20 * y + 50 * z = 20000) : ∃ a b c : ℕ, a + 2 * b + 5 * c = 1000 :=
sorry

end guo_can_pay_exactly_l27_27880


namespace evaluate_exponential_operations_l27_27210

theorem evaluate_exponential_operations (a : ℝ) :
  (2 * a^2 - a^2 ≠ 2) ∧
  (a^2 * a^4 = a^6) ∧
  ((a^2)^3 ≠ a^5) ∧
  (a^6 / a^2 ≠ a^3) := by
  sorry

end evaluate_exponential_operations_l27_27210


namespace angle_CBD_is_48_degrees_l27_27483

theorem angle_CBD_is_48_degrees :
  ∀ (A B D C : Type) (α β γ δ : ℝ), 
    α = 28 ∧ β = 46 ∧ C ∈ [B, D] ∧ γ = 30 → 
    δ = 48 := 
by 
  sorry

end angle_CBD_is_48_degrees_l27_27483


namespace value_of_a_is_minus_one_l27_27920

-- Define the imaginary unit i
def imaginary_unit_i : Complex := Complex.I

-- Define the complex number condition
def complex_number_condition (a : ℝ) : Prop :=
  let z := (a + imaginary_unit_i) / (1 + imaginary_unit_i)
  (Complex.re z) = 0 ∧ (Complex.im z) ≠ 0

-- Prove that the value of the real number a is -1 given the condition
theorem value_of_a_is_minus_one (a : ℝ) (h : complex_number_condition a) : a = -1 :=
sorry

end value_of_a_is_minus_one_l27_27920


namespace total_saltwater_animals_l27_27202

variable (numSaltwaterAquariums : Nat)
variable (animalsPerAquarium : Nat)

theorem total_saltwater_animals (h1 : numSaltwaterAquariums = 22) (h2 : animalsPerAquarium = 46) : 
    numSaltwaterAquariums * animalsPerAquarium = 1012 := 
  by
    sorry

end total_saltwater_animals_l27_27202


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27262

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27262


namespace range_of_x_of_sqrt_x_plus_3_l27_27145

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l27_27145


namespace min_value_f_l27_27868

noncomputable def f (x : ℝ) : ℝ := x^3 + 9 * x + 81 / x^4

theorem min_value_f : ∃ x > 0, f x = 21 ∧ ∀ y > 0, f y ≥ 21 := by
  sorry

end min_value_f_l27_27868


namespace error_difference_l27_27399

noncomputable def total_income_without_error (T: ℝ) : ℝ :=
  T + 110000

noncomputable def total_income_with_error (T: ℝ) : ℝ :=
  T + 1100000

noncomputable def mean_without_error (T: ℝ) : ℝ :=
  (T + 110000) / 500

noncomputable def mean_with_error (T: ℝ) : ℝ :=
  (T + 1100000) / 500

theorem error_difference (T: ℝ) :
  mean_with_error T - mean_without_error T = 1980 :=
by
  sorry

end error_difference_l27_27399


namespace ellipse_with_foci_on_x_axis_l27_27119

theorem ellipse_with_foci_on_x_axis {a : ℝ} (h1 : a - 5 > 0) (h2 : 2 > 0) (h3 : a - 5 > 2) :
  a > 7 :=
by
  sorry

end ellipse_with_foci_on_x_axis_l27_27119


namespace simplify_expression_solve_fractional_eq_l27_27051

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l27_27051


namespace no_solution_a_squared_plus_b_squared_eq_2023_l27_27809

theorem no_solution_a_squared_plus_b_squared_eq_2023 :
  ∀ (a b : ℤ), a^2 + b^2 ≠ 2023 := 
by
  sorry

end no_solution_a_squared_plus_b_squared_eq_2023_l27_27809


namespace transformed_roots_l27_27649

theorem transformed_roots (b c : ℝ) (h₁ : (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c).roots = {2, -3}) :
  (Polynomial.C 1 * (Polynomial.X - Polynomial.C 4)^2 + Polynomial.C b * (Polynomial.X - Polynomial.C 4) + Polynomial.C c).roots = {1, 6} :=
by
  sorry

end transformed_roots_l27_27649


namespace monkey_slip_distance_l27_27065

theorem monkey_slip_distance
  (height : ℕ)
  (climb_per_hour : ℕ)
  (hours : ℕ)
  (s : ℕ)
  (total_hours : ℕ)
  (final_climb : ℕ)
  (reach_top : height = hours * (climb_per_hour - s) + final_climb)
  (total_hours_constraint : total_hours = 17)
  (climb_per_hour_constraint : climb_per_hour = 3)
  (height_constraint : height = 19)
  (final_climb_constraint : final_climb = 3)
  (hours_constraint : hours = 16) :
  s = 2 := sorry

end monkey_slip_distance_l27_27065


namespace part_I_part_II_l27_27875

section Problem

def point_F1 : (ℝ × ℝ) := (-1, 0)
def point_F2 : (ℝ × ℝ) := (1, 0)
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def ellipse_C (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem part_I : 
  (∀ (M : ℝ × ℝ), ∃ P : ℝ × ℝ, 
    circle_F1 P.1 P.2 ∧ 
    (|M.1 - point_F1.1| + |M.2 - point_F1.2|) + (|M.1 - point_F2.1| + |M.2 - point_F2.2|) = 4 
    → ellipse_C M.1 M.2) := sorry

theorem part_II : 
  (∃ l : ℝ × ℝ → Prop, ∀ B1 B2 : ℝ × ℝ, 
    l B1 ∧ l B2 ∧ ellipse_C B1.1 B1.2 ∧ ellipse_C B2.1 B2.2 ∧ (∃ A1 A2 : ℝ × ℝ, parabola A1.1 A1.2 ∧ parabola A2.1 A2.2 ∧ |A1.1 - A2.1| = |A1.2 - A2.2| = 0) 
    → |A1.1 - A2.1| = 64/9) := sorry

end Problem

end part_I_part_II_l27_27875


namespace triangle_smallest_angle_l27_27190

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end triangle_smallest_angle_l27_27190


namespace quadratic_vertex_on_x_axis_l27_27904

theorem quadratic_vertex_on_x_axis (k : ℝ) :
  (∃ x : ℝ, (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end quadratic_vertex_on_x_axis_l27_27904


namespace positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l27_27738

theorem positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5 : 
  ∃ (x : ℕ), (x = 594) ∧ (18 ∣ x) ∧ (24 ≤ Real.sqrt (x) ∧ Real.sqrt (x) ≤ 24.5) := 
by 
  sorry

end positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l27_27738


namespace right_triangle_least_side_l27_27901

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l27_27901


namespace least_third_side_length_l27_27890

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l27_27890


namespace find_number_that_satisfies_congruences_l27_27066

theorem find_number_that_satisfies_congruences :
  ∃ m : ℕ, 
  (m % 13 = 12) ∧ 
  (m % 11 = 10) ∧ 
  (m % 7 = 6) ∧ 
  (m % 5 = 4) ∧ 
  (m % 3 = 2) ∧ 
  m = 15014 :=
by
  sorry

end find_number_that_satisfies_congruences_l27_27066


namespace difference_in_gems_l27_27236

theorem difference_in_gems (r d : ℕ) (h : d = 3 * r) : d - r = 2 * r := 
by 
  sorry

end difference_in_gems_l27_27236


namespace fencing_required_l27_27718

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (hL : L = 20) (hA : A = 80) (hW : A = L * W) :
  (L + 2 * W) = 28 :=
by {
  sorry
}

end fencing_required_l27_27718


namespace scientific_notation_of_86000000_l27_27523

theorem scientific_notation_of_86000000 :
  ∃ (x : ℝ) (y : ℤ), 86000000 = x * 10^y ∧ x = 8.6 ∧ y = 7 :=
by
  use 8.6
  use 7
  sorry

end scientific_notation_of_86000000_l27_27523


namespace possible_integer_roots_l27_27854

theorem possible_integer_roots (x : ℤ) :
  x^3 + 3 * x^2 - 4 * x - 13 = 0 →
  x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13 :=
by sorry

end possible_integer_roots_l27_27854


namespace correct_option_is_B_l27_27840

noncomputable def smallest_absolute_value := 0

theorem correct_option_is_B :
  (∀ x : ℝ, |x| ≥ 0) ∧ |(0 : ℝ)| = 0 :=
by
  sorry

end correct_option_is_B_l27_27840


namespace minimum_value_of_x_y_l27_27771

noncomputable def minimum_value (x y : ℝ) : ℝ :=
  x + y

theorem minimum_value_of_x_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 - x) * (-y) = x) : minimum_value x y = 4 :=
  sorry

end minimum_value_of_x_y_l27_27771


namespace find_f_of_2_l27_27587

theorem find_f_of_2 (f g : ℝ → ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) 
                    (h₂ : ∀ x : ℝ, g x = f x + 9) (h₃ : g (-2) = 3) :
                    f 2 = 6 :=
by
  sorry

end find_f_of_2_l27_27587


namespace find_RS_length_l27_27473

-- Define the conditions and the problem in Lean

theorem find_RS_length
  (radius : ℝ)
  (P Q R S T : ℝ)
  (center_to_T : ℝ)
  (PT : ℝ)
  (PQ : ℝ)
  (RT TS : ℝ)
  (h_radius : radius = 7)
  (h_center_to_T : center_to_T = 3)
  (h_PT : PT = 8)
  (h_bisect_PQ : PQ = 2 * PT)
  (h_intersecting_chords : PT * (PQ / 2) = RT * TS)
  (h_perfect_square : ∃ k : ℝ, k^2 = RT * TS) :
  RS = 16 :=
by
  sorry

end find_RS_length_l27_27473


namespace num_apartments_per_floor_l27_27222

-- Definitions used in the proof
def num_buildings : ℕ := 2
def floors_per_building : ℕ := 12
def doors_per_apartment : ℕ := 7
def total_doors_needed : ℕ := 1008

-- Lean statement to proof the number of apartments per floor
theorem num_apartments_per_floor : 
  (total_doors_needed / (doors_per_apartment * num_buildings * floors_per_building)) = 6 :=
by
  sorry

end num_apartments_per_floor_l27_27222


namespace find_n_l27_27036

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 101) (h3 : 100 * n % 101 = 72) : n = 29 := 
by
  sorry

end find_n_l27_27036


namespace taylor_scores_l27_27196

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end taylor_scores_l27_27196


namespace line_through_two_points_l27_27959

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ x y : ℝ, (x, y) = (-2, 4) ∨ (x, y) = (-1, 3) → y = m * x + b) ∧ b = 2 ∧ m = -1 :=
by
  sorry

end line_through_two_points_l27_27959


namespace problem_solution_l27_27977

theorem problem_solution :
  (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := 
  by 
  sorry

end problem_solution_l27_27977


namespace toby_initial_photos_l27_27200

-- Defining the problem conditions and proving the initial number of photos Toby had.
theorem toby_initial_photos (X : ℕ) 
  (h1 : ∃ n, X = n - 7) 
  (h2 : ∃ m, m = (n - 7) + 15) 
  (h3 : ∃ k, k = m) 
  (h4 : (k - 3) = 84) 
  : X = 79 :=
sorry

end toby_initial_photos_l27_27200


namespace at_least_one_inequality_holds_l27_27170

theorem at_least_one_inequality_holds
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l27_27170


namespace constant_term_expansion_l27_27978

theorem constant_term_expansion :
  (∃ k : ℕ, k ∈ finset.range 13 ∧ 
            (∃ (term : ℚ), term = binom 12 k * (x ^ (k / 3) * (4 / x ^ 2) ^ (12 - k))) ∧
            is_constant term) →
  term.coeff = 126720 :=
by
  sorry

end constant_term_expansion_l27_27978


namespace common_ratio_geometric_sequence_l27_27105

variables {a : ℕ → ℝ} -- 'a' is a sequence of positive real numbers
variable {q : ℝ} -- 'q' is the common ratio of the geometric sequence

-- Definition of a geometric sequence with common ratio 'q'
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Condition from the problem statement
def condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  2 * a 5 - 3 * a 4 = 2 * a 3

-- Main theorem: If the sequence {a_n} is a geometric sequence with positive terms and satisfies the condition, 
-- then the common ratio q = 2
theorem common_ratio_geometric_sequence :
  (∀ n, 0 < a n) → geometric_sequence a q → condition a q → q = 2 :=
by
  intro h_pos h_geom h_cond
  sorry

end common_ratio_geometric_sequence_l27_27105


namespace additional_machines_l27_27095

theorem additional_machines (r : ℝ) (M : ℝ) : 
  (5 * r * 20 = 1) ∧ (M * r * 10 = 1) → (M - 5 = 95) :=
by
  sorry

end additional_machines_l27_27095


namespace smallest_cube_volume_l27_27001

noncomputable def sculpture_height : ℝ := 15
noncomputable def sculpture_base_radius : ℝ := 8
noncomputable def cube_side_length : ℝ := 16

theorem smallest_cube_volume :
  ∀ (h r s : ℝ), 
    h = sculpture_height ∧
    r = sculpture_base_radius ∧
    s = cube_side_length →
    s ^ 3 = 4096 :=
by
  intros h r s 
  intro h_def
  sorry

end smallest_cube_volume_l27_27001


namespace line_through_points_l27_27650

theorem line_through_points (x1 y1 x2 y2 : ℕ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (3, 8)) : 
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = 2 := 
by
  sorry

end line_through_points_l27_27650


namespace c_sum_formula_l27_27755

noncomputable section

def arithmetic_sequence (a : Nat -> ℚ) : Prop :=
  a 3 = 2 ∧ (a 1 + 2 * ((a 2 - a 1) : ℚ)) = 2

def geometric_sequence (b : Nat -> ℚ) (a : Nat -> ℚ) : Prop :=
  b 1 = a 1 ∧ b 4 = a 15

def c_sequence (a : Nat -> ℚ) (b : Nat -> ℚ) (n : Nat) : ℚ :=
  a n + b n

def Tn (c : Nat -> ℚ) (n : Nat) : ℚ :=
  (Finset.range n).sum c

theorem c_sum_formula
  (a b c : Nat -> ℚ)
  (k : Nat) 
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b a)
  (hc : ∀ n, c n = c_sequence a b n) :
  Tn c k = k * (k + 3) / 4 + 2^k - 1 :=
by
  sorry

end c_sum_formula_l27_27755


namespace emily_sixth_quiz_score_l27_27737

theorem emily_sixth_quiz_score (q1 q2 q3 q4 q5 target_mean : ℕ) (required_sum : ℕ) (current_sum : ℕ) (s6 : ℕ)
  (h1 : q1 = 94) (h2 : q2 = 97) (h3 : q3 = 88) (h4 : q4 = 91) (h5 : q5 = 102) (h_target_mean : target_mean = 95)
  (h_required_sum : required_sum = 6 * target_mean) (h_current_sum : current_sum = q1 + q2 + q3 + q4 + q5)
  (h6 : s6 = required_sum - current_sum) :
  s6 = 98 :=
by
  sorry

end emily_sixth_quiz_score_l27_27737


namespace probability_product_multiple_of_10_l27_27122

open Finset

def S : Finset ℕ := {2, 3, 5, 6, 9}

theorem probability_product_multiple_of_10 :
  (S.choose 3).filter (λ t => 10 ∣ t.prod id).card = 3 →
  (S.choose 3).card = 10 →
  ((S.choose 3).filter (λ t => 10 ∣ t.prod id).card / (S.choose 3).card : ℚ) = 3 / 10 := by
  intro h_favorable h_total
  rw [div_eq_mul_inv, int.coe_nat_div, h_favorable, h_total]
  norm_num
  sorry

end probability_product_multiple_of_10_l27_27122


namespace smallest_four_digit_divisible_by_53_l27_27984

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27984


namespace invertible_2x2_matrix_l27_27440

open Matrix

def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℚ

def A : matrix_2x2 := ![![4, 5], ![-2, 9]]

def inv_A : matrix_2x2 := ![![9/46, -5/46], ![1/23, 2/23]]

theorem invertible_2x2_matrix :
  det A ≠ 0 → (inv A = inv_A) := 
by
  sorry

end invertible_2x2_matrix_l27_27440


namespace total_people_present_l27_27685

def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698
def total_people : ℕ := number_of_parents + number_of_pupils

theorem total_people_present : total_people = 803 :=
by
  sorry

end total_people_present_l27_27685


namespace sqrt_domain_l27_27131

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l27_27131


namespace remainder_of_N_mod_37_l27_27225

theorem remainder_of_N_mod_37 (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_of_N_mod_37_l27_27225


namespace multiple_of_six_as_four_cubes_integer_as_five_cubes_l27_27047

-- Part (a)
theorem multiple_of_six_as_four_cubes (n : ℤ) : ∃ a b c d : ℤ, 6 * n = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 :=
by
  sorry

-- Part (b)
theorem integer_as_five_cubes (k : ℤ) : ∃ a b c d e : ℤ, k = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 + e ^ 3 :=
by
  have h := multiple_of_six_as_four_cubes
  sorry

end multiple_of_six_as_four_cubes_integer_as_five_cubes_l27_27047


namespace dandelion_average_l27_27414

theorem dandelion_average :
  let Billy_initial := 36
  let George_initial := Billy_initial / 3
  let Billy_total := Billy_initial + 10
  let George_total := George_initial + 10
  let total := Billy_total + George_total
  let average := total / 2
  average = 34 :=
by
  -- placeholder for the proof
  sorry

end dandelion_average_l27_27414


namespace proof_main_proof_l27_27705

noncomputable def main_proof : Prop :=
  2 * Real.logb 5 10 + Real.logb 5 0.25 = 2

theorem proof_main_proof : main_proof :=
  by
    sorry

end proof_main_proof_l27_27705


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27278

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27278


namespace overtime_hourly_rate_l27_27076

theorem overtime_hourly_rate
  (hourly_rate_first_40_hours: ℝ)
  (hours_first_40: ℝ)
  (gross_pay: ℝ)
  (overtime_hours: ℝ)
  (total_pay_first_40: ℝ := hours_first_40 * hourly_rate_first_40_hours)
  (pay_overtime: ℝ := gross_pay - total_pay_first_40)
  (hourly_rate_overtime: ℝ := pay_overtime / overtime_hours)
  (h1: hourly_rate_first_40_hours = 11.25)
  (h2: hours_first_40 = 40)
  (h3: gross_pay = 622)
  (h4: overtime_hours = 10.75) :
  hourly_rate_overtime = 16 := 
by
  sorry

end overtime_hourly_rate_l27_27076


namespace jill_tax_on_other_items_l27_27935

noncomputable def tax_on_other_items (total_spent clothing_tax_percent total_tax_percent : ℝ) : ℝ :=
  let clothing_spent := 0.5 * total_spent
  let food_spent := 0.25 * total_spent
  let other_spent := 0.25 * total_spent
  let clothing_tax := clothing_tax_percent * clothing_spent
  let total_tax := total_tax_percent * total_spent
  let tax_on_others := total_tax - clothing_tax
  (tax_on_others / other_spent) * 100

theorem jill_tax_on_other_items :
  let total_spent := 100
  let clothing_tax_percent := 0.1
  let total_tax_percent := 0.1
  tax_on_other_items total_spent clothing_tax_percent total_tax_percent = 20 := by
  sorry

end jill_tax_on_other_items_l27_27935


namespace t_shirt_sale_revenue_per_minute_l27_27515

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end t_shirt_sale_revenue_per_minute_l27_27515


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27277

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27277


namespace BD_equals_regular_10_gon_side_length_l27_27255

-- Define the points and lengths
variable {A B C D E : ℝ}

-- Define the properties and conditions
axiom h1 : Dist A D = Dist D E
axiom h2 : Dist A D = Dist A C 
axiom h3 : Dist B D = Dist A E
axiom h4 : Parallel (Segment D E) (Segment B C)

-- Main theorem statement
theorem BD_equals_regular_10_gon_side_length (A B C D E : ℝ) 
  (h1 : Dist A D = Dist D E)
  (h2 : Dist A D = Dist A C)
  (h3 : Dist B D = Dist A E)
  (h4 : Parallel (Segment D E) (Segment B C)) :
  Dist B D = 2 * Dist A C * sin (π / 10) := sorry

end BD_equals_regular_10_gon_side_length_l27_27255


namespace point_on_x_axis_coord_l27_27480

theorem point_on_x_axis_coord (m : ℝ) (h : (m - 1, 2 * m).snd = 0) : (m - 1, 2 * m) = (-1, 0) :=
by
  sorry

end point_on_x_axis_coord_l27_27480


namespace t_shirt_jersey_price_difference_l27_27012

theorem t_shirt_jersey_price_difference :
  ∀ (T J : ℝ), (0.9 * T = 192) → (0.9 * J = 34) → (T - J = 175.55) :=
by
  intros T J hT hJ
  sorry

end t_shirt_jersey_price_difference_l27_27012


namespace arithmetic_sequence_sum_l27_27086

theorem arithmetic_sequence_sum :
  ∃ a b : ℕ, ∀ d : ℕ,
    d = 5 →
    a = 28 →
    b = 33 →
    a + b = 61 :=
by
  sorry

end arithmetic_sequence_sum_l27_27086


namespace sqrt_domain_l27_27129

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l27_27129


namespace sum_of_fifth_powers_l27_27274

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l27_27274


namespace distance_from_Idaho_to_Nevada_l27_27565

theorem distance_from_Idaho_to_Nevada (d1 d2 s1 s2 t total_time : ℝ) 
  (h1 : d1 = 640)
  (h2 : s1 = 80)
  (h3 : s2 = 50)
  (h4 : total_time = 19)
  (h5 : t = total_time - (d1 / s1)) :
  d2 = s2 * t :=
by
  sorry

end distance_from_Idaho_to_Nevada_l27_27565


namespace marissas_sunflower_height_l27_27931

def height_of_marissas_sunflower (sister_height_feet : ℤ) (sister_height_inches : ℤ) (additional_inches : ℤ) : Prop :=
  (sister_height_feet = 4) →
  (sister_height_inches = 3) →
  (additional_inches = 21) →
  sister_height_feet * 12 + sister_height_inches + additional_inches = 72

-- Prove that Marissa's sunflower height in feet is 6
theorem marissas_sunflower_height :
  height_of_marissas_sunflower 4 3 21 →
  72 / 12 = 6 :=
by
  assume h,
  rw Nat.div_eq_of_eq_mul_left sorry,
  sorry

end marissas_sunflower_height_l27_27931


namespace walter_fraction_fewer_bananas_l27_27314

theorem walter_fraction_fewer_bananas (f : ℚ) (h1 : 56 + (56 - 56 * f) = 98) : f = 1 / 4 :=
sorry

end walter_fraction_fewer_bananas_l27_27314


namespace smallest_four_digit_divisible_by_53_l27_27985

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27985


namespace find_g_53_l27_27651

variable (g : ℝ → ℝ)

axiom functional_eq (x y : ℝ) : g (x * y) = y * g x
axiom g_one : g 1 = 10

theorem find_g_53 : g 53 = 530 :=
by
  sorry

end find_g_53_l27_27651


namespace peter_pizza_total_l27_27339

theorem peter_pizza_total (total_slices : ℕ) (whole_slice : ℕ) (shared_slice : ℚ) (shared_parts : ℕ) :
  total_slices = 16 ∧ whole_slice = 1 ∧ shared_parts = 3 ∧ shared_slice = 1 / (total_slices * shared_parts) →
  whole_slice / total_slices + shared_slice = 1 / 12 :=
by
  sorry

end peter_pizza_total_l27_27339


namespace number_of_pencils_purchased_l27_27547

variable {total_pens : ℕ} (total_cost : ℝ) (avg_price_pencil avg_price_pen : ℝ)

theorem number_of_pencils_purchased 
  (h1 : total_pens = 30)
  (h2 : total_cost = 570)
  (h3 : avg_price_pencil = 2.00)
  (h4 : avg_price_pen = 14)
  : 
  ∃ P : ℕ, P = 75 :=
by
  sorry

end number_of_pencils_purchased_l27_27547


namespace product_remainder_mod_7_l27_27671

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l27_27671


namespace volume_of_rectangular_prism_l27_27068

-- Definition of the given conditions
variables (a b c : ℝ)

def condition1 : Prop := a * b = 24
def condition2 : Prop := b * c = 15
def condition3 : Prop := a * c = 10

-- The statement we want to prove
theorem volume_of_rectangular_prism
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c) :
  a * b * c = 60 :=
by sorry

end volume_of_rectangular_prism_l27_27068


namespace probability_of_moving_to_Q_after_n_seconds_l27_27726

noncomputable def probability_after_n_seconds (n : ℕ) : ℝ :=
  let M : Matrix (Fin 3) (Fin 3) ℝ :=
    ![![2/3, 1/6, 1/6], ![1/6, 2/3, 1/6], ![1/6, 1/6, 2/3]]
  let v0 : Fin 3 → ℝ := fun i => if i = 0 then 1 else 0
  let v_n := M^n • v0
  v_n 1

theorem probability_of_moving_to_Q_after_n_seconds (n : ℕ) : probability_after_n_seconds n = sorry := sorry

end probability_of_moving_to_Q_after_n_seconds_l27_27726


namespace problem1_problem2_l27_27417

theorem problem1 : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := 
by 
  sorry
  
theorem problem2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1) ^ 2 = 14 + 4 * Real.sqrt 3 := 
by 
  sorry

end problem1_problem2_l27_27417


namespace max_value_of_quadratic_at_2_l27_27543

def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

theorem max_value_of_quadratic_at_2 : ∃ (x : ℝ), x = 2 ∧ ∀ y : ℝ, f y ≤ f x :=
by
  use 2
  sorry

end max_value_of_quadratic_at_2_l27_27543


namespace valid_sandwiches_bob_can_order_l27_27647

def total_breads := 5
def total_meats := 7
def total_cheeses := 6

def undesired_combinations_count : Nat :=
  let turkey_swiss := total_breads
  let roastbeef_rye := total_cheeses
  let roastbeef_swiss := total_breads
  turkey_swiss + roastbeef_rye + roastbeef_swiss

def total_sandwiches : Nat :=
  total_breads * total_meats * total_cheeses

def valid_sandwiches_count : Nat :=
  total_sandwiches - undesired_combinations_count

theorem valid_sandwiches_bob_can_order : valid_sandwiches_count = 194 := by
  sorry

end valid_sandwiches_bob_can_order_l27_27647


namespace product_remainder_mod_7_l27_27681

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l27_27681


namespace standard_eq_circle_l27_27870

noncomputable def circle_eq (x y : ℝ) (r : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1

theorem standard_eq_circle {x y : ℝ}
  (h1 : 5 * x - 3 * y = 8)
  (h2 : abs x = abs y) :
  ∃ r : ℝ, circle_eq x y r :=
by {
  sorry
}

end standard_eq_circle_l27_27870


namespace min_boxes_to_eliminate_for_one_third_chance_l27_27789

-- Define the number of boxes
def total_boxes := 26

-- Define the number of boxes with at least $250,000
def boxes_with_at_least_250k := 6

-- Define the condition for having a 1/3 chance
def one_third_chance (remaining_boxes : ℕ) : Prop :=
  6 / remaining_boxes = 1 / 3

-- Define the target number of boxes to eliminate
def boxes_to_eliminate := total_boxes - 18

theorem min_boxes_to_eliminate_for_one_third_chance :
  ∃ remaining_boxes : ℕ, one_third_chance remaining_boxes ∧ total_boxes - remaining_boxes = boxes_to_eliminate :=
sorry

end min_boxes_to_eliminate_for_one_third_chance_l27_27789


namespace nine_a_minus_six_b_l27_27883

-- Define the variables and conditions.
variables (a b : ℚ)

-- Assume the given conditions.
def condition1 : Prop := 3 * a + 4 * b = 0
def condition2 : Prop := a = 2 * b - 3

-- Formalize the statement to prove.
theorem nine_a_minus_six_b (h1 : condition1 a b) (h2 : condition2 a b) : 9 * a - 6 * b = -81 / 5 :=
sorry

end nine_a_minus_six_b_l27_27883


namespace two_pow_n_plus_one_divisible_by_three_l27_27730

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end two_pow_n_plus_one_divisible_by_three_l27_27730


namespace repeating_decimal_division_l27_27540

theorem repeating_decimal_division:
  let x := (54 / 99 : ℚ)
  let y := (18 / 99 : ℚ)
  (x / y) * (1 / 2) = (3 / 2 : ℚ) := by
    sorry

end repeating_decimal_division_l27_27540


namespace find_possible_values_of_n_l27_27235

theorem find_possible_values_of_n (n : ℕ) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ 
    (2*n*(2*n + 1))/2 - (n*k + (n*(n-1))/2) = 1615) ↔ (n = 34 ∨ n = 38) :=
by
  sorry

end find_possible_values_of_n_l27_27235


namespace difference_of_cubes_l27_27166

theorem difference_of_cubes (x y : ℕ) (h1 : x = y + 3) (h2 : x + y = 5) : x^3 - y^3 = 63 :=
by sorry

end difference_of_cubes_l27_27166


namespace find_m_value_l27_27479

-- Condition: P(-m^2, 3) lies on the axis of symmetry of the parabola y^2 = mx
def point_on_axis_of_symmetry (m : ℝ) : Prop :=
  let P := (-m^2, 3)
  let axis_of_symmetry := (-m / 4)
  P.1 = axis_of_symmetry

theorem find_m_value (m : ℝ) (h : point_on_axis_of_symmetry m) : m = 1 / 4 :=
  sorry

end find_m_value_l27_27479


namespace time_shortened_by_opening_both_pipes_l27_27849

theorem time_shortened_by_opening_both_pipes 
  (a b p : ℝ) 
  (hp : a * p > 0) -- To ensure p > 0 and reservoir volume is positive
  (h1 : p = (a * p) / a) -- Given that pipe A alone takes p hours
  : p - (a * p) / (a + b) = (b * p) / (a + b) := 
sorry

end time_shortened_by_opening_both_pipes_l27_27849


namespace factorize_expression_l27_27573

-- The problem is about factorizing the expression x^3y - xy
theorem factorize_expression (x y : ℝ) : x^3 * y - x * y = x * y * (x - 1) * (x + 1) := 
by sorry

end factorize_expression_l27_27573


namespace savings_for_mother_l27_27634

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l27_27634


namespace point_in_first_quadrant_of_complex_number_l27_27955

open Complex

theorem point_in_first_quadrant_of_complex_number :
  let z := complex.sin (real.pi * (100 / 180)) - complex.I * complex.cos (real.pi * (100 / 180))
  let Z := (z.re, z.im)
  (0 < z.re) ∧ (0 < z.im) :=
by
  let z := complex.sin (real.pi * (100 / 180)) - complex.I * complex.cos (real.pi * (100 / 180))
  let Z := (z.re, z.im)
  sorry

end point_in_first_quadrant_of_complex_number_l27_27955


namespace min_value_fraction_l27_27453

theorem min_value_fraction (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b^2 - 4 * a * c ≤ 0) :
  (a + b + c) / (2 * a) ≥ 2 :=
  sorry

end min_value_fraction_l27_27453


namespace chi_square_test_l27_27061

-- Conditions
def n : ℕ := 100
def a : ℕ := 5
def b : ℕ := 55
def c : ℕ := 15
def d : ℕ := 25

-- Critical chi-square value for alpha = 0.001
def chi_square_critical : ℝ := 10.828

-- Calculated chi-square value
noncomputable def chi_square_value : ℝ :=
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement to prove
theorem chi_square_test : chi_square_value > chi_square_critical :=
by sorry

end chi_square_test_l27_27061


namespace largest_four_digit_perfect_cube_is_9261_l27_27830

-- Define the notion of a four-digit number and perfect cube
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

-- The main theorem statement
theorem largest_four_digit_perfect_cube_is_9261 :
  ∃ n, is_four_digit n ∧ is_perfect_cube n ∧ (∀ m, is_four_digit m ∧ is_perfect_cube m → m ≤ n) ∧ n = 9261 :=
sorry -- Proof is omitted

end largest_four_digit_perfect_cube_is_9261_l27_27830


namespace emerson_distance_l27_27089

theorem emerson_distance (d1 : ℕ) : 
  (d1 + 15 + 18 = 39) → d1 = 6 := 
by
  intro h
  have h1 : 33 = 39 - d1 := sorry -- Steps to manipulate equation to find d1
  sorry

end emerson_distance_l27_27089


namespace set_different_l27_27731

-- Definitions of the sets ①, ②, ③, and ④
def set1 : Set ℤ := {x | x = 1}
def set2 : Set ℤ := {y | (y - 1)^2 = 0}
def set3 : Set ℤ := {x | x = 1}
def set4 : Set ℤ := {1}

-- Lean statement to prove that set3 is different from the others
theorem set_different : set3 ≠ set1 ∧ set3 ≠ set2 ∧ set3 ≠ set4 :=
by
  -- Skipping the proof with sorry
  sorry

end set_different_l27_27731


namespace sum_of_squares_l27_27024

theorem sum_of_squares :
  (2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2 = 10) :=
by
  sorry

end sum_of_squares_l27_27024


namespace quadratic_minimum_value_l27_27452

theorem quadratic_minimum_value (p q : ℝ) (h_min_value : ∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) :
  q = 10 + p^2 / 8 :=
by
  sorry

end quadratic_minimum_value_l27_27452


namespace pure_gold_to_add_eq_46_67_l27_27075

-- Define the given conditions
variable (initial_alloy_weight : ℝ) (initial_gold_percentage : ℝ) (final_gold_percentage : ℝ)
variable (added_pure_gold : ℝ)

-- State the proof problem
theorem pure_gold_to_add_eq_46_67 :
  initial_alloy_weight = 20 ∧
  initial_gold_percentage = 0.50 ∧
  final_gold_percentage = 0.85 ∧
  (10 + added_pure_gold) / (20 + added_pure_gold) = 0.85 →
  added_pure_gold = 46.67 :=
by
  sorry

end pure_gold_to_add_eq_46_67_l27_27075


namespace solution_exists_l27_27815

def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f' (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem solution_exists (a b : ℝ) :
    f 1 a b = 10 ∧ f' 1 a b = 0 ↔ (a = -4 ∧ b = 11) :=
by 
  sorry

end solution_exists_l27_27815


namespace possible_values_of_x_l27_27472

theorem possible_values_of_x (x : ℝ) (h : (x^2 - 1) / x = 0) (hx : x ≠ 0) : x = 1 ∨ x = -1 :=
  sorry

end possible_values_of_x_l27_27472


namespace sqrt_domain_l27_27130

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l27_27130


namespace find_a_plus_b_l27_27318

open Complex Polynomial

noncomputable def a : ℝ := -5
noncomputable def b : ℝ := 44

theorem find_a_plus_b 
  (ha : a = -5)
  (hb : b = 44)
  (h₁ : a + b = 39)
  (hroots : (2 + I * Real.sqrt 7) ∈ (RootSet (X^3 + C(a) * X + C(b)) ℂ))
  (hconjugate : (2 - I * Real.sqrt 7) ∈ (RootSet (X^3 + C(a) * X + C(b)) ℂ))
  (hthird : -4 ∈ (RootSet (X^3 + C(a) * X + C(b)) ℂ)) : a + b = 39 :=
by
  have ha : a = -5 := ha
  have hb : b = 44 := hb
  have h : a + b = 39 := h
  rw [ha, hb]
  exact h

end find_a_plus_b_l27_27318


namespace exists_cycle_not_divisible_by_three_l27_27950

open Combinatorics

theorem exists_cycle_not_divisible_by_three (V : Type) [Finite V] 
  (G : SimpleGraph V) (h : ∀ v, 3 ≤ G.degree v) : 
  ∃ (c : List V), (G.isCycle c) ∧ (¬ (c.length % 3 = 0)) :=
sorry

end exists_cycle_not_divisible_by_three_l27_27950


namespace simplify_expression_l27_27810

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 :=
by
  sorry

end simplify_expression_l27_27810


namespace three_tangent_lines_l27_27461

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

def g (t : ℝ) : ℝ := t^2 / Real.exp t

theorem three_tangent_lines (a : ℝ) : 
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ g t1 = a ∧ g t2 = a ∧ g t3 = a) ↔ (0 < a ∧ a < 4 / Real.exp 2) :=
begin
  sorry
end

end three_tangent_lines_l27_27461


namespace product_remainder_mod_7_l27_27665

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l27_27665


namespace scaled_multiplication_l27_27747

theorem scaled_multiplication
  (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 :=
by
  sorry

end scaled_multiplication_l27_27747


namespace estimate_students_in_range_l27_27313

noncomputable def n_students := 3000
noncomputable def score_range_low := 70
noncomputable def score_range_high := 80
noncomputable def est_students_in_range := 408

theorem estimate_students_in_range : ∀ (n : ℕ) (k : ℕ), n = n_students →
  k = est_students_in_range →
  normal_distribution :=
sorry

end estimate_students_in_range_l27_27313


namespace find_k_eq_neg2_l27_27113

theorem find_k_eq_neg2 (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by sorry

end find_k_eq_neg2_l27_27113


namespace jeff_total_cabinets_l27_27619

def initial_cabinets : ℕ := 3
def cabinets_per_counter : ℕ := 2 * initial_cabinets
def total_cabinets_installed : ℕ := 3 * cabinets_per_counter + 5
def total_cabinets (initial : ℕ) (installed : ℕ) : ℕ := initial + installed

theorem jeff_total_cabinets : total_cabinets initial_cabinets total_cabinets_installed = 26 :=
by
  sorry

end jeff_total_cabinets_l27_27619


namespace remainder_of_product_mod_7_l27_27673

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l27_27673


namespace sin_theta_value_l27_27775

theorem sin_theta_value (θ : ℝ) (h₁ : 8 * (Real.tan θ) = 3 * (Real.cos θ)) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 := 
by sorry

end sin_theta_value_l27_27775


namespace total_number_of_people_l27_27612

def total_people_at_park(hikers bike_riders : Nat) : Nat :=
  hikers + bike_riders

theorem total_number_of_people 
  (bike_riders : Nat)
  (hikers : Nat)
  (hikers_eq_bikes_plus_178 : hikers = bike_riders + 178)
  (bikes_eq_249 : bike_riders = 249) :
  total_people_at_park hikers bike_riders = 676 :=
by
  sorry

end total_number_of_people_l27_27612


namespace solve_for_x_l27_27444

theorem solve_for_x :
  ∃ x : ℕ, (12 ^ 3) * (6 ^ x) / 432 = 144 ∧ x = 2 := by
  sorry

end solve_for_x_l27_27444


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27263

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27263


namespace smallest_four_digit_multiple_of_53_l27_27994

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27994


namespace distance_comparison_l27_27489

def distance_mart_to_home : ℕ := 800
def distance_home_to_academy : ℕ := 1300
def distance_academy_to_restaurant : ℕ := 1700

theorem distance_comparison :
  (distance_mart_to_home + distance_home_to_academy) - distance_academy_to_restaurant = 400 :=
by
  sorry

end distance_comparison_l27_27489


namespace function_decreasing_on_interval_l27_27506

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem function_decreasing_on_interval : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → 1 ≤ x₂ → x₁ ≤ x₂ → f x₁ ≥ f x₂ := by
  sorry

end function_decreasing_on_interval_l27_27506


namespace solve_equation_l27_27645

theorem solve_equation (x : ℝ) : 
  16 * (x - 1) ^ 2 - 9 = 0 ↔ (x = 7 / 4 ∨ x = 1 / 4) := by
  sorry

end solve_equation_l27_27645


namespace total_toes_on_bus_l27_27335

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l27_27335


namespace transformed_center_coordinates_l27_27723

theorem transformed_center_coordinates (S : (ℝ × ℝ)) (hS : S = (3, -4)) : 
  let reflected_S := (S.1, -S.2)
  let translated_S := (reflected_S.1, reflected_S.2 + 5)
  translated_S = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l27_27723


namespace taylor_scores_l27_27198

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end taylor_scores_l27_27198


namespace return_trip_time_is_15_or_67_l27_27229

variable (d p w : ℝ)

-- Conditions
axiom h1 : (d / (p - w)) = 100
axiom h2 : ∃ t : ℝ, t = d / p ∧ (d / (p + w)) = t - 15

-- Correct answer to prove: time for the return trip is 15 minutes or 67 minutes
theorem return_trip_time_is_15_or_67 : (d / (p + w)) = 15 ∨ (d / (p + w)) = 67 := 
by 
  sorry

end return_trip_time_is_15_or_67_l27_27229


namespace calculate_expression_l27_27238

theorem calculate_expression :
  (1/4 * 6.16^2) - (4 * 1.04^2) = 5.16 :=
by
  sorry

end calculate_expression_l27_27238


namespace total_heads_eq_fifteen_l27_27852

-- Definitions for types of passengers and their attributes
def cats_heads : Nat := 7
def cats_legs : Nat := 7 * 4
def total_legs : Nat := 43
def captain_heads : Nat := 1
def captain_legs : Nat := 1

noncomputable def crew_heads (C : Nat) : Nat := C
noncomputable def crew_legs (C : Nat) : Nat := 2 * C

theorem total_heads_eq_fifteen : 
  ∃ (C : Nat),
    cats_legs + crew_legs C + captain_legs = total_legs ∧
    cats_heads + crew_heads C + captain_heads = 15 :=
by
  sorry

end total_heads_eq_fifteen_l27_27852


namespace ring_groups_in_first_tree_l27_27237

variable (n : ℕ) (y1 y2 : ℕ) (t : ℕ) (groupsPerYear : ℕ := 6)

-- each tree's rings are in groups of 2 fat rings and 4 thin rings, representing 6 years
def group_represents_years : ℕ := groupsPerYear

-- second tree has 40 ring groups, so it is 40 * 6 = 240 years old
def second_tree_groups : ℕ := 40

-- first tree is 180 years older, so its age in years
def first_tree_age : ℕ := (second_tree_groups * groupsPerYear) + 180

-- number of ring groups in the first tree
def number_of_ring_groups_in_first_tree := first_tree_age / groupsPerYear

theorem ring_groups_in_first_tree :
  number_of_ring_groups_in_first_tree = 70 :=
by
  sorry

end ring_groups_in_first_tree_l27_27237


namespace molecular_weight_C4H10_l27_27380

theorem molecular_weight_C4H10
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (C4H10_C_atoms : ℕ)
  (C4H10_H_atoms : ℕ)
  (moles : ℝ) : 
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  C4H10_C_atoms = 4 →
  C4H10_H_atoms = 10 →
  moles = 6 →
  (C4H10_C_atoms * atomic_weight_C + C4H10_H_atoms * atomic_weight_H) * moles = 348.72 :=
by
  sorry

end molecular_weight_C4H10_l27_27380


namespace toes_on_bus_is_164_l27_27328

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l27_27328


namespace total_trolls_l27_27426

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l27_27426


namespace calculate_x_times_a_l27_27412

-- Define variables and assumptions
variables (a b x y : ℕ)
variable (hb : b = 4)
variable (hy : y = 2)
variable (h1 : a = 2 * b)
variable (h2 : x = 3 * y)
variable (h3 : a + b = x * y)

-- The statement to be proved
theorem calculate_x_times_a : x * a = 48 :=
by sorry

end calculate_x_times_a_l27_27412


namespace car_speed_conversion_l27_27060

theorem car_speed_conversion (V_kmph : ℕ) (h : V_kmph = 36) : (V_kmph * 1000 / 3600) = 10 := by
  sorry

end car_speed_conversion_l27_27060


namespace cyclic_sum_inequality_l27_27581

variable {a b c x y z : ℝ}

-- Define the conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : x = a + 1 / b - 1
axiom h5 : y = b + 1 / c - 1
axiom h6 : z = c + 1 / a - 1
axiom h7 : x > 0
axiom h8 : y > 0
axiom h9 : z > 0

-- The statement we need to prove
theorem cyclic_sum_inequality : (x * y) / (Real.sqrt (x * y) + 2) + (y * z) / (Real.sqrt (y * z) + 2) + (z * x) / (Real.sqrt (z * x) + 2) ≥ 1 :=
sorry

end cyclic_sum_inequality_l27_27581


namespace ball_draw_probability_l27_27250

/-- 
Four balls labeled with numbers 1, 2, 3, 4 are placed in an urn. 
A ball is drawn, its number is recorded, and then the ball is returned to the urn. 
This process is repeated three times. Each ball is equally likely to be drawn on each occasion. 
Given that the sum of the numbers recorded is 7, the probability that the ball numbered 2 was drawn twice is 1/4. 
-/
theorem ball_draw_probability :
  let draws := [(1, 1, 5),(1, 2, 4),(1, 3, 3),(2, 2, 3)]
  (3 / 12 = 1 / 4) :=
by
  sorry

end ball_draw_probability_l27_27250


namespace quartic_polynomial_sum_l27_27553

theorem quartic_polynomial_sum :
  ∀ (q : ℤ → ℤ),
    q 1 = 4 →
    q 8 = 26 →
    q 12 = 14 →
    q 15 = 34 →
    q 19 = 44 →
    (q 1 + q 2 + q 3 + q 4 + q 5 + q 6 + q 7 + q 8 + q 9 + q 10 +
     q 11 + q 12 + q 13 + q 14 + q 15 + q 16 + q 17 + q 18 + q 19 + q 20) = 252 :=
by
  intros
  sorry

end quartic_polynomial_sum_l27_27553


namespace stratified_sampling_numbers_l27_27908

-- Definitions of the conditions
def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

-- Hypothesis of proportions
def proportion_senior := senior_teachers / total_teachers
def proportion_intermediate := intermediate_teachers / total_teachers
def proportion_junior := junior_teachers / total_teachers

-- Expected sample counts using stratified sampling method
def expected_senior_drawn := proportion_senior * sample_size
def expected_intermediate_drawn := proportion_intermediate * sample_size
def expected_junior_drawn := proportion_junior * sample_size

-- Proof goal
theorem stratified_sampling_numbers :
  (expected_senior_drawn = 12) ∧ 
  (expected_intermediate_drawn = 20) ∧ 
  (expected_junior_drawn = 8) :=
by
  sorry

end stratified_sampling_numbers_l27_27908


namespace complement_of_intersection_l27_27768

def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

theorem complement_of_intersection (AuB AcB : Set ℤ) :
  (A ∪ B) = AuB ∧ (A ∩ B) = AcB → 
  A ∪ B = ∅ ∨ A ∪ B = AuB → 
  (AuB \ AcB) = {-1, 1} :=
by
  -- Proof construction method placeholder.
  sorry

end complement_of_intersection_l27_27768


namespace quadratic_no_real_roots_l27_27018

theorem quadratic_no_real_roots 
  (a b c m : ℝ) 
  (h1 : c > 0) 
  (h2 : c = a * m^2) 
  (h3 : c = b * m)
  : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0) :=
by 
  sorry

end quadratic_no_real_roots_l27_27018


namespace invertible_2x2_matrix_l27_27441

open Matrix

def matrix_2x2 := Matrix (Fin 2) (Fin 2) ℚ

def A : matrix_2x2 := ![![4, 5], ![-2, 9]]

def inv_A : matrix_2x2 := ![![9/46, -5/46], ![1/23, 2/23]]

theorem invertible_2x2_matrix :
  det A ≠ 0 → (inv A = inv_A) := 
by
  sorry

end invertible_2x2_matrix_l27_27441


namespace range_of_a_l27_27292

def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0

theorem range_of_a (a : ℝ) : (-2 / 3 : ℝ) ≤ a ∧ a < 0 := sorry

end range_of_a_l27_27292


namespace sugar_percentage_after_additions_l27_27548

noncomputable def initial_solution_volume : ℝ := 440
noncomputable def initial_water_percentage : ℝ := 0.88
noncomputable def initial_kola_percentage : ℝ := 0.08
noncomputable def initial_sugar_percentage : ℝ := 1 - initial_water_percentage - initial_kola_percentage
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8

noncomputable def initial_sugar_amount := initial_sugar_percentage * initial_solution_volume
noncomputable def new_sugar_amount := initial_sugar_amount + sugar_added
noncomputable def new_solution_volume := initial_solution_volume + sugar_added + water_added + kola_added

noncomputable def final_sugar_percentage := (new_sugar_amount / new_solution_volume) * 100

theorem sugar_percentage_after_additions :
    final_sugar_percentage = 4.52 :=
by
    sorry

end sugar_percentage_after_additions_l27_27548


namespace smaller_square_area_l27_27150

theorem smaller_square_area (A_L : ℝ) (h : A_L = 100) : ∃ A_S : ℝ, A_S = 50 := 
by
  sorry

end smaller_square_area_l27_27150


namespace marco_score_percentage_less_l27_27519

theorem marco_score_percentage_less
  (average_score : ℕ)
  (margaret_score : ℕ)
  (margaret_more_than_marco : ℕ)
  (h1 : average_score = 90)
  (h2 : margaret_score = 86)
  (h3 : margaret_more_than_marco = 5) :
  (average_score - (margaret_score - margaret_more_than_marco)) * 100 / average_score = 10 :=
by
  sorry

end marco_score_percentage_less_l27_27519


namespace value_of_sine_neg_10pi_over_3_l27_27023

theorem value_of_sine_neg_10pi_over_3 : Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end value_of_sine_neg_10pi_over_3_l27_27023


namespace greatest_four_digit_divisible_by_3_5_6_l27_27541

theorem greatest_four_digit_divisible_by_3_5_6 : 
  ∃ n, n ≤ 9999 ∧ n ≥ 1000 ∧ (∀ m, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n = 9990 :=
by 
  sorry

end greatest_four_digit_divisible_by_3_5_6_l27_27541


namespace sum_of_first_4_terms_l27_27614

noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem sum_of_first_4_terms (a r : ℝ) 
  (h1 : a * (1 + r + r^2) = 13) (h2 : a * (1 + r + r^2 + r^3 + r^4) = 121) : 
  a * (1 + r + r^2 + r^3) = 27.857 :=
by
  sorry

end sum_of_first_4_terms_l27_27614


namespace solve_inequality_l27_27175

theorem solve_inequality (x : ℝ) :
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ (Set.Iio 3 ∪ Set.Ioo 3 5) :=
by
  sorry

end solve_inequality_l27_27175


namespace parallel_line_through_point_l27_27867

theorem parallel_line_through_point :
  ∀ {x y : ℝ}, (3 * x + 4 * y + 1 = 0) ∧ (∃ (a b : ℝ), a = 1 ∧ b = 2 ∧ (3 * a + 4 * b + x0 = 0) → (x = -11)) :=
sorry

end parallel_line_through_point_l27_27867


namespace increasing_function_range_l27_27765

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1/2 ≤ a ∧ a ≤ 3/2) :=
sorry

end increasing_function_range_l27_27765


namespace price_of_second_oil_l27_27055

open Real

-- Define conditions
def litres_of_first_oil : ℝ := 10
def price_per_litre_first_oil : ℝ := 50
def litres_of_second_oil : ℝ := 5
def total_volume_of_mixture : ℝ := 15
def rate_of_mixture : ℝ := 55.67
def total_cost_of_mixture : ℝ := total_volume_of_mixture * rate_of_mixture

-- Define total cost of the first oil
def total_cost_first_oil : ℝ := litres_of_first_oil * price_per_litre_first_oil

-- Define total cost of the second oil in terms of unknown price P
def total_cost_second_oil (P : ℝ) : ℝ := litres_of_second_oil * P

-- Theorem to prove price per litre of the second oil
theorem price_of_second_oil : ∃ P : ℝ, total_cost_first_oil + (total_cost_second_oil P) = total_cost_of_mixture ∧ P = 67.01 :=
by
  sorry

end price_of_second_oil_l27_27055


namespace circumcircle_radius_of_sector_l27_27069

theorem circumcircle_radius_of_sector (θ : Real) (r : Real) (cos_val : Real) (R : Real) :
  θ = 30 * Real.pi / 180 ∧ r = 8 ∧ cos_val = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ R = 8 * (Real.sqrt 6 - Real.sqrt 2) →
  R = 8 * (Real.sqrt 6 - Real.sqrt 2) :=
by
  sorry

end circumcircle_radius_of_sector_l27_27069


namespace find_ABC_plus_DE_l27_27835

theorem find_ABC_plus_DE (ABCDE : Nat) (h1 : ABCDE = 13579 * 6) : (ABCDE / 1000 + ABCDE % 1000 % 100) = 888 :=
by
  sorry

end find_ABC_plus_DE_l27_27835


namespace smallest_n_divisor_lcm_gcd_l27_27469

theorem smallest_n_divisor_lcm_gcd :
  ∀ n : ℕ, n > 0 ∧ (∀ a b : ℕ, 60 = a ∧ n = b → (Nat.lcm a b / Nat.gcd a b = 50)) → n = 750 :=
by
  sorry

end smallest_n_divisor_lcm_gcd_l27_27469


namespace perpendicular_lines_sufficient_l27_27285

noncomputable def line1_slope (a : ℝ) : ℝ :=
-((a + 2) / (3 * a))

noncomputable def line2_slope (a : ℝ) : ℝ :=
-((a - 2) / (a + 2))

theorem perpendicular_lines_sufficient (a : ℝ) (h : a = -2) :
  line1_slope a * line2_slope a = -1 :=
by
  sorry

end perpendicular_lines_sufficient_l27_27285


namespace negation_exists_l27_27654

theorem negation_exists:
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by
  sorry

end negation_exists_l27_27654


namespace find_value_l27_27788

variable {a b c : ℝ}

def ellipse_eqn (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

theorem find_value 
  (h1 : a^2 + b^2 - 3*c^2 = 0)
  (h2 : a^2 = b^2 + c^2) :
  (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := 
  sorry

end find_value_l27_27788


namespace find_certain_number_l27_27734

theorem find_certain_number (x : ℤ) (h : ((x / 4) + 25) * 3 = 150) : x = 100 :=
by
  sorry

end find_certain_number_l27_27734


namespace combinations_of_coins_l27_27298

theorem combinations_of_coins (p n d : ℕ) (h₁ : p ≥ 0) (h₂ : n ≥ 0) (h₃ : d ≥ 0) 
  (value_eq : p + 5 * n + 10 * d = 25) : 
  ∃! c : ℕ, c = 12 :=
sorry

end combinations_of_coins_l27_27298


namespace max_prime_p_l27_27513

-- Define the variables and conditions
variable (a b : ℕ)
variable (p : ℝ)

-- Define the prime condition
def is_prime (n : ℝ) : Prop := sorry -- Placeholder for the prime definition

-- Define the equation condition
def p_eq (p : ℝ) (a b : ℕ) : Prop := 
  p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))

-- The theorem to prove
theorem max_prime_p (a b : ℕ) (p_max : ℝ) :
  (∃ p, is_prime p ∧ p_eq p a b) → p_max = 5 := 
sorry

end max_prime_p_l27_27513


namespace problem1_simplification_problem2_solve_fraction_l27_27050

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l27_27050


namespace remainder_of_product_mod_7_l27_27672

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l27_27672


namespace value_of_y_l27_27744

theorem value_of_y (y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) :
  a = 10^3 → b = 10^4 → 
  a^y * 10^(3 * y) = (b^4) → 
  y = 8 / 3 :=
by 
  intro ha hb hc
  rw [ha, hb] at hc
  sorry

end value_of_y_l27_27744


namespace equation_of_line_through_A_parallel_to_given_line_l27_27960

theorem equation_of_line_through_A_parallel_to_given_line :
  ∃ c : ℝ, 
    (∀ x y : ℝ, 2 * x - y + c = 0 ↔ ∃ a b : ℝ, a = -1 ∧ b = 0 ∧ 2 * a - b + 1 = 0) :=
sorry

end equation_of_line_through_A_parallel_to_given_line_l27_27960


namespace simplest_quadratic_radical_l27_27838

def is_simplest_radical (r : ℝ) : Prop :=
  ∀ x : ℝ, (∃ a b : ℝ, r = a * Real.sqrt b) → (∃ c d : ℝ, x = c * Real.sqrt d) → a ≤ c

def sqrt_12 := Real.sqrt 12
def sqrt_2_3 := Real.sqrt (2 / 3)
def sqrt_0_3 := Real.sqrt 0.3
def sqrt_7 := Real.sqrt 7

theorem simplest_quadratic_radical :
  is_simplest_radical sqrt_7 ∧
  ¬ is_simplest_radical sqrt_12 ∧
  ¬ is_simplest_radical sqrt_2_3 ∧
  ¬ is_simplest_radical sqrt_0_3 :=
by
  sorry

end simplest_quadratic_radical_l27_27838


namespace scientific_notation_correct_l27_27151

def distance_moon_km : ℕ := 384000

def scientific_notation (n : ℕ) : ℝ := 3.84 * 10^5

theorem scientific_notation_correct : scientific_notation distance_moon_km = 3.84 * 10^5 := by
  sorry

end scientific_notation_correct_l27_27151


namespace football_defense_stats_l27_27616

/-- Given:
1. Team 1 has an average of 1.5 goals conceded per match.
2. Team 1 has a standard deviation of 1.1 for the total number of goals conceded throughout the year.
3. Team 2 has an average of 2.1 goals conceded per match.
4. Team 2 has a standard deviation of 0.4 for the total number of goals conceded throughout the year.

Prove:
There are exactly 3 correct statements out of the 4 listed statements. -/
theorem football_defense_stats
  (avg_goals_team1 : ℝ := 1.5)
  (std_dev_team1 : ℝ := 1.1)
  (avg_goals_team2 : ℝ := 2.1)
  (std_dev_team2 : ℝ := 0.4) :
  ∃ correct_statements : ℕ, correct_statements = 3 := 
by
  sorry

end football_defense_stats_l27_27616


namespace fourth_bell_interval_l27_27721

-- Define intervals of the first three bells
def interval_bell1 : ℕ := 5
def interval_bell2 : ℕ := 8
def interval_bell3 : ℕ := 11

-- Interval for all bells tolling together
def all_toll_together : ℕ := 1320

theorem fourth_bell_interval (interval_bell4 : ℕ) :
  Nat.gcd (Nat.lcm (Nat.lcm interval_bell1 interval_bell2) interval_bell3) interval_bell4 = 1 →
  Nat.lcm (Nat.lcm (Nat.lcm interval_bell1 interval_bell2) interval_bell3) interval_bell4 = all_toll_together →
  interval_bell4 = 1320 := sorry

end fourth_bell_interval_l27_27721


namespace max_value_of_a_plus_b_l27_27386

theorem max_value_of_a_plus_b (a b : ℕ) (h1 : 7 * a + 19 * b = 213) (h2 : a > 0) (h3 : b > 0) : a + b = 27 :=
sorry

end max_value_of_a_plus_b_l27_27386


namespace find_second_number_l27_27847

theorem find_second_number (x : ℕ) : 9548 + x = 3362 + 13500 → x = 7314 := by
  sorry

end find_second_number_l27_27847


namespace min_value_expression_min_value_expression_achieved_at_1_l27_27494

noncomputable def min_value_expr (a b : ℝ) (n : ℕ) : ℝ :=
  (1 / (1 + a^n)) + (1 / (1 + b^n))

theorem min_value_expression (a b : ℝ) (n : ℕ) (h1 : a + b = 2) (h2 : 0 < a) (h3 : 0 < b) : 
  (min_value_expr a b n) ≥ 1 :=
sorry

theorem min_value_expression_achieved_at_1 (n : ℕ) :
  (min_value_expr 1 1 n = 1) :=
sorry

end min_value_expression_min_value_expression_achieved_at_1_l27_27494


namespace binary_10101_to_decimal_l27_27242

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.zipWith (λ digit idx => digit * 2^idx) (List.range b.length) |>.sum

theorem binary_10101_to_decimal : binary_to_decimal [1, 0, 1, 0, 1] = 21 := by
  sorry

end binary_10101_to_decimal_l27_27242


namespace tetrahedron_face_area_inequality_l27_27174

theorem tetrahedron_face_area_inequality
  (T_ABC T_ABD T_ACD T_BCD : ℝ)
  (h : T_ABC ≥ 0 ∧ T_ABD ≥ 0 ∧ T_ACD ≥ 0 ∧ T_BCD ≥ 0) :
  T_ABC < T_ABD + T_ACD + T_BCD :=
sorry

end tetrahedron_face_area_inequality_l27_27174


namespace toes_on_bus_is_164_l27_27329

def num_toes_hoopit : Nat := 3 * 4
def num_toes_neglart : Nat := 2 * 5

def num_hoopits : Nat := 7
def num_neglarts : Nat := 8

def total_toes_on_bus : Nat :=
  num_hoopits * num_toes_hoopit + num_neglarts * num_toes_neglart

theorem toes_on_bus_is_164 : total_toes_on_bus = 164 := by
  sorry

end toes_on_bus_is_164_l27_27329


namespace sufficient_condition_for_q_l27_27585

def p (a : ℝ) : Prop := a ≥ 0
def q (a : ℝ) : Prop := a^2 + a ≥ 0

theorem sufficient_condition_for_q (a : ℝ) : p a → q a := by 
  sorry

end sufficient_condition_for_q_l27_27585


namespace original_price_of_suit_l27_27368

theorem original_price_of_suit (P : ℝ) (hP : 0.70 * 1.30 * P = 182) : P = 200 :=
by
  sorry

end original_price_of_suit_l27_27368


namespace solve_equation_l27_27006

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end solve_equation_l27_27006


namespace calculate_expression_l27_27833

theorem calculate_expression :
  3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 :=
by
  sorry

end calculate_expression_l27_27833


namespace relationship_between_exponents_l27_27921

theorem relationship_between_exponents 
  (p r : ℝ) (u v s t m n : ℝ)
  (h1 : p^u = r^s)
  (h2 : r^v = p^t)
  (h3 : m = r^s)
  (h4 : n = r^v)
  (h5 : m^2 = n^3) :
  (s / u = v / t) ∧ (2 * s = 3 * v) :=
  by
  sorry

end relationship_between_exponents_l27_27921


namespace point_C_correct_l27_27231

-- Definitions of point A and B
def A : ℝ × ℝ := (4, -4)
def B : ℝ × ℝ := (18, 6)

-- Coordinate of C obtained from the conditions of the problem
def C : ℝ × ℝ := (25, 11)

-- Proof statement
theorem point_C_correct :
  ∃ C : ℝ × ℝ, (∃ (BC : ℝ × ℝ), BC = (1/2) • (B.1 - A.1, B.2 - A.2) ∧ C = (B.1 + BC.1, B.2 + BC.2)) ∧ C = (25, 11) :=
by
  sorry

end point_C_correct_l27_27231


namespace smallest_four_digit_multiple_of_53_l27_27998

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27998


namespace find_number_of_numbers_l27_27180

theorem find_number_of_numbers (S : ℝ) (n : ℝ) (h1 : S - 30 = 16 * n) (h2 : S = 19 * n) : n = 10 :=
by
  sorry

end find_number_of_numbers_l27_27180


namespace amount_borrowed_eq_4137_84_l27_27801

noncomputable def compound_interest (initial : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  initial * (1 + rate/100) ^ time

theorem amount_borrowed_eq_4137_84 :
  ∃ P : ℝ, 
    (compound_interest (compound_interest (compound_interest P 6 3) 8 4) 10 2 = 8110) 
    ∧ (P = 4137.84) :=
by
  sorry

end amount_borrowed_eq_4137_84_l27_27801


namespace total_men_wages_l27_27056

-- Define our variables and parameters
variable (M W B : ℝ)
variable (W_women : ℝ)

-- Conditions from the problem:
-- 1. 12M = WW (where WW is W_women)
-- 2. WW = 20B
-- 3. 12M + WW + 20B = 450
axiom eq_12M_WW : 12 * M = W_women
axiom eq_WW_20B : W_women = 20 * B
axiom eq_total_earnings : 12 * M + W_women + 20 * B = 450

-- Prove total wages of the men is Rs. 150
theorem total_men_wages : 12 * M = 150 := by
  sorry

end total_men_wages_l27_27056


namespace sumata_family_miles_driven_l27_27178

def total_miles_driven (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

theorem sumata_family_miles_driven :
  total_miles_driven 5 50 = 250 :=
by
  sorry

end sumata_family_miles_driven_l27_27178


namespace milk_amount_in_ounces_l27_27400

theorem milk_amount_in_ounces :
  ∀ (n_packets : ℕ) (ml_per_packet : ℕ) (ml_per_ounce : ℕ),
  n_packets = 150 →
  ml_per_packet = 250 →
  ml_per_ounce = 30 →
  (n_packets * ml_per_packet) / ml_per_ounce = 1250 :=
by
  intros n_packets ml_per_packet ml_per_ounce h_packets h_packet_ml h_ounce_ml
  rw [h_packets, h_packet_ml, h_ounce_ml]
  sorry

end milk_amount_in_ounces_l27_27400


namespace poly_divisible_by_seven_l27_27466

-- Define the given polynomial expression
def poly_expr (x n : ℕ) : ℕ := (1 + x)^n - 1

-- Define the proof statement
theorem poly_divisible_by_seven :
  ∀ x n : ℕ, x = 5 ∧ n = 4 → poly_expr x n % 7 = 0 :=
by
  intro x n h
  cases h
  sorry

end poly_divisible_by_seven_l27_27466


namespace debby_pancakes_l27_27724

def total_pancakes (B A P : ℕ) : ℕ := B + A + P

theorem debby_pancakes : 
  total_pancakes 20 24 23 = 67 := by 
  sorry

end debby_pancakes_l27_27724


namespace determine_b_for_inverse_function_l27_27358

theorem determine_b_for_inverse_function (b : ℝ) :
  (∀ x, (2 - 3 * (1 / (2 * x + b))) / (3 * (1 / (2 * x + b))) = x) ↔ b = 3 / 2 := by
  sorry

end determine_b_for_inverse_function_l27_27358


namespace cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l27_27906

-- Definitions for angles A, B, and C forming an arithmetic sequence and their sum being 180 degrees
variables {A B C : ℝ}

-- Definitions for side lengths a, b, and c forming a geometric sequence
variables {a b c : ℝ}

-- Question 1: Prove that cos B = 1/2 under the given conditions
theorem cos_B_equals_half 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) : 
  Real.cos B = 1 / 2 :=
sorry

-- Question 2: Prove that sin A * sin C = 3/4 under the given conditions
theorem sin_A_mul_sin_C_equals_three_fourths 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) 
  (h3 : b^2 = a * c) : 
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

end cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l27_27906


namespace capacity_of_initial_20_buckets_l27_27708

theorem capacity_of_initial_20_buckets (x : ℝ) (h : 20 * x = 270) : x = 13.5 :=
by 
  sorry

end capacity_of_initial_20_buckets_l27_27708


namespace probability_yellow_side_l27_27395

theorem probability_yellow_side :
  let num_cards := 8
  let num_blue_blue := 4
  let num_blue_yellow := 2
  let num_yellow_yellow := 2
  let total_yellow_sides := (2 * num_yellow_yellow + 1 * num_blue_yellow)
  let yellow_yellow_sides := (2 * num_yellow_yellow)
  (total_yellow_sides = 6) →
  (yellow_yellow_sides = 4) →
  (yellow_yellow_sides / total_yellow_sides = (2 : ℚ) / 3) := 
by 
  intros
  sorry

end probability_yellow_side_l27_27395


namespace rain_third_day_l27_27032

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end rain_third_day_l27_27032


namespace median_length_range_l27_27905

/-- Define the structure of the triangle -/
structure Triangle :=
  (A B C : ℝ) -- vertices of the triangle
  (AD AE AF : ℝ) -- lengths of altitude, angle bisector, and median
  (angleA : AngleType) -- type of angle A (acute, orthogonal, obtuse)

-- Define the angle type as a custom type
inductive AngleType
| acute
| orthogonal
| obtuse

def m_range (t : Triangle) : Set ℝ :=
  match t.angleA with
  | AngleType.acute => {m : ℝ | 13 < m ∧ m < (2028 / 119)}
  | AngleType.orthogonal => {m : ℝ | m = (2028 / 119)}
  | AngleType.obtuse => {m : ℝ | (2028 / 119) < m}

-- Lean statement for proving the problem
theorem median_length_range (t : Triangle)
  (hAD : t.AD = 12)
  (hAE : t.AE = 13) : t.AF ∈ m_range t :=
by
  sorry

end median_length_range_l27_27905


namespace equal_points_probability_l27_27688

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l27_27688


namespace finished_year_eq_183_l27_27913

theorem finished_year_eq_183 (x : ℕ) (h1 : x < 200) 
  (h2 : x ^ 13 = 258145266804692077858261512663) : x = 183 :=
sorry

end finished_year_eq_183_l27_27913


namespace range_of_x_in_sqrt_x_plus_3_l27_27127

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l27_27127


namespace range_of_y_l27_27423

noncomputable def operation (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_y :
  (∀ x : ℝ, operation (x - y) (x + y) < 1) ↔ - (1 : ℝ) / 2 < y ∧ y < (3 : ℝ) / 2 :=
by
  sorry

end range_of_y_l27_27423


namespace eldest_sibling_age_correct_l27_27787

-- Definitions and conditions
def youngest_sibling_age (x : ℝ) := x
def second_youngest_sibling_age (x : ℝ) := x + 4
def third_youngest_sibling_age (x : ℝ) := x + 8
def fourth_youngest_sibling_age (x : ℝ) := x + 12
def fifth_youngest_sibling_age (x : ℝ) := x + 16
def sixth_youngest_sibling_age (x : ℝ) := x + 20
def seventh_youngest_sibling_age (x : ℝ) := x + 28
def eldest_sibling_age (x : ℝ) := x + 32

def combined_age_of_eight_siblings (x : ℝ) : ℝ := 
  youngest_sibling_age x +
  second_youngest_sibling_age x +
  third_youngest_sibling_age x +
  fourth_youngest_sibling_age x +
  fifth_youngest_sibling_age x +
  sixth_youngest_sibling_age x +
  seventh_youngest_sibling_age x +
  eldest_sibling_age x

-- Proving the combined age part
theorem eldest_sibling_age_correct (x : ℝ) (h : combined_age_of_eight_siblings x - youngest_sibling_age (x + 24) = 140) : 
  eldest_sibling_age x = 34.5 := by
  sorry

end eldest_sibling_age_correct_l27_27787


namespace sum_of_fifth_powers_l27_27276

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l27_27276


namespace two_pow_n_plus_one_divisible_by_three_l27_27729

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end two_pow_n_plus_one_divisible_by_three_l27_27729


namespace goats_more_than_pigs_l27_27537

-- Defining the number of goats
def number_of_goats : ℕ := 66

-- Condition: there are twice as many chickens as goats
def number_of_chickens : ℕ := 2 * number_of_goats

-- Calculating the total number of goats and chickens
def total_goats_and_chickens : ℕ := number_of_goats + number_of_chickens

-- Condition: the number of ducks is half of the total number of goats and chickens
def number_of_ducks : ℕ := total_goats_and_chickens / 2

-- Condition: the number of pigs is a third of the number of ducks
def number_of_pigs : ℕ := number_of_ducks / 3

-- The statement we need to prove
theorem goats_more_than_pigs : number_of_goats - number_of_pigs = 33 := by
  -- The proof is omitted as instructed
  sorry

end goats_more_than_pigs_l27_27537


namespace total_wheels_in_garage_l27_27192

theorem total_wheels_in_garage 
    (num_bicycles : ℕ)
    (num_cars : ℕ)
    (wheels_per_bicycle : ℕ)
    (wheels_per_car : ℕ) 
    (num_bicycles_eq : num_bicycles = 9)
    (num_cars_eq : num_cars = 16)
    (wheels_per_bicycle_eq : wheels_per_bicycle = 2)
    (wheels_per_car_eq : wheels_per_car = 4) :
    num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car = 82 := 
by
    sorry

end total_wheels_in_garage_l27_27192


namespace four_lines_set_l27_27725

-- Define the ⬩ operation
def clubsuit (a b : ℝ) := a^3 * b - a * b^3

-- Define the main theorem
theorem four_lines_set (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (y = 0 ∨ x = 0 ∨ y = x ∨ y = -x) :=
by sorry

end four_lines_set_l27_27725


namespace remainder_of_product_mod_7_l27_27676

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l27_27676


namespace oranges_savings_l27_27629

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l27_27629


namespace period_of_sin3x_plus_cos3x_l27_27381

noncomputable def period_of_trig_sum (x : ℝ) : ℝ := 
  let y := (fun x => Real.sin (3 * x) + Real.cos (3 * x))
  (2 * Real.pi) / 3

theorem period_of_sin3x_plus_cos3x : (fun x => Real.sin (3 * x) + Real.cos (3 * x)) =
  (fun x => Real.sin (3 * (x + period_of_trig_sum x)) + Real.cos (3 * (x + period_of_trig_sum x))) :=
by
  sorry

end period_of_sin3x_plus_cos3x_l27_27381


namespace num_turtles_on_sand_l27_27405

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l27_27405


namespace solve_equation_l27_27005

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end solve_equation_l27_27005


namespace value_expression_l27_27607

theorem value_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by 
  sorry

end value_expression_l27_27607


namespace students_count_inconsistent_l27_27817

-- Define the conditions
variables (total_students boys_more_than_girls : ℤ)

-- Define the main theorem: The computed number of girls is not an integer
theorem students_count_inconsistent 
  (h1 : total_students = 3688) 
  (h2 : boys_more_than_girls = 373) 
  : ¬ ∃ x : ℤ, 2 * x + boys_more_than_girls = total_students := 
by
  sorry

end students_count_inconsistent_l27_27817


namespace two_pow_n_plus_one_divisible_by_three_l27_27727

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end two_pow_n_plus_one_divisible_by_three_l27_27727


namespace simplify_rationalize_l27_27352

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end simplify_rationalize_l27_27352


namespace complement_intersection_l27_27109

-- Definitions for the sets
def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2, 3}

-- Statement to be proved
theorem complement_intersection (hU : U = {0, 1, 2, 3}) (hA : A = {0, 1}) (hB : B = {1, 2, 3}) :
  ((U \ A) ∩ B) = {2, 3} :=
by
  -- Greek delta: skip proof details
  sorry

end complement_intersection_l27_27109


namespace no_triples_exist_l27_27574

theorem no_triples_exist (m p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m > 0) :
  2^m * p^2 + 1 ≠ q^7 :=
sorry

end no_triples_exist_l27_27574


namespace find_ordered_pairs_l27_27090

theorem find_ordered_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (2 * m ∣ 3 * n - 2 ∧ 2 * n ∣ 3 * m - 2) ↔ (m, n) = (2, 2) ∨ (m, n) = (10, 14) ∨ (m, n) = (14, 10) :=
by
  sorry

end find_ordered_pairs_l27_27090


namespace find_sum_squares_l27_27604

variables (x y : ℝ)

theorem find_sum_squares (h1 : y + 4 = (x - 2)^2) (h2 : x + 4 = (y - 2)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 15 :=
sorry

end find_sum_squares_l27_27604


namespace obtain_any_natural_from_4_l27_27211

/-- Definitions of allowed operations:
  - Append the digit 4.
  - Append the digit 0.
  - Divide by 2, if the number is even.
--/
def append4 (n : ℕ) : ℕ := 10 * n + 4
def append0 (n : ℕ) : ℕ := 10 * n
def divide2 (n : ℕ) : ℕ := n / 2

/-- We'll also define if a number is even --/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Define the set of operations applied on a number --/
inductive operations : ℕ → ℕ → Prop
| initial : operations 4 4
| append4_step (n m : ℕ) : operations n m → operations n (append4 m)
| append0_step (n m : ℕ) : operations n m → operations n (append0 m)
| divide2_step (n m : ℕ) : is_even m → operations n m → operations n (divide2 m)

/-- The main theorem proving that any natural number can be obtained from 4 using the allowed operations --/
theorem obtain_any_natural_from_4 (n : ℕ) : ∃ m, operations 4 m ∧ m = n :=
by sorry

end obtain_any_natural_from_4_l27_27211


namespace remainder_of_product_mod_7_l27_27675

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l27_27675


namespace simplify_rationalize_l27_27350

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l27_27350


namespace cupcake_ratio_l27_27507

theorem cupcake_ratio (C B : ℕ) (hC : C = 4) (hTotal : C + B = 12) : B / C = 2 :=
by
  sorry

end cupcake_ratio_l27_27507


namespace sqrt_16_eq_plus_minus_4_l27_27965

theorem sqrt_16_eq_plus_minus_4 : ∀ x : ℝ, (x^2 = 16) ↔ (x = 4 ∨ x = -4) :=
by sorry

end sqrt_16_eq_plus_minus_4_l27_27965


namespace fifth_powers_sum_eq_l27_27271

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l27_27271


namespace jill_second_bus_ride_time_l27_27029

theorem jill_second_bus_ride_time :
  let wait_time := 12
  let ride_time := 30
  let combined_time := wait_time + ride_time
  let second_bus_ride_time := combined_time / 2
  second_bus_ride_time = 21 :=
by
  -- Introduce the definitions
  let wait_time := 12
  let ride_time := 30
  let combined_time := wait_time + ride_time
  let second_bus_ride_time := combined_time / 2
  -- State and confirm the goal
  have : second_bus_ride_time = 21 := by rfl
  exact this
  sorry

end jill_second_bus_ride_time_l27_27029


namespace range_of_sqrt_x_plus_3_l27_27141

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l27_27141


namespace minimum_questionnaires_l27_27043

theorem minimum_questionnaires (responses_needed : ℕ) (response_rate : ℝ)
  (h1 : responses_needed = 300) (h2 : response_rate = 0.70) :
  ∃ (n : ℕ), n = Nat.ceil (responses_needed / response_rate) ∧ n = 429 :=
by
  sorry

end minimum_questionnaires_l27_27043


namespace black_pork_zongzi_price_reduction_l27_27087

def price_reduction_15_dollars (initial_profit initial_boxes extra_boxes_per_dollar x : ℕ) : Prop :=
  initial_profit > x ∧ (initial_profit - x) * (initial_boxes + extra_boxes_per_dollar * x) = 2800 -> x = 15

-- Applying the problem conditions explicitly and stating the proposition to prove
theorem black_pork_zongzi_price_reduction:
  price_reduction_15_dollars 50 50 2 15 :=
by
  -- Here we state the question as a proposition based on the identified conditions and correct answer
  sorry

end black_pork_zongzi_price_reduction_l27_27087


namespace monotonic_intervals_range_of_m_l27_27580

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 3 - 2 * x)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 2 * x + m - 3

theorem monotonic_intervals :
  ∀ k : ℤ,
    (
      (∀ x, -Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 5 * Real.pi / 12 + k * Real.pi → ∃ (d : ℝ), f x = d)
      ∧
      (∀ x, 5 * Real.pi / 12 + k * Real.pi ≤ x ∧ x ≤ 11 * Real.pi / 12 + k * Real.pi → ∃ (i : ℝ), f x = i)
    ) := sorry

theorem range_of_m (m : ℝ) :
  (∀ x1 : ℝ, Real.pi / 12 ≤ x1 ∧ x1 ≤ Real.pi / 2 → ∃ x2 : ℝ, -2 ≤ x2 ∧ x2 ≤ m ∧ f x1 = g x2 m) ↔ -1 ≤ m ∧ m ≤ 3 := sorry

end monotonic_intervals_range_of_m_l27_27580


namespace company_A_at_least_two_correct_company_A_higher_chance_l27_27641

-- Defining the condition where Company A can answer 4 out of the 6 questions correctly
def company_A_correct_questions := 4

-- Defining the condition where Company B can answer each question correctly with probability 2/3
def company_B_correct_prob := 2 / 3

-- Proving the probability that Company A answers at least 2 questions correctly
theorem company_A_at_least_two_correct : ProbMassFunction.prob (λ (x : ℕ), (company_A_correct_questions.choose x * (6 - company_A_correct_questions).choose (3 - x)) / 6.choose 3) (λ x, 2 ≤ x) = 4 / 5 := 
sorry

-- Defining expectations and variances for both companies
def expectation_variance_A :=
  let E_X := 2 in
  let D_X := 2/5 in
  (E_X, D_X)

def expectation_variance_B :=
  let E_Y := 2 in
  let D_Y := 2/3 in
  (E_Y, D_Y)

-- Proving that Company A has a higher chance of winning based on lower variance
theorem company_A_higher_chance : ∀ (E_X D_X E_Y D_Y : ℝ), 
  (E_X = 2 ∧ D_X = 2/5) ∧ (E_Y = 2 ∧ D_Y = 2/3) → D_X < D_Y → "Company A wins" := 
sorry

end company_A_at_least_two_correct_company_A_higher_chance_l27_27641


namespace total_balls_l27_27780

def black_balls : ℕ := 8
def white_balls : ℕ := 6 * black_balls
theorem total_balls : white_balls + black_balls = 56 := 
by 
  sorry

end total_balls_l27_27780


namespace find_hyperbola_equation_l27_27740

/-- 
  Given a hyperbola with the equation x²/2 - y² = 1 and an ellipse with the equation y²/8 + x²/2 = 1,
  we want to find the equation of a hyperbola that shares the same asymptotes as the given hyperbola 
  and shares a common focus with the ellipse.
-/

noncomputable def hyperbola_equation : Prop :=
  ∃ (x y : ℝ), (y^2 / 2) - (x^2 / 4) = 1 ∧ 
               -- Asymptotes same as x²/2 - y² = 1
               ∀ (x y : ℝ), 2 * y = ± (sqrt (1/8)) * x + C ∧ 
               -- Shares a common focus with y²/8 + x²/2 = 1 (focus distance sqrt 6)
               ∀ (x y : ℝ), focus_distance = sqrt 6

theorem find_hyperbola_equation :
  hyperbola_equation :=
sorry

end find_hyperbola_equation_l27_27740


namespace least_third_side_of_right_triangle_l27_27895

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l27_27895


namespace cassidy_current_posters_l27_27240

theorem cassidy_current_posters : 
  ∃ P : ℕ, 
    (P + 6 = 2 * 14) → 
    P = 22 :=
begin
  sorry
end

end cassidy_current_posters_l27_27240


namespace minimize_y_at_x_l27_27796

-- Define the function y
def y (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * (x - a) ^ 2 + (x - b) ^ 2 

-- State the theorem
theorem minimize_y_at_x (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x' a b ≥ y ((3 * a + b) / 4) a b) :=
by
  sorry

end minimize_y_at_x_l27_27796


namespace find_number_l27_27220

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
by {
  sorry
}

end find_number_l27_27220


namespace units_digit_diff_l27_27760

theorem units_digit_diff (p : ℕ) (hp : p > 0) (even_p : p % 2 = 0) (units_p1_7 : (p + 1) % 10 = 7) : (p^3 % 10) = (p^2 % 10) :=
by
  sorry

end units_digit_diff_l27_27760


namespace find_larger_number_l27_27605

theorem find_larger_number (x y : ℝ) (h1 : 4 * y = 6 * x) (h2 : x + y = 36) : y = 21.6 :=
by
  sorry

end find_larger_number_l27_27605


namespace even_integers_diff_digits_200_to_800_l27_27772

theorem even_integers_diff_digits_200_to_800 :
  ∃ n : ℕ, n = 131 ∧ (∀ x : ℕ, 200 ≤ x ∧ x < 800 ∧ (x % 2 = 0) ∧ (∀ i j : ℕ, i ≠ j → (x / 10^i % 10) ≠ (x / 10^j % 10)) ↔ x < n) :=
sorry

end even_integers_diff_digits_200_to_800_l27_27772


namespace point_in_fourth_quadrant_l27_27617

-- Define the point (2, -3)
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 2, y := -3 }

-- Define what it means for a point to be in a specific quadrant
def inFirstQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y > 0

def inSecondQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y > 0

def inThirdQuadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

def inFourthQuadrant (P : Point) : Prop :=
  P.x > 0 ∧ P.y < 0

-- Define the theorem to prove that the point A lies in the fourth quadrant
theorem point_in_fourth_quadrant : inFourthQuadrant A :=
  sorry

end point_in_fourth_quadrant_l27_27617


namespace profit_percentage_each_portion_l27_27232

theorem profit_percentage_each_portion (P : ℝ) (total_apples : ℝ) 
  (portion1_percentage : ℝ) (portion2_percentage : ℝ) (total_profit_percentage : ℝ) :
  total_apples = 280 →
  portion1_percentage = 0.4 →
  portion2_percentage = 0.6 →
  total_profit_percentage = 0.3 →
  portion1_percentage * P + portion2_percentage * P = total_profit_percentage →
  P = 0.3 :=
by
  intros
  sorry

end profit_percentage_each_portion_l27_27232


namespace reaches_school_early_l27_27974

theorem reaches_school_early (R : ℝ) (T : ℝ) (F : ℝ) (T' : ℝ)
    (h₁ : F = (6/5) * R)
    (h₂ : T = 24)
    (h₃ : R * T = F * T')
    : T - T' = 4 := by
  -- All the given conditions are set; fill in the below placeholder with the proof.
  sorry

end reaches_school_early_l27_27974


namespace ordering_eight_four_three_l27_27379

noncomputable def eight_pow_ten := 8 ^ 10
noncomputable def four_pow_fifteen := 4 ^ 15
noncomputable def three_pow_twenty := 3 ^ 20

theorem ordering_eight_four_three :
  eight_pow_ten < three_pow_twenty ∧ three_pow_twenty < four_pow_fifteen :=
by
  sorry

end ordering_eight_four_three_l27_27379


namespace function_domain_l27_27134

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l27_27134


namespace number_of_technicians_l27_27014

-- Definitions of the conditions
def average_salary_all_workers := 10000
def average_salary_technicians := 12000
def average_salary_rest := 8000
def total_workers := 14

-- Variables for the number of technicians and the rest of the workers
variable (T R : ℕ)

-- Problem statement in Lean
theorem number_of_technicians :
  (T + R = total_workers) →
  (T * average_salary_technicians + R * average_salary_rest = total_workers * average_salary_all_workers) →
  T = 7 :=
by
  -- leaving the proof as sorry
  sorry

end number_of_technicians_l27_27014


namespace increase_in_p_does_not_imply_increase_in_equal_points_probability_l27_27690

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end increase_in_p_does_not_imply_increase_in_equal_points_probability_l27_27690


namespace fractional_eq_solution_l27_27446

theorem fractional_eq_solution (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) →
  k ≠ -3 ∧ k ≠ 5 :=
by 
  sorry

end fractional_eq_solution_l27_27446


namespace nested_roots_identity_l27_27777

theorem nested_roots_identity (x : ℝ) (hx : x ≥ 0) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) :=
sorry

end nested_roots_identity_l27_27777


namespace taylor_scores_l27_27199

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end taylor_scores_l27_27199


namespace probability_triangle_or_hexagon_l27_27909

theorem probability_triangle_or_hexagon 
  (total_shapes : ℕ) 
  (num_triangles : ℕ) 
  (num_squares : ℕ) 
  (num_circles : ℕ) 
  (num_hexagons : ℕ)
  (htotal : total_shapes = 10)
  (htriangles : num_triangles = 3)
  (hsquares : num_squares = 4)
  (hcircles : num_circles = 2)
  (hhexagons : num_hexagons = 1):
  (num_triangles + num_hexagons) / total_shapes = 2 / 5 := 
by 
  sorry

end probability_triangle_or_hexagon_l27_27909


namespace cassidy_number_of_posters_l27_27241

/-- Cassidy's current number of posters -/
def current_posters (C : ℕ) : Prop := 
  C + 6 = 2 * 14

theorem cassidy_number_of_posters : ∃ C : ℕ, current_posters C := 
  Exists.intro 22 sorry

end cassidy_number_of_posters_l27_27241


namespace gain_percentage_is_30_l27_27702

-- Definitions based on the conditions
def selling_price : ℕ := 195
def gain : ℕ := 45
def cost_price : ℕ := selling_price - gain
def gain_percentage : ℕ := (gain * 100) / cost_price

-- The statement to prove the gain percentage
theorem gain_percentage_is_30 : gain_percentage = 30 := 
by 
  -- Allow usage of fictive sorry for incomplete proof
  sorry

end gain_percentage_is_30_l27_27702


namespace product_of_x_values_product_of_all_possible_x_values_l27_27302

theorem product_of_x_values (x : ℚ) (h : abs ((18 : ℚ) / x - 4) = 3) :
  x = 18 ∨ x = 18 / 7 :=
sorry

theorem product_of_all_possible_x_values (x1 x2 : ℚ) (h1 : abs ((18 : ℚ) / x1 - 4) = 3) (h2 : abs ((18 : ℚ) / x2 - 4) = 3) :
  x1 * x2 = 324 / 7 :=
sorry

end product_of_x_values_product_of_all_possible_x_values_l27_27302


namespace sin_cos_15_degree_l27_27085

theorem sin_cos_15_degree :
  (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end sin_cos_15_degree_l27_27085


namespace function_domain_l27_27133

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l27_27133


namespace tino_jellybeans_l27_27686

theorem tino_jellybeans (Tino Lee Arnold Joshua : ℕ)
  (h1 : Tino = Lee + 24)
  (h2 : Arnold = Lee / 2)
  (h3 : Joshua = 3 * Arnold)
  (h4 : Arnold = 5) : Tino = 34 := by
sorry

end tino_jellybeans_l27_27686


namespace quadratic_points_relation_l27_27455

theorem quadratic_points_relation
  (k y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -((-1) - 1)^2 + k)
  (hB : y₂ = -(2 - 1)^2 + k)
  (hC : y₃ = -(4 - 1)^2 + k) : y₃ < y₁ ∧ y₁ < y₂ :=
by
  sorry

end quadratic_points_relation_l27_27455


namespace arithmetic_sequence_sum_l27_27370

theorem arithmetic_sequence_sum (c d e : ℕ) (h1 : 10 - 3 = 7) (h2 : 17 - 10 = 7) (h3 : c - 17 = 7) (h4 : d - c = 7) (h5 : e - d = 7) : 
  c + d + e = 93 :=
sorry

end arithmetic_sequence_sum_l27_27370


namespace least_sum_p_q_r_l27_27323

theorem least_sum_p_q_r (p q r : ℕ) (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (h : 17 * (p + 1) = 28 * (q + 1) ∧ 28 * (q + 1) = 35 * (r + 1)) : p + q + r = 290 :=
  sorry

end least_sum_p_q_r_l27_27323


namespace impossible_to_form_palindrome_l27_27843

-- Define the possible cards
inductive Card
| abc | bca | cab

-- Define the rule for palindrome formation
def canFormPalindrome (w : List Card) : Prop :=
  sorry  -- Placeholder for the actual formation rule

-- Define the theorem statement
theorem impossible_to_form_palindrome (w : List Card) :
  ¬canFormPalindrome w :=
sorry

end impossible_to_form_palindrome_l27_27843


namespace value_at_7_5_l27_27496

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = -f x
axiom interval_condition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x

theorem value_at_7_5 : f 7.5 = -0.5 := by
  sorry

end value_at_7_5_l27_27496


namespace height_of_sunflower_in_feet_l27_27930

def height_of_sister_in_feet : ℕ := 4
def height_of_sister_in_inches : ℕ := 3
def additional_height_of_sunflower : ℕ := 21

theorem height_of_sunflower_in_feet 
  (h_sister_feet : height_of_sister_in_feet = 4)
  (h_sister_inches : height_of_sister_in_inches = 3)
  (h_additional : additional_height_of_sunflower = 21) :
  (4 * 12 + 3 + 21) / 12 = 6 :=
by simp [h_sister_feet, h_sister_inches, h_additional]; norm_num; sorry

end height_of_sunflower_in_feet_l27_27930


namespace binomial_expansion_value_l27_27384

theorem binomial_expansion_value : 
  105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end binomial_expansion_value_l27_27384


namespace quadratic_to_binomial_square_l27_27039

theorem quadratic_to_binomial_square (m : ℝ) : 
  (∃ c : ℝ, (x : ℝ) → x^2 - 12 * x + m = (x + c)^2) ↔ m = 36 := 
sorry

end quadratic_to_binomial_square_l27_27039


namespace eval_expr_l27_27572

theorem eval_expr : (2/5) + (3/8) - (1/10) = 27/40 :=
by
  sorry

end eval_expr_l27_27572


namespace product_mod_7_l27_27661

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l27_27661


namespace dot_product_example_l27_27770

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example 
  (ha : a = (-1, 1)) 
  (hb : b = (3, -2)) : dot_product a b = -5 := by
  sorry

end dot_product_example_l27_27770


namespace cos_C_value_l27_27296

theorem cos_C_value (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 3 * c * Real.cos C)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  : Real.cos C = (Real.sqrt 10) / 10 :=
sorry

end cos_C_value_l27_27296


namespace oranges_in_total_l27_27720

def number_of_boxes := 3
def oranges_per_box := 8
def total_oranges := 24

theorem oranges_in_total : number_of_boxes * oranges_per_box = total_oranges := 
by {
  -- sorry skips the proof part
  sorry 
}

end oranges_in_total_l27_27720


namespace find_fractions_l27_27957

-- Define the numerators and denominators
def p1 := 75
def p2 := 70
def q1 := 34
def q2 := 51

-- Define the fractions
def frac1 := p1 / q1
def frac2 := p1 / q2

-- Define the greatest common divisor (gcd) condition
def gcd_condition := Nat.gcd p1 p2 = p1 - p2

-- Define the least common multiple (lcm) condition
def lcm_condition := Nat.lcm p1 p2 = 1050

-- Define the difference condition
def difference_condition := (frac1 - frac2) = (5 / 6)

-- Lean proof statement
theorem find_fractions :
  gcd_condition ∧ lcm_condition ∧ difference_condition :=
by
  sorry

end find_fractions_l27_27957


namespace circles_tangent_length_of_chord_l27_27563

open Real

-- Definitions based on given conditions
def r₁ := 4
def r₂ := 10
def r₃ := 14
def length_of_chord : ℝ := (8 * real.sqrt 390) / 7

-- Main theorem statement
theorem circles_tangent_length_of_chord (m n p : ℕ) 
  (h₁ : ∃ m n p, length_of_chord = (m * real.sqrt n) / p)
  (h₂ : m.gcd p = 1)
  (h₃ : ∀ k : ℕ, k^2 ∣ n → k=1) 
  : m + n + p = 405 :=
by 
  use 8, 390, 7
  sorry

end circles_tangent_length_of_chord_l27_27563


namespace units_digit_diff_is_seven_l27_27652

noncomputable def units_digit_resulting_difference (a b c : ℕ) (h1 : a = c - 3) :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let difference := original - reversed
  difference % 10

theorem units_digit_diff_is_seven (a b c : ℕ) (h1 : a = c - 3) :
  units_digit_resulting_difference a b c h1 = 7 :=
by sorry

end units_digit_diff_is_seven_l27_27652


namespace simplify_expr_l27_27346

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l27_27346


namespace ram_task_completion_days_l27_27171

theorem ram_task_completion_days (R : ℕ) (h1 : ∀ k : ℕ, k = R / 2) (h2 : 1 / R + 2 / R = 1 / 12) : R = 36 :=
sorry

end ram_task_completion_days_l27_27171


namespace range_of_x_in_sqrt_x_plus_3_l27_27125

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l27_27125


namespace solve_for_X_l27_27947

theorem solve_for_X (X : ℝ) (h : (X ^ (5 / 4)) = 32 * (32 ^ (1 / 16))) :
  X =  16 * (2 ^ (1 / 4)) :=
sorry

end solve_for_X_l27_27947


namespace second_smallest_sum_l27_27535

theorem second_smallest_sum (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
                           (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
                           (h7 : a + b + c = 180) (h8 : a + c + d = 197)
                           (h9 : b + c + d = 208) (h10 : a + b + d = 222) :
  208 ≠ 180 ∧ 208 ≠ 197 ∧ 208 ≠ 222 := 
sorry

end second_smallest_sum_l27_27535


namespace ball_hits_ground_at_t_l27_27017

noncomputable def ball_height (t : ℝ) : ℝ := -6 * t^2 - 10 * t + 56

theorem ball_hits_ground_at_t :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 7 / 3 := by
  sorry

end ball_hits_ground_at_t_l27_27017


namespace rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l27_27362

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l27_27362


namespace num_turtles_on_sand_l27_27403

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l27_27403


namespace molecular_weight_of_one_mole_l27_27982

noncomputable def molecular_weight (total_weight : ℝ) (moles : ℕ) : ℝ :=
total_weight / moles

theorem molecular_weight_of_one_mole (h : molecular_weight 252 6 = 42) : molecular_weight 252 6 = 42 := by
  exact h

end molecular_weight_of_one_mole_l27_27982


namespace fish_left_in_tank_l27_27800

theorem fish_left_in_tank (initial_fish : ℕ) (fish_taken_out : ℕ) (fish_left : ℕ) 
  (h1 : initial_fish = 19) (h2 : fish_taken_out = 16) : fish_left = initial_fish - fish_taken_out :=
by
  simp [h1, h2]
  sorry

end fish_left_in_tank_l27_27800


namespace initial_pigs_count_l27_27327

theorem initial_pigs_count (P : ℕ) (h1 : 2 + P + 6 + 3 + 5 + 2 = 21) : P = 3 :=
by
  sorry

end initial_pigs_count_l27_27327


namespace abs_diff_of_prod_and_sum_l27_27320

theorem abs_diff_of_prod_and_sum (m n : ℝ) (h1 : m * n = 8) (h2 : m + n = 6) : |m - n| = 2 :=
by
  -- The proof is not required as per the instructions.
  sorry

end abs_diff_of_prod_and_sum_l27_27320


namespace birds_find_more_than_half_millet_on_thursday_l27_27326

def millet_on_day (n : ℕ) : ℝ :=
  2 - 2 * (0.7 ^ n)

def more_than_half_millet (day : ℕ) : Prop :=
  millet_on_day day > 1

theorem birds_find_more_than_half_millet_on_thursday : more_than_half_millet 4 :=
by
  sorry

end birds_find_more_than_half_millet_on_thursday_l27_27326


namespace mia_money_l27_27503

variable (DarwinMoney MiaMoney : ℕ)

theorem mia_money :
  (MiaMoney = 2 * DarwinMoney + 20) → (DarwinMoney = 45) → MiaMoney = 110 := by
  intros h1 h2
  rw [h2] at h1
  rw [h1]
  sorry

end mia_money_l27_27503


namespace product_remainder_mod_7_l27_27679

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l27_27679


namespace isosceles_triangle_perimeter_l27_27784

-- Define the lengths of the sides
def side1 : ℕ := 4
def side2 : ℕ := 7

-- Condition: The given sides form an isosceles triangle
def is_isosceles_triangle (a b : ℕ) : Prop := a = b ∨ a = 4 ∧ b = 7 ∨ a = 7 ∧ b = 4

-- Condition: The triangle inequality theorem must be satisfied
def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem we want to prove
theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : is_isosceles_triangle a b) (h2 : triangle_inequality a a b ∨ triangle_inequality b b a) :
  a + a + b = 15 ∨ b + b + a = 18 := 
sorry

end isosceles_triangle_perimeter_l27_27784


namespace jake_more_balloons_than_allan_l27_27411

-- Define the initial and additional balloons for Allan
def initial_allan_balloons : Nat := 2
def additional_allan_balloons : Nat := 3

-- Total balloons Allan has in the park
def total_allan_balloons : Nat := initial_allan_balloons + additional_allan_balloons

-- Number of balloons Jake has
def jake_balloons : Nat := 6

-- The proof statement
theorem jake_more_balloons_than_allan : jake_balloons - total_allan_balloons = 1 := by
  sorry

end jake_more_balloons_than_allan_l27_27411


namespace trisect_diagonal_l27_27311

theorem trisect_diagonal 
  (A B C D E F : Point)
  (G H : Point) 
  (h1 : Rectangle ABCD) 
  (h2 : Midpoint E B C) 
  (h3 : Midpoint F C D) 
  (h4 : IntersectPoint G A E B D)
  (h5 : IntersectPoint H A F B D) : 
  SegmentEqual B G G H ∧ SegmentEqual G H H D :=
sorry

end trisect_diagonal_l27_27311


namespace fifth_powers_sum_eq_l27_27267

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l27_27267


namespace find_shortage_l27_27844

def total_capacity (T : ℝ) : Prop :=
  0.70 * T = 14

def normal_level (normal : ℝ) : Prop :=
  normal = 14 / 2

def capacity_shortage (T : ℝ) (normal : ℝ) : Prop :=
  T - normal = 13

theorem find_shortage (T : ℝ) (normal : ℝ) : 
  total_capacity T →
  normal_level normal →
  capacity_shortage T normal :=
by
  sorry

end find_shortage_l27_27844


namespace count_of_numbers_with_digit_3_eq_71_l27_27599

-- Define the problem space
def count_numbers_without_digit_3 : ℕ := 729
def total_numbers : ℕ := 800
def count_numbers_with_digit_3 : ℕ := total_numbers - count_numbers_without_digit_3

-- Prove that the count of numbers from 1 to 800 containing at least one digit 3 is 71
theorem count_of_numbers_with_digit_3_eq_71 :
  count_numbers_with_digit_3 = 71 :=
by
  sorry

end count_of_numbers_with_digit_3_eq_71_l27_27599


namespace greatest_five_digit_common_multiple_l27_27209

theorem greatest_five_digit_common_multiple (n : ℕ) :
  (n % 18 = 0) ∧ (10000 ≤ n) ∧ (n ≤ 99999) → n = 99990 :=
by
  sorry

end greatest_five_digit_common_multiple_l27_27209


namespace tony_water_trips_calculation_l27_27822

noncomputable def tony_drinks_water_after_every_n_trips (bucket_capacity_sand : ℤ) 
                                                        (sandbox_depth : ℤ) (sandbox_width : ℤ) 
                                                        (sandbox_length : ℤ) (sand_weight_cubic_foot : ℤ) 
                                                        (water_consumption : ℤ) (water_bottle_ounces : ℤ) 
                                                        (water_bottle_cost : ℤ) (money_with_tony : ℤ) 
                                                        (expected_change : ℤ) : ℤ :=
  let volume_sandbox := sandbox_depth * sandbox_width * sandbox_length
  let total_sand_weight := volume_sandbox * sand_weight_cubic_foot
  let trips_needed := total_sand_weight / bucket_capacity_sand
  let money_spent_on_water := money_with_tony - expected_change
  let water_bottles_bought := money_spent_on_water / water_bottle_cost
  let total_water_ounces := water_bottles_bought * water_bottle_ounces
  let drinking_sessions := total_water_ounces / water_consumption
  trips_needed / drinking_sessions

theorem tony_water_trips_calculation : 
  tony_drinks_water_after_every_n_trips 2 2 4 5 3 3 15 2 10 4 = 4 := 
by 
  sorry

end tony_water_trips_calculation_l27_27822


namespace Mia_biking_speed_l27_27866

theorem Mia_biking_speed
    (Eugene_speed : ℝ)
    (Carlos_ratio : ℝ)
    (Mia_ratio : ℝ)
    (Mia_speed : ℝ)
    (h1 : Eugene_speed = 5)
    (h2 : Carlos_ratio = 3 / 4)
    (h3 : Mia_ratio = 4 / 3)
    (h4 : Mia_speed = Mia_ratio * (Carlos_ratio * Eugene_speed)) :
    Mia_speed = 5 :=
by
  sorry

end Mia_biking_speed_l27_27866


namespace arithmetic_mean_q_r_l27_27013

theorem arithmetic_mean_q_r (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) 
  (h3 : r - p = 24) : 
  (q + r) / 2 = 22 := 
by
  sorry

end arithmetic_mean_q_r_l27_27013


namespace smallest_positive_value_l27_27467

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℚ), k = (a^2 + b^2) / (a^2 - b^2) + (a^2 - b^2) / (a^2 + b^2) ∧ k = 2 :=
sorry

end smallest_positive_value_l27_27467


namespace largest_k_for_3_in_g_l27_27863

theorem largest_k_for_3_in_g (k : ℝ) :
  (∃ x : ℝ, 2*x^2 - 8*x + k = 3) ↔ k ≤ 11 :=
by
  sorry

end largest_k_for_3_in_g_l27_27863


namespace find_train_parameters_l27_27059

-- Definitions based on the problem statement
def bridge_length : ℕ := 1000
def time_total : ℕ := 60
def time_on_bridge : ℕ := 40
def speed_train (x : ℕ) := (40 * x = bridge_length)
def length_train (x y : ℕ) := (60 * x = bridge_length + y)

-- Stating the problem to be proved
theorem find_train_parameters (x y : ℕ) (h₁ : speed_train x) (h₂ : length_train x y) :
  x = 20 ∧ y = 200 :=
sorry

end find_train_parameters_l27_27059


namespace smallest_four_digit_divisible_by_53_l27_27987

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27987


namespace sum_of_fifth_powers_l27_27272

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l27_27272


namespace population_present_l27_27816

theorem population_present (P : ℝ) (h : P * (1.1)^3 = 79860) : P = 60000 :=
sorry

end population_present_l27_27816


namespace pete_travel_time_l27_27560

-- Definitions for the given conditions
def map_distance := 5.0          -- in inches
def scale := 0.05555555555555555 -- in inches per mile
def speed := 60.0                -- in miles per hour
def real_distance := map_distance / scale

-- The theorem to state the proof problem
theorem pete_travel_time : 
  real_distance = 90 → -- Based on condition deduced from earlier
  real_distance / speed = 1.5 := 
by 
  intro h1
  rw[h1]
  norm_num
  sorry

end pete_travel_time_l27_27560


namespace intersection_A_B_intersection_CA_B_intersection_CA_CB_l27_27161

-- Set definitions
def A := {x : ℝ | -5 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | x < -2 ∨ x > 4}
def C_A := {x : ℝ | x < -5 ∨ x > 3}  -- Complement of A
def C_B := {x : ℝ | -2 ≤ x ∧ x ≤ 4}  -- Complement of B

-- Lean statements proving the intersections
theorem intersection_A_B : {x : ℝ | -5 ≤ x ∧ x ≤ 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | -5 ≤ x ∧ x < -2} :=
by sorry

theorem intersection_CA_B : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | x < -2 ∨ x > 4} = {x : ℝ | x < -5 ∨ x > 4} :=
by sorry

theorem intersection_CA_CB : {x : ℝ | x < -5 ∨ x > 3} ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 4} = {x : ℝ | 3 < x ∧ x ≤ 4} :=
by sorry

end intersection_A_B_intersection_CA_B_intersection_CA_CB_l27_27161


namespace simplify_expr_l27_27347

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l27_27347


namespace sequence_term_is_100th_term_l27_27790

theorem sequence_term_is_100th_term (a : ℕ → ℝ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  (∃ n : ℕ, a n = 2 / 101) ∧ ((∃ n : ℕ, a n = 2 / 101) → n = 100) :=
by
  sorry

end sequence_term_is_100th_term_l27_27790


namespace smallest_integer_in_set_l27_27310

theorem smallest_integer_in_set : 
  ∀ (n : ℤ), (n + 6 < 2 * (n + 3)) → n ≥ 1 :=
by 
  sorry

end smallest_integer_in_set_l27_27310


namespace net_profit_correct_l27_27942

-- Define the conditions
def unit_price : ℝ := 1.25
def selling_price : ℝ := 12
def num_patches : ℕ := 100

-- Define the required total cost
def total_cost : ℝ := num_patches * unit_price

-- Define the required total revenue
def total_revenue : ℝ := num_patches * selling_price

-- Define the net profit calculation
def net_profit : ℝ := total_revenue - total_cost

-- The theorem we need to prove
theorem net_profit_correct : net_profit = 1075 := by
    sorry

end net_profit_correct_l27_27942


namespace time_to_fill_tank_l27_27825

noncomputable def pipeA_rate := (1 : ℚ) / 10
noncomputable def pipeB_rate := (1 : ℚ) / 15
noncomputable def pipeC_rate := - (1 : ℚ) / 20
noncomputable def combined_rate := pipeA_rate + pipeB_rate + pipeC_rate
noncomputable def time_to_fill := 1 / combined_rate

theorem time_to_fill_tank : time_to_fill = 60 / 7 :=
by
  sorry

end time_to_fill_tank_l27_27825


namespace shaniqua_earnings_correct_l27_27944

noncomputable def calc_earnings : ℝ :=
  let haircut_tuesday := 5 * 10
  let haircut_normal := 5 * 12
  let styling_vip := (6 * 25) * (1 - 0.2)
  let styling_regular := 4 * 25
  let coloring_friday := (7 * 35) * (1 - 0.15)
  let coloring_normal := 3 * 35
  let treatment_senior := (3 * 50) * (1 - 0.1)
  let treatment_other := 4 * 50
  haircut_tuesday + haircut_normal + styling_vip + styling_regular + coloring_friday + coloring_normal + treatment_senior + treatment_other

theorem shaniqua_earnings_correct : calc_earnings = 978.25 := by
  sorry

end shaniqua_earnings_correct_l27_27944


namespace money_made_per_minute_l27_27516

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end money_made_per_minute_l27_27516


namespace average_rounds_rounded_eq_4_l27_27958

def rounds_distribution : List (Nat × Nat) := [(1, 4), (2, 3), (4, 4), (5, 2), (6, 6)]

def total_rounds : Nat := rounds_distribution.foldl (λ acc (rounds, golfers) => acc + rounds * golfers) 0

def total_golfers : Nat := rounds_distribution.foldl (λ acc (_, golfers) => acc + golfers) 0

def average_rounds : Float := total_rounds.toFloat / total_golfers.toFloat

theorem average_rounds_rounded_eq_4 : Float.round average_rounds = 4 := by
  sorry

end average_rounds_rounded_eq_4_l27_27958


namespace f_g_of_3_l27_27794

def f (x : ℤ) : ℤ := 2 * x + 3
def g (x : ℤ) : ℤ := x^3 - 6

theorem f_g_of_3 : f (g 3) = 45 := by
  sorry

end f_g_of_3_l27_27794


namespace reduced_price_after_discount_l27_27042

theorem reduced_price_after_discount (P R : ℝ) (h1 : R = 0.8 * P) (h2 : 1500 / R - 1500 / P = 10) :
  R = 30 := 
by
  sorry

end reduced_price_after_discount_l27_27042


namespace sum_of_final_numbers_l27_27966

variable {x y T : ℝ}

theorem sum_of_final_numbers (h : x + y = T) : 3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by 
  -- The place for the proof steps, which will later be filled
  sorry

end sum_of_final_numbers_l27_27966


namespace geom_seq_product_l27_27876

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  2 * a 3 - (a 8) ^ 2 + 2 * a 13 = 0

def geometric_seq (b : ℕ → ℤ) (a8 : ℤ) : Prop :=
  b 8 = a8

theorem geom_seq_product (a b : ℕ → ℤ) (a8 : ℤ) 
  (h1 : arithmetic_seq a)
  (h2 : geometric_seq b a8)
  (h3 : a8 = 4)
: b 4 * b 12 = 16 := sorry

end geom_seq_product_l27_27876


namespace increase_p_does_not_always_increase_equal_points_l27_27692

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l27_27692


namespace Guido_costs_42840_l27_27637

def LightningMcQueenCost : ℝ := 140000
def MaterCost : ℝ := 0.1 * LightningMcQueenCost
def SallyCostBeforeModifications : ℝ := 3 * MaterCost
def SallyCostAfterModifications : ℝ := SallyCostBeforeModifications + 0.2 * SallyCostBeforeModifications
def GuidoCost : ℝ := SallyCostAfterModifications - 0.15 * SallyCostAfterModifications

theorem Guido_costs_42840 :
  GuidoCost = 42840 :=
sorry

end Guido_costs_42840_l27_27637


namespace number_of_chicks_is_8_l27_27969

-- Define the number of total chickens
def total_chickens : ℕ := 15

-- Define the number of hens
def hens : ℕ := 3

-- Define the number of roosters
def roosters : ℕ := total_chickens - hens

-- Define the number of chicks
def chicks : ℕ := roosters - 4

-- State the main theorem
theorem number_of_chicks_is_8 : chicks = 8 := 
by
  -- the solution follows from the given definitions and conditions
  sorry

end number_of_chicks_is_8_l27_27969


namespace total_toes_on_bus_l27_27332

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l27_27332


namespace denominator_of_second_fraction_l27_27701

theorem denominator_of_second_fraction :
  let a := 2007
  let b := 2999
  let c := 8001
  let d := 2001
  let e := 3999
  let sum := 3.0035428163476343
  let first_fraction := (2007 : ℝ) / 2999
  let third_fraction := (2001 : ℝ) / 3999
  ∃ x : ℤ, (first_fraction + (8001 : ℝ) / x + third_fraction) = 3.0035428163476343 ∧ x = 4362 := 
by
  sorry

end denominator_of_second_fraction_l27_27701


namespace min_students_orchestra_l27_27855

theorem min_students_orchestra (n : ℕ) 
  (h1 : n % 9 = 0)
  (h2 : n % 10 = 0)
  (h3 : n % 11 = 0) : 
  n ≥ 990 ∧ ∃ k, n = 990 * k :=
by
  sorry

end min_students_orchestra_l27_27855


namespace inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l27_27706

theorem inequality_solution_set (x : ℝ) : (|x - 1| + |2 * x + 5| < 8) ↔ (-4 < x ∧ x < 4 / 3) :=
by
  sorry

theorem ab2_bc_ca_a3b_ge_1_4 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^2 / (b + 3 * c) + b^2 / (c + 3 * a) + c^2 / (a + 3 * b) ≥ 1 / 4) :=
by
  sorry

end inequality_solution_set_ab2_bc_ca_a3b_ge_1_4_l27_27706


namespace find_x_condition_l27_27762

theorem find_x_condition :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  intros x h
  have num_zero : x^2 - 1 = 0 := by
    -- Proof that the numerator is zero
    sorry
  have denom_nonzero : x ≠ -1 := by
    -- Proof that the denominator is non-zero
    sorry
  have x_solves : x = 1 := by
    -- Final proof to show x = 1
    sorry
  exact x_solves

end find_x_condition_l27_27762


namespace decreasing_function_range_l27_27304

theorem decreasing_function_range (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end decreasing_function_range_l27_27304


namespace subtraction_decimal_nearest_hundredth_l27_27356

theorem subtraction_decimal_nearest_hundredth : 
  (845.59 - 249.27 : ℝ) = 596.32 :=
by
  sorry

end subtraction_decimal_nearest_hundredth_l27_27356


namespace consecutive_numbers_probability_l27_27442

theorem consecutive_numbers_probability :
  let total_ways := Nat.choose 20 5
  let non_consecutive_ways := Nat.choose 16 5
  let probability_of_non_consecutive := (non_consecutive_ways : ℚ) / (total_ways : ℚ)
  let probability_of_consecutive := 1 - probability_of_non_consecutive
  probability_of_consecutive = 232 / 323 :=
by
  sorry

end consecutive_numbers_probability_l27_27442


namespace total_pencils_children_l27_27820

theorem total_pencils_children :
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  c1 + c2 + c3 + c4 + c5 = 60 :=
by
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  show c1 + c2 + c3 + c4 + c5 = 60
  sorry

end total_pencils_children_l27_27820


namespace gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l27_27980

theorem gcd_3_pow_1007_minus_1_3_pow_1018_minus_1 :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 177146 :=
by
  -- Proof follows from the Euclidean algorithm and factoring, skipping the proof here.
  sorry

end gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l27_27980


namespace number_of_pieces_of_bubble_gum_l27_27538

theorem number_of_pieces_of_bubble_gum (cost_per_piece total_cost : ℤ) (h1 : cost_per_piece = 18) (h2 : total_cost = 2448) :
  total_cost / cost_per_piece = 136 :=
by
  rw [h1, h2]
  norm_num

end number_of_pieces_of_bubble_gum_l27_27538


namespace turtles_still_on_sand_l27_27407

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l27_27407


namespace mia_has_110_l27_27501

def darwin_money : ℕ := 45
def mia_money (darwin_money : ℕ) : ℕ := 2 * darwin_money + 20

theorem mia_has_110 :
  mia_money darwin_money = 110 := sorry

end mia_has_110_l27_27501


namespace seventh_triangular_number_eq_28_l27_27365

noncomputable def triangular_number (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem seventh_triangular_number_eq_28 :
  triangular_number 7 = 28 :=
by
  sorry

end seventh_triangular_number_eq_28_l27_27365


namespace time_after_1456_minutes_l27_27182

noncomputable def hours_in_minutes := 1456 / 60
noncomputable def minutes_remainder := 1456 % 60

def current_time : Nat := 6 * 60  -- 6:00 a.m. in minutes
def added_time : Nat := current_time + 1456

def six_sixteen_am : Nat := (6 * 60) + 16  -- 6:16 a.m. in minutes the next day

theorem time_after_1456_minutes : added_time % (24 * 60) = six_sixteen_am :=
by
  sorry

end time_after_1456_minutes_l27_27182


namespace possible_to_position_guards_l27_27484

-- Define the conditions
def guard_sees (d : ℝ) : Prop := d = 100

-- Prove that it is possible to arrange guards around a point object so that neither the object nor the guards can be approached unnoticed
theorem possible_to_position_guards (num_guards : ℕ) (d : ℝ) (h : guard_sees d) : 
  (0 < num_guards) → 
  (∀ θ : ℕ, θ < num_guards → (θ * (360 / num_guards)) < 360) → 
  True :=
by 
  -- Details of the proof would go here
  sorry

end possible_to_position_guards_l27_27484


namespace percent_absent_is_correct_l27_27508

theorem percent_absent_is_correct (total_students boys girls absent_boys absent_girls : ℝ) 
(h1 : total_students = 100)
(h2 : boys = 50)
(h3 : girls = 50)
(h4 : absent_boys = boys * (1 / 5))
(h5 : absent_girls = girls * (1 / 4)):
  (absent_boys + absent_girls) / total_students * 100 = 22.5 :=
by 
  sorry

end percent_absent_is_correct_l27_27508


namespace cows_relationship_l27_27828

theorem cows_relationship (H : ℕ) (W : ℕ) (T : ℕ) (hcows : W = 17) (tcows : T = 70) (together : H + W = T) : H = 53 :=
by
  rw [hcows, tcows] at together
  linarith
  -- sorry

end cows_relationship_l27_27828


namespace simplify_and_rationalize_denominator_l27_27344

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l27_27344


namespace distance_is_correct_l27_27856

noncomputable def distance_from_center_to_plane
  (O : Point)
  (radius : ℝ)
  (vertices : Point × Point × Point)
  (side_lengths : (ℝ × ℝ × ℝ)) :
  ℝ :=
  8.772

theorem distance_is_correct
  (O : Point)
  (radius : ℝ)
  (A B C : Point)
  (h_radius : radius = 10)
  (h_sides : side_lengths = (17, 17, 16))
  (vertices := (A, B, C)) :
  distance_from_center_to_plane O radius vertices side_lengths = 8.772 := by
  sorry

end distance_is_correct_l27_27856


namespace ratio_x_w_l27_27251

variable {x y z w : ℕ}

theorem ratio_x_w (h1 : x / y = 24) (h2 : z / y = 8) (h3 : z / w = 1 / 12) : x / w = 1 / 4 := by
  sorry

end ratio_x_w_l27_27251


namespace probability_roots_real_l27_27067

-- Define the polynomial
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + 3*b*x^3 + (3*b - 5)*x^2 + (-6*b + 4)*x - 3

-- Define the intervals for b
def interval_b1 := Set.Icc (-(15:ℝ)) (20:ℝ)
def interval_b2 := Set.Icc (-(15:ℝ)) (-2/3)
def interval_b3 := Set.Icc (4/3) (20:ℝ)

-- Calculate the lengths of the intervals
def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def length_b1 := length_interval (-(15:ℝ)) (20:ℝ)
noncomputable def length_b2 := length_interval (-(15:ℝ)) (-2/3)
noncomputable def length_b3 := length_interval (4/3) (20:ℝ)
noncomputable def effective_length := length_b2 + length_b3

-- The probability is the ratio of effective lengths
noncomputable def probability := effective_length / length_b1

-- The theorem we want to prove
theorem probability_roots_real : probability = 33/35 :=
  sorry

end probability_roots_real_l27_27067


namespace Maria_drove_approximately_517_miles_l27_27929

noncomputable def carRentalMaria (daily_rate per_mile_charge discount_rate insurance_rate rental_duration total_invoice : ℝ) (discount_threshold : ℕ) : ℝ :=
  let total_rental_cost := rental_duration * daily_rate
  let discount := if rental_duration ≥ discount_threshold then discount_rate * total_rental_cost else 0
  let discounted_cost := total_rental_cost - discount
  let insurance_cost := rental_duration * insurance_rate
  let cost_without_mileage := discounted_cost + insurance_cost
  let mileage_cost := total_invoice - cost_without_mileage
  mileage_cost / per_mile_charge

noncomputable def approx_equal (a b : ℝ) (epsilon : ℝ := 1) : Prop :=
  abs (a - b) < epsilon

theorem Maria_drove_approximately_517_miles :
  approx_equal (carRentalMaria 35 0.09 0.10 5 4 192.50 3) 517 :=
by
  sorry

end Maria_drove_approximately_517_miles_l27_27929


namespace find_ratio_l27_27149

-- Define the geometric sequence properties and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions stated in the problem
axiom h₁ : a 5 * a 11 = 3
axiom h₂ : a 3 + a 13 = 4

-- The goal is to find the values of a_15 / a_5
theorem find_ratio (h₁ : a 5 * a 11 = 3) (h₂ : a 3 + a 13 = 4) :
  ∃ r : ℝ, r = a 15 / a 5 ∧ (r = 3 ∨ r = 1 / 3) :=
sorry

end find_ratio_l27_27149


namespace germination_probability_l27_27962

open Nat

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_of_success (p : ℚ) (k : ℕ) (n : ℕ) : ℚ :=
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem germination_probability :
  probability_of_success 0.9 5 7 = 0.124 := by
  sorry

end germination_probability_l27_27962


namespace simplify_and_evaluate_l27_27643

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/6) : 
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5 / 2 := by
  rw [h]
  sorry

end simplify_and_evaluate_l27_27643


namespace simplify_and_rationalize_denominator_l27_27343

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l27_27343


namespace remainder_when_55_times_57_divided_by_8_l27_27831

theorem remainder_when_55_times_57_divided_by_8 :
  (55 * 57) % 8 = 7 :=
by
  -- Insert the proof here
  sorry

end remainder_when_55_times_57_divided_by_8_l27_27831


namespace no_such_rectangle_exists_l27_27244

theorem no_such_rectangle_exists :
  ¬(∃ (x y : ℝ), (∃ a b c d : ℕ, x = a + b * Real.sqrt 3 ∧ y = c + d * Real.sqrt 3) ∧ 
                (x * y = (3 * Real.sqrt 3) / 2 + n * (Real.sqrt 3 / 2))) :=
sorry

end no_such_rectangle_exists_l27_27244


namespace product_mod_7_l27_27657

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l27_27657


namespace find_multiple_l27_27008

-- Given conditions
variables (P W m : ℕ)
variables (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2)

-- The statement to prove
theorem find_multiple (P W m : ℕ) (h1 : P * 24 = W) (h2 : m * P * 6 = W / 2) : m = 4 :=
by
  sorry

end find_multiple_l27_27008


namespace solution_set_of_inequality_l27_27186

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ (0 < x ∧ x < 1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l27_27186


namespace cost_to_produce_program_l27_27711

theorem cost_to_produce_program
  (advertisement_revenue : ℝ)
  (number_of_copies : ℝ)
  (price_per_copy : ℝ)
  (desired_profit : ℝ)
  (total_revenue : ℝ)
  (revenue_from_sales : ℝ)
  (cost_to_produce : ℝ) :
  advertisement_revenue = 15000 →
  number_of_copies = 35000 →
  price_per_copy = 0.5 →
  desired_profit = 8000 →
  total_revenue = advertisement_revenue + desired_profit →
  revenue_from_sales = number_of_copies * price_per_copy →
  total_revenue = revenue_from_sales + cost_to_produce →
  cost_to_produce = 5500 :=
by
  sorry

end cost_to_produce_program_l27_27711


namespace tangent_line_sum_l27_27470

theorem tangent_line_sum (a b : ℝ) :
  (∃ x₀ : ℝ, (e^(x₀ - 1) = 1) ∧ (x₀ + a = e^(x₀-1) * (1 - x₀) - b + 1)) → a + b = 1 :=
by
  sorry

end tangent_line_sum_l27_27470


namespace molecular_weight_compound_l27_27542

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999
def atomic_weight_N : ℝ := 14.007
def atomic_weight_Cl : ℝ := 35.453

def molecular_weight (nH nC nO nN nCl : ℕ) : ℝ :=
  nH * atomic_weight_H + nC * atomic_weight_C + nO * atomic_weight_O + nN * atomic_weight_N + nCl * atomic_weight_Cl

theorem molecular_weight_compound :
  molecular_weight 4 2 3 1 2 = 160.964 := by
  sorry

end molecular_weight_compound_l27_27542


namespace range_of_x_of_sqrt_x_plus_3_l27_27148

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l27_27148


namespace mass_percentage_of_Br_in_BaBr2_l27_27203

theorem mass_percentage_of_Br_in_BaBr2 :
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  mass_percentage_Br = 53.80 :=
by
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  sorry

end mass_percentage_of_Br_in_BaBr2_l27_27203


namespace fastest_route_time_l27_27009

theorem fastest_route_time (d1 d2 : ℕ) (s1 s2 : ℕ) (h1 : d1 = 1500) (h2 : d2 = 750) (h3 : s1 = 75) (h4 : s2 = 25) :
  min (d1 / s1) (d2 / s2) = 20 := by
  sorry

end fastest_route_time_l27_27009


namespace systematic_sampling_first_group_number_l27_27857

-- Given conditions
def total_students := 160
def group_size := 8
def groups := total_students / group_size
def number_in_16th_group := 126

-- Theorem Statement
theorem systematic_sampling_first_group_number :
  ∃ x : ℕ, (120 + x = number_in_16th_group) ∧ x = 6 :=
by
  -- Proof can be filled here
  sorry

end systematic_sampling_first_group_number_l27_27857


namespace q_is_false_l27_27610

variable {p q : Prop}

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end q_is_false_l27_27610


namespace savings_for_mother_l27_27635

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l27_27635


namespace equal_points_probability_not_always_increasing_l27_27694

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l27_27694


namespace product_remainder_mod_7_l27_27680

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l27_27680


namespace quarts_of_water_required_l27_27600

-- Define the ratio of water to juice
def ratio_water_to_juice : Nat := 5 / 3

-- Define the total punch to prepare in gallons
def total_punch_in_gallons : Nat := 2

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : Nat := 4

-- Define the total number of parts
def total_parts : Nat := 5 + 3

-- Define the total punch in quarts
def total_punch_in_quarts : Nat := total_punch_in_gallons * quarts_per_gallon

-- Define the amount of water per part
def quarts_per_part : Nat := total_punch_in_quarts / total_parts

-- Prove the required amount of water in quarts
theorem quarts_of_water_required : quarts_per_part * 5 = 5 := 
by
  -- Proof is omitted, represented by sorry
  sorry

end quarts_of_water_required_l27_27600


namespace sum_of_digits_of_m_l27_27373

-- Define the logarithms and intermediate expressions
noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_m :
  ∃ m : ℕ, log_b 3 (log_b 81 m) = log_b 9 (log_b 9 m) ∧ sum_of_digits m = 10 := 
by
  sorry

end sum_of_digits_of_m_l27_27373


namespace last_rope_length_l27_27823

def totalRopeLength : ℝ := 35
def rope1 : ℝ := 8
def rope2 : ℝ := 20
def rope3a : ℝ := 2
def rope3b : ℝ := 2
def rope3c : ℝ := 2
def knotLoss : ℝ := 1.2
def numKnots : ℝ := 4

theorem last_rope_length : 
  (35 + (4 * 1.2)) = (8 + 20 + 2 + 2 + 2 + x) → (x = 5.8) :=
sorry

end last_rope_length_l27_27823


namespace total_toes_on_bus_l27_27333

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l27_27333


namespace arithmetic_series_remainder_l27_27077

noncomputable def arithmetic_series_sum_mod (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d) / 2) % 10

theorem arithmetic_series_remainder :
  let a := 3
  let d := 5
  let n := 21
  arithmetic_series_sum_mod a d n = 3 :=
by
  sorry

end arithmetic_series_remainder_l27_27077


namespace money_distribution_l27_27409

variable (A B C : ℝ)

theorem money_distribution
  (h₁ : A + B + C = 500)
  (h₂ : A + C = 200)
  (h₃ : C = 60) :
  B + C = 360 :=
by
  sorry

end money_distribution_l27_27409


namespace graph_does_not_pass_second_quadrant_l27_27450

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h₀ : 1 < a) (h₁ : b < -1) : 
∀ x : ℝ, ¬ (y = a^x + b ∧ y > 0 ∧ x < 0) :=
by
  sorry

end graph_does_not_pass_second_quadrant_l27_27450


namespace sum_of_fifth_powers_l27_27275

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l27_27275


namespace floor_ceil_diff_l27_27286

theorem floor_ceil_diff (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) : ⌊x⌋ + x - ⌈x⌉ = x - 1 :=
sorry

end floor_ceil_diff_l27_27286


namespace find_y_l27_27418

variable (A B C : Point)

def carla_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees clockwise about point B lands at point C
  sorry

def devon_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees counterclockwise about point B lands at point C
  sorry

theorem find_y
  (h1 : carla_rotate 690 A B C)
  (h2 : ∀ y, devon_rotate y A B C)
  (h3 : y < 360) :
  ∃ y, y = 30 :=
by
  sorry

end find_y_l27_27418


namespace consecutive_integers_divisor_l27_27100

theorem consecutive_integers_divisor {m n : ℕ} (hm : m < n) (a : ℕ) :
  ∃ i j : ℕ, i ≠ j ∧ (a + i) * (a + j) % (m * n) = 0 :=
by
  sorry

end consecutive_integers_divisor_l27_27100


namespace parabola_intersections_l27_27697

open Real

-- Definition of the two parabolas
def parabola1 (x : ℝ) : ℝ := 3*x^2 - 6*x + 2
def parabola2 (x : ℝ) : ℝ := 9*x^2 - 4*x - 5

-- Theorem stating the intersections are (-7/3, 9) and (0.5, -0.25)
theorem parabola_intersections : 
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} =
  {(-7/3, 9), (0.5, -0.25)} :=
by 
  sorry

end parabola_intersections_l27_27697


namespace problem_statement_l27_27282

variable (f g : ℝ → ℝ)
variable (f' g' : ℝ → ℝ)
variable (a b : ℝ)

theorem problem_statement (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, HasDerivAt g (g' x) x)
                         (h3 : ∀ x, f' x < g' x)
                         (h4 : a = Real.log 2 / Real.log 5)
                         (h5 : b = Real.log 3 / Real.log 8) :
                         f a + g b > g a + f b := 
     sorry

end problem_statement_l27_27282


namespace work_done_l27_27338

-- Definitions of given conditions
def is_cyclic_process (gas: Type) (a b c: gas) : Prop := sorry
def isothermal_side (a b: Type) : Prop := sorry
def line_through_origin (b c: Type) : Prop := sorry
def parabolic_arc_through_origin (c a: Type) : Prop := sorry

def temperature_equality (T_a T_c: ℝ) : Prop :=
  T_a = T_c

def half_pressure (P_a P_c: ℝ) : Prop :=
  P_a = 0.5 * P_c

-- Main theorem statement
theorem work_done (T_0 P_a P_c: ℝ) (a b c: Type) 
  (H_cycle: is_cyclic_process gas a b c)
  (H_isothermal: isothermal_side a b)
  (H_line_origin: line_through_origin b c)
  (H_parabolic_arc: parabolic_arc_through_origin c a)
  (H_temp_eq: temperature_equality T_0 320)
  (H_pressure_half: half_pressure P_a P_c) :
  (work_done gas a b c) = 665 := sorry

end work_done_l27_27338


namespace balcony_more_than_orchestra_l27_27233

-- Conditions
def total_tickets (O B : ℕ) : Prop := O + B = 340
def total_cost (O B : ℕ) : Prop := 12 * O + 8 * B = 3320

-- The statement we need to prove based on the conditions
theorem balcony_more_than_orchestra (O B : ℕ) (h1 : total_tickets O B) (h2 : total_cost O B) :
  B - O = 40 :=
sorry

end balcony_more_than_orchestra_l27_27233


namespace sarah_class_choices_l27_27916

-- Conditions 
def total_classes : ℕ := 10
def choose_classes : ℕ := 4
def specific_classes : ℕ := 2

-- Statement
theorem sarah_class_choices : 
  ∃ (n : ℕ), n = Nat.choose (total_classes - specific_classes) 3 ∧ n = 56 :=
by 
  sorry

end sarah_class_choices_l27_27916


namespace range_x_plus_y_l27_27757

theorem range_x_plus_y (x y: ℝ) (h: x^2 + y^2 - 4 * x + 3 = 0) : 
  2 - Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + Real.sqrt 2 :=
by 
  sorry

end range_x_plus_y_l27_27757


namespace simplify_and_rationalize_denominator_l27_27345

noncomputable def problem (x : ℚ) := 
  x = 1 / (1 + 1 / (Real.sqrt 5 + 2))

theorem simplify_and_rationalize_denominator (x : ℚ) (h: problem x) :
  x = (Real.sqrt 5 - 1) / 4 :=
by
  sorry

end simplify_and_rationalize_denominator_l27_27345


namespace original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l27_27364

-- Condition declarations
variables (x y : ℕ)
variables (hx : x > 0) (hy : y > 0)
variables (h_sum : x + y = 25)
variables (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196)

-- Lean 4 statements for the proof problem
theorem original_rectangle_perimeter (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 25) : 2 * (x + y) = 50 := by
  sorry

theorem difference_is_multiple_of_7 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_diff : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) : 7 ∣ ((x + 5) * (y + 5) - (x - 2) * (y - 2)) := by
  sorry

theorem relationship_for_seamless_combination (x y : ℕ) (h_sum : x + y = 25) (h_seamless : (x+5)*(y+5) = (x*(y+5))) : x = y + 5 := by
  sorry

end original_rectangle_perimeter_difference_is_multiple_of_7_relationship_for_seamless_combination_l27_27364


namespace proportion_margin_l27_27121

theorem proportion_margin (S M C : ℝ) (n : ℝ) (hM : M = S / n) (hC : C = (1 - 1 / n) * S) :
  M / C = 1 / (n - 1) :=
by
  sorry

end proportion_margin_l27_27121


namespace intersection_P_Q_l27_27769

def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

theorem intersection_P_Q : P ∩ Q = {y | y ≤ 2} :=
sorry

end intersection_P_Q_l27_27769


namespace intersection_M_N_l27_27594

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def N : Set ℝ := { y | 0 < y }

theorem intersection_M_N : (M ∩ N) = { z | 0 < z ∧ z ≤ 2 } :=
by
  -- proof to be completed
  sorry

end intersection_M_N_l27_27594


namespace x4_y4_value_l27_27586

theorem x4_y4_value (x y : ℝ) (h1 : x^4 + x^2 = 3) (h2 : y^4 - y^2 = 3) : x^4 + y^4 = 7 := by
  sorry

end x4_y4_value_l27_27586


namespace julian_notes_problem_l27_27305

theorem julian_notes_problem (x y : ℤ) (h1 : 3 * x + 4 * y = 151) (h2 : x = 19 ∨ y = 19) :
  x = 25 ∨ y = 25 := 
by
  sorry

end julian_notes_problem_l27_27305


namespace least_number_of_teams_l27_27064

theorem least_number_of_teams
  (total_athletes : ℕ)
  (max_team_size : ℕ)
  (h_total : total_athletes = 30)
  (h_max : max_team_size = 12) :
  ∃ (number_of_teams : ℕ) (team_size : ℕ),
    number_of_teams * team_size = total_athletes ∧
    team_size ≤ max_team_size ∧
    number_of_teams = 3 :=
by
  sorry

end least_number_of_teams_l27_27064


namespace evaluate_propositions_l27_27283

variable (x y : ℝ)

def p : Prop := (x > y) → (-x < -y)
def q : Prop := (x < y) → (x^2 > y^2)

theorem evaluate_propositions : (p x y ∨ q x y) ∧ (p x y ∧ ¬q x y) := by
  -- Correct answer: \( \boxed{\text{C}} \)
  sorry

end evaluate_propositions_l27_27283


namespace sum_of_fifth_powers_l27_27273

variables {a b c d : ℝ}

theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : a^5 + b^5 = c^5 + d^5 := 
sorry

end sum_of_fifth_powers_l27_27273


namespace evaluate_expr_l27_27046

noncomputable def expr : ℚ :=
  2013 * (5.7 * 4.2 + (21 / 5) * 4.3) / ((14 / 73) * 15 + (5 / 73) * 177 + 656)

theorem evaluate_expr : expr = 126 := by
  sorry

end evaluate_expr_l27_27046


namespace gandalf_reachability_l27_27534

theorem gandalf_reachability :
  ∀ (k : ℕ), ∃ (s : ℕ → ℕ) (m : ℕ), (s 0 = 1) ∧ (s m = k) ∧ (∀ i < m, s (i + 1) = 2 * s i ∨ s (i + 1) = 3 * s i + 1) := 
by
  sorry

end gandalf_reachability_l27_27534


namespace find_middle_number_l27_27532

theorem find_middle_number (x y z : ℤ) (h1 : x + y = 22) (h2 : x + z = 29) (h3 : y + z = 37) (h4 : x < y) (h5 : y < z) : y = 15 :=
by
  sorry

end find_middle_number_l27_27532


namespace arrangement_problem_l27_27025
   
   def numberOfArrangements (n : Nat) : Nat :=
     n.factorial

   def exclusiveArrangements (total people : Nat) (positions : Nat) : Nat :=
     (positions.choose 2) * (total - 2).factorial

   theorem arrangement_problem : 
     (numberOfArrangements 5) - (exclusiveArrangements 5 3) = 84 := 
   by
     sorry
   
end arrangement_problem_l27_27025


namespace marbles_selection_l27_27155

theorem marbles_selection : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ marbles : Finset ℕ, marbles.card = 15 ∧
  ∃ rgb : Finset ℕ, rgb ⊆ marbles ∧ rgb.card = 3 ∧
  ∃ yellow : ℕ, yellow ∈ marbles ∧ yellow ∉ rgb ∧ 
  ∀ (selection : Finset ℕ), selection.card = 5 →
  (∃ red green blue : ℕ, red ∈ rgb ∧ green ∈ rgb ∧ blue ∈ rgb ∧ 
  (red ∈ selection ∨ green ∈ selection ∨ blue ∈ selection) ∧ yellow ∉ selection) → 
  (selection.card = 5) :=
by
  sorry

end marbles_selection_l27_27155


namespace math_problem_l27_27208

theorem math_problem :
  ( (1 / 3 * 9) ^ 2 * (1 / 27 * 81) ^ 2 * (1 / 243 * 729) ^ 2) = 729 := by
  sorry

end math_problem_l27_27208


namespace distinct_arrangements_on_3x3_grid_l27_27307

def is_valid_position (pos : ℤ × ℤ) : Prop :=
  0 ≤ pos.1 ∧ pos.1 < 3 ∧ 0 ≤ pos.2 ∧ pos.2 < 3

def rotations_equiv (pos1 pos2 : ℤ × ℤ) : Prop :=
  pos1 = pos2 ∨ pos1 = (2 - pos2.2, pos2.1) ∨ pos1 = (2 - pos2.1, 2 - pos2.2) ∨ pos1 = (pos2.2, 2 - pos2.1)

def distinct_positions_count (grid_size : ℕ) : ℕ :=
  10  -- given from the problem solution

theorem distinct_arrangements_on_3x3_grid : distinct_positions_count 3 = 10 := sorry

end distinct_arrangements_on_3x3_grid_l27_27307


namespace distance_AB_l27_27805

def A : ℝ := -1
def B : ℝ := 2023

theorem distance_AB : |B - A| = 2024 := by
  sorry

end distance_AB_l27_27805


namespace simplify_frac_and_find_cd_l27_27002

theorem simplify_frac_and_find_cd :
  ∀ (m : ℤ), ∃ (c d : ℤ), 
    (c * m + d = (6 * m + 12) / 3) ∧ (c = 2) ∧ (d = 4) ∧ (c / d = 1 / 2) :=
by
  sorry

end simplify_frac_and_find_cd_l27_27002


namespace find_d_l27_27922

variable {x1 x2 k d : ℝ}

axiom h₁ : x1 ≠ x2
axiom h₂ : 4 * x1^2 - k * x1 = d
axiom h₃ : 4 * x2^2 - k * x2 = d
axiom h₄ : x1 + x2 = 2

theorem find_d : d = -12 := by
  sorry

end find_d_l27_27922


namespace incorrect_calculation_l27_27836

theorem incorrect_calculation (a : ℝ) : (2 * a) ^ 3 ≠ 6 * a ^ 3 :=
by {
  sorry
}

end incorrect_calculation_l27_27836


namespace cake_heavier_than_bread_l27_27684

-- Definitions
def weight_of_7_cakes_eq_1950_grams (C : ℝ) := 7 * C = 1950
def weight_of_5_cakes_12_breads_eq_2750_grams (C B : ℝ) := 5 * C + 12 * B = 2750

-- Statement
theorem cake_heavier_than_bread (C B : ℝ)
  (h1 : weight_of_7_cakes_eq_1950_grams C)
  (h2 : weight_of_5_cakes_12_breads_eq_2750_grams C B) :
  C - B = 165.47 :=
by {
  sorry
}

end cake_heavier_than_bread_l27_27684


namespace car_speed_first_hour_l27_27964

theorem car_speed_first_hour (x : ℝ) (h1 : (x + 75) / 2 = 82.5) : x = 90 :=
sorry

end car_speed_first_hour_l27_27964


namespace sum_mnp_is_405_l27_27564

theorem sum_mnp_is_405 :
  let C1_radius := 4
  let C2_radius := 10
  let C3_radius := C1_radius + C2_radius
  let chord_length := (8 * Real.sqrt 390) / 7
  ∃ m n p : ℕ,
    m * Real.sqrt n / p = chord_length ∧
    m.gcd p = 1 ∧
    (∀ k : ℕ, k^2 ∣ n → k = 1) ∧
    m + n + p = 405 :=
by
  sorry

end sum_mnp_is_405_l27_27564


namespace mushrooms_weight_change_l27_27448

-- Conditions
variables (x W : ℝ)
variable (initial_weight : ℝ := 100 * x)
variable (dry_weight : ℝ := x)
variable (final_weight_dry : ℝ := 2 * W / 100)

-- Given fresh mushrooms have moisture content of 99%
-- and dried mushrooms have moisture content of 98%
theorem mushrooms_weight_change 
  (h1 : dry_weight = x) 
  (h2 : final_weight_dry = x / 0.02) 
  (h3 : W = x / 0.02) 
  (initial_weight : ℝ := 100 * x) : 
  2 * W = initial_weight / 2 :=
by
  -- This is a placeholder for the proof steps which we skip
  sorry

end mushrooms_weight_change_l27_27448


namespace find_c_l27_27602

theorem find_c (c : ℝ) (h : ∀ x : ℝ, ∃ a : ℝ, (x + a)^2 = x^2 + 200 * x + c) : c = 10000 :=
sorry

end find_c_l27_27602


namespace log_equivalence_l27_27884

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end log_equivalence_l27_27884


namespace parallelogram_area_correct_l27_27552

def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem parallelogram_area_correct :
  parallelogram_area 15 5 = 75 :=
by
  sorry

end parallelogram_area_correct_l27_27552


namespace ad_value_l27_27457

variable (a b c d : ℝ)

-- Conditions
def geom_seq := b^2 = a * c ∧ c^2 = b * d
def vertex_of_parabola := (b = 1 ∧ c = 2)

-- Question
theorem ad_value (h_geom : geom_seq a b c d) (h_vertex : vertex_of_parabola b c) : a * d = 2 := by
  sorry

end ad_value_l27_27457


namespace volume_in_cubic_meters_l27_27653

noncomputable def mass_condition : ℝ := 100 -- mass in kg
noncomputable def volume_per_gram : ℝ := 10 -- volume in cubic centimeters per gram
noncomputable def volume_per_kg : ℝ := volume_per_gram * 1000 -- volume in cubic centimeters per kg
noncomputable def mass_in_kg : ℝ := mass_condition

theorem volume_in_cubic_meters (h : mass_in_kg = 100)
    (v_per_kg : volume_per_kg = volume_per_gram * 1000) :
  (mass_in_kg * volume_per_kg) / 1000000 = 1 := by
  sorry

end volume_in_cubic_meters_l27_27653


namespace sum_first_n_terms_arithmetic_sequence_l27_27797

theorem sum_first_n_terms_arithmetic_sequence 
  (S : ℕ → ℕ) (m : ℕ) (h1 : S m = 2) (h2 : S (2 * m) = 10) :
  S (3 * m) = 24 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l27_27797


namespace value_of_r_for_n_3_l27_27925

theorem value_of_r_for_n_3 :
  ∀ (r s : ℕ), 
  (r = 4^s + 3 * s) → 
  (s = 2^3 + 2) → 
  r = 1048606 :=
by
  intros r s h1 h2
  sorry

end value_of_r_for_n_3_l27_27925


namespace range_of_independent_variable_l27_27139

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l27_27139


namespace correct_conclusions_l27_27459

-- Definitions for conditions
variables (a b c m : ℝ)
variables (h_parabola : ∀ x, a * x^2 + b * x + c ≥ 0)
variables (h_a_pos : 0 < a)
variables (h_axis : b = 2 * a)
variables (h_intersect : 0 < m ∧ m < 1)
variables (h_point : a * (1 / 2)^2 + b * (1 / 2) + c = 2)
variables (x1 x2 : ℝ)

-- Correct conclusions to prove
theorem correct_conclusions :
  (4 * a + c > 0) ∧ (∀ t : ℝ, a - b * t ≤ a * t^2 + b) :=
sorry

end correct_conclusions_l27_27459


namespace rectangle_area_theorem_l27_27230

def rectangle_area (d : ℝ) (area : ℝ) : Prop :=
  ∃ w : ℝ, 0 < w ∧ 9 * w^2 + w^2 = d^2 ∧ area = 3 * w^2

theorem rectangle_area_theorem (d : ℝ) : rectangle_area d (3 * d^2 / 10) :=
sorry

end rectangle_area_theorem_l27_27230


namespace trapezoid_area_l27_27791

-- Define the given conditions in the problem
variables (EF GH h EG FH : ℝ)
variables (EF_parallel_GH : true) -- EF and GH are parallel (not used in the calculation)
variables (EF_eq_70 : EF = 70)
variables (GH_eq_40 : GH = 40)
variables (h_eq_15 : h = 15)
variables (EG_eq_20 : EG = 20)
variables (FH_eq_25 : FH = 25)

-- Define the main theorem to prove
theorem trapezoid_area (EF GH h EG FH : ℝ) 
  (EF_eq_70 : EF = 70) 
  (GH_eq_40 : GH = 40) 
  (h_eq_15 : h = 15) 
  (EG_eq_20 : EG = 20) 
  (FH_eq_25 : FH = 25) : 
  0.5 * (EF + GH) * h = 825 := 
by 
  sorry

end trapezoid_area_l27_27791


namespace vector_subtraction_l27_27593

def a : ℝ × ℝ := (5, 3)
def b : ℝ × ℝ := (1, -2)
def scalar : ℝ := 2

theorem vector_subtraction :
  a.1 - scalar * b.1 = 3 ∧ a.2 - scalar * b.2 = 7 :=
by {
  -- here goes the proof
  sorry
}

end vector_subtraction_l27_27593


namespace simplify_expression_l27_27239

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 2) (h₂ : a ≠ -2) : 
  (2 * a / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2)) :=
by
  -- proof to be added
  sorry

end simplify_expression_l27_27239


namespace solve_equation1_solve_equation2_l27_27007

-- Define the two equations
def equation1 (x : ℝ) := 3 * x - 4 = -2 * (x - 1)
def equation2 (x : ℝ) := 1 + (2 * x + 1) / 3 = (3 * x - 2) / 2

-- The statements to prove
theorem solve_equation1 : ∃ x : ℝ, equation1 x ∧ x = 1.2 :=
by
  sorry

theorem solve_equation2 : ∃ x : ℝ, equation2 x ∧ x = 2.8 :=
by
  sorry

end solve_equation1_solve_equation2_l27_27007


namespace distinct_naturals_and_power_of_prime_l27_27624

theorem distinct_naturals_and_power_of_prime (a b : ℕ) (p k : ℕ) (h1 : a ≠ b) (h2 : a^2 + b ∣ b^2 + a) (h3 : ∃ (p : ℕ) (k : ℕ), b^2 + a = p^k) : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) :=
sorry

end distinct_naturals_and_power_of_prime_l27_27624


namespace dot_product_equals_6_l27_27597

-- Define the vectors
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication and addition
def scaled_added_vector : ℝ × ℝ := (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2)

-- Define the dot product
def dot_product : ℝ := scaled_added_vector.1 * vec_a.1 + scaled_added_vector.2 * vec_a.2

-- Assertion that the dot product is equal to 6
theorem dot_product_equals_6 : dot_product = 6 :=
by
  sorry

end dot_product_equals_6_l27_27597


namespace divisor_increase_by_10_5_l27_27804

def condition_one (n t : ℕ) : Prop :=
  n * (t + 7) = t * (n + 2)

def condition_two (n t z : ℕ) : Prop :=
  n * (t + z) = t * (n + 3)

theorem divisor_increase_by_10_5 (n t : ℕ) (hz : ℕ) (nz : n ≠ 0) (tz : t ≠ 0)
  (h1 : condition_one n t) (h2 : condition_two n t hz) : hz = 21 / 2 :=
by {
  sorry
}

end divisor_increase_by_10_5_l27_27804


namespace correct_statement_l27_27589

theorem correct_statement (a b : ℝ) (h_a : a ≥ 0) (h_b : b ≥ 0) : (a ≥ 0 ∧ b ≥ 0) :=
by
  exact ⟨h_a, h_b⟩

end correct_statement_l27_27589


namespace root_product_minus_sums_l27_27792

variable {b c : ℝ}

theorem root_product_minus_sums
  (h1 : 3 * b^2 + 5 * b - 2 = 0)
  (h2 : 3 * c^2 + 5 * c - 2 = 0)
  : (b - 1) * (c - 1) = 2 := 
by
  sorry

end root_product_minus_sums_l27_27792


namespace exceeding_fraction_l27_27084

def repeatingDecimal : ℚ := 8 / 33
def decimalFraction : ℚ := 6 / 25
def difference : ℚ := repeatingDecimal - decimalFraction

theorem exceeding_fraction :
  difference = 2 / 825 := by
  sorry

end exceeding_fraction_l27_27084


namespace sqrt_of_9_l27_27054

theorem sqrt_of_9 : Real.sqrt 9 = 3 :=
by 
  sorry

end sqrt_of_9_l27_27054


namespace equilateral_triangle_l27_27424

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ) (p R : ℝ)
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = p / (9 * R)) :
  a = b ∧ b = c ∧ a = c :=
sorry

end equilateral_triangle_l27_27424


namespace units_digit_same_units_and_tens_digit_same_l27_27218

theorem units_digit_same (n : ℕ) : 
  (∃ a : ℕ, a ∈ [0, 1, 5, 6] ∧ n % 10 = a ∧ n^2 % 10 = a) := 
sorry

theorem units_and_tens_digit_same (n : ℕ) : 
  n ∈ [0, 1, 25, 76] ↔ (n % 100 = n^2 % 100) := 
sorry

end units_digit_same_units_and_tens_digit_same_l27_27218


namespace exists_third_degree_poly_with_positive_and_negative_roots_l27_27485

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end exists_third_degree_poly_with_positive_and_negative_roots_l27_27485


namespace geometric_sequence_division_condition_l27_27481

variable {a : ℕ → ℝ}
variable {q : ℝ}

/-- a is a geometric sequence with common ratio q -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = a 1 * q ^ (n - 1)

/-- 3a₁, 1/2a₅, and 2a₃ forming an arithmetic sequence -/
def arithmetic_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  3 * a 1 + 2 * (a 1 * q ^ 2) = 2 * (1 / 2 * (a 1 * q ^ 4))

theorem geometric_sequence_division_condition
  (h1 : is_geometric_sequence a q)
  (h2 : arithmetic_sequence_condition a q) :
  (a 9 + a 10) / (a 7 + a 8) = 3 :=
sorry

end geometric_sequence_division_condition_l27_27481


namespace opposite_of_neg_2023_is_2023_l27_27367

theorem opposite_of_neg_2023_is_2023 :
  opposite (-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_is_2023_l27_27367


namespace annual_interest_rate_is_10_percent_l27_27829

noncomputable def principal (P : ℝ) := P = 1500
noncomputable def total_amount (A : ℝ) := A = 1815
noncomputable def time_period (t : ℝ) := t = 2
noncomputable def compounding_frequency (n : ℝ) := n = 1
noncomputable def interest_rate_compound_interest_formula (P A t n : ℝ) (r : ℝ) := 
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate_is_10_percent : 
  ∀ (P A t n : ℝ) (r : ℝ), principal P → total_amount A → time_period t → compounding_frequency n → 
  interest_rate_compound_interest_formula P A t n r → r = 0.1 :=
by
  intros P A t n r hP hA ht hn h_formula
  sorry

end annual_interest_rate_is_10_percent_l27_27829


namespace problem_statement_l27_27392

def permutations (n r : ℕ) : ℕ := n.factorial / (n - r).factorial
def combinations (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : permutations 4 2 - combinations 4 3 = 8 := 
by 
  sorry

end problem_statement_l27_27392


namespace find_p_l27_27527

theorem find_p (a b p : ℝ) (h1: a ≠ 0) (h2: b ≠ 0) 
  (h3: a^2 - 4 * b = 0) 
  (h4: a + b = 5 * p) 
  (h5: a * b = 2 * p^3) : p = 3 := 
sorry

end find_p_l27_27527


namespace q1_monotonic_increasing_intervals_q2_proof_l27_27462

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem q1_monotonic_increasing_intervals (a : ℝ) (h : a > 0) :
  (a > 1/2 ∧ (∀ x, (0 < x ∧ x < 1/a) ∨ (2 < x) → f a x > 0)) ∨
  (a = 1/2 ∧ (∀ x, 0 < x → f a x ≥ 0)) ∨
  (0 < a ∧ a < 1/2 ∧ (∀ x, (0 < x ∧ x < 2) ∨ (1/a < x) → f a x > 0)) := sorry

theorem q2_proof (x : ℝ) :
  (a = 0 ∧ x > 0 → f 0 x < 2 * Real.exp x - x - 4) := sorry

end q1_monotonic_increasing_intervals_q2_proof_l27_27462


namespace circle_area_from_intersection_l27_27756

-- Statement of the problem
theorem circle_area_from_intersection (r : ℝ) (A B : ℝ × ℝ)
  (h_circle : ∀ x y, (x + 2) ^ 2 + y ^ 2 = r ^ 2 ↔ (x, y) = A ∨ (x, y) = B)
  (h_parabola : ∀ x y, y ^ 2 = 20 * x ↔ (x, y) = A ∨ (x, y) = B)
  (h_axis_sym : A.1 = -5 ∧ B.1 = -5)
  (h_AB_dist : |A.2 - B.2| = 8) : π * r ^ 2 = 25 * π :=
by
  sorry

end circle_area_from_intersection_l27_27756


namespace carol_remaining_distance_l27_27722

def fuel_efficiency : ℕ := 25 -- miles per gallon
def gas_tank_capacity : ℕ := 18 -- gallons
def distance_to_home : ℕ := 350 -- miles

def total_distance_on_full_tank : ℕ := fuel_efficiency * gas_tank_capacity
def distance_after_home : ℕ := total_distance_on_full_tank - distance_to_home

theorem carol_remaining_distance :
  distance_after_home = 100 :=
sorry

end carol_remaining_distance_l27_27722


namespace longer_part_length_l27_27858

-- Conditions
def total_length : ℕ := 180
def diff_length : ℕ := 32

-- Hypothesis for the shorter part of the wire
def shorter_part (x : ℕ) : Prop :=
  x + (x + diff_length) = total_length

-- The goal is to find the longer part's length
theorem longer_part_length (x : ℕ) (h : shorter_part x) : x + diff_length = 106 := by
  sorry

end longer_part_length_l27_27858


namespace geometric_series_problem_l27_27918

theorem geometric_series_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ)
  (h_seq : ∀ n, a n + a (n + 1) = 3 * 2^n) :
  S (k + 2) - 2 * S (k + 1) + S k = 2^(k + 1) :=
sorry

end geometric_series_problem_l27_27918


namespace inequality_solutions_l27_27739

theorem inequality_solutions (y : ℝ) :
  (2 / (y + 2) + 4 / (y + 8) ≥ 1 ↔ (y > -8 ∧ y ≤ -4) ∨ (y ≥ -2 ∧ y ≤ 2)) :=
by
  sorry

end inequality_solutions_l27_27739


namespace vote_count_l27_27774

theorem vote_count 
(h_total: 200 = h_votes + l_votes + y_votes)
(h_hl: 3 * l_votes = 2 * h_votes)
(l_ly: 6 * y_votes = 5 * l_votes):
h_votes = 90 ∧ l_votes = 60 ∧ y_votes = 50 := by 
sorry

end vote_count_l27_27774


namespace simplify_expression_l27_27079

variable {a b c : ℤ}

theorem simplify_expression (a b c : ℤ) : 3 * a - (4 * a - 6 * b - 3 * c) - 5 * (c - b) = -a + 11 * b - 2 * c :=
by
  sorry

end simplify_expression_l27_27079


namespace find_f_5_l27_27588

def f (x : ℝ) : ℝ := sorry -- we need to create a function under our condition

theorem find_f_5 : f 5 = 0 :=
sorry

end find_f_5_l27_27588


namespace other_investment_interest_rate_l27_27317

open Real

-- Definitions of the given conditions
def total_investment : ℝ := 22000
def investment_at_8_percent : ℝ := 17000
def total_interest : ℝ := 1710
def interest_rate_8_percent : ℝ := 0.08

-- Derived definitions from the conditions
def other_investment_amount : ℝ := total_investment - investment_at_8_percent
def interest_from_8_percent : ℝ := investment_at_8_percent * interest_rate_8_percent
def interest_from_other : ℝ := total_interest - interest_from_8_percent

-- Proof problem: Prove that the percentage of the other investment is 0.07 (or 7%).
theorem other_investment_interest_rate :
  interest_from_other / other_investment_amount = 0.07 := by
  sorry

end other_investment_interest_rate_l27_27317


namespace complement_union_eq_complement_l27_27627

open Set

variable (U : Set ℤ) 
variable (A : Set ℤ) 
variable (B : Set ℤ)

theorem complement_union_eq_complement : 
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} :=
by
  intros hU hA hB
  -- sorry to skip the proof
  sorry

end complement_union_eq_complement_l27_27627


namespace square_side_length_l27_27382

theorem square_side_length (A : ℝ) (h : A = 625) : ∃ l : ℝ, l^2 = A ∧ l = 25 :=
by {
  sorry
}

end square_side_length_l27_27382


namespace fraction_taken_out_is_one_sixth_l27_27941

-- Define the conditions
def original_cards : ℕ := 43
def cards_added_by_Sasha : ℕ := 48
def cards_left_after_Karen_took_out : ℕ := 83

-- Calculate the total number of cards initially after Sasha added hers
def total_cards_after_Sasha : ℕ := original_cards + cards_added_by_Sasha

-- Calculate the number of cards Karen took out
def cards_taken_out_by_Karen : ℕ := total_cards_after_Sasha - cards_left_after_Karen_took_out

-- Define the fraction of the cards Sasha added that Karen took out
def fraction_taken_out : ℚ := cards_taken_out_by_Karen / cards_added_by_Sasha

-- Proof statement: Fraction of the cards Sasha added that Karen took out is 1/6
theorem fraction_taken_out_is_one_sixth : fraction_taken_out = 1 / 6 :=
by
    -- Sorry is a placeholder for the proof, which is not required.
    sorry

end fraction_taken_out_is_one_sixth_l27_27941


namespace bat_wings_area_l27_27917

-- Defining a rectangle and its properties.
structure Rectangle where
  PQ : ℝ
  QR : ℝ
  PT : ℝ
  TR : ℝ
  RQ : ℝ

-- Example rectangle from the problem
def PQRS : Rectangle := { PQ := 5, QR := 3, PT := 1, TR := 1, RQ := 1 }

-- Calculate area of "bat wings" if the rectangle is specified as in the above structure.
-- Expected result is 3.5
theorem bat_wings_area (r : Rectangle) (hPQ : r.PQ = 5) (hQR : r.QR = 3) 
    (hPT : r.PT = 1) (hTR : r.TR = 1) (hRQ : r.RQ = 1) : 
    ∃ area : ℝ, area = 3.5 :=
by
  -- Adding the proof would involve geometric calculations.
  -- Skipping the proof for now.
  sorry

end bat_wings_area_l27_27917


namespace diane_total_harvest_l27_27733

def total_harvest (h1 i1 i2 : Nat) : Nat :=
  h1 + (h1 + i1) + ((h1 + i1) + i2)

theorem diane_total_harvest :
  total_harvest 2479 6085 7890 = 27497 := 
by 
  sorry

end diane_total_harvest_l27_27733


namespace range_of_independent_variable_l27_27137

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l27_27137


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27280

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27280


namespace expected_value_monicas_winnings_l27_27638

noncomputable def prob_roll_odd_sum : ℝ := 1 / 2
noncomputable def prob_roll_even_sum : ℝ := 1 / 2
noncomputable def prob_roll_doubles : ℝ := 1 / 6
noncomputable def prob_roll_even_non_doubles : ℝ := 1 / 3
noncomputable def expected_sum_two_dice : ℝ := 7
noncomputable def expected_winnings_odd_sum : ℝ := 4 * expected_sum_two_dice
noncomputable def expected_winnings_even_doubles : ℝ := 2 * expected_sum_two_dice
noncomputable def loss_even_non_doubles : ℝ := -6

theorem expected_value_monicas_winnings : 
  (prob_roll_odd_sum * expected_winnings_odd_sum + 
  prob_roll_doubles * expected_winnings_even_doubles + 
  prob_roll_even_non_doubles * loss_even_non_doubles) = 14.33 :=
by 
  -- Proof goes here
  sorry

end expected_value_monicas_winnings_l27_27638


namespace unique_7tuple_exists_l27_27575

theorem unique_7tuple_exists 
  (x : Fin 7 → ℝ) 
  (h : (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7) 
  : ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7 :=
sorry

end unique_7tuple_exists_l27_27575


namespace missing_condition_l27_27391

theorem missing_condition (x y : ℕ) 
  (h1 : y = 2 * x + 9) 
  (h2 : y = 3 * (x - 2)) : 
  "Three people ride in one car, and there are two empty cars" :=
by sorry

end missing_condition_l27_27391


namespace slower_pump_time_l27_27827

theorem slower_pump_time (R : ℝ) (hours : ℝ) (combined_rate : ℝ) (faster_rate_adj : ℝ) (time_both : ℝ) :
  (combined_rate = R * (1 + faster_rate_adj)) →
  (faster_rate_adj = 1.5) →
  (time_both = 5) →
  (combined_rate * time_both = 1) →
  (hours = 1 / R) →
  hours = 12.5 :=
by
  sorry

end slower_pump_time_l27_27827


namespace total_books_l27_27194

/-- Define Tim’s and Sam’s number of books. -/
def TimBooks : ℕ := 44
def SamBooks : ℕ := 52

/-- Prove that together they have 96 books. -/
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l27_27194


namespace number_of_people_in_first_group_l27_27951

-- Define variables representing the work done by one person in one day (W) and the number of people in the first group (P)
variable (W : ℕ) (P : ℕ)

-- Conditions from the problem
-- Some people can do 3 times a particular work in 3 days
def condition1 : Prop := P * 3 * W = 3 * W

-- It takes 6 people 3 days to do 6 times of that particular work
def condition2 : Prop := 6 * 3 * W = 6 * W

-- The statement to prove
theorem number_of_people_in_first_group 
  (h1 : condition1 W P) 
  (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l27_27951


namespace hexahedron_octahedron_ratio_l27_27570

open Real

theorem hexahedron_octahedron_ratio (a : ℝ) (h_a_pos : 0 < a) :
  let r1 := (sqrt 6 * a / 9)
  let r2 := (sqrt 6 * a / 6)
  let ratio := r1 / r2
  ∃ m n : ℕ, gcd m n = 1 ∧ (ratio = (m : ℝ) / (n : ℝ)) ∧ (m * n = 6) :=
by {
  sorry
}

end hexahedron_octahedron_ratio_l27_27570


namespace inheritance_amount_l27_27802

def is_inheritance_amount (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_fed := x - federal_tax
  let state_tax := 0.12 * remaining_after_fed
  let total_tax_paid := federal_tax + state_tax
  total_tax_paid = 15600

theorem inheritance_amount : 
  ∃ x, is_inheritance_amount x ∧ x = 45882 := 
by
  sorry

end inheritance_amount_l27_27802


namespace solve_for_y_l27_27004

theorem solve_for_y (y : ℝ) : y^2 - 6 * y + 5 = 0 ↔ y = 1 ∨ y = 5 :=
by
  sorry

end solve_for_y_l27_27004


namespace number_of_liars_l27_27546

/-- There are 25 people in line, each of whom either tells the truth or lies.
The person at the front of the line says: "Everyone behind me is lying."
Everyone else says: "The person directly in front of me is lying."
Prove that the number of liars among these 25 people is 13. -/
theorem number_of_liars : 
  ∀ (persons : Fin 25 → Prop), 
    (persons 0 → ∀ n > 0, ¬persons n) →
    (∀ n : Nat, (1 ≤ n → n < 25 → persons n ↔ ¬persons (n - 1))) →
    (∃ l, l = 13 ∧ ∀ n : Nat, (0 ≤ n → n < 25 → persons n ↔ (n % 2 = 0))) :=
by
  sorry

end number_of_liars_l27_27546


namespace least_third_side_length_l27_27889

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l27_27889


namespace isabella_hair_length_l27_27154

theorem isabella_hair_length (h : ℕ) (g : h + 4 = 22) : h = 18 := by
  sorry

end isabella_hair_length_l27_27154


namespace range_of_m_l27_27609

theorem range_of_m (m : ℝ) : (∃ x1 x2 x3 : ℝ, 
    (x1 - 1) * (x1^2 - 2*x1 + m) = 0 ∧ 
    (x2 - 1) * (x2^2 - 2*x2 + m) = 0 ∧ 
    (x3 - 1) * (x3^2 - 2*x3 + m) = 0 ∧ 
    x1 = 1 ∧ 
    x2^2 - 2*x2 + m = 0 ∧ 
    x3^2 - 2*x3 + m = 0 ∧ 
    x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1 ∧ 
    x1 > 0 ∧ x2 > 0 ∧ x3 > 0) ↔ 3 / 4 < m ∧ m ≤ 1 := 
by
  sorry

end range_of_m_l27_27609


namespace smallest_four_digit_divisible_by_53_l27_27990

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27990


namespace product_remainder_mod_7_l27_27670

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l27_27670


namespace positive_solution_is_perfect_square_l27_27491

theorem positive_solution_is_perfect_square
  (t : ℤ)
  (n : ℕ)
  (h : n > 0)
  (root_cond : (n : ℤ)^2 + (4 * t - 1) * n + 4 * t^2 = 0) :
  ∃ k : ℕ, n = k^2 :=
sorry

end positive_solution_is_perfect_square_l27_27491


namespace solve_for_nonzero_x_l27_27385

open Real

theorem solve_for_nonzero_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 :=
by
  sorry

end solve_for_nonzero_x_l27_27385


namespace number_properties_l27_27074

-- Define what it means for a digit to be in a specific place
def digit_at_place (n place : ℕ) (d : ℕ) : Prop := 
  (n / 10 ^ place) % 10 = d

-- The given number
def specific_number : ℕ := 670154500

-- Conditions: specific number has specific digit in defined places
theorem number_properties : (digit_at_place specific_number 7 7) ∧ (digit_at_place specific_number 2 5) :=
by
  -- Proof of the theorem
  sorry

end number_properties_l27_27074


namespace andy_max_cookies_l27_27826

theorem andy_max_cookies (total_cookies : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ)
  (h1 : total_cookies = 30)
  (h2 : bella_cookies = 2 * andy_cookies)
  (h3 : andy_cookies + bella_cookies = total_cookies) :
  andy_cookies = 10 := by
  sorry

end andy_max_cookies_l27_27826


namespace commute_times_l27_27398

theorem commute_times (x y : ℝ) 
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) : |x - y| = 4 := 
sorry

end commute_times_l27_27398


namespace value_of_x2_minus_y2_l27_27116

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x2_minus_y2_l27_27116


namespace find_function_satisfaction_l27_27048

theorem find_function_satisfaction :
  ∃ (a b : ℚ) (f : ℚ × ℚ → ℚ), (∀ (x y z : ℚ),
  f (x, y) + f (y, z) + f (z, x) = f (0, x + y + z)) ∧ 
  (∀ (x y : ℚ), f (x, y) = a * y^2 + 2 * a * x * y + b * y) := sorry

end find_function_satisfaction_l27_27048


namespace sin_n_theta_eq_product_l27_27000

noncomputable
def sin_product_eq (n : ℕ) (θ : ℝ) : Prop :=
  sin (n * θ) = 2 ^ (n - 1) * ∏ k in finset.range n, sin (θ + (k * real.pi / n))

theorem sin_n_theta_eq_product (n : ℕ) (θ : ℝ) : sin_product_eq n θ :=
by sorry

end sin_n_theta_eq_product_l27_27000


namespace meryll_remaining_questions_l27_27163

variables (total_mc total_ps total_tf : ℕ)
variables (frac_mc frac_ps frac_tf : ℚ)

-- Conditions as Lean definitions:
def written_mc (total_mc : ℕ) (frac_mc : ℚ) := (frac_mc * total_mc).floor
def written_ps (total_ps : ℕ) (frac_ps : ℚ) := (frac_ps * total_ps).floor
def written_tf (total_tf : ℕ) (frac_tf : ℚ) := (frac_tf * total_tf).floor

def remaining_mc (total_mc : ℕ) (frac_mc : ℚ) := total_mc - written_mc total_mc frac_mc
def remaining_ps (total_ps : ℕ) (frac_ps : ℚ) := total_ps - written_ps total_ps frac_ps
def remaining_tf (total_tf : ℕ) (frac_tf : ℚ) := total_tf - written_tf total_tf frac_tf

def total_remaining (total_mc total_ps total_tf : ℕ) (frac_mc frac_ps frac_tf : ℚ) :=
  remaining_mc total_mc frac_mc + remaining_ps total_ps frac_ps + remaining_tf total_tf frac_tf

-- The statement to prove:
theorem meryll_remaining_questions :
  total_remaining 50 30 40 (5/8) (7/12) (2/5) = 56 :=
by
  sorry

end meryll_remaining_questions_l27_27163


namespace smallest_possible_sum_l27_27020

theorem smallest_possible_sum (A B C D : ℤ) 
  (h1 : A + B = 2 * C)
  (h2 : B * D = C * C)
  (h3 : 3 * C = 7 * B)
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D) : 
  A + B + C + D = 76 :=
sorry

end smallest_possible_sum_l27_27020


namespace intersection_line_through_circles_l27_27297

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x + 2 * y - 14 = 0

theorem intersection_line_through_circles : 
  (∀ x y : ℝ, circle1_equation x y → circle2_equation x y → x + y - 2 = 0) :=
by
  intros x y h1 h2
  sorry

end intersection_line_through_circles_l27_27297


namespace number_of_yogurts_l27_27088

def slices_per_yogurt : Nat := 8
def slices_per_banana : Nat := 10
def number_of_bananas : Nat := 4

theorem number_of_yogurts (slices_per_yogurt slices_per_banana number_of_bananas : Nat) : 
  slices_per_yogurt = 8 → 
  slices_per_banana = 10 → 
  number_of_bananas = 4 → 
  (number_of_bananas * slices_per_banana) / slices_per_yogurt = 5 :=
by
  intros h1 h2 h3
  sorry

end number_of_yogurts_l27_27088


namespace increase_p_does_not_always_increase_equal_points_l27_27693

-- Define the function representing equal points probability
def equal_points_probability (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

-- The main theorem states that increasing the probability 'p' of a draw 
-- does not necessarily increase the probability of the teams acquiring equal points.
theorem increase_p_does_not_always_increase_equal_points :
  ∃ p₁ p₂ : ℝ, 0 ≤ p₁ ∧ p₁ < p₂ ∧ p₂ ≤ 1 ∧ equal_points_probability p₁ ≥ equal_points_probability p₂ :=
by
  sorry

end increase_p_does_not_always_increase_equal_points_l27_27693


namespace union_of_sets_l27_27295

open Set

theorem union_of_sets (A B : Set ℝ) (hA : A = {x | -2 < x ∧ x < 1}) (hB : B = {x | 0 < x ∧ x < 2}) :
  A ∪ B = {x | -2 < x ∧ x < 2} :=
sorry

end union_of_sets_l27_27295


namespace probability_five_blue_marbles_is_correct_l27_27308

noncomputable def probability_of_five_blue_marbles : ℝ :=
let p_blue := (9 : ℝ) / 15
let p_red := (6 : ℝ) / 15
let specific_sequence_prob := p_blue ^ 5 * p_red ^ 3
let number_of_ways := (Nat.choose 8 5 : ℝ)
(number_of_ways * specific_sequence_prob)

theorem probability_five_blue_marbles_is_correct :
  probability_of_five_blue_marbles = 0.279 := by
sorry

end probability_five_blue_marbles_is_correct_l27_27308


namespace problem1_simplification_problem2_solve_fraction_l27_27049

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end problem1_simplification_problem2_solve_fraction_l27_27049


namespace households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l27_27475

namespace VehicleHouseholds

-- Definitions for the conditions
def totalHouseholds : ℕ := 250
def householdsNoVehicles : ℕ := 25
def householdsAllVehicles : ℕ := 36
def householdsCarOnly : ℕ := 62
def householdsBikeOnly : ℕ := 45
def householdsScooterOnly : ℕ := 30

-- Proof Statements
theorem households_with_two_types_of_vehicles :
  (totalHouseholds - householdsNoVehicles - householdsAllVehicles - 
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly)) = 52 := by
  sorry

theorem households_with_exactly_one_type_of_vehicle :
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly) = 137 := by
  sorry

theorem households_with_at_least_one_type_of_vehicle :
  (totalHouseholds - householdsNoVehicles) = 225 := by
  sorry

end VehicleHouseholds

end households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l27_27475


namespace external_tangent_b_value_l27_27080

theorem external_tangent_b_value:
  ∀ {C1 C2 : ℝ × ℝ} (r1 r2 : ℝ) (m b : ℝ),
  C1 = (3, -2) ∧ r1 = 3 ∧ 
  C2 = (15, 8) ∧ r2 = 8 ∧
  m = (60 / 11) →
  (∃ b, y = m * x + b ∧ b = 720 / 11) :=
by 
  sorry

end external_tangent_b_value_l27_27080


namespace product_of_two_numbers_in_ratio_l27_27972

theorem product_of_two_numbers_in_ratio (x y : ℚ) 
  (h1 : x - y = d)
  (h2 : x + y = 8 * d)
  (h3 : x * y = 15 * d) :
  x * y = 100 / 7 :=
by
  sorry

end product_of_two_numbers_in_ratio_l27_27972


namespace equal_points_probability_l27_27689

theorem equal_points_probability (p : ℝ) (prob_draw_increases : 0 ≤ p ∧ p ≤ 1) :
  (∀ q : ℝ, (0 ≤ q ∧ q < p) → (q^2 + (1 - q)^2 / 2 < p^2 + (1 - p)^2 / 2)) → False :=
begin
  sorry
end

end equal_points_probability_l27_27689


namespace total_toes_on_bus_l27_27336

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l27_27336


namespace no_solutions_abs_eq_3x_plus_6_l27_27092

theorem no_solutions_abs_eq_3x_plus_6 : ¬ ∃ x : ℝ, |x| = 3 * (|x| + 2) :=
by {
  sorry
}

end no_solutions_abs_eq_3x_plus_6_l27_27092


namespace stella_profit_l27_27512

theorem stella_profit 
    (dolls : ℕ) (doll_price : ℕ) 
    (clocks : ℕ) (clock_price : ℕ) 
    (glasses : ℕ) (glass_price : ℕ) 
    (cost : ℕ) :
    dolls = 3 →
    doll_price = 5 →
    clocks = 2 →
    clock_price = 15 →
    glasses = 5 →
    glass_price = 4 →
    cost = 40 →
    (dolls * doll_price + clocks * clock_price + glasses * glass_price - cost) = 25 := 
by 
  intros h_dolls h_doll_price h_clocks h_clock_price h_glasses h_glass_price h_cost
  rw [h_dolls, h_doll_price, h_clocks, h_clock_price, h_glasses, h_glass_price, h_cost]
  norm_num
  sorry

end stella_profit_l27_27512


namespace find_divisor_l27_27545

theorem find_divisor (d : ℕ) (h1 : 127 = d * 5 + 2) : d = 25 :=
sorry

end find_divisor_l27_27545


namespace jill_second_bus_time_l27_27030

-- Define constants representing the times
def wait_time_first_bus : ℕ := 12
def ride_time_first_bus : ℕ := 30

-- Define a function to calculate the total time for the first bus
def total_time_first_bus (wait : ℕ) (ride : ℕ) : ℕ :=
  wait + ride

-- Define a function to calculate the time for the second bus
def time_second_bus (total_first_bus_time : ℕ) : ℕ :=
  total_first_bus_time / 2

-- The theorem to prove
theorem jill_second_bus_time : 
  time_second_bus (total_time_first_bus wait_time_first_bus ride_time_first_bus) = 21 := by
  sorry

end jill_second_bus_time_l27_27030


namespace find_number_l27_27393

theorem find_number (N : ℝ) (h1 : (4/5) * (3/8) * N = some_number)
                    (h2 : 2.5 * N = 199.99999999999997) :
  N = 79.99999999999999 := 
sorry

end find_number_l27_27393


namespace sean_net_profit_l27_27943

noncomputable def total_cost (num_patches : ℕ) (cost_per_patch : ℝ) : ℝ :=
  num_patches * cost_per_patch

noncomputable def total_revenue (num_patches : ℕ) (selling_price_per_patch : ℝ) : ℝ :=
  num_patches * selling_price_per_patch

noncomputable def net_profit (total_revenue : ℝ) (total_cost : ℝ) : ℝ :=
  total_revenue - total_cost

-- Variables based on conditions
def num_patches := 100
def cost_per_patch := 1.25
def selling_price_per_patch := 12.00

theorem sean_net_profit : net_profit (total_revenue num_patches selling_price_per_patch) (total_cost num_patches cost_per_patch) = 1075 :=
by
  sorry

end sean_net_profit_l27_27943


namespace fifth_powers_sum_eq_l27_27268

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l27_27268


namespace turtles_still_on_sand_l27_27406

-- Define the total number of baby sea turtles
def total_turtles := 42

-- Define the function for calculating the number of swept turtles
def swept_turtles (total : Nat) : Nat := total / 3

-- Define the function for calculating the number of turtles still on the sand
def turtles_on_sand (total : Nat) (swept : Nat) : Nat := total - swept

-- Set parameters for the proof
def swept := swept_turtles total_turtles
def on_sand := turtles_on_sand total_turtles swept

-- Prove the statement
theorem turtles_still_on_sand : on_sand = 28 :=
by
  -- proof steps to be added here
  sorry

end turtles_still_on_sand_l27_27406


namespace find_distance_l27_27806

variable (y : ℚ) -- The circumference of the bicycle wheel
variable (x : ℚ) -- The distance between the village and the field

-- Condition 1: The circumference of the truck's wheel is 4/3 of the bicycle's wheel
def circum_truck_eq : Prop := (4 / 3 : ℚ) * y = y

-- Condition 2: The circumference of the truck's wheel is 2 meters shorter than the tractor's track
def circum_truck_less : Prop := (4 / 3 : ℚ) * y + 2 = y + 2

-- Condition 3: Truck's wheel makes 100 fewer revolutions than the bicycle's wheel
def truck_100_fewer : Prop := x / ((4 / 3 : ℚ) * y) = (x / y) - 100

-- Condition 4: Truck's wheel makes 150 more revolutions than the tractor track
def truck_150_more : Prop := x / ((4 / 3 : ℚ) * y) = (x / ((4 / 3 : ℚ) * y + 2)) + 150

theorem find_distance (y : ℚ) (x : ℚ) :
  circum_truck_eq y →
  circum_truck_less y →
  truck_100_fewer x y →
  truck_150_more x y →
  x = 600 :=
by
  intros
  sorry

end find_distance_l27_27806


namespace min_C_over_D_l27_27253

-- Define y + 1/y = D and y^2 + 1/y^2 = C.
theorem min_C_over_D (y C D : ℝ) (hy_pos : 0 < y) (hC : y ^ 2 + 1 / (y ^ 2) = C) (hD : y + 1 / y = D) (hC_pos : 0 < C) (hD_pos : 0 < D) :
  C / D = 2 := by
  sorry

end min_C_over_D_l27_27253


namespace bananas_to_oranges_l27_27010

theorem bananas_to_oranges :
  (3 / 4) * 16 * (1 / 1 : ℝ) = 10 * (1 / 1 : ℝ) → 
  (3 / 5) * 15 * (1 / 1 : ℝ) = 7.5 * (1 / 1 : ℝ) := 
by
  intros h
  sorry

end bananas_to_oranges_l27_27010


namespace product_remainder_mod_7_l27_27669

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l27_27669


namespace polynomial_roots_l27_27864

theorem polynomial_roots :
  (∃ x : ℝ, x^4 - 16*x^3 + 91*x^2 - 216*x + 180 = 0) ↔ (x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6) := 
sorry

end polynomial_roots_l27_27864


namespace range_of_x_l27_27460

theorem range_of_x (x a1 a2 y : ℝ) (d r : ℝ) (hx : x ≠ 0) 
  (h_arith : a1 = x + d ∧ a2 = x + 2 * d ∧ y = x + 3 * d)
  (h_geom : b1 = x * r ∧ b2 = x * r^2 ∧ y = x * r^3) : 4 ≤ x :=
by
  -- Assume x ≠ 0 as given and the sequences are arithmetic and geometric
  have hx3d := h_arith.2.2
  have hx3r := h_geom.2.2
  -- Substituting y in both sequences
  simp only [hx3d, hx3r] at *
  -- Solving for d and determining constraints
  sorry

end range_of_x_l27_27460


namespace interval_sum_l27_27761

theorem interval_sum (m n : ℚ) (h : ∀ x : ℚ, m < x ∧ x < n ↔ (mx - 1) / (x + 3) > 0) :
  m + n = -10 / 3 :=
sorry

end interval_sum_l27_27761


namespace range_of_m_l27_27106

theorem range_of_m (m n : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |m * x| - |x - n|) 
  (h_n_pos : 0 < n) (h_n_m : n < 1 + m) 
  (h_integer_sol : ∃ xs : Finset ℤ, xs.card = 3 ∧ ∀ x ∈ xs, f x < 0) : 
  1 < m ∧ m < 3 := 
sorry

end range_of_m_l27_27106


namespace cost_of_eight_books_l27_27783

theorem cost_of_eight_books (x : ℝ) (h : 2 * x = 34) : 8 * x = 136 :=
by
  sorry

end cost_of_eight_books_l27_27783


namespace remainder_of_product_mod_7_l27_27674

theorem remainder_of_product_mod_7 
  (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 5) : 
  (a * b * c) % 7 = 2 :=
sorry

end remainder_of_product_mod_7_l27_27674


namespace simplify_expression_solve_fractional_eq_l27_27052

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l27_27052


namespace instantaneous_angular_velocity_at_1_6_l27_27293

noncomputable def angular_velocity (t : ℝ) : ℝ := (2 * Real.pi / 0.64) * t^2

theorem instantaneous_angular_velocity_at_1_6 : 
  deriv angular_velocity 1.6 = 10 * Real.pi := 
by
  sorry

end instantaneous_angular_velocity_at_1_6_l27_27293


namespace ken_pencils_kept_l27_27157

-- Define the known quantities and conditions
def initial_pencils : ℕ := 250
def manny_pencils : ℕ := 25
def nilo_pencils : ℕ := manny_pencils * 2
def carlos_pencils : ℕ := nilo_pencils / 2
def tina_pencils : ℕ := carlos_pencils + 10
def rina_pencils : ℕ := tina_pencils - 20

-- Formulate the total pencils given away
def total_given_away : ℕ :=
  manny_pencils + nilo_pencils + carlos_pencils + tina_pencils + rina_pencils

-- Prove the final number of pencils Ken kept.
theorem ken_pencils_kept : initial_pencils - total_given_away = 100 :=
by
  sorry

end ken_pencils_kept_l27_27157


namespace coin_value_l27_27162

variables (n d q : ℕ)  -- Number of nickels, dimes, and quarters
variable (total_coins : n + d + q = 30)  -- Total coins condition

-- Original value in cents
def original_value : ℕ := 5 * n + 10 * d + 25 * q

-- Swapped values in cents
def swapped_value : ℕ := 10 * n + 25 * d + 5 * q

-- Condition given about the value difference
variable (value_difference : swapped_value = original_value + 150)

-- Prove the total value of coins is $5.00 (500 cents)
theorem coin_value : original_value = 500 :=
by
  sorry

end coin_value_l27_27162


namespace employee_payment_l27_27970

theorem employee_payment
  (A B C : ℝ)
  (h_total : A + B + C = 1500)
  (h_A : A = 1.5 * B)
  (h_C : C = 0.8 * B) :
  A = 682 ∧ B = 454 ∧ C = 364 := by
  sorry

end employee_payment_l27_27970


namespace addition_of_decimals_l27_27967

theorem addition_of_decimals : (0.3 + 0.03 : ℝ) = 0.33 := by
  sorry

end addition_of_decimals_l27_27967


namespace fifth_powers_sum_eq_l27_27269

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l27_27269


namespace missing_condition_l27_27390

theorem missing_condition (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : y = 3 * (x - 2)) :
  true := -- The equivalent mathematical statement asserts the correct missing condition.
sorry

end missing_condition_l27_27390


namespace intersection_S_T_l27_27294

def S : Set ℤ := {-4, -3, 6, 7}
def T : Set ℤ := {x | x^2 > 4 * x}

theorem intersection_S_T : S ∩ T = {-4, -3, 6, 7} :=
by
  sorry

end intersection_S_T_l27_27294


namespace triangle_smallest_angle_l27_27189

theorem triangle_smallest_angle (a b c : ℝ) (h1 : a + b + c = 180) (h2 : a = 5 * c) (h3 : b = 3 * c) : c = 20 :=
by
  sorry

end triangle_smallest_angle_l27_27189


namespace real_roots_range_of_m_l27_27579

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem real_roots_range_of_m :
  (∃ x : ℝ, x^2 + 4 * m * x + 4 * m^2 + 2 * m + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (2 * m + 1) * x + m^2 = 0) ↔ 
  m ≤ -3 / 2 ∨ m ≥ -1 / 4 :=
by
  sorry

end real_roots_range_of_m_l27_27579


namespace roller_coaster_cars_l27_27026

theorem roller_coaster_cars
  (people : ℕ)
  (runs : ℕ)
  (seats_per_car : ℕ)
  (people_per_run : ℕ)
  (h1 : people = 84)
  (h2 : runs = 6)
  (h3 : seats_per_car = 2)
  (h4 : people_per_run = people / runs) :
  (people_per_run / seats_per_car) = 7 :=
by
  sorry

end roller_coaster_cars_l27_27026


namespace greatest_missed_problems_l27_27907

theorem greatest_missed_problems (total_problems : ℕ) (passing_percentage : ℝ) (missed_problems : ℕ) : 
  total_problems = 50 ∧ passing_percentage = 0.85 → missed_problems = 7 :=
by
  sorry

end greatest_missed_problems_l27_27907


namespace area_conversion_correct_l27_27912

-- Define the legs of the right triangle
def leg1 : ℕ := 60
def leg2 : ℕ := 80

-- Define the conversion factor
def square_feet_in_square_yard : ℕ := 9

-- Calculate the area of the triangle in square feet
def area_in_square_feet : ℕ := (leg1 * leg2) / 2

-- Calculate the area of the triangle in square yards
def area_in_square_yards : ℚ := area_in_square_feet / square_feet_in_square_yard

-- The theorem stating the problem
theorem area_conversion_correct : area_in_square_yards = 266 + 2 / 3 := by
  sorry

end area_conversion_correct_l27_27912


namespace abcde_sum_to_628_l27_27219

theorem abcde_sum_to_628 (a b c d e : ℕ) (h_distinct : (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5) ∧ 
                                                 (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5) ∧ 
                                                 (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5) ∧ 
                                                 (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5) ∧ 
                                                 (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 5) ∧
                                                 a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                                                 b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                                                 c ≠ d ∧ c ≠ e ∧
                                                 d ≠ e)
  (h1 : b ≤ d)
  (h2 : c ≥ a)
  (h3 : a ≤ e)
  (h4 : b ≥ e)
  (h5 : d ≠ 5) :
  a^b + c^d + e = 628 := sorry

end abcde_sum_to_628_l27_27219


namespace radius_of_curvature_correct_l27_27248

open Real

noncomputable def radius_of_curvature_squared (a b t_0 : ℝ) : ℝ :=
  (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2)

theorem radius_of_curvature_correct (a b t_0 : ℝ) (h : a > 0) (h₁ : b > 0) :
  radius_of_curvature_squared a b t_0 = (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2) :=
sorry

end radius_of_curvature_correct_l27_27248


namespace magician_card_drawings_l27_27714

theorem magician_card_drawings : 
  let total_cards := 256
  let doubles_count := 16
  let valid_cards_per_double := total_cards - doubles_count - (15 * 2)
  let only_one_double_selections := doubles_count * valid_cards_per_double
  let both_doubles_selections := Nat.choose 16 2
  let total_ways := both_doubles_selections + only_one_double_selections
  total_ways = 3480 :=
by
  let total_cards := 256
  let doubles_count := 16
  let valid_cards_per_double := total_cards - doubles_count - 30
  let only_one_double_selections := doubles_count * valid_cards_per_double
  let both_doubles_selections := Nat.choose 16 2
  let total_ways := both_doubles_selections + only_one_double_selections
  exact Eq.refl total_ways

end magician_card_drawings_l27_27714


namespace sum_gcd_lcm_of_4_and_10_l27_27206

theorem sum_gcd_lcm_of_4_and_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 :=
by
  sorry

end sum_gcd_lcm_of_4_and_10_l27_27206


namespace sam_total_pennies_l27_27509

theorem sam_total_pennies : 
  ∀ (initial_pennies found_pennies total_pennies : ℕ),
  initial_pennies = 98 → 
  found_pennies = 93 → 
  total_pennies = initial_pennies + found_pennies → 
  total_pennies = 191 := by
  intros
  sorry

end sam_total_pennies_l27_27509


namespace range_of_sqrt_x_plus_3_l27_27143

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l27_27143


namespace twenty_four_point_game_l27_27177

theorem twenty_four_point_game : (9 + 7) * 3 / 2 = 24 := by
  sorry -- Proof to be provided

end twenty_four_point_game_l27_27177


namespace base_sum_correct_l27_27434

theorem base_sum_correct :
  let C := 12
  let a := 3 * 9^2 + 5 * 9^1 + 7 * 9^0
  let b := 4 * 13^2 + C * 13^1 + 2 * 13^0
  a + b = 1129 :=
by
  sorry

end base_sum_correct_l27_27434


namespace intersection_point_of_given_lines_l27_27981

theorem intersection_point_of_given_lines :
  ∃ (x y : ℚ), 2 * y = -x + 3 ∧ -y = 5 * x + 1 ∧ x = -5 / 9 ∧ y = 16 / 9 :=
by
  sorry

end intersection_point_of_given_lines_l27_27981


namespace number_of_owls_joined_l27_27709

-- Define the initial condition
def initial_owls : ℕ := 3

-- Define the current condition
def current_owls : ℕ := 5

-- Define the problem statement as a theorem
theorem number_of_owls_joined : (current_owls - initial_owls) = 2 :=
by
  sorry

end number_of_owls_joined_l27_27709


namespace percent_of_240_l27_27975

theorem percent_of_240 (h : (3 / 8 / 100 : ℝ) = 3 / 800) : 
  (3 / 800 * 240 = 0.9) :=
begin
  sorry
end

end percent_of_240_l27_27975


namespace jerry_age_l27_27933

theorem jerry_age (M J : ℕ) (h1 : M = 20) (h2 : M = 2 * J - 8) : J = 14 := 
by
  sorry

end jerry_age_l27_27933


namespace exists_third_degree_poly_with_positive_and_negative_roots_l27_27486

theorem exists_third_degree_poly_with_positive_and_negative_roots :
  ∃ (P : ℝ → ℝ), (∃ x : ℝ, P x = 0 ∧ x > 0) ∧ (∃ y : ℝ, (deriv P) y = 0 ∧ y < 0) :=
sorry

end exists_third_degree_poly_with_positive_and_negative_roots_l27_27486


namespace trapezoid_sides_l27_27812

-- The given conditions: area = 8 cm^2, base angle = 30 degrees
def isosceles_trapezoid_circumscribed (a b c R : ℝ) (area : ℝ) (angle : ℝ) :=
  -- Conditions of the problem
  area = 8 ∧ 
  angle = π / 6 ∧
  -- Definitions for an isosceles trapezoid and the properties of the circle
  2 * R = c / 2 ∧
  area = (1/2) * (a + b) * (2 * R)

-- The proof goal: determine the sides of the trapezoid
theorem trapezoid_sides :
  ∃ (a b c : ℝ),
    isosceles_trapezoid_circumscribed a b c (c / 4) 8 (π / 6) ∧
    c = 4 ∧
    a = 4 - 2 * real.sqrt 3 ∧
    b = 4 + 2 * real.sqrt 3 :=
by { sorry }

end trapezoid_sides_l27_27812


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27265

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27265


namespace five_letter_arrangements_l27_27598

theorem five_letter_arrangements (A B C D E F G H : Type) :
  let letters := {A, B, C, D, E, F, G, H} in
  ∃ n : ℕ,
  (n = 5) ∧                                  -- The length of the arrangement is 5
  (D ∈ letters) ∧                             -- D must be in the first position
  (A ∈ letters) ∧                             -- A must be in the last position
  (E ∈ letters) ∧                             -- E must be one of the letters
  (∀ x : letters, x ≠ x) →                     -- No letter can be used more than once
  n = 60 :=
by
  sorry

end five_letter_arrangements_l27_27598


namespace smallest_k_divides_l27_27443

-- Given Problem: z^{12} + z^{11} + z^8 + z^7 + z^6 + z^3 + 1 divides z^k - 1
theorem smallest_k_divides (
  k : ℕ
) : (∀ z : ℂ, (z ^ 12 + z ^ 11 + z ^ 8 + z ^ 7 + z ^ 6 + z ^ 3 + 1) ∣ (z ^ k - 1) ↔ k = 182) :=
sorry

end smallest_k_divides_l27_27443


namespace quadratic_inequality_solution_l27_27865

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (3 * x^2 - 5 * x - 2 < 0) ↔ (-1/3 < x ∧ x < 2) :=
by
  sorry

end quadratic_inequality_solution_l27_27865


namespace a_n_nonzero_l27_27752

/-- Recurrence relation for the sequence a_n --/
def a : ℕ → ℤ
| 0 => 1
| 1 => 2
| (n + 2) => if (a n * a (n + 1)) % 2 = 1 then 5 * a (n + 1) - 3 * a n else a (n + 1) - a n

/-- Proof that for all n, a_n is non-zero --/
theorem a_n_nonzero : ∀ n : ℕ, a n ≠ 0 := 
sorry

end a_n_nonzero_l27_27752


namespace maximize_profit_l27_27234

theorem maximize_profit : 
  ∃ (a b : ℕ), 
  a ≤ 8 ∧ 
  b ≤ 7 ∧ 
  2 * a + b ≤ 19 ∧ 
  a + b ≤ 12 ∧ 
  10 * a + 6 * b ≥ 72 ∧ 
  (a * 450 + b * 350) = 4900 :=
by
  sorry

end maximize_profit_l27_27234


namespace find_exponent_l27_27118

theorem find_exponent (y : ℕ) (b : ℕ) (h_b : b = 2)
  (h : 1 / 8 * 2 ^ 40 = b ^ y) : y = 37 :=
by
  sorry

end find_exponent_l27_27118


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27266

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27266


namespace triangle_inequality_iff_inequality_l27_27168

theorem triangle_inequality_iff_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * (a^4 + b^4 + c^4) < (a^2 + b^2 + c^2)^2) ↔ (a + b > c ∧ b + c > a ∧ c + a > b) :=
by
  sorry

end triangle_inequality_iff_inequality_l27_27168


namespace savings_for_mother_l27_27636

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end savings_for_mother_l27_27636


namespace incorrect_statement_b_l27_27099

-- Defining the equation of the circle
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 25

-- Defining the point not on the circle
def is_not_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 25

-- The proposition to be proved
theorem incorrect_statement_b : ¬ ∀ p : ℝ × ℝ, is_not_on_circle p.1 p.2 → ¬ is_on_circle p.1 p.2 :=
by
  -- Here we should provide the proof, but this is not required based on the instructions.
  sorry

end incorrect_statement_b_l27_27099


namespace selling_price_with_discount_l27_27520

variable (a : ℝ)

theorem selling_price_with_discount (h : a ≥ 0) : (a * 1.2 * 0.91) = (a * 1.2 * 0.91) :=
by
  sorry

end selling_price_with_discount_l27_27520


namespace function_domain_l27_27135

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l27_27135


namespace part1_part2_l27_27104

-- Part (1)
theorem part1 (B : ℝ) (b : ℝ) (S : ℝ) (a c : ℝ) (B_eq : B = Real.pi / 3) 
  (b_eq : b = Real.sqrt 7) (S_eq : S = (3 * Real.sqrt 3) / 2) :
  a + c = 5 := 
sorry

-- Part (2)
theorem part2 (C : ℝ) (c : ℝ) (dot_BA_BC AB_AC : ℝ) 
  (C_cond : 2 * Real.cos C * (dot_BA_BC + AB_AC) = c^2) :
  C = Real.pi / 3 := 
sorry

end part1_part2_l27_27104


namespace find_slope_l27_27798

theorem find_slope (k : ℝ) :
  (∀ x y : ℝ, y = -2 * x + 3 → y = k * x + 4 → (x, y) = (1, 1)) → k = -3 :=
by
  sorry

end find_slope_l27_27798


namespace rainfall_third_day_is_18_l27_27033

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end rainfall_third_day_is_18_l27_27033


namespace pigs_remaining_l27_27309

def initial_pigs : ℕ := 364
def pigs_joined : ℕ := 145
def pigs_moved : ℕ := 78

theorem pigs_remaining : initial_pigs + pigs_joined - pigs_moved = 431 := by
  sorry

end pigs_remaining_l27_27309


namespace least_third_side_length_l27_27891

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l27_27891


namespace simplify_expression_l27_27934

theorem simplify_expression
  (x y : ℝ)
  (h : (x + 2)^3 ≠ (y - 2)^3) :
  ( (x + 2)^3 + (y + x)^3 ) / ( (x + 2)^3 - (y - 2)^3 ) = (2 * x + y + 2) / (x - y + 4) :=
sorry

end simplify_expression_l27_27934


namespace a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27257

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l27_27257


namespace circle_through_A_B_and_tangent_to_m_l27_27591

noncomputable def circle_equation (x y : ℚ) : Prop :=
  x^2 + (y - 1/3)^2 = 16/9

theorem circle_through_A_B_and_tangent_to_m :
  ∃ (c : ℚ × ℚ) (r : ℚ),
    (c = (0, 1/3)) ∧
    (r = 4/3) ∧
    (∀ (x y : ℚ),
      (x = 0 ∧ y = -1 ∨ x = 4/3 ∧ y = 1/3 → (x^2 + (y - 1/3)^2 = 16/9)) ∧
      (x = 4/3 → x = r)) :=
by
  sorry

end circle_through_A_B_and_tangent_to_m_l27_27591


namespace erin_trolls_count_l27_27431

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l27_27431


namespace geometric_sequence_arithmetic_sequence_l27_27160

def seq₃ := 7
def rec_rel (a : ℕ → ℕ) := ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + a 2 - 2

-- Problem Part 1: Prove that {a_n+1} is a geometric sequence
theorem geometric_sequence (a : ℕ → ℕ) (h_rec_rel : rec_rel a) :
  ∃ r, ∀ n, n ≥ 1 → (a n + 1) = r * (a (n - 1) + 1) :=
sorry

-- Problem Part 2: Given a general formula, prove n, a_n, and S_n form an arithmetic sequence
def general_formula (a : ℕ → ℕ) := ∀ n, a n = 2^n - 1
def sum_formula (S : ℕ → ℕ) := ∀ n, S n = 2^(n+1) - n - 2

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_general : general_formula a) (h_sum : sum_formula S) :
  ∀ n, n + S n = 2 * a n :=
sorry

end geometric_sequence_arithmetic_sequence_l27_27160


namespace max_elements_in_A_l27_27683

noncomputable def max_possible_elements_in_A : ℕ :=
  Nat.choose 2022 1011

structure valid_function (f : ℕ → ℕ) : Prop :=
  (non_increasing : ∀ (x y : ℕ), (1 ≤ x ∧ x < y ∧ y ≤ 2023) → f x ≥ f y)
  (composition_property : ∀ (g : ℕ → ℕ) (x : ℕ), (1 ≤ x ∧ x ≤ 2023) →
    f (g x) = g (f (g x)))

def is_valid_A (A : Finset (ℕ → ℕ)) : Prop :=
  ∀ f ∈ A, valid_function f

theorem max_elements_in_A :
  ∃ (A : Finset (ℕ → ℕ)), is_valid_A A ∧ A.card = max_possible_elements_in_A := by
  sorry

end max_elements_in_A_l27_27683


namespace speed_of_second_train_correct_l27_27698

noncomputable def length_first_train : ℝ := 140 -- in meters
noncomputable def length_second_train : ℝ := 160 -- in meters
noncomputable def time_to_cross : ℝ := 10.799136069114471 -- in seconds
noncomputable def speed_first_train : ℝ := 60 -- in km/hr
noncomputable def speed_second_train : ℝ := 40 -- in km/hr

theorem speed_of_second_train_correct :
  (length_first_train + length_second_train)/time_to_cross - (speed_first_train * (5/18)) = speed_second_train * (5/18) :=
by
  sorry

end speed_of_second_train_correct_l27_27698


namespace least_side_is_8_l27_27887

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l27_27887


namespace initial_hair_length_l27_27172

-- Definitions based on the conditions
def hair_cut_off : ℕ := 13
def current_hair_length : ℕ := 1

-- The problem statement to be proved
theorem initial_hair_length : (current_hair_length + hair_cut_off = 14) :=
by
  sorry

end initial_hair_length_l27_27172


namespace wall_area_in_square_meters_l27_27611

variable {W H : ℤ} -- We treat W and H as integers referring to centimeters

theorem wall_area_in_square_meters 
  (h₁ : W / 30 = 8) 
  (h₂ : H / 30 = 5) : 
  (W / 100) * (H / 100) = 360 / 100 :=
by 
  sorry

end wall_area_in_square_meters_l27_27611


namespace solve_expression_l27_27655

theorem solve_expression :
  (27 ^ (2 / 3) - 2 ^ (Real.log 3 / Real.log 2) * (Real.logb 2 (1 / 8)) +
    Real.logb 10 4 + Real.logb 10 25 = 20) :=
by
  sorry

end solve_expression_l27_27655


namespace fifth_equation_l27_27164

theorem fifth_equation :
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) := 
by sorry

end fifth_equation_l27_27164


namespace utilities_cost_l27_27396

theorem utilities_cost
    (rent1 : ℝ) (utility1 : ℝ) (rent2 : ℝ) (utility2 : ℝ)
    (distance1 : ℝ) (distance2 : ℝ) 
    (cost_per_mile : ℝ) 
    (drive_days : ℝ) (cost_difference : ℝ)
    (h1 : rent1 = 800)
    (h2 : rent2 = 900)
    (h3 : utility2 = 200)
    (h4 : distance1 = 31)
    (h5 : distance2 = 21)
    (h6 : cost_per_mile = 0.58)
    (h7 : drive_days = 20)
    (h8 : cost_difference = 76)
    : utility1 = 259.60 := 
by
  sorry

end utilities_cost_l27_27396


namespace find_x_l27_27152

-- Definitions of the conditions in Lean 4
def angle_sum_180 (A B C : ℝ) : Prop := A + B + C = 180
def angle_BAC_eq_90 (A : ℝ) : Prop := A = 90
def angle_BCA_eq_2x (C x : ℝ) : Prop := C = 2 * x
def angle_ABC_eq_3x (B x : ℝ) : Prop := B = 3 * x

-- The theorem we need to prove
theorem find_x (A B C x : ℝ) 
  (h1 : angle_sum_180 A B C) 
  (h2 : angle_BAC_eq_90 A)
  (h3 : angle_BCA_eq_2x C x) 
  (h4 : angle_ABC_eq_3x B x) : x = 18 :=
by 
  sorry

end find_x_l27_27152


namespace find_a9_l27_27583

theorem find_a9 (a : ℕ → ℕ) 
  (h_add : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
  (h_a2 : a 2 = 4) 
  : a 9 = 18 :=
sorry

end find_a9_l27_27583


namespace melted_mixture_weight_l27_27704

theorem melted_mixture_weight (Z C : ℝ) (h_ratio : Z / C = 9 / 11) (h_zinc : Z = 28.8) : Z + C = 64 :=
by
  sorry

end melted_mixture_weight_l27_27704


namespace arctan_tan_75_minus_2_tan_30_eq_75_l27_27419

theorem arctan_tan_75_minus_2_tan_30_eq_75 : 
  arctan (tan 75 - 2 * tan 30) = 75 :=
by
  sorry

end arctan_tan_75_minus_2_tan_30_eq_75_l27_27419


namespace rectangle_length_is_16_l27_27544

-- Define the conditions
def side_length_square : ℕ := 8
def width_rectangle : ℕ := 4
def area_square : ℕ := side_length_square ^ 2  -- Area of the square
def area_rectangle (length : ℕ) : ℕ := width_rectangle * length  -- Area of the rectangle

-- Lean 4 statement
theorem rectangle_length_is_16 (L : ℕ) (h : area_square = area_rectangle L) : L = 16 :=
by
  /- Proof will be inserted here -/
  sorry

end rectangle_length_is_16_l27_27544


namespace find_brick_width_l27_27093

def SurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

theorem find_brick_width :
  ∃ width : ℝ, SurfaceArea 10 width 3 = 164 ∧ width = 4 :=
by
  sorry

end find_brick_width_l27_27093


namespace find_probability_l27_27848

open ProbabilityTheory

noncomputable def probability_of_eleventh_draw (X : ℕ → ℕ) (p : ℕ → ℚ) : ℚ :=
  let redProb := 1/3
  let whiteProb := 2/3
  let comb := Nat.choose 10 8
  comb * (redProb ^ 9) * (whiteProb ^ 2)

theorem find_probability (X : ℕ → ℕ) (p : ℕ → ℚ) (h : ∀ n, p n = if n ≤ 9 then 1 else 0): 
  probability_of_eleventh_draw X p = Nat.choose 10 8 * (1 / 3) ^ 9 * (2 / 3) ^ 2 :=
by
  sorry

end find_probability_l27_27848


namespace incorrect_statement_l27_27073

-- Define the relationship between the length of the spring and the mass of the object
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

-- Formalize statements A, B, C, and D
def statementA : Prop := spring_length 0 = 10

def statementB : Prop :=
  ¬ ∃ (length : ℝ) (mass : ℝ), (spring_length mass = length ∧ mass = (length - 10) / 2.5)

def statementC : Prop :=
  ∀ m : ℝ, spring_length (m + 1) = spring_length m + 2.5

def statementD : Prop := spring_length 4 = 20

-- The Lean statement to prove that statement B is incorrect
theorem incorrect_statement (hA : statementA) (hC : statementC) (hD : statementD) : ¬ statementB := by
  sorry

end incorrect_statement_l27_27073


namespace max_value_g_l27_27183

def g : ℕ → ℤ
| n => if n < 5 then n + 10 else g (n - 3)

theorem max_value_g : ∃ x, (∀ n : ℕ, g n ≤ x) ∧ (∃ y, g y = x) ∧ x = 14 := 
by
  sorry

end max_value_g_l27_27183


namespace area_of_inscribed_square_l27_27952

theorem area_of_inscribed_square (XY YZ : ℝ) (hXY : XY = 18) (hYZ : YZ = 30) :
  ∃ (s : ℝ), s^2 = 540 :=
by
  sorry

end area_of_inscribed_square_l27_27952


namespace no_integer_solution_exists_l27_27811

theorem no_integer_solution_exists : ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8 * t - 1 := 
by sorry

end no_integer_solution_exists_l27_27811


namespace cos_arith_prog_impossible_l27_27212

theorem cos_arith_prog_impossible
  (x y z : ℝ)
  (sin_arith_prog : 2 * Real.sin y = Real.sin x + Real.sin z) :
  ¬ (2 * Real.cos y = Real.cos x + Real.cos z) :=
by
  sorry

end cos_arith_prog_impossible_l27_27212


namespace milk_volume_in_ounces_l27_27401

theorem milk_volume_in_ounces
  (packets : ℕ)
  (volume_per_packet_ml : ℕ)
  (ml_per_oz : ℕ)
  (total_volume_ml : ℕ)
  (total_volume_oz : ℕ)
  (h1 : packets = 150)
  (h2 : volume_per_packet_ml = 250)
  (h3 : ml_per_oz = 30)
  (h4 : total_volume_ml = packets * volume_per_packet_ml)
  (h5 : total_volume_oz = total_volume_ml / ml_per_oz) :
  total_volume_oz = 1250 :=
by
  sorry

end milk_volume_in_ounces_l27_27401


namespace range_of_x_in_sqrt_x_plus_3_l27_27128

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l27_27128


namespace rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l27_27361

variables (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0)

def S1 := (x + 5) * (y + 5)
def S2 := (x - 2) * (y - 2)
def perimeter := 2 * (x + y)

theorem rectangle_perimeter (h : S1 - S2 = 196) :
  perimeter = 50 :=
sorry

theorem difference_multiple_of_7 (h : S1 - S2 = 196) :
  ∃ k : ℕ, S1 - S2 = 7 * k :=
sorry

theorem area_seamless_combination (h : S1 - S2 = 196) :
  S1 - x * y = (x + 5) * (y + 5) - x * y ∧ x = y + 5 :=
sorry

end rectangle_perimeter_difference_multiple_of_7_area_seamless_combination_l27_27361


namespace trees_in_garden_l27_27474

theorem trees_in_garden (yard_length : ℕ) (distance_between_trees : ℕ) (H1 : yard_length = 400) (H2 : distance_between_trees = 16) : 
  (yard_length / distance_between_trees) + 1 = 26 :=
by
  -- Adding sorry to skip the proof
  sorry

end trees_in_garden_l27_27474


namespace find_k_find_m_l27_27878

-- Condition definitions
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

-- Proof problem statements
theorem find_k (k : ℝ) :
  (3 * a.fst - b.fst) / (a.fst + k * b.fst) = (3 * a.snd - b.snd) / (a.snd + k * b.snd) →
  k = -1 / 3 :=
sorry

theorem find_m (m : ℝ) :
  a.fst * (m * a.fst - b.fst) + a.snd * (m * a.snd - b.snd) = 0 →
  m = -4 / 5 :=
sorry

end find_k_find_m_l27_27878


namespace goose_eggs_count_l27_27044

theorem goose_eggs_count (E : ℕ) (h1 : E % 3 = 0) 
(h2 : ((4 / 5) * (1 / 3) * E) * (2 / 5) = 120) : E = 1125 := 
sorry

end goose_eggs_count_l27_27044


namespace emily_remainder_l27_27571

theorem emily_remainder (c d : ℤ) (h1 : c % 60 = 53) (h2 : d % 42 = 35) : (c + d) % 21 = 4 :=
by
  sorry

end emily_remainder_l27_27571


namespace product_remainder_mod_7_l27_27664

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l27_27664


namespace fraction_transformation_l27_27468

theorem fraction_transformation (a b x: ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2 * b) / (a - 2 * b) = (x + 2) / (x - 2) :=
by sorry

end fraction_transformation_l27_27468


namespace e_n_max_value_l27_27492

def b (n : ℕ) : ℕ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem e_n_max_value (n : ℕ) : e n = 1 := 
by sorry

end e_n_max_value_l27_27492


namespace fraction_zero_l27_27288

theorem fraction_zero (x : ℝ) (h : (x - 1) * (x + 2) = 0) (hne : x^2 - 1 ≠ 0) : x = -2 :=
by
  sorry

end fraction_zero_l27_27288


namespace company_budget_salaries_degrees_l27_27710

theorem company_budget_salaries_degrees :
  let transportation := 0.20
  let research_and_development := 0.09
  let utilities := 0.05
  let equipment := 0.04
  let supplies := 0.02
  let total_budget := 1.0
  let total_percentage := transportation + research_and_development + utilities + equipment + supplies
  let salaries_percentage := total_budget - total_percentage
  let total_degrees := 360.0
  let degrees_salaries := salaries_percentage * total_degrees
  degrees_salaries = 216 :=
by
  sorry

end company_budget_salaries_degrees_l27_27710


namespace pants_to_shirts_ratio_l27_27413

-- Conditions
def shirts : ℕ := 4
def total_clothes : ℕ := 16

-- Given P as the number of pants and S as the number of shorts
variable (P S : ℕ)

-- State the conditions as hypotheses
axiom shorts_half_pants : S = P / 2
axiom total_clothes_condition : 4 + P + S = 16

-- Question: Prove that the ratio of pants to shirts is 2
theorem pants_to_shirts_ratio : P = 2 * shirts :=
by {
  -- insert proof steps here
  sorry
}

end pants_to_shirts_ratio_l27_27413


namespace jerusha_earnings_l27_27620

theorem jerusha_earnings (L : ℕ) (h1 : 5 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l27_27620


namespace fifth_powers_sum_eq_l27_27270

section PowerProof

variables (a b c d : ℝ)

-- Conditions:
def condition1 : a + b = c + d := sorry
def condition2 : a^3 + b^3 = c^3 + d^3 := sorry

-- Claim for fifth powers:
theorem fifth_powers_sum_eq : a + b = c + d → a^3 + b^3 = c^3 + d^3 → a^5 + b^5 = c^5 + d^5 := by
  intros h1 h2
  sorry

-- Clauses for disproving fourth powers under generality:
example : ¬ (∀ a b c d : ℝ, (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) :=
  by{
    sorry
  }

end PowerProof

end fifth_powers_sum_eq_l27_27270


namespace aluminum_phosphate_molecular_weight_l27_27699

theorem aluminum_phosphate_molecular_weight :
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  (Al + P + 4 * O) = 121.95 :=
by
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  sorry

end aluminum_phosphate_molecular_weight_l27_27699


namespace radio_selling_price_l27_27181

theorem radio_selling_price (CP LP Loss SP : ℝ) (h1 : CP = 1500) (h2 : LP = 11)
  (h3 : Loss = (LP / 100) * CP) (h4 : SP = CP - Loss) : SP = 1335 := 
  by
  -- hint: Apply the given conditions.
  sorry

end radio_selling_price_l27_27181


namespace polynomial_no_real_roots_l27_27808

def f (x : ℝ) : ℝ := 4 * x ^ 8 - 2 * x ^ 7 + x ^ 6 - 3 * x ^ 4 + x ^ 2 - x + 1

theorem polynomial_no_real_roots : ∀ x : ℝ, f x > 0 := by
  sorry

end polynomial_no_real_roots_l27_27808


namespace smallest_four_digit_multiple_of_53_l27_27996

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27996


namespace reciprocal_eq_self_l27_27782

open Classical

theorem reciprocal_eq_self (a : ℝ) (h : a = 1 / a) : a = 1 ∨ a = -1 := 
sorry

end reciprocal_eq_self_l27_27782


namespace total_vertical_distance_l27_27551

theorem total_vertical_distance (thickness top_diameter bottom_diameter : ℕ) 
  (h_thickness : thickness = 2) 
  (h_top_diameter : top_diameter = 20) 
  (h_bottom_diameter : bottom_diameter = 4) 
  (h_outside_decrease : ∀ n : ℕ, n > 0 → (top_diameter - 2 * n) > bottom_diameter → (top_diameter - 2 * (n + 1)) = (top_diameter - 2 * n) - 2) :
  (let n := 9 in
   let a := (top_diameter - 4) in
   let d := -2 in
   ∑ i in finset.range n, a + i * d) = 72 := 
by 
  sorry

end total_vertical_distance_l27_27551


namespace remainder_55_57_div_8_l27_27045

def remainder (a b n : ℕ) := (a * b) % n

theorem remainder_55_57_div_8 : remainder 55 57 8 = 7 := by
  -- proof omitted
  sorry

end remainder_55_57_div_8_l27_27045


namespace no_solution_l27_27939

theorem no_solution : ∀ x y z t : ℕ, 16^x + 21^y + 26^z ≠ t^2 :=
by
  intro x y z t
  sorry

end no_solution_l27_27939


namespace speed_of_train_b_l27_27846

-- Defining the known data
def train_a_speed := 60 -- km/h
def train_a_time_after_meeting := 9 -- hours
def train_b_time_after_meeting := 4 -- hours

-- Statement we want to prove
theorem speed_of_train_b : ∃ (V_b : ℝ), V_b = 135 :=
by
  -- Sorry placeholder, as the proof is not required
  sorry

end speed_of_train_b_l27_27846


namespace num_ways_arith_prog_l27_27914

theorem num_ways_arith_prog : 
  ∑ (d : ℕ) in finset.range 334, 1000 - 3 * d = 166167 :=
by
  sorry

end num_ways_arith_prog_l27_27914


namespace arithmetic_proof_l27_27247

theorem arithmetic_proof : (28 + 48 / 69) * 69 = 1980 :=
by
  sorry

end arithmetic_proof_l27_27247


namespace correspond_half_l27_27736

theorem correspond_half (m n : ℕ) 
  (H : ∀ h : Fin m, ∃ g_set : Finset (Fin n), (g_set.card = n / 2) ∧ (∀ g : Fin n, g ∈ g_set))
  (G : ∀ g : Fin n, ∃ h_set : Finset (Fin m), (h_set.card ≤ m / 2) ∧ (∀ h : Fin m, h ∈ h_set)) :
  (∀ h : Fin m, ∀ g_set : Finset (Fin n), g_set.card = n / 2) ∧ (∀ g : Fin n, ∀ h_set : Finset (Fin m), h_set.card = m / 2) :=
by
  sorry

end correspond_half_l27_27736


namespace no_hikers_in_morning_l27_27057

-- Given Conditions
def morning_rowers : ℕ := 13
def afternoon_rowers : ℕ := 21
def total_rowers : ℕ := 34

-- Statement to be proven
theorem no_hikers_in_morning : (total_rowers - afternoon_rowers = morning_rowers) →
                              (total_rowers - afternoon_rowers = morning_rowers) →
                              0 = 34 - 21 - morning_rowers :=
by
  intros h1 h2
  sorry

end no_hikers_in_morning_l27_27057


namespace longest_side_eq_24_l27_27525

noncomputable def x : Real := 19 / 3

def side1 (x : Real) : Real := x + 3
def side2 (x : Real) : Real := 2 * x - 1
def side3 (x : Real) : Real := 3 * x + 5

def perimeter (x : Real) : Prop :=
  side1 x + side2 x + side3 x = 45

theorem longest_side_eq_24 : perimeter x → max (max (side1 x) (side2 x)) (side3 x) = 24 :=
by
  sorry

end longest_side_eq_24_l27_27525


namespace asthma_distribution_l27_27696

noncomputable def total_children := 490
noncomputable def boys := 280
noncomputable def general_asthma_ratio := 2 / 7
noncomputable def boys_asthma_ratio := 1 / 9

noncomputable def total_children_with_asthma := general_asthma_ratio * total_children
noncomputable def boys_with_asthma := boys_asthma_ratio * boys
noncomputable def girls_with_asthma := total_children_with_asthma - boys_with_asthma

theorem asthma_distribution
  (h_general_asthma: general_asthma_ratio = 2 / 7)
  (h_total_children: total_children = 490)
  (h_boys: boys = 280)
  (h_boys_asthma: boys_asthma_ratio = 1 / 9):
  boys_with_asthma = 31 ∧ girls_with_asthma = 109 :=
by
  sorry

end asthma_distribution_l27_27696


namespace num_turtles_on_sand_l27_27404

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l27_27404


namespace find_general_term_l27_27071

variable (a : ℕ → ℝ) (a1 : a 1 = 1)

def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

def isArithmeticSequence (u v w : ℝ) :=
  2 * v = u + w

theorem find_general_term (h1 : a 1 = 1)
  (h2 : (isGeometricSequence a (1 / 2)))
  (h3 : isArithmeticSequence (1 / a 1) (1 / a 3) (1 / a 4 - 1)) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
sorry

end find_general_term_l27_27071


namespace _l27_27438

open Matrix

noncomputable def matrix_2x2_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 5], ![-2, 9]]

@[simp] theorem inverse_correctness : 
  invOf (matrix_2x2_inverse) = ![![9/46, -5/46], ![2/46, 4/46]] :=
by
  sorry

end _l27_27438


namespace students_suggested_pasta_l27_27003

-- Define the conditions as variables in Lean
variable (total_students : ℕ := 470)
variable (suggested_mashed_potatoes : ℕ := 230)
variable (suggested_bacon : ℕ := 140)

-- The problem statement to prove
theorem students_suggested_pasta : 
  total_students - (suggested_mashed_potatoes + suggested_bacon) = 100 := by
  sorry

end students_suggested_pasta_l27_27003


namespace petya_coin_difference_20_l27_27937

-- Definitions for the problem conditions
variables (n k : ℕ) -- n: number of 5-ruble coins Petya has, k: number of 2-ruble coins Petya has

-- Condition: Petya has 60 rubles more than Vanya
def petya_has_60_more (n k : ℕ) : Prop := (5 * n + 2 * k = 5 * k + 2 * n + 60)

-- Theorem to prove Petya has 20 more 5-ruble coins than 2-ruble coins
theorem petya_coin_difference_20 (n k : ℕ) (h : petya_has_60_more n k) : n - k = 20 :=
sorry

end petya_coin_difference_20_l27_27937


namespace white_bread_served_l27_27936

theorem white_bread_served (total_bread : ℝ) (wheat_bread : ℝ) (white_bread : ℝ) 
  (h1 : total_bread = 0.9) (h2 : wheat_bread = 0.5) : white_bread = 0.4 :=
by
  sorry

end white_bread_served_l27_27936


namespace find_a_l27_27750

def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem find_a : (f_prime a (-1) = 3) → a = 3 :=
by
  sorry

end find_a_l27_27750


namespace gift_cost_calc_l27_27824

theorem gift_cost_calc (C N : ℕ) (hN : N = 12)
    (h : C / (N - 4) = C / N + 10) : C = 240 := by
  sorry

end gift_cost_calc_l27_27824


namespace minimum_value_of_f_maximum_value_of_k_l27_27758

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f : ∃ x : ℝ, 0 < x ∧ f x = -1 / Real.exp 1 :=
sorry

theorem maximum_value_of_k : ∀ x > 2, ∀ k : ℤ, (f x ≥ k * x - 2 * (k + 1)) → k ≤ 3 :=
sorry

end minimum_value_of_f_maximum_value_of_k_l27_27758


namespace radius_ratio_ge_sqrt2plus1_l27_27926

theorem radius_ratio_ge_sqrt2plus1 (r R a h : ℝ) (h1 : 2 * a ≠ 0) (h2 : h ≠ 0) 
  (hr : r = a * h / (a + Real.sqrt (a ^ 2 + h ^ 2)))
  (hR : R = (2 * a ^ 2 + h ^ 2) / (2 * h)) : 
  R / r ≥ 1 + Real.sqrt 2 := 
sorry

end radius_ratio_ge_sqrt2plus1_l27_27926


namespace marley_total_fruits_l27_27928

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end marley_total_fruits_l27_27928


namespace petya_coloring_failure_7_petya_coloring_failure_10_l27_27451

theorem petya_coloring_failure_7 :
  ¬ ∀ (points : Fin 200 → Fin 7) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

theorem petya_coloring_failure_10 :
  ¬ ∀ (points : Fin 200 → Fin 10) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

end petya_coloring_failure_7_petya_coloring_failure_10_l27_27451


namespace a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27264

variables (a b c d : ℝ)

-- Given conditions
def first_condition : Prop := a + b = c + d
def second_condition : Prop := a^3 + b^3 = c^3 + d^3

-- Proof problem for part (a)
theorem a_b_fifth_power_equals_c_d_fifth_power 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : a^5 + b^5 = c^5 + d^5 := 
sorry

-- Proof problem for part (b)
theorem cannot_conclude_fourth_powers 
  (h1 : first_condition a b c d) 
  (h2 : second_condition a b c d) : ¬ (a^4 + b^4 = c^4 + d^4) :=
sorry

end a_b_fifth_power_equals_c_d_fifth_power_cannot_conclude_fourth_powers_l27_27264


namespace remaining_hours_needed_l27_27687

noncomputable
def hours_needed_to_finish (x : ℚ) : Prop :=
  (1/5 : ℚ) * (2 + x) + (1/8 : ℚ) * x = 1

theorem remaining_hours_needed :
  ∃ x : ℚ, hours_needed_to_finish x ∧ x = 24/13 :=
by
  use 24/13
  sorry

end remaining_hours_needed_l27_27687


namespace road_length_kopatych_to_losyash_l27_27316

variable (T Krosh_dist Yozhik_dist : ℕ)
variable (d_k d_y r_k r_y : ℕ)

theorem road_length_kopatych_to_losyash : 
    (d_k = 20) → (d_y = 16) → (r_k = 30) → (r_y = 60) → 
    (Krosh_dist = 5 * T / 9) → (Yozhik_dist = 4 * T / 9) → 
    (T = Krosh_dist + r_k) →
    (T = Yozhik_dist + r_y) → 
    (T = 180) :=
by
  intros
  sorry

end road_length_kopatych_to_losyash_l27_27316


namespace even_square_minus_self_l27_27807

theorem even_square_minus_self (a : ℤ) : 2 ∣ (a^2 - a) :=
sorry

end even_square_minus_self_l27_27807


namespace sequence_arithmetic_l27_27582

-- Define the sequence and sum conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ) (p : ℝ)

-- We are given that the sum of the first n terms is Sn = n * p * a_n
axiom sum_condition (n : ℕ) (hpos : n > 0) : S n = n * p * a n

-- Also, given that a_1 ≠ a_2
axiom a1_ne_a2 : a 1 ≠ a 2

-- Define what we need to prove
theorem sequence_arithmetic (n : ℕ) (hn : n ≥ 2) :
  ∃ (a2 : ℝ), p = 1/2 ∧ a n = (n-1) * a2 :=
by
  sorry

end sequence_arithmetic_l27_27582


namespace masha_can_generate_all_integers_up_to_1093_l27_27482

theorem masha_can_generate_all_integers_up_to_1093 :
  ∃ (f : ℕ → ℤ), (∀ n, 1 ≤ n → n ≤ 1093 → f n ∈ {k | ∃ (a b c d e f g : ℤ), a * 1 + b * 3 + c * 9 + d * 27 + e * 81 + f * 243 + g * 729 = k}) :=
sorry

end masha_can_generate_all_integers_up_to_1093_l27_27482


namespace average_dandelions_picked_l27_27415

def Billy_initial_pick : ℕ := 36
def George_initial_ratio : ℚ := 1 / 3
def additional_picks : ℕ := 10

theorem average_dandelions_picked :
  let Billy_initial := Billy_initial_pick,
      George_initial := (George_initial_ratio * Billy_initial).toNat,
      Billy_total := Billy_initial + additional_picks,
      George_total := George_initial + additional_picks,
      total_picked := Billy_total + George_total in
  total_picked / 2 = 34 :=
  by
  let Billy_initial := Billy_initial_pick
  let George_initial := (George_initial_ratio * Billy_initial).toNat
  let Billy_total := Billy_initial + additional_picks
  let George_total := George_initial + additional_picks
  let total_picked := Billy_total + George_total
  sorry

end average_dandelions_picked_l27_27415


namespace total_cost_of_umbrellas_l27_27622

theorem total_cost_of_umbrellas : 
  ∀ (h_umbrellas c_umbrellas cost_per_umbrella : ℕ),
  h_umbrellas = 2 → 
  c_umbrellas = 1 → 
  cost_per_umbrella = 8 → 
  (h_umbrellas + c_umbrellas) * cost_per_umbrella = 24 :=
by 
  intros h_umbrellas c_umbrellas cost_per_umbrella h_eq c_eq cost_eq
  rw [h_eq, c_eq, cost_eq]
  sorry

end total_cost_of_umbrellas_l27_27622


namespace number_multiply_increase_l27_27227

theorem number_multiply_increase (x : ℕ) (h : 25 * x = 25 + 375) : x = 16 := by
  sorry

end number_multiply_increase_l27_27227


namespace bridge_extension_length_l27_27814

theorem bridge_extension_length (river_width bridge_length : ℕ) (h_river : river_width = 487) (h_bridge : bridge_length = 295) : river_width - bridge_length = 192 :=
by
  sorry

end bridge_extension_length_l27_27814


namespace max_value_of_e_l27_27493

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (5^n - 1) / 4

-- Define e_n as the gcd of b_n and b_(n+1)
def e (n : ℕ) : ℚ := Int.gcd (b n) (b (n + 1))

-- The theorem we need to prove is that e_n is always 1
theorem max_value_of_e (n : ℕ) : e n = 1 :=
  sorry

end max_value_of_e_l27_27493


namespace second_discount_percentage_l27_27549

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ) :
  original_price = 10000 →
  first_discount = 0.20 →
  final_price = 6840 →
  second_discount = 14.5 :=
by
  sorry

end second_discount_percentage_l27_27549


namespace mahesh_worked_days_l27_27499

-- Definitions
def mahesh_work_days := 45
def rajesh_work_days := 30
def total_work_days := 54

-- Theorem statement
theorem mahesh_worked_days (maheshrate : ℕ := mahesh_work_days) (rajeshrate : ℕ := rajesh_work_days) (totaldays : ℕ := total_work_days) :
  ∃ x : ℕ, x = totaldays - rajesh_work_days := by
  apply Exists.intro (54 - 30)
  simp
  sorry

end mahesh_worked_days_l27_27499


namespace number_eq_1925_l27_27606

theorem number_eq_1925 (x : ℝ) (h : x / 7 - x / 11 = 100) : x = 1925 :=
sorry

end number_eq_1925_l27_27606


namespace new_average_mark_of_remaining_students_l27_27179

def new_average (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ) : ℕ :=
  ((total_students * avg_marks) - (excluded_students * excluded_avg_marks)) / (total_students - excluded_students)

theorem new_average_mark_of_remaining_students 
  (total_students : ℕ) (excluded_students : ℕ) (avg_marks : ℕ) (excluded_avg_marks : ℕ)
  (h1 : total_students = 33)
  (h2 : excluded_students = 3)
  (h3 : avg_marks = 90)
  (h4 : excluded_avg_marks = 40) : 
  new_average total_students excluded_students avg_marks excluded_avg_marks = 95 :=
by
  sorry

end new_average_mark_of_remaining_students_l27_27179


namespace relationship_y1_y2_y3_l27_27456

theorem relationship_y1_y2_y3 :
  ∀ (y1 y2 y3 : ℝ), y1 = 6 ∧ y2 = 3 ∧ y3 = -2 → y1 > y2 ∧ y2 > y3 :=
by 
  intros y1 y2 y3 h
  sorry

end relationship_y1_y2_y3_l27_27456


namespace function_domain_l27_27136

theorem function_domain (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end function_domain_l27_27136


namespace product_remainder_mod_7_l27_27663

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l27_27663


namespace circle_in_quad_radius_l27_27862

theorem circle_in_quad_radius (AB BC CD DA : ℝ) (r : ℝ) (h₁ : AB = 15) (h₂ : BC = 10) (h₃ : CD = 8) (h₄ : DA = 13) :
  r = 2 * Real.sqrt 10 := 
by {
  sorry
  }

end circle_in_quad_radius_l27_27862


namespace money_made_per_minute_l27_27517

theorem money_made_per_minute (total_tshirts : ℕ) (time_minutes : ℕ) (black_tshirt_price white_tshirt_price : ℕ) (num_black num_white : ℕ) :
  total_tshirts = 200 →
  time_minutes = 25 →
  black_tshirt_price = 30 →
  white_tshirt_price = 25 →
  num_black = total_tshirts / 2 →
  num_white = total_tshirts / 2 →
  (num_black * black_tshirt_price + num_white * white_tshirt_price) / time_minutes = 220 :=
by
  sorry

end money_made_per_minute_l27_27517


namespace partition_exists_l27_27159

noncomputable def partition_nats (c : ℚ) (hc : c ≠ 1) : Prop :=
  ∃ (A B : Finset ℕ), (A ∩ B = ∅) ∧ (A ∪ B = Finset.univ) ∧ ∀ m n ∈ A, m ≠ n → m / n ≠ c ∧ ∀ m n ∈ B, m ≠ n → m / n ≠ c

theorem partition_exists (c : ℚ) (hc : c ≠ 1) : partition_nats c hc :=
  sorry

end partition_exists_l27_27159


namespace positive_integer_M_l27_27037

theorem positive_integer_M (M : ℕ) (h : 14^2 * 35^2 = 70^2 * M^2) : M = 7 :=
sorry

end positive_integer_M_l27_27037


namespace single_reduction_equivalent_l27_27369

/-- If a price is first reduced by 25%, and the new price is further reduced by 30%, 
the single percentage reduction equivalent to these two reductions together is 47.5%. -/
theorem single_reduction_equivalent :
  ∀ P : ℝ, (1 - 0.25) * (1 - 0.30) * P = P * (1 - 0.475) :=
by
  intros
  sorry

end single_reduction_equivalent_l27_27369


namespace rectangles_in_grid_at_least_three_cells_l27_27522

theorem rectangles_in_grid_at_least_three_cells :
  let number_of_rectangles (n : ℕ) := (n + 1).choose 2 * (n + 1).choose 2
  let single_cell_rectangles (n : ℕ) := n * n
  let one_by_two_or_two_by_one_rectangles (n : ℕ) := n * (n - 1) * 2
  let total_rectangles (n : ℕ) := number_of_rectangles n - (single_cell_rectangles n + one_by_two_or_two_by_one_rectangles n)
  total_rectangles 6 = 345 :=
by
  sorry

end rectangles_in_grid_at_least_three_cells_l27_27522


namespace pizza_slices_left_over_l27_27554

def small_pizza_slices : ℕ := 4
def large_pizza_slices : ℕ := 8
def small_pizzas_purchased : ℕ := 3
def large_pizzas_purchased : ℕ := 2
def george_slices : ℕ := 3
def bob_slices : ℕ := george_slices + 1
def susie_slices : ℕ := bob_slices / 2
def bill_slices : ℕ := 3
def fred_slices : ℕ := 3
def mark_slices : ℕ := 3

def total_pizza_slices : ℕ := (small_pizzas_purchased * small_pizza_slices) + (large_pizzas_purchased * large_pizza_slices)
def total_slices_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

theorem pizza_slices_left_over : total_pizza_slices - total_slices_eaten = 10 :=
by sorry

end pizza_slices_left_over_l27_27554


namespace two_pow_n_plus_one_divisible_by_three_l27_27728

theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h1 : n > 0) :
  (2 ^ n + 1) % 3 = 0 ↔ n % 2 = 1 := 
sorry

end two_pow_n_plus_one_divisible_by_three_l27_27728


namespace gain_percent_is_30_l27_27041

-- Given conditions
def CostPrice : ℕ := 100
def SellingPrice : ℕ := 130
def Gain : ℕ := SellingPrice - CostPrice
def GainPercent : ℕ := (Gain * 100) / CostPrice

-- The theorem to be proven
theorem gain_percent_is_30 :
  GainPercent = 30 := sorry

end gain_percent_is_30_l27_27041


namespace comparison_inequalities_l27_27749

open Real

theorem comparison_inequalities
  (m : ℝ) (h1 : 3 ^ m = Real.exp 1) 
  (a : ℝ) (h2 : a = cos m) 
  (b : ℝ) (h3 : b = 1 - 1/2 * m^2)
  (c : ℝ) (h4 : c = sin m / m) :
  c > a ∧ a > b := by
  sorry

end comparison_inequalities_l27_27749


namespace least_third_side_of_right_triangle_l27_27896

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l27_27896


namespace smallest_four_digit_multiple_of_53_l27_27995

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27995


namespace correct_equation_l27_27568

theorem correct_equation (x : ℝ) : 3 * x + 20 = 4 * x - 25 :=
by sorry

end correct_equation_l27_27568


namespace number_of_elements_l27_27518

theorem number_of_elements
  (init_avg : ℕ → ℝ)
  (correct_avg : ℕ → ℝ)
  (incorrect_num correct_num : ℝ)
  (h1 : ∀ n : ℕ, init_avg n = 17)
  (h2 : ∀ n : ℕ, correct_avg n = 20)
  (h3 : incorrect_num = 26)
  (h4 : correct_num = 56)
  : ∃ n : ℕ, n = 10 := sorry

end number_of_elements_l27_27518


namespace ninth_number_l27_27953

theorem ninth_number (S1 S2 Total N : ℕ)
  (h1 : S1 = 9 * 56)
  (h2 : S2 = 9 * 63)
  (h3 : Total = 17 * 59)
  (h4 : Total = S1 + S2 - N) :
  N = 68 :=
by 
  -- The proof is omitted, only the statement is needed.
  sorry

end ninth_number_l27_27953


namespace range_of_x_in_sqrt_x_plus_3_l27_27126

theorem range_of_x_in_sqrt_x_plus_3 (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 :=
by
  sorry

end range_of_x_in_sqrt_x_plus_3_l27_27126


namespace grocer_second_month_sale_l27_27851

theorem grocer_second_month_sale (sale_1 sale_3 sale_4 sale_5 sale_6 avg_sale n : ℕ) 
(h1 : sale_1 = 6435) 
(h3 : sale_3 = 6855) 
(h4 : sale_4 = 7230) 
(h5 : sale_5 = 6562) 
(h6 : sale_6 = 7391) 
(havg : avg_sale = 6900) 
(hn : n = 6) : 
  sale_2 = 6927 :=
by
  sorry

end grocer_second_month_sale_l27_27851


namespace find_p_plus_q_l27_27021

noncomputable def f (k p : ℚ) : ℚ := 5 * k^2 - 2 * k + p
noncomputable def g (k q : ℚ) : ℚ := 4 * k^2 + q * k - 6

theorem find_p_plus_q (p q : ℚ) (h : ∀ k : ℚ, f k p * g k q = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) :
  p + q = -3 :=
sorry

end find_p_plus_q_l27_27021


namespace even_function_f_l27_27743

noncomputable def f (a b c x : ℝ) := a * Real.cos x + b * x^2 + c

theorem even_function_f (a b c : ℝ) (h1 : f a b c 1 = 1) : f a b c (-1) = f a b c 1 := by
  sorry

end even_function_f_l27_27743


namespace third_smallest_four_digit_in_pascals_triangle_l27_27207

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (i j : ℕ), j ≤ i ∧ n = Nat.choose i j

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n : ℕ, is_in_pascals_triangle n ∧ is_four_digit_number n ∧
  (∀ m : ℕ, is_in_pascals_triangle m ∧ is_four_digit_number m 
   → m = 1000 ∨ m = 1001 ∨ m = n) ∧ n = 1002 := sorry

end third_smallest_four_digit_in_pascals_triangle_l27_27207


namespace _l27_27439

open Matrix

noncomputable def matrix_2x2_inverse : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 5], ![-2, 9]]

@[simp] theorem inverse_correctness : 
  invOf (matrix_2x2_inverse) = ![![9/46, -5/46], ![2/46, 4/46]] :=
by
  sorry

end _l27_27439


namespace germination_percentage_l27_27445

theorem germination_percentage (total_seeds_plot1 total_seeds_plot2 germinated_plot2_percentage total_germinated_percentage germinated_plot1_percentage : ℝ) 
  (plant1 : total_seeds_plot1 = 300) 
  (plant2 : total_seeds_plot2 = 200) 
  (germination2 : germinated_plot2_percentage = 0.35) 
  (total_germination : total_germinated_percentage = 0.23)
  (germinated_plot1 : germinated_plot1_percentage = 0.15) :
  (total_germinated_percentage * (total_seeds_plot1 + total_seeds_plot2) = 
    (germinated_plot2_percentage * total_seeds_plot2) + (germinated_plot1_percentage * total_seeds_plot1)) :=
by
  sorry

end germination_percentage_l27_27445


namespace solve_for_sum_l27_27596

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := -1
noncomputable def c : ℝ := Real.sqrt 26

theorem solve_for_sum :
  (a * (a - 4) = 5) ∧ (b * (b - 4) = 5) ∧ (c * (c - 4) = 5) ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a^2 + b^2 = c^2) → (a + b + c = 4 + Real.sqrt 26) :=
by
  sorry

end solve_for_sum_l27_27596


namespace ratio_d_s_proof_l27_27072

noncomputable def ratio_d_s (n : ℕ) (s d : ℝ) : ℝ :=
  d / s

theorem ratio_d_s_proof : ∀ (n : ℕ) (s d : ℝ), 
  (n = 30) → 
  ((n ^ 2 * s ^ 2) / (n * s + 2 * n * d) ^ 2 = 0.81) → 
  ratio_d_s n s d = 1 / 18 :=
by
  intros n s d h_n h_area
  sorry

end ratio_d_s_proof_l27_27072


namespace convex_polygon_max_interior_angles_l27_27613

theorem convex_polygon_max_interior_angles (n : ℕ) (h1 : n ≥ 3) (h2 : n < 360) :
  ∃ x, x ≤ 4 ∧ ∀ k, k > 4 → False :=
by
  sorry

end convex_polygon_max_interior_angles_l27_27613


namespace cos_neg_75_eq_l27_27562

noncomputable def cos_75_degrees : Real := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem cos_neg_75_eq : Real.cos (-(75 * Real.pi / 180)) = cos_75_degrees := by
  sorry

end cos_neg_75_eq_l27_27562


namespace solve_missing_number_l27_27948

theorem solve_missing_number (n : ℤ) (h : 121 * n = 75625) : n = 625 :=
sorry

end solve_missing_number_l27_27948


namespace inequality_solution_set_l27_27577

theorem inequality_solution_set (x : ℝ) :
  2 * x^2 - x ≤ 0 → 0 ≤ x ∧ x ≤ 1 / 2 :=
sorry

end inequality_solution_set_l27_27577


namespace peaches_eaten_l27_27374

theorem peaches_eaten (P B Baskets P_each R Boxes P_box : ℕ) 
  (h1 : B = 5) 
  (h2 : P_each = 25)
  (h3 : Baskets = B * P_each)
  (h4 : R = 8) 
  (h5 : P_box = 15)
  (h6 : Boxes = R * P_box)
  (h7 : P = Baskets - Boxes) : P = 5 :=
by sorry

end peaches_eaten_l27_27374


namespace storks_equal_other_birds_l27_27558

-- Definitions of initial numbers of birds
def initial_sparrows := 2
def initial_crows := 1
def initial_storks := 3
def initial_egrets := 0

-- Birds arriving initially
def sparrows_arrived := 1
def crows_arrived := 3
def storks_arrived := 6
def egrets_arrived := 4

-- Birds leaving after 15 minutes
def sparrows_left := 2
def crows_left := 0
def storks_left := 0
def egrets_left := 1

-- Additional birds arriving after 30 minutes
def additional_sparrows := 0
def additional_crows := 4
def additional_storks := 3
def additional_egrets := 0

-- Final counts
def final_sparrows := initial_sparrows + sparrows_arrived - sparrows_left + additional_sparrows
def final_crows := initial_crows + crows_arrived - crows_left + additional_crows
def final_storks := initial_storks + storks_arrived - storks_left + additional_storks
def final_egrets := initial_egrets + egrets_arrived - egrets_left + additional_egrets

def total_other_birds := final_sparrows + final_crows + final_egrets

-- Theorem statement
theorem storks_equal_other_birds : final_storks - total_other_birds = 0 := by
  sorry

end storks_equal_other_birds_l27_27558


namespace least_third_side_of_right_triangle_l27_27898

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l27_27898


namespace counting_arithmetic_progressions_l27_27915

open Finset

/-- The number of ways to choose 4 numbers from the first 1000 natural numbers to form an increasing arithmetic progression is 166167. -/
theorem counting_arithmetic_progressions :
  let n := 1000 in
  let count := ∑ d in range 334, n - 3*d in
  count = 166167 :=
by
  sorry

end counting_arithmetic_progressions_l27_27915


namespace girls_came_in_classroom_l27_27536

theorem girls_came_in_classroom (initial_boys initial_girls boys_left final_children girls_in_classroom : ℕ)
  (h1 : initial_boys = 5)
  (h2 : initial_girls = 4)
  (h3 : boys_left = 3)
  (h4 : final_children = 8)
  (h5 : girls_in_classroom = final_children - (initial_boys - boys_left)) :
  girls_in_classroom - initial_girls = 2 :=
by
  sorry

end girls_came_in_classroom_l27_27536


namespace coat_lifetime_15_l27_27623

noncomputable def coat_lifetime : ℕ :=
  let cost_coat_expensive := 300
  let cost_coat_cheap := 120
  let years_cheap := 5
  let year_saving := 120
  let duration_comparison := 30
  let yearly_cost_cheaper := cost_coat_cheap / years_cheap
  let yearly_savings := year_saving / duration_comparison
  let cost_savings := yearly_cost_cheaper * duration_comparison - cost_coat_expensive * duration_comparison / (yearly_savings + (cost_coat_expensive / cost_coat_cheap))
  cost_savings

theorem coat_lifetime_15 : coat_lifetime = 15 := by
  sorry

end coat_lifetime_15_l27_27623


namespace right_triangle_least_side_l27_27903

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l27_27903


namespace fabric_cut_l27_27111

/-- Given a piece of fabric that is 2/3 meter long,
we can cut a piece measuring 1/2 meter
by folding the original piece into four equal parts and removing one part. -/
theorem fabric_cut :
  ∃ (f : ℚ), f = (2/3 : ℚ) → ∃ (half : ℚ), half = (1/2 : ℚ) ∧ half = f * (3/4 : ℚ) :=
by
  sorry

end fabric_cut_l27_27111


namespace range_of_a_l27_27449

-- Definitions of sets A and B
def A (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| < 2
def B (x a : ℝ) : Prop := x^2 - (a + 1) * x + a < 0

-- The condition A ∩ B ≠ ∅
def nonempty_intersection (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a

-- Proving the required range of a
theorem range_of_a : {a : ℝ | nonempty_intersection a} = {a : ℝ | a < 1 ∨ a > 3} := by
  sorry

end range_of_a_l27_27449


namespace heat_of_reaction_correct_l27_27078

def delta_H_f_NH4Cl : ℝ := -314.43  -- Enthalpy of formation of NH4Cl in kJ/mol
def delta_H_f_H2O : ℝ := -285.83    -- Enthalpy of formation of H2O in kJ/mol
def delta_H_f_HCl : ℝ := -92.31     -- Enthalpy of formation of HCl in kJ/mol
def delta_H_f_NH4OH : ℝ := -80.29   -- Enthalpy of formation of NH4OH in kJ/mol

def delta_H_rxn : ℝ :=
  ((2 * delta_H_f_NH4OH) + (2 * delta_H_f_HCl)) -
  ((2 * delta_H_f_NH4Cl) + (2 * delta_H_f_H2O))

theorem heat_of_reaction_correct :
  delta_H_rxn = 855.32 :=
  by
    -- Calculation and proof steps go here
    sorry

end heat_of_reaction_correct_l27_27078


namespace simplify_rationalize_l27_27349

theorem simplify_rationalize : (1 : ℝ) / (1 + (1 / (real.sqrt 5 + 2))) = (real.sqrt 5 + 1) / 4 :=
by sorry

end simplify_rationalize_l27_27349


namespace average_age_of_inhabitants_l27_27818

theorem average_age_of_inhabitants (H M : ℕ) (avg_age_men avg_age_women : ℕ)
  (ratio_condition : 2 * M = 3 * H)
  (men_avg_age_condition : avg_age_men = 37)
  (women_avg_age_condition : avg_age_women = 42) :
  ((H * 37) + (M * 42)) / (H + M) = 40 :=
by
  sorry

end average_age_of_inhabitants_l27_27818


namespace range_of_x_of_sqrt_x_plus_3_l27_27146

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l27_27146


namespace arctan_tan_expr_is_75_degrees_l27_27420

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end arctan_tan_expr_is_75_degrees_l27_27420


namespace problem_solution_l27_27256

-- Define the given circle equation C
def circle_C_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 3 = 0

-- Define the line of symmetry
def line_symmetry_eq (x y : ℝ) : Prop := y = -x - 4

-- Define the symmetric circle equation
def sym_circle_eq (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

theorem problem_solution (x y : ℝ)
  (H1 : circle_C_eq x y)
  (H2 : line_symmetry_eq x y) :
  sym_circle_eq x y :=
sorry

end problem_solution_l27_27256


namespace larger_integer_of_two_with_difference_8_and_product_168_l27_27973

theorem larger_integer_of_two_with_difference_8_and_product_168 :
  ∃ (x y : ℕ), x > y ∧ x - y = 8 ∧ x * y = 168 ∧ x = 14 :=
by
  sorry

end larger_integer_of_two_with_difference_8_and_product_168_l27_27973


namespace rectangle_length_to_width_ratio_l27_27355

-- Define the side length of the square
def s : ℝ := 1 -- Since we only need the ratio, the actual length does not matter

-- Define the length and width of the large rectangle
def length_of_large_rectangle : ℝ := 3 * s
def width_of_large_rectangle : ℝ := 3 * s

-- Define the dimensions of the small rectangle
def length_of_rectangle : ℝ := 3 * s
def width_of_rectangle : ℝ := s

-- Proving that the length of the rectangle is 3 times its width
theorem rectangle_length_to_width_ratio : length_of_rectangle = 3 * width_of_rectangle := 
by
  -- The proof is omitted
  sorry

end rectangle_length_to_width_ratio_l27_27355


namespace right_triangle_shorter_leg_l27_27476

theorem right_triangle_shorter_leg (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
sorry

end right_triangle_shorter_leg_l27_27476


namespace coronavirus_transmission_l27_27963

theorem coronavirus_transmission:
  (∃ x: ℝ, (1 + x)^2 = 225) :=
by
  sorry

end coronavirus_transmission_l27_27963


namespace kids_go_to_camp_l27_27158

theorem kids_go_to_camp (total_kids : ℕ) (kids_stay_home : ℕ) (h1 : total_kids = 898051) (h2 : kids_stay_home = 268627) : total_kids - kids_stay_home = 629424 :=
by
  sorry

end kids_go_to_camp_l27_27158


namespace base_7_minus_base_8_to_decimal_l27_27416

theorem base_7_minus_base_8_to_decimal : 
  (5 * 7^4 + 4 * 7^3 + 3 * 7^2 + 2 * 7^1 + 1 * 7^0) - (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 8190 :=
by sorry

end base_7_minus_base_8_to_decimal_l27_27416


namespace andy_paint_total_l27_27557

-- Define the given ratio condition and green paint usage
def paint_ratio (blue green white : ℕ) : Prop :=
  blue / green = 1 / 2 ∧ white / green = 5 / 2

def green_paint_used (green : ℕ) : Prop :=
  green = 6

-- Define the proof goal: total paint used
def total_paint_used (blue green white : ℕ) : ℕ :=
  blue + green + white

-- The statement to be proved
theorem andy_paint_total (blue green white : ℕ)
  (h_ratio : paint_ratio blue green white)
  (h_green : green_paint_used green) :
  total_paint_used blue green white = 24 :=
  sorry

end andy_paint_total_l27_27557


namespace oranges_savings_l27_27628

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l27_27628


namespace rose_clothing_tax_l27_27165

theorem rose_clothing_tax {total_spent total_tax tax_other tax_clothing amount_clothing amount_food amount_other clothing_tax_rate : ℝ} 
  (h_total_spent : total_spent = 100)
  (h_amount_clothing : amount_clothing = 0.5 * total_spent)
  (h_amount_food : amount_food = 0.2 * total_spent)
  (h_amount_other : amount_other = 0.3 * total_spent)
  (h_no_tax_food : True)
  (h_tax_other_rate : tax_other = 0.08 * amount_other)
  (h_total_tax_rate : total_tax = 0.044 * total_spent)
  (h_calculate_tax_clothing : tax_clothing = total_tax - tax_other) :
  clothing_tax_rate = (tax_clothing / amount_clothing) * 100 → 
  clothing_tax_rate = 4 := 
by
  sorry

end rose_clothing_tax_l27_27165


namespace two_digit_integer_one_less_than_lcm_of_3_4_7_l27_27834

theorem two_digit_integer_one_less_than_lcm_of_3_4_7 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n + 1) % (Nat.lcm (Nat.lcm 3 4) 7) = 0 ∧ n = 83 := by
  sorry

end two_digit_integer_one_less_than_lcm_of_3_4_7_l27_27834


namespace shortest_side_of_triangle_l27_27556

noncomputable def triangle_shortest_side (AB : ℝ) (AD : ℝ) (DB : ℝ) (radius : ℝ) : ℝ :=
  let x := 6
  let y := 5
  2 * y

theorem shortest_side_of_triangle :
  let AB := 16
  let AD := 7
  let DB := 9
  let radius := 5
  AB = AD + DB →
  (AD = 7) ∧ (DB = 9) ∧ (radius = 5) →
  triangle_shortest_side AB AD DB radius = 10 :=
by
  intros h1 h2
  -- proof goes here
  sorry

end shortest_side_of_triangle_l27_27556


namespace fraction_simplification_l27_27213

theorem fraction_simplification : 
  ((2 * 7) * (6 * 14)) / ((14 * 6) * (2 * 7)) = 1 :=
by
  sorry

end fraction_simplification_l27_27213


namespace line_equation_sum_l27_27713

theorem line_equation_sum (m b x y : ℝ) (hx : x = 4) (hy : y = 2) (hm : m = -5) (hline : y = m * x + b) : m + b = 17 := by
  sorry

end line_equation_sum_l27_27713


namespace Beth_crayons_proof_l27_27561

def Beth_packs_of_crayons (packs_crayons : ℕ) (total_crayons extra_crayons : ℕ) : ℕ :=
  total_crayons - extra_crayons

theorem Beth_crayons_proof
  (packs_crayons : ℕ)
  (each_pack_contains total_crayons extra_crayons : ℕ)
  (h_each_pack : each_pack_contains = 10) 
  (h_extra : extra_crayons = 6)
  (h_total : total_crayons = 40) 
  (valid_packs : packs_crayons = (Beth_packs_of_crayons total_crayons extra_crayons / each_pack_contains)) :
  packs_crayons = 3 :=
by
  rw [h_each_pack, h_extra, h_total] at valid_packs
  sorry

end Beth_crayons_proof_l27_27561


namespace smallest_four_digit_multiple_of_53_l27_27993

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l27_27993


namespace monotone_increasing_interval_l27_27526

def f (x : ℝ) := x^2 - 2

theorem monotone_increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y :=
by
  sorry

end monotone_increasing_interval_l27_27526


namespace maria_savings_l27_27500

variable (S : ℝ) -- Define S as a real number (amount saved initially)

-- Conditions
def bike_cost : ℝ := 600
def additional_money : ℝ := 250 + 230

-- Theorem statement
theorem maria_savings : S + additional_money = bike_cost → S = 120 :=
by
  intro h -- Assume the hypothesis (condition)
  sorry -- Proof will go here

end maria_savings_l27_27500


namespace right_triangle_least_side_l27_27902

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l27_27902


namespace pencils_combined_length_l27_27312

theorem pencils_combined_length (length_pencil1 length_pencil2 : Nat) (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) :
  length_pencil1 + length_pencil2 = 24 := by
  sorry

end pencils_combined_length_l27_27312


namespace no_valid_pairs_l27_27576

theorem no_valid_pairs : ∀ (x y : ℕ), x > 0 → y > 0 → x^2 + y^2 + 1 = x^3 → false := 
by
  intros x y hx hy h
  sorry

end no_valid_pairs_l27_27576


namespace find_index_l27_27185

-- Declaration of sequence being arithmetic with first term 1 and common difference 3
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 1 + (n - 1) * 3

-- The theorem to be proven
theorem find_index (a : ℕ → ℤ) (h1 : arithmetic_sequence a) (h2 : a 672 = 2014) : 672 = 672 :=
by 
  sorry

end find_index_l27_27185


namespace algebraic_expression_evaluates_to_2_l27_27873

theorem algebraic_expression_evaluates_to_2 (x : ℝ) (h : x^2 + x - 5 = 0) : 
(x - 1)^2 - x * (x - 3) + (x + 2) * (x - 2) = 2 := 
by 
  sorry

end algebraic_expression_evaluates_to_2_l27_27873


namespace value_of_x2_minus_y2_l27_27115

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x2_minus_y2_l27_27115


namespace floor_equiv_l27_27872

theorem floor_equiv {n : ℤ} (h : n > 2) : 
  Int.floor ((n * (n + 1) : ℚ) / (4 * n - 2 : ℚ)) = Int.floor ((n + 1 : ℚ) / 4) := 
sorry

end floor_equiv_l27_27872


namespace total_toes_on_bus_l27_27331

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end total_toes_on_bus_l27_27331


namespace not_geometric_sequence_of_transformed_l27_27879

theorem not_geometric_sequence_of_transformed (a b c : ℝ) (q : ℝ) (hq : q ≠ 1) 
  (h_geometric : b = a * q ∧ c = b * q) :
  ¬ (∃ q' : ℝ, 1 - b = (1 - a) * q' ∧ 1 - c = (1 - b) * q') :=
by
  sorry

end not_geometric_sequence_of_transformed_l27_27879


namespace book_pages_l27_27505

theorem book_pages (P : ℝ) (h1 : 2/3 * P = 1/3 * P + 20) : P = 60 :=
by
  sorry

end book_pages_l27_27505


namespace quadratic_inequality_solution_l27_27019

theorem quadratic_inequality_solution
  (a : ℝ) :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end quadratic_inequality_solution_l27_27019


namespace cos2alpha_plus_sin2alpha_l27_27287

def point_angle_condition (x y : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  x = -3 ∧ y = 4 ∧ r = 5 ∧ x^2 + y^2 = r^2

theorem cos2alpha_plus_sin2alpha (α : ℝ) (x y r : ℝ)
  (h : point_angle_condition x y r α) : 
  (Real.cos (2 * α) + Real.sin (2 * α)) = -31/25 :=
by
  sorry

end cos2alpha_plus_sin2alpha_l27_27287


namespace parabola_focus_directrix_l27_27608

-- Definitions and conditions
def parabola (y a x : ℝ) : Prop := y^2 = a * x
def distance_from_focus_to_directrix (d : ℝ) : Prop := d = 2

-- Statement of the problem
theorem parabola_focus_directrix {a : ℝ} (h : parabola y a x) (h2 : distance_from_focus_to_directrix d) : 
  a = 4 ∨ a = -4 :=
sorry

end parabola_focus_directrix_l27_27608


namespace original_total_cost_l27_27477

-- Definitions based on the conditions
def price_jeans : ℝ := 14.50
def price_shirt : ℝ := 9.50
def price_jacket : ℝ := 21.00

def jeans_count : ℕ := 2
def shirts_count : ℕ := 4
def jackets_count : ℕ := 1

-- The proof statement
theorem original_total_cost :
  (jeans_count * price_jeans) + (shirts_count * price_shirt) + (jackets_count * price_jacket) = 88 := 
by
  sorry

end original_total_cost_l27_27477


namespace discriminant_quadratic_eqn_l27_27016

def a := 1
def b := 1
def c := -2
def Δ : ℤ := b^2 - 4 * a * c

theorem discriminant_quadratic_eqn : Δ = 9 := by
  sorry

end discriminant_quadratic_eqn_l27_27016


namespace find_n_l27_27062

theorem find_n 
  (num_engineers : ℕ) (num_technicians : ℕ) (num_workers : ℕ)
  (total_population : ℕ := num_engineers + num_technicians + num_workers)
  (systematic_sampling_inclusion_exclusion : ∀ n : ℕ, ∃ k : ℕ, n ∣ total_population ↔ n + 1 ≠ total_population) 
  (stratified_sampling_lcm : ∃ lcm : ℕ, lcm = Nat.lcm (Nat.lcm num_engineers num_technicians) num_workers)
  (total_population_is_36 : total_population = 36)
  (num_engineers_is_6 : num_engineers = 6)
  (num_technicians_is_12 : num_technicians = 12)
  (num_workers_is_18 : num_workers = 18) :
  ∃ n : ℕ, n = 6 :=
by
  sorry

end find_n_l27_27062


namespace cauchy_schwarz_equivalent_iag_l27_27578

theorem cauchy_schwarz_equivalent_iag (a b c d : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (Real.sqrt x * Real.sqrt y) ≤ (x + y) / 2) ↔
  ((a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)) := by
  sorry

end cauchy_schwarz_equivalent_iag_l27_27578


namespace lorry_sand_capacity_l27_27640

def cost_cement (bags : ℕ) (cost_per_bag : ℕ) : ℕ := bags * cost_per_bag
def total_cost (cement_cost : ℕ) (sand_cost : ℕ) : ℕ := cement_cost + sand_cost
def total_sand (sand_cost : ℕ) (cost_per_ton : ℕ) : ℕ := sand_cost / cost_per_ton
def sand_per_lorry (total_sand : ℕ) (lorries : ℕ) : ℕ := total_sand / lorries

theorem lorry_sand_capacity : 
  cost_cement 500 10 + (total_cost 5000 (total_sand 8000 40)) = 13000 ∧
  total_cost 5000 8000 = 13000 ∧
  total_sand 8000 40 = 200 ∧
  sand_per_lorry 200 20 = 10 :=
by
  sorry

end lorry_sand_capacity_l27_27640


namespace smallest_n_satisfying_inequality_l27_27081

noncomputable def sigma_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), Real.logb 10 (1 + 1 / 10 ^ (2 ^ k))

noncomputable def L : ℝ :=
  1 + Real.logb 10 (999 / 1000)

theorem smallest_n_satisfying_inequality :
  ∃ n : ℕ, (σ : sigma_n n) ≥ L ∧
           (∀ m : ℕ, (m < n) → (sigma_n m < L)) :=
by
  use 2
  sorry

end smallest_n_satisfying_inequality_l27_27081


namespace notAPrpos_l27_27841

def isProposition (s : String) : Prop :=
  s = "6 > 4" ∨ s = "If f(x) is a sine function, then f(x) is a periodic function." ∨ s = "1 ∈ {1, 2, 3}"

theorem notAPrpos (s : String) : ¬isProposition "Is a linear function an increasing function?" :=
by
  sorry

end notAPrpos_l27_27841


namespace simplify_expression_l27_27946

theorem simplify_expression (a : ℝ) :
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a :=
by
  sorry

end simplify_expression_l27_27946


namespace geom_sequence_sum_first_ten_terms_l27_27753

noncomputable def geom_sequence_sum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_sum_first_ten_terms (a : ℕ) (q : ℕ) (h1 : a * (1 + q) = 6) (h2 : a * q^3 * (1 + q) = 48) :
  geom_sequence_sum a q 10 = 2046 :=
sorry

end geom_sequence_sum_first_ten_terms_l27_27753


namespace square_side_increase_l27_27176

theorem square_side_increase (s : ℝ) :
  let new_side := 1.5 * s
  let new_area := new_side^2
  let original_area := s^2
  let new_perimeter := 4 * new_side
  let original_perimeter := 4 * s
  let new_diagonal := new_side * Real.sqrt 2
  let original_diagonal := s * Real.sqrt 2
  (new_area - original_area) / original_area * 100 = 125 ∧
  (new_perimeter - original_perimeter) / original_perimeter * 100 = 50 ∧
  (new_diagonal - original_diagonal) / original_diagonal * 100 = 50 :=
by
  sorry

end square_side_increase_l27_27176


namespace largest_corner_sum_l27_27566

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end largest_corner_sum_l27_27566


namespace sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27279

variable {a b c d : ℝ}

-- Part (a)
theorem sum_of_fifth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := sorry

-- Part (b)
theorem cannot_conclude_fourth_powers (h1 : a + b = c + d) (h2 : a^3 + b^3 = c^3 + d^3) : 
  ¬ (a^4 + b^4 = c^4 + d^4) := sorry

end sum_of_fifth_powers_cannot_conclude_fourth_powers_l27_27279


namespace man_and_son_together_days_l27_27715

noncomputable def man_days : ℝ := 7
noncomputable def son_days : ℝ := 5.25
noncomputable def combined_days : ℝ := man_days * son_days / (man_days + son_days)

theorem man_and_son_together_days :
  combined_days = 7 / 5 :=
by
  sorry

end man_and_son_together_days_l27_27715


namespace min_value_of_number_l27_27103

theorem min_value_of_number (a b c d : ℕ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 9) (h6 : 1 ≤ d) : 
  a + b * 10 + c * 100 + d * 1000 = 1119 :=
by
  sorry

end min_value_of_number_l27_27103


namespace eighth_grade_students_l27_27377

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end eighth_grade_students_l27_27377


namespace positive_integer_solutions_count_3x_plus_4y_eq_1024_l27_27528

theorem positive_integer_solutions_count_3x_plus_4y_eq_1024 :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = 1024) ∧ 
  (∀ n, n = 85 → ∃! (s : ℕ × ℕ), s.fst > 0 ∧ s.snd > 0 ∧ 3 * s.fst + 4 * s.snd = 1024 ∧ n = 85) := 
sorry

end positive_integer_solutions_count_3x_plus_4y_eq_1024_l27_27528


namespace derivative_sum_of_distinct_real_roots_l27_27291

noncomputable def f (a x : ℝ) := a * x - Real.log x
noncomputable def f' (a x : ℝ) := a - 1 / x

theorem derivative_sum_of_distinct_real_roots (a x1 x2 : ℝ) (h1 : a ∈ Set.Icc 0 (Real.log x2 + Real.log x1) / x2)
  (h2 : f a x1 = 0) (h3 : f a x2 = 0) (h4 : 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) :
  f' a x1 + f' a x2 < 0 := sorry

end derivative_sum_of_distinct_real_roots_l27_27291


namespace marching_band_formations_l27_27397

open Nat

theorem marching_band_formations :
  ∃ g, (g = 9) ∧ ∀ s t : ℕ, (s * t = 480 ∧ 15 ≤ t ∧ t ≤ 60) ↔ 
    (t = 15 ∨ t = 16 ∨ t = 20 ∨ t = 24 ∨ t = 30 ∨ t = 32 ∨ t = 40 ∨ t = 48 ∨ t = 60) :=
by
  -- Skipped proof.
  sorry

end marching_band_formations_l27_27397


namespace total_students_l27_27646

-- Define the problem statement in Lean 4
theorem total_students (n : ℕ) (h1 : n < 400)
  (h2 : n % 17 = 15) (h3 : n % 19 = 10) : n = 219 :=
sorry

end total_students_l27_27646


namespace range_of_independent_variable_l27_27140

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l27_27140


namespace pie_chart_shows_percentage_l27_27859

-- Define the different types of graphs
inductive GraphType
| PieChart
| BarGraph
| LineGraph
| Histogram

-- Define conditions of the problem
def shows_percentage_of_whole (g : GraphType) : Prop :=
  g = GraphType.PieChart

def displays_with_rectangular_bars (g : GraphType) : Prop :=
  g = GraphType.BarGraph

def shows_trends (g : GraphType) : Prop :=
  g = GraphType.LineGraph

def shows_frequency_distribution (g : GraphType) : Prop :=
  g = GraphType.Histogram

-- We need to prove that a pie chart satisfies the condition of showing percentages of parts in a whole
theorem pie_chart_shows_percentage : shows_percentage_of_whole GraphType.PieChart :=
  by
    -- Proof is skipped
    sorry

end pie_chart_shows_percentage_l27_27859


namespace least_side_is_8_l27_27888

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l27_27888


namespace number_of_intersections_of_lines_l27_27869

theorem number_of_intersections_of_lines : 
  let L1 := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 12}
  let L2 := {p : ℝ × ℝ | 5 * p.1 - 2 * p.2 = 10}
  let L3 := {p : ℝ × ℝ | p.1 = 3}
  let L4 := {p : ℝ × ℝ | p.2 = 1}
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ p1 ∈ L1 ∧ p1 ∈ L2 ∧ p2 ∈ L3 ∧ p2 ∈ L4 :=
by
  sorry

end number_of_intersections_of_lines_l27_27869


namespace school_selection_theorem_l27_27648

-- Define the basic setup and conditions
def school_selection_problem : Prop :=
  let schools := ["A", "B", "C", "D"]
  let total_schools := 4
  let selected_schools := 2
  let combinations := Nat.choose total_schools selected_schools
  let favorable_outcomes := Nat.choose (total_schools - 1) (selected_schools - 1)
  let probability := (favorable_outcomes : ℚ) / (combinations : ℚ)
  probability = 1 / 2

-- Proof is yet to be provided
theorem school_selection_theorem : school_selection_problem := sorry

end school_selection_theorem_l27_27648


namespace Marissa_sunflower_height_l27_27932

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end Marissa_sunflower_height_l27_27932


namespace rainfall_third_day_is_18_l27_27034

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end rainfall_third_day_is_18_l27_27034


namespace little_john_gave_to_each_friend_l27_27927

noncomputable def little_john_total : ℝ := 10.50
noncomputable def sweets : ℝ := 2.25
noncomputable def remaining : ℝ := 3.85

theorem little_john_gave_to_each_friend :
  (little_john_total - sweets - remaining) / 2 = 2.20 :=
by
  sorry

end little_john_gave_to_each_friend_l27_27927


namespace smallest_value_of_x_l27_27205

theorem smallest_value_of_x : ∃ x, (2 * x^2 + 30 * x - 84 = x * (x + 15)) ∧ (∀ y, (2 * y^2 + 30 * y - 84 = y * (y + 15)) → x ≤ y) ∧ x = -28 := by
  sorry

end smallest_value_of_x_l27_27205


namespace erin_trolls_count_l27_27433

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end erin_trolls_count_l27_27433


namespace range_of_sqrt_x_plus_3_l27_27142

theorem range_of_sqrt_x_plus_3 (x : ℝ) : 
  (∃ (y : ℝ), y = sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end range_of_sqrt_x_plus_3_l27_27142


namespace water_fee_20_water_fee_55_l27_27478

-- Define the water charge method as a function
def water_fee (a : ℕ) : ℝ :=
  if a ≤ 15 then 2 * a else 2.5 * a - 7.5

-- Prove the specific cases
theorem water_fee_20 :
  water_fee 20 = 42.5 :=
by sorry

theorem water_fee_55 :
  (∃ a : ℕ, water_fee a = 55) ↔ (a = 25) :=
by sorry

end water_fee_20_water_fee_55_l27_27478


namespace find_a_l27_27763

noncomputable def curve (x a : ℝ) : ℝ := 1/x + (Real.log x)/a
noncomputable def curve_derivative (x a : ℝ) : ℝ := 
  (-1/(x^2)) + (1/(a * x))

theorem find_a (a : ℝ) : 
  (curve_derivative 1 a = 3/2) ∧ ((∃ l : ℝ, curve 1 a = l) → ∃ m : ℝ, m * (-2/3) = -1)  → a = 2/5 :=
by
  sorry

end find_a_l27_27763


namespace burger_cost_l27_27410

theorem burger_cost 
    (b s : ℕ) 
    (h1 : 5 * b + 3 * s = 500) 
    (h2 : 3 * b + 2 * s = 310) :
    b = 70 := by
  sorry

end burger_cost_l27_27410


namespace smallest_four_digit_divisible_by_53_l27_27986

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27986


namespace Iesha_num_books_about_school_l27_27301

theorem Iesha_num_books_about_school (total_books sports_books : ℕ) (h1 : total_books = 58) (h2 : sports_books = 39) : total_books - sports_books = 19 :=
by
  sorry

end Iesha_num_books_about_school_l27_27301


namespace sqrt_domain_l27_27132

theorem sqrt_domain :
  ∀ x : ℝ, (∃ y : ℝ, y = real.sqrt (x + 3)) ↔ x ≥ -3 :=
by sorry

end sqrt_domain_l27_27132


namespace base_four_30121_eq_793_l27_27813

-- Definition to convert a base-four (radix 4) number 30121_4 to its base-ten equivalent
def base_four_to_base_ten (d4 d3 d2 d1 d0 : ℕ) : ℕ :=
  d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

theorem base_four_30121_eq_793 : base_four_to_base_ten 3 0 1 2 1 = 793 := 
by
  sorry

end base_four_30121_eq_793_l27_27813


namespace quadratic_inequality_solution_set_l27_27454

theorem quadratic_inequality_solution_set (m t : ℝ)
  (h : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - m*x + t < 0) : 
  m - t = -1 := sorry

end quadratic_inequality_solution_set_l27_27454


namespace min_value_of_a_l27_27871

theorem min_value_of_a :
  ∀ (x y : ℝ), |x| + |y| ≤ 1 → (|2 * x - 3 * y + 3 / 2| + |y - 1| + |2 * y - x - 3| ≤ 23 / 2) :=
by
  intros x y h
  sorry

end min_value_of_a_l27_27871


namespace range_of_x_of_sqrt_x_plus_3_l27_27147

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end range_of_x_of_sqrt_x_plus_3_l27_27147


namespace sample_mean_unbiased_unbiased_variance_unbiased_t_distribution_confidence_interval_μ_confidence_interval_σ_l27_27341

variables {α : Type*} [Fintype α] 
variables (a σ : ℝ) (n : ℕ) (xi : α → ℝ)

-- Define sample mean
def sample_mean (xi : α → ℝ) : ℝ := 
  (1 / n) * ∑ i, xi i 

-- Define unbiased estimator for variance
def unbiased_variance (xi : α → ℝ) (x̄ : ℝ) : ℝ := 
  (1 / (n - 1)) * ∑ i, (xi i - x̄) ^ 2

-- Define unbiased estimator for standard deviation
def unbiased_sd (xi : α → ℝ) (x̄ : ℝ) : ℝ := 
  sqrt (unbiased_variance xi x̄)

-- Sample mean as unbiased estimator of a 
theorem sample_mean_unbiased (h: n > 1) : 
  (sample_mean xi) = a := 
begin
  sorry
end

-- Unbiased variance as unbiased estimator of σ^2 
theorem unbiased_variance_unbiased (h: n > 1) : 
  (unbiased_variance xi (sample_mean xi)) = σ^2 := 
begin
  sorry
end

-- t_{n-1}(x) follows Student's t-distribution with n-1 degrees of freedom
theorem t_distribution (h: n > 1) : 
  ∃ t : α, (t (xi)) = (sample_mean xi - a) / (unbiased_sd xi (sample_mean xi) / sqrt n) := 
begin
  sorry
end

-- Construct 1-α confidence intervals for a
theorem confidence_interval_μ (h: n > 1) (α : ℝ) : 
  ∃ b, (b = sample_mean xi ± (Quantile.t_inv (α/2) (n - 1)) * (unbiased_sd xi (sample_mean xi) / sqrt n)) := 
begin
  sorry
end

-- Construct 1-α confidence intervals for σ
theorem confidence_interval_σ (h: n > 1) (α : ℝ) : 
  ∃ l u, (l, u = sqrt ((n - 1) * (unbiased_variance xi (sample_mean xi)) / Quantile.chi2 (1 - α / 2) (n - 1)), 
                      sqrt ((n - 1) * (unbiased_variance xi (sample_mean xi)) / Quantile.chi2 (α / 2) (n - 1))) := 
begin
  sorry
end

end sample_mean_unbiased_unbiased_variance_unbiased_t_distribution_confidence_interval_μ_confidence_interval_σ_l27_27341


namespace factorize_expression_l27_27435

variable (x : ℝ)

theorem factorize_expression : x^2 + x = x * (x + 1) :=
by
  sorry

end factorize_expression_l27_27435


namespace inequality_holds_l27_27961

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def even_function : Prop := ∀ x : ℝ, f x = f (-x)
def decreasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f y ≤ f x

-- Proof goal
theorem inequality_holds (h_even : even_function f) (h_decreasing : decreasing_on_pos f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := 
by
  sorry

end inequality_holds_l27_27961


namespace sons_ages_l27_27773

theorem sons_ages (x y : ℕ) (h1 : 2 * x = x + y + 18) (h2 : y = (x - y) - 6) : 
  x = 30 ∧ y = 12 := by
  sorry

end sons_ages_l27_27773


namespace speed_of_stream_l27_27022

theorem speed_of_stream (v : ℝ) 
    (h1 : ∀ (v : ℝ), v ≠ 0 → (80 / (36 + v) = 40 / (36 - v))) : 
    v = 12 := 
by 
    sorry

end speed_of_stream_l27_27022


namespace weight_difference_l27_27803

theorem weight_difference :
  let Box_A := 2.4
  let Box_B := 5.3
  let Box_C := 13.7
  let Box_D := 7.1
  let Box_E := 10.2
  let Box_F := 3.6
  let Box_G := 9.5
  max Box_A (max Box_B (max Box_C (max Box_D (max Box_E (max Box_F Box_G))))) -
  min Box_A (min Box_B (min Box_C (min Box_D (min Box_E (min Box_F Box_G))))) = 11.3 :=
by
  sorry

end weight_difference_l27_27803


namespace product_remainder_mod_7_l27_27668

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l27_27668


namespace probability_at_least_one_l27_27123

variable (p_A p_B : ℚ) (hA : p_A = 1 / 4) (hB : p_B = 2 / 5)

theorem probability_at_least_one (h : p_A * (1 - p_B) + (1 - p_A) * p_B + p_A * p_B = 11 / 20) : 
  (1 - (1 - p_A) * (1 - p_B) = 11 / 20) :=
by
  rw [hA, hB,←h]
  sorry

end probability_at_least_one_l27_27123


namespace verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l27_27035

variable (A B C P M N : ℝ)

-- Verification of Subtraction by Addition
theorem verify_sub_by_add (h : A - B = C) : C + B = A :=
sorry

-- Verification of Subtraction by Subtraction
theorem verify_sub_by_sub (h : A - B = C) : A - C = B :=
sorry

-- Verification of Multiplication by Division (1)
theorem verify_mul_by_div1 (h : M * N = P) : P / N = M :=
sorry

-- Verification of Multiplication by Division (2)
theorem verify_mul_by_div2 (h : M * N = P) : P / M = N :=
sorry

-- Verification of Multiplication by Multiplication
theorem verify_mul_by_mul (h : M * N = P) : M * N = P :=
sorry

end verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l27_27035


namespace remainder_when_divided_l27_27223

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_l27_27223


namespace bus_departure_l27_27193

theorem bus_departure (current_people : ℕ) (min_people : ℕ) (required_people : ℕ) 
  (h1 : current_people = 9) (h2 : min_people = 16) : required_people = 7 :=
by 
  sorry

end bus_departure_l27_27193


namespace smallest_m_satisfying_conditions_l27_27832

theorem smallest_m_satisfying_conditions :
  ∃ m : ℕ, m = 4 ∧ (∃ k : ℕ, 0 ≤ k ∧ k ≤ m ∧ (m^2 + m) % k ≠ 0) ∧ (∀ k : ℕ, (0 ≤ k ∧ k ≤ m) → (k ≠ 0 → (m^2 + m) % k = 0)) :=
sorry

end smallest_m_satisfying_conditions_l27_27832


namespace problem_I_l27_27707

theorem problem_I (x m : ℝ) (h1 : |x - m| < 1) (h2 : (1/3 : ℝ) < x ∧ x < (1/2 : ℝ)) : (-1/2 : ℝ) ≤ m ∧ m ≤ (4/3 : ℝ) :=
sorry

end problem_I_l27_27707


namespace box_volume_l27_27732

-- Given conditions
variables (a b c : ℝ)
axiom ab_eq : a * b = 30
axiom bc_eq : b * c = 18
axiom ca_eq : c * a = 45

-- Prove that the volume of the box (a * b * c) equals 90 * sqrt(3)
theorem box_volume : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end box_volume_l27_27732


namespace find_R_when_S_eq_5_l27_27923

theorem find_R_when_S_eq_5
  (g : ℚ)
  (h1 : ∀ S, R = g * S^2 - 6)
  (h2 : R = 15 ∧ S = 3) :
  R = 157 / 3 := by
    sorry

end find_R_when_S_eq_5_l27_27923


namespace total_soccer_balls_purchased_l27_27195

theorem total_soccer_balls_purchased : 
  (∃ (x : ℝ), 
    800 / x * 2 = 1560 / (x - 2)) → 
  (800 / x + 1560 / (x - 2) = 30) :=
by
  sorry

end total_soccer_balls_purchased_l27_27195


namespace product_remainder_mod_7_l27_27666

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l27_27666


namespace min_value_of_y_l27_27759

variable {x k : ℝ}

theorem min_value_of_y (h₁ : ∀ x > 0, 0 < k) 
  (h₂ : ∀ x > 0, (x^2 + k / x) ≥ 3) : k = 2 :=
sorry

end min_value_of_y_l27_27759


namespace prime_square_condition_no_prime_cube_condition_l27_27091

-- Part (a): Prove p = 3 given 8*p + 1 = n^2 and p is a prime
theorem prime_square_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 2) : 
  p = 3 :=
sorry

-- Part (b): Prove no p exists given 8*p + 1 = n^3 and p is a prime
theorem no_prime_cube_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 3) : 
  False :=
sorry

end prime_square_condition_no_prime_cube_condition_l27_27091


namespace minimum_ticket_cost_l27_27324

theorem minimum_ticket_cost :
  let num_people := 12
  let num_adults := 8
  let num_children := 4
  let adult_ticket_cost := 100
  let child_ticket_cost := 50
  let group_ticket_cost := 70
  num_people = num_adults + num_children →
  (num_people >= 10) →
  ∃ (cost : ℕ), cost = min (num_adults * adult_ticket_cost + num_children * child_ticket_cost) (group_ticket_cost * num_people) ∧
  cost = min (group_ticket_cost * 10 + child_ticket_cost * (num_people - 10)) (group_ticket_cost * num_people) →
  cost = 800 :=
by
  intro h1 h2
  sorry

end minimum_ticket_cost_l27_27324


namespace david_money_l27_27215

theorem david_money (S : ℝ) (h_initial : 1500 - S = S - 500) : 1500 - S = 500 :=
by
  sorry

end david_money_l27_27215


namespace op_exp_eq_l27_27464

-- Define the operation * on natural numbers
def op (a b : ℕ) : ℕ := a ^ b

-- The theorem to be proven
theorem op_exp_eq (a b n : ℕ) : (op a b)^n = op a (b^n) := by
  sorry

end op_exp_eq_l27_27464


namespace face_opposite_A_is_F_l27_27567

structure Cube where
  adjacency : String → String → Prop
  exists_face : ∃ a b c d e f : String, True

variable 
  (C : Cube)
  (adjA_B : C.adjacency "A" "B")
  (adjA_C : C.adjacency "A" "C")
  (adjB_D : C.adjacency "B" "D")

theorem face_opposite_A_is_F : 
  ∃ f : String, f = "F" ∧ ∀ g : String, (C.adjacency "A" g → g ≠ "F") :=
by 
  sorry

end face_opposite_A_is_F_l27_27567


namespace orange_ring_weight_l27_27315

theorem orange_ring_weight :
  ∀ (p w t o : ℝ), 
  p = 0.33 → w = 0.42 → t = 0.83 → t - (p + w) = o → 
  o = 0.08 :=
by
  intro p w t o hp hw ht h
  rw [hp, hw, ht] at h
  -- Additional steps would go here, but
  sorry -- Skipping the proof as instructed

end orange_ring_weight_l27_27315


namespace opposite_of_negative_2023_l27_27366

-- Define the opposite condition
def is_opposite (y x : Int) : Prop := y + x = 0

theorem opposite_of_negative_2023 : ∃ x : Int, is_opposite (-2023) x ∧ x = 2023 :=
by 
  use 2023
  sorry

end opposite_of_negative_2023_l27_27366


namespace intersection_M_N_l27_27595

noncomputable def set_M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
noncomputable def set_N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_M_N :
  (set_M ∩ set_N) = { x | -1 ≤ x ∧ x ≤ Real.sqrt 2 } := sorry

end intersection_M_N_l27_27595


namespace base_7_multiplication_addition_l27_27639

theorem base_7_multiplication_addition :
  (25 * 3 + 144) % 7^3 = 303 :=
by sorry

end base_7_multiplication_addition_l27_27639


namespace smallest_four_digit_divisible_by_53_l27_27983

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27983


namespace book_selling_price_l27_27394

def cost_price : ℕ := 225
def profit_percentage : ℚ := 0.20
def selling_price := cost_price + (profit_percentage * cost_price)

theorem book_selling_price :
  selling_price = 270 :=
by
  sorry

end book_selling_price_l27_27394


namespace proof_problem_l27_27114

noncomputable def log (x : ℝ) : ℝ := real.log x

theorem proof_problem :
  let a := log 8
  let b := log 25
  5^(a/b) + 2^(b/a) = 2 * real.sqrt 2 + 5^(2/3) :=
by
  sorry

end proof_problem_l27_27114


namespace triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l27_27530

-- Definition of the sides according to Plato's rule
def triangle_sides (p : ℕ) : ℕ × ℕ × ℕ :=
  (2 * p, p^2 - 1, p^2 + 1)

-- Function to check if the given sides form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Theorems to verify the sides of the triangle for given p values
theorem triangle_sides_p2 : triangle_sides 2 = (4, 3, 5) ∧ is_right_triangle 4 3 5 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p3 : triangle_sides 3 = (6, 8, 10) ∧ is_right_triangle 6 8 10 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p4 : triangle_sides 4 = (8, 15, 17) ∧ is_right_triangle 8 15 17 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p5 : triangle_sides 5 = (10, 24, 26) ∧ is_right_triangle 10 24 26 :=
by {
  sorry -- Proof goes here
}

end triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l27_27530


namespace num_sets_N_l27_27107

open Set

noncomputable def M : Set ℤ := {-1, 0}

theorem num_sets_N (N : Set ℤ) : M ∪ N = {-1, 0, 1} → 
  (N = {1} ∨ N = {0, 1} ∨ N = {-1, 1} ∨ N = {0, -1, 1}) := 
sorry

end num_sets_N_l27_27107


namespace log2_square_eq_37_l27_27938

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_square_eq_37
  {x y : ℝ}
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_log : log2 x = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (log2 (x / y))^2 = 37 := by
  sorry

end log2_square_eq_37_l27_27938


namespace simplest_quadratic_radical_l27_27837

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end simplest_quadratic_radical_l27_27837


namespace string_length_is_correct_l27_27550

noncomputable def calculate_string_length (circumference height : ℝ) (loops : ℕ) : ℝ :=
  let vertical_distance_per_loop := height / loops
  let hypotenuse_length := Real.sqrt ((circumference ^ 2) + (vertical_distance_per_loop ^ 2))
  loops * hypotenuse_length

theorem string_length_is_correct : calculate_string_length 6 16 5 = 34 := 
  sorry

end string_length_is_correct_l27_27550


namespace necessary_but_not_sufficient_l27_27956

theorem necessary_but_not_sufficient (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 → m < 2 :=
by
  sorry

end necessary_but_not_sufficient_l27_27956


namespace positive_divisors_d17_l27_27217

theorem positive_divisors_d17 (n : ℕ) (d : ℕ → ℕ) (k : ℕ) (h_order : d 1 = 1 ∧ ∀ i, 1 ≤ i → i ≤ k → d i < d (i + 1)) 
  (h_last : d k = n) (h_pythagorean : d 7 ^ 2 + d 15 ^ 2 = d 16 ^ 2) : 
  d 17 = 28 :=
sorry

end positive_divisors_d17_l27_27217


namespace equation_solution_unique_or_not_l27_27447

theorem equation_solution_unique_or_not (a b : ℝ) :
  (∃ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x - a) / (x - 2) + (x - b) / (x - 3) = 2) ↔ 
  (a = 2 ∧ b = 3) ∨ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
by
  sorry

end equation_solution_unique_or_not_l27_27447


namespace expected_non_empty_urns_correct_l27_27786

open ProbabilityTheory

noncomputable def expected_non_empty_urns (n k : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n) ^ k)

theorem expected_non_empty_urns_correct (n k : ℕ) : expected_non_empty_urns n k = n * (1 - ((n - 1) / n) ^ k) :=
by 
  sorry

end expected_non_empty_urns_correct_l27_27786


namespace sum_of_three_fractions_is_one_l27_27094

theorem sum_of_three_fractions_is_one (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a : ℝ) + 1 / (b : ℝ) + 1 / (c : ℝ) = 1 ↔ 
  (a = 3 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 6 ∧ c = 3) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 6) ∨ 
  (a = 3 ∧ b = 6 ∧ c = 2) :=
by sorry

end sum_of_three_fractions_is_one_l27_27094


namespace salary_for_january_l27_27845

theorem salary_for_january (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8700)
  (h_may : May = 6500) :
  J = 3700 :=
by
  sorry

end salary_for_january_l27_27845


namespace intersection_M_N_l27_27498

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l27_27498


namespace total_people_selected_l27_27971

-- Define the number of residents in each age group
def residents_21_to_35 : Nat := 840
def residents_36_to_50 : Nat := 700
def residents_51_to_65 : Nat := 560

-- Define the number of people selected from the 36 to 50 age group
def selected_36_to_50 : Nat := 100

-- Define the total number of residents
def total_residents : Nat := residents_21_to_35 + residents_36_to_50 + residents_51_to_65

-- Theorem: Prove that the total number of people selected in this survey is 300
theorem total_people_selected : (100 : ℕ) / (700 : ℕ) * (residents_21_to_35 + residents_36_to_50 + residents_51_to_65) = 300 :=
  by 
    sorry

end total_people_selected_l27_27971


namespace sin_2_angle_CPD_correct_l27_27340

noncomputable def sin_2_angle_CPD (cos_alpha cos_beta : ℝ) (h_cos_alpha : cos_alpha = 3/5) (h_cos_beta : cos_beta = 2/5) : ℝ :=
  let alpha := real.arccos cos_alpha
  let beta := real.arccos cos_beta
  let gamma := alpha + beta
  2 * real.sin gamma * real.cos gamma

theorem sin_2_angle_CPD_correct : 
  (sin_2_angle_CPD (3/5) (2/5) (by norm_num) (by norm_num)) = (64 + 24 * real.sqrt 21) / 125 :=
by
  sorry

end sin_2_angle_CPD_correct_l27_27340


namespace find_t_l27_27101

theorem find_t
  (x y t : ℝ)
  (h1 : 2 ^ x = t)
  (h2 : 5 ^ y = t)
  (h3 : 1 / x + 1 / y = 2)
  (h4 : t ≠ 1) : 
  t = Real.sqrt 10 := 
by
  sorry

end find_t_l27_27101


namespace equal_points_probability_not_always_increasing_l27_27695

theorem equal_points_probability_not_always_increasing 
  (p q : ℝ) (h₀ : 0 ≤ p) (h₁ : p ≤ q) (h₂ : q ≤ 1) :
  ¬ ∀ p₀ p₁, 0 ≤ p₀ ∧ p₀ ≤ p₁ ∧ p₁ ≤ 1 → 
    let f := λ x : ℝ, (3 * x^2 - 2 * x + 1) / 4 in
    f p₀ ≤ f p₁ := by
    sorry

end equal_points_probability_not_always_increasing_l27_27695


namespace domain_of_f_l27_27245

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ -3) ↔ ((x < -3) ∨ (-3 < x ∧ x < 3) ∨ (x > 3)) :=
by
  sorry

end domain_of_f_l27_27245


namespace smallest_four_digit_divisible_by_53_l27_27991

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l27_27991


namespace sum_gt_two_l27_27290

noncomputable def f (x : ℝ) : ℝ := ((x - 1) * Real.log x) / x

theorem sum_gt_two (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ > 2 := 
sorry

end sum_gt_two_l27_27290


namespace cost_of_one_dozen_pens_is_780_l27_27360

-- Defining the cost of pens and pencils
def cost_of_pens (n : ℕ) := n * 65

def cost_of_pencils (m : ℕ) := m * 13

-- Given conditions
def total_cost (x y : ℕ) := cost_of_pens x + cost_of_pencils y

theorem cost_of_one_dozen_pens_is_780
  (h1 : total_cost 3 5 = 260)
  (h2 : 65 = 5 * 13)
  (h3 : 65 = 65) :
  12 * 65 = 780 := by
    sorry

end cost_of_one_dozen_pens_is_780_l27_27360


namespace martha_flower_cost_l27_27325

theorem martha_flower_cost :
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  total_cost = 2700 :=
by
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  -- Proof to be added here
  sorry

end martha_flower_cost_l27_27325
