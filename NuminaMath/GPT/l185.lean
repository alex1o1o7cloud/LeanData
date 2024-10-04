import Mathlib

namespace ratio_of_areas_l185_185130

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l185_185130


namespace interior_angles_sum_l185_185114

def sum_of_interior_angles (sides : ℕ) : ℕ :=
  180 * (sides - 2)

theorem interior_angles_sum (n : ℕ) (h : sum_of_interior_angles n = 1800) :
  sum_of_interior_angles (n + 4) = 2520 :=
sorry

end interior_angles_sum_l185_185114


namespace james_spends_252_per_week_l185_185832

noncomputable def cost_pistachios_per_ounce := 10 / 5
noncomputable def cost_almonds_per_ounce := 8 / 4
noncomputable def cost_walnuts_per_ounce := 12 / 6

noncomputable def daily_consumption_pistachios := 30 / 5
noncomputable def daily_consumption_almonds := 24 / 4
noncomputable def daily_consumption_walnuts := 18 / 3

noncomputable def weekly_consumption_pistachios := daily_consumption_pistachios * 7
noncomputable def weekly_consumption_almonds := daily_consumption_almonds * 7
noncomputable def weekly_consumption_walnuts := daily_consumption_walnuts * 7

noncomputable def weekly_cost_pistachios := weekly_consumption_pistachios * cost_pistachios_per_ounce
noncomputable def weekly_cost_almonds := weekly_consumption_almonds * cost_almonds_per_ounce
noncomputable def weekly_cost_walnuts := weekly_consumption_walnuts * cost_walnuts_per_ounce

noncomputable def total_weekly_cost := weekly_cost_pistachios + weekly_cost_almonds + weekly_cost_walnuts

theorem james_spends_252_per_week :
  total_weekly_cost = 252 := by
  sorry

end james_spends_252_per_week_l185_185832


namespace unique_pair_natural_numbers_l185_185667

theorem unique_pair_natural_numbers (a b : ℕ) :
  (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by
  sorry

end unique_pair_natural_numbers_l185_185667


namespace problem1_problem2_l185_185192

def f (x : ℝ) := |x - 1| + |x + 2|

def T (a : ℝ) := -Real.sqrt 3 < a ∧ a < Real.sqrt 3

theorem problem1 (a : ℝ) : (∀ x : ℝ, f x > a^2) ↔ T a :=
by
  sorry

theorem problem2 (m n : ℝ) (h1 : T m) (h2 : T n) : Real.sqrt 3 * |m + n| < |m * n + 3| :=
by
  sorry

end problem1_problem2_l185_185192


namespace test_methods_first_last_test_methods_within_six_l185_185941

open Classical

def perms (n k : ℕ) : ℕ := sorry -- placeholder for permutation function

theorem test_methods_first_last
  (prod_total : ℕ) (defective : ℕ) (first_test : ℕ) (last_test : ℕ) 
  (A4_2 : ℕ) (A5_2 : ℕ) (A6_4 : ℕ) : first_test = 2 → last_test = 8 → 
  perms 4 2 * perms 5 2 * perms 6 4 = A4_2 * A5_2 * A6_4 :=
by
  intro h_first_test h_last_test
  simp [perms]
  sorry

theorem test_methods_within_six
  (prod_total : ℕ) (defective : ℕ) 
  (A4_4 : ℕ) (A4_3_A6_1 : ℕ) (A5_3_A6_2 : ℕ) (A6_6 : ℕ)
  : perms 4 4 + 4 * perms 4 3 * perms 6 1 + 4 * perms 5 3 * perms 6 2 + perms 6 6 
  = A4_4 + 4 * A4_3_A6_1 + 4 * A5_3_A6_2 + A6_6 :=
by
  simp [perms]
  sorry

end test_methods_first_last_test_methods_within_six_l185_185941


namespace find_g_inv_84_l185_185213

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end find_g_inv_84_l185_185213


namespace rearrange_letters_no_adjacent_repeats_l185_185956

-- Factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Problem conditions
def distinct_permutations (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  factorial (String.length word) / (factorial freq_I * factorial freq_L)

-- No-adjacent-repeated permutations
def no_adjacent_repeats (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  let total_permutations := distinct_permutations word freq_I freq_L
  let i_superletter_permutations := distinct_permutations (String.dropRight word 1) (freq_I - 1) freq_L
  let l_superletter_permutations := distinct_permutations (String.dropRight word 1) freq_I (freq_L - 1)
  let both_superletter_permutations := factorial (String.length word - 2)
  total_permutations - (i_superletter_permutations + l_superletter_permutations - both_superletter_permutations)

-- Given problem definition
def word := "BRILLIANT"
def freq_I := 2
def freq_L := 2

-- Proof problem statement
theorem rearrange_letters_no_adjacent_repeats :
  no_adjacent_repeats word freq_I freq_L = 55440 := by
  sorry

end rearrange_letters_no_adjacent_repeats_l185_185956


namespace find_x_pow_y_l185_185791

theorem find_x_pow_y (x y : ℝ) : |x + 2| + (y - 3)^2 = 0 → x ^ y = -8 :=
by
  sorry

end find_x_pow_y_l185_185791


namespace value_of_b_l185_185861

noncomputable def k := 675

theorem value_of_b (a b : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) (h4 : a = -12) :
  b = -56.25 := by
  sorry

end value_of_b_l185_185861


namespace smallest_sum_97_l185_185111

theorem smallest_sum_97 (X Y Z W : ℕ) 
  (h1 : X + Y + Z = 3)
  (h2 : 4 * Z = 7 * Y)
  (h3 : 16 ∣ Y) : 
  X + Y + Z + W = 97 :=
by
  sorry

end smallest_sum_97_l185_185111


namespace max_points_of_intersection_l185_185580

-- Define the conditions
variable {α : Type*} [DecidableEq α]
variable (L : Fin 100 → α → α → Prop) -- Representation of the lines

-- Define property of being parallel
variable (are_parallel : ∀ {n : ℕ}, L (5 * n) = L (5 * n + 5))

-- Define property of passing through point B
variable (passes_through_B : ∀ {n : ℕ}, ∃ P B, L (5 * n - 4) P B)

-- Prove the stated result
theorem max_points_of_intersection : 
  ∃ max_intersections, max_intersections = 4571 :=
by {
  sorry
}

end max_points_of_intersection_l185_185580


namespace whisky_replacement_l185_185763

variable (V x : ℝ)

/-- The initial whisky in the jar contains 40% alcohol -/
def initial_volume_of_alcohol (V : ℝ) : ℝ := 0.4 * V

/-- A part (x liters) of this whisky is replaced by another containing 19% alcohol -/
def volume_replaced_whisky (x : ℝ) : ℝ := x
def remaining_whisky (V x : ℝ) : ℝ := V - x

/-- The percentage of alcohol in the jar after replacement is 24% -/
def final_volume_of_alcohol (V x : ℝ) : ℝ := 0.4 * (remaining_whisky V x) + 0.19 * (volume_replaced_whisky x)

/- Prove that the quantity of whisky replaced is 0.16/0.21 times the total volume -/
theorem whisky_replacement :
  final_volume_of_alcohol V x = 0.24 * V → x = (0.16 / 0.21) * V :=
by sorry

end whisky_replacement_l185_185763


namespace weight_problem_l185_185006

variable (M T : ℕ)

theorem weight_problem
  (h1 : 220 = 3 * M + 10)
  (h2 : T = 2 * M)
  (h3 : 2 * T = 220) :
  M = 70 ∧ T = 140 :=
by
  sorry

end weight_problem_l185_185006


namespace equation_of_line_AB_l185_185354

def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def on_circle (C : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - C.1) ^ 2 + P.2 ^ 2 = r ^ 2

theorem equation_of_line_AB : 
  ∃ A B : ℝ × ℝ, 
    is_midpoint (2, -1) A B ∧ 
    on_circle (1, 0) 5 A ∧ 
    on_circle (1, 0) 5 B ∧ 
    ∀ x y : ℝ, (x - y - 3 = 0) ∧ 
    ∃ t : ℝ, ∃ u : ℝ, (t - u - 3 = 0) := 
sorry

end equation_of_line_AB_l185_185354


namespace eval_expression_l185_185927

theorem eval_expression : (538 * 538) - (537 * 539) = 1 :=
by
  sorry

end eval_expression_l185_185927


namespace find_unknown_number_l185_185511

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l185_185511


namespace sum_of_digits_of_n_l185_185245

theorem sum_of_digits_of_n : 
  ∃ n : ℕ, n > 1500 ∧ 
    (Nat.gcd 40 (n + 105) = 10) ∧ 
    (Nat.gcd (n + 40) 105 = 35) ∧ 
    (Nat.digits 10 n).sum = 8 :=
by 
  sorry

end sum_of_digits_of_n_l185_185245


namespace intersection_A_B_l185_185536

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l185_185536


namespace square_tiles_count_l185_185879

theorem square_tiles_count (t s p : ℕ) (h1 : t + s + p = 30) (h2 : 3 * t + 4 * s + 5 * p = 108) : s = 6 := by
  sorry

end square_tiles_count_l185_185879


namespace continuity_at_x0_l185_185751

noncomputable def f (x : ℝ) : ℝ := -4 * x^2 - 7

theorem continuity_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → |f x - f 1| < ε :=
by
  sorry

end continuity_at_x0_l185_185751


namespace apple_tree_total_apples_l185_185892

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l185_185892


namespace inradius_of_triangle_l185_185823

theorem inradius_of_triangle (A p s r : ℝ) 
  (h1 : A = (1/2) * p) 
  (h2 : p = 2 * s) 
  (h3 : A = r * s) : 
  r = 1 :=
by
  sorry

end inradius_of_triangle_l185_185823


namespace probability_is_18_over_25_l185_185643

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l185_185643


namespace trig_expression_value_quadratic_roots_l185_185908

theorem trig_expression_value :
  (Real.tan (Real.pi / 6))^2 + 2 * Real.sin (Real.pi / 4) - 2 * Real.cos (Real.pi / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

theorem quadratic_roots :
  (∀ x : ℝ, 2 * x^2 + 4 * x + 1 = 0 ↔ x = (-2 + Real.sqrt 2) / 2 ∨ x = (-2 - Real.sqrt 2) / 2) := by
  sorry

end trig_expression_value_quadratic_roots_l185_185908


namespace incorrect_statement_l185_185582

/-- Conditions for meiosis and fertilization for determining the incorrect statement. -/
constants (A B D : Prop)
  
/-- After proving the conditions of meiosis and fertilization correctly, we need to prove that C is incorrect. -/
theorem incorrect_statement (A_correct : A) (B_correct : B) (D_correct : D) : ¬C :=
sorry

end incorrect_statement_l185_185582


namespace eq_neg_one_fifth_l185_185906

theorem eq_neg_one_fifth : 
  ((1 : ℝ) / ((-5) ^ 4) ^ 2 * (-5) ^ 7) = -1 / 5 := by
  sorry

end eq_neg_one_fifth_l185_185906


namespace cone_volume_proof_l185_185050

noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cone_volume_proof :
  (cone_volume 1 (Real.sqrt 3)) = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end cone_volume_proof_l185_185050


namespace shells_total_l185_185085

theorem shells_total (a s v : ℕ) 
  (h1 : s = v + 16) 
  (h2 : v = a - 5) 
  (h3 : a = 20) : 
  s + v + a = 66 := 
by
  sorry

end shells_total_l185_185085


namespace factorize_x9_minus_512_l185_185912

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l185_185912


namespace avg_waiting_time_waiting_time_equivalence_l185_185480

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l185_185480


namespace domain_of_f_2x_minus_1_l185_185542

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) →
  ∀ x, (1 / 2) ≤ x ∧ x ≤ (3 / 2) → ∃ y, f y = (2 * x - 1) :=
by
  intros h x hx
  sorry

end domain_of_f_2x_minus_1_l185_185542


namespace totalCostOfFencing_l185_185813

def numberOfSides : ℕ := 4
def costPerSide : ℕ := 79

theorem totalCostOfFencing (n : ℕ) (c : ℕ) (hn : n = numberOfSides) (hc : c = costPerSide) : n * c = 316 :=
by 
  rw [hn, hc]
  exact rfl

end totalCostOfFencing_l185_185813


namespace evalCeilingOfNegativeSqrt_l185_185178

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end evalCeilingOfNegativeSqrt_l185_185178


namespace q_joins_after_2_days_l185_185622

-- Define the conditions
def work_rate_p := 1 / 10
def work_rate_q := 1 / 6
def total_days := 5

-- Define the proof problem
theorem q_joins_after_2_days (a b : ℝ) (t x : ℕ) : 
  a = work_rate_p → b = work_rate_q → t = total_days →
  x * a + (t - x) * (a + b) = 1 → 
  x = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end q_joins_after_2_days_l185_185622


namespace algebraic_expression_evaluation_l185_185201

theorem algebraic_expression_evaluation (x m : ℝ) (h1 : 5 * (2 - 1) + 3 * m * 2 = -7) (h2 : m = -2) :
  5 * (x - 1) + 3 * m * x = -1 ↔ x = -4 :=
by
  sorry

end algebraic_expression_evaluation_l185_185201


namespace square_lawn_side_length_l185_185704

theorem square_lawn_side_length (length width : ℕ) (h_length : length = 18) (h_width : width = 8) : 
  ∃ x : ℕ, x * x = length * width ∧ x = 12 := by
  -- Assume the necessary definitions and theorems to build the proof
  sorry

end square_lawn_side_length_l185_185704


namespace coins_problem_l185_185210

theorem coins_problem
  (N Q : ℕ)
  (h1 : N + Q = 21)
  (h2 : 0.05 * N + 0.25 * Q = 3.65) :
  Q = 13 :=
by
  -- To be proved
  sorry

end coins_problem_l185_185210


namespace simplify_expression_l185_185993

theorem simplify_expression (x : ℤ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 24 = 45 * x + 42 := 
by 
  -- proof steps
  sorry

end simplify_expression_l185_185993


namespace pear_sales_l185_185454

theorem pear_sales (sale_afternoon : ℕ) (h1 : sale_afternoon = 260)
  (h2 : ∃ sale_morning : ℕ, sale_afternoon = 2 * sale_morning) :
  sale_afternoon / 2 + sale_afternoon = 390 :=
by
  sorry

end pear_sales_l185_185454


namespace parabola_coordinates_and_area_l185_185042

theorem parabola_coordinates_and_area
  (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (3, 0))
  (hC : C = (5 / 2, 1 / 4))
  (h_vertex : ∀ x y, y = -x^2 + 5 * x - 6 → 
                   ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C)) :
  A = (2, 0) ∧ B = (3, 0) ∧ C = (5 / 2, 1 / 4)
  ∧ (1 / 2 * (3 - 2) * (1 / 4) = 1 / 8) := 
by
  sorry

end parabola_coordinates_and_area_l185_185042


namespace jason_seashells_initial_count_l185_185974

variable (initialSeashells : ℕ) (seashellsGivenAway : ℕ)
variable (seashellsNow : ℕ) (initialSeashells := 49)
variable (seashellsGivenAway := 13) (seashellsNow := 36)

theorem jason_seashells_initial_count :
  initialSeashells - seashellsGivenAway = seashellsNow → initialSeashells = 49 := by
  sorry

end jason_seashells_initial_count_l185_185974


namespace Rose_has_20_crystal_beads_l185_185584

noncomputable def num_crystal_beads (metal_beads_Nancy : ℕ) (pearl_beads_more_than_metal : ℕ) (beads_per_bracelet : ℕ)
    (total_bracelets : ℕ) (stone_to_crystal_ratio : ℕ) : ℕ :=
  let pearl_beads_Nancy := metal_beads_Nancy + pearl_beads_more_than_metal
  let total_beads_Nancy := metal_beads_Nancy + pearl_beads_Nancy
  let beads_needed := beads_per_bracelet * total_bracelets
  let beads_Rose := beads_needed - total_beads_Nancy
  beads_Rose / stone_to_crystal_ratio.succ

theorem Rose_has_20_crystal_beads :
  num_crystal_beads 40 20 8 20 2 = 20 :=
by
  sorry

end Rose_has_20_crystal_beads_l185_185584


namespace odometer_problem_l185_185172

theorem odometer_problem :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a + b + c ≤ 10 ∧ (11 * c - 10 * a - b) % 6 = 0 ∧ a^2 + b^2 + c^2 = 54 :=
by
  sorry

end odometer_problem_l185_185172


namespace bananas_count_l185_185924

/-- Elias bought some bananas and ate 1 of them. 
    After eating, he has 11 bananas left.
    Prove that Elias originally bought 12 bananas. -/
theorem bananas_count (x : ℕ) (h1 : x - 1 = 11) : x = 12 := by
  sorry

end bananas_count_l185_185924


namespace tenth_term_of_arithmetic_sequence_l185_185058

theorem tenth_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 2 * d = 14)
  (h2 : a + 5 * d = 32) : 
  (a + 9 * d = 56) ∧ (d = 6) := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l185_185058


namespace multiple_of_669_l185_185330

theorem multiple_of_669 (k : ℕ) (h : ∃ a : ℤ, 2007 ∣ (a + k : ℤ)^3 - a^3) : 669 ∣ k :=
sorry

end multiple_of_669_l185_185330


namespace profit_percentage_l185_185151

-- Define the selling price
def selling_price : ℝ := 900

-- Define the profit
def profit : ℝ := 100

-- Define the cost price as selling price minus profit
def cost_price : ℝ := selling_price - profit

-- Statement of the profit percentage calculation
theorem profit_percentage : (profit / cost_price) * 100 = 12.5 := by
  sorry

end profit_percentage_l185_185151


namespace focus_of_parabola_l185_185787

theorem focus_of_parabola : 
  ∀ x y : ℝ, y = -2 * x ^ 2 → ∃ focus: ℝ × ℝ, focus = (0, -1/8) :=
by
  intro x y h
  -- added temporarily prove
  use (0, -1/8)
  sorry

end focus_of_parabola_l185_185787


namespace find_m_l185_185984

theorem find_m (x1 x2 m : ℝ)
  (h1 : ∀ x, x^2 - 4 * x + m = 0 → x = x1 ∨ x = x2)
  (h2 : x1 + x2 - x1 * x2 = 1) :
  m = 3 :=
sorry

end find_m_l185_185984


namespace find_unknown_number_l185_185510

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l185_185510


namespace Sues_necklace_total_beads_l185_185460

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l185_185460


namespace number_of_boys_in_second_class_l185_185257

def boys_in_first_class : ℕ := 28
def portion_of_second_class (b2 : ℕ) : ℚ := 7 / 8 * b2

theorem number_of_boys_in_second_class (b2 : ℕ) (h : portion_of_second_class b2 = boys_in_first_class) : b2 = 32 :=
by 
  sorry

end number_of_boys_in_second_class_l185_185257


namespace matrix_pow_101_l185_185834

noncomputable def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ]

theorem matrix_pow_101 :
  matrixA ^ 101 =
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] :=
sorry

end matrix_pow_101_l185_185834


namespace polynomial_value_l185_185551

theorem polynomial_value
  (x : ℝ)
  (h : x^2 + 2 * x - 2 = 0) :
  4 - 2 * x - x^2 = 2 :=
by
  sorry

end polynomial_value_l185_185551


namespace frankie_candies_l185_185187

theorem frankie_candies (M D F : ℕ) (h1 : M = 92) (h2 : D = 18) (h3 : F = M - D) : F = 74 :=
by
  sorry

end frankie_candies_l185_185187


namespace verify_grazing_non_overlap_verify_chain_length_percentage_l185_185937

noncomputable def grazing_non_overlap (R : ℝ) : Prop :=
  let A := 0
  let B := 2 * Real.pi / 3
  let C := 4 * Real.pi / 3
  let ρ := R / 2
  ∀ θ ∈ {A, B, C}, 
    ∀ φ ∈ {A, B, C}, 
    θ ≠ φ → ∥θ - φ∥ ≥ ρ

noncomputable def chain_length_percentage (R : ℝ) : ℝ :=
  let ρ := 0.775 * R
  ρ

theorem verify_grazing_non_overlap (R : ℝ) : grazing_non_overlap R :=
begin
  sorry,
end

theorem verify_chain_length_percentage (R : ℝ) : chain_length_percentage R = 0.775 * R :=
begin
  refl,
end

end verify_grazing_non_overlap_verify_chain_length_percentage_l185_185937


namespace haleigh_cats_l185_185351

open Nat

def total_pairs := 14
def dog_leggings := 4
def legging_per_animal := 1

theorem haleigh_cats : ∀ (dogs cats : ℕ), 
  dogs = 4 → 
  total_pairs = dogs * legging_per_animal + cats * legging_per_animal → 
  cats = 10 :=
by
  intros dogs cats h1 h2
  sorry

end haleigh_cats_l185_185351


namespace donny_cost_of_apples_l185_185171

def cost_of_apples (small_cost medium_cost big_cost : ℝ) (n_small n_medium n_big : ℕ) : ℝ := 
  n_small * small_cost + n_medium * medium_cost + n_big * big_cost

theorem donny_cost_of_apples :
  cost_of_apples 1.5 2 3 6 6 8 = 45 :=
by
  sorry

end donny_cost_of_apples_l185_185171


namespace tan_alpha_sub_pi_over_8_l185_185559

theorem tan_alpha_sub_pi_over_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (Real.pi / 8)) :
  Real.tan (α - Real.pi / 8) = (5 * Real.sqrt 2 + 1) / 49 :=
by sorry

end tan_alpha_sub_pi_over_8_l185_185559


namespace subtraction_problem_l185_185408

variable (x : ℕ) -- Let's assume x is a natural number for this problem

theorem subtraction_problem (h : x - 46 = 15) : x - 29 = 32 := 
by 
  sorry -- Proof to be filled in

end subtraction_problem_l185_185408


namespace probability_digits_all_different_l185_185652

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l185_185652


namespace probability_f_times_f_zero_l185_185203

-- Define the function f
def f (x : ℕ) : ℝ := Real.sin (Real.pi * x / 6)

-- Define the set M
def M := {0, 1, 2, 3, 4, 5, 6, 7, 8}

-- Define the condition where f(m) * f(n) = 0
def f_times_f_zero (m n : ℕ) : Prop := f m * f n = 0

-- Define the total pairs
def total_pairs : Finset (ℕ × ℕ) := Finset.filter (λ p, p.1 ≠ p.2) (Finset.product M M)

-- Define the pairs where f(m) * f(n) = 0
def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p, f_times_f_zero p.1 p.2) total_pairs

-- Define the probability
noncomputable def probability : ℝ :=
  (Finset.card valid_pairs : ℝ) / (Finset.card total_pairs : ℝ)

-- Assert the probability is 5/12
theorem probability_f_times_f_zero :
  probability = 5 / 12 :=
sorry

end probability_f_times_f_zero_l185_185203


namespace abs_neg_frac_l185_185098

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l185_185098


namespace find_number_l185_185523

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l185_185523


namespace parabola_translation_l185_185280

theorem parabola_translation :
  (∀ x : ℝ, y = x^2 → y' = (x - 1)^2 + 3) :=
sorry

end parabola_translation_l185_185280


namespace second_customer_headphones_l185_185882

theorem second_customer_headphones
  (H : ℕ)
  (M : ℕ)
  (x : ℕ)
  (H_eq : H = 30)
  (eq1 : 5 * M + 8 * H = 840)
  (eq2 : 3 * M + x * H = 480) :
  x = 4 :=
by
  sorry

end second_customer_headphones_l185_185882


namespace circle_equation_l185_185030

theorem circle_equation (C : ℝ → ℝ → Prop)
  (h₁ : C 1 0)
  (h₂ : C 0 (Real.sqrt 3))
  (h₃ : C (-3) 0) :
  ∃ D E F : ℝ, (∀ x y, C x y ↔ x^2 + y^2 + D * x + E * y + F = 0) ∧ D = 2 ∧ E = 0 ∧ F = -3 := 
by
  sorry

end circle_equation_l185_185030


namespace birth_rate_calculation_l185_185105

theorem birth_rate_calculation (D : ℕ) (G : ℕ) (P : ℕ) (NetGrowth : ℕ) (B : ℕ) (h1 : D = 16) (h2 : G = 12) (h3 : P = 3000) (h4 : NetGrowth = G * P / 100) (h5 : NetGrowth = B - D) : B = 52 := by
  sorry

end birth_rate_calculation_l185_185105


namespace average_income_P_Q_l185_185998

   variable (P Q R : ℝ)

   theorem average_income_P_Q
     (h1 : (Q + R) / 2 = 6250)
     (h2 : (P + R) / 2 = 5200)
     (h3 : P = 4000) :
     (P + Q) / 2 = 5050 := by
   sorry
   
end average_income_P_Q_l185_185998


namespace tan_ratio_l185_185387

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end tan_ratio_l185_185387


namespace no_nonneg_rational_sol_for_equation_l185_185469

theorem no_nonneg_rational_sol_for_equation :
  ¬ ∃ (x y z : ℚ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^5 + 2 * y^5 + 5 * z^5 = 11 :=
by
  sorry

end no_nonneg_rational_sol_for_equation_l185_185469


namespace smallest_prime_after_seven_non_primes_l185_185971

-- Define the property of being non-prime
def non_prime (n : ℕ) : Prop :=
¬Nat.Prime n

-- Statement of the proof problem
theorem smallest_prime_after_seven_non_primes :
  ∃ m : ℕ, (∀ i : ℕ, (m - 7 ≤ i ∧ i < m) → non_prime i) ∧ Nat.Prime m ∧
  (∀ p : ℕ, (∀ i : ℕ, (p - 7 ≤ i ∧ i < p) → non_prime i) → Nat.Prime p → m ≤ p) :=
sorry

end smallest_prime_after_seven_non_primes_l185_185971


namespace find_number_l185_185507

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l185_185507


namespace ratio_calculation_l185_185043

theorem ratio_calculation (A B C : ℚ)
  (h_ratio : (A / B = 3 / 2) ∧ (B / C = 2 / 5)) :
  (4 * A + 3 * B) / (5 * C - 2 * B) = 15 / 23 := by
  sorry

end ratio_calculation_l185_185043


namespace friends_with_john_l185_185976

def total_slices (pizzas slices_per_pizza : Nat) : Nat := pizzas * slices_per_pizza

def total_people (total_slices slices_per_person : Nat) : Nat := total_slices / slices_per_person

def number_of_friends (total_people john : Nat) : Nat := total_people - john

theorem friends_with_john (pizzas slices_per_pizza slices_per_person john friends : Nat) (h_pizzas : pizzas = 3) 
                          (h_slices_per_pizza : slices_per_pizza = 8) (h_slices_per_person : slices_per_person = 4)
                          (h_john : john = 1) (h_friends : friends = 5) :
  number_of_friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) john = friends := by
  sorry

end friends_with_john_l185_185976


namespace no_positive_abc_exists_l185_185495

theorem no_positive_abc_exists 
  (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h1 : b^2 ≥ 4 * a * c)
  (h2 : c^2 ≥ 4 * b * a)
  (h3 : a^2 ≥ 4 * b * c)
  : false :=
sorry

end no_positive_abc_exists_l185_185495


namespace tens_digit_of_11_pow_12_pow_13_l185_185333

theorem tens_digit_of_11_pow_12_pow_13 :
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  tens_digit = 2 :=
by 
  let n := 12^13
  let t := 10
  let tens_digit := (11^n % 100) / 10 % 10
  show tens_digit = 2
  sorry

end tens_digit_of_11_pow_12_pow_13_l185_185333


namespace adults_not_wearing_blue_is_10_l185_185901

section JohnsonFamilyReunion

-- Define the number of children
def children : ℕ := 45

-- Define the ratio between adults and children
def adults : ℕ := children / 3

-- Define the ratio of adults who wore blue
def adults_wearing_blue : ℕ := adults / 3

-- Define the number of adults who did not wear blue
def adults_not_wearing_blue : ℕ := adults - adults_wearing_blue

-- Theorem stating the number of adults who did not wear blue
theorem adults_not_wearing_blue_is_10 : adults_not_wearing_blue = 10 :=
by
  -- This is a placeholder for the actual proof
  sorry

end JohnsonFamilyReunion

end adults_not_wearing_blue_is_10_l185_185901


namespace volume_of_prism_l185_185423

theorem volume_of_prism :
  ∃ (a b c : ℝ), ab * bc * ac = 762 ∧ (ab = 56) ∧ (bc = 63) ∧ (ac = 72) ∧ (b = 2 * a) :=
sorry

end volume_of_prism_l185_185423


namespace price_of_pen_l185_185287

theorem price_of_pen (price_pen : ℚ) (price_notebook : ℚ) :
  (price_pen + 3 * price_notebook = 36.45) →
  (price_notebook = 15 / 4 * price_pen) →
  price_pen = 3 :=
by
  intros h1 h2
  sorry

end price_of_pen_l185_185287


namespace min_sum_of_abc_conditions_l185_185394

theorem min_sum_of_abc_conditions
  (a b c d : ℕ)
  (hab : a + b = 2)
  (hac : a + c = 3)
  (had : a + d = 4)
  (hbc : b + c = 5)
  (hbd : b + d = 6)
  (hcd : c + d = 7) :
  a + b + c + d = 9 :=
sorry

end min_sum_of_abc_conditions_l185_185394


namespace loss_percentage_on_first_book_l185_185353

variable (C1 C2 SP L : ℝ)
variable (total_cost : ℝ := 540)
variable (C1_value : ℝ := 315)
variable (gain_percentage : ℝ := 0.19)
variable (common_selling_price : ℝ := 267.75)

theorem loss_percentage_on_first_book :
  C1 = C1_value →
  C2 = total_cost - C1 →
  SP = 1.19 * C2 →
  SP = C1 - (L / 100 * C1) →
  L = 15 :=
sorry

end loss_percentage_on_first_book_l185_185353


namespace length_of_field_l185_185103

-- Define the known conditions
def width := 50
def total_distance_run := 1800
def num_laps := 6

-- Define the problem statement
theorem length_of_field :
  ∃ L : ℕ, 6 * (2 * (L + width)) = total_distance_run ∧ L = 100 :=
by
  sorry

end length_of_field_l185_185103


namespace trivia_game_answer_l185_185616

theorem trivia_game_answer (correct_first_half : Nat)
    (points_per_question : Nat) (final_score : Nat) : 
    correct_first_half = 8 → 
    points_per_question = 8 →
    final_score = 80 →
    (final_score - correct_first_half * points_per_question) / points_per_question = 2 :=
by
    intros h1 h2 h3
    sorry

end trivia_game_answer_l185_185616


namespace find_number_l185_185509

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l185_185509


namespace average_speed_of_horse_l185_185762

/-- Definitions of the conditions given in the problem. --/
def pony_speed : ℕ := 20
def pony_head_start_hours : ℕ := 3
def horse_chase_hours : ℕ := 4

-- Define a proof problem for the average speed of the horse.
theorem average_speed_of_horse : (pony_head_start_hours * pony_speed + horse_chase_hours * pony_speed) / horse_chase_hours = 35 := by
  -- Setting up the necessary distances
  let pony_head_start_distance := pony_head_start_hours * pony_speed
  let pony_additional_distance := horse_chase_hours * pony_speed
  let total_pony_distance := pony_head_start_distance + pony_additional_distance
  -- Asserting the average speed of the horse
  let horse_average_speed := total_pony_distance / horse_chase_hours
  show horse_average_speed = 35
  sorry

end average_speed_of_horse_l185_185762


namespace problem1_problem2_l185_185987

noncomputable def f (x a : ℝ) := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x (-1) ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 :=
sorry

theorem problem2 (a : ℝ) : (∀ x, f x a ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end problem1_problem2_l185_185987


namespace fifteen_percent_of_x_is_ninety_l185_185519

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l185_185519


namespace solve_fractional_equation_1_solve_fractional_equation_2_l185_185270

-- Proof Problem 1
theorem solve_fractional_equation_1 (x : ℝ) (h : 6 * x - 2 ≠ 0) :
  (3 / 2 - 1 / (3 * x - 1) = 5 / (6 * x - 2)) ↔ (x = 10 / 9) :=
sorry

-- Proof Problem 2
theorem solve_fractional_equation_2 (x : ℝ) (h1 : 3 * x - 6 ≠ 0) :
  (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 → false :=
sorry

end solve_fractional_equation_1_solve_fractional_equation_2_l185_185270


namespace peanuts_remaining_l185_185421

theorem peanuts_remaining (initial : ℕ) (brock_fraction : ℚ) (bonita_count : ℕ) (h_initial : initial = 148) (h_brock_fraction : brock_fraction = 1/4) (h_bonita_count : bonita_count = 29) : initial - (initial * brock_fraction).natValue - bonita_count = 82 := 
by 
  sorry

end peanuts_remaining_l185_185421


namespace count_distinct_even_numbers_l185_185938

theorem count_distinct_even_numbers : 
  ∃ c, c = 37 ∧ ∀ d1 d2 d3, d1 ≠ d2 → d2 ≠ d3 → d1 ≠ d3 → (d1 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d2 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d3 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (∃ n : ℕ, n / 10 ^ 2 = d1 ∧ (n / 10) % 10 = d2 ∧ n % 10 = d3 ∧ n % 2 = 0) :=
sorry

end count_distinct_even_numbers_l185_185938


namespace sufficient_to_know_unitary_sum_ineq_deg_bound_l185_185241

noncomputable theory

namespace RealPolyRoots

-- Define the set E
def is_in_E (P : Polynomial ℝ) : Prop :=
  (∀ x : ℝ, Polynomial.eval x P = 0 → P.coeffs.all (λ c, c ∈ {-1, 0, 1})) ∧
  (∀ x : ℝ, Polynomial.eval x P = 0 → ∃ a : ℝ, x = a)

-- Question 1: Sufficient to know unitary polynomials in E that do not vanish at 0.
theorem sufficient_to_know_unitary (P : Polynomial ℝ) (hPE : is_in_E P) :
  ∀ (Q : Polynomial ℝ), Q.Monic ∧ Q ≠ 0 → is_in_E Q :=
sorry

-- Question 2: Proving the inequality
theorem sum_ineq (n : ℕ) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ i j, a i / a j) ≥ n ^ 2 :=
sorry

-- Question 3: Bounding the degree of polynomials in E to 3
theorem deg_bound (P : Polynomial ℝ) (hPE : is_in_E P) (hMonic : P.Monic) (hNonZero : P.eval 0 ≠ 0) :
  P.degree ≤ 3 :=
sorry

end RealPolyRoots

end sufficient_to_know_unitary_sum_ineq_deg_bound_l185_185241


namespace part_a_l185_185289

theorem part_a (n : ℕ) (h_n : n ≥ 3) (x : Fin n → ℝ) (hx : ∀ i j : Fin n, i ≠ j → x i ≠ x j) (hx_pos : ∀ i : Fin n, 0 < x i) :
  ∃ (i j : Fin n), i ≠ j ∧ 0 < (x i - x j) / (1 + (x i) * (x j)) ∧ (x i - x j) / (1 + (x i) * (x j)) < Real.tan (π / (2 * (n - 1))) :=
by
  sorry

end part_a_l185_185289


namespace leaf_raking_earnings_l185_185483

variable {S M L P : ℕ}

theorem leaf_raking_earnings (h1 : 5 * 4 + 7 * 2 + 10 * 1 + 3 * 1 = 47)
                             (h2 : 5 * 2 + 3 * 1 + 7 * 1 + 10 * 2 = 40)
                             (h3 : 163 - 87 = 76) :
  5 * S + 7 * M + 10 * L + 3 * P = 76 :=
by
  sorry

end leaf_raking_earnings_l185_185483


namespace water_added_to_mixture_is_11_l185_185821

noncomputable def initial_mixture_volume : ℕ := 45
noncomputable def initial_milk_ratio : ℚ := 4
noncomputable def initial_water_ratio : ℚ := 1
noncomputable def final_milk_ratio : ℚ := 9
noncomputable def final_water_ratio : ℚ := 5

theorem water_added_to_mixture_is_11 :
  ∃ x : ℚ, (initial_milk_ratio * initial_mixture_volume / 
            (initial_water_ratio * initial_mixture_volume + x)) = (final_milk_ratio / final_water_ratio)
  ∧ x = 11 :=
by
  -- Proof here
  sorry

end water_added_to_mixture_is_11_l185_185821


namespace total_ticket_count_is_59_l185_185608

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end total_ticket_count_is_59_l185_185608


namespace age_difference_l185_185868

variable {A B C : ℕ}

-- Definition of conditions
def condition1 (A B C : ℕ) : Prop := A + B > B + C
def condition2 (A C : ℕ) : Prop := C = A - 16

-- The theorem stating the math problem
theorem age_difference (h1 : condition1 A B C) (h2 : condition2 A C) :
  (A + B) - (B + C) = 16 := by
  sorry

end age_difference_l185_185868


namespace lcm_1_to_5_l185_185430

theorem lcm_1_to_5 : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5 = 60 := by
  sorry

end lcm_1_to_5_l185_185430


namespace sum_a_b_is_95_l185_185566

-- Define the conditions
def product_condition (a b : ℕ) : Prop :=
  (a : ℤ) / 3 = 16 ∧ b = a - 1

-- Define the theorem to be proven
theorem sum_a_b_is_95 (a b : ℕ) (h : product_condition a b) : a + b = 95 :=
by
  sorry

end sum_a_b_is_95_l185_185566


namespace smallest_solution_x_abs_x_eq_3x_plus_2_l185_185528

theorem smallest_solution_x_abs_x_eq_3x_plus_2 : ∃ x : ℝ, (x * abs x = 3 * x + 2) ∧ (∀ y : ℝ, (y * abs y = 3 * y + 2) → x ≤ y) ∧ x = -2 :=
by
  sorry

end smallest_solution_x_abs_x_eq_3x_plus_2_l185_185528


namespace B_subset_A_implies_range_m_l185_185540

variable {x m : ℝ}

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -m < x ∧ x < m}

theorem B_subset_A_implies_range_m (m : ℝ) (h : B m ⊆ A) : m ≤ 1 := by
  sorry

end B_subset_A_implies_range_m_l185_185540


namespace sum_cubes_eq_power_l185_185611

/-- Given the conditions, prove that 1^3 + 2^3 + 3^3 + 4^3 = 10^2 -/
theorem sum_cubes_eq_power : 1 + 2 + 3 + 4 = 10 → 1^3 + 2^3 + 3^3 + 4^3 = 10^2 :=
by
  intro h
  sorry

end sum_cubes_eq_power_l185_185611


namespace intersection_sum_l185_185543

theorem intersection_sum (h j : ℝ → ℝ)
  (H1 : h 3 = 3 ∧ j 3 = 3)
  (H2 : h 6 = 9 ∧ j 6 = 9)
  (H3 : h 9 = 18 ∧ j 9 = 18)
  (H4 : h 12 = 18 ∧ j 12 = 18) :
  ∃ a b : ℕ, h (3 * a) = b ∧ 3 * j a = b ∧ (a + b = 33) :=
by {
  sorry
}

end intersection_sum_l185_185543


namespace find_k_l185_185565

theorem find_k (k : ℝ) :
  (∀ x, x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) →
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 :=
by
  sorry

end find_k_l185_185565


namespace choose_student_B_l185_185856

-- Define the scores for students A and B
def scores_A : List ℕ := [72, 85, 86, 90, 92]
def scores_B : List ℕ := [76, 83, 85, 87, 94]

-- Function to calculate the average of scores
def average (scores : List ℕ) : ℚ :=
  scores.sum / scores.length

-- Function to calculate the variance of scores
def variance (scores : List ℕ) : ℚ :=
  let mean := average scores
  (scores.map (λ x => (x - mean) * (x - mean))).sum / scores.length

-- Calculate the average scores for A and B
def avg_A : ℚ := average scores_A
def avg_B : ℚ := average scores_B

-- Calculate the variances for A and B
def var_A : ℚ := variance scores_A
def var_B : ℚ := variance scores_B

-- The theorem to be proved
theorem choose_student_B : var_B < var_A :=
  by sorry

end choose_student_B_l185_185856


namespace cube_root_neg_eight_l185_185104

theorem cube_root_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by {
  sorry
}

end cube_root_neg_eight_l185_185104


namespace quadratic_root_property_l185_185531

theorem quadratic_root_property (a x1 x2 : ℝ) 
  (h_eq : ∀ x, a * x^2 - (3 * a + 1) * x + 2 * (a + 1) = 0)
  (h_distinct : x1 ≠ x2)
  (h_relation : x1 - x1 * x2 + x2 = 1 - a) : a = -1 :=
sorry

end quadratic_root_property_l185_185531


namespace additional_seasons_is_one_l185_185710

-- Definitions for conditions
def episodes_per_season : Nat := 22
def episodes_last_season : Nat := episodes_per_season + 4
def episodes_in_9_seasons : Nat := 9 * episodes_per_season
def hours_per_episode : Nat := 1 / 2 -- Stored as half units

-- Given conditions
def total_hours_to_watch_after_last_season: Nat := 112 * 2 -- converted to half-hours
def time_watched_in_9_seasons: Nat := episodes_in_9_seasons * hours_per_episode
def additional_hours: Nat := total_hours_to_watch_after_last_season - time_watched_in_9_seasons

-- Theorem to prove
theorem additional_seasons_is_one : additional_hours / hours_per_episode = episodes_last_season -> 
      additional_hours / hours_per_episode / episodes_per_season = 1 :=
by
  sorry

end additional_seasons_is_one_l185_185710


namespace zeke_estimate_smaller_l185_185425

variable (x y k : ℝ)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)
variable (h_inequality : x > 2 * y)
variable (hk_pos : 0 < k)

theorem zeke_estimate_smaller : (x + k) - 2 * (y + k) < x - 2 * y :=
by
  sorry

end zeke_estimate_smaller_l185_185425


namespace triangle_BDC_is_isosceles_l185_185719

-- Define the given conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (AB AC BC AD DC : ℝ)
variables (a : ℝ)
variables (α : ℝ)

-- Given conditions
def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) : Prop :=
AB = AC

def angle_BAC_120 (α : ℝ) : Prop :=
α = 120

def point_D_extension (AD AB : ℝ) : Prop :=
AD = 2 * AB

-- Let triangle ABC be isosceles with AB = AC and angle BAC = 120 degrees
axiom isosceles_triangle_ABC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB AC : ℝ) : is_isosceles_triangle A B C AB AC

axiom angle_BAC (α : ℝ) : angle_BAC_120 α

axiom point_D (AD AB : ℝ) : point_D_extension AD AB

-- Prove that triangle BDC is isosceles
theorem triangle_BDC_is_isosceles 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB AC BC AD DC : ℝ) 
  (α : ℝ) 
  (h1 : is_isosceles_triangle A B C AB AC)
  (h2 : angle_BAC_120 α)
  (h3 : point_D_extension AD AB) :
  BC = DC :=
sorry

end triangle_BDC_is_isosceles_l185_185719


namespace average_waiting_time_for_first_bite_l185_185478

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l185_185478


namespace expand_product_l185_185328

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 :=
by
  sorry

end expand_product_l185_185328


namespace tan_alpha_value_l185_185672

theorem tan_alpha_value (α β : ℝ) (h₁ : Real.tan (α + β) = 3) (h₂ : Real.tan β = 2) : 
  Real.tan α = 1 / 7 := 
by 
  sorry

end tan_alpha_value_l185_185672


namespace ratio_of_areas_l185_185129

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l185_185129


namespace jason_initial_cards_l185_185381

-- Definitions based on conditions
def cards_given_away : ℕ := 4
def cards_left : ℕ := 5

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 9 :=
by sorry

end jason_initial_cards_l185_185381


namespace range_of_b_l185_185049

theorem range_of_b (b : ℤ) : 
  (∃ x1 x2 : ℤ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ x1 - b > 0 ∧ x2 - b > 0 ∧ (∀ x : ℤ, x < 0 ∧ x - b > 0 → (x = x1 ∨ x = x2))) ↔ (-3 ≤ b ∧ b < -2) :=
by sorry

end range_of_b_l185_185049


namespace find_b_l185_185216

variable (p q r b : ℤ)

-- Conditions
def condition1 : Prop := p - q = 2
def condition2 : Prop := p - r = 1

-- The main statement to prove
def problem_statement : Prop :=
  b = (r - q) * ((p - q)^2 + (p - q) * (p - r) + (p - r)^2) → b = 7

theorem find_b (h1 : condition1 p q) (h2 : condition2 p r) (h3 : problem_statement p q r b) : b = 7 :=
sorry

end find_b_l185_185216


namespace sum_of_ages_is_37_l185_185581

def maries_age : ℕ := 12
def marcos_age (M : ℕ) : ℕ := 2 * M + 1

theorem sum_of_ages_is_37 : maries_age + marcos_age maries_age = 37 := 
by
  -- Inserting the proof details
  sorry

end sum_of_ages_is_37_l185_185581


namespace rod_length_l185_185816

theorem rod_length (L : ℝ) (weight : ℝ → ℝ) (weight_6m : weight 6 = 14.04) (weight_L : weight L = 23.4) :
  L = 10 :=
by 
  sorry

end rod_length_l185_185816


namespace sufficient_but_not_necessary_condition_l185_185189

-- The conditions of the problem
variables (a b : ℝ)

-- The proposition to be proved
theorem sufficient_but_not_necessary_condition (h : a + b = 1) : 4 * a * b ≤ 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l185_185189


namespace triangle_side_length_uniqueness_l185_185824

-- Define the conditions as axioms
variable (n : ℕ)
variable (h : n > 0)
variable (A1 : 3 * n + 9 > 5 * n - 4)
variable (A2 : 5 * n - 4 > 4 * n + 6)

-- The theorem stating the constraints and expected result
theorem triangle_side_length_uniqueness :
  (4 * n + 6) + (3 * n + 9) > (5 * n - 4) ∧
  (3 * n + 9) + (5 * n - 4) > (4 * n + 6) ∧
  (5 * n - 4) + (4 * n + 6) > (3 * n + 9) ∧
  3 * n + 9 > 5 * n - 4 ∧
  5 * n - 4 > 4 * n + 6 → 
  n = 11 :=
by {
  -- Proof steps can be filled here
  sorry
}

end triangle_side_length_uniqueness_l185_185824


namespace part_I_part_II_l185_185547

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

theorem part_I (x : ℝ) : f x > 4 ↔ x < -1.5 ∨ x > 2.5 := 
sorry

theorem part_II (a : ℝ) : (∀ x, f x ≥ a) ↔ a ≤ 3 := 
sorry

end part_I_part_II_l185_185547


namespace largest_of_five_consecutive_sum_180_l185_185305

theorem largest_of_five_consecutive_sum_180 (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 180) :
  n + 4 = 38 :=
by
  sorry

end largest_of_five_consecutive_sum_180_l185_185305


namespace b_should_pay_348_48_l185_185618

/-- Definitions for the given conditions --/

def horses_a : ℕ := 12
def months_a : ℕ := 8

def horses_b : ℕ := 16
def months_b : ℕ := 9

def horses_c : ℕ := 18
def months_c : ℕ := 6

def total_rent : ℕ := 841

/-- Calculate the individual and total contributions in horse-months --/

def contribution_a : ℕ := horses_a * months_a
def contribution_b : ℕ := horses_b * months_b
def contribution_c : ℕ := horses_c * months_c

def total_contributions : ℕ := contribution_a + contribution_b + contribution_c

/-- Calculate cost per horse-month and b's share of the rent --/

def cost_per_horse_month : ℚ := total_rent / total_contributions
def b_share : ℚ := contribution_b * cost_per_horse_month

/-- Lean statement to check b's share --/

theorem b_should_pay_348_48 : b_share = 348.48 := by
  sorry

end b_should_pay_348_48_l185_185618


namespace arithmetic_sequence_common_difference_l185_185544

theorem arithmetic_sequence_common_difference 
  (a1 a2 a3 a4 d : ℕ)
  (S : ℕ → ℕ)
  (h1 : S 2 = a1 + a2)
  (h2 : S 4 = a1 + a2 + a3 + a4)
  (h3 : S 2 = 4)
  (h4 : S 4 = 20)
  (h5 : a2 = a1 + d)
  (h6 : a3 = a2 + d)
  (h7 : a4 = a3 + d) :
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l185_185544


namespace cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l185_185327

theorem cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2 : 
  Real.cos (- (11 / 4) * Real.pi) = - Real.sqrt 2 / 2 := 
sorry

end cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l185_185327


namespace fifteen_percent_of_x_is_ninety_l185_185504

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l185_185504


namespace number_of_integers_with_square_fraction_l185_185027

theorem number_of_integers_with_square_fraction : 
  ∃! (S : Finset ℤ), (∀ (n : ℤ), n ∈ S ↔ ∃ (k : ℤ), (n = 15 * k^2) ∨ (15 - n = k^2)) ∧ S.card = 2 := 
sorry

end number_of_integers_with_square_fraction_l185_185027


namespace female_salmon_returned_l185_185015

theorem female_salmon_returned :
  let total_salmon : ℕ := 971639
  let male_salmon : ℕ := 712261
  total_salmon - male_salmon = 259378 :=
by
  let total_salmon := 971639
  let male_salmon := 712261
  calc
    971639 - 712261 = 259378 := by norm_num

end female_salmon_returned_l185_185015


namespace find_b_l185_185356

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end find_b_l185_185356


namespace quadratic_no_real_roots_l185_185953

theorem quadratic_no_real_roots (b c : ℝ) (h : ∀ x : ℝ, x^2 + b * x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 :=
sorry

end quadratic_no_real_roots_l185_185953


namespace standard_polar_representation_l185_185228

theorem standard_polar_representation {r θ : ℝ} (hr : r < 0) (hθ : θ = 5 * Real.pi / 6) :
  ∃ (r' θ' : ℝ), r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * Real.pi ∧ (r', θ') = (5, 11 * Real.pi / 6) := 
by {
  sorry
}

end standard_polar_representation_l185_185228


namespace one_less_than_neg_one_is_neg_two_l185_185447

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end one_less_than_neg_one_is_neg_two_l185_185447


namespace tournament_player_count_l185_185570

theorem tournament_player_count (n : ℕ) :
  (∃ points_per_game : ℕ, points_per_game = (n * (n - 1)) / 2) →
  (∃ T : ℕ, T = 90) →
  (n * (n - 1)) / 4 = 90 →
  n = 19 :=
by
  intros h1 h2 h3
  sorry

end tournament_player_count_l185_185570


namespace meaningful_sqrt_neg_x_squared_l185_185533

theorem meaningful_sqrt_neg_x_squared (x : ℝ) : (x = 0) ↔ (-(x^2) ≥ 0) :=
by
  sorry

end meaningful_sqrt_neg_x_squared_l185_185533


namespace range_of_a_l185_185036

variable (a : ℝ)

def p := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q := ∃ x : ℝ, x^2 + (a-1)*x + 1 < 0
def r := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a
  (h₀ : p a ∨ q a)
  (h₁ : ¬ (p a ∧ q a)) :
  r a :=
sorry

end range_of_a_l185_185036


namespace missed_both_shots_l185_185774

variables (p q : Prop)

theorem missed_both_shots : (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
by sorry

end missed_both_shots_l185_185774


namespace sum_of_squares_l185_185358

open Int

theorem sum_of_squares (p q r s t u : ℤ) (h : ∀ x : ℤ, 343 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3506 :=
sorry

end sum_of_squares_l185_185358


namespace fraction_simplification_l185_185267

def numerator : Int := 5^4 + 5^2 + 5
def denominator : Int := 5^3 - 2 * 5

theorem fraction_simplification :
  (numerator : ℚ) / (denominator : ℚ) = 27 + (14 / 23) := by
  sorry

end fraction_simplification_l185_185267


namespace find_abc_l185_185294

theorem find_abc (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < 4)
  (h4 : a + b + c = a * b * c) : (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                 (a = -3 ∧ b = -2 ∧ c = -1) ∨ 
                                 (a = -1 ∧ b = 0 ∧ c = 1) ∨ 
                                 (a = -2 ∧ b = 0 ∧ c = 2) ∨ 
                                 (a = -3 ∧ b = 0 ∧ c = 3) :=
sorry

end find_abc_l185_185294


namespace abs_neg_three_halves_l185_185091

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l185_185091


namespace max_expression_value_l185_185108

theorem max_expression_value : 
  ∃ a b c d e f : ℕ, 1 ≤ a ∧ a ≤ 6 ∧
                   1 ≤ b ∧ b ≤ 6 ∧
                   1 ≤ c ∧ c ≤ 6 ∧
                   1 ≤ d ∧ d ≤ 6 ∧
                   1 ≤ e ∧ e ≤ 6 ∧
                   1 ≤ f ∧ f ≤ 6 ∧
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                   d ≠ e ∧ d ≠ f ∧
                   e ≠ f ∧
                   (f * (a * d + b * c) / (b * d * e) = 14) :=
sorry

end max_expression_value_l185_185108


namespace angle_hyperbola_l185_185349

theorem angle_hyperbola (a b : ℝ) (e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (hyperbola_eq : ∀ (x y : ℝ), ((x^2)/(a^2) - (y^2)/(b^2) = 1)) 
  (eccentricity_eq : e = 2 + Real.sqrt 6 - Real.sqrt 3 - Real.sqrt 2) :
  ∃ α : ℝ, α = 15 :=
by
  sorry

end angle_hyperbola_l185_185349


namespace least_area_of_figure_l185_185239

theorem least_area_of_figure (c : ℝ) (hc : c > 1) : 
  ∃ A : ℝ, A = (4 / 3) * (c - 1)^(3 / 2) :=
by
  sorry

end least_area_of_figure_l185_185239


namespace log_identity_l185_185357

theorem log_identity (c b : ℝ) (h1 : c = Real.log 81 / Real.log 4) (h2 : b = Real.log 3 / Real.log 2) : c = 2 * b := by
  sorry

end log_identity_l185_185357


namespace simplify_power_expression_l185_185404

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^5 = 243 * x^20 :=
by
  sorry

end simplify_power_expression_l185_185404


namespace sum_first_two_integers_l185_185332

/-- Prove that the sum of the first two integers n > 1 such that 3^n is divisible by n 
and 3^n - 1 is divisible by n - 1 is equal to 30. -/
theorem sum_first_two_integers (n : ℕ) (h1 : n > 1) (h2 : 3 ^ n % n = 0) (h3 : (3 ^ n - 1) % (n - 1) = 0) : 
  n = 3 ∨ n = 27 → n + 3 + 27 = 30 :=
sorry

end sum_first_two_integers_l185_185332


namespace probability_neither_red_nor_purple_l185_185288

theorem probability_neither_red_nor_purple (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) : 
  total_balls = 60 →
  white_balls = 22 →
  green_balls = 18 →
  yellow_balls = 2 →
  red_balls = 15 →
  purple_balls = 3 →
  (total_balls - red_balls - purple_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_neither_red_nor_purple_l185_185288


namespace olympic_iberic_sets_containing_33_l185_185455

/-- A set of positive integers is iberic if it is a subset of {2, 3, ..., 2018},
    and whenever m, n are both in the set, gcd(m, n) is also in the set. -/
def is_iberic_set (X : Set ℕ) : Prop :=
  X ⊆ {n | 2 ≤ n ∧ n ≤ 2018} ∧ ∀ m n, m ∈ X → n ∈ X → Nat.gcd m n ∈ X

/-- An iberic set is olympic if it is not properly contained in any other iberic set. -/
def is_olympic_set (X : Set ℕ) : Prop :=
  is_iberic_set X ∧ ∀ Y, is_iberic_set Y → X ⊂ Y → False

/-- The olympic iberic sets containing 33 are exactly {3, 6, 9, ..., 2016} and {11, 22, 33, ..., 2013}. -/
theorem olympic_iberic_sets_containing_33 :
  ∀ X, is_iberic_set X ∧ 33 ∈ X → X = {n | 3 ∣ n ∧ 2 ≤ n ∧ n ≤ 2016} ∨ X = {n | 11 ∣ n ∧ 11 ≤ n ∧ n ≤ 2013} :=
by
  sorry

end olympic_iberic_sets_containing_33_l185_185455


namespace smallest_n_congruent_l185_185429

theorem smallest_n_congruent (n : ℕ) (h : 635 * n ≡ 1251 * n [MOD 30]) : n = 15 :=
sorry

end smallest_n_congruent_l185_185429


namespace derivative_of_f_at_pi_over_2_l185_185694

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = -5 :=
sorry

end derivative_of_f_at_pi_over_2_l185_185694


namespace quadrilateral_correct_choice_l185_185615

/-- Define the triangle inequality theorem for four line segments.
    A quadrilateral can be formed if for any:
    - The sum of the lengths of any three segments is greater than the length of the fourth segment.
-/
def is_quadrilateral (a b c d : ℕ) : Prop :=
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a)

/-- Determine which set of three line segments can form a quadrilateral with a fourth line segment of length 5.
    We prove that the correct choice is the set (3, 3, 3). --/
theorem quadrilateral_correct_choice :
  is_quadrilateral 3 3 3 5 ∧  ¬ is_quadrilateral 1 1 1 5 ∧  ¬ is_quadrilateral 1 1 8 5 ∧  ¬ is_quadrilateral 1 2 2 5 :=
by
  sorry

end quadrilateral_correct_choice_l185_185615


namespace triangle_inequality_l185_185537

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) :=
sorry

end triangle_inequality_l185_185537


namespace arithmetic_sequence_general_term_and_sum_l185_185060

theorem arithmetic_sequence_general_term_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 2 = 2) →
  (a 4 = 4) →
  (∀ n, a n = n) →
  (∀ n, b n = 2 ^ (a n)) →
  (∀ n, S n = 2 * (2 ^ n - 1)) :=
by
  intros h1 h2 h3 h4
  -- Proof part is skipped
  sorry

end arithmetic_sequence_general_term_and_sum_l185_185060


namespace find_unknown_number_l185_185514

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l185_185514


namespace consecutive_odds_coprime_l185_185290

theorem consecutive_odds_coprime (a : ℤ) : Nat.coprime a (a + 2) :=
sorry

end consecutive_odds_coprime_l185_185290


namespace sue_necklace_total_beads_l185_185462

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l185_185462


namespace original_number_not_800_l185_185885

theorem original_number_not_800 (x : ℕ) (h : 10 * x = x + 720) : x ≠ 800 :=
by {
  sorry
}

end original_number_not_800_l185_185885


namespace solution_set_inequality_l185_185736

theorem solution_set_inequality (x : ℝ) : (0 < x ∧ x < 1) ↔ (1 / (x - 1) < -1) :=
by
  sorry

end solution_set_inequality_l185_185736


namespace no_integer_k_sq_plus_k_plus_one_divisible_by_101_l185_185401

theorem no_integer_k_sq_plus_k_plus_one_divisible_by_101 (k : ℤ) : 
  (k^2 + k + 1) % 101 ≠ 0 := 
by
  sorry

end no_integer_k_sq_plus_k_plus_one_divisible_by_101_l185_185401


namespace part_to_third_fraction_is_six_five_l185_185078

noncomputable def ratio_of_part_to_third_fraction (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ℝ :=
  P / (N / 3)

theorem part_to_third_fraction_is_six_five (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ratio_of_part_to_third_fraction P N h1 h2 = 6 / 5 :=
  sorry

end part_to_third_fraction_is_six_five_l185_185078


namespace ratio_of_areas_of_triangles_l185_185125

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l185_185125


namespace time_difference_l185_185437

-- Define the capacity of the tanks
def capacity : ℕ := 20

-- Define the inflow rates of tanks A and B in litres per hour
def inflow_rate_A : ℕ := 2
def inflow_rate_B : ℕ := 4

-- Define the times to fill tanks A and B
def time_A : ℕ := capacity / inflow_rate_A
def time_B : ℕ := capacity / inflow_rate_B

-- Proving the time difference between filling tanks A and B
theorem time_difference : (time_A - time_B) = 5 := by
  sorry

end time_difference_l185_185437


namespace count_four_digit_numbers_with_digit_sum_5_count_four_digit_numbers_with_digit_sum_6_count_four_digit_numbers_with_digit_sum_7_l185_185922

open Nat List

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def digit_sum (n : ℕ) : ℕ := (to_digits 10 n).sum

theorem count_four_digit_numbers_with_digit_sum_5 : 
  (finset.filter (λ n, digit_sum n = 5) (finset.filter is_four_digit (finset.range 10000))).card = 35 :=
sorry

theorem count_four_digit_numbers_with_digit_sum_6 : 
  (finset.filter (λ n, digit_sum n = 6) (finset.filter is_four_digit (finset.range 10000))).card = 56 :=
sorry

theorem count_four_digit_numbers_with_digit_sum_7 : 
  (finset.filter (λ n, digit_sum n = 7) (finset.filter is_four_digit (finset.range 10000))).card = 84 :=
sorry

end count_four_digit_numbers_with_digit_sum_5_count_four_digit_numbers_with_digit_sum_6_count_four_digit_numbers_with_digit_sum_7_l185_185922


namespace div_eq_four_l185_185361

theorem div_eq_four (x : ℝ) (h : 64 / x = 4) : x = 16 :=
sorry

end div_eq_four_l185_185361


namespace apple_tree_total_apples_l185_185893

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l185_185893


namespace number_of_terminating_decimals_l185_185335

theorem number_of_terminating_decimals :
  ∃ (count : ℕ), count = 64 ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 449 → (∃ k : ℕ, n = 7 * k) → (∃ k : ℕ, (∃ m : ℕ, 560 = 2^m * 5^k * n)) :=
sorry

end number_of_terminating_decimals_l185_185335


namespace ray_steps_problem_l185_185848

theorem ray_steps_problem : ∃ n, n > 15 ∧ n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧ n = 71 :=
by
  sorry

end ray_steps_problem_l185_185848


namespace x_greater_than_y_l185_185806

theorem x_greater_than_y (x y z : ℝ) (h1 : x + y + z = 28) (h2 : 2 * x - y = 32) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 
  x > y :=
by 
  sorry

end x_greater_than_y_l185_185806


namespace almost_square_as_quotient_l185_185870

-- Defining what almost squares are
def isAlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

-- Statement of the theorem
theorem almost_square_as_quotient (n : ℕ) (hn : n > 0) :
  ∃ a b : ℕ, isAlmostSquare a ∧ isAlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end almost_square_as_quotient_l185_185870


namespace ratio_of_areas_GHI_to_JKL_l185_185121

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l185_185121


namespace triangles_in_extended_figure_l185_185690

theorem triangles_in_extended_figure : 
  ∀ (row1_tri : ℕ) (row2_tri : ℕ) (row3_tri : ℕ) (row4_tri : ℕ) 
  (row1_2_med_tri : ℕ) (row2_3_med_tri : ℕ) (row3_4_med_tri : ℕ) 
  (large_tri : ℕ), 
  row1_tri = 6 →
  row2_tri = 5 →
  row3_tri = 4 →
  row4_tri = 3 →
  row1_2_med_tri = 5 →
  row2_3_med_tri = 2 →
  row3_4_med_tri = 1 →
  large_tri = 1 →
  row1_tri + row2_tri + row3_tri + row4_tri
  + row1_2_med_tri + row2_3_med_tri + row3_4_med_tri
  + large_tri = 27 :=
by
  intro row1_tri row2_tri row3_tri row4_tri
  intro row1_2_med_tri row2_3_med_tri row3_4_med_tri
  intro large_tri
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end triangles_in_extended_figure_l185_185690


namespace min_binary_questions_to_determine_number_l185_185256

theorem min_binary_questions_to_determine_number (x : ℕ) (h : 10 ≤ x ∧ x ≤ 19) : 
  ∃ (n : ℕ), n = 3 := 
sorry

end min_binary_questions_to_determine_number_l185_185256


namespace shiela_used_seven_colors_l185_185001

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ) 
    (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : 
    total_blocks / blocks_per_color = 7 :=
by
  sorry

end shiela_used_seven_colors_l185_185001


namespace sqrt_of_square_neg_five_eq_five_l185_185484

theorem sqrt_of_square_neg_five_eq_five :
  Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end sqrt_of_square_neg_five_eq_five_l185_185484


namespace population_30_3_million_is_30300000_l185_185964

theorem population_30_3_million_is_30300000 :
  let million := 1000000
  let population_1998 := 30.3 * million
  population_1998 = 30300000 :=
by
  -- Proof goes here
  sorry

end population_30_3_million_is_30300000_l185_185964


namespace calculation_result_l185_185907

theorem calculation_result : 
  (16 = 2^4) → 
  (8 = 2^3) → 
  (4 = 2^2) → 
  (16^6 * 8^3 / 4^10 = 8192) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end calculation_result_l185_185907


namespace gcf_lcm_problem_l185_185247

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_problem :
  GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end gcf_lcm_problem_l185_185247


namespace range_of_m_l185_185807

def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B (m : ℝ) : Set ℝ := {x | abs (x - 3) ≤ m}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) (m : ℝ) : Prop := x ∈ B m

theorem range_of_m (m : ℝ) (hm : m > 0):
  (∀ x, p x → q x m) ↔ (6 ≤ m) := by
  sorry

end range_of_m_l185_185807


namespace expression_independent_of_a_l185_185724

theorem expression_independent_of_a (a : ℝ) :
  7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 :=
by sorry

end expression_independent_of_a_l185_185724


namespace problem_l185_185340

-- Definitions for the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a (n : ℕ) : ℤ := sorry -- Define the arithmetic sequence a_n based on conditions

-- Problem statement
theorem problem : 
  (a 1 = 4) ∧
  (a 2 + a 4 = 4) →
  (∃ d : ℤ, arithmetic_sequence a d ∧ a 10 = -5) :=
by {
  sorry
}

end problem_l185_185340


namespace insurance_covers_80_percent_of_medical_bills_l185_185119

theorem insurance_covers_80_percent_of_medical_bills 
    (vaccine_cost : ℕ) (num_vaccines : ℕ) (doctor_visit_cost trip_cost : ℕ) (amount_tom_pays : ℕ) 
    (total_cost := num_vaccines * vaccine_cost + doctor_visit_cost) 
    (total_trip_cost := trip_cost + total_cost)
    (insurance_coverage := total_trip_cost - amount_tom_pays)
    (percent_covered := (insurance_coverage * 100) / total_cost) :
    vaccine_cost = 45 → num_vaccines = 10 → doctor_visit_cost = 250 → trip_cost = 1200 → amount_tom_pays = 1340 →
    percent_covered = 80 := 
by
  sorry

end insurance_covers_80_percent_of_medical_bills_l185_185119


namespace boys_in_class_l185_185075

theorem boys_in_class (total_students : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ)
    (h_ratio : ratio_girls = 3) (h_ratio_boys : ratio_boys = 4)
    (h_total_students : total_students = 35) :
    ∃ boys, boys = 20 :=
by
  let k := total_students / (ratio_girls + ratio_boys)
  have hk : k = 5 := by sorry
  let boys := ratio_boys * k
  have h_boys : boys = 20 := by sorry
  exact ⟨boys, h_boys⟩

end boys_in_class_l185_185075


namespace find_cos_alpha_l185_185959

variable (α β : ℝ)

-- Conditions
def acute_angles (α β : ℝ) : Prop := 0 < α ∧ α < (Real.pi / 2) ∧ 0 < β ∧ β < (Real.pi / 2)
def cos_alpha_beta : Prop := Real.cos (α + β) = 12 / 13
def cos_2alpha_beta : Prop := Real.cos (2 * α + β) = 3 / 5

-- Main theorem
theorem find_cos_alpha (h1 : acute_angles α β) (h2 : cos_alpha_beta α β) (h3 : cos_2alpha_beta α β) : 
  Real.cos α = 56 / 65 :=
sorry

end find_cos_alpha_l185_185959


namespace intersection_points_x_axis_vertex_on_line_inequality_c_l185_185706

section
variable {r : ℝ}
def quadratic_function (x m : ℝ) : ℝ := -0.5 * (x - 2*m)^2 + 3 - m

theorem intersection_points_x_axis (m : ℝ) (h : m = 2) : 
  ∃ x1 x2 : ℝ, quadratic_function x1 m = 0 ∧ quadratic_function x2 m = 0 ∧ x1 ≠ x2 :=
by
  sorry

theorem vertex_on_line (m : ℝ) (h : true) : 
  ∀ m : ℝ, (2*m, 3-m) ∈ {p : ℝ × ℝ | p.2 = -0.5 * p.1 + 3} :=
by
  sorry

theorem inequality_c (a c m : ℝ) (hP : quadratic_function (a+1) m = c) (hQ : quadratic_function ((4*m-5)+a) m = c) : 
  c ≤ 13/8 :=
by
  sorry
end

end intersection_points_x_axis_vertex_on_line_inequality_c_l185_185706


namespace at_least_one_hit_l185_185322

-- Introduce the predicates
variable (p q : Prop)

-- State the theorem
theorem at_least_one_hit : (¬ (¬ p ∧ ¬ q)) = (p ∨ q) :=
by
  sorry

end at_least_one_hit_l185_185322


namespace no_solution_inequalities_l185_185371

theorem no_solution_inequalities (a : ℝ) : (¬ ∃ x : ℝ, 2 * x - 4 > 0 ∧ x - a < 0) → a ≤ 2 := 
by 
  sorry

end no_solution_inequalities_l185_185371


namespace find_xyz_l185_185811

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 198) (h5 : y * (z + x) = 216) (h6 : z * (x + y) = 234) :
  x * y * z = 1080 :=
sorry

end find_xyz_l185_185811


namespace hotel_charge_l185_185411

variable (R G P : ℝ)

theorem hotel_charge (h1 : P = 0.60 * R) (h2 : P = 0.90 * G) : (R - G) / G = 0.50 :=
by
  sorry

end hotel_charge_l185_185411


namespace find_ec_l185_185949

theorem find_ec (angle_A : ℝ) (BC : ℝ) (BD_perp_AC : Prop) (CE_perp_AB : Prop)
  (angle_DBC_2_angle_ECB : Prop) :
  angle_A = 45 ∧ 
  BC = 8 ∧
  BD_perp_AC ∧
  CE_perp_AB ∧
  angle_DBC_2_angle_ECB → 
  ∃ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 2 ∧ a + b + c = 7 :=
sorry

end find_ec_l185_185949


namespace expression_evaluation_l185_185770

theorem expression_evaluation:
  ( (1/3)^2000 * 27^669 + Real.sin (60 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) + (2009 + Real.sin (25 * Real.pi / 180))^0 ) = 
  (2 + 29/54) := by
  sorry

end expression_evaluation_l185_185770


namespace inequality_proof_l185_185837

variables (a b c : ℝ)

theorem inequality_proof
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (cond : a^2 + b^2 + c^2 + ab + bc + ca ≤ 2) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 := 
sorry

end inequality_proof_l185_185837


namespace largest_digit_to_correct_sum_l185_185854

theorem largest_digit_to_correct_sum :
  (725 + 864 + 991 = 2570) → (∃ (d : ℕ), d = 9 ∧ 
  (∃ (n1 : ℕ), n1 ∈ [702, 710, 711, 721, 715] ∧ 
  ∃ (n2 : ℕ), n2 ∈ [806, 805, 814, 854, 864] ∧ 
  ∃ (n3 : ℕ), n3 ∈ [918, 921, 931, 941, 981, 991] ∧ 
  n1 + n2 + n3 = n1 + n2 + n3 - 10))
    → d = 9 :=
by
  sorry

end largest_digit_to_correct_sum_l185_185854


namespace ratio_problem_l185_185804

/-
  Given the ratio A : B : C = 3 : 2 : 5, we need to prove that 
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19.
-/

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 :=
by sorry

end ratio_problem_l185_185804


namespace problem1_problem2_problem3_problem4_l185_185147

theorem problem1 (h : Real.cos 75 * Real.sin 75 = 1 / 2) : False :=
by
  sorry

theorem problem2 : (1 + Real.tan 15) / (1 - Real.tan 15) = Real.sqrt 3 :=
by
  sorry

theorem problem3 : Real.tan 20 + Real.tan 25 + Real.tan 20 * Real.tan 25 = 1 :=
by
  sorry

theorem problem4 (θ : Real) (h1 : Real.sin (2 * θ) ≠ 0) : (1 / Real.tan θ - 1 / Real.tan (2 * θ) = 1 / Real.sin (2 * θ)) :=
by
  sorry

end problem1_problem2_problem3_problem4_l185_185147


namespace abs_neg_frac_l185_185095

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l185_185095


namespace optimal_garden_dimensions_l185_185409

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), l ≥ 100 ∧ w ≥ 60 ∧ l + w = 180 ∧ l * w = 8000 := by
  sorry

end optimal_garden_dimensions_l185_185409


namespace soy_sauce_bottle_size_l185_185996

theorem soy_sauce_bottle_size 
  (ounces_per_cup : ℕ)
  (cups_recipe1 : ℕ)
  (cups_recipe2 : ℕ)
  (cups_recipe3 : ℕ)
  (number_of_bottles : ℕ)
  (total_ounces_needed : ℕ)
  (ounces_per_bottle : ℕ) :
  ounces_per_cup = 8 →
  cups_recipe1 = 2 →
  cups_recipe2 = 1 →
  cups_recipe3 = 3 →
  number_of_bottles = 3 →
  total_ounces_needed = (cups_recipe1 + cups_recipe2 + cups_recipe3) * ounces_per_cup →
  ounces_per_bottle = total_ounces_needed / number_of_bottles →
  ounces_per_bottle = 16 :=
by
  sorry

end soy_sauce_bottle_size_l185_185996


namespace ratio_doubled_to_original_l185_185884

theorem ratio_doubled_to_original (x : ℝ) (h : 3 * (2 * x + 9) = 69) : (2 * x) / x = 2 :=
by
  -- We skip the proof here.
  sorry

end ratio_doubled_to_original_l185_185884


namespace isosceles_triangle_cosines_l185_185930

theorem isosceles_triangle_cosines 
  (a b c : ℝ)
  (h_isosceles : a = c)
  (α : ℝ)
  (h_triangle_angles : ∠ABC = α)
  (H : ∠BCA = α)
  (O : Type)
  (B : ℝ)
  (D : ℝ)
  (H_orthocenter_bisect : B / 2 = D) :
  cos (α) = sqrt 3 / 3 ∧ cos (∠ABC) = 1 / 3 :=
sorry

end isosceles_triangle_cosines_l185_185930


namespace boys_belong_to_other_communities_l185_185822

-- Definitions for the given problem
def total_boys : ℕ := 850
def percent_muslims : ℝ := 0.34
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10
def percent_other : ℝ := 1 - (percent_muslims + percent_hindus + percent_sikhs)

-- Statement to prove that the number of boys belonging to other communities is 238
theorem boys_belong_to_other_communities : 
  (percent_other * total_boys) = 238 := by 
  sorry

end boys_belong_to_other_communities_l185_185822


namespace minimum_value_at_x_eq_3_l185_185286

theorem minimum_value_at_x_eq_3 (b : ℝ) : 
  ∃ m : ℝ, (∀ x : ℝ, 3 * x^2 - 18 * x + b ≥ m) ∧ (3 * 3^2 - 18 * 3 + b = m) :=
by
  sorry

end minimum_value_at_x_eq_3_l185_185286


namespace calculate_A_plus_B_l185_185716

theorem calculate_A_plus_B (A B : ℝ) (h1 : A ≠ B) 
  (h2 : ∀ x : ℝ, (A * (B * x^2 + A * x + 1)^2 + B * (B * x^2 + A * x + 1) + 1) 
                - (B * (A * x^2 + B * x + 1)^2 + A * (A * x^2 + B * x + 1) + 1) 
                = x^4 + 5 * x^3 + x^2 - 4 * x) : A + B = 0 :=
by
  sorry

end calculate_A_plus_B_l185_185716


namespace bus_seat_capacity_l185_185569

theorem bus_seat_capacity (x : ℕ) : 15 * x + (15 - 3) * x + 11 = 92 → x = 3 :=
by
  sorry

end bus_seat_capacity_l185_185569


namespace apple_tree_total_production_l185_185895

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l185_185895


namespace binom_1300_2_eq_844350_l185_185318

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l185_185318


namespace winning_strategy_l185_185339

/-- Given a square table n x n, two players A and B are playing the following game: 
  - At the beginning, all cells of the table are empty.
  - Player A has the first move, and in each of their moves, a player will put a coin on some cell 
    that doesn't contain a coin and is not adjacent to any of the cells that already contain a coin. 
  - The player who makes the last move wins. 

  Cells are adjacent if they share an edge.

  - If n is even, player B has the winning strategy.
  - If n is odd, player A has the winning strategy.
-/
theorem winning_strategy (n : ℕ) : (n % 2 = 0 → ∃ (B_strat : winning_strategy_for_B), True) ∧ (n % 2 = 1 → ∃ (A_strat : winning_strategy_for_A), True) :=
by {
  admit
}

end winning_strategy_l185_185339


namespace simplify_expression_l185_185589

theorem simplify_expression (x y : ℝ) : 7 * x + 8 * y - 3 * x + 4 * y + 10 = 4 * x + 12 * y + 10 :=
by
  sorry

end simplify_expression_l185_185589


namespace roots_of_quadratic_l185_185344

theorem roots_of_quadratic (a b c : ℝ) (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) :
  ¬ ∃ (x : ℝ), x^2 + (a + b + c) * x + a^2 + b^2 + c^2 = 0 :=
by
  sorry

end roots_of_quadratic_l185_185344


namespace paper_sufficient_to_cover_cube_l185_185636

noncomputable def edge_length_cube : ℝ := 1
noncomputable def side_length_sheet : ℝ := 2.5

noncomputable def surface_area_cube : ℝ := 6
noncomputable def area_sheet : ℝ := 6.25

theorem paper_sufficient_to_cover_cube : area_sheet ≥ surface_area_cube :=
  by
    sorry

end paper_sufficient_to_cover_cube_l185_185636


namespace find_number_l185_185505

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l185_185505


namespace find_a_l185_185827

theorem find_a (a t : ℝ) 
    (h1 : (a + t) / 2 = 2020) 
    (h2 : t / 2 = 11) : 
    a = 4018 := 
by 
    sorry

end find_a_l185_185827


namespace simplify_and_evaluate_expression_l185_185994

theorem simplify_and_evaluate_expression :
  (2 * (-1/2) + 3 * 1)^2 - (2 * (-1/2) + 1) * (2 * (-1/2) - 1) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l185_185994


namespace mimi_spent_on_clothes_l185_185842

theorem mimi_spent_on_clothes : 
  let A := 800
  let N := 2 * A
  let S := 4 * A
  let P := 1 / 2 * N
  let total_spending := 10000
  let total_sneaker_spending := A + N + S + P
  let amount_spent_on_clothes := total_spending - total_sneaker_spending
  amount_spent_on_clothes = 3600 := 
by
  sorry

end mimi_spent_on_clothes_l185_185842


namespace max_sum_of_factors_l185_185277

theorem max_sum_of_factors (p q : ℕ) (hpq : p * q = 100) : p + q ≤ 101 :=
sorry

end max_sum_of_factors_l185_185277


namespace children_absent_l185_185845

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end children_absent_l185_185845


namespace highest_number_of_years_of_service_l185_185416

theorem highest_number_of_years_of_service
  (years_of_service : Fin 8 → ℕ)
  (h_range : ∃ L, ∃ H, H - L = 14)
  (h_second_highest : ∃ second_highest, second_highest = 16) :
  ∃ highest, highest = 17 := by
  sorry

end highest_number_of_years_of_service_l185_185416


namespace old_clock_slower_l185_185468

-- Given conditions
def old_clock_coincidence_minutes : ℕ := 66

-- Standard clock coincidences in 24 hours
def standard_clock_coincidences_in_24_hours : ℕ := 22

-- Standard 24 hours in minutes
def standard_24_hours_in_minutes : ℕ := 24 * 60

-- Total time for old clock in minutes over what should be 24 hours
def total_time_for_old_clock : ℕ := standard_clock_coincidences_in_24_hours * old_clock_coincidence_minutes

-- Problem statement: prove that the old clock's 24 hours is 12 minutes slower 
theorem old_clock_slower : total_time_for_old_clock = standard_24_hours_in_minutes + 12 := by
  sorry

end old_clock_slower_l185_185468


namespace prime_factor_of_difference_l185_185733

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA2 : A ≤ 9) (hB : 0 ≤ B) (hB2 : B ≤ 9) (hC : 1 ≤ C) (hC2 : C ≤ 9) (h : A ≠ C) :
  3 ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) :=
by
  dsimp at *
  rw [sub_sub, add_sub_assoc, sub_self, zero_add, sub_sub, sub_sub, add_sub_add_right_eq_sub]
  rw [mul_sub, mul_sub]
  exact dvd_mul_right 3 33 (A - C)

end prime_factor_of_difference_l185_185733


namespace no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l185_185586

-- Conditions: Expressing the sum of three reciprocals
def sum_of_reciprocals (a b c : ℕ) : ℚ := (1 / a) + (1 / b) + (1 / c)

-- Proof Problem 1: Prove that the sum of the reciprocals of any three positive integers cannot equal 9/11
theorem no_three_reciprocals_sum_to_nine_eleven :
  ∀ (a b c : ℕ), sum_of_reciprocals a b c ≠ 9 / 11 := sorry

-- Proof Problem 2: Prove that there exists no rational number between 41/42 and 1 that can be expressed as the sum of the reciprocals of three positive integers other than 41/42
theorem no_rational_between_fortyone_fortytwo_and_one :
  ∀ (K : ℚ), 41 / 42 < K ∧ K < 1 → ¬ (∃ (a b c : ℕ), sum_of_reciprocals a b c = K) := sorry

end no_three_reciprocals_sum_to_nine_eleven_no_rational_between_fortyone_fortytwo_and_one_l185_185586


namespace no_real_solution_l185_185012

theorem no_real_solution : ∀ x : ℝ, ¬ ((2*x - 3*x + 7)^2 + 4 = -|2*x|) :=
by
  intro x
  have h1 : (2*x - 3*x + 7)^2 + 4 ≥ 4 := by
    sorry
  have h2 : -|2*x| ≤ 0 := by
    sorry
  -- The main contradiction follows from comparing h1 and h2
  sorry

end no_real_solution_l185_185012


namespace chess_or_basketball_students_l185_185225

-- Definitions based on the conditions
def percentage_likes_basketball : ℝ := 0.4
def percentage_likes_chess : ℝ := 0.1
def total_students : ℕ := 250

-- Main statement to prove
theorem chess_or_basketball_students : 
  (percentage_likes_basketball + percentage_likes_chess) * total_students = 125 :=
by
  sorry

end chess_or_basketball_students_l185_185225


namespace average_waiting_time_l185_185470

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l185_185470


namespace total_mileage_pay_l185_185718

-- Conditions
def distance_first_package : ℕ := 10
def distance_second_package : ℕ := 28
def distance_third_package : ℕ := distance_second_package / 2
def total_miles_driven : ℕ := distance_first_package + distance_second_package + distance_third_package
def pay_per_mile : ℕ := 2

-- Proof statement
theorem total_mileage_pay (X : ℕ) : 
  X + (total_miles_driven * pay_per_mile) = X + 104 := by
sorry

end total_mileage_pay_l185_185718


namespace remainder_of_4521_l185_185527

theorem remainder_of_4521 (h1 : ∃ d : ℕ, d = 88)
  (h2 : 3815 % 88 = 31) : 4521 % 88 = 33 :=
sorry

end remainder_of_4521_l185_185527


namespace order_theorems_l185_185398

theorem order_theorems : 
  ∃ a b c d e f g : String,
    (a = "H") ∧ (b = "M") ∧ (c = "P") ∧ (d = "C") ∧ 
    (e = "V") ∧ (f = "S") ∧ (g = "E") ∧
    (a = "Heron's Theorem") ∧
    (b = "Menelaus' Theorem") ∧
    (c = "Pascal's Theorem") ∧
    (d = "Ceva's Theorem") ∧
    (e = "Varignon's Theorem") ∧
    (f = "Stewart's Theorem") ∧
    (g = "Euler's Theorem") := 
  sorry

end order_theorems_l185_185398


namespace average_price_of_pen_l185_185295

theorem average_price_of_pen (c_total : ℝ) (n_pens n_pencils : ℕ) (p_pencil : ℝ)
  (h1 : c_total = 450) (h2 : n_pens = 30) (h3 : n_pencils = 75) (h4 : p_pencil = 2) :
  (c_total - (n_pencils * p_pencil)) / n_pens = 10 :=
by
  sorry

end average_price_of_pen_l185_185295


namespace find_g_inv_84_l185_185214

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end find_g_inv_84_l185_185214


namespace log_equation_l185_185345

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_equation (x : ℝ) (h1 : x > 1) (h2 : (log_base_10 x)^2 - log_base_10 (x^4) = 32) :
  (log_base_10 x)^4 - log_base_10 (x^4) = 4064 :=
by
  sorry

end log_equation_l185_185345


namespace five_diff_numbers_difference_l185_185530

theorem five_diff_numbers_difference (S : Finset ℕ) (hS_size : S.card = 5) 
    (hS_range : ∀ x ∈ S, x ≤ 10) : 
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a - b = c - d ∧ a - b ≠ 0 :=
by
  sorry

end five_diff_numbers_difference_l185_185530


namespace unique_wxyz_solution_l185_185048

theorem unique_wxyz_solution (w x y z : ℕ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : w.factorial = x.factorial + y.factorial + z.factorial) : (w, x, y, z) = (3, 2, 2, 2) :=
by
  sorry

end unique_wxyz_solution_l185_185048


namespace fraction_expression_evaluation_l185_185142

theorem fraction_expression_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/4) = 1 := 
by
  sorry

end fraction_expression_evaluation_l185_185142


namespace percentage_difference_between_M_and_J_is_34_74_percent_l185_185988

-- Definitions of incomes and relationships
variables (J T M : ℝ)
variables (h1 : T = 0.80 * J)
variables (h2 : M = 1.60 * T)

-- Definitions of savings and expenses
variables (Msavings : ℝ := 0.15 * M)
variables (Mexpenses : ℝ := 0.25 * M)
variables (Tsavings : ℝ := 0.12 * T)
variables (Texpenses : ℝ := 0.30 * T)
variables (Jsavings : ℝ := 0.18 * J)
variables (Jexpenses : ℝ := 0.20 * J)

-- Total savings and expenses
variables (Mtotal : ℝ := Msavings + Mexpenses)
variables (Jtotal : ℝ := Jsavings + Jexpenses)

-- Prove the percentage difference between Mary's and Juan's total savings and expenses combined
theorem percentage_difference_between_M_and_J_is_34_74_percent :
  M = 1.28 * J → 
  Mtotal = 0.40 * M →
  Jtotal = 0.38 * J →
  ( (Mtotal - Jtotal) / Jtotal ) * 100 = 34.74 :=
by
  sorry

end percentage_difference_between_M_and_J_is_34_74_percent_l185_185988


namespace cos_seven_pi_over_six_l185_185781

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end cos_seven_pi_over_six_l185_185781


namespace binomial_variance_is_one_l185_185545

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem binomial_variance_is_one :
  binomial_variance 4 (1 / 2) = 1 := by
  sorry

end binomial_variance_is_one_l185_185545


namespace complex_right_triangle_l185_185246

open Complex

theorem complex_right_triangle {z1 z2 a b : ℂ}
  (h1 : z2 = I * z1)
  (h2 : z1 + z2 = -a)
  (h3 : z1 * z2 = b) :
  a^2 / b = 2 :=
by sorry

end complex_right_triangle_l185_185246


namespace sequence_formula_correct_l185_185232

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 2

-- Define the general term of the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 1 then -1 else 2 * n - 1

-- Theorem to prove that for the given S_n, the defined a_n is correct
theorem sequence_formula_correct (n : ℕ) (h : n > 0) : 
  a n = if n = 1 then -1 else S n - S (n - 1) :=
by sorry

end sequence_formula_correct_l185_185232


namespace function_has_two_zeros_for_a_eq_2_l185_185677

noncomputable def f (a x : ℝ) : ℝ := a ^ x - x - 1

theorem function_has_two_zeros_for_a_eq_2 :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f 2 x1 = 0 ∧ f 2 x2 = 0) := sorry

end function_has_two_zeros_for_a_eq_2_l185_185677


namespace smallest_positive_integer_a_l185_185047

theorem smallest_positive_integer_a (a : ℕ) (hpos : a > 0) :
  (∃ k, 5880 * a = k ^ 2) → a = 15 := 
by
  sorry

end smallest_positive_integer_a_l185_185047


namespace ratio_XY_7_l185_185772

variable (Z : ℕ)
variable (population_Z : ℕ := Z)
variable (population_Y : ℕ := 2 * Z)
variable (population_X : ℕ := 14 * Z)

theorem ratio_XY_7 :
  population_X / population_Y = 7 := by
  sorry

end ratio_XY_7_l185_185772


namespace polynomial_divisible_by_five_l185_185248

open Polynomial

theorem polynomial_divisible_by_five
  (a b c d m : ℤ)
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ (n : ℤ), (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := 
  sorry

end polynomial_divisible_by_five_l185_185248


namespace min_distance_PQ_l185_185970

theorem min_distance_PQ :
  let P_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = 4
  let Q_circle (ρ θ : ℝ) := ρ^2 = 4 * ρ * Real.cos θ - 3
  ∃ (P Q : ℝ × ℝ), 
    (∃ ρP θP, P = (ρP * Real.cos θP, ρP * Real.sin θP) ∧ P_line ρP θP) ∧
    (∃ ρQ θQ, Q = (ρQ * Real.cos θQ, ρQ * Real.sin θQ) ∧ Q_circle ρQ θQ) ∧
    ∀ R S : ℝ × ℝ, 
      (∃ ρR θR, R = (ρR * Real.cos θR, ρR * Real.sin θR) ∧ P_line ρR θR) →
      (∃ ρS θS, S = (ρS * Real.cos θS, ρS * Real.sin θS) ∧ Q_circle ρS θS) →
      dist P Q ≤ dist R S :=
  sorry

end min_distance_PQ_l185_185970


namespace move_decimal_point_one_place_right_l185_185392

theorem move_decimal_point_one_place_right (x : ℝ) (h : x = 76.08) : x * 10 = 760.8 :=
by
  rw [h]
  -- Here, you would provide proof steps, but we'll use sorry to indicate the proof is omitted.
  sorry

end move_decimal_point_one_place_right_l185_185392


namespace fraction_of_income_from_tips_l185_185635

variable (S T I : ℚ)

theorem fraction_of_income_from_tips
  (h₁ : T = (9 / 4) * S)
  (h₂ : I = S + T) : 
  T / I = 9 / 13 := 
sorry

end fraction_of_income_from_tips_l185_185635


namespace decreasing_function_in_interval_l185_185244

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 4)

theorem decreasing_function_in_interval (ω : ℝ) (h_omega_pos : ω > 0) (h_period : Real.pi / 3 < 2 * Real.pi / (2 * ω) ∧ 2 * Real.pi / (2 * ω) < Real.pi / 2)
    (h_symmetry : 2 * ω * 3 * Real.pi / 4 + Real.pi / 4 = (4:ℤ) * Real.pi) :
    ∀ x : ℝ, Real.pi / 6 < x ∧ x < Real.pi / 4 → f ω x < f ω (x + Real.pi / 100) :=
by
    intro x h_interval
    have ω_value : ω = 5 / 2 := sorry
    exact sorry

end decreasing_function_in_interval_l185_185244


namespace initial_deposit_l185_185159

theorem initial_deposit (x : ℝ) 
  (h1 : x - (1 / 4) * x - (4 / 9) * ((3 / 4) * x) - 640 = (3 / 20) * x) 
  : x = 2400 := 
by 
  sorry

end initial_deposit_l185_185159


namespace length_of_cloth_l185_185876

theorem length_of_cloth (L : ℝ) (h : 35 = (L + 4) * (35 / L - 1)) : L = 10 :=
sorry

end length_of_cloth_l185_185876


namespace trig_identity_l185_185197

open Real

theorem trig_identity (α : ℝ) (h_tan : tan α = 2) (h_quad : 0 < α ∧ α < π / 2) :
  sin (2 * α) + cos α = (4 + sqrt 5) / 5 :=
sorry

end trig_identity_l185_185197


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l185_185269

-- Problem 1
theorem solve_quadratic_1 (x : ℝ) : (x - 1) ^ 2 - 4 = 0 ↔ (x = -1 ∨ x = 3) :=
by
  sorry

-- Problem 2
theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1) * (x + 3) = 4 ↔ (x = -7 / 2 ∨ x = 1) :=
by
  sorry

-- Problem 3
theorem solve_quadratic_3 (x : ℝ) : 2 * x ^ 2 - 5 * x + 2 = 0 ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l185_185269


namespace least_whole_number_clock_equivalent_l185_185585

theorem least_whole_number_clock_equivalent :
  ∃ h : ℕ, h > 6 ∧ h ^ 2 % 24 = h % 24 ∧ ∀ k : ℕ, k > 6 ∧ k ^ 2 % 24 = k % 24 → h ≤ k := sorry

end least_whole_number_clock_equivalent_l185_185585


namespace find_a_b_sum_l185_185738

theorem find_a_b_sum (a b : ℕ) (h1 : 830 - (400 + 10 * a + 7) = 300 + 10 * b + 4)
    (h2 : ∃ k : ℕ, 300 + 10 * b + 4 = 7 * k) : a + b = 2 :=
by
  sorry

end find_a_b_sum_l185_185738


namespace tire_circumference_constant_l185_185755

/--
Given the following conditions:
1. Car speed v = 120 km/h
2. Tire rotation rate n = 400 rpm
3. Tire pressure P = 32 psi
4. Tire radius changes according to the formula R = R_0(1 + kP)
5. R_0 is the initial tire radius
6. k is a constant relating to the tire's elasticity
7. Change in tire pressure due to the incline is negligible

Prove that the circumference C of the tire is 5 meters.
-/
theorem tire_circumference_constant (v : ℝ) (n : ℝ) (P : ℝ) (R_0 : ℝ) (k : ℝ) 
  (h1 : v = 120 * 1000 / 3600) -- Car speed in m/s
  (h2 : n = 400 / 60)           -- Tire rotation rate in rps
  (h3 : P = 32)                 -- Tire pressure in psi
  (h4 : ∀ R P, R = R_0 * (1 + k * P)) -- Tire radius formula
  (h5 : ∀ P, P = 0)             -- Negligible change in tire pressure
  : C = 5 :=
  sorry

end tire_circumference_constant_l185_185755


namespace hyperbola_min_focal_asymptote_eq_l185_185874

theorem hyperbola_min_focal_asymptote_eq {x y m : ℝ}
  (h1 : -2 ≤ m)
  (h2 : m < 0)
  (h_eq : x^2 / m^2 - y^2 / (2 * m + 6) = 1)
  (h_min_focal : m = -1) :
  y = 2 * x ∨ y = -2 * x :=
by
  sorry

end hyperbola_min_focal_asymptote_eq_l185_185874


namespace gcd_64_144_l185_185742

theorem gcd_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end gcd_64_144_l185_185742


namespace dice_roll_probability_is_correct_l185_185890

/-- Define the probability calculation based on conditions of the problem. --/
def dice_rolls_probability_diff_by_two (successful_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

/-- Given the problem conditions, there are 8 successful outcomes and 36 total outcomes. --/
theorem dice_roll_probability_is_correct :
  dice_rolls_probability_diff_by_two 8 36 = 2 / 9 :=
by
  sorry

end dice_roll_probability_is_correct_l185_185890


namespace product_of_two_numbers_l185_185867

-- State the conditions and the proof problem
theorem product_of_two_numbers (x y : ℤ) (h_sum : x + y = 30) (h_diff : x - y = 6) :
  x * y = 216 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end product_of_two_numbers_l185_185867


namespace find_y_l185_185624

theorem find_y (x y : ℤ) (q : ℤ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x = q * y + 6) (h4 : (x : ℚ) / y = 96.15) : y = 40 :=
sorry

end find_y_l185_185624


namespace perfect_square_probability_l185_185303

theorem perfect_square_probability :
  let outcomes := Finset.pi (Finset.range 6) (fun _ => Finset.range 6)
  let is_perfect_square (l : List Nat) := ∃ n, l.prod = n * n
  let desired_outcome_count := (Finset.filter (fun l => is_perfect_square l) outcomes).card
  let total_outcomes := 6^4
  let probability := desired_outcome_count / total_outcomes
  probability = 25 / 162 :=
by
  sorry

end perfect_square_probability_l185_185303


namespace complement_of_A_l185_185044

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set A
def A : Set ℕ := {2, 4, 5}

-- Define the complement of A with respect to U
def CU : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- State the theorem that the complement of A with respect to U is {1, 3, 6, 7}
theorem complement_of_A : CU = {1, 3, 6, 7} := by
  sorry

end complement_of_A_l185_185044


namespace number_of_graphing_calculators_in_class_l185_185377

-- Define a structure for the problem
structure ClassData where
  num_boys : ℕ
  num_girls : ℕ
  num_scientific_calculators : ℕ
  num_girls_with_calculators : ℕ
  num_graphing_calculators : ℕ
  no_overlap : Prop

-- Instantiate the problem using given conditions
def mrs_anderson_class : ClassData :=
{
  num_boys := 20,
  num_girls := 18,
  num_scientific_calculators := 30,
  num_girls_with_calculators := 15,
  num_graphing_calculators := 10,
  no_overlap := true
}

-- Lean statement for the proof problem
theorem number_of_graphing_calculators_in_class (data : ClassData) :
  data.num_graphing_calculators = 10 :=
by
  sorry

end number_of_graphing_calculators_in_class_l185_185377


namespace geometric_sequence_common_ratio_l185_185830

theorem geometric_sequence_common_ratio
  (a_n : ℕ → ℝ)
  (q : ℝ)
  (h1 : a_n 3 = 7)
  (h2 : a_n 1 + a_n 2 + a_n 3 = 21) :
  q = 1 ∨ q = -1 / 2 :=
sorry

end geometric_sequence_common_ratio_l185_185830


namespace number_of_terms_l185_185278

noncomputable def Sn (n : ℕ) : ℝ := sorry

def an_arithmetic_seq (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

theorem number_of_terms {a : ℕ → ℝ}
  (h_arith : an_arithmetic_seq a)
  (cond1 : a 1 + a 2 + a 3 + a 4 = 1)
  (cond2 : a 5 + a 6 + a 7 + a 8 = 2)
  (cond3 : Sn = 15) :
  ∃ n, n = 16 :=
sorry

end number_of_terms_l185_185278


namespace unique_records_l185_185728

variable (Samantha_records : Nat)
variable (shared_records : Nat)
variable (Lily_unique_records : Nat)

theorem unique_records (h1 : Samantha_records = 24) (h2 : shared_records = 15) (h3 : Lily_unique_records = 9) :
  let Samantha_unique_records := Samantha_records - shared_records
  Samantha_unique_records + Lily_unique_records = 18 :=
by
  sorry

end unique_records_l185_185728


namespace rhombus_other_diagonal_length_l185_185100

theorem rhombus_other_diagonal_length (area_square : ℝ) (side_length_square : ℝ) (d1_rhombus : ℝ) (d2_expected: ℝ) 
  (h1 : area_square = side_length_square^2) 
  (h2 : side_length_square = 8) 
  (h3 : d1_rhombus = 16) 
  (h4 : (d1_rhombus * d2_expected) / 2 = area_square) :
  d2_expected = 8 := 
by
  sorry

end rhombus_other_diagonal_length_l185_185100


namespace lila_substituted_value_l185_185841

theorem lila_substituted_value:
  let a := 2
  let b := 3
  let c := 4
  let d := 5
  let f := 6
  ∃ e : ℚ, 20 * e = 2 * (3 - 4 * (5 - (e / 6))) ∧ e = -51 / 28 := sorry

end lila_substituted_value_l185_185841


namespace find_monthly_income_l185_185767

-- Define the percentages spent on various categories
def household_items_percentage : ℝ := 0.35
def clothing_percentage : ℝ := 0.18
def medicines_percentage : ℝ := 0.06
def entertainment_percentage : ℝ := 0.11
def transportation_percentage : ℝ := 0.12
def mutual_fund_percentage : ℝ := 0.05
def taxes_percentage : ℝ := 0.07

-- Define the savings amount
def savings_amount : ℝ := 12500

-- Total spent percentage
def total_spent_percentage := household_items_percentage + clothing_percentage + medicines_percentage + entertainment_percentage + transportation_percentage + mutual_fund_percentage + taxes_percentage

-- Percentage saved
def savings_percentage := 1 - total_spent_percentage

-- Prove that Ajay's monthly income is Rs. 208,333.33
theorem find_monthly_income (I : ℝ) (h : I * savings_percentage = savings_amount) : I = 208333.33 := by
  sorry

end find_monthly_income_l185_185767


namespace max_non_div_by_3_l185_185640

theorem max_non_div_by_3 (s : Finset ℕ) (h_len : s.card = 7) (h_prod : 3 ∣ s.prod id) : 
  ∃ n, n ≤ 6 ∧ ∀ x ∈ s, ¬ (3 ∣ x) → n = 6 :=
sorry

end max_non_div_by_3_l185_185640


namespace transistor_count_2010_l185_185637

-- Define the known constants and conditions
def initial_transistors : ℕ := 2000000
def doubling_period : ℕ := 2
def years_elapsed : ℕ := 2010 - 1995
def number_of_doublings := years_elapsed / doubling_period -- we want floor division

-- The theorem statement we need to prove
theorem transistor_count_2010 : initial_transistors * 2^number_of_doublings = 256000000 := by
  sorry

end transistor_count_2010_l185_185637


namespace one_less_than_neg_one_is_neg_two_l185_185448

theorem one_less_than_neg_one_is_neg_two : (-1 - 1 = -2) :=
by
  sorry

end one_less_than_neg_one_is_neg_two_l185_185448


namespace abs_neg_frac_l185_185096

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l185_185096


namespace find_unknown_number_l185_185600

theorem find_unknown_number :
  (0.86 ^ 3 - 0.1 ^ 3) / (0.86 ^ 2) + x + 0.1 ^ 2 = 0.76 → 
  x = 0.115296 :=
sorry

end find_unknown_number_l185_185600


namespace chantel_bracelets_at_end_l185_185933

-- Definitions based on conditions
def bracelets_day1 := 4
def days1 := 7
def given_away1 := 8

def bracelets_day2 := 5
def days2 := 10
def given_away2 := 12

-- Computation based on conditions
def total_bracelets := days1 * bracelets_day1 - given_away1 + days2 * bracelets_day2 - given_away2

-- The proof statement
theorem chantel_bracelets_at_end : total_bracelets = 58 := by
  sorry

end chantel_bracelets_at_end_l185_185933


namespace Vikas_submitted_6_questions_l185_185260

theorem Vikas_submitted_6_questions (R V A : ℕ) (h1 : 7 * V = 3 * R) (h2 : 2 * V = 3 * A) (h3 : R + V + A = 24) : V = 6 :=
by
  sorry

end Vikas_submitted_6_questions_l185_185260


namespace uniform_pdf_normalization_l185_185932

open MeasureTheory

variables (α β : ℝ) (p : ℝ → ℝ)

-- Define the probability density function for the uniform distribution
def uniform_pdf (c : ℝ) (x : ℝ) : ℝ :=
  if (α ≤ x ∧ x ≤ β) then c else 0

-- The integral condition for any probability density function
def integral_condition (p : ℝ → ℝ) : Prop :=
  ∫ x, p x = 1

-- The uniform_pdf function must satisfy the integral condition
theorem uniform_pdf_normalization (h : α < β) :
  integral_condition (uniform_pdf (1 / (β - α)) α β) :=
by
  sorry

end uniform_pdf_normalization_l185_185932


namespace tangent_line_to_curve_perpendicular_l185_185698

noncomputable def perpendicular_tangent_line (x y : ℝ) : Prop :=
  y = x^4 ∧ (4*x - y - 3 = 0)

theorem tangent_line_to_curve_perpendicular {x y : ℝ} (h : y = x^4 ∧ (4*x - y - 3 = 0)) :
  ∃ (x y : ℝ), (x+4*y-8=0) ∧ (4*x - y - 3 = 0) :=
by
  sorry

end tangent_line_to_curve_perpendicular_l185_185698


namespace sequence_monotonic_b_gt_neg3_l185_185947

theorem sequence_monotonic_b_gt_neg3 (b : ℝ) :
  (∀ n : ℕ, n > 0 → (n+1)^2 + b*(n+1) > n^2 + b*n) ↔ b > -3 :=
by sorry

end sequence_monotonic_b_gt_neg3_l185_185947


namespace find_n_l185_185851

theorem find_n (n : ℕ) (d : ℕ) (h_pos : n > 0) (h_digit : d < 10) (h_equiv : n * 999 = 810 * (100 * d + 25)) : n = 750 :=
  sorry

end find_n_l185_185851


namespace exists_25_consecutive_odd_numbers_l185_185321

theorem exists_25_consecutive_odd_numbers
    (seq : Fin 25 → ℤ)
    (h1 : ∀ i : Fin 25, seq i = -23 + 2 * ↑i) :
    ∃ (sum_prod_is_square : ∃ (S P : ℤ), S = (Finset.univ.sum seq) ∧ P = (Finset.univ.prod seq) ∧ S = k^2 ∧ P = m^2) :=
begin
    -- Proof is omitted
    sorry
end

end exists_25_consecutive_odd_numbers_l185_185321


namespace range_of_a_l185_185359

theorem range_of_a (x a : ℝ) (h₁ : 0 < x) (h₂ : x < 2) (h₃ : a - 1 < x) (h₄ : x ≤ a) :
  1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l185_185359


namespace percent_decrease_area_square_l185_185820

/-- 
In a configuration, two figures, an equilateral triangle and a square, are initially given. 
The equilateral triangle has an area of 27√3 square inches, and the square has an area of 27 square inches.
If the side length of the square is decreased by 10%, prove that the percent decrease in the area of the square is 19%.
-/
theorem percent_decrease_area_square 
  (triangle_area : ℝ := 27 * Real.sqrt 3)
  (square_area : ℝ := 27)
  (percentage_decrease : ℝ := 0.10) : 
  let new_square_side := Real.sqrt square_area * (1 - percentage_decrease)
  let new_square_area := new_square_side ^ 2
  let area_decrease := square_area - new_square_area
  let percent_decrease := (area_decrease / square_area) * 100
  percent_decrease = 19 := 
by
  sorry

end percent_decrease_area_square_l185_185820


namespace math_problem_l185_185553

noncomputable def a : ℝ := 0.137
noncomputable def b : ℝ := 0.098
noncomputable def c : ℝ := 0.123
noncomputable def d : ℝ := 0.086

theorem math_problem : 
  ( ((a + b)^2 - (a - b)^2) / (c * d) + (d^3 - c^3) / (a * b * (a + b)) ) = 4.6886 := 
  sorry

end math_problem_l185_185553


namespace average_rainfall_l185_185376

theorem average_rainfall {R D H: ℕ} (hR : R = 320) (hD : D = 30) (hH: H = 24) :
  (R / (D * H) : ℚ) = 4 / 9 :=
by {
  -- start of the proof
  sorry
}

end average_rainfall_l185_185376


namespace unique_solution_c_exceeds_s_l185_185452

-- Problem Conditions
def steers_cost : ℕ := 35
def cows_cost : ℕ := 40
def total_budget : ℕ := 1200

-- Definition of the solution conditions
def valid_purchase (s c : ℕ) : Prop := 
  steers_cost * s + cows_cost * c = total_budget ∧ s > 0 ∧ c > 0

-- Statement to prove
theorem unique_solution_c_exceeds_s :
  ∃ s c : ℕ, valid_purchase s c ∧ c > s ∧ ∀ (s' c' : ℕ), valid_purchase s' c' → s' = 8 ∧ c' = 17 :=
sorry

end unique_solution_c_exceeds_s_l185_185452


namespace max_AB_CD_value_l185_185840

def is_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

noncomputable def max_AB_CD : ℕ :=
  let A := 9
  let B := 8
  let C := 7
  let D := 6
  (A + B) + (C + D)

theorem max_AB_CD_value :
  ∀ (A B C D : ℕ), 
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (A + B) + (C + D) ≤ max_AB_CD :=
by
  sorry

end max_AB_CD_value_l185_185840


namespace donuts_left_for_coworkers_l185_185909

theorem donuts_left_for_coworkers :
  ∀ (total_donuts gluten_free regular gluten_free_chocolate gluten_free_plain regular_chocolate regular_plain consumed_gluten_free consumed_regular afternoon_gluten_free_chocolate afternoon_gluten_free_plain afternoon_regular_chocolate afternoon_regular_plain left_gluten_free_chocolate left_gluten_free_plain left_regular_chocolate left_regular_plain),
  total_donuts = 30 →
  gluten_free = 12 →
  regular = 18 →
  gluten_free_chocolate = 6 →
  gluten_free_plain = 6 →
  regular_chocolate = 11 →
  regular_plain = 7 →
  consumed_gluten_free = 1 →
  consumed_regular = 1 →
  afternoon_gluten_free_chocolate = 2 →
  afternoon_gluten_free_plain = 1 →
  afternoon_regular_chocolate = 2 →
  afternoon_regular_plain = 1 →
  left_gluten_free_chocolate = gluten_free_chocolate - consumed_gluten_free * 0.5 - afternoon_gluten_free_chocolate →
  left_gluten_free_plain = gluten_free_plain - consumed_gluten_free * 0.5 - afternoon_gluten_free_plain →
  left_regular_chocolate = regular_chocolate - consumed_regular * 1 - afternoon_regular_chocolate →
  left_regular_plain = regular_plain - consumed_regular * 0 - afternoon_regular_plain →
  left_gluten_free_chocolate + left_gluten_free_plain + left_regular_chocolate + left_regular_plain = 23 :=
by
  intros
  sorry

end donuts_left_for_coworkers_l185_185909


namespace fifteen_percent_of_x_is_ninety_l185_185502

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l185_185502


namespace num_real_a_with_int_roots_l185_185184

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end num_real_a_with_int_roots_l185_185184


namespace inequality_holds_for_all_xyz_in_unit_interval_l185_185265

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end inequality_holds_for_all_xyz_in_unit_interval_l185_185265


namespace constant_term_binomial_expansion_l185_185661

theorem constant_term_binomial_expansion : 
  let a := (1 : ℚ) / (x : ℚ) -- Note: Here 'x' is not bound, in actual Lean code x should be a declared variable in ℚ.
  let b := 2 * (x : ℚ)
  let n := 6
  let T (r : ℕ) := (Nat.choose n r : ℚ) * a^(n - r) * b^r
  (T 3) = (160 : ℚ) := by
  sorry

end constant_term_binomial_expansion_l185_185661


namespace monotone_increasing_function_range_l185_185679

theorem monotone_increasing_function_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2 : ℝ) (3 : ℝ), (1 / x + 2 * a * x - 3) ≥ 0) ↔ a ≥ 9 / 8 := 
by 
  sorry

end monotone_increasing_function_range_l185_185679


namespace marbles_leftover_l185_185144

theorem marbles_leftover (r p : ℤ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 := by
  sorry

end marbles_leftover_l185_185144


namespace find_expression_l185_185041

theorem find_expression (x y : ℝ) (h1 : 4 * x + y = 17) (h2 : x + 4 * y = 23) :
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 :=
by
  sorry

end find_expression_l185_185041


namespace no_polygon_with_half_parallel_diagonals_l185_185496

open Set

noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

def is_parallel_diagonal (n i j : ℕ) : Bool := 
  -- Here, you should define the mathematical condition of a diagonal being parallel to a side
  ((j - i) % n = 0) -- This is a placeholder; the actual condition would depend on the precise geometric definition.

theorem no_polygon_with_half_parallel_diagonals (n : ℕ) (h1 : n ≥ 3) :
  ¬(∃ (k : ℕ), k = num_diagonals n ∧ (∀ (i j : ℕ), i < j ∧ is_parallel_diagonal n i j = true → k = num_diagonals n / 2)) :=
by
  sorry

end no_polygon_with_half_parallel_diagonals_l185_185496


namespace coffee_shop_ratio_l185_185498

theorem coffee_shop_ratio (morning_usage afternoon_multiplier weekly_usage days_per_week : ℕ) (r : ℕ) 
  (h_morning : morning_usage = 3)
  (h_afternoon : afternoon_multiplier = 3)
  (h_weekly : weekly_usage = 126)
  (h_days : days_per_week = 7):
  weekly_usage = days_per_week * (morning_usage + afternoon_multiplier * morning_usage + r * morning_usage) →
  r = 2 :=
by
  intros h_eq
  sorry

end coffee_shop_ratio_l185_185498


namespace jesse_needs_more_carpet_l185_185709

def additional_carpet_needed (carpet : ℕ) (length : ℕ) (width : ℕ) : ℕ :=
  let room_area := length * width
  room_area - carpet

theorem jesse_needs_more_carpet
  (carpet : ℕ) (length : ℕ) (width : ℕ)
  (h_carpet : carpet = 18)
  (h_length : length = 4)
  (h_width : width = 20) :
  additional_carpet_needed carpet length width = 62 :=
by {
  -- the proof goes here
  sorry
}

end jesse_needs_more_carpet_l185_185709


namespace avg_waiting_time_is_1_point_2_minutes_l185_185475

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l185_185475


namespace total_time_for_12000_dolls_l185_185978

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end total_time_for_12000_dolls_l185_185978


namespace rotated_line_eq_l185_185403

theorem rotated_line_eq :
  ∀ (x y : ℝ), 
  (x - y + 4 = 0) ∨ (x - y - 4 = 0) ↔ 
  ∃ (x' y' : ℝ), (-x', -y') = (x, y) ∧ (x' - y' + 4 = 0) :=
by
  sorry

end rotated_line_eq_l185_185403


namespace product_of_terms_geometric_sequence_l185_185343

variable {a : ℕ → ℝ}
variable {q : ℝ}
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem product_of_terms_geometric_sequence
  (ha: geometric_sequence a q)
  (h3_4: a 3 * a 4 = 6) :
  a 2 * a 5 = 6 :=
by
  sorry

end product_of_terms_geometric_sequence_l185_185343


namespace sum_of_first_ten_nicely_odd_numbers_is_775_l185_185919

def is_nicely_odd (n : ℕ) : Prop :=
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ (Odd p ∧ Odd q) ∧ n = p * q)
  ∨ (∃ p : ℕ, Nat.Prime p ∧ Odd p ∧ n = p ^ 3)

theorem sum_of_first_ten_nicely_odd_numbers_is_775 :
  let nicely_odd_nums := [15, 27, 21, 35, 125, 33, 77, 343, 55, 39]
  ∃ (nums : List ℕ), List.length nums = 10 ∧
  (∀ n ∈ nums, is_nicely_odd n) ∧ List.sum nums = 775 := by
  sorry

end sum_of_first_ten_nicely_odd_numbers_is_775_l185_185919


namespace intersection_value_unique_l185_185115

theorem intersection_value_unique (x : ℝ) :
  (∃ y : ℝ, y = 8 / (x^2 + 4) ∧ x + y = 2) → x = 0 :=
by
  sorry

end intersection_value_unique_l185_185115


namespace minimum_value_expression_l185_185660

theorem minimum_value_expression (x : ℝ) (h : -3 < x ∧ x < 2) :
  ∃ y, y = (x^2 + 4 * x + 5) / (2 * x + 6) ∧ y = 3 / 4 :=
by
  sorry

end minimum_value_expression_l185_185660


namespace range_of_m_l185_185194

variable {m x x1 x2 y1 y2 : ℝ}

noncomputable def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (2 + m)

theorem range_of_m (h1 : x1 < x2) (h2 : y1 = linear_function m x1) (h3 : y2 = linear_function m x2) (h4 : y1 > y2) : m < 2 :=
by
  sorry

end range_of_m_l185_185194


namespace limit_example_l185_185400

open Real

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ → abs ((2 * x^2 + 6 * x - 8) / (x + 4) + 10) < ε) :=
by
  sorry

end limit_example_l185_185400


namespace A_minus_B_l185_185836

def A : ℕ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℕ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem A_minus_B : A - B = 128 := by
  sorry

end A_minus_B_l185_185836


namespace more_trees_died_than_survived_l185_185207

def haley_trees : ℕ := 14
def died_in_typhoon : ℕ := 9
def survived_trees := haley_trees - died_in_typhoon

theorem more_trees_died_than_survived : (died_in_typhoon - survived_trees) = 4 := by
  -- proof goes here
  sorry

end more_trees_died_than_survived_l185_185207


namespace eric_containers_l185_185324

theorem eric_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) 
  (h1 : initial_pencils = 150) (h2 : additional_pencils = 30) (h3 : pencils_per_container = 36) :
  (initial_pencils + additional_pencils) / pencils_per_container = 5 := 
by {
  sorry
}

end eric_containers_l185_185324


namespace probability_all_different_digits_l185_185644

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l185_185644


namespace perpendicular_line_slope_l185_185539

theorem perpendicular_line_slope (a : ℝ) :
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  k_MN * (-a / 2) = -1 → a = 1 :=
by
  intros M N k_MN H
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  sorry

end perpendicular_line_slope_l185_185539


namespace total_marbles_l185_185076

-- Define the number of marbles Mary has
def marblesMary : Nat := 9 

-- Define the number of marbles Joan has
def marblesJoan : Nat := 3 

-- Theorem to prove the total number of marbles
theorem total_marbles : marblesMary + marblesJoan = 12 := 
by sorry

end total_marbles_l185_185076


namespace new_average_daily_production_l185_185669

theorem new_average_daily_production (n : ℕ) (avg_past : ℕ) (production_today : ℕ) (new_avg : ℕ)
  (h1 : n = 9)
  (h2 : avg_past = 50)
  (h3 : production_today = 100)
  (h4 : new_avg = (avg_past * n + production_today) / (n + 1)) :
  new_avg = 55 :=
by
  -- Using the provided conditions, it will be shown in the proof stage that new_avg equals 55
  sorry

end new_average_daily_production_l185_185669


namespace find_d_k_l185_185674

open Matrix

noncomputable def matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![6, d]]

noncomputable def inv_matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let detA := 3 * d - 24
  (1 / detA) • ![![d, -4], ![-6, 3]]

theorem find_d_k (d k : ℝ) (h : inv_matrix_A d = k • matrix_A d) :
    (d, k) = (-3, 1/33) := by
  sorry

end find_d_k_l185_185674


namespace larger_segment_of_triangle_l185_185866

theorem larger_segment_of_triangle (a b c : ℝ) (h : ℝ) (hc : c = 100) (ha : a = 40) (hb : b = 90) 
  (h_triangle : a^2 + h^2 = x^2)
  (h_triangle2 : b^2 + h^2 = (100 - x)^2) :
  100 - x = 82.5 :=
sorry

end larger_segment_of_triangle_l185_185866


namespace length_of_shorter_train_l185_185427

noncomputable def relativeSpeedInMS (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  (speed1_kmh + speed2_kmh) * (5 / 18)

noncomputable def totalDistanceCovered (relativeSpeed_ms time_s : ℝ) : ℝ :=
  relativeSpeed_ms * time_s

noncomputable def lengthOfShorterTrain (longerTrainLength_m time_s : ℝ) (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relativeSpeed_ms := relativeSpeedInMS speed1_kmh speed2_kmh
  let totalDistance := totalDistanceCovered relativeSpeed_ms time_s
  totalDistance - longerTrainLength_m

theorem length_of_shorter_train :
  lengthOfShorterTrain 160 10.07919366450684 60 40 = 117.8220467912412 := 
sorry

end length_of_shorter_train_l185_185427


namespace picked_clovers_when_one_four_found_l185_185575

-- Definition of conditions
def total_leaves : ℕ := 100
def leaves_three_leaved_clover : ℕ := 3
def leaves_four_leaved_clover : ℕ := 4
def one_four_leaved_clover : ℕ := 1

-- Proof Statement
theorem picked_clovers_when_one_four_found (three_leaved_count : ℕ) :
  (total_leaves - leaves_four_leaved_clover) / leaves_three_leaved_clover = three_leaved_count → 
  three_leaved_count = 32 :=
by
  sorry

end picked_clovers_when_one_four_found_l185_185575


namespace problem_solution_l185_185067

theorem problem_solution (a b : ℝ) (h1 : a^3 - 15 * a^2 + 25 * a - 75 = 0) (h2 : 8 * b^3 - 60 * b^2 - 310 * b + 2675 = 0) :
  a + b = 15 / 2 :=
sorry

end problem_solution_l185_185067


namespace identity_eq_coefficients_l185_185217

theorem identity_eq_coefficients (a b c d : ℝ) :
  (∀ x : ℝ, a * x + b = c * x + d) ↔ (a = c ∧ b = d) :=
by
  sorry

end identity_eq_coefficients_l185_185217


namespace probability_all_digits_different_l185_185653

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l185_185653


namespace exponentiation_addition_l185_185499

theorem exponentiation_addition : (3^3)^2 + 1 = 730 := by
  sorry

end exponentiation_addition_l185_185499


namespace molecular_weight_l185_185612

-- Definitions of the molar masses of the elements
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_I : ℝ := 126.90
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.07
def molar_mass_O : ℝ := 16.00

-- Definition of the molar masses of the compounds
def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Number of moles
def moles_NH4I : ℝ := 3
def moles_CaSO4 : ℝ := 2

-- Total mass calculation
def total_mass : ℝ :=
  moles_NH4I * molar_mass_NH4I + 
  moles_CaSO4 * molar_mass_CaSO4

-- Problem statement
theorem molecular_weight : total_mass = 707.15 := by
  sorry

end molecular_weight_l185_185612


namespace andrea_avg_km_per_day_l185_185004

theorem andrea_avg_km_per_day
  (total_distance : ℕ := 168)
  (total_days : ℕ := 6)
  (completed_fraction : ℚ := 3/7)
  (completed_days : ℕ := 3) :
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := 
sorry

end andrea_avg_km_per_day_l185_185004


namespace Gargamel_bought_tires_l185_185188

def original_price_per_tire := 84
def sale_price_per_tire := 75
def total_savings := 36
def discount_per_tire := original_price_per_tire - sale_price_per_tire
def num_tires (total_savings : ℕ) (discount_per_tire : ℕ) := total_savings / discount_per_tire

theorem Gargamel_bought_tires :
  num_tires total_savings discount_per_tire = 4 :=
by
  sorry

end Gargamel_bought_tires_l185_185188


namespace bill_weight_training_l185_185005

theorem bill_weight_training (jugs : ℕ) (gallons_per_jug : ℝ) (percent_filled : ℝ) (density : ℝ) 
  (h_jugs : jugs = 2)
  (h_gallons_per_jug : gallons_per_jug = 2)
  (h_percent_filled : percent_filled = 0.70)
  (h_density : density = 5) :
  jugs * gallons_per_jug * percent_filled * density = 14 := 
by
  subst h_jugs
  subst h_gallons_per_jug
  subst h_percent_filled
  subst h_density
  norm_num
  done

end bill_weight_training_l185_185005


namespace sam_age_l185_185769

-- Definitions
variables (B J S : ℕ)
axiom H1 : B = 2 * J
axiom H2 : B + J = 60
axiom H3 : S = (B + J) / 2

-- Problem statement
theorem sam_age : S = 30 :=
sorry

end sam_age_l185_185769


namespace ruby_shares_with_9_friends_l185_185726

theorem ruby_shares_with_9_friends
    (total_candies : ℕ) (candies_per_friend : ℕ)
    (h1 : total_candies = 36) (h2 : candies_per_friend = 4) :
    total_candies / candies_per_friend = 9 := by
  sorry

end ruby_shares_with_9_friends_l185_185726


namespace semicircle_problem_l185_185230

theorem semicircle_problem (N : ℕ) (r : ℝ) (π : ℝ) (hπ : 0 < π) 
  (h1 : ∀ (r : ℝ), ∃ (A B : ℝ), A = N * (π * r^2 / 2) ∧ B = (π * (N^2 * r^2 / 2) - N * (π * r^2 / 2)) ∧ A / B = 1 / 3) :
  N = 4 :=
by
  sorry

end semicircle_problem_l185_185230


namespace escalator_length_l185_185166

theorem escalator_length
  (escalator_speed : ℕ)
  (person_speed : ℕ)
  (time_taken : ℕ)
  (combined_speed : ℕ)
  (condition1 : escalator_speed = 12)
  (condition2 : person_speed = 2)
  (condition3 : time_taken = 14)
  (condition4 : combined_speed = escalator_speed + person_speed)
  (condition5 : combined_speed * time_taken = 196) :
  combined_speed * time_taken = 196 := 
by
  -- the proof would go here
  sorry

end escalator_length_l185_185166


namespace max_edges_partitioned_square_l185_185362

theorem max_edges_partitioned_square (n v e : ℕ) 
  (h : v - e + n = 1) : e ≤ 3 * n + 1 := 
sorry

end max_edges_partitioned_square_l185_185362


namespace haley_total_expenditure_l185_185208

-- Definition of conditions
def ticket_cost : ℕ := 4
def tickets_bought_for_self_and_friends : ℕ := 3
def tickets_bought_for_others : ℕ := 5
def total_tickets : ℕ := tickets_bought_for_self_and_friends + tickets_bought_for_others

-- Proof statement
theorem haley_total_expenditure : total_tickets * ticket_cost = 32 := by
  sorry

end haley_total_expenditure_l185_185208


namespace julia_drove_214_miles_l185_185756

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end julia_drove_214_miles_l185_185756


namespace multiplication_decomposition_l185_185626

theorem multiplication_decomposition :
  100 * 3 = 100 + 100 + 100 :=
sorry

end multiplication_decomposition_l185_185626


namespace distance_between_points_on_parabola_l185_185300

theorem distance_between_points_on_parabola (x1 y1 x2 y2 : ℝ) 
  (h_parabola : ∀ (x : ℝ), 4 * ((x^2)/4) = x^2) 
  (h_focus : F = (0, 1))
  (h_line : y1 = k * x1 + 1 ∧ y2 = k * x2 + 1)
  (h_intersects : x1^2 = 4 * y1 ∧ x2^2 = 4 * y2)
  (h_y_sum : y1 + y2 = 6) :
  |dist (x1, y1) (x2, y2)| = 8 := sorry

end distance_between_points_on_parabola_l185_185300


namespace obtuse_triangle_area_side_l185_185282

theorem obtuse_triangle_area_side (a b : ℝ) (C : ℝ) 
  (h1 : a = 8) 
  (h2 : C = 150 * (π / 180)) -- converting degrees to radians
  (h3 : 1 / 2 * a * b * Real.sin C = 24) : 
  b = 12 :=
by sorry

end obtuse_triangle_area_side_l185_185282


namespace base6_problem_l185_185850

theorem base6_problem
  (x y : ℕ)
  (h1 : 453 = 2 * x * 10 + y) -- Constraint from base-6 to base-10 conversion
  (h2 : 0 ≤ x ∧ x ≤ 9) -- x is a base-10 digit
  (h3 : 0 ≤ y ∧ y ≤ 9) -- y is a base-10 digit
  (h4 : 4 * 6^2 + 5 * 6 + 3 = 177) -- Conversion result for 453_6
  (h5 : 2 * x * 10 + y = 177) -- Conversion from condition
  (hx : x = 7) -- x value from solution
  (hy : y = 7) -- y value from solution
  : (x * y) / 10 = 49 / 10 := 
by 
  sorry

end base6_problem_l185_185850


namespace volume_of_pyramid_l185_185486

theorem volume_of_pyramid (A B C : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (28, 0)) (hC : C = (12, 20))
  (D : ℝ × ℝ) (hD : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (E : ℝ × ℝ) (hE : E = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (F : ℝ × ℝ) (hF : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (∃ h : ℝ, h = 10 ∧ ∃ V : ℝ, V = (1 / 3) * 70 * h ∧ V = 700 / 3) :=
by sorry

end volume_of_pyramid_l185_185486


namespace seashells_total_l185_185081

theorem seashells_total :
  let sally := 9.5
  let tom := 7.2
  let jessica := 5.3
  let alex := 12.8
  sally + tom + jessica + alex = 34.8 :=
by
  sorry

end seashells_total_l185_185081


namespace determine_number_of_solutions_l185_185662

noncomputable def num_solutions_eq : Prop :=
  let f (x : ℝ) := (3 * x ^ 2 - 15 * x) / (x ^ 2 - 7 * x + 10)
  let g (x : ℝ) := x - 4
  ∃ S : Finset ℝ, 
    (∀ x ∈ S, (x ≠ 2 ∧ x ≠ 5) ∧ f x = g x) ∧
    S.card = 2

theorem determine_number_of_solutions : num_solutions_eq :=
  by
  sorry

end determine_number_of_solutions_l185_185662


namespace simplify_expr_l185_185992

theorem simplify_expr (x : ℝ) : (3 * x)^5 + (4 * x) * (x^4) = 247 * x^5 :=
by
  sorry

end simplify_expr_l185_185992


namespace total_houses_l185_185054

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end total_houses_l185_185054


namespace vector_on_line_l185_185010

theorem vector_on_line (t : ℝ) (x y : ℝ) : 
  (x = 3 * t + 1) → (y = 2 * t + 3) → 
  ∃ t, (∃ x y, (x = 3 * t + 1) ∧ (y = 2 * t + 3) ∧ (x = 23 / 2) ∧ (y = 10)) :=
  by
  sorry

end vector_on_line_l185_185010


namespace number_is_square_l185_185670

theorem number_is_square (x y : ℕ) : (∃ n : ℕ, (1100 * x + 11 * y = n^2)) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_is_square_l185_185670


namespace circular_garden_area_l185_185757

open Real

theorem circular_garden_area (r : ℝ) (h₁ : r = 8)
      (h₂ : 2 * π * r = (1 / 4) * π * r ^ 2) :
  π * r ^ 2 = 64 * π :=
by
  -- The proof will go here
  sorry

end circular_garden_area_l185_185757


namespace selection_ways_l185_185302

/-- 
A math interest group in a vocational school consists of 4 boys and 3 girls. 
If 3 students are randomly selected from these 7 students to participate in a math competition, 
and the selection must include both boys and girls, then the number of different ways to select the 
students is 30.
-/
theorem selection_ways (B G : ℕ) (students : ℕ) (selections : ℕ) (condition_boys_girls : B = 4 ∧ G = 3)
  (condition_students : students = B + G) (condition_selections : selections = 3) :
  (B = 4 ∧ G = 3 ∧ students = 7 ∧ selections = 3) → 
  ∃ (res : ℕ), res = 30 :=
by
  sorry

end selection_ways_l185_185302


namespace fleas_after_treatment_l185_185758

theorem fleas_after_treatment
  (F : ℕ)  -- F is the number of fleas the dog has left after the treatments
  (half_fleas : ℕ → ℕ)  -- Function representing halving fleas
  (initial_fleas := F + 210)  -- Initial number of fleas before treatment
  (half_fleas_def : ∀ n, half_fleas n = n / 2)  -- Definition of half_fleas function
  (condition : F = (half_fleas (half_fleas (half_fleas (half_fleas initial_fleas)))))  -- Condition given in the problem
  :
  F = 14 := 
  sorry

end fleas_after_treatment_l185_185758


namespace probability_of_same_color_balls_l185_185753

theorem probability_of_same_color_balls :
  let num_balls := 5
  let num_white := 3
  let num_black := 2
  let total_drawn := 2
  let total_outcomes := nat.choose num_balls total_drawn
  let white_outcomes := nat.choose num_white total_drawn
  let black_outcomes := nat.choose num_black total_drawn
  let favorable_outcomes := white_outcomes + black_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability = (2 : ℚ) / 5 :=
by
  sorry

end probability_of_same_color_balls_l185_185753


namespace number_of_real_a_l185_185183

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end number_of_real_a_l185_185183


namespace min_sum_product_l185_185940

theorem min_sum_product (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 9/n = 1) :
  m * n = 48 :=
sorry

end min_sum_product_l185_185940


namespace max_wx_plus_xy_plus_yz_l185_185983

theorem max_wx_plus_xy_plus_yz (w x y z : ℝ) (h1 : w ≥ 0) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) (h_sum : w + x + y + z = 200) : wx + xy + yz ≤ 10000 :=
sorry

end max_wx_plus_xy_plus_yz_l185_185983


namespace no_distributive_laws_hold_l185_185577

def tripledAfterAdding (a b : ℝ) : ℝ := 3 * (a + b)

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (tripledAfterAdding x (y + z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) ∧
  ¬ (x + (tripledAfterAdding y z) = tripledAfterAdding (x + y) (x + z)) ∧
  ¬ (tripledAfterAdding x (tripledAfterAdding y z) = tripledAfterAdding (tripledAfterAdding x y) (tripledAfterAdding x z)) :=
by sorry

end no_distributive_laws_hold_l185_185577


namespace spherical_to_rectangular_correct_l185_185490

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end spherical_to_rectangular_correct_l185_185490


namespace railway_tunnel_construction_days_l185_185881

theorem railway_tunnel_construction_days
  (a b t : ℝ)
  (h1 : a = 1/3)
  (h2 : b = 20/100)
  (h3 : t = 4/5 ∨ t = 0.8)
  (total_days : ℝ)
  (h_total_days : total_days = 185)
  : total_days = 180 := 
sorry

end railway_tunnel_construction_days_l185_185881


namespace cupcakes_per_child_l185_185279

theorem cupcakes_per_child (total_cupcakes children : ℕ) (h1 : total_cupcakes = 96) (h2 : children = 8) : total_cupcakes / children = 12 :=
by
  sorry

end cupcakes_per_child_l185_185279


namespace new_person_weight_l185_185410

noncomputable def weight_of_new_person (weight_of_replaced : ℕ) (number_of_persons : ℕ) (increase_in_average : ℕ) := 
  weight_of_replaced + number_of_persons * increase_in_average

theorem new_person_weight:
  weight_of_new_person 70 8 3 = 94 :=
  by
  -- Proof omitted
  sorry

end new_person_weight_l185_185410


namespace simplify_fraction_l185_185995

theorem simplify_fraction :
  ( (3 * 5 * 7 : ℚ) / (9 * 11 * 13) ) * ( (7 * 9 * 11 * 15) / (3 * 5 * 14) ) = 15 / 26 :=
by
  sorry

end simplify_fraction_l185_185995


namespace geometric_series_sum_l185_185614

open Real

theorem geometric_series_sum :
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  S = -716637955 / 16777216 :=
by
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  have h : S = -716637955 / 16777216 := sorry
  exact h

end geometric_series_sum_l185_185614


namespace fraction_product_eq_one_l185_185771

theorem fraction_product_eq_one :
  (7 / 4 : ℚ) * (8 / 14) * (21 / 12) * (16 / 28) * (49 / 28) * (24 / 42) * (63 / 36) * (32 / 56) = 1 := by
  sorry

end fraction_product_eq_one_l185_185771


namespace binom_1300_2_eq_l185_185314

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l185_185314


namespace min_area_and_line_eq_l185_185299

theorem min_area_and_line_eq (a b : ℝ) (l : ℝ → ℝ → Prop)
    (h1 : l 3 2)
    (h2: ∀ x y: ℝ, l x y → (x/a + y/b = 1))
    (h3: a > 0)
    (h4: b > 0)
    : 
    a = 6 ∧ b = 4 ∧ 
    (∀ x y : ℝ, l x y ↔ (4 * x + 6 * y - 24 = 0)) ∧ 
    (∃ min_area : ℝ, min_area = 12) :=
by
  sorry

end min_area_and_line_eq_l185_185299


namespace quadratic_m_condition_l185_185558

theorem quadratic_m_condition (m : ℝ) (h_eq : (m - 2) * x ^ (m ^ 2 - 2) - m * x + 1 = 0) (h_pow : m ^ 2 - 2 = 2) :
  m = -2 :=
by sorry

end quadratic_m_condition_l185_185558


namespace digits_probability_l185_185650

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l185_185650


namespace ratio_of_areas_GHI_to_JKL_l185_185123

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l185_185123


namespace beka_flew_more_l185_185625

def bekaMiles := 873
def jacksonMiles := 563

theorem beka_flew_more : bekaMiles - jacksonMiles = 310 := by
  -- proof here
  sorry

end beka_flew_more_l185_185625


namespace operation_is_double_l185_185668

theorem operation_is_double (x : ℝ) (operation : ℝ → ℝ) (h1: x^2 = 25) (h2: operation x = x / 5 + 9) : operation x = 2 * x :=
by
  sorry

end operation_is_double_l185_185668


namespace find_ab_l185_185707

theorem find_ab (a b : ℝ) (h1 : a - b = 26) (h2 : a + b = 15) :
  a = 41 / 2 ∧ b = 11 / 2 :=
sorry

end find_ab_l185_185707


namespace point_M_quadrant_l185_185800

theorem point_M_quadrant (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) :
  (0 < Real.sin θ) ∧ (Real.cos θ < 0) :=
by
  sorry

end point_M_quadrant_l185_185800


namespace ratio_of_areas_l185_185128

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l185_185128


namespace inequality_proof_l185_185263

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end inequality_proof_l185_185263


namespace lincoln_high_fraction_of_girls_l185_185169

noncomputable def fraction_of_girls_in_science_fair (total_girls total_boys : ℕ) (frac_girls_participated frac_boys_participated : ℚ) : ℚ :=
  let participating_girls := frac_girls_participated * total_girls
  let participating_boys := frac_boys_participated * total_boys
  participating_girls / (participating_girls + participating_boys)

theorem lincoln_high_fraction_of_girls 
  (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_participated : ℚ) (frac_boys_participated : ℚ)
  (h1 : total_girls = 150) (h2 : total_boys = 100)
  (h3 : frac_girls_participated = 4/5) (h4 : frac_boys_participated = 3/4) :
  fraction_of_girls_in_science_fair total_girls total_boys frac_girls_participated frac_boys_participated = 8/13 := 
by
  sorry

end lincoln_high_fraction_of_girls_l185_185169


namespace abs_neg_three_halves_l185_185087

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l185_185087


namespace handshake_problem_l185_185420

theorem handshake_problem (n : ℕ) (h : n * (n - 1) / 2 = 1770) : n = 60 :=
sorry

end handshake_problem_l185_185420


namespace obtain_half_not_obtain_one_l185_185440

theorem obtain_half (x : ℕ) : (10 + x) / (97 + x) = 1 / 2 ↔ x = 77 := 
by
  sorry

theorem not_obtain_one (x k : ℕ) : ¬ ((10 + x) / (97 + x) = 1 ∨ (10 * k) / (97 * k) = 1) := 
by
  sorry

end obtain_half_not_obtain_one_l185_185440


namespace other_root_l185_185951

/-- Given the quadratic equation x^2 - 3x + k = 0 has one root as 1, 
    prove that the other root is 2. -/
theorem other_root (k : ℝ) (h : 1^2 - 3 * 1 + k = 0) : 
  2^2 - 3 * 2 + k = 0 := 
by 
  sorry

end other_root_l185_185951


namespace function_properties_l185_185682

theorem function_properties
  (f : ℝ → ℝ)
  (h1 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : ∀ x, f (x - t) = f (x + t)) 
  (h3_even : ∀ x, f (-x) = f x)
  (h3_decreasing : ∀ x1 x2, x1 < x2 ∧ x2 < 0 → f x1 > f x2)
  (h3_at_neg2 : f (-2) = 0)
  (h4_odd : ∀ x, f (-x) = -f x) : 
  ((∀ x1 x2, x1 < x2 → f x1 > f x2) ∧
   (¬∀ x, (f x > 0) ↔ (-2 < x ∧ x < 2)) ∧
   (∀ x, f (x) * f (|x|) = - f (-x) * f |x|) ∧
   (¬∀ x, f (x) = f (x + 2 * t))) :=
by 
  sorry

end function_properties_l185_185682


namespace domain_of_sqrt_fun_l185_185595

theorem domain_of_sqrt_fun : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 7 → 7 + 6 * x - x^2 ≥ 0) :=
sorry

end domain_of_sqrt_fun_l185_185595


namespace problem_Ashwin_Sah_l185_185917

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end problem_Ashwin_Sah_l185_185917


namespace smallest_t_for_sin_theta_circle_l185_185857

theorem smallest_t_for_sin_theta_circle :
  ∃ t > 0, ∀ θ, 0 ≤ θ ∧ θ ≤ t → sin θ = r ↔ (r, θ) completes_circle ∧ t = π :=
by
  sorry

end smallest_t_for_sin_theta_circle_l185_185857


namespace height_of_taller_tree_l185_185602

-- Define the conditions as hypotheses:
variables (h₁ h₂ : ℝ)
-- The top of one tree is 24 feet higher than the top of another tree
variables (h_difference : h₁ = h₂ + 24)
-- The heights of the two trees are in the ratio 2:3
variables (h_ratio : h₂ / h₁ = 2 / 3)

theorem height_of_taller_tree : h₁ = 72 :=
by
  -- This is the place where the solution steps would be applied
  sorry

end height_of_taller_tree_l185_185602


namespace roots_interlaced_l185_185950

variable {α : Type*} [LinearOrderedField α]
variables {f g : α → α}

theorem roots_interlaced
    (x1 x2 x3 x4 : α)
    (h1 : x1 < x2) (h2 : x3 < x4)
    (hfx1 : f x1 = 0) (hfx2 : f x2 = 0)
    (hfx_distinct : x1 ≠ x2)
    (hgx3 : g x3 = 0) (hgx4 : g x4 = 0)
    (hgx_distinct : x3 ≠ x4)
    (hgx1_ne_0 : g x1 ≠ 0) (hgx2_ne_0 : g x2 ≠ 0)
    (hgx1_gx2_lt_0 : g x1 * g x2 < 0) :
    (x1 < x3 ∧ x3 < x2 ∧ x2 < x4) ∨ (x3 < x1 ∧ x1 < x4 ∧ x4 < x2) :=
sorry

end roots_interlaced_l185_185950


namespace length_of_escalator_l185_185307

-- Given conditions
def escalator_speed : ℝ := 12 -- ft/sec
def person_speed : ℝ := 8 -- ft/sec
def time : ℝ := 8 -- seconds

-- Length of the escalator
def length : ℝ := 160 -- feet

-- Theorem stating the length of the escalator given the conditions
theorem length_of_escalator
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 8)
  (h3 : time = 8)
  (combined_speed := escalator_speed + person_speed) :
  combined_speed * time = length :=
by
  -- Here the proof would go, but it's omitted as per instructions
  sorry

end length_of_escalator_l185_185307


namespace wrongly_read_number_l185_185272

theorem wrongly_read_number (initial_avg correct_avg n wrong_correct_sum : ℝ) : 
  initial_avg = 23 ∧ correct_avg = 24 ∧ n = 10 ∧ wrong_correct_sum = 36
  → ∃ (X : ℝ), 36 - X = 10 ∧ X = 26 :=
by
  intro h
  sorry

end wrongly_read_number_l185_185272


namespace max_value_l185_185038

-- Define the vector types
structure Vector2 where
  x : ℝ
  y : ℝ

-- Define the properties given in the problem
def a_is_unit_vector (a : Vector2) : Prop :=
  a.x^2 + a.y^2 = 1

def a_plus_b (a b : Vector2) : Prop :=
  a.x + b.x = 3 ∧ a.y + b.y = 4

-- Define dot product for the vectors
def dot_product (a b : Vector2) : ℝ :=
  a.x * b.x + a.y * b.y

-- The theorem statement
theorem max_value (a b : Vector2) (h1 : a_is_unit_vector a) (h2 : a_plus_b a b) :
  ∃ m, m = 5 ∧ ∀ c : ℝ, |1 + dot_product a b| ≤ m :=
  sorry

end max_value_l185_185038


namespace find_unknown_number_l185_185513

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l185_185513


namespace MrKozelGarden_l185_185221

theorem MrKozelGarden :
  ∀ (x y : ℕ), 
  (y = 3 * x + 1) ∧ (y = 4 * (x - 1)) → (x = 5 ∧ y = 16) := 
by
  intros x y h
  sorry

end MrKozelGarden_l185_185221


namespace largest_perimeter_regular_polygons_l185_185739

theorem largest_perimeter_regular_polygons :
  ∃ (p q r : ℕ), 
    (p ≥ 3 ∧ q ≥ 3 ∧ r >= 3) ∧
    (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (180 * (p - 2)/p + 180 * (q - 2)/q + 180 * (r - 2)/r = 360) ∧
    ((p + q + r - 6) = 9) :=
sorry

end largest_perimeter_regular_polygons_l185_185739


namespace articles_produced_l185_185693

theorem articles_produced (a b c d f p q r g : ℕ) :
  (a * b * c = d) → 
  ((p * q * r * d * g) / (a * b * c * f) = pqr * d * g / (abc * f)) :=
by
  sorry

end articles_produced_l185_185693


namespace alcohol_percentage_correct_in_mixed_solution_l185_185750

-- Define the ratios of alcohol to water
def ratio_A : ℚ := 21 / 25
def ratio_B : ℚ := 2 / 5

-- Define the mixing ratio of solutions A and B
def mix_ratio_A : ℚ := 5 / 11
def mix_ratio_B : ℚ := 6 / 11

-- Define the function to compute the percentage of alcohol in the mixed solution
def alcohol_percentage_mixed : ℚ := 
  (mix_ratio_A * ratio_A + mix_ratio_B * ratio_B) * 100

-- The theorem to be proven
theorem alcohol_percentage_correct_in_mixed_solution : 
  alcohol_percentage_mixed = 60 :=
by
  sorry

end alcohol_percentage_correct_in_mixed_solution_l185_185750


namespace average_waiting_time_l185_185471

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l185_185471


namespace brother_to_madeline_ratio_l185_185254

theorem brother_to_madeline_ratio (M B T : ℕ) (hM : M = 48) (hT : T = 72) (hSum : M + B = T) : B / M = 1 / 2 := by
  sorry

end brother_to_madeline_ratio_l185_185254


namespace ratio_of_areas_l185_185135

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l185_185135


namespace fraction_given_to_cousin_l185_185485

theorem fraction_given_to_cousin
  (initial_candies : ℕ)
  (brother_share sister_share : ℕ)
  (eaten_candies left_candies : ℕ)
  (remaining_candies : ℕ)
  (given_to_cousin : ℕ)
  (fraction : ℚ)
  (h1 : initial_candies = 50)
  (h2 : brother_share = 5)
  (h3 : sister_share = 5)
  (h4 : eaten_candies = 12)
  (h5 : left_candies = 18)
  (h6 : initial_candies - brother_share - sister_share = remaining_candies)
  (h7 : remaining_candies - given_to_cousin - eaten_candies = left_candies)
  (h8 : fraction = (given_to_cousin : ℚ) / (remaining_candies : ℚ))
  : fraction = 1 / 4 := 
sorry

end fraction_given_to_cousin_l185_185485


namespace abs_neg_three_halves_l185_185094

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l185_185094


namespace ceil_minus_eq_zero_l185_185136

theorem ceil_minus_eq_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 :=
sorry

end ceil_minus_eq_zero_l185_185136


namespace total_money_is_correct_l185_185691

-- Define conditions as constants
def numChocolateCookies : ℕ := 220
def pricePerChocolateCookie : ℕ := 1
def numVanillaCookies : ℕ := 70
def pricePerVanillaCookie : ℕ := 2

-- Total money made from selling chocolate cookies
def moneyFromChocolateCookies : ℕ := numChocolateCookies * pricePerChocolateCookie

-- Total money made from selling vanilla cookies
def moneyFromVanillaCookies : ℕ := numVanillaCookies * pricePerVanillaCookie

-- Total money made from selling all cookies
def totalMoneyMade : ℕ := moneyFromChocolateCookies + moneyFromVanillaCookies

-- The statement to prove, with the expected result
theorem total_money_is_correct : totalMoneyMade = 360 := by
  sorry

end total_money_is_correct_l185_185691


namespace farm_field_ploughing_l185_185426

theorem farm_field_ploughing (A D : ℕ) 
  (h1 : ∀ farmerA_initial_capacity: ℕ, farmerA_initial_capacity = 120)
  (h2 : ∀ farmerB_initial_capacity: ℕ, farmerB_initial_capacity = 100)
  (h3 : ∀ farmerA_adjustment: ℕ, farmerA_adjustment = 10)
  (h4 : ∀ farmerA_reduced_capacity: ℕ, farmerA_reduced_capacity = farmerA_initial_capacity - (farmerA_adjustment * farmerA_initial_capacity / 100))
  (h5 : ∀ farmerB_reduced_capacity: ℕ, farmerB_reduced_capacity = 90)
  (h6 : ∀ extra_days: ℕ, extra_days = 3)
  (h7 : ∀ remaining_hectares: ℕ, remaining_hectares = 60)
  (h8 : ∀ initial_combined_effort: ℕ, initial_combined_effort = (farmerA_initial_capacity + farmerB_initial_capacity) * D)
  (h9 : ∀ total_combined_effort: ℕ, total_combined_effort = (farmerA_reduced_capacity + farmerB_reduced_capacity) * (D + extra_days))
  (h10 : ∀ area_covered: ℕ, area_covered = total_combined_effort + remaining_hectares)
  : initial_combined_effort = A ∧ D = 30 ∧ A = 6600 :=
by
  sorry

end farm_field_ploughing_l185_185426


namespace max_value_at_x0_l185_185220

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_at_x0 {x0 : ℝ} (h : ∃ x0, ∀ x, f x ≤ f x0) : 
  f x0 = x0 :=
sorry

end max_value_at_x0_l185_185220


namespace plane_through_Ox_and_point_plane_parallel_Oz_and_points_l185_185931

-- Definitions for first plane problem
def plane1_through_Ox_axis (y z : ℝ) : Prop := 3 * y + 2 * z = 0

-- Definitions for second plane problem
def plane2_parallel_Oz (x y : ℝ) : Prop := x + 3 * y - 1 = 0

theorem plane_through_Ox_and_point : plane1_through_Ox_axis 2 (-3) := 
by {
  -- Hint: Prove that substituting y = 2 and z = -3 in the equation results in LHS equals RHS.
  -- proof
  sorry 
}

theorem plane_parallel_Oz_and_points : 
  plane2_parallel_Oz 1 0 ∧ plane2_parallel_Oz (-2) 1 :=
by {
  -- Hint: Prove that substituting the points (1, 0) and (-2, 1) in the equation results in LHS equals RHS.
  -- proof
  sorry
}

end plane_through_Ox_and_point_plane_parallel_Oz_and_points_l185_185931


namespace geometric_sequence_common_ratio_l185_185273

theorem geometric_sequence_common_ratio
  (q a_1 : ℝ)
  (h1: a_1 * q = 1)
  (h2: a_1 + a_1 * q^2 = -2) :
  q = -1 :=
by
  sorry

end geometric_sequence_common_ratio_l185_185273


namespace committee_count_l185_185534

theorem committee_count (students teachers : ℕ) (committee_size : ℕ) 
  (h_students : students = 11) (h_teachers : teachers = 3) 
  (h_committee_size : committee_size = 8) : 
  ∑ (k : ℕ) in finset.range committee_size.succ, (nat.choose (students + teachers) committee_size) - (nat.choose students committee_size) = 2838 := 
by 
  sorry

end committee_count_l185_185534


namespace measure_of_angle_C_l185_185253

theorem measure_of_angle_C (m l : ℝ) (angle_A angle_B angle_D angle_C : ℝ)
  (h_parallel : l = m)
  (h_angle_A : angle_A = 130)
  (h_angle_B : angle_B = 140)
  (h_angle_D : angle_D = 100) :
  angle_C = 90 :=
by
  sorry

end measure_of_angle_C_l185_185253


namespace number_of_workers_in_second_group_l185_185567

theorem number_of_workers_in_second_group (w₁ w₂ d₁ d₂ : ℕ) (total_wages₁ total_wages₂ : ℝ) (daily_wage : ℝ) :
  w₁ = 15 ∧ d₁ = 6 ∧ total_wages₁ = 9450 ∧ 
  w₂ * d₂ * daily_wage = total_wages₂ ∧ d₂ = 5 ∧ total_wages₂ = 9975 ∧ 
  daily_wage = 105 
  → w₂ = 19 :=
by
  sorry

end number_of_workers_in_second_group_l185_185567


namespace smallest_pos_d_l185_185434

theorem smallest_pos_d (d : ℕ) (h : d > 0) (hd : ∃ k : ℕ, 3150 * d = k * k) : d = 14 := 
by 
  sorry

end smallest_pos_d_l185_185434


namespace ed_lighter_than_al_l185_185639

theorem ed_lighter_than_al :
  let Al := Ben + 25
  let Ben := Carl - 16
  let Ed := 146
  let Carl := 175
  Al - Ed = 38 :=
by
  sorry

end ed_lighter_than_al_l185_185639


namespace find_x_when_y4_l185_185592

theorem find_x_when_y4 
  (k : ℝ) 
  (h_var : ∀ y : ℝ, ∃ x : ℝ, x = k * y^2)
  (h_initial : ∃ x : ℝ, x = 6 ∧ 1 = k) :
  ∃ x : ℝ, x = 96 :=
by 
  sorry

end find_x_when_y4_l185_185592


namespace three_digit_identical_divisible_by_37_l185_185723

theorem three_digit_identical_divisible_by_37 (A : ℕ) (h : A ≤ 9) : 37 ∣ (111 * A) :=
sorry

end three_digit_identical_divisible_by_37_l185_185723


namespace same_terminal_side_eq_l185_185013

theorem same_terminal_side_eq (α : ℝ) : 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 3) ↔ α = 5 * Real.pi / 3 :=
by sorry

end same_terminal_side_eq_l185_185013


namespace family_gathering_total_people_l185_185620

theorem family_gathering_total_people (P : ℕ) 
  (h1 : P / 2 = 10) : 
  P = 20 := by
  sorry

end family_gathering_total_people_l185_185620


namespace gcd_459_357_is_51_l185_185414

-- Define the problem statement
theorem gcd_459_357_is_51 : Nat.gcd 459 357 = 51 :=
by
  -- Proof here
  sorry

end gcd_459_357_is_51_l185_185414


namespace range_of_BD_l185_185293

-- Define the types of points and triangle
variables {α : Type*} [MetricSpace α]

-- Hypothesis: AD is the median of triangle ABC
-- Definition of lengths AB, AC, and that BD = CD.
def isMedianOnBC (A B C D : α) : Prop :=
  dist A B = 5 ∧ dist A C = 7 ∧ dist B D = dist C D

-- The theorem to be proven
theorem range_of_BD {A B C D : α} (h : isMedianOnBC A B C D) : 
  1 < dist B D ∧ dist B D < 6 :=
by
  sorry

end range_of_BD_l185_185293


namespace avg_waiting_time_is_1_point_2_minutes_l185_185474

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l185_185474


namespace prime_factor_of_difference_l185_185711

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA9 : A ≤ 9) (hC : 1 ≤ C) (hC9 : C ≤ 9) (hA_ne_C : A ≠ C) :
  ∃ p : ℕ, Prime p ∧ p = 3 ∧ p ∣ 3 * (100 * A + 10 * B + C - (100 * C + 10 * B + A)) := by
  sorry

end prime_factor_of_difference_l185_185711


namespace max_amount_paul_received_l185_185846

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end max_amount_paul_received_l185_185846


namespace wood_blocks_after_days_l185_185587

-- Defining the known conditions
def blocks_per_tree : Nat := 3
def trees_per_day : Nat := 2
def days : Nat := 5

-- Stating the theorem to prove the total number of blocks of wood after 5 days
theorem wood_blocks_after_days : blocks_per_tree * trees_per_day * days = 30 :=
by
  sorry

end wood_blocks_after_days_l185_185587


namespace oil_truck_radius_l185_185632

/-- 
A full stationary oil tank that is a right circular cylinder has a radius of 100 feet 
and a height of 25 feet. Oil is pumped from the stationary tank to an oil truck that 
has a tank that is a right circular cylinder. The oil level dropped 0.025 feet in the stationary tank. 
The oil truck's tank has a height of 10 feet. The radius of the oil truck's tank is 5 feet. 
--/
theorem oil_truck_radius (r_stationary : ℝ) (h_stationary : ℝ) (h_truck : ℝ) 
  (Δh : ℝ) (r_truck : ℝ) 
  (h_stationary_pos : 0 < h_stationary) (h_truck_pos : 0 < h_truck) (r_stationary_pos : 0 < r_stationary) :
  r_stationary = 100 → h_stationary = 25 → Δh = 0.025 → h_truck = 10 → r_truck = 5 → 
  π * (r_stationary ^ 2) * Δh = π * (r_truck ^ 2) * h_truck :=
by 
  -- Use the conditions and perform algebra to show the equality.
  sorry

end oil_truck_radius_l185_185632


namespace correct_option_for_sentence_completion_l185_185720

-- Define the mathematical formalization of the problem
def sentence_completion_problem : String × (List String) := 
    ("One of the most important questions they had to consider was _ of public health.", 
     ["what", "this", "that", "which"])

-- Define the correct answer
def correct_answer : String := "that"

-- The formal statement of the problem in Lean 4
theorem correct_option_for_sentence_completion 
    (problem : String × (List String)) (answer : String) :
    answer = "that" :=
by
  sorry  -- Proof to be completed

end correct_option_for_sentence_completion_l185_185720


namespace find_symmetric_sequence_l185_185031

noncomputable def symmetric_sequence (b : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → b k = b (n - k + 1)

noncomputable def arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, b 2 = b 1 + d ∧ b 3 = b 2 + d ∧ b 4 = b 3 + d

theorem find_symmetric_sequence :
  ∃ b : ℕ → ℤ, symmetric_sequence b 7 ∧ arithmetic_sequence b ∧ b 1 = 2 ∧ b 2 + b 4 = 16 ∧
  (b 1 = 2 ∧ b 2 = 5 ∧ b 3 = 8 ∧ b 4 = 11 ∧ b 5 = 8 ∧ b 6 = 5 ∧ b 7 = 2) :=
by {
  sorry
}

end find_symmetric_sequence_l185_185031


namespace solve_for_k_l185_185696

theorem solve_for_k (k : ℕ) (h : 16 / k = 4) : k = 4 :=
sorry

end solve_for_k_l185_185696


namespace inequality_proof_l185_185722

theorem inequality_proof (x y : ℝ) : 
  -1 / 2 ≤ (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ∧
  (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ≤ 1 / 2 :=
sorry

end inequality_proof_l185_185722


namespace functional_relationship_l185_185944

variable (x y k1 k2 : ℝ)

axiom h1 : y = k1 * x + k2 / (x - 2)
axiom h2 : (y = -1) ↔ (x = 1)
axiom h3 : (y = 5) ↔ (x = 3)

theorem functional_relationship :
  (∀ x y, y = k1 * x + k2 / (x - 2) ∧
    ((x = 1) → y = -1) ∧
    ((x = 3) → y = 5) → y = x + 2 / (x - 2)) :=
by
  sorry

end functional_relationship_l185_185944


namespace adult_tickets_sold_l185_185880

theorem adult_tickets_sold (A C : ℕ) (h1 : A + C = 85) (h2 : 5 * A + 2 * C = 275) : A = 35 := by
  sorry

end adult_tickets_sold_l185_185880


namespace ranch_cows_variance_l185_185451

variable (n : ℕ)
variable (p : ℝ)

-- Definition of the variance of a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem ranch_cows_variance : 
  binomial_variance 10 0.02 = 0.196 :=
by
  sorry

end ranch_cows_variance_l185_185451


namespace ratio_of_volumes_l185_185864

noncomputable def volumeSphere (p : ℝ) : ℝ := (4/3) * Real.pi * (p^3)

noncomputable def volumeHemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * (3*p)^3

theorem ratio_of_volumes (p : ℝ) (hp : p > 0) : volumeSphere p / volumeHemisphere p = 2 / 27 :=
by
  sorry

end ratio_of_volumes_l185_185864


namespace AlissaMorePresents_l185_185325

/-- Ethan has 31 presents -/
def EthanPresents : ℕ := 31

/-- Alissa has 53 presents -/
def AlissaPresents : ℕ := 53

/-- How many more presents does Alissa have than Ethan? -/
theorem AlissaMorePresents : AlissaPresents - EthanPresents = 22 := by
  -- Place the proof here
  sorry

end AlissaMorePresents_l185_185325


namespace possible_values_of_a_and_b_l185_185395

theorem possible_values_of_a_and_b (a b : ℕ) : 
  (a = 22 ∨ a = 33 ∨ a = 40 ∨ a = 42) ∧ 
  (b = 21 ∨ b = 10 ∨ b = 3 ∨ b = 1) ∧ 
  (a % (b + 1) = 0) ∧ (43 % (a + b) = 0) :=
sorry

end possible_values_of_a_and_b_l185_185395


namespace correct_operation_l185_185148

theorem correct_operation : 
  ¬(3 * x^2 + 2 * x^2 = 6 * x^4) ∧ 
  ¬((-2 * x^2)^3 = -6 * x^6) ∧ 
  ¬(x^3 * x^2 = x^6) ∧ 
  (-6 * x^2 * y^3 / (2 * x^2 * y^2) = -3 * y) :=
by
  sorry

end correct_operation_l185_185148


namespace eighth_term_of_arithmetic_sequence_l185_185202

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℤ),
  (a 1 = 11) →
  (a 2 = 8) →
  (a 3 = 5) →
  (∃ (d : ℤ), ∀ n, a (n + 1) = a n + d) →
  a 8 = -10 :=
by
  intros a h1 h2 h3 arith
  sorry

end eighth_term_of_arithmetic_sequence_l185_185202


namespace inequality_holds_for_all_xyz_in_unit_interval_l185_185264

theorem inequality_holds_for_all_xyz_in_unit_interval :
  ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) → 
  (x / (y + z + 1) + y / (z + x + 1) + z / (x + y + 1) ≤ 1 - (1 - x) * (1 - y) * (1 - z)) :=
by
  intros x y z hx hy hz
  sorry

end inequality_holds_for_all_xyz_in_unit_interval_l185_185264


namespace max_cubes_fit_l185_185284

-- Define the conditions
def box_volume (length : ℕ) (width : ℕ) (height : ℕ) : ℕ := length * width * height
def cube_volume : ℕ := 27
def total_cubes (V_box : ℕ) (V_cube : ℕ) : ℕ := V_box / V_cube

-- Statement of the problem
theorem max_cubes_fit (length width height : ℕ) (V_box : ℕ) (V_cube q : ℕ) :
  length = 8 → width = 9 → height = 12 → V_box = box_volume length width height →
  V_cube = cube_volume → q = total_cubes V_box V_cube → q = 32 :=
by sorry

end max_cubes_fit_l185_185284


namespace abs_neg_three_halves_l185_185088

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l185_185088


namespace distance_between_intersections_is_sqrt3_l185_185229

noncomputable def intersection_distance : ℝ :=
  let C1_polar := (θ : ℝ) → θ = (2 * Real.pi / 3)
  let C2_standard := (x y : ℝ) → (x + Real.sqrt 3)^2 + (y + 2)^2 = 1
  let C3 := (θ : ℝ) → θ = (Real.pi / 3) 
  let C3_cartesian := (x y : ℝ) → y = Real.sqrt 3 * x
  let center := (-Real.sqrt 3, -2)
  let dist_to_C3 := abs (-3 + 2) / 2
  2 * Real.sqrt (1 - (dist_to_C3)^2)

theorem distance_between_intersections_is_sqrt3:
  intersection_distance = Real.sqrt 3 := by
  sorry

end distance_between_intersections_is_sqrt3_l185_185229


namespace total_beads_correct_l185_185464

def purple_beads : ℕ := 7
def blue_beads : ℕ := 2 * purple_beads
def green_beads : ℕ := blue_beads + 11
def total_beads : ℕ := purple_beads + blue_beads + green_beads

theorem total_beads_correct : total_beads = 46 := 
by
  have h1 : purple_beads = 7 := rfl
  have h2 : blue_beads = 2 * 7 := rfl
  have h3 : green_beads = 14 + 11 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_beads_correct_l185_185464


namespace race_distance_l185_185223

-- Definitions for the conditions
def A_time : ℕ := 20
def B_time : ℕ := 25
def A_beats_B_by : ℕ := 14

-- Definition of the function to calculate whether the total distance D is correct
def total_distance : ℕ := 56

-- The theorem statement without proof
theorem race_distance (D : ℕ) (A_time B_time A_beats_B_by : ℕ)
  (hA : A_time = 20)
  (hB : B_time = 25)
  (hAB : A_beats_B_by = 14)
  (h_eq : (D / A_time) * B_time = D + A_beats_B_by) : 
  D = total_distance :=
sorry

end race_distance_l185_185223


namespace sin_C_eq_sin_A_minus_B_eq_l185_185818

open Real

-- Problem 1
theorem sin_C_eq (A B C : ℝ) (a b c : ℝ)
  (hB : B = π / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) :
  sin C = (sqrt 3 + 3 * sqrt 2) / 6 :=
sorry

-- Problem 2
theorem sin_A_minus_B_eq (A B C : ℝ) (a b c : ℝ)
  (h_cosC : cos C = 2 / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) 
  (hA_acute : 0 < A ∧ A < π / 2)
  (hB_acute : 0 < B ∧ B < π / 2) :
  sin (A - B) = -sqrt 5 / 3 :=
sorry

end sin_C_eq_sin_A_minus_B_eq_l185_185818


namespace common_value_of_7a_and_2b_l185_185697

variable (a b : ℝ)

theorem common_value_of_7a_and_2b (h1 : 7 * a = 2 * b) (h2 : 42 * a * b = 674.9999999999999) :
  7 * a = 15 :=
by
  -- This place will contain the proof steps
  sorry

end common_value_of_7a_and_2b_l185_185697


namespace ratio_of_areas_GHI_to_JKL_l185_185120

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l185_185120


namespace binom_1300_2_eq_l185_185316

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l185_185316


namespace find_a_l185_185368

theorem find_a (a : ℝ) :
  {x : ℝ | (x + a) / ((x + 1) * (x + 3)) > 0} = {x : ℝ | x > -3 ∧ x ≠ -1} →
  a = 1 := 
by sorry

end find_a_l185_185368


namespace loom_weaving_rate_l185_185768

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) 
    (h1 : total_cloth = 25) (h2 : total_time = 195.3125) : 
    total_cloth / total_time = 0.128 :=
sorry

end loom_weaving_rate_l185_185768


namespace ratio_of_areas_l185_185133

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l185_185133


namespace ceil_sqrt_fraction_eq_neg2_l185_185175

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end ceil_sqrt_fraction_eq_neg2_l185_185175


namespace binom_1300_2_eq_844350_l185_185319

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l185_185319


namespace pos_sol_eq_one_l185_185072

theorem pos_sol_eq_one (n : ℕ) (hn : 1 < n) :
  ∀ x : ℝ, 0 < x → (x ^ n - n * x + n - 1 = 0) → x = 1 := by
  -- The proof goes here
  sorry

end pos_sol_eq_one_l185_185072


namespace sparrows_on_fence_l185_185380

-- Define the number of sparrows initially on the fence
def initial_sparrows : ℕ := 2

-- Define the number of sparrows that joined later
def additional_sparrows : ℕ := 4

-- Define the number of sparrows that flew away
def sparrows_flew_away : ℕ := 3

-- Define the final number of sparrows on the fence
def final_sparrows : ℕ := initial_sparrows + additional_sparrows - sparrows_flew_away

-- Prove that the final number of sparrows on the fence is 3
theorem sparrows_on_fence : final_sparrows = 3 := by
  sorry

end sparrows_on_fence_l185_185380


namespace number_of_keepers_l185_185153

theorem number_of_keepers (k : ℕ)
  (hens : ℕ := 50)
  (goats : ℕ := 45)
  (camels : ℕ := 8)
  (hen_feet : ℕ := 2)
  (goat_feet : ℕ := 4)
  (camel_feet : ℕ := 4)
  (keeper_feet : ℕ := 2)
  (feet_more_than_heads : ℕ := 224)
  (total_heads : ℕ := hens + goats + camels + k)
  (total_feet : ℕ := (hens * hen_feet) + (goats * goat_feet) + (camels * camel_feet) + (k * keeper_feet)):
  total_feet = total_heads + feet_more_than_heads → k = 15 :=
by
  sorry

end number_of_keepers_l185_185153


namespace total_points_scored_l185_185413

theorem total_points_scored :
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  a + b + c + d + e + f + g + h = 54 :=
by
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  sorry

end total_points_scored_l185_185413


namespace min_value_512_l185_185715

noncomputable def min_value (a b c d e f g h : ℝ) : ℝ :=
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2

theorem min_value_512 
  (a b c d e f g h : ℝ)
  (H1 : a * b * c * d = 8)
  (H2 : e * f * g * h = 16) : 
  ∃ (min_val : ℝ), min_val = 512 ∧ min_value a b c d e f g h = min_val :=
sorry

end min_value_512_l185_185715


namespace evaporation_period_l185_185443

theorem evaporation_period
  (total_water : ℕ)
  (daily_evaporation_rate : ℝ)
  (percentage_evaporated : ℝ)
  (evaporation_period_days : ℕ)
  (h_total_water : total_water = 10)
  (h_daily_evaporation_rate : daily_evaporation_rate = 0.006)
  (h_percentage_evaporated : percentage_evaporated = 0.03)
  (h_evaporation_period_days : evaporation_period_days = 50):
  (percentage_evaporated * total_water) / daily_evaporation_rate = evaporation_period_days := by
  sorry

end evaporation_period_l185_185443


namespace discount_given_l185_185393

variables (initial_money : ℕ) (extra_fraction : ℕ) (additional_money_needed : ℕ)
variables (total_with_discount : ℕ) (discount_amount : ℕ)

def total_without_discount (initial_money : ℕ) (extra_fraction : ℕ) : ℕ :=
  initial_money + extra_fraction

def discount (initial_money : ℕ) (total_without_discount : ℕ) (total_with_discount : ℕ) : ℕ :=
  total_without_discount - total_with_discount

def discount_percentage (discount_amount : ℕ) (total_without_discount : ℕ) : ℚ :=
  (discount_amount : ℚ) / (total_without_discount : ℚ) * 100

theorem discount_given 
  (initial_money : ℕ := 500)
  (extra_fraction : ℕ := 200)
  (additional_money_needed : ℕ := 95)
  (total_without_discount₀ : ℕ := total_without_discount initial_money extra_fraction)
  (total_with_discount₀ : ℕ := initial_money + additional_money_needed)
  (discount_amount₀ : ℕ := discount initial_money total_without_discount₀ total_with_discount₀)
  : discount_percentage discount_amount₀ total_without_discount₀ = 15 :=
by sorry

end discount_given_l185_185393


namespace binomial_1300_2_eq_844350_l185_185311

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l185_185311


namespace fifteen_percent_of_x_is_ninety_l185_185517

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l185_185517


namespace gift_sequences_count_l185_185583

def num_students : ℕ := 11
def num_meetings : ℕ := 4
def sequences : ℕ := num_students ^ num_meetings

theorem gift_sequences_count : sequences = 14641 := by
  sorry

end gift_sequences_count_l185_185583


namespace smaller_of_two_numbers_in_ratio_l185_185610

theorem smaller_of_two_numbers_in_ratio (x y a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : x / y = a / b) (h3 : x + y = c) : 
  min x y = (a * c) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_in_ratio_l185_185610


namespace jamies_class_girls_count_l185_185375

theorem jamies_class_girls_count 
  (g b : ℕ)
  (h_ratio : 4 * g = 3 * b)
  (h_total : g + b = 35) 
  : g = 15 := 
by 
  sorry 

end jamies_class_girls_count_l185_185375


namespace fifteen_percent_of_x_is_ninety_l185_185515

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l185_185515


namespace quadrant_of_points_l185_185360

theorem quadrant_of_points (x y : ℝ) (h : |3 * x + 2| + |2 * y - 1| = 0) : 
  ((x < 0) ∧ (y > 0) ∧ (x + 1 > 0) ∧ (y - 2 < 0)) :=
by
  sorry

end quadrant_of_points_l185_185360


namespace balloon_permutations_l185_185808

theorem balloon_permutations : 
  (Nat.factorial 7 / 
  ((Nat.factorial 1) * 
  (Nat.factorial 1) * 
  (Nat.factorial 2) * 
  (Nat.factorial 2) * 
  (Nat.factorial 1))) = 1260 := by
  sorry

end balloon_permutations_l185_185808


namespace avg_waiting_time_waiting_time_equivalence_l185_185479

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l185_185479


namespace x_expression_l185_185838

noncomputable def f (t : ℝ) : ℝ := t / (1 - t)

theorem x_expression {x y : ℝ} (hx : x ≠ 1) (hy : y = f x) : x = y / (1 + y) :=
by {
  sorry
}

end x_expression_l185_185838


namespace three_character_license_plates_l185_185209

theorem three_character_license_plates :
  let consonants := 20
  let vowels := 6
  (consonants * consonants * vowels = 2400) :=
by
  sorry

end three_character_license_plates_l185_185209


namespace find_unknown_number_l185_185512

theorem find_unknown_number (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end find_unknown_number_l185_185512


namespace total_houses_l185_185053

theorem total_houses (houses_one_side : ℕ) (houses_other_side : ℕ) (h1 : houses_one_side = 40) (h2 : houses_other_side = 3 * houses_one_side) : houses_one_side + houses_other_side = 160 :=
by sorry

end total_houses_l185_185053


namespace expression_value_l185_185285

theorem expression_value :
  (35 + 12) ^ 2 - (12 ^ 2 + 35 ^ 2 - 2 * 12 * 35) = 1680 :=
by
  sorry

end expression_value_l185_185285


namespace units_digit_of_calculation_l185_185141

-- Base definitions for units digits of given numbers
def units_digit (n : ℕ) : ℕ := n % 10

-- Main statement to prove
theorem units_digit_of_calculation : 
  units_digit ((25 ^ 3 + 17 ^ 3) * 12 ^ 2) = 2 :=
by
  -- This is where the proof would go, but it's omitted as requested
  sorry

end units_digit_of_calculation_l185_185141


namespace range_of_k_l185_185532

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

def is_ktimes_value_function (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  0 < k ∧ a < b ∧ f a = k * a ∧ f b = k * b

theorem range_of_k (k : ℝ) : (∃ a b : ℝ, is_ktimes_value_function f k a b) ↔ 1 < k ∧ k < 1 + 1 / Real.exp 1 := by
  sorry

end range_of_k_l185_185532


namespace smallest_X_l185_185242

noncomputable def T : ℕ := 1110
noncomputable def X : ℕ := T / 6

theorem smallest_X (hT_digits : (∀ d ∈ T.digits 10, d = 0 ∨ d = 1))
  (hT_positive : T > 0)
  (hT_div_6 : T % 6 = 0) :
  X = 185 := by
  sorry

end smallest_X_l185_185242


namespace compare_2_roses_3_carnations_l185_185346

variable (x y : ℝ)

def condition1 : Prop := 6 * x + 3 * y > 24
def condition2 : Prop := 4 * x + 5 * y < 22

theorem compare_2_roses_3_carnations (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x > 3 * y := sorry

end compare_2_roses_3_carnations_l185_185346


namespace fish_per_black_duck_l185_185334

theorem fish_per_black_duck :
  ∀ (W_d B_d M_d : ℕ) (fish_per_W fish_per_M total_fish : ℕ),
    (fish_per_W = 5) →
    (fish_per_M = 12) →
    (W_d = 3) →
    (B_d = 7) →
    (M_d = 6) →
    (total_fish = 157) →
    (total_fish - (W_d * fish_per_W + M_d * fish_per_M)) = 70 →
    (70 / B_d) = 10 :=
by
  intros W_d B_d M_d fish_per_W fish_per_M total_fish hW hM hW_d hB_d hM_d htotal_fish hcalculation
  sorry

end fish_per_black_duck_l185_185334


namespace exists_polynomial_h_l185_185240

variable {R : Type} [CommRing R] [IsDomain R] [CharZero R]

noncomputable def f (x : R) : ℝ := sorry -- define the polynomial f(x) here
noncomputable def g (x : R) : ℝ := sorry -- define the polynomial g(x) here

theorem exists_polynomial_h (m n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h_mn : m + n > 0)
  (h_fg_squares : ∀ x : ℝ, (∃ k : ℤ, f x = k^2) ↔ (∃ l : ℤ, g x = l^2)) :
  ∃ h : ℝ → ℝ, ∀ x : ℝ, f x * g x = (h x)^2 :=
sorry

end exists_polynomial_h_l185_185240


namespace exists_increasing_or_decreasing_subsequence_l185_185796

theorem exists_increasing_or_decreasing_subsequence (n : ℕ) (a : Fin (n^2 + 1) → ℝ) :
  ∃ (b : Fin (n + 1) → ℝ), (StrictMono b ∨ StrictAnti b) :=
sorry

end exists_increasing_or_decreasing_subsequence_l185_185796


namespace number_of_men_l185_185297

theorem number_of_men (M : ℕ) (h : M * 40 = 20 * 68) : M = 34 :=
by
  sorry

end number_of_men_l185_185297


namespace a_perfect_square_l185_185238

theorem a_perfect_square (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_div : 2 * a * b ∣ a^2 + b^2 - a) : ∃ k : ℕ, a = k^2 := 
sorry

end a_perfect_square_l185_185238


namespace smallest_six_digit_divisible_by_111_l185_185623

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 :=
by {
  sorry
}

end smallest_six_digit_divisible_by_111_l185_185623


namespace intersection_of_sets_l185_185073

open Set Real

theorem intersection_of_sets :
  let A := {x : ℝ | x^2 - 2*x - 3 < 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = sin x}
  A ∩ B = Ioc (-1) 1 := by
  sorry

end intersection_of_sets_l185_185073


namespace pool_maintenance_cost_l185_185062

theorem pool_maintenance_cost 
  {d_cleaning : Nat}
  {cleaning_cost : ℕ}
  {tip_rate : ℝ}
  {d_month : Nat}
  {cleanings_per_month : ℕ}
  {use_chem_freq : ℕ}
  {chem_cost : ℕ}
  {total_cleaning_cost : ℕ}
  {total_chem_cost : ℕ}
  {total_monthly_cost : ℕ} 
  (hc1 : d_cleaning = 3)
  (hc2 : cleaning_cost = 150)
  (hc3 : tip_rate = 0.1)
  (hc4 : d_month = 30)
  (hc5 : cleanings_per_month = d_month / d_cleaning)
  (hc6 : use_chem_freq = 2)
  (hc7 : chem_cost = 200)
  (hc8 : total_cleaning_cost = cleanings_per_month * (cleaning_cost + (cleaning_cost * tip_rate).toNat))
  (hc9 : total_chem_cost = use_chem_freq * chem_cost)
  (hc10 : total_monthly_cost = total_cleaning_cost + total_chem_cost) :
  total_monthly_cost = 2050 :=
by
  sorry

end pool_maintenance_cost_l185_185062


namespace remaining_employees_earn_rate_l185_185379

theorem remaining_employees_earn_rate
  (total_employees : ℕ)
  (employees_12_per_hour : ℕ)
  (employees_14_per_hour : ℕ)
  (total_cost : ℝ)
  (hourly_rate_12 : ℝ)
  (hourly_rate_14 : ℝ)
  (shift_hours : ℝ)
  (remaining_employees : ℕ)
  (remaining_hourly_rate : ℝ) :
  total_employees = 300 →
  employees_12_per_hour = 200 →
  employees_14_per_hour = 40 →
  total_cost = 31840 →
  hourly_rate_12 = 12 →
  hourly_rate_14 = 14 →
  shift_hours = 8 →
  remaining_employees = 60 →
  remaining_hourly_rate = 
    (total_cost - (employees_12_per_hour * hourly_rate_12 * shift_hours) - 
    (employees_14_per_hour * hourly_rate_14 * shift_hours)) / 
    (remaining_employees * shift_hours) →
  remaining_hourly_rate = 17 :=
by
  sorry

end remaining_employees_earn_rate_l185_185379


namespace min_dist_on_circle_l185_185599

theorem min_dist_on_circle :
  let P (θ : ℝ) := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)
  let M := (0, 2)
  ∃ θ_min : ℝ, 
    (∀ θ : ℝ, 
      let dist (P : ℝ × ℝ) (M : ℝ × ℝ) := Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)
      dist (P θ) M ≥ dist (P θ_min) M) ∧ 
    dist (P θ_min) M = Real.sqrt 5 - 1 := sorry

end min_dist_on_circle_l185_185599


namespace reciprocal_pair_c_l185_185641

def is_reciprocal (a b : ℝ) : Prop :=
  a * b = 1

theorem reciprocal_pair_c :
  is_reciprocal (-2) (-1/2) :=
by sorry

end reciprocal_pair_c_l185_185641


namespace triangle_perimeter_l185_185598

-- Given conditions
def inradius : ℝ := 2.5
def area : ℝ := 40

-- The formula relating inradius, area, and perimeter
def perimeter_formula (r a p : ℝ) : Prop := a = r * p / 2

-- Prove the perimeter p of the triangle
theorem triangle_perimeter : ∃ (p : ℝ), perimeter_formula inradius area p ∧ p = 32 := by
  sorry

end triangle_perimeter_l185_185598


namespace array_sum_remainder_l185_185441

def entry_value (r c : ℕ) : ℚ :=
  (1 / (2 * 1013) ^ r) * (1 / 1013 ^ c)

def array_sum : ℚ :=
  (1 / (2 * 1013 - 1)) * (1 / (1013 - 1))

def m : ℤ := 1
def n : ℤ := 2046300
def mn_sum : ℤ := m + n

theorem array_sum_remainder :
  (mn_sum % 1013) = 442 :=
by
  sorry

end array_sum_remainder_l185_185441


namespace exists_isodynamic_points_l185_185576

theorem exists_isodynamic_points (ABC : Triangle)
  (h_non_equilateral : ¬ (ABC.isEquilateral)) :
  ∃ (P Q : IsodynamicPoint ABC), 
    P.insideCircumcircle ∧ ¬ Q.insideCircumcircle ∧
    (equilateral (ABC.projections P)) ∧ (equilateral (ABC.projections Q)) :=
by
  sorry

end exists_isodynamic_points_l185_185576


namespace hoseok_divides_number_l185_185143

theorem hoseok_divides_number (x : ℕ) (h : x / 6 = 11) : x = 66 := by
  sorry

end hoseok_divides_number_l185_185143


namespace Jeremy_songs_l185_185975

theorem Jeremy_songs (songs_yesterday : ℕ) (songs_difference : ℕ) (songs_today : ℕ) (total_songs : ℕ) :
  songs_yesterday = 9 ∧ songs_difference = 5 ∧ songs_today = songs_yesterday + songs_difference ∧ 
  total_songs = songs_yesterday + songs_today → total_songs = 23 :=
by
  intros h
  sorry

end Jeremy_songs_l185_185975


namespace min_value_of_reciprocals_l185_185069

theorem min_value_of_reciprocals (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (1 / m + 1 / n) = 2 :=
sorry

end min_value_of_reciprocals_l185_185069


namespace number_less_than_neg_one_is_neg_two_l185_185450

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end number_less_than_neg_one_is_neg_two_l185_185450


namespace simplify_expression_l185_185268

theorem simplify_expression (x : ℤ) (h1 : 2 * (x - 1) < x + 1) (h2 : 5 * x + 3 ≥ 2 * x) :
  (x = 2) → (2 / (x^2 + x) / (1 - (x - 1) / (x^2 - 1)) = 1 / 2) :=
by
  sorry

end simplify_expression_l185_185268


namespace sum_of_midpoints_l185_185737

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
by
  sorry

end sum_of_midpoints_l185_185737


namespace norris_savings_l185_185844

theorem norris_savings:
  ∀ (N : ℕ), 
  (29 + 25 + N = 85) → N = 31 :=
by
  intros N h
  sorry

end norris_savings_l185_185844


namespace total_beads_correct_l185_185465

def purple_beads : ℕ := 7
def blue_beads : ℕ := 2 * purple_beads
def green_beads : ℕ := blue_beads + 11
def total_beads : ℕ := purple_beads + blue_beads + green_beads

theorem total_beads_correct : total_beads = 46 := 
by
  have h1 : purple_beads = 7 := rfl
  have h2 : blue_beads = 2 * 7 := rfl
  have h3 : green_beads = 14 + 11 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_beads_correct_l185_185465


namespace area_difference_8_7_area_difference_9_8_l185_185009

-- Define the side lengths of the tablets
def side_length_7 : ℕ := 7
def side_length_8 : ℕ := 8
def side_length_9 : ℕ := 9

-- Define the areas of the tablets
def area_7 := side_length_7 * side_length_7
def area_8 := side_length_8 * side_length_8
def area_9 := side_length_9 * side_length_9

-- Prove the differences in area
theorem area_difference_8_7 : area_8 - area_7 = 15 := by sorry
theorem area_difference_9_8 : area_9 - area_8 = 17 := by sorry

end area_difference_8_7_area_difference_9_8_l185_185009


namespace vitamin_C_in_apple_juice_l185_185077

theorem vitamin_C_in_apple_juice (A O : ℝ) 
  (h₁ : A + O = 185) 
  (h₂ : 2 * A + 3 * O = 452) :
  A = 103 :=
sorry

end vitamin_C_in_apple_juice_l185_185077


namespace range_of_a_l185_185805

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), a_seq n = if n < 6 then (1 / 2 - a) * n + 1 else a ^ (n - 5))
  (h2 : ∀ (n : ℕ), n > 0 → a_seq n > a_seq (n + 1)) :
  (1 / 2 : ℝ) < a ∧ a < (7 / 12 : ℝ) :=
sorry

end range_of_a_l185_185805


namespace order_of_a_b_c_l185_185713

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem order_of_a_b_c : b < a ∧ a < c :=
by
  sorry

end order_of_a_b_c_l185_185713


namespace numbering_tube_contacts_l185_185887

theorem numbering_tube_contacts {n : ℕ} (hn : n = 7) :
  ∃ (f g : ℕ → ℕ), (∀ k : ℕ, f k = k % n) ∧ (∀ k : ℕ, g k = (n - k) % n) ∧ 
  (∀ m : ℕ, ∃ k : ℕ, f (k + m) % n = g k % n) :=
by
  sorry

end numbering_tube_contacts_l185_185887


namespace false_inverse_proposition_l185_185276

theorem false_inverse_proposition (a b : ℝ) : (a^2 = b^2) → (a = b ∨ a = -b) := sorry

end false_inverse_proposition_l185_185276


namespace average_waiting_time_for_first_bite_l185_185477

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l185_185477


namespace no_regular_polygon_with_half_parallel_diagonals_l185_185497

-- Define the concept of a regular polygon with n sides
def is_regular_polygon (n : ℕ) : Prop :=
  n ≥ 3  -- A polygon has at least 3 sides

-- Define the concept of diagonals in the polygon
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the concept of diagonals being parallel to the sides
def parallell_diagonals (n : ℕ) : ℕ :=
  -- This needs more formalization if specified, here's a placeholder
  sorry

-- The main theorem to prove
theorem no_regular_polygon_with_half_parallel_diagonals (n : ℕ) (h : is_regular_polygon n) :
  ¬ (∃ k : ℕ, k = num_diagonals n / 2 ∧ k = parallell_diagonals n) :=
begin
  sorry
end

end no_regular_polygon_with_half_parallel_diagonals_l185_185497


namespace lcm_of_28_and_24_is_168_l185_185990

/-- Racing car A completes the track in 28 seconds.
    Racing car B completes the track in 24 seconds.
    Both cars start at the same time.
    We want to prove that the time after which both cars will be side by side again
    (least common multiple of their lap times) is 168 seconds. -/
theorem lcm_of_28_and_24_is_168 :
  Nat.lcm 28 24 = 168 :=
sorry

end lcm_of_28_and_24_is_168_l185_185990


namespace x_is_integer_l185_185934

theorem x_is_integer 
  (x : ℝ)
  (h1 : ∃ a : ℤ, a = x^1960 - x^1919)
  (h2 : ∃ b : ℤ, b = x^2001 - x^1960) :
  ∃ k : ℤ, x = k :=
sorry

end x_is_integer_l185_185934


namespace binary_subtraction_to_decimal_l185_185137

theorem binary_subtraction_to_decimal :
  (511 - 63 = 448) :=
by
  sorry

end binary_subtraction_to_decimal_l185_185137


namespace CorrectChoice_l185_185204

open Classical

-- Define the integer n
variable (n : ℤ)

-- Define proposition p: 2n - 1 is always odd
def p : Prop := ∃ k : ℤ, 2 * k + 1 = 2 * n - 1

-- Define proposition q: 2n + 1 is always even
def q : Prop := ∃ k : ℤ, 2 * k = 2 * n + 1

-- The theorem we want to prove
theorem CorrectChoice : (p n ∨ q n) :=
by
  sorry

end CorrectChoice_l185_185204


namespace ratio_of_areas_of_triangles_l185_185124

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l185_185124


namespace value_of_f_l185_185803

def f (x z : ℕ) (y : ℕ) : ℕ := 2 * x^2 + y - z

theorem value_of_f (y : ℕ) (h1 : f 2 3 y = 100) : f 5 7 y = 138 := by
  sorry

end value_of_f_l185_185803


namespace unique_solution_exists_l185_185779

theorem unique_solution_exists (a x y z : ℝ) 
  (h1 : z = a * (x + 2 * y + 5 / 2)) 
  (h2 : x^2 + y^2 + 2 * x - y + a * (x + 2 * y + 5 / 2) = 0) :
  a = 1 → x = -3 / 2 ∧ y = -1 / 2 ∧ z = 0 := 
by
  sorry

end unique_solution_exists_l185_185779


namespace two_digit_number_l185_185666

theorem two_digit_number (a : ℕ) (N M : ℕ) :
  (10 ≤ a) ∧ (a ≤ 99) ∧ (2 * a + 1 = N^2) ∧ (3 * a + 1 = M^2) → a = 40 :=
by
  sorry

end two_digit_number_l185_185666


namespace total_animals_sighted_l185_185020

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end total_animals_sighted_l185_185020


namespace shirts_bought_by_peter_l185_185905

-- Define the constants and assumptions
variables (P S x : ℕ)

-- State the conditions given in the problem
def condition1 : P = 6 :=
by sorry

def condition2 : 2 * S = 20 :=
by sorry

def condition3 : 2 * P + x * S = 62 :=
by sorry

-- State the theorem to be proven
theorem shirts_bought_by_peter : x = 5 :=
by sorry

end shirts_bought_by_peter_l185_185905


namespace production_units_l185_185810

-- Define the production function U
def U (women hours days : ℕ) : ℕ := women * hours * days

-- State the theorem
theorem production_units (x z : ℕ) (hx : ¬ x = 0) :
  U z z z = (z^3 / x) :=
  sorry

end production_units_l185_185810


namespace spherical_to_rectangular_correct_l185_185489

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end spherical_to_rectangular_correct_l185_185489


namespace harry_spends_1920_annually_l185_185688

def geckoCount : Nat := 3
def iguanaCount : Nat := 2
def snakeCount : Nat := 4

def geckoFeedTimesPerMonth : Nat := 2
def iguanaFeedTimesPerMonth : Nat := 3
def snakeFeedTimesPerMonth : Nat := 1 / 2

def geckoFeedCostPerMeal : Nat := 8
def iguanaFeedCostPerMeal : Nat := 12
def snakeFeedCostPerMeal : Nat := 20

def annualCostHarrySpends (geckoCount guCount scCount : Nat) (geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth : Nat) (geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal : Nat) : Nat :=
  let geckoAnnualCost := geckoCount * (geckoFeedTimesPerMonth * 12 * geckoFeedCostPerMeal)
  let iguanaAnnualCost := iguanaCount * (iguanaFeedTimesPerMonth * 12 * iguanaFeedCostPerMeal)
  let snakeAnnualCost := snakeCount * ((12 / (2 : Nat)) * snakeFeedCostPerMeal)
  geckoAnnualCost + iguanaAnnualCost + snakeAnnualCost

theorem harry_spends_1920_annually : annualCostHarrySpends geckoCount iguanaCount snakeCount geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal = 1920 := 
  sorry

end harry_spends_1920_annually_l185_185688


namespace digits_all_different_l185_185651

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l185_185651


namespace students_not_reading_l185_185444

theorem students_not_reading (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_reading : ℚ) (frac_boys_reading : ℚ)
  (h1 : total_girls = 12) (h2 : total_boys = 10)
  (h3 : frac_girls_reading = 5 / 6) (h4 : frac_boys_reading = 4 / 5) :
  let girls_not_reading := total_girls - total_girls * frac_girls_reading
  let boys_not_reading := total_boys - total_boys * frac_boys_reading
  let total_not_reading := girls_not_reading + boys_not_reading
  total_not_reading = 4 := sorry

end students_not_reading_l185_185444


namespace solutions_of_quadratic_l185_185839

theorem solutions_of_quadratic 
  (p q : ℚ) 
  (h₁ : 2 * p * p + 11 * p - 21 = 0) 
  (h₂ : 2 * q * q + 11 * q - 21 = 0) : 
  (p - q) * (p - q) = 289 / 4 := 
sorry

end solutions_of_quadratic_l185_185839


namespace quadratic_inequality_solution_l185_185790

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x - 21 ≤ 0 ↔ -3 ≤ x ∧ x ≤ 7 :=
sorry

end quadratic_inequality_solution_l185_185790


namespace carrots_remaining_l185_185752

theorem carrots_remaining 
  (total_carrots : ℕ)
  (weight_20_carrots : ℕ)
  (removed_carrots : ℕ)
  (avg_weight_remaining : ℕ)
  (avg_weight_removed : ℕ)
  (h1 : total_carrots = 20)
  (h2 : weight_20_carrots = 3640)
  (h3 : removed_carrots = 4)
  (h4 : avg_weight_remaining = 180)
  (h5 : avg_weight_removed = 190) :
  total_carrots - removed_carrots = 16 :=
by 
  -- h1 : 20 carrots in total
  -- h2 : total weight of 20 carrots is 3640 grams
  -- h3 : 4 carrots are removed
  -- h4 : average weight of remaining carrots is 180 grams
  -- h5 : average weight of removed carrots is 190 grams
  sorry

end carrots_remaining_l185_185752


namespace find_largest_n_l185_185686

theorem find_largest_n 
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (x y : ℕ)
  (h_a1 : a 1 = 1)
  (h_b1 : b 1 = 1)
  (h_arith_a : ∀ n : ℕ, a n = 1 + (n - 1) * x)
  (h_arith_b : ∀ n : ℕ, b n = 1 + (n - 1) * y)
  (h_order : x ≤ y)
  (h_product : ∃ n : ℕ, a n * b n = 4021) :
  ∃ n : ℕ, a n * b n = 4021 ∧ n ≤ 11 := 
by
  sorry

end find_largest_n_l185_185686


namespace tetrahedron_faces_equal_l185_185438

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end tetrahedron_faces_equal_l185_185438


namespace find_k_l185_185673

-- Define the problem conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- Define the dot product for 2D vectors
def dot_prod (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- State the theorem
theorem find_k (k : ℝ) (h : dot_prod b (c k) = 0) : k = -3/2 :=
by
  sorry

end find_k_l185_185673


namespace arithmetic_sequence_has_11_terms_l185_185035

theorem arithmetic_sequence_has_11_terms
  (a1 d : ℝ)
  (h_sum_first_four : 4 * a1 + 6 * d = 26)
  (h_sum_last_four : ∀ n, 4 * a1 + (4 * n - 10) * d = 110)
  (h_total_sum : ∃ n, n / 2 * (2 * a1 + (n - 1) * d) = 187) :
  ∃ n : ℝ, n = 11 := by
  sorry

end arithmetic_sequence_has_11_terms_l185_185035


namespace complementary_events_B_l185_185568

-- Definitions of events based on the problem's conditions
def white_balls : Finset ℕ := {1, 2, 3}
def black_balls : Finset ℕ := {4, 5, 6, 7}
def bag : Finset ℕ := white_balls ∪ black_balls
def draws : Finset (Finset ℕ) := bag.powerset.filter (λ s, s.card = 3)

def at_least_one_white_ball (s : Finset ℕ) : Prop := ∃ b ∈ s, b ∈ white_balls
def all_black_balls (s : Finset ℕ) : Prop := ∀ b ∈ s, b ∈ black_balls

-- Lean 4 statement to prove the pair are complementary events
theorem complementary_events_B : 
  (∀ s ∈ draws, at_least_one_white_ball s → ¬all_black_balls s) ∧
  (∀ s ∈ draws, (at_least_one_white_ball s ∨ all_black_balls s)) :=
by
  sorry

end complementary_events_B_l185_185568


namespace sufficient_but_not_necessary_condition_l185_185155

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x - 2

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) : 
  a ≤ 0 :=
sorry

end sufficient_but_not_necessary_condition_l185_185155


namespace range_of_a_l185_185798

variable (a x : ℝ)
def A (a : ℝ) := {x : ℝ | 2 * a ≤ x ∧ x ≤ a ^ 2 + 1}
def B (a : ℝ) := {x : ℝ | (x - 2) * (x - (3 * a + 1)) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ A a → x ∈ B a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by sorry

end range_of_a_l185_185798


namespace simplify_fraction_l185_185405

theorem simplify_fraction : 
  (5 / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108)) = (5 * Real.sqrt 3 / 54) :=
by
  -- Proof will be provided here
  sorry

end simplify_fraction_l185_185405


namespace lawrence_walked_total_distance_l185_185562

noncomputable def distance_per_day : ℝ := 4.0
noncomputable def number_of_days : ℝ := 3.0
noncomputable def total_distance_walked (distance_per_day : ℝ) (number_of_days : ℝ) : ℝ :=
  distance_per_day * number_of_days

theorem lawrence_walked_total_distance :
  total_distance_walked distance_per_day number_of_days = 12.0 :=
by
  -- The detailed proof is omitted as per the instructions.
  sorry

end lawrence_walked_total_distance_l185_185562


namespace parallel_lines_necessary_not_sufficient_l185_185350

theorem parallel_lines_necessary_not_sufficient {a : ℝ} 
  (h1 : ∀ x y : ℝ, a * x + (a + 2) * y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + a * y + 2 = 0) 
  (h3 : ∀ x y : ℝ, a * (1 * y + 2) = 1 * (a * y + 2)) : 
  (a = -1) -> (a = 2 ∨ a = -1 ∧ ¬(∀ b, a = b → a = -1)) :=
by
  -- proof goes here
  sorry

end parallel_lines_necessary_not_sufficient_l185_185350


namespace probability_digits_different_l185_185649

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l185_185649


namespace centroid_condition_l185_185703

-- Define the 4030 x 4030 grid and the selection of lines 
def grid_size : ℕ := 4030

def horizontal_lines : Finset ℤ := {n | -2015 ≤ n ∧ n ≤ 2015}
def vertical_lines : Finset ℤ := {n | -2015 ≤ n ∧ n ≤ 2015}

def intersection_points (h_lines : Finset ℤ) (v_lines : Finset ℤ) : Finset (ℤ × ℤ) := 
  (h_lines.product v_lines)

-- Define the centroid calculation
noncomputable def centroid (points : Finset (ℤ × ℤ)) : ℤ × ℤ :=
  let x_sum := points.sum (λ p, p.1)
  let y_sum := points.sum (λ p, p.2)
  let n := points.card
  (x_sum / n, y_sum / n)

-- The main theorem statement
theorem centroid_condition :
  ∀ (h_lines : Finset ℤ) (v_lines : Finset ℤ),
  horizontal_lines ⊆ h_lines →
  vertical_lines ⊆ v_lines →
  h_lines.card = 2017 →
  v_lines.card = 2017 →
  ∃ (A B C D E F : ℤ × ℤ),
    A ∈ intersection_points(h_lines, v_lines) ∧
    B ∈ intersection_points(h_lines, v_lines) ∧
    C ∈ intersection_points(h_lines, v_lines) ∧
    D ∈ intersection_points(h_lines, v_lines) ∧
    E ∈ intersection_points(h_lines, v_lines) ∧
    F ∈ intersection_points(h_lines, v_lines) ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    centroid {A, B, C} = (0, 0) ∧
    centroid {D, E, F} = (0, 0) :=
sorry

end centroid_condition_l185_185703


namespace domain_of_log_function_l185_185526

noncomputable def domain_f (k : ℤ) : Set ℝ :=
  {x : ℝ | (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
           (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3)}

theorem domain_of_log_function :
  ∀ x : ℝ, (∃ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
                      (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3))
  ↔ (3 - 4 * Real.sin x ^ 2 > 0) :=
by {
  sorry
}

end domain_of_log_function_l185_185526


namespace chessboard_probability_l185_185258

theorem chessboard_probability :
  ∀ (total_squares perimeter_squares : ℕ),
  total_squares = 100 →
  perimeter_squares = 36 →
  (total_squares - perimeter_squares) / total_squares = 16 / 25 := by
  intros total_squares perimeter_squares h_total h_perim
  rw [h_total, h_perim]
  norm_num
  sorry

end chessboard_probability_l185_185258


namespace time_required_painting_rooms_l185_185459

-- Definitions based on the conditions
def alice_rate := 1 / 4
def bob_rate := 1 / 6
def charlie_rate := 1 / 8
def combined_rate := 13 / 24
def required_time : ℚ := 74 / 13

-- Proof problem statement
theorem time_required_painting_rooms (t : ℚ) :
  (combined_rate) * (t - 2) = 2 ↔ t = required_time :=
by
  sorry

end time_required_painting_rooms_l185_185459


namespace light2011_is_green_l185_185923

def light_pattern : list string := ["green", "yellow", "yellow", "red", "red", "red"]

def color_of_light (n : ℕ) : string :=
  light_pattern[(n - 1) % 6]

theorem light2011_is_green : color_of_light 2011 = "green" :=
  by sorry

end light2011_is_green_l185_185923


namespace infinitely_many_primes_of_form_6n_plus_5_l185_185588

theorem infinitely_many_primes_of_form_6n_plus_5 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ p % 6 = 5 :=
sorry

end infinitely_many_primes_of_form_6n_plus_5_l185_185588


namespace determine_a_and_theta_l185_185549

noncomputable def f (a θ : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin (2 * x + θ)

theorem determine_a_and_theta :
  (∃ a θ : ℝ, 0 < θ ∧ θ < π ∧ a ≠ 0 ∧ (∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f a θ x ∈ Set.Icc (-2 : ℝ) 2) ∧ 
  (∀ (x1 x2 : ℝ), x1 ∈ Set.Icc (-5 * π / 12) (π / 12) → x2 ∈ Set.Icc (-5 * π / 12) (π / 12) → x1 < x2 → f a θ x1 > f a θ x2)) →
  (a = -1) ∧ (θ = π / 3) :=
sorry

end determine_a_and_theta_l185_185549


namespace ashley_family_spending_l185_185167

theorem ashley_family_spending:
  let child_ticket := 4.25
  let adult_ticket := child_ticket + 3.50
  let senior_ticket := adult_ticket - 1.75
  let morning_discount := 0.10
  let total_morning_tickets := 2 * adult_ticket + 4 * child_ticket + senior_ticket
  let morning_tickets_after_discount := total_morning_tickets * (1 - morning_discount)
  let buy_2_get_1_free_discount := child_ticket
  let discount_for_5_or_more := 4.00
  let total_tickets_after_vouchers := morning_tickets_after_discount - buy_2_get_1_free_discount - discount_for_5_or_more
  let popcorn := 5.25
  let soda := 3.50
  let candy := 4.00
  let concession_total := 3 * popcorn + 2 * soda + candy
  let concession_discount := concession_total * 0.10
  let concession_after_discount := concession_total - concession_discount
  let final_total := total_tickets_after_vouchers + concession_after_discount
  final_total = 50.47 := by
  sorry

end ashley_family_spending_l185_185167


namespace nancy_first_album_pictures_l185_185989

theorem nancy_first_album_pictures (total_pics : ℕ) (total_albums : ℕ) (pics_per_album : ℕ)
    (h1 : total_pics = 51) (h2 : total_albums = 8) (h3 : pics_per_album = 5) :
    (total_pics - total_albums * pics_per_album = 11) :=
by
    sorry

end nancy_first_album_pictures_l185_185989


namespace find_a_b_l185_185952

noncomputable def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x : ℝ | -3 < x ∧ x < 2 }
noncomputable def sol_set (a b : ℝ) : Set ℝ := { x : ℝ | x^2 + a * x + b < 0 }

theorem find_a_b :
  (sol_set (-2) (3 - 6)) = A ∩ B → (-1) + (-2) = -3 :=
by
  intros h1
  sorry

end find_a_b_l185_185952


namespace fifteen_percent_of_x_is_ninety_l185_185518

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l185_185518


namespace polygon_parallel_edges_l185_185034

theorem polygon_parallel_edges (n : ℕ) (h : n > 2) :
  (∃ i j, i ≠ j ∧ (i + 1) % n = (j + 1) % n) ↔ (∃ k, n = 2 * k) :=
  sorry

end polygon_parallel_edges_l185_185034


namespace sequence_sum_l185_185865

open Nat

-- Define the sequence
def a : ℕ → ℕ
| 0     => 1
| (n+1) => a n + (n + 1)

-- Define the sum of reciprocals up to the 2016 term
def sum_reciprocals : ℕ → ℚ
| 0     => 1 / (a 0)
| (n+1) => sum_reciprocals n + 1 / (a (n+1))

-- Define the property we wish to prove
theorem sequence_sum :
  sum_reciprocals 2015 = 4032 / 2017 :=
sorry

end sequence_sum_l185_185865


namespace percentage_defective_meters_l185_185642

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (h1 : total_meters = 150) (h2 : defective_meters = 15) : 
  (defective_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
sorry

end percentage_defective_meters_l185_185642


namespace calculate_f_zero_l185_185681

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem calculate_f_zero
  (ω φ : ℝ)
  (h_inc : ∀ x y : ℝ, (π / 6 < x ∧ x < y ∧ y < 2 * π / 3) → f ω φ x < f ω φ y)
  (h_symmetry1 : ∀ x : ℝ, f ω φ (π / 6 - x) = f ω φ (π / 6 + x))
  (h_symmetry2 : ∀ x : ℝ, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x)) :
  f ω φ 0 = -1 / 2 :=
sorry

end calculate_f_zero_l185_185681


namespace simplify_sum_of_polynomials_l185_185082

-- Definitions of the given polynomials
def P (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15
def Q (x : ℝ) : ℝ := -5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9

-- Statement to prove that the sum of P and Q equals the simplified polynomial
theorem simplify_sum_of_polynomials (x : ℝ) : 
  P x + Q x = 2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := 
sorry

end simplify_sum_of_polynomials_l185_185082


namespace edward_spent_13_l185_185664

-- Define the initial amount of money Edward had
def initial_amount : ℕ := 19
-- Define the current amount of money Edward has now
def current_amount : ℕ := 6
-- Define the amount of money Edward spent
def amount_spent : ℕ := initial_amount - current_amount

-- The proof we need to show
theorem edward_spent_13 : amount_spent = 13 := by
  -- The proof goes here.
  sorry

end edward_spent_13_l185_185664


namespace problem_statement_l185_185156

theorem problem_statement : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 :=
by sorry

end problem_statement_l185_185156


namespace candy_bar_price_l185_185431

theorem candy_bar_price (total_money bread_cost candy_bar_price remaining_money : ℝ) 
    (h1 : total_money = 32)
    (h2 : bread_cost = 3)
    (h3 : remaining_money = 18)
    (h4 : total_money - bread_cost - candy_bar_price - (1 / 3) * (total_money - bread_cost - candy_bar_price) = remaining_money) :
    candy_bar_price = 1.33 := 
sorry

end candy_bar_price_l185_185431


namespace buoy_radius_l185_185754

-- Define the conditions based on the given problem
def is_buoy_hole (width : ℝ) (depth : ℝ) : Prop :=
  width = 30 ∧ depth = 10

-- Define the statement to prove the radius of the buoy
theorem buoy_radius : ∀ r x : ℝ, is_buoy_hole 30 10 → (x^2 + 225 = (x + 10)^2) → r = x + 10 → r = 16.25 := by
  intros r x h_cond h_eq h_add
  sorry

end buoy_radius_l185_185754


namespace number_less_than_neg_one_is_neg_two_l185_185449

theorem number_less_than_neg_one_is_neg_two : ∃ x : ℤ, x = -1 - 1 ∧ x = -2 := by
  sorry

end number_less_than_neg_one_is_neg_two_l185_185449


namespace problem_remainders_l185_185717

open Int

theorem problem_remainders (x : ℤ) :
  (x + 2) % 45 = 7 →
  ((x + 2) % 20 = 7 ∧ x % 19 = 5) :=
by
  sorry

end problem_remainders_l185_185717


namespace cos_sin_ratio_l185_185675

open Real

-- Given conditions
variables {α β : Real}
axiom tan_alpha_beta : tan (α + β) = 2 / 5
axiom tan_beta_pi_over_4 : tan (β - π / 4) = 1 / 4

-- Theorem to be proven
theorem cos_sin_ratio (hαβ : tan (α + β) = 2 / 5) (hβ : tan (β - π / 4) = 1 / 4) :
  (cos α + sin α) / (cos α - sin α) = 3 / 22 :=
sorry

end cos_sin_ratio_l185_185675


namespace trains_distance_apart_l185_185628

-- Define the initial conditions
def cattle_train_speed : ℝ := 56
def diesel_train_speed : ℝ := cattle_train_speed - 33
def cattle_train_time : ℝ := 6 + 12
def diesel_train_time : ℝ := 12

-- Calculate distances
def cattle_train_distance : ℝ := cattle_train_speed * cattle_train_time
def diesel_train_distance : ℝ := diesel_train_speed * diesel_train_time

-- Define total distance apart
def distance_apart : ℝ := cattle_train_distance + diesel_train_distance

-- The theorem to prove
theorem trains_distance_apart :
  distance_apart = 1284 :=
by
  -- Skip the proof
  sorry

end trains_distance_apart_l185_185628


namespace systematic_sampling_student_selection_l185_185378

theorem systematic_sampling_student_selection
    (total_students : ℕ)
    (num_groups : ℕ)
    (students_per_group : ℕ)
    (third_group_selected : ℕ)
    (third_group_num : ℕ)
    (eighth_group_num : ℕ)
    (h1 : total_students = 50)
    (h2 : num_groups = 10)
    (h3 : students_per_group = total_students / num_groups)
    (h4 : students_per_group = 5)
    (h5 : 11 ≤ third_group_selected ∧ third_group_selected ≤ 15)
    (h6 : third_group_selected = 12)
    (h7 : third_group_num = 3)
    (h8 : eighth_group_num = 8) :
  eighth_group_selected = 37 :=
by
  sorry

end systematic_sampling_student_selection_l185_185378


namespace find_a_find_b_find_T_l185_185795

open Real

def S (n : ℕ) : ℝ := 2 * n^2 + n

def a (n : ℕ) : ℝ := if n = 1 then 3 else S n - S (n - 1)

def b (n : ℕ) : ℝ := 2^(n - 1)

def T (n : ℕ) : ℝ := (4 * n - 5) * 2^n + 5

theorem find_a (n : ℕ) (hn : n > 0) : a n = 4 * n - 1 :=
by sorry

theorem find_b (n : ℕ) (hn : n > 0) : b n = 2^(n-1) :=
by sorry

theorem find_T (n : ℕ) (hn : n > 0) (a_def : ∀ n, a n = 4 * n - 1) (b_def : ∀ n, b n = 2^(n-1)) : T n = (4 * n - 5) * 2^n + 5 :=
by sorry

end find_a_find_b_find_T_l185_185795


namespace simplify_expression_l185_185849

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 :=
by sorry

end simplify_expression_l185_185849


namespace sin_270_eq_neg_one_l185_185914

theorem sin_270_eq_neg_one : 
  let Q := (0, -1) in
  ∃ (θ : ℝ), θ = 270 * Real.pi / 180 ∧ ∃ (Q : ℝ × ℝ), 
    Q = ⟨Real.cos θ, Real.sin θ⟩ ∧ Real.sin θ = -1 :=
by 
  sorry

end sin_270_eq_neg_one_l185_185914


namespace cos_seven_pi_over_six_l185_185780

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end cos_seven_pi_over_six_l185_185780


namespace smallest_gcd_bc_l185_185211

theorem smallest_gcd_bc (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (gcd_ab : Nat.gcd a b = 168) (gcd_ac : Nat.gcd a c = 693) : Nat.gcd b c = 21 := 
sorry

end smallest_gcd_bc_l185_185211


namespace solution_set_inequality_l185_185250

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ s → b ∈ s → a ≤ b → f a ≤ f b

def f_increasing_on_pos : Prop := is_increasing_on f (Set.Ioi 0)

def f_at_one_zero : Prop := f 1 = 0

theorem solution_set_inequality : 
    is_odd f →
    f_increasing_on_pos →
    f_at_one_zero →
    {x : ℝ | x * (f x - f (-x)) < 0} = {x : ℝ | -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1} :=
sorry

end solution_set_inequality_l185_185250


namespace final_notebooks_l185_185165

def initial_notebooks : ℕ := 10
def ordered_notebooks : ℕ := 6
def lost_notebooks : ℕ := 2

theorem final_notebooks : initial_notebooks + ordered_notebooks - lost_notebooks = 14 :=
by
  sorry

end final_notebooks_l185_185165


namespace other_eigenvalue_and_eigenvector_l185_185683

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]

def λ1 : ℝ := -1

def e1 : Fin 2 → ℝ := ![1, -1]

theorem other_eigenvalue_and_eigenvector : 
  let λ2 := 4
  let e2 := ![2, 3]
  eigenvalue M λ2 ∧ (M.mul_vec e2 = λ2 • e2) := by
  sorry

end other_eigenvalue_and_eigenvector_l185_185683


namespace probability_sum_odd_l185_185424

open Classical
open Probability

noncomputable def probability_odd_sum : ℚ := sorry

theorem probability_sum_odd (n : ℕ) (h : n = 3) : 
  probability_odd_sum = 7 / 16 := by
  sorry

end probability_sum_odd_l185_185424


namespace apple_production_total_l185_185899

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l185_185899


namespace real_roots_range_of_k_l185_185366

theorem real_roots_range_of_k (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k + 3) = 0) ↔ (k ≤ 3 / 2) :=
sorry

end real_roots_range_of_k_l185_185366


namespace negation_of_proposition_l185_185954

theorem negation_of_proposition (a : ℝ) :
  (¬ (∀ x : ℝ, (x - a) ^ 2 + 2 > 0)) ↔ (∃ x : ℝ, (x - a) ^ 2 + 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l185_185954


namespace general_term_b_l185_185195

noncomputable def S (n : ℕ) : ℚ := sorry -- Define the sum of the first n terms sequence S_n
noncomputable def a (n : ℕ) : ℚ := sorry -- Define the sequence a_n
noncomputable def b (n : ℕ) : ℤ := Int.log 3 (|a n|) -- Define the sequence b_n using log base 3

-- Theorem stating the general formula for the sequence b_n
theorem general_term_b (n : ℕ) (h : 0 < n) :
  b n = -n :=
sorry -- We skip the proof, focusing on statement declaration

end general_term_b_l185_185195


namespace solve_system_l185_185731

theorem solve_system :
  ∃ (x y z : ℝ), x + y + z = 9 ∧ (1/x + 1/y + 1/z = 1) ∧ (x * y + x * z + y * z = 27) ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end solve_system_l185_185731


namespace binomial_1300_2_eq_844350_l185_185313

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l185_185313


namespace probability_all_digits_different_l185_185648

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l185_185648


namespace num_subsets_satisfy_property_M_l185_185338

-- Define property M for subsets
def property_M (A : Finset ℕ) : Prop :=
  (∃ k ∈ A, k + 1 ∈ A) ∧ ∀ k ∈ A, k - 2 ∉ A

-- The main statement - number of 3-element subsets with property M
theorem num_subsets_satisfy_property_M :
  (Finset.filter property_M { A : Finset ℕ // A.card = 3 } (Finset.powerset (Finset.range' 1 6))).card = 6 :=
sorry

end num_subsets_satisfy_property_M_l185_185338


namespace ratio_of_areas_l185_185134

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l185_185134


namespace max_min_product_l185_185383

theorem max_min_product (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h_sum : p + q + r = 13) (h_prod_sum : p * q + q * r + r * p = 30) :
  ∃ n, n = min (p * q) (min (q * r) (r * p)) ∧ n = 10 :=
by
  sorry

end max_min_product_l185_185383


namespace triangle_area_is_two_l185_185701

noncomputable def triangle_area (b c : ℝ) (angle_A : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_A

theorem triangle_area_is_two
  (A B C : ℝ) (a b c : ℝ)
  (hA : A = π / 4)
  (hCondition : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B)
  (hBC : b * c = 4 * Real.sqrt 2) : 
  triangle_area b c A = 2 :=
by
  -- actual proof omitted
  sorry

end triangle_area_is_two_l185_185701


namespace ratio_of_areas_of_triangles_l185_185127

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l185_185127


namespace optionA_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_not_right_triangle_l185_185199

-- Four conditions for the triangle ABC
axiom condA : ∀ (A B C : ℝ), A + B = C
axiom condB : ∀ (A B C : ℝ), 2 * A = B ∧ 3 * A = C
axiom condC : ∀ (a b c : ℝ), a^2 = b^2 - c^2
axiom condD : ∀ (a b c : ℝ), a^2 = 5 ∧ b^2 = 12 ∧ c^2 = 13

-- The angles and sides of triangle ABC
variable (A B C : ℝ)
variable (a b c : ℝ)

-- The proof
theorem optionA_right_triangle : condA A B C → A + B = 90 := by 
  sorry
theorem optionB_right_triangle : condB A B C → C = 90 := by 
  sorry
theorem optionC_right_triangle : condC a b c → a^2 + c^2 = b^2 := by 
  sorry
theorem optionD_not_right_triangle : condD a b c → ¬(a^2 + b^2 = c^2) := by 
  sorry

end optionA_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_not_right_triangle_l185_185199


namespace two_hundredth_digit_of_fraction_l185_185138

theorem two_hundredth_digit_of_fraction (h1 : (17 : ℚ) / 70 = (1 / 10) * (17 / 7))
    (h2 : ∃ r : ℝ, (((17 : ℚ) / 7) : ℝ) = r ∧ r = 2 + (0.428571).over) :
    (∃ d : ℕ, d = 2 ∧ ∀ n : ℕ, n = 200 → Digit_At n (17 / 70) d) := 
by
  sorry

end two_hundredth_digit_of_fraction_l185_185138


namespace general_formula_a_n_sum_first_n_terms_T_n_l185_185538

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Condition: S_n = 2a_n - 3
axiom condition_S (n : ℕ) : S_n n = 2 * (a_n n) - 3

-- (I) General formula for a_n
theorem general_formula_a_n (n : ℕ) : a_n n = 3 * 2^(n - 1) := 
sorry

-- (II) General formula for T_n
theorem sum_first_n_terms_T_n (n : ℕ) : T_n n = 3 * (n - 1) * 2^n + 3 := 
sorry

end general_formula_a_n_sum_first_n_terms_T_n_l185_185538


namespace evaluate_expression_l185_185935

noncomputable def greatest_integer (x : Real) : Int := ⌊x⌋

theorem evaluate_expression (y : Real) (h : y = 7.2) :
  greatest_integer 6.5 * greatest_integer (2 / 3)
  + greatest_integer 2 * y
  + greatest_integer 8.4 - 6.0 = 16.4 := by
  simp [greatest_integer, h]
  sorry

end evaluate_expression_l185_185935


namespace prime_719_exists_l185_185529

theorem prime_719_exists (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) :
  (a^4 + b^4 + c^4 - 3 = 719) → Nat.Prime (a^4 + b^4 + c^4 - 3) := sorry

end prime_719_exists_l185_185529


namespace probability_of_drawing_red_ball_l185_185627

theorem probability_of_drawing_red_ball (total_balls red_balls : ℕ) (h_total : total_balls = 10) (h_red : red_balls = 7) : (red_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l185_185627


namespace train_length_360_l185_185304

variable (time_to_cross : ℝ) (speed_of_train : ℝ)

theorem train_length_360 (h1 : time_to_cross = 12) (h2 : speed_of_train = 30) :
  speed_of_train * time_to_cross = 360 :=
by
  rw [h1, h2]
  norm_num

end train_length_360_l185_185304


namespace least_five_digit_congruent_to_7_mod_17_l185_185872

theorem least_five_digit_congruent_to_7_mod_17 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 17 = 7 ∧ (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 7 → n ≤ m) :=
sorry

end least_five_digit_congruent_to_7_mod_17_l185_185872


namespace average_waiting_time_for_first_bite_l185_185476

theorem average_waiting_time_for_first_bite
  (bites_first_rod : ℝ)
  (bites_second_rod: ℝ)
  (total_bites: ℝ)
  (time_interval: ℝ)
  (H1 : bites_first_rod = 3)
  (H2 : bites_second_rod = 2)
  (H3 : total_bites = 5)
  (H4 : time_interval = 6) :
  1 / (total_bites / time_interval) = 1.2 :=
by
  rw [H3, H4]
  simp
  norm_num
  rw [div_eq_mul_inv, inv_div, inv_inv]
  norm_num
  sorry

end average_waiting_time_for_first_bite_l185_185476


namespace find_f_one_l185_185367

noncomputable def f_inv (x : ℝ) : ℝ := 2^(x + 1)

theorem find_f_one : ∃ f : ℝ → ℝ, (∀ y, f (f_inv y) = y) ∧ f 1 = -1 :=
by
  sorry

end find_f_one_l185_185367


namespace sixth_largest_divisor_correct_l185_185744

noncomputable def sixth_largest_divisor_of_4056600000 : ℕ :=
  50707500

theorem sixth_largest_divisor_correct : sixth_largest_divisor_of_4056600000 = 50707500 :=
sorry

end sixth_largest_divisor_correct_l185_185744


namespace unique_flavors_l185_185853

theorem unique_flavors (x y : ℕ) (h₀ : x = 5) (h₁ : y = 4) : 
  (∃ f : ℕ, f = 17) :=
sorry

end unique_flavors_l185_185853


namespace quadratic_real_roots_range_l185_185033

theorem quadratic_real_roots_range (k : ℝ) (h : ∀ x : ℝ, (k - 1) * x^2 - 2 * x + 1 = 0) : k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_range_l185_185033


namespace probability_of_three_heads_in_eight_tosses_l185_185760

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end probability_of_three_heads_in_eight_tosses_l185_185760


namespace winning_candidate_percentage_l185_185306

theorem winning_candidate_percentage (total_membership: ℕ)
  (votes_cast: ℕ) (winning_percentage: ℝ) (h1: total_membership = 1600)
  (h2: votes_cast = 525) (h3: winning_percentage = 19.6875)
  : (winning_percentage / 100 * total_membership / votes_cast * 100 = 60) :=
by
  sorry

end winning_candidate_percentage_l185_185306


namespace sqrt_five_minus_one_range_l185_185665

theorem sqrt_five_minus_one_range (h : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) : 
  1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := 
by 
  sorry

end sqrt_five_minus_one_range_l185_185665


namespace evaluate_division_l185_185152

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l185_185152


namespace evaluate_expression_l185_185926

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l185_185926


namespace radii_of_circles_l185_185831

theorem radii_of_circles
  (r s : ℝ)
  (h_ratio : r / s = 9 / 4)
  (h_right_triangle : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)
  (h_tangent : (r + s)^2 = (r - s)^2 + 12^2) :
   r = 20 / 47 ∧ s = 45 / 47 :=
by
  sorry

end radii_of_circles_l185_185831


namespace sam_after_joan_took_marbles_l185_185261

theorem sam_after_joan_took_marbles
  (original_yellow : ℕ)
  (marbles_taken_by_joan : ℕ)
  (remaining_yellow : ℕ)
  (h1 : original_yellow = 86)
  (h2 : marbles_taken_by_joan = 25)
  (h3 : remaining_yellow = original_yellow - marbles_taken_by_joan) :
  remaining_yellow = 61 :=
by
  sorry

end sam_after_joan_took_marbles_l185_185261


namespace find_fraction_value_l185_185835

variable {x y : ℂ}

theorem find_fraction_value
    (h1 : (x^2 + y^2) / (x + y) = 4)
    (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
    (x^6 + y^6) / (x^5 + y^5) = 4 := by
  sorry

end find_fraction_value_l185_185835


namespace total_houses_is_160_l185_185056

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end total_houses_is_160_l185_185056


namespace non_square_solution_equiv_l185_185541

theorem non_square_solution_equiv 
  (a b : ℤ) (h1 : ¬∃ k : ℤ, a = k^2) (h2 : ¬∃ k : ℤ, b = k^2) :
  (∃ x y z w : ℤ, x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) ↔
  (∃ x y z : ℤ, x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0)) :=
by sorry

end non_square_solution_equiv_l185_185541


namespace no_distinct_natural_numbers_exist_l185_185494

theorem no_distinct_natural_numbers_exist 
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ¬ (a + 1 / a = (1 / 2) * (b + 1 / b + c + 1 / c)) :=
sorry

end no_distinct_natural_numbers_exist_l185_185494


namespace sports_club_membership_l185_185224

theorem sports_club_membership (B T Both Neither : ℕ) (hB : B = 17) (hT : T = 19) (hBoth : Both = 11) (hNeither : Neither = 2) :
  B + T - Both + Neither = 27 := by
  sorry

end sports_club_membership_l185_185224


namespace math_problem_l185_185329

theorem math_problem :
  2537 + 240 * 3 / 60 - 347 = 2202 :=
by
  sorry

end math_problem_l185_185329


namespace James_age_is_11_l185_185064

-- Define the ages of Julio and James.
def Julio_age := 36

-- The age condition in 14 years.
def Julio_age_in_14_years := Julio_age + 14

-- James' age in 14 years and the relation as per the condition.
def James_age_in_14_years (J : ℕ) := J + 14

-- The main proof statement.
theorem James_age_is_11 (J : ℕ) 
  (h1 : Julio_age_in_14_years = 2 * James_age_in_14_years J) : J = 11 :=
by
  sorry

end James_age_is_11_l185_185064


namespace find_number_l185_185506

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l185_185506


namespace eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l185_185406

-- Equation (1): 2x^2 + 2x - 1 = 0
theorem eq_1_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) := by
  sorry

-- Equation (2): x(x-1) = 2(x-1)
theorem eq_2_solution (x : ℝ) :
  x * (x - 1) = 2 * (x - 1) ↔ (x = 1 ∨ x = 2) := by
  sorry

-- Equation (3): 4(x-2)^2 = 9(2x+1)^2
theorem eq_3_solution (x : ℝ) :
  4 * (x - 2)^2 = 9 * (2 * x + 1)^2 ↔ (x = -7 / 4 ∨ x = 1 / 8) := by
  sorry

-- Equation (4): (2x-1)^2 - 3(2x-1) = 4
theorem eq_4_solution (x : ℝ) :
  (2 * x - 1)^2 - 3 * (2 * x - 1) = 4 ↔ (x = 5 / 2 ∨ x = 0) := by
  sorry

end eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l185_185406


namespace committee_formations_l185_185535

theorem committee_formations :
  let students := 11
  let teachers := 3
  let total_people := students + teachers
  let committee_size := 8
  (nat.choose total_people committee_size) - (nat.choose students committee_size) = 2838 :=
by
  sorry

end committee_formations_l185_185535


namespace binom_1300_2_eq_844350_l185_185317

theorem binom_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 :=
by
  sorry

end binom_1300_2_eq_844350_l185_185317


namespace largest_divisor_if_n_sq_div_72_l185_185364

theorem largest_divisor_if_n_sq_div_72 (n : ℕ) (h : n > 0) (h72 : 72 ∣ n^2) : ∃ m, m = 12 ∧ m ∣ n :=
by { sorry }

end largest_divisor_if_n_sq_div_72_l185_185364


namespace arccos_sin_three_l185_185657

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l185_185657


namespace johan_painted_green_fraction_l185_185833

theorem johan_painted_green_fraction :
  let total_rooms := 10
  let walls_per_room := 8
  let purple_walls := 32
  let purple_rooms := purple_walls / walls_per_room
  let green_rooms := total_rooms - purple_rooms
  (green_rooms : ℚ) / total_rooms = 3 / 5 := by
  sorry

end johan_painted_green_fraction_l185_185833


namespace percentage_of_y_l185_185052

theorem percentage_of_y (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y :=
by
  sorry

end percentage_of_y_l185_185052


namespace range_of_k_l185_185678

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 2 then 2 / x else (x - 1)^3

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 - k = 0 ∧ f x2 - k = 0) ↔ (0 < k ∧ k < 1) := sorry

end range_of_k_l185_185678


namespace trees_died_l185_185116

theorem trees_died 
  (original_trees : ℕ) 
  (cut_trees : ℕ) 
  (remaining_trees : ℕ) 
  (died_trees : ℕ)
  (h1 : original_trees = 86)
  (h2 : cut_trees = 23)
  (h3 : remaining_trees = 48)
  (h4 : original_trees - died_trees - cut_trees = remaining_trees) : 
  died_trees = 15 :=
by
  sorry

end trees_died_l185_185116


namespace find_m_no_solution_l185_185218

-- Define the condition that the equation has no solution
def no_solution (m : ℤ) : Prop :=
  ∀ x : ℤ, (x + m)/(4 - x^2) + x / (x - 2) ≠ 1

-- State the proof problem in Lean 4
theorem find_m_no_solution : ∀ m : ℤ, no_solution m → (m = 2 ∨ m = 6) :=
by
  sorry

end find_m_no_solution_l185_185218


namespace apple_tree_total_production_l185_185897

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l185_185897


namespace abs_neg_three_halves_l185_185089

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l185_185089


namespace find_number_l185_185521

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l185_185521


namespace tickets_sold_total_l185_185606

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end tickets_sold_total_l185_185606


namespace weight_of_new_person_l185_185999

theorem weight_of_new_person
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (replaced_weight : ℝ)
  (weight_increase_total : ℝ)
  (W : ℝ)
  (h1 : avg_increase = 4.5)
  (h2 : num_persons = 8)
  (h3 : replaced_weight = 65)
  (h4 : weight_increase_total = 8 * 4.5)
  (h5 : W = replaced_weight + weight_increase_total) :
  W = 101 :=
by
  sorry

end weight_of_new_person_l185_185999


namespace values_of_a_plus_b_l185_185695

theorem values_of_a_plus_b (a b : ℝ) (h1 : abs (-a) = abs (-1)) (h2 : b^2 = 9) (h3 : abs (a - b) = b - a) : a + b = 2 ∨ a + b = 4 := 
by 
  sorry

end values_of_a_plus_b_l185_185695


namespace all_increased_quadratics_have_integer_roots_l185_185732

def original_quadratic (p q : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -p ∧ α * β = q

def increased_quadratic (p q n : ℤ) : Prop :=
  ∃ α β : ℤ, α + β = -(p + n) ∧ α * β = (q + n)

theorem all_increased_quadratics_have_integer_roots (p q : ℤ) :
  original_quadratic p q →
  (∀ n, 0 ≤ n ∧ n ≤ 9 → increased_quadratic p q n) :=
sorry

end all_increased_quadratics_have_integer_roots_l185_185732


namespace original_number_divisible_l185_185337

theorem original_number_divisible (N M R : ℕ) (n : ℕ) (hN : N = 1000 * M + R)
  (hDiff : (M - R) % n = 0) (hn : n = 7 ∨ n = 11 ∨ n = 13) : N % n = 0 :=
by
  sorry

end original_number_divisible_l185_185337


namespace quadratic_intersects_xaxis_once_l185_185815

theorem quadratic_intersects_xaxis_once (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0) ↔ k = 1 :=
by
  sorry

end quadratic_intersects_xaxis_once_l185_185815


namespace min_detectors_correct_l185_185740

noncomputable def min_detectors (M N : ℕ) : ℕ :=
  ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊

theorem min_detectors_correct (M N : ℕ) (hM : 2 ≤ M) (hN : 2 ≤ N) :
  min_detectors M N = ⌈(M : ℝ) / 2⌉₊ + ⌈(N : ℝ) / 2⌉₊ :=
by {
  -- The proof goes here
  sorry
}

end min_detectors_correct_l185_185740


namespace midpoint_distance_trapezoid_l185_185705

theorem midpoint_distance_trapezoid (x : ℝ) : 
  let AD := x
  let BC := 5
  PQ = (|x - 5| / 2) :=
sorry

end midpoint_distance_trapezoid_l185_185705


namespace abs_neg_three_halves_l185_185090

theorem abs_neg_three_halves : |(-3 : ℚ) / 2| = 3 / 2 := 
sorry

end abs_neg_three_halves_l185_185090


namespace original_cost_price_l185_185301

theorem original_cost_price (C : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) 
    (h3 : S_new = 1.25 * C - 14.70) (h4 : S_new = 1.04 * C) : C = 70 := 
by {
  sorry
}

end original_cost_price_l185_185301


namespace find_f3_l185_185548

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_f3
  (a b : ℝ)
  (h1 : f a b 3 1 = 7)
  (h2 : f a b 3 2 = 12) :
  f a b 3 3 = 18 :=
sorry

end find_f3_l185_185548


namespace number_of_rows_l185_185721

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

end number_of_rows_l185_185721


namespace number_of_perfect_cubes_l185_185555

theorem number_of_perfect_cubes (n : ℤ) : 
  (∃ (count : ℤ), (∀ (x : ℤ), (100 < x^3 ∧ x^3 < 400) ↔ x = 5 ∨ x = 6 ∨ x = 7) ∧ (count = 3)) := 
sorry

end number_of_perfect_cubes_l185_185555


namespace base_of_numbering_system_l185_185259

-- Definitions based on conditions
def num_children := 100
def num_boys := 24
def num_girls := 32

-- Problem statement: Prove the base of numbering system used is 6
theorem base_of_numbering_system (n: ℕ) (h: n ≠ 0):
    n^2 = (2 * n + 4) + (3 * n + 2) → n = 6 := 
  by
    sorry

end base_of_numbering_system_l185_185259


namespace distance_traveled_l185_185574

-- Define the conditions
def rate : Real := 60  -- rate of 60 miles per hour
def total_break_time : Real := 1  -- total break time of 1 hour
def total_trip_time : Real := 9  -- total trip time of 9 hours

-- The theorem to prove the distance traveled
theorem distance_traveled : rate * (total_trip_time - total_break_time) = 480 := 
by
  sorry

end distance_traveled_l185_185574


namespace dividend_calculation_l185_185991

theorem dividend_calculation 
  (D : ℝ) (Q : ℕ) (R : ℕ) 
  (hD : D = 164.98876404494382)
  (hQ : Q = 89)
  (hR : R = 14) :
  ⌈D * Q + R⌉ = 14698 :=
sorry

end dividend_calculation_l185_185991


namespace distinct_ints_divisibility_l185_185985

theorem distinct_ints_divisibility
  (x y z : ℤ) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : z ≠ x) : 
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * (y - z) * (z - x) * (x - y) * k := 
by 
  sorry

end distinct_ints_divisibility_l185_185985


namespace find_point_M_l185_185794

def parabola (x y : ℝ) := x^2 = 4 * y
def focus_dist (M : ℝ × ℝ) := dist M (0, 1) = 2
def point_on_parabola (M : ℝ × ℝ) := parabola M.1 M.2

theorem find_point_M (M : ℝ × ℝ) (h1 : point_on_parabola M) (h2 : focus_dist M) :
  M = (2, 1) ∨ M = (-2, 1) := by
  sorry

end find_point_M_l185_185794


namespace complex_square_l185_185170

theorem complex_square (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end complex_square_l185_185170


namespace total_ticket_count_is_59_l185_185607

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end total_ticket_count_is_59_l185_185607


namespace employees_cycle_l185_185158

theorem employees_cycle (total_employees : ℕ) (drivers_percentage walkers_percentage cyclers_percentage: ℕ) (walk_cycle_ratio_walk walk_cycle_ratio_cycle: ℕ)
    (h_total : total_employees = 500)
    (h_drivers_perc : drivers_percentage = 35)
    (h_transit_perc : walkers_percentage = 25)
    (h_walkers_cyclers_ratio_walk : walk_cycle_ratio_walk = 3)
    (h_walkers_cyclers_ratio_cycle : walk_cycle_ratio_cycle = 7) :
    cyclers_percentage = 140 :=
by
  sorry

end employees_cycle_l185_185158


namespace average_waiting_time_l185_185472

theorem average_waiting_time 
  (bites_rod1 : ℕ) (bites_rod2 : ℕ) (total_time : ℕ)
  (avg_bites_rod1 : bites_rod1 = 3)
  (avg_bites_rod2 : bites_rod2 = 2)
  (total_bites : bites_rod1 + bites_rod2 = 5)
  (interval : total_time = 6) :
  (total_time : ℝ) / (bites_rod1 + bites_rod2 : ℝ) = 1.2 :=
by
  sorry

end average_waiting_time_l185_185472


namespace solve_for_x_l185_185046

-- Define the operation
def triangle (a b : ℝ) : ℝ := 2 * a - b

-- Define the necessary conditions and the goal
theorem solve_for_x :
  (∀ (a b : ℝ), triangle a b = 2 * a - b) →
  (∃ x : ℝ, triangle x (triangle 1 3) = 2) →
  ∃ x : ℝ, x = 1 / 2 :=
by 
  intros h_main h_eqn
  -- We can skip the proof part as requested.
  sorry

end solve_for_x_l185_185046


namespace number_of_tables_l185_185965

/-- Problem Statement
  In a hall used for a conference, each table is surrounded by 8 stools and 4 chairs. Each stool has 3 legs,
  each chair has 4 legs, and each table has 4 legs. If the total number of legs for all tables, stools, and chairs is 704,
  the number of tables in the hall is 16. -/
theorem number_of_tables (legs_per_stool legs_per_chair legs_per_table total_legs t : ℕ) 
  (Hstools : ∀ tables, stools = 8 * tables)
  (Hchairs : ∀ tables, chairs = 4 * tables)
  (Hlegs : 3 * stools + 4 * chairs + 4 * t = total_legs)
  (Hleg_values : legs_per_stool = 3 ∧ legs_per_chair = 4 ∧ legs_per_table = 4)
  (Htotal_legs : total_legs = 704) :
  t = 16 := by
  sorry

end number_of_tables_l185_185965


namespace number_of_adults_attending_concert_l185_185101

-- We have to define the constants and conditions first.
variable (A C : ℕ)
variable (h1 : A + C = 578)
variable (h2 : 2 * A + 3 / 2 * C = 985)

-- Now we state the theorem that given these conditions, A is equal to 236.

theorem number_of_adults_attending_concert : A = 236 :=
by sorry

end number_of_adults_attending_concert_l185_185101


namespace geom_seq_sum_first_10_terms_l185_185829

variable (a : ℕ → ℝ) (a₁ : ℝ) (q : ℝ)
variable (h₀ : a₁ = 1/4)
variable (h₁ : ∀ n, a (n + 1) = a₁ * q ^ n)
variable (S : ℕ → ℝ)
variable (h₂ : S n = a₁ * (1 - q ^ n) / (1 - q))

theorem geom_seq_sum_first_10_terms :
  a 1 = 1 / 4 →
  (a 3) * (a 5) = 4 * ((a 4) - 1) →
  S 10 = 1023 / 4 :=
by
  sorry

end geom_seq_sum_first_10_terms_l185_185829


namespace tetrahedron_faces_equal_l185_185439

theorem tetrahedron_faces_equal {a b c a' b' c' : ℝ} (h₁ : a + b + c = a + b' + c') (h₂ : a + b + c = a' + b + b') (h₃ : a + b + c = c' + c + a') :
  (a = a') ∧ (b = b') ∧ (c = c') :=
by
  sorry

end tetrahedron_faces_equal_l185_185439


namespace probability_digits_all_different_l185_185646

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l185_185646


namespace max_g_eq_25_l185_185773

-- Define the function g on positive integers.
def g : ℕ → ℤ
| n => if n < 12 then n + 14 else g (n - 7)

-- Prove that the maximum value of g is 25.
theorem max_g_eq_25 : ∀ n : ℕ, 1 ≤ n → g n ≤ 25 ∧ (∃ n : ℕ, 1 ≤ n ∧ g n = 25) := by
  sorry

end max_g_eq_25_l185_185773


namespace bellas_score_l185_185255

-- Definitions from the problem conditions
def n : Nat := 17
def x : Nat := 75
def new_n : Nat := n + 1
def y : Nat := 76

-- Assertion that Bella's score is 93
theorem bellas_score : (new_n * y) - (n * x) = 93 :=
by
  -- This is where the proof would go
  sorry

end bellas_score_l185_185255


namespace union_of_M_N_l185_185799

-- Define the sets M and N
def M : Set ℕ := {0, 2, 3}
def N : Set ℕ := {1, 3}

-- State the theorem to prove that M ∪ N = {0, 1, 2, 3}
theorem union_of_M_N : M ∪ N = {0, 1, 2, 3} :=
by
  sorry -- Proof goes here

end union_of_M_N_l185_185799


namespace cara_constant_speed_l185_185107

noncomputable def cara_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

theorem cara_constant_speed
  ( distance : ℕ := 120 )
  ( dan_speed : ℕ := 40 )
  ( dan_time_offset : ℕ := 1 ) :
  cara_speed distance (3 + dan_time_offset) = 30 := 
by
  -- skip proof
  sorry

end cara_constant_speed_l185_185107


namespace painters_time_l185_185973

-- Define the initial conditions
def n1 : ℕ := 3
def d1 : ℕ := 2
def W := n1 * d1
def n2 : ℕ := 2
def d2 := W / n2
def d_r := (3 * d2) / 4

-- Theorem statement
theorem painters_time (h : d_r = 9 / 4) : d_r = 9 / 4 := by
  sorry

end painters_time_l185_185973


namespace largest_x_l185_185788

theorem largest_x (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 := 
sorry

end largest_x_l185_185788


namespace avg_waiting_time_waiting_time_equivalence_l185_185481

-- The first rod receives an average of 3 bites in 6 minutes
def firstRodBites : ℝ := 3 / 6
-- The second rod receives an average of 2 bites in 6 minutes
def secondRodBites : ℝ := 2 / 6
-- Together, they receive an average of 5 bites in 6 minutes
def combinedBites : ℝ := firstRodBites + secondRodBites

-- We need to prove the average waiting time for the first bite
theorem avg_waiting_time : combinedBites = 5 / 6 → (1 / combinedBites) = 6 / 5 :=
by
  intro h
  rw h
  sorry

-- Convert 1.2 minutes into minutes and seconds
def minutes := 1
def seconds := 12

-- Prove the equivalence of waiting time in minutes and seconds
theorem waiting_time_equivalence : (6 / 5 = minutes + seconds / 60) :=
by
  simp [minutes, seconds]
  sorry

end avg_waiting_time_waiting_time_equivalence_l185_185481


namespace triangle_two_solutions_range_of_a_l185_185708

noncomputable def range_of_a (a b : ℝ) (A : ℝ) : Prop :=
b * Real.sin A < a ∧ a < b

theorem triangle_two_solutions_range_of_a (a : ℝ) (A : ℝ := Real.pi / 6) (b : ℝ := 2) :
  range_of_a a b A ↔ 1 < a ∧ a < 2 := by
sorry

end triangle_two_solutions_range_of_a_l185_185708


namespace cannot_determine_right_triangle_l185_185200

/-- Proof that the condition \(a^2 = 5\), \(b^2 = 12\), \(c^2 = 13\) cannot determine that \(\triangle ABC\) is a right triangle. -/
theorem cannot_determine_right_triangle (a b c : ℝ) (ha : a^2 = 5) (hb : b^2 = 12) (hc : c^2 = 13) : 
  ¬(a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := 
by
  sorry

end cannot_determine_right_triangle_l185_185200


namespace exists_m_inequality_l185_185206

theorem exists_m_inequality (a b : ℝ) (h : a > b) : ∃ m : ℝ, m < 0 ∧ a * m < b * m :=
by
  sorry

end exists_m_inequality_l185_185206


namespace ratio_of_areas_of_squares_l185_185591

theorem ratio_of_areas_of_squares (sideC sideD : ℕ) (hC : sideC = 45) (hD : sideD = 60) : 
  (sideC ^ 2) / (sideD ^ 2) = 9 / 16 := 
by
  sorry

end ratio_of_areas_of_squares_l185_185591


namespace find_percentage_ryegrass_in_seed_mixture_X_l185_185161

open Real

noncomputable def percentage_ryegrass_in_seed_mixture_X (R : ℝ) : Prop := 
  let proportion_X : ℝ := 2 / 3
  let percentage_Y_ryegrass : ℝ := 25 / 100
  let proportion_Y : ℝ := 1 / 3
  let final_percentage_ryegrass : ℝ := 35 / 100
  final_percentage_ryegrass = (R / 100 * proportion_X) + (percentage_Y_ryegrass * proportion_Y)

/-
  Given the conditions:
  - Seed mixture Y is 25 percent ryegrass.
  - A mixture of seed mixtures X (66.67% of the mixture) and Y (33.33% of the mixture) contains 35 percent ryegrass.

  Prove:
  The percentage of ryegrass in seed mixture X is 40%.
-/
theorem find_percentage_ryegrass_in_seed_mixture_X : 
  percentage_ryegrass_in_seed_mixture_X 40 := 
  sorry

end find_percentage_ryegrass_in_seed_mixture_X_l185_185161


namespace min_cost_for_boxes_l185_185154

def box_volume (l w h : ℕ) : ℕ := l * w * h
def total_boxes_needed (total_volume box_volume : ℕ) : ℕ := (total_volume + box_volume - 1) / box_volume
def total_cost (num_boxes : ℕ) (cost_per_box : ℚ) : ℚ := num_boxes * cost_per_box

theorem min_cost_for_boxes : 
  let l := 20
  let w := 20
  let h := 15
  let cost_per_box := (7 : ℚ) / 10
  let total_volume := 3060000
  let volume_box := box_volume l w h
  let num_boxes_needed := total_boxes_needed total_volume volume_box
  (num_boxes_needed = 510) → 
  (total_cost num_boxes_needed cost_per_box = 357) :=
by
  intros
  sorry

end min_cost_for_boxes_l185_185154


namespace number_of_hens_is_50_l185_185222

def number_goats : ℕ := 45
def number_camels : ℕ := 8
def number_keepers : ℕ := 15
def extra_feet : ℕ := 224

def total_heads (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  number_hens + number_goats + number_camels + number_keepers

def total_feet (number_hens number_goats number_camels number_keepers : ℕ) : ℕ :=
  2 * number_hens + 4 * number_goats + 4 * number_camels + 2 * number_keepers

theorem number_of_hens_is_50 (H : ℕ) :
  total_feet H number_goats number_camels number_keepers = (total_heads H number_goats number_camels number_keepers) + extra_feet → H = 50 :=
sorry

end number_of_hens_is_50_l185_185222


namespace line_circle_intersection_l185_185231

noncomputable def line_parametric (a t : ℝ) : ℝ × ℝ := (a + real.sqrt 3 * t, t)

def circle_cartesian (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

def distance_to_line (a x y : ℝ) : ℝ := real.abs (2 - a) / real.sqrt (1 + 3)

theorem line_circle_intersection (a : ℝ) :
  (∃ t, let ⟨x, y⟩ := line_parametric a t in circle_cartesian x y) ↔ -2 ≤ a ∧ a ≤ 6 :=
by
  sorry

end line_circle_intersection_l185_185231


namespace quadratic_condition_l185_185331

theorem quadratic_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1^2 - x2^2 = c^2 / a^2) ↔
  b^4 - c^4 = 4 * a * b^2 * c :=
sorry

end quadratic_condition_l185_185331


namespace ceil_sqrt_fraction_eq_neg2_l185_185176

theorem ceil_sqrt_fraction_eq_neg2 :
  (Int.ceil (-Real.sqrt (36 / 9))) = -2 :=
by
  sorry

end ceil_sqrt_fraction_eq_neg2_l185_185176


namespace ratio_of_squares_l185_185916

theorem ratio_of_squares : (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3 / 10 := 
sorry

end ratio_of_squares_l185_185916


namespace vans_needed_for_trip_l185_185291

theorem vans_needed_for_trip (total_people : ℕ) (van_capacity : ℕ) (h_total_people : total_people = 24) (h_van_capacity : van_capacity = 8) : ℕ :=
  let exact_vans := total_people / van_capacity
  let vans_needed := if total_people % van_capacity = 0 then exact_vans else exact_vans + 1
  have h_exact : exact_vans = 3 := by sorry
  have h_vans_needed : vans_needed = 4 := by sorry
  vans_needed

end vans_needed_for_trip_l185_185291


namespace product_fraction_simplification_l185_185863

theorem product_fraction_simplification :
  (1 - (1 / 3)) * (1 - (1 / 4)) * (1 - (1 / 5)) = 2 / 5 :=
by
  sorry

end product_fraction_simplification_l185_185863


namespace inequality_proof_l185_185262

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end inequality_proof_l185_185262


namespace total_games_single_elimination_l185_185888

theorem total_games_single_elimination (teams : ℕ) (h_teams : teams = 24)
  (preliminary_matches : ℕ) (h_preliminary_matches : preliminary_matches = 8)
  (preliminary_teams : ℕ) (h_preliminary_teams : preliminary_teams = 16)
  (idle_teams : ℕ) (h_idle_teams : idle_teams = 8)
  (main_draw_teams : ℕ) (h_main_draw_teams : main_draw_teams = 16) :
  (games : ℕ) -> games = 23 :=
by
  sorry

end total_games_single_elimination_l185_185888


namespace solution_inequality_1_range_of_a_l185_185802

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 2)

theorem solution_inequality_1 :
  {x : ℝ | f x < 3} = {x : ℝ | - (1/2) < x ∧ x < (5/2)} :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
by
  sorry

end solution_inequality_1_range_of_a_l185_185802


namespace value_of_star_l185_185877

theorem value_of_star :
  ∀ x : ℕ, 45 - (28 - (37 - (15 - x))) = 55 → x = 16 :=
by
  intro x
  intro h
  sorry

end value_of_star_l185_185877


namespace cos_seven_pi_six_l185_185782

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end cos_seven_pi_six_l185_185782


namespace cannot_achieve_80_cents_l185_185181

def is_possible_value (n : ℕ) : Prop :=
  ∃ (n_nickels n_dimes n_quarters n_half_dollars : ℕ), 
    n_nickels + n_dimes + n_quarters + n_half_dollars = 5 ∧
    5 * n_nickels + 10 * n_dimes + 25 * n_quarters + 50 * n_half_dollars = n

theorem cannot_achieve_80_cents : ¬ is_possible_value 80 :=
by sorry

end cannot_achieve_80_cents_l185_185181


namespace find_g_inv_84_l185_185215

def g (x : ℝ) : ℝ := 3 * x ^ 3 + 3

theorem find_g_inv_84 (x : ℝ) (h : g x = 84) : x = 3 :=
by 
  unfold g at h
  -- Begin proof steps here, but we will use sorry to denote placeholder 

  sorry

end find_g_inv_84_l185_185215


namespace largest_integer_less_than_100_with_remainder_7_divided_9_l185_185023

theorem largest_integer_less_than_100_with_remainder_7_divided_9 :
  ∃ x : ℕ, (∀ m : ℤ, x = 9 * m + 7 → 9 * m + 7 < 100) ∧ x = 97 :=
sorry

end largest_integer_less_than_100_with_remainder_7_divided_9_l185_185023


namespace inequality_solution_l185_185601

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-6) ∪ Set.Ioi (-2) :=
by
  sorry

end inequality_solution_l185_185601


namespace endpoint_of_vector_a_l185_185676

theorem endpoint_of_vector_a (x y : ℝ) (h : (x - 3) / -3 = (y + 1) / 4) : 
    x = 13 / 5 ∧ y = 2 / 5 :=
by sorry

end endpoint_of_vector_a_l185_185676


namespace some_number_is_five_l185_185789

theorem some_number_is_five (x : ℕ) (some_number : ℕ) (h1 : x = 5) (h2 : x / some_number + 3 = 4) : some_number = 5 := by
  sorry

end some_number_is_five_l185_185789


namespace part_I_part_II_l185_185819

noncomputable def triangle_conditions (a b c : ℝ) (B C : ℝ) : Prop :=
  (a > c) ∧
  (cos B = 1/3) ∧
  (a * c = 6) ∧
  (b = 3)

theorem part_I (a b c : ℝ) (B C : ℝ) (h : triangle_conditions a b c B C) :
  a = 3 ∧ cos C = 7/9 :=
sorry

theorem part_II (a b c : ℝ) (B C : ℝ) (h : triangle_conditions a b c B C) :
  cos (2 * C + real.pi / 3) = (17 - 56 * real.sqrt 6) / 162 :=
sorry

end part_I_part_II_l185_185819


namespace maximum_area_of_right_angled_triangle_l185_185685

noncomputable def max_area_right_angled_triangle (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 48) : ℕ := 
  max (a * b / 2) 288

theorem maximum_area_of_right_angled_triangle (a b c : ℕ) 
  (h1 : a^2 + b^2 = c^2)    -- Pythagorean theorem
  (h2 : a + b + c = 48)     -- Perimeter condition
  (h3 : 0 < a)              -- Positive integer side length condition
  (h4 : 0 < b)              -- Positive integer side length condition
  (h5 : 0 < c)              -- Positive integer side length condition
  : max_area_right_angled_triangle a b c h1 h2 = 288 := 
sorry

end maximum_area_of_right_angled_triangle_l185_185685


namespace factor_27x6_minus_512y6_sum_coeffs_is_152_l185_185855

variable {x y : ℤ}

theorem factor_27x6_minus_512y6_sum_coeffs_is_152 :
  ∃ a b c d e f g h j k : ℤ, 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) ∧ 
    (a + b + c + d + e + f + g + h + j + k = 152) := 
sorry

end factor_27x6_minus_512y6_sum_coeffs_is_152_l185_185855


namespace simplify_exponents_l185_185145

theorem simplify_exponents : (10^0.5) * (10^0.3) * (10^0.2) * (10^0.1) * (10^0.9) = 100 := 
by 
  sorry

end simplify_exponents_l185_185145


namespace find_line_eq_of_given_conditions_l185_185193

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * y + 5 = 0
def line_perpendicular (a b : ℝ) : Prop := a + b + 1 = 0
def is_center (x y : ℝ) : Prop := (x, y) = (0, 3)
def is_eq_of_line (x y : ℝ) : Prop := x - y + 3 = 0

theorem find_line_eq_of_given_conditions (x y : ℝ) (h1 : circle_eq x y) (h2 : line_perpendicular x y) (h3 : is_center x y) : is_eq_of_line x y :=
by
  sorry

end find_line_eq_of_given_conditions_l185_185193


namespace f_g_2_eq_36_l185_185563

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end f_g_2_eq_36_l185_185563


namespace symmetric_circle_equation_l185_185275

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), (x + 2) ^ 2 + y ^ 2 = 5 → (x - 2) ^ 2 + y ^ 2 = 5 :=
by 
  sorry

end symmetric_circle_equation_l185_185275


namespace abs_neg_three_halves_l185_185093

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l185_185093


namespace factor_polynomial_l185_185179

theorem factor_polynomial :
  ∀ u : ℝ, (u^4 - 81 * u^2 + 144) = (u^2 - 72) * (u - 3) * (u + 3) :=
by
  intro u
  -- Establish the polynomial and its factorization in Lean
  have h : u^4 - 81 * u^2 + 144 = (u^2 - 72) * (u - 3) * (u + 3) := sorry
  exact h

end factor_polynomial_l185_185179


namespace number_of_real_a_l185_185182

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end number_of_real_a_l185_185182


namespace peach_count_l185_185604

theorem peach_count (n : ℕ) : n % 4 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ 120 ≤ n ∧ n ≤ 150 → n = 142 :=
sorry

end peach_count_l185_185604


namespace arithmetic_sequence_a5_value_l185_185967

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_a2 : a 2 = 1)
  (h_a8 : a 8 = 2 * a 6 + a 4) : 
  a 5 = -1 / 2 :=
by
  sorry

end arithmetic_sequence_a5_value_l185_185967


namespace minimum_score_118_l185_185168

noncomputable def minimum_score (μ σ : ℝ) (p : ℝ) : ℝ :=
  sorry

theorem minimum_score_118 :
  minimum_score 98 10 (9100 / 400000) = 118 :=
by sorry

end minimum_score_118_l185_185168


namespace circle_C_equation_l185_185826

/-- Definitions of circles C1 and C2 -/
def circle_C1 := ∀ (x y : ℝ), (x - 4) ^ 2 + (y - 8) ^ 2 = 1
def circle_C2 := ∀ (x y : ℝ), (x - 6) ^ 2 + (y + 6) ^ 2 = 9

/-- Condition that the center of circle C is on the x-axis -/
def center_on_x_axis (x : ℝ) : Prop := ∃ y : ℝ, y = 0

/-- Bisection condition circle C bisects circumferences of circles C1 and C2 -/
def bisects_circumferences (x : ℝ) : Prop := 
  (∀ (y1 y2 : ℝ), ((x - 4) ^ 2 + (y1 - 8) ^ 2 + 1 = (x - 6) ^ 2 + (y2 + 6) ^ 2 + 9)) ∧ 
  center_on_x_axis x

/-- Statement to prove -/
theorem circle_C_equation : ∃ x y : ℝ, bisects_circumferences x ∧ (x^2 + y^2 = 81) := 
sorry

end circle_C_equation_l185_185826


namespace calculate_expression_l185_185656

theorem calculate_expression : 1^345 + 5^10 / 5^7 = 126 := by
  sorry

end calculate_expression_l185_185656


namespace sum_of_interior_angles_l185_185274

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1800) : 180 * ((n - 3) - 2) = 1260 :=
by
  sorry

end sum_of_interior_angles_l185_185274


namespace tan_ratio_l185_185385

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end tan_ratio_l185_185385


namespace sum_of_divisors_117_l185_185140

-- Defining the conditions in Lean
def n : ℕ := 117
def is_factorization : n = 3^2 * 13 := by rfl

-- The sum-of-divisors function can be defined based on the problem
def sum_of_divisors (n : ℕ) : ℕ :=
  (1 + 3 + 3^2) * (1 + 13)

-- Assertion of the correct answer
theorem sum_of_divisors_117 : sum_of_divisors n = 182 := by
  sorry

end sum_of_divisors_117_l185_185140


namespace correct_statements_l185_185032

theorem correct_statements (f : ℝ → ℝ)
  (h_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
  (h_pos : ∀ x : ℝ, x > 0 → f (x) > 0) :
  (f 0 ≠ 1) ∧
  (∀ x : ℝ, f (-x) = -f (x)) ∧
  ¬ (∀ x : ℝ, |f (x)| = |f (-x)|) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f (x₁) < f (x₂)) ∧
  ¬ (∀ x : ℝ, f (x) + 1 < f (x + 1)) :=
by
  sorry

end correct_statements_l185_185032


namespace find_number_l185_185508

-- Define the conditions as stated in the problem
def fifteen_percent_of_x_is_ninety (x : ℝ) : Prop :=
  (15 / 100) * x = 90

-- Define the theorem to prove that given the condition, x must be 600
theorem find_number (x : ℝ) (h : fifteen_percent_of_x_is_ninety x) : x = 600 :=
sorry

end find_number_l185_185508


namespace avg_waiting_time_is_1_point_2_minutes_l185_185473

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end avg_waiting_time_is_1_point_2_minutes_l185_185473


namespace AdultsNotWearingBlue_l185_185904

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end AdultsNotWearingBlue_l185_185904


namespace dante_eggs_l185_185918

theorem dante_eggs (E F : ℝ) (h1 : F = E / 2) (h2 : F + E = 90) : E = 60 :=
by
  sorry

end dante_eggs_l185_185918


namespace function_inequality_m_l185_185801

theorem function_inequality_m (m : ℝ) : (∀ x : ℝ, (1 / 2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) ↔ m ≥ (3 / 2) := sorry

end function_inequality_m_l185_185801


namespace john_tour_days_l185_185234

noncomputable def numberOfDaysInTourProgram (d e : ℕ) : Prop :=
  d * e = 800 ∧ (d + 7) * (e - 5) = 800

theorem john_tour_days :
  ∃ (d e : ℕ), numberOfDaysInTourProgram d e ∧ d = 28 :=
by
  sorry

end john_tour_days_l185_185234


namespace smallest_number_is_33_l185_185749

theorem smallest_number_is_33 
  (x : ℕ) 
  (h1 : ∀ y z, y = 2 * x → z = 4 * x → (x + y + z) / 3 = 77) : 
  x = 33 :=
by
  sorry

end smallest_number_is_33_l185_185749


namespace ratio_of_areas_l185_185131

-- Definitions of the side lengths of the triangles
noncomputable def sides_GHI : (ℕ × ℕ × ℕ) := (7, 24, 25)
noncomputable def sides_JKL : (ℕ × ℕ × ℕ) := (9, 40, 41)

-- Function to compute the area of a right triangle given its legs
def area_right_triangle (a b : ℕ) : ℚ :=
  (a * b) / 2

-- Areas of the triangles
noncomputable def area_GHI := area_right_triangle 7 24
noncomputable def area_JKL := area_right_triangle 9 40

-- Theorem: Ratio of the areas of the triangles GHI to JKL
theorem ratio_of_areas : (area_GHI / area_JKL) = 7 / 15 :=
by {
  sorry -- Proof is skipped as per instructions
}

end ratio_of_areas_l185_185131


namespace cups_of_baking_mix_planned_l185_185886

-- Definitions
def butter_per_cup := 2 -- 2 ounces of butter per 1 cup of baking mix
def coconut_oil_per_butter := 2 -- 2 ounces of coconut oil can substitute 2 ounces of butter
def butter_remaining := 4 -- Chef had 4 ounces of butter
def coconut_oil_used := 8 -- Chef used 8 ounces of coconut oil

-- Statement to be proven
theorem cups_of_baking_mix_planned : 
  (butter_remaining / butter_per_cup) + (coconut_oil_used / coconut_oil_per_butter) = 6 := 
by 
  sorry

end cups_of_baking_mix_planned_l185_185886


namespace rectangular_solid_surface_area_l185_185775

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_volume : a * b * c = 1001) :
  2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end rectangular_solid_surface_area_l185_185775


namespace r_exceeds_s_by_two_l185_185955

theorem r_exceeds_s_by_two (x y r s : ℝ) (h1 : 3 * x + 2 * y = 16) (h2 : 5 * x + 3 * y = 26)
  (hr : r = x) (hs : s = y) : r - s = 2 :=
by
  sorry

end r_exceeds_s_by_two_l185_185955


namespace boys_from_Pine_l185_185237

/-
We need to prove that the number of boys from Pine Middle School is 70
given the following conditions:
1. There were 150 students in total.
2. 90 were boys and 60 were girls.
3. 50 students were from Maple Middle School.
4. 100 students were from Pine Middle School.
5. 30 of the girls were from Maple Middle School.
-/
theorem boys_from_Pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h_total : total_students = 150) (h_boys : total_boys = 90)
  (h_girls : total_girls = 60) (h_maple : maple_students = 50)
  (h_pine : pine_students = 100) (h_maple_girls : maple_girls = 30) :
  total_boys - maple_students + maple_girls = 70 :=
by
  sorry

end boys_from_Pine_l185_185237


namespace simplify_expression_correct_l185_185083

def simplify_expression : ℚ :=
  15 * (7 / 10) * (1 / 9)

theorem simplify_expression_correct : simplify_expression = 7 / 6 :=
by
  unfold simplify_expression
  sorry

end simplify_expression_correct_l185_185083


namespace fish_weight_l185_185146

variables (W G T : ℕ)

-- Define the known conditions
axiom tail_weight : W = 1
axiom head_weight : G = W + T / 2
axiom torso_weight : T = G + W

-- Define the proof statement
theorem fish_weight : W + G + T = 8 :=
by
  sorry

end fish_weight_l185_185146


namespace max_angle_AFB_l185_185734

noncomputable def focus_of_parabola := (2, 0)
def parabola (x y : ℝ) := y^2 = 8 * x
def on_parabola (A B : ℝ × ℝ) := parabola A.1 A.2 ∧ parabola B.1 B.2
def condition (x1 x2 : ℝ) (AB : ℝ) := x1 + x2 + 4 = (2 * Real.sqrt 3 / 3) * AB

theorem max_angle_AFB (A B : ℝ × ℝ) (x1 x2 : ℝ) (AB : ℝ)
  (h1 : on_parabola A B)
  (h2 : condition x1 x2 AB)
  (hA : A.1 = x1)
  (hB : B.1 = x2) :
  ∃ θ, θ ≤ Real.pi * 2 / 3 := 
  sorry

end max_angle_AFB_l185_185734


namespace balls_in_boxes_wrong_positions_l185_185399

-- Define the total number of ways to place the balls in the boxes
def numberOfWays (n m : ℕ) : ℕ :=
  Nat.choose n m * 9

-- Prove that the total number of ways to place 9 balls into 9 boxes 
-- such that exactly 4 balls do not match the numbers of their respective boxes is 1134
theorem balls_in_boxes_wrong_positions :
  numberOfWays 9 5 = 1134 :=
by
  -- (5 balls placed correctly means exactly 4 balls placed incorrectly)
  unfold numberOfWays
  sorry

end balls_in_boxes_wrong_positions_l185_185399


namespace negation_of_universal_proposition_l185_185860

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l185_185860


namespace businessmen_drink_neither_l185_185655

theorem businessmen_drink_neither (n c t b : ℕ) 
  (h_n : n = 30) 
  (h_c : c = 15) 
  (h_t : t = 13) 
  (h_b : b = 7) : 
  n - (c + t - b) = 9 := 
  by
  sorry

end businessmen_drink_neither_l185_185655


namespace rook_reaches_upper_right_in_70_l185_185658

open Classical
open Set Function
open BigOperators nnreal

-- Define the problem parameters
def rook_grid : Type := Fin 8 × Fin 8

-- Defining the movement with equal probability on the grid
def move (pos : rook_grid) : rook_grid :=
by sorry -- Define movement as probabilistic

-- Define the expected time to reach the upper-right corner from a given position
noncomputable def expected_time (pos : rook_grid) : ℝ :=
by sorry -- Define recursive expected time

-- Prove the expected time to reach the upper-right corner from (0,0) is 70 minutes
theorem rook_reaches_upper_right_in_70 (pos : rook_grid) :
  pos = (0, 0) → expected_time pos = 70 :=
by sorry

end rook_reaches_upper_right_in_70_l185_185658


namespace Sues_necklace_total_beads_l185_185461

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end Sues_necklace_total_beads_l185_185461


namespace cost_price_article_l185_185621

variable (SP : ℝ := 21000)
variable (d : ℝ := 0.10)
variable (p : ℝ := 0.08)

theorem cost_price_article : (SP * (1 - d)) / (1 + p) = 17500 := by
  sorry

end cost_price_article_l185_185621


namespace polygon_is_hexagon_l185_185369

theorem polygon_is_hexagon (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  have hd : (n - 2) * 180 = 720 := by rw [h]
  have hn : n - 2 = 4 := by linarith
  rw [← hd, ← hn]
  linarith

end polygon_is_hexagon_l185_185369


namespace spherical_to_rectangular_correct_l185_185488

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) := by
  sorry

end spherical_to_rectangular_correct_l185_185488


namespace probability_of_minimal_arrangement_l185_185396

open Finset

def arrangements := {l : List ℕ | l.length = 9 ∧ l.toFinset = (finset.range 1 10)}

noncomputable def all_arrangements := arrangements.card / 2

noncomputable def minimal_arrangements := (2 ^ 6)

theorem probability_of_minimal_arrangement :
  minimal_arrangements / (all_arrangements : ℝ) = 1 / 315 := 
sorry

end probability_of_minimal_arrangement_l185_185396


namespace angle_terminal_side_l185_185022

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * (180 / Real.pi)

theorem angle_terminal_side :
  ∃ k : ℤ, rad_to_deg (π / 12) + 360 * k = 375 :=
sorry

end angle_terminal_side_l185_185022


namespace mod_inverse_9_mod_23_l185_185024

theorem mod_inverse_9_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (9 * a) % 23 = 1 :=
by
  use 18
  sorry

end mod_inverse_9_mod_23_l185_185024


namespace value_of_t_eq_3_over_4_l185_185373

-- Define the values x and y as per the conditions
def x (t : ℝ) : ℝ := 1 - 2 * t
def y (t : ℝ) : ℝ := 2 * t - 2

-- Statement only, proof is omitted using sorry
theorem value_of_t_eq_3_over_4 (t : ℝ) (h : x t = y t) : t = 3 / 4 :=
by
  sorry

end value_of_t_eq_3_over_4_l185_185373


namespace increasing_function_cond_l185_185793

theorem increasing_function_cond (f : ℝ → ℝ)
  (h : ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0) :
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end increasing_function_cond_l185_185793


namespace determine_range_of_a_l185_185680

-- Define the function f(x).
def f (a x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Define the derivative of f(x).
def f' (a x : ℝ) : ℝ := 1 / x + 2 * a * x - 3

-- Define the function g(x) as used in the solution.
def g (x : ℝ) : ℝ := (3 / (2 * x)) - (1 / (2 * x^2))

-- Define the interval.
def I : set ℝ := {x : ℝ | 1/2 < x ∧ x < 3}

-- Define the condition for f(x) to be monotonically increasing on the interval I.
def monotonic_increasing_on (a : ℝ) : Prop :=
  ∀ x ∈ I, f'(a, x) ≥ 0

-- State the theorem.
theorem determine_range_of_a :
  ∀ a : ℝ, monotonic_increasing_on a ↔ a ∈ set.Ici (9/8) :=
begin
  sorry
end

end determine_range_of_a_l185_185680


namespace log_property_l185_185039

theorem log_property (x : ℝ) (h₁ : Real.log x > 0) (h₂ : x > 1) : x > Real.exp 1 := by 
  sorry

end log_property_l185_185039


namespace sum_of_numbers_l185_185235

theorem sum_of_numbers (x y : ℕ) (hx : 100 ≤ x ∧ x < 1000) (hy : 1000 ≤ y ∧ y < 10000) (h : 10000 * x + y = 12 * x * y) :
  x + y = 1083 :=
sorry

end sum_of_numbers_l185_185235


namespace evalCeilingOfNegativeSqrt_l185_185177

noncomputable def ceiling_of_negative_sqrt : ℤ :=
  Int.ceil (-(Real.sqrt (36 / 9)))

theorem evalCeilingOfNegativeSqrt : ceiling_of_negative_sqrt = -2 := by
  sorry

end evalCeilingOfNegativeSqrt_l185_185177


namespace repeating_decimal_as_fraction_l185_185778

theorem repeating_decimal_as_fraction :
  (3 + 45 / 99) = 38 / 11 :=
by
  -- Here you would perform the necessary steps and computations to show the equivalency.
  sorry

end repeating_decimal_as_fraction_l185_185778


namespace investment_period_l185_185308

theorem investment_period (P A : ℝ) (r n t : ℝ)
  (hP : P = 4000)
  (hA : A = 4840.000000000001)
  (hr : r = 0.10)
  (hn : n = 1)
  (hC : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 := by
-- Adding a sorry to skip the actual proof.
sorry

end investment_period_l185_185308


namespace problem_statement_l185_185809

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 12) : a^3 + 1/a^3 = 18 * Real.sqrt 3 :=
by
  -- We'll skip the proof as per instruction
  sorry

end problem_statement_l185_185809


namespace total_time_for_12000_dolls_l185_185977

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end total_time_for_12000_dolls_l185_185977


namespace proof_equation_of_line_l185_185412
   
   -- Define the point P
   structure Point where
     x : ℝ
     y : ℝ
     
   -- Define conditions
   def passesThroughP (line : ℝ → ℝ → Prop) : Prop :=
     line 2 (-1)
     
   def interceptRelation (line : ℝ → ℝ → Prop) : Prop :=
     ∃ a : ℝ, a ≠ 0 ∧ (∀ x y, line x y ↔ (x / a + y / (2 * a) = 1))
   
   -- Define the line equation
   def line_equation (line : ℝ → ℝ → Prop) : Prop :=
     passesThroughP line ∧ interceptRelation line
     
   -- The final statement
   theorem proof_equation_of_line (line : ℝ → ℝ → Prop) :
     line_equation line →
     (∀ x y, line x y ↔ (2 * x + y = 3)) ∨ (∀ x y, line x y ↔ (x + 2 * y = 0)) :=
   by
     sorry
   
end proof_equation_of_line_l185_185412


namespace range_of_a_l185_185920

noncomputable def matrix_det_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem range_of_a : 
  {a : ℝ | matrix_det_2x2 (a^2) 1 3 2 < matrix_det_2x2 a 0 4 1} = {a : ℝ | -1 < a ∧ a < 3/2} :=
by
  sorry

end range_of_a_l185_185920


namespace probability_of_log_ge_than_1_l185_185453

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end probability_of_log_ge_than_1_l185_185453


namespace perpendicular_tangent_inequality_l185_185603

variable {A B C : Type} 

-- Definitions according to conditions in part a)
def isAcuteAngledTriangle (a b c : Type) : Prop :=
  -- A triangle being acute-angled in Euclidean geometry
  sorry

def triangleArea (a b c : Type) : ℝ :=
  -- Definition of the area of a triangle
  sorry

def perpendicularLengthToLine (point line : Type) : ℝ :=
  -- Length of the perpendicular from a point to a line
  sorry

def tangentOfAngleA (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle A in the triangle
  sorry

def tangentOfAngleB (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle B in the triangle
  sorry

def tangentOfAngleC (a b c : Type) : ℝ :=
  -- Definition of the tangent of angle C in the triangle
  sorry

theorem perpendicular_tangent_inequality (a b c line : Type) 
  (ht : isAcuteAngledTriangle a b c)
  (u := perpendicularLengthToLine a line)
  (v := perpendicularLengthToLine b line)
  (w := perpendicularLengthToLine c line):
  u^2 * tangentOfAngleA a b c + v^2 * tangentOfAngleB a b c + w^2 * tangentOfAngleC a b c ≥ 
  2 * triangleArea a b c :=
sorry

end perpendicular_tangent_inequality_l185_185603


namespace cone_base_circumference_l185_185157

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (C : ℝ) : 
  r = 5 → θ = 300 → C = (θ / 360) * (2 * Real.pi * r) → C = (25 / 3) * Real.pi :=
by
  sorry

end cone_base_circumference_l185_185157


namespace fifteen_percent_of_x_is_ninety_l185_185501

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l185_185501


namespace tan_quadruple_angle_l185_185560

theorem tan_quadruple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 :=
sorry

end tan_quadruple_angle_l185_185560


namespace avg_payment_correct_l185_185630

def first_payment : ℕ := 410
def additional_amount : ℕ := 65
def num_first_payments : ℕ := 8
def num_remaining_payments : ℕ := 44
def total_installments : ℕ := num_first_payments + num_remaining_payments

def total_first_payments : ℕ := num_first_payments * first_payment
def remaining_payment : ℕ := first_payment + additional_amount
def total_remaining_payments : ℕ := num_remaining_payments * remaining_payment

def total_payment : ℕ := total_first_payments + total_remaining_payments
def average_payment : ℚ := total_payment / total_installments

theorem avg_payment_correct : average_payment = 465 := by
  sorry

end avg_payment_correct_l185_185630


namespace total_animals_sighted_l185_185019

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end total_animals_sighted_l185_185019


namespace first_digit_base9_650_l185_185283

theorem first_digit_base9_650 : ∃ d : ℕ, 
  d = 8 ∧ (∃ k : ℕ, 650 = d * 9^2 + k ∧ k < 9^2) :=
by {
  sorry
}

end first_digit_base9_650_l185_185283


namespace time_to_cover_escalator_l185_185467

def escalator_speed := 11 -- ft/sec
def escalator_length := 126 -- feet
def person_speed := 3 -- ft/sec

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 9 := by
  sorry

end time_to_cover_escalator_l185_185467


namespace binary_digit_difference_l185_185747

theorem binary_digit_difference (n1 n2 : ℕ) (h1 : n1 = 300) (h2 : n2 = 1400) : 
  (nat.bit_length n2 - nat.bit_length n1) = 2 := by
  sorry

end binary_digit_difference_l185_185747


namespace sue_necklace_total_beads_l185_185463

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sue_necklace_total_beads_l185_185463


namespace quiz_scores_dropped_students_l185_185102

theorem quiz_scores_dropped_students (T S : ℝ) :
  T = 30 * 60.25 →
  T - S = 26 * 63.75 →
  S = 150 :=
by
  intros hT h_rem
  -- Additional steps would be implemented here.
  sorry

end quiz_scores_dropped_students_l185_185102


namespace captain_smollett_problem_l185_185008

/-- 
Given the captain's age, the number of children he has, and the length of his schooner, 
prove that the unique solution to the product condition is age = 53 years, children = 6, 
and length = 101 feet, under the given constraints.
-/
theorem captain_smollett_problem
  (age children length : ℕ)
  (h1 : age < 100)
  (h2 : children > 3)
  (h3 : age * children * length = 32118) : age = 53 ∧ children = 6 ∧ length = 101 :=
by {
  -- Proof will be filled in later
  sorry
}

end captain_smollett_problem_l185_185008


namespace isosceles_trapezoid_legs_squared_l185_185066

theorem isosceles_trapezoid_legs_squared
  (A B C D : Type)
  (AB CD AD BC : ℝ)
  (isosceles_trapezoid : AB = 50 ∧ CD = 14 ∧ AD = BC)
  (circle_tangent : ∃ M : ℝ, M = 25 ∧ ∀ x : ℝ, MD = 7 ↔ AD = x ∧ BC = x) :
  AD^2 = 800 := 
by
  sorry

end isosceles_trapezoid_legs_squared_l185_185066


namespace chords_triangle_count_l185_185397

-- Defining the problem constraints as a Lean 4 statement
theorem chords_triangle_count (h₁ : ∃ points : Finset ℝ, points.card = 9)
  (h₂ : ∀ (p1 p2 p3 : ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3)
  : ∃ n : ℕ, n = 315500 :=
by
  sorry

end chords_triangle_count_l185_185397


namespace monthly_pool_cost_is_correct_l185_185063

def cost_of_cleaning : ℕ := 150
def tip_percentage : ℕ := 10
def number_of_cleanings_in_a_month : ℕ := 30 / 3
def cost_of_chemicals_per_use : ℕ := 200
def number_of_chemical_uses_in_a_month : ℕ := 2

def monthly_cost_of_pool : ℕ :=
  let cost_per_cleaning := cost_of_cleaning + (cost_of_cleaning * tip_percentage / 100)
  let total_cleaning_cost := number_of_cleanings_in_a_month * cost_per_cleaning
  let total_chemical_cost := number_of_chemical_uses_in_a_month * cost_of_chemicals_per_use
  total_cleaning_cost + total_chemical_cost

theorem monthly_pool_cost_is_correct : monthly_cost_of_pool = 2050 :=
by
  sorry

end monthly_pool_cost_is_correct_l185_185063


namespace find_p_plus_q_l185_185609

noncomputable def calculate_p_plus_q (DE EF FD WX : ℕ) (Area : ℕ → ℝ) : ℕ :=
  let s := (DE + EF + FD) / 2
  let triangle_area := (Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))) / 2
  let delta := triangle_area / (225 * WX)
  let gcd := Nat.gcd 41 225
  let p := 41 / gcd
  let q := 225 / gcd
  p + q

theorem find_p_plus_q : calculate_p_plus_q 13 30 19 15 (fun θ => 30 * θ - (41 / 225) * θ^2) = 266 := by
  sorry

end find_p_plus_q_l185_185609


namespace files_remaining_correct_l185_185891

-- Definitions for the original number of files
def music_files_original : ℕ := 4
def video_files_original : ℕ := 21
def document_files_original : ℕ := 12
def photo_files_original : ℕ := 30
def app_files_original : ℕ := 7

-- Definitions for the number of deleted files
def video_files_deleted : ℕ := 15
def document_files_deleted : ℕ := 10
def photo_files_deleted : ℕ := 18
def app_files_deleted : ℕ := 3

-- Definitions for the remaining number of files
def music_files_remaining : ℕ := music_files_original
def video_files_remaining : ℕ := video_files_original - video_files_deleted
def document_files_remaining : ℕ := document_files_original - document_files_deleted
def photo_files_remaining : ℕ := photo_files_original - photo_files_deleted
def app_files_remaining : ℕ := app_files_original - app_files_deleted

-- The proof problem statement
theorem files_remaining_correct : 
  music_files_remaining + video_files_remaining + document_files_remaining + photo_files_remaining + app_files_remaining = 28 :=
by
  rw [music_files_remaining, video_files_remaining, document_files_remaining, photo_files_remaining, app_files_remaining]
  exact rfl


end files_remaining_correct_l185_185891


namespace minneapolis_st_louis_temperature_l185_185310

theorem minneapolis_st_louis_temperature (N M L : ℝ) (h1 : M = L + N)
                                         (h2 : M - 7 = L + N - 7)
                                         (h3 : L + 5 = L + 5)
                                         (h4 : (M - 7) - (L + 5) = |(L + N - 7) - (L + 5)|) :
  ∃ (N1 N2 : ℝ), (|N - 12| = 4) ∧ N1 = 16 ∧ N2 = 8 ∧ N1 * N2 = 128 :=
by {
  sorry
}

end minneapolis_st_louis_temperature_l185_185310


namespace zyka_expense_increase_l185_185432

theorem zyka_expense_increase (C_k C_c : ℝ) (h1 : 0.5 * C_k = 0.2 * C_c) : 
  (((1.2 * C_c) - C_c) / C_c) * 100 = 20 := by
  sorry

end zyka_expense_increase_l185_185432


namespace probability_of_three_heads_in_eight_tosses_l185_185759

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end probability_of_three_heads_in_eight_tosses_l185_185759


namespace relationship_between_a_b_c_l185_185336

-- Definitions as given in the problem
def a : ℝ := 2 ^ 0.5
def b : ℝ := Real.log 5 / Real.log 2
def c : ℝ := Real.log 10 / Real.log 4

-- Statement we need to prove
theorem relationship_between_a_b_c : a < c ∧ c < b := by
  sorry

end relationship_between_a_b_c_l185_185336


namespace exponent_multiplication_l185_185596

variable (a : ℝ) (m : ℤ)

theorem exponent_multiplication (a : ℝ) (m : ℤ) : a^(2 * m + 2) = a^(2 * m) * a^2 := 
sorry

end exponent_multiplication_l185_185596


namespace gauravi_walks_4500m_on_tuesday_l185_185671

def initial_distance : ℕ := 500
def increase_per_day : ℕ := 500
def target_distance : ℕ := 4500

def distance_after_days (n : ℕ) : ℕ :=
  initial_distance + n * increase_per_day

def day_of_week_after (start_day : ℕ) (n : ℕ) : ℕ :=
  (start_day + n) % 7

def monday : ℕ := 0 -- Represent Monday as 0

theorem gauravi_walks_4500m_on_tuesday :
  distance_after_days 8 = target_distance ∧ day_of_week_after monday 8 = 2 :=
by 
  sorry

end gauravi_walks_4500m_on_tuesday_l185_185671


namespace intersection_points_on_hyperbola_l185_185936

theorem intersection_points_on_hyperbola (p x y : ℝ) :
  (2*p*x - 3*y - 4*p = 0) ∧ (4*x - 3*p*y - 6 = 0) → 
  (∃ a b : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1) :=
by
  intros h
  sorry

end intersection_points_on_hyperbola_l185_185936


namespace algebraic_simplification_l185_185150

variables (a b : ℝ)

theorem algebraic_simplification (h : a > b ∧ b > 0) : 
  ((a + b) / ((Real.sqrt a - Real.sqrt b)^2)) * 
  (((3 * a * b - b * Real.sqrt (a * b) + a * Real.sqrt (a * b) - 3 * b^2) / 
    (1/2 * Real.sqrt (1/4 * ((a / b + b / a)^2) - 1)) + 
   (4 * a * b * Real.sqrt a + 9 * a * b * Real.sqrt b - 9 * b^2 * Real.sqrt a) / 
   (3/2 * Real.sqrt b - 2 * Real.sqrt a))) 
  = -2 * b * (a + 3 * Real.sqrt (a * b)) :=
sorry

end algebraic_simplification_l185_185150


namespace vans_needed_l185_185173

-- Definitions of conditions
def students : Nat := 2
def adults : Nat := 6
def capacity_per_van : Nat := 4

-- Main theorem to prove
theorem vans_needed : (students + adults) / capacity_per_van = 2 := by
  sorry

end vans_needed_l185_185173


namespace problem_correct_statements_l185_185011

def T (a b x y : ℚ) : ℚ := a * x * y + b * x - 4

theorem problem_correct_statements (a b : ℚ) (h₁ : T a b 2 1 = 2) (h₂ : T a b (-1) 2 = -8) :
  (a = 1 ∧ b = 2) ∧
  (∀ m n : ℚ, T 1 2 m n = 0 ∧ n ≠ -2 → m = 4 / (n + 2)) ∧
  ¬ (∃ m n : ℤ, T 1 2 m n = 0 ∧ n ≠ -2 ∧ m + n = 3) ∧
  (∀ k x y : ℚ, T 1 2 (k * x) y = T 1 2 (k * x) y → y = -2) ∧
  (∀ k x y : ℚ, x ≠ y → T 1 2 (k * x) y = T 1 2 (k * y) x → k = 0) :=
by
  sorry

end problem_correct_statements_l185_185011


namespace triangle_has_three_altitudes_l185_185352

-- Assuming a triangle in ℝ² space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Definition of an altitude in the context of Lean
def altitude (T : Triangle) (p : ℝ × ℝ) := 
  ∃ (a : ℝ) (b : ℝ), T.A.1 * p.1 + T.A.2 * p.2 = a * p.1 + b -- Placeholder, real definition of altitude may vary

-- Prove that a triangle has exactly 3 altitudes
theorem triangle_has_three_altitudes (T : Triangle) : ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
  altitude T p₁ ∧ altitude T p₂ ∧ altitude T p₃ :=
sorry

end triangle_has_three_altitudes_l185_185352


namespace apple_production_total_l185_185900

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l185_185900


namespace tan_of_angle_in_second_quadrant_l185_185037

theorem tan_of_angle_in_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.cos (π / 2 - α) = 4 / 5) : Real.tan α = -4 / 3 :=
by
  sorry

end tan_of_angle_in_second_quadrant_l185_185037


namespace dance_arrangement_possible_l185_185014

variables {B G : Type}
variables (boys : Fin 10 → B) (girls : Fin 10 → G)

structure GirlAttrs :=
(beauty : ℕ) 
(intelligence : ℕ)

variables (beauty intelligence : G → ℕ)
variables (initial_pairing second_pairing : Fin 10 → Fin 10)

def valid_initial_pairing : Prop :=
  ∀ i : Fin 10, initial_pairing i = i

def valid_second_pairing : Prop :=
  ∀ i : Fin 10, 
    (i < 9 → beauty (girls (second_pairing i)) > beauty (girls (initial_pairing i)) ∧ intelligence (girls (second_pairing i)) > intelligence (girls (initial_pairing i))) ∧
    (i = 9 → second_pairing i = 10 - 1) ∧ 
    (i = 10 - 1 → second_pairing i = 0)

def ratio_greater_beauty_intelligence : Prop :=
  (∑ i in Finset.range 9, if beauty (girls (second_pairing i)) > beauty (girls (initial_pairing i)) ∧ intelligence (girls (second_pairing i)) > intelligence (girls (initial_pairing i)) then 1 else 0) ≥ (8 * 1)

theorem dance_arrangement_possible (boys : Fin 10 → B) (girls : Fin 10 → G) 
  [∀ i : Fin 10, valid_initial_pairing initial_pairing] 
  [∀ i : Fin 10, valid_second_pairing second_pairing] :
  ratio_greater_beauty_intelligence beauty intelligence initial_pairing second_pairing :=
sorry

end dance_arrangement_possible_l185_185014


namespace train_length_approx_l185_185433

noncomputable def length_of_train (distance_km : ℝ) (time_min : ℝ) (time_sec : ℝ) : ℝ :=
  let distance_m := distance_km * 1000 -- Convert km to meters
  let time_s := time_min * 60 -- Convert min to seconds
  let speed := distance_m / time_s -- Speed in meters/second
  speed * time_sec -- Length of train in meters

theorem train_length_approx :
  length_of_train 10 15 10 = 111.1 :=
by
  sorry

end train_length_approx_l185_185433


namespace choose_two_fruits_l185_185958

theorem choose_two_fruits :
  let n := 5
  let k := 2
  Nat.choose n k = 10 := 
by 
  let n := 5
  let k := 2
  sorry

end choose_two_fruits_l185_185958


namespace math_problem_l185_185663

theorem math_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^b + 3 = b^a) (h4 : 3 * a^b = b^a + 13) : 
  (a = 2) ∧ (b = 3) :=
sorry

end math_problem_l185_185663


namespace matrix_inverse_problem_l185_185814

variables {α : Type*} [field α] [decidable_eq α] {n : ℕ}
variables (B : matrix (fin n) (fin n) α)
variables (I : matrix (fin n) (fin n) α) [invertible B]

-- Conditions 
def matrix_condition1 : Prop := (B - 3 • I) ⬝ (B - 5 • I) = 0

-- Theorem we aim to prove
theorem matrix_inverse_problem (h_inv : invertible B) (h_cond : matrix_condition1 B I):
  B + 15 • (⅟B) = 8 • I :=
sorry

end matrix_inverse_problem_l185_185814


namespace f_g_2_eq_36_l185_185564

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end f_g_2_eq_36_l185_185564


namespace range_of_a_l185_185597

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → (2 * x + 2 * (a - 1)) ≤ 0) → a ≤ -3 :=
by
  sorry

end range_of_a_l185_185597


namespace cricket_target_runs_l185_185828

def target_runs (first_10_overs_run_rate remaining_40_overs_run_rate : ℝ) : ℝ :=
  10 * first_10_overs_run_rate + 40 * remaining_40_overs_run_rate

theorem cricket_target_runs : target_runs 4.2 6 = 282 := by
  sorry

end cricket_target_runs_l185_185828


namespace exists_equidistant_point_l185_185869

-- Define three points A, B, and C in 2D space
variables {A B C P: ℝ × ℝ}

-- Assume the points A, B, and C are not collinear
def not_collinear (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) ≠ (C.1 - A.1) * (B.2 - A.2)

-- Define the concept of a point being equidistant from three given points
def equidistant (P A B C : ℝ × ℝ) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C

-- Define the intersection of the perpendicular bisectors of the sides of the triangle formed by A, B, and C
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- placeholder for the actual construction

-- The main theorem statement: If A, B, and C are not collinear, then there exists a unique point P that is equidistant from A, B, and C
theorem exists_equidistant_point (h: not_collinear A B C) :
  ∃! P, equidistant P A B C := 
sorry

end exists_equidistant_point_l185_185869


namespace find_g_l185_185982

noncomputable def g (x : ℝ) : ℝ := 2 - 4 * x

theorem find_g :
  g 0 = 2 ∧ (∀ x y : ℝ, g (x * y) = g ((3 * x ^ 2 + y ^ 2) / 4) + 3 * (x - y) ^ 2) → ∀ x : ℝ, g x = 2 - 4 * x :=
by
  sorry

end find_g_l185_185982


namespace multiple_of_spending_on_wednesday_l185_185939

-- Definitions based on the conditions
def monday_spending : ℤ := 60
def tuesday_spending : ℤ := 4 * monday_spending
def total_spending : ℤ := 600

-- Problem to prove
theorem multiple_of_spending_on_wednesday (x : ℤ) : 
  monday_spending + tuesday_spending + x * monday_spending = total_spending → 
  x = 5 := by
  sorry

end multiple_of_spending_on_wednesday_l185_185939


namespace fifteen_percent_of_x_is_ninety_l185_185503

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l185_185503


namespace programs_produce_same_output_l185_185725

def sum_program_a : ℕ :=
  let S := (Finset.range 1000).sum (λ i => i + 1)
  S

def sum_program_b : ℕ :=
  let S := (Finset.range 1000).sum (λ i => 1000 - i)
  S

theorem programs_produce_same_output :
  sum_program_a = sum_program_b := by
  sorry

end programs_produce_same_output_l185_185725


namespace smallest_number_is_a_l185_185003

def smallest_number_among_options : ℤ :=
  let a: ℤ := -3
  let b: ℤ := 0
  let c: ℤ := -(-1)
  let d: ℤ := (-1)^2
  min a (min b (min c d))

theorem smallest_number_is_a : smallest_number_among_options = -3 :=
  by
    sorry

end smallest_number_is_a_l185_185003


namespace percent_difference_l185_185957

theorem percent_difference : 
  let a := 0.60 * 50
  let b := 0.45 * 30
  a - b = 16.5 :=
by
  let a := 0.60 * 50
  let b := 0.45 * 30
  sorry

end percent_difference_l185_185957


namespace abs_neg_frac_l185_185097

theorem abs_neg_frac : abs (-3 / 2) = 3 / 2 := 
by sorry

end abs_neg_frac_l185_185097


namespace solve_system_l185_185407

theorem solve_system : ∃ s t : ℝ, (11 * s + 7 * t = 240) ∧ (s = 1 / 2 * t + 3) ∧ (t = 414 / 25) :=
by
  sorry

end solve_system_l185_185407


namespace ValleyFalcons_all_items_l185_185026

noncomputable def num_fans_receiving_all_items (capacity : ℕ) (tshirt_interval : ℕ) 
  (cap_interval : ℕ) (wristband_interval : ℕ) : ℕ :=
  (capacity / Nat.lcm (Nat.lcm tshirt_interval cap_interval) wristband_interval)

theorem ValleyFalcons_all_items:
  num_fans_receiving_all_items 3000 50 25 60 = 10 :=
by
  -- This is where the mathematical proof would go
  sorry

end ValleyFalcons_all_items_l185_185026


namespace fraction_of_second_year_given_not_third_year_l185_185057

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_second_year_given_not_third_year_l185_185057


namespace triangle_BC_length_l185_185817

theorem triangle_BC_length (A B C X : Type) 
  (AB AC : ℕ) (BX CX BC : ℕ)
  (h1 : AB = 100)
  (h2 : AC = 121)
  (h3 : ∃ x y : ℕ, x = BX ∧ y = CX ∧ AB = 100 ∧ x + y = BC)
  (h4 : x * y = 31 * 149 ∧ x + y = 149) :
  BC = 149 := 
by
  sorry

end triangle_BC_length_l185_185817


namespace min_value_of_reciprocals_l185_185070

theorem min_value_of_reciprocals (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (1 / m + 1 / n) = 2 :=
sorry

end min_value_of_reciprocals_l185_185070


namespace polynomial_coeffs_identity_l185_185786

theorem polynomial_coeffs_identity : 
  (∀ a b c : ℝ, (2 * x^4 + x^3 - 41 * x^2 + 83 * x - 45 = 
                (a * x^2 + b * x + c) * (x^2 + 4 * x + 9))
                  → a = 2 ∧ b = -7 ∧ c = -5) :=
by
  intros a b c h
  have h₁ : a = 2 := 
    sorry-- prove that a = 2
  have h₂ : b = -7 := 
    sorry-- prove that b = -7
  have h₃ : c = -5 := 
    sorry-- prove that c = -5
  exact ⟨h₁, h₂, h₃⟩

end polynomial_coeffs_identity_l185_185786


namespace tan_ratio_l185_185384

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end tan_ratio_l185_185384


namespace solve_abs_eq_2x_plus_1_l185_185945

theorem solve_abs_eq_2x_plus_1 (x : ℝ) (h : |x| = 2 * x + 1) : x = -1 / 3 :=
by 
  sorry

end solve_abs_eq_2x_plus_1_l185_185945


namespace original_mixture_volume_l185_185629

theorem original_mixture_volume (x : ℝ) (h1 : 0.20 * x / (x + 3) = 1 / 6) : x = 15 :=
  sorry

end original_mixture_volume_l185_185629


namespace polynomial_divisible_by_five_l185_185249

open Polynomial

theorem polynomial_divisible_by_five
  (a b c d m : ℤ)
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ (n : ℤ), (d * n^3 + c * n^2 + b * n + a) % 5 = 0 := 
  sorry

end polynomial_divisible_by_five_l185_185249


namespace algae_coverage_double_l185_185099

theorem algae_coverage_double (algae_cov : ℕ → ℝ) (h1 : ∀ n : ℕ, algae_cov (n + 2) = 2 * algae_cov n)
  (h2 : algae_cov 24 = 1) : algae_cov 18 = 0.125 :=
by
  sorry

end algae_coverage_double_l185_185099


namespace factorize_x9_minus_512_l185_185910

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l185_185910


namespace parabola_and_line_solutions_l185_185684

-- Definition of the parabola with its focus
def parabola_with_focus (p : ℝ) : Prop :=
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (x : ℝ), x = 1 / 2)

-- Definitions of conditions for intersection and orthogonal vectors
def line_intersecting_parabola (slope t : ℝ) (p : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), 
  (y1 = 2 * x1 + t) ∧ (y2 = 2 * x2 + t) ∧
  (y1^2 = 2 * x1) ∧ (y2^2 = 2 * x2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧
  (x1 * x2 = (t^2) / 4) ∧ (x1 * x2 + y1 * y2 = 0)

-- Lean statement for the proof problem
theorem parabola_and_line_solutions :
  ∀ p t : ℝ, 
  parabola_with_focus p → 
  (line_intersecting_parabola 2 t p → t = -4)
  → p = 1 :=
by
  intros p t h_parabola h_line
  sorry

end parabola_and_line_solutions_l185_185684


namespace shorter_leg_length_l185_185859

theorem shorter_leg_length (m h x : ℝ) (H1 : m = 15) (H2 : h = 3 * x) (H3 : m = 0.5 * h) : x = 10 :=
by
  sorry

end shorter_leg_length_l185_185859


namespace apple_production_total_l185_185898

def apples_first_year := 40
def apples_second_year := 8 + 2 * apples_first_year
def apples_third_year := apples_second_year - (1 / 4) * apples_second_year
def total_apples := apples_first_year + apples_second_year + apples_third_year

-- Math proof problem statement
theorem apple_production_total : total_apples = 194 :=
  sorry

end apple_production_total_l185_185898


namespace fifteen_percent_of_x_is_ninety_l185_185516

theorem fifteen_percent_of_x_is_ninety (x : ℝ) (h : (15 / 100) * x = 90) : x = 600 :=
sorry

end fifteen_percent_of_x_is_ninety_l185_185516


namespace fifteen_percent_of_x_is_ninety_l185_185500

theorem fifteen_percent_of_x_is_ninety :
  ∃ (x : ℝ), (15 / 100) * x = 90 ↔ x = 600 :=
by
  sorry

end fifteen_percent_of_x_is_ninety_l185_185500


namespace difference_digits_in_base2_l185_185746

def binaryDigitCount (n : Nat) : Nat := Nat.log2 n + 1

theorem difference_digits_in_base2 : binaryDigitCount 1400 - binaryDigitCount 300 = 2 :=
by
  sorry

end difference_digits_in_base2_l185_185746


namespace students_identified_chess_or_basketball_l185_185226

theorem students_identified_chess_or_basketball (total_students : ℕ) (p_basketball : ℝ) (p_chess : ℝ) (p_soccer : ℝ) :
    total_students = 250 → 
    p_basketball = 0.4 → 
    p_chess = 0.1 →
    p_soccer = 0.28 → 
    (p_basketball * total_students + p_chess * total_students) = 125 :=
begin 
  intros h1 h2 h3 h4,
  sorry
end

end students_identified_chess_or_basketball_l185_185226


namespace total_animals_seen_l185_185017

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end total_animals_seen_l185_185017


namespace find_borrowed_amount_l185_185765

noncomputable def borrowed_amount (P : ℝ) : Prop :=
  let interest_paid := P * (4 / 100) * 2
  let interest_earned := P * (6 / 100) * 2
  let total_gain := 120 * 2
  interest_earned - interest_paid = total_gain

theorem find_borrowed_amount : ∃ P : ℝ, borrowed_amount P ∧ P = 3000 :=
by
  use 3000
  unfold borrowed_amount
  simp
  sorry

end find_borrowed_amount_l185_185765


namespace find_number_l185_185524

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l185_185524


namespace tangency_theorem_l185_185572

variable {A B C D P Q : Point}

-- Definitions based on the conditions
def is_parallelogram (A B C D : Point) : Prop :=
  same_relation (<<AB>> <<CD>>) ∧ same_relation (<<AD>> <<BC>>)

def is_on_arc (P : Point) (A B C : Triangle) : Prop :=
  ∃ O : Point, is_circumcenter O A B C ∧ (¬ circle_contains O A P)

def on_segment (Q : Point) (A C : Point) := lies_on_line_segment Q A C

def has_equal_angles (P B C Q D : Point) : Prop :=
  ∠PBC = ∠CDQ

-- Statement of the theorem
theorem tangency_theorem (h_parallelogram : is_parallelogram A B C D)
    (h_P_arc : is_on_arc P A B C)
    (h_Q_segment: on_segment Q A C) 
    (h_angles : has_equal_angles P B C Q D) :
    tangent_to (circumcircle A P Q) A B :=
sorry

end tangency_theorem_l185_185572


namespace find_a2_b2_l185_185190

theorem find_a2_b2 (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : a^2 + b^2 = 8 :=
by
  sorry

end find_a2_b2_l185_185190


namespace equidistant_divisors_multiple_of_6_l185_185492

open Nat

theorem equidistant_divisors_multiple_of_6 (n : ℕ) :
  (∃ a b : ℕ, a ≠ b ∧ a ∣ n ∧ b ∣ n ∧ 
    (a + b = 2 * (n / 3))) → 
  (∃ k : ℕ, n = 6 * k) := 
by
  sorry

end equidistant_divisors_multiple_of_6_l185_185492


namespace percentage_increase_is_50_l185_185889

-- Defining the conditions
def new_wage : ℝ := 51
def original_wage : ℝ := 34
def increase : ℝ := new_wage - original_wage

-- Proving the required percentage increase is 50%
theorem percentage_increase_is_50 :
  (increase / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l185_185889


namespace max_possible_value_of_C_l185_185594

theorem max_possible_value_of_C (A B C D : ℕ) (h₁ : A + B + C + D = 200) (h₂ : A + B = 70) (h₃ : 0 < A) (h₄ : 0 < B) (h₅ : 0 < C) (h₆ : 0 < D) :
  C ≤ 129 :=
by
  sorry

end max_possible_value_of_C_l185_185594


namespace greatest_integer_inequality_l185_185743

theorem greatest_integer_inequality : 
  ⌊ (3 ^ 100 + 2 ^ 100 : ℝ) / (3 ^ 96 + 2 ^ 96) ⌋ = 80 :=
by
  sorry

end greatest_integer_inequality_l185_185743


namespace frogs_climbed_onto_logs_l185_185556

-- Definitions of the conditions
def f_lily : ℕ := 5
def f_rock : ℕ := 24
def f_total : ℕ := 32

-- The final statement we want to prove
theorem frogs_climbed_onto_logs : f_total - (f_lily + f_rock) = 3 :=
by
  sorry

end frogs_climbed_onto_logs_l185_185556


namespace find_number_l185_185699

theorem find_number (x : ℤ) (h : 3 * x + 4 = 19) : x = 5 :=
by {
  sorry
}

end find_number_l185_185699


namespace greatest_value_x_is_correct_l185_185180

noncomputable def greatest_value_x : ℝ :=
-8 + Real.sqrt 6

theorem greatest_value_x_is_correct :
  ∀ x : ℝ, (x ≠ 9) → ((x^2 - x - 90) / (x - 9) = 2 / (x + 6)) → x ≤ greatest_value_x :=
by
  sorry

end greatest_value_x_is_correct_l185_185180


namespace domain_of_f_l185_185921

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_f :
  { x : ℝ | 2 * x - 1 > 0 } = { x : ℝ | x > 1 / 2 } :=
by
  sorry

end domain_of_f_l185_185921


namespace video_games_expenditure_l185_185557

theorem video_games_expenditure (allowance : ℝ) (books_expense : ℝ) (snacks_expense : ℝ) (clothes_expense : ℝ) 
    (initial_allowance : allowance = 50)
    (books_fraction : books_expense = 1 / 7 * allowance)
    (snacks_fraction : snacks_expense = 1 / 2 * allowance)
    (clothes_fraction : clothes_expense = 3 / 14 * allowance) :
    50 - (books_expense + snacks_expense + clothes_expense) = 7.15 :=
by
  sorry

end video_games_expenditure_l185_185557


namespace remainder_is_correct_l185_185025

def P (x : ℝ) : ℝ := x^6 + 2 * x^5 - 3 * x^4 + x^2 - 8
def D (x : ℝ) : ℝ := x^2 - 1

theorem remainder_is_correct : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, P x = D x * q x + (2.5 * x - 9.5) :=
by
  sorry

end remainder_is_correct_l185_185025


namespace find_x_l185_185968

-- Definitions for the angles
def angle1 (x : ℝ) := 3 * x
def angle2 (x : ℝ) := 7 * x
def angle3 (x : ℝ) := 4 * x
def angle4 (x : ℝ) := 2 * x
def angle5 (x : ℝ) := x

-- The condition that the sum of the angles equals 360 degrees
def sum_of_angles (x : ℝ) := angle1 x + angle2 x + angle3 x + angle4 x + angle5 x = 360

-- The statement to prove
theorem find_x (x : ℝ) (hx : sum_of_angles x) : x = 360 / 17 := by
  -- Proof to be written here
  sorry

end find_x_l185_185968


namespace spherical_to_rectangular_correct_l185_185491

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_correct : spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 3) = (3, Real.sqrt 3, 2) :=
by
  sorry

end spherical_to_rectangular_correct_l185_185491


namespace quadratic_roots_square_l185_185028

theorem quadratic_roots_square (q : ℝ) :
  (∃ a : ℝ, a + a^2 = 12 ∧ q = a * a^2) → (q = 27 ∨ q = -64) :=
by
  sorry

end quadratic_roots_square_l185_185028


namespace unique_nonzero_b_l185_185812

variable (a b m n : ℝ)
variable (h_ne : m ≠ n)
variable (h_m_nonzero : m ≠ 0)
variable (h_n_nonzero : n ≠ 0)

theorem unique_nonzero_b (h : (a * m + b * n + m)^2 - (a * m + b * n + n)^2 = (m - n)^2) : 
  a = 0 ∧ b = -1 :=
sorry

end unique_nonzero_b_l185_185812


namespace seq_is_arithmetic_l185_185963

-- Define the sequence sum S_n and the sequence a_n
noncomputable def S (a : ℕ) (n : ℕ) : ℕ := a * n^2 + n
noncomputable def a_n (a : ℕ) (n : ℕ) : ℕ := S a n - S a (n - 1)

-- Define the property of being an arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, n ≥ 1 → (a_n (n + 1) : ℤ) - (a_n n : ℤ) = d

-- The theorem to be proven
theorem seq_is_arithmetic (a : ℕ) (h : 0 < a) : is_arithmetic_seq (a_n a) :=
by
  sorry

end seq_is_arithmetic_l185_185963


namespace initial_ratio_l185_185761

theorem initial_ratio (partners associates associates_after_hiring : ℕ)
  (h_partners : partners = 20)
  (h_associates_after_hiring : associates_after_hiring = 20 * 34)
  (h_assoc_equation : associates + 50 = associates_after_hiring) :
  (partners : ℚ) / associates = 2 / 63 :=
by
  sorry

end initial_ratio_l185_185761


namespace max_students_with_equal_distribution_l185_185766

theorem max_students_with_equal_distribution (pens pencils : ℕ) (h_pens : pens = 3540) (h_pencils : pencils = 2860) :
  gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  -- Proof steps will go here
  sorry

end max_students_with_equal_distribution_l185_185766


namespace triangle_height_l185_185309

def width := 10
def length := 2 * width
def area_rectangle := width * length
def base_triangle := width

theorem triangle_height (h : ℝ) : (1 / 2) * base_triangle * h = area_rectangle → h = 40 :=
by
  sorry

end triangle_height_l185_185309


namespace symmetric_circle_l185_185419

theorem symmetric_circle (x y : ℝ) :
  let C := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }
  let L := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  ∃ C' : ℝ × ℝ → Prop, (∀ p, C' p ↔ (p.1)^2 + (p.2)^2 = 1) :=
sorry

end symmetric_circle_l185_185419


namespace train_speed_l185_185634

theorem train_speed (v : ℕ) :
  (∀ (d : ℕ), d = 480 → ∀ (ship_speed : ℕ), ship_speed = 60 → 
  (∀ (ship_time : ℕ), ship_time = d / ship_speed →
  (∀ (train_time : ℕ), train_time = ship_time + 2 →
  v = d / train_time))) → v = 48 :=
by
  sorry

end train_speed_l185_185634


namespace sin_270_eq_neg_one_l185_185913

theorem sin_270_eq_neg_one : Real.sin (270 * Real.pi / 180) = -1 := 
by
  sorry

end sin_270_eq_neg_one_l185_185913


namespace expand_product_l185_185928

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end expand_product_l185_185928


namespace solve_diophantine_equation_l185_185493

def is_solution (m n : ℕ) : Prop := 2^m - 3^n = 1

theorem solve_diophantine_equation : 
  { (m, n) : ℕ × ℕ | is_solution m n } = { (1, 0), (2, 1) } :=
by
  sorry

end solve_diophantine_equation_l185_185493


namespace relationship_between_abc_l185_185943

noncomputable def a : Real := Real.sqrt 1.2
noncomputable def b : Real := Real.exp 0.1
noncomputable def c : Real := 1 + Real.log 1.1

theorem relationship_between_abc : b > a ∧ a > c :=
by {
  -- a = sqrt(1.2)
  -- b = exp(0.1)
  -- c = 1 + log(1.1)
  -- We need to prove: b > a > c
  sorry
}

end relationship_between_abc_l185_185943


namespace benny_days_worked_l185_185482

/-- Benny works 3 hours a day and in total he worked for 18 hours. 
We need to prove that he worked for 6 days. -/
theorem benny_days_worked (hours_per_day : ℕ) (total_hours : ℕ)
  (h1 : hours_per_day = 3)
  (h2 : total_hours = 18) :
  total_hours / hours_per_day = 6 := 
by sorry

end benny_days_worked_l185_185482


namespace ratio_of_areas_GHI_to_JKL_l185_185122

-- Define the side lengths of the triangles
def side_lengths_GHI := (7, 24, 25)
def side_lengths_JKL := (9, 40, 41)

-- Define the areas of the triangles
def area_triangle (a b : ℕ) : ℕ :=
  (a * b) / 2

def area_GHI := area_triangle 7 24
def area_JKL := area_triangle 9 40

-- Define the ratio of the areas
def ratio_areas (area1 area2 : ℕ) : ℚ :=
  area1 / area2

-- Prove the ratio of the areas
theorem ratio_of_areas_GHI_to_JKL :
  ratio_areas area_GHI area_JKL = (7 : ℚ) / 15 :=
by {
  sorry
}

end ratio_of_areas_GHI_to_JKL_l185_185122


namespace staircase_toothpicks_l185_185654

theorem staircase_toothpicks :
  ∀ (T : ℕ → ℕ), 
  (T 4 = 28) →
  (∀ n : ℕ, T (n + 1) = T n + (12 + 3 * (n - 3))) →
  T 6 - T 4 = 33 :=
by
  intros T T4_step H_increase
  -- proof goes here
  sorry

end staircase_toothpicks_l185_185654


namespace num_real_a_with_int_roots_l185_185185

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end num_real_a_with_int_roots_l185_185185


namespace common_ratio_geometric_series_l185_185929

theorem common_ratio_geometric_series
  (a₁ a₂ a₃ : ℚ)
  (h₁ : a₁ = 7 / 8)
  (h₂ : a₂ = -14 / 27)
  (h₃ : a₃ = 56 / 81) :
  (a₂ / a₁ = a₃ / a₂) ∧ (a₂ / a₁ = -2 / 3) :=
by
  -- The proof will follow here
  sorry

end common_ratio_geometric_series_l185_185929


namespace juice_oranges_l185_185554

theorem juice_oranges (oranges_per_glass : ℕ) (glasses : ℕ) (total_oranges : ℕ)
  (h1 : oranges_per_glass = 3)
  (h2 : glasses = 10)
  (h3 : total_oranges = oranges_per_glass * glasses) :
  total_oranges = 30 :=
by
  rw [h1, h2] at h3
  exact h3

end juice_oranges_l185_185554


namespace neg_P_l185_185552

/-
Proposition: There exists a natural number n such that 2^n > 1000.
-/
def P : Prop := ∃ n : ℕ, 2^n > 1000

/-
Theorem: The negation of the above proposition P is:
For all natural numbers n, 2^n ≤ 1000.
-/
theorem neg_P : ¬ P ↔ ∀ n : ℕ, 2^n ≤ 1000 :=
by
  sorry

end neg_P_l185_185552


namespace find_a_range_l185_185347

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 / 4 * x + 1 else Real.log x

theorem find_a_range : 
  {a : ℝ | ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a * x1 ∧ f x2 = a * x2} = [1 / 4, 1 / Real.e) := 
sorry

end find_a_range_l185_185347


namespace magic_square_l185_185702

-- Define a 3x3 grid with positions a, b, c and unknowns x, y, z, t, u, v
variables (a b c x y z t u v : ℝ)

-- State the theorem: there exists values for x, y, z, t, u, v
-- such that the sums in each row, column, and both diagonals are the same
theorem magic_square (h1: x = (b + 3*c - 2*a) / 2)
  (h2: y = a + b - c)
  (h3: z = (b + c) / 2)
  (h4: t = 2*c - a)
  (h5: u = b + c - a)
  (h6: v = (2*a + b - c) / 2) :
  x + a + b = y + z + t ∧
  y + z + t = u ∧
  z + t + u = b + z + c ∧
  t + u + v = a + u + c ∧
  x + t + v = u + y + c ∧
  by sorry :=
sorry

end magic_square_l185_185702


namespace proof_problem_l185_185858

def label_sum_of_domains_specified (labels: List Nat) (domains: List Nat) : Nat :=
  let relevant_labels := labels.filter (fun l => domains.contains l)
  relevant_labels.foldl (· + ·) 0

def label_product_of_continuous_and_invertible (labels: List Nat) (properties: List Bool) : Nat :=
  let relevant_labels := labels.zip properties |>.filter (fun (_, p) => p) |>.map (·.fst)
  relevant_labels.foldl (· * ·) 1

theorem proof_problem :
  label_sum_of_domains_specified [1, 2, 3, 4] [4] = 4 ∧ label_product_of_continuous_and_invertible [1, 2, 3, 4] [true, false, true, false] = 3 :=
by
  sorry

end proof_problem_l185_185858


namespace no_minimum_and_inf_S_eq_zero_l185_185487

open Function Real Interval

def A : Set (ℝ → ℝ) :=
  {f | ContDiffOn ℝ 1 f (Icc (-1 : ℝ) 1) ∧ f (-1) = -1 ∧ f 1 = 1}

def S (f : ℝ → ℝ) : ℝ :=
  ∫ x in -1..1, x^2 * (deriv f x) ^ 2

theorem no_minimum_and_inf_S_eq_zero :
  (¬ ∃ f ∈ A, (∀ g ∈ A, S f ≤ S g)) ∧ (inf {S f | f ∈ A} = 0) :=
by
  sorry

end no_minimum_and_inf_S_eq_zero_l185_185487


namespace even_function_f3_l185_185961

theorem even_function_f3 (a : ℝ) (h : ∀ x : ℝ, (x + 2) * (x - a) = (-x + 2) * (-x - a)) : (3 + 2) * (3 - a) = 5 := by
  sorry

end even_function_f3_l185_185961


namespace decomposition_of_x_l185_185149

-- Definitions derived from the conditions
def x : ℝ × ℝ × ℝ := (11, 5, -3)
def p : ℝ × ℝ × ℝ := (1, 0, 2)
def q : ℝ × ℝ × ℝ := (-1, 0, 1)
def r : ℝ × ℝ × ℝ := (2, 5, -3)

-- Theorem statement proving the decomposition
theorem decomposition_of_x : x = (3 : ℝ) • p + (-6 : ℝ) • q + (1 : ℝ) • r := by
  sorry

end decomposition_of_x_l185_185149


namespace sum_of_medians_bounds_l185_185402

theorem sum_of_medians_bounds (a b c m_a m_b m_c : ℝ) 
    (h1 : m_a < (b + c) / 2)
    (h2 : m_b < (a + c) / 2)
    (h3 : m_c < (a + b) / 2)
    (h4 : ∀a b c : ℝ, a + b > c) :
    (3 / 4) * (a + b + c) < m_a + m_b + m_c ∧ m_a + m_b + m_c < a + b + c := 
by
  sorry

end sum_of_medians_bounds_l185_185402


namespace abs_neg_three_halves_l185_185092

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l185_185092


namespace equilateral_triangle_BJ_l185_185825

-- Define points G, F, H, J and their respective lengths on sides AB and BC
def equilateral_triangle_AG_GF_HJ_FC (AG GF HJ FC BJ : ℕ) : Prop :=
  AG = 3 ∧ GF = 11 ∧ HJ = 5 ∧ FC = 4 ∧ 
    (∀ (side_length : ℕ), side_length = AG + GF + HJ + FC → 
    (∀ (length_J : ℕ), length_J = side_length - (AG + HJ) → BJ = length_J))

-- Example usage statement
theorem equilateral_triangle_BJ : 
  ∃ BJ, equilateral_triangle_AG_GF_HJ_FC 3 11 5 4 BJ ∧ BJ = 15 :=
by
  use 15
  sorry

end equilateral_triangle_BJ_l185_185825


namespace songs_today_is_14_l185_185573

-- Define the number of songs Jeremy listened to yesterday
def songs_yesterday (x : ℕ) : ℕ := x

-- Define the number of songs Jeremy listened to today
def songs_today (x : ℕ) : ℕ := x + 5

-- Given conditions
def total_songs (x : ℕ) : Prop := songs_yesterday x + songs_today x = 23

-- Prove the number of songs Jeremy listened to today
theorem songs_today_is_14 : ∃ x: ℕ, total_songs x ∧ songs_today x = 14 :=
by {
  sorry
}

end songs_today_is_14_l185_185573


namespace non_empty_set_A_l185_185946

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {x | x ^ 2 = a}

-- Theorem statement
theorem non_empty_set_A (a : ℝ) (h : (A a).Nonempty) : 0 ≤ a :=
by
  sorry

end non_empty_set_A_l185_185946


namespace frank_maze_time_l185_185186

theorem frank_maze_time 
    (n mazes : ℕ)
    (avg_time_per_maze completed_time total_allowable_time remaining_maze_time extra_time_inside current_time : ℕ) 
    (h1 : mazes = 5)
    (h2 : avg_time_per_maze = 60)
    (h3 : completed_time = 200)
    (h4 : total_allowable_time = mazes * avg_time_per_maze)
    (h5 : total_allowable_time = 300)
    (h6 : remaining_maze_time = total_allowable_time - completed_time) 
    (h7 : extra_time_inside = 55)
    (h8 : current_time + extra_time_inside ≤ remaining_maze_time) :
  current_time = 45 :=
by
  sorry

end frank_maze_time_l185_185186


namespace inequality_a_b_c_l185_185942

noncomputable def a := Real.log (Real.pi / 3)
noncomputable def b := Real.log (Real.exp 1 / 3)
noncomputable def c := Real.exp (0.5)

theorem inequality_a_b_c : c > a ∧ a > b := by
  sorry

end inequality_a_b_c_l185_185942


namespace cars_transfer_equation_l185_185086

theorem cars_transfer_equation (x : ℕ) : 100 - x = 68 + x :=
sorry

end cars_transfer_equation_l185_185086


namespace good_numbers_100_2010_ex_good_and_not_good_x_y_l185_185160

-- Definition of a good number
def is_good_number (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + 161 * b^2

-- (1) Prove 100 and 2010 are good numbers
theorem good_numbers_100_2010 : is_good_number 100 ∧ is_good_number 2010 :=
by sorry

-- (2) Prove there exist positive integers x and y such that x^161 + y^161 is a good number, 
-- but x + y is not a good number
theorem ex_good_and_not_good_x_y : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ is_good_number (x^161 + y^161) ∧ ¬ is_good_number (x + y) :=
by sorry

end good_numbers_100_2010_ex_good_and_not_good_x_y_l185_185160


namespace tickets_sold_total_l185_185605

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end tickets_sold_total_l185_185605


namespace best_years_to_scrap_l185_185080

-- Define the conditions from the problem
def purchase_cost : ℕ := 150000
def annual_cost : ℕ := 15000
def maintenance_initial : ℕ := 3000
def maintenance_difference : ℕ := 3000

-- Define the total_cost function
def total_cost (n : ℕ) : ℕ :=
  purchase_cost + annual_cost * n + (n * (2 * maintenance_initial + (n - 1) * maintenance_difference)) / 2

-- Define the average annual cost function
def average_annual_cost (n : ℕ) : ℕ :=
  total_cost n / n

-- Statement to be proven: the best number of years to minimize average annual cost is 10
theorem best_years_to_scrap : 
  (∀ n : ℕ, average_annual_cost 10 ≤ average_annual_cost n) :=
by
  sorry
  
end best_years_to_scrap_l185_185080


namespace budget_equality_year_l185_185619

theorem budget_equality_year :
  ∃ n : ℕ, 540000 + 30000 * n = 780000 - 10000 * n ∧ 1990 + n = 1996 :=
by
  sorry

end budget_equality_year_l185_185619


namespace f_of_f_inv_e_eq_inv_e_l185_185191

noncomputable def f : ℝ → ℝ := λ x =>
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_of_f_inv_e_eq_inv_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_of_f_inv_e_eq_inv_e_l185_185191


namespace era_slices_burger_l185_185776

theorem era_slices_burger (slices_per_burger : ℕ) (h : 5 * slices_per_burger = 10) : slices_per_burger = 2 :=
by 
  sorry

end era_slices_burger_l185_185776


namespace side_length_equilateral_l185_185997

-- Define our parameters
variables {A B C Q : Point} (t : ℝ)

-- Define the conditions
def is_equilateral (A B C : Point) (t : ℝ) := dist A B = t ∧ dist B C = t ∧ dist C A = t
def distances_to_point (A B C Q : Point) := dist A Q = 2 ∧ dist B Q = 2 * Real.sqrt 2 ∧ dist C Q = 3

-- The theorem to prove
theorem side_length_equilateral (A B C Q : Point) (t : ℝ) 
  (h_eq : is_equilateral A B C t)
  (h_dist : distances_to_point A B C Q) :
  t = Real.sqrt 15 :=
sorry

end side_length_equilateral_l185_185997


namespace even_function_derivative_at_zero_l185_185219

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_diff : Differentiable ℝ f)

theorem even_function_derivative_at_zero : deriv f 0 = 0 :=
by 
  -- proof omitted
  sorry

end even_function_derivative_at_zero_l185_185219


namespace trajectory_of_center_l185_185966

theorem trajectory_of_center :
  ∃ (x y : ℝ), (x + 1) ^ 2 + y ^ 2 = 49 / 4 ∧ (x - 1) ^ 2 + y ^ 2 = 1 / 4 ∧ ( ∀ P, (P = (x, y) → (P.1^2) / 4 + (P.2^2) / 3 = 1) ) := sorry

end trajectory_of_center_l185_185966


namespace tomatoes_first_shipment_l185_185391

theorem tomatoes_first_shipment :
  ∃ X : ℕ, 
    (∀Y : ℕ, 
      (Y = 300) → -- Saturday sale
      (X - Y = X - 300) ∧
      (∀Z : ℕ, 
        (Z = 200) → -- Sunday rotting
        (X - 300 - Z = X - 500) ∧
        (∀W : ℕ, 
          (W = 2 * X) → -- Monday new shipment
          (X - 500 + W = 2500) →
          (X = 1000)
        )
      )
    ) :=
by
  sorry

end tomatoes_first_shipment_l185_185391


namespace prob_all_digits_different_l185_185645

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l185_185645


namespace find_y_coordinate_l185_185446

theorem find_y_coordinate (y : ℝ) (h : y > 0) (dist_eq : (10 - 2)^2 + (y - 5)^2 = 13^2) : y = 16 :=
by
  sorry

end find_y_coordinate_l185_185446


namespace bead_bracelet_problem_l185_185059

-- Define the condition Bead A and Bead B are always next to each other
def adjacent (A B : ℕ) (l : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), l = l1 ++ A :: B :: l2 ∨ l = l1 ++ B :: A :: l2

-- Define the context and translate the problem
def bracelet_arrangements (n : ℕ) : ℕ :=
  if n = 8 then 720 else 0

theorem bead_bracelet_problem : bracelet_arrangements 8 = 720 :=
by {
  -- Place proof here
  sorry 
}

end bead_bracelet_problem_l185_185059


namespace sum_first_7_terms_eq_105_l185_185797

variable {a : ℕ → ℤ}

-- Definitions from conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a)

def a_4_eq_15 : a 4 = 15 := sorry

-- Sum definition specific for 7 terms of an arithmetic sequence.
def sum_first_7_terms (a : ℕ → ℤ) : ℤ := (7 / 2 : ℤ) * (a 1 + a 7)

-- The theorem to prove.
theorem sum_first_7_terms_eq_105 
    (arith_seq : is_arithmetic_sequence a) 
    (a4 : a 4 = 15) : 
  sum_first_7_terms a = 105 := 
sorry

end sum_first_7_terms_eq_105_l185_185797


namespace exponent_property_l185_185292

theorem exponent_property :
  4^4 * 9^4 * 4^9 * 9^9 = 36^13 :=
by
  -- Add the proof here
  sorry

end exponent_property_l185_185292


namespace find_a8_l185_185341

variable (a : ℕ → ℤ)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)

theorem find_a8 (h1 : a 7 + a 9 = 16) (h2 : arithmetic_sequence a) : a 8 = 8 := by
  -- proof would go here
  sorry

end find_a8_l185_185341


namespace distance_from_apex_l185_185281

theorem distance_from_apex (a₁ a₂ : ℝ) (d : ℝ)
  (ha₁ : a₁ = 150 * Real.sqrt 3)
  (ha₂ : a₂ = 300 * Real.sqrt 3)
  (hd : d = 10) :
  ∃ h : ℝ, h = 10 * Real.sqrt 2 :=
by
  sorry

end distance_from_apex_l185_185281


namespace find_b_l185_185355

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end find_b_l185_185355


namespace sqrt_eq_cubrt_l185_185962

theorem sqrt_eq_cubrt (x : ℝ) (h : Real.sqrt x = x^(1/3)) : x = 0 ∨ x = 1 :=
by
  sorry

end sqrt_eq_cubrt_l185_185962


namespace M_subset_N_l185_185205

def M : Set ℚ := { x | ∃ k : ℤ, x = k / 2 + 1 / 4 }
def N : Set ℚ := { x | ∃ k : ℤ, x = k / 4 + 1 / 2 }

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l185_185205


namespace necessary_but_not_sufficient_condition_l185_185714

open Classical

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 3) → (a + b > 4) ∧ ¬((a + b > 4) → (a > 1 ∧ b > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l185_185714


namespace find_g_inv_84_l185_185212

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end find_g_inv_84_l185_185212


namespace Steven_more_than_Jill_l185_185233

variable (Jill Jake Steven : ℕ)

def Jill_peaches : Jill = 87 := by sorry
def Jake_peaches_more : Jake = Jill + 13 := by sorry
def Steven_peaches_more : Steven = Jake + 5 := by sorry

theorem Steven_more_than_Jill : Steven - Jill = 18 := by
  -- Proof steps to be filled
  sorry

end Steven_more_than_Jill_l185_185233


namespace apple_tree_total_production_l185_185896

-- Definitions for conditions
def first_year_production : ℕ := 40
def second_year_production : ℕ := 2 * first_year_production + 8
def third_year_production : ℕ := second_year_production - second_year_production / 4

-- Theorem statement
theorem apple_tree_total_production :
  first_year_production + second_year_production + third_year_production = 194 :=
by
  sorry

end apple_tree_total_production_l185_185896


namespace tan_ratio_l185_185386

open Real

variable (x y : ℝ)

-- Conditions
def sin_add_eq : sin (x + y) = 5 / 8 := sorry
def sin_sub_eq : sin (x - y) = 1 / 4 := sorry

-- Proof problem statement
theorem tan_ratio : sin (x + y) = 5 / 8 → sin (x - y) = 1 / 4 → (tan x) / (tan y) = 2 := sorry

end tan_ratio_l185_185386


namespace intersection_of_A_B_find_a_b_l185_185390

-- Lean 4 definitions based on the given conditions
def setA (x : ℝ) : Prop := 4 - x^2 > 0
def setB (x : ℝ) (y : ℝ) : Prop := y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0

-- Prove the intersection of sets A and B
theorem intersection_of_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | ∃ y : ℝ, setB x y} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

-- On the roots of the quadratic equation and solution interval of inequality
theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, 2 * x^2 + a * x + b < 0 ↔ -3 < x ∧ x < 1) →
  a = 4 ∧ b = -6 :=
by
  sorry

end intersection_of_A_B_find_a_b_l185_185390


namespace necessary_but_not_sufficient_condition_l185_185320

-- Definitions of conditions
def condition_p (x : ℝ) := (x - 1) * (x + 2) ≤ 0
def condition_q (x : ℝ) := abs (x + 1) ≤ 1

-- The theorem statement
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (∀ x, condition_q x → condition_p x) ∧ ¬(∀ x, condition_p x → condition_q x) := 
by
  sorry

end necessary_but_not_sufficient_condition_l185_185320


namespace three_b_minus_a_eq_neg_five_l185_185960

theorem three_b_minus_a_eq_neg_five (a b : ℤ) (h : |a - 2| + (b + 1)^2 = 0) : 3 * b - a = -5 :=
sorry

end three_b_minus_a_eq_neg_five_l185_185960


namespace soccer_points_l185_185271

def total_points (wins draws losses : ℕ) (points_per_win points_per_draw points_per_loss : ℕ) : ℕ :=
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

theorem soccer_points : total_points 14 4 2 3 1 0 = 46 :=
by
  sorry

end soccer_points_l185_185271


namespace total_houses_is_160_l185_185055

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end total_houses_is_160_l185_185055


namespace polygon_with_given_angle_sums_is_hexagon_l185_185370

theorem polygon_with_given_angle_sums_is_hexagon
  (n : ℕ)
  (h_interior : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end polygon_with_given_angle_sums_is_hexagon_l185_185370


namespace regular_hexagon_interior_angle_l185_185689

theorem regular_hexagon_interior_angle : ∀ (n : ℕ), n = 6 → ∀ (angle_sum : ℕ), angle_sum = (n - 2) * 180 → (∀ (angle : ℕ), angle = angle_sum / n → angle = 120) :=
by sorry

end regular_hexagon_interior_angle_l185_185689


namespace index_card_area_l185_185687

theorem index_card_area :
  ∀ (length width : ℕ), length = 5 → width = 7 →
  (length - 2) * width = 21 →
  length * (width - 1) = 30 :=
by
  intros length width h_length h_width h_condition
  sorry

end index_card_area_l185_185687


namespace total_machine_operation_time_l185_185979

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end total_machine_operation_time_l185_185979


namespace probability_of_selecting_two_girls_l185_185883

def total_students : ℕ := 5
def boys : ℕ := 2
def girls : ℕ := 3
def selected_students : ℕ := 2

theorem probability_of_selecting_two_girls :
  (Nat.choose girls selected_students : ℝ) / (Nat.choose total_students selected_students : ℝ) = 0.3 := by
  sorry

end probability_of_selecting_two_girls_l185_185883


namespace solution_set_of_inequality_l185_185418

variable (m x : ℝ)

-- Defining the condition
def inequality (m x : ℝ) := x^2 - (2 * m - 1) * x + m^2 - m > 0

-- Problem statement
theorem solution_set_of_inequality (h : inequality m x) : x < m-1 ∨ x > m :=
  sorry

end solution_set_of_inequality_l185_185418


namespace enclosedArea_l185_185139

theorem enclosedArea (x y : ℝ) :
  (x^2 + y^2 = 2 * (|x| + |y|)) → (area {p : ℝ × ℝ | p.1^2 + p.2^2 = 2 * (|p.1| + |p.2|)}) = 2 * real.pi :=
by
  sorry

end enclosedArea_l185_185139


namespace determine_ABC_l185_185251

theorem determine_ABC : 
  ∀ (A B C : ℝ), 
    A = 2 * B - 3 * C ∧ 
    B = 2 * C - 5 ∧ 
    A + B + C = 100 → 
    A = 18.75 ∧ B = 52.5 ∧ C = 28.75 :=
by
  intro A B C h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end determine_ABC_l185_185251


namespace adults_not_wearing_blue_is_10_l185_185902

section JohnsonFamilyReunion

-- Define the number of children
def children : ℕ := 45

-- Define the ratio between adults and children
def adults : ℕ := children / 3

-- Define the ratio of adults who wore blue
def adults_wearing_blue : ℕ := adults / 3

-- Define the number of adults who did not wear blue
def adults_not_wearing_blue : ℕ := adults - adults_wearing_blue

-- Theorem stating the number of adults who did not wear blue
theorem adults_not_wearing_blue_is_10 : adults_not_wearing_blue = 10 :=
by
  -- This is a placeholder for the actual proof
  sorry

end JohnsonFamilyReunion

end adults_not_wearing_blue_is_10_l185_185902


namespace ratio_of_areas_of_triangles_l185_185126

noncomputable def area_of_triangle (a b c : ℕ) : ℕ :=
  if a * a + b * b = c * c then (a * b) / 2 else 0

theorem ratio_of_areas_of_triangles :
  let area_GHI := area_of_triangle 7 24 25
  let area_JKL := area_of_triangle 9 40 41
  (area_GHI : ℚ) / area_JKL = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l185_185126


namespace linear_equation_value_l185_185365

-- Define the conditions of the equation
def equation_is_linear (m : ℝ) : Prop :=
  |m| = 1 ∧ m - 1 ≠ 0

-- Prove the equivalence statement
theorem linear_equation_value (m : ℝ) (h : equation_is_linear m) : m = -1 := 
sorry

end linear_equation_value_l185_185365


namespace find_number_l185_185520

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l185_185520


namespace area_enclosed_by_equation_l185_185871

theorem area_enclosed_by_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 10 * y = -20) → (∃ r : ℝ, r^2 = 9 ∧ ∃ c : ℝ × ℝ, (∃ a b, (x - a)^2 + (y - b)^2 = r^2)) :=
by
  sorry

end area_enclosed_by_equation_l185_185871


namespace geom_progression_lines_common_point_l185_185164

theorem geom_progression_lines_common_point
  (a c b : ℝ) (r : ℝ)
  (h_geom_prog : c = a * r ∧ b = a * r^2) :
  ∃ (P : ℝ × ℝ), ∀ (a c b : ℝ), c = a * r ∧ b = a * r^2 → (P = (0, 0) ∧ a ≠ 0) :=
by
  sorry

end geom_progression_lines_common_point_l185_185164


namespace max_volume_at_6_l185_185745

noncomputable def volume (x : ℝ) : ℝ :=
  x * (36 - 2 * x)^2

theorem max_volume_at_6 :
  ∃ x : ℝ, (0 < x) ∧ (x < 18) ∧ 
  (∀ y : ℝ, (0 < y) ∧ (y < 18) → volume y ≤ volume 6) :=
by
  sorry

end max_volume_at_6_l185_185745


namespace orangeade_price_second_day_l185_185878

theorem orangeade_price_second_day :
  ∀ (X O : ℝ), (2 * X * 0.60 = 3 * X * E) → (E = 2 * 0.60 / 3) →
  E = 0.40 := by
  intros X O h₁ h₂
  sorry

end orangeade_price_second_day_l185_185878


namespace solve_problem_l185_185525

def question : ℝ := -7.8
def answer : ℕ := 22

theorem solve_problem : 2 * (⌊|question|⌋) + (|⌊question⌋|) = answer := by
  sorry

end solve_problem_l185_185525


namespace possible_triplets_l185_185659

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem possible_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (is_power_of_two (a * b - c) ∧ is_power_of_two (b * c - a) ∧ is_power_of_two (c * a - b)) ↔ 
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) :=
by
  sorry

end possible_triplets_l185_185659


namespace find_n_eq_6_l185_185021

theorem find_n_eq_6 (n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) : 2^n + n^2 + 25 = p^3 → n = 6 := by
  sorry

end find_n_eq_6_l185_185021


namespace triangle_circle_square_value_l185_185000

theorem triangle_circle_square_value (Δ : ℝ) (bigcirc : ℝ) (square : ℝ) 
  (h1 : 2 * Δ + 3 * bigcirc + square = 45)
  (h2 : Δ + 5 * bigcirc + 2 * square = 58)
  (h3 : 3 * Δ + bigcirc + 3 * square = 62) :
  Δ + 2 * bigcirc + square = 35 :=
sorry

end triangle_circle_square_value_l185_185000


namespace cos_seven_pi_six_l185_185783

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end cos_seven_pi_six_l185_185783


namespace probability_all_balls_same_color_probability_4_white_balls_l185_185040

-- Define initial conditions
def initial_white_balls : ℕ := 6
def initial_yellow_balls : ℕ := 4
def total_initial_balls : ℕ := initial_white_balls + initial_yellow_balls

-- Define the probability calculation for drawing balls as described
noncomputable def draw_probability_same_color_after_4_draws : ℚ :=
  (6 / 10) * (7 / 10) * (8 / 10) * (9 / 10)

noncomputable def draw_probability_4_white_balls_after_4_draws : ℚ :=
  (6 / 10) * (3 / 10) * (4 / 10) * (5 / 10) + 
  3 * ((4 / 10) * (5 / 10) * (4 / 10) * (5 / 10))

-- The theorem we want to prove about the probabilities
theorem probability_all_balls_same_color :
  draw_probability_same_color_after_4_draws = 189 / 625 := by
  sorry

theorem probability_4_white_balls :
  draw_probability_4_white_balls_after_4_draws = 19 / 125 := by
  sorry

end probability_all_balls_same_color_probability_4_white_balls_l185_185040


namespace sum_of_positive_integers_eq_32_l185_185106

noncomputable def sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : ℕ :=
  x + y

theorem sum_of_positive_integers_eq_32 (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : sum_of_integers x y h1 h2 = 32 :=
  sorry

end sum_of_positive_integers_eq_32_l185_185106


namespace calculate_area_l185_185969

def leftmost_rectangle_area (height width : ℕ) : ℕ := height * width
def middle_rectangle_area (height width : ℕ) : ℕ := height * width
def rightmost_rectangle_area (height width : ℕ) : ℕ := height * width

theorem calculate_area : 
  let leftmost_segment_height := 7
  let bottom_width := 6
  let segment_above_3 := 3
  let segment_above_2 := 2
  let rightmost_width := 5
  leftmost_rectangle_area leftmost_segment_height bottom_width + 
  middle_rectangle_area segment_above_3 segment_above_3 + 
  rightmost_rectangle_area segment_above_2 rightmost_width = 
  61 := by
    sorry

end calculate_area_l185_185969


namespace gcd_multiples_l185_185561

theorem gcd_multiples (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : Nat.gcd p q = 15) : Nat.gcd (8 * p) (18 * q) = 30 :=
by sorry

end gcd_multiples_l185_185561


namespace simplify_expression_l185_185777

theorem simplify_expression (x : ℝ) :
  ( ( ((x + 1) ^ 3 * (x ^ 2 - x + 1) ^ 3) / (x ^ 3 + 1) ^ 3 ) ^ 2 *
    ( ((x - 1) ^ 3 * (x ^ 2 + x + 1) ^ 3) / (x ^ 3 - 1) ^ 3 ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l185_185777


namespace peanuts_remaining_l185_185422

theorem peanuts_remaining (initial_peanuts brock_ate bonita_ate brock_fraction : ℕ) (h_initial : initial_peanuts = 148) (h_brock_fraction : brock_fraction = 4) (h_brock_ate : brock_ate = initial_peanuts / brock_fraction) (h_bonita_ate : bonita_ate = 29) :
  (initial_peanuts - brock_ate - bonita_ate) = 82 :=
by
  sorry

end peanuts_remaining_l185_185422


namespace doughnut_machine_completion_time_l185_185296

-- Define the start time and the time when half the job is completed
def start_time := 8 * 60 -- 8:00 AM in minutes
def half_job_time := 10 * 60 + 30 -- 10:30 AM in minutes

-- Given the machine completes half of the day's job by 10:30 AM
-- Prove that the doughnut machine will complete the entire job by 1:00 PM
theorem doughnut_machine_completion_time :
  half_job_time - start_time = 150 → 
  (start_time + 2 * 150) % (24 * 60) = 13 * 60 :=
by
  sorry

end doughnut_machine_completion_time_l185_185296


namespace wickets_before_last_match_l185_185764

theorem wickets_before_last_match
  (W : ℝ)  -- Number of wickets before last match
  (R : ℝ)  -- Total runs before last match
  (h1 : R = 12.4 * W)
  (h2 : (R + 26) / (W + 8) = 12.0)
  : W = 175 :=
sorry

end wickets_before_last_match_l185_185764


namespace range_of_a_l185_185342

variable {x a : ℝ}

def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

theorem range_of_a (A_union_B_R : A ∪ B a = Set.univ) : a ∈ Set.Ici 3 :=
  sorry

end range_of_a_l185_185342


namespace ratio_of_adult_to_kid_charge_l185_185236

variable (A : ℝ)  -- Charge for adults

-- Conditions
def kids_charge : ℝ := 3
def num_kids_per_day : ℝ := 8
def num_adults_per_day : ℝ := 10
def weekly_earnings : ℝ := 588
def days_per_week : ℝ := 7

-- Hypothesis for the relationship between charges and total weekly earnings
def total_weekly_earnings_eq : Prop :=
  days_per_week * (num_kids_per_day * kids_charge + num_adults_per_day * A) = weekly_earnings

-- Statement to be proved
theorem ratio_of_adult_to_kid_charge (h : total_weekly_earnings_eq A) : (A / kids_charge) = 2 := 
by 
  sorry

end ratio_of_adult_to_kid_charge_l185_185236


namespace messages_per_member_per_day_l185_185638

theorem messages_per_member_per_day (initial_members removed_members remaining_members total_weekly_messages total_daily_messages : ℕ)
  (h1 : initial_members = 150)
  (h2 : removed_members = 20)
  (h3 : remaining_members = initial_members - removed_members)
  (h4 : total_weekly_messages = 45500)
  (h5 : total_daily_messages = total_weekly_messages / 7)
  (h6 : 7 * total_daily_messages = total_weekly_messages) -- ensures that total_daily_messages calculated is correct
  : total_daily_messages / remaining_members = 50 := 
by
  sorry

end messages_per_member_per_day_l185_185638


namespace coffee_blend_l185_185002

variable (pA pB : ℝ) (cA cB : ℝ) (total_cost : ℝ) 

theorem coffee_blend (hA : pA = 4.60) 
                     (hB : pB = 5.95) 
                     (h_ratio : cB = 2 * cA) 
                     (h_total : 4.60 * cA + 5.95 * cB = 511.50) : 
                     cA = 31 := 
by
  sorry

end coffee_blend_l185_185002


namespace probability_same_color_is_19_over_39_l185_185748
-- Step d): Rewrite in Lean 4 statement

def probability_same_color : ℚ :=
  let total_balls := 13
  let green_balls := 5
  let white_balls := 8
  let total_ways := Nat.choose total_balls 2
  let green_ways := Nat.choose green_balls 2
  let white_ways := Nat.choose white_balls 2
  (green_ways + white_ways) / total_ways

theorem probability_same_color_is_19_over_39 :
  probability_same_color = 19 / 39 :=
by
  sorry

end probability_same_color_is_19_over_39_l185_185748


namespace olivia_race_time_l185_185016

variable (O E : ℕ)

theorem olivia_race_time (h1 : O + E = 112) (h2 : E = O - 4) : O = 58 :=
sorry

end olivia_race_time_l185_185016


namespace ratio_of_areas_l185_185132

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l185_185132


namespace simplify_complex_div_l185_185729

theorem simplify_complex_div (a b c d : ℝ) (i : ℂ)
  (h1 : (a = 3) ∧ (b = 5) ∧ (c = -2) ∧ (d = 7) ∧ (i = Complex.I)) :
  ((Complex.mk a b) / (Complex.mk c d) = (Complex.mk (29/53) (-31/53))) :=
by
  sorry

end simplify_complex_div_l185_185729


namespace total_machine_operation_time_l185_185980

theorem total_machine_operation_time 
  (num_dolls : ℕ) 
  (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll time_per_accessory : ℕ)
  (num_shoes num_bags num_cosmetics num_hats num_accessories : ℕ) 
  (total_doll_time total_accessory_time total_time : ℕ) :
  num_dolls = 12000 →
  shoes_per_doll = 2 →
  bags_per_doll = 3 →
  cosmetics_per_doll = 1 →
  hats_per_doll = 5 →
  time_per_doll = 45 →
  time_per_accessory = 10 →
  num_shoes = num_dolls * shoes_per_doll →
  num_bags = num_dolls * bags_per_doll →
  num_cosmetics = num_dolls * cosmetics_per_doll →
  num_hats = num_dolls * hats_per_doll →
  num_accessories = num_shoes + num_bags + num_cosmetics + num_hats →
  total_doll_time = num_dolls * time_per_doll →
  total_accessory_time = num_accessories * time_per_accessory →
  total_time = total_doll_time + total_accessory_time →
  total_time = 1860000 := 
sorry

end total_machine_operation_time_l185_185980


namespace simplify_expression_l185_185617

variable (d : ℤ)

theorem simplify_expression :
  (5 + 4 * d) / 9 - 3 + 1 / 3 = (4 * d - 19) / 9 := by
  sorry

end simplify_expression_l185_185617


namespace table_area_l185_185118

theorem table_area (A : ℝ) 
  (combined_area : ℝ)
  (coverage_percentage : ℝ)
  (area_two_layers : ℝ)
  (area_three_layers : ℝ)
  (combined_area_eq : combined_area = 220)
  (coverage_percentage_eq : coverage_percentage = 0.80 * A)
  (area_two_layers_eq : area_two_layers = 24)
  (area_three_layers_eq : area_three_layers = 28) :
  A = 275 :=
by
  -- Assumptions and derivations can be filled in.
  sorry

end table_area_l185_185118


namespace line_tangent_constant_sum_l185_185550

noncomputable def parabolaEquation (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

noncomputable def isTangent (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  l x = y ∧ ((x - 2) ^ 2 + y ^ 2 = 4)

theorem line_tangent_constant_sum (l : ℝ → ℝ) (A B P : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabolaEquation x₁ y₁ →
  parabolaEquation x₂ y₂ →
  isTangent l (4 / 5) (8 / 5) →
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  let F := (1, 0)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := (Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))
  (distance F A) + (distance F B) - (distance A B) = 2 :=
sorry

end line_tangent_constant_sum_l185_185550


namespace average_sales_l185_185007

theorem average_sales (jan feb mar apr : ℝ) (h_jan : jan = 100) (h_feb : feb = 60) (h_mar : mar = 40) (h_apr : apr = 120) : 
  (jan + feb + mar + apr) / 4 = 80 :=
by {
  sorry
}

end average_sales_l185_185007


namespace find_unknown_number_l185_185112

def unknown_number (x : ℝ) : Prop :=
  (0.5^3) - (0.1^3 / 0.5^2) + x + (0.1^2) = 0.4

theorem find_unknown_number : ∃ (x : ℝ), unknown_number x ∧ x = 0.269 :=
by
  sorry

end find_unknown_number_l185_185112


namespace expression_evaluation_l185_185730

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = -1 ^ 2023) :
  (2 * m + n) * (2 * m - n) - (2 * m - n) ^ 2 + 2 * n * (m + n) = -12 := by
  sorry

end expression_evaluation_l185_185730


namespace round_trip_by_car_time_l185_185457

variable (time_walk time_car : ℕ)
variable (h1 : time_walk + time_car = 20)
variable (h2 : 2 * time_walk = 32)

theorem round_trip_by_car_time : 2 * time_car = 8 :=
by
  sorry

end round_trip_by_car_time_l185_185457


namespace number_of_speaking_orders_l185_185631

-- Define the original set of people
def group := fin 7

-- Define the subset of people including A and B
def A_and_B := {0, 1} -- Let's assume person 0 is A and person 1 is B

-- Define the condition that at least one of A and B must participate
def at_least_one_AB (S : finset group) : Prop :=
  0 ∈ S ∨ 1 ∈ S

-- Define the main theorem
theorem number_of_speaking_orders :
  ∃ (S : finset group), S.card = 4 ∧ at_least_one_AB S ∧ S.order_count = 720 :=
sorry

end number_of_speaking_orders_l185_185631


namespace BRAIN_7225_cycle_line_number_l185_185415

def BRAIN_cycle : Nat := 5
def _7225_cycle : Nat := 4

theorem BRAIN_7225_cycle_line_number : Nat.lcm BRAIN_cycle _7225_cycle = 20 :=
by
  sorry

end BRAIN_7225_cycle_line_number_l185_185415


namespace numberOfWaysToDistributeMedals_correct_l185_185298

-- Define the medals and their constraints
noncomputable def numberOfWaysToDistributeMedals : ℕ :=
  (Nat.choose (12 - 1) (3 - 1))

theorem numberOfWaysToDistributeMedals_correct :
  numberOfWaysToDistributeMedals = 55 := by
  sorry

end numberOfWaysToDistributeMedals_correct_l185_185298


namespace domain_of_function_l185_185174

theorem domain_of_function :
  ∀ x : ℝ, (x > 0) ∧ (x ≤ 2) ∧ (x ≠ 1) ↔ ∀ x, (∃ y : ℝ, y = (1 / (Real.log x / Real.log 10) + Real.sqrt (2 - x))) :=
by
  sorry

end domain_of_function_l185_185174


namespace seating_arrangements_count_l185_185593

-- Define the main entities: the three teams and the conditions
inductive Person
| Jupitarian
| Saturnian
| Neptunian

open Person

-- Define the seating problem constraints
def valid_arrangement (seating : Fin 12 → Person) : Prop :=
  seating 0 = Jupitarian ∧ seating 11 = Neptunian ∧
  (∀ i, seating (i % 12) = Jupitarian → seating ((i + 11) % 12) ≠ Neptunian) ∧
  (∀ i, seating (i % 12) = Neptunian → seating ((i + 11) % 12) ≠ Saturnian) ∧
  (∀ i, seating (i % 12) = Saturnian → seating ((i + 11) % 12) ≠ Jupitarian)

-- Main theorem: The number of valid arrangements is 225 * (4!)^3
theorem seating_arrangements_count :
  ∃ M : ℕ, (M = 225) ∧ ∃ arrangements : Fin 12 → Person, valid_arrangement arrangements :=
sorry

end seating_arrangements_count_l185_185593


namespace orchestra_ticket_cost_l185_185456

noncomputable def cost_balcony : ℝ := 8  -- cost of balcony tickets
noncomputable def total_sold : ℝ := 340  -- total tickets sold
noncomputable def total_revenue : ℝ := 3320  -- total revenue
noncomputable def extra_balcony : ℝ := 40  -- extra tickets sold for balcony than orchestra

theorem orchestra_ticket_cost (x y : ℝ) (h1 : x + extra_balcony = total_sold)
    (h2 : y = x + extra_balcony) (h3 : x + y = total_sold)
    (h4 : x + cost_balcony * y = total_revenue) : 
    cost_balcony = 8 → x = 12 :=
by
  sorry

end orchestra_ticket_cost_l185_185456


namespace layla_goldfish_count_l185_185981

def goldfish_count (total_food : ℕ) (swordtails_count : ℕ) (swordtails_food : ℕ) (guppies_count : ℕ) (guppies_food : ℕ) (goldfish_food : ℕ) : ℕ :=
  total_food - (swordtails_count * swordtails_food + guppies_count * guppies_food) / goldfish_food

theorem layla_goldfish_count : goldfish_count 12 3 2 8 1 1 = 2 := by
  sorry

end layla_goldfish_count_l185_185981


namespace six_digit_increasing_check_six_digit_increasing_l185_185712

theorem six_digit_increasing (N : ℕ) : 
  N = number_of_six_digit_increasing_nonstarting_with_six 1 2 3 4 5 6 := by 
  sorry

/--
Define the number of six-digit integers where digits are in increasing order 
and do not start with the digit 6.
-/
def number_of_six_digit_increasing_nonstarting_with_six
  (d1 d2 d3 d4 d5 d6 : ℕ) : ℕ :=
  let total := Nat.choose (6 + 5) 5
  let exclude6 := Nat.choose (5 + 4) 4
  total - exclude6

noncomputable def number_of_six_digit_increasing_nonstarting_with_six_value : ℕ :=
  number_of_six_digit_increasing_nonstarting_with_six 1 2 3 4 5 6

theorem check_six_digit_increasing :
    number_of_six_digit_increasing_nonstarting_with_six_value = 336 := by
  sorry

end six_digit_increasing_check_six_digit_increasing_l185_185712


namespace min_value_a_3b_9c_l185_185986

theorem min_value_a_3b_9c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 
  a + 3 * b + 9 * c ≥ 27 := 
sorry

end min_value_a_3b_9c_l185_185986


namespace probability_all_digits_different_l185_185647

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l185_185647


namespace no_combination_of_three_coins_sums_to_52_cents_l185_185117

def is_valid_coin (c : ℕ) : Prop :=
  c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50 ∨ c = 100

theorem no_combination_of_three_coins_sums_to_52_cents :
  ¬ ∃ a b c : ℕ, is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ a + b + c = 52 :=
by 
  sorry

end no_combination_of_three_coins_sums_to_52_cents_l185_185117


namespace problem_statement_l185_185948

open Real

-- Define a synchronous property
def synchronous (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  f m = g m ∧ f n = g n

/-- The given problem translated to Lean 4 statement -/
theorem problem_statement :
  ∀ (f g : ℝ → ℝ),
  (∀ m n : ℝ, synchronous f g m n → (m,n) ⊆ Icc 0 1) ∧
  ¬ synchronous (λ x, x^2) (λ x, 2 * x) 1 4 ∧
  (∃ n ∈ Ioo (1/2 : ℝ) 1, synchronous (λ x, exp x - 1) (λ x, sin (π * x)) 0 n) ∧
  (∀ (m n : ℝ), synchronous (λ x, a * log x) (λ x, x ^ 2) m n → a > 2 * exp 1) ∧
  ∃ m n : ℝ, ¬ synchronous (λ x, x + 1) (λ x, log (x + 1)) m n :=
sorry

end problem_statement_l185_185948


namespace endpoint_sum_l185_185110

theorem endpoint_sum
  (x y : ℤ)
  (H_midpoint_x : (x + 15) / 2 = 10)
  (H_midpoint_y : (y - 8) / 2 = -3) :
  x + y = 7 :=
sorry

end endpoint_sum_l185_185110


namespace adam_final_amount_l185_185458

def initial_savings : ℝ := 1579.37
def money_received_monday : ℝ := 21.85
def money_received_tuesday : ℝ := 33.28
def money_spent_wednesday : ℝ := 87.41

def total_money_received : ℝ := money_received_monday + money_received_tuesday
def new_total_after_receiving : ℝ := initial_savings + total_money_received
def final_amount : ℝ := new_total_after_receiving - money_spent_wednesday

theorem adam_final_amount : final_amount = 1547.09 := by
  -- proof omitted
  sorry

end adam_final_amount_l185_185458


namespace part_one_part_two_l185_185348

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2

theorem part_one (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m * n > 1) : f m >= 0 ∨ f n >= 0 :=
sorry

theorem part_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hf : f a = f b) : a + b < 4 / 3 :=
sorry

end part_one_part_two_l185_185348


namespace net_effect_on_sale_value_l185_185435

theorem net_effect_on_sale_value (P Q : ℝ) (hP : P > 0) (hQ : Q > 0) :
  let original_sale_value := P * Q
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let new_sale_value := new_price * new_quantity
  let net_effect := (new_sale_value / original_sale_value - 1) * 100
  net_effect = 54.16 :=
by
  sorry

end net_effect_on_sale_value_l185_185435


namespace triangle_angle_A_is_60_degrees_l185_185045

theorem triangle_angle_A_is_60_degrees
  (a b c : ℚ) 
  (h1 : (a + Real.sqrt 2)^2 = (b + Real.sqrt 2) * (c + Real.sqrt 2)) : 
  ∠A = 60 := 
sorry

end triangle_angle_A_is_60_degrees_l185_185045


namespace B_can_complete_work_in_6_days_l185_185875

theorem B_can_complete_work_in_6_days (A B : ℝ) (h1 : (A + B) = 1 / 4) (h2 : A = 1 / 12) : B = 1 / 6 := 
by
  sorry

end B_can_complete_work_in_6_days_l185_185875


namespace remaining_dimes_l185_185727

-- Conditions
def initial_pennies : Nat := 7
def initial_dimes : Nat := 8
def borrowed_dimes : Nat := 4

-- Define the theorem
theorem remaining_dimes : initial_dimes - borrowed_dimes = 4 := by
  -- Use the conditions to state the remaining dimes
  sorry

end remaining_dimes_l185_185727


namespace sales_on_same_days_l185_185442

-- Definitions representing the conditions
def bookstore_sales_days : List ℕ := [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
def toy_store_sales_days : List ℕ := [2, 9, 16, 23, 30]

-- Lean statement to prove the number of common sale days
theorem sales_on_same_days : (bookstore_sales_days ∩ toy_store_sales_days).length = 2 :=
by sorry

end sales_on_same_days_l185_185442


namespace each_student_contribution_l185_185445

-- Definitions for conditions in the problem
def numberOfStudents : ℕ := 30
def totalAmount : ℕ := 480
def numberOfFridaysInTwoMonths : ℕ := 8

-- Statement to prove
theorem each_student_contribution (numberOfStudents : ℕ) (totalAmount : ℕ) (numberOfFridaysInTwoMonths : ℕ) : 
  totalAmount / (numberOfFridaysInTwoMonths * numberOfStudents) = 2 := 
by
  sorry

end each_student_contribution_l185_185445


namespace initial_bags_count_l185_185741

theorem initial_bags_count
  (points_per_bag : ℕ)
  (non_recycled_bags : ℕ)
  (total_possible_points : ℕ)
  (points_earned : ℕ)
  (B : ℕ)
  (h1 : points_per_bag = 5)
  (h2 : non_recycled_bags = 2)
  (h3 : total_possible_points = 45)
  (h4 : points_earned = 5 * (B - non_recycled_bags))
  : B = 11 :=
by {
  sorry
}

end initial_bags_count_l185_185741


namespace larger_number_is_28_l185_185363

theorem larger_number_is_28
  (x y : ℕ)
  (h1 : 4 * y = 7 * x)
  (h2 : y - x = 12) : y = 28 :=
sorry

end larger_number_is_28_l185_185363


namespace simplify_fraction_l185_185266

-- Define the fractions and the product
def fraction1 : ℚ := 18 / 11
def fraction2 : ℚ := -42 / 45
def product : ℚ := 15 * fraction1 * fraction2

-- State the theorem to prove the correctness of the simplification
theorem simplify_fraction : product = -23 + 1 / 11 :=
by
  -- Adding this as a placeholder. The proof would go here.
  sorry

end simplify_fraction_l185_185266


namespace AdultsNotWearingBlue_l185_185903

theorem AdultsNotWearingBlue (number_of_children : ℕ) (number_of_adults : ℕ) (adults_who_wore_blue : ℕ) :
  number_of_children = 45 → 
  number_of_adults = number_of_children / 3 → 
  adults_who_wore_blue = number_of_adults / 3 → 
  number_of_adults - adults_who_wore_blue = 10 :=
by
  sorry

end AdultsNotWearingBlue_l185_185903


namespace find_n_value_l185_185784

theorem find_n_value (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 9) : n = 210 := sorry

end find_n_value_l185_185784


namespace simplify_fraction_l185_185590

theorem simplify_fraction (a b : ℝ) :
  ( (3 * b) / (2 * a^2) )^3 = 27 * b^3 / (8 * a^6) :=
by
  sorry

end simplify_fraction_l185_185590


namespace six_digit_number_contains_7_l185_185252

theorem six_digit_number_contains_7
  (a b k : ℤ)
  (h1 : 100 ≤ 7 * a + k ∧ 7 * a + k < 1000)
  (h2 : 100 ≤ 7 * b + k ∧ 7 * b + k < 1000) :
  7 ∣ (1000 * (7 * a + k) + (7 * b + k)) :=
by
  sorry

end six_digit_number_contains_7_l185_185252


namespace Polly_lunch_time_l185_185079

-- Define the conditions
def breakfast_time_per_day := 20
def total_days_in_week := 7
def dinner_time_4_days := 10
def remaining_days_in_week := 3
def remaining_dinner_time_per_day := 30
def total_cooking_time := 305

-- Define the total time Polly spends cooking breakfast in a week
def total_breakfast_time := breakfast_time_per_day * total_days_in_week

-- Define the total time Polly spends cooking dinner in a week
def total_dinner_time := (dinner_time_4_days * 4) + (remaining_dinner_time_per_day * remaining_days_in_week)

-- Define the time Polly spends cooking lunch in a week
def lunch_time := total_cooking_time - (total_breakfast_time + total_dinner_time)

-- The theorem to prove Polly's lunch time
theorem Polly_lunch_time : lunch_time = 35 :=
by
  sorry

end Polly_lunch_time_l185_185079


namespace find_b_l185_185109

theorem find_b
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/12) * x^2 + a * x + b)
  (A C: ℝ × ℝ)
  (hA : A = (x1, 0))
  (hC : C = (x2, 0))
  (T : ℝ × ℝ)
  (hT : T = (3, 3))
  (h_TA : dist (3, 3) (x1, 0) = dist (3, 3) (0, b))
  (h_TB : dist (3, 3) (0, b) = dist (3, 3) (x2, 0))
  (vietas : x1 * x2 = 12 * b)
  : b = -6 := 
sorry

end find_b_l185_185109


namespace mr_klinker_twice_as_old_l185_185843

theorem mr_klinker_twice_as_old (x : ℕ) (current_age_klinker : ℕ) (current_age_daughter : ℕ)
  (h1 : current_age_klinker = 35) (h2 : current_age_daughter = 10) 
  (h3 : current_age_klinker + x = 2 * (current_age_daughter + x)) : 
  x = 15 :=
by 
  -- We include sorry to indicate where the proof should be
  sorry

end mr_klinker_twice_as_old_l185_185843


namespace factorize_x9_minus_512_l185_185911

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l185_185911


namespace find_pairs_l185_185785

theorem find_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : (m^2 - n) ∣ (m + n^2)) (h4 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3) := by
  sorry

end find_pairs_l185_185785


namespace apple_tree_total_apples_l185_185894

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l185_185894


namespace range_of_dot_product_l185_185546

theorem range_of_dot_product
  (x y : ℝ)
  (on_ellipse : x^2 / 2 + y^2 = 1) :
  ∃ m n : ℝ, (m = 0) ∧ (n = 1) ∧ m ≤ x^2 / 2 ∧ x^2 / 2 ≤ n :=
sorry

end range_of_dot_product_l185_185546


namespace total_animals_seen_l185_185018

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end total_animals_seen_l185_185018


namespace inequality_holds_l185_185198

variable {f : ℝ → ℝ}

-- Conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_monotonic_on_nonneg_interval (f : ℝ → ℝ) : Prop := ∀ x y, (0 ≤ x ∧ x < y ∧ y < 8) → f y ≤ f x

axiom condition1 : is_even f
axiom condition2 : is_monotonic_on_nonneg_interval f
axiom condition3 : f (-3) < f 2

-- The statement to be proven
theorem inequality_holds : f 5 < f (-3) ∧ f (-3) < f (-1) :=
by
  sorry

end inequality_holds_l185_185198


namespace smallest_possible_e_l185_185735

-- Definitions based on given conditions
def polynomial (x : ℝ) (a b c d e : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- The given polynomial has roots -3, 4, 8, and -1/4, and e is positive integer
theorem smallest_possible_e :
  ∃ (a b c d e : ℤ), polynomial x a b c d e = 4*x^4 - 32*x^3 - 23*x^2 + 104*x + 96 ∧ e > 0 ∧ e = 96 :=
by
  sorry

end smallest_possible_e_l185_185735


namespace distinct_values_of_c_l185_185388

theorem distinct_values_of_c {c p q : ℂ} 
  (h_distinct : p ≠ q) 
  (h_eq : ∀ z : ℂ, (z - p) * (z - q) = (z - c * p) * (z - c * q)) :
  (∃ c_values : ℕ, c_values = 2) :=
sorry

end distinct_values_of_c_l185_185388


namespace find_x_in_terms_of_a_b_l185_185071

variable (a b x : ℝ)
variable (ha : a > 0) (hb : b > 0) (hx : x > 0) (r : ℝ)
variable (h1 : r = (4 * a)^(3 * b))
variable (h2 : r = a ^ b * x ^ b)

theorem find_x_in_terms_of_a_b 
  (ha : a > 0) (hb : b > 0) (hx : x > 0)
  (h1 : (4 * a)^(3 * b) = r)
  (h2 : r = a^b * x^b) :
  x = 64 * a^2 :=
by
  sorry

end find_x_in_terms_of_a_b_l185_185071


namespace joe_eats_different_fruits_l185_185382

noncomputable def joe_probability : ℚ :=
  let single_fruit_prob := (1 / 3) ^ 4
  let all_same_fruit_prob := 3 * single_fruit_prob
  let at_least_two_diff_fruits_prob := 1 - all_same_fruit_prob
  at_least_two_diff_fruits_prob

theorem joe_eats_different_fruits :
  joe_probability = 26 / 27 :=
by
  -- The proof is omitted for this task
  sorry

end joe_eats_different_fruits_l185_185382


namespace truck_capacity_cost_function_minimum_cost_l185_185084

theorem truck_capacity :
  ∃ (m n : ℕ),
    3 * m + 4 * n = 27 ∧ 
    4 * m + 5 * n = 35 ∧
    m = 5 ∧ 
    n = 3 :=
by {
  sorry
}

theorem cost_function (a : ℕ) (h : a ≤ 5) :
  ∃ (w : ℕ),
    w = 50 * a + 2250 :=
by {
  sorry
}

theorem minimum_cost :
  ∃ (w : ℕ),
    w = 2250 ∧ 
    ∀ (a : ℕ), a ≤ 5 → (50 * a + 2250) ≥ 2250 :=
by {
  sorry
}

end truck_capacity_cost_function_minimum_cost_l185_185084


namespace find_J_l185_185613

-- Define the problem conditions
def eq1 : Nat := 32
def eq2 : Nat := 4

-- Define the target equation form
def target_eq (J : Nat) : Prop := (eq1^3) * (eq2^3) = 2^J

theorem find_J : ∃ J : Nat, target_eq J ∧ J = 21 :=
by
  -- Rest of the proof goes here
  sorry

end find_J_l185_185613


namespace effect_on_revenue_l185_185436

-- Define the conditions using parameters and variables

variables {P Q : ℝ} -- Original price and quantity of TV sets

def new_price (P : ℝ) : ℝ := P * 1.60 -- New price after 60% increase
def new_quantity (Q : ℝ) : ℝ := Q * 0.80 -- New quantity after 20% decrease

def original_revenue (P Q : ℝ) : ℝ := P * Q -- Original revenue
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q) -- New revenue

theorem effect_on_revenue
  (P Q : ℝ) :
  new_revenue P Q = original_revenue P Q * 1.28 :=
by
  sorry

end effect_on_revenue_l185_185436


namespace certain_number_l185_185374

theorem certain_number (x : ℝ) (h : x - 4 = 2) : x^2 - 3 * x = 18 :=
by
  -- Proof yet to be completed
  sorry

end certain_number_l185_185374


namespace max_amount_paul_received_l185_185847

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end max_amount_paul_received_l185_185847


namespace domain_of_v_l185_185428

def domain_v (x : ℝ) : Prop :=
  x ≥ 2 ∧ x ≠ 5

theorem domain_of_v :
  {x : ℝ | domain_v x} = { x | 2 < x ∧ x < 5 } ∪ { x | 5 < x }
:= by
  sorry

end domain_of_v_l185_185428


namespace binom_1300_2_eq_l185_185315

theorem binom_1300_2_eq : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_eq_l185_185315


namespace solve_inequality_l185_185051

theorem solve_inequality (a b : ℝ) (h : ∀ x, (x > 1 ∧ x < 2) ↔ (x - a) * (x - b) < 0) : a + b = 3 :=
sorry

end solve_inequality_l185_185051


namespace student_correct_sums_l185_185162

theorem student_correct_sums (x wrong total : ℕ) (h1 : wrong = 2 * x) (h2 : total = x + wrong) (h3 : total = 54) : x = 18 :=
by
  sorry

end student_correct_sums_l185_185162


namespace cos_pi_over_3_plus_2alpha_l185_185029

theorem cos_pi_over_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_l185_185029


namespace smallest_angle_WYZ_l185_185243

-- Define the given angle measures.
def angle_XYZ : ℝ := 40
def angle_XYW : ℝ := 15

-- The theorem statement proving the smallest possible degree measure for ∠WYZ
theorem smallest_angle_WYZ : angle_XYZ - angle_XYW = 25 :=
by
  -- Add the proof here
  sorry

end smallest_angle_WYZ_l185_185243


namespace password_correct_l185_185571

-- conditions
def poly1 (x y : ℤ) : ℤ := x ^ 4 - y ^ 4
def factor1 (x y : ℤ) : ℤ := (x - y) * (x + y) * (x ^ 2 + y ^ 2)

def poly2 (x y : ℤ) : ℤ := x ^ 3 - x * y ^ 2
def factor2 (x y : ℤ) : ℤ := x * (x - y) * (x + y)

-- given values
def x := 18
def y := 5

-- goal
theorem password_correct : factor2 x y = 18 * 13 * 23 :=
by
  -- We setup the goal with the equivalent sequence of the password generation
  sorry

end password_correct_l185_185571


namespace find_x_l185_185068

noncomputable def h (x : ℝ) : ℝ := (2 * x^2 + 3 * x + 1)^(1 / 3) / 5^(1/3)

theorem find_x (x : ℝ) :
  h (3 * x) = 3 * h x ↔ x = -1 + (10^(1/2)) / 3 ∨ x = -1 - (10^(1/2)) / 3 := by
  sorry

end find_x_l185_185068


namespace quadratic_solution_l185_185113

theorem quadratic_solution (x : ℝ) :
  (x^2 + 2 * x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end quadratic_solution_l185_185113


namespace length_of_parallel_at_60N_l185_185417

noncomputable def parallel_length (R : ℝ) (lat_deg : ℝ) : ℝ :=
  2 * Real.pi * R * Real.cos (Real.pi * lat_deg / 180)

theorem length_of_parallel_at_60N :
  parallel_length 20 60 = 20 * Real.pi :=
by
  sorry

end length_of_parallel_at_60N_l185_185417


namespace evaluate_expression_l185_185925

theorem evaluate_expression : (∃ (x : Real), 6 < x ∧ x < 7 ∧ x = Real.sqrt 45) → (Int.floor (Real.sqrt 45))^2 + 2*Int.floor (Real.sqrt 45) + 1 = 49 := 
by
  sorry

end evaluate_expression_l185_185925


namespace fraction_expression_simplifies_to_313_l185_185915

theorem fraction_expression_simplifies_to_313 :
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324) /
  (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324) = 313 :=
by
  sorry

end fraction_expression_simplifies_to_313_l185_185915


namespace maximum_special_points_l185_185852

theorem maximum_special_points (n : ℕ) (h : n = 11) : 
  ∃ p : ℕ, p = 91 := 
sorry

end maximum_special_points_l185_185852


namespace calculate_fg1_l185_185579

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem calculate_fg1 : f (g 1) = -1 :=
by {
  sorry
}

end calculate_fg1_l185_185579


namespace incorrect_statement_A_l185_185792

theorem incorrect_statement_A (x_1 x_2 y_1 y_2 : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y - 4 = 0) ∧
  x_1 = 1 - Real.sqrt 5 ∧
  x_2 = 1 + Real.sqrt 5 ∧
  y_1 = 2 - 2 * Real.sqrt 2 ∧
  y_2 = 2 + 2 * Real.sqrt 2 →
  x_1 + x_2 ≠ -2 := by
  intro h
  sorry

end incorrect_statement_A_l185_185792


namespace distance_between_A_and_B_l185_185323

theorem distance_between_A_and_B 
    (Time_E : ℝ) (Time_F : ℝ) (D_AC : ℝ) (V_ratio : ℝ)
    (E_time : Time_E = 3) (F_time : Time_F = 4) 
    (AC_distance : D_AC = 300) (speed_ratio : V_ratio = 4) : 
    ∃ D_AB : ℝ, D_AB = 900 :=
by
  sorry

end distance_between_A_and_B_l185_185323


namespace lcm_inequality_l185_185065

theorem lcm_inequality
  (a b c d e : ℤ)
  (h1 : 1 ≤ a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (1 : ℚ) / Int.lcm a b + (1 : ℚ) / Int.lcm b c + 
  (1 : ℚ) / Int.lcm c d + (1 : ℚ) / Int.lcm d e ≤ (15 : ℚ) / 16 := by
  sorry

end lcm_inequality_l185_185065


namespace problem_statement_l185_185074

noncomputable def f (x : ℝ) : ℝ := x - Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem problem_statement (x : ℝ) (h : 0 < x ∧ x ≤ Real.exp 1) : 
  f x > g x + 1/2 :=
sorry

end problem_statement_l185_185074


namespace seeds_in_fourth_pot_l185_185326

theorem seeds_in_fourth_pot (total_seeds : ℕ) (total_pots : ℕ) (seeds_per_pot : ℕ) (first_three_pots : ℕ)
  (h1 : total_seeds = 10) (h2 : total_pots = 4) (h3 : seeds_per_pot = 3) (h4 : first_three_pots = 3) : 
  (total_seeds - (seeds_per_pot * first_three_pots)) = 1 :=
by
  sorry

end seeds_in_fourth_pot_l185_185326


namespace find_a_n_find_b_n_find_T_n_l185_185196

-- definitions of sequences and common ratios
variable (a_n b_n : ℕ → ℕ)
variable (S_n T_n : ℕ → ℕ)
variable (q : ℝ)
variable (n : ℕ)

-- conditions
axiom a1 : a_n 1 = 1
axiom S3 : S_n 3 = 9
axiom b1 : b_n 1 = 1
axiom b3 : b_n 3 = 20
axiom q_pos : q > 0
axiom geo_seq : (∀ n, b_n n / a_n n = q ^ (n - 1))

-- goals to prove
theorem find_a_n : ∀ n, a_n n = 2 * n - 1 := 
by sorry

theorem find_b_n : ∀ n, b_n n = (2 * n - 1) * 2 ^ (n - 1) := 
by sorry

theorem find_T_n : ∀ n, T_n n = (2 * n - 3) * 2 ^ n + 3 :=
by sorry

end find_a_n_find_b_n_find_T_n_l185_185196


namespace truck_loading_time_l185_185163

theorem truck_loading_time :
  let worker1_rate := (1:ℝ) / 6
  let worker2_rate := (1:ℝ) / 5
  let combined_rate := worker1_rate + worker2_rate
  (combined_rate != 0) → 
  (1 / combined_rate = (30:ℝ) / 11) :=
by
  sorry

end truck_loading_time_l185_185163


namespace remainder_when_s_div_6_is_5_l185_185862

theorem remainder_when_s_div_6_is_5 (s t : ℕ) (h1 : s > t) (Rs Rt : ℕ) (h2 : s % 6 = Rs) (h3 : t % 6 = Rt) (h4 : (s - t) % 6 = 5) : Rs = 5 := 
by
  sorry

end remainder_when_s_div_6_is_5_l185_185862


namespace arithmetic_sequence_sum_l185_185227

open Nat

theorem arithmetic_sequence_sum (m n : Nat) (d : ℤ) (a_1 : ℤ)
    (hnm : n ≠ m)
    (hSn : (n * (2 * a_1 + (n - 1) * d) / 2) = n / m)
    (hSm : (m * (2 * a_1 + (m - 1) * d) / 2) = m / n) :
  ((m + n) * (2 * a_1 + (m + n - 1) * d) / 2) > 4 := by
  sorry

end arithmetic_sequence_sum_l185_185227


namespace b_sequence_periodic_l185_185578

theorem b_sequence_periodic (b : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h_b1 : b 1 = 2 + Real.sqrt 3)
  (h_b2021 : b 2021 = 11 + Real.sqrt 3) :
  b 2048 = b 2 :=
sorry

end b_sequence_periodic_l185_185578


namespace reducible_fraction_implies_divisibility_l185_185389

theorem reducible_fraction_implies_divisibility
  (a b c d l k : ℤ)
  (m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n)
  : k ∣ (a * d - b * c) :=
by
  sorry

end reducible_fraction_implies_divisibility_l185_185389


namespace equal_cost_sharing_l185_185466

variable (X Y Z : ℝ)
variable (h : X < Y ∧ Y < Z)

theorem equal_cost_sharing :
  ∃ (amount : ℝ), amount = (Y + Z - 2 * X) / 3 := 
sorry

end equal_cost_sharing_l185_185466


namespace range_of_a_l185_185372

theorem range_of_a (a : ℝ) :
    (∀ x : ℤ, x + 1 > 0 → 3 * x - a ≤ 0 → x = 0 ∨ x = 1 ∨ x = 2) ↔ 6 ≤ a ∧ a < 9 :=
by
  sorry

end range_of_a_l185_185372


namespace binomial_1300_2_eq_844350_l185_185312

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end binomial_1300_2_eq_844350_l185_185312


namespace math_proof_problem_l185_185972

variable (n : ℕ) (a b : ℕ → ℕ) (S T : ℕ → ℕ)

-- Conditions
def condition_a1 : Prop := a 1 = 1
def condition_b1 : Prop := b 1 = 4
def condition_sum : Prop := ∀ n ∈ ℕ, n * S (n+1) - (n+3) * S n = 0
def condition_geom_mean : Prop := ∀ n ∈ ℕ, 2 * a (n+1) = ((b n) * (b (n+1))) / 4

-- Proof goals
def goal_a2 : Prop := a 2 = 3
def goal_b2 : Prop := b 2 = 9
def goal_general_terms_a : Prop := ∀ n ∈ ℕ, a n = n * (n+1) / 2
def goal_general_terms_b : Prop := ∀ n ∈ ℕ, b n = (n+1)^2
def goal_t : Prop := ∀ n ≥ 3, |T n| < 2 * n^2

-- Main theorem statement
theorem math_proof_problem :
  condition_a1 ∧ condition_b1 ∧ condition_sum ∧ condition_geom_mean →
  (goal_a2 ∧ goal_b2) ∧
  (goal_general_terms_a ∧ goal_general_terms_b) ∧
  goal_t :=
 by sorry

end math_proof_problem_l185_185972


namespace expected_value_of_X_l185_185633

-- Define the conditions
def row_size : ℕ := 6

-- X is the random variable denoting the number of students disturbed by the first student to exit.
noncomputable def X : ℕ → ℚ := sorry -- Define the random variable X properly to denote the disturbance

-- Probability Distribution and Expected Value calculation are based on the given conditions
theorem expected_value_of_X :
  let E : ℚ := 0 * (32 / (6!)) + 1 * (160 / (6!)) + 2 * (280 / (6!)) + 3 * (200 / (6!)) + 4 * (48 / (6!)) in
  E = 21 / 10 :=
by
  sorry

end expected_value_of_X_l185_185633


namespace smallest_positive_integer_mod_l185_185873

theorem smallest_positive_integer_mod (a : ℕ) (h1 : a ≡ 4 [MOD 5]) (h2 : a ≡ 6 [MOD 7]) : a = 34 :=
by
  sorry

end smallest_positive_integer_mod_l185_185873


namespace find_number_l185_185522

theorem find_number (x : ℝ) (h : 0.15 * x = 90) : x = 600 :=
by
  sorry

end find_number_l185_185522


namespace valid_outfit_combinations_l185_185692

theorem valid_outfit_combinations :
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  total_combinations - invalid_combinations = 205 :=
by
  let shirts := 8
  let pants := 5
  let hats := 6
  let shared_colors := 5
  let extra_colors := 2
  let total_combinations := shirts * pants * hats
  let invalid_shared_combinations := shared_colors * pants
  let invalid_extra_combinations := extra_colors * pants
  let invalid_combinations := invalid_shared_combinations + invalid_extra_combinations
  have h : total_combinations - invalid_combinations = 205 := sorry
  exact h

end valid_outfit_combinations_l185_185692


namespace minimize_sum_of_distances_l185_185061

open_locale real

-- Define the circle and points A, B
variables {O A B M : Point}
variables (circle : Circle O r)
variables (N : Point) -- Midpoint of AB

-- Define the conditions: OA = OB and M is on the circle
def is_midpoint (A B N : Point) : Prop :=
  dist A N = dist B N ∧ 2 * dist A N = dist A B

def point_on_circle (circle : Circle O r) (M : Point) : Prop :=
  dist O M = r

def minimizes_distance_sum (M A B : Point) : Prop :=
  ∀ M' : Point, M' ∈ circle → (dist M A + dist M B) ≤ (dist M' A + dist M' B)

-- The statement to prove
theorem minimize_sum_of_distances
  (h1 : dist O A = dist O B)
  (h2 : is_midpoint A B N)
  (M_in_circle : point_on_circle circle M)
  (N_midpoint : N = midpoint A B) :
  minimizes_distance_sum M A B :=
sorry

end minimize_sum_of_distances_l185_185061


namespace tangent_circle_equation_l185_185700

theorem tangent_circle_equation :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi →
    ∃ c : ℝ × ℝ, ∃ r : ℝ,
      (∀ (a b : ℝ), c = (a, b) →
        (|a * Real.cos θ + b * Real.sin θ - Real.cos θ - 2 * Real.sin θ - 2| = r) ∧
        (r = 2)) ∧
      (∃ x y : ℝ, (x - 1) ^ 2 + (y - 2) ^ 2 = r^2)) :=
by
  sorry

end tangent_circle_equation_l185_185700
