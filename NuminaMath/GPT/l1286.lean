import Mathlib

namespace solve_nine_sections_bamboo_problem_l1286_128616

-- Define the bamboo stick problem
noncomputable def nine_sections_bamboo_problem : Prop :=
∃ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) ∧ -- Arithmetic sequence
  (a 1 + a 2 + a 3 + a 4 = 3) ∧ -- Top 4 sections' total volume
  (a 7 + a 8 + a 9 = 4) ∧ -- Bottom 3 sections' total volume
  (a 5 = 67 / 66) -- Volume of the 5th section

theorem solve_nine_sections_bamboo_problem : nine_sections_bamboo_problem :=
sorry

end solve_nine_sections_bamboo_problem_l1286_128616


namespace range_of_b_l1286_128670

theorem range_of_b (b : ℝ) :
  (∀ x : ℤ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ↔ 5 < b ∧ b < 7 := 
sorry

end range_of_b_l1286_128670


namespace selling_price_is_320_l1286_128669

noncomputable def sales_volume (x : ℝ) : ℝ := 8000 / x

def cost_price : ℝ := 180

def desired_profit : ℝ := 3500

def selling_price_for_desired_profit (x : ℝ) : Prop :=
  (x - cost_price) * sales_volume x = desired_profit

/-- The selling price of the small electrical appliance to achieve a daily sales profit 
    of $3500 dollars is $320 dollars. -/
theorem selling_price_is_320 : selling_price_for_desired_profit 320 :=
by
  -- We skip the proof as per instructions
  sorry

end selling_price_is_320_l1286_128669


namespace factor_as_complete_square_l1286_128650

theorem factor_as_complete_square (k : ℝ) : (∃ a : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := 
sorry

end factor_as_complete_square_l1286_128650


namespace leonine_cats_l1286_128692

theorem leonine_cats (n : ℕ) (h : n = (4 / 5 * n) + (4 / 5)) : n = 4 :=
by
  sorry

end leonine_cats_l1286_128692


namespace value_of_M_l1286_128648

theorem value_of_M (G A M E: ℕ) (hG : G = 15)
(hGAME : G + A + M + E = 50)
(hMEGA : M + E + G + A = 55)
(hAGE : A + G + E = 40) : 
M = 15 := sorry

end value_of_M_l1286_128648


namespace sector_area_l1286_128653

theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) (hα : α = 60 * Real.pi / 180) (hl : l = 6 * Real.pi) : S = 54 * Real.pi :=
sorry

end sector_area_l1286_128653


namespace circle_x_intersect_l1286_128626

theorem circle_x_intersect (x y : ℝ) : 
  (x, y) = (0, 0) ∨ (x, y) = (10, 0) → (x = 10) :=
by
  -- conditions:
  -- The endpoints of the diameter are (0,0) and (10,10)
  -- (proving that the second intersect on x-axis has x-coordinate 10)
  sorry

end circle_x_intersect_l1286_128626


namespace apples_per_pie_l1286_128612

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

end apples_per_pie_l1286_128612


namespace animal_count_in_hollow_l1286_128659

theorem animal_count_in_hollow (heads legs : ℕ) (animals_with_odd_legs animals_with_even_legs : ℕ) :
  heads = 18 →
  legs = 24 →
  (∀ n, n % 2 = 1 → animals_with_odd_legs * 2 = heads - 2 * n) →
  (∀ m, m % 2 = 0 → animals_with_even_legs * 1 = heads - m) →
  (animals_with_odd_legs + animals_with_even_legs = 10 ∨
   animals_with_odd_legs + animals_with_even_legs = 12 ∨
   animals_with_odd_legs + animals_with_even_legs = 14) :=
sorry

end animal_count_in_hollow_l1286_128659


namespace three_over_x_solution_l1286_128680

theorem three_over_x_solution (x : ℝ) (h : 1 - 9 / x + 9 / (x^2) = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end three_over_x_solution_l1286_128680


namespace circle_area_from_circumference_l1286_128617

theorem circle_area_from_circumference (C : ℝ) (hC : C = 48 * Real.pi) : 
  ∃ m : ℝ, (∀ r : ℝ, C = 2 * Real.pi * r → (Real.pi * r^2 = m * Real.pi)) ∧ m = 576 :=
by
  sorry

end circle_area_from_circumference_l1286_128617


namespace union_A_B_correct_l1286_128639

def A : Set ℕ := {0, 1}
def B : Set ℕ := {x | 0 < x ∧ x < 3}

theorem union_A_B_correct : A ∪ B = {0, 1, 2} :=
by sorry

end union_A_B_correct_l1286_128639


namespace rosy_has_14_fish_l1286_128629

-- Define the number of Lilly's fish
def lilly_fish : ℕ := 10

-- Define the total number of fish
def total_fish : ℕ := 24

-- Define the number of Rosy's fish, which we need to prove equals 14
def rosy_fish : ℕ := total_fish - lilly_fish

-- Prove that Rosy has 14 fish
theorem rosy_has_14_fish : rosy_fish = 14 := by
  sorry

end rosy_has_14_fish_l1286_128629


namespace fraction_spent_on_dvd_l1286_128657

theorem fraction_spent_on_dvd (r l m d x : ℝ) (h1 : r = 200) (h2 : l = (1/4) * r) (h3 : m = r - l) (h4 : x = 50) (h5 : d = m - x) : d / r = 1 / 2 :=
by
  sorry

end fraction_spent_on_dvd_l1286_128657


namespace mineral_samples_per_shelf_l1286_128673

theorem mineral_samples_per_shelf (total_samples : ℕ) (num_shelves : ℕ) (h1 : total_samples = 455) (h2 : num_shelves = 7) :
  total_samples / num_shelves = 65 :=
by
  sorry

end mineral_samples_per_shelf_l1286_128673


namespace hulk_jump_kilometer_l1286_128658

theorem hulk_jump_kilometer (n : ℕ) (h : ∀ n : ℕ, n ≥ 1 → (2^(n-1) : ℕ) ≤ 1000 → n-1 < 10) : n = 11 :=
by
  sorry

end hulk_jump_kilometer_l1286_128658


namespace unique_zero_of_f_inequality_of_x1_x2_l1286_128679

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end unique_zero_of_f_inequality_of_x1_x2_l1286_128679


namespace shoes_to_belts_ratio_l1286_128695

variable (hats : ℕ) (belts : ℕ) (shoes : ℕ)

theorem shoes_to_belts_ratio (hats_eq : hats = 5)
                            (belts_eq : belts = hats + 2)
                            (shoes_eq : shoes = 14) : 
  (shoes / (Nat.gcd shoes belts)) = 2 ∧ (belts / (Nat.gcd shoes belts)) = 1 := 
by
  sorry

end shoes_to_belts_ratio_l1286_128695


namespace q_is_false_l1286_128683

theorem q_is_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end q_is_false_l1286_128683


namespace sufficient_condition_l1286_128642

theorem sufficient_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a = 0 → a < 1) ↔ 
  (∀ c : ℝ, x^2 - 2 * x + c = 0 ↔ 4 - 4 * c ≥ 0 ∧ c < 1 → ¬ (∀ d : ℝ, d ≤ 1 → d < 1)) := 
by 
sorry

end sufficient_condition_l1286_128642


namespace exists_two_positive_integers_dividing_3003_l1286_128693

theorem exists_two_positive_integers_dividing_3003 : 
  ∃ (m1 m2 : ℕ), m1 > 0 ∧ m2 > 0 ∧ m1 ≠ m2 ∧ (3003 % (m1^2 + 2) = 0) ∧ (3003 % (m2^2 + 2) = 0) :=
by
  sorry

end exists_two_positive_integers_dividing_3003_l1286_128693


namespace igor_number_proof_l1286_128611

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

end igor_number_proof_l1286_128611


namespace excircle_side_formula_l1286_128631

theorem excircle_side_formula 
  (a b c r_a r_b r_c : ℝ)
  (h1 : r_c = Real.sqrt (r_a * r_b)) :
  c = (a^2 + b^2) / (a + b) :=
sorry

end excircle_side_formula_l1286_128631


namespace additional_cats_l1286_128691

theorem additional_cats {M R C : ℕ} (h1 : 20 * R = M) (h2 : 4 + 2 * C = 10) : C = 3 := 
  sorry

end additional_cats_l1286_128691


namespace octadecagon_diagonals_l1286_128655

def num_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octadecagon_diagonals : num_of_diagonals 18 = 135 := by
  sorry

end octadecagon_diagonals_l1286_128655


namespace division_theorem_l1286_128600

noncomputable def p (z : ℝ) : ℝ := 4 * z ^ 3 - 8 * z ^ 2 + 9 * z - 7
noncomputable def d (z : ℝ) : ℝ := 4 * z + 2
noncomputable def q (z : ℝ) : ℝ := z ^ 2 - 2.5 * z + 3.5
def r : ℝ := -14

theorem division_theorem (z : ℝ) : p z = d z * q z + r := 
by
  sorry

end division_theorem_l1286_128600


namespace arc_length_of_f_l1286_128678

noncomputable def f (x : ℝ) : ℝ := 2 - Real.exp x

theorem arc_length_of_f :
  ∫ x in Real.log (Real.sqrt 3)..Real.log (Real.sqrt 8), Real.sqrt (1 + (Real.exp x)^2) = 1 + 1/2 * Real.log (3 / 2) :=
by
  sorry

end arc_length_of_f_l1286_128678


namespace remainder_17_pow_63_mod_7_l1286_128615

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l1286_128615


namespace rhombus_area_l1286_128687

-- Define the given conditions: diagonals and side length
def d1 : ℕ := 40
def d2 : ℕ := 18
def s : ℕ := 25

-- Prove that the area of the rhombus is 360 square units given the conditions
theorem rhombus_area :
  (d1 * d2) / 2 = 360 :=
by
  sorry

end rhombus_area_l1286_128687


namespace simplify_radical_expr_l1286_128605

-- Define the variables and expressions
variables {x : ℝ} (hx : 0 ≤ x) 

-- State the problem
theorem simplify_radical_expr (hx : 0 ≤ x) :
  (Real.sqrt (100 * x)) * (Real.sqrt (3 * x)) * (Real.sqrt (18 * x)) = 30 * x * Real.sqrt (6 * x) :=
sorry

end simplify_radical_expr_l1286_128605


namespace arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l1286_128645

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * (-2)) → 
  a 2 = 1 → 
  a 5 = -5 → 
  ∀ n : ℕ, a n = -2 * n + 5 :=
by
  intros h₁ h₂ h₅
  sorry

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (-2)) →
  a 2 = 1 → 
  a 5 = -5 → 
  ∃ n : ℕ, n = 2 ∧ S n = 4 :=
by
  intros hSn h₂ h₅
  sorry

end arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l1286_128645


namespace range_of_a_l1286_128601

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else 2 + Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x, 3 ≤ f x a) ∧ (0 < a) ∧ (a ≠ 1) → 1 < a ∧ a ≤ 2 :=
by
  intro h
  sorry

end range_of_a_l1286_128601


namespace system_of_equations_solution_l1286_128667

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 : ℝ), 
    (x1 + 2 * x2 = 10) ∧
    (3 * x1 + 2 * x2 + x3 = 23) ∧
    (x2 + 2 * x3 = 13) ∧
    (x1 = 4) ∧
    (x2 = 3) ∧
    (x3 = 5) :=
sorry

end system_of_equations_solution_l1286_128667


namespace problem_statement_l1286_128690

noncomputable def general_term (a : ℕ → ℕ) (n : ℕ) : Prop :=
a n = n

noncomputable def sum_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n, S n = (n * (n + 1)) / 2

noncomputable def b_def (S : ℕ → ℕ) (b : ℕ → ℚ) : Prop :=
∀ n, b n = (2 : ℚ) / (S n)

noncomputable def sum_b_first_n_terms (b : ℕ → ℚ) (T : ℕ → ℚ) : Prop :=
∀ n, T n = (4 * n) / (n + 1)

theorem problem_statement (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (∀ n, a n = 1 + (n - 1) * 1) →
  a 1 = 1 →
  (∀ n, a (n + 1) - a n ≠ 0) →
  a 3 ^ 2 = a 1 * a 9 →
  general_term a 1 →
  sum_first_n_terms a S →
  b_def S b →
  sum_b_first_n_terms b T :=
by
  intro arithmetic_seq
  intro a_1_eq_1
  intro non_zero_diff
  intro geometric_seq
  intro gen_term_cond
  intro sum_terms_cond
  intro b_def_cond
  intro sum_b_terms_cond
  -- The proof goes here.
  sorry

end problem_statement_l1286_128690


namespace distinguishable_arrangements_l1286_128623

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

end distinguishable_arrangements_l1286_128623


namespace quadrant_of_angle_l1286_128654

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃! (q : ℕ), q = 2 :=
sorry

end quadrant_of_angle_l1286_128654


namespace percentage_of_y_in_relation_to_25_percent_of_x_l1286_128656

variable (y x : ℕ) (p : ℕ)

-- Conditions
def condition1 : Prop := (y = (p * 25 * x) / 10000)
def condition2 : Prop := (y * x = 100 * 100)
def condition3 : Prop := (y = 125)

-- The proof goal
theorem percentage_of_y_in_relation_to_25_percent_of_x :
  condition1 y x p ∧ condition2 y x ∧ condition3 y → ((y * 100) / (25 * x / 100) = 625)
:= by
-- Here we would insert the proof steps, but they are omitted as per the requirements.
sorry

end percentage_of_y_in_relation_to_25_percent_of_x_l1286_128656


namespace expression_in_terms_of_p_and_q_l1286_128682

theorem expression_in_terms_of_p_and_q (x : ℝ) :
  let p := (1 - Real.cos x) * (1 + Real.sin x)
  let q := (1 + Real.cos x) * (1 - Real.sin x)
  (Real.cos x ^ 2 - Real.cos x ^ 4 - Real.sin (2 * x) + 2) = p * q - (p + q) :=
by
  sorry

end expression_in_terms_of_p_and_q_l1286_128682


namespace x_intercept_of_perpendicular_line_l1286_128646

theorem x_intercept_of_perpendicular_line (x y : ℝ) (b : ℕ) :
  let line1 := 2 * x + 3 * y
  let slope1 := -2/3
  let slope2 := 3/2
  let y_intercept := -1
  let perp_line := slope2 * x + y_intercept
  let x_intercept := 2/3
  line1 = 12 → perp_line = 0 → x = x_intercept :=
by
  sorry

end x_intercept_of_perpendicular_line_l1286_128646


namespace profit_june_correct_l1286_128661

-- Define conditions
def profit_in_May : ℝ := 20000
def profit_in_July : ℝ := 28800

-- Define the monthly growth rate variable
variable (x : ℝ)

-- The growth factor per month
def growth_factor : ℝ := 1 + x

-- Given condition translated to an equation
def profit_relation (x : ℝ) : Prop :=
  profit_in_May * (growth_factor x) * (growth_factor x) = profit_in_July

-- The profit in June should be computed
def profit_in_June (x : ℝ) : ℝ :=
  profit_in_May * (growth_factor x)

-- The target profit in June we want to prove
def target_profit_in_June := 24000

-- Statement to prove
theorem profit_june_correct (h : profit_relation x) : profit_in_June x = target_profit_in_June :=
  sorry  -- proof to be completed

end profit_june_correct_l1286_128661


namespace yogurt_combinations_l1286_128625

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

end yogurt_combinations_l1286_128625


namespace minimum_operations_to_transfer_beer_l1286_128665

-- Definition of the initial conditions
structure InitialState where
  barrel_quarts : ℕ := 108
  seven_quart_vessel : ℕ := 0
  five_quart_vessel : ℕ := 0

-- Definition of the desired final state after minimum steps
structure FinalState where
  operations : ℕ := 17

-- Main theorem statement
theorem minimum_operations_to_transfer_beer (s : InitialState) : FinalState :=
  sorry

end minimum_operations_to_transfer_beer_l1286_128665


namespace quadratic_inequality_solution_l1286_128613

theorem quadratic_inequality_solution (b c : ℝ) 
    (h1 : ∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + b * x + c < 0) :
    b + c = -1 :=
sorry

end quadratic_inequality_solution_l1286_128613


namespace different_color_socks_l1286_128685

def total_socks := 15
def white_socks := 6
def brown_socks := 5
def blue_socks := 4

theorem different_color_socks (total : ℕ) (white : ℕ) (brown : ℕ) (blue : ℕ) :
  total = white + brown + blue →
  white ≠ 0 → brown ≠ 0 → blue ≠ 0 →
  (white * brown + brown * blue + white * blue) = 74 :=
by
  intros
  -- proof goes here
  sorry

end different_color_socks_l1286_128685


namespace no_positive_integer_k_for_rational_solutions_l1286_128622

theorem no_positive_integer_k_for_rational_solutions :
  ∀ k : ℕ, k > 0 → ¬ ∃ m : ℤ, 12 * (27 - k ^ 2) = m ^ 2 := by
  sorry

end no_positive_integer_k_for_rational_solutions_l1286_128622


namespace quadrilateral_rectangle_ratio_l1286_128608

theorem quadrilateral_rectangle_ratio
  (s x y : ℝ)
  (h_area : (s + 2 * x) ^ 2 = 4 * s ^ 2)
  (h_y : 2 * y = s) :
  y / x = 1 :=
by
  sorry

end quadrilateral_rectangle_ratio_l1286_128608


namespace blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l1286_128689

variables (length magnitude : ℕ)
variable (price : ℝ)
variable (area : ℕ)

-- Definitions based on the conditions
def length_is_about_4 (length : ℕ) : Prop := length = 4
def price_is_about_9_50 (price : ℝ) : Prop := price = 9.50
def large_area_is_about_3 (area : ℕ) : Prop := area = 3
def small_area_is_about_1 (area : ℕ) : Prop := area = 1

-- Proof problem statements
theorem blackboard_length_is_meters : length_is_about_4 length → length = 4 := by sorry
theorem pencil_case_price_is_yuan : price_is_about_9_50 price → price = 9.50 := by sorry
theorem campus_area_is_hectares : large_area_is_about_3 area → area = 3 := by sorry
theorem fingernail_area_is_square_centimeters : small_area_is_about_1 area → area = 1 := by sorry

end blackboard_length_is_meters_pencil_case_price_is_yuan_campus_area_is_hectares_fingernail_area_is_square_centimeters_l1286_128689


namespace total_payment_correct_l1286_128604

def payment_X (payment_Y : ℝ) : ℝ := 1.2 * payment_Y
def payment_Y : ℝ := 254.55
def total_payment (payment_X payment_Y : ℝ) : ℝ := payment_X + payment_Y

theorem total_payment_correct :
  total_payment (payment_X payment_Y) payment_Y = 560.01 :=
by
  sorry

end total_payment_correct_l1286_128604


namespace skipping_times_eq_l1286_128628

theorem skipping_times_eq (x : ℝ) (h : x > 0) :
  180 / x = 240 / (x + 5) :=
sorry

end skipping_times_eq_l1286_128628


namespace question_b_l1286_128627

theorem question_b (a b c : ℝ) (h : c ≠ 0) (h_eq : a / c = b / c) : a = b := 
by
  sorry

end question_b_l1286_128627


namespace trisha_total_distance_l1286_128688

-- Define each segment of Trisha's walk in miles
def hotel_to_postcard : ℝ := 0.1111111111111111
def postcard_to_tshirt : ℝ := 0.2222222222222222
def tshirt_to_keychain : ℝ := 0.7777777777777778
def keychain_to_toy : ℝ := 0.5555555555555556
def meters_to_miles (m : ℝ) : ℝ := m * 0.000621371
def toy_to_bookstore : ℝ := meters_to_miles 400
def bookstore_to_hotel : ℝ := 0.6666666666666666

-- Sum of all distances
def total_distance : ℝ :=
  hotel_to_postcard +
  postcard_to_tshirt +
  tshirt_to_keychain +
  keychain_to_toy +
  toy_to_bookstore +
  bookstore_to_hotel

-- Proof statement
theorem trisha_total_distance : total_distance = 1.5818817333333333 := by
  sorry

end trisha_total_distance_l1286_128688


namespace distance_between_house_and_school_l1286_128677

theorem distance_between_house_and_school (T D : ℕ) 
    (h1 : D = 10 * (T + 2)) 
    (h2 : D = 20 * (T - 1)) : 
    D = 60 := by
  sorry

end distance_between_house_and_school_l1286_128677


namespace solve_abs_eq_l1286_128696

theorem solve_abs_eq (x : ℝ) : |2*x - 6| = 3*x + 6 ↔ x = 0 :=
by 
  sorry

end solve_abs_eq_l1286_128696


namespace rationalize_denominator_l1286_128668

theorem rationalize_denominator (t : ℝ) (h : t = 1 / (1 - Real.sqrt (Real.sqrt 2))) : 
  t = -(1 + Real.sqrt (Real.sqrt 2)) * (1 + Real.sqrt 2) :=
by
  sorry

end rationalize_denominator_l1286_128668


namespace range_of_m_l1286_128674

theorem range_of_m :
  ∀ m, (∀ x, m ≤ x ∧ x ≤ 4 → (0 ≤ -x^2 + 4*x ∧ -x^2 + 4*x ≤ 4)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l1286_128674


namespace total_animals_l1286_128697

theorem total_animals (total_legs : ℕ) (number_of_sheep : ℕ)
  (legs_per_chicken : ℕ) (legs_per_sheep : ℕ)
  (H1 : total_legs = 60) 
  (H2 : number_of_sheep = 10)
  (H3 : legs_per_chicken = 2)
  (H4 : legs_per_sheep = 4) : 
  number_of_sheep + (total_legs - number_of_sheep * legs_per_sheep) / legs_per_chicken = 20 :=
by {
  sorry
}

end total_animals_l1286_128697


namespace target_destroyed_probability_l1286_128635

noncomputable def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  let miss1 := 1 - p1
  let miss2 := 1 - p2
  let miss3 := 1 - p3
  let prob_all_miss := miss1 * miss2 * miss3
  let prob_one_hit := (p1 * miss2 * miss3) + (miss1 * p2 * miss3) + (miss1 * miss2 * p3)
  let prob_destroyed := 1 - (prob_all_miss + prob_one_hit)
  prob_destroyed

theorem target_destroyed_probability :
  probability_hit 0.9 0.9 0.8 = 0.954 :=
sorry

end target_destroyed_probability_l1286_128635


namespace roots_seventh_sum_l1286_128651

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end roots_seventh_sum_l1286_128651


namespace find_integer_k_l1286_128694

noncomputable def P : ℤ → ℤ := sorry

theorem find_integer_k :
  P 1 = 2019 ∧ P 2019 = 1 ∧ ∃ k : ℤ, P k = k ∧ k = 1010 :=
by
  sorry

end find_integer_k_l1286_128694


namespace game_winner_Aerith_first_game_winner_Bob_first_l1286_128643

-- Conditions: row of 20 squares, players take turns crossing out one square,
-- game ends when there are two squares left, Aerith wins if two remaining squares
-- are adjacent, Bob wins if they are not adjacent.

-- Definition of the game and winning conditions
inductive Player
| Aerith
| Bob

-- Function to determine the winner given the initial player
def winning_strategy (initial_player : Player) : Player :=
  match initial_player with
  | Player.Aerith => Player.Bob  -- Bob wins if Aerith goes first
  | Player.Bob    => Player.Aerith  -- Aerith wins if Bob goes first

-- Statement to prove
theorem game_winner_Aerith_first : 
  winning_strategy Player.Aerith = Player.Bob :=
by 
  sorry -- Proof is to be done

theorem game_winner_Bob_first :
  winning_strategy Player.Bob = Player.Aerith :=
by
  sorry -- Proof is to be done

end game_winner_Aerith_first_game_winner_Bob_first_l1286_128643


namespace find_a_l1286_128641

open Real

variable (a : ℝ)

theorem find_a (h : 4 * a + -5 * 3 = 0) : a = 15 / 4 :=
sorry

end find_a_l1286_128641


namespace cricket_players_count_l1286_128602

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

end cricket_players_count_l1286_128602


namespace correct_final_positions_l1286_128606

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

end correct_final_positions_l1286_128606


namespace geometric_proportion_exists_l1286_128699

theorem geometric_proportion_exists (x y : ℝ) (h1 : x + (24 - x) = 24) 
  (h2 : y + (16 - y) = 16) (h3 : x^2 + y^2 + (16 - y)^2 + (24 - x)^2 = 580) : 
  (21 / 7 = 9 / 3) :=
  sorry

end geometric_proportion_exists_l1286_128699


namespace inequality_proof_l1286_128671

variable {a b : ℝ}

theorem inequality_proof (h : a > b) : 2 - a < 2 - b :=
by
  sorry

end inequality_proof_l1286_128671


namespace vector_at_t_zero_l1286_128644

theorem vector_at_t_zero :
  ∃ a d : ℝ × ℝ, (a + d = (2, 5) ∧ a + 4 * d = (11, -7)) ∧ a = (-1, 9) ∧ a + 0 * d = (-1, 9) :=
by {
  sorry
}

end vector_at_t_zero_l1286_128644


namespace gas_volume_at_12_l1286_128684

variable (VolumeTemperature : ℕ → ℕ) -- a function representing the volume of gas at a given temperature 

axiom condition1 : ∀ t : ℕ, VolumeTemperature (t + 4) = VolumeTemperature t + 5

axiom condition2 : VolumeTemperature 28 = 35

theorem gas_volume_at_12 :
  VolumeTemperature 12 = 15 := 
sorry

end gas_volume_at_12_l1286_128684


namespace total_number_of_trees_l1286_128672

variable {T : ℕ} -- Define T as a natural number (total number of trees)
variable (h1 : 70 / 100 * T + 105 = T) -- Indicates 30% of T is 105

theorem total_number_of_trees (h1 : 70 / 100 * T + 105 = T) : T = 350 :=
by
sorry

end total_number_of_trees_l1286_128672


namespace find_number_l1286_128633

theorem find_number (x y : ℝ) (h1 : x = y + 0.25 * y) (h2 : x = 110) : y = 88 := 
by
  sorry

end find_number_l1286_128633


namespace original_price_l1286_128610

theorem original_price (P: ℝ) (h: 0.80 * 1.15 * P = 46) : P = 50 :=
by sorry

end original_price_l1286_128610


namespace problem1_problem2_l1286_128632

def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

theorem problem1 (x : ℝ) : f x (-1) ≤ 2 ↔ -1 / 2 ≤ x ∧ x ≤ 1 / 2 :=
by sorry

theorem problem2 (a : ℝ) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 1, f x a ≤ |2 * x + 1|) → (0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l1286_128632


namespace relationship_l1286_128663

noncomputable def a : ℝ := Real.log (Real.log Real.pi)
noncomputable def b : ℝ := Real.log Real.pi
noncomputable def c : ℝ := 2^Real.log Real.pi

theorem relationship (a b c : ℝ) (ha : a = Real.log (Real.log Real.pi)) (hb : b = Real.log Real.pi) (hc : c = 2^Real.log Real.pi)
: a < b ∧ b < c := 
by
  sorry

end relationship_l1286_128663


namespace angle_measure_l1286_128698

theorem angle_measure (x : ℝ) (h1 : x + 3 * x^2 + 10 = 90) : x = 5 :=
by
  sorry

end angle_measure_l1286_128698


namespace total_expenditure_correct_l1286_128647

-- Define the weekly costs based on the conditions
def cost_white_bread : Float := 2 * 3.50
def cost_baguette : Float := 1.50
def cost_sourdough_bread : Float := 2 * 4.50
def cost_croissant : Float := 2.00

-- Total weekly cost calculation
def weekly_cost : Float := cost_white_bread + cost_baguette + cost_sourdough_bread + cost_croissant

-- Total cost over 4 weeks
def total_cost_4_weeks (weeks : Float) : Float := weekly_cost * weeks

-- The assertion that needs to be proved
theorem total_expenditure_correct :
  total_cost_4_weeks 4 = 78.00 := by
  sorry

end total_expenditure_correct_l1286_128647


namespace range_of_a_l1286_128636

theorem range_of_a
  (a : ℝ)
  (h : ∀ x : ℝ, |x + 1| + |x - 3| ≥ a) : a ≤ 4 :=
sorry

end range_of_a_l1286_128636


namespace find_four_numbers_l1286_128649

theorem find_four_numbers (a b c d : ℕ) (h1 : b^2 = a * c) (h2 : a * b * c = 216) (h3 : 2 * c = b + d) (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 :=
sorry

end find_four_numbers_l1286_128649


namespace min_dot_product_PA_PB_l1286_128603

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

end min_dot_product_PA_PB_l1286_128603


namespace total_ants_correct_l1286_128638

-- Define the conditions
def park_width_ft : ℕ := 450
def park_length_ft : ℕ := 600
def ants_per_sq_inch_first_half : ℕ := 2
def ants_per_sq_inch_second_half : ℕ := 4

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Convert width and length from feet to inches
def park_width_inch : ℕ := park_width_ft * feet_to_inches
def park_length_inch : ℕ := park_length_ft * feet_to_inches

-- Define the area of each half of the park in square inches
def half_length_inch : ℕ := park_length_inch / 2
def area_first_half_sq_inch : ℕ := park_width_inch * half_length_inch
def area_second_half_sq_inch : ℕ := park_width_inch * half_length_inch

-- Define the number of ants in each half
def ants_first_half : ℕ := ants_per_sq_inch_first_half * area_first_half_sq_inch
def ants_second_half : ℕ := ants_per_sq_inch_second_half * area_second_half_sq_inch

-- Define the total number of ants
def total_ants : ℕ := ants_first_half + ants_second_half

-- The proof problem
theorem total_ants_correct : total_ants = 116640000 := by
  sorry

end total_ants_correct_l1286_128638


namespace union_with_complement_l1286_128609

open Set

-- Definitions based on the conditions given in the problem.
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

-- The statement that needs to be proved
theorem union_with_complement : (M ∪ (U \ N)) = {0, 2, 4, 6, 8} :=
by
  sorry

end union_with_complement_l1286_128609


namespace same_root_implies_a_vals_l1286_128676

-- Define the first function f(x) = x - a
def f (x a : ℝ) : ℝ := x - a

-- Define the second function g(x) = x^2 + ax - 2
def g (x a : ℝ) : ℝ := x^2 + a * x - 2

-- Theorem statement
theorem same_root_implies_a_vals (a : ℝ) (x : ℝ) (hf : f x a = 0) (hg : g x a = 0) : a = 1 ∨ a = -1 := 
sorry

end same_root_implies_a_vals_l1286_128676


namespace solve_phi_l1286_128607

-- Define the problem
noncomputable def f (phi x : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + phi)
noncomputable def f' (phi x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + phi)
noncomputable def g (phi x : ℝ) : ℝ := f phi x + f' phi x

-- Define the main theorem
theorem solve_phi (phi : ℝ) (h : -Real.pi < phi ∧ phi < 0) 
  (even_g : ∀ x, g phi x = g phi (-x)) : phi = -Real.pi / 3 :=
sorry

end solve_phi_l1286_128607


namespace depth_of_water_in_smaller_container_l1286_128686

theorem depth_of_water_in_smaller_container 
  (H_big : ℝ) (R_big : ℝ) (h_water : ℝ) 
  (H_small : ℝ) (R_small : ℝ) (expected_depth : ℝ) 
  (v_water_small : ℝ) 
  (v_water_big : ℝ) 
  (h_total_water : ℝ)
  (above_brim : ℝ) 
  (v_water_final : ℝ) : 

  H_big = 20 ∧ R_big = 6 ∧ h_water = 17 ∧ H_small = 18 ∧ R_small = 5 ∧ expected_depth = 2.88 ∧
  v_water_big = π * R_big^2 * H_big ∧ v_water_small = π * R_small^2 * H_small ∧ 
  h_total_water = π * R_big^2 * h_water ∧ above_brim = π * R_big^2 * (H_big - H_small) ∧ 
  v_water_final = above_brim →

  expected_depth = v_water_final / (π * R_small^2) :=
by
  intro h
  sorry

end depth_of_water_in_smaller_container_l1286_128686


namespace camels_in_caravan_l1286_128637

theorem camels_in_caravan : 
  ∃ (C : ℕ), 
  (60 + 35 + 10 + C) * 1 + 60 * 2 + 35 * 4 + 10 * 2 + 4 * C - (60 + 35 + 10 + C) = 193 ∧ 
  C = 6 :=
by
  sorry

end camels_in_caravan_l1286_128637


namespace Kyle_rose_cost_l1286_128630

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

end Kyle_rose_cost_l1286_128630


namespace exists_prime_mod_greater_remainder_l1286_128675

theorem exists_prime_mod_greater_remainder (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∃ p : ℕ, Prime p ∧ a % p > b % p :=
sorry

end exists_prime_mod_greater_remainder_l1286_128675


namespace median_of_right_triangle_l1286_128634

theorem median_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  c / 2 = 5 :=
by
  rw [h3]
  norm_num

end median_of_right_triangle_l1286_128634


namespace polynomial_divisibility_l1286_128621

theorem polynomial_divisibility (a b : ℤ) :
  (∀ x : ℤ, x^2 - 1 ∣ x^5 - 3 * x^4 + a * x^3 + b * x^2 - 5 * x - 5) ↔ (a = 4 ∧ b = 8) :=
sorry

end polynomial_divisibility_l1286_128621


namespace greatest_possible_remainder_l1286_128620

theorem greatest_possible_remainder (x : ℕ) : ∃ r : ℕ, r < 12 ∧ r ≠ 0 ∧ x % 12 = r ∧ r = 11 :=
by 
  sorry

end greatest_possible_remainder_l1286_128620


namespace remainder_mod_105_l1286_128664

theorem remainder_mod_105 (x : ℤ) 
  (h1 : 3 + x ≡ 4 [ZMOD 27])
  (h2 : 5 + x ≡ 9 [ZMOD 125])
  (h3 : 7 + x ≡ 25 [ZMOD 343]) :
  x % 105 = 4 :=
  sorry

end remainder_mod_105_l1286_128664


namespace album_ways_10_l1286_128681

noncomputable def total_album_ways : ℕ := 
  let photo_albums := 2
  let stamp_albums := 3
  let total_albums := 4
  let friends := 4
  ((total_albums.choose photo_albums) * (total_albums - photo_albums).choose stamp_albums) / friends

theorem album_ways_10 :
  total_album_ways = 10 := 
by sorry

end album_ways_10_l1286_128681


namespace determine_range_a_l1286_128618

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, 1 ≤ x → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

theorem determine_range_a (a : ℝ) (h : ¬ prop_p a ∧ (prop_p a ∨ prop_q a)) : 
  a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8) :=
sorry

end determine_range_a_l1286_128618


namespace total_amount_is_47_69_l1286_128662

noncomputable def Mell_order_cost : ℝ :=
  2 * 4 + 7

noncomputable def friend_order_cost : ℝ :=
  2 * 4 + 7 + 3

noncomputable def total_cost_before_discount : ℝ :=
  Mell_order_cost + 2 * friend_order_cost

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def sales_tax : ℝ :=
  0.10 * total_after_discount

noncomputable def total_to_pay : ℝ :=
  total_after_discount + sales_tax

theorem total_amount_is_47_69 : total_to_pay = 47.69 :=
by
  sorry

end total_amount_is_47_69_l1286_128662


namespace domain_and_range_of_g_l1286_128619

noncomputable def f : ℝ → ℝ := sorry-- Given: a function f with domain [0,2] and range [0,1]
noncomputable def g (x : ℝ) := 1 - f (x / 2 + 1)

theorem domain_and_range_of_g :
  let dom_g := { x | -2 ≤ x ∧ x ≤ 2 }
  let range_g := { y | 0 ≤ y ∧ y ≤ 1 }
  ∀ (x : ℝ), (x ∈ dom_g → (g x) ∈ range_g) := 
sorry

end domain_and_range_of_g_l1286_128619


namespace find_c_value_l1286_128614

def finds_c (c : ℝ) : Prop :=
  6 * (-(c / 6)) + 9 * (-(c / 9)) + c = 0 ∧ (-(c / 6) + -(c / 9) = 30)

theorem find_c_value : ∃ c : ℝ, finds_c c ∧ c = -108 :=
by
  use -108
  sorry

end find_c_value_l1286_128614


namespace bananas_used_l1286_128652

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end bananas_used_l1286_128652


namespace circle_passes_through_points_l1286_128666

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l1286_128666


namespace selling_price_eq_l1286_128624

noncomputable def cost_price : ℝ := 1300
noncomputable def selling_price_loss : ℝ := 1280
noncomputable def selling_price_profit_25_percent : ℝ := 1625

theorem selling_price_eq (cp sp_loss sp_profit sp: ℝ) 
  (h1 : sp_profit = 1.25 * cp)
  (h2 : sp_loss = cp - 20)
  (h3 : sp = cp + 20) :
  sp = 1320 :=
sorry

end selling_price_eq_l1286_128624


namespace regular_octagon_side_length_sum_l1286_128640

theorem regular_octagon_side_length_sum (s : ℝ) (h₁ : s = 2.3) (h₂ : 1 = 100) : 
  8 * (s * 100) = 1840 :=
by
  sorry

end regular_octagon_side_length_sum_l1286_128640


namespace abc_divisibility_l1286_128660

theorem abc_divisibility (a b c : ℕ) (h1 : c ∣ a^b) (h2 : a ∣ b^c) (h3 : b ∣ c^a) : abc ∣ (a + b + c)^(a + b + c) := 
sorry

end abc_divisibility_l1286_128660
