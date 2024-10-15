import Mathlib

namespace NUMINAMATH_GPT_bart_total_pages_l2389_238935

theorem bart_total_pages (total_spent : ℝ) (cost_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_spent = 10) (h2 : cost_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_spent / cost_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end NUMINAMATH_GPT_bart_total_pages_l2389_238935


namespace NUMINAMATH_GPT_find_sister_candy_l2389_238938

/-- Define Katie's initial amount of candy -/
def Katie_candy : ℕ := 10

/-- Define the amount of candy eaten the first night -/
def eaten_candy : ℕ := 9

/-- Define the amount of candy left after the first night -/
def remaining_candy : ℕ := 7

/-- Define the number of candies Katie's sister had -/
def sister_candy (S : ℕ) : Prop :=
  Katie_candy + S - eaten_candy = remaining_candy

/-- Theorem stating that Katie's sister had 6 pieces of candy -/
theorem find_sister_candy : ∃ S, sister_candy S ∧ S = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_sister_candy_l2389_238938


namespace NUMINAMATH_GPT_sixty_five_percent_of_40_minus_four_fifths_of_25_l2389_238942

theorem sixty_five_percent_of_40_minus_four_fifths_of_25 : 
  (0.65 * 40) - (0.8 * 25) = 6 := 
by
  sorry

end NUMINAMATH_GPT_sixty_five_percent_of_40_minus_four_fifths_of_25_l2389_238942


namespace NUMINAMATH_GPT_greatest_divisor_with_sum_of_digits_four_l2389_238939

/-- Define the given numbers -/
def a := 4665
def b := 6905

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Define the greatest number n that divides both a and b, leaving the same remainder and having a sum of digits equal to 4 -/
theorem greatest_divisor_with_sum_of_digits_four :
  ∃ (n : ℕ), (∀ (d : ℕ), (d ∣ a - b ∧ sum_of_digits d = 4) → d ≤ n) ∧ (n ∣ a - b) ∧ (sum_of_digits n = 4) ∧ n = 40 := sorry

end NUMINAMATH_GPT_greatest_divisor_with_sum_of_digits_four_l2389_238939


namespace NUMINAMATH_GPT_smallest_rel_prime_210_l2389_238916

theorem smallest_rel_prime_210 : ∃ x : ℕ, x > 1 ∧ gcd x 210 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 210 = 1 → x ≤ y := 
by
  sorry

end NUMINAMATH_GPT_smallest_rel_prime_210_l2389_238916


namespace NUMINAMATH_GPT_infinitely_many_primes_l2389_238917

theorem infinitely_many_primes : ∀ (p : ℕ) (h_prime : Nat.Prime p), ∃ (q : ℕ), Nat.Prime q ∧ q > p :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_primes_l2389_238917


namespace NUMINAMATH_GPT_sum_of_numbers_l2389_238966

theorem sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) : a + b + c = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l2389_238966


namespace NUMINAMATH_GPT_rectangle_probability_l2389_238936

theorem rectangle_probability (m n : ℕ) (h_m : m = 1003^2) (h_n : n = 1003 * 2005) :
  (1 - (m / n)) = 1002 / 2005 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_probability_l2389_238936


namespace NUMINAMATH_GPT_math_problems_not_a_set_l2389_238990

-- Define the conditions in Lean
def is_well_defined (α : Type) : Prop := sorry

-- Type definitions for the groups of objects
def table_tennis_players : Type := sorry
def positive_integers_less_than_5 : Type := sorry
def irrational_numbers : Type := sorry
def math_problems_2023_college_exam : Type := sorry

-- Defining specific properties of each group
def well_defined_table_tennis_players : is_well_defined table_tennis_players := sorry
def well_defined_positive_integers_less_than_5 : is_well_defined positive_integers_less_than_5 := sorry
def well_defined_irrational_numbers : is_well_defined irrational_numbers := sorry

-- The key property that math problems from 2023 college entrance examination cannot form a set.
theorem math_problems_not_a_set : ¬ is_well_defined math_problems_2023_college_exam := sorry

end NUMINAMATH_GPT_math_problems_not_a_set_l2389_238990


namespace NUMINAMATH_GPT_intersection_with_y_axis_is_correct_l2389_238912

theorem intersection_with_y_axis_is_correct (x y : ℝ) (h : y = 5 * x + 1) (hx : x = 0) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_y_axis_is_correct_l2389_238912


namespace NUMINAMATH_GPT_minimum_value_g_l2389_238914

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if a > 1 then 
    a * (-1/a) + 1 
  else 
    if 0 < a then 
      a^2 + 1 
    else 
      0  -- adding a default value to make it computable

theorem minimum_value_g (a : ℝ) (m : ℝ) : 0 < a ∧ a < 2 ∧ ∃ x₀, f x₀ a = m → m ≥ 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_g_l2389_238914


namespace NUMINAMATH_GPT_steve_halfway_longer_than_danny_l2389_238955

theorem steve_halfway_longer_than_danny :
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  (T_s / 2) - (T_d / 2) = 15.5 :=
by
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  show (T_s / 2) - (T_d / 2) = 15.5
  sorry

end NUMINAMATH_GPT_steve_halfway_longer_than_danny_l2389_238955


namespace NUMINAMATH_GPT_expression_evaluation_l2389_238977

noncomputable def x := Real.sqrt 5 + 1
noncomputable def y := Real.sqrt 5 - 1

theorem expression_evaluation : 
  ( ( (5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2) ) / (1 / (x^2 * y - x * y^2)) ) = 12 := 
by 
  -- Provide a proof here
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2389_238977


namespace NUMINAMATH_GPT_number_of_blocks_needed_to_form_cube_l2389_238972

-- Define the dimensions of the rectangular block
def block_length : ℕ := 5
def block_width : ℕ := 4
def block_height : ℕ := 3

-- Define the side length of the cube
def cube_side_length : ℕ := 60

-- The expected number of rectangular blocks needed
def expected_number_of_blocks : ℕ := 3600

-- Statement to prove the number of rectangular blocks needed to form the cube
theorem number_of_blocks_needed_to_form_cube
  (l : ℕ) (w : ℕ) (h : ℕ) (cube_side : ℕ) (expected_count : ℕ)
  (h_l : l = block_length)
  (h_w : w = block_width)
  (h_h : h = block_height)
  (h_cube_side : cube_side = cube_side_length)
  (h_expected : expected_count = expected_number_of_blocks) :
  (cube_side ^ 3) / (l * w * h) = expected_count :=
sorry

end NUMINAMATH_GPT_number_of_blocks_needed_to_form_cube_l2389_238972


namespace NUMINAMATH_GPT_find_n_l2389_238965

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.sin (n * Real.pi / 180) = Real.sin (782 * Real.pi / 180)) :
  n = 62 ∨ n = -62 := 
sorry

end NUMINAMATH_GPT_find_n_l2389_238965


namespace NUMINAMATH_GPT_combined_mpg_l2389_238924

theorem combined_mpg (ray_mpg tom_mpg ray_miles tom_miles : ℕ) 
  (h1 : ray_mpg = 50) (h2 : tom_mpg = 8) 
  (h3 : ray_miles = 100) (h4 : tom_miles = 200) : 
  (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 100 / 9 :=
by
  sorry

end NUMINAMATH_GPT_combined_mpg_l2389_238924


namespace NUMINAMATH_GPT_raja_monthly_income_l2389_238944

theorem raja_monthly_income (X : ℝ) 
  (h1 : 0.1 * X = 5000) : X = 50000 :=
sorry

end NUMINAMATH_GPT_raja_monthly_income_l2389_238944


namespace NUMINAMATH_GPT_archibald_percentage_wins_l2389_238974

def archibald_wins : ℕ := 12
def brother_wins : ℕ := 18
def total_games_played : ℕ := archibald_wins + brother_wins

def percentage_archibald_wins : ℚ := (archibald_wins : ℚ) / (total_games_played : ℚ) * 100

theorem archibald_percentage_wins : percentage_archibald_wins = 40 := by
  sorry

end NUMINAMATH_GPT_archibald_percentage_wins_l2389_238974


namespace NUMINAMATH_GPT_product_lcm_gcd_l2389_238922

def a : ℕ := 6
def b : ℕ := 8

theorem product_lcm_gcd : Nat.lcm a b * Nat.gcd a b = 48 := by
  sorry

end NUMINAMATH_GPT_product_lcm_gcd_l2389_238922


namespace NUMINAMATH_GPT_range_of_f_l2389_238918

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem range_of_f : Set.Ioo 0 3 ∪ {3} = { y | ∃ x, f x = y } :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l2389_238918


namespace NUMINAMATH_GPT_largest_integer_not_greater_than_expr_l2389_238993

theorem largest_integer_not_greater_than_expr (x : ℝ) (hx : 20 * Real.sin x = 22 * Real.cos x) :
    ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := 
sorry

end NUMINAMATH_GPT_largest_integer_not_greater_than_expr_l2389_238993


namespace NUMINAMATH_GPT_sqrt_of_9_eq_3_l2389_238998

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_of_9_eq_3_l2389_238998


namespace NUMINAMATH_GPT_tangent_line_at_P_l2389_238931

noncomputable def tangent_line (x : ℝ) (y : ℝ) := (8 * x - y - 12 = 0)

def curve (x : ℝ) := x^3 - x^2

def derivative (f : ℝ → ℝ) (x : ℝ) := 3 * x^2 - 2 * x

theorem tangent_line_at_P :
    tangent_line 2 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_P_l2389_238931


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_123_l2389_238950

theorem last_three_digits_of_7_pow_123 : 7^123 % 1000 = 773 := 
by sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_123_l2389_238950


namespace NUMINAMATH_GPT_number_of_blue_marbles_l2389_238920

-- Definitions based on the conditions
def total_marbles : ℕ := 20
def red_marbles : ℕ := 9
def probability_red_or_white : ℚ := 0.7

-- The question to prove: the number of blue marbles (B)
theorem number_of_blue_marbles (B W : ℕ) (h1 : B + W + red_marbles = total_marbles)
  (h2: (red_marbles + W : ℚ) / total_marbles = probability_red_or_white) : 
  B = 6 := 
by
  sorry

end NUMINAMATH_GPT_number_of_blue_marbles_l2389_238920


namespace NUMINAMATH_GPT_sum_of_roots_l2389_238901

theorem sum_of_roots (y1 y2 k m : ℝ) (h1 : y1 ≠ y2) (h2 : 5 * y1^2 - k * y1 = m) (h3 : 5 * y2^2 - k * y2 = m) : 
  y1 + y2 = k / 5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2389_238901


namespace NUMINAMATH_GPT_production_movie_count_l2389_238933

theorem production_movie_count
  (LJ_annual : ℕ)
  (H1 : LJ_annual = 220)
  (H2 : ∀ n, n = 275 → n = LJ_annual + (LJ_annual * 25 / 100))
  (years : ℕ)
  (H3 : years = 5) :
  (LJ_annual + 275) * years = 2475 :=
by {
  sorry
}

end NUMINAMATH_GPT_production_movie_count_l2389_238933


namespace NUMINAMATH_GPT_polygon_sides_l2389_238971

-- Define the conditions
def sum_interior_angles (x : ℕ) : ℝ := 180 * (x - 2)
def sum_given_angles (x : ℕ) : ℝ := 160 + 112 * (x - 1)

-- State the theorem
theorem polygon_sides (x : ℕ) (h : sum_interior_angles x = sum_given_angles x) : x = 6 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2389_238971


namespace NUMINAMATH_GPT_correctly_calculated_expression_l2389_238951

theorem correctly_calculated_expression (x : ℝ) :
  ¬ (x^3 + x^2 = x^5) ∧ 
  ¬ (x^3 * x^2 = x^6) ∧ 
  (x^3 / x^2 = x) ∧ 
  ¬ ((x^3)^2 = x^9) := by
sorry

end NUMINAMATH_GPT_correctly_calculated_expression_l2389_238951


namespace NUMINAMATH_GPT_Mikaela_initially_planned_walls_l2389_238953

/-- 
Mikaela bought 16 containers of paint to cover a certain number of equally-sized walls in her bathroom.
At the last minute, she decided to put tile on one wall and paint flowers on the ceiling with one 
container of paint instead. She had 3 containers of paint left over. 
Prove she initially planned to paint 13 walls.
-/
theorem Mikaela_initially_planned_walls
  (PaintContainers : ℕ)
  (CeilingPaint : ℕ)
  (LeftOverPaint : ℕ)
  (TiledWalls : ℕ) : PaintContainers = 16 → CeilingPaint = 1 → LeftOverPaint = 3 → TiledWalls = 1 → 
    (PaintContainers - CeilingPaint - LeftOverPaint + TiledWalls = 13) :=
by
  -- Given conditions:
  intros h1 h2 h3 h4
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_Mikaela_initially_planned_walls_l2389_238953


namespace NUMINAMATH_GPT_no_real_solutions_l2389_238968

theorem no_real_solutions :
  ∀ y : ℝ, ( (-2 * y + 7)^2 + 2 = -2 * |y| ) → false := by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l2389_238968


namespace NUMINAMATH_GPT_real_solutions_count_l2389_238978

theorem real_solutions_count :
  ∃ n : ℕ, n = 2 ∧ ∀ x : ℝ, |x + 1| = |x - 3| + |x - 4| → x = 2 ∨ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_count_l2389_238978


namespace NUMINAMATH_GPT_find_two_digit_number_l2389_238973

theorem find_two_digit_number
  (X : ℕ)
  (h1 : 57 + (10 * X + 6) = 123)
  (h2 : two_digit_number = 10 * X + 9) :
  two_digit_number = 69 :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l2389_238973


namespace NUMINAMATH_GPT_value_of_k_l2389_238958

theorem value_of_k :
  ∀ (x k : ℝ), (x + 6) * (x - 5) = x^2 + k * x - 30 → k = 1 :=
by
  intros x k h
  sorry

end NUMINAMATH_GPT_value_of_k_l2389_238958


namespace NUMINAMATH_GPT_find_number_l2389_238947

theorem find_number (x : ℝ) (h : 160 = 3.2 * x) : x = 50 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l2389_238947


namespace NUMINAMATH_GPT_star_calculation_l2389_238997

-- Define the operation '*' via the given table
def star_table : Matrix (Fin 5) (Fin 5) (Fin 5) :=
  ![
    ![0, 1, 2, 3, 4],
    ![1, 0, 4, 2, 3],
    ![2, 3, 1, 4, 0],
    ![3, 4, 0, 1, 2],
    ![4, 2, 3, 0, 1]
  ]

def star (a b : Fin 5) : Fin 5 := star_table a b

-- Prove (3 * 5) * (2 * 4) = 3
theorem star_calculation : star (star 2 4) (star 4 1) = 2 := by
  sorry

end NUMINAMATH_GPT_star_calculation_l2389_238997


namespace NUMINAMATH_GPT_challenge_Jane_l2389_238904

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def card_pairs : List (Char ⊕ ℕ) :=
  [Sum.inl 'A', Sum.inl 'T', Sum.inl 'U', Sum.inr 5, Sum.inr 8, Sum.inr 10, Sum.inr 14]

def Jane_claim (c : Char ⊕ ℕ) : Prop :=
  match c with
  | Sum.inl v => is_vowel v → ∃ n, Sum.inr n ∈ card_pairs ∧ is_even n
  | Sum.inr n => false

theorem challenge_Jane (cards : List (Char ⊕ ℕ)) (h : card_pairs = cards) :
  ∃ c ∈ cards, c = Sum.inr 5 ∧ ¬Jane_claim (Sum.inr 5) :=
sorry

end NUMINAMATH_GPT_challenge_Jane_l2389_238904


namespace NUMINAMATH_GPT_add_fractions_11_12_7_15_l2389_238949

/-- A theorem stating that the sum of 11/12 and 7/15 is 83/60. -/
theorem add_fractions_11_12_7_15 : (11 / 12) + (7 / 15) = (83 / 60) := 
by
  sorry

end NUMINAMATH_GPT_add_fractions_11_12_7_15_l2389_238949


namespace NUMINAMATH_GPT_donny_total_cost_eq_45_l2389_238980

-- Definitions for prices of each type of apple
def price_small : ℝ := 1.5
def price_medium : ℝ := 2
def price_big : ℝ := 3

-- Quantities purchased by Donny
def count_small : ℕ := 6
def count_medium : ℕ := 6
def count_big : ℕ := 8

-- Total cost calculation
def total_cost (count_small count_medium count_big : ℕ) : ℝ := 
  (count_small * price_small) + (count_medium * price_medium) + (count_big * price_big)

-- Theorem stating the total cost
theorem donny_total_cost_eq_45 : total_cost count_small count_medium count_big = 45 := by
  sorry

end NUMINAMATH_GPT_donny_total_cost_eq_45_l2389_238980


namespace NUMINAMATH_GPT_binom_18_7_l2389_238932

theorem binom_18_7 : Nat.choose 18 7 = 31824 := by sorry

end NUMINAMATH_GPT_binom_18_7_l2389_238932


namespace NUMINAMATH_GPT_gcd_of_72_120_168_l2389_238913

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_72_120_168_l2389_238913


namespace NUMINAMATH_GPT_ellen_smoothie_ingredients_l2389_238979

theorem ellen_smoothie_ingredients :
  let strawberries := 0.2
  let yogurt := 0.1
  let orange_juice := 0.2
  strawberries + yogurt + orange_juice = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_ellen_smoothie_ingredients_l2389_238979


namespace NUMINAMATH_GPT_orchard_yield_correct_l2389_238925

-- Definitions for conditions
def gala3YrTreesYield : ℕ := 10 * 120
def gala2YrTreesYield : ℕ := 10 * 150
def galaTotalYield : ℕ := gala3YrTreesYield + gala2YrTreesYield

def fuji4YrTreesYield : ℕ := 5 * 180
def fuji5YrTreesYield : ℕ := 5 * 200
def fujiTotalYield : ℕ := fuji4YrTreesYield + fuji5YrTreesYield

def redhaven6YrTreesYield : ℕ := 15 * 50
def redhaven4YrTreesYield : ℕ := 15 * 60
def redhavenTotalYield : ℕ := redhaven6YrTreesYield + redhaven4YrTreesYield

def elberta2YrTreesYield : ℕ := 5 * 70
def elberta3YrTreesYield : ℕ := 5 * 75
def elberta5YrTreesYield : ℕ := 5 * 80
def elbertaTotalYield : ℕ := elberta2YrTreesYield + elberta3YrTreesYield + elberta5YrTreesYield

def appleTotalYield : ℕ := galaTotalYield + fujiTotalYield
def peachTotalYield : ℕ := redhavenTotalYield + elbertaTotalYield
def orchardTotalYield : ℕ := appleTotalYield + peachTotalYield

-- Theorem to prove
theorem orchard_yield_correct : orchardTotalYield = 7375 := 
by sorry

end NUMINAMATH_GPT_orchard_yield_correct_l2389_238925


namespace NUMINAMATH_GPT_smallest_possible_number_of_apples_l2389_238986

theorem smallest_possible_number_of_apples :
  ∃ (M : ℕ), M > 2 ∧ M % 9 = 2 ∧ M % 10 = 2 ∧ M % 11 = 2 ∧ M = 200 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_number_of_apples_l2389_238986


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2389_238954

theorem solution_set_of_inequality (x : ℝ) : (x - 1) * (2 - x) > 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2389_238954


namespace NUMINAMATH_GPT_sum_of_x_y_l2389_238915

theorem sum_of_x_y (m x y : ℝ) (h₁ : x + m = 4) (h₂ : y - 3 = m) : x + y = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_x_y_l2389_238915


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l2389_238903

theorem cos_seven_pi_over_six :
  Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l2389_238903


namespace NUMINAMATH_GPT_find_natural_number_l2389_238989

-- Define the problem statement
def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ (2 * n^2 - 2) = k * (n^3 - n)

-- The main theorem
theorem find_natural_number (n : ℕ) : satisfies_condition n ↔ n = 2 :=
sorry

end NUMINAMATH_GPT_find_natural_number_l2389_238989


namespace NUMINAMATH_GPT_average_student_headcount_l2389_238975

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end NUMINAMATH_GPT_average_student_headcount_l2389_238975


namespace NUMINAMATH_GPT_distance_is_18_l2389_238930

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  let faster := (x + 1) * (3 * t / 4) = d
  let slower := (x - 1) * (t + 3) = d
  let normal := x * t = d
  faster ∧ slower ∧ normal

theorem distance_is_18 : 
  ∃ (x t : ℝ), distance_walked x t 18 :=
by
  sorry

end NUMINAMATH_GPT_distance_is_18_l2389_238930


namespace NUMINAMATH_GPT_transformed_parabola_eq_l2389_238995

noncomputable def initial_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3
def shift_left (h : ℝ) (c : ℝ): ℝ := h - c
def shift_down (k : ℝ) (d : ℝ): ℝ := k - d

theorem transformed_parabola_eq :
  ∃ (x : ℝ), (initial_parabola (shift_left x 2) - 1 = 2 * (x + 1)^2 + 2) :=
sorry

end NUMINAMATH_GPT_transformed_parabola_eq_l2389_238995


namespace NUMINAMATH_GPT_range_of_a_l2389_238981

noncomputable def in_range (a : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∨ (a ≥ 1)

theorem range_of_a (a : ℝ) (p q : Prop) (h1 : p ↔ (0 < a ∧ a < 1)) (h2 : q ↔ (a ≥ 1 / 2)) (h3 : p ∨ q) (h4 : ¬ (p ∧ q)) :
  in_range a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2389_238981


namespace NUMINAMATH_GPT_systematic_sampling_interval_l2389_238994

-- Definitions for the given conditions
def total_students : ℕ := 1203
def sample_size : ℕ := 40

-- Theorem statement to be proven
theorem systematic_sampling_interval (N n : ℕ) (hN : N = total_students) (hn : n = sample_size) : 
  N % n ≠ 0 → ∃ k : ℕ, k = 30 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_interval_l2389_238994


namespace NUMINAMATH_GPT_sally_balance_fraction_l2389_238996

variable (G : ℝ) (x : ℝ)
-- spending limit on gold card is G
-- spending limit on platinum card is 2G
-- Balance on platinum card is G/2
-- After transfer, 0.5833333333333334 portion of platinum card remains unspent

theorem sally_balance_fraction
  (h1 : (5/12) * 2 * G = G / 2 + x * G) : x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sally_balance_fraction_l2389_238996


namespace NUMINAMATH_GPT_production_today_l2389_238983

def average_production (P : ℕ) (n : ℕ) := P / n

theorem production_today :
  ∀ (T P n : ℕ), n = 9 → average_production P n = 50 → average_production (P + T) (n + 1) = 54 → T = 90 :=
by
  intros T P n h1 h2 h3
  sorry

end NUMINAMATH_GPT_production_today_l2389_238983


namespace NUMINAMATH_GPT_simplify_expression_l2389_238941

variable (a : ℝ)

theorem simplify_expression : 2 * a * (2 * a ^ 2 + a) - a ^ 2 = 4 * a ^ 3 + a ^ 2 := 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2389_238941


namespace NUMINAMATH_GPT_total_votes_l2389_238952

theorem total_votes (V : ℝ) (h1 : 0.32 * V = 0.32 * V) (h2 : 0.32 * V + 1908 = 0.68 * V) : V = 5300 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l2389_238952


namespace NUMINAMATH_GPT_count_distinct_reals_a_with_integer_roots_l2389_238959

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end NUMINAMATH_GPT_count_distinct_reals_a_with_integer_roots_l2389_238959


namespace NUMINAMATH_GPT_finite_odd_divisors_condition_l2389_238921

theorem finite_odd_divisors_condition (k : ℕ) (hk : 0 < k) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → ¬ (n % 2 = 1 ∧ n ∣ k^n + 1)) ↔ (∃ c : ℕ, k + 1 = 2^c) :=
by sorry

end NUMINAMATH_GPT_finite_odd_divisors_condition_l2389_238921


namespace NUMINAMATH_GPT_seq_identity_l2389_238988

-- Define the sequence (a_n)
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 0 ∧ a 2 = 1 ∧ ∀ n, a (n + 3) = a (n + 1) + 1998 * a n

theorem seq_identity (a : ℕ → ℕ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * (a (n - 1))^2 :=
sorry

end NUMINAMATH_GPT_seq_identity_l2389_238988


namespace NUMINAMATH_GPT_find_number_l2389_238963

theorem find_number :
  ∃ x : ℕ, (x / 5 = 80 + x / 6) ∧ x = 2400 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l2389_238963


namespace NUMINAMATH_GPT_domain_of_function_y_eq_sqrt_2x_3_div_x_2_l2389_238940

def domain (x : ℝ) : Prop :=
  (2 * x - 3 ≥ 0) ∧ (x ≠ 2)

theorem domain_of_function_y_eq_sqrt_2x_3_div_x_2 :
  ∀ x : ℝ, domain x ↔ ((x ≥ 3 / 2) ∧ (x ≠ 2)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_y_eq_sqrt_2x_3_div_x_2_l2389_238940


namespace NUMINAMATH_GPT_negation_of_proposition_l2389_238961

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^2 + x_0 - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l2389_238961


namespace NUMINAMATH_GPT_lcm_gcf_ratio_120_504_l2389_238962

theorem lcm_gcf_ratio_120_504 : 
  let a := 120
  let b := 504
  (Int.lcm a b) / (Int.gcd a b) = 105 := by
  sorry

end NUMINAMATH_GPT_lcm_gcf_ratio_120_504_l2389_238962


namespace NUMINAMATH_GPT_miguel_socks_probability_l2389_238991

theorem miguel_socks_probability :
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  probability = 5 / 21 :=
by
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  sorry

end NUMINAMATH_GPT_miguel_socks_probability_l2389_238991


namespace NUMINAMATH_GPT_find_n_l2389_238926

theorem find_n (n : ℤ) (h : (1 : ℤ)^2 + 3 * 1 + n = 0) : n = -4 :=
sorry

end NUMINAMATH_GPT_find_n_l2389_238926


namespace NUMINAMATH_GPT_min_packs_needed_l2389_238928

-- Define pack sizes
def pack_sizes : List ℕ := [6, 12, 24, 30]

-- Define the total number of cans needed
def total_cans : ℕ := 150

-- Define the minimum number of packs needed to buy exactly 150 cans of soda
theorem min_packs_needed : ∃ packs : List ℕ, (∀ p ∈ packs, p ∈ pack_sizes) ∧ List.sum packs = total_cans ∧ packs.length = 5 := by
  sorry

end NUMINAMATH_GPT_min_packs_needed_l2389_238928


namespace NUMINAMATH_GPT_train_speed_l2389_238911

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 300) (h_time : time = 15) : 
  (length / time) * 3.6 = 72 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l2389_238911


namespace NUMINAMATH_GPT_house_painting_l2389_238985

theorem house_painting (n : ℕ) (h1 : n = 1000)
  (occupants : Fin n → Fin n) (perm : ∀ i, occupants i ≠ i) :
  ∃ (coloring : Fin n → Fin 3), ∀ i, coloring i ≠ coloring (occupants i) :=
by
  sorry

end NUMINAMATH_GPT_house_painting_l2389_238985


namespace NUMINAMATH_GPT_books_about_outer_space_l2389_238919

variable (x : ℕ)

theorem books_about_outer_space :
  160 + 48 + 16 * x = 224 → x = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_books_about_outer_space_l2389_238919


namespace NUMINAMATH_GPT_vector_calc_l2389_238976

def vec1 : ℝ × ℝ := (5, -8)
def vec2 : ℝ × ℝ := (2, 6)
def vec3 : ℝ × ℝ := (-1, 4)
def scalar : ℝ := 5

theorem vector_calc :
  (vec1.1 - scalar * vec2.1 + vec3.1, vec1.2 - scalar * vec2.2 + vec3.2) = (-6, -34) :=
sorry

end NUMINAMATH_GPT_vector_calc_l2389_238976


namespace NUMINAMATH_GPT_paths_for_content_l2389_238967

def grid := [
  [none, none, none, none, none, none, some 'C', none, none, none, none, none, none, none],
  [none, none, none, none, none, some 'C', some 'O', some 'C', none, none, none, none, none, none],
  [none, none, none, none, some 'C', some 'O', some 'N', some 'O', some 'C', none, none, none, none, none],
  [none, none, none, some 'C', some 'O', some 'N', some 'T', some 'N', some 'O', some 'C', none, none, none, none],
  [none, none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'T', some 'N', some 'O', some 'C', none, none, none],
  [none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C', none, none],
  [some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'T', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C']
]

def spelling_paths : Nat :=
  -- Skipping the actual calculation and providing the given total for now
  127

theorem paths_for_content : spelling_paths = 127 := sorry

end NUMINAMATH_GPT_paths_for_content_l2389_238967


namespace NUMINAMATH_GPT_percent_sparrows_not_pigeons_l2389_238910

-- Definitions of percentages
def crows_percent : ℝ := 0.20
def sparrows_percent : ℝ := 0.40
def pigeons_percent : ℝ := 0.15
def doves_percent : ℝ := 0.25

-- The statement to prove
theorem percent_sparrows_not_pigeons :
  (sparrows_percent / (1 - pigeons_percent)) = 0.47 :=
by
  sorry

end NUMINAMATH_GPT_percent_sparrows_not_pigeons_l2389_238910


namespace NUMINAMATH_GPT_gasVolume_at_20_l2389_238909

variable (V : ℕ → ℕ)

/-- Given conditions:
 1. The gas volume expands by 3 cubic centimeters for every 5 degree rise in temperature.
 2. The volume is 30 cubic centimeters when the temperature is 30 degrees.
  -/
def gasVolume : Prop :=
  (∀ T ΔT, ΔT = 5 → V (T + ΔT) = V T + 3) ∧ V 30 = 30

theorem gasVolume_at_20 :
  gasVolume V → V 20 = 24 :=
by
  intro h
  -- Proof steps would go here.
  sorry

end NUMINAMATH_GPT_gasVolume_at_20_l2389_238909


namespace NUMINAMATH_GPT_lines_are_parallel_l2389_238992

theorem lines_are_parallel : 
  ∀ (x y : ℝ), (2 * x - y = 7) → (2 * x - y - 1 = 0) → False :=
by
  sorry  -- Proof will be filled in later

end NUMINAMATH_GPT_lines_are_parallel_l2389_238992


namespace NUMINAMATH_GPT_points_scored_by_others_l2389_238984

-- Define the conditions as hypothesis
variables (P_total P_Jessie : ℕ)
  (H1 : P_total = 311)
  (H2 : P_Jessie = 41)
  (H3 : ∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie)

-- Define what we need to prove
theorem points_scored_by_others (P_others : ℕ) :
  P_total = 311 → P_Jessie = 41 → 
  (∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie) → 
  P_others = 188 :=
by
  sorry

end NUMINAMATH_GPT_points_scored_by_others_l2389_238984


namespace NUMINAMATH_GPT_order_of_exponents_l2389_238945

theorem order_of_exponents (p q r : ℕ) (hp : p = 2^3009) (hq : q = 3^2006) (hr : r = 5^1003) : r < p ∧ p < q :=
by {
  sorry -- Proof will go here
}

end NUMINAMATH_GPT_order_of_exponents_l2389_238945


namespace NUMINAMATH_GPT_women_in_business_class_l2389_238957

theorem women_in_business_class 
  (total_passengers : ℕ) 
  (percent_women : ℝ) 
  (percent_women_in_business : ℝ) 
  (H1 : total_passengers = 300)
  (H2 : percent_women = 0.70)
  (H3 : percent_women_in_business = 0.08) : 
  ∃ (num_women_business_class : ℕ), num_women_business_class = 16 := 
by
  sorry

end NUMINAMATH_GPT_women_in_business_class_l2389_238957


namespace NUMINAMATH_GPT_rational_inequality_solution_l2389_238900

theorem rational_inequality_solution (x : ℝ) (h : x ≠ 4) :
  (4 < x ∧ x ≤ 5) ↔ (x - 2) / (x - 4) ≤ 3 :=
sorry

end NUMINAMATH_GPT_rational_inequality_solution_l2389_238900


namespace NUMINAMATH_GPT_radius_of_circle_l2389_238946

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 64 * π) : r = 8 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_l2389_238946


namespace NUMINAMATH_GPT_liquid_X_percentage_36_l2389_238902

noncomputable def liquid_X_percentage (m : ℕ) (pX : ℕ) (m_evaporate : ℕ) (m_add : ℕ) (p_add : ℕ) : ℕ :=
  let m_X_initial := (pX * m / 100)
  let m_water_initial := ((100 - pX) * m / 100)
  let m_X_after_evaporation := m_X_initial
  let m_water_after_evaporation := m_water_initial - m_evaporate
  let m_X_additional := (p_add * m_add / 100)
  let m_water_additional := ((100 - p_add) * m_add / 100)
  let m_X_new := m_X_after_evaporation + m_X_additional
  let m_water_new := m_water_after_evaporation + m_water_additional
  let m_total_new := m_X_new + m_water_new
  (m_X_new * 100 / m_total_new)

theorem liquid_X_percentage_36 :
  liquid_X_percentage 10 30 2 2 30 = 36 := by
  sorry

end NUMINAMATH_GPT_liquid_X_percentage_36_l2389_238902


namespace NUMINAMATH_GPT_find_intercept_l2389_238948

theorem find_intercept (avg_height : ℝ) (avg_shoe_size : ℝ) (a : ℝ)
  (h1 : avg_height = 170)
  (h2 : avg_shoe_size = 40) 
  (h3 : 3 * avg_shoe_size + a = avg_height) : a = 50 := 
by
  sorry

end NUMINAMATH_GPT_find_intercept_l2389_238948


namespace NUMINAMATH_GPT_find_y_intercept_l2389_238964

theorem find_y_intercept (m : ℝ) (x_intercept: ℝ × ℝ) : (x_intercept.snd = 0) → (x_intercept = (-4, 0)) → m = 3 → (0, m * 4 - m * (-4)) = (0, 12) :=
by
  sorry

end NUMINAMATH_GPT_find_y_intercept_l2389_238964


namespace NUMINAMATH_GPT_part1_part2_l2389_238969

noncomputable def f (a x : ℝ) : ℝ := (a / 2) * x * x - (a - 2) * x - 2 * x * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := a * x - a - 2 * Real.log x

theorem part1 (a : ℝ) : (∀ x > 0, f' a x ≥ 0) ↔ a = 2 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : f' a x1 = 0) (h4 : f' a x2 = 0) (h5 : x1 < x2) : 
  x2 - x1 > 4 / a - 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2389_238969


namespace NUMINAMATH_GPT_sum_cos_4x_4y_4z_l2389_238970

theorem sum_cos_4x_4y_4z (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_cos_4x_4y_4z_l2389_238970


namespace NUMINAMATH_GPT_find_A_l2389_238906

noncomputable def A_value (A B C : ℝ) := (A = 1/4) 

theorem find_A : 
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x - 1) + B / (x - 3) + C / (x - 3)^2)) →
  A_value A B C :=
by 
  sorry

end NUMINAMATH_GPT_find_A_l2389_238906


namespace NUMINAMATH_GPT_total_tomatoes_l2389_238908

def tomatoes_first_plant : Nat := 2 * 12
def tomatoes_second_plant : Nat := (tomatoes_first_plant / 2) + 5
def tomatoes_third_plant : Nat := tomatoes_second_plant + 2

theorem total_tomatoes :
  (tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant) = 60 := by
  sorry

end NUMINAMATH_GPT_total_tomatoes_l2389_238908


namespace NUMINAMATH_GPT_sacks_harvested_per_section_l2389_238907

theorem sacks_harvested_per_section (total_sacks : ℕ) (sections : ℕ) (sacks_per_section : ℕ) 
  (h1 : total_sacks = 360) 
  (h2 : sections = 8) 
  (h3 : total_sacks = sections * sacks_per_section) :
  sacks_per_section = 45 :=
by sorry

end NUMINAMATH_GPT_sacks_harvested_per_section_l2389_238907


namespace NUMINAMATH_GPT_largest_integer_satisfying_inequality_l2389_238927

theorem largest_integer_satisfying_inequality : ∃ (x : ℤ), (5 * x - 4 < 3 - 2 * x) ∧ (∀ (y : ℤ), (5 * y - 4 < 3 - 2 * y) → y ≤ x) ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_satisfying_inequality_l2389_238927


namespace NUMINAMATH_GPT_gavin_shirts_l2389_238923

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end NUMINAMATH_GPT_gavin_shirts_l2389_238923


namespace NUMINAMATH_GPT_fencing_required_l2389_238956

theorem fencing_required (L W : ℕ) (area : ℕ) (hL : L = 20) (hA : area = 120) (hW : area = L * W) :
  2 * W + L = 32 :=
by
  -- Steps and proof logic to be provided here
  sorry

end NUMINAMATH_GPT_fencing_required_l2389_238956


namespace NUMINAMATH_GPT_gondor_laptops_wednesday_l2389_238934

/-- Gondor's phone repair earnings per unit -/
def phone_earning : ℕ := 10

/-- Gondor's laptop repair earnings per unit -/
def laptop_earning : ℕ := 20

/-- Number of phones repaired on Monday -/
def phones_monday : ℕ := 3

/-- Number of phones repaired on Tuesday -/
def phones_tuesday : ℕ := 5

/-- Number of laptops repaired on Thursday -/
def laptops_thursday : ℕ := 4

/-- Total earnings of Gondor -/
def total_earnings : ℕ := 200

/-- Number of laptops repaired on Wednesday, which we need to prove equals 2 -/
def laptops_wednesday : ℕ := 2

theorem gondor_laptops_wednesday : 
    (phones_monday * phone_earning + phones_tuesday * phone_earning + 
    laptops_thursday * laptop_earning + laptops_wednesday * laptop_earning = total_earnings) :=
by
    sorry

end NUMINAMATH_GPT_gondor_laptops_wednesday_l2389_238934


namespace NUMINAMATH_GPT_evening_customers_l2389_238929

-- Define the conditions
def matinee_price : ℕ := 5
def evening_price : ℕ := 7
def opening_night_price : ℕ := 10
def popcorn_price : ℕ := 10
def num_matinee_customers : ℕ := 32
def num_opening_night_customers : ℕ := 58
def total_revenue : ℕ := 1670

-- Define the number of evening customers as a variable
variable (E : ℕ)

-- Prove that the number of evening customers E equals 40 given the conditions
theorem evening_customers :
  5 * num_matinee_customers +
  7 * E +
  10 * num_opening_night_customers +
  10 * (num_matinee_customers + E + num_opening_night_customers) / 2 = total_revenue
  → E = 40 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_evening_customers_l2389_238929


namespace NUMINAMATH_GPT_max_matching_pairs_l2389_238960

theorem max_matching_pairs (total_pairs : ℕ) (lost_individual : ℕ) (left_pair : ℕ) : 
  total_pairs = 25 ∧ lost_individual = 9 → left_pair = 20 :=
by
  sorry

end NUMINAMATH_GPT_max_matching_pairs_l2389_238960


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2389_238999

theorem solve_equation_1 :
  ∀ x : ℝ, 3 * x - 5 = 6 * x - 8 → x = 1 :=
by
  intro x
  intro h
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, (x + 1) / 2 - (2 * x - 1) / 3 = 1 → x = -1 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l2389_238999


namespace NUMINAMATH_GPT_jori_water_left_l2389_238937

theorem jori_water_left (initial_gallons used_gallons : ℚ) (h1 : initial_gallons = 3) (h2 : used_gallons = 11 / 4) :
  initial_gallons - used_gallons = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_jori_water_left_l2389_238937


namespace NUMINAMATH_GPT_simpleInterest_500_l2389_238905

def simpleInterest (P R T : ℝ) : ℝ := P * R * T

theorem simpleInterest_500 :
  simpleInterest 10000 0.05 1 = 500 :=
by
  sorry

end NUMINAMATH_GPT_simpleInterest_500_l2389_238905


namespace NUMINAMATH_GPT_Elizabeth_More_Revenue_Than_Banks_l2389_238982

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end NUMINAMATH_GPT_Elizabeth_More_Revenue_Than_Banks_l2389_238982


namespace NUMINAMATH_GPT_coefficient_6th_term_expansion_l2389_238943

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, k => if k > n then 0 else Nat.choose n k

-- Define the coefficient of the general term of binomial expansion
def binomial_coeff (n r : ℕ) : ℤ := (-1)^r * binom n r

-- Define the theorem to show the coefficient of the 6th term in the expansion of (x-1)^10
theorem coefficient_6th_term_expansion :
  binomial_coeff 10 5 = -binom 10 5 :=
by sorry

end NUMINAMATH_GPT_coefficient_6th_term_expansion_l2389_238943


namespace NUMINAMATH_GPT_smallest_cube_ends_in_584_l2389_238987

theorem smallest_cube_ends_in_584 (n : ℕ) : n^3 ≡ 584 [MOD 1000] → n = 34 := by
  sorry

end NUMINAMATH_GPT_smallest_cube_ends_in_584_l2389_238987
