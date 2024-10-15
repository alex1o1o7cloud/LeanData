import Mathlib

namespace NUMINAMATH_GPT_arc_length_of_sector_l419_41966

theorem arc_length_of_sector 
  (R : ℝ) (θ : ℝ) (hR : R = Real.pi) (hθ : θ = 2 * Real.pi / 3) : 
  (R * θ = 2 * Real.pi^2 / 3) := 
by
  rw [hR, hθ]
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l419_41966


namespace NUMINAMATH_GPT_find_a3_l419_41971

noncomputable def geometric_term (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n-1)

noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (q^n - 1) / (q - 1)

theorem find_a3 (a : ℝ) (q : ℝ) (h_q : q = 3)
  (h_sum : geometric_sum a q 3 + geometric_sum a q 4 = 53 / 3) :
  geometric_term a q 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l419_41971


namespace NUMINAMATH_GPT_arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l419_41914

-- Arithmetic Progression
theorem arithmetic_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 - x2 = x2 - x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = (2 * a^3 + 27 * c) / (9 * a)) :=
sorry

-- Geometric Progression
theorem geometric_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x2 / x1 = x3 / x2 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = a * c^(1/3)) :=
sorry

-- Harmonic Sequence
theorem harmonic_sequence_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x1 - x2) / (x2 - x3) = x1 / x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (a = (2 * b^3 + 27 * c) / (9 * b^2)) :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l419_41914


namespace NUMINAMATH_GPT_net_increase_in_wealth_l419_41996

-- Definitions for yearly changes and fees
def firstYearChange (initialAmt : ℝ) : ℝ := initialAmt * 1.75 - 0.02 * initialAmt * 1.75
def secondYearChange (amt : ℝ) : ℝ := amt * 0.7 - 0.02 * amt * 0.7
def thirdYearChange (amt : ℝ) : ℝ := amt * 1.45 - 0.02 * amt * 1.45
def fourthYearChange (amt : ℝ) : ℝ := amt * 0.85 - 0.02 * amt * 0.85

-- Total Value after 4th year accounting all changes and fees
def totalAfterFourYears (initialAmt : ℝ) : ℝ :=
  let afterFirstYear := firstYearChange initialAmt
  let afterSecondYear := secondYearChange afterFirstYear
  let afterThirdYear := thirdYearChange afterSecondYear
  fourthYearChange afterThirdYear

-- Capital gains tax calculation
def capitalGainsTax (initialAmt finalAmt : ℝ) : ℝ :=
  0.20 * (finalAmt - initialAmt)

-- Net value after taxes
def netValueAfterTaxes (initialAmt : ℝ) : ℝ :=
  let total := totalAfterFourYears initialAmt
  total - capitalGainsTax initialAmt total

-- Main theorem statement
theorem net_increase_in_wealth :
  ∀ (initialAmt : ℝ), netValueAfterTaxes initialAmt = initialAmt * 1.31408238206 := sorry

end NUMINAMATH_GPT_net_increase_in_wealth_l419_41996


namespace NUMINAMATH_GPT_simplify_expr_l419_41947

theorem simplify_expr : 2 - 2 / (1 + Real.sqrt 2) - 2 / (1 - Real.sqrt 2) = -2 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l419_41947


namespace NUMINAMATH_GPT_total_spent_is_49_l419_41919

-- Define the prices of items
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define Paula's purchases
def paula_bracelets := 3
def paula_keychains := 2
def paula_coloring_book := 1
def paula_stickers := 4

-- Define Olive's purchases
def olive_bracelets := 2
def olive_coloring_book := 1
def olive_toy_car := 1
def olive_stickers := 3

-- Calculate total expenses
def paula_total := paula_bracelets * price_bracelet + paula_keychains * price_keychain + paula_coloring_book * price_coloring_book + paula_stickers * price_sticker
def olive_total := olive_coloring_book * price_coloring_book + olive_bracelets * price_bracelet + olive_toy_car * price_toy_car + olive_stickers * price_sticker
def total_expense := paula_total + olive_total

-- Prove the total expenses amount to $49
theorem total_spent_is_49 : total_expense = 49 :=
by
  have : paula_total = (3 * 4) + (2 * 5) + (1 * 3) + (4 * 1) := rfl
  have : olive_total = (1 * 3) + (2 * 4) + (1 *6) + (3 * 1) := rfl
  have : paula_total = 29 := rfl
  have : olive_total = 20 := rfl
  have : total_expense = 29 + 20 := rfl
  exact rfl

end NUMINAMATH_GPT_total_spent_is_49_l419_41919


namespace NUMINAMATH_GPT_units_digit_product_l419_41969

theorem units_digit_product (a b : ℕ) (h1 : (a % 10 ≠ 0) ∧ (b % 10 ≠ 0)) : (a * b % 10 = 0) ∨ (a * b % 10 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_product_l419_41969


namespace NUMINAMATH_GPT_find_m_l419_41987

theorem find_m (m : ℤ) (h1 : m + 1 ≠ 0) (h2 : m^2 + 3 * m + 1 = -1) : m = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l419_41987


namespace NUMINAMATH_GPT_francis_had_2_muffins_l419_41922

noncomputable def cost_of_francis_breakfast (m : ℕ) : ℕ := 2 * m + 6
noncomputable def cost_of_kiera_breakfast : ℕ := 4 + 3
noncomputable def total_cost (m : ℕ) : ℕ := cost_of_francis_breakfast m + cost_of_kiera_breakfast

theorem francis_had_2_muffins (m : ℕ) : total_cost m = 17 → m = 2 :=
by
  -- Sorry is used here to leave the proof steps blank.
  sorry

end NUMINAMATH_GPT_francis_had_2_muffins_l419_41922


namespace NUMINAMATH_GPT_min_value_expression_l419_41980

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  ∃ c, c = (1/(a+1) + 4/(b+1)) ∧ c ≥ 9/4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l419_41980


namespace NUMINAMATH_GPT_probability_not_same_level_is_four_fifths_l419_41907

-- Definitions of the conditions
def nobility_levels := 5
def total_outcomes := nobility_levels * nobility_levels
def same_level_outcomes := nobility_levels

-- Definition of the probability
def probability_not_same_level := 1 - (same_level_outcomes / total_outcomes : ℚ)

-- The theorem statement
theorem probability_not_same_level_is_four_fifths :
  probability_not_same_level = 4 / 5 := 
  by sorry

end NUMINAMATH_GPT_probability_not_same_level_is_four_fifths_l419_41907


namespace NUMINAMATH_GPT_find_prime_pairs_l419_41908

open Nat

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def valid_prime_pairs (p q : ℕ): Prop :=
  is_prime p ∧ is_prime q ∧ divides p (30 * q - 1) ∧ divides q (30 * p - 1)

theorem find_prime_pairs :
  { (p, q) | valid_prime_pairs p q } = { (7, 11), (11, 7), (59, 61), (61, 59) } :=
sorry

end NUMINAMATH_GPT_find_prime_pairs_l419_41908


namespace NUMINAMATH_GPT_solve_system_l419_41960

-- The system of equations as conditions in Lean
def system1 (x y : ℤ) : Prop := 5 * x + 2 * y = 25
def system2 (x y : ℤ) : Prop := 3 * x + 4 * y = 15

-- The statement that asserts the solution is (x = 5, y = 0)
theorem solve_system : ∃ x y : ℤ, system1 x y ∧ system2 x y ∧ x = 5 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l419_41960


namespace NUMINAMATH_GPT_angle_coterminal_l419_41916

theorem angle_coterminal (k : ℤ) : 
  ∃ α : ℝ, α = 30 + k * 360 :=
sorry

end NUMINAMATH_GPT_angle_coterminal_l419_41916


namespace NUMINAMATH_GPT_average_visitors_in_month_of_30_days_starting_with_sunday_l419_41903

def average_visitors_per_day (sundays_visitors : ℕ) (other_days_visitors : ℕ) (num_sundays : ℕ) (num_other_days : ℕ) : ℕ :=
  (sundays_visitors * num_sundays + other_days_visitors * num_other_days) / (num_sundays + num_other_days)

theorem average_visitors_in_month_of_30_days_starting_with_sunday :
  average_visitors_per_day 1000 700 5 25 = 750 := sorry

end NUMINAMATH_GPT_average_visitors_in_month_of_30_days_starting_with_sunday_l419_41903


namespace NUMINAMATH_GPT_triangle_vertices_l419_41975

theorem triangle_vertices : 
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ x - y = -4 ∧ x = 2 / 3 ∧ y = 14 / 3) ∧ 
  (∃ (x y : ℚ), x - y = -4 ∧ y = -1 ∧ x = -5) ∧
  (∃ (x y : ℚ), 2 * x + y = 6 ∧ y = -1 ∧ x = 7 / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_vertices_l419_41975


namespace NUMINAMATH_GPT_max_prime_p_l419_41993

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

end NUMINAMATH_GPT_max_prime_p_l419_41993


namespace NUMINAMATH_GPT_sum_of_numbers_l419_41999

theorem sum_of_numbers (a b c d : ℕ) (h1 : a > d) (h2 : a * b = c * d) (h3 : a + b + c + d = a * c) (h4 : ∀ x y z w: ℕ, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ) : a + b + c + d = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l419_41999


namespace NUMINAMATH_GPT_eating_time_proof_l419_41938

noncomputable def combined_eating_time (time_fat time_thin weight : ℝ) : ℝ :=
  let rate_fat := 1 / time_fat
  let rate_thin := 1 / time_thin
  let combined_rate := rate_fat + rate_thin
  weight / combined_rate

theorem eating_time_proof :
  let time_fat := 12
  let time_thin := 40
  let weight := 5
  combined_eating_time time_fat time_thin weight = (600 / 13) :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_eating_time_proof_l419_41938


namespace NUMINAMATH_GPT_volume_of_cube_is_correct_surface_area_of_cube_is_correct_l419_41917

-- Define the conditions: total edge length of the cube frame
def total_edge_length : ℕ := 60
def number_of_edges : ℕ := 12

-- Define the edge length of the cube
def edge_length (total_edge_length number_of_edges : ℕ) : ℕ := total_edge_length / number_of_edges

-- Define the volume of the cube
def cube_volume (a : ℕ) : ℕ := a ^ 3

-- Define the surface area of the cube
def cube_surface_area (a : ℕ) : ℕ := 6 * (a ^ 2)

-- Volume Proof Statement
theorem volume_of_cube_is_correct : cube_volume (edge_length total_edge_length number_of_edges) = 125 :=
by
  sorry

-- Surface Area Proof Statement
theorem surface_area_of_cube_is_correct : cube_surface_area (edge_length total_edge_length number_of_edges) = 150 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cube_is_correct_surface_area_of_cube_is_correct_l419_41917


namespace NUMINAMATH_GPT_select_4_people_arrangement_3_day_new_year_l419_41939

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_select_4_people_arrangement_3_day_new_year_l419_41939


namespace NUMINAMATH_GPT_cost_price_of_article_l419_41982

-- Define the conditions
variable (C : ℝ) -- Cost price of the article
variable (SP : ℝ) -- Selling price of the article

-- Conditions according to the problem
def condition1 : Prop := SP = 0.75 * C
def condition2 : Prop := SP + 500 = 1.15 * C

-- The theorem to prove the cost price
theorem cost_price_of_article (h₁ : condition1 C SP) (h₂ : condition2 C SP) : C = 1250 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l419_41982


namespace NUMINAMATH_GPT_shopkeepers_total_profit_percentage_l419_41942

noncomputable def calculateProfitPercentage : ℝ :=
  let oranges := 1000
  let bananas := 800
  let apples := 750
  let rotten_oranges_percentage := 0.12
  let rotten_bananas_percentage := 0.05
  let rotten_apples_percentage := 0.10
  let profit_oranges_percentage := 0.20
  let profit_bananas_percentage := 0.25
  let profit_apples_percentage := 0.15
  let cost_per_orange := 2.5
  let cost_per_banana := 1.5
  let cost_per_apple := 2.0

  let rotten_oranges := rotten_oranges_percentage * oranges
  let rotten_bananas := rotten_bananas_percentage * bananas
  let rotten_apples := rotten_apples_percentage * apples

  let good_oranges := oranges - rotten_oranges
  let good_bananas := bananas - rotten_bananas
  let good_apples := apples - rotten_apples

  let cost_oranges := cost_per_orange * oranges
  let cost_bananas := cost_per_banana * bananas
  let cost_apples := cost_per_apple * apples

  let total_cost := cost_oranges + cost_bananas + cost_apples

  let selling_price_oranges := cost_per_orange * (1 + profit_oranges_percentage) * good_oranges
  let selling_price_bananas := cost_per_banana * (1 + profit_bananas_percentage) * good_bananas
  let selling_price_apples := cost_per_apple * (1 + profit_apples_percentage) * good_apples

  let total_selling_price := selling_price_oranges + selling_price_bananas + selling_price_apples

  let total_profit := total_selling_price - total_cost

  (total_profit / total_cost) * 100

theorem shopkeepers_total_profit_percentage :
  calculateProfitPercentage = 8.03 := sorry

end NUMINAMATH_GPT_shopkeepers_total_profit_percentage_l419_41942


namespace NUMINAMATH_GPT_sin_cos_theta_l419_41930

open Real

theorem sin_cos_theta (θ : ℝ) (H1 : θ > π / 2 ∧ θ < π) (H2 : tan (θ + π / 4) = 1 / 2) :
  sin θ + cos θ = -sqrt 10 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_theta_l419_41930


namespace NUMINAMATH_GPT_motorcycles_in_anytown_l419_41924

variable (t s m : ℕ) -- t: number of trucks, s: number of sedans, m: number of motorcycles
variable (r_trucks r_sedans r_motorcycles : ℕ) -- r_trucks : truck ratio, r_sedans : sedan ratio, r_motorcycles : motorcycle ratio
variable (n_sedans : ℕ) -- n_sedans: number of sedans

theorem motorcycles_in_anytown
  (h1 : r_trucks = 3) -- ratio of trucks
  (h2 : r_sedans = 7) -- ratio of sedans
  (h3 : r_motorcycles = 2) -- ratio of motorcycles
  (h4 : s = 9100) -- number of sedans
  (h5 : s = (r_sedans * n_sedans)) -- relationship between sedans and parts
  (h6 : t = (r_trucks * n_sedans)) -- relationship between trucks and parts
  (h7 : m = (r_motorcycles * n_sedans)) -- relationship between motorcycles and parts
  : m = 2600 := by
    sorry

end NUMINAMATH_GPT_motorcycles_in_anytown_l419_41924


namespace NUMINAMATH_GPT_number_of_ways_to_fill_l419_41948

-- Definitions and conditions
def triangular_array (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the triangular array structure
  sorry 

def sum_based (row : ℕ) (col : ℕ) : Prop :=
  -- Placeholder definition for the sum-based condition
  sorry 

def valid_filling (x : Fin 13 → ℕ) :=
  (∀ i, x i = 0 ∨ x i = 1) ∧
  (x 0 + x 12) % 5 = 0

theorem number_of_ways_to_fill (x : Fin 13 → ℕ) :
  triangular_array 13 1 → sum_based 13 1 →
  valid_filling x → 
  (∃ (count : ℕ), count = 4096) :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_fill_l419_41948


namespace NUMINAMATH_GPT_find_value_l419_41952

variable {a b : ℝ}

theorem find_value (h : 2 * a + b + 1 = 0) : 1 + 4 * a + 2 * b = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l419_41952


namespace NUMINAMATH_GPT_find_blue_highlighters_l419_41976

theorem find_blue_highlighters
(h_pink : P = 9)
(h_yellow : Y = 8)
(h_total : T = 22)
(h_sum : P + Y + B = T) :
  B = 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_blue_highlighters_l419_41976


namespace NUMINAMATH_GPT_find_a_and_union_l419_41936

noncomputable def A (a : ℝ) : Set ℝ := { -4, 2 * a - 1, a ^ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { a - 5, 1 - a, 9 }

theorem find_a_and_union {a : ℝ}
  (h : A a ∩ B a = {9}): 
  a = -3 ∧ A a ∪ B a = {-8, -7, -4, 4, 9} :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_union_l419_41936


namespace NUMINAMATH_GPT_sales_tax_difference_l419_41978

theorem sales_tax_difference :
  let price_before_tax := 40
  let tax_rate_8_percent := 0.08
  let tax_rate_7_percent := 0.07
  let sales_tax_8_percent := price_before_tax * tax_rate_8_percent
  let sales_tax_7_percent := price_before_tax * tax_rate_7_percent
  sales_tax_8_percent - sales_tax_7_percent = 0.4 := 
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l419_41978


namespace NUMINAMATH_GPT_ball_drawing_ways_l419_41992

theorem ball_drawing_ways :
    ∃ (r w y : ℕ), 
      0 ≤ r ∧ r ≤ 2 ∧
      0 ≤ w ∧ w ≤ 3 ∧
      0 ≤ y ∧ y ≤ 5 ∧
      r + w + y = 5 ∧
      10 ≤ 5 * r + 2 * w + y ∧ 
      5 * r + 2 * w + y ≤ 15 := 
sorry

end NUMINAMATH_GPT_ball_drawing_ways_l419_41992


namespace NUMINAMATH_GPT_range_of_x_l419_41931

-- Defining the vectors as given in the conditions
def a (x : ℝ) : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (2, -1)

-- Defining the condition that the angle is obtuse
def is_obtuse (x : ℝ) : Prop := 
  let dot_product := (a x).1 * b.1 + (a x).2 * b.2
  dot_product < 0

-- Defining the condition that vectors are not in opposite directions
def not_opposite_directions (x : ℝ) : Prop := x ≠ -6

-- Proving the required range of x
theorem range_of_x (x : ℝ) :
  is_obtuse x → not_opposite_directions x → x < 3 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_l419_41931


namespace NUMINAMATH_GPT_cos_330_eq_sqrt_3_div_2_l419_41946

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_cos_330_eq_sqrt_3_div_2_l419_41946


namespace NUMINAMATH_GPT_prob_two_girls_is_one_fourth_l419_41937

-- Define the probability of giving birth to a girl
def prob_girl : ℚ := 1 / 2

-- Define the probability of having two girls
def prob_two_girls : ℚ := prob_girl * prob_girl

-- Theorem statement: The probability of having two girls is 1/4
theorem prob_two_girls_is_one_fourth : prob_two_girls = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_prob_two_girls_is_one_fourth_l419_41937


namespace NUMINAMATH_GPT_polynomial_remainder_l419_41967

theorem polynomial_remainder :
  ∀ (x : ℝ), (x^4 + 2 * x^3 - 3 * x^2 + 4 * x - 5) % (x^2 - 3 * x + 2) = (24 * x - 25) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l419_41967


namespace NUMINAMATH_GPT_price_verification_l419_41956

noncomputable def price_on_hot_day : ℚ :=
  let P : ℚ := 225 / 172
  1.25 * P

theorem price_verification :
  (32 * 7 * (225 / 172) + 32 * 3 * (1.25 * (225 / 172)) - (32 * 10 * 0.75)) = 210 :=
sorry

end NUMINAMATH_GPT_price_verification_l419_41956


namespace NUMINAMATH_GPT_a_2013_is_4_l419_41944

theorem a_2013_is_4
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 7)
  (h3 : ∀ n : ℕ, a (n+2) = (a n * a (n+1)) % 10) :
  a 2013 = 4 :=
sorry

end NUMINAMATH_GPT_a_2013_is_4_l419_41944


namespace NUMINAMATH_GPT_min_value_abs_plus_2023_proof_l419_41965

noncomputable def min_value_abs_plus_2023 (a : ℚ) : Prop :=
  |a| + 2023 ≥ 2023

theorem min_value_abs_plus_2023_proof (a : ℚ) : min_value_abs_plus_2023 a :=
  by
  sorry

end NUMINAMATH_GPT_min_value_abs_plus_2023_proof_l419_41965


namespace NUMINAMATH_GPT_different_meal_combinations_l419_41933

-- Defining the conditions explicitly
def items_on_menu : ℕ := 12

-- A function representing possible combinations of choices for Yann and Camille
def meal_combinations (menu_items : ℕ) : ℕ :=
  menu_items * (menu_items - 1)

-- Theorem stating that given 12 items on the menu, the different combinations of meals is 132
theorem different_meal_combinations : meal_combinations items_on_menu = 132 :=
by
  sorry

end NUMINAMATH_GPT_different_meal_combinations_l419_41933


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l419_41904

open Classical

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x : ℕ, x^2 ≤ x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l419_41904


namespace NUMINAMATH_GPT_smallest_num_rectangles_to_cover_square_l419_41994

-- Define essential conditions
def area_3by4_rectangle : ℕ := 3 * 4
def area_square (side_length : ℕ) : ℕ := side_length * side_length
def can_be_tiled_with_3by4 (side_length : ℕ) : Prop := (area_square side_length) % area_3by4_rectangle = 0

-- Define the main theorem
theorem smallest_num_rectangles_to_cover_square :
  can_be_tiled_with_3by4 12 → ∃ n : ℕ, n = (area_square 12) / area_3by4_rectangle ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_num_rectangles_to_cover_square_l419_41994


namespace NUMINAMATH_GPT_factorize_x4_plus_81_l419_41990

theorem factorize_x4_plus_81 : 
  ∀ x : ℝ, 
    (x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9)) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorize_x4_plus_81_l419_41990


namespace NUMINAMATH_GPT_max_n_for_regular_polygons_l419_41923

theorem max_n_for_regular_polygons (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 3)
  (h3 : (7 * (m - 2) * n) = (8 * (n - 2) * m)) : 
  n ≤ 112 ∧ (∃ m, (14 * n = (n - 16) * m)) :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_regular_polygons_l419_41923


namespace NUMINAMATH_GPT_sam_earnings_difference_l419_41958

def hours_per_dollar := 1 / 10  -- Sam earns $10 per hour, so it takes 1/10 hour per dollar earned.

theorem sam_earnings_difference
  (hours_per_dollar : ℝ := 1 / 10)
  (E1 : ℝ := 200)  -- Earnings in the first month are $200.
  (total_hours : ℝ := 55)  -- Total hours he worked over two months.
  (total_hourly_earning : ℝ := total_hours / hours_per_dollar)  -- Total earnings over two months.
  (E2 : ℝ := total_hourly_earning - E1) :  -- Earnings in the second month.

  E2 - E1 = 150 :=  -- The difference in earnings between the second month and the first month is $150.
sorry

end NUMINAMATH_GPT_sam_earnings_difference_l419_41958


namespace NUMINAMATH_GPT_min_value_expression_l419_41935

theorem min_value_expression : ∃ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 = 2 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l419_41935


namespace NUMINAMATH_GPT_arithmetic_seq_a8_l419_41912

theorem arithmetic_seq_a8 : ∀ (a : ℕ → ℤ), 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → 
  (a 5 + a 6 = 22) → 
  (a 3 = 7) → 
  a 8 = 15 :=
by
  intros a ha_arithmetic hsum h3
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a8_l419_41912


namespace NUMINAMATH_GPT_value_of_fraction_l419_41984

variable {x y : ℝ}

theorem value_of_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 * x + y) / (x - 3 * y) = -2) :
  (x + 3 * y) / (3 * x - y) = 2 :=
sorry

end NUMINAMATH_GPT_value_of_fraction_l419_41984


namespace NUMINAMATH_GPT_original_volume_of_ice_cube_l419_41955

theorem original_volume_of_ice_cube
  (V : ℝ)
  (h1 : V * (1/2) * (2/3) * (3/4) * (4/5) = 30)
  : V = 150 :=
sorry

end NUMINAMATH_GPT_original_volume_of_ice_cube_l419_41955


namespace NUMINAMATH_GPT_cars_selected_l419_41953

theorem cars_selected (num_cars num_clients selections_made total_selections : ℕ)
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_made = 2)
  (h4 : total_selections = num_clients * selections_made) :
  num_cars * (total_selections / num_cars) = 48 :=
by
  sorry

end NUMINAMATH_GPT_cars_selected_l419_41953


namespace NUMINAMATH_GPT_find_r_condition_l419_41954

variable {x y z w r : ℝ}

axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : z ≠ 0
axiom h4 : w ≠ 0
axiom h5 : (x ≠ y) ∧ (x ≠ z) ∧ (x ≠ w) ∧ (y ≠ z) ∧ (y ≠ w) ∧ (z ≠ w)

noncomputable def is_geometric_progression (a b c d : ℝ) (r : ℝ) : Prop :=
  b = a * r ∧ c = a * r^2 ∧ d = a * r^3

theorem find_r_condition :
  is_geometric_progression (x * (y - z)) (y * (z - x)) (z * (x - y)) (w * (y - x)) r →
  r^3 + r^2 + r + 1 = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_r_condition_l419_41954


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l419_41974

theorem greatest_three_digit_multiple_of_17 : ∃ x : ℕ, 100 ≤ x ∧ x ≤ 999 ∧ 17 ∣ x ∧ ∀ y : ℕ, 100 ≤ y ∧ y ≤ 999 ∧ 17 ∣ y → y ≤ x :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l419_41974


namespace NUMINAMATH_GPT_emilia_donut_holes_count_l419_41977

noncomputable def surface_area (r : ℕ) : ℕ := 4 * r^2

def lcm (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

def donut_holes := 5103

theorem emilia_donut_holes_count :
  ∀ (S1 S2 S3 : ℕ), 
  S1 = surface_area 5 → 
  S2 = surface_area 7 → 
  S3 = surface_area 9 → 
  donut_holes = lcm S1 S2 S3 / S1 :=
by
  intros S1 S2 S3 hS1 hS2 hS3
  sorry

end NUMINAMATH_GPT_emilia_donut_holes_count_l419_41977


namespace NUMINAMATH_GPT_calculate_cherry_pies_l419_41988

-- Definitions for the conditions
def total_pies : ℕ := 40
def ratio_parts_apple : ℕ := 2
def ratio_parts_blueberry : ℕ := 5
def ratio_parts_cherry : ℕ := 3
def total_ratio_parts := ratio_parts_apple + ratio_parts_blueberry + ratio_parts_cherry

-- Calculating the number of pies per part and then the number of cherry pies
def pies_per_part : ℕ := total_pies / total_ratio_parts
def cherry_pies : ℕ := ratio_parts_cherry * pies_per_part

-- Proof statement
theorem calculate_cherry_pies : cherry_pies = 12 :=
by
  -- Lean proof goes here
  sorry

end NUMINAMATH_GPT_calculate_cherry_pies_l419_41988


namespace NUMINAMATH_GPT_rational_solution_counts_l419_41957

theorem rational_solution_counts :
  (∃ (x y : ℚ), x^2 + y^2 = 2) ∧ 
  (¬ ∃ (x y : ℚ), x^2 + y^2 = 3) := 
by 
  sorry

end NUMINAMATH_GPT_rational_solution_counts_l419_41957


namespace NUMINAMATH_GPT_range_of_a_intersection_l419_41902

theorem range_of_a_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x - 2 * k + 2 ∧ y = a * x^2 - 2 * a * x - 3 * a) ↔ (a ≤ -2/3 ∨ a > 0) := by
  sorry

end NUMINAMATH_GPT_range_of_a_intersection_l419_41902


namespace NUMINAMATH_GPT_problem_statement_l419_41945

theorem problem_statement (x y : ℝ) (h : -x + 2 * y = 5) :
  5 * (x - 2 * y) ^ 2 - 3 * (x - 2 * y) - 60 = 80 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l419_41945


namespace NUMINAMATH_GPT_hypotenuse_of_454590_triangle_l419_41943

theorem hypotenuse_of_454590_triangle (l : ℝ) (angle : ℝ) (h : ℝ) (h_leg : l = 15) (h_angle : angle = 45) :
  h = l * Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_hypotenuse_of_454590_triangle_l419_41943


namespace NUMINAMATH_GPT_solve_equation_l419_41973

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end NUMINAMATH_GPT_solve_equation_l419_41973


namespace NUMINAMATH_GPT_find_FC_l419_41910

variable (DC : ℝ) (CB : ℝ) (AB AD ED : ℝ)
variable (FC : ℝ)
variable (h1 : DC = 9)
variable (h2 : CB = 6)
variable (h3 : AB = (1/3) * AD)
variable (h4 : ED = (2/3) * AD)

theorem find_FC : FC = 9 :=
by sorry

end NUMINAMATH_GPT_find_FC_l419_41910


namespace NUMINAMATH_GPT_alayas_fruit_salads_l419_41913

theorem alayas_fruit_salads (A : ℕ) (H1 : 2 * A + A = 600) : A = 200 := 
by
  sorry

end NUMINAMATH_GPT_alayas_fruit_salads_l419_41913


namespace NUMINAMATH_GPT_parabola_focus_directrix_distance_l419_41932

theorem parabola_focus_directrix_distance {a : ℝ} (h₀ : a > 0):
  (∃ (b : ℝ), ∃ (x1 x2 : ℝ), (x1 + x2 = 1 / a) ∧ (1 / (2 * a) = 1)) → 
  (1 / (2 * a) / 2 = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_distance_l419_41932


namespace NUMINAMATH_GPT_sin_double_angle_identity_l419_41962

open Real 

theorem sin_double_angle_identity 
  (A : ℝ) 
  (h1 : 0 < A) 
  (h2 : A < π / 2) 
  (h3 : cos A = 3 / 5) : 
  sin (2 * A) = 24 / 25 :=
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_identity_l419_41962


namespace NUMINAMATH_GPT_willam_tax_paid_l419_41983

-- Define our conditions
variables (T : ℝ) (tax_collected : ℝ) (willam_percent : ℝ)

-- Initialize the conditions according to the problem statement
def is_tax_collected (tax_collected : ℝ) : Prop := tax_collected = 3840
def is_farm_tax_levied_on_cultivated_land : Prop := true -- Essentially means we acknowledge it is 50%
def is_willam_taxable_land_percentage (willam_percent : ℝ) : Prop := willam_percent = 0.25

-- The final theorem that states Mr. Willam's tax payment is $960 given the conditions
theorem willam_tax_paid  : 
  ∀ (T : ℝ),
  is_tax_collected 3840 → 
  is_farm_tax_levied_on_cultivated_land →
  is_willam_taxable_land_percentage 0.25 →
  0.25 * 3840 = 960 :=
sorry

end NUMINAMATH_GPT_willam_tax_paid_l419_41983


namespace NUMINAMATH_GPT_derivative_at_2_l419_41981

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_at_2 : (deriv f 2) = 1 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_derivative_at_2_l419_41981


namespace NUMINAMATH_GPT_total_number_of_workers_l419_41940

theorem total_number_of_workers 
  (W : ℕ) 
  (avg_all : ℕ) 
  (n_technicians : ℕ) 
  (avg_technicians : ℕ) 
  (avg_non_technicians : ℕ) :
  avg_all * W = avg_technicians * n_technicians + avg_non_technicians * (W - n_technicians) →
  avg_all = 8000 →
  n_technicians = 7 →
  avg_technicians = 12000 →
  avg_non_technicians = 6000 →
  W = 21 :=
by 
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_number_of_workers_l419_41940


namespace NUMINAMATH_GPT_greatest_integer_solution_l419_41928

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 36 ≤ 0) : n ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_solution_l419_41928


namespace NUMINAMATH_GPT_expected_number_of_digits_on_fair_icosahedral_die_l419_41905

noncomputable def expected_digits_fair_icosahedral_die : ℚ :=
  let prob_one_digit := (9 : ℚ) / 20
  let prob_two_digits := (11 : ℚ) / 20
  (prob_one_digit * 1) + (prob_two_digits * 2)

theorem expected_number_of_digits_on_fair_icosahedral_die : expected_digits_fair_icosahedral_die = 1.55 := by
  sorry

end NUMINAMATH_GPT_expected_number_of_digits_on_fair_icosahedral_die_l419_41905


namespace NUMINAMATH_GPT_age_difference_l419_41920

variable (A B C : ℕ)

-- Conditions: C is 11 years younger than A
axiom h1 : C = A - 11

-- Statement: Prove the difference (A + B) - (B + C) is 11
theorem age_difference : (A + B) - (B + C) = 11 := by
  sorry

end NUMINAMATH_GPT_age_difference_l419_41920


namespace NUMINAMATH_GPT_solve_inequalities_l419_41972

theorem solve_inequalities (x : ℝ) (h₁ : x - 2 > 1) (h₂ : x < 4) : 3 < x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l419_41972


namespace NUMINAMATH_GPT_total_triangles_correct_l419_41901

-- Define the rectangle and additional constructions
structure Rectangle :=
  (A B C D : Type)
  (midpoint_AB midpoint_BC midpoint_CD midpoint_DA : Type)
  (AC BD diagonals : Type)

-- Hypothesize the structure
variables (rect : Rectangle)

-- Define the number of triangles
def number_of_triangles (r : Rectangle) : Nat := 16

-- The theorem statement
theorem total_triangles_correct : number_of_triangles rect = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_triangles_correct_l419_41901


namespace NUMINAMATH_GPT_systematic_sampling_first_group_draw_l419_41900

noncomputable def index_drawn_from_group (x n : ℕ) : ℕ := x + 8 * (n - 1)

theorem systematic_sampling_first_group_draw (k : ℕ) (fifteenth_group : index_drawn_from_group k 15 = 116) :
  index_drawn_from_group k 1 = 4 := 
sorry

end NUMINAMATH_GPT_systematic_sampling_first_group_draw_l419_41900


namespace NUMINAMATH_GPT_angle_bisectors_and_median_inequality_l419_41911

open Real

variables (A B C : Point)
variables (a b c : ℝ) -- sides of the triangle
variables (p : ℝ) -- semi-perimeter of the triangle
variables (la lb mc : ℝ) -- angle bisectors and median lengths

-- Assume the given conditions
axiom angle_bisector_la (A B C : Point) : ℝ -- lengths of the angle bisector of ∠BAC
axiom angle_bisector_lb (A B C : Point) : ℝ -- lengths of the angle bisector of ∠ABC
axiom median_mc (A B C : Point) : ℝ -- length of the median from vertex C
axiom semi_perimeter (a b c : ℝ) : ℝ -- semi-perimeter of the triangle

-- The statement of the theorem
theorem angle_bisectors_and_median_inequality (la lb mc p : ℝ) :
  la + lb + mc ≤ sqrt 3 * p :=
sorry

end NUMINAMATH_GPT_angle_bisectors_and_median_inequality_l419_41911


namespace NUMINAMATH_GPT_find_cost_price_l419_41989

/-- Define the given conditions -/
def selling_price : ℝ := 100
def profit_percentage : ℝ := 0.15
def cost_price : ℝ := 86.96

/-- Define the relationship between selling price and cost price -/
def relation (CP SP : ℝ) : Prop := SP = CP * (1 + profit_percentage)

/-- State the theorem based on the conditions and required proof -/
theorem find_cost_price 
  (SP : ℝ) (CP : ℝ) 
  (h1 : SP = selling_price) 
  (h2 : relation CP SP) : 
  CP = cost_price := 
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l419_41989


namespace NUMINAMATH_GPT_bus_stop_time_per_hour_l419_41986

theorem bus_stop_time_per_hour
  (speed_no_stops : ℝ)
  (speed_with_stops : ℝ)
  (h1 : speed_no_stops = 50)
  (h2 : speed_with_stops = 35) : 
  18 = (60 * (1 - speed_with_stops / speed_no_stops)) :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_per_hour_l419_41986


namespace NUMINAMATH_GPT_scientific_notation_of_population_l419_41934

theorem scientific_notation_of_population : (85000000 : ℝ) = 8.5 * 10^7 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_population_l419_41934


namespace NUMINAMATH_GPT_compare_xyz_l419_41915

theorem compare_xyz
  (a b c d : ℝ) (h : a < b ∧ b < c ∧ c < d)
  (x : ℝ) (hx : x = (a + b) * (c + d))
  (y : ℝ) (hy : y = (a + c) * (b + d))
  (z : ℝ) (hz : z = (a + d) * (b + c)) :
  x < y ∧ y < z :=
by sorry

end NUMINAMATH_GPT_compare_xyz_l419_41915


namespace NUMINAMATH_GPT_possible_winning_scores_count_l419_41968

def total_runners := 15
def total_score := (total_runners * (total_runners + 1)) / 2

def min_score := 15
def max_potential_score := 39

def is_valid_winning_score (score : ℕ) : Prop :=
  min_score ≤ score ∧ score ≤ max_potential_score

theorem possible_winning_scores_count : 
  ∃ scores : Finset ℕ, ∀ score ∈ scores, is_valid_winning_score score ∧ Finset.card scores = 25 := 
sorry

end NUMINAMATH_GPT_possible_winning_scores_count_l419_41968


namespace NUMINAMATH_GPT_min_value_48_l419_41951

noncomputable def min_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : ℝ :=
  1 / a + 27 / b

theorem min_value_48 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : 
  min_value ha hb h = 48 := 
sorry

end NUMINAMATH_GPT_min_value_48_l419_41951


namespace NUMINAMATH_GPT_average_marks_of_all_students_l419_41998

/-
Consider two classes:
- The first class has 12 students with an average mark of 40.
- The second class has 28 students with an average mark of 60.

We are to prove that the average marks of all students from both classes combined is 54.
-/

theorem average_marks_of_all_students (s1 s2 : ℕ) (m1 m2 : ℤ)
  (h1 : s1 = 12) (h2 : m1 = 40) (h3 : s2 = 28) (h4 : m2 = 60) :
  (s1 * m1 + s2 * m2) / (s1 + s2) = 54 :=
by
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_average_marks_of_all_students_l419_41998


namespace NUMINAMATH_GPT_asha_remaining_money_l419_41918

-- Define the borrowed amounts, gift, and savings
def borrowed_from_brother : ℤ := 20
def borrowed_from_father : ℤ := 40
def borrowed_from_mother : ℤ := 30
def gift_from_granny : ℤ := 70
def savings : ℤ := 100

-- Total amount of money Asha has
def total_amount : ℤ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + savings

-- Amount spent by Asha
def amount_spent : ℤ := (3 * total_amount) / 4

-- Amount of money Asha remains with
def amount_left : ℤ := total_amount - amount_spent

-- The proof statement
theorem asha_remaining_money : amount_left = 65 := by
  sorry

end NUMINAMATH_GPT_asha_remaining_money_l419_41918


namespace NUMINAMATH_GPT_remaining_macaroons_weight_is_103_l419_41949

-- Definitions based on the conditions
def coconutMacaroonsInitialCount := 12
def coconutMacaroonWeight := 5
def coconutMacaroonsBags := 4

def almondMacaroonsInitialCount := 8
def almondMacaroonWeight := 8
def almondMacaroonsBags := 2

def whiteChocolateMacaroonsInitialCount := 2
def whiteChocolateMacaroonWeight := 10

def steveAteCoconutMacaroons := coconutMacaroonsInitialCount / coconutMacaroonsBags
def steveAteAlmondMacaroons := (almondMacaroonsInitialCount / almondMacaroonsBags) / 2
def steveAteWhiteChocolateMacaroons := 1

-- Calculation of remaining macaroons weights
def remainingCoconutMacaroonsCount := coconutMacaroonsInitialCount - steveAteCoconutMacaroons
def remainingAlmondMacaroonsCount := almondMacaroonsInitialCount - steveAteAlmondMacaroons
def remainingWhiteChocolateMacaroonsCount := whiteChocolateMacaroonsInitialCount - steveAteWhiteChocolateMacaroons

-- Calculation of total remaining weight
def remainingCoconutMacaroonsWeight := remainingCoconutMacaroonsCount * coconutMacaroonWeight
def remainingAlmondMacaroonsWeight := remainingAlmondMacaroonsCount * almondMacaroonWeight
def remainingWhiteChocolateMacaroonsWeight := remainingWhiteChocolateMacaroonsCount * whiteChocolateMacaroonWeight

def totalRemainingWeight := remainingCoconutMacaroonsWeight + remainingAlmondMacaroonsWeight + remainingWhiteChocolateMacaroonsWeight

-- Statement to be proved
theorem remaining_macaroons_weight_is_103 :
  totalRemainingWeight = 103 := by
  sorry

end NUMINAMATH_GPT_remaining_macaroons_weight_is_103_l419_41949


namespace NUMINAMATH_GPT_chemistry_problem_l419_41997

theorem chemistry_problem 
(C : ℝ)  -- concentration of the original salt solution
(h_mix : 1 * C / 100 = 15 * 2 / 100) : 
  C = 30 := 
sorry

end NUMINAMATH_GPT_chemistry_problem_l419_41997


namespace NUMINAMATH_GPT_find_minutes_per_mile_l419_41927

-- Conditions
def num_of_movies : ℕ := 2
def avg_length_of_movie_hours : ℝ := 1.5
def total_distance_miles : ℝ := 15

-- Question and proof target
theorem find_minutes_per_mile :
  (num_of_movies * avg_length_of_movie_hours * 60) / total_distance_miles = 12 :=
by
  -- Insert the proof here (not required as per the task instructions)
  sorry

end NUMINAMATH_GPT_find_minutes_per_mile_l419_41927


namespace NUMINAMATH_GPT_reciprocal_sum_neg_l419_41921

theorem reciprocal_sum_neg (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c = 8) : (1/a) + (1/b) + (1/c) < 0 := 
sorry

end NUMINAMATH_GPT_reciprocal_sum_neg_l419_41921


namespace NUMINAMATH_GPT_permutations_of_BANANA_l419_41941

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end NUMINAMATH_GPT_permutations_of_BANANA_l419_41941


namespace NUMINAMATH_GPT_unique_arrangements_of_BANANA_l419_41991

-- Define the conditions as separate definitions in Lean 4
def word := "BANANA"
def total_letters := 6
def count_A := 3
def count_N := 2
def count_B := 1

-- State the theorem to be proven
theorem unique_arrangements_of_BANANA : 
  (total_letters.factorial) / (count_A.factorial * count_N.factorial * count_B.factorial) = 60 := 
by
  sorry

end NUMINAMATH_GPT_unique_arrangements_of_BANANA_l419_41991


namespace NUMINAMATH_GPT_find_x_between_0_and_180_l419_41961

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end NUMINAMATH_GPT_find_x_between_0_and_180_l419_41961


namespace NUMINAMATH_GPT_smallest_n_logarithm_l419_41926

theorem smallest_n_logarithm :
  ∃ n : ℕ, 0 < n ∧ 
  (Real.log (Real.log n / Real.log 3) / Real.log 3^2 =
  Real.log (Real.log n / Real.log 2) / Real.log 2^3) ∧ 
  n = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_logarithm_l419_41926


namespace NUMINAMATH_GPT_ammonium_bromide_total_weight_l419_41959

noncomputable def nitrogen_weight : ℝ := 14.01
noncomputable def hydrogen_weight : ℝ := 1.01
noncomputable def bromine_weight : ℝ := 79.90
noncomputable def ammonium_bromide_weight : ℝ := nitrogen_weight + 4 * hydrogen_weight + bromine_weight
noncomputable def moles : ℝ := 5
noncomputable def total_weight : ℝ := moles * ammonium_bromide_weight

theorem ammonium_bromide_total_weight :
  total_weight = 489.75 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_ammonium_bromide_total_weight_l419_41959


namespace NUMINAMATH_GPT_smallest_angle_of_triangle_l419_41979

theorem smallest_angle_of_triangle (k : ℕ) (h : 4 * k + 5 * k + 9 * k = 180) : 4 * k = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_angle_of_triangle_l419_41979


namespace NUMINAMATH_GPT_range_of_a_l419_41929

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x - 1) / x > 2 * a) ↔ a ∈ (Set.Ici (-1/2) : Set ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l419_41929


namespace NUMINAMATH_GPT_twelve_hens_lay_48_eggs_in_twelve_days_l419_41925

theorem twelve_hens_lay_48_eggs_in_twelve_days :
  (∀ (hens eggs days : ℕ), hens = 3 → eggs = 3 → days = 3 → eggs / (hens * days) = 1/3) → 
  ∀ (hens days : ℕ), hens = 12 → days = 12 → hens * days * (1/3) = 48 :=
by
  sorry

end NUMINAMATH_GPT_twelve_hens_lay_48_eggs_in_twelve_days_l419_41925


namespace NUMINAMATH_GPT_find_f_two_l419_41970

-- Define the function f with the given properties
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 1

-- Given conditions
variable (a b : ℝ)
axiom f_neg_two_zero : f (-2) a b = 0

-- Statement to be proven
theorem find_f_two : f 2 a b = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_f_two_l419_41970


namespace NUMINAMATH_GPT_find_sixth_term_l419_41985

noncomputable def arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def sum_first_n_terms (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem find_sixth_term :
  ∀ (a1 S3 : ℕ),
  a1 = 2 →
  S3 = 12 →
  ∃ d : ℕ, sum_first_n_terms a1 d 3 = S3 ∧ arithmetic_sequence a1 d 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_sixth_term_l419_41985


namespace NUMINAMATH_GPT_consumer_installment_credit_l419_41909

theorem consumer_installment_credit (A C : ℝ) (h1 : A = 0.36 * C) (h2 : 35 = (1 / 3) * A) :
  C = 291.67 :=
by 
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_consumer_installment_credit_l419_41909


namespace NUMINAMATH_GPT_sum_of_perimeters_l419_41995

theorem sum_of_perimeters (a : ℕ → ℝ) (h₁ : a 0 = 180) (h₂ : ∀ n, a (n + 1) = 1 / 2 * a n) :
  (∑' n, a n) = 360 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l419_41995


namespace NUMINAMATH_GPT_initial_salary_increase_l419_41963

theorem initial_salary_increase :
  ∃ x : ℝ, 5000 * (1 + x/100) * 0.95 = 5225 := by
  sorry

end NUMINAMATH_GPT_initial_salary_increase_l419_41963


namespace NUMINAMATH_GPT_maria_total_eggs_l419_41964

def total_eggs (boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  boxes * eggs_per_box

theorem maria_total_eggs :
  total_eggs 3 7 = 21 :=
by
  -- Here, you would normally show the steps of computation
  -- which we can skip with sorry
  sorry

end NUMINAMATH_GPT_maria_total_eggs_l419_41964


namespace NUMINAMATH_GPT_find_a_l419_41906

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l419_41906


namespace NUMINAMATH_GPT_find_number_of_math_problems_l419_41950

-- Define the number of social studies problems
def social_studies_problems : ℕ := 6

-- Define the number of science problems
def science_problems : ℕ := 10

-- Define the time to solve each type of problem in minutes
def time_per_math_problem : ℝ := 2
def time_per_social_studies_problem : ℝ := 0.5
def time_per_science_problem : ℝ := 1.5

-- Define the total time to solve all problems in minutes
def total_time : ℝ := 48

-- Define the theorem to find the number of math problems
theorem find_number_of_math_problems (M : ℕ) :
  time_per_math_problem * M + time_per_social_studies_problem * social_studies_problems + time_per_science_problem * science_problems = total_time → 
  M = 15 :=
by {
  -- proof is not required to be written, hence expressing the unresolved part
  sorry
}

end NUMINAMATH_GPT_find_number_of_math_problems_l419_41950
