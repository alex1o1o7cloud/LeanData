import Mathlib

namespace recurring_decimal_to_fraction_l3460_346014

theorem recurring_decimal_to_fraction :
  ∃ (x : ℚ), x = 3 + 145 / 999 ∧ x = 3142 / 999 := by
  sorry

end recurring_decimal_to_fraction_l3460_346014


namespace exists_monochromatic_triangle_l3460_346042

/-- A type representing the scientists -/
def Scientist : Type := Fin 17

/-- A type representing the topics -/
def Topic : Type := Fin 3

/-- A function representing the correspondence between scientists on a specific topic -/
def corresponds (s1 s2 : Scientist) : Topic :=
  sorry

/-- The main theorem stating that there exists a monochromatic triangle -/
theorem exists_monochromatic_triangle :
  ∃ (s1 s2 s3 : Scientist) (t : Topic),
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
    corresponds s1 s2 = t ∧
    corresponds s2 s3 = t ∧
    corresponds s1 s3 = t :=
  sorry

end exists_monochromatic_triangle_l3460_346042


namespace julia_jonny_stairs_fraction_l3460_346097

theorem julia_jonny_stairs_fraction (jonny_stairs : ℕ) (total_stairs : ℕ) 
  (h1 : jonny_stairs = 1269)
  (h2 : total_stairs = 1685) :
  (total_stairs - jonny_stairs : ℚ) / jonny_stairs = 416 / 1269 := by
  sorry

end julia_jonny_stairs_fraction_l3460_346097


namespace gcd_840_1764_l3460_346080

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l3460_346080


namespace parallel_lines_not_always_equal_l3460_346025

-- Define a line in a plane
structure Line :=
  (extends_infinitely : Bool)
  (can_be_measured : Bool)

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.extends_infinitely ∧ l2.extends_infinitely ∧ ¬l1.can_be_measured ∧ ¬l2.can_be_measured

-- Theorem: Two parallel lines are not always equal
theorem parallel_lines_not_always_equal :
  ∃ l1 l2 : Line, parallel l1 l2 ∧ l1 ≠ l2 :=
sorry

end parallel_lines_not_always_equal_l3460_346025


namespace sammy_remaining_problems_l3460_346012

theorem sammy_remaining_problems (total : ℕ) (fractions decimals multiplication division : ℕ)
  (completed_fractions completed_decimals completed_multiplication completed_division : ℕ)
  (h1 : total = 115)
  (h2 : fractions = 35)
  (h3 : decimals = 40)
  (h4 : multiplication = 20)
  (h5 : division = 20)
  (h6 : completed_fractions = 11)
  (h7 : completed_decimals = 17)
  (h8 : completed_multiplication = 9)
  (h9 : completed_division = 5)
  (h10 : total = fractions + decimals + multiplication + division) :
  total - (completed_fractions + completed_decimals + completed_multiplication + completed_division) = 73 := by
sorry

end sammy_remaining_problems_l3460_346012


namespace triangle_problem_l3460_346003

theorem triangle_problem (a b c A B C : ℝ) (h1 : a = 3) 
  (h2 : (a + b) * Real.sin B = (Real.sin A + Real.sin C) * (a + b - c))
  (h3 : a * Real.cos B + b * Real.cos A = Real.sqrt 3) :
  A = π / 3 ∧ (1 / 2 : ℝ) * a * c = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_problem_l3460_346003


namespace container_volume_ratio_l3460_346085

theorem container_volume_ratio : 
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (4 / 5 : ℚ) * volume_container1 = (2 / 3 : ℚ) * volume_container2 →
  volume_container1 / volume_container2 = 5 / 6 := by
  sorry

end container_volume_ratio_l3460_346085


namespace smallest_sum_of_reciprocals_l3460_346053

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12 → x.val + y.val ≥ 49 := by
  sorry

end smallest_sum_of_reciprocals_l3460_346053


namespace seashell_count_after_six_weeks_l3460_346021

/-- Calculates the number of seashells in Jar A after n weeks -/
def shellsInJarA (n : ℕ) : ℕ := sorry

/-- Calculates the number of seashells in Jar B after n weeks -/
def shellsInJarB (n : ℕ) : ℕ := sorry

/-- Calculates the total number of seashells in both jars after n weeks -/
def totalShells (n : ℕ) : ℕ := shellsInJarA n + shellsInJarB n

theorem seashell_count_after_six_weeks :
  shellsInJarA 0 = 50 →
  shellsInJarB 0 = 30 →
  (∀ k : ℕ, shellsInJarA (k + 1) = shellsInJarA k + 20) →
  (∀ k : ℕ, shellsInJarB (k + 1) = shellsInJarB k * 2) →
  (∀ k : ℕ, k % 3 = 0 → shellsInJarA (k + 1) = shellsInJarA k / 2) →
  (∀ k : ℕ, k % 3 = 0 → shellsInJarB (k + 1) = shellsInJarB k / 2) →
  totalShells 6 = 97 := by
  sorry

end seashell_count_after_six_weeks_l3460_346021


namespace union_M_N_is_half_open_interval_l3460_346006

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {y | ∃ x, y = 2^x ∧ x < 0}

-- State the theorem
theorem union_M_N_is_half_open_interval :
  M ∪ N = Set.Icc 0 1 \ {1} :=
sorry

end union_M_N_is_half_open_interval_l3460_346006


namespace ellipse_dimensions_l3460_346044

/-- Given an ellipse and a parabola with specific properties, prove the dimensions of the ellipse. -/
theorem ellipse_dimensions (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (∀ x y, x^2 / m^2 + y^2 / n^2 = 1) →  -- Ellipse equation
  (∃ x₀, ∀ y, y^2 = 8*x₀ ∧ x₀ = 2) →   -- Parabola focus
  (let c := Real.sqrt (m^2 - n^2);
   c / m = 1 / 2) →                    -- Eccentricity
  m^2 = 16 ∧ n^2 = 12 := by
sorry

end ellipse_dimensions_l3460_346044


namespace complex_distance_to_origin_l3460_346040

theorem complex_distance_to_origin : 
  let z : ℂ := (I^2016 - 2*I^2014) / (2 - I)^2
  Complex.abs z = 3/5 := by
  sorry

end complex_distance_to_origin_l3460_346040


namespace retailer_profit_percent_l3460_346008

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given values is approximately 18.58%. -/
theorem retailer_profit_percent :
  let ε := 0.01
  let result := profit_percent 225 28 300
  (result > 18.58 - ε) ∧ (result < 18.58 + ε) :=
sorry

end retailer_profit_percent_l3460_346008


namespace min_rubles_to_win_l3460_346050

/-- Represents the state of the game --/
structure GameState :=
  (score : ℕ)
  (rubles_spent : ℕ)

/-- The rules of the game --/
def apply_rule (state : GameState) (coin : ℕ) : GameState :=
  match coin with
  | 1 => ⟨state.score + 1, state.rubles_spent + 1⟩
  | 2 => ⟨state.score * 2, state.rubles_spent + 2⟩
  | _ => state

/-- Check if the game is won --/
def is_won (state : GameState) : Bool :=
  state.score = 50

/-- Check if the game is lost --/
def is_lost (state : GameState) : Bool :=
  state.score > 50

/-- The main theorem to prove --/
theorem min_rubles_to_win :
  ∃ (sequence : List ℕ),
    let final_state := sequence.foldl apply_rule ⟨0, 0⟩
    is_won final_state ∧
    final_state.rubles_spent = 11 ∧
    (∀ (other_sequence : List ℕ),
      let other_final_state := other_sequence.foldl apply_rule ⟨0, 0⟩
      is_won other_final_state →
      other_final_state.rubles_spent ≥ 11) :=
by sorry

end min_rubles_to_win_l3460_346050


namespace monotone_decreasing_implies_a_leq_neg_three_l3460_346075

/-- A quadratic function f(x) = x^2 - 2ax + a - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a - 3

/-- The theorem stating that if f(x) is monotonically decreasing on (-∞, -1/4),
    then a ≤ -3 -/
theorem monotone_decreasing_implies_a_leq_neg_three (a : ℝ) :
  (∀ x y, x < y → x < -1/4 → f a x > f a y) → a ≤ -3 :=
by
  sorry

end monotone_decreasing_implies_a_leq_neg_three_l3460_346075


namespace double_counted_page_l3460_346020

theorem double_counted_page (n : ℕ) : 
  (n * (n + 1)) / 2 + 80 = 2550 → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
  (n * (n + 1)) / 2 + k = 2550 → 
  k = 80 := by
sorry

end double_counted_page_l3460_346020


namespace sum_of_reciprocals_l3460_346055

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end sum_of_reciprocals_l3460_346055


namespace only_KC2H3O2_turns_pink_l3460_346010

-- Define the set of solutes
inductive Solute
| NaCl
| KC2H3O2
| LiBr
| NH4NO3

-- Define a function to determine if a solution is basic
def isBasic (s : Solute) : Prop :=
  match s with
  | Solute.KC2H3O2 => True
  | _ => False

-- Define a function to check if phenolphthalein turns pink
def turnsPink (s : Solute) : Prop := isBasic s

-- Theorem statement
theorem only_KC2H3O2_turns_pink :
  ∀ s : Solute, turnsPink s ↔ s = Solute.KC2H3O2 :=
by sorry

end only_KC2H3O2_turns_pink_l3460_346010


namespace no_rain_probability_l3460_346065

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end no_rain_probability_l3460_346065


namespace expression_equality_l3460_346023

theorem expression_equality : 2⁻¹ - Real.sqrt 3 * Real.tan (60 * π / 180) + (π - 2011)^0 + |(-1/2)| = -1 := by
  sorry

end expression_equality_l3460_346023


namespace statue_original_cost_l3460_346013

theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 660)
  (h2 : profit_percentage = 20) :
  let original_cost := selling_price / (1 + profit_percentage / 100)
  original_cost = 550 := by
sorry

end statue_original_cost_l3460_346013


namespace flower_producing_plants_l3460_346089

theorem flower_producing_plants 
  (daisy_seeds sunflower_seeds : ℕ)
  (daisy_germination_rate sunflower_germination_rate flower_production_rate : ℚ)
  (h1 : daisy_seeds = 25)
  (h2 : sunflower_seeds = 25)
  (h3 : daisy_germination_rate = 3/5)
  (h4 : sunflower_germination_rate = 4/5)
  (h5 : flower_production_rate = 4/5) :
  ⌊(daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate⌋ = 28 :=
by sorry

end flower_producing_plants_l3460_346089


namespace nine_integer_segments_l3460_346086

/-- Right triangle XYZ with integer leg lengths -/
structure RightTriangle where
  xy : ℕ
  yz : ℕ

/-- The number of different integer length line segments from Y to XZ -/
def countIntegerSegments (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle with XY = 15 and YZ = 20 -/
def specialTriangle : RightTriangle :=
  { xy := 15, yz := 20 }

/-- Theorem stating that the number of integer length segments is 9 -/
theorem nine_integer_segments :
  countIntegerSegments specialTriangle = 9 :=
sorry

end nine_integer_segments_l3460_346086


namespace geometric_sequence_sixth_term_l3460_346039

/-- Given a geometric sequence where a₁ = 3 and a₃ = 1/9, prove that a₆ = 1/81 -/
theorem geometric_sequence_sixth_term (a : ℕ → ℚ) (h1 : a 1 = 3) (h3 : a 3 = 1/9) :
  a 6 = 1/81 := by
  sorry

end geometric_sequence_sixth_term_l3460_346039


namespace sales_increase_percentage_l3460_346037

def saturday_sales : ℕ := 60
def total_sales : ℕ := 150

def sunday_sales : ℕ := total_sales - saturday_sales

def percentage_increase : ℚ := (sunday_sales - saturday_sales : ℚ) / saturday_sales * 100

theorem sales_increase_percentage :
  sunday_sales > saturday_sales →
  percentage_increase = 50 := by
  sorry

end sales_increase_percentage_l3460_346037


namespace multiply_algebraic_expression_l3460_346047

theorem multiply_algebraic_expression (a b : ℝ) : -3 * a * b * (2 * a) = -6 * a^2 * b := by
  sorry

end multiply_algebraic_expression_l3460_346047


namespace slope_interpretation_l3460_346087

/-- Regression line equation for poverty and education data -/
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

/-- Theorem stating the relationship between changes in x and y -/
theorem slope_interpretation (x₁ x₂ : ℝ) (h : x₂ = x₁ + 1) :
  regression_line x₂ - regression_line x₁ = 0.8 := by
  sorry

end slope_interpretation_l3460_346087


namespace fully_filled_boxes_l3460_346005

theorem fully_filled_boxes (total_cards : ℕ) (max_per_box : ℕ) (h1 : total_cards = 94) (h2 : max_per_box = 8) :
  (total_cards / max_per_box : ℕ) = 11 :=
by sorry

end fully_filled_boxes_l3460_346005


namespace potatoes_cooked_l3460_346098

theorem potatoes_cooked (total : ℕ) (cooking_time : ℕ) (remaining_time : ℕ) : 
  total = 15 → 
  cooking_time = 8 → 
  remaining_time = 72 → 
  total - (remaining_time / cooking_time) = 6 := by
sorry

end potatoes_cooked_l3460_346098


namespace continuous_fraction_equality_l3460_346067

theorem continuous_fraction_equality : 1 + 2 / (3 + 6/7) = 41/27 := by
  sorry

end continuous_fraction_equality_l3460_346067


namespace carrie_iphone_weeks_l3460_346052

/-- Proves that Carrie needs to work 7 weeks to buy the iPhone -/
theorem carrie_iphone_weeks : 
  ∀ (iphone_cost trade_in_value weekly_earnings : ℕ),
    iphone_cost = 800 →
    trade_in_value = 240 →
    weekly_earnings = 80 →
    (iphone_cost - trade_in_value) / weekly_earnings = 7 := by
  sorry

end carrie_iphone_weeks_l3460_346052


namespace complex_equation_roots_l3460_346092

theorem complex_equation_roots : 
  let z₁ : ℂ := (1 + 2 * Real.sqrt 7 - Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (1 - 2 * Real.sqrt 7 + Complex.I * Real.sqrt 7) / 2
  (z₁ ^ 2 - z₁ = 3 - 7 * Complex.I) ∧ (z₂ ^ 2 - z₂ = 3 - 7 * Complex.I) := by
  sorry

end complex_equation_roots_l3460_346092


namespace remainder_1999_div_7_l3460_346030

theorem remainder_1999_div_7 : 1999 % 7 = 1 := by
  sorry

end remainder_1999_div_7_l3460_346030


namespace smallest_digit_for_divisibility_by_nine_l3460_346036

theorem smallest_digit_for_divisibility_by_nine :
  ∀ d : Nat, d ≤ 9 →
    (526000 + d * 1000 + 45) % 9 = 0 →
    d ≥ 5 :=
by sorry

end smallest_digit_for_divisibility_by_nine_l3460_346036


namespace sum_of_quadratic_solutions_l3460_346096

theorem sum_of_quadratic_solutions : 
  let f (x : ℝ) := x^2 - 6*x - 22 - (2*x + 18)
  let roots := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ + x₂ = 8 := by
  sorry

end sum_of_quadratic_solutions_l3460_346096


namespace milk_water_ratio_after_addition_l3460_346049

/-- Calculates the final ratio of milk to water after adding water to a mixture -/
theorem milk_water_ratio_after_addition
  (initial_volume : ℚ)
  (initial_milk_ratio : ℚ)
  (initial_water_ratio : ℚ)
  (added_water : ℚ)
  (h1 : initial_volume = 45)
  (h2 : initial_milk_ratio = 4)
  (h3 : initial_water_ratio = 1)
  (h4 : added_water = 21) :
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water
  let final_milk_ratio := initial_milk / final_water
  let final_water_ratio := final_water / final_water
  (final_milk_ratio : ℚ) / (final_water_ratio : ℚ) = 6 / 5 := by
  sorry

end milk_water_ratio_after_addition_l3460_346049


namespace triangle_height_l3460_346032

theorem triangle_height (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) : 
  let area := (x^3 * y)^2
  let side := (2 * x * y)^2
  let height := (1/2) * x^4
  area = (1/2) * side * height := by
sorry

end triangle_height_l3460_346032


namespace intersection_of_A_and_B_l3460_346045

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end intersection_of_A_and_B_l3460_346045


namespace system_solution_l3460_346083

theorem system_solution (x y a : ℝ) : 
  x + 2 * y = a ∧ 
  x - 2 * y = 2 ∧ 
  x = 4 → 
  a = 6 ∧ y = 1 := by
sorry

end system_solution_l3460_346083


namespace apple_picking_contest_l3460_346061

/-- The number of apples picked by Marin -/
def marin_apples : ℕ := 9

/-- The number of apples picked by Donald -/
def donald_apples : ℕ := 11

/-- The number of apples picked by Ana -/
def ana_apples : ℕ := 2 * (marin_apples + donald_apples)

/-- The total number of apples picked by all three participants -/
def total_apples : ℕ := marin_apples + donald_apples + ana_apples

theorem apple_picking_contest :
  total_apples = 60 := by sorry

end apple_picking_contest_l3460_346061


namespace slope_of_line_parallel_lines_coefficient_l3460_346094

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
theorem slope_of_line (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a / b) * x - c / b :=
sorry

theorem parallel_lines_coefficient (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = 0) → a = -6 :=
sorry

end slope_of_line_parallel_lines_coefficient_l3460_346094


namespace ellipse_condition_l3460_346058

/-- If the equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 represents an ellipse, then m > 5 -/
theorem ellipse_condition (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2 ↔
      (x^2 / a^2) + ((y + 1)^2 / b^2) = 1) →
  m > 5 :=
by sorry

end ellipse_condition_l3460_346058


namespace factor_expression_l3460_346051

theorem factor_expression (x : ℝ) : 81 * x^3 + 27 * x^2 = 27 * x^2 * (3 * x + 1) := by
  sorry

end factor_expression_l3460_346051


namespace class_division_transfer_l3460_346011

/-- 
Given a class divided into two groups with 26 and 22 people respectively,
prove that the number of people transferred (x) from the second group to the first
satisfies the equation x + 26 = 3(22 - x) when the first group becomes
three times the size of the second group after the transfer.
-/
theorem class_division_transfer (x : ℤ) : x + 26 = 3 * (22 - x) ↔ 
  (26 + x = 3 * (22 - x) ∧ 
   26 + x > 0 ∧
   22 - x > 0) := by
  sorry

#check class_division_transfer

end class_division_transfer_l3460_346011


namespace max_value_rational_function_l3460_346024

theorem max_value_rational_function : 
  ∃ (n : ℕ), n = 97 ∧ 
  (∀ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) ≤ n) ∧
  (∀ m : ℕ, m > n → ∃ x : ℝ, (4 * x^2 + 12 * x + 29) / (4 * x^2 + 12 * x + 5) < m) :=
by sorry

end max_value_rational_function_l3460_346024


namespace rumor_day_seven_l3460_346009

def rumor_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

theorem rumor_day_seven :
  (∀ k < 7, rumor_spread k < 3280) ∧ rumor_spread 7 ≥ 3280 := by
  sorry

end rumor_day_seven_l3460_346009


namespace roots_sequence_property_l3460_346031

/-- Given x₁ and x₂ are roots of x² - 6x + 1 = 0, prove that for all natural numbers n,
    aₙ = x₁ⁿ + x₂ⁿ is an integer and not a multiple of 5. -/
theorem roots_sequence_property (x₁ x₂ : ℝ) (h : x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0) :
  ∀ n : ℕ, ∃ k : ℤ, (x₁^n + x₂^n = k) ∧ ¬(5 ∣ k) := by
  sorry

end roots_sequence_property_l3460_346031


namespace range_of_a_minus_b_l3460_346095

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 1) (hb : 0 < b ∧ b < 4) :
  ∀ x, (∃ y z, -2 < y ∧ y < 1 ∧ 0 < z ∧ z < 4 ∧ x = y - z) ↔ -6 < x ∧ x < 1 :=
by sorry

end range_of_a_minus_b_l3460_346095


namespace arithmetic_geometric_mean_inequality_l3460_346043

theorem arithmetic_geometric_mean_inequality {x y : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) ∧
  ((x + y) / 2 = Real.sqrt (x * y) ↔ x = y) := by
  sorry

end arithmetic_geometric_mean_inequality_l3460_346043


namespace father_son_age_relation_l3460_346059

/-- Proves the number of years it takes for a father to be twice as old as his son -/
theorem father_son_age_relation (father_age : ℕ) (son_age : ℕ) (years : ℕ) : 
  father_age = 45 →
  father_age = 3 * son_age →
  father_age + years = 2 * (son_age + years) →
  years = 15 := by
sorry

end father_son_age_relation_l3460_346059


namespace order_of_integrals_l3460_346073

theorem order_of_integrals : 
  let a : ℝ := ∫ x in (0:ℝ)..2, x^2
  let b : ℝ := ∫ x in (0:ℝ)..2, x^3
  let c : ℝ := ∫ x in (0:ℝ)..2, Real.sin x
  c < a ∧ a < b := by sorry

end order_of_integrals_l3460_346073


namespace total_paintable_area_l3460_346015

def bedroom_type1_length : ℝ := 14
def bedroom_type1_width : ℝ := 11
def bedroom_type1_height : ℝ := 9
def bedroom_type2_length : ℝ := 13
def bedroom_type2_width : ℝ := 12
def bedroom_type2_height : ℝ := 9
def num_bedrooms : ℕ := 4
def unpaintable_area : ℝ := 70

def wall_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℝ) : ℝ :=
  total_area - unpaintable_area

theorem total_paintable_area :
  let type1_area := wall_area bedroom_type1_length bedroom_type1_width bedroom_type1_height
  let type2_area := wall_area bedroom_type2_length bedroom_type2_width bedroom_type2_height
  let total_area := (num_bedrooms / 2) * (paintable_area type1_area unpaintable_area + 
                                          paintable_area type2_area unpaintable_area)
  total_area = 1520 := by
  sorry

end total_paintable_area_l3460_346015


namespace grunters_win_probability_l3460_346069

theorem grunters_win_probability (num_games : ℕ) (win_prob : ℚ) :
  num_games = 6 →
  win_prob = 3/5 →
  (win_prob ^ num_games : ℚ) = 729/15625 := by
  sorry

end grunters_win_probability_l3460_346069


namespace sales_volume_estimate_l3460_346060

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := -5 * x + 150

-- Define the theorem
theorem sales_volume_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |regression_equation 10 - 100| < ε :=
sorry

end sales_volume_estimate_l3460_346060


namespace janet_number_problem_l3460_346068

theorem janet_number_problem (x : ℝ) : ((x - 3) * 3 + 3) / 3 = 10 → x = 12 := by
  sorry

end janet_number_problem_l3460_346068


namespace zero_of_f_l3460_346026

def f (x : ℝ) : ℝ := 4 * x - 2

theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1/2 := by
  sorry

end zero_of_f_l3460_346026


namespace quadratic_x_axis_intersection_l3460_346007

theorem quadratic_x_axis_intersection (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 4 * x + 1 = 0) ↔ (a ≤ 4 ∧ a ≠ 0) := by sorry

end quadratic_x_axis_intersection_l3460_346007


namespace stacys_farm_chickens_l3460_346022

theorem stacys_farm_chickens :
  ∀ (total_animals sick_animals piglets goats : ℕ),
    piglets = 40 →
    goats = 34 →
    sick_animals = 50 →
    2 * sick_animals = total_animals →
    total_animals = piglets + goats + (total_animals - piglets - goats) →
    total_animals - piglets - goats = 26 := by
  sorry

end stacys_farm_chickens_l3460_346022


namespace dog_weight_ratio_l3460_346082

/-- Represents the weight of a dog at different ages --/
structure DogWeight where
  week7 : ℝ
  week9 : ℝ
  month3 : ℝ
  month5 : ℝ
  year1 : ℝ

/-- Theorem stating the ratio of a dog's weight at 9 weeks to 7 weeks --/
theorem dog_weight_ratio (w : DogWeight) 
  (h1 : w.week7 = 6)
  (h2 : w.month3 = 2 * w.week9)
  (h3 : w.month5 = 2 * w.month3)
  (h4 : w.year1 = w.month5 + 30)
  (h5 : w.year1 = 78) :
  w.week9 / w.week7 = 2 := by
  sorry

#check dog_weight_ratio

end dog_weight_ratio_l3460_346082


namespace monthly_income_p_l3460_346081

def average_income (x y : ℕ) := (x + y) / 2

theorem monthly_income_p (p q r : ℕ) 
  (h1 : average_income p q = 5050)
  (h2 : average_income q r = 6250)
  (h3 : average_income p r = 5200) :
  p = 4000 := by
  sorry

end monthly_income_p_l3460_346081


namespace builder_project_l3460_346091

/-- A builder's project involving bolts and nuts -/
theorem builder_project (bolt_boxes : ℕ) (bolts_per_box : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ)
  (leftover_bolts : ℕ) (leftover_nuts : ℕ) :
  bolt_boxes = 7 →
  bolts_per_box = 11 →
  nut_boxes = 3 →
  nuts_per_box = 15 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  (bolt_boxes * bolts_per_box - leftover_bolts) + (nut_boxes * nuts_per_box - leftover_nuts) = 113 :=
by sorry

end builder_project_l3460_346091


namespace orthocenter_of_triangle_PQR_l3460_346084

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle PQR is (3/2, 13/2, 5) -/
theorem orthocenter_of_triangle_PQR :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  let Q : ℝ × ℝ × ℝ := (6, 4, 2)
  let R : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter P Q R = (3/2, 13/2, 5) :=
by
  sorry

end orthocenter_of_triangle_PQR_l3460_346084


namespace batsman_average_l3460_346088

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) :
  total_innings = 12 →
  last_innings_score = 65 →
  average_increase = 2 →
  (∃ (prev_average : ℕ),
    (prev_average * (total_innings - 1) + last_innings_score) / total_innings = prev_average + average_increase) →
  (((total_innings - 1) * ((last_innings_score + (total_innings - 1) * average_increase) / total_innings - average_increase) + last_innings_score) / total_innings) = 43 :=
by sorry

end batsman_average_l3460_346088


namespace april_plant_arrangement_l3460_346041

/-- The number of ways to arrange plants with specific conditions -/
def plant_arrangements (n_basil : ℕ) (n_tomato : ℕ) : ℕ :=
  (n_basil + n_tomato - 1).factorial * (n_basil - 1).factorial

/-- Theorem stating the number of arrangements for the given problem -/
theorem april_plant_arrangement :
  plant_arrangements 5 3 = 576 := by
  sorry

end april_plant_arrangement_l3460_346041


namespace complex_number_properties_l3460_346093

/-- Given a complex number z and a real number m, where z = m^2 - m - 2 + (5m^2 - 20)i -/
theorem complex_number_properties (m : ℝ) (z : ℂ) 
  (h : z = (m^2 - m - 2 : ℝ) + (5 * m^2 - 20 : ℝ) * Complex.I) :
  (z.im = 0 ↔ m = 2 ∨ m = -2) ∧ 
  (z.re = 0 ∧ z.im ≠ 0 ↔ m = -1) := by
  sorry

end complex_number_properties_l3460_346093


namespace fixed_point_of_exponential_function_l3460_346090

/-- The function f(x) = a^(x-2016) + 1 has a fixed point at (2016, 2) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = a^(x - 2016) + 1) ∧ f 2016 = 2 :=
by sorry

end fixed_point_of_exponential_function_l3460_346090


namespace root_value_range_l3460_346099

theorem root_value_range (a : ℝ) (h : a^2 + 3*a - 1 = 0) :
  2 < a^2 + 3*a + Real.sqrt 3 ∧ a^2 + 3*a + Real.sqrt 3 < 3 := by
  sorry

end root_value_range_l3460_346099


namespace defective_shipped_percentage_l3460_346002

theorem defective_shipped_percentage 
  (defective_rate : Real) 
  (shipped_rate : Real) 
  (h1 : defective_rate = 0.08) 
  (h2 : shipped_rate = 0.04) : 
  defective_rate * shipped_rate = 0.0032 := by
  sorry

#check defective_shipped_percentage

end defective_shipped_percentage_l3460_346002


namespace trigonometric_equation_solution_l3460_346077

theorem trigonometric_equation_solution (x y : ℝ) : 
  x = π / 6 → 
  Real.sin x * Real.cos x * y - 2 * Real.sin x * Real.sin x * y + Real.cos x * y = 1/2 → 
  y = (6 * Real.sqrt 3 + 4) / 23 := by
sorry

end trigonometric_equation_solution_l3460_346077


namespace anna_chargers_l3460_346029

theorem anna_chargers (phone_chargers : ℕ) (laptop_chargers : ℕ) : 
  laptop_chargers = 5 * phone_chargers →
  phone_chargers + laptop_chargers = 24 →
  phone_chargers = 4 := by
sorry

end anna_chargers_l3460_346029


namespace patent_agency_employment_relation_l3460_346000

/-- Data for graduate students and their preferences for patent agency employment --/
structure GraduateData where
  total : ℕ
  male_like : ℕ
  male_dislike : ℕ
  female_like : ℕ
  female_dislike : ℕ

/-- Calculate the probability of selecting at least 2 students who like employment
    in patent agency when 3 are selected --/
def probability_at_least_two (data : GraduateData) : ℚ :=
  let p := (data.male_like + data.female_like : ℚ) / data.total
  3 * p^2 * (1 - p) + p^3

/-- Calculate the chi-square statistic for the given data --/
def chi_square (data : GraduateData) : ℚ :=
  let n := data.total
  let a := data.male_like
  let b := data.male_dislike
  let c := data.female_like
  let d := data.female_dislike
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The main theorem to be proved --/
theorem patent_agency_employment_relation (data : GraduateData)
  (h_total : data.total = 200)
  (h_male_like : data.male_like = 60)
  (h_male_dislike : data.male_dislike = 40)
  (h_female_like : data.female_like = 80)
  (h_female_dislike : data.female_dislike = 20) :
  probability_at_least_two data = 98/125 ∧ chi_square data > 7879/1000 :=
sorry


end patent_agency_employment_relation_l3460_346000


namespace total_problems_is_550_l3460_346004

/-- The total number of math problems practiced by Marvin, Arvin, and Kevin over two days -/
def totalProblems (marvinYesterday : ℕ) : ℕ :=
  let marvinToday := 3 * marvinYesterday
  let arvinYesterday := 2 * marvinYesterday
  let arvinToday := 2 * marvinToday
  let kevinYesterday := 30
  let kevinToday := kevinYesterday + 10
  (marvinYesterday + marvinToday) + (arvinYesterday + arvinToday) + (kevinYesterday + kevinToday)

/-- Theorem stating that the total number of problems practiced is 550 -/
theorem total_problems_is_550 : totalProblems 40 = 550 := by
  sorry

end total_problems_is_550_l3460_346004


namespace lewis_earnings_l3460_346034

/-- Lewis's earnings during harvest season --/
theorem lewis_earnings (weekly_earnings weekly_rent : ℕ) (harvest_weeks : ℕ) : 
  weekly_earnings = 403 → 
  weekly_rent = 49 → 
  harvest_weeks = 233 → 
  (weekly_earnings * harvest_weeks) - (weekly_rent * harvest_weeks) = 82482 := by
  sorry

end lewis_earnings_l3460_346034


namespace line_through_point_equal_intercepts_l3460_346048

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equalIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- The main theorem
theorem line_through_point_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (pointOnLine ⟨3, -2⟩ l₁) ∧
    (pointOnLine ⟨3, -2⟩ l₂) ∧
    (equalIntercepts l₁ ∨ (l₁.a = 0 ∧ l₁.b = 0)) ∧
    (equalIntercepts l₂ ∨ (l₂.a = 0 ∧ l₂.b = 0)) ∧
    ((l₁.a = 2 ∧ l₁.b = 3 ∧ l₁.c = 0) ∨ (l₂.a = 1 ∧ l₂.b = 1 ∧ l₂.c = -1)) :=
  sorry

end line_through_point_equal_intercepts_l3460_346048


namespace path_length_eq_three_times_PQ_l3460_346066

/-- The length of the segment PQ -/
def PQ_length : ℝ := 73

/-- The length of the path along the squares constructed on the segments of PQ -/
def path_length : ℝ := 3 * PQ_length

theorem path_length_eq_three_times_PQ : path_length = 3 * PQ_length := by
  sorry

end path_length_eq_three_times_PQ_l3460_346066


namespace world_cup_investment_scientific_notation_l3460_346054

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem world_cup_investment_scientific_notation :
  toScientificNotation 220000000000 = ScientificNotation.mk 2.2 11 (by norm_num) :=
sorry

end world_cup_investment_scientific_notation_l3460_346054


namespace other_endpoint_of_line_segment_l3460_346038

/-- Given a line segment with midpoint (3, -1) and one endpoint (7, 3), 
    prove that the other endpoint is (-1, -5). -/
theorem other_endpoint_of_line_segment 
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, -1))
  (h_endpoint1 : endpoint1 = (7, 3)) :
  ∃ endpoint2 : ℝ × ℝ, 
    endpoint2 = (-1, -5) ∧ 
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end other_endpoint_of_line_segment_l3460_346038


namespace max_value_of_f_l3460_346078

-- Define the quadratic function
def f (x : ℝ) : ℝ := -8 * x^2 + 32 * x - 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 31 := by
  sorry

end max_value_of_f_l3460_346078


namespace gcd_lcm_sum_l3460_346033

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 8 32 = 46 := by
  sorry

end gcd_lcm_sum_l3460_346033


namespace largest_five_digit_distinct_odd_number_l3460_346064

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d < 10

def is_five_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_are_distinct (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 5 ∧ 0 ≤ j ∧ j < 5 → i ≠ j →
    (n / 10^i) % 10 ≠ (n / 10^j) % 10

def all_digits_odd (n : ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i < 5 → is_odd_digit ((n / 10^i) % 10)

theorem largest_five_digit_distinct_odd_number :
  ∀ n : ℕ, is_five_digit_number n → digits_are_distinct n → all_digits_odd n →
    n ≤ 97531 := by sorry

end largest_five_digit_distinct_odd_number_l3460_346064


namespace valid_numbers_l3460_346063

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  x + y + z = (10 * x + y) - (10 * y + z)

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {209, 428, 647, 866, 214, 433, 652, 871} :=
by sorry

end valid_numbers_l3460_346063


namespace mowing_time_calculation_l3460_346074

/-- Calculates the time required to mow a rectangular lawn -/
def mowing_time (length width swath overlap speed : ℚ) : ℚ :=
  let effective_swath := (swath - overlap) / 12
  let strips := width / effective_swath
  let total_distance := strips * length
  total_distance / speed

/-- Theorem stating the time required to mow the lawn under given conditions -/
theorem mowing_time_calculation :
  mowing_time 100 180 (30/12) (6/12) 4000 = 2.25 := by
  sorry

end mowing_time_calculation_l3460_346074


namespace complex_distance_sum_constant_l3460_346028

theorem complex_distance_sum_constant (w : ℂ) (h : Complex.abs (w - (3 + 2*I)) = 3) :
  Complex.abs (w - (2 - 3*I))^2 + Complex.abs (w - (4 + 5*I))^2 = 71 := by
  sorry

end complex_distance_sum_constant_l3460_346028


namespace committee_selection_l3460_346019

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end committee_selection_l3460_346019


namespace exact_defective_selection_l3460_346072

def total_products : ℕ := 100
def defective_products : ℕ := 3
def products_to_select : ℕ := 4
def defective_to_select : ℕ := 2

theorem exact_defective_selection :
  (Nat.choose defective_products defective_to_select) *
  (Nat.choose (total_products - defective_products) (products_to_select - defective_to_select)) = 13968 := by
  sorry

end exact_defective_selection_l3460_346072


namespace solution_set_of_inequalities_l3460_346018

theorem solution_set_of_inequalities :
  let S := { x : ℝ | x - 2 > 1 ∧ x < 4 }
  S = { x : ℝ | 3 < x ∧ x < 4 } := by
  sorry

end solution_set_of_inequalities_l3460_346018


namespace prop_equivalence_l3460_346076

theorem prop_equivalence (p q : Prop) : (p ∧ q) ↔ ¬(¬p ∨ ¬q) := by
  sorry

end prop_equivalence_l3460_346076


namespace permutations_count_l3460_346046

def word_length : ℕ := 12
def repeated_letter_count : ℕ := 2

theorem permutations_count :
  (word_length.factorial / repeated_letter_count.factorial) = 239500800 := by
  sorry

end permutations_count_l3460_346046


namespace parabola_intercepts_sum_l3460_346062

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

/-- The y-intercept of the parabola -/
def d : ℝ := parabola 0

/-- The x-intercepts of the parabola -/
noncomputable def e : ℝ := (9 + Real.sqrt 33) / 6
noncomputable def f : ℝ := (9 - Real.sqrt 33) / 6

/-- Theorem stating that the sum of the y-intercept and x-intercepts is 7 -/
theorem parabola_intercepts_sum : d + e + f = 7 := by sorry

end parabola_intercepts_sum_l3460_346062


namespace coin_payment_difference_l3460_346017

/-- Represents the available coin denominations in cents -/
inductive Coin : Type
  | OneCent : Coin
  | TenCent : Coin
  | TwentyCent : Coin

/-- The value of a coin in cents -/
def coin_value : Coin → ℕ
  | Coin.OneCent => 1
  | Coin.TenCent => 10
  | Coin.TwentyCent => 20

/-- A function that returns true if a list of coins sums to the target amount -/
def sum_to_target (coins : List Coin) (target : ℕ) : Prop :=
  (coins.map coin_value).sum = target

/-- The proposition to be proved -/
theorem coin_payment_difference (target : ℕ := 50) :
  ∃ (min_coins max_coins : List Coin),
    sum_to_target min_coins target ∧
    sum_to_target max_coins target ∧
    (max_coins.length - min_coins.length = 47) :=
  sorry

end coin_payment_difference_l3460_346017


namespace angle_measure_in_triangle_l3460_346056

theorem angle_measure_in_triangle (y : ℝ) : 
  let angle_ABC : ℝ := 180
  let angle_CBD : ℝ := 115
  let angle_BAD : ℝ := 31
  angle_ABC = 180 ∧ 
  angle_CBD = 115 ∧ 
  angle_BAD = 31 ∧
  y + angle_BAD + (angle_ABC - angle_CBD) = 180
  → y = 84 := by sorry

end angle_measure_in_triangle_l3460_346056


namespace machine_doesnt_require_repair_l3460_346016

/-- Represents a weighing machine --/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  all_deviations_bounded : Prop
  standard_deviation_bounded : Prop

/-- Determines if a weighing machine requires repair --/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨
  ¬m.all_deviations_bounded ∨
  ¬m.standard_deviation_bounded

/-- Theorem stating that the machine does not require repair --/
theorem machine_doesnt_require_repair (m : WeighingMachine)
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.all_deviations_bounded)
  (h4 : m.standard_deviation_bounded) :
  ¬(requires_repair m) :=
sorry

end machine_doesnt_require_repair_l3460_346016


namespace division_problem_l3460_346079

theorem division_problem : ∃ A : ℕ, 23 = 6 * A + 5 ∧ A = 3 := by
  sorry

end division_problem_l3460_346079


namespace function_value_ordering_l3460_346070

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is increasing on [0, +∞) if f(x) ≤ f(y) for all 0 ≤ x ≤ y -/
def IncreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem function_value_ordering (f : ℝ → ℝ) 
    (heven : EvenFunction f) (hincr : IncreasingOnNonnegative f) :
    f 1 < f (-2) ∧ f (-2) < f (-3) := by
  sorry

end function_value_ordering_l3460_346070


namespace equilateral_triangle_perimeter_l3460_346035

/-- The perimeter of an equilateral triangle with side length 5 cm is 15 cm. -/
theorem equilateral_triangle_perimeter :
  ∀ (side_length perimeter : ℝ),
  side_length = 5 →
  perimeter = 3 * side_length →
  perimeter = 15 :=
by sorry

end equilateral_triangle_perimeter_l3460_346035


namespace cookie_boxes_problem_l3460_346057

theorem cookie_boxes_problem (type1_per_box type3_per_box : ℕ)
  (type1_boxes type2_boxes type3_boxes : ℕ)
  (total_cookies : ℕ)
  (h1 : type1_per_box = 12)
  (h2 : type3_per_box = 16)
  (h3 : type1_boxes = 50)
  (h4 : type2_boxes = 80)
  (h5 : type3_boxes = 70)
  (h6 : total_cookies = 3320)
  (h7 : type1_per_box * type1_boxes + type2_boxes * type2_per_box + type3_per_box * type3_boxes = total_cookies) :
  type2_per_box = 20 := by
  sorry


end cookie_boxes_problem_l3460_346057


namespace summit_academy_contestants_l3460_346071

theorem summit_academy_contestants (s j : ℕ) (h : s / 3 = j * 3 / 4) : s = 4 * j := by
  sorry

end summit_academy_contestants_l3460_346071


namespace smallest_d_for_g_range_three_l3460_346001

/-- The function g(x) defined as x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- Theorem stating that 7 is the smallest value of d for which 3 is in the range of g(x) -/
theorem smallest_d_for_g_range_three :
  (∃ (d : ℝ), (∃ (x : ℝ), g d x = 3) ∧ (∀ (d' : ℝ), d' < d → ¬∃ (x : ℝ), g d' x = 3)) ∧
  (∃ (x : ℝ), g 7 x = 3) :=
sorry

end smallest_d_for_g_range_three_l3460_346001


namespace sphere_volume_after_drilling_l3460_346027

/-- The remaining volume of a sphere after drilling cylindrical holes -/
theorem sphere_volume_after_drilling (sphere_diameter : ℝ) 
  (hole1_depth hole1_diameter : ℝ) 
  (hole2_depth hole2_diameter : ℝ) 
  (hole3_depth hole3_diameter : ℝ) : 
  sphere_diameter = 24 →
  hole1_depth = 10 → hole1_diameter = 3 →
  hole2_depth = 10 → hole2_diameter = 3 →
  hole3_depth = 5 → hole3_diameter = 4 →
  (4 / 3 * π * (sphere_diameter / 2)^3) - 
  (π * (hole1_diameter / 2)^2 * hole1_depth) - 
  (π * (hole2_diameter / 2)^2 * hole2_depth) - 
  (π * (hole3_diameter / 2)^2 * hole3_depth) = 2239 * π := by
  sorry

end sphere_volume_after_drilling_l3460_346027
