import Mathlib

namespace NUMINAMATH_CALUDE_find_A_l1181_118196

theorem find_A : ∃ A : ℕ, 
  (1047 % A = 23) ∧ 
  (1047 % (A + 1) = 7) ∧ 
  (A = 64) := by
sorry

end NUMINAMATH_CALUDE_find_A_l1181_118196


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l1181_118167

theorem two_digit_number_puzzle :
  ∀ x y : ℕ,
  (10 ≤ 10 * x + y) ∧ (10 * x + y < 100) →  -- two-digit number condition
  (x + y) * 3 = 10 * x + y - 2 →             -- puzzle condition
  x = 2 :=                                   -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l1181_118167


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1181_118120

theorem smaller_number_in_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Two positive numbers
  b / a = 11 / 7 ∧  -- In the ratio 7:11
  b - a = 16  -- Larger number exceeds smaller by 16
  → a = 28 := by  -- The smaller number is 28
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1181_118120


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1181_118175

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := n + 1

-- Define the sum of the first n terms of 2a_n
def S (n : ℕ) : ℝ := 2^(n+2) - 4

-- Theorem statement
theorem arithmetic_sequence_properties :
  (a 1 = 2) ∧ 
  (a 1 + a 2 + a 3 = 9) ∧
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, S n = 2^(n+2) - 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1181_118175


namespace NUMINAMATH_CALUDE_kennel_dogs_l1181_118114

theorem kennel_dogs (total : ℕ) (long_fur : ℕ) (brown : ℕ) (long_fur_and_brown : ℕ)
  (h_total : total = 45)
  (h_long_fur : long_fur = 29)
  (h_brown : brown = 17)
  (h_long_fur_and_brown : long_fur_and_brown = 9) :
  total - (long_fur + brown - long_fur_and_brown) = 8 :=
by sorry

end NUMINAMATH_CALUDE_kennel_dogs_l1181_118114


namespace NUMINAMATH_CALUDE_marching_band_formation_l1181_118161

/-- A function that returns the list of divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A function that returns the number of divisors of a natural number within a given range -/
def countDivisorsInRange (n m l : ℕ) : ℕ := sorry

theorem marching_band_formation (total_musicians : ℕ) (min_per_row : ℕ) (num_formations : ℕ) 
  (h1 : total_musicians = 240)
  (h2 : min_per_row = 8)
  (h3 : num_formations = 8) :
  ∃ (max_per_row : ℕ), 
    countDivisorsInRange total_musicians min_per_row max_per_row = num_formations ∧ 
    max_per_row = 80 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_formation_l1181_118161


namespace NUMINAMATH_CALUDE_f_geq_m_range_l1181_118177

/-- The function f(x) = x^2 - 2mx + 2 -/
def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2

/-- The theorem stating the range of m for which f(x) ≥ m holds for all x ∈ [-1, +∞) -/
theorem f_geq_m_range (m : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x m ≥ m) ↔ -3 ≤ m ∧ m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_geq_m_range_l1181_118177


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l1181_118171

def total_crayons : ℕ := 15
def red_crayons : ℕ := 3
def selected_crayons : ℕ := 5

theorem crayon_selection_theorem :
  (Nat.choose total_crayons selected_crayons - 
   Nat.choose (total_crayons - red_crayons) selected_crayons) = 2211 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l1181_118171


namespace NUMINAMATH_CALUDE_equations_different_graphs_l1181_118158

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := y = (2 * x^2 - 18) / (x + 3)
def eq3 (x y : ℝ) : Prop := (x + 3) * y = 2 * x^2 - 18

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem equations_different_graphs :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end NUMINAMATH_CALUDE_equations_different_graphs_l1181_118158


namespace NUMINAMATH_CALUDE_margie_change_is_six_l1181_118131

/-- The change Margie received after buying apples -/
def margieChange (numApples : ℕ) (costPerApple : ℚ) (amountPaid : ℚ) : ℚ :=
  amountPaid - (numApples : ℚ) * costPerApple

/-- Theorem: Margie's change is $6.00 -/
theorem margie_change_is_six :
  margieChange 5 (80 / 100) 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_margie_change_is_six_l1181_118131


namespace NUMINAMATH_CALUDE_vendor_profit_calculation_l1181_118106

/-- Calculates the profit for a vendor selling apples and oranges --/
def vendor_profit (apple_buy_price : ℚ) (apple_sell_price : ℚ) 
                  (orange_buy_price : ℚ) (orange_sell_price : ℚ) 
                  (apples_sold : ℕ) (oranges_sold : ℕ) : ℚ :=
  let apple_profit := apple_sell_price - apple_buy_price
  let orange_profit := orange_sell_price - orange_buy_price
  apple_profit * apples_sold + orange_profit * oranges_sold

theorem vendor_profit_calculation :
  let apple_buy_price : ℚ := 3 / 2  -- $3 for 2 apples
  let apple_sell_price : ℚ := 2     -- $10 for 5 apples, so $2 each
  let orange_buy_price : ℚ := 9 / 10  -- $2.70 for 3 oranges
  let orange_sell_price : ℚ := 1    -- $1 each
  let apples_sold : ℕ := 5
  let oranges_sold : ℕ := 5
  vendor_profit apple_buy_price apple_sell_price orange_buy_price orange_sell_price apples_sold oranges_sold = 3 := by
  sorry


end NUMINAMATH_CALUDE_vendor_profit_calculation_l1181_118106


namespace NUMINAMATH_CALUDE_salmon_migration_result_l1181_118197

/-- The total number of salmon in a river after migration -/
def total_salmon (initial : ℕ) (increase_factor : ℕ) : ℕ :=
  initial + initial * increase_factor

/-- Theorem: Given 500 initial salmon and a tenfold increase, the total is 5500 -/
theorem salmon_migration_result :
  total_salmon 500 10 = 5500 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_result_l1181_118197


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_square_difference_150_l1181_118143

theorem no_integer_solutions_for_square_difference_150 :
  ∀ m n : ℕ+, m ≥ n → m^2 - n^2 ≠ 150 := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_square_difference_150_l1181_118143


namespace NUMINAMATH_CALUDE_subset_intersection_theorem_l1181_118181

theorem subset_intersection_theorem (α : ℝ) (h_pos : α > 0) (h_bound : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ+) (S T : Fin p → Finset (Fin n)),
    p > α * 2^(n : ℝ) ∧
    (∀ i j : Fin p, i ≠ j → S i ≠ S j) ∧
    (∀ i j : Fin p, i ≠ j → T i ≠ T j) ∧
    (∀ i j : Fin p, (S i ∩ T j).Nonempty) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_theorem_l1181_118181


namespace NUMINAMATH_CALUDE_equation_solution_l1181_118112

theorem equation_solution :
  ∀ x y : ℝ,
  y = 3 * x →
  (5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)) ↔
  (x = 1/3 ∨ x = -2/9) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1181_118112


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1181_118186

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| := by
sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l1181_118186


namespace NUMINAMATH_CALUDE_salary_percent_increase_l1181_118174

theorem salary_percent_increase 
  (x y : ℝ) (z : ℝ) 
  (hx : x > 0) 
  (hy : y ≥ 0) 
  (hz : z = (y / x) * 100) : 
  z = (y / x) * 100 := by
sorry

end NUMINAMATH_CALUDE_salary_percent_increase_l1181_118174


namespace NUMINAMATH_CALUDE_pipe_a_fills_in_12_hours_l1181_118126

/-- Represents the time (in hours) taken by pipe A to fill the cistern -/
def pipe_a_time : ℝ := 12

/-- Represents the time (in hours) taken by pipe B to leak out the cistern -/
def pipe_b_time : ℝ := 18

/-- Represents the time (in hours) taken to fill the cistern when both pipes are open -/
def both_pipes_time : ℝ := 36

/-- Proves that pipe A fills the cistern in 12 hours given the conditions -/
theorem pipe_a_fills_in_12_hours :
  (1 / pipe_a_time) - (1 / pipe_b_time) = (1 / both_pipes_time) :=
by sorry

end NUMINAMATH_CALUDE_pipe_a_fills_in_12_hours_l1181_118126


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1181_118135

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 2 + 3^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1181_118135


namespace NUMINAMATH_CALUDE_max_cos_value_l1181_118179

theorem max_cos_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) :
  ∀ x : ℝ, Real.cos a ≤ 1 ∧ (Real.cos x ≤ Real.cos a → x = a) :=
by sorry

end NUMINAMATH_CALUDE_max_cos_value_l1181_118179


namespace NUMINAMATH_CALUDE_inverse_function_property_l1181_118146

-- Define the function f and its inverse g
variable (f g : ℝ → ℝ)

-- Define the property that g is the inverse of f
variable (h₁ : ∀ x, g (f x) = x)
variable (h₂ : ∀ x, f (g x) = x)

-- Define the given property of f
variable (h₃ : ∀ a b, f (a * b) = f a + f b)

-- Theorem to prove
theorem inverse_function_property :
  ∀ a b, g (a + b) = g a * g b :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1181_118146


namespace NUMINAMATH_CALUDE_triangle_properties_l1181_118128

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a = 2)
  (h2 : abc.c = 1)
  (h3 : Real.tan abc.A + Real.tan abc.B = -(Real.tan abc.A * Real.tan abc.B)) :
  (Real.tan (abc.A + abc.B) = 1) ∧ 
  (((2 : Real) - Real.sqrt 2) / 2 = 1/2 * abc.a * abc.b * Real.sin abc.C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1181_118128


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l1181_118159

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (a 6 + a 8 - a 5 = a 7 - (a 6 + a 8)) →  -- a_5, a_6 + a_8, a_7 form an arithmetic sequence
  q = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l1181_118159


namespace NUMINAMATH_CALUDE_count_arrangements_no_adjacent_girls_count_arrangements_AB_adjacent_l1181_118153

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 4

/-- The number of arrangements where no two girls are adjacent -/
def arrangements_no_adjacent_girls : ℕ := 144

/-- The number of arrangements where boys A and B are adjacent -/
def arrangements_AB_adjacent : ℕ := 240

/-- Theorem stating the number of arrangements where no two girls are adjacent -/
theorem count_arrangements_no_adjacent_girls :
  (num_boys.factorial * num_girls.factorial) = arrangements_no_adjacent_girls := by
  sorry

/-- Theorem stating the number of arrangements where boys A and B are adjacent -/
theorem count_arrangements_AB_adjacent :
  ((num_boys + num_girls - 1).factorial * 2) = arrangements_AB_adjacent := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_no_adjacent_girls_count_arrangements_AB_adjacent_l1181_118153


namespace NUMINAMATH_CALUDE_exists_integer_solution_l1181_118133

theorem exists_integer_solution : ∃ x : ℤ, 2 * x^2 - 3 * x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_solution_l1181_118133


namespace NUMINAMATH_CALUDE_harvey_sam_race_l1181_118109

theorem harvey_sam_race (sam_miles harvey_miles : ℕ) : 
  sam_miles = 12 → 
  harvey_miles > sam_miles → 
  sam_miles + harvey_miles = 32 → 
  harvey_miles - sam_miles = 8 := by
sorry

end NUMINAMATH_CALUDE_harvey_sam_race_l1181_118109


namespace NUMINAMATH_CALUDE_nine_digit_number_bounds_l1181_118132

theorem nine_digit_number_bounds (A B : ℕ) : 
  (∃ C b : ℕ, B = 10 * C + b ∧ b < 10 ∧ A = 10^8 * b + C) →
  B > 22222222 →
  Nat.gcd B 18 = 1 →
  A ≥ 122222224 ∧ A ≤ 999999998 :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_number_bounds_l1181_118132


namespace NUMINAMATH_CALUDE_min_red_cells_for_win_thirteen_red_cells_win_l1181_118108

/-- Represents an 8x8 grid where some cells are colored red -/
def Grid := Fin 8 → Fin 8 → Bool

/-- Returns true if the given cell is covered by the selected rows and columns -/
def isCovered (rows columns : Finset (Fin 8)) (i j : Fin 8) : Prop :=
  i ∈ rows ∨ j ∈ columns

/-- Returns the number of red cells in the grid -/
def redCount (g : Grid) : Nat :=
  (Finset.univ.filter (λ i => Finset.univ.filter (λ j => g i j) ≠ ∅)).card

/-- Returns true if there exists an uncovered red cell -/
def hasUncoveredRed (g : Grid) (rows columns : Finset (Fin 8)) : Prop :=
  ∃ i j, g i j ∧ ¬isCovered rows columns i j

theorem min_red_cells_for_win :
  ∀ n : Nat, n < 13 →
    ∃ g : Grid, redCount g = n ∧
      ∃ rows columns : Finset (Fin 8),
        rows.card = 4 ∧ columns.card = 4 ∧ ¬hasUncoveredRed g rows columns :=
by sorry

theorem thirteen_red_cells_win :
  ∃ g : Grid, redCount g = 13 ∧
    ∀ rows columns : Finset (Fin 8),
      rows.card = 4 ∧ columns.card = 4 → hasUncoveredRed g rows columns :=
by sorry

end NUMINAMATH_CALUDE_min_red_cells_for_win_thirteen_red_cells_win_l1181_118108


namespace NUMINAMATH_CALUDE_total_students_in_schools_l1181_118180

theorem total_students_in_schools (capacity1 capacity2 : ℕ) 
  (h1 : capacity1 = 400) 
  (h2 : capacity2 = 340) : 
  2 * capacity1 + 2 * capacity2 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_schools_l1181_118180


namespace NUMINAMATH_CALUDE_clock_angle_at_eight_thirty_l1181_118111

/-- The angle between clock hands at 8:30 -/
theorem clock_angle_at_eight_thirty :
  let hour_angle : ℝ := (8 * 30 + 30 / 2)
  let minute_angle : ℝ := 180
  let angle_diff : ℝ := |hour_angle - minute_angle|
  angle_diff = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_eight_thirty_l1181_118111


namespace NUMINAMATH_CALUDE_trig_equation_proof_l1181_118169

theorem trig_equation_proof (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_proof_l1181_118169


namespace NUMINAMATH_CALUDE_roll_five_dice_probability_l1181_118105

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of ways to roll an equal number of 1's and 6's -/
def equal_ones_and_sixes : ℕ := 2424

/-- The probability of rolling more 1's than 6's -/
def prob_more_ones_than_sixes : ℚ := 167 / 486

theorem roll_five_dice_probability :
  prob_more_ones_than_sixes = 1 / 2 * (1 - equal_ones_and_sixes / total_outcomes) :=
sorry

end NUMINAMATH_CALUDE_roll_five_dice_probability_l1181_118105


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1181_118119

theorem quadratic_roots_product (a b : ℝ) : 
  (a^2 + 2012*a + 1 = 0) → 
  (b^2 + 2012*b + 1 = 0) → 
  (2 + 2013*a + a^2) * (2 + 2013*b + b^2) = -2010 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l1181_118119


namespace NUMINAMATH_CALUDE_volleyball_scores_l1181_118138

/-- Volleyball competition scores -/
theorem volleyball_scores (lizzie_score : ℕ) (nathalie_score : ℕ) (aimee_score : ℕ) (team_score : ℕ) :
  lizzie_score = 4 →
  nathalie_score = lizzie_score + 3 →
  aimee_score = 2 * (lizzie_score + nathalie_score) →
  team_score = 50 →
  team_score - (lizzie_score + nathalie_score + aimee_score) = 17 := by
sorry


end NUMINAMATH_CALUDE_volleyball_scores_l1181_118138


namespace NUMINAMATH_CALUDE_age_calculation_l1181_118139

/-- Given Luke's current age and Mr. Bernard's future age relative to Luke's,
    calculate 10 years less than their average current age. -/
theorem age_calculation (luke_age : ℕ) (bernard_future_age_factor : ℕ) (years_in_future : ℕ) : 
  luke_age = 20 →
  years_in_future = 8 →
  bernard_future_age_factor = 3 →
  10 < luke_age →
  (luke_age + (bernard_future_age_factor * luke_age - years_in_future)) / 2 - 10 = 26 := by
sorry

end NUMINAMATH_CALUDE_age_calculation_l1181_118139


namespace NUMINAMATH_CALUDE_roots_equation_l1181_118172

open Real

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := x^2 - 2 * cos θ * x + 1

theorem roots_equation (θ : ℝ) (α : ℝ) 
  (h1 : f θ (sin α) = 1/4 + cos θ) 
  (h2 : f θ (cos α) = 1/4 + cos θ) : 
  (tan α)^2 + 1 / tan α = (16 + 4 * sqrt 11) / 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l1181_118172


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1181_118100

theorem sum_of_numbers : 3 + 33 + 333 + 3.33 = 372.33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1181_118100


namespace NUMINAMATH_CALUDE_egg_weight_calculation_l1181_118163

/-- Given the total weight of eggs and the number of dozens, 
    calculate the weight of a single egg. -/
theorem egg_weight_calculation 
  (total_weight : ℝ) 
  (dozens : ℕ) 
  (h1 : total_weight = 6) 
  (h2 : dozens = 8) : 
  total_weight / (dozens * 12) = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_egg_weight_calculation_l1181_118163


namespace NUMINAMATH_CALUDE_r_upper_bound_r_7_upper_bound_l1181_118168

/-- The maximum number of pieces that can be placed on an n × n chessboard
    without forming a rectangle with sides parallel to grid lines. -/
def r (n : ℕ) : ℕ := sorry

/-- Theorem: Upper bound for r(n) -/
theorem r_upper_bound (n : ℕ) : r n ≤ (n + n * Real.sqrt (4 * n - 3)) / 2 := by sorry

/-- Theorem: Upper bound for r(7) -/
theorem r_7_upper_bound : r 7 ≤ 21 := by sorry

end NUMINAMATH_CALUDE_r_upper_bound_r_7_upper_bound_l1181_118168


namespace NUMINAMATH_CALUDE_stratified_sampling_example_l1181_118110

/-- The number of ways to select students using stratified sampling -/
def stratified_sampling_selection (female_count male_count total_selected : ℕ) : ℕ :=
  let female_selected := (female_count * total_selected) / (female_count + male_count)
  let male_selected := total_selected - female_selected
  (Nat.choose female_count female_selected) * (Nat.choose male_count male_selected)

/-- Theorem: The number of ways to select 3 students from 8 female and 4 male students
    using stratified sampling by gender ratio is 112 -/
theorem stratified_sampling_example : stratified_sampling_selection 8 4 3 = 112 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_example_l1181_118110


namespace NUMINAMATH_CALUDE_day_of_week_theorem_l1181_118136

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Given a year and a day number, returns the day of the week -/
def dayOfWeek (year : ℤ) (dayNumber : ℕ) : DayOfWeek := sorry

theorem day_of_week_theorem (M : ℤ) :
  dayOfWeek M 200 = DayOfWeek.Monday →
  dayOfWeek (M + 2) 300 = DayOfWeek.Monday →
  dayOfWeek (M - 1) 100 = DayOfWeek.Tuesday :=
by sorry

end NUMINAMATH_CALUDE_day_of_week_theorem_l1181_118136


namespace NUMINAMATH_CALUDE_overlapping_strips_l1181_118117

theorem overlapping_strips (total_length width : ℝ) 
  (left_length right_length : ℝ) 
  (left_area right_area : ℝ) : 
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_length + right_length = total_length →
  left_area = 27 →
  right_area = 18 →
  (left_area + (left_length * width)) / (right_area + (right_length * width)) = left_length / right_length →
  ∃ overlap_area : ℝ, overlap_area = 13.5 ∧ 
    (left_area + overlap_area) / (right_area + overlap_area) = left_length / right_length :=
by sorry

end NUMINAMATH_CALUDE_overlapping_strips_l1181_118117


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1181_118116

theorem line_tangent_to_circle (m : ℝ) :
  (∀ x y : ℝ, x + y - m = 0 ∧ x^2 + y^2 = 2 → (∀ ε > 0, ∃ x' y' : ℝ, x' + y' - m = 0 ∧ x'^2 + y'^2 < 2 ∧ (x' - x)^2 + (y' - y)^2 < ε)) ↔
  (m > 2 ∨ m < -2) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1181_118116


namespace NUMINAMATH_CALUDE_box_height_rounding_equivalence_l1181_118129

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem box_height_rounding_equivalence :
  let height1 : ℕ := 53
  let height2 : ℕ := 78
  let correct_sum := height1 + height2
  let alice_sum := height1 + round_to_nearest_ten height2
  round_to_nearest_ten correct_sum = round_to_nearest_ten alice_sum :=
by
  sorry

end NUMINAMATH_CALUDE_box_height_rounding_equivalence_l1181_118129


namespace NUMINAMATH_CALUDE_tony_total_cost_l1181_118102

/-- Represents the total cost of Tony's purchases at the toy store -/
def total_cost (lego_price toy_sword_price play_dough_price : ℝ)
               (lego_sets toy_swords play_doughs : ℕ)
               (first_day_discount second_day_discount sales_tax : ℝ) : ℝ :=
  let first_day_cost := (2 * lego_price + 3 * toy_sword_price) * (1 - first_day_discount) * (1 + sales_tax)
  let second_day_cost := ((lego_sets - 2) * lego_price + (toy_swords - 3) * toy_sword_price + play_doughs * play_dough_price) * (1 - second_day_discount) * (1 + sales_tax)
  first_day_cost + second_day_cost

/-- Theorem stating that Tony's total cost matches the calculated amount -/
theorem tony_total_cost :
  total_cost 250 120 35 3 5 10 0.2 0.1 0.05 = 1516.20 := by
  sorry

end NUMINAMATH_CALUDE_tony_total_cost_l1181_118102


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l1181_118103

theorem isosceles_triangle_base (t α : ℝ) (h_t : t > 0) (h_α : 0 < α ∧ α < π) :
  ∃ a : ℝ, a > 0 ∧ a = 2 * Real.sqrt (t * Real.tan (α / 2)) ∧
    ∃ b : ℝ, b > 0 ∧
      let m := b * Real.cos (α / 2)
      t = (1 / 2) * a * m ∧
      α = 2 * Real.arccos (m / b) :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l1181_118103


namespace NUMINAMATH_CALUDE_product_abcd_l1181_118182

/-- Given positive real numbers a, b, c, and d satisfying the specified conditions,
    prove that their product equals 14400. -/
theorem product_abcd (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 762)
  (h_sum_ab_cd : a * b + c * d = 260)
  (h_sum_ac_bd : a * c + b * d = 365)
  (h_sum_ad_bc : a * d + b * c = 244) :
  a * b * c * d = 14400 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l1181_118182


namespace NUMINAMATH_CALUDE_subtraction_problem_l1181_118115

theorem subtraction_problem (A B : ℕ) : 
  (A ≥ 10 ∧ A ≤ 99) → 
  (B ≥ 10 ∧ B ≤ 99) → 
  A = 23 - 8 → 
  B + 7 = 18 → 
  A - B = 4 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_l1181_118115


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1181_118195

def is_valid_arrangement (perm : List Nat) : Prop :=
  perm.length = 8 ∧
  (∀ n, n ∈ perm → n ∈ [1, 2, 3, 4, 5, 6, 8, 9]) ∧
  (∀ i, i < perm.length - 1 → (10 * perm[i]! + perm[i+1]!) % 7 = 0)

theorem no_valid_arrangement : ¬∃ perm : List Nat, is_valid_arrangement perm := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l1181_118195


namespace NUMINAMATH_CALUDE_caroline_score_l1181_118176

structure Player where
  name : String
  score : ℕ

def winning_score : ℕ := 21

theorem caroline_score (caroline anthony leo : Player)
  (h1 : anthony.score = 19)
  (h2 : leo.score = 28)
  (h3 : ∃ p : Player, p ∈ [caroline, anthony, leo] ∧ p.score = winning_score) :
  caroline.score = winning_score :=
sorry

end NUMINAMATH_CALUDE_caroline_score_l1181_118176


namespace NUMINAMATH_CALUDE_initial_ratio_of_men_to_women_l1181_118134

theorem initial_ratio_of_men_to_women 
  (initial_men : ℕ) 
  (initial_women : ℕ) 
  (final_men : ℕ) 
  (final_women : ℕ) 
  (h1 : final_men = initial_men + 2)
  (h2 : final_women = 2 * (initial_women - 3))
  (h3 : final_men = 14)
  (h4 : final_women = 24) :
  initial_men / initial_women = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_initial_ratio_of_men_to_women_l1181_118134


namespace NUMINAMATH_CALUDE_egg_problem_l1181_118165

theorem egg_problem (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 100 →
  5 * x + 6 * y + 9 * z = 600 →
  (x = y ∨ y = z ∨ x = z) →
  x = 60 ∧ y = 20 := by
  sorry

end NUMINAMATH_CALUDE_egg_problem_l1181_118165


namespace NUMINAMATH_CALUDE_lakeview_academy_teachers_l1181_118162

/-- Represents the number of teachers at Lakeview Academy -/
def num_teachers (total_students : ℕ) (classes_per_student : ℕ) (class_size : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (total_students * classes_per_student * 2) / (class_size * classes_per_teacher)

/-- Theorem stating the number of teachers at Lakeview Academy -/
theorem lakeview_academy_teachers :
  num_teachers 1500 6 25 5 = 144 := by
  sorry

#eval num_teachers 1500 6 25 5

end NUMINAMATH_CALUDE_lakeview_academy_teachers_l1181_118162


namespace NUMINAMATH_CALUDE_negative_root_condition_l1181_118190

theorem negative_root_condition (p : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ x^4 - 4*p*x^3 + x^2 - 4*p*x + 1 = 0) ↔ p ≥ -3/8 := by sorry

end NUMINAMATH_CALUDE_negative_root_condition_l1181_118190


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l1181_118122

theorem binomial_expansion_example : 100 + 2 * (10 * 3) + 9 = (10 + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l1181_118122


namespace NUMINAMATH_CALUDE_acute_triangle_inequality_l1181_118155

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  R : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c
  inradius_positive : 0 < r
  circumradius_positive : 0 < R
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  acute_angles : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- State the theorem
theorem acute_triangle_inequality (t : AcuteTriangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * (t.R + t.r)^2 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_inequality_l1181_118155


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1181_118184

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 480) 
  (h2 : height = 15) : 
  area / height = 32 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1181_118184


namespace NUMINAMATH_CALUDE_andreas_living_room_area_l1181_118192

theorem andreas_living_room_area :
  ∀ (room_area : ℝ),
  (0.60 * room_area = 4 * 9) →
  room_area = 60 := by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_l1181_118192


namespace NUMINAMATH_CALUDE_distance_negative_five_to_origin_l1181_118150

theorem distance_negative_five_to_origin : 
  abs (-5 : ℝ) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_negative_five_to_origin_l1181_118150


namespace NUMINAMATH_CALUDE_intersection_point_l1181_118191

def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℚ) : Prop := -2 * y = 6 * x + 4

theorem intersection_point : 
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-12/7, 22/7) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_l1181_118191


namespace NUMINAMATH_CALUDE_total_leaked_equals_1958_l1181_118154

/-- Represents the data for an oil pipe leak -/
structure PipeLeak where
  name : String
  leakRate : ℕ  -- gallons per hour
  fixTime : ℕ  -- hours

/-- Calculates the total amount of oil leaked from a pipe during repair -/
def totalLeakedDuringRepair (pipe : PipeLeak) : ℕ :=
  pipe.leakRate * pipe.fixTime

/-- The set of all pipe leaks -/
def pipeLeaks : List PipeLeak := [
  { name := "A", leakRate := 25, fixTime := 10 },
  { name := "B", leakRate := 37, fixTime := 7 },
  { name := "C", leakRate := 55, fixTime := 12 },
  { name := "D", leakRate := 41, fixTime := 9 },
  { name := "E", leakRate := 30, fixTime := 14 }
]

/-- Calculates the total amount of oil leaked from all pipes during repair -/
def totalLeaked : ℕ :=
  (pipeLeaks.map totalLeakedDuringRepair).sum

theorem total_leaked_equals_1958 : totalLeaked = 1958 := by
  sorry

#eval totalLeaked  -- This will print the result

end NUMINAMATH_CALUDE_total_leaked_equals_1958_l1181_118154


namespace NUMINAMATH_CALUDE_project_completion_time_l1181_118198

/-- The number of days it takes A to complete the project alone -/
def a_days : ℝ := 10

/-- The number of days it takes B to complete the project alone -/
def b_days : ℝ := 30

/-- The number of days before project completion that A quits -/
def a_quit_days : ℝ := 10

/-- The total number of days to complete the project with A and B working together, with A quitting early -/
def total_days : ℝ := 15

theorem project_completion_time :
  let a_rate : ℝ := 1 / a_days
  let b_rate : ℝ := 1 / b_days
  (total_days - a_quit_days) * a_rate + total_days * b_rate = 1 :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1181_118198


namespace NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l1181_118123

theorem fraction_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 0 →
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = 2*x / (x + 2) ∧
  (2 * (-1)) / ((-1) + 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_evaluation_l1181_118123


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1181_118151

theorem polynomial_simplification (x : ℝ) : 
  x^2 * (4*x^3 - 3*x + 1) - 6*(x^3 - 3*x^2 + 4*x - 5) = 
  4*x^5 - 9*x^3 + 19*x^2 - 24*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1181_118151


namespace NUMINAMATH_CALUDE_two_digit_minus_reverse_63_l1181_118140

/-- Reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_minus_reverse_63 (n : ℕ) :
  is_two_digit n ∧ n - reverse_two_digit n = 63 → n = 81 ∨ n = 92 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_minus_reverse_63_l1181_118140


namespace NUMINAMATH_CALUDE_cone_angle_theorem_l1181_118148

/-- A cone with vertex A -/
structure Cone where
  vertexAngle : ℝ

/-- The configuration of four cones as described in the problem -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  cone4 : Cone
  cone1_eq_cone2 : cone1 = cone2
  cone3_angle : cone3.vertexAngle = π / 3
  cone4_angle : cone4.vertexAngle = 5 * π / 6
  external_tangent : True  -- Represents that cone1, cone2, and cone3 are externally tangent
  internal_tangent : True  -- Represents that cone4 is internally tangent to the other three

theorem cone_angle_theorem (config : ConeConfiguration) :
  config.cone1.vertexAngle = 2 * Real.arctan (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_cone_angle_theorem_l1181_118148


namespace NUMINAMATH_CALUDE_R_value_when_S_is_12_l1181_118183

-- Define the relationship between R and S
def R (g : ℝ) (S : ℝ) : ℝ := g * S - 6

-- State the theorem
theorem R_value_when_S_is_12 : 
  ∃ g : ℝ, (R g 6 = 12) → (R g 12 = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_R_value_when_S_is_12_l1181_118183


namespace NUMINAMATH_CALUDE_remainder_23_pow_2003_mod_7_l1181_118188

theorem remainder_23_pow_2003_mod_7 : 23^2003 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_23_pow_2003_mod_7_l1181_118188


namespace NUMINAMATH_CALUDE_claire_crafting_time_l1181_118189

/-- Represents the system of equations for Claire's time allocation --/
structure ClaireTimeSystem where
  x : ℝ
  y : ℝ
  z : ℝ
  crafting : ℝ
  tailoring : ℝ
  eq1 : (2 * y) + y + (y - 1) + crafting + crafting + 8 = 24
  eq2 : x = 2 * y
  eq3 : z = y - 1
  eq4 : crafting = tailoring
  eq5 : 2 * crafting = 9 - tailoring

/-- Theorem stating that in any valid ClaireTimeSystem, the crafting time is 3 hours --/
theorem claire_crafting_time (s : ClaireTimeSystem) : s.crafting = 3 := by
  sorry

end NUMINAMATH_CALUDE_claire_crafting_time_l1181_118189


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_l1181_118157

/-- The parabola defined by y^2 = 8x -/
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line y = (1/4)x - 2 -/
def SymmetryLine (x y : ℝ) : Prop := y = (1/4)*x - 2

/-- Two points are symmetric about a line if their midpoint lies on that line -/
def SymmetricPoints (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  SymmetryLine ((x₁ + x₂)/2) ((y₁ + y₂)/2)

/-- The equation of line AB: 4x + y - 15 = 0 -/
def LineAB (x y : ℝ) : Prop := 4*x + y - 15 = 0

theorem parabola_symmetric_points (x₁ y₁ x₂ y₂ : ℝ) :
  Parabola x₁ y₁ → Parabola x₂ y₂ → SymmetricPoints x₁ y₁ x₂ y₂ →
  ∀ x y, LineAB x y ↔ (y - y₁)/(x - x₁) = (y₂ - y)/(x₂ - x) :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_l1181_118157


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1181_118107

theorem arithmetic_mean_of_fractions : 
  (1/3 : ℚ) * ((3/4 : ℚ) + (5/6 : ℚ) + (9/10 : ℚ)) = 149/180 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1181_118107


namespace NUMINAMATH_CALUDE_composite_sequences_exist_l1181_118130

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def consecutive_composites (start : ℕ) (len : ℕ) : Prop :=
  ∀ i, i ∈ Finset.range len → is_composite (start + i)

theorem composite_sequences_exist :
  (∃ start : ℕ, start ≤ 500 - 9 + 1 ∧ consecutive_composites start 9) ∧
  (∃ start : ℕ, start ≤ 500 - 11 + 1 ∧ consecutive_composites start 11) :=
sorry

end NUMINAMATH_CALUDE_composite_sequences_exist_l1181_118130


namespace NUMINAMATH_CALUDE_function_property_l1181_118101

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

theorem function_property (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1181_118101


namespace NUMINAMATH_CALUDE_non_working_games_l1181_118144

theorem non_working_games (total : ℕ) (working : ℕ) (h1 : total = 30) (h2 : working = 17) :
  total - working = 13 := by
  sorry

end NUMINAMATH_CALUDE_non_working_games_l1181_118144


namespace NUMINAMATH_CALUDE_correct_weighted_mean_l1181_118193

def total_values : ℕ := 30
def incorrect_mean : ℝ := 150
def first_error : ℝ := 135 - 165
def second_error : ℝ := 170 - 200
def weight_first_half : ℝ := 2
def weight_second_half : ℝ := 3

theorem correct_weighted_mean :
  let original_sum := incorrect_mean * total_values
  let total_error := first_error + second_error
  let corrected_sum := original_sum - total_error
  let total_weight := weight_first_half * (total_values / 2) + weight_second_half * (total_values / 2)
  corrected_sum / total_weight = 59.2 := by sorry

end NUMINAMATH_CALUDE_correct_weighted_mean_l1181_118193


namespace NUMINAMATH_CALUDE_rectangle_width_l1181_118147

theorem rectangle_width (w : ℝ) (l : ℝ) (P : ℝ) : 
  l = 2 * w + 6 →  -- length is 6 more than twice the width
  P = 2 * l + 2 * w →  -- perimeter formula
  P = 120 →  -- given perimeter
  w = 18 :=  -- width to prove
by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1181_118147


namespace NUMINAMATH_CALUDE_coinciding_vertices_l1181_118152

/-- A point in the plane -/
structure Point :=
  (x y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- An isosceles right triangle defined by two points of the quadrilateral and a third point -/
structure IsoscelesRightTriangle :=
  (P Q R : Point)

/-- Predicate to check if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if two points coincide -/
def coincide (P Q : Point) : Prop := P.x = Q.x ∧ P.y = Q.y

/-- Theorem: If O₁ and O₃ coincide, then O₂ and O₄ coincide -/
theorem coinciding_vertices 
  (q : Quadrilateral) 
  (t1 : IsoscelesRightTriangle) 
  (t2 : IsoscelesRightTriangle) 
  (t3 : IsoscelesRightTriangle) 
  (t4 : IsoscelesRightTriangle) 
  (h1 : is_convex q)
  (h2 : t1.P = q.A ∧ t1.Q = q.B)
  (h3 : t2.P = q.B ∧ t2.Q = q.C)
  (h4 : t3.P = q.C ∧ t3.Q = q.D)
  (h5 : t4.P = q.D ∧ t4.Q = q.A)
  (h6 : coincide t1.R t3.R) :
  coincide t2.R t4.R := by sorry

end NUMINAMATH_CALUDE_coinciding_vertices_l1181_118152


namespace NUMINAMATH_CALUDE_triangle_area_cosine_sum_maximum_l1181_118127

theorem triangle_area_cosine_sum_maximum (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = Real.sqrt 3 →
  a^2 = b^2 + c^2 + b*c →
  S = (1/2) * a * b * Real.sin C →
  (∃ (k : ℝ), S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ k) →
  (∀ (k : ℝ), S + Real.sqrt 3 * Real.cos B * Real.cos C ≤ k → k ≥ Real.sqrt 3) :=
by sorry

#check triangle_area_cosine_sum_maximum

end NUMINAMATH_CALUDE_triangle_area_cosine_sum_maximum_l1181_118127


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_to_one_l1181_118124

theorem quadratic_roots_sum_to_one (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x + y = 1 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ↔ b = -a :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_to_one_l1181_118124


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_between_400_and_700_l1181_118199

def first_term : ℕ := 401
def last_term : ℕ := 699
def common_difference : ℕ := 2

def number_of_terms : ℕ := (last_term - first_term) / common_difference + 1

theorem sum_of_odd_integers_between_400_and_700 :
  (number_of_terms : ℝ) / 2 * (first_term + last_term : ℝ) = 82500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_between_400_and_700_l1181_118199


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1181_118156

theorem quadratic_factorization : ∀ x : ℝ, 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1181_118156


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1181_118113

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 = 2*x} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1181_118113


namespace NUMINAMATH_CALUDE_part_one_part_two_l1181_118166

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) :
  let a := 2
  f a x ≥ 7 - |x - 1| ↔ x ∈ Set.Iic (-2) ∪ Set.Ici 5 := by sorry

-- Part II
theorem part_two (m n : ℝ) (h1 : m > 0) (h2 : n > 0) :
  (∀ x, f 1 x ≤ 1 ↔ x ∈ Set.Icc 0 2) →
  m^2 + 2*n^2 = 1 →
  m + 4*n ≤ 3 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m^2 + 2*n^2 = 1 ∧ m + 4*n = 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1181_118166


namespace NUMINAMATH_CALUDE_c1_minus_c4_equals_9_l1181_118149

def f (c1 c2 c3 c4 x : ℕ) : ℕ := 
  (x^2 - 8*x + c1) * (x^2 - 8*x + c2) * (x^2 - 8*x + c3) * (x^2 - 8*x + c4)

theorem c1_minus_c4_equals_9 
  (c1 c2 c3 c4 : ℕ) 
  (h1 : c1 ≥ c2) 
  (h2 : c2 ≥ c3) 
  (h3 : c3 ≥ c4)
  (h4 : ∃ (M : Finset ℕ), M.card = 7 ∧ ∀ x ∈ M, f c1 c2 c3 c4 x = 0) :
  c1 - c4 = 9 := by
sorry

end NUMINAMATH_CALUDE_c1_minus_c4_equals_9_l1181_118149


namespace NUMINAMATH_CALUDE_odd_prime_condition_l1181_118185

theorem odd_prime_condition (p : ℕ) : 
  (Prime p ∧ Odd p) →
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 → Prime (1 + k * (p - 1))) →
  p = 3 ∨ p = 7 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_condition_l1181_118185


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1181_118164

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_third : a 3 = 2 * a 1 + a 2)
  (h_exist : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l1181_118164


namespace NUMINAMATH_CALUDE_series_convergence_power_l1181_118104

theorem series_convergence_power (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) 
  (h_conv : Summable a) :
  Summable (fun n => (a n) ^ (n / (n + 1))) := by
sorry

end NUMINAMATH_CALUDE_series_convergence_power_l1181_118104


namespace NUMINAMATH_CALUDE_dilin_gave_sword_l1181_118141

-- Define the types for individuals and gifts
inductive Individual : Type
| Ilse : Individual
| Elsa : Individual
| Bilin : Individual
| Dilin : Individual

inductive Gift : Type
| Sword : Gift
| Necklace : Gift

-- Define the type for statements
inductive Statement : Type
| GiftWasSword : Statement
| IDidNotGive : Statement
| IlseGaveNecklace : Statement
| BilinGaveSword : Statement

-- Define a function to determine if an individual is an elf
def isElf (i : Individual) : Prop :=
  i = Individual.Ilse ∨ i = Individual.Elsa

-- Define a function to determine if an individual is a dwarf
def isDwarf (i : Individual) : Prop :=
  i = Individual.Bilin ∨ i = Individual.Dilin

-- Define the truth value of a statement given who made it and who gave the gift
def isTruthful (speaker : Individual) (giver : Individual) (gift : Gift) (s : Statement) : Prop :=
  match s with
  | Statement.GiftWasSword => gift = Gift.Sword
  | Statement.IDidNotGive => speaker ≠ giver
  | Statement.IlseGaveNecklace => giver = Individual.Ilse ∧ gift = Gift.Necklace
  | Statement.BilinGaveSword => giver = Individual.Bilin ∧ gift = Gift.Sword

-- Define the conditions of truthfulness based on the problem statement
def meetsConditions (speaker : Individual) (giver : Individual) (gift : Gift) (s : Statement) : Prop :=
  (isElf speaker ∧ isDwarf giver → ¬isTruthful speaker giver gift s) ∧
  (isDwarf speaker ∧ (s = Statement.IDidNotGive ∨ s = Statement.IlseGaveNecklace) → ¬isTruthful speaker giver gift s) ∧
  (¬(isElf speaker ∧ isDwarf giver) ∧ ¬(isDwarf speaker ∧ (s = Statement.IDidNotGive ∨ s = Statement.IlseGaveNecklace)) → isTruthful speaker giver gift s)

-- The theorem to be proved
theorem dilin_gave_sword :
  ∃ (speakers : Fin 4 → Individual),
    (∃ (statements : Fin 4 → Statement),
      (∀ i : Fin 4, meetsConditions (speakers i) Individual.Dilin Gift.Sword (statements i)) ∧
      (∃ i : Fin 4, statements i = Statement.GiftWasSword) ∧
      (∃ i : Fin 4, statements i = Statement.IDidNotGive) ∧
      (∃ i : Fin 4, statements i = Statement.IlseGaveNecklace) ∧
      (∃ i : Fin 4, statements i = Statement.BilinGaveSword)) :=
sorry

end NUMINAMATH_CALUDE_dilin_gave_sword_l1181_118141


namespace NUMINAMATH_CALUDE_difference_number_and_three_fifths_l1181_118170

theorem difference_number_and_three_fifths (n : ℚ) : n = 160 → n - (3 / 5 * n) = 64 := by
  sorry

end NUMINAMATH_CALUDE_difference_number_and_three_fifths_l1181_118170


namespace NUMINAMATH_CALUDE_shooting_scores_theorem_l1181_118145

def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

theorem shooting_scores_theorem :
  let avg_A := (scores_A.sum) / (scores_A.length : ℝ)
  let avg_B := (scores_B.sum) / (scores_B.length : ℝ)
  let avg_total := ((scores_A ++ scores_B).sum) / ((scores_A ++ scores_B).length : ℝ)
  (avg_A < avg_B) ∧ (avg_total = 6.6) := by
  sorry

end NUMINAMATH_CALUDE_shooting_scores_theorem_l1181_118145


namespace NUMINAMATH_CALUDE_reverse_digits_problem_l1181_118121

/-- Given two two-digit numbers where the second is the reverse of the first,
    if their quotient is 1.75 and the product of the first with its tens digit
    is 3.5 times the second, then the numbers are 21 and 12. -/
theorem reverse_digits_problem (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧  -- x is a two-digit number
  10 ≤ y ∧ y < 100 ∧  -- y is a two-digit number
  y = (x % 10) * 10 + (x / 10) ∧  -- y is the reverse of x
  (x : ℚ) / y = 1.75 ∧  -- their quotient is 1.75
  x * (x / 10) = (7 * y) / 2  -- product of x and its tens digit is 3.5 times y
  → x = 21 ∧ y = 12 := by
sorry

end NUMINAMATH_CALUDE_reverse_digits_problem_l1181_118121


namespace NUMINAMATH_CALUDE_cone_base_radius_l1181_118142

theorem cone_base_radius (surface_area : ℝ) (r : ℝ) : 
  surface_area = 12 * Real.pi ∧ 
  (∃ l : ℝ, l = 2 * r ∧ surface_area = Real.pi * r^2 + Real.pi * r * l) → 
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1181_118142


namespace NUMINAMATH_CALUDE_function_value_at_half_l1181_118118

theorem function_value_at_half (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc 0 (π / 2), f (Real.sin x) = x) →
  f (1 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_half_l1181_118118


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1181_118125

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | x ≥ 3}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1181_118125


namespace NUMINAMATH_CALUDE_major_axis_length_for_given_cylinder_l1181_118194

/-- The length of the major axis of an ellipse formed by cutting a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis of an ellipse formed by cutting a right circular cylinder
    with radius 2 is 5.6, given that the major axis is 40% longer than the minor axis --/
theorem major_axis_length_for_given_cylinder :
  major_axis_length 2 1.4 = 5.6 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_for_given_cylinder_l1181_118194


namespace NUMINAMATH_CALUDE_sandy_age_multiple_is_ten_l1181_118160

/-- The multiple of Sandy's age that equals her monthly phone bill expense -/
def sandy_age_multiple : ℕ → ℕ → ℕ → ℕ
| kim_age, sandy_future_age, sandy_expense =>
  let sandy_current_age := sandy_future_age - 2
  sandy_expense / sandy_current_age

/-- Theorem stating the multiple of Sandy's age that equals her monthly phone bill expense -/
theorem sandy_age_multiple_is_ten :
  sandy_age_multiple 10 36 340 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sandy_age_multiple_is_ten_l1181_118160


namespace NUMINAMATH_CALUDE_mathematics_arrangements_l1181_118173

def word : String := "MATHEMATICS"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowel_count (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def consonant_count (s : String) : Nat :=
  s.length - vowel_count s

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def multiset_permutations (total : Nat) (duplicates : List Nat) : Nat :=
  factorial total / (duplicates.map factorial |>.prod)

theorem mathematics_arrangements :
  let vowels := vowel_count word
  let consonants := consonant_count word
  let vowel_arrangements := multiset_permutations vowels [2]
  let consonant_arrangements := multiset_permutations consonants [2, 2]
  vowel_arrangements * consonant_arrangements = 15120 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_arrangements_l1181_118173


namespace NUMINAMATH_CALUDE_jellybean_problem_l1181_118137

theorem jellybean_problem (initial_bags : ℕ) (initial_average : ℕ) (average_increase : ℕ) :
  initial_bags = 34 →
  initial_average = 117 →
  average_increase = 7 →
  let total_initial := initial_bags * initial_average
  let new_average := initial_average + average_increase
  let total_new := (initial_bags + 1) * new_average
  total_new - total_initial = 362 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1181_118137


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1181_118178

theorem arithmetic_equality : 4 * 5 + 5 * 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1181_118178


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1181_118187

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1181_118187
