import Mathlib

namespace NUMINAMATH_CALUDE_hash_twelve_six_l241_24159

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the theorem
theorem hash_twelve_six :
  (∀ r s : ℝ, hash r 0 = r) →
  (∀ r s : ℝ, hash r s = hash s r) →
  (∀ r s : ℝ, hash (r + 2) s = hash r s + 2 * s + 2) →
  hash 12 6 = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_hash_twelve_six_l241_24159


namespace NUMINAMATH_CALUDE_train_length_calculation_l241_24136

/-- The length of a train given specific conditions -/
theorem train_length_calculation (crossing_time : Real) (man_speed : Real) (train_speed : Real) :
  let relative_speed := (train_speed - man_speed) * (5 / 18)
  let train_length := relative_speed * crossing_time
  crossing_time = 35.99712023038157 ∧ 
  man_speed = 3 ∧ 
  train_speed = 63 →
  ∃ ε > 0, |train_length - 600| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l241_24136


namespace NUMINAMATH_CALUDE_distinct_sentences_count_l241_24135

/-- Represents the number of variations for each phrase -/
def phrase_variations : Fin 4 → ℕ
  | 0 => 3  -- Phrase I
  | 1 => 2  -- Phrase II
  | 2 => 1  -- Phrase III (mandatory)
  | 3 => 2  -- Phrase IV

/-- Calculates the total number of combinations -/
def total_combinations : ℕ := 
  (phrase_variations 0) * (phrase_variations 1) * (phrase_variations 2) * (phrase_variations 3)

/-- The number of distinct meaningful sentences -/
def distinct_sentences : ℕ := total_combinations - 1

theorem distinct_sentences_count : distinct_sentences = 23 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sentences_count_l241_24135


namespace NUMINAMATH_CALUDE_initial_eggs_count_l241_24116

theorem initial_eggs_count (eggs_used : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  eggs_used = 5 → chickens = 2 → eggs_per_chicken = 3 → final_eggs = 11 →
  ∃ initial_eggs : ℕ, initial_eggs = 10 ∧ initial_eggs - eggs_used + chickens * eggs_per_chicken = final_eggs :=
by
  sorry


end NUMINAMATH_CALUDE_initial_eggs_count_l241_24116


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l241_24131

-- First expression
theorem simplify_expression_1 : 
  (Real.sqrt 12 + Real.sqrt 20) - (3 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 3 := by
  sorry

-- Second expression
theorem simplify_expression_2 : 
  Real.sqrt 8 * Real.sqrt 6 - 3 * Real.sqrt 6 + Real.sqrt 2 = 4 * Real.sqrt 3 - 3 * Real.sqrt 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l241_24131


namespace NUMINAMATH_CALUDE_exists_x_satisfying_inequality_l241_24144

theorem exists_x_satisfying_inequality (a b c : ℝ) 
  (h : ∃ (x y : ℝ), (x = a ∧ y = b) ∨ (x = a ∧ y = c) ∨ (x = b ∧ y = c) ∧ |x - y| > 1 / (2 * Real.sqrt 2)) :
  ∃ (x : ℤ), x^2 - 4*(a+b+c)*x + 12*(a*b+b*c+c*a) < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_satisfying_inequality_l241_24144


namespace NUMINAMATH_CALUDE_activity_popularity_order_l241_24145

def soccer_popularity : ℚ := 13/40
def swimming_popularity : ℚ := 9/24
def baseball_popularity : ℚ := 11/30
def hiking_popularity : ℚ := 3/10

def activity_order : List String := ["Swimming", "Baseball", "Soccer", "Hiking"]

theorem activity_popularity_order :
  swimming_popularity > baseball_popularity ∧
  baseball_popularity > soccer_popularity ∧
  soccer_popularity > hiking_popularity :=
by sorry

end NUMINAMATH_CALUDE_activity_popularity_order_l241_24145


namespace NUMINAMATH_CALUDE_range_of_a_l241_24104

def set_A : Set ℝ := {x : ℝ | (3 * x) / (x + 1) ≤ 2}

def set_B (a : ℝ) : Set ℝ := {x : ℝ | a - 2 < x ∧ x < 2 * a + 1}

theorem range_of_a (a : ℝ) :
  set_A = set_B a → a ∈ Set.Ioo (1/2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l241_24104


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l241_24176

theorem right_triangle_hypotenuse : 
  ∀ (short_leg long_leg hypotenuse : ℝ),
  short_leg > 0 →
  long_leg = 2 * short_leg + 3 →
  (1 / 2) * short_leg * long_leg = 84 →
  short_leg ^ 2 + long_leg ^ 2 = hypotenuse ^ 2 →
  hypotenuse = Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l241_24176


namespace NUMINAMATH_CALUDE_ice_chests_filled_example_l241_24160

/-- Given an ice machine with a total number of ice cubes and a fixed number of ice cubes per chest,
    calculate the number of ice chests that can be filled. -/
def ice_chests_filled (total_ice_cubes : ℕ) (ice_cubes_per_chest : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_chest

/-- Prove that with 294 ice cubes in total and 42 ice cubes per chest, 7 ice chests can be filled. -/
theorem ice_chests_filled_example : ice_chests_filled 294 42 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_chests_filled_example_l241_24160


namespace NUMINAMATH_CALUDE_power_two_half_equals_two_l241_24147

theorem power_two_half_equals_two : 2^(2/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_half_equals_two_l241_24147


namespace NUMINAMATH_CALUDE_square_impossibility_l241_24185

theorem square_impossibility (n : ℕ) : n^2 = 24 → False := by
  sorry

end NUMINAMATH_CALUDE_square_impossibility_l241_24185


namespace NUMINAMATH_CALUDE_books_sold_l241_24181

theorem books_sold (initial_books remaining_books : ℕ) 
  (h1 : initial_books = 136) 
  (h2 : remaining_books = 27) : 
  initial_books - remaining_books = 109 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l241_24181


namespace NUMINAMATH_CALUDE_prime_plus_three_prime_l241_24153

theorem prime_plus_three_prime (p : ℕ) (hp : Nat.Prime p) (hp3 : Nat.Prime (p + 3)) :
  p^11 - 52 = 1996 := by
  sorry

end NUMINAMATH_CALUDE_prime_plus_three_prime_l241_24153


namespace NUMINAMATH_CALUDE_greatest_x_cube_less_than_2000_l241_24172

theorem greatest_x_cube_less_than_2000 :
  ∃ (x : ℕ), x > 0 ∧ ∃ (k : ℕ), x = 5 * k ∧ x^3 < 2000 ∧
  ∀ (y : ℕ), y > 0 → (∃ (m : ℕ), y = 5 * m) → y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_x_cube_less_than_2000_l241_24172


namespace NUMINAMATH_CALUDE_min_value_theorem_l241_24127

theorem min_value_theorem (x y a : ℝ) 
  (h1 : (x - 3)^3 + 2016 * (x - 3) = a) 
  (h2 : (2 * y - 3)^3 + 2016 * (2 * y - 3) = -a) : 
  ∃ (m : ℝ), m = 28 ∧ ∀ (x y : ℝ), x^2 + 4 * y^2 + 4 * x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l241_24127


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_i_l241_24151

theorem complex_sum_of_powers_i (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_i_l241_24151


namespace NUMINAMATH_CALUDE_three_color_plane_coloring_l241_24112

-- Define a type for colors
inductive Color
| Red
| Green
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for lines in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a predicate to check if a point is on a line
def IsOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a predicate to check if a line contains at most two colors
def LineContainsAtMostTwoColors (coloring : Coloring) (l : Line) : Prop :=
  ∃ (c1 c2 : Color), ∀ (p : Point), IsOnLine p l → coloring p = c1 ∨ coloring p = c2

-- Define a predicate to check if all three colors are used
def AllColorsUsed (coloring : Coloring) : Prop :=
  (∃ (p : Point), coloring p = Color.Red) ∧
  (∃ (p : Point), coloring p = Color.Green) ∧
  (∃ (p : Point), coloring p = Color.Blue)

-- Theorem statement
theorem three_color_plane_coloring :
  ∃ (coloring : Coloring),
    (∀ (l : Line), LineContainsAtMostTwoColors coloring l) ∧
    AllColorsUsed coloring :=
by
  sorry

end NUMINAMATH_CALUDE_three_color_plane_coloring_l241_24112


namespace NUMINAMATH_CALUDE_boat_animals_correct_number_of_dogs_l241_24194

theorem boat_animals (sheep_initial : ℕ) (cows_initial : ℕ) (sheep_drowned : ℕ) (animals_survived : ℕ) : ℕ :=
  let cows_drowned := 2 * sheep_drowned
  let sheep_survived := sheep_initial - sheep_drowned
  let cows_survived := cows_initial - cows_drowned
  let dogs := animals_survived - sheep_survived - cows_survived
  dogs

theorem correct_number_of_dogs : 
  boat_animals 20 10 3 35 = 14 := by
  sorry

end NUMINAMATH_CALUDE_boat_animals_correct_number_of_dogs_l241_24194


namespace NUMINAMATH_CALUDE_line_relationships_l241_24141

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationships
  (α β : Plane) (a b l : Line)
  (h1 : intersect α β = l)
  (h2 : subset a α)
  (h3 : subset b β)
  (h4 : ¬ perp α β)
  (h5 : ¬ perp_line a l)
  (h6 : ¬ perp_line b l) :
  (∃ (a' b' : Line), parallel a' b' ∧ a' = a ∧ b' = b) ∧
  (∃ (a'' b'' : Line), perp_line a'' b'' ∧ a'' = a ∧ b'' = b) :=
sorry

end NUMINAMATH_CALUDE_line_relationships_l241_24141


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l241_24155

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l241_24155


namespace NUMINAMATH_CALUDE_no_solution_for_four_l241_24173

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def three_digit_number (x y : ℕ) : ℕ :=
  100 * x + 30 + y

theorem no_solution_for_four :
  ∀ y : ℕ, y < 10 →
    ¬(is_divisible_by_11 (three_digit_number 4 y)) ∧
    (∀ x : ℕ, x < 10 → x ≠ 4 →
      ∃ y : ℕ, y < 10 ∧ is_divisible_by_11 (three_digit_number x y)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_four_l241_24173


namespace NUMINAMATH_CALUDE_product_sequence_sum_l241_24177

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 12) (h2 : b = a - 1) : a + b = 71 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l241_24177


namespace NUMINAMATH_CALUDE_victors_weekly_earnings_l241_24171

/-- Calculates the total earnings for a week given an hourly wage and hours worked each day -/
def weeklyEarnings (hourlyWage : ℕ) (hoursWorked : List ℕ) : ℕ :=
  hourlyWage * (hoursWorked.sum)

/-- Theorem: Victor's weekly earnings -/
theorem victors_weekly_earnings :
  let hourlyWage : ℕ := 12
  let hoursWorked : List ℕ := [5, 6, 7, 4, 8]
  weeklyEarnings hourlyWage hoursWorked = 360 := by
  sorry

end NUMINAMATH_CALUDE_victors_weekly_earnings_l241_24171


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l241_24178

theorem half_abs_diff_squares_21_19 : (1 / 2 : ℝ) * |21^2 - 19^2| = 40 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l241_24178


namespace NUMINAMATH_CALUDE_unique_x_with_square_conditions_l241_24137

theorem unique_x_with_square_conditions : ∃! (x : ℕ), 
  x > 0 ∧ 
  (∃ (n : ℕ), 2 * x + 1 = n^2) ∧ 
  (∀ (k : ℕ), (2 * x + 2 ≤ k) ∧ (k ≤ 3 * x + 2) → ¬∃ (m : ℕ), k = m^2) ∧
  x = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_square_conditions_l241_24137


namespace NUMINAMATH_CALUDE_min_team_a_size_l241_24199

theorem min_team_a_size (a b : ℕ) : 
  (∃ c : ℕ, 2 * (a - 90) = b + 90 ∧ a + c = 6 * (b - c)) →
  a ≥ 153 :=
sorry

end NUMINAMATH_CALUDE_min_team_a_size_l241_24199


namespace NUMINAMATH_CALUDE_cost_per_minute_is_twelve_cents_l241_24180

/-- Calculates the cost per minute for a phone service -/
def costPerMinute (monthlyFee : ℚ) (totalBill : ℚ) (minutesUsed : ℕ) : ℚ :=
  (totalBill - monthlyFee) / minutesUsed

/-- Proof that the cost per minute is $0.12 given the specified conditions -/
theorem cost_per_minute_is_twelve_cents :
  let monthlyFee : ℚ := 2
  let totalBill : ℚ := 23.36
  let minutesUsed : ℕ := 178
  costPerMinute monthlyFee totalBill minutesUsed = 0.12 := by
  sorry

#eval costPerMinute 2 23.36 178

end NUMINAMATH_CALUDE_cost_per_minute_is_twelve_cents_l241_24180


namespace NUMINAMATH_CALUDE_inequality_solution_set_l241_24150

-- Define the condition that x^2 - 2ax + a > 0 holds for all real x
def always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*a*x + a > 0

-- Define the solution set
def solution_set (t : ℝ) : Prop :=
  t < -3 ∨ t > 1

-- State the theorem
theorem inequality_solution_set :
  ∀ a : ℝ, always_positive a →
    (∀ t : ℝ, a^(t^2) + 2*t - 3 < 1 ↔ solution_set t) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l241_24150


namespace NUMINAMATH_CALUDE_derivative_f_at_sqrt2_over_2_l241_24143

noncomputable def f (x : ℝ) := x^3 - 3*x + 1

theorem derivative_f_at_sqrt2_over_2 :
  deriv f (Real.sqrt 2 / 2) = -3/2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_sqrt2_over_2_l241_24143


namespace NUMINAMATH_CALUDE_regression_line_unit_change_l241_24126

/-- Regression line equation parameters -/
structure RegressionParams where
  slope : ℝ
  intercept : ℝ

/-- Change in y for a unit change in x -/
def change_in_y (params : RegressionParams) : ℝ :=
  params.slope

/-- The main theorem: for the given regression line, 
    the change in y for a unit change in x is 0.254 -/
theorem regression_line_unit_change 
  (params : RegressionParams) 
  (h : params = ⟨0.254, 0.321⟩) : 
  change_in_y params = 0.254 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_unit_change_l241_24126


namespace NUMINAMATH_CALUDE_angle_C_measure_l241_24123

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem angle_C_measure (t : Triangle) : 
  t.A = 39 * π / 180 ∧ 
  (t.a^2 - t.b^2) * (t.a^2 + t.a * t.c - t.b^2) = t.b^2 * t.c^2 ∧
  t.A + t.B + t.C = π →
  t.C = 115 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l241_24123


namespace NUMINAMATH_CALUDE_min_value_a_l241_24124

theorem min_value_a (x y z : ℝ) (h : x^2 + 4*y^2 + z^2 = 6) :
  (∃ (a : ℝ), ∀ (x y z : ℝ), x^2 + 4*y^2 + z^2 = 6 → x + 2*y + 3*z ≤ a) ∧
  (∀ (b : ℝ), (∀ (x y z : ℝ), x^2 + 4*y^2 + z^2 = 6 → x + 2*y + 3*z ≤ b) → Real.sqrt 66 ≤ b) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l241_24124


namespace NUMINAMATH_CALUDE_min_floor_sum_l241_24103

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (a₀ b₀ c₀ : ℝ) (ha₀ : a₀ > 0) (hb₀ : b₀ > 0) (hc₀ : c₀ > 0),
    (⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(c + 2 * a) / b⌋ ≥ 9) ∧
    (⌊(2 * a₀ + b₀) / c₀⌋ + ⌊(b₀ + 2 * c₀) / a₀⌋ + ⌊(c₀ + 2 * a₀) / b₀⌋ = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l241_24103


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l241_24130

def f (m n : ℕ) : ℕ := m * n

theorem f_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := by
sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l241_24130


namespace NUMINAMATH_CALUDE_store_revenue_l241_24101

theorem store_revenue (december : ℝ) (h1 : december > 0) : 
  let november := (2 / 5 : ℝ) * december
  let january := (1 / 3 : ℝ) * november
  let average := (november + january) / 2
  december = 5 * average := by
sorry

end NUMINAMATH_CALUDE_store_revenue_l241_24101


namespace NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l241_24191

theorem power_of_five_mod_ten_thousand : 5^2023 % 10000 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_ten_thousand_l241_24191


namespace NUMINAMATH_CALUDE_line_through_point_l241_24146

/-- Given a line equation bx + (b-1)y = b+3 that passes through the point (3, -7), prove that b = 4/5 -/
theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 1) * (-7) = b + 3) → b = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l241_24146


namespace NUMINAMATH_CALUDE_no_integer_solutions_l241_24142

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^3 + 4*x^2 + x = 18*y^3 + 18*y^2 + 6*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l241_24142


namespace NUMINAMATH_CALUDE_range_of_p_l241_24195

/-- The function p(x) = x^4 - 4x^2 + 4 -/
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 4

/-- The theorem stating that the range of p(x) over [0, ∞) is [0, ∞) -/
theorem range_of_p :
  ∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, x ≥ 0 ∧ p x = y :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_l241_24195


namespace NUMINAMATH_CALUDE_square_side_length_l241_24138

theorem square_side_length (side : ℝ) : 
  (5 * side) * (side / 2) = 160 → side = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l241_24138


namespace NUMINAMATH_CALUDE_remainder_problem_l241_24156

theorem remainder_problem (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l241_24156


namespace NUMINAMATH_CALUDE_batsman_average_after_11th_inning_l241_24109

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalScore : ℕ
  averageScore : ℚ

/-- Calculates the new average score after an inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalScore + newInningScore : ℚ) / (stats.inningsPlayed + 1 : ℚ)

/-- Theorem: Given the conditions, the batsman's average after the 11th inning is 45 -/
theorem batsman_average_after_11th_inning
  (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 10)
  (h2 : newAverage stats 95 = stats.averageScore + 5) :
  newAverage stats 95 = 45 := by
  sorry

#check batsman_average_after_11th_inning

end NUMINAMATH_CALUDE_batsman_average_after_11th_inning_l241_24109


namespace NUMINAMATH_CALUDE_oscar_cd_distribution_l241_24154

/-- Represents the number of CDs Oscar can pack in each box -/
def max_cds_per_box : ℕ := 2

/-- Represents the number of rock CDs Oscar needs to ship -/
def rock_cds : ℕ := 14

/-- Represents the number of pop CDs Oscar needs to ship -/
def pop_cds : ℕ := 8

/-- Theorem stating that for any non-negative integer n, if Oscar ships 2n classical CDs
    along with the rock and pop CDs, the total number of CDs can be evenly distributed
    into boxes of 2 CDs each -/
theorem oscar_cd_distribution (n : ℕ) :
  ∃ (total_boxes : ℕ), (rock_cds + 2*n + pop_cds) = max_cds_per_box * total_boxes :=
sorry

end NUMINAMATH_CALUDE_oscar_cd_distribution_l241_24154


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l241_24162

theorem stratified_sampling_problem (total_population : ℕ) (first_stratum : ℕ) (sample_first_stratum : ℕ) (total_sample : ℕ) :
  total_population = 1500 →
  first_stratum = 700 →
  sample_first_stratum = 14 →
  (sample_first_stratum : ℚ) / total_sample = (first_stratum : ℚ) / total_population →
  total_sample = 30 :=
by
  sorry

#check stratified_sampling_problem

end NUMINAMATH_CALUDE_stratified_sampling_problem_l241_24162


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l241_24133

theorem cot_thirty_degrees : 
  let cos_thirty : ℝ := Real.sqrt 3 / 2
  let sin_thirty : ℝ := 1 / 2
  let cot (θ : ℝ) : ℝ := (Real.cos θ) / (Real.sin θ)
  cot (30 * π / 180) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l241_24133


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l241_24193

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 1 = 1 →
  (∀ n, S n = (a 1 - a (n + 1) * (a 2 / a 1)^n) / (1 - a 2 / a 1)) →
  1 / a 1 - 1 / a 2 = 2 / a 3 →
  S 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l241_24193


namespace NUMINAMATH_CALUDE_sum_of_squared_digits_l241_24169

/-- The number of digits in 222222222 -/
def n : ℕ := 9

/-- The number whose square we're considering -/
def num : ℕ := 222222222

/-- Function to calculate the sum of digits of a number -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem sum_of_squared_digits : sum_of_digits (num ^ 2) = 162 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_digits_l241_24169


namespace NUMINAMATH_CALUDE_comprehensive_survey_suitable_for_grade_8_1_l241_24152

/-- Represents a type of survey -/
inductive SurveyType
| Sampling
| Comprehensive

/-- Represents a population to be surveyed -/
structure Population where
  size : ℕ
  accessibility : Bool
  variability : Bool

/-- Determines if a survey type is suitable for a given population -/
def is_suitable (st : SurveyType) (p : Population) : Prop :=
  match st with
  | SurveyType.Sampling => p.size > 1000 ∨ p.accessibility = false ∨ p.variability = true
  | SurveyType.Comprehensive => p.size ≤ 1000 ∧ p.accessibility = true ∧ p.variability = false

/-- Represents the population of Grade 8 (1) students in a certain school -/
def grade_8_1_population : Population :=
  { size := 50,  -- Assuming a typical class size
    accessibility := true,
    variability := false }

/-- Theorem stating that a comprehensive survey is suitable for the Grade 8 (1) population -/
theorem comprehensive_survey_suitable_for_grade_8_1 :
  is_suitable SurveyType.Comprehensive grade_8_1_population :=
by
  sorry


end NUMINAMATH_CALUDE_comprehensive_survey_suitable_for_grade_8_1_l241_24152


namespace NUMINAMATH_CALUDE_system_solution_l241_24108

theorem system_solution : ∃! (u v : ℝ), 5 * u = -7 - 2 * v ∧ 3 * u = 4 * v - 25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l241_24108


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l241_24122

theorem base_4_9_digit_difference (n : ℕ) : n = 1296 →
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l241_24122


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l241_24113

theorem complex_modulus_problem (i : ℂ) (a : ℝ) :
  i^2 = -1 →
  (∃ (b : ℝ), (2 - i) / (a + i) = b * i) →
  Complex.abs ((2 * a + 1) + Real.sqrt 2 * i) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l241_24113


namespace NUMINAMATH_CALUDE_weight_of_a_l241_24182

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 80 →
  (a + b + c + d) / 4 = 82 →
  e = d + 3 →
  (b + c + d + e) / 4 = 81 →
  a = 95 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l241_24182


namespace NUMINAMATH_CALUDE_intersection_with_complement_l241_24115

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l241_24115


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l241_24129

theorem consecutive_integers_sum (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + b + c + d = 274) → (b = 68) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l241_24129


namespace NUMINAMATH_CALUDE_cube_root_simplification_l241_24107

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 + 8000)^(1/3) = 8 * (1500)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l241_24107


namespace NUMINAMATH_CALUDE_simplify_expression_l241_24161

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l241_24161


namespace NUMINAMATH_CALUDE_triangular_prism_has_nine_edges_l241_24166

/-- The number of sides in the base polygon of a triangular prism -/
def triangular_prism_base_sides : ℕ := 3

/-- The number of edges in a prism given the number of sides in its base polygon -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- Theorem: A triangular prism has 9 edges -/
theorem triangular_prism_has_nine_edges :
  prism_edges triangular_prism_base_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_has_nine_edges_l241_24166


namespace NUMINAMATH_CALUDE_log_equality_implies_ln_a_l241_24134

theorem log_equality_implies_ln_a (a : ℝ) (h : a > 0) :
  (Real.log (8 * a) / Real.log (9 * a) = Real.log (2 * a) / Real.log (3 * a)) →
  (Real.log a = (Real.log 2 * Real.log 3) / (Real.log 3 - 2 * Real.log 2)) := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ln_a_l241_24134


namespace NUMINAMATH_CALUDE_tan_105_degrees_l241_24192

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l241_24192


namespace NUMINAMATH_CALUDE_min_fraction_sum_l241_24187

def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem min_fraction_sum (W X Y Z : ℕ) 
  (h_distinct : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z)
  (h_in_set : W ∈ digits ∧ X ∈ digits ∧ Y ∈ digits ∧ Z ∈ digits) :
  (W : ℚ) / X + (Y : ℚ) / Z ≥ 17 / 30 :=
sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l241_24187


namespace NUMINAMATH_CALUDE_inequality_proof_l241_24190

theorem inequality_proof (α : ℝ) (hα : α > 0) (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^(Real.sin α)^2 * y^(Real.cos α)^2 < x + y :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l241_24190


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_l241_24198

theorem largest_n_binomial_sum : ∃ (n : ℕ), (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 m → m ≤ n) ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_l241_24198


namespace NUMINAMATH_CALUDE_green_ball_probability_l241_24125

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of selecting a green ball -/
def totalGreenProbability (containers : List Container) : ℚ :=
  (containers.map greenProbability).sum / containers.length

theorem green_ball_probability :
  let containers : List Container := [
    { red := 8, green := 4 },
    { red := 3, green := 5 },
    { red := 4, green := 4 }
  ]
  totalGreenProbability containers = 35 / 72 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l241_24125


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l241_24117

theorem cube_sum_of_roots (p q r : ℂ) : 
  (p^3 - 2*p^2 + 3*p - 4 = 0) →
  (q^3 - 2*q^2 + 3*q - 4 = 0) →
  (r^3 - 2*r^2 + 3*r - 4 = 0) →
  p^3 + q^3 + r^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l241_24117


namespace NUMINAMATH_CALUDE_product_abcd_equals_1280_l241_24119

theorem product_abcd_equals_1280 
  (a b c d : ℝ) 
  (eq1 : 2*a + 4*b + 6*c + 8*d = 48)
  (eq2 : 4*d + 2*c = 2*b)
  (eq3 : 4*b + 2*c = 2*a)
  (eq4 : c - 2 = d)
  (eq5 : d + b = 10) :
  a * b * c * d = 1280 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_equals_1280_l241_24119


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l241_24114

/-- Given a natural number n, prove that if the sum of all coefficients in the expansion of (√x + 3/x)^n
    plus the sum of binomial coefficients equals 72, then the constant term in the expansion is 9. -/
theorem binomial_expansion_constant_term (n : ℕ) : 
  (4^n + 2^n = 72) → 
  (∃ (r : ℕ), r < n ∧ (n.choose r) * 3^r = 9) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l241_24114


namespace NUMINAMATH_CALUDE_intersection_equals_N_l241_24118

def M : Set ℝ := {x | x < 2011}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_N_l241_24118


namespace NUMINAMATH_CALUDE_infinite_primes_with_solutions_l241_24165

theorem infinite_primes_with_solutions (S : Finset Nat) (h : ∀ p ∈ S, Nat.Prime p) :
  ∃ p : Nat, p ∉ S ∧ Nat.Prime p ∧ ∃ x : ℤ, x^2 + x + 1 = p := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_with_solutions_l241_24165


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l241_24111

theorem function_value_at_negative_one (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  f (-1) = 12 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l241_24111


namespace NUMINAMATH_CALUDE_inclination_angle_range_l241_24121

/-- The range of inclination angles for a line with equation x*cos(θ) + √3*y - 1 = 0 -/
theorem inclination_angle_range (θ : ℝ) :
  let l : Set (ℝ × ℝ) := {(x, y) | x * Real.cos θ + Real.sqrt 3 * y - 1 = 0}
  let α := Real.arctan (-Real.sqrt 3 / 3 * Real.cos θ)
  α ∈ Set.union (Set.Icc 0 (Real.pi / 6)) (Set.Icc (5 * Real.pi / 6) Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_inclination_angle_range_l241_24121


namespace NUMINAMATH_CALUDE_nancy_total_games_l241_24183

/-- The total number of games Nancy will attend over three months -/
def total_games (this_month : ℕ) (last_month : ℕ) (next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Theorem stating that Nancy will attend 24 games in total -/
theorem nancy_total_games : 
  total_games 9 8 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_nancy_total_games_l241_24183


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l241_24148

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (|x| - 2) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l241_24148


namespace NUMINAMATH_CALUDE_inequality_implication_l241_24110

theorem inequality_implication (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2) → 
  a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l241_24110


namespace NUMINAMATH_CALUDE_m_range_characterization_l241_24139

/-- Proposition P: The equation x^2 + mx + 1 = 0 has two distinct negative roots -/
def P (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition Q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The range of values for m satisfying the given conditions -/
def m_range (m : ℝ) : Prop := m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3

theorem m_range_characterization :
  ∀ m : ℝ, ((P m ∨ Q m) ∧ ¬(P m ∧ Q m)) ↔ m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l241_24139


namespace NUMINAMATH_CALUDE_grace_garden_medium_bed_rows_l241_24132

/-- Represents a raised bed garden with large and medium beds -/
structure RaisedBedGarden where
  large_beds : Nat
  medium_beds : Nat
  large_bed_rows : Nat
  large_bed_seeds_per_row : Nat
  medium_bed_seeds_per_row : Nat
  total_seeds : Nat

/-- Calculates the number of rows in medium beds -/
def medium_bed_rows (garden : RaisedBedGarden) : Nat :=
  let large_bed_seeds := garden.large_beds * garden.large_bed_rows * garden.large_bed_seeds_per_row
  let medium_bed_seeds := garden.total_seeds - large_bed_seeds
  medium_bed_seeds / garden.medium_bed_seeds_per_row

/-- Theorem stating that for the given garden configuration, medium beds have 6 rows -/
theorem grace_garden_medium_bed_rows :
  let garden : RaisedBedGarden := {
    large_beds := 2,
    medium_beds := 2,
    large_bed_rows := 4,
    large_bed_seeds_per_row := 25,
    medium_bed_seeds_per_row := 20,
    total_seeds := 320
  }
  medium_bed_rows garden = 6 := by
  sorry

end NUMINAMATH_CALUDE_grace_garden_medium_bed_rows_l241_24132


namespace NUMINAMATH_CALUDE_decryption_works_l241_24179

-- Define the Russian alphabet (excluding 'ё')
def russian_alphabet : List Char := ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

-- Define the encryption steps
def swap_adjacent (s : String) : String := sorry

def shift_right (s : String) (n : Nat) : String := sorry

def reverse_string (s : String) : String := sorry

-- Define the decryption steps
def shift_left (s : String) (n : Nat) : String := sorry

-- Define the full encryption and decryption processes
def encrypt (s : String) : String :=
  reverse_string (shift_right (swap_adjacent s) 2)

def decrypt (s : String) : String :=
  swap_adjacent (shift_left (reverse_string s) 2)

-- Theorem to prove
theorem decryption_works (encrypted : String) (decrypted : String) :
  encrypted = "врпвл терпраиэ вйзгцфпз" ∧ 
  decrypted = "нефте базы южного района" →
  decrypt encrypted = decrypted := by sorry

end NUMINAMATH_CALUDE_decryption_works_l241_24179


namespace NUMINAMATH_CALUDE_lily_account_balance_l241_24100

def initial_amount : ℕ := 55
def shirt_cost : ℕ := 7

theorem lily_account_balance :
  initial_amount - (shirt_cost + 3 * shirt_cost) = 27 :=
by sorry

end NUMINAMATH_CALUDE_lily_account_balance_l241_24100


namespace NUMINAMATH_CALUDE_g_of_fifty_l241_24170

/-- A function g satisfying g(xy) = xg(y) for all real x and y, and g(1) = 30 -/
def g : ℝ → ℝ :=
  fun x => x * 30

/-- Theorem stating that g(50) = 1500 -/
theorem g_of_fifty : g 50 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_g_of_fifty_l241_24170


namespace NUMINAMATH_CALUDE_circle_intersection_range_l241_24186

/-- The problem statement translated to Lean 4 --/
theorem circle_intersection_range :
  ∀ (a : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x - a)^2 + (y - (a - 3))^2 = 1) ↔ 0 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l241_24186


namespace NUMINAMATH_CALUDE_divisible_by_eight_count_l241_24174

theorem divisible_by_eight_count : 
  (Finset.filter (fun n => n % 8 = 0) (Finset.Icc 200 400)).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_count_l241_24174


namespace NUMINAMATH_CALUDE_complex_cube_root_l241_24120

theorem complex_cube_root : ∃ (z : ℂ), z^2 + 2 = 0 → z^3 = 2 * Real.sqrt 2 * I ∨ z^3 = -2 * Real.sqrt 2 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l241_24120


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l241_24184

theorem sqrt_equation_solution (z : ℚ) : 
  Real.sqrt (5 - 4 * z + 1) = 7 → z = -43 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l241_24184


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l241_24197

theorem angle_in_second_quadrant : 
  let θ : Real := (2 * Real.pi) / 3
  0 < θ ∧ θ < Real.pi / 2 → False ∧ 
  Real.pi / 2 < θ ∧ θ ≤ Real.pi → True ∧
  Real.pi < θ ∧ θ < 3 * Real.pi / 2 → False ∧
  3 * Real.pi / 2 ≤ θ ∧ θ < 2 * Real.pi → False :=
by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l241_24197


namespace NUMINAMATH_CALUDE_gcf_of_45_135_60_l241_24163

theorem gcf_of_45_135_60 : Nat.gcd 45 (Nat.gcd 135 60) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_45_135_60_l241_24163


namespace NUMINAMATH_CALUDE_student_allowance_proof_l241_24157

/-- The student's weekly allowance in dollars -/
def weekly_allowance : ℝ := 4.50

theorem student_allowance_proof :
  ∃ (arcade_spent toy_store_spent : ℝ),
    arcade_spent = (3/5) * weekly_allowance ∧
    toy_store_spent = (1/3) * (weekly_allowance - arcade_spent) ∧
    weekly_allowance - arcade_spent - toy_store_spent = 1.20 :=
by sorry

end NUMINAMATH_CALUDE_student_allowance_proof_l241_24157


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l241_24158

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the line x = a²/c intersects its asymptotes at points A and B,
    and triangle ABF is a right-angled triangle (where F is the right focus),
    then the eccentricity of the hyperbola is √2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  A.1 = a^2 / c →
  B.1 = a^2 / c →
  F.1 = c →
  F.2 = 0 →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
  c / a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l241_24158


namespace NUMINAMATH_CALUDE_chess_and_go_purchase_l241_24188

theorem chess_and_go_purchase (m : ℕ) : 
  (m + (120 - m) = 120) →
  (m ≥ 2 * (120 - m)) →
  (30 * m + 25 * (120 - m) ≤ 3500) →
  (80 ≤ m ∧ m ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_chess_and_go_purchase_l241_24188


namespace NUMINAMATH_CALUDE_syllogism_structure_l241_24102

/-- A syllogism in deductive reasoning -/
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

/-- The components of a syllogism -/
def syllogism_components (s : Syllogism) : List Prop :=
  [s.major_premise, s.minor_premise, s.conclusion]

/-- Theorem stating that a syllogism consists of major premise, minor premise, and conclusion -/
theorem syllogism_structure :
  ∀ (s : Syllogism), syllogism_components s = [s.major_premise, s.minor_premise, s.conclusion] :=
by
  sorry

#check syllogism_structure

end NUMINAMATH_CALUDE_syllogism_structure_l241_24102


namespace NUMINAMATH_CALUDE_smallest_population_satisfying_conditions_l241_24189

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem smallest_population_satisfying_conditions :
  ∃ (n : ℕ),
    (is_perfect_square n) ∧
    (is_perfect_square (n + 100)) ∧
    (∃ k : ℕ, n + 50 = k * k + 1) ∧
    (n % 3 = 0) ∧
    (∀ m : ℕ, m < n →
      ¬(is_perfect_square m ∧
        is_perfect_square (m + 100) ∧
        (∃ k : ℕ, m + 50 = k * k + 1) ∧
        (m % 3 = 0))) ∧
    n = 576 :=
by sorry

end NUMINAMATH_CALUDE_smallest_population_satisfying_conditions_l241_24189


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_46_l241_24164

theorem consecutive_integers_sum_46 :
  ∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧
  y = x + 1 ∧ z = y + 1 ∧ w = z + 1 ∧
  x + y + z + w = 46 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_46_l241_24164


namespace NUMINAMATH_CALUDE_ship_passengers_l241_24140

theorem ship_passengers : 
  ∀ (P : ℕ), 
    (P / 4 : ℚ) + (P / 8 : ℚ) + (P / 12 : ℚ) + (P / 6 : ℚ) + 36 = P → 
    P = 96 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l241_24140


namespace NUMINAMATH_CALUDE_cookies_difference_l241_24106

theorem cookies_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 8 →
  initial_salty = 6 →
  eaten_sweet = 20 →
  eaten_salty = 34 →
  eaten_salty - eaten_sweet = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_difference_l241_24106


namespace NUMINAMATH_CALUDE_investment_interest_l241_24149

/-- Calculates the interest earned on an investment with annual compounding --/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- The interest earned on a $500 investment at 2% annual interest for 3 years is $31 --/
theorem investment_interest : 
  ∃ ε > 0, |interest_earned 500 0.02 3 - 31| < ε :=
sorry

end NUMINAMATH_CALUDE_investment_interest_l241_24149


namespace NUMINAMATH_CALUDE_david_widget_production_l241_24128

theorem david_widget_production (w t : ℕ) (h : w = 3 * t) :
  w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_david_widget_production_l241_24128


namespace NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l241_24196

/-- Calculates the number of trips to the water fountain given the distance to the fountain and total distance walked. -/
def trips_to_fountain (distance_to_fountain : ℕ) (total_distance_walked : ℕ) : ℕ :=
  total_distance_walked / (2 * distance_to_fountain)

/-- Theorem stating that given a distance of 30 feet to the fountain and 120 feet walked, the number of trips is 2. -/
theorem mrs_hilt_fountain_trips :
  trips_to_fountain 30 120 = 2 := by
  sorry


end NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l241_24196


namespace NUMINAMATH_CALUDE_parabola_opens_left_is_ellipse_parabola_ellipse_system_correct_l241_24167

/-- Represents a parabola and an ellipse in a 2D coordinate system -/
structure ParabolaEllipseSystem where
  m : ℝ
  n : ℝ
  hm : m > 0
  hn : n > 0

/-- The parabola equation: mx + ny² = 0 -/
def parabola_equation (sys : ParabolaEllipseSystem) (x y : ℝ) : Prop :=
  sys.m * x + sys.n * y^2 = 0

/-- The ellipse equation: mx² + ny² = 1 -/
def ellipse_equation (sys : ParabolaEllipseSystem) (x y : ℝ) : Prop :=
  sys.m * x^2 + sys.n * y^2 = 1

/-- Theorem stating that the parabola opens to the left -/
theorem parabola_opens_left (sys : ParabolaEllipseSystem) :
  ∀ x y, parabola_equation sys x y → x ≤ 0 :=
sorry

/-- Theorem stating that the equation represents an ellipse -/
theorem is_ellipse (sys : ParabolaEllipseSystem) :
  ∃ a b, a > 0 ∧ b > 0 ∧ ∀ x y, ellipse_equation sys x y ↔ (x/a)^2 + (y/b)^2 = 1 :=
sorry

/-- Main theorem: The system represents a left-opening parabola and an ellipse -/
theorem parabola_ellipse_system_correct (sys : ParabolaEllipseSystem) :
  (∀ x y, parabola_equation sys x y → x ≤ 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ ∀ x y, ellipse_equation sys x y ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_opens_left_is_ellipse_parabola_ellipse_system_correct_l241_24167


namespace NUMINAMATH_CALUDE_davids_biology_marks_l241_24105

theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 36)
  (h2 : mathematics = 35)
  (h3 : physics = 42)
  (h4 : chemistry = 57)
  (h5 : average = 45)
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) :
  biology = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l241_24105


namespace NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l241_24168

theorem derivative_inequality_implies_function_inequality 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x > 0, deriv f x - f x / x > 0) → 3 * f 4 > 4 * f 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_inequality_implies_function_inequality_l241_24168


namespace NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l241_24175

/-- The sum of interior angles of a regular hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  ∀ (n : ℕ) (sum : ℝ),
  n = 6 →
  sum = (n - 2) * 180 →
  sum = 720 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l241_24175
