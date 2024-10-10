import Mathlib

namespace complex_sum_theorem_l1765_176572

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 →
  g = -a - c - e →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) + (g + h * Complex.I) = -2 * Complex.I →
  d + f + h = -4 := by
sorry

end complex_sum_theorem_l1765_176572


namespace afternoon_bike_sales_l1765_176564

theorem afternoon_bike_sales (morning_sales : ℕ) (total_clamps : ℕ) (clamps_per_bike : ℕ) :
  morning_sales = 19 →
  total_clamps = 92 →
  clamps_per_bike = 2 →
  ∃ (afternoon_sales : ℕ), 
    afternoon_sales = 27 ∧
    total_clamps = clamps_per_bike * (morning_sales + afternoon_sales) :=
by sorry

end afternoon_bike_sales_l1765_176564


namespace baker_pastries_cakes_difference_l1765_176544

theorem baker_pastries_cakes_difference (cakes pastries : ℕ) 
  (h1 : cakes = 19) 
  (h2 : pastries = 131) : 
  pastries - cakes = 112 := by
sorry

end baker_pastries_cakes_difference_l1765_176544


namespace octal_724_equals_468_l1765_176569

/-- Converts an octal number represented as a list of digits to its decimal equivalent. -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the number -/
def octal_number : List Nat := [4, 2, 7]

theorem octal_724_equals_468 :
  octal_to_decimal octal_number = 468 := by
  sorry

end octal_724_equals_468_l1765_176569


namespace unique_number_with_property_l1765_176573

/-- Calculate the total number of digits needed to write all integers from 1 to n -/
def totalDigits (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

/-- The property that the number, when doubled, equals the total number of digits -/
def hasProperty (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ 2 * x = totalDigits x

theorem unique_number_with_property :
  ∃! x : ℕ, hasProperty x ∧ x = 108 := by
  sorry

end unique_number_with_property_l1765_176573


namespace quadratic_root_condition_l1765_176581

theorem quadratic_root_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ + m - 1 = 0 ∧ 
                x₂^2 - 4*x₂ + m - 1 = 0 ∧ 
                3*x₁*x₂ - x₁ - x₂ > 2) →
  3 < m ∧ m ≤ 5 := by
sorry

end quadratic_root_condition_l1765_176581


namespace special_function_upper_bound_l1765_176508

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧ 
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

/-- The main theorem -/
theorem special_function_upper_bound 
  (f : ℝ → ℝ) (h : SpecialFunction f) : 
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by sorry

end special_function_upper_bound_l1765_176508


namespace inequality_proof_l1765_176556

theorem inequality_proof (a b c d : ℝ) (h1 : a * b > 0) (h2 : -c / a < -d / b) :
  b * c > a * d := by
  sorry

end inequality_proof_l1765_176556


namespace min_floor_sum_l1765_176548

theorem min_floor_sum (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋ ≥ 9 := by
  sorry

end min_floor_sum_l1765_176548


namespace head_probability_l1765_176587

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
  | Head
  | Tail

/-- A fair coin toss -/
def FairCoin : Type := CoinOutcome

/-- The probability of an outcome in a fair coin toss -/
def prob (outcome : CoinOutcome) : ℚ :=
  1 / 2

theorem head_probability (c : FairCoin) :
  prob CoinOutcome.Head = 1 / 2 := by
  sorry

end head_probability_l1765_176587


namespace negation_of_cube_odd_is_odd_l1765_176580

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ Even (n^3)) :=
by sorry

end negation_of_cube_odd_is_odd_l1765_176580


namespace A_share_of_profit_l1765_176595

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_A_share_of_profit (a_initial : ℕ) (b_initial : ℕ) (a_withdrawal : ℕ) (b_addition : ℕ) (months_before_change : ℕ) (total_months : ℕ) (total_profit : ℕ) : ℚ :=
  let a_investment_months := a_initial * months_before_change + (a_initial - a_withdrawal) * (total_months - months_before_change)
  let b_investment_months := b_initial * months_before_change + (b_initial + b_addition) * (total_months - months_before_change)
  let total_investment_months := a_investment_months + b_investment_months
  (a_investment_months : ℚ) / total_investment_months * total_profit

theorem A_share_of_profit :
  calculate_A_share_of_profit 3000 4000 1000 1000 8 12 630 = 240 := by
  sorry

end A_share_of_profit_l1765_176595


namespace f_properties_l1765_176574

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_properties :
  is_smallest_positive_period π f ∧
  is_monotonically_decreasing f (π / 3) (5 * π / 6) ∧
  ∀ α : ℝ, 
    (3 * π / 2 < α ∧ α < 2 * π) →  -- α in fourth quadrant
    Real.cos α = 3 / 5 → 
    f (α / 2 + 7 * π / 12) = 8 / 5 :=
by sorry

end f_properties_l1765_176574


namespace folded_paper_distance_l1765_176585

theorem folded_paper_distance (sheet_area : ℝ) (folded_leg : ℝ) : 
  sheet_area = 6 →
  folded_leg ^ 2 / 2 = sheet_area - folded_leg ^ 2 →
  Real.sqrt (2 * folded_leg ^ 2) = 2 * Real.sqrt 2 := by
  sorry

end folded_paper_distance_l1765_176585


namespace subtract_preserves_inequality_l1765_176528

theorem subtract_preserves_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end subtract_preserves_inequality_l1765_176528


namespace fraction_subtraction_l1765_176543

theorem fraction_subtraction : (18 : ℚ) / 42 - (3 : ℚ) / 8 = (3 : ℚ) / 56 := by
  sorry

end fraction_subtraction_l1765_176543


namespace geometric_sequence_property_l1765_176537

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : 1 / (a 2 * a 4) + 2 / (a 4 * a 4) + 1 / (a 4 * a 6) = 81) :
  1 / a 3 + 1 / a 5 = 9 := by
  sorry

end geometric_sequence_property_l1765_176537


namespace inequality_proof_l1765_176516

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
by sorry

end inequality_proof_l1765_176516


namespace cheryl_egg_difference_l1765_176578

/-- The number of eggs found by Kevin -/
def kevin_eggs : ℕ := 5

/-- The number of eggs found by Bonnie -/
def bonnie_eggs : ℕ := 13

/-- The number of eggs found by George -/
def george_eggs : ℕ := 9

/-- The number of eggs found by Cheryl -/
def cheryl_eggs : ℕ := 56

/-- Theorem stating that Cheryl found 29 more eggs than the other three children combined -/
theorem cheryl_egg_difference : 
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end cheryl_egg_difference_l1765_176578


namespace book_pages_count_l1765_176533

theorem book_pages_count (total_notebooks : ℕ) (sum_of_four_pages : ℕ) : 
  total_notebooks = 12 ∧ sum_of_four_pages = 338 → 
  ∃ (total_pages : ℕ), 
    total_pages = 288 ∧
    (total_pages / total_notebooks : ℚ) + 1 + 
    (total_pages / total_notebooks : ℚ) + 2 + 
    (total_pages / 3 : ℚ) - 1 + 
    (total_pages / 3 : ℚ) = sum_of_four_pages := by
  sorry

#check book_pages_count

end book_pages_count_l1765_176533


namespace easter_eggs_per_basket_l1765_176535

theorem easter_eggs_per_basket : ∃ (n : ℕ), n ≥ 5 ∧ n ∣ 30 ∧ n ∣ 42 ∧ ∀ (m : ℕ), m ≥ 5 ∧ m ∣ 30 ∧ m ∣ 42 → m ≤ n :=
by sorry

end easter_eggs_per_basket_l1765_176535


namespace xiaoli_estimate_l1765_176570

theorem xiaoli_estimate (x y z : ℝ) (hxy : x > y) (hy : y > 0) (hz : z > 0) :
  (x + z) + (y - z) = x + y := by
  sorry

end xiaoli_estimate_l1765_176570


namespace complex_root_quadratic_l1765_176515

theorem complex_root_quadratic (b c : ℝ) : 
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 → 
  b = -2 ∧ c = 3 := by
  sorry

end complex_root_quadratic_l1765_176515


namespace mushroom_picking_ratio_l1765_176500

/-- Proves the ratio of mushrooms picked on the last day to the second day -/
theorem mushroom_picking_ratio : 
  ∀ (total_mushrooms first_day_revenue second_day_picked price_per_mushroom : ℕ),
  total_mushrooms = 65 →
  first_day_revenue = 58 →
  second_day_picked = 12 →
  price_per_mushroom = 2 →
  (total_mushrooms - first_day_revenue / price_per_mushroom - second_day_picked) / second_day_picked = 2 := by
  sorry

end mushroom_picking_ratio_l1765_176500


namespace function_property_l1765_176529

/-- Given a function g : ℝ → ℝ satisfying g(x)g(y) - g(xy) = x - y for all real x and y,
    prove that g(3) = -2 -/
theorem function_property (g : ℝ → ℝ) 
    (h : ∀ x y : ℝ, g x * g y - g (x * y) = x - y) : 
    g 3 = -2 := by
  sorry

end function_property_l1765_176529


namespace savings_difference_l1765_176523

def savings_problem : Prop :=
  let dick_1989 := 5000
  let jane_1989 := 5000
  let dick_1990 := dick_1989 * 1.10
  let jane_1990 := jane_1989 * 0.95
  let dick_1991 := dick_1990 * 1.07
  let jane_1991 := jane_1990 * 1.08
  let dick_1992 := dick_1991 * 0.88
  let jane_1992 := jane_1991 * 1.15
  let dick_total := dick_1989 + dick_1990 + dick_1991 + dick_1992
  let jane_total := jane_1989 + jane_1990 + jane_1991 + jane_1992
  dick_total - jane_total = 784.30

theorem savings_difference : savings_problem := by
  sorry

end savings_difference_l1765_176523


namespace f_ln_2_value_l1765_176590

/-- A function f is monotonically decreasing on (0, +∞) if for any a, b ∈ (0, +∞) with a < b, f(a) ≥ f(b) -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 < a → a < b → f a ≥ f b

/-- The main theorem -/
theorem f_ln_2_value (f : ℝ → ℝ) 
  (h_mono : MonoDecreasing f)
  (h_domain : ∀ x, x > 0 → f x ≠ 0)
  (h_eq : ∀ x, x > 0 → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3/2 := by
  sorry

end f_ln_2_value_l1765_176590


namespace ramesh_profit_share_l1765_176517

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  (investment2 * total_profit) / (investment1 + investment2)

/-- Theorem stating that Ramesh's share of the profit is 11,875 --/
theorem ramesh_profit_share :
  calculate_profit_share 24000 40000 19000 = 11875 := by
  sorry

end ramesh_profit_share_l1765_176517


namespace arithmetic_geometric_sequence_l1765_176546

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 2 * a 8 = a 4 * a 4) →     -- a_2, a_4, a_8 form a geometric sequence
  a 4 = 12 :=
by sorry

end arithmetic_geometric_sequence_l1765_176546


namespace intersection_A_B_l1765_176598

-- Define the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end intersection_A_B_l1765_176598


namespace value_of_expression_l1765_176567

theorem value_of_expression (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end value_of_expression_l1765_176567


namespace madeline_water_goal_l1765_176506

/-- The amount of water Madeline wants to drink in a day -/
def waterGoal (bottleCapacity : ℕ) (refills : ℕ) (additionalWater : ℕ) : ℕ :=
  bottleCapacity * refills + additionalWater

/-- Proves that Madeline's water goal is 100 ounces -/
theorem madeline_water_goal :
  waterGoal 12 7 16 = 100 := by
  sorry

end madeline_water_goal_l1765_176506


namespace parabola_standard_equation_l1765_176560

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - 2 * y - 4 = 0

-- Define the parabola equations
def parabola_equation_1 (x y : ℝ) : Prop := y^2 = 16 * x
def parabola_equation_2 (x y : ℝ) : Prop := x^2 = -8 * y

-- Define a parabola type
structure Parabola where
  focus : ℝ × ℝ
  is_on_line : line_equation focus.1 focus.2

-- Theorem statement
theorem parabola_standard_equation (p : Parabola) :
  (∃ x y : ℝ, parabola_equation_1 x y) ∨ (∃ x y : ℝ, parabola_equation_2 x y) :=
sorry

end parabola_standard_equation_l1765_176560


namespace volume_ratio_volume_112oz_l1765_176562

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℝ
  /-- Assumption that k is positive -/
  k_pos : k > 0

/-- Volume of the substance given its weight -/
def volume (s : Substance) (weight : ℝ) : ℝ :=
  s.k * weight

/-- Theorem stating the relationship between volumes of different weights -/
theorem volume_ratio (s : Substance) (w1 w2 v1 : ℝ) (hw1 : w1 > 0) (hw2 : w2 > 0) (hv1 : v1 > 0)
    (h : volume s w1 = v1) :
    volume s w2 = v1 * (w2 / w1) := by
  sorry

/-- Main theorem proving the volume for 112 ounces given the volume for 84 ounces -/
theorem volume_112oz (s : Substance) (h : volume s 84 = 36) :
    volume s 112 = 48 := by
  sorry

end volume_ratio_volume_112oz_l1765_176562


namespace peter_mowing_time_l1765_176583

/-- The time it takes Nancy to mow the yard alone (in hours) -/
def nancy_time : ℝ := 3

/-- The time it takes Nancy and Peter together to mow the yard (in hours) -/
def combined_time : ℝ := 1.71428571429

/-- The time it takes Peter to mow the yard alone (in hours) -/
def peter_time : ℝ := 4

/-- Theorem stating that given Nancy's time and the combined time, Peter's individual time is approximately 4 hours -/
theorem peter_mowing_time (ε : ℝ) (h_ε : ε > 0) :
  ∃ (t : ℝ), abs (t - peter_time) < ε ∧ 
  1 / nancy_time + 1 / t = 1 / combined_time :=
sorry


end peter_mowing_time_l1765_176583


namespace inequality_proof_l1765_176507

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (b * (a + b)) + 1 / (c * (b + c)) + 1 / (a * (c + a)) ≥ 27 / (2 * (a + b + c)^2) := by
  sorry

end inequality_proof_l1765_176507


namespace color_film_fraction_l1765_176594

theorem color_film_fraction (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) : 
  let total_bw := 30 * x
  let total_color := 6 * y
  let bw_selected_percent := y / x
  let bw_selected := bw_selected_percent * total_bw / 100
  let color_selected := total_color
  let total_selected := bw_selected + color_selected
  color_selected / total_selected = 20 / 21 := by
sorry

end color_film_fraction_l1765_176594


namespace constant_b_value_l1765_176550

theorem constant_b_value (a b : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x - a) = x^2 - b*x - 10) → b = -1/3 := by
  sorry

end constant_b_value_l1765_176550


namespace ln_graph_rotation_l1765_176514

open Real

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := log x

-- Define the rotation angle
variable (θ : ℝ)

-- State the theorem
theorem ln_graph_rotation (h : ∃ x > 0, f x * cos θ + x * sin θ = 0) :
  sin θ = ℯ * cos θ :=
sorry

end ln_graph_rotation_l1765_176514


namespace fourteen_trucks_sufficient_l1765_176521

/-- Represents the number of packages of each size -/
structure PackageDistribution where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the capacity of a Type B truck for each package size -/
structure TruckCapacity where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the number of Type B trucks needed given a package distribution and truck capacity -/
def trucksNeeded (packages : PackageDistribution) (capacity : TruckCapacity) : ℕ :=
  let smallTrucks := (packages.small + capacity.small - 1) / capacity.small
  let mediumTrucks := (packages.medium + capacity.medium - 1) / capacity.medium
  let largeTrucks := (packages.large + capacity.large - 1) / capacity.large
  smallTrucks + mediumTrucks + largeTrucks

/-- Theorem stating that 14 Type B trucks are sufficient for the given package distribution -/
theorem fourteen_trucks_sufficient 
  (packages : PackageDistribution)
  (capacity : TruckCapacity)
  (h1 : packages.small + packages.medium + packages.large = 1000)
  (h2 : packages.small = 2 * packages.medium)
  (h3 : packages.medium = 3 * packages.large)
  (h4 : capacity.small = 90)
  (h5 : capacity.medium = 60)
  (h6 : capacity.large = 50) :
  trucksNeeded packages capacity ≤ 14 :=
sorry

end fourteen_trucks_sufficient_l1765_176521


namespace expression_simplification_l1765_176552

theorem expression_simplification (m : ℝ) : 
  ((7*m + 3) - 3*m*2)*4 + (5 - 2/4)*(8*m - 12) = 40*m - 42 := by
sorry

end expression_simplification_l1765_176552


namespace subset_relation_l1765_176554

theorem subset_relation (x : ℝ) : x > 1 → x^2 - x > 0 := by
  sorry

end subset_relation_l1765_176554


namespace video_game_lives_l1765_176538

theorem video_game_lives (initial_players : Nat) (quitting_players : Nat) (lives_per_player : Nat) : 
  initial_players = 20 → quitting_players = 10 → lives_per_player = 7 → 
  (initial_players - quitting_players) * lives_per_player = 70 := by
  sorry

end video_game_lives_l1765_176538


namespace right_triangle_hypotenuse_l1765_176571

theorem right_triangle_hypotenuse : 
  ∀ (hypotenuse : ℝ), 
  hypotenuse > 0 →
  (hypotenuse - 1)^2 + 7^2 = hypotenuse^2 →
  hypotenuse = 25 := by
sorry

end right_triangle_hypotenuse_l1765_176571


namespace magnitude_of_vector_2_neg1_l1765_176519

/-- The magnitude of a 2D vector (2, -1) is √5 -/
theorem magnitude_of_vector_2_neg1 :
  let a : Fin 2 → ℝ := ![2, -1]
  Real.sqrt ((a 0) ^ 2 + (a 1) ^ 2) = Real.sqrt 5 := by
  sorry

end magnitude_of_vector_2_neg1_l1765_176519


namespace determinant_transformation_l1765_176510

theorem determinant_transformation (p q r s : ℝ) 
  (h : Matrix.det !![p, q; r, s] = 6) : 
  Matrix.det !![p, 9*p + 4*q; r, 9*r + 4*s] = 24 := by
  sorry

end determinant_transformation_l1765_176510


namespace square_root_of_four_l1765_176527

theorem square_root_of_four :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 :=
sorry

end square_root_of_four_l1765_176527


namespace jasons_correct_answers_l1765_176540

theorem jasons_correct_answers
  (total_problems : ℕ)
  (points_for_correct : ℕ)
  (points_for_incorrect : ℕ)
  (final_score : ℕ)
  (h1 : total_problems = 12)
  (h2 : points_for_correct = 4)
  (h3 : points_for_incorrect = 1)
  (h4 : final_score = 33) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_problems ∧
    points_for_correct * correct_answers -
    points_for_incorrect * (total_problems - correct_answers) = final_score ∧
    correct_answers = 9 :=
by
  sorry

end jasons_correct_answers_l1765_176540


namespace fourth_root_equivalence_l1765_176534

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : (x^2 * x^(1/2))^(1/4) = x^(5/8) := by
  sorry

end fourth_root_equivalence_l1765_176534


namespace basketball_free_throws_l1765_176524

/-- Represents the scoring of a basketball team -/
structure BasketballScore where
  two_pointers : ℕ
  three_pointers : ℕ
  free_throws : ℕ

/-- Checks if the given BasketballScore satisfies the problem conditions -/
def is_valid_score (score : BasketballScore) : Prop :=
  3 * score.three_pointers = 2 * 2 * score.two_pointers ∧
  score.free_throws = 2 * score.two_pointers - 3 ∧
  2 * score.two_pointers + 3 * score.three_pointers + score.free_throws = 73

theorem basketball_free_throws (score : BasketballScore) :
  is_valid_score score → score.free_throws = 21 := by
  sorry

end basketball_free_throws_l1765_176524


namespace route_redistribution_possible_l1765_176518

/-- Represents an airline with its routes -/
structure Airline where
  id : Nat
  routes : Finset (Nat × Nat)

/-- Represents the initial configuration of airlines -/
def initial_airlines (k : Nat) : Finset Airline :=
  sorry

/-- Checks if an airline complies with the one-route-per-city law -/
def complies_with_law (a : Airline) : Prop :=
  sorry

/-- Checks if all airlines have the same number of routes -/
def equal_routes (airlines : Finset Airline) : Prop :=
  sorry

theorem route_redistribution_possible (k : Nat) :
  ∃ (new_airlines : Finset Airline),
    (∀ a ∈ new_airlines, complies_with_law a) ∧
    equal_routes new_airlines ∧
    new_airlines.card = (initial_airlines k).card :=
  sorry

end route_redistribution_possible_l1765_176518


namespace transportation_cost_optimization_l1765_176555

/-- The transportation cost problem -/
theorem transportation_cost_optimization (a : ℝ) :
  let distance : ℝ := 300
  let fuel_cost_constant : ℝ := 1/2
  let other_costs : ℝ := 800
  let cost_function (x : ℝ) : ℝ := 150 * (x + 1600 / x)
  let optimal_speed : ℝ := if a ≥ 40 then 40 else a
  0 < a →
  (∀ x > 0, x ≤ a → cost_function optimal_speed ≤ cost_function x) :=
by sorry

end transportation_cost_optimization_l1765_176555


namespace count_100_digit_even_numbers_l1765_176596

/-- A function that represents the count of n-digit even numbers where each digit is 0, 1, or 3 -/
def countEvenNumbers (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * 3^(n - 2)

/-- Theorem stating that the count of 100-digit even numbers where each digit is 0, 1, or 3 is 2 * 3^98 -/
theorem count_100_digit_even_numbers :
  countEvenNumbers 100 = 2 * 3^98 := by
  sorry


end count_100_digit_even_numbers_l1765_176596


namespace not_divides_power_diff_l1765_176531

theorem not_divides_power_diff (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) 
  (hm_odd : Odd m) (hn_odd : Odd n) : 
  ¬ ((2^m - 1) ∣ (3^n - 1)) := by
  sorry

end not_divides_power_diff_l1765_176531


namespace imaginary_part_of_one_over_one_plus_i_l1765_176513

theorem imaginary_part_of_one_over_one_plus_i :
  Complex.im (1 / (1 + Complex.I)) = -1/2 := by
  sorry

end imaginary_part_of_one_over_one_plus_i_l1765_176513


namespace series_sum_equals_inverse_sqrt5_minus_1_l1765_176591

/-- The sum of the series $\sum_{k=0}^{\infty} \frac{5^{2^k}}{25^{2^k} - 1}$ is equal to $\frac{1}{\sqrt{5}-1}$ -/
theorem series_sum_equals_inverse_sqrt5_minus_1 :
  let series_term (k : ℕ) := (5 ^ (2 ^ k)) / ((25 ^ (2 ^ k)) - 1)
  ∑' (k : ℕ), series_term k = 1 / (Real.sqrt 5 - 1) := by
  sorry

end series_sum_equals_inverse_sqrt5_minus_1_l1765_176591


namespace cos_pi_plus_2alpha_l1765_176512

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π + 2 * α) = 7 / 9 := by
  sorry

end cos_pi_plus_2alpha_l1765_176512


namespace suv_fuel_efficiency_l1765_176525

theorem suv_fuel_efficiency (highway_mpg city_mpg : ℝ) (max_distance gallons : ℝ) 
  (h1 : highway_mpg = 12.2)
  (h2 : city_mpg = 7.6)
  (h3 : max_distance = 268.4)
  (h4 : gallons = 22)
  (h5 : max_distance = highway_mpg * gallons) :
  city_mpg * gallons = 167.2 := by
  sorry

end suv_fuel_efficiency_l1765_176525


namespace trivia_team_absentees_l1765_176599

/-- Proves that 6 members didn't show up to a trivia game --/
theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 15 →
  points_per_member = 3 →
  total_points = 27 →
  total_members - (total_points / points_per_member) = 6 := by
  sorry


end trivia_team_absentees_l1765_176599


namespace johns_workday_end_l1765_176530

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_def : minutes < 60

/-- Calculates the difference between two times in hours -/
def time_diff (t1 t2 : Time) : ℚ :=
  (t2.hours - t1.hours : ℚ) + (t2.minutes - t1.minutes : ℚ) / 60

/-- Adds hours and minutes to a given time -/
def add_time (t : Time) (h : ℕ) (m : ℕ) : Time :=
  let total_minutes := t.hours * 60 + t.minutes + h * 60 + m
  { hours := total_minutes / 60,
    minutes := total_minutes % 60,
    inv_def := by sorry }

theorem johns_workday_end (work_hours : ℕ) (lunch_break : Time) (start_time end_time : Time) :
  work_hours = 9 →
  lunch_break.hours = 1 ∧ lunch_break.minutes = 15 →
  start_time.hours = 6 ∧ start_time.minutes = 30 →
  time_diff start_time { hours := 11, minutes := 30, inv_def := by sorry } = 5 →
  add_time { hours := 11, minutes := 30, inv_def := by sorry } lunch_break.hours lunch_break.minutes = { hours := 12, minutes := 45, inv_def := by sorry } →
  add_time { hours := 12, minutes := 45, inv_def := by sorry } 4 0 = end_time →
  end_time.hours = 16 ∧ end_time.minutes = 45 :=
by sorry

end johns_workday_end_l1765_176530


namespace parabola_values_l1765_176584

/-- A parabola passing through (1, 1) with a specific tangent line -/
def Parabola (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x - 7

theorem parabola_values (a b : ℝ) :
  (Parabola a b 1 = 1) ∧ 
  (4 * 1 - Parabola a b 1 - 3 = 0) ∧
  (2 * a * 1 + b = 4) →
  a = -4 ∧ b = 12 := by sorry

end parabola_values_l1765_176584


namespace jessica_purchases_total_cost_l1765_176502

/-- The total cost of Jessica's purchases is $41.44 -/
theorem jessica_purchases_total_cost :
  let cat_toy := 10.22
  let cage := 11.73
  let cat_food := 8.15
  let collar := 4.35
  let litter_box := 6.99
  cat_toy + cage + cat_food + collar + litter_box = 41.44 := by
  sorry

end jessica_purchases_total_cost_l1765_176502


namespace least_cubes_for_6x9x12_block_l1765_176575

/-- Represents the dimensions of a cuboidal block in centimeters -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the least number of equal cubes that can be cut from a block -/
def leastNumberOfEqualCubes (d : BlockDimensions) : ℕ :=
  (d.length * d.width * d.height) / (Nat.gcd d.length (Nat.gcd d.width d.height))^3

/-- Theorem stating that for a 6x9x12 cm block, the least number of equal cubes is 24 -/
theorem least_cubes_for_6x9x12_block :
  leastNumberOfEqualCubes ⟨6, 9, 12⟩ = 24 := by
  sorry

#eval leastNumberOfEqualCubes ⟨6, 9, 12⟩

end least_cubes_for_6x9x12_block_l1765_176575


namespace percentage_of_female_students_l1765_176579

theorem percentage_of_female_students 
  (total_students : ℕ) 
  (female_percentage : ℝ) 
  (brunette_percentage : ℝ) 
  (under_5ft_percentage : ℝ) 
  (under_5ft_count : ℕ) :
  total_students = 200 →
  brunette_percentage = 50 →
  under_5ft_percentage = 50 →
  under_5ft_count = 30 →
  (female_percentage / 100) * (brunette_percentage / 100) * (under_5ft_percentage / 100) * total_students = under_5ft_count →
  female_percentage = 60 := by
sorry

end percentage_of_female_students_l1765_176579


namespace will_toy_cost_l1765_176592

def toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) : ℚ :=
  (initial_money - game_cost) / num_toys

theorem will_toy_cost :
  toy_cost 57 27 5 = 6 := by
  sorry

end will_toy_cost_l1765_176592


namespace simplify_expression_l1765_176520

theorem simplify_expression (y : ℝ) : 2 - (2 * (1 - (3 - (2 * (2 - y))))) = -2 + 4 * y := by
  sorry

end simplify_expression_l1765_176520


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1765_176563

/-- An isosceles triangle with side lengths 5 and 11 has a perimeter of 27. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 5 ∧ b = 11 ∧ c = 11) ∨ (a = 11 ∧ b = 5 ∧ c = 11) →
    IsoscelesTriangle a b c →
    a + b + c = 27
  where
    IsoscelesTriangle a b c := (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)

theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 5 11 11 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1765_176563


namespace trisection_dot_product_l1765_176505

/-- Given three points A, B, C in 2D space, and E, F as trisection points of BC,
    prove that the dot product of vectors AE and AF is 3. -/
theorem trisection_dot_product (A B C E F : ℝ × ℝ) : 
  A = (1, 2) →
  B = (2, -1) →
  C = (2, 2) →
  E = B + (1/3 : ℝ) • (C - B) →
  F = B + (2/3 : ℝ) • (C - B) →
  (E.1 - A.1) * (F.1 - A.1) + (E.2 - A.2) * (F.2 - A.2) = 3 := by
  sorry

#check trisection_dot_product

end trisection_dot_product_l1765_176505


namespace backpacks_sold_at_swap_meet_l1765_176557

/-- Prove that 17 backpacks were sold at the swap meet given the problem conditions -/
theorem backpacks_sold_at_swap_meet :
  ∀ (x : ℕ),
  (
    -- Total number of backpacks
    48 : ℕ
  ) = (
    -- Backpacks sold at swap meet
    x
  ) + (
    -- Backpacks sold to department store
    10 : ℕ
  ) + (
    -- Remaining backpacks
    48 - x - 10
  ) ∧
  (
    -- Total revenue
    1018 : ℕ
  ) = (
    -- Revenue from swap meet
    18 * x
  ) + (
    -- Revenue from department store
    10 * 25
  ) + (
    -- Revenue from remaining backpacks
    22 * (48 - x - 10)
  ) ∧
  (
    -- Total revenue
    1018 : ℕ
  ) = (
    -- Cost of backpacks
    576 : ℕ
  ) + (
    -- Profit
    442 : ℕ
  ) →
  x = 17 := by
  sorry


end backpacks_sold_at_swap_meet_l1765_176557


namespace bill_insurance_cost_l1765_176589

def monthly_plan_price : ℚ := 500
def hourly_rate : ℚ := 25
def weekly_hours : ℚ := 30
def weeks_per_month : ℚ := 4
def months_per_year : ℚ := 12

def annual_income (rate : ℚ) (hours : ℚ) (weeks : ℚ) (months : ℚ) : ℚ :=
  rate * hours * weeks * months

def subsidy_rate (income : ℚ) : ℚ :=
  if income < 10000 then 0.9
  else if income ≤ 40000 then 0.5
  else 0.2

def annual_insurance_cost (plan_price : ℚ) (income : ℚ) (months : ℚ) : ℚ :=
  plan_price * (1 - subsidy_rate income) * months

theorem bill_insurance_cost :
  let income := annual_income hourly_rate weekly_hours weeks_per_month months_per_year
  annual_insurance_cost monthly_plan_price income months_per_year = 3000 :=
by sorry

end bill_insurance_cost_l1765_176589


namespace total_divisors_xyz_l1765_176582

-- Define the variables and their properties
variable (p q r : ℕ) -- Natural numbers for primes
variable (hp : Prime p) -- p is prime
variable (hq : Prime q) -- q is prime
variable (hr : Prime r) -- r is prime
variable (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) -- p, q, and r are distinct

-- Define x, y, and z
def x : ℕ := p^2
def y : ℕ := q^2
def z : ℕ := r^4

-- State the theorem
theorem total_divisors_xyz (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  (Finset.card (Nat.divisors ((x p)^3 * (y q)^4 * (z r)^2))) = 567 := by
  sorry

end total_divisors_xyz_l1765_176582


namespace alisha_todd_ratio_l1765_176509

/-- Represents the number of gumballs given to each person and the total purchased -/
structure GumballDistribution where
  total : ℕ
  todd : ℕ
  alisha : ℕ
  bobby : ℕ
  remaining : ℕ

/-- Defines the conditions of the gumball distribution problem -/
def gumball_problem (g : GumballDistribution) : Prop :=
  g.total = 45 ∧
  g.todd = 4 ∧
  g.bobby = 4 * g.alisha - 5 ∧
  g.remaining = 6 ∧
  g.total = g.todd + g.alisha + g.bobby + g.remaining

/-- Theorem stating the ratio of gumballs given to Alisha vs Todd -/
theorem alisha_todd_ratio (g : GumballDistribution) 
  (h : gumball_problem g) : g.alisha = 2 * g.todd := by
  sorry


end alisha_todd_ratio_l1765_176509


namespace complex_root_modulus_l1765_176549

theorem complex_root_modulus (z : ℂ) : z^2 - 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_root_modulus_l1765_176549


namespace brian_cards_left_l1765_176511

/-- Given that Brian has 76 cards initially and Wayne takes 59 cards away,
    prove that Brian will have 17 cards left. -/
theorem brian_cards_left (initial_cards : ℕ) (cards_taken : ℕ) (cards_left : ℕ) : 
  initial_cards = 76 → cards_taken = 59 → cards_left = initial_cards - cards_taken → cards_left = 17 := by
  sorry

end brian_cards_left_l1765_176511


namespace estate_division_l1765_176501

theorem estate_division (total_estate : ℚ) : 
  total_estate > 0 → 
  ∃ (son_share mother_share daughter_share : ℚ),
    son_share = (4 : ℚ) / 7 * total_estate ∧
    mother_share = (2 : ℚ) / 7 * total_estate ∧
    daughter_share = (1 : ℚ) / 7 * total_estate ∧
    son_share + mother_share + daughter_share = total_estate ∧
    son_share = 2 * mother_share ∧
    mother_share = 2 * daughter_share :=
by sorry

end estate_division_l1765_176501


namespace complement_of_M_l1765_176545

/-- The complement of set M in the real numbers -/
theorem complement_of_M (x : ℝ) :
  x ∈ (Set.univ : Set ℝ) \ {x : ℝ | x^2 - 4 ≤ 0} ↔ x > 2 ∨ x < -2 := by
  sorry

end complement_of_M_l1765_176545


namespace g_zero_iff_a_eq_seven_fifths_l1765_176561

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(a) = 0 if and only if a = 7/5 -/
theorem g_zero_iff_a_eq_seven_fifths :
  ∀ a : ℝ, g a = 0 ↔ a = 7 / 5 := by
  sorry

end g_zero_iff_a_eq_seven_fifths_l1765_176561


namespace inverse_proportion_problem_l1765_176586

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  InverselyProportional x y →
  x + y = 40 →
  x - y = 8 →
  (∃ y' : ℝ, InverselyProportional 7 y' ∧ y' = 384 / 7) :=
by sorry

end inverse_proportion_problem_l1765_176586


namespace hyperbola_sum_l1765_176597

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 5 ∧ 
  k = 0 ∧ 
  c = 10 ∧ 
  a = 5 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 10 + 5 * Real.sqrt 3 := by
sorry

end hyperbola_sum_l1765_176597


namespace semicircle_perimeter_approx_l1765_176577

/-- The perimeter of a semicircle with radius 12 units is approximately 61.7 units. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 12
  let perimeter := 2 * r + π * r
  ∃ ε > 0, abs (perimeter - 61.7) < ε :=
by sorry

end semicircle_perimeter_approx_l1765_176577


namespace point_lies_on_graph_l1765_176542

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define a point lying on the graph of a function
def LiesOnGraph (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Theorem statement
theorem point_lies_on_graph (f : ℝ → ℝ) (a : ℝ) 
  (h : EvenFunction f) : LiesOnGraph f (-a) (f a) := by
  sorry

end point_lies_on_graph_l1765_176542


namespace equivalent_ratios_l1765_176539

theorem equivalent_ratios (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end equivalent_ratios_l1765_176539


namespace first_floor_rooms_l1765_176526

theorem first_floor_rooms : ∃ (x : ℕ), x > 0 ∧ 
  (6 * (x - 1) = 5 * x + 4) ∧ x = 10 := by
  sorry

end first_floor_rooms_l1765_176526


namespace pet_store_dogs_l1765_176547

/-- The number of dogs in a pet store with dogs and parakeets -/
def num_dogs : ℕ := 6

/-- The number of parakeets in the pet store -/
def num_parakeets : ℕ := 15 - num_dogs

/-- The total number of heads in the pet store -/
def total_heads : ℕ := 15

/-- The total number of feet in the pet store -/
def total_feet : ℕ := 42

theorem pet_store_dogs :
  num_dogs + num_parakeets = total_heads ∧
  4 * num_dogs + 2 * num_parakeets = total_feet :=
by sorry

end pet_store_dogs_l1765_176547


namespace max_sum_arithmetic_sequence_l1765_176504

theorem max_sum_arithmetic_sequence (x y : ℝ) (h : x^2 + y^2 = 4) :
  (∃ (z : ℝ), (3/4) * (x + 3*y) ≤ z) ∧ (∀ (z : ℝ), (3/4) * (x + 3*y) ≤ z → 3 * Real.sqrt 10 / 2 ≤ z) :=
by sorry

end max_sum_arithmetic_sequence_l1765_176504


namespace expression_evaluation_l1765_176566

theorem expression_evaluation (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = Real.sqrt 3 := by
  sorry

end expression_evaluation_l1765_176566


namespace planes_parallel_if_perpendicular_to_parallel_lines_l1765_176541

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeparallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β) :
  planeparallel α β :=
sorry

end planes_parallel_if_perpendicular_to_parallel_lines_l1765_176541


namespace quadratic_equal_roots_l1765_176593

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x - 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y - 2 * y + 12 = 0 → y = x) ↔ 
  (k = 10 ∨ k = -14) := by
sorry

end quadratic_equal_roots_l1765_176593


namespace ratio_multiple_choice_to_free_response_l1765_176565

/-- Represents the number of problems of each type in Stacy's homework assignment --/
structure HomeworkAssignment where
  total : Nat
  truefalse : Nat
  freeresponse : Nat
  multiplechoice : Nat

/-- Conditions for Stacy's homework assignment --/
def stacysHomework : HomeworkAssignment where
  total := 45
  truefalse := 6
  freeresponse := 13  -- 6 + 7
  multiplechoice := 26 -- 45 - (13 + 6)

theorem ratio_multiple_choice_to_free_response :
  (stacysHomework.multiplechoice : ℚ) / stacysHomework.freeresponse = 2 / 1 := by
  sorry

#check ratio_multiple_choice_to_free_response

end ratio_multiple_choice_to_free_response_l1765_176565


namespace function_characterization_l1765_176558

/-- A function from natural numbers to natural numbers. -/
def NatFunction := ℕ → ℕ

/-- The property that f(3x + 2y) = f(x)f(y) for all x, y ∈ ℕ. -/
def SatisfiesProperty (f : NatFunction) : Prop :=
  ∀ x y : ℕ, f (3 * x + 2 * y) = f x * f y

/-- The constant zero function. -/
def ZeroFunction : NatFunction := λ _ => 0

/-- The constant one function. -/
def OneFunction : NatFunction := λ _ => 1

/-- The function that is 1 at 0 and 0 elsewhere. -/
def ZeroOneFunction : NatFunction := λ n => if n = 0 then 1 else 0

/-- The main theorem stating that any function satisfying the property
    must be one of the three specified functions. -/
theorem function_characterization (f : NatFunction) 
  (h : SatisfiesProperty f) : 
  f = ZeroFunction ∨ f = OneFunction ∨ f = ZeroOneFunction :=
sorry

end function_characterization_l1765_176558


namespace max_elephants_l1765_176588

/-- The number of union members --/
def unionMembers : ℕ := 28

/-- The number of non-members --/
def nonMembers : ℕ := 37

/-- The total number of attendees --/
def totalAttendees : ℕ := unionMembers + nonMembers

/-- A function to check if a distribution is valid --/
def isValidDistribution (elephants : ℕ) : Prop :=
  ∃ (unionElephants nonUnionElephants : ℕ),
    elephants = unionElephants + nonUnionElephants ∧
    unionElephants % unionMembers = 0 ∧
    nonUnionElephants % nonMembers = 0 ∧
    unionElephants / unionMembers ≥ 1 ∧
    nonUnionElephants / nonMembers ≥ 1

/-- The theorem stating the maximum number of elephants --/
theorem max_elephants :
  ∃! (maxElephants : ℕ),
    isValidDistribution maxElephants ∧
    ∀ (n : ℕ), n > maxElephants → ¬isValidDistribution n :=
by
  sorry

end max_elephants_l1765_176588


namespace water_displaced_by_sphere_l1765_176559

/-- The volume of water displaced by a completely submerged sphere -/
theorem water_displaced_by_sphere (diameter : ℝ) (volume_displaced : ℝ) :
  diameter = 8 →
  volume_displaced = (4/3) * Real.pi * (diameter/2)^3 →
  volume_displaced = (256/3) * Real.pi := by
  sorry

end water_displaced_by_sphere_l1765_176559


namespace factors_of_M_l1765_176536

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 7^1 · 11^2 -/
def num_factors (M : ℕ) : ℕ :=
  if M = 2^4 * 3^3 * 7^1 * 11^2 then 120 else 0

/-- Theorem stating that the number of natural-number factors of M is 120 -/
theorem factors_of_M :
  ∀ M : ℕ, M = 2^4 * 3^3 * 7^1 * 11^2 → num_factors M = 120 := by
  sorry

end factors_of_M_l1765_176536


namespace least_factorial_divisible_by_7350_l1765_176551

theorem least_factorial_divisible_by_7350 : ∃ (n : ℕ), n > 0 ∧ 7350 ∣ n.factorial ∧ ∀ (m : ℕ), m > 0 → 7350 ∣ m.factorial → n ≤ m :=
  sorry

end least_factorial_divisible_by_7350_l1765_176551


namespace fish_population_calculation_l1765_176503

/-- Calculates the number of fish in a lake on May 1 based on sampling data --/
theorem fish_population_calculation (tagged_may : ℕ) (caught_sept : ℕ) (tagged_sept : ℕ)
  (death_rate : ℚ) (new_fish_rate : ℚ) 
  (h1 : tagged_may = 60)
  (h2 : caught_sept = 70)
  (h3 : tagged_sept = 3)
  (h4 : death_rate = 1/4)
  (h5 : new_fish_rate = 2/5)
  (h6 : tagged_sept ≤ caught_sept) :
  ∃ (fish_may : ℕ), fish_may = 840 := by
  sorry

end fish_population_calculation_l1765_176503


namespace age_difference_l1765_176576

/-- Given two people A and B, where B is currently 37 years old,
    and in 10 years A will be twice as old as B was 10 years ago,
    prove that A is currently 7 years older than B. -/
theorem age_difference (a b : ℕ) (h1 : b = 37) 
    (h2 : a + 10 = 2 * (b - 10)) : a - b = 7 := by
  sorry

end age_difference_l1765_176576


namespace total_flowers_l1765_176532

theorem total_flowers (class_a_students class_b_students flowers_per_student : ℕ) 
  (h1 : class_a_students = 48)
  (h2 : class_b_students = 48)
  (h3 : flowers_per_student = 16) :
  class_a_students * flowers_per_student + class_b_students * flowers_per_student = 1536 := by
  sorry

end total_flowers_l1765_176532


namespace annie_cookies_count_l1765_176553

/-- The number of cookies Annie ate on Monday -/
def monday_cookies : ℕ := 5

/-- The number of cookies Annie ate on Tuesday -/
def tuesday_cookies : ℕ := 2 * monday_cookies

/-- The number of cookies Annie ate on Wednesday -/
def wednesday_cookies : ℕ := tuesday_cookies + (tuesday_cookies * 2 / 5)

/-- The total number of cookies Annie ate during the three days -/
def total_cookies : ℕ := monday_cookies + tuesday_cookies + wednesday_cookies

theorem annie_cookies_count : total_cookies = 29 := by
  sorry

end annie_cookies_count_l1765_176553


namespace sqrt_sum_irrational_l1765_176522

theorem sqrt_sum_irrational (a b : ℚ) 
  (ha : Irrational (Real.sqrt a)) 
  (hb : Irrational (Real.sqrt b)) : 
  Irrational (Real.sqrt a + Real.sqrt b) := by
  sorry

end sqrt_sum_irrational_l1765_176522


namespace sequence_formulas_l1765_176568

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, b (n + 1) = q * b n

theorem sequence_formulas
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a235 : a 2 + a 3 = a 5)
  (h_a4b12 : a 4 = 4 * b 1 - b 2)
  (h_b3a35 : b 3 = a 3 + a 5) :
  (∀ n, a n = n) ∧ (∀ n, b n = 2^n) :=
sorry

end sequence_formulas_l1765_176568
