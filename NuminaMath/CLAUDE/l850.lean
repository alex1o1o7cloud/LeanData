import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l850_85074

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- Theorem: For a hyperbola with given asymptotes and passing through a specific point, a + h = 16/3 -/
theorem hyperbola_a_plus_h_value (H : Hyperbola) 
  (asymptote1 : ∀ x y : ℝ, y = 3*x + 3 → (∀ t : ℝ, (y - H.k)^2/(H.a^2) - (x - H.h)^2/(H.b^2) = t))
  (asymptote2 : ∀ x y : ℝ, y = -3*x - 1 → (∀ t : ℝ, (y - H.k)^2/(H.a^2) - (x - H.h)^2/(H.b^2) = t))
  (point_on_hyperbola : (11 - H.k)^2/(H.a^2) - (2 - H.h)^2/(H.b^2) = 1) :
  H.a + H.h = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l850_85074


namespace NUMINAMATH_CALUDE_train_passing_time_l850_85036

/-- The time taken for two trains to completely pass each other -/
theorem train_passing_time (length_A length_B : ℝ) (speed_A speed_B : ℝ) 
  (h1 : length_A = 150)
  (h2 : length_B = 150)
  (h3 : speed_A = 54 * (5/18))
  (h4 : speed_B = 36 * (5/18))
  (h5 : speed_A > 0)
  (h6 : speed_B > 0) :
  (length_A + length_B) / (speed_A + speed_B) = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l850_85036


namespace NUMINAMATH_CALUDE_comparison_and_inequality_l850_85071

theorem comparison_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + b^2 ≥ 2*(2*a - b) - 5 ∧ 
  a^a * b^b ≥ (a*b)^((a+b)/2) ∧ 
  (a^a * b^b = (a*b)^((a+b)/2) ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_comparison_and_inequality_l850_85071


namespace NUMINAMATH_CALUDE_quadratic_equation_equal_coefficients_l850_85066

/-- A quadratic equation with coefficients forming an arithmetic sequence and reciprocal roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_reciprocal : ∃ (r s : ℝ), r * s = 1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0
  coeff_arithmetic : b - a = c - b

/-- The coefficients of a quadratic equation with reciprocal roots and coefficients in arithmetic sequence are equal -/
theorem quadratic_equation_equal_coefficients (eq : QuadraticEquation) : eq.a = eq.b ∧ eq.b = eq.c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equal_coefficients_l850_85066


namespace NUMINAMATH_CALUDE_equation_solutions_may_days_l850_85081

-- Define the interval [0°, 360°]
def angle_interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 360}

-- Define the equation cos³α - cosα = 0
def equation (α : ℝ) : Prop := Real.cos α ^ 3 - Real.cos α = 0

-- Define the day of the week as an enumeration
inductive DayOfWeek
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to get the day of the week for a given day in May
def day_in_may (day : ℕ) : DayOfWeek := sorry

-- Theorem 1: There are exactly 5 values of α in [0°, 360°] that satisfy cos³α - cosα = 0
theorem equation_solutions :
  ∃ (S : Finset ℝ), S.card = 5 ∧ (∀ α ∈ S, α ∈ angle_interval ∧ equation α) ∧
    (∀ α, α ∈ angle_interval → equation α → α ∈ S) :=
  sorry

-- Theorem 2: If the 5th day of May is Thursday, then the 16th day of May is Monday
theorem may_days :
  day_in_may 5 = DayOfWeek.Thursday → day_in_may 16 = DayOfWeek.Monday :=
  sorry

end NUMINAMATH_CALUDE_equation_solutions_may_days_l850_85081


namespace NUMINAMATH_CALUDE_points_per_bag_l850_85002

theorem points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) : 
  total_bags = 11 → unrecycled_bags = 2 → total_points = 45 → 
  (total_points / (total_bags - unrecycled_bags) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_bag_l850_85002


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l850_85091

theorem arithmetic_mean_geq_geometric_mean {x y : ℝ} (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l850_85091


namespace NUMINAMATH_CALUDE_elevator_stop_time_is_three_l850_85073

/-- Represents the race to the top of a building --/
structure BuildingRace where
  stories : ℕ
  lola_time_per_story : ℕ
  elevator_time_per_story : ℕ
  total_time : ℕ

/-- Calculates the time the elevator stops on each floor --/
def elevator_stop_time (race : BuildingRace) : ℕ :=
  let lola_total_time := race.stories * race.lola_time_per_story
  let elevator_move_time := race.stories * race.elevator_time_per_story
  let total_stop_time := race.total_time - elevator_move_time
  total_stop_time / (race.stories - 1)

/-- The theorem stating that the elevator stops for 3 seconds on each floor --/
theorem elevator_stop_time_is_three (race : BuildingRace) 
    (h1 : race.stories = 20)
    (h2 : race.lola_time_per_story = 10)
    (h3 : race.elevator_time_per_story = 8)
    (h4 : race.total_time = 220) :
  elevator_stop_time race = 3 := by
  sorry

#eval elevator_stop_time { stories := 20, lola_time_per_story := 10, elevator_time_per_story := 8, total_time := 220 }

end NUMINAMATH_CALUDE_elevator_stop_time_is_three_l850_85073


namespace NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_l850_85062

/-- A convex octagon -/
structure ConvexOctagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of angles in an octagon -/
def num_angles : ℕ := 8

/-- The sum of exterior angles in any polygon -/
def sum_exterior_angles : ℕ := 360

/-- Theorem: In a convex octagon, the minimum number of obtuse interior angles is 5 -/
theorem min_obtuse_angles_convex_octagon (O : ConvexOctagon) : 
  ∃ (n : ℕ), n ≥ 5 ∧ n = (num_angles - (sum_exterior_angles / 90)) := by
  sorry

end NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_l850_85062


namespace NUMINAMATH_CALUDE_apple_pies_count_l850_85056

theorem apple_pies_count (pecan_pies : ℕ) (total_rows : ℕ) (pies_per_row : ℕ) : 
  pecan_pies = 16 →
  total_rows = 6 →
  pies_per_row = 5 →
  ∃ (apple_pies : ℕ), apple_pies = total_rows * pies_per_row - pecan_pies ∧ apple_pies = 14 := by
  sorry

end NUMINAMATH_CALUDE_apple_pies_count_l850_85056


namespace NUMINAMATH_CALUDE_inequality_proof_l850_85012

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 9 / 4) : 
  a^3 + b^3 + c^3 > a * Real.sqrt (b + c) + b * Real.sqrt (c + a) + c * Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l850_85012


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l850_85049

theorem solution_to_linear_equation :
  let x : ℝ := 4
  let y : ℝ := 2
  2 * x - y = 6 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l850_85049


namespace NUMINAMATH_CALUDE_carnation_percentage_l850_85080

/-- Represents a floral arrangement with different types of flowers -/
structure FloralArrangement where
  total : ℕ
  pink_roses : ℕ
  red_roses : ℕ
  white_roses : ℕ
  pink_carnations : ℕ
  red_carnations : ℕ
  white_carnations : ℕ

/-- Conditions for the floral arrangement -/
def valid_arrangement (f : FloralArrangement) : Prop :=
  -- Half of the pink flowers are roses
  f.pink_roses = f.pink_carnations
  -- One-third of the red flowers are carnations
  ∧ 3 * f.red_carnations = f.red_roses + f.red_carnations
  -- Three-fifths of the flowers are pink
  ∧ 5 * (f.pink_roses + f.pink_carnations) = 3 * f.total
  -- Total flowers equals sum of all flower types
  ∧ f.total = f.pink_roses + f.red_roses + f.white_roses + 
              f.pink_carnations + f.red_carnations + f.white_carnations

/-- Theorem: The percentage of carnations in a valid floral arrangement is 50% -/
theorem carnation_percentage (f : FloralArrangement) 
  (h : valid_arrangement f) : 
  (f.pink_carnations + f.red_carnations + f.white_carnations) * 2 = f.total := by
  sorry

#check carnation_percentage

end NUMINAMATH_CALUDE_carnation_percentage_l850_85080


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l850_85077

/-- The trajectory of the midpoint of a line segment with one fixed endpoint and the other moving on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, (x₀ + 1)^2 + y₀^2 = 4 ∧ x = (x₀ + 4)/2 ∧ y = (y₀ + 3)/2) → 
  (x - 3/2)^2 + (y - 3/2)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l850_85077


namespace NUMINAMATH_CALUDE_taxi_distance_calculation_l850_85008

/-- Calculates the total distance of a taxi ride given the fare structure and total charge --/
theorem taxi_distance_calculation (initial_charge : ℚ) (initial_distance : ℚ) 
  (additional_charge : ℚ) (additional_distance : ℚ) (total_charge : ℚ) :
  initial_charge = 2.5 →
  initial_distance = 1/5 →
  additional_charge = 0.4 →
  additional_distance = 1/5 →
  total_charge = 18.1 →
  ∃ (total_distance : ℚ), total_distance = 8 ∧
    total_charge = initial_charge + 
      (total_distance - initial_distance) / additional_distance * additional_charge :=
by
  sorry


end NUMINAMATH_CALUDE_taxi_distance_calculation_l850_85008


namespace NUMINAMATH_CALUDE_martha_butterflies_l850_85033

def butterfly_collection (blue yellow black : ℕ) : Prop :=
  blue = 2 * yellow ∧ black = 5 ∧ blue = 4

theorem martha_butterflies :
  ∀ blue yellow black : ℕ,
  butterfly_collection blue yellow black →
  blue + yellow + black = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_butterflies_l850_85033


namespace NUMINAMATH_CALUDE_nonzero_y_solution_l850_85065

theorem nonzero_y_solution (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_y_solution_l850_85065


namespace NUMINAMATH_CALUDE_total_animals_hunted_l850_85029

theorem total_animals_hunted (sam rob mark peter : ℕ) : 
  sam = 6 →
  rob = sam / 2 →
  mark = (sam + rob) / 3 →
  peter = 3 * mark →
  sam + rob + mark + peter = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_total_animals_hunted_l850_85029


namespace NUMINAMATH_CALUDE_f_two_roots_range_l850_85095

/-- The cubic function f(x) = x^3 - 3x + 5 -/
def f (x : ℝ) : ℝ := x^3 - 3*x + 5

/-- Theorem stating the range of a for which f(x) = a has at least two distinct real roots -/
theorem f_two_roots_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f x = a ∧ f y = a) ↔ 3 ≤ a ∧ a ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_f_two_roots_range_l850_85095


namespace NUMINAMATH_CALUDE_elephant_count_theorem_l850_85013

/-- The total number of elephants in two parks, given the number in one park
    and a multiplier for the other park. -/
def total_elephants (park1_count : ℕ) (multiplier : ℕ) : ℕ :=
  park1_count + multiplier * park1_count

/-- Theorem stating that the total number of elephants in two parks is 280,
    given that one park has 70 elephants and the other has 3 times as many. -/
theorem elephant_count_theorem :
  total_elephants 70 3 = 280 := by
  sorry

end NUMINAMATH_CALUDE_elephant_count_theorem_l850_85013


namespace NUMINAMATH_CALUDE_judes_current_age_jude_is_two_years_old_l850_85063

/-- Proves Jude's current age given Heath's current age and their future age relationship -/
theorem judes_current_age (heath_current_age : ℕ) (future_years : ℕ) (future_age_ratio : ℕ) : ℕ :=
  let heath_future_age := heath_current_age + future_years
  let jude_future_age := heath_future_age / future_age_ratio
  let age_difference := heath_future_age - jude_future_age
  heath_current_age - age_difference

/-- The main theorem that proves Jude's current age is 2 years old -/
theorem jude_is_two_years_old : judes_current_age 16 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_judes_current_age_jude_is_two_years_old_l850_85063


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l850_85039

/-- The equation x³ - x - 2/(3√3) = 0 has exactly three real roots: 2/√3, -1/√3, and -1/√3 -/
theorem cubic_equation_roots :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^3 - x - 2/(3*Real.sqrt 3) = 0 ∧ 
  (2/Real.sqrt 3 ∈ s ∧ -1/Real.sqrt 3 ∈ s) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l850_85039


namespace NUMINAMATH_CALUDE_quadratic_function_range_l850_85021

theorem quadratic_function_range (a b : ℝ) : 
  (∀ x ∈ Set.Ioo 2 5, a * x^2 + b * x + 2 > 0) →
  (a * 1^2 + b * 1 + 2 = 1) →
  a > 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l850_85021


namespace NUMINAMATH_CALUDE_football_points_sum_l850_85041

theorem football_points_sum : 
  let zach_points : Float := 42.0
  let ben_points : Float := 21.0
  let sarah_points : Float := 18.5
  let emily_points : Float := 27.5
  zach_points + ben_points + sarah_points + emily_points = 109.0 := by
  sorry

end NUMINAMATH_CALUDE_football_points_sum_l850_85041


namespace NUMINAMATH_CALUDE_executive_board_selection_l850_85010

theorem executive_board_selection (n : ℕ) (r : ℕ) : n = 12 ∧ r = 5 → Nat.choose n r = 792 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_selection_l850_85010


namespace NUMINAMATH_CALUDE_correct_calculation_result_proof_correct_calculation_l850_85052

theorem correct_calculation_result : ℤ → Prop :=
  fun x => (x + 9 = 30) → (x - 7 = 14)

-- The proof is omitted
theorem proof_correct_calculation : correct_calculation_result 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_proof_correct_calculation_l850_85052


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l850_85083

theorem quadratic_inequality_solution (a b : ℝ) 
  (h1 : (1 : ℝ) / 3 * 1 = -1 / a) 
  (h2 : (1 : ℝ) / 3 + 1 = -b / a) 
  (h3 : a < 0) : 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l850_85083


namespace NUMINAMATH_CALUDE_insect_jumps_l850_85096

theorem insect_jumps (s : ℝ) (h_s : 1/2 < s ∧ s < 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) :
  ∀ ε > 0, ∃ (n : ℕ) (x : ℕ → ℝ),
    (x 0 = 0 ∨ x 0 = 1) ∧
    (∀ i, i < n → (x (i + 1) = x i * s ∨ x (i + 1) = (x i - 1) * s + 1)) ∧
    |x n - c| < ε :=
by sorry

end NUMINAMATH_CALUDE_insect_jumps_l850_85096


namespace NUMINAMATH_CALUDE_cost_price_calculation_l850_85047

/-- Proves that if an article is sold for $1200 with a 20% profit, then the cost price is $1000. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 1200 ∧ profit_percentage = 20 →
  (selling_price = (100 + profit_percentage) / 100 * 1000) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l850_85047


namespace NUMINAMATH_CALUDE_skating_time_for_average_l850_85068

def minutes_per_day_1 : ℕ := 80
def days_1 : ℕ := 4
def minutes_per_day_2 : ℕ := 105
def days_2 : ℕ := 3
def total_days : ℕ := 8
def target_average : ℕ := 95

theorem skating_time_for_average :
  (minutes_per_day_1 * days_1 + minutes_per_day_2 * days_2 + 125) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_skating_time_for_average_l850_85068


namespace NUMINAMATH_CALUDE_probability_of_specific_combination_l850_85022

def shirts : ℕ := 6
def shorts : ℕ := 7
def socks : ℕ := 8
def hats : ℕ := 3
def total_items : ℕ := shirts + shorts + socks + hats
def items_chosen : ℕ := 4

theorem probability_of_specific_combination :
  (shirts.choose 1 * shorts.choose 1 * socks.choose 1 * hats.choose 1) / total_items.choose items_chosen = 144 / 1815 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_combination_l850_85022


namespace NUMINAMATH_CALUDE_two_lines_at_45_degrees_l850_85099

/-- The equation represents two lines that intersect at a 45° angle when k = 80 -/
theorem two_lines_at_45_degrees (x y : ℝ) :
  let k : ℝ := 80
  let equation := x^2 + x*y - 6*y^2 - 20*x - 20*y + k
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, equation = 0 ↔ (l₁ x y ∨ l₂ x y)) ∧
    (∃ x₀ y₀, l₁ x₀ y₀ ∧ l₂ x₀ y₀) ∧
    (∀ x₀ y₀, l₁ x₀ y₀ ∧ l₂ x₀ y₀ → 
      ∃ (v₁ v₂ : ℝ × ℝ),
        (v₁.1 ≠ 0 ∨ v₁.2 ≠ 0) ∧
        (v₂.1 ≠ 0 ∨ v₂.2 ≠ 0) ∧
        (v₁.1 * v₂.1 + v₁.2 * v₂.2) / (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = Real.cos (π/4)) :=
by sorry


end NUMINAMATH_CALUDE_two_lines_at_45_degrees_l850_85099


namespace NUMINAMATH_CALUDE_factor_x10_minus_1024_l850_85085

theorem factor_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x-2)*(x+2)*(x^4 + 2*x^3 + 4*x^2 + 8*x + 16)*(x^4 - 2*x^3 + 4*x^2 - 8*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x10_minus_1024_l850_85085


namespace NUMINAMATH_CALUDE_cubic_expression_value_l850_85015

theorem cubic_expression_value (r s : ℝ) : 
  3 * r^2 - 4 * r - 7 = 0 →
  3 * s^2 - 4 * s - 7 = 0 →
  r ≠ s →
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l850_85015


namespace NUMINAMATH_CALUDE_platform_length_platform_length_approx_l850_85082

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) : ℝ :=
  let train_speed := train_length / pole_time
  let platform_length := train_speed * platform_time - train_length
  platform_length

/-- The platform length is approximately 300 meters -/
theorem platform_length_approx :
  let result := platform_length 300 36 18
  ∃ ε > 0, abs (result - 300) < ε :=
sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_approx_l850_85082


namespace NUMINAMATH_CALUDE_not_adjacent_in_sorted_consecutive_l850_85089

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sorted_by_digit_sum (a b : ℕ) : Prop :=
  (sum_of_digits a < sum_of_digits b) ∨ 
  (sum_of_digits a = sum_of_digits b ∧ a ≤ b)

theorem not_adjacent_in_sorted_consecutive (start : ℕ) : 
  ¬ ∃ i : ℕ, i < 99 ∧ 
    (sorted_by_digit_sum (start + i) 2010 ∧ sorted_by_digit_sum 2010 2011 ∧ sorted_by_digit_sum 2011 (start + (i + 1))) ∨
    (sorted_by_digit_sum (start + i) 2011 ∧ sorted_by_digit_sum 2011 2010 ∧ sorted_by_digit_sum 2010 (start + (i + 1))) :=
sorry

end NUMINAMATH_CALUDE_not_adjacent_in_sorted_consecutive_l850_85089


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l850_85067

theorem quadratic_roots_relation (m n p : ℝ) : 
  m ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ s₁ s₂ : ℝ, (s₁ * s₂ = m) ∧ 
               (s₁ + s₂ = -p) ∧
               ((3 * s₁) * (3 * s₂) = n)) →
  n / p = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l850_85067


namespace NUMINAMATH_CALUDE_interest_calculation_time_l850_85069

-- Define the given values
def simple_interest : ℚ := 345/100
def principal : ℚ := 23
def rate_paise : ℚ := 5

-- Convert rate from paise to rupees
def rate : ℚ := rate_paise / 100

-- Define the simple interest formula
def calculate_time (si p r : ℚ) : ℚ := si / (p * r)

-- State the theorem
theorem interest_calculation_time :
  calculate_time simple_interest principal rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_time_l850_85069


namespace NUMINAMATH_CALUDE_subtract_problem_l850_85046

theorem subtract_problem (x : ℕ) (h : 913 - x = 514) : 514 - x = 115 := by
  sorry

end NUMINAMATH_CALUDE_subtract_problem_l850_85046


namespace NUMINAMATH_CALUDE_diamond_eight_five_l850_85040

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := (a + b) * ((a - b)^2)

-- Theorem statement
theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end NUMINAMATH_CALUDE_diamond_eight_five_l850_85040


namespace NUMINAMATH_CALUDE_months_C_is_three_l850_85090

/-- Represents the number of months C put his oxen for grazing -/
def months_C : ℕ := sorry

/-- Total rent of the pasture in rupees -/
def total_rent : ℕ := 280

/-- Number of oxen A put for grazing -/
def oxen_A : ℕ := 10

/-- Number of months A put his oxen for grazing -/
def months_A : ℕ := 7

/-- Number of oxen B put for grazing -/
def oxen_B : ℕ := 12

/-- Number of months B put his oxen for grazing -/
def months_B : ℕ := 5

/-- Number of oxen C put for grazing -/
def oxen_C : ℕ := 15

/-- C's share of rent in rupees -/
def rent_C : ℕ := 72

/-- Theorem stating that C put his oxen for grazing for 3 months -/
theorem months_C_is_three : months_C = 3 := by sorry

end NUMINAMATH_CALUDE_months_C_is_three_l850_85090


namespace NUMINAMATH_CALUDE_max_product_constraint_l850_85079

theorem max_product_constraint (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a + b = 4) :
  (a - 1) * (b - 1) ≤ 1 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 1 ∧ b₀ > 1 ∧ a₀ + b₀ = 4 ∧ (a₀ - 1) * (b₀ - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l850_85079


namespace NUMINAMATH_CALUDE_front_view_length_l850_85060

theorem front_view_length
  (body_diagonal : ℝ)
  (side_view : ℝ)
  (top_view : ℝ)
  (h1 : body_diagonal = 5 * Real.sqrt 2)
  (h2 : side_view = 5)
  (h3 : top_view = Real.sqrt 34) :
  ∃ front_view : ℝ,
    front_view = Real.sqrt 41 ∧
    side_view ^ 2 + top_view ^ 2 + front_view ^ 2 = body_diagonal ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_front_view_length_l850_85060


namespace NUMINAMATH_CALUDE_positive_real_inequality_l850_85055

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l850_85055


namespace NUMINAMATH_CALUDE_minimize_reciprocal_sum_l850_85020

theorem minimize_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (1 / a + 4 / b ≥ 8 / 15) ∧
  (1 / a + 4 / b = 8 / 15 ↔ a = 15 / 4 ∧ b = 15) := by
sorry

end NUMINAMATH_CALUDE_minimize_reciprocal_sum_l850_85020


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l850_85076

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) : 
  x^2 + y^2 = 458 := by sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l850_85076


namespace NUMINAMATH_CALUDE_sum_and_difference_bounds_l850_85037

theorem sum_and_difference_bounds (a b : ℝ) 
  (ha : 60 ≤ a ∧ a ≤ 84) (hb : 28 ≤ b ∧ b ≤ 33) : 
  (88 ≤ a + b ∧ a + b ≤ 117) ∧ (27 ≤ a - b ∧ a - b ≤ 56) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_difference_bounds_l850_85037


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l850_85072

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z * (Complex.I + 1) = 2 / (Complex.I - 1)) : 
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l850_85072


namespace NUMINAMATH_CALUDE_probability_sum_10_l850_85061

def die_faces : Nat := 6

def total_outcomes : Nat := die_faces ^ 3

def favorable_outcomes : Nat := 30

theorem probability_sum_10 : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_10_l850_85061


namespace NUMINAMATH_CALUDE_system_solution_l850_85031

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, a * x - y = 4 ∧ 3 * x + b * y = 4 ∧ x = 2 ∧ y = -2) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l850_85031


namespace NUMINAMATH_CALUDE_platform_length_calculation_l850_85048

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is
    approximately 350.13 meters. -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ (platform_length : ℝ), abs (platform_length - 350.13) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l850_85048


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus2_l850_85024

theorem rationalize_denominator_sqrt3_minus2 :
  1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus2_l850_85024


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_l850_85028

def consecutive_sum (n : ℕ) : ℕ := 10 * (2 * n + 19)

theorem smallest_perfect_square_sum :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), consecutive_sum m = k^2) ∧
    (∃ (k : ℕ), consecutive_sum n = k^2) ∧
    consecutive_sum n = 1000 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_l850_85028


namespace NUMINAMATH_CALUDE_walking_meeting_point_l850_85011

/-- Represents the meeting of two people walking towards each other --/
theorem walking_meeting_point (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) 
  (deceleration_a : ℝ) (acceleration_b : ℝ) (h : ℕ) :
  total_distance = 100 ∧ 
  speed_a = 5 ∧ 
  speed_b = 4 ∧ 
  deceleration_a = 0.4 ∧ 
  acceleration_b = 0.5 →
  (h : ℝ) * (2 * speed_a - (h - 1) * deceleration_a) / 2 + 
  (h : ℝ) * (2 * speed_b + (h - 1) * acceleration_b) / 2 = total_distance ∧ 
  (h : ℝ) * (2 * speed_a - (h - 1) * deceleration_a) / 2 = 
  total_distance / 2 - 31 := by
  sorry

#check walking_meeting_point

end NUMINAMATH_CALUDE_walking_meeting_point_l850_85011


namespace NUMINAMATH_CALUDE_sequence_formula_l850_85057

theorem sequence_formula (n : ℕ+) (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h1 : ∀ k, S k = a k / 2 + 1 / a k - 1)
  (h2 : ∀ k, a k > 0) :
  a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l850_85057


namespace NUMINAMATH_CALUDE_part_one_solution_part_two_solution_l850_85092

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + a - 4| + x + 1

-- Part I
theorem part_one_solution :
  let a : ℝ := 2
  ∀ x : ℝ, f a x < 9 ↔ -6 < x ∧ x < 10/3 :=
sorry

-- Part II
theorem part_two_solution :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 2 → f a x ≤ (x + 2)^2) ↔ -3 ≤ a ∧ a ≤ 17/3 :=
sorry

end NUMINAMATH_CALUDE_part_one_solution_part_two_solution_l850_85092


namespace NUMINAMATH_CALUDE_correct_inequalities_l850_85032

/-- 
Given a student's estimated scores in Chinese and Mathematics after a mock final exam,
this theorem proves that the correct system of inequalities representing the situation is
x > 85 and y ≥ 80, where x is the Chinese score and y is the Mathematics score.
-/
theorem correct_inequalities (x y : ℝ) 
  (h1 : x > 85)  -- Chinese score is higher than 85 points
  (h2 : y ≥ 80)  -- Mathematics score is not less than 80 points
  : x > 85 ∧ y ≥ 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_inequalities_l850_85032


namespace NUMINAMATH_CALUDE_water_added_to_bowl_l850_85044

theorem water_added_to_bowl (C : ℝ) (h1 : C > 0) : 
  (C / 2 + (14 - C / 2) = 0.7 * C) → (14 - C / 2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_bowl_l850_85044


namespace NUMINAMATH_CALUDE_solution_set_min_value_g_inequality_proof_l850_85034

-- Define the absolute value function
def f (x : ℝ) : ℝ := |x|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Statement for part 1
theorem solution_set (x : ℝ) :
  f ((1 / 2^x) - 2) ≤ 1 ↔ x ∈ Set.Icc (Real.log 3 / Real.log 2) 0 :=
sorry

-- Statement for part 2
theorem min_value_g :
  ∃ (m : ℝ), m = 1 ∧ ∀ x, g x ≥ m :=
sorry

-- Statement for part 3
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_min_value_g_inequality_proof_l850_85034


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l850_85075

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem perpendicular_vectors (x : ℝ) : 
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l850_85075


namespace NUMINAMATH_CALUDE_system_solutions_l850_85058

theorem system_solutions (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^2 / (1 + x₁^2) = x₂ ∧ 
   2 * x₂^2 / (1 + x₂^2) = x₃ ∧ 
   2 * x₃^2 / (1 + x₃^2) = x₁) ↔ 
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ 
   (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l850_85058


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l850_85097

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (1 + I) = a + b*I → a + b = 2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l850_85097


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l850_85094

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first, third, and fifth terms is 9. -/
def SumOdd (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 = 9

/-- The sum of the second, fourth, and sixth terms is 15. -/
def SumEven (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 15

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SumOdd a) 
  (h3 : SumEven a) : 
  a 3 + a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l850_85094


namespace NUMINAMATH_CALUDE_parallelogram_uniqueness_l850_85017

/-- Represents a parallelogram in 2D space -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- The measure of an angle in radians -/
def Angle := ℝ

/-- The length of a line segment -/
def Length := ℝ

/-- Checks if two parallelograms are congruent -/
def are_congruent (p1 p2 : Parallelogram) : Prop :=
  sorry

/-- Constructs a parallelogram given the required parameters -/
def construct_parallelogram (α ε : Angle) (bd : Length) : Parallelogram :=
  sorry

/-- Theorem stating the uniqueness of the constructed parallelogram -/
theorem parallelogram_uniqueness (α ε : Angle) (bd : Length) :
  ∀ p1 p2 : Parallelogram,
    (p1 = construct_parallelogram α ε bd) →
    (p2 = construct_parallelogram α ε bd) →
    are_congruent p1 p2 :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_uniqueness_l850_85017


namespace NUMINAMATH_CALUDE_max_value_expression_l850_85009

theorem max_value_expression (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : x^2 = 1) : 
  ∃ (m : ℝ), ∀ y, y^2 = 1 → x^2 + a + b + c * d * x ≤ m ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l850_85009


namespace NUMINAMATH_CALUDE_not_proportional_l850_85023

-- Define the notion of direct proportionality
def is_directly_proportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, y t = k * x t

-- Define the notion of inverse proportionality
def is_inversely_proportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

-- Define our equation
def our_equation (x y : ℝ) : Prop :=
  2 * x + 3 * y = 6

-- Theorem statement
theorem not_proportional :
  ¬ (∃ x y : ℝ → ℝ, (∀ t : ℝ, our_equation (x t) (y t)) ∧
    (is_directly_proportional x y ∨ is_inversely_proportional x y)) :=
sorry

end NUMINAMATH_CALUDE_not_proportional_l850_85023


namespace NUMINAMATH_CALUDE_no_self_inverse_plus_one_function_l850_85087

theorem no_self_inverse_plus_one_function : ¬∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_self_inverse_plus_one_function_l850_85087


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l850_85014

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l850_85014


namespace NUMINAMATH_CALUDE_set_intersection_equals_interval_l850_85051

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
def N : Set ℝ := {x | x > 0 ∧ x ≠ 1}

-- Define the interval (0,1) ∪ (1,2]
def interval : Set ℝ := {x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)}

-- State the theorem
theorem set_intersection_equals_interval : M ∩ N = interval := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equals_interval_l850_85051


namespace NUMINAMATH_CALUDE_min_value_a_plus_5b_l850_85093

theorem min_value_a_plus_5b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b + b^2 = b + 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x * y + y^2 = y + 1 → a + 5 * b ≤ x + 5 * y ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x * y + y^2 = y + 1 ∧ x + 5 * y = 7/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_5b_l850_85093


namespace NUMINAMATH_CALUDE_sequence_and_inequality_problem_l850_85004

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def positive_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0

theorem sequence_and_inequality_problem
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_pos : positive_sequence b)
  (h_a1 : a 1 = 2)
  (h_b1 : b 1 = 3)
  (h_sum1 : a 3 + b 5 = 56)
  (h_sum2 : a 5 + b 3 = 26)
  (h_ineq : ∀ n : ℕ, n > 0 → ∀ x : ℝ, -x^2 + 3*x ≤ (2 * b n) / (2 * ↑n + 1)) :
  (∀ n : ℕ, a n = 3 * ↑n - 1) ∧
  (∀ n : ℕ, b n = 3 * 2^(n-1)) ∧
  (∀ x : ℝ, (-x^2 + 3*x ≤ 2) ↔ (x ≥ 2 ∨ x ≤ 1)) :=
sorry

end NUMINAMATH_CALUDE_sequence_and_inequality_problem_l850_85004


namespace NUMINAMATH_CALUDE_smallest_n_l850_85000

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_n : 
  let n : ℕ := 9075
  ∀ m : ℕ, m > 0 → 
    (is_factor (5^2) (m * (2^5) * (6^2) * (7^3) * (13^4)) ∧
     is_factor (3^3) (m * (2^5) * (6^2) * (7^3) * (13^4)) ∧
     is_factor (11^2) (m * (2^5) * (6^2) * (7^3) * (13^4))) →
    m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_l850_85000


namespace NUMINAMATH_CALUDE_sector_central_angle_l850_85027

/-- Given a sector with arc length 2π cm and radius 2 cm, its central angle is π radians. -/
theorem sector_central_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2 * Real.pi) (h2 : radius = 2) :
  arc_length / radius = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l850_85027


namespace NUMINAMATH_CALUDE_max_pieces_of_cake_l850_85086

/-- The size of the cake in inches -/
def cake_size : ℕ := 16

/-- The size of each piece in inches -/
def piece_size : ℕ := 4

/-- The area of the cake in square inches -/
def cake_area : ℕ := cake_size * cake_size

/-- The area of each piece in square inches -/
def piece_area : ℕ := piece_size * piece_size

/-- The maximum number of pieces that can be cut from the cake -/
def max_pieces : ℕ := cake_area / piece_area

theorem max_pieces_of_cake :
  max_pieces = 16 :=
sorry

end NUMINAMATH_CALUDE_max_pieces_of_cake_l850_85086


namespace NUMINAMATH_CALUDE_sqrt_six_irrational_l850_85098

theorem sqrt_six_irrational : Irrational (Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_irrational_l850_85098


namespace NUMINAMATH_CALUDE_candy_difference_is_twenty_l850_85006

/-- The number of candies Bryan has compared to Ben -/
def candy_difference : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  λ bryan_skittles bryan_gummy bryan_lollipops ben_mm ben_jelly ben_lollipops =>
    (bryan_skittles + bryan_gummy + bryan_lollipops) - (ben_mm + ben_jelly + ben_lollipops)

theorem candy_difference_is_twenty :
  candy_difference 50 30 15 20 45 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_is_twenty_l850_85006


namespace NUMINAMATH_CALUDE_brads_money_l850_85059

theorem brads_money (total : ℚ) (josh_brad_ratio : ℚ) (josh_doug_ratio : ℚ) :
  total = 68 →
  josh_brad_ratio = 2 →
  josh_doug_ratio = 3/4 →
  ∃ (brad josh doug : ℚ),
    brad + josh + doug = total ∧
    josh = josh_brad_ratio * brad ∧
    josh = josh_doug_ratio * doug ∧
    brad = 12 :=
by sorry

end NUMINAMATH_CALUDE_brads_money_l850_85059


namespace NUMINAMATH_CALUDE_birthday_celebration_attendees_l850_85016

theorem birthday_celebration_attendees :
  ∀ (n : ℕ), 
  (12 * (n + 2) = 16 * n) → 
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_birthday_celebration_attendees_l850_85016


namespace NUMINAMATH_CALUDE_prism_volume_l850_85003

/-- A right rectangular prism with face areas 12, 18, and 24 square inches has a volume of 72 cubic inches -/
theorem prism_volume (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0)
  (area1 : l * w = 12) (area2 : w * h = 18) (area3 : l * h = 24) :
  l * w * h = 72 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l850_85003


namespace NUMINAMATH_CALUDE_proportional_scaling_l850_85001

/-- Proportional scaling of a rectangle -/
theorem proportional_scaling (w h new_w : ℝ) (hw : w > 0) (hh : h > 0) (hnew_w : new_w > 0) :
  let scale_factor := new_w / w
  let new_h := h * scale_factor
  w = 3 ∧ h = 2 ∧ new_w = 12 → new_h = 8 := by sorry

end NUMINAMATH_CALUDE_proportional_scaling_l850_85001


namespace NUMINAMATH_CALUDE_gcd_of_45_75_90_l850_85025

theorem gcd_of_45_75_90 : Nat.gcd 45 (Nat.gcd 75 90) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_75_90_l850_85025


namespace NUMINAMATH_CALUDE_debbie_large_boxes_l850_85053

def large_box_tape : ℕ := 5
def medium_box_tape : ℕ := 3
def small_box_tape : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5
def total_tape_used : ℕ := 44

theorem debbie_large_boxes :
  ∃ (large_boxes : ℕ),
    large_boxes * large_box_tape +
    medium_boxes_packed * medium_box_tape +
    small_boxes_packed * small_box_tape = total_tape_used ∧
    large_boxes = 2 :=
by sorry

end NUMINAMATH_CALUDE_debbie_large_boxes_l850_85053


namespace NUMINAMATH_CALUDE_clara_weight_l850_85070

/-- Given two positive real numbers representing weights in pounds,
    prove that one of them (Clara's weight) is equal to 960/7 pounds,
    given the specified conditions. -/
theorem clara_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight > 0)
  (h2 : clara_weight > 0)
  (h3 : alice_weight + clara_weight = 240)
  (h4 : clara_weight - alice_weight = alice_weight / 3) :
  clara_weight = 960 / 7 := by
  sorry

end NUMINAMATH_CALUDE_clara_weight_l850_85070


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l850_85019

theorem hemisphere_surface_area (r : ℝ) (h : r = 6) :
  let sphere_area := λ r : ℝ => 4 * π * r^2
  let base_area := π * r^2
  let hemisphere_area := sphere_area r / 2 + base_area
  hemisphere_area = 108 * π := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l850_85019


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l850_85084

/-- A cubic equation x^3 + px + q = 0 has three distinct roots in (-2, 4) if and only if
    its coefficients p and q satisfy the given conditions. -/
theorem cubic_three_distinct_roots_in_interval
  (p q : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    -2 < x₁ ∧ x₁ < 4 ∧ -2 < x₂ ∧ x₂ < 4 ∧ -2 < x₃ ∧ x₃ < 4 ∧
    x₁^3 + p*x₁ + q = 0 ∧ x₂^3 + p*x₂ + q = 0 ∧ x₃^3 + p*x₃ + q = 0) ↔
  (4*p^3 + 27*q^2 < 0 ∧ -4*p - 64 < q ∧ q < 2*p + 8) :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l850_85084


namespace NUMINAMATH_CALUDE_min_value_implies_m_l850_85043

/-- Given a function f(x) = x + m / (x - 2) where x > 2 and m > 0,
    if the minimum value of f(x) is 6, then m = 4. -/
theorem min_value_implies_m (m : ℝ) (h_m_pos : m > 0) :
  (∀ x > 2, x + m / (x - 2) ≥ 6) ∧
  (∃ x > 2, x + m / (x - 2) = 6) →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_m_l850_85043


namespace NUMINAMATH_CALUDE_tony_errands_halfway_distance_l850_85035

theorem tony_errands_halfway_distance (groceries haircut doctor : ℕ) 
  (h1 : groceries = 10)
  (h2 : haircut = 15)
  (h3 : doctor = 5) :
  (groceries + haircut + doctor) / 2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_tony_errands_halfway_distance_l850_85035


namespace NUMINAMATH_CALUDE_count_congruent_integers_l850_85054

theorem count_congruent_integers (n : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < 2000 ∧ x % 13 = 3) (Finset.range 2000)).card = 154 := by
  sorry

end NUMINAMATH_CALUDE_count_congruent_integers_l850_85054


namespace NUMINAMATH_CALUDE_lcm_of_12_18_24_l850_85064

theorem lcm_of_12_18_24 : Nat.lcm 12 (Nat.lcm 18 24) = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_18_24_l850_85064


namespace NUMINAMATH_CALUDE_phone_plan_comparison_l850_85078

/-- Represents a mobile phone plan with a monthly fee and a per-minute call charge. -/
structure PhonePlan where
  monthly_fee : ℝ
  per_minute_charge : ℝ

/-- Calculates the monthly bill for a given phone plan and call duration. -/
def monthly_bill (plan : PhonePlan) (duration : ℝ) : ℝ :=
  plan.monthly_fee + plan.per_minute_charge * duration

/-- Plan A with a monthly fee of 15 yuan and a call charge of 0.1 yuan per minute. -/
def plan_a : PhonePlan := ⟨15, 0.1⟩

/-- Plan B with no monthly fee and a call charge of 0.15 yuan per minute. -/
def plan_b : PhonePlan := ⟨0, 0.15⟩

theorem phone_plan_comparison :
  /- 1. Functional relationships are correct -/
  (∀ x, monthly_bill plan_a x = 15 + 0.1 * x) ∧
  (∀ x, monthly_bill plan_b x = 0.15 * x) ∧
  /- 2. For Plan A, a monthly bill of 50 yuan corresponds to 350 minutes -/
  (monthly_bill plan_a 350 = 50) ∧
  /- 3. For 280 minutes, Plan B is more cost-effective -/
  (monthly_bill plan_b 280 < monthly_bill plan_a 280) := by
  sorry

#eval monthly_bill plan_a 350  -- Should output 50
#eval monthly_bill plan_b 280  -- Should output 42
#eval monthly_bill plan_a 280  -- Should output 43

end NUMINAMATH_CALUDE_phone_plan_comparison_l850_85078


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_bisector_length_l850_85050

theorem isosceles_triangle_angle_bisector_length 
  (AB BC AC : ℝ) (h_isosceles : AC = BC) (h_base : AB = 5) (h_lateral : AC = 20) :
  let AD := Real.sqrt (AB * AC * (1 - BC / (AB + AC)))
  AD = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_bisector_length_l850_85050


namespace NUMINAMATH_CALUDE_juice_distribution_l850_85030

theorem juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2/3) * C
  let cups := 4
  let juice_per_cup := juice_volume / cups
  juice_per_cup / C = 1/6 := by sorry

end NUMINAMATH_CALUDE_juice_distribution_l850_85030


namespace NUMINAMATH_CALUDE_replaced_person_weight_l850_85045

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (num_persons : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - (num_persons * avg_weight_increase)

/-- Theorem stating the weight of the replaced person under the given conditions -/
theorem replaced_person_weight :
  weight_of_replaced_person 5 10.0 90 = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l850_85045


namespace NUMINAMATH_CALUDE_intersection_M_N_l850_85007

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def N : Set ℝ := {x | ∃ y, y = Real.sqrt x + Real.log (1 - x)}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l850_85007


namespace NUMINAMATH_CALUDE_smaller_number_problem_l850_85018

theorem smaller_number_problem (x y : ℤ) : 
  x + y = 64 → y = x + 12 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l850_85018


namespace NUMINAMATH_CALUDE_correct_equation_l850_85005

theorem correct_equation (a b : ℝ) : 5 * a^2 * b - 6 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l850_85005


namespace NUMINAMATH_CALUDE_digits_at_1100_to_1102_l850_85026

/-- Represents a list of integers starting with 2 in increasing order -/
def listStartingWith2 : List ℕ := sorry

/-- Returns the nth digit in the concatenated string of all numbers in the list -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 1100th, 1101st, and 1102nd digits are 2, 1, and 9 respectively -/
theorem digits_at_1100_to_1102 :
  (nthDigit 1100 = 2) ∧ (nthDigit 1101 = 1) ∧ (nthDigit 1102 = 9) := by sorry

end NUMINAMATH_CALUDE_digits_at_1100_to_1102_l850_85026


namespace NUMINAMATH_CALUDE_eulers_formula_l850_85042

/-- A connected planar graph -/
structure ConnectedPlanarGraph where
  s : ℕ  -- number of vertices
  f : ℕ  -- number of faces
  a : ℕ  -- number of edges

/-- Euler's formula for connected planar graphs -/
theorem eulers_formula (G : ConnectedPlanarGraph) : G.f = G.a - G.s + 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l850_85042


namespace NUMINAMATH_CALUDE_expression_equals_zero_l850_85088

theorem expression_equals_zero :
  (-1 : ℝ) ^ 2022 + |-2| - (1/2 : ℝ) ^ 0 - 2 * Real.tan (π/4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l850_85088


namespace NUMINAMATH_CALUDE_machinery_cost_l850_85038

def total_amount : ℝ := 7428.57
def raw_materials : ℝ := 5000
def cash_percentage : ℝ := 0.30

theorem machinery_cost :
  ∃ (machinery : ℝ),
    machinery = total_amount - raw_materials - (cash_percentage * total_amount) ∧
    machinery = 200 := by
  sorry

end NUMINAMATH_CALUDE_machinery_cost_l850_85038
