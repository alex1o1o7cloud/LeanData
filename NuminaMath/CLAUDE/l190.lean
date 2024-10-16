import Mathlib

namespace NUMINAMATH_CALUDE_infinitely_many_L_for_fibonacci_ratio_l190_19098

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- The sequence defined in the problem -/
def a (L : ℕ) : ℕ → ℚ
  | 0 => 0
  | (n + 1) => 1 / (L - a L n)

theorem infinitely_many_L_for_fibonacci_ratio :
  ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k, ∃ i j,
    ∀ n, a (lucas (f k)) n = (fib i) / (fib j) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_L_for_fibonacci_ratio_l190_19098


namespace NUMINAMATH_CALUDE_soda_price_increase_l190_19052

theorem soda_price_increase (original_price : ℝ) (new_price : ℝ) (increase_percentage : ℝ) : 
  new_price = 15 ∧ increase_percentage = 50 ∧ new_price = original_price * (1 + increase_percentage / 100) → 
  original_price = 10 := by
sorry

end NUMINAMATH_CALUDE_soda_price_increase_l190_19052


namespace NUMINAMATH_CALUDE_dog_feed_mix_problem_l190_19051

/-- The cost per pound of the cheaper kind of feed -/
def cheaper_feed_cost : ℝ := 0.18

theorem dog_feed_mix_problem :
  -- Total weight of the mix
  let total_weight : ℝ := 35
  -- Cost per pound of the final mix
  let final_mix_cost : ℝ := 0.36
  -- Cost per pound of the more expensive feed
  let expensive_feed_cost : ℝ := 0.53
  -- Weight of the cheaper feed used
  let cheaper_feed_weight : ℝ := 17
  -- Weight of the more expensive feed used
  let expensive_feed_weight : ℝ := total_weight - cheaper_feed_weight
  -- Total value of the final mix
  let total_value : ℝ := total_weight * final_mix_cost
  -- Value of the more expensive feed
  let expensive_feed_value : ℝ := expensive_feed_weight * expensive_feed_cost
  -- Equation for the total value
  cheaper_feed_weight * cheaper_feed_cost + expensive_feed_value = total_value →
  cheaper_feed_cost = 0.18 := by
sorry

end NUMINAMATH_CALUDE_dog_feed_mix_problem_l190_19051


namespace NUMINAMATH_CALUDE_curve_equation_k_value_l190_19086

-- Define the curve C
def C (x y : ℝ) : Prop :=
  Real.sqrt ((x - 0)^2 + (y - Real.sqrt 3)^2) +
  Real.sqrt ((x - 0)^2 + (y + Real.sqrt 3)^2) = 4

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

-- Define the perpendicularity condition
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Theorem for the equation of C
theorem curve_equation :
  ∀ x y : ℝ, C x y ↔ x^2 + y^2/4 = 1 :=
sorry

-- Theorem for the value of k
theorem k_value (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    C x₁ y₁ ∧ C x₂ y₂ ∧
    Line k x₁ y₁ ∧ Line k x₂ y₂ ∧
    Perpendicular x₁ y₁ x₂ y₂) →
  k = 1/2 ∨ k = -1/2 :=
sorry

end NUMINAMATH_CALUDE_curve_equation_k_value_l190_19086


namespace NUMINAMATH_CALUDE_star_A_B_equals_result_l190_19065

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {y | y ≥ 1}

-- Define the operation *
def star (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- State the theorem
theorem star_A_B_equals_result : star A B = {x | (0 ≤ x ∧ x < 1) ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_star_A_B_equals_result_l190_19065


namespace NUMINAMATH_CALUDE_b_plus_3b_squared_positive_l190_19012

theorem b_plus_3b_squared_positive (b : ℝ) (h1 : -0.5 < b) (h2 : b < 0) : 
  b + 3 * b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_3b_squared_positive_l190_19012


namespace NUMINAMATH_CALUDE_max_cables_is_150_l190_19064

/-- Represents the maximum number of cables that can be installed between
    Brand A and Brand B computers under given conditions. -/
def max_cables (total_employees : ℕ) (brand_a_computers : ℕ) (brand_b_computers : ℕ) 
               (connectable_brand_b : ℕ) : ℕ :=
  brand_a_computers * connectable_brand_b

/-- Theorem stating that the maximum number of cables is 150 under the given conditions. -/
theorem max_cables_is_150 :
  max_cables 50 15 35 10 = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_max_cables_is_150_l190_19064


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l190_19096

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_parallel_to_x_axis :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ (deriv f x₀ = 0) ∧ (f x₀ = 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l190_19096


namespace NUMINAMATH_CALUDE_reach_50_from_49_l190_19063

def double (n : ℕ) : ℕ := n * 2

def eraseLast (n : ℕ) : ℕ := n / 10

def canReach (start target : ℕ) : Prop :=
  ∃ (steps : ℕ), ∃ (moves : Fin steps → Bool),
    (start = target) ∨
    (∃ (intermediate : Fin (steps + 1) → ℕ),
      intermediate 0 = start ∧
      intermediate (Fin.last steps) = target ∧
      ∀ i : Fin steps,
        (moves i = true → intermediate (i.succ) = double (intermediate i)) ∧
        (moves i = false → intermediate (i.succ) = eraseLast (intermediate i)))

theorem reach_50_from_49 : canReach 49 50 := by
  sorry

end NUMINAMATH_CALUDE_reach_50_from_49_l190_19063


namespace NUMINAMATH_CALUDE_abs_inequality_l190_19015

theorem abs_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l190_19015


namespace NUMINAMATH_CALUDE_zoe_calories_l190_19037

/-- The number of calories Zoe ate from her snack -/
def total_calories (strawberry_count : ℕ) (yogurt_ounces : ℕ) (calories_per_strawberry : ℕ) (calories_per_yogurt_ounce : ℕ) : ℕ :=
  strawberry_count * calories_per_strawberry + yogurt_ounces * calories_per_yogurt_ounce

/-- Theorem stating that Zoe ate 150 calories -/
theorem zoe_calories : total_calories 12 6 4 17 = 150 := by
  sorry

end NUMINAMATH_CALUDE_zoe_calories_l190_19037


namespace NUMINAMATH_CALUDE_number_of_boys_l190_19095

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 → 
  boys + girls = total → 
  girls = boys → 
  boys = 50 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l190_19095


namespace NUMINAMATH_CALUDE_half_to_fourth_power_l190_19007

theorem half_to_fourth_power : (1/2 : ℚ)^4 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_half_to_fourth_power_l190_19007


namespace NUMINAMATH_CALUDE_circle_area_equals_circumference_l190_19058

theorem circle_area_equals_circumference (r : ℝ) (h : r > 0) :
  π * r^2 = 2 * π * r → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equals_circumference_l190_19058


namespace NUMINAMATH_CALUDE_ceiling_times_self_216_l190_19054

theorem ceiling_times_self_216 :
  ∃! x : ℝ, ⌈x⌉ * x = 216 ∧ x = 14.4 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_216_l190_19054


namespace NUMINAMATH_CALUDE_solution_set_correct_l190_19061

/-- An odd function f: ℝ → ℝ with specific properties -/
class OddFunction (f : ℝ → ℝ) :=
  (odd : ∀ x, f (-x) = -f x)
  (deriv_pos : ∀ x < 0, deriv f x > 0)
  (zero_at_neg_half : f (-1/2) = 0)

/-- The solution set for f(x) < 0 given an odd function with specific properties -/
def solution_set (f : ℝ → ℝ) [OddFunction f] : Set ℝ :=
  {x | x < -1/2 ∨ (0 < x ∧ x < 1/2)}

/-- Theorem stating that the solution set is correct -/
theorem solution_set_correct (f : ℝ → ℝ) [OddFunction f] :
  ∀ x, f x < 0 ↔ x ∈ solution_set f :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_l190_19061


namespace NUMINAMATH_CALUDE_min_red_tulips_for_arrangement_l190_19088

/-- Represents the number of tulips in a bouquet -/
structure Bouquet where
  white : ℕ
  red : ℕ

/-- Represents the total number of tulips and bouquets -/
structure TulipArrangement where
  whiteTotal : ℕ
  redTotal : ℕ
  bouquetCount : ℕ

/-- Checks if a TulipArrangement is valid according to the problem constraints -/
def isValidArrangement (arr : TulipArrangement) : Prop :=
  ∃ (b : Bouquet),
    arr.whiteTotal = arr.bouquetCount * b.white ∧
    arr.redTotal = arr.bouquetCount * b.red

/-- The main theorem to prove -/
theorem min_red_tulips_for_arrangement :
  ∀ (arr : TulipArrangement),
    arr.whiteTotal = 21 →
    arr.bouquetCount = 7 →
    isValidArrangement arr →
    arr.redTotal ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_min_red_tulips_for_arrangement_l190_19088


namespace NUMINAMATH_CALUDE_certain_number_proof_l190_19067

theorem certain_number_proof (z : ℤ) (h1 : z % 9 = 6) 
  (h2 : ∃ x : ℤ, ∃ m : ℤ, (z + x) / 9 = m) : 
  ∃ x : ℤ, x = 3 ∧ ∃ m : ℤ, (z + x) / 9 = m :=
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l190_19067


namespace NUMINAMATH_CALUDE_adults_on_bus_l190_19092

/-- Given a bus with 60 passengers where children make up 25% of the riders,
    prove that there are 45 adults on the bus. -/
theorem adults_on_bus (total_passengers : ℕ) (children_percentage : ℚ) : 
  total_passengers = 60 →
  children_percentage = 25 / 100 →
  (total_passengers : ℚ) * (1 - children_percentage) = 45 := by
sorry

end NUMINAMATH_CALUDE_adults_on_bus_l190_19092


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l190_19009

/-- The conic section defined by the equation 10x - 2xy - 2y + 1 = 0 -/
def ConicSection (x y : ℝ) : Prop :=
  10 * x - 2 * x * y - 2 * y + 1 = 0

/-- The eccentricity of a conic section -/
def Eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 2

theorem conic_section_eccentricity :
  ∀ x y : ℝ, ConicSection x y → ∃ e : ℝ, Eccentricity e := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l190_19009


namespace NUMINAMATH_CALUDE_team_winning_percentage_l190_19072

theorem team_winning_percentage 
  (first_games : ℕ) 
  (total_games : ℕ) 
  (first_win_rate : ℚ) 
  (remaining_win_rate : ℚ) 
  (h1 : first_games = 30)
  (h2 : total_games = 60)
  (h3 : first_win_rate = 2/5)
  (h4 : remaining_win_rate = 4/5) : 
  (first_win_rate * first_games + remaining_win_rate * (total_games - first_games)) / total_games = 3/5 := by
sorry

end NUMINAMATH_CALUDE_team_winning_percentage_l190_19072


namespace NUMINAMATH_CALUDE_forbidden_city_area_scientific_notation_l190_19074

/-- The area of the Forbidden City in square meters -/
def forbidden_city_area : ℝ := 720000

/-- Scientific notation representation of the Forbidden City's area -/
def scientific_notation : ℝ := 7.2 * (10 ^ 5)

theorem forbidden_city_area_scientific_notation :
  forbidden_city_area = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_forbidden_city_area_scientific_notation_l190_19074


namespace NUMINAMATH_CALUDE_cookies_per_bag_l190_19068

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 75) (h2 : num_bags = 25) :
  total_cookies / num_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l190_19068


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l190_19056

/-- Given a quadratic equation x^2 + 12x + k = 0 where k is a real number,
    if the nonzero roots are in the ratio 3:1, then k = 27. -/
theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 12*x + k = 0 ∧ y^2 + 12*y + k = 0) → k = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l190_19056


namespace NUMINAMATH_CALUDE_log_difference_sqrt_l190_19018

theorem log_difference_sqrt (x : ℝ) : 
  x = Real.sqrt (Real.log 8 / Real.log 4 - Real.log 16 / Real.log 8) → x = 1 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_sqrt_l190_19018


namespace NUMINAMATH_CALUDE_window_area_l190_19011

-- Define the number of panes
def num_panes : ℕ := 8

-- Define the length of each pane in inches
def pane_length : ℕ := 12

-- Define the width of each pane in inches
def pane_width : ℕ := 8

-- Theorem statement
theorem window_area :
  (num_panes * pane_length * pane_width) = 768 := by
  sorry

end NUMINAMATH_CALUDE_window_area_l190_19011


namespace NUMINAMATH_CALUDE_remainder_divisibility_l190_19043

theorem remainder_divisibility (x : ℤ) : 
  (∃ k : ℤ, x = 63 * k + 27) → (∃ m : ℤ, x = 8 * m + 3) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l190_19043


namespace NUMINAMATH_CALUDE_problem_solution_l190_19028

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m

theorem problem_solution (m : ℝ) (h_m : m > 0) :
  (∀ x, f m (x - 3) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 2) →
  m = 2 ∧
  (∃ t : ℝ, ∀ x, ∃ y : ℝ, f 2 y ≥ |2*x - 1| - t^2 + (3/2)*t + 1 ↔ t ≤ 1/2 ∨ t ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l190_19028


namespace NUMINAMATH_CALUDE_f_range_l190_19046

def f (x : ℝ) : ℝ := -x^2

theorem f_range :
  ∀ y ∈ Set.range (f ∘ (Set.Icc (-3) 1).restrict f), -9 ≤ y ∧ y ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l190_19046


namespace NUMINAMATH_CALUDE_smaller_std_dev_more_stable_smaller_variance_more_stable_smaller_mean_not_necessarily_more_stable_l190_19042

-- Define a dataset as a list of real numbers
def Dataset := List ℝ

-- Define standard deviation
def standardDeviation (data : Dataset) : ℝ :=
  sorry

-- Define variance
def variance (data : Dataset) : ℝ :=
  sorry

-- Define mean
def mean (data : Dataset) : ℝ :=
  sorry

-- Define a measure of concentration and stability
def isConcentratedAndStable (data : Dataset) : Prop :=
  sorry

-- Theorem stating that smaller standard deviation implies more concentrated and stable distribution
theorem smaller_std_dev_more_stable (data1 data2 : Dataset) :
  standardDeviation data1 < standardDeviation data2 →
  isConcentratedAndStable data1 → isConcentratedAndStable data2 :=
sorry

-- Theorem stating that smaller variance implies more concentrated and stable distribution
theorem smaller_variance_more_stable (data1 data2 : Dataset) :
  variance data1 < variance data2 →
  isConcentratedAndStable data1 → isConcentratedAndStable data2 :=
sorry

-- Theorem stating that smaller mean does not necessarily imply more concentrated and stable distribution
theorem smaller_mean_not_necessarily_more_stable :
  ∃ (data1 data2 : Dataset), mean data1 < mean data2 ∧
  isConcentratedAndStable data2 ∧ ¬isConcentratedAndStable data1 :=
sorry

end NUMINAMATH_CALUDE_smaller_std_dev_more_stable_smaller_variance_more_stable_smaller_mean_not_necessarily_more_stable_l190_19042


namespace NUMINAMATH_CALUDE_number_above_200_is_91_l190_19002

/-- Represents the array where the k-th row contains the first 2k natural numbers -/
def array_sum (k : ℕ) : ℕ := k * (2 * k + 1) / 2

/-- The row number in which 200 is located -/
def row_of_200 : ℕ := 14

/-- The starting number of the row containing 200 -/
def start_of_row_200 : ℕ := array_sum (row_of_200 - 1) + 1

/-- The position of 200 in its row -/
def position_of_200 : ℕ := 200 - start_of_row_200 + 1

/-- The number directly above 200 -/
def number_above_200 : ℕ := array_sum (row_of_200 - 1)

theorem number_above_200_is_91 : number_above_200 = 91 := by
  sorry

end NUMINAMATH_CALUDE_number_above_200_is_91_l190_19002


namespace NUMINAMATH_CALUDE_reservoir_capacity_proof_l190_19055

theorem reservoir_capacity_proof (current_amount : ℝ) (normal_level : ℝ) (total_capacity : ℝ) 
  (h1 : current_amount = 14)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.7 * total_capacity) :
  total_capacity - normal_level = 13 := by
sorry

end NUMINAMATH_CALUDE_reservoir_capacity_proof_l190_19055


namespace NUMINAMATH_CALUDE_racetrack_circumference_difference_l190_19014

/-- The difference in circumferences of two concentric circles -/
theorem racetrack_circumference_difference (inner_diameter outer_diameter : ℝ) 
  (h1 : inner_diameter = 55)
  (h2 : outer_diameter = inner_diameter + 2 * 15) :
  π * outer_diameter - π * inner_diameter = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_racetrack_circumference_difference_l190_19014


namespace NUMINAMATH_CALUDE_max_value_of_z_l190_19078

-- Define the objective function
def z (x y : ℝ) : ℝ := 4 * x + 3 * y

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x - y - 2 ≥ 0 ∧ 2 * x + y - 2 ≤ 0 ∧ y + 4 ≥ 0

-- Theorem statement
theorem max_value_of_z :
  ∃ (max : ℝ), max = 8 ∧
  (∀ x y : ℝ, feasible_region x y → z x y ≤ max) ∧
  (∃ x y : ℝ, feasible_region x y ∧ z x y = max) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l190_19078


namespace NUMINAMATH_CALUDE_f_sum_zero_four_l190_19024

def f (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem f_sum_zero_four (a b c d : ℝ) :
  f a b c d 1 = 1 →
  f a b c d 2 = 2 →
  f a b c d 3 = 3 →
  f a b c d 0 + f a b c d 4 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_f_sum_zero_four_l190_19024


namespace NUMINAMATH_CALUDE_simplify_square_roots_l190_19071

theorem simplify_square_roots : Real.sqrt 12 * Real.sqrt 27 - 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l190_19071


namespace NUMINAMATH_CALUDE_fraction_addition_l190_19017

theorem fraction_addition (c : ℝ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l190_19017


namespace NUMINAMATH_CALUDE_guppy_ratio_l190_19008

/-- Represents the number of guppies each person has -/
structure Guppies where
  haylee : ℕ
  jose : ℕ
  charliz : ℕ
  nicolai : ℕ

/-- The conditions of the guppy problem -/
def guppy_conditions (g : Guppies) : Prop :=
  g.haylee = 36 ∧
  g.charliz = g.jose / 3 ∧
  g.nicolai = 4 * g.charliz ∧
  g.haylee + g.jose + g.charliz + g.nicolai = 84

/-- The theorem stating the ratio of Jose's guppies to Haylee's guppies -/
theorem guppy_ratio (g : Guppies) (h : guppy_conditions g) : 
  g.jose * 2 = g.haylee :=
sorry

end NUMINAMATH_CALUDE_guppy_ratio_l190_19008


namespace NUMINAMATH_CALUDE_savings_theorem_l190_19003

/-- Represents the prices of food items and meals --/
structure FoodPrices where
  burger : ℝ
  fries : ℝ
  drink : ℝ
  burgerMeal : ℝ
  kidsBurger : ℝ
  kidsFries : ℝ
  kidsJuice : ℝ
  kidsMeal : ℝ

/-- Calculates the savings when buying meals instead of individual items --/
def calculateSavings (prices : FoodPrices) : ℝ :=
  let individualCost := 
    2 * (prices.burger + prices.fries + prices.drink) +
    2 * (prices.kidsBurger + prices.kidsFries + prices.kidsJuice)
  let mealCost := 2 * prices.burgerMeal + 2 * prices.kidsMeal
  individualCost - mealCost

/-- The savings theorem --/
theorem savings_theorem (prices : FoodPrices) 
  (h1 : prices.burger = 5)
  (h2 : prices.fries = 3)
  (h3 : prices.drink = 3)
  (h4 : prices.burgerMeal = 9.5)
  (h5 : prices.kidsBurger = 3)
  (h6 : prices.kidsFries = 2)
  (h7 : prices.kidsJuice = 2)
  (h8 : prices.kidsMeal = 5) :
  calculateSavings prices = 7 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l190_19003


namespace NUMINAMATH_CALUDE_candy_bar_cost_l190_19006

/-- The cost of a candy bar given initial and remaining amounts -/
theorem candy_bar_cost (initial_amount : ℚ) (remaining_amount : ℚ) 
  (h1 : initial_amount = 3)
  (h2 : remaining_amount = 2) :
  initial_amount - remaining_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l190_19006


namespace NUMINAMATH_CALUDE_jason_cantaloupes_l190_19035

theorem jason_cantaloupes (total keith fred : ℕ) (h1 : total = 65) (h2 : keith = 29) (h3 : fred = 16) :
  total - keith - fred = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_cantaloupes_l190_19035


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l190_19080

-- Define the binomial expansion function
def binomial_expansion (a : ℝ) (n : ℕ) : ℝ → ℝ := sorry

-- Define the sum of coefficients function
def sum_of_coefficients (a : ℝ) (n : ℕ) : ℝ := sorry

-- Define the sum of binomial coefficients function
def sum_of_binomial_coefficients (n : ℕ) : ℕ := sorry

-- Define the coefficient of x^2 function
def coefficient_of_x_squared (a : ℝ) (n : ℕ) : ℝ := sorry

theorem binomial_expansion_theorem (a : ℝ) (n : ℕ) :
  sum_of_coefficients a n = -1 ∧
  sum_of_binomial_coefficients n = 32 →
  coefficient_of_x_squared a n = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l190_19080


namespace NUMINAMATH_CALUDE_problem_solution_l190_19081

theorem problem_solution (x : ℤ) (h : x = 40) : x * 6 - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l190_19081


namespace NUMINAMATH_CALUDE_min_troupe_size_l190_19021

def is_valid_troupe_size (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ n > 50

theorem min_troupe_size :
  ∃ (n : ℕ), is_valid_troupe_size n ∧ ∀ (m : ℕ), is_valid_troupe_size m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_troupe_size_l190_19021


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l190_19030

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    imaginary axis length of 4, and focal distance of 4√3,
    prove that its asymptotes are given by y = ±(√2/2)x -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_imaginary_axis : b = 2) 
  (h_focal_distance : 2 * Real.sqrt ((a^2 + b^2) : ℝ) = 4 * Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 / 2 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotes_l190_19030


namespace NUMINAMATH_CALUDE_water_cost_for_family_of_six_l190_19053

/-- The cost of fresh water for a family for one day -/
def water_cost (family_size : ℕ) (purification_cost : ℚ) (water_per_person : ℚ) : ℚ :=
  family_size * water_per_person * purification_cost

/-- Proof that the water cost for a family of 6 is $3 -/
theorem water_cost_for_family_of_six :
  water_cost 6 1 (1/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_water_cost_for_family_of_six_l190_19053


namespace NUMINAMATH_CALUDE_discount_difference_l190_19066

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  bill = 8000 ∧ 
  single_discount = 0.3 ∧ 
  first_discount = 0.26 ∧ 
  second_discount = 0.05 → 
  (bill * (1 - first_discount) * (1 - second_discount)) - (bill * (1 - single_discount)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l190_19066


namespace NUMINAMATH_CALUDE_square_perimeter_l190_19033

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (∃ (r : ℝ × ℝ), r.1 = s/2 ∧ r.2 = s ∧ 2*(r.1 + r.2) = 24) → 4*s = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l190_19033


namespace NUMINAMATH_CALUDE_process_never_stops_l190_19082

/-- Represents a large number as a list of digits -/
def LargeNumber := List Nat

/-- The initial number with 900 digits, all 1s -/
def initial_number : LargeNumber := List.replicate 900 1

/-- Extracts the last two digits of a LargeNumber -/
def last_two_digits (n : LargeNumber) : Nat :=
  match n.reverse with
  | d1 :: d2 :: _ => d1 + 10 * d2
  | _ => 0

/-- Applies the transformation rule to a LargeNumber -/
def transform (n : LargeNumber) : Nat :=
  let a := n.foldl (fun acc d => acc * 10 + d) 0 / 100
  let b := last_two_digits n
  2 * a + 8 * b

/-- Predicate to check if a number is less than 100 -/
def is_less_than_100 (n : Nat) : Prop := n < 100

/-- Main theorem: The process will never stop -/
theorem process_never_stops :
  ∀ n : Nat, ∃ m : Nat, m > n ∧ ¬(is_less_than_100 (transform (List.replicate m 1))) :=
  sorry


end NUMINAMATH_CALUDE_process_never_stops_l190_19082


namespace NUMINAMATH_CALUDE_polynomial_value_l190_19013

theorem polynomial_value (x y : ℚ) (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l190_19013


namespace NUMINAMATH_CALUDE_election_margin_theorem_l190_19079

/-- Represents an election with two candidates -/
structure Election where
  total_votes : ℕ
  winner_votes : ℕ
  winner_percentage : ℚ

/-- Calculates the margin of victory in an election -/
def margin_of_victory (e : Election) : ℕ :=
  e.winner_votes - (e.total_votes - e.winner_votes)

/-- Theorem stating the margin of victory for the given election scenario -/
theorem election_margin_theorem (e : Election) 
  (h1 : e.winner_percentage = 65 / 100)
  (h2 : e.winner_votes = 650) :
  margin_of_victory e = 300 := by
sorry

#eval margin_of_victory { total_votes := 1000, winner_votes := 650, winner_percentage := 65 / 100 }

end NUMINAMATH_CALUDE_election_margin_theorem_l190_19079


namespace NUMINAMATH_CALUDE_marias_water_bottles_l190_19034

theorem marias_water_bottles (initial bottles_drunk final : ℕ) 
  (h1 : initial = 14)
  (h2 : bottles_drunk = 8)
  (h3 : final = 51) :
  final - (initial - bottles_drunk) = 45 := by
  sorry

end NUMINAMATH_CALUDE_marias_water_bottles_l190_19034


namespace NUMINAMATH_CALUDE_part_one_part_two_l190_19036

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2| - |x + a|

-- Part 1
theorem part_one :
  {x : ℝ | f 3 x ≤ 1/2} = {x : ℝ | x ≥ -11/4} := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  ({x : ℝ | f a x ≤ a} = Set.univ) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l190_19036


namespace NUMINAMATH_CALUDE_manuscript_solution_l190_19069

/-- Represents the problem of determining the number of pages in a manuscript. -/
def ManuscriptProblem (copies : ℕ) (printCost : ℚ) (bindCost : ℚ) (totalCost : ℚ) : Prop :=
  ∃ (pages : ℕ),
    (copies : ℚ) * printCost * (pages : ℚ) + (copies : ℚ) * bindCost = totalCost ∧
    pages = 400

/-- The solution to the manuscript problem. -/
theorem manuscript_solution :
  ManuscriptProblem 10 (5/100) 5 250 := by
  sorry

#check manuscript_solution

end NUMINAMATH_CALUDE_manuscript_solution_l190_19069


namespace NUMINAMATH_CALUDE_entertainment_unit_theorem_l190_19022

/-- A structure representing the entertainment unit -/
structure EntertainmentUnit where
  singers : ℕ
  dancers : ℕ
  total : ℕ
  both : ℕ
  h_singers : singers = 4
  h_dancers : dancers = 5
  h_total : total = singers + dancers - both
  h_all_can : total ≤ singers + dancers

/-- The probability of selecting at least one person who can both sing and dance -/
def prob_at_least_one (u : EntertainmentUnit) : ℚ :=
  1 - (Nat.choose (u.total - u.both) 2 : ℚ) / (Nat.choose u.total 2 : ℚ)

/-- The probability distribution of ξ -/
def prob_dist (u : EntertainmentUnit) : ℕ → ℚ
| 0 => (Nat.choose (u.total - u.both) 2 : ℚ) / (Nat.choose u.total 2 : ℚ)
| 1 => (u.both * (u.total - u.both) : ℚ) / (Nat.choose u.total 2 : ℚ)
| 2 => (Nat.choose u.both 2 : ℚ) / (Nat.choose u.total 2 : ℚ)
| _ => 0

/-- The expected value of ξ -/
def expected_value (u : EntertainmentUnit) : ℚ :=
  0 * prob_dist u 0 + 1 * prob_dist u 1 + 2 * prob_dist u 2

/-- The main theorem -/
theorem entertainment_unit_theorem (u : EntertainmentUnit) 
  (h_prob : prob_at_least_one u = 11/21) : 
  u.total = 7 ∧ 
  prob_dist u 0 = 10/21 ∧ 
  prob_dist u 1 = 10/21 ∧ 
  prob_dist u 2 = 1/21 ∧
  expected_value u = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_entertainment_unit_theorem_l190_19022


namespace NUMINAMATH_CALUDE_class_average_problem_l190_19094

/-- Given a class of 25 students where 10 students average 88% and the overall average is 79%,
    this theorem proves that the average percentage of the remaining 15 students is 73%. -/
theorem class_average_problem (total_students : Nat) (group_a_students : Nat) (group_b_students : Nat)
    (group_b_average : ℝ) (overall_average : ℝ) :
    total_students = 25 →
    group_a_students = 15 →
    group_b_students = 10 →
    group_b_average = 88 →
    overall_average = 79 →
    (group_a_students * x + group_b_students * group_b_average) / total_students = overall_average →
    x = 73 :=
  by sorry


end NUMINAMATH_CALUDE_class_average_problem_l190_19094


namespace NUMINAMATH_CALUDE_area_of_large_rectangle_l190_19040

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Area of large rectangle formed by three identical smaller rectangles -/
theorem area_of_large_rectangle (small_rect : Rectangle) 
    (h1 : small_rect.width = 7)
    (h2 : small_rect.height ≥ small_rect.width) : 
  (Rectangle.area { width := 3 * small_rect.height, height := small_rect.width }) = 294 := by
  sorry

#check area_of_large_rectangle

end NUMINAMATH_CALUDE_area_of_large_rectangle_l190_19040


namespace NUMINAMATH_CALUDE_find_other_number_l190_19076

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 4 * b = 161) 
  (h2 : a = 17 ∨ b = 17) : (a = 31 ∧ b = 17) ∨ (a = 17 ∧ b = 31) :=
sorry

end NUMINAMATH_CALUDE_find_other_number_l190_19076


namespace NUMINAMATH_CALUDE_survey_analysis_l190_19004

-- Define the survey data
structure SurveyData where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  male_like : ℕ
  female_like : ℕ
  male_dislike : ℕ
  female_dislike : ℕ

-- Define the theorem
theorem survey_analysis (data : SurveyData) 
  (h1 : data.total_students = 400 + data.female_like + data.male_dislike)
  (h2 : data.male_students = 280 + data.male_dislike)
  (h3 : data.female_students = 120 + data.female_like)
  (h4 : data.male_students = (4 : ℚ) / 7 * data.total_students)
  (h5 : data.female_like = (3 : ℚ) / 5 * data.female_students)
  (h6 : data.male_like = 280)
  (h7 : data.female_dislike = 120) :
  data.female_like = 180 ∧ 
  data.male_dislike = 120 ∧ 
  ((700 : ℚ) * (280 * 120 - 180 * 120)^2 / (460 * 240 * 400 * 300) < (10828 : ℚ) / 1000) :=
sorry


end NUMINAMATH_CALUDE_survey_analysis_l190_19004


namespace NUMINAMATH_CALUDE_least_coins_seventeen_coins_coins_in_jar_l190_19091

theorem least_coins (n : ℕ) : 
  (n % 7 = 3) ∧ (n % 4 = 1) ∧ (n % 6 = 5) → n ≥ 17 :=
by
  sorry

theorem seventeen_coins : 
  (17 % 7 = 3) ∧ (17 % 4 = 1) ∧ (17 % 6 = 5) :=
by
  sorry

theorem coins_in_jar : 
  ∃ (n : ℕ), (n % 7 = 3) ∧ (n % 4 = 1) ∧ (n % 6 = 5) ∧ 
  (∀ (m : ℕ), (m % 7 = 3) ∧ (m % 4 = 1) ∧ (m % 6 = 5) → m ≥ n) :=
by
  sorry

end NUMINAMATH_CALUDE_least_coins_seventeen_coins_coins_in_jar_l190_19091


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l190_19027

/-- A line with equal x and y intercepts passing through (-1, 2) -/
structure EqualInterceptLine where
  -- The slope of the line
  m : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (-1, 2)
  point_condition : 2 = m * (-1) + b
  -- The line has equal x and y intercepts
  equal_intercepts : b ≠ 0 → -b/m = b

/-- The equation of an EqualInterceptLine is either 2x + y = 0 or x + y - 1 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = -2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l190_19027


namespace NUMINAMATH_CALUDE_simplify_fraction_l190_19039

theorem simplify_fraction : 
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 2 + Real.sqrt 7) = 
  -3.6 * (1 + Real.sqrt 2 - 2 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l190_19039


namespace NUMINAMATH_CALUDE_shop_profit_calculation_l190_19060

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℕ := 210

/-- The additional cost of a t-shirt compared to a jersey -/
def tshirt_additional_cost : ℕ := 30

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℕ := jersey_profit + tshirt_additional_cost

theorem shop_profit_calculation :
  tshirt_profit = 240 :=
by sorry

end NUMINAMATH_CALUDE_shop_profit_calculation_l190_19060


namespace NUMINAMATH_CALUDE_no_integer_solutions_l190_19099

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 4*x*y + 3*y^2 - z^2 = 24) ∧
  (-x^2 + 3*y*z + 5*z^2 = 60) ∧
  (x^2 + 2*x*y + 5*z^2 = 85) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l190_19099


namespace NUMINAMATH_CALUDE_league_games_l190_19057

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 20 →
  total_games = 760 →
  games_per_matchup = 4 →
  games_per_matchup * (num_teams - 1) * num_teams / 2 = total_games :=
by
  sorry

end NUMINAMATH_CALUDE_league_games_l190_19057


namespace NUMINAMATH_CALUDE_total_birds_count_l190_19025

def birds_monday : ℕ := 70

def birds_tuesday : ℕ := birds_monday / 2

def birds_wednesday : ℕ := birds_tuesday + 8

theorem total_birds_count : birds_monday + birds_tuesday + birds_wednesday = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l190_19025


namespace NUMINAMATH_CALUDE_one_real_root_condition_l190_19089

theorem one_real_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (a * x^2 + a * x + 2) = a * x + 2) ↔ 
  (a = -8 ∨ a ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_one_real_root_condition_l190_19089


namespace NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l190_19070

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h2 : a 2 = 9) (h5 : a 5 = 33) : 
  ∃ d : ℝ, d = 8 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_common_difference_of_arithmetic_sequence_l190_19070


namespace NUMINAMATH_CALUDE_fraction_sum_product_equality_l190_19020

theorem fraction_sum_product_equality : (1 : ℚ) / 2 + ((1 : ℚ) / 2 * (1 : ℚ) / 2) = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_equality_l190_19020


namespace NUMINAMATH_CALUDE_principal_is_2000_l190_19001

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- Proves that the principal is 2000 given the conditions of the problem. -/
theorem principal_is_2000 (rate : ℚ) (time : ℚ) (interest : ℚ) 
    (h_rate : rate = 5)
    (h_time : time = 13)
    (h_interest : interest = 1300)
    (h_simple_interest : simpleInterest principal rate time = interest) :
  principal = 2000 := by
  sorry

#check principal_is_2000

end NUMINAMATH_CALUDE_principal_is_2000_l190_19001


namespace NUMINAMATH_CALUDE_triangle_intersection_theorem_l190_19097

/-- A triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop := sorry

/-- Constructs the next triangle from the given triangle -/
def nextTriangle (t : Triangle) : Triangle := sorry

/-- Counts the number of intersection points between two triangles -/
def intersectionPoints (t1 t2 : Triangle) : ℕ := sorry

/-- The main theorem -/
theorem triangle_intersection_theorem (A₀B₀C₀ : Triangle) (h : isAcute A₀B₀C₀) :
  ∀ n : ℕ, intersectionPoints ((nextTriangle^[n]) A₀B₀C₀) ((nextTriangle^[n+1]) A₀B₀C₀) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_intersection_theorem_l190_19097


namespace NUMINAMATH_CALUDE_line_through_point_l190_19023

/-- Given a line 3x + ay - 5 = 0 that passes through the point (1, 2), prove that a = 1 --/
theorem line_through_point (a : ℝ) : (3 * 1 + a * 2 - 5 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l190_19023


namespace NUMINAMATH_CALUDE_smallest_d_value_l190_19073

def σ (v : Fin 4 → ℕ) : Finset (Fin 4 → ℕ) := sorry

theorem smallest_d_value (a b c d : ℕ) :
  0 < a → a < b → b < c → c < d →
  (∃ (s : ℕ), ∃ (v₁ v₂ v₃ : Fin 4 → ℕ),
    v₁ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₂ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₃ ∈ σ (fun i => [a, b, c, d].get i) ∧
    v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧
    (∀ i : Fin 4, v₁ i + v₂ i + v₃ i = s)) →
  d ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_value_l190_19073


namespace NUMINAMATH_CALUDE_blue_marbles_fraction_l190_19016

theorem blue_marbles_fraction (total : ℚ) (h : total > 0) :
  let initial_blue := (2 : ℚ) / 3 * total
  let initial_red := total - initial_blue
  let new_blue := 2 * initial_blue
  let new_total := new_blue + initial_red
  new_blue / new_total = (4 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_fraction_l190_19016


namespace NUMINAMATH_CALUDE_porter_buns_problem_l190_19047

/-- Calculates the maximum number of buns that can be transported to a construction site. -/
def max_buns_transported (total_buns : ℕ) (buns_per_trip : ℕ) (buns_eaten_per_way : ℕ) : ℕ :=
  let num_trips : ℕ := total_buns / buns_per_trip
  let buns_eaten : ℕ := 2 * (num_trips - 1) * buns_eaten_per_way + buns_eaten_per_way
  total_buns - buns_eaten

/-- Theorem stating that given 200 total buns, 40 buns carried per trip, and 1 bun eaten per one-way trip,
    the maximum number of buns that can be transported to the construction site is 191. -/
theorem porter_buns_problem :
  max_buns_transported 200 40 1 = 191 := by
  sorry

end NUMINAMATH_CALUDE_porter_buns_problem_l190_19047


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l190_19075

theorem coefficient_x_squared_expansion : 
  let f : ℕ → ℕ → ℕ := fun n k => Nat.choose n k
  let g : ℕ → ℤ := fun n => (-1)^n
  (f 3 0) * (f 4 2) + (f 3 1) * (f 4 1) * (g 1) + (f 3 2) * 2^2 * (f 4 0) = -6 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l190_19075


namespace NUMINAMATH_CALUDE_double_room_cost_is_60_l190_19032

/-- Represents the hotel booking scenario -/
structure HotelBooking where
  total_rooms : ℕ
  single_room_cost : ℕ
  total_revenue : ℕ
  single_rooms_booked : ℕ

/-- Calculates the cost of a double room given the hotel booking information -/
def double_room_cost (booking : HotelBooking) : ℕ :=
  let double_rooms := booking.total_rooms - booking.single_rooms_booked
  let single_room_revenue := booking.single_rooms_booked * booking.single_room_cost
  let double_room_revenue := booking.total_revenue - single_room_revenue
  double_room_revenue / double_rooms

/-- Theorem stating that the double room cost is 60 for the given scenario -/
theorem double_room_cost_is_60 (booking : HotelBooking) 
  (h1 : booking.total_rooms = 260)
  (h2 : booking.single_room_cost = 35)
  (h3 : booking.total_revenue = 14000)
  (h4 : booking.single_rooms_booked = 64) :
  double_room_cost booking = 60 := by
  sorry

end NUMINAMATH_CALUDE_double_room_cost_is_60_l190_19032


namespace NUMINAMATH_CALUDE_max_min_difference_is_five_l190_19059

/-- Given non-zero real numbers a and b satisfying a² + b² = 25,
    prove that the difference between the maximum and minimum values
    of the function y = (ax + b) / (x² + 1) is 5. -/
theorem max_min_difference_is_five (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 + b^2 = 25) :
  let f : ℝ → ℝ := λ x => (a * x + b) / (x^2 + 1)
  ∃ y₁ y₂ : ℝ, (∀ x, f x ≤ y₁) ∧ (∀ x, f x ≥ y₂) ∧ y₁ - y₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_is_five_l190_19059


namespace NUMINAMATH_CALUDE_a_range_l190_19049

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom increasing_after_one : ∀ x y, 1 ≤ x → x < y → f x < f y
axiom inequality_condition : ∀ x, 1/2 ≤ x → x ≤ 1 → ∃ a, f (a * x) < f (x - 1)

-- Define the theorem
theorem a_range (a : ℝ) : 
  (∀ x, 1/2 ≤ x → x ≤ 1 → f (a * x) < f (x - 1)) → 0 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_a_range_l190_19049


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l190_19005

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 40
  let hole1_diameter : ℝ := 2
  let hole2_diameter : ℝ := 4
  let hole3_diameter : ℝ := 4
  let hole_depth : ℝ := 10
  let sphere_volume := (4/3) * π * (sphere_diameter/2)^3
  let hole1_volume := π * (hole1_diameter/2)^2 * hole_depth
  let hole2_volume := π * (hole2_diameter/2)^2 * hole_depth
  let hole3_volume := π * (hole3_diameter/2)^2 * hole_depth
  sphere_volume - (hole1_volume + hole2_volume + hole3_volume) = (31710/3) * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l190_19005


namespace NUMINAMATH_CALUDE_cafeteria_apples_theorem_l190_19029

/-- Given the initial number of apples, the number of pies made, and the number of apples per pie,
    calculate the number of apples handed out to students. -/
def apples_handed_out (initial_apples : ℕ) (pies_made : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - pies_made * apples_per_pie

/-- Theorem stating that for the given problem, 30 apples were handed out to students. -/
theorem cafeteria_apples_theorem :
  apples_handed_out 86 7 8 = 30 := by
  sorry

#eval apples_handed_out 86 7 8

end NUMINAMATH_CALUDE_cafeteria_apples_theorem_l190_19029


namespace NUMINAMATH_CALUDE_three_over_x_equals_one_l190_19087

theorem three_over_x_equals_one (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_over_x_equals_one_l190_19087


namespace NUMINAMATH_CALUDE_walk_to_restaurant_time_l190_19050

/-- Represents the time in minutes for various walks -/
structure WalkTimes where
  parkToHidden : ℕ
  hiddenToPark : ℕ
  totalTime : ℕ

/-- Calculates the time to walk from Park Office to Lake Park restaurant -/
def timeToLakeParkRestaurant (w : WalkTimes) : ℕ :=
  w.totalTime - (w.parkToHidden + w.hiddenToPark)

/-- Theorem stating that the time to walk from Park Office to Lake Park restaurant is 10 minutes -/
theorem walk_to_restaurant_time (w : WalkTimes) 
  (h1 : w.parkToHidden = 15)
  (h2 : w.hiddenToPark = 7)
  (h3 : w.totalTime = 32) :
  timeToLakeParkRestaurant w = 10 := by
  sorry


end NUMINAMATH_CALUDE_walk_to_restaurant_time_l190_19050


namespace NUMINAMATH_CALUDE_squares_ending_in_identical_digits_l190_19093

def endsIn (n : ℤ) (d : ℤ) : Prop := n % (10 ^ (d.natAbs + 1)) = d

theorem squares_ending_in_identical_digits :
  (∀ n : ℤ, (endsIn n 12 ∨ endsIn n 38 ∨ endsIn n 62 ∨ endsIn n 88) → endsIn (n^2) 44) ∧
  (∀ m : ℤ, (endsIn m 038 ∨ endsIn m 462 ∨ endsIn m 538 ∨ endsIn m 962) → endsIn (m^2) 444) :=
by sorry

end NUMINAMATH_CALUDE_squares_ending_in_identical_digits_l190_19093


namespace NUMINAMATH_CALUDE_water_to_concentrate_ratio_l190_19048

/-- Represents the number of ounces in a serving of orange juice -/
def serving_size : ℕ := 6

/-- Represents the number of servings to be prepared -/
def total_servings : ℕ := 320

/-- Represents the number of ounces in a can of concentrate -/
def can_size : ℕ := 12

/-- Represents the number of cans of concentrate required -/
def concentrate_cans : ℕ := 40

/-- Calculates the total volume of orange juice in ounces -/
def total_volume : ℕ := total_servings * serving_size

/-- Calculates the total number of cans of orange juice -/
def total_cans : ℕ := total_volume / can_size

/-- Calculates the number of cans of water needed -/
def water_cans : ℕ := total_cans - concentrate_cans

/-- Theorem stating that the ratio of water cans to concentrate cans is 3:1 -/
theorem water_to_concentrate_ratio :
  water_cans / concentrate_cans = 3 ∧ water_cans % concentrate_cans = 0 := by
  sorry


end NUMINAMATH_CALUDE_water_to_concentrate_ratio_l190_19048


namespace NUMINAMATH_CALUDE_triangle_area_l190_19019

/-- Given a triangle ABC with side lengths a, b, c, prove that its area is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  b * Real.cos C = c * Real.cos B →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l190_19019


namespace NUMINAMATH_CALUDE_two_numbers_problem_l190_19084

theorem two_numbers_problem (x y : ℝ) :
  (2 * (x + y) = x^2 - y^2) ∧ (2 * (x + y) = x * y / 4 - 56) →
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l190_19084


namespace NUMINAMATH_CALUDE_score_difference_is_1_25_l190_19010

def score_distribution : List (Float × Float) :=
  [(0.20, 70), (0.20, 80), (0.25, 85), (0.25, 90), (0.10, 100)]

def median_score : Float := 85

def mean_score : Float :=
  score_distribution.foldl (λ acc (percent, score) => acc + percent * score) 0

theorem score_difference_is_1_25 :
  median_score - mean_score = 1.25 := by sorry

end NUMINAMATH_CALUDE_score_difference_is_1_25_l190_19010


namespace NUMINAMATH_CALUDE_tobys_journey_l190_19031

/-- Toby's sled-pulling journey --/
theorem tobys_journey (unloaded_speed loaded_speed : ℝ)
  (distance1 distance2 distance3 distance4 : ℝ)
  (h1 : unloaded_speed = 20)
  (h2 : loaded_speed = 10)
  (h3 : distance1 = 180)
  (h4 : distance2 = 120)
  (h5 : distance3 = 80)
  (h6 : distance4 = 140) :
  distance1 / loaded_speed + distance2 / unloaded_speed +
  distance3 / loaded_speed + distance4 / unloaded_speed = 39 := by
  sorry

end NUMINAMATH_CALUDE_tobys_journey_l190_19031


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l190_19062

theorem baseball_card_value_decrease : 
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.1)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 19 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l190_19062


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l190_19045

-- Define the ⊗ operation
noncomputable def bowtie (a b : ℝ) : ℝ := a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem bowtie_equation_solution (h : ℝ) :
  bowtie 4 h = 10 → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l190_19045


namespace NUMINAMATH_CALUDE_unique_face_reconstruction_l190_19077

/-- Represents the numbers on the faces of a cube -/
structure CubeFaces where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Represents the sums on the edges of a cube -/
structure CubeEdges where
  ab : ℝ
  ac : ℝ
  ad : ℝ
  ae : ℝ
  bc : ℝ
  bf : ℝ
  cf : ℝ
  df : ℝ
  de : ℝ
  ef : ℝ
  bd : ℝ
  ce : ℝ

/-- Function to calculate edge sums from face numbers -/
def edgeSumsFromFaces (faces : CubeFaces) : CubeEdges :=
  { ab := faces.a + faces.b
  , ac := faces.a + faces.c
  , ad := faces.a + faces.d
  , ae := faces.a + faces.e
  , bc := faces.b + faces.c
  , bf := faces.b + faces.f
  , cf := faces.c + faces.f
  , df := faces.d + faces.f
  , de := faces.d + faces.e
  , ef := faces.e + faces.f
  , bd := faces.b + faces.d
  , ce := faces.c + faces.e }

/-- Theorem stating that face numbers can be uniquely reconstructed from edge sums -/
theorem unique_face_reconstruction (edges : CubeEdges) : 
  ∃! faces : CubeFaces, edgeSumsFromFaces faces = edges := by
  sorry


end NUMINAMATH_CALUDE_unique_face_reconstruction_l190_19077


namespace NUMINAMATH_CALUDE_rectangle_area_l190_19090

/-- Given a rectangle ABCD divided into six identical squares with a perimeter of 160 cm,
    its area is 1536 square centimeters. -/
theorem rectangle_area (a : ℝ) (h1 : a > 0) : 
  (2 * (3 * a + 2 * a) = 160) → (3 * a) * (2 * a) = 1536 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l190_19090


namespace NUMINAMATH_CALUDE_min_value_implies_a_eq_one_l190_19038

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x - 2 + 2 * a * Real.log x

theorem min_value_implies_a_eq_one (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f a x = 0) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_eq_one_l190_19038


namespace NUMINAMATH_CALUDE_gcd_m5_plus_125_m_plus_3_l190_19041

theorem gcd_m5_plus_125_m_plus_3 (m : ℕ) (h : m > 16) :
  Nat.gcd (m^5 + 5^3) (m + 3) = if (m + 3) % 27 ≠ 0 then 1 else Nat.gcd 27 (m + 3) := by
  sorry

end NUMINAMATH_CALUDE_gcd_m5_plus_125_m_plus_3_l190_19041


namespace NUMINAMATH_CALUDE_functional_equation_solution_l190_19026

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f x * f y) = x * f (x + y)

/-- The theorem stating that any function satisfying the functional equation
    must be one of the three specified functions -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l190_19026


namespace NUMINAMATH_CALUDE_andrey_gifts_l190_19000

theorem andrey_gifts :
  ∀ (n : ℕ) (a : ℕ),
    n > 2 →
    n * (n - 2) = a * (n - 1) + 16 →
    n = 18 :=
by sorry

end NUMINAMATH_CALUDE_andrey_gifts_l190_19000


namespace NUMINAMATH_CALUDE_third_number_in_expression_l190_19083

theorem third_number_in_expression (x : ℝ) : 
  (26.3 * 12 * x) / 3 + 125 = 2229 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_expression_l190_19083


namespace NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_approx_42_l190_19085

/-- The diameter of a circular field given the fencing cost per meter and total fencing cost -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- Proof that the diameter of the circular field is approximately 42 meters -/
theorem circular_field_diameter_approx_42 :
  ∃ ε > 0, |circular_field_diameter 5 659.73 - 42| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_approx_42_l190_19085


namespace NUMINAMATH_CALUDE_lassis_and_smoothies_count_l190_19044

/-- Represents the number of lassis that can be made from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (15 * mangoes) / 3

/-- Represents the number of smoothies that can be made from given numbers of mangoes and bananas -/
def smoothies_from_ingredients (mangoes bananas : ℕ) : ℕ :=
  min mangoes (bananas / 2)

/-- Theorem stating the number of lassis and smoothies that can be made -/
theorem lassis_and_smoothies_count :
  lassis_from_mangoes 18 = 90 ∧ smoothies_from_ingredients 18 36 = 18 :=
by sorry

end NUMINAMATH_CALUDE_lassis_and_smoothies_count_l190_19044
