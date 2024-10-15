import Mathlib

namespace NUMINAMATH_CALUDE_max_value_cos_sin_l2218_221814

theorem max_value_cos_sin : 
  ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y + 4 * Real.sin y = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l2218_221814


namespace NUMINAMATH_CALUDE_jeff_tennis_time_l2218_221836

/-- Proves that Jeff played tennis for 2 hours given the conditions -/
theorem jeff_tennis_time (
  points_per_match : ℕ) 
  (minutes_per_point : ℕ) 
  (matches_won : ℕ) 
  (h1 : points_per_match = 8)
  (h2 : minutes_per_point = 5)
  (h3 : matches_won = 3)
  : (points_per_match * matches_won * minutes_per_point) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jeff_tennis_time_l2218_221836


namespace NUMINAMATH_CALUDE_least_multiple_of_smallest_primes_gt_5_l2218_221849

def smallest_primes_gt_5 : List Nat := [7, 11, 13]

theorem least_multiple_of_smallest_primes_gt_5 :
  (∀ n : Nat, n > 0 ∧ (∀ p ∈ smallest_primes_gt_5, p ∣ n) → n ≥ 1001) ∧
  (∀ p ∈ smallest_primes_gt_5, p ∣ 1001) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_of_smallest_primes_gt_5_l2218_221849


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2218_221815

/-- A regular nonagon is a 9-sided polygon with all sides and angles equal -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- The probability that two randomly chosen diagonals of a regular nonagon intersect inside the nonagon -/
def intersectionProbability (n : RegularNonagon) : ℚ :=
  6 / 13

/-- Theorem: The probability that two randomly chosen diagonals of a regular nonagon 
    intersect inside the nonagon is 6/13 -/
theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) : 
  intersectionProbability n = 6 / 13 := by
  sorry


end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l2218_221815


namespace NUMINAMATH_CALUDE_min_value_vector_expr_l2218_221802

/-- Given plane vectors a, b, and c satisfying certain conditions, 
    the minimum value of a specific vector expression is 1/2. -/
theorem min_value_vector_expr 
  (a b c : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖c‖ = 2) 
  (h4 : a + b + c = (0, 0)) :
  ∃ (min : ℝ), min = 1/2 ∧ 
  ∀ (x y : ℝ), 0 ≤ x → x ≤ 1/2 → 1/2 ≤ y → y ≤ 1 →
  ‖x • (a - c) + y • (b - c) + c‖ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_vector_expr_l2218_221802


namespace NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l2218_221829

/-- The infinite sum of n / (n^4 - 4n^2 + 8) from n=1 to infinity equals 5/24 -/
theorem infinite_sum_equals_five_twentyfourths :
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 - 4*(n : ℝ)^2 + 8) = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_five_twentyfourths_l2218_221829


namespace NUMINAMATH_CALUDE_present_age_of_B_l2218_221859

-- Define the ages of A and B as natural numbers
variable (A B : ℕ)

-- Define the conditions
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 7

-- Theorem statement
theorem present_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 := by
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_l2218_221859


namespace NUMINAMATH_CALUDE_apples_to_friends_l2218_221816

theorem apples_to_friends (initial_apples : ℕ) (apples_left : ℕ) (apples_to_teachers : ℕ) (apples_eaten : ℕ) :
  initial_apples = 25 →
  apples_left = 3 →
  apples_to_teachers = 16 →
  apples_eaten = 1 →
  initial_apples - apples_left - apples_to_teachers - apples_eaten = 5 :=
by sorry

end NUMINAMATH_CALUDE_apples_to_friends_l2218_221816


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2218_221820

/-- Given an equation mx^2 - my^2 = n where m and n are real numbers and mn < 0,
    the curve represented by this equation is a hyperbola with foci on the y-axis. -/
theorem equation_represents_hyperbola (m n : ℝ) (h : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), m * x^2 - m * y^2 = n ↔ y^2 / a^2 - x^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2218_221820


namespace NUMINAMATH_CALUDE_gravel_path_cost_l2218_221882

/-- Calculate the cost of gravelling a path around a rectangular plot -/
theorem gravel_path_cost 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm : ℝ) : 
  plot_length = 110 →
  plot_width = 65 →
  path_width = 2.5 →
  cost_per_sqm = 0.7 →
  ((plot_length + 2 * path_width) * (plot_width + 2 * path_width) - plot_length * plot_width) * cost_per_sqm = 630 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l2218_221882


namespace NUMINAMATH_CALUDE_parallel_non_existent_slopes_intersect_one_non_existent_slope_line_equation_through_two_points_l2218_221898

-- Define a straight line in a coordinate plane
structure Line where
  slope : Option ℝ
  point : ℝ × ℝ

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

-- Theorem 1: If the slopes of two lines do not exist, then the two lines are parallel
theorem parallel_non_existent_slopes (l1 l2 : Line) :
  l1.slope = none ∧ l2.slope = none → parallel l1 l2 := by sorry

-- Theorem 2: If one of two lines has a non-existent slope and the other has a slope, 
-- then the two lines intersect
theorem intersect_one_non_existent_slope (l1 l2 : Line) :
  (l1.slope = none ∧ l2.slope ≠ none) ∨ (l1.slope ≠ none ∧ l2.slope = none) 
  → intersect l1 l2 := by sorry

-- Theorem 3: The equation of the line passing through any two different points 
-- P₁(x₁, y₁), P₂(x₂, y₂) is (x₂-x₁)(y-y₁)=(y₂-y₁)(x-x₁)
theorem line_equation_through_two_points (P1 P2 : ℝ × ℝ) (x y : ℝ) :
  P1 ≠ P2 → (P2.1 - P1.1) * (y - P1.2) = (P2.2 - P1.2) * (x - P1.1) := by sorry

end NUMINAMATH_CALUDE_parallel_non_existent_slopes_intersect_one_non_existent_slope_line_equation_through_two_points_l2218_221898


namespace NUMINAMATH_CALUDE_truck_rental_miles_driven_l2218_221844

theorem truck_rental_miles_driven 
  (rental_fee : ℝ) 
  (charge_per_mile : ℝ) 
  (total_paid : ℝ) 
  (h1 : rental_fee = 20.99)
  (h2 : charge_per_mile = 0.25)
  (h3 : total_paid = 95.74) : 
  ⌊(total_paid - rental_fee) / charge_per_mile⌋ = 299 := by
sorry


end NUMINAMATH_CALUDE_truck_rental_miles_driven_l2218_221844


namespace NUMINAMATH_CALUDE_candy_distribution_properties_l2218_221845

-- Define the people
inductive Person : Type
| Chun : Person
| Tian : Person
| Zhen : Person
| Mei : Person
| Li : Person

-- Define the order of taking candies
def Order := Fin 5 → Person

-- Define the number of candies taken by each person
def CandiesTaken := Person → ℕ

-- Define the properties of the candy distribution
structure CandyDistribution where
  order : Order
  candiesTaken : CandiesTaken
  initialCandies : ℕ
  allDifferent : ∀ (p q : Person), p ≠ q → candiesTaken p ≠ candiesTaken q
  tianHalf : candiesTaken Person.Tian = (initialCandies - candiesTaken Person.Chun) / 2
  zhenTwoThirds : candiesTaken Person.Zhen = 2 * (initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian - candiesTaken Person.Li) / 3
  meiAll : candiesTaken Person.Mei = initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian - candiesTaken Person.Zhen - candiesTaken Person.Li
  liHalf : candiesTaken Person.Li = (initialCandies - candiesTaken Person.Chun - candiesTaken Person.Tian) / 2

-- Theorem statement
theorem candy_distribution_properties (d : CandyDistribution) :
  (∃ i : Fin 5, d.order i = Person.Zhen ∧ i = 3) ∧
  d.initialCandies ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_properties_l2218_221845


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2218_221826

/-- A quadratic function f(x) = (x + a)(bx + 2a) where a, b ∈ ℝ, 
    which is even and has a range of (-∞, 4] -/
def quadratic_function (a b : ℝ) : ℝ → ℝ := fun x ↦ (x + a) * (b * x + 2 * a)

/-- The property of being an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The range of a function -/
def has_range (f : ℝ → ℝ) (S : Set ℝ) : Prop := ∀ y, y ∈ S ↔ ∃ x, f x = y

theorem quadratic_function_theorem (a b : ℝ) :
  is_even (quadratic_function a b) ∧ 
  has_range (quadratic_function a b) {y | y ≤ 4} →
  quadratic_function a b = fun x ↦ -2 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2218_221826


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l2218_221896

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l2218_221896


namespace NUMINAMATH_CALUDE_haley_candy_count_l2218_221899

/-- The number of candy pieces Haley has at the end -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem stating that Haley's final candy count is 35 -/
theorem haley_candy_count :
  final_candy_count 33 17 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_haley_candy_count_l2218_221899


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l2218_221879

/-- Given an ellipse that passes through a point (a, b), prove the minimum value of m + n -/
theorem ellipse_min_sum (a b m n : ℝ) : 
  m > 0 → n > 0 → m > n → a ≠ 0 → b ≠ 0 → abs a ≠ abs b →
  (a^2 / m^2) + (b^2 / n^2) = 1 →
  ∀ m' n', m' > 0 → n' > 0 → m' > n' → (a^2 / m'^2) + (b^2 / n'^2) = 1 →
  m + n ≤ m' + n' →
  m + n = (a^(2/3) + b^(2/3))^(3/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l2218_221879


namespace NUMINAMATH_CALUDE_turban_price_l2218_221853

theorem turban_price (annual_salary : ℝ) (turban_price : ℝ) (work_fraction : ℝ) (partial_payment : ℝ) :
  annual_salary = 90 ∧ 
  work_fraction = 3/4 ∧ 
  work_fraction * (annual_salary + turban_price) = partial_payment + turban_price ∧
  partial_payment = 45 →
  turban_price = 90 := by sorry

end NUMINAMATH_CALUDE_turban_price_l2218_221853


namespace NUMINAMATH_CALUDE_justin_tim_emily_games_l2218_221828

/-- The total number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in the larger game -/
def larger_game_players : ℕ := 7

/-- The number of specific players (Justin, Tim, and Emily) -/
def specific_players : ℕ := 3

theorem justin_tim_emily_games (h : total_players = 12 ∧ larger_game_players = 7 ∧ specific_players = 3) :
  Nat.choose (total_players - specific_players) (larger_game_players - specific_players) = 126 := by
  sorry

end NUMINAMATH_CALUDE_justin_tim_emily_games_l2218_221828


namespace NUMINAMATH_CALUDE_total_bees_after_changes_l2218_221839

/-- Represents a bee hive with initial bees and changes in population --/
structure BeeHive where
  initial : ℕ
  fly_in : ℕ
  fly_out : ℕ

/-- Calculates the final number of bees in a hive after changes --/
def final_bees (hive : BeeHive) : ℕ :=
  hive.initial + hive.fly_in - hive.fly_out

/-- Represents the bee colony --/
def BeeColony : List BeeHive := [
  { initial := 45, fly_in := 12, fly_out := 8 },
  { initial := 60, fly_in := 15, fly_out := 20 },
  { initial := 75, fly_in := 10, fly_out := 5 }
]

/-- Theorem stating the total number of bees after changes --/
theorem total_bees_after_changes :
  (BeeColony.map final_bees).sum = 184 := by
  sorry

end NUMINAMATH_CALUDE_total_bees_after_changes_l2218_221839


namespace NUMINAMATH_CALUDE_pulled_pork_sandwiches_l2218_221804

def total_sauce : ℚ := 5
def burger_sauce : ℚ := 1/4
def sandwich_sauce : ℚ := 1/6
def num_burgers : ℕ := 8

theorem pulled_pork_sandwiches :
  ∃ (n : ℕ), n * sandwich_sauce + num_burgers * burger_sauce = total_sauce ∧ n = 18 :=
by sorry

end NUMINAMATH_CALUDE_pulled_pork_sandwiches_l2218_221804


namespace NUMINAMATH_CALUDE_pass_rate_two_steps_l2218_221876

/-- The pass rate of a product going through two independent processing steps -/
def product_pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

/-- Theorem stating that the pass rate of a product going through two independent
    processing steps with defect rates a and b is (1-a) * (1-b) -/
theorem pass_rate_two_steps (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  product_pass_rate a b = (1 - a) * (1 - b) := by
  sorry

#check pass_rate_two_steps

end NUMINAMATH_CALUDE_pass_rate_two_steps_l2218_221876


namespace NUMINAMATH_CALUDE_water_volume_for_four_balls_l2218_221822

/-- The volume of water needed to cover four touching balls in a cylinder -/
theorem water_volume_for_four_balls (r ball_radius container_radius : ℝ) 
  (h_ball_radius : ball_radius = 0.5)
  (h_container_radius : container_radius = 1) :
  let water_height := container_radius + ball_radius
  let cylinder_volume := π * container_radius^2 * water_height
  let ball_volume := (4/3) * π * ball_radius^3
  cylinder_volume - 4 * ball_volume = (2/3) * π := by sorry

end NUMINAMATH_CALUDE_water_volume_for_four_balls_l2218_221822


namespace NUMINAMATH_CALUDE_intersection_A_B_l2218_221866

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - 2^x)}

-- Define set B
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2218_221866


namespace NUMINAMATH_CALUDE_triangle_inequality_l2218_221871

/-- Triangle inequality for sides and area -/
theorem triangle_inequality (a b c S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : S = Real.sqrt (((a + b + c) / 2) * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l2218_221871


namespace NUMINAMATH_CALUDE_tangent_values_l2218_221830

/-- Two linear functions with parallel non-vertical graphs -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  hf : f = λ x => a * x + b
  hg : g = λ x => a * x + c
  ha : a ≠ 0

/-- The property that (f x)^2 is tangent to -12(g x) -/
def is_tangent_f_g (p : ParallelLinearFunctions) : Prop :=
  ∃! x, (p.f x)^2 = -12 * (p.g x)

/-- The main theorem -/
theorem tangent_values (p : ParallelLinearFunctions) 
  (h : is_tangent_f_g p) :
  ∃ A : Set ℝ, A = {0, 12} ∧ 
  ∀ a : ℝ, a ∈ A ↔ ∃! x, (p.g x)^2 = a * (p.f x) := by
  sorry

end NUMINAMATH_CALUDE_tangent_values_l2218_221830


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2218_221807

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → x₂^2 - 3*x₂ - 1 = 0 → x₁ ≠ x₂ → 
  (1/x₁ + 1/x₂ : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2218_221807


namespace NUMINAMATH_CALUDE_dogsled_race_time_difference_l2218_221838

theorem dogsled_race_time_difference 
  (course_length : ℝ) 
  (speed_T : ℝ) 
  (speed_difference : ℝ) :
  course_length = 300 →
  speed_T = 20 →
  speed_difference = 5 →
  let speed_A := speed_T + speed_difference
  let time_T := course_length / speed_T
  let time_A := course_length / speed_A
  time_T - time_A = 3 := by
sorry

end NUMINAMATH_CALUDE_dogsled_race_time_difference_l2218_221838


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2218_221894

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 9 = 0 ∧ 
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = (65 + 81 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2218_221894


namespace NUMINAMATH_CALUDE_factor_implies_q_value_l2218_221831

theorem factor_implies_q_value (m q : ℤ) : 
  (∃ k : ℤ, m^2 - q*m - 24 = (m - 8) * k) → q = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_q_value_l2218_221831


namespace NUMINAMATH_CALUDE_mixture_weight_l2218_221842

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_volume : ℝ) : 
  weight_a = 900 →
  weight_b = 800 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  (((ratio_a : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ))) * total_volume * weight_a +
   ((ratio_b : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ))) * total_volume * weight_b) / 1000 = 3.44 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_mixture_weight_l2218_221842


namespace NUMINAMATH_CALUDE_double_angle_formulas_l2218_221840

open Real

theorem double_angle_formulas (α p q : ℝ) (h : tan α = p / q) :
  sin (2 * α) = (2 * p * q) / (p^2 + q^2) ∧
  cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  tan (2 * α) = (2 * p * q) / (q^2 - p^2) := by
  sorry

end NUMINAMATH_CALUDE_double_angle_formulas_l2218_221840


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l2218_221893

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The sum of coordinates of two points -/
def sum_of_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_sum_coordinates :
  let C : Point := { x := 3, y := 8 }
  let D : Point := reflect_over_y_axis C
  sum_of_coordinates C D = 16 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l2218_221893


namespace NUMINAMATH_CALUDE_function_upper_bound_l2218_221818

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, f x ≥ 0)
  (h2 : f 1 = 1)
  (h3 : ∀ x₁ x₂, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≤ f x₁ + f x₂) :
  ∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x :=
by
  sorry


end NUMINAMATH_CALUDE_function_upper_bound_l2218_221818


namespace NUMINAMATH_CALUDE_final_price_is_66_percent_l2218_221805

/-- The percentage of the suggested retail price paid after discounts and tax -/
def final_price_percentage (initial_discount : ℝ) (clearance_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  (1 - initial_discount) * (1 - clearance_discount) * (1 + sales_tax)

/-- Theorem stating that the final price is 66% of the suggested retail price -/
theorem final_price_is_66_percent :
  final_price_percentage 0.2 0.25 0.1 = 0.66 := by
  sorry

#eval final_price_percentage 0.2 0.25 0.1

end NUMINAMATH_CALUDE_final_price_is_66_percent_l2218_221805


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2218_221869

theorem complex_absolute_value (z : ℂ) : z = 7 + 3*I → Complex.abs (z^2 + 8*z + 65) = Real.sqrt 30277 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2218_221869


namespace NUMINAMATH_CALUDE_students_present_l2218_221803

/-- Given a class of 100 students with 14% absent, prove that the number of students present is 86. -/
theorem students_present (total_students : ℕ) (absent_percentage : ℚ) : 
  total_students = 100 → 
  absent_percentage = 14/100 → 
  (total_students : ℚ) * (1 - absent_percentage) = 86 :=
by sorry

end NUMINAMATH_CALUDE_students_present_l2218_221803


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_open_2_closed_3_l2218_221841

-- Define set A
def A : Set ℝ := {x | -1 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 5}

-- Define set B (domain of log(-x^2 + 6x - 8))
def B : Set ℝ := {x | -x^2 + 6*x - 8 > 0}

-- Theorem to prove
theorem A_intersect_B_equals_open_2_closed_3 : A ∩ B = {x | 2 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_open_2_closed_3_l2218_221841


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_planes_l2218_221863

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_planes 
  (m : Line) (α β : Plane) :
  parallel m α → perpendicular m β → perpendicularPlanes α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_planes_l2218_221863


namespace NUMINAMATH_CALUDE_S_minimum_at_n_min_l2218_221809

/-- The sequence a_n with general term 2n - 49 -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

/-- The value of n for which S_n reaches its minimum -/
def n_min : ℕ := 24

theorem S_minimum_at_n_min :
  ∀ k : ℕ, k ≠ 0 → S n_min ≤ S k :=
sorry

end NUMINAMATH_CALUDE_S_minimum_at_n_min_l2218_221809


namespace NUMINAMATH_CALUDE_fraction_problem_l2218_221856

theorem fraction_problem :
  ∃ x : ℚ, x * 180 = 18 ∧ x < 0.15 → x = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2218_221856


namespace NUMINAMATH_CALUDE_least_common_multiple_problem_l2218_221837

def is_divisible_by_all (n : ℕ) : Prop :=
  n % 24 = 0 ∧ n % 32 = 0 ∧ n % 36 = 0 ∧ n % 54 = 0

theorem least_common_multiple_problem : 
  ∃! x : ℕ, (is_divisible_by_all (856 + x) ∧ 
    ∀ y : ℕ, y < x → ¬is_divisible_by_all (856 + y)) ∧ 
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_problem_l2218_221837


namespace NUMINAMATH_CALUDE_beads_per_necklace_l2218_221843

theorem beads_per_necklace (total_beads : ℕ) (total_necklaces : ℕ) 
  (h1 : total_beads = 20) 
  (h2 : total_necklaces = 4) : 
  total_beads / total_necklaces = 5 :=
by sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l2218_221843


namespace NUMINAMATH_CALUDE_animals_remaining_l2218_221858

theorem animals_remaining (cows dogs : ℕ) : 
  cows = 2 * dogs →
  cows = 184 →
  (184 - 184 / 4) + (dogs - 3 * dogs / 4) = 161 := by
sorry

end NUMINAMATH_CALUDE_animals_remaining_l2218_221858


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2218_221887

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 2 * |x - 1| - 1 < 0} = {x : ℝ | 1/2 < x ∧ x < 3/2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2218_221887


namespace NUMINAMATH_CALUDE_triangle_properties_l2218_221864

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.c * Real.cos t.B = (2 * t.a - t.b) * Real.cos t.C ∧
  t.c = 2 ∧
  t.a + t.b + t.c = 2 * Real.sqrt 3 + 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  t.C = π / 3 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C : ℝ) = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2218_221864


namespace NUMINAMATH_CALUDE_percentage_difference_l2218_221825

theorem percentage_difference : 
  (60 * 80 / 100) - (25 * 4 / 5) = 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2218_221825


namespace NUMINAMATH_CALUDE_largest_unreachable_proof_l2218_221813

/-- The largest integer that cannot be expressed as a non-negative linear combination of 17 and 11 -/
def largest_unreachable : ℕ := 159

/-- The width of the paper in half-inches -/
def paper_width : ℕ := 17

/-- The length of the paper in inches -/
def paper_length : ℕ := 11

/-- A predicate that checks if a natural number can be expressed as a non-negative linear combination of paper_width and paper_length -/
def is_reachable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * paper_width + b * paper_length

theorem largest_unreachable_proof :
  (∀ n > largest_unreachable, is_reachable n) ∧
  ¬is_reachable largest_unreachable :=
sorry

end NUMINAMATH_CALUDE_largest_unreachable_proof_l2218_221813


namespace NUMINAMATH_CALUDE_third_square_perimeter_l2218_221895

/-- Given two squares with perimeters 40 cm and 32 cm, prove that a third square
    whose area is equal to the difference of the areas of the first two squares
    has a perimeter of 24 cm. -/
theorem third_square_perimeter (square1 square2 square3 : Real → Real → Real) :
  (∀ s, square1 s s = s * s) →
  (∀ s, square2 s s = s * s) →
  (∀ s, square3 s s = s * s) →
  (4 * 10 = 40) →
  (4 * 8 = 32) →
  (square1 10 10 - square2 8 8 = square3 6 6) →
  (4 * 6 = 24) := by
sorry

end NUMINAMATH_CALUDE_third_square_perimeter_l2218_221895


namespace NUMINAMATH_CALUDE_game_a_vs_game_b_l2218_221860

def p_heads : ℚ := 2/3
def p_tails : ℚ := 1/3

def p_win_game_a : ℚ := p_heads^3 + p_tails^3

def p_same_pair : ℚ := p_heads^2 + p_tails^2
def p_win_game_b : ℚ := p_same_pair^2

theorem game_a_vs_game_b : p_win_game_a - p_win_game_b = 2/81 := by
  sorry

end NUMINAMATH_CALUDE_game_a_vs_game_b_l2218_221860


namespace NUMINAMATH_CALUDE_projectile_max_height_l2218_221874

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 45

/-- Theorem: The maximum height reached by the projectile is 153 feet -/
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 153 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2218_221874


namespace NUMINAMATH_CALUDE_line_inclination_through_origin_and_negative_one_l2218_221823

/-- The angle of inclination of a line passing through (0, 0) and (-1, -1) is 45°. -/
theorem line_inclination_through_origin_and_negative_one : ∃ (α : ℝ), 
  (∀ (x y : ℝ), y = x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) →
  α * (π / 180) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_inclination_through_origin_and_negative_one_l2218_221823


namespace NUMINAMATH_CALUDE_exponent_division_l2218_221824

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2218_221824


namespace NUMINAMATH_CALUDE_side_significant_digits_l2218_221850

/-- The area of the square in square meters -/
def area : ℝ := 2.7509

/-- The precision of the area measurement in square meters -/
def precision : ℝ := 0.0001

/-- The number of significant digits in the measurement of the side of the square -/
def significant_digits : ℕ := 5

/-- Theorem stating that the number of significant digits in the measurement of the side of the square is 5 -/
theorem side_significant_digits : 
  ∀ (side : ℝ), side^2 = area → significant_digits = 5 := by
  sorry

end NUMINAMATH_CALUDE_side_significant_digits_l2218_221850


namespace NUMINAMATH_CALUDE_tshirt_production_l2218_221862

/-- The number of minutes in an hour -/
def minutesPerHour : ℕ := 60

/-- The rate of t-shirt production in the first hour (minutes per t-shirt) -/
def rateFirstHour : ℕ := 12

/-- The rate of t-shirt production in the second hour (minutes per t-shirt) -/
def rateSecondHour : ℕ := 6

/-- The total number of t-shirts produced in two hours -/
def totalTShirts : ℕ := minutesPerHour / rateFirstHour + minutesPerHour / rateSecondHour

theorem tshirt_production : totalTShirts = 15 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_production_l2218_221862


namespace NUMINAMATH_CALUDE_min_top_managers_bound_l2218_221832

/-- Represents the structure of a company with its employees and order distribution system. -/
structure Company where
  total_employees : ℕ
  direct_connections : ℕ
  distribution_days : ℕ
  (total_employees_positive : total_employees > 0)
  (direct_connections_positive : direct_connections > 0)
  (distribution_days_positive : distribution_days > 0)

/-- Calculates the minimum number of top-level managers in the company. -/
def min_top_managers (c : Company) : ℕ :=
  ((c.total_employees - 1) / (c.direct_connections^(c.distribution_days + 1) - 1)) + 1

/-- Theorem stating that a company with 50,000 employees, 7 direct connections per employee, 
    and 4 distribution days has at least 28 top-level managers. -/
theorem min_top_managers_bound (c : Company) 
  (h1 : c.total_employees = 50000)
  (h2 : c.direct_connections = 7)
  (h3 : c.distribution_days = 4) :
  min_top_managers c ≥ 28 := by
  sorry

#eval min_top_managers ⟨50000, 7, 4, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_min_top_managers_bound_l2218_221832


namespace NUMINAMATH_CALUDE_find_T_l2218_221861

theorem find_T : ∃ T : ℚ, (1/3 : ℚ) * (1/4 : ℚ) * T = (1/3 : ℚ) * (1/5 : ℚ) * 120 ∧ T = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l2218_221861


namespace NUMINAMATH_CALUDE_basketball_scores_l2218_221819

/-- The number of different total point scores for a basketball player who made 7 baskets,
    each worth either 2 or 3 points. -/
def differentScores : ℕ := by sorry

theorem basketball_scores :
  let totalBaskets : ℕ := 7
  let twoPointValue : ℕ := 2
  let threePointValue : ℕ := 3
  differentScores = 8 := by sorry

end NUMINAMATH_CALUDE_basketball_scores_l2218_221819


namespace NUMINAMATH_CALUDE_bookshelf_cost_price_l2218_221884

/-- The cost price of a bookshelf sold at a loss and would have made a profit with additional revenue -/
theorem bookshelf_cost_price (C : ℝ) : C = 1071.43 :=
  let SP := 0.76 * C
  have h1 : SP = 0.76 * C := by rfl
  have h2 : SP + 450 = 1.18 * C := by sorry
  sorry

end NUMINAMATH_CALUDE_bookshelf_cost_price_l2218_221884


namespace NUMINAMATH_CALUDE_number_equation_solution_l2218_221881

theorem number_equation_solution : ∃ x : ℝ, (27 + 2 * x = 39) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2218_221881


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_square_sum_digits_l2218_221857

def isValidNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (let digits := [n / 100, (n / 10) % 10, n % 10]
   digits.toFinset.card = 3 ∧
   n % ((digits.sum)^2) = 0)

theorem three_digit_divisible_by_square_sum_digits :
  {n : ℕ | isValidNumber n} =
  {162, 243, 324, 405, 512, 648, 729, 810, 972} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_square_sum_digits_l2218_221857


namespace NUMINAMATH_CALUDE_percentage_of_students_taking_music_l2218_221890

/-- Calculates the percentage of students taking music in a school -/
theorem percentage_of_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (drama_students : ℕ)
  (h1 : total_students = 2000)
  (h2 : dance_students = 450)
  (h3 : art_students = 680)
  (h4 : drama_students = 370) :
  (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_taking_music_l2218_221890


namespace NUMINAMATH_CALUDE_iterated_forward_difference_of_exponential_l2218_221848

def f (n : ℕ) : ℕ := 3^n

def forwardDifference (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

def iteratedForwardDifference (g : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => g
  | k + 1 => forwardDifference (iteratedForwardDifference g k)

theorem iterated_forward_difference_of_exponential (k : ℕ) (h : k ≥ 1) :
  ∀ n, iteratedForwardDifference f k n = 2^k * 3^n := by
  sorry

end NUMINAMATH_CALUDE_iterated_forward_difference_of_exponential_l2218_221848


namespace NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l2218_221855

/-- The "Heap of Stones" game --/
structure HeapOfStones where
  initial_stones : Nat
  max_stones_per_turn : Nat

/-- A player in the game --/
inductive Player
  | Petya
  | Computer

/-- The game state --/
structure GameState where
  stones_left : Nat
  current_player : Player

/-- The result of a game --/
inductive GameResult
  | PetyaWins
  | ComputerWins

/-- The strategy for the computer player --/
def computer_strategy : GameState → Nat := sorry

/-- The random strategy for Petya --/
def petya_random_strategy : GameState → Nat := sorry

/-- Play a single game --/
def play_game (game : HeapOfStones) : GameResult := sorry

/-- Calculate the probability of Petya winning --/
def petya_win_probability (game : HeapOfStones) : ℚ := sorry

/-- The main theorem --/
theorem petya_win_probability_is_1_256 :
  let game : HeapOfStones := ⟨16, 4⟩
  petya_win_probability game = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l2218_221855


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l2218_221865

theorem quadratic_equation_completion_square (x : ℝ) :
  (16 * x^2 - 32 * x - 512 = 0) →
  ∃ (k m : ℝ), ((x + k)^2 = m) ∧ (m = 65) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l2218_221865


namespace NUMINAMATH_CALUDE_unique_angle_l2218_221833

def is_valid_angle (a b c d e f : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  10 * a + b < 90 ∧
  10 * c + d < 60 ∧
  10 * e + f < 60

def is_complement (a b c d e f a1 b1 c1 d1 e1 f1 : ℕ) : Prop :=
  (10 * a + b) + (10 * a1 + b1) = 89 ∧
  (10 * c + d) + (10 * c1 + d1) = 59 ∧
  (10 * e + f) + (10 * e1 + f1) = 60

def is_rearrangement (a b c d e f a1 b1 c1 d1 e1 f1 : ℕ) : Prop :=
  ∃ (n m : ℕ), n + m = 6 ∧ n ≤ m ∧
  (10^n + 1) * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) +
  (10^m + 1) * (100000 * a1 + 10000 * b1 + 1000 * c1 + 100 * d1 + 10 * e1 + f1) = 895960

theorem unique_angle :
  ∀ (a b c d e f : ℕ),
    is_valid_angle a b c d e f →
    (∃ (a1 b1 c1 d1 e1 f1 : ℕ),
      is_valid_angle a1 b1 c1 d1 e1 f1 ∧
      is_complement a b c d e f a1 b1 c1 d1 e1 f1 ∧
      is_rearrangement a b c d e f a1 b1 c1 d1 e1 f1) →
    a = 4 ∧ b = 5 ∧ c = 4 ∧ d = 4 ∧ e = 1 ∧ f = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_angle_l2218_221833


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2218_221810

theorem ellipse_eccentricity_range (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), 
    (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    ((x + c)^2 + y^2) * ((x - c)^2 + y^2) = (2*c^2)^2 →
    (1/2 : ℝ) ≤ c/a ∧ c/a ≤ (Real.sqrt 3)/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2218_221810


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l2218_221812

theorem integer_ratio_problem (a b : ℤ) : 
  1996 * a + b / 96 = a + b → b / a = 2016 ∨ a / b = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l2218_221812


namespace NUMINAMATH_CALUDE_evaluate_expression_l2218_221800

theorem evaluate_expression : (-2 : ℤ) ^ (4^2) + 2^(4^2) = 2^17 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2218_221800


namespace NUMINAMATH_CALUDE_train_problem_solution_l2218_221875

/-- Represents the train problem scenario -/
structure TrainProblem where
  totalDistance : ℝ
  trainBTime : ℝ
  meetingPointA : ℝ
  trainATime : ℝ

/-- The solution to the train problem -/
def solveTrain (p : TrainProblem) : Prop :=
  p.totalDistance = 125 ∧
  p.trainBTime = 8 ∧
  p.meetingPointA = 50 ∧
  p.trainATime = 12

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem train_problem_solution :
  ∀ (p : TrainProblem),
    p.totalDistance = 125 ∧
    p.trainBTime = 8 ∧
    p.meetingPointA = 50 →
    solveTrain p :=
by
  sorry

#check train_problem_solution

end NUMINAMATH_CALUDE_train_problem_solution_l2218_221875


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2218_221892

theorem unique_solution_for_equation :
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2218_221892


namespace NUMINAMATH_CALUDE_saree_sale_price_l2218_221867

/-- Calculates the final price of a saree after discounts and tax -/
def finalSalePrice (initialPrice : ℝ) (discount1 discount2 discount3 taxRate : ℝ) : ℝ :=
  let price1 := initialPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * (1 + taxRate)

/-- The final sale price of a saree is approximately 298.55 Rs -/
theorem saree_sale_price :
  ∃ ε > 0, abs (finalSalePrice 560 0.2 0.3 0.15 0.12 - 298.55) < ε :=
sorry

end NUMINAMATH_CALUDE_saree_sale_price_l2218_221867


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l2218_221854

theorem quadratic_non_real_roots (b : ℝ) :
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l2218_221854


namespace NUMINAMATH_CALUDE_count_factorizable_pairs_eq_325_l2218_221873

/-- Counts the number of ordered pairs (a,b) satisfying the factorization condition -/
def count_factorizable_pairs : ℕ :=
  (Finset.range 50).sum (λ a => (a + 1) / 2)

/-- The main theorem stating that the count of factorizable pairs is 325 -/
theorem count_factorizable_pairs_eq_325 : count_factorizable_pairs = 325 := by
  sorry


end NUMINAMATH_CALUDE_count_factorizable_pairs_eq_325_l2218_221873


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2218_221870

theorem complex_fraction_calculation : 
  (2 + 2/3 : ℚ) * ((1/3 - 1/11) / (1/11 + 1/5)) / (8/27) = 7 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2218_221870


namespace NUMINAMATH_CALUDE_triangle_property_l2218_221821

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  2 * b * Real.cos A = c * Real.cos A + a * Real.cos C ∧
  a = 4 →
  A = π / 3 ∧ 
  (∀ (b' c' : ℝ), b' > 0 → c' > 0 → 
    4 * 4 = b' * b' + c' * c' - b' * c' → 
    1/2 * b' * c' * Real.sin A ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2218_221821


namespace NUMINAMATH_CALUDE_complex_number_properties_l2218_221834

theorem complex_number_properties (z : ℂ) (h : (2 + Complex.I) * z = 1 + 3 * Complex.I) :
  (Complex.abs z = Real.sqrt 2) ∧ (z^2 - 2*z + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2218_221834


namespace NUMINAMATH_CALUDE_product_closest_to_1200_l2218_221872

def product : ℝ := 0.000315 * 3928500

def options : List ℝ := [1100, 1200, 1300, 1400]

theorem product_closest_to_1200 : 
  1200 ∈ options ∧ ∀ x ∈ options, |product - 1200| ≤ |product - x| :=
by sorry

end NUMINAMATH_CALUDE_product_closest_to_1200_l2218_221872


namespace NUMINAMATH_CALUDE_trig_inequality_l2218_221835

theorem trig_inequality : ∃ (a b c : ℝ), 
  a = Real.cos 1 ∧ 
  b = Real.sin 1 ∧ 
  c = Real.tan 1 ∧ 
  a < b ∧ b < c :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l2218_221835


namespace NUMINAMATH_CALUDE_chi_square_test_win_probability_not_C_given_not_win_l2218_221889

-- Define the data from the problem
def flavor1_C : ℕ := 20
def flavor1_nonC : ℕ := 75
def flavor2_C : ℕ := 10
def flavor2_nonC : ℕ := 45
def total_samples : ℕ := 150

-- Define the chi-square test statistic function
def chi_square (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the proportions of card types
def prob_A : ℚ := 2 / 5
def prob_B : ℚ := 2 / 5
def prob_C : ℚ := 1 / 5

-- Theorem statements
theorem chi_square_test :
  chi_square flavor1_C flavor1_nonC flavor2_C flavor2_nonC total_samples < 6635 / 1000 :=
sorry

theorem win_probability :
  (3 * prob_A * prob_B * prob_C : ℚ) = 24 / 125 :=
sorry

theorem not_C_given_not_win :
  ((1 - prob_C)^3 : ℚ) / (1 - 3 * prob_A * prob_B * prob_C) = 64 / 101 :=
sorry

end NUMINAMATH_CALUDE_chi_square_test_win_probability_not_C_given_not_win_l2218_221889


namespace NUMINAMATH_CALUDE_max_xy_value_l2218_221846

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3*x*y = 2015) : 
  ∀ a b : ℝ, a^2 + b^2 + 3*a*b = 2015 → x*y ≤ 403 ∧ ∃ c d : ℝ, c^2 + d^2 + 3*c*d = 2015 ∧ c*d = 403 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l2218_221846


namespace NUMINAMATH_CALUDE_analysis_method_sufficient_conditions_l2218_221897

/-- The analysis method in mathematical proofs -/
structure AnalysisMethod where
  /-- The method starts from the conclusion to be proved -/
  starts_from_conclusion : Bool
  /-- The method progressively searches for conditions -/
  progressive_search : Bool
  /-- The type of conditions the method searches for -/
  condition_type : Type

/-- Definition of sufficient conditions -/
def SufficientCondition : Type := Unit

/-- Theorem: The analysis method searches for sufficient conditions -/
theorem analysis_method_sufficient_conditions (am : AnalysisMethod) :
  am.starts_from_conclusion ∧ am.progressive_search →
  am.condition_type = SufficientCondition := by
  sorry

end NUMINAMATH_CALUDE_analysis_method_sufficient_conditions_l2218_221897


namespace NUMINAMATH_CALUDE_hexagon_same_length_probability_l2218_221878

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of elements (sides and diagonals) in a regular hexagon -/
def total_elements : ℕ := num_sides + num_diagonals

/-- The probability of selecting two segments of the same length from a regular hexagon -/
def prob_same_length : ℚ := 17/35

theorem hexagon_same_length_probability :
  (num_sides * (num_sides - 1) + num_diagonals * (num_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length := by
  sorry

end NUMINAMATH_CALUDE_hexagon_same_length_probability_l2218_221878


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2218_221827

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 140 → 
  b = 210 → 
  c^2 = a^2 + b^2 → 
  c = 70 * Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2218_221827


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l2218_221891

theorem max_value_quadratic_function :
  let f : ℝ → ℝ := fun x ↦ -x^2 + 2*x + 1
  ∃ (m : ℝ), m = 2 ∧ ∀ x, f x ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l2218_221891


namespace NUMINAMATH_CALUDE_brendas_blisters_l2218_221883

/-- The number of blisters Brenda has on each arm -/
def blisters_per_arm : ℕ := 60

/-- The number of blisters Brenda has on the rest of her body -/
def blisters_on_body : ℕ := 80

/-- The number of arms Brenda has -/
def number_of_arms : ℕ := 2

/-- The total number of blisters Brenda has -/
def total_blisters : ℕ := blisters_per_arm * number_of_arms + blisters_on_body

theorem brendas_blisters : total_blisters = 200 := by
  sorry

end NUMINAMATH_CALUDE_brendas_blisters_l2218_221883


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2218_221877

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2218_221877


namespace NUMINAMATH_CALUDE_modulus_of_z_squared_l2218_221886

theorem modulus_of_z_squared (z : ℂ) (h : z^2 = 3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_squared_l2218_221886


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2218_221851

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem: Given the specified conditions, the man's speed against the current is 9.4 km/hr -/
theorem mans_speed_against_current :
  speedAgainstCurrent 15 2.8 = 9.4 := by
  sorry

#eval speedAgainstCurrent 15 2.8

end NUMINAMATH_CALUDE_mans_speed_against_current_l2218_221851


namespace NUMINAMATH_CALUDE_circle_point_range_l2218_221847

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (0, a)

-- Define the condition for point M
def condition_M (M : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := M
  circle_C x y ∧ (x^2 + (y - a)^2 = 2 * (x^2 + y^2))

-- Theorem statement
theorem circle_point_range (a : ℝ) :
  a > 0 →
  (∃ M : ℝ × ℝ, condition_M M a) →
  Real.sqrt 3 ≤ a ∧ a ≤ 4 + Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l2218_221847


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2218_221801

def n : ℕ := 245700

theorem sum_of_distinct_prime_factors :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id : ℕ)
    = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l2218_221801


namespace NUMINAMATH_CALUDE_plane_equation_through_point_perpendicular_to_vector_l2218_221868

/-- A plane passing through a point and perpendicular to a non-zero vector -/
theorem plane_equation_through_point_perpendicular_to_vector
  (x₀ y₀ z₀ : ℝ) (a b c : ℝ) (h : (a, b, c) ≠ (0, 0, 0)) :
  ∀ x y z : ℝ,
  (a * (x - x₀) + b * (y - y₀) + c * (z - z₀) = 0) ↔
  ((x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (x₀, y₀, z₀) + t • (a, b, c)}ᶜ) :=
by sorry


end NUMINAMATH_CALUDE_plane_equation_through_point_perpendicular_to_vector_l2218_221868


namespace NUMINAMATH_CALUDE_triangle_problem_l2218_221817

/-- Given an acute triangle ABC with collinear vectors m and n, prove B = π/6 and a + c = 2 + √3 -/
theorem triangle_problem (A B C : ℝ) (a b c : ℝ) : 
  0 < B → B < π/2 →  -- B is acute
  (2 * Real.sin (A + C)) * (2 * Real.cos (B/2)^2 - 1) = Real.sqrt 3 * Real.cos (2*B) →  -- m and n are collinear
  b = 1 →  -- given condition
  a * c * Real.sin B / 2 = Real.sqrt 3 / 2 →  -- area condition
  (B = π/6 ∧ a + c = 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2218_221817


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_l2218_221852

/-- The smallest positive integer x such that 2520x is a perfect cube -/
def smallest_x : ℕ := 3675

/-- 2520 as a natural number -/
def given_number : ℕ := 2520

theorem smallest_x_is_correct :
  (∀ y : ℕ, y > 0 ∧ y < smallest_x → ¬∃ M : ℕ, given_number * y = M^3) ∧
  ∃ M : ℕ, given_number * smallest_x = M^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_is_correct_l2218_221852


namespace NUMINAMATH_CALUDE_range_of_f_l2218_221885

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x + 2)

-- State the theorem
theorem range_of_f :
  Set.range f = {y : ℝ | y < 21 ∨ y > 21} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2218_221885


namespace NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l2218_221880

/-- The function g(x) as defined in the problem -/
def g (x r : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 5 + r

/-- Theorem stating that g(-1) = 0 when r = 14 -/
theorem g_equals_zero_at_negative_one (r : ℝ) : g (-1) r = 0 ↔ r = 14 := by sorry

end NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l2218_221880


namespace NUMINAMATH_CALUDE_football_team_progress_l2218_221888

/-- The progress of a football team after a series of gains and losses -/
theorem football_team_progress 
  (L1 G1 L2 G2 G3 : ℤ) 
  (hL1 : L1 = 17)
  (hG1 : G1 = 35)
  (hL2 : L2 = 22)
  (hG2 : G2 = 8) :
  (G1 + G2 - (L1 + L2)) + G3 = 4 + G3 :=
by sorry

end NUMINAMATH_CALUDE_football_team_progress_l2218_221888


namespace NUMINAMATH_CALUDE_function_inequality_l2218_221808

/-- Given a function f(x) = x - 1 - ln x, if f(x) ≥ kx - 2 for all x > 0, 
    then k ≤ 1 - 1/e² -/
theorem function_inequality (k : ℝ) : 
  (∀ x > 0, x - 1 - Real.log x ≥ k * x - 2) → k ≤ 1 - 1 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2218_221808


namespace NUMINAMATH_CALUDE_two_face_cards_probability_l2218_221806

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of face cards in a standard deck
def face_cards : ℕ := 12

-- Define the probability of selecting two face cards
def prob_two_face_cards : ℚ := 22 / 442

-- Theorem statement
theorem two_face_cards_probability :
  (face_cards / total_cards) * ((face_cards - 1) / (total_cards - 1)) = prob_two_face_cards := by
  sorry

end NUMINAMATH_CALUDE_two_face_cards_probability_l2218_221806


namespace NUMINAMATH_CALUDE_existence_of_a_value_of_a_l2218_221811

-- Define the sets A, B, and C as functions of real numbers
def A (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + 4*a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem 1
theorem existence_of_a : ∃ a : ℝ, A a = B ∧ a = 1/2 := by sorry

-- Theorem 2
theorem value_of_a (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -1 := by sorry

end NUMINAMATH_CALUDE_existence_of_a_value_of_a_l2218_221811
