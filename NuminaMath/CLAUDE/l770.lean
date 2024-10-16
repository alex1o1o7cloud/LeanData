import Mathlib

namespace NUMINAMATH_CALUDE_total_lifting_capacity_is_250_l770_77090

/-- Calculates the new combined total lifting capacity given initial weights and increases -/
def new_total_lifting_capacity (initial_clean_and_jerk : ℝ) (initial_snatch : ℝ) : ℝ :=
  (2 * initial_clean_and_jerk) + (initial_snatch * 1.8)

/-- Proves that the new combined total lifting capacity is 250 kg -/
theorem total_lifting_capacity_is_250 :
  new_total_lifting_capacity 80 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_total_lifting_capacity_is_250_l770_77090


namespace NUMINAMATH_CALUDE_range_of_a_l770_77084

def A (a : ℝ) : Set ℝ := {x | (a * x - 1) / (x - a) < 0}

theorem range_of_a : ∀ a : ℝ, 
  (2 ∈ A a ∧ 3 ∉ A a) ↔ 
  (a ∈ Set.Icc (1/3 : ℝ) (1/2 : ℝ) ∪ Set.Ioc (2 : ℝ) (3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l770_77084


namespace NUMINAMATH_CALUDE_sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2_l770_77068

theorem sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2 :
  Real.sqrt 50 + 2 / (Real.sqrt 2 + 1) - 4 * Real.sqrt (1/2) + 2 * (Real.sqrt 2 - 1)^0 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_fraction_minus_sqrt_half_plus_power_eq_5_sqrt_2_l770_77068


namespace NUMINAMATH_CALUDE_basketball_season_games_l770_77056

/-- The total number of games played by a basketball team in a season -/
def total_games : ℕ := 93

/-- The number of games in the first segment -/
def first_segment : ℕ := 40

/-- The number of games in the second segment -/
def second_segment : ℕ := 30

/-- The win rate for the first segment -/
def first_rate : ℚ := 1/2

/-- The win rate for the second segment -/
def second_rate : ℚ := 3/5

/-- The win rate for the remaining games -/
def remaining_rate : ℚ := 17/20

/-- The overall win rate for the season -/
def overall_rate : ℚ := 31/50

theorem basketball_season_games :
  let remaining_games := total_games - first_segment - second_segment
  let total_wins := (first_rate * first_segment) + (second_rate * second_segment) + (remaining_rate * remaining_games)
  total_wins = overall_rate * total_games := by sorry

#eval total_games

end NUMINAMATH_CALUDE_basketball_season_games_l770_77056


namespace NUMINAMATH_CALUDE_farmer_additional_earnings_l770_77039

/-- Represents the farmer's market transactions and wheelbarrow sale --/
def farmer_earnings (duck_price chicken_price : ℕ) (ducks_sold chickens_sold : ℕ) : ℕ :=
  let total_earnings := duck_price * ducks_sold + chicken_price * chickens_sold
  let wheelbarrow_cost := total_earnings / 2
  let wheelbarrow_sale_price := wheelbarrow_cost * 2
  wheelbarrow_sale_price - wheelbarrow_cost

/-- Proves that the farmer's additional earnings from selling the wheelbarrow is $30 --/
theorem farmer_additional_earnings :
  farmer_earnings 10 8 2 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_farmer_additional_earnings_l770_77039


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l770_77044

theorem arithmetic_simplification : (4 + 6 + 2) / 3 - 2 / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l770_77044


namespace NUMINAMATH_CALUDE_partition_rational_points_l770_77015

/-- Rational points in the plane -/
def RationalPoints : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | true}

/-- The theorem statement -/
theorem partition_rational_points :
  ∃ (A B : Set (ℚ × ℚ)),
    A ∩ B = ∅ ∧
    A ∪ B = RationalPoints ∧
    (∀ t : ℚ, Set.Finite {y : ℚ | (t, y) ∈ A}) ∧
    (∀ t : ℚ, Set.Finite {x : ℚ | (x, t) ∈ B}) :=
sorry

end NUMINAMATH_CALUDE_partition_rational_points_l770_77015


namespace NUMINAMATH_CALUDE_digit_150_is_1_l770_77076

/-- The decimal expansion of 5/31 -/
def decimal_expansion : ℚ := 5 / 31

/-- The length of the repeating part in the decimal expansion of 5/31 -/
def repetition_length : ℕ := 15

/-- The position we're interested in -/
def target_position : ℕ := 150

/-- The function that returns the nth digit after the decimal point in the decimal expansion of 5/31 -/
noncomputable def nth_digit (n : ℕ) : ℕ := 
  sorry

theorem digit_150_is_1 : nth_digit target_position = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_is_1_l770_77076


namespace NUMINAMATH_CALUDE_card_area_problem_l770_77057

theorem card_area_problem (length width : ℝ) 
  (h1 : length = 4 ∧ width = 6)
  (h2 : (length - 1) * width = 18 ∨ length * (width - 1) = 18) :
  (if (length - 1) * width = 18 
   then length * (width - 1) 
   else (length - 1) * width) = 20 := by
  sorry

end NUMINAMATH_CALUDE_card_area_problem_l770_77057


namespace NUMINAMATH_CALUDE_specific_isosceles_triangle_area_l770_77013

/-- Represents an isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ
  leg_difference : ℝ

/-- Calculates the area of the isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles triangle -/
theorem specific_isosceles_triangle_area :
  let t : IsoscelesTriangle := {
    altitude := 10,
    perimeter := 40,
    leg_difference := 2
  }
  area t = 81.2 := by sorry

end NUMINAMATH_CALUDE_specific_isosceles_triangle_area_l770_77013


namespace NUMINAMATH_CALUDE_bowlfuls_in_box_l770_77099

/-- Represents the number of clusters of oats in each spoonful -/
def clusters_per_spoonful : ℕ := 4

/-- Represents the number of spoonfuls in each bowl of cereal -/
def spoonfuls_per_bowl : ℕ := 25

/-- Represents the total number of clusters of oats in each box -/
def clusters_per_box : ℕ := 500

/-- Calculates the number of bowlfuls of cereal in each box -/
def bowlfuls_per_box : ℕ :=
  clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl)

/-- Theorem stating that the number of bowlfuls of cereal in each box is 5 -/
theorem bowlfuls_in_box : bowlfuls_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_bowlfuls_in_box_l770_77099


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_l770_77009

/-- Represents the array of odd numbers -/
def oddNumberArray (row : ℕ) (position : ℕ) : ℕ :=
  2 * (row * (row - 1) / 2 + position) - 1

/-- The theorem to prove -/
theorem fifth_number_21st_row :
  oddNumberArray 21 5 = 809 :=
sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_l770_77009


namespace NUMINAMATH_CALUDE_homework_time_is_48_minutes_l770_77081

def math_problems : ℕ := 15
def social_studies_problems : ℕ := 6
def science_problems : ℕ := 10

def math_time_per_problem : ℚ := 2
def social_studies_time_per_problem : ℚ := 1/2
def science_time_per_problem : ℚ := 3/2

def total_homework_time : ℚ :=
  math_problems * math_time_per_problem +
  social_studies_problems * social_studies_time_per_problem +
  science_problems * science_time_per_problem

theorem homework_time_is_48_minutes :
  total_homework_time = 48 := by sorry

end NUMINAMATH_CALUDE_homework_time_is_48_minutes_l770_77081


namespace NUMINAMATH_CALUDE_problem_statement_l770_77038

theorem problem_statement (x : ℝ) (h : x + 1/x = 2) : x^5 - 5*x^3 + 6*x = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l770_77038


namespace NUMINAMATH_CALUDE_ellipse_area_irrational_l770_77003

/-- The area of an ellipse with rational semi-axes is irrational -/
theorem ellipse_area_irrational (a b : ℚ) (h1 : a > 0) (h2 : b > 0) : 
  Irrational (Real.pi * (a * b)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_area_irrational_l770_77003


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_when_intersection_equals_given_set_l770_77026

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x : ℝ | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem m_value_when_intersection_equals_given_set :
  (∃ m : ℝ, A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) → 
  (∃ m : ℝ, A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4} ∧ m = 8) := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_when_intersection_equals_given_set_l770_77026


namespace NUMINAMATH_CALUDE_total_cost_theorem_l770_77030

/-- Calculates the total cost of purchasing two laptops with accessories --/
def total_cost (first_laptop_price : ℝ) (second_laptop_multiplier : ℝ) 
  (second_laptop_discount : ℝ) (hard_drive_price : ℝ) (mouse_price : ℝ) 
  (software_subscription_price : ℝ) (insurance_rate : ℝ) : ℝ :=
  let first_laptop_total := first_laptop_price + hard_drive_price + mouse_price + 
    software_subscription_price + (insurance_rate * first_laptop_price)
  let second_laptop_price := first_laptop_price * second_laptop_multiplier
  let second_laptop_discounted := second_laptop_price * (1 - second_laptop_discount)
  let second_laptop_total := second_laptop_discounted + hard_drive_price + mouse_price + 
    (2 * software_subscription_price) + (insurance_rate * second_laptop_discounted)
  first_laptop_total + second_laptop_total

/-- Theorem stating the total cost of purchasing both laptops with accessories --/
theorem total_cost_theorem : 
  total_cost 500 3 0.15 80 20 120 0.1 = 2512.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l770_77030


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l770_77072

theorem six_digit_multiple_of_nine (n : ℕ) (h1 : n ≥ 734601 ∧ n ≤ 734691) 
  (h2 : n % 9 = 0) : 
  ∃ d : ℕ, (d = 6 ∨ d = 9) ∧ n = 734601 + d * 100 :=
sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l770_77072


namespace NUMINAMATH_CALUDE_roys_height_l770_77050

/-- Given the heights of Sara, Joe, and Roy, prove Roy's height -/
theorem roys_height
  (sara_height : ℕ)
  (sara_joe_diff : ℕ)
  (joe_roy_diff : ℕ)
  (h_sara_height : sara_height = 45)
  (h_sara_joe : sara_height = sara_joe_diff + joe_height)
  (h_joe_roy : joe_height = joe_roy_diff + roy_height)
  : roy_height = 36 := by
  sorry


end NUMINAMATH_CALUDE_roys_height_l770_77050


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l770_77063

theorem imaginary_part_of_complex_fraction :
  Complex.im ((4 - 5 * Complex.I) / Complex.I) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l770_77063


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l770_77032

def p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0

theorem necessary_but_not_sufficient_condition :
  (∃ (a b : ℝ), (a = -1 ∧ b = 2 ∨ a = -2 ∧ b = 2) ∧
    (∀ x, p x → a < x ∧ x < b) ∧
    (∃ y, a < y ∧ y < b ∧ ¬(p y))) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l770_77032


namespace NUMINAMATH_CALUDE_ratio_proportion_problem_l770_77073

theorem ratio_proportion_problem (x : ℝ) :
  (2975.75 / 7873.125 = 12594.5 / x) → x = 33333.75 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proportion_problem_l770_77073


namespace NUMINAMATH_CALUDE_extremum_values_l770_77086

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_values (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 1| < ε → f a b x ≤ f a b 1) ∧
  (∃ (δ : ℝ), ∀ (x : ℝ), |x - 1| < δ → f a b x ≥ f a b 1) ∧
  f a b 1 = 10 →
  a = 4 ∧ b = -11 := by sorry

end NUMINAMATH_CALUDE_extremum_values_l770_77086


namespace NUMINAMATH_CALUDE_x_remaining_time_l770_77046

-- Define the work rates and time worked
def x_rate : ℚ := 1 / 20
def y_rate : ℚ := 1 / 15
def y_time_worked : ℚ := 9

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem statement
theorem x_remaining_time (x_rate : ℚ) (y_rate : ℚ) (y_time_worked : ℚ) (total_work : ℚ) :
  x_rate = 1 / 20 →
  y_rate = 1 / 15 →
  y_time_worked = 9 →
  total_work = 1 →
  (total_work - y_rate * y_time_worked) / x_rate = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_x_remaining_time_l770_77046


namespace NUMINAMATH_CALUDE_unique_plane_through_and_parallel_l770_77002

-- Define the concept of skew lines
def SkewLines (l₁ l₂ : Set (Point)) : Prop := sorry

-- Define a plane passing through a line and parallel to another line
def PlaneThroughAndParallel (π : Set (Point)) (l₁ l₂ : Set (Point)) : Prop := sorry

theorem unique_plane_through_and_parallel 
  (l₁ l₂ : Set (Point)) 
  (h : SkewLines l₁ l₂) : 
  ∃! π, PlaneThroughAndParallel π l₁ l₂ := by sorry

end NUMINAMATH_CALUDE_unique_plane_through_and_parallel_l770_77002


namespace NUMINAMATH_CALUDE_optimal_ships_l770_77069

/-- Revenue function -/
def R (x : ℕ) : ℚ := 3700 * x + 45 * x^2 - 10 * x^3

/-- Cost function -/
def C (x : ℕ) : ℚ := 460 * x + 5000

/-- Profit function -/
def P (x : ℕ) : ℚ := R x - C x

/-- The maximum number of ships that can be built annually -/
def max_capacity : ℕ := 20

/-- Theorem: The number of ships that maximizes annual profit is 12 -/
theorem optimal_ships :
  ∃ (x : ℕ), x ≤ max_capacity ∧ x > 0 ∧
  ∀ (y : ℕ), y ≤ max_capacity ∧ y > 0 → P x ≥ P y ∧
  x = 12 :=
sorry

end NUMINAMATH_CALUDE_optimal_ships_l770_77069


namespace NUMINAMATH_CALUDE_star_six_three_l770_77022

-- Define the * operation
def star (a b : ℤ) : ℤ := 4*a + 5*b - 2*a*b

-- Theorem statement
theorem star_six_three : star 6 3 = 3 := by sorry

end NUMINAMATH_CALUDE_star_six_three_l770_77022


namespace NUMINAMATH_CALUDE_fixed_point_line_l770_77092

/-- Given a line that always passes through a fixed point, prove that the line
    passing through this fixed point and the origin has the equation y = 2x -/
theorem fixed_point_line (a : ℝ) : 
  (∃ (x₀ y₀ : ℝ), ∀ (x y : ℝ), a * x + y + a + 2 = 0 → x = x₀ ∧ y = y₀) → 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    (a * x₀ + y₀ + a + 2 = 0 ∧ 
     y - y₀ = m * (x - x₀) ∧ 
     0 - y₀ = m * (0 - x₀)) → 
    y = 2 * x :=
sorry

end NUMINAMATH_CALUDE_fixed_point_line_l770_77092


namespace NUMINAMATH_CALUDE_cube_division_l770_77034

theorem cube_division (n : ℕ) (h1 : n ≥ 6) (h2 : Even n) :
  ∃ (m : ℕ), m^3 = (3 * n * (n - 2)) / 4 + 2 :=
sorry

end NUMINAMATH_CALUDE_cube_division_l770_77034


namespace NUMINAMATH_CALUDE_die_roll_probability_l770_77052

def standard_die_roll := Fin 6

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_five (roll : standard_die_roll) : Prop := roll.val + 1 = 5

def probability_odd_roll : ℚ := 1/2

def probability_not_five : ℚ := 5/6

def num_rolls : ℕ := 8

theorem die_roll_probability :
  (probability_odd_roll ^ num_rolls) * (1 - probability_not_five ^ num_rolls) = 1288991/429981696 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l770_77052


namespace NUMINAMATH_CALUDE_parabola_coefficient_b_l770_77082

/-- Given a parabola y = ax^2 + bx + c with vertex at (q, -q) and y-intercept at (0, q),
    where q ≠ 0, the coefficient b is equal to -4. -/
theorem parabola_coefficient_b (a b c q : ℝ) (hq : q ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - q)^2 - q) →
  (a * 0^2 + b * 0 + c = q) →
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_b_l770_77082


namespace NUMINAMATH_CALUDE_pants_count_l770_77033

/-- Represents the number of each type of clothing item in a dresser -/
structure DresserContents where
  pants : ℕ
  shorts : ℕ
  shirts : ℕ

/-- The ratio of pants to shorts to shirts in the dresser -/
def clothingRatio : ℕ × ℕ × ℕ := (7, 7, 10)

/-- The number of shirts in the dresser -/
def shirtCount : ℕ := 20

/-- Checks if the given DresserContents satisfies the ratio condition -/
def satisfiesRatio (contents : DresserContents) : Prop :=
  contents.pants * clothingRatio.2.2 = contents.shirts * clothingRatio.1 ∧
  contents.shorts * clothingRatio.2.2 = contents.shirts * clothingRatio.2.1

theorem pants_count (contents : DresserContents) 
  (h_ratio : satisfiesRatio contents) 
  (h_shirts : contents.shirts = shirtCount) : 
  contents.pants = 14 := by
  sorry

end NUMINAMATH_CALUDE_pants_count_l770_77033


namespace NUMINAMATH_CALUDE_golf_ball_goal_l770_77058

theorem golf_ball_goal (goal : ℕ) (saturday : ℕ) (sunday : ℕ) : 
  goal = 48 → saturday = 16 → sunday = 18 → 
  goal - (saturday + sunday) = 14 :=
by sorry

end NUMINAMATH_CALUDE_golf_ball_goal_l770_77058


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l770_77091

/-- A quadratic function with specific properties -/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * (x + 2)^2 + b

/-- The chord length intercepted by the x-axis is 2√3 -/
def chord_length (a b : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ |x₁ - x₂| = 2 * Real.sqrt 3

/-- The function passes through (0, 1) -/
def passes_through_origin (a b : ℝ) : Prop := f a b 0 = 1

/-- The function passes through (-2+√3, 0) -/
def passes_through_intercept (a b : ℝ) : Prop := f a b (-2 + Real.sqrt 3) = 0

theorem quadratic_function_properties :
  ∀ a b : ℝ, chord_length a b → passes_through_origin a b → passes_through_intercept a b →
  (∀ x, f a b x = (x + 2)^2 - 3) ∧
  (∀ k, k < 13/4 → ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a b ((1/2 : ℝ)^x) > k) :=
sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l770_77091


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l770_77093

theorem rectangle_perimeter (L B : ℝ) 
  (h1 : L - B = 23)
  (h2 : L * B = 2520) :
  2 * (L + B) = 206 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l770_77093


namespace NUMINAMATH_CALUDE_integer_solutions_xy_eq_x_plus_y_l770_77097

theorem integer_solutions_xy_eq_x_plus_y :
  ∀ x y : ℤ, x * y = x + y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_xy_eq_x_plus_y_l770_77097


namespace NUMINAMATH_CALUDE_gravel_path_cost_l770_77049

/-- Represents the dimensions of a rectangular plot -/
structure PlotDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular path inside a plot -/
def pathArea (plot : PlotDimensions) (pathWidth : ℝ) : ℝ :=
  plot.length * plot.width - (plot.length - 2 * pathWidth) * (plot.width - 2 * pathWidth)

/-- Calculates the cost of gravelling a path -/
def gravellingCost (area : ℝ) (costPerSqMetre : ℝ) : ℝ :=
  area * costPerSqMetre

/-- Theorem: The cost of gravelling the path is 680 Rupees -/
theorem gravel_path_cost :
  let plot := PlotDimensions.mk 110 65
  let pathWidth := 2.5
  let costPerSqMetre := 0.8 -- 80 paise = 0.8 Rupees
  gravellingCost (pathArea plot pathWidth) costPerSqMetre = 680 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l770_77049


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l770_77012

/-- Proves that the number of meters of cloth sold is 85, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 8500 →
  profit_per_meter = 15 →
  cost_price_per_meter = 85 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
sorry

end NUMINAMATH_CALUDE_cloth_sale_meters_l770_77012


namespace NUMINAMATH_CALUDE_blender_price_difference_l770_77087

def in_store_price : ℚ := 75.99
def tv_payment : ℚ := 17.99
def shipping_fee : ℚ := 6.50
def handling_charge : ℚ := 2.50

theorem blender_price_difference :
  (4 * tv_payment + shipping_fee + handling_charge - in_store_price) * 100 = 497 := by
  sorry

end NUMINAMATH_CALUDE_blender_price_difference_l770_77087


namespace NUMINAMATH_CALUDE_smallest_valid_tv_order_l770_77098

def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d ∈ [1, 2, 3, 4, 5, 6]

theorem smallest_valid_tv_order : ∃! n : ℕ, n > 0 ∧ valid_digits (1994 * n) ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬valid_digits (1994 * m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_tv_order_l770_77098


namespace NUMINAMATH_CALUDE_power_of_three_equation_l770_77017

theorem power_of_three_equation (x : ℝ) : 
  4 * (3 : ℝ)^x = 243 → (x + 1) * (x - 1) = 16.696 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equation_l770_77017


namespace NUMINAMATH_CALUDE_determinant_problem_l770_77074

theorem determinant_problem (a b c d : ℝ) : 
  let M₁ : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let M₂ : Matrix (Fin 2) (Fin 2) ℝ := !![a+2*c, b+2*d; 3*c, 3*d]
  Matrix.det M₁ = -7 → Matrix.det M₂ = -21 := by
  sorry

end NUMINAMATH_CALUDE_determinant_problem_l770_77074


namespace NUMINAMATH_CALUDE_complex_equation_solution_l770_77064

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l770_77064


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l770_77020

theorem cubic_root_sum_product (p q r : ℂ) : 
  (5 * p^3 - 10 * p^2 + 17 * p - 7 = 0) →
  (5 * q^3 - 10 * q^2 + 17 * q - 7 = 0) →
  (5 * r^3 - 10 * r^2 + 17 * r - 7 = 0) →
  p * q + q * r + r * p = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l770_77020


namespace NUMINAMATH_CALUDE_AX_length_l770_77023

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
def diameter : ℝ := 1

-- Define the points on the circle
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define point X on diameter AD
def X : Point := sorry

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the angle function
def angle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem AX_length (h1 : distance A D = diameter)
                  (h2 : distance B X = distance C X)
                  (h3 : 3 * angle B A C = angle B X C)
                  (h4 : angle B X C = 30 * π / 180) :
  distance A X = Real.cos (10 * π / 180) * Real.sin (20 * π / 180) * (1 / Real.sin (15 * π / 180)) :=
sorry

end NUMINAMATH_CALUDE_AX_length_l770_77023


namespace NUMINAMATH_CALUDE_intersection_x_value_l770_77021

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m1 b1 a2 b2 c2 : ℚ) : ℚ :=
  (c2 - b2 + b1) / (m1 + a2)

/-- Theorem: The x-coordinate of the intersection point of y = 4x - 29 and 3x + y = 105 is 134/7 -/
theorem intersection_x_value :
  intersection_x 4 (-29) 3 1 105 = 134 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_value_l770_77021


namespace NUMINAMATH_CALUDE_coefficient_of_x3y2z5_in_expansion_l770_77028

/-- The coefficient of x^3y^2z^5 in the expansion of (2x+y+z)^10 -/
def coefficient : ℕ := 20160

/-- The exponent of the trinomial expression -/
def exponent : ℕ := 10

/-- Theorem stating that the coefficient of x^3y^2z^5 in (2x+y+z)^10 is 20160 -/
theorem coefficient_of_x3y2z5_in_expansion : 
  coefficient = (2^3 : ℕ) * Nat.choose exponent 3 * Nat.choose (exponent - 3) 2 * Nat.choose ((exponent - 3) - 2) 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x3y2z5_in_expansion_l770_77028


namespace NUMINAMATH_CALUDE_min_treasures_is_15_l770_77029

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Predicate that checks if a given number of treasures satisfies all conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ total_trees ∧
  (n ≠ 15 ∨ signs_15 = total_trees - n) ∧
  (n ≠ 8 ∨ signs_8 = total_trees - n) ∧
  (n ≠ 4 ∨ signs_4 = total_trees - n) ∧
  (n ≠ 3 ∨ signs_3 = total_trees - n)

/-- Theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasures_is_15 :
  ∃ (n : ℕ), n = 15 ∧ satisfies_conditions n ∧ ∀ (m : ℕ), m < n → ¬satisfies_conditions m :=
by sorry

end NUMINAMATH_CALUDE_min_treasures_is_15_l770_77029


namespace NUMINAMATH_CALUDE_kate_stickers_l770_77077

theorem kate_stickers (kate_stickers jenna_stickers : ℕ) : 
  (kate_stickers : ℚ) / jenna_stickers = 7 / 4 →
  jenna_stickers = 12 →
  kate_stickers = 21 := by
sorry

end NUMINAMATH_CALUDE_kate_stickers_l770_77077


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_sixteen_l770_77024

theorem sum_of_solutions_equals_sixteen :
  let f : ℝ → ℝ := λ x => Real.sqrt x + Real.sqrt (9 / x) + Real.sqrt (x + 9 / x)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 9 ∧ f x₂ = 9 ∧ x₁ + x₂ = 16 ∧
  ∀ (x : ℝ), f x = 9 → x = x₁ ∨ x = x₂ :=
by sorry


end NUMINAMATH_CALUDE_sum_of_solutions_equals_sixteen_l770_77024


namespace NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l770_77019

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
def B : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}

-- Theorem 1: When m = 2, A ∩ B = [1, 2]
theorem intersection_when_m_is_two : 
  A 2 ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: A ⊆ (A ∩ B) if and only if -2 ≤ m ≤ 1/2
theorem subset_condition (m : ℝ) : 
  A m ⊆ (A m ∩ B) ↔ -2 ≤ m ∧ m ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l770_77019


namespace NUMINAMATH_CALUDE_largest_five_digit_multiple_of_3_and_4_l770_77007

theorem largest_five_digit_multiple_of_3_and_4 : 
  ∀ n : ℕ, n ≤ 99999 ∧ n ≥ 10000 ∧ 3 ∣ n ∧ 4 ∣ n → n ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_multiple_of_3_and_4_l770_77007


namespace NUMINAMATH_CALUDE_solve_for_y_l770_77089

theorem solve_for_y (x y : ℤ) (h1 : x^2 + x + 4 = y - 4) (h2 : x = -7) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l770_77089


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l770_77031

theorem tax_free_items_cost 
  (total_paid : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_paid = 40) 
  (h2 : sales_tax = 1.28) 
  (h3 : tax_rate = 0.08) : 
  total_paid - (sales_tax / tax_rate + sales_tax) = 22.72 := by
sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l770_77031


namespace NUMINAMATH_CALUDE_bagel_count_is_three_l770_77067

/-- Represents the number of items bought at each price point -/
structure PurchaseCount where
  sixtyCount : ℕ
  eightyCount : ℕ
  hundredCount : ℕ

/-- Calculates the total cost in cents for a given purchase count -/
def totalCost (p : PurchaseCount) : ℕ :=
  60 * p.sixtyCount + 80 * p.eightyCount + 100 * p.hundredCount

/-- Theorem stating that under the given conditions, the number of 80-cent items is 3 -/
theorem bagel_count_is_three :
  ∃ (p : PurchaseCount),
    p.sixtyCount + p.eightyCount + p.hundredCount = 5 ∧
    totalCost p = 400 ∧
    p.eightyCount = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bagel_count_is_three_l770_77067


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l770_77096

/-- Represents the number of rooms that can be painted with the available paint -/
def initialRooms : ℕ := 50

/-- Represents the number of paint cans misplaced -/
def misplacedCans : ℕ := 5

/-- Represents the number of rooms that can be painted after misplacing some cans -/
def remainingRooms : ℕ := 37

/-- Calculates the number of cans used to paint the remaining rooms -/
def cansUsed : ℕ := 15

theorem paint_cans_theorem : 
  ∀ (initial : ℕ) (misplaced : ℕ) (remaining : ℕ),
  initial = initialRooms → 
  misplaced = misplacedCans → 
  remaining = remainingRooms → 
  cansUsed = 15 :=
by sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l770_77096


namespace NUMINAMATH_CALUDE_library_books_remaining_l770_77043

theorem library_books_remaining (initial_books : ℕ) 
  (day1_borrowers : ℕ) (books_per_borrower : ℕ) (day2_borrowed : ℕ) : 
  initial_books = 100 →
  day1_borrowers = 5 →
  books_per_borrower = 2 →
  day2_borrowed = 20 →
  initial_books - (day1_borrowers * books_per_borrower + day2_borrowed) = 70 :=
by sorry

end NUMINAMATH_CALUDE_library_books_remaining_l770_77043


namespace NUMINAMATH_CALUDE_hour_hand_rotation_l770_77025

/-- Represents the number of degrees in a complete rotation. -/
def complete_rotation : ℕ := 360

/-- Represents the number of hours in a day. -/
def hours_per_day : ℕ := 24

/-- Represents the number of complete rotations the hour hand makes. -/
def rotations : ℕ := 12

/-- Represents the number of days in which the rotations occur. -/
def days : ℕ := 6

/-- Calculates the number of degrees the hour hand rotates per hour. -/
def degrees_per_hour : ℚ :=
  (rotations * complete_rotation) / (days * hours_per_day)

theorem hour_hand_rotation :
  degrees_per_hour = 30 := by sorry

end NUMINAMATH_CALUDE_hour_hand_rotation_l770_77025


namespace NUMINAMATH_CALUDE_M_remainder_mod_45_l770_77078

/-- The number of digits in M -/
def num_digits : ℕ := 95

/-- The last integer in the sequence forming M -/
def last_int : ℕ := 50

/-- M is the number formed by concatenating integers from 1 to last_int -/
def M : ℕ := sorry

theorem M_remainder_mod_45 : M % 45 = 15 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_45_l770_77078


namespace NUMINAMATH_CALUDE_parrots_per_cage_l770_77079

theorem parrots_per_cage (num_cages : ℕ) (total_birds : ℕ) : 
  num_cages = 9 →
  total_birds = 36 →
  (∃ (parrots_per_cage : ℕ), 
    parrots_per_cage * num_cages * 2 = total_birds ∧ 
    parrots_per_cage = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l770_77079


namespace NUMINAMATH_CALUDE_value_set_of_x_l770_77018

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- State the theorem
theorem value_set_of_x (x : ℝ) :
  (∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|) →
  x ≤ -1 ∨ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_value_set_of_x_l770_77018


namespace NUMINAMATH_CALUDE_star_computation_l770_77080

def star (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem star_computation :
  star 2 (star 3 (star 4 5)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_star_computation_l770_77080


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l770_77011

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + 5 * x + b > 0}) : 
  {x : ℝ | b * x^2 - 5 * x + a > 0} = Set.Ioo (-1/2) (-1/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l770_77011


namespace NUMINAMATH_CALUDE_sum_Q_mod_500_l770_77051

/-- The set of distinct remainders when 3^k is divided by 500, for 0 ≤ k < 200 -/
def Q : Finset ℕ :=
  (Finset.range 200).image (fun k => (3^k : ℕ) % 500)

/-- The sum of all elements in Q -/
def sum_Q : ℕ := Q.sum id

/-- The theorem to prove -/
theorem sum_Q_mod_500 :
  sum_Q % 500 = (Finset.range 200).sum (fun k => (3^k : ℕ) % 500) % 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_Q_mod_500_l770_77051


namespace NUMINAMATH_CALUDE_four_points_on_circle_l770_77042

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points lie on the same circle -/
def on_same_circle (A B C D : Point) : Prop :=
  ∃ (center : Point) (r : ℝ),
    (center.x - A.x)^2 + (center.y - A.y)^2 = r^2 ∧
    (center.x - B.x)^2 + (center.y - B.y)^2 = r^2 ∧
    (center.x - C.x)^2 + (center.y - C.y)^2 = r^2 ∧
    (center.x - D.x)^2 + (center.y - D.y)^2 = r^2

theorem four_points_on_circle :
  let A : Point := ⟨-1, 5⟩
  let B : Point := ⟨5, 5⟩
  let C : Point := ⟨-3, 1⟩
  let D : Point := ⟨6, -2⟩
  on_same_circle A B C D :=
by
  sorry

end NUMINAMATH_CALUDE_four_points_on_circle_l770_77042


namespace NUMINAMATH_CALUDE_seating_arrangements_l770_77027

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where a group of k people must sit together -/
def groupedArrangements (n k : ℕ) : ℕ := 
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of people to be seated -/
def totalPeople : ℕ := 10

/-- The number of people in the group that can't sit together -/
def groupSize : ℕ := 4

theorem seating_arrangements :
  totalArrangements totalPeople - groupedArrangements totalPeople groupSize = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l770_77027


namespace NUMINAMATH_CALUDE_triangle_inequality_l770_77066

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l770_77066


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_gcd_of_54_87_172_l770_77014

theorem greatest_common_divisor_of_differences : Int → Int → Int → Prop :=
  fun a b c => 
    let diff1 := b - a
    let diff2 := c - b
    let diff3 := c - a
    Nat.gcd (Nat.gcd (Int.natAbs diff1) (Int.natAbs diff2)) (Int.natAbs diff3) = 1

theorem gcd_of_54_87_172 : greatest_common_divisor_of_differences 54 87 172 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_gcd_of_54_87_172_l770_77014


namespace NUMINAMATH_CALUDE_point_M_properties_segment_MN_length_l770_77085

def M (m : ℝ) : ℝ × ℝ := (2*m + 1, m + 3)

theorem point_M_properties (m : ℝ) :
  (M m).1 > 0 ∧ (M m).2 > 0 ∧  -- M is in the first quadrant
  (M m).2 = 2 * (M m).1  -- distance to x-axis is twice distance to y-axis
  → m = 1/3 := by sorry

def N : ℝ × ℝ := (2, 1)

theorem segment_MN_length (m : ℝ) :
  (M m).2 = N.2  -- MN is parallel to x-axis
  → |N.1 - (M m).1| = 5 := by sorry

end NUMINAMATH_CALUDE_point_M_properties_segment_MN_length_l770_77085


namespace NUMINAMATH_CALUDE_inequality_solution_l770_77036

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x + 3) / ((x - 2)^2) < 0 ↔ 1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l770_77036


namespace NUMINAMATH_CALUDE_problem_statement_l770_77035

theorem problem_statement :
  (∀ n : ℕ, n > 1 → ¬(n ∣ (2^n - 1))) ∧
  (∀ n : ℕ, Nat.Prime n → (n^2 ∣ (2^n + 1)) → n = 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l770_77035


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l770_77000

/-- Given a quadratic x^2 - 40x + 121 that can be written as (x+b)^2 + c,
    prove that b + c = -299 -/
theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 40*x + 121 = (x + b)^2 + c) → b + c = -299 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l770_77000


namespace NUMINAMATH_CALUDE_graduation_ceremony_chairs_l770_77083

/-- Calculates the number of chairs needed for a graduation ceremony --/
def chairs_needed (graduates : ℕ) (parents_per_graduate : ℕ) (teachers : ℕ) : ℕ :=
  let parent_chairs := graduates * parents_per_graduate
  let graduate_and_parent_chairs := graduates + parent_chairs
  let administrator_chairs := teachers / 2
  graduate_and_parent_chairs + teachers + administrator_chairs

theorem graduation_ceremony_chairs :
  chairs_needed 50 2 20 = 180 :=
by sorry

end NUMINAMATH_CALUDE_graduation_ceremony_chairs_l770_77083


namespace NUMINAMATH_CALUDE_complex_equation_solution_l770_77070

theorem complex_equation_solution : ∃ z : ℂ, (z + 2) * (1 + Complex.I ^ 3) = 2 * Complex.I ∧ z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l770_77070


namespace NUMINAMATH_CALUDE_abs_two_over_z_minus_z_equals_two_l770_77016

/-- Given a complex number z = 1 + i, prove that |2/z - z| = 2 -/
theorem abs_two_over_z_minus_z_equals_two :
  let z : ℂ := 1 + Complex.I
  Complex.abs (2 / z - z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_over_z_minus_z_equals_two_l770_77016


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l770_77095

theorem quadratic_root_k_value (k : ℝ) : 
  ((-2 : ℝ)^2 - k * (-2) - 6 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l770_77095


namespace NUMINAMATH_CALUDE_rose_incorrect_answers_l770_77061

theorem rose_incorrect_answers 
  (total_items : ℕ) 
  (liza_percentage : ℚ) 
  (rose_additional_correct : ℕ) 
  (h1 : total_items = 60)
  (h2 : liza_percentage = 90 / 100)
  (h3 : rose_additional_correct = 2) :
  total_items - (total_items * liza_percentage + rose_additional_correct) = 4 :=
by sorry

end NUMINAMATH_CALUDE_rose_incorrect_answers_l770_77061


namespace NUMINAMATH_CALUDE_regular_dinosaur_weight_is_800_l770_77054

/-- The weight of a regular dinosaur in pounds -/
def regular_dinosaur_weight : ℝ := sorry

/-- The weight of Barney the dinosaur in pounds -/
def barney_weight : ℝ := 5 * regular_dinosaur_weight + 1500

/-- The total weight of Barney and five regular dinosaurs in pounds -/
def total_weight : ℝ := 9500

/-- Theorem stating that each regular dinosaur weighs 800 pounds -/
theorem regular_dinosaur_weight_is_800 : regular_dinosaur_weight = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_dinosaur_weight_is_800_l770_77054


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_three_spheres_l770_77006

/-- Given two spheres touching a plane at points B and C, with sum of radii 11 and distance between
    centers 5√17, and a third sphere of radius 8 at point A externally tangent to the other two,
    the radius of the circumcircle of triangle ABC is 2√19. -/
theorem circumcircle_radius_of_three_spheres (R1 R2 : ℝ) (d : ℝ) (R3 : ℝ) :
  R1 + R2 = 11 →
  d = 5 * Real.sqrt 17 →
  R3 = 8 →
  R1 + R2 + 2 * R3 = d →
  ∃ (R : ℝ), R = 2 * Real.sqrt 19 ∧ R = d / 2 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_three_spheres_l770_77006


namespace NUMINAMATH_CALUDE_complex_equation_sum_l770_77005

theorem complex_equation_sum (z : ℂ) (a b : ℝ) : 
  z = a + b * I → z * (1 + I^3) = 2 + I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l770_77005


namespace NUMINAMATH_CALUDE_complement_of_A_l770_77001

def A : Set ℝ := {y | ∃ x, y = 2^x}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l770_77001


namespace NUMINAMATH_CALUDE_einstein_snack_sale_l770_77094

/-- The number of potato fries packs sold by Einstein --/
def potato_fries_packs : ℕ := sorry

theorem einstein_snack_sale :
  let goal : ℚ := 500
  let pizza_price : ℚ := 12
  let fries_price : ℚ := 0.30
  let soda_price : ℚ := 2
  let pizza_sold : ℕ := 15
  let soda_sold : ℕ := 25
  let remaining : ℚ := 258
  
  (pizza_price * pizza_sold + fries_price * potato_fries_packs + soda_price * soda_sold = goal - remaining) ∧
  (potato_fries_packs = 40) := by sorry

end NUMINAMATH_CALUDE_einstein_snack_sale_l770_77094


namespace NUMINAMATH_CALUDE_parabola_intersection_l770_77065

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := 9 * x^2 + 6 * x + 2
  ∀ x y : ℝ, f x = y ∧ g x = y ↔ (x = 0 ∧ y = 2) ∨ (x = -5/3 ∧ y = 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l770_77065


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l770_77048

theorem fixed_point_on_line (k : ℝ) : 1 = k * (-2) + 2 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l770_77048


namespace NUMINAMATH_CALUDE_gcd_problem_l770_77041

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ a = 17 * k) :
  Nat.gcd (Int.natAbs (2 * a ^ 2 + 33 * a + 85)) (Int.natAbs (a + 17)) = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l770_77041


namespace NUMINAMATH_CALUDE_bounded_sequence_convergence_l770_77037

def is_bounded (s : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |s n| ≤ M

theorem bounded_sequence_convergence
  (a : ℕ → ℝ)
  (h_rec : ∀ n : ℕ, a (n + 1) = 3 * a n - 4)
  (h_bounded : is_bounded a) :
  ∀ n : ℕ, a n = 2 :=
sorry

end NUMINAMATH_CALUDE_bounded_sequence_convergence_l770_77037


namespace NUMINAMATH_CALUDE_tim_travel_distance_l770_77008

/-- Represents the distance traveled by Tim and Élan -/
structure TravelDistance where
  tim : ℝ
  elan : ℝ

/-- Calculates the distance traveled in one hour given initial speeds -/
def distanceInHour (timSpeed : ℝ) (elanSpeed : ℝ) : TravelDistance :=
  { tim := timSpeed, elan := elanSpeed }

/-- Theorem: Tim travels 60 miles before meeting Élan -/
theorem tim_travel_distance (initialDistance : ℝ) (timInitialSpeed : ℝ) (elanInitialSpeed : ℝ) :
  initialDistance = 90 ∧ timInitialSpeed = 10 ∧ elanInitialSpeed = 5 →
  (let d1 := distanceInHour timInitialSpeed elanInitialSpeed
   let d2 := distanceInHour (2 * timInitialSpeed) (2 * elanInitialSpeed)
   let d3 := distanceInHour (4 * timInitialSpeed) (4 * elanInitialSpeed)
   d1.tim + d2.tim + (initialDistance - d1.tim - d1.elan - d2.tim - d2.elan) * (4 * timInitialSpeed) / (4 * timInitialSpeed + 4 * elanInitialSpeed) = 60) :=
by
  sorry


end NUMINAMATH_CALUDE_tim_travel_distance_l770_77008


namespace NUMINAMATH_CALUDE_binomial_12_6_l770_77045

theorem binomial_12_6 : Nat.choose 12 6 = 1848 := by sorry

end NUMINAMATH_CALUDE_binomial_12_6_l770_77045


namespace NUMINAMATH_CALUDE_valid_pairs_count_l770_77071

/-- A function that checks if a positive integer has a zero digit. -/
def has_zero_digit (n : ℕ+) : Prop := sorry

/-- The count of ordered pairs (a,b) of positive integers where a + b = 500 and neither a nor b has a zero digit. -/
def count_valid_pairs : ℕ := sorry

/-- Theorem stating that the count of valid pairs is 329. -/
theorem valid_pairs_count : count_valid_pairs = 329 := by sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l770_77071


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l770_77062

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Represents the number of days between June 1 and June 10 -/
def days_between_june1_and_june10 : ℕ := 9

/-- Represents the number of hours between noon and 6 PM -/
def hours_noon_to_6pm : ℕ := 6

/-- Represents the number of minutes the watch loses per day -/
def minutes_lost_per_day : ℕ := 3

/-- Represents the number of minutes added on June 4 -/
def minutes_added_june4 : ℕ := 5

/-- Represents the number of days between June 4 and June 10 -/
def days_between_june4_and_june10 : ℕ := 6

/-- Theorem stating the correct positive correction in minutes -/
theorem watch_correction_theorem :
  let total_hours := days_between_june1_and_june10 * 24 + hours_noon_to_6pm
  let total_minutes_lost := days_between_june1_and_june10 * minutes_lost_per_day
  let minutes_lost_after_adjustment := days_between_june4_and_june10 * minutes_lost_per_day
  total_minutes_lost - minutes_added_june4 + minutes_lost_after_adjustment = 22 := by
  sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l770_77062


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l770_77047

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 2) : y = 2 - 3 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l770_77047


namespace NUMINAMATH_CALUDE_congruence_solution_l770_77055

theorem congruence_solution (n : ℕ) : n ∈ Finset.range 29 → (8 * n ≡ 5 [ZMOD 29]) ↔ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l770_77055


namespace NUMINAMATH_CALUDE_polynomial_unique_value_l770_77075

theorem polynomial_unique_value (P : ℤ → ℤ) :
  (∃ x₁ x₂ x₃ : ℤ, P x₁ = 1 ∧ P x₂ = 2 ∧ P x₃ = 3 ∧ (x₁ = x₂ - 1 ∨ x₁ = x₂ + 1) ∧ (x₂ = x₃ - 1 ∨ x₂ = x₃ + 1)) →
  (∃! x : ℤ, P x = 5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_unique_value_l770_77075


namespace NUMINAMATH_CALUDE_total_buttons_for_order_l770_77040

/-- The number of shirts ordered for each type -/
def shirts_per_type : ℕ := 200

/-- The number of buttons on the first type of shirt -/
def buttons_type1 : ℕ := 3

/-- The number of buttons on the second type of shirt -/
def buttons_type2 : ℕ := 5

/-- Theorem: The total number of buttons needed for the order is 1600 -/
theorem total_buttons_for_order :
  shirts_per_type * buttons_type1 + shirts_per_type * buttons_type2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_for_order_l770_77040


namespace NUMINAMATH_CALUDE_line_intersection_intersection_point_l770_77004

/-- Two lines intersect at a unique point -/
theorem line_intersection (s t : ℝ) : ∃! (p : ℝ × ℝ), 
  (∃ s, p = (1 + 3*s, 2 - 7*s)) ∧ 
  (∃ t, p = (-5 + 5*t, 3 - 8*t)) :=
by sorry

/-- The intersection point of the two lines is (7, -12) -/
theorem intersection_point : 
  ∃ (s t : ℝ), (1 + 3*s, 2 - 7*s) = (-5 + 5*t, 3 - 8*t) ∧ 
                (1 + 3*s, 2 - 7*s) = (7, -12) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_intersection_point_l770_77004


namespace NUMINAMATH_CALUDE_min_value_abc_l770_77088

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 1/a + 1/b + 1/c = 9) : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 9 ∧ 
  x^2 * y^3 * z = 1/108 ∧ ∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 → 
  1/a' + 1/b' + 1/c' = 9 → a'^2 * b'^3 * c' ≥ 1/108 := by
sorry

end NUMINAMATH_CALUDE_min_value_abc_l770_77088


namespace NUMINAMATH_CALUDE_no_valid_partition_l770_77053

theorem no_valid_partition : ¬∃ (A B C : Set ℕ), 
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅) ∧
  (A ∪ B ∪ C = Set.univ) ∧
  (∀ a b, a ∈ A → b ∈ B → a + b + 1 ∈ C) ∧
  (∀ b c, b ∈ B → c ∈ C → b + c + 1 ∈ A) ∧
  (∀ c a, c ∈ C → a ∈ A → c + a + 1 ∈ B) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_partition_l770_77053


namespace NUMINAMATH_CALUDE_card_area_theorem_l770_77010

/-- Represents the dimensions of a rectangular card -/
structure CardDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a card given its dimensions -/
def cardArea (d : CardDimensions) : ℝ := d.length * d.width

/-- Theorem: If shortening one side of a 5x7 card by 2 inches results in an area of 15 square inches,
    then shortening the other side by 2 inches results in an area of 21 square inches -/
theorem card_area_theorem (original : CardDimensions) 
    (h1 : original.length = 5 ∧ original.width = 7)
    (h2 : ∃ (shortened : CardDimensions), 
      (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
      (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
      cardArea shortened = 15) :
  ∃ (other_shortened : CardDimensions),
    ((other_shortened.length = original.length - 2 ∧ other_shortened.width = original.width) ∨
     (other_shortened.length = original.length ∧ other_shortened.width = original.width - 2)) ∧
    cardArea other_shortened = 21 := by
  sorry

end NUMINAMATH_CALUDE_card_area_theorem_l770_77010


namespace NUMINAMATH_CALUDE_intersection_point_translated_line_l770_77060

/-- The intersection point of the line y = 3x + 6 with the x-axis is (-2, 0) -/
theorem intersection_point_translated_line (x y : ℝ) :
  y = 3 * x + 6 ∧ y = 0 → x = -2 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_point_translated_line_l770_77060


namespace NUMINAMATH_CALUDE_perfect_squares_difference_l770_77059

theorem perfect_squares_difference (m : ℕ+) : 
  (∃ a : ℕ, m - 4 = a^2) ∧ (∃ b : ℕ, m + 5 = b^2) → m = 20 ∨ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_l770_77059
