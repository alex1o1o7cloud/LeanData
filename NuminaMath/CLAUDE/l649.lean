import Mathlib

namespace NUMINAMATH_CALUDE_heating_pad_cost_per_use_l649_64989

/-- The cost per use of a heating pad -/
def cost_per_use (total_cost : ℚ) (uses_per_week : ℕ) (weeks : ℕ) : ℚ :=
  total_cost / (uses_per_week * weeks)

/-- Theorem: The cost per use of a $30 heating pad used 3 times a week for 2 weeks is $5 -/
theorem heating_pad_cost_per_use :
  cost_per_use 30 3 2 = 5 := by sorry

end NUMINAMATH_CALUDE_heating_pad_cost_per_use_l649_64989


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l649_64957

/-- Given a geometric sequence {a_n} with common ratio q, 
    if a₁ + a₃ = 20 and a₂ + a₄ = 40, then q = 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = q * a n) 
  (h2 : a 1 + a 3 = 20) 
  (h3 : a 2 + a 4 = 40) : 
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l649_64957


namespace NUMINAMATH_CALUDE_toms_original_amount_l649_64960

theorem toms_original_amount (tom sara jim : ℝ) : 
  tom + sara + jim = 1200 →
  (tom - 200) + (3 * sara) + (2 * jim) = 1800 →
  tom = 400 := by
sorry

end NUMINAMATH_CALUDE_toms_original_amount_l649_64960


namespace NUMINAMATH_CALUDE_arithmetic_simplifications_l649_64900

theorem arithmetic_simplifications :
  (∀ (a b c : Rat), a / 16 - b / 16 + c / 16 = (a - b + c) / 16) ∧
  (∀ (d e f : Rat), d / 12 - e / 12 + f / 12 = (d - e + f) / 12) ∧
  (∀ (g h i j k l m : Nat), g + h + i + j + k + l + m = 736) ∧
  (∀ (n p q r : Rat), n - p / 9 - q / 9 + (1 + r / 99) = 2 + r / 99) →
  5 / 16 - 3 / 16 + 7 / 16 = 9 / 16 ∧
  3 / 12 - 4 / 12 + 6 / 12 = 5 / 12 ∧
  64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 ∧
  2 - 8 / 9 - 1 / 9 + (1 + 98 / 99) = 2 + 98 / 99 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_simplifications_l649_64900


namespace NUMINAMATH_CALUDE_complex_equation_solution_l649_64977

theorem complex_equation_solution (z : ℂ) : (1 - z = z * Complex.I) → z = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l649_64977


namespace NUMINAMATH_CALUDE_age_difference_l649_64928

/-- Given four persons a, b, c, and d with ages A, B, C, and D respectively,
    where the total age of a and b is 11 years more than the total age of b and c,
    prove that c is 11 + D years younger than the sum of the ages of a and d. -/
theorem age_difference (A B C D : ℤ) (h : A + B = B + C + 11) :
  C - (A + D) = -11 - D := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l649_64928


namespace NUMINAMATH_CALUDE_cubic_curve_triangle_problem_l649_64919

/-- A point on the curve y = x^3 -/
structure CubicPoint where
  x : ℝ
  y : ℝ
  cubic_cond : y = x^3

/-- The problem statement -/
theorem cubic_curve_triangle_problem :
  ∃ (A B C : CubicPoint),
    -- A, B, C are distinct
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    -- BC is parallel to x-axis
    B.y = C.y ∧
    -- Area condition
    |C.x - B.x| * |A.y - B.y| = 2000 ∧
    -- Sum of digits of A's x-coordinate is 1
    (∃ (n : ℕ), A.x = 10 * n + 1 ∧ 0 ≤ n ∧ n < 10) := by
  sorry

end NUMINAMATH_CALUDE_cubic_curve_triangle_problem_l649_64919


namespace NUMINAMATH_CALUDE_complex_real_condition_l649_64981

theorem complex_real_condition (a : ℝ) :
  (((a - Complex.I) / (2 + Complex.I)).im = 0) → a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l649_64981


namespace NUMINAMATH_CALUDE_fraction_order_l649_64902

theorem fraction_order : 
  let f1 := 16/12
  let f2 := 21/14
  let f3 := 18/13
  let f4 := 20/15
  f1 < f3 ∧ f3 < f2 ∧ f2 < f4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l649_64902


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l649_64985

-- Problem 1
theorem problem_1 : 211 * (-455) + 365 * 455 - 211 * 545 + 545 * 365 = 154000 := by
  sorry

-- Problem 2
theorem problem_2 : (-7/5 * (-5/2) - 1) / 9 / (1/(-3/4)^2) - |2 + (-1/2)^3 * 5^2| = -31/32 := by
  sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (3*x + 2)*(x + 1) + 2*(x - 3)*(x + 2) = 5*x^2 + 3*x - 10 := by
  sorry

-- Problem 4
theorem problem_4 : ∃ (x : ℚ), (2*x + 3)/6 - (2*x - 1)/4 = 1 ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l649_64985


namespace NUMINAMATH_CALUDE_bicycle_profit_percentage_l649_64948

theorem bicycle_profit_percentage 
  (final_price : ℝ) 
  (initial_cost : ℝ) 
  (intermediate_profit_percentage : ℝ) 
  (h1 : final_price = 225)
  (h2 : initial_cost = 112.5)
  (h3 : intermediate_profit_percentage = 25) :
  let intermediate_cost := final_price / (1 + intermediate_profit_percentage / 100)
  let initial_profit_percentage := (intermediate_cost - initial_cost) / initial_cost * 100
  initial_profit_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_bicycle_profit_percentage_l649_64948


namespace NUMINAMATH_CALUDE_boards_per_package_not_integer_l649_64913

theorem boards_per_package_not_integer (total_boards : ℕ) (num_packages : ℕ) 
  (h1 : total_boards = 154) (h2 : num_packages = 52) : 
  ¬ ∃ (n : ℕ), (total_boards : ℚ) / (num_packages : ℚ) = n := by
  sorry

end NUMINAMATH_CALUDE_boards_per_package_not_integer_l649_64913


namespace NUMINAMATH_CALUDE_h_properties_l649_64947

-- Define the functions f, g, and h
def f : ℝ → ℝ := λ x => x

-- g is symmetric to f with respect to y = x
def g : ℝ → ℝ := λ x => x

def h : ℝ → ℝ := λ x => g (1 - |x|)

-- Theorem statement
theorem h_properties :
  (∀ x, h x = h (-x)) ∧  -- h is an even function
  (∃ m, ∀ x, h x ≥ m ∧ ∃ x₀, h x₀ = m ∧ m = 0) -- The minimum value of h is 0
  := by sorry

end NUMINAMATH_CALUDE_h_properties_l649_64947


namespace NUMINAMATH_CALUDE_toy_distribution_ratio_l649_64939

theorem toy_distribution_ratio (total_toys : ℕ) (num_friends : ℕ) 
  (h1 : total_toys = 118) (h2 : num_friends = 4) :
  ∃ (toys_per_friend : ℕ), 
    toys_per_friend * num_friends ≤ total_toys ∧
    toys_per_friend * num_friends > total_toys - num_friends ∧
    (toys_per_friend : ℚ) / total_toys = 1 / 4 := by
  sorry

#check toy_distribution_ratio

end NUMINAMATH_CALUDE_toy_distribution_ratio_l649_64939


namespace NUMINAMATH_CALUDE_proportional_y_value_l649_64942

/-- Given that y is directly proportional to x+1 and y=4 when x=1, 
    prove that y=6 when x=2 -/
theorem proportional_y_value (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = k * (x + 1)) →  -- y is directly proportional to x+1
  (4 = k * (1 + 1)) →                    -- when x=1, y=4
  (6 = k * (2 + 1)) :=                   -- prove y=6 when x=2
by
  sorry


end NUMINAMATH_CALUDE_proportional_y_value_l649_64942


namespace NUMINAMATH_CALUDE_linear_equation_solution_l649_64904

theorem linear_equation_solution :
  ∀ x : ℝ, x - 2 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l649_64904


namespace NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l649_64995

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- Two angles are adjacent if they share a common vertex and a common side,
    but have no common interior points -/
def Adjacent (α β : ℝ) : Prop := True  -- We simplify this for the statement

theorem adjacent_complementary_angles_are_complementary 
  (α β : ℝ) (h1 : Adjacent α β) (h2 : Complementary α β) : Complementary α β := by
  sorry

end NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l649_64995


namespace NUMINAMATH_CALUDE_wendy_first_level_treasures_l649_64987

/-- Represents the game scenario where Wendy finds treasures on two levels --/
structure GameScenario where
  pointsPerTreasure : ℕ
  treasuresOnSecondLevel : ℕ
  totalScore : ℕ

/-- Calculates the number of treasures found on the first level --/
def treasuresOnFirstLevel (game : GameScenario) : ℕ :=
  (game.totalScore - game.pointsPerTreasure * game.treasuresOnSecondLevel) / game.pointsPerTreasure

/-- Theorem stating that Wendy found 4 treasures on the first level --/
theorem wendy_first_level_treasures :
  let game : GameScenario := {
    pointsPerTreasure := 5,
    treasuresOnSecondLevel := 3,
    totalScore := 35
  }
  treasuresOnFirstLevel game = 4 := by sorry

end NUMINAMATH_CALUDE_wendy_first_level_treasures_l649_64987


namespace NUMINAMATH_CALUDE_katies_miles_l649_64996

/-- Proves that Katie's miles run is 10, given Adam's miles and the difference between their runs -/
theorem katies_miles (adam_miles : ℕ) (difference : ℕ) (h1 : adam_miles = 35) (h2 : difference = 25) :
  adam_miles - difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_katies_miles_l649_64996


namespace NUMINAMATH_CALUDE_susan_is_eleven_l649_64903

/-- Susan's age -/
def susan_age : ℕ := sorry

/-- Ann's age -/
def ann_age : ℕ := sorry

/-- Ann is 5 years older than Susan -/
axiom age_difference : ann_age = susan_age + 5

/-- The sum of their ages is 27 -/
axiom age_sum : ann_age + susan_age = 27

/-- Proof that Susan is 11 years old -/
theorem susan_is_eleven : susan_age = 11 := by sorry

end NUMINAMATH_CALUDE_susan_is_eleven_l649_64903


namespace NUMINAMATH_CALUDE_greatest_x_value_l649_64935

theorem greatest_x_value (x : ℝ) : 
  x ≠ 6 → x ≠ -4 → (x^2 - 3*x - 18) / (x - 6) = 2 / (x + 4) → 
  x ≤ -2 ∧ ∃ y : ℝ, y ≠ 6 ∧ y ≠ -4 ∧ (y^2 - 3*y - 18) / (y - 6) = 2 / (y + 4) ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l649_64935


namespace NUMINAMATH_CALUDE_translation_product_l649_64967

/-- Given a point P(-3, y) translated 3 units down and 2 units left to obtain point Q(x, -1), 
    the product xy equals -10. -/
theorem translation_product (y : ℝ) : 
  let x : ℝ := -3 - 2
  let y' : ℝ := y - 3
  x * y = -10 ∧ y' = -1 := by sorry

end NUMINAMATH_CALUDE_translation_product_l649_64967


namespace NUMINAMATH_CALUDE_quadratic_completion_square_constant_term_value_l649_64908

theorem quadratic_completion_square (x : ℝ) : 
  x^2 - 8*x + 3 = (x - 4)^2 - 13 :=
by sorry

theorem constant_term_value : 
  ∃ (a h : ℝ), ∀ (x : ℝ), x^2 - 8*x + 3 = a*(x - h)^2 - 13 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_constant_term_value_l649_64908


namespace NUMINAMATH_CALUDE_continuity_at_two_l649_64974

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^2 - 4)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_two_l649_64974


namespace NUMINAMATH_CALUDE_car_trip_average_mpg_l649_64990

/-- Proves that the average miles per gallon for a car trip is 450/11, given specific conditions. -/
theorem car_trip_average_mpg :
  -- Define the distance from B to C as x
  ∀ x : ℝ,
  x > 0 →
  -- Distance from A to B is twice the distance from B to C
  let dist_ab := 2 * x
  let dist_bc := x
  -- Define the fuel efficiencies
  let mpg_ab := 25
  let mpg_bc := 30
  -- Calculate total distance and total fuel used
  let total_dist := dist_ab + dist_bc
  let total_fuel := dist_ab / mpg_ab + dist_bc / mpg_bc
  -- The average MPG for the entire trip
  let avg_mpg := total_dist / total_fuel
  -- Prove that the average MPG equals 450/11
  avg_mpg = 450 / 11 := by
    sorry

#eval (450 : ℚ) / 11

end NUMINAMATH_CALUDE_car_trip_average_mpg_l649_64990


namespace NUMINAMATH_CALUDE_find_b_l649_64984

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 3 * x - b else 2^x

theorem find_b : ∃ b : ℝ, f b (f b (5/6)) = 4 ∧ b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l649_64984


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l649_64924

theorem absolute_value_sum_zero (a b : ℝ) :
  |a - 2| + |b + 3| = 0 → b^a = 9 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l649_64924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l649_64998

theorem arithmetic_sequence_sum_divisibility :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x c : ℕ), x > 0 → c ≥ 0 → 
    n ∣ (10 * x + 45 * c)) ∧
  (∀ (m : ℕ), m > n → 
    ∃ (x c : ℕ), x > 0 ∧ c ≥ 0 ∧ 
      ¬(m ∣ (10 * x + 45 * c))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l649_64998


namespace NUMINAMATH_CALUDE_wand_price_theorem_l649_64932

theorem wand_price_theorem (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 8 →
  purchase_price = (1/8) * original_price →
  original_price = 64 := by
sorry

end NUMINAMATH_CALUDE_wand_price_theorem_l649_64932


namespace NUMINAMATH_CALUDE_sergey_ndfl_calculation_l649_64999

/-- Calculates the personal income tax (NDFL) for a Russian resident --/
def calculate_ndfl (monthly_income : ℕ) (bonus : ℕ) (car_sale : ℕ) (land_purchase : ℕ) : ℕ :=
  let annual_income := monthly_income * 12
  let total_income := annual_income + bonus + car_sale
  let total_deductions := car_sale + land_purchase
  let taxable_income := total_income - total_deductions
  (taxable_income * 13) / 100

/-- Theorem stating that the NDFL for Sergey's income is 10400 rubles --/
theorem sergey_ndfl_calculation :
  calculate_ndfl 30000 20000 250000 300000 = 10400 := by
  sorry

end NUMINAMATH_CALUDE_sergey_ndfl_calculation_l649_64999


namespace NUMINAMATH_CALUDE_set_union_condition_l649_64945

theorem set_union_condition (m : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
  let B : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
  A ∪ B = A → m ≤ -2 ∨ (-1 ≤ m ∧ m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_union_condition_l649_64945


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l649_64994

theorem easter_egg_distribution (red_eggs orange_eggs min_eggs : ℕ) 
  (h1 : red_eggs = 30)
  (h2 : orange_eggs = 45)
  (h3 : min_eggs = 5) :
  ∃ (eggs_per_basket : ℕ), 
    eggs_per_basket ≥ min_eggs ∧ 
    eggs_per_basket ∣ red_eggs ∧ 
    eggs_per_basket ∣ orange_eggs ∧
    ∀ (n : ℕ), n > eggs_per_basket → ¬(n ∣ red_eggs ∧ n ∣ orange_eggs) :=
by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l649_64994


namespace NUMINAMATH_CALUDE_distribute_books_count_l649_64931

/-- The number of ways to distribute books among students -/
def distribute_books : ℕ :=
  let num_students : ℕ := 4
  let num_novels : ℕ := 4
  let num_anthologies : ℕ := 1
  -- Category 1: Each student gets 1 novel, anthology to any student
  let category1 : ℕ := num_students
  -- Category 2: Anthology to one student, novels distributed to others
  let category2 : ℕ := num_students * (num_students - 1)
  category1 + category2

/-- Theorem stating that the number of distribution methods is 16 -/
theorem distribute_books_count : distribute_books = 16 := by
  sorry

end NUMINAMATH_CALUDE_distribute_books_count_l649_64931


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l649_64965

theorem intersection_point_x_coordinate 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : ∃! x y : ℝ, x^2 + 2*a*x + 6*b = x^2 + 2*b*x + 6*a) : 
  ∃ x y : ℝ, x^2 + 2*a*x + 6*b = x^2 + 2*b*x + 6*a ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l649_64965


namespace NUMINAMATH_CALUDE_intersection_A_B_l649_64943

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l649_64943


namespace NUMINAMATH_CALUDE_initial_average_height_calculation_l649_64927

theorem initial_average_height_calculation (n : ℕ) (error : ℝ) (actual_avg : ℝ) :
  n = 35 ∧ error = 60 ∧ actual_avg = 178 →
  (n * actual_avg + error) / n = 179.71 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_height_calculation_l649_64927


namespace NUMINAMATH_CALUDE_fourth_post_length_l649_64940

/-- Given a total rope length and the lengths used for the first three posts,
    calculate the length of rope used for the fourth post. -/
def rope_for_fourth_post (total : ℕ) (first : ℕ) (second : ℕ) (third : ℕ) : ℕ :=
  total - (first + second + third)

/-- Theorem stating that given the specific lengths in the problem,
    the rope used for the fourth post is 12 inches. -/
theorem fourth_post_length :
  rope_for_fourth_post 70 24 20 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_post_length_l649_64940


namespace NUMINAMATH_CALUDE_kim_shirts_left_l649_64936

def initial_shirts : ℚ := 4.5 * 12
def bought_shirts : ℕ := 7
def lost_shirts : ℕ := 2
def fraction_given : ℚ := 2 / 5

theorem kim_shirts_left : 
  let total_before_giving := initial_shirts + bought_shirts - lost_shirts
  let given_to_sister := ⌊fraction_given * total_before_giving⌋
  total_before_giving - given_to_sister = 36 := by
  sorry

end NUMINAMATH_CALUDE_kim_shirts_left_l649_64936


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l649_64991

theorem candy_sampling_percentage
  (caught_sampling : Real)
  (total_sampling : Real)
  (h1 : caught_sampling = 22)
  (h2 : total_sampling = 27.5)
  : total_sampling - caught_sampling = 5.5 := by
sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l649_64991


namespace NUMINAMATH_CALUDE_sum_is_composite_l649_64929

theorem sum_is_composite (a b : ℤ) (h : 56 * a = 65 * b) : 
  ∃ (x y : ℤ), x > 1 ∧ y > 1 ∧ a + b = x * y := by
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l649_64929


namespace NUMINAMATH_CALUDE_xy_equals_one_l649_64992

theorem xy_equals_one (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 36 → y = 0.16666666666666666 → x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l649_64992


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l649_64980

theorem quadratic_equation_roots (k : ℤ) :
  let f := fun x : ℝ => k * x^2 - (4*k + 1) * x + 3*k + 3
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₁^2 + x₂^2 = (3 * Real.sqrt 5 / 2)^2 → k = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l649_64980


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l649_64956

theorem arithmetic_calculation : (40 * 30 + (12 + 8) * 3) / 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l649_64956


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l649_64993

theorem quadratic_perfect_square (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (x + b)^2) ↔ k = 100 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l649_64993


namespace NUMINAMATH_CALUDE_total_wheels_count_l649_64986

def bicycle_count : Nat := 3
def tricycle_count : Nat := 4
def unicycle_count : Nat := 7

def bicycle_wheels : Nat := 2
def tricycle_wheels : Nat := 3
def unicycle_wheels : Nat := 1

theorem total_wheels_count : 
  bicycle_count * bicycle_wheels + 
  tricycle_count * tricycle_wheels + 
  unicycle_count * unicycle_wheels = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l649_64986


namespace NUMINAMATH_CALUDE_inequality_solution_l649_64949

theorem inequality_solution (x : ℝ) : 
  (9*x^2 + 27*x - 64) / ((3*x - 5)*(x + 3)) < 2 ↔ 
  x < -3 ∨ (-17/3 < x ∧ x < 5/3) ∨ 2 < x := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l649_64949


namespace NUMINAMATH_CALUDE_average_first_five_primes_gt_50_l649_64911

def first_five_primes_gt_50 : List Nat := [53, 59, 61, 67, 71]

def average (lst : List Nat) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem average_first_five_primes_gt_50 :
  average first_five_primes_gt_50 = 62.2 := by
  sorry

end NUMINAMATH_CALUDE_average_first_five_primes_gt_50_l649_64911


namespace NUMINAMATH_CALUDE_prank_combinations_l649_64941

/-- Represents the number of choices for each day of the week --/
def choices : List Nat := [1, 2, 6, 3, 1]

/-- Calculates the total number of combinations --/
def totalCombinations (choices : List Nat) : Nat :=
  choices.prod

/-- Theorem: The total number of combinations for the given choices is 36 --/
theorem prank_combinations :
  totalCombinations choices = 36 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l649_64941


namespace NUMINAMATH_CALUDE_first_trail_length_is_20_l649_64955

/-- The length of the first trail in miles -/
def first_trail_length : ℝ := 20

/-- The speed of hiking the first trail in miles per hour -/
def first_trail_speed : ℝ := 5

/-- The length of the second trail in miles -/
def second_trail_length : ℝ := 12

/-- The speed of hiking the second trail in miles per hour -/
def second_trail_speed : ℝ := 3

/-- The duration of the break during the second trail in hours -/
def break_duration : ℝ := 1

/-- The time difference between the two trails in hours -/
def time_difference : ℝ := 1

theorem first_trail_length_is_20 :
  first_trail_length = 20 ∧
  first_trail_length / first_trail_speed = 
    (second_trail_length / second_trail_speed + break_duration) - time_difference :=
by sorry

end NUMINAMATH_CALUDE_first_trail_length_is_20_l649_64955


namespace NUMINAMATH_CALUDE_calvins_roaches_l649_64997

theorem calvins_roaches (total insects : ℕ) (scorpions : ℕ) (roaches crickets caterpillars : ℕ) : 
  insects = 27 →
  scorpions = 3 →
  crickets = roaches / 2 →
  caterpillars = 2 * scorpions →
  insects = roaches + scorpions + crickets + caterpillars →
  roaches = 12 := by
sorry

end NUMINAMATH_CALUDE_calvins_roaches_l649_64997


namespace NUMINAMATH_CALUDE_lassie_bones_problem_l649_64952

theorem lassie_bones_problem (B : ℝ) : 
  (4/5 * (3/4 * (2/3 * B + 5) + 8) + 15 = 60) → B = 89 := by
  sorry

end NUMINAMATH_CALUDE_lassie_bones_problem_l649_64952


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean_55_l649_64954

theorem max_ratio_two_digit_mean_55 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 55 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 55 →
  x / y ≤ 9 ∧ x / y ≥ a / b :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean_55_l649_64954


namespace NUMINAMATH_CALUDE_largest_number_l649_64922

theorem largest_number (a b c d e : ℝ) 
  (ha : a = 0.993) 
  (hb : b = 0.9899) 
  (hc : c = 0.990) 
  (hd : d = 0.989) 
  (he : e = 0.9909) : 
  a > b ∧ a > c ∧ a > d ∧ a > e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l649_64922


namespace NUMINAMATH_CALUDE_mark_deck_project_cost_l649_64966

/-- Calculates the total cost of a multi-layered deck project --/
def deck_project_cost (length width : ℝ) 
                      (material_a_cost material_b_cost material_c_cost : ℝ) 
                      (beam_cost sealant_cost : ℝ) 
                      (railing_cost_30 railing_cost_40 : ℝ) 
                      (tax_rate : ℝ) : ℝ :=
  let area := length * width
  let material_cost := area * (material_a_cost + material_b_cost + material_c_cost)
  let beam_cost_total := area * beam_cost * 2
  let sealant_cost_total := area * sealant_cost
  let railing_cost_total := 2 * (railing_cost_30 + railing_cost_40)
  let subtotal := material_cost + beam_cost_total + sealant_cost_total + railing_cost_total
  let tax := subtotal * tax_rate
  subtotal + tax

/-- The total cost of Mark's deck project is $25423.20 --/
theorem mark_deck_project_cost : 
  deck_project_cost 30 40 3 5 8 2 1 120 160 0.07 = 25423.20 := by
  sorry

end NUMINAMATH_CALUDE_mark_deck_project_cost_l649_64966


namespace NUMINAMATH_CALUDE_intersection_distance_l649_64920

-- Define the line x = 4
def line (x : ℝ) : Prop := x = 4

-- Define the curve x = t², y = t³
def curve (t x y : ℝ) : Prop := x = t^2 ∧ y = t^3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 ∧ curve 2 A.1 A.2 ∧
  line B.1 ∧ curve (-2) B.1 B.2

-- Theorem statement
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l649_64920


namespace NUMINAMATH_CALUDE_non_integer_factors_integer_products_l649_64917

theorem non_integer_factors_integer_products :
  ∃ (a b c : ℝ),
    (¬ ∃ (n : ℤ), a = n) ∧
    (¬ ∃ (n : ℤ), b = n) ∧
    (¬ ∃ (n : ℤ), c = n) ∧
    (∃ (m : ℤ), a * b = m) ∧
    (∃ (m : ℤ), b * c = m) ∧
    (∃ (m : ℤ), c * a = m) ∧
    (∃ (m : ℤ), a * b * c = m) :=
by sorry

end NUMINAMATH_CALUDE_non_integer_factors_integer_products_l649_64917


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l649_64938

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l649_64938


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt13_l649_64983

/-- A circle with two given points on its circumference and its center on the y-axis -/
structure CircleWithPoints where
  center : ℝ × ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  point1_on_circle : (point1.1 - center.1)^2 + (point1.2 - center.2)^2 = (point2.1 - center.1)^2 + (point2.2 - center.2)^2

/-- The radius of the circle is √13 -/
theorem circle_radius_is_sqrt13 (c : CircleWithPoints) 
  (h1 : c.point1 = (2, 5)) 
  (h2 : c.point2 = (3, 6)) : 
  Real.sqrt ((c.point1.1 - c.center.1)^2 + (c.point1.2 - c.center.2)^2) = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_circle_radius_is_sqrt13_l649_64983


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l649_64958

theorem decagon_adjacent_vertices_probability :
  let n : ℕ := 10  -- number of vertices in a decagon
  let adjacent_pairs : ℕ := 2  -- number of adjacent vertices for any chosen vertex
  let total_choices : ℕ := n - 1  -- total number of choices for the second vertex
  (adjacent_pairs : ℚ) / total_choices = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l649_64958


namespace NUMINAMATH_CALUDE_infinite_solutions_infinitely_many_solutions_l649_64914

/-- The type of solutions to the equation x^3 + y^3 = z^4 - t^2 -/
def Solution := ℤ × ℤ × ℤ × ℤ

/-- Predicate to check if a tuple (x, y, z, t) is a solution to the equation -/
def is_solution (s : Solution) : Prop :=
  let (x, y, z, t) := s
  x^3 + y^3 = z^4 - t^2

/-- Function to transform a solution using an integer k -/
def transform (k : ℤ) (s : Solution) : Solution :=
  let (x, y, z, t) := s
  (k^4 * x, k^4 * y, k^3 * z, k^6 * t)

/-- Theorem stating that if (x, y, z, t) is a solution, then (k^4*x, k^4*y, k^3*z, k^6*t) is also a solution for any integer k -/
theorem infinite_solutions (s : Solution) (k : ℤ) :
  is_solution s → is_solution (transform k s) := by
  sorry

/-- Corollary: There are infinitely many solutions to the equation -/
theorem infinitely_many_solutions :
  ∃ f : ℕ → Solution, ∀ n : ℕ, is_solution (f n) ∧ ∀ m : ℕ, m ≠ n → f m ≠ f n := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_infinitely_many_solutions_l649_64914


namespace NUMINAMATH_CALUDE_melanie_missed_games_l649_64978

theorem melanie_missed_games (total_games attended_games : ℕ) 
  (h1 : total_games = 64)
  (h2 : attended_games = 32) :
  total_games - attended_games = 32 := by
  sorry

end NUMINAMATH_CALUDE_melanie_missed_games_l649_64978


namespace NUMINAMATH_CALUDE_remainder_theorem_l649_64934

theorem remainder_theorem (n : ℕ) 
  (h1 : n % 22 = 7) 
  (h2 : n % 33 = 18) : 
  n % 66 = 51 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l649_64934


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_l649_64909

theorem gcd_of_powers_of_two : Nat.gcd (2^1502 - 1) (2^1513 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_l649_64909


namespace NUMINAMATH_CALUDE_second_reduction_percentage_l649_64950

/-- Given two successive price reductions, where the first is 25% and the combined effect
    is equivalent to a single 47.5% reduction, proves that the second reduction is 30%. -/
theorem second_reduction_percentage (P : ℝ) (x : ℝ) 
  (h1 : P > 0)  -- Assume positive initial price
  (h2 : 0 ≤ x ∧ x ≤ 1)  -- Second reduction percentage is between 0 and 1
  (h3 : (1 - x) * (P - 0.25 * P) = P - 0.475 * P)  -- Combined reduction equation
  : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_l649_64950


namespace NUMINAMATH_CALUDE_min_value_abc_l649_64933

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  a^2 * b^3 * c^4 ≥ 1/1728 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^2 * b₀^3 * c₀^4 = 1/1728 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l649_64933


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l649_64959

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l649_64959


namespace NUMINAMATH_CALUDE_percentage_commutation_l649_64930

theorem percentage_commutation (n : ℝ) (h : 0.3 * (0.4 * n) = 24) : 0.4 * (0.3 * n) = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l649_64930


namespace NUMINAMATH_CALUDE_quadratic_inequality_set_l649_64961

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * m * x + 5 * m + 1

-- State the theorem
theorem quadratic_inequality_set :
  {m : ℝ | ∀ x : ℝ, f m x > 0} = {m : ℝ | 0 ≤ m ∧ m < 1/4} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_set_l649_64961


namespace NUMINAMATH_CALUDE_compare_powers_l649_64937

theorem compare_powers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a < 1) (hba : 1 < b) :
  a^4 < 1 ∧ 1 < b^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l649_64937


namespace NUMINAMATH_CALUDE_second_number_calculation_l649_64921

theorem second_number_calculation (A B : ℝ) : 
  A = 700 → 
  0.3 * A = 0.6 * B + 120 → 
  B = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l649_64921


namespace NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l649_64912

theorem semicircles_to_circle_area_ratio : 
  let r₁ : ℝ := 10  -- radius of the larger circle
  let r₂ : ℝ := 8   -- radius of the first semicircle
  let r₃ : ℝ := 6   -- radius of the second semicircle
  let circle_area := π * r₁^2
  let semicircle_area_1 := (π * r₂^2) / 2
  let semicircle_area_2 := (π * r₃^2) / 2
  let combined_semicircle_area := semicircle_area_1 + semicircle_area_2
  (combined_semicircle_area / circle_area) = (1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_semicircles_to_circle_area_ratio_l649_64912


namespace NUMINAMATH_CALUDE_triangle_height_from_rectangle_l649_64976

/-- Given a 9x27 rectangle cut into two congruent trapezoids and rearranged to form a triangle with base 9, the height of the resulting triangle is 54 units. -/
theorem triangle_height_from_rectangle (rectangle_length : ℝ) (rectangle_width : ℝ) (triangle_base : ℝ) :
  rectangle_length = 27 →
  rectangle_width = 9 →
  triangle_base = rectangle_width →
  (1 / 2 : ℝ) * triangle_base * 54 = rectangle_length * rectangle_width :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_from_rectangle_l649_64976


namespace NUMINAMATH_CALUDE_tangent_line_of_quartic_curve_l649_64970

/-- The curve y = x^4 has a tangent line parallel to x + 2y - 8 = 0, 
    and this tangent line has the equation 8x + 16y + 3 = 0 -/
theorem tangent_line_of_quartic_curve (x y : ℝ) : 
  y = x^4 → 
  ∃ (x₀ y₀ : ℝ), y₀ = x₀^4 ∧ 
    (∀ (x' y' : ℝ), y' - y₀ = 4 * x₀^3 * (x' - x₀) → 
      ∃ (k : ℝ), y' - y₀ = k * (x' - x₀) ∧ k = -1/2) →
    8 * x₀ + 16 * y₀ + 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_of_quartic_curve_l649_64970


namespace NUMINAMATH_CALUDE_regular_octagon_side_length_l649_64905

/-- A regular octagon with a perimeter of 23.6 cm has sides of length 2.95 cm. -/
theorem regular_octagon_side_length : 
  ∀ (perimeter side_length : ℝ),
  perimeter = 23.6 →
  perimeter = 8 * side_length →
  side_length = 2.95 := by
sorry

end NUMINAMATH_CALUDE_regular_octagon_side_length_l649_64905


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l649_64964

def father_age : ℕ := 40
def son_age : ℕ := 10

theorem father_son_age_ratio :
  (father_age : ℚ) / son_age = 4 ∧
  father_age + 20 = 2 * (son_age + 20) :=
by sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l649_64964


namespace NUMINAMATH_CALUDE_yellow_ball_players_l649_64979

theorem yellow_ball_players (total : ℕ) (white : ℕ) (both : ℕ) (yellow : ℕ) : 
  total = 35 → white = 26 → both = 19 → yellow = 28 → 
  total = white + yellow - both :=
by sorry

end NUMINAMATH_CALUDE_yellow_ball_players_l649_64979


namespace NUMINAMATH_CALUDE_tiffany_score_l649_64988

/-- The score for each treasure found -/
def points_per_treasure : ℕ := 6

/-- The number of treasures found on the first level -/
def treasures_level1 : ℕ := 3

/-- The number of treasures found on the second level -/
def treasures_level2 : ℕ := 5

/-- Tiffany's total score -/
def total_score : ℕ := points_per_treasure * (treasures_level1 + treasures_level2)

theorem tiffany_score : total_score = 48 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_score_l649_64988


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_is_two_l649_64944

/-- The sequence a_n defined as n^2! + n -/
def a (n : ℕ) : ℕ := (Nat.factorial (n^2)) + n

/-- The theorem stating that the maximum GCD of consecutive terms in the sequence is 2 -/
theorem max_gcd_consecutive_terms_is_two :
  ∃ (k : ℕ), (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧ 
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) ∧
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_is_two_l649_64944


namespace NUMINAMATH_CALUDE_max_value_f_one_l649_64951

/-- Given a function f(x) = x^2 + abx + a + 2b where f(0) = 4,
    the maximum value of f(1) is 7. -/
theorem max_value_f_one (a b : ℝ) :
  let f := fun x : ℝ => x^2 + a*b*x + a + 2*b
  f 0 = 4 →
  (∀ x : ℝ, f 1 ≤ 7) ∧ (∃ x : ℝ, f 1 = 7) :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_one_l649_64951


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l649_64975

theorem imaginary_part_of_2_minus_3i :
  Complex.im (2 - 3 * Complex.I) = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l649_64975


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l649_64971

theorem similar_triangles_leg_sum (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → A > 0 → B > 0 → C > 0 →
  (1/2) * a * b = 6 →
  (1/2) * A * B = 150 →
  c = 5 →
  a^2 + b^2 = c^2 →
  A^2 + B^2 = C^2 →
  (a/A)^2 = (b/B)^2 →
  (a/A)^2 = (c/C)^2 →
  A + B = 35 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l649_64971


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l649_64962

theorem square_minus_product_plus_square : 7^2 - 4*5 + 6^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l649_64962


namespace NUMINAMATH_CALUDE_root_sum_product_l649_64973

theorem root_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 2 = 0) → 
  (b^4 - 6*b - 2 = 0) → 
  (a ≠ b) →
  (a*b + a + b = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_product_l649_64973


namespace NUMINAMATH_CALUDE_system_solution_l649_64907

/-- Given a system of equations and a partial solution, prove the complete solution -/
theorem system_solution (a : ℝ) :
  (∃ x y : ℝ, 2*x + y = a ∧ 2*x - y = 12 ∧ x = 5) →
  (∃ x y : ℝ, 2*x + y = a ∧ 2*x - y = 12 ∧ x = 5 ∧ y = -2 ∧ a = 8) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l649_64907


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l649_64963

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 5

/-- The total number of possible chords between num_points points -/
def total_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of ways to select num_selected_chords from total_chords -/
def ways_to_select_chords (n : ℕ) : ℕ := (total_chords n).choose num_selected_chords

/-- The number of ways to choose 5 points from num_points points -/
def convex_pentagons (n : ℕ) : ℕ := n.choose 5

/-- The probability of forming a convex pentagon -/
def probability : ℚ := (convex_pentagons num_points : ℚ) / (ways_to_select_chords num_points : ℚ)

theorem convex_pentagon_probability :
  probability = 1 / 1755 :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l649_64963


namespace NUMINAMATH_CALUDE_car_speed_problem_l649_64901

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 42 ∧ average_speed = 66 →
  ∃ speed_first_hour : ℝ,
    speed_first_hour = 90 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l649_64901


namespace NUMINAMATH_CALUDE_scooter_initial_value_l649_64968

/-- Proves that the initial value of a scooter is 40000 given its depreciation rate and value after 2 years -/
theorem scooter_initial_value (depreciation_rate : ℚ) (value_after_two_years : ℚ) :
  depreciation_rate = 3 / 4 →
  value_after_two_years = 22500 →
  depreciation_rate * (depreciation_rate * 40000) = value_after_two_years :=
by sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l649_64968


namespace NUMINAMATH_CALUDE_division_equality_l649_64926

theorem division_equality (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l649_64926


namespace NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l649_64918

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  first_term : ℝ  -- First term of the sequence
  common_diff : ℝ  -- Common difference
  sum_formula : ∀ n : ℕ, S n = n / 2 * (2 * first_term + (n - 1) * common_diff)
  seq_formula : ∀ n : ℕ, a n = first_term + (n - 1) * common_diff

/-- Theorem: If (S_4 / 4) - (S_2 / 2) = 2 for an arithmetic sequence, 
    then its common difference is 2 -/
theorem arithmetic_seq_common_diff 
  (seq : ArithmeticSequence) 
  (h : seq.S 4 / 4 - seq.S 2 / 2 = 2) : 
  seq.common_diff = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_common_diff_l649_64918


namespace NUMINAMATH_CALUDE_group_size_proof_l649_64969

theorem group_size_proof (total_spent : ℕ) (mango_price : ℕ) (pineapple_price : ℕ) (pineapple_spent : ℕ) :
  total_spent = 94 →
  mango_price = 5 →
  pineapple_price = 6 →
  pineapple_spent = 54 →
  ∃ (mango_count pineapple_count : ℕ),
    mango_count * mango_price + pineapple_count * pineapple_price = total_spent ∧
    pineapple_count * pineapple_price = pineapple_spent ∧
    mango_count + pineapple_count = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l649_64969


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l649_64906

theorem sandwich_jam_cost 
  (N B J : ℕ) 
  (h1 : N > 1) 
  (h2 : B > 0) 
  (h3 : J > 0) 
  (h4 : N * (4 * B + 5 * J + 20) = 414) : 
  N * 5 * J = 225 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l649_64906


namespace NUMINAMATH_CALUDE_remaining_sample_is_nineteen_l649_64972

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Theorem stating that the remaining sample number is 19 -/
theorem remaining_sample_is_nineteen (s : SystematicSampling)
  (h1 : s.total_students = 56)
  (h2 : s.sample_size = 4)
  (h3 : s.known_samples = [5, 33, 47])
  : ∃ (remaining : ℕ), remaining = 19 ∧ remaining ∉ s.known_samples :=
by sorry

end NUMINAMATH_CALUDE_remaining_sample_is_nineteen_l649_64972


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l649_64946

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) * (n - 2) = 6 * (n * (n - 1) * (n - 2) * (n - 3)) / 24) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l649_64946


namespace NUMINAMATH_CALUDE_set_operations_l649_64953

-- Define the universal set U
def U : Set ℕ := {x | x < 10}

-- Define set A
def A : Set ℕ := {x ∈ U | ∃ k, x = 2 * k}

-- Define set B
def B : Set ℕ := {x | x^2 - 3*x + 2 = 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {2}) ∧
  (A ∪ B = {0, 1, 2, 4, 6, 8}) ∧
  (U \ A = {1, 3, 5, 7, 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l649_64953


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l649_64925

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, (2 * (x1 - 1)^2 = 18 ∧ x1 = 4) ∧ (2 * (x2 - 1)^2 = 18 ∧ x2 = -2)) ∧
  (∃ y1 y2 : ℝ, (y1^2 - 4*y1 - 3 = 0 ∧ y1 = 2 + Real.sqrt 7) ∧ (y2^2 - 4*y2 - 3 = 0 ∧ y2 = 2 - Real.sqrt 7)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l649_64925


namespace NUMINAMATH_CALUDE_problem_solving_probability_l649_64915

theorem problem_solving_probability : 
  let p_arthur : ℚ := 1/4
  let p_bella : ℚ := 3/10
  let p_xavier : ℚ := 1/6
  let p_yvonne : ℚ := 1/2
  let p_zelda : ℚ := 5/8
  let p_not_zelda : ℚ := 1 - p_zelda
  let p_four_solve : ℚ := p_arthur * p_yvonne * p_bella * p_xavier * p_not_zelda
  p_four_solve = 9/3840 := by sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l649_64915


namespace NUMINAMATH_CALUDE_trader_theorem_l649_64910

def trader_problem (profit goal donations : ℕ) : Prop :=
  let half_profit := profit / 2
  let total_available := half_profit + donations
  total_available - goal = 180

theorem trader_theorem : trader_problem 960 610 310 := by
  sorry

end NUMINAMATH_CALUDE_trader_theorem_l649_64910


namespace NUMINAMATH_CALUDE_engineer_designer_ratio_l649_64916

/-- Represents the ratio of engineers to designers in a team -/
structure TeamRatio where
  engineers : ℕ
  designers : ℕ

/-- Proves that the ratio of engineers to designers is 2:1 given the average ages -/
theorem engineer_designer_ratio (team_avg : ℝ) (engineer_avg : ℝ) (designer_avg : ℝ) 
    (h1 : team_avg = 52) (h2 : engineer_avg = 48) (h3 : designer_avg = 60) : 
    ∃ (ratio : TeamRatio), ratio.engineers = 2 ∧ ratio.designers = 1 := by
  sorry

#check engineer_designer_ratio

end NUMINAMATH_CALUDE_engineer_designer_ratio_l649_64916


namespace NUMINAMATH_CALUDE_parabola_through_points_l649_64982

/-- A parabola passing through three specific points -/
def parabola (x y : ℝ) : Prop :=
  y = -x^2 + 2*x + 3

theorem parabola_through_points :
  parabola (-1) 0 ∧ parabola 3 0 ∧ parabola 0 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_points_l649_64982


namespace NUMINAMATH_CALUDE_distance_between_foci_l649_64923

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 := by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l649_64923
