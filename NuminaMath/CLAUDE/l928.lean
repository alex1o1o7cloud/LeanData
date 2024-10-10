import Mathlib

namespace hollow_block_3_9_5_cubes_l928_92884

/-- Calculates the number of unit cubes needed to construct the outer shell of a hollow rectangular block. -/
def hollow_block_cubes (length width depth : ℕ) : ℕ :=
  2 * (length * width) +  -- top and bottom
  2 * ((width * depth) - (width * 2)) +  -- longer sides
  2 * ((length * depth) - (length * 2) - 2)  -- shorter sides

/-- Theorem stating that a hollow rectangular block with dimensions 3 x 9 x 5 requires 122 unit cubes. -/
theorem hollow_block_3_9_5_cubes : 
  hollow_block_cubes 3 9 5 = 122 := by
  sorry

#eval hollow_block_cubes 3 9 5  -- Should output 122

end hollow_block_3_9_5_cubes_l928_92884


namespace arithmetic_sequence_length_l928_92871

theorem arithmetic_sequence_length : 
  ∀ (a d : ℤ) (n : ℕ), 
    a - d * (n - 1) = 39 → 
    a = 147 → 
    d = 3 → 
    n = 37 :=
by
  sorry

end arithmetic_sequence_length_l928_92871


namespace roots_of_quadratic_equation_l928_92820

theorem roots_of_quadratic_equation (α β : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = α ∨ x = β) →
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end roots_of_quadratic_equation_l928_92820


namespace fraction_sum_equals_two_l928_92866

theorem fraction_sum_equals_two (a b : ℝ) (ha : a ≠ 0) : 
  (2*b + a) / a + (a - 2*b) / a = 2 := by
sorry

end fraction_sum_equals_two_l928_92866


namespace triangle_inequality_inside_l928_92886

/-- A point is inside a triangle if it's in the interior of the triangle --/
def PointInsideTriangle (A B C M : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
  M = α • A + β • B + γ • C

/-- The theorem statement --/
theorem triangle_inequality_inside (A B C M : EuclideanSpace ℝ (Fin 2)) 
  (h : PointInsideTriangle A B C M) : 
  dist M B + dist M C < dist A B + dist A C := by
  sorry

end triangle_inequality_inside_l928_92886


namespace smallest_solution_abs_equation_l928_92844

theorem smallest_solution_abs_equation :
  ∃ x : ℝ, (∀ y : ℝ, |y - 2| = |y - 3| + 1 → x ≤ y) ∧ |x - 2| = |x - 3| + 1 :=
by
  sorry

end smallest_solution_abs_equation_l928_92844


namespace greatest_whole_number_satisfying_inequality_l928_92862

theorem greatest_whole_number_satisfying_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ (5 * x - 4 : ℝ) < (3 - 2 * x : ℝ) := by
  sorry

end greatest_whole_number_satisfying_inequality_l928_92862


namespace households_with_bike_only_l928_92880

theorem households_with_bike_only 
  (total : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90) 
  (h2 : neither = 11) 
  (h3 : both = 20) 
  (h4 : with_car = 44) : 
  total - neither - (with_car - both) - both = 35 := by
sorry

end households_with_bike_only_l928_92880


namespace inequality_proof_l928_92883

theorem inequality_proof (x y z u : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (hx1 : x ≠ 1) (hy1 : y ≠ 1) (hz1 : z ≠ 1) (hu1 : u ≠ 1)
  (h_all_gt : (x > 1 ∧ y > 1 ∧ z > 1 ∧ u > 1) ∨ (x < 1 ∧ y < 1 ∧ z < 1 ∧ u < 1)) :
  (Real.log x ^ 3 / Real.log y) / (x + y + z) +
  (Real.log y ^ 3 / Real.log z) / (y + z + u) +
  (Real.log z ^ 3 / Real.log u) / (z + u + x) +
  (Real.log u ^ 3 / Real.log x) / (u + x + y) ≥
  16 / (x + y + z + u) :=
sorry

end inequality_proof_l928_92883


namespace quadricycle_count_l928_92846

theorem quadricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_wheels = 30) : ∃ (b t q : ℕ), 
  b + t + q = total_children ∧ 
  2 * b + 3 * t + 4 * q = total_wheels ∧ 
  q = 2 := by
  sorry

end quadricycle_count_l928_92846


namespace xiaoming_scoring_frequency_l928_92849

/-- The frequency of scoring given total shots and goals -/
def scoring_frequency (total_shots : ℕ) (goals : ℕ) : ℚ :=
  (goals : ℚ) / (total_shots : ℚ)

/-- Theorem stating that given 80 total shots and 50 goals, the frequency of scoring is 0.625 -/
theorem xiaoming_scoring_frequency :
  scoring_frequency 80 50 = 0.625 := by
  sorry

end xiaoming_scoring_frequency_l928_92849


namespace set_union_problem_l928_92833

theorem set_union_problem (a b : ℕ) :
  let M : Set ℕ := {3, 2^a}
  let N : Set ℕ := {a, b}
  M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
  sorry

end set_union_problem_l928_92833


namespace sum_of_roots_l928_92875

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x = 1) 
  (hy : y^3 - 3*y^2 + 5*y = 5) : 
  x + y = 2 := by
sorry

end sum_of_roots_l928_92875


namespace largest_power_of_two_dividing_difference_l928_92869

theorem largest_power_of_two_dividing_difference : ∃ k : ℕ, 
  (2^k : ℤ) ∣ (17^4 - 13^4) ∧ 
  ∀ m : ℕ, (2^m : ℤ) ∣ (17^4 - 13^4) → m ≤ k :=
by
  -- The proof would go here
  sorry

end largest_power_of_two_dividing_difference_l928_92869


namespace min_plates_for_five_colors_l928_92803

/-- The minimum number of plates to pull out to guarantee a matching pair -/
def min_plates_for_match (num_colors : ℕ) : ℕ :=
  num_colors + 1

/-- Theorem stating that for 5 colors, the minimum number of plates to pull out for a match is 6 -/
theorem min_plates_for_five_colors :
  min_plates_for_match 5 = 6 := by
  sorry

end min_plates_for_five_colors_l928_92803


namespace tan_beta_value_l928_92810

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -3/4)
  (h2 : Real.tan (α + β) = 1) : 
  Real.tan β = 7 := by
sorry

end tan_beta_value_l928_92810


namespace tan_is_odd_l928_92802

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define the property of being an odd function
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem tan_is_odd : IsOdd Real.tan := by
  sorry

end tan_is_odd_l928_92802


namespace fraction_meaningful_condition_l928_92888

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x + 1)) ↔ x ≠ -1 := by
sorry

end fraction_meaningful_condition_l928_92888


namespace max_value_of_f_l928_92811

-- Define the function to be maximized
def f (a b c : ℝ) : ℝ := a * b + b * c + 2 * a * c

-- State the theorem
theorem max_value_of_f :
  ∀ a b c : ℝ,
  a ≥ 0 → b ≥ 0 → c ≥ 0 →
  a + b + c = 1 →
  f a b c ≤ 1/2 :=
sorry

end max_value_of_f_l928_92811


namespace power_multiplication_l928_92829

theorem power_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end power_multiplication_l928_92829


namespace cloth_sales_calculation_l928_92850

/-- Calculates the total sales given the commission rate and commission amount -/
def totalSales (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem: Given a commission rate of 2.5% and a commission of 18, the total sales is 720 -/
theorem cloth_sales_calculation :
  totalSales (2.5 : ℚ) 18 = 720 := by
  sorry

end cloth_sales_calculation_l928_92850


namespace point_outside_circle_l928_92842

/-- Given a line ax + by = 1 and a circle x^2 + y^2 = 1 that intersect at two distinct points,
    prove that the point (a, b) lies outside the circle. -/
theorem point_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  (∃ x' y' : ℝ, x' ≠ y' ∧ a * x' + b * y' = 1 ∧ x'^2 + y'^2 = 1) →
  a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l928_92842


namespace max_intersections_10_points_l928_92895

/-- The maximum number of intersection points of perpendicular bisectors for n points -/
def max_intersections (n : ℕ) : ℕ :=
  Nat.choose n 3 + 3 * Nat.choose n 4

/-- Theorem stating the maximum number of intersection points for 10 points -/
theorem max_intersections_10_points :
  max_intersections 10 = 750 := by sorry

end max_intersections_10_points_l928_92895


namespace primes_rounding_to_40_l928_92839

def roundToNearestTen (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem primes_rounding_to_40 :
  ∃! (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p ∧ roundToNearestTen p = 40) ∧ 
    (∀ p, Nat.Prime p → roundToNearestTen p = 40 → p ∈ S) ∧ 
    S.card = 3 :=
by sorry

end primes_rounding_to_40_l928_92839


namespace barn_paint_area_l928_92877

/-- Represents the dimensions of a rectangular prism -/
structure BarnDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in the barn -/
def total_paint_area (dim : BarnDimensions) : ℝ :=
  let end_wall_area := 2 * dim.width * dim.height
  let side_wall_area := 2 * dim.length * dim.height
  let ceiling_area := dim.length * dim.width
  let partition_area := 2 * dim.length * dim.height
  2 * (end_wall_area + side_wall_area) + ceiling_area + partition_area

/-- The barn dimensions -/
def barn : BarnDimensions :=
  { length := 15
  , width := 12
  , height := 6 }

theorem barn_paint_area :
  total_paint_area barn = 1008 := by
  sorry

end barn_paint_area_l928_92877


namespace altered_difference_larger_l928_92852

theorem altered_difference_larger (a b : ℤ) (h1 : a > b) (h2 : b > 0) :
  (1.03 : ℝ) * (a : ℝ) - 0.98 * (b : ℝ) > (a : ℝ) - (b : ℝ) := by
  sorry

end altered_difference_larger_l928_92852


namespace inscribed_sphere_volume_l928_92813

/-- The volume of a sphere inscribed in a cube with edge length 10 inches -/
theorem inscribed_sphere_volume :
  let edge_length : ℝ := 10
  let radius : ℝ := edge_length / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume = (500 / 3) * Real.pi := by sorry

end inscribed_sphere_volume_l928_92813


namespace b_upper_bound_l928_92854

theorem b_upper_bound (b : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (1/2), Real.sqrt (1 - x^2) > x + b) → 
  b < 0 := by
sorry

end b_upper_bound_l928_92854


namespace point_in_region_l928_92831

def is_in_region (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 25 ∧ 
  (x + y ≠ 0 → -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1) ∧
  (x ≠ 0 → 0 ≤ x / y ∧ x / y ≤ 1)

theorem point_in_region (x y : ℝ) :
  is_in_region x y ↔ 
    (x^2 + y^2 ≤ 25 ∧ 
     ((x ≠ 0 ∧ y ≠ 0 ∧ 0 ≤ x / y ∧ x / y ≤ 1) ∨ 
      (x = 0 ∧ -5 ≤ y ∧ y ≤ 5) ∨ 
      (y = 0 ∧ 0 ≤ x ∧ x ≤ 5))) :=
  sorry

end point_in_region_l928_92831


namespace linear_equation_solution_l928_92805

theorem linear_equation_solution :
  ∃ x : ℚ, 3 * x + 5 = 500 - (4 * x + 6 * x) ∧ x = 495 / 13 := by
  sorry

end linear_equation_solution_l928_92805


namespace number_of_arrangements_l928_92845

/-- The number of volunteers --/
def n : ℕ := 6

/-- The number of exhibition areas --/
def m : ℕ := 4

/-- The number of exhibition areas that should have one person --/
def k : ℕ := 2

/-- The number of exhibition areas that should have two people --/
def l : ℕ := 2

/-- The constraint that two specific volunteers cannot be in the same group --/
def constraint : Prop := True

/-- The function that calculates the number of arrangement plans --/
def arrangement_plans (n m k l : ℕ) (constraint : Prop) : ℕ := sorry

/-- Theorem stating that the number of arrangement plans is 156 --/
theorem number_of_arrangements :
  arrangement_plans n m k l constraint = 156 := by sorry

end number_of_arrangements_l928_92845


namespace not_all_distinct_l928_92896

/-- A sequence of non-negative rational numbers satisfying a(m) + a(n) = a(mn) -/
def NonNegativeSequence (a : ℕ → ℚ) : Prop :=
  (∀ n, a n ≥ 0) ∧ (∀ m n, a m + a n = a (m * n))

/-- The theorem stating that not all elements of the sequence can be distinct -/
theorem not_all_distinct (a : ℕ → ℚ) (h : NonNegativeSequence a) :
  ∃ i j, i ≠ j ∧ a i = a j :=
sorry

end not_all_distinct_l928_92896


namespace integer_pair_divisibility_l928_92834

theorem integer_pair_divisibility (m n : ℕ+) :
  (∃ k : ℤ, (m : ℤ) + n^2 = k * ((m : ℤ)^2 - n)) ∧
  (∃ l : ℤ, (m : ℤ)^2 + n = l * (n^2 - m)) →
  ((m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨
   (m = 2 ∧ n = 1) ∨ (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2)) :=
by sorry

end integer_pair_divisibility_l928_92834


namespace percentage_calculation_l928_92861

theorem percentage_calculation (x : ℝ) (h : 70 = 0.56 * x) : 1.25 * x = 156.25 := by
  sorry

end percentage_calculation_l928_92861


namespace always_odd_l928_92827

theorem always_odd (n : ℤ) : ∃ k : ℤ, (n + 1)^3 - n^3 = 2*k + 1 := by
  sorry

end always_odd_l928_92827


namespace floor_expression_equals_eight_l928_92891

theorem floor_expression_equals_eight :
  ⌊(2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023)⌋ = 8 := by
  sorry

end floor_expression_equals_eight_l928_92891


namespace cartesian_to_polar_equivalence_curve_transformation_l928_92892

-- Part I
theorem cartesian_to_polar_equivalence :
  let x : ℝ := -Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  let r : ℝ := 2
  let θ : ℝ := 3 * Real.pi / 4
  x = r * Real.cos θ ∧ y = r * Real.sin θ := by sorry

-- Part II
theorem curve_transformation (x y x' y' : ℝ) :
  x' = 5 * x →
  y' = 3 * y →
  (2 * x' ^ 2 + 8 * y' ^ 2 = 1) →
  (25 * x ^ 2 + 36 * y ^ 2 = 1) := by sorry

end cartesian_to_polar_equivalence_curve_transformation_l928_92892


namespace percentage_problem_l928_92879

theorem percentage_problem :
  ∃ x : ℝ, (18 : ℝ) / x = (45 : ℝ) / 100 ∧ x = 40 := by
  sorry

end percentage_problem_l928_92879


namespace savings_goal_theorem_l928_92897

/-- Calculates the amount to save per paycheck given the total savings goal,
    number of months, and number of paychecks per month. -/
def amount_per_paycheck (total_savings : ℚ) (num_months : ℕ) (paychecks_per_month : ℕ) : ℚ :=
  total_savings / (num_months * paychecks_per_month)

/-- Proves that saving $100 per paycheck for 15 months with 2 paychecks per month
    results in a total savings of $3000. -/
theorem savings_goal_theorem :
  amount_per_paycheck 3000 15 2 = 100 := by
  sorry

#eval amount_per_paycheck 3000 15 2

end savings_goal_theorem_l928_92897


namespace sequence_inequality_l928_92865

theorem sequence_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let z := Complex.mk a b
  let seq := fun (n : ℕ+) => z ^ n.val
  let a_n := fun (n : ℕ+) => (seq n).re
  let b_n := fun (n : ℕ+) => (seq n).im
  ∀ n : ℕ+, (Complex.abs (a_n (n + 1)) + Complex.abs (b_n (n + 1))) / (Complex.abs (a_n n) + Complex.abs (b_n n)) ≥ (a^2 + b^2) / (a + b) :=
by sorry


end sequence_inequality_l928_92865


namespace max_books_robert_can_buy_l928_92878

theorem max_books_robert_can_buy (book_cost : ℚ) (available_money : ℚ) : 
  book_cost = 875/100 → available_money = 250 → 
  (∃ n : ℕ, n * book_cost ≤ available_money ∧ 
    ∀ m : ℕ, m * book_cost ≤ available_money → m ≤ n) → 
  (∃ n : ℕ, n * book_cost ≤ available_money ∧ 
    ∀ m : ℕ, m * book_cost ≤ available_money → m ≤ n) ∧ n = 28 := by
  sorry

end max_books_robert_can_buy_l928_92878


namespace parabola_line_intersection_l928_92806

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line with 60° inclination passing through the focus
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define a point in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the intersection point A
def point_A (x y : ℝ) : Prop :=
  parabola x y ∧ line x y ∧ first_quadrant x y

-- The main theorem
theorem parabola_line_intersection :
  ∀ x y : ℝ, point_A x y → |((x, y) : ℝ × ℝ) - focus| = 4 :=
sorry

end parabola_line_intersection_l928_92806


namespace star_composition_l928_92835

-- Define the star operations
def star_right (y : ℝ) : ℝ := 10 - y
def star_left (y : ℝ) : ℝ := y - 10

-- State the theorem
theorem star_composition : star_left (star_right 15) = -15 := by
  sorry

end star_composition_l928_92835


namespace min_floor_sum_l928_92821

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 := by
  sorry

end min_floor_sum_l928_92821


namespace quadratic_inequality_range_l928_92836

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) → 
  (-1 < k ∧ k < 0) :=
by sorry

end quadratic_inequality_range_l928_92836


namespace opposite_numbers_equation_l928_92847

theorem opposite_numbers_equation (x : ℚ) : 
  x / 5 + (3 - 2 * x) / 2 = 0 → x = 15 / 8 := by
  sorry

end opposite_numbers_equation_l928_92847


namespace batsman_average_increase_l928_92824

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ

/-- Calculate the average runs per inning -/
def average (b : Batsman) : ℚ :=
  b.totalRuns / b.innings

/-- Calculate the increase in average -/
def averageIncrease (before after : ℚ) : ℚ :=
  after - before

theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 11) 
  (h2 : b.lastInningRuns = 90) 
  (h3 : average b = 40) :
  averageIncrease 
    (average { innings := b.innings - 1, totalRuns := b.totalRuns - b.lastInningRuns, lastInningRuns := 0 }) 
    (average b) = 5 := by
  sorry

end batsman_average_increase_l928_92824


namespace cos_negative_thirteen_pi_fourths_l928_92874

theorem cos_negative_thirteen_pi_fourths : 
  Real.cos (-13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end cos_negative_thirteen_pi_fourths_l928_92874


namespace game_probability_limit_l928_92893

/-- Represents the state of money distribution among players -/
inductive GameState
  | AllOne
  | TwoOneZero

/-- Transition probability matrix for the game -/
def transitionMatrix : Matrix GameState GameState ℝ := sorry

/-- The probability of all players having $1 after n bell rings -/
def prob_all_one (n : ℕ) : ℝ := sorry

/-- The limit of prob_all_one as n approaches infinity -/
def limit_prob_all_one : ℝ := sorry

theorem game_probability_limit :
  limit_prob_all_one = 1/4 := by sorry

end game_probability_limit_l928_92893


namespace water_depth_relationship_l928_92882

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : Real
  baseDiameter : Real
  horizontalWaterDepth : Real

/-- Calculates the water depth when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) : Real :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_relationship (tank : WaterTank) 
  (h : tank.height = 20) 
  (d : tank.baseDiameter = 5) 
  (w : tank.horizontalWaterDepth = 2) : 
  ∃ ε > 0, abs (verticalWaterDepth tank - 0.9) < ε :=
sorry

end water_depth_relationship_l928_92882


namespace binary_10101_is_21_l928_92809

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end binary_10101_is_21_l928_92809


namespace monotonic_intervals_max_value_when_a_2_l928_92828

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 4 * (a + 1) * x^3 + 6 * a * x^2 - 12

-- Theorem for the intervals of monotonic increase
theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ a → f a x < f a y) ∧
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) ∧
  (a = 1 → ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (a > 1 → (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ∧
           (∀ x y, a ≤ x ∧ x < y → f a x < f a y)) :=
sorry

-- Theorem for the maximum value when a = 2
theorem max_value_when_a_2 :
  ∀ x, f 2 x ≤ f 2 1 ∧ f 2 1 = -9 :=
sorry

end monotonic_intervals_max_value_when_a_2_l928_92828


namespace parabola_theorem_l928_92814

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define line l passing through K
def l (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the condition for A and B being on C and l
def intersectionPoints (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ l m A.1 A.2 ∧ l m B.1 B.2

-- Define the symmetry condition for A and D
def symmetricPoints (A D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = -D.2

-- Define the dot product condition
def dotProductCondition (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 8/9

-- State the theorem
theorem parabola_theorem (m : ℝ) (A B D : ℝ × ℝ) :
  intersectionPoints m A B →
  symmetricPoints A D →
  dotProductCondition A B →
  (∃ (t : ℝ), F.1 = D.1 + t * (B.1 - D.1) ∧ F.2 = D.2 + t * (B.2 - D.2)) ∧
  (∃ (M : ℝ × ℝ), M.1 = 1/9 ∧ M.2 = 0 ∧
    ∀ (x y : ℝ), (x - M.1)^2 + (y - M.2)^2 = 4/9 →
      (x - K.1)^2 + (y - K.2)^2 ≥ 4/9 ∧
      (x - B.1)^2 + (y - B.2)^2 ≥ 4/9 ∧
      (x - D.1)^2 + (y - D.2)^2 ≥ 4/9) :=
by sorry

end parabola_theorem_l928_92814


namespace point_equidistant_from_axes_l928_92807

theorem point_equidistant_from_axes (a : ℝ) : 
  (∀ (x y : ℝ), x = a - 2 ∧ y = 6 - 2*a → |x| = |y|) → 
  (a = 8/3 ∨ a = 4) :=
sorry

end point_equidistant_from_axes_l928_92807


namespace triangle_circumradius_l928_92837

theorem triangle_circumradius (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) :
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c))) = 8.5 := by
  sorry

end triangle_circumradius_l928_92837


namespace product_evaluation_l928_92818

theorem product_evaluation (a : ℤ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end product_evaluation_l928_92818


namespace show_attendance_ratio_l928_92876

/-- The ratio of attendees at the second showing to the debut show is 4 -/
theorem show_attendance_ratio : 
  ∀ (debut_attendance second_attendance ticket_price total_revenue : ℕ),
    debut_attendance = 200 →
    ticket_price = 25 →
    total_revenue = 20000 →
    second_attendance = total_revenue / ticket_price →
    second_attendance / debut_attendance = 4 := by
  sorry

end show_attendance_ratio_l928_92876


namespace face_value_from_discounts_l928_92898

/-- Face value calculation given banker's discount and true discount -/
theorem face_value_from_discounts
  (BD : ℚ) -- Banker's discount
  (TD : ℚ) -- True discount
  (h1 : BD = 42)
  (h2 : TD = 36)
  : BD = TD + (BD - TD) :=
by
  sorry

#check face_value_from_discounts

end face_value_from_discounts_l928_92898


namespace fraction_to_decimal_l928_92817

theorem fraction_to_decimal : (45 : ℚ) / (5^3) = (360 : ℚ) / 1000 := by sorry

end fraction_to_decimal_l928_92817


namespace equation_represents_two_lines_solution_set_is_axes_l928_92863

theorem equation_represents_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x * y = 0 :=
by sorry

-- The following definitions are to establish the connection
-- between the algebraic equation and its geometric interpretation

def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

theorem solution_set_is_axes :
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2} = x_axis ∪ y_axis :=
by sorry

end equation_represents_two_lines_solution_set_is_axes_l928_92863


namespace hotel_room_cost_l928_92832

theorem hotel_room_cost (original_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  original_friends = 5 →
  additional_friends = 2 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    total_cost / original_friends - total_cost / (original_friends + additional_friends) = cost_decrease ∧
    total_cost = 262.5 := by
  sorry

end hotel_room_cost_l928_92832


namespace sufficient_not_necessary_condition_l928_92819

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > b + 1 → a > b) ∧ ¬(a > b → a > b + 1) := by sorry

end sufficient_not_necessary_condition_l928_92819


namespace susans_remaining_money_is_830_02_l928_92812

/-- Calculates Susan's remaining money after expenses --/
def susans_remaining_money (swimming_earnings babysitting_earnings online_earnings_euro : ℚ)
  (exchange_rate tax_rate clothes_percent books_percent gifts_percent : ℚ) : ℚ :=
  let online_earnings_dollar := online_earnings_euro * exchange_rate
  let total_earnings := swimming_earnings + babysitting_earnings + online_earnings_dollar
  let tax_amount := online_earnings_dollar * tax_rate
  let after_tax := online_earnings_dollar - tax_amount
  let clothes_spend := total_earnings * clothes_percent
  let after_clothes := total_earnings - clothes_spend
  let books_spend := after_clothes * books_percent
  let after_books := after_clothes - books_spend
  let gifts_spend := after_books * gifts_percent
  after_books - gifts_spend

/-- Theorem stating that Susan's remaining money is $830.02 --/
theorem susans_remaining_money_is_830_02 :
  susans_remaining_money 1000 500 300 1.20 0.02 0.30 0.25 0.15 = 830.02 := by
  sorry

end susans_remaining_money_is_830_02_l928_92812


namespace division_sum_equals_two_l928_92858

theorem division_sum_equals_two : (101 : ℚ) / 101 + (99 : ℚ) / 99 = 2 := by
  sorry

end division_sum_equals_two_l928_92858


namespace equation_solution_l928_92800

theorem equation_solution : ∃ x : ℝ, (x / 2 - 1 = 3) ∧ (x = 8) := by
  sorry

end equation_solution_l928_92800


namespace flamingo_percentage_among_non_parrots_l928_92887

/-- Given the distribution of birds in a wildlife reserve, this theorem proves
    that flamingos constitute 50% of the non-parrot birds. -/
theorem flamingo_percentage_among_non_parrots :
  let total_percentage : ℝ := 100
  let flamingo_percentage : ℝ := 40
  let parrot_percentage : ℝ := 20
  let eagle_percentage : ℝ := 15
  let owl_percentage : ℝ := total_percentage - flamingo_percentage - parrot_percentage - eagle_percentage
  let non_parrot_percentage : ℝ := total_percentage - parrot_percentage
  (flamingo_percentage / non_parrot_percentage) * 100 = 50 := by
  sorry

end flamingo_percentage_among_non_parrots_l928_92887


namespace negation_of_existence_negation_of_quadratic_inequality_l928_92889

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x, f x) ↔ ∀ x, ¬ f x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l928_92889


namespace B_subset_A_l928_92899

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 2 ∧ p.1 + 2 * p.2 = 6}

-- Theorem statement
theorem B_subset_A : B ⊆ A := by
  sorry

end B_subset_A_l928_92899


namespace max_xy_value_l928_92885

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

end max_xy_value_l928_92885


namespace three_arcs_must_intersect_l928_92873

/-- Represents a great circle arc on a sphere --/
structure GreatCircleArc where
  length : ℝ
  start_point : Sphere
  end_point : Sphere

/-- Defines a sphere --/
class Sphere where
  center : Point
  radius : ℝ

/-- Checks if two great circle arcs intersect or share an endpoint --/
def arcs_intersect (arc1 arc2 : GreatCircleArc) : Prop :=
  sorry

/-- Theorem: It's impossible to place three 300° great circle arcs on a sphere without intersections --/
theorem three_arcs_must_intersect (s : Sphere) :
  ∀ (arc1 arc2 arc3 : GreatCircleArc),
    arc1.length = 300 ∧ arc2.length = 300 ∧ arc3.length = 300 →
    arcs_intersect arc1 arc2 ∨ arcs_intersect arc2 arc3 ∨ arcs_intersect arc1 arc3 :=
by
  sorry

end three_arcs_must_intersect_l928_92873


namespace correct_num_episodes_l928_92894

/-- The number of episodes in a TV mini series -/
def num_episodes : ℕ := 6

/-- The length of each episode in minutes -/
def episode_length : ℕ := 50

/-- The total watching time in hours -/
def total_watching_time : ℕ := 5

/-- Theorem stating that the number of episodes is correct -/
theorem correct_num_episodes :
  num_episodes * episode_length = total_watching_time * 60 := by
  sorry

end correct_num_episodes_l928_92894


namespace jessie_weight_loss_l928_92823

/-- Jessie's weight loss journey -/
theorem jessie_weight_loss (initial_weight lost_weight : ℕ) 
  (h1 : initial_weight = 192)
  (h2 : lost_weight = 126) :
  initial_weight - lost_weight = 66 :=
by sorry

end jessie_weight_loss_l928_92823


namespace cube_surface_area_l928_92864

def edge_length : ℝ := 7

def surface_area_of_cube (edge : ℝ) : ℝ := 6 * edge^2

theorem cube_surface_area : 
  surface_area_of_cube edge_length = 294 := by sorry

end cube_surface_area_l928_92864


namespace circular_well_diameter_l928_92815

/-- Proves that a circular well with given depth and volume has a specific diameter -/
theorem circular_well_diameter 
  (depth : ℝ) 
  (volume : ℝ) 
  (h_depth : depth = 8) 
  (h_volume : volume = 25.132741228718345) : 
  2 * (volume / (Real.pi * depth))^(1/2 : ℝ) = 2 := by
  sorry

end circular_well_diameter_l928_92815


namespace complement_union_A_B_l928_92848

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 2, 4} := by sorry

end complement_union_A_B_l928_92848


namespace pipeline_construction_equation_l928_92890

theorem pipeline_construction_equation 
  (total_length : ℝ) 
  (efficiency_increase : ℝ) 
  (days_ahead : ℝ) 
  (x : ℝ) 
  (h1 : total_length = 3000)
  (h2 : efficiency_increase = 0.2)
  (h3 : days_ahead = 10)
  (h4 : x > 0) :
  total_length / x - total_length / ((1 + efficiency_increase) * x) = days_ahead :=
sorry

end pipeline_construction_equation_l928_92890


namespace intersection_distance_product_l928_92851

/-- Parabola defined by y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Line with equation y = x - 2 -/
def line (x y : ℝ) : Prop := y = x - 2

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- Theorem stating that the product of distances from focus to intersection points is 32 -/
theorem intersection_distance_product : 
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    (A.1 - focus.1)^2 + (A.2 - focus.2)^2 * 
    (B.1 - focus.1)^2 + (B.2 - focus.2)^2 = 32^2 :=
sorry

end intersection_distance_product_l928_92851


namespace perfect_33rd_power_l928_92841

theorem perfect_33rd_power (x y : ℕ+) (h : ∃ k : ℕ+, (x * y^10 : ℕ) = k^33) :
  ∃ m : ℕ+, (x^10 * y : ℕ) = m^33 := by
  sorry

end perfect_33rd_power_l928_92841


namespace star_distance_l928_92830

/-- The distance between a star and Earth given the speed of light and time taken for light to reach Earth -/
theorem star_distance (c : ℝ) (t : ℝ) (y : ℝ) (h1 : c = 3 * 10^5) (h2 : t = 10) (h3 : y = 3.1 * 10^7) :
  c * (t * y) = 9.3 * 10^13 := by
  sorry

end star_distance_l928_92830


namespace park_length_l928_92872

/-- A rectangular park with given dimensions and tree density. -/
structure Park where
  width : ℝ
  length : ℝ
  treeCount : ℕ
  treeDensity : ℝ

/-- The park satisfies the given conditions. -/
def validPark (p : Park) : Prop :=
  p.width = 2000 ∧
  p.treeCount = 100000 ∧
  p.treeDensity = 1 / 20

/-- The theorem stating the length of the park given the conditions. -/
theorem park_length (p : Park) (h : validPark p) : p.length = 1000 := by
  sorry

#check park_length

end park_length_l928_92872


namespace length_breadth_difference_l928_92867

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  perimeter : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ

/-- Theorem stating the difference between length and breadth of the plot -/
theorem length_breadth_difference (plot : RectangularPlot)
  (h1 : plot.length = 61)
  (h2 : plot.perimeter * plot.fencing_rate = plot.fencing_cost)
  (h3 : plot.fencing_rate = 26.50)
  (h4 : plot.fencing_cost = 5300)
  (h5 : plot.perimeter = 2 * (plot.length + plot.breadth))
  (h6 : plot.length > plot.breadth) :
  plot.length - plot.breadth = 22 := by
  sorry

end length_breadth_difference_l928_92867


namespace f_composition_eq_inverse_e_l928_92881

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_eq_inverse_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_composition_eq_inverse_e_l928_92881


namespace fraction_change_l928_92838

theorem fraction_change (x : ℚ) : 
  (1.2 * (5 : ℚ)) / (7 * (1 - x / 100)) = 20 / 21 → x = 10 := by
  sorry

end fraction_change_l928_92838


namespace airway_graph_diameter_at_most_two_l928_92804

/-- A simple graph with 20 vertices and 172 edges -/
structure AirwayGraph where
  V : Finset (Fin 20)
  E : Finset (Fin 20 × Fin 20)
  edge_count : E.card = 172
  simple : ∀ (e : Fin 20 × Fin 20), e ∈ E → e.1 ≠ e.2
  undirected : ∀ (e : Fin 20 × Fin 20), e ∈ E → (e.2, e.1) ∈ E
  at_most_one : ∀ (u v : Fin 20), u ≠ v → ({(u, v), (v, u)} ∩ E).card ≤ 1

/-- The diameter of an AirwayGraph is at most 2 -/
theorem airway_graph_diameter_at_most_two (G : AirwayGraph) :
  ∀ (u v : Fin 20), u ≠ v → ∃ (w : Fin 20), (u = w ∨ (u, w) ∈ G.E) ∧ (w = v ∨ (w, v) ∈ G.E) :=
sorry

end airway_graph_diameter_at_most_two_l928_92804


namespace investment_result_l928_92801

/-- Calculates the final amount after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the given investment scenario results in approximately $3045.28 --/
theorem investment_result :
  let principal : ℝ := 1500
  let rate : ℝ := 0.04
  let time : ℕ := 21
  let result := compound_interest principal rate time
  ∃ ε > 0, |result - 3045.28| < ε :=
sorry

end investment_result_l928_92801


namespace nancy_coffee_expenditure_l928_92826

/-- Represents Nancy's coffee consumption and expenditure over a period of time. -/
structure CoffeeConsumption where
  morning_price : ℝ
  afternoon_price : ℝ
  days : ℕ

/-- Calculates the total expenditure on coffee given Nancy's consumption pattern. -/
def total_expenditure (c : CoffeeConsumption) : ℝ :=
  c.days * (c.morning_price + c.afternoon_price)

/-- Theorem stating that Nancy's total expenditure on coffee over 20 days is $110.00. -/
theorem nancy_coffee_expenditure :
  let c : CoffeeConsumption := {
    morning_price := 3.00,
    afternoon_price := 2.50,
    days := 20
  }
  total_expenditure c = 110.00 := by
  sorry

end nancy_coffee_expenditure_l928_92826


namespace wintersweet_bouquet_solution_l928_92840

/-- Represents the number of branches in a bouquet --/
structure BouquetComposition where
  typeA : ℕ
  typeB : ℕ

/-- Represents the total number of branches available --/
structure TotalBranches where
  typeA : ℕ
  typeB : ℕ

/-- Represents the number of bouquets of each type --/
structure BouquetCounts where
  alpha : ℕ
  beta : ℕ

def totalBranches : TotalBranches := { typeA := 142, typeB := 104 }

def alphaBouquet : BouquetComposition := { typeA := 6, typeB := 4 }
def betaBouquet : BouquetComposition := { typeA := 5, typeB := 4 }

/-- The theorem states that given the total branches and bouquet compositions,
    the solution of 12 Alpha bouquets and 14 Beta bouquets is correct --/
theorem wintersweet_bouquet_solution :
  ∃ (solution : BouquetCounts),
    solution.alpha = 12 ∧
    solution.beta = 14 ∧
    solution.alpha * alphaBouquet.typeA + solution.beta * betaBouquet.typeA = totalBranches.typeA ∧
    solution.alpha * alphaBouquet.typeB + solution.beta * betaBouquet.typeB = totalBranches.typeB :=
by sorry

end wintersweet_bouquet_solution_l928_92840


namespace solution_set_inequality_l928_92816

theorem solution_set_inequality (x : ℝ) : 
  Set.Icc 1 2 = {x | (x - 1) * (x - 2) ≤ 0} := by sorry

end solution_set_inequality_l928_92816


namespace base7_to_base10_conversion_l928_92843

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number -/
def base7Number : List Nat := [6, 4, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 132 := by
  sorry

end base7_to_base10_conversion_l928_92843


namespace nabla_computation_l928_92855

def nabla (a b : ℕ) : ℕ := 3 + a^b

theorem nabla_computation : nabla (nabla 3 2) 1 = 15 := by
  sorry

end nabla_computation_l928_92855


namespace triangle_count_l928_92825

theorem triangle_count : ∃! (n : ℕ), n = 59 ∧
  (∀ (a b c : ℕ), (a < b ∧ b < c) →
    (b = 60) →
    (c - b = b - a) →
    (a + b + c = 180) →
    (0 < a ∧ a < b ∧ b < c) →
    (∃ (d : ℕ), a = 60 - d ∧ c = 60 + d ∧ 0 < d ∧ d < 60)) :=
by sorry

end triangle_count_l928_92825


namespace lucy_fraction_of_edna_l928_92868

-- Define the field length
def field_length : ℚ := 24

-- Define Mary's distance as a fraction of the field length
def mary_distance : ℚ := 3/8 * field_length

-- Define Edna's distance as a fraction of Mary's distance
def edna_distance : ℚ := 2/3 * mary_distance

-- Define Lucy's distance as mary_distance - 4
def lucy_distance : ℚ := mary_distance - 4

-- Theorem to prove
theorem lucy_fraction_of_edna : lucy_distance / edna_distance = 5/6 := by
  sorry

end lucy_fraction_of_edna_l928_92868


namespace fraction_meaningful_range_l928_92822

theorem fraction_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x - 2)) ↔ x ≠ 2 := by
  sorry

end fraction_meaningful_range_l928_92822


namespace fraction_value_l928_92857

theorem fraction_value (m n : ℝ) (h : |m - 1/4| + (n + 3)^2 = 0) : n / m = -12 := by
  sorry

end fraction_value_l928_92857


namespace sufficient_conditions_for_x_squared_less_than_one_l928_92870

theorem sufficient_conditions_for_x_squared_less_than_one :
  (∀ x : ℝ, (0 < x ∧ x < 1) → x^2 < 1) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 0) → x^2 < 1) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 1) → x^2 < 1) ∧
  (∃ x : ℝ, x < 1 ∧ ¬(x^2 < 1)) :=
by sorry

end sufficient_conditions_for_x_squared_less_than_one_l928_92870


namespace ten_markers_five_friends_l928_92860

/-- The number of ways to distribute n identical markers among k friends,
    where each friend must have at least one marker -/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical markers among 5 friends,
    where each friend must have at least one marker, is 126 -/
theorem ten_markers_five_friends :
  distributionWays 10 5 = 126 := by sorry

end ten_markers_five_friends_l928_92860


namespace jimin_yuna_problem_l928_92859

/-- Given a line of students ordered by height, calculate the number of students between two specific positions. -/
def students_between (total : ℕ) (pos1 : ℕ) (pos2 : ℕ) : ℕ :=
  if pos1 > pos2 then pos1 - pos2 - 1 else pos2 - pos1 - 1

theorem jimin_yuna_problem :
  let total_students : ℕ := 32
  let jimin_position : ℕ := 27
  let yuna_position : ℕ := 11
  students_between total_students jimin_position yuna_position = 15 := by
  sorry

end jimin_yuna_problem_l928_92859


namespace value_of_y_l928_92853

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end value_of_y_l928_92853


namespace neg_p_sufficient_not_necessary_for_neg_q_l928_92808

-- Define the conditions
def p (x : ℝ) := x^2 - 1 > 0
def q (x : ℝ) := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x) := by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l928_92808


namespace airplane_seats_theorem_l928_92856

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 300

/-- Represents the number of First Class seats -/
def first_class_seats : ℕ := 30

/-- Represents the percentage of Business Class seats -/
def business_class_percentage : ℚ := 20 / 100

/-- Represents the percentage of Economy Class seats -/
def economy_class_percentage : ℚ := 70 / 100

/-- Theorem stating that the total number of seats is 300 -/
theorem airplane_seats_theorem :
  (first_class_seats : ℚ) +
  (business_class_percentage * total_seats) +
  (economy_class_percentage * total_seats) =
  total_seats :=
sorry

end airplane_seats_theorem_l928_92856
