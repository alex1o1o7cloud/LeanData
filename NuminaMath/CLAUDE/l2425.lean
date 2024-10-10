import Mathlib

namespace arithmetic_calculation_l2425_242555

theorem arithmetic_calculation : -12 * 5 - (-8 * -4) + (-15 * -6) = -2 := by
  sorry

end arithmetic_calculation_l2425_242555


namespace integer_root_values_l2425_242530

def polynomial (x b : ℤ) : ℤ := x^4 + 4*x^3 + 2*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-34, -19, -10, -9, -3, 2, 4, 6, 8, 11} :=
sorry

end integer_root_values_l2425_242530


namespace jasons_books_count_l2425_242546

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 45

/-- The number of shelves Jason needs -/
def shelves_needed : ℕ := 7

/-- The total number of books Jason has -/
def total_books : ℕ := books_per_shelf * shelves_needed

theorem jasons_books_count : total_books = 315 := by
  sorry

end jasons_books_count_l2425_242546


namespace sixth_roll_sum_l2425_242586

/-- Represents the sum of numbers on a single die over 6 rolls -/
def single_die_sum : ℕ := 21

/-- Represents the number of dice -/
def num_dice : ℕ := 6

/-- Represents the sums of the top faces for the first 5 rolls -/
def first_five_rolls : List ℕ := [21, 19, 20, 18, 25]

/-- Theorem: The sum of the top faces on the 6th roll is 23 -/
theorem sixth_roll_sum :
  (num_dice * single_die_sum) - (first_five_rolls.sum) = 23 := by
  sorry

end sixth_roll_sum_l2425_242586


namespace range_of_a_l2425_242545

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (1 ∉ A a) → a < 1 := by
  sorry

end range_of_a_l2425_242545


namespace sidney_thursday_jumping_jacks_l2425_242544

/-- The number of jumping jacks Sidney did on Thursday -/
def thursday_jumping_jacks : ℕ := by sorry

/-- The total number of jumping jacks Sidney did from Monday to Wednesday -/
def monday_to_wednesday : ℕ := 20 + 36 + 40

/-- The total number of jumping jacks Brooke did -/
def brooke_total : ℕ := 438

theorem sidney_thursday_jumping_jacks :
  thursday_jumping_jacks = 50 :=
by
  have sidney_total : ℕ := brooke_total / 3
  have h1 : sidney_total = monday_to_wednesday + thursday_jumping_jacks := by sorry
  sorry


end sidney_thursday_jumping_jacks_l2425_242544


namespace max_leap_years_in_200_years_l2425_242573

/-- 
In a calendrical system where leap years occur every three years without exception,
the maximum number of leap years in a 200-year period is 66.
-/
theorem max_leap_years_in_200_years : 
  ∀ (leap_year_count : ℕ → ℕ),
  (∀ n : ℕ, leap_year_count (3 * n) = n) →
  leap_year_count 200 = 66 := by
sorry

end max_leap_years_in_200_years_l2425_242573


namespace polynomial_division_remainder_l2425_242521

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^2 - 22 * X + 63 = (X - 3) * q + 24 := by sorry

end polynomial_division_remainder_l2425_242521


namespace inverse_proportion_problem_l2425_242576

-- Define the inverse proportionality relationship
def inverse_proportional (x y : ℝ) := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Theorem statement
theorem inverse_proportion_problem (x y : ℝ → ℝ) :
  (∀ a b : ℝ, inverse_proportional (x a) (y a)) →
  x 2 = 4 →
  x (-3) = -8/3 ∧ x 6 = 4/3 :=
by sorry

end inverse_proportion_problem_l2425_242576


namespace sum_g_formula_l2425_242548

/-- g(n) is the largest odd divisor of the positive integer n -/
def g (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of g(k) for k from 1 to 2^n -/
def sum_g (n : ℕ) : ℕ+ :=
  sorry

/-- Theorem: The sum of g(k) for k from 1 to 2^n equals (4^n + 5) / 3 -/
theorem sum_g_formula (n : ℕ) : 
  (sum_g n : ℚ) = (4^n + 5) / 3 := by
  sorry

end sum_g_formula_l2425_242548


namespace quadratic_roots_sum_l2425_242568

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + a - 2024 = 0) → (b^2 + b - 2024 = 0) → (a^2 + 2*a + b = 2023) := by
  sorry

end quadratic_roots_sum_l2425_242568


namespace fraction_equality_l2425_242535

theorem fraction_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 := by
  sorry

end fraction_equality_l2425_242535


namespace parallel_vectors_k_value_l2425_242574

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : (ℝ × ℝ)) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

theorem parallel_vectors_k_value
  (a b : ℝ × ℝ)  -- a and b are plane vectors
  (h_not_collinear : ¬ are_parallel a b)  -- a and b are non-collinear
  (m : ℝ × ℝ)
  (h_m : m = (a.1 - 2 * b.1, a.2 - 2 * b.2))  -- m = a - 2b
  (k : ℝ)
  (n : ℝ × ℝ)
  (h_n : n = (3 * a.1 + k * b.1, 3 * a.2 + k * b.2))  -- n = 3a + kb
  (h_parallel : are_parallel m n)  -- m is parallel to n
  : k = -6 := by
  sorry

end parallel_vectors_k_value_l2425_242574


namespace complex_simplification_l2425_242528

/-- Proof of complex number simplification -/
theorem complex_simplification :
  let i : ℂ := Complex.I
  (3 + 5*i) / (-2 + 7*i) = 29/53 - (31/53)*i :=
by sorry

end complex_simplification_l2425_242528


namespace equation_solution_l2425_242501

theorem equation_solution (x : ℝ) : (24 / 36 : ℝ) = Real.sqrt (x / 36) → x = 16 := by
  sorry

end equation_solution_l2425_242501


namespace equation_transformation_l2425_242500

theorem equation_transformation (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) + 3 = 3 * x / (1 - x) → 1 + 3 * (x - 1) = -3 * x := by
  sorry

end equation_transformation_l2425_242500


namespace simplify_expression_l2425_242569

theorem simplify_expression (x : ℝ) : 2*x*(4*x^2 - 3) - 4*(x^2 - 3*x + 8) = 8*x^3 - 4*x^2 + 6*x - 32 := by
  sorry

end simplify_expression_l2425_242569


namespace tangent_curve_intersection_l2425_242516

-- Define the curve C
def C (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 - 9 * x^2 + 4

-- Define the point M
def M : ℝ × ℝ := (1, -4)

-- Define the tangent line l
def l (x : ℝ) : ℝ := -12 * (x - 1) - 4

-- Define a function to count common points
def count_common_points (f g : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem tangent_curve_intersection :
  count_common_points C l = 3 :=
sorry

end tangent_curve_intersection_l2425_242516


namespace lasagna_cost_l2425_242533

def cheese_quantity : Real := 1.5
def meat_quantity : Real := 0.550
def pasta_quantity : Real := 0.280
def tomatoes_quantity : Real := 2.2

def cheese_price : Real := 6.30
def meat_price : Real := 8.55
def pasta_price : Real := 2.40
def tomatoes_price : Real := 1.79

def cheese_tax : Real := 0.07
def meat_tax : Real := 0.06
def pasta_tax : Real := 0.08
def tomatoes_tax : Real := 0.05

def total_cost (cq mq pq tq : Real) (cp mp pp tp : Real) (ct mt pt tt : Real) : Real :=
  (cq * cp * (1 + ct)) + (mq * mp * (1 + mt)) + (pq * pp * (1 + pt)) + (tq * tp * (1 + tt))

theorem lasagna_cost :
  total_cost cheese_quantity meat_quantity pasta_quantity tomatoes_quantity
              cheese_price meat_price pasta_price tomatoes_price
              cheese_tax meat_tax pasta_tax tomatoes_tax = 19.9568 := by
  sorry

end lasagna_cost_l2425_242533


namespace james_diet_result_l2425_242508

/-- Represents James' food intake and exercise routine --/
structure JamesDiet where
  cheezitBags : ℕ
  cheezitOuncesPerBag : ℕ
  cheezitCaloriesPerOunce : ℕ
  chocolateBars : ℕ
  chocolateBarCalories : ℕ
  popcornCalories : ℕ
  runningMinutes : ℕ
  runningCaloriesPerMinute : ℕ
  swimmingMinutes : ℕ
  swimmingCaloriesPerMinute : ℕ
  cyclingMinutes : ℕ
  cyclingCaloriesPerMinute : ℕ
  caloriesPerPound : ℕ

/-- Calculates the total calories consumed --/
def totalCaloriesConsumed (d : JamesDiet) : ℕ :=
  d.cheezitBags * d.cheezitOuncesPerBag * d.cheezitCaloriesPerOunce +
  d.chocolateBars * d.chocolateBarCalories +
  d.popcornCalories

/-- Calculates the total calories burned --/
def totalCaloriesBurned (d : JamesDiet) : ℕ :=
  d.runningMinutes * d.runningCaloriesPerMinute +
  d.swimmingMinutes * d.swimmingCaloriesPerMinute +
  d.cyclingMinutes * d.cyclingCaloriesPerMinute

/-- Calculates the excess calories --/
def excessCalories (d : JamesDiet) : ℤ :=
  (totalCaloriesConsumed d : ℤ) - (totalCaloriesBurned d : ℤ)

/-- Calculates the potential weight gain in pounds --/
def potentialWeightGain (d : JamesDiet) : ℚ :=
  (excessCalories d : ℚ) / d.caloriesPerPound

/-- Theorem stating James' excess calorie consumption and potential weight gain --/
theorem james_diet_result (d : JamesDiet) 
  (h1 : d.cheezitBags = 3)
  (h2 : d.cheezitOuncesPerBag = 2)
  (h3 : d.cheezitCaloriesPerOunce = 150)
  (h4 : d.chocolateBars = 2)
  (h5 : d.chocolateBarCalories = 250)
  (h6 : d.popcornCalories = 500)
  (h7 : d.runningMinutes = 40)
  (h8 : d.runningCaloriesPerMinute = 12)
  (h9 : d.swimmingMinutes = 30)
  (h10 : d.swimmingCaloriesPerMinute = 15)
  (h11 : d.cyclingMinutes = 20)
  (h12 : d.cyclingCaloriesPerMinute = 10)
  (h13 : d.caloriesPerPound = 3500) :
  excessCalories d = 770 ∧ potentialWeightGain d = 11/50 := by
  sorry

end james_diet_result_l2425_242508


namespace average_equation_l2425_242571

theorem average_equation (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 4) + (4*x + 6) + (5*x + 3)) = 3*x + 5 → x = 1 := by
  sorry

end average_equation_l2425_242571


namespace min_distance_line_curve_l2425_242596

/-- The minimum distance between a point on the line y = 1 - x and a point on the curve y = -e^x is √2 -/
theorem min_distance_line_curve : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = 1 - P.1) → 
    (Q.2 = -Real.exp Q.1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
sorry

end min_distance_line_curve_l2425_242596


namespace ratio_calculation_l2425_242572

theorem ratio_calculation (A B C : ℚ) (h : A/B = 3/2 ∧ B/C = 2/5) :
  (4*A + 3*B) / (5*C - 2*B) = 15/23 := by
  sorry

end ratio_calculation_l2425_242572


namespace boxed_flowers_cost_l2425_242589

theorem boxed_flowers_cost (first_batch_total : ℕ) (second_batch_total : ℕ) 
  (second_batch_multiplier : ℕ) (price_difference : ℕ) :
  first_batch_total = 2000 →
  second_batch_total = 4200 →
  second_batch_multiplier = 3 →
  price_difference = 6 →
  ∃ (x : ℕ), 
    x * first_batch_total = second_batch_multiplier * second_batch_total * (x - price_difference) ∧
    x = 20 :=
by sorry

end boxed_flowers_cost_l2425_242589


namespace smurf_score_difference_l2425_242522

/-- The number of Smurfs in the village -/
def total_smurfs : ℕ := 45

/-- The number of top and bottom Smurfs with known average scores -/
def known_scores_count : ℕ := 25

/-- The average score of the top 25 Smurfs -/
def top_average : ℚ := 93

/-- The average score of the bottom 25 Smurfs -/
def bottom_average : ℚ := 89

/-- The number of top and bottom Smurfs we're comparing -/
def comparison_count : ℕ := 20

/-- The theorem stating the difference between top and bottom scores -/
theorem smurf_score_difference :
  (top_average * known_scores_count - bottom_average * known_scores_count : ℚ) = 100 := by
  sorry

end smurf_score_difference_l2425_242522


namespace board_transformation_impossibility_l2425_242504

/-- Represents a board state as a list of integers -/
def Board := List Int

/-- Performs one move on the board, pairing integers and replacing with their sum and difference -/
def move (b : Board) : Board :=
  sorry

/-- Checks if a board contains 1000 consecutive integers -/
def isConsecutive1000 (b : Board) : Prop :=
  sorry

/-- Calculates the sum of squares of all integers on the board -/
def sumOfSquares (b : Board) : Int :=
  sorry

theorem board_transformation_impossibility (initial : Board) :
  (sumOfSquares initial) % 8 = 0 →
  ∀ n : Nat, ¬(isConsecutive1000 (n.iterate move initial)) :=
sorry

end board_transformation_impossibility_l2425_242504


namespace direct_proportion_k_value_l2425_242587

def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x

theorem direct_proportion_k_value :
  ∀ k : ℝ,
  (∀ x y : ℝ, y = (k - 1) * x + k^2 - 1) →
  (is_direct_proportion (λ x => (k - 1) * x + k^2 - 1)) →
  k = -1 :=
by sorry

end direct_proportion_k_value_l2425_242587


namespace even_coverings_for_odd_height_l2425_242580

/-- Represents a covering of the lateral surface of a rectangular parallelepiped -/
def Covering (a b c : ℕ) := Unit

/-- Count the number of valid coverings for a rectangular parallelepiped -/
def countCoverings (a b c : ℕ) : ℕ := sorry

/-- Theorem: The number of valid coverings is even when the height is odd -/
theorem even_coverings_for_odd_height (a b c : ℕ) (h : c % 2 = 1) :
  ∃ k : ℕ, countCoverings a b c = 2 * k := by sorry

end even_coverings_for_odd_height_l2425_242580


namespace puppy_cost_puppy_cost_proof_l2425_242594

/-- The cost of a puppy in a pet shop, given the following conditions:
  * There are 2 puppies and 4 kittens in the pet shop.
  * A kitten costs $15.
  * The total stock is worth $100. -/
theorem puppy_cost : ℕ :=
  let num_puppies : ℕ := 2
  let num_kittens : ℕ := 4
  let kitten_cost : ℕ := 15
  let total_stock_value : ℕ := 100
  20

/-- Proof that the cost of a puppy is $20. -/
theorem puppy_cost_proof : puppy_cost = 20 := by
  sorry

end puppy_cost_puppy_cost_proof_l2425_242594


namespace hyperbola_eccentricity_l2425_242515

/-- 
Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
if the distance from a vertex to one of its asymptotes is b/2,
then its eccentricity is 2.
-/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_distance : (b * a) / Real.sqrt (a^2 + b^2) = b / 2) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 := by
  sorry

end hyperbola_eccentricity_l2425_242515


namespace distance_between_points_l2425_242510

/-- The distance between the points (3, -2) and (10, 8) is √149 units. -/
theorem distance_between_points : Real.sqrt 149 = Real.sqrt ((10 - 3)^2 + (8 - (-2))^2) := by
  sorry

end distance_between_points_l2425_242510


namespace unique_solution_cube_equation_l2425_242592

theorem unique_solution_cube_equation :
  ∃! (x : ℕ+), (2 * x.val)^3 - x.val = 726 := by
  sorry

end unique_solution_cube_equation_l2425_242592


namespace increasing_magnitude_l2425_242512

-- Define the variables and conditions
theorem increasing_magnitude (a : ℝ) 
  (h1 : 0.8 < a) (h2 : a < 0.9)
  (y : ℝ) (hy : y = a^a)
  (z : ℝ) (hz : z = a^(a^a))
  (w : ℝ) (hw : w = a^(Real.log a)) :
  a < z ∧ z < y ∧ y < w := by sorry

end increasing_magnitude_l2425_242512


namespace regular_polygon_with_18_degree_exterior_angles_has_20_sides_l2425_242520

/-- Theorem: A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_with_18_degree_exterior_angles_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by
  sorry

end regular_polygon_with_18_degree_exterior_angles_has_20_sides_l2425_242520


namespace zeros_in_5000_to_50_l2425_242541

theorem zeros_in_5000_to_50 : ∃ n : ℕ, (5000 ^ 50 : ℕ) = n * (10 ^ 150) ∧ n % 10 ≠ 0 := by
  sorry

end zeros_in_5000_to_50_l2425_242541


namespace least_integer_for_triangle_with_integer_area_l2425_242595

theorem least_integer_for_triangle_with_integer_area : 
  ∃ (a : ℕ), a > 14 ∧ 
  (∀ b : ℕ, b > 14 ∧ b < a → 
    ¬(∃ A : ℕ, A^2 = (3*b^2/4) * ((b^2/4) - 1))) ∧
  (∃ A : ℕ, A^2 = (3*a^2/4) * ((a^2/4) - 1)) ∧
  a = 52 := by
sorry

end least_integer_for_triangle_with_integer_area_l2425_242595


namespace inscribed_sphere_volume_l2425_242529

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (base_diameter : ℝ) (h : base_diameter = 12 * Real.sqrt 3) :
  let cone_height : ℝ := base_diameter / 2
  let sphere_radius : ℝ := 3 * Real.sqrt 6 - 3 * Real.sqrt 3
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (4 / 3) * Real.pi * (3 * Real.sqrt 6 - 3 * Real.sqrt 3) ^ 3 :=
by sorry

end inscribed_sphere_volume_l2425_242529


namespace points_collinear_l2425_242588

/-- Three points A, B, and C in the plane are collinear if there exists a real number k such that 
    vector AC = k * vector AB. -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (C.1 - A.1, C.2 - A.2) = (k * (B.1 - A.1), k * (B.2 - A.2))

/-- The points A(-1, -2), B(2, -1), and C(8, 1) are collinear. -/
theorem points_collinear : collinear (-1, -2) (2, -1) (8, 1) := by
  sorry


end points_collinear_l2425_242588


namespace exists_four_digit_divisible_by_23_digit_sum_23_l2425_242505

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Proposition: There exists a four-digit number divisible by 23 with digit sum 23 -/
theorem exists_four_digit_divisible_by_23_digit_sum_23 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ digit_sum n = 23 ∧ n % 23 = 0 := by
sorry

end exists_four_digit_divisible_by_23_digit_sum_23_l2425_242505


namespace unique_I_value_l2425_242561

def addition_problem (E I G T W O : Nat) : Prop :=
  E ≠ I ∧ E ≠ G ∧ E ≠ T ∧ E ≠ W ∧ E ≠ O ∧
  I ≠ G ∧ I ≠ T ∧ I ≠ W ∧ I ≠ O ∧
  G ≠ T ∧ G ≠ W ∧ G ≠ O ∧
  T ≠ W ∧ T ≠ O ∧
  W ≠ O ∧
  E < 10 ∧ I < 10 ∧ G < 10 ∧ T < 10 ∧ W < 10 ∧ O < 10 ∧
  E = 4 ∧
  G % 2 = 1 ∧
  100 * T + 10 * W + O = 100 * E + 10 * I + G + 100 * E + 10 * I + G

theorem unique_I_value :
  ∀ E I G T W O : Nat,
    addition_problem E I G T W O →
    I = 2 :=
by sorry

end unique_I_value_l2425_242561


namespace fixed_point_of_exponential_function_l2425_242558

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a^(x-1) + 4
  f 1 = 5 := by
sorry

end fixed_point_of_exponential_function_l2425_242558


namespace kid_ticket_price_l2425_242523

theorem kid_ticket_price 
  (total_sales : ℕ) 
  (adult_price : ℕ) 
  (num_adults : ℕ) 
  (total_people : ℕ) : 
  total_sales = 3864 ∧ 
  adult_price = 28 ∧ 
  num_adults = 51 ∧ 
  total_people = 254 → 
  (total_sales - num_adults * adult_price) / (total_people - num_adults) = 12 := by
  sorry

#eval (3864 - 51 * 28) / (254 - 51)

end kid_ticket_price_l2425_242523


namespace complex_equation_solution_l2425_242503

theorem complex_equation_solution (z : ℂ) : z * Complex.I = Complex.I - 1 → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l2425_242503


namespace fruit_box_theorem_l2425_242566

theorem fruit_box_theorem (total_fruits : ℕ) 
  (h_total : total_fruits = 56)
  (h_oranges : total_fruits / 4 = total_fruits / 4)  -- One-fourth are oranges
  (h_peaches : total_fruits / 8 = total_fruits / 8)  -- Half as many peaches as oranges
  (h_apples : 5 * (total_fruits / 8) = 5 * (total_fruits / 8))  -- Five times as many apples as peaches
  (h_mixed : total_fruits / 4 = total_fruits / 4)  -- Twice as many mixed fruits as peaches
  : (5 * (total_fruits / 8) = 35) ∧ 
    (total_fruits / 4 : ℚ) / total_fruits = 1 / 4 := by
  sorry

end fruit_box_theorem_l2425_242566


namespace algae_growth_l2425_242506

/-- Calculates the population of algae after a given time period. -/
def algaePopulation (initialPopulation : ℕ) (minutes : ℕ) : ℕ :=
  initialPopulation * 2^(minutes / 5)

/-- Theorem stating that the algae population grows from 50 to 6400 in 35 minutes. -/
theorem algae_growth :
  algaePopulation 50 35 = 6400 :=
by
  sorry

#eval algaePopulation 50 35

end algae_growth_l2425_242506


namespace sakshi_work_duration_l2425_242525

-- Define the efficiency ratio between Tanya and Sakshi
def efficiency_ratio : ℝ := 1.25

-- Define Tanya's work duration in days
def tanya_days : ℝ := 4

-- Theorem stating that Sakshi takes 5 days to complete the work
theorem sakshi_work_duration :
  efficiency_ratio * tanya_days = 5 := by
  sorry

end sakshi_work_duration_l2425_242525


namespace hawks_score_l2425_242514

theorem hawks_score (total_points margin : ℕ) (h1 : total_points = 50) (h2 : margin = 18) :
  (total_points - margin) / 2 = 16 := by
  sorry

end hawks_score_l2425_242514


namespace floor_tiles_proof_l2425_242591

/-- Represents the number of tiles in a row given its position -/
def tiles_in_row (n : ℕ) : ℕ := 53 - 2 * (n - 1)

/-- Represents the total number of tiles in the first n rows -/
def total_tiles (n : ℕ) : ℕ := n * (tiles_in_row 1 + tiles_in_row n) / 2

/-- The number of rows in the floor -/
def num_rows : ℕ := 9

theorem floor_tiles_proof :
  (total_tiles num_rows = 405) ∧
  (∀ i : ℕ, i > 0 → i ≤ num_rows → tiles_in_row i > 0) :=
sorry

#eval num_rows

end floor_tiles_proof_l2425_242591


namespace expression_simplification_l2425_242549

theorem expression_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 + b^2 ≠ 0) :
  (1 / (a - b) - (2 * a * b) / (a^3 - a^2 * b + a * b^2 - b^3)) /
  ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + b / (a^2 + b^2)) =
  (a - b) / (a + b) :=
sorry

end expression_simplification_l2425_242549


namespace valid_numbers_l2425_242556

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    (1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a) / 143 = 136

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {9949, 9859, 9769, 9679, 9589, 9499} :=
by sorry

end valid_numbers_l2425_242556


namespace diana_wins_prob_l2425_242590

/-- Represents the number of sides on Diana's die -/
def diana_sides : ℕ := 8

/-- Represents the number of sides on Apollo's die -/
def apollo_sides : ℕ := 6

/-- Calculates the probability of Diana rolling higher than Apollo -/
def prob_diana_higher : ℚ :=
  (diana_sides * (diana_sides - 1) - apollo_sides * (apollo_sides - 1)) / (2 * diana_sides * apollo_sides)

/-- Theorem stating that the probability of Diana rolling higher than Apollo is 9/16 -/
theorem diana_wins_prob : prob_diana_higher = 9/16 := by
  sorry

end diana_wins_prob_l2425_242590


namespace cyclist_distance_l2425_242532

/-- Represents the distance traveled by a cyclist at a constant speed -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that if a cyclist travels 24 km in 40 minutes at a constant speed, 
    then they will travel 18 km in 30 minutes -/
theorem cyclist_distance 
  (speed : ℝ) 
  (h1 : speed > 0) 
  (h2 : distance_traveled speed (40 / 60) = 24) : 
  distance_traveled speed (30 / 60) = 18 := by
sorry

end cyclist_distance_l2425_242532


namespace tim_weekly_earnings_l2425_242585

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 1.2

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end tim_weekly_earnings_l2425_242585


namespace simplify_and_evaluate_l2425_242598

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = -1) 
  (hy : y = 1/3) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = -2/3 := by
  sorry

end simplify_and_evaluate_l2425_242598


namespace solve_chocolates_problem_l2425_242543

def chocolates_problem (nick_chocolates : ℕ) (alix_multiplier : ℕ) (difference_after : ℕ) : Prop :=
  let initial_alix_chocolates := nick_chocolates * alix_multiplier
  let alix_chocolates_after := nick_chocolates + difference_after
  let chocolates_taken := initial_alix_chocolates - alix_chocolates_after
  chocolates_taken = 5

theorem solve_chocolates_problem :
  chocolates_problem 10 3 15 := by
  sorry

end solve_chocolates_problem_l2425_242543


namespace f_3_minus_f_4_l2425_242570

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_3_minus_f_4 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 5)
  (h_f_1 : f 1 = 1)
  (h_f_2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
sorry

end f_3_minus_f_4_l2425_242570


namespace composite_expression_l2425_242562

theorem composite_expression (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b := by
  sorry

end composite_expression_l2425_242562


namespace sqrt_eight_and_three_ninths_simplification_l2425_242599

theorem sqrt_eight_and_three_ninths_simplification :
  Real.sqrt (8 + 3 / 9) = 5 * Real.sqrt 3 / 3 := by
  sorry

end sqrt_eight_and_three_ninths_simplification_l2425_242599


namespace arithmetic_sequence_sum_l2425_242563

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 1 + a 2 = 4 →
  d = 2 →
  a 7 + a 8 = 28 := by
sorry

end arithmetic_sequence_sum_l2425_242563


namespace slag_transport_allocation_l2425_242507

/-- Represents the daily rental income for slag transport vehicles --/
def daily_rental_income (x : ℕ) : ℕ := 80000 - 200 * x

/-- Theorem stating the properties of the slag transport vehicle allocation problem --/
theorem slag_transport_allocation :
  (∀ x : ℕ, x ≤ 20 → daily_rental_income x = 80000 - 200 * x) ∧
  (∀ x : ℕ, x ≤ 20 → (daily_rental_income x ≥ 79600 ↔ x ≤ 2)) ∧
  (∀ x : ℕ, x ≤ 20 → daily_rental_income x ≤ 80000) ∧
  (daily_rental_income 0 = 80000) := by
  sorry

#check slag_transport_allocation

end slag_transport_allocation_l2425_242507


namespace inequality_condition_l2425_242554

theorem inequality_condition (x y : ℝ) : 
  (((x^3 + y^3) / 2)^(1/3) ≥ ((x^2 + y^2) / 2)^(1/2)) ↔ (x + y ≥ 0 ∨ x + y ≤ 0) := by
  sorry

end inequality_condition_l2425_242554


namespace xiaoming_age_is_10_l2425_242502

/-- Xiao Ming's age this year -/
def xiaoming_age : ℕ := sorry

/-- Father's age this year -/
def father_age : ℕ := 4 * xiaoming_age

/-- The sum of their ages 25 years later -/
def sum_ages_25_years_later : ℕ := (xiaoming_age + 25) + (father_age + 25)

theorem xiaoming_age_is_10 :
  xiaoming_age = 10 ∧ father_age = 4 * xiaoming_age ∧ sum_ages_25_years_later = 100 :=
sorry

end xiaoming_age_is_10_l2425_242502


namespace complex_number_properties_l2425_242531

theorem complex_number_properties (z : ℂ) (h : (z - 2*Complex.I)/z = 2 + Complex.I) :
  (∃ (x y : ℝ), z = x + y*Complex.I ∧ y = -1) ∧
  (∀ (z₁ : ℂ), Complex.abs (z₁ - z) = 1 → 
    Real.sqrt 2 - 1 ≤ Complex.abs z₁ ∧ Complex.abs z₁ ≤ Real.sqrt 2 + 1) :=
by sorry

end complex_number_properties_l2425_242531


namespace root_difference_equals_2000_l2425_242537

theorem root_difference_equals_2000 : ∃ (a b : ℝ), 
  ((1998 * a)^2 - 1997 * 1999 * a - 1 = 0 ∧ 
   ∀ x, (1998 * x)^2 - 1997 * 1999 * x - 1 = 0 → x ≤ a) ∧
  (b^2 + 1998 * b - 1999 = 0 ∧ 
   ∀ y, y^2 + 1998 * y - 1999 = 0 → b ≤ y) ∧
  a - b = 2000 := by
sorry

end root_difference_equals_2000_l2425_242537


namespace polynomial_simplification_l2425_242575

theorem polynomial_simplification (x : ℝ) :
  (15 * x^12 + 8 * x^10 + 11 * x^9) + (5 * x^12 + 3 * x^10 + x^9 + 6 * x^7 + 4 * x^4 + 7 * x^2 + 10) =
  20 * x^12 + 11 * x^10 + 12 * x^9 + 6 * x^7 + 4 * x^4 + 7 * x^2 + 10 := by
sorry

end polynomial_simplification_l2425_242575


namespace unique_two_digit_number_divisible_by_eight_l2425_242564

theorem unique_two_digit_number_divisible_by_eight :
  ∃! n : ℕ, 70 < n ∧ n < 80 ∧ n % 8 = 0 :=
by
  sorry

end unique_two_digit_number_divisible_by_eight_l2425_242564


namespace roger_trays_first_table_l2425_242547

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := 4

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The number of trays Roger picked up from the second table -/
def trays_from_second_table : ℕ := 2

/-- The number of trays Roger picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * num_trips - trays_from_second_table

theorem roger_trays_first_table :
  trays_from_first_table = 10 := by sorry

end roger_trays_first_table_l2425_242547


namespace cyclic_sum_inequality_l2425_242524

theorem cyclic_sum_inequality (a b c : ℝ) (n : ℕ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1) :
  (a + b + c^n) / (a^(2*n+3) + b^(2*n+3) + a*b) +
  (b + c + a^n) / (b^(2*n+3) + c^(2*n+3) + b*c) +
  (c + a + b^n) / (c^(2*n+3) + a^(2*n+3) + c*a) ≤
  a^(n+1) + b^(n+1) + c^(n+1) := by sorry

end cyclic_sum_inequality_l2425_242524


namespace seating_arrangements_l2425_242526

/-- The number of seats in the row -/
def num_seats : ℕ := 8

/-- The number of people to be seated -/
def num_people : ℕ := 3

/-- The number of empty seats required between people and at the ends -/
def num_partitions : ℕ := 4

/-- The number of ways to arrange the double empty seat -/
def double_seat_arrangements : ℕ := 4

/-- The number of ways to arrange the people -/
def people_arrangements : ℕ := 6

/-- The total number of seating arrangements -/
def total_arrangements : ℕ := double_seat_arrangements * people_arrangements

theorem seating_arrangements :
  total_arrangements = 24 :=
sorry

end seating_arrangements_l2425_242526


namespace saras_quarters_l2425_242560

/-- Given that Sara initially had 783 quarters and now has 1054 quarters,
    prove that the number of quarters Sara's dad gave her is 271. -/
theorem saras_quarters (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 783)
  (h2 : final_quarters = 1054) :
  final_quarters - initial_quarters = 271 := by
  sorry

end saras_quarters_l2425_242560


namespace geometric_sequence_minimum_value_l2425_242542

/-- A positive geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ) (q : ℝ) (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  GeometricSequence a q →
  (a m * a n).sqrt = 4 * a 1 →
  a 7 = a 6 + 2 * a 5 →
  (1 : ℝ) / m + 5 / n ≥ 7 / 4 :=
by sorry

end geometric_sequence_minimum_value_l2425_242542


namespace a_minus_b_value_l2425_242577

theorem a_minus_b_value (a b : ℝ) : 
  (|a - 2| = 5) → (|b| = 9) → (a + b < 0) → (a - b = 16 ∨ a - b = 6) := by
  sorry

end a_minus_b_value_l2425_242577


namespace fifteen_fishers_tomorrow_l2425_242593

/-- Represents the fishing schedule in a coastal village over three days -/
structure FishingSchedule where
  everyday : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterdayCount : Nat
  todayCount : Nat

/-- Calculates the number of people fishing tomorrow given a FishingSchedule -/
def tomorrowFishers (schedule : FishingSchedule) : Nat :=
  schedule.everyday +
  schedule.everyThreeDay +
  (schedule.everyOtherDay - (schedule.yesterdayCount - schedule.everyday))

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow (schedule : FishingSchedule)
  (h1 : schedule.everyday = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterdayCount = 12)
  (h5 : schedule.todayCount = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

#eval tomorrowFishers { everyday := 7, everyOtherDay := 8, everyThreeDay := 3, yesterdayCount := 12, todayCount := 10 }

end fifteen_fishers_tomorrow_l2425_242593


namespace probability_of_perfect_square_sum_l2425_242538

/-- The number of faces on a standard die -/
def standardDieFaces : ℕ := 6

/-- The set of possible sums when rolling two dice -/
def possibleSums : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

/-- The set of perfect squares within the possible sums -/
def perfectSquareSums : Set ℕ := {4, 9}

/-- The number of ways to get a sum of 4 -/
def waysToGetFour : ℕ := 3

/-- The number of ways to get a sum of 9 -/
def waysToGetNine : ℕ := 4

/-- The total number of favorable outcomes -/
def favorableOutcomes : ℕ := waysToGetFour + waysToGetNine

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := standardDieFaces * standardDieFaces

/-- Theorem: The probability of rolling two standard 6-sided dice and getting a sum that is a perfect square is 7/36 -/
theorem probability_of_perfect_square_sum :
  (favorableOutcomes : ℚ) / totalOutcomes = 7 / 36 := by
  sorry

end probability_of_perfect_square_sum_l2425_242538


namespace congruence_problem_l2425_242553

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 25 [ZMOD 42]) (h2 : b ≡ 63 [ZMOD 42]) :
  ∃ n : ℤ, 200 ≤ n ∧ n ≤ 241 ∧ a - b ≡ n [ZMOD 42] ∧ n = 214 := by
  sorry

end congruence_problem_l2425_242553


namespace blue_candy_probability_l2425_242597

/-- The probability of selecting a blue candy from a bag with green, blue, and red candies. -/
theorem blue_candy_probability
  (green : ℕ) (blue : ℕ) (red : ℕ)
  (h_green : green = 5)
  (h_blue : blue = 3)
  (h_red : red = 4) :
  (blue : ℚ) / (green + blue + red) = 1 / 4 :=
sorry

end blue_candy_probability_l2425_242597


namespace emmalyn_earnings_l2425_242582

/-- Calculates the total amount earned from painting fences. -/
def total_amount_earned (price_per_meter : ℚ) (num_fences : ℕ) (fence_length : ℕ) : ℚ :=
  price_per_meter * (num_fences : ℚ) * (fence_length : ℚ)

/-- Proves that Emmalyn earned $5,000 from painting fences. -/
theorem emmalyn_earnings : 
  total_amount_earned (20 / 100) 50 500 = 5000 := by
  sorry

#eval total_amount_earned (20 / 100) 50 500

end emmalyn_earnings_l2425_242582


namespace theater_tickets_l2425_242513

theorem theater_tickets (orchestra_price balcony_price : ℕ) 
  (total_tickets total_cost : ℕ) : 
  orchestra_price = 12 →
  balcony_price = 8 →
  total_tickets = 380 →
  total_cost = 3320 →
  ∃ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost ∧
    balcony_tickets - orchestra_tickets = 240 :=
by
  sorry

end theater_tickets_l2425_242513


namespace area_covered_five_strips_l2425_242511

/-- The area covered by overlapping rectangular strips -/
def area_covered (n : ℕ) (length width : ℝ) (intersection_width : ℝ) : ℝ :=
  n * length * width - (n.choose 2) * 2 * intersection_width^2

/-- Theorem stating the area covered by the specific configuration of strips -/
theorem area_covered_five_strips :
  area_covered 5 15 2 2 = 70 := by sorry

end area_covered_five_strips_l2425_242511


namespace hundred_passengers_sixteen_stops_l2425_242581

/-- The number of ways passengers can disembark from a train -/
def ways_to_disembark (num_passengers : ℕ) (num_stops : ℕ) : ℕ :=
  num_stops ^ num_passengers

/-- Theorem: 100 passengers disembarking at 16 stops results in 16^100 possibilities -/
theorem hundred_passengers_sixteen_stops :
  ways_to_disembark 100 16 = 16^100 := by
  sorry

end hundred_passengers_sixteen_stops_l2425_242581


namespace angle_alpha_trig_l2425_242539

theorem angle_alpha_trig (α : Real) (m : Real) :
  m ≠ 0 →
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = m ∧ x^2 + y^2 = (Real.cos α)^2 + (Real.sin α)^2) →
  Real.sin α = (Real.sqrt 2 / 4) * m →
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  ((m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
   (m < 0 → Real.tan α = Real.sqrt 15 / 3)) :=
by sorry

end angle_alpha_trig_l2425_242539


namespace angle_C_measure_l2425_242559

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem angle_C_measure (abc : Triangle) (h : abc.A + abc.B = 80) : abc.C = 100 := by
  sorry

end angle_C_measure_l2425_242559


namespace max_gold_coins_max_gold_coins_proof_l2425_242536

/-- The largest number of gold coins that can be distributed among 15 friends
    with 4 coins left over and a total less than 150. -/
theorem max_gold_coins : ℕ :=
  let num_friends : ℕ := 15
  let extra_coins : ℕ := 4
  let max_total : ℕ := 149  -- less than 150
  
  have h1 : ∃ (k : ℕ), num_friends * k + extra_coins ≤ max_total :=
    sorry
  
  have h2 : ∀ (n : ℕ), num_friends * n + extra_coins > max_total → n > 9 :=
    sorry
  
  139

theorem max_gold_coins_proof (n : ℕ) :
  n ≤ max_gold_coins ∧
  (∃ (k : ℕ), n = 15 * k + 4) ∧
  n < 150 :=
by sorry

end max_gold_coins_max_gold_coins_proof_l2425_242536


namespace smallest_n_divisible_by_247_l2425_242567

theorem smallest_n_divisible_by_247 :
  ∀ n : ℕ, n > 0 ∧ n < 37 → ¬(247 ∣ n * (n + 1) * (n + 2)) ∧ (247 ∣ 37 * 38 * 39) :=
by sorry

end smallest_n_divisible_by_247_l2425_242567


namespace product_digit_count_l2425_242527

def number1 : ℕ := 925743857234987123123
def number2 : ℕ := 10345678909876

theorem product_digit_count : (String.length (toString (number1 * number2))) = 36 := by
  sorry

end product_digit_count_l2425_242527


namespace square_of_negative_two_m_squared_l2425_242519

theorem square_of_negative_two_m_squared (m : ℝ) : (-2 * m^2)^2 = 4 * m^4 := by
  sorry

end square_of_negative_two_m_squared_l2425_242519


namespace line_equation_sum_l2425_242579

/-- Given a line with slope 5 passing through the point (2,4), prove that m + b = -1 --/
theorem line_equation_sum (m b : ℝ) : 
  m = 5 →                   -- The slope is 5
  4 = 5 * 2 + b →           -- The line passes through (2,4)
  m + b = -1 :=             -- Prove that m + b = -1
by sorry

end line_equation_sum_l2425_242579


namespace college_student_ticket_cost_l2425_242578

/-- Proves that the cost of a college student ticket is $4 given the specified conditions -/
theorem college_student_ticket_cost : 
  ∀ (total_visitors : ℕ) 
    (nyc_resident_ratio : ℚ) 
    (college_student_ratio : ℚ) 
    (total_revenue : ℚ),
  total_visitors = 200 →
  nyc_resident_ratio = 1/2 →
  college_student_ratio = 3/10 →
  total_revenue = 120 →
  (total_visitors : ℚ) * nyc_resident_ratio * college_student_ratio * 4 = total_revenue :=
by
  sorry


end college_student_ticket_cost_l2425_242578


namespace power_of_three_equation_l2425_242509

theorem power_of_three_equation (m : ℤ) : 
  3^2001 - 2 * 3^2000 - 3^1999 + 5 * 3^1998 = m * 3^1998 → m = 11 := by
  sorry

end power_of_three_equation_l2425_242509


namespace computer_price_increase_l2425_242540

theorem computer_price_increase (x : ℝ) (h : 2 * x = 540) : 
  (351 - x) / x * 100 = 30 := by sorry

end computer_price_increase_l2425_242540


namespace school_distance_is_150km_l2425_242550

/-- The distance from Xiaoming's home to school in kilometers. -/
def school_distance : ℝ := 150

/-- Xiaoming's walking speed in km/h. -/
def walking_speed : ℝ := 5

/-- The car speed in km/h. -/
def car_speed : ℝ := 15

/-- The time difference between going to school and returning home in hours. -/
def time_difference : ℝ := 2

/-- Theorem stating the distance from Xiaoming's home to school is 150 km. -/
theorem school_distance_is_150km :
  let d := school_distance
  let v_walk := walking_speed
  let v_car := car_speed
  let t_diff := time_difference
  (d / (2 * v_walk) + d / (2 * v_car) = d / (3 * v_car) + 2 * d / (3 * v_walk) + t_diff) →
  d = 150 := by
  sorry


end school_distance_is_150km_l2425_242550


namespace initial_horses_l2425_242583

theorem initial_horses (sheep : ℕ) (chickens : ℕ) (goats : ℕ) (male_animals : ℕ) : 
  sheep = 29 → 
  chickens = 9 → 
  goats = 37 → 
  male_animals = 53 → 
  ∃ (horses : ℕ), 
    horses = 100 ∧ 
    (horses + sheep + chickens) / 2 + goats = male_animals * 2 :=
by sorry

end initial_horses_l2425_242583


namespace max_students_above_mean_l2425_242551

theorem max_students_above_mean (n : ℕ) (h : n = 107) :
  ∃ (scores : Fin n → ℝ), ∃ (count : ℕ),
    count = (Finset.univ.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n)).card ∧
    count ≤ n - 1 ∧
    ∀ (other_count : ℕ),
      (∃ (other_scores : Fin n → ℝ),
        other_count = (Finset.univ.filter (λ i => other_scores i > (Finset.sum Finset.univ other_scores) / n)).card) →
      other_count ≤ count :=
by sorry

end max_students_above_mean_l2425_242551


namespace consecutive_numbers_multiple_l2425_242552

/-- Given three consecutive numbers where the first is 4.2, 
    prove that the multiple of the first number that satisfies 
    the equation is 9. -/
theorem consecutive_numbers_multiple (m : ℝ) : 
  m * 4.2 = 2 * (4.2 + 4) + 2 * (4.2 + 2) + 9 → m = 9 := by
  sorry

end consecutive_numbers_multiple_l2425_242552


namespace roller_coaster_line_length_l2425_242517

theorem roller_coaster_line_length 
  (num_cars : ℕ) 
  (people_per_car : ℕ) 
  (num_runs : ℕ) 
  (h1 : num_cars = 7)
  (h2 : people_per_car = 2)
  (h3 : num_runs = 6) :
  num_cars * people_per_car * num_runs = 84 :=
by sorry

end roller_coaster_line_length_l2425_242517


namespace absolute_value_equation_solution_l2425_242565

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 3| = |x + 2| :=
by
  -- The unique solution is x = 1/2
  use 1/2
  sorry

end absolute_value_equation_solution_l2425_242565


namespace xiaolins_age_l2425_242584

/-- Represents a person's age as a two-digit number -/
structure TwoDigitAge where
  tens : Nat
  units : Nat
  h_tens : tens < 10
  h_units : units < 10

/-- Swaps the digits of a two-digit age -/
def swapDigits (age : TwoDigitAge) : TwoDigitAge :=
  { tens := age.units,
    units := age.tens,
    h_tens := age.h_units,
    h_units := age.h_tens }

/-- Calculates the numeric value of a two-digit age -/
def toNumber (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.units

theorem xiaolins_age :
  ∀ (grandpa : TwoDigitAge),
    let dad := swapDigits grandpa
    toNumber grandpa - toNumber dad = 5 * 9 →
    9 = 9 := by sorry

end xiaolins_age_l2425_242584


namespace total_collection_l2425_242534

/-- A group of students collecting money -/
structure StudentGroup where
  members : ℕ
  contribution_per_member : ℕ
  total_paise : ℕ

/-- Conversion rate from paise to rupees -/
def paise_to_rupees (paise : ℕ) : ℚ :=
  (paise : ℚ) / 100

/-- Theorem stating the total amount collected by the group -/
theorem total_collection (group : StudentGroup) 
    (h1 : group.members = 54)
    (h2 : group.contribution_per_member = group.members)
    (h3 : group.total_paise = group.members * group.contribution_per_member) : 
    paise_to_rupees group.total_paise = 29.16 := by
  sorry

end total_collection_l2425_242534


namespace marks_animals_legs_l2425_242518

def total_legs (num_kangaroos : ℕ) (num_goats : ℕ) : ℕ :=
  2 * num_kangaroos + 4 * num_goats

theorem marks_animals_legs : 
  let num_kangaroos : ℕ := 23
  let num_goats : ℕ := 3 * num_kangaroos
  total_legs num_kangaroos num_goats = 322 := by
sorry

end marks_animals_legs_l2425_242518


namespace quadratic_inequality_solution_l2425_242557

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a + 1) * x + a * b > 0 ↔ x < -1 ∨ x > 4) →
  a + b = -3 := by
  sorry

end quadratic_inequality_solution_l2425_242557
