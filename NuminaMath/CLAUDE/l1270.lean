import Mathlib

namespace sufficient_not_necessary_l1270_127070

/-- A complex number z is in the first quadrant if its real part is positive and its imaginary part is positive. -/
def in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

/-- Given a real number a, construct the complex number z = (3i - a) / i -/
def z (a : ℝ) : ℂ :=
  Complex.I * 3 - a

/-- The condition a > -1 is sufficient but not necessary for z(a) to be in the first quadrant -/
theorem sufficient_not_necessary (a : ℝ) :
  (∃ a₁ : ℝ, a₁ > -1 ∧ in_first_quadrant (z a₁)) ∧
  (∃ a₂ : ℝ, in_first_quadrant (z a₂) ∧ ¬(a₂ > -1)) :=
sorry

end sufficient_not_necessary_l1270_127070


namespace student_selection_probability_l1270_127024

theorem student_selection_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (excluded_students : ℕ) 
  (h1 : total_students = 2008) 
  (h2 : selected_students = 50) 
  (h3 : excluded_students = 8) :
  (selected_students : ℚ) / total_students = 25 / 1004 :=
sorry

end student_selection_probability_l1270_127024


namespace fish_tagging_ratio_l1270_127055

theorem fish_tagging_ratio : 
  ∀ (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) (total_fish : ℕ),
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  total_fish = 3200 →
  (tagged_in_second : ℚ) / second_catch = 1 / 40 := by
sorry

end fish_tagging_ratio_l1270_127055


namespace divisibility_theorem_l1270_127057

theorem divisibility_theorem (a b c : ℕ) (h1 : a ∣ b * c) (h2 : Nat.gcd a b = 1) : a ∣ c := by
  sorry

end divisibility_theorem_l1270_127057


namespace least_likely_score_l1270_127078

def class_size : ℕ := 50
def average_score : ℝ := 82
def score_variance : ℝ := 8.2

def score_options : List ℝ := [60, 70, 80, 100]

def distance_from_mean (score : ℝ) : ℝ :=
  |score - average_score|

theorem least_likely_score :
  ∃ (score : ℝ), score ∈ score_options ∧
    ∀ (other : ℝ), other ∈ score_options → other ≠ score →
      distance_from_mean score > distance_from_mean other :=
by sorry

end least_likely_score_l1270_127078


namespace coin_toss_sequences_count_l1270_127021

/-- The number of ways to insert k items into n bins -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (n - 1)

/-- The number of different sequences of 17 coin tosses with specific subsequence counts -/
def coinTossSequences : ℕ := 
  let hh_insertions := starsAndBars 5 3  -- Insert 3 H into 5 existing H positions
  let tt_insertions := starsAndBars 4 6  -- Insert 6 T into 4 existing T positions
  hh_insertions * tt_insertions

/-- Theorem stating the number of coin toss sequences -/
theorem coin_toss_sequences_count :
  coinTossSequences = 2940 := by sorry

end coin_toss_sequences_count_l1270_127021


namespace apple_distribution_l1270_127044

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 26 3 3 = 171 := by
  sorry

end apple_distribution_l1270_127044


namespace bug_position_after_2023_jumps_l1270_127095

/-- Represents the possible positions on the circle -/
inductive Position : Type
| one : Position
| two : Position
| three : Position
| four : Position
| five : Position
| six : Position
| seven : Position

/-- Determines if a position is odd-numbered -/
def is_odd (p : Position) : Bool :=
  match p with
  | Position.one => true
  | Position.two => false
  | Position.three => true
  | Position.four => false
  | Position.five => true
  | Position.six => false
  | Position.seven => true

/-- Represents a single jump of the bug -/
def jump (p : Position) : Position :=
  match p with
  | Position.one => Position.three
  | Position.two => Position.five
  | Position.three => Position.five
  | Position.four => Position.seven
  | Position.five => Position.seven
  | Position.six => Position.two
  | Position.seven => Position.two

/-- Represents multiple jumps of the bug -/
def multi_jump (p : Position) (n : Nat) : Position :=
  match n with
  | 0 => p
  | n + 1 => jump (multi_jump p n)

/-- The main theorem to prove -/
theorem bug_position_after_2023_jumps :
  multi_jump Position.seven 2023 = Position.two := by
  sorry


end bug_position_after_2023_jumps_l1270_127095


namespace no_real_solutions_l1270_127071

theorem no_real_solutions : ¬∃ x : ℝ, |x| - 4 = (3 * |x|) / 2 := by
  sorry

end no_real_solutions_l1270_127071


namespace beka_jackson_miles_difference_l1270_127000

/-- The difference in miles flown between Beka and Jackson -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating the difference in miles flown between Beka and Jackson -/
theorem beka_jackson_miles_difference :
  miles_difference 873 563 = 310 :=
by sorry

end beka_jackson_miles_difference_l1270_127000


namespace slips_theorem_l1270_127003

/-- The number of slips in the bag -/
def total_slips : ℕ := 15

/-- The expected value of a randomly drawn slip -/
def expected_value : ℚ := 46/10

/-- The value on some of the slips -/
def value1 : ℕ := 3

/-- The value on the rest of the slips -/
def value2 : ℕ := 8

/-- The number of slips with value1 -/
def slips_with_value1 : ℕ := 10

theorem slips_theorem : 
  ∃ (x : ℕ), x = slips_with_value1 ∧ 
  x ≤ total_slips ∧
  (x : ℚ) / total_slips * value1 + (total_slips - x : ℚ) / total_slips * value2 = expected_value :=
sorry

end slips_theorem_l1270_127003


namespace alice_basic_salary_l1270_127017

/-- Calculates the monthly basic salary given total sales, commission rate, and savings. -/
def calculate_basic_salary (total_sales : ℝ) (commission_rate : ℝ) (savings : ℝ) : ℝ :=
  let total_earnings := savings * 10
  let commission := total_sales * commission_rate
  total_earnings - commission

/-- Proves that given the specified conditions, Alice's monthly basic salary is $240. -/
theorem alice_basic_salary :
  let total_sales : ℝ := 2500
  let commission_rate : ℝ := 0.02
  let savings : ℝ := 29
  calculate_basic_salary total_sales commission_rate savings = 240 := by
  sorry

#eval calculate_basic_salary 2500 0.02 29

end alice_basic_salary_l1270_127017


namespace inequality_proof_l1270_127053

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (7 * a^2 + b^2 + c^2)) + 
  (b / Real.sqrt (a^2 + 7 * b^2 + c^2)) + 
  (c / Real.sqrt (a^2 + b^2 + 7 * c^2)) ≤ 1 := by
sorry

end inequality_proof_l1270_127053


namespace sum_of_consecutive_odd_numbers_l1270_127010

theorem sum_of_consecutive_odd_numbers : 
  let odd_numbers := [997, 999, 1001, 1003, 1005]
  (List.sum odd_numbers) = 5100 - 95 := by
sorry

end sum_of_consecutive_odd_numbers_l1270_127010


namespace fraction_sum_l1270_127030

theorem fraction_sum (w x y : ℝ) (h1 : (w + x) / 2 = 0.5) (h2 : w * x = y) : 
  5 / w + 5 / x = 20 := by
sorry

end fraction_sum_l1270_127030


namespace similar_triangles_side_length_l1270_127073

/-- Two triangles are similar -/
structure SimilarTriangles (Triangle1 Triangle2 : Type) :=
  (similar : Triangle1 → Triangle2 → Prop)

/-- Definition of a triangle with side lengths -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Theorem: If triangles DEF and ABC are similar, and DE = 6, EF = 12, BC = 18, then AB = 9 -/
theorem similar_triangles_side_length 
  (DEF ABC : Triangle)
  (h_similar : SimilarTriangles Triangle Triangle)
  (h_similar_triangles : h_similar.similar DEF ABC)
  (h_DE : DEF.side1 = 6)
  (h_EF : DEF.side2 = 12)
  (h_BC : ABC.side2 = 18) :
  ABC.side1 = 9 :=
sorry

end similar_triangles_side_length_l1270_127073


namespace max_shot_radius_l1270_127069

/-- Given a sphere of radius 3 cm from which 27 shots can be made, 
    prove that the maximum radius of each shot is 1 cm. -/
theorem max_shot_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 3 → n = 27 → (4 / 3 * Real.pi * R^3 = n * (4 / 3 * Real.pi * r^3)) → r ≤ 1 := by
  sorry

#check max_shot_radius

end max_shot_radius_l1270_127069


namespace cube_sum_difference_l1270_127039

theorem cube_sum_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 210 → a + b + c = 7 := by
  sorry

end cube_sum_difference_l1270_127039


namespace car_distance_l1270_127067

/-- The distance traveled by a car in 30 minutes, given that it travels at 2/3 the speed of a train moving at 90 miles per hour -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (time : ℝ) : 
  train_speed = 90 →
  car_speed_ratio = 2 / 3 →
  time = 1 / 2 →
  car_speed_ratio * train_speed * time = 30 := by
sorry

end car_distance_l1270_127067


namespace subset_condition_l1270_127082

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | -m < x ∧ x < m}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 1 := by
  sorry

end subset_condition_l1270_127082


namespace geometric_progressions_existence_l1270_127033

theorem geometric_progressions_existence :
  (∃ a r : ℚ, 
    (∀ k : ℕ, k < 4 → 200 ≤ a * r^k ∧ a * r^k ≤ 1200) ∧
    (∀ k : ℕ, k < 4 → ∃ n : ℕ, a * r^k = n)) ∧
  (∃ b s : ℚ, 
    (∀ k : ℕ, k < 6 → 200 ≤ b * s^k ∧ b * s^k ≤ 1200) ∧
    (∀ k : ℕ, k < 6 → ∃ n : ℕ, b * s^k = n)) :=
by sorry

end geometric_progressions_existence_l1270_127033


namespace pencil_sharpening_ishas_pencil_l1270_127042

/-- The length sharpened off a pencil is equal to the difference between
    the original length and the new length after sharpening. -/
theorem pencil_sharpening (original_length new_length : ℝ) :
  original_length ≥ new_length →
  original_length - new_length = original_length - new_length :=
by
  sorry

/-- Isha's pencil problem -/
theorem ishas_pencil :
  let original_length : ℝ := 31
  let new_length : ℝ := 14
  original_length - new_length = 17 :=
by
  sorry

end pencil_sharpening_ishas_pencil_l1270_127042


namespace toys_donation_problem_l1270_127041

theorem toys_donation_problem (leila_bags : ℕ) (leila_toys_per_bag : ℕ) 
  (mohamed_bags : ℕ) (extra_toys : ℕ) :
  leila_bags = 2 →
  leila_toys_per_bag = 25 →
  mohamed_bags = 3 →
  extra_toys = 7 →
  (mohamed_bags * ((leila_bags * leila_toys_per_bag + extra_toys) / mohamed_bags) = 
   leila_bags * leila_toys_per_bag + extra_toys) ∧
  ((leila_bags * leila_toys_per_bag + extra_toys) / mohamed_bags = 19) :=
by sorry

end toys_donation_problem_l1270_127041


namespace circle_radius_from_spherical_coordinates_l1270_127008

theorem circle_radius_from_spherical_coordinates : 
  let r : ℝ := Real.sqrt 3 / 2
  ∀ θ : ℝ, 
    let x : ℝ := Real.sin (π/3) * Real.cos θ
    let y : ℝ := Real.sin (π/3) * Real.sin θ
    Real.sqrt (x^2 + y^2) = r := by sorry

end circle_radius_from_spherical_coordinates_l1270_127008


namespace probability_sum_three_l1270_127094

/-- Represents the color of a ball --/
inductive BallColor
  | Red
  | Yellow
  | Blue

/-- Represents the score of a ball --/
def score (color : BallColor) : ℕ :=
  match color with
  | BallColor.Red => 1
  | BallColor.Yellow => 2
  | BallColor.Blue => 3

/-- The total number of balls in the bag --/
def totalBalls : ℕ := 6

/-- The number of possible outcomes when drawing two balls with replacement --/
def totalOutcomes : ℕ := totalBalls * totalBalls

/-- The number of favorable outcomes (sum of scores is 3) --/
def favorableOutcomes : ℕ := 12

/-- Theorem stating that the probability of drawing two balls with a sum of scores equal to 3 is 1/3 --/
theorem probability_sum_three (h : favorableOutcomes = 12) :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 3 := by
  sorry


end probability_sum_three_l1270_127094


namespace fraction_problem_l1270_127006

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/3) * F * N = 18) 
  (h2 : (3/10) * N = 64.8) : 
  F = 1/4 := by
sorry

end fraction_problem_l1270_127006


namespace necessary_and_sufficient_l1270_127091

theorem necessary_and_sufficient (p q : Prop) : 
  (p → q) → (q → p) → (p ↔ q) := by
  sorry

end necessary_and_sufficient_l1270_127091


namespace fruit_basket_theorem_l1270_127051

/-- Calculates the number of possible fruit baskets given a number of apples and oranges. -/
def fruitBasketCount (apples : ℕ) (oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - 1

/-- Theorem stating that the number of fruit baskets with 4 apples and 8 oranges is 44. -/
theorem fruit_basket_theorem :
  fruitBasketCount 4 8 = 44 := by
  sorry

end fruit_basket_theorem_l1270_127051


namespace egg_groups_l1270_127066

theorem egg_groups (total_eggs : ℕ) (eggs_per_group : ℕ) (h1 : total_eggs = 35) (h2 : eggs_per_group = 7) :
  total_eggs / eggs_per_group = 5 := by
  sorry

end egg_groups_l1270_127066


namespace expression_evaluation_l1270_127045

theorem expression_evaluation : (4 + 5 + 6) / 3 * 2 - 2 / 2 = 9 := by
  sorry

end expression_evaluation_l1270_127045


namespace x_plus_q_equals_five_plus_two_q_l1270_127068

theorem x_plus_q_equals_five_plus_two_q (x q : ℝ) 
  (h1 : |x - 5| = q) 
  (h2 : x > 5) : 
  x + q = 5 + 2*q := by
sorry

end x_plus_q_equals_five_plus_two_q_l1270_127068


namespace tangent_line_equality_l1270_127050

theorem tangent_line_equality (x₁ x₂ y₁ y₂ : ℝ) :
  (∃ m b : ℝ, (∀ x : ℝ, y₁ + (Real.exp x₁) * (x - x₁) = m * x + b) ∧
              (∀ x : ℝ, y₂ + (1 / x₂) * (x - x₂) = m * x + b)) →
  (x₁ + 1) * (x₂ - 1) = -2 :=
by sorry

end tangent_line_equality_l1270_127050


namespace discriminant_of_5x2_plus_3x_minus_8_l1270_127031

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 + 3x - 8 is 169 -/
theorem discriminant_of_5x2_plus_3x_minus_8 :
  discriminant 5 3 (-8) = 169 := by
  sorry

end discriminant_of_5x2_plus_3x_minus_8_l1270_127031


namespace anne_wandering_l1270_127027

/-- Anne's wandering problem -/
theorem anne_wandering (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2.0 → time = 1.5 → distance = speed * time → distance = 3.0 := by
  sorry

end anne_wandering_l1270_127027


namespace ball_probabilities_l1270_127037

def red_balls : ℕ := 5
def black_balls : ℕ := 7
def additional_balls : ℕ := 6

def probability_red : ℚ := red_balls / (red_balls + black_balls)
def probability_black : ℚ := black_balls / (red_balls + black_balls)

def new_red_balls : ℕ := red_balls + 4
def new_black_balls : ℕ := black_balls + 2

theorem ball_probabilities :
  (probability_black > probability_red) ∧
  (new_red_balls / (new_red_balls + new_black_balls) = new_black_balls / (new_red_balls + new_black_balls)) :=
by sorry

end ball_probabilities_l1270_127037


namespace alternating_digit_sum_2017_l1270_127049

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Alternating sum of digit sums from 1 to n -/
def alternating_digit_sum (n : ℕ) : ℤ :=
  (Finset.range n).sum (fun i => (-1)^i.succ * (digit_sum (i + 1) : ℤ))

/-- The alternating sum of digit sums for integers from 1 to 2017 is equal to 1009 -/
theorem alternating_digit_sum_2017 : alternating_digit_sum 2017 = 1009 := by sorry

end alternating_digit_sum_2017_l1270_127049


namespace sum_and_product_problem_l1270_127099

theorem sum_and_product_problem (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 ∧ x^2 + y^2 = 80 := by
  sorry

end sum_and_product_problem_l1270_127099


namespace midpoint_distance_range_l1270_127036

/-- Given two parallel lines and a point constrained to lie between them, 
    prove the range of the squared distance from this point to the origin. -/
theorem midpoint_distance_range (x₀ y₀ : ℝ) : 
  (∃ x y u v : ℝ, 
    x - 2*y - 2 = 0 ∧ 
    u - 2*v - 6 = 0 ∧ 
    x₀ = (x + u) / 2 ∧ 
    y₀ = (y + v) / 2 ∧
    (x₀ - 2)^2 + (y₀ + 1)^2 ≤ 5) →
  16/5 ≤ x₀^2 + y₀^2 ∧ x₀^2 + y₀^2 ≤ 16 :=
by sorry

end midpoint_distance_range_l1270_127036


namespace ancient_chinese_math_problem_l1270_127081

/-- Represents the cost of animals in taels of silver -/
structure AnimalCost where
  cow : ℝ
  sheep : ℝ

/-- The total cost of a group of animals -/
def totalCost (c : AnimalCost) (numCows numSheep : ℕ) : ℝ :=
  c.cow * (numCows : ℝ) + c.sheep * (numSheep : ℝ)

/-- The theorem representing the ancient Chinese mathematical problem -/
theorem ancient_chinese_math_problem (c : AnimalCost) : 
  totalCost c 5 2 = 19 ∧ totalCost c 2 3 = 12 ↔ 
  (5 * c.cow + 2 * c.sheep = 19 ∧ 2 * c.cow + 3 * c.sheep = 12) := by
  sorry

end ancient_chinese_math_problem_l1270_127081


namespace product_xyz_l1270_127013

theorem product_xyz (x y z : ℝ) 
  (sphere_eq : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (plane_eq : x + y + z = 12) :
  x * y * z = 42 := by
  sorry

end product_xyz_l1270_127013


namespace expression_simplification_l1270_127009

theorem expression_simplification : (((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4)) = 12.75 := by
  sorry

end expression_simplification_l1270_127009


namespace cost_of_ingredients_l1270_127087

-- Define the given values
def popcorn_sales_per_day : ℕ := 50
def cotton_candy_multiplier : ℕ := 3
def activity_duration : ℕ := 5
def rent : ℕ := 30
def final_earnings : ℕ := 895

-- Define the theorem
theorem cost_of_ingredients :
  let total_daily_sales := popcorn_sales_per_day + cotton_candy_multiplier * popcorn_sales_per_day
  let total_sales := total_daily_sales * activity_duration
  let earnings_after_rent := total_sales - rent
  earnings_after_rent - final_earnings = 75 := by
sorry

end cost_of_ingredients_l1270_127087


namespace complex_fraction_equals_i_l1270_127060

theorem complex_fraction_equals_i : (1 + Complex.I) / (1 - Complex.I) = Complex.I := by
  sorry

end complex_fraction_equals_i_l1270_127060


namespace blue_socks_cost_l1270_127023

/-- The cost of blue socks given the total cost, number of red and blue socks, and cost of red socks -/
def cost_of_blue_socks (total_cost : ℚ) (num_red : ℕ) (num_blue : ℕ) (cost_red : ℚ) : ℚ :=
  (total_cost - num_red * cost_red) / num_blue

/-- Theorem stating the cost of each pair of blue socks -/
theorem blue_socks_cost :
  cost_of_blue_socks 42 4 6 3 = 5 := by
  sorry

end blue_socks_cost_l1270_127023


namespace percent_of_a_is_4b_l1270_127084

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) :
  (4 * b) / a * 100 = 333.33 := by
  sorry

end percent_of_a_is_4b_l1270_127084


namespace monic_quadratic_polynomial_l1270_127018

theorem monic_quadratic_polynomial (f : ℝ → ℝ) :
  (∃ a b : ℝ, ∀ x, f x = x^2 + a*x + b) →  -- monic quadratic polynomial
  f 1 = 3 →                               -- f(1) = 3
  f 2 = 12 →                              -- f(2) = 12
  ∀ x, f x = x^2 + 6*x - 4 :=              -- f(x) = x^2 + 6x - 4
by sorry

end monic_quadratic_polynomial_l1270_127018


namespace cost_price_calculation_l1270_127079

-- Define the selling price
def selling_price : ℚ := 715

-- Define the profit percentage
def profit_percentage : ℚ := 10 / 100

-- Define the cost price
def cost_price : ℚ := 650

-- Theorem to prove
theorem cost_price_calculation :
  cost_price = selling_price / (1 + profit_percentage) :=
sorry

end cost_price_calculation_l1270_127079


namespace dodecagon_areas_l1270_127047

/-- A regular dodecagon with side length 1 cm -/
structure RegularDodecagon where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- An equilateral triangle within the dodecagon -/
structure EquilateralTriangle where
  area : ℝ
  is_one_cm_squared : area = 1

/-- A square within the dodecagon -/
structure Square where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- A regular hexagon within the dodecagon -/
structure RegularHexagon where
  side_length : ℝ
  is_one_cm : side_length = 1

/-- The decomposition of the dodecagon -/
structure DodecagonDecomposition where
  triangles : Finset EquilateralTriangle
  squares : Finset Square
  hexagon : RegularHexagon
  triangle_count : triangles.card = 6
  square_count : squares.card = 6

theorem dodecagon_areas 
  (d : RegularDodecagon) 
  (decomp : DodecagonDecomposition) : 
  /- 1. The area of the hexagon is 6 cm² -/
  decomp.hexagon.side_length ^ 2 * Real.sqrt 3 / 4 * 6 = 6 ∧ 
  /- 2. The area of the figure formed by removing 12 equilateral triangles is 6 cm² -/
  (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12 + 6 * d.side_length ^ 2) - 
    (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12) = 6 ∧
  /- 3. The area of the figure formed by removing 2 regular hexagons is 6 cm² -/
  (d.side_length ^ 2 * Real.sqrt 3 / 4 * 12 + 6 * d.side_length ^ 2) - 
    (2 * (decomp.hexagon.side_length ^ 2 * Real.sqrt 3 / 4 * 6)) = 6 :=
by sorry

end dodecagon_areas_l1270_127047


namespace shaded_region_perimeter_l1270_127052

/-- The perimeter of the shaded region formed by three identical touching circles --/
theorem shaded_region_perimeter (c : ℝ) (n : ℕ) (α : ℝ) : 
  c = 48 → n = 3 → α = 90 → c * (α / 360) * n = 36 := by sorry

end shaded_region_perimeter_l1270_127052


namespace cd_price_correct_l1270_127004

/-- The price of a CD in dollars -/
def price_cd : ℝ := 14

/-- The price of a cassette in dollars -/
def price_cassette : ℝ := 9

/-- The total amount Leanna has to spend in dollars -/
def total_money : ℝ := 37

/-- The amount left over when buying one CD and two cassettes in dollars -/
def money_left : ℝ := 5

theorem cd_price_correct : 
  (2 * price_cd + price_cassette = total_money) ∧ 
  (price_cd + 2 * price_cassette = total_money - money_left) :=
by sorry

end cd_price_correct_l1270_127004


namespace pencil_boxes_filled_l1270_127046

/-- Given 648 pencils and 4 pencils per box, prove that the number of filled boxes is 162. -/
theorem pencil_boxes_filled (total_pencils : ℕ) (pencils_per_box : ℕ) (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) :
  total_pencils / pencils_per_box = 162 := by
  sorry

end pencil_boxes_filled_l1270_127046


namespace solution_characterization_l1270_127092

/-- The set of solutions to the equation x y z + x y + y z + z x + x + y + z = 1977 --/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(1, 22, 42), (1, 42, 22), (22, 1, 42), (22, 42, 1), (42, 1, 22), (42, 22, 1)}

/-- The equation x y z + x y + y z + z x + x + y + z = 1977 --/
def SatisfiesEquation (x y z : ℕ) : Prop :=
  x * y * z + x * y + y * z + z * x + x + y + z = 1977

theorem solution_characterization :
  ∀ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0) →
    (SatisfiesEquation x y z ↔ (x, y, z) ∈ SolutionSet) := by
  sorry

end solution_characterization_l1270_127092


namespace tan_alpha_for_point_on_terminal_side_l1270_127043

theorem tan_alpha_for_point_on_terminal_side (α : Real) :
  let P : ℝ × ℝ := (1, -2)
  (P.1 = 1 ∧ P.2 = -2) →  -- Point P(1, -2) lies on the terminal side of angle α
  Real.tan α = -2 :=
by sorry

end tan_alpha_for_point_on_terminal_side_l1270_127043


namespace count_valid_license_plates_l1270_127025

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of possible digits -/
def digit_range : ℕ := 10

/-- The number of letter positions in a license plate -/
def letter_positions : ℕ := 3

/-- The number of digit positions in a license plate -/
def digit_positions : ℕ := 2

/-- Calculates the total number of valid license plates -/
def valid_license_plates : ℕ := alphabet_size ^ letter_positions * digit_range ^ digit_positions

theorem count_valid_license_plates : valid_license_plates = 1757600 := by
  sorry

end count_valid_license_plates_l1270_127025


namespace q_value_l1270_127040

theorem q_value (t R m q : ℝ) (h : R = t / ((2 + m) ^ q)) :
  q = Real.log (t / R) / Real.log (2 + m) := by
  sorry

end q_value_l1270_127040


namespace candidate_vote_percentage_l1270_127038

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15 / 100) 
  (h3 : candidate_valid_votes = 333200) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 70 / 100 := by
sorry

end candidate_vote_percentage_l1270_127038


namespace square_area_on_parabola_l1270_127012

/-- The area of a square with one side on y = 5 and endpoints on y = x^2 + 3x + 2 is 21 -/
theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 3*x₁ + 2 = 5) ∧
  (x₂^2 + 3*x₂ + 2 = 5) ∧
  (x₁ ≠ x₂) ∧
  ((x₂ - x₁)^2 = 21) := by
  sorry

end square_area_on_parabola_l1270_127012


namespace jane_calculation_l1270_127032

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 17) 
  (h2 : x - y - z = 5) : 
  x - y = 11 := by
sorry

end jane_calculation_l1270_127032


namespace division_of_powers_l1270_127085

theorem division_of_powers (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a * b^2) / (a * b) = b :=
by sorry

end division_of_powers_l1270_127085


namespace probability_non_yellow_jelly_bean_l1270_127086

/-- The probability of selecting a non-yellow jelly bean from a bag -/
theorem probability_non_yellow_jelly_bean 
  (red : ℕ) (green : ℕ) (yellow : ℕ) (blue : ℕ)
  (h_red : red = 4)
  (h_green : green = 5)
  (h_yellow : yellow = 9)
  (h_blue : blue = 10) :
  (red + green + blue : ℚ) / (red + green + yellow + blue) = 19 / 28 := by
sorry

end probability_non_yellow_jelly_bean_l1270_127086


namespace quadratic_rewrite_l1270_127080

theorem quadratic_rewrite (x : ℝ) :
  ∃ (a b c : ℤ), 16 * x^2 - 40 * x + 24 = (a * x + b : ℝ)^2 + c ∧ a * b = -20 :=
by sorry

end quadratic_rewrite_l1270_127080


namespace inequality_solution_l1270_127074

theorem inequality_solution (x : ℝ) :
  x ≠ 1 →
  (x^3 - 3*x^2 + 2*x + 1) / (x^2 - 2*x + 1) ≤ 2 ↔ 
  (2 - Real.sqrt 3 < x ∧ x < 1) ∨ (1 < x ∧ x < 2 + Real.sqrt 3) :=
by sorry

end inequality_solution_l1270_127074


namespace handshake_count_l1270_127063

theorem handshake_count (n : Nat) (h : n = 8) : 
  (n * (n - 1)) / 2 = 28 := by
  sorry

end handshake_count_l1270_127063


namespace carly_shipping_cost_l1270_127088

/-- Calculates the shipping cost given a flat fee, per-pound cost, and weight -/
def shipping_cost (flat_fee : ℝ) (per_pound_cost : ℝ) (weight : ℝ) : ℝ :=
  flat_fee + per_pound_cost * weight

/-- Theorem: The shipping cost for Carly's package is $9.00 -/
theorem carly_shipping_cost :
  shipping_cost 5 0.8 5 = 9 := by
sorry

end carly_shipping_cost_l1270_127088


namespace min_distance_sliding_ruler_l1270_127054

/-- The minimum distance between a point and the endpoint of a sliding ruler -/
theorem min_distance_sliding_ruler (h s : ℝ) (h_pos : h > 0) (s_pos : s > 0) (h_gt_s : h > s) :
  let min_distance := Real.sqrt (h^2 - s^2)
  ∀ (distance : ℝ), distance ≥ min_distance :=
sorry

end min_distance_sliding_ruler_l1270_127054


namespace first_patient_therapy_hours_l1270_127076

/-- Represents the cost structure and patient charges for a psychologist's therapy sessions. -/
structure TherapyCost where
  first_hour : ℕ           -- Cost of the first hour
  additional_hour : ℕ      -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  two_hour_total : ℕ       -- Total charge for a patient receiving 2 hours

/-- Calculates the number of therapy hours for the first patient given the cost structure. -/
def calculate_therapy_hours (cost : TherapyCost) : ℕ :=
  -- The implementation is not provided as per the instructions
  sorry

/-- Theorem stating that given the specific cost structure, the first patient received 5 hours of therapy. -/
theorem first_patient_therapy_hours 
  (cost : TherapyCost)
  (h1 : cost.first_hour = cost.additional_hour + 35)
  (h2 : cost.two_hour_total = 161)
  (h3 : cost.first_patient_total = 350) :
  calculate_therapy_hours cost = 5 := by
  sorry

end first_patient_therapy_hours_l1270_127076


namespace connie_marbles_l1270_127029

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℝ := 183.0

/-- The number of marbles Connie has left -/
def marbles_left : ℝ := 593.0

/-- The initial number of marbles Connie had -/
def initial_marbles : ℝ := marbles_given + marbles_left

theorem connie_marbles : initial_marbles = 776.0 := by
  sorry

end connie_marbles_l1270_127029


namespace liz_scored_three_three_pointers_l1270_127016

/-- Represents the basketball game scenario described in the problem -/
structure BasketballGame where
  initial_deficit : ℕ
  free_throws : ℕ
  jump_shots : ℕ
  opponent_points : ℕ
  final_deficit : ℕ

/-- Calculates the number of three-pointers Liz scored -/
def three_pointers (game : BasketballGame) : ℕ :=
  let points_needed := game.initial_deficit - game.final_deficit + game.opponent_points
  let points_from_other_shots := game.free_throws + 2 * game.jump_shots
  (points_needed - points_from_other_shots) / 3

/-- Theorem stating that Liz scored 3 three-pointers -/
theorem liz_scored_three_three_pointers :
  let game := BasketballGame.mk 20 5 4 10 8
  three_pointers game = 3 := by sorry

end liz_scored_three_three_pointers_l1270_127016


namespace reciprocal_product_l1270_127035

theorem reciprocal_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) :
  (1 / x) * (1 / y) = 1 / 8 := by
sorry

end reciprocal_product_l1270_127035


namespace power_function_uniqueness_l1270_127020

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_uniqueness 
  (f : ℝ → ℝ) 
  (h1 : is_power_function f) 
  (h2 : f 27 = 3) : 
  ∀ x : ℝ, f x = x ^ (1/3) :=
sorry

end power_function_uniqueness_l1270_127020


namespace problem1_problem2_l1270_127058

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the axioms
axiom parallel_trans_LP {l1 l2 : Line} {p : Plane} :
  parallel l1 l2 → parallelLP l2 p → (parallelLP l1 p ∨ subset l1 p)

axiom parallel_trans_PP {p1 p2 p3 : Plane} :
  parallelPP p1 p2 → parallelPP p2 p3 → parallelPP p1 p3

axiom perpendicular_parallel {l : Line} {p1 p2 : Plane} :
  perpendicular l p1 → parallelPP p1 p2 → perpendicular l p2

-- State the theorems
theorem problem1 {m n : Line} {α : Plane} :
  parallel m n → parallelLP n α → (parallelLP m α ∨ subset m α) :=
by sorry

theorem problem2 {m : Line} {α β γ : Plane} :
  parallelPP α β → parallelPP β γ → perpendicular m α → perpendicular m γ :=
by sorry

end problem1_problem2_l1270_127058


namespace min_product_of_three_numbers_l1270_127019

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (z_eq_3x : z = 3 * x)
  (ordered : x ≤ y ∧ y ≤ z)
  (max_triple : z ≤ 3 * x) : 
  ∃ (min_prod : ℝ), min_prod = 9 / 343 ∧ x * y * z ≥ min_prod :=
sorry

end min_product_of_three_numbers_l1270_127019


namespace train_length_l1270_127061

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ length : ℝ, 
  (abs (length - 250.05) < 0.01) ∧ (length = speed * 1000 / 3600 * time) :=
sorry

end train_length_l1270_127061


namespace equation1_solution_equation2_no_solution_l1270_127028

-- Define the equations
def equation1 (x : ℝ) : Prop := 4 / (x - 6) = 3 / (x + 1)
def equation2 (x : ℝ) : Prop := (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1

-- Theorem for equation1
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x = -22 :=
sorry

-- Theorem for equation2
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x :=
sorry

end equation1_solution_equation2_no_solution_l1270_127028


namespace motel_rent_reduction_l1270_127093

/-- Represents a motel with rooms rented at two different prices -/
structure Motel :=
  (price1 : ℕ)
  (price2 : ℕ)
  (total_rent : ℕ)
  (room_change : ℕ)

/-- The percentage reduction in total rent when changing room prices -/
def rent_reduction_percentage (m : Motel) : ℚ :=
  ((m.price2 - m.price1) * m.room_change : ℚ) / m.total_rent * 100

/-- Theorem stating that for a motel with specific conditions, 
    changing 10 rooms from $60 to $40 results in a 10% rent reduction -/
theorem motel_rent_reduction 
  (m : Motel) 
  (h1 : m.price1 = 40)
  (h2 : m.price2 = 60)
  (h3 : m.total_rent = 2000)
  (h4 : m.room_change = 10) :
  rent_reduction_percentage m = 10 := by
  sorry

#eval rent_reduction_percentage ⟨40, 60, 2000, 10⟩

end motel_rent_reduction_l1270_127093


namespace vector_parallel_condition_l1270_127001

/-- Prove that given vectors a, b, u, and v with specific conditions, x = 1/2 --/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  let u : Fin 2 → ℝ := a + 2 • b
  let v : Fin 2 → ℝ := 2 • a - b
  (∃ (k : ℝ), u = k • v) → x = 1/2 := by
sorry

end vector_parallel_condition_l1270_127001


namespace min_clerks_problem_l1270_127098

theorem min_clerks_problem : ∃ n : ℕ, n > 0 ∧ (Nat.choose n 4 = 3 * Nat.choose n 3) ∧ ∀ m : ℕ, m > 0 ∧ m < n → Nat.choose m 4 ≠ 3 * Nat.choose m 3 := by
  sorry

end min_clerks_problem_l1270_127098


namespace rectangle_inscribed_circle_circumference_l1270_127026

theorem rectangle_inscribed_circle_circumference 
  (width : ℝ) (height : ℝ) (circumference : ℝ) :
  width = 7 ∧ height = 24 →
  circumference = Real.pi * Real.sqrt (width^2 + height^2) →
  circumference = 25 * Real.pi :=
by sorry

end rectangle_inscribed_circle_circumference_l1270_127026


namespace positive_x_solution_l1270_127056

theorem positive_x_solution (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (x_pos : x > 0) :
  x = Real.sqrt (14 * 17 * 60) / 17 - 2 := by
  sorry

end positive_x_solution_l1270_127056


namespace danny_found_seven_caps_l1270_127072

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem: Danny found 7 bottle caps at the park -/
theorem danny_found_seven_caps : bottle_caps_found 25 32 = 7 := by
  sorry

end danny_found_seven_caps_l1270_127072


namespace product_greater_than_sum_minus_one_l1270_127075

theorem product_greater_than_sum_minus_one {a₁ a₂ : ℝ} 
  (h₁ : 0 < a₁) (h₂ : a₁ < 1) (h₃ : 0 < a₂) (h₄ : a₂ < 1) : 
  a₁ * a₂ > a₁ + a₂ - 1 := by
  sorry

end product_greater_than_sum_minus_one_l1270_127075


namespace buttons_multiple_l1270_127065

theorem buttons_multiple (sue_buttons kendra_buttons mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : kendra_buttons = 2 * sue_buttons)
  (h3 : ∃ m : ℕ, mari_buttons = m * kendra_buttons + 4)
  (h4 : mari_buttons = 64) : 
  ∃ m : ℕ, mari_buttons = m * kendra_buttons + 4 ∧ m = 5 :=
by sorry

end buttons_multiple_l1270_127065


namespace greatest_three_digit_divisible_by_eight_l1270_127090

theorem greatest_three_digit_divisible_by_eight :
  ∃ n : ℕ, n = 992 ∧ 
  n ≥ 100 ∧ n < 1000 ∧
  n % 8 = 0 ∧
  ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 8 = 0 → m ≤ n :=
by sorry

end greatest_three_digit_divisible_by_eight_l1270_127090


namespace equation_solution_l1270_127048

theorem equation_solution (M : ℚ) : (5 + 7 + 9) / 3 = (4020 + 4021 + 4022) / M → M = 1723 := by
  sorry

end equation_solution_l1270_127048


namespace circle_radius_in_ellipse_l1270_127062

/-- Two circles of radius r are externally tangent to each other and internally tangent to the ellipse x² + 4y² = 5. -/
theorem circle_radius_in_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x - r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end circle_radius_in_ellipse_l1270_127062


namespace souvenirs_for_45_colleagues_l1270_127022

def souvenir_pattern : Nat → Nat
| 0 => 1
| 1 => 3
| 2 => 5
| 3 => 7
| n + 4 => souvenir_pattern n

def total_souvenirs (n : Nat) : Nat :=
  (List.range n).map souvenir_pattern |>.sum

theorem souvenirs_for_45_colleagues :
  total_souvenirs 45 = 177 := by
  sorry

end souvenirs_for_45_colleagues_l1270_127022


namespace photo_rectangle_perimeters_l1270_127059

/-- Represents a photograph with length and width -/
structure Photo where
  length : ℝ
  width : ℝ

/-- Represents a rectangle composed of photographs -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The problem statement -/
theorem photo_rectangle_perimeters 
  (photo : Photo)
  (rect1 rect2 rect3 : Rectangle)
  (h1 : 2 * (photo.length + photo.width) = 20)
  (h2 : 2 * (rect2.length + rect2.width) = 56)
  (h3 : rect1.length = 2 * photo.length ∧ rect1.width = 2 * photo.width)
  (h4 : rect2.length = photo.length ∧ rect2.width = 4 * photo.width)
  (h5 : rect3.length = 4 * photo.length ∧ rect3.width = photo.width) :
  2 * (rect1.length + rect1.width) = 40 ∧ 2 * (rect3.length + rect3.width) = 44 := by
  sorry

end photo_rectangle_perimeters_l1270_127059


namespace set_closure_under_difference_l1270_127002

theorem set_closure_under_difference (A : Set ℝ) 
  (h1 : ∀ a ∈ A, (2 * a) ∈ A) 
  (h2 : ∀ a b, a ∈ A → b ∈ A → (a + b) ∈ A) : 
  ∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A := by
  sorry

end set_closure_under_difference_l1270_127002


namespace ellipse_sine_intersections_l1270_127097

/-- An ellipse with center (h, k) and semi-major and semi-minor axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of an ellipse -/
def ellipse_eq (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

/-- The graph of y = sin x -/
def sine_graph (x y : ℝ) : Prop :=
  y = Real.sin x

/-- A point (x, y) is an intersection point if it satisfies both equations -/
def is_intersection_point (e : Ellipse) (x y : ℝ) : Prop :=
  ellipse_eq e x y ∧ sine_graph x y

/-- The theorem stating that there exists an ellipse with more than 8 intersection points -/
theorem ellipse_sine_intersections :
  ∃ (e : Ellipse), ∃ (points : Finset (ℝ × ℝ)),
    (∀ (p : ℝ × ℝ), p ∈ points → is_intersection_point e p.1 p.2) ∧
    points.card > 8 :=
  sorry

end ellipse_sine_intersections_l1270_127097


namespace bike_truck_travel_time_indeterminate_equal_travel_time_l1270_127077

/-- Given a bike and a truck with the same speed covering the same distance,
    prove that their travel times are equal but indeterminate without knowing the speed. -/
theorem bike_truck_travel_time (distance : ℝ) (speed : ℝ) : 
  distance > 0 → speed > 0 → ∃ (time : ℝ), 
    time = distance / speed ∧ 
    (∀ (bike_time truck_time : ℝ), 
      bike_time = distance / speed → 
      truck_time = distance / speed → 
      bike_time = truck_time) :=
by sorry

/-- The specific distance covered by both vehicles -/
def covered_distance : ℝ := 72

/-- The speed difference between the bike and the truck -/
def speed_difference : ℝ := 0

/-- Theorem stating that the travel time for both vehicles is the same 
    but cannot be determined without knowing the speed -/
theorem indeterminate_equal_travel_time :
  ∃ (time : ℝ), 
    (∀ (bike_speed : ℝ), bike_speed > 0 →
      time = covered_distance / bike_speed) ∧
    (∀ (truck_speed : ℝ), truck_speed > 0 →
      time = covered_distance / truck_speed) ∧
    (∀ (bike_time truck_time : ℝ),
      bike_time = covered_distance / bike_speed →
      truck_time = covered_distance / truck_speed →
      bike_time = truck_time) :=
by sorry

end bike_truck_travel_time_indeterminate_equal_travel_time_l1270_127077


namespace megan_initial_files_l1270_127014

-- Define the problem parameters
def added_files : ℝ := 21.0
def files_per_folder : ℝ := 8.0
def final_folders : ℝ := 14.25

-- Define the theorem
theorem megan_initial_files :
  ∃ (initial_files : ℝ),
    (initial_files + added_files) / files_per_folder = final_folders ∧
    initial_files = 93.0 := by
  sorry

end megan_initial_files_l1270_127014


namespace quotient_problem_l1270_127015

theorem quotient_problem (q d1 d2 : ℝ) 
  (h1 : q = 6 * d1)  -- quotient is 6 times the dividend
  (h2 : q = 15 * d2) -- quotient is 15 times the divisor
  (h3 : d1 / d2 = q) -- definition of quotient
  : q = 2.5 := by
  sorry

end quotient_problem_l1270_127015


namespace monogram_cost_per_backpack_l1270_127083

/-- Proves the cost of monogramming each backpack --/
theorem monogram_cost_per_backpack 
  (num_backpacks : ℕ)
  (original_price : ℚ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (h1 : num_backpacks = 5)
  (h2 : original_price = 20)
  (h3 : discount_percent = 20 / 100)
  (h4 : total_cost = 140) :
  (total_cost - num_backpacks * (original_price * (1 - discount_percent))) / num_backpacks = 12 := by
  sorry

#check monogram_cost_per_backpack

end monogram_cost_per_backpack_l1270_127083


namespace first_phase_revenue_calculation_l1270_127096

/-- Represents a two-phase sales scenario -/
structure SalesScenario where
  total_purchase : ℝ
  first_markup : ℝ
  second_markup : ℝ
  total_revenue_increase : ℝ

/-- Calculates the revenue from the first phase of sales -/
def first_phase_revenue (s : SalesScenario) : ℝ :=
  sorry

/-- Theorem stating the first phase revenue for the given scenario -/
theorem first_phase_revenue_calculation (s : SalesScenario) 
  (h1 : s.total_purchase = 180000)
  (h2 : s.first_markup = 0.25)
  (h3 : s.second_markup = 0.16)
  (h4 : s.total_revenue_increase = 0.20) :
  first_phase_revenue s = 100000 :=
sorry

end first_phase_revenue_calculation_l1270_127096


namespace extreme_value_cubic_l1270_127064

/-- Given a cubic function f(x) = x^3 + ax^2 + bx + a^2 with an extreme value of 10 at x = 1,
    prove that f(2) = 18. -/
theorem extreme_value_cubic (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + a^2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧
  f 1 = 10 →
  f 2 = 18 := by
  sorry

end extreme_value_cubic_l1270_127064


namespace find_y_l1270_127089

theorem find_y (x y : ℤ) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := by
  sorry

end find_y_l1270_127089


namespace set_properties_l1270_127011

-- Define set A
def A : Set Int := {x | ∃ m n : Int, x = m^2 - n^2}

-- Define set B
def B : Set Int := {x | ∃ k : Int, x = 2*k + 1}

-- Theorem statement
theorem set_properties :
  (8 ∈ A ∧ 9 ∈ A ∧ 10 ∉ A) ∧
  (∀ x, x ∈ B → x ∈ A) ∧
  (∃ x, x ∈ A ∧ x ∉ B) ∧
  (∀ x, x ∈ A ∧ Even x ↔ ∃ k : Int, x = 4*k) :=
by sorry

end set_properties_l1270_127011


namespace exponential_equality_l1270_127034

theorem exponential_equality (x a b : ℝ) (h1 : 3^x = a) (h2 : 5^x = b) : 45^x = a^2 * b := by
  sorry

end exponential_equality_l1270_127034


namespace starting_lineup_count_l1270_127005

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def special_players : ℕ := 3

theorem starting_lineup_count : 
  (lineup_size.choose (team_size - special_players)) + 
  (special_players * (lineup_size - 1).choose (team_size - special_players)) = 2277 :=
by sorry

end starting_lineup_count_l1270_127005


namespace expression_evaluation_l1270_127007

theorem expression_evaluation : 
  (2002 : ℤ)^3 - 2001 * 2002^2 - 2001^2 * 2002 + 2001^3 + (2002 - 2001)^3 = 4004 := by
  sorry

end expression_evaluation_l1270_127007
