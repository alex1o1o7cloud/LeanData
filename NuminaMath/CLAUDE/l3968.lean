import Mathlib

namespace bicolored_angles_bound_l3968_396803

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A coloring of segments -/
def Coloring (n : ℕ) := Fin (n + 1) → Fin (n + 1) → Fin n

/-- The number of bicolored angles for a given coloring -/
def bicoloredAngles (n k : ℕ) (points : Fin (n + 1) → Point) (coloring : Coloring k) : ℕ :=
  sorry

/-- Three points are collinear if they lie on the same line -/
def collinear (p q r : Point) : Prop :=
  sorry

theorem bicolored_angles_bound (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 3) 
  (points : Fin (n + 1) → Point) 
  (h3 : ∀ (i j l : Fin (n + 1)), i ≠ j → j ≠ l → i ≠ l → ¬collinear (points i) (points j) (points l)) :
  ∃ (coloring : Coloring k), bicoloredAngles n k points coloring > n * (n / k)^2 * (k.choose 2) :=
sorry

end bicolored_angles_bound_l3968_396803


namespace candy_cost_proof_l3968_396830

/-- Represents the cost per pound of the second type of candy -/
def second_candy_cost : ℝ := 5

/-- Represents the weight of the first type of candy in pounds -/
def first_candy_weight : ℝ := 10

/-- Represents the cost per pound of the first type of candy -/
def first_candy_cost : ℝ := 8

/-- Represents the weight of the second type of candy in pounds -/
def second_candy_weight : ℝ := 20

/-- Represents the cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- Represents the total weight of the mixture in pounds -/
def total_weight : ℝ := first_candy_weight + second_candy_weight

theorem candy_cost_proof :
  first_candy_weight * first_candy_cost + second_candy_weight * second_candy_cost =
  total_weight * mixture_cost :=
by sorry

end candy_cost_proof_l3968_396830


namespace no_nonzero_solution_for_equation_l3968_396824

theorem no_nonzero_solution_for_equation (a b c d : ℤ) :
  a^2 + b^2 = 3*(c^2 + d^2) → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end no_nonzero_solution_for_equation_l3968_396824


namespace optimal_pan_dimensions_l3968_396868

def is_valid_pan (m n : ℕ) : Prop :=
  (m - 2) * (n - 2) = 2 * m + 2 * n - 4

def perimeter (m n : ℕ) : ℕ := 2 * m + 2 * n

def area (m n : ℕ) : ℕ := m * n

theorem optimal_pan_dimensions :
  ∀ m n : ℕ, m > 2 ∧ n > 2 → is_valid_pan m n →
    (perimeter m n ≥ perimeter 6 8) ∧
    (perimeter m n = perimeter 6 8 → area m n ≤ area 6 8) ∧
    is_valid_pan 6 8 :=
by sorry

end optimal_pan_dimensions_l3968_396868


namespace kates_savings_l3968_396858

/-- Kate's savings and purchases problem -/
theorem kates_savings (march april may june : ℕ) 
  (keyboard mouse headset video_game : ℕ) : 
  march = 27 → 
  april = 13 → 
  may = 28 → 
  june = 35 → 
  keyboard = 49 → 
  mouse = 5 → 
  headset = 15 → 
  video_game = 25 → 
  (march + april + may + june + 2 * april) - 
  (keyboard + mouse + headset + video_game) = 35 := by
  sorry

end kates_savings_l3968_396858


namespace cos_negative_45_degrees_l3968_396809

theorem cos_negative_45_degrees : Real.cos (-(45 * π / 180)) = 1 / Real.sqrt 2 := by sorry

end cos_negative_45_degrees_l3968_396809


namespace x_value_approximation_l3968_396882

/-- The value of x in the given equation is approximately 179692.08 -/
theorem x_value_approximation : 
  let x := 3.5 * ((3.6 * 0.48 * 2.50)^2 / (0.12 * 0.09 * 0.5)) * Real.log (2.5 * 4.3)
  ∃ ε > 0, |x - 179692.08| < ε :=
by sorry

end x_value_approximation_l3968_396882


namespace pen_collection_l3968_396896

theorem pen_collection (initial_pens : ℕ) (received_pens : ℕ) (given_away : ℕ) : 
  initial_pens = 5 → received_pens = 20 → given_away = 10 → 
  ((initial_pens + received_pens) * 2 - given_away) = 40 := by
  sorry

end pen_collection_l3968_396896


namespace equal_diagonal_quadrilateral_multiple_shapes_l3968_396811

/-- A quadrilateral with equal-length diagonals -/
structure EqualDiagonalQuadrilateral where
  /-- The length of the diagonals -/
  diagonal_length : ℝ
  /-- The quadrilateral has positive area -/
  positive_area : ℝ
  area_pos : positive_area > 0

/-- Possible shapes of a quadrilateral -/
inductive QuadrilateralShape
  | Square
  | Rectangle
  | IsoscelesTrapezoid
  | Other

/-- A function that determines if a given shape is possible for an equal-diagonal quadrilateral -/
def is_possible_shape (q : EqualDiagonalQuadrilateral) (shape : QuadrilateralShape) : Prop :=
  ∃ (quad : EqualDiagonalQuadrilateral), quad.diagonal_length = q.diagonal_length ∧ 
    quad.positive_area = q.positive_area ∧ 
    (match shape with
      | QuadrilateralShape.Square => true
      | QuadrilateralShape.Rectangle => true
      | QuadrilateralShape.IsoscelesTrapezoid => true
      | QuadrilateralShape.Other => true)

theorem equal_diagonal_quadrilateral_multiple_shapes (q : EqualDiagonalQuadrilateral) :
  (is_possible_shape q QuadrilateralShape.Square) ∧
  (is_possible_shape q QuadrilateralShape.Rectangle) ∧
  (is_possible_shape q QuadrilateralShape.IsoscelesTrapezoid) :=
sorry

end equal_diagonal_quadrilateral_multiple_shapes_l3968_396811


namespace olympic_medal_awards_l3968_396813

/-- The number of ways to award medals in the Olympic 100-meter finals -/
def medal_award_ways (total_sprinters : ℕ) (canadian_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_canadian_sprinters := total_sprinters - canadian_sprinters
  let no_canadian_medal := non_canadian_sprinters * (non_canadian_sprinters - 1) * (non_canadian_sprinters - 2)
  let one_canadian_medal := canadian_sprinters * medals * (non_canadian_sprinters) * (non_canadian_sprinters - 1)
  no_canadian_medal + one_canadian_medal

/-- Theorem: The number of ways to award medals in the given scenario is 480 -/
theorem olympic_medal_awards : medal_award_ways 10 4 3 = 480 := by
  sorry

end olympic_medal_awards_l3968_396813


namespace mandy_coin_value_l3968_396820

/-- Represents the number of cents in a coin -/
inductive Coin
| Dime : Coin
| Quarter : Coin

def coin_value : Coin → Nat
| Coin.Dime => 10
| Coin.Quarter => 25

/-- Represents Mandy's coin collection -/
structure CoinCollection where
  dimes : Nat
  quarters : Nat
  total_coins : Nat
  coin_balance : dimes + quarters = total_coins
  dime_quarter_relation : dimes + 2 = quarters

def collection_value (c : CoinCollection) : Nat :=
  c.dimes * coin_value Coin.Dime + c.quarters * coin_value Coin.Quarter

theorem mandy_coin_value :
  ∃ c : CoinCollection, c.total_coins = 17 ∧ collection_value c = 320 := by
  sorry

end mandy_coin_value_l3968_396820


namespace marble_probability_l3968_396819

/-- Given a bag of marbles with 5 red, 4 blue, and 6 yellow marbles,
    the probability of drawing one marble that is either red or blue is 3/5. -/
theorem marble_probability : 
  let red : ℕ := 5
  let blue : ℕ := 4
  let yellow : ℕ := 6
  let total : ℕ := red + blue + yellow
  let target : ℕ := red + blue
  (target : ℚ) / total = 3 / 5 := by sorry

end marble_probability_l3968_396819


namespace all_multiples_of_three_after_four_iterations_no_2020_on_tenth_page_l3968_396885

/-- Represents the numbers written by three schoolchildren on their notebooks. -/
structure SchoolchildrenNumbers where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Performs one iteration of the number writing process. -/
def iterate (nums : SchoolchildrenNumbers) : SchoolchildrenNumbers :=
  { a := nums.c - nums.b
  , b := nums.a - nums.c
  , c := nums.b - nums.a }

/-- Performs n iterations of the number writing process. -/
def iterateN (n : ℕ) (nums : SchoolchildrenNumbers) : SchoolchildrenNumbers :=
  match n with
  | 0 => nums
  | n + 1 => iterate (iterateN n nums)

/-- Theorem stating that after 4 iterations, all numbers are multiples of 3. -/
theorem all_multiples_of_three_after_four_iterations (initial : SchoolchildrenNumbers) :
  ∃ k l m : ℤ, 
    let result := iterateN 4 initial
    result.a = 3 * k ∧ result.b = 3 * l ∧ result.c = 3 * m :=
  sorry

/-- Theorem stating that 2020 cannot appear on the 10th page. -/
theorem no_2020_on_tenth_page (initial : SchoolchildrenNumbers) :
  let result := iterateN 9 initial
  result.a ≠ 2020 ∧ result.b ≠ 2020 ∧ result.c ≠ 2020 :=
  sorry

end all_multiples_of_three_after_four_iterations_no_2020_on_tenth_page_l3968_396885


namespace linear_expression_bounds_l3968_396844

/-- Given a system of equations and constraints, prove the bounds of a linear expression. -/
theorem linear_expression_bounds (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  x - 2*y - 3*z = -10 → 
  x + 2*y + z = 6 → 
  ∃ (A_min A_max : ℝ), 
    (∀ A, A = 1.5*x + y - z → A ≥ A_min ∧ A ≤ A_max) ∧
    A_min = -1 ∧ A_max = 0 := by
  sorry

end linear_expression_bounds_l3968_396844


namespace closest_integer_to_cube_root_l3968_396850

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 + 10 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| := by
  sorry

end closest_integer_to_cube_root_l3968_396850


namespace missing_village_population_l3968_396873

def village_population_problem (total_villages : Nat) 
                               (average_population : Nat) 
                               (known_populations : List Nat) : Prop :=
  total_villages = 7 ∧
  average_population = 1000 ∧
  known_populations = [803, 900, 1023, 945, 980, 1249] ∧
  known_populations.length = 6 ∧
  (List.sum known_populations + 1100) / total_villages = average_population

theorem missing_village_population :
  ∀ (total_villages : Nat) (average_population : Nat) (known_populations : List Nat),
  village_population_problem total_villages average_population known_populations →
  1100 = total_villages * average_population - List.sum known_populations :=
by
  sorry

end missing_village_population_l3968_396873


namespace stratified_sampling_result_l3968_396806

/-- Represents the number of residents in different age groups and the sampling size for one group -/
structure CommunityData where
  residents_35_to_45 : ℕ
  residents_46_to_55 : ℕ
  residents_56_to_65 : ℕ
  sampled_46_to_55 : ℕ

/-- Calculates the total number of people selected in a stratified sampling survey -/
def totalSampled (data : CommunityData) : ℕ :=
  (data.residents_35_to_45 + data.residents_46_to_55 + data.residents_56_to_65) / 
  (data.residents_46_to_55 / data.sampled_46_to_55)

/-- Theorem: Given the community data, the total number of people selected in the sampling survey is 140 -/
theorem stratified_sampling_result (data : CommunityData) 
  (h1 : data.residents_35_to_45 = 450)
  (h2 : data.residents_46_to_55 = 750)
  (h3 : data.residents_56_to_65 = 900)
  (h4 : data.sampled_46_to_55 = 50) :
  totalSampled data = 140 := by
  sorry

end stratified_sampling_result_l3968_396806


namespace remainder_after_adding_3006_l3968_396841

theorem remainder_after_adding_3006 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end remainder_after_adding_3006_l3968_396841


namespace complex_roots_equilateral_triangle_l3968_396890

theorem complex_roots_equilateral_triangle (p q z₁ z₂ : ℂ) :
  z₂^2 + p*z₂ + q = 0 →
  z₁^2 + p*z₁ + q = 0 →
  z₂ = Complex.exp (2*Real.pi*Complex.I/3) * z₁ →
  p^2 / q = 0 := by
  sorry

end complex_roots_equilateral_triangle_l3968_396890


namespace afternoon_campers_l3968_396804

theorem afternoon_campers (morning_campers : ℕ) (total_campers : ℕ) 
  (h1 : morning_campers = 15) 
  (h2 : total_campers = 32) : 
  total_campers - morning_campers = 17 := by
sorry

end afternoon_campers_l3968_396804


namespace max_concave_polygons_in_square_l3968_396812

/-- A concave polygon with sides parallel to a square's sides -/
structure ConcavePolygon where
  vertices : List (ℝ × ℝ)
  is_concave : Bool
  sides_parallel_to_square : Bool

/-- A square divided into concave polygons -/
structure DividedSquare where
  polygons : List ConcavePolygon
  no_parallel_translation : Bool

/-- The maximum number of equal concave polygons a square can be divided into -/
def max_concave_polygons : ℕ := 8

/-- Theorem stating the maximum number of equal concave polygons a square can be divided into -/
theorem max_concave_polygons_in_square :
  ∀ (d : DividedSquare),
    d.no_parallel_translation →
    (∀ p ∈ d.polygons, p.is_concave ∧ p.sides_parallel_to_square) →
    (List.length d.polygons ≤ max_concave_polygons) :=
by sorry

end max_concave_polygons_in_square_l3968_396812


namespace abs_five_implies_plus_minus_five_l3968_396864

theorem abs_five_implies_plus_minus_five (a : ℝ) : |a| = 5 → a = 5 ∨ a = -5 := by
  sorry

end abs_five_implies_plus_minus_five_l3968_396864


namespace transfer_equation_l3968_396878

theorem transfer_equation (x : ℤ) : 
  let initial_A : ℤ := 232
  let initial_B : ℤ := 146
  let final_A : ℤ := initial_A + x
  let final_B : ℤ := initial_B - x
  (final_A = 3 * final_B) ↔ (232 + x = 3 * (146 - x)) :=
by sorry

end transfer_equation_l3968_396878


namespace pure_imaginary_condition_l3968_396835

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (z : ℂ), z = m^2 - 1 + (m + 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = 1 := by
  sorry

end pure_imaginary_condition_l3968_396835


namespace davids_biology_marks_l3968_396839

theorem davids_biology_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (chemistry : ℕ) 
  (average : ℕ) 
  (h1 : english = 96) 
  (h2 : mathematics = 95) 
  (h3 : physics = 82) 
  (h4 : chemistry = 97) 
  (h5 : average = 93) 
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) : 
  biology = 95 := by
  sorry

end davids_biology_marks_l3968_396839


namespace paradise_park_ferris_wheel_small_seat_capacity_l3968_396892

/-- Represents the Ferris wheel in paradise park -/
structure FerrisWheel where
  small_seats : Nat
  large_seats : Nat
  small_seat_capacity : Nat

/-- Calculates the total capacity of small seats on the Ferris wheel -/
def total_small_seat_capacity (fw : FerrisWheel) : Nat :=
  fw.small_seats * fw.small_seat_capacity

theorem paradise_park_ferris_wheel_small_seat_capacity :
  ∃ (fw : FerrisWheel), 
    fw.small_seats = 2 ∧ 
    fw.large_seats = 23 ∧ 
    fw.small_seat_capacity = 14 ∧ 
    total_small_seat_capacity fw = 28 := by
  sorry

end paradise_park_ferris_wheel_small_seat_capacity_l3968_396892


namespace sunflower_height_l3968_396832

-- Define the height of Marissa's sister in inches
def sister_height_inches : ℕ := 4 * 12 + 3

-- Define the height difference between the sunflower and Marissa's sister
def height_difference : ℕ := 21

-- Theorem to prove the height of the sunflower
theorem sunflower_height :
  (sister_height_inches + height_difference) / 12 = 6 :=
by sorry

end sunflower_height_l3968_396832


namespace exists_odd_power_function_l3968_396807

/-- A function satisfying the given conditions -/
def special_function (f : ℕ → ℕ) : Prop :=
  (∀ m n : ℕ, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ, (m + n) ∣ (f m + f n))

/-- The main theorem -/
theorem exists_odd_power_function (f : ℕ → ℕ) (hf : special_function f) :
  ∃ k : ℕ, Odd k ∧ ∀ n : ℕ, f n = n^k :=
sorry

end exists_odd_power_function_l3968_396807


namespace square_sequence_20th_figure_l3968_396833

theorem square_sequence_20th_figure :
  let square_count : ℕ → ℕ := λ n => 2 * n^2 - 2 * n + 1
  (square_count 1 = 1) ∧
  (square_count 2 = 5) ∧
  (square_count 3 = 13) ∧
  (square_count 4 = 25) →
  square_count 20 = 761 :=
by
  sorry

end square_sequence_20th_figure_l3968_396833


namespace combined_share_proof_l3968_396848

/-- Proves that given $12,000 to be distributed among 5 children in the ratio 2 : 4 : 3 : 1 : 5,
    the combined share of the children with ratios 1 and 5 is $4,800. -/
theorem combined_share_proof (total_money : ℕ) (num_children : ℕ) (ratio : List ℕ) :
  total_money = 12000 →
  num_children = 5 →
  ratio = [2, 4, 3, 1, 5] →
  (ratio.sum * 800 = total_money) →
  (List.get! ratio 3 * 800 + List.get! ratio 4 * 800 = 4800) :=
by sorry

end combined_share_proof_l3968_396848


namespace simplify_fraction_1_simplify_fraction_2_l3968_396814

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1) : 
  1 / (a - 1) - a + 1 = (2*a - a^2) / (a - 1) := by sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  ((x + 2) / (x^2 - 2*x) - 1 / (x - 2)) / (2 / x) = 1 / (x - 2) := by sorry

end simplify_fraction_1_simplify_fraction_2_l3968_396814


namespace largest_gcd_of_sum_and_product_l3968_396893

theorem largest_gcd_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 1130)
  (prod_eq : x * y = 100000) :
  ∃ (a b : ℕ+), a + b = 1130 ∧ a * b = 100000 ∧ 
    ∀ (c d : ℕ+), c + d = 1130 → c * d = 100000 → Nat.gcd c d ≤ Nat.gcd a b ∧ Nat.gcd a b = 2 :=
by sorry

end largest_gcd_of_sum_and_product_l3968_396893


namespace sum_of_numbers_in_ratio_l3968_396849

/-- Given three numbers in ratio 5:7:9 with LCM 6300, their sum is 14700 -/
theorem sum_of_numbers_in_ratio (a b c : ℕ) : 
  (a : ℚ) / 5 = (b : ℚ) / 7 ∧ (b : ℚ) / 7 = (c : ℚ) / 9 →
  Nat.lcm a (Nat.lcm b c) = 6300 →
  a + b + c = 14700 := by
sorry

end sum_of_numbers_in_ratio_l3968_396849


namespace difference_of_squares_601_597_l3968_396889

theorem difference_of_squares_601_597 : 601^2 - 597^2 = 4792 := by
  sorry

end difference_of_squares_601_597_l3968_396889


namespace area_of_enclosed_region_l3968_396825

/-- The curve defined by the equation x^2 + y^2 = 2(|x| + |y|) -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * (abs x + abs y)

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of the region enclosed by the curve x^2 + y^2 = 2(|x| + |y|) is 2π -/
theorem area_of_enclosed_region : area enclosed_region = 2 * Real.pi := by sorry

end area_of_enclosed_region_l3968_396825


namespace negation_of_existence_is_universal_l3968_396877

variable (a : ℝ)

theorem negation_of_existence_is_universal :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end negation_of_existence_is_universal_l3968_396877


namespace augmented_matrix_solution_sum_l3968_396810

/-- Given an augmented matrix representing a system of linear equations,
    if the solution exists, then the sum of certain elements in the matrix is determined. -/
theorem augmented_matrix_solution_sum (m n : ℝ) : 
  (∃ x y : ℝ, m * x = 6 ∧ 3 * y = n ∧ x = -3 ∧ y = 4) → m + n = 10 := by
  sorry

end augmented_matrix_solution_sum_l3968_396810


namespace rectangular_shingle_area_l3968_396834

/-- The area of a rectangular roof shingle -/
theorem rectangular_shingle_area (length width : ℝ) (h1 : length = 10) (h2 : width = 7) :
  length * width = 70 := by
  sorry

end rectangular_shingle_area_l3968_396834


namespace fraction_division_specific_fraction_division_l3968_396894

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((4 : ℚ) / 5) = 15 / 28 :=
by sorry

end fraction_division_specific_fraction_division_l3968_396894


namespace decreasing_function_inequality_l3968_396862

/-- A function f is decreasing on ℝ if for all x y, x < y implies f x > f y -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingOn f) (h_inequality : f (3 * a) < f (-2 * a + 10)) : 
  a > 2 := by
  sorry

end decreasing_function_inequality_l3968_396862


namespace candy_difference_l3968_396884

/- Define the number of candies each person can eat -/
def nellie_candies : ℕ := 12
def jacob_candies : ℕ := nellie_candies / 2
def lana_candies : ℕ := jacob_candies - 3

/- Define the total number of candies in the bucket -/
def total_candies : ℕ := 30

/- Define the number of remaining candies after they ate -/
def remaining_candies : ℕ := 9

/- Theorem statement -/
theorem candy_difference :
  jacob_candies - lana_candies = 3 :=
by sorry

end candy_difference_l3968_396884


namespace max_value_of_sin_cos_combination_l3968_396880

/-- The function f(x) = 3 sin x + 4 cos x has a maximum value of 5 -/
theorem max_value_of_sin_cos_combination :
  ∃ (M : ℝ), M = 5 ∧ ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ M :=
by sorry

end max_value_of_sin_cos_combination_l3968_396880


namespace lizette_overall_average_l3968_396866

def average_first_two_quizzes : ℝ := 95
def third_quiz_score : ℝ := 92
def number_of_quizzes : ℕ := 3

theorem lizette_overall_average :
  (average_first_two_quizzes * 2 + third_quiz_score) / number_of_quizzes = 94 := by
  sorry

end lizette_overall_average_l3968_396866


namespace arithmetic_sequence_product_l3968_396875

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1 : ℝ) * seq.d

theorem arithmetic_sequence_product (seq : ArithmeticSequence) 
  (h1 : seq.nthTerm 8 = 20)
  (h2 : seq.d = 2) :
  seq.nthTerm 2 * seq.nthTerm 3 = 80 := by
  sorry

#check arithmetic_sequence_product

end arithmetic_sequence_product_l3968_396875


namespace simplify_expression_l3968_396843

theorem simplify_expression (y : ℝ) : 
  5 * y - 6 * y^2 + 9 - (4 - 5 * y + 2 * y^2) = -8 * y^2 + 10 * y + 5 := by
  sorry

end simplify_expression_l3968_396843


namespace at_least_one_angle_not_greater_than_60_l3968_396827

-- Define a triangle as a triple of angles
def Triangle := (ℝ × ℝ × ℝ)

-- Define a predicate for a valid triangle (sum of angles is 180°)
def is_valid_triangle (t : Triangle) : Prop :=
  let (a, b, c) := t
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

-- Theorem statement
theorem at_least_one_angle_not_greater_than_60 (t : Triangle) 
  (h : is_valid_triangle t) : 
  ∃ θ, θ ∈ [t.1, t.2.1, t.2.2] ∧ θ ≤ 60 := by
  sorry

end at_least_one_angle_not_greater_than_60_l3968_396827


namespace parallel_segment_length_l3968_396847

theorem parallel_segment_length (base : ℝ) (a b c : ℝ) :
  base = 18 →
  a + b + c = 1 →
  a = (1/4 : ℝ) →
  b = (1/2 : ℝ) →
  c = (1/4 : ℝ) →
  ∃ (middle_segment : ℝ), middle_segment = 9 * Real.sqrt 3 :=
by sorry

end parallel_segment_length_l3968_396847


namespace sqrt_difference_equals_one_l3968_396886

theorem sqrt_difference_equals_one : Real.sqrt 25 - Real.sqrt 16 = 1 := by
  sorry

end sqrt_difference_equals_one_l3968_396886


namespace fraction_subtraction_l3968_396852

theorem fraction_subtraction : 
  (((3 : ℚ) + 6 + 9) / ((2 : ℚ) + 5 + 8) - ((2 : ℚ) + 5 + 8) / ((3 : ℚ) + 6 + 9)) = 11 / 30 := by
  sorry

end fraction_subtraction_l3968_396852


namespace largest_five_digit_congruent_to_17_mod_29_l3968_396861

theorem largest_five_digit_congruent_to_17_mod_29 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n ≡ 17 [MOD 29] → n ≤ 99982 :=
by sorry

end largest_five_digit_congruent_to_17_mod_29_l3968_396861


namespace complex_modulus_problem_l3968_396817

theorem complex_modulus_problem : Complex.abs (Complex.I / (1 - Complex.I)) = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l3968_396817


namespace thursday_tea_consumption_l3968_396859

/-- Represents the relationship between hours grading and liters of tea consumed -/
structure TeaGrading where
  hours : ℝ
  liters : ℝ
  inv_prop : hours * liters = hours * liters

/-- The constant of proportionality derived from Wednesday's data -/
def wednesday_constant : ℝ := 5 * 4

/-- Theorem stating that given Wednesday's data and Thursday's hours, the teacher drinks 2.5 liters of tea on Thursday -/
theorem thursday_tea_consumption (wednesday : TeaGrading) (thursday : TeaGrading) 
    (h_wednesday : wednesday.hours = 5 ∧ wednesday.liters = 4)
    (h_thursday : thursday.hours = 8)
    (h_constant : wednesday.hours * wednesday.liters = thursday.hours * thursday.liters) :
    thursday.liters = 2.5 := by
  sorry

end thursday_tea_consumption_l3968_396859


namespace zou_win_probability_l3968_396831

/-- Represents the outcome of a race -/
inductive RaceOutcome
| Win
| Loss

/-- Calculates the probability of winning a race given the previous outcome -/
def winProbability (previousOutcome : RaceOutcome) : ℚ :=
  match previousOutcome with
  | RaceOutcome.Win => 2/3
  | RaceOutcome.Loss => 1/3

/-- Represents a sequence of race outcomes -/
def RaceSequence := List RaceOutcome

/-- Calculates the probability of a given race sequence -/
def sequenceProbability (sequence : RaceSequence) : ℚ :=
  sequence.foldl (fun acc outcome => acc * winProbability outcome) 1

/-- Generates all possible race sequences where Zou wins exactly 5 out of 6 races -/
def winningSequences : List RaceSequence := sorry

theorem zou_win_probability :
  let totalProbability := (winningSequences.map sequenceProbability).sum
  totalProbability = 80/243 := by sorry

end zou_win_probability_l3968_396831


namespace circles_externally_tangent_l3968_396872

/-- Two circles are externally tangent when the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

theorem circles_externally_tangent :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 3
  let d : ℝ := 5
  externally_tangent r₁ r₂ d := by sorry

end circles_externally_tangent_l3968_396872


namespace geometric_sequence_ratio_l3968_396856

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 3 * a 7 = 6)
  (h_sum : a 2 + a 8 = 5) :
  a 10 / a 4 = 3/2 :=
sorry

end geometric_sequence_ratio_l3968_396856


namespace arithmetic_sequence_sum_6_l3968_396801

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  s : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, s n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: For an arithmetic sequence with given conditions, S_6 = 6 -/
theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) 
    (h1 : seq.a 1 = 6)
    (h2 : seq.a 3 + seq.a 5 = 0) : 
  seq.s 6 = 6 := by
  sorry

end arithmetic_sequence_sum_6_l3968_396801


namespace largest_equal_cost_number_l3968_396829

/-- Calculates the cost of transmitting a number using Option 1 (decimal representation) -/
def option1Cost (n : Nat) : Nat :=
  sorry

/-- Calculates the cost of transmitting a number using Option 2 (binary representation) -/
def option2Cost (n : Nat) : Nat :=
  sorry

/-- Checks if the costs are equal for both options -/
def costsEqual (n : Nat) : Bool :=
  option1Cost n = option2Cost n

/-- Theorem stating that 1118 is the largest number less than 2000 with equal costs -/
theorem largest_equal_cost_number :
  (∀ n : Nat, n < 2000 → costsEqual n → n ≤ 1118) ∧ costsEqual 1118 := by
  sorry

end largest_equal_cost_number_l3968_396829


namespace complex_square_sum_l3968_396840

theorem complex_square_sum (a b : ℝ) (h : (1 + Complex.I)^2 = Complex.mk a b) : a + b = 2 := by
  sorry

end complex_square_sum_l3968_396840


namespace permutation_equation_solution_l3968_396826

/-- Permutation function: number of ways to arrange k items out of m items -/
def A (m : ℕ) (k : ℕ) : ℕ := m.factorial / (m - k).factorial

/-- The theorem states that the equation 3A₈ⁿ⁻¹ = 4A₉ⁿ⁻² is satisfied when n = 9 -/
theorem permutation_equation_solution :
  ∃ n : ℕ, 3 * A 8 (n - 1) = 4 * A 9 (n - 2) ∧ n = 9 := by
  sorry

end permutation_equation_solution_l3968_396826


namespace clever_cat_academy_count_l3968_396871

theorem clever_cat_academy_count :
  let jump : ℕ := 60
  let spin : ℕ := 35
  let fetch : ℕ := 40
  let jump_and_spin : ℕ := 25
  let spin_and_fetch : ℕ := 20
  let jump_and_fetch : ℕ := 22
  let all_three : ℕ := 12
  let none : ℕ := 10
  jump + spin + fetch - jump_and_spin - spin_and_fetch - jump_and_fetch + all_three + none = 92 :=
by sorry

end clever_cat_academy_count_l3968_396871


namespace product_one_when_equal_absolute_log_l3968_396845

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem product_one_when_equal_absolute_log 
  (a b : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : f a = f b) : 
  a * b = 1 := by
  sorry

end product_one_when_equal_absolute_log_l3968_396845


namespace integer_solutions_equation_l3968_396802

theorem integer_solutions_equation :
  ∀ (a b c : ℤ),
    c ≤ 94 →
    (a + Real.sqrt c)^2 + (b + Real.sqrt c)^2 = 60 + 20 * Real.sqrt c →
    ((a = 3 ∧ b = 7 ∧ c = 41) ∨
     (a = 4 ∧ b = 6 ∧ c = 44) ∨
     (a = 5 ∧ b = 5 ∧ c = 45) ∨
     (a = 6 ∧ b = 4 ∧ c = 44) ∨
     (a = 7 ∧ b = 3 ∧ c = 41)) :=
by sorry

end integer_solutions_equation_l3968_396802


namespace repeating_decimal_as_fraction_l3968_396869

/-- Represents the repeating decimal 0.53246246246... -/
def repeating_decimal : ℚ := 0.53 + (0.246 / 999)

/-- The denominator of the target fraction -/
def target_denominator : ℕ := 999900

theorem repeating_decimal_as_fraction :
  ∃ x : ℕ, (x : ℚ) / target_denominator = repeating_decimal ∧ x = 531714 := by
  sorry

end repeating_decimal_as_fraction_l3968_396869


namespace average_weight_solution_l3968_396853

def average_weight_problem (a b c : ℝ) : Prop :=
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43 ∧
  b = 31

theorem average_weight_solution :
  ∀ a b c : ℝ, average_weight_problem a b c → (a + b + c) / 3 = 45 := by
  sorry

end average_weight_solution_l3968_396853


namespace max_value_of_product_l3968_396897

theorem max_value_of_product (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a + b = 4) :
  (∀ x y : ℝ, x > 1 → y > 1 → x + y = 4 → (x - 1) * (y - 1) ≤ (a - 1) * (b - 1)) →
  (a - 1) * (b - 1) = 1 :=
by sorry

end max_value_of_product_l3968_396897


namespace no_linear_term_implies_m_value_l3968_396800

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) → m = -3 := by
  sorry

end no_linear_term_implies_m_value_l3968_396800


namespace initial_money_theorem_l3968_396828

def meat_cost : ℝ := 17
def chicken_cost : ℝ := 22
def veggie_cost : ℝ := 43
def egg_cost : ℝ := 5
def dog_food_cost : ℝ := 45
def cat_food_cost : ℝ := 18
def discount_rate : ℝ := 0.1
def money_left : ℝ := 35

def total_spent : ℝ := meat_cost + chicken_cost + veggie_cost + egg_cost + dog_food_cost + (cat_food_cost * (1 - discount_rate))

theorem initial_money_theorem :
  total_spent + money_left = 183.20 := by
  sorry

end initial_money_theorem_l3968_396828


namespace sector_central_angle_l3968_396808

theorem sector_central_angle (circumference area : ℝ) (h1 : circumference = 4) (h2 : area = 1) :
  let r := (4 - circumference) / 2
  let l := circumference - 2 * r
  l / r = 2 := by sorry

end sector_central_angle_l3968_396808


namespace triangle_side_length_l3968_396863

-- Define a triangle XYZ
structure Triangle :=
  (x y z : ℝ)
  (X Y Z : ℝ)

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (hy : t.y = 7)
  (hz : t.z = 6)
  (hcos : Real.cos (t.Y - t.Z) = 1/2) :
  t.x = Real.sqrt 73 := by
  sorry

end triangle_side_length_l3968_396863


namespace decreasing_function_implies_b_geq_4_l3968_396883

-- Define the function y
def y (x b : ℝ) : ℝ := x^3 - 3*b*x + 1

-- State the theorem
theorem decreasing_function_implies_b_geq_4 :
  ∀ b : ℝ, (∀ x ∈ Set.Ioo 1 2, ∀ h > 0, x + h ∈ Set.Ioo 1 2 → y (x + h) b < y x b) →
  b ≥ 4 := by sorry

end decreasing_function_implies_b_geq_4_l3968_396883


namespace average_of_combined_results_l3968_396899

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) 
  (h₁ : n₁ = 55) (h₂ : n₂ = 28) (h₃ : avg₁ = 28) (h₄ : avg₂ = 55) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = (55 * 28 + 28 * 55) / (55 + 28) := by
sorry

end average_of_combined_results_l3968_396899


namespace jack_quarantine_days_l3968_396805

/-- Calculates the number of days spent in quarantine given the total wait time and customs time. -/
def quarantine_days (total_hours : ℕ) (customs_hours : ℕ) : ℕ :=
  (total_hours - customs_hours) / 24

/-- Theorem stating that given a total wait time of 356 hours, including 20 hours for customs,
    the number of days spent in quarantine is 14. -/
theorem jack_quarantine_days :
  quarantine_days 356 20 = 14 := by
  sorry

end jack_quarantine_days_l3968_396805


namespace cara_charge_account_l3968_396846

/-- Represents the simple interest calculation for Cara's charge account --/
theorem cara_charge_account (initial_charge : ℝ) : 
  initial_charge * (1 + 0.05) = 56.7 → initial_charge = 54 := by
  sorry

end cara_charge_account_l3968_396846


namespace watch_angle_difference_l3968_396822

/-- Represents the angle between the hour and minute hands of a watch -/
def watchAngle (hours minutes : ℝ) : ℝ :=
  |30 * hours - 5.5 * minutes|

/-- Theorem stating that the time difference between two 120° angles of watch hands between 7:00 PM and 8:00 PM is 30 minutes -/
theorem watch_angle_difference : ∃ (t₁ t₂ : ℝ),
  0 < t₁ ∧ t₁ < t₂ ∧ t₂ < 60 ∧
  watchAngle (7 + t₁ / 60) t₁ = 120 ∧
  watchAngle (7 + t₂ / 60) t₂ = 120 ∧
  t₂ - t₁ = 30 := by
  sorry

end watch_angle_difference_l3968_396822


namespace matrix_equation_solution_l3968_396876

def B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 0, 1, 2; 1, 0, 1]

theorem matrix_equation_solution :
  B^3 + (-5 : ℚ) • B^2 + 3 • B + (-6 : ℚ) • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 := by
  sorry

end matrix_equation_solution_l3968_396876


namespace company_n_profit_change_l3968_396816

def CompanyN (R : ℝ) : Prop :=
  let profit1998 := 0.10 * R
  let revenue1999 := 0.70 * R
  let profit1999 := 0.15 * revenue1999
  let revenue2000 := 1.20 * revenue1999
  let profit2000 := 0.18 * revenue2000
  let percentageChange := (profit2000 - profit1998) / profit1998 * 100
  percentageChange = 51.2

theorem company_n_profit_change (R : ℝ) (h : R > 0) : CompanyN R := by
  sorry

end company_n_profit_change_l3968_396816


namespace angle_measure_in_special_triangle_l3968_396818

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + bc + c², then the measure of angle A is 120°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a^2 = b^2 + b*c + c^2 →
  A = 2 * Real.pi / 3 :=
by sorry

end angle_measure_in_special_triangle_l3968_396818


namespace profit_doubling_l3968_396895

theorem profit_doubling (cost : ℝ) (price : ℝ) (h1 : price = 1.5 * cost) :
  let double_price := 2 * price
  (double_price - cost) / cost * 100 = 200 :=
by sorry

end profit_doubling_l3968_396895


namespace second_agency_daily_charge_proof_l3968_396838

/-- The daily charge of the first agency in dollars -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first agency in dollars -/
def first_agency_mile_charge : ℝ := 0.14

/-- The per-mile charge of the second agency in dollars -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the agencies' costs are equal -/
def equal_cost_miles : ℝ := 25.0

/-- The daily charge of the second agency in dollars -/
def second_agency_daily_charge : ℝ := 18.25

theorem second_agency_daily_charge_proof :
  first_agency_daily_charge + first_agency_mile_charge * equal_cost_miles =
  second_agency_daily_charge + second_agency_mile_charge * equal_cost_miles :=
by sorry

end second_agency_daily_charge_proof_l3968_396838


namespace functional_equation_solution_l3968_396879

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + y * f x

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) (h2 : f 1 = 3) : f 501 = 503 := by
  sorry

end functional_equation_solution_l3968_396879


namespace symmetry_axis_implies_phi_l3968_396821

theorem symmetry_axis_implies_phi (φ : ℝ) : 
  (∀ x, 2 * Real.sin (3 * x + φ) = 2 * Real.sin (3 * (π / 6 - x) + φ)) →
  |φ| < π / 2 →
  φ = π / 4 := by
sorry

end symmetry_axis_implies_phi_l3968_396821


namespace difference_of_squares_special_case_l3968_396842

theorem difference_of_squares_special_case : (532 * 532) - (531 * 533) = 1 := by
  sorry

end difference_of_squares_special_case_l3968_396842


namespace leaps_per_meter_calculation_l3968_396867

/-- Represents the number of leaps in one meter given the relationships between strides, leaps, bounds, and meters. -/
def leaps_per_meter (x y z w u v : ℚ) : ℚ :=
  (u * w) / (v * z)

/-- Theorem stating that given the relationships between units, one meter equals (uw/vz) leaps. -/
theorem leaps_per_meter_calculation
  (x y z w u v : ℚ)
  (h1 : x * 1 = y)  -- x strides = y leaps
  (h2 : z * 1 = w)  -- z bounds = w leaps
  (h3 : u * 1 = v)  -- u bounds = v meters
  : leaps_per_meter x y z w u v = (u * w) / (v * z) := by
  sorry

#check leaps_per_meter_calculation

end leaps_per_meter_calculation_l3968_396867


namespace game_draw_fraction_l3968_396851

theorem game_draw_fraction (jack_win : ℚ) (emma_win : ℚ) 
  (h1 : jack_win = 4/9) (h2 : emma_win = 5/14) : 
  1 - (jack_win + emma_win) = 25/126 := by
  sorry

end game_draw_fraction_l3968_396851


namespace marker_sale_savings_l3968_396857

/-- Calculates the savings when buying markers during a sale --/
def calculate_savings (original_price : ℚ) (num_markers : ℕ) (discount_rate : ℚ) : ℚ :=
  let original_total := original_price * num_markers
  let discounted_price := original_price * (1 - discount_rate)
  let free_markers := num_markers / 4
  let effective_markers := num_markers + free_markers
  let sale_total := discounted_price * num_markers
  original_total - sale_total

theorem marker_sale_savings :
  let original_price : ℚ := 3
  let num_markers : ℕ := 8
  let discount_rate : ℚ := 0.3
  calculate_savings original_price num_markers discount_rate = 36/5 := by sorry

end marker_sale_savings_l3968_396857


namespace square_of_number_ending_in_five_l3968_396874

theorem square_of_number_ending_in_five (a : ℤ) : (10 * a + 5)^2 = 100 * a * (a + 1) + 25 := by
  sorry

end square_of_number_ending_in_five_l3968_396874


namespace assistant_professor_pencils_l3968_396887

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by
  sorry

end assistant_professor_pencils_l3968_396887


namespace algebraic_expression_equality_l3968_396891

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → 3*x^2 + 9*x - 2 = 4 := by
  sorry

end algebraic_expression_equality_l3968_396891


namespace angelina_walk_speeds_l3968_396854

theorem angelina_walk_speeds (v : ℝ) :
  v > 0 ∧
  960 / v - 40 = 480 / (2 * v) ∧
  480 / (2 * v) - 20 = 720 / (3 * v) →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by sorry

end angelina_walk_speeds_l3968_396854


namespace largest_gcd_of_sum_1023_l3968_396888

theorem largest_gcd_of_sum_1023 :
  ∃ (a b : ℕ+), a + b = 1023 ∧
  ∀ (c d : ℕ+), c + d = 1023 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 341 :=
by sorry

end largest_gcd_of_sum_1023_l3968_396888


namespace bea_highest_profit_l3968_396865

/-- Represents a lemonade seller with their sales information -/
structure LemonadeSeller where
  name : String
  price : ℕ
  soldGlasses : ℕ
  variableCost : ℕ

/-- Calculates the profit for a lemonade seller -/
def calculateProfit (seller : LemonadeSeller) : ℕ :=
  seller.price * seller.soldGlasses - seller.variableCost * seller.soldGlasses

/-- Theorem stating that Bea makes the most profit -/
theorem bea_highest_profit (bea dawn carla : LemonadeSeller)
  (h_bea : bea = { name := "Bea", price := 25, soldGlasses := 10, variableCost := 10 })
  (h_dawn : dawn = { name := "Dawn", price := 28, soldGlasses := 8, variableCost := 12 })
  (h_carla : carla = { name := "Carla", price := 35, soldGlasses := 6, variableCost := 15 }) :
  calculateProfit bea ≥ calculateProfit dawn ∧ calculateProfit bea ≥ calculateProfit carla :=
by sorry

end bea_highest_profit_l3968_396865


namespace muffin_buyers_count_l3968_396860

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_buyers : ℕ := 50

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- Define the probability of selecting a buyer who purchases neither cake nor muffin mix
def prob_neither : ℚ := 29/100

-- Theorem to prove
theorem muffin_buyers_count : 
  ∃ (muffin_buyers : ℕ), 
    muffin_buyers = total_buyers - cake_buyers - (total_buyers * prob_neither).num + both_buyers := by
  sorry

end muffin_buyers_count_l3968_396860


namespace number_properties_l3968_396837

def is_even (n : ℕ) := n % 2 = 0
def is_odd (n : ℕ) := n % 2 ≠ 0
def is_prime (n : ℕ) := n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)
def is_composite (n : ℕ) := n > 1 ∧ ¬(is_prime n)

theorem number_properties :
  (∀ n : ℕ, n ≤ 10 → (is_even n ∧ ¬is_composite n) → n = 2) ∧
  (∀ n : ℕ, n ≤ 10 → (is_odd n ∧ ¬is_prime n) → n = 1) ∧
  (∀ n : ℕ, n ≤ 10 → (is_odd n ∧ is_composite n) → n = 9) ∧
  (∀ n : ℕ, is_prime n → n ≥ 2) ∧
  (∀ n : ℕ, is_composite n → n ≥ 4) :=
by sorry

end number_properties_l3968_396837


namespace complex_fourth_power_l3968_396881

theorem complex_fourth_power (z : ℂ) : z = Complex.I * Real.sqrt 2 → z^4 = 4 := by
  sorry

end complex_fourth_power_l3968_396881


namespace power_of_negative_cube_l3968_396898

theorem power_of_negative_cube (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := by
  sorry

end power_of_negative_cube_l3968_396898


namespace school_teachers_calculation_l3968_396870

/-- Calculates the number of teachers required in a school given specific conditions -/
theorem school_teachers_calculation (total_students : ℕ) (lessons_per_student : ℕ) 
  (lessons_per_teacher : ℕ) (students_per_class : ℕ) : 
  total_students = 1200 →
  lessons_per_student = 5 →
  lessons_per_teacher = 4 →
  students_per_class = 30 →
  (total_students * lessons_per_student) / (students_per_class * lessons_per_teacher) = 50 := by
  sorry

#check school_teachers_calculation

end school_teachers_calculation_l3968_396870


namespace complex_product_equality_complex_product_equality_proof_l3968_396836

theorem complex_product_equality : Complex → Prop :=
  fun i =>
    i * i = -1 →
    (1 + i) * (2 - i) = 3 + i

-- The proof is omitted
theorem complex_product_equality_proof : complex_product_equality Complex.I :=
  sorry

end complex_product_equality_complex_product_equality_proof_l3968_396836


namespace factors_of_81_l3968_396855

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end factors_of_81_l3968_396855


namespace triangle_problem_l3968_396815

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The main theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * Real.sin t.A = t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = Real.sqrt 3 * Real.sin t.A) : 
  t.B = π / 6 ∧ t.a = 3 ∧ t.c = 3 * Real.sqrt 3 := by
  sorry


end triangle_problem_l3968_396815


namespace six_digit_divisibility_l3968_396823

theorem six_digit_divisibility (a b c : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) (h4 : c ≤ 9) :
  ∃ k : Nat, (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c) = 143 * k := by
  sorry

end six_digit_divisibility_l3968_396823
