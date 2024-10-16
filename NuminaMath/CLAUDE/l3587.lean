import Mathlib

namespace NUMINAMATH_CALUDE_unique_four_digit_reverse_l3587_358709

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem unique_four_digit_reverse : ∃! n : ℕ, is_four_digit n ∧ n + 8802 = reverse_digits n :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_reverse_l3587_358709


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l3587_358708

theorem power_function_not_through_origin (m : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (m^2 - 3*m + 3) * x^(m^2 - m - 2) ≠ 0) →
  m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l3587_358708


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l3587_358792

/-- Proves that the number of meters of cloth sold is 45 given the specified conditions -/
theorem cloth_sale_problem (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  total_selling_price = 4500 →
  profit_per_meter = 14 →
  cost_price_per_meter = 86 →
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 45 := by
  sorry

#check cloth_sale_problem

end NUMINAMATH_CALUDE_cloth_sale_problem_l3587_358792


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3587_358799

theorem quadratic_solution_sum (c d : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 2 * x + 8 = 0 ↔ x = c + d * I ∨ x = c - d * I) → 
  c + d^2 = 44/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3587_358799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3587_358783

/-- Given an arithmetic sequence {a_n} with a_1 = -2 and S_3 = 0, 
    where S_n is the sum of the first n terms, 
    prove that the common difference is 2. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, S n = (n : ℚ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →  -- Definition of S_n
  a 1 = -2 →                                                     -- a_1 = -2
  S 3 = 0 →                                                      -- S_3 = 0
  a 2 - a 1 = 2 :=                                               -- Common difference is 2
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3587_358783


namespace NUMINAMATH_CALUDE_power_zero_eq_one_negative_two_power_zero_l3587_358701

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem negative_two_power_zero : (-2 : ℝ)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_negative_two_power_zero_l3587_358701


namespace NUMINAMATH_CALUDE_rabbit_cat_age_ratio_l3587_358700

/-- Given the ages of a cat, dog, and rabbit, prove the ratio of rabbit's age to cat's age --/
theorem rabbit_cat_age_ratio 
  (cat_age : ℕ) 
  (dog_age : ℕ) 
  (rabbit_age : ℕ) 
  (h1 : cat_age = 8) 
  (h2 : dog_age = 12) 
  (h3 : rabbit_age * 3 = dog_age) : 
  (rabbit_age : ℚ) / cat_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_cat_age_ratio_l3587_358700


namespace NUMINAMATH_CALUDE_minimize_sum_distances_on_x_axis_l3587_358790

/-- The point that minimizes the sum of distances to two given points -/
def minimize_sum_distances (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem minimize_sum_distances_on_x_axis 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h_A : A = (-1, 2)) 
  (h_B : B = (2, 1)) :
  minimize_sum_distances A B = (1, 0) :=
sorry

end NUMINAMATH_CALUDE_minimize_sum_distances_on_x_axis_l3587_358790


namespace NUMINAMATH_CALUDE_problem_solution_l3587_358723

theorem problem_solution : (1 / (Real.sqrt 2 + 1) - Real.sqrt 8 + (Real.sqrt 3 + 1) ^ 0) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3587_358723


namespace NUMINAMATH_CALUDE_petya_vasya_game_l3587_358731

theorem petya_vasya_game (k : ℚ) : ∃ (a b c : ℚ), 
  ∃ (x y : ℂ), x ≠ y ∧ 
  (x^3 + a*x^2 + b*x + c = 0) ∧ 
  (y^3 + a*y^2 + b*y + c = 0) ∧ 
  (x - y = 2014 ∨ y - x = 2014) :=
sorry

end NUMINAMATH_CALUDE_petya_vasya_game_l3587_358731


namespace NUMINAMATH_CALUDE_divisibility_property_l3587_358713

theorem divisibility_property (n : ℕ) : 
  ∃ (x y : ℤ), (x^2 + y^2 - 2018) % n = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3587_358713


namespace NUMINAMATH_CALUDE_square_roots_problem_l3587_358706

theorem square_roots_problem (m : ℝ) :
  (2*m - 4)^2 = (3*m - 1)^2 → (2*m - 4)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3587_358706


namespace NUMINAMATH_CALUDE_complement_I_M_l3587_358791

def M : Set ℕ := {0, 1}
def I : Set ℕ := {0, 1, 2, 3, 4, 5}

theorem complement_I_M : (I \ M) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_I_M_l3587_358791


namespace NUMINAMATH_CALUDE_double_negation_l3587_358740

theorem double_negation (n : ℤ) : -(-n) = n := by
  sorry

end NUMINAMATH_CALUDE_double_negation_l3587_358740


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l3587_358745

theorem min_value_a_squared_plus_b_squared :
  ∀ a b : ℝ,
  ((-2)^2 + a*(-2) + 2*b = 0) →
  ∀ c d : ℝ,
  (c^2 + d^2 ≥ a^2 + b^2) →
  (a^2 + b^2 ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_b_squared_l3587_358745


namespace NUMINAMATH_CALUDE_line_equation_l3587_358733

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallelism
def parallel (a b c d e f : ℝ) : Prop := a*e = b*d

-- Define the line l
def line_l (x y : ℝ) : Prop := 4*x - 3*y - 6 = 0

-- Theorem statement
theorem line_equation : 
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧ 
  parallel 4 (-3) (-6) 4 (-3) (-7) ∧
  line_l x₀ y₀ := by sorry

end NUMINAMATH_CALUDE_line_equation_l3587_358733


namespace NUMINAMATH_CALUDE_parallel_transitive_l3587_358781

-- Define a type for lines in a plane
variable {Line : Type}

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_transitive (A B C : Line) :
  parallel A C → parallel B C → parallel A B :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l3587_358781


namespace NUMINAMATH_CALUDE_olivia_chocolate_sales_l3587_358777

/-- Calculates the money made from selling chocolate bars --/
def money_made (cost_per_bar : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * cost_per_bar

/-- Proves that Olivia made $9 from selling chocolate bars --/
theorem olivia_chocolate_sales : money_made 3 7 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_olivia_chocolate_sales_l3587_358777


namespace NUMINAMATH_CALUDE_binary_1010101_equals_octal_125_l3587_358705

/-- Converts a binary number represented as a list of bits to a natural number -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

/-- The binary representation of 1010101₂ -/
def binary_1010101 : List Bool := [true, false, true, false, true, false, true]

/-- The octal representation of 125₈ -/
def octal_125 : List ℕ := [5, 2, 1]

theorem binary_1010101_equals_octal_125 :
  natural_to_octal (binary_to_natural binary_1010101) = octal_125 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_equals_octal_125_l3587_358705


namespace NUMINAMATH_CALUDE_three_cubes_volume_and_area_l3587_358762

/-- Calculates the volume of a cube given its edge length -/
def cube_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

/-- Calculates the surface area of a cube given its edge length -/
def cube_surface_area (edge_length : ℝ) : ℝ := 6 * edge_length ^ 2

/-- Theorem about the total volume and surface area of three cubic boxes -/
theorem three_cubes_volume_and_area (edge1 edge2 edge3 : ℝ) 
  (h1 : edge1 = 3) (h2 : edge2 = 5) (h3 : edge3 = 6) : 
  cube_volume edge1 + cube_volume edge2 + cube_volume edge3 = 368 ∧ 
  cube_surface_area edge1 + cube_surface_area edge2 + cube_surface_area edge3 = 420 := by
  sorry

#check three_cubes_volume_and_area

end NUMINAMATH_CALUDE_three_cubes_volume_and_area_l3587_358762


namespace NUMINAMATH_CALUDE_square_plot_area_l3587_358751

/-- Proves that a square plot with a fence costing Rs. 58 per foot and Rs. 3944 in total has an area of 289 square feet. -/
theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) :
  price_per_foot = 58 →
  total_cost = 3944 →
  ∃ (side_length : ℝ),
    4 * side_length * price_per_foot = total_cost ∧
    side_length^2 = 289 :=
by sorry


end NUMINAMATH_CALUDE_square_plot_area_l3587_358751


namespace NUMINAMATH_CALUDE_tri_divisible_iff_l3587_358730

/-- A polynomial is tri-divisible if 3 divides f(k) for any integer k -/
def TriDivisible (f : Polynomial ℤ) : Prop :=
  ∀ k : ℤ, (3 : ℤ) ∣ (f.eval k)

/-- The necessary and sufficient condition for a polynomial to be tri-divisible -/
theorem tri_divisible_iff (f : Polynomial ℤ) :
  TriDivisible f ↔ ∃ (Q : Polynomial ℤ) (a b c : ℤ),
    f = (X - 1) * (X - 2) * X * Q + 3 * (a * X^2 + b * X + c) :=
sorry

end NUMINAMATH_CALUDE_tri_divisible_iff_l3587_358730


namespace NUMINAMATH_CALUDE_tan_alpha_plus_beta_l3587_358722

theorem tan_alpha_plus_beta (α β : ℝ) 
  (h1 : 3 * Real.tan (α / 2) + Real.tan (α / 2) ^ 2 = 1)
  (h2 : Real.sin β = 3 * Real.sin (2 * α + β)) :
  Real.tan (α + β) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_beta_l3587_358722


namespace NUMINAMATH_CALUDE_quiz_team_payment_l3587_358743

/-- The set of possible values for B in the number 2B5 -/
def possible_B : Set Nat :=
  {b | b ∈ Finset.range 10 ∧ (200 + 10 * b + 5) % 15 = 0}

/-- The theorem stating that the only possible values for B are 2, 5, and 8 -/
theorem quiz_team_payment :
  possible_B = {2, 5, 8} := by sorry

end NUMINAMATH_CALUDE_quiz_team_payment_l3587_358743


namespace NUMINAMATH_CALUDE_product_remainder_mod_seven_l3587_358729

def product_sequence : List ℕ := List.range 10 |>.map (λ i => 3 + 10 * i)

theorem product_remainder_mod_seven :
  (product_sequence.prod) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_seven_l3587_358729


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l3587_358739

theorem polynomial_product_equality (x y z : ℝ) :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2) =
  27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l3587_358739


namespace NUMINAMATH_CALUDE_tournament_has_king_l3587_358717

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Lose

/-- A tournament with m teams -/
structure Tournament (m : ℕ) where
  /-- The result of a match between two teams -/
  result : Fin m → Fin m → MatchResult
  /-- Each pair of teams has competed exactly once -/
  competed_once : ∀ i j : Fin m, i ≠ j → (result i j = MatchResult.Win ∧ result j i = MatchResult.Lose) ∨
                                        (result i j = MatchResult.Lose ∧ result j i = MatchResult.Win)

/-- Definition of a king in the tournament -/
def is_king (t : Tournament m) (x : Fin m) : Prop :=
  ∀ y : Fin m, y ≠ x → 
    (t.result x y = MatchResult.Win) ∨ 
    (∃ z : Fin m, t.result x z = MatchResult.Win ∧ t.result z y = MatchResult.Win)

/-- Theorem: Every tournament has a king -/
theorem tournament_has_king (m : ℕ) (t : Tournament m) : ∃ x : Fin m, is_king t x := by
  sorry

end NUMINAMATH_CALUDE_tournament_has_king_l3587_358717


namespace NUMINAMATH_CALUDE_rectangle_width_l3587_358732

/-- A rectangle with area 50 square meters and perimeter 30 meters has a width of 5 meters. -/
theorem rectangle_width (length width : ℝ) 
  (area_eq : length * width = 50)
  (perimeter_eq : 2 * length + 2 * width = 30) :
  width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3587_358732


namespace NUMINAMATH_CALUDE_room_selection_equivalence_l3587_358718

def total_rooms : ℕ := 6

def select_at_least_two (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ k => if k ≥ 2 then Nat.choose n k else 0)

def sum_of_combinations (n : ℕ) : ℕ :=
  Nat.choose n 3 + 2 * Nat.choose n 4 + Nat.choose n 5 + Nat.choose n 6

def power_minus_seven (n : ℕ) : ℕ :=
  2^n - 7

theorem room_selection_equivalence :
  select_at_least_two total_rooms = sum_of_combinations total_rooms ∧
  select_at_least_two total_rooms = power_minus_seven total_rooms := by
  sorry

end NUMINAMATH_CALUDE_room_selection_equivalence_l3587_358718


namespace NUMINAMATH_CALUDE_levi_initial_score_proof_l3587_358752

/-- Levi's initial score in a basketball game with his brother -/
def levi_initial_score : ℕ := 8

/-- Levi's brother's initial score -/
def brother_initial_score : ℕ := 12

/-- The minimum difference in scores Levi wants to achieve -/
def score_difference : ℕ := 5

/-- Additional scores by Levi's brother -/
def brother_additional_score : ℕ := 3

/-- Additional scores Levi needs to reach his goal -/
def levi_additional_score : ℕ := 12

theorem levi_initial_score_proof :
  levi_initial_score = 8 ∧
  brother_initial_score = 12 ∧
  score_difference = 5 ∧
  brother_additional_score = 3 ∧
  levi_additional_score = 12 ∧
  levi_initial_score + levi_additional_score = 
    brother_initial_score + brother_additional_score + score_difference :=
by sorry

end NUMINAMATH_CALUDE_levi_initial_score_proof_l3587_358752


namespace NUMINAMATH_CALUDE_candle_count_l3587_358744

/-- The number of candles Alex used -/
def used_candles : ℕ := 32

/-- The number of candles Alex has left -/
def leftover_candles : ℕ := 12

/-- The total number of candles Alex had initially -/
def initial_candles : ℕ := used_candles + leftover_candles

theorem candle_count : initial_candles = 44 := by
  sorry

end NUMINAMATH_CALUDE_candle_count_l3587_358744


namespace NUMINAMATH_CALUDE_truck_travel_distance_l3587_358747

/-- Given a truck that travels 300 miles on 10 gallons of diesel,
    prove that it will travel 450 miles on 15 gallons of diesel. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ)
    (h1 : initial_distance = 300)
    (h2 : initial_fuel = 10)
    (h3 : new_fuel = 15) :
    (initial_distance / initial_fuel) * new_fuel = 450 :=
by sorry

end NUMINAMATH_CALUDE_truck_travel_distance_l3587_358747


namespace NUMINAMATH_CALUDE_pushup_difference_l3587_358711

theorem pushup_difference (zachary_pushups : Real) (david_more_than_zachary : Real) (john_less_than_david : Real)
  (h1 : zachary_pushups = 15.5)
  (h2 : david_more_than_zachary = 39.2)
  (h3 : john_less_than_david = 9.3) :
  let david_pushups := zachary_pushups + david_more_than_zachary
  let john_pushups := david_pushups - john_less_than_david
  john_pushups - zachary_pushups = 29.9 := by
sorry

end NUMINAMATH_CALUDE_pushup_difference_l3587_358711


namespace NUMINAMATH_CALUDE_equation_solution_l3587_358772

theorem equation_solution : ∃ x : ℚ, (3/x + (1/x) / (5/x) + 1/(2*x) = 5/4) ∧ (x = 10/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3587_358772


namespace NUMINAMATH_CALUDE_smallest_top_number_l3587_358774

/-- Represents the pyramid structure -/
structure Pyramid :=
  (layer1 : Fin 15 → ℕ)
  (layer2 : Fin 10 → ℕ)
  (layer3 : Fin 6 → ℕ)
  (layer4 : Fin 3 → ℕ)
  (layer5 : ℕ)

/-- The numbering rule for layers 2-5 -/
def validNumbering (p : Pyramid) : Prop :=
  (∀ i : Fin 10, p.layer2 i = p.layer1 (3*i) + p.layer1 (3*i+1) + p.layer1 (3*i+2)) ∧
  (∀ i : Fin 6, p.layer3 i = p.layer2 (3*i) + p.layer2 (3*i+1) + p.layer2 (3*i+2)) ∧
  (∀ i : Fin 3, p.layer4 i = p.layer3 (2*i) + p.layer3 (2*i+1) + p.layer3 (2*i+2)) ∧
  (p.layer5 = p.layer4 0 + p.layer4 1 + p.layer4 2)

/-- The bottom layer contains numbers 1 to 15 -/
def validBottomLayer (p : Pyramid) : Prop :=
  (∀ i : Fin 15, p.layer1 i ∈ Finset.range 16 \ {0}) ∧
  (∀ i j : Fin 15, i ≠ j → p.layer1 i ≠ p.layer1 j)

/-- The theorem stating the smallest possible number for the top block -/
theorem smallest_top_number (p : Pyramid) 
  (h1 : validNumbering p) (h2 : validBottomLayer p) : 
  p.layer5 ≥ 155 :=
sorry

end NUMINAMATH_CALUDE_smallest_top_number_l3587_358774


namespace NUMINAMATH_CALUDE_remaining_integers_l3587_358779

theorem remaining_integers (T : Finset ℕ) : 
  T = Finset.range 100 →
  (Finset.filter (fun n => ¬(n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0)) T).card = 26 :=
by sorry

end NUMINAMATH_CALUDE_remaining_integers_l3587_358779


namespace NUMINAMATH_CALUDE_seating_arrangements_l3587_358707

theorem seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) :
  (n.factorial / (n - k).factorial) = 720 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3587_358707


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l3587_358796

/-- Represents the survey data --/
structure SurveyData where
  male_total : Nat
  male_opposing : Nat
  female_total : Nat
  female_opposing : Nat

/-- Represents different statistical methods --/
inductive StatisticalMethod
  | MeanAndVariance
  | RegressionLine
  | IndependenceTest
  | Probability

/-- Determines the most appropriate method for analyzing the relationship between gender and judgment --/
def most_appropriate_method (data : SurveyData) : StatisticalMethod :=
  StatisticalMethod.IndependenceTest

/-- Theorem stating that the Independence test is the most appropriate method for the given survey data --/
theorem independence_test_most_appropriate (data : SurveyData) :
  most_appropriate_method data = StatisticalMethod.IndependenceTest :=
sorry

end NUMINAMATH_CALUDE_independence_test_most_appropriate_l3587_358796


namespace NUMINAMATH_CALUDE_subtraction_problem_l3587_358737

theorem subtraction_problem (v : Nat) : v < 10 → 400 + 10 * v + 7 - 189 = 268 → v = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3587_358737


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3587_358776

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let A : ℝ × ℝ := (x, 6)
  let B : ℝ × ℝ := reflect_over_y_axis A
  A.1 + A.2 + B.1 + B.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3587_358776


namespace NUMINAMATH_CALUDE_marbles_combination_l3587_358724

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of marbles -/
def total_marbles : ℕ := 9

/-- The number of marbles to choose -/
def marbles_to_choose : ℕ := 4

/-- Theorem stating that choosing 4 marbles from 9 results in 126 ways -/
theorem marbles_combination :
  choose total_marbles marbles_to_choose = 126 := by
  sorry

end NUMINAMATH_CALUDE_marbles_combination_l3587_358724


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l3587_358761

/-- Calculates the number of egg boxes filled per week given the number of hens,
    eggs per hen per day, eggs per box, and days per week. -/
def boxes_per_week (hens : ℕ) (eggs_per_hen_per_day : ℕ) (eggs_per_box : ℕ) (days_per_week : ℕ) : ℕ :=
  (hens * eggs_per_hen_per_day * days_per_week) / eggs_per_box

/-- Proves that given 270 hens, each laying one egg per day, packed in boxes of 6,
    collected 7 days a week, the total number of boxes filled per week is 315. -/
theorem boisjoli_farm_egg_boxes :
  boxes_per_week 270 1 6 7 = 315 := by
  sorry

end NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l3587_358761


namespace NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l3587_358775

theorem angle_with_complement_40percent_of_supplement (x : ℝ) : 
  (90 - x = (2/5) * (180 - x)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_complement_40percent_of_supplement_l3587_358775


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3587_358759

theorem sum_of_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 10 = (7 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3587_358759


namespace NUMINAMATH_CALUDE_binary_101101_conversion_l3587_358756

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, x) => acc + (if x then 2^i else 0)) 0

def decimal_to_base7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_conversion :
  (binary_to_decimal binary_101101 = 45) ∧
  (decimal_to_base7 45 = [6, 3]) := by sorry

end NUMINAMATH_CALUDE_binary_101101_conversion_l3587_358756


namespace NUMINAMATH_CALUDE_xyz_inequality_l3587_358763

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 + 2*x*y*z = 1) : 
  8*x*y*z ≤ 1 ∧ 
  (8*x*y*z = 1 ↔ 
    ((x, y, z) = (1/2, 1/2, 1/2) ∨ 
     (x, y, z) = (-1/2, -1/2, 1/2) ∨ 
     (x, y, z) = (-1/2, 1/2, -1/2) ∨ 
     (x, y, z) = (1/2, -1/2, -1/2))) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3587_358763


namespace NUMINAMATH_CALUDE_rectangle_unique_property_l3587_358725

-- Define the properties
def opposite_sides_equal (shape : Type) : Prop := sorry
def opposite_angles_equal (shape : Type) : Prop := sorry
def diagonals_equal (shape : Type) : Prop := sorry
def opposite_sides_parallel (shape : Type) : Prop := sorry

-- Define rectangles and parallelograms
class Rectangle (shape : Type) : Prop :=
  (opp_sides_eq : opposite_sides_equal shape)
  (opp_angles_eq : opposite_angles_equal shape)
  (diag_eq : diagonals_equal shape)
  (opp_sides_para : opposite_sides_parallel shape)

class Parallelogram (shape : Type) : Prop :=
  (opp_sides_eq : opposite_sides_equal shape)
  (opp_angles_eq : opposite_angles_equal shape)
  (opp_sides_para : opposite_sides_parallel shape)

-- Theorem statement
theorem rectangle_unique_property :
  ∀ (shape : Type),
    Rectangle shape →
    Parallelogram shape →
    (diagonals_equal shape ↔ Rectangle shape) ∧
    (¬(opposite_sides_equal shape ↔ Rectangle shape)) ∧
    (¬(opposite_angles_equal shape ↔ Rectangle shape)) ∧
    (¬(opposite_sides_parallel shape ↔ Rectangle shape)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_unique_property_l3587_358725


namespace NUMINAMATH_CALUDE_max_ab_tangent_circles_l3587_358766

/-- Two externally tangent circles -/
structure TangentCircles where
  a : ℝ
  b : ℝ
  c1 : (x : ℝ) → (y : ℝ) → (x - a)^2 + (y + 2)^2 = 4
  c2 : (x : ℝ) → (y : ℝ) → (x + b)^2 + (y + 2)^2 = 1
  tangent : a + b = 3

/-- The maximum value of ab for externally tangent circles -/
theorem max_ab_tangent_circles (tc : TangentCircles) : 
  ∃ (max : ℝ), max = 9/4 ∧ tc.a * tc.b ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_ab_tangent_circles_l3587_358766


namespace NUMINAMATH_CALUDE_sqrt_a_minus_b_is_natural_l3587_358767

theorem sqrt_a_minus_b_is_natural (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) :
  ∃ k : ℕ, a - b = k^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_minus_b_is_natural_l3587_358767


namespace NUMINAMATH_CALUDE_no_same_line_l3587_358787

/-- Two lines are the same if and only if they have the same slope and y-intercept -/
def same_line (m1 m2 b1 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

/-- The first line equation: ax + 3y + d = 0 -/
def line1 (a d : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + d = 0

/-- The second line equation: 4x - ay + 8 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 4 * x - a * y + 8 = 0

/-- Theorem: There are no real values of a and d such that ax+3y+d=0 and 4x-ay+8=0 represent the same line -/
theorem no_same_line : ¬∃ (a d : ℝ), ∀ (x y : ℝ), line1 a d x y ↔ line2 a x y := by
  sorry

end NUMINAMATH_CALUDE_no_same_line_l3587_358787


namespace NUMINAMATH_CALUDE_length_of_BC_l3587_358773

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  -- A is at the origin
  t.A = (0, 0) ∧
  -- All vertices lie on the parabola
  t.A.2 = parabola t.A.1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1 ∧
  -- BC is parallel to x-axis
  t.B.2 = t.C.2 ∧
  -- Area of the triangle is 128
  abs ((t.B.1 - t.A.1) * (t.C.2 - t.A.2) - (t.C.1 - t.A.1) * (t.B.2 - t.A.2)) / 2 = 128

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : validTriangle t) : 
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_length_of_BC_l3587_358773


namespace NUMINAMATH_CALUDE_cheap_handcuff_time_is_6_l3587_358738

/-- The time it takes to pick the lock on a cheap pair of handcuffs -/
def cheap_handcuff_time : ℝ := 6

/-- The time it takes to pick the lock on an expensive pair of handcuffs -/
def expensive_handcuff_time : ℝ := 8

/-- The number of friends to rescue -/
def num_friends : ℕ := 3

/-- The total time it takes to free all friends -/
def total_rescue_time : ℝ := 42

/-- Theorem stating that the time to pick a cheap handcuff lock is 6 minutes -/
theorem cheap_handcuff_time_is_6 :
  cheap_handcuff_time = 6 ∧
  num_friends * (cheap_handcuff_time + expensive_handcuff_time) = total_rescue_time :=
by sorry

end NUMINAMATH_CALUDE_cheap_handcuff_time_is_6_l3587_358738


namespace NUMINAMATH_CALUDE_angle_value_l3587_358760

theorem angle_value (θ : Real) (h : θ > 0 ∧ θ < 90) : 
  (∃ (x y : Real), x = Real.sin (10 * π / 180) ∧ 
                   y = 1 + Real.sin (80 * π / 180) ∧ 
                   x = Real.sin θ ∧ 
                   y = Real.cos θ) → 
  θ = 85 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_value_l3587_358760


namespace NUMINAMATH_CALUDE_prime_squares_congruence_l3587_358780

theorem prime_squares_congruence (p : ℕ) (hp : Prime p) :
  (∀ a : ℕ, ¬(p ∣ a) → a^2 % p = 1) → p = 2 ∨ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_squares_congruence_l3587_358780


namespace NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l3587_358703

/-- Given that Joe has 50 toy cars initially and wants to have 62 cars in total,
    prove that the number of additional cars he needs is 12. -/
theorem joe_needs_twelve_more_cars (initial_cars : ℕ) (target_cars : ℕ) 
    (h1 : initial_cars = 50) (h2 : target_cars = 62) : 
    target_cars - initial_cars = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l3587_358703


namespace NUMINAMATH_CALUDE_regular_polygon_distance_sum_l3587_358746

theorem regular_polygon_distance_sum (n : ℕ) (h : ℝ) (h_list : List ℝ) :
  n > 2 →
  h > 0 →
  h_list.length = n →
  (∀ x ∈ h_list, x > 0) →
  h_list.sum = n * h :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_distance_sum_l3587_358746


namespace NUMINAMATH_CALUDE_gumballs_eaten_l3587_358742

/-- The number of gumballs in each package -/
def gumballs_per_package : ℝ := 5.0

/-- The number of packages Nathan ate -/
def packages_eaten : ℝ := 20.0

/-- The total number of gumballs Nathan ate -/
def total_gumballs : ℝ := gumballs_per_package * packages_eaten

theorem gumballs_eaten :
  total_gumballs = 100.0 := by sorry

end NUMINAMATH_CALUDE_gumballs_eaten_l3587_358742


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l3587_358757

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_conditions : 
  let N : ℕ := 2329089562800
  ∀ k : ℕ, k ≤ 30 → k ≠ 28 → k ≠ 29 → is_divisible N k ∧ 
  ¬is_divisible N 28 ∧ 
  ¬is_divisible N 29 ∧
  consecutive 28 29 ∧
  28 > 15 ∧ 29 > 15 ∧
  (∀ m : ℕ, m < N → 
    ¬(∀ j : ℕ, j ≤ 30 → j ≠ 28 → j ≠ 29 → is_divisible m j) ∨ 
    is_divisible m 28 ∨ 
    is_divisible m 29 ∨
    ¬(∃ p q : ℕ, p > 15 ∧ q > 15 ∧ consecutive p q ∧ ¬is_divisible m p ∧ ¬is_divisible m q)
  ) :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_conditions_l3587_358757


namespace NUMINAMATH_CALUDE_max_qed_value_l3587_358785

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

theorem max_qed_value 
  (D E L M Q : Digit) 
  (h_distinct : D ≠ E ∧ D ≠ L ∧ D ≠ M ∧ D ≠ Q ∧ 
                E ≠ L ∧ E ≠ M ∧ E ≠ Q ∧ 
                L ≠ M ∧ L ≠ Q ∧ 
                M ≠ Q)
  (h_equation : 91 * E.val + 10 * L.val + 101 * M.val = 100 * Q.val + D.val) :
  (∀ (D' E' Q' : Digit), 
    D' ≠ E' ∧ D' ≠ Q' ∧ E' ≠ Q' → 
    100 * Q'.val + 10 * E'.val + D'.val ≤ 893) :=
sorry

end NUMINAMATH_CALUDE_max_qed_value_l3587_358785


namespace NUMINAMATH_CALUDE_sandy_marks_lost_l3587_358735

theorem sandy_marks_lost (marks_per_correct : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_sums : ℕ) :
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  ∃ (marks_lost_per_incorrect : ℕ), 
    marks_lost_per_incorrect = 2 ∧
    total_marks = correct_sums * marks_per_correct - (total_attempts - correct_sums) * marks_lost_per_incorrect :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_lost_l3587_358735


namespace NUMINAMATH_CALUDE_sector_arc_length_l3587_358789

/-- The length of an arc in a circular sector with given central angle and radius -/
def arc_length (central_angle : Real) (radius : Real) : Real :=
  central_angle * radius

theorem sector_arc_length :
  let central_angle : Real := π / 3
  let radius : Real := 2
  arc_length central_angle radius = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3587_358789


namespace NUMINAMATH_CALUDE_quadratic_sum_l3587_358741

/-- Given a quadratic function f(x) = 10x^2 + 100x + 1000, 
    proves that when written in the form a(x+b)^2 + c, 
    the sum of a, b, and c is 765 -/
theorem quadratic_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
    (∀ x, 10 * x^2 + 100 * x + 1000 = a * (x + b)^2 + c) ∧
    a + b + c = 765 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3587_358741


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l3587_358749

theorem rectangular_plot_breadth (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 2700 →
  width = 30 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l3587_358749


namespace NUMINAMATH_CALUDE_expression_value_l3587_358715

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (z_neq_zero : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3587_358715


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3587_358784

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ 
  (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3587_358784


namespace NUMINAMATH_CALUDE_unique_five_numbers_l3587_358754

def triple_sums (a b c d e : ℝ) : List ℝ :=
  [a + b + c, a + b + d, a + b + e, a + c + d, a + c + e, a + d + e,
   b + c + d, b + c + e, b + d + e, c + d + e]

theorem unique_five_numbers :
  ∃! (a b c d e : ℝ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧
    triple_sums a b c d e = [3, 4, 6, 7, 9, 10, 11, 14, 15, 17] :=
by
  sorry

end NUMINAMATH_CALUDE_unique_five_numbers_l3587_358754


namespace NUMINAMATH_CALUDE_min_ones_in_valid_grid_l3587_358736

/-- A grid of zeros and ones -/
def Grid := Matrix (Fin 11) (Fin 11) Bool

/-- The sum of elements in a 2x2 subgrid is odd -/
def valid_subgrid (g : Grid) (i j : Fin 10) : Prop :=
  (g i j).toNat + (g i (j+1)).toNat + (g (i+1) j).toNat + (g (i+1) (j+1)).toNat % 2 = 1

/-- A grid is valid if all its 2x2 subgrids have odd sum -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j : Fin 10, valid_subgrid g i j

/-- Count the number of ones in a grid -/
def count_ones (g : Grid) : Nat :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (g i j).toNat))

/-- The main theorem: the minimum number of ones in a valid 11x11 grid is 25 -/
theorem min_ones_in_valid_grid :
  ∃ (g : Grid), valid_grid g ∧ count_ones g = 25 ∧
  ∀ (h : Grid), valid_grid h → count_ones h ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_min_ones_in_valid_grid_l3587_358736


namespace NUMINAMATH_CALUDE_sqrt_40_simplification_l3587_358793

theorem sqrt_40_simplification : Real.sqrt 40 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_40_simplification_l3587_358793


namespace NUMINAMATH_CALUDE_function_properties_l3587_358721

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def IsSymmetricAboutPoint (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def IsSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (x + 3/2) + f x = 0)
    (h2 : IsOddFunction (fun x ↦ f (x - 3/4))) :
  (IsPeriodic f 3 ∧ 
   ¬ IsPeriodic f (3/2)) ∧ 
  IsSymmetricAboutPoint f (-3/4) ∧ 
  ¬ IsSymmetricAboutYAxis f :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3587_358721


namespace NUMINAMATH_CALUDE_pyramid_arrangements_10_l3587_358734

/-- The number of distinguishable ways to form a pyramid with n distinct pool balls -/
def pyramid_arrangements (n : ℕ) : ℕ :=
  n.factorial / 9

/-- The theorem stating that the number of distinguishable ways to form a pyramid
    with 10 distinct pool balls is 403,200 -/
theorem pyramid_arrangements_10 :
  pyramid_arrangements 10 = 403200 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_arrangements_10_l3587_358734


namespace NUMINAMATH_CALUDE_hank_carwash_earnings_l3587_358788

/-- Proves that Hank made $100 in the carwash given the donation information -/
theorem hank_carwash_earnings :
  ∀ (carwash_earnings : ℝ),
    -- Conditions
    (carwash_earnings * 0.9 + 80 * 0.75 + 50 = 200) →
    -- Conclusion
    carwash_earnings = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_hank_carwash_earnings_l3587_358788


namespace NUMINAMATH_CALUDE_wax_left_after_detailing_l3587_358794

/-- The amount of wax needed to detail Kellan's car in ounces. -/
def car_wax : ℕ := 3

/-- The amount of wax needed to detail Kellan's SUV in ounces. -/
def suv_wax : ℕ := 4

/-- The amount of wax in the bottle Kellan bought in ounces. -/
def bought_wax : ℕ := 11

/-- The amount of wax Kellan spilled in ounces. -/
def spilled_wax : ℕ := 2

/-- The theorem states that given the above conditions, 
    the amount of wax Kellan has left after waxing his car and SUV is 2 ounces. -/
theorem wax_left_after_detailing : 
  bought_wax - spilled_wax - (car_wax + suv_wax) = 2 := by
  sorry

end NUMINAMATH_CALUDE_wax_left_after_detailing_l3587_358794


namespace NUMINAMATH_CALUDE_two_students_know_same_number_l3587_358765

/-- Represents the number of students a given student knows -/
def StudentsKnown := Fin 81

/-- The set of all students in the course -/
def Students := Fin 81

theorem two_students_know_same_number (f : Students → StudentsKnown) :
  ∃ (i j : Students), i ≠ j ∧ f i = f j :=
sorry

end NUMINAMATH_CALUDE_two_students_know_same_number_l3587_358765


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3587_358770

theorem book_sale_loss_percentage 
  (selling_price_loss : ℝ) 
  (selling_price_gain : ℝ) 
  (gain_percentage : ℝ) :
  selling_price_loss = 800 →
  selling_price_gain = 1100 →
  gain_percentage = 10 →
  (1 - selling_price_loss / (selling_price_gain / (1 + gain_percentage / 100))) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3587_358770


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l3587_358716

-- Define the points
variable (A B C D E F D₁ E₁ F₁ : ℝ × ℝ)

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define points D, E, F on sides of triangle ABC
def point_on_side (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (1 - t) • Q + t • R

-- Define parallel lines
def parallel_lines (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ Q₁ - P₁ = k • (Q₂ - P₂)

-- Define the theorem
theorem triangle_ratio_theorem 
  (h_triangle : triangle_ABC A B C)
  (h_D : point_on_side D B C)
  (h_E : point_on_side E C A)
  (h_F : point_on_side F A B)
  (h_D₁ : parallel_lines A E₁ E F)
  (h_E₁ : parallel_lines B F₁ F D)
  (h_F₁ : parallel_lines C D₁ D E) :
  (B.1 - D.1) / (D.1 - C.1) = (F₁.1 - A.1) / (A.1 - E₁.1) ∧
  (C.1 - E.1) / (E.1 - A.1) = (D₁.1 - B.1) / (B.1 - F₁.1) ∧
  (A.1 - F.1) / (F.1 - B.1) = (E₁.1 - C.1) / (C.1 - D₁.1) :=
sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l3587_358716


namespace NUMINAMATH_CALUDE_xyz_sum_root_l3587_358755

theorem xyz_sum_root (x y z : ℝ) 
  (h1 : y + z = 22 / 2)
  (h2 : z + x = 24 / 2)
  (h3 : x + y = 26 / 2) :
  Real.sqrt (x * y * z * (x + y + z)) = 3 * Real.sqrt 70 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l3587_358755


namespace NUMINAMATH_CALUDE_determinant_implies_cosine_l3587_358712

theorem determinant_implies_cosine (α : Real) : 
  (Real.cos (75 * π / 180) * Real.cos α + Real.sin (75 * π / 180) * Real.sin α = 1/3) →
  (Real.cos ((30 * π / 180) + 2 * α) = 7/9) := by
sorry

end NUMINAMATH_CALUDE_determinant_implies_cosine_l3587_358712


namespace NUMINAMATH_CALUDE_average_sale_is_7000_l3587_358704

/-- Calculates the average sale for six months given the sales of five months and a required sale for the sixth month. -/
def average_sale (sales : List ℕ) (required_sale : ℕ) : ℚ :=
  (sales.sum + required_sale) / 6

/-- Theorem stating that the average sale for the given problem is 7000. -/
theorem average_sale_is_7000 :
  let sales : List ℕ := [4000, 6524, 5689, 7230, 6000]
  let required_sale : ℕ := 12557
  average_sale sales required_sale = 7000 := by
  sorry

#eval average_sale [4000, 6524, 5689, 7230, 6000] 12557

end NUMINAMATH_CALUDE_average_sale_is_7000_l3587_358704


namespace NUMINAMATH_CALUDE_unique_triple_solution_l3587_358719

theorem unique_triple_solution : 
  ∀ (a b p : ℕ+), 
    Nat.Prime p.val → 
    (a.val + b.val : ℕ) ^ p.val = p.val ^ a.val + p.val ^ b.val → 
    a = 1 ∧ b = 1 ∧ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l3587_358719


namespace NUMINAMATH_CALUDE_bobbys_candy_problem_l3587_358714

/-- The problem of Bobby's candy consumption -/
theorem bobbys_candy_problem (initial_candy : ℕ) : 
  initial_candy + 42 = 70 → initial_candy = 28 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_candy_problem_l3587_358714


namespace NUMINAMATH_CALUDE_quadratic_sum_l3587_358769

theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → 
  a + h + k = -23.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3587_358769


namespace NUMINAMATH_CALUDE_problem_solution_l3587_358798

theorem problem_solution : (2210 - 2137)^2 + (2137 - 2028)^2 = 64 * 268.90625 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3587_358798


namespace NUMINAMATH_CALUDE_school_pairing_fraction_l3587_358710

theorem school_pairing_fraction (t s : ℕ) (ht : t > 0) (hs : s > 0) : 
  (t / 4 : ℚ) = (s / 3 : ℚ) → 
  ((t / 4 + s / 3) : ℚ) / ((t + s) : ℚ) = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_school_pairing_fraction_l3587_358710


namespace NUMINAMATH_CALUDE_tangent_line_curve_intersection_l3587_358702

/-- Given a line y = kx + 1 tangent to the curve y = x^3 + ax + b at the point (1, 3), 
    prove that b = -3 -/
theorem tangent_line_curve_intersection (k a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 
    (k * x + 1 = x^3 + a * x + b → x = 1) ∧ 
    (k * 1 + 1 = 1^3 + a * 1 + b) ∧
    (k = 3 * 1^2 + a)) → 
  (∃ b : ℝ, k * 1 + 1 = 1^3 + a * 1 + b ∧ b = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_curve_intersection_l3587_358702


namespace NUMINAMATH_CALUDE_sum_consecutive_odd_integers_minus_20_l3587_358748

/-- Three consecutive odd integers with a specific sum property -/
structure ConsecutiveOddIntegers where
  first : ℤ
  is_odd : Odd first
  sum_first_third : first + (first + 4) = 150

/-- The sum of three consecutive odd integers minus 20 equals 205 -/
theorem sum_consecutive_odd_integers_minus_20 (n : ConsecutiveOddIntegers) :
  n.first + (n.first + 2) + (n.first + 4) - 20 = 205 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_odd_integers_minus_20_l3587_358748


namespace NUMINAMATH_CALUDE_moremom_arrangements_count_l3587_358795

/-- The number of unique arrangements of letters in MOREMOM -/
def moremom_arrangements : ℕ := 420

/-- The total number of letters in MOREMOM -/
def total_letters : ℕ := 7

/-- The number of M's in MOREMOM -/
def m_count : ℕ := 3

/-- The number of O's in MOREMOM -/
def o_count : ℕ := 2

/-- Theorem stating that the number of unique arrangements of letters in MOREMOM is 420 -/
theorem moremom_arrangements_count :
  moremom_arrangements = Nat.factorial total_letters /(Nat.factorial m_count * Nat.factorial o_count) :=
by sorry

end NUMINAMATH_CALUDE_moremom_arrangements_count_l3587_358795


namespace NUMINAMATH_CALUDE_marble_244_is_white_l3587_358764

/-- Represents the color of a marble -/
inductive MarbleColor
  | White
  | Gray
  | Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cyclePosition := n % 12
  if cyclePosition ≤ 4 then MarbleColor.White
  else if cyclePosition ≤ 9 then MarbleColor.Gray
  else MarbleColor.Black

/-- Theorem: The 244th marble in the sequence is white -/
theorem marble_244_is_white : marbleColor 244 = MarbleColor.White := by
  sorry


end NUMINAMATH_CALUDE_marble_244_is_white_l3587_358764


namespace NUMINAMATH_CALUDE_playground_run_distance_l3587_358728

theorem playground_run_distance (length width laps : ℕ) : 
  length = 55 → width = 35 → laps = 2 → 
  2 * (length + width) * laps = 360 := by
sorry

end NUMINAMATH_CALUDE_playground_run_distance_l3587_358728


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3587_358750

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  (∃ x : ℝ, |x - 1| < 2 ∧ x * (x - 3) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3587_358750


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3587_358782

theorem sin_330_degrees : Real.sin (330 * π / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3587_358782


namespace NUMINAMATH_CALUDE_power_function_monotonicity_l3587_358753

/-- A power function is monotonically increasing on (0, +∞) -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x > 0, ∀ y > 0, x < y → (m^2 - m - 1) * x^m < (m^2 - m - 1) * y^m

/-- The condition |m-2| < 1 -/
def condition_q (m : ℝ) : Prop := |m - 2| < 1

theorem power_function_monotonicity (m : ℝ) :
  (is_monotone_increasing m → condition_q m) ∧
  ¬(condition_q m → is_monotone_increasing m) :=
sorry

end NUMINAMATH_CALUDE_power_function_monotonicity_l3587_358753


namespace NUMINAMATH_CALUDE_plane_through_points_l3587_358778

-- Define the points A, B, and C
def A (a : ℝ) : ℝ × ℝ × ℝ := (a, 0, 0)
def B (b : ℝ) : ℝ × ℝ × ℝ := (0, b, 0)
def C (c : ℝ) : ℝ × ℝ × ℝ := (0, 0, c)

-- Define the plane equation
def plane_equation (a b c x y z : ℝ) : Prop :=
  x / a + y / b + z / c = 1

-- Theorem statement
theorem plane_through_points (a b c : ℝ) (h : a * b * c ≠ 0) :
  ∃ (f : ℝ × ℝ × ℝ → Prop),
    (∀ x y z, f (x, y, z) ↔ plane_equation a b c x y z) ∧
    f (A a) ∧ f (B b) ∧ f (C c) :=
sorry

end NUMINAMATH_CALUDE_plane_through_points_l3587_358778


namespace NUMINAMATH_CALUDE_no_consecutive_integers_sum_75_l3587_358720

theorem no_consecutive_integers_sum_75 : 
  ¬∃ (a n : ℕ), n ≥ 2 ∧ (n * (2 * a + n - 1) / 2 = 75) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_sum_75_l3587_358720


namespace NUMINAMATH_CALUDE_increasing_absolute_value_function_l3587_358797

-- Define the function f(x) = |x - a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem increasing_absolute_value_function (a : ℝ) :
  (∀ x y, 1 ≤ x → x < y → f a x < f a y) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_absolute_value_function_l3587_358797


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3587_358727

def set_A : Set ℝ := {x | -1 ≤ 2*x+1 ∧ 2*x+1 ≤ 3}
def set_B : Set ℝ := {x | x ≠ 0 ∧ (x-2)/x ≤ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3587_358727


namespace NUMINAMATH_CALUDE_spending_theorem_l3587_358726

-- Define the fraction of savings spent on the stereo
def stereo_fraction : ℚ := 1/4

-- Define the fraction less spent on the television compared to the stereo
def tv_fraction_less : ℚ := 2/3

-- Calculate the fraction spent on the television
def tv_fraction : ℚ := stereo_fraction - tv_fraction_less * stereo_fraction

-- Define the total fraction spent on both items
def total_fraction : ℚ := stereo_fraction + tv_fraction

-- Theorem statement
theorem spending_theorem : total_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_spending_theorem_l3587_358726


namespace NUMINAMATH_CALUDE_eh_length_l3587_358786

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 7
  (ex - fx)^2 + (ey - fy)^2 = 7^2 ∧
  -- FG = 21
  (fx - gx)^2 + (fy - gy)^2 = 21^2 ∧
  -- GH = 7
  (gx - hx)^2 + (gy - hy)^2 = 7^2 ∧
  -- HE = 13
  (hx - ex)^2 + (hy - ey)^2 = 13^2 ∧
  -- Angle at H is a right angle
  (ex - hx) * (gx - hx) + (ey - hy) * (gy - hy) = 0

-- Theorem statement
theorem eh_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (hx, hy) := q.H
  (ex - hx)^2 + (ey - hy)^2 = 24^2 :=
sorry

end NUMINAMATH_CALUDE_eh_length_l3587_358786


namespace NUMINAMATH_CALUDE_connor_date_cost_l3587_358771

/-- The cost of a movie date for Connor and his date -/
def movie_date_cost (ticket_price : ℚ) (combo_meal_price : ℚ) (candy_price : ℚ) : ℚ :=
  2 * ticket_price + combo_meal_price + 2 * candy_price

/-- Theorem stating the total cost of Connor's movie date -/
theorem connor_date_cost :
  movie_date_cost 10 11 2.5 = 36 :=
by sorry

end NUMINAMATH_CALUDE_connor_date_cost_l3587_358771


namespace NUMINAMATH_CALUDE_max_books_read_l3587_358768

def reading_speed : ℕ := 120
def pages_per_book : ℕ := 360
def reading_time : ℕ := 8

theorem max_books_read : 
  (reading_speed * reading_time) / pages_per_book = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_books_read_l3587_358768


namespace NUMINAMATH_CALUDE_swim_meet_capacity_theorem_l3587_358758

/-- Represents the swimming club's transportation scenario -/
structure SwimMeetTransport where
  num_cars : ℕ
  num_vans : ℕ
  people_per_car : ℕ
  people_per_van : ℕ
  max_car_capacity : ℕ
  max_van_capacity : ℕ

/-- Calculates the number of additional people that could have ridden with the swim team -/
def additional_capacity (t : SwimMeetTransport) : ℕ :=
  (t.num_cars * t.max_car_capacity + t.num_vans * t.max_van_capacity) -
  (t.num_cars * t.people_per_car + t.num_vans * t.people_per_van)

/-- Theorem stating that 17 more people could have ridden with the swim team -/
theorem swim_meet_capacity_theorem (t : SwimMeetTransport)
  (h1 : t.num_cars = 2)
  (h2 : t.num_vans = 3)
  (h3 : t.people_per_car = 5)
  (h4 : t.people_per_van = 3)
  (h5 : t.max_car_capacity = 6)
  (h6 : t.max_van_capacity = 8) :
  additional_capacity t = 17 := by
  sorry

#eval additional_capacity {
  num_cars := 2,
  num_vans := 3,
  people_per_car := 5,
  people_per_van := 3,
  max_car_capacity := 6,
  max_van_capacity := 8
}

end NUMINAMATH_CALUDE_swim_meet_capacity_theorem_l3587_358758
