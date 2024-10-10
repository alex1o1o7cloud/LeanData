import Mathlib

namespace smallest_positive_integer_congruence_l1134_113481

theorem smallest_positive_integer_congruence :
  ∃ y : ℕ+, 
    (∀ z : ℕ+, 58 * z + 14 ≡ 4 [ZMOD 36] → y ≤ z) ∧ 
    (58 * y + 14 ≡ 4 [ZMOD 36]) ∧
    y = 26 := by
  sorry

end smallest_positive_integer_congruence_l1134_113481


namespace arithmetic_mean_of_numbers_l1134_113469

def numbers : List ℝ := [12, 18, 25, 33, 40]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 25.6 := by
  sorry

end arithmetic_mean_of_numbers_l1134_113469


namespace sum_of_max_min_f_l1134_113499

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem sum_of_max_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ (M + m = 2) :=
by sorry

end sum_of_max_min_f_l1134_113499


namespace find_k_value_l1134_113491

theorem find_k_value (x : ℝ) (k : ℝ) : 
  x = 2 → 
  k / (x - 3) - 1 / (3 - x) = 1 → 
  k = -2 := by
sorry

end find_k_value_l1134_113491


namespace function_inequality_l1134_113477

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_inequality (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x < deriv f x) : 
  f 2 > Real.exp 2 * f 0 ∧ f 2017 > Real.exp 2017 * f 0 := by
  sorry

end function_inequality_l1134_113477


namespace complex_multiplication_l1134_113429

theorem complex_multiplication (z : ℂ) : 
  (z.re = 2 ∧ z.im = -1) → z * (2 + I) = 5 := by
  sorry

end complex_multiplication_l1134_113429


namespace seven_thirteenths_repeating_block_length_l1134_113402

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def least_repeating_block_length : ℕ := 6

/-- Theorem stating that the least number of digits in a repeating block of 7/13 is 6 -/
theorem seven_thirteenths_repeating_block_length :
  least_repeating_block_length = 6 := by sorry

end seven_thirteenths_repeating_block_length_l1134_113402


namespace sum_of_terms_l1134_113422

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_terms (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 8 = 10 →
  a 1 + a 3 + a 5 + a 7 + a 9 = 25 :=
by
  sorry

end sum_of_terms_l1134_113422


namespace fraction_ordering_l1134_113414

theorem fraction_ordering : 
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14
  let c := (6 : ℚ) / 17
  let d := b - (1 : ℚ) / 56
  d < c ∧ c < a :=
by sorry

end fraction_ordering_l1134_113414


namespace infinitely_many_common_divisors_l1134_113460

theorem infinitely_many_common_divisors :
  ∀ k : ℕ, ∃ n : ℕ, ∃ d : ℕ, d > 1 ∧ d ∣ (2 * n - 3) ∧ d ∣ (3 * n - 2) :=
by sorry

end infinitely_many_common_divisors_l1134_113460


namespace definite_integral_x_plus_sin_x_l1134_113415

open Real MeasureTheory

theorem definite_integral_x_plus_sin_x : ∫ x in (-1)..1, (x + Real.sin x) = 0 := by
  sorry

end definite_integral_x_plus_sin_x_l1134_113415


namespace carries_mountain_dew_oz_per_can_l1134_113455

/-- Represents the punch recipe and serving information -/
structure PunchRecipe where
  mountain_dew_cans : ℕ
  ice_oz : ℕ
  fruit_juice_oz : ℕ
  total_servings : ℕ
  oz_per_serving : ℕ

/-- Calculates the ounces of Mountain Dew per can -/
def mountain_dew_oz_per_can (recipe : PunchRecipe) : ℕ :=
  let total_oz := recipe.total_servings * recipe.oz_per_serving
  let non_mountain_dew_oz := recipe.ice_oz + recipe.fruit_juice_oz
  let total_mountain_dew_oz := total_oz - non_mountain_dew_oz
  total_mountain_dew_oz / recipe.mountain_dew_cans

/-- Carrie's punch recipe -/
def carries_recipe : PunchRecipe := {
  mountain_dew_cans := 6
  ice_oz := 28
  fruit_juice_oz := 40
  total_servings := 14
  oz_per_serving := 10
}

/-- Theorem stating that each can of Mountain Dew in Carrie's recipe contains 12 oz -/
theorem carries_mountain_dew_oz_per_can :
  mountain_dew_oz_per_can carries_recipe = 12 := by
  sorry

end carries_mountain_dew_oz_per_can_l1134_113455


namespace block_dimension_l1134_113418

/-- The number of positions to place a 2x1x1 block in a layer of 11x10 --/
def positions_in_layer : ℕ := 199

/-- The number of positions to place a 2x1x1 block across two adjacent layers --/
def positions_across_layers : ℕ := 110

/-- The total number of positions to place a 2x1x1 block in a nx11x10 block --/
def total_positions (n : ℕ) : ℕ := n * positions_in_layer + (n - 1) * positions_across_layers

theorem block_dimension (n : ℕ) :
  total_positions n = 2362 → n = 8 := by
  sorry

end block_dimension_l1134_113418


namespace larger_number_proof_l1134_113492

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 9660) :
  max a b = 460 := by
sorry

end larger_number_proof_l1134_113492


namespace score_54_recorded_as_negative_6_l1134_113475

/-- Calculates the recorded score based on the base score and actual score -/
def recordedScore (baseScore actualScore : Int) : Int :=
  actualScore - baseScore

/-- Theorem: A score of 54 points is recorded as -6 points when the base score is 60 -/
theorem score_54_recorded_as_negative_6 :
  recordedScore 60 54 = -6 := by
  sorry

end score_54_recorded_as_negative_6_l1134_113475


namespace interest_rate_is_six_percent_l1134_113426

/-- Calculates the interest rate given the principal, time, and interest amount -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  (interest * 100) / (principal * time)

theorem interest_rate_is_six_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (interest : ℚ) 
  (h1 : principal = 1050)
  (h2 : time = 6)
  (h3 : interest = principal - 672) :
  calculate_interest_rate principal time interest = 6 := by
  sorry

end interest_rate_is_six_percent_l1134_113426


namespace simplify_expression_l1134_113470

theorem simplify_expression (a b m : ℝ) (h1 : a + b = m) (h2 : a * b = -4) :
  (a - 2) * (b - 2) = -2 * m := by
  sorry

end simplify_expression_l1134_113470


namespace sum_of_three_numbers_l1134_113468

theorem sum_of_three_numbers (a b c : ℝ) 
  (eq1 : 2 * a + b = 46)
  (eq2 : b + 2 * c = 53)
  (eq3 : 2 * c + a = 29) :
  a + b + c = 146.5 / 3 := by
  sorry

end sum_of_three_numbers_l1134_113468


namespace range_of_a_l1134_113425

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 ≠ 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Iic (-2) ∪ Set.Icc 1 2 :=
by sorry

end range_of_a_l1134_113425


namespace subtract_negative_one_l1134_113456

theorem subtract_negative_one : 3 - (-1) = 4 := by
  sorry

end subtract_negative_one_l1134_113456


namespace max_value_of_d_l1134_113428

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_prod_eq : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 ∧ 
  ∃ (a' b' c' d' : ℝ), a' + b' + c' + d' = 10 ∧ 
    a' * b' + a' * c' + a' * d' + b' * c' + b' * d' + c' * d' = 20 ∧
    d' = (5 + Real.sqrt 105) / 2 :=
by sorry

end max_value_of_d_l1134_113428


namespace correct_mark_l1134_113408

theorem correct_mark (num_pupils : ℕ) (wrong_mark : ℕ) (correct_mark : ℕ) : 
  num_pupils = 44 →
  wrong_mark = 67 →
  (wrong_mark - correct_mark : ℚ) = num_pupils / 2 →
  correct_mark = 45 := by
sorry

end correct_mark_l1134_113408


namespace domain_transformation_l1134_113480

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_x_plus_1 (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -2 < x ∧ x < -1}

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | -1 < x ∧ x < -1/2}

-- Theorem statement
theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, x ∈ domain_f_x_plus_1 f ↔ f (x + 1) ∈ Set.univ) →
  (∀ x, x ∈ domain_f_2x_plus_1 f ↔ f (2*x + 1) ∈ Set.univ) :=
sorry

end domain_transformation_l1134_113480


namespace ahmed_min_grade_l1134_113400

/-- The number of assignments excluding the final one -/
def num_assignments : ℕ := 9

/-- Ahmed's current average score -/
def ahmed_average : ℕ := 91

/-- Emily's current average score -/
def emily_average : ℕ := 92

/-- Sarah's current average score -/
def sarah_average : ℕ := 94

/-- The minimum passing score -/
def min_score : ℕ := 70

/-- The maximum possible score -/
def max_score : ℕ := 100

/-- Emily's score on the final assignment -/
def emily_final : ℕ := 90

/-- Function to calculate the total score -/
def total_score (average : ℕ) (final : ℕ) : ℕ :=
  average * num_assignments + final

/-- Theorem stating the minimum grade Ahmed needs -/
theorem ahmed_min_grade :
  ∀ x : ℕ, 
    (x ≤ max_score) →
    (total_score ahmed_average x > total_score emily_average emily_final) →
    (total_score ahmed_average x > total_score sarah_average min_score) →
    (∀ y : ℕ, y < x → (total_score ahmed_average y ≤ total_score emily_average emily_final ∨
                       total_score ahmed_average y ≤ total_score sarah_average min_score)) →
    x = 98 := by
  sorry

end ahmed_min_grade_l1134_113400


namespace square_roots_theorem_l1134_113403

theorem square_roots_theorem (a x : ℝ) : 
  x > 0 ∧ (2*a - 1)^2 = x ∧ (-a + 2)^2 = x → x = 9 ∨ x = 1 := by
  sorry

end square_roots_theorem_l1134_113403


namespace a_range_l1134_113489

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ {x | x^2 - 2*x + a > 0} ↔ x^2 - 2*x + a > 0) →
  1 ∉ {x : ℝ | x^2 - 2*x + a > 0} →
  a ≤ 1 := by
sorry

end a_range_l1134_113489


namespace max_value_of_f_l1134_113401

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The maximum value of f(x) is 2 -/
theorem max_value_of_f : ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end max_value_of_f_l1134_113401


namespace daniels_animals_legs_l1134_113440

/-- The number of legs an animal has -/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | _ => 0

/-- The number of animals Daniel has -/
def animal_count (animal : String) : ℕ :=
  match animal with
  | "horse" => 2
  | "dog" => 5
  | "cat" => 7
  | "turtle" => 3
  | "goat" => 1
  | _ => 0

/-- The total number of legs of all animals -/
def total_legs : ℕ :=
  (animal_count "horse" * legs "horse") +
  (animal_count "dog" * legs "dog") +
  (animal_count "cat" * legs "cat") +
  (animal_count "turtle" * legs "turtle") +
  (animal_count "goat" * legs "goat")

theorem daniels_animals_legs : total_legs = 72 := by
  sorry

end daniels_animals_legs_l1134_113440


namespace certain_number_problem_l1134_113463

theorem certain_number_problem (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) 
  (h3 : a * (a - 4) = b * (b - 4)) : a * (a - 4) = -3 := by
  sorry

end certain_number_problem_l1134_113463


namespace right_triangle_with_consecutive_legs_and_hypotenuse_31_l1134_113447

theorem right_triangle_with_consecutive_legs_and_hypotenuse_31 :
  ∃ (a b : ℕ), 
    a + 1 = b ∧ 
    a^2 + b^2 = 31^2 ∧ 
    a + b = 43 := by
  sorry

end right_triangle_with_consecutive_legs_and_hypotenuse_31_l1134_113447


namespace triangle_angle_sum_contradiction_l1134_113485

theorem triangle_angle_sum_contradiction :
  ∀ (left right top : ℝ),
  right = 60 →
  left = 2 * right →
  top = 70 →
  left + right + top ≠ 180 :=
by
  sorry

end triangle_angle_sum_contradiction_l1134_113485


namespace stratified_sampling_medium_stores_stratified_sampling_medium_stores_correct_l1134_113495

theorem stratified_sampling_medium_stores 
  (total_stores medium_stores sample_size : ℕ) 
  (h1 : total_stores > 0) 
  (h2 : medium_stores ≤ total_stores) 
  (h3 : sample_size ≤ total_stores) : 
  ℕ :=
  let medium_stores_to_draw := (medium_stores * sample_size) / total_stores
  medium_stores_to_draw

#check stratified_sampling_medium_stores

theorem stratified_sampling_medium_stores_correct 
  (total_stores medium_stores sample_size : ℕ) 
  (h1 : total_stores > 0) 
  (h2 : medium_stores ≤ total_stores) 
  (h3 : sample_size ≤ total_stores) : 
  stratified_sampling_medium_stores total_stores medium_stores sample_size h1 h2 h3 = 
  (medium_stores * sample_size) / total_stores :=
by
  sorry

end stratified_sampling_medium_stores_stratified_sampling_medium_stores_correct_l1134_113495


namespace determine_coin_weight_in_two_weighings_l1134_113487

-- Define the set of possible coin weights
def coin_weights : Finset ℕ := {7, 8, 9, 10, 11, 12, 13}

-- Define a type for the balance scale comparison result
inductive ComparisonResult
| Equal : ComparisonResult
| LeftHeavier : ComparisonResult
| RightHeavier : ComparisonResult

-- Define a function to simulate a weighing
def weigh (left right : ℕ) : ComparisonResult :=
  if left = right then ComparisonResult.Equal
  else if left > right then ComparisonResult.LeftHeavier
  else ComparisonResult.RightHeavier

-- Define the theorem
theorem determine_coin_weight_in_two_weighings :
  ∀ (x : ℕ), x ∈ coin_weights →
    ∃ (w₁ w₂ : ℕ × ℕ),
      (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.Equal ∨
       (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.LeftHeavier ∧
        weigh (70 * x) (w₂.1 * 70) = ComparisonResult.Equal) ∨
       (weigh (70 * x) (w₁.1 * 70) = ComparisonResult.RightHeavier ∧
        weigh (70 * x) (w₂.2 * 70) = ComparisonResult.Equal)) :=
by sorry

end determine_coin_weight_in_two_weighings_l1134_113487


namespace tan_22_5_decomposition_l1134_113479

theorem tan_22_5_decomposition :
  ∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (a ≥ b) ∧ (b ≥ c) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a - Real.sqrt b - c) ∧
    (a + b + c = 4) := by
  sorry

end tan_22_5_decomposition_l1134_113479


namespace remainder_theorem_l1134_113465

theorem remainder_theorem (n : ℤ) (h : n % 9 = 3) : (5 * n - 12) % 9 = 3 := by
  sorry

end remainder_theorem_l1134_113465


namespace fixed_point_of_exponential_function_l1134_113412

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 2 + a^(x - 1)
  f 1 = 3 :=
sorry

end fixed_point_of_exponential_function_l1134_113412


namespace catherine_wins_l1134_113438

/-- Represents a point on a circle -/
structure CirclePoint where
  -- Define necessary properties for a point on a circle

/-- Represents a triangle formed by three points on the circle -/
structure Triangle where
  vertex1 : CirclePoint
  vertex2 : CirclePoint
  vertex3 : CirclePoint

/-- Represents the state of the game -/
structure GameState where
  chosenTriangles : List Triangle
  currentPlayer : Bool  -- True for Peter, False for Catherine

/-- Checks if a set of triangles has a common interior point -/
def hasCommonInteriorPoint (triangles : List Triangle) : Bool :=
  sorry

/-- Checks if a triangle is valid to be chosen -/
def isValidTriangle (triangle : Triangle) (state : GameState) : Bool :=
  sorry

/-- Represents a move in the game -/
def makeMove (state : GameState) (triangle : Triangle) : Option GameState :=
  sorry

/-- Theorem stating Catherine has a winning strategy -/
theorem catherine_wins (points : List CirclePoint) 
  (h1 : points.length = 100)
  (h2 : points.Nodup) : 
  ∃ (strategy : GameState → Triangle), 
    ∀ (finalState : GameState), 
      (finalState.currentPlayer = false → 
        ∃ (move : Triangle), isValidTriangle move finalState) ∧
      (finalState.currentPlayer = true → 
        ¬∃ (move : Triangle), isValidTriangle move finalState) :=
  sorry

end catherine_wins_l1134_113438


namespace candy_distribution_l1134_113466

/-- The number of children in the circle -/
def num_children : ℕ := 73

/-- The total number of candies distributed -/
def total_candies : ℕ := 2020

/-- The position of the n-th candy distribution -/
def candy_position (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of unique positions reached after distributing all candies -/
def unique_positions : ℕ := 37

theorem candy_distribution :
  num_children - unique_positions = 36 :=
sorry

end candy_distribution_l1134_113466


namespace equation_solution_l1134_113435

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -3/2 ∧ x₂ = 2 ∧
  (∀ x : ℝ, 2*x^2 - 4*x = 6 - 3*x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l1134_113435


namespace problem_1_problem_2_problem_3_l1134_113462

-- Problem 1
theorem problem_1 : (1) - 2 + 3 - 4 + 5 = 3 := by sorry

-- Problem 2
theorem problem_2 : (-4/7) / (8/49) = -7/2 := by sorry

-- Problem 3
theorem problem_3 : (1/2 - 3/5 + 2/3) * (-15) = -17/2 := by sorry

end problem_1_problem_2_problem_3_l1134_113462


namespace walter_chores_l1134_113497

theorem walter_chores (total_days : ℕ) (normal_pay exceptional_pay : ℚ) 
  (total_earnings : ℚ) (min_exceptional_days : ℕ) :
  total_days = 15 →
  normal_pay = 4 →
  exceptional_pay = 6 →
  total_earnings = 70 →
  min_exceptional_days = 5 →
  ∃ (normal_days exceptional_days : ℕ),
    normal_days + exceptional_days = total_days ∧
    normal_days * normal_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days ≥ min_exceptional_days ∧
    exceptional_days = 5 :=
by sorry

end walter_chores_l1134_113497


namespace isosceles_triangle_side_length_l1134_113473

/-- An isosceles triangle with perimeter 53 and base 11 has equal sides of length 21 -/
theorem isosceles_triangle_side_length : 
  ∀ (x : ℝ), 
  x > 0 → -- Ensure positive side length
  x + x + 11 = 53 → -- Perimeter condition
  x = 21 := by
sorry

end isosceles_triangle_side_length_l1134_113473


namespace cube_volume_ratio_l1134_113486

theorem cube_volume_ratio (edge1 edge2 : ℝ) (h : edge2 = 6 * edge1) :
  (edge1^3) / (edge2^3) = 1 / 216 := by
  sorry

end cube_volume_ratio_l1134_113486


namespace line_intersection_canonical_equation_l1134_113471

/-- The canonical equation of the line of intersection of two planes -/
theorem line_intersection_canonical_equation 
  (plane1 : ℝ → ℝ → ℝ → Prop) 
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ x - y + z - 2 = 0)
  (h2 : ∀ x y z, plane2 x y z ↔ x - 2*y - z + 4 = 0) :
  ∃ t : ℝ, ∀ x y z, 
    (plane1 x y z ∧ plane2 x y z) ↔ 
    ((x - 8) / 3 = t ∧ (y - 6) / 2 = t ∧ z / (-1) = t) :=
sorry

end line_intersection_canonical_equation_l1134_113471


namespace pan_division_l1134_113472

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of chocolate cake -/
def pan : Dimensions := ⟨24, 30⟩

/-- Represents a piece of the chocolate cake -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can be divided into exactly 120 pieces -/
theorem pan_division :
  (area pan) / (area piece) = 120 := by sorry

end pan_division_l1134_113472


namespace quadratic_roots_problem_l1134_113459

theorem quadratic_roots_problem (a b : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  x₁^2 + a^2*x₁ + b = 0 ∧
  x₂^2 + a^2*x₂ + b = 0 ∧
  y₁^2 + 5*a*y₁ + 7 = 0 ∧
  y₂^2 + 5*a*y₂ + 7 = 0 ∧
  x₁ - y₁ = 2 ∧
  x₂ - y₂ = 2 →
  a = 4 ∧ b = -29 := by
sorry

end quadratic_roots_problem_l1134_113459


namespace quadratic_minimum_l1134_113404

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x + 3

-- State the theorem
theorem quadratic_minimum :
  ∃ (x_min y_min : ℝ), x_min = -1 ∧ y_min = 1 ∧
  ∀ x, f x ≥ f x_min := by
  sorry

end quadratic_minimum_l1134_113404


namespace sum_of_polynomials_l1134_113450

/-- Given polynomials f, g, and h, prove their sum is equal to the specified polynomial -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => -3*x^2 + x - 4
  let g := fun (x : ℝ) => -5*x^2 + 3*x - 8
  let h := fun (x : ℝ) => 5*x^2 + 5*x + 1
  f x + g x + h x = -3*x^2 + 9*x - 11 := by
  sorry

end sum_of_polynomials_l1134_113450


namespace exchange_rate_l1134_113452

def goose_to_duck : ℕ := 2
def pigeon_to_duck : ℕ := 5

theorem exchange_rate (geese : ℕ) : 
  geese * (goose_to_duck * pigeon_to_duck) = 
  geese * 10 := by sorry

end exchange_rate_l1134_113452


namespace games_per_season_l1134_113457

/-- Given the following conditions:
  - Louie scored 4 goals in the last match
  - Louie scored 40 goals in previous matches
  - Louie's brother scored twice as many goals as Louie in the last match
  - Louie's brother has played for 3 seasons
  - The total number of goals scored by both brothers is 1244
Prove that there are 50 games in each season -/
theorem games_per_season (louie_last_match : ℕ) (louie_previous : ℕ) 
  (brother_multiplier : ℕ) (brother_seasons : ℕ) (total_goals : ℕ) :
  louie_last_match = 4 →
  louie_previous = 40 →
  brother_multiplier = 2 →
  brother_seasons = 3 →
  total_goals = 1244 →
  ∃ (games_per_season : ℕ), 
    louie_last_match + louie_previous + 
    brother_multiplier * louie_last_match * games_per_season * brother_seasons = 
    total_goals ∧ games_per_season = 50 :=
by sorry

end games_per_season_l1134_113457


namespace mountain_height_l1134_113498

/-- The relative height of a mountain given temperature conditions -/
theorem mountain_height (temp_decrease_rate : ℝ) (summit_temp : ℝ) (base_temp : ℝ) :
  temp_decrease_rate = 0.7 →
  summit_temp = 14.1 →
  base_temp = 26 →
  (base_temp - summit_temp) / temp_decrease_rate * 100 = 1700 := by
  sorry

#check mountain_height

end mountain_height_l1134_113498


namespace factorization_proof_l1134_113482

theorem factorization_proof (a : ℝ) : 
  45 * a^2 + 135 * a + 90 * a^3 = 45 * a * (90 * a^2 + a + 3) := by
  sorry

end factorization_proof_l1134_113482


namespace ant_ratio_l1134_113442

theorem ant_ratio (abe beth cece duke : ℕ) : 
  abe = 4 →
  beth = abe + abe / 2 →
  cece = 2 * abe →
  abe + beth + cece + duke = 20 →
  duke = 2 ∧ duke * 2 = abe := by sorry

end ant_ratio_l1134_113442


namespace translation_exists_l1134_113432

-- Define the set of line segments
def LineSegments : Set (Set ℝ) := sorry

-- Define the property that the total length of line segments is less than 1
def TotalLengthLessThanOne (segments : Set (Set ℝ)) : Prop := sorry

-- Define a set of n points on the line
def Points (n : ℕ) : Set ℝ := sorry

-- Define a translation vector
def TranslationVector : ℝ := sorry

-- Define the property that the translation vector length does not exceed n/2
def TranslationLengthValid (v : ℝ) (n : ℕ) : Prop := 
  abs v ≤ n / 2

-- Define the translated points
def TranslatedPoints (points : Set ℝ) (v : ℝ) : Set ℝ := sorry

-- Define the property that no translated point intersects with any line segment
def NoIntersection (translatedPoints : Set ℝ) (segments : Set (Set ℝ)) : Prop := sorry

-- The main theorem
theorem translation_exists (n : ℕ) (segments : Set (Set ℝ)) (points : Set ℝ) 
  (h1 : TotalLengthLessThanOne segments) 
  (h2 : points = Points n) :
  ∃ v : ℝ, TranslationLengthValid v n ∧ 
    NoIntersection (TranslatedPoints points v) segments := by sorry

end translation_exists_l1134_113432


namespace equation_c_is_linear_l1134_113493

/-- Definition of a linear equation with one variable -/
def is_linear_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 3 = 5 -/
def f (x : ℝ) : ℝ := 2 * x - 3

theorem equation_c_is_linear : is_linear_one_var f :=
sorry

end equation_c_is_linear_l1134_113493


namespace expression_value_when_b_is_negative_one_l1134_113467

theorem expression_value_when_b_is_negative_one :
  let b : ℚ := -1
  let expr := (3 * b⁻¹ + (2 * b⁻¹) / 3) / b
  expr = 11 / 3 :=
by sorry

end expression_value_when_b_is_negative_one_l1134_113467


namespace sqrt_sum_diff_complex_expression_system_of_equations_l1134_113448

-- Problem 1
theorem sqrt_sum_diff (a b c : ℝ) (ha : a = 3) (hb : b = 27) (hc : c = 12) :
  Real.sqrt a + Real.sqrt b - Real.sqrt c = 2 * Real.sqrt 3 := by sorry

-- Problem 2
theorem complex_expression (a b c d e : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 20) (hd : d = 15) (he : e = 5) :
  (Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b) - (Real.sqrt c - Real.sqrt d) / Real.sqrt e = Real.sqrt 3 - 1 := by sorry

-- Problem 3
theorem system_of_equations (x y : ℝ) (h1 : 2 * (x + 1) - y = 6) (h2 : x = y - 1) :
  x = 5 ∧ y = 6 := by sorry

end sqrt_sum_diff_complex_expression_system_of_equations_l1134_113448


namespace hundredthOddPositiveInteger_l1134_113437

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- The 100th odd positive integer is 199 -/
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end hundredthOddPositiveInteger_l1134_113437


namespace arithmetic_sequence_difference_l1134_113411

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n and common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, S n = n * (2 * (a 1) + (n - 1) * d) / 2

/-- The theorem to be proved -/
theorem arithmetic_sequence_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a S d →
  (S 2017 / 2017) - (S 17 / 17) = 100 →
  d = 1/10 := by
  sorry

end arithmetic_sequence_difference_l1134_113411


namespace polynomial_factor_l1134_113476

-- Define the polynomial
def P (a b d x : ℝ) : ℝ := a * x^4 + b * x^3 + 27 * x^2 + d * x + 10

-- Define the factor
def F (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2

-- Theorem statement
theorem polynomial_factor (a b d : ℝ) :
  (∃ (e f : ℝ), ∀ x, P a b d x = F x * (e * x^2 + f * x + 5)) →
  a = 2 ∧ b = -13 := by
  sorry

end polynomial_factor_l1134_113476


namespace percentage_decrease_l1134_113436

theorem percentage_decrease (initial : ℝ) (increase : ℝ) (final : ℝ) :
  initial = 1500 →
  increase = 20 →
  final = 1080 →
  ∃ y : ℝ, y = 40 ∧ final = (initial * (1 + increase / 100)) * (1 - y / 100) :=
by sorry

end percentage_decrease_l1134_113436


namespace circumcircle_of_triangle_l1134_113446

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which point C lies
def line_C (x y : ℝ) : Prop := 3 * x - y + 3 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Define the possible equations of the circumcircle
def circumcircle_eq1 (x y : ℝ) : Prop :=
  x^2 + y^2 - (1/2) * x - 5 * y - (3/2) = 0

def circumcircle_eq2 (x y : ℝ) : Prop :=
  x^2 + y^2 - (25/6) * x - (89/9) * y + (347/18) = 0

-- The theorem to be proved
theorem circumcircle_of_triangle :
  ∃ (C : ℝ × ℝ),
    line_C C.1 C.2 ∧
    (∀ (x y : ℝ), circumcircle_eq1 x y ∨ circumcircle_eq2 x y) :=
  sorry

end circumcircle_of_triangle_l1134_113446


namespace circle_trajectory_l1134_113431

/-- Circle 1 with center (a/2, -1) and radius sqrt((a/2)^2 + 2) -/
def circle1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a/2)^2 + (p.2 + 1)^2 = (a/2)^2 + 2}

/-- Circle 2 with center (0, 0) and radius 1 -/
def circle2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line y = x - 1 -/
def symmetryLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- Point C(-a, a) -/
def pointC (a : ℝ) : ℝ × ℝ := (-a, a)

/-- Circle P passing through point C(-a, a) and tangent to y-axis -/
def circleP (a : ℝ) (center : ℝ × ℝ) : Prop :=
  (center.1 + a)^2 + (center.2 - a)^2 = center.1^2

/-- Trajectory of the center of circle P -/
def trajectoryP : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 + 4*p.1 - 4*p.2 + 8 = 0}

theorem circle_trajectory :
  ∃ (a : ℝ), 
    (∀ (p : ℝ × ℝ), p ∈ symmetryLine → (p ∈ circle1 a ↔ p ∈ circle2)) ∧
    (a = 2) ∧
    (∀ (center : ℝ × ℝ), circleP a center → center ∈ trajectoryP) :=
  sorry

end circle_trajectory_l1134_113431


namespace childrens_home_toddlers_l1134_113410

theorem childrens_home_toddlers (total : ℕ) (newborns : ℕ) :
  total = 40 →
  newborns = 4 →
  ∃ (toddlers teenagers : ℕ),
    toddlers + teenagers + newborns = total ∧
    teenagers = 5 * toddlers ∧
    toddlers = 6 :=
by sorry

end childrens_home_toddlers_l1134_113410


namespace quadratic_root_value_l1134_113443

theorem quadratic_root_value (c : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + 20 * x + c = 0 ↔ x = (-20 + Real.sqrt 16) / 8 ∨ x = (-20 - Real.sqrt 16) / 8) 
  → c = 24 := by
  sorry

end quadratic_root_value_l1134_113443


namespace car_owners_without_motorcycles_l1134_113420

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) 
  (car_owners : ℕ) 
  (motorcycle_owners : ℕ) 
  (h1 : total_adults = 400)
  (h2 : car_owners = 350)
  (h3 : motorcycle_owners = 60)
  (h4 : total_adults ≤ car_owners + motorcycle_owners) :
  car_owners - (car_owners + motorcycle_owners - total_adults) = 340 :=
by sorry

end car_owners_without_motorcycles_l1134_113420


namespace smallest_number_proof_l1134_113474

theorem smallest_number_proof (a b c : ℚ) : 
  b = 4 * a →
  c = 2 * b →
  (a + b + c) / 3 = 78 →
  a = 18 := by
sorry

end smallest_number_proof_l1134_113474


namespace distribute_nine_balls_three_boxes_l1134_113458

/-- The number of ways to distribute n identical balls into k distinct boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical balls into k distinct boxes,
    where each box contains at least one ball -/
def distributeAtLeastOne (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n identical balls into k distinct boxes,
    where each box contains at least one ball and the number of balls in each box is different -/
def distributeAtLeastOneDifferent (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 9 identical balls into 3 distinct boxes,
    where each box contains at least one ball and the number of balls in each box is different, is 18 -/
theorem distribute_nine_balls_three_boxes : distributeAtLeastOneDifferent 9 3 = 18 := by
  sorry

end distribute_nine_balls_three_boxes_l1134_113458


namespace adjacent_same_face_exists_l1134_113454

/-- Represents a coin, which can be either heads or tails -/
inductive Coin
| Heads
| Tails

/-- Represents a circular arrangement of 11 coins -/
def CoinArrangement := Fin 11 → Coin

/-- Two positions in the circle are adjacent if they differ by 1 modulo 11 -/
def adjacent (i j : Fin 11) : Prop :=
  (i.val + 1) % 11 = j.val ∨ (j.val + 1) % 11 = i.val

/-- Main theorem: In any arrangement of 11 coins, there exists a pair of adjacent coins showing the same face -/
theorem adjacent_same_face_exists (arrangement : CoinArrangement) :
  ∃ (i j : Fin 11), adjacent i j ∧ arrangement i = arrangement j := by
  sorry

end adjacent_same_face_exists_l1134_113454


namespace total_hotdogs_is_125_l1134_113451

/-- The total number of hotdogs brought by two neighbors, where one neighbor brings 75 hotdogs
    and the other brings 25 fewer hotdogs than the first. -/
def total_hotdogs : ℕ :=
  let first_neighbor := 75
  let second_neighbor := first_neighbor - 25
  first_neighbor + second_neighbor

/-- Theorem stating that the total number of hotdogs brought by the neighbors is 125. -/
theorem total_hotdogs_is_125 : total_hotdogs = 125 := by
  sorry

end total_hotdogs_is_125_l1134_113451


namespace jackson_souvenirs_l1134_113453

/-- Calculates the total number of souvenirs collected by Jackson -/
def total_souvenirs (hermit_crabs : ℕ) (shells_per_crab : ℕ) (starfish_per_shell : ℕ) : ℕ :=
  let spiral_shells := hermit_crabs * shells_per_crab
  let starfish := spiral_shells * starfish_per_shell
  hermit_crabs + spiral_shells + starfish

/-- Proves that Jackson collects 450 souvenirs in total -/
theorem jackson_souvenirs :
  total_souvenirs 45 3 2 = 450 := by
  sorry

#eval total_souvenirs 45 3 2

end jackson_souvenirs_l1134_113453


namespace intersection_M_N_l1134_113445

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end intersection_M_N_l1134_113445


namespace equal_points_iff_odd_participants_l1134_113441

/-- A round-robin chess tournament with no draws -/
structure ChessTournament where
  n : ℕ  -- number of participants
  no_draws : Bool

/-- The total number of games played in a round-robin tournament -/
def total_games (t : ChessTournament) : ℕ := t.n * (t.n - 1) / 2

/-- The total number of points scored in the tournament -/
def total_points (t : ChessTournament) : ℕ := total_games t

/-- Whether all participants have the same number of points -/
def all_equal_points (t : ChessTournament) : Prop :=
  ∃ k : ℕ, k * t.n = total_points t

/-- The main theorem: all participants can have equal points iff the number of participants is odd -/
theorem equal_points_iff_odd_participants (t : ChessTournament) (h : t.no_draws = true) :
  all_equal_points t ↔ Odd t.n :=
sorry

end equal_points_iff_odd_participants_l1134_113441


namespace special_rectangle_perimeter_l1134_113478

/-- A rectangle with the property that increasing both its length and width by 6
    results in an area increase of 114 -/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  area_increase : (length + 6) * (width + 6) - length * width = 114

theorem special_rectangle_perimeter (rect : SpecialRectangle) :
  2 * (rect.length + rect.width) = 26 :=
sorry

end special_rectangle_perimeter_l1134_113478


namespace complex_absolute_value_l1134_113430

theorem complex_absolute_value (ω : ℂ) (h : ω = 5 + 3*I) : 
  Complex.abs (ω^2 + 4*ω + 34) = Real.sqrt 6664 := by
  sorry

end complex_absolute_value_l1134_113430


namespace responses_needed_l1134_113413

/-- Given a 65% response rate and 461.54 questionnaires mailed, prove that 300 responses are needed -/
theorem responses_needed (response_rate : ℝ) (questionnaires_mailed : ℝ) : 
  response_rate = 0.65 → 
  questionnaires_mailed = 461.54 → 
  ⌊response_rate * questionnaires_mailed⌋ = 300 := by
sorry

end responses_needed_l1134_113413


namespace inequality_solution_set_l1134_113409

theorem inequality_solution_set (x : ℝ) : 
  (2 * x - 2) / (x^2 - 5 * x + 6) ≤ 3 ↔ 
  (5/3 < x ∧ x ≤ 2) ∨ (3 ≤ x ∧ x ≤ 4) :=
by sorry

end inequality_solution_set_l1134_113409


namespace largest_three_digit_divisible_by_digits_l1134_113434

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, 800 ≤ n → n < 900 → is_divisible_by_digits n → n ≤ 888 :=
by sorry

end largest_three_digit_divisible_by_digits_l1134_113434


namespace prob_top_joker_modified_deck_l1134_113461

/-- A deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (standard_cards : ℕ)
  (jokers : ℕ)
  (h_total : total_cards = standard_cards + jokers)
  (h_jokers : jokers = 2)

/-- The probability of drawing a joker from the top of a shuffled deck -/
def prob_top_joker (d : Deck) : ℚ :=
  d.jokers / d.total_cards

/-- Theorem stating the probability of drawing a joker from a modified 54-card deck -/
theorem prob_top_joker_modified_deck :
  ∃ (d : Deck), d.total_cards = 54 ∧ d.standard_cards = 52 ∧ prob_top_joker d = 1 / 27 := by
  sorry


end prob_top_joker_modified_deck_l1134_113461


namespace three_digit_number_sum_property_l1134_113416

theorem three_digit_number_sum_property :
  ∃! (N : ℕ), ∃ (a b c : ℕ),
    (100 ≤ N) ∧ (N < 1000) ∧
    (1 ≤ a) ∧ (a ≤ 9) ∧
    (0 ≤ b) ∧ (b ≤ 9) ∧
    (0 ≤ c) ∧ (c ≤ 9) ∧
    (N = 100 * a + 10 * b + c) ∧
    (N = 11 * (a + b + c)) ∧
    (N = 198) := by
  sorry

end three_digit_number_sum_property_l1134_113416


namespace bob_cleaning_time_l1134_113444

/-- 
Given that Alice takes 32 minutes to clean her room and Bob takes 3/4 of Alice's time,
prove that Bob takes 24 minutes to clean his room.
-/
theorem bob_cleaning_time : 
  let alice_time : ℚ := 32
  let bob_fraction : ℚ := 3/4
  let bob_time : ℚ := alice_time * bob_fraction
  bob_time = 24 := by sorry

end bob_cleaning_time_l1134_113444


namespace students_liking_both_mountains_and_sea_l1134_113484

/-- Given a school with the following properties:
  * There are 500 total students
  * 289 students like mountains
  * 337 students like the sea
  * 56 students like neither mountains nor the sea
  Then the number of students who like both mountains and the sea is 182. -/
theorem students_liking_both_mountains_and_sea 
  (total : ℕ) 
  (like_mountains : ℕ) 
  (like_sea : ℕ) 
  (like_neither : ℕ) 
  (h1 : total = 500)
  (h2 : like_mountains = 289)
  (h3 : like_sea = 337)
  (h4 : like_neither = 56) :
  like_mountains + like_sea - (total - like_neither) = 182 := by
  sorry

end students_liking_both_mountains_and_sea_l1134_113484


namespace logarithmic_equation_solution_l1134_113449

theorem logarithmic_equation_solution :
  ∀ x : ℝ, x > 0 → x ≠ 1 →
  (6 - (1 + 4 * 9^(4 - 2 * (Real.log 3 / Real.log (Real.sqrt 3)))) * (Real.log x / Real.log 7) = Real.log 7 / Real.log x) →
  (x = 7 ∨ x = Real.rpow 7 (1/5)) := by
  sorry

end logarithmic_equation_solution_l1134_113449


namespace candy_soda_price_before_increase_l1134_113464

/-- Proves that the total price of a candy box and a soda can before a price increase is 16 pounds, given their initial prices and percentage increases. -/
theorem candy_soda_price_before_increase 
  (candy_price : ℝ) 
  (soda_price : ℝ) 
  (candy_increase : ℝ) 
  (soda_increase : ℝ) 
  (h1 : candy_price = 10) 
  (h2 : soda_price = 6) 
  (h3 : candy_increase = 0.25) 
  (h4 : soda_increase = 0.50) : 
  candy_price + soda_price = 16 := by
  sorry

#check candy_soda_price_before_increase

end candy_soda_price_before_increase_l1134_113464


namespace max_value_a4a8_l1134_113433

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = r * a n

theorem max_value_a4a8 (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_cond : a 2 * a 6 + a 5 * a 11 = 16) : 
    (∀ x, a 4 * a 8 ≤ x → x = 8) :=
  sorry

end max_value_a4a8_l1134_113433


namespace system_solution_l1134_113490

theorem system_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = 1) ∧ (5 * x + 2 * y = 6) ∧ (x = 1) ∧ (y = 1/2) := by
  sorry

end system_solution_l1134_113490


namespace rectangular_plot_length_l1134_113496

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 28 →
  perimeter = 2 * length + 2 * breadth →
  perimeter = 5300 / 26.5 →
  length = 64 := by
sorry

end rectangular_plot_length_l1134_113496


namespace value_of_expression_l1134_113406

theorem value_of_expression (x : ℤ) (h : x = -4) : 5 * x - 2 = -22 := by
  sorry

end value_of_expression_l1134_113406


namespace max_value_of_function_l1134_113417

theorem max_value_of_function (x : ℝ) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ y, y = Real.sin (2 * x) - 2 * (Real.sin x)^2 + 1 → y ≤ M :=
sorry

end max_value_of_function_l1134_113417


namespace mencius_view_contradicts_option_a_l1134_113419

-- Define the philosophical views
def MenciusView := "Human nature is inherently good"
def OptionAView := "Human nature is evil"

-- Define the passage content
def PassageContent := "Discussion on choices between fish, bear's paws, life, and righteousness"

-- Define Mencius's philosophy
def MenciusPhilosophy := "Advocate for inherent goodness of human nature"

-- Theorem to prove
theorem mencius_view_contradicts_option_a :
  (PassageContent = "Discussion on choices between fish, bear's paws, life, and righteousness") →
  (MenciusPhilosophy = "Advocate for inherent goodness of human nature") →
  (MenciusView ≠ OptionAView) :=
by
  sorry


end mencius_view_contradicts_option_a_l1134_113419


namespace exponent_equality_l1134_113424

theorem exponent_equality (y x : ℕ) (h1 : 16^y = 4^x) (h2 : y = 8) : x = 16 := by
  sorry

end exponent_equality_l1134_113424


namespace range_of_f_l1134_113423

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def domain (a : ℝ) : Set ℝ := Set.Icc (a - 1) (2 * a)

theorem range_of_f (a b : ℝ) (h1 : is_even_function (f a b))
  (h2 : ∀ x ∈ domain a, f a b x ∈ Set.Icc 1 (31/27)) :
  Set.range (f a b) = Set.Icc 1 (31/27) := by
  sorry

end range_of_f_l1134_113423


namespace triangle_perimeter_bound_l1134_113494

theorem triangle_perimeter_bound : 
  ∀ (s : ℝ), s > 0 → 7 + s > 19 → 19 + s > 7 → s + 7 + 19 < 53 :=
by sorry

end triangle_perimeter_bound_l1134_113494


namespace petya_sequence_l1134_113407

theorem petya_sequence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (eq1 : (a + b) * (a + c) = a)
  (eq2 : (a + b) * (b + c) = b)
  (eq3 : (a + c) * (b + c) = c) :
  a = 1/4 ∧ b = 1/4 ∧ c = 1/4 := by
sorry

end petya_sequence_l1134_113407


namespace y_derivative_l1134_113421

noncomputable def y (x : ℝ) : ℝ := Real.cos (Real.log 13) - (1 / 44) * (Real.cos (22 * x))^2 / Real.sin (44 * x)

theorem y_derivative (x : ℝ) (h : Real.sin (22 * x) ≠ 0) : 
  deriv y x = 1 / (4 * (Real.sin (22 * x))^2) := by sorry

end y_derivative_l1134_113421


namespace sphere_cube_ratios_l1134_113439

theorem sphere_cube_ratios (R : ℝ) (a : ℝ) (h : a = 2 * R / Real.sqrt 3) :
  let sphere_surface := 4 * Real.pi * R^2
  let cube_surface := 6 * a^2
  let sphere_volume := 4 / 3 * Real.pi * R^3
  let cube_volume := a^3
  (sphere_surface / cube_surface = Real.pi / 2) ∧
  (sphere_volume / cube_volume = Real.pi * Real.sqrt 3 / 2) := by
sorry

end sphere_cube_ratios_l1134_113439


namespace arithmetic_geometric_mean_log_sum_l1134_113483

theorem arithmetic_geometric_mean_log_sum (a b c x y z m : ℝ) 
  (hb : b = (a + c) / 2)
  (hy : y^2 = x * z)
  (hx : x > 0)
  (hy_pos : y > 0)
  (hz : z > 0)
  (hm : m > 0 ∧ m ≠ 1) :
  (b - c) * (Real.log x / Real.log m) + 
  (c - a) * (Real.log y / Real.log m) + 
  (a - b) * (Real.log z / Real.log m) = 0 :=
sorry

end arithmetic_geometric_mean_log_sum_l1134_113483


namespace divisibility_by_396_l1134_113488

def is_divisible_by_396 (n : ℕ) : Prop :=
  n % 396 = 0

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem divisibility_by_396 (n : ℕ) :
  (n ≥ 10000 ∧ n < 100000) →
  (is_divisible_by_396 n ↔ 
    (last_two_digits n % 4 = 0 ∧ 
    (digit_sum n = 18 ∨ digit_sum n = 27))) :=
by sorry

#check divisibility_by_396

end divisibility_by_396_l1134_113488


namespace problem_statement_l1134_113405

-- Define the propositions
def universal_prop := ∀ x : ℝ, 2*x + 5 > 0
def equation_prop := ∀ x : ℝ, x^2 + 5*x = 6

-- Define logical variables
variable (p q : Prop)

-- Theorem statement
theorem problem_statement :
  (universal_prop) ∧
  (¬(equation_prop) ≠ ∃ x : ℝ, x^2 + 5*x ≠ 6) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) ∧
  ((¬(p ∨ q)) → (¬p ∧ ¬q)) :=
sorry

end problem_statement_l1134_113405


namespace journey_time_reduction_l1134_113427

/-- Given a person's journey where increasing speed by 10% reduces time by x minutes, 
    prove the original journey time was 11x minutes. -/
theorem journey_time_reduction (d : ℝ) (s : ℝ) (x : ℝ) 
  (h1 : d > 0) (h2 : s > 0) (h3 : x > 0) 
  (h4 : d / s - d / (1.1 * s) = x) : 
  d / s = 11 * x := by
sorry

end journey_time_reduction_l1134_113427
