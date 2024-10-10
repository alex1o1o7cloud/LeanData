import Mathlib

namespace nested_fraction_evaluation_l2272_227237

theorem nested_fraction_evaluation :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_evaluation_l2272_227237


namespace expression_simplification_l2272_227224

theorem expression_simplification (x y z : ℝ) :
  (x - 3 * (y * z)) - ((x - 3 * y) * z) = -x * z := by
  sorry

end expression_simplification_l2272_227224


namespace multiple_with_binary_digits_l2272_227204

theorem multiple_with_binary_digits (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m ≠ 0 ∧ n ∣ m ∧ (Nat.digits 10 m).length ≤ n ∧ ∀ d ∈ Nat.digits 10 m, d = 0 ∨ d = 1 := by
  sorry

end multiple_with_binary_digits_l2272_227204


namespace apple_eraser_distribution_l2272_227267

/-- Given a total of 84 items consisting of apples and erasers, prove the number of apples each friend receives and the number of erasers the teacher receives. -/
theorem apple_eraser_distribution (a e : ℕ) (h : a + e = 84) :
  ∃ (friend_apples teacher_erasers : ℚ),
    friend_apples = a / 3 ∧
    teacher_erasers = e / 2 := by
  sorry

end apple_eraser_distribution_l2272_227267


namespace perfect_squares_among_expressions_l2272_227259

-- Define the expressions
def A : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^10 * 3^12 * 7^14
def B : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^12 * 3^15 * 7^10
def C : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^9 * 3^18 * 7^15
def D : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^20 * 3^16 * 7^12

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

-- Theorem statement
theorem perfect_squares_among_expressions :
  (is_perfect_square (A 2 3 7)) ∧
  (¬ is_perfect_square (B 2 3 7)) ∧
  (¬ is_perfect_square (C 2 3 7)) ∧
  (is_perfect_square (D 2 3 7)) := by
  sorry

end perfect_squares_among_expressions_l2272_227259


namespace product_of_five_consecutive_integers_divisible_by_ten_l2272_227269

theorem product_of_five_consecutive_integers_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = 10 * k := by
  sorry

end product_of_five_consecutive_integers_divisible_by_ten_l2272_227269


namespace quadratic_function_value_l2272_227254

/-- Given a quadratic function f(x) = -(x+h)^2 with axis of symmetry at x=-3,
    prove that f(0) = -9 -/
theorem quadratic_function_value (h : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = -(x + h)^2) →
  (∀ x < -3, ∀ y > x, f y > f x) →
  (∀ x > -3, ∀ y > x, f y < f x) →
  f 0 = -9 :=
sorry

end quadratic_function_value_l2272_227254


namespace correct_addition_l2272_227264

def original_sum : ℕ := 2002
def correct_sum : ℕ := 2502
def num1 : ℕ := 736
def num2 : ℕ := 941
def num3 : ℕ := 825

def smallest_digit_change (d : ℕ) : Prop :=
  d ≤ 9 ∧ 
  (num1 - d * 100 + num2 + num3 = correct_sum) ∧
  ∀ e, e < d → (num1 - e * 100 + num2 + num3 ≠ correct_sum)

theorem correct_addition :
  smallest_digit_change 5 :=
sorry

end correct_addition_l2272_227264


namespace min_wrapping_paper_dimensions_l2272_227290

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular wrapping paper -/
structure WrappingPaperDimensions where
  width : ℝ
  length : ℝ

/-- Checks if the wrapping paper can cover the box completely -/
def canCoverBox (box : BoxDimensions) (paper : WrappingPaperDimensions) : Prop :=
  paper.width ≥ box.width + 2 * box.height ∧
  paper.length ≥ box.length + 2 * box.height

/-- The main theorem stating the minimum dimensions of wrapping paper required -/
theorem min_wrapping_paper_dimensions (w : ℝ) (hw : w > 0) :
  ∀ paper : WrappingPaperDimensions,
    let box : BoxDimensions := ⟨w, 2*w, w⟩
    canCoverBox box paper →
    paper.width ≥ 3*w ∧ paper.length ≥ 4*w :=
  sorry

end min_wrapping_paper_dimensions_l2272_227290


namespace pizza_sharing_l2272_227298

theorem pizza_sharing (total_slices : ℕ) (difference : ℕ) (y : ℕ) : 
  total_slices = 10 →
  difference = 2 →
  y + (y + difference) = total_slices →
  y = 4 := by
sorry

end pizza_sharing_l2272_227298


namespace distribute_5_3_l2272_227205

/-- The number of ways to distribute n distinct items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags -/
theorem distribute_5_3 : distribute 5 3 = 36 := by sorry

end distribute_5_3_l2272_227205


namespace prob_sum_24_is_prob_four_sixes_l2272_227231

/-- Represents a fair, standard six-sided die -/
def Die : Type := Fin 6

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : Die) : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 24

/-- The probability of rolling four 6s with four fair, standard six-sided dice -/
def prob_four_sixes : ℚ := (1 / 6) ^ 4

theorem prob_sum_24_is_prob_four_sixes : 
  prob_four_sixes = 1 / 1296 :=
sorry

end prob_sum_24_is_prob_four_sixes_l2272_227231


namespace second_distribution_boys_l2272_227297

theorem second_distribution_boys (total_amount : ℕ) (first_boys : ℕ) (difference : ℕ) : 
  total_amount = 5040 →
  first_boys = 14 →
  difference = 80 →
  ∃ (second_boys : ℕ), 
    (total_amount / first_boys = total_amount / second_boys + difference) ∧
    second_boys = 18 :=
by sorry

end second_distribution_boys_l2272_227297


namespace proposition_truth_l2272_227200

-- Define the propositions
def proposition1 (m : ℝ) : Prop := m > 0 ↔ ∃ (x y : ℝ), x^2 + m*y^2 = 1 ∧ ¬(x^2 + y^2 = 1)

def proposition2 (a : ℝ) : Prop := (a = 1 → ∃ (k : ℝ), ∀ (x y : ℝ), a*x + y - 1 = k*(x + a*y - 2)) ∧
                                   ¬(∀ (a : ℝ), a = 1 → ∃ (k : ℝ), ∀ (x y : ℝ), a*x + y - 1 = k*(x + a*y - 2))

def proposition3 (m : ℝ) : Prop := (∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁^3 + m*x₁ < x₂^3 + m*x₂) ↔ m > 0

def proposition4 (p q : Prop) : Prop := ((p ∨ q) → (p ∧ q)) ∧ ((p ∧ q) → (p ∨ q))

-- Theorem stating which propositions are true and which are false
theorem proposition_truth : 
  (∃ (m : ℝ), ¬proposition1 m) ∧ 
  (∀ (a : ℝ), proposition2 a) ∧
  (∃ (m : ℝ), ¬proposition3 m) ∧
  (∀ (p q : Prop), proposition4 p q) := by
  sorry

end proposition_truth_l2272_227200


namespace solution_equation1_solution_equation2_l2272_227243

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x - 2) = 9 / x) ∧ (x ≠ 2) ∧ (x ≠ 0)

def equation2 (x : ℝ) : Prop := (x / (x + 1) = 2 * x / (3 * x + 3) - 1) ∧ (x ≠ -1) ∧ (3 * x + 3 ≠ 0)

-- State the theorems
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 3 := by sorry

theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -3/4 := by sorry

end solution_equation1_solution_equation2_l2272_227243


namespace number_square_relationship_l2272_227282

theorem number_square_relationship (n : ℕ) (h : n = 14) : n + n^2 = 210 := by
  sorry

end number_square_relationship_l2272_227282


namespace book_selection_theorem_l2272_227256

def select_books (total : ℕ) (to_select : ℕ) (must_include : ℕ) : ℕ :=
  Nat.choose (total - must_include) (to_select - must_include)

theorem book_selection_theorem :
  select_books 8 5 1 = 35 :=
by sorry

end book_selection_theorem_l2272_227256


namespace smallest_square_cover_l2272_227272

/-- The side length of the smallest square that can be covered by 3-by-4 rectangles -/
def minSquareSide : ℕ := 12

/-- The area of a 3-by-4 rectangle -/
def rectangleArea : ℕ := 3 * 4

/-- The number of 3-by-4 rectangles required to cover the smallest square -/
def numRectangles : ℕ := 9

theorem smallest_square_cover :
  (minSquareSide * minSquareSide) % rectangleArea = 0 ∧
  numRectangles * rectangleArea = minSquareSide * minSquareSide ∧
  ∀ n : ℕ, n < minSquareSide → (n * n) % rectangleArea ≠ 0 := by
  sorry

#check smallest_square_cover

end smallest_square_cover_l2272_227272


namespace arithmetic_mean_geq_geometric_mean_l2272_227263

theorem arithmetic_mean_geq_geometric_mean {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end arithmetic_mean_geq_geometric_mean_l2272_227263


namespace basketball_weight_l2272_227280

theorem basketball_weight (skateboard_weight : ℝ) (num_skateboards num_basketballs : ℕ) :
  skateboard_weight = 20 →
  num_skateboards = 4 →
  num_basketballs = 5 →
  num_basketballs * (skateboard_weight * num_skateboards / num_basketballs) = num_skateboards * skateboard_weight →
  skateboard_weight * num_skateboards / num_basketballs = 16 :=
by sorry

end basketball_weight_l2272_227280


namespace quadratic_function_properties_l2272_227286

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_properties :
  ∀ (a b : ℝ),
  (∀ x : ℝ, f a b x = f a b (2 - x)) →  -- Symmetry about x=1
  f a b 0 = 0 →                        -- Passes through origin
  (∀ x : ℝ, f a b x = x^2 - 2*x) ∧     -- Explicit expression
  Set.Icc (-1) 3 = {y | ∃ x ∈ Set.Ioo 0 3, f a b x = y} -- Range on (0, 3]
  := by sorry

end quadratic_function_properties_l2272_227286


namespace river_rowing_time_l2272_227234

/-- Conversion factor from yards to meters -/
def yards_to_meters : ℝ := 0.9144

/-- Initial width of the river in yards -/
def initial_width_yards : ℝ := 50

/-- Final width of the river in yards -/
def final_width_yards : ℝ := 80

/-- Rate of river width increase in yards per 10 meters -/
def width_increase_rate : ℝ := 2

/-- Rowing speed in meters per second -/
def rowing_speed : ℝ := 5

/-- Time taken to row from initial width to final width -/
def time_taken : ℝ := 30

theorem river_rowing_time :
  let initial_width_meters := initial_width_yards * yards_to_meters
  let final_width_meters := final_width_yards * yards_to_meters
  let width_difference := final_width_meters - initial_width_meters
  let width_increase_per_10m := width_increase_rate * yards_to_meters
  let distance := (width_difference / width_increase_per_10m) * 10
  distance / rowing_speed = time_taken :=
by sorry

end river_rowing_time_l2272_227234


namespace five_fourths_of_eight_thirds_l2272_227226

theorem five_fourths_of_eight_thirds (x : ℚ) : 
  x = 8 / 3 → (5 / 4) * x = 10 / 3 := by
  sorry

end five_fourths_of_eight_thirds_l2272_227226


namespace subtract_negative_l2272_227209

theorem subtract_negative : -3 - 1 = -4 := by
  sorry

end subtract_negative_l2272_227209


namespace colored_cards_permutations_l2272_227295

/-- The number of distinct permutations of a multiset -/
def multiset_permutations (n : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial n / (frequencies.map Nat.factorial).prod

/-- The problem statement -/
theorem colored_cards_permutations :
  let total_cards : ℕ := 11
  let card_frequencies : List ℕ := [5, 3, 2, 1]
  multiset_permutations total_cards card_frequencies = 27720 := by
  sorry

end colored_cards_permutations_l2272_227295


namespace prank_combinations_l2272_227217

/-- The number of choices for each day of the prank --/
def choices : List Nat := [1, 3, 6, 4, 3]

/-- The total number of combinations --/
def total_combinations : Nat := 216

/-- Theorem stating that the product of the choices equals the total combinations --/
theorem prank_combinations : choices.prod = total_combinations := by
  sorry

end prank_combinations_l2272_227217


namespace range_of_a_when_f_has_four_zeros_l2272_227260

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x^2 else Real.exp (-x) + a * x^2

/-- Theorem stating the range of a when f has four zeros -/
theorem range_of_a_when_f_has_four_zeros :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) →
  a < -Real.exp 2 / 4 ∧ ∀ y : ℝ, y < -Real.exp 2 / 4 → ∃ x : ℝ, f x y = 0 :=
sorry

end range_of_a_when_f_has_four_zeros_l2272_227260


namespace length_AF_is_5_l2272_227236

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the condition for the circle intersecting y-axis at only one point
def circle_intersects_y_axis_once (A : ℝ × ℝ) : Prop :=
  let (x, y) := A
  x - 2*y + 4 = 0

-- Main theorem
theorem length_AF_is_5 (A : ℝ × ℝ) :
  let (x, y) := A
  parabola x y →
  circle_intersects_y_axis_once A →
  Real.sqrt ((x - focus.1)^2 + (y - focus.2)^2) = 5 := by
  sorry

end length_AF_is_5_l2272_227236


namespace smallest_valid_strategy_l2272_227299

/-- Represents a 9x9 game board -/
def GameBoard := Fin 9 → Fin 9 → Bool

/-- Represents an L-shaped tromino -/
structure Tromino :=
  (x : Fin 9) (y : Fin 9) (orientation : Fin 4)

/-- Checks if a tromino covers a given cell -/
def covers (t : Tromino) (x y : Fin 9) : Bool :=
  sorry

/-- Checks if a marking strategy allows unique determination of tromino placement -/
def is_valid_strategy (board : GameBoard) : Prop :=
  ∀ t1 t2 : Tromino, t1 ≠ t2 →
    ∃ x y : Fin 9, board x y ∧ (covers t1 x y ≠ covers t2 x y)

/-- Counts the number of marked cells on the board -/
def count_marked (board : GameBoard) : Nat :=
  sorry

theorem smallest_valid_strategy :
  ∃ (board : GameBoard),
    is_valid_strategy board ∧
    count_marked board = 68 ∧
    ∀ (other_board : GameBoard),
      is_valid_strategy other_board →
      count_marked other_board ≥ 68 :=
sorry

end smallest_valid_strategy_l2272_227299


namespace schedule_theorem_l2272_227216

/-- The number of lessons to be scheduled -/
def total_lessons : ℕ := 6

/-- The number of morning periods -/
def morning_periods : ℕ := 4

/-- The number of afternoon periods -/
def afternoon_periods : ℕ := 2

/-- The number of ways to arrange the schedule -/
def schedule_arrangements : ℕ := 192

theorem schedule_theorem :
  (morning_periods.choose 1) * (afternoon_periods.choose 1) * (total_lessons - 2).factorial = schedule_arrangements :=
sorry

end schedule_theorem_l2272_227216


namespace rainy_days_count_l2272_227202

/-- Proves that the number of rainy days in a week is 2, given the conditions of Mo's drinking habits. -/
theorem rainy_days_count (n : ℤ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 3 * NR = 20 ∧ 
    3 * NR = n * R + 10) → 
  (∃ (R : ℕ), R = 2) :=
by sorry

end rainy_days_count_l2272_227202


namespace range_of_a_l2272_227266

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → a ∈ Set.Iic 1 ∪ Set.Ici 3 := by
  sorry

end range_of_a_l2272_227266


namespace labor_practice_problem_l2272_227201

-- Define the problem parameters
def type_a_capacity : ℕ := 35
def type_b_capacity : ℕ := 30
def type_a_rental : ℕ := 400
def type_b_rental : ℕ := 320
def max_rental : ℕ := 3000

-- Define the theorem
theorem labor_practice_problem :
  ∃ (teachers students : ℕ) (type_a_buses : ℕ),
    -- Conditions from the problem
    students = 30 * teachers + 7 ∧
    31 * teachers = students + 1 ∧
    -- Solution part 1
    teachers = 8 ∧
    students = 247 ∧
    -- Solution part 2
    3 ≤ type_a_buses ∧ type_a_buses ≤ 5 ∧
    type_a_capacity * type_a_buses + type_b_capacity * (teachers - type_a_buses) ≥ teachers + students ∧
    type_a_rental * type_a_buses + type_b_rental * (teachers - type_a_buses) ≤ max_rental ∧
    -- Solution part 3
    (∀ m : ℕ, 3 ≤ m ∧ m ≤ 5 →
      type_a_rental * 3 + type_b_rental * 5 ≤ type_a_rental * m + type_b_rental * (8 - m)) :=
by sorry


end labor_practice_problem_l2272_227201


namespace real_part_of_complex_square_l2272_227233

theorem real_part_of_complex_square : Complex.re ((5 : ℂ) + 2 * Complex.I) ^ 2 = 21 := by
  sorry

end real_part_of_complex_square_l2272_227233


namespace food_waste_scientific_notation_l2272_227211

/-- The amount of food wasted in China annually in kilograms -/
def food_waste : ℕ := 500000000000

/-- Prove that the food waste in China is equivalent to 5 × 10^10 kg -/
theorem food_waste_scientific_notation : food_waste = 5 * (10 ^ 10) := by
  sorry

end food_waste_scientific_notation_l2272_227211


namespace linear_coefficient_is_negative_two_l2272_227271

def polynomial (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem linear_coefficient_is_negative_two :
  ∃ a b c : ℝ, polynomial = λ x => a * x^2 + (-2) * x + c :=
sorry

end linear_coefficient_is_negative_two_l2272_227271


namespace gift_cost_l2272_227208

/-- Proves that the cost of the gift is $250 given the specified conditions --/
theorem gift_cost (erika_savings : ℕ) (cake_cost : ℕ) (leftover : ℕ) :
  erika_savings = 155 →
  cake_cost = 25 →
  leftover = 5 →
  ∃ (gift_cost : ℕ), 
    gift_cost = 250 ∧
    erika_savings + gift_cost / 2 = gift_cost + cake_cost + leftover :=
by
  sorry

end gift_cost_l2272_227208


namespace remaining_requests_after_seven_days_l2272_227268

/-- The number of days -/
def days : ℕ := 7

/-- The number of requests received per day -/
def requests_per_day : ℕ := 8

/-- The number of requests completed per day -/
def completed_per_day : ℕ := 4

/-- The number of remaining requests after a given number of days -/
def remaining_requests (d : ℕ) : ℕ :=
  (requests_per_day - completed_per_day) * d + requests_per_day * d

theorem remaining_requests_after_seven_days :
  remaining_requests days = 84 := by
  sorry

end remaining_requests_after_seven_days_l2272_227268


namespace shirt_discount_problem_l2272_227206

theorem shirt_discount_problem (original_price : ℝ) : 
  (0.75 * (0.75 * original_price) = 19) → 
  original_price = 33.78 := by
  sorry

end shirt_discount_problem_l2272_227206


namespace find_y_l2272_227239

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(2*y) = 4) (h2 : x = 4) : y = 1/2 := by
  sorry

end find_y_l2272_227239


namespace postman_return_speed_l2272_227251

/-- Proves that given a round trip with specified conditions, the return speed is 6 miles/hour -/
theorem postman_return_speed 
  (total_distance : ℝ) 
  (first_half_time : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 4) 
  (h2 : first_half_time = 1) 
  (h3 : average_speed = 3) : 
  (total_distance / 2) / (total_distance / average_speed - first_half_time) = 6 := by
  sorry

end postman_return_speed_l2272_227251


namespace total_seats_calculation_l2272_227218

/-- The number of trains at the station -/
def num_trains : ℕ := 3

/-- The number of cars per train -/
def cars_per_train : ℕ := 12

/-- The number of seats per car -/
def seats_per_car : ℕ := 24

/-- The total number of seats on all trains at the station -/
def total_seats : ℕ := num_trains * cars_per_train * seats_per_car

theorem total_seats_calculation : total_seats = 864 := by
  sorry

end total_seats_calculation_l2272_227218


namespace reciprocal_sum_fractions_l2272_227291

theorem reciprocal_sum_fractions : 
  (1 / (1/4 + 1/6) : ℚ) = 12/5 := by sorry

end reciprocal_sum_fractions_l2272_227291


namespace not_p_and_not_q_true_l2272_227214

theorem not_p_and_not_q_true (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬(p ∨ q)) : 
  (¬p ∧ ¬q) := by
  sorry

end not_p_and_not_q_true_l2272_227214


namespace ratio_problem_l2272_227210

theorem ratio_problem (A B C : ℚ) 
  (sum_eq : A + B + C = 98)
  (ratio_bc : B / C = 5 / 8)
  (b_eq : B = 30) :
  A / B = 2 / 3 := by
sorry

end ratio_problem_l2272_227210


namespace committee_election_count_l2272_227229

def group_size : ℕ := 15
def women_count : ℕ := 5
def committee_size : ℕ := 4
def min_women : ℕ := 2

def elect_committee : ℕ := sorry

theorem committee_election_count : 
  elect_committee = 555 := by sorry

end committee_election_count_l2272_227229


namespace x_plus_y_value_l2272_227278

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 14)
  (h2 : x + |y| - y = 16) : 
  x + y = 26/5 := by
  sorry

end x_plus_y_value_l2272_227278


namespace white_surface_area_fraction_l2272_227258

theorem white_surface_area_fraction (large_cube_edge : ℕ) (small_cube_edge : ℕ) 
  (total_small_cubes : ℕ) (white_small_cubes : ℕ) (black_small_cubes : ℕ) :
  large_cube_edge = 4 →
  small_cube_edge = 1 →
  total_small_cubes = 64 →
  white_small_cubes = 56 →
  black_small_cubes = 8 →
  white_small_cubes + black_small_cubes = total_small_cubes →
  black_small_cubes = large_cube_edge^2 →
  (((6 * large_cube_edge^2) - large_cube_edge^2) : ℚ) / (6 * large_cube_edge^2) = 5/6 := by
  sorry

end white_surface_area_fraction_l2272_227258


namespace profit_margin_calculation_l2272_227289

/-- The originally anticipated profit margin given a 6.4% decrease in purchase price
    and an 8 percentage point increase in profit margin -/
def original_profit_margin : ℝ := 117

/-- The decrease in purchase price as a percentage -/
def price_decrease : ℝ := 6.4

/-- The increase in profit margin in percentage points -/
def margin_increase : ℝ := 8

theorem profit_margin_calculation :
  let new_purchase_price : ℝ := 100 - price_decrease
  let new_profit_margin : ℝ := original_profit_margin + margin_increase
  (100 + original_profit_margin) * 100 = new_purchase_price * (100 + new_profit_margin) := by
  sorry

end profit_margin_calculation_l2272_227289


namespace jackson_monday_earnings_l2272_227279

/-- Represents Jackson's fundraising activities for a week -/
structure FundraisingWeek where
  goal : ℕ
  monday_earnings : ℕ
  tuesday_earnings : ℕ
  houses_per_day : ℕ
  earnings_per_four_houses : ℕ
  working_days : ℕ

/-- Theorem stating that Jackson's Monday earnings were $300 -/
theorem jackson_monday_earnings 
  (week : FundraisingWeek)
  (h1 : week.goal = 1000)
  (h2 : week.tuesday_earnings = 40)
  (h3 : week.houses_per_day = 88)
  (h4 : week.earnings_per_four_houses = 10)
  (h5 : week.working_days = 5) :
  week.monday_earnings = 300 := by
  sorry


end jackson_monday_earnings_l2272_227279


namespace min_value_reciprocal_sum_l2272_227265

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3/4) :
  (4/x + 1/y) ≥ 12 :=
by sorry

end min_value_reciprocal_sum_l2272_227265


namespace calculate_expression_l2272_227296

theorem calculate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (8 * y) * (1 / (4 * y)^3) = 9/4 * y := by
  sorry

end calculate_expression_l2272_227296


namespace gcd_16_12_is_4_l2272_227257

theorem gcd_16_12_is_4 : Nat.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_is_4_l2272_227257


namespace y_share_is_36_l2272_227262

/-- Given a sum divided among x, y, and z, where y gets 45 paisa and z gets 30 paisa for each rupee x gets,
    and the total amount is Rs. 140, prove that y's share is Rs. 36. -/
theorem y_share_is_36 
  (total : ℝ) 
  (x_share : ℝ) 
  (y_share : ℝ) 
  (z_share : ℝ) 
  (h1 : total = 140) 
  (h2 : y_share = 0.45 * x_share) 
  (h3 : z_share = 0.30 * x_share) 
  (h4 : total = x_share + y_share + z_share) : 
  y_share = 36 := by
sorry


end y_share_is_36_l2272_227262


namespace point_distance_product_l2272_227293

theorem point_distance_product : 
  ∀ y₁ y₂ : ℝ,
  (((1 : ℝ) - 5)^2 + (y₁ - 2)^2 = 12^2) →
  (((1 : ℝ) - 5)^2 + (y₂ - 2)^2 = 12^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -28 :=
by
  sorry

end point_distance_product_l2272_227293


namespace sum_of_digits_of_square_not_1991_l2272_227250

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Theorem statement
theorem sum_of_digits_of_square_not_1991 :
  ∀ n : ℕ, sumOfDigits (n^2) ≠ 1991 :=
by
  sorry

end sum_of_digits_of_square_not_1991_l2272_227250


namespace binary_to_quaternary_conversion_l2272_227238

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

/-- The binary number 10101110₂ -/
def binaryNumber : ℕ := 174  -- 10101110₂ in decimal is 174

theorem binary_to_quaternary_conversion :
  base2ToBase4 binaryNumber = 2232 := by sorry

end binary_to_quaternary_conversion_l2272_227238


namespace two_digit_addition_proof_l2272_227281

theorem two_digit_addition_proof (A B C : ℕ) : 
  A ≠ B → B ≠ C → A ≠ C →
  A ≤ 9 → B ≤ 9 → C ≤ 9 →
  A ≠ 0 → C ≠ 0 →
  (10 * A + B) + (10 * C + B) = 100 * C + C * 10 + 6 →
  B = 8 := by
sorry

end two_digit_addition_proof_l2272_227281


namespace polynomial_division_remainder_l2272_227253

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 2*X^2 + 3 : Polynomial ℝ) = (X^2 - 2*X + 4) * q + (4*X - 13) := by
  sorry

end polynomial_division_remainder_l2272_227253


namespace digits_s_200_l2272_227244

/-- s(n) is the number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- count_digits n is the number of digits in the decimal representation of n -/
def count_digits (n : ℕ) : ℕ := sorry

/-- The number of digits in s(200) is 492 -/
theorem digits_s_200 : count_digits (s 200) = 492 := by sorry

end digits_s_200_l2272_227244


namespace correct_reasoning_directions_l2272_227241

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define a function that describes the direction of each reasoning type
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating the correct reasoning directions
theorem correct_reasoning_directions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end correct_reasoning_directions_l2272_227241


namespace min_triangles_17gon_is_six_l2272_227232

/-- The minimum number of triangles needed to divide a 17-gon -/
def min_triangles_17gon : ℕ := 6

/-- A polygon with 17 sides -/
structure Polygon17 :=
  (vertices : Fin 17 → ℝ × ℝ)

/-- A triangulation of a polygon -/
structure Triangulation (P : Polygon17) :=
  (num_triangles : ℕ)
  (is_valid : num_triangles ≥ min_triangles_17gon)

/-- Theorem: The minimum number of triangles to divide a 17-gon is 6 -/
theorem min_triangles_17gon_is_six (P : Polygon17) :
  ∀ (T : Triangulation P), T.num_triangles ≥ min_triangles_17gon :=
sorry

end min_triangles_17gon_is_six_l2272_227232


namespace quadratic_inequality_always_negative_l2272_227230

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -12 * x^2 + 3 * x - 5 < 0 := by
  sorry

end quadratic_inequality_always_negative_l2272_227230


namespace correct_ages_l2272_227294

/-- Teacher Zhang's current age -/
def zhang_age : ℕ := sorry

/-- Wang Bing's current age -/
def wang_age : ℕ := sorry

/-- The relationship between Teacher Zhang's and Wang Bing's ages -/
axiom age_relation : zhang_age = 3 * wang_age + 4

/-- The relationship between their ages 10 years ago and 10 years from now -/
axiom age_time_relation : zhang_age - 10 = wang_age + 10

/-- Theorem stating the correct ages -/
theorem correct_ages : zhang_age = 28 ∧ wang_age = 8 := by sorry

end correct_ages_l2272_227294


namespace probability_theorem_l2272_227245

/-- The probability of opening all safes given the number of keys and safes -/
def probability_open_all_safes (k n : ℕ) : ℚ :=
  if n > k then k / n else 1

/-- The theorem stating the probability of opening all safes -/
theorem probability_theorem (k n : ℕ) (h : n > k) :
  probability_open_all_safes k n = k / n := by
  sorry

#eval probability_open_all_safes 2 94

end probability_theorem_l2272_227245


namespace bread_slice_cost_l2272_227225

/-- Given the conditions of Tim's bread purchase, prove that each slice costs 40 cents. -/
theorem bread_slice_cost :
  let num_loaves : ℕ := 3
  let slices_per_loaf : ℕ := 20
  let payment : ℕ := 2 * 20
  let change : ℕ := 16
  let total_cost : ℕ := payment - change
  let total_slices : ℕ := num_loaves * slices_per_loaf
  let cost_per_slice : ℚ := total_cost / total_slices
  cost_per_slice * 100 = 40 := by
  sorry

end bread_slice_cost_l2272_227225


namespace equation_solution_l2272_227255

theorem equation_solution (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) ↔ y = 1 := by
  sorry

end equation_solution_l2272_227255


namespace monotonic_cubic_range_l2272_227221

/-- Given a function f(x) = -x^3 + ax^2 - x - 1 that is monotonic on ℝ,
    the range of the real number a is [-√3, √3]. -/
theorem monotonic_cubic_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_cubic_range_l2272_227221


namespace other_root_of_quadratic_l2272_227220

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 2 + m = 0) → ((-1)^2 - (-1) + m = 0) := by
  sorry

end other_root_of_quadratic_l2272_227220


namespace abs_sum_min_value_l2272_227213

theorem abs_sum_min_value (x : ℚ) : 
  ∃ (min : ℚ), min = 5 ∧ (∀ y : ℚ, |y - 2| + |y + 3| ≥ min) ∧ (|x - 2| + |x + 3| = min) :=
sorry

end abs_sum_min_value_l2272_227213


namespace shopping_tax_calculation_l2272_227276

theorem shopping_tax_calculation (total : ℝ) (total_positive : 0 < total) : 
  let clothing_percent : ℝ := 0.5
  let food_percent : ℝ := 0.2
  let other_percent : ℝ := 0.3
  let clothing_tax_rate : ℝ := 0.04
  let food_tax_rate : ℝ := 0
  let other_tax_rate : ℝ := 0.08
  let clothing_amount := clothing_percent * total
  let food_amount := food_percent * total
  let other_amount := other_percent * total
  let clothing_tax := clothing_tax_rate * clothing_amount
  let food_tax := food_tax_rate * food_amount
  let other_tax := other_tax_rate * other_amount
  let total_tax := clothing_tax + food_tax + other_tax
  (total_tax / total) = 0.044 := by sorry

end shopping_tax_calculation_l2272_227276


namespace f_range_theorem_l2272_227242

open Real

noncomputable def f (k x : ℝ) : ℝ := (k * x + 4) * log x - x

def has_unique_integer_root (k : ℝ) : Prop :=
  ∃ s t : ℝ, s < 2 ∧ 2 < t ∧ 
    (∀ x, 1 < x → (s < x ∧ x < t ↔ 0 < f k x)) ∧
    (∀ n : ℤ, (s < ↑n ∧ ↑n < t) → n = 2)

theorem f_range_theorem :
  ∀ k : ℝ, has_unique_integer_root k ↔ 
    (1 / log 2 - 2 < k ∧ k ≤ 1 / log 3 - 4 / 3) :=
sorry

end f_range_theorem_l2272_227242


namespace hotel_meal_expenditure_l2272_227247

theorem hotel_meal_expenditure (num_persons : ℕ) (regular_cost : ℕ) (extra_cost : ℕ) (total_cost : ℕ) : 
  num_persons = 9 →
  regular_cost = 12 →
  extra_cost = 8 →
  total_cost = 117 →
  ∃ (x : ℕ), (num_persons - 1) * regular_cost + (x + extra_cost) = total_cost ∧ x = 13 := by
sorry

end hotel_meal_expenditure_l2272_227247


namespace cloth_selling_price_l2272_227215

theorem cloth_selling_price 
  (meters : ℕ) 
  (profit_per_meter : ℕ) 
  (cost_price_per_meter : ℕ) 
  (h1 : meters = 75)
  (h2 : profit_per_meter = 15)
  (h3 : cost_price_per_meter = 51) :
  meters * (cost_price_per_meter + profit_per_meter) = 4950 := by
  sorry

end cloth_selling_price_l2272_227215


namespace balls_picked_proof_l2272_227252

def total_balls : ℕ := 9
def red_balls : ℕ := 3
def blue_balls : ℕ := 2
def green_balls : ℕ := 4

theorem balls_picked_proof (n : ℕ) : 
  total_balls = red_balls + blue_balls + green_balls →
  (red_balls.choose 2 : ℚ) / (total_balls.choose n) = 1 / 12 →
  n = 2 := by
sorry

end balls_picked_proof_l2272_227252


namespace part1_part2_l2272_227275

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Define the solution set for part 1
def solution_set_part1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ -1/2}

-- Define the solution set for part 2
def solution_set_part2 (a : ℝ) : Set ℝ :=
  if a = -1 then {-1}
  else if a < -1 then {x | -1 ≤ x ∧ x ≤ 1/a}
  else {x | 1/a ≤ x ∧ x ≤ -1}

-- Theorem for part 1
theorem part1 :
  ∀ x ∈ solution_set_part1, quadratic_inequality (-2) x ∧
  ∀ a ≠ -2, ∃ x ∈ solution_set_part1, ¬(quadratic_inequality a x) :=
sorry

-- Theorem for part 2
theorem part2 (a : ℝ) (h : a < 0) :
  ∀ x, quadratic_inequality a x ↔ x ∈ solution_set_part2 a :=
sorry

end part1_part2_l2272_227275


namespace functional_equation_solution_l2272_227273

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

/-- The theorem stating the form of functions satisfying the equation -/
theorem functional_equation_solution
    (f : ℝ → ℝ)
    (h_smooth : ContDiff ℝ ⊤ f)
    (h_satisfies : SatisfiesEquation f) :
    ∃ a : ℝ, ∀ x : ℝ, f x = x^2 + a * x := by
  sorry

end functional_equation_solution_l2272_227273


namespace trader_gain_percentage_l2272_227283

/-- Represents the gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gain : ℕ) : ℚ :=
  (num_gain : ℚ) / (num_sold : ℚ) * 100

/-- Theorem stating that selling 90 pens and gaining the cost of 30 pens results in a 33.33% gain -/
theorem trader_gain_percentage : 
  gain_percentage 90 30 = 33.33 := by sorry

end trader_gain_percentage_l2272_227283


namespace pizza_theorem_l2272_227287

def pizza_problem (total_pizzas : ℕ) (first_day_fraction : ℚ) (subsequent_day_fraction : ℚ) (daily_limit_fraction : ℚ) : Prop :=
  ∀ (monday tuesday wednesday thursday friday : ℕ),
    -- Total pizzas condition
    total_pizzas = 1000 →
    -- First day condition
    monday = (total_pizzas : ℚ) * first_day_fraction →
    -- Subsequent days conditions
    tuesday = min ((total_pizzas - monday : ℚ) * subsequent_day_fraction) (monday * daily_limit_fraction) →
    wednesday = min ((total_pizzas - monday - tuesday : ℚ) * subsequent_day_fraction) (tuesday * daily_limit_fraction) →
    thursday = min ((total_pizzas - monday - tuesday - wednesday : ℚ) * subsequent_day_fraction) (wednesday * daily_limit_fraction) →
    friday = min ((total_pizzas - monday - tuesday - wednesday - thursday : ℚ) * subsequent_day_fraction) (thursday * daily_limit_fraction) →
    -- Conclusion
    friday ≤ 2

theorem pizza_theorem : pizza_problem 1000 (7/10) (4/5) (9/10) :=
sorry

end pizza_theorem_l2272_227287


namespace smallest_repunit_divisible_by_97_l2272_227261

theorem smallest_repunit_divisible_by_97 : 
  (∀ k < 96, ∃ r, (10^k - 1) / 9 = 97 * r + 1) ∧ 
  ∃ q, (10^96 - 1) / 9 = 97 * q :=
sorry

end smallest_repunit_divisible_by_97_l2272_227261


namespace color_partition_impossibility_l2272_227227

theorem color_partition_impossibility : ¬ ∃ (A B C : Set ℕ),
  (∀ n : ℕ, n > 1 → (n ∈ A ∨ n ∈ B ∨ n ∈ C)) ∧
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧
  (∀ x y, x ∈ A → y ∈ B → x * y ∈ C) ∧
  (∀ x z, x ∈ A → z ∈ C → x * z ∈ B) ∧
  (∀ y z, y ∈ B → z ∈ C → y * z ∈ A) :=
sorry

end color_partition_impossibility_l2272_227227


namespace expected_winnings_is_one_sixth_l2272_227207

/-- A strange die with 6 sides -/
inductive DieSide
  | one
  | two
  | three
  | four
  | five
  | six

/-- Probability of rolling each side of the die -/
def probability (s : DieSide) : ℚ :=
  match s with
  | DieSide.one => 1/4
  | DieSide.two => 1/4
  | DieSide.three => 1/6
  | DieSide.four => 1/6
  | DieSide.five => 1/6
  | DieSide.six => 1/12

/-- Winnings (or losses) for each outcome -/
def winnings (s : DieSide) : ℤ :=
  match s with
  | DieSide.one => 2
  | DieSide.two => 2
  | DieSide.three => 4
  | DieSide.four => 4
  | DieSide.five => -6
  | DieSide.six => -12

/-- Expected value of winnings -/
def expected_winnings : ℚ :=
  (probability DieSide.one * winnings DieSide.one) +
  (probability DieSide.two * winnings DieSide.two) +
  (probability DieSide.three * winnings DieSide.three) +
  (probability DieSide.four * winnings DieSide.four) +
  (probability DieSide.five * winnings DieSide.five) +
  (probability DieSide.six * winnings DieSide.six)

theorem expected_winnings_is_one_sixth :
  expected_winnings = 1/6 := by sorry

end expected_winnings_is_one_sixth_l2272_227207


namespace right_triangle_hypotenuse_l2272_227212

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) 
  (h3 : c^2 = a^2 + b^2) : c = Real.sqrt 10 := by
  sorry

end right_triangle_hypotenuse_l2272_227212


namespace second_largest_power_of_ten_in_170_factorial_l2272_227246

theorem second_largest_power_of_ten_in_170_factorial : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ n → (170 : ℕ).factorial % (10 ^ k) = 0) ∧ 
  (170 : ℕ).factorial % (10 ^ (n + 1)) ≠ 0 ∧ 
  n = 40 := by
  sorry

end second_largest_power_of_ten_in_170_factorial_l2272_227246


namespace complex_simplification_l2272_227270

theorem complex_simplification :
  let z : ℂ := (2 + Complex.I) / Complex.I
  z = 1 - 2 * Complex.I :=
by sorry

end complex_simplification_l2272_227270


namespace total_stick_length_l2272_227285

/-- The length of Jazel's sticks -/
def stick_length (n : Nat) : ℝ :=
  match n with
  | 1 => 3
  | 2 => 2 * stick_length 1
  | 3 => stick_length 2 - 1
  | _ => 0

/-- The theorem stating the total length of Jazel's sticks -/
theorem total_stick_length :
  stick_length 1 + stick_length 2 + stick_length 3 = 14 := by
  sorry

end total_stick_length_l2272_227285


namespace trevor_remaining_eggs_l2272_227248

def chicken_eggs : List Nat := [4, 3, 2, 2, 5, 1, 3]

def total_eggs : Nat := chicken_eggs.sum

theorem trevor_remaining_eggs :
  total_eggs - 2 - 3 = 15 := by
  sorry

end trevor_remaining_eggs_l2272_227248


namespace bobby_bought_two_packets_l2272_227277

/-- The number of packets of candy Bobby bought -/
def bobby_candy_packets : ℕ :=
  let candies_per_packet : ℕ := 18
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let weeks : ℕ := 3
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let candies_per_week : ℕ := weekdays * candies_per_weekday + weekend_days * candies_per_weekend_day
  let total_candies : ℕ := candies_per_week * weeks
  total_candies / candies_per_packet

theorem bobby_bought_two_packets : bobby_candy_packets = 2 := by
  sorry

end bobby_bought_two_packets_l2272_227277


namespace variance_is_five_ninths_l2272_227228

/-- A random variable with a discrete distribution over {-1, 0, 1} -/
structure DiscreteRV where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_one : a + b + c = 1
  arithmetic_seq : 2 * b = a + c

/-- Expected value of the random variable -/
def expected_value (ξ : DiscreteRV) : ℝ := -1 * ξ.a + 1 * ξ.c

/-- Variance of the random variable -/
def variance (ξ : DiscreteRV) : ℝ :=
  (-1 - expected_value ξ)^2 * ξ.a +
  (0 - expected_value ξ)^2 * ξ.b +
  (1 - expected_value ξ)^2 * ξ.c

theorem variance_is_five_ninths (ξ : DiscreteRV) 
  (h : expected_value ξ = 1/3) : variance ξ = 5/9 := by
  sorry

end variance_is_five_ninths_l2272_227228


namespace bianca_recycling_points_l2272_227203

theorem bianca_recycling_points 
  (points_per_bag : ℕ) 
  (total_bags : ℕ) 
  (bags_not_recycled : ℕ) 
  (h1 : points_per_bag = 5)
  (h2 : total_bags = 17)
  (h3 : bags_not_recycled = 8) :
  (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by sorry

end bianca_recycling_points_l2272_227203


namespace P_in_third_quadrant_l2272_227274

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def is_in_third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given point P -/
def P : Point :=
  { x := -3, y := -4 }

/-- Theorem: Point P is in the third quadrant -/
theorem P_in_third_quadrant : is_in_third_quadrant P := by
  sorry

end P_in_third_quadrant_l2272_227274


namespace percent_equality_l2272_227288

theorem percent_equality : (25 : ℚ) / 100 * 2004 = (50 : ℚ) / 100 * 1002 := by
  sorry

end percent_equality_l2272_227288


namespace pete_ran_least_l2272_227223

-- Define the set of runners
inductive Runner
| Phil
| Tom
| Pete
| Amal
| Sanjay

-- Define a function that maps each runner to their distance run
def distance : Runner → ℝ
| Runner.Phil => 4
| Runner.Tom => 6
| Runner.Pete => 2
| Runner.Amal => 8
| Runner.Sanjay => 7

-- Theorem: Pete ran the least distance
theorem pete_ran_least : ∀ r : Runner, distance Runner.Pete ≤ distance r :=
by sorry

end pete_ran_least_l2272_227223


namespace special_right_triangle_hypotenuse_l2272_227222

/-- A right triangle with specific leg relationship and area -/
structure SpecialRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  leg_relationship : longer_leg = 3 * shorter_leg - 3
  area_condition : (1 / 2) * shorter_leg * longer_leg = 84
  right_angle : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The hypotenuse of the special right triangle is √505 -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) : t.hypotenuse = Real.sqrt 505 := by
  sorry

#check special_right_triangle_hypotenuse

end special_right_triangle_hypotenuse_l2272_227222


namespace division_remainder_sum_l2272_227240

theorem division_remainder_sum (n : ℕ) : 
  (n / 7 = 13 ∧ n % 7 = 1) → ((n + 9) / 8 + (n + 9) % 8 = 17) := by
  sorry

end division_remainder_sum_l2272_227240


namespace new_person_weight_l2272_227219

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 35 →
  ∃ (new_weight : ℝ), new_weight = 55 ∧
    new_weight = replaced_weight + initial_count * weight_increase :=
by sorry

end new_person_weight_l2272_227219


namespace intersection_equality_theorem_l2272_227284

/-- The set A of solutions to x^2 + 2x - 3 = 0 -/
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}

/-- The set B of solutions to x^2 - (k+1)x + k = 0 for a given k -/
def B (k : ℝ) : Set ℝ := {x | x^2 - (k+1)*x + k = 0}

/-- The theorem stating that the set of k values satisfying A ∩ B = B is {1, -3} -/
theorem intersection_equality_theorem :
  {k : ℝ | A ∩ B k = B k} = {1, -3} := by sorry

end intersection_equality_theorem_l2272_227284


namespace brownies_degrees_in_pie_chart_l2272_227235

/-- Calculates the degrees for brownies in a pie chart given the class composition -/
theorem brownies_degrees_in_pie_chart 
  (total_students : ℕ) 
  (cookie_lovers : ℕ) 
  (muffin_lovers : ℕ) 
  (cupcake_lovers : ℕ) 
  (h1 : total_students = 45)
  (h2 : cookie_lovers = 15)
  (h3 : muffin_lovers = 9)
  (h4 : cupcake_lovers = 7)
  (h5 : (total_students - (cookie_lovers + muffin_lovers + cupcake_lovers)) % 2 = 0) :
  (((total_students - (cookie_lovers + muffin_lovers + cupcake_lovers)) / 2) : ℚ) / total_students * 360 = 56 := by
sorry

end brownies_degrees_in_pie_chart_l2272_227235


namespace winning_probability_l2272_227249

theorem winning_probability (total_products winning_products : ℕ) 
  (h1 : total_products = 6)
  (h2 : winning_products = 2) :
  (winning_products : ℚ) / total_products = 1 / 3 := by
  sorry

end winning_probability_l2272_227249


namespace linear_regression_intercept_l2272_227292

theorem linear_regression_intercept 
  (x_mean y_mean b a : ℝ) 
  (h1 : y_mean = b * x_mean + a) 
  (h2 : b = 0.51) 
  (h3 : x_mean = 61.75) 
  (h4 : y_mean = 38.14) : 
  a = 6.65 := by sorry

end linear_regression_intercept_l2272_227292
