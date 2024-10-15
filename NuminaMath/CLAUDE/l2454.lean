import Mathlib

namespace NUMINAMATH_CALUDE_digits_s_200_l2454_245446

/-- s(n) is the number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- count_digits n is the number of digits in the decimal representation of n -/
def count_digits (n : ℕ) : ℕ := sorry

/-- The number of digits in s(200) is 492 -/
theorem digits_s_200 : count_digits (s 200) = 492 := by sorry

end NUMINAMATH_CALUDE_digits_s_200_l2454_245446


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2454_245472

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 2 + m = 0) → ((-1)^2 - (-1) + m = 0) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2454_245472


namespace NUMINAMATH_CALUDE_postman_return_speed_l2454_245438

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

end NUMINAMATH_CALUDE_postman_return_speed_l2454_245438


namespace NUMINAMATH_CALUDE_min_triangles_17gon_is_six_l2454_245421

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

end NUMINAMATH_CALUDE_min_triangles_17gon_is_six_l2454_245421


namespace NUMINAMATH_CALUDE_second_largest_power_of_ten_in_170_factorial_l2454_245444

theorem second_largest_power_of_ten_in_170_factorial : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ n → (170 : ℕ).factorial % (10 ^ k) = 0) ∧ 
  (170 : ℕ).factorial % (10 ^ (n + 1)) ≠ 0 ∧ 
  n = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_power_of_ten_in_170_factorial_l2454_245444


namespace NUMINAMATH_CALUDE_subset_range_of_a_l2454_245489

theorem subset_range_of_a (a : ℝ) : 
  let A := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
  let B := {x : ℝ | a < x ∧ x < a + 1}
  B ⊆ A → 1 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_subset_range_of_a_l2454_245489


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l2454_245406

theorem quadratic_reciprocal_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x * y = 1 ∧ x^2 - 2*(m+2)*x + m^2 - 4 = 0 ∧ y^2 - 2*(m+2)*y + m^2 - 4 = 0) 
  → m = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l2454_245406


namespace NUMINAMATH_CALUDE_winning_probability_l2454_245404

theorem winning_probability (total_products winning_products : ℕ) 
  (h1 : total_products = 6)
  (h2 : winning_products = 2) :
  (winning_products : ℚ) / total_products = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_winning_probability_l2454_245404


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l2454_245475

theorem nested_fraction_evaluation :
  1 / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l2454_245475


namespace NUMINAMATH_CALUDE_color_partition_impossibility_l2454_245447

theorem color_partition_impossibility : ¬ ∃ (A B C : Set ℕ),
  (∀ n : ℕ, n > 1 → (n ∈ A ∨ n ∈ B ∨ n ∈ C)) ∧
  (A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅) ∧
  (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅) ∧
  (∀ x y, x ∈ A → y ∈ B → x * y ∈ C) ∧
  (∀ x z, x ∈ A → z ∈ C → x * z ∈ B) ∧
  (∀ y z, y ∈ B → z ∈ C → y * z ∈ A) :=
sorry

end NUMINAMATH_CALUDE_color_partition_impossibility_l2454_245447


namespace NUMINAMATH_CALUDE_pete_ran_least_l2454_245455

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

end NUMINAMATH_CALUDE_pete_ran_least_l2454_245455


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l2454_245467

/-- The height of a cylinder resulting from a melted sphere, given constant volume --/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3)
  (h_cylinder : r_cylinder = 10) :
  (4 / 3 * π * r_sphere ^ 3) / (π * r_cylinder ^ 2) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_melted_ice_cream_height_l2454_245467


namespace NUMINAMATH_CALUDE_length_AF_is_5_l2454_245454

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

end NUMINAMATH_CALUDE_length_AF_is_5_l2454_245454


namespace NUMINAMATH_CALUDE_linear_regression_intercept_l2454_245407

theorem linear_regression_intercept 
  (x_mean y_mean b a : ℝ) 
  (h1 : y_mean = b * x_mean + a) 
  (h2 : b = 0.51) 
  (h3 : x_mean = 61.75) 
  (h4 : y_mean = 38.14) : 
  a = 6.65 := by sorry

end NUMINAMATH_CALUDE_linear_regression_intercept_l2454_245407


namespace NUMINAMATH_CALUDE_prob_sum_24_is_prob_four_sixes_l2454_245442

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

end NUMINAMATH_CALUDE_prob_sum_24_is_prob_four_sixes_l2454_245442


namespace NUMINAMATH_CALUDE_scientific_notation_308000000_l2454_245405

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_308000000 :
  to_scientific_notation 308000000 = ScientificNotation.mk 3.08 8 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_308000000_l2454_245405


namespace NUMINAMATH_CALUDE_total_stick_length_l2454_245414

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

end NUMINAMATH_CALUDE_total_stick_length_l2454_245414


namespace NUMINAMATH_CALUDE_colored_cards_permutations_l2454_245465

/-- The number of distinct permutations of a multiset -/
def multiset_permutations (n : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial n / (frequencies.map Nat.factorial).prod

/-- The problem statement -/
theorem colored_cards_permutations :
  let total_cards : ℕ := 11
  let card_frequencies : List ℕ := [5, 3, 2, 1]
  multiset_permutations total_cards card_frequencies = 27720 := by
  sorry

end NUMINAMATH_CALUDE_colored_cards_permutations_l2454_245465


namespace NUMINAMATH_CALUDE_square_root_of_square_l2454_245492

theorem square_root_of_square (x : ℝ) : {y : ℝ | y^2 = x^2} = {x, -x} := by sorry

end NUMINAMATH_CALUDE_square_root_of_square_l2454_245492


namespace NUMINAMATH_CALUDE_davids_math_marks_l2454_245497

/-- Represents the marks obtained by David in various subjects -/
structure Marks where
  english : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ

/-- Calculates the average marks given the total marks and number of subjects -/
def average (total : ℕ) (subjects : ℕ) : ℚ :=
  (total : ℚ) / (subjects : ℚ)

/-- Theorem stating that given David's marks in other subjects and his average,
    his Mathematics marks must be 60 -/
theorem davids_math_marks (m : Marks) (h1 : m.english = 72) (h2 : m.physics = 35)
    (h3 : m.chemistry = 62) (h4 : m.biology = 84)
    (h5 : average (m.english + m.physics + m.chemistry + m.biology + m.mathematics) 5 = 62.6) :
    m.mathematics = 60 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l2454_245497


namespace NUMINAMATH_CALUDE_variance_is_five_ninths_l2454_245448

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

end NUMINAMATH_CALUDE_variance_is_five_ninths_l2454_245448


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l2454_245451

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

end NUMINAMATH_CALUDE_shopping_tax_calculation_l2454_245451


namespace NUMINAMATH_CALUDE_min_wrapping_paper_dimensions_l2454_245403

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

end NUMINAMATH_CALUDE_min_wrapping_paper_dimensions_l2454_245403


namespace NUMINAMATH_CALUDE_balls_picked_proof_l2454_245459

def total_balls : ℕ := 9
def red_balls : ℕ := 3
def blue_balls : ℕ := 2
def green_balls : ℕ := 4

theorem balls_picked_proof (n : ℕ) : 
  total_balls = red_balls + blue_balls + green_balls →
  (red_balls.choose 2 : ℚ) / (total_balls.choose n) = 1 / 12 →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_balls_picked_proof_l2454_245459


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2454_245415

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 14)
  (h2 : x + |y| - y = 16) : 
  x + y = 26/5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2454_245415


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2454_245469

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

end NUMINAMATH_CALUDE_functional_equation_solution_l2454_245469


namespace NUMINAMATH_CALUDE_sin_theta_value_l2454_245496

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 29) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l2454_245496


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2454_245425

def select_books (total : ℕ) (to_select : ℕ) (must_include : ℕ) : ℕ :=
  Nat.choose (total - must_include) (to_select - must_include)

theorem book_selection_theorem :
  select_books 8 5 1 = 35 :=
by sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2454_245425


namespace NUMINAMATH_CALUDE_bobby_bought_two_packets_l2454_245452

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

end NUMINAMATH_CALUDE_bobby_bought_two_packets_l2454_245452


namespace NUMINAMATH_CALUDE_apple_eraser_distribution_l2454_245410

/-- Given a total of 84 items consisting of apples and erasers, prove the number of apples each friend receives and the number of erasers the teacher receives. -/
theorem apple_eraser_distribution (a e : ℕ) (h : a + e = 84) :
  ∃ (friend_apples teacher_erasers : ℚ),
    friend_apples = a / 3 ∧
    teacher_erasers = e / 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_eraser_distribution_l2454_245410


namespace NUMINAMATH_CALUDE_intersection_equality_theorem_l2454_245482

/-- The set A of solutions to x^2 + 2x - 3 = 0 -/
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}

/-- The set B of solutions to x^2 - (k+1)x + k = 0 for a given k -/
def B (k : ℝ) : Set ℝ := {x | x^2 - (k+1)*x + k = 0}

/-- The theorem stating that the set of k values satisfying A ∩ B = B is {1, -3} -/
theorem intersection_equality_theorem :
  {k : ℝ | A ∩ B k = B k} = {1, -3} := by sorry

end NUMINAMATH_CALUDE_intersection_equality_theorem_l2454_245482


namespace NUMINAMATH_CALUDE_correct_addition_l2454_245486

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

end NUMINAMATH_CALUDE_correct_addition_l2454_245486


namespace NUMINAMATH_CALUDE_smallest_repunit_divisible_by_97_l2454_245462

theorem smallest_repunit_divisible_by_97 : 
  (∀ k < 96, ∃ r, (10^k - 1) / 9 = 97 * r + 1) ∧ 
  ∃ q, (10^96 - 1) / 9 = 97 * q :=
sorry

end NUMINAMATH_CALUDE_smallest_repunit_divisible_by_97_l2454_245462


namespace NUMINAMATH_CALUDE_f_range_theorem_l2454_245483

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

end NUMINAMATH_CALUDE_f_range_theorem_l2454_245483


namespace NUMINAMATH_CALUDE_linear_coefficient_is_negative_two_l2454_245408

def polynomial (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem linear_coefficient_is_negative_two :
  ∃ a b c : ℝ, polynomial = λ x => a * x^2 + (-2) * x + c :=
sorry

end NUMINAMATH_CALUDE_linear_coefficient_is_negative_two_l2454_245408


namespace NUMINAMATH_CALUDE_new_person_weight_l2454_245471

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 35 →
  ∃ (new_weight : ℝ), new_weight = 55 ∧
    new_weight = replaced_weight + initial_count * weight_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2454_245471


namespace NUMINAMATH_CALUDE_smallest_square_cover_l2454_245409

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

end NUMINAMATH_CALUDE_smallest_square_cover_l2454_245409


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l2454_245485

theorem arithmetic_mean_geq_geometric_mean {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l2454_245485


namespace NUMINAMATH_CALUDE_expression_simplification_l2454_245456

theorem expression_simplification (x y z : ℝ) :
  (x - 3 * (y * z)) - ((x - 3 * y) * z) = -x * z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2454_245456


namespace NUMINAMATH_CALUDE_jackson_monday_earnings_l2454_245416

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


end NUMINAMATH_CALUDE_jackson_monday_earnings_l2454_245416


namespace NUMINAMATH_CALUDE_range_of_a_l2454_245401

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → a ∈ Set.Iic 1 ∪ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2454_245401


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_not_1991_l2454_245439

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

-- Theorem statement
theorem sum_of_digits_of_square_not_1991 :
  ∀ n : ℕ, sumOfDigits (n^2) ≠ 1991 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_not_1991_l2454_245439


namespace NUMINAMATH_CALUDE_schedule_theorem_l2454_245473

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

end NUMINAMATH_CALUDE_schedule_theorem_l2454_245473


namespace NUMINAMATH_CALUDE_perfect_squares_among_expressions_l2454_245460

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

end NUMINAMATH_CALUDE_perfect_squares_among_expressions_l2454_245460


namespace NUMINAMATH_CALUDE_pizza_sharing_l2454_245436

theorem pizza_sharing (total_slices : ℕ) (difference : ℕ) (y : ℕ) : 
  total_slices = 10 →
  difference = 2 →
  y + (y + difference) = total_slices →
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_pizza_sharing_l2454_245436


namespace NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l2454_245488

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

end NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l2454_245488


namespace NUMINAMATH_CALUDE_can_cut_one_more_square_l2454_245498

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square region in the grid -/
structure Square :=
  (size : ℕ)

/-- The number of 2x2 squares that can fit in a grid -/
def count_2x2_squares (g : Grid) : ℕ :=
  ((g.rows - 1) / 2) * ((g.cols - 1) / 2)

theorem can_cut_one_more_square (g : Grid) (s : Square) (n : ℕ) :
  g.rows = 29 →
  g.cols = 29 →
  s.size = 2 →
  n = 99 →
  n < count_2x2_squares g →
  ∃ (remaining : ℕ), remaining > 0 ∧ remaining = count_2x2_squares g - n :=
by sorry

end NUMINAMATH_CALUDE_can_cut_one_more_square_l2454_245498


namespace NUMINAMATH_CALUDE_cubic_decreasing_iff_l2454_245490

theorem cubic_decreasing_iff (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x) ((3 * a * x^2) - 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x) > (a * y^3 - y)) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_iff_l2454_245490


namespace NUMINAMATH_CALUDE_equation_solution_l2454_245419

theorem equation_solution (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) ↔ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2454_245419


namespace NUMINAMATH_CALUDE_brownies_degrees_in_pie_chart_l2454_245453

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

end NUMINAMATH_CALUDE_brownies_degrees_in_pie_chart_l2454_245453


namespace NUMINAMATH_CALUDE_calculate_expression_l2454_245434

theorem calculate_expression (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (8 * y) * (1 / (4 * y)^3) = 9/4 * y := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2454_245434


namespace NUMINAMATH_CALUDE_bread_slice_cost_l2454_245412

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

end NUMINAMATH_CALUDE_bread_slice_cost_l2454_245412


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l2454_245491

/-- Calculates the area of carpet required for a room and corridor -/
theorem carpet_area_calculation (main_length main_width corridor_length corridor_width : ℝ) 
  (h_main_length : main_length = 15)
  (h_main_width : main_width = 12)
  (h_corridor_length : corridor_length = 10)
  (h_corridor_width : corridor_width = 3)
  (h_feet_to_yard : 3 = 1) :
  (main_length * main_width + corridor_length * corridor_width) / 9 = 23.33 := by
sorry

#eval (15 * 12 + 10 * 3) / 9

end NUMINAMATH_CALUDE_carpet_area_calculation_l2454_245491


namespace NUMINAMATH_CALUDE_river_rowing_time_l2454_245427

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

end NUMINAMATH_CALUDE_river_rowing_time_l2454_245427


namespace NUMINAMATH_CALUDE_decomposition_count_l2454_245449

theorem decomposition_count (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (c d : ℕ), (c, d) ∈ s ↔ 
      c * d = p^2 * q^2 ∧ 
      c < d ∧ 
      d < p * q) := by sorry

end NUMINAMATH_CALUDE_decomposition_count_l2454_245449


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2454_245418

/-- Given a quadratic function f(x) = -(x+h)^2 with axis of symmetry at x=-3,
    prove that f(0) = -9 -/
theorem quadratic_function_value (h : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = -(x + h)^2) →
  (∀ x < -3, ∀ y > x, f y > f x) →
  (∀ x > -3, ∀ y > x, f y < f x) →
  f 0 = -9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2454_245418


namespace NUMINAMATH_CALUDE_range_of_a_when_f_has_four_zeros_l2454_245461

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x^2 else Real.exp (-x) + a * x^2

/-- Theorem stating the range of a when f has four zeros -/
theorem range_of_a_when_f_has_four_zeros :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) →
  a < -Real.exp 2 / 4 ∧ ∀ y : ℝ, y < -Real.exp 2 / 4 → ∃ x : ℝ, f x y = 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_f_has_four_zeros_l2454_245461


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2454_245476

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

/-- The binary number 10101110₂ -/
def binaryNumber : ℕ := 174  -- 10101110₂ in decimal is 174

theorem binary_to_quaternary_conversion :
  base2ToBase4 binaryNumber = 2232 := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2454_245476


namespace NUMINAMATH_CALUDE_prank_combinations_l2454_245457

/-- The number of choices for each day of the prank --/
def choices : List Nat := [1, 3, 6, 4, 3]

/-- The total number of combinations --/
def total_combinations : Nat := 216

/-- Theorem stating that the product of the choices equals the total combinations --/
theorem prank_combinations : choices.prod = total_combinations := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l2454_245457


namespace NUMINAMATH_CALUDE_remaining_requests_after_seven_days_l2454_245411

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

end NUMINAMATH_CALUDE_remaining_requests_after_seven_days_l2454_245411


namespace NUMINAMATH_CALUDE_solve_clothing_problem_l2454_245463

def clothing_problem (total : ℕ) (num_loads : ℕ) (pieces_per_load : ℕ) : Prop :=
  let remaining := num_loads * pieces_per_load
  let first_load := total - remaining
  first_load = 19

theorem solve_clothing_problem :
  clothing_problem 39 5 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_clothing_problem_l2454_245463


namespace NUMINAMATH_CALUDE_gcd_16_12_is_4_l2454_245428

theorem gcd_16_12_is_4 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16_12_is_4_l2454_245428


namespace NUMINAMATH_CALUDE_division_remainder_sum_l2454_245478

theorem division_remainder_sum (n : ℕ) : 
  (n / 7 = 13 ∧ n % 7 = 1) → ((n + 9) / 8 + (n + 9) % 8 = 17) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_sum_l2454_245478


namespace NUMINAMATH_CALUDE_y_share_is_36_l2454_245484

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


end NUMINAMATH_CALUDE_y_share_is_36_l2454_245484


namespace NUMINAMATH_CALUDE_committee_election_count_l2454_245440

def group_size : ℕ := 15
def women_count : ℕ := 5
def committee_size : ℕ := 4
def min_women : ℕ := 2

def elect_committee : ℕ := sorry

theorem committee_election_count : 
  elect_committee = 555 := by sorry

end NUMINAMATH_CALUDE_committee_election_count_l2454_245440


namespace NUMINAMATH_CALUDE_second_distribution_boys_l2454_245435

theorem second_distribution_boys (total_amount : ℕ) (first_boys : ℕ) (difference : ℕ) : 
  total_amount = 5040 →
  first_boys = 14 →
  difference = 80 →
  ∃ (second_boys : ℕ), 
    (total_amount / first_boys = total_amount / second_boys + difference) ∧
    second_boys = 18 :=
by sorry

end NUMINAMATH_CALUDE_second_distribution_boys_l2454_245435


namespace NUMINAMATH_CALUDE_total_seats_calculation_l2454_245458

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

end NUMINAMATH_CALUDE_total_seats_calculation_l2454_245458


namespace NUMINAMATH_CALUDE_even_odd_sum_l2454_245424

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x ∈ ℝ -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

/-- Given f and g are even and odd functions respectively, and f(x) - g(x) = x^3 + x^2 + 1,
    prove that f(1) + g(1) = 1 -/
theorem even_odd_sum (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g)
    (h : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_l2454_245424


namespace NUMINAMATH_CALUDE_basketball_weight_l2454_245443

theorem basketball_weight (skateboard_weight : ℝ) (num_skateboards num_basketballs : ℕ) :
  skateboard_weight = 20 →
  num_skateboards = 4 →
  num_basketballs = 5 →
  num_basketballs * (skateboard_weight * num_skateboards / num_basketballs) = num_skateboards * skateboard_weight →
  skateboard_weight * num_skateboards / num_basketballs = 16 :=
by sorry

end NUMINAMATH_CALUDE_basketball_weight_l2454_245443


namespace NUMINAMATH_CALUDE_point_distance_product_l2454_245413

theorem point_distance_product : 
  ∀ y₁ y₂ : ℝ,
  (((1 : ℝ) - 5)^2 + (y₁ - 2)^2 = 12^2) →
  (((1 : ℝ) - 5)^2 + (y₂ - 2)^2 = 12^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -28 :=
by
  sorry

end NUMINAMATH_CALUDE_point_distance_product_l2454_245413


namespace NUMINAMATH_CALUDE_cards_ratio_l2454_245499

/-- Prove the ratio of cards given to initial cards is 1:2 -/
theorem cards_ratio (brandon_cards : ℕ) (malcom_extra : ℕ) (malcom_left : ℕ)
  (h1 : brandon_cards = 20)
  (h2 : malcom_extra = 8)
  (h3 : malcom_left = 14) :
  (brandon_cards + malcom_extra - malcom_left) / (brandon_cards + malcom_extra) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cards_ratio_l2454_245499


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2454_245400

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3/4) :
  (4/x + 1/y) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2454_245400


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l2454_245481

/-- Represents the gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gain : ℕ) : ℚ :=
  (num_gain : ℚ) / (num_sold : ℚ) * 100

/-- Theorem stating that selling 90 pens and gaining the cost of 30 pens results in a 33.33% gain -/
theorem trader_gain_percentage : 
  gain_percentage 90 30 = 33.33 := by sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l2454_245481


namespace NUMINAMATH_CALUDE_two_digit_addition_proof_l2454_245417

theorem two_digit_addition_proof (A B C : ℕ) : 
  A ≠ B → B ≠ C → A ≠ C →
  A ≤ 9 → B ≤ 9 → C ≤ 9 →
  A ≠ 0 → C ≠ 0 →
  (10 * A + B) + (10 * C + B) = 100 * C + C * 10 + 6 →
  B = 8 := by
sorry

end NUMINAMATH_CALUDE_two_digit_addition_proof_l2454_245417


namespace NUMINAMATH_CALUDE_james_dance_duration_l2454_245493

/-- Represents the number of calories burned per hour while walking -/
def calories_walking : ℕ := 300

/-- Represents the number of calories burned per week from dancing -/
def calories_dancing_weekly : ℕ := 2400

/-- Represents the number of times James dances per week -/
def dance_sessions_per_week : ℕ := 4

/-- Represents the ratio of calories burned dancing compared to walking -/
def dancing_to_walking_ratio : ℕ := 2

/-- Proves that James dances for 1 hour each time given the conditions -/
theorem james_dance_duration :
  (calories_dancing_weekly / (dancing_to_walking_ratio * calories_walking)) / dance_sessions_per_week = 1 :=
by sorry

end NUMINAMATH_CALUDE_james_dance_duration_l2454_245493


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_l2454_245429

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

end NUMINAMATH_CALUDE_white_surface_area_fraction_l2454_245429


namespace NUMINAMATH_CALUDE_max_dot_product_ellipses_l2454_245466

/-- The maximum dot product of vectors to points on two specific ellipses -/
theorem max_dot_product_ellipses : 
  let C₁ := {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 9 = 1}
  ∃ (max : ℝ), max = 15 ∧ 
    ∀ (M N : ℝ × ℝ), M ∈ C₁ → N ∈ C₂ → 
      (M.1 * N.1 + M.2 * N.2 : ℝ) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_ellipses_l2454_245466


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2454_245445

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x - 2) = 9 / x) ∧ (x ≠ 2) ∧ (x ≠ 0)

def equation2 (x : ℝ) : Prop := (x / (x + 1) = 2 * x / (3 * x + 3) - 1) ∧ (x ≠ -1) ∧ (3 * x + 3 ≠ 0)

-- State the theorems
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 3 := by sorry

theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -3/4 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l2454_245445


namespace NUMINAMATH_CALUDE_percent_equality_l2454_245431

theorem percent_equality : (25 : ℚ) / 100 * 2004 = (50 : ℚ) / 100 * 1002 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l2454_245431


namespace NUMINAMATH_CALUDE_correct_ages_l2454_245464

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

end NUMINAMATH_CALUDE_correct_ages_l2454_245464


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2454_245474

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 2*X^2 + 3 : Polynomial ℝ) = (X^2 - 2*X + 4) * q + (4*X - 13) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2454_245474


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2454_245441

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -12 * x^2 + 3 * x - 5 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l2454_245441


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2454_245450

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

end NUMINAMATH_CALUDE_quadratic_function_properties_l2454_245450


namespace NUMINAMATH_CALUDE_number_square_relationship_l2454_245480

theorem number_square_relationship (n : ℕ) (h : n = 14) : n + n^2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_number_square_relationship_l2454_245480


namespace NUMINAMATH_CALUDE_find_y_l2454_245477

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(2*y) = 4) (h2 : x = 4) : y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2454_245477


namespace NUMINAMATH_CALUDE_pizza_theorem_l2454_245430

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

end NUMINAMATH_CALUDE_pizza_theorem_l2454_245430


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2454_245423

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2454_245423


namespace NUMINAMATH_CALUDE_trevor_remaining_eggs_l2454_245433

def chicken_eggs : List Nat := [4, 3, 2, 2, 5, 1, 3]

def total_eggs : Nat := chicken_eggs.sum

theorem trevor_remaining_eggs :
  total_eggs - 2 - 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_trevor_remaining_eggs_l2454_245433


namespace NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l2454_245494

/-- A cubic polynomial Q with specific properties -/
structure CubicPolynomial (p : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + p
  at_zero : Q 0 = p
  at_one : Q 1 = 3 * p
  at_neg_one : Q (-1) = 4 * p

/-- The sum of Q(2) and Q(-2) for a specific cubic polynomial Q -/
theorem sum_at_two_and_neg_two (p : ℝ) (Q : CubicPolynomial p) :
  Q.Q 2 + Q.Q (-2) = 22 * p := by
  sorry

end NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l2454_245494


namespace NUMINAMATH_CALUDE_complex_simplification_l2454_245468

theorem complex_simplification :
  let z : ℂ := (2 + Complex.I) / Complex.I
  z = 1 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2454_245468


namespace NUMINAMATH_CALUDE_inequality_holds_l2454_245495

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^3 + k * log x

theorem inequality_holds (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : k ≥ -3) 
  (h2 : x₁ ≥ 1) 
  (h3 : x₂ ≥ 1) 
  (h4 : x₁ > x₂) : 
  (deriv (f k) x₁ + deriv (f k) x₂) / 2 > (f k x₁ - f k x₂) / (x₁ - x₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_l2454_245495


namespace NUMINAMATH_CALUDE_profit_margin_calculation_l2454_245402

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

end NUMINAMATH_CALUDE_profit_margin_calculation_l2454_245402


namespace NUMINAMATH_CALUDE_correct_reasoning_directions_l2454_245479

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

end NUMINAMATH_CALUDE_correct_reasoning_directions_l2454_245479


namespace NUMINAMATH_CALUDE_monotonic_cubic_range_l2454_245487

/-- Given a function f(x) = -x^3 + ax^2 - x - 1 that is monotonic on ℝ,
    the range of the real number a is [-√3, √3]. -/
theorem monotonic_cubic_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ 
  a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_range_l2454_245487


namespace NUMINAMATH_CALUDE_interior_angle_sum_difference_l2454_245420

/-- The sum of interior angles of a convex n-sided polygon in degrees -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

theorem interior_angle_sum_difference (n : ℕ) (h : n ≥ 3) :
  interior_angle_sum (n + 1) - interior_angle_sum n = 180 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_difference_l2454_245420


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_ten_l2454_245437

theorem product_of_five_consecutive_integers_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_ten_l2454_245437


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_average_l2454_245422

theorem quadratic_equation_roots_average (a b : ℝ) (h : a ≠ 0) : 
  let f : ℝ → ℝ := λ x ↦ 3*a*x^2 - 6*a*x + 2*b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_average_l2454_245422


namespace NUMINAMATH_CALUDE_real_part_of_complex_square_l2454_245426

theorem real_part_of_complex_square : Complex.re ((5 : ℂ) + 2 * Complex.I) ^ 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_square_l2454_245426


namespace NUMINAMATH_CALUDE_P_in_third_quadrant_l2454_245470

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

end NUMINAMATH_CALUDE_P_in_third_quadrant_l2454_245470


namespace NUMINAMATH_CALUDE_hotel_meal_expenditure_l2454_245432

theorem hotel_meal_expenditure (num_persons : ℕ) (regular_cost : ℕ) (extra_cost : ℕ) (total_cost : ℕ) : 
  num_persons = 9 →
  regular_cost = 12 →
  extra_cost = 8 →
  total_cost = 117 →
  ∃ (x : ℕ), (num_persons - 1) * regular_cost + (x + extra_cost) = total_cost ∧ x = 13 := by
sorry

end NUMINAMATH_CALUDE_hotel_meal_expenditure_l2454_245432
