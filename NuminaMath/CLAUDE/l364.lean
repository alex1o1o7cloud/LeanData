import Mathlib

namespace NUMINAMATH_CALUDE_coin_combination_theorem_l364_36472

/-- Represents the number of different coin values obtainable -/
def different_values (num_five_cent : ℕ) (num_ten_cent : ℕ) : ℕ :=
  29 - num_five_cent

/-- Theorem stating that given 15 coins with 22 different obtainable values, there must be 8 10-cent coins -/
theorem coin_combination_theorem :
  ∀ (num_five_cent num_ten_cent : ℕ),
    num_five_cent + num_ten_cent = 15 →
    different_values num_five_cent num_ten_cent = 22 →
    num_ten_cent = 8 := by
  sorry

#check coin_combination_theorem

end NUMINAMATH_CALUDE_coin_combination_theorem_l364_36472


namespace NUMINAMATH_CALUDE_wheat_bags_weight_l364_36486

def standard_weight : ℕ := 150
def num_bags : ℕ := 10
def deviations : List ℤ := [-6, -3, -1, -2, 7, 3, 4, -3, -2, 1]

theorem wheat_bags_weight :
  (List.sum deviations = -2) ∧
  (num_bags * standard_weight + List.sum deviations = 1498) :=
sorry

end NUMINAMATH_CALUDE_wheat_bags_weight_l364_36486


namespace NUMINAMATH_CALUDE_increasing_and_second_derivative_l364_36469

open Set

-- Define the properties
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def HasPositiveSecondDerivative (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → (deriv (deriv f)) x > 0

-- Theorem statement
theorem increasing_and_second_derivative (f : ℝ → ℝ) (a b : ℝ) :
  (HasPositiveSecondDerivative f a b → IsIncreasing f a b) ∧
  ∃ g : ℝ → ℝ, IsIncreasing g a b ∧ ∃ x, a < x ∧ x < b ∧ (deriv (deriv g)) x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_and_second_derivative_l364_36469


namespace NUMINAMATH_CALUDE_exponent_division_l364_36470

theorem exponent_division (a : ℝ) : a^10 / a^5 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l364_36470


namespace NUMINAMATH_CALUDE_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l364_36471

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 - 1)

-- Theorem for real number condition
theorem real_number_condition (m : ℝ) : (z m).im = 0 ↔ m = 1 ∨ m = -1 := by sorry

-- Theorem for imaginary number condition
theorem imaginary_number_condition (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 1 ∧ m ≠ -1 := by sorry

-- Theorem for pure imaginary number condition
theorem pure_imaginary_number_condition (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l364_36471


namespace NUMINAMATH_CALUDE_square_of_arithmetic_mean_geq_product_l364_36408

theorem square_of_arithmetic_mean_geq_product (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : c = (a + b) / 2) : c^2 ≥ a * b := by
  sorry

end NUMINAMATH_CALUDE_square_of_arithmetic_mean_geq_product_l364_36408


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l364_36450

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  ∃ (a : ℝ), (a = 1 → |a| = 1) ∧ (|a| = 1 → ¬(a = 1 ↔ |a| = 1)) := by
  sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l364_36450


namespace NUMINAMATH_CALUDE_complex_equation_solution_l364_36427

theorem complex_equation_solution (z : ℂ) :
  Complex.abs z + z = 2 + 4 * Complex.I → z = -3 + 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l364_36427


namespace NUMINAMATH_CALUDE_complex_multiplication_l364_36438

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l364_36438


namespace NUMINAMATH_CALUDE_pocket_money_problem_l364_36447

/-- Pocket money problem -/
theorem pocket_money_problem (a b c d e : ℕ) : 
  (a + b + c + d + e) / 5 = 2300 →
  (a + b) / 2 = 3000 →
  (b + c) / 2 = 2100 →
  (c + d) / 2 = 2750 →
  a = b + 800 →
  d = 3900 := by
sorry

end NUMINAMATH_CALUDE_pocket_money_problem_l364_36447


namespace NUMINAMATH_CALUDE_library_reorganization_l364_36428

theorem library_reorganization (total_books : Nat) (books_per_new_stack : Nat) 
    (h1 : total_books = 1450)
    (h2 : books_per_new_stack = 45) : 
  total_books % books_per_new_stack = 10 := by
  sorry

end NUMINAMATH_CALUDE_library_reorganization_l364_36428


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l364_36416

theorem stratified_sampling_sample_size 
  (high_school_students : ℕ) 
  (junior_high_students : ℕ) 
  (high_school_sample : ℕ) 
  (h1 : high_school_students = 3500)
  (h2 : junior_high_students = 1500)
  (h3 : high_school_sample = 70) :
  let total_students := high_school_students + junior_high_students
  let sample_proportion := high_school_sample / high_school_students
  let total_sample_size := total_students * sample_proportion
  total_sample_size = 100 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l364_36416


namespace NUMINAMATH_CALUDE_symmetry_problem_l364_36499

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given point P -/
def P : Point3D := { x := -2, y := 1, z := 4 }

/-- Given point A -/
def A : Point3D := { x := 1, y := 0, z := 2 }

/-- Reflect a point about the xOy plane -/
def reflectXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Find the point symmetric to a given point about another point -/
def symmetricPoint (p q : Point3D) : Point3D :=
  { x := 2 * p.x - q.x,
    y := 2 * p.y - q.y,
    z := 2 * p.z - q.z }

theorem symmetry_problem :
  reflectXOY P = { x := -2, y := 1, z := -4 } ∧
  symmetricPoint P A = { x := -5, y := 2, z := 6 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_problem_l364_36499


namespace NUMINAMATH_CALUDE_candy_boxes_total_l364_36435

theorem candy_boxes_total (x y z : ℕ) : 
  x = y / 2 → 
  x + z = 24 → 
  y + z = 34 → 
  x + y + z = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_boxes_total_l364_36435


namespace NUMINAMATH_CALUDE_cups_sold_after_day_one_l364_36455

theorem cups_sold_after_day_one 
  (initial_sales : ℕ) 
  (total_days : ℕ) 
  (average_sales : ℚ) 
  (h1 : initial_sales = 86)
  (h2 : total_days = 12)
  (h3 : average_sales = 53) :
  ∃ (daily_sales : ℕ), 
    (initial_sales + (total_days - 1) * daily_sales) / total_days = average_sales ∧
    daily_sales = 50 := by
  sorry

end NUMINAMATH_CALUDE_cups_sold_after_day_one_l364_36455


namespace NUMINAMATH_CALUDE_angle_equality_l364_36433

/-- Given a straight line split into two angles and a triangle with specific properties,
    prove that one of the angles equals 60 degrees. -/
theorem angle_equality (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →  -- Straight line condition
  angle1 + angle3 + 60 = 180 →  -- Triangle angle sum
  angle3 = angle4 →  -- Given equality
  angle4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l364_36433


namespace NUMINAMATH_CALUDE_playground_students_count_l364_36429

/-- Represents the seating arrangement on the playground -/
structure PlaygroundSeating where
  left : Nat
  right : Nat
  front : Nat
  back : Nat

/-- Calculates the total number of students on the playground -/
def totalStudents (s : PlaygroundSeating) : Nat :=
  ((s.left + s.right - 1) * (s.front + s.back - 1))

/-- Theorem stating the total number of students on the playground -/
theorem playground_students_count (yujeong : PlaygroundSeating) 
  (h1 : yujeong.left = 12)
  (h2 : yujeong.right = 11)
  (h3 : yujeong.front = 18)
  (h4 : yujeong.back = 8) :
  totalStudents yujeong = 550 := by
  sorry

#check playground_students_count

end NUMINAMATH_CALUDE_playground_students_count_l364_36429


namespace NUMINAMATH_CALUDE_min_rectangle_side_l364_36490

/-- Given a rectangle with one side of length 1, divided into four smaller rectangles
    by two perpendicular lines, where three of the smaller rectangles have areas of
    at least 1 and the fourth has an area of at least 2, the minimum length of the
    other side of the original rectangle is 3 + 2√2. -/
theorem min_rectangle_side (a b c d : ℝ) : 
  a + b = 1 →
  a * c ≥ 1 →
  a * d ≥ 1 →
  b * c ≥ 1 →
  b * d ≥ 2 →
  c + d ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_rectangle_side_l364_36490


namespace NUMINAMATH_CALUDE_toddler_difference_l364_36476

/-- Represents the group of toddlers playing in the sandbox. -/
structure ToddlerGroup where
  total : ℕ
  forgot_bucket : ℕ
  forgot_shovel : ℕ
  bucket_implies_shovel : Bool

/-- The difference between toddlers with shovel but no bucket and toddlers with bucket -/
def shovel_no_bucket_minus_bucket (group : ToddlerGroup) : ℕ :=
  (group.total - group.forgot_shovel) - (group.total - group.forgot_bucket) - (group.total - group.forgot_bucket)

/-- The main theorem stating the difference is 4 -/
theorem toddler_difference (group : ToddlerGroup) 
  (h1 : group.total = 12)
  (h2 : group.forgot_bucket = 9)
  (h3 : group.forgot_shovel = 2)
  (h4 : group.bucket_implies_shovel = true) :
  shovel_no_bucket_minus_bucket group = 4 := by
  sorry

end NUMINAMATH_CALUDE_toddler_difference_l364_36476


namespace NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l364_36483

theorem pascals_triangle_51st_row_third_number : 
  (Nat.choose 51 2) = 1275 := by sorry

end NUMINAMATH_CALUDE_pascals_triangle_51st_row_third_number_l364_36483


namespace NUMINAMATH_CALUDE_visitor_growth_rate_l364_36437

theorem visitor_growth_rate (initial_visitors : ℕ) (final_visitors : ℕ) 
  (h1 : initial_visitors = 420000) 
  (h2 : final_visitors = 1339100) : 
  ∃ x : ℝ, 42 * (1 + x)^2 = 133.91 := by
  sorry

end NUMINAMATH_CALUDE_visitor_growth_rate_l364_36437


namespace NUMINAMATH_CALUDE_tshirt_cost_l364_36446

def total_spent : ℝ := 199
def num_tshirts : ℕ := 20

theorem tshirt_cost : total_spent / num_tshirts = 9.95 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l364_36446


namespace NUMINAMATH_CALUDE_factorial_quotient_l364_36465

theorem factorial_quotient : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_quotient_l364_36465


namespace NUMINAMATH_CALUDE_erroneous_product_equals_correct_l364_36492

/-- Given a positive integer, reverse its digits --/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is two-digit --/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem erroneous_product_equals_correct (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ is_two_digit b ∧ a * (reverse_digits b) = 180 → a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_erroneous_product_equals_correct_l364_36492


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l364_36420

/-- Given that the monomials 2a^(4)b^(-2m+7) and 3a^(2m)b^(n+2) are like terms, prove that m + n = 3 -/
theorem like_terms_exponent_sum (a b : ℝ) (m n : ℤ) 
  (h : ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 2 * a^4 * b^(-2*m+7) = 3 * a^(2*m) * b^(n+2)) : 
  m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l364_36420


namespace NUMINAMATH_CALUDE_circle_symmetry_about_origin_l364_36441

/-- Given a circle with equation (x-1)^2+(y+2)^2=5, 
    prove that (x+1)^2+(y-2)^2=5 is its symmetric about the origin -/
theorem circle_symmetry_about_origin :
  let original_circle := (fun (x y : ℝ) => (x - 1)^2 + (y + 2)^2 = 5)
  let symmetric_circle := (fun (x y : ℝ) => (x + 1)^2 + (y - 2)^2 = 5)
  ∀ (x y : ℝ), original_circle (-x) (-y) ↔ symmetric_circle x y :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_about_origin_l364_36441


namespace NUMINAMATH_CALUDE_negative_distance_is_west_l364_36496

/-- Represents the direction of travel -/
inductive Direction
| East
| West

/-- Represents the distance and direction of travel -/
structure Travel where
  distance : ℤ
  direction : Direction

/-- Defines the interpretation of signed distances -/
def interpret_travel (d : ℤ) : Travel :=
  if d ≥ 0 then
    { distance := d, direction := Direction.East }
  else
    { distance := -d, direction := Direction.West }

/-- Theorem: A negative distance represents westward travel -/
theorem negative_distance_is_west (d : ℤ) (h : d < 0) :
  (interpret_travel d).direction = Direction.West := by
  sorry

end NUMINAMATH_CALUDE_negative_distance_is_west_l364_36496


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_5_l364_36434

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + a

-- State the theorem
theorem max_value_implies_a_equals_5 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≤ 5) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 5) → a = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_5_l364_36434


namespace NUMINAMATH_CALUDE_trig_expression_value_l364_36458

theorem trig_expression_value (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α ^ 2 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l364_36458


namespace NUMINAMATH_CALUDE_temple_storage_cost_l364_36478

/-- Calculates the total cost for storing items for a group of people -/
def totalCost (numPeople : ℕ) (numPeopleWithGloves : ℕ) (costPerObject : ℕ) : ℕ :=
  let numObjectsPerPerson := 2 + 2 + 1 + 1  -- 2 shoes, 2 socks, 1 mobile, 1 umbrella
  let totalObjects := numPeople * numObjectsPerPerson + numPeopleWithGloves * 2
  totalObjects * costPerObject

/-- Proves that the total cost for the given scenario is 374 dollars -/
theorem temple_storage_cost : totalCost 5 2 11 = 374 := by
  sorry

end NUMINAMATH_CALUDE_temple_storage_cost_l364_36478


namespace NUMINAMATH_CALUDE_lottery_probability_calculation_l364_36444

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallsDrawn : ℕ := 5
def specialBallCount : ℕ := 45

def lotteryProbability : ℚ :=
  1 / (megaBallCount * (winnerBallCount.choose winnerBallsDrawn) * specialBallCount)

theorem lottery_probability_calculation :
  lotteryProbability = 1 / 2861184000 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_calculation_l364_36444


namespace NUMINAMATH_CALUDE_selling_price_difference_difference_is_approximately_56_l364_36440

/-- The difference in selling prices given the original selling price and profit percentages -/
theorem selling_price_difference (original_selling_price : ℝ) : ℝ :=
  let original_profit_rate := 0.1
  let new_purchase_discount := 0.1
  let new_profit_rate := 0.3
  
  let original_purchase_price := original_selling_price / (1 + original_profit_rate)
  let new_purchase_price := original_purchase_price * (1 - new_purchase_discount)
  let new_selling_price := new_purchase_price * (1 + new_profit_rate)
  
  new_selling_price - original_selling_price

/-- The difference in selling prices is approximately $56 -/
theorem difference_is_approximately_56 :
  ∃ ε > 0, abs (selling_price_difference 879.9999999999993 - 56) < ε :=
sorry

end NUMINAMATH_CALUDE_selling_price_difference_difference_is_approximately_56_l364_36440


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l364_36415

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 + Complex.I) ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l364_36415


namespace NUMINAMATH_CALUDE_min_items_for_matching_pair_l364_36456

/-- Represents a tea set with a cup and a saucer -/
structure TeaSet :=
  (cup : Nat)
  (saucer : Nat)

/-- Represents a box containing either cups or saucers -/
inductive Box
| Cups : Box
| Saucers : Box

/-- The number of distinct tea sets -/
def num_sets : Nat := 6

/-- The total number of items in each box -/
def items_per_box : Nat := 6

/-- A function that selects a given number of items from a box -/
def select_items (b : Box) (n : Nat) : Finset Nat := sorry

/-- Predicate to check if a selection guarantees a matching pair -/
def guarantees_matching_pair (cups : Finset Nat) (saucers : Finset Nat) : Prop := sorry

/-- The main theorem stating the minimum number of items needed -/
theorem min_items_for_matching_pair :
  ∀ (n : Nat),
    (∀ (cups saucers : Finset Nat),
      cups.card + saucers.card = n →
      cups.card ≤ items_per_box →
      saucers.card ≤ items_per_box →
      ¬ guarantees_matching_pair cups saucers) ↔
    n < 32 :=
sorry

end NUMINAMATH_CALUDE_min_items_for_matching_pair_l364_36456


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l364_36442

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, a^2+1, 2*a-1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l364_36442


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l364_36402

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_root_existence (a b c : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, x > 0.6 ∧ x < 0.7 ∧ quadratic_function a b c x = 0 :=
by
  have h1 : quadratic_function a b c 0.6 < 0 := by sorry
  have h2 : quadratic_function a b c 0.7 > 0 := by sorry
  sorry

#check quadratic_root_existence

end NUMINAMATH_CALUDE_quadratic_root_existence_l364_36402


namespace NUMINAMATH_CALUDE_first_number_is_five_l364_36474

/-- A sequence where each sum is 1 less than the actual sum of two numbers -/
def SpecialSequence (seq : List (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (a b c : ℕ), (a, b, c) ∈ seq → a + b = c + 1

/-- The first equation in the sequence is x + 7 = 12 -/
def FirstEquation (x : ℕ) : Prop :=
  x + 7 = 12

theorem first_number_is_five (seq : List (ℕ × ℕ × ℕ)) (x : ℕ) 
  (h1 : SpecialSequence seq) (h2 : FirstEquation x) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_five_l364_36474


namespace NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l364_36451

def existing_scores : List ℝ := [76, 82, 79, 84, 91]
def target_mean : ℝ := 85
def sixth_score : ℝ := 98

theorem sixth_score_achieves_target_mean :
  let all_scores := existing_scores ++ [sixth_score]
  (all_scores.sum / all_scores.length : ℝ) = target_mean := by sorry

end NUMINAMATH_CALUDE_sixth_score_achieves_target_mean_l364_36451


namespace NUMINAMATH_CALUDE_a3_range_l364_36462

/-- A sequence {aₙ} is convex if (aₙ + aₙ₊₂)/2 ≤ aₙ₊₁ for all positive integers n. -/
def is_convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)

/-- The function bₙ = n² - 6n + 10 -/
def b (n : ℕ) : ℝ := (n : ℝ)^2 - 6*(n : ℝ) + 10

theorem a3_range (a : ℕ → ℝ) 
  (h_convex : is_convex_sequence a)
  (h_a1 : a 1 = 1)
  (h_a10 : a 10 = 28)
  (h_bound : ∀ n : ℕ, 1 ≤ n → n < 10 → |a n - b n| ≤ 20) :
  7 ≤ a 3 ∧ a 3 ≤ 19 := by sorry

end NUMINAMATH_CALUDE_a3_range_l364_36462


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l364_36481

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem smallest_slope_tangent_line :
  ∃ (x₀ : ℝ), 
    (∀ (x : ℝ), f' x₀ ≤ f' x) ∧ 
    (∀ (x y : ℝ), y = f' x₀ * (x - x₀) + f x₀ ↔ y = -3 * x) :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l364_36481


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l364_36417

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l364_36417


namespace NUMINAMATH_CALUDE_book_length_l364_36494

theorem book_length (P : ℕ) 
  (h1 : 2 * P = 3 * ((2 * P) / 3 - P / 3 + 100)) : P = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_length_l364_36494


namespace NUMINAMATH_CALUDE_cubic_roots_theorem_l364_36452

open Complex

-- Define the cubic equation
def cubic_equation (p q x : ℂ) : Prop := x^3 + p*x + q = 0

-- Define the condition for roots forming an equilateral triangle
def roots_form_equilateral_triangle (r₁ r₂ r₃ : ℂ) : Prop :=
  abs (r₁ - r₂) = Real.sqrt 3 ∧
  abs (r₂ - r₃) = Real.sqrt 3 ∧
  abs (r₃ - r₁) = Real.sqrt 3

theorem cubic_roots_theorem (p q : ℂ) :
  (∃ r₁ r₂ r₃ : ℂ, 
    cubic_equation p q r₁ ∧
    cubic_equation p q r₂ ∧
    cubic_equation p q r₃ ∧
    roots_form_equilateral_triangle r₁ r₂ r₃) →
  arg q = 2 * Real.pi / 3 →
  p + q = -1/2 + (Real.sqrt 3 / 2) * I :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_theorem_l364_36452


namespace NUMINAMATH_CALUDE_power_function_through_point_l364_36491

/-- A power function passing through (3, √3) evaluates to 1/2 at x = 1/4 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x > 0, f x = x^α) →  -- f is a power function
  f 3 = Real.sqrt 3 →     -- f passes through (3, √3)
  f (1/4) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l364_36491


namespace NUMINAMATH_CALUDE_complex_number_location_l364_36407

theorem complex_number_location (a : ℝ) (h : 0 < a ∧ a < 1) :
  let z : ℂ := Complex.mk a (a - 1)
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l364_36407


namespace NUMINAMATH_CALUDE_existence_of_special_fractions_l364_36401

theorem existence_of_special_fractions : 
  ∃ (a b c d : ℕ), (a : ℚ) / b + (c : ℚ) / d = 1 ∧ (a : ℚ) / d + (d : ℚ) / b = 2008 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_fractions_l364_36401


namespace NUMINAMATH_CALUDE_floor_sum_equality_implies_integer_difference_l364_36400

theorem floor_sum_equality_implies_integer_difference (a b c d : ℝ) 
  (h : ∀ (n : ℕ+), ⌊n * a⌋ + ⌊n * b⌋ = ⌊n * c⌋ + ⌊n * d⌋) : 
  (∃ (z : ℤ), a + b = z) ∨ (∃ (z : ℤ), a - c = z) ∨ (∃ (z : ℤ), a - d = z) := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equality_implies_integer_difference_l364_36400


namespace NUMINAMATH_CALUDE_line_passes_through_point_l364_36421

theorem line_passes_through_point (m : ℝ) : m * 1 - 1 + 1 - m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l364_36421


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l364_36461

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 3 + a 4 = 9 →
  a 7 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l364_36461


namespace NUMINAMATH_CALUDE_focus_of_given_parabola_l364_36413

/-- The parabola equation is y = (1/8)x^2 -/
def parabola_equation (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola with equation x^2 = 4py is at (0, p) -/
def focus_of_standard_parabola (p : ℝ) : ℝ × ℝ := (0, p)

/-- The theorem stating that the focus of the parabola y = (1/8)x^2 is at (0, 2) -/
theorem focus_of_given_parabola :
  ∃ (p : ℝ), (∀ x y : ℝ, parabola_equation x y ↔ x^2 = 4*p*y) ∧
             focus_of_standard_parabola p = (0, 2) := by sorry

end NUMINAMATH_CALUDE_focus_of_given_parabola_l364_36413


namespace NUMINAMATH_CALUDE_original_average_proof_l364_36477

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) : 
  n = 12 → 
  new_avg = 72 → 
  new_avg = 2 * original_avg → 
  original_avg = 36 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l364_36477


namespace NUMINAMATH_CALUDE_michael_goals_multiplier_l364_36426

theorem michael_goals_multiplier (bruce_goals : ℕ) (total_goals : ℕ) : 
  bruce_goals = 4 → total_goals = 16 → 
  ∃ x : ℕ, x * bruce_goals = total_goals - bruce_goals ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_michael_goals_multiplier_l364_36426


namespace NUMINAMATH_CALUDE_factorization_equality_l364_36405

theorem factorization_equality (a b : ℝ) : (a^2 + b^2)^2 - 4*a^2*b^2 = (a + b)^2 * (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l364_36405


namespace NUMINAMATH_CALUDE_wrong_to_right_ratio_l364_36443

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) (h1 : total = 48) (h2 : correct = 16) :
  (total - correct) / correct = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_right_ratio_l364_36443


namespace NUMINAMATH_CALUDE_frog_jump_probability_l364_36479

/-- The probability of reaching a vertical side when starting from a given point -/
def P (x y : ℕ) : ℚ :=
  sorry

/-- The square grid size -/
def gridSize : ℕ := 5

theorem frog_jump_probability :
  P 2 1 = 13 / 24 :=
by
  have h1 : ∀ x y, x = 0 ∨ x = gridSize → P x y = 1 := sorry
  have h2 : ∀ x y, y = 0 ∨ y = gridSize → P x y = 0 := sorry
  have h3 : ∀ x y, 0 < x ∧ x < gridSize ∧ 0 < y ∧ y < gridSize →
    P x y = (P (x-1) y + P (x+1) y + P x (y-1) + P x (y+1)) / 4 := sorry
  sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l364_36479


namespace NUMINAMATH_CALUDE_handshake_problem_l364_36410

theorem handshake_problem (n : ℕ) : n * (n - 1) / 2 = 78 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l364_36410


namespace NUMINAMATH_CALUDE_usual_time_calculation_l364_36448

/-- Given a man who takes 24 minutes more to cover a distance when walking at 75% of his usual speed, 
    his usual time to cover this distance is 72 minutes. -/
theorem usual_time_calculation (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0) 
  (h2 : usual_speed > 0)
  (h3 : usual_speed * usual_time = 0.75 * usual_speed * (usual_time + 24)) : 
  usual_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_calculation_l364_36448


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l364_36463

theorem difference_of_squares_special_case : (1025 : ℤ) * 1025 - 1023 * 1027 = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l364_36463


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l364_36430

theorem ellipse_foci_distance (x y : ℝ) :
  (x^2 / 45 + y^2 / 5 = 9) → (∃ f : ℝ, f = 12 * Real.sqrt 10 ∧ f = distance_between_foci) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l364_36430


namespace NUMINAMATH_CALUDE_probability_different_colors_is_83_128_l364_36484

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probabilityDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.yellow + counts.red
  let pBlue := counts.blue / total
  let pYellow := counts.yellow / total
  let pRed := counts.red / total
  pBlue * (pYellow + pRed) + pYellow * (pBlue + pRed) + pRed * (pBlue + pYellow)

/-- The main theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_is_83_128 :
  probabilityDifferentColors ⟨7, 5, 4⟩ = 83 / 128 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_83_128_l364_36484


namespace NUMINAMATH_CALUDE_quadrilateral_point_D_l364_36418

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define a property for parallel sides
def parallel_sides (q : Quadrilateral) : Prop :=
  (q.A.x - q.B.x) * (q.C.y - q.D.y) = (q.A.y - q.B.y) * (q.C.x - q.D.x) ∧
  (q.A.x - q.D.x) * (q.B.y - q.C.y) = (q.A.y - q.D.y) * (q.B.x - q.C.x)

-- Theorem statement
theorem quadrilateral_point_D (q : Quadrilateral) :
  q.A = Point2D.mk (-2) 0 ∧
  q.B = Point2D.mk 6 8 ∧
  q.C = Point2D.mk 8 6 ∧
  parallel_sides q →
  q.D = Point2D.mk 0 (-2) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_point_D_l364_36418


namespace NUMINAMATH_CALUDE_coefficient_x5_eq_11_l364_36453

/-- The coefficient of x^5 in the expansion of (x^2 + x - 1)^5 -/
def coefficient_x5 : ℤ :=
  (Nat.choose 5 0) * (Nat.choose 5 5) -
  (Nat.choose 5 1) * (Nat.choose 4 3) +
  (Nat.choose 5 2) * (Nat.choose 3 1)

/-- Theorem stating that the coefficient of x^5 in (x^2 + x - 1)^5 is 11 -/
theorem coefficient_x5_eq_11 : coefficient_x5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_eq_11_l364_36453


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l364_36488

theorem triangle_angle_measure (A B C : ℝ) : 
  A = 40 → B = 2 * C → A + B + C = 180 → C = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l364_36488


namespace NUMINAMATH_CALUDE_sqrt_x_minus_4_real_range_l364_36412

theorem sqrt_x_minus_4_real_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_4_real_range_l364_36412


namespace NUMINAMATH_CALUDE_bottles_from_625_l364_36482

/-- The number of new bottles that can be made from a given number of plastic bottles -/
def new_bottles (initial : ℕ) : ℕ :=
  if initial < 3 then 0
  else (initial / 5) + new_bottles (initial / 5)

/-- Theorem stating the number of new bottles that can be made from 625 plastic bottles -/
theorem bottles_from_625 : new_bottles 625 = 156 := by
  sorry

end NUMINAMATH_CALUDE_bottles_from_625_l364_36482


namespace NUMINAMATH_CALUDE_thirteen_ts_possible_l364_36431

/-- Represents a grid with horizontal and vertical lines -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Represents a T shape on the grid -/
structure TShape :=
  (intersections : ℕ)

/-- The problem setup -/
def problem_setup : Prop :=
  ∃ (g : Grid) (t : TShape),
    g.horizontal_lines = 9 ∧
    g.vertical_lines = 9 ∧
    t.intersections = 5

/-- The theorem to be proved -/
theorem thirteen_ts_possible (h : problem_setup) : 
  ∃ (n : ℕ), n = 13 ∧ n * 5 ≤ 9 * 9 :=
sorry

end NUMINAMATH_CALUDE_thirteen_ts_possible_l364_36431


namespace NUMINAMATH_CALUDE_shelter_cats_l364_36466

theorem shelter_cats (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 15 / 7 →
  cats / (dogs + 12) = 15 / 11 →
  cats = 45 := by
sorry

end NUMINAMATH_CALUDE_shelter_cats_l364_36466


namespace NUMINAMATH_CALUDE_sin_15_sin_105_equals_1_l364_36464

theorem sin_15_sin_105_equals_1 : 4 * Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_sin_105_equals_1_l364_36464


namespace NUMINAMATH_CALUDE_tangent_identity_l364_36459

theorem tangent_identity (α β γ : Real) (h : α + β + γ = π/4) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_identity_l364_36459


namespace NUMINAMATH_CALUDE_reduced_oil_price_l364_36468

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  original_price : ℝ
  reduced_price : ℝ
  reduction_percentage : ℝ
  additional_quantity : ℝ
  total_cost : ℝ

/-- Theorem stating the reduced price of oil given the conditions --/
theorem reduced_oil_price 
  (scenario : OilPriceReduction)
  (h1 : scenario.reduction_percentage = 0.4)
  (h2 : scenario.additional_quantity = 8)
  (h3 : scenario.total_cost = 2400)
  (h4 : scenario.reduced_price = scenario.original_price * (1 - scenario.reduction_percentage))
  (h5 : scenario.total_cost = (scenario.total_cost / scenario.original_price + scenario.additional_quantity) * scenario.reduced_price) :
  scenario.reduced_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_reduced_oil_price_l364_36468


namespace NUMINAMATH_CALUDE_fruit_display_ratio_l364_36493

theorem fruit_display_ratio (apples oranges bananas : ℕ) : 
  apples = 2 * oranges →
  apples + oranges + bananas = 35 →
  bananas = 5 →
  oranges = 2 * bananas :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_display_ratio_l364_36493


namespace NUMINAMATH_CALUDE_inequality_exists_n_l364_36480

theorem inequality_exists_n : ∃ n : ℕ+, ∀ x : ℝ, x ≥ 0 → (x - 1) * (x^2005 - 2005*x^(n.val + 1) + 2005*x^n.val - 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_exists_n_l364_36480


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l364_36498

/-- The probability of rolling a 1 on a fair six-sided die -/
def p_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a fair six-sided die -/
def p_not_one : ℚ := 5/6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll a 1 -/
def num_ones : ℕ := 4

/-- The number of ways to choose the positions for the non-1 roll -/
def num_arrangements : ℕ := 5

theorem probability_four_ones_in_five_rolls :
  num_arrangements * p_one^num_ones * p_not_one^(num_rolls - num_ones) = 25/7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_l364_36498


namespace NUMINAMATH_CALUDE_circle_diameter_l364_36432

theorem circle_diameter (x y : ℝ) (h : x + y = 100 * Real.pi) : ∃ (r : ℝ), 
  x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ 2 * r = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l364_36432


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l364_36436

theorem joel_age_when_dad_twice_as_old (joel_current_age : ℕ) (dad_current_age : ℕ) : 
  joel_current_age = 5 → dad_current_age = 32 →
  ∃ (years : ℕ), dad_current_age + years = 2 * (joel_current_age + years) ∧ joel_current_age + years = 27 :=
by sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l364_36436


namespace NUMINAMATH_CALUDE_variableCostIncrease_is_ten_percent_l364_36449

/-- Represents the annual breeding cost model for a certain breeder -/
structure BreedingCost where
  fixedCost : ℝ
  initialVariableCost : ℝ
  variableCostIncrease : ℝ

/-- Calculates the total breeding cost for a given year -/
def totalCost (model : BreedingCost) (year : ℕ) : ℝ :=
  model.fixedCost + model.initialVariableCost * (1 + model.variableCostIncrease) ^ (year - 1)

/-- Theorem stating that the percentage increase in variable costs is 10% -/
theorem variableCostIncrease_is_ten_percent (model : BreedingCost) :
  model.fixedCost = 40000 →
  model.initialVariableCost = 26000 →
  totalCost model 3 = 71460 →
  model.variableCostIncrease = 0.1 := by
  sorry


end NUMINAMATH_CALUDE_variableCostIncrease_is_ten_percent_l364_36449


namespace NUMINAMATH_CALUDE_pyramid_tiers_count_l364_36454

/-- Calculates the surface area of a pyramid with n tiers built from 1 cm³ cubes -/
def pyramidSurfaceArea (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

/-- A pyramid built from 1 cm³ cubes with a surface area of 2352 cm² has 24 tiers -/
theorem pyramid_tiers_count : ∃ n : ℕ, pyramidSurfaceArea n = 2352 ∧ n = 24 := by
  sorry

#eval pyramidSurfaceArea 24  -- Should output 2352

end NUMINAMATH_CALUDE_pyramid_tiers_count_l364_36454


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l364_36419

theorem tank_volume_ratio : 
  ∀ (tank1_volume tank2_volume : ℝ), 
  tank1_volume > 0 → tank2_volume > 0 →
  (3/4 : ℝ) * tank1_volume = (5/8 : ℝ) * tank2_volume →
  tank1_volume / tank2_volume = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l364_36419


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l364_36403

theorem min_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧
  (∀ (c d : ℝ), c > 0 → d > 0 → c + d = 1 →
    Real.sqrt (c^2 + 1) + Real.sqrt (d^2 + 4) ≥ Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 4)) ∧
  Real.sqrt (x^2 + 1) + Real.sqrt (y^2 + 4) = Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l364_36403


namespace NUMINAMATH_CALUDE_fair_coin_heads_then_tails_l364_36414

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting tails on a single flip of a fair coin. -/
def prob_tails : ℚ := 1/2

/-- The probability of getting heads on the first flip and tails on the second flip
    of a fair coin. -/
def prob_heads_then_tails : ℚ := prob_heads * prob_tails

theorem fair_coin_heads_then_tails :
  prob_heads_then_tails = 1/4 := by sorry

end NUMINAMATH_CALUDE_fair_coin_heads_then_tails_l364_36414


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l364_36411

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 5 * a^2 + 7 * a + 2 = 1) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ (x : ℝ), 5 * x^2 + 7 * x + 2 = 1 → 3 * x + 2 ≥ m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l364_36411


namespace NUMINAMATH_CALUDE_floor_painting_two_solutions_l364_36495

/-- The number of ordered pairs (a, b) satisfying the floor painting conditions -/
def floor_painting_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let a := p.1
    let b := p.2
    b > a ∧ (a - 4) * (b - 4) = 2 * a * b / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- There are exactly two solutions to the floor painting problem -/
theorem floor_painting_two_solutions : floor_painting_solutions = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_two_solutions_l364_36495


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l364_36473

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -3) :
  ∃ (max : ℝ), max = 6 * Real.sqrt 3 ∧
  ∀ (a b c : ℝ), a + b + c = 3 → a ≥ -1 → b ≥ -2 → c ≥ -3 →
  Real.sqrt (4 * a + 4) + Real.sqrt (4 * b + 8) + Real.sqrt (4 * c + 12) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l364_36473


namespace NUMINAMATH_CALUDE_power_of_three_squared_to_fourth_l364_36409

theorem power_of_three_squared_to_fourth : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_squared_to_fourth_l364_36409


namespace NUMINAMATH_CALUDE_apple_weight_difference_l364_36457

/-- Given two baskets of apples with a total weight and the weight of one basket,
    prove the difference in weight between the baskets. -/
theorem apple_weight_difference (total_weight weight_a : ℕ) 
  (h1 : total_weight = 72)
  (h2 : weight_a = 42) :
  weight_a - (total_weight - weight_a) = 12 := by
  sorry

#check apple_weight_difference

end NUMINAMATH_CALUDE_apple_weight_difference_l364_36457


namespace NUMINAMATH_CALUDE_product_sum_relation_l364_36489

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 14 → b = 8 → b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l364_36489


namespace NUMINAMATH_CALUDE_kevins_prizes_l364_36487

theorem kevins_prizes (total_prizes stuffed_animals frisbees : ℕ) 
  (h1 : total_prizes = 50)
  (h2 : stuffed_animals = 14)
  (h3 : frisbees = 18) :
  total_prizes - (stuffed_animals + frisbees) = 18 := by
  sorry

end NUMINAMATH_CALUDE_kevins_prizes_l364_36487


namespace NUMINAMATH_CALUDE_complex_expression_equality_l364_36425

theorem complex_expression_equality : 
  (125 : ℝ) ^ (1/3) - (-Real.sqrt 3)^2 + (1 + 1/Real.sqrt 2 - Real.sqrt 2) * Real.sqrt 2 - (-1)^2023 = Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l364_36425


namespace NUMINAMATH_CALUDE_square_diagonal_point_theorem_l364_36404

/-- A square with side length 10 -/
structure Square :=
  (E F G H : ℝ × ℝ)
  (is_square : 
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = 100 ∧
    (F.1 - G.1)^2 + (F.2 - G.2)^2 = 100 ∧
    (G.1 - H.1)^2 + (G.2 - H.2)^2 = 100 ∧
    (H.1 - E.1)^2 + (H.2 - E.2)^2 = 100)

/-- Point Q on diagonal EH -/
def Q (s : Square) : ℝ × ℝ := sorry

/-- R1 is the circumcenter of triangle EFQ -/
def R1 (s : Square) : ℝ × ℝ := sorry

/-- R2 is the circumcenter of triangle GHQ -/
def R2 (s : Square) : ℝ × ℝ := sorry

/-- The angle between R1, Q, and R2 is 150° -/
def angle_R1QR2 (s : Square) : ℝ := sorry

theorem square_diagonal_point_theorem (s : Square) 
  (h1 : (Q s).1 > s.E.1 ∧ (Q s).1 < s.H.1)  -- EQ > HQ
  (h2 : angle_R1QR2 s = 150 * π / 180) :
  let EQ := Real.sqrt ((Q s).1 - s.E.1)^2 + ((Q s).2 - s.E.2)^2
  EQ = Real.sqrt 100 + Real.sqrt 150 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_point_theorem_l364_36404


namespace NUMINAMATH_CALUDE_divisibility_condition_l364_36422

theorem divisibility_condition (N : ℤ) : 
  (7 * N + 55) ∣ (N^2 - 71) ↔ N = 57 ∨ N = -8 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l364_36422


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l364_36406

theorem inequality_implies_upper_bound (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) →
  m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l364_36406


namespace NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l364_36439

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem: f(-2) = 11 -/
theorem f_neg_two_eq_eleven : f (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l364_36439


namespace NUMINAMATH_CALUDE_total_candies_l364_36467

/-- The number of boxes Linda has -/
def x : ℕ := 3

/-- The number of candy bags Chloe has -/
def y : ℕ := 2

/-- The number of candy bars Olivia has -/
def z : ℕ := 5

/-- The number of candies in each of Linda's boxes -/
def candies_per_box : ℕ := 2

/-- The number of candies in each of Chloe's bags -/
def candies_per_bag : ℕ := 4

/-- The number of candies equivalent to each of Olivia's candy bars -/
def candies_per_bar : ℕ := 3

/-- The number of candies Linda has -/
def linda_candies : ℕ := 2 * x + 6

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 4 * y + 7

/-- The number of candies Olivia has -/
def olivia_candies : ℕ := 3 * z - 5

theorem total_candies : linda_candies + chloe_candies + olivia_candies = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l364_36467


namespace NUMINAMATH_CALUDE_cone_lateral_area_l364_36423

/-- The lateral area of a cone with base radius 2 cm and height 1 cm is 2√5π cm² -/
theorem cone_lateral_area : 
  let base_radius : ℝ := 2
  let height : ℝ := 1
  let slant_height : ℝ := Real.sqrt (base_radius ^ 2 + height ^ 2)
  let lateral_area : ℝ := π * base_radius * slant_height
  lateral_area = 2 * Real.sqrt 5 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l364_36423


namespace NUMINAMATH_CALUDE_cos_increasing_interval_l364_36497

theorem cos_increasing_interval (a : Real) : 
  (∀ x₁ x₂, -Real.pi ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_increasing_interval_l364_36497


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l364_36485

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l364_36485


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l364_36460

def sum_odd_integers (n : ℕ) : ℕ :=
  let count := (n + 1) / 2
  count * (1 + n) / 2

def sum_even_integers (n : ℕ) : ℕ :=
  let count := n / 2
  count * (2 + n) / 2

theorem odd_even_sum_difference : sum_odd_integers 215 - sum_even_integers 100 = 9114 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l364_36460


namespace NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l364_36475

/-- The volume ratio of an octahedron formed by connecting the centers of adjacent faces of a cube
    to the volume of the cube itself is 1/6, given that the cube has an edge length of 2 units. -/
theorem octahedron_cube_volume_ratio :
  let cube_edge : ℝ := 2
  let cube_volume : ℝ := cube_edge ^ 3
  let octahedron_edge : ℝ := Real.sqrt 8
  let octahedron_volume : ℝ := (Real.sqrt 2 / 3) * octahedron_edge ^ 3
  octahedron_volume / cube_volume = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_octahedron_cube_volume_ratio_l364_36475


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l364_36424

/-- Proves the cost of the second candy in a mixture given specific conditions --/
theorem candy_mixture_cost 
  (weight_first : ℝ) 
  (cost_first : ℝ) 
  (weight_total : ℝ) 
  (cost_mixture : ℝ) : 
  weight_first = 25 ∧ 
  cost_first = 8 ∧ 
  weight_total = 75 ∧ 
  cost_mixture = 6 → 
  (cost_mixture * weight_total - cost_first * weight_first) / (weight_total - weight_first) = 5 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l364_36424


namespace NUMINAMATH_CALUDE_students_liking_both_sports_l364_36445

/-- The number of students who like both basketball and cricket -/
def students_liking_both (b c t : ℕ) : ℕ := b + c - t

/-- Theorem: Given the conditions, prove that 3 students like both basketball and cricket -/
theorem students_liking_both_sports :
  let b := 7  -- number of students who like basketball
  let c := 5  -- number of students who like cricket
  let t := 9  -- total number of students who like basketball or cricket or both
  students_liking_both b c t = 3 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_l364_36445
