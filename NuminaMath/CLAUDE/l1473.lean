import Mathlib

namespace NUMINAMATH_CALUDE_roommate_condition_not_satisfied_l1473_147342

-- Define the functions for John's and Bob's roommates
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 1
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 5

-- Theorem stating that the condition is not satisfied after 3 years
theorem roommate_condition_not_satisfied : f 3 ≠ 2 * g 3 + 5 := by
  sorry

end NUMINAMATH_CALUDE_roommate_condition_not_satisfied_l1473_147342


namespace NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l1473_147394

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

-- Theorem 1
theorem pure_imaginary_product (m : ℝ) :
  (z₁ m * z₂ m).re = 0 → m = 0 := by sorry

-- Theorem 2
theorem imaginary_part_quotient (m : ℝ) :
  z₁ m ^ 2 - 2 * z₁ m + 2 = 0 →
  (z₂ m / z₁ m).im = -1/2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_imaginary_part_quotient_l1473_147394


namespace NUMINAMATH_CALUDE_distance_XY_is_24_l1473_147327

/-- The distance between points X and Y in miles. -/
def distance_XY : ℝ := 24

/-- Yolanda's walking rate in miles per hour. -/
def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour. -/
def bob_rate : ℝ := 4

/-- The distance Bob has walked when they meet, in miles. -/
def bob_distance : ℝ := 12

/-- The time difference between Yolanda and Bob's start, in hours. -/
def time_difference : ℝ := 1

theorem distance_XY_is_24 : 
  distance_XY = yolanda_rate * (bob_distance / bob_rate + time_difference) + bob_distance :=
sorry

end NUMINAMATH_CALUDE_distance_XY_is_24_l1473_147327


namespace NUMINAMATH_CALUDE_min_a_2005_l1473_147345

theorem min_a_2005 (a : Fin 2005 → ℕ+) 
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_product : ∀ i j k, i ≠ j → i < 2005 → j < 2005 → k < 2005 → a i * a j ≠ a k) :
  a 2004 ≥ 2048 := by
  sorry

end NUMINAMATH_CALUDE_min_a_2005_l1473_147345


namespace NUMINAMATH_CALUDE_tobys_money_l1473_147336

/-- 
Proves that if Toby gives 1/7 of his money to each of his two brothers 
and is left with $245, then the initial amount of money he received was $343.
-/
theorem tobys_money (initial_amount : ℚ) : 
  (initial_amount * (1 - 2 * (1 / 7)) = 245) → initial_amount = 343 := by
  sorry

end NUMINAMATH_CALUDE_tobys_money_l1473_147336


namespace NUMINAMATH_CALUDE_simplify_expression_l1473_147341

theorem simplify_expression : 0.3 * 0.8 + 0.1 * 0.5 = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1473_147341


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1473_147386

theorem simplify_fraction_product : 5 * (14 / 3) * (27 / (-35)) * (9 / 7) = -6 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1473_147386


namespace NUMINAMATH_CALUDE_percentage_equality_l1473_147348

theorem percentage_equality (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 20) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1473_147348


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1473_147307

/-- Given an arithmetic sequence {a_n} where a₂ = 9 and a₅ = 33, 
    prove that the common difference is 8. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Definition of arithmetic sequence
  (h2 : a 2 = 9) -- Given: a₂ = 9
  (h3 : a 5 = 33) -- Given: a₅ = 33
  : a 2 - a 1 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1473_147307


namespace NUMINAMATH_CALUDE_no_right_obtuse_triangle_l1473_147392

-- Define a triangle
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define properties of triangles
def Triangle.isValid (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

def Triangle.hasRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.hasObtuseAngle (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: A right obtuse triangle cannot exist
theorem no_right_obtuse_triangle (t : Triangle) :
  t.isValid → ¬(t.hasRightAngle ∧ t.hasObtuseAngle) :=
by
  sorry


end NUMINAMATH_CALUDE_no_right_obtuse_triangle_l1473_147392


namespace NUMINAMATH_CALUDE_line_equations_l1473_147315

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem line_equations (l₁ : Line) (p : Point) :
  l₁.a = 2 ∧ l₁.b = 4 ∧ l₁.c = -1 ∧ p.x = 1 ∧ p.y = -2 →
  (∃ l₂ : Line, l₂.parallel l₁ ∧ l₂.contains p ∧ l₂.a = 1 ∧ l₂.b = 2 ∧ l₂.c = 3) ∧
  (∃ l₂ : Line, l₂.perpendicular l₁ ∧ l₂.contains p ∧ l₂.a = 2 ∧ l₂.b = -1 ∧ l₂.c = -4) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l1473_147315


namespace NUMINAMATH_CALUDE_permutation_17_14_l1473_147362

/-- The falling factorial function -/
def fallingFactorial (n m : ℕ) : ℕ :=
  match m with
  | 0 => 1
  | m + 1 => n * fallingFactorial (n - 1) m

/-- The permutation function -/
def permutation (n m : ℕ) : ℕ := fallingFactorial n m

theorem permutation_17_14 :
  ∃ (n m : ℕ), permutation n m = (17 * 16 * 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4) ∧ n = 17 ∧ m = 14 := by
  sorry

#check permutation_17_14

end NUMINAMATH_CALUDE_permutation_17_14_l1473_147362


namespace NUMINAMATH_CALUDE_cleanup_drive_total_l1473_147374

/-- The total amount of garbage collected by three groups in a cleanup drive -/
theorem cleanup_drive_total (group1_pounds group2_pounds group3_ounces : ℕ) 
  (h1 : group1_pounds = 387)
  (h2 : group2_pounds = group1_pounds - 39)
  (h3 : group3_ounces = 560)
  (h4 : ∀ (x : ℕ), x * 16 = x * 1 * 16) :
  group1_pounds + group2_pounds + (group3_ounces / 16) = 770 := by
  sorry

end NUMINAMATH_CALUDE_cleanup_drive_total_l1473_147374


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_equality_l1473_147387

theorem binomial_expansion_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_equality_l1473_147387


namespace NUMINAMATH_CALUDE_inequalities_and_minimum_l1473_147373

theorem inequalities_and_minimum (a b : ℝ) :
  (a > b ∧ b > 0 → a - 1/a > b - 1/b) ∧
  (a > 0 ∧ b > 0 ∧ 2*a + b = 1 → 2/a + 1/b ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_and_minimum_l1473_147373


namespace NUMINAMATH_CALUDE_congruence_problem_l1473_147305

theorem congruence_problem (x : ℤ) : 
  (5 * x + 11) % 19 = 3 → (3 * x + 7) % 19 = 6 := by sorry

end NUMINAMATH_CALUDE_congruence_problem_l1473_147305


namespace NUMINAMATH_CALUDE_units_digit_product_l1473_147360

theorem units_digit_product (a b c : ℕ) : 
  (3^1004 * 7^1003 * 17^1002) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l1473_147360


namespace NUMINAMATH_CALUDE_pure_imaginary_number_l1473_147363

theorem pure_imaginary_number (x : ℝ) : 
  (((x - 2008) : ℂ) + (x + 2007)*I).re = 0 ∧ (((x - 2008) : ℂ) + (x + 2007)*I).im ≠ 0 → x = 2008 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_number_l1473_147363


namespace NUMINAMATH_CALUDE_box_cube_volume_l1473_147359

theorem box_cube_volume (length width height : ℝ) (num_cubes : ℕ) :
  length = 12 →
  width = 16 →
  height = 6 →
  num_cubes = 384 →
  (length * width * height) / num_cubes = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_box_cube_volume_l1473_147359


namespace NUMINAMATH_CALUDE_mushroom_count_l1473_147343

/-- The number of vegetables Maria needs to cut for her stew -/
def vegetable_counts (potatoes : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
  let carrots := 6 * potatoes
  let onions := 2 * carrots
  let green_beans := onions / 3
  let bell_peppers := 4 * green_beans
  let mushrooms := 3 * bell_peppers
  (potatoes, carrots, onions, green_beans, bell_peppers, mushrooms)

/-- Theorem stating the number of mushrooms Maria needs to cut -/
theorem mushroom_count (potatoes : ℕ) (h : potatoes = 3) :
  (vegetable_counts potatoes).2.2.2.2.2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_count_l1473_147343


namespace NUMINAMATH_CALUDE_fresh_driving_hours_l1473_147318

/-- Calculates the number of hours driving fresh given total distance, total time, and speeds -/
theorem fresh_driving_hours (total_distance : ℝ) (total_time : ℝ) (fresh_speed : ℝ) (fatigued_speed : ℝ) 
  (h1 : total_distance = 152)
  (h2 : total_time = 9)
  (h3 : fresh_speed = 25)
  (h4 : fatigued_speed = 15) :
  ∃ x : ℝ, x = 17 / 10 ∧ fresh_speed * x + fatigued_speed * (total_time - x) = total_distance :=
by
  sorry

end NUMINAMATH_CALUDE_fresh_driving_hours_l1473_147318


namespace NUMINAMATH_CALUDE_alice_prob_is_nine_twentyfifths_l1473_147313

/-- Represents the person holding the ball -/
inductive Person : Type
| Alice : Person
| Bob : Person

/-- The probability of tossing the ball to the other person -/
def toss_prob (p : Person) : ℚ :=
  match p with
  | Person.Alice => 3/5
  | Person.Bob => 1/3

/-- The probability of keeping the ball -/
def keep_prob (p : Person) : ℚ :=
  1 - toss_prob p

/-- The probability that Alice has the ball after two turns, given she starts with it -/
def prob_alice_after_two_turns : ℚ :=
  toss_prob Person.Alice * toss_prob Person.Bob +
  keep_prob Person.Alice * keep_prob Person.Alice

theorem alice_prob_is_nine_twentyfifths :
  prob_alice_after_two_turns = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_alice_prob_is_nine_twentyfifths_l1473_147313


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1473_147335

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 100 →
  x = 50 →
  y = 60 →
  x ∈ S →
  y ∈ S →
  (S.sum id) / S.card = 45 →
  ((S.sum id - (x + y)) / (S.card - 2)) = 44.8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1473_147335


namespace NUMINAMATH_CALUDE_hyperbola_y_axis_implies_m_negative_l1473_147356

/-- A curve represented by the equation x²/m + y²/(1-m) = 1 -/
def is_curve (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2/m + y^2/(1-m) = 1

/-- The curve is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_axis (m : ℝ) : Prop :=
  is_curve m ∧ ∃ c : ℝ, c > 0 ∧ ∀ x y : ℝ, y^2/(1-m) - x^2/(-m) = 1

/-- The theorem stating that if the curve is a hyperbola with foci on the y-axis, then m < 0 -/
theorem hyperbola_y_axis_implies_m_negative (m : ℝ) :
  is_hyperbola_y_axis m → m < 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_y_axis_implies_m_negative_l1473_147356


namespace NUMINAMATH_CALUDE_first_box_nonempty_count_l1473_147377

def total_boxes : Nat := 4
def total_balls : Nat := 3

def ways_with_first_box_nonempty : Nat :=
  total_boxes ^ total_balls - (total_boxes - 1) ^ total_balls

theorem first_box_nonempty_count :
  ways_with_first_box_nonempty = 37 := by
  sorry

end NUMINAMATH_CALUDE_first_box_nonempty_count_l1473_147377


namespace NUMINAMATH_CALUDE_car_trip_duration_l1473_147351

/-- Proves that a car trip with given conditions has a total duration of 6 hours -/
theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) 
  (h1 : initial_speed = 75)
  (h2 : initial_time = 4)
  (h3 : additional_speed = 60)
  (h4 : average_speed = 70) : 
  ∃ (additional_time : ℝ),
    (initial_speed * initial_time + additional_speed * additional_time) / (initial_time + additional_time) = average_speed ∧
    initial_time + additional_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l1473_147351


namespace NUMINAMATH_CALUDE_estimate_passing_papers_l1473_147334

/-- Estimates the number of passing papers in a population based on a sample --/
theorem estimate_passing_papers 
  (total_papers : ℕ) 
  (sample_size : ℕ) 
  (passing_in_sample : ℕ) 
  (h1 : total_papers = 10000)
  (h2 : sample_size = 500)
  (h3 : passing_in_sample = 420) :
  ⌊(total_papers : ℝ) * (passing_in_sample : ℝ) / (sample_size : ℝ)⌋ = 8400 :=
sorry

end NUMINAMATH_CALUDE_estimate_passing_papers_l1473_147334


namespace NUMINAMATH_CALUDE_number_equation_l1473_147322

theorem number_equation : ∃ n : ℝ, 2 * 2 + n = 6 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1473_147322


namespace NUMINAMATH_CALUDE_right_angle_zdels_l1473_147311

/-- The number of zdels in a full circle -/
def zdels_in_full_circle : ℕ := 400

/-- The fraction of a full circle that constitutes a right angle -/
def right_angle_fraction : ℚ := 1 / 3

/-- The number of zdels in a right angle -/
def zdels_in_right_angle : ℚ := zdels_in_full_circle * right_angle_fraction

theorem right_angle_zdels : zdels_in_right_angle = 400 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_zdels_l1473_147311


namespace NUMINAMATH_CALUDE_sum_g_h_equals_negative_eight_l1473_147300

theorem sum_g_h_equals_negative_eight (g h : ℝ) :
  (∀ d : ℝ, (8*d^2 - 4*d + g) * (4*d^2 + h*d + 7) = 32*d^4 + (4*h-16)*d^3 - (14*d^2 - 28*d - 56)) →
  g + h = -8 := by sorry

end NUMINAMATH_CALUDE_sum_g_h_equals_negative_eight_l1473_147300


namespace NUMINAMATH_CALUDE_fruit_stand_problem_l1473_147378

/-- Proves that the price of each apple is $0.80 given the conditions of the fruit stand problem -/
theorem fruit_stand_problem (total_cost : ℝ) (total_fruits : ℕ) (banana_price : ℝ) 
  (h1 : total_cost = 5.60)
  (h2 : total_fruits = 9)
  (h3 : banana_price = 0.60) :
  ∃ (apple_price : ℝ) (num_apples : ℕ),
    apple_price = 0.80 ∧
    num_apples ≤ total_fruits ∧
    apple_price * num_apples + banana_price * (total_fruits - num_apples) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_fruit_stand_problem_l1473_147378


namespace NUMINAMATH_CALUDE_sticker_difference_l1473_147340

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23

theorem sticker_difference : 
  total_stickers - first_box_stickers - first_box_stickers = 12 := by
  sorry

end NUMINAMATH_CALUDE_sticker_difference_l1473_147340


namespace NUMINAMATH_CALUDE_series_sum_equals_one_third_l1473_147357

/-- The sum of the infinite series ∑(k=1 to ∞) [2^k / (8^k - 1)] is equal to 1/3 -/
theorem series_sum_equals_one_third :
  ∑' k, (2 : ℝ)^k / ((8 : ℝ)^k - 1) = 1/3 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_third_l1473_147357


namespace NUMINAMATH_CALUDE_solve_equation_l1473_147314

theorem solve_equation (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (6 * x + 45)) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1473_147314


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l1473_147347

/-- Represents a cube with smaller cubes removed from its corners -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (c : ModifiedCube) : ℕ :=
  -- The actual calculation would go here
  24

/-- Theorem stating that a cube of side length 4 with unit cubes removed from corners has 24 edges -/
theorem modified_cube_edge_count :
  ∀ (c : ModifiedCube), 
    c.originalSideLength = 4 ∧ 
    c.removedCubeSideLength = 1 → 
    edgeCount c = 24 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l1473_147347


namespace NUMINAMATH_CALUDE_pizza_slices_l1473_147371

theorem pizza_slices (total_slices : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) : 
  total_slices = 16 → 
  num_pizzas = 2 → 
  total_slices = num_pizzas * slices_per_pizza → 
  slices_per_pizza = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l1473_147371


namespace NUMINAMATH_CALUDE_five_digit_with_eight_count_l1473_147323

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def contains_eight (n : ℕ) : Prop := ∃ (d : ℕ), d < 5 ∧ (n / 10^d) % 10 = 8

def count_five_digit : ℕ := 90000

def count_without_eight : ℕ := 52488

theorem five_digit_with_eight_count :
  (count_five_digit - count_without_eight) = 37512 :=
sorry

end NUMINAMATH_CALUDE_five_digit_with_eight_count_l1473_147323


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1473_147389

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of side BC -/
  bc : ℝ
  /-- The perpendicular distance from point P on BC to side AB -/
  p_to_ab : ℝ
  /-- The perpendicular distance from point P on BC to side AC -/
  p_to_ac : ℝ
  /-- Assertion that BC = 65 -/
  h_bc : bc = 65
  /-- Assertion that the perpendicular distance from P to AB is 24 -/
  h_p_to_ab : p_to_ab = 24
  /-- Assertion that the perpendicular distance from P to AC is 36 -/
  h_p_to_ac : p_to_ac = 36

/-- The area of the isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ := 2535

/-- Theorem stating that the area of the isosceles triangle is 2535 -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : area t = 2535 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1473_147389


namespace NUMINAMATH_CALUDE_prime_square_mod_six_l1473_147358

theorem prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  p^2 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_six_l1473_147358


namespace NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l1473_147326

theorem lemonade_pitcher_capacity (total_glasses : ℕ) (total_pitchers : ℕ) 
  (h1 : total_glasses = 30) (h2 : total_pitchers = 6) :
  total_glasses / total_pitchers = 5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_pitcher_capacity_l1473_147326


namespace NUMINAMATH_CALUDE_max_pairs_sum_l1473_147346

theorem max_pairs_sum (n : ℕ) (h : n = 3009) :
  let S := Finset.range n
  ∃ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    k = 1504 ∧
    pairs.card = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (m : ℕ) (pairs' : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ n) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      m = pairs'.card →
      m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l1473_147346


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1473_147376

/-- Given a line with equation y - 3 = -3(x - 6), prove that the sum of its x-intercept and y-intercept is 28. -/
theorem line_intercepts_sum (x y : ℝ) :
  (y - 3 = -3 * (x - 6)) →
  (∃ x_int y_int : ℝ,
    (y_int - 3 = -3 * (x_int - 6) ∧ y_int = 0) ∧
    (0 - 3 = -3 * (0 - 6) ∧ y = y_int) ∧
    x_int + y_int = 28) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1473_147376


namespace NUMINAMATH_CALUDE_dodecagon_enclosed_by_dodecagons_l1473_147320

/-- The number of sides of the inner regular polygon -/
def inner_sides : ℕ := 12

/-- The number of outer regular polygons -/
def num_outer_polygons : ℕ := 12

/-- The number of sides of each outer regular polygon -/
def outer_sides : ℕ := 12

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ :=
  (n - 2 : ℚ) * 180 / n

/-- The exterior angle of a regular polygon with n sides -/
def exterior_angle (n : ℕ) : ℚ :=
  360 / n

/-- Theorem stating that a regular 12-sided polygon can be exactly enclosed
    by 12 regular 12-sided polygons -/
theorem dodecagon_enclosed_by_dodecagons :
  2 * (exterior_angle outer_sides / 2) = 
  180 - interior_angle inner_sides :=
sorry

end NUMINAMATH_CALUDE_dodecagon_enclosed_by_dodecagons_l1473_147320


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1473_147333

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 4) ↔ 
  (-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ 
  (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2) := by
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1473_147333


namespace NUMINAMATH_CALUDE_division_result_l1473_147303

theorem division_result : (0.075 : ℚ) / (0.005 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1473_147303


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_implies_a_eq_two_l1473_147396

/-- If (1 + ai) / (2 - i) is a pure imaginary number, then a = 2 -/
theorem pure_imaginary_fraction_implies_a_eq_two (a : ℝ) :
  (∃ b : ℝ, (1 + a * Complex.I) / (2 - Complex.I) = b * Complex.I) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_implies_a_eq_two_l1473_147396


namespace NUMINAMATH_CALUDE_janette_camping_duration_l1473_147338

/-- Calculates the number of days Janette went camping based on her beef jerky consumption --/
def camping_days (
  initial_jerky : ℕ
  ) (daily_consumption : ℕ
  ) (final_jerky : ℕ
  ) : ℕ :=
  (initial_jerky - 2 * final_jerky) / daily_consumption

/-- Proves that Janette went camping for 5 days --/
theorem janette_camping_duration :
  camping_days 40 4 10 = 5 := by
  sorry

#eval camping_days 40 4 10

end NUMINAMATH_CALUDE_janette_camping_duration_l1473_147338


namespace NUMINAMATH_CALUDE_orange_harvest_days_l1473_147367

/-- The number of days required to harvest a given number of sacks of ripe oranges -/
def harvest_days (total_sacks : ℕ) (sacks_per_day : ℕ) : ℕ :=
  total_sacks / sacks_per_day

/-- Theorem stating that it takes 25 days to harvest 2050 sacks of ripe oranges when harvesting 82 sacks per day -/
theorem orange_harvest_days : harvest_days 2050 82 = 25 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_days_l1473_147367


namespace NUMINAMATH_CALUDE_pedestrian_meets_cart_time_l1473_147304

/-- Represents a participant in the scenario -/
inductive Participant
| Pedestrian
| Cyclist
| Cart
| Car

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents an event involving two participants -/
structure Event where
  participant1 : Participant
  participant2 : Participant
  time : Time

/-- The scenario with all participants and events -/
structure Scenario where
  cyclist_overtakes_pedestrian : Event
  pedestrian_meets_car : Event
  cyclist_meets_cart : Event
  cyclist_meets_car : Event
  car_meets_cyclist : Event
  car_meets_pedestrian : Event
  car_overtakes_cart : Event

def is_valid_scenario (s : Scenario) : Prop :=
  s.cyclist_overtakes_pedestrian.time = Time.mk 10 0 ∧
  s.pedestrian_meets_car.time = Time.mk 11 0 ∧
  s.cyclist_meets_cart.time.hours - s.cyclist_overtakes_pedestrian.time.hours = 
    s.cyclist_meets_car.time.hours - s.cyclist_meets_cart.time.hours ∧
  s.cyclist_meets_cart.time.minutes - s.cyclist_overtakes_pedestrian.time.minutes = 
    s.cyclist_meets_car.time.minutes - s.cyclist_meets_cart.time.minutes ∧
  s.car_meets_pedestrian.time.hours - s.car_meets_cyclist.time.hours = 
    s.car_overtakes_cart.time.hours - s.car_meets_pedestrian.time.hours ∧
  s.car_meets_pedestrian.time.minutes - s.car_meets_cyclist.time.minutes = 
    s.car_overtakes_cart.time.minutes - s.car_meets_pedestrian.time.minutes

theorem pedestrian_meets_cart_time (s : Scenario) (h : is_valid_scenario s) :
  ∃ (t : Event), t.participant1 = Participant.Pedestrian ∧ 
                 t.participant2 = Participant.Cart ∧ 
                 t.time = Time.mk 10 40 :=
sorry

end NUMINAMATH_CALUDE_pedestrian_meets_cart_time_l1473_147304


namespace NUMINAMATH_CALUDE_lunchroom_students_l1473_147383

/-- The number of students sitting at each table -/
def students_per_table : ℕ := 6

/-- The number of tables in the lunchroom -/
def number_of_tables : ℕ := 34

/-- The total number of students in the lunchroom -/
def total_students : ℕ := students_per_table * number_of_tables

theorem lunchroom_students : total_students = 204 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_students_l1473_147383


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1473_147310

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n, a (n + 1) = q * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → a n < a m

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : increasing_sequence a)
  (h3 : a 5 ^ 2 = a 10)
  (h4 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  a 5 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1473_147310


namespace NUMINAMATH_CALUDE_line_point_sum_l1473_147369

/-- The line equation y = -5/3x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T is on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * P.1 + (1 - t) * Q.1 ∧ 
  s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop := 
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 = 
  4 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

/-- Theorem statement -/
theorem line_point_sum (r s : ℝ) : 
  line_equation r s → T_on_PQ r s → area_condition r s → r + s = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_line_point_sum_l1473_147369


namespace NUMINAMATH_CALUDE_computer_price_is_150_l1473_147337

/-- The price per computer in a factory with given production and earnings -/
def price_per_computer (daily_production : ℕ) (weekly_earnings : ℕ) : ℚ :=
  weekly_earnings / (daily_production * 7)

/-- Theorem stating that the price per computer is $150 -/
theorem computer_price_is_150 :
  price_per_computer 1500 1575000 = 150 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_is_150_l1473_147337


namespace NUMINAMATH_CALUDE_max_excellent_videos_l1473_147382

/-- A micro-video with likes and expert score -/
structure MicroVideo where
  likes : ℕ
  expertScore : ℕ

/-- Determines if one video is not inferior to another -/
def notInferior (a b : MicroVideo) : Prop :=
  a.likes ≥ b.likes ∨ a.expertScore ≥ b.expertScore

/-- Determines if a video is excellent among a list of videos -/
def isExcellent (v : MicroVideo) (videos : List MicroVideo) : Prop :=
  ∀ u ∈ videos, notInferior v u

/-- The main theorem to prove -/
theorem max_excellent_videos (videos : List MicroVideo) 
  (h : videos.length = 5) :
  ∃ (excellentVideos : List MicroVideo), 
    excellentVideos.length ≤ 5 ∧ 
    ∀ v ∈ excellentVideos, isExcellent v videos ∧
    ∀ v ∈ videos, isExcellent v videos → v ∈ excellentVideos :=
  sorry

end NUMINAMATH_CALUDE_max_excellent_videos_l1473_147382


namespace NUMINAMATH_CALUDE_journey_average_mpg_l1473_147344

/-- Represents a car's journey with odometer readings and gas fill-ups -/
structure CarJourney where
  initial_odometer : ℕ
  initial_gas : ℕ
  intermediate_odometer : ℕ
  intermediate_gas : ℕ
  final_odometer : ℕ
  final_gas : ℕ

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (journey : CarJourney) : ℚ :=
  let total_distance : ℕ := journey.final_odometer - journey.initial_odometer
  let total_gas : ℕ := journey.initial_gas + journey.intermediate_gas + journey.final_gas
  (total_distance : ℚ) / total_gas

/-- Theorem stating that the average mpg for the given journey is 15.2 -/
theorem journey_average_mpg :
  let journey : CarJourney := {
    initial_odometer := 35200,
    initial_gas := 10,
    intermediate_odometer := 35480,
    intermediate_gas := 15,
    final_odometer := 35960,
    final_gas := 25
  }
  average_mpg journey = 152 / 10 := by sorry

end NUMINAMATH_CALUDE_journey_average_mpg_l1473_147344


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_zero_solution_l1473_147379

theorem smallest_n_for_unique_zero_solution :
  ∃ (n : ℕ), n ≥ 1 ∧
  (∀ (a b c d : ℤ), a^2 + b^2 + c^2 - n * d^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∧
  (∀ (m : ℕ), m < n →
    ∃ (a b c d : ℤ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0) ∧ a^2 + b^2 + c^2 - m * d^2 = 0) ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_zero_solution_l1473_147379


namespace NUMINAMATH_CALUDE_cab_driver_income_l1473_147390

theorem cab_driver_income (incomes : Fin 5 → ℕ) 
  (h1 : incomes 1 = 50)
  (h2 : incomes 2 = 60)
  (h3 : incomes 3 = 65)
  (h4 : incomes 4 = 70)
  (h_avg : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 58) :
  incomes 0 = 45 := by
sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1473_147390


namespace NUMINAMATH_CALUDE_repair_cost_is_288_l1473_147316

/-- The amount spent on repairs for a scooter, given the purchase price, selling price, and gain percentage. -/
def repair_cost (purchase_price selling_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  selling_price * (1 - gain_percentage / 100) - purchase_price

/-- Theorem stating that the repair cost is $288 given the specific conditions. -/
theorem repair_cost_is_288 :
  repair_cost 900 1320 10 = 288 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_288_l1473_147316


namespace NUMINAMATH_CALUDE_seagrass_study_l1473_147380

/-- Represents the sample statistics for a town -/
structure TownSample where
  size : ℕ
  mean : ℝ
  variance : ℝ

/-- Represents the competition probabilities for town A -/
structure CompetitionProbs where
  win_in_A : ℝ
  win_in_B : ℝ

theorem seagrass_study (town_A : TownSample) (town_B : TownSample) (probs : CompetitionProbs)
  (h_A_size : town_A.size = 12)
  (h_A_mean : town_A.mean = 18)
  (h_A_var : town_A.variance = 19)
  (h_B_size : town_B.size = 18)
  (h_B_mean : town_B.mean = 36)
  (h_B_var : town_B.variance = 70)
  (h_prob_A : probs.win_in_A = 3/5)
  (h_prob_B : probs.win_in_B = 1/2) :
  let total_mean := (town_A.size * town_A.mean + town_B.size * town_B.mean) / (town_A.size + town_B.size)
  let total_variance := (1 / (town_A.size + town_B.size)) *
    (town_A.size * town_A.variance + town_A.size * (town_A.mean - total_mean)^2 +
     town_B.size * town_B.variance + town_B.size * (town_B.mean - total_mean)^2)
  let expected_score := 0 * (1 - probs.win_in_A)^2 + 1 * (2 * probs.win_in_A * (1 - probs.win_in_B) * (1 - probs.win_in_A)) +
    2 * (1 - (1 - probs.win_in_A)^2 - 2 * probs.win_in_A * (1 - probs.win_in_B) * (1 - probs.win_in_A))
  total_mean = 28.8 ∧ total_variance = 127.36 ∧ expected_score = 36/25 := by
  sorry

end NUMINAMATH_CALUDE_seagrass_study_l1473_147380


namespace NUMINAMATH_CALUDE_bin_game_expected_win_l1473_147306

/-- The number of yellow balls in the bin -/
def yellow_balls : ℕ := 7

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 3

/-- The amount won when drawing a yellow ball -/
def yellow_win : ℚ := 3

/-- The amount lost when drawing a blue ball -/
def blue_loss : ℚ := 1

/-- The expected amount won from playing the game -/
def expected_win : ℚ := 1

/-- Theorem stating that the expected amount won is 1 dollar
    given the specified number of yellow and blue balls and win/loss amounts -/
theorem bin_game_expected_win :
  (yellow_balls * yellow_win + blue_balls * (-blue_loss)) / (yellow_balls + blue_balls) = expected_win :=
sorry

end NUMINAMATH_CALUDE_bin_game_expected_win_l1473_147306


namespace NUMINAMATH_CALUDE_sum_from_simple_interest_and_true_discount_l1473_147354

/-- Given a sum, time, and rate, if the simple interest is 85 and the true discount is 80, then the sum is 1360 -/
theorem sum_from_simple_interest_and_true_discount 
  (P T R : ℝ) 
  (h_simple_interest : (P * T * R) / 100 = 85)
  (h_true_discount : (P * T * R) / (100 + T * R) = 80) :
  P = 1360 := by
  sorry

end NUMINAMATH_CALUDE_sum_from_simple_interest_and_true_discount_l1473_147354


namespace NUMINAMATH_CALUDE_council_vote_change_l1473_147309

theorem council_vote_change (total_members : ℕ) 
  (initial_for initial_against : ℚ) 
  (revote_for revote_against : ℚ) : 
  total_members = 500 ∧ 
  initial_for + initial_against = total_members ∧
  initial_against > initial_for ∧
  revote_for + revote_against = total_members ∧
  revote_for > revote_against ∧
  revote_for - revote_against = (3/2) * (initial_against - initial_for) ∧
  revote_for = (11/10) * initial_against →
  revote_for - initial_for = 156.25 := by
sorry

end NUMINAMATH_CALUDE_council_vote_change_l1473_147309


namespace NUMINAMATH_CALUDE_tax_fraction_proof_l1473_147385

theorem tax_fraction_proof (gross_income : ℝ) (car_payment : ℝ) (car_payment_percentage : ℝ) :
  gross_income = 3000 →
  car_payment = 400 →
  car_payment_percentage = 0.20 →
  car_payment = car_payment_percentage * (gross_income * (1 - (1/3))) →
  1/3 = (gross_income - (car_payment / car_payment_percentage)) / gross_income :=
by sorry

end NUMINAMATH_CALUDE_tax_fraction_proof_l1473_147385


namespace NUMINAMATH_CALUDE_quiz_competition_score_l1473_147308

/-- Calculates the final score in a quiz competition given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - (incorrect : ℚ) * (1 / 4)

/-- Represents the quiz competition problem -/
theorem quiz_competition_score :
  let total_questions : ℕ := 35
  let correct_answers : ℕ := 17
  let incorrect_answers : ℕ := 12
  let unanswered_questions : ℕ := 6
  correct_answers + incorrect_answers + unanswered_questions = total_questions →
  calculate_score correct_answers incorrect_answers unanswered_questions = 14 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_score_l1473_147308


namespace NUMINAMATH_CALUDE_second_rectangle_height_l1473_147353

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Proves that the height of the second rectangle is 6 inches -/
theorem second_rectangle_height (r1 r2 : Rectangle) 
  (h1 : r1.width = 4)
  (h2 : r1.height = 5)
  (h3 : r2.width = 3)
  (h4 : area r1 = area r2 + 2) : 
  r2.height = 6 := by
  sorry

#check second_rectangle_height

end NUMINAMATH_CALUDE_second_rectangle_height_l1473_147353


namespace NUMINAMATH_CALUDE_no_equivalent_expressions_l1473_147301

theorem no_equivalent_expressions (x : ℝ) (h : x > 0) : 
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ 2*(y+1)^y) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ (y+1)^(2*y+2)) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ 2*(y+0.5*y)^y) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ (2*y+2)^(2*y+2)) :=
by sorry

end NUMINAMATH_CALUDE_no_equivalent_expressions_l1473_147301


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1473_147329

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s : ℝ, s = -(b / a) ∧ s = x + y) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2023 * x - 2024
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s : ℝ, s = -2023 ∧ s = x + y) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1473_147329


namespace NUMINAMATH_CALUDE_intersecting_circles_B_coords_l1473_147349

/-- Two circles with centers on the line y = 1 - x, intersecting at points A and B -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  centers_on_line : ∀ c, c = O₁ ∨ c = O₂ → c.2 = 1 - c.1
  A_coords : A = (-7, 9)

/-- The theorem stating that point B has coordinates (-8, 8) -/
theorem intersecting_circles_B_coords (circles : IntersectingCircles) : 
  circles.B = (-8, 8) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_B_coords_l1473_147349


namespace NUMINAMATH_CALUDE_projective_transformation_uniqueness_l1473_147330

/-- A projective transformation on a line -/
structure ProjectiveTransformation (α : Type*) where
  transform : α → α

/-- The statement that two projective transformations are equal if they agree on three distinct points -/
theorem projective_transformation_uniqueness 
  {α : Type*} [LinearOrder α] 
  (P Q : ProjectiveTransformation α) 
  (A B C : α) 
  (hABC : A < B ∧ B < C) 
  (hP : P.transform A = Q.transform A ∧ 
        P.transform B = Q.transform B ∧ 
        P.transform C = Q.transform C) : 
  P = Q :=
sorry

end NUMINAMATH_CALUDE_projective_transformation_uniqueness_l1473_147330


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1473_147366

def f (x : ℝ) := -x^2 + 4*x + 5

theorem min_value_of_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc 1 4 ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ Set.Icc 1 4 → f y ≥ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l1473_147366


namespace NUMINAMATH_CALUDE_special_polynomial_value_at_one_l1473_147398

/-- A non-constant quadratic polynomial satisfying the given equation -/
def SpecialPolynomial (g : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, g x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x)^2 / (2023 * x))

theorem special_polynomial_value_at_one
  (g : ℝ → ℝ) (h : SpecialPolynomial g) : g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_at_one_l1473_147398


namespace NUMINAMATH_CALUDE_min_value_expression_l1473_147319

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 12) / Real.sqrt (x - 4) ≥ 8 ∧ ∃ y : ℝ, y > 4 ∧ (y + 12) / Real.sqrt (y - 4) = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1473_147319


namespace NUMINAMATH_CALUDE_largest_possible_BD_l1473_147328

/-- A cyclic quadrilateral with side lengths that are distinct primes less than 20 -/
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  cyclic : Bool
  distinct_primes : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  all_prime : Nat.Prime AB ∧ Nat.Prime BC ∧ Nat.Prime CD ∧ Nat.Prime DA
  all_less_than_20 : AB < 20 ∧ BC < 20 ∧ CD < 20 ∧ DA < 20
  AB_is_11 : AB = 11
  product_condition : BC * CD = AB * DA

/-- The diagonal BD of the cyclic quadrilateral -/
def diagonal_BD (q : CyclicQuadrilateral) : ℝ := sorry

theorem largest_possible_BD (q : CyclicQuadrilateral) :
  ∃ (max_bd : ℝ), diagonal_BD q ≤ max_bd ∧ max_bd = Real.sqrt 290 := by
  sorry

end NUMINAMATH_CALUDE_largest_possible_BD_l1473_147328


namespace NUMINAMATH_CALUDE_correct_mean_after_errors_l1473_147324

theorem correct_mean_after_errors (n : ℕ) (initial_mean : ℚ) 
  (error1_actual error1_copied error2_actual error2_copied error3_actual error3_copied : ℚ) :
  n = 50 ∧ 
  initial_mean = 325 ∧
  error1_actual = 200 ∧ error1_copied = 150 ∧
  error2_actual = 175 ∧ error2_copied = 220 ∧
  error3_actual = 592 ∧ error3_copied = 530 →
  let incorrect_sum := n * initial_mean
  let correction := (error1_actual - error1_copied) + (error3_actual - error3_copied) - (error2_actual - error2_copied)
  let corrected_sum := incorrect_sum + correction
  let correct_mean := corrected_sum / n
  correct_mean = 326.34 := by
sorry


end NUMINAMATH_CALUDE_correct_mean_after_errors_l1473_147324


namespace NUMINAMATH_CALUDE_system_solution_implies_2a_minus_3b_equals_6_l1473_147384

theorem system_solution_implies_2a_minus_3b_equals_6
  (a b : ℝ)
  (eq1 : a * 2 - b * 1 = 4)
  (eq2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_implies_2a_minus_3b_equals_6_l1473_147384


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l1473_147397

/-- Represents an ellipse with center (h, k), semi-major axis a, and semi-minor axis b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse centered at (3, -5) with vertical semi-major axis 8 and semi-minor axis 4,
    the sum of h, k, a, and b equals 10 -/
theorem ellipse_sum_theorem (e : Ellipse)
    (h_center : e.h = 3 ∧ e.k = -5)
    (h_axes : e.a = 8 ∧ e.b = 4)
    (h_vertical : e.a > e.b) :
    ellipse_sum e = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l1473_147397


namespace NUMINAMATH_CALUDE_courtyard_width_l1473_147312

theorem courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length brick_width : ℝ) :
  length = 25 ∧ 
  num_bricks = 20000 ∧ 
  brick_length = 0.2 ∧ 
  brick_width = 0.1 → 
  (num_bricks : ℝ) * brick_length * brick_width / length = 16 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_l1473_147312


namespace NUMINAMATH_CALUDE_largest_x_value_l1473_147399

theorem largest_x_value (x : ℝ) :
  (x / 3 + 1 / (7 * x) = 1 / 2) →
  x ≤ (21 + Real.sqrt 105) / 28 ∧
  ∃ y : ℝ, y / 3 + 1 / (7 * y) = 1 / 2 ∧ y = (21 + Real.sqrt 105) / 28 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l1473_147399


namespace NUMINAMATH_CALUDE_point_adding_procedure_l1473_147361

theorem point_adding_procedure (x : ℕ+) : ∃ x, 9 * x - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_point_adding_procedure_l1473_147361


namespace NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1473_147370

/-- The number of values of 'a' for which the line y = x + a passes through
    the vertex of the parabola y = x^2 + a^2 is exactly 2. -/
theorem line_passes_through_parabola_vertex :
  let line := λ (x a : ℝ) => x + a
  let parabola := λ (x a : ℝ) => x^2 + a^2
  let vertex := λ (a : ℝ) => (0, a^2)
  ∃! (s : Finset ℝ), (∀ a ∈ s, line 0 a = (vertex a).2) ∧ s.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1473_147370


namespace NUMINAMATH_CALUDE_circle_equation_l1473_147395

-- Define the line y = x + 1
def line_symmetry (x y : ℝ) : Prop := y = x + 1

-- Define the point P
def point_P : ℝ × ℝ := (-2, 1)

-- Define the line 3x + 4y - 11 = 0
def line_intersect (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0

-- Define the length of AB
def length_AB : ℝ := 6

-- Define the circle C
def circle_C (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), line_symmetry x y → 
      (center.1 + point_P.1) / 2 = x ∧ (center.2 + point_P.2) / 2 = y) →
    (∃ (A B : ℝ × ℝ), line_intersect A.1 A.2 ∧ line_intersect B.1 B.2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = length_AB^2) →
    (∀ (x y : ℝ), circle_C center radius x y ↔ x^2 + (y+1)^2 = 18) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1473_147395


namespace NUMINAMATH_CALUDE_mul_exp_analogy_l1473_147339

-- Define multiplication recursively
def mul_rec (k : ℕ) : ℕ → ℕ
| 0     => 0                   -- Base case
| n + 1 => k + mul_rec k n     -- Recursive step

-- Define exponentiation recursively
def exp_rec (k : ℕ) : ℕ → ℕ
| 0     => 1                   -- Base case
| n + 1 => k * exp_rec k n     -- Recursive step

-- Theorem stating the analogy between multiplication and exponentiation
theorem mul_exp_analogy :
  (∀ k n : ℕ, mul_rec k (n + 1) = k + mul_rec k n) ↔
  (∀ k n : ℕ, exp_rec k (n + 1) = k * exp_rec k n) :=
sorry

end NUMINAMATH_CALUDE_mul_exp_analogy_l1473_147339


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1473_147368

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2*x - (3*y - 5*x) + 7*y = 7*x + 4*y :=
by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  (-x^2 + 4*x) - 3*x^2 + 2*(2*x^2 - 3*x) = -2*x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1473_147368


namespace NUMINAMATH_CALUDE_six_people_arrangement_l1473_147321

theorem six_people_arrangement (n : ℕ) (h : n = 6) : 
  n.factorial - 2 * (n-1).factorial - 2 * 2 * (n-1).factorial + 2 * 2 * (n-2).factorial = 96 :=
by sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l1473_147321


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l1473_147375

theorem sqrt_x_minus_8_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l1473_147375


namespace NUMINAMATH_CALUDE_bicycle_fog_problem_l1473_147365

/-- Bicycle and fog bank problem -/
theorem bicycle_fog_problem (v_bicycle : ℝ) (v_fog : ℝ) (r_fog : ℝ) (initial_distance : ℝ) :
  v_bicycle = 1/2 →
  v_fog = 1/3 * Real.sqrt 2 →
  r_fog = 40 →
  initial_distance = 100 →
  ∃ t₁ t₂ : ℝ,
    t₁ < t₂ ∧
    (∀ t, t₁ ≤ t ∧ t ≤ t₂ →
      (initial_distance - v_fog * t)^2 + (v_bicycle * t - v_fog * t)^2 ≤ r_fog^2) ∧
    (t₁ + t₂) / 2 = 240 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_fog_problem_l1473_147365


namespace NUMINAMATH_CALUDE_nested_root_simplification_l1473_147352

theorem nested_root_simplification (b : ℝ) :
  (((b^16)^(1/8))^(1/4))^6 * (((b^16)^(1/4))^(1/8))^6 = b^6 := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l1473_147352


namespace NUMINAMATH_CALUDE_person_age_l1473_147391

/-- The age of a person satisfying a specific equation is 32 years old. -/
theorem person_age : ∃ (age : ℕ), 4 * (age + 4) - 4 * (age - 4) = age ∧ age = 32 := by
  sorry

end NUMINAMATH_CALUDE_person_age_l1473_147391


namespace NUMINAMATH_CALUDE_common_terms_count_l1473_147381

theorem common_terms_count : 
  (Finset.filter (fun k => 15 * k + 8 ≤ 2018) (Finset.range (2019 / 15 + 1))).card = 135 :=
by sorry

end NUMINAMATH_CALUDE_common_terms_count_l1473_147381


namespace NUMINAMATH_CALUDE_earphone_cost_l1473_147325

def mean_expenditure : ℝ := 500
def mon_expenditure : ℝ := 450
def tue_expenditure : ℝ := 600
def wed_expenditure : ℝ := 400
def thu_expenditure : ℝ := 500
def sat_expenditure : ℝ := 550
def sun_expenditure : ℝ := 300
def pen_cost : ℝ := 30
def notebook_cost : ℝ := 50
def num_days : ℕ := 7

theorem earphone_cost :
  let total_expenditure := mean_expenditure * num_days
  let known_expenditures := mon_expenditure + tue_expenditure + wed_expenditure + 
                            thu_expenditure + sat_expenditure + sun_expenditure
  let friday_expenditure := total_expenditure - known_expenditures
  let other_items_cost := pen_cost + notebook_cost
  friday_expenditure - other_items_cost = 620 := by
sorry

end NUMINAMATH_CALUDE_earphone_cost_l1473_147325


namespace NUMINAMATH_CALUDE_oil_tank_capacity_oil_tank_capacity_proof_l1473_147317

theorem oil_tank_capacity : ℝ → Prop :=
  fun t => 
    (∃ o : ℝ, o / t = 1 / 6 ∧ (o + 4) / t = 1 / 3) → t = 24

-- The proof is omitted
theorem oil_tank_capacity_proof : oil_tank_capacity 24 :=
  sorry

end NUMINAMATH_CALUDE_oil_tank_capacity_oil_tank_capacity_proof_l1473_147317


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l1473_147388

/-- Calculates the number of books Robert can read given his reading speed, book size, and available time. -/
def books_read (reading_speed : ℕ) (pages_per_book : ℕ) (available_time : ℕ) : ℕ :=
  (reading_speed * available_time) / pages_per_book

theorem robert_reading_capacity : books_read 200 400 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l1473_147388


namespace NUMINAMATH_CALUDE_males_in_choir_is_twelve_l1473_147331

/-- Represents the number of musicians in each group -/
structure MusicianCounts where
  orchestra_males : ℕ
  orchestra_females : ℕ
  choir_females : ℕ
  total_musicians : ℕ

/-- Calculates the number of males in the choir based on given conditions -/
def males_in_choir (counts : MusicianCounts) : ℕ :=
  let orchestra_total := counts.orchestra_males + counts.orchestra_females
  let band_total := 2 * orchestra_total
  let choir_total := counts.total_musicians - (orchestra_total + band_total)
  choir_total - counts.choir_females

/-- Theorem stating that the number of males in the choir is 12 -/
theorem males_in_choir_is_twelve (counts : MusicianCounts)
  (h1 : counts.orchestra_males = 11)
  (h2 : counts.orchestra_females = 12)
  (h3 : counts.choir_females = 17)
  (h4 : counts.total_musicians = 98) :
  males_in_choir counts = 12 := by
  sorry

#eval males_in_choir ⟨11, 12, 17, 98⟩

end NUMINAMATH_CALUDE_males_in_choir_is_twelve_l1473_147331


namespace NUMINAMATH_CALUDE_circle_radius_in_rectangle_l1473_147350

theorem circle_radius_in_rectangle (r : ℝ) : 
  r > 0 → 
  (π * r^2 = 72 / 2) → 
  r = 6 / Real.sqrt π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_in_rectangle_l1473_147350


namespace NUMINAMATH_CALUDE_sum_has_five_digits_l1473_147364

theorem sum_has_five_digits (A B : ℕ) (hA : A ≠ 0 ∧ A < 10) (hB : B ≠ 0 ∧ B < 10) :
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n = 9876 + (100 * A + 32) + (10 * B + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_has_five_digits_l1473_147364


namespace NUMINAMATH_CALUDE_average_of_numbers_l1473_147355

def numbers : List ℤ := [-5, -2, 0, 4, 8]

theorem average_of_numbers : (numbers.sum : ℚ) / numbers.length = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1473_147355


namespace NUMINAMATH_CALUDE_average_score_is_two_l1473_147372

/-- Represents the score distribution of a test --/
structure ScoreDistribution where
  three_points : ℝ
  two_points : ℝ
  one_point : ℝ
  zero_points : ℝ
  sum_to_one : three_points + two_points + one_point + zero_points = 1

/-- Calculates the average score given a score distribution --/
def average_score (sd : ScoreDistribution) : ℝ :=
  3 * sd.three_points + 2 * sd.two_points + 1 * sd.one_point + 0 * sd.zero_points

/-- The main theorem stating that the average score is 2 points --/
theorem average_score_is_two (sd : ScoreDistribution)
  (h1 : sd.three_points = 0.3)
  (h2 : sd.two_points = 0.5)
  (h3 : sd.one_point = 0.1)
  (h4 : sd.zero_points = 0.1) :
  average_score sd = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_two_l1473_147372


namespace NUMINAMATH_CALUDE_sine_cosine_equation_l1473_147393

theorem sine_cosine_equation (x y : ℝ) 
  (h : (Real.sin x ^ 2 - Real.cos x ^ 2 + Real.cos x ^ 2 * Real.cos y ^ 2 - Real.sin x ^ 2 * Real.sin y ^ 2) / Real.sin (x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * k * Real.pi + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_equation_l1473_147393


namespace NUMINAMATH_CALUDE_sequence_limit_uniqueness_l1473_147332

theorem sequence_limit_uniqueness (a : ℕ → ℝ) (l₁ l₂ : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l₁| < ε) →
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l₂| < ε) →
  l₁ = l₂ :=
by sorry

end NUMINAMATH_CALUDE_sequence_limit_uniqueness_l1473_147332


namespace NUMINAMATH_CALUDE_ten_points_guarantees_win_ten_is_smallest_winning_score_l1473_147302

/-- Represents the possible positions a team can finish in a match -/
inductive Position
| first
| second
| third
| fourth

/-- Returns the points awarded for a given position -/
def points_for_position (p : Position) : ℕ :=
  match p with
  | Position.first => 4
  | Position.second => 3
  | Position.third => 2
  | Position.fourth => 1

/-- Represents the results of a team in three matches -/
structure TeamResults :=
  (match1 : Position)
  (match2 : Position)
  (match3 : Position)

/-- Calculates the total points for a team's results -/
def total_points (results : TeamResults) : ℕ :=
  points_for_position results.match1 +
  points_for_position results.match2 +
  points_for_position results.match3

/-- Theorem: 10 points guarantees more points than any other team -/
theorem ten_points_guarantees_win :
  ∀ (results : TeamResults),
    total_points results ≥ 10 →
    ∀ (other_results : TeamResults),
      other_results ≠ results →
      total_points results > total_points other_results :=
by sorry

/-- Theorem: 10 is the smallest number of points that guarantees a win -/
theorem ten_is_smallest_winning_score :
  ∀ n : ℕ,
    n < 10 →
    ∃ (results other_results : TeamResults),
      total_points results = n ∧
      other_results ≠ results ∧
      total_points other_results ≥ total_points results :=
by sorry

end NUMINAMATH_CALUDE_ten_points_guarantees_win_ten_is_smallest_winning_score_l1473_147302
