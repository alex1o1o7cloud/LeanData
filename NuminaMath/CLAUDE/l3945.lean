import Mathlib

namespace NUMINAMATH_CALUDE_power_two_1000_mod_13_l3945_394586

theorem power_two_1000_mod_13 : 2^1000 % 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_two_1000_mod_13_l3945_394586


namespace NUMINAMATH_CALUDE_third_shot_scores_l3945_394583

/-- Represents a shooter's scores across 5 shots -/
structure ShooterScores where
  scores : Fin 5 → ℕ

/-- The problem setup -/
def ShootingProblem (shooter1 shooter2 : ShooterScores) : Prop :=
  -- The first three shots resulted in the same number of points
  (shooter1.scores 0 + shooter1.scores 1 + shooter1.scores 2 =
   shooter2.scores 0 + shooter2.scores 1 + shooter2.scores 2) ∧
  -- In the last three shots, the first shooter scored three times as many points as the second shooter
  (shooter1.scores 2 + shooter1.scores 3 + shooter1.scores 4 =
   3 * (shooter2.scores 2 + shooter2.scores 3 + shooter2.scores 4))

/-- The theorem to prove -/
theorem third_shot_scores (shooter1 shooter2 : ShooterScores)
    (h : ShootingProblem shooter1 shooter2) :
    shooter1.scores 2 = 10 ∧ shooter2.scores 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_third_shot_scores_l3945_394583


namespace NUMINAMATH_CALUDE_strongest_signal_l3945_394552

def signal_strength (x : ℤ) : ℝ := |x|

def is_stronger (x y : ℤ) : Prop := signal_strength x < signal_strength y

theorem strongest_signal :
  let signals : List ℤ := [-50, -60, -70, -80]
  ∀ s ∈ signals, s ≠ -50 → is_stronger (-50) s :=
sorry

end NUMINAMATH_CALUDE_strongest_signal_l3945_394552


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3945_394542

theorem polynomial_simplification (x : ℝ) :
  (5 * x^12 - 3 * x^9 + 6 * x^8 - 2 * x^7) + 
  (7 * x^12 + 2 * x^11 - x^9 + 4 * x^7 + 2 * x^5 - x + 3) = 
  12 * x^12 + 2 * x^11 - 4 * x^9 + 6 * x^8 + 2 * x^7 + 2 * x^5 - x + 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3945_394542


namespace NUMINAMATH_CALUDE_cake_triangles_l3945_394551

/-- The number of triangular pieces that can be cut from a rectangular cake -/
theorem cake_triangles (cake_length cake_width triangle_base triangle_height : ℝ) 
  (h1 : cake_length = 24)
  (h2 : cake_width = 20)
  (h3 : triangle_base = 2)
  (h4 : triangle_height = 2) :
  (cake_length * cake_width) / (1/2 * triangle_base * triangle_height) = 240 :=
by sorry

end NUMINAMATH_CALUDE_cake_triangles_l3945_394551


namespace NUMINAMATH_CALUDE_sara_coin_collection_value_l3945_394546

/-- Calculates the total value in cents of a coin collection --/
def total_cents (quarters dimes nickels pennies : ℕ) : ℕ :=
  quarters * 25 + dimes * 10 + nickels * 5 + pennies

/-- Proves that Sara's coin collection totals 453 cents --/
theorem sara_coin_collection_value :
  total_cents 11 8 15 23 = 453 := by
  sorry

end NUMINAMATH_CALUDE_sara_coin_collection_value_l3945_394546


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l3945_394538

theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  -- ABCDEF is a convex hexagon (sum of angles is 720°)
  A + B + C + D + E + F = 720 →
  -- Angles A, B, C, and D are congruent
  A = B ∧ B = C ∧ C = D →
  -- Angles E and F are congruent
  E = F →
  -- Measure of angle A is 30° less than measure of angle E
  A + 30 = E →
  -- Conclusion: Measure of angle E is 140°
  E = 140 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l3945_394538


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3945_394505

theorem quadratic_inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3945_394505


namespace NUMINAMATH_CALUDE_inequality_holds_l3945_394592

theorem inequality_holds (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3945_394592


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3945_394522

theorem complex_magnitude_problem : 
  let z : ℂ := (2 - Complex.I)^2 / Complex.I
  Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3945_394522


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3945_394557

theorem arithmetic_equality : 54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3945_394557


namespace NUMINAMATH_CALUDE_power_two_99_mod_7_l3945_394580

theorem power_two_99_mod_7 : 2^99 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_99_mod_7_l3945_394580


namespace NUMINAMATH_CALUDE_find_a_and_b_l3945_394509

-- Define the curve equation
def curve (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 36

-- Define the theorem
theorem find_a_and_b :
  ∀ a b : ℝ, curve 0 (-12) a b → curve 0 0 a b → a = 0 ∧ b = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l3945_394509


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l3945_394533

/-- The number of roses available for selection. -/
def num_roses : ℕ := 4

/-- The number of tulips available for selection. -/
def num_tulips : ℕ := 3

/-- The number of flower arrangements where exactly one of the roses or tulips is the same. -/
def arrangements_with_one_same : ℕ := 
  (num_roses * (num_tulips * (num_tulips - 1))) + 
  (num_tulips * (num_roses * (num_roses - 1)))

/-- Theorem stating that the number of flower arrangements where exactly one of the roses or tulips is the same is 60. -/
theorem flower_arrangement_count : arrangements_with_one_same = 60 := by
  sorry

end NUMINAMATH_CALUDE_flower_arrangement_count_l3945_394533


namespace NUMINAMATH_CALUDE_charlie_golden_delicious_l3945_394531

/-- The number of bags of Golden Delicious apples Charlie picked -/
def golden_delicious : ℝ :=
  0.67 - (0.17 + 0.33)

theorem charlie_golden_delicious :
  golden_delicious = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_charlie_golden_delicious_l3945_394531


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3945_394518

theorem simplify_trig_expression :
  Real.sqrt (2 + Real.cos (20 * π / 180) - Real.sin (10 * π / 180)^2) = Real.sqrt 3 * Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3945_394518


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l3945_394535

/-- The probability of selecting at least one woman when choosing 4 people at random from a group of 10 men and 5 women is 29/36. -/
theorem prob_at_least_one_woman (total : ℕ) (men : ℕ) (women : ℕ) (selection : ℕ) : 
  total = 15 → men = 10 → women = 5 → selection = 4 → 
  (1 - (men.choose selection / total.choose selection : ℚ)) = 29/36 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l3945_394535


namespace NUMINAMATH_CALUDE_max_n_for_factorizable_quadratic_l3945_394584

/-- Given a quadratic expression 5x^2 + nx + 60 that can be factored as the product
    of two linear factors with integer coefficients, the maximum possible value of n is 301. -/
theorem max_n_for_factorizable_quadratic : 
  ∀ n : ℤ, 
  (∃ a b : ℤ, ∀ x : ℤ, 5 * x^2 + n * x + 60 = (5 * x + a) * (x + b)) →
  n ≤ 301 :=
by sorry

end NUMINAMATH_CALUDE_max_n_for_factorizable_quadratic_l3945_394584


namespace NUMINAMATH_CALUDE_right_triangle_exists_l3945_394519

theorem right_triangle_exists (a : ℤ) (h : a ≥ 5) :
  ∃ b c : ℤ, c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_exists_l3945_394519


namespace NUMINAMATH_CALUDE_fraction_simplification_l3945_394596

theorem fraction_simplification :
  (-45 : ℚ) / 25 / (15 : ℚ) / 40 = -24 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3945_394596


namespace NUMINAMATH_CALUDE_ball_box_problem_l3945_394560

/-- Given an opaque box with balls of three colors: red, yellow, and blue. -/
structure BallBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ
  total_eq : total = red + yellow + blue
  yellow_eq : yellow = 2 * blue

/-- The probability of drawing a blue ball from the box -/
def blue_probability (box : BallBox) : ℚ :=
  box.blue / box.total

/-- The number of additional blue balls needed to make the probability 1/2 -/
def additional_blue_balls (box : BallBox) : ℕ :=
  let new_total := box.total + 14
  let new_blue := box.blue + 14
  14

/-- Theorem stating the properties of the specific box in the problem -/
theorem ball_box_problem :
  ∃ (box : BallBox),
    box.total = 30 ∧
    box.red = 6 ∧
    blue_probability box = 4 / 15 ∧
    additional_blue_balls box = 14 ∧
    blue_probability ⟨box.total + 14, box.red, box.blue + 14, box.yellow,
      by sorry, by sorry⟩ = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ball_box_problem_l3945_394560


namespace NUMINAMATH_CALUDE_painter_problem_l3945_394566

theorem painter_problem (total_rooms : ℕ) (time_per_room : ℕ) (time_left : ℕ) 
  (h1 : total_rooms = 11)
  (h2 : time_per_room = 7)
  (h3 : time_left = 63) :
  total_rooms - (time_left / time_per_room) = 2 := by
  sorry

end NUMINAMATH_CALUDE_painter_problem_l3945_394566


namespace NUMINAMATH_CALUDE_function_constant_l3945_394526

/-- A function satisfying the given functional equation is constant -/
theorem function_constant (f : ℝ → ℝ) 
    (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (Real.sqrt (x * y)) = f ((x + y) / 2)) :
  ∀ (a b : ℝ), a > 0 → b > 0 → f a = f b := by sorry

end NUMINAMATH_CALUDE_function_constant_l3945_394526


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3945_394575

theorem quadratic_factorization (a b : ℤ) :
  (∀ x : ℝ, 20 * x^2 - 90 * x - 22 = (5 * x + a) * (4 * x + b)) →
  a + 3 * b = -65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3945_394575


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3945_394545

theorem regular_polygon_sides (n : ℕ) (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 140) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3945_394545


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3945_394506

theorem x_plus_y_value (x y : ℤ) (hx : -x = 3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3945_394506


namespace NUMINAMATH_CALUDE_city_population_problem_l3945_394558

/-- Given three cities with the following conditions:
    - Richmond has 1000 more people than Victoria
    - Victoria has 4 times as many people as another city
    - Richmond has 3000 people
    Prove that the other city has 500 people. -/
theorem city_population_problem (richmond victoria other : ℕ) : 
  richmond = victoria + 1000 →
  victoria = 4 * other →
  richmond = 3000 →
  other = 500 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l3945_394558


namespace NUMINAMATH_CALUDE_quadratic_trinomial_existence_l3945_394532

theorem quadratic_trinomial_existence : ∃ f : ℝ → ℝ, 
  (∀ x, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧ 
  f 2014 = 2015 ∧ 
  f 2015 = 0 ∧ 
  f 2016 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_existence_l3945_394532


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_six_l3945_394504

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

theorem largest_five_digit_divisible_by_six :
  ∀ n : ℕ, is_five_digit n → divisible_by_six n → n ≤ 99996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_six_l3945_394504


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3945_394594

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l3945_394594


namespace NUMINAMATH_CALUDE_sin_2x_value_l3945_394569

theorem sin_2x_value (x : Real) (h : (1 + Real.tan x) / (1 - Real.tan x) = 2) : 
  Real.sin (2 * x) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l3945_394569


namespace NUMINAMATH_CALUDE_birthday_cake_icing_l3945_394547

/-- Represents a 3D cube --/
structure Cube :=
  (size : ℕ)

/-- Represents the icing configuration on the cube --/
structure IcingConfig :=
  (top : Bool)
  (bottom : Bool)
  (side1 : Bool)
  (side2 : Bool)
  (side3 : Bool)
  (side4 : Bool)

/-- Counts the number of unit cubes with exactly two iced sides --/
def countTwoSidedIcedCubes (c : Cube) (ic : IcingConfig) : ℕ :=
  sorry

/-- The main theorem --/
theorem birthday_cake_icing (c : Cube) (ic : IcingConfig) :
  c.size = 5 →
  ic.top = true →
  ic.bottom = true →
  ic.side1 = true →
  ic.side2 = true →
  ic.side3 = false →
  ic.side4 = false →
  countTwoSidedIcedCubes c ic = 20 :=
sorry

end NUMINAMATH_CALUDE_birthday_cake_icing_l3945_394547


namespace NUMINAMATH_CALUDE_probability_six_even_numbers_l3945_394582

def integers_range : Set ℤ := {x | -9 ≤ x ∧ x ≤ 9}

def even_numbers (S : Set ℤ) : Set ℤ := {x ∈ S | x % 2 = 0}

def total_count : ℕ := Finset.card (Finset.range 19)

def even_count (S : Set ℤ) : ℕ := Finset.card (Finset.filter (λ x => x % 2 = 0) (Finset.range 19))

theorem probability_six_even_numbers :
  let S := integers_range
  let n := total_count
  let k := even_count S
  (k.choose 6 : ℚ) / (n.choose 6 : ℚ) = 1 / 76 := by sorry

end NUMINAMATH_CALUDE_probability_six_even_numbers_l3945_394582


namespace NUMINAMATH_CALUDE_rational_sum_product_quotient_zero_l3945_394537

theorem rational_sum_product_quotient_zero (a b : ℚ) :
  (a + b) / (a * b) = 0 → a ≠ 0 ∧ b ≠ 0 ∧ a = -b := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_product_quotient_zero_l3945_394537


namespace NUMINAMATH_CALUDE_max_k_value_l3945_394530

theorem max_k_value (A B C : ℕ) (k : ℕ+) : 
  (A ≠ 0) →
  (A < 10) → 
  (B < 10) → 
  (C < 10) → 
  (k * (10 * A + B) = 100 * A + 10 * C + B) → 
  k ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3945_394530


namespace NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l3945_394521

theorem eight_digit_increasing_remainder (n : ℕ) (h : n = 8) :
  (Nat.choose (n + 9 - 1) n) % 1000 = 870 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_remainder_l3945_394521


namespace NUMINAMATH_CALUDE_equation_represents_point_l3945_394572

theorem equation_represents_point :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 6*y + 7 = 0 ↔ x = 2 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_represents_point_l3945_394572


namespace NUMINAMATH_CALUDE_math_paths_count_l3945_394559

/-- Represents the number of adjacent positions a letter can move to -/
def adjacent_positions : ℕ := 8

/-- Represents the length of the word "MATH" -/
def word_length : ℕ := 4

/-- Calculates the number of paths to spell "MATH" -/
def num_paths : ℕ := adjacent_positions ^ (word_length - 1)

/-- Theorem stating that the number of paths to spell "MATH" is 512 -/
theorem math_paths_count : num_paths = 512 := by sorry

end NUMINAMATH_CALUDE_math_paths_count_l3945_394559


namespace NUMINAMATH_CALUDE_three_students_left_l3945_394513

/-- Calculates the number of students who left during the year. -/
def students_left (initial : ℕ) (new : ℕ) (final : ℕ) : ℕ :=
  initial + new - final

/-- Proves that 3 students left during the year given the initial, new, and final student counts. -/
theorem three_students_left : students_left 4 42 43 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_students_left_l3945_394513


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l3945_394502

/-- Represents a point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space --/
structure Line2D where
  k : ℝ
  b : ℝ

/-- Checks if a point is in the second quadrant --/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is on the given line --/
def isOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.k * p.x + l.b

/-- Theorem: A line with positive slope and negative y-intercept does not pass through the second quadrant --/
theorem line_not_in_second_quadrant (l : Line2D) 
  (h1 : l.k > 0) (h2 : l.b < 0) : 
  ¬ ∃ p : Point2D, isInSecondQuadrant p ∧ isOnLine p l :=
by
  sorry


end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l3945_394502


namespace NUMINAMATH_CALUDE_right_triangle_tangent_l3945_394567

theorem right_triangle_tangent (n : ℕ) (a h : ℝ) (α : ℝ) :
  Odd n →
  0 < n →
  0 < a →
  0 < h →
  0 < α →
  α < π / 2 →
  Real.tan α = (4 * n * h) / ((n^2 - 1) * a) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_l3945_394567


namespace NUMINAMATH_CALUDE_fine_arts_packaging_volume_l3945_394585

/-- The volume needed to package a fine arts collection given box dimensions, cost per box, and minimum total cost. -/
theorem fine_arts_packaging_volume 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (cost_per_box : ℝ) 
  (min_total_cost : ℝ)
  (h1 : box_length = 20)
  (h2 : box_width = 20)
  (h3 : box_height = 12)
  (h4 : cost_per_box = 0.5)
  (h5 : min_total_cost = 225) :
  (min_total_cost / cost_per_box) * (box_length * box_width * box_height) = 2160000 := by
  sorry

#check fine_arts_packaging_volume

end NUMINAMATH_CALUDE_fine_arts_packaging_volume_l3945_394585


namespace NUMINAMATH_CALUDE_factorization_equality_l3945_394524

theorem factorization_equality (c : ℝ) : 196 * c^3 + 28 * c^2 = 28 * c^2 * (7 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3945_394524


namespace NUMINAMATH_CALUDE_abc_inequality_l3945_394578

theorem abc_inequality : 
  let a : ℝ := Real.rpow 7 (1/3)
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3945_394578


namespace NUMINAMATH_CALUDE_odd_spaced_stone_selections_count_l3945_394528

/-- The number of ways to select 5 stones from 15 stones in a line, 
    such that there are an odd number of stones between any two selected stones. -/
def oddSpacedStoneSelections : ℕ := 77

/-- The total number of stones in the line. -/
def totalStones : ℕ := 15

/-- The number of stones to be selected. -/
def stonesToSelect : ℕ := 5

/-- The number of odd-numbered stones in the line. -/
def oddNumberedStones : ℕ := 8

/-- The number of even-numbered stones in the line. -/
def evenNumberedStones : ℕ := 7

theorem odd_spaced_stone_selections_count :
  oddSpacedStoneSelections = Nat.choose oddNumberedStones stonesToSelect + Nat.choose evenNumberedStones stonesToSelect :=
by sorry

end NUMINAMATH_CALUDE_odd_spaced_stone_selections_count_l3945_394528


namespace NUMINAMATH_CALUDE_line_point_k_value_l3945_394553

/-- A line contains the points (7, 10), (-3, k), and (-11, 5). The value of k is 65/9. -/
theorem line_point_k_value :
  ∀ (k : ℚ),
  (∃ (m b : ℚ),
    (7 : ℚ) * m + b = 10 ∧
    (-3 : ℚ) * m + b = k ∧
    (-11 : ℚ) * m + b = 5) →
  k = 65 / 9 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3945_394553


namespace NUMINAMATH_CALUDE_bill_profit_l3945_394548

-- Define the given conditions
def total_milk : ℚ := 16
def butter_ratio : ℚ := 1/4
def sour_cream_ratio : ℚ := 1/4
def milk_to_butter : ℚ := 4
def milk_to_sour_cream : ℚ := 2
def butter_price : ℚ := 5
def sour_cream_price : ℚ := 6
def whole_milk_price : ℚ := 3

-- Define the theorem
theorem bill_profit : 
  let milk_for_butter := total_milk * butter_ratio
  let milk_for_sour_cream := total_milk * sour_cream_ratio
  let butter_gallons := milk_for_butter / milk_to_butter
  let sour_cream_gallons := milk_for_sour_cream / milk_to_sour_cream
  let whole_milk_gallons := total_milk - milk_for_butter - milk_for_sour_cream
  let butter_profit := butter_gallons * butter_price
  let sour_cream_profit := sour_cream_gallons * sour_cream_price
  let whole_milk_profit := whole_milk_gallons * whole_milk_price
  butter_profit + sour_cream_profit + whole_milk_profit = 41 := by
sorry

end NUMINAMATH_CALUDE_bill_profit_l3945_394548


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3945_394501

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- First line equation: 3y - 3b = 9x -/
def line1 (x y b : ℝ) : Prop := 3 * y - 3 * b = 9 * x

/-- Second line equation: y - 2 = (b + 9)x -/
def line2 (x y b : ℝ) : Prop := y - 2 = (b + 9) * x

theorem perpendicular_lines_b_value :
  ∀ b : ℝ, (∃ x y : ℝ, line1 x y b ∧ line2 x y b ∧
    perpendicular 3 (b + 9)) → b = -28/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l3945_394501


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l3945_394516

theorem positive_reals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) : 
  (a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c) ∧ 
  (a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l3945_394516


namespace NUMINAMATH_CALUDE_inequality_solution_and_a_range_l3945_394564

def f (x : ℝ) := |3*x + 2|

theorem inequality_solution_and_a_range :
  (∃ S : Set ℝ, S = {x : ℝ | -5/4 < x ∧ x < 1/2} ∧
    ∀ x, x ∈ S ↔ f x < 4 - |x - 1|) ∧
  ∀ m n : ℝ, m > 0 → n > 0 → m + n = 1 →
    (∀ a : ℝ, a > 0 →
      (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) →
        0 < a ∧ a ≤ 10/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_a_range_l3945_394564


namespace NUMINAMATH_CALUDE_marble_problem_l3945_394563

theorem marble_problem (x : ℚ) 
  (angela : ℚ) (brian : ℚ) (caden : ℚ) (daryl : ℚ)
  (h1 : angela = x)
  (h2 : brian = 3 * x)
  (h3 : caden = 2 * brian)
  (h4 : daryl = 4 * caden)
  (h5 : angela + brian + caden + daryl = 144) :
  x = 72 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l3945_394563


namespace NUMINAMATH_CALUDE_frog_jump_probability_l3945_394562

/-- Represents a position on the grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the grid -/
def Grid := {p : Position // p.x ≤ 5 ∧ p.y ≤ 5}

/-- The blocked cell -/
def blockedCell : Position := ⟨3, 3⟩

/-- Check if a position is on the grid boundary -/
def isOnBoundary (p : Position) : Bool :=
  p.x = 0 ∨ p.x = 5 ∨ p.y = 0 ∨ p.y = 5

/-- Check if a position is on a vertical side of the grid -/
def isOnVerticalSide (p : Position) : Bool :=
  p.x = 0 ∨ p.x = 5

/-- Probability of ending on a vertical side starting from a given position -/
noncomputable def probabilityVerticalSide (p : Position) : Real :=
  sorry

/-- Theorem: The probability of ending on a vertical side starting from (2,2) is 5/8 -/
theorem frog_jump_probability :
  probabilityVerticalSide ⟨2, 2⟩ = 5/8 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l3945_394562


namespace NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l3945_394570

/-- The number of shaded triangles in the nth diagram -/
def shaded_triangles (n : ℕ) : ℕ := n^2

/-- The total number of triangles in the nth diagram -/
def total_triangles (n : ℕ) : ℕ := n^2

/-- The fraction of shaded triangles in the nth diagram -/
def shaded_fraction (n : ℕ) : ℚ := shaded_triangles n / total_triangles n

theorem eighth_diagram_fully_shaded :
  shaded_fraction 8 = 1 := by sorry

end NUMINAMATH_CALUDE_eighth_diagram_fully_shaded_l3945_394570


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3945_394577

theorem quadratic_one_solution (k : ℝ) :
  (∃! x, 4 * x^2 + k * x + 4 = 0) ↔ (k = 8 ∨ k = -8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3945_394577


namespace NUMINAMATH_CALUDE_max_points_for_successful_teams_l3945_394598

/-- Represents the number of teams in the tournament -/
def num_teams : ℕ := 15

/-- Represents the number of teams that scored at least N points -/
def num_successful_teams : ℕ := 6

/-- Represents the points awarded for a win -/
def win_points : ℕ := 3

/-- Represents the points awarded for a draw -/
def draw_points : ℕ := 1

/-- Represents the points awarded for a loss -/
def loss_points : ℕ := 0

/-- Theorem stating the maximum value of N -/
theorem max_points_for_successful_teams :
  ∃ (N : ℕ), 
    (∀ (n : ℕ), n > N → 
      ¬∃ (team_scores : Fin num_teams → ℕ),
        (∀ i j, i ≠ j → team_scores i + team_scores j ≤ win_points) ∧
        (∃ (successful : Fin num_teams → Prop),
          (∃ (k : Fin num_successful_teams), ∀ i, successful i ↔ team_scores i ≥ n))) ∧
    (∃ (team_scores : Fin num_teams → ℕ),
      (∀ i j, i ≠ j → team_scores i + team_scores j ≤ win_points) ∧
      (∃ (successful : Fin num_teams → Prop),
        (∃ (k : Fin num_successful_teams), ∀ i, successful i ↔ team_scores i ≥ N))) ∧
    N = 34 := by
  sorry

end NUMINAMATH_CALUDE_max_points_for_successful_teams_l3945_394598


namespace NUMINAMATH_CALUDE_fraction_value_in_system_l3945_394573

theorem fraction_value_in_system (a b x y : ℝ) (hb : b ≠ 0) 
  (eq1 : 4 * x - 2 * y = a) (eq2 : 5 * y - 10 * x = b) : a / b = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_in_system_l3945_394573


namespace NUMINAMATH_CALUDE_subset_sums_determine_set_l3945_394523

def three_element_subset_sums (A : Finset ℤ) : Finset ℤ :=
  (A.powerset.filter (λ s => s.card = 3)).image (λ s => s.sum id)

theorem subset_sums_determine_set :
  ∀ A : Finset ℤ,
    A.card = 4 →
    three_element_subset_sums A = {-1, 3, 5, 8} →
    A = {-3, 0, 2, 6} := by
  sorry

end NUMINAMATH_CALUDE_subset_sums_determine_set_l3945_394523


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3945_394565

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ x y : ℝ, hyperbola x y → 
  ∃ a b : ℝ, ellipse a b ∧ 
  (∀ c d : ℝ, hyperbola c 0 → (a = c ∨ a = -c) ∧ (b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3945_394565


namespace NUMINAMATH_CALUDE_morning_ride_l3945_394599

theorem morning_ride (x : ℝ) (h : x + 5*x = 12) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_morning_ride_l3945_394599


namespace NUMINAMATH_CALUDE_sandras_mother_contribution_sandras_mother_contribution_proof_l3945_394549

theorem sandras_mother_contribution : ℝ → Prop :=
  fun m =>
    let savings : ℝ := 10
    let father_contribution : ℝ := 2 * m
    let total_money : ℝ := savings + m + father_contribution
    let candy_cost : ℝ := 0.5
    let jelly_bean_cost : ℝ := 0.2
    let candy_quantity : ℕ := 14
    let jelly_bean_quantity : ℕ := 20
    let total_cost : ℝ := candy_cost * candy_quantity + jelly_bean_cost * jelly_bean_quantity
    let money_left : ℝ := 11
    total_money = total_cost + money_left → m = 4

theorem sandras_mother_contribution_proof : ∃ m, sandras_mother_contribution m :=
  sorry

end NUMINAMATH_CALUDE_sandras_mother_contribution_sandras_mother_contribution_proof_l3945_394549


namespace NUMINAMATH_CALUDE_train_crossing_time_l3945_394587

/-- The time taken for a train to cross a telegraph post -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 320 ∧ train_speed_kmh = 72 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 16 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3945_394587


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3945_394555

theorem polygon_sides_from_angle_sum (sum_of_angles : ℝ) :
  sum_of_angles = 1080 → ∃ n : ℕ, n = 8 ∧ sum_of_angles = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3945_394555


namespace NUMINAMATH_CALUDE_lcm_36_105_l3945_394536

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_105_l3945_394536


namespace NUMINAMATH_CALUDE_division_problem_l3945_394512

theorem division_problem : ∃ x : ℝ, (3.242 * 15) / x = 0.04863 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3945_394512


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3945_394550

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_sequence :
  arithmetic_sequence 2 3 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l3945_394550


namespace NUMINAMATH_CALUDE_valid_pictures_invalid_pictures_l3945_394568

-- Define a 4x4 grid
def Grid := Fin 4 → Fin 4 → Option ℕ

-- Define adjacency in the grid
def adjacent (x₁ y₁ x₂ y₂ : Fin 4) : Prop :=
  (x₁ = x₂ ∧ y₁.val + 1 = y₂.val) ∨
  (x₁ = x₂ ∧ y₂.val + 1 = y₁.val) ∨
  (y₁ = y₂ ∧ x₁.val + 1 = x₂.val) ∨
  (y₁ = y₂ ∧ x₂.val + 1 = x₁.val)

-- Define a valid grid configuration
def valid_grid (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 15 →
    ∃ x₁ y₁ x₂ y₂ : Fin 4,
      g x₁ y₁ = some n ∧
      g x₂ y₂ = some (n + 1) ∧
      adjacent x₁ y₁ x₂ y₂

-- Define the specific configurations for Pictures 3 and 5
def picture3 : Grid := fun x y =>
  match x, y with
  | 0, 0 => some 1 | 0, 1 => some 2 | 0, 2 => some 7 | 0, 3 => some 8
  | 1, 0 => some 14 | 1, 1 => some 3 | 1, 2 => some 6 | 1, 3 => some 9
  | 2, 0 => some 15 | 2, 1 => some 4 | 2, 2 => some 5 | 2, 3 => some 10
  | 3, 0 => some 16 | 3, 1 => none | 3, 2 => none | 3, 3 => some 11
  
def picture5 : Grid := fun x y =>
  match x, y with
  | 0, 0 => none | 0, 1 => some 4 | 0, 2 => some 5 | 0, 3 => some 6
  | 1, 0 => none | 1, 1 => some 3 | 1, 2 => none | 1, 3 => some 7
  | 2, 0 => some 14 | 2, 1 => some 2 | 2, 2 => some 9 | 2, 3 => some 8
  | 3, 0 => some 15 | 3, 1 => some 1 | 3, 2 => some 10 | 3, 3 => none

-- Theorem stating that Pictures 3 and 5 are valid configurations
theorem valid_pictures :
  valid_grid picture3 ∧ valid_grid picture5 := by sorry

-- Theorem stating that Pictures 1, 2, 4, and 6 are not valid configurations
theorem invalid_pictures :
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃,
      g x₁ y₁ = some 3 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 1 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧
      (∃ x₄ y₄ x₅ y₅, g x₄ y₄ = some 11 ∧ g x₅ y₅ = some 10 ∧ ¬adjacent x₄ y₄ x₅ y₅))) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃,
      g x₁ y₁ = some 1 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 3 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ ¬adjacent x₂ y₂ x₃ y₃)) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      g x₁ y₁ = some 1 ∧ g x₂ y₂ = some 2 ∧ g x₃ y₃ = some 3 ∧ g x₄ y₄ = some 4 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧ ¬adjacent x₃ y₃ x₄ y₄)) ∧
  ¬ (∃ g : Grid, valid_grid g ∧
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,
      g x₁ y₁ = some 4 ∧ g x₂ y₂ = some 5 ∧ g x₃ y₃ = some 6 ∧ g x₄ y₄ = some 7 ∧
      adjacent x₁ y₁ x₂ y₂ ∧ adjacent x₂ y₂ x₃ y₃ ∧ ¬adjacent x₃ y₃ x₄ y₄)) := by sorry

end NUMINAMATH_CALUDE_valid_pictures_invalid_pictures_l3945_394568


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l3945_394520

theorem geometric_mean_problem (k : ℝ) :
  (k + 9) * (6 - k) = (2 * k)^2 → k = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l3945_394520


namespace NUMINAMATH_CALUDE_exponential_function_max_min_sum_l3945_394507

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_max_min_sum (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ max (f a 0) (f a 1)) ∧
  (∀ x ∈ Set.Icc 0 1, f a x ≥ min (f a 0) (f a 1)) ∧
  (max (f a 0) (f a 1) + min (f a 0) (f a 1) = 3) →
  a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_function_max_min_sum_l3945_394507


namespace NUMINAMATH_CALUDE_num_triples_eq_three_l3945_394544

/-- The number of triples (a, b, c) of positive integers satisfying a + ab + abc = 11 -/
def num_triples : Nat :=
  (Finset.filter (fun t : Nat × Nat × Nat =>
    let (a, b, c) := t
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a + a * b + a * b * c = 11)
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12)))).card

/-- Theorem stating that there are exactly 3 triples (a, b, c) of positive integers
    satisfying a + ab + abc = 11 -/
theorem num_triples_eq_three : num_triples = 3 := by
  sorry

end NUMINAMATH_CALUDE_num_triples_eq_three_l3945_394544


namespace NUMINAMATH_CALUDE_angle_mor_measure_l3945_394589

/-- A regular octagon with vertices LMNOPQR -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices (i + 1)) = dist (vertices j) (vertices (j + 1))

/-- The measure of an angle in radians -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the measure of angle MOR in a regular octagon is π/8 radians (22.5°) -/
theorem angle_mor_measure (octagon : RegularOctagon) :
  let vertices := octagon.vertices
  angle_measure (vertices 0) (vertices 3) (vertices 5) = π / 8 := by sorry

end NUMINAMATH_CALUDE_angle_mor_measure_l3945_394589


namespace NUMINAMATH_CALUDE_laundry_day_lcm_l3945_394595

theorem laundry_day_lcm : Nat.lcm 6 (Nat.lcm 9 (Nat.lcm 12 15)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_laundry_day_lcm_l3945_394595


namespace NUMINAMATH_CALUDE_product_of_roots_l3945_394597

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 50*x + 35 = 0) → 
  (∃ a b c : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - a) * (x - b) * (x - c) ∧ a * b * c = -35) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3945_394597


namespace NUMINAMATH_CALUDE_expression_simplification_l3945_394576

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (18 * x^3) * (8 * y) * (1 / (6 * x * y)^2) = 4 * x / y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3945_394576


namespace NUMINAMATH_CALUDE_max_value_theorem_l3945_394515

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : 2*x + 2*y + z = 1) : 
  3*x*y + y*z + z*x ≤ 1/5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3945_394515


namespace NUMINAMATH_CALUDE_total_length_of_stationery_l3945_394571

/-- Given the lengths of a rubber, pen, and pencil with specific relationships,
    prove that their total length is 29 centimeters. -/
theorem total_length_of_stationery (rubber pen pencil : ℝ) : 
  pen = rubber + 3 →
  pencil = pen + 2 →
  pencil = 12 →
  rubber + pen + pencil = 29 := by
sorry

end NUMINAMATH_CALUDE_total_length_of_stationery_l3945_394571


namespace NUMINAMATH_CALUDE_sum_s_1_to_321_l3945_394508

-- Define s(n) as the sum of all odd digits of n
def s (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_s_1_to_321 : 
  (Finset.range 321).sum s + s 321 = 1727 := by sorry

end NUMINAMATH_CALUDE_sum_s_1_to_321_l3945_394508


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3945_394540

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3945_394540


namespace NUMINAMATH_CALUDE_solve_equation_l3945_394534

theorem solve_equation : ∃ x : ℝ, 15 * x = 5.7 ∧ x = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3945_394534


namespace NUMINAMATH_CALUDE_problem_solution_l3945_394529

theorem problem_solution (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225/73 → x = -647/177 :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3945_394529


namespace NUMINAMATH_CALUDE_subset_sum_inequality_l3945_394510

theorem subset_sum_inequality (n m : ℕ) (A : Finset ℕ) (h_m : m > 0) (h_n : n > 0) 
  (h_subset : A ⊆ Finset.range n)
  (h_closure : ∀ (i j : ℕ), i ∈ A → j ∈ A → i + j ≤ n → i + j ∈ A) :
  (A.sum id) / m ≥ (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_inequality_l3945_394510


namespace NUMINAMATH_CALUDE_triangle_area_implies_sin_A_l3945_394543

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the area of the triangle
def area (t : Triangle) : ℝ := t.a^2 - (t.b - t.c)^2

-- State the theorem
theorem triangle_area_implies_sin_A (t : Triangle) (h_area : area t = t.a^2 - (t.b - t.c)^2) :
  let A := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
  Real.sin A = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_sin_A_l3945_394543


namespace NUMINAMATH_CALUDE_average_of_eight_numbers_l3945_394593

theorem average_of_eight_numbers :
  ∀ (a₁ a₂ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ),
    (a₁ + a₂) / 2 = 20 →
    (b₁ + b₂ + b₃) / 3 = 26 →
    c₁ = c₂ - 4 →
    c₁ = c₃ - 6 →
    c₃ = 30 →
    (a₁ + a₂ + b₁ + b₂ + b₃ + c₁ + c₂ + c₃) / 8 = 25 := by
  sorry


end NUMINAMATH_CALUDE_average_of_eight_numbers_l3945_394593


namespace NUMINAMATH_CALUDE_initial_cats_in_shelter_l3945_394556

theorem initial_cats_in_shelter (initial_cats : ℕ) : 
  (initial_cats / 3 : ℚ) = (initial_cats / 3 : ℕ) →
  (4 * initial_cats / 3 + 8 * initial_cats / 3 : ℚ) = 60 →
  initial_cats = 15 := by
sorry

end NUMINAMATH_CALUDE_initial_cats_in_shelter_l3945_394556


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3945_394503

/-- Given a hyperbola with equation x²/4 - y² = 1, prove its asymptotes and eccentricity -/
theorem hyperbola_properties (x y : ℝ) :
  x^2 / 4 - y^2 = 1 →
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) ∧
  (∃ (e : ℝ), e = Real.sqrt 5 / 2 ∧ e > 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3945_394503


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3945_394514

/-- A quadratic function y = x^2 + bx + c -/
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_property (b c m n : ℝ) :
  (∀ x ≤ 2, quadratic_function b c (x + 0.01) < quadratic_function b c x) →
  quadratic_function b c m = n →
  quadratic_function b c (m + 1) = n →
  m ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3945_394514


namespace NUMINAMATH_CALUDE_max_value_constraint_l3945_394500

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3945_394500


namespace NUMINAMATH_CALUDE_equal_intercepts_implies_a_value_l3945_394517

/-- Given two points A(0, 1) and B(4, a) on a line, if the x-intercept and y-intercept of the line are equal, then a = -3. -/
theorem equal_intercepts_implies_a_value (a : ℝ) : 
  let A : ℝ × ℝ := (0, 1)
  let B : ℝ × ℝ := (4, a)
  let m : ℝ := (a - 1) / 4  -- Slope of the line AB
  let x_intercept : ℝ := 4 / (1 - m)  -- x-intercept formula
  let y_intercept : ℝ := a - m * 4  -- y-intercept formula
  x_intercept = y_intercept → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_equal_intercepts_implies_a_value_l3945_394517


namespace NUMINAMATH_CALUDE_remaining_students_l3945_394561

/-- Given a group of students divided into 3 groups of 8, with 2 students leaving early,
    prove that 22 students remain. -/
theorem remaining_students (initial_groups : Nat) (students_per_group : Nat) (students_left : Nat) :
  initial_groups = 3 →
  students_per_group = 8 →
  students_left = 2 →
  initial_groups * students_per_group - students_left = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_l3945_394561


namespace NUMINAMATH_CALUDE_binomial_20_choose_7_l3945_394581

theorem binomial_20_choose_7 : Nat.choose 20 7 = 5536 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_choose_7_l3945_394581


namespace NUMINAMATH_CALUDE_fraction_ordering_l3945_394525

theorem fraction_ordering : (4 : ℚ) / 13 < 12 / 37 ∧ 12 / 37 < 15 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3945_394525


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l3945_394511

theorem sum_of_squares_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 75) (h5 : a*b + b*c + c*a = 40) (h6 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l3945_394511


namespace NUMINAMATH_CALUDE_abc_sum_equals_888_l3945_394554

/-- Given ABC + ABC + ABC = 888, where A, B, and C are all different single digit numbers, prove A = 2 -/
theorem abc_sum_equals_888 (A B C : ℕ) : 
  (100 * A + 10 * B + C) * 3 = 888 →
  A < 10 ∧ B < 10 ∧ C < 10 →
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 2 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_equals_888_l3945_394554


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l3945_394590

/-- A square with area numerically equal to its side length has a perimeter of 4 units. -/
theorem square_perimeter_when_area_equals_side : ∀ s : ℝ,
  s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l3945_394590


namespace NUMINAMATH_CALUDE_tylers_and_brothers_age_sum_l3945_394539

theorem tylers_and_brothers_age_sum : 
  ∀ (tyler_age brother_age : ℕ),
    tyler_age = 7 →
    brother_age = tyler_age + 3 →
    tyler_age + brother_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_tylers_and_brothers_age_sum_l3945_394539


namespace NUMINAMATH_CALUDE_tangent_angle_at_one_l3945_394574

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_angle_at_one (x : ℝ) :
  let slope := f' 1
  let angle := Real.arctan slope
  angle = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_at_one_l3945_394574


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3945_394579

-- Define the function f(x)
def f (x : ℝ) : ℝ := 6 - 12*x + x^3

-- Define the interval
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem max_min_values_of_f :
  (∃ x ∈ interval, f x = 22 ∧ ∀ y ∈ interval, f y ≤ 22) ∧
  (∃ x ∈ interval, f x = -5 ∧ ∀ y ∈ interval, f y ≥ -5) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3945_394579


namespace NUMINAMATH_CALUDE_dave_pays_more_than_doug_l3945_394588

/-- Represents the cost and composition of a pizza -/
structure Pizza where
  slices : Nat
  base_cost : Nat
  olive_slices : Nat
  olive_cost : Nat
  mushroom_slices : Nat
  mushroom_cost : Nat

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : Nat :=
  p.base_cost + p.olive_cost + p.mushroom_cost

/-- Calculates the cost of a given number of slices -/
def slice_cost (p : Pizza) (n : Nat) (with_olive : Nat) (with_mushroom : Nat) : Nat :=
  let base := n * p.base_cost / p.slices
  let olive := with_olive * p.olive_cost / p.olive_slices
  let mushroom := with_mushroom * p.mushroom_cost / p.mushroom_slices
  base + olive + mushroom

/-- Theorem: Dave pays 10 dollars more than Doug -/
theorem dave_pays_more_than_doug (p : Pizza) 
    (h1 : p.slices = 12)
    (h2 : p.base_cost = 12)
    (h3 : p.olive_slices = 3)
    (h4 : p.olive_cost = 3)
    (h5 : p.mushroom_slices = 6)
    (h6 : p.mushroom_cost = 4) :
  slice_cost p 8 2 6 - slice_cost p 4 0 0 = 10 := by
  sorry


end NUMINAMATH_CALUDE_dave_pays_more_than_doug_l3945_394588


namespace NUMINAMATH_CALUDE_spring_length_dependent_on_mass_l3945_394591

/-- Represents the relationship between spring length and object mass -/
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

theorem spring_length_dependent_on_mass :
  ∃ (f : ℝ → ℝ), ∀ (mass : ℝ), spring_length mass = f mass ∧
  ¬ (∃ (g : ℝ → ℝ), ∀ (length : ℝ), mass = g length) :=
sorry

end NUMINAMATH_CALUDE_spring_length_dependent_on_mass_l3945_394591


namespace NUMINAMATH_CALUDE_least_common_multiple_of_primes_l3945_394541

theorem least_common_multiple_of_primes : ∃ n : ℕ,
  (n > 0) ∧
  (n % 7 = 0) ∧ (n % 11 = 0) ∧ (n % 13 = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m % 7 = 0 ∧ m % 11 = 0 ∧ m % 13 = 0 → m ≥ n) ∧
  n = 1001 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_primes_l3945_394541


namespace NUMINAMATH_CALUDE_prob_at_least_two_women_l3945_394527

/-- The probability of selecting at least 2 women from a group of 8 men and 4 women when choosing 4 people at random -/
theorem prob_at_least_two_women (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) : 
  total = men + women →
  men = 8 →
  women = 4 →
  selected = 4 →
  (1 : ℚ) - (Nat.choose men selected / Nat.choose total selected + 
    (Nat.choose women 1 * Nat.choose men (selected - 1)) / Nat.choose total selected) = 67 / 165 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_women_l3945_394527
