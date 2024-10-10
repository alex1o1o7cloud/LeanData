import Mathlib

namespace correct_change_l976_97679

/-- The change Bomi should receive after buying candy and chocolate -/
def bomi_change (candy_cost chocolate_cost paid : ℕ) : ℕ :=
  paid - (candy_cost + chocolate_cost)

/-- Theorem stating the correct change Bomi should receive -/
theorem correct_change : bomi_change 350 500 1000 = 150 := by
  sorry

end correct_change_l976_97679


namespace chefs_flour_calculation_l976_97645

theorem chefs_flour_calculation (recipe_ratio : ℚ) (eggs_needed : ℕ) (flour_used : ℚ) : 
  recipe_ratio = 7 / 2 →
  eggs_needed = 28 →
  flour_used = eggs_needed / recipe_ratio →
  flour_used = 8 := by
sorry

end chefs_flour_calculation_l976_97645


namespace angle_sum_in_special_figure_l976_97625

theorem angle_sum_in_special_figure (A B C x y : ℝ) : 
  A = 34 → B = 80 → C = 30 →
  (A + B + (360 - x) + 90 + (120 - y) = 720) →
  x + y = 36 := by sorry

end angle_sum_in_special_figure_l976_97625


namespace normal_distribution_probability_l976_97613

def normal_distribution (μ σ : ℝ) : Type := ℝ

def probability {α : Type} (p : Set α) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : normal_distribution 0 3) : 
  probability {x : ℝ | -3 < x ∧ x < 6} = 0.8185 := by sorry

end normal_distribution_probability_l976_97613


namespace traditionalist_fraction_l976_97675

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) (num_progressives : ℚ) :
  num_provinces = 15 →
  num_traditionalists_per_province = num_progressives / 20 →
  (num_provinces : ℚ) * num_traditionalists_per_province / ((num_provinces : ℚ) * num_traditionalists_per_province + num_progressives) = 3 / 7 := by
  sorry

end traditionalist_fraction_l976_97675


namespace square_on_hypotenuse_side_length_l976_97640

/-- Given a right triangle PQR with leg PR = 9 and leg PQ = 12, 
    prove that a square with one side along the hypotenuse and 
    one vertex each on legs PR and PQ has a side length of 5 5/7 -/
theorem square_on_hypotenuse_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (P.1 - Q.1) * (P.1 - R.1) + (P.2 - Q.2) * (P.2 - R.2) = 0)
  (leg_PR : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 9)
  (leg_PQ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 12)
  (S : ℝ × ℝ) 
  (T : ℝ × ℝ) 
  (square_side_on_hypotenuse : ∃ U : ℝ × ℝ, 
    (S.1 - T.1) * (Q.1 - R.1) + (S.2 - T.2) * (Q.2 - R.2) = 0 ∧
    Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2) = 
    Real.sqrt ((S.1 - U.1)^2 + (S.2 - U.2)^2) ∧
    Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) = 
    Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2))
  (S_on_PR : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ S = (P.1 + t * (R.1 - P.1), P.2 + t * (R.2 - P.2)))
  (T_on_PQ : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ T = (P.1 + s * (Q.1 - P.1), P.2 + s * (Q.2 - P.2))) :
  Real.sqrt ((S.1 - T.1)^2 + (S.2 - T.2)^2) = 5 + 5/7 := by
  sorry


end square_on_hypotenuse_side_length_l976_97640


namespace angle_inequality_l976_97627

theorem angle_inequality (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) := by
sorry

end angle_inequality_l976_97627


namespace optimal_AD_length_l976_97616

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

/-- Point D on AB -/
def D (t : Triangle) := ℝ

/-- Expected value of EF -/
noncomputable def expectedEF (t : Triangle) (d : D t) : ℝ := sorry

/-- Theorem statement -/
theorem optimal_AD_length (t : Triangle) 
  (h1 : t.AB = 14) 
  (h2 : t.AC = 13) 
  (h3 : t.BC = 15) : 
  ∃ (d : D t), 
    (∀ (d' : D t), expectedEF t d ≥ expectedEF t d') ∧ 
    d = Real.sqrt 70 :=
sorry

end optimal_AD_length_l976_97616


namespace max_value_complex_expression_l976_97634

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + z^2 - 5*z + 3) ≤ 128 * Real.sqrt 3 / 27 := by
  sorry

end max_value_complex_expression_l976_97634


namespace probability_two_white_balls_l976_97665

/-- The probability of drawing two white balls sequentially without replacement from a box containing 7 white balls and 8 black balls is 1/5. -/
theorem probability_two_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) :
  total_balls = white_balls + black_balls →
  white_balls = 7 →
  black_balls = 8 →
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 1 / 5 :=
by sorry

end probability_two_white_balls_l976_97665


namespace exists_greater_or_equal_scores_64_exists_greater_or_equal_scores_49_l976_97636

/-- Represents a student's scores on three problems -/
structure StudentScores where
  problem1 : Nat
  problem2 : Nat
  problem3 : Nat
  h1 : problem1 ≤ 7
  h2 : problem2 ≤ 7
  h3 : problem3 ≤ 7

/-- Checks if one student's scores are greater than or equal to another's -/
def scoresGreaterOrEqual (a b : StudentScores) : Prop :=
  a.problem1 ≥ b.problem1 ∧ a.problem2 ≥ b.problem2 ∧ a.problem3 ≥ b.problem3

/-- Main theorem for part (a) -/
theorem exists_greater_or_equal_scores_64 :
  ∀ (students : Fin 64 → StudentScores),
  ∃ (i j : Fin 64), i ≠ j ∧ scoresGreaterOrEqual (students i) (students j) := by
  sorry

/-- Main theorem for part (b) -/
theorem exists_greater_or_equal_scores_49 :
  ∀ (students : Fin 49 → StudentScores),
  ∃ (i j : Fin 49), i ≠ j ∧ scoresGreaterOrEqual (students i) (students j) := by
  sorry

end exists_greater_or_equal_scores_64_exists_greater_or_equal_scores_49_l976_97636


namespace hex_multiplication_l976_97661

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a hexadecimal digit to its decimal value --/
def hex_to_dec (d : HexDigit) : Nat :=
  match d with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Represents a two-digit hexadecimal number --/
structure HexNumber :=
  (msb : HexDigit)
  (lsb : HexDigit)

/-- Converts a two-digit hexadecimal number to its decimal value --/
def hex_number_to_dec (h : HexNumber) : Nat :=
  16 * (hex_to_dec h.msb) + (hex_to_dec h.lsb)

/-- The main theorem to prove --/
theorem hex_multiplication :
  let a := HexNumber.mk HexDigit.A HexDigit.A
  let b := HexNumber.mk HexDigit.B HexDigit.B
  let result := HexNumber.mk HexDigit.D6 HexDigit.E
  hex_number_to_dec a * hex_number_to_dec b = hex_number_to_dec result := by
  sorry

end hex_multiplication_l976_97661


namespace greatest_sum_of_digits_l976_97690

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the sum of digits for a given time -/
def sumOfDigits (t : Time) : Nat :=
  (t.hours / 10) + (t.hours % 10) + (t.minutes / 10) + (t.minutes % 10)

/-- States that 19:59 has the greatest sum of digits among all possible times -/
theorem greatest_sum_of_digits :
  ∀ t : Time, sumOfDigits t ≤ sumOfDigits ⟨19, 59, by norm_num, by norm_num⟩ := by
  sorry

end greatest_sum_of_digits_l976_97690


namespace cards_given_to_jeff_l976_97670

theorem cards_given_to_jeff (initial_cards : ℕ) (cards_to_john : ℕ) (cards_left : ℕ) :
  initial_cards = 573 →
  cards_to_john = 195 →
  cards_left = 210 →
  initial_cards - cards_to_john - cards_left = 168 :=
by sorry

end cards_given_to_jeff_l976_97670


namespace absolute_value_inequality_l976_97693

theorem absolute_value_inequality (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end absolute_value_inequality_l976_97693


namespace tenth_term_of_sequence_l976_97628

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

theorem tenth_term_of_sequence (a : ℚ) (r : ℚ) (h : a = 4 ∧ r = 1) :
  geometric_sequence a r 10 = 4 := by
  sorry

end tenth_term_of_sequence_l976_97628


namespace greatest_divisor_four_consecutive_integers_squared_l976_97697

theorem greatest_divisor_four_consecutive_integers_squared (n : ℕ) :
  ∃ (k : ℕ), k = 144 ∧ (∀ m : ℕ, m > k → ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))^2)) ∧
  (k ∣ (n * (n + 1) * (n + 2) * (n + 3))^2) := by
  sorry

end greatest_divisor_four_consecutive_integers_squared_l976_97697


namespace min_length_of_rectangle_l976_97652

theorem min_length_of_rectangle (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = a^2 → min x y ≥ a :=
by sorry

end min_length_of_rectangle_l976_97652


namespace floor_ceil_sum_l976_97603

theorem floor_ceil_sum : ⌊(1.99 : ℝ)⌋ + ⌈(3.02 : ℝ)⌉ = 5 := by
  sorry

end floor_ceil_sum_l976_97603


namespace wand_price_l976_97671

theorem wand_price (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 4 → 
  purchase_price = (1/8) * original_price → 
  original_price = 32 := by
sorry

end wand_price_l976_97671


namespace intersection_midpoint_l976_97678

/-- The midpoint of the line segment connecting the intersection points of y = x and y^2 = 4x is (2,2) -/
theorem intersection_midpoint :
  let line := {(x, y) : ℝ × ℝ | y = x}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4*x}
  let intersection := line ∩ parabola
  ∃ (a b : ℝ × ℝ), a ∈ intersection ∧ b ∈ intersection ∧ a ≠ b ∧
    (a.1 + b.1) / 2 = 2 ∧ (a.2 + b.2) / 2 = 2 :=
by sorry

end intersection_midpoint_l976_97678


namespace three_color_circle_existence_l976_97656

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- State the theorem
theorem three_color_circle_existence 
  (coloring : Coloring) 
  (all_colors_used : ∀ c : Color, ∃ p : Point, coloring p = c) :
  ∃ circ : Circle, ∀ c : Color, ∃ p : Point, 
    coloring p = c ∧ (p.x - circ.center.x)^2 + (p.y - circ.center.y)^2 ≤ circ.radius^2 :=
sorry

end three_color_circle_existence_l976_97656


namespace product_of_reals_l976_97611

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end product_of_reals_l976_97611


namespace head_start_time_l976_97674

/-- Proves that given a runner completes a 1000-meter race in 190 seconds,
    the time equivalent to a 50-meter head start is 9.5 seconds. -/
theorem head_start_time (race_distance : ℝ) (race_time : ℝ) (head_start_distance : ℝ) : 
  race_distance = 1000 →
  race_time = 190 →
  head_start_distance = 50 →
  (head_start_distance / (race_distance / race_time)) = 9.5 := by
  sorry

end head_start_time_l976_97674


namespace complex_equation_solution_l976_97687

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end complex_equation_solution_l976_97687


namespace weeding_rate_calculation_l976_97655

/-- The hourly rate for mowing lawns -/
def mowing_rate : ℝ := 4

/-- The number of hours spent mowing lawns in September -/
def mowing_hours : ℝ := 25

/-- The number of hours spent pulling weeds in September -/
def weeding_hours : ℝ := 3

/-- The total earnings for September and October -/
def total_earnings : ℝ := 248

/-- The hourly rate for pulling weeds -/
def weeding_rate : ℝ := 8

theorem weeding_rate_calculation :
  2 * (mowing_rate * mowing_hours + weeding_rate * weeding_hours) = total_earnings :=
by sorry

end weeding_rate_calculation_l976_97655


namespace first_stack_height_is_5_l976_97647

/-- The height of the first stack of blocks -/
def first_stack_height : ℕ := sorry

/-- The height of the second stack of blocks -/
def second_stack_height : ℕ := first_stack_height + 2

/-- The height of the third stack of blocks -/
def third_stack_height : ℕ := second_stack_height - 5

/-- The height of the fourth stack of blocks -/
def fourth_stack_height : ℕ := third_stack_height + 5

/-- The total number of blocks used -/
def total_blocks : ℕ := 21

theorem first_stack_height_is_5 : 
  first_stack_height = 5 ∧ 
  first_stack_height + second_stack_height + third_stack_height + fourth_stack_height = total_blocks :=
sorry

end first_stack_height_is_5_l976_97647


namespace back_wheel_circumference_l976_97649

/-- Given a cart with front and back wheels, this theorem proves the circumference of the back wheel
    based on the given conditions. -/
theorem back_wheel_circumference
  (front_circumference : ℝ)
  (distance : ℝ)
  (revolution_difference : ℕ)
  (h1 : front_circumference = 30)
  (h2 : distance = 1650)
  (h3 : revolution_difference = 5) :
  ∃ (back_circumference : ℝ),
    back_circumference * (distance / front_circumference - revolution_difference) = distance ∧
    back_circumference = 33 :=
by sorry

end back_wheel_circumference_l976_97649


namespace simplified_expression_value_l976_97639

theorem simplified_expression_value (a b : ℚ) 
  (h1 : a = -1) 
  (h2 : b = 1/4) : 
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end simplified_expression_value_l976_97639


namespace parallel_line_through_point_l976_97604

/-- Given a line L1: 2x + y - 3 = 0 and a point P(0, 1), 
    prove that the line L2: 2x + y - 1 = 0 passes through P and is parallel to L1. -/
theorem parallel_line_through_point (x y : ℝ) : 
  let L1 := {(x, y) | 2 * x + y - 3 = 0}
  let P := (0, 1)
  let L2 := {(x, y) | 2 * x + y - 1 = 0}
  (P ∈ L2) ∧ (∀ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ L1 → (x2, y2) ∈ L1 → 
    (x2 - x1) * 1 = (y2 - y1) * 2 ↔ 
    (x2 - x1) * 1 = (y2 - y1) * 2) := by
  sorry


end parallel_line_through_point_l976_97604


namespace triangle_existence_l976_97683

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the function to check if a line cuts off equal segments from a point on two other lines
def cutsEqualSegments (l : Line) (A : Point) (AB AC : Line) : Prop :=
  ∃ (P Q : Point), isPointOnLine P AB ∧ isPointOnLine Q AC ∧
    isPointOnLine P l ∧ isPointOnLine Q l ∧
    (P.x - A.x)^2 + (P.y - A.y)^2 = (Q.x - A.x)^2 + (Q.y - A.y)^2

-- State the theorem
theorem triangle_existence 
  (A O : Point) 
  (l : Line) 
  (h_euler : isPointOnLine O l)
  (h_equal_segments : ∃ (AB AC : Line), cutsEqualSegments l A AB AC) :
  ∃ (T : Triangle), T.A = A ∧ 
    isPointOnLine T.B l ∧ isPointOnLine T.C l ∧
    (T.B.x - O.x)^2 + (T.B.y - O.y)^2 = (T.C.x - O.x)^2 + (T.C.y - O.y)^2 :=
sorry

end triangle_existence_l976_97683


namespace sine_cosine_inequality_l976_97637

theorem sine_cosine_inequality (x : ℝ) (n m : ℕ) 
  (h1 : 0 < x) (h2 : x < π / 2) (h3 : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| := by
  sorry

end sine_cosine_inequality_l976_97637


namespace quadratic_sum_of_squares_l976_97667

theorem quadratic_sum_of_squares (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (∃ (x y z : ℝ),
    (x^2 + a*x + b = 0 ∧ y^2 + b*y + c = 0 ∧ x = y) ∧
    (y^2 + b*y + c = 0 ∧ z^2 + c*z + a = 0 ∧ y = z) ∧
    (z^2 + c*z + a = 0 ∧ x^2 + a*x + b = 0 ∧ z = x)) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end quadratic_sum_of_squares_l976_97667


namespace combination_sum_equals_seven_l976_97698

theorem combination_sum_equals_seven (n : ℕ) 
  (h1 : 0 ≤ 5 - n ∧ 5 - n ≤ n) 
  (h2 : 0 ≤ 10 - n ∧ 10 - n ≤ n + 1) : 
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 := by
  sorry

end combination_sum_equals_seven_l976_97698


namespace complex_fraction_simplification_l976_97612

theorem complex_fraction_simplification :
  2017 * (2016 / 2017) / (2019 * (1 / 2016)) + 1 / 2017 = 1 := by
  sorry

end complex_fraction_simplification_l976_97612


namespace expression_evaluation_l976_97623

theorem expression_evaluation :
  let a : ℚ := -1/3
  let b : ℚ := -3
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := by
  sorry

end expression_evaluation_l976_97623


namespace find_b_l976_97680

def is_valid_set (x b : ℕ) : Prop :=
  x > 0 ∧ x + 2 > 0 ∧ x + b > 0 ∧ x + 7 > 0 ∧ x + 32 > 0

def median (x b : ℕ) : ℚ := x + b

def mean (x b : ℕ) : ℚ := (x + (x + 2) + (x + b) + (x + 7) + (x + 32)) / 5

theorem find_b (x : ℕ) :
  ∃ b : ℕ, is_valid_set x b ∧ mean x b = median x b + 5 → b = 4 := by
  sorry

end find_b_l976_97680


namespace area_enclosed_by_line_and_parabola_l976_97646

/-- The area of the region enclosed by y = (a/6)x and y = x^2, 
    where a is the constant term in (x + 2/x)^n and 
    the sum of coefficients in the expansion is 81 -/
theorem area_enclosed_by_line_and_parabola (n : ℕ) (a : ℝ) : 
  (3 : ℝ)^n = 81 →
  (∃ k, (Nat.choose 4 2) * 2^2 = k ∧ k = a) →
  (∫ x in (0)..(a/4), (a/6 * x - x^2)) = 32/3 := by
sorry

end area_enclosed_by_line_and_parabola_l976_97646


namespace jennifer_cookie_sales_l976_97600

theorem jennifer_cookie_sales (kim_sales : ℕ) (jennifer_extra : ℕ) 
  (h1 : kim_sales = 54)
  (h2 : jennifer_extra = 17) : 
  kim_sales + jennifer_extra = 71 := by
  sorry

end jennifer_cookie_sales_l976_97600


namespace amanda_lost_notebooks_l976_97659

/-- The number of notebooks Amanda lost -/
def notebooks_lost (initial : ℕ) (ordered : ℕ) (current : ℕ) : ℕ :=
  initial + ordered - current

theorem amanda_lost_notebooks : notebooks_lost 10 6 14 = 2 := by
  sorry

end amanda_lost_notebooks_l976_97659


namespace hilton_marbles_l976_97609

/-- Calculates the final number of marbles Hilton has --/
def final_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

/-- Proves that Hilton ends up with 42 marbles given the initial conditions --/
theorem hilton_marbles : final_marbles 26 6 10 = 42 := by
  sorry

end hilton_marbles_l976_97609


namespace book_sale_profit_l976_97664

/-- Represents the profit calculation for a book sale with and without discount -/
theorem book_sale_profit (cost_price : ℝ) (discount_percent : ℝ) (profit_with_discount_percent : ℝ) :
  discount_percent = 5 →
  profit_with_discount_percent = 23.5 →
  let selling_price_with_discount := cost_price * (1 + profit_with_discount_percent / 100 - discount_percent / 100)
  let selling_price_without_discount := selling_price_with_discount + cost_price * (discount_percent / 100)
  let profit_without_discount_percent := (selling_price_without_discount - cost_price) / cost_price * 100
  profit_without_discount_percent = 23.5 :=
by sorry

end book_sale_profit_l976_97664


namespace exists_expression_for_100_l976_97630

/-- A type representing arithmetic expressions using only the number 7 --/
inductive Expr
  | seven : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluate an expression to a rational number --/
def eval : Expr → ℚ
  | Expr.seven => 7
  | Expr.add e₁ e₂ => eval e₁ + eval e₂
  | Expr.sub e₁ e₂ => eval e₁ - eval e₂
  | Expr.mul e₁ e₂ => eval e₁ * eval e₂
  | Expr.div e₁ e₂ => eval e₁ / eval e₂

/-- Count the number of sevens in an expression --/
def countSevens : Expr → ℕ
  | Expr.seven => 1
  | Expr.add e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.sub e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.mul e₁ e₂ => countSevens e₁ + countSevens e₂
  | Expr.div e₁ e₂ => countSevens e₁ + countSevens e₂

/-- There exists an expression using fewer than 10 sevens that evaluates to 100 --/
theorem exists_expression_for_100 : ∃ e : Expr, eval e = 100 ∧ countSevens e < 10 := by
  sorry

end exists_expression_for_100_l976_97630


namespace max_red_balls_l976_97607

/-- Given a pile of red and white balls, with the total number not exceeding 50,
    and the number of red balls being three times the number of white balls,
    prove that the maximum number of red balls is 36. -/
theorem max_red_balls (r w : ℕ) : 
  r + w ≤ 50 →  -- Total number of balls not exceeding 50
  r = 3 * w →   -- Number of red balls is three times the number of white balls
  r ≤ 36        -- Maximum number of red balls is 36
  := by sorry

end max_red_balls_l976_97607


namespace malcom_brandon_card_difference_l976_97619

theorem malcom_brandon_card_difference :
  ∀ (brandon_cards malcom_cards_initial malcom_cards_after : ℕ),
    brandon_cards = 20 →
    malcom_cards_initial > brandon_cards →
    malcom_cards_after = 14 →
    malcom_cards_after * 2 = malcom_cards_initial →
    malcom_cards_initial - brandon_cards = 8 :=
by sorry

end malcom_brandon_card_difference_l976_97619


namespace peggy_initial_dolls_l976_97601

theorem peggy_initial_dolls :
  ∀ (initial : ℕ) (grandmother_gift : ℕ) (birthday_christmas : ℕ),
    grandmother_gift = 30 →
    birthday_christmas = grandmother_gift / 2 →
    initial + grandmother_gift + birthday_christmas = 51 →
    initial = 6 :=
by
  sorry

end peggy_initial_dolls_l976_97601


namespace compare_sqrt_l976_97651

theorem compare_sqrt : 2 * Real.sqrt 11 < 3 * Real.sqrt 5 := by
  sorry

end compare_sqrt_l976_97651


namespace ad_greater_than_bc_l976_97686

theorem ad_greater_than_bc (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (sum_eq : a + d = b + c)
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
sorry

end ad_greater_than_bc_l976_97686


namespace star_calculation_l976_97643

-- Define the ☆ operation for rational numbers
def star (a b : ℚ) : ℚ := 2 * a - b + 1

-- Theorem statement
theorem star_calculation : star 1 (star 2 (-3)) = -5 := by sorry

end star_calculation_l976_97643


namespace sqrt_expression_simplification_l976_97682

theorem sqrt_expression_simplification :
  Real.sqrt 24 - 3 * Real.sqrt (1/6) + Real.sqrt 6 = (5 * Real.sqrt 6) / 2 := by
  sorry

end sqrt_expression_simplification_l976_97682


namespace movie_duration_l976_97629

theorem movie_duration (flight_duration : ℕ) (tv_time : ℕ) (sleep_time : ℕ) (remaining_time : ℕ) (num_movies : ℕ) :
  flight_duration = 600 →
  tv_time = 75 →
  sleep_time = 270 →
  remaining_time = 45 →
  num_movies = 2 →
  ∃ (movie_duration : ℕ),
    flight_duration = tv_time + sleep_time + num_movies * movie_duration + remaining_time ∧
    movie_duration = 105 := by
  sorry

end movie_duration_l976_97629


namespace arithmetic_sequence_common_difference_l976_97610

/-- An arithmetic sequence with sum S_n for the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- The common difference of an arithmetic sequence given specific conditions -/
theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence)
  (h1 : seq.a 4 + seq.a 5 = 24)
  (h2 : seq.S 6 = 48) :
  seq.d = 4 := by
  sorry


end arithmetic_sequence_common_difference_l976_97610


namespace simplify_power_sum_l976_97617

theorem simplify_power_sum : (-2)^2003 + 2^2004 + (-2)^2005 - 2^2006 = 5 * 2^2003 := by
  sorry

end simplify_power_sum_l976_97617


namespace polynomial_equality_l976_97622

theorem polynomial_equality : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end polynomial_equality_l976_97622


namespace third_term_is_64_l976_97673

/-- A geometric sequence with positive integer terms -/
structure GeometricSequence where
  terms : ℕ → ℕ
  first_term : terms 1 = 4
  is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, terms (n + 1) = (terms n : ℚ) * r

/-- The theorem stating that for a geometric sequence with first term 4 and fourth term 256, the third term is 64 -/
theorem third_term_is_64 (seq : GeometricSequence) (h : seq.terms 4 = 256) : seq.terms 3 = 64 := by
  sorry

end third_term_is_64_l976_97673


namespace conic_section_classification_l976_97620

/-- The equation y^4 - 9x^4 = 3y^2 - 4 represents the union of two hyperbolas -/
theorem conic_section_classification (x y : ℝ) :
  (y^4 - 9*x^4 = 3*y^2 - 4) ↔
  ((y^2 - 3*x^2 = 5/2) ∨ (y^2 - 3*x^2 = 1)) :=
sorry

end conic_section_classification_l976_97620


namespace isosceles_triangle_base_length_l976_97626

/-- An isosceles triangle with two sides of 7 cm each and a perimeter of 23 cm has a base of 9 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
    base > 0 → 
    7 + 7 + base = 23 → 
    base = 9 := by
  sorry

end isosceles_triangle_base_length_l976_97626


namespace number_difference_l976_97663

theorem number_difference (L S : ℝ) (h1 : L = 1650) (h2 : L = 6 * S + 15) : 
  L - S = 1377.5 := by
sorry

end number_difference_l976_97663


namespace maria_chocolate_chip_cookies_l976_97684

/-- Calculates the number of chocolate chip cookies Maria had -/
def chocolateChipCookies (cookiesPerBag : ℕ) (oatmealCookies : ℕ) (numBags : ℕ) : ℕ :=
  cookiesPerBag * numBags - oatmealCookies

/-- Proves that Maria had 5 chocolate chip cookies -/
theorem maria_chocolate_chip_cookies :
  chocolateChipCookies 8 19 3 = 5 := by
  sorry

end maria_chocolate_chip_cookies_l976_97684


namespace inlet_fill_rate_l976_97608

/-- Given a tank with the following properties:
  * Capacity: 12960 liters
  * Time to empty with leak alone: 9 hours
  * Time to empty with leak and inlet: 12 hours
  Prove that the rate at which the inlet pipe fills water is 2520 liters per hour. -/
theorem inlet_fill_rate 
  (tank_capacity : ℝ) 
  (empty_time_leak : ℝ) 
  (empty_time_leak_and_inlet : ℝ) 
  (h1 : tank_capacity = 12960)
  (h2 : empty_time_leak = 9)
  (h3 : empty_time_leak_and_inlet = 12) :
  (tank_capacity / empty_time_leak) + (tank_capacity / empty_time_leak_and_inlet) = 2520 :=
by sorry

end inlet_fill_rate_l976_97608


namespace number_of_girls_in_school_l976_97648

theorem number_of_girls_in_school (total_boys : ℕ) (total_sections : ℕ) 
  (h1 : total_boys = 408)
  (h2 : total_sections = 27)
  (h3 : total_boys % total_sections = 0) -- Boys are divided into equal sections
  : ∃ (total_girls : ℕ), 
    total_girls = 324 ∧ 
    total_girls % total_sections = 0 ∧ -- Girls are divided into equal sections
    (total_boys / total_sections + total_girls / total_sections = total_sections) :=
by
  sorry


end number_of_girls_in_school_l976_97648


namespace asha_win_probability_l976_97676

theorem asha_win_probability (lose_prob tie_prob : ℚ) 
  (lose_eq : lose_prob = 3 / 7)
  (tie_eq : tie_prob = 1 / 7)
  (total_prob : lose_prob + tie_prob + (1 - lose_prob - tie_prob) = 1) :
  1 - lose_prob - tie_prob = 3 / 7 := by
sorry

end asha_win_probability_l976_97676


namespace max_value_of_expression_l976_97691

theorem max_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (2 * a) / (a^2 + b) + b / (a + b^2) ≤ (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end max_value_of_expression_l976_97691


namespace arithmetic_sequence_common_difference_l976_97681

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_fifth : a 5 = 8)
  (h_sum : a 1 + a 2 + a 3 = 6) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l976_97681


namespace total_people_shook_hands_l976_97657

/-- The number of schools participating in the debate -/
def num_schools : ℕ := 5

/-- The number of students in the fourth school -/
def students_fourth : ℕ := 150

/-- The number of faculty members per school -/
def faculty_per_school : ℕ := 10

/-- The number of event staff per school -/
def event_staff_per_school : ℕ := 5

/-- Calculate the number of students in the third school -/
def students_third : ℕ := (3 * students_fourth) / 2

/-- Calculate the number of students in the second school -/
def students_second : ℕ := students_third + 50

/-- Calculate the number of students in the first school -/
def students_first : ℕ := 2 * students_second

/-- Calculate the number of students in the fifth school -/
def students_fifth : ℕ := students_fourth - 120

/-- Calculate the total number of students -/
def total_students : ℕ := students_first + students_second + students_third + students_fourth + students_fifth

/-- Calculate the total number of faculty and staff -/
def total_faculty_staff : ℕ := num_schools * (faculty_per_school + event_staff_per_school)

/-- The theorem to prove -/
theorem total_people_shook_hands : total_students + total_faculty_staff = 1305 := by
  sorry

end total_people_shook_hands_l976_97657


namespace tangent_line_parabola_l976_97618

/-- The equation of the tangent line to the parabola y = x^2 that is parallel to the line y = 2x is 2x - y - 1 = 0 -/
theorem tangent_line_parabola (x y : ℝ) : 
  (y = x^2) →  -- parabola equation
  (∃ m : ℝ, m = 2) →  -- parallel to y = 2x
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ 
    ∀ x₀ y₀ : ℝ, y₀ = x₀^2 → (y₀ - (x₀^2) = m * (x - x₀))) →  -- tangent line equation
  (2 * x - y - 1 = 0) :=
by sorry

end tangent_line_parabola_l976_97618


namespace special_numbers_count_l976_97660

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Counts the number of two-digit integers x for which digit_sum(digit_sum(x)) = 4 -/
def count_special_numbers : ℕ := sorry

theorem special_numbers_count : count_special_numbers = 10 := by sorry

end special_numbers_count_l976_97660


namespace b_not_played_e_l976_97606

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents the number of matches played by each team -/
def matches_played : Team → Nat
| Team.A => 5
| Team.B => 4
| Team.C => 3
| Team.D => 2
| Team.E => 1
| Team.F => 0  -- Inferred from the problem

/-- Predicate to check if two teams have played against each other -/
def has_played_against : Team → Team → Prop := sorry

/-- The theorem stating that team B has not played against team E -/
theorem b_not_played_e : ¬(has_played_against Team.B Team.E) := by
  sorry

end b_not_played_e_l976_97606


namespace tan_beta_value_l976_97614

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by sorry

end tan_beta_value_l976_97614


namespace polynomial_factor_coefficient_l976_97662

theorem polynomial_factor_coefficient (a b : ℤ) : 
  (∃ (c d : ℤ), ∀ (x : ℝ), 
    a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8 = (2 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 4)) →
  a = 112 ∧ b = -152 := by
sorry

end polynomial_factor_coefficient_l976_97662


namespace floor_sqrt_ten_l976_97677

theorem floor_sqrt_ten : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end floor_sqrt_ten_l976_97677


namespace modular_arithmetic_problem_l976_97615

theorem modular_arithmetic_problem (m : ℕ) : 
  m < 41 ∧ (5 * m) % 41 = 1 → (3^m % 41)^2 % 41 - 3 % 41 = 6 % 41 := by
  sorry

end modular_arithmetic_problem_l976_97615


namespace toy_ratio_l976_97621

def total_toys : ℕ := 240
def elder_son_toys : ℕ := 60

def younger_son_toys : ℕ := total_toys - elder_son_toys

theorem toy_ratio :
  (younger_son_toys : ℚ) / elder_son_toys = 3 / 1 := by sorry

end toy_ratio_l976_97621


namespace min_value_theorem_min_value_achieved_l976_97650

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  a + 2*b + 3*c ≥ 18 :=
by sorry

theorem min_value_achieved (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 2/b + 3/c = 2) :
  (a + 2*b + 3*c = 18) ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by sorry

end min_value_theorem_min_value_achieved_l976_97650


namespace lines_theorem_l976_97688

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2*x + 3*y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₃ (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 2*x + y - 3 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

theorem lines_theorem :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →  -- P exists as intersection of l₁ and l₂
  (parallel_line (P.1) (P.2)) ∧    -- Parallel line passes through P
  (∀ (x y : ℝ), parallel_line x y → (∃ (k : ℝ), y - P.2 = k * (x - P.1) ∧ y = -2*x + 5)) ∧  -- Parallel line is parallel to l₃
  (perpendicular_line (P.1) (P.2)) ∧  -- Perpendicular line passes through P
  (∀ (x y : ℝ), perpendicular_line x y → 
    (∃ (k₁ k₂ : ℝ), y - P.2 = k₁ * (x - P.1) ∧ y = k₂ * x - 5/2 ∧ k₁ * k₂ = -1)) -- Perpendicular line is perpendicular to l₃
  := by sorry

end lines_theorem_l976_97688


namespace price_calculation_equivalence_l976_97633

theorem price_calculation_equivalence 
  (initial_price tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) 
  (discount_rate_pos : 0 < discount_rate) 
  (tax_rate_bound : tax_rate < 1) 
  (discount_rate_bound : discount_rate < 1) :
  initial_price * (1 + tax_rate) * (1 - discount_rate) = 
  initial_price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end price_calculation_equivalence_l976_97633


namespace negation_equivalence_l976_97669

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l976_97669


namespace quadratic_roots_positive_conditions_l976_97692

theorem quadratic_roots_positive_conditions (a b c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) →
  (b^2 - 4*a*c ≥ 0 ∧ a*c > 0 ∧ a*b < 0) ∧
  ¬(b^2 - 4*a*c ≥ 0 ∧ a*c > 0 ∧ a*b < 0 → 
    ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end quadratic_roots_positive_conditions_l976_97692


namespace color_assignment_count_l976_97638

theorem color_assignment_count : ∀ (n m : ℕ), n = 5 ∧ m = 3 →
  (n * (n - 1) * (n - 2)) = 60 := by
  sorry

end color_assignment_count_l976_97638


namespace line_equation_l976_97644

/-- Proves that the line represented by the given parametric equations has the equation y = 2x - 4 -/
theorem line_equation (t : ℝ) :
  let x := 3 * t + 1
  let y := 6 * t - 2
  y = 2 * x - 4 := by sorry

end line_equation_l976_97644


namespace license_plate_count_l976_97631

/-- The number of possible letters in each position of the license plate -/
def num_letters : ℕ := 26

/-- The number of odd digits available for the first position -/
def num_odd_digits : ℕ := 5

/-- The number of even digits available for the second position -/
def num_even_digits : ℕ := 5

/-- The number of digits that are multiples of 3 available for the third position -/
def num_multiples_of_3 : ℕ := 4

/-- The total number of license plates satisfying the given conditions -/
def total_license_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_even_digits * num_multiples_of_3

theorem license_plate_count : total_license_plates = 878800 := by
  sorry

end license_plate_count_l976_97631


namespace solution_satisfies_equations_l976_97668

theorem solution_satisfies_equations :
  ∃ (x y : ℝ), 3 * x - 8 * y = 2 ∧ 4 * y - x = 6 ∧ x = 14 ∧ y = 5 := by
  sorry

end solution_satisfies_equations_l976_97668


namespace abs_equality_l976_97635

theorem abs_equality (x : ℝ) : 
  (|x| = Real.sqrt (x^2)) ∧ 
  (|x| = if x ≥ 0 then x else -x) := by sorry

end abs_equality_l976_97635


namespace fare_ratio_proof_l976_97666

theorem fare_ratio_proof (passenger_ratio : ℚ) (total_amount : ℕ) (second_class_amount : ℕ) :
  passenger_ratio = 1 / 50 →
  total_amount = 1325 →
  second_class_amount = 1250 →
  ∃ (first_class_fare second_class_fare : ℕ),
    first_class_fare / second_class_fare = 3 :=
by sorry

end fare_ratio_proof_l976_97666


namespace triangle_area_from_lines_l976_97699

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines (f g h : ℝ → ℝ) :
  (f = fun x ↦ x + 2) →
  (g = fun x ↦ -3*x + 9) →
  (h = fun _ ↦ 2) →
  let p₁ := (0, 2)
  let p₂ := (7/3, 2)
  let p₃ := (7/4, 15/4)
  let base := p₂.1 - p₁.1
  let height := p₃.2 - 2
  1/2 * base * height = 49/24 := by
  sorry

end triangle_area_from_lines_l976_97699


namespace imaginary_part_of_product_l976_97602

theorem imaginary_part_of_product : Complex.im ((3 - Complex.I) * (2 + Complex.I)) = 1 := by
  sorry

end imaginary_part_of_product_l976_97602


namespace target_probabilities_l976_97653

def prob_hit : ℝ := 0.8
def total_shots : ℕ := 4

theorem target_probabilities :
  let prob_miss := 1 - prob_hit
  (1 - prob_miss ^ total_shots = 0.9984) ∧
  (prob_hit ^ 3 * prob_miss * total_shots + prob_hit ^ total_shots = 0.8192) ∧
  (prob_miss ^ total_shots + total_shots * prob_hit * prob_miss ^ 3 = 0.2576) := by
  sorry

#check target_probabilities

end target_probabilities_l976_97653


namespace polygon_interior_exterior_angle_sum_l976_97642

theorem polygon_interior_exterior_angle_sum (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 2 * 360) → 
  n = 6 := by
  sorry

end polygon_interior_exterior_angle_sum_l976_97642


namespace total_age_l976_97695

def kate_age : ℕ := 19
def maggie_age : ℕ := 17
def sue_age : ℕ := 12

theorem total_age : kate_age + maggie_age + sue_age = 48 := by
  sorry

end total_age_l976_97695


namespace existence_of_special_integers_l976_97624

theorem existence_of_special_integers : 
  ∃ (a b : ℕ+), 
    (¬ (7 ∣ a.val)) ∧ 
    (¬ (7 ∣ b.val)) ∧ 
    (¬ (7 ∣ (a.val + b.val))) ∧ 
    (7^7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) := by
  sorry

end existence_of_special_integers_l976_97624


namespace cost_price_correct_l976_97658

/-- The cost price of a product satisfying given conditions -/
def cost_price : ℝ := 90

/-- The marked price of the product -/
def marked_price : ℝ := 120

/-- The discount rate applied to the product -/
def discount_rate : ℝ := 0.1

/-- The profit rate relative to the cost price -/
def profit_rate : ℝ := 0.2

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct : 
  cost_price * (1 + profit_rate) = marked_price * (1 - discount_rate) := by
  sorry

#eval cost_price -- Should output 90

end cost_price_correct_l976_97658


namespace square_area_from_perimeter_l976_97672

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 48 → area = (perimeter / 4)^2 → area = 144 := by
  sorry

end square_area_from_perimeter_l976_97672


namespace coin_combination_theorem_l976_97694

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  fiveCent : ℕ
  tenCent : ℕ
  twentyFiveCent : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def differentValues (coins : CoinCounts) : ℕ := sorry

theorem coin_combination_theorem (coins : CoinCounts) :
  coins.fiveCent + coins.tenCent + coins.twentyFiveCent = 15 →
  differentValues coins = 23 →
  coins.twentyFiveCent = 3 := by sorry

end coin_combination_theorem_l976_97694


namespace work_completion_time_l976_97632

-- Define the work completion time for B
def b_time : ℝ := 8

-- Define the work completion time for A and B together
def ab_time : ℝ := 4.444444444444445

-- Define the work completion time for A
def a_time : ℝ := 10

-- Theorem statement
theorem work_completion_time :
  b_time = 8 ∧ ab_time = 4.444444444444445 →
  a_time = 10 :=
by sorry

end work_completion_time_l976_97632


namespace m_plus_n_values_l976_97696

theorem m_plus_n_values (m n : ℤ) (hm : |m| = 4) (hn : |n| = 5) (hn_neg : n < 0) :
  m + n = -1 ∨ m + n = -9 := by
  sorry

end m_plus_n_values_l976_97696


namespace square_sum_divided_l976_97689

theorem square_sum_divided : (2005^2 + 2 * 2005 * 1995 + 1995^2) / 800 = 20000 := by
  sorry

end square_sum_divided_l976_97689


namespace unique_solution_condition_l976_97641

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by sorry

end unique_solution_condition_l976_97641


namespace regular_milk_consumption_l976_97685

def total_milk : ℝ := 0.6
def soy_milk : ℝ := 0.1

theorem regular_milk_consumption : total_milk - soy_milk = 0.5 := by
  sorry

end regular_milk_consumption_l976_97685


namespace sergey_age_l976_97654

/-- Calculates the number of full years given a person's age components --/
def fullYears (years months weeks days hours : ℕ) : ℕ :=
  years + (months / 12) + ((weeks * 7 + days) / 365)

/-- Theorem stating that given the specific age components, the result is 39 full years --/
theorem sergey_age : fullYears 36 36 36 36 36 = 39 := by
  sorry

end sergey_age_l976_97654


namespace square_rectangle_area_relation_l976_97605

theorem square_rectangle_area_relation :
  let square_side : ℝ → ℝ := λ x => x - 5
  let rect_length : ℝ → ℝ := λ x => x - 4
  let rect_width : ℝ → ℝ := λ x => x + 5
  let square_area : ℝ → ℝ := λ x => (square_side x) ^ 2
  let rect_area : ℝ → ℝ := λ x => (rect_length x) * (rect_width x)
  ∃ x₁ x₂ : ℝ, x₁ > 5 ∧ x₂ > 5 ∧
    2 * x₁^2 - 31 * x₁ + 95 = 0 ∧
    2 * x₂^2 - 31 * x₂ + 95 = 0 ∧
    3 * (square_area x₁) = rect_area x₁ ∧
    3 * (square_area x₂) = rect_area x₂ ∧
    x₁ + x₂ = 31/2 :=
by
  sorry

end square_rectangle_area_relation_l976_97605
