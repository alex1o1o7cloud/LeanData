import Mathlib

namespace NUMINAMATH_CALUDE_solution_comparison_l1698_169871

theorem solution_comparison (p q r s : ℝ) (hp : p ≠ 0) (hr : r ≠ 0) :
  (-q / p > -s / r) ↔ (s * r > q * p) :=
by sorry

end NUMINAMATH_CALUDE_solution_comparison_l1698_169871


namespace NUMINAMATH_CALUDE_probability_consecutive_is_one_eighteenth_l1698_169875

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling four dice -/
def AllOutcomes : Finset (Die × Die × Die × Die) := sorry

/-- A function to check if four numbers are consecutive -/
def AreConsecutive (a b c d : ℕ) : Prop := sorry

/-- The set of favorable outcomes (four consecutive numbers in any order) -/
def FavorableOutcomes : Finset (Die × Die × Die × Die) := sorry

/-- The probability of rolling four consecutive numbers in any order -/
def ProbabilityConsecutive : ℚ :=
  (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

/-- The main theorem: the probability is 1/18 -/
theorem probability_consecutive_is_one_eighteenth :
  ProbabilityConsecutive = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_probability_consecutive_is_one_eighteenth_l1698_169875


namespace NUMINAMATH_CALUDE_simplify_fraction_l1698_169865

theorem simplify_fraction : (98 : ℚ) / 210 = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1698_169865


namespace NUMINAMATH_CALUDE_abie_chips_bought_l1698_169808

theorem abie_chips_bought (initial_bags : ℕ) (given_away : ℕ) (final_bags : ℕ) 
  (h1 : initial_bags = 20)
  (h2 : given_away = 4)
  (h3 : final_bags = 22) :
  final_bags - (initial_bags - given_away) = 6 :=
by sorry

end NUMINAMATH_CALUDE_abie_chips_bought_l1698_169808


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1698_169872

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1698_169872


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l1698_169811

/-- The difference between the area of a circle with diameter 10 inches
    and the area of a square with diagonal 10 inches is approximately 28.5 square inches. -/
theorem circle_square_area_difference :
  let square_diagonal : ℝ := 10
  let circle_diameter : ℝ := 10
  let square_area : ℝ := (square_diagonal ^ 2) / 2
  let circle_area : ℝ := π * ((circle_diameter / 2) ^ 2)
  ∃ ε > 0, ε < 0.1 ∧ |circle_area - square_area - 28.5| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_circle_square_area_difference_l1698_169811


namespace NUMINAMATH_CALUDE_mike_red_notebooks_l1698_169895

/-- Represents the number of red notebooks Mike bought -/
def red_notebooks : ℕ := sorry

/-- Represents the number of blue notebooks Mike bought -/
def blue_notebooks : ℕ := sorry

/-- The total cost of all notebooks -/
def total_cost : ℕ := 37

/-- The total number of notebooks -/
def total_notebooks : ℕ := 12

/-- The cost of each red notebook -/
def red_cost : ℕ := 4

/-- The number of green notebooks -/
def green_notebooks : ℕ := 2

/-- The cost of each green notebook -/
def green_cost : ℕ := 2

/-- The cost of each blue notebook -/
def blue_cost : ℕ := 3

theorem mike_red_notebooks : 
  red_notebooks = 3 ∧
  red_notebooks + green_notebooks + blue_notebooks = total_notebooks ∧
  red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks * blue_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_mike_red_notebooks_l1698_169895


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l1698_169839

/-- Theorem: Tripling the radius and doubling the height of a cylinder increases its volume by a factor of 18. -/
theorem cylinder_volume_change (r h V : ℝ) (hV : V = π * r^2 * h) :
  π * (3*r)^2 * (2*h) = 18 * V := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l1698_169839


namespace NUMINAMATH_CALUDE_smallest_square_with_rook_l1698_169882

/-- Represents a chessboard with rooks -/
structure ChessBoard (n : ℕ) where
  size : ℕ := 3 * n
  rooks : Set (ℕ × ℕ)
  beats_entire_board : ∀ (x y : ℕ), x ≤ size ∧ y ≤ size → 
    ∃ (rx ry : ℕ), (rx, ry) ∈ rooks ∧ (rx = x ∨ ry = y)
  beats_at_most_one : ∀ (r1 r2 : ℕ × ℕ), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 →
    (r1.1 = r2.1 ∧ r1.2 ≠ r2.2) ∨ (r1.1 ≠ r2.1 ∧ r1.2 = r2.2)

/-- The main theorem to be proved -/
theorem smallest_square_with_rook (n : ℕ) (h : n > 0) (board : ChessBoard n) :
  (∀ (k : ℕ), k > 2 * n → 
    ∀ (x y : ℕ), x ≤ board.size - k + 1 → y ≤ board.size - k + 1 →
      ∃ (rx ry : ℕ), (rx, ry) ∈ board.rooks ∧ rx ≥ x ∧ rx < x + k ∧ ry ≥ y ∧ ry < y + k) ∧
  (∃ (x y : ℕ), x ≤ board.size - 2 * n + 1 ∧ y ≤ board.size - 2 * n + 1 ∧
    ∀ (rx ry : ℕ), (rx, ry) ∈ board.rooks → (rx < x ∨ rx ≥ x + 2 * n ∨ ry < y ∨ ry ≥ y + 2 * n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_with_rook_l1698_169882


namespace NUMINAMATH_CALUDE_difference_of_squares_l1698_169807

theorem difference_of_squares : 55^2 - 45^2 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1698_169807


namespace NUMINAMATH_CALUDE_min_cards_to_form_square_l1698_169817

/-- Represents the width of the rectangular card in centimeters -/
def card_width : ℕ := 20

/-- Represents the length of the rectangular card in centimeters -/
def card_length : ℕ := 8

/-- Represents the area of a single card in square centimeters -/
def card_area : ℕ := card_width * card_length

/-- Represents the side length of the smallest square that can be formed -/
def square_side : ℕ := Nat.lcm card_width card_length

/-- Represents the area of the smallest square that can be formed -/
def square_area : ℕ := square_side * square_side

/-- The minimum number of cards needed to form the smallest square -/
def min_cards : ℕ := square_area / card_area

theorem min_cards_to_form_square : min_cards = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_cards_to_form_square_l1698_169817


namespace NUMINAMATH_CALUDE_special_sequence_properties_l1698_169890

/-- A sequence of 2000 positive integers satisfying the given conditions -/
def special_sequence : Fin 2000 → ℕ
  | ⟨i, _⟩ => 2^(2000 + i) * 3^(3999 - i)

theorem special_sequence_properties :
  ∃ (seq : Fin 2000 → ℕ),
    (∀ i j, i ≠ j → ¬(seq i ∣ seq j)) ∧
    (∀ i j, i ≠ j → (seq i)^2 ∣ seq j) :=
by
  use special_sequence
  sorry

end NUMINAMATH_CALUDE_special_sequence_properties_l1698_169890


namespace NUMINAMATH_CALUDE_power_function_value_l1698_169899

-- Define a power function that passes through (2, 8)
def f : ℝ → ℝ := fun x ↦ x^3

-- Theorem statement
theorem power_function_value : f 2 = 8 ∧ f (-3) = -27 := by
  sorry


end NUMINAMATH_CALUDE_power_function_value_l1698_169899


namespace NUMINAMATH_CALUDE_no_valid_y_exists_l1698_169891

theorem no_valid_y_exists : ¬∃ (y : ℝ), y^3 + y - 2 = 0 ∧ abs y < 1 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_y_exists_l1698_169891


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1698_169860

theorem hyperbola_equation (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  ∃ (x y : ℝ), (x^2 / 25 - y^2 / 24 = 1) ∨ (y^2 / 25 - x^2 / 24 = 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1698_169860


namespace NUMINAMATH_CALUDE_least_subtraction_l1698_169809

theorem least_subtraction (n : Nat) (a b c : Nat) (r : Nat) : 
  (∀ m : Nat, m < n → 
    ((2590 - m) % a ≠ r ∨ (2590 - m) % b ≠ r ∨ (2590 - m) % c ≠ r)) →
  (2590 - n) % a = r ∧ (2590 - n) % b = r ∧ (2590 - n) % c = r →
  n = 16 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l1698_169809


namespace NUMINAMATH_CALUDE_circles_intersect_iff_l1698_169889

/-- Two circles intersect if and only if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
theorem circles_intersect_iff (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_circles_intersect_iff_l1698_169889


namespace NUMINAMATH_CALUDE_dividend_calculation_l1698_169816

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 19)
  (h3 : remainder = 2) : 
  divisor * quotient + remainder = 686 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1698_169816


namespace NUMINAMATH_CALUDE_sam_remaining_money_l1698_169884

/-- Calculates the remaining money in cents after Sam's purchases -/
def remaining_money (initial_dimes : Nat) (initial_quarters : Nat) 
  (candy_bars : Nat) (candy_bar_cost : Nat) (lollipop_cost : Nat) : Nat :=
  let remaining_dimes := initial_dimes - candy_bars * candy_bar_cost
  let remaining_quarters := initial_quarters - 1
  remaining_dimes * 10 + remaining_quarters * 25

/-- Proves that Sam has 195 cents left after her purchases -/
theorem sam_remaining_money :
  remaining_money 19 6 4 3 1 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sam_remaining_money_l1698_169884


namespace NUMINAMATH_CALUDE_vacuum_time_difference_l1698_169838

/-- Given vacuuming times, proves the difference between upstairs time and twice downstairs time -/
theorem vacuum_time_difference (total_time upstairs_time downstairs_time : ℕ) 
  (h1 : total_time = 38)
  (h2 : upstairs_time = 27)
  (h3 : total_time = upstairs_time + downstairs_time)
  (h4 : upstairs_time > 2 * downstairs_time) :
  upstairs_time - 2 * downstairs_time = 5 := by
  sorry


end NUMINAMATH_CALUDE_vacuum_time_difference_l1698_169838


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1698_169805

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 3| = |x + 2| :=
by
  -- The unique solution is x = 1/2
  use 1/2
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1698_169805


namespace NUMINAMATH_CALUDE_joe_weight_lifting_ratio_l1698_169896

/-- Joe's weight-lifting competition problem -/
theorem joe_weight_lifting_ratio :
  ∀ (total first second : ℕ),
  total = first + second →
  first = 600 →
  total = 1500 →
  first = 2 * (second - 300) →
  first = second :=
λ total first second h1 h2 h3 h4 =>
  sorry

end NUMINAMATH_CALUDE_joe_weight_lifting_ratio_l1698_169896


namespace NUMINAMATH_CALUDE_C_power_50_l1698_169815

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -8, -5]

theorem C_power_50 : C^50 = !![(-199 : ℤ), -100; 400, 199] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l1698_169815


namespace NUMINAMATH_CALUDE_F_inequalities_l1698_169803

/-- A function is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is monotonically increasing on [0,+∞) if for any x ≥ y ≥ 0, f(x) ≥ f(y) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ y → y ≤ x → f y ≤ f x

/-- Definition of the function F -/
def F (f g : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + g (1 - x) - |f x - g (1 - x)|

theorem F_inequalities (f g : ℝ → ℝ) (a : ℝ)
    (hf_even : EvenFunction f) (hg_even : EvenFunction g)
    (hf_mono : MonoIncreasing f) (hg_mono : MonoIncreasing g)
    (ha : a > 0) :
    F f g (-a) ≥ F f g a ∧ F f g (1 + a) ≥ F f g (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_F_inequalities_l1698_169803


namespace NUMINAMATH_CALUDE_polynomial_equivalence_l1698_169861

theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0 ↔ x^2*(y^2 + 2*y - 5) = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equivalence_l1698_169861


namespace NUMINAMATH_CALUDE_difference_expression_correct_l1698_169855

/-- The expression that represents "the difference between the opposite of a and 5 times b" -/
def difference_expression (a b : ℝ) : ℝ := -a - 5*b

/-- The difference_expression correctly represents "the difference between the opposite of a and 5 times b" -/
theorem difference_expression_correct (a b : ℝ) :
  difference_expression a b = (-a) - (5*b) := by sorry

end NUMINAMATH_CALUDE_difference_expression_correct_l1698_169855


namespace NUMINAMATH_CALUDE_vector_properties_l1698_169835

/-- Given vectors in ℝ², prove properties about their relationships -/
theorem vector_properties (a b c : ℝ × ℝ) (k : ℝ) :
  a = (1, 2) →
  b = (-3, 2) →
  ‖c‖ = 2 * Real.sqrt 5 →
  ∃ (t : ℝ), c = t • a →
  (∃ (t : ℝ), (k • a + 2 • b) = t • (2 • a - 4 • b) → k = -1) ∧
  ((k • a + 2 • b) • (2 • a - 4 • b) = 0 → k = 50/3) ∧
  (c = (2, 4) ∨ c = (-2, -4)) := by
sorry

end NUMINAMATH_CALUDE_vector_properties_l1698_169835


namespace NUMINAMATH_CALUDE_equation_solution_l1698_169876

theorem equation_solution :
  ∃ x : ℚ, (x - 60) / 3 = (5 - 3 * x) / 4 ∧ x = 255 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1698_169876


namespace NUMINAMATH_CALUDE_scientific_notation_of_55000000_l1698_169888

theorem scientific_notation_of_55000000 :
  55000000 = 5.5 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_55000000_l1698_169888


namespace NUMINAMATH_CALUDE_sum_of_outer_arcs_equals_180_degrees_l1698_169847

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an arc on a circle -/
structure Arc where
  circle : Circle
  startAngle : ℝ
  endAngle : ℝ

/-- Three circles arranged in a triangle formation -/
structure TriangleFormation where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  equalRadius : c1.radius = c2.radius ∧ c2.radius = c3.radius
  triangleArrangement : Bool  -- Represents that the circles are arranged in a triangle

/-- The theorem to be proved -/
theorem sum_of_outer_arcs_equals_180_degrees 
  (formation : TriangleFormation) 
  (arc1 : Arc) 
  (arc2 : Arc) 
  (arc3 : Arc) 
  (h1 : arc1.circle = formation.c1) 
  (h2 : arc2.circle = formation.c2) 
  (h3 : arc3.circle = formation.c3) 
  (h4 : arc1.startAngle < arc1.endAngle) 
  (h5 : arc2.startAngle < arc2.endAngle) 
  (h6 : arc3.startAngle < arc3.endAngle) :
  (arc1.endAngle - arc1.startAngle) + 
  (arc2.endAngle - arc2.startAngle) + 
  (arc3.endAngle - arc3.startAngle) = π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_outer_arcs_equals_180_degrees_l1698_169847


namespace NUMINAMATH_CALUDE_borrow_methods_eq_seven_l1698_169893

/-- The number of ways to borrow at least one book from a set of three books -/
def borrow_methods : ℕ :=
  2^3 - 1

/-- Theorem stating that the number of ways to borrow at least one book from three books is 7 -/
theorem borrow_methods_eq_seven : borrow_methods = 7 := by
  sorry

end NUMINAMATH_CALUDE_borrow_methods_eq_seven_l1698_169893


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1698_169848

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1698_169848


namespace NUMINAMATH_CALUDE_inequality_proof_l1698_169842

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1698_169842


namespace NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l1698_169828

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: There exist two consecutive natural numbers whose sums of digits are both divisible by 7 -/
theorem exist_consecutive_lucky_tickets : ∃ n : ℕ, 7 ∣ sum_of_digits n ∧ 7 ∣ sum_of_digits (n + 1) :=
sorry

end NUMINAMATH_CALUDE_exist_consecutive_lucky_tickets_l1698_169828


namespace NUMINAMATH_CALUDE_walkway_problem_l1698_169898

theorem walkway_problem (walkway_length : ℝ) (time_with_walkway : ℝ) (time_without_walkway : ℝ) :
  walkway_length = 60 →
  time_with_walkway = 30 →
  time_without_walkway = 48 →
  ∃ time_against_walkway : ℝ,
    time_against_walkway = 120 ∧
    (walkway_length / time_with_walkway - walkway_length / time_without_walkway) * time_against_walkway = walkway_length :=
by sorry

end NUMINAMATH_CALUDE_walkway_problem_l1698_169898


namespace NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l1698_169892

theorem magnitude_of_4_minus_15i :
  let z : ℂ := 4 - 15 * I
  Complex.abs z = Real.sqrt 241 := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_4_minus_15i_l1698_169892


namespace NUMINAMATH_CALUDE_unique_two_digit_number_divisible_by_eight_l1698_169804

theorem unique_two_digit_number_divisible_by_eight :
  ∃! n : ℕ, 70 < n ∧ n < 80 ∧ n % 8 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_divisible_by_eight_l1698_169804


namespace NUMINAMATH_CALUDE_product_of_fractions_l1698_169823

theorem product_of_fractions : 
  (((3^4 - 1) / (3^4 + 1)) * ((4^4 - 1) / (4^4 + 1)) * ((5^4 - 1) / (5^4 + 1)) * 
   ((6^4 - 1) / (6^4 + 1)) * ((7^4 - 1) / (7^4 + 1))) = 25 / 210 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1698_169823


namespace NUMINAMATH_CALUDE_retractable_door_unique_non_triangle_l1698_169894

/-- A design that may or may not utilize the stability of a triangle. -/
inductive Design
  | RetractableDoor
  | BicycleFrame
  | WindowFrame
  | CameraTripod

/-- Predicate indicating whether a design utilizes the stability of a triangle. -/
def utilizesTriangleStability (d : Design) : Prop :=
  match d with
  | Design.RetractableDoor => False
  | Design.BicycleFrame => True
  | Design.WindowFrame => True
  | Design.CameraTripod => True

/-- Theorem stating that only the retractable door does not utilize triangle stability. -/
theorem retractable_door_unique_non_triangle :
    ∀ (d : Design), ¬(utilizesTriangleStability d) ↔ d = Design.RetractableDoor := by
  sorry

end NUMINAMATH_CALUDE_retractable_door_unique_non_triangle_l1698_169894


namespace NUMINAMATH_CALUDE_powerlifting_bodyweight_l1698_169877

theorem powerlifting_bodyweight (initial_total : ℝ) (total_gain_percent : ℝ) (weight_gain : ℝ) (final_ratio : ℝ) :
  initial_total = 2200 →
  total_gain_percent = 15 →
  weight_gain = 8 →
  final_ratio = 10 →
  ∃ initial_weight : ℝ,
    initial_weight > 0 ∧
    (initial_total * (1 + total_gain_percent / 100)) / (initial_weight + weight_gain) = final_ratio ∧
    initial_weight = 245 := by
  sorry

#check powerlifting_bodyweight

end NUMINAMATH_CALUDE_powerlifting_bodyweight_l1698_169877


namespace NUMINAMATH_CALUDE_negative_power_product_l1698_169883

theorem negative_power_product (x : ℝ) : -x^2 * x = -x^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_power_product_l1698_169883


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l1698_169856

theorem min_value_abs_sum (x : ℝ) : 
  |x - 4| + |x + 7| + |x - 5| ≥ 4 ∧ ∃ y : ℝ, |y - 4| + |y + 7| + |y - 5| = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l1698_169856


namespace NUMINAMATH_CALUDE_expand_polynomial_l1698_169863

theorem expand_polynomial (x : ℝ) : (4 * x + 3) * (2 * x - 7) + x = 8 * x^2 - 21 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1698_169863


namespace NUMINAMATH_CALUDE_single_digit_sum_l1698_169870

theorem single_digit_sum (a b : ℕ) : 
  a ∈ Finset.range 10 ∧ a ≠ 0 ∧
  b ∈ Finset.range 10 ∧ b ≠ 0 ∧
  82 * 10 * a + 7 + 6 * b = 190 →
  a + 2 * b = 7 := by
sorry

end NUMINAMATH_CALUDE_single_digit_sum_l1698_169870


namespace NUMINAMATH_CALUDE_tina_homework_time_l1698_169825

/-- Tina's keyboard cleaning and homework problem -/
theorem tina_homework_time (initial_keys : ℕ) (cleaning_time_per_key : ℕ) 
  (remaining_keys : ℕ) (total_time : ℕ) : 
  initial_keys = 15 →
  cleaning_time_per_key = 3 →
  remaining_keys = 14 →
  total_time = 52 →
  total_time - (remaining_keys * cleaning_time_per_key) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tina_homework_time_l1698_169825


namespace NUMINAMATH_CALUDE_tan_squared_gamma_equals_tan_alpha_tan_beta_l1698_169836

theorem tan_squared_gamma_equals_tan_alpha_tan_beta 
  (α β γ : Real) 
  (h : (Real.sin γ)^2 / (Real.sin α)^2 = 1 - Real.tan (α - β) / Real.tan α) : 
  (Real.tan γ)^2 = Real.tan α * Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_tan_squared_gamma_equals_tan_alpha_tan_beta_l1698_169836


namespace NUMINAMATH_CALUDE_probability_three_same_is_one_third_l1698_169885

/-- Represents the outcome of rolling five dice -/
structure DiceRoll :=
  (pair : Fin 6)
  (different : Fin 6)
  (reroll1 : Fin 6)
  (reroll2 : Fin 6)
  (pair_count : Nat)
  (different_from_pair : pair ≠ different)
  (rerolls_different : reroll1 ≠ reroll2)

/-- The probability of getting at least three dice with the same value after rerolling -/
def probability_three_same (roll : DiceRoll) : ℚ :=
  sorry

/-- Theorem stating that the probability is 1/3 -/
theorem probability_three_same_is_one_third :
  ∀ roll : DiceRoll, probability_three_same roll = 1/3 :=
sorry

end NUMINAMATH_CALUDE_probability_three_same_is_one_third_l1698_169885


namespace NUMINAMATH_CALUDE_square_product_equals_sum_implies_zero_l1698_169854

theorem square_product_equals_sum_implies_zero (x y : ℤ) 
  (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_product_equals_sum_implies_zero_l1698_169854


namespace NUMINAMATH_CALUDE_equation_solutions_l1698_169827

theorem equation_solutions :
  (∀ x : ℝ, 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2) ∧
  (∀ x : ℝ, 4 * x^2 - 2 * Real.sqrt 3 * x - 1 = 0 ↔ 
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1698_169827


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1698_169880

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ratio
  (a : ℕ → ℝ)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : arithmetic_sequence a)
  (h_sub_sequence : ∃ k : ℝ, a 1 + k = (1/2) * a 3 ∧ (1/2) * a 3 + k = 2 * a 2) :
  (a 8 + a 9) / (a 7 + a 8) = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1698_169880


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1698_169826

-- Part 1
def solution_set_1 (x : ℝ) : Prop := x < -1 ∨ x > 5

theorem inequality_solution_1 : 
  {x : ℝ | -x^2 + 4*x + 5 < 0} = {x : ℝ | solution_set_1 x} := by sorry

-- Part 2
def solution_set_2 (a : ℝ) (x : ℝ) : Prop :=
  (a = -1 ∧ False) ∨
  (a > -1 ∧ -1 < x ∧ x < a) ∨
  (a < -1 ∧ a < x ∧ x < -1)

theorem inequality_solution_2 (a : ℝ) : 
  {x : ℝ | x^2 + (1-a)*x - a < 0} = {x : ℝ | solution_set_2 a x} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1698_169826


namespace NUMINAMATH_CALUDE_katie_sold_four_pastries_l1698_169857

/-- The number of pastries sold at a bake sale -/
def pastries_sold (cupcakes cookies leftover : ℕ) : ℕ :=
  cupcakes + cookies - leftover

/-- Proof that Katie sold 4 pastries at the bake sale -/
theorem katie_sold_four_pastries :
  pastries_sold 7 5 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_katie_sold_four_pastries_l1698_169857


namespace NUMINAMATH_CALUDE_roots_sum_powers_l1698_169852

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 6*a + 8 = 0 → 
  b^2 - 6*b + 8 = 0 → 
  a^5 + a^3*b^3 + b^5 = -568 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l1698_169852


namespace NUMINAMATH_CALUDE_fruit_box_theorem_l1698_169806

theorem fruit_box_theorem (total_fruits : ℕ) 
  (h_total : total_fruits = 56)
  (h_oranges : total_fruits / 4 = total_fruits / 4)  -- One-fourth are oranges
  (h_peaches : total_fruits / 8 = total_fruits / 8)  -- Half as many peaches as oranges
  (h_apples : 5 * (total_fruits / 8) = 5 * (total_fruits / 8))  -- Five times as many apples as peaches
  (h_mixed : total_fruits / 4 = total_fruits / 4)  -- Twice as many mixed fruits as peaches
  : (5 * (total_fruits / 8) = 35) ∧ 
    (total_fruits / 4 : ℚ) / total_fruits = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_box_theorem_l1698_169806


namespace NUMINAMATH_CALUDE_equation_relation_l1698_169801

theorem equation_relation (x y : ℝ) (h : 2 * x - y = 4) : 6 * x - 3 * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_relation_l1698_169801


namespace NUMINAMATH_CALUDE_point_shift_theorem_l1698_169886

theorem point_shift_theorem (original_coord final_coord shift : ℤ) :
  final_coord = original_coord + shift →
  final_coord = 8 →
  shift = 13 →
  original_coord = -5 := by
sorry

end NUMINAMATH_CALUDE_point_shift_theorem_l1698_169886


namespace NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_root_range_l1698_169832

def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

theorem quadratic_function_properties (q : ℝ) :
  (∃ (min : ℝ), ∀ (x : ℝ), f q x ≥ min ∧ (∃ (x_min : ℝ), f q x_min = min) ∧ min = -60) →
  q = 1 :=
sorry

theorem quadratic_function_root_range (q : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc (-1) 1 ∧ f q x = 0) →
  q ∈ Set.Icc (-20) 12 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_quadratic_function_root_range_l1698_169832


namespace NUMINAMATH_CALUDE_min_value_expression_l1698_169879

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (min : ℝ), min = 2 * Real.sqrt 3 ∧
  ∀ (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0),
    x^2 + y^2 + z^2 + 1/x^2 + y/x + z/y ≥ min ∧
    ∃ (a' b' c' : ℝ) (ha' : a' ≠ 0) (hb' : b' ≠ 0) (hc' : c' ≠ 0),
      a'^2 + b'^2 + c'^2 + 1/a'^2 + b'/a' + c'/b' = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1698_169879


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l1698_169867

theorem pentagon_area_sum (a b : ℤ) (h1 : 0 < b) (h2 : b < a) : 
  let P := (a, b)
  let Q := (b, a)
  let R := (-b, a)
  let S := (-b, -a)
  let T := (b, -a)
  let pentagon_area := a * (3 * b + a)
  pentagon_area = 792 → a + b = 45 := by
sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l1698_169867


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1698_169859

theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 = 6*x - 10*y + 24 →
  ∃ (center_x center_y : ℝ),
    (∀ (p_x p_y : ℝ), (p_x - center_x)^2 + (p_y - center_y)^2 = (x - center_x)^2 + (y - center_y)^2) ∧
    center_x + center_y = -2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1698_169859


namespace NUMINAMATH_CALUDE_apples_added_l1698_169869

theorem apples_added (initial_apples final_apples : ℕ) 
  (h1 : initial_apples = 8)
  (h2 : final_apples = 13) :
  final_apples - initial_apples = 5 := by
  sorry

end NUMINAMATH_CALUDE_apples_added_l1698_169869


namespace NUMINAMATH_CALUDE_shopping_expenditure_theorem_l1698_169858

theorem shopping_expenditure_theorem (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 ∧
  x / 100 + 0.3 + 0.3 = 1 ∧
  0.04 * (x / 100) + 0.08 * 0.3 = 0.04 →
  x = 40 := by sorry

end NUMINAMATH_CALUDE_shopping_expenditure_theorem_l1698_169858


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1698_169818

theorem solution_set_of_equation (x : ℝ) : 
  {x | 3 * x - 4 = 2} = {2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1698_169818


namespace NUMINAMATH_CALUDE_die_roll_probability_l1698_169829

theorem die_roll_probability (p_greater_than_four : ℚ) 
  (h : p_greater_than_four = 1/3) : 
  1 - p_greater_than_four = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1698_169829


namespace NUMINAMATH_CALUDE_transformed_graph_equivalence_l1698_169887

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Define the horizontal shift transformation
noncomputable def shift (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the horizontal compression transformation
noncomputable def compress (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (k * x)

-- Theorem statement
theorem transformed_graph_equivalence :
  ∀ x : ℝ, g x = (compress (1/2) (shift (-1/2) f)) x := by sorry

end NUMINAMATH_CALUDE_transformed_graph_equivalence_l1698_169887


namespace NUMINAMATH_CALUDE_no_geometric_subsequence_in_arithmetic_sequence_l1698_169810

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n + r

def contains_one_and_sqrt_two (a : ℕ → ℝ) : Prop :=
  ∃ k l : ℕ, a k = 1 ∧ a l = Real.sqrt 2

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem no_geometric_subsequence_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_contains : contains_one_and_sqrt_two a) :
  ¬ ∃ m n p : ℕ, m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ is_geometric_sequence (a m) (a n) (a p) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_subsequence_in_arithmetic_sequence_l1698_169810


namespace NUMINAMATH_CALUDE_expression_evaluation_l1698_169837

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (x + 3) * (x - 3) - x * (x - 2) = 2 * Real.sqrt 2 - 11 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1698_169837


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l1698_169819

theorem sum_of_roots_eq_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l1698_169819


namespace NUMINAMATH_CALUDE_friend_savings_rate_l1698_169878

/-- Proves that given the initial amounts and saving rates, after 25 weeks,
    both people will have the same amount of money if and only if the friend saves 5 dollars per week. -/
theorem friend_savings_rate (your_initial : ℕ) (friend_initial : ℕ) (your_weekly_savings : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  friend_initial = 210 →
  your_weekly_savings = 7 →
  weeks = 25 →
  (your_initial + your_weekly_savings * weeks = friend_initial + 5 * weeks) :=
by sorry

end NUMINAMATH_CALUDE_friend_savings_rate_l1698_169878


namespace NUMINAMATH_CALUDE_triangle_height_l1698_169821

/-- Given a triangle with base 4 meters and area 10 square meters, prove its height is 5 meters. -/
theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 4 → area = 10 → area = (base * height) / 2 → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1698_169821


namespace NUMINAMATH_CALUDE_sin_2x_plus_1_equals_shifted_cos_l1698_169849

theorem sin_2x_plus_1_equals_shifted_cos (x : ℝ) : 
  Real.sin (2 * x) + 1 = Real.cos (2 * (x - π / 4)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_plus_1_equals_shifted_cos_l1698_169849


namespace NUMINAMATH_CALUDE_expansion_properties_l1698_169845

/-- Represents the coefficient of x^(k/3) in the expansion of (∛x - 3/∛x)^n -/
def coeff (n : ℕ) (k : ℤ) : ℚ :=
  sorry

/-- The sixth term in the expansion -/
def sixth_term (n : ℕ) : ℚ := coeff n (n - 10)

/-- The coefficient of x² in the expansion -/
def x_squared_coeff (n : ℕ) : ℚ := coeff n 6

theorem expansion_properties (n : ℕ) :
  sixth_term n = 0 →
  n = 10 ∧ x_squared_coeff 10 = 405 := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l1698_169845


namespace NUMINAMATH_CALUDE_terry_age_l1698_169840

/-- Given the following conditions:
    1. In 10 years, Terry will be 4 times Nora's current age.
    2. Nora is currently 10 years old.
    3. In 5 years, Nora will be half Sam's age.
    4. Sam is currently 6 years older than Terry.
    Prove that Terry is currently 19 years old. -/
theorem terry_age (nora_age : ℕ) (terry_future_age : ℕ → ℕ) (sam_age : ℕ → ℕ) :
  nora_age = 10 ∧
  terry_future_age 10 = 4 * nora_age ∧
  sam_age 5 = 2 * (nora_age + 5) ∧
  sam_age 0 = terry_future_age 0 + 6 →
  terry_future_age 0 = 19 := by
  sorry

end NUMINAMATH_CALUDE_terry_age_l1698_169840


namespace NUMINAMATH_CALUDE_existence_of_special_point_l1698_169897

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

/-- Checks if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  sorry -- Definition of line-circle intersection

/-- Main theorem -/
theorem existence_of_special_point (c1 c2 : Circle) :
  ∃ p : Point, (isOutside p c1 ∧ isOutside p c2) ∧
    ∀ l : Line, l.point = p → (intersects l c1 ∨ intersects l c2) :=
  sorry

end NUMINAMATH_CALUDE_existence_of_special_point_l1698_169897


namespace NUMINAMATH_CALUDE_eggs_left_is_five_l1698_169866

-- Define the problem parameters
def total_eggs : ℕ := 30
def total_cost : ℕ := 500  -- in cents
def price_per_egg : ℕ := 20  -- in cents

-- Define the function to calculate eggs left after recovering capital
def eggs_left_after_recovery : ℕ :=
  total_eggs - (total_cost / price_per_egg)

-- Theorem statement
theorem eggs_left_is_five : eggs_left_after_recovery = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_is_five_l1698_169866


namespace NUMINAMATH_CALUDE_mosquito_blood_consumption_proof_l1698_169814

/-- The number of drops of blood per liter -/
def drops_per_liter : ℕ := 5000

/-- The number of liters of blood loss that leads to death -/
def lethal_blood_loss : ℕ := 3

/-- The number of mosquitoes that would cause death by feeding -/
def lethal_mosquito_count : ℕ := 750

/-- The number of drops of blood a single mosquito sucks in one feeding -/
def mosquito_blood_consumption : ℕ := 20

theorem mosquito_blood_consumption_proof :
  mosquito_blood_consumption = (drops_per_liter * lethal_blood_loss) / lethal_mosquito_count :=
by sorry

end NUMINAMATH_CALUDE_mosquito_blood_consumption_proof_l1698_169814


namespace NUMINAMATH_CALUDE_sum_of_pentagram_angles_l1698_169831

/-- A self-intersecting five-pointed star (pentagram) -/
structure Pentagram where
  vertices : Fin 5 → Point2
  is_self_intersecting : Bool

/-- The sum of angles at the vertices of a pentagram -/
def sum_of_vertex_angles (p : Pentagram) : ℝ := sorry

/-- Theorem: The sum of angles at the vertices of a self-intersecting pentagram is 180° -/
theorem sum_of_pentagram_angles (p : Pentagram) (h : p.is_self_intersecting = true) :
  sum_of_vertex_angles p = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_pentagram_angles_l1698_169831


namespace NUMINAMATH_CALUDE_equation_solution_l1698_169834

theorem equation_solution : 
  ∃! x : ℚ, (x - 17) / 3 = (3 * x + 8) / 6 :=
by
  use (-42 : ℚ)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1698_169834


namespace NUMINAMATH_CALUDE_colorcrafter_secret_codes_l1698_169802

/-- The number of available colors in the ColorCrafter game -/
def num_colors : ℕ := 8

/-- The number of slots to fill in the ColorCrafter game -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the ColorCrafter game -/
def num_secret_codes : ℕ := num_colors ^ num_slots

theorem colorcrafter_secret_codes :
  num_secret_codes = 32768 :=
by sorry

end NUMINAMATH_CALUDE_colorcrafter_secret_codes_l1698_169802


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1698_169800

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a + 1) * x + a * b > 0 ↔ x < -1 ∨ x > 4) →
  a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1698_169800


namespace NUMINAMATH_CALUDE_x_is_twenty_percent_greater_than_80_l1698_169850

/-- If x is 20 percent greater than 80, then x equals 96. -/
theorem x_is_twenty_percent_greater_than_80 : ∀ x : ℝ, x = 80 * (1 + 20 / 100) → x = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_x_is_twenty_percent_greater_than_80_l1698_169850


namespace NUMINAMATH_CALUDE_line_parameterization_l1698_169862

/-- Given a line y = 2x + 5 parameterized as (x, y) = (s, -2) + t(3, m), prove that s = -7/2 and m = 6 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ t : ℝ, ∀ x y : ℝ, x = s + 3*t ∧ y = -2 + m*t → y = 2*x + 5) →
  s = -7/2 ∧ m = 6 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1698_169862


namespace NUMINAMATH_CALUDE_milk_water_ratio_change_l1698_169844

/-- Proves that adding 60 litres of water to a 60-litre mixture with initial milk to water ratio of 2:1 results in a new ratio of 1:2 -/
theorem milk_water_ratio_change (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) :
  initial_volume = 60 →
  initial_milk_ratio = 2 →
  initial_water_ratio = 1 →
  added_water = 60 →
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let new_water := initial_water + added_water
  let new_milk_ratio := initial_milk / new_water
  let new_water_ratio := new_water / new_water
  new_milk_ratio = 1 ∧ new_water_ratio = 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_change_l1698_169844


namespace NUMINAMATH_CALUDE_unique_f_3_l1698_169812

/-- A function satisfying the given functional equation and initial condition -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * (f x) * y) ∧ f 1 = 1

/-- The theorem stating that for any function satisfying the conditions, f(3) must equal 9 -/
theorem unique_f_3 (f : ℝ → ℝ) (hf : special_function f) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_3_l1698_169812


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1698_169833

theorem expand_and_simplify (x : ℝ) : (7 * x - 3) * 3 * x^2 = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1698_169833


namespace NUMINAMATH_CALUDE_binomial_n_minus_two_l1698_169851

theorem binomial_n_minus_two (n : ℕ) (h : n ≥ 2) : 
  (n.choose (n - 2)) = n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_n_minus_two_l1698_169851


namespace NUMINAMATH_CALUDE_ellipse_sum_specific_l1698_169824

/-- Represents an ellipse with center (h, k) and semi-axes lengths a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

theorem ellipse_sum_specific : ∃ (e : Ellipse), 
  e.h = 3 ∧ 
  e.k = -1 ∧ 
  e.a = 6 ∧ 
  e.b = 4 ∧ 
  ellipse_sum e = 12 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_specific_l1698_169824


namespace NUMINAMATH_CALUDE_existence_of_a_and_b_l1698_169873

theorem existence_of_a_and_b : ∃ (a b : ℝ), a = b + 1 ∧ a^4 = b^4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_and_b_l1698_169873


namespace NUMINAMATH_CALUDE_total_distance_traveled_l1698_169846

theorem total_distance_traveled (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 2) (h2 : v2 = 6) (h3 : v3 = 6) (h4 : t = 11 / 60) :
  let d := t * (v1⁻¹ + v2⁻¹ + v3⁻¹)⁻¹
  3 * d = 33 / 50 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l1698_169846


namespace NUMINAMATH_CALUDE_fair_coin_prob_heads_l1698_169881

/-- A fair coin is a coin where the probability of getting heads is equal to the probability of getting tails -/
def is_fair_coin (coin : Type) (prob_heads : coin → ℝ) : Prop :=
  ∀ c : coin, prob_heads c = 1 - prob_heads c

/-- The probability of an event is independent of previous events if the probability remains constant regardless of previous outcomes -/
def is_independent_event {α : Type} (prob : α → ℝ) : Prop :=
  ∀ (a b : α), prob a = prob b

/-- Theorem: For a fair coin, the probability of getting heads on any single toss is 1/2, regardless of previous tosses -/
theorem fair_coin_prob_heads {coin : Type} (prob_heads : coin → ℝ) 
  (h_fair : is_fair_coin coin prob_heads) 
  (h_indep : is_independent_event prob_heads) :
  ∀ c : coin, prob_heads c = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_fair_coin_prob_heads_l1698_169881


namespace NUMINAMATH_CALUDE_family_juice_consumption_l1698_169830

/-- The amount of juice consumed by a family in a week -/
def juice_consumption_per_week (juice_per_serving : ℝ) (servings_per_day : ℕ) (days_per_week : ℕ) : ℝ :=
  juice_per_serving * (servings_per_day : ℝ) * (days_per_week : ℝ)

/-- Theorem stating that a family drinking 0.2 liters of juice three times a day consumes 4.2 liters in a week -/
theorem family_juice_consumption :
  juice_consumption_per_week 0.2 3 7 = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_family_juice_consumption_l1698_169830


namespace NUMINAMATH_CALUDE_guppies_needed_per_day_l1698_169874

/-- The number of guppies eaten by a moray eel per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish -/
def num_betta_fish : ℕ := 5

/-- The number of guppies eaten by each betta fish per day -/
def betta_fish_guppies : ℕ := 7

/-- The total number of guppies needed per day -/
def total_guppies : ℕ := moray_eel_guppies + num_betta_fish * betta_fish_guppies

theorem guppies_needed_per_day : total_guppies = 55 := by
  sorry

end NUMINAMATH_CALUDE_guppies_needed_per_day_l1698_169874


namespace NUMINAMATH_CALUDE_scheduling_methods_count_l1698_169864

/-- Represents the number of days in the schedule -/
def num_days : ℕ := 7

/-- Represents the number of volunteers -/
def num_volunteers : ℕ := 4

/-- Calculates the number of scheduling methods -/
def scheduling_methods : ℕ :=
  -- This function should implement the logic to calculate the number of scheduling methods
  -- based on the given conditions
  sorry

/-- Theorem stating that the number of scheduling methods is 420 -/
theorem scheduling_methods_count : scheduling_methods = 420 := by
  sorry

end NUMINAMATH_CALUDE_scheduling_methods_count_l1698_169864


namespace NUMINAMATH_CALUDE_routes_in_grid_l1698_169813

/-- The number of routes in a 3x3 grid from top-left to bottom-right -/
def num_routes : ℕ := Nat.choose 6 3

/-- The dimensions of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required -/
def total_moves : ℕ := 2 * grid_size

/-- The number of moves in each direction -/
def moves_per_direction : ℕ := grid_size

theorem routes_in_grid :
  num_routes = Nat.choose total_moves moves_per_direction :=
sorry

end NUMINAMATH_CALUDE_routes_in_grid_l1698_169813


namespace NUMINAMATH_CALUDE_sum_of_jenna_and_darius_ages_l1698_169843

def sum_of_ages (jenna_age : ℕ) (darius_age : ℕ) : ℕ :=
  jenna_age + darius_age

theorem sum_of_jenna_and_darius_ages :
  ∀ (jenna_age : ℕ) (darius_age : ℕ),
    jenna_age = darius_age + 5 →
    jenna_age = 13 →
    darius_age = 8 →
    sum_of_ages jenna_age darius_age = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_jenna_and_darius_ages_l1698_169843


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l1698_169868

theorem quadratic_roots_product (b c : ℝ) :
  (∀ x, x^2 + b*x + c = 0 → ∃ y, y^2 + b*y + c = 0 ∧ x * y = 20) →
  c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l1698_169868


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l1698_169822

theorem multiplication_addition_equality : 3.6 * 0.5 + 1.2 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l1698_169822


namespace NUMINAMATH_CALUDE_ferry_hat_count_l1698_169820

theorem ferry_hat_count : ∀ (total_adults : ℕ) (children : ℕ) 
  (women_hat_percent : ℚ) (men_hat_percent : ℚ) (children_hat_percent : ℚ),
  total_adults = 3000 →
  children = 500 →
  women_hat_percent = 25 / 100 →
  men_hat_percent = 15 / 100 →
  children_hat_percent = 30 / 100 →
  ∃ (total_with_hats : ℕ),
    total_with_hats = 
      (total_adults / 2 * women_hat_percent).floor +
      (total_adults / 2 * men_hat_percent).floor +
      (children * children_hat_percent).floor ∧
    total_with_hats = 750 := by
  sorry

end NUMINAMATH_CALUDE_ferry_hat_count_l1698_169820


namespace NUMINAMATH_CALUDE_thirty_two_team_tournament_games_l1698_169841

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_teams : ℕ

/-- The number of games played in a single-elimination tournament -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.num_teams - 1

theorem thirty_two_team_tournament_games :
  ∀ (t : SingleEliminationTournament),
    t.num_teams = 32 →
    games_played t = 31 := by
  sorry

end NUMINAMATH_CALUDE_thirty_two_team_tournament_games_l1698_169841


namespace NUMINAMATH_CALUDE_festival_guests_selection_l1698_169853

theorem festival_guests_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 5) : 
  Nat.choose n m = 252 := by
  sorry

end NUMINAMATH_CALUDE_festival_guests_selection_l1698_169853
