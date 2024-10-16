import Mathlib

namespace NUMINAMATH_CALUDE_at_operation_four_three_l757_75780

def at_operation (a b : ℝ) : ℝ := 4 * a^2 - 2 * b

theorem at_operation_four_three : at_operation 4 3 = 58 := by
  sorry

end NUMINAMATH_CALUDE_at_operation_four_three_l757_75780


namespace NUMINAMATH_CALUDE_least_N_for_P_condition_l757_75796

/-- The probability that at least 3/5 of N green balls are on the same side of a red ball
    when arranged randomly in a line -/
def P (N : ℕ) : ℚ :=
  (⌊(2 * N : ℚ) / 5⌋ + 1 + (N - ⌈(3 * N : ℚ) / 5⌉ + 1)) / (N + 1)

theorem least_N_for_P_condition :
  ∀ N : ℕ, N % 5 = 0 → N > 0 →
    (∀ k : ℕ, k % 5 = 0 → k > 0 → k < N → P k ≥ 321/400) ∧
    P N < 321/400 →
    N = 480 :=
sorry

end NUMINAMATH_CALUDE_least_N_for_P_condition_l757_75796


namespace NUMINAMATH_CALUDE_remainder_when_c_divided_by_b_l757_75720

theorem remainder_when_c_divided_by_b (a b c : ℕ) 
  (h1 : b = 3 * a + 3) 
  (h2 : c = 9 * a + 11) : 
  c % b = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_when_c_divided_by_b_l757_75720


namespace NUMINAMATH_CALUDE_probability_theorem_l757_75723

def num_events : ℕ := 5
def prob_success : ℚ := 3/4

theorem probability_theorem :
  (prob_success ^ num_events = 243/1024) ∧
  (1 - (1 - prob_success) ^ num_events = 1023/1024) :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l757_75723


namespace NUMINAMATH_CALUDE_symmetry_lines_sum_l757_75794

/-- Two parabolas intersecting at two points -/
structure IntersectingParabolas where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h1 : -(3 - a)^2 + b = 6
  h2 : (3 - c)^2 + d = 6
  h3 : -(9 - a)^2 + b = 0
  h4 : (9 - c)^2 + d = 0

/-- The sum of x-axis symmetry lines of two intersecting parabolas equals 12 -/
theorem symmetry_lines_sum (p : IntersectingParabolas) : p.a + p.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_lines_sum_l757_75794


namespace NUMINAMATH_CALUDE_plot_length_is_60_l757_75714

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ

/-- Calculates the perimeter of a rectangular plot. -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth)

/-- Theorem stating the length of the plot given the conditions. -/
theorem plot_length_is_60 (plot : RectangularPlot)
  (h1 : plot.length = plot.breadth + 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300)
  (h4 : plot.totalFencingCost = plot.fencingCostPerMeter * perimeter plot) :
  plot.length = 60 := by
  sorry

#check plot_length_is_60

end NUMINAMATH_CALUDE_plot_length_is_60_l757_75714


namespace NUMINAMATH_CALUDE_fraction_sum_bounds_l757_75781

theorem fraction_sum_bounds (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_log_sum : Real.log a + Real.log b + Real.log c = 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 := by
  sorry


end NUMINAMATH_CALUDE_fraction_sum_bounds_l757_75781


namespace NUMINAMATH_CALUDE_journey_distance_l757_75734

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  (total_time * speed1 * speed2) / (speed1 + speed2) = 224 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l757_75734


namespace NUMINAMATH_CALUDE_slope_of_line_l757_75772

-- Define a line with equation y = 3x + 1
def line (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem: the slope of this line is 3
theorem slope_of_line :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (line x₂ - line x₁) / (x₂ - x₁) = 3) := by
sorry

end NUMINAMATH_CALUDE_slope_of_line_l757_75772


namespace NUMINAMATH_CALUDE_cost_per_flower_is_15_l757_75757

/-- Represents the number of centerpieces -/
def num_centerpieces : ℕ := 6

/-- Represents the number of roses per centerpiece -/
def roses_per_centerpiece : ℕ := 8

/-- Represents the number of lilies per centerpiece -/
def lilies_per_centerpiece : ℕ := 6

/-- Represents the total budget in dollars -/
def total_budget : ℕ := 2700

/-- Calculates the total number of roses -/
def total_roses : ℕ := num_centerpieces * roses_per_centerpiece

/-- Calculates the total number of orchids -/
def total_orchids : ℕ := 2 * total_roses

/-- Calculates the total number of lilies -/
def total_lilies : ℕ := num_centerpieces * lilies_per_centerpiece

/-- Calculates the total number of flowers -/
def total_flowers : ℕ := total_roses + total_orchids + total_lilies

/-- Theorem: The cost per flower is $15 -/
theorem cost_per_flower_is_15 : total_budget / total_flowers = 15 := by
  sorry


end NUMINAMATH_CALUDE_cost_per_flower_is_15_l757_75757


namespace NUMINAMATH_CALUDE_sum_of_digits_n_l757_75798

/-- The least 6-digit number that leaves a remainder of 2 when divided by 4, 610, and 15 -/
def n : ℕ := 102482

/-- Condition: n is at least 100000 (6-digit number) -/
axiom n_six_digits : n ≥ 100000

/-- Condition: n leaves remainder 2 when divided by 4 -/
axiom n_mod_4 : n % 4 = 2

/-- Condition: n leaves remainder 2 when divided by 610 -/
axiom n_mod_610 : n % 610 = 2

/-- Condition: n leaves remainder 2 when divided by 15 -/
axiom n_mod_15 : n % 15 = 2

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else m % 10 + sum_of_digits (m / 10)

/-- Theorem: The sum of digits of n is 17 -/
theorem sum_of_digits_n : sum_of_digits n = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_l757_75798


namespace NUMINAMATH_CALUDE_product_sum_inequality_l757_75737

theorem product_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l757_75737


namespace NUMINAMATH_CALUDE_charity_arrangements_l757_75753

/-- The number of people selected from the class -/
def total_people : ℕ := 6

/-- The maximum number of people that can participate in each activity -/
def max_per_activity : ℕ := 4

/-- The number of charity activities -/
def num_activities : ℕ := 2

/-- The function to calculate the number of different arrangements -/
def num_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ := sorry

theorem charity_arrangements :
  num_arrangements total_people max_per_activity num_activities = 50 := by sorry

end NUMINAMATH_CALUDE_charity_arrangements_l757_75753


namespace NUMINAMATH_CALUDE_upper_limit_is_1575_l757_75765

def upper_limit (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 1 ∧
  (∃ (S : Finset ℕ), S.card = 8 ∧
    (∀ x ∈ S, x ≥ 1 ∧ x ≤ m ∧ 25 ∣ x ∧ 35 ∣ x) ∧
    (∀ y : ℕ, y ≥ 1 ∧ y ≤ m ∧ 25 ∣ y ∧ 35 ∣ y → y ∈ S)) ∧
  (∀ k < m, ¬∃ (T : Finset ℕ), T.card = 8 ∧
    (∀ x ∈ T, x ≥ 1 ∧ x ≤ k ∧ 25 ∣ x ∧ 35 ∣ x) ∧
    (∀ y : ℕ, y ≥ 1 ∧ y ≤ k ∧ 25 ∣ y ∧ 35 ∣ y → y ∈ T))

theorem upper_limit_is_1575 : upper_limit 1575 :=
sorry

end NUMINAMATH_CALUDE_upper_limit_is_1575_l757_75765


namespace NUMINAMATH_CALUDE_union_covers_reals_l757_75719

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem union_covers_reals (a : ℝ) : A ∪ B a = Set.univ → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l757_75719


namespace NUMINAMATH_CALUDE_x_plus_y_eq_1_is_linear_l757_75707

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The function representing x + y = 1 -/
def f (x y : ℝ) : ℝ := x + y - 1

/-- Theorem stating that x + y = 1 is a linear equation with two variables -/
theorem x_plus_y_eq_1_is_linear : IsLinearEquationTwoVars f := by
  sorry


end NUMINAMATH_CALUDE_x_plus_y_eq_1_is_linear_l757_75707


namespace NUMINAMATH_CALUDE_probability_of_no_defective_pens_l757_75799

theorem probability_of_no_defective_pens (total_pens : Nat) (defective_pens : Nat) (pens_bought : Nat) :
  total_pens = 12 →
  defective_pens = 3 →
  pens_bought = 2 →
  (1 - defective_pens / total_pens) * (1 - (defective_pens) / (total_pens - 1)) = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_no_defective_pens_l757_75799


namespace NUMINAMATH_CALUDE_oldest_child_age_l757_75777

def average_age : ℝ := 7
def younger_child1_age : ℝ := 4
def younger_child2_age : ℝ := 7

theorem oldest_child_age :
  ∃ (oldest_age : ℝ),
    (younger_child1_age + younger_child2_age + oldest_age) / 3 = average_age ∧
    oldest_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l757_75777


namespace NUMINAMATH_CALUDE_arcMTN_constant_l757_75731

/-- Represents an equilateral triangle ABC with a circle rolling along side AB -/
structure RollingCircleTriangle where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of the circle, equal to the triangle's altitude -/
  radius : ℝ
  /-- The circle's radius is equal to the triangle's altitude -/
  radius_eq_altitude : radius = side * Real.sqrt 3 / 2

/-- The measure of arc MTN in degrees -/
def arcMTN (rct : RollingCircleTriangle) : ℝ :=
  60

/-- Theorem stating that arc MTN always measures 60° -/
theorem arcMTN_constant (rct : RollingCircleTriangle) :
  arcMTN rct = 60 := by
  sorry

end NUMINAMATH_CALUDE_arcMTN_constant_l757_75731


namespace NUMINAMATH_CALUDE_rectangle_diagonal_maximum_l757_75725

theorem rectangle_diagonal_maximum (l w : ℝ) : 
  (2 * l + 2 * w = 40) → 
  (∀ l' w' : ℝ, (2 * l' + 2 * w' = 40) → (l'^2 + w'^2 ≤ l^2 + w^2)) →
  l^2 + w^2 = 200 :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_maximum_l757_75725


namespace NUMINAMATH_CALUDE_real_nut_findable_l757_75773

/-- Represents the type of a nut -/
inductive NutType
| Real
| Artificial

/-- Represents the result of a weighing -/
inductive WeighResult
| Equal
| LeftHeavier
| RightHeavier

/-- Represents a collection of nuts -/
structure NutCollection :=
  (nuts : Fin 6 → NutType)
  (realCount : Nat)
  (artificialCount : Nat)
  (real_count_correct : realCount = 4)
  (artificial_count_correct : artificialCount = 2)

/-- Represents a weighing operation -/
def weighNuts (collection : NutCollection) (left right sacrificed : Fin 6) : WeighResult :=
  sorry

/-- Represents the process of finding a real nut -/
def findRealNut (collection : NutCollection) : Fin 6 :=
  sorry

/-- The main theorem: it's possible to find a real nut without sacrificing it -/
theorem real_nut_findable (collection : NutCollection) :
  ∃ (n : Fin 6), collection.nuts n = NutType.Real ∧ n ≠ findRealNut collection :=
sorry

end NUMINAMATH_CALUDE_real_nut_findable_l757_75773


namespace NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l757_75783

/-- A standard deck of cards. -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- The suit of a card. -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- The rank of a card. -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A playing card. -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- The probability of drawing a heart first and a king second from a standard 52-card deck. -/
def prob_heart_then_king (d : Deck) : ℚ :=
  1 / 52

/-- Theorem stating that the probability of drawing a heart first and a king second
    from a standard 52-card deck is 1/52. -/
theorem prob_heart_then_king_is_one_fiftytwo (d : Deck) :
  prob_heart_then_king d = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l757_75783


namespace NUMINAMATH_CALUDE_y_range_l757_75782

theorem y_range (x y : ℝ) (h1 : |y - 2*x| = x^2) (h2 : -1 < x) (h3 : x < 0) :
  ∃ (a b : ℝ), a = -3 ∧ b = 0 ∧ a < y ∧ y < b ∧
  ∀ (z : ℝ), (∃ (w : ℝ), -1 < w ∧ w < 0 ∧ |z - 2*w| = w^2) → a ≤ z ∧ z ≤ b :=
sorry

end NUMINAMATH_CALUDE_y_range_l757_75782


namespace NUMINAMATH_CALUDE_initial_value_exists_and_unique_l757_75701

theorem initial_value_exists_and_unique : 
  ∃! x : ℤ, ∃ k : ℤ, x + 7 = k * 456 := by sorry

end NUMINAMATH_CALUDE_initial_value_exists_and_unique_l757_75701


namespace NUMINAMATH_CALUDE_order_of_powers_l757_75762

theorem order_of_powers : 
  let a : ℕ := 3^55
  let b : ℕ := 4^44
  let c : ℕ := 5^33
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_powers_l757_75762


namespace NUMINAMATH_CALUDE_quadrilaterals_with_fixed_point_l757_75766

theorem quadrilaterals_with_fixed_point (n : ℕ) (k : ℕ) :
  n = 11 ∧ k = 3 → Nat.choose n k = 165 := by sorry

end NUMINAMATH_CALUDE_quadrilaterals_with_fixed_point_l757_75766


namespace NUMINAMATH_CALUDE_rectangle_area_18_l757_75754

def rectangle_pairs : Set (Nat × Nat) :=
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}

theorem rectangle_area_18 :
  ∀ (w l : Nat), w > 0 ∧ l > 0 →
  (w * l = 18 ↔ (w, l) ∈ rectangle_pairs) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_18_l757_75754


namespace NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l757_75735

-- Define the probabilities for each ring
def p10 : ℝ := 0.21
def p9 : ℝ := 0.23
def p8 : ℝ := 0.25
def p7 : ℝ := 0.28

-- Theorem for the probability of hitting either 10 ring or 7 ring
theorem prob_10_or_7 : p10 + p7 = 0.49 := by sorry

-- Theorem for the probability of scoring below 7 ring
theorem prob_below_7 : 1 - (p10 + p9 + p8 + p7) = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l757_75735


namespace NUMINAMATH_CALUDE_equal_color_squares_count_l757_75718

/-- Represents a 5x5 grid with some cells painted black -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Counts the number of squares in the grid with equal black and white cells -/
def countEqualColorSquares (g : Grid) : ℕ :=
  let count2x2 := (5 - 2 + 1)^2 - 2  -- Total 2x2 squares minus those containing the center
  let count4x4 := 2  -- Lower two 4x4 squares meet the criterion
  count2x2 + count4x4

/-- Theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_color_squares_count (g : Grid) : countEqualColorSquares g = 16 := by
  sorry

end NUMINAMATH_CALUDE_equal_color_squares_count_l757_75718


namespace NUMINAMATH_CALUDE_typist_salary_problem_l757_75712

theorem typist_salary_problem (S : ℝ) : 
  S * 1.1 * 0.95 = 3135 → S = 3000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l757_75712


namespace NUMINAMATH_CALUDE_max_value_expression_l757_75750

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({0, 1, 2, 3} : Set ℕ) →
  b ∈ ({0, 1, 2, 3} : Set ℕ) →
  c ∈ ({0, 1, 2, 3} : Set ℕ) →
  d ∈ ({0, 1, 2, 3} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  d ≠ 0 →
  c * a^b - d ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l757_75750


namespace NUMINAMATH_CALUDE_modulus_of_z_l757_75764

theorem modulus_of_z (z : ℂ) (h : (2 * z) / (1 - z) = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l757_75764


namespace NUMINAMATH_CALUDE_water_level_rise_l757_75775

/-- Given a cube with edge length 15 cm and a rectangular vessel with base dimensions 20 cm × 15 cm,
    prove that the rise in water level when the cube is fully immersed is 11.25 cm. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  cube_edge = 15 →
  vessel_length = 20 →
  vessel_width = 15 →
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 := by
  sorry

#check water_level_rise

end NUMINAMATH_CALUDE_water_level_rise_l757_75775


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l757_75776

theorem quadratic_real_roots (n : ℕ+) :
  (∃ x : ℝ, x^2 - 4*x + n.val = 0) ↔ n.val = 1 ∨ n.val = 2 ∨ n.val = 3 ∨ n.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l757_75776


namespace NUMINAMATH_CALUDE_hotdog_sales_l757_75713

theorem hotdog_sales (small_hotdogs : ℕ) (total_hotdogs : ℕ) (large_hotdogs : ℕ)
  (h1 : small_hotdogs = 58)
  (h2 : total_hotdogs = 79)
  (h3 : total_hotdogs = small_hotdogs + large_hotdogs) :
  large_hotdogs = 21 := by
  sorry

end NUMINAMATH_CALUDE_hotdog_sales_l757_75713


namespace NUMINAMATH_CALUDE_max_value_S_l757_75769

theorem max_value_S (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 5) :
  ∃ (max : ℝ), max = 18 ∧ ∀ (a' b' c' : ℝ), a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → a' + b' + c' = 5 →
    2*a' + 2*a'*b' + a'*b'*c' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_S_l757_75769


namespace NUMINAMATH_CALUDE_choose_four_from_seven_l757_75787

-- Define the number of available paints
def n : ℕ := 7

-- Define the number of paints to be chosen
def k : ℕ := 4

-- Theorem stating that choosing 4 paints from 7 different ones results in 35 ways
theorem choose_four_from_seven :
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_seven_l757_75787


namespace NUMINAMATH_CALUDE_function_characterization_l757_75749

def iterate (f : ℕ → ℕ) : ℕ → ℕ → ℕ
  | 0, x => x
  | n + 1, x => f (iterate f n x)

theorem function_characterization (f : ℕ → ℕ) : 
  (∀ a b : ℕ, a > 0 → b > 0 → 
    (iterate f a b + iterate f b a) ∣ (2 * (f (a * b) + b^2 - 1))) → 
  ((∀ x : ℕ, f x = x + 1) ∨ (f 1 ∣ 4)) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l757_75749


namespace NUMINAMATH_CALUDE_polynomial_equality_l757_75733

theorem polynomial_equality (x : ℝ) : 
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = (2*x)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l757_75733


namespace NUMINAMATH_CALUDE_inscribed_square_area_l757_75742

/-- The parabola function y = x^2 - 10x + 21 --/
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bound by the parabola and the x-axis --/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  side : ℝ    -- length of the square's side
  h1 : center - side/2 ≥ 0  -- Left side of square is non-negative
  h2 : center + side/2 ≤ 10 -- Right side of square is at most the x-intercept
  h3 : parabola (center - side/2) = 0  -- Left bottom corner on x-axis
  h4 : parabola (center + side/2) = 0  -- Right bottom corner on x-axis
  h5 : parabola center = side          -- Top of square touches parabola

/-- The theorem stating the area of the inscribed square --/
theorem inscribed_square_area (s : InscribedSquare) :
  s.side^2 = 24 - 8*Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l757_75742


namespace NUMINAMATH_CALUDE_max_candy_leftover_l757_75795

theorem max_candy_leftover (y : ℕ) : ∃ (q r : ℕ), y = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l757_75795


namespace NUMINAMATH_CALUDE_system_is_linear_l757_75704

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- A system of two equations is linear if both equations are linear and they involve exactly two variables. -/
def is_linear_system (f g : ℝ → ℝ → ℝ) : Prop :=
  is_linear_equation f ∧ is_linear_equation g

/-- The given system of equations -/
def equation1 (x y : ℝ) : ℝ := x - y - 11
def equation2 (x y : ℝ) : ℝ := 4 * x - y - 1

theorem system_is_linear : is_linear_system equation1 equation2 := by
  sorry

end NUMINAMATH_CALUDE_system_is_linear_l757_75704


namespace NUMINAMATH_CALUDE_two_cyclists_problem_l757_75771

/-- Two cyclists problem -/
theorem two_cyclists_problem (north_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  north_speed = 30 →
  time = 0.7142857142857143 →
  distance = 50 →
  ∃ (south_speed : ℝ), south_speed = 40 ∧ distance = (north_speed + south_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_two_cyclists_problem_l757_75771


namespace NUMINAMATH_CALUDE_actual_lawn_area_l757_75736

/-- Actual area of a lawn given its blueprint measurements and scale -/
theorem actual_lawn_area 
  (blueprint_area : ℝ) 
  (blueprint_side : ℝ) 
  (actual_side : ℝ) 
  (h1 : blueprint_area = 300) 
  (h2 : blueprint_side = 5) 
  (h3 : actual_side = 1500) : 
  (actual_side / blueprint_side)^2 * blueprint_area = 2700 * 10000 := by
  sorry

end NUMINAMATH_CALUDE_actual_lawn_area_l757_75736


namespace NUMINAMATH_CALUDE_fraction_equivalence_l757_75748

theorem fraction_equivalence (x : ℝ) : (x + 1) / (x + 3) = 1 / 3 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l757_75748


namespace NUMINAMATH_CALUDE_cut_depth_proof_l757_75789

theorem cut_depth_proof (sheet_width sheet_height : ℕ) 
  (cut_width_1 cut_width_2 cut_width_3 : ℕ → ℕ) 
  (remaining_area : ℕ) : 
  sheet_width = 80 → 
  sheet_height = 15 → 
  (∀ d : ℕ, cut_width_1 d = 5 * d) →
  (∀ d : ℕ, cut_width_2 d = 15 * d) →
  (∀ d : ℕ, cut_width_3 d = 10 * d) →
  remaining_area = 990 →
  ∃ d : ℕ, d = 7 ∧ 
    sheet_width * sheet_height - (cut_width_1 d + cut_width_2 d + cut_width_3 d) = remaining_area :=
by sorry

end NUMINAMATH_CALUDE_cut_depth_proof_l757_75789


namespace NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l757_75717

theorem consecutive_product_prime_power_and_perfect_power (m : ℕ) : m ≥ 1 → (
  (∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m * (m + 1) = p ^ k) ↔ m = 1
) ∧ (
  ¬∃ (a k : ℕ), a ≥ 1 ∧ k ≥ 2 ∧ m * (m + 1) = a ^ k
) := by sorry

end NUMINAMATH_CALUDE_consecutive_product_prime_power_and_perfect_power_l757_75717


namespace NUMINAMATH_CALUDE_inequality_proof_l757_75728

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l757_75728


namespace NUMINAMATH_CALUDE_dogs_not_liking_any_food_l757_75747

/-- Given a kennel of dogs with specified food preferences, prove the number of dogs
    that don't like any of watermelon, salmon, or chicken. -/
theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : ℕ)
  (watermelon_and_salmon watermelon_and_chicken_not_salmon salmon_and_chicken_not_watermelon : ℕ)
  (h1 : total = 80)
  (h2 : watermelon = 21)
  (h3 : salmon = 58)
  (h4 : watermelon_and_salmon = 12)
  (h5 : chicken = 15)
  (h6 : watermelon_and_chicken_not_salmon = 7)
  (h7 : salmon_and_chicken_not_watermelon = 10) :
  total - (watermelon_and_salmon + (salmon - watermelon_and_salmon - salmon_and_chicken_not_watermelon) +
           (watermelon - watermelon_and_salmon - watermelon_and_chicken_not_salmon) +
           salmon_and_chicken_not_watermelon + watermelon_and_chicken_not_salmon) = 13 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_liking_any_food_l757_75747


namespace NUMINAMATH_CALUDE_pages_difference_l757_75755

/-- Represents the number of pages in a purple book -/
def purple_pages : ℕ := 230

/-- Represents the number of pages in an orange book -/
def orange_pages : ℕ := 510

/-- Represents the number of purple books Mirella read -/
def purple_books_read : ℕ := 5

/-- Represents the number of orange books Mirella read -/
def orange_books_read : ℕ := 4

/-- Theorem stating the difference in pages read between orange and purple books -/
theorem pages_difference : 
  orange_pages * orange_books_read - purple_pages * purple_books_read = 890 := by
  sorry

end NUMINAMATH_CALUDE_pages_difference_l757_75755


namespace NUMINAMATH_CALUDE_algebraic_simplification_l757_75730

theorem algebraic_simplification (a b : ℝ) :
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l757_75730


namespace NUMINAMATH_CALUDE_expression_simplification_l757_75786

theorem expression_simplification (x : ℝ) (h : x = 2) :
  (1 / (x + 1) - 1) / ((x^3 - x) / (x^2 + 2*x + 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l757_75786


namespace NUMINAMATH_CALUDE_zero_location_l757_75722

def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

theorem zero_location (f : ℝ → ℝ) :
  has_unique_zero f 0 16 ∧
  has_unique_zero f 0 8 ∧
  has_unique_zero f 0 4 ∧
  has_unique_zero f 0 2 →
  ∀ x, 2 ≤ x ∧ x < 16 → f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_location_l757_75722


namespace NUMINAMATH_CALUDE_cone_surface_area_l757_75711

/-- The surface area of a cone given its slant height and lateral surface central angle -/
theorem cone_surface_area (s : ℝ) (θ : ℝ) (h_s : s = 3) (h_θ : θ = 2 * Real.pi / 3) :
  s * θ / 2 + Real.pi * (s * θ / (2 * Real.pi))^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l757_75711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l757_75741

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 8 = 15 - a 5 → a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l757_75741


namespace NUMINAMATH_CALUDE_rectangle_from_triangles_l757_75744

/-- Represents a right-angled triangle tile with integer side lengths -/
structure Triangle :=
  (a b c : ℕ)

/-- Represents a rectangle with integer side lengths -/
structure Rectangle :=
  (width height : ℕ)

/-- Checks if a triangle is valid (right-angled and positive sides) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ t.a^2 + t.b^2 = t.c^2

/-- Checks if a rectangle can be formed from a given number of triangles -/
def canFormRectangle (r : Rectangle) (t : Triangle) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ 2 * n * t.a * t.b = r.width * r.height

theorem rectangle_from_triangles 
  (jackTile : Triangle)
  (targetRect : Rectangle)
  (h1 : isValidTriangle jackTile)
  (h2 : jackTile.a = 3 ∧ jackTile.b = 4 ∧ jackTile.c = 5)
  (h3 : targetRect.width = 2016 ∧ targetRect.height = 2021) :
  canFormRectangle targetRect jackTile :=
sorry

end NUMINAMATH_CALUDE_rectangle_from_triangles_l757_75744


namespace NUMINAMATH_CALUDE_total_pages_in_textbooks_l757_75778

/-- Represents the number of pages in each textbook and calculates the total --/
def textbook_pages : ℕ → ℕ → ℕ → ℕ → ℕ := fun history geography math science =>
  history + geography + math + science

/-- Theorem stating the total number of pages in Suzanna's textbooks --/
theorem total_pages_in_textbooks : ∃ (history geography math science : ℕ),
  history = 160 ∧
  geography = history + 70 ∧
  math = (history + geography) / 2 ∧
  science = 2 * history ∧
  textbook_pages history geography math science = 905 := by
  sorry

#eval textbook_pages 160 230 195 320

end NUMINAMATH_CALUDE_total_pages_in_textbooks_l757_75778


namespace NUMINAMATH_CALUDE_equality_of_exponential_equation_l757_75779

theorem equality_of_exponential_equation (a b : ℝ) : 
  0 < a → 0 < b → a < 1 → a^b = b^a → a = b := by sorry

end NUMINAMATH_CALUDE_equality_of_exponential_equation_l757_75779


namespace NUMINAMATH_CALUDE_square_area_equals_perimeter_l757_75740

theorem square_area_equals_perimeter (s : ℝ) (h : s > 0) :
  s^2 = 4*s → 4*s = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_perimeter_l757_75740


namespace NUMINAMATH_CALUDE_opposite_corner_not_always_farthest_l757_75709

/-- A rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- A point on the surface of a box -/
structure SurfacePoint (b : Box) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = b.length) ∨ (y = 0 ∨ y = b.width) ∨ (z = 0 ∨ z = b.height)

/-- The distance between two points on the surface of a box -/
noncomputable def surface_distance (b : Box) (p1 p2 : SurfacePoint b) : ℝ :=
  sorry

/-- The corner opposite to (0, 0, 0) -/
def opposite_corner (b : Box) : SurfacePoint b :=
  { x := b.length, y := b.width, z := b.height,
    on_surface := by simp }

/-- Theorem: The opposite corner is not necessarily the point with the greatest distance from a corner -/
theorem opposite_corner_not_always_farthest (b : Box) :
  ∃ (p : SurfacePoint b), surface_distance b ⟨0, 0, 0, by simp⟩ p > 
                           surface_distance b ⟨0, 0, 0, by simp⟩ (opposite_corner b) :=
sorry

end NUMINAMATH_CALUDE_opposite_corner_not_always_farthest_l757_75709


namespace NUMINAMATH_CALUDE_number_equation_l757_75743

theorem number_equation (x : ℝ) : (1/4 : ℝ) * x + 15 = 27 ↔ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l757_75743


namespace NUMINAMATH_CALUDE_same_color_probability_l757_75729

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose blue_plates plates_selected) /
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l757_75729


namespace NUMINAMATH_CALUDE_real_condition_implies_a_equals_one_l757_75724

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that a complex number is real
def is_real (z : ℂ) : Prop := z.im = 0

-- Theorem statement
theorem real_condition_implies_a_equals_one (a : ℝ) :
  is_real ((1 + i) * (1 - a * i)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_condition_implies_a_equals_one_l757_75724


namespace NUMINAMATH_CALUDE_probability_both_above_400_l757_75715

def total_students : ℕ := 600
def male_students : ℕ := 220
def female_students : ℕ := 380
def selected_students : ℕ := 10
def selected_females : ℕ := 6
def females_above_400 : ℕ := 3
def discussion_group_size : ℕ := 2

theorem probability_both_above_400 :
  (female_students = total_students - male_students) →
  (selected_females ≤ selected_students) →
  (females_above_400 ≤ selected_females) →
  (discussion_group_size ≤ selected_females) →
  (Nat.choose females_above_400 discussion_group_size) / (Nat.choose selected_females discussion_group_size) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_above_400_l757_75715


namespace NUMINAMATH_CALUDE_right_cone_diameter_l757_75708

theorem right_cone_diameter (h : ℝ) (s : ℝ) (d : ℝ) :
  h = 3 →
  s = 5 →
  s^2 = h^2 + (d/2)^2 →
  d = 8 :=
by sorry

end NUMINAMATH_CALUDE_right_cone_diameter_l757_75708


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l757_75760

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l757_75760


namespace NUMINAMATH_CALUDE_shopping_spree_cost_equalization_l757_75745

/-- Given the spending amounts and agreement to equally share costs, 
    prove that the difference between what Charlie gives to Bob and 
    what Alice gives to Bob is 30. -/
theorem shopping_spree_cost_equalization 
  (charlie_spent : ℝ) 
  (alice_spent : ℝ) 
  (bob_spent : ℝ) 
  (h1 : charlie_spent = 150)
  (h2 : alice_spent = 180)
  (h3 : bob_spent = 210)
  (c : ℝ)  -- amount Charlie gives to Bob
  (a : ℝ)  -- amount Alice gives to Bob
  (h4 : c = (charlie_spent + alice_spent + bob_spent) / 3 - charlie_spent)
  (h5 : a = (charlie_spent + alice_spent + bob_spent) / 3 - alice_spent) :
  c - a = 30 := by
sorry


end NUMINAMATH_CALUDE_shopping_spree_cost_equalization_l757_75745


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l757_75770

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1) + n * (n - 1)

-- Define b_n
def b (n : ℕ) : ℚ := (-1)^(n-1) * (4 * n) / (a n * a (n+1))

-- Define T_n (sum of first n terms of b_n)
def T (n : ℕ) : ℚ :=
  if n % 2 = 0 then (2 * n) / (2 * n + 1)
  else (2 * n + 2) / (2 * n + 1)

-- Theorem statement
theorem arithmetic_sequence_proof :
  (∀ n : ℕ, S (n+1) - S n = 2) ∧  -- Common difference is 2
  (S 2)^2 = S 1 * S 4 ∧           -- S_1, S_2, S_4 form a geometric sequence
  (∀ n : ℕ, a n = 2 * n - 1) ∧    -- General formula for a_n
  (∀ n : ℕ, T n = if n % 2 = 0 then (2 * n) / (2 * n + 1) else (2 * n + 2) / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l757_75770


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l757_75793

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = x^2 - 2*x + 2}

-- Theorem statement
theorem complement_A_intersect_B :
  (Set.univ : Set ℝ) \ (A ∩ B) = {x : ℝ | x < 1 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l757_75793


namespace NUMINAMATH_CALUDE_same_color_probability_l757_75726

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates
def plates_selected : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_selected + Nat.choose blue_plates plates_selected) / 
  Nat.choose total_plates plates_selected = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l757_75726


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l757_75763

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (j : ℕ), 5 * n = j^3) ∧
  (∀ (m : ℕ), m > 0 → 
    ((∃ (k : ℕ), 4 * m = k^2) ∧ (∃ (j : ℕ), 5 * m = j^3)) → 
    m ≥ 500) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l757_75763


namespace NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l757_75727

theorem johnson_family_reunion_ratio : 
  let num_children : ℕ := 45
  let num_adults : ℕ := num_children / 3
  let adults_not_blue : ℕ := 10
  let adults_blue : ℕ := num_adults - adults_not_blue
  adults_blue / num_adults = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_johnson_family_reunion_ratio_l757_75727


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l757_75788

theorem min_x_prime_factorization_sum (x y : ℕ+) (h : 3 * x ^ 12 = 5 * y ^ 17) :
  ∃ (a b c d : ℕ),
    (∀ (p : ℕ), p.Prime → p ∣ x → p = a ∨ p = b) ∧
    x = a ^ c * b ^ d ∧
    (∀ (x' : ℕ+), 3 * x' ^ 12 = 5 * y ^ 17 → x ≤ x') ∧
    a + b + c + d = 30 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l757_75788


namespace NUMINAMATH_CALUDE_problem_statement_l757_75768

theorem problem_statement (x y : ℝ) : 
  let a := x^3 * y
  let b := x^2 * y^2
  let c := x * y^3
  (a * c + b^2 - 2 * x^4 * y^4 = 0) ∧ 
  (a * y^2 + c * x^2 = 2 * x * y * b) ∧ 
  ¬(∀ x y : ℝ, a * b * c + b^3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l757_75768


namespace NUMINAMATH_CALUDE_jumping_contest_l757_75758

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_jump = 15)
  (h3 : mouse_jump + 44 = frog_jump) :
  grasshopper_jump - frog_jump = 4 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l757_75758


namespace NUMINAMATH_CALUDE_gummy_bears_count_l757_75751

/-- The number of gummy bears produced per minute -/
def production_rate : ℕ := 300

/-- The time taken to produce enough gummy bears to fill the packets (in minutes) -/
def production_time : ℕ := 40

/-- The number of packets filled with the gummy bears produced -/
def num_packets : ℕ := 240

/-- The number of gummy bears in each packet -/
def gummy_bears_per_packet : ℕ := production_rate * production_time / num_packets

theorem gummy_bears_count : gummy_bears_per_packet = 50 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bears_count_l757_75751


namespace NUMINAMATH_CALUDE_cuboid_from_rectangular_projections_l757_75721

/-- Represents a solid object in 3D space -/
structure Solid :=
  (shape : Type)

/-- Represents an orthographic projection (view) of a solid -/
inductive Projection
  | Rectangle
  | Other

/-- Defines the front view of a solid -/
def front_view (s : Solid) : Projection := sorry

/-- Defines the top view of a solid -/
def top_view (s : Solid) : Projection := sorry

/-- Defines the side view of a solid -/
def side_view (s : Solid) : Projection := sorry

/-- Defines a cuboid -/
def is_cuboid (s : Solid) : Prop := sorry

/-- Theorem: If all three orthographic projections of a solid are rectangles, then the solid is a cuboid -/
theorem cuboid_from_rectangular_projections (s : Solid) :
  front_view s = Projection.Rectangle →
  top_view s = Projection.Rectangle →
  side_view s = Projection.Rectangle →
  is_cuboid s :=
sorry

end NUMINAMATH_CALUDE_cuboid_from_rectangular_projections_l757_75721


namespace NUMINAMATH_CALUDE_time_between_ticks_at_6_l757_75705

/-- The number of ticks at 6 o'clock -/
def ticks_at_6 : ℕ := 6

/-- The number of ticks at 8 o'clock -/
def ticks_at_8 : ℕ := 8

/-- The time between the first and last ticks at 8 o'clock in seconds -/
def time_at_8 : ℕ := 42

/-- The theorem stating the time between the first and last ticks at 6 o'clock -/
theorem time_between_ticks_at_6 : ℕ := by
  -- Assume the time between each tick is constant for any hour
  -- Calculate the time between ticks at 6 o'clock
  sorry

end NUMINAMATH_CALUDE_time_between_ticks_at_6_l757_75705


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_to_polynomial_l757_75703

-- Problem 1
theorem simplify_and_evaluate (a : ℚ) : 
  a = -2 → (a^2 + a) / (a^2 - 3*a) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2/3 := by
  sorry

-- Problem 2
theorem simplify_to_polynomial (x : ℚ) : 
  (x^2 - 1) / (x - 4) / ((x + 1) / (4 - x)) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_simplify_to_polynomial_l757_75703


namespace NUMINAMATH_CALUDE_chord_equation_l757_75791

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an ellipse -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)

/-- Checks if a point is inside an ellipse -/
def isInside (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) < 1

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Checks if a point bisects a chord of an ellipse -/
def bisectsChord (p : Point) (e : Ellipse) : Prop :=
  sorry  -- Definition of bisecting a chord

theorem chord_equation (e : Ellipse) (m : Point) :
  e.a = 4 →
  e.b = 2 →
  m.x = 2 →
  m.y = 1 →
  isInside m e →
  bisectsChord m e →
  ∃ l : Line, l.a = 1 ∧ l.b = 2 ∧ l.c = -4 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l757_75791


namespace NUMINAMATH_CALUDE_greatest_abcba_div_by_11_and_3_l757_75790

def is_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_div_by_11_and_3 : 
  (∀ n : ℕ, is_abcba n → n % 11 = 0 → n % 3 = 0 → n ≤ 96569) ∧ 
  is_abcba 96569 ∧ 
  96569 % 11 = 0 ∧ 
  96569 % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_div_by_11_and_3_l757_75790


namespace NUMINAMATH_CALUDE_plane_division_theorem_l757_75738

/-- A line in the plane --/
structure Line where
  -- We don't need to define the actual properties of a line for this statement

/-- A set of lines in the plane --/
def LineSet := Set Line

/-- Predicate to check if all lines in a set are parallel to one of them --/
def allParallel (ls : LineSet) : Prop := sorry

/-- Number of regions formed by a set of lines --/
def numRegions (ls : LineSet) : ℕ := sorry

/-- Statement of the theorem --/
theorem plane_division_theorem :
  ∃ (k₀ : ℕ), ∀ (k : ℕ), k > k₀ →
    ∃ (ls : LineSet), ls.Finite ∧ ¬allParallel ls ∧ numRegions ls = k :=
by
  -- Let k₀ = 5
  use 5
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_plane_division_theorem_l757_75738


namespace NUMINAMATH_CALUDE_jellybean_problem_l757_75700

theorem jellybean_problem (eat_rate : ℝ) (days : ℕ) (remaining : ℕ) (original : ℕ) : 
  eat_rate = 0.25 →
  days = 3 →
  remaining = 27 →
  (1 - eat_rate) ^ days * original = remaining →
  original = 64 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l757_75700


namespace NUMINAMATH_CALUDE_solve_equation_l757_75732

theorem solve_equation : ∃ x : ℝ, x + 2*x = 400 - (3*x + 4*x) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l757_75732


namespace NUMINAMATH_CALUDE_xy_value_l757_75797

theorem xy_value (x y : ℝ) (h : |x + 2*y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l757_75797


namespace NUMINAMATH_CALUDE_fraction_simplification_l757_75785

theorem fraction_simplification : 
  (20 : ℚ) / 19 * 15 / 28 * 76 / 45 = 95 / 84 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l757_75785


namespace NUMINAMATH_CALUDE_inequality_equivalence_l757_75756

/-- The inequality holds for all positive q if and only if p is in the interval [0, 2) -/
theorem inequality_equivalence (p : ℝ) : 
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l757_75756


namespace NUMINAMATH_CALUDE_cookies_per_bag_l757_75767

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : num_bags = 7)
  (h3 : total_cookies = num_bags * cookies_per_bag) :
  cookies_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l757_75767


namespace NUMINAMATH_CALUDE_class_size_theorem_l757_75774

theorem class_size_theorem :
  ∀ (m d : ℕ),
  (∃ (r : ℕ), r = 3 * m ∧ r = 5 * d) →
  30 < m + d →
  m + d < 40 →
  m + d = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_theorem_l757_75774


namespace NUMINAMATH_CALUDE_nine_trailing_zeros_l757_75746

def binary_trailing_zeros (n : ℕ) : ℕ :=
  (n.digits 2).reverse.takeWhile (· = 0) |>.length

theorem nine_trailing_zeros (n : ℕ) : binary_trailing_zeros (n * 1024 + 4 * 64 + 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_trailing_zeros_l757_75746


namespace NUMINAMATH_CALUDE_calculation_result_l757_75752

-- Define the numbers in their respective bases
def num_base_8 : ℚ := 2 * 8^3 + 4 * 8^2 + 6 * 8^1 + 8 * 8^0
def num_base_5 : ℚ := 1 * 5^2 + 2 * 5^1 + 1 * 5^0
def num_base_9 : ℚ := 1 * 9^3 + 3 * 9^2 + 5 * 9^1 + 7 * 9^0
def num_base_10 : ℚ := 2048

-- Define the result of the calculation
def result : ℚ := num_base_8 / num_base_5 - num_base_9 + num_base_10

-- State the theorem
theorem calculation_result : result = 1061.1111 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l757_75752


namespace NUMINAMATH_CALUDE_decimal_multiplication_equivalence_l757_75706

theorem decimal_multiplication_equivalence (given : 268 * 74 = 19832) :
  ∃ x : ℝ, 2.68 * x = 1.9832 ∧ x = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_equivalence_l757_75706


namespace NUMINAMATH_CALUDE_total_vowels_written_l757_75759

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 3

/-- Theorem: The total number of vowels written on the board is 15 -/
theorem total_vowels_written : num_vowels * times_written = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_vowels_written_l757_75759


namespace NUMINAMATH_CALUDE_ellipse_and_fixed_point_l757_75716

/-- Ellipse C₁ -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Parabola C₂ -/
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- Tangent line to parabola -/
def tangent_line (b : ℝ) (x y : ℝ) : Prop :=
  y = x + b

/-- Circle with diameter AB passing through T -/
def circle_passes_through (A B T : ℝ × ℝ) : Prop :=
  (T.1 - A.1) * (T.1 - B.1) + (T.2 - A.2) * (T.2 - B.2) = 0

theorem ellipse_and_fixed_point 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : a^2 - b^2 = a^2 / 2) -- Eccentricity condition
  (h4 : ∃ (x y : ℝ), tangent_line 1 x y ∧ parabola x y) :
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (A B : ℝ × ℝ), 
    ellipse a b A.1 A.2 → 
    ellipse a b B.1 B.2 → 
    (∃ (k : ℝ), A.2 = k * A.1 - 1/3 ∧ B.2 = k * B.1 - 1/3) →
    circle_passes_through A B (0, 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_fixed_point_l757_75716


namespace NUMINAMATH_CALUDE_max_distance_with_specific_tires_l757_75710

/-- Represents the maximum distance a car can travel with tire switching -/
def maxDistanceWithTireSwitching (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  min frontTireLife rearTireLife

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_with_specific_tires :
  maxDistanceWithTireSwitching 42000 56000 = 42000 := by
  sorry

#check max_distance_with_specific_tires

end NUMINAMATH_CALUDE_max_distance_with_specific_tires_l757_75710


namespace NUMINAMATH_CALUDE_machine_production_difference_l757_75739

/-- The number of widgets produced in total -/
def total_widgets : ℕ := 1080

/-- The number of widgets Machine X produces per hour -/
def machine_x_rate : ℝ := 3

/-- The difference in hours between Machine X and Machine Y to produce the total widgets -/
def hour_difference : ℕ := 60

/-- The percentage difference in production rate between Machine Y and Machine X -/
def percentage_difference : ℝ := 20

theorem machine_production_difference :
  let machine_x_hours : ℝ := total_widgets / machine_x_rate
  let machine_y_hours : ℝ := machine_x_hours - hour_difference
  let machine_y_rate : ℝ := total_widgets / machine_y_hours
  (machine_y_rate - machine_x_rate) / machine_x_rate * 100 = percentage_difference :=
by sorry

end NUMINAMATH_CALUDE_machine_production_difference_l757_75739


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l757_75702

theorem cubic_equation_real_root (a b : ℝ) : ∃ x : ℝ, x^3 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l757_75702


namespace NUMINAMATH_CALUDE_inverse_of_A_is_B_l757_75761

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 7; 2, 3]

def B : Matrix (Fin 2) (Fin 2) ℚ := !![-3/2, 7/2; 1, -2]

theorem inverse_of_A_is_B :
  (Matrix.det A ≠ 0) → (A * B = 1 ∧ B * A = 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_A_is_B_l757_75761


namespace NUMINAMATH_CALUDE_james_weekly_earnings_l757_75784

/-- Calculates the weekly earnings from car rental -/
def weekly_earnings (rate : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  rate * hours_per_day * days_per_week

/-- Proof that James' weekly earnings from car rental are $640 -/
theorem james_weekly_earnings :
  weekly_earnings 20 8 4 = 640 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_earnings_l757_75784


namespace NUMINAMATH_CALUDE_ellipse_properties_l757_75792

/-- Given an ellipse with equation x²/25 + y²/9 = 1, prove its semi-major axis length and eccentricity -/
theorem ellipse_properties : ∃ (a b c : ℝ), 
  (∀ x y : ℝ, x^2/25 + y^2/9 = 1 → 
    a = 5 ∧ 
    b = 3 ∧ 
    c^2 = a^2 - b^2 ∧ 
    a = 5 ∧ 
    c/a = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l757_75792
