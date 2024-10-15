import Mathlib

namespace NUMINAMATH_CALUDE_lollipop_cost_is_two_l1428_142862

/-- The cost of a single lollipop in dollars -/
def lollipop_cost : ℝ := 2

/-- The number of lollipops bought -/
def num_lollipops : ℕ := 4

/-- The number of chocolate packs bought -/
def num_chocolate_packs : ℕ := 6

/-- The number of $10 bills used for payment -/
def num_ten_dollar_bills : ℕ := 6

/-- The amount of change received in dollars -/
def change_received : ℝ := 4

theorem lollipop_cost_is_two :
  lollipop_cost = 2 ∧
  num_lollipops * lollipop_cost + num_chocolate_packs * (4 * lollipop_cost) = 
    num_ten_dollar_bills * 10 - change_received :=
by sorry

end NUMINAMATH_CALUDE_lollipop_cost_is_two_l1428_142862


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l1428_142833

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem murtha_pebble_collection :
  arithmetic_sum 1 1 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l1428_142833


namespace NUMINAMATH_CALUDE_solution_range_l1428_142872

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, a * x < 6 ∧ (3 * x - 6 * a) / 2 > a / 3 - 1) → 
  a ≤ -3/2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l1428_142872


namespace NUMINAMATH_CALUDE_coral_age_conversion_l1428_142825

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_number (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * 8^i) 0

theorem coral_age_conversion :
  octal_to_decimal_number [7, 3, 4] = 476 := by
  sorry

end NUMINAMATH_CALUDE_coral_age_conversion_l1428_142825


namespace NUMINAMATH_CALUDE_impossibility_of_tiling_101_square_l1428_142867

theorem impossibility_of_tiling_101_square : ¬ ∃ (a b : ℕ), 4*a + 9*b = 101*101 := by sorry

end NUMINAMATH_CALUDE_impossibility_of_tiling_101_square_l1428_142867


namespace NUMINAMATH_CALUDE_bike_clamps_promotion_l1428_142854

/-- The number of bike clamps given per bicycle purchase -/
def clamps_per_bike (morning_bikes : ℕ) (afternoon_bikes : ℕ) (total_clamps : ℕ) : ℚ :=
  total_clamps / (morning_bikes + afternoon_bikes)

/-- Theorem stating that the number of bike clamps given per bicycle purchase is 2 -/
theorem bike_clamps_promotion (morning_bikes afternoon_bikes total_clamps : ℕ)
  (h1 : morning_bikes = 19)
  (h2 : afternoon_bikes = 27)
  (h3 : total_clamps = 92) :
  clamps_per_bike morning_bikes afternoon_bikes total_clamps = 2 := by
  sorry

end NUMINAMATH_CALUDE_bike_clamps_promotion_l1428_142854


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l1428_142818

theorem geometric_sequence_b_value (b : ℝ) (h_positive : b > 0) 
  (h_sequence : ∃ r : ℝ, 250 * r = b ∧ b * r = 81 / 50) : 
  b = 9 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l1428_142818


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l1428_142885

theorem sqrt_expressions_equality :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (Real.sqrt (24 * a) - Real.sqrt (18 * b)) - Real.sqrt (6 * c) = 
    Real.sqrt (6 * c) - 3 * Real.sqrt (2 * b)) ∧
  (∀ d e f : ℝ, d > 0 → e > 0 → f > 0 →
    2 * Real.sqrt (12 * d) * Real.sqrt ((1 / 8) * e) + 5 * Real.sqrt (2 * f) = 
    Real.sqrt (6 * d) + 5 * Real.sqrt (2 * f)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l1428_142885


namespace NUMINAMATH_CALUDE_max_dot_product_on_circle_l1428_142829

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ),
  P.1^2 + P.2^2 = 1 →
  let A : ℝ × ℝ := (-2, 0)
  let O : ℝ × ℝ := (0, 0)
  let AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)
  let AP : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)
  (AO.1 * AP.1 + AO.2 * AP.2) ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_on_circle_l1428_142829


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1428_142865

theorem quadratic_equation_solution : 
  let x₁ := 2 + Real.sqrt 5
  let x₂ := 2 - Real.sqrt 5
  (x₁^2 - 4*x₁ - 1 = 0) ∧ (x₂^2 - 4*x₂ - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1428_142865


namespace NUMINAMATH_CALUDE_problem_solution_l1428_142863

theorem problem_solution (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1428_142863


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1428_142890

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of 2a + 3b is 4√5. -/
theorem parallel_vectors_magnitude (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), a = k • b) →
  ‖(2 : ℝ) • a + (3 : ℝ) • b‖ = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1428_142890


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1428_142841

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The square is inscribed in the region bound by the parabola and x-axis -/
def is_inscribed_square (s : ℝ) : Prop :=
  ∃ (center : ℝ), 
    parabola (center - s) = 0 ∧
    parabola (center + s) = 0 ∧
    parabola (center + s) = 2*s

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∃ (s : ℝ), is_inscribed_square s ∧ (2*s)^2 = 64 - 16*Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1428_142841


namespace NUMINAMATH_CALUDE_sequence_nth_term_l1428_142853

theorem sequence_nth_term (u : ℕ → ℝ) (u₀ a b : ℝ) (h : ∀ n : ℕ, u (n + 1) = a * u n + b) :
  ∀ n : ℕ, u n = if a = 1
    then u₀ + n * b
    else a^n * u₀ + b * (1 - a^(n + 1)) / (1 - a) :=
by sorry

end NUMINAMATH_CALUDE_sequence_nth_term_l1428_142853


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1428_142811

theorem trigonometric_identity : 
  let a : Real := 2 * Real.pi / 3
  Real.sin (Real.pi - a / 2) + Real.tan (a - 5 * Real.pi / 12) = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1428_142811


namespace NUMINAMATH_CALUDE_exercise_weights_after_training_l1428_142828

def calculate_final_weight (initial_weight : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_weight

def bench_press_changes : List ℝ := [-0.8, 0.6, -0.2, 2.0]
def squat_changes : List ℝ := [-0.5, 0.4, 1.0]
def deadlift_changes : List ℝ := [-0.3, 0.8, -0.4, 0.5]

theorem exercise_weights_after_training (initial_bench : ℝ) (initial_squat : ℝ) (initial_deadlift : ℝ) 
    (h1 : initial_bench = 500) 
    (h2 : initial_squat = 400) 
    (h3 : initial_deadlift = 600) :
  (calculate_final_weight initial_bench bench_press_changes = 384) ∧
  (calculate_final_weight initial_squat squat_changes = 560) ∧
  (calculate_final_weight initial_deadlift deadlift_changes = 680.4) := by
  sorry

#eval calculate_final_weight 500 bench_press_changes
#eval calculate_final_weight 400 squat_changes
#eval calculate_final_weight 600 deadlift_changes

end NUMINAMATH_CALUDE_exercise_weights_after_training_l1428_142828


namespace NUMINAMATH_CALUDE_area_of_original_figure_l1428_142891

/-- Represents the properties of an oblique diametric view of a figure -/
structure ObliqueView where
  is_isosceles_trapezoid : Bool
  base_angle : ℝ
  leg_length : ℝ
  top_base_length : ℝ

/-- Calculates the area of the original plane figure given its oblique diametric view -/
def original_area (view : ObliqueView) : ℝ :=
  sorry

/-- Theorem stating the area of the original plane figure given specific oblique view properties -/
theorem area_of_original_figure (view : ObliqueView) 
  (h1 : view.is_isosceles_trapezoid = true)
  (h2 : view.base_angle = π / 4)  -- 45° in radians
  (h3 : view.leg_length = 1)
  (h4 : view.top_base_length = 1) :
  original_area view = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_area_of_original_figure_l1428_142891


namespace NUMINAMATH_CALUDE_max_profit_price_l1428_142831

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- The initial purchase price of the product -/
def initial_purchase_price : ℝ := 8

/-- The initial selling price of the product -/
def initial_selling_price : ℝ := 10

/-- The initial daily sales volume -/
def initial_daily_sales : ℝ := 100

/-- The decrease in daily sales for each yuan increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Theorem: The selling price that maximizes profit is 14 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x > initial_selling_price ∧ 
  ∀ (y : ℝ), y > initial_selling_price → profit_function x ≥ profit_function y :=
sorry

end NUMINAMATH_CALUDE_max_profit_price_l1428_142831


namespace NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l1428_142827

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 3]

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → ‖vector_a x‖ = 5) ∧
  (∃ y : ℝ, y ≠ 4 ∧ ‖vector_a y‖ = 5) :=
by sorry

end NUMINAMATH_CALUDE_x_4_sufficient_not_necessary_l1428_142827


namespace NUMINAMATH_CALUDE_tomato_price_theorem_l1428_142852

/-- The original price per pound of tomatoes -/
def original_price : ℝ := 0.80

/-- The percentage of tomatoes that can be sold -/
def sellable_percentage : ℝ := 0.90

/-- The selling price per pound of tomatoes -/
def selling_price : ℝ := 0.96

/-- The profit percentage of the cost -/
def profit_percentage : ℝ := 0.08

/-- Theorem stating that the original price satisfies the given conditions -/
theorem tomato_price_theorem :
  selling_price * sellable_percentage = 
  original_price * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_tomato_price_theorem_l1428_142852


namespace NUMINAMATH_CALUDE_train_speed_l1428_142819

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 450) (h2 : time = 12) :
  length / time = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1428_142819


namespace NUMINAMATH_CALUDE_union_complement_equality_l1428_142805

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x < 1}

theorem union_complement_equality : A ∪ (U \ B) = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l1428_142805


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1428_142855

theorem consecutive_integers_square_sum : 
  ∀ (a b c : ℕ), 
    a > 0 → 
    b = a + 1 → 
    c = b + 1 → 
    a * b * c = 6 * (a + b + c) → 
    a^2 + b^2 + c^2 = 77 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l1428_142855


namespace NUMINAMATH_CALUDE_soccer_team_biology_count_l1428_142870

theorem soccer_team_biology_count :
  ∀ (total_players physics_count chemistry_count all_three_count physics_and_chemistry_count : ℕ),
    total_players = 15 →
    physics_count = 8 →
    chemistry_count = 6 →
    all_three_count = 3 →
    physics_and_chemistry_count = 4 →
    ∃ (biology_count : ℕ),
      biology_count = 9 ∧
      biology_count = total_players - (physics_count - physics_and_chemistry_count) - (chemistry_count - physics_and_chemistry_count) :=
by
  sorry

#check soccer_team_biology_count

end NUMINAMATH_CALUDE_soccer_team_biology_count_l1428_142870


namespace NUMINAMATH_CALUDE_valid_numbers_l1428_142896

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 = (n / 100) % 10) ∧
  ((n / 100) % 10 = n % 10) ∧
  (n ^ 2) % ((n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)) = 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {1111, 1212, 1515, 2424, 3636} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1428_142896


namespace NUMINAMATH_CALUDE_sum_powers_of_i_l1428_142874

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_powers_of_i :
  i^300 + i^301 + i^302 + i^303 + i^304 + i^305 + i^306 + i^307 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_powers_of_i_l1428_142874


namespace NUMINAMATH_CALUDE_solve_for_z_l1428_142837

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem solve_for_z : ∃ z : ℝ, euro (euro 4 5) z = 560 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l1428_142837


namespace NUMINAMATH_CALUDE_largest_fraction_addition_l1428_142803

def is_proper_fraction (n d : ℤ) : Prop := 0 < n ∧ n < d

def denominator_less_than_8 (d : ℤ) : Prop := 0 < d ∧ d < 8

theorem largest_fraction_addition :
  ∀ n d : ℤ,
    is_proper_fraction n d →
    denominator_less_than_8 d →
    is_proper_fraction (6 * n + d) (6 * d) →
    n * 7 ≤ 5 * d :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_addition_l1428_142803


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_x_in_1_to_3_l1428_142887

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem range_of_a_when_x_in_1_to_3 :
  {a : ℝ | ∀ x ∈ Set.Icc 1 3, f x a ≤ 3} = Set.Icc (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_x_in_1_to_3_l1428_142887


namespace NUMINAMATH_CALUDE_factorization_equality_l1428_142844

theorem factorization_equality (a b : ℝ) : a * b^3 - 4 * a * b = a * b * (b + 2) * (b - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1428_142844


namespace NUMINAMATH_CALUDE_y_derivative_l1428_142851

noncomputable def y (x : ℝ) : ℝ := 
  (1/12) * Real.log ((x^4 - x^2 + 1) / (x^2 + 1)^2) - 
  (1 / (2 * Real.sqrt 3)) * Real.arctan (Real.sqrt 3 / (2*x^2 - 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = x^3 / ((x^4 - x^2 + 1) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l1428_142851


namespace NUMINAMATH_CALUDE_white_balls_count_l1428_142876

theorem white_balls_count (total : ℕ) (prob : ℚ) (w : ℕ) : 
  total = 15 → 
  prob = 1 / 21 → 
  (w : ℚ) / total * ((w - 1) : ℚ) / (total - 1) = prob → 
  w = 5 := by sorry

end NUMINAMATH_CALUDE_white_balls_count_l1428_142876


namespace NUMINAMATH_CALUDE_intersection_count_l1428_142835

-- Define the line L
def line_L (x y : ℝ) : Prop := y = 2 + Real.sqrt 3 - Real.sqrt 3 * x

-- Define the ellipse C'
def ellipse_C' (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define a point on the line L
def point_on_L : Prop := line_L 1 2

-- Theorem statement
theorem intersection_count :
  point_on_L →
  ∃ (p q : ℝ × ℝ),
    p ≠ q ∧
    line_L p.1 p.2 ∧
    line_L q.1 q.2 ∧
    ellipse_C' p.1 p.2 ∧
    ellipse_C' q.1 q.2 ∧
    ∀ (r : ℝ × ℝ), line_L r.1 r.2 ∧ ellipse_C' r.1 r.2 → r = p ∨ r = q :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l1428_142835


namespace NUMINAMATH_CALUDE_train_speed_conversion_l1428_142888

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Speed of the train in meters per second -/
def train_speed_mps : ℝ := 52.5042

/-- Theorem stating the conversion of train speed from m/s to km/h -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 189.01512 := by sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l1428_142888


namespace NUMINAMATH_CALUDE_total_installments_count_l1428_142860

/-- Proves that the total number of installments is 52 given the specified payment conditions -/
theorem total_installments_count (first_25_payment : ℝ) (remaining_payment : ℝ) (average_payment : ℝ) :
  first_25_payment = 500 →
  remaining_payment = 600 →
  average_payment = 551.9230769230769 →
  ∃ n : ℕ, n = 52 ∧ 
    n * average_payment = 25 * first_25_payment + (n - 25) * remaining_payment :=
by sorry

end NUMINAMATH_CALUDE_total_installments_count_l1428_142860


namespace NUMINAMATH_CALUDE_difference_of_squares_l1428_142845

theorem difference_of_squares (x y : ℚ) 
  (h1 : x + y = 15/26) 
  (h2 : x - y = 2/65) : 
  x^2 - y^2 = 15/845 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1428_142845


namespace NUMINAMATH_CALUDE_product_inequality_with_sum_constraint_l1428_142883

theorem product_inequality_with_sum_constraint (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_constraint : x + y + z = 1) :
  (1 + 1/x) * (1 + 1/y) * (1 + 1/z) ≥ 64 ∧
  ((1 + 1/x) * (1 + 1/y) * (1 + 1/z) = 64 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_with_sum_constraint_l1428_142883


namespace NUMINAMATH_CALUDE_rectangles_bounded_by_lines_l1428_142836

/-- The number of rectangles bounded by p parallel lines and q perpendicular lines -/
def num_rectangles (p q : ℕ) : ℚ :=
  (p * q * (p - 1) * (q - 1)) / 4

/-- Theorem stating the number of rectangles bounded by p parallel lines and q perpendicular lines -/
theorem rectangles_bounded_by_lines (p q : ℕ) :
  num_rectangles p q = (p * q * (p - 1) * (q - 1)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_bounded_by_lines_l1428_142836


namespace NUMINAMATH_CALUDE_prob_sum_less_than_10_given_first_6_l1428_142812

/-- The probability that the sum of two dice is less than 10, given that the first die shows 6 -/
theorem prob_sum_less_than_10_given_first_6 :
  let outcomes : Finset ℕ := Finset.range 6
  let favorable_outcomes : Finset ℕ := Finset.filter (λ x => x + 6 < 10) outcomes
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_10_given_first_6_l1428_142812


namespace NUMINAMATH_CALUDE_prop_p_prop_q_l1428_142806

-- Define the set of real numbers excluding 1
def RealExcludingOne : Set ℝ := {x : ℝ | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioi 1)}

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Proposition p
theorem prop_p : ∀ a ∈ RealExcludingOne, log a 1 = 0 := by sorry

-- Proposition q
theorem prop_q : ∀ x : ℕ, x^3 ≥ x^2 := by sorry

end NUMINAMATH_CALUDE_prop_p_prop_q_l1428_142806


namespace NUMINAMATH_CALUDE_mass_percentage_H_is_correct_l1428_142849

/-- The mass percentage of H in a certain compound -/
def mass_percentage_H : ℝ := 1.69

/-- Theorem stating that the mass percentage of H is 1.69% -/
theorem mass_percentage_H_is_correct : mass_percentage_H = 1.69 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_H_is_correct_l1428_142849


namespace NUMINAMATH_CALUDE_fathers_age_l1428_142816

theorem fathers_age (father daughter : ℕ) 
  (h1 : father = 4 * daughter)
  (h2 : father + daughter + 10 = 50) : 
  father = 32 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l1428_142816


namespace NUMINAMATH_CALUDE_train_length_l1428_142899

/-- The length of a train given its speed and time to pass a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 10 → speed_kmh * (1000 / 3600) * time_s = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1428_142899


namespace NUMINAMATH_CALUDE_group_meal_cost_example_l1428_142884

def group_meal_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

theorem group_meal_cost_example : group_meal_cost 9 2 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_example_l1428_142884


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1428_142814

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ -1) :
  (2 * a^2 - 3) / (a + 1) - (a^2 - 2) / (a + 1) = a - 1 := by
  sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x / (x^2 - 4)) / (x / (4 - 2*x)) = -2 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1428_142814


namespace NUMINAMATH_CALUDE_dana_earnings_l1428_142846

def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

theorem dana_earnings : 
  hourly_rate * (friday_hours + saturday_hours + sunday_hours) = 286 := by
  sorry

end NUMINAMATH_CALUDE_dana_earnings_l1428_142846


namespace NUMINAMATH_CALUDE_square_with_semicircles_perimeter_l1428_142864

theorem square_with_semicircles_perimeter (π : Real) (h : π > 0) :
  let side_length := 2 / π
  let semicircle_radius := side_length / 2
  let semicircle_arc_length := π * semicircle_radius
  4 * semicircle_arc_length = 4 := by sorry

end NUMINAMATH_CALUDE_square_with_semicircles_perimeter_l1428_142864


namespace NUMINAMATH_CALUDE_coin_division_problem_l1428_142886

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 5) → 
  (n % 7 = 2) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 5 ∨ m % 7 ≠ 2)) →
  (n % 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l1428_142886


namespace NUMINAMATH_CALUDE_f_min_value_inequality_proof_l1428_142804

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f
theorem f_min_value : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 3 := by sorry

-- Theorem for the inequality
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq : a + b + c = 3) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_proof_l1428_142804


namespace NUMINAMATH_CALUDE_total_cost_plates_and_spoons_l1428_142823

theorem total_cost_plates_and_spoons :
  let num_plates : ℕ := 9
  let price_per_plate : ℚ := 2
  let num_spoons : ℕ := 4
  let price_per_spoon : ℚ := 3/2
  (num_plates : ℚ) * price_per_plate + (num_spoons : ℚ) * price_per_spoon = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_plates_and_spoons_l1428_142823


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1428_142881

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x) → 
  m = 6 ∨ m = -6 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1428_142881


namespace NUMINAMATH_CALUDE_more_science_than_math_books_l1428_142801

def total_budget : ℕ := 500
def math_books : ℕ := 4
def math_book_price : ℕ := 20
def science_book_price : ℕ := 10
def art_book_price : ℕ := 20
def music_book_cost : ℕ := 160

theorem more_science_than_math_books :
  ∃ (science_books : ℕ) (art_books : ℕ),
    science_books > math_books ∧
    art_books = 2 * math_books ∧
    total_budget = math_books * math_book_price + science_books * science_book_price + 
                   art_books * art_book_price + music_book_cost ∧
    science_books - math_books = 6 :=
by sorry

end NUMINAMATH_CALUDE_more_science_than_math_books_l1428_142801


namespace NUMINAMATH_CALUDE_counterexample_exists_l1428_142809

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1428_142809


namespace NUMINAMATH_CALUDE_total_homework_time_l1428_142840

def jacob_time : ℕ := 18

def greg_time (jacob_time : ℕ) : ℕ := jacob_time - 6

def patrick_time (greg_time : ℕ) : ℕ := 2 * greg_time - 4

def samantha_time (patrick_time : ℕ) : ℕ := (3 * patrick_time) / 2

theorem total_homework_time :
  jacob_time + greg_time jacob_time + patrick_time (greg_time jacob_time) + samantha_time (patrick_time (greg_time jacob_time)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_homework_time_l1428_142840


namespace NUMINAMATH_CALUDE_symmetry_across_origin_l1428_142868

/-- Given two points A and B in a 2D plane, where B is symmetrical to A with respect to the origin,
    this theorem proves that if A has coordinates (2, -6), then B has coordinates (-2, 6). -/
theorem symmetry_across_origin (A B : ℝ × ℝ) :
  A = (2, -6) → B = (-A.1, -A.2) → B = (-2, 6) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_across_origin_l1428_142868


namespace NUMINAMATH_CALUDE_martin_martina_ages_l1428_142892

/-- Martin's age -/
def martin_age : ℕ := 33

/-- Martina's age -/
def martina_age : ℕ := 22

/-- The condition from Martin's statement -/
def martin_condition (x y : ℕ) : Prop :=
  x = 3 * (y - (x - y))

/-- The condition from Martina's statement -/
def martina_condition (x y : ℕ) : Prop :=
  x + (x + (x - y)) = 77

theorem martin_martina_ages :
  martin_condition martin_age martina_age ∧
  martina_condition martin_age martina_age :=
by sorry

end NUMINAMATH_CALUDE_martin_martina_ages_l1428_142892


namespace NUMINAMATH_CALUDE_largest_divisor_of_fifth_power_minus_self_l1428_142858

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n^5 - n for all composite n -/
def LargestCommonDivisor : ℕ := 6

theorem largest_divisor_of_fifth_power_minus_self :
  ∀ n : ℕ, IsComposite n → (n^5 - n) % LargestCommonDivisor = 0 ∧
  ∀ k : ℕ, k > LargestCommonDivisor → ∃ m : ℕ, IsComposite m ∧ (m^5 - m) % k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_fifth_power_minus_self_l1428_142858


namespace NUMINAMATH_CALUDE_evaluate_g_l1428_142838

/-- The function g(x) = 3x^2 - 5x + 7 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

/-- Theorem: 3g(2) + 2g(-4) = 177 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-4) = 177 := by sorry

end NUMINAMATH_CALUDE_evaluate_g_l1428_142838


namespace NUMINAMATH_CALUDE_smallest_side_length_is_correct_l1428_142866

/-- Represents a triangle ABC with a point D on AC --/
structure TriangleABCD where
  -- The side length of the equilateral triangle
  side_length : ℕ
  -- The length of CD
  cd_length : ℕ
  -- Ensures that CD is not longer than AC
  h_cd_le_side : cd_length ≤ side_length

/-- The smallest possible side length of an equilateral triangle ABC 
    with a point D on AC such that BD is perpendicular to AC, 
    BD² = 65, and AC and CD are integers --/
def smallest_side_length : ℕ := 8

theorem smallest_side_length_is_correct (t : TriangleABCD) : 
  (t.side_length : ℝ)^2 / 4 + 65 = (t.side_length : ℝ)^2 →
  t.side_length ≥ smallest_side_length := by
  sorry

#check smallest_side_length_is_correct

end NUMINAMATH_CALUDE_smallest_side_length_is_correct_l1428_142866


namespace NUMINAMATH_CALUDE_equation_solutions_count_l1428_142822

theorem equation_solutions_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ (p : ℕ × ℕ) => (p.1 - 4)^2 - 35 = (p.2 - 3)^2) 
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l1428_142822


namespace NUMINAMATH_CALUDE_umbrella_boots_probability_l1428_142839

theorem umbrella_boots_probability
  (total_umbrellas : ℕ)
  (total_boots : ℕ)
  (prob_boots_and_umbrella : ℚ)
  (h1 : total_umbrellas = 40)
  (h2 : total_boots = 60)
  (h3 : prob_boots_and_umbrella = 1/3) :
  (prob_boots_and_umbrella * total_boots : ℚ) / total_umbrellas = 1/2 :=
sorry

end NUMINAMATH_CALUDE_umbrella_boots_probability_l1428_142839


namespace NUMINAMATH_CALUDE_sqrt_three_seven_plus_four_sqrt_three_l1428_142894

theorem sqrt_three_seven_plus_four_sqrt_three :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_seven_plus_four_sqrt_three_l1428_142894


namespace NUMINAMATH_CALUDE_quadratic_cubic_relation_l1428_142898

theorem quadratic_cubic_relation (x : ℝ) : x^2 + x - 1 = 0 → 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_cubic_relation_l1428_142898


namespace NUMINAMATH_CALUDE_sample_size_major_C_l1428_142800

/-- Represents the number of students in each major -/
structure CollegeMajors where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Calculates the total number of students across all majors -/
def totalStudents (majors : CollegeMajors) : Nat :=
  majors.A + majors.B + majors.C + majors.D

/-- Calculates the number of students to be sampled from a specific major -/
def sampleSize (majors : CollegeMajors) (totalSample : Nat) (majorSize : Nat) : Nat :=
  (majorSize * totalSample) / totalStudents majors

/-- Theorem: The number of students to be sampled from major C is 16 -/
theorem sample_size_major_C :
  let majors : CollegeMajors := { A := 150, B := 150, C := 400, D := 300 }
  let totalSample : Nat := 40
  sampleSize majors totalSample majors.C = 16 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_major_C_l1428_142800


namespace NUMINAMATH_CALUDE_black_car_speed_proof_l1428_142810

/-- The speed of the red car in miles per hour -/
def red_car_speed : ℝ := 10

/-- The initial distance between the cars in miles -/
def initial_distance : ℝ := 20

/-- The time it takes for the black car to overtake the red car in hours -/
def overtake_time : ℝ := 0.5

/-- The speed of the black car in miles per hour -/
def black_car_speed : ℝ := 50

theorem black_car_speed_proof :
  red_car_speed * overtake_time + initial_distance = black_car_speed * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_black_car_speed_proof_l1428_142810


namespace NUMINAMATH_CALUDE_fancy_shape_charge_proof_l1428_142856

/-- The cost to trim up a single boxwood -/
def trim_cost : ℚ := 5

/-- The total number of boxwoods -/
def total_boxwoods : ℕ := 30

/-- The number of boxwoods to be trimmed into fancy shapes -/
def fancy_boxwoods : ℕ := 4

/-- The total charge for the job -/
def total_charge : ℚ := 210

/-- The charge for trimming a boxwood into a fancy shape -/
def fancy_shape_charge : ℚ := 15

theorem fancy_shape_charge_proof :
  fancy_shape_charge * fancy_boxwoods + trim_cost * total_boxwoods = total_charge :=
sorry

end NUMINAMATH_CALUDE_fancy_shape_charge_proof_l1428_142856


namespace NUMINAMATH_CALUDE_two_digit_integers_mod_seven_l1428_142879

theorem two_digit_integers_mod_seven : 
  (Finset.filter (fun n => n ≥ 10 ∧ n < 100 ∧ n % 7 = 3) (Finset.range 100)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integers_mod_seven_l1428_142879


namespace NUMINAMATH_CALUDE_triangle_side_length_l1428_142802

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 1 → c = Real.sqrt 3 → A = π / 6 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  b = 1 ∨ b = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1428_142802


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1428_142878

theorem interest_rate_calculation (simple_interest principal time_period : ℚ) 
  (h1 : simple_interest = 4016.25)
  (h2 : time_period = 5)
  (h3 : principal = 80325) :
  simple_interest * 100 / (principal * time_period) = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1428_142878


namespace NUMINAMATH_CALUDE_russel_carousel_rides_l1428_142817

/-- The number of times Russel rode the carousel -/
def carousel_rides (total_tickets jen_games shooting_cost carousel_cost : ℕ) : ℕ :=
  (total_tickets - jen_games * shooting_cost) / carousel_cost

/-- Proof that Russel rode the carousel 3 times -/
theorem russel_carousel_rides : 
  carousel_rides 19 2 5 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_russel_carousel_rides_l1428_142817


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_half_right_angle_l1428_142834

/-- Given an angle that is half of 90 degrees, prove that the degree measure of
    the supplement of its complement is 135 degrees. -/
theorem supplement_of_complement_of_half_right_angle :
  let α : ℝ := 90 / 2
  let complement_α : ℝ := 90 - α
  let supplement_complement_α : ℝ := 180 - complement_α
  supplement_complement_α = 135 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_half_right_angle_l1428_142834


namespace NUMINAMATH_CALUDE_polynomial_value_l1428_142859

theorem polynomial_value (x y : ℝ) (h : x - 2*y + 3 = 8) : x - 2*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1428_142859


namespace NUMINAMATH_CALUDE_bryan_bookshelves_l1428_142869

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 27

/-- The total number of books Bryan has -/
def total_books : ℕ := 621

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := total_books / books_per_shelf

theorem bryan_bookshelves : num_bookshelves = 23 := by
  sorry

end NUMINAMATH_CALUDE_bryan_bookshelves_l1428_142869


namespace NUMINAMATH_CALUDE_highest_a_divisible_by_8_first_digit_is_three_l1428_142830

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem highest_a_divisible_by_8 :
  ∃ (a : ℕ), a ≤ 9 ∧
  is_divisible_by_8 (3 * 100000 + a * 1000 + 524) ∧
  (∀ (b : ℕ), b ≤ 9 → b > a →
    ¬is_divisible_by_8 (3 * 100000 + b * 1000 + 524)) ∧
  a = 8 :=
sorry

theorem first_digit_is_three :
  ∀ (a : ℕ), a ≤ 9 →
  (3 * 100000 + a * 1000 + 524) / 100000 = 3 :=
sorry

end NUMINAMATH_CALUDE_highest_a_divisible_by_8_first_digit_is_three_l1428_142830


namespace NUMINAMATH_CALUDE_games_expenditure_l1428_142821

def allowance : ℚ := 48

def clothes_fraction : ℚ := 1/4
def books_fraction : ℚ := 1/3
def snacks_fraction : ℚ := 1/6

def amount_on_games : ℚ := allowance - (clothes_fraction * allowance + books_fraction * allowance + snacks_fraction * allowance)

theorem games_expenditure : amount_on_games = 12 := by
  sorry

end NUMINAMATH_CALUDE_games_expenditure_l1428_142821


namespace NUMINAMATH_CALUDE_distinct_triangles_in_square_pyramid_l1428_142882

-- Define the number of vertices in a square pyramid
def num_vertices : ℕ := 5

-- Define the number of vertices needed to form a triangle
def vertices_per_triangle : ℕ := 3

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ := 
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem distinct_triangles_in_square_pyramid :
  combinations num_vertices vertices_per_triangle = 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_square_pyramid_l1428_142882


namespace NUMINAMATH_CALUDE_taimour_painting_time_l1428_142875

theorem taimour_painting_time (jamshid_rate taimour_rate : ℝ) 
  (h1 : jamshid_rate = 2 * taimour_rate) 
  (h2 : jamshid_rate + taimour_rate = 1 / 6) : 
  taimour_rate = 1 / 18 :=
by sorry

end NUMINAMATH_CALUDE_taimour_painting_time_l1428_142875


namespace NUMINAMATH_CALUDE_consecutive_divisible_numbers_l1428_142877

theorem consecutive_divisible_numbers :
  ∃ n : ℕ, 100 ≤ n ∧ n < 200 ∧ 
    3 ∣ n ∧ 5 ∣ (n + 1) ∧ 7 ∣ (n + 2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_divisible_numbers_l1428_142877


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_second_greater_first_l1428_142815

/-- A geometric sequence with positive first term -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IsIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) > s n

theorem geometric_sequence_increasing_iff_second_greater_first (seq : GeometricSequence) :
  (seq.a 2 > seq.a 1) ↔ IsIncreasing seq.a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_second_greater_first_l1428_142815


namespace NUMINAMATH_CALUDE_cubic_equation_with_complex_root_l1428_142842

theorem cubic_equation_with_complex_root (k : ℝ) : 
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  (k = -1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_complex_root_l1428_142842


namespace NUMINAMATH_CALUDE_ratio_equality_l1428_142807

theorem ratio_equality (x : ℝ) : (1 : ℝ) / 3 = (5 : ℝ) / (3 * x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1428_142807


namespace NUMINAMATH_CALUDE_polar_bear_trout_consumption_l1428_142848

/-- The amount of fish eaten daily by the polar bear -/
def total_fish : ℝ := 0.6

/-- The amount of salmon eaten daily by the polar bear -/
def salmon : ℝ := 0.4

/-- The amount of trout eaten daily by the polar bear -/
def trout : ℝ := total_fish - salmon

theorem polar_bear_trout_consumption : trout = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_trout_consumption_l1428_142848


namespace NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_parallel_l1428_142847

/-- A type representing lines in three-dimensional space -/
structure Line3D where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Parallel relation between two lines in 3D space -/
def parallel (l m : Line3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines in 3D space -/
def perpendicular (l m : Line3D) : Prop :=
  sorry

/-- Theorem: If two lines are parallel to the same line, they are parallel to each other -/
theorem parallel_transitive (l m n : Line3D) :
  parallel l m → parallel m n → parallel l n :=
sorry

/-- Theorem: If a line is perpendicular to one of two parallel lines, it is perpendicular to the other -/
theorem perpendicular_to_parallel (l m n : Line3D) :
  perpendicular l m → parallel m n → perpendicular l n :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_perpendicular_to_parallel_l1428_142847


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1428_142880

def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / k + y^2 / (k - 3) = 1 ∧ k ≠ 0 ∧ k ≠ 3

theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ 0 < k ∧ k < 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1428_142880


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_2012_l1428_142843

theorem units_digit_of_2_pow_2012 : ∃ n : ℕ, 2^2012 ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_2012_l1428_142843


namespace NUMINAMATH_CALUDE_candies_equalization_l1428_142832

theorem candies_equalization (basket_a basket_b added : ℕ) : 
  basket_a = 8 → basket_b = 17 → basket_a + added = basket_b → added = 9 := by
sorry

end NUMINAMATH_CALUDE_candies_equalization_l1428_142832


namespace NUMINAMATH_CALUDE_overtime_hours_l1428_142897

theorem overtime_hours (regular_rate : ℝ) (regular_hours : ℝ) (total_pay : ℝ) :
  regular_rate = 3 →
  regular_hours = 40 →
  total_pay = 198 →
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate = 13 := by sorry

end NUMINAMATH_CALUDE_overtime_hours_l1428_142897


namespace NUMINAMATH_CALUDE_twenty_customers_without_fish_l1428_142893

/-- Represents the fish market scenario -/
structure FishMarket where
  total_customers : ℕ
  num_tuna : ℕ
  tuna_weight : ℕ
  customer_request : ℕ

/-- Calculates the number of customers who will go home without fish -/
def customers_without_fish (market : FishMarket) : ℕ :=
  market.total_customers - (market.num_tuna * market.tuna_weight / market.customer_request)

/-- Theorem stating that in the given scenario, 20 customers will go home without fish -/
theorem twenty_customers_without_fish :
  let market : FishMarket := {
    total_customers := 100,
    num_tuna := 10,
    tuna_weight := 200,
    customer_request := 25
  }
  customers_without_fish market = 20 := by sorry

end NUMINAMATH_CALUDE_twenty_customers_without_fish_l1428_142893


namespace NUMINAMATH_CALUDE_bacteria_growth_rate_l1428_142813

/-- The growth rate of a bacteria colony -/
def growth_rate : ℝ := 2

/-- The number of days for a single colony to reach the habitat's limit -/
def single_colony_days : ℕ := 22

/-- The number of days for two colonies to reach the habitat's limit -/
def double_colony_days : ℕ := 21

/-- The theorem stating the growth rate of the bacteria colony -/
theorem bacteria_growth_rate :
  (growth_rate ^ single_colony_days : ℝ) = 2 * (growth_rate ^ double_colony_days : ℝ) :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_rate_l1428_142813


namespace NUMINAMATH_CALUDE_modulus_of_z_l1428_142808

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 * I) : abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1428_142808


namespace NUMINAMATH_CALUDE_father_son_age_sum_l1428_142826

theorem father_son_age_sum :
  ∀ (F S : ℕ),
  F > 0 ∧ S > 0 →
  F / S = 7 / 4 →
  (F + 10) / (S + 10) = 5 / 3 →
  F + S = 220 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_l1428_142826


namespace NUMINAMATH_CALUDE_correct_employee_count_l1428_142820

/-- The number of employees in John's company --/
def number_of_employees : ℕ := 85

/-- The cost of each turkey in dollars --/
def cost_per_turkey : ℕ := 25

/-- The total amount spent on turkeys in dollars --/
def total_spent : ℕ := 2125

/-- Theorem stating that the number of employees is correct given the conditions --/
theorem correct_employee_count :
  number_of_employees * cost_per_turkey = total_spent :=
by sorry

end NUMINAMATH_CALUDE_correct_employee_count_l1428_142820


namespace NUMINAMATH_CALUDE_bad_carrots_l1428_142824

theorem bad_carrots (vanessa_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  vanessa_carrots = 17 → mom_carrots = 14 → good_carrots = 24 → 
  vanessa_carrots + mom_carrots - good_carrots = 7 := by
sorry

end NUMINAMATH_CALUDE_bad_carrots_l1428_142824


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l1428_142850

theorem complex_sum_of_parts (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 : ℂ) + 2 * i = a + b * i) : a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l1428_142850


namespace NUMINAMATH_CALUDE_unique_base_solution_l1428_142871

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (λ (i, d) acc => acc + d * b^i) 0

/-- The equation 142₂ + 163₂ = 315₂ holds in base b --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [1, 4, 2] b + to_decimal [1, 6, 3] b = to_decimal [3, 1, 5] b

theorem unique_base_solution :
  ∃! b : Nat, b > 6 ∧ equation_holds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l1428_142871


namespace NUMINAMATH_CALUDE_prod_mod_seven_l1428_142857

theorem prod_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_prod_mod_seven_l1428_142857


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l1428_142895

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 10) :
  (∃ (m' n' : ℕ+), Nat.gcd m' n' = 10 ∧ Nat.gcd (8 * m') (12 * n') = 40) ∧
  (∀ (m'' n'' : ℕ+), Nat.gcd m'' n'' = 10 → Nat.gcd (8 * m'') (12 * n'') ≥ 40) :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l1428_142895


namespace NUMINAMATH_CALUDE_factor_expression_l1428_142861

theorem factor_expression (x y : ℝ) : 100 - 25 * x^2 + 16 * y^2 = (10 - 5*x + 4*y) * (10 + 5*x - 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1428_142861


namespace NUMINAMATH_CALUDE_x_fourth_gt_x_minus_half_l1428_142889

theorem x_fourth_gt_x_minus_half (x : ℝ) : x^4 - x + (1/2 : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_gt_x_minus_half_l1428_142889


namespace NUMINAMATH_CALUDE_country_club_cost_l1428_142873

/-- Calculates the amount one person pays for the first year of a country club membership,
    given they pay half the total cost for a group. -/
theorem country_club_cost
  (num_people : ℕ)
  (joining_fee : ℕ)
  (monthly_cost : ℕ)
  (months_in_year : ℕ)
  (h_num_people : num_people = 4)
  (h_joining_fee : joining_fee = 4000)
  (h_monthly_cost : monthly_cost = 1000)
  (h_months_in_year : months_in_year = 12) :
  (num_people * joining_fee + num_people * monthly_cost * months_in_year) / 2 = 32000 := by
  sorry

#check country_club_cost

end NUMINAMATH_CALUDE_country_club_cost_l1428_142873
