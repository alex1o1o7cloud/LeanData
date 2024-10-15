import Mathlib

namespace NUMINAMATH_CALUDE_pineapple_weight_l2090_209029

theorem pineapple_weight (P : ℝ) 
  (h1 : P > 0)
  (h2 : P / 6 + 2 / 5 * (5 / 6 * P) + 2 / 3 * (P / 2) + 120 = P) : 
  P = 720 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_weight_l2090_209029


namespace NUMINAMATH_CALUDE_ghee_mixture_original_quantity_l2090_209063

/-- Proves that the original quantity of a ghee mixture is 10 kg given specific conditions -/
theorem ghee_mixture_original_quantity :
  ∀ (x : ℝ),
  (0.6 * x = x - 0.4 * x) →  -- 60% pure ghee, 40% vanaspati in original mixture
  (0.2 * (x + 10) = 0.4 * x) →  -- 20% vanaspati after adding 10 kg pure ghee
  x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ghee_mixture_original_quantity_l2090_209063


namespace NUMINAMATH_CALUDE_square_land_area_l2090_209000

/-- A square land plot with side length 30 units has an area of 900 square units. -/
theorem square_land_area (side_length : ℝ) (h1 : side_length = 30) :
  side_length * side_length = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l2090_209000


namespace NUMINAMATH_CALUDE_print_shop_cost_difference_l2090_209019

/-- The cost difference between two print shops for a given number of copies -/
def cost_difference (price_x price_y : ℚ) (num_copies : ℕ) : ℚ :=
  (price_y - price_x) * num_copies

/-- Theorem stating the cost difference between print shops Y and X for 40 copies -/
theorem print_shop_cost_difference :
  cost_difference (120/100) (170/100) 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_cost_difference_l2090_209019


namespace NUMINAMATH_CALUDE_min_people_to_ask_for_hat_color_l2090_209057

/-- Represents the minimum number of people to ask to ensure a majority of truthful answers -/
def min_people_to_ask (knights : ℕ) (civilians : ℕ) : ℕ :=
  civilians + (civilians + 1)

/-- Theorem stating the minimum number of people to ask in the given scenario -/
theorem min_people_to_ask_for_hat_color (knights : ℕ) (civilians : ℕ) 
  (h1 : knights = 50) (h2 : civilians = 15) :
  min_people_to_ask knights civilians = 31 := by
  sorry

#eval min_people_to_ask 50 15

end NUMINAMATH_CALUDE_min_people_to_ask_for_hat_color_l2090_209057


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l2090_209021

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 :=
by sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l2090_209021


namespace NUMINAMATH_CALUDE_cab_driver_income_l2090_209013

/-- Given a cab driver's income for 5 days, prove that the income on the third day is $450 -/
theorem cab_driver_income (income : Fin 5 → ℕ) 
  (day1 : income 0 = 600)
  (day2 : income 1 = 250)
  (day4 : income 3 = 400)
  (day5 : income 4 = 800)
  (avg_income : (income 0 + income 1 + income 2 + income 3 + income 4) / 5 = 500) :
  income 2 = 450 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l2090_209013


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2090_209051

theorem arithmetic_sequence_product (a : ℝ) (d : ℝ) : 
  (a + 6 * d = 20) → (d = 2) → (a * (a + d) = 80) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2090_209051


namespace NUMINAMATH_CALUDE_pens_and_pencils_equation_system_l2090_209099

theorem pens_and_pencils_equation_system (x y : ℕ) : 
  (x + y = 30 ∧ x = 2 * y - 3) ↔ 
  (x + y = 30 ∧ x = 2 * y - 3 ∧ x < 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_pens_and_pencils_equation_system_l2090_209099


namespace NUMINAMATH_CALUDE_quadratic_residue_mod_prime_l2090_209058

theorem quadratic_residue_mod_prime (p : Nat) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∃ a : Int, (a ^ 2) % p = (p - 1) % p) ↔ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_residue_mod_prime_l2090_209058


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2090_209067

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 14*x + 24

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = {-4, -3, 3} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2090_209067


namespace NUMINAMATH_CALUDE_line_equation_proof_l2090_209038

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if two lines are parallel
def linesParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Theorem statement
theorem line_equation_proof (l : Line) (p : Point) (given_line : Line) :
  pointOnLine p l ∧ 
  p = Point.mk 0 3 ∧ 
  linesParallel l given_line ∧ 
  given_line = Line.mk 1 (-1) (-1) →
  l = Line.mk 1 (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2090_209038


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l2090_209061

theorem triangle_area_from_squares (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 121) (h3 : c^2 = 169)
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 44 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l2090_209061


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l2090_209020

/-- The volume of a rectangular parallelepiped with given diagonal, angle, and base perimeter. -/
theorem parallelepiped_volume (l P α : ℝ) (hl : l > 0) (hP : P > 0) (hα : 0 < α ∧ α < π / 2) :
  ∃ V : ℝ, V = (l * (P^2 - 4 * l^2 * Real.sin α ^ 2) * Real.cos α) / 8 ∧
    V > 0 ∧
    ∀ (x y h : ℝ),
      x > 0 → y > 0 → h > 0 →
      x + y = P / 2 →
      x^2 + y^2 = l^2 * Real.sin α ^ 2 →
      h = l * Real.cos α →
      V = x * y * h :=
by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l2090_209020


namespace NUMINAMATH_CALUDE_max_product_decomposition_l2090_209045

theorem max_product_decomposition (a : ℝ) (ha : a > 0) :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y = a →
  x * y ≤ (a / 2) * (a / 2) ∧
  (x * y = (a / 2) * (a / 2) ↔ x = a / 2 ∧ y = a / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_product_decomposition_l2090_209045


namespace NUMINAMATH_CALUDE_complement_of_union_l2090_209015

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2090_209015


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2090_209091

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l2090_209091


namespace NUMINAMATH_CALUDE_triple_area_right_triangle_l2090_209049

/-- Given a right triangle with hypotenuse a+b and legs a and b, 
    the area of a triangle that is three times the area of this right triangle is 3/2ab. -/
theorem triple_area_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  3 * (1/2 * a * b) = 3/2 * a * b := by sorry

end NUMINAMATH_CALUDE_triple_area_right_triangle_l2090_209049


namespace NUMINAMATH_CALUDE_angela_insect_count_l2090_209046

theorem angela_insect_count (dean_insects jacob_insects angela_insects : ℕ) : 
  dean_insects = 30 →
  jacob_insects = 5 * dean_insects →
  angela_insects = jacob_insects / 2 →
  angela_insects = 75 := by
  sorry

end NUMINAMATH_CALUDE_angela_insect_count_l2090_209046


namespace NUMINAMATH_CALUDE_drink_expense_l2090_209076

def initial_amount : ℝ := 9
def final_amount : ℝ := 6
def additional_expense : ℝ := 1.25

theorem drink_expense : 
  initial_amount - final_amount - additional_expense = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_drink_expense_l2090_209076


namespace NUMINAMATH_CALUDE_average_marks_l2090_209088

theorem average_marks (avg_five : ℝ) (sixth_mark : ℝ) : 
  avg_five = 74 → sixth_mark = 80 → 
  ((avg_five * 5 + sixth_mark) / 6 : ℝ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l2090_209088


namespace NUMINAMATH_CALUDE_fifteenth_even_multiple_of_3_l2090_209030

/-- The nth positive even integer that is a multiple of 3 -/
def evenMultipleOf3 (n : ℕ) : ℕ := 6 * n

/-- The 15th positive even integer that is a multiple of 3 is 90 -/
theorem fifteenth_even_multiple_of_3 : evenMultipleOf3 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_even_multiple_of_3_l2090_209030


namespace NUMINAMATH_CALUDE_high_five_problem_l2090_209075

theorem high_five_problem (n : ℕ) (h : n > 0) :
  (∀ (person : Fin n), (person.val < n → 2 * 2021 = n - 1)) →
  (n = 4043 ∧ Nat.choose n 3 = 11024538580) := by
  sorry

end NUMINAMATH_CALUDE_high_five_problem_l2090_209075


namespace NUMINAMATH_CALUDE_garden_area_l2090_209048

theorem garden_area (total_posts : ℕ) (post_spacing : ℕ) 
  (h1 : total_posts = 20)
  (h2 : post_spacing = 4)
  (h3 : ∃ (short_posts long_posts : ℕ), 
    short_posts > 1 ∧ 
    long_posts > 1 ∧ 
    short_posts + long_posts = total_posts / 2 + 2 ∧ 
    long_posts = 2 * short_posts) :
  ∃ (width length : ℕ), 
    width * length = 336 ∧ 
    width = post_spacing * (short_posts - 1) ∧ 
    length = post_spacing * (long_posts - 1) :=
by sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l2090_209048


namespace NUMINAMATH_CALUDE_line_point_distance_l2090_209025

/-- Given five points O, A, B, C, D on a line, with Q and P also on the line,
    prove that OP = 2q under the given conditions. -/
theorem line_point_distance (a b c d q : ℝ) : 
  ∀ (x : ℝ), 
  (0 < a) → (a < b) → (b < c) → (c < d) →  -- Points are in order
  (0 < q) → (q < d) →  -- Q is on the line
  (b ≤ x) → (x ≤ c) →  -- P is between B and C
  ((a - x) / (x - d) = (b - x) / (x - c)) →  -- AP : PD = BP : PC
  (x = 2 * q) →  -- P is twice as far from O as Q is
  x = 2 * q := by
  sorry

end NUMINAMATH_CALUDE_line_point_distance_l2090_209025


namespace NUMINAMATH_CALUDE_hexagon_side_length_equals_square_side_l2090_209086

/-- Represents a hexagon with side length y -/
structure Hexagon where
  y : ℝ

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with side length s -/
structure Square where
  s : ℝ

/-- Given a 12 × 12 rectangle divided into two congruent hexagons that can form a square without overlap,
    the side length of each hexagon is 12. -/
theorem hexagon_side_length_equals_square_side 
  (rect : Rectangle)
  (hex1 hex2 : Hexagon)
  (sq : Square)
  (h1 : rect.length = 12 ∧ rect.width = 12)
  (h2 : hex1 = hex2)
  (h3 : rect.length * rect.width = sq.s * sq.s)
  (h4 : hex1.y = sq.s) :
  hex1.y = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_equals_square_side_l2090_209086


namespace NUMINAMATH_CALUDE_one_correct_statement_l2090_209096

-- Define a sequence as a function from natural numbers to real numbers
def Sequence := ℕ → ℝ

-- Statement 1: A sequence represented graphically appears as a group of isolated points
def graphical_representation (s : Sequence) : Prop :=
  ∀ n : ℕ, ∃ ε > 0, ∀ m : ℕ, m ≠ n → |s m - s n| > ε

-- Statement 2: The terms of a sequence are finite
def finite_terms (s : Sequence) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n > N → s n = 0

-- Statement 3: If a sequence is decreasing, then the sequence must be finite
def decreasing_implies_finite (s : Sequence) : Prop :=
  (∀ n : ℕ, s (n + 1) ≤ s n) → finite_terms s

-- Theorem stating that only one of the above statements is correct
theorem one_correct_statement :
  (∀ s : Sequence, graphical_representation s) ∧
  (∃ s : Sequence, ¬finite_terms s) ∧
  (∃ s : Sequence, (∀ n : ℕ, s (n + 1) ≤ s n) ∧ ¬finite_terms s) :=
sorry

end NUMINAMATH_CALUDE_one_correct_statement_l2090_209096


namespace NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l2090_209066

theorem angle_with_special_complement_supplement : ∀ x : ℝ,
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by sorry

end NUMINAMATH_CALUDE_angle_with_special_complement_supplement_l2090_209066


namespace NUMINAMATH_CALUDE_product_of_sums_geq_one_l2090_209073

theorem product_of_sums_geq_one (a b c d : ℝ) 
  (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := by sorry

end NUMINAMATH_CALUDE_product_of_sums_geq_one_l2090_209073


namespace NUMINAMATH_CALUDE_line_shift_theorem_l2090_209005

-- Define the original line
def original_line (x : ℝ) : ℝ := -2 * x + 1

-- Define the shift amount
def shift : ℝ := 2

-- Define the shifted line
def shifted_line (x : ℝ) : ℝ := original_line (x + shift)

-- Theorem statement
theorem line_shift_theorem :
  ∀ x : ℝ, shifted_line x = -2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_line_shift_theorem_l2090_209005


namespace NUMINAMATH_CALUDE_problem_solution_l2090_209027

theorem problem_solution (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x * y = 3) 
  (h3 : x^3 - x^2 - 4*x + 4 = 0) 
  (h4 : y^3 - y^2 - 4*y + 4 = 0) : 
  x + x^3/y^2 + y^3/x^2 + y = 174 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2090_209027


namespace NUMINAMATH_CALUDE_fort_men_count_l2090_209031

/-- Represents the initial number of men in the fort -/
def initial_men : ℕ := 150

/-- Represents the number of days the initial provision would last -/
def initial_days : ℕ := 45

/-- Represents the number of days after which some men left -/
def days_before_leaving : ℕ := 10

/-- Represents the number of men who left the fort -/
def men_who_left : ℕ := 25

/-- Represents the number of days the remaining food lasted -/
def remaining_days : ℕ := 42

/-- Theorem stating that given the conditions, the initial number of men in the fort was 150 -/
theorem fort_men_count :
  initial_men * (initial_days - days_before_leaving) = 
  (initial_men - men_who_left) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_fort_men_count_l2090_209031


namespace NUMINAMATH_CALUDE_expression_simplification_l2090_209023

theorem expression_simplification :
  1 + (1 : ℝ) / (1 + Real.sqrt 2) - 1 / (1 - Real.sqrt 5) =
  1 + (-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2090_209023


namespace NUMINAMATH_CALUDE_alternating_number_composite_l2090_209033

def alternating_number (k : ℕ) : ℕ := 
  (10^(2*k+1) - 1) / 99

theorem alternating_number_composite (k : ℕ) (h : k ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ alternating_number k = a * b :=
sorry

end NUMINAMATH_CALUDE_alternating_number_composite_l2090_209033


namespace NUMINAMATH_CALUDE_marble_probability_difference_l2090_209044

theorem marble_probability_difference :
  let total_marbles : ℕ := 4000
  let red_marbles : ℕ := 1500
  let black_marbles : ℕ := 2500
  let p_same : ℚ := (red_marbles.choose 2 + black_marbles.choose 2) / total_marbles.choose 2
  let p_different : ℚ := (red_marbles * black_marbles) / total_marbles.choose 2
  |p_same - p_different| = 3 / 50 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l2090_209044


namespace NUMINAMATH_CALUDE_fill_675_cans_l2090_209008

/-- A machine that fills paint cans at a specific rate -/
structure PaintMachine where
  cans_per_batch : ℕ
  minutes_per_batch : ℕ

/-- Calculate the time needed to fill a given number of cans -/
def time_to_fill (machine : PaintMachine) (total_cans : ℕ) : ℕ := 
  (total_cans * machine.minutes_per_batch + machine.cans_per_batch - 1) / machine.cans_per_batch

/-- Theorem: The given machine takes 36 minutes to fill 675 cans -/
theorem fill_675_cans (machine : PaintMachine) 
  (h1 : machine.cans_per_batch = 150) 
  (h2 : machine.minutes_per_batch = 8) : 
  time_to_fill machine 675 = 36 := by sorry

end NUMINAMATH_CALUDE_fill_675_cans_l2090_209008


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l2090_209053

theorem complex_modulus_equation (t : ℝ) : 
  t > 0 → Complex.abs (8 + 3 * t * Complex.I) = 13 → t = Real.sqrt 105 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l2090_209053


namespace NUMINAMATH_CALUDE_van_rental_cost_equation_l2090_209062

theorem van_rental_cost_equation (x : ℝ) (h : x > 2) :
  180 / (x - 2) - 180 / x = 3 :=
sorry

end NUMINAMATH_CALUDE_van_rental_cost_equation_l2090_209062


namespace NUMINAMATH_CALUDE_student_weights_l2090_209003

/-- Theorem: Total and average weight of students
Given 10 students with a base weight and weight deviations, 
prove the total weight and average weight. -/
theorem student_weights (base_weight : ℝ) (weight_deviations : List ℝ) : 
  base_weight = 50 ∧ 
  weight_deviations = [2, 3, -7.5, -3, 5, -8, 3.5, 4.5, 8, -1.5] →
  (List.sum weight_deviations + 10 * base_weight = 509) ∧
  ((List.sum weight_deviations + 10 * base_weight) / 10 = 50.9) := by
  sorry

#check student_weights

end NUMINAMATH_CALUDE_student_weights_l2090_209003


namespace NUMINAMATH_CALUDE_coin_placement_coloring_l2090_209074

theorem coin_placement_coloring (n : ℕ) (h1 : 1 < n) (h2 : n < 2010) :
  (∃ (coloring : Fin 2010 → Fin n) (initial_positions : Fin n → Fin 2010),
    ∀ (t : ℕ) (i j : Fin n),
      i ≠ j →
      coloring ((initial_positions i + t) % 2010) ≠
      coloring ((initial_positions j + t) % 2010)) ↔
  2010 % n = 0 :=
sorry

end NUMINAMATH_CALUDE_coin_placement_coloring_l2090_209074


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2090_209041

def C : Set Nat := {33, 35, 37, 39, 41}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, (Nat.minFac n ≤ Nat.minFac m)

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 33 C ∧ has_smallest_prime_factor 39 C ∧
  ∀ x ∈ C, has_smallest_prime_factor x C → (x = 33 ∨ x = 39) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2090_209041


namespace NUMINAMATH_CALUDE_sum_of_xy_l2090_209024

theorem sum_of_xy (x y : ℕ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x < 30) 
  (h4 : y < 30) 
  (h5 : x + y + x * y = 119) : 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2090_209024


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_inradius_l2090_209079

theorem right_triangle_arithmetic_progression_inradius (a b c d : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a < b ∧ b < c →  -- Ordered side lengths
  a^2 + b^2 = c^2 →  -- Right triangle (Pythagorean theorem)
  b = a + d ∧ c = a + 2*d →  -- Arithmetic progression
  d = (a*b*c) / (a + b + c)  -- d equals inradius
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_inradius_l2090_209079


namespace NUMINAMATH_CALUDE_percentage_passed_l2090_209026

def total_students : ℕ := 800
def failed_students : ℕ := 520

theorem percentage_passed : 
  (((total_students - failed_students) : ℚ) / total_students) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_l2090_209026


namespace NUMINAMATH_CALUDE_contrapositive_false_l2090_209072

theorem contrapositive_false : 
  ¬(∀ x : ℝ, x^2 - 1 = 0 → x = 1) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_false_l2090_209072


namespace NUMINAMATH_CALUDE_kim_easy_round_answers_l2090_209093

/-- Represents the number of points for each round in the math contest -/
structure ContestPoints where
  easy : ℕ
  average : ℕ
  hard : ℕ

/-- Represents the number of correct answers for each round -/
structure ContestAnswers where
  easy : ℕ
  average : ℕ
  hard : ℕ

def totalPoints (points : ContestPoints) (answers : ContestAnswers) : ℕ :=
  points.easy * answers.easy + points.average * answers.average + points.hard * answers.hard

theorem kim_easy_round_answers 
  (points : ContestPoints) 
  (answers : ContestAnswers) 
  (h1 : points.easy = 2) 
  (h2 : points.average = 3) 
  (h3 : points.hard = 5)
  (h4 : answers.average = 2)
  (h5 : answers.hard = 4)
  (h6 : totalPoints points answers = 38) : 
  answers.easy = 6 := by
sorry

end NUMINAMATH_CALUDE_kim_easy_round_answers_l2090_209093


namespace NUMINAMATH_CALUDE_purple_candies_count_l2090_209068

/-- The number of purple candies in a box of rainbow nerds -/
def purple_candies : ℕ := 10

/-- The number of yellow candies in a box of rainbow nerds -/
def yellow_candies : ℕ := purple_candies + 4

/-- The number of green candies in a box of rainbow nerds -/
def green_candies : ℕ := yellow_candies - 2

/-- The total number of candies in the box -/
def total_candies : ℕ := 36

/-- Theorem stating that the number of purple candies is 10 -/
theorem purple_candies_count : 
  purple_candies = 10 ∧ 
  yellow_candies = purple_candies + 4 ∧ 
  green_candies = yellow_candies - 2 ∧ 
  purple_candies + yellow_candies + green_candies = total_candies :=
by sorry

end NUMINAMATH_CALUDE_purple_candies_count_l2090_209068


namespace NUMINAMATH_CALUDE_unique_divisor_property_l2090_209022

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_divisor_property : ∃! n : ℕ, n > 0 ∧ n = 100 * divisor_count n :=
  sorry

end NUMINAMATH_CALUDE_unique_divisor_property_l2090_209022


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l2090_209018

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 3 * a + b = a^2 + a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3 * x + y = x^2 + x * y → 2 * x + y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l2090_209018


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2090_209037

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → ∃ (a b : ℕ), a^2 - b^2 = 187 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 205 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2090_209037


namespace NUMINAMATH_CALUDE_remaining_bottle_caps_l2090_209070

-- Define the initial number of bottle caps
def initial_caps : ℕ := 34

-- Define the number of bottle caps eaten
def eaten_caps : ℕ := 8

-- Theorem to prove
theorem remaining_bottle_caps : initial_caps - eaten_caps = 26 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bottle_caps_l2090_209070


namespace NUMINAMATH_CALUDE_stratified_sampling_arts_students_l2090_209035

theorem stratified_sampling_arts_students 
  (total_students : ℕ) 
  (arts_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1000)
  (h2 : arts_students = 200)
  (h3 : sample_size = 100) :
  (arts_students : ℚ) / total_students * sample_size = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_arts_students_l2090_209035


namespace NUMINAMATH_CALUDE_divide_fractions_l2090_209056

theorem divide_fractions : (7 : ℚ) / 3 / ((5 : ℚ) / 4) = 28 / 15 := by sorry

end NUMINAMATH_CALUDE_divide_fractions_l2090_209056


namespace NUMINAMATH_CALUDE_cube_root_last_three_digits_l2090_209089

theorem cube_root_last_three_digits :
  ∃ (n : ℕ+) (a : ℕ+) (b : ℕ),
    n = 1000 * a + b ∧
    b < 1000 ∧
    n = a^3 ∧
    n = 32768 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_last_three_digits_l2090_209089


namespace NUMINAMATH_CALUDE_football_team_throwers_l2090_209081

/-- Represents the number of throwers on a football team. -/
def num_throwers : ℕ := 52

/-- Represents the total number of players on the football team. -/
def total_players : ℕ := 70

/-- Represents the total number of right-handed players on the team. -/
def right_handed_players : ℕ := 64

theorem football_team_throwers :
  num_throwers = 52 ∧
  total_players = 70 ∧
  right_handed_players = 64 ∧
  num_throwers ≤ total_players ∧
  num_throwers ≤ right_handed_players ∧
  (total_players - num_throwers) % 3 = 0 ∧
  right_handed_players = num_throwers + 2 * ((total_players - num_throwers) / 3) :=
by sorry

end NUMINAMATH_CALUDE_football_team_throwers_l2090_209081


namespace NUMINAMATH_CALUDE_max_product_constrained_l2090_209036

theorem max_product_constrained (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_constraint : 3 * x + 8 * y = 72) : 
  x * y ≤ 54 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + 8 * y₀ = 72 ∧ x₀ * y₀ = 54 :=
sorry

end NUMINAMATH_CALUDE_max_product_constrained_l2090_209036


namespace NUMINAMATH_CALUDE_max_xy_value_l2090_209097

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x/3 + y/4 = 1) :
  ∃ (M : ℝ), M = 3 ∧ xy ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀/3 + y₀/4 = 1 ∧ x₀*y₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l2090_209097


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2090_209017

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2090_209017


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l2090_209080

/-- The number of counselors needed at Camp Cedar -/
def counselors_needed (num_boys : ℕ) (girl_to_boy_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girl_to_boy_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

/-- Theorem stating the number of counselors needed at Camp Cedar -/
theorem camp_cedar_counselors :
  counselors_needed 40 3 8 = 20 := by
  sorry

#eval counselors_needed 40 3 8

end NUMINAMATH_CALUDE_camp_cedar_counselors_l2090_209080


namespace NUMINAMATH_CALUDE_largest_prime_factor_3434_l2090_209011

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_3434 : largest_prime_factor 3434 = 7 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_3434_l2090_209011


namespace NUMINAMATH_CALUDE_prob_drawing_10_red_in_12_draws_l2090_209039

-- Define the number of white and red balls
def white_balls : ℕ := 5
def red_balls : ℕ := 3
def total_balls : ℕ := white_balls + red_balls

-- Define the probability of drawing a red ball
def prob_red : ℚ := red_balls / total_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Define the number of draws
def total_draws : ℕ := 12

-- Define the number of red balls needed to stop
def red_balls_to_stop : ℕ := 10

-- Define the probability of the event
def prob_event : ℚ := (Nat.choose (total_draws - 1) (red_balls_to_stop - 1)) * 
                      (prob_red ^ red_balls_to_stop) * 
                      (prob_white ^ (total_draws - red_balls_to_stop))

-- Theorem statement
theorem prob_drawing_10_red_in_12_draws : 
  prob_event = (Nat.choose 11 9) * ((3 / 8) ^ 10) * ((5 / 8) ^ 2) :=
sorry

end NUMINAMATH_CALUDE_prob_drawing_10_red_in_12_draws_l2090_209039


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l2090_209034

theorem polynomial_equation_solution (p : ℝ → ℝ) :
  (∀ x : ℝ, p (5 * x)^2 - 3 = p (5 * x^2 + 1)) →
  (p = λ _ ↦ (1 + Real.sqrt 13) / 2) ∨ (p = λ _ ↦ (1 - Real.sqrt 13) / 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l2090_209034


namespace NUMINAMATH_CALUDE_green_peaches_count_l2090_209052

/-- Given a basket of peaches, prove the number of green peaches. -/
theorem green_peaches_count (red : ℕ) (green : ℕ) : 
  red = 7 → green = red + 1 → green = 8 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2090_209052


namespace NUMINAMATH_CALUDE_greatest_k_for_inequality_l2090_209084

theorem greatest_k_for_inequality : 
  ∃ (k : ℤ), k = 5 ∧ 
  (∀ (j : ℤ), j > k → 
    ∃ (n : ℕ), n ≥ 2 ∧ ⌊n / Real.sqrt 3⌋ + 1 ≤ n^2 / Real.sqrt (3 * n^2 - j)) ∧
  (∀ (n : ℕ), n ≥ 2 → ⌊n / Real.sqrt 3⌋ + 1 > n^2 / Real.sqrt (3 * n^2 - k)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_inequality_l2090_209084


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l2090_209085

/-- A function that counts the number of positive integer divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 12 positive integer divisors -/
def has_12_divisors (n : ℕ) : Prop := count_divisors n = 12

/-- Theorem stating that 60 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_12_divisors : 
  (has_12_divisors 60) ∧ (∀ m : ℕ, 0 < m ∧ m < 60 → ¬(has_12_divisors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l2090_209085


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2090_209028

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 250) 
  (h2 : train_speed_kmph = 72) 
  (h3 : bridge_length = 1250) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 75 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2090_209028


namespace NUMINAMATH_CALUDE_pen_notebook_difference_l2090_209032

theorem pen_notebook_difference (notebooks pens : ℕ) : 
  notebooks = 30 →
  notebooks + pens = 110 →
  pens > notebooks →
  pens - notebooks = 50 := by
sorry

end NUMINAMATH_CALUDE_pen_notebook_difference_l2090_209032


namespace NUMINAMATH_CALUDE_points_on_same_side_l2090_209090

/-- The time when two points moving on a square are first on the same side -/
def time_on_same_side (square_side : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  25

/-- Theorem stating that the time when the points are first on the same side is 25 seconds -/
theorem points_on_same_side (square_side : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : square_side = 100)
  (h2 : speed_A = 5)
  (h3 : speed_B = 10) :
  time_on_same_side square_side speed_A speed_B = 25 :=
by
  sorry

#check points_on_same_side

end NUMINAMATH_CALUDE_points_on_same_side_l2090_209090


namespace NUMINAMATH_CALUDE_i_pow_2006_l2090_209050

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- Theorem to prove
theorem i_pow_2006 : i^2006 = -1 := by
  sorry

end NUMINAMATH_CALUDE_i_pow_2006_l2090_209050


namespace NUMINAMATH_CALUDE_simplify_expressions_l2090_209047

theorem simplify_expressions :
  (1 + (-0.5) = 0.5) ∧
  (2 - 10.1 = -10.1) ∧
  (3 + 7 = 10) ∧
  (4 - (-20) = 24) ∧
  (5 + |-(2/3)| = 17/3) ∧
  (6 - |-(4/5)| = 26/5) ∧
  (7 + (-(-10)) = 17) ∧
  (8 - (-(-20/7)) = -12/7) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2090_209047


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l2090_209014

def f (x : ℝ) := -x^2

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l2090_209014


namespace NUMINAMATH_CALUDE_arrange_algebra_and_calculus_books_l2090_209092

/-- The number of ways to arrange books on a shelf --/
def arrange_books (algebra_copies : ℕ) (calculus_copies : ℕ) : ℕ :=
  Nat.choose (algebra_copies + calculus_copies) algebra_copies

/-- Theorem: Arranging 4 algebra books and 5 calculus books yields 126 possibilities --/
theorem arrange_algebra_and_calculus_books :
  arrange_books 4 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_arrange_algebra_and_calculus_books_l2090_209092


namespace NUMINAMATH_CALUDE_dividing_line_halves_area_l2090_209016

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the L-shaped region -/
def LShapedRegion : Set Point := {p | 
  (0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 4) ∨
  (4 < p.x ∧ p.x ≤ 7 ∧ 0 ≤ p.y ∧ p.y ≤ 2)
}

/-- Calculates the area of a region -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The line y = (5/7)x -/
def dividingLine (p : Point) : Prop := p.y = (5/7) * p.x

/-- Regions above and below the dividing line -/
def upperRegion : Set Point := {p ∈ LShapedRegion | p.y ≥ (5/7) * p.x}
def lowerRegion : Set Point := {p ∈ LShapedRegion | p.y ≤ (5/7) * p.x}

theorem dividing_line_halves_area : 
  area upperRegion = area lowerRegion := by sorry

end NUMINAMATH_CALUDE_dividing_line_halves_area_l2090_209016


namespace NUMINAMATH_CALUDE_probability_five_consecutive_heads_eight_flips_l2090_209060

/-- A sequence of coin flips -/
def CoinFlipSequence := List Bool

/-- The length of a coin flip sequence -/
def sequenceLength : CoinFlipSequence → Nat :=
  List.length

/-- Checks if a sequence has at least n consecutive heads -/
def hasConsecutiveHeads (n : Nat) : CoinFlipSequence → Bool :=
  sorry

/-- All possible outcomes of flipping a coin n times -/
def allOutcomes (n : Nat) : List CoinFlipSequence :=
  sorry

/-- Count of sequences with at least n consecutive heads -/
def countConsecutiveHeads (n : Nat) (totalFlips : Nat) : Nat :=
  sorry

/-- Probability of getting at least n consecutive heads in m flips -/
def probabilityConsecutiveHeads (n : Nat) (m : Nat) : Rat :=
  sorry

theorem probability_five_consecutive_heads_eight_flips :
  probabilityConsecutiveHeads 5 8 = 23 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_five_consecutive_heads_eight_flips_l2090_209060


namespace NUMINAMATH_CALUDE_fibonacci_7_l2090_209064

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_7 : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_7_l2090_209064


namespace NUMINAMATH_CALUDE_fraction_equality_l2090_209059

theorem fraction_equality (a b : ℝ) (h : (a - b) / a = 2 / 3) : b / a = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2090_209059


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_solution_range_for_a_l2090_209006

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x - 1| > (1/2) * (x + 1)} = {x : ℝ | x > 3 ∨ x < 1/3} := by sorry

-- Part 2
theorem solution_range_for_a :
  ∀ a : ℝ, (∃ x : ℝ, |x - a| + |x - 2| ≤ 3) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_solution_range_for_a_l2090_209006


namespace NUMINAMATH_CALUDE_petes_backward_speed_l2090_209043

/-- Pete's backward walking speed problem -/
theorem petes_backward_speed (petes_hand_speed tracy_cartwheel_speed susans_speed petes_backward_speed : ℝ) : 
  petes_hand_speed = 2 →
  petes_hand_speed = (1 / 4) * tracy_cartwheel_speed →
  tracy_cartwheel_speed = 2 * susans_speed →
  petes_backward_speed = 3 * susans_speed →
  petes_backward_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_petes_backward_speed_l2090_209043


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2090_209012

/-- Given polynomials f, g, and h, prove their sum equals the simplified polynomial -/
theorem sum_of_polynomials (x : ℝ) :
  let f := fun x : ℝ => -4 * x^2 + 2 * x - 5
  let g := fun x : ℝ => -6 * x^2 + 4 * x - 9
  let h := fun x : ℝ => 6 * x^2 + 6 * x + 2
  f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2090_209012


namespace NUMINAMATH_CALUDE_half_power_five_decimal_l2090_209078

theorem half_power_five_decimal : (1/2)^5 = 0.03125 := by
  sorry

end NUMINAMATH_CALUDE_half_power_five_decimal_l2090_209078


namespace NUMINAMATH_CALUDE_shirt_tie_belt_combinations_l2090_209004

/-- Given a number of shirts, ties, and belts, calculates the total number of
    shirt-and-tie or shirt-and-belt combinations -/
def total_combinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * ties + shirts * belts

/-- Theorem stating that with 7 shirts, 6 ties, and 4 belts, 
    the total number of combinations is 70 -/
theorem shirt_tie_belt_combinations :
  total_combinations 7 6 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_belt_combinations_l2090_209004


namespace NUMINAMATH_CALUDE_yard_raking_time_l2090_209042

theorem yard_raking_time (your_time brother_time together_time : ℝ) 
  (h1 : brother_time = 45)
  (h2 : together_time = 18)
  (h3 : 1 / your_time + 1 / brother_time = 1 / together_time) :
  your_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_yard_raking_time_l2090_209042


namespace NUMINAMATH_CALUDE_student_A_more_stable_l2090_209071

/-- Represents a student's jumping rope performance -/
structure JumpRopePerformance where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one performance is more stable than another -/
def more_stable (a b : JumpRopePerformance) : Prop :=
  a.variance < b.variance

theorem student_A_more_stable (
  student_A student_B : JumpRopePerformance
) (h1 : student_A.average_score = student_B.average_score)
  (h2 : student_A.variance = 0.06)
  (h3 : student_B.variance = 0.35) :
  more_stable student_A student_B :=
sorry

end NUMINAMATH_CALUDE_student_A_more_stable_l2090_209071


namespace NUMINAMATH_CALUDE_museum_exhibit_group_size_l2090_209087

/-- Represents the ticket sales data for a museum exhibit --/
structure TicketSales where
  regular_price : ℕ
  student_price : ℕ
  total_revenue : ℕ
  regular_to_student_ratio : ℕ
  start_time : ℕ  -- in minutes since midnight
  end_time : ℕ    -- in minutes since midnight
  interval : ℕ    -- in minutes

/-- Calculates the number of people in each group for the given ticket sales data --/
def people_per_group (sales : TicketSales) : ℕ :=
  let student_tickets := sales.total_revenue / (sales.regular_price * sales.regular_to_student_ratio + sales.student_price)
  let regular_tickets := student_tickets * sales.regular_to_student_ratio
  let total_tickets := student_tickets + regular_tickets
  let num_groups := (sales.end_time - sales.start_time) / sales.interval
  total_tickets / num_groups

/-- Theorem stating that for the given conditions, the number of people in each group is 30 --/
theorem museum_exhibit_group_size :
  let sales : TicketSales := {
    regular_price := 10,
    student_price := 5,
    total_revenue := 28350,
    regular_to_student_ratio := 3,
    start_time := 9 * 60,      -- 9:00 AM in minutes
    end_time := 17 * 60 + 55,  -- 5:55 PM in minutes
    interval := 5
  }
  people_per_group sales = 30 := by
  sorry


end NUMINAMATH_CALUDE_museum_exhibit_group_size_l2090_209087


namespace NUMINAMATH_CALUDE_paula_candy_problem_l2090_209077

theorem paula_candy_problem (initial_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ)
  (h1 : initial_candies = 20)
  (h2 : num_friends = 6)
  (h3 : candies_per_friend = 4) :
  num_friends * candies_per_friend - initial_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_paula_candy_problem_l2090_209077


namespace NUMINAMATH_CALUDE_order_inequality_l2090_209098

theorem order_inequality (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) :
  x^2 > a*x ∧ a*x > a*b ∧ a*b > a^2 := by
  sorry

end NUMINAMATH_CALUDE_order_inequality_l2090_209098


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2090_209002

theorem inequality_equivalence (x : ℝ) : 
  (x ≠ 5) → ((x^2 + 2*x + 1) / ((x-5)^2) ≥ 15 ↔ 
    ((76 - 3*Real.sqrt 60) / 14 ≤ x ∧ x < 5) ∨ 
    (5 < x ∧ x ≤ (76 + 3*Real.sqrt 60) / 14)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2090_209002


namespace NUMINAMATH_CALUDE_henry_lawn_mowing_l2090_209094

/-- The number of lawns Henry was supposed to mow -/
def total_lawns : ℕ := 12

/-- The amount Henry earns per lawn -/
def earnings_per_lawn : ℕ := 5

/-- The number of lawns Henry forgot to mow -/
def forgotten_lawns : ℕ := 7

/-- The amount Henry actually earned -/
def actual_earnings : ℕ := 25

theorem henry_lawn_mowing :
  total_lawns = (actual_earnings / earnings_per_lawn) + forgotten_lawns :=
by sorry

end NUMINAMATH_CALUDE_henry_lawn_mowing_l2090_209094


namespace NUMINAMATH_CALUDE_parrots_per_cage_l2090_209095

theorem parrots_per_cage (num_cages : ℕ) (parakeets_per_cage : ℕ) (total_birds : ℕ) :
  num_cages = 4 →
  parakeets_per_cage = 2 →
  total_birds = 40 →
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l2090_209095


namespace NUMINAMATH_CALUDE_obtuse_triangle_necessary_not_sufficient_l2090_209009

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Definition of an obtuse triangle --/
def isObtuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

theorem obtuse_triangle_necessary_not_sufficient :
  (∀ t : Triangle, isObtuse t → (t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2)) ∧
  (∃ t : Triangle, (t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2) ∧ ¬isObtuse t) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_necessary_not_sufficient_l2090_209009


namespace NUMINAMATH_CALUDE_ratio_equality_l2090_209040

theorem ratio_equality (x : ℝ) : (0.60 / x = 6 / 2) → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2090_209040


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l2090_209082

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + c * a) = 166) 
  (edge_sum : 4 * (a + b + c) = 64) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 12 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l2090_209082


namespace NUMINAMATH_CALUDE_triangle_property_l2090_209065

open Real

theorem triangle_property (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  b * cos C + sqrt 3 * b * sin C = a + c →
  B = π / 3 ∧
  (b = sqrt 3 → -sqrt 3 < 2 * a - c ∧ 2 * a - c < 2 * sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l2090_209065


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2090_209007

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 9) →
  (a 4 + a 5 + a 6 = 27) →
  (a 7 + a 8 + a 9 = 45) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2090_209007


namespace NUMINAMATH_CALUDE_distance_between_specific_planes_l2090_209054

/-- The distance between two planes given by their equations -/
def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  sorry

/-- Theorem: The distance between the planes x - 2y + 2z = 9 and 2x - 4y + 4z = 18 is 0 -/
theorem distance_between_specific_planes :
  distance_between_planes 1 (-2) 2 9 2 (-4) 4 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_specific_planes_l2090_209054


namespace NUMINAMATH_CALUDE_base_2016_remainder_l2090_209083

theorem base_2016_remainder (N A B C k : ℕ) : 
  (N = A * 2016^2 + B * 2016 + C) →
  (A < 2016 ∧ B < 2016 ∧ C < 2016) →
  (1 ≤ k ∧ k ≤ 2015) →
  (N - (A + B + C + k)) % 2015 = 2015 - k := by
  sorry

end NUMINAMATH_CALUDE_base_2016_remainder_l2090_209083


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l2090_209069

theorem bicycle_cost_price (final_price : ℝ) (profit_percentage : ℝ) : 
  final_price = 225 →
  profit_percentage = 25 →
  ∃ (original_cost : ℝ), 
    original_cost * (1 + profit_percentage / 100)^2 = final_price ∧
    original_cost = 144 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l2090_209069


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l2090_209001

theorem hexagon_largest_angle (a b c d e f : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 2 →
  f / a = 5 / 2 →
  a + b + c + d + e + f = 720 →
  f = 1200 / 7 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l2090_209001


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2090_209055

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (startingNumber : ℕ) (sampleIndex : ℕ) : ℕ :=
  startingNumber + (sampleIndex - 1) * (totalEmployees / sampleSize)

/-- Theorem: If the 5th sample is 23 in a systematic sampling of 40 from 200, then the 8th sample is 38 -/
theorem systematic_sampling_theorem (totalEmployees sampleSize startingNumber : ℕ) 
    (h1 : totalEmployees = 200)
    (h2 : sampleSize = 40)
    (h3 : systematicSample totalEmployees sampleSize startingNumber 5 = 23) :
  systematicSample totalEmployees sampleSize startingNumber 8 = 38 := by
  sorry

#eval systematicSample 200 40 3 5  -- Should output 23
#eval systematicSample 200 40 3 8  -- Should output 38

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2090_209055


namespace NUMINAMATH_CALUDE_column_arrangement_l2090_209010

theorem column_arrangement (total_people : ℕ) 
  (h1 : total_people = 30 * 16) 
  (h2 : ∃ (people_per_column : ℕ), total_people = people_per_column * 10) : 
  ∃ (people_per_column : ℕ), total_people = people_per_column * 10 ∧ people_per_column = 48 :=
by sorry

end NUMINAMATH_CALUDE_column_arrangement_l2090_209010
