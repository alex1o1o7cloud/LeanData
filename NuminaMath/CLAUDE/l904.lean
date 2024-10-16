import Mathlib

namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_nonnegative_l904_90474

theorem empty_solution_set_iff_a_nonnegative (a : ℝ) :
  (∀ x : ℝ, ¬(2*x < 5 - 3*x ∧ (x-1)/2 > a)) ↔ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_nonnegative_l904_90474


namespace NUMINAMATH_CALUDE_solve_system_l904_90447

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : 2 * x + y = 21) : 
  y = 27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l904_90447


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l904_90480

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a)
  (h_eq : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_prod : a 9 * a 10 = -8) :
  a 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l904_90480


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l904_90472

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of possible face arrangements for n coins where no two adjacent coins are face to face -/
def faceArrangements (n : ℕ) : ℕ := n + 1

theorem coin_stack_arrangements :
  let totalCoins : ℕ := 8
  let goldCoins : ℕ := 5
  let silverCoins : ℕ := 3
  (binomial totalCoins goldCoins) * (faceArrangements totalCoins) = 504 := by sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l904_90472


namespace NUMINAMATH_CALUDE_total_subjects_is_41_l904_90445

/-- The total number of subjects taken by Millie, Monica, and Marius -/
def total_subjects (monica_subjects marius_subjects millie_subjects : ℕ) : ℕ :=
  monica_subjects + marius_subjects + millie_subjects

/-- Theorem stating the total number of subjects taken by all three students -/
theorem total_subjects_is_41 :
  ∃ (monica_subjects marius_subjects millie_subjects : ℕ),
    monica_subjects = 10 ∧
    marius_subjects = monica_subjects + 4 ∧
    millie_subjects = marius_subjects + 3 ∧
    total_subjects monica_subjects marius_subjects millie_subjects = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_total_subjects_is_41_l904_90445


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l904_90451

/-- Given a hyperbola with equation x²/9 - y²/4 = 1, 
    its asymptotes have the equation y = ±(2/3)x -/
theorem hyperbola_asymptotes : 
  ∃ (f : ℝ → ℝ), 
    (∀ x y : ℝ, x^2/9 - y^2/4 = 1 → 
      (y = f x ∨ y = -f x) ∧ 
      f x = (2/3) * x) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l904_90451


namespace NUMINAMATH_CALUDE_min_value_2a_minus_ab_l904_90494

def is_valid (a b : ℕ) : Prop := 0 < a ∧ a < 8 ∧ 0 < b ∧ b < 8

theorem min_value_2a_minus_ab :
  ∃ (a₀ b₀ : ℕ), is_valid a₀ b₀ ∧
  (∀ (a b : ℕ), is_valid a b → (2 * a - a * b : ℤ) ≥ (2 * a₀ - a₀ * b₀ : ℤ)) ∧
  (2 * a₀ - a₀ * b₀ : ℤ) = -35 :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_minus_ab_l904_90494


namespace NUMINAMATH_CALUDE_intersection_when_m_3_intersection_equals_B_l904_90414

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3 * m - 2}

theorem intersection_when_m_3 : A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 4} := by sorry

theorem intersection_equals_B (m : ℝ) : A ∩ B m = B m ↔ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_intersection_equals_B_l904_90414


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l904_90483

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^3 / b^3 = 125 / 1 → a / b = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l904_90483


namespace NUMINAMATH_CALUDE_reina_kevin_marble_ratio_l904_90421

/-- Proves that the ratio of Reina's marbles to Kevin's marbles is 4:1 -/
theorem reina_kevin_marble_ratio :
  let kevin_counters : ℕ := 40
  let kevin_marbles : ℕ := 50
  let reina_counters : ℕ := 3 * kevin_counters
  let reina_total : ℕ := 320
  let reina_marbles : ℕ := reina_total - reina_counters
  (reina_marbles : ℚ) / kevin_marbles = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_reina_kevin_marble_ratio_l904_90421


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l904_90455

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- The sequence is increasing -/
def IncreasingSequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  IncreasingSequence b →
  b 4 * b 5 = 30 →
  (b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l904_90455


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l904_90484

def P : Set Nat := {1, 3, 6, 9}
def Q : Set Nat := {1, 2, 4, 6, 8}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l904_90484


namespace NUMINAMATH_CALUDE_rectangle_area_l904_90430

/-- Given a rectangle with perimeter 120 cm and length twice the width, prove its area is 800 cm² -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  (2 * (length + width) = 120) →  -- Perimeter condition
  (length = 2 * width) →          -- Length-width relationship
  (length * width = 800) :=       -- Area to prove
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l904_90430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l904_90427

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℚ := (n^2 + 3*n) / 2

-- Define the general term of the arithmetic sequence
def a (n : ℕ) : ℚ := n + 1

-- Define the terms of the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a (2*n - 1) * a (2*n + 1))

-- Define the sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℚ := n / (4*n + 4)

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → a n = n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (4*n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l904_90427


namespace NUMINAMATH_CALUDE_son_work_time_l904_90438

theorem son_work_time (man_time son_time combined_time : ℚ) : 
  man_time = 6 →
  combined_time = 3 →
  1 / man_time + 1 / son_time = 1 / combined_time →
  son_time = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l904_90438


namespace NUMINAMATH_CALUDE_students_remaining_in_school_l904_90486

theorem students_remaining_in_school (total_students : ℕ) 
  (h1 : total_students = 1000)
  (h2 : ∃ trip_students : ℕ, trip_students = total_students / 2)
  (h3 : ∃ remaining_after_trip : ℕ, remaining_after_trip = total_students - (total_students / 2))
  (h4 : ∃ sent_home : ℕ, sent_home = remaining_after_trip / 2)
  : total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2) = 250 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_in_school_l904_90486


namespace NUMINAMATH_CALUDE_rich_walk_ratio_l904_90409

theorem rich_walk_ratio : 
  ∀ (x : ℝ), 
    (20 : ℝ) + 200 + 220 * x + ((20 + 200 + 220 * x) / 2) = 990 → 
    (220 * x) / (20 + 200) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rich_walk_ratio_l904_90409


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_eq_three_times_abs_minus_two_l904_90441

theorem product_of_solutions_abs_eq_three_times_abs_minus_two :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3*(|y₁| - 2)) ∧ (|y₂| = 3*(|y₂| - 2)) ∧ (y₁ ≠ y₂) ∧ (y₁ * y₂ = -9) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_eq_three_times_abs_minus_two_l904_90441


namespace NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_60_l904_90498

theorem greatest_common_divisor_420_90_under_60 : 
  ∃ (n : ℕ), n ∣ 420 ∧ n ∣ 90 ∧ n < 60 ∧ 
  ∀ (m : ℕ), m ∣ 420 ∧ m ∣ 90 ∧ m < 60 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_60_l904_90498


namespace NUMINAMATH_CALUDE_mans_upstream_rate_l904_90499

/-- Given a man's rowing rates and current speed, calculate his upstream rate -/
theorem mans_upstream_rate
  (downstream_rate : ℝ)
  (still_water_rate : ℝ)
  (current_rate : ℝ)
  (h1 : downstream_rate = 24)
  (h2 : still_water_rate = 15.5)
  (h3 : current_rate = 8.5) :
  still_water_rate - current_rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_rate_l904_90499


namespace NUMINAMATH_CALUDE_solve_for_x_l904_90402

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 20) (h2 : y = 5) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l904_90402


namespace NUMINAMATH_CALUDE_officer_average_salary_l904_90440

/-- Represents the average salary of officers in an office, given the following conditions:
  * The average salary of all employees is 120 Rs/month
  * The average salary of non-officers is 110 Rs/month
  * There are 15 officers
  * There are 480 non-officers
-/
theorem officer_average_salary :
  let total_employees : ℕ := 15 + 480
  let all_average : ℚ := 120
  let non_officer_average : ℚ := 110
  let non_officer_count : ℕ := 480
  let officer_count : ℕ := 15
  (total_employees : ℚ) * all_average =
    (non_officer_count : ℚ) * non_officer_average +
    (officer_count : ℚ) * ((total_employees : ℚ) * all_average - (non_officer_count : ℚ) * non_officer_average) / (officer_count : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_officer_average_salary_l904_90440


namespace NUMINAMATH_CALUDE_smallest_alpha_is_eight_l904_90482

/-- A quadratic polynomial P(x) = ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The derivative of a quadratic polynomial at x = 0 -/
def QuadraticPolynomial.deriv_at_zero (P : QuadraticPolynomial) : ℝ :=
  P.b

/-- The property that |P(x)| ≤ 1 for x ∈ [0,1] -/
def bounded_on_unit_interval (P : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |P.eval x| ≤ 1

theorem smallest_alpha_is_eight :
  (∃ α : ℝ, ∀ P : QuadraticPolynomial, bounded_on_unit_interval P → |P.deriv_at_zero| ≤ α) ∧
  (∀ β : ℝ, (∀ P : QuadraticPolynomial, bounded_on_unit_interval P → |P.deriv_at_zero| ≤ β) → 8 ≤ β) :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_is_eight_l904_90482


namespace NUMINAMATH_CALUDE_eleven_pictures_left_to_color_l904_90449

/-- The number of pictures left to color given two coloring books and some already colored pictures. -/
def pictures_left_to_color (book1_pictures book2_pictures colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem stating that given the specific numbers in the problem, 11 pictures are left to color. -/
theorem eleven_pictures_left_to_color :
  pictures_left_to_color 23 32 44 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_pictures_left_to_color_l904_90449


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l904_90466

theorem sum_of_squares_of_roots (k l m n a b c : ℕ) :
  k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n →
  ((a * k^2 - b * k + c = 0 ∨ c * k^2 - 16 * b * k + 256 * a = 0) ∧
   (a * l^2 - b * l + c = 0 ∨ c * l^2 - 16 * b * l + 256 * a = 0) ∧
   (a * m^2 - b * m + c = 0 ∨ c * m^2 - 16 * b * m + 256 * a = 0) ∧
   (a * n^2 - b * n + c = 0 ∨ c * n^2 - 16 * b * n + 256 * a = 0)) →
  k^2 + l^2 + m^2 + n^2 = 325 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l904_90466


namespace NUMINAMATH_CALUDE_number_plus_seven_equals_six_l904_90433

theorem number_plus_seven_equals_six : 
  ∃ x : ℤ, x + 7 = 6 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_seven_equals_six_l904_90433


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l904_90479

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x + 1/x = 3) :
  x^4 + 1/x^4 = 47 := by sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_fourth_l904_90479


namespace NUMINAMATH_CALUDE_lucas_february_bill_l904_90462

/-- Calculates the total cost of a cell phone plan based on given parameters. -/
def calculate_phone_bill (base_cost : ℚ) (text_cost : ℚ) (extra_cost_30_31 : ℚ) 
  (extra_cost_beyond_31 : ℚ) (num_texts : ℕ) (talk_time : ℚ) : ℚ :=
  let text_total := num_texts * text_cost
  let extra_time := max (talk_time - 30) 0
  let extra_cost := 
    if extra_time ≤ 1 then
      extra_time * 60 * extra_cost_30_31
    else
      60 * extra_cost_30_31 + (extra_time - 1) * 60 * extra_cost_beyond_31
  base_cost + text_total + extra_cost

/-- Theorem stating that Lucas's phone bill for February is $55.00 -/
theorem lucas_february_bill : 
  calculate_phone_bill 25 0.1 0.15 0.2 150 31.5 = 55 := by
  sorry


end NUMINAMATH_CALUDE_lucas_february_bill_l904_90462


namespace NUMINAMATH_CALUDE_min_value_quadratic_l904_90497

theorem min_value_quadratic (x : ℝ) :
  let z := 4 * x^2 + 8 * x + 16
  ∀ y : ℝ, z ≤ y → 12 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l904_90497


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l904_90428

theorem complex_number_in_third_quadrant :
  let z : ℂ := ((-1 : ℂ) - 2*I) / (1 - 2*I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l904_90428


namespace NUMINAMATH_CALUDE_brett_red_marbles_l904_90450

/-- The number of red marbles Brett has -/
def red_marbles : ℕ := sorry

/-- The number of blue marbles Brett has -/
def blue_marbles : ℕ := sorry

/-- Brett has 24 more blue marbles than red marbles -/
axiom more_blue : blue_marbles = red_marbles + 24

/-- Brett has 5 times as many blue marbles as red marbles -/
axiom five_times : blue_marbles = 5 * red_marbles

theorem brett_red_marbles : red_marbles = 6 := by sorry

end NUMINAMATH_CALUDE_brett_red_marbles_l904_90450


namespace NUMINAMATH_CALUDE_trigonometric_identities_l904_90458

theorem trigonometric_identities :
  (Real.cos (75 * π / 180))^2 = (2 - Real.sqrt 3) / 4 ∧
  Real.tan (1 * π / 180) + Real.tan (44 * π / 180) + Real.tan (1 * π / 180) * Real.tan (44 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l904_90458


namespace NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l904_90431

/-- A structure representing a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A structure representing two inscribed rectangles sharing a common vertex --/
structure InscribedRectangles where
  outer : Rectangle
  common_vertex : ℝ  -- Position of K on AB, 0 ≤ common_vertex ≤ outer.width

/-- Calculate the area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculate the sum of areas of the two inscribed rectangles --/
def InscribedRectangles.sumOfAreas (ir : InscribedRectangles) : ℝ :=
  ir.common_vertex * ir.outer.height

/-- Theorem stating that the sum of areas of inscribed rectangles equals the area of the outer rectangle --/
theorem inscribed_rectangles_area_sum (ir : InscribedRectangles) :
  ir.sumOfAreas = ir.outer.area := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l904_90431


namespace NUMINAMATH_CALUDE_tree_planting_cost_l904_90437

/-- The cost to plant one tree given temperature drop and total cost -/
theorem tree_planting_cost 
  (temp_drop_per_tree : ℝ) 
  (total_temp_drop : ℝ) 
  (total_cost : ℝ) : 
  temp_drop_per_tree = 0.1 → 
  total_temp_drop = 1.8 → 
  total_cost = 108 → 
  (total_cost / (total_temp_drop / temp_drop_per_tree) = 6) :=
by
  sorry

#check tree_planting_cost

end NUMINAMATH_CALUDE_tree_planting_cost_l904_90437


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l904_90443

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l904_90443


namespace NUMINAMATH_CALUDE_complex_distance_theorem_l904_90469

theorem complex_distance_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3)
  (h₂ : Complex.abs z₂ = 5)
  (h₃ : Complex.abs (z₁ + z₂) = 6) :
  Complex.abs (z₁ - z₂) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_distance_theorem_l904_90469


namespace NUMINAMATH_CALUDE_cereal_box_initial_price_l904_90403

/-- The initial price of a cereal box given a price reduction and total purchase amount -/
theorem cereal_box_initial_price 
  (price_reduction : ℝ) 
  (num_boxes : ℕ) 
  (total_paid : ℝ) 
  (h1 : price_reduction = 24)
  (h2 : num_boxes = 20)
  (h3 : total_paid = 1600) : 
  ∃ (initial_price : ℝ), 
    num_boxes * (initial_price - price_reduction) = total_paid ∧ 
    initial_price = 104 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_initial_price_l904_90403


namespace NUMINAMATH_CALUDE_inequalities_hold_l904_90467

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  ab ≤ 1 ∧ Real.sqrt a + Real.sqrt b ≤ 2 ∧ a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l904_90467


namespace NUMINAMATH_CALUDE_roots_imply_composite_sum_of_squares_l904_90425

theorem roots_imply_composite_sum_of_squares (a b : ℤ) :
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) →
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ m * n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_imply_composite_sum_of_squares_l904_90425


namespace NUMINAMATH_CALUDE_seashells_solution_l904_90481

/-- The number of seashells found by Sam, Mary, John, and Emily -/
def seashells_problem (sam mary john emily : ℕ) : Prop :=
  sam = 18 ∧ mary = 47 ∧ john = 32 ∧ emily = 26 →
  sam + mary + john + emily = 123

/-- Theorem stating the solution to the seashells problem -/
theorem seashells_solution : seashells_problem 18 47 32 26 := by
  sorry

end NUMINAMATH_CALUDE_seashells_solution_l904_90481


namespace NUMINAMATH_CALUDE_angle_d_measure_l904_90457

/-- A scalene triangle with specific angle relationships -/
structure ScaleneTriangle where
  /-- Measure of angle D in degrees -/
  angleD : ℝ
  /-- Measure of angle E in degrees -/
  angleE : ℝ
  /-- Measure of angle F in degrees -/
  angleF : ℝ
  /-- Triangle is scalene -/
  scalene : angleD ≠ angleE ∧ angleE ≠ angleF ∧ angleD ≠ angleF
  /-- Angle E is twice angle D -/
  e_twice_d : angleE = 2 * angleD
  /-- Angle F is 40 degrees -/
  f_is_40 : angleF = 40
  /-- Sum of angles in a triangle is 180 degrees -/
  angle_sum : angleD + angleE + angleF = 180

/-- Theorem: In a scalene triangle DEF with the given conditions, angle D measures 140/3 degrees -/
theorem angle_d_measure (t : ScaleneTriangle) : t.angleD = 140 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_d_measure_l904_90457


namespace NUMINAMATH_CALUDE_all_approximations_valid_l904_90415

/-- Represents an approximation with its estimated value, absolute error, and relative error. -/
structure Approximation where
  value : ℝ
  absoluteError : ℝ
  relativeError : ℝ

/-- The approximations given in the problem. -/
def classSize : Approximation := ⟨40, 5, 0.125⟩
def hallPeople : Approximation := ⟨1500, 100, 0.067⟩
def itemPrice : Approximation := ⟨100, 5, 0.05⟩
def pageCharacters : Approximation := ⟨40000, 500, 0.0125⟩

/-- Checks if the relative error is correctly calculated from the absolute error and value. -/
def isValidApproximation (a : Approximation) : Prop :=
  a.relativeError = a.absoluteError / a.value

/-- Proves that all given approximations are valid. -/
theorem all_approximations_valid :
  isValidApproximation classSize ∧
  isValidApproximation hallPeople ∧
  isValidApproximation itemPrice ∧
  isValidApproximation pageCharacters :=
sorry

end NUMINAMATH_CALUDE_all_approximations_valid_l904_90415


namespace NUMINAMATH_CALUDE_angle_C_is_60_degrees_area_is_10_sqrt_3_l904_90436

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  (t.a - t.c) * (Real.sin t.A + Real.sin t.C) = (t.a - t.b) * Real.sin t.B

-- Theorem 1: Measure of angle C
theorem angle_C_is_60_degrees (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.C = Real.pi / 3 :=
sorry

-- Theorem 2: Area of triangle when a = 5 and c = 7
theorem area_is_10_sqrt_3 (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t)
  (h3 : t.a = 5)
  (h4 : t.c = 7) : 
  (1/2) * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_angle_C_is_60_degrees_area_is_10_sqrt_3_l904_90436


namespace NUMINAMATH_CALUDE_square_inequality_l904_90452

theorem square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l904_90452


namespace NUMINAMATH_CALUDE_sum_of_ten_and_hundredth_l904_90422

theorem sum_of_ten_and_hundredth : 10 + 0.01 = 10.01 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_and_hundredth_l904_90422


namespace NUMINAMATH_CALUDE_rhombus_equations_l904_90424

/-- A rhombus with given properties -/
structure Rhombus where
  /-- Point A of the rhombus -/
  A : ℝ × ℝ
  /-- Point C of the rhombus -/
  C : ℝ × ℝ
  /-- Point P on the line BC -/
  P : ℝ × ℝ
  /-- Assertion that ABCD is a rhombus -/
  is_rhombus : A = (-4, 7) ∧ C = (2, -3) ∧ P = (3, -1)

/-- The equation of line AD in a rhombus -/
def line_AD (r : Rhombus) : ℝ → ℝ → Prop :=
  fun x y => 2 * x - y + 15 = 0

/-- The equation of diagonal BD in a rhombus -/
def diagonal_BD (r : Rhombus) : ℝ → ℝ → Prop :=
  fun x y => 3 * x - 5 * y + 13 = 0

/-- Main theorem about the equations of line AD and diagonal BD in the given rhombus -/
theorem rhombus_equations (r : Rhombus) :
  (∀ x y, line_AD r x y ↔ y = 2 * x + 15) ∧
  (∀ x y, diagonal_BD r x y ↔ y = (3 * x + 13) / 5) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_equations_l904_90424


namespace NUMINAMATH_CALUDE_polynomial_remainder_l904_90487

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 5*x^8 - 3*x^7 + 2*x^6 - 4*x^3 + x^2 - 9
  let d : ℝ → ℝ := λ x => 3*x - 9
  ∃ q : ℝ → ℝ, p = λ x => d x * q x + 39594 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l904_90487


namespace NUMINAMATH_CALUDE_absolute_value_plus_inverse_l904_90495

theorem absolute_value_plus_inverse : |(-2 : ℝ)| + 3⁻¹ = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_inverse_l904_90495


namespace NUMINAMATH_CALUDE_inequality_solution_l904_90478

/-- Theorem: Solutions to the inequality ax^2 - 2 ≥ 2x - ax for a < 0 -/
theorem inequality_solution (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, ¬(a * x^2 - 2 ≥ 2 * x - a * x) ∨ (a = -2 ∧ x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l904_90478


namespace NUMINAMATH_CALUDE_basketball_team_selection_l904_90417

def total_players : ℕ := 12
def team_size : ℕ := 5
def captain_count : ℕ := 1
def regular_player_count : ℕ := 4

theorem basketball_team_selection :
  (total_players.choose captain_count) * ((total_players - captain_count).choose regular_player_count) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l904_90417


namespace NUMINAMATH_CALUDE_sum_of_opposite_sign_l904_90419

/-- Two real numbers are opposite in sign if their product is less than or equal to zero -/
def opposite_sign (a b : ℝ) : Prop := a * b ≤ 0

/-- If two real numbers are opposite in sign, then their sum is zero -/
theorem sum_of_opposite_sign (a b : ℝ) : opposite_sign a b → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_opposite_sign_l904_90419


namespace NUMINAMATH_CALUDE_triangle_inequality_l904_90426

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a^2 - b^2| / c + |b^2 - c^2| / a ≥ |c^2 - a^2| / b := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l904_90426


namespace NUMINAMATH_CALUDE_gummy_bear_distribution_l904_90444

theorem gummy_bear_distribution (initial_candies : ℕ) (num_siblings : ℕ) (josh_eat : ℕ) (leftover : ℕ) :
  initial_candies = 100 →
  num_siblings = 3 →
  josh_eat = 16 →
  leftover = 19 →
  ∃ (sibling_candies : ℕ),
    sibling_candies * num_siblings + 2 * (josh_eat + leftover) = initial_candies ∧
    sibling_candies = 10 :=
by sorry

end NUMINAMATH_CALUDE_gummy_bear_distribution_l904_90444


namespace NUMINAMATH_CALUDE_range_of_a_l904_90408

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : 
  (B a ⊆ A) ↔ (a ≤ -1 ∨ a = 1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l904_90408


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l904_90454

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l904_90454


namespace NUMINAMATH_CALUDE_possible_sums_B_l904_90489

theorem possible_sums_B (a b c d : ℕ+) 
  (h1 : a * b = 2 * (c + d))
  (h2 : c * d = 2 * (a + b))
  (h3 : a + b ≥ c + d) :
  c + d = 13 ∨ c + d = 10 ∨ c + d = 9 ∨ c + d = 8 := by
sorry

end NUMINAMATH_CALUDE_possible_sums_B_l904_90489


namespace NUMINAMATH_CALUDE_leo_has_largest_answer_l904_90476

def starting_number : ℕ := 12

def rodrigo_process (n : ℕ) : ℕ := ((n - 3)^2 + 4)

def samantha_process (n : ℕ) : ℕ := (n^2 - 5 + 4)

def leo_process (n : ℕ) : ℕ := ((n - 3 + 4)^2)

theorem leo_has_largest_answer :
  leo_process starting_number > rodrigo_process starting_number ∧
  leo_process starting_number > samantha_process starting_number :=
sorry

end NUMINAMATH_CALUDE_leo_has_largest_answer_l904_90476


namespace NUMINAMATH_CALUDE_successive_price_reduction_l904_90406

theorem successive_price_reduction (initial_reduction : ℝ) (subsequent_reduction : ℝ) 
  (initial_reduction_percent : initial_reduction = 0.25) 
  (subsequent_reduction_percent : subsequent_reduction = 0.40) : 
  1 - (1 - initial_reduction) * (1 - subsequent_reduction) = 0.55 := by
sorry

end NUMINAMATH_CALUDE_successive_price_reduction_l904_90406


namespace NUMINAMATH_CALUDE_price_per_small_bottle_l904_90435

/-- Calculates the price per small bottle given the number of large and small bottles,
    the price of large bottles, and the average price of all bottles. -/
theorem price_per_small_bottle
  (num_large : ℕ)
  (num_small : ℕ)
  (price_large : ℚ)
  (avg_price : ℚ)
  (h1 : num_large = 1325)
  (h2 : num_small = 750)
  (h3 : price_large = 189/100)
  (h4 : avg_price = 17057/10000) :
  ∃ (price_small : ℚ),
    abs (price_small - 13828/10000) < 1/10000 ∧
    (num_large * price_large + num_small * price_small) / (num_large + num_small) = avg_price :=
sorry

end NUMINAMATH_CALUDE_price_per_small_bottle_l904_90435


namespace NUMINAMATH_CALUDE_smallest_a_for_integer_roots_and_product_condition_l904_90413

theorem smallest_a_for_integer_roots_and_product_condition : 
  (∃ (a : ℕ+), 
    (∀ (x : ℤ), x^2 + a*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z) ∧ 
    (∀ (b : ℕ+), b < a → 
      ¬(∀ (x : ℤ), x^2 + b*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z))) ∧
  (∀ (a : ℕ+), 
    (∀ (x : ℤ), x^2 + a*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z) ∧ 
    (∀ (b : ℕ+), b < a → 
      ¬(∀ (x : ℤ), x^2 + b*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z)) → 
    a = 11) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_integer_roots_and_product_condition_l904_90413


namespace NUMINAMATH_CALUDE_president_vp_committee_selection_l904_90496

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem president_vp_committee_selection (n : ℕ) (h : n = 10) : 
  n * (n - 1) * choose (n - 2) 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_president_vp_committee_selection_l904_90496


namespace NUMINAMATH_CALUDE_four_number_puzzle_l904_90492

theorem four_number_puzzle :
  ∀ (a b c d : ℕ),
    a + b + c + d = 243 →
    ∃ (x : ℚ),
      (a + 8 : ℚ) = x ∧
      (b - 8 : ℚ) = x ∧
      (c * 8 : ℚ) = x ∧
      (d / 8 : ℚ) = x →
    (max (max a b) (max c d)) * (min (min a b) (min c d)) = 576 := by
  sorry

end NUMINAMATH_CALUDE_four_number_puzzle_l904_90492


namespace NUMINAMATH_CALUDE_sum_of_first_5n_integers_l904_90439

theorem sum_of_first_5n_integers (n : ℕ) : 
  (3*n*(3*n + 1))/2 = (n*(n + 1))/2 + 210 → 
  (5*n*(5*n + 1))/2 = 630 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_5n_integers_l904_90439


namespace NUMINAMATH_CALUDE_inequality_solution_set_l904_90446

theorem inequality_solution_set (m : ℝ) (h : m < -3) :
  {x : ℝ | (m + 3) * x^2 - (2 * m + 3) * x + m > 0} = {x : ℝ | 1 < x ∧ x < m / (m + 3)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l904_90446


namespace NUMINAMATH_CALUDE_high_school_total_students_l904_90423

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  total_sample : ℕ
  grade1_sample : ℕ
  grade2_sample : ℕ
  grade3_total : ℕ
  sample_sum : total_sample = grade1_sample + grade2_sample + (total_sample - grade1_sample - grade2_sample)
  grade3_prob : (total_sample - grade1_sample - grade2_sample) / grade3_total = 1 / 20

/-- The total number of students in the high school is 3600. -/
theorem high_school_total_students (h : HighSchool) 
  (h_total_sample : h.total_sample = 180)
  (h_grade1_sample : h.grade1_sample = 70)
  (h_grade2_sample : h.grade2_sample = 60)
  (h_grade3_total : h.grade3_total = 1000) : 
  h.total_sample * 20 = 3600 := by
  sorry

#check high_school_total_students

end NUMINAMATH_CALUDE_high_school_total_students_l904_90423


namespace NUMINAMATH_CALUDE_polynomial_difference_l904_90470

theorem polynomial_difference (a : ℝ) : (6 * a^2 - 5*a + 3) - (5 * a^2 + 2*a - 1) = a^2 - 7*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_l904_90470


namespace NUMINAMATH_CALUDE_triangle_vertices_l904_90471

/-- The lines forming the triangle --/
def line1 (x y : ℚ) : Prop := 2 * x + y - 6 = 0
def line2 (x y : ℚ) : Prop := x - y + 4 = 0
def line3 (x y : ℚ) : Prop := y + 1 = 0

/-- The vertices of the triangle --/
def vertex1 : ℚ × ℚ := (2/3, 14/3)
def vertex2 : ℚ × ℚ := (-5, -1)
def vertex3 : ℚ × ℚ := (7/2, -1)

/-- Theorem stating that the given points are the vertices of the triangle --/
theorem triangle_vertices : 
  (line1 vertex1.1 vertex1.2 ∧ line2 vertex1.1 vertex1.2) ∧
  (line2 vertex2.1 vertex2.2 ∧ line3 vertex2.1 vertex2.2) ∧
  (line1 vertex3.1 vertex3.2 ∧ line3 vertex3.1 vertex3.2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vertices_l904_90471


namespace NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l904_90475

theorem cos_squared_30_minus_2_minus_pi_to_0 :
  Real.cos (30 * π / 180) ^ 2 - (2 - π) ^ 0 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l904_90475


namespace NUMINAMATH_CALUDE_certain_number_proof_l904_90432

theorem certain_number_proof (p q : ℝ) (h1 : 3 / q = 18) (h2 : p - q = 7/12) : 3 / p = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l904_90432


namespace NUMINAMATH_CALUDE_emily_egg_collection_l904_90400

/-- The total number of eggs collected by Emily -/
def total_eggs : ℕ :=
  let set_a := 200 * 36 + 250 * 24
  let set_b := 375 * 42 - 80
  let set_c := (560 / 2) * 50 + (560 / 2) * 32
  set_a + set_b + set_c

/-- Theorem stating that Emily collected 51830 eggs in total -/
theorem emily_egg_collection : total_eggs = 51830 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l904_90400


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l904_90456

/-- Represents the starting age of smoking -/
def X : Type := ℕ

/-- Represents the relative risk of lung cancer for different starting ages -/
def Y : Type := ℝ

/-- Represents the number of cigarettes smoked per day -/
def U : Type := ℕ

/-- Represents the relative risk of lung cancer for different numbers of cigarettes -/
def V : Type := ℝ

/-- The linear correlation coefficient between X and Y -/
def r1 : ℝ := sorry

/-- The linear correlation coefficient between U and V -/
def r2 : ℝ := sorry

/-- Theorem stating the relationship between r1 and r2 -/
theorem correlation_coefficient_relationship : r1 < 0 ∧ 0 < r2 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l904_90456


namespace NUMINAMATH_CALUDE_min_sum_of_arithmetic_sequence_l904_90488

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem min_sum_of_arithmetic_sequence (seq : ArithmeticSequence) :
  seq.a 1 = -7 →
  sumOfTerms seq 3 = -15 →
  ∀ n : ℕ, sumOfTerms seq n ≥ -16 ∧ 
  (∃ m : ℕ, sumOfTerms seq m = -16) := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_arithmetic_sequence_l904_90488


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l904_90412

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 10003 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l904_90412


namespace NUMINAMATH_CALUDE_two_numbers_difference_l904_90442

theorem two_numbers_difference (a b : ℝ) 
  (sum_condition : a + b = 9)
  (square_difference_condition : a^2 - b^2 = 45) :
  |a - b| = 5 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l904_90442


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l904_90463

/-- The value of p for which the focus of the parabola y^2 = 2px coincides with 
    the right focus of the hyperbola x^2/3 - y^2/1 = 1 -/
theorem parabola_hyperbola_focus_coincide : ∃ p : ℝ, 
  (∀ x y : ℝ, y^2 = 2*p*x → x^2/3 - y^2 = 1 → 
   ∃ f : ℝ × ℝ, f = (p, 0) ∧ f = (2, 0)) → 
  p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l904_90463


namespace NUMINAMATH_CALUDE_problem_solution_l904_90468

theorem problem_solution : ∃ x : ℚ, (70 / 100) * x - (1 / 3) * x = 110 ∧ x = 300 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l904_90468


namespace NUMINAMATH_CALUDE_max_children_in_class_l904_90460

/-- The maximum number of children in a class given the distribution of items -/
theorem max_children_in_class (total_apples total_cookies total_chocolates : ℕ)
  (leftover_apples leftover_cookies leftover_chocolates : ℕ)
  (h1 : total_apples = 55)
  (h2 : total_cookies = 114)
  (h3 : total_chocolates = 83)
  (h4 : leftover_apples = 3)
  (h5 : leftover_cookies = 10)
  (h6 : leftover_chocolates = 5) :
  Nat.gcd (total_apples - leftover_apples)
    (Nat.gcd (total_cookies - leftover_cookies) (total_chocolates - leftover_chocolates)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_max_children_in_class_l904_90460


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l904_90410

theorem common_roots_cubic_polynomials :
  ∀ (a b : ℝ),
  (∃ (r s : ℝ), r ≠ s ∧
    (r^3 + a*r^2 + 15*r + 10 = 0) ∧
    (r^3 + b*r^2 + 18*r + 12 = 0) ∧
    (s^3 + a*s^2 + 15*s + 10 = 0) ∧
    (s^3 + b*s^2 + 18*s + 12 = 0)) →
  a = 6 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l904_90410


namespace NUMINAMATH_CALUDE_right_triangle_area_l904_90407

theorem right_triangle_area (a b : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 81) :
  (1/2) * a * b = 36 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l904_90407


namespace NUMINAMATH_CALUDE_gigi_remaining_pieces_l904_90464

/-- The number of remaining mushroom pieces after cutting and using some -/
def remaining_pieces (total_mushrooms : ℕ) (pieces_per_mushroom : ℕ) 
  (used_by_kenny : ℕ) (used_by_karla : ℕ) : ℕ :=
  total_mushrooms * pieces_per_mushroom - (used_by_kenny + used_by_karla)

/-- Theorem stating the number of remaining mushroom pieces in GiGi's scenario -/
theorem gigi_remaining_pieces : 
  remaining_pieces 22 4 38 42 = 8 := by sorry

end NUMINAMATH_CALUDE_gigi_remaining_pieces_l904_90464


namespace NUMINAMATH_CALUDE_ava_lily_trees_l904_90411

/-- The number of apple trees planted by Ava and Lily -/
def total_trees (ava_trees lily_trees : ℕ) : ℕ :=
  ava_trees + lily_trees

/-- Theorem stating the total number of apple trees planted by Ava and Lily -/
theorem ava_lily_trees :
  ∀ (ava_trees lily_trees : ℕ),
    ava_trees = 9 →
    ava_trees = lily_trees + 3 →
    total_trees ava_trees lily_trees = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ava_lily_trees_l904_90411


namespace NUMINAMATH_CALUDE_find_m_l904_90490

theorem find_m (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) :
  m = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l904_90490


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l904_90477

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 3)
  (h2 : new_price = 5) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l904_90477


namespace NUMINAMATH_CALUDE_dot_product_in_triangle_l904_90493

/-- Given a triangle ABC where AB = (2, 3) and AC = (3, 4), prove that the dot product of AB and BC is 5. -/
theorem dot_product_in_triangle (A B C : ℝ × ℝ) : 
  B - A = (2, 3) → C - A = (3, 4) → (B - A) • (C - B) = 5 := by sorry

end NUMINAMATH_CALUDE_dot_product_in_triangle_l904_90493


namespace NUMINAMATH_CALUDE_hall_length_is_30_l904_90491

/-- Represents a rectangular hall with specific properties -/
structure RectangularHall where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_breadth_relation : length = breadth + 5
  area_formula : area = length * breadth

/-- Theorem stating that a rectangular hall with the given properties has a length of 30 meters -/
theorem hall_length_is_30 (hall : RectangularHall) (h : hall.area = 750) : hall.length = 30 := by
  sorry

#check hall_length_is_30

end NUMINAMATH_CALUDE_hall_length_is_30_l904_90491


namespace NUMINAMATH_CALUDE_derivative_of_y_l904_90416

noncomputable def y (x : ℝ) : ℝ := (Real.sin (x^2))^3

theorem derivative_of_y (x : ℝ) :
  deriv y x = 3 * x * Real.sin (x^2) * Real.sin (2 * x^2) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_y_l904_90416


namespace NUMINAMATH_CALUDE_polynomial_identity_l904_90420

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l904_90420


namespace NUMINAMATH_CALUDE_carmen_dogs_l904_90453

def problem (initial_cats : ℕ) (adopted_cats : ℕ) (cat_dog_difference : ℕ) : Prop :=
  let remaining_cats := initial_cats - adopted_cats
  ∃ (dogs : ℕ), remaining_cats = dogs + cat_dog_difference

theorem carmen_dogs : 
  problem 28 3 7 → ∃ (dogs : ℕ), dogs = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_carmen_dogs_l904_90453


namespace NUMINAMATH_CALUDE_altitude_bisector_median_inequality_l904_90429

/-- Triangle structure with altitude, angle bisector, and median from vertex A -/
structure Triangle :=
  (A B C : Point)
  (ha : ℝ) -- altitude from A to BC
  (βa : ℝ) -- angle bisector from A to BC
  (ma : ℝ) -- median from A to BC

/-- Theorem stating the inequality between altitude, angle bisector, and median -/
theorem altitude_bisector_median_inequality (t : Triangle) : t.ha ≤ t.βa ∧ t.βa ≤ t.ma := by
  sorry

end NUMINAMATH_CALUDE_altitude_bisector_median_inequality_l904_90429


namespace NUMINAMATH_CALUDE_andy_inappropriate_joke_demerits_l904_90418

/-- Represents the number of demerits Andy got for making an inappropriate joke -/
def inappropriate_joke_demerits : ℕ := sorry

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of demerits Andy gets per instance of being late -/
def late_demerits_per_instance : ℕ := 2

/-- The number of times Andy was late -/
def late_instances : ℕ := 6

/-- The number of additional demerits Andy can get this month before getting fired -/
def remaining_demerits : ℕ := 23

theorem andy_inappropriate_joke_demerits :
  inappropriate_joke_demerits = 
    max_demerits - remaining_demerits - (late_demerits_per_instance * late_instances) :=
by sorry

end NUMINAMATH_CALUDE_andy_inappropriate_joke_demerits_l904_90418


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l904_90404

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (true_discount : ℝ) (interest_rate : ℝ) (time_period : ℝ) :
  true_discount = 60.00000000000001 →
  interest_rate = 0.12 →
  time_period = 1 →
  let face_value := (true_discount * (1 + interest_rate * time_period)) / (interest_rate * time_period)
  let bankers_discount := face_value * interest_rate * time_period
  bankers_discount - true_discount = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_bankers_gain_calculation_l904_90404


namespace NUMINAMATH_CALUDE_kim_total_water_consumption_l904_90434

/-- The amount of water Kim drinks from various sources -/
def kim_water_consumption (quart_to_ounce : Real) (bottle_quarts : Real) (can_ounces : Real) 
  (shared_bottle_ounces : Real) (jake_fraction : Real) : Real :=
  let bottle_ounces := bottle_quarts * quart_to_ounce
  let kim_shared_fraction := 1 - jake_fraction
  bottle_ounces + can_ounces + (kim_shared_fraction * shared_bottle_ounces)

/-- Theorem stating that Kim's total water consumption is 79.2 ounces -/
theorem kim_total_water_consumption :
  kim_water_consumption 32 1.5 12 32 (2/5) = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_kim_total_water_consumption_l904_90434


namespace NUMINAMATH_CALUDE_yuna_has_biggest_number_l904_90459

-- Define the type for students
inductive Student : Type
  | Yoongi : Student
  | Jungkook : Student
  | Yuna : Student
  | Yoojung : Student

-- Define a function that assigns numbers to students
def studentNumber : Student → Nat
  | Student.Yoongi => 7
  | Student.Jungkook => 6
  | Student.Yuna => 9
  | Student.Yoojung => 8

-- Theorem statement
theorem yuna_has_biggest_number :
  (∀ s : Student, studentNumber s ≤ studentNumber Student.Yuna) ∧
  studentNumber Student.Yuna = 9 := by
  sorry

end NUMINAMATH_CALUDE_yuna_has_biggest_number_l904_90459


namespace NUMINAMATH_CALUDE_tenth_term_of_inverse_proportional_sequence_l904_90401

/-- A sequence where each term after the first is inversely proportional to the preceding term -/
def InverseProportionalSequence (a : ℕ → ℚ) : Prop :=
  ∃ k : ℚ, k ≠ 0 ∧ ∀ n : ℕ, n > 0 → a n * a (n + 1) = k

theorem tenth_term_of_inverse_proportional_sequence
  (a : ℕ → ℚ)
  (h_seq : InverseProportionalSequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 4) :
  a 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_inverse_proportional_sequence_l904_90401


namespace NUMINAMATH_CALUDE_intersection_of_sets_l904_90485

theorem intersection_of_sets : 
  let M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}
  let N : Set ℤ := {x | x^2 = x}
  M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l904_90485


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l904_90448

-- Define an isosceles triangle with side lengths a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ a + c > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
    ((t.a = 3 ∧ t.b = 7) ∨ (t.a = 7 ∧ t.b = 3) ∨ 
     (t.b = 3 ∧ t.c = 7) ∨ (t.b = 7 ∧ t.c = 3) ∨ 
     (t.a = 3 ∧ t.c = 7) ∨ (t.a = 7 ∧ t.c = 3)) →
    perimeter t = 17 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l904_90448


namespace NUMINAMATH_CALUDE_alices_journey_time_l904_90465

/-- Represents the problem of Alice's journey to the library -/
theorem alices_journey_time :
  ∀ (d : ℝ) (r_w : ℝ),
    r_w > 0 →
    (3/4 * d) / r_w = 9 →
    (1/4 * d) / (4 * r_w) + 9 = 9.75 :=
by sorry

end NUMINAMATH_CALUDE_alices_journey_time_l904_90465


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l904_90405

def total_balls : ℕ := 9
def white_balls : ℕ := 5
def black_balls : ℕ := 4

def prob_first_white : ℚ := white_balls / total_balls
def prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)

def prob_two_white : ℚ := prob_first_white * prob_second_white

theorem two_white_balls_probability :
  prob_two_white = 5 / 18 := by sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l904_90405


namespace NUMINAMATH_CALUDE_sock_pair_count_l904_90461

def white_socks : ℕ := 5
def brown_socks : ℕ := 3
def blue_socks : ℕ := 2
def black_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + black_socks

def choose (n k : ℕ) : ℕ := Nat.choose n k

def same_color_pairs : ℕ :=
  choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 + choose black_socks 2

theorem sock_pair_count : same_color_pairs = 15 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l904_90461


namespace NUMINAMATH_CALUDE_chess_tournament_players_l904_90473

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 240) :
  ∃ n : ℕ, n > 0 ∧ 2 * n * (n - 1) = total_games ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l904_90473
