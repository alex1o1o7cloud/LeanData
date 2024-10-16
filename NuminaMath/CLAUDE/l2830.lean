import Mathlib

namespace NUMINAMATH_CALUDE_point_n_from_m_l2830_283094

/-- Given two points M and N in a 2D plane, prove that N can be obtained
    from M by moving 4 units upward. -/
theorem point_n_from_m (M N : ℝ × ℝ) : 
  M = (-1, -1) → N = (-1, 3) → N.2 - M.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_n_from_m_l2830_283094


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2830_283072

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  use -4
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2830_283072


namespace NUMINAMATH_CALUDE_sum_base6_numbers_l2830_283073

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of the given base 6 numbers equals 3153₆ --/
theorem sum_base6_numbers :
  let n1 := [4, 3, 2, 1]  -- 1234₆
  let n2 := [4, 5, 6]     -- 654₆
  let n3 := [1, 2, 3]     -- 321₆
  let n4 := [6, 5]        -- 56₆
  base10ToBase6 (base6ToBase10 n1 + base6ToBase10 n2 + base6ToBase10 n3 + base6ToBase10 n4) = [3, 1, 5, 3] :=
by sorry

end NUMINAMATH_CALUDE_sum_base6_numbers_l2830_283073


namespace NUMINAMATH_CALUDE_square_root_problem_l2830_283078

theorem square_root_problem (x y : ℝ) 
  (h1 : (5 * x - 1).sqrt = 3)
  (h2 : (4 * x + 2 * y + 1)^(1/3) = 1) :
  (4 * x - 2 * y).sqrt = 4 ∨ (4 * x - 2 * y).sqrt = -4 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2830_283078


namespace NUMINAMATH_CALUDE_f_neg_five_eq_twelve_l2830_283076

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem f_neg_five_eq_twelve : f (-5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_five_eq_twelve_l2830_283076


namespace NUMINAMATH_CALUDE_integral_reciprocal_sqrt_one_minus_x_squared_l2830_283070

open Real MeasureTheory

theorem integral_reciprocal_sqrt_one_minus_x_squared : 
  ∫ x in (Set.Icc 0 (1 / Real.sqrt 2)), 1 / ((1 - x^2) * Real.sqrt (1 - x^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_sqrt_one_minus_x_squared_l2830_283070


namespace NUMINAMATH_CALUDE_jake_balloons_l2830_283099

theorem jake_balloons (allan_balloons : ℕ) (difference : ℕ) : 
  allan_balloons = 5 → difference = 2 → allan_balloons - difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_jake_balloons_l2830_283099


namespace NUMINAMATH_CALUDE_principal_is_2500_l2830_283082

/-- Given a simple interest, interest rate, and time period, calculates the principal amount. -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that given the specified conditions, the principal amount is 2500. -/
theorem principal_is_2500 :
  let simple_interest : ℚ := 1000
  let rate : ℚ := 10
  let time : ℚ := 4
  calculate_principal simple_interest rate time = 2500 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_2500_l2830_283082


namespace NUMINAMATH_CALUDE_inequalities_proof_l2830_283003

theorem inequalities_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2830_283003


namespace NUMINAMATH_CALUDE_average_of_three_l2830_283044

theorem average_of_three (total : ℝ) (avg_all : ℝ) (avg_two : ℝ) :
  total = 5 →
  avg_all = 10 →
  avg_two = 19 →
  (total * avg_all - 2 * avg_two) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_l2830_283044


namespace NUMINAMATH_CALUDE_min_squared_distance_to_line_l2830_283006

/-- The minimum squared distance from a point on the line x - y - 1 = 0 to the point (2, 2) -/
theorem min_squared_distance_to_line (x y : ℝ) :
  x - y - 1 = 0 → (∀ x' y' : ℝ, x' - y' - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≤ (x' - 2)^2 + (y' - 2)^2) →
  (x - 2)^2 + (y - 2)^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_line_l2830_283006


namespace NUMINAMATH_CALUDE_one_fourth_of_12_8_l2830_283019

theorem one_fourth_of_12_8 :
  let x : ℚ := 12.8 / 4
  x = 16 / 5 ∧ x = 3 + 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_one_fourth_of_12_8_l2830_283019


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l2830_283091

theorem mississippi_arrangements : 
  (11 : ℕ).factorial / ((4 : ℕ).factorial * (4 : ℕ).factorial * (2 : ℕ).factorial) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l2830_283091


namespace NUMINAMATH_CALUDE_odd_number_between_bounds_l2830_283086

theorem odd_number_between_bounds (N : ℕ) : 
  N % 2 = 1 → (9.5 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 10.5) → N = 39 ∨ N = 41 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_between_bounds_l2830_283086


namespace NUMINAMATH_CALUDE_polynomial_identity_l2830_283097

theorem polynomial_identity (x y : ℝ) : (x - y) * (x^2 + x*y + y^2) = x^3 - y^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2830_283097


namespace NUMINAMATH_CALUDE_circle_radius_l2830_283059

theorem circle_radius (x y : ℝ) : 
  (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68 = 0) → 
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2830_283059


namespace NUMINAMATH_CALUDE_solution_equivalence_l2830_283067

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (Real.sqrt 22, Real.sqrt 22), (-Real.sqrt 22, -Real.sqrt 22),
   (Real.sqrt 20, -Real.sqrt 20), (-Real.sqrt 20, Real.sqrt 20),
   (((-3 + Real.sqrt 5) / 2) * (2 * Real.sqrt (3 + Real.sqrt 5)), 2 * Real.sqrt (3 + Real.sqrt 5)),
   (((-3 + Real.sqrt 5) / 2) * (-2 * Real.sqrt (3 + Real.sqrt 5)), -2 * Real.sqrt (3 + Real.sqrt 5)),
   (((-3 - Real.sqrt 5) / 2) * (2 * Real.sqrt (3 - Real.sqrt 5)), 2 * Real.sqrt (3 - Real.sqrt 5)),
   (((-3 - Real.sqrt 5) / 2) * (-2 * Real.sqrt (3 - Real.sqrt 5)), -2 * Real.sqrt (3 - Real.sqrt 5))}

theorem solution_equivalence :
  {(x, y) : ℝ × ℝ | x^5 = 21*x^3 + y^3 ∧ y^5 = x^3 + 21*y^3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l2830_283067


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2830_283027

/-- Given distinct digits a, b, and c different from 1,
    prove that abb × c = bcb1 implies a = 5, b = 3, and c = 7 -/
theorem multiplication_puzzle :
  ∀ a b c : ℕ,
    a ≠ b → b ≠ c → a ≠ c →
    a ≠ 1 → b ≠ 1 → c ≠ 1 →
    a < 10 → b < 10 → c < 10 →
    (100 * a + 10 * b + b) * c = 1000 * b + 100 * c + 10 * b + 1 →
    a = 5 ∧ b = 3 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2830_283027


namespace NUMINAMATH_CALUDE_gwen_candy_weight_l2830_283074

/-- The amount of candy Gwen received given the total amount and Frank's amount -/
def gwens_candy (total frank : ℕ) : ℕ := total - frank

/-- Theorem stating that Gwen received 7 pounds of candy -/
theorem gwen_candy_weight :
  let total := 17
  let frank := 10
  gwens_candy total frank = 7 := by sorry

end NUMINAMATH_CALUDE_gwen_candy_weight_l2830_283074


namespace NUMINAMATH_CALUDE_min_area_over_sqrt_t_l2830_283052

/-- The area bounded by the tangent lines and the parabola -/
noncomputable def S (t : ℝ) : ℝ := (2 / 3) * (1 + t^2)^(3/2)

/-- The main theorem statement -/
theorem min_area_over_sqrt_t (t : ℝ) (ht : t > 0) :
  ∃ (min_value : ℝ), min_value = (2 * 6^(3/2)) / (3 * 5^(5/4)) ∧
  ∀ (t : ℝ), t > 0 → S t / Real.sqrt t ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_area_over_sqrt_t_l2830_283052


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2830_283054

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop := (3 - k) * (k - 1) < 0

/-- The condition k > 3 -/
def condition (k : ℝ) : Prop := k > 3

theorem hyperbola_condition (k : ℝ) :
  (condition k → is_hyperbola k) ∧ ¬(is_hyperbola k → condition k) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2830_283054


namespace NUMINAMATH_CALUDE_problem_solution_l2830_283012

theorem problem_solution (x : ℝ) : (3 * x + 20 = (1 / 3) * (7 * x + 60)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2830_283012


namespace NUMINAMATH_CALUDE_sodium_hypochlorite_weight_approx_l2830_283017

/-- The atomic weight of sodium in g/mol -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of sodium hypochlorite (NaOCl) in g/mol -/
def sodium_hypochlorite_weight : ℝ := sodium_weight + oxygen_weight + chlorine_weight

/-- The given molecular weight of a certain substance -/
def given_weight : ℝ := 74

/-- Theorem stating that the molecular weight of sodium hypochlorite is approximately equal to the given weight -/
theorem sodium_hypochlorite_weight_approx : 
  ∃ ε > 0, |sodium_hypochlorite_weight - given_weight| < ε :=
sorry

end NUMINAMATH_CALUDE_sodium_hypochlorite_weight_approx_l2830_283017


namespace NUMINAMATH_CALUDE_solution_set_a_1_find_a_l2830_283018

-- Define the function f(x) = |x - a| + x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Theorem 1: Solution set for a = 1
theorem solution_set_a_1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Theorem 2: Finding a when solution set is {x | x ≥ 2}
theorem find_a (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3 * x} = {x : ℝ | x ≥ 2}) → a = 6 :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_1_find_a_l2830_283018


namespace NUMINAMATH_CALUDE_unit_digit_of_product_is_zero_l2830_283036

/-- Get the unit digit of a natural number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The product of the given numbers -/
def productOfNumbers : ℕ := 785846 * 1086432 * 4582735 * 9783284 * 5167953 * 3821759 * 7594683

theorem unit_digit_of_product_is_zero :
  unitDigit productOfNumbers = 0 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_product_is_zero_l2830_283036


namespace NUMINAMATH_CALUDE_simplify_expression_l2830_283023

theorem simplify_expression :
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2830_283023


namespace NUMINAMATH_CALUDE_percent_relation_l2830_283025

theorem percent_relation (x y z w v : ℝ) 
  (hx : x = 1.3 * y) 
  (hy : y = 0.6 * z) 
  (hw : w = 1.25 * x) 
  (hv : v = 0.85 * w) : 
  v = 0.82875 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l2830_283025


namespace NUMINAMATH_CALUDE_always_defined_division_by_two_l2830_283045

theorem always_defined_division_by_two (a : ℝ) : ∃ (x : ℝ), x = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_always_defined_division_by_two_l2830_283045


namespace NUMINAMATH_CALUDE_volleyball_net_theorem_l2830_283031

/-- Represents a rectangular grid -/
structure RectangularGrid where
  rows : ℕ
  cols : ℕ

/-- Calculates the number of vertices in a rectangular grid -/
def num_vertices (g : RectangularGrid) : ℕ := (g.rows + 1) * (g.cols + 1)

/-- Calculates the number of edges in a rectangular grid -/
def num_edges (g : RectangularGrid) : ℕ := g.rows * (g.cols + 1) + g.cols * (g.rows + 1)

/-- Calculates the maximum number of removable edges while keeping the grid connected -/
def max_removable_edges (g : RectangularGrid) : ℕ := 
  num_edges g - (num_vertices g - 1)

/-- Theorem stating that for a 50 × 600 grid, the maximum number of removable edges is 30,000 -/
theorem volleyball_net_theorem :
  let g : RectangularGrid := ⟨50, 600⟩
  max_removable_edges g = 30000 := by sorry

end NUMINAMATH_CALUDE_volleyball_net_theorem_l2830_283031


namespace NUMINAMATH_CALUDE_cubic_root_h_value_l2830_283095

theorem cubic_root_h_value : ∀ h : ℚ, 
  (3 : ℚ)^3 + h * 3 - 20 = 0 → h = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_h_value_l2830_283095


namespace NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l2830_283032

/-- The surface area of the Earth in square kilometers -/
def earth_surface_area : ℝ := 510000000

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Conversion of a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem earth_surface_area_scientific_notation :
  to_scientific_notation earth_surface_area = ScientificNotation.mk 5.1 8 sorry := by
  sorry

end NUMINAMATH_CALUDE_earth_surface_area_scientific_notation_l2830_283032


namespace NUMINAMATH_CALUDE_inequality_solution_l2830_283093

theorem inequality_solution (x : ℝ) : 
  (|(7 - 2*x) / 4| < 3) ↔ (-5/2 < x ∧ x < 19/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2830_283093


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2830_283065

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2830_283065


namespace NUMINAMATH_CALUDE_marbles_from_henry_l2830_283068

theorem marbles_from_henry (initial_marbles end_marbles marbles_from_henry : ℕ) 
  (h1 : initial_marbles = 95)
  (h2 : end_marbles = 104)
  (h3 : end_marbles = initial_marbles + marbles_from_henry) :
  marbles_from_henry = 9 := by
  sorry

end NUMINAMATH_CALUDE_marbles_from_henry_l2830_283068


namespace NUMINAMATH_CALUDE_gain_percent_for_fifty_cost_twenty_eight_sell_l2830_283030

/-- If the cost price of 50 articles equals the selling price of 28 articles,
    then the gain percent is 78.57%. -/
theorem gain_percent_for_fifty_cost_twenty_eight_sell :
  ∀ (C S : ℝ), C > 0 → S > 0 →
  50 * C = 28 * S →
  (S - C) / C * 100 = 78.57 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_for_fifty_cost_twenty_eight_sell_l2830_283030


namespace NUMINAMATH_CALUDE_special_polyhedron_value_l2830_283089

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  T : ℕ
  P : ℕ
  is_convex : Prop
  face_count : faces = 32
  face_types : faces = triangles + pentagons
  vertex_config : Prop
  euler_formula : vertices - edges + faces = 2

/-- Theorem stating the specific value for the polyhedron -/
theorem special_polyhedron_value (poly : SpecialPolyhedron) :
  100 * poly.P + 10 * poly.T + poly.vertices = 250 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_value_l2830_283089


namespace NUMINAMATH_CALUDE_log_equation_solution_l2830_283058

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_solution (m n b : ℝ) (h : lg m = b - lg n) : m = 10^b / n := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2830_283058


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2830_283010

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (5 * x + 9) = 12) ∧ (x = 27) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2830_283010


namespace NUMINAMATH_CALUDE_test_score_calculation_l2830_283098

theorem test_score_calculation (total_questions correct_answers incorrect_answers score : ℕ) : 
  total_questions = 100 →
  correct_answers + incorrect_answers = total_questions →
  score = correct_answers - 2 * incorrect_answers →
  score = 70 →
  correct_answers = 90 := by
sorry

end NUMINAMATH_CALUDE_test_score_calculation_l2830_283098


namespace NUMINAMATH_CALUDE_even_factors_count_l2830_283034

/-- The number of even natural-number factors of 2^2 * 3^1 * 7^2 -/
def num_even_factors : ℕ := 12

/-- The prime factorization of n -/
def n : ℕ := 2^2 * 3^1 * 7^2

/-- A function that counts the number of even natural-number factors of n -/
def count_even_factors (n : ℕ) : ℕ := sorry

theorem even_factors_count :
  count_even_factors n = num_even_factors := by sorry

end NUMINAMATH_CALUDE_even_factors_count_l2830_283034


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l2830_283050

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 * (1000 / 3600))  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * (1000 / 3600))  -- 45 km/hr in m/s
  (h3 : train_length = 120)                -- 120 m
  (h4 : initial_distance = 180)            -- 180 m
  : (initial_distance + train_length) / (train_speed - jogger_speed) = 30 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_jogger_time_l2830_283050


namespace NUMINAMATH_CALUDE_jack_email_difference_l2830_283062

/-- Given the number of emails Jack received at different times of the day,
    prove that he received 2 more emails in the morning than in the afternoon. -/
theorem jack_email_difference (morning afternoon evening : ℕ) 
    (h1 : morning = 5)
    (h2 : afternoon = 3)
    (h3 : evening = 16) :
    morning - afternoon = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_email_difference_l2830_283062


namespace NUMINAMATH_CALUDE_bank_queue_theorem_l2830_283042

/-- Represents a bank queue with simple and long operations -/
structure BankQueue where
  total_people : Nat
  simple_ops : Nat
  long_ops : Nat
  simple_time : Nat
  long_time : Nat

/-- Calculates the minimum wasted person-minutes -/
def min_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the maximum wasted person-minutes -/
def max_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Calculates the expected wasted person-minutes assuming random order -/
def expected_wasted_time (q : BankQueue) : Nat :=
  sorry

/-- Main theorem about the bank queue problem -/
theorem bank_queue_theorem (q : BankQueue) 
  (h1 : q.total_people = 8)
  (h2 : q.simple_ops = 5)
  (h3 : q.long_ops = 3)
  (h4 : q.simple_time = 1)
  (h5 : q.long_time = 5) :
  min_wasted_time q = 40 ∧ 
  max_wasted_time q = 100 ∧ 
  expected_wasted_time q = 84 := by
  sorry

end NUMINAMATH_CALUDE_bank_queue_theorem_l2830_283042


namespace NUMINAMATH_CALUDE_translated_parabola_vertex_l2830_283037

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation
def translation_left : ℝ := 3
def translation_down : ℝ := 2

-- Theorem stating the vertex of the translated parabola
theorem translated_parabola_vertex :
  let vertex_x : ℝ := 2 - translation_left
  let vertex_y : ℝ := original_parabola 2 - translation_down
  (vertex_x, vertex_y) = (-1, -4) := by sorry

end NUMINAMATH_CALUDE_translated_parabola_vertex_l2830_283037


namespace NUMINAMATH_CALUDE_total_water_consumed_water_consumed_is_686_l2830_283007

/-- Represents a medication schedule --/
structure MedicationSchedule where
  name : String
  timesPerDay : Nat
  waterPerDose : Nat

/-- Represents missed doses for a medication --/
structure MissedDoses where
  medication : String
  count : Nat

/-- Calculates the total water consumed for a medication over two weeks --/
def waterConsumedForMedication (schedule : MedicationSchedule) : Nat :=
  schedule.timesPerDay * schedule.waterPerDose * 7 * 2

/-- Calculates the water missed due to skipped doses --/
def waterMissedForMedication (schedule : MedicationSchedule) (missed : Nat) : Nat :=
  schedule.waterPerDose * missed

/-- The main theorem to prove --/
theorem total_water_consumed 
  (schedules : List MedicationSchedule)
  (missedDoses : List MissedDoses) : Nat :=
  let totalWater := schedules.map waterConsumedForMedication |>.sum
  let missedWater := missedDoses.map (fun m => 
    let schedule := schedules.find? (fun s => s.name == m.medication)
    match schedule with
    | some s => waterMissedForMedication s m.count
    | none => 0
  ) |>.sum
  totalWater - missedWater

/-- The specific medication schedules --/
def medicationSchedules : List MedicationSchedule := [
  { name := "A", timesPerDay := 3, waterPerDose := 4 },
  { name := "B", timesPerDay := 4, waterPerDose := 5 },
  { name := "C", timesPerDay := 2, waterPerDose := 6 },
  { name := "D", timesPerDay := 1, waterPerDose := 8 }
]

/-- The specific missed doses --/
def missedDosesList : List MissedDoses := [
  { medication := "A", count := 3 },
  { medication := "B", count := 2 },
  { medication := "C", count := 2 },
  { medication := "D", count := 1 }
]

/-- The main theorem for this specific problem --/
theorem water_consumed_is_686 : 
  total_water_consumed medicationSchedules missedDosesList = 686 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumed_water_consumed_is_686_l2830_283007


namespace NUMINAMATH_CALUDE_rectangle_division_l2830_283000

theorem rectangle_division (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  let x := b / 3
  let y := a / 3
  (x * a) / ((b - x) * a) = 1 / 2 ∧
  (2 * a + 2 * x) / (2 * a + 2 * (b - x)) = 3 / 5 →
  (y * b) / ((a - y) * b) = 1 / 2 →
  (2 * y + 2 * b) / (2 * (a - y) + 2 * b) = 20 / 19 :=
by sorry


end NUMINAMATH_CALUDE_rectangle_division_l2830_283000


namespace NUMINAMATH_CALUDE_guaranteed_babysitting_hours_is_eight_l2830_283043

/-- Calculates the number of guaranteed babysitting hours on Saturday given Donna's work schedule and earnings. -/
def guaranteed_babysitting_hours (
  dog_walking_hours : ℕ)
  (dog_walking_rate : ℚ)
  (dog_walking_days : ℕ)
  (card_shop_hours : ℕ)
  (card_shop_rate : ℚ)
  (card_shop_days : ℕ)
  (babysitting_rate : ℚ)
  (total_earnings : ℚ) : ℚ :=
  let dog_walking_earnings := ↑dog_walking_hours * dog_walking_rate * ↑dog_walking_days
  let card_shop_earnings := ↑card_shop_hours * card_shop_rate * ↑card_shop_days
  let other_earnings := dog_walking_earnings + card_shop_earnings
  let babysitting_earnings := total_earnings - other_earnings
  babysitting_earnings / babysitting_rate

theorem guaranteed_babysitting_hours_is_eight :
  guaranteed_babysitting_hours 2 10 5 2 (25/2) 5 10 305 = 8 := by
  sorry

end NUMINAMATH_CALUDE_guaranteed_babysitting_hours_is_eight_l2830_283043


namespace NUMINAMATH_CALUDE_solve_for_r_l2830_283009

theorem solve_for_r (k : ℝ) (r : ℝ) 
  (h1 : 5 = k * 3^r) 
  (h2 : 45 = k * 9^r) : 
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_r_l2830_283009


namespace NUMINAMATH_CALUDE_algebraic_expression_evaluation_l2830_283069

theorem algebraic_expression_evaluation : 
  let x : ℚ := -1
  let y : ℚ := 1/2
  2 * (x^2 - 5*x*y) - 3 * (x^2 - 6*x*y) = 3 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_evaluation_l2830_283069


namespace NUMINAMATH_CALUDE_salary_distribution_l2830_283026

/-- Represents the salary distribution problem for three teams of workers --/
theorem salary_distribution
  (total_value : ℝ)
  (team1_people team1_days : ℕ)
  (team2_people team2_days : ℕ)
  (team3_days : ℕ)
  (team3_people_ratio : ℝ)
  (h1 : total_value = 325500)
  (h2 : team1_people = 15)
  (h3 : team1_days = 21)
  (h4 : team2_people = 14)
  (h5 : team2_days = 25)
  (h6 : team3_days = 20)
  (h7 : team3_people_ratio = 1.4) :
  ∃ (salary_per_day : ℝ),
    let team1_salary := salary_per_day * team1_people * team1_days
    let team2_salary := salary_per_day * team2_people * team2_days
    let team3_salary := salary_per_day * (team3_people_ratio * team1_people) * team3_days
    team1_salary + team2_salary + team3_salary = total_value ∧
    team1_salary = 94500 ∧
    team2_salary = 105000 ∧
    team3_salary = 126000 :=
by sorry

end NUMINAMATH_CALUDE_salary_distribution_l2830_283026


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2830_283057

theorem regular_polygon_sides (n : ℕ) (h : n > 0) :
  (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2830_283057


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l2830_283004

/-- Two lines intersect in the first quadrant if and only if k is in the open interval (-2/3, 2) -/
theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + k + 2 ∧ y = -2 * x + 4) ↔ 
  -2/3 < k ∧ k < 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l2830_283004


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l2830_283015

/-- Strawberry picking problem -/
theorem strawberry_picking_problem 
  (brother_baskets : ℕ) 
  (strawberries_per_basket : ℕ) 
  (kimberly_multiplier : ℕ) 
  (equal_share : ℕ) 
  (family_members : ℕ) 
  (h1 : brother_baskets = 3)
  (h2 : strawberries_per_basket = 15)
  (h3 : kimberly_multiplier = 8)
  (h4 : equal_share = 168)
  (h5 : family_members = 4) :
  kimberly_multiplier * (brother_baskets * strawberries_per_basket) - 
  (family_members * equal_share - 
   kimberly_multiplier * (brother_baskets * strawberries_per_basket) - 
   (brother_baskets * strawberries_per_basket)) = 93 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_problem_l2830_283015


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l2830_283066

theorem consecutive_integers_square_difference :
  ∃ n : ℕ, 
    (n > 0) ∧ 
    (n + (n + 1) + (n + 2) < 150) ∧ 
    ((n + 2)^2 - n^2 = 144) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l2830_283066


namespace NUMINAMATH_CALUDE_red_ants_count_l2830_283081

theorem red_ants_count (total : ℕ) (black : ℕ) (red : ℕ) : 
  total = 900 → black = 487 → total = red + black → red = 413 := by
  sorry

end NUMINAMATH_CALUDE_red_ants_count_l2830_283081


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l2830_283087

/-- Given an angle α whose terminal side passes through the point (-a, 2a) where a < 0,
    prove that sin α = -2√5/5 -/
theorem sin_alpha_for_point (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : ∃ k : ℝ, k > 0 ∧ k * Real.cos α = -a ∧ k * Real.sin α = 2*a) : 
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l2830_283087


namespace NUMINAMATH_CALUDE_ab_value_l2830_283024

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2830_283024


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l2830_283060

theorem largest_integer_less_than_100_remainder_5_mod_8 : 
  ∀ n : ℕ, n < 100 ∧ n % 8 = 5 → n ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_8_l2830_283060


namespace NUMINAMATH_CALUDE_smallest_natural_numbers_for_nested_root_l2830_283092

theorem smallest_natural_numbers_for_nested_root (a b : ℕ) : 
  (b > 1) → 
  (Real.sqrt (a * Real.sqrt (a * Real.sqrt a)) = b) → 
  (∀ a' b' : ℕ, b' > 1 → Real.sqrt (a' * Real.sqrt (a' * Real.sqrt a')) = b' → a ≤ a' ∧ b ≤ b') →
  a = 256 ∧ b = 128 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_numbers_for_nested_root_l2830_283092


namespace NUMINAMATH_CALUDE_three_digit_product_sum_l2830_283040

theorem three_digit_product_sum (P A U : ℕ) : 
  P ≠ A → P ≠ U → A ≠ U →
  P ≥ 1 → P ≤ 9 →
  A ≥ 0 → A ≤ 9 →
  U ≥ 0 → U ≤ 9 →
  100 * P + 10 * A + U ≥ 100 →
  100 * P + 10 * A + U ≤ 999 →
  (P + A + U) * P * A * U = 300 →
  ∃ (PAU : ℕ), PAU = 100 * P + 10 * A + U ∧ 
               (PAU.div 100 + (PAU.mod 100).div 10 + PAU.mod 10) * 
               PAU.div 100 * (PAU.mod 100).div 10 * PAU.mod 10 = 300 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_product_sum_l2830_283040


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l2830_283016

theorem complex_sum_of_powers (i : ℂ) : i^2 = -1 → i + i^2 + i^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l2830_283016


namespace NUMINAMATH_CALUDE_common_root_pairs_l2830_283048

theorem common_root_pairs (n : ℕ) (hn : n > 1) :
  ∀ (a b : ℤ), (∃ (x : ℝ), x^n + a*x - 2008 = 0 ∧ x^n + b*x - 2009 = 0) ↔
    ((a = 2007 ∧ b = 2008) ∨ (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end NUMINAMATH_CALUDE_common_root_pairs_l2830_283048


namespace NUMINAMATH_CALUDE_rogers_cookie_price_l2830_283005

/-- Represents a cookie shape -/
inductive CookieShape
| Trapezoid
| Rectangle

/-- Represents a baker's cookie production -/
structure Baker where
  name : String
  shape : CookieShape
  numCookies : ℕ
  pricePerCookie : ℕ

/-- Calculates the total earnings for a baker -/
def totalEarnings (baker : Baker) : ℕ :=
  baker.numCookies * baker.pricePerCookie

/-- Theorem: Roger's cookie price for equal earnings -/
theorem rogers_cookie_price 
  (art roger : Baker)
  (h1 : art.shape = CookieShape.Trapezoid)
  (h2 : roger.shape = CookieShape.Rectangle)
  (h3 : art.numCookies = 12)
  (h4 : art.pricePerCookie = 60)
  (h5 : totalEarnings art = totalEarnings roger) :
  roger.pricePerCookie = 40 :=
sorry

end NUMINAMATH_CALUDE_rogers_cookie_price_l2830_283005


namespace NUMINAMATH_CALUDE_remainder_problem_l2830_283056

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : k < 42) 
  (h3 : k % 5 = 2) (h4 : k % 6 = 5) : k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2830_283056


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2830_283047

theorem sum_of_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y + x) : x + y + z = 17 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2830_283047


namespace NUMINAMATH_CALUDE_fraction_inequality_l2830_283014

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / c > a / d := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2830_283014


namespace NUMINAMATH_CALUDE_max_value_product_sum_l2830_283049

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 15 →
    A' * M' * C' + A' * M' + M' * C' + C' * A' ≤ A * M * C + A * M + M * C + C * A) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l2830_283049


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l2830_283033

/-- Represents a cylinder with given dimensions and arc --/
structure Cylinder :=
  (radius : ℝ)
  (height : ℝ)
  (arc_angle : ℝ)

/-- Calculates the area of the cross-section of the cylinder --/
def cross_section_area (c : Cylinder) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime --/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Main theorem about the cross-section area of the specific cylinder --/
theorem cylinder_cross_section_area :
  let c := Cylinder.mk 7 10 (150 * π / 180)
  ∃ (d e : ℕ) (f : ℕ),
    cross_section_area c = d * π + e * Real.sqrt f ∧
    not_divisible_by_square_prime f ∧
    d = 60 ∧ e = 70 ∧ f = 3 := by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l2830_283033


namespace NUMINAMATH_CALUDE_perpendicular_condition_false_l2830_283022

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpLine : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpPlane : Plane → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (subset : Line → Plane → Prop)

theorem perpendicular_condition_false
  (α β : Plane) (b : Line)
  (h_diff : α ≠ β)
  (h_subset : subset b β) :
  ¬(∀ (α β : Plane) (b : Line),
    α ≠ β →
    subset b β →
    (perpLine b α → perpPlane α β) ∧
    ¬(perpPlane α β → perpLine b α)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_false_l2830_283022


namespace NUMINAMATH_CALUDE_unattainable_value_l2830_283046

/-- The function f(x) = (1-2x) / (3x+4) cannot attain the value -2/3 for any real x ≠ -4/3. -/
theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  (1 - 2*x) / (3*x + 4) ≠ -2/3 := by
  sorry


end NUMINAMATH_CALUDE_unattainable_value_l2830_283046


namespace NUMINAMATH_CALUDE_lesser_fraction_l2830_283085

theorem lesser_fraction (x y : ℚ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2830_283085


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2830_283064

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a, a ≤ 0 ∧ a^2 + a ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2830_283064


namespace NUMINAMATH_CALUDE_hospital_workers_count_l2830_283008

theorem hospital_workers_count :
  let total_workers : ℕ := 2 + 3  -- Jack, Jill, and 3 others
  let interview_size : ℕ := 2
  let prob_jack_and_jill : ℚ := 1 / 10  -- 0.1 as a rational number
  total_workers = 5 ∧
  interview_size = 2 ∧
  prob_jack_and_jill = 1 / Nat.choose total_workers interview_size :=
by sorry

end NUMINAMATH_CALUDE_hospital_workers_count_l2830_283008


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2830_283038

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2830_283038


namespace NUMINAMATH_CALUDE_discount_rate_pony_jeans_discount_rate_pony_jeans_proof_l2830_283055

theorem discount_rate_pony_jeans : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (fox_price pony_price total_savings discount_sum : ℝ) =>
    let fox_pairs : ℝ := 3
    let pony_pairs : ℝ := 2
    let total_pairs : ℝ := fox_pairs + pony_pairs
    fox_price = 15 ∧ 
    pony_price = 18 ∧ 
    total_savings = 9 ∧ 
    discount_sum = 22 ∧
    ∃ (fox_discount pony_discount : ℝ),
      fox_discount + pony_discount = discount_sum ∧
      fox_pairs * (fox_discount / 100 * fox_price) + 
        pony_pairs * (pony_discount / 100 * pony_price) = total_savings ∧
      pony_discount = 10

theorem discount_rate_pony_jeans_proof 
  (fox_price pony_price total_savings discount_sum : ℝ) :
  discount_rate_pony_jeans fox_price pony_price total_savings discount_sum :=
by sorry

end NUMINAMATH_CALUDE_discount_rate_pony_jeans_discount_rate_pony_jeans_proof_l2830_283055


namespace NUMINAMATH_CALUDE_exists_meaningful_sqrt_l2830_283088

theorem exists_meaningful_sqrt : ∃ x : ℝ, x - 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_meaningful_sqrt_l2830_283088


namespace NUMINAMATH_CALUDE_smallest_marble_count_l2830_283041

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing two marbles of one color and two of another --/
def prob_two_two (m : MarbleCount) (c1 c2 : ℕ) : ℚ :=
  (c1.choose 2 * c2.choose 2 : ℚ) / (m.red + m.white + m.blue + m.green).choose 4

/-- Calculates the probability of drawing one marble of each color --/
def prob_one_each (m : MarbleCount) : ℚ :=
  (m.red * m.white * m.blue * m.green : ℚ) / (m.red + m.white + m.blue + m.green).choose 4

/-- Checks if the probabilities of the three events are equal --/
def probabilities_equal (m : MarbleCount) : Prop :=
  prob_two_two m m.red m.blue = prob_two_two m m.white m.green ∧
  prob_two_two m m.red m.blue = prob_one_each m

/-- The theorem stating that 10 is the smallest number of marbles satisfying the conditions --/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), 
    (m.red + m.white + m.blue + m.green = 10) ∧ 
    probabilities_equal m ∧
    (∀ (n : MarbleCount), 
      (n.red + n.white + n.blue + n.green < 10) → ¬probabilities_equal n) :=
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l2830_283041


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l2830_283053

theorem cube_roots_of_unity_sum (ω ω_bar : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω_bar = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω_bar^3 = 1 →
  ω^9 + ω_bar^9 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l2830_283053


namespace NUMINAMATH_CALUDE_society_member_sum_or_double_l2830_283080

theorem society_member_sum_or_double {n : ℕ} (hn : n = 1978) :
  ∀ (f : Fin n → Fin 6),
  ∃ (i : Fin 6) (a b c : Fin n),
    f a = i ∧ f b = i ∧ f c = i ∧
    (a.val + 1 = b.val + c.val + 2 ∨ a.val + 1 = 2 * (b.val + 1)) := by
  sorry


end NUMINAMATH_CALUDE_society_member_sum_or_double_l2830_283080


namespace NUMINAMATH_CALUDE_simplify_expression_l2830_283028

theorem simplify_expression : 8 * (15 / 11) * (-25 / 40) = -15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2830_283028


namespace NUMINAMATH_CALUDE_people_speaking_neither_language_l2830_283071

theorem people_speaking_neither_language (total : ℕ) (latin : ℕ) (french : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_both : both = 9)
  : total - (latin + french - both) = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_speaking_neither_language_l2830_283071


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2830_283002

theorem quadratic_inequality_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) ↔ m > 13 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2830_283002


namespace NUMINAMATH_CALUDE_max_y_value_l2830_283061

/-- Given that x and y are negative integers satisfying y = 10x / (10 - x), 
    the maximum value of y is -5 -/
theorem max_y_value (x y : ℤ) 
  (h1 : x < 0) 
  (h2 : y < 0) 
  (h3 : y = 10 * x / (10 - x)) : 
  (∀ z : ℤ, z < 0 ∧ ∃ w : ℤ, w < 0 ∧ z = 10 * w / (10 - w) → z ≤ -5) ∧ 
  (∃ u : ℤ, u < 0 ∧ -5 = 10 * u / (10 - u)) :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l2830_283061


namespace NUMINAMATH_CALUDE_monic_quadratic_unique_l2830_283079

/-- A monic quadratic polynomial is a polynomial of the form x^2 + bx + c -/
def MonicQuadraticPolynomial (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem monic_quadratic_unique (b c : ℝ) :
  let g := MonicQuadraticPolynomial b c
  g 0 = 8 ∧ g 1 = 14 → b = 5 ∧ c = 8 := by sorry

end NUMINAMATH_CALUDE_monic_quadratic_unique_l2830_283079


namespace NUMINAMATH_CALUDE_voldemort_remaining_calories_voldemort_specific_remaining_calories_l2830_283077

/-- Calculates the remaining calories Voldemort can consume given his intake and limit -/
theorem voldemort_remaining_calories (cake_calories : ℕ) (chips_calories : ℕ) 
  (coke_calories : ℕ) (breakfast_calories : ℕ) (lunch_calories : ℕ) 
  (daily_limit : ℕ) : ℕ :=
  by
  have dinner_calories : ℕ := cake_calories + chips_calories + coke_calories
  have breakfast_lunch_calories : ℕ := breakfast_calories + lunch_calories
  have total_consumed : ℕ := dinner_calories + breakfast_lunch_calories
  exact daily_limit - total_consumed

/-- Proves that Voldemort's remaining calories is 525 given specific intake values -/
theorem voldemort_specific_remaining_calories : 
  voldemort_remaining_calories 110 310 215 560 780 2500 = 525 :=
by
  sorry

end NUMINAMATH_CALUDE_voldemort_remaining_calories_voldemort_specific_remaining_calories_l2830_283077


namespace NUMINAMATH_CALUDE_positive_sum_inequalities_l2830_283063

theorem positive_sum_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 4) : 
  a^2 + b^2/4 + c^2/9 ≥ 8/7 ∧ 
  1/(a+c) + 1/(a+b) + 1/(b+c) ≥ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_inequalities_l2830_283063


namespace NUMINAMATH_CALUDE_monkey_peaches_l2830_283020

theorem monkey_peaches (x : ℕ) : 
  (x / 2 - 12 + (x / 2 + 12) / 2 + 12 = x - 19) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_monkey_peaches_l2830_283020


namespace NUMINAMATH_CALUDE_percentage_difference_l2830_283029

theorem percentage_difference (x y : ℝ) (h : x = 6 * y) :
  (x - y) / x * 100 = 500 / 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2830_283029


namespace NUMINAMATH_CALUDE_direct_proportion_properties_l2830_283090

/-- A function representing direct proportion --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Theorem stating the properties of the function f --/
theorem direct_proportion_properties :
  ∀ k : ℝ, k ≠ 0 →
  (f k 2 = 4) →
  ∃ a : ℝ, (f k a = 3 ∧ k = 2 ∧ a = 3/2) := by
  sorry

#check direct_proportion_properties

end NUMINAMATH_CALUDE_direct_proportion_properties_l2830_283090


namespace NUMINAMATH_CALUDE_samanthas_number_l2830_283021

theorem samanthas_number (x : ℚ) : 5 * ((3 * x + 6) / 2) = 100 → x = 34 / 3 := by
  sorry

end NUMINAMATH_CALUDE_samanthas_number_l2830_283021


namespace NUMINAMATH_CALUDE_helen_cookies_l2830_283051

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 19

/-- The number of chocolate chip cookies Helen baked this morning -/
def today_chocolate : ℕ := 237

/-- The difference between the number of chocolate chip cookies and raisin cookies Helen baked -/
def chocolate_raisin_diff : ℕ := 25

/-- The number of raisin cookies Helen baked -/
def raisin_cookies : ℕ := 231

theorem helen_cookies :
  raisin_cookies = (yesterday_chocolate + today_chocolate) - chocolate_raisin_diff := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l2830_283051


namespace NUMINAMATH_CALUDE_abs_negative_seventeen_l2830_283096

theorem abs_negative_seventeen : |(-17 : ℤ)| = 17 := by sorry

end NUMINAMATH_CALUDE_abs_negative_seventeen_l2830_283096


namespace NUMINAMATH_CALUDE_range_of_b_and_m_l2830_283039

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x - 1

-- Define the set of b values
def B : Set ℝ := {b | b < 0 ∨ b > 4}

-- Define the function F
def F (x m : ℝ) : ℝ := x^2 - m*(x - 1) + 1 - m - m^2

-- Define the set of m values
def M : Set ℝ := {m | -Real.sqrt (4/5) ≤ m ∧ m ≤ Real.sqrt (4/5) ∨ m ≥ 2}

theorem range_of_b_and_m :
  (∀ b : ℝ, (∃ x : ℝ, f x < b * g x) ↔ b ∈ B) ∧
  (∀ m : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → |F x m| < |F y m|) → m ∈ M) :=
sorry

end NUMINAMATH_CALUDE_range_of_b_and_m_l2830_283039


namespace NUMINAMATH_CALUDE_dad_age_is_36_l2830_283075

-- Define the current ages
def talia_age : ℕ := 13
def mom_age : ℕ := 39
def dad_age : ℕ := 36
def grandpa_age : ℕ := 18

-- Define the theorem
theorem dad_age_is_36 :
  (talia_age + 7 = 20) ∧
  (mom_age = 3 * talia_age) ∧
  (dad_age + 2 = grandpa_age + 2 + 5) ∧
  (dad_age + 3 = mom_age) ∧
  (grandpa_age + 3 = (mom_age + 3) / 2) →
  dad_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_dad_age_is_36_l2830_283075


namespace NUMINAMATH_CALUDE_range_of_a_l2830_283001

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) : (¬(p a) ∧ q a) → (1 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2830_283001


namespace NUMINAMATH_CALUDE_max_sum_abc_l2830_283011

def An (a n : ℕ) : ℚ := a * (10^n - 1) / 9
def Bn (b n : ℕ) : ℚ := b * (10^(2*n) - 1) / 9
def Cn (c n : ℕ) : ℚ := c * (10^(2*n) - 1) / 9

theorem max_sum_abc (a b c n : ℕ) :
  (a ∈ Finset.range 10 ∧ a ≠ 0) →
  (b ∈ Finset.range 10 ∧ b ≠ 0) →
  (c ∈ Finset.range 10 ∧ c ≠ 0) →
  (∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ Cn c n₁ - Bn b n₁ = (An a n₁)^2 ∧ Cn c n₂ - Bn b n₂ = (An a n₂)^2) →
  a + b + c ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abc_l2830_283011


namespace NUMINAMATH_CALUDE_ellipse_equation_and_slope_l2830_283084

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- Theorem about the equation of the ellipse and the slope of line l -/
theorem ellipse_equation_and_slope (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.eccentricity = Real.sqrt 3 / 2)
  (h4 : e.passes_through = (Real.sqrt 2, Real.sqrt 2 / 2)) :
  (∃ (x y : ℝ), x^2 / 4 + y^2 = 1) ∧ 
  (∃ (k : ℝ), k = 1/2 ∨ k = -1/2) := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_slope_l2830_283084


namespace NUMINAMATH_CALUDE_remaining_card_is_seven_l2830_283013

def cards : List Nat := [2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_relatively_prime (a b : Nat) : Prop := Nat.gcd a b = 1

def is_consecutive (a b : Nat) : Prop := a.succ = b ∨ b.succ = a

def is_composite (n : Nat) : Prop := n > 3 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def is_multiple (a b : Nat) : Prop := ∃ k, k > 1 ∧ (a = k * b ∨ b = k * a)

theorem remaining_card_is_seven (A B C D : List Nat) : 
  A.length = 2 ∧ B.length = 2 ∧ C.length = 2 ∧ D.length = 2 →
  (∀ x ∈ A, x ∈ cards) ∧ (∀ x ∈ B, x ∈ cards) ∧ (∀ x ∈ C, x ∈ cards) ∧ (∀ x ∈ D, x ∈ cards) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  (∀ a ∈ A, ∀ b ∈ A, a ≠ b → is_relatively_prime a b ∧ is_consecutive a b) →
  (∀ a ∈ B, ∀ b ∈ B, a ≠ b → ¬is_relatively_prime a b ∧ ¬is_multiple a b) →
  (∀ a ∈ C, ∀ b ∈ C, a ≠ b → is_composite a ∧ is_composite b ∧ is_relatively_prime a b) →
  (∀ a ∈ D, ∀ b ∈ D, a ≠ b → is_multiple a b ∧ ¬is_relatively_prime a b) →
  ∃! x, x ∈ cards ∧ x ∉ A ∧ x ∉ B ∧ x ∉ C ∧ x ∉ D ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_remaining_card_is_seven_l2830_283013


namespace NUMINAMATH_CALUDE_heptagon_fencing_cost_l2830_283083

/-- Calculates the total cost of fencing around a heptagon-shaped field -/
def total_fencing_cost (sides : List ℝ) (costs : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) sides costs)

/-- The theorem stating the total cost of fencing for the given heptagon -/
theorem heptagon_fencing_cost : 
  let sides : List ℝ := [14, 20, 35, 40, 15, 30, 25]
  let costs : List ℝ := [2.5, 3, 3.5, 4, 2.75, 3.25, 3.75]
  total_fencing_cost sides costs = 610 := by
  sorry

#eval total_fencing_cost [14, 20, 35, 40, 15, 30, 25] [2.5, 3, 3.5, 4, 2.75, 3.25, 3.75]

end NUMINAMATH_CALUDE_heptagon_fencing_cost_l2830_283083


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_l2830_283035

theorem cubic_equation_root_sum (p q : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + p * (Complex.I * Real.sqrt 2 + 2) + q = 0 → 
  p + q = 14 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_l2830_283035
