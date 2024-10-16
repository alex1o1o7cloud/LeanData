import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l640_64044

theorem equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 4 * Real.sqrt (9 + x) + 4 * Real.sqrt (9 - x) = 10 * Real.sqrt 3 ∧
  x = Real.sqrt 80.859375 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l640_64044


namespace NUMINAMATH_CALUDE_p_minus_q_empty_iff_a_nonneg_l640_64082

/-- The set P as defined in the problem -/
def P : Set ℝ :=
  {y | ∃ x, 1 - Real.sqrt 2 / 2 < x ∧ x < 3/2 ∧ y = -x^2 + 2*x - 1/2}

/-- The set Q as defined in the problem -/
def Q (a : ℝ) : Set ℝ :=
  {x | x^2 + (a-1)*x - a < 0}

/-- The main theorem stating the equivalence between P - Q being empty and a being in [0, +∞) -/
theorem p_minus_q_empty_iff_a_nonneg (a : ℝ) :
  P \ Q a = ∅ ↔ a ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_p_minus_q_empty_iff_a_nonneg_l640_64082


namespace NUMINAMATH_CALUDE_smallest_multiple_of_9_and_6_l640_64014

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_multiple_of_9_and_6 : 
  (∀ n : ℕ, n > 0 ∧ is_multiple 9 n ∧ is_multiple 6 n → n ≥ 18) ∧
  (18 > 0 ∧ is_multiple 9 18 ∧ is_multiple 6 18) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_9_and_6_l640_64014


namespace NUMINAMATH_CALUDE_four_bb_two_divisible_by_9_l640_64086

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (B : ℕ) : ℕ :=
  4 + B + B + 2

theorem four_bb_two_divisible_by_9 (B : ℕ) (h1 : B < 10) :
  is_divisible_by_9 (4000 + 100 * B + 10 * B + 2) ↔ B = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_four_bb_two_divisible_by_9_l640_64086


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l640_64034

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: For an arithmetic sequence, a_1 < a_3 iff the sequence is monotonically increasing -/
theorem arithmetic_sequence_increasing_iff_a1_lt_a3 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 < a 3 ↔ MonotonicallyIncreasing a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_a1_lt_a3_l640_64034


namespace NUMINAMATH_CALUDE_function_value_theorem_l640_64076

/-- Given a function f(x) = ax³ - bx + |x| - 1 where f(-8) = 3, prove that f(8) = 11 -/
theorem function_value_theorem (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 - b * x + |x| - 1
  (f (-8) = 3) → (f 8 = 11) := by
sorry

end NUMINAMATH_CALUDE_function_value_theorem_l640_64076


namespace NUMINAMATH_CALUDE_product_xyz_equals_42_l640_64011

theorem product_xyz_equals_42 (x y z : ℝ) 
  (h1 : y = x + 1) 
  (h2 : x + y = 2 * z) 
  (h3 : x = 3) : 
  x * y * z = 42 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_42_l640_64011


namespace NUMINAMATH_CALUDE_sector_area_l640_64091

/-- Given a circular sector with central angle 2π/3 and chord length 2√3, 
    its area is 4π/3 -/
theorem sector_area (θ : Real) (chord_length : Real) (area : Real) : 
  θ = 2 * Real.pi / 3 →
  chord_length = 2 * Real.sqrt 3 →
  area = 4 * Real.pi / 3 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l640_64091


namespace NUMINAMATH_CALUDE_equation_solution_l640_64036

theorem equation_solution : ∃! x : ℝ, (2 / 3) * x - 2 = 4 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l640_64036


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l640_64079

def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l640_64079


namespace NUMINAMATH_CALUDE_first_project_questions_l640_64047

/-- Calculates the number of questions for the first project given the total questions per day, 
    number of days, and questions for the second project. -/
def questions_for_first_project (questions_per_day : ℕ) (days : ℕ) (questions_second_project : ℕ) : ℕ :=
  questions_per_day * days - questions_second_project

/-- Proves that given the specified conditions, the number of questions for the first project is 518. -/
theorem first_project_questions : 
  questions_for_first_project 142 7 476 = 518 := by
sorry

end NUMINAMATH_CALUDE_first_project_questions_l640_64047


namespace NUMINAMATH_CALUDE_matrix_sum_squares_invertible_l640_64098

open Matrix

variable {n : ℕ}

/-- Given real n×n matrices M and N satisfying the conditions, M² + N² is invertible iff M and N are invertible -/
theorem matrix_sum_squares_invertible (M N : Matrix (Fin n) (Fin n) ℝ)
  (h_neq : M ≠ N)
  (h_cube : M^3 = N^3)
  (h_comm : M^2 * N = N^2 * M) :
  IsUnit (M^2 + N^2) ↔ IsUnit M ∧ IsUnit N := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_squares_invertible_l640_64098


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_sixth_term_l640_64077

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop := sorry

/-- Sum of the first n terms of a sequence -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := sorry

theorem arithmetic_geometric_sum_sixth_term 
  (a : ℕ → ℝ) 
  (h_ag : arithmetic_geometric_sequence a)
  (h_s2 : S a 2 = 1)
  (h_s4 : S a 4 = 3) : 
  S a 6 = 7 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_sixth_term_l640_64077


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_46_l640_64069

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0 ∨ m = 1)

theorem no_primes_divisible_by_46 :
  ∀ p : ℕ, is_prime p → ¬(p % 46 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_46_l640_64069


namespace NUMINAMATH_CALUDE_three_number_sum_l640_64007

theorem three_number_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 8)
  (h4 : (a + b + c) / 3 = c - 18)
  (h5 : b = 12) :
  a + b + c = 66 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l640_64007


namespace NUMINAMATH_CALUDE_parabola_coefficient_l640_64009

/-- Given a parabola y = ax^2 + bx + c with vertex (q/2, q/2) and y-intercept (0, -2q),
    where q ≠ 0, prove that b = 10 -/
theorem parabola_coefficient (a b c q : ℝ) (h_q : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (q/2, q/2) = (-(b / (2 * a)), a * (-(b / (2 * a)))^2 + b * (-(b / (2 * a))) + c) →
  c = -2 * q →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l640_64009


namespace NUMINAMATH_CALUDE_geom_seq_sum_property_l640_64061

/-- Represents a geometric sequence and its properties -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)
  geom_seq : ∀ n, a (n + 1) = q * a n

/-- 
Given a geometric sequence with S_4 = 1 and S_12 = 13,
prove that a_13 + a_14 + a_15 + a_16 = 27
-/
theorem geom_seq_sum_property (g : GeometricSequence) 
  (h1 : g.S 4 = 1) (h2 : g.S 12 = 13) :
  g.a 13 + g.a 14 + g.a 15 + g.a 16 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geom_seq_sum_property_l640_64061


namespace NUMINAMATH_CALUDE_room_length_l640_64033

/-- The length of a room satisfying given conditions -/
theorem room_length : ∃ (L : ℝ), 
  (L > 0) ∧ 
  (9 * (2 * 12 * (L + 15) - (6 * 3 + 3 * 4 * 3)) = 8154) → 
  L = 25 := by
  sorry

end NUMINAMATH_CALUDE_room_length_l640_64033


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l640_64053

-- Define the side length of the inner square
def inner_square_side : ℝ := 1

-- Define the shorter side of the rectangle
def rectangle_short_side : ℝ := inner_square_side

-- Define the longer side of the rectangle
def rectangle_long_side : ℝ := 2 * inner_square_side

-- Define the side length of the outer square
def outer_square_side : ℝ := inner_square_side + 2 * rectangle_short_side

-- State the theorem
theorem rectangle_ratio_is_two :
  (outer_square_side ^ 2 = 9 * inner_square_side ^ 2) →
  (rectangle_long_side / rectangle_short_side = 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l640_64053


namespace NUMINAMATH_CALUDE_triangle_side_length_l640_64020

/-- Given an acute triangle ABC with sides a, b, and c, 
    if a = 4, b = 3, and the area is 3√3, then c = √13 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  a = 4 → 
  b = 3 → 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 →
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  c = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l640_64020


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l640_64078

/-- An arithmetic sequence with common difference 2 -/
def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Condition that a_1, a_3, and a_4 form a geometric sequence -/
def geometricSubsequence (a : ℕ → ℤ) : Prop :=
  (a 3) ^ 2 = a 1 * a 4

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) 
  (h_arith : arithmeticSequence a) 
  (h_geom : geometricSubsequence a) : 
  a 2 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l640_64078


namespace NUMINAMATH_CALUDE_anayet_speed_l640_64087

/-- Calculates Anayet's speed given the total distance, Amoli's speed and driving time,
    Anayet's driving time, and the remaining distance. -/
theorem anayet_speed
  (total_distance : ℝ)
  (amoli_speed : ℝ)
  (amoli_time : ℝ)
  (anayet_time : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_time = 2)
  (h5 : remaining_distance = 121) :
  (total_distance - (amoli_speed * amoli_time) - remaining_distance) / anayet_time = 61 :=
by sorry

end NUMINAMATH_CALUDE_anayet_speed_l640_64087


namespace NUMINAMATH_CALUDE_ellipse_left_vertex_l640_64026

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
def ellipse_conditions (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ b = 4 ∧ (3 : ℝ)^2 = a^2 - b^2

-- Theorem statement
theorem ellipse_left_vertex (a b : ℝ) :
  ellipse_conditions a b →
  ∃ (x y : ℝ), ellipse a b x y ∧ x = -5 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_left_vertex_l640_64026


namespace NUMINAMATH_CALUDE_cow_count_is_83_l640_64041

/-- Calculates the final number of cows given the initial count and changes in the herd -/
def final_cow_count (initial : ℕ) (died : ℕ) (sold : ℕ) (increased : ℕ) (bought : ℕ) (gifted : ℕ) : ℕ :=
  initial - died - sold + increased + bought + gifted

/-- Theorem stating that given the specific changes in the herd, the final count is 83 -/
theorem cow_count_is_83 :
  final_cow_count 39 25 6 24 43 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_83_l640_64041


namespace NUMINAMATH_CALUDE_points_on_line_y_relation_l640_64012

/-- Given two points A(1, y₁) and B(-1, y₂) on the line y = -3x + 2, 
    prove that y₁ < y₂ -/
theorem points_on_line_y_relation (y₁ y₂ : ℝ) : 
  (1 : ℝ) > (-1 : ℝ) → -- x₁ > x₂
  y₁ = -3 * (1 : ℝ) + 2 → -- Point A satisfies the line equation
  y₂ = -3 * (-1 : ℝ) + 2 → -- Point B satisfies the line equation
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_points_on_line_y_relation_l640_64012


namespace NUMINAMATH_CALUDE_function_inequality_l640_64031

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem function_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f a x₀ ≤ 0) →
  (a ≥ (Real.exp 2 + 1) / (Real.exp 1 - 1) ∨ a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l640_64031


namespace NUMINAMATH_CALUDE_proposition_p_and_not_q_l640_64037

theorem proposition_p_and_not_q :
  (∃ x : ℝ, x - 2 > Real.log x) ∧ ¬(∀ x : ℝ, Real.sin x < x) := by sorry

end NUMINAMATH_CALUDE_proposition_p_and_not_q_l640_64037


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l640_64048

theorem ceiling_floor_product (y : ℝ) :
  y < 0 →
  ⌈y⌉ * ⌊y⌋ = 72 →
  y ∈ Set.Icc (-9 : ℝ) (-8 : ℝ) ∧ y ≠ -8 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l640_64048


namespace NUMINAMATH_CALUDE_parabola_shift_down_2_l640_64035

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Shifts a parabola vertically -/
def shift_parabola (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

theorem parabola_shift_down_2 :
  let original := Parabola.mk 2 4
  let shifted := shift_parabola original (-2)
  shifted = Parabola.mk 2 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_down_2_l640_64035


namespace NUMINAMATH_CALUDE_product_cost_change_l640_64002

theorem product_cost_change (initial_cost : ℝ) (h : initial_cost > 0) : 
  initial_cost * (1 + 0.2)^2 * (1 - 0.2)^2 < initial_cost := by
  sorry

end NUMINAMATH_CALUDE_product_cost_change_l640_64002


namespace NUMINAMATH_CALUDE_equality_multiplication_l640_64095

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_l640_64095


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l640_64016

theorem gcd_lcm_product_24_36 : 
  (Nat.gcd 24 36) * (Nat.lcm 24 36) = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l640_64016


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l640_64062

/-- Fred's initial amount of money in dollars -/
def fred_initial : ℕ := 19

/-- Fred's final amount of money in dollars -/
def fred_final : ℕ := 40

/-- Fred's earnings over the weekend in dollars -/
def fred_earnings : ℕ := fred_final - fred_initial

theorem fred_weekend_earnings : fred_earnings = 21 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l640_64062


namespace NUMINAMATH_CALUDE_special_numbers_l640_64015

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_four_digit n ∧ n % 11 = 0 ∧ digit_sum n = 11

theorem special_numbers :
  {n : ℕ | satisfies_conditions n} =
  {2090, 3080, 4070, 5060, 6050, 7040, 8030, 9020} := by sorry

end NUMINAMATH_CALUDE_special_numbers_l640_64015


namespace NUMINAMATH_CALUDE_catch_up_theorem_l640_64064

/-- The number of days after which the second student catches up with the first student -/
def catch_up_day : ℕ := 13

/-- The distance walked by the first student each day -/
def first_student_daily_distance : ℕ := 7

/-- The distance walked by the second student on the nth day -/
def second_student_daily_distance (n : ℕ) : ℕ := n

/-- The total distance walked by the first student after n days -/
def first_student_total_distance (n : ℕ) : ℕ :=
  n * first_student_daily_distance

/-- The total distance walked by the second student after n days -/
def second_student_total_distance (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem catch_up_theorem :
  first_student_total_distance catch_up_day = second_student_total_distance catch_up_day :=
by sorry

end NUMINAMATH_CALUDE_catch_up_theorem_l640_64064


namespace NUMINAMATH_CALUDE_not_perp_planes_implies_no_perp_line_l640_64083

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the "line within plane" relation
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem not_perp_planes_implies_no_perp_line (α β : Plane) :
  ¬(∀ (α β : Plane), ¬(perp_planes α β) → ∀ (l : Line), line_in_plane l α → ¬(perp_line_plane l β)) :=
sorry

end NUMINAMATH_CALUDE_not_perp_planes_implies_no_perp_line_l640_64083


namespace NUMINAMATH_CALUDE_probability_club_then_heart_l640_64055

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of clubs in a standard deck
def num_clubs : ℕ := 13

-- Define the number of hearts in a standard deck
def num_hearts : ℕ := 13

-- Theorem statement
theorem probability_club_then_heart :
  (num_clubs : ℚ) / total_cards * num_hearts / (total_cards - 1) = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_probability_club_then_heart_l640_64055


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l640_64094

/-- Represents a repeating decimal with a whole number part and a repeating part -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeating_decimal_to_rational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (99 : ℚ)

/-- The theorem stating that the division of two specific repeating decimals equals 3/10 -/
theorem repeating_decimal_division :
  let d1 := RepeatingDecimal.mk 0 81
  let d2 := RepeatingDecimal.mk 2 72
  (repeating_decimal_to_rational d1) / (repeating_decimal_to_rational d2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l640_64094


namespace NUMINAMATH_CALUDE_value_of_expression_l640_64090

theorem value_of_expression (x : ℝ) (h : x = 2) : 3^x - x^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l640_64090


namespace NUMINAMATH_CALUDE_distance_between_vertices_l640_64005

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 1) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -2)

-- Theorem statement
theorem distance_between_vertices :
  let (x1, y1) := vertex1
  let (x2, y2) := vertex2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l640_64005


namespace NUMINAMATH_CALUDE_riverbend_prep_distance_l640_64058

/-- Represents a relay race team -/
structure RelayTeam where
  name : String
  members : Nat
  raceLength : Nat

/-- Calculates the total distance covered by a relay team -/
def totalDistance (team : RelayTeam) : Nat :=
  team.members * team.raceLength

/-- Theorem stating that the total distance covered by Riverbend Prep is 1500 meters -/
theorem riverbend_prep_distance :
  let riverbendPrep : RelayTeam := ⟨"Riverbend Prep", 6, 250⟩
  totalDistance riverbendPrep = 1500 := by sorry

end NUMINAMATH_CALUDE_riverbend_prep_distance_l640_64058


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l640_64021

theorem quadratic_root_condition (a : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*a*x + a + 1 = 0 ∧ x^2 + 2*a*y + a + 1 = 0 ∧ x > 2 ∧ y < 2) → 
  a < -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l640_64021


namespace NUMINAMATH_CALUDE_inequality_equivalence_l640_64045

theorem inequality_equivalence (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -4/3) :
  (x + 3) / (x - 1) > (4 * x + 5) / (3 * x + 4) ↔ 7 - Real.sqrt 66 < x ∧ x < 7 + Real.sqrt 66 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l640_64045


namespace NUMINAMATH_CALUDE_age_difference_is_four_l640_64000

/-- The age difference between Angelina and Justin -/
def ageDifference (angelinaFutureAge : ℕ) (justinCurrentAge : ℕ) : ℕ :=
  (angelinaFutureAge - 5) - justinCurrentAge

/-- Theorem stating that the age difference between Angelina and Justin is 4 years -/
theorem age_difference_is_four :
  ageDifference 40 31 = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_four_l640_64000


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l640_64065

-- Problem 1
theorem problem_1 : Real.sqrt 27 - 6 * Real.sqrt (1/3) + Real.sqrt ((-2)^2) = Real.sqrt 3 + 2 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 2 + Real.sqrt 3) (hy : y = 2 - Real.sqrt 3) :
  x^2 * y + x * y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l640_64065


namespace NUMINAMATH_CALUDE_house_transaction_loss_l640_64057

def initial_value : ℝ := 12000
def loss_percentage : ℝ := 0.15
def gain_percentage : ℝ := 0.20

theorem house_transaction_loss :
  let first_sale := initial_value * (1 - loss_percentage)
  let second_sale := first_sale * (1 + gain_percentage)
  second_sale - initial_value = 240 := by sorry

end NUMINAMATH_CALUDE_house_transaction_loss_l640_64057


namespace NUMINAMATH_CALUDE_line_parabola_circle_intersection_l640_64043

/-- A line intersecting a parabola and a circle with specific conditions -/
theorem line_parabola_circle_intersection
  (k m : ℝ)
  (l : Set (ℝ × ℝ))
  (A B C D : ℝ × ℝ)
  (h_line : l = {(x, y) | y = k * x + m})
  (h_parabola : A ∈ l ∧ B ∈ l ∧ A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2)
  (h_midpoint : (A.1 + B.1) / 2 = 1)
  (h_circle : C ∈ l ∧ D ∈ l ∧ C.1^2 + C.2^2 = 12 ∧ D.1^2 + D.2^2 = 12)
  (h_equal_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2) :
  k = 1 ∧ m = 2 := by sorry

end NUMINAMATH_CALUDE_line_parabola_circle_intersection_l640_64043


namespace NUMINAMATH_CALUDE_semicircle_point_height_l640_64046

theorem semicircle_point_height :
  let P : ℝ × ℝ := (-4, 0)
  let Q : ℝ × ℝ := (16, 0)
  let R : ℝ → ℝ × ℝ := fun t => (0, t)
  ∀ t : ℝ, t > 0 →
    (∃ C : ℝ × ℝ, 
      (C.1 = (P.1 + Q.1) / 2 ∧ C.2 = (P.2 + Q.2) / 2) ∧
      (Real.sqrt ((R t).1 - C.1)^2 + ((R t).2 - C.2)^2) = (Q.1 - P.1) / 2) →
    t = 8 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_point_height_l640_64046


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l640_64040

theorem binary_to_octal_conversion : 
  (1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 
  (1 * 8^2 + 1 * 8^1 + 5 * 8^0) :=
by sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l640_64040


namespace NUMINAMATH_CALUDE_correct_average_l640_64029

theorem correct_average (total_numbers : Nat) (initial_average : ℚ) 
  (correct_number1 correct_number2 correct_number3 : ℚ)
  (incorrect_number1 incorrect_number2 incorrect_number3 : ℚ) :
  total_numbers = 12 ∧
  initial_average = 22 ∧
  correct_number1 = 52 ∧ incorrect_number1 = 32 ∧
  correct_number2 = 47 ∧ incorrect_number2 = 27 ∧
  correct_number3 = 68 ∧ incorrect_number3 = 45 →
  (total_numbers : ℚ) * initial_average - 
    (incorrect_number1 + incorrect_number2 + incorrect_number3) + 
    (correct_number1 + correct_number2 + correct_number3) = 
  (total_numbers : ℚ) * (327 / 12) :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l640_64029


namespace NUMINAMATH_CALUDE_income_comparison_l640_64067

/-- Represents the problem of calculating incomes relative to Juan's base income -/
theorem income_comparison (J : ℝ) (J_pos : J > 0) : 
  let tim_base := 0.7 * J
  let mary_total := 1.12 * J * 1.1
  let lisa_base := 0.63 * J
  let lisa_total := lisa_base * 1.03
  let alan_base := lisa_base / 1.15
  let nina_base := 1.25 * J
  let nina_total := nina_base * 1.07
  (mary_total + lisa_total + nina_total) / J = 3.2184 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l640_64067


namespace NUMINAMATH_CALUDE_log_equation_implies_ratio_l640_64024

theorem log_equation_implies_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x - y) + Real.log (x + 2*y) = Real.log 2 + Real.log x + Real.log y) : 
  x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_ratio_l640_64024


namespace NUMINAMATH_CALUDE_factor_expression_l640_64052

theorem factor_expression (x : ℝ) : 63 * x^2 + 28 * x = 7 * x * (9 * x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l640_64052


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_main_theorem_l640_64097

theorem repeating_decimal_sum : ∀ (a b c : ℕ), a < 10 → b < 10 → c < 10 →
  (a : ℚ) / 9 + (b : ℚ) / 9 - (c : ℚ) / 9 = (a + b - c : ℚ) / 9 :=
by sorry

theorem main_theorem : (8 : ℚ) / 9 + (2 : ℚ) / 9 - (6 : ℚ) / 9 = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_main_theorem_l640_64097


namespace NUMINAMATH_CALUDE_regression_unit_increase_survey_regression_unit_increase_l640_64051

/-- Linear regression equation parameters -/
structure RegressionParams where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted y value for a given x -/
def predict (params : RegressionParams) (x : ℝ) : ℝ :=
  params.slope * x + params.intercept

/-- Theorem: The difference in predicted y when x increases by 1 is equal to the slope -/
theorem regression_unit_increase (params : RegressionParams) :
  ∀ x : ℝ, predict params (x + 1) - predict params x = params.slope := by
  sorry

/-- The specific regression equation from the problem -/
def survey_regression : RegressionParams :=
  { slope := 0.254, intercept := 0.321 }

/-- Theorem: For the given survey regression, the difference in predicted y
    when x increases by 1 is equal to 0.254 -/
theorem survey_regression_unit_increase :
  ∀ x : ℝ, predict survey_regression (x + 1) - predict survey_regression x = 0.254 := by
  sorry

end NUMINAMATH_CALUDE_regression_unit_increase_survey_regression_unit_increase_l640_64051


namespace NUMINAMATH_CALUDE_min_value_of_z_l640_64073

theorem min_value_of_z (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4^x + 2^y) : 
  z ≥ 2 * Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), 2 * x₀ + y₀ = 1 ∧ 4^x₀ + 2^y₀ = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l640_64073


namespace NUMINAMATH_CALUDE_adults_in_group_l640_64084

theorem adults_in_group (children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) (adults : ℕ) : 
  children = 5 → 
  meal_cost = 3 → 
  total_bill = 21 → 
  adults * meal_cost + children * meal_cost = total_bill → 
  adults = 2 := by
sorry

end NUMINAMATH_CALUDE_adults_in_group_l640_64084


namespace NUMINAMATH_CALUDE_function_inequality_condition_l640_64001

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^2 + x + 1) →
  a > 0 →
  b > 0 →
  (∀ x, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l640_64001


namespace NUMINAMATH_CALUDE_triangle_segment_sum_l640_64092

/-- Given a triangle ABC with vertices A(0,0), B(7,0), and C(3,4), and a line
    passing through (6-2√2, 3-√2) intersecting AC at P and BC at Q,
    if the area of triangle PQC is 14/3, then |CP| + |CQ| = 63. -/
theorem triangle_segment_sum (P Q : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (7, 0)
  let C : ℝ × ℝ := (3, 4)
  let line_point : ℝ × ℝ := (6 - 2 * Real.sqrt 2, 3 - Real.sqrt 2)
  (∃ (t₁ t₂ : ℝ), 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧
    P = (t₁ * C.1 + (1 - t₁) * A.1, t₁ * C.2 + (1 - t₁) * A.2) ∧
    Q = (t₂ * C.1 + (1 - t₂) * B.1, t₂ * C.2 + (1 - t₂) * B.2) ∧
    ∃ (s : ℝ), P = (line_point.1 + s * (Q.1 - line_point.1), 
                    line_point.2 + s * (Q.2 - line_point.2))) →
  (abs (P.1 * Q.2 - P.2 * Q.1 + Q.1 * C.2 - Q.2 * C.1 + C.1 * P.2 - C.2 * P.1) / 2 = 14/3) →
  Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) + Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) = 63 :=
by sorry

end NUMINAMATH_CALUDE_triangle_segment_sum_l640_64092


namespace NUMINAMATH_CALUDE_b_inequalities_l640_64042

theorem b_inequalities (a : ℝ) (h : a ∈ Set.Icc 0 1) :
  let b := a^3 + 1 / (1 + a)
  (b ≥ 1 - a + a^2) ∧ (3/4 < b ∧ b ≤ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_b_inequalities_l640_64042


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l640_64032

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x^3 - 24*x^2 + 151*x - 650 = (x - p)*(x - q)*(x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 24*s^2 + 151*s - 650) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 251 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l640_64032


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l640_64096

theorem trigonometric_equation_solution (x : ℝ) : 
  (abs (Real.cos x) - Real.cos (3 * x)) / (Real.cos x * Real.sin (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 6 + 2 * k * π ∨ x = 5 * π / 6 + 2 * k * π ∨ x = 4 * π / 3 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l640_64096


namespace NUMINAMATH_CALUDE_exponential_function_property_l640_64017

theorem exponential_function_property (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → (fun x => a^x) (x + y) = (fun x => a^x) x * (fun x => a^x) y :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l640_64017


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l640_64070

-- Define the two circles
def C₁ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 169
def C₂ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 9

-- Define the moving circle
def MovingCircle (x y r : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), C₁ x₀ y₀ ∧ C₂ x₀ y₀ ∧
  ((x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  ((x - 4)^2 + y^2 = (13 - r)^2) ∧
  ((x + 4)^2 + y^2 = (r + 3)^2)

-- Theorem statement
theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), (∃ (r : ℝ), MovingCircle x y r) →
  (x^2 / 64 + y^2 / 48 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l640_64070


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l640_64004

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equals_set : B ∪ (U \ A) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l640_64004


namespace NUMINAMATH_CALUDE_homework_problem_count_l640_64038

/-- The number of sub tasks per homework problem -/
def sub_tasks_per_problem : ℕ := 5

/-- The total number of sub tasks to solve -/
def total_sub_tasks : ℕ := 200

/-- The total number of homework problems -/
def total_problems : ℕ := total_sub_tasks / sub_tasks_per_problem

theorem homework_problem_count :
  total_problems = 40 :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_count_l640_64038


namespace NUMINAMATH_CALUDE_existence_of_bounded_irreducible_factorization_l640_64071

def is_irreducible (S : Set ℕ) (x : ℕ) : Prop :=
  x ∈ S ∧ ∀ y z : ℕ, y ∈ S → z ∈ S → x = y * z → (y = 1 ∨ z = 1)

theorem existence_of_bounded_irreducible_factorization 
  (a b : ℕ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_gcd : ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ Nat.gcd a b ∧ q ∣ Nat.gcd a b) :
  ∃ t : ℕ, ∀ x ∈ {n : ℕ | n > 0 ∧ n % b = a % b}, 
    ∃ (factors : List ℕ), 
      (∀ f ∈ factors, is_irreducible {n : ℕ | n > 0 ∧ n % b = a % b} f) ∧
      (factors.prod = x) ∧
      (factors.length ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_bounded_irreducible_factorization_l640_64071


namespace NUMINAMATH_CALUDE_max_intersections_count_l640_64075

/-- The number of points on the x-axis segment -/
def n : ℕ := 15

/-- The number of points on the y-axis segment -/
def m : ℕ := 10

/-- The maximum number of intersection points -/
def max_intersections : ℕ := n.choose 2 * m.choose 2

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersections_count :
  max_intersections = 4725 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_count_l640_64075


namespace NUMINAMATH_CALUDE_second_concert_attendance_l640_64049

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_attendees : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : additional_attendees = 119) :
  first_concert + additional_attendees = 66018 := by
sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l640_64049


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l640_64060

/-- A geometric sequence with first term 1 and the sum of the third and fifth terms equal to 6 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  (∃ q : ℝ, ∀ n : ℕ, a n = q ^ (n - 1)) ∧
  a 3 + a 5 = 6

/-- The sum of the fifth and seventh terms of the geometric sequence is 12 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 5 + a 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l640_64060


namespace NUMINAMATH_CALUDE_arccos_sin_three_l640_64010

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_three_l640_64010


namespace NUMINAMATH_CALUDE_carters_baseball_cards_l640_64006

/-- Given that Marcus has 350 baseball cards and 95 more cards than Carter,
    prove that Carter has 255 baseball cards. -/
theorem carters_baseball_cards : 
  ∀ (marcus_cards carter_cards : ℕ), 
    marcus_cards = 350 → 
    marcus_cards = carter_cards + 95 →
    carter_cards = 255 := by
  sorry

end NUMINAMATH_CALUDE_carters_baseball_cards_l640_64006


namespace NUMINAMATH_CALUDE_coin_problem_l640_64085

/-- Represents the number of different values that can be produced with given coins -/
def different_values (five_cent_coins ten_cent_coins : ℕ) : ℕ :=
  29 - five_cent_coins

theorem coin_problem (total_coins : ℕ) (distinct_values : ℕ) 
  (h1 : total_coins = 15)
  (h2 : distinct_values = 26) :
  ∃ (five_cent_coins ten_cent_coins : ℕ),
    five_cent_coins + ten_cent_coins = total_coins ∧
    different_values five_cent_coins ten_cent_coins = distinct_values ∧
    ten_cent_coins = 12 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l640_64085


namespace NUMINAMATH_CALUDE_money_difference_l640_64027

/-- Calculates the difference between final and initial amounts given monetary transactions --/
theorem money_difference (initial chores birthday neighbor candy lost : ℕ) : 
  initial = 2 →
  chores = 5 →
  birthday = 10 →
  neighbor = 7 →
  candy = 3 →
  lost = 2 →
  (initial + chores + birthday + neighbor - candy - lost) - initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l640_64027


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l640_64008

theorem geometric_mean_of_4_and_16 (x : ℝ) :
  x ^ 2 = 4 * 16 → x = 8 ∨ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l640_64008


namespace NUMINAMATH_CALUDE_jacket_price_proof_l640_64081

/-- The original price of the jacket -/
def original_price : ℝ := 250

/-- The regular discount percentage -/
def regular_discount : ℝ := 0.4

/-- The weekend additional discount percentage -/
def weekend_discount : ℝ := 0.1

/-- The final price after both discounts -/
def final_price : ℝ := original_price * (1 - regular_discount) * (1 - weekend_discount)

theorem jacket_price_proof : final_price = 135 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_proof_l640_64081


namespace NUMINAMATH_CALUDE_inequality_proof_l640_64056

theorem inequality_proof (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l640_64056


namespace NUMINAMATH_CALUDE_find_m_l640_64013

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

theorem find_m : 
  ∃ (m : ℝ), (U \ A m = {1, 2}) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l640_64013


namespace NUMINAMATH_CALUDE_stating_assignment_ways_l640_64022

/-- Represents the number of student volunteers -/
def num_volunteers : ℕ := 5

/-- Represents the number of posts -/
def num_posts : ℕ := 4

/-- Represents the number of ways A and B can be assigned to posts -/
def ways_to_assign_A_and_B : ℕ := num_posts * (num_posts - 1)

/-- 
Theorem stating that the number of ways for A and B to each independently 
take charge of one post, while ensuring each post is staffed by at least 
one volunteer, is equal to 72.
-/
theorem assignment_ways : 
  ∃ (f : ℕ → ℕ → ℕ), 
    f ways_to_assign_A_and_B (num_volunteers - 2) = 72 ∧ 
    (∀ x y, f x y ≤ x * (y^(num_posts - 2))) := by
  sorry

end NUMINAMATH_CALUDE_stating_assignment_ways_l640_64022


namespace NUMINAMATH_CALUDE_f_1987_equals_1984_l640_64068

/-- 
Represents the function f(n) where f(n) = m if the 10^n th digit in the sequence
forms part of an m-digit number.
-/
def f (n : ℕ) : ℕ := sorry

/-- 
Represents the cumulative count of digits up to and including m-digit numbers
in the sequence of concatenated positive integers.
-/
def digitCount (m : ℕ) : ℕ := sorry

theorem f_1987_equals_1984 : f 1987 = 1984 := by sorry

end NUMINAMATH_CALUDE_f_1987_equals_1984_l640_64068


namespace NUMINAMATH_CALUDE_cubic_polynomial_three_distinct_roots_l640_64063

/-- A cubic polynomial with specific properties has three distinct real roots -/
theorem cubic_polynomial_three_distinct_roots 
  (f : ℝ → ℝ) (a b c : ℝ) 
  (h_cubic : ∀ x, f x = x^3 + a*x^2 + b*x + c) 
  (h_b_neg : b < 0) 
  (h_ab_9c : a * b = 9 * c) : 
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_three_distinct_roots_l640_64063


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l640_64080

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a₁ + a₂ = 5 and a₃ + a₄ = 7,
    prove that a₅ + a₆ = 9 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 = 5)
  (h_sum2 : a 3 + a 4 = 7) :
  a 5 + a 6 = 9 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l640_64080


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l640_64018

theorem smallest_n_divisibility : ∃ (n : ℕ), n = 4058209 ∧
  (∃ (m : ℤ), n + 2015 = 2016 * m) ∧
  (∃ (k : ℤ), n + 2016 = 2015 * k) ∧
  (∀ (n' : ℕ), n' < n →
    (∃ (m : ℤ), n' + 2015 = 2016 * m) →
    (∃ (k : ℤ), n' + 2016 = 2015 * k) → False) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l640_64018


namespace NUMINAMATH_CALUDE_rectangle_area_stage_8_l640_64028

/-- The area of a rectangle formed by adding n squares of side length s --/
def rectangleArea (n : ℕ) (s : ℝ) : ℝ := n * (s * s)

/-- Theorem: The area of a rectangle formed by adding 8 squares, each 4 inches by 4 inches, is 128 square inches --/
theorem rectangle_area_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_8_l640_64028


namespace NUMINAMATH_CALUDE_range_of_m_when_not_two_distinct_positive_roots_l640_64019

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + m*x + 1 = 0

-- Define the condition for two distinct positive real roots
def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- The theorem to prove
theorem range_of_m_when_not_two_distinct_positive_roots :
  {m : ℝ | ¬(has_two_distinct_positive_roots m)} = Set.Ici (-2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_when_not_two_distinct_positive_roots_l640_64019


namespace NUMINAMATH_CALUDE_normal_dist_symmetry_normal_dist_property_l640_64023

/-- A normal distribution with mean 0 and standard deviation σ -/
def normal_dist (σ : ℝ) : Type := ℝ

/-- Probability measure for the normal distribution -/
def P (σ : ℝ) : Set ℝ → ℝ := sorry

theorem normal_dist_symmetry 
  (σ : ℝ) (a : ℝ) : 
  P σ {x | -a ≤ x ∧ x ≤ 0} = P σ {x | 0 ≤ x ∧ x ≤ a} :=
sorry

theorem normal_dist_property 
  (σ : ℝ) (h : P σ {x | -2 ≤ x ∧ x ≤ 0} = 0.3) : 
  P σ {x | x > 2} = 0.2 :=
sorry

end NUMINAMATH_CALUDE_normal_dist_symmetry_normal_dist_property_l640_64023


namespace NUMINAMATH_CALUDE_garden_perimeter_l640_64099

/-- The perimeter of a rectangular garden with width 8 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 64 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let playground_area : ℝ := playground_length * playground_width
  let garden_width : ℝ := 8
  let garden_length : ℝ := playground_area / garden_width
  let garden_perimeter : ℝ := 2 * (garden_length + garden_width)
  garden_perimeter = 64 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l640_64099


namespace NUMINAMATH_CALUDE_ball_returns_after_12_throws_l640_64025

/-- Represents the number of girls in the circle -/
def n : ℕ := 15

/-- Represents the number of girls skipped in each throw -/
def skip : ℕ := 4

/-- The function that determines the next girl to receive the ball -/
def next (x : ℕ) : ℕ := (x + skip + 1) % n + 1

/-- Represents the sequence of girls receiving the ball -/
def ball_sequence (k : ℕ) : ℕ := 
  Nat.iterate next 1 k

theorem ball_returns_after_12_throws : 
  ball_sequence 12 = 1 := by sorry

end NUMINAMATH_CALUDE_ball_returns_after_12_throws_l640_64025


namespace NUMINAMATH_CALUDE_class_item_distribution_l640_64066

/-- Calculates the total number of items distributed in a class --/
def calculate_total_items (num_children : ℕ) 
                          (initial_pencils : ℕ) 
                          (initial_erasers : ℕ) 
                          (initial_crayons : ℕ) 
                          (extra_pencils : ℕ) 
                          (extra_crayons : ℕ) 
                          (extra_erasers : ℕ) 
                          (num_children_extra_pencils_crayons : ℕ) : ℕ × ℕ × ℕ :=
  let total_pencils := num_children * initial_pencils + num_children_extra_pencils_crayons * extra_pencils
  let total_erasers := num_children * initial_erasers + (num_children - num_children_extra_pencils_crayons) * extra_erasers
  let total_crayons := num_children * initial_crayons + num_children_extra_pencils_crayons * extra_crayons
  (total_pencils, total_erasers, total_crayons)

theorem class_item_distribution :
  let num_children : ℕ := 18
  let initial_pencils : ℕ := 6
  let initial_erasers : ℕ := 3
  let initial_crayons : ℕ := 12
  let extra_pencils : ℕ := 5
  let extra_crayons : ℕ := 8
  let extra_erasers : ℕ := 2
  let num_children_extra_pencils_crayons : ℕ := 10
  
  calculate_total_items num_children initial_pencils initial_erasers initial_crayons
                        extra_pencils extra_crayons extra_erasers
                        num_children_extra_pencils_crayons = (158, 70, 296) := by
  sorry

end NUMINAMATH_CALUDE_class_item_distribution_l640_64066


namespace NUMINAMATH_CALUDE_remaining_money_calculation_l640_64054

/-- Calculates the remaining money after expenses given a salary and expense ratios -/
def remaining_money (salary : ℚ) (food_ratio : ℚ) (rent_ratio : ℚ) (clothes_ratio : ℚ) : ℚ :=
  salary * (1 - (food_ratio + rent_ratio + clothes_ratio))

/-- Theorem stating that given the specific salary and expense ratios, the remaining money is 17000 -/
theorem remaining_money_calculation :
  remaining_money 170000 (1/5) (1/10) (3/5) = 17000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_calculation_l640_64054


namespace NUMINAMATH_CALUDE_min_value_of_expression_l640_64093

/-- The set of available numbers -/
def S : Finset Int := {-9, -6, -4, -1, 3, 5, 7, 12}

/-- The expression to be minimized -/
def f (p q r s t u v w : Int) : ℚ :=
  ((p + q + r + s : ℚ) ^ 2 + (t + u + v + w : ℚ) ^ 2 : ℚ)

/-- The theorem stating the minimum value of the expression -/
theorem min_value_of_expression :
  ∀ p q r s t u v w : Int,
    p ∈ S → q ∈ S → r ∈ S → s ∈ S → t ∈ S → u ∈ S → v ∈ S → w ∈ S →
    p ≠ q → p ≠ r → p ≠ s → p ≠ t → p ≠ u → p ≠ v → p ≠ w →
    q ≠ r → q ≠ s → q ≠ t → q ≠ u → q ≠ v → q ≠ w →
    r ≠ s → r ≠ t → r ≠ u → r ≠ v → r ≠ w →
    s ≠ t → s ≠ u → s ≠ v → s ≠ w →
    t ≠ u → t ≠ v → t ≠ w →
    u ≠ v → u ≠ w →
    v ≠ w →
    f p q r s t u v w ≥ 26.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l640_64093


namespace NUMINAMATH_CALUDE_ice_cream_difference_l640_64039

-- Define the number of scoops for Oli and Victoria
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := 2 * oli_scoops

-- Theorem statement
theorem ice_cream_difference : victoria_scoops - oli_scoops = 4 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_difference_l640_64039


namespace NUMINAMATH_CALUDE_peach_problem_l640_64059

theorem peach_problem (jake steven jill : ℕ) 
  (h1 : jake = steven - 6)
  (h2 : steven = jill + 18)
  (h3 : jake = 17) : 
  jill = 5 := by
sorry

end NUMINAMATH_CALUDE_peach_problem_l640_64059


namespace NUMINAMATH_CALUDE_sin_negative_330_degrees_l640_64003

theorem sin_negative_330_degrees : Real.sin ((-330 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_330_degrees_l640_64003


namespace NUMINAMATH_CALUDE_consecutive_zeros_in_power_of_five_l640_64030

theorem consecutive_zeros_in_power_of_five : ∃ n : ℕ, n < 10^6 ∧ 5^n % 10^20 < 10^14 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_zeros_in_power_of_five_l640_64030


namespace NUMINAMATH_CALUDE_fraction_of_number_minus_constant_l640_64072

theorem fraction_of_number_minus_constant (a b c d : ℕ) (h : a ≤ b) : 
  (a : ℚ) / b * c - d = 39 → a = 7 ∧ b = 8 ∧ c = 48 ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_number_minus_constant_l640_64072


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l640_64050

theorem contrapositive_equivalence (a b : ℝ) : 
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l640_64050


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l640_64088

theorem adult_ticket_cost (num_children num_adults : ℕ) (child_ticket_cost total_cost : ℚ) 
  (h1 : num_children = 6)
  (h2 : num_adults = 10)
  (h3 : child_ticket_cost = 10)
  (h4 : total_cost = 220)
  : (total_cost - num_children * child_ticket_cost) / num_adults = 16 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l640_64088


namespace NUMINAMATH_CALUDE_christopher_karen_difference_l640_64089

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of quarters Karen has -/
def karen_quarters : ℕ := 32

/-- The number of quarters Christopher has -/
def christopher_quarters : ℕ := 64

/-- The difference in money between Christopher and Karen -/
def money_difference : ℚ := (christopher_quarters - karen_quarters) * quarter_value

theorem christopher_karen_difference : money_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_christopher_karen_difference_l640_64089


namespace NUMINAMATH_CALUDE_john_weight_loss_days_l640_64074

/-- Calculates the number of days needed to lose a certain amount of weight given daily calorie intake, daily calorie burn, calories needed to lose one pound, and desired weight loss. -/
def days_to_lose_weight (calories_eaten : ℕ) (calories_burned : ℕ) (calories_per_pound : ℕ) (pounds_to_lose : ℕ) : ℕ :=
  let net_calories_burned := calories_burned - calories_eaten
  let total_calories_to_burn := calories_per_pound * pounds_to_lose
  total_calories_to_burn / net_calories_burned

/-- Theorem stating that it takes 80 days for John to lose 10 pounds given the specified conditions. -/
theorem john_weight_loss_days : 
  days_to_lose_weight 1800 2300 4000 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_john_weight_loss_days_l640_64074
