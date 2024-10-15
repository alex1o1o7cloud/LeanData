import Mathlib

namespace NUMINAMATH_CALUDE_no_real_solutions_l0_61

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 8*x - 12*y + 36 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l0_61


namespace NUMINAMATH_CALUDE_grape_sales_profit_l0_86

/-- Profit function for grape sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 510 * x - 7500

/-- Theorem stating the properties of the profit function -/
theorem grape_sales_profit :
  let w := profit_function
  (w 28 = 1040) ∧
  (∀ x, w x ≤ w (51/2)) ∧
  (w (51/2) = 1102.5) := by
  sorry

end NUMINAMATH_CALUDE_grape_sales_profit_l0_86


namespace NUMINAMATH_CALUDE_exam_question_distribution_l0_83

theorem exam_question_distribution (total_questions : ℕ) 
  (group_a_marks : ℕ → ℕ) (group_b_marks : ℕ → ℕ) (group_c_marks : ℕ → ℕ)
  (group_b_count : ℕ) :
  total_questions = 100 →
  group_b_count = 23 →
  (∀ n, group_a_marks n = n) →
  (∀ n, group_b_marks n = 2 * n) →
  (∀ n, group_c_marks n = 3 * n) →
  (∀ a b c, a + b + c = total_questions → a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1) →
  (∀ a b c, a + b + c = total_questions → 
    group_a_marks a ≥ (3 * (group_a_marks a + group_b_marks b + group_c_marks c)) / 5) →
  (∀ a b c, a + b + c = total_questions → 
    group_b_marks b ≤ (group_a_marks a + group_b_marks b + group_c_marks c) / 4) →
  ∃ a c, a + group_b_count + c = total_questions ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_question_distribution_l0_83


namespace NUMINAMATH_CALUDE_sequence_with_2018_distinct_elements_l0_82

theorem sequence_with_2018_distinct_elements :
  ∃ a : ℝ, ∃ (x : ℕ → ℝ), 
    (x 1 = a) ∧ 
    (∀ n : ℕ, x (n + 1) = (1 / 2) * (x n - 1 / x n)) ∧
    (∃ m : ℕ, m ≤ 2018 ∧ x m = 0) ∧
    (∀ i j : ℕ, i < j ∧ j ≤ 2018 → x i ≠ x j) ∧
    (∀ k : ℕ, k > 2018 → x k = 0) :=
by sorry

end NUMINAMATH_CALUDE_sequence_with_2018_distinct_elements_l0_82


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l0_14

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 24 → b = 10 → h^2 = a^2 + b^2 → h = 26 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l0_14


namespace NUMINAMATH_CALUDE_art_club_theorem_l0_77

/-- Represents the distribution of students in a school's clubs -/
structure SchoolClubs where
  total_students : ℕ
  music_students : ℕ
  recitation_offset : ℕ
  dance_offset : ℤ

/-- Calculates the number of students in the art club -/
def art_club_students (sc : SchoolClubs) : ℤ :=
  sc.total_students - sc.music_students - (sc.music_students / 2 + sc.recitation_offset) - 
  (sc.music_students + 2 * sc.recitation_offset + sc.dance_offset)

/-- Theorem stating the number of students in the art club -/
theorem art_club_theorem (sc : SchoolClubs) 
  (h1 : sc.total_students = 220)
  (h2 : sc.dance_offset = -40) :
  art_club_students sc = 260 - (5/2 : ℚ) * sc.music_students - 3 * sc.recitation_offset :=
by sorry

end NUMINAMATH_CALUDE_art_club_theorem_l0_77


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l0_39

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ ∃ y, y^2 + p*y + m = 0 ∧ x = 3*y) →
  n / p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l0_39


namespace NUMINAMATH_CALUDE_investment_proportional_to_profit_share_q_investment_l0_91

/-- Represents the investment and profit share of an investor -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Given two investors with their investments and profit shares, 
    proves that their investments are proportional to their profit shares -/
theorem investment_proportional_to_profit_share 
  (p q : Investor) 
  (h1 : p.investment = 500000) 
  (h2 : p.profitShare = 2) 
  (h3 : q.profitShare = 4) : 
  q.investment = 1000000 := by
sorry

/-- Main theorem that proves Q's investment given P's investment and their profit sharing ratio -/
theorem q_investment 
  (p q : Investor) 
  (h1 : p.investment = 500000) 
  (h2 : p.profitShare = 2) 
  (h3 : q.profitShare = 4) : 
  q.investment = 1000000 := by
sorry

end NUMINAMATH_CALUDE_investment_proportional_to_profit_share_q_investment_l0_91


namespace NUMINAMATH_CALUDE_cubelets_one_color_count_l0_63

/-- Represents a cube divided into cubelets -/
structure CubeletCube where
  size : Nat
  total_cubelets : Nat
  painted_faces : Fin 3 → Fin 6

/-- The number of cubelets painted with exactly one color -/
def cubelets_with_one_color (c : CubeletCube) : Nat :=
  6 * (c.size - 2) * (c.size - 2)

/-- Theorem: In a 6x6x6 cube painted as described, 96 cubelets are painted with exactly one color -/
theorem cubelets_one_color_count :
  ∀ c : CubeletCube, c.size = 6 → c.total_cubelets = 216 → cubelets_with_one_color c = 96 := by
  sorry

end NUMINAMATH_CALUDE_cubelets_one_color_count_l0_63


namespace NUMINAMATH_CALUDE_chris_jogging_time_l0_3

/-- Represents the time in minutes -/
def Time := ℝ

/-- Represents the distance in miles -/
def Distance := ℝ

/-- Chris's jogging rate in minutes per mile -/
def chris_rate : ℝ := sorry

/-- Alex's walking rate in minutes per mile -/
def alex_rate : ℝ := sorry

theorem chris_jogging_time 
  (h1 : chris_rate * 4 = 2 * alex_rate * 2)  -- Chris's 4-mile time is twice Alex's 2-mile time
  (h2 : alex_rate * 2 = 40)                  -- Alex's 2-mile time is 40 minutes
  : chris_rate * 6 = 120 :=                  -- Chris's 6-mile time is 120 minutes
sorry

end NUMINAMATH_CALUDE_chris_jogging_time_l0_3


namespace NUMINAMATH_CALUDE_sofia_card_theorem_l0_8

theorem sofia_card_theorem (y : Real) (h1 : 0 < y) (h2 : y < Real.pi / 2) 
  (h3 : Real.tan y > Real.sin y) (h4 : Real.tan y > Real.cos y) : y = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_sofia_card_theorem_l0_8


namespace NUMINAMATH_CALUDE_marbles_distribution_l0_89

theorem marbles_distribution (n : ℕ) (initial_marbles : ℕ) : 
  n = 12 ∧ initial_marbles = 50 → 
  (n * (n + 1)) / 2 - initial_marbles = 28 := by
sorry

end NUMINAMATH_CALUDE_marbles_distribution_l0_89


namespace NUMINAMATH_CALUDE_smallest_k_for_cosine_equation_l0_84

theorem smallest_k_for_cosine_equation :
  let f : ℕ → Prop := λ k => Real.cos (k^2 + 8^2 : ℝ)^2 = 1
  ∃ (k₁ k₂ : ℕ), k₁ < k₂ ∧ f k₁ ∧ f k₂ ∧ k₁ = 10 ∧ k₂ = 12 ∧
    ∀ (k : ℕ), 0 < k ∧ k < k₁ → ¬f k :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_cosine_equation_l0_84


namespace NUMINAMATH_CALUDE_deck_size_l0_12

theorem deck_size (toothpicks_per_card : ℕ) (unused_cards : ℕ) (boxes : ℕ) (toothpicks_per_box : ℕ) :
  toothpicks_per_card = 75 →
  unused_cards = 16 →
  boxes = 6 →
  toothpicks_per_box = 450 →
  boxes * toothpicks_per_box / toothpicks_per_card + unused_cards = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_deck_size_l0_12


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l0_2

/-- Two parabolas with given properties -/
structure ParabolaPair where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : y₁ = a * x₁^2 + b * x₁ + c  -- Vertex condition for N₁
  h₂ : y₂ = -a * x₂^2 + d * x₂ + e  -- Vertex condition for N₂
  h₃ : 21 = a * 12^2 + b * 12 + c  -- A(12, 21) lies on N₁
  h₄ : 3 = a * 28^2 + b * 28 + c  -- B(28, 3) lies on N₁
  h₅ : 21 = -a * 12^2 + d * 12 + e  -- A(12, 21) lies on N₂
  h₆ : 3 = -a * 28^2 + d * 28 + e  -- B(28, 3) lies on N₂

/-- The main theorem -/
theorem parabola_intersection_ratio (p : ParabolaPair) :
  (p.x₁ + p.x₂) / (p.y₁ + p.y₂) = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_ratio_l0_2


namespace NUMINAMATH_CALUDE_parabola_tangents_perpendicular_iff_P_on_line_l0_29

/-- Parabola C: x^2 = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l: y = -1 -/
def line (x y : ℝ) : Prop := y = -1

/-- Point P is on line l -/
def P_on_line (P : ℝ × ℝ) : Prop := line P.1 P.2

/-- PA and PB are perpendicular -/
def tangents_perpendicular (P A B : ℝ × ℝ) : Prop :=
  let slope_PA := (A.2 - P.2) / (A.1 - P.1)
  let slope_PB := (B.2 - P.2) / (B.1 - P.1)
  slope_PA * slope_PB = -1

/-- A and B are points on the parabola -/
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

/-- PA and PB are tangent to the parabola at A and B respectively -/
def tangent_lines (P A B : ℝ × ℝ) : Prop :=
  points_on_parabola A B ∧
  (A.2 - P.2) / (A.1 - P.1) = A.1 / 2 ∧
  (B.2 - P.2) / (B.1 - P.1) = B.1 / 2

theorem parabola_tangents_perpendicular_iff_P_on_line
  (P A B : ℝ × ℝ) :
  tangent_lines P A B →
  (P_on_line P ↔ tangents_perpendicular P A B) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangents_perpendicular_iff_P_on_line_l0_29


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l0_35

theorem rectangle_length_calculation (w : ℝ) (l_increase : ℝ) (w_decrease : ℝ) :
  w = 40 →
  l_increase = 0.30 →
  w_decrease = 0.17692307692307693 →
  (1 + l_increase) * (1 - w_decrease) * w = w →
  ∃ l : ℝ, l = 40 / 1.3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_length_calculation_l0_35


namespace NUMINAMATH_CALUDE_sector_max_area_l0_72

/-- Given a sector with fixed perimeter P, prove that the maximum area is P^2/16
    and this maximum is achieved when the radius is P/4. -/
theorem sector_max_area (P : ℝ) (h : P > 0) :
  let max_area := P^2 / 16
  let max_radius := P / 4
  ∀ R l, R > 0 → l > 0 → 2 * R + l = P →
    (1/2 * R * l ≤ max_area) ∧
    (1/2 * max_radius * (P - 2 * max_radius) = max_area) :=
by sorry


end NUMINAMATH_CALUDE_sector_max_area_l0_72


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l0_46

/-- Simple interest calculation -/
theorem simple_interest_calculation (principal interest_rate simple_interest : ℚ) : 
  principal = 8 →
  interest_rate = 5 / 100 →
  simple_interest = 4.8 →
  ∃ (months : ℚ), months = 12 ∧ simple_interest = principal * interest_rate * months :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l0_46


namespace NUMINAMATH_CALUDE_no_playful_numbers_l0_99

/-- A two-digit positive integer is playful if it equals the sum of the cube of its tens digit and the square of its units digit. -/
def IsPlayful (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = a^3 + b^2

/-- The number of playful two-digit positive integers is zero. -/
theorem no_playful_numbers : ∀ n : ℕ, ¬(IsPlayful n) := by sorry

end NUMINAMATH_CALUDE_no_playful_numbers_l0_99


namespace NUMINAMATH_CALUDE_power_last_digit_match_l0_20

theorem power_last_digit_match : ∃ (m n : ℕ), 
  100 ≤ 2^m ∧ 2^m < 1000 ∧ 
  100 ≤ 3^n ∧ 3^n < 1000 ∧ 
  2^m % 10 = 3^n % 10 ∧ 
  2^m % 10 = 3 := by
sorry

end NUMINAMATH_CALUDE_power_last_digit_match_l0_20


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l0_21

theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q ≠ 1 →  -- common ratio is not 1
  a 1 + a 4 > a 2 + a 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l0_21


namespace NUMINAMATH_CALUDE_sin_15_75_simplification_l0_13

theorem sin_15_75_simplification : 2 * Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_75_simplification_l0_13


namespace NUMINAMATH_CALUDE_quadratic_root_l0_68

theorem quadratic_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : 9*a - 3*b + c = 0) : 
  a*(-3)^2 + b*(-3) + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_l0_68


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l0_19

theorem mixed_fraction_product (X Y : ℕ) : 
  (X > 0) →
  (Y > 0) →
  (5 : ℚ) + 1 / X > 5 →
  (5 : ℚ) + 1 / X ≤ 11 / 2 →
  (5 + 1 / X) * (Y + 1 / 2) = 43 →
  X = 17 ∧ Y = 8 := by
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l0_19


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l0_15

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ y : ℝ, 1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l0_15


namespace NUMINAMATH_CALUDE_variable_order_l0_76

theorem variable_order (a b c d : ℝ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : 
  c > a ∧ a > b ∧ b > d := by
  sorry

end NUMINAMATH_CALUDE_variable_order_l0_76


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_h_eq_two_l0_81

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (c : ℝ), l1.direction = c • l2.direction

/-- The first line parameterized by s -/
def line1 (h : ℝ) : Line3D :=
  { point := (1, 0, 4),
    direction := (2, -1, h) }

/-- The second line parameterized by t -/
def line2 : Line3D :=
  { point := (0, 0, -6),
    direction := (3, 1, -2) }

/-- The main theorem stating the condition for coplanarity -/
theorem lines_coplanar_iff_h_eq_two :
  ∀ h : ℝ, are_coplanar (line1 h) line2 ↔ h = 2 := by
  sorry


end NUMINAMATH_CALUDE_lines_coplanar_iff_h_eq_two_l0_81


namespace NUMINAMATH_CALUDE_cricket_team_size_l0_0

/-- The number of players on a cricket team satisfying specific conditions -/
theorem cricket_team_size :
  ∀ (total_players throwers non_throwers right_handed : ℕ),
    throwers = 37 →
    non_throwers = total_players - throwers →
    right_handed = 51 →
    right_handed = throwers + (2 * non_throwers / 3) →
    total_players = 58 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l0_0


namespace NUMINAMATH_CALUDE_fifth_point_coordinate_l0_79

/-- A sequence of 16 numbers where each number (except the first and last) is the average of its two adjacent numbers -/
def ArithmeticSequence (a : Fin 16 → ℝ) : Prop :=
  a 0 = 2 ∧ 
  a 15 = 47 ∧ 
  ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2

theorem fifth_point_coordinate (a : Fin 16 → ℝ) (h : ArithmeticSequence a) : a 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_point_coordinate_l0_79


namespace NUMINAMATH_CALUDE_system1_neither_necessary_nor_sufficient_l0_42

-- Define the two systems of inequalities
def system1 (x y a b : ℝ) : Prop := x > a ∧ y > b
def system2 (x y a b : ℝ) : Prop := x + y > a + b ∧ x * y > a * b

-- Theorem stating that system1 is neither necessary nor sufficient for system2
theorem system1_neither_necessary_nor_sufficient :
  ¬(∀ x y a b : ℝ, system1 x y a b → system2 x y a b) ∧
  ¬(∀ x y a b : ℝ, system2 x y a b → system1 x y a b) :=
sorry

end NUMINAMATH_CALUDE_system1_neither_necessary_nor_sufficient_l0_42


namespace NUMINAMATH_CALUDE_equation_solution_l0_94

theorem equation_solution : 
  ∃! x : ℝ, x ≠ -1 ∧ x ≠ -(3/2) ∧ x ≠ 1/2 ∧ x ≠ -(1/2) ∧
  (((((2*x+1)/(2*x-1))-1)/(1-((2*x-1)/(2*x+1)))) + 
   ((((2*x+1)/(2*x-1))-2)/(2-((2*x-1)/(2*x+1)))) +
   ((((2*x+1)/(2*x-1))-3)/(3-((2*x-1)/(2*x+1))))) = 0 ∧
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l0_94


namespace NUMINAMATH_CALUDE_complex_equation_solution_l0_28

theorem complex_equation_solution (z : ℂ) : (1 + z * Complex.I = z + Complex.I) → z = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l0_28


namespace NUMINAMATH_CALUDE_set_a_condition_l0_22

theorem set_a_condition (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a > 0}
  1 ∉ A → a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_set_a_condition_l0_22


namespace NUMINAMATH_CALUDE_distance_between_foci_l0_67

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y - 3)^2) + Real.sqrt ((x + 4)^2 + (y - 5)^2) = 18

-- Define the foci
def focus1 : ℝ × ℝ := (2, 3)
def focus2 : ℝ × ℝ := (-4, 5)

-- Theorem statement
theorem distance_between_foci :
  let d := Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2)
  d = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l0_67


namespace NUMINAMATH_CALUDE_marks_fruit_consumption_l0_37

/-- Given that Mark had 10 pieces of fruit for the week, kept 2 for next week,
    and brought 3 to school on Friday, prove that he ate 5 pieces in the first four days. -/
theorem marks_fruit_consumption
  (total_fruit : ℕ)
  (kept_for_next_week : ℕ)
  (brought_to_school : ℕ)
  (h1 : total_fruit = 10)
  (h2 : kept_for_next_week = 2)
  (h3 : brought_to_school = 3) :
  total_fruit - kept_for_next_week - brought_to_school = 5 := by
  sorry

end NUMINAMATH_CALUDE_marks_fruit_consumption_l0_37


namespace NUMINAMATH_CALUDE_inequality_proof_l0_36

open Real BigOperators

theorem inequality_proof (n : ℕ) (r s t u v : Fin n → ℝ) 
  (hr : ∀ i, r i > 1) (hs : ∀ i, s i > 1) (ht : ∀ i, t i > 1) (hu : ∀ i, u i > 1) (hv : ∀ i, v i > 1) :
  let R := (∑ i, r i) / n
  let S := (∑ i, s i) / n
  let T := (∑ i, t i) / n
  let U := (∑ i, u i) / n
  let V := (∑ i, v i) / n
  ∑ i, ((r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) ≥ 
    ((R * S * T * U * V + 1) / (R * S * T * U * V - 1)) ^ n :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l0_36


namespace NUMINAMATH_CALUDE_thread_length_calculation_l0_38

/-- The total length of thread required given an original length and an additional fraction -/
def total_length (original : ℝ) (additional_fraction : ℝ) : ℝ :=
  original + original * additional_fraction

/-- Theorem: Given a 12 cm thread and an additional three-quarters requirement, the total length is 21 cm -/
theorem thread_length_calculation : total_length 12 (3/4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_thread_length_calculation_l0_38


namespace NUMINAMATH_CALUDE_normal_dist_peak_l0_57

/-- A normal distribution with probability 0.5 of falling within the interval (0.2, +∞) -/
structure NormalDist where
  pdf : ℝ → ℝ
  cdf : ℝ → ℝ
  right_tail_prob : cdf 0.2 = 0.5

/-- The peak of the probability density function occurs at x = 0.2 -/
theorem normal_dist_peak (d : NormalDist) : 
  ∀ x : ℝ, d.pdf x ≤ d.pdf 0.2 :=
sorry

end NUMINAMATH_CALUDE_normal_dist_peak_l0_57


namespace NUMINAMATH_CALUDE_min_squares_covering_sqrt63_l0_32

theorem min_squares_covering_sqrt63 :
  ∀ n : ℕ, n ≥ 2 → (4 * n - 4 ≥ Real.sqrt 63 ↔ n ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_squares_covering_sqrt63_l0_32


namespace NUMINAMATH_CALUDE_delta_k_zero_iff_ge_four_l0_7

def u (n : ℕ) : ℕ := n^3 + n

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => λ n => f (n + 1) - f n

def Δk : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ Δk k

theorem delta_k_zero_iff_ge_four (k : ℕ) :
  (∀ n, Δk k u n = 0) ↔ k ≥ 4 := by sorry

end NUMINAMATH_CALUDE_delta_k_zero_iff_ge_four_l0_7


namespace NUMINAMATH_CALUDE_chocolate_discount_l0_25

/-- Calculates the discount amount given the original price and final price -/
def discount (original_price final_price : ℚ) : ℚ :=
  original_price - final_price

/-- Proves that the discount on a chocolate with original price $2.00 and final price $1.43 is $0.57 -/
theorem chocolate_discount :
  let original_price : ℚ := 2
  let final_price : ℚ := 143/100
  discount original_price final_price = 57/100 := by
sorry

end NUMINAMATH_CALUDE_chocolate_discount_l0_25


namespace NUMINAMATH_CALUDE_sup_good_is_ln_2_l0_24

/-- A positive real number d is good if there exists an infinite sequence
    a₁, a₂, a₃, ... ∈ (0,d) such that for each n, the points a₁, a₂, ..., aₙ
    partition the interval [0,d] into segments of length at most 1/n each. -/
def IsGood (d : ℝ) : Prop :=
  d > 0 ∧ ∃ a : ℕ → ℝ, ∀ n : ℕ,
    (∀ i : ℕ, i ≤ n → 0 < a i ∧ a i < d) ∧
    (∀ i : ℕ, i ≤ n → ∀ j : ℕ, j ≤ n → i ≠ j → |a i - a j| ≤ 1 / n) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ d → ∃ i : ℕ, i ≤ n ∧ |x - a i| ≤ 1 / n)

/-- The supremum of the set of all good numbers is ln 2. -/
theorem sup_good_is_ln_2 : sSup {d : ℝ | IsGood d} = Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_sup_good_is_ln_2_l0_24


namespace NUMINAMATH_CALUDE_joans_kittens_l0_93

/-- Represents the number of kittens Joan gave to her friends -/
def kittens_given_away (initial_kittens current_kittens : ℕ) : ℕ :=
  initial_kittens - current_kittens

/-- Proves that Joan gave away 2 kittens -/
theorem joans_kittens : kittens_given_away 8 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joans_kittens_l0_93


namespace NUMINAMATH_CALUDE_shoe_alteration_cost_l0_74

theorem shoe_alteration_cost (pairs : ℕ) (total_cost : ℕ) (cost_per_shoe : ℕ) :
  pairs = 17 →
  total_cost = 986 →
  cost_per_shoe = total_cost / (pairs * 2) →
  cost_per_shoe = 29 := by
  sorry

end NUMINAMATH_CALUDE_shoe_alteration_cost_l0_74


namespace NUMINAMATH_CALUDE_prob_between_30_and_40_l0_48

/-- Represents the age groups in the population -/
inductive AgeGroup
  | LessThan20
  | Between20And30
  | Between30And40
  | MoreThan40

/-- Represents the population with their age distribution -/
structure Population where
  total : ℕ
  ageDist : AgeGroup → ℕ
  sum_eq_total : (ageDist AgeGroup.LessThan20) + (ageDist AgeGroup.Between20And30) + 
                 (ageDist AgeGroup.Between30And40) + (ageDist AgeGroup.MoreThan40) = total

/-- The probability of selecting a person from a specific age group -/
def prob (p : Population) (ag : AgeGroup) : ℚ :=
  (p.ageDist ag : ℚ) / (p.total : ℚ)

/-- The given population -/
def givenPopulation : Population where
  total := 200
  ageDist := fun
    | AgeGroup.LessThan20 => 20
    | AgeGroup.Between20And30 => 30
    | AgeGroup.Between30And40 => 70
    | AgeGroup.MoreThan40 => 80
  sum_eq_total := by sorry

theorem prob_between_30_and_40 : 
  prob givenPopulation AgeGroup.Between30And40 = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_prob_between_30_and_40_l0_48


namespace NUMINAMATH_CALUDE_function_range_and_inequality_l0_90

theorem function_range_and_inequality (a b c m : ℝ) : 
  (∀ x, -x^2 + a*x + b ≤ 0) →
  (∀ x, -x^2 + a*x + b > c - 1 ↔ m - 4 < x ∧ x < m + 1) →
  c = 29/4 := by sorry

end NUMINAMATH_CALUDE_function_range_and_inequality_l0_90


namespace NUMINAMATH_CALUDE_hypotenuse_length_l0_45

/-- A right triangle with specific properties -/
structure RightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  longer_leg_property : longer_leg = 3 * shorter_leg - 1
  area_property : (1 / 2) * shorter_leg * longer_leg = 24
  pythagorean_theorem : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The length of the hypotenuse in the specific right triangle is √137 -/
theorem hypotenuse_length (t : RightTriangle) : t.hypotenuse = Real.sqrt 137 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l0_45


namespace NUMINAMATH_CALUDE_stating_convex_polygon_decomposition_iff_centrally_symmetric_l0_55

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields and conditions for a convex polygon
  is_convex : Bool

/-- A parallelogram. -/
structure Parallelogram where
  -- Add necessary fields for a parallelogram

/-- Represents a decomposition of a polygon into parallelograms. -/
def Decomposition (p : ConvexPolygon) := List Parallelogram

/-- Checks if a decomposition is valid for a given polygon. -/
def is_valid_decomposition (p : ConvexPolygon) (d : Decomposition p) : Prop :=
  sorry

/-- Checks if a polygon is centrally symmetric. -/
def is_centrally_symmetric (p : ConvexPolygon) : Prop :=
  sorry

/-- 
Theorem stating that a convex polygon can be decomposed into a finite number of parallelograms 
if and only if it is centrally symmetric.
-/
theorem convex_polygon_decomposition_iff_centrally_symmetric (p : ConvexPolygon) :
  (∃ d : Decomposition p, is_valid_decomposition p d) ↔ is_centrally_symmetric p :=
sorry

end NUMINAMATH_CALUDE_stating_convex_polygon_decomposition_iff_centrally_symmetric_l0_55


namespace NUMINAMATH_CALUDE_A_equals_B_l0_41

/-- The number of digits written when listing integers from 1 to 10^(n-1) -/
def A (n : ℕ) : ℕ := sorry

/-- The number of zeros written when listing integers from 1 to 10^n -/
def B (n : ℕ) : ℕ := sorry

/-- Theorem stating that A(n) equals B(n) for all positive integers n -/
theorem A_equals_B (n : ℕ) (h : n > 0) : A n = B n := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l0_41


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l0_44

/-- Given an arithmetic sequence with first term 2/3 and second term 4/3,
    prove that its tenth term is 20/3. -/
theorem tenth_term_of_arithmetic_sequence :
  let a₁ : ℚ := 2/3  -- First term
  let a₂ : ℚ := 4/3  -- Second term
  let d : ℚ := a₂ - a₁  -- Common difference
  let a₁₀ : ℚ := a₁ + 9 * d  -- Tenth term
  a₁₀ = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l0_44


namespace NUMINAMATH_CALUDE_oranges_used_proof_l0_80

/-- Calculates the total number of oranges used to make juice -/
def total_oranges (oranges_per_glass : ℕ) (glasses : ℕ) : ℕ :=
  oranges_per_glass * glasses

/-- Proves that the total number of oranges used is 30 -/
theorem oranges_used_proof (oranges_per_glass : ℕ) (glasses : ℕ) 
  (h1 : oranges_per_glass = 3) 
  (h2 : glasses = 10) : 
  total_oranges oranges_per_glass glasses = 30 := by
sorry

end NUMINAMATH_CALUDE_oranges_used_proof_l0_80


namespace NUMINAMATH_CALUDE_average_first_five_multiples_of_five_l0_31

/-- The average of the first 5 multiples of 5 is 15 -/
theorem average_first_five_multiples_of_five : 
  (List.sum (List.map (· * 5) (List.range 5))) / 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_first_five_multiples_of_five_l0_31


namespace NUMINAMATH_CALUDE_number_equation_l0_5

theorem number_equation : ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l0_5


namespace NUMINAMATH_CALUDE_max_value_interval_range_l0_78

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The interval (a, 6-a^2) --/
def interval (a : ℝ) : Set ℝ := {x | a < x ∧ x < 6 - a^2}

/-- Theorem stating the range of a for which f has a maximum on the interval --/
theorem max_value_interval_range :
  ∀ a : ℝ, (∃ x_max ∈ interval a, ∀ x ∈ interval a, f x ≤ f x_max) →
    a > -Real.sqrt 7 ∧ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_interval_range_l0_78


namespace NUMINAMATH_CALUDE_intersection_with_complement_l0_27

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 4, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l0_27


namespace NUMINAMATH_CALUDE_lowella_score_l0_16

/-- Given a 100-item exam, prove that Lowella's score is 22% when:
    - Pamela's score is 20 percentage points higher than Lowella's
    - Mandy's score is twice Pamela's score
    - Mandy's score is 84% -/
theorem lowella_score (pamela_score mandy_score lowella_score : ℚ) : 
  pamela_score = lowella_score + 20 →
  mandy_score = 2 * pamela_score →
  mandy_score = 84 →
  lowella_score = 22 := by
  sorry

#check lowella_score

end NUMINAMATH_CALUDE_lowella_score_l0_16


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l0_66

theorem employee_pay_percentage (total_pay y_pay : ℝ) (h1 : total_pay = 570) (h2 : y_pay = 259.09) :
  let x_pay := total_pay - y_pay
  (x_pay / y_pay) * 100 = 120.03 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l0_66


namespace NUMINAMATH_CALUDE_quadratic_function_property_l0_23

/-- A quadratic function of the form y = 3(x - a)² -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := 3 * (x - a)^2

/-- The property that y increases as x increases when x > 2 -/
def increasing_when_x_gt_2 (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → quadratic_function a x₂ > quadratic_function a x₁

theorem quadratic_function_property (a : ℝ) :
  increasing_when_x_gt_2 a → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l0_23


namespace NUMINAMATH_CALUDE_combined_age_is_28_l0_92

/-- Represents the ages of Michael and his brothers -/
structure FamilyAges where
  michael : ℕ
  younger_brother : ℕ
  older_brother : ℕ

/-- Defines the conditions for the ages of Michael and his brothers -/
def valid_ages (ages : FamilyAges) : Prop :=
  ages.younger_brother = 5 ∧
  ages.older_brother = 3 * ages.younger_brother ∧
  ages.older_brother = 2 * (ages.michael - 1) + 1

/-- Theorem stating that the combined age of Michael and his brothers is 28 years -/
theorem combined_age_is_28 (ages : FamilyAges) (h : valid_ages ages) :
  ages.michael + ages.younger_brother + ages.older_brother = 28 := by
  sorry


end NUMINAMATH_CALUDE_combined_age_is_28_l0_92


namespace NUMINAMATH_CALUDE_special_function_value_l0_51

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 3) ≤ f x + 3) ∧
  (∀ x, f (x + 2) ≥ f x + 2) ∧
  (f 4 = 2008)

/-- The theorem to be proved -/
theorem special_function_value (f : ℝ → ℝ) (h : SpecialFunction f) : f 2008 = 4012 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l0_51


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l0_56

theorem min_value_sum_reciprocals (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a^n)) + (1 / (1 + b^n)) ≥ 1 ∧ 
  ((1 / (1 + a^n)) + (1 / (1 + b^n)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

#check min_value_sum_reciprocals

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l0_56


namespace NUMINAMATH_CALUDE_a_plus_b_value_l0_60

theorem a_plus_b_value (a b : ℝ) 
  (h1 : |a| = 4)
  (h2 : Real.sqrt (b^2) = 3)
  (h3 : a + b > 0) :
  a + b = 1 ∨ a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l0_60


namespace NUMINAMATH_CALUDE_largest_intersection_point_l0_53

-- Define the polynomial P(x)
def P (x b : ℝ) : ℝ := x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3

-- Define the line L(x)
def L (x c d : ℝ) : ℝ := c*x - d

-- Theorem statement
theorem largest_intersection_point (b c d : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (∀ x : ℝ, P x b = L x c d ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) →
  (∃ x_max : ℝ, P x_max b = L x_max c d ∧ 
    ∀ x : ℝ, P x b = L x c d → x ≤ x_max) →
  (∃ x_max : ℝ, P x_max b = L x_max c d ∧ 
    ∀ x : ℝ, P x b = L x c d → x ≤ x_max ∧ x_max = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_largest_intersection_point_l0_53


namespace NUMINAMATH_CALUDE_sum_of_cubes_l0_10

theorem sum_of_cubes (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (sum_prod_eq : x*y + y*z + z*x = -3) 
  (prod_eq : x*y*z = 2) : 
  x^3 + y^3 + z^3 = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l0_10


namespace NUMINAMATH_CALUDE_intersection_point_correct_l0_52

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The first line: y = 3x + 4 -/
def line₁ (x y : ℚ) : Prop := y = m₁ * x + 4

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The point through which the perpendicular line passes -/
def point : (ℚ × ℚ) := (3, 2)

/-- The perpendicular line passing through (3, 2) -/
def line₂ (x y : ℚ) : Prop := y - point.2 = m₂ * (x - point.1)

/-- The intersection point of the two lines -/
def intersection_point : (ℚ × ℚ) := (-3/10, 31/10)

/-- Theorem stating that the intersection point is correct -/
theorem intersection_point_correct :
  line₁ intersection_point.1 intersection_point.2 ∧
  line₂ intersection_point.1 intersection_point.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l0_52


namespace NUMINAMATH_CALUDE_g_minus_one_eq_zero_iff_s_eq_neg_six_l0_73

/-- The function g(x) as defined in the problem -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 2 * x^2 + x + s

/-- Theorem stating that g(-1) = 0 if and only if s = -6 -/
theorem g_minus_one_eq_zero_iff_s_eq_neg_six :
  ∀ s : ℝ, g s (-1) = 0 ↔ s = -6 := by sorry

end NUMINAMATH_CALUDE_g_minus_one_eq_zero_iff_s_eq_neg_six_l0_73


namespace NUMINAMATH_CALUDE_twelve_people_in_line_l0_17

/-- The number of people in a line with Jeanne, given the number of people in front and behind her -/
def people_in_line (people_in_front : ℕ) (people_behind : ℕ) : ℕ :=
  people_in_front + 1 + people_behind

/-- Theorem stating that there are 12 people in the line -/
theorem twelve_people_in_line :
  people_in_line 4 7 = 12 := by
  sorry

#check twelve_people_in_line

end NUMINAMATH_CALUDE_twelve_people_in_line_l0_17


namespace NUMINAMATH_CALUDE_initial_charge_value_l0_65

/-- The charge for the first 1/5 of a minute in cents -/
def initial_charge : ℝ := sorry

/-- The charge for each additional 1/5 of a minute in cents -/
def additional_charge : ℝ := 0.40

/-- The total charge for an 8-minute call in cents -/
def total_charge : ℝ := 18.70

/-- The number of 1/5 minute intervals in 8 minutes -/
def total_intervals : ℕ := 8 * 5

/-- The number of additional 1/5 minute intervals after the first one -/
def additional_intervals : ℕ := total_intervals - 1

theorem initial_charge_value :
  initial_charge = 3.10 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_charge_value_l0_65


namespace NUMINAMATH_CALUDE_ibrahim_palace_count_l0_98

/-- Represents a square grid of rooms -/
structure RoomGrid where
  size : Nat
  has_door_between_rooms : Bool
  has_window_on_outer_wall : Bool

/-- Calculates the number of windows in the grid -/
def count_windows (grid : RoomGrid) : Nat :=
  if grid.has_window_on_outer_wall then
    4 * grid.size
  else
    0

/-- Calculates the number of doors in the grid -/
def count_doors (grid : RoomGrid) : Nat :=
  if grid.has_door_between_rooms then
    2 * grid.size * (grid.size - 1)
  else
    0

/-- Theorem stating the number of windows and doors in the specific 10x10 grid -/
theorem ibrahim_palace_count (grid : RoomGrid)
  (h_size : grid.size = 10)
  (h_door : grid.has_door_between_rooms = true)
  (h_window : grid.has_window_on_outer_wall = true) :
  count_windows grid = 40 ∧ count_doors grid = 180 := by
  sorry


end NUMINAMATH_CALUDE_ibrahim_palace_count_l0_98


namespace NUMINAMATH_CALUDE_some_number_value_l0_40

theorem some_number_value (t k some_number : ℝ) 
  (h1 : t = 5 / 9 * (k - some_number))
  (h2 : t = 35)
  (h3 : k = 95) : 
  some_number = 32 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l0_40


namespace NUMINAMATH_CALUDE_mrs_hilt_money_left_l0_47

def initial_amount : ℕ := 10
def truck_cost : ℕ := 3
def pencil_case_cost : ℕ := 2

theorem mrs_hilt_money_left : 
  initial_amount - (truck_cost + pencil_case_cost) = 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_money_left_l0_47


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_six_l0_88

theorem no_linear_term_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (2*x + m) * (x - 3) = 2*x^2 - 3*m) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_six_l0_88


namespace NUMINAMATH_CALUDE_hotel_beds_count_l0_97

theorem hotel_beds_count (total_rooms : ℕ) (two_bed_rooms : ℕ) (beds_in_two_bed_room : ℕ) (beds_in_three_bed_room : ℕ) 
    (h1 : total_rooms = 13)
    (h2 : two_bed_rooms = 8)
    (h3 : beds_in_two_bed_room = 2)
    (h4 : beds_in_three_bed_room = 3) :
  two_bed_rooms * beds_in_two_bed_room + (total_rooms - two_bed_rooms) * beds_in_three_bed_room = 31 := by
  sorry

#eval 8 * 2 + (13 - 8) * 3  -- This should output 31

end NUMINAMATH_CALUDE_hotel_beds_count_l0_97


namespace NUMINAMATH_CALUDE_find_k_l0_4

def is_max_solution (k : ℝ) : Prop :=
  ∀ x : ℝ, |x^2 - 4*x + k| + |x - 3| ≤ 5 → x ≤ 3

theorem find_k : ∃! k : ℝ, is_max_solution k ∧ k = 8 := by sorry

end NUMINAMATH_CALUDE_find_k_l0_4


namespace NUMINAMATH_CALUDE_triangle_properties_l0_34

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  b + a * Real.cos C = 0 →
  Real.sin A = 2 * Real.sin (A + C) →
  C = 2 * π / 3 ∧ c / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l0_34


namespace NUMINAMATH_CALUDE_markus_more_marbles_l0_58

theorem markus_more_marbles (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) 
  (markus_bags : ℕ) (markus_marbles_per_bag : ℕ) 
  (h1 : mara_bags = 12) (h2 : mara_marbles_per_bag = 2) 
  (h3 : markus_bags = 2) (h4 : markus_marbles_per_bag = 13) : 
  markus_bags * markus_marbles_per_bag - mara_bags * mara_marbles_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_markus_more_marbles_l0_58


namespace NUMINAMATH_CALUDE_shoes_cost_calculation_l0_75

def budget : ℚ := 200
def shirt_cost : ℚ := 30
def pants_cost : ℚ := 46
def coat_cost : ℚ := 38
def socks_cost : ℚ := 11
def belt_cost : ℚ := 18
def necktie_cost : ℚ := 22
def remaining_money : ℚ := 16

def other_items_cost : ℚ := shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + necktie_cost

theorem shoes_cost_calculation :
  ∃ (shoes_cost : ℚ), 
    shoes_cost = budget - remaining_money - other_items_cost ∧
    shoes_cost = 19 :=
by sorry

end NUMINAMATH_CALUDE_shoes_cost_calculation_l0_75


namespace NUMINAMATH_CALUDE_janet_practice_days_l0_43

def total_miles : ℕ := 72
def miles_per_day : ℕ := 8

theorem janet_practice_days : 
  total_miles / miles_per_day = 9 := by sorry

end NUMINAMATH_CALUDE_janet_practice_days_l0_43


namespace NUMINAMATH_CALUDE_roots_of_equation_l0_18

theorem roots_of_equation (x : ℝ) : 
  (3 * Real.sqrt x + 3 / Real.sqrt x = 7) ↔ 
  (x = ((7 + Real.sqrt 13) / 6)^2 ∨ x = ((7 - Real.sqrt 13) / 6)^2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l0_18


namespace NUMINAMATH_CALUDE_distance_on_line_l0_62

/-- Given two points (a, b) and (c, d) on a line y = mx + k, 
    the distance between them is |a - c|√(1 + m²) -/
theorem distance_on_line (m k a b c d : ℝ) :
  b = m * a + k →
  d = m * c + k →
  Real.sqrt ((c - a)^2 + (d - b)^2) = |a - c| * Real.sqrt (1 + m^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_on_line_l0_62


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l0_30

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

theorem tangent_line_y_intercept (c1 c2 : Circle) (l : Line) :
  c1.center = (3, 0) →
  c1.radius = 3 →
  c2.center = (7, 0) →
  c2.radius = 2 →
  is_tangent l c1 →
  is_tangent l c2 →
  l.y_intercept = 12 * Real.sqrt 17 / 17 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l0_30


namespace NUMINAMATH_CALUDE_same_side_of_line_l0_95

/-- Define a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the line equation --/
def line_equation (p : Point) : ℝ := p.x + p.y - 1

/-- Check if a point is on the positive side of the line --/
def is_positive_side (p : Point) : Prop := line_equation p > 0

/-- The reference point (1,2) --/
def reference_point : Point := ⟨1, 2⟩

/-- The point to be checked (-1,3) --/
def check_point : Point := ⟨-1, 3⟩

/-- Theorem statement --/
theorem same_side_of_line : 
  is_positive_side reference_point → is_positive_side check_point :=
by sorry

end NUMINAMATH_CALUDE_same_side_of_line_l0_95


namespace NUMINAMATH_CALUDE_skew_parallel_relationship_l0_33

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- But for simplicity, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not parallel and do not intersect
  sorry

-- Define what it means for two lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they have the same direction vector
  sorry

-- Define what it means for two lines to intersect
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Two lines intersect if they share a point
  sorry

-- The main theorem
theorem skew_parallel_relationship (a b c : Line3D) :
  are_skew a b → are_parallel a c → (are_skew c b ∨ do_intersect c b) :=
by
  sorry

end NUMINAMATH_CALUDE_skew_parallel_relationship_l0_33


namespace NUMINAMATH_CALUDE_rotation_270_degrees_l0_70

theorem rotation_270_degrees (z : ℂ) : z = -8 - 4*I → z * (-I) = -4 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_rotation_270_degrees_l0_70


namespace NUMINAMATH_CALUDE_train_overtake_l0_71

/-- Proves that Train B overtakes Train A in 120 minutes given the specified conditions -/
theorem train_overtake (speed_a speed_b : ℝ) (head_start : ℝ) (overtake_time : ℝ) : 
  speed_a = 60 →
  speed_b = 80 →
  head_start = 40 / 60 →
  overtake_time = 120 / 60 →
  speed_a * (head_start + overtake_time) = speed_b * overtake_time :=
by sorry

end NUMINAMATH_CALUDE_train_overtake_l0_71


namespace NUMINAMATH_CALUDE_bakery_pie_distribution_l0_1

theorem bakery_pie_distribution (initial_pie : ℚ) (additional_percentage : ℚ) (num_employees : ℕ) :
  initial_pie = 8/9 →
  additional_percentage = 1/10 →
  num_employees = 4 →
  (initial_pie + initial_pie * additional_percentage) / num_employees = 11/45 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_distribution_l0_1


namespace NUMINAMATH_CALUDE_assistant_end_time_l0_64

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a bracelet producer -/
structure Producer where
  startTime : Time
  endTime : Time
  rate : Nat
  interval : Nat
  deriving Repr

def craftsman : Producer := {
  startTime := { hours := 8, minutes := 0 }
  endTime := { hours := 12, minutes := 0 }
  rate := 6
  interval := 20
}

def assistant : Producer := {
  startTime := { hours := 9, minutes := 0 }
  endTime := { hours := 0, minutes := 0 }  -- To be determined
  rate := 8
  interval := 30
}

def calculateProduction (p : Producer) : Nat :=
  sorry

def calculateEndTime (p : Producer) (targetProduction : Nat) : Time :=
  sorry

theorem assistant_end_time :
  calculateEndTime assistant (calculateProduction craftsman) = { hours := 13, minutes := 30 } :=
sorry

end NUMINAMATH_CALUDE_assistant_end_time_l0_64


namespace NUMINAMATH_CALUDE_mishas_current_dollars_l0_54

theorem mishas_current_dollars (current_dollars target_dollars needed_dollars : ℕ) 
  (h1 : target_dollars = 47)
  (h2 : needed_dollars = 13)
  (h3 : current_dollars + needed_dollars = target_dollars) :
  current_dollars = 34 := by
  sorry

end NUMINAMATH_CALUDE_mishas_current_dollars_l0_54


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l0_50

theorem cubic_expression_evaluation : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l0_50


namespace NUMINAMATH_CALUDE_min_value_expression_l0_85

theorem min_value_expression (x y z k : ℝ) 
  (hx : -2 < x ∧ x < 2) 
  (hy : -2 < y ∧ y < 2) 
  (hz : -2 < z ∧ z < 2) 
  (hk : k > 0) :
  (k / ((2 - x) * (2 - y) * (2 - z))) + (k / ((2 + x) * (2 + y) * (2 + z))) ≥ 2 * k :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l0_85


namespace NUMINAMATH_CALUDE_sum_of_cubes_difference_l0_87

theorem sum_of_cubes_difference (p q r : ℕ+) :
  (p + q + r : ℕ)^3 - p^3 - q^3 - r^3 = 200 →
  (p : ℕ) + q + r = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_difference_l0_87


namespace NUMINAMATH_CALUDE_last_number_proof_l0_96

theorem last_number_proof (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 3 →
  a + d = 13 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l0_96


namespace NUMINAMATH_CALUDE_miscalculation_correction_l0_6

theorem miscalculation_correction (x : ℝ) : 
  63 + x = 69 → 36 / x = 6 := by sorry

end NUMINAMATH_CALUDE_miscalculation_correction_l0_6


namespace NUMINAMATH_CALUDE_square_dissection_existence_l0_26

theorem square_dissection_existence :
  ∃ (S a b c : ℝ), 
    S > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    S^2 = a^2 + 3*b^2 + 5*c^2 :=
by sorry

end NUMINAMATH_CALUDE_square_dissection_existence_l0_26


namespace NUMINAMATH_CALUDE_dandelion_count_l0_9

/-- Proves the original number of yellow and white dandelions given the initial and final conditions --/
theorem dandelion_count : ∀ y w : ℕ,
  y + w = 35 →
  y - 2 = 2 * (w - 6) →
  y = 20 ∧ w = 15 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_count_l0_9


namespace NUMINAMATH_CALUDE_prob_no_dessert_is_35_percent_l0_11

/-- Represents the probability of different order combinations -/
structure OrderProbabilities where
  dessert_coffee : ℝ
  dessert_only : ℝ
  coffee_only : ℝ
  appetizer_dessert : ℝ
  appetizer_coffee : ℝ
  appetizer_dessert_coffee : ℝ

/-- Calculate the probability of not ordering dessert -/
def prob_no_dessert (p : OrderProbabilities) : ℝ :=
  1 - (p.dessert_coffee + p.dessert_only + p.appetizer_dessert + p.appetizer_dessert_coffee)

/-- Theorem: The probability of not ordering dessert is 35% -/
theorem prob_no_dessert_is_35_percent (p : OrderProbabilities)
  (h1 : p.dessert_coffee = 0.60)
  (h2 : p.dessert_only = 0.15)
  (h3 : p.coffee_only = 0.10)
  (h4 : p.appetizer_dessert = 0.05)
  (h5 : p.appetizer_coffee = 0.08)
  (h6 : p.appetizer_dessert_coffee = 0.03) :
  prob_no_dessert p = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_prob_no_dessert_is_35_percent_l0_11


namespace NUMINAMATH_CALUDE_smallest_k_for_omega_inequality_l0_49

/-- ω(n) denotes the number of positive prime divisors of n -/
def omega (n : ℕ) : ℕ := sorry

/-- The theorem states that 5 is the smallest positive integer k 
    such that 2^ω(n) ≤ k∙n^(1/4) for all positive integers n -/
theorem smallest_k_for_omega_inequality : 
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → (2 : ℝ)^(omega n : ℝ) ≤ k * (n : ℝ)^(1/4)) ∧ 
  (∀ k : ℕ, 0 < k → k < 5 → ∃ n : ℕ, n > 0 ∧ (2 : ℝ)^(omega n : ℝ) > k * (n : ℝ)^(1/4)) ∧
  (∀ n : ℕ, n > 0 → (2 : ℝ)^(omega n : ℝ) ≤ 5 * (n : ℝ)^(1/4)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_omega_inequality_l0_49


namespace NUMINAMATH_CALUDE_johns_allowance_l0_59

theorem johns_allowance (A : ℚ) : 
  (A > 0) →
  (3/5 * A + 1/3 * (A - 3/5 * A) + 92/100 = A) →
  A = 345/100 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l0_59


namespace NUMINAMATH_CALUDE_distance_AB_is_correct_l0_69

/-- The distance between two points A and B, given the conditions of the problem. -/
def distance_AB : ℝ :=
  let first_meeting_distance : ℝ := 700
  let second_meeting_distance : ℝ := 400
  -- Define the distance as a variable to be solved
  let d : ℝ := 1700
  d

theorem distance_AB_is_correct : distance_AB = 1700 := by
  -- Unfold the definition of distance_AB
  unfold distance_AB
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_distance_AB_is_correct_l0_69
