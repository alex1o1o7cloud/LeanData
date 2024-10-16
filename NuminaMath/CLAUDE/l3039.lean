import Mathlib

namespace NUMINAMATH_CALUDE_percentage_difference_l3039_303907

theorem percentage_difference : 
  (40 * 80 / 100) - (25 * 4 / 5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3039_303907


namespace NUMINAMATH_CALUDE_last_term_value_l3039_303954

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => b
  | 2 => 5 * a
  | 3 => 7
  | 4 => 3 * b
  | n + 5 => arithmetic_sequence a b n

-- Define the sum of the sequence
def sequence_sum (a b : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (arithmetic_sequence a b) |>.sum

-- Theorem statement
theorem last_term_value (a b : ℝ) (n : ℕ) :
  sequence_sum a b n = 2500 →
  ∃ c, arithmetic_sequence a b (n - 1) = c ∧ c = 99 := by
  sorry

end NUMINAMATH_CALUDE_last_term_value_l3039_303954


namespace NUMINAMATH_CALUDE_no_third_quadrant_implies_m_leq_1_l3039_303962

def linear_function (x m : ℝ) : ℝ := -2 * x + 1 - m

theorem no_third_quadrant_implies_m_leq_1 :
  ∀ m : ℝ, (∀ x y : ℝ, y = linear_function x m → (x < 0 → y ≥ 0)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_third_quadrant_implies_m_leq_1_l3039_303962


namespace NUMINAMATH_CALUDE_angle_sum_is_ninety_degrees_l3039_303919

theorem angle_sum_is_ninety_degrees (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_ninety_degrees_l3039_303919


namespace NUMINAMATH_CALUDE_prob_two_aces_full_deck_prob_two_aces_after_two_kings_l3039_303937

/-- Represents a deck of cards with Aces, Kings, and Queens -/
structure Deck :=
  (num_aces : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing two Aces from a given deck -/
def prob_two_aces (d : Deck) : ℚ :=
  (d.num_aces.choose 2 : ℚ) / (d.num_aces + d.num_kings + d.num_queens).choose 2

/-- The full deck with 4 each of Aces, Kings, and Queens -/
def full_deck : Deck := ⟨4, 4, 4⟩

/-- The deck after two Kings have been drawn -/
def deck_after_two_kings : Deck := ⟨4, 2, 4⟩

theorem prob_two_aces_full_deck :
  prob_two_aces full_deck = 1 / 11 :=
sorry

theorem prob_two_aces_after_two_kings :
  prob_two_aces deck_after_two_kings = 2 / 15 :=
sorry

end NUMINAMATH_CALUDE_prob_two_aces_full_deck_prob_two_aces_after_two_kings_l3039_303937


namespace NUMINAMATH_CALUDE_trapezoid_area_in_circle_l3039_303913

/-- The area of a trapezoid inscribed in a circle -/
theorem trapezoid_area_in_circle (R : ℝ) (α : ℝ) (h : 0 < α ∧ α < π) :
  let trapezoid_area := R^2 * (1 + Real.sin (α/2)) * Real.cos (α/2)
  let diameter := 2 * R
  let chord := 2 * R * Real.sin (α/2)
  let height := R * Real.cos (α/2)
  trapezoid_area = (diameter + chord) * height / 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_circle_l3039_303913


namespace NUMINAMATH_CALUDE_colored_squares_count_l3039_303961

/-- The size of the square grid -/
def gridSize : ℕ := 101

/-- The number of L-shaped layers in the grid -/
def numLayers : ℕ := gridSize / 2

/-- The number of squares colored in the nth L-shaped layer -/
def squaresInLayer (n : ℕ) : ℕ := 8 * n

/-- The total number of colored squares in the grid -/
def totalColoredSquares : ℕ := 1 + (numLayers * (numLayers + 1) * 4)

/-- Theorem stating that the total number of colored squares is 10201 -/
theorem colored_squares_count :
  totalColoredSquares = 10201 := by sorry

end NUMINAMATH_CALUDE_colored_squares_count_l3039_303961


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3039_303978

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "outside of plane" relation
variable (outside_of_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : outside_of_plane m α)
  (h2 : outside_of_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l3039_303978


namespace NUMINAMATH_CALUDE_solve_for_x_l3039_303991

theorem solve_for_x (x y : ℝ) (h1 : x - y = 7) (h2 : x + y = 11) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3039_303991


namespace NUMINAMATH_CALUDE_sin_product_equality_l3039_303946

theorem sin_product_equality : 
  Real.sin (9 * π / 180) * Real.sin (45 * π / 180) * Real.sin (69 * π / 180) * Real.sin (81 * π / 180) = 
  (Real.sin (39 * π / 180) * Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_product_equality_l3039_303946


namespace NUMINAMATH_CALUDE_fundraiser_result_l3039_303988

def fundraiser (num_students : ℕ) (initial_needed : ℕ) (additional_needed : ℕ) 
               (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) (num_half_days : ℕ) : ℕ :=
  let total_per_student := initial_needed + additional_needed
  let total_needed := num_students * total_per_student
  let first_three_days := day1 + day2 + day3
  let half_day_amount := first_three_days / 2
  let total_raised := first_three_days + num_half_days * half_day_amount
  total_raised - total_needed

theorem fundraiser_result : 
  fundraiser 6 450 475 600 900 400 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_result_l3039_303988


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3039_303940

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  (x - 3) / (4 * x) = 0 ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3039_303940


namespace NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l3039_303984

-- Define a tetrahedron type
structure Tetrahedron where
  -- The volume of the tetrahedron
  volume : ℝ
  -- The distances between opposite edges
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  -- Ensure all distances are positive
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  d₃_pos : d₃ > 0

-- State the theorem
theorem tetrahedron_volume_lower_bound (t : Tetrahedron) : 
  t.volume ≥ (1/3) * t.d₁ * t.d₂ * t.d₃ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_lower_bound_l3039_303984


namespace NUMINAMATH_CALUDE_exists_sum_a1_b1_l3039_303955

-- Define the sequence (aₙ, bₙ)
def a_b : ℕ → ℝ × ℝ
| 0 => (0, 0)  -- We define the 0th term arbitrarily as it's not used
| (n+1) => let (aₙ, bₙ) := a_b n; (2*aₙ - bₙ, 2*bₙ + aₙ)

-- State the theorem
theorem exists_sum_a1_b1 : ∃ x : ℝ, 
  (a_b 50).1 = 3 ∧ (a_b 50).2 = 5 → 
  (a_b 1).1 + (a_b 1).2 = x := by
  sorry

end NUMINAMATH_CALUDE_exists_sum_a1_b1_l3039_303955


namespace NUMINAMATH_CALUDE_tank_inflow_rate_l3039_303994

theorem tank_inflow_rate (capacity : ℝ) (time_diff : ℝ) (slow_rate : ℝ) : 
  capacity > 0 → time_diff > 0 → slow_rate > 0 →
  let slow_time := capacity / slow_rate
  let fast_time := slow_time - time_diff
  fast_time > 0 →
  capacity / fast_time = 2 * slow_rate := by
  sorry

-- Example usage with given values
example : 
  let capacity := 20
  let time_diff := 5
  let slow_rate := 2
  let slow_time := capacity / slow_rate
  let fast_time := slow_time - time_diff
  capacity / fast_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_tank_inflow_rate_l3039_303994


namespace NUMINAMATH_CALUDE_females_together_arrangements_l3039_303971

/-- Represents the number of students of each gender -/
def num_males : ℕ := 2
def num_females : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_males + num_females

/-- The number of ways to arrange the students with females next to each other -/
def arrangements_with_females_together : ℕ := 12

/-- Theorem stating that the number of arrangements with females together is 12 -/
theorem females_together_arrangements :
  (arrangements_with_females_together = 12) ∧
  (num_males = 2) ∧
  (num_females = 2) ∧
  (total_students = 4) := by
  sorry

end NUMINAMATH_CALUDE_females_together_arrangements_l3039_303971


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3039_303936

theorem rectangle_dimensions : ∃ (l w : ℝ), 
  (l = 9 ∧ w = 8) ∧
  (l - 3 = w - 2) ∧
  ((l - 3)^2 = (1/2) * l * w) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3039_303936


namespace NUMINAMATH_CALUDE_root_equation_value_l3039_303997

theorem root_equation_value (m : ℝ) : 
  (2 * m^2 + 3 * m - 1 = 0) → (4 * m^2 + 6 * m - 2019 = -2017) := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3039_303997


namespace NUMINAMATH_CALUDE_power_equation_solution_l3039_303926

theorem power_equation_solution : ∃ x : ℝ, (5^5 * 9^3 : ℝ) = 3 * 15^x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3039_303926


namespace NUMINAMATH_CALUDE_x_value_proof_l3039_303905

theorem x_value_proof (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 3 * Real.sqrt 54 ∨ x = -3 * Real.sqrt 54 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3039_303905


namespace NUMINAMATH_CALUDE_school_ratio_problem_l3039_303908

theorem school_ratio_problem (S T : ℕ) : 
  S / T = 50 →
  (S + 50) / (T + 5) = 25 →
  T = 3 :=
by sorry

end NUMINAMATH_CALUDE_school_ratio_problem_l3039_303908


namespace NUMINAMATH_CALUDE_combined_weight_l3039_303999

/-- The combined weight of Tracy, John, and Jake is 150 kg -/
theorem combined_weight (tracy_weight : ℕ) (jake_weight : ℕ) (john_weight : ℕ)
  (h1 : tracy_weight = 52)
  (h2 : jake_weight = tracy_weight + 8)
  (h3 : jake_weight - john_weight = 14 ∨ tracy_weight - john_weight = 14) :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

#check combined_weight

end NUMINAMATH_CALUDE_combined_weight_l3039_303999


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3039_303924

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 2*x + 5)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3039_303924


namespace NUMINAMATH_CALUDE_complex_abs_value_l3039_303921

theorem complex_abs_value : Complex.abs (-3 - (8/5)*Complex.I) = 17/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_value_l3039_303921


namespace NUMINAMATH_CALUDE_ball_max_height_l3039_303982

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 - 40 * t + 50

-- State the theorem
theorem ball_max_height :
  ∃ (max : ℝ), max = 70 ∧ ∀ (t : ℝ), h t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3039_303982


namespace NUMINAMATH_CALUDE_rhombohedron_volume_l3039_303960

/-- The volume of a rhombohedron formed by extruding a rhombus -/
theorem rhombohedron_volume
  (d1 : ℝ) (d2 : ℝ) (h : ℝ)
  (hd1 : d1 = 25)
  (hd2 : d2 = 50)
  (hh : h = 20) :
  (d1 * d2 / 2) * h = 12500 := by
  sorry

end NUMINAMATH_CALUDE_rhombohedron_volume_l3039_303960


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_equal_l3039_303981

theorem quadratic_roots_sum_squares_equal (a : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 + 2*x₁ + a = 0 ∧
    x₂^2 + 2*x₂ + a = 0 ∧
    y₁^2 + a*y₁ + 2 = 0 ∧
    y₂^2 + a*y₂ + 2 = 0 ∧
    x₁^2 + x₂^2 = y₁^2 + y₂^2) →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_equal_l3039_303981


namespace NUMINAMATH_CALUDE_solution_set_for_a_neg_one_range_of_a_l3039_303932

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |3*x - 1|

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x ≤ |3*x + 1|}

-- Statement for part 1
theorem solution_set_for_a_neg_one :
  {x : ℝ | f (-1) x ≤ 1} = {x : ℝ | 1/4 ≤ x ∧ x ≤ 1/2} :=
sorry

-- Statement for part 2
theorem range_of_a (a : ℝ) :
  (Set.Icc (1/4 : ℝ) 1 ⊆ M a) → -7/3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_neg_one_range_of_a_l3039_303932


namespace NUMINAMATH_CALUDE_unique_root_sum_l3039_303964

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Theorem statement
theorem unique_root_sum (a b : ℤ) : 
  (∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) →  -- Exactly one root in (a, b)
  (b - a = 1) →                          -- b - a = 1
  (a + b = -3) :=                        -- Conclusion: a + b = -3
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_unique_root_sum_l3039_303964


namespace NUMINAMATH_CALUDE_chipped_marbles_count_l3039_303939

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [20, 22, 25, 30, 32, 34, 36]

/-- Represents the number of bags Jane takes -/
def jane_bags : Nat := 3

/-- Represents the number of bags George takes -/
def george_bags : Nat := 3

/-- The number of chipped marbles -/
def chipped_marbles : Nat := 22

theorem chipped_marbles_count :
  ∃ (jane_selection george_selection : List Nat),
    jane_selection.length = jane_bags ∧
    george_selection.length = george_bags ∧
    (∀ x, x ∈ jane_selection ∨ x ∈ george_selection → x ∈ bags) ∧
    (∀ x, x ∈ jane_selection → x ∉ george_selection) ∧
    (∀ x, x ∈ george_selection → x ∉ jane_selection) ∧
    (∃ remaining, remaining ∈ bags ∧
      remaining ∉ jane_selection ∧
      remaining ∉ george_selection ∧
      remaining = chipped_marbles ∧
      (jane_selection.sum + george_selection.sum = 3 * remaining)) :=
sorry

end NUMINAMATH_CALUDE_chipped_marbles_count_l3039_303939


namespace NUMINAMATH_CALUDE_difference_of_squares_65_35_l3039_303923

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_65_35_l3039_303923


namespace NUMINAMATH_CALUDE_number_of_roses_l3039_303918

theorem number_of_roses (total : ℕ) (rose_lily_diff : ℕ) (tulip_rose_diff : ℕ)
  (h1 : total = 100)
  (h2 : rose_lily_diff = 22)
  (h3 : tulip_rose_diff = 20) :
  ∃ (roses lilies tulips : ℕ),
    roses + lilies + tulips = total ∧
    roses = lilies + rose_lily_diff ∧
    tulips = roses + tulip_rose_diff ∧
    roses = 34 := by
  sorry

end NUMINAMATH_CALUDE_number_of_roses_l3039_303918


namespace NUMINAMATH_CALUDE_unique_solution_l3039_303970

/-- The vector [2, -3] -/
def v : Fin 2 → ℝ := ![2, -3]

/-- The vector [4, 7] -/
def w : Fin 2 → ℝ := ![4, 7]

/-- The equation to be solved -/
def equation (k : ℝ) : Prop :=
  ‖k • v - w‖ = 2 * Real.sqrt 13

/-- Theorem stating that k = -1 is the only solution -/
theorem unique_solution :
  ∃! k : ℝ, equation k ∧ k = -1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3039_303970


namespace NUMINAMATH_CALUDE_farrah_match_sticks_l3039_303949

/-- Calculates the total number of match sticks ordered given the number of boxes, 
    matchboxes per box, and sticks per matchbox. -/
def total_match_sticks (boxes : ℕ) (matchboxes_per_box : ℕ) (sticks_per_matchbox : ℕ) : ℕ :=
  boxes * matchboxes_per_box * sticks_per_matchbox

/-- Proves that the total number of match sticks ordered by Farrah is 122,500. -/
theorem farrah_match_sticks : 
  total_match_sticks 7 35 500 = 122500 := by
  sorry

end NUMINAMATH_CALUDE_farrah_match_sticks_l3039_303949


namespace NUMINAMATH_CALUDE_juggling_improvement_l3039_303902

/-- 
Given:
- start_objects: The number of objects Jeanette starts juggling with
- weeks: The number of weeks Jeanette practices
- end_objects: The number of objects Jeanette can juggle at the end
- weekly_improvement: The number of additional objects Jeanette can juggle each week

Prove that with the given conditions, the weekly improvement is 2.
-/
theorem juggling_improvement 
  (start_objects : ℕ) 
  (weeks : ℕ) 
  (end_objects : ℕ) 
  (weekly_improvement : ℕ) 
  (h1 : start_objects = 3)
  (h2 : weeks = 5)
  (h3 : end_objects = 13)
  (h4 : end_objects = start_objects + weeks * weekly_improvement) : 
  weekly_improvement = 2 := by
  sorry


end NUMINAMATH_CALUDE_juggling_improvement_l3039_303902


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l3039_303934

/-- A parabola y = ax² + bx + 7 is tangent to the line y = 2x + 3 if and only if b = 2 ± 4√a -/
theorem parabola_tangent_to_line (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x^2 + b * x + 7 ∧ y = 2 * x + 3 ∧
    ∀ x' : ℝ, a * x'^2 + b * x' + 7 ≥ 2 * x' + 3) ↔
  (b = 2 + 4 * Real.sqrt a ∨ b = 2 - 4 * Real.sqrt a) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l3039_303934


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l3039_303925

theorem quadratic_function_sum (a b : ℝ) : 
  a > 0 → 
  (∀ x ∈ Set.Icc 2 3, (a * x^2 - 2 * a * x + 1 + b) ≤ 4) →
  (∀ x ∈ Set.Icc 2 3, (a * x^2 - 2 * a * x + 1 + b) ≥ 1) →
  (∃ x ∈ Set.Icc 2 3, a * x^2 - 2 * a * x + 1 + b = 4) →
  (∃ x ∈ Set.Icc 2 3, a * x^2 - 2 * a * x + 1 + b = 1) →
  a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_l3039_303925


namespace NUMINAMATH_CALUDE_quadrilateral_vector_sum_l3039_303972

/-- Given a quadrilateral ABCD in a real inner product space, with M as the intersection of its diagonals,
    prove that for any point O not equal to M, the sum of vectors from O to each vertex
    is equal to four times the vector from O to M. -/
theorem quadrilateral_vector_sum (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (A B C D M O : V) : 
  M ≠ O →  -- O is not equal to M
  (A - C) = (D - B) →  -- M is the midpoint of AC
  (B - D) = (C - A) →  -- M is the midpoint of BD
  (O - A) + (O - B) + (O - C) + (O - D) = 4 • (O - M) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_vector_sum_l3039_303972


namespace NUMINAMATH_CALUDE_simplify_expression_l3039_303985

theorem simplify_expression (x y : ℝ) (m : ℤ) :
  (x + y) ^ (2 * m + 1) / (x + y) ^ (m - 1) = (x + y) ^ (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3039_303985


namespace NUMINAMATH_CALUDE_lizas_account_balance_l3039_303911

/-- Calculates the final balance in Liza's account after all transactions -/
def final_balance (initial_balance rent paycheck electricity internet phone : ℤ) : ℤ :=
  initial_balance - rent + paycheck - electricity - internet - phone

/-- Theorem stating that Liza's final account balance is correct -/
theorem lizas_account_balance :
  final_balance 800 450 1500 117 100 70 = 1563 := by
  sorry

end NUMINAMATH_CALUDE_lizas_account_balance_l3039_303911


namespace NUMINAMATH_CALUDE_brittany_vacation_duration_l3039_303901

/-- The duration of Brittany's vacation --/
def vacation_duration (rebecca_age : ℕ) (age_difference : ℕ) (brittany_age_after : ℕ) : ℕ :=
  brittany_age_after - (rebecca_age + age_difference)

/-- Theorem stating that Brittany's vacation lasted 4 years --/
theorem brittany_vacation_duration :
  vacation_duration 25 3 32 = 4 := by
  sorry

end NUMINAMATH_CALUDE_brittany_vacation_duration_l3039_303901


namespace NUMINAMATH_CALUDE_food_drive_cans_l3039_303952

theorem food_drive_cans (rachel jaydon mark : ℕ) : 
  jaydon = 2 * rachel + 5 →
  mark = 4 * jaydon →
  rachel + jaydon + mark = 135 →
  mark = 100 := by
sorry

end NUMINAMATH_CALUDE_food_drive_cans_l3039_303952


namespace NUMINAMATH_CALUDE_deepak_age_l3039_303983

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) 
  (years_ahead : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 3 →
  rahul_future_age = 38 →
  years_ahead = 6 →
  ∃ (x : ℕ), rahul_ratio * x + years_ahead = rahul_future_age ∧ 
             deepak_ratio * x = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l3039_303983


namespace NUMINAMATH_CALUDE_zilla_savings_l3039_303904

/-- Given Zilla's monthly earnings and spending habits, calculate her savings --/
theorem zilla_savings (E : ℝ) (h1 : E * 0.07 = 133) (h2 : E > 0) : E - (E * 0.07 + E * 0.5) = 817 := by
  sorry

end NUMINAMATH_CALUDE_zilla_savings_l3039_303904


namespace NUMINAMATH_CALUDE_equal_fish_count_l3039_303998

def herring_fat : ℕ := 40
def eel_fat : ℕ := 20
def pike_fat : ℕ := eel_fat + 10
def total_fat : ℕ := 3600

theorem equal_fish_count (x : ℕ) 
  (h : x * herring_fat + x * eel_fat + x * pike_fat = total_fat) : 
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_fish_count_l3039_303998


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3039_303950

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  (2 * y - 4 * x) / 3 + 2 * x = 4 ∧ y - 2 * x + 3 = 6

-- Theorem for the first system
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 4 ∧ y = 3 := by
  sorry

-- Theorem for the second system
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 1 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3039_303950


namespace NUMINAMATH_CALUDE_equation_condition_l3039_303980

/-- The floor function (greatest integer function) -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The condition for the equation to have no solutions except x = y -/
def no_other_solutions (A B : ℝ) : Prop :=
  ∀ x y : ℝ, A * x + B * (floor x) = A * y + B * (floor y) → x = y

/-- The theorem stating the necessary and sufficient condition -/
theorem equation_condition (A B : ℝ) :
  no_other_solutions A B ↔ (|A + B| ≥ |A| ∧ |A| > 0) :=
sorry

end NUMINAMATH_CALUDE_equation_condition_l3039_303980


namespace NUMINAMATH_CALUDE_x_over_y_is_negative_two_l3039_303993

theorem x_over_y_is_negative_two (x y : ℝ) 
  (h1 : 1 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 4)
  (h2 : (x + y) / (x - y) ≠ 1)
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_is_negative_two_l3039_303993


namespace NUMINAMATH_CALUDE_three_digit_powers_of_three_l3039_303974

theorem three_digit_powers_of_three :
  (∃! (s : Finset ℕ), s = {n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999} ∧ Finset.card s = 2) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_three_l3039_303974


namespace NUMINAMATH_CALUDE_intersection_coordinates_l3039_303930

/-- A line perpendicular to the x-axis passing through a point -/
structure VerticalLine where
  x : ℝ

/-- A line perpendicular to the y-axis passing through a point -/
structure HorizontalLine where
  y : ℝ

/-- The intersection point of a vertical and a horizontal line -/
def intersectionPoint (v : VerticalLine) (h : HorizontalLine) : ℝ × ℝ :=
  (v.x, h.y)

/-- Theorem: The intersection of the specific vertical and horizontal lines -/
theorem intersection_coordinates :
  let v := VerticalLine.mk (-3)
  let h := HorizontalLine.mk (-3)
  intersectionPoint v h = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_coordinates_l3039_303930


namespace NUMINAMATH_CALUDE_investment_difference_l3039_303989

def compound_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

def emma_investment (initial : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest initial rate1) rate2) rate3

def briana_investment (initial : ℝ) (rate1 rate2 rate3 : ℝ) : ℝ :=
  compound_interest (compound_interest (compound_interest initial rate1) rate2) rate3

theorem investment_difference :
  let emma_initial := 300
  let briana_initial := 500
  let emma_rate1 := 0.15
  let emma_rate2 := 0.12
  let emma_rate3 := 0.18
  let briana_rate1 := 0.10
  let briana_rate2 := 0.08
  let briana_rate3 := 0.14
  briana_investment briana_initial briana_rate1 briana_rate2 briana_rate3 -
  emma_investment emma_initial emma_rate1 emma_rate2 emma_rate3 = 220.808 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l3039_303989


namespace NUMINAMATH_CALUDE_largest_circle_radius_on_chessboard_l3039_303906

/-- Represents a chessboard with the usual coloring of fields -/
structure Chessboard where
  size : Nat
  is_black : Nat → Nat → Bool

/-- Represents a circle on the chessboard -/
structure Circle where
  center : Nat × Nat
  radius : Real

/-- Check if a circle intersects a white field on the chessboard -/
def intersects_white (board : Chessboard) (circle : Circle) : Bool :=
  sorry

/-- The largest possible circle radius on a chessboard without intersecting white fields -/
def largest_circle_radius (board : Chessboard) : Real :=
  sorry

/-- Theorem stating the largest possible circle radius on a standard chessboard -/
theorem largest_circle_radius_on_chessboard :
  ∀ (board : Chessboard),
    board.size = 8 →
    (∀ i j, board.is_black i j = ((i + j) % 2 = 1)) →
    largest_circle_radius board = (1 / 2) * Real.sqrt 10 :=
  sorry

end NUMINAMATH_CALUDE_largest_circle_radius_on_chessboard_l3039_303906


namespace NUMINAMATH_CALUDE_arccos_cos_nine_l3039_303976

theorem arccos_cos_nine :
  Real.arccos (Real.cos 9) = 9 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_nine_l3039_303976


namespace NUMINAMATH_CALUDE_average_difference_l3039_303928

def num_students : ℕ := 120
def num_teachers : ℕ := 6
def class_sizes : List ℕ := [40, 35, 25, 10, 5, 5]

def t : ℚ := (List.sum class_sizes) / num_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes)) / num_students

theorem average_difference : t - s = -10 := by sorry

end NUMINAMATH_CALUDE_average_difference_l3039_303928


namespace NUMINAMATH_CALUDE_polygon_sides_l3039_303957

theorem polygon_sides (sum_interior_angles : ℕ) : sum_interior_angles = 1440 → ∃ n : ℕ, n = 10 ∧ (n - 2) * 180 = sum_interior_angles :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l3039_303957


namespace NUMINAMATH_CALUDE_system_solution_l3039_303959

theorem system_solution : ∃ (x y : ℝ), 2 * x + y = 4 ∧ 3 * x - 2 * y = 13 := by
  use 3, -2
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3039_303959


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l3039_303996

theorem triangle_side_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) = b / (a + c) + c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l3039_303996


namespace NUMINAMATH_CALUDE_blue_balls_in_jar_l3039_303967

theorem blue_balls_in_jar (total_balls : ℕ) (removed_blue : ℕ) (prob_after : ℚ) : 
  total_balls = 25 → 
  removed_blue = 5 → 
  prob_after = 1/5 → 
  ∃ initial_blue : ℕ, 
    initial_blue = 9 ∧ 
    (initial_blue - removed_blue : ℚ) / (total_balls - removed_blue : ℚ) = prob_after :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_in_jar_l3039_303967


namespace NUMINAMATH_CALUDE_cosine_range_theorem_l3039_303914

theorem cosine_range_theorem (f : ℝ → ℝ) (x : ℝ) :
  (f = λ x => Real.cos (x - π/3)) →
  (x ∈ Set.Icc 0 (π/2)) →
  (∀ y, y ∈ Set.range f ↔ y ∈ Set.Icc (1/2) 1) :=
sorry

end NUMINAMATH_CALUDE_cosine_range_theorem_l3039_303914


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l3039_303992

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x < 10}) :=
sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  C a ⊆ B → a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l3039_303992


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3039_303977

theorem smallest_max_sum (a b c d e f g : ℕ+) 
  (sum_eq : a + b + c + d + e + f + g = 2024) : 
  (∃ M : ℕ, 
    (M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g)))))) ∧ 
    (∀ M' : ℕ, 
      (M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g)))))) → 
      M ≤ M') ∧
    M = 338) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3039_303977


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3039_303975

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a = 0 → a * b = 0) ∧
  (∃ a b, a * b = 0 ∧ a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3039_303975


namespace NUMINAMATH_CALUDE_passing_percentage_l3039_303958

def max_marks : ℕ := 300
def obtained_marks : ℕ := 160
def failed_by : ℕ := 20

theorem passing_percentage :
  (((obtained_marks + failed_by : ℚ) / max_marks) * 100 : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l3039_303958


namespace NUMINAMATH_CALUDE_no_integer_solution_l3039_303938

theorem no_integer_solution : ¬ ∃ (n : ℤ), (n + 15 > 18) ∧ (-3*n > -9) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3039_303938


namespace NUMINAMATH_CALUDE_terms_before_five_l3039_303986

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_five (a₁ : ℤ) (d : ℤ) :
  a₁ = 105 ∧ d = -5 →
  ∃ n : ℕ, 
    arithmetic_sequence a₁ d n = 5 ∧ 
    (∀ k : ℕ, k < n → arithmetic_sequence a₁ d k > 5) ∧
    n - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_terms_before_five_l3039_303986


namespace NUMINAMATH_CALUDE_right_triangle_cosine_sine_l3039_303973

-- Define the right triangle XYZ
def RightTriangleXYZ (X Y Z : ℝ) : Prop :=
  X^2 + Y^2 = Z^2 ∧ X = 8 ∧ Z = 17

-- Theorem statement
theorem right_triangle_cosine_sine 
  (X Y Z : ℝ) (h : RightTriangleXYZ X Y Z) : 
  Real.cos (Real.arccos (X / Z)) = 15 / 17 ∧ Real.sin (Real.arcsin 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_sine_l3039_303973


namespace NUMINAMATH_CALUDE_divisible_by_120_l3039_303947

theorem divisible_by_120 (n : ℕ) : ∃ k : ℤ, n * (n^2 - 1) * (n^2 - 5*n + 26) = 120 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l3039_303947


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l3039_303910

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f(x) = log_a(x+1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + 2

-- Theorem statement
theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l3039_303910


namespace NUMINAMATH_CALUDE_unit_digit_of_3_power_2023_l3039_303929

def unit_digit_pattern : List Nat := [3, 9, 7, 1]

theorem unit_digit_of_3_power_2023 :
  (3^2023 : ℕ) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_3_power_2023_l3039_303929


namespace NUMINAMATH_CALUDE_airplane_seats_l3039_303969

theorem airplane_seats : 
  ∀ (total : ℕ),
  (24 : ℕ) + (total / 4 : ℕ) + (2 * total / 3 : ℕ) = total →
  total = 288 := by
sorry

end NUMINAMATH_CALUDE_airplane_seats_l3039_303969


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3039_303922

theorem unique_solution_sqrt_equation :
  ∃! (x : ℝ), 2 * x + Real.sqrt (x - 3) = 7 :=
by
  -- The unique solution is x = 3.25
  use 3.25
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3039_303922


namespace NUMINAMATH_CALUDE_power_multiplication_l3039_303979

theorem power_multiplication (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3039_303979


namespace NUMINAMATH_CALUDE_integer_solution_inequalities_l3039_303920

theorem integer_solution_inequalities :
  ∀ x : ℤ, (x + 7 > 5 ∧ -3*x > -9) ↔ x ∈ ({-1, 0, 1, 2} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_inequalities_l3039_303920


namespace NUMINAMATH_CALUDE_queen_high_school_teachers_queen_high_school_teachers_correct_l3039_303953

theorem queen_high_school_teachers (num_students : ℕ) (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  let total_classes := num_students * classes_per_student
  let unique_classes := total_classes / students_per_class
  unique_classes / classes_per_teacher

theorem queen_high_school_teachers_correct : 
  queen_high_school_teachers 1500 6 5 25 = 72 := by
  sorry

end NUMINAMATH_CALUDE_queen_high_school_teachers_queen_high_school_teachers_correct_l3039_303953


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3039_303912

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.cos (α + π / 4) = -3 / 5) : Real.tan α = 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3039_303912


namespace NUMINAMATH_CALUDE_basketball_substitutions_remainder_l3039_303915

/-- Represents the number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players starters max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem about the basketball substitutions problem -/
theorem basketball_substitutions_remainder :
  substitution_ways 15 5 4 % 1000 = 301 := by sorry

end NUMINAMATH_CALUDE_basketball_substitutions_remainder_l3039_303915


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l3039_303942

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (boat_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  boat_speed + stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : boat_speed = 11)
  (h2 : boat_distance boat_speed (downstream_distance - boat_speed) = 13) :
  boat_distance boat_speed (boat_speed - downstream_distance) = 9 := by
sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l3039_303942


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3039_303903

/-- The function f(x) = a^(x-2) + 1 passes through the point (2, 2) when a > 0 and a ≠ 1 -/
theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3039_303903


namespace NUMINAMATH_CALUDE_book_pages_digits_l3039_303941

/-- Given a book with n pages, calculate the total number of digits used to number all pages. -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigits := min n 9
  let doubleDigits := max (min n 99 - 9) 0
  let tripleDigits := max (n - 99) 0
  singleDigits + 2 * doubleDigits + 3 * tripleDigits

/-- Theorem stating that a book with 360 pages requires exactly 972 digits to number all its pages. -/
theorem book_pages_digits : totalDigits 360 = 972 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_digits_l3039_303941


namespace NUMINAMATH_CALUDE_photos_framed_by_jack_or_taken_by_octavia_or_sam_l3039_303943

/-- Represents a photographer in the exhibition -/
inductive Photographer
| Octavia
| Sam
| Alice
| Max

/-- Represents a framer in the exhibition -/
inductive Framer
| Jack
| Jane

/-- The number of photographs framed by each framer for each photographer -/
def framed_photos (f : Framer) (p : Photographer) : ℕ :=
  match f, p with
  | Framer.Jack, Photographer.Octavia => 24
  | Framer.Jack, Photographer.Sam => 12
  | Framer.Jack, Photographer.Alice => 8
  | Framer.Jack, Photographer.Max => 0
  | Framer.Jane, Photographer.Octavia => 0
  | Framer.Jane, Photographer.Sam => 10
  | Framer.Jane, Photographer.Alice => 6
  | Framer.Jane, Photographer.Max => 18

/-- The total number of photographs taken by each photographer -/
def total_photos (p : Photographer) : ℕ :=
  match p with
  | Photographer.Octavia => 36
  | Photographer.Sam => 20
  | Photographer.Alice => 14
  | Photographer.Max => 32

/-- Theorem stating the number of photographs either framed by Jack or taken by Octavia or Sam -/
theorem photos_framed_by_jack_or_taken_by_octavia_or_sam :
  (framed_photos Framer.Jack Photographer.Octavia +
   framed_photos Framer.Jack Photographer.Sam +
   framed_photos Framer.Jack Photographer.Alice) +
  (total_photos Photographer.Octavia +
   total_photos Photographer.Sam) = 100 := by
  sorry

end NUMINAMATH_CALUDE_photos_framed_by_jack_or_taken_by_octavia_or_sam_l3039_303943


namespace NUMINAMATH_CALUDE_exists_parallel_line_l3039_303909

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Prop)
variable (not_perpendicular : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem exists_parallel_line 
  (α β γ : Plane)
  (h1 : perpendicular β γ)
  (h2 : intersects α γ)
  (h3 : not_perpendicular α γ) :
  ∃ (a : Line), in_plane a α ∧ parallel a γ :=
sorry

end NUMINAMATH_CALUDE_exists_parallel_line_l3039_303909


namespace NUMINAMATH_CALUDE_fraction_product_proof_l3039_303995

theorem fraction_product_proof : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_proof_l3039_303995


namespace NUMINAMATH_CALUDE_general_term_formula_correct_l3039_303963

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- General term formula -/
def generalTerm (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem general_term_formula_correct :
  ∀ n : ℕ, n > 0 → arithmeticSequence n = generalTerm n := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_correct_l3039_303963


namespace NUMINAMATH_CALUDE_ball_picking_probabilities_l3039_303951

/-- The probability of picking ball 3 using method one -/
def P₁ : ℚ := 1/3

/-- The probability of picking ball 3 using method two -/
def P₂ : ℚ := 1/2

/-- The probability of picking ball 3 using method three -/
def P₃ : ℚ := 2/3

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem ball_picking_probabilities :
  (P₁ < P₂) ∧ (P₁ < P₃) ∧ (P₂ ≠ P₃) ∧ (2 * P₁ = P₃) := by
  sorry

end NUMINAMATH_CALUDE_ball_picking_probabilities_l3039_303951


namespace NUMINAMATH_CALUDE_root_implies_m_values_l3039_303927

theorem root_implies_m_values (m : ℝ) : 
  ((m + 2) * 1^2 - 2 * 1 + m^2 - 2 * m - 6 = 0) → (m = -2 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_root_implies_m_values_l3039_303927


namespace NUMINAMATH_CALUDE_time_after_elapsed_minutes_l3039_303917

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2015, month := 3, day := 3, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def elapsedMinutes : Nat := 4350

/-- The expected result DateTime -/
def expectedResult : DateTime :=
  { year := 2015, month := 3, day := 6, hour := 0, minute := 30 }

theorem time_after_elapsed_minutes :
  addMinutes startTime elapsedMinutes = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_time_after_elapsed_minutes_l3039_303917


namespace NUMINAMATH_CALUDE_range_of_m_l3039_303931

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 1/x + 4/y = 1) (h2 : ∃ m : ℝ, x + y < m^2 - 8*m) : 
  ∃ m : ℝ, (m < -1 ∨ m > 9) ∧ x + y < m^2 - 8*m := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3039_303931


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposing_l3039_303965

/-- A bag containing two red balls and two black balls -/
structure Bag :=
  (red_balls : ℕ := 2)
  (black_balls : ℕ := 2)

/-- The event of drawing exactly one black ball -/
def exactly_one_black (bag : Bag) : Set (Fin 2 → Bool) :=
  {draw | (draw 0 = true ∧ draw 1 = false) ∨ (draw 0 = false ∧ draw 1 = true)}

/-- The event of drawing exactly two black balls -/
def exactly_two_black (bag : Bag) : Set (Fin 2 → Bool) :=
  {draw | draw 0 = true ∧ draw 1 = true}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutually_exclusive (E F : Set (Fin 2 → Bool)) : Prop :=
  E ∩ F = ∅

/-- Two events are opposing if their union is the entire sample space -/
def opposing (E F : Set (Fin 2 → Bool)) : Prop :=
  E ∪ F = Set.univ

theorem mutually_exclusive_not_opposing (bag : Bag) :
  mutually_exclusive (exactly_one_black bag) (exactly_two_black bag) ∧
  ¬opposing (exactly_one_black bag) (exactly_two_black bag) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposing_l3039_303965


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3039_303987

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3039_303987


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l3039_303966

theorem cubic_roots_sum_of_squares_reciprocal (a b c : ℝ) 
  (sum_eq : a + b + c = 12)
  (sum_prod_eq : a * b + b * c + c * a = 20)
  (prod_eq : a * b * c = -5) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 20.8 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_reciprocal_l3039_303966


namespace NUMINAMATH_CALUDE_expression_evaluation_l3039_303916

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = 1) :
  2 * (x - y) - 3 * (2 * x - y) + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3039_303916


namespace NUMINAMATH_CALUDE_vector_parallel_proof_l3039_303945

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, -1]
def c (m n : ℝ) : Fin 2 → ℝ := ![m - 2, -n]

theorem vector_parallel_proof (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_parallel : ∃ (k : ℝ), ∀ i, (a - b) i = k * c m n i) :
  (2 * m + n = 4) ∧ (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 4 → x * y ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_proof_l3039_303945


namespace NUMINAMATH_CALUDE_square_perimeter_l3039_303933

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 468 → perimeter = 24 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3039_303933


namespace NUMINAMATH_CALUDE_sues_mix_nuts_percent_l3039_303944

/-- Sue's trail mix composition -/
structure SuesMix where
  nuts : ℝ
  dried_fruit : ℝ
  dried_fruit_percent : dried_fruit = 70
  sum_to_100 : nuts + dried_fruit = 100

/-- Jane's trail mix composition -/
structure JanesMix where
  nuts : ℝ
  chocolate_chips : ℝ
  nuts_percent : nuts = 60
  chocolate_chips_percent : chocolate_chips = 40
  sum_to_100 : nuts + chocolate_chips = 100

/-- Combined mixture composition -/
structure CombinedMix where
  nuts : ℝ
  dried_fruit : ℝ
  nuts_percent : nuts = 45
  dried_fruit_percent : dried_fruit = 35

/-- Theorem stating that Sue's trail mix contains 30% nuts -/
theorem sues_mix_nuts_percent (s : SuesMix) (j : JanesMix) (c : CombinedMix) : s.nuts = 30 :=
sorry

end NUMINAMATH_CALUDE_sues_mix_nuts_percent_l3039_303944


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3039_303935

theorem unique_prime_solution : ∃! (p : ℕ), 
  Prime p ∧ 
  ∃ (x y : ℕ), 
    x > 0 ∧ 
    y > 0 ∧ 
    p + 49 = 2 * x^2 ∧ 
    p^2 + 49 = 2 * y^2 ∧ 
    p = 23 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3039_303935


namespace NUMINAMATH_CALUDE_physics_class_grades_l3039_303968

theorem physics_class_grades (total_students : ℕ) (prob_A prob_B prob_C : ℚ) :
  total_students = 42 →
  prob_A = 2 * prob_B →
  prob_C = 1.2 * prob_B →
  prob_A + prob_B + prob_C = 1 →
  (prob_B * total_students : ℚ) = 10 :=
by sorry

end NUMINAMATH_CALUDE_physics_class_grades_l3039_303968


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l3039_303900

/-- Calculates Tim's total weekly earnings including bonuses -/
def timWeeklyEarnings (tasksPerDay : ℕ) (workDaysPerWeek : ℕ) 
  (tasksPay1 tasksPay2 tasksPay3 : ℕ) (rate1 rate2 rate3 : ℚ) 
  (bonusThreshold : ℕ) (bonusAmount : ℚ) 
  (performanceBonusThreshold : ℕ) (performanceBonusAmount : ℚ) : ℚ :=
  let dailyEarnings := tasksPay1 * rate1 + tasksPay2 * rate2 + tasksPay3 * rate3
  let dailyBonuses := (tasksPerDay / bonusThreshold) * bonusAmount
  let weeklyEarnings := (dailyEarnings + dailyBonuses) * workDaysPerWeek
  let performanceBonus := if tasksPerDay ≥ performanceBonusThreshold then performanceBonusAmount else 0
  weeklyEarnings + performanceBonus

/-- Tim's total weekly earnings are $1058 -/
theorem tim_weekly_earnings :
  timWeeklyEarnings 100 6 40 30 30 (6/5) (3/2) 2 50 10 90 20 = 1058 := by
  sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l3039_303900


namespace NUMINAMATH_CALUDE_function_satisfying_condition_l3039_303948

theorem function_satisfying_condition (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = 1 + x * y + f (x + y)) →
  ((∀ x : ℝ, f x = 2 * x - 1) ∨ (∀ x : ℝ, f x = x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_l3039_303948


namespace NUMINAMATH_CALUDE_slope_of_l₃_is_two_l3039_303956

-- Define the points and lines
def A : ℝ × ℝ := (2, 0)
def l₁ : ℝ → ℝ := λ x => 4 - 2*x
def l₂ : ℝ → ℝ := λ _ => 2

-- Define the properties of the lines and points
axiom l₁_through_A : l₁ A.1 = A.2
axiom l₂_intersects_l₁ : ∃ B : ℝ × ℝ, l₁ B.1 = B.2 ∧ l₂ B.1 = B.2

-- Define the properties of l₃
axiom l₃_positive_slope : ∃ m : ℝ, m > 0 ∧ ∃ b : ℝ, ∀ x : ℝ, (m * x + b - A.2) / (x - A.1) = m
axiom l₃_through_A : ∃ m : ℝ, m > 0 ∧ ∃ b : ℝ, m * A.1 + b = A.2
axiom l₃_intersects_l₂ : ∃ C : ℝ × ℝ, ∃ m : ℝ, m > 0 ∧ ∃ b : ℝ, m * C.1 + b = C.2 ∧ l₂ C.1 = C.2

-- Define the area of triangle ABC
axiom triangle_area : ∃ B C : ℝ × ℝ, 
  l₁ B.1 = B.2 ∧ l₂ B.1 = B.2 ∧ 
  (∃ m : ℝ, m > 0 ∧ ∃ b : ℝ, m * C.1 + b = C.2) ∧ l₂ C.1 = C.2 ∧
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 2

-- Theorem to prove
theorem slope_of_l₃_is_two : 
  ∃ m : ℝ, m = 2 ∧ 
  (∀ x : ℝ, (m * x + (A.2 - m * A.1) - A.2) / (x - A.1) = m) ∧
  (m * A.1 + (A.2 - m * A.1) = A.2) ∧
  (∃ C : ℝ × ℝ, m * C.1 + (A.2 - m * A.1) = C.2 ∧ l₂ C.1 = C.2) :=
sorry

end NUMINAMATH_CALUDE_slope_of_l₃_is_two_l3039_303956


namespace NUMINAMATH_CALUDE_spinner_probability_l3039_303990

-- Define the spinner regions
inductive Region
| A
| B1
| B2
| C

-- Define the probability function
def P : Region → ℚ
| Region.A  => 3/8
| Region.B1 => 1/8
| Region.B2 => 1/4
| Region.C  => 1/4  -- This is what we want to prove

-- State the theorem
theorem spinner_probability :
  P Region.C = 1/4 :=
by
  sorry

-- Additional lemmas to help with the proof
lemma total_probability :
  P Region.A + P Region.B1 + P Region.B2 + P Region.C = 1 :=
by
  sorry

lemma b_subregions :
  P Region.B1 + P Region.B2 = 3/8 :=
by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3039_303990
