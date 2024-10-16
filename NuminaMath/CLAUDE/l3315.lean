import Mathlib

namespace NUMINAMATH_CALUDE_complex_circle_extrema_l3315_331517

theorem complex_circle_extrema (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 1) :
  (∃ w : ℂ, Complex.abs (w - (1 + 2*I)) = 1 ∧ Complex.abs (w - (3 + I)) = Real.sqrt 5 + 1) ∧
  (∃ v : ℂ, Complex.abs (v - (1 + 2*I)) = 1 ∧ Complex.abs (v - (3 + I)) = Real.sqrt 5 - 1) ∧
  (∀ u : ℂ, Complex.abs (u - (1 + 2*I)) = 1 →
    Real.sqrt 5 - 1 ≤ Complex.abs (u - (3 + I)) ∧ Complex.abs (u - (3 + I)) ≤ Real.sqrt 5 + 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_circle_extrema_l3315_331517


namespace NUMINAMATH_CALUDE_equation_solution_l3315_331579

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (21 / (x₁^2 - 9) - 3 / (x₁ - 3) = 2) ∧ 
                 (21 / (x₂^2 - 9) - 3 / (x₂ - 3) = 2) ∧ 
                 (abs (x₁ - 4.695) < 0.001) ∧ 
                 (abs (x₂ + 3.195) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3315_331579


namespace NUMINAMATH_CALUDE_marble_count_theorem_l3315_331557

/-- Represents the total number of marbles in a bag given the ratio of colors and the number of green marbles -/
def total_marbles (red blue green yellow : ℕ) (green_count : ℕ) : ℕ :=
  (red + blue + green + yellow) * green_count / green

/-- Theorem stating that given the specific ratio and number of green marbles, the total is 120 -/
theorem marble_count_theorem :
  total_marbles 1 3 2 4 24 = 120 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_theorem_l3315_331557


namespace NUMINAMATH_CALUDE_inequality_relations_l3315_331500

theorem inequality_relations :
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) ∧
  (∀ a b : ℝ, a^2 > b^2 → |a| > |b|) ∧
  (∀ a b c : ℝ, a > b ↔ a + c > b + c) :=
sorry

end NUMINAMATH_CALUDE_inequality_relations_l3315_331500


namespace NUMINAMATH_CALUDE_tim_found_37_seashells_l3315_331523

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := 13

/-- The total number of seashells Tim and Sally found together -/
def total_seashells : ℕ := 50

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := total_seashells - sally_seashells

theorem tim_found_37_seashells : tim_seashells = 37 := by
  sorry

end NUMINAMATH_CALUDE_tim_found_37_seashells_l3315_331523


namespace NUMINAMATH_CALUDE_determine_set_N_l3315_331573

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define subset M
def M : Set Nat := {1, 4}

-- Define the theorem
theorem determine_set_N (N : Set Nat) 
  (h1 : N ⊆ U)
  (h2 : M ∩ N = {1})
  (h3 : N ∩ (U \ M) = {3, 5}) :
  N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_determine_set_N_l3315_331573


namespace NUMINAMATH_CALUDE_range_of_a_l3315_331503

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  (a ∈ Set.Icc (-1) 1) ∨ (a > 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3315_331503


namespace NUMINAMATH_CALUDE_inequality_sufficient_conditions_l3315_331529

theorem inequality_sufficient_conditions (a b x y : ℝ) :
  (a < 0 ∧ 0 < b → x < y) ∧
  (0 < b ∧ b < a → x < y) ∧
  ¬(b < a ∧ a < 0 → x < y) ∧
  ¬(b < 0 ∧ 0 < a → x < y) :=
by sorry

end NUMINAMATH_CALUDE_inequality_sufficient_conditions_l3315_331529


namespace NUMINAMATH_CALUDE_identify_participants_with_2k_minus_3_questions_l3315_331581

/-- Represents the type of participant -/
inductive Participant
| Chemist
| Alchemist

/-- Represents the state of the identification process -/
structure IdentificationState where
  participants : Nat
  chemists : Nat
  alchemists : Nat
  questions_asked : Nat

/-- The main theorem stating that 2k-3 questions are sufficient -/
theorem identify_participants_with_2k_minus_3_questions 
  (k : Nat) 
  (h : k > 0) 
  (more_chemists : ∃ (c a : Nat), c > a ∧ c + a = k) :
  ∃ (strategy : IdentificationState → Participant), 
    (∀ (state : IdentificationState), 
      state.participants = k → 
      state.chemists > state.alchemists → 
      state.questions_asked ≤ 2 * k - 3 → 
      (∀ p, strategy state = p → 
        (p = Participant.Chemist → state.chemists > 0) ∧ 
        (p = Participant.Alchemist → state.alchemists > 0))) :=
sorry

end NUMINAMATH_CALUDE_identify_participants_with_2k_minus_3_questions_l3315_331581


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3315_331522

theorem sphere_surface_area (r h d : ℝ) (h_positive : 0 < h) (d_positive : 0 < d) : 
  (4 / 3 * π * r^3 = π * (d / 2)^2 * h) → 
  (4 * π * r^2 = 576 * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3315_331522


namespace NUMINAMATH_CALUDE_tournament_teams_l3315_331599

-- Define the number of teams
def n : ℕ := sorry

-- Define the total number of matches
def total_matches : ℕ := 28

-- Theorem: The number of teams in the tournament is 8
theorem tournament_teams : n = 8 := by
  -- Define the formula for the number of matches in a round-robin tournament
  have round_robin_formula : total_matches = n * (n - 1) / 2 := sorry
  
  -- Prove that n = 8 satisfies the formula
  sorry


end NUMINAMATH_CALUDE_tournament_teams_l3315_331599


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3315_331549

theorem geometric_sequence_sum (a : ℝ) : 
  (a + 2*a + 4*a + 8*a = 1) →  -- Sum of first 4 terms equals 1
  (a + 2*a + 4*a + 8*a + 16*a + 32*a + 64*a + 128*a = 17) :=  -- Sum of first 8 terms equals 17
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3315_331549


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3315_331543

theorem solve_system_of_equations (x y : ℝ) 
  (eq1 : (x - y) * (x * y) = 30)
  (eq2 : (x + y) * (x * y) = 120) :
  x = 5 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3315_331543


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3315_331521

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2

/-- Line l₁: 2x + my - 7 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  2 * x + m * y - 7 = 0

/-- Line l₂: mx + 8y - 14 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  m * x + 8 * y - 14 = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (parallel_lines 2 m (-7) m 8 (-14)) → m = -4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3315_331521


namespace NUMINAMATH_CALUDE_expected_rainfall_is_19_25_l3315_331582

/-- Weather forecast for a single day -/
structure DailyForecast where
  prob_sun : ℝ
  prob_rain3 : ℝ
  prob_rain8 : ℝ
  sum_to_one : prob_sun + prob_rain3 + prob_rain8 = 1

/-- Calculate expected rainfall for a single day -/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.prob_sun * 0 + f.prob_rain3 * 3 + f.prob_rain8 * 8

/-- Weather forecast for the week -/
def weekly_forecast : DailyForecast :=
  { prob_sun := 0.3
    prob_rain3 := 0.35
    prob_rain8 := 0.35
    sum_to_one := by norm_num }

/-- Number of days in the forecast -/
def num_days : ℕ := 5

/-- Expected total rainfall for the week -/
def expected_total_rainfall : ℝ :=
  (expected_daily_rainfall weekly_forecast) * num_days

theorem expected_rainfall_is_19_25 :
  expected_total_rainfall = 19.25 := by sorry

end NUMINAMATH_CALUDE_expected_rainfall_is_19_25_l3315_331582


namespace NUMINAMATH_CALUDE_wedding_rsvp_yes_percentage_l3315_331520

theorem wedding_rsvp_yes_percentage 
  (total_guests : ℕ) 
  (no_response_percentage : ℚ) 
  (no_reply_guests : ℕ) : 
  total_guests = 200 →
  no_response_percentage = 9 / 100 →
  no_reply_guests = 16 →
  (↑(total_guests - (total_guests * no_response_percentage).floor - no_reply_guests) / total_guests : ℚ) = 83 / 100 := by
sorry

end NUMINAMATH_CALUDE_wedding_rsvp_yes_percentage_l3315_331520


namespace NUMINAMATH_CALUDE_track_circumference_l3315_331550

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (v1 v2 t : ℝ) (h1 : v1 = 4.5) (h2 : v2 = 3.75) (h3 : t = 5.28 / 60) :
  v1 * t + v2 * t = 0.726 := by
  sorry

end NUMINAMATH_CALUDE_track_circumference_l3315_331550


namespace NUMINAMATH_CALUDE_cubic_factorization_l3315_331546

theorem cubic_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3315_331546


namespace NUMINAMATH_CALUDE_carnival_spending_theorem_l3315_331509

def carnival_spending (initial_amount food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let game_cost := 2 * food_cost
  initial_amount - (food_cost + ride_cost + game_cost)

theorem carnival_spending_theorem :
  carnival_spending 80 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carnival_spending_theorem_l3315_331509


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l3315_331556

theorem simplify_trig_fraction (x : Real) :
  let u := Real.sin (x/2) * (Real.cos (x/2) + Real.sin (x/2))
  (2 - Real.sin x + Real.cos x) / (2 + Real.sin x - Real.cos x) = (3 - 2*u) / (1 + 2*u) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l3315_331556


namespace NUMINAMATH_CALUDE_trig_simplification_l3315_331542

theorem trig_simplification (α : ℝ) : 
  (2 * (Real.cos α)^2 - 1) / (2 * Real.tan (π/4 - α) * (Real.sin (π/4 + α))^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3315_331542


namespace NUMINAMATH_CALUDE_hotel_distance_l3315_331548

/-- Calculates the remaining distance to the hotel given the total distance and four driving segments. -/
def remaining_distance (total_distance : ℝ) 
  (speed1 speed2 speed3 speed4 : ℝ) 
  (time1 time2 time3 time4 : ℝ) : ℝ :=
  total_distance - (speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4)

/-- Theorem stating that the remaining distance to the hotel is 270 miles. -/
theorem hotel_distance : 
  remaining_distance 1200 60 70 50 80 2 3 4 5 = 270 := by
  sorry

end NUMINAMATH_CALUDE_hotel_distance_l3315_331548


namespace NUMINAMATH_CALUDE_rental_cost_equality_l3315_331527

/-- The daily rate for Sunshine Car Rentals in dollars -/
def sunshine_base : ℝ := 17.99

/-- The per-mile rate for Sunshine Car Rentals in dollars -/
def sunshine_per_mile : ℝ := 0.18

/-- The daily rate for City Rentals in dollars -/
def city_base : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_per_mile : ℝ := 0.16

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 48

theorem rental_cost_equality :
  sunshine_base + sunshine_per_mile * equal_cost_mileage =
  city_base + city_per_mile * equal_cost_mileage :=
by sorry

end NUMINAMATH_CALUDE_rental_cost_equality_l3315_331527


namespace NUMINAMATH_CALUDE_triangle_area_l3315_331578

/-- Given a triangle with side lengths a, b, c where:
  - a = 13
  - The angle opposite side a is 60°
  - b : c = 4 : 3
  Prove that the area of the triangle is 39√3 -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (h1 : a = 13) (h2 : A = π / 3)
    (h3 : ∃ (k : ℝ), b = 4 * k ∧ c = 3 * k) :
    (1 / 2) * b * c * Real.sin A = 39 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3315_331578


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3315_331537

theorem solution_set_inequality (x : ℝ) : 
  (2 * x + 3) * (4 - x) > 0 ↔ -3/2 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3315_331537


namespace NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_eighth_power_l3315_331577

theorem smallest_integer_larger_than_root_sum_eighth_power :
  ∃ n : ℤ, n = 1631 ∧ (∀ m : ℤ, m > (Real.sqrt 5 + Real.sqrt 3)^8 → m ≥ n) ∧
  (n - 1 : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_larger_than_root_sum_eighth_power_l3315_331577


namespace NUMINAMATH_CALUDE_scrabble_multiplier_is_three_l3315_331594

/-- Represents a three-letter word in Scrabble --/
structure ScrabbleWord where
  first_letter_value : ℕ
  middle_letter_value : ℕ
  last_letter_value : ℕ

/-- Calculates the multiplier for a given Scrabble word and final score --/
def calculate_multiplier (word : ScrabbleWord) (final_score : ℕ) : ℚ :=
  final_score / (word.first_letter_value + word.middle_letter_value + word.last_letter_value)

theorem scrabble_multiplier_is_three :
  let word : ScrabbleWord := {
    first_letter_value := 1,
    middle_letter_value := 8,
    last_letter_value := 1
  }
  let final_score : ℕ := 30
  calculate_multiplier word final_score = 3 := by
    sorry

end NUMINAMATH_CALUDE_scrabble_multiplier_is_three_l3315_331594


namespace NUMINAMATH_CALUDE_cubic_roots_sum_product_l3315_331538

theorem cubic_roots_sum_product (α β γ : ℂ) (u v w : ℂ) : 
  (∀ x : ℂ, x^3 + 5*x^2 + 7*x - 13 = (x - α) * (x - β) * (x - γ)) →
  (∀ x : ℂ, x^3 + u*x^2 + v*x + w = (x - (α + β)) * (x - (β + γ)) * (x - (γ + α))) →
  w = 48 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_product_l3315_331538


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3315_331561

theorem similar_triangle_perimeter :
  ∀ (a b c d e f : ℝ),
  a^2 + b^2 = c^2 →  -- right triangle condition
  d^2 + e^2 = f^2 →  -- right triangle condition
  a / d = b / e →    -- similarity condition
  a / d = c / f →    -- similarity condition
  (a, b, c) = (6, 8, 10) →  -- given smaller triangle
  f = 30 →           -- given longest side of larger triangle
  d + e + f = 72 :=  -- perimeter of larger triangle
by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3315_331561


namespace NUMINAMATH_CALUDE_area_triangle_BFE_l3315_331512

/-- Given a rectangle ABCD with area 48 square units and points E and F dividing sides AD and BC
    in a 2:1 ratio, the area of triangle BFE is 24 square units. -/
theorem area_triangle_BFE (A B C D E F : ℝ × ℝ) : 
  let rectangle_area := 48
  let ratio := (2 : ℝ) / 3
  (∃ u v : ℝ, 
    A = (0, 0) ∧ 
    B = (3*u, 0) ∧ 
    C = (3*u, 3*v) ∧ 
    D = (0, 3*v) ∧
    E = (0, 2*v) ∧ 
    F = (2*u, 0) ∧
    3*u*3*v = rectangle_area ∧
    (D.2 - E.2) / D.2 = ratio ∧
    (C.1 - F.1) / C.1 = ratio) →
  (1/2 * |B.1*(E.2 - F.2) + E.1*(F.2 - B.2) + F.1*(B.2 - E.2)| = 24) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_BFE_l3315_331512


namespace NUMINAMATH_CALUDE_trapezoid_parallel_line_length_l3315_331588

/-- Represents a trapezoid with bases of lengths a and b -/
structure Trapezoid (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- 
Given a trapezoid with bases of lengths a and b, 
if a line parallel to the bases divides the trapezoid into two equal-area trapezoids,
then the length of the segment of this line between the non-parallel sides 
is sqrt((a^2 + b^2)/2).
-/
theorem trapezoid_parallel_line_length 
  (a b : ℝ) (trap : Trapezoid a b) : 
  ∃ (x : ℝ), x > 0 ∧ x = Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_parallel_line_length_l3315_331588


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3315_331515

theorem polynomial_product_expansion (x : ℝ) :
  (1 + x + x^3) * (1 - x^4) = 1 + x + x^3 - x^4 - x^5 - x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l3315_331515


namespace NUMINAMATH_CALUDE_group_size_problem_l3315_331551

theorem group_size_problem (total_collection : ℕ) 
  (h1 : total_collection = 2916) : ∃ n : ℕ, n * n = total_collection ∧ n = 54 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3315_331551


namespace NUMINAMATH_CALUDE_largest_B_term_l3315_331597

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the expansion of (1+0.1)^500 -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem largest_B_term : 
  ∀ k : ℕ, k ≤ 500 → k ≠ 45 → B 45 > B k := by sorry

end NUMINAMATH_CALUDE_largest_B_term_l3315_331597


namespace NUMINAMATH_CALUDE_unique_N_exists_l3315_331559

theorem unique_N_exists : ∃! N : ℝ, 
  ∃ a b c : ℝ, 
    a + b + c = 120 ∧
    a + 8 = N ∧
    8 * b = N ∧
    c / 8 = N := by
  sorry

end NUMINAMATH_CALUDE_unique_N_exists_l3315_331559


namespace NUMINAMATH_CALUDE_smallest_q_value_l3315_331598

theorem smallest_q_value (p q : ℕ+) 
  (h1 : (72 : ℚ) / 487 < p.val / q.val)
  (h2 : p.val / q.val < (18 : ℚ) / 121) :
  ∀ (q' : ℕ+), ((72 : ℚ) / 487 < p.val / q'.val ∧ p.val / q'.val < (18 : ℚ) / 121) → q.val ≤ q'.val →
  q.val = 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_q_value_l3315_331598


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l3315_331505

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l3315_331505


namespace NUMINAMATH_CALUDE_negation_equivalence_l3315_331587

theorem negation_equivalence (x : ℝ) :
  ¬(x^2 - x ≥ 0 → x > 2) ↔ (x^2 - x < 0 → x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3315_331587


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3315_331516

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 8)
  (h_sum2 : a 2 + a 4 = 12) :
  (∀ n : ℕ, a n = 2 * n) ∧
  (∃ n : ℕ, n * (n + 1) = 420) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3315_331516


namespace NUMINAMATH_CALUDE_billboard_count_l3315_331545

theorem billboard_count (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (total_hours : ℕ) (avg : ℕ) 
  (h1_count : h1 = 17)
  (h2_count : h2 = 20)
  (hours : total_hours = 3)
  (average : avg = 20)
  (avg_def : avg * total_hours = h1 + h2 + h3) :
  h3 = 23 := by
sorry

end NUMINAMATH_CALUDE_billboard_count_l3315_331545


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_iff_strictly_increasing_l3315_331571

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The condition that a_{n+2} > a_n for all positive integers n -/
def Condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) > a n

/-- The sequence is strictly increasing -/
def StrictlyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_condition_iff_strictly_increasing
  (a : ℕ → ℝ) (h : GeometricSequence a) :
  Condition a ↔ StrictlyIncreasing a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_iff_strictly_increasing_l3315_331571


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3315_331568

/-- The surface area of a cuboid -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 12, width 14, and height 7 is 700 -/
theorem cuboid_surface_area_example : cuboid_surface_area 12 14 7 = 700 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3315_331568


namespace NUMINAMATH_CALUDE_max_value_expression_l3315_331552

theorem max_value_expression (a b c d : ℝ) 
  (ha : -7.5 ≤ a ∧ a ≤ 7.5)
  (hb : -7.5 ≤ b ∧ b ≤ 7.5)
  (hc : -7.5 ≤ c ∧ c ≤ 7.5)
  (hd : -7.5 ≤ d ∧ d ≤ 7.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 240 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    (-7.5 ≤ a₀ ∧ a₀ ≤ 7.5) ∧
    (-7.5 ≤ b₀ ∧ b₀ ≤ 7.5) ∧
    (-7.5 ≤ c₀ ∧ c₀ ≤ 7.5) ∧
    (-7.5 ≤ d₀ ∧ d₀ ≤ 7.5) ∧
    (a₀ + 2*b₀ + c₀ + 2*d₀ - a₀*b₀ - b₀*c₀ - c₀*d₀ - d₀*a₀) = 240 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3315_331552


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l3315_331574

/-- The number of red lava lamps -/
def num_red : ℕ := 4

/-- The number of blue lava lamps -/
def num_blue : ℕ := 4

/-- The number of green lava lamps -/
def num_green : ℕ := 4

/-- The total number of lava lamps -/
def total_lamps : ℕ := num_red + num_blue + num_green

/-- The number of lamps that are turned on -/
def num_on : ℕ := 6

/-- The probability of the leftmost lamp being green and off, and the rightmost lamp being blue and on -/
def prob_specific_arrangement : ℚ := 80 / 1313

theorem lava_lamp_probability :
  prob_specific_arrangement = (Nat.choose (total_lamps - 2) num_red * Nat.choose (total_lamps - 2 - num_red) (num_blue - 1) * Nat.choose (total_lamps - 1) (num_on - 1)) /
  (Nat.choose total_lamps num_red * Nat.choose (total_lamps - num_red) num_blue * Nat.choose total_lamps num_on) :=
sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l3315_331574


namespace NUMINAMATH_CALUDE_remainder_problem_l3315_331506

theorem remainder_problem : (2^300 + 300) % (2^150 + 2^75 + 1) = 298 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3315_331506


namespace NUMINAMATH_CALUDE_triangle_area_implies_q_value_l3315_331530

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, q), 
    prove that if the area of the triangle is 50, then q = 125/12 -/
theorem triangle_area_implies_q_value (q : ℝ) : 
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, q)
  let triangle_area := (abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))) / 2
  triangle_area = 50 → q = 125 / 12 := by
  sorry

#check triangle_area_implies_q_value

end NUMINAMATH_CALUDE_triangle_area_implies_q_value_l3315_331530


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l3315_331534

/-- Represents a 3D coordinate --/
structure Coord :=
  (x y z : ℕ)

/-- Represents a cube --/
structure Cube :=
  (side_length : ℕ)

/-- Represents a plane perpendicular to the main diagonal of a cube --/
structure DiagonalPlane :=
  (cube : Cube)
  (passes_through_center : Bool)

/-- Counts the number of unit cubes intersected by a diagonal plane in a larger cube --/
def count_intersected_cubes (c : Cube) (p : DiagonalPlane) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem intersected_cubes_count (c : Cube) (p : DiagonalPlane) :
  c.side_length = 5 →
  p.cube = c →
  p.passes_through_center = true →
  count_intersected_cubes c p = 55 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l3315_331534


namespace NUMINAMATH_CALUDE_book_ratio_l3315_331572

/-- The number of books Pete read last year -/
def P : ℕ := sorry

/-- The number of books Matt read last year -/
def M : ℕ := sorry

/-- Pete doubles his reading this year -/
axiom pete_doubles : P * 2 = 300 - P

/-- Matt reads 50% more this year -/
axiom matt_increases : M * 3/2 = 75

/-- Pete read 300 books across both years -/
axiom pete_total : P + P * 2 = 300

/-- Matt read 75 books in his second year -/
axiom matt_second_year : M * 3/2 = 75

/-- The ratio of books Pete read last year to books Matt read last year is 2:1 -/
theorem book_ratio : P / M = 2 := by sorry

end NUMINAMATH_CALUDE_book_ratio_l3315_331572


namespace NUMINAMATH_CALUDE_james_total_score_l3315_331504

-- Define the number of field goals and shots
def field_goals : ℕ := 13
def shots : ℕ := 20

-- Define the point values for field goals and shots
def field_goal_points : ℕ := 3
def shot_points : ℕ := 2

-- Define the total points scored
def total_points : ℕ := field_goals * field_goal_points + shots * shot_points

-- Theorem stating that the total points scored is 79
theorem james_total_score : total_points = 79 := by
  sorry

end NUMINAMATH_CALUDE_james_total_score_l3315_331504


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_real_l3315_331567

open Set

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem union_M_complement_N_equals_real : M ∪ Nᶜ = univ :=
sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_real_l3315_331567


namespace NUMINAMATH_CALUDE_function_composition_ratio_l3315_331570

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 4 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l3315_331570


namespace NUMINAMATH_CALUDE_negation_of_forall_squared_plus_one_nonnegative_l3315_331535

theorem negation_of_forall_squared_plus_one_nonnegative :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_squared_plus_one_nonnegative_l3315_331535


namespace NUMINAMATH_CALUDE_ounces_per_pound_l3315_331580

-- Define constants
def pounds_per_ton : ℝ := 2500
def gunny_bag_capacity_tons : ℝ := 13
def num_packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

-- Define the theorem
theorem ounces_per_pound : ∃ (x : ℝ), 
  (gunny_bag_capacity_tons * pounds_per_ton = num_packets * (packet_weight_pounds + packet_weight_ounces / x)) → 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_pound_l3315_331580


namespace NUMINAMATH_CALUDE_system_solutions_l3315_331536

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  26 * x^2 - 42 * x * y + 17 * y^2 = 10 ∧
  10 * x^2 - 18 * x * y + 8 * y^2 = 6

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(-1, -2), (1, 2), (-11, -14), (11, 14)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x y : ℝ, system x y ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3315_331536


namespace NUMINAMATH_CALUDE_max_value_implies_a_l3315_331533

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 4, f a x = 3) →
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_max_value_implies_a_l3315_331533


namespace NUMINAMATH_CALUDE_impossible_c_value_l3315_331565

theorem impossible_c_value (a b c : ℤ) : 
  (∀ x : ℝ, (x + a) * (x + b) = x^2 + c*x - 8) → c ≠ 4 := by
sorry

end NUMINAMATH_CALUDE_impossible_c_value_l3315_331565


namespace NUMINAMATH_CALUDE_school_capacity_l3315_331566

/-- The total number of students that can be taught at a time by four primary schools -/
def total_students (capacity1 capacity2 : ℕ) : ℕ :=
  2 * capacity1 + 2 * capacity2

/-- Theorem stating that the total number of students is 1480 -/
theorem school_capacity : total_students 400 340 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_school_capacity_l3315_331566


namespace NUMINAMATH_CALUDE_sum_of_sixth_root_arguments_l3315_331544

open Complex

/-- The complex number whose sixth power is equal to -1/√3 - i√(2/3) -/
def z : ℂ := sorry

/-- The argument of z^6 in radians -/
def arg_z6 : ℝ := sorry

/-- The list of arguments of the sixth roots of z^6 in radians -/
def root_args : List ℝ := sorry

theorem sum_of_sixth_root_arguments :
  (root_args.sum * (180 / Real.pi)) = 1140 ∧ 
  (∀ φ ∈ root_args, 0 ≤ φ * (180 / Real.pi) ∧ φ * (180 / Real.pi) < 360) ∧
  (List.length root_args = 6) ∧
  (∀ φ ∈ root_args, Complex.exp (φ * Complex.I) ^ 6 = z^6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sixth_root_arguments_l3315_331544


namespace NUMINAMATH_CALUDE_interval_condition_l3315_331510

theorem interval_condition (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 1/2 < x ∧ x < 3/5 :=
by sorry

end NUMINAMATH_CALUDE_interval_condition_l3315_331510


namespace NUMINAMATH_CALUDE_no_solution_exists_l3315_331507

theorem no_solution_exists : ¬ ∃ (a b c t x₁ x₂ x₃ : ℝ),
  (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧
  (a * x₁^2 + b * t * x₁ + c = 0) ∧
  (a * x₂^2 + b * t * x₂ + c = 0) ∧
  (b * x₂^2 + c * x₂ + a = 0) ∧
  (b * x₃^2 + c * x₃ + a = 0) ∧
  (c * x₃^2 + a * t * x₃ + b = 0) ∧
  (c * x₁^2 + a * t * x₁ + b = 0) :=
sorry

#check no_solution_exists

end NUMINAMATH_CALUDE_no_solution_exists_l3315_331507


namespace NUMINAMATH_CALUDE_no_geometric_subsequence_of_three_l3315_331596

theorem no_geometric_subsequence_of_three (a : ℕ → ℤ) :
  (∀ n, a n = 3^n - 2^n) →
  ¬ ∃ r s t : ℕ, r < s ∧ s < t ∧ ∃ b : ℚ, b ≠ 0 ∧
    (a s : ℚ) / (a r : ℚ) = b ∧ (a t : ℚ) / (a s : ℚ) = b :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_subsequence_of_three_l3315_331596


namespace NUMINAMATH_CALUDE_decreasing_cubic_implies_nonpositive_a_l3315_331519

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The cubic function f(x) = ax³ - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x + 1

theorem decreasing_cubic_implies_nonpositive_a :
  ∀ a : ℝ, DecreasingFunction (f a) → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_cubic_implies_nonpositive_a_l3315_331519


namespace NUMINAMATH_CALUDE_ratio_of_percentages_l3315_331526

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.30 * Q)
  (hQ : Q = 0.20 * P)
  (hN : N = 0.50 * P)
  (hP : P ≠ 0) :
  M / N = 3 / 25 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_percentages_l3315_331526


namespace NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l3315_331590

/-- Sequence definition -/
def a : ℕ → ℕ → ℕ
  | m, 0 => 0
  | m, 1 => 1
  | m, k + 2 => m * a m (k + 1) - a m k

/-- Main theorem -/
theorem consecutive_terms_iff_equation (m : ℕ) :
  ∀ x y : ℕ, x^2 - m*x*y + y^2 = 1 ↔ ∃ k : ℕ, x = a m k ∧ y = a m (k + 1) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_terms_iff_equation_l3315_331590


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_9x_l3315_331589

theorem factorization_x_squared_minus_9x (x : ℝ) : x^2 - 9*x = x*(x - 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_9x_l3315_331589


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l3315_331525

theorem sqrt_difference_approximation : 
  |Real.sqrt 100 - Real.sqrt 96 - 0.20| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l3315_331525


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l3315_331539

/-- A function passes through the first and third quadrants if for any non-zero x,
    x and f(x) have the same sign. -/
def passes_through_first_and_third_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ f x > 0) ∨ (x < 0 ∧ f x < 0)

/-- Theorem: If the graph of y = kx passes through the first and third quadrants,
    then k is positive. -/
theorem proportional_function_quadrants (k : ℝ) :
  passes_through_first_and_third_quadrants (λ x => k * x) → k > 0 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l3315_331539


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3315_331511

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 8 players, where each player plays every other player 
    exactly once, the total number of games played is 28. -/
theorem chess_tournament_games :
  num_games 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3315_331511


namespace NUMINAMATH_CALUDE_hexagon_area_ratio_l3315_331576

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- A point on a side of the hexagon -/
def SidePoint (h : RegularHexagon) (i : Fin 6) := ℝ × ℝ

/-- The ratio of areas of two polygons -/
def AreaRatio (p1 p2 : Set (ℝ × ℝ)) : ℚ := sorry

theorem hexagon_area_ratio 
  (ABCDEF : RegularHexagon)
  (P : SidePoint ABCDEF 0) (Q : SidePoint ABCDEF 1) (R : SidePoint ABCDEF 2)
  (S : SidePoint ABCDEF 3) (T : SidePoint ABCDEF 4) (U : SidePoint ABCDEF 5)
  (h_P : P = (2/3 : ℝ) • ABCDEF.vertices 0 + (1/3 : ℝ) • ABCDEF.vertices 1)
  (h_Q : Q = (2/3 : ℝ) • ABCDEF.vertices 1 + (1/3 : ℝ) • ABCDEF.vertices 2)
  (h_R : R = (2/3 : ℝ) • ABCDEF.vertices 2 + (1/3 : ℝ) • ABCDEF.vertices 3)
  (h_S : S = (2/3 : ℝ) • ABCDEF.vertices 3 + (1/3 : ℝ) • ABCDEF.vertices 4)
  (h_T : T = (2/3 : ℝ) • ABCDEF.vertices 4 + (1/3 : ℝ) • ABCDEF.vertices 5)
  (h_U : U = (2/3 : ℝ) • ABCDEF.vertices 5 + (1/3 : ℝ) • ABCDEF.vertices 0) :
  let inner_hexagon := {ABCDEF.vertices 0, R, ABCDEF.vertices 2, T, ABCDEF.vertices 4, P}
  let outer_hexagon := {ABCDEF.vertices i | i : Fin 6}
  AreaRatio inner_hexagon outer_hexagon = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_ratio_l3315_331576


namespace NUMINAMATH_CALUDE_square_difference_l3315_331583

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) :
  (x - y)^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3315_331583


namespace NUMINAMATH_CALUDE_matrix_N_properties_l3315_331518

def N : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; -1/2, 2]

theorem matrix_N_properties :
  let v1 : Matrix (Fin 2) (Fin 1) ℚ := !![2; -1]
  let v2 : Matrix (Fin 2) (Fin 1) ℚ := !![0; 3]
  let r1 : Matrix (Fin 2) (Fin 1) ℚ := !![5; -3]
  let r2 : Matrix (Fin 2) (Fin 1) ℚ := !![-9; 6]
  (N * v1 = r1) ∧
  (N * v2 = r2) ∧
  (N 0 0 - N 1 1 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_matrix_N_properties_l3315_331518


namespace NUMINAMATH_CALUDE_systematic_sampling_used_l3315_331508

/-- Represents the sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents the auditorium setup and sampling process --/
structure AuditoriumSampling where
  total_seats : Nat
  seats_per_row : Nat
  selected_seat_number : Nat
  num_selected : Nat

/-- Determines the sampling method based on the auditorium setup and selection process --/
def determine_sampling_method (setup : AuditoriumSampling) : SamplingMethod :=
  sorry

/-- Theorem stating that the sampling method used is systematic sampling --/
theorem systematic_sampling_used (setup : AuditoriumSampling) 
  (h1 : setup.total_seats = 25)
  (h2 : setup.seats_per_row = 20)
  (h3 : setup.selected_seat_number = 15)
  (h4 : setup.num_selected = 25) :
  determine_sampling_method setup = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_used_l3315_331508


namespace NUMINAMATH_CALUDE_cosine_square_root_pi_eighths_l3315_331502

theorem cosine_square_root_pi_eighths :
  Real.sqrt ((3 - Real.cos (π / 8) ^ 2) * (3 - Real.cos (3 * π / 8) ^ 2)) = 3 * Real.sqrt 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_square_root_pi_eighths_l3315_331502


namespace NUMINAMATH_CALUDE_calculate_expression_l3315_331593

theorem calculate_expression : 
  2 * Real.sin (π / 4) + |(-Real.sqrt 2)| - (π - 2023)^0 - Real.sqrt 2 = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3315_331593


namespace NUMINAMATH_CALUDE_cloud_9_diving_refund_l3315_331563

/-- Cloud 9 Diving Company Cancellation Refund Problem -/
theorem cloud_9_diving_refund (individual_bookings group_bookings total_after_cancellations : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : total_after_cancellations = 26400) :
  individual_bookings + group_bookings - total_after_cancellations = 1600 := by
  sorry

end NUMINAMATH_CALUDE_cloud_9_diving_refund_l3315_331563


namespace NUMINAMATH_CALUDE_sports_club_membership_l3315_331553

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : tennis = 18)
  (h4 : both = 3) :
  total - (badminton + tennis - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l3315_331553


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l3315_331591

theorem real_solutions_quadratic (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 - 3 * x * y + x + 8 = 0) ↔ x ≤ -4 ∨ x ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l3315_331591


namespace NUMINAMATH_CALUDE_inequality_relationship_l3315_331514

theorem inequality_relationship (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l3315_331514


namespace NUMINAMATH_CALUDE_edward_earnings_l3315_331595

theorem edward_earnings : 
  let lawn_pay : ℕ := 8
  let garden_pay : ℕ := 12
  let lawns_mowed : ℕ := 5
  let gardens_cleaned : ℕ := 3
  let fuel_cost : ℕ := 10
  let equipment_cost : ℕ := 15
  let initial_savings : ℕ := 7

  let total_earnings := lawn_pay * lawns_mowed + garden_pay * gardens_cleaned
  let total_expenses := fuel_cost + equipment_cost
  let final_amount := total_earnings + initial_savings - total_expenses

  final_amount = 58 := by sorry

end NUMINAMATH_CALUDE_edward_earnings_l3315_331595


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l3315_331585

theorem positive_numbers_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2*a*b) :
  (a + 2*b ≥ 4) ∧ (a^2 + 4*b^2 ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l3315_331585


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3315_331528

theorem unique_integer_satisfying_conditions (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  7 > x ∧ x > -1 ∧
  4 > x ∧ x > 0 ∧
  x + 1 < 5 →
  x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3315_331528


namespace NUMINAMATH_CALUDE_intersection_point_expression_l3315_331575

theorem intersection_point_expression (m n : ℝ) : 
  n = m - 2022 → 
  n = -2022 / m → 
  (2022 / m) + ((m^2 - 2022*m) / n) = 2022 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_expression_l3315_331575


namespace NUMINAMATH_CALUDE_inequality_condition_l3315_331540

theorem inequality_condition (a b : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3315_331540


namespace NUMINAMATH_CALUDE_kishore_savings_l3315_331541

/-- Proves that given the total expenses and the fact that they represent 90% of the salary,
    the 10% savings amount to the correct value. -/
theorem kishore_savings (total_expenses : ℕ) (monthly_salary : ℕ) : 
  total_expenses = 20700 →
  total_expenses = (90 * monthly_salary) / 100 →
  (10 * monthly_salary) / 100 = 2300 :=
by sorry

end NUMINAMATH_CALUDE_kishore_savings_l3315_331541


namespace NUMINAMATH_CALUDE_ellipse_equation_l3315_331524

-- Define the ellipse
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def on_line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ (x1 y1 x2 y2 : ℝ), 
    is_ellipse x1 y1 a b ∧ 
    is_ellipse x2 y2 a b ∧ 
    on_line x1 y1 ∧ 
    on_line x2 y2 ∧ 
    ((x1 = a ∧ y1 = 0) ∨ (x2 = a ∧ y2 = 0)) ∧ 
    ((x1 = 0 ∧ y1 = b) ∨ (x2 = 0 ∧ y2 = b))) →
  (∀ x y : ℝ, is_ellipse x y a b ↔ x^2 / 16 + y^2 / 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3315_331524


namespace NUMINAMATH_CALUDE_fraction_comparison_l3315_331558

theorem fraction_comparison (n : ℕ) (hn : n > 0) :
  (n + 1 : ℝ) ^ (n + 3) / (n + 3 : ℝ) ^ (n + 1) > n ^ (n + 2) / (n + 2 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3315_331558


namespace NUMINAMATH_CALUDE_black_area_after_changes_l3315_331560

/-- The fraction of black area remaining after one change -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of changes -/
def num_changes : ℕ := 4

/-- The fraction of the original black area remaining after 'num_changes' changes -/
def final_fraction : ℚ := remaining_fraction ^ num_changes

theorem black_area_after_changes : final_fraction = 4096 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_changes_l3315_331560


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3315_331586

/-- Proves that the ratio of boys to girls in a school with 90 students, of which 60 are girls, is 1:2 -/
theorem boys_to_girls_ratio (total_students : Nat) (girls : Nat) (h1 : total_students = 90) (h2 : girls = 60) :
  (total_students - girls) / girls = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3315_331586


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3315_331592

theorem container_volume_ratio : 
  ∀ (A B : ℝ),  -- A and B are the volumes of the first and second containers
  A > 0 → B > 0 →  -- Both volumes are positive
  (4/5 * A - 1/5 * A) = 2/3 * B →  -- Amount poured equals 2/3 of second container
  A / B = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3315_331592


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_36_l3315_331564

/-- The sum of consecutive integers starting from a given integer -/
def sumConsecutiveIntegers (start : ℤ) (count : ℕ) : ℤ :=
  (count : ℤ) * (2 * start + (count : ℤ) - 1) / 2

/-- The property that the sum of a sequence of consecutive integers is 36 -/
def hasSumThirtySix (start : ℤ) (count : ℕ) : Prop :=
  sumConsecutiveIntegers start count = 36

/-- The theorem stating that 72 is the greatest number of consecutive integers whose sum is 36 -/
theorem greatest_consecutive_integers_sum_36 :
  (∃ start : ℤ, hasSumThirtySix start 72) ∧
  (∀ n : ℕ, n > 72 → ∀ start : ℤ, ¬hasSumThirtySix start n) :=
sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_36_l3315_331564


namespace NUMINAMATH_CALUDE_tate_total_years_l3315_331555

/-- Calculate the total years Tate spent in education, travel, and work -/
def totalYears : ℕ :=
  let typicalHighSchool := 4
  let highSchool := typicalHighSchool - 1
  let gapYears := 2
  let bachelors := 2 * highSchool
  let certification := 1
  let workExperience := 1
  let masters := bachelors / 2
  let phd := 3 * (highSchool + bachelors + masters)
  highSchool + gapYears + bachelors + certification + workExperience + masters + phd

/-- Theorem stating that the total years Tate spent is 52 -/
theorem tate_total_years : totalYears = 52 := by
  sorry

end NUMINAMATH_CALUDE_tate_total_years_l3315_331555


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_bicycle_price_proof_l3315_331554

theorem bicycle_price_after_discounts (original_price : ℝ) 
  (first_discount_percent : ℝ) (second_discount_percent : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let final_price := price_after_first_discount * (1 - second_discount_percent / 100)
  final_price

theorem bicycle_price_proof : 
  bicycle_price_after_discounts 200 20 25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_bicycle_price_proof_l3315_331554


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_slope_l3315_331531

/-- Circle ω₁ -/
def ω₁ (x y : ℝ) : Prop := x^2 + y^2 + 12*x - 20*y - 100 = 0

/-- Circle ω₂ -/
def ω₂ (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 20*y + 196 = 0

/-- A circle is externally tangent to ω₂ -/
def externally_tangent_ω₂ (x y r : ℝ) : Prop :=
  r + 8 = Real.sqrt ((x - 6)^2 + (y - 10)^2)

/-- A circle is internally tangent to ω₁ -/
def internally_tangent_ω₁ (x y r : ℝ) : Prop :=
  16 - r = Real.sqrt ((x + 6)^2 + (y - 10)^2)

/-- The main theorem -/
theorem smallest_tangent_circle_slope :
  ∃ (m : ℝ), m > 0 ∧ m^2 = 160/99 ∧
  (∀ (a : ℝ), a > 0 → a < m →
    ¬∃ (x y r : ℝ), y = a*x ∧
      externally_tangent_ω₂ x y r ∧
      internally_tangent_ω₁ x y r) ∧
  (∃ (x y r : ℝ), y = m*x ∧
    externally_tangent_ω₂ x y r ∧
    internally_tangent_ω₁ x y r) :=
sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_slope_l3315_331531


namespace NUMINAMATH_CALUDE_arcade_change_machine_l3315_331501

theorem arcade_change_machine (total_value : ℕ) (one_dollar_bills : ℕ) : 
  total_value = 300 → one_dollar_bills = 175 → 
  ∃ (five_dollar_bills : ℕ), 
    one_dollar_bills + five_dollar_bills = 200 ∧ 
    one_dollar_bills + 5 * five_dollar_bills = total_value :=
by sorry

end NUMINAMATH_CALUDE_arcade_change_machine_l3315_331501


namespace NUMINAMATH_CALUDE_triangular_prism_ratio_l3315_331513

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D

/-- Checks if two planes are perpendicular -/
def arePlanesPerp (p1 p2 p3 q1 q2 q3 : Point3D) : Prop := sorry

/-- Checks if two vectors are perpendicular -/
def areVectorsPerp (v1 v2 : Point3D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Checks if a point lies on a line segment -/
def isOnLineSegment (p1 p2 p : Point3D) : Prop := sorry

/-- Main theorem -/
theorem triangular_prism_ratio 
  (prism : TriangularPrism)
  (D : Point3D)
  (h1 : distance prism.A prism.A₁ = 4)
  (h2 : distance prism.A prism.C = 4)
  (h3 : distance prism.A₁ prism.C₁ = 4)
  (h4 : distance prism.C prism.C₁ = 4)
  (h5 : arePlanesPerp prism.A prism.B prism.C prism.A prism.A₁ prism.C₁)
  (h6 : distance prism.A prism.B = 3)
  (h7 : distance prism.B prism.C = 5)
  (h8 : isOnLineSegment prism.B prism.C₁ D)
  (h9 : areVectorsPerp (Point3D.mk (D.x - prism.A.x) (D.y - prism.A.y) (D.z - prism.A.z))
                       (Point3D.mk (prism.B.x - prism.A₁.x) (prism.B.y - prism.A₁.y) (prism.B.z - prism.A₁.z))) :
  distance prism.B D / distance prism.B prism.C₁ = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_ratio_l3315_331513


namespace NUMINAMATH_CALUDE_greatest_divisible_by_cubes_l3315_331547

theorem greatest_divisible_by_cubes : ∃ (n : ℕ), n = 60 ∧ 
  (∀ (m : ℕ), m^3 ≤ n → n % m = 0) ∧
  (∀ (k : ℕ), k > n → ∃ (m : ℕ), m^3 ≤ k ∧ k % m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisible_by_cubes_l3315_331547


namespace NUMINAMATH_CALUDE_binomial_product_l3315_331569

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l3315_331569


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3315_331584

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_property (a b c : ℝ) :
  -- The axis of symmetry is at x = 3.5
  (∀ x : ℝ, QuadraticFunction a b c (3.5 - x) = QuadraticFunction a b c (3.5 + x)) →
  -- The function passes through the point (2, -1)
  QuadraticFunction a b c 2 = -1 →
  -- p(5) is an integer
  ∃ n : ℤ, QuadraticFunction a b c 5 = n →
  -- Then p(5) = -1
  QuadraticFunction a b c 5 = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3315_331584


namespace NUMINAMATH_CALUDE_worker_save_fraction_l3315_331562

/-- Represents the worker's monthly savings scenario -/
structure WorkerSavings where
  monthly_pay : ℝ
  save_fraction : ℝ
  (monthly_pay_positive : monthly_pay > 0)
  (save_fraction_valid : 0 ≤ save_fraction ∧ save_fraction ≤ 1)

/-- The total amount saved over a year -/
def yearly_savings (w : WorkerSavings) : ℝ := 12 * w.save_fraction * w.monthly_pay

/-- The amount not saved from monthly pay -/
def monthly_unsaved (w : WorkerSavings) : ℝ := (1 - w.save_fraction) * w.monthly_pay

/-- Theorem stating the fraction of monthly take-home pay saved -/
theorem worker_save_fraction (w : WorkerSavings) 
  (h : yearly_savings w = 5 * monthly_unsaved w) : 
  w.save_fraction = 5 / 17 := by
  sorry

end NUMINAMATH_CALUDE_worker_save_fraction_l3315_331562


namespace NUMINAMATH_CALUDE_sum_of_squares_l3315_331532

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) : a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3315_331532
