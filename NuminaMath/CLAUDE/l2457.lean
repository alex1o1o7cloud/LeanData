import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l2457_245740

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 271 →
  divisor = 30 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2457_245740


namespace NUMINAMATH_CALUDE_sin6_cos2_integral_l2457_245726

theorem sin6_cos2_integral : ∫ x in (0 : ℝ)..(2 * Real.pi), (Real.sin x)^6 * (Real.cos x)^2 = (5 * Real.pi) / 64 := by
  sorry

end NUMINAMATH_CALUDE_sin6_cos2_integral_l2457_245726


namespace NUMINAMATH_CALUDE_sevenPeopleArrangements_eq_3600_l2457_245743

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters. -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- The number of arrangements of seven people where two specific people must not be adjacent. -/
def sevenPeopleArrangements : ℕ := 
  permutations 5 5 * permutations 6 2

theorem sevenPeopleArrangements_eq_3600 : 
  sevenPeopleArrangements = 3600 := by sorry

end NUMINAMATH_CALUDE_sevenPeopleArrangements_eq_3600_l2457_245743


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2457_245765

theorem sum_of_solutions (a : ℝ) (h : a > 2) : 
  ∃ x₁ x₂ : ℝ, (Real.sqrt (a - Real.sqrt (a + x₁)) = x₁ + 1) ∧ 
              (Real.sqrt (a - Real.sqrt (a + x₂)) = x₂ + 1) ∧ 
              (x₁ + x₂ = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2457_245765


namespace NUMINAMATH_CALUDE_bus_speed_relation_l2457_245750

/-- Represents the speed and stoppage characteristics of a bus -/
structure Bus where
  speed_with_stops : ℝ
  stop_time : ℝ
  speed_without_stops : ℝ

/-- Theorem stating the relationship between bus speeds and stop time -/
theorem bus_speed_relation (b : Bus) 
  (h1 : b.speed_with_stops = 12)
  (h2 : b.stop_time = 45)
  : b.speed_without_stops = 48 := by
  sorry

#check bus_speed_relation

end NUMINAMATH_CALUDE_bus_speed_relation_l2457_245750


namespace NUMINAMATH_CALUDE_fiona_reach_probability_l2457_245792

/-- Represents a lily pad with its number and whether it contains a predator -/
structure LilyPad :=
  (number : Nat)
  (hasPredator : Bool)

/-- Represents Fiona's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the frog's journey -/
def FrogJourney := List LilyPad

/-- The probability of each move -/
def moveProbability : Move → ℚ
  | Move.Hop => 1/2
  | Move.Jump => 1/2

/-- The number of pads to move for each move type -/
def moveDistance : Move → Nat
  | Move.Hop => 1
  | Move.Jump => 2

/-- The lily pads in the pond -/
def lilyPads : List LilyPad :=
  List.range 16 |> List.map (λ n => ⟨n, n ∈ [4, 7, 11]⟩)

/-- Check if a journey is safe (doesn't land on predator pads) -/
def isSafeJourney (journey : FrogJourney) : Bool :=
  journey.all (λ pad => !pad.hasPredator)

/-- Calculate the probability of a specific journey -/
def journeyProbability (journey : FrogJourney) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem fiona_reach_probability :
  ∃ (safeJourneys : List FrogJourney),
    (∀ j ∈ safeJourneys, j.head? = some ⟨0, false⟩ ∧
                         j.getLast? = some ⟨14, false⟩ ∧
                         isSafeJourney j) ∧
    (safeJourneys.map journeyProbability).sum = 3/256 :=
  sorry

end NUMINAMATH_CALUDE_fiona_reach_probability_l2457_245792


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2457_245714

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 7 → 
  (a + b + c) / 3 = a + 8 → 
  (a + b + c) / 3 = c - 20 → 
  a + b + c = 57 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2457_245714


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2457_245762

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 1
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2457_245762


namespace NUMINAMATH_CALUDE_bernardo_wins_l2457_245797

theorem bernardo_wins (N : ℕ) : N ≤ 999 ∧ 72 * N < 1000 ∧ 36 * N < 1000 ∧ ∀ m : ℕ, m < N → (72 * m ≥ 1000 ∨ 36 * m ≥ 1000) → N = 13 := by
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l2457_245797


namespace NUMINAMATH_CALUDE_not_all_diagonal_cells_good_l2457_245712

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 13
  col : Fin 13

/-- Represents the table -/
def Table := Fin 13 → Fin 13 → Fin 25

/-- Checks if a cell is "good" -/
def is_good (t : Table) (c : Cell) : Prop :=
  ∀ n : Fin 25, (∃! i : Fin 13, t i c.col = n) ∧ (∃! j : Fin 13, t c.row j = n)

/-- Represents the main diagonal -/
def main_diagonal : List Cell :=
  List.map (λ i => ⟨i, i⟩) (List.range 13)

/-- The theorem to be proved -/
theorem not_all_diagonal_cells_good (t : Table) : 
  ¬(∀ c ∈ main_diagonal, is_good t c) := by
  sorry


end NUMINAMATH_CALUDE_not_all_diagonal_cells_good_l2457_245712


namespace NUMINAMATH_CALUDE_income_ratio_uma_bala_l2457_245742

theorem income_ratio_uma_bala (uma_income : ℕ) (uma_expenditure bala_expenditure : ℕ) 
  (h1 : uma_income = 16000)
  (h2 : uma_expenditure = 7 * bala_expenditure / 6)
  (h3 : uma_income - uma_expenditure = 2000)
  (h4 : bala_income - bala_expenditure = 2000)
  : uma_income / (uma_income - 2000) = 8 / 7 :=
by sorry

end NUMINAMATH_CALUDE_income_ratio_uma_bala_l2457_245742


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2457_245785

theorem quadratic_inequality (x : ℝ) : 
  (2 * x^2 - 5 * x - 12 > 0 ↔ x < -3/2 ∨ x > 4) ∧
  (2 * x^2 - 5 * x - 12 < 0 ↔ -3/2 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2457_245785


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l2457_245713

theorem complex_in_first_quadrant (z : ℂ) : z = Complex.mk (Real.sqrt 3) 1 → z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l2457_245713


namespace NUMINAMATH_CALUDE_correct_calculation_l2457_245771

theorem correct_calculation : 
  (5 + (-6) = -1) ∧ 
  (1 / Real.sqrt 2 ≠ Real.sqrt 2) ∧ 
  (3 * (-2) ≠ 6) ∧ 
  (Real.sin (30 * π / 180) ≠ Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_l2457_245771


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2457_245777

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- Definition of S_n
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Arithmetic sequence condition
  S 3 / S 6 = 1 / 3 →
  S 6 / S 12 = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2457_245777


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_l2457_245774

theorem simplify_inverse_sum (k x y : ℝ) (hk : k ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  (k * x⁻¹ + k * y⁻¹)⁻¹ = (x * y) / (k * (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_l2457_245774


namespace NUMINAMATH_CALUDE_orange_weight_after_water_loss_orange_weight_problem_l2457_245757

/-- Calculates the new weight of oranges after water loss -/
theorem orange_weight_after_water_loss 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (evaporation_loss_percentage : ℝ) 
  (skin_loss_percentage : ℝ) : ℝ :=
  let initial_water_weight := initial_weight * initial_water_percentage
  let dry_weight := initial_weight - initial_water_weight
  let evaporation_loss := initial_water_weight * evaporation_loss_percentage
  let remaining_water_after_evaporation := initial_water_weight - evaporation_loss
  let skin_loss := remaining_water_after_evaporation * skin_loss_percentage
  let total_water_loss := evaporation_loss + skin_loss
  let new_water_weight := initial_water_weight - total_water_loss
  new_water_weight + dry_weight

/-- The new weight of oranges after water loss is approximately 4.67225 kg -/
theorem orange_weight_problem : 
  ∃ ε > 0, |orange_weight_after_water_loss 5 0.95 0.05 0.02 - 4.67225| < ε :=
sorry

end NUMINAMATH_CALUDE_orange_weight_after_water_loss_orange_weight_problem_l2457_245757


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l2457_245781

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 56 = 3 ∧
  n % 78 = 3 ∧
  n % 9 = 0 ∧
  ∀ m : ℕ, m > 0 ∧ m % 56 = 3 ∧ m % 78 = 3 ∧ m % 9 = 0 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l2457_245781


namespace NUMINAMATH_CALUDE_smallest_K_for_divisibility_l2457_245793

def repeatedDigit (d : ℕ) (K : ℕ) : ℕ :=
  d * (10^K - 1) / 9

theorem smallest_K_for_divisibility (K : ℕ) : 
  (∀ n : ℕ, n < K → ¬(198 ∣ repeatedDigit 2 n)) ∧ 
  (198 ∣ repeatedDigit 2 K) → 
  K = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_K_for_divisibility_l2457_245793


namespace NUMINAMATH_CALUDE_perpendicular_line_coordinates_l2457_245741

/-- Given two points P and Q in a 2D plane, where Q has fixed coordinates
    and P's coordinates depend on a parameter 'a', prove that if the line PQ
    is perpendicular to the y-axis, then P has specific coordinates. -/
theorem perpendicular_line_coordinates 
  (Q : ℝ × ℝ) 
  (P : ℝ → ℝ × ℝ) 
  (h1 : Q = (2, -3))
  (h2 : ∀ a, P a = (2*a + 2, a - 5))
  (h3 : ∀ a, (P a).1 = Q.1) :
  ∃ a, P a = (6, -3) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_coordinates_l2457_245741


namespace NUMINAMATH_CALUDE_infinite_solution_equation_non_solutions_l2457_245724

/-- Given an equation with infinitely many solutions, prove the number and sum of non-solutions -/
theorem infinite_solution_equation_non_solutions (A B C : ℚ) : 
  (∀ x, (x + B) * (A * x + 42) = 3 * (x + C) * (x + 9)) →
  (∃! s : Finset ℚ, s.card = 2 ∧ 
    (∀ x ∈ s, (x + B) * (A * x + 42) ≠ 3 * (x + C) * (x + 9)) ∧
    (∀ x ∉ s, (x + B) * (A * x + 42) = 3 * (x + C) * (x + 9)) ∧
    s.sum id = -187/13) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solution_equation_non_solutions_l2457_245724


namespace NUMINAMATH_CALUDE_evaluate_expression_l2457_245768

theorem evaluate_expression : (2^13 : ℚ) / (5 * 4^3) = 128 / 5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2457_245768


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l2457_245722

theorem at_least_one_equation_has_two_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  ∃ (x y : ℝ),
    (x ≠ y ∧
      ((a * x^2 + 2 * b * x + c = 0 ∧ a * y^2 + 2 * b * y + c = 0) ∨
       (b * x^2 + 2 * c * x + a = 0 ∧ b * y^2 + 2 * c * y + a = 0) ∨
       (c * x^2 + 2 * a * x + b = 0 ∧ c * y^2 + 2 * a * y + b = 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_roots_l2457_245722


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_necessary_condition_range_l2457_245747

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m+1}

theorem intersection_when_m_is_one :
  A ∩ B 1 = {x : ℝ | 0 < x ∧ x ≤ 3/2} := by sorry

theorem necessary_condition_range :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ↔ m ≤ 1/6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_necessary_condition_range_l2457_245747


namespace NUMINAMATH_CALUDE_f_is_odd_l2457_245729

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom add_property : ∀ x y : ℝ, f (x + y) = f x + f y
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0

-- Define what it means for f to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem f_is_odd : is_odd f := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l2457_245729


namespace NUMINAMATH_CALUDE_triangle_relations_l2457_245759

/-- Given a triangle ABC with side lengths a, b, c, altitudes h_a, h_b, h_c, 
    inradius r, and exradii r_a, r_b, r_c, the following equations hold -/
theorem triangle_relations (a b c h_a h_b h_c r r_a r_b r_c : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (hr : r > 0) (hr_a : r_a > 0) (hr_b : r_b > 0) (hr_c : r_c > 0) 
    (hh_a : h_a > 0) (hh_b : h_b > 0) (hh_c : h_c > 0) :
  h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) ∧
  1 / h_a + 1 / h_b + 1 / h_c = 1 / r ∧
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c ∧
  (h_a + h_b + h_c) * (1 / h_a + 1 / h_b + 1 / h_c) = (a + b + c) * (1 / a + 1 / b + 1 / c) ∧
  (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_relations_l2457_245759


namespace NUMINAMATH_CALUDE_simplify_expression_l2457_245794

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (Real.sqrt a * 3 * a^2) = 1 / Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2457_245794


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2457_245705

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (-1 + Real.sqrt 5) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 5) / 2
  ∀ x : ℝ, x^2 + x - 1 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2457_245705


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l2457_245702

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_unit_cubes : Nat
  painted_grid_size : Nat

/-- Calculates the number of unpainted unit cubes in the painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := {
    size := 6,
    total_unit_cubes := 216,
    painted_grid_size := 4
  }
  unpainted_cubes cube = 176 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l2457_245702


namespace NUMINAMATH_CALUDE_trapezoid_division_common_side_l2457_245778

theorem trapezoid_division_common_side
  (a b k p : ℝ)
  (h1 : a > b)
  (h2 : k > 0)
  (h3 : p > 0) :
  let x := Real.sqrt ((k * a^2 + p * b^2) / (p + k))
  ∃ (h1 h2 : ℝ), 
    h1 > 0 ∧ h2 > 0 ∧
    (b + x) * h1 / ((a + x) * h2) = k / p ∧
    x > b ∧ x < a :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_division_common_side_l2457_245778


namespace NUMINAMATH_CALUDE_incorrect_steps_count_l2457_245773

theorem incorrect_steps_count (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : 
  ∃ (s1 s2 s3 : Prop),
    (s1 ↔ (a * c > b * c ∧ b * c > b * d)) ∧
    (s2 ↔ (a * c > b * c ∧ b * c > b * d → a * c > b * d)) ∧
    (s3 ↔ (a * c > b * d → a / d > b / c)) ∧
    (¬s1 ∧ s2 ∧ ¬s3) :=
by sorry


end NUMINAMATH_CALUDE_incorrect_steps_count_l2457_245773


namespace NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l2457_245749

theorem cos_four_minus_sin_four_equals_cos_double (θ : ℝ) :
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by
  sorry

end NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l2457_245749


namespace NUMINAMATH_CALUDE_heartsuit_three_four_l2457_245786

-- Define the ⊛ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_four : heartsuit 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_four_l2457_245786


namespace NUMINAMATH_CALUDE_total_cost_is_24_l2457_245744

/-- The number of index fingers on a person's hands. -/
def number_of_index_fingers : ℕ := 2

/-- The cost of one gold ring in dollars. -/
def cost_per_ring : ℕ := 12

/-- The total cost of buying gold rings for all index fingers. -/
def total_cost : ℕ := number_of_index_fingers * cost_per_ring

/-- Theorem stating that the total cost of buying gold rings for all index fingers is $24. -/
theorem total_cost_is_24 : total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_24_l2457_245744


namespace NUMINAMATH_CALUDE_harrys_travel_time_l2457_245717

/-- Harry's travel time calculation -/
theorem harrys_travel_time :
  let bus_time_so_far : ℕ := 15
  let remaining_bus_time : ℕ := 25
  let total_bus_time : ℕ := bus_time_so_far + remaining_bus_time
  let walking_time : ℕ := total_bus_time / 2
  total_bus_time + walking_time = 60 := by sorry

end NUMINAMATH_CALUDE_harrys_travel_time_l2457_245717


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l2457_245754

/-- Given points P, Q, R in a plane, and G as the midpoint of PQ,
    prove that the sum of the slope and y-intercept of line RG is 9/2. -/
theorem slope_intercept_sum (P Q R G : ℝ × ℝ) : 
  P = (0, 10) →
  Q = (0, 0) →
  R = (10, 0) →
  G = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  let slope := (G.2 - R.2) / (G.1 - R.1)
  let y_intercept := G.2 - slope * G.1
  slope + y_intercept = 9/2 := by
sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l2457_245754


namespace NUMINAMATH_CALUDE_smallest_sector_angle_l2457_245728

theorem smallest_sector_angle (n : ℕ) (a d : ℤ) : 
  n = 8 ∧ 
  (∀ i : ℕ, i < n → (a + i * d : ℤ) > 0) ∧
  (∀ i : ℕ, i < n → (a + i * d : ℤ).natAbs = a + i * d) ∧
  (n : ℤ) * (2 * a + (n - 1) * d) = 360 * 2 →
  a ≥ 38 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sector_angle_l2457_245728


namespace NUMINAMATH_CALUDE_fraction_reciprocal_difference_l2457_245704

theorem fraction_reciprocal_difference : 
  let f : ℚ := 4/5
  let r : ℚ := 5/4  -- reciprocal of f
  r - f = 9/20 := by sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_difference_l2457_245704


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l2457_245763

/-- The slope of a chord in an ellipse with midpoint (-2, 1) -/
theorem ellipse_chord_slope :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  (x₁ + x₂ = -4) →
  (y₁ + y₂ = 2) →
  ((y₂ - y₁) / (x₂ - x₁) = 9 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l2457_245763


namespace NUMINAMATH_CALUDE_tims_sock_drawer_probability_l2457_245730

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer where
  gray : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the probability of picking a matching pair of socks -/
def probabilityOfMatchingPair (drawer : SockDrawer) : ℚ :=
  let totalSocks := drawer.gray + drawer.white + drawer.black
  let totalPairs := (totalSocks * (totalSocks - 1)) / 2
  let matchingPairs := (drawer.gray * (drawer.gray - 1) + 
                        drawer.white * (drawer.white - 1) + 
                        drawer.black * (drawer.black - 1)) / 2
  matchingPairs / totalPairs

/-- Theorem stating that the probability of picking a matching pair 
    from Tim's sock drawer is 1/3 -/
theorem tims_sock_drawer_probability : 
  probabilityOfMatchingPair ⟨12, 10, 6⟩ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tims_sock_drawer_probability_l2457_245730


namespace NUMINAMATH_CALUDE_intersecting_chords_probability_2023_l2457_245708

/-- Given a circle with 2023 evenly spaced points, this function calculates
    the probability that when selecting four distinct points A, B, C, and D randomly,
    chord AB intersects chord CD and chord AC intersects chord BD. -/
def intersecting_chords_probability (n : ℕ) : ℚ :=
  if n = 2023 then 1/6 else 0

/-- Theorem stating that the probability of the specific chord intersection
    scenario for 2023 points is 1/6. -/
theorem intersecting_chords_probability_2023 :
  intersecting_chords_probability 2023 = 1/6 := by sorry

end NUMINAMATH_CALUDE_intersecting_chords_probability_2023_l2457_245708


namespace NUMINAMATH_CALUDE_unique_solution_for_all_y_l2457_245791

theorem unique_solution_for_all_y : ∃! x : ℝ, ∀ y : ℝ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_y_l2457_245791


namespace NUMINAMATH_CALUDE_sqrt_two_nine_two_equals_six_l2457_245769

theorem sqrt_two_nine_two_equals_six : Real.sqrt (2 * 9 * 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_nine_two_equals_six_l2457_245769


namespace NUMINAMATH_CALUDE_cube_divisors_count_l2457_245764

-- Define a natural number with exactly two prime divisors
def has_two_prime_divisors (n : ℕ) : Prop :=
  ∃ p q α β : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p^α * q^β

-- Define the number of divisors function
noncomputable def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- State the theorem
theorem cube_divisors_count
  (n : ℕ)
  (h1 : has_two_prime_divisors n)
  (h2 : num_divisors (n^2) = 35) :
  num_divisors (n^3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_cube_divisors_count_l2457_245764


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2457_245798

theorem initial_money_calculation (initial_amount : ℚ) : 
  (2/5 : ℚ) * initial_amount = 600 → initial_amount = 1500 := by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l2457_245798


namespace NUMINAMATH_CALUDE_cubic_factorization_l2457_245737

theorem cubic_factorization (m : ℝ) : m^3 - 16*m = m*(m+4)*(m-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2457_245737


namespace NUMINAMATH_CALUDE_inequality_solution_l2457_245787

theorem inequality_solution (a : ℝ) :
  (a < 1/2 → ∀ x, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1-a) ∧
  (a > 1/2 → ∀ x, x^2 - x + a - a^2 < 0 ↔ 1-a < x ∧ x < a) ∧
  (a = 1/2 → ∀ x, ¬(x^2 - x + a - a^2 < 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2457_245787


namespace NUMINAMATH_CALUDE_bicycle_race_finishers_l2457_245784

theorem bicycle_race_finishers :
  let initial_racers : ℕ := 50
  let joined_racers : ℕ := 30
  let dropped_racers : ℕ := 30
  let racers_after_joining := initial_racers + joined_racers
  let racers_after_doubling := 2 * racers_after_joining
  let finishers := racers_after_doubling - dropped_racers
  finishers = 130 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_race_finishers_l2457_245784


namespace NUMINAMATH_CALUDE_modifiedLucas_100th_term_mod_10_l2457_245707

def modifiedLucas : ℕ → ℕ
  | 0 => 2
  | 1 => 5
  | n + 2 => (modifiedLucas n + modifiedLucas (n + 1)) % 10

theorem modifiedLucas_100th_term_mod_10 :
  modifiedLucas 99 % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modifiedLucas_100th_term_mod_10_l2457_245707


namespace NUMINAMATH_CALUDE_incorrect_fraction_equality_l2457_245733

theorem incorrect_fraction_equality (x y : ℝ) (h : x ≠ -y) :
  ¬ ((x - y) / (x + y) = (y - x) / (y + x)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_fraction_equality_l2457_245733


namespace NUMINAMATH_CALUDE_find_y_l2457_245772

theorem find_y (x : ℝ) (y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2457_245772


namespace NUMINAMATH_CALUDE_general_solution_valid_particular_solution_valid_l2457_245751

-- Define the general solution function
def f (C : ℝ) (x : ℝ) : ℝ := x^2 + C

-- Define the particular solution function
def g (x : ℝ) : ℝ := x^2 + 1

-- Theorem for the general solution
theorem general_solution_valid (C : ℝ) : 
  ∀ x, HasDerivAt (f C) (2 * x) x :=
sorry

-- Theorem for the particular solution
theorem particular_solution_valid : 
  g 1 = 2 ∧ ∀ x, HasDerivAt g (2 * x) x :=
sorry

end NUMINAMATH_CALUDE_general_solution_valid_particular_solution_valid_l2457_245751


namespace NUMINAMATH_CALUDE_intersection_M_N_l2457_245767

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {x ∈ U | x^2 ≤ x}

def N : Set Int := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2457_245767


namespace NUMINAMATH_CALUDE_twenty_cows_twenty_days_l2457_245782

/-- The number of bags of husk eaten by a group of cows over a period of days -/
def bags_eaten (num_cows : ℕ) (num_days : ℕ) : ℚ :=
  (num_cows : ℚ) * (num_days : ℚ) * (1 / 20 : ℚ)

/-- Theorem stating that 20 cows eat 20 bags of husk in 20 days -/
theorem twenty_cows_twenty_days : bags_eaten 20 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_cows_twenty_days_l2457_245782


namespace NUMINAMATH_CALUDE_design_area_is_16_l2457_245700

/-- A point on a 2D grid --/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A right-angled triangle on a grid --/
structure RightTriangle where
  vertex1 : GridPoint
  vertex2 : GridPoint
  vertex3 : GridPoint

/-- The design formed by two right-angled triangles --/
structure Design where
  triangle1 : RightTriangle
  triangle2 : RightTriangle

/-- Function to calculate the area of a right-angled triangle using Pick's theorem --/
def triangleArea (t : RightTriangle) : ℕ := sorry

/-- Function to check if a design is symmetrical about the diagonal --/
def isSymmetrical (d : Design) : Prop := sorry

/-- The main theorem --/
theorem design_area_is_16 (d : Design) :
  d.triangle1.vertex1 = ⟨0, 0⟩ ∧
  d.triangle1.vertex2 = ⟨4, 0⟩ ∧
  d.triangle1.vertex3 = ⟨0, 4⟩ ∧
  d.triangle2.vertex1 = ⟨4, 0⟩ ∧
  d.triangle2.vertex2 = ⟨4, 4⟩ ∧
  d.triangle2.vertex3 = ⟨0, 4⟩ ∧
  isSymmetrical d →
  triangleArea d.triangle1 + triangleArea d.triangle2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_design_area_is_16_l2457_245700


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2457_245783

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2457_245783


namespace NUMINAMATH_CALUDE_sum_of_xy_l2457_245770

theorem sum_of_xy (x y : ℕ) 
  (pos_x : x > 0) (pos_y : y > 0)
  (bound_x : x < 30) (bound_y : y < 30)
  (eq : x + y + x * y = 94) : x + y = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xy_l2457_245770


namespace NUMINAMATH_CALUDE_find_a_l2457_245723

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- State the theorem
theorem find_a : ∃ a : ℝ, (A a ∪ B a = {0, 1, 2, 4, 16}) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2457_245723


namespace NUMINAMATH_CALUDE_area_between_curves_l2457_245789

theorem area_between_curves : 
  let f (x : ℝ) := Real.sqrt x
  let g (x : ℝ) := x^2
  ∫ x in (0 : ℝ)..1, (f x - g x) = (1 : ℝ) / 3 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l2457_245789


namespace NUMINAMATH_CALUDE_f_5_equals_2015_l2457_245796

/-- Horner's method representation of a polynomial --/
def horner_poly (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 - 2x^4 + x^3 + x^2 - x - 5 --/
def f (x : ℤ) : ℤ := horner_poly [-5, -1, 1, 1, -2, 1] x

theorem f_5_equals_2015 : f 5 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_2015_l2457_245796


namespace NUMINAMATH_CALUDE_sister_age_2021_l2457_245710

def kelsey_birth_year (kelsey_age_1999 : ℕ) : ℕ := 1999 - kelsey_age_1999

def sister_birth_year (kelsey_birth : ℕ) (age_difference : ℕ) : ℕ := kelsey_birth - age_difference

def current_age (birth_year : ℕ) (current_year : ℕ) : ℕ := current_year - birth_year

theorem sister_age_2021 (kelsey_age_1999 : ℕ) (age_difference : ℕ) (current_year : ℕ) :
  kelsey_age_1999 = 25 →
  age_difference = 3 →
  current_year = 2021 →
  current_age (sister_birth_year (kelsey_birth_year kelsey_age_1999) age_difference) current_year = 50 :=
by sorry

end NUMINAMATH_CALUDE_sister_age_2021_l2457_245710


namespace NUMINAMATH_CALUDE_final_movie_length_l2457_245761

/-- Given an original movie length of 60 minutes and a cut scene of 6 minutes,
    the final movie length is 54 minutes. -/
theorem final_movie_length (original_length cut_length : ℕ) 
  (h1 : original_length = 60)
  (h2 : cut_length = 6) :
  original_length - cut_length = 54 := by
  sorry

end NUMINAMATH_CALUDE_final_movie_length_l2457_245761


namespace NUMINAMATH_CALUDE_bug_triangle_probability_l2457_245739

/-- Probability of the bug being at the starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => (1 - P n) / 2

/-- The bug's movement on an equilateral triangle -/
theorem bug_triangle_probability :
  P 12 = 683 / 2048 :=
by sorry

end NUMINAMATH_CALUDE_bug_triangle_probability_l2457_245739


namespace NUMINAMATH_CALUDE_symmetry_theorem_l2457_245780

/-- The line about which the points are symmetrical -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Defines a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry between two points about a line -/
def symmetric_about_line (P Q : Point) : Prop :=
  -- The midpoint of PQ lies on the symmetry line
  symmetry_line ((P.x + Q.x) / 2) ((P.y + Q.y) / 2) ∧
  -- The slope of PQ is perpendicular to the slope of the symmetry line
  (Q.y - P.y) / (Q.x - P.x) = -1

/-- The theorem to be proved -/
theorem symmetry_theorem (a b : ℝ) :
  let P : Point := ⟨3, 4⟩
  let Q : Point := ⟨a, b⟩
  symmetric_about_line P Q → a = 5 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_theorem_l2457_245780


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_four_satisfies_inequality_four_is_smallest_integer_l2457_245788

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (x : ℚ) / 4 + 3 / 7 > 4 / 3 → x ≥ 4 :=
by
  sorry

theorem four_satisfies_inequality :
  (4 : ℚ) / 4 + 3 / 7 > 4 / 3 :=
by
  sorry

theorem four_is_smallest_integer :
  ∀ x : ℤ, x < 4 → (x : ℚ) / 4 + 3 / 7 ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_four_satisfies_inequality_four_is_smallest_integer_l2457_245788


namespace NUMINAMATH_CALUDE_lucas_pet_beds_lucas_pet_beds_solution_l2457_245727

theorem lucas_pet_beds (initial_beds : ℕ) (beds_per_pet : ℕ) (pets_capacity : ℕ) : ℕ :=
  let total_beds_needed := pets_capacity * beds_per_pet
  let additional_beds := total_beds_needed - initial_beds
  additional_beds

theorem lucas_pet_beds_solution :
  lucas_pet_beds 12 2 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_lucas_pet_beds_lucas_pet_beds_solution_l2457_245727


namespace NUMINAMATH_CALUDE_increase_in_average_age_l2457_245775

/-- Calculates the increase in average age when two men in a group are replaced -/
theorem increase_in_average_age
  (n : ℕ) -- Total number of men
  (age1 age2 : ℕ) -- Ages of the two men being replaced
  (new_avg : ℚ) -- Average age of the two new men
  (h1 : n = 15)
  (h2 : age1 = 21)
  (h3 : age2 = 23)
  (h4 : new_avg = 37) :
  (2 * new_avg - (age1 + age2 : ℚ)) / n = 2 := by sorry

end NUMINAMATH_CALUDE_increase_in_average_age_l2457_245775


namespace NUMINAMATH_CALUDE_distance_difference_l2457_245748

/-- Given distances between locations, prove the difference in total distances -/
theorem distance_difference (orchard_to_house house_to_pharmacy pharmacy_to_school : ℕ) 
  (h1 : orchard_to_house = 800)
  (h2 : house_to_pharmacy = 1300)
  (h3 : pharmacy_to_school = 1700) :
  (orchard_to_house + house_to_pharmacy) - pharmacy_to_school = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2457_245748


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2457_245734

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 120 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ k : ℤ, k ≤ 120 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2457_245734


namespace NUMINAMATH_CALUDE_red_balls_count_l2457_245732

theorem red_balls_count (w r : ℕ) : 
  (w : ℚ) / r = 5 / 3 →  -- ratio of white to red balls
  w + 15 + r = 50 →     -- total after adding 15 white balls
  r = 12 := by           -- number of red balls
sorry

end NUMINAMATH_CALUDE_red_balls_count_l2457_245732


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_i_over_three_l2457_245758

theorem complex_exp_thirteen_pi_i_over_three :
  Complex.exp (13 * π * Complex.I / 3) = (1 / 2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_i_over_three_l2457_245758


namespace NUMINAMATH_CALUDE_widget_production_theorem_l2457_245795

/-- Represents the widget production difference between Monday and Tuesday -/
def widget_production_difference (t : ℝ) : ℝ :=
  let w := 3 * t  -- Monday's production rate
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 3)
  monday_production - tuesday_production

/-- Theorem stating the widget production difference -/
theorem widget_production_theorem (t : ℝ) :
  widget_production_difference t = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_widget_production_theorem_l2457_245795


namespace NUMINAMATH_CALUDE_total_filled_boxes_is_16_l2457_245760

/-- Represents the types of trading cards --/
inductive CardType
  | Magic
  | Rare
  | Common

/-- Represents the types of boxes --/
inductive BoxType
  | Small
  | Large

/-- Defines the capacity of each box type for each card type --/
def boxCapacity (b : BoxType) (c : CardType) : ℕ :=
  match b, c with
  | BoxType.Small, CardType.Magic => 5
  | BoxType.Small, CardType.Rare => 5
  | BoxType.Small, CardType.Common => 6
  | BoxType.Large, CardType.Magic => 10
  | BoxType.Large, CardType.Rare => 10
  | BoxType.Large, CardType.Common => 15

/-- Calculates the number of fully filled boxes of a given type for a specific card type --/
def filledBoxes (cardCount : ℕ) (b : BoxType) (c : CardType) : ℕ :=
  cardCount / boxCapacity b c

/-- The main theorem stating that the total number of fully filled boxes is 16 --/
theorem total_filled_boxes_is_16 :
  let magicCards := 33
  let rareCards := 28
  let commonCards := 33
  let smallBoxesMagic := filledBoxes magicCards BoxType.Small CardType.Magic
  let smallBoxesRare := filledBoxes rareCards BoxType.Small CardType.Rare
  let smallBoxesCommon := filledBoxes commonCards BoxType.Small CardType.Common
  smallBoxesMagic + smallBoxesRare + smallBoxesCommon = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_total_filled_boxes_is_16_l2457_245760


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l2457_245731

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = abs (r₁ - r₂)

/-- Given two circles with radii 5 cm and 3 cm, with centers 2 cm apart,
    prove that they are internally tangent -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 5  -- radius of larger circle
  let r₂ : ℝ := 3  -- radius of smaller circle
  let d  : ℝ := 2  -- distance between centers
  internally_tangent r₁ r₂ d := by
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l2457_245731


namespace NUMINAMATH_CALUDE_foot_to_total_distance_ratio_l2457_245799

/-- Proves that the ratio of distance traveled by foot to total distance is 1:4 -/
theorem foot_to_total_distance_ratio :
  let total_distance : ℝ := 40
  let bus_distance : ℝ := total_distance / 2
  let car_distance : ℝ := 10
  let foot_distance : ℝ := total_distance - bus_distance - car_distance
  foot_distance / total_distance = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_foot_to_total_distance_ratio_l2457_245799


namespace NUMINAMATH_CALUDE_john_popcorn_profit_l2457_245701

/-- Calculates the profit for John's popcorn business --/
theorem john_popcorn_profit :
  let regular_price : ℚ := 4
  let discount_rate : ℚ := 0.1
  let adult_price : ℚ := 8
  let child_price : ℚ := 6
  let packaging_cost : ℚ := 0.5
  let transport_fee : ℚ := 10
  let adult_bags : ℕ := 20
  let child_bags : ℕ := 10
  let total_bags : ℕ := adult_bags + child_bags
  let discounted_price : ℚ := regular_price * (1 - discount_rate)
  let total_cost : ℚ := discounted_price * total_bags + packaging_cost * total_bags + transport_fee
  let total_revenue : ℚ := adult_price * adult_bags + child_price * child_bags
  let profit : ℚ := total_revenue - total_cost
  profit = 87 := by
    sorry

end NUMINAMATH_CALUDE_john_popcorn_profit_l2457_245701


namespace NUMINAMATH_CALUDE_complex_multiplication_l2457_245736

theorem complex_multiplication :
  let i : ℂ := Complex.I
  (1 - 2*i) * (2 + i) = 4 - 3*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2457_245736


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2457_245753

theorem travel_time_calculation (total_time subway_time : ℕ) 
  (h1 : total_time = 38)
  (h2 : subway_time = 10)
  (h3 : total_time = subway_time + 2 * subway_time + (total_time - subway_time - 2 * subway_time)) :
  total_time - subway_time - 2 * subway_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l2457_245753


namespace NUMINAMATH_CALUDE_equation_solutions_l2457_245703

def equation (x : ℝ) : Prop :=
  x ≠ 1 ∧ (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2457_245703


namespace NUMINAMATH_CALUDE_f_one_equals_phi_l2457_245746

noncomputable section

def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the properties of function f
def IsValidF (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 → y > 0 → x < y → f x < f y) ∧ 
  (∀ x, x > 0 → f x * f (f x + 1/x) = 1)

-- State the theorem
theorem f_one_equals_phi (f : ℝ → ℝ) (h : IsValidF f) : f 1 = φ := by
  sorry

end

end NUMINAMATH_CALUDE_f_one_equals_phi_l2457_245746


namespace NUMINAMATH_CALUDE_no_periodic_sum_with_given_periods_l2457_245752

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

/-- The period of a function. -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- Theorem: There do not exist periodic functions g and h with periods 2 and π/2 respectively,
    such that g + h is also a periodic function. -/
theorem no_periodic_sum_with_given_periods :
  ¬ ∃ (g h : ℝ → ℝ),
    Periodic g ∧ Periodic h ∧ Period g 2 ∧ Period h (π / 2) ∧ Periodic (g + h) :=
by sorry

end NUMINAMATH_CALUDE_no_periodic_sum_with_given_periods_l2457_245752


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l2457_245715

theorem unique_solution_cube_equation :
  ∀ (x y z : ℤ), x^3 + 2*y^3 = 4*z^3 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l2457_245715


namespace NUMINAMATH_CALUDE_vector_a_magnitude_l2457_245790

def vector_a : ℝ × ℝ := (3, -2)

theorem vector_a_magnitude : ‖vector_a‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_a_magnitude_l2457_245790


namespace NUMINAMATH_CALUDE_m_intersect_n_l2457_245718

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem m_intersect_n : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_l2457_245718


namespace NUMINAMATH_CALUDE_chess_pawn_placement_l2457_245779

theorem chess_pawn_placement (n : ℕ) (hn : n = 5) : 
  (Finset.card (Finset.univ : Finset (Fin n → Fin n))) * 
  (Finset.card (Finset.univ : Finset (Equiv.Perm (Fin n)))) = 14400 :=
by sorry

end NUMINAMATH_CALUDE_chess_pawn_placement_l2457_245779


namespace NUMINAMATH_CALUDE_fraction_sum_l2457_245721

theorem fraction_sum (p q : ℝ) (h : p ≠ 0 ∧ q ≠ 0) 
  (h1 : 1/p + 1/q = 1/(p+q)) : p/q + q/p = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2457_245721


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l2457_245716

def is_reducible (n : ℕ) : Prop :=
  (n > 15) ∧ (Nat.gcd (n - 15) (3 * n + 4) > 1)

theorem least_reducible_fraction :
  ∀ k : ℕ, k < 22 → ¬(is_reducible k) ∧ is_reducible 22 :=
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l2457_245716


namespace NUMINAMATH_CALUDE_no_m_exists_for_subset_l2457_245745

theorem no_m_exists_for_subset : ¬ ∃ m : ℝ, m > 1 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 4 → 1 - m ≤ x ∧ x ≤ 3 * m - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_subset_l2457_245745


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l2457_245766

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ)
  (h_blue : blue = 7)
  (h_red : red = 5)
  (h_yellow : yellow = 4) :
  let total := blue + red + yellow
  let p_blue := blue / total
  let p_red := red / total
  let p_yellow := yellow / total
  let p_not_blue := (red + yellow) / total
  let p_not_red := (blue + yellow) / total
  let p_not_yellow := (blue + red) / total
  p_blue * p_not_blue + p_red * p_not_red + p_yellow * p_not_yellow = 83 / 128 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l2457_245766


namespace NUMINAMATH_CALUDE_trapezoid_bases_l2457_245725

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  -- The lengths of the two bases
  base1 : ℝ
  base2 : ℝ
  -- The length of the side
  side : ℝ
  -- The ratio of areas divided by the midline
  areaRatio : ℝ
  -- Conditions
  side_length : side = 3 ∨ side = 5
  inscribable : base1 + base2 = 2 * side
  area_ratio : areaRatio = 5 / 11

/-- The theorem stating the lengths of the bases -/
theorem trapezoid_bases (t : IsoscelesTrapezoid) : 
  (t.base1 = 1 ∧ t.base2 = 7) ∨ (t.base1 = 7 ∧ t.base2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l2457_245725


namespace NUMINAMATH_CALUDE_initial_members_family_e_l2457_245755

/-- The number of families in Indira Nagar -/
def num_families : ℕ := 6

/-- The initial number of members in family a -/
def family_a : ℕ := 7

/-- The initial number of members in family b -/
def family_b : ℕ := 8

/-- The initial number of members in family c -/
def family_c : ℕ := 10

/-- The initial number of members in family d -/
def family_d : ℕ := 13

/-- The initial number of members in family f -/
def family_f : ℕ := 10

/-- The number of members that left each family -/
def members_left : ℕ := 1

/-- The average number of members in each family after some left -/
def new_average : ℕ := 8

/-- The initial number of members in family e -/
def family_e : ℕ := 6

theorem initial_members_family_e :
  family_a + family_b + family_c + family_d + family_e + family_f - 
  (num_families * members_left) = num_families * new_average := by
  sorry

end NUMINAMATH_CALUDE_initial_members_family_e_l2457_245755


namespace NUMINAMATH_CALUDE_least_possible_x_l2457_245706

theorem least_possible_x (x y z : ℤ) : 
  (∃ k : ℤ, x = 2 * k) →  -- x is even
  (∃ m : ℤ, y = 2 * m + 1) →  -- y is odd
  (∃ n : ℤ, z = 2 * n + 1) →  -- z is odd
  y - x > 5 →
  z - x ≥ 9 →
  (∀ w : ℤ, (∃ j : ℤ, w = 2 * j) → w ≥ x) →
  x = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_x_l2457_245706


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2457_245776

def A : Set ℤ := {0, 1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 0, 1, 6}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2457_245776


namespace NUMINAMATH_CALUDE_correlation_coefficients_relation_l2457_245709

def X : List ℝ := [16, 18, 20, 22]
def Y : List ℝ := [15.10, 12.81, 9.72, 3.21]
def U : List ℝ := [10, 20, 30]
def V : List ℝ := [7.5, 9.5, 16.6]

def r1 : ℝ := sorry
def r2 : ℝ := sorry

theorem correlation_coefficients_relation : r1 < 0 ∧ 0 < r2 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficients_relation_l2457_245709


namespace NUMINAMATH_CALUDE_expression_simplification_l2457_245720

theorem expression_simplification (a b : ℤ) : 
  (a = 1) → (b = -a) → 3*a^2*b + 2*(a*b - 3/2*a^2*b) - (2*a*b^2 - (3*a*b^2 - a*b)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2457_245720


namespace NUMINAMATH_CALUDE_james_daily_trips_l2457_245735

/-- The number of bags James can carry per trip -/
def bags_per_trip : ℕ := 10

/-- The total number of bags James delivers in 5 days -/
def total_bags : ℕ := 1000

/-- The number of days James works -/
def total_days : ℕ := 5

/-- The number of trips James takes each day -/
def trips_per_day : ℕ := total_bags / (bags_per_trip * total_days)

theorem james_daily_trips : trips_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_daily_trips_l2457_245735


namespace NUMINAMATH_CALUDE_fraction_of_one_third_is_one_fifth_l2457_245756

theorem fraction_of_one_third_is_one_fifth : (1 : ℚ) / 5 / ((1 : ℚ) / 3) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_one_third_is_one_fifth_l2457_245756


namespace NUMINAMATH_CALUDE_plaid_shirts_count_l2457_245738

/-- Prove that the number of plaid shirts is 3 -/
theorem plaid_shirts_count (total_shirts : ℕ) (total_pants : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ) : ℕ :=
  by
  have total_items : ℕ := total_shirts + total_pants
  have plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
  have plaid_shirts : ℕ := plaid_or_purple - purple_pants
  exact plaid_shirts

#check plaid_shirts_count 5 24 5 21

end NUMINAMATH_CALUDE_plaid_shirts_count_l2457_245738


namespace NUMINAMATH_CALUDE_arun_weight_upper_limit_l2457_245719

theorem arun_weight_upper_limit (arun_lower : ℝ) (arun_upper : ℝ) 
  (brother_lower : ℝ) (brother_upper : ℝ) (mother_upper : ℝ) (average : ℝ) :
  arun_lower = 61 →
  arun_upper = 72 →
  brother_lower = 60 →
  brother_upper = 70 →
  average = 63 →
  arun_lower < mother_upper →
  mother_upper ≤ brother_upper →
  (arun_lower + mother_upper) / 2 = average →
  mother_upper = 65 := by
sorry

end NUMINAMATH_CALUDE_arun_weight_upper_limit_l2457_245719


namespace NUMINAMATH_CALUDE_unique_value_not_in_range_l2457_245711

/-- A function f with specific properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The theorem stating the properties of f and its unique value not in its range -/
theorem unique_value_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h19 : f a b c d 19 = 19)
  (h97 : f a b c d 97 = 97)
  (hinv : ∀ x, x ≠ -d/c → f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 58 :=
sorry

end NUMINAMATH_CALUDE_unique_value_not_in_range_l2457_245711
