import Mathlib

namespace NUMINAMATH_CALUDE_student_number_problem_l2705_270554

theorem student_number_problem (x : ℝ) : 2 * x - 200 = 110 → x = 155 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2705_270554


namespace NUMINAMATH_CALUDE_five_by_five_uncoverable_l2705_270555

/-- A checkerboard that can be completely covered by dominoes. -/
structure CoverableCheckerboard where
  rows : ℕ
  cols : ℕ
  even_rows : Even rows
  even_cols : Even cols
  even_total : Even (rows * cols)

/-- A domino covers exactly two squares. -/
def domino_covers : ℕ := 2

/-- Theorem stating that a 5x5 checkerboard cannot be completely covered by dominoes. -/
theorem five_by_five_uncoverable :
  ¬ ∃ (c : CoverableCheckerboard), c.rows = 5 ∧ c.cols = 5 :=
sorry

end NUMINAMATH_CALUDE_five_by_five_uncoverable_l2705_270555


namespace NUMINAMATH_CALUDE_base4_divisibility_by_17_l2705_270572

def base4_to_decimal (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4^1 + d * 4^0

def is_base4_digit (x : ℕ) : Prop :=
  x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3

theorem base4_divisibility_by_17 (x : ℕ) :
  is_base4_digit x →
  (base4_to_decimal 2 3 x 2 ∣ 17) ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_base4_divisibility_by_17_l2705_270572


namespace NUMINAMATH_CALUDE_magical_stack_with_79_fixed_l2705_270568

/-- A stack of cards is magical if it satisfies certain conditions -/
def magical_stack (n : ℕ) : Prop :=
  ∃ (card_position : ℕ → ℕ),
    (∀ i, i ≤ 2*n → card_position i ≤ 2*n) ∧
    (∃ i ≤ n, card_position i = i) ∧
    (∃ i > n, i ≤ 2*n ∧ card_position i = i) ∧
    (∀ i ≤ 2*n, i % 2 = 1 → card_position i ≤ n) ∧
    (∀ i ≤ 2*n, i % 2 = 0 → card_position i > n)

theorem magical_stack_with_79_fixed (n : ℕ) :
  magical_stack n ∧ n ≥ 79 ∧ (∃ card_position : ℕ → ℕ, card_position 79 = 79) →
  2 * n = 236 :=
sorry

end NUMINAMATH_CALUDE_magical_stack_with_79_fixed_l2705_270568


namespace NUMINAMATH_CALUDE_min_fraction_sum_l2705_270525

theorem min_fraction_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  1 ≤ a ∧ a ≤ 10 →
  1 ≤ b ∧ b ≤ 10 →
  1 ≤ c ∧ c ≤ 10 →
  1 ≤ d ∧ d ≤ 10 →
  (a : ℚ) / b + (c : ℚ) / d ≥ 14 / 45 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l2705_270525


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l2705_270597

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem fourth_term_of_sequence (a : ℕ → ℝ) :
  a 1 = 1 → (∀ n, a (n + 1) = 2 * a n) → a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l2705_270597


namespace NUMINAMATH_CALUDE_impossible_coverage_l2705_270576

/-- Represents a 5x5 board --/
def Board := Fin 5 → Fin 5 → Bool

/-- Represents a stromino (3x1 rectangle) --/
structure Stromino where
  start_row : Fin 5
  start_col : Fin 5
  is_horizontal : Bool

/-- Checks if a stromino is valid (fits within the board) --/
def is_valid_stromino (s : Stromino) : Bool :=
  if s.is_horizontal then
    s.start_col < 3
  else
    s.start_row < 3

/-- Counts how many strominos cover a given square --/
def count_coverage (board : Board) (strominos : List Stromino) (row col : Fin 5) : Nat :=
  sorry

/-- Checks if a given arrangement of strominos is valid --/
def is_valid_arrangement (strominos : List Stromino) : Bool :=
  sorry

/-- The main theorem stating that it's impossible to cover the board with 16 strominos --/
theorem impossible_coverage : ¬ ∃ (strominos : List Stromino),
  strominos.length = 16 ∧
  is_valid_arrangement strominos ∧
  ∀ (row col : Fin 5),
    let coverage := count_coverage (λ _ _ => true) strominos row col
    coverage = 1 ∨ coverage = 2 :=
  sorry

end NUMINAMATH_CALUDE_impossible_coverage_l2705_270576


namespace NUMINAMATH_CALUDE_incircle_radius_eq_area_div_semiperimeter_l2705_270521

/-- Triangle DEF with given angles and side length --/
structure TriangleDEF where
  D : ℝ
  E : ℝ
  F : ℝ
  DE : ℝ
  angle_D_eq : D = 75
  angle_E_eq : E = 45
  angle_F_eq : F = 60
  side_DE_eq : DE = 20

/-- The radius of the incircle of triangle DEF --/
def incircle_radius (t : TriangleDEF) : ℝ := sorry

/-- The semi-perimeter of triangle DEF --/
def semi_perimeter (t : TriangleDEF) : ℝ := sorry

/-- The area of triangle DEF --/
def triangle_area (t : TriangleDEF) : ℝ := sorry

/-- Theorem: The radius of the incircle is equal to the area divided by the semi-perimeter --/
theorem incircle_radius_eq_area_div_semiperimeter (t : TriangleDEF) :
  incircle_radius t = triangle_area t / semi_perimeter t := by sorry

end NUMINAMATH_CALUDE_incircle_radius_eq_area_div_semiperimeter_l2705_270521


namespace NUMINAMATH_CALUDE_prob_four_of_a_kind_after_reroll_l2705_270528

/-- Represents the outcome of rolling five dice -/
structure DiceRoll where
  pairs : Nat -- Number of pairs
  fourOfAKind : Bool -- Whether there's a four-of-a-kind

/-- Represents the possible outcomes after re-rolling the fifth die -/
inductive ReRollOutcome
  | fourOfAKind : ReRollOutcome
  | nothingSpecial : ReRollOutcome

/-- The probability of getting at least four of a kind after re-rolling -/
def probFourOfAKind (initialRoll : DiceRoll) : ℚ :=
  sorry

theorem prob_four_of_a_kind_after_reroll :
  ∀ (initialRoll : DiceRoll),
    initialRoll.pairs = 2 ∧ ¬initialRoll.fourOfAKind →
    probFourOfAKind initialRoll = 1 / 3 :=
  sorry

end NUMINAMATH_CALUDE_prob_four_of_a_kind_after_reroll_l2705_270528


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2705_270520

theorem vector_magnitude_proof (OA AB : ℂ) (h1 : OA = -2 + I) (h2 : AB = 3 + 2*I) :
  Complex.abs (OA + AB) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2705_270520


namespace NUMINAMATH_CALUDE_anne_drawings_per_marker_l2705_270590

/-- Given:
  * Anne has 12 markers
  * She has already made 8 drawings
  * She can make 10 more drawings before running out of markers
  Prove that Anne can make 1.5 drawings with one marker -/
theorem anne_drawings_per_marker (markers : ℕ) (made_drawings : ℕ) (remaining_drawings : ℕ) 
  (h1 : markers = 12)
  (h2 : made_drawings = 8)
  (h3 : remaining_drawings = 10) :
  (made_drawings + remaining_drawings : ℚ) / markers = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_anne_drawings_per_marker_l2705_270590


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l2705_270516

/-- Represents a system of pipes filling or draining a tank -/
structure PipeSystem where
  fill_time_1 : ℝ  -- Time for first pipe to fill the tank
  drain_time : ℝ   -- Time for drain pipe to empty the tank
  combined_time : ℝ -- Time to fill the tank with all pipes open
  fill_time_2 : ℝ  -- Time for second pipe to fill the tank (to be proven)

/-- The theorem stating the relationship between the pipes' fill times -/
theorem second_pipe_fill_time (ps : PipeSystem) 
  (h1 : ps.fill_time_1 = 5)
  (h2 : ps.drain_time = 20)
  (h3 : ps.combined_time = 2.5) : 
  ps.fill_time_2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_second_pipe_fill_time_l2705_270516


namespace NUMINAMATH_CALUDE_pencils_on_desk_l2705_270595

theorem pencils_on_desk (drawer : ℕ) (added : ℕ) (total : ℕ) (initial : ℕ) : 
  drawer = 43 → added = 16 → total = 78 → initial + added + drawer = total → initial = 19 := by
sorry

end NUMINAMATH_CALUDE_pencils_on_desk_l2705_270595


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l2705_270540

variable (a b : ℝ)
variable (u : ℕ → ℝ)

def arithmetic_geometric_sequence (u : ℕ → ℝ) (a b : ℝ) : Prop :=
  ∀ n, u (n + 1) = a * u n + b

theorem arithmetic_geometric_sequence_closed_form
  (h : arithmetic_geometric_sequence u a b) :
  ∀ n, u n = a^n * u 0 + (a^n - 1) / (a - 1) * b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_closed_form_l2705_270540


namespace NUMINAMATH_CALUDE_min_value_of_a_l2705_270537

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

theorem min_value_of_a (a : ℝ) (h : A ∩ B a = ∅) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2705_270537


namespace NUMINAMATH_CALUDE_sandy_payment_l2705_270573

def amount_paid (football_cost baseball_cost change : ℚ) : ℚ :=
  football_cost + baseball_cost + change

theorem sandy_payment (football_cost baseball_cost change : ℚ) 
  (h1 : football_cost = 9.14)
  (h2 : baseball_cost = 6.81)
  (h3 : change = 4.05) :
  amount_paid football_cost baseball_cost change = 20 :=
by sorry

end NUMINAMATH_CALUDE_sandy_payment_l2705_270573


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2705_270549

theorem cubic_equation_solution (x y z a : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ z ≠ x →
  x^3 + a = -3*(y + z) →
  y^3 + a = -3*(x + z) →
  z^3 + a = -3*(x + y) →
  a ∈ Set.Ioo (-2 : ℝ) 2 \ {0} := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2705_270549


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l2705_270501

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 3 50 = 149 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l2705_270501


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2705_270513

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2705_270513


namespace NUMINAMATH_CALUDE_circles_M_N_common_tangents_l2705_270506

/-- Circle M with equation x^2 + y^2 - 4y = 0 -/
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.2 = 0}

/-- Circle N with equation (x - 1)^2 + (y - 1)^2 = 1 -/
def circle_N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The number of common tangents between two circles -/
def num_common_tangents (C1 C2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem stating that circles M and N have exactly 2 common tangents -/
theorem circles_M_N_common_tangents :
  num_common_tangents circle_M circle_N = 2 :=
sorry

end NUMINAMATH_CALUDE_circles_M_N_common_tangents_l2705_270506


namespace NUMINAMATH_CALUDE_school_bus_capacity_l2705_270514

/-- The number of rows of seats in the school bus -/
def num_rows : ℕ := 20

/-- The number of kids that can sit in each row -/
def kids_per_row : ℕ := 4

/-- The total number of kids that can ride the school bus -/
def total_capacity : ℕ := num_rows * kids_per_row

theorem school_bus_capacity : total_capacity = 80 := by
  sorry

end NUMINAMATH_CALUDE_school_bus_capacity_l2705_270514


namespace NUMINAMATH_CALUDE_f_inverse_composition_l2705_270551

def f : Fin 6 → Fin 6
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 5
| 5 => 3
| 6 => 1

theorem f_inverse_composition (h : Function.Bijective f) :
  (Function.invFun f (Function.invFun f (Function.invFun f 6))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_inverse_composition_l2705_270551


namespace NUMINAMATH_CALUDE_solve_equation_l2705_270509

/-- Proves that for x = 3.3333333333333335, the equation √((x * y) / 3) = x is satisfied when y = 10 -/
theorem solve_equation (x : ℝ) (y : ℝ) (h1 : x = 3.3333333333333335) (h2 : y = 10) :
  Real.sqrt ((x * y) / 3) = x := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2705_270509


namespace NUMINAMATH_CALUDE_tan_sum_identity_l2705_270594

theorem tan_sum_identity (α : Real) (h : Real.tan α = Real.sqrt 2) :
  Real.tan (α + π / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l2705_270594


namespace NUMINAMATH_CALUDE_roots_reality_l2705_270507

theorem roots_reality (p q : ℝ) (h : p^2 - 4*q > 0) :
  ∀ a : ℝ, (2*a + 3*p)^2 - 4*3*(q + a*p) > 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_reality_l2705_270507


namespace NUMINAMATH_CALUDE_family_savings_correct_l2705_270561

def income_tax_rate : ℝ := 0.13

def ivan_salary : ℝ := 55000
def vasilisa_salary : ℝ := 45000
def vasilisa_mother_salary : ℝ := 18000
def vasilisa_father_salary : ℝ := 20000
def son_state_stipend : ℝ := 3000
def son_non_state_stipend : ℝ := 15000

def vasilisa_mother_pension : ℝ := 10000

def monthly_expenses : ℝ := 74000

def net_income (gross_income : ℝ) : ℝ :=
  gross_income * (1 - income_tax_rate)

def total_income_before_may2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income vasilisa_mother_salary + net_income vasilisa_father_salary + 
  son_state_stipend

def total_income_may_to_aug2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  vasilisa_mother_pension + net_income vasilisa_father_salary + 
  son_state_stipend

def total_income_from_sep2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  vasilisa_mother_pension + net_income vasilisa_father_salary + 
  son_state_stipend + net_income son_non_state_stipend

theorem family_savings_correct :
  (total_income_before_may2018 - monthly_expenses = 49060) ∧
  (total_income_may_to_aug2018 - monthly_expenses = 43400) ∧
  (total_income_from_sep2018 - monthly_expenses = 56450) := by
  sorry

end NUMINAMATH_CALUDE_family_savings_correct_l2705_270561


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l2705_270557

theorem smallest_absolute_value : ∀ (a b c : ℝ),
  a = 4.1 → b = 13 → c = 3 →
  |(-Real.sqrt 7)| < |a| ∧ |(-Real.sqrt 7)| < Real.sqrt b ∧ |(-Real.sqrt 7)| < |c| :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l2705_270557


namespace NUMINAMATH_CALUDE_volleyball_advancement_l2705_270560

def can_advance (k : ℕ) (t : ℕ) : Prop :=
  t ≤ k ∧ t * (t - 1) ≤ 2 * t * 1 ∧ (k - t) * (k - t - 1) ≥ 2 * (k - t) * (1 - t + 1)

theorem volleyball_advancement (k : ℕ) (h : k = 5 ∨ k = 6) :
  ∃ t : ℕ, t ≥ 0 ∧ t ≤ 3 ∧ can_advance k t :=
sorry

end NUMINAMATH_CALUDE_volleyball_advancement_l2705_270560


namespace NUMINAMATH_CALUDE_square_roots_problem_l2705_270541

theorem square_roots_problem (a : ℝ) : 
  (a + 3) ^ 2 = (2 * a - 6) ^ 2 → (a + 3) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2705_270541


namespace NUMINAMATH_CALUDE_unit_digit_of_large_exponentiation_l2705_270571

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_of_large_exponentiation : 
  unit_digit ((23^100000 * 56^150000) / Nat.gcd 23 56) = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_large_exponentiation_l2705_270571


namespace NUMINAMATH_CALUDE_butane_molecular_weight_l2705_270522

/-- The molecular weight of Butane in grams per mole. -/
def molecular_weight_butane : ℝ := 65

/-- The number of moles used in the given condition. -/
def given_moles : ℝ := 4

/-- The total molecular weight of the given moles in grams. -/
def given_total_weight : ℝ := 260

/-- Theorem stating that the molecular weight of Butane is 65 grams/mole. -/
theorem butane_molecular_weight : 
  molecular_weight_butane = given_total_weight / given_moles := by
  sorry

end NUMINAMATH_CALUDE_butane_molecular_weight_l2705_270522


namespace NUMINAMATH_CALUDE_percentage_without_full_time_jobs_survey_result_l2705_270529

theorem percentage_without_full_time_jobs 
  (mother_ratio : Real) 
  (father_ratio : Real) 
  (women_ratio : Real) : Real :=
  let total_parents := 100
  let women_count := women_ratio * total_parents
  let men_count := total_parents - women_count
  let employed_women := mother_ratio * women_count
  let employed_men := father_ratio * men_count
  let total_employed := employed_women + employed_men
  let unemployed := total_parents - total_employed
  unemployed / total_parents * 100

theorem survey_result : 
  percentage_without_full_time_jobs (5/6) (3/4) 0.6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_without_full_time_jobs_survey_result_l2705_270529


namespace NUMINAMATH_CALUDE_alpha_value_l2705_270562

theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 18) = Real.sqrt 3 / 2) : α = π / 180 * 70 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2705_270562


namespace NUMINAMATH_CALUDE_power_calculation_l2705_270566

theorem power_calculation : 3^2022 * (1/3)^2023 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2705_270566


namespace NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_has_12_sides_l2705_270559

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_deg_interior_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_deg_interior_has_12_sides_l2705_270559


namespace NUMINAMATH_CALUDE_regular_tetrahedron_edge_sum_plus_three_l2705_270535

/-- A regular tetrahedron is a tetrahedron with all faces congruent equilateral triangles. -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The number of edges in a tetrahedron -/
def tetrahedronEdgeCount : ℕ := 6

/-- Calculate the sum of edge lengths of a regular tetrahedron plus an additional length -/
def sumEdgeLengthsPlusExtra (t : RegularTetrahedron) (extra : ℝ) : ℝ :=
  (t.sideLength * tetrahedronEdgeCount : ℝ) + extra

/-- Theorem: For a regular tetrahedron with side length 3 cm, 
    the sum of its edge lengths plus 3 cm is 21 cm -/
theorem regular_tetrahedron_edge_sum_plus_three
  (t : RegularTetrahedron)
  (h : t.sideLength = 3) :
  sumEdgeLengthsPlusExtra t 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_edge_sum_plus_three_l2705_270535


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2705_270574

/-- Given an arithmetic sequence with common difference 2 where a₁, a₂, a₅ form a geometric sequence, prove a₂ = 3 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 1 * a 5 = a 2 * a 2) →     -- a₁, a₂, a₅ form a geometric sequence
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2705_270574


namespace NUMINAMATH_CALUDE_wine_drinking_time_is_correct_l2705_270564

/-- Represents the time taken for three assistants to drink 40 liters of wine -/
def wine_drinking_time : ℚ :=
  let rate1 := (40 : ℚ) / 12  -- Rate of the first assistant
  let rate2 := (40 : ℚ) / 10  -- Rate of the second assistant
  let rate3 := (40 : ℚ) / 8   -- Rate of the third assistant
  let total_rate := rate1 + rate2 + rate3
  (40 : ℚ) / total_rate

/-- The wine drinking time is equal to 3 9/37 hours -/
theorem wine_drinking_time_is_correct : wine_drinking_time = 3 + 9 / 37 := by
  sorry

#eval wine_drinking_time

end NUMINAMATH_CALUDE_wine_drinking_time_is_correct_l2705_270564


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2705_270593

/-- 
Theorem: Area increase of a rectangle
Given a rectangle with its length increased by 30% and width increased by 15%,
prove that the area increases by 49.5%.
-/
theorem rectangle_area_increase : 
  ∀ (l w : ℝ), l > 0 → w > 0 → 
  (1.3 * l) * (1.15 * w) = 1.495 * (l * w) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2705_270593


namespace NUMINAMATH_CALUDE_exists_periodic_functions_with_nonperiodic_difference_l2705_270548

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ (∃ p > 0, ∀ x, f (x + p) = f x)

/-- The period of a function. -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

/-- Theorem stating that there exist periodic functions g and h with periods 6 and 2π respectively,
    such that their difference is not periodic. -/
theorem exists_periodic_functions_with_nonperiodic_difference :
  ∃ (g h : ℝ → ℝ),
    IsPeriodic g ∧ Period g 6 ∧
    IsPeriodic h ∧ Period h (2 * Real.pi) ∧
    ¬IsPeriodic (g - h) := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_functions_with_nonperiodic_difference_l2705_270548


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l2705_270553

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 4 -/
def curve (p : Point) : Prop := p.x * p.y = 4

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

theorem fourth_intersection_point (c : Circle) 
    (h1 : curve (Point.mk 4 1) ∧ onCircle (Point.mk 4 1) c)
    (h2 : curve (Point.mk (-2) (-2)) ∧ onCircle (Point.mk (-2) (-2)) c)
    (h3 : curve (Point.mk 8 (1/2)) ∧ onCircle (Point.mk 8 (1/2)) c)
    (h4 : ∃ p : Point, curve p ∧ onCircle p c ∧ p ≠ Point.mk 4 1 ∧ p ≠ Point.mk (-2) (-2) ∧ p ≠ Point.mk 8 (1/2)) :
    ∃ p : Point, p = Point.mk (-1/4) (-16) ∧ curve p ∧ onCircle p c := by
  sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l2705_270553


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2705_270583

theorem units_digit_of_product (n : ℕ) : 
  (2^2101 * 5^2102 * 11^2103) % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2705_270583


namespace NUMINAMATH_CALUDE_unique_equal_sums_l2705_270526

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem unique_equal_sums : ∃! (n : ℕ), n > 0 ∧ 
  arithmetic_sum 3 7 n = arithmetic_sum 5 3 n := by sorry

end NUMINAMATH_CALUDE_unique_equal_sums_l2705_270526


namespace NUMINAMATH_CALUDE_match_committee_count_l2705_270589

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members the host team contributes to the committee -/
def host_contribution : ℕ := 4

/-- The number of members each non-host team contributes to the committee -/
def non_host_contribution : ℕ := 3

/-- The total size of the match committee -/
def committee_size : ℕ := 13

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem match_committee_count :
  num_teams * (choose team_size host_contribution) * (choose team_size non_host_contribution) ^ (num_teams - 1) = 262609375 := by
  sorry

end NUMINAMATH_CALUDE_match_committee_count_l2705_270589


namespace NUMINAMATH_CALUDE_millet_percentage_in_mix_l2705_270505

/-- Theorem: Millet percentage in a birdseed mix -/
theorem millet_percentage_in_mix
  (brand_a_millet : ℝ)
  (brand_b_millet : ℝ)
  (mix_brand_a : ℝ)
  (h1 : brand_a_millet = 0.60)
  (h2 : brand_b_millet = 0.65)
  (h3 : mix_brand_a = 0.60)
  (h4 : 0 ≤ mix_brand_a ∧ mix_brand_a ≤ 1) :
  mix_brand_a * brand_a_millet + (1 - mix_brand_a) * brand_b_millet = 0.62 := by
  sorry


end NUMINAMATH_CALUDE_millet_percentage_in_mix_l2705_270505


namespace NUMINAMATH_CALUDE_shooting_probabilities_l2705_270547

/-- The probability of A hitting the target -/
def prob_A_hit : ℚ := 2/3

/-- The probability of B hitting the target -/
def prob_B_hit : ℚ := 3/4

/-- The probability that A shoots 3 times and misses at least once -/
def prob_A_miss_at_least_once : ℚ := 19/27

/-- The probability that A shoots twice and hits both times, while B shoots twice and hits exactly once -/
def prob_A_hit_twice_B_hit_once : ℚ := 1/6

theorem shooting_probabilities 
  (hA : prob_A_hit = 2/3)
  (hB : prob_B_hit = 3/4)
  (indep : ∀ (n : ℕ) (m : ℕ), (prob_A_hit ^ n) * ((1 - prob_A_hit) ^ (m - n)) = 
    (2/3 ^ n) * ((1/3) ^ (m - n))) :
  prob_A_miss_at_least_once = 19/27 ∧ 
  prob_A_hit_twice_B_hit_once = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l2705_270547


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2705_270563

/-- The x-intercept of the line 4x + 7y = 28 is the point (7, 0). -/
theorem x_intercept_of_line (x y : ℝ) : 
  (4 * x + 7 * y = 28) → (x = 7 ∧ y = 0 → 4 * x + 7 * y = 28) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2705_270563


namespace NUMINAMATH_CALUDE_triangle_value_l2705_270542

theorem triangle_value (triangle r : ℝ) 
  (h1 : triangle + r = 72)
  (h2 : (triangle + r) + r = 117) : 
  triangle = 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l2705_270542


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2705_270546

def a : ℝ × ℝ := (1, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)
def c : ℝ × ℝ := (1, 2)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 + (b x).1) * c.1 + (a.2 + (b x).2) * c.2 = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2705_270546


namespace NUMINAMATH_CALUDE_jeans_cost_theorem_l2705_270518

def total_cost : ℕ := 110
def coat_cost : ℕ := 40
def shoes_cost : ℕ := 30
def num_jeans : ℕ := 2

theorem jeans_cost_theorem :
  ∃ (jeans_cost : ℕ),
    jeans_cost * num_jeans + coat_cost + shoes_cost = total_cost ∧
    jeans_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_jeans_cost_theorem_l2705_270518


namespace NUMINAMATH_CALUDE_perfectSquareFactorsOf1800_l2705_270527

/-- The number of positive factors of 1800 that are perfect squares -/
def perfectSquareFactors : ℕ := 8

/-- 1800 as a natural number -/
def n : ℕ := 1800

/-- A function that returns the number of positive factors of a natural number that are perfect squares -/
def countPerfectSquareFactors (m : ℕ) : ℕ := sorry

theorem perfectSquareFactorsOf1800 : countPerfectSquareFactors n = perfectSquareFactors := by sorry

end NUMINAMATH_CALUDE_perfectSquareFactorsOf1800_l2705_270527


namespace NUMINAMATH_CALUDE_complex_power_sum_l2705_270502

theorem complex_power_sum : ∃ (i : ℂ), i^2 = -1 ∧ (1 - i)^2016 + (1 + i)^2016 = 2^1009 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2705_270502


namespace NUMINAMATH_CALUDE_spinner_probability_l2705_270515

-- Define the spinners
def spinner_C : Finset ℕ := {1, 3, 5, 7}
def spinner_D : Finset ℕ := {2, 4, 6}

-- Define the probability space
def Ω : Finset (ℕ × ℕ) := spinner_C.product spinner_D

-- Define the event of sum being divisible by 3
def E : Finset (ℕ × ℕ) := Ω.filter (λ p => (p.1 + p.2) % 3 = 0)

theorem spinner_probability : 
  (Finset.card E : ℚ) / (Finset.card Ω : ℚ) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_spinner_probability_l2705_270515


namespace NUMINAMATH_CALUDE_infinitely_many_with_1989_ones_l2705_270552

/-- Count the number of ones in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The theorem stating that there are infinitely many positive integers
    with 1989 ones in their binary representation -/
theorem infinitely_many_with_1989_ones :
  ∀ k : ℕ, ∃ m : ℕ, m > k ∧ countOnes m = 1989 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_with_1989_ones_l2705_270552


namespace NUMINAMATH_CALUDE_reservoir_ratio_l2705_270581

theorem reservoir_ratio : 
  ∀ (total_capacity normal_level end_month_amount : ℝ),
  end_month_amount = 6 →
  end_month_amount = 0.6 * total_capacity →
  normal_level = total_capacity - 5 →
  end_month_amount / normal_level = 1.2 := by
sorry

end NUMINAMATH_CALUDE_reservoir_ratio_l2705_270581


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2705_270539

theorem cubic_equation_solution (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 24*x^3) 
  (h3 : a - b = x) : 
  a = (x*(3 + Real.sqrt 92))/6 ∨ a = (x*(3 - Real.sqrt 92))/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2705_270539


namespace NUMINAMATH_CALUDE_pencils_remaining_l2705_270536

theorem pencils_remaining (x : ℕ) : ℕ :=
  let initial_pencils_per_child : ℕ := 2
  let number_of_children : ℕ := 15
  let total_initial_pencils : ℕ := initial_pencils_per_child * number_of_children
  let pencils_given_away : ℕ := number_of_children * x
  total_initial_pencils - pencils_given_away

#check pencils_remaining

end NUMINAMATH_CALUDE_pencils_remaining_l2705_270536


namespace NUMINAMATH_CALUDE_jerry_money_left_l2705_270578

/-- The amount of money Jerry has left after grocery shopping -/
def money_left (mustard_oil_price : ℝ) (mustard_oil_quantity : ℝ)
                (pasta_price : ℝ) (pasta_quantity : ℝ)
                (sauce_price : ℝ) (sauce_quantity : ℝ)
                (initial_money : ℝ) : ℝ :=
  initial_money - (mustard_oil_price * mustard_oil_quantity +
                   pasta_price * pasta_quantity +
                   sauce_price * sauce_quantity)

/-- Theorem stating that Jerry has $7 left after grocery shopping -/
theorem jerry_money_left :
  money_left 13 2 4 3 5 1 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_left_l2705_270578


namespace NUMINAMATH_CALUDE_max_value_of_a_l2705_270587

theorem max_value_of_a (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (product_sum_condition : a * b + a * c + b * c = 11) :
  a ≤ 2 + (2 * Real.sqrt 15) / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2705_270587


namespace NUMINAMATH_CALUDE_group_size_l2705_270544

/-- The number of people in the group -/
def N : ℕ := sorry

/-- The weight of the person being replaced (in kg) -/
def original_weight : ℕ := 65

/-- The weight of the new person (in kg) -/
def new_weight : ℕ := 89

/-- The average weight increase (in kg) -/
def avg_increase : ℕ := 3

theorem group_size :
  (new_weight - original_weight : ℤ) = N * avg_increase →
  N = 8 := by sorry

end NUMINAMATH_CALUDE_group_size_l2705_270544


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2705_270592

theorem min_value_of_sum (x y : ℝ) : 
  x > 1 → 
  y > 1 → 
  2 * Real.log 2 = Real.log x + Real.log y → 
  x + y ≥ 200 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ 2 * Real.log 2 = Real.log x + Real.log y ∧ x + y = 200 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2705_270592


namespace NUMINAMATH_CALUDE_cubic_function_property_l2705_270508

/-- Given a cubic function f(x) = ax³ + bx - 2 where f(2015) = 7, prove that f(-2015) = -11 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 2
  f 2015 = 7 → f (-2015) = -11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2705_270508


namespace NUMINAMATH_CALUDE_train_speed_l2705_270585

/-- The speed of a train passing through a tunnel -/
theorem train_speed (train_length : Real) (tunnel_length : Real) (time_minutes : Real) :
  train_length = 0.1 →
  tunnel_length = 2.9 →
  time_minutes = 2.5 →
  ∃ (speed : Real), abs (speed - 71.94) < 0.01 ∧ 
    speed = (tunnel_length + train_length) / (time_minutes / 60) := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l2705_270585


namespace NUMINAMATH_CALUDE_base_conversion_difference_l2705_270598

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100000) * 7^5 +
  ((n / 10000) % 10) * 7^4 +
  ((n / 1000) % 10) * 7^3 +
  ((n / 100) % 10) * 7^2 +
  ((n / 10) % 10) * 7^1 +
  (n % 10) * 7^0

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 10000) * 8^4 +
  ((n / 1000) % 10) * 8^3 +
  ((n / 100) % 10) * 8^2 +
  ((n / 10) % 10) * 8^1 +
  (n % 10) * 8^0

theorem base_conversion_difference :
  base7ToBase10 543210 - base8ToBase10 43210 = 76717 := by
  sorry

#eval base7ToBase10 543210 - base8ToBase10 43210

end NUMINAMATH_CALUDE_base_conversion_difference_l2705_270598


namespace NUMINAMATH_CALUDE_total_fruits_is_41_l2705_270524

/-- The number of oranges Mike received -/
def mike_oranges : ℕ := 3

/-- The number of apples Matt received -/
def matt_apples : ℕ := 2 * mike_oranges

/-- The number of bananas Mark received -/
def mark_bananas : ℕ := mike_oranges + matt_apples

/-- The number of grapes Mary received -/
def mary_grapes : ℕ := mike_oranges + matt_apples + mark_bananas + 5

/-- The total number of fruits received by all four children -/
def total_fruits : ℕ := mike_oranges + matt_apples + mark_bananas + mary_grapes

theorem total_fruits_is_41 : total_fruits = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_41_l2705_270524


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_specific_geometric_series_sum_l2705_270550

/-- The sum of an infinite geometric series with first term a and common ratio r is a / (1 - r),
    given that |r| < 1 -/
theorem infinite_geometric_series_sum (a r : ℚ) (h : |r| < 1) :
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

/-- The sum of the specific infinite geometric series is 10/9 -/
theorem specific_geometric_series_sum :
  let a : ℚ := 5/3
  let r : ℚ := -1/2
  (∑' n, a * r^n) = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_specific_geometric_series_sum_l2705_270550


namespace NUMINAMATH_CALUDE_min_value_sum_l2705_270543

theorem min_value_sum (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ a₀ + 3 * b₀ + 9 * c₀ = 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l2705_270543


namespace NUMINAMATH_CALUDE_intersection_A_B_l2705_270545

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≥ 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2705_270545


namespace NUMINAMATH_CALUDE_cube_root_of_a_times_sqrt_a_l2705_270523

theorem cube_root_of_a_times_sqrt_a (a : ℝ) (ha : a > 0) : 
  (a * a^(1/2))^(1/3) = a^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_times_sqrt_a_l2705_270523


namespace NUMINAMATH_CALUDE_eunji_gymnastics_count_l2705_270575

/-- Represents the position of a student in a rectangular arrangement -/
structure StudentPosition where
  leftColumn : Nat
  rightColumn : Nat
  frontRow : Nat
  backRow : Nat

/-- Calculates the total number of students in a rectangular arrangement -/
def totalStudents (pos : StudentPosition) : Nat :=
  let totalColumns := pos.leftColumn + pos.rightColumn - 1
  let totalRows := pos.frontRow + pos.backRow - 1
  totalColumns * totalRows

/-- Theorem: Given Eunji's position, the total number of students is 441 -/
theorem eunji_gymnastics_count :
  let eunjiPosition : StudentPosition := {
    leftColumn := 8,
    rightColumn := 14,
    frontRow := 7,
    backRow := 15
  }
  totalStudents eunjiPosition = 441 := by
  sorry

end NUMINAMATH_CALUDE_eunji_gymnastics_count_l2705_270575


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2705_270538

/-- Given a school with the following student composition:
    - Total number of boys: 650
    - 44% are Muslims
    - 28% are Hindus
    - 117 boys are from other communities
    This theorem proves that 10% of the boys are Sikhs. -/
theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 650 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 117 →
  (total - (muslim_percent * total + hindu_percent * total + other)) / total = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2705_270538


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2705_270586

/-- Represents a trapezoid ABCD with given side lengths and a right angle at BCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  AD : ℝ
  angle_BCD_is_right : Bool

/-- Calculate the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.CD + t.BC + t.AD

/-- Theorem: The perimeter of the given trapezoid is 118 units -/
theorem trapezoid_perimeter : 
  ∀ (t : Trapezoid), 
  t.AB = 33 ∧ t.CD = 15 ∧ t.BC = 45 ∧ t.AD = 25 ∧ t.angle_BCD_is_right = true → 
  perimeter t = 118 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2705_270586


namespace NUMINAMATH_CALUDE_negative_square_inequality_l2705_270533

theorem negative_square_inequality (a b : ℝ) : a < b → b < 0 → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_inequality_l2705_270533


namespace NUMINAMATH_CALUDE_B_subset_A_l2705_270532

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2*x = 0}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l2705_270532


namespace NUMINAMATH_CALUDE_sum_and_equal_after_changes_l2705_270503

theorem sum_and_equal_after_changes (a b c : ℝ) 
  (sum_eq : a + b + c = 120)
  (equal_after_changes : a + 4 = b - 12 ∧ b - 12 = 3 * c) :
  b = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_equal_after_changes_l2705_270503


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l2705_270500

/-- A quadratic function of the form y = -(x+1)² + c -/
def quadratic_function (c : ℝ) (x : ℝ) : ℝ := -(x + 1)^2 + c

theorem quadratic_point_ordering (c : ℝ) :
  let y₁ := quadratic_function c (-13/4)
  let y₂ := quadratic_function c (-1)
  let y₃ := quadratic_function c 0
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l2705_270500


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2705_270584

theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 1875 → ¬(∃ a : ℕ, 3 * m = a ^ 2) ∨ ¬(∃ b : ℕ, 5 * m = b ^ 3)) ∧ 
  (∃ a : ℕ, 3 * 1875 = a ^ 2) ∧ 
  (∃ b : ℕ, 5 * 1875 = b ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2705_270584


namespace NUMINAMATH_CALUDE_parabolas_intersection_l2705_270588

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 4 * x - 7
def parabola2 (x : ℝ) : ℝ := x^2 + 5

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(-2, 9), (2, 9)}

-- Theorem statement
theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, p ∈ intersection_points ↔
    (parabola1 p.1 = parabola2 p.1 ∧ p.2 = parabola1 p.1) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l2705_270588


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2705_270511

theorem polynomial_simplification (s r : ℝ) :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2705_270511


namespace NUMINAMATH_CALUDE_probability_same_length_is_33_105_l2705_270512

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of longer diagonals in a regular hexagon -/
def num_longer_diagonals : ℕ := 6

/-- The number of shorter diagonals in a regular hexagon -/
def num_shorter_diagonals : ℕ := 3

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := Finset.range (num_sides + num_diagonals)

/-- The probability of selecting two segments of the same length from S -/
def probability_same_length : ℚ :=
  (Nat.choose num_sides 2 + Nat.choose num_longer_diagonals 2 + Nat.choose num_shorter_diagonals 2) /
  Nat.choose S.card 2

theorem probability_same_length_is_33_105 :
  probability_same_length = 33 / 105 := by sorry

end NUMINAMATH_CALUDE_probability_same_length_is_33_105_l2705_270512


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2705_270519

theorem complex_fraction_equality : 1 + 1 / (2 + 1 / 3) = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2705_270519


namespace NUMINAMATH_CALUDE_not_sum_of_three_squares_l2705_270558

theorem not_sum_of_three_squares (n : ℕ) : ¬ ∃ (a b c : ℕ+), (8 * n - 1 : ℤ) = a ^ 2 + b ^ 2 + c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_sum_of_three_squares_l2705_270558


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l2705_270504

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)

theorem tangent_slope_angle (x : ℝ) : 
  let slope := (deriv f) 1
  Real.arctan slope = π / 4 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l2705_270504


namespace NUMINAMATH_CALUDE_house_transaction_gain_l2705_270570

/-- Calculates the net gain from selling a house at a profit and buying it back at a loss --/
def net_gain (initial_worth : ℝ) (sell_profit_percent : ℝ) (buy_loss_percent : ℝ) : ℝ :=
  let sell_price := initial_worth * (1 + sell_profit_percent)
  let buy_back_price := sell_price * (1 - buy_loss_percent)
  sell_price - buy_back_price

/-- Theorem stating that selling a $12,000 house at 20% profit and buying it back at 15% loss results in $2,160 gain --/
theorem house_transaction_gain :
  net_gain 12000 0.2 0.15 = 2160 := by
  sorry

end NUMINAMATH_CALUDE_house_transaction_gain_l2705_270570


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2705_270582

theorem sin_2theta_value (θ : Real) (h : Real.cos θ + Real.sin θ = 3/2) : 
  Real.sin (2 * θ) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2705_270582


namespace NUMINAMATH_CALUDE_pond_filling_time_l2705_270534

/-- Represents the time needed to fill a pond under specific conditions. -/
def time_to_fill_pond (initial_flow_ratio : ℚ) (initial_fill_ratio : ℚ) (initial_days : ℚ) : ℚ :=
  let total_volume := 18 * initial_flow_ratio * initial_days / initial_fill_ratio
  let remaining_volume := total_volume * (1 - initial_fill_ratio)
  remaining_volume / 1

theorem pond_filling_time :
  let initial_flow_ratio : ℚ := 3/4
  let initial_fill_ratio : ℚ := 2/3
  let initial_days : ℚ := 16
  time_to_fill_pond initial_flow_ratio initial_fill_ratio initial_days = 6 := by
  sorry

#eval time_to_fill_pond (3/4) (2/3) 16

end NUMINAMATH_CALUDE_pond_filling_time_l2705_270534


namespace NUMINAMATH_CALUDE_regular_price_correct_l2705_270530

/-- The regular price of one tire -/
def regular_price : ℝ := 108

/-- The sale price for three tires -/
def sale_price : ℝ := 270

/-- The theorem stating that the regular price of one tire is correct given the sale conditions -/
theorem regular_price_correct : 
  2 * regular_price + regular_price / 2 = sale_price :=
by sorry

end NUMINAMATH_CALUDE_regular_price_correct_l2705_270530


namespace NUMINAMATH_CALUDE_insects_in_lab_l2705_270580

/-- The number of insects in a laboratory given the total number of legs --/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem: The number of insects in the laboratory is 5 --/
theorem insects_in_lab : number_of_insects 30 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_insects_in_lab_l2705_270580


namespace NUMINAMATH_CALUDE_patrick_less_than_twice_greg_l2705_270599

def homework_hours (jacob greg patrick : ℕ) : Prop :=
  jacob = 18 ∧ 
  greg = jacob - 6 ∧ 
  jacob + greg + patrick = 50

theorem patrick_less_than_twice_greg : 
  ∀ jacob greg patrick : ℕ, 
  homework_hours jacob greg patrick → 
  2 * greg - patrick = 4 := by
sorry

end NUMINAMATH_CALUDE_patrick_less_than_twice_greg_l2705_270599


namespace NUMINAMATH_CALUDE_expression_simplification_l2705_270531

theorem expression_simplification (a : ℝ) (h : a = 2) :
  (a^2 / (a - 1) - a) / ((a + a^2) / (1 - 2*a + a^2)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2705_270531


namespace NUMINAMATH_CALUDE_room_length_is_25_l2705_270556

/-- Represents the dimensions and whitewashing details of a room --/
structure Room where
  length : ℝ
  width : ℝ
  height : ℝ
  whitewashCost : ℝ
  doorArea : ℝ
  windowArea : ℝ
  totalCost : ℝ

/-- Calculates the whitewashable area of the room --/
def whitewashableArea (r : Room) : ℝ :=
  2 * (r.length * r.height + r.width * r.height) - r.doorArea - 3 * r.windowArea

/-- Theorem stating that the room length is 25 feet given the specified conditions --/
theorem room_length_is_25 (r : Room) 
    (h1 : r.width = 15)
    (h2 : r.height = 12)
    (h3 : r.whitewashCost = 6)
    (h4 : r.doorArea = 18)
    (h5 : r.windowArea = 12)
    (h6 : r.totalCost = 5436)
    (h7 : r.totalCost = r.whitewashCost * whitewashableArea r) :
    r.length = 25 := by
  sorry


end NUMINAMATH_CALUDE_room_length_is_25_l2705_270556


namespace NUMINAMATH_CALUDE_hole_pattern_symmetry_l2705_270596

/-- Represents a rectangular piece of paper --/
structure Paper where
  length : ℝ
  width : ℝ

/-- Represents a point on the paper --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold operation --/
inductive Fold
  | LeftToRight
  | TopToBottom
  | Diagonal

/-- Represents the hole pattern after unfolding --/
inductive HolePattern
  | SymmetricAll
  | SingleCenter
  | VerticalOnly
  | HorizontalOnly

/-- Performs a series of folds on the paper --/
def foldPaper (p : Paper) (folds : List Fold) : Paper :=
  sorry

/-- Punches a hole at a specific location on the folded paper --/
def punchHole (p : Paper) (loc : Point) : Paper :=
  sorry

/-- Unfolds the paper and determines the resulting hole pattern --/
def unfoldAndAnalyze (p : Paper) : HolePattern :=
  sorry

/-- Main theorem: The hole pattern is symmetrical across all axes --/
theorem hole_pattern_symmetry 
  (initialPaper : Paper)
  (folds : List Fold)
  (holeLocation : Point) :
  initialPaper.length = 8 ∧ 
  initialPaper.width = 4 ∧
  folds = [Fold.LeftToRight, Fold.TopToBottom, Fold.Diagonal] ∧
  holeLocation.x = 1/4 ∧ 
  holeLocation.y = 3/4 →
  unfoldAndAnalyze (punchHole (foldPaper initialPaper folds) holeLocation) = HolePattern.SymmetricAll :=
by
  sorry

end NUMINAMATH_CALUDE_hole_pattern_symmetry_l2705_270596


namespace NUMINAMATH_CALUDE_problem_solution_l2705_270565

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a + 1 / c = 7) (h6 : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2705_270565


namespace NUMINAMATH_CALUDE_triangle_midpoint_sum_l2705_270569

theorem triangle_midpoint_sum (a b c : ℝ) : 
  a + b + c = 15 → 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_midpoint_sum_l2705_270569


namespace NUMINAMATH_CALUDE_new_jasmine_percentage_l2705_270510

/-- Calculates the new jasmine percentage in a solution after adding jasmine and water -/
theorem new_jasmine_percentage
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 90)
  (h2 : initial_jasmine_percentage = 5)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 2) :
  let initial_jasmine := initial_volume * (initial_jasmine_percentage / 100)
  let new_jasmine := initial_jasmine + added_jasmine
  let new_volume := initial_volume + added_jasmine + added_water
  let new_percentage := (new_jasmine / new_volume) * 100
  new_percentage = 12.5 := by
sorry

end NUMINAMATH_CALUDE_new_jasmine_percentage_l2705_270510


namespace NUMINAMATH_CALUDE_acute_triangles_in_cuboid_l2705_270517

-- Define a rectangular cuboid
structure RectangularCuboid where
  vertices : Finset (ℝ × ℝ × ℝ)
  vertex_count : vertices.card = 8

-- Define a function to count acute triangles
def count_acute_triangles (rc : RectangularCuboid) : ℕ := sorry

-- Theorem statement
theorem acute_triangles_in_cuboid (rc : RectangularCuboid) :
  count_acute_triangles rc = 8 := by sorry

end NUMINAMATH_CALUDE_acute_triangles_in_cuboid_l2705_270517


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2705_270567

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 4 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 9 / (5 + 6 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2705_270567


namespace NUMINAMATH_CALUDE_good_number_decomposition_l2705_270577

/-- A natural number is good if it can be represented as the product of two consecutive natural numbers. -/
def is_good (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

/-- Theorem: Any good number greater than 6 can be represented as the sum of a good number and a number that is 3 times a good number. -/
theorem good_number_decomposition (a : ℕ) (h1 : is_good a) (h2 : a > 6) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * (y * (y + 1)) :=
sorry

end NUMINAMATH_CALUDE_good_number_decomposition_l2705_270577


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2705_270579

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∃ (x y : ℝ), mx + 2*y + 1 = 0 ∧ x - m^2*y + 1/2 = 0) →
  (m * 1 + 2 * (-m^2) = 0) →
  (m = 0 ∨ m = 1/2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2705_270579


namespace NUMINAMATH_CALUDE_difference_of_squares_640_360_l2705_270591

theorem difference_of_squares_640_360 : 640^2 - 360^2 = 280000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_640_360_l2705_270591
