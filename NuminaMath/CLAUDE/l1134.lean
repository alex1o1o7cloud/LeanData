import Mathlib

namespace NUMINAMATH_CALUDE_profit_sharing_problem_l1134_113417

/-- Given a profit shared between two partners in the ratio 2:5, where the second partner
    receives $2500, prove that the first partner will have $800 left after spending $200. -/
theorem profit_sharing_problem (total_parts : ℕ) (first_partner_parts second_partner_parts : ℕ) 
    (second_partner_share : ℕ) (shirt_cost : ℕ) :
  total_parts = first_partner_parts + second_partner_parts →
  first_partner_parts = 2 →
  second_partner_parts = 5 →
  second_partner_share = 2500 →
  shirt_cost = 200 →
  (first_partner_parts * second_partner_share / second_partner_parts) - shirt_cost = 800 :=
by sorry


end NUMINAMATH_CALUDE_profit_sharing_problem_l1134_113417


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1134_113429

/-- Given that 5y varies inversely as the square of x, and y = 4 when x = 2, 
    prove that y = 1 when x = 4 -/
theorem inverse_variation_problem (k : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → 5 * y = k / (x ^ 2)) →
  (5 * 4 = k / (2 ^ 2)) →
  ∃ y : ℝ, 5 * y = k / (4 ^ 2) ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1134_113429


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l1134_113461

theorem opposite_of_negative_one_half :
  -(-(1/2)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l1134_113461


namespace NUMINAMATH_CALUDE_combined_salaries_of_abce_l1134_113456

def average_salary : ℕ := 8800
def number_of_people : ℕ := 5
def d_salary : ℕ := 7000

theorem combined_salaries_of_abce :
  (average_salary * number_of_people) - d_salary = 37000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_of_abce_l1134_113456


namespace NUMINAMATH_CALUDE_rectangle_width_l1134_113486

/-- Given a rectangle with area 1638 square inches, where ten such rectangles
    would have a total length of 390 inches, prove that its width is 42 inches. -/
theorem rectangle_width (area : ℝ) (total_length : ℝ) (h1 : area = 1638) 
    (h2 : total_length = 390) : ∃ (width : ℝ), width = 42 ∧ 
    ∃ (length : ℝ), area = length * width ∧ total_length = 10 * length :=
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1134_113486


namespace NUMINAMATH_CALUDE_best_player_hits_l1134_113495

/-- Represents a baseball team -/
structure BaseballTeam where
  totalPlayers : ℕ
  averageHitsPerGame : ℕ
  gamesPlayed : ℕ
  otherPlayersAverageHits : ℕ
  otherPlayersGames : ℕ

/-- Calculates the total hits of the best player -/
def bestPlayerTotalHits (team : BaseballTeam) : ℕ :=
  team.averageHitsPerGame * team.gamesPlayed - 
  (team.totalPlayers - 1) * team.otherPlayersAverageHits

/-- Theorem stating the best player's total hits -/
theorem best_player_hits (team : BaseballTeam) 
  (h1 : team.totalPlayers = 11)
  (h2 : team.averageHitsPerGame = 15)
  (h3 : team.gamesPlayed = 5)
  (h4 : team.otherPlayersAverageHits = 6)
  (h5 : team.otherPlayersGames = 6) :
  bestPlayerTotalHits team = 25 := by
  sorry

#eval bestPlayerTotalHits { 
  totalPlayers := 11, 
  averageHitsPerGame := 15, 
  gamesPlayed := 5, 
  otherPlayersAverageHits := 6, 
  otherPlayersGames := 6
}

end NUMINAMATH_CALUDE_best_player_hits_l1134_113495


namespace NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l1134_113440

/-- The cost of blackberry jam used in Elmo's sandwiches -/
theorem elmo_sandwich_jam_cost :
  ∀ (N B J : ℕ),
    N > 1 →
    B > 0 →
    J > 0 →
    N * (6 * B + 7 * J) = 396 →
    (N * J * 7 : ℚ) / 100 = 378 / 100 := by
  sorry

end NUMINAMATH_CALUDE_elmo_sandwich_jam_cost_l1134_113440


namespace NUMINAMATH_CALUDE_fraction_inequality_l1134_113438

theorem fraction_inequality (a b : ℚ) (h : a / b = 2 / 3) :
  ¬(∀ (x y : ℚ), x / y = 2 / 3 → x / y = (x + 2) / (y + 2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1134_113438


namespace NUMINAMATH_CALUDE_nine_by_n_grid_rectangles_l1134_113457

theorem nine_by_n_grid_rectangles (n : ℕ) : 
  (9 : ℕ) > 1 → n > 1 → (Nat.choose 9 2 * Nat.choose n 2 = 756) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_nine_by_n_grid_rectangles_l1134_113457


namespace NUMINAMATH_CALUDE_problem_statement_l1134_113439

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_eq : a^2 + b^2 = a + b) : 
  ((a + b)^2 ≤ 2*(a^2 + b^2)) ∧ ((a + 1)*(b + 1) ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1134_113439


namespace NUMINAMATH_CALUDE_half_full_one_minute_before_end_l1134_113434

/-- Represents the filling process of a box with marbles -/
def FillingProcess (total_time : ℕ) : Type :=
  ℕ → ℝ

/-- The quantity doubles every minute -/
def DoublesEveryMinute (process : FillingProcess n) : Prop :=
  ∀ t, t < n → process (t + 1) = 2 * process t

/-- The process is complete at the total time -/
def CompleteAtEnd (process : FillingProcess n) : Prop :=
  process n = 1

/-- The box is half full at a given time -/
def HalfFullAt (process : FillingProcess n) (t : ℕ) : Prop :=
  process t = 1/2

theorem half_full_one_minute_before_end 
  (process : FillingProcess 10) 
  (h1 : DoublesEveryMinute process) 
  (h2 : CompleteAtEnd process) :
  HalfFullAt process 9 :=
sorry

end NUMINAMATH_CALUDE_half_full_one_minute_before_end_l1134_113434


namespace NUMINAMATH_CALUDE_age_difference_is_six_l1134_113408

-- Define Claire's future age
def claire_future_age : ℕ := 20

-- Define the number of years until Claire reaches her future age
def years_until_future : ℕ := 2

-- Define Jessica's current age
def jessica_current_age : ℕ := 24

-- Theorem to prove
theorem age_difference_is_six :
  jessica_current_age - (claire_future_age - years_until_future) = 6 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_is_six_l1134_113408


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1134_113430

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- First line equation: 3y - 3b = 9x -/
def line1 (x y b : ℝ) : Prop := 3 * y - 3 * b = 9 * x

/-- Second line equation: y - 2 = (b + 9)x -/
def line2 (x y b : ℝ) : Prop := y - 2 = (b + 9) * x

theorem perpendicular_lines_b_value :
  ∀ b : ℝ, (∃ x y : ℝ, line1 x y b ∧ line2 x y b ∧
    perpendicular 3 (b + 9)) → b = -28/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1134_113430


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l1134_113413

/-- The complex number z = (1+2i)/(1-2i) is in the second quadrant -/
theorem complex_in_second_quadrant : 
  let z : ℂ := (1 + 2*I) / (1 - 2*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l1134_113413


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1134_113415

theorem absolute_value_equation_solutions : 
  {x : ℝ | x + 1 = |x + 3| - |x - 1|} = {3, -1, -5} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1134_113415


namespace NUMINAMATH_CALUDE_opposite_expressions_l1134_113475

theorem opposite_expressions (x : ℝ) : (x + 1) + (3 * x - 5) = 0 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_l1134_113475


namespace NUMINAMATH_CALUDE_f_equals_g_l1134_113442

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1
def g (t : ℝ) : ℝ := 2 * t - 1

-- Theorem stating that f and g are the same function
theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1134_113442


namespace NUMINAMATH_CALUDE_unique_odd_natural_from_primes_l1134_113496

theorem unique_odd_natural_from_primes :
  ∃! (n : ℕ), 
    n % 2 = 1 ∧ 
    ∃ (p q : ℕ), 
      Prime p ∧ Prime q ∧ p > q ∧ 
      n = (p + q) / (p - q) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_odd_natural_from_primes_l1134_113496


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_two_l1134_113494

theorem set_equality_implies_a_equals_two (A B : Set ℕ) (a : ℕ) 
  (h1 : A = {1, 2})
  (h2 : B = {1, a})
  (h3 : A = B) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_two_l1134_113494


namespace NUMINAMATH_CALUDE_range_of_a_l1134_113462

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  (-1 ≤ a ∧ a < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1134_113462


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l1134_113460

theorem sqrt_plus_square_zero_implies_diff (x y : ℝ) :
  Real.sqrt (y - 3) + (2 * x - 4)^2 = 0 → 2 * x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l1134_113460


namespace NUMINAMATH_CALUDE_square_difference_formula_l1134_113423

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1134_113423


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_35_l1134_113418

-- Define a function that represents angles with the same terminal side as a given angle
def sameTerminalSide (angle : ℝ) : ℤ → ℝ := fun k => k * 360 + angle

-- Theorem statement
theorem angle_with_same_terminal_side_as_negative_35 :
  ∃ (x : ℝ), 0 ≤ x ∧ x < 360 ∧ ∃ (k : ℤ), x = sameTerminalSide (-35) k ∧ x = 325 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_as_negative_35_l1134_113418


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1134_113481

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ z : ℂ, z = a + 1 - a * Complex.I ∧ z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1134_113481


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_l1134_113474

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- The event of drawing a specific combination of balls -/
structure Event where
  red : ℕ
  white : ℕ

/-- The bag in the problem -/
def problem_bag : Bag := { red := 5, white := 5 }

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- The event of drawing 3 red balls -/
def event_all_red : Event := { red := 3, white := 0 }

/-- The event of drawing at least 1 white ball -/
def event_at_least_one_white : Event := { red := drawn_balls - 1, white := 1 }

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (e1 e2 : Event) : Prop :=
  e1.red + e1.white = drawn_balls ∧ e2.red + e2.white = drawn_balls ∧ 
  (e1.red + e2.red > problem_bag.red ∨ e1.white + e2.white > problem_bag.white)

/-- The main theorem stating that the two events are mutually exclusive -/
theorem events_mutually_exclusive : 
  mutually_exclusive event_all_red event_at_least_one_white :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_l1134_113474


namespace NUMINAMATH_CALUDE_sean_houses_problem_l1134_113459

theorem sean_houses_problem (initial_houses : ℕ) : 
  initial_houses - 8 + 12 = 31 → initial_houses = 27 := by
  sorry

end NUMINAMATH_CALUDE_sean_houses_problem_l1134_113459


namespace NUMINAMATH_CALUDE_johns_country_club_payment_l1134_113427

/-- Represents the cost John pays for the country club membership in the first year -/
def johns_payment (num_members : ℕ) (joining_fee_pp : ℕ) (monthly_cost_pp : ℕ) : ℕ :=
  let total_joining_fee := num_members * joining_fee_pp
  let total_monthly_cost := num_members * monthly_cost_pp * 12
  let total_cost := total_joining_fee + total_monthly_cost
  total_cost / 2

/-- Proves that John's payment for the first year is $32000 given the problem conditions -/
theorem johns_country_club_payment :
  johns_payment 4 4000 1000 = 32000 := by
sorry

end NUMINAMATH_CALUDE_johns_country_club_payment_l1134_113427


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1134_113425

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (x_ge_2 : x ≥ 2)
  (y_ge_2 : y ≥ 2)
  (z_ge_2 : z ≥ 2) :
  ∃ (max : ℝ), max = Real.sqrt 69 ∧ 
    ∀ a b c : ℝ, a + b + c = 7 → a ≥ 2 → b ≥ 2 → c ≥ 2 →
      Real.sqrt (2 * a + 3) + Real.sqrt (2 * b + 3) + Real.sqrt (2 * c + 3) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1134_113425


namespace NUMINAMATH_CALUDE_smallest_M_for_inequality_l1134_113468

theorem smallest_M_for_inequality : 
  ∃ M : ℝ, (∀ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧ 
  (∀ M' : ℝ, (∀ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_M_for_inequality_l1134_113468


namespace NUMINAMATH_CALUDE_time_per_braid_l1134_113409

/-- The time it takes to braid one braid, given the number of dancers, braids per dancer, and total time -/
theorem time_per_braid (num_dancers : ℕ) (braids_per_dancer : ℕ) (total_time_minutes : ℕ) : 
  num_dancers = 8 → 
  braids_per_dancer = 5 → 
  total_time_minutes = 20 → 
  (total_time_minutes * 60) / (num_dancers * braids_per_dancer) = 30 := by
  sorry

#check time_per_braid

end NUMINAMATH_CALUDE_time_per_braid_l1134_113409


namespace NUMINAMATH_CALUDE_triangle_side_relation_l1134_113458

theorem triangle_side_relation (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (angle_B : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l1134_113458


namespace NUMINAMATH_CALUDE_sum_15_is_120_l1134_113406

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℚ
  /-- The common difference of the sequence -/
  d : ℚ
  /-- The sum of the first 5 terms is 10 -/
  sum_5 : (5 : ℚ) / 2 * (2 * a₁ + 4 * d) = 10
  /-- The sum of the first 10 terms is 50 -/
  sum_10 : (10 : ℚ) / 2 * (2 * a₁ + 9 * d) = 50

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a₁ + (n - 1 : ℚ) * seq.d)

/-- Theorem: The sum of the first 15 terms is 120 -/
theorem sum_15_is_120 (seq : ArithmeticSequence) : sum_n seq 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_15_is_120_l1134_113406


namespace NUMINAMATH_CALUDE_line_slope_l1134_113493

/-- Given a line passing through points (1,2) and (4,2+√3), its slope is √3/3 -/
theorem line_slope : ∃ (k : ℝ), k = (2 + Real.sqrt 3 - 2) / (4 - 1) ∧ k = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1134_113493


namespace NUMINAMATH_CALUDE_photocopy_discount_is_25_percent_l1134_113401

/-- The discount percentage for bulk photocopy orders -/
def discount_percentage (cost_per_copy : ℚ) (copies_for_discount : ℕ) 
  (steve_copies : ℕ) (dinley_copies : ℕ) (individual_savings : ℚ) : ℚ :=
  let total_copies := steve_copies + dinley_copies
  let total_cost_without_discount := cost_per_copy * total_copies
  let total_savings := individual_savings * 2
  let total_cost_with_discount := total_cost_without_discount - total_savings
  (total_cost_without_discount - total_cost_with_discount) / total_cost_without_discount * 100

theorem photocopy_discount_is_25_percent :
  discount_percentage 0.02 100 80 80 0.40 = 25 := by
  sorry

end NUMINAMATH_CALUDE_photocopy_discount_is_25_percent_l1134_113401


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1134_113466

theorem consecutive_integers_product (n : ℤ) : 
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1134_113466


namespace NUMINAMATH_CALUDE_triangle_determinant_l1134_113478

theorem triangle_determinant (A B C : Real) (h1 : A + B + C = π) (h2 : A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2) : 
  let M : Matrix (Fin 3) (Fin 3) Real := !![2*Real.sin A, 1, 1; 1, 2*Real.sin B, 1; 1, 1, 2*Real.sin C]
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_l1134_113478


namespace NUMINAMATH_CALUDE_sales_tax_difference_l1134_113435

-- Define the item price
def item_price : ℝ := 50

-- Define the tax rates
def tax_rate_high : ℝ := 0.075
def tax_rate_low : ℝ := 0.07

-- Define the sales tax calculation function
def sales_tax (price : ℝ) (rate : ℝ) : ℝ := price * rate

-- Theorem statement
theorem sales_tax_difference :
  sales_tax item_price tax_rate_high - sales_tax item_price tax_rate_low = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_sales_tax_difference_l1134_113435


namespace NUMINAMATH_CALUDE_capacity_variation_l1134_113464

/-- Given positive constants e, R, and r, prove that the function C(n) = en / (R + nr^2) 
    first increases and then decreases as n increases. -/
theorem capacity_variation (e R r : ℝ) (he : e > 0) (hR : R > 0) (hr : r > 0) :
  ∃ n₀ : ℝ, n₀ > 0 ∧
    (∀ n₁ n₂ : ℝ, 0 < n₁ ∧ n₁ < n₂ ∧ n₂ < n₀ → 
      (e * n₁) / (R + n₁ * r^2) < (e * n₂) / (R + n₂ * r^2)) ∧
    (∀ n₁ n₂ : ℝ, n₀ < n₁ ∧ n₁ < n₂ → 
      (e * n₁) / (R + n₁ * r^2) > (e * n₂) / (R + n₂ * r^2)) :=
sorry

end NUMINAMATH_CALUDE_capacity_variation_l1134_113464


namespace NUMINAMATH_CALUDE_inequality_proof_l1134_113453

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a + b + c) + Real.sqrt a) / (b + c) +
  (Real.sqrt (a + b + c) + Real.sqrt b) / (c + a) +
  (Real.sqrt (a + b + c) + Real.sqrt c) / (a + b) ≥
  (9 + 3 * Real.sqrt 3) / (2 * Real.sqrt (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1134_113453


namespace NUMINAMATH_CALUDE_circle_equation_with_given_endpoints_l1134_113437

/-- The standard equation of a circle with diameter endpoints M(2,0) and N(0,4) -/
theorem circle_equation_with_given_endpoints :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ↔ 
  ∃ (t : ℝ), x = 2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 4 * t ∧ 0 ≤ t ∧ t ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_endpoints_l1134_113437


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_75_degrees_l1134_113480

/-- Given a 75-degree angle, prove that the degree measure of the supplement of its complement is 165°. -/
theorem supplement_of_complement_of_75_degrees :
  let angle : ℝ := 75
  let complement : ℝ := 90 - angle
  let supplement : ℝ := 180 - complement
  supplement = 165 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_75_degrees_l1134_113480


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1134_113465

theorem arithmetic_calculations :
  (12 - (-18) + (-7) - 20 = 3) ∧ (-4 / (1/2) * 8 = -64) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1134_113465


namespace NUMINAMATH_CALUDE_g_value_proof_l1134_113444

def nabla (g h : ℝ) : ℝ := g^2 - h^2

theorem g_value_proof (g : ℝ) (h_pos : g > 0) (h_eq : nabla g 6 = 45) : g = 9 := by
  sorry

end NUMINAMATH_CALUDE_g_value_proof_l1134_113444


namespace NUMINAMATH_CALUDE_three_color_theorem_l1134_113448

/-- Represents a country on the island -/
structure Country where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents the entire island -/
structure Island where
  countries : List Country
  adjacent : Country → Country → Bool

/-- A coloring of the island -/
def Coloring := Country → Fin 3

theorem three_color_theorem (I : Island) :
  ∃ (c : Coloring), ∀ (x y : Country),
    I.adjacent x y → c x ≠ c y :=
sorry

end NUMINAMATH_CALUDE_three_color_theorem_l1134_113448


namespace NUMINAMATH_CALUDE_total_amc8_students_l1134_113433

/-- Represents a math class at Euclid Middle School -/
structure MathClass where
  teacher : String
  totalStudents : Nat
  olympiadStudents : Nat

/-- Calculates the number of students in a class taking only AMC 8 -/
def studentsOnlyAMC8 (c : MathClass) : Nat :=
  c.totalStudents - c.olympiadStudents

/-- Theorem: The total number of students only taking AMC 8 is 26 -/
theorem total_amc8_students (germain newton young : MathClass)
  (h_germain : germain = { teacher := "Mrs. Germain", totalStudents := 13, olympiadStudents := 3 })
  (h_newton : newton = { teacher := "Mr. Newton", totalStudents := 10, olympiadStudents := 2 })
  (h_young : young = { teacher := "Mrs. Young", totalStudents := 12, olympiadStudents := 4 }) :
  studentsOnlyAMC8 germain + studentsOnlyAMC8 newton + studentsOnlyAMC8 young = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_amc8_students_l1134_113433


namespace NUMINAMATH_CALUDE_age_ratio_sandy_molly_l1134_113428

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the passage of time in years -/
def yearsLater (a : Age) (n : ℕ) : Age :=
  ⟨a.years + n⟩

theorem age_ratio_sandy_molly :
  ∀ (sandy_current : Age) (molly_current : Age),
    yearsLater sandy_current 6 = Age.mk 42 →
    molly_current = Age.mk 27 →
    (sandy_current.years : ℚ) / molly_current.years = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sandy_molly_l1134_113428


namespace NUMINAMATH_CALUDE_min_group_size_l1134_113492

theorem min_group_size (adult_group_size children_group_size : ℕ) 
  (h1 : adult_group_size = 17)
  (h2 : children_group_size = 15)
  (h3 : ∃ n : ℕ, n > 0 ∧ n % adult_group_size = 0 ∧ n % children_group_size = 0) :
  (Nat.lcm adult_group_size children_group_size = 255) := by
  sorry

end NUMINAMATH_CALUDE_min_group_size_l1134_113492


namespace NUMINAMATH_CALUDE_total_distance_to_grandma_l1134_113407

/-- The distance to Grandma's house -/
def distance_to_grandma (distance_to_pie_shop : ℕ) (distance_to_gas_station : ℕ) (remaining_distance : ℕ) : ℕ :=
  distance_to_pie_shop + distance_to_gas_station + remaining_distance

/-- Theorem: The total distance to Grandma's house is 78 miles -/
theorem total_distance_to_grandma : 
  distance_to_grandma 35 18 25 = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_to_grandma_l1134_113407


namespace NUMINAMATH_CALUDE_square_of_complex_is_pure_imaginary_l1134_113412

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem square_of_complex_is_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) ^ 2) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_is_pure_imaginary_l1134_113412


namespace NUMINAMATH_CALUDE_prob_select_one_from_couple_l1134_113443

/-- The probability of selecting exactly one person from a couple, given their individual selection probabilities -/
theorem prob_select_one_from_couple (p_husband p_wife : ℝ) 
  (h_husband : p_husband = 1/7)
  (h_wife : p_wife = 1/5) :
  p_husband * (1 - p_wife) + p_wife * (1 - p_husband) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_one_from_couple_l1134_113443


namespace NUMINAMATH_CALUDE_v3_equals_55_l1134_113422

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^5 + 8x^4 - 3x^3 + 5x^2 + 12x - 6 -/
def f : List ℤ := [3, 8, -3, 5, 12, -6]

/-- Theorem: V_3 equals 55 when x = 2 for the given polynomial using Horner's method -/
theorem v3_equals_55 : horner f 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_v3_equals_55_l1134_113422


namespace NUMINAMATH_CALUDE_walking_speed_is_4_l1134_113441

/-- The speed at which Jack and Jill walked -/
def walking_speed : ℝ → ℝ := λ x => x^3 - 5*x^2 - 14*x + 104

/-- The distance Jill walked -/
def jill_distance : ℝ → ℝ := λ x => x^2 - 7*x - 60

/-- The time Jill walked -/
def jill_time : ℝ → ℝ := λ x => x + 7

theorem walking_speed_is_4 :
  ∃ x : ℝ, x ≠ -7 ∧ walking_speed x = (jill_distance x) / (jill_time x) ∧ walking_speed x = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_is_4_l1134_113441


namespace NUMINAMATH_CALUDE_product_perfect_square_l1134_113426

theorem product_perfect_square (nums : Finset ℕ) : 
  (nums.card = 17) →
  (∀ n ∈ nums, ∃ (a b c d : ℕ), n = 2^a * 3^b * 5^c * 7^d) →
  ∃ (n1 n2 : ℕ), n1 ∈ nums ∧ n2 ∈ nums ∧ n1 ≠ n2 ∧ ∃ (m : ℕ), n1 * n2 = m^2 :=
by sorry

end NUMINAMATH_CALUDE_product_perfect_square_l1134_113426


namespace NUMINAMATH_CALUDE_sequence_problem_l1134_113452

theorem sequence_problem (b : ℕ → ℝ) 
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 1987 = 17 + Real.sqrt 11) :
  b 2015 = (3 - Real.sqrt 11) / 8 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1134_113452


namespace NUMINAMATH_CALUDE_probability_different_with_three_l1134_113421

/-- The number of faces on a fair die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (different numbers with one being 3) -/
def favorableOutcomes : ℕ := 2 * (numFaces - 1)

/-- The probability of getting different numbers on two fair dice with one showing 3 -/
def probabilityDifferentWithThree : ℚ := favorableOutcomes / totalOutcomes

theorem probability_different_with_three :
  probabilityDifferentWithThree = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_with_three_l1134_113421


namespace NUMINAMATH_CALUDE_max_log_sum_l1134_113491

theorem max_log_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : a + 2*b = 6) :
  ∃ (max : ℝ), max = 3 * Real.log 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 6 → Real.log x + 2 * Real.log y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l1134_113491


namespace NUMINAMATH_CALUDE_no_solution_exists_l1134_113487

theorem no_solution_exists : ¬∃ (k t : ℕ), 
  (1 ≤ k ∧ k ≤ 9) ∧ 
  (1 ≤ t ∧ t ≤ 9) ∧ 
  (808 + 10 * k) - (800 + 88 * k) = 1606 + 10 * t :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1134_113487


namespace NUMINAMATH_CALUDE_complex_modulus_l1134_113403

theorem complex_modulus (z : ℂ) : (1 + Complex.I * Real.sqrt 3) * z = 1 + Complex.I →
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1134_113403


namespace NUMINAMATH_CALUDE_rachel_initial_lives_l1134_113467

/-- Rachel's initial number of lives -/
def initial_lives : ℕ := 10

/-- The number of lives Rachel lost -/
def lives_lost : ℕ := 4

/-- The number of lives Rachel gained -/
def lives_gained : ℕ := 26

/-- The final number of lives Rachel had -/
def final_lives : ℕ := 32

/-- Theorem stating that Rachel's initial number of lives was 10 -/
theorem rachel_initial_lives :
  initial_lives = 10 ∧
  final_lives = initial_lives - lives_lost + lives_gained :=
sorry

end NUMINAMATH_CALUDE_rachel_initial_lives_l1134_113467


namespace NUMINAMATH_CALUDE_skaters_meeting_distance_l1134_113447

/-- Represents the meeting point of two skaters --/
structure MeetingPoint where
  time : ℝ
  distance_allie : ℝ
  distance_billie : ℝ

/-- Calculates the meeting point of two skaters --/
def calculate_meeting_point (speed_allie speed_billie distance_ab angle : ℝ) : MeetingPoint :=
  sorry

/-- The theorem to be proved --/
theorem skaters_meeting_distance 
  (speed_allie : ℝ)
  (speed_billie : ℝ)
  (distance_ab : ℝ)
  (angle : ℝ)
  (h1 : speed_allie = 8)
  (h2 : speed_billie = 7)
  (h3 : distance_ab = 100)
  (h4 : angle = π / 3) -- 60 degrees in radians
  : 
  let meeting := calculate_meeting_point speed_allie speed_billie distance_ab angle
  meeting.distance_allie = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_skaters_meeting_distance_l1134_113447


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1134_113402

/-- Given two lines in the form of linear equations,
    returns true if they are perpendicular. -/
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The slope of the first line 3y + 2x - 6 = 0 -/
def m1 : ℚ := -2/3

/-- The slope of the second line 4y + ax - 5 = 0 in terms of a -/
def m2 (a : ℚ) : ℚ := -a/4

/-- Theorem stating that if the two given lines are perpendicular, then a = -6 -/
theorem perpendicular_lines_a_value :
  are_perpendicular m1 (m2 a) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1134_113402


namespace NUMINAMATH_CALUDE_inscribed_cone_volume_ratio_l1134_113445

/-- A right circular cone inscribed in a right prism -/
structure InscribedCone where
  /-- Radius of the cone's base -/
  r : ℝ
  /-- Height of both the cone and the prism -/
  h : ℝ
  /-- The radius and height are positive -/
  r_pos : r > 0
  h_pos : h > 0

/-- Theorem: The ratio of the volume of the inscribed cone to the volume of the prism is π/12 -/
theorem inscribed_cone_volume_ratio (c : InscribedCone) :
  (1 / 3 * π * c.r^2 * c.h) / (4 * c.r^2 * c.h) = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cone_volume_ratio_l1134_113445


namespace NUMINAMATH_CALUDE_john_share_l1134_113479

def total_amount : ℕ := 6000
def john_ratio : ℕ := 2
def jose_ratio : ℕ := 4
def binoy_ratio : ℕ := 6

theorem john_share :
  let total_ratio := john_ratio + jose_ratio + binoy_ratio
  (john_ratio : ℚ) / total_ratio * total_amount = 1000 := by sorry

end NUMINAMATH_CALUDE_john_share_l1134_113479


namespace NUMINAMATH_CALUDE_total_games_played_l1134_113497

/-- Given that Carla won 20 games and Frankie won half as many games as Carla,
    prove that the total number of games played is 30. -/
theorem total_games_played (carla_games frankie_games : ℕ) : 
  carla_games = 20 →
  frankie_games = carla_games / 2 →
  carla_games + frankie_games = 30 := by
sorry

end NUMINAMATH_CALUDE_total_games_played_l1134_113497


namespace NUMINAMATH_CALUDE_vector_coefficient_theorem_l1134_113463

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (A B C P O : V)

-- Define the condition that P lies in the plane of triangle ABC
def in_plane (A B C P : V) : Prop := 
  ∃ (α β γ : ℝ), α + β + γ = 1 ∧ P = α • A + β • B + γ • C

-- Define the vector equation
def vector_equation (A B C P O : V) (x : ℝ) : Prop :=
  P - O = (1/2) • (A - O) + (1/3) • (B - O) + x • (C - O)

-- State the theorem
theorem vector_coefficient_theorem 
  (h_plane : in_plane V A B C P)
  (h_eq : vector_equation V A B C P O x) :
  x = 1/6 := by sorry

end NUMINAMATH_CALUDE_vector_coefficient_theorem_l1134_113463


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l1134_113490

/-- Represents a triangle -/
structure Triangle where
  side_length : ℝ

/-- Represents a square -/
structure Square where
  side_length : ℝ

/-- Calculates the path length of a vertex of a triangle rotating inside a square -/
def path_length (t : Triangle) (s : Square) : ℝ :=
  sorry

/-- Theorem stating the path length for the given triangle and square -/
theorem triangle_rotation_path_length :
  let t : Triangle := { side_length := 3 }
  let s : Square := { side_length := 6 }
  path_length t s = 24 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l1134_113490


namespace NUMINAMATH_CALUDE_periodic_placement_exists_l1134_113482

/-- A function that maps integer coordinates to natural numbers -/
def f : ℤ × ℤ → ℕ := sorry

/-- Theorem stating the existence of a function satisfying the required properties -/
theorem periodic_placement_exists : 
  (∀ n : ℕ, ∃ x y : ℤ, f (x, y) = n) ∧ 
  (∀ a b c : ℤ, a ≠ 0 ∨ b ≠ 0 → c ≠ 0 → 
    ∃ k m : ℤ, ∀ x y : ℤ, a * x + b * y = c → 
      f (x + k, y + m) = f (x, y)) :=
by sorry

end NUMINAMATH_CALUDE_periodic_placement_exists_l1134_113482


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l1134_113470

theorem initial_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 8) = 84 → x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l1134_113470


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1134_113489

/-- Proposition p: am² < bm² -/
def p (a b m : ℝ) : Prop := a * m^2 < b * m^2

/-- Proposition q: a < b -/
def q (a b : ℝ) : Prop := a < b

/-- p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ a b m : ℝ, p a b m → q a b) ∧
  ¬(∀ a b : ℝ, q a b → ∀ m : ℝ, p a b m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1134_113489


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1134_113477

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1134_113477


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1134_113451

/-- An arithmetic sequence is monotonically increasing if its common difference is positive -/
def IsMonoIncreasingArithmeticSeq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_mono : IsMonoIncreasingArithmeticSeq a)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3/4) :
  a 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1134_113451


namespace NUMINAMATH_CALUDE_comic_book_ratio_l1134_113449

/-- Represents the number of comic books Sandy has at different stages -/
structure ComicBooks where
  initial : ℕ
  sold : ℕ
  bought : ℕ
  final : ℕ

/-- The ratio of sold books to initial books is 1:2 -/
theorem comic_book_ratio (s : ComicBooks) 
  (h1 : s.initial = 14)
  (h2 : s.bought = 6)
  (h3 : s.final = 13)
  (h4 : s.initial - s.sold + s.bought = s.final) :
  s.sold / s.initial = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_comic_book_ratio_l1134_113449


namespace NUMINAMATH_CALUDE_coat_price_reduction_l1134_113446

/-- Given a coat with an original price and a price reduction, 
    calculate the percent reduction. -/
theorem coat_price_reduction 
  (original_price : ℝ) 
  (price_reduction : ℝ) 
  (h1 : original_price = 500) 
  (h2 : price_reduction = 150) : 
  (price_reduction / original_price) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_reduction_l1134_113446


namespace NUMINAMATH_CALUDE_train_length_l1134_113484

/-- Calculates the length of a train given its speed, the speed of a motorbike it overtakes, and the time it takes to overtake. -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) : 
  train_speed = 100 → 
  motorbike_speed = 64 → 
  overtake_time = 20 → 
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 200 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1134_113484


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l1134_113410

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l1134_113410


namespace NUMINAMATH_CALUDE_q_value_at_two_l1134_113483

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define q(x) as p(p(x))
def q (x : ℝ) : ℝ := p (p x)

-- Theorem statement
theorem q_value_at_two (h : ∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ q a = 0 ∧ q b = 0 ∧ q c = 0) :
  q 2 = -1 := by
  sorry


end NUMINAMATH_CALUDE_q_value_at_two_l1134_113483


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1134_113404

theorem necessary_not_sufficient_condition (a b : ℝ) : 
  (∀ a b, a > b → a + 1 > b) ∧ 
  (∃ a b, a + 1 > b ∧ a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1134_113404


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l1134_113469

theorem solution_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + 12*x - m^2 = 0 ∧ x = 2) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l1134_113469


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1134_113455

theorem digit_sum_problem (J K L : ℕ) : 
  J ≠ K ∧ J ≠ L ∧ K ≠ L →
  J < 10 ∧ K < 10 ∧ L < 10 →
  100 * J + 10 * K + L + 100 * J + 10 * L + L + 100 * J + 10 * K + L = 479 →
  J + K + L = 11 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1134_113455


namespace NUMINAMATH_CALUDE_unique_intersection_l1134_113411

/-- The value of k for which the graphs of y = kx^2 - 5x + 4 and y = 2x - 6 intersect at exactly one point -/
def intersection_k : ℚ := 49/40

/-- First equation: y = kx^2 - 5x + 4 -/
def equation1 (k : ℚ) (x : ℚ) : ℚ := k * x^2 - 5*x + 4

/-- Second equation: y = 2x - 6 -/
def equation2 (x : ℚ) : ℚ := 2*x - 6

/-- Theorem stating that the graphs intersect at exactly one point if and only if k = 49/40 -/
theorem unique_intersection :
  ∀ k : ℚ, (∃! x : ℚ, equation1 k x = equation2 x) ↔ k = intersection_k :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1134_113411


namespace NUMINAMATH_CALUDE_correct_average_price_l1134_113498

/-- The average price of books given two purchases -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price of books is calculated correctly -/
theorem correct_average_price :
  average_price 27 20 581 594 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_price_l1134_113498


namespace NUMINAMATH_CALUDE_f_value_at_2_l1134_113416

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 10 → f a b 2 = 6 := by sorry

end NUMINAMATH_CALUDE_f_value_at_2_l1134_113416


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1134_113485

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.inningsPlayed + 1)

/-- Theorem: A batsman's average after 17th inning is 39, given the conditions -/
theorem batsman_average_after_17th_inning
  (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 16)
  (h2 : newAverage stats 87 = stats.average + 3)
  : newAverage stats 87 = 39 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1134_113485


namespace NUMINAMATH_CALUDE_new_line_properties_new_line_equation_correct_l1134_113414

/-- Given two lines in the plane -/
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

/-- The intersection point of the two lines -/
def intersection : ℝ × ℝ := (1, 1)

/-- The new line passing through the intersection point and having y-intercept -5 -/
def new_line (x y : ℝ) : Prop := 6 * x - y - 5 = 0

/-- Theorem stating that the new line passes through the intersection point and has y-intercept -5 -/
theorem new_line_properties :
  (line1 intersection.1 intersection.2) ∧
  (line2 intersection.1 intersection.2) ∧
  (new_line intersection.1 intersection.2) ∧
  (new_line 0 (-5)) :=
sorry

/-- Main theorem proving that the new line equation is correct -/
theorem new_line_equation_correct (x y : ℝ) :
  (line1 x y ∧ line2 x y) →
  (∃ t : ℝ, new_line (x + t * (intersection.1 - x)) (y + t * (intersection.2 - y))) :=
sorry

end NUMINAMATH_CALUDE_new_line_properties_new_line_equation_correct_l1134_113414


namespace NUMINAMATH_CALUDE_range_of_m_l1134_113488

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - m| < 5) ↔ -2 < m ∧ m < 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1134_113488


namespace NUMINAMATH_CALUDE_complex_quadrant_l1134_113454

theorem complex_quadrant (z : ℂ) (h : (1 + Complex.I * Real.sqrt 3) * z = 2 - Complex.I * Real.sqrt 3) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1134_113454


namespace NUMINAMATH_CALUDE_courtyard_width_prove_courtyard_width_l1134_113471

/-- The width of a rectangular courtyard given specific conditions -/
theorem courtyard_width : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (length width stone_side num_stones : ℝ) =>
    length = 30 ∧
    stone_side = 2 ∧
    num_stones = 135 ∧
    length * width = num_stones * stone_side * stone_side →
    width = 18

/-- Proof of the courtyard width theorem -/
theorem prove_courtyard_width :
  ∃ (length width stone_side num_stones : ℝ),
    courtyard_width length width stone_side num_stones :=
by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_prove_courtyard_width_l1134_113471


namespace NUMINAMATH_CALUDE_estimate_larger_than_actual_l1134_113419

theorem estimate_larger_than_actual (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  ⌈x⌉ - ⌊y⌋ > x - y :=
sorry

end NUMINAMATH_CALUDE_estimate_larger_than_actual_l1134_113419


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l1134_113473

/-- For positive integers a and b, where b > 1, if a^b is the greatest possible value less than 500, then a + b = 24 -/
theorem greatest_power_under_500 (a b : ℕ) (ha : a > 0) (hb : b > 1) 
  (h_greatest : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 500 → a^b ≥ x^y) 
  (h_less_500 : a^b < 500) : a + b = 24 := by
  sorry


end NUMINAMATH_CALUDE_greatest_power_under_500_l1134_113473


namespace NUMINAMATH_CALUDE_largest_inscribed_sphere_surface_area_l1134_113499

/-- The surface area of the largest sphere inscribed in a cone -/
theorem largest_inscribed_sphere_surface_area
  (base_radius : ℝ)
  (slant_height : ℝ)
  (h_base_radius : base_radius = 1)
  (h_slant_height : slant_height = 3) :
  ∃ (sphere_surface_area : ℝ),
    sphere_surface_area = 2 * Real.pi ∧
    ∀ (other_sphere_surface_area : ℝ),
      other_sphere_surface_area ≤ sphere_surface_area :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_sphere_surface_area_l1134_113499


namespace NUMINAMATH_CALUDE_expression_value_l1134_113450

theorem expression_value (m n x y : ℤ) 
  (h1 : m - n = 100) 
  (h2 : x + y = -1) : 
  (n + x) - (m - y) = -101 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1134_113450


namespace NUMINAMATH_CALUDE_sin_210_degrees_l1134_113420

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l1134_113420


namespace NUMINAMATH_CALUDE_triangle_interior_lines_sum_bound_l1134_113472

-- Define a triangle with side lengths x, y, z
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  hxy : x ≤ y
  hyz : y ≤ z

-- Define the sum s
def s (t : Triangle) (XX' YY' ZZ' : ℝ) : ℝ := XX' + YY' + ZZ'

-- Theorem statement
theorem triangle_interior_lines_sum_bound (t : Triangle) 
  (XX' YY' ZZ' : ℝ) (hXX' : XX' ≥ 0) (hYY' : YY' ≥ 0) (hZZ' : ZZ' ≥ 0) : 
  s t XX' YY' ZZ' ≤ t.x + t.y + t.z := by
  sorry

end NUMINAMATH_CALUDE_triangle_interior_lines_sum_bound_l1134_113472


namespace NUMINAMATH_CALUDE_addie_stamp_ratio_l1134_113424

theorem addie_stamp_ratio (parker_initial stamps : ℕ) (parker_final : ℕ) (addie_total : ℕ) : 
  parker_initial = 18 → 
  parker_final = 36 → 
  addie_total = 72 → 
  (parker_final - parker_initial) * 4 = addie_total := by
sorry

end NUMINAMATH_CALUDE_addie_stamp_ratio_l1134_113424


namespace NUMINAMATH_CALUDE_cindys_cycling_speed_l1134_113436

/-- Cindy's cycling problem -/
theorem cindys_cycling_speed :
  -- Cindy leaves school at the same time every day
  ∀ (leave_time : ℝ),
  -- Define the distance from school to home
  ∀ (distance : ℝ),
  -- If she cycles at 20 km/h, she arrives home at 4:30 PM
  (distance / 20 = 4.5 - leave_time) →
  -- If she cycles at 10 km/h, she arrives home at 5:15 PM
  (distance / 10 = 5.25 - leave_time) →
  -- Then the speed at which she must cycle to arrive home at 5:00 PM is 12 km/h
  (distance / 12 = 5 - leave_time) :=
by sorry

end NUMINAMATH_CALUDE_cindys_cycling_speed_l1134_113436


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1134_113431

/-- Represents a point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space --/
structure Line2D where
  k : ℝ
  b : ℝ

/-- Checks if a point is in the second quadrant --/
def isInSecondQuadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is on the given line --/
def isOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.k * p.x + l.b

/-- Theorem: A line with positive slope and negative y-intercept does not pass through the second quadrant --/
theorem line_not_in_second_quadrant (l : Line2D) 
  (h1 : l.k > 0) (h2 : l.b < 0) : 
  ¬ ∃ p : Point2D, isInSecondQuadrant p ∧ isOnLine p l :=
by
  sorry


end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1134_113431


namespace NUMINAMATH_CALUDE_gary_paycheck_l1134_113432

/-- Calculates the total paycheck for an employee with overtime --/
def calculate_paycheck (normal_wage : ℚ) (total_hours : ℕ) (regular_hours : ℕ) (overtime_multiplier : ℚ) : ℚ :=
  let regular_pay := normal_wage * regular_hours
  let overtime_hours := total_hours - regular_hours
  let overtime_pay := normal_wage * overtime_multiplier * overtime_hours
  regular_pay + overtime_pay

/-- Gary's paycheck calculation --/
theorem gary_paycheck :
  let normal_wage : ℚ := 12
  let total_hours : ℕ := 52
  let regular_hours : ℕ := 40
  let overtime_multiplier : ℚ := 3/2
  calculate_paycheck normal_wage total_hours regular_hours overtime_multiplier = 696 := by
  sorry


end NUMINAMATH_CALUDE_gary_paycheck_l1134_113432


namespace NUMINAMATH_CALUDE_marts_income_percentage_l1134_113476

theorem marts_income_percentage (juan tim mart : ℝ) : 
  tim = 0.6 * juan →
  mart = 0.9599999999999999 * juan →
  (mart - tim) / tim * 100 = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_marts_income_percentage_l1134_113476


namespace NUMINAMATH_CALUDE_leapYearsIn200Years_l1134_113405

/-- Definition of a leap year in the modified calendar system -/
def isLeapYear (year : ℕ) : Bool :=
  year % 4 == 0 && year % 128 ≠ 0

/-- Count of leap years in a given period -/
def countLeapYears (period : ℕ) : ℕ :=
  (List.range period).filter isLeapYear |>.length

/-- Theorem: There are 49 leap years in a 200-year period -/
theorem leapYearsIn200Years : countLeapYears 200 = 49 := by
  sorry

end NUMINAMATH_CALUDE_leapYearsIn200Years_l1134_113405


namespace NUMINAMATH_CALUDE_parabola_equation_l1134_113400

theorem parabola_equation (p : ℝ) (h_p : p > 0) :
  (∃ x y : ℝ, y^2 = 2*p*x ∧ 
    (x + p/2)^2 + y^2 = 100 ∧ 
    y^2 = 36) → 
  p = 2 ∨ p = 18 := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1134_113400
