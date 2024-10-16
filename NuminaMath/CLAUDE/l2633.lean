import Mathlib

namespace NUMINAMATH_CALUDE_well_digging_payment_l2633_263397

/-- Calculates the total payment for a group of workers given their daily work hours and hourly rate -/
def totalPayment (numWorkers : ℕ) (dailyHours : List ℕ) (hourlyRate : ℕ) : ℕ :=
  numWorkers * (dailyHours.sum * hourlyRate)

/-- Proves that the total payment for 3 workers working 12, 10, 8, and 14 hours on four days at $15 per hour is $1980 -/
theorem well_digging_payment :
  totalPayment 3 [12, 10, 8, 14] 15 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l2633_263397


namespace NUMINAMATH_CALUDE_trapezoid_area_l2633_263332

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the isosceles triangle PQR -/
structure IsoscelesTriangle extends Triangle

/-- Represents the trapezoid TQRS -/
structure Trapezoid where
  area : ℝ

/-- The problem setup -/
axiom smallest_triangle : Triangle
axiom smallest_triangle_area : smallest_triangle.area = 2

axiom PQR : IsoscelesTriangle
axiom PQR_area : PQR.area = 72

axiom PTQ : Triangle
axiom PTQ_composition : PTQ.area = 5 * smallest_triangle.area

axiom TQRS : Trapezoid
axiom TQRS_formation : TQRS.area = PQR.area - PTQ.area

/-- The theorem to prove -/
theorem trapezoid_area : TQRS.area = 62 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2633_263332


namespace NUMINAMATH_CALUDE_tic_tac_toe_probability_l2633_263344

/-- Represents a tic-tac-toe board -/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- The number of cells in a tic-tac-toe board -/
def boardSize : Nat := 9

/-- The number of noughts on the board -/
def noughtsCount : Nat := 3

/-- The number of crosses on the board -/
def crossesCount : Nat := 6

/-- The number of ways to choose noughts positions -/
def totalPositions : Nat := Nat.choose boardSize noughtsCount

/-- The number of winning positions for noughts -/
def winningPositions : Nat := 8

/-- Theorem: The probability of 3 noughts being in a winning position is 2/21 -/
theorem tic_tac_toe_probability : 
  (winningPositions : ℚ) / totalPositions = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_probability_l2633_263344


namespace NUMINAMATH_CALUDE_original_numbers_proof_l2633_263365

theorem original_numbers_proof : ∃ (a b c : ℕ), 
  a + b = 39 ∧ 
  b + c = 96 ∧ 
  a = 21 ∧ 
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_original_numbers_proof_l2633_263365


namespace NUMINAMATH_CALUDE_expression_bounds_l2633_263384

theorem expression_bounds (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  14/27 ≤ x^3 + 2*y^2 + (10/3)*z ∧ x^3 + 2*y^2 + (10/3)*z ≤ 10/3 := by
sorry

end NUMINAMATH_CALUDE_expression_bounds_l2633_263384


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2633_263348

/-- A geometric sequence with fifth term 243 and sixth term 729 has first term 3 -/
theorem geometric_sequence_first_term : ∀ (a : ℝ) (r : ℝ),
  a * r^4 = 243 →
  a * r^5 = 729 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2633_263348


namespace NUMINAMATH_CALUDE_pelt_costs_l2633_263340

/-- Proof of pelt costs given total cost and individual profits -/
theorem pelt_costs (total_cost : ℝ) (total_profit_percent : ℝ) 
  (profit_percent_1 : ℝ) (profit_percent_2 : ℝ) 
  (h1 : total_cost = 22500)
  (h2 : total_profit_percent = 40)
  (h3 : profit_percent_1 = 25)
  (h4 : profit_percent_2 = 50) :
  ∃ (cost_1 cost_2 : ℝ),
    cost_1 + cost_2 = total_cost ∧
    cost_1 * (1 + profit_percent_1 / 100) + cost_2 * (1 + profit_percent_2 / 100) 
      = total_cost * (1 + total_profit_percent / 100) ∧
    cost_1 = 9000 ∧
    cost_2 = 13500 := by
  sorry

end NUMINAMATH_CALUDE_pelt_costs_l2633_263340


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l2633_263366

-- Define the properties of rectangle R1
def R1_side : ℝ := 3
def R1_area : ℝ := 24

-- Define the diagonal of rectangle R2
def R2_diagonal : ℝ := 20

-- Theorem statement
theorem area_of_similar_rectangle :
  let R1_other_side := R1_area / R1_side
  let ratio := R1_other_side / R1_side
  let R2_side := (R2_diagonal^2 / (1 + ratio^2))^(1/2)
  R2_side * (ratio * R2_side) = 28800 / 219 :=
by sorry

end NUMINAMATH_CALUDE_area_of_similar_rectangle_l2633_263366


namespace NUMINAMATH_CALUDE_cupcake_cookie_price_ratio_l2633_263358

theorem cupcake_cookie_price_ratio :
  ∀ (cookie_price cupcake_price : ℚ),
    cookie_price > 0 →
    cupcake_price > 0 →
    5 * cookie_price + 3 * cupcake_price = 23 →
    4 * cookie_price + 4 * cupcake_price = 21 →
    cupcake_price / cookie_price = 13 / 29 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_cookie_price_ratio_l2633_263358


namespace NUMINAMATH_CALUDE_ratio_problem_l2633_263313

theorem ratio_problem (x y z : ℚ) (h1 : x / y = 3) (h2 : z / y = 4) :
  (y + z) / (x + z) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l2633_263313


namespace NUMINAMATH_CALUDE_village_foods_monthly_sales_l2633_263342

/-- Represents the monthly sales data for Village Foods --/
structure VillageFoodsSales where
  customers_per_month : ℕ
  lettuce_per_customer : ℕ
  lettuce_price : ℚ
  tomatoes_per_customer : ℕ
  tomato_price : ℚ

/-- Calculates the total monthly sales of lettuce and tomatoes --/
def total_monthly_sales (sales : VillageFoodsSales) : ℚ :=
  sales.customers_per_month * 
  (sales.lettuce_per_customer * sales.lettuce_price + 
   sales.tomatoes_per_customer * sales.tomato_price)

/-- Theorem stating that the total monthly sales of lettuce and tomatoes is $2000 --/
theorem village_foods_monthly_sales :
  let sales : VillageFoodsSales := {
    customers_per_month := 500,
    lettuce_per_customer := 2,
    lettuce_price := 1,
    tomatoes_per_customer := 4,
    tomato_price := 1/2
  }
  total_monthly_sales sales = 2000 := by sorry

end NUMINAMATH_CALUDE_village_foods_monthly_sales_l2633_263342


namespace NUMINAMATH_CALUDE_reunion_attendance_l2633_263337

/-- The number of people attending a family reunion. -/
def n : ℕ := sorry

/-- The age of the youngest person at the reunion. -/
def youngest_age : ℕ := sorry

/-- The age of the oldest person at the reunion. -/
def oldest_age : ℕ := sorry

/-- The sum of ages of all people at the reunion. -/
def total_age_sum : ℕ := sorry

/-- The average age of members excluding the oldest person is 18 years old. -/
axiom avg_without_oldest : (total_age_sum - oldest_age) / (n - 1) = 18

/-- The average age of members excluding the youngest person is 20 years old. -/
axiom avg_without_youngest : (total_age_sum - youngest_age) / (n - 1) = 20

/-- The age difference between the oldest and youngest person is 40 years. -/
axiom age_difference : oldest_age - youngest_age = 40

/-- The number of people attending the reunion is 21. -/
theorem reunion_attendance : n = 21 := by sorry

end NUMINAMATH_CALUDE_reunion_attendance_l2633_263337


namespace NUMINAMATH_CALUDE_students_walking_home_l2633_263341

theorem students_walking_home (bus auto bike skate : ℚ) 
  (h_bus : bus = 1/3)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/8)
  (h_skate : skate = 1/15) :
  1 - (bus + auto + bike + skate) = 11/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l2633_263341


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_eighteen_l2633_263385

theorem sum_of_roots_equals_eighteen : 
  let f (x : ℝ) := (3 * x^3 + 2 * x^2 - 9 * x + 15) - (4 * x^3 - 16 * x^2 + 27)
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧ r₁ + r₂ + r₃ = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_eighteen_l2633_263385


namespace NUMINAMATH_CALUDE_abigail_spending_l2633_263349

theorem abigail_spending (initial_amount : ℝ) : 
  let food_expense := 0.6 * initial_amount
  let remainder_after_food := initial_amount - food_expense
  let phone_bill := 0.25 * remainder_after_food
  let remainder_after_phone := remainder_after_food - phone_bill
  let entertainment_expense := 20
  let final_amount := remainder_after_phone - entertainment_expense
  (final_amount = 40) → (initial_amount = 200) :=
by
  sorry

end NUMINAMATH_CALUDE_abigail_spending_l2633_263349


namespace NUMINAMATH_CALUDE_intersection_equals_A_intersection_is_empty_l2633_263391

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for the first question
theorem intersection_equals_A (m : ℝ) :
  A ∩ B m = A ↔ m ≤ -2 :=
sorry

-- Theorem for the second question
theorem intersection_is_empty (m : ℝ) :
  A ∩ B m = ∅ ↔ 0 ≤ m :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_A_intersection_is_empty_l2633_263391


namespace NUMINAMATH_CALUDE_max_divisors_1_to_20_l2633_263302

def divisorCount (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def maxDivisorCount : ℕ → ℕ
  | 0 => 0
  | n + 1 => max (maxDivisorCount n) (divisorCount (n + 1))

theorem max_divisors_1_to_20 :
  maxDivisorCount 20 = 6 ∧
  divisorCount 12 = 6 ∧
  divisorCount 18 = 6 ∧
  divisorCount 20 = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → divisorCount n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_divisors_1_to_20_l2633_263302


namespace NUMINAMATH_CALUDE_cylinder_to_sphere_l2633_263324

/-- Given a cylinder with base radius 4 and lateral area 16π/3,
    prove its volume and the radius of an equivalent sphere -/
theorem cylinder_to_sphere (r : ℝ) (L : ℝ) (h : ℝ) (V : ℝ) (R : ℝ) :
  r = 4 →
  L = 16 / 3 * Real.pi →
  L = 2 * Real.pi * r * h →
  V = Real.pi * r^2 * h →
  V = 4 / 3 * Real.pi * R^3 →
  V = 32 / 3 * Real.pi ∧ R = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_to_sphere_l2633_263324


namespace NUMINAMATH_CALUDE_equation_solution_inequality_solution_l2633_263320

-- Definition of permutation
def A (n : ℕ) (m : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - m)

-- Theorem for the equation
theorem equation_solution :
  ∃! x : ℕ, 3 * A 8 x = 4 * A 9 (x - 1) ∧ x = 6 :=
sorry

-- Theorem for the inequality
theorem inequality_solution :
  ∀ x : ℕ, x ≥ 4 ↔ A (x - 2) 2 + x ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_solution_l2633_263320


namespace NUMINAMATH_CALUDE_negation_of_implication_l2633_263321

theorem negation_of_implication (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2633_263321


namespace NUMINAMATH_CALUDE_inverse_difference_evaluation_l2633_263310

theorem inverse_difference_evaluation (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 5*x - 3*y ≠ 0) : 
  (5*x - 3*y)⁻¹ * ((5*x)⁻¹ - (3*y)⁻¹) = -1 / (15*x*y) := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_evaluation_l2633_263310


namespace NUMINAMATH_CALUDE_no_x_term_l2633_263378

theorem no_x_term (m : ℝ) : (∀ x : ℝ, (x + m) * (x + 3) = x^2 + 3*m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_x_term_l2633_263378


namespace NUMINAMATH_CALUDE_green_sequin_rows_jane_green_sequin_rows_l2633_263352

/-- Calculates the number of rows of green sequins in Jane's costume. -/
theorem green_sequin_rows (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_rows : Nat) (purple_per_row : Nat) (green_per_row : Nat) 
  (total_sequins : Nat) : Nat :=
  let blue_sequins := blue_rows * blue_per_row
  let purple_sequins := purple_rows * purple_per_row
  let blue_and_purple := blue_sequins + purple_sequins
  let green_sequins := total_sequins - blue_and_purple
  let green_rows := green_sequins / green_per_row
  green_rows

/-- Proves that Jane sews 9 rows of green sequins. -/
theorem jane_green_sequin_rows : 
  green_sequin_rows 6 8 5 12 6 162 = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_sequin_rows_jane_green_sequin_rows_l2633_263352


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_imply_p_eq_two_l2633_263326

/-- A cubic polynomial with coefficient p -/
def cubic_poly (p : ℝ) (x : ℝ) : ℝ := x^3 - 6*p*x^2 + 5*p*x + 88

/-- The roots of the cubic polynomial form an arithmetic sequence -/
def roots_form_arithmetic_sequence (p : ℝ) : Prop :=
  ∃ (a d : ℝ), Set.range (λ i : Fin 3 => a + i.val * d) = {x | cubic_poly p x = 0}

/-- If the roots of x³ - 6px² + 5px + 88 = 0 form an arithmetic sequence, then p = 2 -/
theorem cubic_roots_arithmetic_imply_p_eq_two :
  ∀ p : ℝ, roots_form_arithmetic_sequence p → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_imply_p_eq_two_l2633_263326


namespace NUMINAMATH_CALUDE_grunters_win_probability_l2633_263316

def number_of_games : ℕ := 5
def win_probability : ℚ := 3/5

theorem grunters_win_probability :
  let p := win_probability
  let n := number_of_games
  (n.choose 4 * p^4 * (1-p)^1) + p^n = 1053/3125 := by sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l2633_263316


namespace NUMINAMATH_CALUDE_positive_integer_solutions_5x_plus_y_11_l2633_263372

theorem positive_integer_solutions_5x_plus_y_11 :
  {(x, y) : ℕ × ℕ | 5 * x + y = 11 ∧ x > 0 ∧ y > 0} = {(1, 6), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_5x_plus_y_11_l2633_263372


namespace NUMINAMATH_CALUDE_baseball_ratio_l2633_263361

theorem baseball_ratio (games_played : ℕ) (games_won : ℕ) 
  (h1 : games_played = 10) (h2 : games_won = 5) :
  (games_played : ℚ) / (games_played - games_won) = 2 := by
  sorry

end NUMINAMATH_CALUDE_baseball_ratio_l2633_263361


namespace NUMINAMATH_CALUDE_percent_to_decimal_l2633_263392

theorem percent_to_decimal (p : ℚ) : p / 100 = p / 100 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_l2633_263392


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2633_263351

/-- The surface area of a sphere circumscribing a cube with side length 4 is 48π. -/
theorem circumscribed_sphere_surface_area (cube_side : ℝ) (h : cube_side = 4) :
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius^2 = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l2633_263351


namespace NUMINAMATH_CALUDE_square_roots_theorem_l2633_263364

theorem square_roots_theorem (a : ℝ) (n : ℝ) : 
  (2 * a + 3)^2 = n ∧ (a - 18)^2 = n → n = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_theorem_l2633_263364


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2633_263300

theorem contrapositive_equivalence (a b : ℝ) : 
  (¬(a - 1 > b - 2) → ¬(a > b)) ↔ (a - 1 ≤ b - 2 → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2633_263300


namespace NUMINAMATH_CALUDE_conic_section_properties_l2633_263381

/-- A conic section defined by the equation x^2 + x + 2y - 2 = 0 -/
def conic_section (x y : ℝ) : Prop := x^2 + x + 2*y - 2 = 0

/-- The first line: x - 2y + 3 = 0 -/
def line1 (x y : ℝ) : Prop := x - 2*y + 3 = 0

/-- The second line: 5x + 2y - 6 = 0 -/
def line2 (x y : ℝ) : Prop := 5*x + 2*y - 6 = 0

/-- Point P -/
def P : ℝ × ℝ := (-1, 1)

/-- Point Q -/
def Q : ℝ × ℝ := (2, -2)

/-- Point R -/
def R : ℝ × ℝ := (1, 0)

/-- The conic section is tangent to line1 at point P, tangent to line2 at point Q, and passes through point R -/
theorem conic_section_properties :
  (conic_section P.1 P.2 ∧ line1 P.1 P.2) ∧
  (conic_section Q.1 Q.2 ∧ line2 Q.1 Q.2) ∧
  conic_section R.1 R.2 :=
sorry

end NUMINAMATH_CALUDE_conic_section_properties_l2633_263381


namespace NUMINAMATH_CALUDE_power_function_through_point_l2633_263359

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = 8) : 
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2633_263359


namespace NUMINAMATH_CALUDE_eight_possible_rankings_l2633_263394

/-- Represents a player in the tournament -/
inductive Player : Type
| X : Player
| Y : Player
| Z : Player
| W : Player

/-- Represents a match between two players -/
structure Match :=
  (player1 : Player)
  (player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
  (day1_match1 : Match)
  (day1_match2 : Match)
  (no_draws : Bool)

/-- Represents a final ranking of players -/
def Ranking := List Player

/-- Function to generate all possible rankings given a tournament structure -/
def generateRankings (t : Tournament) : List Ranking :=
  sorry

/-- Theorem stating that there are exactly 8 possible ranking sequences -/
theorem eight_possible_rankings (t : Tournament) 
  (h1 : t.day1_match1 = ⟨Player.X, Player.Y⟩)
  (h2 : t.day1_match2 = ⟨Player.Z, Player.W⟩)
  (h3 : t.no_draws = true)
  (h4 : (generateRankings t).length > 0)
  (h5 : [Player.X, Player.Z, Player.Y, Player.W] ∈ generateRankings t) :
  (generateRankings t).length = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_possible_rankings_l2633_263394


namespace NUMINAMATH_CALUDE_sum_equals_point_nine_six_repeating_l2633_263327

/-- Represents a repeating decimal where the digit 8 repeats infinitely -/
def repeating_eight : ℚ := 8/9

/-- Represents the decimal 0.07 -/
def seven_hundredths : ℚ := 7/100

/-- Theorem stating that the sum of 0.8̇ and 0.07 is equal to 0.96̇ -/
theorem sum_equals_point_nine_six_repeating :
  repeating_eight + seven_hundredths = 29/30 := by sorry

end NUMINAMATH_CALUDE_sum_equals_point_nine_six_repeating_l2633_263327


namespace NUMINAMATH_CALUDE_shoe_size_ratio_l2633_263343

def jasmine_shoe_size : ℕ := 7
def combined_shoe_size : ℕ := 21

def alexa_shoe_size : ℕ := combined_shoe_size - jasmine_shoe_size

theorem shoe_size_ratio : 
  alexa_shoe_size / jasmine_shoe_size = 2 := by sorry

end NUMINAMATH_CALUDE_shoe_size_ratio_l2633_263343


namespace NUMINAMATH_CALUDE_circle_radius_is_three_l2633_263374

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 10*y + 32 = 0

/-- The radius of a circle given by its equation -/
def CircleRadius (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem circle_radius_is_three :
  CircleRadius CircleEquation = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_three_l2633_263374


namespace NUMINAMATH_CALUDE_chairs_to_remove_proof_l2633_263330

/-- Calculates the number of chairs to remove given the initial setup and expected attendance --/
def chairs_to_remove (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_attendees : ℕ) : ℕ :=
  let rows_needed := (expected_attendees + chairs_per_row - 1) / chairs_per_row
  let chairs_needed := rows_needed * chairs_per_row
  total_chairs - chairs_needed

/-- Proves that given the specific conditions, 105 chairs should be removed --/
theorem chairs_to_remove_proof :
  chairs_to_remove 15 300 180 = 105 := by
  sorry

#eval chairs_to_remove 15 300 180

end NUMINAMATH_CALUDE_chairs_to_remove_proof_l2633_263330


namespace NUMINAMATH_CALUDE_fifth_figure_perimeter_l2633_263328

/-- Represents the outer perimeter of a figure in the sequence -/
def outer_perimeter (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

/-- The outer perimeter of the fifth figure in the sequence is 20 -/
theorem fifth_figure_perimeter :
  outer_perimeter 5 = 20 := by
  sorry

#check fifth_figure_perimeter

end NUMINAMATH_CALUDE_fifth_figure_perimeter_l2633_263328


namespace NUMINAMATH_CALUDE_box_side_face_area_l2633_263373

/-- Represents a rectangular box with length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- Calculates the area of the top face of a box -/
def topFaceArea (b : Box) : ℝ := b.length * b.width

/-- Calculates the area of the front face of a box -/
def frontFaceArea (b : Box) : ℝ := b.width * b.height

/-- Calculates the area of the side face of a box -/
def sideFaceArea (b : Box) : ℝ := b.length * b.height

theorem box_side_face_area (b : Box) 
  (h1 : volume b = 192)
  (h2 : frontFaceArea b = (1/2) * topFaceArea b)
  (h3 : topFaceArea b = (3/2) * sideFaceArea b) :
  sideFaceArea b = 32 := by
  sorry

end NUMINAMATH_CALUDE_box_side_face_area_l2633_263373


namespace NUMINAMATH_CALUDE_tan_sum_ratio_equals_neg_sqrt_three_over_three_l2633_263318

theorem tan_sum_ratio_equals_neg_sqrt_three_over_three : 
  (Real.tan (10 * π / 180) + Real.tan (20 * π / 180) + Real.tan (150 * π / 180)) / 
  (Real.tan (10 * π / 180) * Real.tan (20 * π / 180)) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_ratio_equals_neg_sqrt_three_over_three_l2633_263318


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l2633_263356

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  (∃ (k : ℤ), n^4 - n^2 = 12 * k) ∧ 
  (∀ (m : ℤ), m > 12 → ∃ (n : ℤ), ¬∃ (k : ℤ), n^4 - n^2 = m * k) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l2633_263356


namespace NUMINAMATH_CALUDE_min_rectangles_theorem_l2633_263347

/-- The minimum number of rectangles that can be placed on an n × n grid -/
def min_rectangles (k n : ℕ) : ℕ :=
  if n = k then k
  else min n (2*n - 2*k + 2)

/-- Theorem stating the minimum number of rectangles that can be placed -/
theorem min_rectangles_theorem (k n : ℕ) (h1 : k ≥ 2) (h2 : k ≤ n) (h3 : n ≤ 2*k - 1) :
  min_rectangles k n = 
    if n = k then k
    else min n (2*n - 2*k + 2) := by
  sorry

#check min_rectangles_theorem

end NUMINAMATH_CALUDE_min_rectangles_theorem_l2633_263347


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l2633_263363

/-- An arithmetic progression is represented by its first term and common difference. -/
structure ArithmeticProgression where
  a : ℤ  -- First term
  d : ℤ  -- Common difference

/-- A term in an arithmetic progression. -/
def ArithmeticProgression.term (ap : ArithmeticProgression) (n : ℕ) : ℤ :=
  ap.a + n * ap.d

/-- Predicate to check if a number is a perfect square. -/
def is_square (x : ℤ) : Prop :=
  ∃ k : ℤ, x = k * k

/-- An arithmetic progression contains a square. -/
def contains_square (ap : ArithmeticProgression) : Prop :=
  ∃ n : ℕ, is_square (ap.term n)

/-- An arithmetic progression contains infinitely many squares. -/
def contains_infinite_squares (ap : ArithmeticProgression) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ is_square (ap.term n)

/-- 
If an infinite arithmetic progression contains a square number, 
then it contains infinitely many square numbers.
-/
theorem arithmetic_progression_squares 
  (ap : ArithmeticProgression) 
  (h : contains_square ap) : 
  contains_infinite_squares ap :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l2633_263363


namespace NUMINAMATH_CALUDE_vector_norm_equation_solutions_l2633_263389

theorem vector_norm_equation_solutions :
  let v : ℝ × ℝ := (3, -4)
  let w : ℝ × ℝ := (5, 8)
  let norm_eq : ℝ → Prop := λ k => ‖k • v - w‖ = 5 * Real.sqrt 13
  ∀ k : ℝ, norm_eq k ↔ (k = 123 / 50 ∨ k = -191 / 50) :=
by sorry

end NUMINAMATH_CALUDE_vector_norm_equation_solutions_l2633_263389


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_tan_value_l2633_263377

theorem pure_imaginary_implies_tan_value (θ : ℝ) :
  (Complex.I * (Complex.cos θ - 4/5) = Complex.sin θ - 3/5) →
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_tan_value_l2633_263377


namespace NUMINAMATH_CALUDE_rounding_estimate_less_than_exact_l2633_263375

theorem rounding_estimate_less_than_exact (x y z : ℝ) 
  (hx : 1.5 < x ∧ x < 2) 
  (hy : 9 < y ∧ y < 9.5) 
  (hz : 3 < z ∧ z < 3.5) : 
  1 * 9 + 4 < x * y + z := by
  sorry

end NUMINAMATH_CALUDE_rounding_estimate_less_than_exact_l2633_263375


namespace NUMINAMATH_CALUDE_pressure_calculation_l2633_263346

/-- Prove that given the ideal gas law and specific conditions, the pressure is 1125000 Pa -/
theorem pressure_calculation (v R T V : ℝ) (h1 : v = 30)
  (h2 : R = 8.31) (h3 : T = 300) (h4 : V = 0.06648) :
  v * R * T / V = 1125000 :=
by sorry

end NUMINAMATH_CALUDE_pressure_calculation_l2633_263346


namespace NUMINAMATH_CALUDE_minimum_bills_for_exchange_l2633_263312

/-- Represents the count of bills for each denomination --/
structure BillCounts where
  hundred : ℕ
  fifty : ℕ
  twenty : ℕ
  ten : ℕ
  five : ℕ
  two : ℕ

/-- Calculates the total value of bills --/
def total_value (bills : BillCounts) : ℕ :=
  100 * bills.hundred + 50 * bills.fifty + 20 * bills.twenty +
  10 * bills.ten + 5 * bills.five + 2 * bills.two

/-- Checks if the bill counts satisfy the given constraints --/
def satisfies_constraints (bills : BillCounts) : Prop :=
  bills.hundred ≥ 3 ∧ bills.fifty ≥ 2 ∧ bills.twenty ≤ 4

/-- Initial bill counts --/
def initial_bills : BillCounts :=
  { hundred := 0, fifty := 12, twenty := 10, ten := 8, five := 15, two := 5 }

/-- Theorem stating the minimum number of bills needed for exchange --/
theorem minimum_bills_for_exchange :
  ∃ (exchange_bills : BillCounts),
    total_value exchange_bills = 3000 ∧
    satisfies_constraints exchange_bills ∧
    exchange_bills.hundred = 18 ∧
    exchange_bills.fifty = 3 ∧
    exchange_bills.twenty = 4 ∧
    exchange_bills.five = 1 ∧
    exchange_bills.ten = 0 ∧
    exchange_bills.two = 0 ∧
    (∀ (other_bills : BillCounts),
      total_value other_bills = 3000 →
      satisfies_constraints other_bills →
      total_value other_bills ≥ total_value exchange_bills) :=
sorry

end NUMINAMATH_CALUDE_minimum_bills_for_exchange_l2633_263312


namespace NUMINAMATH_CALUDE_converse_inequality_abs_l2633_263354

theorem converse_inequality_abs (x y : ℝ) : x > |y| → x > y := by
  sorry

end NUMINAMATH_CALUDE_converse_inequality_abs_l2633_263354


namespace NUMINAMATH_CALUDE_option_b_is_best_l2633_263360

-- Define the problem parameters
def total_metal_needed : ℝ := 635
def metal_in_storage : ℝ := 276
def aluminum_percentage : ℝ := 0.60
def steel_percentage : ℝ := 0.40

-- Define supplier options
structure Supplier :=
  (aluminum_price : ℝ)
  (steel_price : ℝ)

def option_a : Supplier := ⟨1.30, 0.90⟩
def option_b : Supplier := ⟨1.10, 1.00⟩
def option_c : Supplier := ⟨1.25, 0.95⟩

-- Calculate additional metal needed
def additional_metal_needed : ℝ := total_metal_needed - metal_in_storage

-- Calculate cost for a supplier
def calculate_cost (s : Supplier) : ℝ :=
  (additional_metal_needed * aluminum_percentage * s.aluminum_price) +
  (additional_metal_needed * steel_percentage * s.steel_price)

-- Theorem to prove
theorem option_b_is_best :
  calculate_cost option_b < calculate_cost option_a ∧
  calculate_cost option_b < calculate_cost option_c :=
by sorry

end NUMINAMATH_CALUDE_option_b_is_best_l2633_263360


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2633_263390

-- Define the inequality system
def inequality_system (x k : ℝ) : Prop :=
  (2 * x + 9 > 6 * x + 1) ∧ (x - k < 1)

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 2

-- Theorem statement
theorem inequality_system_solution (k : ℝ) :
  (∀ x, inequality_system x k ↔ solution_set x) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2633_263390


namespace NUMINAMATH_CALUDE_complementary_angle_of_25_l2633_263319

def complementary_angle (x : ℝ) : ℝ := 90 - x

theorem complementary_angle_of_25 :
  complementary_angle 25 = 65 :=
by sorry

end NUMINAMATH_CALUDE_complementary_angle_of_25_l2633_263319


namespace NUMINAMATH_CALUDE_volleyball_points_product_l2633_263367

def first_10_games : List ℕ := [5, 6, 4, 7, 5, 6, 2, 3, 4, 9]

def total_first_10 : ℕ := first_10_games.sum

theorem volleyball_points_product :
  ∀ (points_11 points_12 : ℕ),
    points_11 < 15 →
    points_12 < 15 →
    (total_first_10 + points_11) % 11 = 0 →
    (total_first_10 + points_11 + points_12) % 12 = 0 →
    points_11 * points_12 = 20 := by
sorry

end NUMINAMATH_CALUDE_volleyball_points_product_l2633_263367


namespace NUMINAMATH_CALUDE_factor_expression_l2633_263362

theorem factor_expression (y : ℝ) : 3*y*(y-4) + 8*(y-4) - 2*(y-4) = 3*(y+2)*(y-4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2633_263362


namespace NUMINAMATH_CALUDE_valuable_heirlooms_percentage_l2633_263357

theorem valuable_heirlooms_percentage
  (useful_percentage : Real)
  (junk_percentage : Real)
  (useful_items : Nat)
  (junk_items : Nat)
  (h1 : useful_percentage = 0.2)
  (h2 : junk_percentage = 0.7)
  (h3 : useful_items = 8)
  (h4 : junk_items = 28) :
  ∃ (total_items : Nat),
    (useful_items : Real) / total_items = useful_percentage ∧
    (junk_items : Real) / total_items = junk_percentage ∧
    1 - useful_percentage - junk_percentage = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_valuable_heirlooms_percentage_l2633_263357


namespace NUMINAMATH_CALUDE_trolleybus_problem_l2633_263336

/-- Trolleybus Problem -/
theorem trolleybus_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (∀ z : ℝ, z > 0 → y * z = 6 * (y - x) ∧ y * z = 3 * (y + x)) →
  (∃ z : ℝ, z = 4 ∧ x = y / 3) :=
by sorry

end NUMINAMATH_CALUDE_trolleybus_problem_l2633_263336


namespace NUMINAMATH_CALUDE_bread_roll_combinations_eq_21_l2633_263311

/-- The number of ways to distribute n identical items into k distinct groups -/
def starsAndBars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of combinations of bread rolls Tom could purchase -/
def breadRollCombinations : ℕ := starsAndBars 5 3

theorem bread_roll_combinations_eq_21 : breadRollCombinations = 21 := by
  sorry

end NUMINAMATH_CALUDE_bread_roll_combinations_eq_21_l2633_263311


namespace NUMINAMATH_CALUDE_solution_set_properties_inequality_properties_l2633_263305

/-- The function f(x) = x² - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

/-- Part I: Solution set properties -/
theorem solution_set_properties (a b : ℝ) :
  (∀ x, f a x ≤ -3 ↔ b ≤ x ∧ x ≤ 3) →
  a = 5 ∧ b = 2 :=
sorry

/-- Part II: Inequality properties -/
theorem inequality_properties (a : ℝ) :
  (∀ x, x ≥ 1/2 → f a x ≥ 1 - x^2) →
  a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_properties_inequality_properties_l2633_263305


namespace NUMINAMATH_CALUDE_ellipse_intersection_range_l2633_263380

/-- Ellipse C with center at origin, right focus at (√3, 0), and eccentricity √3/2 -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

/-- Line l: y = kx + √2 -/
def line_l (k x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

/-- Points A and B are distinct intersections of ellipse C and line l -/
def distinct_intersections (k x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
  line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

/-- Dot product of OA and OB is greater than 2 -/
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ > 2

/-- The range of k satisfies the given conditions -/
theorem ellipse_intersection_range :
  ∀ k : ℝ,
    (∀ x₁ y₁ x₂ y₂ : ℝ,
      distinct_intersections k x₁ y₁ x₂ y₂ →
      dot_product_condition x₁ y₁ x₂ y₂) →
    (k ∈ Set.Ioo (-Real.sqrt 3 / 3) (-1/2) ∪ Set.Ioo (1/2) (Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_range_l2633_263380


namespace NUMINAMATH_CALUDE_sin_inequality_l2633_263355

theorem sin_inequality (n : ℕ) (hn : n > 0) :
  Real.sin (1 / n) + Real.sin (2 / n) > (3 / n) * Real.cos (1 / n) := by
  sorry

end NUMINAMATH_CALUDE_sin_inequality_l2633_263355


namespace NUMINAMATH_CALUDE_equation_satisfied_l2633_263325

theorem equation_satisfied (a b c : ℤ) (h1 : a = b) (h2 : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l2633_263325


namespace NUMINAMATH_CALUDE_carpet_width_l2633_263393

/-- Calculates the width of a carpet given room dimensions and carpeting costs -/
theorem carpet_width
  (room_length : ℝ)
  (room_breadth : ℝ)
  (carpet_cost_paisa : ℝ)
  (total_cost_rupees : ℝ)
  (h1 : room_length = 15)
  (h2 : room_breadth = 6)
  (h3 : carpet_cost_paisa = 30)
  (h4 : total_cost_rupees = 36)
  : ∃ (carpet_width : ℝ), carpet_width = 75 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l2633_263393


namespace NUMINAMATH_CALUDE_coin_game_theorem_l2633_263388

/-- Represents the result of a coin-taking game -/
inductive GameResult
| FirstPlayerWins
| SecondPlayerWins

/-- Defines the coin-taking game and determines the winner -/
def coinGameWinner (n : ℕ) : GameResult :=
  if n = 7 then GameResult.FirstPlayerWins
  else if n = 12 then GameResult.SecondPlayerWins
  else sorry -- For other cases

/-- Theorem stating the winner for specific game configurations -/
theorem coin_game_theorem :
  (coinGameWinner 7 = GameResult.FirstPlayerWins) ∧
  (coinGameWinner 12 = GameResult.SecondPlayerWins) := by
  sorry

/-- The maximum value of coins a player can take in one turn -/
def maxTakeValue : ℕ := 3

/-- The value of a two-pound coin -/
def twoPoundValue : ℕ := 2

/-- The value of a one-pound coin -/
def onePoundValue : ℕ := 1

end NUMINAMATH_CALUDE_coin_game_theorem_l2633_263388


namespace NUMINAMATH_CALUDE_mrs_thompson_chicken_cost_l2633_263368

/-- Given the total cost, number of chickens, and cost of potatoes, 
    calculate the cost of each chicken. -/
def chicken_cost (total : ℚ) (num_chickens : ℕ) (potato_cost : ℚ) : ℚ :=
  (total - potato_cost) / num_chickens

/-- Prove that each chicken costs $3 given the problem conditions -/
theorem mrs_thompson_chicken_cost :
  chicken_cost 15 3 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mrs_thompson_chicken_cost_l2633_263368


namespace NUMINAMATH_CALUDE_certain_number_problem_l2633_263370

theorem certain_number_problem (x : ℝ) : 
  (0.8 * 40 = (4/5) * x + 16) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2633_263370


namespace NUMINAMATH_CALUDE_gcf_of_270_and_180_l2633_263353

theorem gcf_of_270_and_180 : Nat.gcd 270 180 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_270_and_180_l2633_263353


namespace NUMINAMATH_CALUDE_round_trip_speed_l2633_263314

/-- Proves that given the conditions of the round trip, the speed from B to A is 45 miles per hour -/
theorem round_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) (speed_ba : ℝ) : 
  distance = 180 →
  speed_ab = 90 →
  avg_speed = 60 →
  speed_ba = (2 * distance * avg_speed) / (2 * distance - avg_speed * (distance / speed_ab)) →
  speed_ba = 45 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_l2633_263314


namespace NUMINAMATH_CALUDE_beef_pack_weight_l2633_263323

/-- Given the conditions of James' beef purchase, prove the weight of each pack. -/
theorem beef_pack_weight (num_packs : ℕ) (price_per_pound : ℚ) (total_paid : ℚ) 
  (h1 : num_packs = 5)
  (h2 : price_per_pound = 5.5)
  (h3 : total_paid = 110) :
  (total_paid / price_per_pound) / num_packs = 4 := by
  sorry

#check beef_pack_weight

end NUMINAMATH_CALUDE_beef_pack_weight_l2633_263323


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2633_263322

theorem quadratic_roots_property (c : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ + c = 0) → 
  (x₂^2 + x₂ + c = 0) → 
  (x₁^2 * x₂ + x₂^2 * x₁ = 3) → 
  c = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2633_263322


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2633_263304

-- Define set A
def A : Set ℝ := {x | x^2 < 4}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2633_263304


namespace NUMINAMATH_CALUDE_gemma_pizza_order_l2633_263308

/-- The number of pizzas Gemma ordered -/
def number_of_pizzas : ℕ := 4

/-- The cost of each pizza in dollars -/
def pizza_cost : ℕ := 10

/-- The tip amount in dollars -/
def tip_amount : ℕ := 5

/-- The amount Gemma paid with in dollars -/
def payment_amount : ℕ := 50

/-- The change Gemma received in dollars -/
def change_amount : ℕ := 5

theorem gemma_pizza_order :
  number_of_pizzas * pizza_cost + tip_amount = payment_amount - change_amount :=
sorry

end NUMINAMATH_CALUDE_gemma_pizza_order_l2633_263308


namespace NUMINAMATH_CALUDE_novel_essay_arrangement_l2633_263334

/-- The number of ways to arrange 2 novels (which must be placed next to each other) and 3 essays on a bookshelf -/
def arrangement_count : ℕ := 48

/-- The number of novels -/
def novel_count : ℕ := 2

/-- The number of essays -/
def essay_count : ℕ := 3

/-- The total number of items to arrange (treating the novels as a single unit) -/
def total_units : ℕ := essay_count + 1

theorem novel_essay_arrangement :
  arrangement_count = (Nat.factorial total_units) * (Nat.factorial novel_count) :=
sorry

end NUMINAMATH_CALUDE_novel_essay_arrangement_l2633_263334


namespace NUMINAMATH_CALUDE_grade_change_impossible_l2633_263309

theorem grade_change_impossible : ∀ (n1 n2 n3 n4 : ℤ),
  2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
  -n1 + 2 * n2 + n3 - 2 * n4 = -27 →
  False :=
by
  sorry

end NUMINAMATH_CALUDE_grade_change_impossible_l2633_263309


namespace NUMINAMATH_CALUDE_lines_per_page_l2633_263371

theorem lines_per_page (total_words : ℕ) (words_per_line : ℕ) (pages_filled : ℚ) (words_left : ℕ) : 
  total_words = 400 →
  words_per_line = 10 →
  pages_filled = 3/2 →
  words_left = 100 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := by
sorry

end NUMINAMATH_CALUDE_lines_per_page_l2633_263371


namespace NUMINAMATH_CALUDE_james_writing_time_l2633_263387

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  pages_per_day_per_person : ℕ
  people_per_day : ℕ

/-- Calculate the hours spent writing per week -/
def hours_per_week (scenario : WritingScenario) : ℕ :=
  let pages_per_day := scenario.pages_per_day_per_person * scenario.people_per_day
  let pages_per_week := pages_per_day * 7
  pages_per_week / scenario.pages_per_hour

/-- Theorem stating James spends 7 hours a week writing -/
theorem james_writing_time (james : WritingScenario)
  (h1 : james.pages_per_hour = 10)
  (h2 : james.pages_per_day_per_person = 5)
  (h3 : james.people_per_day = 2) :
  hours_per_week james = 7 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_time_l2633_263387


namespace NUMINAMATH_CALUDE_fixed_points_bisector_range_l2633_263329

noncomputable def f (a b x : ℝ) : ℝ := a * x + b + 1

theorem fixed_points_bisector_range (a b : ℝ) :
  (0 < a) → (a < 2) →
  (∃ x₀ : ℝ, f a b x₀ = x₀) →
  (∃ A B : ℝ × ℝ, 
    (f a b A.1 = A.2 ∧ f a b B.1 = B.2) ∧
    (∀ x y : ℝ, y = x + 1 / (2 * a^2 + 1) ↔ 
      ((x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
       2 * x = A.1 + B.1 ∧ 2 * y = A.2 + B.2))) →
  b ∈ Set.Icc (-Real.sqrt 2 / 4) 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_bisector_range_l2633_263329


namespace NUMINAMATH_CALUDE_ashton_pencils_left_l2633_263333

def pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (given_away : ℕ) : ℕ :=
  boxes * pencils_per_box - given_away

theorem ashton_pencils_left : pencils_left 2 14 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ashton_pencils_left_l2633_263333


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2633_263383

theorem initial_money_calculation (X : ℝ) : 
  X * (1 - (0.30 + 0.25 + 0.15)) = 3500 → X = 11666.67 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2633_263383


namespace NUMINAMATH_CALUDE_octahedron_colorings_l2633_263396

/-- The number of faces in a regular octahedron -/
def num_faces : ℕ := 8

/-- The number of rotational symmetries of a regular octahedron -/
def num_rotational_symmetries : ℕ := 24

/-- The number of distinguishable colorings of a regular octahedron -/
def num_distinguishable_colorings : ℕ := Nat.factorial num_faces / num_rotational_symmetries

theorem octahedron_colorings :
  num_distinguishable_colorings = 1680 := by sorry

end NUMINAMATH_CALUDE_octahedron_colorings_l2633_263396


namespace NUMINAMATH_CALUDE_expression_evaluation_l2633_263301

theorem expression_evaluation : 
  81 + (128 / 16) + (15 * 12) - 250 - (180 / 3)^2 = -3581 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2633_263301


namespace NUMINAMATH_CALUDE_john_savings_proof_l2633_263315

/-- Calculates the monthly savings amount given the total savings period, amount spent, and remaining amount. -/
def monthly_savings (savings_period_years : ℕ) (amount_spent : ℕ) (amount_remaining : ℕ) : ℚ :=
  let total_saved : ℕ := amount_spent + amount_remaining
  let total_months : ℕ := savings_period_years * 12
  (total_saved : ℚ) / total_months

/-- Proves that given a savings period of 2 years, $400 spent, and $200 remaining, the monthly savings amount is $25. -/
theorem john_savings_proof :
  monthly_savings 2 400 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_john_savings_proof_l2633_263315


namespace NUMINAMATH_CALUDE_annual_turbans_count_l2633_263386

/-- Represents the annual salary structure and partial payment details --/
structure SalaryInfo where
  annual_cash : ℕ  -- Annual cash component in Rupees
  turban_price : ℕ  -- Price of one turban in Rupees
  partial_months : ℕ  -- Number of months worked
  partial_cash : ℕ  -- Cash received for partial work in Rupees
  partial_turbans : ℕ  -- Number of turbans received for partial work

/-- Calculates the number of turbans in the annual salary --/
def calculate_annual_turbans (info : SalaryInfo) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of turbans in the annual salary is 1 --/
theorem annual_turbans_count (info : SalaryInfo) 
  (h1 : info.annual_cash = 90)
  (h2 : info.turban_price = 50)
  (h3 : info.partial_months = 9)
  (h4 : info.partial_cash = 55)
  (h5 : info.partial_turbans = 1) :
  calculate_annual_turbans info = 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_turbans_count_l2633_263386


namespace NUMINAMATH_CALUDE_min_distance_point_l2633_263395

noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 + (2 * Real.log x - 2 * a)^2

theorem min_distance_point (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ ≤ 4/5) → a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_l2633_263395


namespace NUMINAMATH_CALUDE_problem_statement_l2633_263331

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) :
  3 * a^2008 - 5 * b^2008 = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2633_263331


namespace NUMINAMATH_CALUDE_definite_integral_x_plus_two_cubed_ln_squared_l2633_263399

open Real MeasureTheory

theorem definite_integral_x_plus_two_cubed_ln_squared :
  ∫ x in (-1)..(0), (x + 2)^3 * (log (x + 2))^2 = 4 * (log 2)^2 - 2 * log 2 + 15/32 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_plus_two_cubed_ln_squared_l2633_263399


namespace NUMINAMATH_CALUDE_sum_x_y_equals_two_l2633_263398

theorem sum_x_y_equals_two (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 1))
  (h2 : (5 : ℝ) ^ (2 * y) = 25 ^ (x - 2)) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_two_l2633_263398


namespace NUMINAMATH_CALUDE_square_sum_eq_243_l2633_263317

theorem square_sum_eq_243 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_eq_243_l2633_263317


namespace NUMINAMATH_CALUDE_max_intersections_count_l2633_263339

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 15

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 10

/-- The total number of segments connecting points on x-axis to points on y-axis -/
def num_segments : ℕ := num_x_points * num_y_points

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := (num_x_points.choose 2) * (num_y_points.choose 2)

theorem max_intersections_count :
  max_intersections = 4725 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_count_l2633_263339


namespace NUMINAMATH_CALUDE_german_team_goals_l2633_263306

def journalist1_statement (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2_statement (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3_statement (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1_statement x ∧ journalist2_statement x ∧ ¬journalist3_statement x) ∨
  (journalist1_statement x ∧ ¬journalist2_statement x ∧ journalist3_statement x) ∨
  (¬journalist1_statement x ∧ journalist2_statement x ∧ journalist3_statement x)

theorem german_team_goals :
  {x : ℕ | exactly_two_correct x} = {11, 12, 14, 16, 17} := by sorry

end NUMINAMATH_CALUDE_german_team_goals_l2633_263306


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2633_263369

/-- A quadratic function that takes on specific values for consecutive natural numbers. -/
structure QuadraticFunction where
  f : ℝ → ℝ
  n : ℕ
  h1 : f n = 6
  h2 : f (n + 1) = 14
  h3 : f (n + 2) = 14

/-- The maximum value of a quadratic function with the given properties is 15. -/
theorem quadratic_max_value (qf : QuadraticFunction) : 
  ∃ (x : ℝ), ∀ (y : ℝ), qf.f y ≤ qf.f x ∧ qf.f x = 15 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2633_263369


namespace NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l2633_263335

/-- Given two 2D vectors a and b, prove that a + 3b equals the specified result. -/
theorem vector_addition_scalar_multiplication 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (-1, 5)) : 
  a + 3 • b = (-1, 18) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_multiplication_l2633_263335


namespace NUMINAMATH_CALUDE_pages_left_to_read_is_1000_l2633_263303

/-- Calculates the number of pages left to read in a book series -/
def pagesLeftToRead (totalBooks : ℕ) (pagesPerBook : ℕ) (readFirstMonth : ℕ) : ℕ :=
  let remainingAfterFirstMonth := totalBooks - readFirstMonth
  let readSecondMonth := remainingAfterFirstMonth / 2
  let totalRead := readFirstMonth + readSecondMonth
  let pagesLeft := (totalBooks - totalRead) * pagesPerBook
  pagesLeft

/-- Theorem: Given the specified reading pattern, 1000 pages are left to read -/
theorem pages_left_to_read_is_1000 :
  pagesLeftToRead 14 200 4 = 1000 := by
  sorry

#eval pagesLeftToRead 14 200 4

end NUMINAMATH_CALUDE_pages_left_to_read_is_1000_l2633_263303


namespace NUMINAMATH_CALUDE_kyle_driving_time_l2633_263307

/-- Given the conditions of Joseph and Kyle's driving, prove that Kyle's driving time is 2 hours. -/
theorem kyle_driving_time :
  let joseph_speed : ℝ := 50
  let joseph_time : ℝ := 2.5
  let kyle_speed : ℝ := 62
  let joseph_distance : ℝ := joseph_speed * joseph_time
  let kyle_distance : ℝ := joseph_distance - 1
  kyle_distance / kyle_speed = 2 := by sorry

end NUMINAMATH_CALUDE_kyle_driving_time_l2633_263307


namespace NUMINAMATH_CALUDE_tangent_line_value_l2633_263345

/-- The line x + y = c is tangent to the circle x^2 + y^2 = 8, where c is a positive real number. -/
def is_tangent_line (c : ℝ) : Prop :=
  c > 0 ∧ ∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c ∧
  ∀ (x' y' : ℝ), x' + y' = c → x'^2 + y'^2 ≥ 8

theorem tangent_line_value :
  ∀ c : ℝ, is_tangent_line c → c = 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_value_l2633_263345


namespace NUMINAMATH_CALUDE_football_campers_count_l2633_263382

theorem football_campers_count (total : ℕ) (basketball : ℕ) (soccer : ℕ) 
  (h1 : total = 88) 
  (h2 : basketball = 24) 
  (h3 : soccer = 32) : 
  total - soccer - basketball = 32 := by
sorry

end NUMINAMATH_CALUDE_football_campers_count_l2633_263382


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l2633_263338

theorem jelly_bean_distribution (total_jelly_beans : ℕ) (leftover_jelly_beans : ℕ) : 
  total_jelly_beans = 726 →
  leftover_jelly_beans = 4 →
  ∃ (girls : ℕ),
    let boys := girls + 3
    let students := girls + boys
    let distributed_jelly_beans := boys * boys + girls * (2 * girls + 1)
    distributed_jelly_beans = total_jelly_beans - leftover_jelly_beans →
    students = 31 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l2633_263338


namespace NUMINAMATH_CALUDE_third_episode_duration_l2633_263350

/-- Given a series of four episodes with known durations for three episodes
    and a total duration, this theorem proves the duration of the third episode. -/
theorem third_episode_duration
  (total_duration : ℕ)
  (first_episode : ℕ)
  (second_episode : ℕ)
  (fourth_episode : ℕ)
  (h1 : total_duration = 240)  -- 4 hours in minutes
  (h2 : first_episode = 58)
  (h3 : second_episode = 62)
  (h4 : fourth_episode = 55)
  : total_duration - (first_episode + second_episode + fourth_episode) = 65 := by
  sorry

#check third_episode_duration

end NUMINAMATH_CALUDE_third_episode_duration_l2633_263350


namespace NUMINAMATH_CALUDE_line_through_tangent_intersections_l2633_263379

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

/-- A point inside the ellipse -/
def M : ℝ × ℝ := (3, 2)

/-- A line intersecting the ellipse -/
structure IntersectingLine where
  a : ℝ × ℝ
  b : ℝ × ℝ
  ha : is_ellipse a.1 a.2
  hb : is_ellipse b.1 b.2

/-- The intersection point of tangent lines -/
structure TangentIntersection where
  p : ℝ × ℝ
  line : IntersectingLine
  -- Additional properties for tangent intersection could be added here

/-- The theorem statement -/
theorem line_through_tangent_intersections 
  (ab cd : IntersectingLine) 
  (p q : TangentIntersection) 
  (hp : p.line = ab) 
  (hq : q.line = cd) 
  (hab : ab.a.1 * M.1 / 25 + ab.a.2 * M.2 / 9 = 1)
  (hcd : cd.a.1 * M.1 / 25 + cd.a.2 * M.2 / 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = k * x + (1 - 3 * k / 25) * 9 / 2 ↔ 3 * x / 25 + 2 * y / 9 = 1 :=
sorry

end NUMINAMATH_CALUDE_line_through_tangent_intersections_l2633_263379


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l2633_263376

/-- 
A perfect cube is a number that is the result of multiplying an integer by itself twice.
-/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- 
The smallest positive integer x such that 1152x is a perfect cube is 36.
-/
theorem smallest_x_for_perfect_cube : 
  (∀ y : ℕ+, is_perfect_cube (1152 * y) → y ≥ 36) ∧ 
  is_perfect_cube (1152 * 36) := by
  sorry


end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l2633_263376
