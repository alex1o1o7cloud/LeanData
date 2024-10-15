import Mathlib

namespace NUMINAMATH_CALUDE_intersection_A_B_l3924_392447

def A : Set ℝ := {x | -1 < x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3924_392447


namespace NUMINAMATH_CALUDE_apple_sale_revenue_is_408_l3924_392414

/-- Calculates the money brought in from selling apples in bags -/
def apple_sale_revenue (total_harvest : ℕ) (juice_weight : ℕ) (restaurant_weight : ℕ) (bag_weight : ℕ) (price_per_bag : ℕ) : ℕ :=
  let remaining_weight := total_harvest - juice_weight - restaurant_weight
  let num_bags := remaining_weight / bag_weight
  num_bags * price_per_bag

/-- Theorem stating that the apple sale revenue is $408 given the problem conditions -/
theorem apple_sale_revenue_is_408 :
  apple_sale_revenue 405 90 60 5 8 = 408 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_revenue_is_408_l3924_392414


namespace NUMINAMATH_CALUDE_rectangle_area_l3924_392462

theorem rectangle_area (length width : ℚ) (h1 : length = 2 / 3) (h2 : width = 3 / 5) :
  length * width = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3924_392462


namespace NUMINAMATH_CALUDE_expression_equality_l3924_392405

theorem expression_equality : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3924_392405


namespace NUMINAMATH_CALUDE_combined_surface_area_theorem_l3924_392486

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents the combined shape of two cubes -/
structure CombinedShape where
  largerCube : Cube
  smallerCube : Cube

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength^2

/-- Calculates the surface area of the combined shape -/
def combinedSurfaceArea (cs : CombinedShape) : ℝ :=
  surfaceArea cs.largerCube + surfaceArea cs.smallerCube - 4 * cs.smallerCube.edgeLength^2

/-- The main theorem stating the surface area of the combined shape -/
theorem combined_surface_area_theorem (cs : CombinedShape) 
  (h1 : cs.largerCube.edgeLength = 2)
  (h2 : cs.smallerCube.edgeLength = cs.largerCube.edgeLength / 2) :
  combinedSurfaceArea cs = 32 := by
  sorry

#check combined_surface_area_theorem

end NUMINAMATH_CALUDE_combined_surface_area_theorem_l3924_392486


namespace NUMINAMATH_CALUDE_apartment_ratio_l3924_392440

theorem apartment_ratio (total_floors : ℕ) (max_residents : ℕ) 
  (h1 : total_floors = 12)
  (h2 : max_residents = 264) :
  ∃ (floors_with_6 : ℕ) (floors_with_5 : ℕ),
    floors_with_6 + floors_with_5 = total_floors ∧
    6 * floors_with_6 + 5 * floors_with_5 = max_residents / 4 ∧
    floors_with_6 * 2 = total_floors := by
  sorry

end NUMINAMATH_CALUDE_apartment_ratio_l3924_392440


namespace NUMINAMATH_CALUDE_cupcake_price_correct_l3924_392421

/-- The original price of cupcakes before the discount -/
def original_cupcake_price : ℝ := 3

/-- The original price of cookies before the discount -/
def original_cookie_price : ℝ := 2

/-- The number of cupcakes sold -/
def cupcakes_sold : ℕ := 16

/-- The number of cookies sold -/
def cookies_sold : ℕ := 8

/-- The total revenue from the sale -/
def total_revenue : ℝ := 32

/-- Theorem stating that the original cupcake price satisfies the given conditions -/
theorem cupcake_price_correct : 
  cupcakes_sold * (original_cupcake_price / 2) + cookies_sold * (original_cookie_price / 2) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_cupcake_price_correct_l3924_392421


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3924_392422

def solution_set (a : ℝ) : Set ℝ :=
  if a > 0 then { x | x < -a/4 ∨ x > a/3 }
  else if a = 0 then { x | x ≠ 0 }
  else { x | x > -a/4 ∨ x < a/3 }

theorem inequality_solution_set (a : ℝ) :
  { x : ℝ | 12 * x^2 - a * x > a^2 } = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3924_392422


namespace NUMINAMATH_CALUDE_monomial_exponent_equality_l3924_392420

theorem monomial_exponent_equality (a b : ℤ) : 
  (1 : ℤ) = a - 2 → b + 1 = 3 → (a - b)^(2023 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponent_equality_l3924_392420


namespace NUMINAMATH_CALUDE_sum_of_complex_exponentials_l3924_392410

/-- The sum of 16 complex exponentials with angles that are multiples of 2π/17 -/
theorem sum_of_complex_exponentials (ω : ℂ) (h : ω = Complex.exp (2 * Real.pi * Complex.I / 17)) :
  (Finset.range 16).sum (fun k => ω ^ (k + 1)) = ω := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_exponentials_l3924_392410


namespace NUMINAMATH_CALUDE_max_value_of_f_l3924_392439

/-- The domain of the function f -/
def Domain (c d e : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ c - d * p.1 - e * p.2 > 0}

/-- The function f -/
def f (a b c d e : ℝ) (p : ℝ × ℝ) : ℝ :=
  a * p.1 * b * p.2 * (c - d * p.1 - e * p.2)

/-- Theorem stating the maximum value of f -/
theorem max_value_of_f (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  ∃ M : ℝ, M = (a / d) * (b / e) * (c / 3)^3 ∧
  ∀ p ∈ Domain c d e, f a b c d e p ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3924_392439


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l3924_392424

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define symmetry with respect to x-axis
def symmetricXAxis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_x_axis :
  let M : Point := (3, -4)
  let M' : Point := symmetricXAxis M
  M' = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l3924_392424


namespace NUMINAMATH_CALUDE_dogsled_race_distance_l3924_392416

/-- The distance of a dogsled race course given the speeds and time differences of two teams. -/
theorem dogsled_race_distance
  (team_e_speed : ℝ)
  (team_a_speed_diff : ℝ)
  (team_a_time_diff : ℝ)
  (h1 : team_e_speed = 20)
  (h2 : team_a_speed_diff = 5)
  (h3 : team_a_time_diff = 3) :
  let team_a_speed := team_e_speed + team_a_speed_diff
  let team_e_time := (team_a_speed * team_a_time_diff) / (team_a_speed - team_e_speed)
  let distance := team_e_speed * team_e_time
  distance = 300 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_distance_l3924_392416


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l3924_392487

theorem parallelepiped_volume 
  (a b c : ℝ) 
  (h1 : Real.sqrt (a^2 + b^2 + c^2) = 13)
  (h2 : Real.sqrt (a^2 + b^2) = 3 * Real.sqrt 17)
  (h3 : Real.sqrt (b^2 + c^2) = 4 * Real.sqrt 10) :
  a * b * c = 144 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l3924_392487


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3924_392491

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price = 100)
  (h2 : gain_percent = 15) :
  cost_price * (1 + gain_percent / 100) = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3924_392491


namespace NUMINAMATH_CALUDE_cadence_total_earnings_l3924_392413

/-- Calculates the total earnings of Cadence from two companies given the specified conditions. -/
theorem cadence_total_earnings :
  let old_company_years : ℚ := 3.5
  let old_company_monthly_salary : ℚ := 5000
  let old_company_bonus_rate : ℚ := 0.5
  let new_company_years : ℕ := 4
  let new_company_salary_raise : ℚ := 0.2
  let new_company_bonus_rate : ℚ := 1
  let third_year_deduction_rate : ℚ := 0.02

  let old_company_salary := old_company_years * 12 * old_company_monthly_salary
  let old_company_bonus := (old_company_years.floor * old_company_bonus_rate * old_company_monthly_salary) +
                           (old_company_years - old_company_years.floor) * old_company_bonus_rate * old_company_monthly_salary
  let new_company_monthly_salary := old_company_monthly_salary * (1 + new_company_salary_raise)
  let new_company_salary := new_company_years * 12 * new_company_monthly_salary
  let new_company_bonus := new_company_years * new_company_bonus_rate * new_company_monthly_salary
  let third_year_deduction := third_year_deduction_rate * 12 * new_company_monthly_salary

  let total_earnings := old_company_salary + old_company_bonus + new_company_salary + new_company_bonus - third_year_deduction

  total_earnings = 529310 := by
    sorry

end NUMINAMATH_CALUDE_cadence_total_earnings_l3924_392413


namespace NUMINAMATH_CALUDE_stock_market_value_l3924_392459

/-- Prove that for a stock with an 8% dividend rate and a 20% yield, the market value is 40% of the face value. -/
theorem stock_market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) :
  dividend_rate = 0.08 →
  yield = 0.20 →
  (dividend_rate * face_value) / yield = 0.40 * face_value :=
by sorry

end NUMINAMATH_CALUDE_stock_market_value_l3924_392459


namespace NUMINAMATH_CALUDE_statue_cost_l3924_392430

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) : 
  selling_price = 750 ∧ 
  profit_percentage = 35 ∧ 
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 555.56 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l3924_392430


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l3924_392415

-- Define the property of being a positive number
def IsPositive (x : ℚ) : Prop := x > 0

-- Define the property of being a negative number
def IsNegative (x : ℚ) : Prop := x < 0

-- Theorem statement
theorem zero_neither_positive_nor_negative : 
  ¬(IsPositive 0) ∧ ¬(IsNegative 0) :=
sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l3924_392415


namespace NUMINAMATH_CALUDE_grid_game_winner_parity_second_player_wins_when_even_first_player_wins_when_odd_l3924_392496

/-- Represents the outcome of the grid game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Determines the winner of the grid game based on the dimensions of the grid -/
def gridGameWinner (m n : ℕ) : GameOutcome :=
  if (m + n) % 2 = 0 then
    GameOutcome.SecondPlayerWins
  else
    GameOutcome.FirstPlayerWins

/-- Theorem stating the winning condition for the grid game -/
theorem grid_game_winner_parity (m n : ℕ) :
  gridGameWinner m n = 
    if (m + n) % 2 = 0 then 
      GameOutcome.SecondPlayerWins
    else 
      GameOutcome.FirstPlayerWins := by
  sorry

/-- Corollary: The second player wins when m + n is even -/
theorem second_player_wins_when_even (m n : ℕ) (h : (m + n) % 2 = 0) :
  gridGameWinner m n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Corollary: The first player wins when m + n is odd -/
theorem first_player_wins_when_odd (m n : ℕ) (h : (m + n) % 2 ≠ 0) :
  gridGameWinner m n = GameOutcome.FirstPlayerWins := by
  sorry

end NUMINAMATH_CALUDE_grid_game_winner_parity_second_player_wins_when_even_first_player_wins_when_odd_l3924_392496


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3924_392494

-- Problem 1
theorem problem_1 (a : ℝ) (h : a ≠ 1) : 
  a^2 / (a - 1) - a - 1 = 1 / (a - 1) := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) :
  (2 * x * y) / (x^2 - y^2) / ((1 / (x - y)) + (1 / (x + y))) = y := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3924_392494


namespace NUMINAMATH_CALUDE_regional_math_competition_l3924_392403

theorem regional_math_competition (initial_contestants : ℕ) : 
  (initial_contestants : ℚ) * (2/5) * (1/2) = 30 → initial_contestants = 150 := by
  sorry

end NUMINAMATH_CALUDE_regional_math_competition_l3924_392403


namespace NUMINAMATH_CALUDE_fractional_linear_conjugacy_l3924_392438

/-- Given a fractional linear function f(x) = (ax + b) / (cx + d) where c ≠ 0 and ad ≠ bc,
    there exist functions φ and g such that f(x) = φ⁻¹(g(φ(x))). -/
theorem fractional_linear_conjugacy 
  {a b c d : ℝ} (hc : c ≠ 0) (had : a * d ≠ b * c) :
  ∃ (φ : ℝ → ℝ) (g : ℝ → ℝ),
    Function.Bijective φ ∧
    (∀ x, (a * x + b) / (c * x + d) = φ⁻¹ (g (φ x))) :=
by sorry

end NUMINAMATH_CALUDE_fractional_linear_conjugacy_l3924_392438


namespace NUMINAMATH_CALUDE_parabola_intersection_midpoint_l3924_392456

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at origin and focus at (p/2, 0) -/
structure Parabola where
  p : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  line.a * point.x + line.b * point.y + line.c = 0

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (p : Point) (q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

theorem parabola_intersection_midpoint 
  (parabola : Parabola)
  (C : Point)
  : 
  parabola.p = 2 ∧ C.x = 2 ∧ C.y = 1 →
  ∃ (l : Line) (M N : Point),
    l.a = 2 ∧ l.b = -1 ∧ l.c = -3 ∧
    onLine C l ∧
    onLine M l ∧ onLine N l ∧
    onParabola M parabola ∧ onParabola N parabola ∧
    isMidpoint C M N := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_midpoint_l3924_392456


namespace NUMINAMATH_CALUDE_emails_left_theorem_l3924_392407

/-- Calculates the number of emails left in the inbox after a series of moves -/
def emailsLeftInInbox (initialEmails : ℕ) : ℕ :=
  let afterTrash := initialEmails / 2
  let afterWork := afterTrash - (afterTrash * 2 / 5)
  let afterPersonal := afterWork - (afterWork / 4)
  afterPersonal - (afterPersonal / 10)

/-- Theorem stating that given 500 initial emails, after a series of moves, 102 emails are left in the inbox -/
theorem emails_left_theorem :
  emailsLeftInInbox 500 = 102 := by
  sorry

end NUMINAMATH_CALUDE_emails_left_theorem_l3924_392407


namespace NUMINAMATH_CALUDE_detergent_in_altered_solution_l3924_392442

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℕ
  detergent : ℕ
  water : ℕ

/-- Calculates the new ratio after altering the solution -/
def alter_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := r.detergent,
    water := 2 * r.water }

/-- Theorem: Given the conditions, the altered solution contains 60 liters of detergent -/
theorem detergent_in_altered_solution 
  (original_ratio : SolutionRatio)
  (h_original : original_ratio = ⟨2, 40, 100⟩)
  (h_water : (alter_ratio original_ratio).water = 300) :
  (alter_ratio original_ratio).detergent = 60 :=
sorry

end NUMINAMATH_CALUDE_detergent_in_altered_solution_l3924_392442


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l3924_392433

theorem zoo_ticket_price :
  let monday_children : ℕ := 7
  let monday_adults : ℕ := 5
  let tuesday_children : ℕ := 4
  let tuesday_adults : ℕ := 2
  let child_ticket_price : ℕ := 3
  let total_revenue : ℕ := 61
  ∃ (adult_ticket_price : ℕ),
    (monday_children * child_ticket_price + monday_adults * adult_ticket_price) +
    (tuesday_children * child_ticket_price + tuesday_adults * adult_ticket_price) = total_revenue ∧
    adult_ticket_price = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l3924_392433


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l3924_392437

theorem two_digit_numbers_problem : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧  -- a is a two-digit number
  10 ≤ b ∧ b < 100 ∧  -- b is a two-digit number
  a = 2 * b ∧         -- a is double b
  (a / 10 ≠ b / 10) ∧ (a / 10 ≠ b % 10) ∧ (a % 10 ≠ b / 10) ∧ (a % 10 ≠ b % 10) ∧  -- no common digits
  (a / 10 + a % 10 = b / 10) ∧  -- sum of digits of a equals tens digit of b
  (a % 10 - a / 10 = b % 10) ∧  -- difference of digits of a equals ones digit of b
  a = 34 ∧ b = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l3924_392437


namespace NUMINAMATH_CALUDE_tower_height_proof_l3924_392400

def sum_of_arithmetic_series (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tower_height_proof :
  let initial_blocks := 35
  let additional_blocks := 65
  let initial_height := sum_of_arithmetic_series initial_blocks
  let additional_height := sum_of_arithmetic_series additional_blocks
  initial_height + additional_height = 2775 :=
by sorry

end NUMINAMATH_CALUDE_tower_height_proof_l3924_392400


namespace NUMINAMATH_CALUDE_household_gas_fee_l3924_392484

def gas_fee (usage : ℕ) : ℚ :=
  if usage ≤ 60 then
    0.8 * usage
  else
    0.8 * 60 + 1.2 * (usage - 60)

theorem household_gas_fee :
  ∃ (usage : ℕ),
    usage > 60 ∧
    gas_fee usage / usage = 0.88 ∧
    gas_fee usage = 66 := by
  sorry

end NUMINAMATH_CALUDE_household_gas_fee_l3924_392484


namespace NUMINAMATH_CALUDE_bryan_has_more_candies_l3924_392495

-- Define the number of candies for Bryan and Ben
def bryan_skittles : ℕ := 50
def ben_mms : ℕ := 20

-- Theorem to prove Bryan has more candies and the difference is 30
theorem bryan_has_more_candies : 
  bryan_skittles > ben_mms ∧ bryan_skittles - ben_mms = 30 := by
  sorry

end NUMINAMATH_CALUDE_bryan_has_more_candies_l3924_392495


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3924_392427

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3924_392427


namespace NUMINAMATH_CALUDE_trevor_age_ratio_l3924_392444

/-- The ratio of Trevor's brother's age to Trevor's age 20 years ago -/
def age_ratio : ℚ := 16 / 3

theorem trevor_age_ratio :
  let trevor_age_decade_ago : ℕ := 16
  let brother_current_age : ℕ := 32
  let trevor_current_age : ℕ := trevor_age_decade_ago + 10
  let trevor_age_20_years_ago : ℕ := trevor_current_age - 20
  (brother_current_age : ℚ) / trevor_age_20_years_ago = age_ratio := by
  sorry

end NUMINAMATH_CALUDE_trevor_age_ratio_l3924_392444


namespace NUMINAMATH_CALUDE_only_25_satisfies_l3924_392463

theorem only_25_satisfies : ∀ n : ℕ, 
  (n > 5 * (n % 10) ∧ n ≠ 25) → False :=
by sorry

end NUMINAMATH_CALUDE_only_25_satisfies_l3924_392463


namespace NUMINAMATH_CALUDE_work_completion_time_l3924_392417

/-- Represents the time it takes for worker B to complete the work alone -/
def time_B_alone : ℝ := 10

/-- Represents the time it takes for worker A to complete the work alone -/
def time_A_alone : ℝ := 4

/-- Represents the time A and B work together -/
def time_together : ℝ := 2

/-- Represents the time B works alone after A leaves -/
def time_B_after_A : ℝ := 3.0000000000000004

/-- Theorem stating that given the conditions, B can finish the work alone in 10 days -/
theorem work_completion_time :
  time_B_alone = 10 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3924_392417


namespace NUMINAMATH_CALUDE_yellow_heavier_than_green_l3924_392401

/-- The weight difference between two blocks -/
def weight_difference (yellow_weight green_weight : Real) : Real :=
  yellow_weight - green_weight

/-- Theorem: The yellow block weighs 0.2 pounds more than the green block -/
theorem yellow_heavier_than_green :
  let yellow_weight : Real := 0.6
  let green_weight : Real := 0.4
  weight_difference yellow_weight green_weight = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_heavier_than_green_l3924_392401


namespace NUMINAMATH_CALUDE_set_intersection_complement_l3924_392474

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {1, 2, 3, 5}

-- Define set B
def B : Set Nat := {2, 4, 6}

-- Theorem statement
theorem set_intersection_complement : B ∩ (U \ A) = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l3924_392474


namespace NUMINAMATH_CALUDE_sin_75_times_sin_15_l3924_392435

theorem sin_75_times_sin_15 :
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_75_times_sin_15_l3924_392435


namespace NUMINAMATH_CALUDE_M_when_a_is_one_M_subset_N_iff_a_in_range_N_explicit_M_cases_l3924_392454

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1: When a = 1, M = {x | 0 < x < 2}
theorem M_when_a_is_one : M 1 = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem 2: M ⊆ N if and only if a ∈ [-2, 2]
theorem M_subset_N_iff_a_in_range : 
  ∀ a : ℝ, M a ⊆ N ↔ a ∈ Set.Icc (-2) 2 := by sorry

-- Additional helper theorems to establish the relationship
theorem N_explicit : N = Set.Icc (-1) 3 := by sorry

theorem M_cases (a : ℝ) : 
  (a < -1 → M a = {x | a + 1 < x ∧ x < 0}) ∧
  (a = -1 → M a = ∅) ∧
  (a > -1 → M a = {x | 0 < x ∧ x < a + 1}) := by sorry

end NUMINAMATH_CALUDE_M_when_a_is_one_M_subset_N_iff_a_in_range_N_explicit_M_cases_l3924_392454


namespace NUMINAMATH_CALUDE_employed_females_percentage_l3924_392446

theorem employed_females_percentage (total_population employed_population employed_males : ℝ) :
  employed_population / total_population = 0.7 →
  employed_males / total_population = 0.21 →
  (employed_population - employed_males) / employed_population = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l3924_392446


namespace NUMINAMATH_CALUDE_remaining_erasers_l3924_392467

theorem remaining_erasers (total : ℕ) (yeonju_fraction : ℚ) (minji_fraction : ℚ)
  (h_total : total = 28)
  (h_yeonju : yeonju_fraction = 1 / 4)
  (h_minji : minji_fraction = 3 / 7) :
  total - (↑total * yeonju_fraction).floor - (↑total * minji_fraction).floor = 9 := by
  sorry

end NUMINAMATH_CALUDE_remaining_erasers_l3924_392467


namespace NUMINAMATH_CALUDE_oranges_packed_l3924_392480

/-- Calculates the total number of oranges packed given the number of oranges per box and the number of boxes used. -/
def totalOranges (orangesPerBox : ℕ) (boxesUsed : ℕ) : ℕ :=
  orangesPerBox * boxesUsed

/-- Proves that packing 10 oranges per box in 265 boxes results in 2650 oranges packed. -/
theorem oranges_packed :
  let orangesPerBox : ℕ := 10
  let boxesUsed : ℕ := 265
  totalOranges orangesPerBox boxesUsed = 2650 := by
  sorry

end NUMINAMATH_CALUDE_oranges_packed_l3924_392480


namespace NUMINAMATH_CALUDE_one_fourth_difference_product_sum_l3924_392418

theorem one_fourth_difference_product_sum : 
  (1 / 4 : ℚ) * ((9 * 5) - (7 + 3)) = 35 / 4 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_difference_product_sum_l3924_392418


namespace NUMINAMATH_CALUDE_problem_statement_l3924_392429

theorem problem_statement :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧
  (∀ x : ℝ, x > 2 → Real.sqrt (x^2 + 1) + 4 / Real.sqrt (x^2 + 1) ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3924_392429


namespace NUMINAMATH_CALUDE_first_day_over_1000_l3924_392478

def fungi_count (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_over_1000 : ∃ n : ℕ, fungi_count n > 1000 ∧ ∀ m : ℕ, m < n → fungi_count m ≤ 1000 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_first_day_over_1000_l3924_392478


namespace NUMINAMATH_CALUDE_all_sheep_with_one_peasant_l3924_392457

/-- Represents the state of sheep distribution among peasants -/
structure SheepDistribution where
  peasants : List Nat
  deriving Repr

/-- Represents a single expropriation event -/
def expropriation (dist : SheepDistribution) : SheepDistribution :=
  sorry

/-- The total number of sheep -/
def totalSheep : Nat := 128

/-- Theorem: After 7 expropriations, all sheep end up with one peasant -/
theorem all_sheep_with_one_peasant 
  (initial : SheepDistribution) 
  (h_total : initial.peasants.sum = totalSheep) 
  (h_expropriations : ∃ (d : SheepDistribution), 
    d = (expropriation^[7]) initial ∧ 
    d.peasants.length > 0) : 
  ∃ (final : SheepDistribution), 
    final = (expropriation^[7]) initial ∧ 
    final.peasants = [totalSheep] :=
sorry

end NUMINAMATH_CALUDE_all_sheep_with_one_peasant_l3924_392457


namespace NUMINAMATH_CALUDE_boat_rental_problem_l3924_392475

theorem boat_rental_problem (total_students : ℕ) 
  (large_boat_capacity small_boat_capacity : ℕ) :
  total_students = 104 →
  large_boat_capacity = 12 →
  small_boat_capacity = 5 →
  ∃ (num_large_boats num_small_boats : ℕ),
    num_large_boats * large_boat_capacity + 
    num_small_boats * small_boat_capacity = total_students ∧
    (num_large_boats = 2 ∨ num_large_boats = 7) :=
by sorry

end NUMINAMATH_CALUDE_boat_rental_problem_l3924_392475


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3924_392426

theorem boxes_with_neither (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) 
  (h1 : total = 12)
  (h2 : pencils = 8)
  (h3 : pens = 5)
  (h4 : both = 3) :
  total - (pencils + pens - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3924_392426


namespace NUMINAMATH_CALUDE_trapezoid_constructible_l3924_392481

/-- A trapezoid with side lengths a, b, c, and d, where a and b are the bases and c and d are the legs. -/
structure Trapezoid (a b c d : ℝ) : Prop where
  base1 : a > 0
  base2 : b > 0
  leg1 : c > 0
  leg2 : d > 0

/-- The condition for constructibility of a trapezoid. -/
def isConstructible (a b c d : ℝ) : Prop :=
  c > d ∧ c - d < a - b ∧ a - b < c + d

/-- Theorem stating the necessary and sufficient conditions for constructing a trapezoid. -/
theorem trapezoid_constructible {a b c d : ℝ} (t : Trapezoid a b c d) :
  isConstructible a b c d ↔ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a - b :=
sorry

end NUMINAMATH_CALUDE_trapezoid_constructible_l3924_392481


namespace NUMINAMATH_CALUDE_total_hats_bought_l3924_392482

theorem total_hats_bought (blue_cost green_cost total_price green_count : ℕ)
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 548)
  (h4 : green_count = 38)
  (h5 : ∃ blue_count : ℕ, blue_cost * blue_count + green_cost * green_count = total_price) :
  ∃ total_count : ℕ, total_count = green_count + (total_price - green_cost * green_count) / blue_cost :=
by sorry

end NUMINAMATH_CALUDE_total_hats_bought_l3924_392482


namespace NUMINAMATH_CALUDE_milk_price_increase_day_l3924_392411

/-- The day in June when the milk price increased -/
def price_increase_day : ℕ := 19

/-- The cost of milk before the price increase -/
def initial_price : ℕ := 1500

/-- The cost of milk after the price increase -/
def new_price : ℕ := 1600

/-- The total amount spent on milk in June -/
def total_spent : ℕ := 46200

/-- The number of days in June -/
def days_in_june : ℕ := 30

theorem milk_price_increase_day :
  (price_increase_day - 1) * initial_price +
  (days_in_june - (price_increase_day - 1)) * new_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_milk_price_increase_day_l3924_392411


namespace NUMINAMATH_CALUDE_recipe_pancakes_l3924_392471

/-- The number of pancakes Bobby ate -/
def bobby_pancakes : ℕ := 5

/-- The number of pancakes Bobby's dog ate -/
def dog_pancakes : ℕ := 7

/-- The number of pancakes left -/
def leftover_pancakes : ℕ := 9

/-- The total number of pancakes made by the recipe -/
def total_pancakes : ℕ := bobby_pancakes + dog_pancakes + leftover_pancakes

theorem recipe_pancakes : total_pancakes = 21 := by
  sorry

end NUMINAMATH_CALUDE_recipe_pancakes_l3924_392471


namespace NUMINAMATH_CALUDE_box_height_proof_l3924_392476

/-- Given a box with specified dimensions and cube requirements, prove its height --/
theorem box_height_proof (length width : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h_length : length = 9)
  (h_width : width = 12)
  (h_cube_volume : cube_volume = 3)
  (h_min_cubes : min_cubes = 108) :
  (cube_volume * min_cubes) / (length * width) = 3 := by
  sorry

end NUMINAMATH_CALUDE_box_height_proof_l3924_392476


namespace NUMINAMATH_CALUDE_gcd_lcm_relation_l3924_392499

theorem gcd_lcm_relation (a b c : ℕ+) :
  (Nat.gcd a (Nat.gcd b c))^2 * Nat.lcm a b * Nat.lcm b c * Nat.lcm c a =
  (Nat.lcm a (Nat.lcm b c))^2 * Nat.gcd a b * Nat.gcd b c * Nat.gcd c a :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_relation_l3924_392499


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3924_392488

/-- A point P in ℝ² is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

/-- An angle θ is in the fourth quadrant if sin θ < 0 and cos θ > 0 -/
def in_fourth_quadrant (θ : ℝ) : Prop :=
  Real.sin θ < 0 ∧ Real.cos θ > 0

/-- If P(sin θ cos θ, 2cos θ) is in the second quadrant, then θ is in the fourth quadrant -/
theorem angle_in_fourth_quadrant (θ : ℝ) :
  in_second_quadrant (Real.sin θ * Real.cos θ, 2 * Real.cos θ) → in_fourth_quadrant θ :=
by sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l3924_392488


namespace NUMINAMATH_CALUDE_area_of_U_l3924_392445

noncomputable section

/-- A regular octagon in the complex plane -/
def RegularOctagon : Set ℂ :=
  sorry

/-- The region outside the regular octagon -/
def T : Set ℂ :=
  { z : ℂ | z ∉ RegularOctagon }

/-- The region U, which is the image of T under the transformation z ↦ 1/z -/
def U : Set ℂ :=
  { w : ℂ | ∃ z ∈ T, w = 1 / z }

/-- The area of a set in the complex plane -/
def area : Set ℂ → ℝ :=
  sorry

/-- The main theorem: The area of region U is 4 + 4π -/
theorem area_of_U : area U = 4 + 4 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_area_of_U_l3924_392445


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3924_392490

theorem rectangle_area_proof (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 25 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 2 * rectangle_width →
  rectangle_width * rectangle_length = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3924_392490


namespace NUMINAMATH_CALUDE_equation_solution_l3924_392428

theorem equation_solution (M : ℚ) : 
  (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M → M = 1003 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3924_392428


namespace NUMINAMATH_CALUDE_road_trip_mileage_l3924_392408

/-- Calculates the final mileage of a car after a road trip -/
def final_mileage (initial_mileage : ℕ) (efficiency : ℕ) (tank_capacity : ℕ) (refills : ℕ) : ℕ :=
  initial_mileage + efficiency * tank_capacity * refills

theorem road_trip_mileage :
  final_mileage 1728 30 20 2 = 2928 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_mileage_l3924_392408


namespace NUMINAMATH_CALUDE_watch_time_theorem_l3924_392483

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Represents a watch that loses time at a constant rate -/
structure Watch where
  lossRate : ℚ  -- Rate at which the watch loses time (in seconds per hour)

def Watch.actualTimeWhenShowing (w : Watch) (setTime : Time) (actualSetTime : Time) (showingTime : Time) : Time :=
  sorry  -- Implementation not required for the statement

theorem watch_time_theorem (w : Watch) :
  let noonTime : Time := ⟨12, 0, 0⟩
  let threeTime : Time := ⟨15, 0, 0⟩
  let watchAtThree : Time := ⟨14, 54, 30⟩
  let eightPM : Time := ⟨20, 0, 0⟩
  let actualEightPM : Time := ⟨20, 15, 8⟩
  w.actualTimeWhenShowing noonTime noonTime eightPM = actualEightPM :=
by sorry


end NUMINAMATH_CALUDE_watch_time_theorem_l3924_392483


namespace NUMINAMATH_CALUDE_candy_distribution_l3924_392468

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) : 
  total_candy = 16 → 
  num_bags = 2 → 
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 8 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3924_392468


namespace NUMINAMATH_CALUDE_equation_solutions_l3924_392466

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -1 ∧ 
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3924_392466


namespace NUMINAMATH_CALUDE_expression_simplification_l3924_392469

theorem expression_simplification (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  2 * (a^2 - 3*a*b + 1) - (2*a^2 - b^2) + 5*a*b = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3924_392469


namespace NUMINAMATH_CALUDE_path_area_and_cost_calculation_l3924_392451

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path -/
def construction_cost (area cost_per_sqm : ℝ) : ℝ :=
  area * cost_per_sqm

theorem path_area_and_cost_calculation 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : path_width = 2.5) 
  (h4 : cost_per_sqm = 7) : 
  path_area field_length field_width path_width = 675 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 4725 :=
by
  sorry

end NUMINAMATH_CALUDE_path_area_and_cost_calculation_l3924_392451


namespace NUMINAMATH_CALUDE_z_tetromino_placement_count_l3924_392455

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)

/-- Represents a tetromino -/
structure Tetromino :=
  (shape : String)

/-- Calculates the number of ways to place a rectangle on a chessboard -/
def placeRectangle (board : Chessboard) (width : Nat) (height : Nat) : Nat :=
  (board.size - width + 1) * (board.size - height + 1)

/-- Calculates the total number of ways to place a Z-shaped tetromino on a chessboard -/
def placeZTetromino (board : Chessboard) (tetromino : Tetromino) : Nat :=
  2 * (placeRectangle board 2 3 + placeRectangle board 3 2)

/-- The main theorem stating the number of ways to place a Z-shaped tetromino on an 8x8 chessboard -/
theorem z_tetromino_placement_count :
  let board : Chessboard := ⟨8⟩
  let tetromino : Tetromino := ⟨"Z"⟩
  placeZTetromino board tetromino = 168 := by
  sorry


end NUMINAMATH_CALUDE_z_tetromino_placement_count_l3924_392455


namespace NUMINAMATH_CALUDE_expression_bounds_l3924_392450

theorem expression_bounds (a b c d x : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ x) (hc : 0 ≤ c ∧ c ≤ x) (hd : 0 ≤ d ∧ d ≤ x)
  (hx : 0 < x ∧ x ≤ 10) : 
  2 * x * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (x - b)^2) + Real.sqrt (b^2 + (x - c)^2) + 
    Real.sqrt (c^2 + (x - d)^2) + Real.sqrt (d^2 + (x - a)^2) ∧
  Real.sqrt (a^2 + (x - b)^2) + Real.sqrt (b^2 + (x - c)^2) + 
    Real.sqrt (c^2 + (x - d)^2) + Real.sqrt (d^2 + (x - a)^2) ≤ 4 * x ∧
  ∃ (a' b' c' d' : ℝ), 
    0 ≤ a' ∧ a' ≤ x ∧ 0 ≤ b' ∧ b' ≤ x ∧ 0 ≤ c' ∧ c' ≤ x ∧ 0 ≤ d' ∧ d' ≤ x ∧
    Real.sqrt (a'^2 + (x - b')^2) + Real.sqrt (b'^2 + (x - c')^2) + 
    Real.sqrt (c'^2 + (x - d')^2) + Real.sqrt (d'^2 + (x - a')^2) = 2 * x * Real.sqrt 2 ∧
  ∃ (a'' b'' c'' d'' : ℝ), 
    0 ≤ a'' ∧ a'' ≤ x ∧ 0 ≤ b'' ∧ b'' ≤ x ∧ 0 ≤ c'' ∧ c'' ≤ x ∧ 0 ≤ d'' ∧ d'' ≤ x ∧
    Real.sqrt (a''^2 + (x - b''^2)) + Real.sqrt (b''^2 + (x - c''^2)) + 
    Real.sqrt (c''^2 + (x - d''^2)) + Real.sqrt (d''^2 + (x - a''^2)) = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l3924_392450


namespace NUMINAMATH_CALUDE_multiply_divide_sqrt_l3924_392472

theorem multiply_divide_sqrt (x y : ℝ) : 
  x = 0.7142857142857143 → 
  x ≠ 0 → 
  Real.sqrt ((x * y) / 7) = x → 
  y = 5 := by
sorry

end NUMINAMATH_CALUDE_multiply_divide_sqrt_l3924_392472


namespace NUMINAMATH_CALUDE_tom_weekly_fee_l3924_392419

/-- Represents Tom's car leasing scenario -/
structure CarLease where
  miles_mon_wed_fri : ℕ  -- Miles driven on Monday, Wednesday, Friday
  miles_other_days : ℕ   -- Miles driven on other days
  cost_per_mile : ℚ      -- Cost per mile in dollars
  total_annual_payment : ℚ -- Total annual payment in dollars

/-- Calculate the weekly fee given a car lease scenario -/
def weekly_fee (lease : CarLease) : ℚ :=
  let weekly_miles := 3 * lease.miles_mon_wed_fri + 4 * lease.miles_other_days
  let weekly_mileage_cost := weekly_miles * lease.cost_per_mile
  let annual_mileage_cost := 52 * weekly_mileage_cost
  (lease.total_annual_payment - annual_mileage_cost) / 52

/-- Theorem stating that the weekly fee for Tom's scenario is $95 -/
theorem tom_weekly_fee :
  let tom_lease := CarLease.mk 50 100 (1/10) 7800
  weekly_fee tom_lease = 95 := by sorry

end NUMINAMATH_CALUDE_tom_weekly_fee_l3924_392419


namespace NUMINAMATH_CALUDE_additional_emails_per_day_l3924_392402

theorem additional_emails_per_day 
  (initial_emails_per_day : ℕ)
  (total_days : ℕ)
  (subscription_day : ℕ)
  (total_emails : ℕ)
  (h1 : initial_emails_per_day = 20)
  (h2 : total_days = 30)
  (h3 : subscription_day = 15)
  (h4 : total_emails = 675) :
  ∃ (additional_emails : ℕ),
    additional_emails = 5 ∧
    total_emails = initial_emails_per_day * subscription_day + 
      (initial_emails_per_day + additional_emails) * (total_days - subscription_day) :=
by sorry

end NUMINAMATH_CALUDE_additional_emails_per_day_l3924_392402


namespace NUMINAMATH_CALUDE_exact_blue_marbles_probability_l3924_392425

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def trials : ℕ := 6
def blue_selections : ℕ := 3

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose trials blue_selections : ℚ) *
  probability_blue ^ blue_selections *
  probability_red ^ (trials - blue_selections) =
  3512320 / 11390625 := by
sorry

end NUMINAMATH_CALUDE_exact_blue_marbles_probability_l3924_392425


namespace NUMINAMATH_CALUDE_fifteenth_thirtyseventh_415th_digit_l3924_392412

/-- The decimal representation of 15/37 has a repeating sequence of '405'. -/
def decimal_rep : ℚ → List ℕ := sorry

/-- The nth digit after the decimal point in the decimal representation of a rational number. -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- The 415th digit after the decimal point in the decimal representation of 15/37 is 4. -/
theorem fifteenth_thirtyseventh_415th_digit :
  nth_digit (15 / 37) 415 = 4 := by sorry

end NUMINAMATH_CALUDE_fifteenth_thirtyseventh_415th_digit_l3924_392412


namespace NUMINAMATH_CALUDE_percentage_increase_60_to_80_l3924_392477

/-- The percentage increase when a value changes from 60 to 80 -/
theorem percentage_increase_60_to_80 : 
  (80 - 60) / 60 * 100 = 100 / 3 := by sorry

end NUMINAMATH_CALUDE_percentage_increase_60_to_80_l3924_392477


namespace NUMINAMATH_CALUDE_greatest_a_value_l3924_392449

theorem greatest_a_value (a : ℝ) : 
  (7 * Real.sqrt ((2 * a) ^ 2 + 1 ^ 2) - 4 * a ^ 2 - 1) / (Real.sqrt (1 + 4 * a ^ 2) + 3) = 2 →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_greatest_a_value_l3924_392449


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_three_l3924_392464

theorem smallest_two_digit_multiple_of_three : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 99) ∧ 
  n % 3 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 10 ∧ m ≤ 99) ∧ m % 3 = 0 → n ≤ m) ∧
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_three_l3924_392464


namespace NUMINAMATH_CALUDE_zebra_catches_tiger_l3924_392434

/-- The time it takes for a zebra to catch a tiger given their speeds and the tiger's head start -/
theorem zebra_catches_tiger (zebra_speed tiger_speed : ℝ) (head_start : ℝ) : 
  zebra_speed = 55 →
  tiger_speed = 30 →
  head_start = 5 →
  (head_start * tiger_speed) / (zebra_speed - tiger_speed) = 6 := by
  sorry

end NUMINAMATH_CALUDE_zebra_catches_tiger_l3924_392434


namespace NUMINAMATH_CALUDE_sin_alpha_plus_beta_equals_one_l3924_392432

theorem sin_alpha_plus_beta_equals_one 
  (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = Real.sqrt 3) : 
  Real.sin (α + β) = 1 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_beta_equals_one_l3924_392432


namespace NUMINAMATH_CALUDE_multiply_binomial_l3924_392406

theorem multiply_binomial (x : ℝ) : (-2*x)*(x - 3) = -2*x^2 + 6*x := by
  sorry

end NUMINAMATH_CALUDE_multiply_binomial_l3924_392406


namespace NUMINAMATH_CALUDE_johns_final_push_pace_l3924_392485

/-- Proves that John's pace during his final push was 4.2 m/s given the race conditions --/
theorem johns_final_push_pace (initial_distance : ℝ) (steve_speed : ℝ) (final_distance : ℝ) (push_duration : ℝ) :
  initial_distance = 12 →
  steve_speed = 3.7 →
  final_distance = 2 →
  push_duration = 28 →
  (push_duration * steve_speed + initial_distance + final_distance) / push_duration = 4.2 :=
by sorry

end NUMINAMATH_CALUDE_johns_final_push_pace_l3924_392485


namespace NUMINAMATH_CALUDE_relay_race_last_year_distance_l3924_392409

/-- Represents the relay race setup and calculations -/
def RelayRace (tables : ℕ) (distance_between_1_and_3 : ℝ) (multiplier : ℝ) : Prop :=
  let segment_length := distance_between_1_and_3 / 2
  let total_segments := tables - 1
  let this_year_distance := segment_length * total_segments
  let last_year_distance := this_year_distance / multiplier
  (tables = 6) ∧
  (distance_between_1_and_3 = 400) ∧
  (multiplier = 4) ∧
  (last_year_distance = 250)

/-- Theorem stating that given the conditions, the race distance last year was 250 meters -/
theorem relay_race_last_year_distance :
  ∀ (tables : ℕ) (distance_between_1_and_3 : ℝ) (multiplier : ℝ),
  RelayRace tables distance_between_1_and_3 multiplier :=
by
  sorry

end NUMINAMATH_CALUDE_relay_race_last_year_distance_l3924_392409


namespace NUMINAMATH_CALUDE_perpendicular_lines_exist_l3924_392441

/-- Two lines l₁ and l₂ in the plane -/
structure Lines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop
  l₂ : ℝ × ℝ → Prop
  h₁ : ∀ x y, l₁ (x, y) ↔ x + a * y = 3
  h₂ : ∀ x y, l₂ (x, y) ↔ 3 * x - (a - 2) * y = 2

/-- Perpendicularity condition for two lines -/
def perpendicular (l : Lines) : Prop :=
  1 * 3 + l.a * -(l.a - 2) = 0

/-- Theorem: If the lines are perpendicular, then there exists a real number a satisfying the condition -/
theorem perpendicular_lines_exist (l : Lines) (h : perpendicular l) : 
  ∃ a : ℝ, 1 * 3 + a * -(a - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_exist_l3924_392441


namespace NUMINAMATH_CALUDE_jumping_contest_l3924_392453

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 19 →
  frog_jump = grasshopper_jump + 10 →
  mouse_jump = grasshopper_jump + 30 →
  mouse_jump - frog_jump = 20 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l3924_392453


namespace NUMINAMATH_CALUDE_minimum_employees_proof_l3924_392460

/-- Represents the number of employees handling customer service -/
def customer_service : ℕ := 95

/-- Represents the number of employees handling technical support -/
def technical_support : ℕ := 80

/-- Represents the number of employees handling both customer service and technical support -/
def both : ℕ := 30

/-- Calculates the minimum number of employees needed to be hired -/
def min_employees : ℕ := (customer_service - both) + (technical_support - both) + both

theorem minimum_employees_proof :
  min_employees = 145 :=
sorry

end NUMINAMATH_CALUDE_minimum_employees_proof_l3924_392460


namespace NUMINAMATH_CALUDE_line_parameterization_l3924_392452

/-- Given a line y = 2x - 30 parameterized by (x,y) = (f(t), 20t - 10),
    prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ x y, y = 2 * x - 30 ↔ ∃ t, x = f t ∧ y = 20 * t - 10) →
  f = fun t => 10 * t + 10 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3924_392452


namespace NUMINAMATH_CALUDE_jack_walking_time_l3924_392404

/-- Represents the walking parameters and time for a person -/
structure WalkingData where
  steps_per_minute : ℕ
  step_length : ℕ
  time_to_school : ℚ

/-- Calculates the distance walked based on walking data -/
def distance_walked (data : WalkingData) : ℚ :=
  (data.steps_per_minute : ℚ) * (data.step_length : ℚ) * data.time_to_school / 100

theorem jack_walking_time 
  (dave : WalkingData)
  (jack : WalkingData)
  (h1 : dave.steps_per_minute = 80)
  (h2 : dave.step_length = 80)
  (h3 : dave.time_to_school = 20)
  (h4 : jack.steps_per_minute = 120)
  (h5 : jack.step_length = 50)
  (h6 : distance_walked dave = distance_walked jack) :
  jack.time_to_school = 64/3 := by
  sorry

end NUMINAMATH_CALUDE_jack_walking_time_l3924_392404


namespace NUMINAMATH_CALUDE_fish_pond_population_l3924_392458

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) 
  (h1 : initial_tagged = 70)
  (h2 : second_catch = 50)
  (h3 : tagged_in_second = 2)
  (h4 : (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish) :
  total_fish = 1750 := by
  sorry

#check fish_pond_population

end NUMINAMATH_CALUDE_fish_pond_population_l3924_392458


namespace NUMINAMATH_CALUDE_dividend_calculation_l3924_392470

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : divisor = 5 * remainder)
  (h3 : remainder = 46) :
  divisor * quotient + remainder = 5336 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3924_392470


namespace NUMINAMATH_CALUDE_eggs_to_buy_l3924_392493

theorem eggs_to_buy (total_needed : ℕ) (given_by_andrew : ℕ) 
  (h1 : total_needed = 222) (h2 : given_by_andrew = 155) : 
  total_needed - given_by_andrew = 67 := by
  sorry

end NUMINAMATH_CALUDE_eggs_to_buy_l3924_392493


namespace NUMINAMATH_CALUDE_number_of_inequalities_l3924_392473

-- Define a function to check if an expression is an inequality
def isInequality (expr : String) : Bool :=
  match expr with
  | "3 < 5" => true
  | "x > 0" => true
  | "2x ≠ 3" => true
  | "a = 3" => false
  | "2a + 1" => false
  | "(1-x)/5 > 1" => true
  | _ => false

-- Define the list of expressions
def expressions : List String :=
  ["3 < 5", "x > 0", "2x ≠ 3", "a = 3", "2a + 1", "(1-x)/5 > 1"]

-- Theorem stating that the number of inequalities is 4
theorem number_of_inequalities :
  (expressions.filter isInequality).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_inequalities_l3924_392473


namespace NUMINAMATH_CALUDE_prob_heart_or_king_correct_l3924_392461

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards that are either hearts or kings -/
def heart_or_king_count : ℕ := 16

/-- The probability of drawing at least one heart or king in two draws with replacement -/
def prob_at_least_one_heart_or_king : ℚ :=
  1 - (1 - heart_or_king_count / deck_size) ^ 2

theorem prob_heart_or_king_correct :
  prob_at_least_one_heart_or_king = 88 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_or_king_correct_l3924_392461


namespace NUMINAMATH_CALUDE_minimum_sugar_amount_l3924_392492

theorem minimum_sugar_amount (f s : ℝ) : 
  (f ≥ 8 + (3 * s) / 4) → 
  (f ≤ 2 * s) → 
  s ≥ 32 / 5 :=
by
  sorry

#eval (32 : ℚ) / 5  -- To show that 32/5 = 6.4

end NUMINAMATH_CALUDE_minimum_sugar_amount_l3924_392492


namespace NUMINAMATH_CALUDE_prob_at_most_one_success_in_three_trials_l3924_392431

/-- The probability of at most one success in three independent trials -/
theorem prob_at_most_one_success_in_three_trials (p : ℝ) (h : p = 1/3) :
  p^0 * (1-p)^3 + 3 * p^1 * (1-p)^2 = 20/27 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_success_in_three_trials_l3924_392431


namespace NUMINAMATH_CALUDE_tamtam_orange_shells_l3924_392448

/-- The number of orange shells in Tamtam's collection --/
def orange_shells (total purple pink yellow blue : ℕ) : ℕ :=
  total - (purple + pink + yellow + blue)

/-- Theorem stating the number of orange shells in Tamtam's collection --/
theorem tamtam_orange_shells :
  orange_shells 65 13 8 18 12 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tamtam_orange_shells_l3924_392448


namespace NUMINAMATH_CALUDE_colored_pencils_ratio_l3924_392465

/-- Proves that given the conditions in the problem, the ratio of Cheryl's colored pencils to Cyrus's is 3:1 -/
theorem colored_pencils_ratio (madeline_pencils : ℕ) (total_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : total_pencils = 231) : ∃ (cheryl_pencils cyrus_pencils : ℕ),
  cheryl_pencils = 2 * madeline_pencils ∧
  total_pencils = cheryl_pencils + cyrus_pencils + madeline_pencils ∧
  cheryl_pencils / cyrus_pencils = 3 := by
  sorry


end NUMINAMATH_CALUDE_colored_pencils_ratio_l3924_392465


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3924_392498

/-- The distance from the focus of a hyperbola to its asymptote -/
def distance_focus_to_asymptote (b : ℝ) : ℝ := 
  sorry

/-- The theorem stating the distance from the focus to the asymptote for a specific hyperbola -/
theorem hyperbola_focus_asymptote_distance : 
  ∀ b : ℝ, b > 0 → 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) → 
  (∃ x : ℝ, x^2 / 4 + b^2 = 9) →
  (∀ x y : ℝ, y^2 = 12*x) →
  distance_focus_to_asymptote b = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3924_392498


namespace NUMINAMATH_CALUDE_calculation_proof_l3924_392443

theorem calculation_proof : (3127 - 2972)^3 / 343 = 125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3924_392443


namespace NUMINAMATH_CALUDE_product_difference_sum_l3924_392436

theorem product_difference_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 120 →
  R * S = 120 →
  P - Q = R + S →
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_product_difference_sum_l3924_392436


namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l3924_392479

/-- Represents the cost of Plan X in cents for y gigabytes of data -/
def plan_x_cost (y : ℝ) : ℝ := 25 * y

/-- Represents the cost of Plan Y in cents for y gigabytes of data -/
def plan_y_cost (y : ℝ) : ℝ := 1500 + 15 * y

/-- The minimum number of gigabytes for Plan Y to be more cost-effective -/
def min_gb_for_plan_y : ℝ := 150

theorem plan_y_more_cost_effective :
  ∀ y : ℝ, y ≥ min_gb_for_plan_y → plan_y_cost y < plan_x_cost y :=
by sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l3924_392479


namespace NUMINAMATH_CALUDE_class_trip_problem_l3924_392423

theorem class_trip_problem (x y : ℕ) : 
  ((x + 5) * (y + 6) = x * y + 792) ∧ 
  ((x - 4) * (y + 4) = x * y - 388) → 
  (x = 27 ∧ y = 120) := by
sorry

end NUMINAMATH_CALUDE_class_trip_problem_l3924_392423


namespace NUMINAMATH_CALUDE_frank_payment_l3924_392497

/-- The amount of money Frank handed to the cashier -/
def amount_handed (chocolate_bars : ℕ) (chips : ℕ) (chocolate_price : ℕ) (chips_price : ℕ) (change : ℕ) : ℕ :=
  chocolate_bars * chocolate_price + chips * chips_price + change

/-- Proof that Frank handed $20 to the cashier -/
theorem frank_payment : amount_handed 5 2 2 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_frank_payment_l3924_392497


namespace NUMINAMATH_CALUDE_carson_octopus_legs_l3924_392489

/-- The number of octopuses Carson saw -/
def num_octopuses : ℕ := 5

/-- The number of legs each octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs Carson saw -/
def total_octopus_legs : ℕ := num_octopuses * legs_per_octopus

theorem carson_octopus_legs : total_octopus_legs = 40 := by
  sorry

end NUMINAMATH_CALUDE_carson_octopus_legs_l3924_392489
