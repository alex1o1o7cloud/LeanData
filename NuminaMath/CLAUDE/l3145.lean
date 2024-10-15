import Mathlib

namespace NUMINAMATH_CALUDE_stack_logs_count_l3145_314507

/-- The number of logs in a stack with arithmetic progression of rows -/
def logsInStack (bottomRow : ℕ) (topRow : ℕ) : ℕ :=
  let numRows := bottomRow - topRow + 1
  numRows * (bottomRow + topRow) / 2

theorem stack_logs_count : logsInStack 15 5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_stack_logs_count_l3145_314507


namespace NUMINAMATH_CALUDE_problem_statement_l3145_314570

theorem problem_statement : Real.rpow 81 0.25 * Real.rpow 81 0.2 = 9 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3145_314570


namespace NUMINAMATH_CALUDE_dani_pants_after_five_years_l3145_314523

/-- Calculates the number of pants after a given number of years -/
def pantsAfterYears (initialPants : ℕ) (pairsPerYear : ℕ) (pantsPerPair : ℕ) (years : ℕ) : ℕ :=
  initialPants + years * (pairsPerYear * pantsPerPair)

/-- Theorem: Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  pantsAfterYears 50 4 2 5 = 90 := by
  sorry

#eval pantsAfterYears 50 4 2 5

end NUMINAMATH_CALUDE_dani_pants_after_five_years_l3145_314523


namespace NUMINAMATH_CALUDE_ac_price_l3145_314557

/-- Given a car and an AC with prices in the ratio 3:2, where the car costs $500 more than the AC,
    prove that the price of the AC is $1000. -/
theorem ac_price (car_price ac_price : ℕ) : 
  car_price = 3 * (car_price / 5) ∧ 
  ac_price = 2 * (car_price / 5) ∧ 
  car_price = ac_price + 500 → 
  ac_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_ac_price_l3145_314557


namespace NUMINAMATH_CALUDE_hillarys_weekend_reading_l3145_314587

/-- The total reading time assigned for a weekend, given the reading times for Friday, Saturday, and Sunday. -/
def weekend_reading_time (friday_time saturday_time sunday_time : ℕ) : ℕ :=
  friday_time + saturday_time + sunday_time

/-- Theorem stating that the total reading time for Hillary's weekend assignment is 60 minutes. -/
theorem hillarys_weekend_reading : weekend_reading_time 16 28 16 = 60 := by
  sorry

end NUMINAMATH_CALUDE_hillarys_weekend_reading_l3145_314587


namespace NUMINAMATH_CALUDE_natalia_cycling_distance_l3145_314538

/-- Represents the total distance cycled over four days given specific conditions --/
def total_distance (monday tuesday : ℕ) : ℕ :=
  let wednesday := tuesday / 2
  let thursday := monday + wednesday
  monday + tuesday + wednesday + thursday

/-- Theorem stating that given the specific conditions in the problem, 
    the total distance cycled is 180 km --/
theorem natalia_cycling_distance : total_distance 40 50 = 180 := by
  sorry

end NUMINAMATH_CALUDE_natalia_cycling_distance_l3145_314538


namespace NUMINAMATH_CALUDE_colonization_combinations_eq_77056_l3145_314592

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 12

/-- The resource cost to colonize an Earth-like planet -/
def earth_cost : ℕ := 3

/-- The resource cost to colonize a Mars-like planet -/
def mars_cost : ℕ := 1

/-- The total available resources -/
def total_resources : ℕ := 18

/-- The function to calculate the number of combinations -/
def colonization_combinations : ℕ :=
  (Nat.choose earth_like_planets 2 * Nat.choose mars_like_planets 12) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 3) +
  (Nat.choose earth_like_planets 6 * Nat.choose mars_like_planets 0)

/-- The theorem stating that the number of colonization combinations is 77056 -/
theorem colonization_combinations_eq_77056 : colonization_combinations = 77056 := by
  sorry

end NUMINAMATH_CALUDE_colonization_combinations_eq_77056_l3145_314592


namespace NUMINAMATH_CALUDE_lcm_45_75_180_l3145_314551

theorem lcm_45_75_180 : Nat.lcm 45 (Nat.lcm 75 180) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_45_75_180_l3145_314551


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l3145_314563

theorem raffle_ticket_sales_difference :
  ∀ (friday_sales saturday_sales sunday_sales : ℕ),
    friday_sales = 181 →
    saturday_sales = 2 * friday_sales →
    sunday_sales = 78 →
    saturday_sales - sunday_sales = 284 :=
by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l3145_314563


namespace NUMINAMATH_CALUDE_probability_heart_king_king_ace_l3145_314502

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts excluding King and Ace of hearts -/
def HeartsExcludingKingAce : ℕ := 11

/-- Number of Kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of Aces in a standard deck -/
def AcesInDeck : ℕ := 4

/-- Probability of drawing the specific sequence (Heart, King, King, Ace) -/
def probabilityHeartKingKingAce : ℚ :=
  (HeartsExcludingKingAce : ℚ) / StandardDeck *
  KingsInDeck / (StandardDeck - 1) *
  (KingsInDeck - 1) / (StandardDeck - 2) *
  AcesInDeck / (StandardDeck - 3)

theorem probability_heart_king_king_ace :
  probabilityHeartKingKingAce = 1 / 12317 := by
  sorry

end NUMINAMATH_CALUDE_probability_heart_king_king_ace_l3145_314502


namespace NUMINAMATH_CALUDE_square_minus_circle_area_l3145_314526

theorem square_minus_circle_area (r : ℝ) (s : ℝ) : 
  r = 2 → s = 2 * Real.sqrt 2 → 
  s^2 - π * r^2 = 8 - 4 * π := by
sorry

end NUMINAMATH_CALUDE_square_minus_circle_area_l3145_314526


namespace NUMINAMATH_CALUDE_all_radii_equal_l3145_314584

/-- A circle with radius 2 cm -/
structure Circle :=
  (radius : ℝ)
  (h : radius = 2)

/-- Any radius of the circle is 2 cm -/
theorem all_radii_equal (c : Circle) (r : ℝ) (h : r = c.radius) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_all_radii_equal_l3145_314584


namespace NUMINAMATH_CALUDE_max_k_value_l3145_314571

theorem max_k_value (k : ℤ) : 
  (∃ x : ℕ+, k * x.val - 5 = 2021 * x.val + 2 * k) → k ≤ 6068 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l3145_314571


namespace NUMINAMATH_CALUDE_real_part_of_complex_power_l3145_314524

theorem real_part_of_complex_power : Complex.re ((1 - 2*Complex.I)^5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_power_l3145_314524


namespace NUMINAMATH_CALUDE_dihedral_angle_sum_l3145_314537

/-- A dihedral angle -/
structure DihedralAngle where
  /-- The linear angle of the dihedral angle -/
  linearAngle : ℝ
  /-- The angle between the external normals of the dihedral angle -/
  externalNormalAngle : ℝ
  /-- The linear angle is between 0 and π -/
  linearAngle_bounds : 0 < linearAngle ∧ linearAngle < π
  /-- The external normal angle is between 0 and π -/
  externalNormalAngle_bounds : 0 < externalNormalAngle ∧ externalNormalAngle < π

/-- The sum of the external normal angle and the linear angle of a dihedral angle is π -/
theorem dihedral_angle_sum (d : DihedralAngle) : 
  d.externalNormalAngle + d.linearAngle = π :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_sum_l3145_314537


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l3145_314582

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l3145_314582


namespace NUMINAMATH_CALUDE_new_person_age_l3145_314504

/-- Given a group of 10 persons, prove that if replacing a 40-year-old person
    with a new person decreases the average age by 3 years, then the age of
    the new person is 10 years. -/
theorem new_person_age (T : ℕ) (A : ℕ) : 
  (T / 10 : ℚ) - ((T - 40 + A) / 10 : ℚ) = 3 → A = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l3145_314504


namespace NUMINAMATH_CALUDE_min_voters_for_giraffe_contest_l3145_314532

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure :=
  (total_voters : ℕ)
  (num_districts : ℕ)
  (num_sections_per_district : ℕ)
  (voters_per_section : ℕ)
  (h_total : total_voters = num_districts * num_sections_per_district * voters_per_section)

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : ℕ :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let sections_to_win := (vs.num_sections_per_district + 1) / 2
  let voters_to_win_section := (vs.voters_per_section + 1) / 2
  districts_to_win * sections_to_win * voters_to_win_section

/-- Theorem stating the minimum number of voters required to win the contest -/
theorem min_voters_for_giraffe_contest :
  ∀ (vs : VotingStructure),
  vs.total_voters = 105 ∧
  vs.num_districts = 5 ∧
  vs.num_sections_per_district = 7 ∧
  vs.voters_per_section = 3 →
  min_voters_to_win vs = 24 := by
  sorry

#eval min_voters_to_win {
  total_voters := 105,
  num_districts := 5,
  num_sections_per_district := 7,
  voters_per_section := 3,
  h_total := rfl
}

end NUMINAMATH_CALUDE_min_voters_for_giraffe_contest_l3145_314532


namespace NUMINAMATH_CALUDE_triangle_inequality_l3145_314578

/-- The inequality for triangle sides and area -/
theorem triangle_inequality (a b c : ℝ) (Δ : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : Δ > 0) : 
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 ∧
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l3145_314578


namespace NUMINAMATH_CALUDE_lucius_weekly_earnings_l3145_314559

/-- Lucius's small business model --/
structure Business where
  daily_ingredient_cost : ℝ
  french_fries_price : ℝ
  poutine_price : ℝ
  tax_rate : ℝ
  daily_french_fries_sold : ℝ
  daily_poutine_sold : ℝ

/-- Calculate weekly earnings after taxes and expenses --/
def weekly_earnings_after_taxes_and_expenses (b : Business) : ℝ :=
  let daily_revenue := b.french_fries_price * b.daily_french_fries_sold + b.poutine_price * b.daily_poutine_sold
  let weekly_revenue := daily_revenue * 7
  let weekly_expenses := b.daily_ingredient_cost * 7
  let taxable_income := weekly_revenue
  let tax := taxable_income * b.tax_rate
  weekly_revenue - weekly_expenses - tax

/-- Theorem stating Lucius's weekly earnings --/
theorem lucius_weekly_earnings :
  ∃ (b : Business),
    b.daily_ingredient_cost = 10 ∧
    b.french_fries_price = 12 ∧
    b.poutine_price = 8 ∧
    b.tax_rate = 0.1 ∧
    b.daily_french_fries_sold = 1 ∧
    b.daily_poutine_sold = 1 ∧
    weekly_earnings_after_taxes_and_expenses b = 56 := by
  sorry


end NUMINAMATH_CALUDE_lucius_weekly_earnings_l3145_314559


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l3145_314562

theorem jordan_rectangle_width (carol_length carol_width jordan_length : ℝ)
  (h1 : carol_length = 15)
  (h2 : carol_width = 24)
  (h3 : jordan_length = 8)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_jordan_rectangle_width_l3145_314562


namespace NUMINAMATH_CALUDE_line_equation_proof_l3145_314591

theorem line_equation_proof (m b k : ℝ) : 
  (∃! k, ∀ y₁ y₂, y₁ = k^2 + 6*k + 5 ∧ y₂ = m*k + b → |y₁ - y₂| = 7) →
  (8 = 2*m + b) →
  (b ≠ 0) →
  (m = 10 ∧ b = -12) :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3145_314591


namespace NUMINAMATH_CALUDE_smallest_repeating_block_8_13_l3145_314540

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def repeating_block (l : List ℕ) : List ℕ := sorry

theorem smallest_repeating_block_8_13 :
  (repeating_block (decimal_expansion 8 13)).length = 6 := by sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_8_13_l3145_314540


namespace NUMINAMATH_CALUDE_expression_factorization_l3145_314595

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 45 * x^2 - 15) - (-3 * x^3 + 6 * x^2 - 3) = 3 * (5 * x^3 + 13 * x^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3145_314595


namespace NUMINAMATH_CALUDE_polyhedron_edge_intersection_l3145_314522

/-- A polyhedron with a given number of edges -/
structure Polyhedron where
  edges : ℕ

/-- A plane that can intersect edges of a polyhedron -/
structure IntersectingPlane where
  intersectedEdges : ℕ

/-- Represents a convex polyhedron -/
def ConvexPolyhedron (p : Polyhedron) : Prop := sorry

/-- Represents a non-convex polyhedron -/
def NonConvexPolyhedron (p : Polyhedron) : Prop := sorry

/-- The maximum number of edges that can be intersected by a plane in a convex polyhedron -/
def maxIntersectedEdgesConvex (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  ConvexPolyhedron p ∧ plane.intersectedEdges ≤ 68

/-- The number of edges that can be intersected by a plane in a non-convex polyhedron -/
def intersectedEdgesNonConvex (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  NonConvexPolyhedron p ∧ plane.intersectedEdges = 96

/-- The impossibility of intersecting all edges in any polyhedron -/
def cannotIntersectAllEdges (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  plane.intersectedEdges < p.edges

theorem polyhedron_edge_intersection (p : Polyhedron) (plane : IntersectingPlane) 
    (h : p.edges = 100) : 
    (maxIntersectedEdgesConvex p plane) ∧ 
    (∃ (p' : Polyhedron) (plane' : IntersectingPlane), intersectedEdgesNonConvex p' plane') ∧ 
    (cannotIntersectAllEdges p plane) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_edge_intersection_l3145_314522


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l3145_314510

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l3145_314510


namespace NUMINAMATH_CALUDE_parabola_directrix_l3145_314546

/-- The equation of the directrix of the parabola y^2 = 6x is x = -3/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 6*x → (∃ (k : ℝ), k = -3/2 ∧ x = k) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3145_314546


namespace NUMINAMATH_CALUDE_power_value_from_equation_l3145_314564

theorem power_value_from_equation (a b : ℝ) : 
  |a - 2| + (b + 3)^2 = 0 → b^a = 9 := by
sorry

end NUMINAMATH_CALUDE_power_value_from_equation_l3145_314564


namespace NUMINAMATH_CALUDE_cupcake_distribution_l3145_314552

def dozen : ℕ := 12

theorem cupcake_distribution (total_dozens : ℕ) (cupcakes_per_cousin : ℕ) : 
  total_dozens = 4 → cupcakes_per_cousin = 3 → (dozen * total_dozens) / cupcakes_per_cousin = 16 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l3145_314552


namespace NUMINAMATH_CALUDE_project_contribution_balance_l3145_314575

/-- The contribution of the first worker to the project -/
def first_worker_contribution : ℚ := 1/3

/-- The contribution of the second worker to the project -/
def second_worker_contribution : ℚ := 1/3

/-- The contribution of the third worker to the project -/
def third_worker_contribution : ℚ := 1/3

/-- The total full-time equivalent (FTE) for the project -/
def total_fte : ℚ := 1

theorem project_contribution_balance :
  first_worker_contribution + second_worker_contribution + third_worker_contribution = total_fte :=
sorry

end NUMINAMATH_CALUDE_project_contribution_balance_l3145_314575


namespace NUMINAMATH_CALUDE_parakeet_consumption_l3145_314511

/-- Represents the daily birdseed consumption of each bird type and the total for a week -/
structure BirdseedConsumption where
  parakeet : ℝ
  parrot : ℝ
  finch : ℝ
  total_weekly : ℝ

/-- The number of each type of bird -/
structure BirdCounts where
  parakeets : ℕ
  parrots : ℕ
  finches : ℕ

/-- Calculates the total daily consumption for all birds -/
def total_daily_consumption (c : BirdseedConsumption) (b : BirdCounts) : ℝ :=
  c.parakeet * b.parakeets + c.parrot * b.parrots + c.finch * b.finches

/-- Theorem stating the daily consumption of each parakeet -/
theorem parakeet_consumption (c : BirdseedConsumption) (b : BirdCounts) :
  c.parakeet = 2 ∧
  c.parrot = 14 ∧
  c.finch = c.parakeet / 2 ∧
  b.parakeets = 3 ∧
  b.parrots = 2 ∧
  b.finches = 4 ∧
  c.total_weekly = 266 ∧
  c.total_weekly = 7 * total_daily_consumption c b :=
by sorry

end NUMINAMATH_CALUDE_parakeet_consumption_l3145_314511


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l3145_314579

theorem smallest_m_no_real_roots : 
  ∃ (m : ℤ), (∀ (n : ℤ), n < m → ∃ (x : ℝ), 3*x*(n*x-5) - x^2 + 8 = 0) ∧
             (∀ (x : ℝ), 3*x*(m*x-5) - x^2 + 8 ≠ 0) ∧
             m = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l3145_314579


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3145_314567

-- Define the quadratic function f(x) = ax^2 + 2x + c
def f (a c x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- Define the quadratic function g(x) = -cx^2 + 2x - a
def g (a c x : ℝ) : ℝ := -c * x^2 + 2 * x - a

-- Theorem statement
theorem solution_set_equivalence (a c : ℝ) :
  (∀ x, -1/3 < x ∧ x < 1/2 → f a c x > 0) →
  (∀ x, g a c x > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3145_314567


namespace NUMINAMATH_CALUDE_petri_dishes_count_l3145_314519

/-- The number of petri dishes in a biology lab -/
def number_of_petri_dishes : ℕ :=
  let total_germs : ℕ := 3600  -- 0.036 * 10^5 = 3600
  let germs_per_dish : ℕ := 80 -- Approximating 79.99999999999999 to 80
  total_germs / germs_per_dish

theorem petri_dishes_count : number_of_petri_dishes = 45 := by
  sorry

end NUMINAMATH_CALUDE_petri_dishes_count_l3145_314519


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3145_314558

theorem quadratic_solution_sum (a b : ℕ+) (x : ℝ) :
  x^2 + 16*x = 100 ∧ x > 0 ∧ x = Real.sqrt a - b → a + b = 172 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3145_314558


namespace NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l3145_314505

theorem absolute_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 6) 
  (h2 : p + q = 7) : 
  |p - q| = 5 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l3145_314505


namespace NUMINAMATH_CALUDE_dining_group_size_l3145_314542

theorem dining_group_size (total_bill : ℝ) (tip_percentage : ℝ) (individual_payment : ℝ) : 
  total_bill = 139 ∧ tip_percentage = 0.1 ∧ individual_payment = 25.48 →
  Int.floor ((total_bill * (1 + tip_percentage)) / individual_payment) = 6 := by
sorry

end NUMINAMATH_CALUDE_dining_group_size_l3145_314542


namespace NUMINAMATH_CALUDE_correct_algebraic_expression_l3145_314515

-- Define the set of possible expressions
inductive AlgebraicExpression
  | MixedNumber : AlgebraicExpression  -- represents 1½a
  | ExplicitMultiplication : AlgebraicExpression  -- represents a × b
  | DivisionSign : AlgebraicExpression  -- represents a ÷ b
  | ImplicitMultiplication : AlgebraicExpression  -- represents 2a

-- Define the property of being correctly written
def isCorrectlyWritten (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.ImplicitMultiplication => True
  | _ => False

-- Theorem statement
theorem correct_algebraic_expression :
  isCorrectlyWritten AlgebraicExpression.ImplicitMultiplication ∧
  ¬isCorrectlyWritten AlgebraicExpression.MixedNumber ∧
  ¬isCorrectlyWritten AlgebraicExpression.ExplicitMultiplication ∧
  ¬isCorrectlyWritten AlgebraicExpression.DivisionSign :=
sorry

end NUMINAMATH_CALUDE_correct_algebraic_expression_l3145_314515


namespace NUMINAMATH_CALUDE_area_of_region_t_l3145_314573

/-- Represents a rhombus PQRS -/
structure Rhombus where
  side_length : ℝ
  angle_q : ℝ

/-- Represents the region T inside the rhombus -/
def region_t (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of a given set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the area of region T in the given rhombus -/
theorem area_of_region_t (r : Rhombus) 
  (h1 : r.side_length = 4) 
  (h2 : r.angle_q = 150 * π / 180) : 
  abs (area (region_t r) - 1.034) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_t_l3145_314573


namespace NUMINAMATH_CALUDE_g_ln_inverse_2017_l3145_314503

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

axiom a_positive : a > 0
axiom a_not_one : a ≠ 1
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom g_ln_2017 : g (Real.log 2017) = 2018

theorem g_ln_inverse_2017 : g (Real.log (1 / 2017)) = -2015 := by
  sorry

end NUMINAMATH_CALUDE_g_ln_inverse_2017_l3145_314503


namespace NUMINAMATH_CALUDE_hotel_guests_count_l3145_314530

/-- The number of guests attending at least one reunion -/
def total_guests (oates_guests hall_guests both_guests : ℕ) : ℕ :=
  oates_guests + hall_guests - both_guests

/-- Theorem stating the total number of guests attending at least one reunion -/
theorem hotel_guests_count :
  total_guests 70 52 28 = 94 := by
  sorry

end NUMINAMATH_CALUDE_hotel_guests_count_l3145_314530


namespace NUMINAMATH_CALUDE_nth_equation_l3145_314588

theorem nth_equation (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l3145_314588


namespace NUMINAMATH_CALUDE_min_diff_composites_sum_96_l3145_314517

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem min_diff_composites_sum_96 : 
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → c + d = 96 → 
  (c : ℤ) - (d : ℤ) ≥ 2 ∨ (d : ℤ) - (c : ℤ) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_diff_composites_sum_96_l3145_314517


namespace NUMINAMATH_CALUDE_day_shift_percentage_l3145_314580

theorem day_shift_percentage
  (excel_percentage : ℝ)
  (excel_and_night_percentage : ℝ)
  (h1 : excel_percentage = 0.20)
  (h2 : excel_and_night_percentage = 0.06) :
  1 - (excel_and_night_percentage / excel_percentage) = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_day_shift_percentage_l3145_314580


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3145_314512

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3145_314512


namespace NUMINAMATH_CALUDE_triangle_property_l3145_314544

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- The angle at C is a right angle
  sorry

def altitude_meets_AB (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- The altitude from C meets AB at D
  sorry

def integer_sides (t : Triangle) : Prop :=
  -- The lengths of the sides of triangle ABC are integers
  sorry

def BD_length (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- BD = 29³
  sorry

def cos_B_fraction (t : Triangle) (m n : ℕ) : Prop :=
  -- cos B = m/n, where m and n are relatively prime positive integers
  sorry

theorem triangle_property (t : Triangle) (D : ℝ × ℝ) (m n : ℕ) :
  is_right_triangle t →
  altitude_meets_AB t D →
  integer_sides t →
  BD_length t D →
  cos_B_fraction t m n →
  m + n = 450 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3145_314544


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3145_314597

def solution_set (a b c : ℝ) : Set ℝ :=
  {x : ℝ | x ≤ -3 ∨ x ≥ 4}

theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : solution_set a b c = {x : ℝ | a * x^2 + b * x + c ≥ 0}) :
  (a > 0) ∧ 
  ({x : ℝ | c * x^2 - b * x + a < 0} = {x : ℝ | x < -1/4 ∨ x > 1/3}) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3145_314597


namespace NUMINAMATH_CALUDE_heath_carrot_planting_l3145_314539

/-- Calculates the number of carrots planted per hour given the number of rows, plants per row, and total planting time. -/
def carrots_per_hour (rows : ℕ) (plants_per_row : ℕ) (total_hours : ℕ) : ℕ :=
  (rows * plants_per_row) / total_hours

/-- Proves that given 400 rows of carrots, 300 plants per row, and 20 hours of planting time, the number of carrots planted per hour is 6,000. -/
theorem heath_carrot_planting :
  carrots_per_hour 400 300 20 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_heath_carrot_planting_l3145_314539


namespace NUMINAMATH_CALUDE_clock_angle_4_to_545_l3145_314509

-- Define the clock structure
structure Clock :=
  (numbers : Nat)
  (angle_between_numbers : Real)
  (divisions_between_numbers : Nat)
  (angle_between_divisions : Real)

-- Define the function to calculate the angle turned by the hour hand
def angle_turned (c : Clock) (start_hour : Nat) (end_hour : Nat) (end_minute : Nat) : Real :=
  sorry

-- Theorem statement
theorem clock_angle_4_to_545 (c : Clock) :
  c.numbers = 12 →
  c.angle_between_numbers = 30 →
  c.divisions_between_numbers = 5 →
  c.angle_between_divisions = 6 →
  angle_turned c 4 5 45 = 52.5 :=
sorry

end NUMINAMATH_CALUDE_clock_angle_4_to_545_l3145_314509


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l3145_314545

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 8 ∧ 
  (∀ w x y z : ℝ, (w^2 + x^2 + y^2 + z^2)^3 ≤ k * (w^6 + x^6 + y^6 + z^6)) ∧
  (∀ k' : ℕ, k' < k → 
    ∃ w x y z : ℝ, (w^2 + x^2 + y^2 + z^2)^3 > k' * (w^6 + x^6 + y^6 + z^6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l3145_314545


namespace NUMINAMATH_CALUDE_distance_to_origin_l3145_314576

def z₁ : ℂ := Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem distance_to_origin (z : ℂ := z₁ * z₂) :
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3145_314576


namespace NUMINAMATH_CALUDE_guppies_count_l3145_314547

/-- The number of guppies Rick bought -/
def guppies : ℕ := sorry

/-- The number of clowns Tim bought -/
def clowns : ℕ := sorry

/-- The number of tetras bought -/
def tetras : ℕ := sorry

/-- The total number of animals bought -/
def total_animals : ℕ := 330

theorem guppies_count :
  (tetras = 4 * clowns) →
  (clowns = 2 * guppies) →
  (guppies + clowns + tetras = total_animals) →
  guppies = 30 := by sorry

end NUMINAMATH_CALUDE_guppies_count_l3145_314547


namespace NUMINAMATH_CALUDE_triangle_properties_l3145_314599

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = (2 * Real.sqrt 3 / 3) * t.b * Real.sin (t.A + π/3))
  (h2 : t.a + t.c = 4) :
  t.B = π/3 ∧ 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3145_314599


namespace NUMINAMATH_CALUDE_max_roads_in_graphia_l3145_314598

/-- A graph representing the towns and roads in Graphia. -/
structure Graphia where
  towns : Finset Nat
  roads : Finset (Nat × Nat)

/-- The number of towns in Graphia. -/
def num_towns : Nat := 100

/-- A function representing Peter's travel pattern. -/
def peter_travel (g : Graphia) (start : Nat) : List Nat := sorry

/-- The condition that each town is visited exactly twice. -/
def all_towns_visited_twice (g : Graphia) : Prop :=
  ∀ t ∈ g.towns, (List.count t (List.join (List.map (peter_travel g) (List.range num_towns)))) = 2

/-- The theorem stating the maximum number of roads in Graphia. -/
theorem max_roads_in_graphia :
  ∀ g : Graphia,
    g.towns.card = num_towns →
    all_towns_visited_twice g →
    g.roads.card ≤ 4851 :=
sorry

end NUMINAMATH_CALUDE_max_roads_in_graphia_l3145_314598


namespace NUMINAMATH_CALUDE_bones_fraction_in_beef_l3145_314514

/-- The price of beef with bones in rubles per kilogram -/
def price_beef_with_bones : ℝ := 78

/-- The price of boneless beef in rubles per kilogram -/
def price_boneless_beef : ℝ := 90

/-- The price of bones in rubles per kilogram -/
def price_bones : ℝ := 15

/-- The fraction of bones in a kilogram of beef -/
def fraction_bones : ℝ := 0.16

theorem bones_fraction_in_beef :
  price_bones * fraction_bones + price_boneless_beef * (1 - fraction_bones) = price_beef_with_bones :=
sorry

end NUMINAMATH_CALUDE_bones_fraction_in_beef_l3145_314514


namespace NUMINAMATH_CALUDE_rope_folded_six_times_l3145_314549

/-- The number of segments a rope is cut into after being folded n times and cut along the middle -/
def rope_segments (n : ℕ) : ℕ := 2^n + 1

/-- Theorem: A rope folded in half 6 times and cut along the middle will result in 65 segments -/
theorem rope_folded_six_times : rope_segments 6 = 65 := by
  sorry

end NUMINAMATH_CALUDE_rope_folded_six_times_l3145_314549


namespace NUMINAMATH_CALUDE_video_game_theorem_l3145_314500

def video_game_problem (x : ℝ) (n : ℕ) (y : ℝ) : Prop :=
  x > 0 ∧ n > 0 ∧ y > 0 ∧
  (1/4 : ℝ) * x = (1/2 : ℝ) * n * y ∧
  (1/3 : ℝ) * x = x - ((1/2 : ℝ) * x + (1/6 : ℝ) * x)

theorem video_game_theorem (x : ℝ) (n : ℕ) (y : ℝ) 
  (h : video_game_problem x n y) : True :=
by
  sorry

end NUMINAMATH_CALUDE_video_game_theorem_l3145_314500


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3145_314565

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 4*i) / (1 + i) = (7:ℂ)/2 + (1:ℂ)/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3145_314565


namespace NUMINAMATH_CALUDE_log_power_sum_l3145_314596

/-- Given a = log 25 and b = log 36, prove that 5^(a/b) + 6^(b/a) = 11 -/
theorem log_power_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by sorry

end NUMINAMATH_CALUDE_log_power_sum_l3145_314596


namespace NUMINAMATH_CALUDE_solutions_sum_greater_than_two_l3145_314529

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2/x

-- State the theorem
theorem solutions_sum_greater_than_two 
  (t : ℝ) 
  (h_t : t > 3) 
  (x₁ x₂ : ℝ) 
  (h_x₁ : x₁ > 0) 
  (h_x₂ : x₂ > 0) 
  (h_x₁_neq_x₂ : x₁ ≠ x₂) 
  (h_f_x₁ : f x₁ = t) 
  (h_f_x₂ : f x₂ = t) : 
  x₁ + x₂ > 2 := by
  sorry

end

end NUMINAMATH_CALUDE_solutions_sum_greater_than_two_l3145_314529


namespace NUMINAMATH_CALUDE_exists_unique_equal_power_point_equal_power_point_is_orthogonal_circle_center_l3145_314508

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The power of a point with respect to a circle -/
def powerOfPoint (p : ℝ × ℝ) (c : Circle) : ℝ :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 - c.radius^2

/-- Theorem: Given three circles, there exists a unique point with equal power to all three circles -/
theorem exists_unique_equal_power_point (c1 c2 c3 : Circle) :
  ∃! p : ℝ × ℝ, powerOfPoint p c1 = powerOfPoint p c2 ∧ powerOfPoint p c2 = powerOfPoint p c3 :=
sorry

/-- Theorem: The point with equal power to three circles is the center of a circle 
    that intersects the three given circles at right angles -/
theorem equal_power_point_is_orthogonal_circle_center 
  (c1 c2 c3 : Circle) (p : ℝ × ℝ) 
  (h : powerOfPoint p c1 = powerOfPoint p c2 ∧ powerOfPoint p c2 = powerOfPoint p c3) :
  ∃ r : ℝ, ∀ i : Fin 3, 
    let c := Circle.mk p r
    let ci := [c1, c2, c3].get i
    ∃ x : ℝ × ℝ, (x.1 - c.center.1)^2 + (x.2 - c.center.2)^2 = c.radius^2 ∧
                 (x.1 - ci.center.1)^2 + (x.2 - ci.center.2)^2 = ci.radius^2 ∧
                 ((x.1 - c.center.1) * (x.1 - ci.center.1) + (x.2 - c.center.2) * (x.2 - ci.center.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_exists_unique_equal_power_point_equal_power_point_is_orthogonal_circle_center_l3145_314508


namespace NUMINAMATH_CALUDE_probability_both_red_probability_different_colors_l3145_314590

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents a ball with a label -/
structure Ball where
  color : Color
  label : String

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of balls in the bag -/
def total_balls : Nat := 6

/-- The number of red balls -/
def red_balls : Nat := 4

/-- The number of black balls -/
def black_balls : Nat := 2

/-- The set of all possible combinations when drawing 2 balls -/
def all_combinations : Finset (Ball × Ball) := sorry

/-- The set of combinations where both balls are red -/
def both_red : Finset (Ball × Ball) := sorry

/-- The set of combinations where the balls have different colors -/
def different_colors : Finset (Ball × Ball) := sorry

theorem probability_both_red :
  (Finset.card both_red : ℚ) / (Finset.card all_combinations : ℚ) = 2 / 5 := by sorry

theorem probability_different_colors :
  (Finset.card different_colors : ℚ) / (Finset.card all_combinations : ℚ) = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_probability_both_red_probability_different_colors_l3145_314590


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l3145_314525

/-- Calculates the difference in profit shares between two partners given their investments and the profit share of a third partner. -/
def profit_share_difference (invest_a invest_b invest_c b_profit : ℚ) : ℚ :=
  let total_invest := invest_a + invest_b + invest_c
  let total_profit := (total_invest * b_profit) / invest_b
  let a_share := (invest_a * total_profit) / total_invest
  let c_share := (invest_c * total_profit) / total_invest
  c_share - a_share

/-- Given the investments of A, B, and C as 8000, 10000, and 12000 respectively,
    and B's profit share as 1800, the difference between A's and C's profit shares is 720. -/
theorem profit_share_difference_example :
  profit_share_difference 8000 10000 12000 1800 = 720 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_difference_example_l3145_314525


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3145_314533

/-- Given an arithmetic sequence with first term 2 and common difference 5,
    the 50th term of this sequence is 247. -/
theorem arithmetic_sequence_50th_term :
  let a : ℕ → ℕ := λ n => 2 + (n - 1) * 5
  a 50 = 247 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3145_314533


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3145_314556

/-- A parabola in the Cartesian plane -/
structure Parabola where
  /-- The parameter p of the parabola -/
  p : ℝ
  /-- The vertex is at the origin -/
  vertex_at_origin : True
  /-- The focus is on the x-axis -/
  focus_on_x_axis : True
  /-- The parabola passes through the point (1, 2) -/
  passes_through_point : p = 2

/-- The distance from the focus to the directrix of a parabola -/
def focus_directrix_distance (c : Parabola) : ℝ := c.p

theorem parabola_focus_directrix_distance :
  ∀ c : Parabola, focus_directrix_distance c = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3145_314556


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l3145_314534

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y ∧ y < 4 → f x > f y) →  -- f is decreasing on (0,4)
  (0 < a^2 - a ∧ a^2 - a < 4) →                 -- domain condition
  f (a^2 - a) > f 2 →                           -- given inequality
  (-1 < a ∧ a < 0) ∨ (1 < a ∧ a < 2) :=         -- conclusion
by sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l3145_314534


namespace NUMINAMATH_CALUDE_dividend_calculation_l3145_314535

theorem dividend_calculation (dividend divisor quotient : ℕ) 
  (h1 : dividend = 5 * divisor) 
  (h2 : divisor = 4 * quotient) : 
  dividend = 100 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3145_314535


namespace NUMINAMATH_CALUDE_worker_productivity_ratio_l3145_314501

theorem worker_productivity_ratio :
  ∀ (x y : ℝ),
  (x > 0) →
  (y > 0) →
  (2 * (x + y) = 1) →
  (x / 3 + 3 * y = 1) →
  (y / x = 5 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_worker_productivity_ratio_l3145_314501


namespace NUMINAMATH_CALUDE_range_of_k_l3145_314543

open Set

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}

def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2*k + 1}

theorem range_of_k : ∀ k : ℝ, (Aᶜ ∩ B k = ∅) ↔ (k ≤ 0 ∨ k ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3145_314543


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_sixteen_l3145_314569

theorem sum_of_fractions_equals_sixteen :
  let fractions : List ℚ := [2/10, 4/10, 6/10, 8/10, 10/10, 15/10, 20/10, 25/10, 30/10, 40/10]
  fractions.sum = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_sixteen_l3145_314569


namespace NUMINAMATH_CALUDE_division_equations_for_26_l3145_314554

theorem division_equations_for_26 : 
  {(x, y) : ℕ × ℕ | 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 26 = x * y + 2} = 
  {(3, 8), (4, 6), (6, 4), (8, 3)} := by sorry

end NUMINAMATH_CALUDE_division_equations_for_26_l3145_314554


namespace NUMINAMATH_CALUDE_marked_box_second_row_l3145_314521

/-- Represents the number of cakes in each box of the pyramid -/
structure CakePyramid where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ

/-- The condition that each box in a higher row contains the sum of cakes in the two adjacent boxes below -/
def valid_pyramid (p : CakePyramid) : Prop :=
  p.e = p.a + p.b ∧
  p.f = p.b + p.c ∧
  p.g = p.c + p.d ∧
  p.h = p.e + p.f

/-- The condition that three boxes contain 3, 5, and 6 cakes -/
def marked_boxes (p : CakePyramid) : Prop :=
  (p.a = 3 ∨ p.a = 5 ∨ p.a = 6) ∧
  (p.d = 3 ∨ p.d = 5 ∨ p.d = 6) ∧
  (p.f = 3 ∨ p.f = 5 ∨ p.f = 6) ∧
  p.a ≠ p.d ∧ p.a ≠ p.f ∧ p.d ≠ p.f

/-- The total number of cakes in the pyramid -/
def total_cakes (p : CakePyramid) : ℕ :=
  p.a + p.b + p.c + p.d + p.e + p.f + p.g + p.h

/-- The theorem stating that the marked box in the second row from the bottom contains 3 cakes -/
theorem marked_box_second_row (p : CakePyramid) :
  valid_pyramid p → marked_boxes p → (∀ q : CakePyramid, valid_pyramid q → marked_boxes q → total_cakes q ≥ total_cakes p) → p.f = 3 := by
  sorry


end NUMINAMATH_CALUDE_marked_box_second_row_l3145_314521


namespace NUMINAMATH_CALUDE_condition_implies_isosceles_l3145_314516

-- Define a structure for a triangle in a plane
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to represent a vector from one point to another
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition given in the problem
def satisfies_condition (t : Triangle) : Prop :=
  ∀ O : ℝ × ℝ, 
    let OB := vector O t.B
    let OC := vector O t.C
    let OA := vector O t.A
    dot_product (OB - OC) (OB + OC - 2 • OA) = 0

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  let AB := vector t.A t.B
  let AC := vector t.A t.C
  dot_product AB AB = dot_product AC AC

-- State the theorem
theorem condition_implies_isosceles (t : Triangle) :
  satisfies_condition t → is_isosceles t :=
by
  sorry

end NUMINAMATH_CALUDE_condition_implies_isosceles_l3145_314516


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l3145_314561

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 8, 10} : Set ℤ) →
  (¬(5 ∣ (b * (3 * b^2 - 3 * b - 1))) ↔ (b = 6 ∨ b = 8)) := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l3145_314561


namespace NUMINAMATH_CALUDE_min_value_at_seven_l3145_314583

/-- The quadratic function f(x) = x^2 - 14x + 40 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 40

theorem min_value_at_seven :
  ∀ x : ℝ, f 7 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_at_seven_l3145_314583


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3145_314589

theorem smallest_x_absolute_value_equation :
  ∃ x : ℚ, (∀ y : ℚ, |5 * y - 3| = 45 → x ≤ y) ∧ |5 * x - 3| = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3145_314589


namespace NUMINAMATH_CALUDE_complex_equation_implies_xy_equals_one_l3145_314513

theorem complex_equation_implies_xy_equals_one (x y : ℝ) :
  (x + 1 : ℂ) + y * I = -I + 2 * x →
  x ^ y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_xy_equals_one_l3145_314513


namespace NUMINAMATH_CALUDE_ellipse_to_circle_l3145_314527

theorem ellipse_to_circle 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a ≠ b) 
  (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  x^2 + ((a/b) * y)^2 = a^2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_to_circle_l3145_314527


namespace NUMINAMATH_CALUDE_acid_dilution_l3145_314531

/-- Proves that adding 80 ounces of pure water to 50 ounces of a 26% acid solution
    results in a 10% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.26 →
  water_added = 80 →
  final_concentration = 0.10 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l3145_314531


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l3145_314536

/-- Represents the cost and survival properties of flower types -/
structure FlowerType where
  cost : ℝ
  survivalRate : ℝ

/-- Represents the planting scenario -/
structure PlantingScenario where
  typeA : FlowerType
  typeB : FlowerType
  totalPots : ℕ
  maxReplacement : ℕ

def minimumCost (scenario : PlantingScenario) : ℝ :=
  let m := scenario.totalPots / 2
  m * scenario.typeA.cost + (scenario.totalPots - m) * scenario.typeB.cost

theorem minimum_cost_theorem (scenario : PlantingScenario) :
  scenario.typeA.cost = 30 ∧
  scenario.typeB.cost = 60 ∧
  scenario.totalPots = 400 ∧
  scenario.typeA.survivalRate = 0.7 ∧
  scenario.typeB.survivalRate = 0.9 ∧
  scenario.maxReplacement = 80 ∧
  3 * scenario.typeA.cost + 4 * scenario.typeB.cost = 330 ∧
  4 * scenario.typeA.cost + 3 * scenario.typeB.cost = 300 →
  minimumCost scenario = 18000 :=
sorry

#check minimum_cost_theorem

end NUMINAMATH_CALUDE_minimum_cost_theorem_l3145_314536


namespace NUMINAMATH_CALUDE_tucker_tissues_used_l3145_314560

/-- The number of tissues Tucker used while sick -/
def tissues_used (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_left : ℕ) : ℕ :=
  boxes_bought * tissues_per_box - tissues_left

/-- Theorem stating the number of tissues Tucker used while sick -/
theorem tucker_tissues_used :
  tissues_used 160 3 270 = 210 := by
  sorry

end NUMINAMATH_CALUDE_tucker_tissues_used_l3145_314560


namespace NUMINAMATH_CALUDE_ball_box_arrangements_l3145_314593

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of distinct boxes -/
def num_boxes : ℕ := 4

/-- The number of arrangements when exactly one box remains empty -/
def arrangements_one_empty : ℕ := 144

/-- The number of arrangements when exactly two boxes remain empty -/
def arrangements_two_empty : ℕ := 84

/-- Theorem stating the correct number of arrangements for each case -/
theorem ball_box_arrangements :
  (∀ (n : ℕ), n = num_balls → n = num_boxes) →
  (arrangements_one_empty = 144 ∧ arrangements_two_empty = 84) := by
  sorry

end NUMINAMATH_CALUDE_ball_box_arrangements_l3145_314593


namespace NUMINAMATH_CALUDE_age_ratio_l3145_314574

def arun_future_age : ℕ := 30
def years_to_future : ℕ := 10
def deepak_age : ℕ := 50

theorem age_ratio :
  (arun_future_age - years_to_future) / deepak_age = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l3145_314574


namespace NUMINAMATH_CALUDE_unique_positive_n_l3145_314555

/-- A quadratic equation has exactly one real root if and only if its discriminant is zero -/
axiom discriminant_zero_iff_one_root {a b c : ℝ} (ha : a ≠ 0) :
  b^2 - 4*a*c = 0 ↔ ∃! x, a*x^2 + b*x + c = 0

/-- The quadratic equation y^2 + 6ny + 9n has exactly one real root -/
def has_one_root (n : ℝ) : Prop :=
  ∃! y, y^2 + 6*n*y + 9*n = 0

theorem unique_positive_n :
  ∃! n : ℝ, n > 0 ∧ has_one_root n :=
sorry

end NUMINAMATH_CALUDE_unique_positive_n_l3145_314555


namespace NUMINAMATH_CALUDE_max_four_digit_quotient_l3145_314528

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n ≤ 9

def four_digit_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

def digit_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem max_four_digit_quotient :
  ∀ (a b c d : ℕ),
    is_nonzero_digit a →
    is_digit b →
    is_nonzero_digit c →
    is_nonzero_digit d →
    (four_digit_number a b c d) / (digit_sum a b c d) ≤ 337 :=
by sorry

end NUMINAMATH_CALUDE_max_four_digit_quotient_l3145_314528


namespace NUMINAMATH_CALUDE_purple_yellow_ratio_l3145_314506

/-- Represents the number of flowers of each color in the garden -/
structure GardenFlowers where
  yellow : ℕ
  purple : ℕ
  green : ℕ

/-- Conditions of the garden -/
def gardenConditions (g : GardenFlowers) : Prop :=
  g.yellow = 10 ∧
  g.green = (g.yellow + g.purple) / 4 ∧
  g.yellow + g.purple + g.green = 35

/-- Theorem stating the relationship between purple and yellow flowers -/
theorem purple_yellow_ratio (g : GardenFlowers) 
  (h : gardenConditions g) : g.purple * 10 = g.yellow * 18 := by
  sorry

#check purple_yellow_ratio

end NUMINAMATH_CALUDE_purple_yellow_ratio_l3145_314506


namespace NUMINAMATH_CALUDE_integral_fractional_parts_sum_l3145_314577

theorem integral_fractional_parts_sum (x y : ℝ) : 
  (x = ⌊5 - 2 * Real.sqrt 3⌋) → 
  (y = (5 - 2 * Real.sqrt 3) - ⌊5 - 2 * Real.sqrt 3⌋) → 
  (x + y + 4 / y = 9) := by
  sorry

end NUMINAMATH_CALUDE_integral_fractional_parts_sum_l3145_314577


namespace NUMINAMATH_CALUDE_congruence_problem_l3145_314594

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 151 ∧ (100 * n) % 151 = 93 % 151 → n % 151 = 29 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3145_314594


namespace NUMINAMATH_CALUDE_percentage_problem_l3145_314566

theorem percentage_problem (X : ℝ) (h : 0.2 * X = 300) : 
  ∃ P : ℝ, (P / 100) * X = 1800 ∧ P = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3145_314566


namespace NUMINAMATH_CALUDE_round_trip_ticket_holders_l3145_314548

/-- The percentage of ship passengers holding round-trip tickets -/
def round_trip_percentage : ℝ := 62.5

theorem round_trip_ticket_holders (total_passengers : ℝ) (round_trip_with_car : ℝ) (round_trip_without_car : ℝ)
  (h1 : round_trip_with_car = 0.25 * total_passengers)
  (h2 : round_trip_without_car = 0.6 * (round_trip_with_car + round_trip_without_car)) :
  (round_trip_with_car + round_trip_without_car) / total_passengers * 100 = round_trip_percentage := by
  sorry

end NUMINAMATH_CALUDE_round_trip_ticket_holders_l3145_314548


namespace NUMINAMATH_CALUDE_unique_solution_l3145_314586

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The equation that the four-digit number must satisfy. -/
def SatisfiesEquation (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2*b^6 + 3*c^6 + 4*d^6) = n

/-- The main theorem stating that 2010 is the only four-digit number satisfying the equation. -/
theorem unique_solution :
  ∀ n : ℕ, FourDigitNumber n → SatisfiesEquation n → n = 2010 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3145_314586


namespace NUMINAMATH_CALUDE_eggs_needed_is_84_l3145_314568

/-- Represents the number of eggs in an omelette -/
inductive OmeletteType
  | threeEgg
  | fourEgg

/-- Represents an hour of operation at the cafe -/
structure Hour where
  customers : Nat
  omeletteType : OmeletteType

/-- Represents a day of operation at Theo's cafe -/
structure CafeDay where
  hours : List Hour

/-- Calculates the total number of eggs needed for a given hour -/
def eggsNeededForHour (hour : Hour) : Nat :=
  match hour.omeletteType with
  | OmeletteType.threeEgg => 3 * hour.customers
  | OmeletteType.fourEgg => 4 * hour.customers

/-- Calculates the total number of eggs needed for the entire day -/
def totalEggsNeeded (day : CafeDay) : Nat :=
  day.hours.foldl (fun acc hour => acc + eggsNeededForHour hour) 0

/-- Theorem stating that the total number of eggs needed is 84 -/
theorem eggs_needed_is_84 (day : CafeDay) 
    (h1 : day.hours = [
      { customers := 5, omeletteType := OmeletteType.threeEgg },
      { customers := 7, omeletteType := OmeletteType.fourEgg },
      { customers := 3, omeletteType := OmeletteType.threeEgg },
      { customers := 8, omeletteType := OmeletteType.fourEgg }
    ]) : 
    totalEggsNeeded day = 84 := by
  sorry


end NUMINAMATH_CALUDE_eggs_needed_is_84_l3145_314568


namespace NUMINAMATH_CALUDE_prime_product_divisors_l3145_314518

theorem prime_product_divisors (p q : Nat) (x : Nat) :
  Prime p →
  Prime q →
  (Nat.divisors (p^x * q^5)).card = 30 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_prime_product_divisors_l3145_314518


namespace NUMINAMATH_CALUDE_rectangle_area_l3145_314553

/-- Given a right triangle ABC composed of two right triangles and a rectangle,
    prove that the area of the rectangle is 750 square centimeters. -/
theorem rectangle_area (AE BF : ℝ) (h1 : AE = 30) (h2 : BF = 25) : AE * BF = 750 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3145_314553


namespace NUMINAMATH_CALUDE_modulus_of_neg_one_plus_i_l3145_314585

theorem modulus_of_neg_one_plus_i :
  Complex.abs (-1 + Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_neg_one_plus_i_l3145_314585


namespace NUMINAMATH_CALUDE_g_of_4_equals_26_l3145_314572

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 6

-- Theorem statement
theorem g_of_4_equals_26 : g 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_g_of_4_equals_26_l3145_314572


namespace NUMINAMATH_CALUDE_smallest_odd_island_has_nine_counties_l3145_314550

/-- A rectangular county (graphstum) -/
structure County where
  width : ℕ
  height : ℕ

/-- A rectangular island composed of counties -/
structure Island where
  counties : List County
  isRectangular : Bool
  hasDiagonalRoads : Bool
  hasClosedPath : Bool

/-- The property of having an odd number of counties -/
def hasOddCounties (i : Island) : Prop :=
  Odd (List.length i.counties)

/-- The property of being a valid island configuration -/
def isValidIsland (i : Island) : Prop :=
  i.isRectangular ∧ i.hasDiagonalRoads ∧ i.hasClosedPath ∧ hasOddCounties i

/-- The theorem stating that the smallest valid odd-county island has 9 counties -/
theorem smallest_odd_island_has_nine_counties :
  ∀ i : Island, isValidIsland i → List.length i.counties ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_island_has_nine_counties_l3145_314550


namespace NUMINAMATH_CALUDE_average_score_three_subjects_l3145_314581

theorem average_score_three_subjects 
  (math_score : ℝ)
  (korean_english_avg : ℝ)
  (h1 : math_score = 100)
  (h2 : korean_english_avg = 88) : 
  (math_score + 2 * korean_english_avg) / 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_average_score_three_subjects_l3145_314581


namespace NUMINAMATH_CALUDE_least_possible_qr_l3145_314541

/-- Triangle PQR with side lengths -/
structure TrianglePQR where
  pq : ℝ
  pr : ℝ
  qr : ℝ
  pq_positive : 0 < pq
  pr_positive : 0 < pr
  qr_positive : 0 < qr
  triangle_inequality_1 : qr < pq + pr
  triangle_inequality_2 : pq < qr + pr
  triangle_inequality_3 : pr < pq + qr

/-- Triangle SQR with side lengths -/
structure TriangleSQR where
  sq : ℝ
  sr : ℝ
  qr : ℝ
  sq_positive : 0 < sq
  sr_positive : 0 < sr
  qr_positive : 0 < qr
  triangle_inequality_1 : qr < sq + sr
  triangle_inequality_2 : sq < qr + sr
  triangle_inequality_3 : sr < sq + qr

/-- The theorem stating the least possible integral length of QR -/
theorem least_possible_qr 
  (triangle_pqr : TrianglePQR)
  (triangle_sqr : TriangleSQR)
  (h_pq : triangle_pqr.pq = 7)
  (h_pr : triangle_pqr.pr = 10)
  (h_sq : triangle_sqr.sq = 24)
  (h_sr : triangle_sqr.sr = 15)
  (h_qr_same : triangle_pqr.qr = triangle_sqr.qr)
  (h_qr_int : ∃ n : ℕ, triangle_pqr.qr = n) :
  triangle_pqr.qr = 9 ∧ ∀ m : ℕ, (m : ℝ) = triangle_pqr.qr → m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_least_possible_qr_l3145_314541


namespace NUMINAMATH_CALUDE_pizza_slice_difference_l3145_314520

/-- Given a pizza with 78 slices shared in a ratio of 5:8, prove that the difference
    between the waiter's share and 20 less than the waiter's share is 20 slices. -/
theorem pizza_slice_difference (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) : 
  total_slices = 78 → 
  buzz_ratio = 5 → 
  waiter_ratio = 8 → 
  let waiter_share := (waiter_ratio * total_slices) / (buzz_ratio + waiter_ratio)
  waiter_share - (waiter_share - 20) = 20 :=
by sorry

end NUMINAMATH_CALUDE_pizza_slice_difference_l3145_314520
