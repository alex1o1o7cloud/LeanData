import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1094_109484

theorem quadratic_root_transformation (r s : ℝ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  (r + s = -4/3) →
  (r * s = 2/3) →
  ∃ q : ℝ, r^3 + s^3 = -16/27 ∧ r^3 * s^3 = q ∧ 
    ∀ x : ℝ, x^2 + (16/27) * x + q = 0 ↔ (x = r^3 ∨ x = s^3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1094_109484


namespace NUMINAMATH_CALUDE_alice_commission_percentage_l1094_109433

/-- Alice's sales data and commission calculation --/
theorem alice_commission_percentage
  (sales : ℝ)
  (basic_salary : ℝ)
  (savings : ℝ)
  (savings_percentage : ℝ)
  (h1 : sales = 2500)
  (h2 : basic_salary = 240)
  (h3 : savings = 29)
  (h4 : savings_percentage = 0.1)
  (h5 : savings = savings_percentage * (basic_salary + sales * commission_percentage)) :
  commission_percentage = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_alice_commission_percentage_l1094_109433


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1094_109455

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 
  1 / x + 1 / y = 8 / 75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1094_109455


namespace NUMINAMATH_CALUDE_quartic_sum_l1094_109428

/-- A quartic polynomial Q with specific values at 0, 1, and -1 -/
def QuarticPolynomial (m : ℝ) : ℝ → ℝ := sorry

/-- Properties of the QuarticPolynomial -/
axiom quartic_prop_0 (m : ℝ) : QuarticPolynomial m 0 = m
axiom quartic_prop_1 (m : ℝ) : QuarticPolynomial m 1 = 3 * m
axiom quartic_prop_neg1 (m : ℝ) : QuarticPolynomial m (-1) = 2 * m

/-- Theorem: For a quartic polynomial Q with Q(0) = m, Q(1) = 3m, and Q(-1) = 2m, Q(3) + Q(-3) = 56m -/
theorem quartic_sum (m : ℝ) : 
  QuarticPolynomial m 3 + QuarticPolynomial m (-3) = 56 * m := by sorry

end NUMINAMATH_CALUDE_quartic_sum_l1094_109428


namespace NUMINAMATH_CALUDE_exists_positive_solution_l1094_109494

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a * b^2 + 3 * b - a

/-- Theorem stating the existence of a positive solution -/
theorem exists_positive_solution :
  ∃ x : ℝ, x > 0 ∧ star 5 x = 100 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_solution_l1094_109494


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1094_109453

theorem complex_square_simplification :
  (4 - 3 * Complex.I) ^ 2 = 7 - 24 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1094_109453


namespace NUMINAMATH_CALUDE_problem_solution_l1094_109444

theorem problem_solution (c d : ℚ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1094_109444


namespace NUMINAMATH_CALUDE_geometry_textbook_weight_l1094_109401

/-- The weight of Kelly's chemistry textbook in pounds -/
def chemistry_weight : ℝ := 7.125

/-- The weight difference between the chemistry and geometry textbooks in pounds -/
def weight_difference : ℝ := 6.5

/-- The weight of Kelly's geometry textbook in pounds -/
def geometry_weight : ℝ := chemistry_weight - weight_difference

theorem geometry_textbook_weight :
  geometry_weight = 0.625 := by sorry

end NUMINAMATH_CALUDE_geometry_textbook_weight_l1094_109401


namespace NUMINAMATH_CALUDE_P_on_y_axis_l1094_109418

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the y-axis -/
def is_on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- The point P with coordinates (0, -4) -/
def P : Point :=
  { x := 0, y := -4 }

/-- Theorem: The point P(0, -4) lies on the y-axis -/
theorem P_on_y_axis : is_on_y_axis P := by
  sorry

end NUMINAMATH_CALUDE_P_on_y_axis_l1094_109418


namespace NUMINAMATH_CALUDE_smallest_n_value_l1094_109470

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) : 
  (∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N) → N = 80 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l1094_109470


namespace NUMINAMATH_CALUDE_land_division_theorem_l1094_109400

/-- Represents a rectangular piece of land --/
structure Land where
  length : ℝ
  width : ℝ

/-- Represents a division of land into three sections --/
structure LandDivision where
  section1 : Land
  section2 : Land
  section3 : Land

def Land.area (l : Land) : ℝ := l.length * l.width

def LandDivision.isValid (ld : LandDivision) (totalLand : Land) : Prop :=
  ld.section1.area + ld.section2.area + ld.section3.area = totalLand.area ∧
  ld.section1.area = ld.section2.area ∧
  ld.section2.area = ld.section3.area

def LandDivision.fenceLength (ld : LandDivision) : ℝ :=
  ld.section1.length + ld.section2.length + ld.section3.length

def countValidDivisions (totalLand : Land) : ℕ :=
  sorry

def minFenceLength (totalLand : Land) : ℝ :=
  sorry

theorem land_division_theorem (totalLand : Land) 
  (h1 : totalLand.length = 25)
  (h2 : totalLand.width = 36) :
  countValidDivisions totalLand = 4 ∧ 
  minFenceLength totalLand = 49 := by
  sorry

end NUMINAMATH_CALUDE_land_division_theorem_l1094_109400


namespace NUMINAMATH_CALUDE_inequality_solution_l1094_109458

theorem inequality_solution (x : ℝ) : (x^2 - 4) / (x^2 - 9) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1094_109458


namespace NUMINAMATH_CALUDE_part_one_part_two_l1094_109434

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2 * m}

-- Theorem for part 1
theorem part_one :
  A ∩ (U \ B 3) = {x | 0 ≤ x ∧ x < 1} := by sorry

-- Theorem for part 2
theorem part_two :
  ∀ m : ℝ, A ∪ B m = B m ↔ 3/2 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1094_109434


namespace NUMINAMATH_CALUDE_intersection_perpendicular_implies_m_value_l1094_109440

/-- The curve C: x^2 + y^2 - 2x - 4y + m = 0 -/
def C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

/-- The line L: x + 2y - 3 = 0 -/
def L (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

/-- M and N are points on both C and L -/
def intersection_points (m : ℝ) (M N : ℝ × ℝ) : Prop :=
  C m M.1 M.2 ∧ C m N.1 N.2 ∧ L M.1 M.2 ∧ L N.1 N.2

/-- OM is perpendicular to ON -/
def perpendicular (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

theorem intersection_perpendicular_implies_m_value (m : ℝ) (M N : ℝ × ℝ) :
  intersection_points m M N → perpendicular M N → m = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_implies_m_value_l1094_109440


namespace NUMINAMATH_CALUDE_lindas_minimum_hours_l1094_109408

/-- Calculates the minimum number of hours Linda needs to work to cover college application fees --/
def minimum_hours_to_cover_fees (initial_rate : ℚ) (raise_percentage : ℚ) (hours_per_raise : ℕ) 
  (num_applications : ℕ) (fee_per_application : ℚ) : ℕ :=
  sorry

theorem lindas_minimum_hours : 
  minimum_hours_to_cover_fees 10 (1/10) 5 6 25 = 15 := by sorry

end NUMINAMATH_CALUDE_lindas_minimum_hours_l1094_109408


namespace NUMINAMATH_CALUDE_alyssa_marbles_cost_l1094_109437

/-- The amount Alyssa spent on marbles -/
def marbles_cost (football_cost total_cost : ℚ) : ℚ :=
  total_cost - football_cost

/-- Proof that Alyssa spent $6.59 on marbles -/
theorem alyssa_marbles_cost :
  let football_cost : ℚ := 571/100
  let total_cost : ℚ := 1230/100
  marbles_cost football_cost total_cost = 659/100 := by
sorry

end NUMINAMATH_CALUDE_alyssa_marbles_cost_l1094_109437


namespace NUMINAMATH_CALUDE_michael_basketball_points_l1094_109441

theorem michael_basketball_points :
  ∀ (junior_points : ℝ),
    (junior_points + (junior_points * 1.2) = 572) →
    junior_points = 260 := by
  sorry

end NUMINAMATH_CALUDE_michael_basketball_points_l1094_109441


namespace NUMINAMATH_CALUDE_james_running_distance_l1094_109411

/-- Proves that given the conditions of the problem, the initial running distance was 600 miles per week -/
theorem james_running_distance (initial_distance : ℝ) : 
  (initial_distance + 40 * 3 = 1.2 * initial_distance) → 
  initial_distance = 600 := by
  sorry

end NUMINAMATH_CALUDE_james_running_distance_l1094_109411


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sevenths_l1094_109461

theorem smallest_fraction_greater_than_five_sevenths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (5 : ℚ) / 7 < (a : ℚ) / b →
    (68 : ℚ) / 95 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_five_sevenths_l1094_109461


namespace NUMINAMATH_CALUDE_two_out_of_five_permutation_l1094_109495

theorem two_out_of_five_permutation : 
  (Finset.range 5).card * (Finset.range 4).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_five_permutation_l1094_109495


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l1094_109469

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l1094_109469


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l1094_109477

/-- Given that Bryan has 56 books in each bookshelf and 504 books in total,
    prove that he has 9 bookshelves. -/
theorem bryans_bookshelves (books_per_shelf : ℕ) (total_books : ℕ) 
    (h1 : books_per_shelf = 56) (h2 : total_books = 504) :
    total_books / books_per_shelf = 9 := by
  sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l1094_109477


namespace NUMINAMATH_CALUDE_simplify_expression_l1094_109460

theorem simplify_expression (m n : ℝ) : m - n - (m + n) = -2 * n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1094_109460


namespace NUMINAMATH_CALUDE_range_of_m_l1094_109459

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the condition for f having two zeros
def has_two_zeros (m : ℝ) : Prop := ∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0

-- Define the condition q
def condition_q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, has_two_zeros m ∧ ¬(condition_q m) →
  m < -2 ∨ m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1094_109459


namespace NUMINAMATH_CALUDE_max_value_of_f_l1094_109443

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

theorem max_value_of_f (a : ℝ) :
  (∀ x : ℝ, f a x = f a (π / 3 - x)) →
  (∃ m : ℝ, ∀ x : ℝ, f a x ≤ m ∧ ∃ x₀ : ℝ, f a x₀ = m) →
  (∃ x₀ : ℝ, f a x₀ = 1) ∧ (∀ x : ℝ, f a x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1094_109443


namespace NUMINAMATH_CALUDE_epidemic_competition_theorem_l1094_109413

/-- Represents a participant in the competition -/
structure Participant where
  first_round_prob : ℚ
  second_round_prob : ℚ

/-- Calculates the probability of a participant winning both rounds -/
def win_prob (p : Participant) : ℚ :=
  p.first_round_prob * p.second_round_prob

/-- Calculates the probability of at least one participant winning -/
def at_least_one_wins (p1 p2 : Participant) : ℚ :=
  1 - (1 - win_prob p1) * (1 - win_prob p2)

theorem epidemic_competition_theorem 
  (A B : Participant)
  (h_A_first : A.first_round_prob = 5/6)
  (h_A_second : A.second_round_prob = 2/3)
  (h_B_first : B.first_round_prob = 3/5)
  (h_B_second : B.second_round_prob = 3/4) :
  win_prob A > win_prob B ∧ at_least_one_wins A B = 34/45 := by
  sorry

end NUMINAMATH_CALUDE_epidemic_competition_theorem_l1094_109413


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1094_109496

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first 5 terms of a sequence. -/
def SumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5

theorem arithmetic_sequence_third_term 
  (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SumFirstFive a = 20) : 
  a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1094_109496


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1094_109425

theorem polynomial_factorization (x y : ℝ) :
  x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8 =
  (x^2 - x*y + y^2) * (x^6 - x^3*y^3 + y^6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1094_109425


namespace NUMINAMATH_CALUDE_gas_fill_calculation_l1094_109406

theorem gas_fill_calculation (cost_today : ℝ) (rollback : ℝ) (friday_fill : ℝ) 
  (total_spend : ℝ) (total_liters : ℝ) 
  (h1 : cost_today = 1.4)
  (h2 : rollback = 0.4)
  (h3 : friday_fill = 25)
  (h4 : total_spend = 39)
  (h5 : total_liters = 35) :
  ∃ (today_fill : ℝ), 
    today_fill = 10 ∧ 
    cost_today * today_fill + (cost_today - rollback) * friday_fill = total_spend ∧
    today_fill + friday_fill = total_liters :=
by sorry

end NUMINAMATH_CALUDE_gas_fill_calculation_l1094_109406


namespace NUMINAMATH_CALUDE_prism_lateral_edge_length_l1094_109448

/-- A prism with 12 vertices and a sum of lateral edge lengths of 60 has lateral edges of length 10. -/
theorem prism_lateral_edge_length (num_vertices : ℕ) (sum_lateral_edges : ℝ) :
  num_vertices = 12 →
  sum_lateral_edges = 60 →
  ∃ (lateral_edge_length : ℝ), lateral_edge_length = 10 ∧
    lateral_edge_length * (num_vertices / 2) = sum_lateral_edges :=
by sorry


end NUMINAMATH_CALUDE_prism_lateral_edge_length_l1094_109448


namespace NUMINAMATH_CALUDE_exists_cost_price_l1094_109415

/-- The cost price of a watch satisfying the given conditions --/
def cost_price : ℝ → Prop := fun C =>
  3 * (0.925 * C + 265) = 3 * C * 1.053

/-- Theorem stating the existence of a cost price satisfying the conditions --/
theorem exists_cost_price : ∃ C : ℝ, cost_price C := by
  sorry

end NUMINAMATH_CALUDE_exists_cost_price_l1094_109415


namespace NUMINAMATH_CALUDE_count_non_negative_l1094_109431

def number_set : List ℚ := [-15, 16/3, -23/100, 0, 76/10, 2, -3/5, 314/100]

theorem count_non_negative : (number_set.filter (λ x => x ≥ 0)).length = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_non_negative_l1094_109431


namespace NUMINAMATH_CALUDE_turtle_combination_probability_l1094_109429

/- Define the number of initial turtles -/
def initial_turtles : ℕ := 2017

/- Define the number of combinations -/
def combinations : ℕ := 2015

/- Function to calculate binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/- Probability that a specific turtle is never chosen -/
def prob_never_chosen : ℚ := 1 / (binomial initial_turtles 2)

/- Theorem statement -/
theorem turtle_combination_probability :
  (initial_turtles : ℚ) * prob_never_chosen = 1 / 1008 :=
sorry

end NUMINAMATH_CALUDE_turtle_combination_probability_l1094_109429


namespace NUMINAMATH_CALUDE_purple_ball_count_l1094_109464

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the minimum number of tries to get one blue and one yellow ball -/
def minTries (counts : BallCounts) : ℕ :=
  counts.purple + (counts.blue - 1) + (counts.yellow - 1)

/-- The theorem stating the number of purple balls in the box -/
theorem purple_ball_count : ∃ (counts : BallCounts), 
  counts.blue = 5 ∧ 
  counts.yellow = 11 ∧ 
  minTries counts = 19 ∧ 
  counts.purple = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_ball_count_l1094_109464


namespace NUMINAMATH_CALUDE_min_value_theorem_l1094_109439

/-- A circle C with equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def CircleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- A line l with equation ax + by - 2 = 0 -/
def LineL (a b x y : ℝ) : Prop :=
  a*x + b*y - 2 = 0

/-- The theorem stating the minimum value of 1/a + 2/b -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_symmetry : ∃ (x y : ℝ), CircleC x y ∧ LineL a b x y) :
  (∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    (∃ (x y : ℝ), CircleC x y ∧ LineL a' b' x y) →
    1/a + 2/b ≤ 1/a' + 2/b') →
  1/a + 2/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1094_109439


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l1094_109405

/-- Given two cylinders X and Y with the following properties:
    1. The height of X equals the diameter of Y
    2. The diameter of X equals the height of Y (denoted as k)
    3. The volume of X is three times the volume of Y
    This theorem states that the volume of Y can be expressed as (1/4) π k^3 cubic units. -/
theorem cylinder_volume_relation (k : ℝ) (hk : k > 0) :
  ∃ (r_x h_x r_y : ℝ),
    r_x > 0 ∧ h_x > 0 ∧ r_y > 0 ∧
    h_x = 2 * r_y ∧
    2 * r_x = k ∧
    π * r_x^2 * h_x = 3 * (π * r_y^2 * k) ∧
    π * r_y^2 * k = (1/4) * π * k^3 :=
sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l1094_109405


namespace NUMINAMATH_CALUDE_jamie_ball_collection_l1094_109403

def total_balls (initial_red : ℕ) (blue_multiplier : ℕ) (lost_red : ℕ) (bought_yellow : ℕ) : ℕ :=
  (initial_red - lost_red) + (initial_red * blue_multiplier) + bought_yellow

theorem jamie_ball_collection : total_balls 16 2 6 32 = 74 := by
  sorry

end NUMINAMATH_CALUDE_jamie_ball_collection_l1094_109403


namespace NUMINAMATH_CALUDE_kevin_cards_l1094_109462

theorem kevin_cards (initial_cards lost_cards : ℝ) 
  (h1 : initial_cards = 47.0)
  (h2 : lost_cards = 7.0) :
  initial_cards - lost_cards = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l1094_109462


namespace NUMINAMATH_CALUDE_paint_production_max_profit_l1094_109424

/-- The paint production problem -/
theorem paint_production_max_profit :
  let material_A : ℝ := 120
  let material_B : ℝ := 90
  let total_production : ℝ := 150
  let type_A_material_A : ℝ := 0.6
  let type_A_material_B : ℝ := 0.7
  let type_A_profit : ℝ := 450
  let type_B_material_A : ℝ := 0.9
  let type_B_material_B : ℝ := 0.4
  let type_B_profit : ℝ := 500
  let profit (x : ℝ) := type_A_profit * x + type_B_profit * (total_production - x)
  ∀ x : ℝ, 
    (type_A_material_A * x + type_B_material_A * (total_production - x) ≤ material_A) →
    (type_A_material_B * x + type_B_material_B * (total_production - x) ≤ material_B) →
    profit x ≤ 72500 ∧ 
    (x = 50 → profit x = 72500) :=
by sorry

end NUMINAMATH_CALUDE_paint_production_max_profit_l1094_109424


namespace NUMINAMATH_CALUDE_trade_calculation_l1094_109451

/-- The number of matches per stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of stamps Tonya starts with -/
def tonya_initial_stamps : ℕ := 13

/-- The number of stamps Tonya ends with -/
def tonya_final_stamps : ℕ := 3

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

theorem trade_calculation :
  (tonya_initial_stamps - tonya_final_stamps) * matches_per_stamp = jimmy_matchbooks * matches_per_matchbook :=
by sorry

end NUMINAMATH_CALUDE_trade_calculation_l1094_109451


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1094_109421

/-- The surface area of a rectangular solid with edge lengths 2, 3, and 4 is 52 -/
theorem rectangular_solid_surface_area : 
  let a : ℝ := 4
  let b : ℝ := 3
  let h : ℝ := 2
  2 * (a * b + b * h + a * h) = 52 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1094_109421


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1094_109480

theorem solution_set_inequality (x : ℝ) :
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1094_109480


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l1094_109473

def is_valid_selection (s : Finset Nat) : Prop :=
  s.card = 6 ∧ s ⊆ Finset.range 8

def Q (s : Finset Nat) : Nat :=
  s.prod id

theorem largest_certain_divisor :
  ∀ (s : Finset Nat), is_valid_selection s →
  (2 ∣ Q s) ∧ 
  ∀ (n : Nat), n > 2 → (∃ (t : Finset Nat), is_valid_selection t ∧ ¬(n ∣ Q t)) :=
by sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l1094_109473


namespace NUMINAMATH_CALUDE_arrangements_with_constraint_l1094_109463

theorem arrangements_with_constraint (n : ℕ) (h : n ≥ 2) :
  (n - 1) * Nat.factorial (n - 1) = Nat.factorial n / 2 :=
by
  sorry

#check arrangements_with_constraint 5

end NUMINAMATH_CALUDE_arrangements_with_constraint_l1094_109463


namespace NUMINAMATH_CALUDE_oranges_packed_l1094_109465

/-- Given boxes that hold 10 oranges each and 265 boxes used, prove that the total number of oranges packed is 2650. -/
theorem oranges_packed (oranges_per_box : ℕ) (boxes_used : ℕ) (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) :
  oranges_per_box * boxes_used = 2650 := by
  sorry

end NUMINAMATH_CALUDE_oranges_packed_l1094_109465


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1094_109472

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Checks if a point satisfies the equation of a line -/
def on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The original line 2x - 3y + 2 = 0 -/
def original_line : Line :=
  { a := 2, b := -3, c := 2 }

/-- The symmetric line to be proven -/
def symmetric_line : Line :=
  { a := 2, b := 3, c := 2 }

theorem symmetric_line_correct :
  ∀ p : ℝ × ℝ, on_line symmetric_line p ↔ on_line original_line (reflect_x p) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1094_109472


namespace NUMINAMATH_CALUDE_remainder_polynomial_l1094_109449

-- Define the polynomials
variable (z : ℂ)
variable (Q : ℂ → ℂ)
variable (R : ℂ → ℂ)

-- State the theorem
theorem remainder_polynomial :
  (∀ z, z^2023 + 1 = (z^2 - z + 1) * Q z + R z) →
  (∃ a b : ℂ, ∀ z, R z = a * z + b) →
  (∀ z, R z = z + 1) :=
by sorry

end NUMINAMATH_CALUDE_remainder_polynomial_l1094_109449


namespace NUMINAMATH_CALUDE_a_formula_l1094_109402

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 5
  | n + 1 => ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem a_formula (n : ℕ) : a n = 4 * n + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_a_formula_l1094_109402


namespace NUMINAMATH_CALUDE_saree_price_l1094_109442

theorem saree_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  final_price = 144 ∧ discount1 = 0.20 ∧ discount2 = 0.10 →
  ∃ original_price : ℝ, 
    original_price = 200 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
by
  sorry

end NUMINAMATH_CALUDE_saree_price_l1094_109442


namespace NUMINAMATH_CALUDE_square_fence_perimeter_16_posts_l1094_109412

/-- Calculates the outer perimeter of a square fence given the number of posts,
    post width, and gap between posts. -/
def squareFencePerimeter (numPosts : ℕ) (postWidth : ℚ) (gapWidth : ℚ) : ℚ :=
  let postsPerSide : ℕ := numPosts / 4
  let gapsPerSide : ℕ := postsPerSide - 1
  let sideLength : ℚ := (gapsPerSide : ℚ) * gapWidth + (postsPerSide : ℚ) * postWidth
  4 * sideLength

/-- The outer perimeter of a square fence with 16 posts, each 6 inches wide,
    and 4 feet between posts, is 56 feet. -/
theorem square_fence_perimeter_16_posts :
  squareFencePerimeter 16 (1/2) 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_16_posts_l1094_109412


namespace NUMINAMATH_CALUDE_range_x_and_a_l1094_109435

def P (x a : ℝ) : Prop := -x^2 + 4*a*x - 3*a^2 > 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_x_and_a (a : ℝ) (h : a > 0) :
  (∀ x, P x 1 ∧ q x → x > 2 ∧ x < 3) ∧
  (∀ a, (∀ x, 2 < x ∧ x < 3 → a < x ∧ x < 3*a) ↔ 1 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_range_x_and_a_l1094_109435


namespace NUMINAMATH_CALUDE_inequality_always_holds_l1094_109436

theorem inequality_always_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l1094_109436


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l1094_109468

theorem zoo_ticket_price (total_people : ℕ) (num_children : ℕ) (child_price : ℕ) (total_bill : ℕ) :
  total_people = 201 →
  num_children = 161 →
  child_price = 4 →
  total_bill = 964 →
  (total_people - num_children) * 8 + num_children * child_price = total_bill :=
by sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l1094_109468


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1094_109422

/-- Given a quadratic function f(x) = ax^2 + bx + c with specific conditions,
    prove properties about its coefficients, roots, and values. -/
theorem quadratic_function_properties
  (a b c : ℝ) (m₁ m₂ : ℝ)
  (h_order : a > b ∧ b > c)
  (h_points : a^2 + (a * m₁^2 + b * m₁ + c + a * m₂^2 + b * m₂ + c) * a +
              (a * m₁^2 + b * m₁ + c) * (a * m₂^2 + b * m₂ + c) = 0)
  (h_root : a + b + c = 0) :
  (b ≥ 0) ∧
  (2 ≤ |1 - c/a| ∧ |1 - c/a| < 3) ∧
  (max (a * (m₁ + 3)^2 + b * (m₁ + 3) + c) (a * (m₂ + 3)^2 + b * (m₂ + 3) + c) > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1094_109422


namespace NUMINAMATH_CALUDE_no_perfect_cube_in_range_l1094_109457

theorem no_perfect_cube_in_range : 
  ¬ ∃ (n : ℤ), 4 ≤ n ∧ n ≤ 11 ∧ ∃ (k : ℤ), n^2 + 3*n + 2 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cube_in_range_l1094_109457


namespace NUMINAMATH_CALUDE_max_value_product_l1094_109493

theorem max_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^4 * y^2 * z ≤ 1024 / 7^7 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 1 ∧ x₀^4 * y₀^2 * z₀ = 1024 / 7^7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_l1094_109493


namespace NUMINAMATH_CALUDE_card_distribution_l1094_109417

theorem card_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  ∃ (people_with_fewer : ℕ), people_with_fewer = 3 ∧ 
  people_with_fewer = num_people - (total_cards % num_people) :=
by
  sorry

end NUMINAMATH_CALUDE_card_distribution_l1094_109417


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1094_109492

theorem cube_sum_reciprocal (x : ℝ) (h : 35 = x^6 + 1/x^6) : x^3 + 1/x^3 = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1094_109492


namespace NUMINAMATH_CALUDE_pie_chart_probability_l1094_109410

theorem pie_chart_probability (prob_D prob_E prob_F : ℚ) : 
  prob_D = 1/4 → prob_E = 1/3 → prob_D + prob_E + prob_F = 1 → prob_F = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_probability_l1094_109410


namespace NUMINAMATH_CALUDE_shadow_length_theorem_l1094_109481

theorem shadow_length_theorem (α β : Real) (h : Real) 
  (shadow_length : Real → Real → Real)
  (h_shadow : ∀ θ, shadow_length h θ = h * Real.tan θ)
  (h_first_measurement : Real.tan α = 3)
  (h_angle_diff : Real.tan (α - β) = 1/3) :
  Real.tan β = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_theorem_l1094_109481


namespace NUMINAMATH_CALUDE_least_value_expression_l1094_109471

theorem least_value_expression (x : ℝ) (h : x < -2) :
  (2 * x ≤ x) ∧ (2 * x ≤ x + 2) ∧ (2 * x ≤ (1/2) * x) ∧ (2 * x ≤ x - 2) := by
  sorry

end NUMINAMATH_CALUDE_least_value_expression_l1094_109471


namespace NUMINAMATH_CALUDE_shirts_not_washed_l1094_109498

theorem shirts_not_washed 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 40)
  (h2 : long_sleeve = 23)
  (h3 : washed = 29) :
  short_sleeve + long_sleeve - washed = 34 := by
sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l1094_109498


namespace NUMINAMATH_CALUDE_cosine_translation_symmetry_l1094_109419

theorem cosine_translation_symmetry (x : ℝ) (k : ℤ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * (x + π / 12))
  let axis : ℝ := k * π / 2 - π / 12
  (∀ t, f (axis + t) = f (axis - t)) := by sorry

end NUMINAMATH_CALUDE_cosine_translation_symmetry_l1094_109419


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_not_axiom_l1094_109479

-- Define the type for geometric propositions
inductive GeometricProposition
  | PlanesParallelToSamePlaneAreParallel
  | ThreePointsDetermineUniquePlane
  | LineInPlaneImpliesAllPointsInPlane
  | TwoPlanesWithCommonPointHaveCommonLine

-- Define the set of axioms
def geometryAxioms : Set GeometricProposition :=
  { GeometricProposition.ThreePointsDetermineUniquePlane,
    GeometricProposition.LineInPlaneImpliesAllPointsInPlane,
    GeometricProposition.TwoPlanesWithCommonPointHaveCommonLine }

-- Theorem statement
theorem planes_parallel_to_same_plane_not_axiom :
  GeometricProposition.PlanesParallelToSamePlaneAreParallel ∉ geometryAxioms :=
by sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_not_axiom_l1094_109479


namespace NUMINAMATH_CALUDE_money_fraction_after_two_years_l1094_109467

/-- The simple interest rate per annum as a decimal -/
def interest_rate : ℝ := 0.08333333333333337

/-- The time period in years -/
def time_period : ℝ := 2

/-- The fraction of the sum of money after the given time period -/
def money_fraction : ℝ := 1 + interest_rate * time_period

theorem money_fraction_after_two_years :
  money_fraction = 1.1666666666666667 := by sorry

end NUMINAMATH_CALUDE_money_fraction_after_two_years_l1094_109467


namespace NUMINAMATH_CALUDE_total_books_l1094_109438

/-- The number of books each person has -/
structure Books where
  beatrix : ℕ
  alannah : ℕ
  queen : ℕ

/-- The conditions of the problem -/
def book_conditions (b : Books) : Prop :=
  b.beatrix = 30 ∧
  b.alannah = b.beatrix + 20 ∧
  b.queen = b.alannah + (b.alannah / 5)

/-- The theorem to prove -/
theorem total_books (b : Books) :
  book_conditions b → b.beatrix + b.alannah + b.queen = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1094_109438


namespace NUMINAMATH_CALUDE_red_balls_count_l1094_109489

theorem red_balls_count (total : ℕ) (white : ℕ) (red : ℕ) (prob_white : ℚ) : 
  white = 5 →
  total = white + red →
  prob_white = 1/4 →
  (white : ℚ) / total = prob_white →
  red = 15 := by
sorry

end NUMINAMATH_CALUDE_red_balls_count_l1094_109489


namespace NUMINAMATH_CALUDE_inequality_proof_l1094_109430

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hu₁ : x₁ * y₁ - z₁^2 > 0) (hu₂ : x₂ * y₂ - z₂^2 > 0) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1094_109430


namespace NUMINAMATH_CALUDE_total_soccer_balls_l1094_109446

/-- Represents a school with elementary and middle school classes --/
structure School where
  elementary_classes : ℕ
  middle_classes : ℕ
  elementary_students : List ℕ
  middle_students : List ℕ

/-- Calculates the number of soccer balls for a given number of students in an elementary class --/
def elementary_balls (students : ℕ) : ℕ :=
  if students ≤ 30 then 4 else 5

/-- Calculates the number of soccer balls for a given number of students in a middle school class --/
def middle_balls (students : ℕ) : ℕ :=
  if students ≤ 24 then 6 else 7

/-- Calculates the total number of soccer balls for a school --/
def school_balls (school : School) : ℕ :=
  (school.elementary_students.map elementary_balls).sum +
  (school.middle_students.map middle_balls).sum

/-- The three schools as described in the problem --/
def school_A : School :=
  { elementary_classes := 4
  , middle_classes := 5
  , elementary_students := List.replicate 4 28
  , middle_students := List.replicate 5 25 }

def school_B : School :=
  { elementary_classes := 5
  , middle_classes := 3
  , elementary_students := [32, 32, 32, 30, 30]
  , middle_students := [22, 22, 26] }

def school_C : School :=
  { elementary_classes := 6
  , middle_classes := 4
  , elementary_students := [30, 30, 30, 30, 31, 31]
  , middle_students := List.replicate 4 24 }

/-- The main theorem stating that the total number of soccer balls donated is 143 --/
theorem total_soccer_balls :
  school_balls school_A + school_balls school_B + school_balls school_C = 143 := by
  sorry


end NUMINAMATH_CALUDE_total_soccer_balls_l1094_109446


namespace NUMINAMATH_CALUDE_min_unique_integers_l1094_109485

theorem min_unique_integers (L : List ℕ) (h : L = [1, 2, 3, 4, 5, 6, 7, 8, 9]) : 
  ∃ (f : ℕ → ℕ), 
    (∀ n, f n = n + 2 ∨ f n = n + 5) ∧ 
    (Finset.card (Finset.image f L.toFinset) = 6) ∧
    (∀ g : ℕ → ℕ, (∀ n, g n = n + 2 ∨ g n = n + 5) → 
      Finset.card (Finset.image g L.toFinset) ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_min_unique_integers_l1094_109485


namespace NUMINAMATH_CALUDE_construct_segment_a_construct_segment_b_l1094_109490

-- Part a
theorem construct_segment_a (a : ℝ) (h : a = Real.sqrt 5) : ∃ b : ℝ, b = 1 := by
  sorry

-- Part b
theorem construct_segment_b (a : ℝ) (h : a = 7) : ∃ b : ℝ, b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_construct_segment_a_construct_segment_b_l1094_109490


namespace NUMINAMATH_CALUDE_investment_ratio_proof_l1094_109497

/-- Represents the investment and return for an investor -/
structure Investor where
  investment : ℝ
  returnRate : ℝ

/-- Proves that the ratio of investments is 6:5:4 given the problem conditions -/
theorem investment_ratio_proof 
  (a b c : Investor)
  (return_ratio : a.returnRate / b.returnRate = 6/5 ∧ b.returnRate / c.returnRate = 5/4)
  (b_earns_more : b.investment * b.returnRate = a.investment * a.returnRate + 100)
  (total_earnings : a.investment * a.returnRate + b.investment * b.returnRate + c.investment * c.returnRate = 2900)
  : a.investment / b.investment = 6/5 ∧ b.investment / c.investment = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_proof_l1094_109497


namespace NUMINAMATH_CALUDE_data_mode_and_median_l1094_109476

def data : List ℕ := [6, 3, 5, 4, 3, 3]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem data_mode_and_median : 
  mode data = 3 ∧ median data = 3.5 := by sorry

end NUMINAMATH_CALUDE_data_mode_and_median_l1094_109476


namespace NUMINAMATH_CALUDE_coconut_farm_earnings_l1094_109407

def farm_size : ℕ := 20
def trees_per_square_meter : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_frequency : ℕ := 3  -- in months
def price_per_coconut : ℚ := 1/2  -- $0.50 as a rational number
def time_period : ℕ := 6  -- in months

def total_trees : ℕ := farm_size * trees_per_square_meter
def total_coconuts_per_harvest : ℕ := total_trees * coconuts_per_tree
def number_of_harvests : ℕ := time_period / harvest_frequency
def total_coconuts : ℕ := total_coconuts_per_harvest * number_of_harvests

theorem coconut_farm_earnings : 
  (total_coconuts : ℚ) * price_per_coconut = 240 := by
  sorry

end NUMINAMATH_CALUDE_coconut_farm_earnings_l1094_109407


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1094_109409

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x - 2 * y = 4) ∧ (3 * x + 4 * y = 16) ↔ x = 24/13 ∧ y = 34/13 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1094_109409


namespace NUMINAMATH_CALUDE_equation_solutions_l1094_109445

theorem equation_solutions : 
  ∀ x : ℝ, (x - 2)^2 = 9*x^2 ↔ x = -1 ∨ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1094_109445


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l1094_109478

theorem bingo_prize_distribution (total_prize : ℕ) (first_winner_fraction : ℚ) 
  (remaining_winners : ℕ) (remaining_fraction : ℚ) :
  total_prize = 2400 →
  first_winner_fraction = 1 / 3 →
  remaining_winners = 10 →
  remaining_fraction = 1 / 10 →
  (total_prize - (first_winner_fraction * total_prize).num) / remaining_winners = 160 := by
  sorry

end NUMINAMATH_CALUDE_bingo_prize_distribution_l1094_109478


namespace NUMINAMATH_CALUDE_train_length_l1094_109450

/-- The length of a train given its speed, platform length, and time to cross the platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) →
  platform_length = 240 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 280 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1094_109450


namespace NUMINAMATH_CALUDE_video_library_disk_space_l1094_109452

/-- Calculates the average disk space per hour of video in a library, rounded to the nearest integer -/
def averageDiskSpacePerHour (totalDays : ℕ) (totalSpace : ℕ) : ℕ :=
  let totalHours : ℕ := totalDays * 24
  let exactAverage : ℚ := totalSpace / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Theorem stating that for a 15-day video library occupying 24000 MB, 
    the average disk space per hour rounded to the nearest integer is 67 MB -/
theorem video_library_disk_space :
  averageDiskSpacePerHour 15 24000 = 67 := by
  sorry

end NUMINAMATH_CALUDE_video_library_disk_space_l1094_109452


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1094_109454

theorem complete_square_quadratic (x : ℝ) : 
  (∃ c d : ℝ, x^2 + 6*x - 5 = 0 ↔ (x + c)^2 = d) → 
  (∃ c : ℝ, (x + c)^2 = 14) := by
sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1094_109454


namespace NUMINAMATH_CALUDE_vacation_cost_division_l1094_109432

theorem vacation_cost_division (total_cost : ℝ) (initial_people : ℕ) (cost_reduction : ℝ) (n : ℕ) : 
  total_cost = 375 ∧ 
  initial_people = 3 ∧ 
  cost_reduction = 50 ∧ 
  (total_cost / initial_people) - (total_cost / n) = cost_reduction →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l1094_109432


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1094_109475

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n^4 - 4*n^3 + 22*n^2 - 36*n + 18 = m^2) ↔ (n = 1 ∨ n = 3) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1094_109475


namespace NUMINAMATH_CALUDE_necklace_cost_l1094_109420

/-- The cost of Scarlet's necklace given her savings and expenses -/
theorem necklace_cost (savings : ℕ) (earrings_cost : ℕ) (remaining : ℕ) : 
  savings = 80 → earrings_cost = 23 → remaining = 9 → 
  savings - earrings_cost - remaining = 48 := by
  sorry

end NUMINAMATH_CALUDE_necklace_cost_l1094_109420


namespace NUMINAMATH_CALUDE_line_parameterization_values_l1094_109488

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 3 * x + 2

/-- The parameterization of the line -/
def parameterization (t s m : ℝ) : ℝ × ℝ :=
  (-4 + t * m, s + t * (-7))

/-- Theorem stating the values of s and m for the given line and parameterization -/
theorem line_parameterization_values :
  ∃ (s m : ℝ), 
    (∀ t, line_equation (parameterization t s m).1 (parameterization t s m).2) ∧
    s = -10 ∧ 
    m = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_values_l1094_109488


namespace NUMINAMATH_CALUDE_biancas_album_pictures_l1094_109447

/-- Given that Bianca uploaded 33 pictures and put some into 3 albums with 2 pictures each,
    prove that she put 27 pictures into the first album. -/
theorem biancas_album_pictures :
  ∀ (total_pictures : ℕ) (other_albums : ℕ) (pics_per_album : ℕ),
    total_pictures = 33 →
    other_albums = 3 →
    pics_per_album = 2 →
    total_pictures - (other_albums * pics_per_album) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_biancas_album_pictures_l1094_109447


namespace NUMINAMATH_CALUDE_cube_volume_after_increase_l1094_109414

theorem cube_volume_after_increase (surface_area : ℝ) (increase_factor : ℝ) : 
  surface_area = 864 → increase_factor = 1.5 → 
  (increase_factor * (surface_area / 6).sqrt) ^ 3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_after_increase_l1094_109414


namespace NUMINAMATH_CALUDE_tile_1x1_position_l1094_109416

/-- Represents a position in the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents a 1x3 tile -/
structure Tile1x3 where
  start : Position
  horizontal : Bool

/-- Represents the placement of tiles in the 7x7 grid -/
structure TilePlacement where
  tiles1x3 : Finset Tile1x3
  tile1x1 : Position

/-- Predicate to check if a position is in the center or adjacent to the edges -/
def isCenterOrEdgeAdjacent (p : Position) : Prop :=
  (p.row = 3 ∧ p.col = 3) ∨ 
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6 ∨
  p.row = 1 ∨ p.row = 5 ∨ p.col = 1 ∨ p.col = 5

/-- Main theorem: The 1x1 tile must be in the center or adjacent to the edges -/
theorem tile_1x1_position (placement : TilePlacement) 
  (h1 : placement.tiles1x3.card = 16) 
  (h2 : ∀ t ∈ placement.tiles1x3, t.start.row < 7 ∧ t.start.col < 7) 
  (h3 : ∀ t ∈ placement.tiles1x3, 
    if t.horizontal 
    then t.start.col < 5 
    else t.start.row < 5) :
  isCenterOrEdgeAdjacent placement.tile1x1 :=
sorry

end NUMINAMATH_CALUDE_tile_1x1_position_l1094_109416


namespace NUMINAMATH_CALUDE_total_harvest_is_2000_l1094_109487

/-- Represents the harvest of tomatoes over three days -/
structure TomatoHarvest where
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total harvest over three days -/
def total_harvest (h : TomatoHarvest) : ℕ :=
  h.wednesday + h.thursday + h.friday

/-- Theorem stating the total harvest is 2000 kg given the conditions -/
theorem total_harvest_is_2000 (h : TomatoHarvest) 
  (h_wed : h.wednesday = 400)
  (h_thu : h.thursday = h.wednesday / 2)
  (h_fri : h.friday - 700 = 700) :
  total_harvest h = 2000 := by
  sorry

#check total_harvest_is_2000

end NUMINAMATH_CALUDE_total_harvest_is_2000_l1094_109487


namespace NUMINAMATH_CALUDE_olives_per_jar_l1094_109499

/-- Proves that the number of olives in a jar is 20 given the problem conditions --/
theorem olives_per_jar (
  total_money : ℝ)
  (olives_needed : ℕ)
  (jar_cost : ℝ)
  (change : ℝ)
  (h1 : total_money = 10)
  (h2 : olives_needed = 80)
  (h3 : jar_cost = 1.5)
  (h4 : change = 4)
  : (olives_needed : ℝ) / ((total_money - change) / jar_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_olives_per_jar_l1094_109499


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l1094_109404

/-- The area of triangle ABC given the total area of small triangles and subtracted areas. -/
theorem area_of_triangle_ABC (total_area : ℝ) (subtracted_area : ℝ) 
  (h1 : total_area = 24)
  (h2 : subtracted_area = 14) :
  total_area - subtracted_area = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l1094_109404


namespace NUMINAMATH_CALUDE_g_of_four_value_l1094_109482

/-- A function from positive reals to positive reals satisfying certain conditions -/
def G := {g : ℝ → ℝ // ∀ x > 0, g x > 0 ∧ g 1 = 1 ∧ g (x^2 * g x) = x * g (x^2) + g x}

theorem g_of_four_value (g : G) : g.val 4 = 36/23 := by
  sorry

end NUMINAMATH_CALUDE_g_of_four_value_l1094_109482


namespace NUMINAMATH_CALUDE_probability_three_common_books_l1094_109491

theorem probability_three_common_books (total_books : ℕ) (books_to_select : ℕ) (common_books : ℕ) :
  total_books = 12 →
  books_to_select = 7 →
  common_books = 3 →
  (Nat.choose total_books common_books * Nat.choose (total_books - common_books) (books_to_select - common_books) * Nat.choose (total_books - common_books) (books_to_select - common_books)) /
  (Nat.choose total_books books_to_select * Nat.choose total_books books_to_select) =
  3502800 / 627264 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_common_books_l1094_109491


namespace NUMINAMATH_CALUDE_angle_between_sides_l1094_109486

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] :=
  (a b c d : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)

-- Define the theorem
theorem angle_between_sides (q : CyclicQuadrilateral ℝ) :
  Real.arccos ((q.a^2 + q.b^2 - q.d^2 - q.c^2) / (2 * (q.a * q.b + q.d * q.c))) =
  Real.arccos ((q.a^2 + q.b^2 - q.d^2 - q.c^2) / (2 * (q.a * q.b + q.d * q.c))) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_sides_l1094_109486


namespace NUMINAMATH_CALUDE_height_difference_after_three_years_l1094_109466

/-- Represents the seasons of the year -/
inductive Season
  | spring
  | summer
  | fall
  | winter

/-- Calculates the growth of an object over a season given its monthly growth rate -/
def seasonalGrowth (monthlyRate : ℕ) : ℕ := 3 * monthlyRate

/-- Calculates the total growth over a year given seasonal growth rates -/
def yearlyGrowth (spring summer fall winter : ℕ) : ℕ :=
  seasonalGrowth spring + seasonalGrowth summer + seasonalGrowth fall + seasonalGrowth winter

/-- Theorem: The height difference between the tree and the boy after 3 years is 73 inches -/
theorem height_difference_after_three_years :
  let initialTreeHeight : ℕ := 16
  let initialBoyHeight : ℕ := 24
  let treeGrowth : Season → ℕ
    | Season.spring => 4
    | Season.summer => 6
    | Season.fall => 2
    | Season.winter => 1
  let boyGrowth : Season → ℕ
    | Season.spring => 2
    | Season.summer => 2
    | Season.fall => 0
    | Season.winter => 0
  let treeYearlyGrowth := yearlyGrowth (treeGrowth Season.spring) (treeGrowth Season.summer) (treeGrowth Season.fall) (treeGrowth Season.winter)
  let boyYearlyGrowth := yearlyGrowth (boyGrowth Season.spring) (boyGrowth Season.summer) (boyGrowth Season.fall) (boyGrowth Season.winter)
  let finalTreeHeight := initialTreeHeight + 3 * treeYearlyGrowth
  let finalBoyHeight := initialBoyHeight + 3 * boyYearlyGrowth
  finalTreeHeight - finalBoyHeight = 73 := by
  sorry


end NUMINAMATH_CALUDE_height_difference_after_three_years_l1094_109466


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1094_109483

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1094_109483


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l1094_109423

/-- Proves that given a boat with speed 36 kmph in still water and a stream with speed 12 kmph,
    if the boat covers 80 km downstream in the same time as it covers a certain distance upstream,
    then that upstream distance is 40 km. -/
theorem boat_upstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (h1 : boat_speed = 36)
  (h2 : stream_speed = 12)
  (h3 : downstream_distance = 80)
  (h4 : downstream_distance / (boat_speed + stream_speed) =
        upstream_distance / (boat_speed - stream_speed)) :
  upstream_distance = 40 :=
sorry

end NUMINAMATH_CALUDE_boat_upstream_distance_l1094_109423


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l1094_109426

theorem polynomial_coefficients 
  (a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (x - 1)^3 + (x + 1)^4 = x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) : 
  a₁ = 5 ∧ a₂ + a₃ + a₄ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l1094_109426


namespace NUMINAMATH_CALUDE_product_equals_143_l1094_109456

/-- Convert a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Convert a ternary number (represented as a list of digits) to its decimal equivalent -/
def ternary_to_decimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The binary representation of 1101₂ -/
def binary_num : List Bool := [true, false, true, true]

/-- The ternary representation of 102₃ -/
def ternary_num : List ℕ := [2, 0, 1]

theorem product_equals_143 : 
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_143_l1094_109456


namespace NUMINAMATH_CALUDE_count_odd_increasing_integers_l1094_109427

/-- A three-digit integer with odd digits in strictly increasing order -/
structure OddIncreasingInteger where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_odd_hundreds : Odd hundreds
  is_odd_tens : Odd tens
  is_odd_ones : Odd ones
  is_increasing : hundreds < tens ∧ tens < ones
  is_three_digit : 100 ≤ hundreds * 100 + tens * 10 + ones ∧ hundreds * 100 + tens * 10 + ones < 1000

/-- The count of three-digit integers with odd digits in strictly increasing order -/
def countOddIncreasingIntegers : Nat := sorry

/-- Theorem stating that there are exactly 10 three-digit integers with odd digits in strictly increasing order -/
theorem count_odd_increasing_integers :
  countOddIncreasingIntegers = 10 := by sorry

end NUMINAMATH_CALUDE_count_odd_increasing_integers_l1094_109427


namespace NUMINAMATH_CALUDE_matrix_power_vector_product_l1094_109474

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 4]
def a : Matrix (Fin 2) (Fin 1) ℝ := !![7; 4]

theorem matrix_power_vector_product :
  A^6 * a = !![435; 339] := by sorry

end NUMINAMATH_CALUDE_matrix_power_vector_product_l1094_109474
