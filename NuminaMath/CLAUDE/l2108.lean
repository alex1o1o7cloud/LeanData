import Mathlib

namespace NUMINAMATH_CALUDE_joan_payment_l2108_210822

/-- The amount Joan paid for her purchases, given the costs and change received -/
def amount_paid (cat_toy_cost cage_cost change : ℚ) : ℚ :=
  cat_toy_cost + cage_cost - change

/-- Theorem stating that Joan paid $19.48 for her purchases -/
theorem joan_payment : amount_paid 8.77 10.97 0.26 = 19.48 := by
  sorry

end NUMINAMATH_CALUDE_joan_payment_l2108_210822


namespace NUMINAMATH_CALUDE_work_comparison_l2108_210842

/-- Represents the amount of work that can be done by a group of people in a given time -/
structure WorkCapacity where
  people : ℕ
  work : ℝ
  days : ℕ

/-- Given two work capacities, proves that the first group did twice the initially considered work -/
theorem work_comparison (w1 w2 : WorkCapacity) : 
  w1.people = 3 ∧ 
  w1.days = 3 ∧ 
  w2.people = 6 ∧ 
  w2.days = 3 ∧ 
  w2.work = 6 * w1.work → 
  w1.work = 2 * w1.work := by
sorry

end NUMINAMATH_CALUDE_work_comparison_l2108_210842


namespace NUMINAMATH_CALUDE_partnership_profit_l2108_210894

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ  -- A's investment
  b_investment : ℕ  -- B's investment
  a_period : ℕ      -- A's investment period
  b_period : ℕ      -- B's investment period
  b_profit : ℕ      -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℕ :=
  let ratio := (p.a_investment * p.a_period) / (p.b_investment * p.b_period)
  p.b_profit * (ratio + 1)

/-- Theorem stating the total profit for the given partnership conditions --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_investment = 3 * p.b_investment) 
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.b_profit = 3000) : 
  total_profit p = 21000 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 3000 }

end NUMINAMATH_CALUDE_partnership_profit_l2108_210894


namespace NUMINAMATH_CALUDE_projected_revenue_increase_l2108_210891

theorem projected_revenue_increase 
  (last_year_revenue : ℝ) 
  (h1 : actual_revenue = 0.75 * last_year_revenue) 
  (h2 : actual_revenue = 0.60 * projected_revenue) 
  (projected_revenue := last_year_revenue * (1 + projected_increase / 100)) :
  projected_increase = 25 := by
sorry

end NUMINAMATH_CALUDE_projected_revenue_increase_l2108_210891


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2108_210868

theorem polynomial_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 12) + 2 = -x^4 + 3*x^3 - 6*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2108_210868


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2108_210839

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a^2 + b^2 = c^2) →  -- right-angled triangle condition
  (a^2 + b^2 + c^2 = 4500) →  -- sum of squares condition
  (a = 3*b) →  -- one leg is three times the other
  c = 15 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2108_210839


namespace NUMINAMATH_CALUDE_special_polyhedron_hexagon_count_l2108_210819

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  -- V: vertices, E: edges, F: faces, P: pentagonal faces, H: hexagonal faces
  V : ℕ
  E : ℕ
  F : ℕ
  P : ℕ
  H : ℕ
  vertex_degree : V * 3 = E * 2
  face_types : F = P + H
  euler : V - E + F = 2
  edge_count : E * 2 = P * 5 + H * 6
  both_face_types : P > 0 ∧ H > 0

/-- Theorem: In a SpecialPolyhedron, the number of hexagonal faces is at least 2 -/
theorem special_polyhedron_hexagon_count (poly : SpecialPolyhedron) : poly.H ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_hexagon_count_l2108_210819


namespace NUMINAMATH_CALUDE_profit_achieved_l2108_210863

/-- The minimum number of disks Maria needs to sell to make a profit of $120 -/
def disks_to_sell : ℕ := 219

/-- The cost of buying 5 disks -/
def buy_price : ℚ := 6

/-- The selling price of 4 disks -/
def sell_price : ℚ := 7

/-- The desired profit -/
def target_profit : ℚ := 120

theorem profit_achieved :
  let cost_per_disk : ℚ := buy_price / 5
  let revenue_per_disk : ℚ := sell_price / 4
  let profit_per_disk : ℚ := revenue_per_disk - cost_per_disk
  (disks_to_sell : ℚ) * profit_per_disk ≥ target_profit ∧
  ∀ n : ℕ, (n : ℚ) * profit_per_disk < target_profit → n < disks_to_sell :=
by sorry

end NUMINAMATH_CALUDE_profit_achieved_l2108_210863


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2108_210808

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x| ≤ 2
def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ 
  (∃ x : ℝ, p x ∧ ¬(q x)) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2108_210808


namespace NUMINAMATH_CALUDE_train_crossing_time_l2108_210872

/-- Proves that a train with given length and speed takes a specific time to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 320 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 8 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2108_210872


namespace NUMINAMATH_CALUDE_expression_equals_28_l2108_210803

theorem expression_equals_28 : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_28_l2108_210803


namespace NUMINAMATH_CALUDE_set_operations_l2108_210802

def U : Set ℕ := {x | x ≤ 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem set_operations :
  (A ∩ B = {2, 4}) ∧
  (U \ (A ∪ B) = {0, 3, 7}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2108_210802


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2108_210895

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - x^4 + x^3 - 9 = (x - 1) * (x^4 - x^3 + x^2 - x + 1) + (-9) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2108_210895


namespace NUMINAMATH_CALUDE_power_division_equals_729_l2108_210856

theorem power_division_equals_729 : 3^12 / 27^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l2108_210856


namespace NUMINAMATH_CALUDE_triangle_property_l2108_210821

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b² + c² - a²) / cos A = 2 and (a cos B - b cos A) / (a cos B + b cos A) - b/c = 1,
    then bc = 1 and the area of triangle ABC is √3/4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  ((b^2 + c^2 - a^2) / Real.cos A = 2) →
  ((a * Real.cos B - b * Real.cos A) / (a * Real.cos B + b * Real.cos A) - b / c = 1) →
  (b * c = 1 ∧ (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2108_210821


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l2108_210886

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- number of days
  let k : ℕ := 5  -- number of successful days
  let p : ℚ := 3/4  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l2108_210886


namespace NUMINAMATH_CALUDE_loop_contains_conditional_l2108_210862

/-- Represents a flowchart structure -/
inductive FlowchartStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents the containment relationship between flowchart structures -/
def contains : FlowchartStructure → FlowchartStructure → Prop := sorry

/-- A loop structure must contain a conditional structure -/
theorem loop_contains_conditional :
  ∀ (loop : FlowchartStructure), loop = FlowchartStructure.Loop →
    ∃ (cond : FlowchartStructure), cond = FlowchartStructure.Conditional ∧ contains loop cond :=
  sorry

end NUMINAMATH_CALUDE_loop_contains_conditional_l2108_210862


namespace NUMINAMATH_CALUDE_promotion_difference_l2108_210883

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A  -- Buy one pair, get second pair half price
  | B  -- Buy one pair, get $15 off second pair

/-- Calculates the total cost of two pairs of shoes under a given promotion -/
def calculateCost (p : Promotion) (price1 : ℕ) (price2 : ℕ) : ℕ :=
  match p with
  | Promotion.A => price1 + price2 / 2
  | Promotion.B => price1 + price2 - 15

/-- Theorem stating the difference in cost between Promotion B and A -/
theorem promotion_difference :
  ∀ (price1 price2 : ℕ),
  price1 = 50 →
  price2 = 40 →
  calculateCost Promotion.B price1 price2 - calculateCost Promotion.A price1 price2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_promotion_difference_l2108_210883


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2108_210854

/-- The lateral surface area of a cone with base radius 3 and lateral surface that unfolds into a semicircle is 18π. -/
theorem cone_lateral_surface_area (r : ℝ) (l : ℝ) : 
  r = 3 →
  π * l = 2 * π * r →
  π * r * l = 18 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2108_210854


namespace NUMINAMATH_CALUDE_mean_home_runs_l2108_210833

def number_of_players : ℕ := 3 + 5 + 3 + 1

def total_home_runs : ℕ := 5 * 3 + 8 * 5 + 9 * 3 + 11 * 1

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (number_of_players : ℚ) = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l2108_210833


namespace NUMINAMATH_CALUDE_value_of_a_l2108_210897

/-- Converts paise to rupees -/
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

/-- The problem statement -/
theorem value_of_a (a : ℚ) (h : (0.5 / 100) * a = 75) : 
  paise_to_rupees a = 150 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l2108_210897


namespace NUMINAMATH_CALUDE_final_water_level_change_l2108_210820

def water_level_change (initial_change : ℝ) (subsequent_change : ℝ) : ℝ :=
  initial_change + subsequent_change

theorem final_water_level_change :
  water_level_change (-3) 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_water_level_change_l2108_210820


namespace NUMINAMATH_CALUDE_even_operations_l2108_210809

theorem even_operations (n : ℤ) (h : Even n) :
  Even (5 * n) ∧ Even (n ^ 2) ∧ Even (n ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_even_operations_l2108_210809


namespace NUMINAMATH_CALUDE_exist_a_b_l2108_210815

theorem exist_a_b : ∃ (a b : ℝ),
  (a < 0) ∧
  (b = -a) ∧
  (b > 9/4) ∧
  (∀ x : ℝ, x < -1 → a * x > b) ∧
  (∀ y : ℝ, y^2 + 3*y + b > 0) := by
  sorry

end NUMINAMATH_CALUDE_exist_a_b_l2108_210815


namespace NUMINAMATH_CALUDE_fraction_equals_ratio_l2108_210858

def numerator_terms : List Nat := [12, 28, 44, 60, 76]
def denominator_terms : List Nat := [8, 24, 40, 56, 72]

def fraction_term (n : Nat) : Rat :=
  (n^4 + 400 : Rat) / 1

theorem fraction_equals_ratio : 
  (List.prod (numerator_terms.map fraction_term)) / 
  (List.prod (denominator_terms.map fraction_term)) = 
  6712 / 148 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_ratio_l2108_210858


namespace NUMINAMATH_CALUDE_factorial_gcd_l2108_210818

theorem factorial_gcd : Nat.gcd (Nat.factorial 6) (Nat.factorial 9) = Nat.factorial 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_gcd_l2108_210818


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2108_210853

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in the compound -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 1

/-- The molecular weight of a compound is the sum of the atomic weights of its constituent atoms -/
def molecular_weight (n_weight o_weight : ℝ) (n_count o_count : ℕ) : ℝ :=
  n_weight * n_count + o_weight * o_count

/-- The molecular weight of the compound is 44.02 amu -/
theorem compound_molecular_weight :
  molecular_weight nitrogen_weight oxygen_weight nitrogen_count oxygen_count = 44.02 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2108_210853


namespace NUMINAMATH_CALUDE_distinct_paintings_l2108_210825

/-- The number of disks --/
def n : ℕ := 7

/-- The number of blue disks --/
def blue : ℕ := 4

/-- The number of red disks --/
def red : ℕ := 2

/-- The number of green disks --/
def green : ℕ := 1

/-- The number of symmetry operations (identity and reflection) --/
def symmetries : ℕ := 2

/-- The total number of colorings --/
def total_colorings : ℕ := (Nat.choose n blue) * (Nat.choose (n - blue) red) * (Nat.choose (n - blue - red) green)

/-- The number of colorings fixed by reflection --/
def fixed_colorings : ℕ := 3

/-- The theorem stating the number of distinct paintings --/
theorem distinct_paintings : (total_colorings + fixed_colorings) / symmetries = 54 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paintings_l2108_210825


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l2108_210849

theorem quadratic_equation_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*(a-1)*x + a^2 - 7*a - 4 = 0 ↔ (x = x₁ ∨ x = x₂)) →
  x₁*x₂ - 3*x₁ - 3*x₂ - 2 = 0 →
  (1 + 4/(a^2 - 4)) * (a + 2)/a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l2108_210849


namespace NUMINAMATH_CALUDE_expression_simplification_l2108_210882

theorem expression_simplification (x : ℝ) : 
  x * (x * (x * (x - 3) - 5) + 11) + 2 = x^4 - 3*x^3 - 5*x^2 + 11*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2108_210882


namespace NUMINAMATH_CALUDE_largest_prime_factor_f9_div_f3_l2108_210835

def f (n : ℕ) : ℕ := (3^n + 1) / 2

theorem largest_prime_factor_f9_div_f3 :
  let ratio : ℕ := f 9 / f 3
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ ratio ∧ p = 37 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ ratio → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_f9_div_f3_l2108_210835


namespace NUMINAMATH_CALUDE_basketball_lineup_theorem_l2108_210834

/-- The number of ways to choose 7 starters from 18 players, including a set of 3 triplets,
    with exactly two of the triplets in the starting lineup. -/
def basketball_lineup_count : ℕ := sorry

/-- The total number of players on the team. -/
def total_players : ℕ := 18

/-- The number of triplets in the team. -/
def triplets : ℕ := 3

/-- The number of starters to be chosen. -/
def starters : ℕ := 7

/-- The number of triplets that must be in the starting lineup. -/
def triplets_in_lineup : ℕ := 2

theorem basketball_lineup_theorem : 
  basketball_lineup_count = (Nat.choose triplets triplets_in_lineup) * 
    (Nat.choose (total_players - triplets) (starters - triplets_in_lineup)) := by sorry

end NUMINAMATH_CALUDE_basketball_lineup_theorem_l2108_210834


namespace NUMINAMATH_CALUDE_consecutive_points_ratio_l2108_210888

/-- Given five consecutive points on a line, prove the ratio of distances -/
theorem consecutive_points_ratio (a b c d e : ℝ) : 
  (b - a = 5) → 
  (c - a = 11) → 
  (e - d = 7) → 
  (e - a = 20) → 
  (c - b) / (d - c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_ratio_l2108_210888


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_54_l2108_210869

/-- A triangle with side lengths 9, 12, and 15 units has an area of 54 square units. -/
theorem triangle_area : ℝ :=
  let a := 9
  let b := 12
  let c := 15
  let s := (a + b + c) / 2  -- semi-perimeter
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))  -- Heron's formula
  54

/-- The theorem statement -/
theorem triangle_area_is_54 : triangle_area = 54 := by sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_54_l2108_210869


namespace NUMINAMATH_CALUDE_irrational_ratio_transformation_l2108_210840

theorem irrational_ratio_transformation : ∃ x y : ℝ, 
  (Irrational x) ∧ (Irrational y) ∧ (x ≠ y) ∧ ((7 + x) / (11 + y) = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_irrational_ratio_transformation_l2108_210840


namespace NUMINAMATH_CALUDE_partner_investment_period_l2108_210829

/-- Given two partners P and Q with investment and profit ratios, and Q's investment period,
    calculate P's investment period. -/
theorem partner_investment_period
  (investment_ratio_p investment_ratio_q : ℕ)
  (profit_ratio_p profit_ratio_q : ℕ)
  (q_months : ℕ)
  (h_investment : investment_ratio_p * 5 = investment_ratio_q * 7)
  (h_profit : profit_ratio_p * 9 = profit_ratio_q * 7)
  (h_q_months : q_months = 9) :
  ∃ (p_months : ℕ),
    p_months * profit_ratio_q * investment_ratio_q =
    q_months * profit_ratio_p * investment_ratio_p ∧
    p_months = 5 :=
by sorry

end NUMINAMATH_CALUDE_partner_investment_period_l2108_210829


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2108_210843

theorem polynomial_simplification (x : ℝ) :
  (2 * x^3 - 5 * x^2 + 8 * x - 9) + (3 * x^4 - 2 * x^3 + x^2 - 8 * x + 6) = 3 * x^4 - 4 * x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2108_210843


namespace NUMINAMATH_CALUDE_complex_square_sum_l2108_210838

theorem complex_square_sum (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (1 + 2 / Complex.I)^2 = a + b * Complex.I →
  a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l2108_210838


namespace NUMINAMATH_CALUDE_right_triangle_with_bisected_hypotenuse_l2108_210810

/-- A triangle with vertices A, B, and C -/
structure Triangle (V : Type*) where
  A : V
  B : V
  C : V

/-- A circle with center O and radius r -/
structure Circle (V : Type*) where
  O : V
  r : ℝ

/-- The property of being a right-angled triangle -/
def IsRightAngled {V : Type*} (t : Triangle V) : Prop :=
  sorry

/-- The property that a circle is constructed on a line segment as its diameter -/
def CircleOnDiameter {V : Type*} (c : Circle V) (A B : V) : Prop :=
  sorry

/-- The property that a point is the midpoint of a line segment -/
def IsMidpoint {V : Type*} (M A B : V) : Prop :=
  sorry

/-- The measure of an angle in degrees -/
def AngleMeasure {V : Type*} (A B C : V) : ℝ :=
  sorry

theorem right_triangle_with_bisected_hypotenuse 
  {V : Type*} (t : Triangle V) (c : Circle V) (M : V) :
  IsRightAngled t →
  CircleOnDiameter c t.A t.C →
  IsMidpoint M t.A t.B →
  AngleMeasure t.B t.A t.C = 45 ∧ 
  AngleMeasure t.A t.B t.C = 45 ∧ 
  AngleMeasure t.A t.C t.B = 90 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_with_bisected_hypotenuse_l2108_210810


namespace NUMINAMATH_CALUDE_flag_distance_not_nine_l2108_210860

theorem flag_distance_not_nine (track_length : ℝ) (num_flags : ℕ) : 
  track_length = 90 → 
  num_flags = 10 → 
  (track_length / (num_flags - 1) ≠ 9) :=
by sorry

end NUMINAMATH_CALUDE_flag_distance_not_nine_l2108_210860


namespace NUMINAMATH_CALUDE_sum_of_squares_l2108_210896

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2108_210896


namespace NUMINAMATH_CALUDE_team_selection_ways_l2108_210866

-- Define the number of teachers and students
def num_teachers : ℕ := 5
def num_students : ℕ := 10

-- Define the function to calculate the number of ways to select one person from a group
def select_one (n : ℕ) : ℕ := n

-- Define the function to calculate the total number of ways to form a team
def total_ways : ℕ := select_one num_teachers * select_one num_students

-- Theorem statement
theorem team_selection_ways : total_ways = 50 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_l2108_210866


namespace NUMINAMATH_CALUDE_coffee_blend_price_l2108_210889

/-- Given two blends of coffee, prove the price of the first blend -/
theorem coffee_blend_price 
  (price_blend2 : ℝ) 
  (total_weight : ℝ) 
  (total_price_per_pound : ℝ) 
  (weight_blend2 : ℝ) 
  (h1 : price_blend2 = 8) 
  (h2 : total_weight = 20) 
  (h3 : total_price_per_pound = 8.4) 
  (h4 : weight_blend2 = 12) : 
  ∃ (price_blend1 : ℝ), price_blend1 = 9 := by
sorry

end NUMINAMATH_CALUDE_coffee_blend_price_l2108_210889


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2108_210887

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (abs q > 1) →
  (a 2 + a 7 = 2) →
  (a 4 * a 5 = -15) →
  (a 12 = -25/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2108_210887


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_area_relation_l2108_210879

/-- Represents a triangular pyramid with right angles at each vertex -/
structure RightTriangularPyramid where
  /-- The length of the first base edge -/
  a : ℝ
  /-- The length of the second base edge -/
  b : ℝ
  /-- The length of the third base edge -/
  c : ℝ
  /-- The length of the first edge from apex to base -/
  e₁ : ℝ
  /-- The length of the second edge from apex to base -/
  e₂ : ℝ
  /-- The length of the third edge from apex to base -/
  e₃ : ℝ
  /-- The area of the first lateral face -/
  t₁ : ℝ
  /-- The area of the second lateral face -/
  t₂ : ℝ
  /-- The area of the third lateral face -/
  t₃ : ℝ
  /-- The area of the base -/
  T : ℝ
  /-- Condition: right angles at vertices -/
  right_angles : a^2 = e₁^2 + e₂^2 ∧ b^2 = e₂^2 + e₃^2 ∧ c^2 = e₃^2 + e₁^2
  /-- Condition: lateral face areas -/
  lateral_areas : t₁ = (1/2) * e₁ * e₂ ∧ t₂ = (1/2) * e₂ * e₃ ∧ t₃ = (1/2) * e₃ * e₁
  /-- Condition: base area -/
  base_area : T = (1/4) * Real.sqrt ((a+b+c)*(a+b-c)*(a-b+c)*(b+c-a))

/-- The square of the base area is equal to the sum of the squares of the lateral face areas -/
theorem right_triangular_pyramid_area_relation (p : RightTriangularPyramid) :
  p.T^2 = p.t₁^2 + p.t₂^2 + p.t₃^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_area_relation_l2108_210879


namespace NUMINAMATH_CALUDE_line_condition_l2108_210877

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if two points are on the same side of a line x - y + 2 = 0 -/
def sameSideOfLine (p1 p2 : Point) : Prop :=
  (p1.x - p1.y + 2) * (p2.x - p2.y + 2) > 0

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point on the line y = kx + b -/
def pointOnLine (l : Line) (x : ℝ) : Point :=
  ⟨x, l.k * x + l.b⟩

theorem line_condition (l : Line) : 
  (∀ x : ℝ, sameSideOfLine (pointOnLine l x) origin) → 
  l.k = 1 ∧ l.b < 2 := by
  sorry

end NUMINAMATH_CALUDE_line_condition_l2108_210877


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2108_210847

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 280000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 2.8
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2108_210847


namespace NUMINAMATH_CALUDE_trip_time_difference_l2108_210857

theorem trip_time_difference (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l2108_210857


namespace NUMINAMATH_CALUDE_product_mod_twelve_l2108_210836

theorem product_mod_twelve : (95 * 97) % 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twelve_l2108_210836


namespace NUMINAMATH_CALUDE_jessica_expense_increase_l2108_210890

-- Define Jessica's monthly expenses last year
def last_year_rent : ℝ := 1000
def last_year_food : ℝ := 200
def last_year_car_insurance : ℝ := 100
def last_year_utilities : ℝ := 50
def last_year_healthcare : ℝ := 150

-- Define the increase rates
def rent_increase_rate : ℝ := 0.3
def food_increase_rate : ℝ := 0.5
def car_insurance_increase_rate : ℝ := 2
def utilities_increase_rate : ℝ := 0.2
def healthcare_increase_rate : ℝ := 1

-- Define the theorem
theorem jessica_expense_increase :
  let this_year_rent := last_year_rent * (1 + rent_increase_rate)
  let this_year_food := last_year_food * (1 + food_increase_rate)
  let this_year_car_insurance := last_year_car_insurance * (1 + car_insurance_increase_rate)
  let this_year_utilities := last_year_utilities * (1 + utilities_increase_rate)
  let this_year_healthcare := last_year_healthcare * (1 + healthcare_increase_rate)
  let last_year_total := last_year_rent + last_year_food + last_year_car_insurance + last_year_utilities + last_year_healthcare
  let this_year_total := this_year_rent + this_year_food + this_year_car_insurance + this_year_utilities + this_year_healthcare
  (this_year_total - last_year_total) * 12 = 9120 :=
by sorry


end NUMINAMATH_CALUDE_jessica_expense_increase_l2108_210890


namespace NUMINAMATH_CALUDE_division_problem_l2108_210841

theorem division_problem (n : ℕ) : 
  n / 15 = 9 ∧ n % 15 = 1 → n = 136 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2108_210841


namespace NUMINAMATH_CALUDE_nahco3_equals_nano3_l2108_210832

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction :=
  (naHCO3 : Moles)
  (hNO3 : Moles)
  (naNO3 : Moles)
  (h2O : Moles)
  (cO2 : Moles)

/-- The chemical equation is balanced -/
axiom balanced_equation (r : Reaction) : r.naHCO3 = r.hNO3 ∧ r.naHCO3 = r.naNO3

/-- The number of moles of HNO3 combined equals the number of moles of NaNO3 formed -/
axiom hno3_equals_nano3 (r : Reaction) : r.hNO3 = r.naNO3

/-- The stoichiometric ratio of NaHCO3 to NaNO3 is 1:1 -/
axiom stoichiometric_ratio (r : Reaction) : r.naHCO3 = r.naNO3

/-- Theorem: The number of moles of NaHCO3 combined equals the number of moles of NaNO3 formed -/
theorem nahco3_equals_nano3 (r : Reaction) : r.naHCO3 = r.naNO3 := by
  sorry

end NUMINAMATH_CALUDE_nahco3_equals_nano3_l2108_210832


namespace NUMINAMATH_CALUDE_unique_base_twelve_l2108_210811

/-- Given a base b ≥ 10, this function checks if the equation 166 × 56 = 8590 is valid in base b -/
def is_valid_equation (b : ℕ) : Prop :=
  b ≥ 10 ∧ 
  (1 * b^2 + 6 * b + 6) * (5 * b + 6) = 8 * b^3 + 5 * b^2 + 9 * b + 0

/-- Theorem stating that 12 is the only base ≥ 10 satisfying the equation -/
theorem unique_base_twelve : 
  (∃ (b : ℕ), is_valid_equation b) ∧ 
  (∀ (b : ℕ), is_valid_equation b → b = 12) := by
  sorry

#check unique_base_twelve

end NUMINAMATH_CALUDE_unique_base_twelve_l2108_210811


namespace NUMINAMATH_CALUDE_point_on_y_axis_implies_a_equals_two_l2108_210850

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
axiom point_on_y_axis (x y : ℝ) : (x, y) ∈ {p : ℝ × ℝ | p.1 = 0} ↔ x = 0

/-- The theorem states that if the point A(a-2, 2a+8) lies on the y-axis, then a = 2 -/
theorem point_on_y_axis_implies_a_equals_two (a : ℝ) :
  (a - 2, 2 * a + 8) ∈ {p : ℝ × ℝ | p.1 = 0} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_implies_a_equals_two_l2108_210850


namespace NUMINAMATH_CALUDE_jakes_friend_candy_and_euros_l2108_210824

/-- Proves the number of candies Jake's friend can purchase and the amount in Euros he will receive --/
theorem jakes_friend_candy_and_euros :
  let feeding_allowance : ℝ := 4
  let fraction_given : ℝ := 1/4
  let candy_price : ℝ := 0.2
  let discount : ℝ := 0.15
  let exchange_rate : ℝ := 0.85
  
  let money_given := feeding_allowance * fraction_given
  let discounted_price := candy_price * (1 - discount)
  let candies_purchasable := ⌊money_given / discounted_price⌋
  let euros_received := money_given * exchange_rate
  
  (candies_purchasable = 5) ∧ (euros_received = 0.85) :=
by
  sorry

#check jakes_friend_candy_and_euros

end NUMINAMATH_CALUDE_jakes_friend_candy_and_euros_l2108_210824


namespace NUMINAMATH_CALUDE_local_minimum_of_f_l2108_210817

/-- The function f(x) = x³ - 4x² + 4x -/
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x

/-- The local minimum value of f(x) is 0 -/
theorem local_minimum_of_f :
  ∃ (a : ℝ), ∀ (x : ℝ), ∃ (ε : ℝ), ε > 0 ∧ 
    (∀ (y : ℝ), |y - a| < ε → f y ≥ f a) ∧
    f a = 0 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_of_f_l2108_210817


namespace NUMINAMATH_CALUDE_ab_value_l2108_210844

-- Define the sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

-- State the theorem
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2108_210844


namespace NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l2108_210805

def total_trees : ℕ := 14
def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 5

theorem birch_trees_not_adjacent_probability : 
  let total_arrangements := Nat.choose total_trees birch_trees
  let non_birch_trees := maple_trees + oak_trees
  let valid_arrangements := Nat.choose (non_birch_trees + 1) birch_trees
  (valid_arrangements : ℚ) / total_arrangements = 18 / 143 := by
  sorry

end NUMINAMATH_CALUDE_birch_trees_not_adjacent_probability_l2108_210805


namespace NUMINAMATH_CALUDE_incorrect_statement_l2108_210814

theorem incorrect_statement : ¬(∀ m : ℝ, (∃ x : ℝ, x^2 + x - m = 0) → m > 0) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2108_210814


namespace NUMINAMATH_CALUDE_half_day_percentage_l2108_210846

def total_students : ℕ := 80
def full_day_students : ℕ := 60

theorem half_day_percentage :
  (total_students - full_day_students) / total_students * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_half_day_percentage_l2108_210846


namespace NUMINAMATH_CALUDE_black_balls_count_l2108_210885

theorem black_balls_count (white_balls : ℕ) (prob_white : ℚ) : 
  white_balls = 5 →
  prob_white = 5 / 11 →
  ∃ (total_balls : ℕ), 
    (prob_white = white_balls / total_balls) ∧
    (total_balls - white_balls = 6) :=
by sorry

end NUMINAMATH_CALUDE_black_balls_count_l2108_210885


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l2108_210830

/-- Given a circle x^2 + y^2 = 9 and a line y = x + b, if there are exactly two points
    on the circle that have a distance of 1 to the line, then b is in the range
    (-4√2, -2√2) ∪ (2√2, 4√2) -/
theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), 
    p.1^2 + p.2^2 = 9 ∧ 
    q.1^2 + q.2^2 = 9 ∧ 
    p ≠ q ∧
    (abs (p.2 - p.1 - b) / Real.sqrt 2 = 1) ∧
    (abs (q.2 - q.1 - b) / Real.sqrt 2 = 1)) →
  (b > 2 * Real.sqrt 2 ∧ b < 4 * Real.sqrt 2) ∨ 
  (b < -2 * Real.sqrt 2 ∧ b > -4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l2108_210830


namespace NUMINAMATH_CALUDE_cube_has_eight_vertices_l2108_210878

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := 8

/-- Theorem: A cube has 8 vertices -/
theorem cube_has_eight_vertices (c : Cube) : num_vertices c = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_eight_vertices_l2108_210878


namespace NUMINAMATH_CALUDE_thirteen_in_binary_l2108_210861

theorem thirteen_in_binary : 
  (13 : ℕ).digits 2 = [1, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_thirteen_in_binary_l2108_210861


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l2108_210880

theorem det_trig_matrix_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![1, Real.sin (a - b), Real.sin a],
                                       ![Real.sin (a - b), 1, Real.sin b],
                                       ![Real.sin a, Real.sin b, 1]]
  Matrix.det M = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l2108_210880


namespace NUMINAMATH_CALUDE_largest_seven_digit_divisible_by_337_l2108_210812

theorem largest_seven_digit_divisible_by_337 :
  ∀ n : ℕ, n ≤ 9999999 → n % 337 = 0 → n ≤ 9999829 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_seven_digit_divisible_by_337_l2108_210812


namespace NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l2108_210859

theorem fraction_sum_equals_percentage : (4/20 : ℚ) + (8/200 : ℚ) + (12/2000 : ℚ) = (246/1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_percentage_l2108_210859


namespace NUMINAMATH_CALUDE_a_b_equality_l2108_210852

theorem a_b_equality (a b : ℝ) 
  (h1 : a * b = 1) 
  (h2 : (a + b + 2) / 4 = 1 / (a + 1) + 1 / (b + 1)) : 
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_a_b_equality_l2108_210852


namespace NUMINAMATH_CALUDE_largest_number_game_l2108_210864

theorem largest_number_game (a b c d : ℤ) 
  (eq1 : (a + b + c) / 3 + d = 17)
  (eq2 : (a + b + d) / 3 + c = 21)
  (eq3 : (a + c + d) / 3 + b = 23)
  (eq4 : (b + c + d) / 3 + a = 29) :
  max a (max b (max c d)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_game_l2108_210864


namespace NUMINAMATH_CALUDE_platform_length_l2108_210893

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (post_time : ℝ) (platform_time : ℝ) :
  train_length = 150 →
  post_time = 15 →
  platform_time = 25 →
  ∃ (platform_length : ℝ),
    platform_length = 100 ∧
    train_length / post_time = (train_length + platform_length) / platform_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2108_210893


namespace NUMINAMATH_CALUDE_no_simultaneous_negative_polynomials_l2108_210800

theorem no_simultaneous_negative_polynomials :
  ∀ (m n : ℝ), ¬(3 * m^2 + 4 * m * n - 2 * n^2 < 0 ∧ -m^2 - 4 * m * n + 3 * n^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_negative_polynomials_l2108_210800


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2108_210837

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > y) (h2 : y > 0) 
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) : x + y ≥ 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2108_210837


namespace NUMINAMATH_CALUDE_equidistant_point_x_coord_l2108_210865

/-- The point (x, y) that is equidistant from the x-axis, y-axis, and the line 2x + 3y = 6 -/
def equidistant_point (x y : ℝ) : Prop :=
  let d_x_axis := |y|
  let d_y_axis := |x|
  let d_line := |2*x + 3*y - 6| / Real.sqrt 13
  d_x_axis = d_y_axis ∧ d_x_axis = d_line

/-- The x-coordinate of the equidistant point is 6/5 -/
theorem equidistant_point_x_coord :
  ∃ y : ℝ, equidistant_point (6/5) y :=
sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coord_l2108_210865


namespace NUMINAMATH_CALUDE_x_equals_y_at_half_l2108_210898

theorem x_equals_y_at_half (t : ℝ) : 
  let x := 1 - 4 * t
  let y := 2 * t - 2
  t = 0.5 → x = y := by sorry

end NUMINAMATH_CALUDE_x_equals_y_at_half_l2108_210898


namespace NUMINAMATH_CALUDE_law_school_applicants_l2108_210816

theorem law_school_applicants (total : ℕ) (pol_sci : ℕ) (high_gpa : ℕ) (pol_sci_high_gpa : ℕ) 
  (h1 : total = 40)
  (h2 : pol_sci = 15)
  (h3 : high_gpa = 20)
  (h4 : pol_sci_high_gpa = 5) :
  total - pol_sci - high_gpa + pol_sci_high_gpa = 10 :=
by sorry

end NUMINAMATH_CALUDE_law_school_applicants_l2108_210816


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l2108_210892

theorem least_sum_of_bases (c d : ℕ) (h1 : c > 0) (h2 : d > 0) 
  (h3 : 3 * c + 6 = 6 * d + 3) : 
  (∀ x y : ℕ, x > 0 → y > 0 → 3 * x + 6 = 6 * y + 3 → c + d ≤ x + y) ∧ c + d = 5 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l2108_210892


namespace NUMINAMATH_CALUDE_hexagon_five_layers_dots_l2108_210870

/-- Calculates the number of dots in a hexagonal layer -/
def dots_in_layer (n : ℕ) : ℕ := 6 * n

/-- Calculates the total number of dots up to and including a given layer -/
def total_dots (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => total_dots m + dots_in_layer (m + 1)

theorem hexagon_five_layers_dots :
  total_dots 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_five_layers_dots_l2108_210870


namespace NUMINAMATH_CALUDE_unit_digit_of_fraction_l2108_210899

theorem unit_digit_of_fraction : 
  (998 * 999 * 1000 * 1001 * 1002 * 1003) / 10000 % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_unit_digit_of_fraction_l2108_210899


namespace NUMINAMATH_CALUDE_log_base_2_negative_range_l2108_210826

-- Define the function f(x) = lg x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem log_base_2_negative_range :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  {x : ℝ | f x < 0} = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_log_base_2_negative_range_l2108_210826


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1023rd_term_l2108_210831

def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℚ) : Prop :=
  ∃ (d : ℚ), a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d

def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_1023rd_term (p r : ℚ) :
  arithmetic_sequence (2*p) 15 (4*p+r) (4*p-r) →
  nth_term (2*p) ((4*p-r) - (4*p+r)) 1023 = 61215 / 14 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1023rd_term_l2108_210831


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l2108_210801

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 2) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 2) ∧ 
  (n % 11 = 0) ∧ 
  (∀ m : ℕ, m > 1 → 
    (m % 3 = 2) → 
    (m % 4 = 2) → 
    (m % 5 = 2) → 
    (m % 6 = 2) → 
    (m % 11 = 0) → 
    m ≥ n) ∧
  n = 242 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l2108_210801


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2108_210876

theorem min_value_trig_expression (α β : ℝ) :
  9 * (Real.cos α)^2 - 10 * Real.cos α * Real.sin β - 8 * Real.cos β * Real.sin α + 17 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2108_210876


namespace NUMINAMATH_CALUDE_largest_n_dividing_30_factorial_l2108_210804

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem largest_n_dividing_30_factorial : 
  (∀ n : ℕ, n > 7 → ¬(divides (18^n) (factorial 30))) ∧ 
  (divides (18^7) (factorial 30)) := by
sorry

end NUMINAMATH_CALUDE_largest_n_dividing_30_factorial_l2108_210804


namespace NUMINAMATH_CALUDE_summer_birth_year_divisibility_l2108_210827

theorem summer_birth_year_divisibility : ∃ (x y : ℕ), 
  x < y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  (1961 - x) % x = 0 ∧ 
  (1961 - y) % y = 0 := by
sorry

end NUMINAMATH_CALUDE_summer_birth_year_divisibility_l2108_210827


namespace NUMINAMATH_CALUDE_certain_number_proof_l2108_210873

theorem certain_number_proof : ∃ x : ℝ, (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2108_210873


namespace NUMINAMATH_CALUDE_function_inequality_existence_l2108_210867

theorem function_inequality_existence (f : ℝ → ℝ) 
  (hf : ∀ x, 0 < x → 0 < f x) : 
  ¬(∀ x y, 0 < x ∧ 0 < y → f (x + y) ≥ f x + y * f (f x)) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_existence_l2108_210867


namespace NUMINAMATH_CALUDE_james_run_time_l2108_210806

/-- The time it takes James to run 100 meters given John's performance and their speed differences -/
theorem james_run_time (john_total_time john_initial_distance john_initial_time total_distance
  james_initial_distance james_initial_time speed_difference : ℝ)
  (h1 : john_total_time = 13)
  (h2 : john_initial_distance = 4)
  (h3 : john_initial_time = 1)
  (h4 : total_distance = 100)
  (h5 : james_initial_distance = 10)
  (h6 : james_initial_time = 2)
  (h7 : speed_difference = 2)
  : ∃ james_total_time : ℝ, james_total_time = 11 :=
by sorry

end NUMINAMATH_CALUDE_james_run_time_l2108_210806


namespace NUMINAMATH_CALUDE_tree_growth_fraction_l2108_210875

/-- Represents the growth of a tree over time -/
def TreeGrowth (initial_height : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_height + growth_rate * years

theorem tree_growth_fraction :
  let initial_height : ℝ := 4
  let growth_rate : ℝ := 0.5
  let height_at_4_years := TreeGrowth initial_height growth_rate 4
  let height_at_6_years := TreeGrowth initial_height growth_rate 6
  (height_at_6_years - height_at_4_years) / height_at_4_years = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_fraction_l2108_210875


namespace NUMINAMATH_CALUDE_nested_cube_roots_l2108_210874

theorem nested_cube_roots (N M : ℝ) (hN : N > 1) (hM : M > 1) :
  (N * (M * (N * M^(1/3))^(1/3))^(1/3))^(1/3) = N^(2/3) * M^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_roots_l2108_210874


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l2108_210848

def arithmetic_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℚ)
  (h_seq : arithmetic_geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l2108_210848


namespace NUMINAMATH_CALUDE_unqualified_pieces_l2108_210884

theorem unqualified_pieces (total_products : ℕ) (pass_rate : ℚ) : 
  total_products = 400 → pass_rate = 98 / 100 → 
  ↑total_products * (1 - pass_rate) = 8 := by
  sorry

end NUMINAMATH_CALUDE_unqualified_pieces_l2108_210884


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2108_210845

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) :
  ∃ (n : ℕ), n = 2 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 8 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2108_210845


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2108_210813

theorem batsman_average_increase (total_runs_before : ℕ) : 
  let innings_before : ℕ := 10
  let new_score : ℕ := 80
  let new_average : ℝ := 30
  let old_average : ℝ := total_runs_before / innings_before
  let increase : ℝ := new_average - old_average
  (total_runs_before + new_score) / (innings_before + 1) = new_average →
  increase = 5 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l2108_210813


namespace NUMINAMATH_CALUDE_janice_work_hours_janice_work_hours_unique_l2108_210807

/-- Calculates the total pay for a given number of hours worked -/
def totalPay (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    10 * hours
  else
    400 + 15 * (hours - 40)

/-- Theorem stating that 60 hours of work results in $700 pay -/
theorem janice_work_hours :
  totalPay 60 = 700 :=
by sorry

/-- Theorem stating that 60 is the unique number of hours that results in $700 pay -/
theorem janice_work_hours_unique :
  ∀ h : ℕ, totalPay h = 700 → h = 60 :=
by sorry

end NUMINAMATH_CALUDE_janice_work_hours_janice_work_hours_unique_l2108_210807


namespace NUMINAMATH_CALUDE_valid_triples_equal_solution_set_l2108_210881

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(23, 24, 30), (12, 30, 31), (9, 18, 40), (9, 30, 32), (4, 15, 42), (15, 22, 36), (4, 30, 33)}

theorem valid_triples_equal_solution_set :
  {(a, b, c) | is_valid_triple a b c} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_valid_triples_equal_solution_set_l2108_210881


namespace NUMINAMATH_CALUDE_height_of_cube_with_corner_cut_l2108_210828

/-- The height of a cube with one corner cut off -/
theorem height_of_cube_with_corner_cut (s : ℝ) (h : s = 2) :
  let diagonal := s * Real.sqrt 3
  let cut_face_side := diagonal / Real.sqrt 2
  let cut_face_area := Real.sqrt 3 / 4 * cut_face_side^2
  let pyramid_volume := 1 / 6 * s^3
  let remaining_height := s - (3 * pyramid_volume) / cut_face_area
  remaining_height = 2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_height_of_cube_with_corner_cut_l2108_210828


namespace NUMINAMATH_CALUDE_tree_planting_variance_l2108_210851

def group_data : List (Nat × Nat) := [(5, 3), (6, 4), (7, 3)]

def total_groups : Nat := (group_data.map Prod.snd).sum

theorem tree_planting_variance :
  let mean : Rat := (group_data.map (λ (x, y) => x * y)).sum / total_groups
  let variance : Rat := (group_data.map (λ (x, y) => y * ((x : Rat) - mean)^2)).sum / total_groups
  variance = 6/10 := by sorry

end NUMINAMATH_CALUDE_tree_planting_variance_l2108_210851


namespace NUMINAMATH_CALUDE_exactly_one_inscribed_rhombus_l2108_210871

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop

/-- The first hyperbola C₁: x²/a² - y²/b² = 1 -/
def C₁ (a b : ℝ) : Hyperbola :=
  { a := a
    b := b
    eq := fun x y ↦ x^2 / a^2 - y^2 / b^2 = 1 }

/-- The second hyperbola C₂: y²/b² - x²/a² = 1 -/
def C₂ (a b : ℝ) : Hyperbola :=
  { a := a
    b := b
    eq := fun x y ↦ y^2 / b^2 - x^2 / a^2 = 1 }

/-- A predicate indicating whether a hyperbola has an inscribed rhombus -/
def has_inscribed_rhombus (h : Hyperbola) : Prop := sorry

/-- The main theorem stating that exactly one of C₁ or C₂ has an inscribed rhombus -/
theorem exactly_one_inscribed_rhombus (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (has_inscribed_rhombus (C₁ a b) ∧ ¬has_inscribed_rhombus (C₂ a b)) ∨
  (has_inscribed_rhombus (C₂ a b) ∧ ¬has_inscribed_rhombus (C₁ a b)) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_inscribed_rhombus_l2108_210871


namespace NUMINAMATH_CALUDE_intersection_A_B_l2108_210855

def A : Set ℝ := {x | x^2 - 4 > 0}
def B : Set ℝ := {x | x + 2 < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2108_210855


namespace NUMINAMATH_CALUDE_reseating_problem_l2108_210823

/-- Number of ways n people can be reseated according to the rules -/
def S : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n + 3 => S (n + 2) + S (n + 1)

/-- The reseating problem for 12 people -/
theorem reseating_problem : S 12 = 89 := by
  sorry

end NUMINAMATH_CALUDE_reseating_problem_l2108_210823
