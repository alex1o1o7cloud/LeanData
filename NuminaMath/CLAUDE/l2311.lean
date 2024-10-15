import Mathlib

namespace NUMINAMATH_CALUDE_marriage_year_proof_l2311_231163

def year_of_marriage : ℕ := 1980
def year_child1_born : ℕ := 1982
def year_child2_born : ℕ := 1984
def reference_year : ℕ := 1986

theorem marriage_year_proof :
  (reference_year - year_child1_born) + (reference_year - year_child2_born) = reference_year - year_of_marriage :=
by sorry

end NUMINAMATH_CALUDE_marriage_year_proof_l2311_231163


namespace NUMINAMATH_CALUDE_gideon_age_proof_l2311_231110

/-- The number of years in a century -/
def century : ℕ := 100

/-- Gideon's initial number of marbles -/
def initial_marbles : ℕ := century

/-- The fraction of marbles Gideon gives to his sister -/
def fraction_given : ℚ := 3/4

/-- Gideon's current age -/
def gideon_age : ℕ := 45

theorem gideon_age_proof :
  gideon_age = initial_marbles - (fraction_given * initial_marbles).num - 5 :=
sorry

end NUMINAMATH_CALUDE_gideon_age_proof_l2311_231110


namespace NUMINAMATH_CALUDE_kelly_games_to_give_away_l2311_231132

/-- Given that Kelly has a certain number of Nintendo games and wants to keep a specific number,
    prove that the number of games she needs to give away is the difference between these two numbers. -/
theorem kelly_games_to_give_away (initial_nintendo_games kept_nintendo_games : ℕ) :
  initial_nintendo_games ≥ kept_nintendo_games →
  initial_nintendo_games - kept_nintendo_games =
  initial_nintendo_games - kept_nintendo_games :=
by
  sorry

#check kelly_games_to_give_away 20 12

end NUMINAMATH_CALUDE_kelly_games_to_give_away_l2311_231132


namespace NUMINAMATH_CALUDE_money_division_l2311_231162

theorem money_division (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a = 80) (h2 : a = (2/3) * (b + c)) (h3 : b = (6/9) * (a + c)) : 
  a + b + c = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l2311_231162


namespace NUMINAMATH_CALUDE_equation_solution_l2311_231170

theorem equation_solution : ∃! x : ℝ, (3 / (x - 2) = 6 / (x - 3)) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2311_231170


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2311_231184

theorem smallest_k_with_remainder_one : ∃! k : ℕ, 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 7 = 1 ∧ 
  k % 5 = 1 ∧ 
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 5 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2311_231184


namespace NUMINAMATH_CALUDE_snail_return_time_l2311_231160

/-- Represents the movement of a point on a plane -/
structure PointMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the position of the point at a given time -/
def Position := ℝ × ℝ

/-- Returns the position of the point after a given time -/
noncomputable def positionAfterTime (m : PointMovement) (t : ℝ) : Position :=
  sorry

/-- Checks if the point has returned to its starting position -/
def hasReturnedToStart (m : PointMovement) (t : ℝ) : Prop :=
  positionAfterTime m t = (0, 0)

/-- The main theorem to prove -/
theorem snail_return_time (m : PointMovement) 
    (h1 : m.speed > 0)
    (h2 : m.turnInterval = 15)
    (h3 : m.turnAngle = 90) :
    ∀ t : ℝ, hasReturnedToStart m t → ∃ n : ℕ, t = 60 * n := by
  sorry

end NUMINAMATH_CALUDE_snail_return_time_l2311_231160


namespace NUMINAMATH_CALUDE_minimize_expression_l2311_231128

theorem minimize_expression (a b : ℝ) (ha : a > 0) (hb : b > 2) (hab : a + b = 3) :
  ∃ (min_a : ℝ), min_a = 2/3 ∧
  ∀ (x : ℝ), x > 0 → x + b = 3 →
  (4/x + 1/(b-2)) ≥ (4/min_a + 1/(b-2)) :=
by sorry

end NUMINAMATH_CALUDE_minimize_expression_l2311_231128


namespace NUMINAMATH_CALUDE_hundred_day_previous_year_is_thursday_l2311_231190

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (year : Year) (day : ℕ) : DayOfWeek :=
  sorry

/-- Checks if a year is a leap year -/
def isLeapYear (year : Year) : Bool :=
  sorry

theorem hundred_day_previous_year_is_thursday 
  (N : Year)
  (h1 : dayOfWeek N 300 = DayOfWeek.Tuesday)
  (h2 : dayOfWeek (Year.mk (N.value + 1)) 200 = DayOfWeek.Tuesday) :
  dayOfWeek (Year.mk (N.value - 1)) 100 = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_hundred_day_previous_year_is_thursday_l2311_231190


namespace NUMINAMATH_CALUDE_regression_analysis_properties_l2311_231196

-- Define the basic concepts
def FunctionRelationship : Type := Unit
def CorrelationRelationship : Type := Unit
def RegressionAnalysis : Type := Unit

-- Define properties
def isDeterministic (r : Type) : Prop := sorry
def isNonDeterministic (r : Type) : Prop := sorry
def usedFor (a : Type) (r : Type) : Prop := sorry

-- Theorem statement
theorem regression_analysis_properties :
  isDeterministic FunctionRelationship ∧
  isNonDeterministic CorrelationRelationship ∧
  usedFor RegressionAnalysis CorrelationRelationship :=
by sorry

end NUMINAMATH_CALUDE_regression_analysis_properties_l2311_231196


namespace NUMINAMATH_CALUDE_equation_solution_l2311_231133

theorem equation_solution : ∃! x : ℝ, Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2 :=
by
  -- The unique solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the equation
    sorry
  · -- Prove that any solution must equal 3
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l2311_231133


namespace NUMINAMATH_CALUDE_fabian_sugar_packs_l2311_231112

/-- The number of packs of sugar Fabian wants to buy -/
def sugar_packs : ℕ := 3

/-- The price of apples in dollars per kilogram -/
def apple_price : ℚ := 2

/-- The price of walnuts in dollars per kilogram -/
def walnut_price : ℚ := 6

/-- The price of sugar in dollars per pack -/
def sugar_price : ℚ := apple_price - 1

/-- The amount of apples Fabian wants to buy in kilograms -/
def apple_amount : ℚ := 5

/-- The amount of walnuts Fabian wants to buy in kilograms -/
def walnut_amount : ℚ := 1/2

/-- The total amount Fabian needs to pay in dollars -/
def total_cost : ℚ := 16

theorem fabian_sugar_packs : 
  sugar_packs = (total_cost - apple_price * apple_amount - walnut_price * walnut_amount) / sugar_price := by
  sorry

end NUMINAMATH_CALUDE_fabian_sugar_packs_l2311_231112


namespace NUMINAMATH_CALUDE_at_least_one_hit_l2311_231195

theorem at_least_one_hit (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_hit_l2311_231195


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2311_231108

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : c^2 = a^2 + b^2

/-- Equilateral triangle structure -/
structure EquilateralTriangle where
  side : ℝ

/-- Theorem: Eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (C : Hyperbola) (T : EquilateralTriangle)
  (h1 : T.side = 2 * C.c) -- AF₁ = AF₂ = F₁F₂ = 2c
  (h2 : ∃ (B : ℝ × ℝ), B.1^2 / C.a^2 - B.2^2 / C.b^2 = 1 ∧ 
    (B.1 + C.c)^2 + B.2^2 = (5/4 * T.side)^2) -- B is on the hyperbola and AB = 5/4 * AF₁
  : C.c / C.a = (Real.sqrt 13 + 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2311_231108


namespace NUMINAMATH_CALUDE_subtraction_problem_l2311_231165

theorem subtraction_problem : 
  (845.59 : ℝ) - 249.27 = 596.32 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2311_231165


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2311_231181

theorem quadratic_inequality_always_nonnegative (x : ℝ) : x^2 + 3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_nonnegative_l2311_231181


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_four_l2311_231106

theorem fraction_zero_implies_x_negative_four (x : ℝ) :
  (|x| - 4) / (4 - x) = 0 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_four_l2311_231106


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2311_231111

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a) 
  (h_prod : a 4 * a 8 = 9) : 
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2311_231111


namespace NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_three_largest_interval_l2311_231139

theorem inequality_holds_for_p_greater_than_three (p q : ℝ) (hp : p > 3) (hq : q > 0) :
  (7 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 3 * p^2 * q :=
sorry

theorem largest_interval (p q : ℝ) (hq : q > 0) :
  (∀ q > 0, (7 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 3 * p^2 * q) ↔ p > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_three_largest_interval_l2311_231139


namespace NUMINAMATH_CALUDE_train_length_l2311_231183

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 12 → 
  ∃ length_m : ℝ, abs (length_m - 200.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2311_231183


namespace NUMINAMATH_CALUDE_isabel_homework_problem_l2311_231171

/-- Given a total number of problems, number of finished problems, and number of remaining pages,
    calculate the number of problems per page, assuming each page has an equal number of problems. -/
def problems_per_page (total : ℕ) (finished : ℕ) (pages : ℕ) : ℕ :=
  (total - finished) / pages

/-- Theorem stating that for the given problem, there are 8 problems per page. -/
theorem isabel_homework_problem :
  problems_per_page 72 32 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_problem_l2311_231171


namespace NUMINAMATH_CALUDE_product_QED_l2311_231147

theorem product_QED (Q E D : ℂ) (hQ : Q = 6 + 3*I) (hE : E = -I) (hD : D = 6 - 3*I) :
  Q * E * D = -45 * I :=
by sorry

end NUMINAMATH_CALUDE_product_QED_l2311_231147


namespace NUMINAMATH_CALUDE_min_real_roots_l2311_231101

/-- A polynomial of degree 2010 with real coefficients -/
def RealPolynomial2010 : Type := Polynomial ℝ

/-- The roots of a polynomial -/
def roots (p : RealPolynomial2010) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial2010) : ℕ := sorry

/-- The degree of the polynomial -/
def degree (p : RealPolynomial2010) : ℕ := 2010

theorem min_real_roots (g : RealPolynomial2010) 
  (h1 : degree g = 2010)
  (h2 : distinctAbsValues g = 1006) : 
  realRootCount g ≥ 6 := sorry

end NUMINAMATH_CALUDE_min_real_roots_l2311_231101


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2311_231129

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2311_231129


namespace NUMINAMATH_CALUDE_paths_7x8_grid_l2311_231144

/-- The number of distinct paths on a rectangular grid -/
def gridPaths (width height : ℕ) : ℕ :=
  Nat.choose (width + height) height

/-- Theorem: The number of distinct paths on a 7x8 grid is 6435 -/
theorem paths_7x8_grid :
  gridPaths 7 8 = 6435 := by
  sorry

end NUMINAMATH_CALUDE_paths_7x8_grid_l2311_231144


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_squared_l2311_231121

theorem cube_sum_reciprocal_squared (x : ℝ) (h : 53 = x^6 + 1/x^6) : (x^3 + 1/x^3)^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_squared_l2311_231121


namespace NUMINAMATH_CALUDE_gcd_smallest_prime_factor_subtraction_l2311_231105

theorem gcd_smallest_prime_factor_subtraction : 
  10 - (Nat.minFac (Nat.gcd 105 90)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_smallest_prime_factor_subtraction_l2311_231105


namespace NUMINAMATH_CALUDE_hiking_team_gloves_l2311_231161

/-- The minimum number of gloves needed for a hiking team -/
def minimum_gloves (total_participants small_members medium_members large_members num_activities : ℕ) : ℕ :=
  (small_members + medium_members + large_members) * num_activities

/-- Theorem: The hiking team needs 225 gloves -/
theorem hiking_team_gloves :
  let total_participants := 75
  let small_members := 20
  let medium_members := 38
  let large_members := 17
  let num_activities := 3
  minimum_gloves total_participants small_members medium_members large_members num_activities = 225 := by
  sorry


end NUMINAMATH_CALUDE_hiking_team_gloves_l2311_231161


namespace NUMINAMATH_CALUDE_ex_factory_price_decrease_selling_price_for_profit_l2311_231109

/-- Ex-factory price in 2019 -/
def price_2019 : ℝ := 144

/-- Ex-factory price in 2021 -/
def price_2021 : ℝ := 100

/-- Current selling price -/
def current_price : ℝ := 140

/-- Current daily sales -/
def current_sales : ℝ := 20

/-- Sales increase per price reduction -/
def sales_increase : ℝ := 10

/-- Price reduction step -/
def price_reduction : ℝ := 5

/-- Target daily profit -/
def target_profit : ℝ := 1250

/-- Average yearly percentage decrease in ex-factory price -/
def avg_decrease : ℝ := 16.67

/-- Selling price for desired profit -/
def desired_price : ℝ := 125

theorem ex_factory_price_decrease :
  ∃ (x : ℝ), price_2019 * (1 - x / 100)^2 = price_2021 ∧ x = avg_decrease :=
sorry

theorem selling_price_for_profit :
  ∃ (y : ℝ),
    (y - price_2021) * (current_sales + sales_increase * (current_price - y) / price_reduction) = target_profit ∧
    y = desired_price :=
sorry

end NUMINAMATH_CALUDE_ex_factory_price_decrease_selling_price_for_profit_l2311_231109


namespace NUMINAMATH_CALUDE_mildred_orange_collection_l2311_231158

/-- Mildred's orange collection problem -/
theorem mildred_orange_collection (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 77 → additional = 2 → total = initial + additional → total = 79 := by
  sorry

end NUMINAMATH_CALUDE_mildred_orange_collection_l2311_231158


namespace NUMINAMATH_CALUDE_fraction_product_l2311_231153

theorem fraction_product : (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l2311_231153


namespace NUMINAMATH_CALUDE_laura_debt_l2311_231199

/-- Calculates the total amount owed after applying simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the total amount owed after one year is $37.10 -/
theorem laura_debt : 
  let principal : ℝ := 35
  let rate : ℝ := 0.06
  let time : ℝ := 1
  total_amount_owed principal rate time = 37.10 := by
sorry

end NUMINAMATH_CALUDE_laura_debt_l2311_231199


namespace NUMINAMATH_CALUDE_always_on_iff_odd_l2311_231186

/-- Represents the state of a light bulb -/
inductive BulbState
| On
| Off

/-- Represents a configuration of light bulbs -/
def BulbConfiguration (n : ℕ) := Fin n → BulbState

/-- Function to update the state of bulbs according to the given rule -/
def updateBulbs (n : ℕ) (config : BulbConfiguration n) : BulbConfiguration n :=
  sorry

/-- Predicate to check if a configuration has at least one bulb on -/
def hasOnBulb (n : ℕ) (config : BulbConfiguration n) : Prop :=
  sorry

/-- Theorem stating that there exists a configuration that always has at least one bulb on
    if and only if n is odd -/
theorem always_on_iff_odd (n : ℕ) :
  (∃ (initial : BulbConfiguration n), ∀ (t : ℕ), hasOnBulb n ((updateBulbs n)^[t] initial)) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_always_on_iff_odd_l2311_231186


namespace NUMINAMATH_CALUDE_max_area_rhombus_l2311_231187

/-- Given a rhombus OABC in a rectangular coordinate system xOy with the following properties:
  - The diagonals intersect at point M(x₀, y₀)
  - The hyperbola y = k/x (x > 0) passes through points C and M
  - 2 ≤ x₀ ≤ 4
  Prove that the maximum area of rhombus OABC is 24√2 -/
theorem max_area_rhombus (x₀ y₀ k : ℝ) (hx₀ : 2 ≤ x₀ ∧ x₀ ≤ 4) (hk : k > 0) 
  (h_hyperbola : y₀ = k / x₀) : 
  (∃ (S : ℝ), S = 24 * Real.sqrt 2 ∧ 
    ∀ (A : ℝ), A ≤ S ∧ 
    (∃ (x₁ y₁ : ℝ), 2 ≤ x₁ ∧ x₁ ≤ 4 ∧ 
      y₁ = k / x₁ ∧ 
      A = (3 * Real.sqrt 2 / 2) * x₁^2)) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rhombus_l2311_231187


namespace NUMINAMATH_CALUDE_sons_age_l2311_231138

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 25 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2311_231138


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l2311_231143

open Real

/-- The function f(x) = 3^x - 3^(-x) is odd and increasing on ℝ -/
theorem f_odd_and_increasing :
  let f : ℝ → ℝ := fun x ↦ 3^x - 3^(-x)
  (∀ x, f (-x) = -f x) ∧ StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l2311_231143


namespace NUMINAMATH_CALUDE_andrew_total_donation_l2311_231130

/-- Calculates the total donation amount for a geometric series of donations -/
def totalDonation (initialAmount : ℕ) (commonRatio : ℕ) (startAge : ℕ) (currentAge : ℕ) : ℕ :=
  let numberOfTerms := currentAge - startAge + 1
  initialAmount * (commonRatio ^ numberOfTerms - 1) / (commonRatio - 1)

/-- Theorem stating that Andrew's total donation equals 3,669,609k -/
theorem andrew_total_donation :
  totalDonation 7000 2 11 29 = 3669609000 := by
  sorry


end NUMINAMATH_CALUDE_andrew_total_donation_l2311_231130


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l2311_231103

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 5 -/
def num_y_intercepts : ℕ := 0

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 4 * y + 5

theorem parabola_y_intercepts :
  (∀ y : ℝ, parabola_equation y ≠ 0) ∧ num_y_intercepts = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l2311_231103


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_square_octagon_proof_l2311_231154

/-- The measure of the exterior angle formed by a regular square and a regular octagon that share a common side in a coplanar configuration is 135 degrees. -/
theorem exterior_angle_square_octagon : ℝ → Prop :=
  λ angle : ℝ =>
    let square_interior_angle : ℝ := 90
    let octagon_interior_angle : ℝ := 135
    let total_angle : ℝ := 360
    angle = total_angle - (square_interior_angle + octagon_interior_angle) ∧
    angle = 135

/-- Proof of the theorem -/
theorem exterior_angle_square_octagon_proof :
  ∃ angle : ℝ, exterior_angle_square_octagon angle :=
sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_exterior_angle_square_octagon_proof_l2311_231154


namespace NUMINAMATH_CALUDE_not_or_implies_both_false_l2311_231123

theorem not_or_implies_both_false (p q : Prop) : 
  ¬(p ∨ q) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_or_implies_both_false_l2311_231123


namespace NUMINAMATH_CALUDE_total_pens_l2311_231104

theorem total_pens (black_pens blue_pens : ℕ) :
  black_pens = 4 → blue_pens = 4 → black_pens + blue_pens = 8 :=
by sorry

end NUMINAMATH_CALUDE_total_pens_l2311_231104


namespace NUMINAMATH_CALUDE_eight_members_left_for_treasurer_l2311_231118

/-- Represents a club with members and officer positions -/
structure Club where
  totalMembers : ℕ
  presidentChosen : Bool
  secretaryChosen : Bool

/-- Function to calculate remaining members for treasurer position -/
def remainingMembersForTreasurer (club : Club) : ℕ :=
  club.totalMembers - (if club.presidentChosen then 1 else 0) - (if club.secretaryChosen then 1 else 0)

/-- Theorem stating that in a club of 10 members, after choosing president and secretary,
    there are 8 members left for treasurer position -/
theorem eight_members_left_for_treasurer (club : Club) 
    (h1 : club.totalMembers = 10)
    (h2 : club.presidentChosen = true)
    (h3 : club.secretaryChosen = true) :
  remainingMembersForTreasurer club = 8 := by
  sorry

#eval remainingMembersForTreasurer { totalMembers := 10, presidentChosen := true, secretaryChosen := true }

end NUMINAMATH_CALUDE_eight_members_left_for_treasurer_l2311_231118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2311_231188

/-- 
Given an arithmetic sequence with:
- First term a₁ = -48
- Common difference d = 5
- Last term aₙ = 72

Prove that the sequence has 25 terms.
-/
theorem arithmetic_sequence_length : 
  let a₁ : ℤ := -48  -- First term
  let d : ℤ := 5     -- Common difference
  let aₙ : ℤ := 72   -- Last term
  ∃ n : ℕ, n = 25 ∧ aₙ = a₁ + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2311_231188


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2311_231125

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 - (2*k - 1)*x₁ + k^2 - k = 0 ∧
  x₂^2 - (2*k - 1)*x₂ + k^2 - k = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2311_231125


namespace NUMINAMATH_CALUDE_happy_children_count_l2311_231193

theorem happy_children_count (total : ℕ) (sad : ℕ) (neither : ℕ) (boys : ℕ) (girls : ℕ) 
  (happy_boys : ℕ) (sad_girls : ℕ) (neither_boys : ℕ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : total = 60 := by sorry
  have h2 : sad = 10 := by sorry
  have h3 : neither = 20 := by sorry
  have h4 : boys = 16 := by sorry
  have h5 : girls = 44 := by sorry
  have h6 : happy_boys = 6 := by sorry
  have h7 : sad_girls = 4 := by sorry
  have h8 : neither_boys = 4 := by sorry

  -- Prove that the number of happy children is 30
  have happy_children : ℕ := total - (sad + neither)
  exact happy_children

end NUMINAMATH_CALUDE_happy_children_count_l2311_231193


namespace NUMINAMATH_CALUDE_factorization_equality_l2311_231135

theorem factorization_equality (x : ℝ) : x * (x + 2) - x - 2 = (x + 2) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2311_231135


namespace NUMINAMATH_CALUDE_simplify_expression_l2311_231192

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2311_231192


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l2311_231124

theorem sqrt_difference_approximation : 
  ∃ ε > 0, |Real.sqrt (49 + 81) - Real.sqrt (64 - 36) - 6.1| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l2311_231124


namespace NUMINAMATH_CALUDE_inequality_theorem_l2311_231122

theorem inequality_theorem (a b c : ℝ) (θ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_ineq : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2311_231122


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2311_231175

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * w.x ∧ v.y = k * w.y

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : Vector2D := ⟨x - 1, 2⟩
  let b : Vector2D := ⟨2, 1⟩
  parallel a b → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2311_231175


namespace NUMINAMATH_CALUDE_inequality_preservation_l2311_231182

theorem inequality_preservation (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2311_231182


namespace NUMINAMATH_CALUDE_problem_solution_l2311_231198

def U : Set ℕ := {1, 2, 3, 4, 5}

def A (q : ℤ) : Set ℕ := {x ∈ U | x^2 - 5*x + q = 0}

def B (p : ℤ) : Set ℕ := {x ∈ U | x^2 + p*x + 12 = 0}

theorem problem_solution :
  ∃ (p q : ℤ),
    (U \ A q) ∪ B p = {1, 3, 4, 5} ∧
    p = -7 ∧
    q = 6 ∧
    A q = {2, 3} ∧
    B p = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2311_231198


namespace NUMINAMATH_CALUDE_parabola_transformation_l2311_231131

-- Define the original function
def original_function (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Define the transformation
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1) - 1

-- State the theorem
theorem parabola_transformation :
  ∀ x : ℝ, transform original_function x = x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2311_231131


namespace NUMINAMATH_CALUDE_rational_function_uniqueness_l2311_231174

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The property that f(1) = 2 -/
def HasPropertyOne (f : RationalFunction) : Prop :=
  f 1 = 2

/-- The property that f(xy) = f(x)f(y) - f(x + y) + 1 for all x, y ∈ ℚ -/
def HasPropertyTwo (f : RationalFunction) : Prop :=
  ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

/-- The theorem stating that any function satisfying both properties must be f(x) = x + 1 -/
theorem rational_function_uniqueness (f : RationalFunction)
  (h1 : HasPropertyOne f) (h2 : HasPropertyTwo f) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_uniqueness_l2311_231174


namespace NUMINAMATH_CALUDE_equation_solutions_l2311_231156

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2) / (x + 2) * (x + (15 - x) / (x + 2))
  ∃ (s : Set ℝ), s = {12, -3, -3 + Real.sqrt 33, -3 - Real.sqrt 33} ∧ 
    ∀ x ∈ s, f x = 54 ∧ 
    ∀ y ∉ s, f y ≠ 54 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2311_231156


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l2311_231102

/-- A function f: ℝ → ℝ that is symmetric about x = 2 and has exactly four distinct real roots -/
def SymmetricFourRootFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∃! (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0)

/-- The sum of the four distinct real roots of a SymmetricFourRootFunction is 8 -/
theorem sum_of_roots_is_eight (f : ℝ → ℝ) (h : SymmetricFourRootFunction f) :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    a + b + c + d = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l2311_231102


namespace NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_l2311_231191

theorem largest_four_digit_perfect_cube : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → (∃ m : ℕ, n = m^3) → n ≤ 9261 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_l2311_231191


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2311_231141

/-- Triangle ABC with sides a, b, and c satisfying |a-3| + (b-4)^2 = 0 -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  h : |a - 3| + (b - 4)^2 = 0

/-- The perimeter of an isosceles triangle -/
def isoscelesPerimeter (t : TriangleABC) : Set ℝ :=
  {10, 11}

theorem triangle_abc_properties (t : TriangleABC) :
  t.a = 3 ∧ t.b = 4 ∧ 1 < t.c ∧ t.c < 7 ∧
  (t.a = t.b ∨ t.a = t.c ∨ t.b = t.c → t.a + t.b + t.c ∈ isoscelesPerimeter t) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2311_231141


namespace NUMINAMATH_CALUDE_windows_already_installed_l2311_231194

/-- Proves that the number of windows already installed is 6 -/
theorem windows_already_installed
  (total_windows : ℕ)
  (install_time_per_window : ℕ)
  (time_left : ℕ)
  (h1 : total_windows = 10)
  (h2 : install_time_per_window = 5)
  (h3 : time_left = 20) :
  total_windows - (time_left / install_time_per_window) = 6 := by
  sorry

#check windows_already_installed

end NUMINAMATH_CALUDE_windows_already_installed_l2311_231194


namespace NUMINAMATH_CALUDE_number_problem_l2311_231142

theorem number_problem (N : ℝ) : 
  1.15 * ((1/4) * (1/3) * (2/5) * N) = 23 → 0.5 * N = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2311_231142


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2311_231140

theorem cubic_root_sum_cubes (r s t : ℂ) : 
  (9 * r^3 + 2023 * r + 4047 = 0) →
  (9 * s^3 + 2023 * s + 4047 = 0) →
  (9 * t^3 + 2023 * t + 4047 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1349 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l2311_231140


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l2311_231197

theorem cos_2alpha_value (α : Real) (h : Real.tan (π/4 - α) = -1/3) : 
  Real.cos (2*α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l2311_231197


namespace NUMINAMATH_CALUDE_equation_transformation_l2311_231180

theorem equation_transformation (x : ℝ) (y : ℝ) (h : y = (x^2 + 2) / (x + 1)) :
  ((x^2 + 2) / (x + 1) + (5*x + 5) / (x^2 + 2) = 6) ↔ (y^2 - 6*y + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_transformation_l2311_231180


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2311_231173

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_decreasing : ∀ n, a (n + 1) < a n)
  (h_geom : geometric_sequence a)
  (h_prod : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2311_231173


namespace NUMINAMATH_CALUDE_bernoullis_inequality_l2311_231166

theorem bernoullis_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : n > 0) :
  (1 + x)^n ≥ 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_bernoullis_inequality_l2311_231166


namespace NUMINAMATH_CALUDE_red_cube_possible_l2311_231176

/-- Represents a small cube with colored faces -/
structure SmallCube where
  blue_faces : Nat
  red_faces : Nat

/-- Represents the larger cube assembled from small cubes -/
structure LargeCube where
  small_cubes : List SmallCube
  visible_red_faces : Nat

/-- The theorem to be proved -/
theorem red_cube_possible 
  (cubes : List SmallCube) 
  (h1 : cubes.length = 8)
  (h2 : ∀ c ∈ cubes, c.blue_faces + c.red_faces = 6)
  (h3 : (cubes.map SmallCube.blue_faces).sum = 16)
  (h4 : ∃ lc : LargeCube, lc.small_cubes = cubes ∧ lc.visible_red_faces = 8) :
  ∃ lc : LargeCube, lc.small_cubes = cubes ∧ lc.visible_red_faces = 24 := by
  sorry

end NUMINAMATH_CALUDE_red_cube_possible_l2311_231176


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l2311_231115

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_large_number : sumOfDigits (2^2010 * 5^2008 * 7) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l2311_231115


namespace NUMINAMATH_CALUDE_algebraic_identity_l2311_231116

theorem algebraic_identity (a b : ℝ) : 
  a = (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3) →
  b = (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) →
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2311_231116


namespace NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l2311_231150

/-- Calculates the total cost of fruits with applied discounts -/
def total_cost_with_discounts (apple_kg : ℝ) (apple_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
  (orange_kg : ℝ) (orange_price : ℝ) (banana_kg : ℝ) (banana_price : ℝ)
  (apple_discount : ℝ) (orange_discount : ℝ) : ℝ :=
  let apple_cost := apple_kg * apple_price * (1 - apple_discount)
  let mango_cost := mango_kg * mango_price
  let orange_cost := orange_kg * orange_price * (1 - orange_discount)
  let banana_cost := banana_kg * banana_price
  apple_cost + mango_cost + orange_cost + banana_cost

/-- Theorem stating that the total cost of Tom's fruit purchase is $1391.5 -/
theorem toms_fruit_purchase_cost :
  total_cost_with_discounts 8 70 9 65 5 50 3 30 0.1 0.15 = 1391.5 := by
  sorry

end NUMINAMATH_CALUDE_toms_fruit_purchase_cost_l2311_231150


namespace NUMINAMATH_CALUDE_perfect_square_addition_l2311_231134

theorem perfect_square_addition (n : Nat) : ∃ (m : Nat), (n + 49)^2 = 4440 + 49 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_addition_l2311_231134


namespace NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l2311_231157

def billion : ℕ := 10^9

theorem eleven_billion_scientific_notation : 
  11 * billion = 11 * 10^9 ∧ 11 * 10^9 = 1.1 * 10^10 :=
sorry

end NUMINAMATH_CALUDE_eleven_billion_scientific_notation_l2311_231157


namespace NUMINAMATH_CALUDE_solve_system_l2311_231177

theorem solve_system (x y : ℚ) : 
  (1 / 3 - 1 / 4 = 1 / x) → (x + y = 10) → (x = 12 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2311_231177


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2311_231114

theorem simplify_complex_fraction :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  -(2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2311_231114


namespace NUMINAMATH_CALUDE_parabola_axis_equation_l2311_231137

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + x^2 + x + 3

-- Define the tangent line at x = -1
def tangent_line (x : ℝ) : ℝ := 2*x + 4

-- Define the parabola y = 2px²
def parabola (p : ℝ) (x : ℝ) : ℝ := 2*p*x^2

-- Theorem statement
theorem parabola_axis_equation :
  ∃ (p : ℝ), (∀ (x : ℝ), tangent_line x = parabola p x → x = -1 ∨ x ≠ -1) →
  (∀ (x : ℝ), parabola p x = -(1/4)*x^2) →
  (∀ (x : ℝ), x^2 = -4*1 → x = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_equation_l2311_231137


namespace NUMINAMATH_CALUDE_kolya_mistake_l2311_231119

structure Box where
  blue : ℕ
  green : ℕ

def vasya_correct (b : Box) : Prop := b.blue ≥ 4
def kolya_correct (b : Box) : Prop := b.green ≥ 5
def petya_correct (b : Box) : Prop := b.blue ≥ 3 ∧ b.green ≥ 4
def misha_correct (b : Box) : Prop := b.blue ≥ 4 ∧ b.green ≥ 4

theorem kolya_mistake (b : Box) :
  (vasya_correct b ∧ petya_correct b ∧ misha_correct b ∧ ¬kolya_correct b) ∨
  (vasya_correct b ∧ petya_correct b ∧ misha_correct b ∧ kolya_correct b) :=
by sorry

end NUMINAMATH_CALUDE_kolya_mistake_l2311_231119


namespace NUMINAMATH_CALUDE_businessmen_beverage_theorem_l2311_231178

theorem businessmen_beverage_theorem (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 13)
  (h4 : both = 7) :
  total - (coffee + tea - both) = 9 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_beverage_theorem_l2311_231178


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l2311_231172

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := a^x + 1

theorem tangent_perpendicular_implies_a_value (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : let tangent_slope := (Real.log a)
        let perpendicular_line_slope := -1/2
        tangent_slope * perpendicular_line_slope = -1) :
  a = Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_value_l2311_231172


namespace NUMINAMATH_CALUDE_circle_radius_increase_circle_radius_increase_is_five_over_pi_l2311_231148

/-- Represents the change in radius when a circle's circumference increases from 30 to 40 inches -/
theorem circle_radius_increase : ℝ → Prop :=
  fun Δr =>
    ∃ (r₁ r₂ : ℝ),
      (2 * Real.pi * r₁ = 30) ∧
      (2 * Real.pi * r₂ = 40) ∧
      (r₂ - r₁ = Δr) ∧
      (Δr = 5 / Real.pi)

/-- Proves that the radius increase is 5/π inches -/
theorem circle_radius_increase_is_five_over_pi :
  circle_radius_increase (5 / Real.pi) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_circle_radius_increase_is_five_over_pi_l2311_231148


namespace NUMINAMATH_CALUDE_intersection_of_planes_intersects_at_least_one_line_l2311_231164

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem intersection_of_planes_intersects_at_least_one_line
  (a b l : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : in_plane a α)
  (h3 : in_plane b β)
  (h4 : plane_intersection α β = l) :
  intersects l a ∨ intersects l b :=
sorry

end NUMINAMATH_CALUDE_intersection_of_planes_intersects_at_least_one_line_l2311_231164


namespace NUMINAMATH_CALUDE_odd_function_inequality_l2311_231155

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : IsOdd f) (h_ineq : f a > f b) : f (-a) < f (-b) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l2311_231155


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l2311_231189

theorem fraction_sum_inequality (α β a b : ℝ) (hα : α > 0) (hβ : β > 0)
  (ha : α ≤ a ∧ a ≤ β) (hb : α ≤ b ∧ b ≤ β) :
  b / a + a / b ≤ β / α + α / β ∧
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β) ∨ (a = β ∧ b = α)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l2311_231189


namespace NUMINAMATH_CALUDE_probability_greater_than_two_l2311_231127

/-- A standard die has 6 sides -/
def die_sides : ℕ := 6

/-- The number of outcomes greater than 2 -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a number greater than 2 on a standard six-sided die -/
theorem probability_greater_than_two : 
  (favorable_outcomes : ℚ) / die_sides = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_two_l2311_231127


namespace NUMINAMATH_CALUDE_delta_calculation_l2311_231179

-- Define the operation Δ
def delta (a b : ℝ) : ℝ := a^3 - b^2

-- State the theorem
theorem delta_calculation :
  delta (3^(delta 5 14)) (4^(delta 4 6)) = -4^56 := by
  sorry

end NUMINAMATH_CALUDE_delta_calculation_l2311_231179


namespace NUMINAMATH_CALUDE_intersection_theorem_l2311_231117

def M : Set ℝ := {x | x^2 - 4 > 0}

def N : Set ℝ := {x | (1 - x) / (x - 3) > 0}

theorem intersection_theorem : N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2311_231117


namespace NUMINAMATH_CALUDE_daves_diner_cost_l2311_231107

/-- Represents the pricing and discount structure at Dave's Diner -/
structure DavesDiner where
  burger_price : ℕ
  fries_price : ℕ
  discount_amount : ℕ
  discount_threshold : ℕ

/-- Calculates the total cost of a purchase at Dave's Diner -/
def calculate_total_cost (d : DavesDiner) (num_burgers : ℕ) (num_fries : ℕ) : ℕ :=
  let burger_cost := if num_burgers ≥ d.discount_threshold
    then (d.burger_price - d.discount_amount) * num_burgers
    else d.burger_price * num_burgers
  let fries_cost := d.fries_price * num_fries
  burger_cost + fries_cost

/-- Theorem stating that the total cost of 6 burgers and 5 fries at Dave's Diner is 27 -/
theorem daves_diner_cost : 
  let d : DavesDiner := { 
    burger_price := 4, 
    fries_price := 3, 
    discount_amount := 2, 
    discount_threshold := 4 
  }
  calculate_total_cost d 6 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_daves_diner_cost_l2311_231107


namespace NUMINAMATH_CALUDE_xyz_value_l2311_231149

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : 
  x * y * z = 8 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2311_231149


namespace NUMINAMATH_CALUDE_general_admission_price_is_21_85_l2311_231159

/-- Represents the ticket sales data for a snooker tournament --/
structure TicketSales where
  totalTickets : ℕ
  totalRevenue : ℚ
  vipPrice : ℚ
  vipGenDifference : ℕ

/-- Calculates the price of a general admission ticket --/
def generalAdmissionPrice (sales : TicketSales) : ℚ :=
  let genTickets := (sales.totalTickets + sales.vipGenDifference) / 2
  let vipTickets := sales.totalTickets - genTickets
  (sales.totalRevenue - sales.vipPrice * vipTickets) / genTickets

/-- Theorem stating that the general admission price is $21.85 --/
theorem general_admission_price_is_21_85 (sales : TicketSales) 
  (h1 : sales.totalTickets = 320)
  (h2 : sales.totalRevenue = 7500)
  (h3 : sales.vipPrice = 45)
  (h4 : sales.vipGenDifference = 276) :
  generalAdmissionPrice sales = 21.85 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_price_is_21_85_l2311_231159


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2311_231152

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025 →
  a 3 + a 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2311_231152


namespace NUMINAMATH_CALUDE_book_selling_price_l2311_231167

/-- Proves that the selling price of each book is $1.50 --/
theorem book_selling_price (total_books : ℕ) (records_bought : ℕ) (record_price : ℚ) (money_left : ℚ) :
  total_books = 200 →
  records_bought = 75 →
  record_price = 3 →
  money_left = 75 →
  (total_books : ℚ) * (1.5 : ℚ) = records_bought * record_price + money_left :=
by
  sorry

#check book_selling_price

end NUMINAMATH_CALUDE_book_selling_price_l2311_231167


namespace NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l2311_231100

theorem square_cut_into_three_rectangles :
  ∀ (a b c d e : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + b = 36 ∧ c + d = 36 ∧ a + c = 36 →
  a * e = b * (36 - e) ∧ c * e = d * (36 - e) →
  (∃ x y : ℝ, (x = a ∨ x = b) ∧ (y = c ∨ y = d) ∧ x + y = 36) →
  36 + e = 60 :=
by sorry

end NUMINAMATH_CALUDE_square_cut_into_three_rectangles_l2311_231100


namespace NUMINAMATH_CALUDE_andy_incorrect_answers_l2311_231113

/-- Represents the number of incorrect answers for each person -/
structure TestResults where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The theorem stating that Andy gets 14 questions wrong given the conditions -/
theorem andy_incorrect_answers (results : TestResults) : results.andy = 14 :=
  by
  have h1 : results.andy + results.beth = results.charlie + results.daniel :=
    sorry
  have h2 : results.andy + results.daniel = results.beth + results.charlie + 6 :=
    sorry
  have h3 : results.charlie = 8 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_andy_incorrect_answers_l2311_231113


namespace NUMINAMATH_CALUDE_park_fencing_cost_l2311_231126

/-- Proves that the cost of fencing a rectangular park with given dimensions and fencing cost is 175 rupees -/
theorem park_fencing_cost (length width area perimeter_cost : ℝ) : 
  length / width = 3 / 2 →
  length * width = 3750 →
  perimeter_cost = 0.7 →
  (2 * length + 2 * width) * perimeter_cost = 175 := by
  sorry

#check park_fencing_cost

end NUMINAMATH_CALUDE_park_fencing_cost_l2311_231126


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l2311_231120

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 1| + |x - 2| ≤ a^2 + a + 1)) → 
  -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l2311_231120


namespace NUMINAMATH_CALUDE_quarter_circle_arc_sum_limit_l2311_231185

/-- The limit of the sum of quarter-circle arcs approaches a quarter of the original circle's circumference --/
theorem quarter_circle_arc_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n * (π * (D / n) / 4) - π * D / 4| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_arc_sum_limit_l2311_231185


namespace NUMINAMATH_CALUDE_range_of_a_l2311_231169

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

def B (a : ℝ) : Set ℝ := {x : ℝ | (x - a + 1) * (x - a - 1) ≤ 0}

def p (x : ℝ) : Prop := x ∈ A

def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem range_of_a : 
  {a : ℝ | (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬q a x)} = {a : ℝ | 2 ≤ a ∧ a ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2311_231169


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2311_231145

theorem min_value_and_inequality (x y z : ℝ) (h : x + y + z = 1) :
  ((x - 1)^2 + (y + 1)^2 + (z + 1)^2 ≥ 4/3) ∧
  (∀ a : ℝ, (x - 2)^2 + (y - 1)^2 + (z - a)^2 ≥ 1/3 → a ≤ -3 ∨ a ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2311_231145


namespace NUMINAMATH_CALUDE_li_elevator_journey_l2311_231146

def floor_movements : List Int := [5, -3, 10, -8, 12, -6, -10]
def floor_height : ℝ := 2.8
def electricity_per_meter : ℝ := 0.1

theorem li_elevator_journey :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map (λ x => floor_height * electricity_per_meter * |x|) floor_movements) = 15.12) := by
  sorry

end NUMINAMATH_CALUDE_li_elevator_journey_l2311_231146


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2311_231168

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x < 4} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 7/2} := by sorry

-- Part 2: Range of a when f(x) ≥ 2 for all x
theorem range_of_a :
  (∀ x, f a x ≥ 2) ↔ a ≤ -1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l2311_231168


namespace NUMINAMATH_CALUDE_machine_production_in_10_seconds_l2311_231136

/-- A machine that produces items at a constant rate -/
structure Machine where
  items_per_minute : ℕ

/-- Calculate the number of items produced in a given number of seconds -/
def items_produced (m : Machine) (seconds : ℕ) : ℚ :=
  (m.items_per_minute : ℚ) * (seconds : ℚ) / 60

theorem machine_production_in_10_seconds (m : Machine) 
  (h : m.items_per_minute = 150) : 
  items_produced m 10 = 25 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_in_10_seconds_l2311_231136


namespace NUMINAMATH_CALUDE_sunglasses_profit_ratio_l2311_231151

theorem sunglasses_profit_ratio (selling_price cost_price : ℚ) (pairs_sold : ℕ) (sign_cost : ℚ) :
  selling_price = 30 →
  cost_price = 26 →
  pairs_sold = 10 →
  sign_cost = 20 →
  sign_cost / ((selling_price - cost_price) * pairs_sold) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sunglasses_profit_ratio_l2311_231151
