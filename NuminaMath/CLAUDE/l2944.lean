import Mathlib

namespace inequality_multiplication_l2944_294445

theorem inequality_multiplication (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end inequality_multiplication_l2944_294445


namespace ice_cream_survey_l2944_294422

theorem ice_cream_survey (total_people : ℕ) (ice_cream_angle : ℕ) :
  total_people = 620 →
  ice_cream_angle = 198 →
  ⌊(total_people : ℝ) * (ice_cream_angle : ℝ) / 360⌋ = 341 :=
by
  sorry

end ice_cream_survey_l2944_294422


namespace transform_f1_to_f2_l2944_294442

/-- Represents a quadratic function of the form y = a(x - h)^2 + k -/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies a horizontal and vertical translation to a quadratic function -/
def translate (f : QuadraticFunction) (dx dy : ℝ) : QuadraticFunction :=
  { a := f.a
  , h := f.h - dx
  , k := f.k - dy }

/-- The original quadratic function y = -2(x - 1)^2 + 3 -/
def f1 : QuadraticFunction :=
  { a := -2
  , h := 1
  , k := 3 }

/-- The target quadratic function y = -2x^2 -/
def f2 : QuadraticFunction :=
  { a := -2
  , h := 0
  , k := 0 }

/-- Theorem stating that translating f1 by 1 unit left and 3 units down results in f2 -/
theorem transform_f1_to_f2 : translate f1 1 3 = f2 := by sorry

end transform_f1_to_f2_l2944_294442


namespace probability_two_positive_one_negative_l2944_294454

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

def first_10_terms (a₁ d : ℚ) : List ℚ :=
  List.map (arithmetic_sequence a₁ d) (List.range 10)

theorem probability_two_positive_one_negative
  (a₁ d : ℚ)
  (h₁ : arithmetic_sequence a₁ d 4 = 2)
  (h₂ : arithmetic_sequence a₁ d 7 = -4)
  : (((first_10_terms a₁ d).filter (λ x => x > 0)).length : ℚ) / 10 * 
    (((first_10_terms a₁ d).filter (λ x => x > 0)).length : ℚ) / 10 * 
    (((first_10_terms a₁ d).filter (λ x => x < 0)).length : ℚ) / 10 * 3 = 6 / 25 :=
by sorry

end probability_two_positive_one_negative_l2944_294454


namespace difference_of_cubes_l2944_294456

theorem difference_of_cubes (y : ℝ) : 
  512 * y^3 - 27 = (8*y - 3) * (64*y^2 + 24*y + 9) ∧ 
  (8 + (-3) + 64 + 24 + 9 = 102) := by
  sorry

end difference_of_cubes_l2944_294456


namespace f_range_and_triangle_property_l2944_294425

noncomputable def f (x : Real) : Real :=
  2 * Real.sqrt 3 * Real.sin x * Real.cos x - 3 * Real.sin x ^ 2 - Real.cos x ^ 2 + 3

theorem f_range_and_triangle_property :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc 0 3) ∧
  (∀ (a b c : Real) (A B C : Real),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    A > 0 ∧ A < Real.pi ∧
    B > 0 ∧ B < Real.pi ∧
    C > 0 ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    b / a = Real.sqrt 3 ∧
    Real.sin (2 * A + C) / Real.sin A = 2 + 2 * Real.cos (A + C) →
    f B = 2) := by
  sorry

end f_range_and_triangle_property_l2944_294425


namespace complex_fraction_simplification_l2944_294443

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 3 * i) / (4 + 5 * i + 3 * i^2) = (-1/2 : ℂ) - (1/2 : ℂ) * i :=
by
  -- The proof would go here, but we're skipping it as per instructions
  sorry

end complex_fraction_simplification_l2944_294443


namespace reciprocal_sum_l2944_294406

theorem reciprocal_sum : (1 / (1/4 + 1/5) : ℚ) = 20/9 := by
  sorry

end reciprocal_sum_l2944_294406


namespace diophantine_equation_solutions_l2944_294414

theorem diophantine_equation_solutions : 
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ S ↔ 
      (0 < x ∧ 0 < y ∧ x < y ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 2007)) ∧
    S.card = 7 :=
by sorry

end diophantine_equation_solutions_l2944_294414


namespace wall_width_calculation_l2944_294479

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the area of the wall,
    the side length of the mirror is 34 inches, and the length of the wall is 42.81481481481482 inches,
    then the width of the wall is 54 inches. -/
theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) (wall_width : ℝ) :
  mirror_side = 34 →
  wall_length = 42.81481481481482 →
  mirror_side ^ 2 = (wall_length * wall_width) / 2 →
  wall_width = 54 := by
  sorry

end wall_width_calculation_l2944_294479


namespace chimpanzee_arrangements_l2944_294466

theorem chimpanzee_arrangements : 
  let word := "chimpanzee"
  let total_letters := word.length
  let unique_letters := word.toList.eraseDups.length
  let repeat_letter := 'e'
  let repeat_count := word.toList.filter (· == repeat_letter) |>.length
  (total_letters.factorial / repeat_count.factorial : ℕ) = 1814400 := by
  sorry

end chimpanzee_arrangements_l2944_294466


namespace least_n_satisfying_inequality_l2944_294434

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 → k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 12) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 12) := by
  sorry

end least_n_satisfying_inequality_l2944_294434


namespace isabel_cupcakes_l2944_294421

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 21

/-- The number of packages Isabel could make after Todd ate some cupcakes -/
def packages : ℕ := 6

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 3

/-- The initial number of cupcakes Isabel baked -/
def initial_cupcakes : ℕ := todd_ate + packages * cupcakes_per_package

theorem isabel_cupcakes : initial_cupcakes = 39 := by
  sorry

end isabel_cupcakes_l2944_294421


namespace pizza_toppings_combinations_l2944_294409

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l2944_294409


namespace arithmetic_seq_ratio_theorem_l2944_294469

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_seq_ratio_theorem (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n a n / sum_n b n = (2 * n + 1) / (3 * n + 2)) →
  (a.a 2 + a.a 5 + a.a 17 + a.a 22) / (b.a 8 + b.a 10 + b.a 12 + b.a 16) = 45 / 68 := by
  sorry

end arithmetic_seq_ratio_theorem_l2944_294469


namespace max_sum_of_factors_24_l2944_294493

theorem max_sum_of_factors_24 :
  ∃ (a b : ℕ), a * b = 24 ∧ a + b = 25 ∧
  ∀ (x y : ℕ), x * y = 24 → x + y ≤ 25 := by
  sorry

end max_sum_of_factors_24_l2944_294493


namespace cosine_sum_eleven_l2944_294497

theorem cosine_sum_eleven : 
  Real.cos (π / 11) - Real.cos (2 * π / 11) + Real.cos (3 * π / 11) - 
  Real.cos (4 * π / 11) + Real.cos (5 * π / 11) = 1 / 2 := by
  sorry

end cosine_sum_eleven_l2944_294497


namespace domino_covering_implies_divisibility_by_three_l2944_294447

/-- Represents a domino covering of a square grid -/
structure Covering (n : ℕ) where
  red : Fin (2*n) → Fin (2*n) → Bool
  blue : Fin (2*n) → Fin (2*n) → Bool

/-- Checks if a covering is valid -/
def is_valid_covering (n : ℕ) (c : Covering n) : Prop :=
  ∀ i j, ∃! k l, (c.red i j ∧ c.red k l) ∨ (c.blue i j ∧ c.blue k l)

/-- Represents an integer assignment to each square -/
def Assignment (n : ℕ) := Fin (2*n) → Fin (2*n) → ℤ

/-- Checks if an assignment satisfies the neighbor difference condition -/
def satisfies_difference_condition (n : ℕ) (c : Covering n) (a : Assignment n) : Prop :=
  ∀ i j, ∃ k₁ l₁ k₂ l₂, 
    (c.red i j ∧ c.red k₁ l₁ ∧ c.blue i j ∧ c.blue k₂ l₂) →
    (a i j ≠ 0 ∧ a i j = a k₁ l₁ - a k₂ l₂)

theorem domino_covering_implies_divisibility_by_three (n : ℕ) 
  (h₁ : n > 0)
  (c : Covering n)
  (h₂ : is_valid_covering n c)
  (a : Assignment n)
  (h₃ : satisfies_difference_condition n c a) :
  3 ∣ n :=
sorry

end domino_covering_implies_divisibility_by_three_l2944_294447


namespace problem_statement_l2944_294472

theorem problem_statement (x : ℝ) (h : x * (x + 3) = 154) : (x + 1) * (x + 2) = 156 := by
  sorry

end problem_statement_l2944_294472


namespace simplify_expression_l2944_294495

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end simplify_expression_l2944_294495


namespace fraction_of_x_l2944_294489

theorem fraction_of_x (x y : ℝ) (k : ℝ) 
  (h1 : 5 * x = 3 * y) 
  (h2 : x * y ≠ 0) 
  (h3 : k * x / (1/6 * y) = 0.7200000000000001) : 
  k = 0.04 := by
  sorry

end fraction_of_x_l2944_294489


namespace quadratic_roots_sum_l2944_294416

/-- Given a quadratic equation ax^2 + bx + c = 0 with two real roots,
    s1 is the sum of the roots,
    s2 is the sum of the squares of the roots,
    s3 is the sum of the cubes of the roots.
    This theorem proves that as3 + bs2 + cs1 = 0. -/
theorem quadratic_roots_sum (a b c : ℝ) (s1 s2 s3 : ℝ) 
    (h1 : a ≠ 0)
    (h2 : b^2 - 4*a*c > 0)
    (h3 : s1 = -b/a)
    (h4 : s2 = b^2/a^2 - 2*c/a)
    (h5 : s3 = -b/a * (b^2/a^2 - 3*c/a)) :
  a * s3 + b * s2 + c * s1 = 0 := by
  sorry


end quadratic_roots_sum_l2944_294416


namespace sale_discount_proof_l2944_294444

theorem sale_discount_proof (original_price : ℝ) (sale_price : ℝ) (final_price : ℝ) :
  sale_price = 0.5 * original_price →
  final_price = 0.7 * sale_price →
  final_price = 0.35 * original_price ∧ 
  (1 - final_price / original_price) * 100 = 65 :=
by
  sorry

end sale_discount_proof_l2944_294444


namespace lcm_18_24_l2944_294405

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l2944_294405


namespace A_intersect_B_l2944_294402

-- Define set A
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Theorem stating the intersection of A and B
theorem A_intersect_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end A_intersect_B_l2944_294402


namespace arithmetic_sequence_property_l2944_294452

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The given condition for the sequence -/
def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 2 + 2 * a 8 + a 14 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sequence_condition a) : 
  2 * a 9 - a 10 = 30 := by
  sorry

end arithmetic_sequence_property_l2944_294452


namespace jake_planting_charge_l2944_294438

/-- The hourly rate Jake wants to make -/
def desired_hourly_rate : ℝ := 20

/-- The time it takes to plant flowers in hours -/
def planting_time : ℝ := 2

/-- The amount Jake should charge for planting flowers -/
def planting_charge : ℝ := desired_hourly_rate * planting_time

theorem jake_planting_charge : planting_charge = 40 := by
  sorry

end jake_planting_charge_l2944_294438


namespace curve_slope_implies_a_range_l2944_294429

/-- The curve y = ln x + ax^2 has no tangent lines with negative slopes for all x > 0 -/
def no_negative_slopes (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1 / x + 2 * a * x) ≥ 0

/-- The theorem states that if the curve has no tangent lines with negative slopes,
    then a is in the range [0, +∞) -/
theorem curve_slope_implies_a_range (a : ℝ) :
  no_negative_slopes a → a ∈ Set.Ici (0 : ℝ) :=
sorry

end curve_slope_implies_a_range_l2944_294429


namespace order_of_a_ab2_ab_l2944_294494

theorem order_of_a_ab2_ab (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end order_of_a_ab2_ab_l2944_294494


namespace distance_from_point_to_line_l2944_294449

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (4, 6, 5)
def line_direction : ℝ × ℝ × ℝ := (1, 3, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  distance_to_line point line_point line_direction = Real.sqrt 62 / 3 := by
  sorry

end distance_from_point_to_line_l2944_294449


namespace player_positions_satisfy_distances_l2944_294403

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
| 0 => 0
| 1 => 1
| 2 => 4
| 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end player_positions_satisfy_distances_l2944_294403


namespace tyrone_total_money_l2944_294462

-- Define the currency values
def one_dollar : ℚ := 1
def ten_dollar : ℚ := 10
def five_dollar : ℚ := 5
def quarter : ℚ := 0.25
def half_dollar : ℚ := 0.5
def dime : ℚ := 0.1
def nickel : ℚ := 0.05
def penny : ℚ := 0.01
def two_dollar : ℚ := 2
def fifty_cent : ℚ := 0.5

-- Define Tyrone's currency counts
def one_dollar_bills : ℕ := 3
def ten_dollar_bills : ℕ := 1
def five_dollar_bills : ℕ := 2
def quarters : ℕ := 26
def half_dollar_coins : ℕ := 5
def dimes : ℕ := 45
def nickels : ℕ := 8
def one_dollar_coins : ℕ := 3
def pennies : ℕ := 56
def two_dollar_bills : ℕ := 2
def fifty_cent_coins : ℕ := 4

-- Define the total amount function
def total_amount : ℚ :=
  (one_dollar_bills : ℚ) * one_dollar +
  (ten_dollar_bills : ℚ) * ten_dollar +
  (five_dollar_bills : ℚ) * five_dollar +
  (quarters : ℚ) * quarter +
  (half_dollar_coins : ℚ) * half_dollar +
  (dimes : ℚ) * dime +
  (nickels : ℚ) * nickel +
  (one_dollar_coins : ℚ) * one_dollar +
  (pennies : ℚ) * penny +
  (two_dollar_bills : ℚ) * two_dollar +
  (fifty_cent_coins : ℚ) * fifty_cent

-- Theorem stating that the total amount is $46.46
theorem tyrone_total_money : total_amount = 46.46 := by
  sorry

end tyrone_total_money_l2944_294462


namespace algae_coverage_day_l2944_294465

/-- Represents the coverage of algae in the lake on a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(30 - day)

/-- The problem statement -/
theorem algae_coverage_day : ∃ d : ℕ, d ≤ 30 ∧ algaeCoverage d < (1/10) ∧ (1/10) ≤ algaeCoverage (d+1) :=
  sorry

end algae_coverage_day_l2944_294465


namespace lucy_aquarium_cleaning_l2944_294448

/-- The number of aquariums Lucy can clean in a given time period. -/
def aquariums_cleaned (aquariums_per_period : ℚ) (hours : ℚ) : ℚ :=
  aquariums_per_period * hours

/-- Theorem stating how many aquariums Lucy can clean in 24 hours. -/
theorem lucy_aquarium_cleaning :
  let aquariums_per_3hours : ℚ := 2
  let cleaning_period : ℚ := 3
  let working_hours : ℚ := 24
  aquariums_cleaned (aquariums_per_3hours / cleaning_period) working_hours = 16 := by
  sorry

end lucy_aquarium_cleaning_l2944_294448


namespace simplify_expression_l2944_294404

theorem simplify_expression : 5 * (18 / 7) * (49 / -54) = -(245 / 9) := by
  sorry

end simplify_expression_l2944_294404


namespace smallest_multiple_of_30_and_40_not_16_l2944_294431

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_30_and_40_not_16 : 
  (∀ n : ℕ, n > 0 ∧ is_multiple n 30 ∧ is_multiple n 40 ∧ ¬is_multiple n 16 → n ≥ 120) ∧ 
  (is_multiple 120 30 ∧ is_multiple 120 40 ∧ ¬is_multiple 120 16) :=
sorry

end smallest_multiple_of_30_and_40_not_16_l2944_294431


namespace fraction_3x_3x_minus_2_simplest_form_l2944_294474

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1 -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- The fraction 3x/(3x-2) -/
def f (x : ℤ) : ℚ := (3 * x) / (3 * x - 2)

/-- Theorem: The fraction 3x/(3x-2) is in its simplest form -/
theorem fraction_3x_3x_minus_2_simplest_form (x : ℤ) :
  IsSimplestForm (3 * x) (3 * x - 2) :=
sorry

end fraction_3x_3x_minus_2_simplest_form_l2944_294474


namespace initial_girls_count_l2944_294413

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 20) = b) →
  (7 * (b - 54) = g - 20) →
  g = 39 := by
sorry

end initial_girls_count_l2944_294413


namespace arithmetic_problem_l2944_294486

theorem arithmetic_problem : (36 / (8 + 2 - 3)) * 7 = 36 := by
  sorry

end arithmetic_problem_l2944_294486


namespace conditional_probability_rain_given_east_wind_l2944_294407

def east_wind_prob : ℚ := 9/30
def rain_prob : ℚ := 11/30
def both_prob : ℚ := 8/30

theorem conditional_probability_rain_given_east_wind :
  (both_prob / east_wind_prob : ℚ) = 8/9 := by sorry

end conditional_probability_rain_given_east_wind_l2944_294407


namespace triangle_line_equations_l2944_294451

/-- Triangle ABC with vertices A(-1, 5), B(-2, -1), and C(4, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def triangle_ABC : Triangle :=
  { A := (-1, 5)
    B := (-2, -1)
    C := (4, 3) }

/-- The equation of the line on which side AB lies -/
def line_AB : LineEquation :=
  { a := 6
    b := -1
    c := 11 }

/-- The equation of the line on which the altitude from C to AB lies -/
def altitude_C : LineEquation :=
  { a := 1
    b := 6
    c := -22 }

theorem triangle_line_equations (t : Triangle) (lab : LineEquation) (lc : LineEquation) :
  t = triangle_ABC →
  lab = line_AB →
  lc = altitude_C →
  (∀ x y : ℝ, lab.a * x + lab.b * y + lab.c = 0 ↔ (x, y) ∈ Set.Icc t.A t.B) ∧
  (∀ x y : ℝ, lc.a * x + lc.b * y + lc.c = 0 ↔ 
    (x - t.C.1) * lab.a + (y - t.C.2) * lab.b = 0) :=
sorry

end triangle_line_equations_l2944_294451


namespace canMeasureFourLiters_l2944_294400

/-- Represents a container with a certain capacity -/
structure Container where
  capacity : ℕ
  current : ℕ
  h : current ≤ capacity

/-- Represents the state of the water measuring system -/
structure WaterSystem where
  small : Container
  large : Container

/-- Checks if the given state has 4 liters in the large container -/
def hasFourLiters (state : WaterSystem) : Prop :=
  state.large.current = 4

/-- Defines the possible operations on the water system -/
inductive Operation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies an operation to the water system -/
def applyOperation (op : Operation) (state : WaterSystem) : WaterSystem :=
  sorry

/-- Theorem stating that it's possible to measure 4 liters -/
theorem canMeasureFourLiters :
  ∃ (ops : List Operation),
    let initialState : WaterSystem := {
      small := { capacity := 3, current := 0, h := by simp },
      large := { capacity := 5, current := 0, h := by simp }
    }
    let finalState := ops.foldl (fun state op => applyOperation op state) initialState
    hasFourLiters finalState :=
  sorry

end canMeasureFourLiters_l2944_294400


namespace union_of_sets_l2944_294411

theorem union_of_sets (A B : Set ℕ) (h1 : A = {0, 1}) (h2 : B = {2}) :
  A ∪ B = {0, 1, 2} := by
  sorry

end union_of_sets_l2944_294411


namespace grassy_area_length_l2944_294467

/-- The length of the grassy area in a rectangular plot with a gravel path -/
theorem grassy_area_length 
  (total_length : ℝ) 
  (path_width : ℝ) 
  (h1 : total_length = 110) 
  (h2 : path_width = 2.5) : 
  total_length - 2 * path_width = 105 := by
sorry

end grassy_area_length_l2944_294467


namespace solution_set_f_range_of_a_l2944_294423

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f (x : ℝ) : 
  f x ≤ 5 ↔ -4/3 ≤ x ∧ x ≤ 0 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |a^2 - 3*a| ≤ f x) ↔ -1 ≤ a ∧ a ≤ 4 := by sorry

end solution_set_f_range_of_a_l2944_294423


namespace vasya_no_purchase_days_vasya_no_purchase_days_proof_l2944_294433

theorem vasya_no_purchase_days : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun x y z w =>
    x + y + z + w = 15 ∧
    9 * x + 4 * z = 30 ∧
    2 * y + z = 9 →
    w = 7

-- The proof is omitted
theorem vasya_no_purchase_days_proof : vasya_no_purchase_days 2 3 3 7 := by
  sorry

end vasya_no_purchase_days_vasya_no_purchase_days_proof_l2944_294433


namespace three_to_negative_x_is_exponential_l2944_294461

/-- Definition of an exponential function -/
def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

/-- The function y = 3^(-x) is an exponential function -/
theorem three_to_negative_x_is_exponential :
  is_exponential_function (fun x => 3^(-x)) :=
sorry

end three_to_negative_x_is_exponential_l2944_294461


namespace hyperbola_equation_l2944_294412

theorem hyperbola_equation (a b : ℝ) (h1 : a = 6) (h2 : b = Real.sqrt 35) :
  ∀ x y : ℝ, (y^2 / 36 - x^2 / 35 = 1) ↔ 
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ 
   ∀ F₁ F₂ : ℝ × ℝ, F₁ = (0, c) ∧ F₂ = (0, -c) → 
   (y - F₁.2)^2 + x^2 - (y - F₂.2)^2 - x^2 = 4 * a^2) :=
by sorry

end hyperbola_equation_l2944_294412


namespace sallys_nickels_l2944_294492

theorem sallys_nickels (initial_nickels dad_nickels total_nickels : ℕ) 
  (h1 : initial_nickels = 7)
  (h2 : dad_nickels = 9)
  (h3 : total_nickels = 18) :
  total_nickels - (initial_nickels + dad_nickels) = 2 := by
  sorry

end sallys_nickels_l2944_294492


namespace lucy_grocery_shopping_l2944_294483

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The number of cans of soup Lucy bought -/
def soup : ℕ := 28

/-- The number of boxes of cereals Lucy bought -/
def cereals : ℕ := 5

/-- The number of packs of crackers Lucy bought -/
def crackers : ℕ := 45

/-- The total number of packs and boxes Lucy bought -/
def total_packs_and_boxes : ℕ := cookies + noodles + cereals + crackers

theorem lucy_grocery_shopping :
  total_packs_and_boxes = 78 := by sorry

end lucy_grocery_shopping_l2944_294483


namespace select_three_from_five_l2944_294401

theorem select_three_from_five (n : ℕ) (h : n = 5) : 
  n * (n - 1) * (n - 2) = 60 := by
  sorry

end select_three_from_five_l2944_294401


namespace root_equation_result_l2944_294491

theorem root_equation_result (a b m p : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ∃ r, ((a + 1/b)^2 - p*(a + 1/b) + r = 0) ∧ 
       ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
       r = 16/3 := by
sorry

end root_equation_result_l2944_294491


namespace complex_simplification_l2944_294464

theorem complex_simplification : (1 - Complex.I)^2 + 4 * Complex.I = 2 + 2 * Complex.I := by
  sorry

end complex_simplification_l2944_294464


namespace marble_problem_l2944_294488

/-- The number of marbles in the jar after adjustment -/
def final_marbles : ℕ := 195

/-- Proves that the final number of marbles is 195 given the conditions -/
theorem marble_problem (ben : ℕ) (leo : ℕ) (tim : ℕ) 
  (h1 : ben = 56)
  (h2 : leo = ben + 20)
  (h3 : tim = leo - 15)
  (h4 : ∃ k : ℤ, -5 ≤ k ∧ k ≤ 5 ∧ (ben + leo + tim + k) % 5 = 0) :
  final_marbles = ben + leo + tim + 2 :=
sorry

end marble_problem_l2944_294488


namespace complement_M_intersect_N_l2944_294426

open Set

-- Define the universal set U as the set of integers
def U : Set ℤ := univ

-- Define set M
def M : Set ℤ := {-1, 0, 1}

-- Define set N
def N : Set ℤ := {0, 1, 3}

-- State the theorem
theorem complement_M_intersect_N :
  (U \ M) ∩ N = {3} := by sorry

end complement_M_intersect_N_l2944_294426


namespace right_triangle_a_value_l2944_294408

/-- Proves that for a right triangle with given properties, the value of a is 14 -/
theorem right_triangle_a_value (a b : ℝ) : 
  a > 0 → -- a is positive
  b = 4 → -- b equals 4
  (1/2) * a * b = 28 → -- area of the triangle is 28
  a = 14 := by
sorry

end right_triangle_a_value_l2944_294408


namespace consistent_walnuts_dont_determine_oranges_l2944_294481

/-- Represents the state of trees in a park -/
structure ParkTrees where
  initial_walnuts : ℕ
  cut_walnuts : ℕ
  final_walnuts : ℕ

/-- Checks if the walnut tree information is consistent -/
def consistent_walnuts (park : ParkTrees) : Prop :=
  park.initial_walnuts - park.cut_walnuts = park.final_walnuts

/-- States that the number of orange trees cannot be determined -/
def orange_trees_undetermined (park : ParkTrees) : Prop :=
  ∀ n : ℕ, ∃ park' : ParkTrees, park'.initial_walnuts = park.initial_walnuts ∧
                                park'.cut_walnuts = park.cut_walnuts ∧
                                park'.final_walnuts = park.final_walnuts ∧
                                n ≠ 0  -- Assuming there's at least one orange tree

/-- Theorem stating that consistent walnut information doesn't determine orange tree count -/
theorem consistent_walnuts_dont_determine_oranges (park : ParkTrees) :
  consistent_walnuts park → orange_trees_undetermined park :=
by
  sorry

#check consistent_walnuts_dont_determine_oranges

end consistent_walnuts_dont_determine_oranges_l2944_294481


namespace arithmetic_mean_of_range_l2944_294441

def integer_range : List ℤ := List.range 12 |>.map (λ i => i - 5)

theorem arithmetic_mean_of_range : (integer_range.sum : ℚ) / integer_range.length = 1/2 := by
  sorry

end arithmetic_mean_of_range_l2944_294441


namespace coefficient_x_squared_is_70_l2944_294437

/-- The coefficient of x^2 in the expansion of (2+x)(1-2x)^5 -/
def coefficient_x_squared : ℤ :=
  2 * (Nat.choose 5 2) * (-2)^2 + (Nat.choose 5 1) * (-2)

/-- Theorem stating that the coefficient of x^2 in the expansion of (2+x)(1-2x)^5 is 70 -/
theorem coefficient_x_squared_is_70 : coefficient_x_squared = 70 := by
  sorry

end coefficient_x_squared_is_70_l2944_294437


namespace fraction_equality_l2944_294468

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : s / u = 7 / 8) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -13 / 10 := by
  sorry

end fraction_equality_l2944_294468


namespace unique_solution_system_l2944_294477

theorem unique_solution_system :
  ∃! (x y : ℚ), (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
                 (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
                 x = 12 / 5 ∧ y = 12 / 25 := by
  sorry

end unique_solution_system_l2944_294477


namespace functional_equation_solution_l2944_294470

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, |x| * (f y) + y * (f x) = f (x * y) + f (x^2) + f (f y)) →
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * (|x| - x) :=
by sorry

end functional_equation_solution_l2944_294470


namespace negation_of_absolute_value_less_than_zero_is_true_l2944_294458

theorem negation_of_absolute_value_less_than_zero_is_true : 
  ¬(∃ x : ℝ, |x - 1| < 0) := by
  sorry

end negation_of_absolute_value_less_than_zero_is_true_l2944_294458


namespace bus_children_difference_l2944_294482

/-- Proves that the difference in children on the bus before and after a stop is 23 -/
theorem bus_children_difference (initial_count : Nat) (final_count : Nat)
    (h1 : initial_count = 41)
    (h2 : final_count = 18) :
    initial_count - final_count = 23 := by
  sorry

end bus_children_difference_l2944_294482


namespace vertex_of_our_parabola_l2944_294490

/-- A parabola is defined by the equation y = (x - h)^2 + k, where (h, k) is its vertex -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The parabola y = (x - 2)^2 + 1 -/
def our_parabola : Parabola := { h := 2, k := 1 }

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

theorem vertex_of_our_parabola :
  vertex our_parabola = (2, 1) := by sorry

end vertex_of_our_parabola_l2944_294490


namespace goldfish_equality_l2944_294476

theorem goldfish_equality (n : ℕ) : (∀ k : ℕ, k < n → 4^(k+1) ≠ 128 * 2^k) ∧ 4^(n+1) = 128 * 2^n ↔ n = 5 := by
  sorry

end goldfish_equality_l2944_294476


namespace committee_probability_l2944_294487

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_ways := Nat.choose total_members committee_size
  let all_boys := Nat.choose boys committee_size
  let all_girls := Nat.choose girls committee_size
  let favorable_ways := total_ways - (all_boys + all_girls)
  (favorable_ways : ℚ) / total_ways = 574287 / 593775 := by sorry

end committee_probability_l2944_294487


namespace cos_45_sin_30_product_equation_equivalence_l2944_294418

-- Problem 1
theorem cos_45_sin_30_product : 4 * Real.cos (π / 4) * Real.sin (π / 6) = Real.sqrt 2 := by
  sorry

-- Problem 2
theorem equation_equivalence (x : ℝ) : (x + 2) * (x - 3) = 2 * x - 6 ↔ x^2 - 3 * x = 0 := by
  sorry

end cos_45_sin_30_product_equation_equivalence_l2944_294418


namespace similarity_ratio_bounds_l2944_294480

theorem similarity_ratio_bounds (x y z p : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_p : 0 < p)
  (h_similar : y = x * (z / y) ∧ z = y * (p / z)) :
  let k := z / y
  let φ := (1 + Real.sqrt 5) / 2
  φ⁻¹ < k ∧ k < φ := by
  sorry

end similarity_ratio_bounds_l2944_294480


namespace lorelai_ate_180_jellybeans_l2944_294417

-- Define the number of jellybeans each person has
def gigi_jellybeans : ℕ := 15
def rory_jellybeans : ℕ := gigi_jellybeans + 30

-- Define the total number of jellybeans both girls have
def total_girls_jellybeans : ℕ := gigi_jellybeans + rory_jellybeans

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : ℕ := 3 * total_girls_jellybeans

-- Theorem to prove
theorem lorelai_ate_180_jellybeans : lorelai_jellybeans = 180 := by
  sorry

end lorelai_ate_180_jellybeans_l2944_294417


namespace cone_lateral_surface_area_l2944_294499

/-- Given a cone with vertex S, prove that its lateral surface area is 40√2π -/
theorem cone_lateral_surface_area (S : Point) (A B : Point) :
  let cos_angle_SA_SB : ℝ := 7/8
  let angle_SA_base : ℝ := π/4  -- 45° in radians
  let area_SAB : ℝ := 5 * Real.sqrt 15
  -- Define the lateral surface area
  let lateral_surface_area : ℝ := 
    let SA : ℝ := 4 * Real.sqrt 5  -- derived from area_SAB and cos_angle_SA_SB
    let base_radius : ℝ := SA * Real.sqrt 2 / 2
    π * base_radius * SA
  lateral_surface_area = 40 * Real.sqrt 2 * π := by
  sorry

end cone_lateral_surface_area_l2944_294499


namespace chocolate_chip_cookie_price_l2944_294453

/-- The price of a box of chocolate chip cookies given the following conditions:
  * Total boxes sold: 1,585
  * Combined value of all boxes: $1,586.75
  * Plain cookies price: $0.75 each
  * Number of plain cookie boxes sold: 793.375
-/
theorem chocolate_chip_cookie_price :
  let total_boxes : ℝ := 1585
  let total_value : ℝ := 1586.75
  let plain_cookie_price : ℝ := 0.75
  let plain_cookie_boxes : ℝ := 793.375
  let chocolate_chip_boxes : ℝ := total_boxes - plain_cookie_boxes
  let chocolate_chip_price : ℝ := (total_value - (plain_cookie_price * plain_cookie_boxes)) / chocolate_chip_boxes
  chocolate_chip_price = 1.2525 := by
  sorry

end chocolate_chip_cookie_price_l2944_294453


namespace combined_shape_is_pentahedron_l2944_294478

/-- A regular square pyramid -/
structure RegularSquarePyramid :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (edge_length_pos : edge_length > 0)

/-- The result of combining a regular square pyramid and a regular tetrahedron -/
def CombinedShape (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) :=
  pyramid.edge_length = tetrahedron.edge_length

/-- The number of faces in the resulting shape -/
def num_faces (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) 
  (h : CombinedShape pyramid tetrahedron) : ℕ := 5

theorem combined_shape_is_pentahedron 
  (pyramid : RegularSquarePyramid) (tetrahedron : RegularTetrahedron) 
  (h : CombinedShape pyramid tetrahedron) : 
  num_faces pyramid tetrahedron h = 5 := by sorry

end combined_shape_is_pentahedron_l2944_294478


namespace chocolate_bars_per_box_l2944_294446

theorem chocolate_bars_per_box 
  (total_bars : ℕ) 
  (total_boxes : ℕ) 
  (h1 : total_bars = 710) 
  (h2 : total_boxes = 142) : 
  total_bars / total_boxes = 5 := by
sorry

end chocolate_bars_per_box_l2944_294446


namespace carpet_area_proof_l2944_294435

/-- Calculates the total carpet area required for three rooms -/
def totalCarpetArea (w1 l1 w2 l2 w3 l3 : ℝ) : ℝ :=
  w1 * l1 + w2 * l2 + w3 * l3

/-- Proves that the total carpet area for the given room dimensions is 353 square feet -/
theorem carpet_area_proof :
  totalCarpetArea 12 15 7 9 10 11 = 353 := by
  sorry

#eval totalCarpetArea 12 15 7 9 10 11

end carpet_area_proof_l2944_294435


namespace expression_simplification_l2944_294471

theorem expression_simplification (a : ℝ) (h1 : a^2 - 4 = 0) (h2 : a ≠ -2) :
  (((a^2 + 1) / a - 2) / ((a + 2) * (a - 1) / (a^2 + 2*a))) = 1 := by
  sorry

end expression_simplification_l2944_294471


namespace fixed_distance_to_H_l2944_294432

/-- Given a parabola y^2 = 4x with origin O and moving points A and B on the parabola,
    such that OA ⊥ OB, and OH ⊥ AB where H is the foot of the perpendicular,
    prove that the point (2,0) has a fixed distance to H. -/
theorem fixed_distance_to_H (A B H : ℝ × ℝ) : 
  (∀ (y₁ y₂ : ℝ), A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1) →  -- A and B on parabola
  (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
  (∃ (m n : ℝ), H.1 = m * H.2 + n ∧ 
    A.1 = m * A.2 + n ∧ B.1 = m * B.2 + n) →  -- H on line AB
  (H.1 * 0 + H.2 * 1 = 0) →  -- OH ⊥ AB
  ∃ (r : ℝ), (H.1 - 2)^2 + H.2^2 = r^2 := by
    sorry

end fixed_distance_to_H_l2944_294432


namespace range_of_a_l2944_294439

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := (a - 1) * x > 2
def solution_set (a x : ℝ) : Prop := x < 2 / (a - 1)

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) :
  (∀ x, inequality a x ↔ solution_set a x) → a < 1 := by
  sorry

end range_of_a_l2944_294439


namespace det_invariant_under_row_operation_l2944_294420

/-- Given a 2x2 matrix with determinant 7, prove that modifying the first row
    by adding twice the second row doesn't change the determinant. -/
theorem det_invariant_under_row_operation {a b c d : ℝ} 
  (h : a * d - b * c = 7) :
  (a + 2*c) * d - (b + 2*d) * c = 7 := by
  sorry

#check det_invariant_under_row_operation

end det_invariant_under_row_operation_l2944_294420


namespace quadratic_unique_solution_l2944_294424

theorem quadratic_unique_solution (a : ℚ) :
  (∃! x : ℚ, 2 * a * x^2 + 15 * x + 9 = 0) →
  (a = 25/8 ∧ ∃! x : ℚ, x = -12/5 ∧ 2 * a * x^2 + 15 * x + 9 = 0) :=
by sorry

end quadratic_unique_solution_l2944_294424


namespace garden_area_l2944_294427

/-- The area of a rectangular garden given specific walking conditions -/
theorem garden_area (length width : ℝ) : 
  length * 30 = 1500 →
  2 * (length + width) * 12 = 1500 →
  length * width = 625 := by
  sorry

end garden_area_l2944_294427


namespace height_decreases_as_vertex_angle_increases_l2944_294498

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ  -- Length of equal sides
  φ : ℝ  -- Half of the vertex angle
  h : ℝ  -- Height dropped to the base
  h_eq : h = a * Real.cos φ

-- Theorem statement
theorem height_decreases_as_vertex_angle_increases
  (t1 t2 : IsoscelesTriangle)
  (h_same_side : t1.a = t2.a)
  (h_larger_angle : t1.φ < t2.φ)
  (h_angle_range : 0 < t1.φ ∧ t2.φ < Real.pi / 2) :
  t2.h < t1.h :=
by
  sorry

end height_decreases_as_vertex_angle_increases_l2944_294498


namespace negation_of_or_implies_both_false_l2944_294419

theorem negation_of_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end negation_of_or_implies_both_false_l2944_294419


namespace positive_difference_problem_l2944_294457

theorem positive_difference_problem : 
  ∀ x : ℝ, (33 + x) / 2 = 37 → |x - 33| = 8 := by
sorry

end positive_difference_problem_l2944_294457


namespace toott_permutations_eq_ten_l2944_294460

/-- The number of distinct permutations of the letters in "TOOTT" -/
def toott_permutations : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "TOOTT" is 10 -/
theorem toott_permutations_eq_ten : toott_permutations = 10 := by
  sorry

end toott_permutations_eq_ten_l2944_294460


namespace geometric_sequence_seventh_term_l2944_294459

/-- Given a geometric sequence of positive integers where the first term is 3
    and the fifth term is 243, the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term is 3
  a 5 = 243 →                          -- fifth term is 243
  a 7 = 2187 :=                        -- seventh term is 2187
by sorry

end geometric_sequence_seventh_term_l2944_294459


namespace investment_value_l2944_294463

theorem investment_value (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.15 * 1500 = 0.13 * (x + 1500) →
  x = 500 := by
sorry

end investment_value_l2944_294463


namespace roots_of_polynomial_l2944_294484

def p (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2) :=
by sorry

end roots_of_polynomial_l2944_294484


namespace inequality_problem_l2944_294455

theorem inequality_problem (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 2*x^2 - 9*x + m < 0) → 
  m ≤ 9 := by
  sorry

end inequality_problem_l2944_294455


namespace extreme_value_at_three_increasing_on_negative_l2944_294415

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

theorem extreme_value_at_three (h : ∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 3 → |x - 3| < ε → f a x ≤ f a 3) :
  a = 3 := by sorry

theorem increasing_on_negative (h : ∀ (x y : ℝ), x < y → y < 0 → f a x < f a y) :
  0 ≤ a := by sorry

end extreme_value_at_three_increasing_on_negative_l2944_294415


namespace absolute_value_expression_l2944_294428

theorem absolute_value_expression (x : ℝ) (E : ℝ) :
  x = 10 ∧ 30 - |E| = 26 → E = 4 ∨ E = -4 := by
sorry

end absolute_value_expression_l2944_294428


namespace l_shaped_region_perimeter_l2944_294450

/-- Represents an L-shaped region with a staircase pattern --/
structure LShapedRegion where
  width : ℝ
  height : ℝ
  unit_length : ℝ
  num_steps : ℕ

/-- Calculates the area of the L-shaped region --/
def area (r : LShapedRegion) : ℝ :=
  r.width * r.height - (r.num_steps * r.unit_length^2)

/-- Calculates the perimeter of the L-shaped region --/
def perimeter (r : LShapedRegion) : ℝ :=
  r.width + r.height + r.num_steps * r.unit_length + r.unit_length * (r.num_steps + 1)

/-- Theorem stating that an L-shaped region with specific properties has a perimeter of 39.4 meters --/
theorem l_shaped_region_perimeter :
  ∀ (r : LShapedRegion),
    r.width = 10 ∧
    r.unit_length = 1 ∧
    r.num_steps = 10 ∧
    area r = 72 →
    perimeter r = 39.4 := by
  sorry


end l_shaped_region_perimeter_l2944_294450


namespace math_competition_results_l2944_294475

/-- Represents the number of joint math competitions --/
def num_competitions : ℕ := 5

/-- Represents the probability of ranking in the top 20 in each competition --/
def prob_top20 : ℚ := 1/4

/-- Represents the number of top 20 rankings needed to qualify for provincial training --/
def qualify_threshold : ℕ := 2

/-- Models the outcome of a student's participation in the math competitions --/
structure StudentOutcome where
  num_participated : ℕ
  num_top20 : ℕ
  qualified : Bool

/-- Calculates the probability of a specific outcome --/
noncomputable def prob_outcome (outcome : StudentOutcome) : ℚ :=
  sorry

/-- Calculates the probability of qualifying for provincial training --/
noncomputable def prob_qualify : ℚ :=
  sorry

/-- Calculates the expected number of competitions participated in, given qualification or completion --/
noncomputable def expected_num_competitions : ℚ :=
  sorry

/-- Main theorem stating the probabilities and expected value --/
theorem math_competition_results :
  prob_qualify = 67/256 ∧ expected_num_competitions = 65/16 :=
by sorry

end math_competition_results_l2944_294475


namespace square_area_of_fourth_side_l2944_294473

theorem square_area_of_fourth_side (EF FG GH : ℝ) (h1 : EF^2 = 25) (h2 : FG^2 = 49) (h3 : GH^2 = 64) : 
  ∃ EG EH : ℝ, EG^2 = EF^2 + FG^2 ∧ EH^2 = EG^2 + GH^2 ∧ EH^2 = 138 := by
  sorry

end square_area_of_fourth_side_l2944_294473


namespace system_solvability_l2944_294430

/-- The first equation of the system -/
def equation1 (x y : ℝ) : Prop :=
  (x - 2)^2 + (|y - 1| - 1)^2 = 4

/-- The second equation of the system -/
def equation2 (x y a b : ℝ) : Prop :=
  y = b * |x - 1| + a

/-- The system has a solution for given a and b -/
def has_solution (a b : ℝ) : Prop :=
  ∃ x y, equation1 x y ∧ equation2 x y a b

theorem system_solvability (a : ℝ) :
  (∀ b, has_solution a b) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ 2 + Real.sqrt 3 :=
sorry

end system_solvability_l2944_294430


namespace det_2x2_matrix_l2944_294436

/-- The determinant of a 2x2 matrix [[5, x], [-3, 4]] is 20 + 3x -/
theorem det_2x2_matrix (x : ℝ) : 
  Matrix.det !![5, x; -3, 4] = 20 + 3 * x := by
  sorry

end det_2x2_matrix_l2944_294436


namespace largest_integer_less_than_100_with_remainder_4_mod_7_l2944_294410

theorem largest_integer_less_than_100_with_remainder_4_mod_7 : ∃ n : ℕ, n = 95 ∧ 
  (∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n) ∧ n < 100 ∧ n % 7 = 4 :=
by sorry

end largest_integer_less_than_100_with_remainder_4_mod_7_l2944_294410


namespace compound_interest_problem_l2944_294496

/-- Given a principal amount P, where the simple interest on P for 2 years at 10% per annum is $660,
    prove that the compound interest on P for 2 years at the same rate is $693. -/
theorem compound_interest_problem (P : ℝ) : 
  P * 0.1 * 2 = 660 → P * (1 + 0.1)^2 - P = 693 := by
  sorry

end compound_interest_problem_l2944_294496


namespace intersection_A_complement_B_l2944_294485

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt ((6 / (x + 1)) - 1)}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3) / Real.log 10}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ∉ B}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ complement_B = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

end intersection_A_complement_B_l2944_294485


namespace polynomial_product_sum_l2944_294440

/-- Given two polynomials in d with coefficients g and h, prove their sum equals 15.5 -/
theorem polynomial_product_sum (g h : ℚ) : 
  (∀ d : ℚ, (8*d^2 - 4*d + g) * (5*d^2 + h*d - 10) = 40*d^4 - 75*d^3 - 90*d^2 + 5*d + 20) →
  g + h = 15.5 := by
  sorry

end polynomial_product_sum_l2944_294440
