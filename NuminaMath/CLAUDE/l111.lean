import Mathlib

namespace NUMINAMATH_CALUDE_expand_and_simplify_l111_11100

theorem expand_and_simplify (x y : ℝ) : (-x + y) * (-x - y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l111_11100


namespace NUMINAMATH_CALUDE_field_width_l111_11174

/-- Proves that a rectangular field of length 60 m with a 2.5 m wide path around it,
    having a path area of 1200 sq m, has a width of 175 m. -/
theorem field_width (field_length : ℝ) (path_width : ℝ) (path_area : ℝ) :
  field_length = 60 →
  path_width = 2.5 →
  path_area = 1200 →
  ∃ field_width : ℝ,
    (field_length + 2 * path_width) * (field_width + 2 * path_width) -
    field_length * field_width = path_area ∧
    field_width = 175 := by
  sorry


end NUMINAMATH_CALUDE_field_width_l111_11174


namespace NUMINAMATH_CALUDE_canada_human_beaver_ratio_l111_11196

/-- The ratio of humans to beavers in Canada -/
def human_beaver_ratio (moose_population : ℕ) (human_population : ℕ) : ℚ :=
  human_population / (2 * moose_population)

/-- Theorem stating the ratio of humans to beavers in Canada -/
theorem canada_human_beaver_ratio :
  human_beaver_ratio 1000000 38000000 = 19 := by
  sorry

end NUMINAMATH_CALUDE_canada_human_beaver_ratio_l111_11196


namespace NUMINAMATH_CALUDE_solution_is_eight_l111_11131

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  lg (2^x + 2*x - 16) = x * (1 - lg 5)

-- Theorem statement
theorem solution_is_eight : 
  ∃ (x : ℝ), equation x ∧ x = 8 :=
sorry

end NUMINAMATH_CALUDE_solution_is_eight_l111_11131


namespace NUMINAMATH_CALUDE_product_no_x3_x2_terms_l111_11165

theorem product_no_x3_x2_terms (p q : ℝ) : 
  (∀ x : ℝ, (x^2 + p*x + 8) * (x^2 - 3*x + q) = x^4 + (p*q - 24)*x + 8*q) → 
  p = 3 ∧ q = 1 := by
sorry

end NUMINAMATH_CALUDE_product_no_x3_x2_terms_l111_11165


namespace NUMINAMATH_CALUDE_smallest_n_is_five_l111_11140

/-- A triple of positive integers (x, y, z) such that x + y = 3z -/
structure SpecialTriple where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  sum_condition : x + y = 3 * z

/-- The property that a positive integer n satisfies the condition -/
def SatisfiesCondition (n : ℕ+) : Prop :=
  ∃ (triples : Fin n → SpecialTriple),
    (∀ i j, i ≠ j → (triples i).x ≠ (triples j).x ∧ (triples i).y ≠ (triples j).y ∧ (triples i).z ≠ (triples j).z) ∧
    (∀ k : ℕ+, k ≤ 3*n → ∃ i, (triples i).x = k ∨ (triples i).y = k ∨ (triples i).z = k)

theorem smallest_n_is_five :
  SatisfiesCondition 5 ∧ ∀ m : ℕ+, m < 5 → ¬SatisfiesCondition m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_five_l111_11140


namespace NUMINAMATH_CALUDE_marble_distribution_l111_11172

theorem marble_distribution (y : ℚ) : 
  (4 * y + 2) + (2 * y) + (y + 3) = 31 → y = 26 / 7 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l111_11172


namespace NUMINAMATH_CALUDE_smallest_z_value_l111_11166

/-- Given four positive integers w, x, y, and z such that:
    1. w³, x³, y³, and z³ are distinct, consecutive positive perfect cubes
    2. There's a gap of 1 between w, x, and y
    3. There's a gap of 3 between y and z
    4. w³ + x³ + y³ = z³
    Then the smallest possible value of z is 9. -/
theorem smallest_z_value (w x y z : ℕ+) 
  (h1 : w.val + 1 = x.val)
  (h2 : x.val + 1 = y.val)
  (h3 : y.val + 3 = z.val)
  (h4 : w.val^3 + x.val^3 + y.val^3 = z.val^3)
  (h5 : w.val^3 < x.val^3 ∧ x.val^3 < y.val^3 ∧ y.val^3 < z.val^3) :
  z.val ≥ 9 := by
  sorry

#check smallest_z_value

end NUMINAMATH_CALUDE_smallest_z_value_l111_11166


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l111_11119

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l111_11119


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l111_11133

theorem sqrt_difference_equality : 
  Real.sqrt (49 + 81) - Real.sqrt (36 - 25) = Real.sqrt 130 - Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l111_11133


namespace NUMINAMATH_CALUDE_min_value_theorem_l111_11112

theorem min_value_theorem (a b : ℝ) (h1 : a + b = 45) (h2 : a > 0) (h3 : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 45 → 1/x + 4/y ≥ 1/5) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 45 ∧ 1/x + 4/y = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l111_11112


namespace NUMINAMATH_CALUDE_binomial_cube_constant_l111_11195

theorem binomial_cube_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 27 * x^3 + 9 * x^2 + 36 * x + a = (3 * x + b)^3) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_cube_constant_l111_11195


namespace NUMINAMATH_CALUDE_height_difference_in_inches_l111_11179

-- Define conversion factors
def meters_to_feet : ℝ := 3.28084
def inches_per_foot : ℕ := 12

-- Define heights in meters
def mark_height : ℝ := 1.60
def mike_height : ℝ := 1.85

-- Function to convert meters to inches
def meters_to_inches (m : ℝ) : ℝ := m * meters_to_feet * inches_per_foot

-- Theorem statement
theorem height_difference_in_inches :
  ⌊meters_to_inches mike_height - meters_to_inches mark_height⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_in_inches_l111_11179


namespace NUMINAMATH_CALUDE_josh_found_seven_marbles_l111_11158

/-- The number of marbles Josh had initially -/
def initial_marbles : ℕ := 21

/-- The number of marbles Josh has now -/
def current_marbles : ℕ := 28

/-- The number of marbles Josh found -/
def found_marbles : ℕ := current_marbles - initial_marbles

theorem josh_found_seven_marbles :
  found_marbles = 7 :=
by sorry

end NUMINAMATH_CALUDE_josh_found_seven_marbles_l111_11158


namespace NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l111_11192

/-- The amount of caffeine Lisa consumed over her goal -/
def caffeine_over_goal (caffeine_per_cup : ℕ) (daily_limit : ℕ) (cups_drunk : ℕ) : ℕ :=
  (caffeine_per_cup * cups_drunk) - daily_limit

theorem lisa_caffeine_over_goal :
  let caffeine_per_cup : ℕ := 80
  let daily_limit : ℕ := 200
  let cups_drunk : ℕ := 3
  caffeine_over_goal caffeine_per_cup daily_limit cups_drunk = 40 := by
sorry

end NUMINAMATH_CALUDE_lisa_caffeine_over_goal_l111_11192


namespace NUMINAMATH_CALUDE_remaining_volleyballs_l111_11125

/-- Given an initial number of volleyballs and a number of volleyballs lent out,
    calculate the number of volleyballs remaining. -/
def volleyballs_remaining (initial : ℕ) (lent_out : ℕ) : ℕ :=
  initial - lent_out

/-- Theorem stating that given 9 initial volleyballs and 5 lent out,
    the number of volleyballs remaining is 4. -/
theorem remaining_volleyballs :
  volleyballs_remaining 9 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_volleyballs_l111_11125


namespace NUMINAMATH_CALUDE_symmetry_point_l111_11173

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (mid : Point) (p1 : Point) (p2 : Point) : Prop :=
  mid.x = (p1.x + p2.x) / 2 ∧ mid.y = (p1.y + p2.y) / 2

theorem symmetry_point (m n : ℝ) :
  let M : Point := ⟨4, m⟩
  let N : Point := ⟨n, -3⟩
  let P : Point := ⟨6, -9⟩
  isMidpoint N M P → m = 3 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_l111_11173


namespace NUMINAMATH_CALUDE_equal_sum_sequence_definition_l111_11132

/-- Definition of an equal sum sequence -/
def is_equal_sum_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = a (n + 1) + a (n + 2)

/-- Theorem stating the definition of an equal sum sequence -/
theorem equal_sum_sequence_definition (a : ℕ → ℝ) :
  is_equal_sum_sequence a ↔
    ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = a (n + 1) + a (n + 2) :=
by sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_definition_l111_11132


namespace NUMINAMATH_CALUDE_log_equation_solution_l111_11108

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log (x + 12) :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l111_11108


namespace NUMINAMATH_CALUDE_sqrt_product_property_l111_11178

theorem sqrt_product_property : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_property_l111_11178


namespace NUMINAMATH_CALUDE_remainder_of_power_division_l111_11190

theorem remainder_of_power_division (n : ℕ) : 
  (2^160 + 160) % (2^81 + 2^41 + 1) = 159 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_power_division_l111_11190


namespace NUMINAMATH_CALUDE_complex_equation_solution_l111_11111

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = -2 * Complex.I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l111_11111


namespace NUMINAMATH_CALUDE_combined_tax_rate_approx_l111_11183

/-- Represents the tax system for a group of individuals in a fictional universe. -/
structure TaxSystem where
  mork_tax_rate : ℝ
  mork_deduction : ℝ
  mindy_tax_rate : ℝ
  mindy_income_multiple : ℝ
  mindy_tax_break : ℝ
  bickley_income_multiple : ℝ
  bickley_tax_rate : ℝ
  bickley_deduction : ℝ
  exidor_income_fraction : ℝ
  exidor_tax_rate : ℝ
  exidor_tax_break : ℝ

/-- Calculates the combined tax rate for the group. -/
def combined_tax_rate (ts : TaxSystem) : ℝ :=
  sorry

/-- Theorem stating that the combined tax rate is approximately 23.57% -/
theorem combined_tax_rate_approx (ts : TaxSystem) 
  (h1 : ts.mork_tax_rate = 0.45)
  (h2 : ts.mork_deduction = 0.10)
  (h3 : ts.mindy_tax_rate = 0.20)
  (h4 : ts.mindy_income_multiple = 4)
  (h5 : ts.mindy_tax_break = 0.05)
  (h6 : ts.bickley_income_multiple = 2)
  (h7 : ts.bickley_tax_rate = 0.25)
  (h8 : ts.bickley_deduction = 0.07)
  (h9 : ts.exidor_income_fraction = 0.5)
  (h10 : ts.exidor_tax_rate = 0.30)
  (h11 : ts.exidor_tax_break = 0.08) :
  abs (combined_tax_rate ts - 0.2357) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_approx_l111_11183


namespace NUMINAMATH_CALUDE_parabola_shift_l111_11152

/-- Given a parabola with equation y = 3x², prove that after shifting 2 units right and 5 units up, the new equation is y = 3(x-2)² + 5 -/
theorem parabola_shift (x y : ℝ) : 
  (y = 3 * x^2) → 
  (∃ (new_y : ℝ), new_y = 3 * (x - 2)^2 + 5 ∧ 
    new_y = y + 5 ∧ 
    ∀ (new_x : ℝ), new_x = x - 2 → 3 * new_x^2 = 3 * (x - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l111_11152


namespace NUMINAMATH_CALUDE_vovochka_max_candies_l111_11182

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies Vovochka can keep --/
def max_candies_kept (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- Theorem stating the maximum number of candies Vovochka can keep --/
theorem vovochka_max_candies :
  let cd := CandyDistribution.mk 200 25 16 100
  max_candies_kept cd = 37 := by
  sorry

end NUMINAMATH_CALUDE_vovochka_max_candies_l111_11182


namespace NUMINAMATH_CALUDE_initial_donuts_l111_11122

theorem initial_donuts (remaining : ℕ) (missing_percent : ℚ) : 
  remaining = 9 → missing_percent = 70/100 → 
  (1 - missing_percent) * 30 = remaining :=
by sorry

end NUMINAMATH_CALUDE_initial_donuts_l111_11122


namespace NUMINAMATH_CALUDE_pool_supply_problem_l111_11188

theorem pool_supply_problem (x : ℕ) (h1 : x + 3 * x = 800) : x = 266 := by
  sorry

end NUMINAMATH_CALUDE_pool_supply_problem_l111_11188


namespace NUMINAMATH_CALUDE_painted_cube_probability_l111_11115

/-- Represents a 3x3x3 cube with two adjacent faces painted -/
structure PaintedCube where
  size : Nat
  painted_faces : Nat

/-- Counts the number of cubes with exactly two painted faces -/
def count_two_painted (cube : PaintedCube) : Nat :=
  4  -- The edge cubes between the two painted faces

/-- Counts the number of cubes with no painted faces -/
def count_no_painted (cube : PaintedCube) : Nat :=
  9  -- The interior cubes not visible from any painted face

/-- Calculates the total number of ways to select two cubes -/
def total_selections (cube : PaintedCube) : Nat :=
  (cube.size^3 * (cube.size^3 - 1)) / 2

/-- The main theorem to prove -/
theorem painted_cube_probability (cube : PaintedCube) 
  (h1 : cube.size = 3) 
  (h2 : cube.painted_faces = 2) : 
  (count_two_painted cube * count_no_painted cube) / total_selections cube = 4 / 39 := by
  sorry


end NUMINAMATH_CALUDE_painted_cube_probability_l111_11115


namespace NUMINAMATH_CALUDE_total_meal_cost_l111_11103

def meal_cost (num_people : ℕ) (cost_per_person : ℚ) (tax_rate : ℚ) (tip_percentages : List ℚ) : ℚ :=
  let base_cost := num_people * cost_per_person
  let tax := base_cost * tax_rate
  let cost_with_tax := base_cost + tax
  let avg_tip_percentage := (tip_percentages.sum + 1) / tip_percentages.length
  let tip := cost_with_tax * avg_tip_percentage
  cost_with_tax + tip

theorem total_meal_cost :
  let num_people : ℕ := 5
  let cost_per_person : ℚ := 90
  let tax_rate : ℚ := 825 / 10000
  let tip_percentages : List ℚ := [15/100, 18/100, 20/100, 22/100, 25/100]
  meal_cost num_people cost_per_person tax_rate tip_percentages = 97426 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_meal_cost_l111_11103


namespace NUMINAMATH_CALUDE_train_length_l111_11170

/-- Given a train that crosses three platforms with different lengths and times,
    this theorem proves that the length of the train is 30 meters. -/
theorem train_length (platform1_length platform2_length platform3_length : ℝ)
                     (platform1_time platform2_time platform3_time : ℝ)
                     (h1 : platform1_length = 180)
                     (h2 : platform2_length = 250)
                     (h3 : platform3_length = 320)
                     (h4 : platform1_time = 15)
                     (h5 : platform2_time = 20)
                     (h6 : platform3_time = 25) :
  ∃ (train_length : ℝ), 
    train_length = 30 ∧ 
    (train_length + platform1_length) / platform1_time = 
    (train_length + platform2_length) / platform2_time ∧
    (train_length + platform2_length) / platform2_time = 
    (train_length + platform3_length) / platform3_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l111_11170


namespace NUMINAMATH_CALUDE_A_intersect_B_l111_11137

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 1| ≥ 2}

theorem A_intersect_B : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l111_11137


namespace NUMINAMATH_CALUDE_select_representatives_l111_11107

theorem select_representatives (boys girls total reps : ℕ) 
  (h1 : boys = 6)
  (h2 : girls = 4)
  (h3 : total = boys + girls)
  (h4 : reps = 3) :
  (Nat.choose total reps) - (Nat.choose boys reps) = 100 := by
  sorry

end NUMINAMATH_CALUDE_select_representatives_l111_11107


namespace NUMINAMATH_CALUDE_balloon_distribution_l111_11143

theorem balloon_distribution (red white green chartreuse : ℕ) (friends : ℕ) : 
  red = 22 → white = 40 → green = 70 → chartreuse = 90 → friends = 10 →
  (red + white + green + chartreuse) % friends = 2 :=
by sorry

end NUMINAMATH_CALUDE_balloon_distribution_l111_11143


namespace NUMINAMATH_CALUDE_square_side_length_difference_l111_11162

/-- Given two squares with side lengths x and y, where the perimeter of the smaller square
    is 20 cm less than the perimeter of the larger square, prove that the side length of
    the larger square is 5 cm more than the side length of the smaller square. -/
theorem square_side_length_difference (x y : ℝ) (h : 4 * x + 20 = 4 * y) : y = x + 5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_difference_l111_11162


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l111_11148

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 2) = 15 → ∃ y : ℝ, (y + 3) * (y - 2) = 15 ∧ x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l111_11148


namespace NUMINAMATH_CALUDE_max_children_correct_l111_11157

/-- Represents the types of buses available --/
inductive BusType
| A
| B
| C
| D

/-- Calculates the total number of seats for a given bus type --/
def totalSeats (t : BusType) : ℕ :=
  match t with
  | BusType.A => 36
  | BusType.B => 54
  | BusType.C => 36
  | BusType.D => 36

/-- Represents the safety regulation for maximum number of children per bus type --/
def safetyRegulation (t : BusType) : ℕ :=
  match t with
  | BusType.A => 40
  | BusType.B => 50
  | BusType.C => 35
  | BusType.D => 30

/-- Calculates the maximum number of children that can be accommodated on a given bus type --/
def maxChildren (t : BusType) : ℕ :=
  min (totalSeats t) (safetyRegulation t)

theorem max_children_correct :
  (maxChildren BusType.A = 36) ∧
  (maxChildren BusType.B = 50) ∧
  (maxChildren BusType.C = 35) ∧
  (maxChildren BusType.D = 30) :=
by
  sorry


end NUMINAMATH_CALUDE_max_children_correct_l111_11157


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l111_11123

theorem max_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y/2 + 1/x + 8/y = 10) : 
  ∃ (z : ℝ), z = 2*x + y ∧ ∀ (w : ℝ), (∃ (a b : ℝ) (ha : a > 0) (hb : b > 0), 
    w = 2*a + b ∧ a + b/2 + 1/a + 8/b = 10) → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l111_11123


namespace NUMINAMATH_CALUDE_carla_hits_nine_l111_11185

/-- Represents a player in the dart contest -/
inductive Player : Type
| Anne : Player
| Joe : Player
| Carla : Player
| Larry : Player
| Naomi : Player
| Mike : Player

/-- The score of each player -/
def score (p : Player) : ℕ :=
  match p with
  | Player.Anne => 21
  | Player.Joe => 18
  | Player.Carla => 14
  | Player.Larry => 22
  | Player.Naomi => 25
  | Player.Mike => 13

/-- The set of possible scores for each throw -/
def possible_scores : Set ℕ := Finset.range 15

/-- Predicate to check if a list of scores is valid for a player -/
def valid_scores (s : List ℕ) (p : Player) : Prop :=
  s.length = 3 ∧ 
  s.sum = score p ∧
  s.toFinset.card = 3 ∧
  ∀ x ∈ s, x ∈ possible_scores

theorem carla_hits_nine : 
  ∃! (p : Player), ∃ (s : List ℕ), valid_scores s p ∧ 9 ∈ s ∧ p = Player.Carla :=
sorry

end NUMINAMATH_CALUDE_carla_hits_nine_l111_11185


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l111_11181

/-- Given a rectangle with area 3a² - 3ab + 6a and one side length 3a, its perimeter is 8a - 2b + 4 -/
theorem rectangle_perimeter (a b : ℝ) : 
  let area := 3 * a^2 - 3 * a * b + 6 * a
  let side1 := 3 * a
  let side2 := area / side1
  let perimeter := 2 * (side1 + side2)
  perimeter = 8 * a - 2 * b + 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l111_11181


namespace NUMINAMATH_CALUDE_polynomial_equality_l111_11136

theorem polynomial_equality (a b c : ℝ) : 
  ((a - b) - c = a - b - c) ∧ 
  (a - (b + c) = a - b - c) ∧ 
  (-(b + c - a) = a - b - c) ∧ 
  (a - (b - c) ≠ a - b - c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l111_11136


namespace NUMINAMATH_CALUDE_sqrt_equation_l111_11186

theorem sqrt_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l111_11186


namespace NUMINAMATH_CALUDE_negative_two_inequality_l111_11194

theorem negative_two_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l111_11194


namespace NUMINAMATH_CALUDE_otimes_composition_l111_11199

/-- Define the custom operation ⊗ on integers -/
def otimes (x y : ℤ) : ℤ := x^3 - 2*y

/-- Theorem stating the result of h ⊗ (h ⊗ 2h) -/
theorem otimes_composition (h : ℤ) : otimes h (otimes h (2*h)) = 8*h - h^3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l111_11199


namespace NUMINAMATH_CALUDE_abs_inequality_l111_11160

theorem abs_inequality (x : ℝ) : |5 - x| > 6 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi 11 :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_l111_11160


namespace NUMINAMATH_CALUDE_number_calculation_l111_11126

theorem number_calculation (n : ℝ) : 0.125 * 0.20 * 0.40 * 0.75 * n = 148.5 → n = 23760 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l111_11126


namespace NUMINAMATH_CALUDE_salary_fraction_on_food_l111_11144

theorem salary_fraction_on_food
  (salary : ℝ)
  (rent_fraction : ℝ)
  (clothes_fraction : ℝ)
  (remaining : ℝ)
  (h1 : salary = 180000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remaining = 18000)
  (h5 : ∃ food_fraction : ℝ, 
    food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary) :
  ∃ food_fraction : ℝ, food_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_salary_fraction_on_food_l111_11144


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l111_11175

theorem difference_of_squares_factorization (x y : ℝ) : 
  x^2 - 4*y^2 = (x + 2*y) * (x - 2*y) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l111_11175


namespace NUMINAMATH_CALUDE_line_equation_l111_11142

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 5 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x - 1

-- Define the intersection points
def intersection (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola x y ∧ line k x y

-- Define the midpoint condition
def midpoint_condition (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection k x₁ y₁ ∧
    intersection k x₂ y₂ ∧
    (x₁ + x₂) / 2 = -2/3

theorem line_equation :
  ∀ k : ℝ, midpoint_condition k → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l111_11142


namespace NUMINAMATH_CALUDE_problem_solution_l111_11197

def A (a : ℕ) : Set ℝ := {x : ℝ | |x + 1| > a}

theorem problem_solution (a : ℕ) 
  (h1 : 3/2 ∈ A a) 
  (h2 : 1/2 ∉ A a) 
  (h3 : a > 0) :
  (a = 2) ∧ 
  (∀ m n s : ℝ, m > 0 → n > 0 → s > 0 → m + n + Real.sqrt 2 * s = a → 
    m^2 + n^2 + s^2 ≥ 1 ∧ ∃ m₀ n₀ s₀, m₀ > 0 ∧ n₀ > 0 ∧ s₀ > 0 ∧ 
      m₀ + n₀ + Real.sqrt 2 * s₀ = a ∧ m₀^2 + n₀^2 + s₀^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l111_11197


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_840_l111_11145

theorem least_n_factorial_divisible_by_840 :
  ∀ n : ℕ, n > 0 → (n.factorial % 840 = 0) → n ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_840_l111_11145


namespace NUMINAMATH_CALUDE_inequality_proof_l111_11191

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2 * a^2 + b^2 = 9 * c^2) : 
  (2 * c / a) + (c / b) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l111_11191


namespace NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l111_11167

theorem max_value_product (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  x^2 * y^3 * z ≤ 9/16 := by
  sorry

theorem max_value_achieved (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  ∃ x y z, x^2 * y^3 * z = 9/16 ∧ x + y + z = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_product_max_value_achieved_l111_11167


namespace NUMINAMATH_CALUDE_qing_dynasty_problem_l111_11138

/-- Represents the price of animals in ancient Chinese currency (taels) --/
structure AnimalPrices where
  horse : ℝ
  cattle : ℝ

/-- Represents a combination of horses and cattle --/
structure AnimalCombination where
  horses : ℕ
  cattle : ℕ

/-- Calculates the total cost of a combination of animals given their prices --/
def totalCost (prices : AnimalPrices) (combo : AnimalCombination) : ℝ :=
  prices.horse * combo.horses + prices.cattle * combo.cattle

/-- The theorem representing the original problem --/
theorem qing_dynasty_problem (prices : AnimalPrices) : 
  totalCost prices ⟨4, 6⟩ = 48 ∧ 
  totalCost prices ⟨2, 5⟩ = 38 ↔ 
  4 * prices.horse + 6 * prices.cattle = 48 ∧
  2 * prices.horse + 5 * prices.cattle = 38 := by
  sorry


end NUMINAMATH_CALUDE_qing_dynasty_problem_l111_11138


namespace NUMINAMATH_CALUDE_special_function_property_equivalence_l111_11177

/-- A function satisfying f(ab) ≥ f(a) + f(b) for non-zero integers -/
def SpecialFunction (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, a ≠ 0 → b ≠ 0 → f (a * b) ≥ f a + f b

/-- The property that f(a^n) = nf(a) for all non-zero integers a and natural numbers n -/
def PropertyForAllN (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → ∀ n : ℕ, f (a ^ n) = n * f a

/-- The property that f(a^2) = 2f(a) for all non-zero integers a -/
def PropertyForTwo (f : ℤ → ℤ) : Prop :=
  ∀ a : ℤ, a ≠ 0 → f (a ^ 2) = 2 * f a

/-- Theorem stating the equivalence of the two properties for special functions -/
theorem special_function_property_equivalence (f : ℤ → ℤ) (h : SpecialFunction f) :
  PropertyForAllN f ↔ PropertyForTwo f := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_equivalence_l111_11177


namespace NUMINAMATH_CALUDE_incircle_iff_reciprocal_heights_sum_l111_11104

/-- A quadrilateral with heights h₁, h₂, h₃, h₄ -/
structure Quadrilateral where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₄ : ℝ
  h₁_pos : 0 < h₁
  h₂_pos : 0 < h₂
  h₃_pos : 0 < h₃
  h₄_pos : 0 < h₄

/-- The property of having an incircle -/
def has_incircle (q : Quadrilateral) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∃ (center : ℝ × ℝ), True  -- We don't specify the exact conditions for an incircle

/-- The main theorem: a quadrilateral has an incircle iff the sum of reciprocals of opposite heights are equal -/
theorem incircle_iff_reciprocal_heights_sum (q : Quadrilateral) :
  has_incircle q ↔ 1 / q.h₁ + 1 / q.h₃ = 1 / q.h₂ + 1 / q.h₄ := by
  sorry

end NUMINAMATH_CALUDE_incircle_iff_reciprocal_heights_sum_l111_11104


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_correct_negation_incorrect_disjunction_false_implication_l111_11151

-- Definition for the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

-- Proposition 1
theorem sufficient_not_necessary : 
  (∃ x : ℝ, x ≠ 1 ∧ quadratic_eq x) ∧ 
  (∀ x : ℝ, x = 1 → quadratic_eq x) := by sorry

-- Proposition 2
theorem contrapositive_correct :
  (∀ x : ℝ, quadratic_eq x → x = 1) ↔ 
  (∀ x : ℝ, x ≠ 1 → ¬(quadratic_eq x)) := by sorry

-- Proposition 3
theorem negation_incorrect :
  ¬(∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ≠ 
  (∀ x : ℝ, x ≤ 0 → x^2 + x + 1 ≥ 0) := by sorry

-- Proposition 4
theorem disjunction_false_implication :
  ¬(∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_contrapositive_correct_negation_incorrect_disjunction_false_implication_l111_11151


namespace NUMINAMATH_CALUDE_stating_smallest_n_no_arithmetic_progression_l111_11113

/-- 
A function that checks if there exists an arithmetic progression of 
1999 terms containing exactly n integers
-/
def exists_arithmetic_progression (n : ℕ) : Prop :=
  ∃ (a d : ℝ), ∃ (k : ℕ), 
    k * n + k - 1 ≥ 1999 ∧
    (k + 1) * n - (k + 1) + 1 ≤ 1999

/-- 
Theorem stating that 70 is the smallest positive integer n such that 
there does not exist an arithmetic progression of 1999 terms of real 
numbers containing exactly n integers
-/
theorem smallest_n_no_arithmetic_progression : 
  (∀ m < 70, exists_arithmetic_progression m) ∧ 
  ¬ exists_arithmetic_progression 70 :=
sorry

end NUMINAMATH_CALUDE_stating_smallest_n_no_arithmetic_progression_l111_11113


namespace NUMINAMATH_CALUDE_base_number_proof_l111_11193

theorem base_number_proof (x : ℝ) : x^3 = 1024 * (1/4)^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l111_11193


namespace NUMINAMATH_CALUDE_sin_330_degrees_l111_11159

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l111_11159


namespace NUMINAMATH_CALUDE_jason_has_21_toys_l111_11116

/-- The number of toys Rachel has -/
def rachel_toys : ℕ := 1

/-- The number of toys John has -/
def john_toys : ℕ := rachel_toys + 6

/-- The number of toys Jason has -/
def jason_toys : ℕ := 3 * john_toys

/-- Theorem: Jason has 21 toys -/
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_21_toys_l111_11116


namespace NUMINAMATH_CALUDE_movie_theater_receipts_l111_11135

/-- 
Given a movie theater with the following conditions:
- Child ticket price is $4.50
- Adult ticket price is $6.75
- There are 20 more children than adults
- There are 48 children at the matinee

Prove that the total receipts for today's matinee is $405.
-/
theorem movie_theater_receipts : 
  let child_price : ℚ := 4.5
  let adult_price : ℚ := 6.75
  let child_count : ℕ := 48
  let adult_count : ℕ := child_count - 20
  let total_receipts : ℚ := child_price * child_count + adult_price * adult_count
  total_receipts = 405 := by sorry

end NUMINAMATH_CALUDE_movie_theater_receipts_l111_11135


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l111_11120

/-- Given a polynomial g(x) that satisfies g(x + 1) - g(x) = 8x + 6 for all x,
    prove that the leading coefficient of g(x) is 4. -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) : 
  (∀ x, g (x + 1) - g x = 8 * x + 6) → 
  ∃ a b c : ℝ, (∀ x, g x = 4 * x^2 + a * x + b) ∧ c = 4 ∧ c * x^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l111_11120


namespace NUMINAMATH_CALUDE_symmetry_properties_l111_11114

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ

-- Define a quadratic function type
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to get the symmetric line about x-axis
def symmetricAboutXAxis (l : Line) : Line :=
  { a := -l.a, b := -l.b }

-- Function to get the symmetric line about y-axis
def symmetricAboutYAxis (l : Line) : Line :=
  { a := -l.a, b := l.b }

-- Function to get the symmetric quadratic function about origin
def symmetricAboutOrigin (q : QuadraticFunction) : QuadraticFunction :=
  { a := q.a, b := -q.b, c := -q.c }

-- Theorem statements
theorem symmetry_properties (l : Line) (q : QuadraticFunction) :
  (symmetricAboutXAxis l = { a := -l.a, b := -l.b }) ∧
  (symmetricAboutYAxis l = { a := -l.a, b := l.b }) ∧
  (symmetricAboutOrigin q = { a := q.a, b := -q.b, c := -q.c }) := by
  sorry


end NUMINAMATH_CALUDE_symmetry_properties_l111_11114


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l111_11198

/-- Given an ellipse and a hyperbola with the same foci, prove that the semi-major axis of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 4 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, x^2 / 9 - y^2 / 3 = 1) →   -- Hyperbola equation
  (a > 0) →                              -- a is positive
  (a^2 - 4 = 12) →                       -- Same foci condition
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l111_11198


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l111_11130

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l111_11130


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l111_11105

/-- A quadratic function with vertex at (-1, 4) passing through (2, -5) -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 4

theorem quadratic_function_properties :
  (∀ x, f x = -x^2 - 2*x + 3) ∧
  (f (-1/2) = 11/4) ∧
  (∀ x, f x = 3 ↔ x = 0 ∨ x = -2) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l111_11105


namespace NUMINAMATH_CALUDE_expression_equals_eight_l111_11124

theorem expression_equals_eight :
  ((18^18 / 18^17)^3 * 9^3) / 3^6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_eight_l111_11124


namespace NUMINAMATH_CALUDE_negation_of_implication_or_l111_11164

theorem negation_of_implication_or (p q r : Prop) :
  ¬(r → p ∨ q) ↔ (¬r → ¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_or_l111_11164


namespace NUMINAMATH_CALUDE_emilys_purchase_cost_l111_11118

/-- The total cost of Emily's purchase including installation service -/
theorem emilys_purchase_cost : 
  let curtain_pairs : ℕ := 2
  let curtain_price : ℚ := 30
  let wall_prints : ℕ := 9
  let wall_print_price : ℚ := 15
  let installation_fee : ℚ := 50
  (curtain_pairs : ℚ) * curtain_price + 
  (wall_prints : ℚ) * wall_print_price + 
  installation_fee = 245 :=
by sorry

end NUMINAMATH_CALUDE_emilys_purchase_cost_l111_11118


namespace NUMINAMATH_CALUDE_inverse_function_relation_l111_11141

/-- Given a function h and its inverse f⁻¹, prove the relation between a and b --/
theorem inverse_function_relation (a b : ℝ) :
  (∀ x, 3 * x - 6 = (Function.invFun (fun x => a * x + b)) x - 2) →
  3 * a + 4 * b = 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_relation_l111_11141


namespace NUMINAMATH_CALUDE_store_discount_percentage_l111_11150

/-- Represents the pricing strategy and profit of a store selling turtleneck sweaters -/
theorem store_discount_percentage (C : ℝ) (D : ℝ) : 
  C > 0 → -- Cost price is positive
  (1.20 * C) * 1.25 * (1 - D / 100) = 1.35 * C → -- February selling price equals 35% profit
  D = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l111_11150


namespace NUMINAMATH_CALUDE_triangle_max_area_l111_11154

theorem triangle_max_area (A B C : ℝ) (a b c : ℝ) :
  (Real.cos A / Real.sin B + Real.cos B / Real.sin A = 2) →
  (a + b + c = 12) →
  (∀ a' b' c' : ℝ, a' + b' + c' = 12 → 
    a' * b' * Real.sin C / 2 ≤ 36 * (3 - 2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l111_11154


namespace NUMINAMATH_CALUDE_distributive_property_first_calculation_l111_11184

theorem distributive_property (a b c : ℤ) : a * (b + c) = a * b + a * c := by sorry

theorem first_calculation : -17 * 43 + (-17) * 20 - (-17) * 163 = 1700 := by sorry

end NUMINAMATH_CALUDE_distributive_property_first_calculation_l111_11184


namespace NUMINAMATH_CALUDE_brendas_age_is_real_l111_11117

/-- Represents the ages of individuals --/
structure Ages where
  addison : ℝ
  brenda : ℝ
  carlos : ℝ
  janet : ℝ

/-- The conditions given in the problem --/
def age_conditions (ages : Ages) : Prop :=
  ages.addison = 4 * ages.brenda ∧
  ages.carlos = 2 * ages.brenda ∧
  ages.addison = ages.janet

/-- Theorem stating that Brenda's age is a positive real number --/
theorem brendas_age_is_real (ages : Ages) (h : age_conditions ages) :
  ∃ (B : ℝ), B > 0 ∧ ages.brenda = B :=
sorry

end NUMINAMATH_CALUDE_brendas_age_is_real_l111_11117


namespace NUMINAMATH_CALUDE_triangle_area_l111_11106

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 15√3/4 when b = 7, c = 5, and B = 2π/3 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 7 → c = 5 → B = 2 * π / 3 → 
  (1/2) * b * c * Real.sin B = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l111_11106


namespace NUMINAMATH_CALUDE_function_passes_through_point_l111_11128

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) - 1
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l111_11128


namespace NUMINAMATH_CALUDE_candies_per_packet_is_18_l111_11155

/-- The number of candies in each packet -/
def candies_per_packet : ℕ := 18

/-- The number of packets Bobby has -/
def num_packets : ℕ := 2

/-- The number of days Bobby eats 2 candies -/
def days_eating_two : ℕ := 5

/-- The number of days Bobby eats 1 candy -/
def days_eating_one : ℕ := 2

/-- The number of weeks it takes to finish the packets -/
def weeks_to_finish : ℕ := 3

/-- Theorem stating that the number of candies in each packet is 18 -/
theorem candies_per_packet_is_18 :
  candies_per_packet * num_packets = 
    (days_eating_two * 2 + days_eating_one) * weeks_to_finish :=
by sorry

end NUMINAMATH_CALUDE_candies_per_packet_is_18_l111_11155


namespace NUMINAMATH_CALUDE_rectangle_ratio_sum_l111_11127

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Definition of the specific rectangle in the problem -/
def problemRectangle : Rectangle :=
  { A := ⟨0, 0⟩
  , B := ⟨6, 0⟩
  , C := ⟨6, 3⟩
  , D := ⟨0, 3⟩ }

/-- Point E on BC -/
def E : Point :=
  ⟨6, 1⟩

/-- Point F on CE -/
def F : Point :=
  ⟨6, 2⟩

/-- Theorem statement -/
theorem rectangle_ratio_sum (r s t : ℕ) :
  (r > 0 ∧ s > 0 ∧ t > 0) →
  (Nat.gcd r (Nat.gcd s t) = 1) →
  (∃ (P Q : Point),
    P.x = Q.x ∧ 
    P.y < Q.y ∧
    Q.y < problemRectangle.D.y ∧
    P.x > problemRectangle.A.x ∧
    P.x < problemRectangle.B.x ∧
    (P.x - problemRectangle.A.x) / (Q.x - P.x) = r / s ∧
    (Q.x - P.x) / (problemRectangle.B.x - Q.x) = s / t) →
  r + s + t = 20 := by
    sorry

end NUMINAMATH_CALUDE_rectangle_ratio_sum_l111_11127


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l111_11129

theorem greatest_common_multiple_under_120 : 
  ∃ (n : ℕ), n = 90 ∧ 
  (∀ m : ℕ, m < 120 ∧ 9 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  9 ∣ n ∧ 15 ∣ n ∧ n < 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_120_l111_11129


namespace NUMINAMATH_CALUDE_jakes_drink_volume_l111_11156

/-- Represents the composition of a drink in parts -/
structure DrinkComposition :=
  (coke : ℕ)
  (sprite : ℕ)
  (mountainDew : ℕ)

/-- Calculates the total volume of a drink given its composition and the volume of Coke -/
def totalVolume (composition : DrinkComposition) (cokeVolume : ℚ) : ℚ :=
  let totalParts := composition.coke + composition.sprite + composition.mountainDew
  let volumePerPart := cokeVolume / composition.coke
  totalParts * volumePerPart

/-- Theorem: The total volume of Jake's drink is 18 ounces -/
theorem jakes_drink_volume :
  let composition : DrinkComposition := ⟨2, 1, 3⟩
  let cokeVolume : ℚ := 6
  totalVolume composition cokeVolume = 18 := by
  sorry

end NUMINAMATH_CALUDE_jakes_drink_volume_l111_11156


namespace NUMINAMATH_CALUDE_max_ab_value_l111_11171

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a + 4*b + a*b = 3) :
  a * b ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_ab_value_l111_11171


namespace NUMINAMATH_CALUDE_max_sum_fourth_powers_l111_11146

theorem max_sum_fourth_powers (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  ∃ (M : ℝ), M = 64 ∧ a^4 + b^4 + c^4 + d^4 ≤ M ∧ 
  ∃ (a' b' c' d' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 = 16 ∧ a'^4 + b'^4 + c'^4 + d'^4 = M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_fourth_powers_l111_11146


namespace NUMINAMATH_CALUDE_reflection_coordinates_sum_l111_11102

/-- Given a point C at (3, y+4) and its reflection D over the y-axis, with y = 2,
    the sum of all four coordinates of C and D is equal to 12. -/
theorem reflection_coordinates_sum :
  let y : ℝ := 2
  let C : ℝ × ℝ := (3, y + 4)
  let D : ℝ × ℝ := (-C.1, C.2)  -- Reflection over y-axis
  C.1 + C.2 + D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_reflection_coordinates_sum_l111_11102


namespace NUMINAMATH_CALUDE_prime_sum_product_l111_11101

theorem prime_sum_product : ∃ p q : ℕ, 
  Nat.Prime p ∧ Nat.Prime q ∧ p + q = 97 ∧ p * q = 190 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l111_11101


namespace NUMINAMATH_CALUDE_people_who_didnt_show_up_l111_11109

def invited_people : ℕ := 47
def people_per_table : ℕ := 5
def tables_needed : ℕ := 8

def seated_people : ℕ := people_per_table * tables_needed

theorem people_who_didnt_show_up :
  invited_people - seated_people = 7 := by sorry

end NUMINAMATH_CALUDE_people_who_didnt_show_up_l111_11109


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l111_11149

/-- In a right-angled triangle ABC, the sum of arctan(b/(a+c)) and arctan(c/(a+b)) is equal to π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let triangle_abc : (ℝ × ℝ × ℝ) := (a, b, c)
  b^2 + c^2 = a^2 →
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l111_11149


namespace NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l111_11169

theorem complex_magnitude_fourth_power : 
  Complex.abs ((7/5 : ℂ) + (24/5 : ℂ) * Complex.I) ^ 4 = 625 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_fourth_power_l111_11169


namespace NUMINAMATH_CALUDE_tree_distance_l111_11187

/-- Given a yard of length 180 meters with 11 trees planted at equal distances,
    with one tree at each end, the distance between two consecutive trees is 18 meters. -/
theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 180 →
  num_trees = 11 →
  let num_spaces := num_trees - 1
  yard_length / num_spaces = 18 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l111_11187


namespace NUMINAMATH_CALUDE_speed_ratio_with_delay_l111_11176

/-- Given a normal travel time and a delay, calculate the ratio of new speed to original speed -/
theorem speed_ratio_with_delay (normal_time delay : ℝ) (normal_time_pos : normal_time > 0) 
  (delay_pos : delay > 0) : 
  (normal_time / (normal_time + delay)) = 5 / 6 :=
by
  sorry

#check speed_ratio_with_delay 60 12

end NUMINAMATH_CALUDE_speed_ratio_with_delay_l111_11176


namespace NUMINAMATH_CALUDE_largest_fraction_l111_11153

theorem largest_fraction :
  let f1 := 397 / 101
  let f2 := 487 / 121
  let f3 := 596 / 153
  let f4 := 678 / 173
  let f5 := 796 / 203
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 ∧ f2 > f5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l111_11153


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l111_11163

/-- The diameter of a triangle's circumscribed circle, given one side and its opposite angle -/
theorem circumscribed_circle_diameter 
  (side : ℝ) 
  (angle : ℝ) 
  (h1 : side = 15) 
  (h2 : angle = π / 4) : 
  side / Real.sin angle = 15 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l111_11163


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l111_11168

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and an asymptote y = 2x/3,
    the focal length is 2√13. -/
theorem hyperbola_focal_length (m : ℝ) :
  (∃ (x y : ℝ), x^2/9 - y^2/m = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), y = 2*x/3) →         -- Asymptote equation
  2 * Real.sqrt 13 = 2 * Real.sqrt ((9:ℝ) + m) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l111_11168


namespace NUMINAMATH_CALUDE_bookkeeper_probability_l111_11161

def word_length : ℕ := 10

def num_e : ℕ := 3
def num_o : ℕ := 2
def num_k : ℕ := 2
def num_b : ℕ := 1
def num_p : ℕ := 1
def num_r : ℕ := 1

def adjacent_o : Prop := true
def two_adjacent_e : Prop := true
def no_o_e_at_beginning : Prop := true

def total_arrangements : ℕ := 9600

theorem bookkeeper_probability : 
  word_length = num_e + num_o + num_k + num_b + num_p + num_r →
  adjacent_o →
  two_adjacent_e →
  no_o_e_at_beginning →
  (1 : ℚ) / total_arrangements = (1 : ℚ) / 9600 :=
sorry

end NUMINAMATH_CALUDE_bookkeeper_probability_l111_11161


namespace NUMINAMATH_CALUDE_handshake_count_l111_11189

/-- The number of handshakes in a convention of gremlins and imps -/
theorem handshake_count (num_gremlins num_imps : ℕ) : 
  num_gremlins = 20 →
  num_imps = 15 →
  (num_gremlins * (num_gremlins - 1)) / 2 + num_gremlins * num_imps = 490 := by
  sorry

#check handshake_count

end NUMINAMATH_CALUDE_handshake_count_l111_11189


namespace NUMINAMATH_CALUDE_gwen_homework_problems_l111_11121

/-- Represents the number of problems for each subject -/
structure SubjectProblems where
  math : ℕ
  science : ℕ
  history : ℕ
  english : ℕ

/-- Calculates the total number of problems left for homework -/
def problems_left (initial : SubjectProblems) (completed : SubjectProblems) : ℕ :=
  (initial.math - completed.math) +
  (initial.science - completed.science) +
  (initial.history - completed.history) +
  (initial.english - completed.english)

/-- Theorem: Given Gwen's initial problems and completed problems, she has 19 problems left for homework -/
theorem gwen_homework_problems :
  let initial := SubjectProblems.mk 18 11 15 7
  let completed := SubjectProblems.mk 12 6 10 4
  problems_left initial completed = 19 := by
  sorry

end NUMINAMATH_CALUDE_gwen_homework_problems_l111_11121


namespace NUMINAMATH_CALUDE_impossible_time_travel_l111_11110

/-- Represents a month in the Gregorian calendar --/
inductive Month : Type
| jan | feb | mar | apr | may | jun | jul | aug | sep | oct | nov | dec

/-- Converts a Month to its corresponding index (1-12) --/
def monthToIndex (m : Month) : Nat :=
  match m with
  | .jan => 1 | .feb => 2 | .mar => 3 | .apr => 4
  | .may => 5 | .jun => 6 | .jul => 7 | .aug => 8
  | .sep => 9 | .oct => 10 | .nov => 11 | .dec => 12

/-- Represents a single time jump --/
def timeJump (start : Month) : Month :=
  match (monthToIndex start + 8) % 12 with
  | 0 => Month.dec
  | 1 => Month.jan
  | 2 => Month.feb
  | 3 => Month.mar
  | 4 => Month.apr
  | 5 => Month.may
  | 6 => Month.jun
  | 7 => Month.jul
  | 8 => Month.aug
  | 9 => Month.sep
  | 10 => Month.oct
  | 11 => Month.nov
  | _ => Month.dec  -- This case should never occur

/-- Represents a sequence of time jumps --/
def timeTravel (start : Month) (jumps : Nat) : Month :=
  match jumps with
  | 0 => start
  | n + 1 => timeJump (timeTravel start n)

theorem impossible_time_travel :
  ∀ (jumps : Nat), timeTravel Month.apr jumps ≠ Month.jun :=
by sorry

end NUMINAMATH_CALUDE_impossible_time_travel_l111_11110


namespace NUMINAMATH_CALUDE_greatest_consecutive_sum_48_l111_11139

/-- The sum of N consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (N : ℕ) : ℤ := N * (2 * a + N - 1) / 2

/-- The proposition that 96 is the greatest number of consecutive integers whose sum is 48 -/
theorem greatest_consecutive_sum_48 :
  ∀ N : ℕ, (∃ a : ℤ, sum_consecutive a N = 48) → N ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_sum_48_l111_11139


namespace NUMINAMATH_CALUDE_probability_theorem_l111_11134

/-- The probability of having a child with younger brother, older brother, younger sister, and older sister
    given n > 4 children and equal probability of male and female births -/
def probability (n : ℕ) : ℚ :=
  1 - (n - 2 : ℚ) / 2^(n - 3)

/-- Theorem stating the probability for the given conditions -/
theorem probability_theorem (n : ℕ) (h : n > 4) :
  probability n = 1 - (n - 2 : ℚ) / 2^(n - 3) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l111_11134


namespace NUMINAMATH_CALUDE_log_sqrt_45_l111_11147

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sqrt_45 (a b : ℝ) (h1 : log10 2 = a) (h2 : log10 3 = b) :
  log10 (Real.sqrt 45) = -a/2 + b + 1/2 := by sorry

end NUMINAMATH_CALUDE_log_sqrt_45_l111_11147


namespace NUMINAMATH_CALUDE_iesha_school_books_l111_11180

/-- The number of books Iesha has about school -/
def books_about_school (total_books sports_books : ℕ) : ℕ :=
  total_books - sports_books

/-- Theorem stating that Iesha has 136 books about school -/
theorem iesha_school_books :
  books_about_school 344 208 = 136 := by
  sorry

end NUMINAMATH_CALUDE_iesha_school_books_l111_11180
