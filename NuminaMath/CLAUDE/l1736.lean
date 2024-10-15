import Mathlib

namespace NUMINAMATH_CALUDE_tangent_product_inequality_l1736_173693

theorem tangent_product_inequality (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.tan (α/2) * Real.tan (β/2) * Real.tan (γ/2) ≤ Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_inequality_l1736_173693


namespace NUMINAMATH_CALUDE_max_value_k_l1736_173611

theorem max_value_k (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) : 
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧ 
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) := by
sorry

end NUMINAMATH_CALUDE_max_value_k_l1736_173611


namespace NUMINAMATH_CALUDE_range_of_a_l1736_173646

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + (a - 1) * x + 1 = 0 ∧ 
             y^2 + (a - 1) * y + 1 = 0 ∧ 
             0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2

theorem range_of_a :
  ∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ 
           ¬(proposition_p a ∧ proposition_q a) →
           (a ∈ Set.Ioc (-2) (-3/2) ∪ Set.Icc (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1736_173646


namespace NUMINAMATH_CALUDE_pens_per_student_l1736_173650

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) :
  total_pens = 100 →
  total_pencils = 50 →
  max_students = 50 →
  total_pens / max_students = 2 :=
by sorry

end NUMINAMATH_CALUDE_pens_per_student_l1736_173650


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l1736_173687

/-- The number of ice cream flavors -/
def n : ℕ := 8

/-- The number of scoops in each sundae -/
def k : ℕ := 2

/-- The number of unique two scoop sundaes -/
def unique_sundaes : ℕ := Nat.choose n k

theorem ice_cream_sundaes :
  unique_sundaes = 28 := by sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l1736_173687


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l1736_173615

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the y-axis reflection function
def reflect_y (p : Point) : Point :=
  (-p.1, p.2)

-- Define the problem statement
theorem sum_of_coordinates_after_reflection :
  let C : Point := (3, 8)
  let D : Point := reflect_y C
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l1736_173615


namespace NUMINAMATH_CALUDE_water_formed_l1736_173671

-- Define the substances involved in the reaction
structure Substance where
  name : String
  moles : ℝ
  molar_mass : ℝ

-- Define the reaction
def reaction (naoh : Substance) (hcl : Substance) (water : Substance) : Prop :=
  naoh.name = "Sodium hydroxide" ∧
  hcl.name = "Hydrochloric acid" ∧
  water.name = "Water" ∧
  naoh.moles = 1 ∧
  water.molar_mass = 18 ∧
  water.moles * water.molar_mass = 18

-- Theorem statement
theorem water_formed (naoh hcl water : Substance) :
  reaction naoh hcl water → water.moles = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_l1736_173671


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l1736_173695

/-- Proves that the average speed of a triathlete is 0.125 miles per minute
    given specific conditions for running and swimming. -/
theorem triathlete_average_speed
  (run_distance : ℝ)
  (swim_distance : ℝ)
  (run_speed : ℝ)
  (swim_speed : ℝ)
  (h1 : run_distance = 3)
  (h2 : swim_distance = 3)
  (h3 : run_speed = 10)
  (h4 : swim_speed = 6) :
  (run_distance + swim_distance) / ((run_distance / run_speed + swim_distance / swim_speed) * 60) = 0.125 := by
  sorry

#check triathlete_average_speed

end NUMINAMATH_CALUDE_triathlete_average_speed_l1736_173695


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l1736_173689

-- Define the system of equations
def satisfies_system (x y : ℝ) : Prop :=
  x + 2*y = 2 ∧ |abs x - 2*(abs y)| = 2

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {pair | satisfies_system pair.1 pair.2}

-- Theorem statement
theorem exactly_two_solutions :
  ∃ (a b c d : ℝ), 
    solution_set = {(a, b), (c, d)} ∧
    (a, b) ≠ (c, d) ∧
    ∀ (x y : ℝ), (x, y) ∈ solution_set → (x, y) = (a, b) ∨ (x, y) = (c, d) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l1736_173689


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1736_173669

theorem arithmetic_series_sum (k : ℕ) : 
  let a₁ : ℕ := k^2 + k + 1
  let d : ℕ := 1
  let n : ℕ := 2*k + 3
  let S := n * (2*a₁ + (n-1)*d) / 2
  S = 2*k^3 + 7*k^2 + 10*k + 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l1736_173669


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1736_173625

/-- A function that represents the relationship between x and y -/
def f (x : ℝ) : ℝ := -2 * x + 6

/-- The proposition that f satisfies the given conditions -/
theorem f_satisfies_conditions :
  (∃ k : ℝ, ∀ x : ℝ, f x = k * (x - 3)) ∧  -- y is directly proportional to x-3
  (f 5 = -4)                               -- When x = 5, y = -4
  := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1736_173625


namespace NUMINAMATH_CALUDE_min_value_of_a_l1736_173634

theorem min_value_of_a (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 25) →
  a ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1736_173634


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1736_173619

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 37/6 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1736_173619


namespace NUMINAMATH_CALUDE_coordinates_of_C_l1736_173621

-- Define the points
def A : ℝ × ℝ := (7, 2)
def B : ℝ × ℝ := (-1, 9)
def D : ℝ × ℝ := (2, 7)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC (isosceles triangle)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on line BC
  (D.1 - B.1) * (C.2 - B.2) = (D.2 - B.2) * (C.1 - B.1) ∧
  -- AD is perpendicular to BC (altitude condition)
  (A.1 - D.1) * (C.1 - B.1) + (A.2 - D.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem coordinates_of_C :
  ∃ (C : ℝ × ℝ), triangle_ABC C ∧ C = (5, 5) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_C_l1736_173621


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l1736_173616

/-- Proposition P: For any real number x, ax^2 + ax + 1 > 0 always holds -/
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0

/-- Proposition Q: The equation x^2 - x + a = 0 has real roots -/
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of a satisfying the given conditions -/
def range_of_a : Set ℝ := {a : ℝ | a < 0 ∨ (0 < a ∧ a < 4)}

theorem range_of_a_theorem :
  ∀ a : ℝ, (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l1736_173616


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1736_173633

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1736_173633


namespace NUMINAMATH_CALUDE_max_gross_profit_l1736_173697

/-- The gross profit function L(p) for a store selling goods --/
def L (p : ℝ) : ℝ := (8300 - 170*p - p^2)*(p - 20)

/-- The statement that L(p) achieves its maximum at p = 30 with a value of 23000 --/
theorem max_gross_profit :
  ∃ (p : ℝ), p > 0 ∧ L p = 23000 ∧ ∀ (q : ℝ), q > 0 → L q ≤ L p :=
sorry

end NUMINAMATH_CALUDE_max_gross_profit_l1736_173697


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1736_173614

/-- Proves that the speed of the first train is approximately 120.016 kmph given the conditions -/
theorem train_speed_calculation (length1 length2 speed2 time : ℝ) 
  (h1 : length1 = 290) 
  (h2 : length2 = 210.04)
  (h3 : speed2 = 80)
  (h4 : time = 9)
  : ∃ speed1 : ℝ, abs (speed1 - 120.016) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1736_173614


namespace NUMINAMATH_CALUDE_symmetry_across_x_axis_l1736_173612

def point_symmetrical_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_across_x_axis :
  let M : ℝ × ℝ := (1, 3)
  let N : ℝ × ℝ := (1, -3)
  point_symmetrical_to_x_axis M N :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_across_x_axis_l1736_173612


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1736_173604

theorem solve_linear_equation (x : ℝ) : 5 * x + 3 = 10 * x - 17 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1736_173604


namespace NUMINAMATH_CALUDE_equal_intercept_line_equations_equidistant_point_locus_l1736_173600

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  a : ℝ
  b : ℝ
  passes_through : a + b = 4
  equal_intercepts : a = b

/-- A point equidistant from two parallel lines -/
structure EquidistantPoint where
  x : ℝ
  y : ℝ
  equidistant : |4*x + 6*y - 10| = |4*x + 6*y + 8|

/-- Theorem for the equal intercept line -/
theorem equal_intercept_line_equations (l : EqualInterceptLine) :
  (∀ x y, y = 3*x) ∨ (∀ x y, y = -x + 4) :=
sorry

/-- Theorem for the locus of equidistant points -/
theorem equidistant_point_locus (p : EquidistantPoint) :
  4*p.x + 6*p.y - 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equations_equidistant_point_locus_l1736_173600


namespace NUMINAMATH_CALUDE_final_sum_theorem_l1736_173609

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  2 * (a + 5) + 2 * (b - 5) = 2 * S := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l1736_173609


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1736_173666

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (s : ℝ) (e : ℝ) : 
  s = 7 → e = 90 → (360 / e : ℝ) * s = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1736_173666


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l1736_173602

/-- A polynomial in two variables that represents a^2005 + b^2005 -/
def P : (ℝ → ℝ → ℝ) → Prop :=
  λ p => ∀ a b : ℝ, p (a + b) (a * b) = a^2005 + b^2005

/-- The sum of coefficients of a polynomial in two variables -/
def sum_of_coefficients (p : ℝ → ℝ → ℝ) : ℝ := p 1 1

theorem sum_of_coefficients_is_one (p : ℝ → ℝ → ℝ) (h : P p) : 
  sum_of_coefficients p = 1 := by
  sorry

#check sum_of_coefficients_is_one

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l1736_173602


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1736_173667

/-- A rectangular prism is a three-dimensional shape with 6 faces, 12 edges, and 8 vertices. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat
  faces_eq : faces = 6
  edges_eq : edges = 12
  vertices_eq : vertices = 8

/-- The sum of faces, edges, and vertices of a rectangular prism is 26. -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1736_173667


namespace NUMINAMATH_CALUDE_max_pyramid_volume_l1736_173680

/-- The maximum volume of a pyramid SABC with given conditions -/
theorem max_pyramid_volume (AB AC : ℝ) (sin_BAC : ℝ) (h : ℝ) :
  AB = 5 →
  AC = 8 →
  sin_BAC = 4/5 →
  h ≤ (5 * Real.sqrt 137 * Real.sqrt 3) / 8 →
  (1/3 : ℝ) * (1/2 * AB * AC * sin_BAC) * h ≤ 10 * Real.sqrt (137/3) :=
by sorry

end NUMINAMATH_CALUDE_max_pyramid_volume_l1736_173680


namespace NUMINAMATH_CALUDE_mary_garden_apples_l1736_173658

/-- The number of pies Mary wants to bake -/
def num_pies : ℕ := 10

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 8

/-- The number of additional apples Mary needs to buy -/
def apples_to_buy : ℕ := 30

/-- The number of apples Mary harvested from her garden -/
def apples_from_garden : ℕ := num_pies * apples_per_pie - apples_to_buy

theorem mary_garden_apples : apples_from_garden = 50 := by
  sorry

end NUMINAMATH_CALUDE_mary_garden_apples_l1736_173658


namespace NUMINAMATH_CALUDE_prism_volume_l1736_173656

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 56) (h2 : b * c = 63) (h3 : 2 * a * c = 72) :
  a * b * c = 504 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1736_173656


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_solution_l1736_173659

theorem arcsin_sin_eq_solution (x : ℝ) : 
  Real.arcsin (Real.sin x) = (3 * x) / 4 ∧ 
  -(π / 2) ≤ (3 * x) / 4 ∧ 
  (3 * x) / 4 ≤ π / 2 → 
  x = 0 := by
sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_solution_l1736_173659


namespace NUMINAMATH_CALUDE_problem_1_l1736_173696

theorem problem_1 (x y : ℝ) (h : x^2 + y^2 = 1) :
  x^6 + 3*x^2*y^2 + y^6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1736_173696


namespace NUMINAMATH_CALUDE_min_draws_for_18_l1736_173635

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The actual box contents -/
def boxContents : BallCounts :=
  { red := 30, green := 25, yellow := 22, blue := 15, white := 12, black := 6 }

/-- The main theorem -/
theorem min_draws_for_18 :
  minDraws boxContents 18 = 85 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_18_l1736_173635


namespace NUMINAMATH_CALUDE_longest_leg_of_smallest_triangle_l1736_173651

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse_def : hypotenuse = 2 * shorter_leg
  longer_leg_def : longer_leg = shorter_leg * Real.sqrt 3

/-- Represents a sequence of three 30-60-90 triangles -/
structure TriangleSequence where
  largest : Triangle30_60_90
  middle : Triangle30_60_90
  smallest : Triangle30_60_90
  sequence_property : 
    largest.longer_leg = middle.hypotenuse ∧
    middle.longer_leg = smallest.hypotenuse

theorem longest_leg_of_smallest_triangle 
  (seq : TriangleSequence) 
  (h : seq.largest.hypotenuse = 16) : 
  seq.smallest.longer_leg = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_leg_of_smallest_triangle_l1736_173651


namespace NUMINAMATH_CALUDE_pigeon_chicks_count_l1736_173622

/-- Proves that each pigeon has 6 chicks given the problem conditions -/
theorem pigeon_chicks_count :
  ∀ (total_pigeons : ℕ) (adult_pigeons : ℕ) (remaining_pigeons : ℕ),
    adult_pigeons = 40 →
    remaining_pigeons = 196 →
    (remaining_pigeons : ℚ) = 0.7 * total_pigeons →
    (total_pigeons - adult_pigeons) / adult_pigeons = 6 := by
  sorry


end NUMINAMATH_CALUDE_pigeon_chicks_count_l1736_173622


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1736_173661

theorem chess_tournament_games (num_players : ℕ) (total_games : ℕ) (games_per_pair : ℕ) : 
  num_players = 8 →
  total_games = 56 →
  total_games = (num_players * (num_players - 1) * games_per_pair) / 2 →
  games_per_pair = 2 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1736_173661


namespace NUMINAMATH_CALUDE_length_of_PQ_l1736_173637

/-- The circle C with center (3, 2) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}

/-- The line L defined by y = (3/4)x -/
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = (3/4) * p.1}

/-- The intersection points of C and L -/
def intersection := C ∩ L

/-- Assuming the intersection contains exactly two points -/
axiom two_intersection_points : ∃ P Q : ℝ × ℝ, P ≠ Q ∧ intersection = {P, Q}

/-- The length of the line segment PQ -/
noncomputable def PQ_length : ℝ := sorry

/-- The main theorem: The length of PQ is 4√6/5 -/
theorem length_of_PQ : PQ_length = 4 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_length_of_PQ_l1736_173637


namespace NUMINAMATH_CALUDE_product_of_eight_consecutive_integers_divisible_by_80_l1736_173688

theorem product_of_eight_consecutive_integers_divisible_by_80 (n : ℕ) : 
  80 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_eight_consecutive_integers_divisible_by_80_l1736_173688


namespace NUMINAMATH_CALUDE_remainder_4059_div_32_l1736_173665

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4059_div_32_l1736_173665


namespace NUMINAMATH_CALUDE_square_side_length_average_l1736_173636

theorem square_side_length_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 144) :
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l1736_173636


namespace NUMINAMATH_CALUDE_resident_price_proof_l1736_173644

/-- Calculates the ticket price for residents given the total attendees, number of residents,
    price for non-residents, and total revenue. -/
def resident_price (total_attendees : ℕ) (num_residents : ℕ) (non_resident_price : ℚ) (total_revenue : ℚ) : ℚ :=
  (total_revenue - (total_attendees - num_residents : ℚ) * non_resident_price) / num_residents

/-- Proves that the resident price is approximately $12.95 given the problem conditions. -/
theorem resident_price_proof :
  let total_attendees : ℕ := 586
  let num_residents : ℕ := 219
  let non_resident_price : ℚ := 17.95
  let total_revenue : ℚ := 9423.70
  abs (resident_price total_attendees num_residents non_resident_price total_revenue - 12.95) < 0.01 := by
  sorry

#eval resident_price 586 219 (17.95 : ℚ) (9423.70 : ℚ)

end NUMINAMATH_CALUDE_resident_price_proof_l1736_173644


namespace NUMINAMATH_CALUDE_junior_high_ten_total_games_l1736_173632

/-- Represents a basketball conference -/
structure BasketballConference where
  num_teams : ℕ
  intra_conference_games : ℕ
  non_conference_games : ℕ

/-- Calculates the total number of games in a season for a given basketball conference -/
def total_games (conf : BasketballConference) : ℕ :=
  (conf.num_teams.choose 2 * conf.intra_conference_games) + (conf.num_teams * conf.non_conference_games)

/-- The Junior High Ten conference -/
def junior_high_ten : BasketballConference :=
  { num_teams := 10
  , intra_conference_games := 3
  , non_conference_games := 5 }

theorem junior_high_ten_total_games :
  total_games junior_high_ten = 185 := by
  sorry


end NUMINAMATH_CALUDE_junior_high_ten_total_games_l1736_173632


namespace NUMINAMATH_CALUDE_first_month_sale_correct_l1736_173607

/-- Represents the sales data for a grocery shop -/
structure SalesData where
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the first month given the sales data -/
def calculate_first_month_sale (data : SalesData) : ℕ :=
  data.average * 6 - (data.month2 + data.month3 + data.month4 + data.month5 + data.month6)

/-- Theorem stating that the calculated first month sale is correct -/
theorem first_month_sale_correct (data : SalesData) 
  (h : data = { month2 := 6927, month3 := 6855, month4 := 7230, month5 := 6562, 
                month6 := 5091, average := 6500 }) : 
  calculate_first_month_sale data = 6335 := by
  sorry

#eval calculate_first_month_sale { month2 := 6927, month3 := 6855, month4 := 7230, 
                                   month5 := 6562, month6 := 5091, average := 6500 }

end NUMINAMATH_CALUDE_first_month_sale_correct_l1736_173607


namespace NUMINAMATH_CALUDE_sally_saturday_sandwiches_l1736_173685

/-- The number of sandwiches Sally eats on Saturday -/
def sandwiches_saturday : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sandwiches_sunday : ℕ := 1

/-- The number of pieces of bread used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- Theorem stating that Sally eats 2 sandwiches on Saturday -/
theorem sally_saturday_sandwiches :
  sandwiches_saturday = (total_bread - sandwiches_sunday * bread_per_sandwich) / bread_per_sandwich :=
by sorry

end NUMINAMATH_CALUDE_sally_saturday_sandwiches_l1736_173685


namespace NUMINAMATH_CALUDE_video_game_lives_l1736_173654

theorem video_game_lives (initial_lives hard_part_lives next_level_lives : ℝ) :
  initial_lives + hard_part_lives + next_level_lives =
  initial_lives + (hard_part_lives + next_level_lives) :=
by
  sorry

-- Example usage
def tiffany_game (initial_lives hard_part_lives next_level_lives : ℝ) : ℝ :=
  initial_lives + hard_part_lives + next_level_lives

#eval tiffany_game 43.0 14.0 27.0

end NUMINAMATH_CALUDE_video_game_lives_l1736_173654


namespace NUMINAMATH_CALUDE_initial_bananas_count_l1736_173652

/-- The number of bananas in each package -/
def package_size : ℕ := 13

/-- The number of bananas added to the pile -/
def bananas_added : ℕ := 7

/-- The total number of bananas after adding -/
def total_bananas : ℕ := 9

/-- The initial number of bananas on the desk -/
def initial_bananas : ℕ := total_bananas - bananas_added

theorem initial_bananas_count : initial_bananas = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l1736_173652


namespace NUMINAMATH_CALUDE_total_carrots_l1736_173683

theorem total_carrots (sally_carrots fred_carrots : ℕ) 
  (h1 : sally_carrots = 6) 
  (h2 : fred_carrots = 4) : 
  sally_carrots + fred_carrots = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l1736_173683


namespace NUMINAMATH_CALUDE_solve_equation_l1736_173699

theorem solve_equation (x t : ℝ) : 
  (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1736_173699


namespace NUMINAMATH_CALUDE_second_number_value_l1736_173668

theorem second_number_value (x y z : ℚ) : 
  x + y + z = 120 ∧ 
  x / y = 3 / 4 ∧ 
  y / z = 4 / 7 →
  y = 240 / 7 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1736_173668


namespace NUMINAMATH_CALUDE_geometric_progression_unique_p_l1736_173608

theorem geometric_progression_unique_p : 
  ∃! (p : ℝ), p > 0 ∧ (2 * Real.sqrt p) ^ 2 = (p - 2) * (-3 - p) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_unique_p_l1736_173608


namespace NUMINAMATH_CALUDE_polynomial_equality_solutions_l1736_173694

theorem polynomial_equality_solutions : 
  ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → 
  ((a = 20 ∧ b = -6 ∧ c = -6) ∨ (a = 29 ∧ b = -9 ∧ c = -12)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_solutions_l1736_173694


namespace NUMINAMATH_CALUDE_train_length_l1736_173620

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 27 ∧ time = 20 → speed * time * (5 / 18) = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1736_173620


namespace NUMINAMATH_CALUDE_aku_birthday_cookies_l1736_173692

/-- Given the number of friends, packages, and cookies per package, 
    calculate the number of cookies each child will eat. -/
def cookies_per_child (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  (packages * cookies_per_package) / (friends + 1)

/-- Theorem stating that under the given conditions, each child will eat 15 cookies. -/
theorem aku_birthday_cookies : 
  cookies_per_child 4 3 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_aku_birthday_cookies_l1736_173692


namespace NUMINAMATH_CALUDE_xy_product_l1736_173645

theorem xy_product (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_product_l1736_173645


namespace NUMINAMATH_CALUDE_rogers_money_l1736_173640

theorem rogers_money (initial amount_spent final : ℤ) 
  (h1 : initial = 45)
  (h2 : amount_spent = 20)
  (h3 : final = 71) :
  final - (initial - amount_spent) = 46 := by
  sorry

end NUMINAMATH_CALUDE_rogers_money_l1736_173640


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1736_173613

/-- Given a rectangle with sides L and W, prove that if one side is measured 5% in excess
    and the calculated area has an error of 0.8%, the other side must be measured 4% in deficit. -/
theorem rectangle_measurement_error (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let L' := 1.05 * L
  let W' := W * (1 - p)
  let A := L * W
  let A' := L' * W'
  A' = 1.008 * A →
  p = 0.04
  := by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1736_173613


namespace NUMINAMATH_CALUDE_functions_with_inverses_l1736_173643

-- Define the four functions
def function_A : ℝ → ℝ := sorry
def function_B : ℝ → ℝ := sorry
def function_C : ℝ → ℝ := sorry
def function_D : ℝ → ℝ := sorry

-- Define the property of being a straight line through the origin
def is_straight_line_through_origin (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a downward-opening parabola with vertex at (0, 1)
def is_downward_parabola_vertex_0_1 (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being an upper semicircle with radius 3 centered at origin
def is_upper_semicircle_radius_3 (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a piecewise linear function as described
def is_piecewise_linear_as_described (f : ℝ → ℝ) : Prop := sorry

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := sorry

theorem functions_with_inverses :
  is_straight_line_through_origin function_A ∧
  is_downward_parabola_vertex_0_1 function_B ∧
  is_upper_semicircle_radius_3 function_C ∧
  is_piecewise_linear_as_described function_D →
  has_inverse function_A ∧
  ¬ has_inverse function_B ∧
  ¬ has_inverse function_C ∧
  has_inverse function_D := by sorry

end NUMINAMATH_CALUDE_functions_with_inverses_l1736_173643


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l1736_173626

/-- The function f is monotonic on the interval [1, 2] if and only if a is in the specified range -/
theorem monotonic_quadratic_function (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => x^2 + (2*a + 1)*x + 1)) ↔ 
  a ∈ Set.Iic (-3/2) ∪ Set.Ioi (-5/2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l1736_173626


namespace NUMINAMATH_CALUDE_min_value_sum_squared_fractions_l1736_173648

theorem min_value_sum_squared_fractions (x y z : ℕ+) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y : ℝ) + (x^2 + z^2) / (x + z : ℝ) + (y^2 + z^2) / (y + z : ℝ) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_fractions_l1736_173648


namespace NUMINAMATH_CALUDE_largest_multiple_of_12_negation_greater_than_neg_150_l1736_173684

theorem largest_multiple_of_12_negation_greater_than_neg_150 :
  ∀ n : ℤ, 12 ∣ n ∧ -n > -150 → n ≤ 144 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_12_negation_greater_than_neg_150_l1736_173684


namespace NUMINAMATH_CALUDE_set_union_problem_l1736_173691

theorem set_union_problem (x y : ℝ) :
  let A : Set ℝ := {x, y}
  let B : Set ℝ := {x + 1, 5}
  A ∩ B = {2} →
  A ∪ B = {1, 2, 5} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1736_173691


namespace NUMINAMATH_CALUDE_vectors_collinear_l1736_173682

def a : Fin 3 → ℝ := ![3, -1, 6]
def b : Fin 3 → ℝ := ![5, 7, 10]
def c₁ : Fin 3 → ℝ := λ i => 4 * a i - 2 * b i
def c₂ : Fin 3 → ℝ := λ i => b i - 2 * a i

theorem vectors_collinear : ∃ (k : ℝ), k ≠ 0 ∧ (∀ i : Fin 3, c₁ i = k * c₂ i) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l1736_173682


namespace NUMINAMATH_CALUDE_circle_area_when_radius_equals_six_times_reciprocal_of_circumference_l1736_173686

theorem circle_area_when_radius_equals_six_times_reciprocal_of_circumference :
  ∀ (r : ℝ), r > 0 → (6 * (1 / (2 * Real.pi * r)) = r) → (Real.pi * r^2 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_when_radius_equals_six_times_reciprocal_of_circumference_l1736_173686


namespace NUMINAMATH_CALUDE_fourth_student_guess_l1736_173698

def jellybean_guess (first_guess : ℕ) : ℕ :=
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let average := (first_guess + second_guess + third_guess) / 3
  average + 25

theorem fourth_student_guess :
  jellybean_guess 100 = 525 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_guess_l1736_173698


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_evaluate_expression_2_l1736_173653

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by sorry

-- Problem 2
theorem simplify_and_evaluate_expression_2 (x y : ℝ) :
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = x * y^2 + x * y := by sorry

theorem evaluate_expression_2 :
  let x : ℝ := -3
  let y : ℝ := -2
  3 * x^2 * y - (2 * x * y^2 - 2 * (x * y - 1.5 * x^2 * y) + x * y) + 3 * x * y^2 = -6 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_evaluate_expression_2_l1736_173653


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1736_173670

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1736_173670


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l1736_173679

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 < x ∧ x ≤ 8}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 8} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 9} := by sorry

-- Theorem for the range of a when A ∩ C is non-empty
theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l1736_173679


namespace NUMINAMATH_CALUDE_simplify_polynomial_l1736_173639

theorem simplify_polynomial (x : ℝ) : 3 * (3 * x^2 + 9 * x - 4) - 2 * (x^2 + 7 * x - 14) = 7 * x^2 + 13 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l1736_173639


namespace NUMINAMATH_CALUDE_book_cost_price_l1736_173672

/-- The cost price of a book satisfying given profit conditions -/
theorem book_cost_price : ∃ (C : ℝ), 
  (C > 0) ∧ 
  (1.15 * C = 1.10 * C + 100) ∧ 
  (C = 2000) := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l1736_173672


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1736_173676

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_others : ℕ) 
  (h1 : total_workers = 28) 
  (h2 : num_technicians = 7) 
  (h3 : avg_salary_technicians = 14000) 
  (h4 : avg_salary_others = 6000) :
  (num_technicians * avg_salary_technicians + 
   (total_workers - num_technicians) * avg_salary_others) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1736_173676


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1736_173618

theorem quadratic_equation_roots (k : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 14 * x + k = 0 ↔ x = (-14 + Real.sqrt 10) / 4 ∨ x = (-14 - Real.sqrt 10) / 4) →
  k = 93 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1736_173618


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1736_173677

theorem rectangle_max_area (x y P D : ℝ) (h1 : P = 2*x + 2*y) (h2 : D^2 = x^2 + y^2) 
  (h3 : P = 14) (h4 : D = 5) :
  ∃ (A : ℝ), A = x * y ∧ A ≤ 49/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ = 49/4 := by
  sorry

#check rectangle_max_area

end NUMINAMATH_CALUDE_rectangle_max_area_l1736_173677


namespace NUMINAMATH_CALUDE_horner_method_V3_l1736_173690

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

def horner_V3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let a₅ := 1
  let a₄ := 2
  let a₃ := 1
  let a₂ := -1
  let a₁ := 3
  let a₀ := -5
  let V₀ := a₅
  let V₁ := V₀ * x + a₄
  let V₂ := V₁ * x + a₃
  V₂ * x + a₂

theorem horner_method_V3 :
  horner_V3 f 5 = 179 := by sorry

end NUMINAMATH_CALUDE_horner_method_V3_l1736_173690


namespace NUMINAMATH_CALUDE_joe_watching_schedule_l1736_173605

/-- The number of episodes Joe needs to watch per day to catch up with the season premiere. -/
def episodes_per_day (days_until_premiere : ℕ) (num_seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  (num_seasons * episodes_per_season) / days_until_premiere

/-- Theorem stating that Joe needs to watch 6 episodes per day. -/
theorem joe_watching_schedule :
  episodes_per_day 10 4 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_joe_watching_schedule_l1736_173605


namespace NUMINAMATH_CALUDE_cookie_distribution_l1736_173617

theorem cookie_distribution (total : ℚ) (blue green orange red : ℚ) : 
  blue + green + orange + red = total →
  blue + green + orange = 11 / 12 * total →
  red = 1 / 12 * total →
  blue = 1 / 6 * total →
  green = 5 / 12 * total →
  orange = 1 / 3 * total :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1736_173617


namespace NUMINAMATH_CALUDE_vitya_wins_l1736_173662

/-- Represents a point on the infinite grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the game state --/
structure GameState where
  marked_points : List GridPoint
  current_player : Bool  -- true for Kolya, false for Vitya

/-- Checks if a list of points forms a convex polygon --/
def is_convex_polygon (points : List GridPoint) : Prop :=
  sorry

/-- Checks if a move is valid according to the game rules --/
def is_valid_move (state : GameState) (new_point : GridPoint) : Prop :=
  is_convex_polygon (new_point :: state.marked_points)

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Option GridPoint

/-- Checks if a strategy is winning for a player --/
def is_winning_strategy (strategy : Strategy) (player : Bool) : Prop :=
  sorry

theorem vitya_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy false :=
sorry

end NUMINAMATH_CALUDE_vitya_wins_l1736_173662


namespace NUMINAMATH_CALUDE_expected_total_rolls_leap_year_l1736_173631

/-- Represents the outcome of rolling an eight-sided die -/
inductive DieRoll
| one | two | three | four | five | six | seven | eight

/-- Defines if a roll is a perfect square (1 or 4) -/
def isPerfectSquare (roll : DieRoll) : Prop :=
  roll = DieRoll.one ∨ roll = DieRoll.four

/-- Calculates the probability of rolling a perfect square -/
def probPerfectSquare : ℚ := 1/4

/-- Calculates the probability of not rolling a perfect square -/
def probNotPerfectSquare : ℚ := 3/4

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- The expected number of rolls per day -/
noncomputable def expectedRollsPerDay : ℚ := 4/3

/-- Theorem: The expected total number of rolls in a leap year is 488 -/
theorem expected_total_rolls_leap_year :
  (expectedRollsPerDay * daysInLeapYear : ℚ) = 488 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rolls_leap_year_l1736_173631


namespace NUMINAMATH_CALUDE_problem_solution_l1736_173603

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y ^ 2) 
  (h3 : x / 5 = 5 * y) : 
  x = 625 / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1736_173603


namespace NUMINAMATH_CALUDE_distance_traveled_l1736_173681

-- Define the speed in miles per hour
def speed : ℝ := 16

-- Define the time in hours
def time : ℝ := 5

-- Theorem to prove the distance traveled
theorem distance_traveled : speed * time = 80 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l1736_173681


namespace NUMINAMATH_CALUDE_line_intersection_l1736_173623

theorem line_intersection :
  ∃! p : ℚ × ℚ, 8 * p.1 - 3 * p.2 = 20 ∧ 9 * p.1 + 2 * p.2 = 17 :=
by
  use (91/43, 61/43)
  sorry

end NUMINAMATH_CALUDE_line_intersection_l1736_173623


namespace NUMINAMATH_CALUDE_square_equation_solution_l1736_173628

theorem square_equation_solution (x : ℝ) : (x + 3)^2 = 121 ↔ x = 8 ∨ x = -14 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1736_173628


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1736_173664

/-- Given two perpendicular lines, prove that the distance from (m, 1) to the y-axis is 0 or 5 -/
theorem distance_to_y_axis (m : ℝ) : 
  (∃ x y, mx - (x + 2) * y + 2 = 0 ∧ 3 * x - m * y - 1 = 0) →  -- Lines exist
  (∀ x₁ y₁ x₂ y₂, mx₁ - (x₁ + 2) * y₁ + 2 = 0 ∧ 3 * x₂ - m * y₂ - 1 = 0 → 
    (m * 3 + m * (m + 2) = 0)) →  -- Lines are perpendicular
  (abs m = 0 ∨ abs m = 5) :=  -- Distance from (m, 1) to y-axis is 0 or 5
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1736_173664


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1736_173624

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem z_in_first_quadrant (z₁ z₂ : ℂ) 
  (h₁ : complex_to_point z₁ = (2, 3))
  (h₂ : z₂ = -1 + 2*Complex.I) :
  in_first_quadrant (complex_to_point (z₁ - z₂)) := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1736_173624


namespace NUMINAMATH_CALUDE_pizza_cost_per_piece_l1736_173657

/-- 
Given that Luigi bought 4 pizzas for $80 and each pizza was cut into 5 pieces,
prove that each piece of pizza costs $4.
-/
theorem pizza_cost_per_piece 
  (num_pizzas : ℕ) 
  (total_cost : ℚ) 
  (pieces_per_pizza : ℕ) 
  (h1 : num_pizzas = 4) 
  (h2 : total_cost = 80) 
  (h3 : pieces_per_pizza = 5) : 
  total_cost / (num_pizzas * pieces_per_pizza : ℚ) = 4 := by
sorry

end NUMINAMATH_CALUDE_pizza_cost_per_piece_l1736_173657


namespace NUMINAMATH_CALUDE_parallel_condition_necessary_not_sufficient_l1736_173627

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define parallelism for two lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1 ≠ l2

/-- The first line: 2x + ay - 1 = 0 -/
def line1 (a : ℝ) : Line2D :=
  { a := 2, b := a, c := -1 }

/-- The second line: bx + 2y - 2 = 0 -/
def line2 (b : ℝ) : Line2D :=
  { a := b, b := 2, c := -2 }

/-- The main theorem -/
theorem parallel_condition_necessary_not_sufficient :
  (∀ a b : ℝ, parallel (line1 a) (line2 b) → a * b = 4) ∧
  ¬(∀ a b : ℝ, a * b = 4 → parallel (line1 a) (line2 b)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_condition_necessary_not_sufficient_l1736_173627


namespace NUMINAMATH_CALUDE_total_pens_bought_l1736_173610

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) 
  (h1 : pen_cost > 10)
  (h2 : masha_spent = 357)
  (h3 : olya_spent = 441)
  (h4 : masha_spent % pen_cost = 0)
  (h5 : olya_spent % pen_cost = 0) :
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
sorry

end NUMINAMATH_CALUDE_total_pens_bought_l1736_173610


namespace NUMINAMATH_CALUDE_madeline_unused_crayons_l1736_173678

theorem madeline_unused_crayons :
  let box1to3 := 3 * 30 * (1/2 : ℚ)
  let box4to5 := 2 * 36 * (3/4 : ℚ)
  let box6to7 := 2 * 40 * (2/5 : ℚ)
  let box8 := 1 * 45 * (5/9 : ℚ)
  let box9to10 := 2 * 48 * (7/8 : ℚ)
  let box11 := 1 * 27 * (5/6 : ℚ)
  let box12 := 1 * 54 * (1/2 : ℚ)
  let total_unused := box1to3 + box4to5 + box6to7 + box8 + box9to10 + box11 + box12
  ⌊total_unused⌋ = 289 :=
by sorry

end NUMINAMATH_CALUDE_madeline_unused_crayons_l1736_173678


namespace NUMINAMATH_CALUDE_cake_ingredient_difference_l1736_173642

/-- Given a cake recipe and partially added ingredients, calculate the difference
    between remaining flour and required sugar. -/
theorem cake_ingredient_difference
  (total_sugar : ℕ)
  (total_flour : ℕ)
  (added_flour : ℕ)
  (h1 : total_sugar = 6)
  (h2 : total_flour = 9)
  (h3 : added_flour = 2)
  : total_flour - added_flour - total_sugar = 1 := by
  sorry

end NUMINAMATH_CALUDE_cake_ingredient_difference_l1736_173642


namespace NUMINAMATH_CALUDE_meeting_time_percentage_l1736_173630

def total_work_day : ℕ := 10 -- in hours
def lunch_break : ℕ := 1 -- in hours
def first_meeting : ℕ := 30 -- in minutes
def second_meeting : ℕ := 3 * first_meeting -- in minutes

def actual_work_minutes : ℕ := (total_work_day - lunch_break) * 60
def total_meeting_minutes : ℕ := first_meeting + second_meeting

def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (actual_work_minutes : ℚ) * 100

theorem meeting_time_percentage : 
  ∃ (ε : ℚ), abs (meeting_percentage - 22) < ε ∧ ε > 0 ∧ ε < 1 :=
sorry

end NUMINAMATH_CALUDE_meeting_time_percentage_l1736_173630


namespace NUMINAMATH_CALUDE_base9_725_to_base3_l1736_173660

/-- Converts a base-9 digit to its two-digit base-3 representation -/
def base9_to_base3_digit (d : ℕ) : ℕ × ℕ :=
  (d / 3, d % 3)

/-- Converts a base-9 number to its base-3 representation -/
def base9_to_base3 (n : ℕ) : List ℕ :=
  let digits := n.digits 9
  List.join (digits.map (fun d => let (q, r) := base9_to_base3_digit d; [q, r]))

theorem base9_725_to_base3 :
  base9_to_base3 725 = [2, 1, 0, 2, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base9_725_to_base3_l1736_173660


namespace NUMINAMATH_CALUDE_ab_ab2_a_inequality_l1736_173675

theorem ab_ab2_a_inequality (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_ab_ab2_a_inequality_l1736_173675


namespace NUMINAMATH_CALUDE_last_digits_of_11_power_l1736_173673

theorem last_digits_of_11_power (n : ℕ) (h : n ≥ 1) :
  11^(10^n) ≡ 6 * 10^(n+1) + 1 [MOD 10^(n+2)] := by
sorry

end NUMINAMATH_CALUDE_last_digits_of_11_power_l1736_173673


namespace NUMINAMATH_CALUDE_calculation_proof_l1736_173629

theorem calculation_proof : (2 - Real.pi) ^ 0 - 2⁻¹ + Real.cos (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1736_173629


namespace NUMINAMATH_CALUDE_round_23_36_to_nearest_tenth_l1736_173647

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest tenth. -/
def roundToNearestTenth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The given repeating decimal 23.363636... -/
def givenNumber : RepeatingDecimal :=
  { integerPart := 23, repeatingPart := 36 }

theorem round_23_36_to_nearest_tenth :
  roundToNearestTenth givenNumber = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_round_23_36_to_nearest_tenth_l1736_173647


namespace NUMINAMATH_CALUDE_N_divisible_by_1980_l1736_173655

/-- The number formed by concatenating all two-digit numbers from 19 to 80 inclusive -/
def N : ℕ := sorry

/-- N is divisible by 1980 -/
theorem N_divisible_by_1980 : 1980 ∣ N := by sorry

end NUMINAMATH_CALUDE_N_divisible_by_1980_l1736_173655


namespace NUMINAMATH_CALUDE_three_planes_max_parts_l1736_173641

/-- The maximum number of parts that can be created by n planes in 3D space -/
def maxParts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => maxParts k + k + 1

/-- Theorem: Three planes can divide 3D space into at most 8 parts -/
theorem three_planes_max_parts :
  maxParts 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_max_parts_l1736_173641


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1736_173638

theorem complex_equation_sum (a b : ℝ) (h : (3 * b : ℂ) + (2 * a - 2) * Complex.I = 1 - Complex.I) : 
  a + b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1736_173638


namespace NUMINAMATH_CALUDE_lateral_edges_coplanar_iff_height_eq_edge_l1736_173674

/-- A cube with regular 4-sided pyramids on each face -/
structure PyramidCube where
  -- Edge length of the cube
  a : ℝ
  -- Height of the pyramids
  h : ℝ
  -- Assumption that a and h are positive
  a_pos : 0 < a
  h_pos : 0 < h

/-- The condition for lateral edges to lie in the same plane -/
def lateral_edges_coplanar (cube : PyramidCube) : Prop :=
  cube.h = cube.a

/-- Theorem stating the condition for lateral edges to be coplanar -/
theorem lateral_edges_coplanar_iff_height_eq_edge (cube : PyramidCube) :
  lateral_edges_coplanar cube ↔ cube.h = cube.a :=
sorry


end NUMINAMATH_CALUDE_lateral_edges_coplanar_iff_height_eq_edge_l1736_173674


namespace NUMINAMATH_CALUDE_simple_interest_rate_change_l1736_173606

/-- Given the conditions of a simple interest problem, prove that the new interest rate is 8% -/
theorem simple_interest_rate_change
  (P : ℝ) (R1 T1 SI T2 : ℝ)
  (h1 : R1 = 5)
  (h2 : T1 = 8)
  (h3 : SI = 840)
  (h4 : T2 = 5)
  (h5 : P = (SI * 100) / (R1 * T1))
  (h6 : SI = (P * R1 * T1) / 100)
  (h7 : SI = (P * R2 * T2) / 100)
  : R2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_change_l1736_173606


namespace NUMINAMATH_CALUDE_range_of_a_l1736_173663

/-- The range of a given the conditions in the problem -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) ↔ ¬((x < 1/2) ∨ (1 < x))) →
  (∀ x, ((x - a) * (x - a - 1) ≤ 0) ↔ (a ≤ x ∧ x ≤ a + 1)) →
  (∀ x, ¬((1/2 ≤ x ∧ x ≤ 1)) → ¬((x - a) * (x - a - 1) ≤ 0)) →
  (∃ x, ¬((1/2 ≤ x ∧ x ≤ 1)) ∧ ((x - a) * (x - a - 1) ≤ 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1736_173663


namespace NUMINAMATH_CALUDE_integral_f_minus_one_to_pi_l1736_173649

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 1 else Real.cos x

theorem integral_f_minus_one_to_pi :
  ∫ x in (-1)..(Real.pi), f x = 1 := by sorry

end NUMINAMATH_CALUDE_integral_f_minus_one_to_pi_l1736_173649


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l1736_173601

/-- For z = 1 + i and a ∈ ℝ, if (1 - ai) / z is a pure imaginary number, then a = 1 -/
theorem complex_pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := 1 + I
  (((1 : ℂ) - a * I) / z).re = 0 → (((1 : ℂ) - a * I) / z).im ≠ 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l1736_173601
