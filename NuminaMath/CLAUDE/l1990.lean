import Mathlib

namespace infinite_equal_terms_l1990_199085

/-- An infinite sequence with two ends satisfying the given recurrence relation -/
def InfiniteSequence := ℤ → ℝ

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (a : InfiniteSequence) : Prop :=
  ∀ k : ℤ, a k = (1/4) * (a (k-1) + a (k+1))

theorem infinite_equal_terms
  (a : InfiniteSequence)
  (h_recurrence : SatisfiesRecurrence a)
  (h_equal : ∃ k p : ℤ, k < p ∧ a k = a p) :
  ∀ n : ℕ, ∃ k p : ℤ, k < p ∧ a (k - n) = a (p + n) :=
sorry

end infinite_equal_terms_l1990_199085


namespace angle_Y_is_50_l1990_199087

-- Define the angles in the geometric figure
def angle_X : ℝ := 120
def angle_Y : ℝ := 50
def angle_Z : ℝ := 180 - angle_X

-- Theorem statement
theorem angle_Y_is_50 : 
  angle_X = 120 →
  angle_Y = 50 →
  angle_Z = 180 - angle_X →
  angle_Y = 50 := by
  sorry

end angle_Y_is_50_l1990_199087


namespace binary_addition_subtraction_l1990_199015

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a binary representation (list of bits). -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- The main theorem to prove -/
theorem binary_addition_subtraction :
  let a := [true, false, true, true]   -- 1101₂
  let b := [true, true, true]          -- 111₂
  let c := [false, true, false, true]  -- 1010₂
  let d := [true, false, false, true]  -- 1001₂
  let result := [true, true, true, false, false, true] -- 100111₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d =
  binary_to_nat result :=
by
  sorry


end binary_addition_subtraction_l1990_199015


namespace isosceles_triangle_quadratic_roots_l1990_199055

/-- 
Given an isosceles triangle with side lengths m, n, and 4, where m and n are 
roots of x^2 - 6x + k + 2 = 0, prove that k = 7 or k = 6.
-/
theorem isosceles_triangle_quadratic_roots (m n k : ℝ) : 
  (m > 0 ∧ n > 0) →  -- m and n are positive (side lengths)
  (m = n ∨ m = 4 ∨ n = 4) →  -- isosceles condition
  (m ≠ n ∨ m ≠ 4) →  -- not equilateral
  m^2 - 6*m + k + 2 = 0 →  -- m is a root
  n^2 - 6*n + k + 2 = 0 →  -- n is a root
  k = 7 ∨ k = 6 := by
  sorry


end isosceles_triangle_quadratic_roots_l1990_199055


namespace perpendicular_probability_l1990_199029

/-- A square is a shape with 4 vertices -/
structure Square where
  vertices : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 4

/-- A line in a square is defined by two distinct vertices -/
structure Line (s : Square) where
  v1 : s.vertices
  v2 : s.vertices
  distinct : v1 ≠ v2

/-- Two lines are perpendicular if they form a right angle -/
def perpendicular (s : Square) (l1 l2 : Line s) : Prop := sorry

/-- The total number of possible line pairs in a square -/
def total_line_pairs (s : Square) : ℕ := sorry

/-- The number of perpendicular line pairs in a square -/
def perpendicular_line_pairs (s : Square) : ℕ := sorry

/-- The theorem to be proved -/
theorem perpendicular_probability (s : Square) : 
  (perpendicular_line_pairs s : ℚ) / (total_line_pairs s : ℚ) = 5 / 18 := sorry

end perpendicular_probability_l1990_199029


namespace students_in_cars_l1990_199069

def total_students : ℕ := 375
def num_buses : ℕ := 7
def students_per_bus : ℕ := 53

theorem students_in_cars : 
  total_students - (num_buses * students_per_bus) = 4 := by
  sorry

end students_in_cars_l1990_199069


namespace statement_I_statement_II_statement_III_l1990_199084

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Statement I
theorem statement_I : ∀ x : ℝ, floor (x + 1) = floor x + 1 := by sorry

-- Statement II (negation)
theorem statement_II : ∃ x y : ℝ, ∃ k : ℤ, floor (x + y + k) ≠ floor x + floor y + k := by sorry

-- Statement III (negation)
theorem statement_III : ∃ x y : ℝ, floor (x * y) ≠ floor x * floor y := by sorry

end statement_I_statement_II_statement_III_l1990_199084


namespace linear_function_proof_l1990_199078

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem linear_function_proof (f : ℝ → ℝ) 
  (h1 : is_linear f) 
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 := by sorry

end linear_function_proof_l1990_199078


namespace trajectory_is_parabola_l1990_199025

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the directrix line
def directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the property of the moving circle
def circle_property (center : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  (center.1 - F.1)^2 + (center.2 - F.2)^2 = r^2 ∧
  ∃ (p : ℝ × ℝ), p ∈ directrix ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ (center : ℝ × ℝ), circle_property center → center.2^2 = 4 * center.1 :=
sorry

end trajectory_is_parabola_l1990_199025


namespace complex_quadrant_l1990_199071

theorem complex_quadrant (z : ℂ) (h : (z - 1) * Complex.I = 1 + 2 * Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end complex_quadrant_l1990_199071


namespace rectangular_field_area_l1990_199063

theorem rectangular_field_area (m : ℕ) : 
  (3 * m + 8) * (m - 3) = 76 → m = 4 := by
sorry

end rectangular_field_area_l1990_199063


namespace trigonometric_properties_l1990_199017

theorem trigonometric_properties :
  (∀ α : Real, 0 < α ∧ α < Real.pi / 2 → Real.sin α > 0) ∧
  (∃ α : Real, 0 < α ∧ α < Real.pi / 2 ∧ Real.cos (2 * α) > 0) := by
  sorry

end trigonometric_properties_l1990_199017


namespace club_equation_solution_l1990_199072

def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 5

theorem club_equation_solution :
  ∃! A : ℝ, club A 4 = 58 ∧ A = 15 := by sorry

end club_equation_solution_l1990_199072


namespace first_term_is_seven_l1990_199080

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 2) + (-1)^n * a n = 3 * n - 1

/-- The sum of the first 16 terms of the sequence equals 540 -/
def SumCondition (a : ℕ → ℚ) : Prop :=
  (Finset.range 16).sum a = 540

/-- The theorem stating that a₁ = 7 for the given conditions -/
theorem first_term_is_seven
    (a : ℕ → ℚ)
    (h_recurrence : RecurrenceSequence a)
    (h_sum : SumCondition a) :
    a 1 = 7 := by
  sorry

end first_term_is_seven_l1990_199080


namespace min_phi_for_even_sine_l1990_199027

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem min_phi_for_even_sine (ω φ : ℝ) (h_omega : ω ≠ 0) (h_phi : φ > 0) 
  (h_even : IsEven (fun x ↦ 2 * Real.sin (ω * x + φ))) :
  ∃ (k : ℤ), φ = k * Real.pi + Real.pi / 2 ∧ 
  ∀ (m : ℤ), (m * Real.pi + Real.pi / 2 > 0) → (k * Real.pi + Real.pi / 2 ≤ m * Real.pi + Real.pi / 2) :=
sorry

end min_phi_for_even_sine_l1990_199027


namespace regular_hexagonal_prism_sum_l1990_199052

/-- A regular hexagonal prism -/
structure RegularHexagonalPrism where
  /-- The number of faces of the prism -/
  faces : ℕ
  /-- The number of edges of the prism -/
  edges : ℕ
  /-- The number of vertices of the prism -/
  vertices : ℕ

/-- The sum of faces, edges, and vertices of a regular hexagonal prism is 38 -/
theorem regular_hexagonal_prism_sum (prism : RegularHexagonalPrism) :
  prism.faces + prism.edges + prism.vertices = 38 := by
  sorry

end regular_hexagonal_prism_sum_l1990_199052


namespace inequality_solution_set_l1990_199037

theorem inequality_solution_set (x : ℝ) :
  x ≠ 2 ∧ x ≠ -9/2 →
  ((x + 1) / (x + 2) > (3*x + 4) / (2*x + 9)) ↔
  (x ∈ Set.Ioo (-9/2 : ℝ) (-2) ∪ Set.Ioo ((1 - Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2)) :=
by sorry

end inequality_solution_set_l1990_199037


namespace largest_beta_exponent_l1990_199018

open Real

/-- Given a sequence of points in a plane with specific distance properties, 
    this theorem proves the largest possible exponent β for which r_n ≥ Cn^β holds. -/
theorem largest_beta_exponent 
  (O : ℝ × ℝ) 
  (P : ℕ → ℝ × ℝ) 
  (r : ℕ → ℝ) 
  (α : ℝ) 
  (h_alpha : 0 < α ∧ α < 1)
  (h_distance : ∀ n m : ℕ, n ≠ m → dist (P n) (P m) ≥ (r n) ^ α)
  (h_r_increasing : ∀ n : ℕ, r n ≤ r (n + 1))
  (h_r_def : ∀ n : ℕ, dist O (P n) = r n) :
  ∃ (C : ℝ) (h_C : C > 0), ∀ n : ℕ, r n ≥ C * n ^ (1 / (2 * (1 - α))) ∧ 
  ∀ β : ℝ, (∃ (D : ℝ) (h_D : D > 0), ∀ n : ℕ, r n ≥ D * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end largest_beta_exponent_l1990_199018


namespace union_A_B_l1990_199026

/-- Set A is defined as the set of real numbers between -2 and 3 inclusive -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

/-- Set B is defined as the set of positive real numbers -/
def B : Set ℝ := {x | x > 0}

/-- The union of sets A and B is equal to the set of real numbers greater than or equal to -2 -/
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end union_A_B_l1990_199026


namespace arc_length_for_given_angle_l1990_199096

theorem arc_length_for_given_angle (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 7) :
  r * α = 2 * π / 7 := by
  sorry

end arc_length_for_given_angle_l1990_199096


namespace hyperbola_standard_equation_l1990_199038

/-- A hyperbola is defined by its standard equation parameters a and b,
    where the equation is (x²/a² - y²/b² = 1) or (y²/a² - x²/b² = 1) -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  isVertical : Bool

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space, defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a hyperbola -/
def pointOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  if h.isVertical then
    p.y^2 / h.a^2 - p.x^2 / h.b^2 = 1
  else
    p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a line is an asymptote of a hyperbola -/
def isAsymptote (h : Hyperbola) (l : Line) : Prop :=
  if h.isVertical then
    l.slope = h.a / h.b ∨ l.slope = -h.a / h.b
  else
    l.slope = h.b / h.a ∨ l.slope = -h.b / h.a

theorem hyperbola_standard_equation
  (h : Hyperbola)
  (p : Point)
  (l : Line)
  (h_point : pointOnHyperbola h p)
  (h_asymptote : isAsymptote h l)
  (h_p_coords : p.x = 1 ∧ p.y = 2 * Real.sqrt 2)
  (h_l_equation : l.slope = 2 ∧ l.yIntercept = 0) :
  h.a = 2 ∧ h.b = 1 ∧ h.isVertical = true :=
sorry

end hyperbola_standard_equation_l1990_199038


namespace range_of_a_l1990_199088

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 2| - |x - 1| ≥ a^3 - 4*a^2 - 3) → a ≤ 4 := by
  sorry

end range_of_a_l1990_199088


namespace inequality_solution_l1990_199060

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^3 - 1) > 0 ↔ x < -3 ∨ (-3 < x ∧ x < 1) ∨ x > 3 :=
sorry

end inequality_solution_l1990_199060


namespace train_speed_l1990_199041

/-- Given a train of length 180 meters that crosses a stationary point in 6 seconds,
    prove that its speed is 30 meters per second. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 6) :
  length / time = 30 := by
  sorry

end train_speed_l1990_199041


namespace water_servings_difference_l1990_199004

/-- Proves the difference in servings for Simeon's water consumption --/
theorem water_servings_difference (total_water : ℕ) (old_serving : ℕ) (new_serving : ℕ)
  (h1 : total_water = 64)
  (h2 : old_serving = 8)
  (h3 : new_serving = 16)
  (h4 : old_serving > 0)
  (h5 : new_serving > 0) :
  (total_water / old_serving) - (total_water / new_serving) = 4 := by
  sorry

end water_servings_difference_l1990_199004


namespace sandra_beignets_l1990_199099

/-- The number of beignets Sandra eats in 16 weeks -/
def total_beignets : ℕ := 336

/-- The number of weeks -/
def num_weeks : ℕ := 16

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of beignets Sandra eats every morning -/
def beignets_per_morning : ℕ := total_beignets / (num_weeks * days_per_week)

theorem sandra_beignets : beignets_per_morning = 3 := by
  sorry

end sandra_beignets_l1990_199099


namespace gcd_10010_15015_l1990_199067

theorem gcd_10010_15015 : Nat.gcd 10010 15015 = 5005 := by
  sorry

end gcd_10010_15015_l1990_199067


namespace complex_equation_real_part_l1990_199024

theorem complex_equation_real_part : 
  ∀ z : ℂ, (1 + Complex.I) * z = Complex.I → Complex.re z = (1 : ℝ) / 2 := by
sorry

end complex_equation_real_part_l1990_199024


namespace largest_solution_bound_l1990_199097

theorem largest_solution_bound (x : ℝ) : 
  3 * (9 * x^2 + 15 * x + 20) = x * (9 * x - 60) →
  x ≤ -0.642 ∧ x > -0.643 :=
by sorry

end largest_solution_bound_l1990_199097


namespace arrange_seven_books_three_identical_l1990_199030

/-- The number of ways to arrange books with some identical copies -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  Nat.factorial total / Nat.factorial identical

/-- Theorem: Arranging 7 books with 3 identical copies yields 840 possibilities -/
theorem arrange_seven_books_three_identical :
  arrange_books 7 3 = 840 := by
  sorry

end arrange_seven_books_three_identical_l1990_199030


namespace rhombus_diagonals_bisect_l1990_199002

-- Define the necessary structures
structure Parallelogram :=
  (diagonals_bisect : Bool)

structure Rhombus :=
  (is_parallelogram : Bool)
  (diagonals_bisect : Bool)

-- State the theorem
theorem rhombus_diagonals_bisect :
  (∀ p : Parallelogram, p.diagonals_bisect = true) →
  (∀ r : Rhombus, r.is_parallelogram = true) →
  (∀ r : Rhombus, r.diagonals_bisect = true) :=
by sorry

end rhombus_diagonals_bisect_l1990_199002


namespace real_part_of_z_is_two_l1990_199053

theorem real_part_of_z_is_two : Complex.re (((Complex.I - 1)^2 + 1) / Complex.I^3) = 2 := by
  sorry

end real_part_of_z_is_two_l1990_199053


namespace simplify_expression_l1990_199001

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) := by
  sorry

end simplify_expression_l1990_199001


namespace inequality_solution_l1990_199056

def solution_set : Set ℝ :=
  {x | x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)}

theorem inequality_solution (x : ℝ) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ x ∈ solution_set :=
sorry

end inequality_solution_l1990_199056


namespace inverse_variation_problem_l1990_199039

/-- Given that x^4 varies inversely with the fourth root of w, 
    prove that when x = 6, w = 1/4096, given that x = 3 when w = 16 -/
theorem inverse_variation_problem (x w : ℝ) (k : ℝ) (h1 : x^4 * w^(1/4) = k) 
  (h2 : 3^4 * 16^(1/4) = k) : 
  x = 6 → w = 1/4096 := by
  sorry

end inverse_variation_problem_l1990_199039


namespace min_route_length_5x5_city_l1990_199064

/-- Represents a square grid city -/
structure City where
  size : ℕ
  streets : ℕ

/-- Calculates the minimum route length for an Eulerian circuit in the city -/
def minRouteLength (c : City) : ℕ :=
  2 * c.streets + 8

theorem min_route_length_5x5_city :
  ∃ (c : City), c.size = 5 ∧ c.streets = 30 ∧ minRouteLength c = 68 :=
by sorry

end min_route_length_5x5_city_l1990_199064


namespace trackball_mice_count_l1990_199040

theorem trackball_mice_count (total : ℕ) (wireless_ratio : ℚ) (optical_ratio : ℚ) :
  total = 80 →
  wireless_ratio = 1/2 →
  optical_ratio = 1/4 →
  (wireless_ratio + optical_ratio + (1 - wireless_ratio - optical_ratio) : ℚ) = 1 →
  ↑total * (1 - wireless_ratio - optical_ratio) = 20 :=
by sorry

end trackball_mice_count_l1990_199040


namespace equal_distribution_probability_l1990_199058

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player
| Dana : Player

/-- The state of the game is represented by the money each player has -/
def GameState := Player → ℕ

/-- The initial state of the game where each player has 1 dollar -/
def initialState : GameState := fun _ => 1

/-- A single turn of the game where a player gives 1 dollar to another randomly chosen player -/
def turn (state : GameState) : GameState := sorry

/-- The probability that after 40 turns, each player has 1 dollar -/
def probabilityEqualDistribution (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating that the probability of equal distribution after 40 turns is 1/9 -/
theorem equal_distribution_probability :
  probabilityEqualDistribution 40 = 1/9 := by
  sorry

end equal_distribution_probability_l1990_199058


namespace jinho_remaining_money_l1990_199048

theorem jinho_remaining_money (initial_amount : ℕ) (eraser_cost pencil_cost : ℕ) 
  (eraser_count pencil_count : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 2500 →
  eraser_cost = 120 →
  pencil_cost = 350 →
  eraser_count = 5 →
  pencil_count = 3 →
  remaining_amount = initial_amount - (eraser_cost * eraser_count + pencil_cost * pencil_count) →
  remaining_amount = 850 := by
sorry

end jinho_remaining_money_l1990_199048


namespace safe_journey_exists_l1990_199000

-- Define the duration of the journey
def road_duration : ℕ := 4
def trail_duration : ℕ := 4

-- Define the eruption patterns
def crater1_cycle : ℕ := 18
def crater2_cycle : ℕ := 10

-- Define the safety condition
def is_safe (t : ℕ) : Prop :=
  (t % crater1_cycle ≠ 0) ∧ 
  ((t % crater2_cycle ≠ 0) → (t < road_duration ∨ t ≥ road_duration + trail_duration))

-- Theorem statement
theorem safe_journey_exists :
  ∃ start : ℕ, 
    (∀ t : ℕ, t ≥ start ∧ t < start + 2 * (road_duration + trail_duration) → is_safe t) :=
sorry

end safe_journey_exists_l1990_199000


namespace least_number_divisible_by_first_five_primes_l1990_199090

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def is_divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  ∀ m ∈ list, n % m = 0

theorem least_number_divisible_by_first_five_primes :
  ∃ n : Nat, n > 0 ∧ is_divisible_by_all n first_five_primes ∧
  ∀ k : Nat, k > 0 ∧ is_divisible_by_all k first_five_primes → n ≤ k :=
by sorry

end least_number_divisible_by_first_five_primes_l1990_199090


namespace total_combinations_l1990_199019

/-- The number of students -/
def num_students : ℕ := 20

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The minimum number of members in each group -/
def min_members_per_group : ℕ := 3

/-- The number of topics -/
def num_topics : ℕ := 5

/-- The number of ways to divide students into groups -/
def group_formations : ℕ := 165

/-- The number of ways to assign topics to groups -/
def topic_assignments : ℕ := 120

theorem total_combinations : 
  (group_formations * topic_assignments = 19800) ∧ 
  (num_students ≥ num_groups * min_members_per_group) ∧
  (num_topics > num_groups) :=
sorry

end total_combinations_l1990_199019


namespace special_parallelogram_side_ratio_l1990_199010

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- Adjacent sides of the parallelogram
  a : ℝ
  b : ℝ
  -- Diagonals of the parallelogram
  d1 : ℝ
  d2 : ℝ
  -- Conditions
  a_pos : 0 < a
  b_pos : 0 < b
  d1_pos : 0 < d1
  d2_pos : 0 < d2
  acute_angle : Real.cos (60 * π / 180) = 1 / 2
  diag_ratio : d1^2 / d2^2 = 1 / 3
  diag1_eq : d1^2 = a^2 + b^2 - a * b
  diag2_eq : d2^2 = a^2 + b^2 + a * b

/-- Theorem: In a special parallelogram, the ratio of adjacent sides is 1:1 -/
theorem special_parallelogram_side_ratio (p : SpecialParallelogram) : p.a = p.b := by
  sorry

end special_parallelogram_side_ratio_l1990_199010


namespace polynomial_division_l1990_199009

-- Define the polynomials
def P (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 8 * x - 1
def D (x : ℝ) : ℝ := x - 3
def Q (x : ℝ) : ℝ := 3 * x^2 + 8
def R : ℝ := 23

-- State the theorem
theorem polynomial_division :
  ∀ x : ℝ, P x = D x * Q x + R := by sorry

end polynomial_division_l1990_199009


namespace geometric_sequence_property_l1990_199083

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property
  (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q)
  (h_neg : a 1 * a 2 < 0) :
  a 1 * a 5 > 0 := by
sorry

end geometric_sequence_property_l1990_199083


namespace three_distinct_roots_l1990_199077

/-- The polynomial Q(x) with parameter p -/
def Q (p : ℝ) (x : ℝ) : ℝ := x^3 + p * x^2 - p * x - 1

/-- Theorem stating the condition for Q(x) to have three distinct real roots -/
theorem three_distinct_roots (p : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    Q p x = 0 ∧ Q p y = 0 ∧ Q p z = 0) ↔ 
  (p > 1 ∨ p < -3) :=
sorry

end three_distinct_roots_l1990_199077


namespace quadratic_roots_subset_l1990_199013

/-- Set A is defined as the solution set of x^2 + ax + b = 0 -/
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

/-- Set B is defined as {1, 2} -/
def B : Set ℝ := {1, 2}

/-- The theorem states that given the conditions, (a, b) must be one of the three specified pairs -/
theorem quadratic_roots_subset (a b : ℝ) : 
  A a b ⊆ B ∧ A a b ≠ ∅ → 
  ((a = -2 ∧ b = 1) ∨ (a = -4 ∧ b = 4) ∨ (a = -3 ∧ b = 2)) :=
by sorry

end quadratic_roots_subset_l1990_199013


namespace same_speed_is_two_l1990_199061

-- Define Jack's speed function
def jack_speed (x : ℝ) : ℝ := x^2 - 7*x - 18

-- Define Jill's distance function
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72

-- Define Jill's time function
def jill_time (x : ℝ) : ℝ := x + 8

-- Theorem statement
theorem same_speed_is_two :
  ∀ x : ℝ, 
  x ≠ -8 →  -- Ensure division by zero is avoided
  (jill_distance x) / (jill_time x) = jack_speed x →
  jack_speed x = 2 :=
by sorry

end same_speed_is_two_l1990_199061


namespace missing_number_proof_l1990_199011

theorem missing_number_proof : 
  ∃ x : ℝ, 248 + x - Real.sqrt (- Real.sqrt 0) = 16 ∧ x = -232 := by sorry

end missing_number_proof_l1990_199011


namespace problem_solution_l1990_199016

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else |Real.sin x|

theorem problem_solution (a : ℝ) :
  f a = (1/2) → (a = (1/4) ∨ a = -π/6) := by
  sorry

end problem_solution_l1990_199016


namespace percentage_of_b_grades_l1990_199031

def scores : List ℕ := [91, 82, 68, 99, 79, 86, 88, 76, 71, 58, 80, 89, 65, 85, 93]

def is_b_grade (score : ℕ) : Bool :=
  87 ≤ score && score ≤ 94

def count_b_grades (scores : List ℕ) : ℕ :=
  scores.filter is_b_grade |>.length

theorem percentage_of_b_grades :
  (count_b_grades scores : ℚ) / (scores.length : ℚ) * 100 = 80 / 3 := by
  sorry

end percentage_of_b_grades_l1990_199031


namespace jessica_seashells_l1990_199082

/-- Given that Joan found 6 seashells and the total number of seashells found by Joan and Jessica is 14, prove that Jessica found 8 seashells. -/
theorem jessica_seashells (joan_seashells : ℕ) (total_seashells : ℕ) (h1 : joan_seashells = 6) (h2 : total_seashells = 14) :
  total_seashells - joan_seashells = 8 := by
  sorry

end jessica_seashells_l1990_199082


namespace b_savings_l1990_199076

/-- Given two people a and b with monthly incomes and expenditures, calculate b's savings -/
theorem b_savings (income_a income_b expenditure_a expenditure_b savings_a : ℕ) 
  (h1 : income_a * 6 = income_b * 5)  -- income ratio 5:6
  (h2 : expenditure_a * 4 = expenditure_b * 3)  -- expenditure ratio 3:4
  (h3 : income_a - expenditure_a = savings_a)
  (h4 : savings_a = 1800)
  (h5 : income_b = 7200) :
  income_b - expenditure_b = 1600 := by sorry

end b_savings_l1990_199076


namespace supermarket_eggs_l1990_199006

/-- Represents the number of egg cartons in the supermarket -/
def num_cartons : ℕ := 28

/-- Represents the length of the egg array in each carton -/
def carton_length : ℕ := 33

/-- Represents the width of the egg array in each carton -/
def carton_width : ℕ := 4

/-- Calculates the total number of eggs in the supermarket -/
def total_eggs : ℕ := num_cartons * carton_length * carton_width

/-- Theorem stating that the total number of eggs in the supermarket is 3696 -/
theorem supermarket_eggs : total_eggs = 3696 := by
  sorry

end supermarket_eggs_l1990_199006


namespace largest_c_value_l1990_199074

theorem largest_c_value (c : ℝ) : 
  (∀ x : ℝ, -2*x^2 + 8*x - 6 ≥ 0 → x ≤ c) ↔ c = 3 := by sorry

end largest_c_value_l1990_199074


namespace concentric_circles_theorem_l1990_199043

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) where
  radius_larger : R > r

/-- Point on a circle -/
structure PointOnCircle (center : ℝ × ℝ) (radius : ℝ) where
  point : ℝ × ℝ
  on_circle : (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

/-- Theorem about the sum of squared distances and the locus of midpoint -/
theorem concentric_circles_theorem
  (R r : ℝ) (h : ConcentricCircles R r)
  (O : ℝ × ℝ) -- Center of the circles
  (P : PointOnCircle O r) -- Fixed point on smaller circle
  (B : PointOnCircle O R) -- Moving point on larger circle
  (A : PointOnCircle O r) -- Point on smaller circle determined by perpendicular line from P to BP
  (C : PointOnCircle O R) -- Intersection of BP with larger circle
  : 
  -- Part 1: Sum of squared distances
  (B.point.1 - C.point.1)^2 + (B.point.2 - C.point.2)^2 +
  (C.point.1 - A.point.1)^2 + (C.point.2 - A.point.2)^2 +
  (A.point.1 - B.point.1)^2 + (A.point.2 - B.point.2)^2 = 6 * R^2 + 2 * r^2
  ∧
  -- Part 2: Locus of midpoint of AB
  ∃ (Q : ℝ × ℝ),
    Q = ((A.point.1 + B.point.1) / 2, (A.point.2 + B.point.2) / 2) ∧
    (Q.1 - (O.1 + P.point.1) / 2)^2 + (Q.2 - (O.2 + P.point.2) / 2)^2 = (R / 2)^2 :=
by sorry

end concentric_circles_theorem_l1990_199043


namespace equation_solution_l1990_199042

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ↔ x = -2/3 := by
sorry

end equation_solution_l1990_199042


namespace kendys_initial_balance_l1990_199079

/-- Proves that Kendy's initial account balance was $190 given the conditions of her transfers --/
theorem kendys_initial_balance :
  let mom_transfer : ℕ := 60
  let sister_transfer : ℕ := mom_transfer / 2
  let remaining_balance : ℕ := 100
  let initial_balance : ℕ := remaining_balance + mom_transfer + sister_transfer
  initial_balance = 190 := by
  sorry

end kendys_initial_balance_l1990_199079


namespace quadratic_minimum_l1990_199093

theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := fun x => x^2 - 12*x + 35
  ∃ (min_x : ℝ), ∀ y, f y ≥ f min_x ∧ min_x = 6 :=
by sorry

end quadratic_minimum_l1990_199093


namespace tournament_matches_l1990_199044

def matches_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tournament_matches : 
  let group_a_players : ℕ := 6
  let group_b_players : ℕ := 5
  matches_in_group group_a_players + matches_in_group group_b_players = 25 := by
  sorry

end tournament_matches_l1990_199044


namespace unique_solution_for_m_squared_minus_eight_equals_three_to_n_l1990_199034

theorem unique_solution_for_m_squared_minus_eight_equals_three_to_n :
  ∀ m n : ℕ, m^2 - 8 = 3^n ↔ m = 3 ∧ n = 0 := by
sorry

end unique_solution_for_m_squared_minus_eight_equals_three_to_n_l1990_199034


namespace abs_inequality_solution_set_l1990_199089

theorem abs_inequality_solution_set (x : ℝ) : 
  |3*x + 1| > 2 ↔ x > 1/3 ∨ x < -1 := by sorry

end abs_inequality_solution_set_l1990_199089


namespace parabola_intersection_l1990_199054

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ := {x | 2*x^2 + 3*x - 4 = x^2 + 2*x + 1}

/-- The y-coordinate of the intersection points -/
def intersection_y : ℝ := 4.5

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ := 2*x^2 + 3*x - 4

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem parabola_intersection :
  intersection_x = {(-1 + Real.sqrt 21) / 2, (-1 - Real.sqrt 21) / 2} ∧
  ∀ x ∈ intersection_x, parabola1 x = intersection_y ∧ parabola2 x = intersection_y :=
sorry

end parabola_intersection_l1990_199054


namespace initial_men_count_l1990_199007

/-- The number of men initially colouring the cloth -/
def M : ℕ := sorry

/-- The length of cloth coloured by M men in 2 days -/
def initial_cloth_length : ℝ := 48

/-- The time taken by M men to colour the initial cloth length -/
def initial_time : ℝ := 2

/-- The length of cloth coloured by 8 men in 0.75 days -/
def new_cloth_length : ℝ := 36

/-- The time taken by 8 men to colour the new cloth length -/
def new_time : ℝ := 0.75

/-- The number of men in the new scenario -/
def new_men : ℕ := 8

theorem initial_men_count : M = 4 := by sorry

end initial_men_count_l1990_199007


namespace password_selection_rule_probability_of_A_in_seventh_week_l1990_199008

/-- Represents the probability of password A being used in week k -/
def P (k : ℕ) : ℚ :=
  if k = 1 then 1
  else (3/4) * (-1/3)^(k-2) + 1/4

/-- The condition that the password for each week is chosen randomly from
    the three not used in the previous week -/
theorem password_selection_rule (k : ℕ) :
  k > 1 → P k = (1/3) * (1 - P (k-1)) :=
sorry

theorem probability_of_A_in_seventh_week :
  P 7 = 61/243 :=
sorry

end password_selection_rule_probability_of_A_in_seventh_week_l1990_199008


namespace fruit_display_total_l1990_199094

/-- Represents the number of fruits on a display -/
structure FruitDisplay where
  bananas : ℕ
  oranges : ℕ
  apples : ℕ
  lemons : ℕ

/-- Calculates the total number of fruits on the display -/
def totalFruits (d : FruitDisplay) : ℕ :=
  d.bananas + d.oranges + d.apples + d.lemons

/-- Theorem stating the total number of fruits on the display -/
theorem fruit_display_total (d : FruitDisplay) 
  (h1 : d.bananas = 5)
  (h2 : d.oranges = 2 * d.bananas)
  (h3 : d.apples = 2 * d.oranges)
  (h4 : d.lemons = (d.apples + d.bananas) / 2) :
  totalFruits d = 47 := by
  sorry

#eval totalFruits { bananas := 5, oranges := 10, apples := 20, lemons := 12 }

end fruit_display_total_l1990_199094


namespace merchant_max_profit_optimal_selling_price_l1990_199049

/-- Represents the merchant's profit function -/
def profit (x : ℝ) : ℝ := -10 * x^2 + 80 * x + 200

/-- The optimal price increase that maximizes profit -/
def optimal_increase : ℝ := 4

/-- The maximum achievable profit -/
def max_profit : ℝ := 360

theorem merchant_max_profit :
  (∀ x, 0 ≤ x → x < 10 → profit x ≤ max_profit) ∧
  profit optimal_increase = max_profit :=
sorry

theorem optimal_selling_price :
  optimal_increase + 10 = 14 :=
sorry

end merchant_max_profit_optimal_selling_price_l1990_199049


namespace interest_rate_is_10_percent_l1990_199005

/-- Calculates the simple interest rate given principal, amount, and time. -/
def calculate_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: Given the conditions, the interest rate is 10%. -/
theorem interest_rate_is_10_percent :
  let principal : ℚ := 750
  let amount : ℚ := 900
  let time : ℕ := 2
  calculate_interest_rate principal amount time = 10 := by
  sorry

end interest_rate_is_10_percent_l1990_199005


namespace symmetric_points_product_l1990_199081

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The problem statement -/
theorem symmetric_points_product (a b : ℝ) 
    (h : symmetric_wrt_origin (a + 2) 2 4 (-b)) : a * b = -12 := by
  sorry

end symmetric_points_product_l1990_199081


namespace evas_shoes_l1990_199086

def total_laces : ℕ := 52
def laces_per_pair : ℕ := 2

theorem evas_shoes : 
  total_laces / laces_per_pair = 26 := by sorry

end evas_shoes_l1990_199086


namespace school_population_l1990_199014

theorem school_population (girls : ℕ) (boys : ℕ) (teachers : ℕ) (staff : ℕ)
  (h1 : girls = 542)
  (h2 : boys = 387)
  (h3 : teachers = 45)
  (h4 : staff = 27) :
  girls + boys + teachers + staff = 1001 := by
sorry

end school_population_l1990_199014


namespace area_difference_square_rectangle_l1990_199073

theorem area_difference_square_rectangle :
  ∀ (square_side : ℝ) (rect_length rect_width : ℝ),
  square_side * 4 = 52 →
  rect_length = 15 →
  rect_length * 2 + rect_width * 2 = 52 →
  square_side * square_side - rect_length * rect_width = 4 := by
sorry

end area_difference_square_rectangle_l1990_199073


namespace machine_x_production_rate_l1990_199065

/-- Production rates and times for two machines -/
structure MachineProduction where
  x_rate : ℝ  -- Production rate of Machine X (widgets per hour)
  y_rate : ℝ  -- Production rate of Machine Y (widgets per hour)
  x_time : ℝ  -- Time taken by Machine X to produce 1080 widgets
  y_time : ℝ  -- Time taken by Machine Y to produce 1080 widgets

/-- Theorem stating the production rate of Machine X -/
theorem machine_x_production_rate (m : MachineProduction) :
  m.x_rate = 18 :=
by
  have h1 : m.x_time = m.y_time + 10 := by sorry
  have h2 : m.y_rate = 1.2 * m.x_rate := by sorry
  have h3 : m.x_rate * m.x_time = 1080 := by sorry
  have h4 : m.y_rate * m.y_time = 1080 := by sorry
  sorry

end machine_x_production_rate_l1990_199065


namespace boat_speed_in_still_water_l1990_199023

/-- Proves that the speed of a boat in still water is 57 kmph given the conditions -/
theorem boat_speed_in_still_water : 
  ∀ (t : ℝ) (Vb : ℝ),
    t > 0 →  -- time taken to row downstream is positive
    Vb > 19 →  -- boat speed in still water is greater than stream speed
    (Vb - 19) * (2 * t) = (Vb + 19) * t →  -- equation based on distance = speed * time
    Vb = 57 := by
  sorry

end boat_speed_in_still_water_l1990_199023


namespace brandons_cash_sales_l1990_199035

theorem brandons_cash_sales (total_sales : ℝ) (credit_sales_fraction : ℝ) (cash_sales : ℝ) : 
  total_sales = 80 →
  credit_sales_fraction = 2/5 →
  cash_sales = total_sales * (1 - credit_sales_fraction) →
  cash_sales = 48 := by
sorry

end brandons_cash_sales_l1990_199035


namespace combined_salaries_of_four_l1990_199092

/-- Given 5 individuals with an average monthly salary and one known salary, 
    prove the sum of the other four salaries. -/
theorem combined_salaries_of_four (average_salary : ℕ) (known_salary : ℕ) 
  (h1 : average_salary = 9000)
  (h2 : known_salary = 5000) :
  4 * average_salary - known_salary = 40000 := by
  sorry

end combined_salaries_of_four_l1990_199092


namespace reads_two_days_per_week_l1990_199091

/-- A person's reading habits over a period of weeks -/
structure ReadingHabits where
  booksPerDay : ℕ
  totalBooks : ℕ
  totalWeeks : ℕ

/-- Calculate the number of days per week a person reads based on their reading habits -/
def daysPerWeek (habits : ReadingHabits) : ℚ :=
  (habits.totalBooks / habits.booksPerDay : ℚ) / habits.totalWeeks

/-- Theorem: Given the specific reading habits, prove that the person reads 2 days per week -/
theorem reads_two_days_per_week (habits : ReadingHabits)
  (h1 : habits.booksPerDay = 4)
  (h2 : habits.totalBooks = 48)
  (h3 : habits.totalWeeks = 6) :
  daysPerWeek habits = 2 := by
  sorry

end reads_two_days_per_week_l1990_199091


namespace equal_roots_cubic_l1990_199045

theorem equal_roots_cubic (m n : ℝ) (h : n ≠ 0) :
  ∃ (x : ℝ), (x^3 + m*x - n = 0 ∧ n*x^3 - 2*m^2*x^2 - 5*m*n*x - 2*m^3 - n^2 = 0) →
  ∃ (a : ℝ), a = (n/2)^(1/3) ∧ 
  (∀ y : ℝ, y^3 + m*y - n = 0 ↔ y = a ∨ y = a ∨ y = -2*a) :=
by sorry

end equal_roots_cubic_l1990_199045


namespace trapezoid_lower_side_length_l1990_199028

/-- Proves that the length of the lower side of a trapezoid is 17.65 cm given specific conditions -/
theorem trapezoid_lower_side_length 
  (height : ℝ) 
  (area : ℝ) 
  (side_difference : ℝ) 
  (h1 : height = 5.2)
  (h2 : area = 100.62)
  (h3 : side_difference = 3.4) : 
  ∃ (lower_side : ℝ), lower_side = 17.65 ∧ 
  area = (1/2) * (lower_side + (lower_side + side_difference)) * height :=
sorry

end trapezoid_lower_side_length_l1990_199028


namespace parabola_point_order_l1990_199075

/-- Theorem: For a parabola y = ax² - 2ax + 3 with a > 0, and points A(-1, y₁), B(2, y₂), C(4, y₃) on the parabola, prove that y₂ < y₁ < y₃ -/
theorem parabola_point_order (a : ℝ) (y₁ y₂ y₃ : ℝ) 
  (ha : a > 0)
  (hy₁ : y₁ = a * (-1)^2 - 2 * a * (-1) + 3)
  (hy₂ : y₂ = a * 2^2 - 2 * a * 2 + 3)
  (hy₃ : y₃ = a * 4^2 - 2 * a * 4 + 3) :
  y₂ < y₁ ∧ y₁ < y₃ :=
sorry

end parabola_point_order_l1990_199075


namespace tournament_probability_l1990_199051

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams.choose 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1/2

/-- The probability that no two teams win the same number of games -/
noncomputable def unique_wins_probability : ℚ := (num_teams.factorial : ℚ) / 2^total_games

theorem tournament_probability :
  ∃ (m : ℕ), m % 2 = 1 ∧ unique_wins_probability = (m : ℚ) / 2^409 :=
sorry

end tournament_probability_l1990_199051


namespace max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l1990_199046

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probability_divisible_by_10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

theorem max_probability_divisible_by_10 :
  ∃ s : Segment, probability_divisible_by_10 s = 1 ∧
  ∀ t : Segment, probability_divisible_by_10 t ≤ 1 :=
sorry

theorem min_nonzero_probability_divisible_by_10 :
  ∃ s : Segment, probability_divisible_by_10 s = 1/19 ∧
  ∀ t : Segment, probability_divisible_by_10 t = 0 ∨ probability_divisible_by_10 t ≥ 1/19 :=
sorry

end max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l1990_199046


namespace problem_solution_l1990_199032

theorem problem_solution : ∃ n : ℕ, 2^13 - 2^11 = 3 * n ∧ n = 2048 := by
  sorry

end problem_solution_l1990_199032


namespace cosine_angle_OAB_l1990_199098

/-- Given points A and B in a 2D Cartesian coordinate system with O as the origin,
    prove that the cosine of angle OAB is equal to -√2/10. -/
theorem cosine_angle_OAB (A B : ℝ × ℝ) (h_A : A = (-3, -4)) (h_B : B = (5, -12)) :
  let O : ℝ × ℝ := (0, 0)
  let AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let dot_product := AO.1 * AB.1 + AO.2 * AB.2
  let magnitude_AO := Real.sqrt (AO.1^2 + AO.2^2)
  let magnitude_AB := Real.sqrt (AB.1^2 + AB.2^2)
  dot_product / (magnitude_AO * magnitude_AB) = -Real.sqrt 2 / 10 := by
sorry

end cosine_angle_OAB_l1990_199098


namespace genesis_work_hours_l1990_199059

/-- The number of hours Genesis worked per day on the new project -/
def hoursPerDayNewProject : ℕ := 6

/-- The number of weeks Genesis worked on the new project -/
def weeksNewProject : ℕ := 3

/-- The number of hours Genesis worked per day on the additional task -/
def hoursPerDayAdditionalTask : ℕ := 3

/-- The number of weeks Genesis worked on the additional task -/
def weeksAdditionalTask : ℕ := 2

/-- The number of days in a week -/
def daysPerWeek : ℕ := 7

/-- The total number of hours Genesis worked during the entire period -/
def totalHoursWorked : ℕ :=
  hoursPerDayNewProject * weeksNewProject * daysPerWeek +
  hoursPerDayAdditionalTask * weeksAdditionalTask * daysPerWeek

theorem genesis_work_hours : totalHoursWorked = 168 := by
  sorry

end genesis_work_hours_l1990_199059


namespace zoo_visitors_saturday_l1990_199022

theorem zoo_visitors_saturday (friday_visitors : ℕ) (saturday_multiplier : ℕ) : 
  friday_visitors = 3575 →
  saturday_multiplier = 5 →
  friday_visitors * saturday_multiplier = 17875 :=
by
  sorry

end zoo_visitors_saturday_l1990_199022


namespace seokgi_candies_l1990_199066

theorem seokgi_candies :
  ∀ (original : ℕ),
  (original : ℚ) * (1/2 : ℚ) * (2/3 : ℚ) = 12 →
  original = 36 := by
  sorry

end seokgi_candies_l1990_199066


namespace l_shaped_area_l1990_199068

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (total_side : ℝ) (small_side1 small_side2 large_side : ℝ) :
  total_side = 7 ∧ 
  small_side1 = 2 ∧ 
  small_side2 = 2 ∧ 
  large_side = 5 →
  total_side^2 - (small_side1^2 + small_side2^2 + large_side^2) = 16 := by
  sorry

end l_shaped_area_l1990_199068


namespace initial_population_proof_l1990_199003

/-- Proves that the initial population is 10000 given the conditions --/
theorem initial_population_proof (P : ℝ) : 
  (P * (1 + 0.2)^2 = 14400) → P = 10000 := by
  sorry

end initial_population_proof_l1990_199003


namespace decimal_equivalent_of_one_tenth_squared_l1990_199095

theorem decimal_equivalent_of_one_tenth_squared : (1 / 10 : ℚ) ^ 2 = 0.01 := by
  sorry

end decimal_equivalent_of_one_tenth_squared_l1990_199095


namespace empty_seats_in_theater_l1990_199057

theorem empty_seats_in_theater (total_seats people_watching : ℕ) 
  (h1 : total_seats = 750)
  (h2 : people_watching = 532) :
  total_seats - people_watching = 218 := by
  sorry

end empty_seats_in_theater_l1990_199057


namespace janes_quiz_mean_l1990_199021

theorem janes_quiz_mean : 
  let scores : List ℝ := [86, 91, 89, 95, 88, 94]
  (scores.sum / scores.length : ℝ) = 90.5 := by sorry

end janes_quiz_mean_l1990_199021


namespace arctan_sum_three_four_l1990_199047

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end arctan_sum_three_four_l1990_199047


namespace circle_sum_inequality_l1990_199033

theorem circle_sum_inequality (a : Fin 100 → ℝ) (h : Function.Injective a) :
  ∃ i : Fin 100, a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100) := by
  sorry

end circle_sum_inequality_l1990_199033


namespace middle_of_three_consecutive_sum_30_l1990_199012

/-- Given three consecutive natural numbers whose sum is 30, the middle number is 10. -/
theorem middle_of_three_consecutive_sum_30 :
  ∀ n : ℕ, n + (n + 1) + (n + 2) = 30 → n + 1 = 10 :=
by
  sorry

end middle_of_three_consecutive_sum_30_l1990_199012


namespace train_arrival_time_l1990_199036

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem train_arrival_time 
  (departure : Time)
  (journey_duration : Nat)
  (h1 : departure = { hours := 9, minutes := 45 })
  (h2 : journey_duration = 15) :
  addMinutes departure journey_duration = { hours := 10, minutes := 0 } :=
sorry

end train_arrival_time_l1990_199036


namespace cos_alpha_value_l1990_199050

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.cos α = 1 / 3 := by
  sorry

end cos_alpha_value_l1990_199050


namespace number_of_students_in_section_B_l1990_199020

/-- Given a class with two sections A and B, prove the number of students in section B -/
theorem number_of_students_in_section_B 
  (students_A : ℕ) 
  (avg_weight_A : ℚ) 
  (avg_weight_B : ℚ) 
  (avg_weight_total : ℚ) 
  (h1 : students_A = 50)
  (h2 : avg_weight_A = 50)
  (h3 : avg_weight_B = 70)
  (h4 : avg_weight_total = 61.67) :
  ∃ (students_B : ℕ), students_B = 70 := by
sorry

end number_of_students_in_section_B_l1990_199020


namespace smallest_solution_abs_equation_l1990_199070

theorem smallest_solution_abs_equation :
  let f : ℝ → ℝ := λ x => x * |x| - (2 * x^2 + 3 * x + 1)
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y ∧ x = (3 + Real.sqrt 13) / 2 :=
by sorry

end smallest_solution_abs_equation_l1990_199070


namespace intersecting_sphere_yz_radius_l1990_199062

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the intersection circle with xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the intersection circle with xy-plane -/
  xy_radius : ℝ
  /-- Center of the intersection circle with yz-plane -/
  yz_center : ℝ × ℝ × ℝ
  /-- Radius of the intersection circle with yz-plane -/
  yz_radius : ℝ

/-- The theorem stating the radius of the yz-plane intersection -/
theorem intersecting_sphere_yz_radius (sphere : IntersectingSphere) 
  (h1 : sphere.xy_center = (3, 5, 0))
  (h2 : sphere.xy_radius = 2)
  (h3 : sphere.yz_center = (0, 5, -8)) :
  sphere.yz_radius = Real.sqrt 59 := by
  sorry

end intersecting_sphere_yz_radius_l1990_199062
