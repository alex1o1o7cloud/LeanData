import Mathlib

namespace NUMINAMATH_CALUDE_n_pointed_star_degree_sum_l3317_331778

/-- An n-pointed star formed from a convex polygon -/
structure NPointedStar where
  n : ℕ
  h_n_ge_7 : n ≥ 7

/-- The degree sum of interior angles of an n-pointed star -/
def degree_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The degree sum of interior angles of an n-pointed star is 180(n-2) -/
theorem n_pointed_star_degree_sum (star : NPointedStar) :
  degree_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_n_pointed_star_degree_sum_l3317_331778


namespace NUMINAMATH_CALUDE_roses_planted_l3317_331747

theorem roses_planted (day1 day2 day3 : ℕ) : 
  day2 = day1 + 20 →
  day3 = 2 * day1 →
  day1 + day2 + day3 = 220 →
  day1 = 50 := by
sorry

end NUMINAMATH_CALUDE_roses_planted_l3317_331747


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3317_331787

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1, 3}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3317_331787


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l3317_331775

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beads : ℝ
  rings : ℝ
  silver_coins : ℝ
  gold_coins : ℝ

/-- Theorem stating the percentage of gold coins in the urn --/
theorem gold_coins_percentage (u : UrnComposition) 
  (h1 : u.beads = 0.3)
  (h2 : u.rings = 0.1)
  (h3 : u.silver_coins + u.gold_coins = 0.6)
  (h4 : u.silver_coins = 0.35 * (u.silver_coins + u.gold_coins)) :
  u.gold_coins = 0.39 := by
  sorry


end NUMINAMATH_CALUDE_gold_coins_percentage_l3317_331775


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3317_331750

theorem system_of_equations_solution (x y z : ℤ) 
  (eq1 : x + y + z = 600)
  (eq2 : x - y = 200)
  (eq3 : x + z = 500) :
  x = 300 ∧ y = 100 ∧ z = 200 :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3317_331750


namespace NUMINAMATH_CALUDE_one_sofa_in_room_l3317_331737

/-- Represents the number of sofas in the room -/
def num_sofas : ℕ := 1

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 40

/-- Represents the number of legs on a sofa -/
def legs_per_sofa : ℕ := 4

/-- Represents the number of legs from furniture other than sofas -/
def other_furniture_legs : ℕ := 
  4 * 4 +  -- 4 tables with 4 legs each
  2 * 4 +  -- 2 chairs with 4 legs each
  3 * 3 +  -- 3 tables with 3 legs each
  1 * 1 +  -- 1 table with 1 leg
  1 * 2    -- 1 rocking chair with 2 legs

/-- Theorem stating that there is exactly one sofa in the room -/
theorem one_sofa_in_room : 
  num_sofas * legs_per_sofa + other_furniture_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_one_sofa_in_room_l3317_331737


namespace NUMINAMATH_CALUDE_reflection_creates_symmetry_l3317_331702

/-- Represents a letter in the word --/
inductive Letter
| G | E | O | M | T | R | I | Ya

/-- Represents a position in 2D space --/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a word as a list of letters with their positions --/
def Word := List (Letter × Position)

/-- The original word "ГЕОМЕТРИя" --/
def original_word : Word := sorry

/-- Reflects a position across a vertical axis --/
def reflect_vertical (p : Position) (axis : ℝ) : Position :=
  ⟨2 * axis - p.x, p.y⟩

/-- Reflects a word across a vertical axis --/
def reflect_word_vertical (w : Word) (axis : ℝ) : Word :=
  w.map (fun (l, p) => (l, reflect_vertical p axis))

/-- Checks if a word is symmetrical across a vertical axis --/
def is_symmetrical_vertical (w : Word) (axis : ℝ) : Prop :=
  w = reflect_word_vertical w axis

/-- Theorem: Reflecting the word "ГЕОМЕТРИя" across a vertical axis results in a symmetrical figure --/
theorem reflection_creates_symmetry (axis : ℝ) :
  is_symmetrical_vertical (reflect_word_vertical original_word axis) axis := by
  sorry

end NUMINAMATH_CALUDE_reflection_creates_symmetry_l3317_331702


namespace NUMINAMATH_CALUDE_max_value_of_f_l3317_331745

theorem max_value_of_f (x : ℝ) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ (y : ℝ), min (3 - x^2) (2*x) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3317_331745


namespace NUMINAMATH_CALUDE_max_k_minus_m_is_neg_sqrt_two_l3317_331711

/-- A point on a parabola with complementary lines intersecting the parabola -/
structure ParabolaPoint where
  m : ℝ
  k : ℝ
  h1 : m > 0  -- First quadrant condition
  h2 : k = 1 / (-2 * m)  -- Derived from the problem

/-- The maximum value of k - m for a point on the parabola -/
def max_k_minus_m (p : ParabolaPoint) : ℝ := p.k - p.m

/-- Theorem: The maximum value of k - m is -√2 -/
theorem max_k_minus_m_is_neg_sqrt_two :
  ∃ (p : ParabolaPoint), ∀ (q : ParabolaPoint), max_k_minus_m p ≥ max_k_minus_m q ∧ 
  max_k_minus_m p = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_k_minus_m_is_neg_sqrt_two_l3317_331711


namespace NUMINAMATH_CALUDE_job_completion_time_l3317_331732

/-- Represents the time (in hours) it takes for a single machine to complete the job -/
def single_machine_time : ℝ := 216

/-- Represents the number of machines of each type used -/
def machines_per_type : ℕ := 9

/-- Represents the time (in hours) it takes for all machines working together to complete the job -/
def total_job_time : ℝ := 12

theorem job_completion_time :
  (((1 / single_machine_time) * machines_per_type + 
    (1 / single_machine_time) * machines_per_type) * total_job_time = 1) →
  single_machine_time = 216 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3317_331732


namespace NUMINAMATH_CALUDE_fourth_power_mod_five_l3317_331757

theorem fourth_power_mod_five (a : ℤ) : (a^4) % 5 = 0 ∨ (a^4) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_mod_five_l3317_331757


namespace NUMINAMATH_CALUDE_cookies_eaten_difference_l3317_331712

theorem cookies_eaten_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) :
  initial_sweet = 8 →
  initial_salty = 6 →
  eaten_sweet = 20 →
  eaten_salty = 34 →
  eaten_salty - eaten_sweet = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_difference_l3317_331712


namespace NUMINAMATH_CALUDE_ellipse_existence_in_acute_triangle_l3317_331798

/-- Represents an acute triangle -/
structure AcuteTriangle where
  -- Add necessary fields for an acute triangle
  is_acute : Bool

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Orthocenter of a triangle -/
def orthocenter (t : AcuteTriangle) : Point :=
  sorry

/-- Circumcenter of a triangle -/
def circumcenter (t : AcuteTriangle) : Point :=
  sorry

/-- Theorem: For any acute triangle, there exists an ellipse with one focus
    at the orthocenter and the other at the circumcenter of the triangle -/
theorem ellipse_existence_in_acute_triangle (t : AcuteTriangle) :
  ∃ e : Ellipse, e.focus1 = orthocenter t ∧ e.focus2 = circumcenter t :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_existence_in_acute_triangle_l3317_331798


namespace NUMINAMATH_CALUDE_meaningful_expression_l3317_331734

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (a + 3) / Real.sqrt (a - 3)) ↔ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3317_331734


namespace NUMINAMATH_CALUDE_universal_set_equality_l3317_331718

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 7}

-- Define set A
def A : Finset Nat := {1, 3, 5, 7}

-- Define set B
def B : Finset Nat := {3, 5}

-- Theorem statement
theorem universal_set_equality : U = A ∪ (U \ B) := by
  sorry

end NUMINAMATH_CALUDE_universal_set_equality_l3317_331718


namespace NUMINAMATH_CALUDE_nth_equation_proof_l3317_331772

theorem nth_equation_proof (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

#check nth_equation_proof

end NUMINAMATH_CALUDE_nth_equation_proof_l3317_331772


namespace NUMINAMATH_CALUDE_queenie_work_days_l3317_331779

/-- Calculates the number of days worked given the daily rate, overtime rate, overtime hours, and total payment -/
def days_worked (daily_rate : ℕ) (overtime_rate : ℕ) (overtime_hours : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment / (daily_rate + overtime_rate * overtime_hours))

/-- Proves that given the specified conditions, the number of days worked is 4 -/
theorem queenie_work_days : 
  let daily_rate : ℕ := 150
  let overtime_rate : ℕ := 5
  let overtime_hours : ℕ := 4
  let total_payment : ℕ := 770
  days_worked daily_rate overtime_rate overtime_hours total_payment = 4 := by
sorry

#eval days_worked 150 5 4 770

end NUMINAMATH_CALUDE_queenie_work_days_l3317_331779


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3317_331785

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 8 * x + 3
  ∃ x₁ x₂ : ℝ, x₁ = 2 + (Real.sqrt 10) / 2 ∧
              x₂ = 2 - (Real.sqrt 10) / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_solution_l3317_331785


namespace NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l3317_331726

theorem interior_angle_sum_regular_polygon (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 45 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = 1080 :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l3317_331726


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l3317_331784

theorem zoo_animal_ratio :
  ∀ (birds non_birds : ℕ),
    birds = 450 →
    birds = non_birds + 360 →
    (birds : ℚ) / non_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l3317_331784


namespace NUMINAMATH_CALUDE_second_fish_length_is_02_l3317_331740

/-- The length of the first fish in feet -/
def first_fish_length : ℝ := 0.3

/-- The difference in length between the first and second fish in feet -/
def length_difference : ℝ := 0.1

/-- The length of the second fish in feet -/
def second_fish_length : ℝ := first_fish_length - length_difference

/-- Theorem stating that the second fish is 0.2 foot long -/
theorem second_fish_length_is_02 : second_fish_length = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_second_fish_length_is_02_l3317_331740


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l3317_331746

theorem similar_triangles_perimeter (p_small p_large : ℝ) : 
  p_small > 0 → 
  p_large > 0 → 
  p_small / p_large = 2 / 3 → 
  p_small + p_large = 20 → 
  p_small = 8 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l3317_331746


namespace NUMINAMATH_CALUDE_max_apartments_l3317_331791

/-- Represents an apartment building with specific properties. -/
structure ApartmentBuilding where
  entrances : Nat
  floors : Nat
  apartments_per_floor : Nat
  two_digit_apartments_in_entrance : Nat

/-- The conditions of the apartment building as described in the problem. -/
def building_conditions (b : ApartmentBuilding) : Prop :=
  b.apartments_per_floor = 4 ∧
  b.two_digit_apartments_in_entrance = 10 * b.entrances ∧
  b.two_digit_apartments_in_entrance ≤ 90

/-- The total number of apartments in the building. -/
def total_apartments (b : ApartmentBuilding) : Nat :=
  b.entrances * b.floors * b.apartments_per_floor

/-- Theorem stating the maximum number of apartments in the building. -/
theorem max_apartments (b : ApartmentBuilding) (h : building_conditions b) :
  total_apartments b ≤ 936 := by
  sorry

#check max_apartments

end NUMINAMATH_CALUDE_max_apartments_l3317_331791


namespace NUMINAMATH_CALUDE_genetic_material_distribution_l3317_331793

/-- Represents a diploid organism -/
structure DiploidOrganism :=
  (chromosomes : ℕ)
  (is_diploid : chromosomes % 2 = 0)

/-- Represents genetic material in the cytoplasm -/
structure GeneticMaterial :=
  (amount : ℝ)

/-- Represents a cell of a diploid organism -/
structure Cell :=
  (organism : DiploidOrganism)
  (cytoplasm : GeneticMaterial)

/-- Represents the distribution of genetic material during cell division -/
def genetic_distribution (parent : Cell) (daughter1 daughter2 : Cell) : Prop :=
  (daughter1.cytoplasm.amount + daughter2.cytoplasm.amount = parent.cytoplasm.amount) ∧
  (daughter1.cytoplasm.amount ≠ daughter2.cytoplasm.amount)

/-- Theorem stating that genetic material in the cytoplasm is distributed randomly and unequally during cell division -/
theorem genetic_material_distribution 
  (parent : Cell) 
  (daughter1 daughter2 : Cell) :
  genetic_distribution parent daughter1 daughter2 :=
sorry

end NUMINAMATH_CALUDE_genetic_material_distribution_l3317_331793


namespace NUMINAMATH_CALUDE_simplify_and_solve_for_t_l3317_331715

theorem simplify_and_solve_for_t
  (m Q : ℝ)
  (j : ℝ)
  (h : j ≠ -2)
  (h_pos_m : m > 0)
  (h_pos_Q : Q > 0)
  (h_eq : Q = m / (2 + j) ^ t) :
  t = Real.log (m / Q) / Real.log (2 + j) :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_solve_for_t_l3317_331715


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3317_331706

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 4/9 →
  ((4/3) * Real.pi * r^3) / ((4/3) * Real.pi * R^3) = 8/27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3317_331706


namespace NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l3317_331756

/-- Represents the cost calculation for filling a bathtub with jello. -/
def jello_bathtub_cost (
  jello_mix_per_pound : Real
) (
  bathtub_capacity : Real
) (
  cubic_foot_to_gallon : Real
) (
  gallon_weight : Real
) (
  jello_mix_cost : Real
) : Real :=
  bathtub_capacity * cubic_foot_to_gallon * gallon_weight * jello_mix_per_pound * jello_mix_cost

/-- Theorem stating that the cost to fill the bathtub with jello is $270. -/
theorem jello_bathtub_cost_is_270 :
  jello_bathtub_cost 1.5 6 7.5 8 0.5 = 270 := by
  sorry

#check jello_bathtub_cost_is_270

end NUMINAMATH_CALUDE_jello_bathtub_cost_is_270_l3317_331756


namespace NUMINAMATH_CALUDE_factorization_equality_l3317_331763

theorem factorization_equality (a b : ℝ) : a^3 - 2*a^2*b + a*b^2 = a*(a-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3317_331763


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3317_331790

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (PQ PR PS QR QS RS : ℝ) : ℝ :=
  -- Define the volume calculation here
  sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 15√2 / 2 -/
theorem volume_of_specific_tetrahedron :
  let PQ : ℝ := 6
  let PR : ℝ := 4
  let PS : ℝ := 5
  let QR : ℝ := 5
  let QS : ℝ := 3
  let RS : ℝ := 15 / 4 * Real.sqrt 2
  tetrahedron_volume PQ PR PS QR QS RS = 15 / 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l3317_331790


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3317_331767

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def has_equal_intercepts (l : Line) : Prop :=
  l.a = l.b ∨ l.a = -l.b

theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line),
    (passes_through l1 ⟨1, 2⟩ ∧ has_equal_intercepts l1) ∧
    (passes_through l2 ⟨1, 2⟩ ∧ has_equal_intercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3317_331767


namespace NUMINAMATH_CALUDE_fraction_chain_l3317_331739

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 3)
  (h3 : c / d = 5) :
  d / a = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_chain_l3317_331739


namespace NUMINAMATH_CALUDE_car_speed_l3317_331782

/-- Given a car that travels 390 miles in 6 hours, prove its speed is 65 miles per hour -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 390 ∧ time = 6 ∧ speed = distance / time → speed = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l3317_331782


namespace NUMINAMATH_CALUDE_logarithm_simplification_l3317_331707

theorem logarithm_simplification 
  (m n p q x z : ℝ) 
  (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) (hx : x > 0) (hz : z > 0) : 
  Real.log (m / n) + Real.log (n / p) + Real.log (p / q) - Real.log (m * x / (q * z)) = Real.log (z / x) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l3317_331707


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3317_331792

-- Define the sets M and N
def M : Set ℝ := {x | x < 0 ∨ x > 2}
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem complement_intersection_theorem :
  (N \ (M ∩ N)) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3317_331792


namespace NUMINAMATH_CALUDE_division_problem_l3317_331755

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 167 →
  quotient = 9 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 18 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3317_331755


namespace NUMINAMATH_CALUDE_first_player_wins_l3317_331743

/-- Represents a move in the coin game -/
structure Move where
  player : Nat
  coins : Nat

/-- Represents the state of the game -/
structure GameState where
  coins : Nat
  turn : Nat

/-- Checks if a move is valid for a given player -/
def isValidMove (m : Move) (gs : GameState) : Prop :=
  (m.player = gs.turn % 2) ∧
  (if m.player = 0
   then m.coins % 2 = 1 ∧ m.coins ≥ 1 ∧ m.coins ≤ 99
   else m.coins % 2 = 0 ∧ m.coins ≥ 2 ∧ m.coins ≤ 100) ∧
  (m.coins ≤ gs.coins)

/-- Applies a move to a game state -/
def applyMove (m : Move) (gs : GameState) : GameState :=
  { coins := gs.coins - m.coins, turn := gs.turn + 1 }

/-- Defines a winning strategy for the first player -/
def firstPlayerStrategy (gs : GameState) : Move :=
  if gs.turn = 0 then
    { player := 0, coins := 99 }
  else
    { player := 0, coins := 101 - (gs.coins % 101) }

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (gs : GameState),
      gs.coins = 2019 →
      (∀ (m : Move), isValidMove m gs → 
        ∃ (nextMove : Move), 
          isValidMove nextMove (applyMove m gs) ∧
          strategy (applyMove m gs) = nextMove) ∧
      (∀ (sequence : Nat → Move),
        (∀ (i : Nat), isValidMove (sequence i) (applyMove (sequence (i-1)) gs)) →
        ∃ (n : Nat), ¬isValidMove (sequence n) (applyMove (sequence (n-1)) gs)) :=
sorry


end NUMINAMATH_CALUDE_first_player_wins_l3317_331743


namespace NUMINAMATH_CALUDE_min_distance_line_curve_l3317_331774

/-- The line represented by the parametric equations x = t, y = 6 - 2t -/
def line (t : ℝ) : ℝ × ℝ := (t, 6 - 2*t)

/-- The curve represented by the equation (x - 1)² + (y + 2)² = 5 -/
def curve (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

/-- The minimum distance between a point on the line and a point on the curve -/
theorem min_distance_line_curve :
  ∃ (d : ℝ), d = Real.sqrt 5 / 5 ∧
  ∀ (t θ : ℝ),
    let (x₁, y₁) := line t
    let (x₂, y₂) := (1 + Real.sqrt 5 * Real.cos θ, -2 + Real.sqrt 5 * Real.sin θ)
    curve x₂ y₂ →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_curve_l3317_331774


namespace NUMINAMATH_CALUDE_distribute_five_gifts_to_three_fans_l3317_331780

/-- The number of ways to distribute n identical gifts to k different fans,
    where each fan receives at least one gift -/
def distribute_gifts (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 identical gifts to 3 different fans,
    where each fan receives at least one gift, can be done in 6 ways -/
theorem distribute_five_gifts_to_three_fans :
  distribute_gifts 5 3 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_five_gifts_to_three_fans_l3317_331780


namespace NUMINAMATH_CALUDE_fourth_vertex_coordinates_l3317_331741

/-- A regular tetrahedron with integer coordinates -/
structure RegularTetrahedron where
  v1 : ℤ × ℤ × ℤ
  v2 : ℤ × ℤ × ℤ
  v3 : ℤ × ℤ × ℤ
  v4 : ℤ × ℤ × ℤ
  is_regular : True  -- Placeholder for the regularity condition

/-- The fourth vertex of the regular tetrahedron -/
def fourth_vertex (t : RegularTetrahedron) : ℤ × ℤ × ℤ := t.v4

/-- The theorem stating the coordinates of the fourth vertex -/
theorem fourth_vertex_coordinates (t : RegularTetrahedron) 
  (h1 : t.v1 = (0, 1, 2))
  (h2 : t.v2 = (4, 2, 1))
  (h3 : t.v3 = (3, 1, 5)) :
  fourth_vertex t = (3, -2, 2) := by sorry

end NUMINAMATH_CALUDE_fourth_vertex_coordinates_l3317_331741


namespace NUMINAMATH_CALUDE_discount_comparison_l3317_331714

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.15, 0.15, 0.05]
def option2_discounts : List ℝ := [0.30, 0.10, 0.02]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem discount_comparison :
  apply_successive_discounts initial_amount option1_discounts -
  apply_successive_discounts initial_amount option2_discounts = 1379.50 :=
sorry

end NUMINAMATH_CALUDE_discount_comparison_l3317_331714


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2926_l3317_331730

theorem smallest_prime_factor_of_2926 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2926 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2926 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2926_l3317_331730


namespace NUMINAMATH_CALUDE_polygon_with_30_degree_exterior_angles_has_12_sides_l3317_331731

/-- A polygon with exterior angles each measuring 30° has 12 sides -/
theorem polygon_with_30_degree_exterior_angles_has_12_sides :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 30 →
    (n : ℝ) * exterior_angle = 360 →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_30_degree_exterior_angles_has_12_sides_l3317_331731


namespace NUMINAMATH_CALUDE_residue_calculation_l3317_331764

theorem residue_calculation : (222 * 15 - 35 * 9 + 2^3) % 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l3317_331764


namespace NUMINAMATH_CALUDE_stevens_apples_l3317_331752

/-- The number of apples Steven has set aside to meet his seed collection goal. -/
def apples_set_aside : ℕ :=
  let total_seeds_needed : ℕ := 60
  let seeds_per_apple : ℕ := 6
  let seeds_per_pear : ℕ := 2
  let seeds_per_grape : ℕ := 3
  let pears : ℕ := 3
  let grapes : ℕ := 9
  let seeds_short : ℕ := 3

  let seeds_from_pears : ℕ := pears * seeds_per_pear
  let seeds_from_grapes : ℕ := grapes * seeds_per_grape
  let seeds_collected : ℕ := total_seeds_needed - seeds_short
  let seeds_from_apples : ℕ := seeds_collected - seeds_from_pears - seeds_from_grapes

  seeds_from_apples / seeds_per_apple

theorem stevens_apples :
  apples_set_aside = 4 :=
by sorry

end NUMINAMATH_CALUDE_stevens_apples_l3317_331752


namespace NUMINAMATH_CALUDE_cost_difference_is_two_point_five_l3317_331771

/-- Represents the pizza sharing scenario between Bob and Samantha -/
structure PizzaSharing where
  totalSlices : ℕ
  plainPizzaCost : ℚ
  oliveCost : ℚ
  bobOliveSlices : ℕ
  bobPlainSlices : ℕ

/-- Calculates the cost difference between Bob and Samantha's payments -/
def costDifference (ps : PizzaSharing) : ℚ :=
  let totalCost := ps.plainPizzaCost + ps.oliveCost
  let costPerSlice := totalCost / ps.totalSlices
  let bobCost := costPerSlice * (ps.bobOliveSlices + ps.bobPlainSlices)
  let samanthaCost := costPerSlice * (ps.totalSlices - ps.bobOliveSlices - ps.bobPlainSlices)
  bobCost - samanthaCost

/-- Theorem stating that the cost difference is $2.5 -/
theorem cost_difference_is_two_point_five :
  let ps : PizzaSharing := {
    totalSlices := 12,
    plainPizzaCost := 12,
    oliveCost := 3,
    bobOliveSlices := 4,
    bobPlainSlices := 3
  }
  costDifference ps = 5/2 := by sorry

end NUMINAMATH_CALUDE_cost_difference_is_two_point_five_l3317_331771


namespace NUMINAMATH_CALUDE_gain_percentage_proof_l3317_331754

/-- Proves that the gain percentage is 20% when an article is sold for 168 Rs,
    given that it incurs a 15% loss when sold for 119 Rs. -/
theorem gain_percentage_proof (cost_price : ℝ) : 
  (cost_price * 0.85 = 119) →  -- 15% loss when sold for 119
  ((168 - cost_price) / cost_price * 100 = 20) := by
sorry

end NUMINAMATH_CALUDE_gain_percentage_proof_l3317_331754


namespace NUMINAMATH_CALUDE_factorization_expression1_l3317_331733

theorem factorization_expression1 (x y : ℝ) : 2 * x^2 * y - 4 * x * y + 2 * y = 2 * y * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_expression1_l3317_331733


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l3317_331742

theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h_girls : girls = 17)
  (h_boys : boys = 32)
  (h_callback : callback = 10) :
  girls + boys - callback = 39 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l3317_331742


namespace NUMINAMATH_CALUDE_line_and_circle_properties_l3317_331751

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 2 * k = 0

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the line l₀
def line_l0 (x y : ℝ) : Prop := x - 2 * y + 2 = 0

theorem line_and_circle_properties :
  (∃ k : ℝ, ∀ x y : ℝ, line_l k x y → line_l0 x y → x = y) ∧
  (∀ k : ℝ, ∃ x y : ℝ, line_l k x y ∧ circle_O x y) :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_properties_l3317_331751


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l3317_331736

theorem interval_length_implies_difference (r s : ℝ) : 
  (∀ x, r ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ s) → 
  ((s - 4) / 3 - (r - 4) / 3 = 12) → 
  s - r = 36 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l3317_331736


namespace NUMINAMATH_CALUDE_percentage_equality_l3317_331748

theorem percentage_equality (x y : ℝ) (h : (18 / 100) * x = (9 / 100) * y) :
  (12 / 100) * x = (6 / 100) * y := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3317_331748


namespace NUMINAMATH_CALUDE_ceiling_times_self_156_l3317_331770

theorem ceiling_times_self_156 :
  ∃! (y : ℝ), ⌈y⌉ * y = 156 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ceiling_times_self_156_l3317_331770


namespace NUMINAMATH_CALUDE_chord_slope_l3317_331758

theorem chord_slope (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + (y-1)^2 = 3 ∧ y = k*x - 1) ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + (y1-1)^2 = 3 ∧ y1 = k*x1 - 1 ∧
    x2^2 + (y2-1)^2 = 3 ∧ y2 = k*x2 - 1 ∧
    (x1-x2)^2 + (y1-y2)^2 = 4) →
  k = 1 ∨ k = -1 :=
by sorry

end NUMINAMATH_CALUDE_chord_slope_l3317_331758


namespace NUMINAMATH_CALUDE_distance_after_translation_l3317_331786

/-- The distance between two points after translation --/
theorem distance_after_translation (x1 y1 x2 y2 tx ty : ℝ) :
  let p1 := (x1 + tx, y1 + ty)
  let p2 := (x2 + tx, y2 + ty)
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 73 :=
by
  sorry

#check distance_after_translation 5 3 (-3) 0 3 (-3)

end NUMINAMATH_CALUDE_distance_after_translation_l3317_331786


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_greater_than_four_l3317_331776

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: If point P(4-a, 2) is in the second quadrant, then a > 4 -/
theorem point_in_second_quadrant_implies_a_greater_than_four (a : ℝ) :
  SecondQuadrant ⟨4 - a, 2⟩ → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_a_greater_than_four_l3317_331776


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l3317_331710

theorem complex_magnitude_example : Complex.abs (1 - Complex.I / 2) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l3317_331710


namespace NUMINAMATH_CALUDE_mikes_shopping_expense_l3317_331773

/-- Calculates the total amount Mike spent given the costs and discounts of items. -/
def total_spent (food_cost wallet_cost shirt_cost shoes_cost belt_cost : ℚ)
  (shirt_discount shoes_discount belt_discount : ℚ) : ℚ :=
  food_cost + wallet_cost +
  shirt_cost * (1 - shirt_discount) +
  shoes_cost * (1 - shoes_discount) +
  belt_cost * (1 - belt_discount)

/-- Theorem stating the total amount Mike spent given the conditions. -/
theorem mikes_shopping_expense :
  let food_cost : ℚ := 30
  let wallet_cost : ℚ := food_cost + 60
  let shirt_cost : ℚ := wallet_cost / 3
  let shoes_cost : ℚ := 2 * wallet_cost
  let belt_cost : ℚ := shoes_cost - 45
  let shirt_discount : ℚ := 20 / 100
  let shoes_discount : ℚ := 15 / 100
  let belt_discount : ℚ := 10 / 100
  total_spent food_cost wallet_cost shirt_cost shoes_cost belt_cost
    shirt_discount shoes_discount belt_discount = 418.5 := by sorry

end NUMINAMATH_CALUDE_mikes_shopping_expense_l3317_331773


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l3317_331725

theorem pyramid_base_side_length (area : ℝ) (slant_height : ℝ) (h1 : area = 120) (h2 : slant_height = 40) :
  ∃ (side_length : ℝ), side_length = 6 ∧ (1/2) * side_length * slant_height = area :=
sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l3317_331725


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3317_331783

theorem min_distance_to_origin (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 4 = 0) :
  ∃ (min_dist : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 6*b + 4 = 0 → 
    min_dist ≤ Real.sqrt (a^2 + b^2)) ∧ min_dist = Real.sqrt 13 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3317_331783


namespace NUMINAMATH_CALUDE_lcm_24_36_45_l3317_331761

theorem lcm_24_36_45 : Nat.lcm 24 (Nat.lcm 36 45) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_45_l3317_331761


namespace NUMINAMATH_CALUDE_chairs_to_remove_l3317_331777

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_students : ℕ)
  (h1 : initial_chairs = 156)
  (h2 : chairs_per_row = 13)
  (h3 : expected_students = 95)
  (h4 : initial_chairs % chairs_per_row = 0) -- All rows are initially completely filled
  : ∃ (removed_chairs : ℕ),
    removed_chairs = 52 ∧
    (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧ -- Remaining rows are completely filled
    (initial_chairs - removed_chairs) ≥ expected_students ∧ -- Can accommodate all students
    ∀ (x : ℕ), x < removed_chairs →
      (initial_chairs - x < expected_students ∨ (initial_chairs - x) % chairs_per_row ≠ 0) -- Minimizes empty seats
    := by sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l3317_331777


namespace NUMINAMATH_CALUDE_factor_expression_l3317_331749

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3317_331749


namespace NUMINAMATH_CALUDE_fourth_month_sale_l3317_331799

def sales_4_months : List Int := [6335, 6927, 6855, 6562]
def sale_6th_month : Int := 5091
def average_sale : Int := 6500
def num_months : Int := 6

theorem fourth_month_sale :
  let total_sales := average_sale * num_months
  let sum_known_sales := (sales_4_months.sum + sale_6th_month)
  total_sales - sum_known_sales = 7230 := by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l3317_331799


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l3317_331724

theorem polar_to_cartesian :
  let ρ : ℝ := 4
  let θ : ℝ := 2 * π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  x = -2 ∧ y = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l3317_331724


namespace NUMINAMATH_CALUDE_second_number_11th_row_l3317_331753

/-- Given a lattice with 11 rows, where each row contains 6 numbers,
    and the last number in each row is n × 6 (where n is the row number),
    prove that the second number in the 11th row is 62. -/
theorem second_number_11th_row (rows : Nat) (numbers_per_row : Nat)
    (last_number : Nat → Nat) :
  rows = 11 →
  numbers_per_row = 6 →
  (∀ n, last_number n = n * numbers_per_row) →
  (last_number 10 + 2 = 62) := by
  sorry

end NUMINAMATH_CALUDE_second_number_11th_row_l3317_331753


namespace NUMINAMATH_CALUDE_product_less_than_factor_l3317_331795

theorem product_less_than_factor : ∃ (a b : ℝ), 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ a * b < min a b := by
  sorry

end NUMINAMATH_CALUDE_product_less_than_factor_l3317_331795


namespace NUMINAMATH_CALUDE_special_function_sum_l3317_331722

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x^3) = (f x)^3) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂)

/-- Theorem stating the sum of f(0), f(1), and f(-1) for a special function -/
theorem special_function_sum (f : ℝ → ℝ) (h : SpecialFunction f) :
  f 0 + f 1 + f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_special_function_sum_l3317_331722


namespace NUMINAMATH_CALUDE_right_triangle_solution_l3317_331766

-- Define the right triangle
def RightTriangle (a b c h : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧ a^2 + b^2 = c^2 ∧ h^2 = (a^2 * b^2) / c^2

-- Define the conditions
def TriangleConditions (a b c h e d : ℝ) : Prop :=
  RightTriangle a b c h ∧ 
  (c^2 / (2*h) - h/2 = e) ∧  -- Difference between hypotenuse segments
  (a - b = d)                -- Difference between legs

-- Theorem statement
theorem right_triangle_solution (e d : ℝ) (he : e = 37.0488) (hd : d = 31) :
  ∃ (a b c h : ℝ), TriangleConditions a b c h e d ∧ 
    (a = 40 ∧ b = 9 ∧ c = 41) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_solution_l3317_331766


namespace NUMINAMATH_CALUDE_rabbit_farm_number_l3317_331760

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem rabbit_farm_number : 
  ∃! n : ℕ, is_six_digit n ∧ 
            is_perfect_square n ∧ 
            is_perfect_cube n ∧ 
            is_prime (n - 6) ∧ 
            n = 117649 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_farm_number_l3317_331760


namespace NUMINAMATH_CALUDE_average_after_removal_l3317_331709

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 12 →
  sum / 12 = 90 →
  sum = Finset.sum numbers id →
  68 ∈ numbers →
  75 ∈ numbers →
  82 ∈ numbers →
  (sum - 68 - 75 - 82) / 9 = 95 := by
sorry

end NUMINAMATH_CALUDE_average_after_removal_l3317_331709


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3317_331794

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℤ, x^3 < 1) ↔ (∃ x : ℤ, x^3 ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3317_331794


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3317_331797

theorem binomial_coefficient_ratio (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (3*x - 2)^6 = a₀ + a₁*(2*x - 1) + a₂*(2*x - 1)^2 + a₃*(2*x - 1)^3 + 
                 a₄*(2*x - 1)^4 + a₅*(2*x - 1)^5 + a₆*(2*x - 1)^6 →
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63/65 := by
sorry


end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3317_331797


namespace NUMINAMATH_CALUDE_parallelogram_area_l3317_331727

/-- Given a parallelogram with height 1 and a right triangle within it with legs 40 and (55 - a),
    where a is the length of the shorter side, prove that its area is 200/3. -/
theorem parallelogram_area (a : ℝ) (h : a > 0) :
  let height : ℝ := 1
  let leg1 : ℝ := 40
  let leg2 : ℝ := 55 - a
  let area : ℝ := a * leg1
  (leg1 ^ 2 + leg2 ^ 2 = (height * area) ^ 2) → area = 200 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3317_331727


namespace NUMINAMATH_CALUDE_antonia_pillbox_weeks_l3317_331738

/-- Represents the number of weeks Antonia filled her pillbox -/
def weeks_filled (total_pills : ℕ) (pills_per_week : ℕ) (pills_left : ℕ) : ℕ :=
  (total_pills - pills_left) / pills_per_week

/-- Theorem stating that Antonia filled her pillbox for 2 weeks -/
theorem antonia_pillbox_weeks :
  let num_supplements : ℕ := 5
  let bottles_120 : ℕ := 3
  let bottles_30 : ℕ := 2
  let days_in_week : ℕ := 7
  let pills_left : ℕ := 350

  let total_pills : ℕ := bottles_120 * 120 + bottles_30 * 30
  let pills_per_week : ℕ := num_supplements * days_in_week

  weeks_filled total_pills pills_per_week pills_left = 2 := by
  sorry

#check antonia_pillbox_weeks

end NUMINAMATH_CALUDE_antonia_pillbox_weeks_l3317_331738


namespace NUMINAMATH_CALUDE_tangent_line_intersection_product_l3317_331769

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = x / 2
def asymptote2 (x y : ℝ) : Prop := y = -x / 2

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := hyperbola P.1 P.2

-- Define a line tangent to the hyperbola at point P
def tangent_line (l : ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  point_on_hyperbola P ∧ l P

-- Define intersection points of the tangent line with asymptotes
def intersection_points (l : ℝ × ℝ → Prop) (M N : ℝ × ℝ) : Prop :=
  l M ∧ l N ∧ asymptote1 M.1 M.2 ∧ asymptote2 N.1 N.2

-- Theorem statement
theorem tangent_line_intersection_product (P M N : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  point_on_hyperbola P →
  tangent_line l P →
  intersection_points l M N →
  M.1 * N.1 + M.2 * N.2 = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_product_l3317_331769


namespace NUMINAMATH_CALUDE_starship_sales_l3317_331789

theorem starship_sales (starship_price mech_price ultimate_price : ℕ)
                       (total_items total_revenue : ℕ) :
  starship_price = 8 →
  mech_price = 26 →
  ultimate_price = 33 →
  total_items = 31 →
  total_revenue = 370 →
  ∃ (x y : ℕ),
    x + y ≤ total_items ∧
    (total_items - x - y) % 2 = 0 ∧
    x * starship_price + y * mech_price + 
      ((total_items - x - y) / 2) * ultimate_price = total_revenue ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_starship_sales_l3317_331789


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3317_331708

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (634 * n ≡ 1275 * n [ZMOD 30]) ∧ 
  (∀ (m : ℕ), m > 0 → (634 * m ≡ 1275 * m [ZMOD 30]) → n ≤ m) ∧ 
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3317_331708


namespace NUMINAMATH_CALUDE_subset_implies_t_equals_two_l3317_331703

theorem subset_implies_t_equals_two (t : ℝ) : 
  let A : Set ℝ := {1, t, 2*t}
  let B : Set ℝ := {1, t^2}
  B ⊆ A → t = 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_t_equals_two_l3317_331703


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3317_331735

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 0; 7, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -1; 0, 2]
  A * B = !![15, -3; 35, -11] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3317_331735


namespace NUMINAMATH_CALUDE_f_at_neg_one_l3317_331788

/-- The function f(x) = x^3 + x^2 - 2x -/
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x

/-- Theorem: f(-1) = 2 -/
theorem f_at_neg_one : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_neg_one_l3317_331788


namespace NUMINAMATH_CALUDE_jeff_score_problem_l3317_331705

theorem jeff_score_problem (scores : List ℝ) (desired_average : ℝ) : 
  scores = [89, 92, 88, 95, 91] → 
  desired_average = 93 → 
  (scores.sum + 103) / 6 = desired_average :=
by sorry

end NUMINAMATH_CALUDE_jeff_score_problem_l3317_331705


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l3317_331716

/-- Given a quadratic expression 8k^2 - 12k + 20, prove that when rewritten in the form a(k + b)^2 + r, the value of r/b is -47.33 -/
theorem quadratic_rewrite_ratio : 
  ∃ (a b r : ℝ), 
    (∀ k, 8 * k^2 - 12 * k + 20 = a * (k + b)^2 + r) ∧ 
    (r / b = -47.33) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l3317_331716


namespace NUMINAMATH_CALUDE_fraction_simplification_l3317_331768

variables {a b c x y z : ℝ}

theorem fraction_simplification :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz)
  = a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3317_331768


namespace NUMINAMATH_CALUDE_tim_singles_count_l3317_331719

/-- The number of points for a single line -/
def single_points : ℕ := 1000

/-- The number of points for a tetris -/
def tetris_points : ℕ := 8 * single_points

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- The total number of points Tim scored -/
def tim_total_points : ℕ := 38000

/-- The number of singles Tim scored -/
def tim_singles : ℕ := (tim_total_points - tim_tetrises * tetris_points) / single_points

theorem tim_singles_count : tim_singles = 6 := by
  sorry

end NUMINAMATH_CALUDE_tim_singles_count_l3317_331719


namespace NUMINAMATH_CALUDE_phone_cost_calculation_phone_cost_proof_l3317_331796

theorem phone_cost_calculation (current_percentage : Real) (additional_amount : Real) : Real :=
  let total_cost := additional_amount / (1 - current_percentage)
  total_cost

theorem phone_cost_proof :
  phone_cost_calculation 0.4 780 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_phone_cost_calculation_phone_cost_proof_l3317_331796


namespace NUMINAMATH_CALUDE_log_ratio_simplification_l3317_331701

theorem log_ratio_simplification (x : ℝ) 
  (h1 : 5 * x^3 > 0) (h2 : 7 * x - 3 > 0) : 
  (Real.log (Real.sqrt (7 * x - 3)) / Real.log (5 * x^3)) / Real.log (7 * x - 3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_simplification_l3317_331701


namespace NUMINAMATH_CALUDE_diego_apple_capacity_l3317_331720

/-- The maximum weight of apples Diego can buy given his carrying capacity and other fruit weights -/
theorem diego_apple_capacity (capacity : ℝ) (watermelon grapes oranges bananas : ℝ) 
  (h_capacity : capacity = 50) 
  (h_watermelon : watermelon = 1.5)
  (h_grapes : grapes = 2.75)
  (h_oranges : oranges = 3.5)
  (h_bananas : bananas = 2.7) :
  capacity - (watermelon + grapes + oranges + bananas) = 39.55 := by
  sorry

#check diego_apple_capacity

end NUMINAMATH_CALUDE_diego_apple_capacity_l3317_331720


namespace NUMINAMATH_CALUDE_consecutive_pair_with_17_l3317_331729

theorem consecutive_pair_with_17 (a b : ℤ) : 
  (a = 17 ∨ b = 17) → 
  (abs (a - b) = 1) → 
  (a + b = 35) → 
  (35 % 5 = 0) → 
  ((a = 17 ∧ b = 18) ∨ (a = 18 ∧ b = 17)) := by sorry

end NUMINAMATH_CALUDE_consecutive_pair_with_17_l3317_331729


namespace NUMINAMATH_CALUDE_subtraction_decimal_result_l3317_331713

theorem subtraction_decimal_result : 5.3567 - 2.1456 - 1.0211 = 2.1900 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_decimal_result_l3317_331713


namespace NUMINAMATH_CALUDE_total_buttons_is_1600_l3317_331781

/-- The number of 3-button shirts ordered -/
def shirts_3_button : ℕ := 200

/-- The number of 5-button shirts ordered -/
def shirts_5_button : ℕ := 200

/-- The number of buttons on a 3-button shirt -/
def buttons_per_3_button_shirt : ℕ := 3

/-- The number of buttons on a 5-button shirt -/
def buttons_per_5_button_shirt : ℕ := 5

/-- The total number of buttons used for all shirts -/
def total_buttons : ℕ := shirts_3_button * buttons_per_3_button_shirt + 
                          shirts_5_button * buttons_per_5_button_shirt

theorem total_buttons_is_1600 : total_buttons = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_buttons_is_1600_l3317_331781


namespace NUMINAMATH_CALUDE_area_calculation_l3317_331721

/-- The lower boundary function of the region -/
def lower_bound (x : ℝ) : ℝ := |x - 4|

/-- The upper boundary function of the region -/
def upper_bound (x : ℝ) : ℝ := 5 - |x - 2|

/-- The region in the xy-plane -/
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lower_bound p.1 ≤ p.2 ∧ p.2 ≤ upper_bound p.1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_calculation : area_of_region = 12.875 := by sorry

end NUMINAMATH_CALUDE_area_calculation_l3317_331721


namespace NUMINAMATH_CALUDE_tims_pencils_count_l3317_331765

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 2

/-- The total number of pencils after Tim's action -/
def total_pencils : ℕ := 5

/-- The number of pencils Tim placed in the drawer -/
def tims_pencils : ℕ := total_pencils - initial_pencils

theorem tims_pencils_count : tims_pencils = 3 := by
  sorry

end NUMINAMATH_CALUDE_tims_pencils_count_l3317_331765


namespace NUMINAMATH_CALUDE_quadratic_form_constant_l3317_331728

theorem quadratic_form_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_constant_l3317_331728


namespace NUMINAMATH_CALUDE_unique_solution_system_l3317_331700

theorem unique_solution_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3317_331700


namespace NUMINAMATH_CALUDE_ladder_length_l3317_331744

/-- Given a right triangle with an adjacent side of 6.4 meters and an angle of 59.5 degrees
    between the adjacent side and the hypotenuse, the length of the hypotenuse is
    approximately 12.43 meters. -/
theorem ladder_length (adjacent : ℝ) (angle : ℝ) (hypotenuse : ℝ) 
    (h_adjacent : adjacent = 6.4)
    (h_angle : angle = 59.5 * π / 180) -- Convert degrees to radians
    (h_cos : Real.cos angle = adjacent / hypotenuse) :
  abs (hypotenuse - 12.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l3317_331744


namespace NUMINAMATH_CALUDE_julie_work_hours_l3317_331759

/-- Given Julie's work schedule and earnings, calculate her required weekly hours during the school year --/
theorem julie_work_hours (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ)
                         (school_year_weeks : ℕ) (school_year_earnings : ℕ) :
  summer_weeks = 12 →
  summer_hours_per_week = 48 →
  summer_earnings = 5000 →
  school_year_weeks = 48 →
  school_year_earnings = 5000 →
  (summer_hours_per_week * summer_weeks * school_year_earnings) / (summer_earnings * school_year_weeks) = 12 := by
  sorry

end NUMINAMATH_CALUDE_julie_work_hours_l3317_331759


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3317_331704

theorem exponential_equation_solution :
  ∃ x : ℝ, (2 : ℝ) ^ (x + 6) = 64 ^ (x - 1) ∧ x = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3317_331704


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3317_331723

theorem system_solution_ratio (a b x y : ℝ) (h1 : 4 * x - 2 * y = a) 
  (h2 : 9 * y - 18 * x = b) (h3 : b ≠ 0) : a / b = -2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3317_331723


namespace NUMINAMATH_CALUDE_range_of_m_l3317_331762

def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*m*x + 7*m - 10 ≠ 0

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x^2 - m*x + 4 ≥ 0

theorem range_of_m :
  ∀ m : ℝ, (prop_p m ∨ prop_q m) ∧ (prop_p m ∧ prop_q m) →
  m ∈ Set.Ioo 2 4 ∪ {4} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3317_331762


namespace NUMINAMATH_CALUDE_spatial_geometry_theorem_l3317_331717

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- State the theorem
theorem spatial_geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular m β ∧ perpendicular n β → parallel_lines m n) ∧
  (perpendicular m α ∧ perpendicular m β → parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_spatial_geometry_theorem_l3317_331717
