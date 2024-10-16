import Mathlib

namespace NUMINAMATH_CALUDE_planes_perpendicular_from_line_perpendicular_planes_from_parallel_l994_99416

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem 1
theorem planes_perpendicular_from_line 
  (α β : Plane) (l : Line) :
  line_perpendicular l α → line_parallel l β → perpendicular α β := by sorry

-- Theorem 2
theorem perpendicular_planes_from_parallel 
  (α β γ : Plane) :
  parallel α β → perpendicular α γ → perpendicular β γ := by sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_line_perpendicular_planes_from_parallel_l994_99416


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l994_99435

/-- The cost of pencils and pens -/
theorem pencil_pen_cost (x y : ℚ) 
  (h1 : 8 * x + 3 * y = 5.1)
  (h2 : 3 * x + 5 * y = 4.95) :
  4 * x + 4 * y = 4.488 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l994_99435


namespace NUMINAMATH_CALUDE_incorrect_step_identification_l994_99424

theorem incorrect_step_identification :
  (2 * Real.sqrt 3 = Real.sqrt (2^2 * 3)) ∧
  (2 * Real.sqrt 3 ≠ -2 * Real.sqrt 3) ∧
  (Real.sqrt ((-2)^2 * 3) ≠ -2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_step_identification_l994_99424


namespace NUMINAMATH_CALUDE_problem_solution_l994_99443

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l994_99443


namespace NUMINAMATH_CALUDE_simplify_expression_l994_99485

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l994_99485


namespace NUMINAMATH_CALUDE_quadratic_function_max_a_l994_99495

theorem quadratic_function_max_a (a b c m : ℝ) : 
  a < 0 →
  a * m^2 + b * m + c = b →
  a * (m + 1)^2 + b * (m + 1) + c = a →
  b ≥ a →
  m < 0 →
  (∀ x, a * x^2 + b * x + c ≤ -2) →
  (∀ a', a' < 0 → 
    (∀ x, a' * x^2 + (-a' * m) * x + (-a' * m) ≤ -2) → 
    a' ≤ a) →
  a = -8/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_a_l994_99495


namespace NUMINAMATH_CALUDE_pie_sugar_percentage_l994_99498

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage 
  (total_weight : ℝ) 
  (sugar_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : sugar_weight = 50) : 
  (total_weight - sugar_weight) / total_weight * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_pie_sugar_percentage_l994_99498


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l994_99456

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking_height * water_density

/-- Theorem stating the mass of the man in the given problem. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 9
  let boat_breadth : ℝ := 3
  let boat_sinking_height : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth boat_sinking_height water_density = 270 :=
by sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l994_99456


namespace NUMINAMATH_CALUDE_club_officer_selection_l994_99444

/-- The number of ways to select officers in a club with special conditions -/
def select_officers (total_members : ℕ) (officers_needed : ℕ) (special_members : ℕ) : ℕ :=
  let remaining_members := total_members - special_members
  let case1 := remaining_members * (remaining_members - 1) * (remaining_members - 2) * (remaining_members - 3)
  let case2 := officers_needed * (officers_needed - 1) * (officers_needed - 2) * remaining_members
  case1 + case2

/-- Theorem stating the number of ways to select officers under given conditions -/
theorem club_officer_selection :
  select_officers 25 4 3 = 176088 :=
sorry

end NUMINAMATH_CALUDE_club_officer_selection_l994_99444


namespace NUMINAMATH_CALUDE_person_age_l994_99425

theorem person_age : ∃ x : ℕ, x = 30 ∧ 3 * (x + 5) - 3 * (x - 5) = x := by sorry

end NUMINAMATH_CALUDE_person_age_l994_99425


namespace NUMINAMATH_CALUDE_power_of_negative_square_l994_99474

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l994_99474


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l994_99448

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  asymptote1 : ℝ → ℝ
  asymptote2 : ℝ → ℝ
  point : ℝ × ℝ

/-- The distance between the foci of a hyperbola -/
def focalDistance (h : Hyperbola) : ℝ := sorry

/-- Theorem: For a hyperbola with asymptotes y = x + 3 and y = -x + 5, 
    passing through the point (4,6), the distance between its foci is 2√10 -/
theorem hyperbola_focal_distance :
  let h : Hyperbola := {
    asymptote1 := fun x ↦ x + 3,
    asymptote2 := fun x ↦ -x + 5,
    point := (4, 6)
  }
  focalDistance h = 2 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l994_99448


namespace NUMINAMATH_CALUDE_correct_product_l994_99483

/-- Given two positive integers a and b, where a is a two-digit number,
    if the product of the reversed digits of a and b is 161,
    then the product of a and b is 224. -/
theorem correct_product (a b : ℕ) : 
  a ≥ 10 ∧ a ≤ 99 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (10 * (a % 10) + a / 10) * b = 161 →  -- reversed a * b = 161
  a * b = 224 :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l994_99483


namespace NUMINAMATH_CALUDE_statements_equivalence_l994_99470

-- Define the propositions
variable (S : Prop) -- Saturn is visible from Earth tonight
variable (M : Prop) -- Mars is visible

-- Define the statements
def statement1 : Prop := S → ¬M
def statement2 : Prop := M → ¬S
def statement3 : Prop := ¬S ∨ ¬M

-- Theorem stating the equivalence of the statements
theorem statements_equivalence : statement1 S M ↔ statement2 S M ∧ statement3 S M := by
  sorry

end NUMINAMATH_CALUDE_statements_equivalence_l994_99470


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l994_99468

/-- Represents a right triangle with mutually externally tangent circles at its vertices -/
structure TriangleWithCircles where
  /-- Side lengths of the right triangle -/
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  /-- Radii of the circles centered at the vertices -/
  r1 : ℝ
  r2 : ℝ
  r3 : ℝ
  /-- Conditions for the triangle and circles -/
  triangle_sides : side1^2 + side2^2 = hypotenuse^2
  circle_tangency1 : r1 + r2 = side1
  circle_tangency2 : r1 + r3 = side2
  circle_tangency3 : r2 + r3 = hypotenuse

/-- The sum of the areas of the three circles in a 6-8-10 right triangle with
    mutually externally tangent circles at its vertices is 56π -/
theorem sum_of_circle_areas (t : TriangleWithCircles)
    (h1 : t.side1 = 6)
    (h2 : t.side2 = 8)
    (h3 : t.hypotenuse = 10) :
  π * (t.r1^2 + t.r2^2 + t.r3^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l994_99468


namespace NUMINAMATH_CALUDE_bus_bike_time_difference_l994_99418

/-- Proves that the difference between bus and bike commute times is 10 minutes -/
theorem bus_bike_time_difference :
  ∀ (bus_time : ℕ),
  (30 + 3 * bus_time + 10 = 160) →
  (bus_time - 30 = 10) := by
  sorry

end NUMINAMATH_CALUDE_bus_bike_time_difference_l994_99418


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l994_99460

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l994_99460


namespace NUMINAMATH_CALUDE_sector_area_l994_99422

/-- The area of a sector with given arc length and diameter -/
theorem sector_area (arc_length diameter : ℝ) (h1 : arc_length = 30) (h2 : diameter = 16) :
  (1 / 2) * (diameter / 2) * arc_length = 120 := by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l994_99422


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l994_99408

theorem arithmetic_calculations : 
  ((82 - 15) * (32 + 18) = 3350) ∧ ((25 + 4) * 75 = 2175) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l994_99408


namespace NUMINAMATH_CALUDE_parabola_shift_l994_99427

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shift_right (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ :=
  λ x => f (x - units)

def shift_down (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ :=
  λ x => f x - units

def final_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem parabola_shift :
  ∀ x : ℝ, shift_down (shift_right original_function 2) 1 x = final_function x :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l994_99427


namespace NUMINAMATH_CALUDE_factor_tree_value_l994_99482

theorem factor_tree_value (X Y Z F G : ℕ) : 
  X = Y * Z ∧
  Y = 7 * F ∧
  F = 2 * 5 ∧
  Z = 11 * G ∧
  G = 3 * 7 →
  X = 16170 := by sorry

end NUMINAMATH_CALUDE_factor_tree_value_l994_99482


namespace NUMINAMATH_CALUDE_expression_evaluation_l994_99432

theorem expression_evaluation : 200 * (200 - 3) + (200^2 - 8^2) = 79336 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l994_99432


namespace NUMINAMATH_CALUDE_square_plus_n_eq_square_plus_k_implies_m_le_n_l994_99472

theorem square_plus_n_eq_square_plus_k_implies_m_le_n
  (k m n : ℕ+) 
  (h : m^2 + n = k^2 + k) : 
  m ≤ n := by
sorry

end NUMINAMATH_CALUDE_square_plus_n_eq_square_plus_k_implies_m_le_n_l994_99472


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l994_99428

/-- Represents a parabola in the form y = ax^2 + b --/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a parabola --/
def translate_parabola (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a,
    b := p.a * t.dx^2 + p.b + t.dy }

theorem parabola_translation_theorem (p : Parabola) (t : Translation) :
  p.a = 2 ∧ p.b = 3 ∧ t.dx = 3 ∧ t.dy = 2 →
  let p' := translate_parabola p t
  p'.a = 2 ∧ p'.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l994_99428


namespace NUMINAMATH_CALUDE_expansion_binomial_coefficients_l994_99469

theorem expansion_binomial_coefficients (n : ℕ) : 
  (∃ a d : ℚ, (n.choose 1 : ℚ) = a ∧ 
               (n.choose 2 : ℚ) = a + d ∧ 
               (n.choose 3 : ℚ) = a + 2*d) → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_expansion_binomial_coefficients_l994_99469


namespace NUMINAMATH_CALUDE_total_squat_bench_press_l994_99486

/-- Represents the weight Tony can lift in various exercises --/
structure TonyLift where
  curl : ℝ
  military_press : ℝ
  squat : ℝ
  bench_press : ℝ

/-- Defines Tony's lifting capabilities based on the given conditions --/
def tony_lift : TonyLift where
  curl := 90
  military_press := 2 * 90
  squat := 5 * (2 * 90)
  bench_press := 1.5 * (2 * 90)

/-- Theorem stating the total weight Tony can lift in squat and bench press combined --/
theorem total_squat_bench_press (t : TonyLift) (h : t = tony_lift) : 
  t.squat + t.bench_press = 1170 := by
  sorry

end NUMINAMATH_CALUDE_total_squat_bench_press_l994_99486


namespace NUMINAMATH_CALUDE_largest_sides_is_eight_l994_99490

/-- A convex polygon with exactly five obtuse interior angles -/
structure ConvexPolygon where
  n : ℕ  -- number of sides
  is_convex : Bool
  obtuse_count : ℕ
  h_convex : is_convex = true
  h_obtuse : obtuse_count = 5

/-- The largest possible number of sides for a convex polygon with exactly five obtuse interior angles -/
def largest_sides : ℕ := 8

/-- Theorem stating that the largest possible number of sides for a convex polygon 
    with exactly five obtuse interior angles is 8 -/
theorem largest_sides_is_eight (p : ConvexPolygon) : 
  p.n ≤ largest_sides ∧ 
  ∃ (q : ConvexPolygon), q.n = largest_sides :=
sorry

end NUMINAMATH_CALUDE_largest_sides_is_eight_l994_99490


namespace NUMINAMATH_CALUDE_fib_consecutive_coprime_fib_gcd_l994_99455

/-- Definition of Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- Theorem: Consecutive Fibonacci numbers are coprime -/
theorem fib_consecutive_coprime (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by sorry

/-- Theorem: GCD of Fibonacci numbers -/
theorem fib_gcd (m n : ℕ) : Nat.gcd (fib m) (fib n) = fib (Nat.gcd m n) := by sorry

end NUMINAMATH_CALUDE_fib_consecutive_coprime_fib_gcd_l994_99455


namespace NUMINAMATH_CALUDE_union_of_sets_l994_99480

def set_A : Set ℝ := {x | x * (x + 1) ≤ 0}
def set_B : Set ℝ := {x | -1 < x ∧ x < 1}

theorem union_of_sets : set_A ∪ set_B = {x | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l994_99480


namespace NUMINAMATH_CALUDE_movie_theater_seats_l994_99479

theorem movie_theater_seats (sections : ℕ) (seats_per_section : ℕ) 
  (h1 : sections = 9)
  (h2 : seats_per_section = 30) :
  sections * seats_per_section = 270 := by
sorry

end NUMINAMATH_CALUDE_movie_theater_seats_l994_99479


namespace NUMINAMATH_CALUDE_original_number_proof_l994_99494

theorem original_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 32 = 87 * k) ∧ 
  (∀ m : ℕ, m < 32 → ¬∃ j : ℕ, N - m = 87 * j) → 
  N = 119 :=
sorry

end NUMINAMATH_CALUDE_original_number_proof_l994_99494


namespace NUMINAMATH_CALUDE_picks_theorem_lattice_points_in_triangle_l994_99499

/-- A point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  O : Point

/-- Counts the number of lattice points on a line segment -/
def countLatticePointsOnSegment (p1 p2 : Point) : ℕ :=
  sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℚ :=
  sorry

/-- Counts the number of lattice points inside a triangle -/
def countLatticePointsInside (t : Triangle) : ℕ :=
  sorry

/-- Pick's theorem: S = N + L/2 - 1, where S is the area, N is the number of interior lattice points, and L is the number of boundary lattice points -/
theorem picks_theorem (t : Triangle) (S : ℚ) (N L : ℕ) :
  S = N + L / 2 - 1 →
  S = triangleArea t →
  L = countLatticePointsOnSegment t.A t.B + countLatticePointsOnSegment t.B t.O + countLatticePointsOnSegment t.O t.A →
  N = countLatticePointsInside t :=
  sorry

theorem lattice_points_in_triangle :
  let t : Triangle := { A := { x := 0, y := 30 }, B := { x := 20, y := 10 }, O := { x := 0, y := 0 } }
  countLatticePointsInside t = 271 :=
by sorry

end NUMINAMATH_CALUDE_picks_theorem_lattice_points_in_triangle_l994_99499


namespace NUMINAMATH_CALUDE_unique_six_digit_number_with_permutation_multiples_l994_99404

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧ digits.toFinset.card = 6

def is_permutation (a b : ℕ) : Prop :=
  (a.digits 10).toFinset = (b.digits 10).toFinset

theorem unique_six_digit_number_with_permutation_multiples :
  ∃! n : ℕ, is_six_digit n ∧ has_distinct_digits n ∧
    (∀ k : Fin 5, is_permutation n ((k + 2) * n)) ∧
    n = 142857 := by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_with_permutation_multiples_l994_99404


namespace NUMINAMATH_CALUDE_power_of_64_l994_99450

theorem power_of_64 : (64 : ℝ) ^ (5/6 : ℝ) = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_l994_99450


namespace NUMINAMATH_CALUDE_carousel_candy_leftover_l994_99403

theorem carousel_candy_leftover (num_clowns num_children num_parents num_vendors : ℕ)
  (initial_supply leftover_candies : ℕ)
  (clown_candies child_candies parent_candies vendor_candies : ℕ)
  (prize_candies bulk_purchase_children bulk_purchase_candies : ℕ)
  (h1 : num_clowns = 4)
  (h2 : num_children = 30)
  (h3 : num_parents = 10)
  (h4 : num_vendors = 5)
  (h5 : initial_supply = 2000)
  (h6 : leftover_candies = 700)
  (h7 : clown_candies = 10)
  (h8 : child_candies = 20)
  (h9 : parent_candies = 15)
  (h10 : vendor_candies = 25)
  (h11 : prize_candies = 150)
  (h12 : bulk_purchase_children = 20)
  (h13 : bulk_purchase_candies = 350) :
  initial_supply - (num_clowns * clown_candies + num_children * child_candies +
    num_parents * parent_candies + num_vendors * vendor_candies +
    prize_candies + bulk_purchase_candies) = 685 :=
by sorry

end NUMINAMATH_CALUDE_carousel_candy_leftover_l994_99403


namespace NUMINAMATH_CALUDE_exponential_fraction_simplification_l994_99400

theorem exponential_fraction_simplification :
  (3^1011 + 3^1009) / (3^1011 - 3^1009) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_fraction_simplification_l994_99400


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l994_99453

def A (m : ℝ) : Set ℝ := {2, m^2}
def B : Set ℝ := {0, 1, 3}

theorem sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → A m ∩ B = {1}) ∧
  (∃ m : ℝ, m ≠ 1 ∧ A m ∩ B = {1}) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l994_99453


namespace NUMINAMATH_CALUDE_f_max_value_l994_99419

/-- A function f(x) with specific properties --/
def f (a b : ℝ) (x : ℝ) : ℝ := (4 - x^2) * (a * x^2 + b * x + 5)

/-- The theorem stating the maximum value of f(x) --/
theorem f_max_value (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-3 - x)) →  -- Symmetry condition
  (∃ M : ℝ, ∀ x : ℝ, f a b x ≤ M ∧ ∃ x₀ : ℝ, f a b x₀ = M) →  -- Maximum exists
  (∃ M : ℝ, M = 36 ∧ ∀ x : ℝ, f a b x ≤ M ∧ ∃ x₀ : ℝ, f a b x₀ = M) :=
by
  sorry


end NUMINAMATH_CALUDE_f_max_value_l994_99419


namespace NUMINAMATH_CALUDE_dan_destroyed_balloons_l994_99457

/-- The number of red balloons destroyed by Dan -/
def balloons_destroyed (fred_balloons sam_balloons remaining_balloons : ℝ) : ℝ :=
  fred_balloons + sam_balloons - remaining_balloons

theorem dan_destroyed_balloons :
  balloons_destroyed 10.0 46.0 40 = 16.0 := by
  sorry

end NUMINAMATH_CALUDE_dan_destroyed_balloons_l994_99457


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l994_99459

-- Part 1
theorem simplify_expression_1 (m n : ℝ) :
  (m + n) * (2 * m + n) + n * (m - n) = 2 * m^2 + 4 * m * n := by
  sorry

-- Part 2
theorem simplify_expression_2 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 3) :
  ((x + 3) / x - 2) / ((x^2 - 9) / (4 * x)) = -4 / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l994_99459


namespace NUMINAMATH_CALUDE_prob_select_AB_correct_l994_99466

/-- The number of students in the class -/
def num_students : ℕ := 5

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly A and B -/
def prob_select_AB : ℚ := 1 / 10

theorem prob_select_AB_correct :
  prob_select_AB = (1 : ℚ) / (num_students.choose num_selected) :=
by sorry

end NUMINAMATH_CALUDE_prob_select_AB_correct_l994_99466


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l994_99433

theorem cylinder_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 1) :
  2 * Real.pi * r * (r + h) = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l994_99433


namespace NUMINAMATH_CALUDE_vectors_are_orthogonal_l994_99475

def vector1 : Fin 4 → ℝ := ![2, -4, 3, 1]
def vector2 : Fin 4 → ℝ := ![-3, 1, 4, -2]

theorem vectors_are_orthogonal :
  (Finset.sum Finset.univ (λ i => vector1 i * vector2 i)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_orthogonal_l994_99475


namespace NUMINAMATH_CALUDE_only_possible_knight_count_l994_99447

/-- Represents a person on the island -/
inductive Person
| Knight
| Liar

/-- The total number of people on the island -/
def total_people : Nat := 2021

/-- A function that determines if a person's claim is true given their position and type -/
def claim_is_true (position : Nat) (person_type : Person) (num_knights : Nat) : Prop :=
  match person_type with
  | Person.Knight => total_people - position - (total_people - num_knights) > position - num_knights
  | Person.Liar => total_people - position - (total_people - num_knights) ≤ position - num_knights

/-- The main theorem stating that the only possible number of knights is 1010 -/
theorem only_possible_knight_count :
  ∃! num_knights : Nat,
    num_knights ≤ total_people ∧
    ∀ position : Nat, position < total_people →
      (claim_is_true position Person.Knight num_knights ∧ position < num_knights) ∨
      (claim_is_true position Person.Liar num_knights ∧ position ≥ num_knights) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_only_possible_knight_count_l994_99447


namespace NUMINAMATH_CALUDE_power_sum_equality_l994_99423

theorem power_sum_equality : (-1)^51 + 3^(2^3 + 5^2 - 7^2) = -1 + 1 / 43046721 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l994_99423


namespace NUMINAMATH_CALUDE_van_speed_problem_l994_99446

theorem van_speed_problem (distance : ℝ) (original_time : ℝ) (time_factor : ℝ) 
  (h1 : distance = 600)
  (h2 : original_time = 5)
  (h3 : time_factor = 3 / 2) :
  distance / (original_time * time_factor) = 80 := by
sorry

end NUMINAMATH_CALUDE_van_speed_problem_l994_99446


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l994_99452

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)^2 - 3*(2*x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ = -1/2 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l994_99452


namespace NUMINAMATH_CALUDE_symmetric_function_value_l994_99437

/-- A function with a graph symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOrigin f)
  (h_pos : ∀ x > 0, f x = 2^x - 3) : 
  f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_value_l994_99437


namespace NUMINAMATH_CALUDE_last_remaining_number_l994_99420

/-- Represents the state of a number in Melanie's list -/
inductive NumberState
  | Unmarked
  | Marked
  | Eliminated

/-- Represents a round in Melanie's process -/
structure Round where
  skipCount : Nat
  startNumber : Nat

/-- The list of numbers Melanie works with -/
def initialList : List Nat := List.range 50

/-- Applies the marking and skipping pattern for a single round -/
def applyRound (list : List (Nat × NumberState)) (round : Round) : List (Nat × NumberState) :=
  sorry

/-- Applies all rounds until only one number remains unmarked -/
def applyAllRounds (list : List (Nat × NumberState)) : Nat :=
  sorry

/-- The main theorem stating that the last remaining number is 47 -/
theorem last_remaining_number :
  applyAllRounds (initialList.map (λ n => (n + 1, NumberState.Unmarked))) = 47 :=
sorry

end NUMINAMATH_CALUDE_last_remaining_number_l994_99420


namespace NUMINAMATH_CALUDE_exists_max_k_l994_99414

theorem exists_max_k (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^3 * (x^2/y^2 + y^2/x^2) + k^2 * (x/y + y/x)) :
  ∃ k_max : ℝ, k ≤ k_max ∧
    ∀ k' : ℝ, k' > 0 → 
      (∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 
        5 = k'^3 * (x'^2/y'^2 + y'^2/x'^2) + k'^2 * (x'/y' + y'/x')) →
      k' ≤ k_max :=
sorry

end NUMINAMATH_CALUDE_exists_max_k_l994_99414


namespace NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l994_99489

def repetend (n d : ℕ) : List ℕ :=
  sorry

theorem repetend_of_four_seventeenths :
  repetend 4 17 = [2, 3, 5, 2, 9, 4] :=
sorry

end NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l994_99489


namespace NUMINAMATH_CALUDE_marks_remaining_money_l994_99462

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount : ℕ) (num_books : ℕ) (price_per_book : ℕ) : ℕ :=
  initial_amount - (num_books * price_per_book)

/-- Proves that Mark is left with $35 after his purchase --/
theorem marks_remaining_money :
  remaining_money 85 10 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_marks_remaining_money_l994_99462


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l994_99464

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x - a) / x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x / x

theorem monotonicity_and_range (a : ℝ) :
  (a > -1/2 →
    (∀ x₁ x₂, x₁ > Real.exp (a + 1/2) ∧ x₂ > Real.exp (a + 1/2) ∧ x₁ < x₂ → g a x₁ > g a x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (a + 1/2) → g a x₁ < g a x₂)) ∧
  (a ≥ 0 →
    (∀ x, x > 0 → x^2 * f a x + a ≥ 2 - Real.exp 1) ↔ 0 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l994_99464


namespace NUMINAMATH_CALUDE_problem_statement_l994_99476

theorem problem_statement (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, x₁^2 ≥ (1/2)^x₂ - m) → 
  m ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l994_99476


namespace NUMINAMATH_CALUDE_complement_of_A_l994_99493

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 6}

theorem complement_of_A : (Aᶜ : Set ℕ) = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l994_99493


namespace NUMINAMATH_CALUDE_x_equals_zero_l994_99454

theorem x_equals_zero (a : ℝ) (x : ℝ) 
  (h1 : a > 0) 
  (h2 : (10 : ℝ) ^ x = Real.log (10 * a) + Real.log (a⁻¹)) : 
  x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_zero_l994_99454


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l994_99406

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : ∀ x, f a b c (x + 1) - f a b c x = 2 * x)
  (h2 : f a b c 0 = 1) :
  (∀ x, f a b c x = x^2 - x + 1) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a b c x > 2 * x + m) ↔ m ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l994_99406


namespace NUMINAMATH_CALUDE_equal_weight_lifts_l994_99484

/-- The number of times Max lifts the original weights -/
def original_lifts : ℕ := 10

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 30

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 25

/-- The number of weights Max lifts each time -/
def num_weights : ℕ := 2

/-- The total weight lifted with the original setup in pounds -/
def total_original_weight : ℕ := num_weights * original_weight * original_lifts

/-- The function to calculate the total weight lifted with the new setup -/
def total_new_weight (n : ℕ) : ℕ := num_weights * new_weight * n

theorem equal_weight_lifts :
  ∃ n : ℕ, total_new_weight n = total_original_weight ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_equal_weight_lifts_l994_99484


namespace NUMINAMATH_CALUDE_class_average_theorem_l994_99415

/-- Given a class with three groups of students, where:
    1. 25% of the class averages 80% on a test
    2. 50% of the class averages 65% on the test
    3. The remainder of the class averages 90% on the test
    Prove that the overall class average is 75% -/
theorem class_average_theorem (group1_proportion : Real) (group1_average : Real)
                              (group2_proportion : Real) (group2_average : Real)
                              (group3_proportion : Real) (group3_average : Real) :
  group1_proportion = 0.25 →
  group1_average = 0.80 →
  group2_proportion = 0.50 →
  group2_average = 0.65 →
  group3_proportion = 0.25 →
  group3_average = 0.90 →
  group1_proportion + group2_proportion + group3_proportion = 1 →
  group1_proportion * group1_average +
  group2_proportion * group2_average +
  group3_proportion * group3_average = 0.75 := by
  sorry


end NUMINAMATH_CALUDE_class_average_theorem_l994_99415


namespace NUMINAMATH_CALUDE_sum_inequality_l994_99473

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - 2*a) / (a^2 + b*c) + 
  (c + a - 2*b) / (b^2 + c*a) + 
  (a + b - 2*c) / (c^2 + a*b) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l994_99473


namespace NUMINAMATH_CALUDE_triangle_theorem_l994_99431

/-- Given a triangle ABC with side lengths a, b, and c, and angles A, B, and C opposite to these sides respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle with given conditions -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a = 2 * Real.sqrt 3)
  (h2 : t.b + t.c = 4) :
  t.A = 2 * π / 3 ∧ 
  t.b * t.c = 4 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l994_99431


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l994_99421

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 4

-- State the theorem
theorem f_decreasing_interval :
  (∀ x : ℝ, f (-x) = f x) →  -- f is even
  (∃ a : ℝ, ∀ x y : ℝ, a ≤ x ∧ x < y → f y < f x) ∧
  (∀ a : ℝ, (∀ x y : ℝ, a ≤ x ∧ x < y → f y < f x) → a ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l994_99421


namespace NUMINAMATH_CALUDE_find_A_l994_99409

theorem find_A : ∃ A : ℝ, (∃ B : ℝ, 211.5 = B - A ∧ B = 10 * A) → A = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l994_99409


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l994_99412

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x

theorem f_derivative_at_one :
  deriv f 1 = 2 * Real.log 2 + 1 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l994_99412


namespace NUMINAMATH_CALUDE_right_triangle_sets_l994_99438

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬ is_right_triangle 1.1 1.5 1.9 ∧
  ¬ is_right_triangle 5 11 12 ∧
  is_right_triangle 1.2 1.6 2.0 ∧
  ¬ is_right_triangle 3 4 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l994_99438


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_l994_99440

/-- Given a sphere with surface area 256π cm², prove that the volume of a cylinder
    with the same radius as the sphere and height equal to the sphere's diameter
    is 1024π cm³. -/
theorem sphere_cylinder_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 256 * Real.pi) :
  Real.pi * r^2 * (2 * r) = 1024 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_l994_99440


namespace NUMINAMATH_CALUDE_min_sum_at_six_l994_99492

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- Sum of first n terms
  h1 : a 1 + a 5 = -14  -- Given condition
  h2 : S 9 = -27  -- Given condition
  h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Arithmetic sequence property

/-- The theorem stating that S_n is minimized when n = 6 -/
theorem min_sum_at_six (seq : ArithmeticSequence) : 
  ∃ (n : ℕ), ∀ (m : ℕ), seq.S n ≤ seq.S m ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_six_l994_99492


namespace NUMINAMATH_CALUDE_problem_solution_l994_99434

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 24 →
  x^2 + Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 10525 / 144 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l994_99434


namespace NUMINAMATH_CALUDE_finish_books_in_two_weeks_l994_99491

/-- The number of weeks needed to finish two books given their page counts and daily reading rate -/
def weeks_to_finish (book1_pages book2_pages daily_pages : ℕ) : ℚ :=
  (book1_pages + book2_pages : ℚ) / (daily_pages * 7 : ℚ)

/-- Theorem: It takes 2 weeks to finish two books with 180 and 100 pages when reading 20 pages per day -/
theorem finish_books_in_two_weeks :
  weeks_to_finish 180 100 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_finish_books_in_two_weeks_l994_99491


namespace NUMINAMATH_CALUDE_laura_circle_arrangements_l994_99477

def numbers : List ℕ := [2, 3, 5, 6, 11]

def is_valid_arrangement (num1 num2 num3 denom : ℕ) : Prop :=
  num1 ∈ numbers ∧ num2 ∈ numbers ∧ num3 ∈ numbers ∧ denom ∈ numbers ∧
  num1 ≠ num2 ∧ num1 ≠ num3 ∧ num1 ≠ denom ∧
  num2 ≠ num3 ∧ num2 ≠ denom ∧
  num3 ≠ denom ∧
  (num1 + num2 + num3) % denom = 0

theorem laura_circle_arrangements :
  ∃! (arrangements : List (ℕ × ℕ × ℕ × ℕ)),
    (∀ arr ∈ arrangements, is_valid_arrangement arr.1 arr.2.1 arr.2.2.1 arr.2.2.2) ∧
    arrangements.length = 4 :=
by sorry

end NUMINAMATH_CALUDE_laura_circle_arrangements_l994_99477


namespace NUMINAMATH_CALUDE_fair_distributions_is_square_l994_99458

/-- The number of permutations of 2n elements with all cycles of even length -/
def fair_distributions (n : ℕ) : ℕ := sorry

/-- The double factorial function -/
def double_factorial (n : ℕ) : ℕ := sorry

theorem fair_distributions_is_square (n : ℕ) : 
  fair_distributions n = (double_factorial (2 * n - 1))^2 := by sorry

end NUMINAMATH_CALUDE_fair_distributions_is_square_l994_99458


namespace NUMINAMATH_CALUDE_sqrt_prime_irrational_l994_99441

theorem sqrt_prime_irrational (p : ℕ) (h : Prime p) : Irrational (Real.sqrt p) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_prime_irrational_l994_99441


namespace NUMINAMATH_CALUDE_proposition_and_related_l994_99478

theorem proposition_and_related (a b : ℝ) : 
  (a + b = 1 → a * b ≤ 1/4) ∧ 
  (a * b > 1/4 → a + b ≠ 1) ∧ 
  ¬(a * b ≤ 1/4 → a + b = 1) ∧ 
  ¬(a + b ≠ 1 → a * b > 1/4) := by
sorry

end NUMINAMATH_CALUDE_proposition_and_related_l994_99478


namespace NUMINAMATH_CALUDE_square_points_sum_l994_99467

/-- A square with side length 900 and two points on one of its sides. -/
structure SquareWithPoints where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The angle EOF in degrees -/
  angle_EOF : ℝ
  /-- The length of EF -/
  EF_length : ℝ
  /-- The distance BF expressed as p + q√r -/
  BF_distance : ℝ → ℝ → ℝ → ℝ
  /-- Condition that side_length is 900 -/
  h_side_length : side_length = 900
  /-- Condition that angle EOF is 45° -/
  h_angle_EOF : angle_EOF = 45
  /-- Condition that EF length is 400 -/
  h_EF_length : EF_length = 400
  /-- Condition that BF = p + q√r -/
  h_BF_distance : ∀ p q r, BF_distance p q r = p + q * Real.sqrt r

/-- The theorem stating that p + q + r = 307 for the given conditions -/
theorem square_points_sum (s : SquareWithPoints) (p q r : ℕ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0)
  (h_prime : ∀ (k : ℕ), k > 1 → k ^ 2 ∣ r → k.Prime → False) :
  p + q + r = 307 := by
  sorry

end NUMINAMATH_CALUDE_square_points_sum_l994_99467


namespace NUMINAMATH_CALUDE_milk_production_increase_l994_99463

/-- Given the initial milk production rate and an increase in production rate,
    calculate the new amount of milk produced by double the cows in triple the time. -/
theorem milk_production_increase (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let initial_rate := y / (x * z)
  let increased_rate := initial_rate * 1.1
  let new_production := 2 * x * increased_rate * 3 * z
  new_production = 6.6 * y := by sorry

end NUMINAMATH_CALUDE_milk_production_increase_l994_99463


namespace NUMINAMATH_CALUDE_quadratic_inequality_l994_99451

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x + 2 < 1 ↔ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l994_99451


namespace NUMINAMATH_CALUDE_only_one_always_true_l994_99442

theorem only_one_always_true (a b c : ℝ) : 
  (∃! p : Prop, p = true) ∧ 
  (((a > b → a * c > b * c) = p) ∨
   ((a > b → a^2 * c^2 > b^2 * c^2) = p) ∨
   ((a^2 * c^2 > b^2 * c^2 → a > b) = p)) :=
by sorry

end NUMINAMATH_CALUDE_only_one_always_true_l994_99442


namespace NUMINAMATH_CALUDE_unique_integral_solution_l994_99465

theorem unique_integral_solution :
  ∃! x : ℤ, x - 9 / (x - 2) = 5 - 9 / (x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_integral_solution_l994_99465


namespace NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l994_99449

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then -1 / x else x^2

theorem sum_f_two_and_neg_two : f 2 + f (-2) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_two_and_neg_two_l994_99449


namespace NUMINAMATH_CALUDE_rhombus_area_l994_99405

/-- The area of a rhombus with side length 3 cm and one angle measuring 45° is (9√2)/2 square cm. -/
theorem rhombus_area (side : ℝ) (angle : ℝ) :
  side = 3 →
  angle = π / 4 →
  let height : ℝ := side / Real.sqrt 2
  let area : ℝ := side * height
  area = (9 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l994_99405


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l994_99496

theorem perfect_square_trinomial_condition (a : ℝ) :
  (∃ b c : ℝ, ∀ x y : ℝ, 4*x^2 - (a-1)*x*y + 9*y^2 = (b*x + c*y)^2) →
  (a = 13 ∨ a = -11) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l994_99496


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_only_four_satisfies_l994_99481

theorem sqrt_x_minus_one_meaningful (x : ℝ) : x - 1 ≥ 0 ↔ x ≥ 1 := by sorry

theorem only_four_satisfies :
  (4 - 1 ≥ 0) ∧ 
  ¬(-4 - 1 ≥ 0) ∧ 
  ¬(-1 - 1 ≥ 0) ∧ 
  ¬(0 - 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_only_four_satisfies_l994_99481


namespace NUMINAMATH_CALUDE_custom_mult_four_three_l994_99430

-- Define the custom multiplication operation
def custom_mult (a b : ℝ) : ℝ := a^2 + a*b - b^2

-- State the theorem
theorem custom_mult_four_three : custom_mult 4 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_four_three_l994_99430


namespace NUMINAMATH_CALUDE_translation_of_segment_l994_99487

/-- Translation of a point in 2D space -/
def translate (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 + q.1, p.2 + q.2)

theorem translation_of_segment (A B C : ℝ × ℝ) :
  A = (-2, 5) →
  B = (-3, 0) →
  C = (3, 7) →
  translate A (5, 2) = C →
  translate B (5, 2) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_segment_l994_99487


namespace NUMINAMATH_CALUDE_tenth_digit_theorem_l994_99401

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def some_number : ℕ := 6840

theorem tenth_digit_theorem :
  (((factorial 5 * factorial 5 - factorial 5 * factorial 3) / some_number) % 100) / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tenth_digit_theorem_l994_99401


namespace NUMINAMATH_CALUDE_least_years_to_double_l994_99436

def interest_rate : ℝ := 0.13

def more_than_doubled (years : ℕ) : Prop :=
  (1 + interest_rate) ^ years > 2

theorem least_years_to_double :
  (∀ y : ℕ, y < 6 → ¬(more_than_doubled y)) ∧ 
  more_than_doubled 6 :=
sorry

end NUMINAMATH_CALUDE_least_years_to_double_l994_99436


namespace NUMINAMATH_CALUDE_quadratic_point_on_graph_l994_99488

/-- Given a quadratic function y = -ax² + 2ax + 3 where a > 0,
    if the point (m, 3) lies on the graph and m ≠ 0, then m = 2 -/
theorem quadratic_point_on_graph (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_on_graph_l994_99488


namespace NUMINAMATH_CALUDE_game_ends_in_22_rounds_l994_99402

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : Vector Player 3
  round : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that the game ends after exactly 22 rounds -/
theorem game_ends_in_22_rounds :
  let initialState : GameState := {
    players := ⟨[{tokens := 16}, {tokens := 15}, {tokens := 14}], rfl⟩,
    round := 0
  }
  ∃ (finalState : GameState),
    finalState.round = 22 ∧
    gameEnded finalState ∧
    (∀ (intermediateState : GameState),
      intermediateState.round < 22 →
      ¬gameEnded intermediateState) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_in_22_rounds_l994_99402


namespace NUMINAMATH_CALUDE_no_real_roots_l994_99445

theorem no_real_roots (a b : ℝ) : ¬ ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l994_99445


namespace NUMINAMATH_CALUDE_seventh_twenty_ninth_712th_digit_l994_99411

def decimal_representation (n d : ℕ) : List ℕ :=
  sorry

theorem seventh_twenty_ninth_712th_digit :
  let repr := decimal_representation 7 29
  let cycle_length := 29
  let digit_position := 712 % cycle_length
  List.get! repr digit_position = 1 := by
  sorry

end NUMINAMATH_CALUDE_seventh_twenty_ninth_712th_digit_l994_99411


namespace NUMINAMATH_CALUDE_homework_problem_count_l994_99461

theorem homework_problem_count (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : 
  math_pages = 2 → reading_pages = 4 → problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l994_99461


namespace NUMINAMATH_CALUDE_roll_two_dice_prob_at_least_one_two_l994_99471

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling at least one 2 with two fair 8-sided dice -/
def prob_at_least_one_two : ℚ := 15 / 64

/-- Theorem stating the probability of rolling at least one 2 with two fair 8-sided dice -/
theorem roll_two_dice_prob_at_least_one_two :
  prob_at_least_one_two = 15 / 64 := by
  sorry

#check roll_two_dice_prob_at_least_one_two

end NUMINAMATH_CALUDE_roll_two_dice_prob_at_least_one_two_l994_99471


namespace NUMINAMATH_CALUDE_expand_expression_l994_99439

theorem expand_expression (x : ℝ) : (x + 2) * (3 * x - 6) = 3 * x^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l994_99439


namespace NUMINAMATH_CALUDE_divide_42_problem_l994_99410

theorem divide_42_problem (x : ℚ) (h : 35 / x = 5) : 42 / x = 6 := by
  sorry

end NUMINAMATH_CALUDE_divide_42_problem_l994_99410


namespace NUMINAMATH_CALUDE_largest_x_value_l994_99426

theorem largest_x_value (x : ℝ) :
  (x / 7 + 3 / (7 * x) = 2 / 3) →
  x ≤ (7 + Real.sqrt 22) / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_l994_99426


namespace NUMINAMATH_CALUDE_august_tips_multiple_l994_99413

/-- 
Proves that if a worker's tips for one month (August) are 0.625 of their total tips for 7 months, 
and the August tips are some multiple of the average tips for the other 6 months, then this multiple is 10.
-/
theorem august_tips_multiple (total_months : ℕ) (other_months : ℕ) (august_ratio : ℝ) (M : ℝ) : 
  total_months = 7 → 
  other_months = 6 → 
  august_ratio = 0.625 →
  M * (1 / other_months : ℝ) * (1 - august_ratio) * total_months = august_ratio →
  M = 10 := by
  sorry

end NUMINAMATH_CALUDE_august_tips_multiple_l994_99413


namespace NUMINAMATH_CALUDE_david_chemistry_marks_l994_99429

/-- Calculates the marks in Chemistry given marks in other subjects and the average -/
def chemistry_marks (english math physics biology : ℕ) (average : ℚ) : ℚ :=
  5 * average - (english + math + physics + biology)

/-- Theorem stating that David's Chemistry marks are 67 given his other marks and average -/
theorem david_chemistry_marks :
  let english : ℕ := 76
  let math : ℕ := 65
  let physics : ℕ := 82
  let biology : ℕ := 85
  let average : ℚ := 75
  chemistry_marks english math physics biology average = 67 := by
sorry

#eval chemistry_marks 76 65 82 85 75

end NUMINAMATH_CALUDE_david_chemistry_marks_l994_99429


namespace NUMINAMATH_CALUDE_points_on_line_l994_99417

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- The theorem states that if the given points are collinear, then a = -7/23 -/
theorem points_on_line (a : ℝ) : 
  collinear (3, -5) (-a + 2, 3) (2*a + 3, 2) → a = -7/23 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l994_99417


namespace NUMINAMATH_CALUDE_sale_price_equals_original_l994_99407

theorem sale_price_equals_original (x : ℝ) : x > 0 → 0.8 * (1.25 * x) = x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_equals_original_l994_99407


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l994_99497

theorem sum_of_fractions_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l994_99497
