import Mathlib

namespace parallelogram_area_bound_l3685_368511

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A parallelogram -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- The center of a regular hexagon -/
def center (h : RegularHexagon) : ℝ × ℝ := sorry

/-- The center of symmetry of a parallelogram -/
def center_of_symmetry (p : Parallelogram) : ℝ × ℝ := sorry

/-- Area of a regular hexagon -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- Area of a parallelogram -/
def area_parallelogram (p : Parallelogram) : ℝ := sorry

/-- A parallelogram is inscribed in a regular hexagon if all its vertices
    are on the perimeter of the hexagon -/
def inscribed (p : Parallelogram) (h : RegularHexagon) : Prop := sorry

theorem parallelogram_area_bound (h : RegularHexagon) (p : Parallelogram) 
  (h_inscribed : inscribed p h) 
  (h_center : center_of_symmetry p = center h) : 
  area_parallelogram p ≤ (2/3) * area_hexagon h := sorry

end parallelogram_area_bound_l3685_368511


namespace geometric_sequence_product_l3685_368504

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = 2 * a n

theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h : geometric_sequence a) : a 3 * a 5 = 64 := by
  sorry

end geometric_sequence_product_l3685_368504


namespace adjacent_cells_difference_l3685_368500

/-- A type representing a cell in an n × n grid --/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- A function representing the placement of integers in the grid --/
def GridPlacement (n : ℕ) := Cell n → Fin (n^2)

/-- Two cells are adjacent if they share a side or a corner --/
def adjacent {n : ℕ} (c1 c2 : Cell n) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c2.col.val + 1 = c1.col.val) ∨
  (c1.col = c2.col ∧ c1.row.val + 1 = c2.row.val) ∨
  (c1.col = c2.col ∧ c2.row.val + 1 = c1.row.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c2.col.val + 1 = c1.col.val) ∨
  (c2.row.val + 1 = c1.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c2.row.val + 1 = c1.row.val ∧ c2.col.val + 1 = c1.col.val)

/-- The main theorem --/
theorem adjacent_cells_difference {n : ℕ} (h : n > 0) (g : GridPlacement n) :
  ∃ (c1 c2 : Cell n), adjacent c1 c2 ∧ 
    ((g c1).val + n + 1 ≤ (g c2).val ∨ (g c2).val + n + 1 ≤ (g c1).val) :=
sorry

end adjacent_cells_difference_l3685_368500


namespace highest_power_of_five_in_M_highest_power_is_one_l3685_368509

def M : ℕ := sorry

theorem highest_power_of_five_in_M : 
  ∃ (j : ℕ), (5^j ∣ M) ∧ ∀ (k : ℕ), k > j → ¬(5^k ∣ M) :=
by
  -- The proof would go here
  sorry

theorem highest_power_is_one : 
  ∃ (j : ℕ), j = 1 ∧ (5^j ∣ M) ∧ ∀ (k : ℕ), k > j → ¬(5^k ∣ M) :=
by
  -- The proof would go here
  sorry

end highest_power_of_five_in_M_highest_power_is_one_l3685_368509


namespace xyz_problem_l3685_368568

theorem xyz_problem (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : x * y * z = 10)
  (h5 : (x ^ (Real.log x)) * (y ^ (Real.log y)) * (z ^ (Real.log z)) = 10) :
  ((x = 1 ∧ y = 1 ∧ z = 10) ∨ 
   (x = 10 ∧ y = 1 ∧ z = 1) ∨ 
   (x = 1 ∧ y = 10 ∧ z = 1)) :=
by sorry

end xyz_problem_l3685_368568


namespace chord_length_theorem_l3685_368547

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if two circles are internally tangent -/
def internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (c1 c2 c3 : Circle) 
  (h1 : c1.radius = 6)
  (h2 : c2.radius = 8)
  (h3 : externally_tangent c1 c3)
  (h4 : internally_tangent c2 c3)
  (h5 : collinear c1.center c2.center c3.center) :
  ∃ (chord_length : ℝ), chord_length = 8 * Real.sqrt (2 * c3.radius - 8) :=
by sorry

end chord_length_theorem_l3685_368547


namespace evaluate_expression_l3685_368597

theorem evaluate_expression : (9 ^ 9) * (3 ^ 3) / (3 ^ 30) = 1 / 19683 := by
  sorry

end evaluate_expression_l3685_368597


namespace square_of_quadratic_condition_l3685_368596

/-- 
If a polynomial x^4 + ax^3 + bx^2 + cx + d is the square of a quadratic polynomial,
then ac^2 - 4abd + 8cd = 0.
-/
theorem square_of_quadratic_condition (a b c d : ℝ) : 
  (∃ p q : ℝ, ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + d = (x^2 + p*x + q)^2) →
  a*c^2 - 4*a*b*d + 8*c*d = 0 := by
  sorry

end square_of_quadratic_condition_l3685_368596


namespace apple_baskets_l3685_368532

theorem apple_baskets (apples_per_basket : ℕ) (total_apples : ℕ) (h1 : apples_per_basket = 17) (h2 : total_apples = 629) :
  total_apples / apples_per_basket = 37 := by
  sorry

end apple_baskets_l3685_368532


namespace smallest_cube_side_length_is_four_l3685_368501

/-- A cube that can contain two non-overlapping spheres of radius 1 -/
structure Cube :=
  (side_length : ℝ)
  (contains_spheres : side_length ≥ 4)

/-- The smallest side length of a cube that can contain two non-overlapping spheres of radius 1 -/
def smallest_cube_side_length : ℝ := 4

/-- Theorem: The smallest side length of a cube that can contain two non-overlapping spheres of radius 1 is 4 -/
theorem smallest_cube_side_length_is_four :
  ∀ (c : Cube), c.side_length ≥ smallest_cube_side_length :=
sorry

end smallest_cube_side_length_is_four_l3685_368501


namespace van_rental_equation_l3685_368537

theorem van_rental_equation (x : ℝ) (h1 : x > 2) : 
  (180 / (x - 2)) - (180 / x) = 3 :=
by sorry

end van_rental_equation_l3685_368537


namespace max_cube_in_tetrahedron_l3685_368531

/-- The maximum edge length of a cube that can rotate freely inside a regular tetrahedron -/
def max_cube_edge_length (tetrahedron_edge : ℝ) : ℝ :=
  2

/-- Theorem: The maximum edge length of a cube that can rotate freely inside a regular tetrahedron
    with edge length 6√2 is equal to 2 -/
theorem max_cube_in_tetrahedron :
  max_cube_edge_length (6 * Real.sqrt 2) = 2 := by
  sorry

end max_cube_in_tetrahedron_l3685_368531


namespace test_maximum_marks_l3685_368585

theorem test_maximum_marks (passing_threshold : Real) (student_score : Nat) (failing_margin : Nat) 
  (h1 : passing_threshold = 0.60)
  (h2 : student_score = 80)
  (h3 : failing_margin = 40) :
  (student_score + failing_margin) / passing_threshold = 200 := by
sorry

end test_maximum_marks_l3685_368585


namespace bus_tour_tickets_sold_l3685_368538

/-- A bus tour selling tickets to senior citizens and regular passengers -/
structure BusTour where
  senior_price : ℕ
  regular_price : ℕ
  total_sales : ℕ
  senior_tickets : ℕ

/-- The total number of tickets sold in a bus tour -/
def total_tickets (tour : BusTour) : ℕ :=
  tour.senior_tickets + (tour.total_sales - tour.senior_tickets * tour.senior_price) / tour.regular_price

/-- Theorem stating that for the given conditions, the total number of tickets sold is 65 -/
theorem bus_tour_tickets_sold (tour : BusTour)
  (h1 : tour.senior_price = 10)
  (h2 : tour.regular_price = 15)
  (h3 : tour.total_sales = 855)
  (h4 : tour.senior_tickets = 24) :
  total_tickets tour = 65 := by
  sorry

end bus_tour_tickets_sold_l3685_368538


namespace root_equation_value_l3685_368545

theorem root_equation_value (a : ℝ) (h : a^2 + 3*a - 5 = 0) :
  a^2 + 3*a + 2021 = 2026 := by
sorry

end root_equation_value_l3685_368545


namespace prime_square_minus_one_divisible_by_forty_l3685_368594

theorem prime_square_minus_one_divisible_by_forty (p : ℕ) 
  (h_prime : Nat.Prime p) (h_geq_seven : p ≥ 7) : 
  40 ∣ p^2 - 1 := by
  sorry

end prime_square_minus_one_divisible_by_forty_l3685_368594


namespace photo_arrangement_count_total_arrangement_count_l3685_368569

/-- The number of ways to arrange 6 people with specific conditions -/
def arrangement_count : ℕ := 480

/-- The number of people in the group -/
def total_people : ℕ := 6

/-- The number of people with specific positions (A, B, and C) -/
def specific_people : ℕ := 3

/-- The number of ways A and B can be arranged on one side of C -/
def ab_arrangements : ℕ := 4

/-- The number of ways to arrange the remaining people -/
def remaining_arrangements : ℕ := 120

theorem photo_arrangement_count :
  arrangement_count = ab_arrangements * remaining_arrangements :=
sorry

theorem total_arrangement_count :
  arrangement_count = 480 :=
sorry

end photo_arrangement_count_total_arrangement_count_l3685_368569


namespace trivia_team_points_l3685_368577

/-- Given a trivia team with the following properties:
  * total_members: The total number of team members
  * absent_members: The number of members who didn't show up
  * points_per_member: The number of points scored by each member who showed up
  * total_points: The total points scored by the team

  This theorem proves that the total points scored by the team is equal to
  the product of the number of members who showed up and the points per member.
-/
theorem trivia_team_points 
  (total_members : ℕ) 
  (absent_members : ℕ) 
  (points_per_member : ℕ) 
  (total_points : ℕ) 
  (h1 : total_members = 14) 
  (h2 : absent_members = 7) :
  total_points = (total_members - absent_members) * points_per_member := by
sorry

end trivia_team_points_l3685_368577


namespace hyperbola_eccentricity_l3685_368595

/-- Hyperbola M with equation x^2 - y^2/b^2 = 1 -/
def hyperbola_M (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 1

/-- Line l with slope 1 passing through the left vertex (-1, 0) -/
def line_l (x y : ℝ) : Prop :=
  y = x + 1

/-- Asymptotes of hyperbola M -/
def asymptotes_M (b : ℝ) (x y : ℝ) : Prop :=
  x^2 - y^2/b^2 = 0

/-- Point A is the left vertex of hyperbola M -/
def point_A : ℝ × ℝ :=
  (-1, 0)

/-- Point B is the intersection of line l and one asymptote -/
def point_B (b : ℝ) : ℝ × ℝ :=
  sorry

/-- Point C is the intersection of line l and the other asymptote -/
def point_C (b : ℝ) : ℝ × ℝ :=
  sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The eccentricity of hyperbola M -/
def eccentricity (b : ℝ) : ℝ :=
  sorry

theorem hyperbola_eccentricity (b : ℝ) :
  hyperbola_M b (point_A.1) (point_A.2) →
  line_l (point_B b).1 (point_B b).2 →
  line_l (point_C b).1 (point_C b).2 →
  asymptotes_M b (point_B b).1 (point_B b).2 →
  asymptotes_M b (point_C b).1 (point_C b).2 →
  distance point_A (point_B b) = distance (point_B b) (point_C b) →
  eccentricity b = Real.sqrt 10 :=
sorry

end hyperbola_eccentricity_l3685_368595


namespace rational_square_root_condition_l3685_368549

theorem rational_square_root_condition (n : ℤ) :
  n ≥ 3 →
  (∃ (q : ℚ), q^2 = (n^2 - 5) / (n + 1)) ↔ n = 3 :=
by sorry

end rational_square_root_condition_l3685_368549


namespace johns_drive_time_l3685_368524

theorem johns_drive_time (speed : ℝ) (total_distance : ℝ) (after_lunch_time : ℝ) 
  (h : speed = 55)
  (h' : total_distance = 275)
  (h'' : after_lunch_time = 3) :
  let before_lunch_time := (total_distance - speed * after_lunch_time) / speed
  before_lunch_time = 2 := by sorry

end johns_drive_time_l3685_368524


namespace apple_pile_count_l3685_368560

theorem apple_pile_count (initial_apples added_apples : ℕ) 
  (h1 : initial_apples = 8)
  (h2 : added_apples = 5) : 
  initial_apples + added_apples = 13 := by
  sorry

end apple_pile_count_l3685_368560


namespace max_product_sum_300_l3685_368535

theorem max_product_sum_300 (a b : ℤ) (h : a + b = 300) :
  a * b ≤ 22500 := by
  sorry

end max_product_sum_300_l3685_368535


namespace exactly_two_out_of_three_l3685_368553

def probability_single_shot : ℚ := 2/3

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exactly_two_out_of_three :
  binomial_probability 3 2 probability_single_shot = 4/9 := by
  sorry

end exactly_two_out_of_three_l3685_368553


namespace binary_multiplication_theorem_l3685_368510

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order --/
def BinaryNumber := List Nat

/-- Converts a binary number to its decimal representation --/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

/-- Converts a decimal number to its binary representation --/
def decimal_to_binary (n : Nat) : BinaryNumber :=
  if n = 0 then [0] else
  let rec to_binary_helper (m : Nat) (acc : BinaryNumber) : BinaryNumber :=
    if m = 0 then acc
    else to_binary_helper (m / 2) ((m % 2) :: acc)
  to_binary_helper n []

theorem binary_multiplication_theorem :
  let a : BinaryNumber := [1, 0, 1, 1]  -- 1101₂
  let b : BinaryNumber := [1, 1, 1]     -- 111₂
  let result : BinaryNumber := [1, 1, 1, 1, 0, 0, 0, 0, 1]  -- 100001111₂
  binary_to_decimal (decimal_to_binary (binary_to_decimal a * binary_to_decimal b)) = binary_to_decimal result :=
by sorry

end binary_multiplication_theorem_l3685_368510


namespace complete_square_equivalence_l3685_368548

theorem complete_square_equivalence :
  ∀ x y : ℝ, y = -x^2 + 2*x + 3 ↔ y = -(x - 1)^2 + 4 := by sorry

end complete_square_equivalence_l3685_368548


namespace baba_yaga_hut_inhabitants_l3685_368554

/-- The number of inhabitants in Baba Yaga's hut -/
def num_inhabitants : ℕ := 3

/-- The number of Talking Cats -/
def num_cats : ℕ := 1

/-- The number of Wise Owls -/
def num_owls : ℕ := 1

/-- The number of Mustached Cockroaches -/
def num_cockroaches : ℕ := 1

/-- The total number of non-Talking Cats -/
def non_cats : ℕ := 2

/-- The total number of non-Wise Owls -/
def non_owls : ℕ := 2

theorem baba_yaga_hut_inhabitants :
  num_inhabitants = num_cats + num_owls + num_cockroaches ∧
  non_cats = num_owls + num_cockroaches ∧
  non_owls = num_cats + num_cockroaches ∧
  num_inhabitants = 3 := by
  sorry

end baba_yaga_hut_inhabitants_l3685_368554


namespace correct_new_upstream_time_l3685_368580

/-- Represents the boat's journey on a river with varying current conditions -/
structure RiverJourney where
  downstream_time : ℝ  -- Time from A to C downstream
  upstream_time : ℝ    -- Time from C to A upstream
  new_downstream_time : ℝ  -- Time from A to C with uniform current
  boat_speed : ℝ        -- Boat's own speed (constant)

/-- Calculates the upstream time under new conditions -/
def new_upstream_time (journey : RiverJourney) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the correct upstream time under new conditions -/
theorem correct_new_upstream_time (journey : RiverJourney) 
  (h1 : journey.downstream_time = 6)
  (h2 : journey.upstream_time = 7)
  (h3 : journey.new_downstream_time = 5.5)
  (h4 : journey.boat_speed > 0) :
  new_upstream_time journey = 7.7 := by
  sorry

end correct_new_upstream_time_l3685_368580


namespace no_solutions_to_equation_l3685_368578

theorem no_solutions_to_equation : 
  ¬∃ x : ℝ, (9 - x^2 ≥ 0) ∧ (Real.sqrt (9 - x^2) = x * Real.sqrt (9 - x^2) + x) := by
  sorry

end no_solutions_to_equation_l3685_368578


namespace quadratic_other_intercept_l3685_368581

/-- A quadratic function with vertex (2, 9) and one x-intercept at (3, 0) has its other x-intercept at x = 1 -/
theorem quadratic_other_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 9 - a * (x - 2)^2) →  -- vertex form with vertex (2, 9)
  a * 3^2 + b * 3 + c = 0 →                         -- x-intercept at (3, 0)
  a * 1^2 + b * 1 + c = 0 :=                        -- other x-intercept at (1, 0)
by sorry

end quadratic_other_intercept_l3685_368581


namespace lemonade_sales_difference_l3685_368566

/-- Calculates the total difference in lemonade sales between siblings --/
def total_lemonade_sales_difference (
  stanley_cups_per_hour : ℕ)
  (stanley_price_per_cup : ℚ)
  (carl_cups_per_hour : ℕ)
  (carl_price_per_cup : ℚ)
  (lucy_cups_per_hour : ℕ)
  (lucy_price_per_cup : ℚ)
  (hours : ℕ) : ℚ :=
  let stanley_total := (stanley_cups_per_hour * hours : ℚ) * stanley_price_per_cup
  let carl_total := (carl_cups_per_hour * hours : ℚ) * carl_price_per_cup
  let lucy_total := (lucy_cups_per_hour * hours : ℚ) * lucy_price_per_cup
  |carl_total - stanley_total| + |lucy_total - stanley_total| + |carl_total - lucy_total|

theorem lemonade_sales_difference :
  total_lemonade_sales_difference 4 (3/2) 7 (13/10) 5 (9/5) 3 = 93/5 := by
  sorry

end lemonade_sales_difference_l3685_368566


namespace ceiling_equation_solution_l3685_368584

theorem ceiling_equation_solution :
  ∃! x : ℝ, ⌈x⌉ * x + 15 = 210 :=
by
  -- The unique solution is 195/14
  use 195/14
  sorry

end ceiling_equation_solution_l3685_368584


namespace symmetry_condition_l3685_368567

/-- A function f is symmetric about a line x = k if f(k + t) = f(k - t) for all t. -/
def IsSymmetricAbout (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ t, f (k + t) = f (k - t)

/-- The main theorem stating the condition for symmetry of the given function. -/
theorem symmetry_condition (a : ℝ) :
  IsSymmetricAbout (fun x => a * Real.sin x + Real.cos (x + π/6)) (π/3) ↔ a = 2 := by
  sorry

end symmetry_condition_l3685_368567


namespace min_value_trig_expression_l3685_368557

theorem min_value_trig_expression (α β : Real) :
  ∃ (min : Real), 
    (∀ α β : Real, (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ min) ∧ 
    (∃ α β : Real, (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = min) ∧
    min = 36 := by
  sorry

end min_value_trig_expression_l3685_368557


namespace collinear_points_a_value_l3685_368530

-- Define the points
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ × ℝ := λ a => (a, 0)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Theorem statement
theorem collinear_points_a_value :
  collinear A B (C a) → a = 2 := by
  sorry

end collinear_points_a_value_l3685_368530


namespace integer_solutions_inequality_l3685_368574

theorem integer_solutions_inequality (x : ℤ) : 
  -1 < 2*x + 1 ∧ 2*x + 1 < 5 ↔ x = 0 ∨ x = 1 := by
  sorry

end integer_solutions_inequality_l3685_368574


namespace lower_limit_of_range_l3685_368508

theorem lower_limit_of_range (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
   n < p ∧ p < 87/6 ∧ q < 87/6 ∧
   ∀ r, Prime r → r > n → r < 87/6 → (r = p ∨ r = q)) →
  n ≤ 79 :=
sorry

end lower_limit_of_range_l3685_368508


namespace problem_statement_l3685_368543

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 + 9 * x * y = x^3 + 3 * x^2 * y) : x = 3 := by
  sorry

end problem_statement_l3685_368543


namespace defeated_candidate_vote_percentage_l3685_368598

theorem defeated_candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_votes : ℕ)
  (losing_margin : ℕ)
  (h_total : total_votes = 12600)
  (h_invalid : invalid_votes = 100)
  (h_margin : losing_margin = 5000) :
  (((total_votes - invalid_votes : ℚ) / 2 - losing_margin) / (total_votes - invalid_votes)) * 100 = 30 := by
  sorry

end defeated_candidate_vote_percentage_l3685_368598


namespace cylinder_volume_formula_l3685_368516

/-- Given a cylinder and a plane passing through its element, prove the formula for the cylinder's volume. -/
theorem cylinder_volume_formula (l α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2) (h_β_acute : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ, V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by sorry

end cylinder_volume_formula_l3685_368516


namespace sum_floor_value_l3685_368565

theorem sum_floor_value (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2016 ∧ c^2 + d^2 = 2016)
  (products : a * c = 1024 ∧ b * d = 1024) : 
  ⌊a + b + c + d⌋ = 127 := by
sorry

end sum_floor_value_l3685_368565


namespace no_two_digit_multiple_of_3_5_7_l3685_368583

theorem no_two_digit_multiple_of_3_5_7 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ¬(3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := by
  sorry

end no_two_digit_multiple_of_3_5_7_l3685_368583


namespace quadratic_no_real_roots_l3685_368570

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has no real roots -/
def has_no_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq < 0

theorem quadratic_no_real_roots :
  let eq_A : QuadraticEquation := ⟨1, 1, -2⟩
  let eq_B : QuadraticEquation := ⟨1, -2, 0⟩
  let eq_C : QuadraticEquation := ⟨1, 1, 5⟩
  let eq_D : QuadraticEquation := ⟨1, -2, 1⟩
  has_no_real_roots eq_C ∧
  ¬(has_no_real_roots eq_A ∨ has_no_real_roots eq_B ∨ has_no_real_roots eq_D) :=
by sorry

end quadratic_no_real_roots_l3685_368570


namespace mandy_current_pages_l3685_368525

/-- Calculates the number of pages in books Mandy reads at different ages --/
def pages_at_age (initial_pages : ℕ) (initial_age : ℕ) (current_age : ℕ) : ℕ :=
  if current_age = initial_age then
    initial_pages
  else if current_age = 2 * initial_age then
    5 * initial_pages
  else if current_age = 2 * initial_age + 8 then
    3 * 5 * initial_pages
  else
    4 * 3 * 5 * initial_pages

/-- Theorem stating that Mandy now reads books with 480 pages --/
theorem mandy_current_pages : pages_at_age 8 6 (2 * 6 + 8 + 1) = 480 := by
  sorry

end mandy_current_pages_l3685_368525


namespace range_of_a_l3685_368527

/-- The function f(x) = |x^3 + ax + b| -/
def f (a b x : ℝ) : ℝ := |x^3 + a*x + b|

/-- Theorem stating the range of 'a' given the conditions -/
theorem range_of_a (a b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 →
    f a b x₁ - f a b x₂ ≤ 2 * |x₁ - x₂|) →
  a ∈ Set.Icc (-2) (-1) :=
by sorry

end range_of_a_l3685_368527


namespace smallest_n_congruence_l3685_368522

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (697 * n ≡ 1421 * n [ZMOD 36]) ∧ 
  ∀ (m : ℕ), m > 0 → (697 * m ≡ 1421 * m [ZMOD 36]) → n ≤ m := by
  sorry

end smallest_n_congruence_l3685_368522


namespace larry_initial_amount_l3685_368515

def initial_amount (lunch_cost brother_gift current_amount : ℕ) : ℕ :=
  lunch_cost + brother_gift + current_amount

theorem larry_initial_amount :
  initial_amount 5 2 15 = 22 :=
by sorry

end larry_initial_amount_l3685_368515


namespace case_one_case_two_l3685_368599

-- Define the set M
def M := {f : ℤ → ℝ | f 0 ≠ 0 ∧ ∀ n m : ℤ, f n * f m = f (n + m) + f (n - m)}

-- Statement for the first case
theorem case_one (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = 5/2) :
  ∀ n : ℤ, f n = 2^n + 2^(-n) :=
sorry

-- Statement for the second case
theorem case_two (f : ℤ → ℝ) (hf : f ∈ M) (h1 : f 1 = Real.sqrt 3) :
  ∀ n : ℤ, f n = 2 * Real.cos (π * n / 6) :=
sorry

end case_one_case_two_l3685_368599


namespace height_of_taller_tree_l3685_368528

/-- Given two trees with different base elevations, prove that the height of the taller tree is 60 feet. -/
theorem height_of_taller_tree : 
  ∀ (h1 h2 : ℝ), -- heights of the two trees
  h1 > h2 → -- first tree is taller
  h1 - h2 = 20 → -- first tree's top is 20 feet above the second tree's top
  ∃ (b1 b2 : ℝ), -- base elevations of the two trees
    b1 - b2 = 8 ∧ -- base of the first tree is 8 feet higher
    (h1 / (h2 + (b1 - b2))) = 5/4 → -- ratio of heights from respective bases is 4:5
  h1 = 60 := by
sorry

end height_of_taller_tree_l3685_368528


namespace min_combines_to_goal_l3685_368587

/-- Represents the state of rock stacks -/
structure RockStacks :=
  (stacks : List ℕ)

/-- Allowed operations on rock stacks -/
inductive Operation
  | Split (i : ℕ) (a b : ℕ)
  | Combine (i j : ℕ)

/-- Applies an operation to the rock stacks -/
def applyOperation (s : RockStacks) (op : Operation) : RockStacks :=
  sorry

/-- Checks if the goal state is reached -/
def isGoalReached (s : RockStacks) (n : ℕ) : Prop :=
  sorry

/-- Theorem: Minimum number of combines to reach the goal state -/
theorem min_combines_to_goal (n : ℕ) :
  ∃ (ops : List Operation),
    (ops.filter (λ op => match op with
      | Operation.Combine _ _ => true
      | _ => false)).length = 4 ∧
    isGoalReached (ops.foldl applyOperation ⟨[3 * 2^n]⟩) n :=
  sorry

end min_combines_to_goal_l3685_368587


namespace loan_interest_difference_l3685_368562

/-- Proves that for a loan of 2000 at 3% simple interest for 3 years, 
    the difference between the principal and the interest is 1940 -/
theorem loan_interest_difference : 
  let principal : ℚ := 2000
  let rate : ℚ := 3 / 100
  let time : ℚ := 3
  let interest := principal * rate * time
  principal - interest = 1940 := by sorry

end loan_interest_difference_l3685_368562


namespace boy_walking_time_l3685_368517

/-- Given a boy who walks at 4/3 of his usual rate and arrives at school 4 minutes early,
    his usual time to reach school is 16 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) 
    (h2 : usual_time > 0) 
    (h3 : usual_rate * usual_time = (4/3 * usual_rate) * (usual_time - 4)) : 
  usual_time = 16 := by
sorry

end boy_walking_time_l3685_368517


namespace quadratic_root_implies_k_l3685_368556

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 5*x + k = 0 ∧ x = 2) → k = 6 := by
  sorry

end quadratic_root_implies_k_l3685_368556


namespace toucan_count_l3685_368540

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end toucan_count_l3685_368540


namespace max_s_value_l3685_368579

theorem max_s_value (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_products_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_s_value_l3685_368579


namespace horner_method_v3_l3685_368589

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (a b c d e f x : ℝ) : ℝ :=
  ((((a*x + b)*x + c)*x + d)*x + e)*x + f

theorem horner_method_v3 :
  horner_v3 2 0 (-3) 2 1 (-3) 2 = 12 :=
sorry

end horner_method_v3_l3685_368589


namespace probability_three_correct_out_of_seven_l3685_368526

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of derangements of n objects -/
def derangement (n : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The probability of exactly k people getting the right letter when n letters are randomly distributed to n people -/
def probability_correct_letters (n k : ℕ) : ℚ :=
  (choose n k * derangement (n - k)) / factorial n

theorem probability_three_correct_out_of_seven :
  probability_correct_letters 7 3 = 1 / 16 := by sorry

end probability_three_correct_out_of_seven_l3685_368526


namespace ab_value_l3685_368582

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l3685_368582


namespace minimum_implies_a_range_l3685_368571

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- Theorem: If f has a minimum value in the interval (a, 6-a^2), then a ∈ [-2, 1) -/
theorem minimum_implies_a_range (a : ℝ) 
  (h_min : ∃ (x : ℝ), a < x ∧ x < 6 - a^2 ∧ ∀ (y : ℝ), a < y ∧ y < 6 - a^2 → f y ≥ f x) :
  a ≥ -2 ∧ a < 1 := by sorry

end minimum_implies_a_range_l3685_368571


namespace star_not_associative_l3685_368539

/-- Definition of the binary operation ⋆ -/
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

/-- Theorem stating that the binary operation ⋆ is not associative -/
theorem star_not_associative : ¬ ∀ x y z : ℝ, star (star x y) z = star x (star y z) := by
  sorry

end star_not_associative_l3685_368539


namespace mia_chocolate_amount_l3685_368503

/-- 
Given that Liam has 72/7 pounds of chocolate and divides it into 6 equal piles,
this theorem proves that if he gives 2 piles to Mia, she will receive 24/7 pounds of chocolate.
-/
theorem mia_chocolate_amount 
  (total_chocolate : ℚ) 
  (num_piles : ℕ) 
  (piles_to_mia : ℕ) 
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_to_mia = 2) : 
  piles_to_mia * (total_chocolate / num_piles) = 24 / 7 := by
  sorry

end mia_chocolate_amount_l3685_368503


namespace sqrt_equation_solution_l3685_368561

theorem sqrt_equation_solution (a x : ℝ) (ha : 0 < a ∧ a ≤ 1) (hx : x ≥ 1) :
  Real.sqrt (x + Real.sqrt x) - Real.sqrt (x - Real.sqrt x) = (a + 1) * Real.sqrt (x / (x + Real.sqrt x)) →
  x = ((a^2 + 1) / (2*a))^2 := by
sorry

end sqrt_equation_solution_l3685_368561


namespace solutions_of_fourth_power_equation_l3685_368534

theorem solutions_of_fourth_power_equation :
  {x : ℂ | x^4 - 16 = 0} = {2, -2, 2*I, -2*I} := by sorry

end solutions_of_fourth_power_equation_l3685_368534


namespace union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l3685_368586

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∪ B = {x | 2 < x < 10}
theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

-- Theorem 2: (Cᵤ A) ∩ B = {x | 2 < x ≤ 3 or 7 < x < 10}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | (2 < x ∧ x ≤ 3) ∨ (7 < x ∧ x < 10)} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a ≥ 3
theorem intersection_A_C_nonempty (a : ℝ) : A ∩ C a ≠ ∅ → a ≥ 3 := by sorry

end union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l3685_368586


namespace problem_cube_surface_area_l3685_368555

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initial_size : Nat
  small_cube_size : Nat
  small_cube_count : Nat
  center_face_removed : Bool
  center_small_cube_removed : Bool
  small_cube_faces_removed : Bool

/-- Calculates the surface area of the modified cube structure -/
def surface_area (c : ModifiedCube) : Nat :=
  sorry

/-- The specific cube structure from the problem -/
def problem_cube : ModifiedCube :=
  { initial_size := 12
  , small_cube_size := 3
  , small_cube_count := 64
  , center_face_removed := true
  , center_small_cube_removed := true
  , small_cube_faces_removed := true }

/-- Theorem stating that the surface area of the problem_cube is 4344 -/
theorem problem_cube_surface_area :
  surface_area problem_cube = 4344 :=
sorry

end problem_cube_surface_area_l3685_368555


namespace circle_tangency_distance_l3685_368558

theorem circle_tangency_distance (r_O r_O' d_external : ℝ) : 
  r_O = 5 → 
  d_external = 9 → 
  r_O + r_O' = d_external → 
  |r_O' - r_O| = 1 := by
  sorry

end circle_tangency_distance_l3685_368558


namespace prob_white_then_black_l3685_368544

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents the bag of balls -/
def Bag := Finset Color

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 2

/-- The bag containing the balls -/
def initial_bag : Bag := sorry

/-- The probability of drawing a white ball first and a black ball second without replacement -/
theorem prob_white_then_black : 
  (white_balls / total_balls) * (black_balls / (total_balls - 1)) = 3 / 10 := by sorry

end prob_white_then_black_l3685_368544


namespace marathon_training_percentage_l3685_368514

theorem marathon_training_percentage (total_miles : ℝ) (day3_miles : ℝ) 
  (h1 : total_miles = 70)
  (h2 : day3_miles = 28) : 
  ∃ (p : ℝ), 
    p * total_miles + 0.5 * (total_miles - p * total_miles) + day3_miles = total_miles ∧ 
    p = 0.2 := by
sorry

end marathon_training_percentage_l3685_368514


namespace sum_reciprocals_bound_l3685_368521

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b ≥ 2) ∧ (∀ ε > 0, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1 / a' + 1 / b' < 2 + ε) :=
sorry

end sum_reciprocals_bound_l3685_368521


namespace unique_triple_divisibility_l3685_368563

theorem unique_triple_divisibility (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ 
  (a * b - 1) * (b * c - 1) * (c * a - 1) % (a * b * c) = 0 →
  a = 2 ∧ b = 3 ∧ c = 5 := by
sorry

end unique_triple_divisibility_l3685_368563


namespace unique_positive_solution_l3685_368518

def f (x : ℝ) : ℝ := x^11 + 9*x^10 + 19*x^9 + 2023*x^8 - 1421*x^7 + 5

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end unique_positive_solution_l3685_368518


namespace circle_theorem_l3685_368573

/-- A structure representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A function to check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- A function to check if a point is on a circle -/
def isOn (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- A function to check if four points are concyclic -/
def areConcyclic (p1 p2 p3 p4 : Point) : Prop :=
  ∃ c : Circle, isOn p1 c ∧ isOn p2 c ∧ isOn p3 c ∧ isOn p4 c

theorem circle_theorem (n : ℕ) (points : Fin (2*n+3) → Point) 
  (h : ∀ (a b c d : Fin (2*n+3)), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d → 
    ¬ areConcyclic (points a) (points b) (points c) (points d)) :
  ∃ (c : Circle) (a b d : Fin (2*n+3)), 
    a ≠ b ∧ b ≠ d ∧ a ≠ d ∧
    isOn (points a) c ∧ isOn (points b) c ∧ isOn (points d) c ∧
    (∃ (inside outside : Fin n → Fin (2*n+3)), 
      (∀ i : Fin n, isInside (points (inside i)) c) ∧
      (∀ i : Fin n, ¬isInside (points (outside i)) c)) :=
sorry

end circle_theorem_l3685_368573


namespace max_m_is_6_min_value_is_9_min_value_achievable_l3685_368520

-- Define the condition that |x+2|+|x-4|-m≥0 for all real x
def condition (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0

-- Theorem 1: The maximum value of m is 6
theorem max_m_is_6 (h : condition m) : m ≤ 6 :=
sorry

-- Define the constraint equation for a and b
def constraint (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ 4 / (a + 5*b) + 1 / (3*a + 2*b) = 1

-- Theorem 2: The minimum value of 4a+7b is 9
theorem min_value_is_9 (a b : ℝ) (h : constraint a b) : 
  4*a + 7*b ≥ 9 :=
sorry

-- Theorem 3: The minimum value 9 is achievable
theorem min_value_achievable : 
  ∃ a b : ℝ, constraint a b ∧ 4*a + 7*b = 9 :=
sorry

end max_m_is_6_min_value_is_9_min_value_achievable_l3685_368520


namespace chunks_for_two_dozen_bananas_l3685_368593

/-- The number of chunks needed to purchase a given number of bananas -/
def chunks_needed (bananas : ℚ) : ℚ :=
  (bananas * 3 * 8) / (7 * 5)

theorem chunks_for_two_dozen_bananas :
  chunks_needed 24 = 576 / 35 := by
  sorry

end chunks_for_two_dozen_bananas_l3685_368593


namespace divisor_power_difference_l3685_368506

theorem divisor_power_difference (k : ℕ) : 
  (18 ^ k : ℕ) ∣ 624938 → 6 ^ k - k ^ 6 = 1 := by
  sorry

end divisor_power_difference_l3685_368506


namespace paint_for_snake_2016_l3685_368529

/-- Amount of paint needed for a snake of cubes -/
def paint_for_snake (num_cubes : ℕ) (paint_per_cube : ℕ) : ℕ :=
  let periodic_fragment := 6
  let paint_per_fragment := periodic_fragment * paint_per_cube
  let num_fragments := num_cubes / periodic_fragment
  let paint_for_fragments := num_fragments * paint_per_fragment
  let paint_for_ends := 2 * (paint_per_cube / 3)
  paint_for_fragments + paint_for_ends

theorem paint_for_snake_2016 :
  paint_for_snake 2016 60 = 121000 := by
  sorry

end paint_for_snake_2016_l3685_368529


namespace soap_packing_problem_l3685_368507

theorem soap_packing_problem :
  ∃! N : ℕ, 200 < N ∧ N < 300 ∧ 2007 % N = 5 := by
  sorry

end soap_packing_problem_l3685_368507


namespace rocky_knockouts_l3685_368513

theorem rocky_knockouts (total_fights : ℕ) (knockout_percentage : ℚ) (first_round_percentage : ℚ) :
  total_fights = 190 →
  knockout_percentage = 1/2 →
  first_round_percentage = 1/5 →
  (↑total_fights * knockout_percentage * first_round_percentage : ℚ) = 19 := by
sorry

end rocky_knockouts_l3685_368513


namespace polynomial_factorization_l3685_368576

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  -(x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end polynomial_factorization_l3685_368576


namespace cube_root_of_216_l3685_368592

theorem cube_root_of_216 (x : ℝ) (h : (x ^ (1/2)) ^ 3 = 216) : x = 36 := by
  sorry

end cube_root_of_216_l3685_368592


namespace exponent_addition_l3685_368564

theorem exponent_addition (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_addition_l3685_368564


namespace max_value_of_squared_differences_l3685_368536

theorem max_value_of_squared_differences (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 = 27) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 9 → (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 27) := by
  sorry

end max_value_of_squared_differences_l3685_368536


namespace max_value_when_xy_over_z_maximized_l3685_368519

/-- Given positive real numbers x, y, and z satisfying x^2 - 3xy + 4y^2 - z = 0,
    the maximum value of (2/x + 1/y - 2/z + 2) is 3 when xy/z is maximized. -/
theorem max_value_when_xy_over_z_maximized
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_eq : x^2 - 3*x*y + 4*y^2 - z = 0) :
  let f := fun (x y z : ℝ) => 2/x + 1/y - 2/z + 2
  let g := fun (x y z : ℝ) => x*y/z
  ∃ (x' y' z' : ℝ),
    x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 ∧
    (∀ a b c, a > 0 → b > 0 → c > 0 → a^2 - 3*a*b + 4*b^2 - c = 0 →
      g a b c ≤ g x' y' z') ∧
    f x' y' z' = 3 ∧
    (∀ a b c, a > 0 → b > 0 → c > 0 → a^2 - 3*a*b + 4*b^2 - c = 0 →
      f a b c ≤ 3) :=
by sorry

end max_value_when_xy_over_z_maximized_l3685_368519


namespace correct_division_l3685_368505

theorem correct_division (dividend : ℕ) : 
  (dividend / 47 = 5 ∧ dividend % 47 = 8) → 
  (dividend / 74 = 3 ∧ dividend % 74 = 21) :=
by
  sorry

end correct_division_l3685_368505


namespace job_completion_time_l3685_368542

/-- The time taken for two workers to complete a job together, given their relative speeds and the time taken by one worker alone. -/
theorem job_completion_time 
  (a_speed : ℝ) -- Speed of worker a
  (b_speed : ℝ) -- Speed of worker b
  (a_alone_time : ℝ) -- Time taken by worker a alone
  (h1 : a_speed = 1.5 * b_speed) -- a is 1.5 times as fast as b
  (h2 : a_alone_time = 30) -- a alone can do the work in 30 days
  : (1 / (1 / a_alone_time + 1 / (a_alone_time * 1.5))) = 18 := by
  sorry

end job_completion_time_l3685_368542


namespace absolute_value_and_exponentiation_l3685_368541

theorem absolute_value_and_exponentiation :
  (abs (-2023) = 2023) ∧ ((-1 : ℤ)^2023 = -1) := by sorry

end absolute_value_and_exponentiation_l3685_368541


namespace play_roles_assignment_l3685_368552

/-- The number of ways to assign roles in a play -/
def assignRoles (numMen numWomen numMaleRoles numFemaleRoles numEitherRoles : ℕ) : ℕ :=
  let remainingActors := numMen + numWomen - numMaleRoles - numFemaleRoles
  (numMen.choose numMaleRoles) * 
  (numWomen.choose numFemaleRoles) * 
  (remainingActors.choose numEitherRoles)

theorem play_roles_assignment :
  assignRoles 6 7 3 3 3 = 5292000 := by
  sorry

end play_roles_assignment_l3685_368552


namespace polynomial_value_at_negative_one_l3685_368588

/-- A polynomial of degree 5 with integer coefficients -/
def polynomial (a₁ a₂ a₃ a₄ a₅ : ℤ) (x : ℝ) : ℝ :=
  x^5 + a₁ * x^4 + a₂ * x^3 + a₃ * x^2 + a₄ * x + a₅

/-- Theorem stating the value of f(-1) given specific conditions -/
theorem polynomial_value_at_negative_one
  (a₁ a₂ a₃ a₄ a₅ : ℤ)
  (h1 : polynomial a₁ a₂ a₃ a₄ a₅ (Real.sqrt 3 + Real.sqrt 2) = 0)
  (h2 : polynomial a₁ a₂ a₃ a₄ a₅ 1 + polynomial a₁ a₂ a₃ a₄ a₅ 3 = 0) :
  polynomial a₁ a₂ a₃ a₄ a₅ (-1) = 24 := by
  sorry

end polynomial_value_at_negative_one_l3685_368588


namespace complex_equation_sum_l3685_368575

theorem complex_equation_sum (a t : ℝ) (i : ℂ) : 
  i * i = -1 → 
  a + i = (1 + 2*i) * (t*i) → 
  t + a = -1 := by
sorry

end complex_equation_sum_l3685_368575


namespace isosceles_trapezoid_area_is_96_l3685_368590

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The shorter base of the trapezoid
  short_base : ℝ
  -- The perimeter of the trapezoid
  perimeter : ℝ
  -- The diagonal bisects the obtuse angle
  diagonal_bisects_obtuse_angle : Bool

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating that an isosceles trapezoid with given properties has an area of 96 -/
theorem isosceles_trapezoid_area_is_96 (t : IsoscelesTrapezoid) 
  (h1 : t.short_base = 3)
  (h2 : t.perimeter = 42)
  (h3 : t.diagonal_bisects_obtuse_angle = true) :
  area t = 96 := by sorry

end isosceles_trapezoid_area_is_96_l3685_368590


namespace triangle_has_at_least_two_acute_angles_l3685_368550

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the property that the sum of angles in a triangle is 180°
def validTriangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define an acute angle
def isAcute (angle : Real) : Prop :=
  0 < angle ∧ angle < 90

-- Theorem: A triangle has at least two acute angles
theorem triangle_has_at_least_two_acute_angles (t : Triangle) 
  (h : validTriangle t) : 
  (isAcute t.angle1 ∧ isAcute t.angle2) ∨ 
  (isAcute t.angle1 ∧ isAcute t.angle3) ∨ 
  (isAcute t.angle2 ∧ isAcute t.angle3) :=
by
  sorry

end triangle_has_at_least_two_acute_angles_l3685_368550


namespace rhombus_side_length_l3685_368572

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = p.1^2

def parallel_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

def area (r : Rhombus) : ℝ := 128

-- Define the theorem
theorem rhombus_side_length (ABCD : Rhombus) :
  (on_parabola ABCD.A) →
  (on_parabola ABCD.B) →
  (on_parabola ABCD.C) →
  (on_parabola ABCD.D) →
  (ABCD.A = (0, 0)) →
  (parallel_to_x_axis ABCD.B ABCD.C) →
  (area ABCD = 128) →
  Real.sqrt ((ABCD.B.1 - ABCD.C.1)^2 + (ABCD.B.2 - ABCD.C.2)^2) = 16 :=
by
  sorry

end rhombus_side_length_l3685_368572


namespace largest_option_l3685_368591

open Real

/-- Given a function f: ℝ → ℝ satisfying xf'(x) + f(x) > 0 for all x, 
    and 0 < a < b < 1, prove that log_{ba} · f(log_{ba}) is the largest among the given options -/
theorem largest_option (f : ℝ → ℝ) (f_deriv : ℝ → ℝ) (a b : ℝ) 
  (h_f : ∀ x, x * f_deriv x + f x > 0)
  (h_a : 0 < a) (h_ab : a < b) (h_b : b < 1) :
  let D := (log b / log a) * f (log b / log a)
  D > a * b * f (a * b) ∧
  D > b * a * f (b * a) ∧
  D > (log a / log b) * f (log a / log b) := by
sorry


end largest_option_l3685_368591


namespace intersection_line_l3685_368502

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line
def line (x y : ℝ) : Prop := 3*x - 3*y - 10 = 0

-- Theorem statement
theorem intersection_line :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle1 B.1 B.2) →
  (circle2 A.1 A.2 ∧ circle2 B.1 B.2) →
  A ≠ B →
  (line A.1 A.2 ∧ line B.1 B.2) :=
sorry

end intersection_line_l3685_368502


namespace expand_product_l3685_368546

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 6) = 2*x^2 + 15*x + 18 := by
  sorry

end expand_product_l3685_368546


namespace folded_circle_cut_theorem_l3685_368512

/-- Represents a circular paper folded along its diameter. -/
structure FoldedCircle :=
  (diameter : ℝ)

/-- Represents a straight line drawn on the folded circular paper. -/
structure Line :=
  (angle : ℝ)  -- Angle with respect to the diameter

/-- Calculates the number of pieces resulting from cutting a folded circular paper along given lines. -/
def num_pieces (circle : FoldedCircle) (lines : List Line) : ℕ :=
  sorry

/-- Theorem stating the minimum and maximum number of pieces when cutting a folded circular paper with 3 lines. -/
theorem folded_circle_cut_theorem (circle : FoldedCircle) :
  (∃ (lines : List Line), lines.length = 3 ∧ num_pieces circle lines = 4) ∧
  (∃ (lines : List Line), lines.length = 3 ∧ num_pieces circle lines = 7) ∧
  (∀ (lines : List Line), lines.length = 3 → 4 ≤ num_pieces circle lines ∧ num_pieces circle lines ≤ 7) :=
sorry

end folded_circle_cut_theorem_l3685_368512


namespace jean_jail_time_l3685_368551

theorem jean_jail_time 
  (arson_counts : ℕ)
  (burglary_charges : ℕ)
  (arson_sentence : ℕ)
  (burglary_sentence : ℕ)
  (h1 : arson_counts = 3)
  (h2 : burglary_charges = 2)
  (h3 : arson_sentence = 36)
  (h4 : burglary_sentence = 18)
  : 
  arson_counts * arson_sentence + 
  burglary_charges * burglary_sentence + 
  (6 * burglary_charges) * (burglary_sentence / 3) = 216 := by
  sorry

#check jean_jail_time

end jean_jail_time_l3685_368551


namespace system_solution_l3685_368533

theorem system_solution (x y a : ℝ) : 
  (4 * x + y = a) → 
  (3 * x + 4 * y^2 = 3 * a) → 
  (x = 3) → 
  (a = 15 ∨ a = 9.75) := by
sorry

end system_solution_l3685_368533


namespace negative_terms_min_value_at_min_value_l3685_368559

/-- The sequence a_n defined as n^2 - 5n + 4 -/
def a_n (n : ℝ) : ℝ := n^2 - 5*n + 4

/-- There are exactly two integer values of n for which a_n < 0 -/
theorem negative_terms : ∃! (s : Finset ℤ), (∀ n ∈ s, a_n n < 0) ∧ s.card = 2 :=
sorry

/-- The minimum value of a_n occurs when n = 5/2 -/
theorem min_value_at : ∀ n : ℝ, a_n n ≥ a_n (5/2) :=
sorry

/-- The minimum value of a_n is -1/4 -/
theorem min_value : a_n (5/2) = -1/4 :=
sorry

end negative_terms_min_value_at_min_value_l3685_368559


namespace closest_integer_to_cube_root_l3685_368523

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 : ℝ) → 
  ∃ (n : ℤ), n = 10 ∧ 
  ∀ (m : ℤ), |x^(1/3) - (n : ℝ)| ≤ |x^(1/3) - (m : ℝ)| := by
  sorry

end closest_integer_to_cube_root_l3685_368523
