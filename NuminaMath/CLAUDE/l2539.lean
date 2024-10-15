import Mathlib

namespace NUMINAMATH_CALUDE_handshakes_count_l2539_253926

/-- Represents a social gathering with specific group interactions -/
structure SocialGathering where
  total_people : ℕ
  group_a : ℕ  -- People who all know each other
  group_b : ℕ  -- People who know no one
  group_c : ℕ  -- People who know exactly 15 from group_a
  h_total : total_people = group_a + group_b + group_c
  h_group_a : group_a = 25
  h_group_b : group_b = 10
  h_group_c : group_c = 5

/-- Calculates the number of handshakes in the social gathering -/
def handshakes (sg : SocialGathering) : ℕ :=
  let ab_handshakes := sg.group_b * (sg.group_a + sg.group_c)
  let b_internal_handshakes := sg.group_b * (sg.group_b - 1) / 2
  let c_handshakes := sg.group_c * (sg.group_a - 15 + sg.group_c)
  ab_handshakes + b_internal_handshakes + c_handshakes

/-- Theorem stating that the number of handshakes in the given social gathering is 420 -/
theorem handshakes_count (sg : SocialGathering) : handshakes sg = 420 := by
  sorry

#eval handshakes { total_people := 40, group_a := 25, group_b := 10, group_c := 5,
                   h_total := rfl, h_group_a := rfl, h_group_b := rfl, h_group_c := rfl }

end NUMINAMATH_CALUDE_handshakes_count_l2539_253926


namespace NUMINAMATH_CALUDE_senate_committee_arrangements_l2539_253964

/-- The number of ways to arrange n distinguishable people around a circular table -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of politicians on the Senate committee -/
def numPoliticians : ℕ := 12

theorem senate_committee_arrangements :
  circularArrangements numPoliticians = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_arrangements_l2539_253964


namespace NUMINAMATH_CALUDE_equation_solution_l2539_253965

theorem equation_solution : 
  ∃! y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2539_253965


namespace NUMINAMATH_CALUDE_coefficient_x5y4_in_expansion_x_plus_y_9_l2539_253925

theorem coefficient_x5y4_in_expansion_x_plus_y_9 :
  (Finset.range 10).sum (λ k => Nat.choose 9 k * X^k * Y^(9 - k)) =
  126 * X^5 * Y^4 + (Finset.range 10).sum (λ k => if k ≠ 5 then Nat.choose 9 k * X^k * Y^(9 - k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5y4_in_expansion_x_plus_y_9_l2539_253925


namespace NUMINAMATH_CALUDE_average_weight_problem_l2539_253907

/-- Given the weights of three people with specific relationships, prove their average weight. -/
theorem average_weight_problem (jalen_weight ponce_weight ishmael_weight : ℕ) : 
  jalen_weight = 160 ∧ 
  ponce_weight = jalen_weight - 10 ∧ 
  ishmael_weight = ponce_weight + 20 → 
  (jalen_weight + ponce_weight + ishmael_weight) / 3 = 160 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_problem_l2539_253907


namespace NUMINAMATH_CALUDE_eighteen_percent_of_500_is_90_l2539_253969

theorem eighteen_percent_of_500_is_90 : 
  (18 : ℚ) / 100 * 500 = 90 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_percent_of_500_is_90_l2539_253969


namespace NUMINAMATH_CALUDE_log_equality_implies_ln_a_l2539_253909

theorem log_equality_implies_ln_a (a : ℝ) (h : a > 0) :
  (Real.log (8 * a) / Real.log (9 * a) = Real.log (2 * a) / Real.log (3 * a)) →
  (Real.log a = (Real.log 2 * Real.log 3) / (Real.log 3 - 2 * Real.log 2)) := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ln_a_l2539_253909


namespace NUMINAMATH_CALUDE_real_part_of_z_l2539_253988

theorem real_part_of_z (z : ℂ) (h : (z + 1).re = 0) : z.re = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2539_253988


namespace NUMINAMATH_CALUDE_fraction_division_l2539_253980

theorem fraction_division (x y z : ℚ) (hx : x = 3) (hy : y = 4) (hz : z = 5) :
  (z / y) / (z / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l2539_253980


namespace NUMINAMATH_CALUDE_sine_inequality_l2539_253952

theorem sine_inequality : 
  (∀ x y, x ∈ Set.Icc 0 (π/2) → y ∈ Set.Icc 0 (π/2) → x < y → Real.sin x < Real.sin y) →
  3*π/7 > 2*π/5 →
  3*π/7 ∈ Set.Icc 0 (π/2) →
  2*π/5 ∈ Set.Icc 0 (π/2) →
  Real.sin (3*π/7) > Real.sin (2*π/5) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l2539_253952


namespace NUMINAMATH_CALUDE_expression_equals_20_times_10_pow_1500_l2539_253903

theorem expression_equals_20_times_10_pow_1500 :
  (2^1500 + 5^1501)^2 - (2^1500 - 5^1501)^2 = 20 * 10^1500 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_20_times_10_pow_1500_l2539_253903


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2539_253912

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 6 = 0 ∧ x = 2) → 
  (∃ x : ℝ, x^2 + k*x + 6 = 0 ∧ x = 3 ∧ k = -5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2539_253912


namespace NUMINAMATH_CALUDE_diameter_is_chord_l2539_253993

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
def isChord (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

-- Define a diameter
def isDiameter (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 4 * c.radius^2

-- Theorem: A diameter is a chord
theorem diameter_is_chord (c : Circle) (p q : ℝ × ℝ) :
  isDiameter c p q → isChord c p q :=
by
  sorry


end NUMINAMATH_CALUDE_diameter_is_chord_l2539_253993


namespace NUMINAMATH_CALUDE_circus_revenue_l2539_253922

/-- Calculates the total revenue from circus ticket sales -/
theorem circus_revenue (lower_price upper_price : ℕ) (total_tickets lower_tickets : ℕ) :
  lower_price = 30 →
  upper_price = 20 →
  total_tickets = 80 →
  lower_tickets = 50 →
  lower_price * lower_tickets + upper_price * (total_tickets - lower_tickets) = 2100 := by
sorry

end NUMINAMATH_CALUDE_circus_revenue_l2539_253922


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l2539_253906

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line
def L (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles :
  ∃ (A B : ℝ × ℝ), 
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ 
    A ≠ B ∧
    L ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
    (B.2 - A.2) * ((A.1 + B.1) / 2 - A.1) = (A.1 - B.1) * ((A.2 + B.2) / 2 - A.2) :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l2539_253906


namespace NUMINAMATH_CALUDE_f_2_eq_0_l2539_253982

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_2_eq_0 :
  f 2 = horner_eval [1, -12, 60, -160, 240, -192, 64] 2 ∧
  horner_eval [1, -12, 60, -160, 240, -192, 64] 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_2_eq_0_l2539_253982


namespace NUMINAMATH_CALUDE_min_doors_for_safety_l2539_253937

/-- Represents a spaceship with a given number of corridors -/
structure Spaceship :=
  (corridors : ℕ)

/-- Represents the state of doors in the spaceship -/
def DoorState := Fin 23 → Bool

/-- Checks if there exists a path from reactor to lounge -/
def hasPath (s : Spaceship) (state : DoorState) : Prop :=
  sorry -- Definition of path existence

/-- Counts the number of closed doors -/
def closedDoors (state : DoorState) : ℕ :=
  sorry -- Count of closed doors

/-- Theorem stating the minimum number of doors to close for safety -/
theorem min_doors_for_safety (s : Spaceship) :
  (s.corridors = 23) →
  (∀ (state : DoorState), closedDoors state ≥ 22 → ¬hasPath s state) ∧
  (∃ (state : DoorState), closedDoors state = 21 ∧ hasPath s state) :=
sorry

#check min_doors_for_safety

end NUMINAMATH_CALUDE_min_doors_for_safety_l2539_253937


namespace NUMINAMATH_CALUDE_water_for_chickens_l2539_253992

/-- Calculates the amount of water needed for chickens given the total water needed and the water needed for pigs and horses. -/
theorem water_for_chickens 
  (num_pigs : ℕ) 
  (num_horses : ℕ) 
  (water_per_pig : ℕ) 
  (total_water : ℕ) 
  (h1 : num_pigs = 8)
  (h2 : num_horses = 10)
  (h3 : water_per_pig = 3)
  (h4 : total_water = 114) :
  total_water - (num_pigs * water_per_pig + num_horses * (2 * water_per_pig)) = 30 := by
  sorry

#check water_for_chickens

end NUMINAMATH_CALUDE_water_for_chickens_l2539_253992


namespace NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l2539_253905

-- Define the circles
def circle1 : Real × Real × Real := (0, 0, 3)  -- (center_x, center_y, radius)
def circle2 : Real × Real × Real := (12, 0, 5)  -- (center_x, center_y, radius)

-- Define the theorem
theorem tangent_intersection_x_coordinate :
  ∃ (x : Real),
    x > 0 ∧  -- Intersection to the right of origin
    (let (x1, y1, r1) := circle1
     let (x2, y2, r2) := circle2
     (x - x1) / (x - x2) = r1 / r2) ∧
    x = 18 := by
  sorry


end NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l2539_253905


namespace NUMINAMATH_CALUDE_sum_equals_negative_twenty_six_thirds_l2539_253977

theorem sum_equals_negative_twenty_six_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 10) : 
  a + b + c + d = -26/3 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_twenty_six_thirds_l2539_253977


namespace NUMINAMATH_CALUDE_expression_evaluation_l2539_253936

-- Define the expression as a function
def f (x : ℚ) : ℚ := (4 + x * (4 + x) - 4^2) / (x - 4 + x^2 + 2*x)

-- State the theorem
theorem expression_evaluation :
  f (-3) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2539_253936


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l2539_253962

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define an interior point of a triangle
def is_interior_point (P : Point) (t : Triangle) : Prop := sorry

-- Define parallel lines
def parallel_line (P : Point) (l : Line) : Line := sorry

-- Define the division of a triangle by parallel lines
def divide_triangle (t : Triangle) (P : Point) : Prop := sorry

-- Define the areas of the smaller triangles
def small_triangle_areas (t : Triangle) (P : Point) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_area_inequality (ABC : Triangle) (P : Point) :
  is_interior_point P ABC →
  divide_triangle ABC P →
  let (S1, S2, S3) := small_triangle_areas ABC P
  area ABC ≤ 3 * (S1 + S2 + S3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l2539_253962


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2539_253930

theorem max_area_rectangle (perimeter : ℕ) (area : ℕ → ℕ → ℕ) :
  perimeter = 150 →
  (∀ w h : ℕ, area w h = w * h) →
  (∀ w h : ℕ, 2 * w + 2 * h = perimeter → area w h ≤ 1406) ∧
  (∃ w h : ℕ, 2 * w + 2 * h = perimeter ∧ area w h = 1406) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2539_253930


namespace NUMINAMATH_CALUDE_total_students_l2539_253986

theorem total_students (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 →
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_l2539_253986


namespace NUMINAMATH_CALUDE_impossible_arrangement_l2539_253928

theorem impossible_arrangement : ¬ ∃ (a b : Fin 2005 → Fin 4010),
  (∀ i : Fin 2005, a i < b i) ∧
  (∀ i : Fin 2005, b i - a i = i.val + 1) ∧
  (∀ k : Fin 4010, ∃! i : Fin 2005, a i = k ∨ b i = k) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l2539_253928


namespace NUMINAMATH_CALUDE_cost_of_300_candies_l2539_253985

/-- The cost of a single candy in cents -/
def candy_cost : ℕ := 5

/-- The number of candies -/
def num_candies : ℕ := 300

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 300 candies is 15 dollars -/
theorem cost_of_300_candies :
  (num_candies * candy_cost) / cents_per_dollar = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_300_candies_l2539_253985


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l2539_253966

/-- The number of ways to arrange animals in cages -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  (Nat.factorial 4) * 
  (Nat.factorial chickens) * 
  (Nat.factorial dogs) * 
  (Nat.factorial cats) * 
  (Nat.factorial rabbits)

/-- Theorem stating the number of arrangements for the given problem -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 3 3 5 2 = 207360 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l2539_253966


namespace NUMINAMATH_CALUDE_basketball_season_games_l2539_253927

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team -/
def intra_conference_games : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams.choose 2 * intra_conference_games) + (num_teams * non_conference_games)

theorem basketball_season_games :
  total_games = 150 := by
sorry

end NUMINAMATH_CALUDE_basketball_season_games_l2539_253927


namespace NUMINAMATH_CALUDE_robin_initial_distance_l2539_253951

/-- The distance Robin walked before realizing he forgot his bag -/
def initial_distance : ℝ := sorry

/-- The distance between Robin's house and the city center -/
def house_to_center : ℝ := 500

/-- The total distance Robin walked -/
def total_distance : ℝ := 900

theorem robin_initial_distance :
  initial_distance = 200 :=
by
  have journey_equation : 2 * initial_distance + house_to_center = total_distance := by sorry
  sorry

end NUMINAMATH_CALUDE_robin_initial_distance_l2539_253951


namespace NUMINAMATH_CALUDE_root_sum_product_l2539_253967

theorem root_sum_product (a b : ℝ) : 
  (a^4 + 2*a^3 - 4*a - 1 = 0) →
  (b^4 + 2*b^3 - 4*b - 1 = 0) →
  (a ≠ b) →
  (a*b + a + b = Real.sqrt 3 - 2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l2539_253967


namespace NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l2539_253995

theorem smallest_integer_solution_inequality :
  ∀ (x : ℤ), (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 ∧
  ∃ (y : ℤ), y < -2 ∧ (9 * y + 8) / 6 - y / 3 < -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_inequality_l2539_253995


namespace NUMINAMATH_CALUDE_preimage_of_one_l2539_253946

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem preimage_of_one (x : ℝ) : f x = 1 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_one_l2539_253946


namespace NUMINAMATH_CALUDE_pascal_triangle_100th_row_10th_number_l2539_253997

theorem pascal_triangle_100th_row_10th_number :
  let n : ℕ := 99  -- row number (100 numbers in the row, so n + 1 = 100)
  let k : ℕ := 9   -- 10th number (0-indexed)
  (n.choose k) = (Nat.choose 99 9) := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_100th_row_10th_number_l2539_253997


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l2539_253908

theorem cot_thirty_degrees : 
  let cos_thirty : ℝ := Real.sqrt 3 / 2
  let sin_thirty : ℝ := 1 / 2
  let cot (θ : ℝ) : ℝ := (Real.cos θ) / (Real.sin θ)
  cot (30 * π / 180) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l2539_253908


namespace NUMINAMATH_CALUDE_possible_m_values_l2539_253950

theorem possible_m_values (M N : Set ℝ) (m : ℝ) :
  M = {x : ℝ | 2 * x^2 - 5 * x - 3 = 0} →
  N = {x : ℝ | m * x = 1} →
  N ⊆ M →
  {m | ∃ (x : ℝ), x ∈ N} = {-2, 1/3} :=
by sorry

end NUMINAMATH_CALUDE_possible_m_values_l2539_253950


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l2539_253998

theorem fixed_point_of_linear_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * x + 2
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l2539_253998


namespace NUMINAMATH_CALUDE_sum_of_digits_of_gcd_l2539_253996

-- Define the numbers given in the problem
def a : ℕ := 1305
def b : ℕ := 4665
def c : ℕ := 6905

-- Define the function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- State the theorem
theorem sum_of_digits_of_gcd : sum_of_digits (Nat.gcd (b - a) (Nat.gcd (c - b) (c - a))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_gcd_l2539_253996


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l2539_253948

/-- Represents a cylinder formed by rotating a rectangle around one of its sides. -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Theorem: Given a cylinder formed by rotating a rectangle with a diagonal of 26 cm
    around one of its sides, if a perpendicular plane equidistant from the bases has
    a total surface area of 2720 cm², then the height of the cylinder is 24 cm and
    its base radius is 10 cm. -/
theorem cylinder_dimensions (c : Cylinder) :
  c.height ^ 2 + c.radius ^ 2 = 26 ^ 2 →
  8 * c.radius ^ 2 + 8 * c.radius * c.height = 2720 →
  c.height = 24 ∧ c.radius = 10 := by
  sorry

#check cylinder_dimensions

end NUMINAMATH_CALUDE_cylinder_dimensions_l2539_253948


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l2539_253945

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 12 = 0) 
  (h2 : n ≥ 24) : ∃ (blue red green yellow : ℕ),
  blue = n / 3 ∧ 
  red = n / 4 ∧ 
  green = 6 ∧ 
  yellow = n - (blue + red + green) ∧ 
  blue + red + green + yellow = n ∧
  yellow ≥ 4 ∧
  (∀ m : ℕ, m < n → ¬(∃ b r g y : ℕ, 
    b = m / 3 ∧ 
    r = m / 4 ∧ 
    g = 6 ∧ 
    y = m - (b + r + g) ∧ 
    b + r + g + y = m ∧ 
    y ≥ 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l2539_253945


namespace NUMINAMATH_CALUDE_sock_purchase_theorem_l2539_253963

/-- Represents the number of pairs of socks at each price point -/
structure SockPurchase where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the SockPurchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.four_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 4 * p.four_dollar + 5 * p.five_dollar = 38 ∧
  p.two_dollar ≥ 1 ∧ p.four_dollar ≥ 1 ∧ p.five_dollar ≥ 1

theorem sock_purchase_theorem :
  ∃ (p : SockPurchase), is_valid_purchase p ∧ p.two_dollar = 12 :=
by sorry

end NUMINAMATH_CALUDE_sock_purchase_theorem_l2539_253963


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2539_253924

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of a for which the given lines are parallel -/
theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, 2 * x + a * y + 2 = 0 ↔ a * x + (a + 4) * y - 1 = 0) ↔ (a = 4 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2539_253924


namespace NUMINAMATH_CALUDE_yellow_given_popped_prob_l2539_253940

-- Define the probabilities of kernel colors in the bag
def white_prob : ℚ := 1/2
def yellow_prob : ℚ := 1/3
def blue_prob : ℚ := 1/6

-- Define the probabilities of popping for each color
def white_pop_prob : ℚ := 2/3
def yellow_pop_prob : ℚ := 1/2
def blue_pop_prob : ℚ := 3/4

-- State the theorem
theorem yellow_given_popped_prob :
  let total_pop_prob := white_prob * white_pop_prob + yellow_prob * yellow_pop_prob + blue_prob * blue_pop_prob
  (yellow_prob * yellow_pop_prob) / total_pop_prob = 4/23 := by
  sorry

end NUMINAMATH_CALUDE_yellow_given_popped_prob_l2539_253940


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_500_l2539_253933

theorem closest_integer_to_cube_root_500 : 
  ∀ n : ℤ, |n - ⌊(500 : ℝ)^(1/3)⌋| ≥ |8 - ⌊(500 : ℝ)^(1/3)⌋| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_500_l2539_253933


namespace NUMINAMATH_CALUDE_trapezoid_area_l2539_253990

-- Define the rectangle ABCD
structure Rectangle where
  width : ℝ
  height : ℝ
  area : ℝ
  area_eq : area = width * height

-- Define the trapezoid DEFG
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

-- Define the problem setup
def problem_setup (rect : Rectangle) (trap : Trapezoid) : Prop :=
  rect.area = 108 ∧
  trap.base1 = rect.height / 2 ∧
  trap.base2 = rect.width / 2 ∧
  trap.height = rect.height / 2

-- Theorem to prove
theorem trapezoid_area 
  (rect : Rectangle) 
  (trap : Trapezoid) 
  (h : problem_setup rect trap) : 
  (trap.base1 + trap.base2) / 2 * trap.height = 27 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2539_253990


namespace NUMINAMATH_CALUDE_simplify_roots_l2539_253914

theorem simplify_roots : (625 : ℝ)^(1/4) * (125 : ℝ)^(1/3) = 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_roots_l2539_253914


namespace NUMINAMATH_CALUDE_unknown_number_problem_l2539_253968

theorem unknown_number_problem (x : ℝ) : 
  (0.1 * 30 + 0.15 * x = 10.5) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l2539_253968


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2539_253991

/-- Given a triangle ABC with side lengths a, b, and c satisfying (a+b+c)(b+c-a) = bc,
    prove that the measure of angle A is 120 degrees. -/
theorem triangle_angle_measure (a b c : ℝ) (h : (a + b + c) * (b + c - a) = b * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  A = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2539_253991


namespace NUMINAMATH_CALUDE_power_equation_solution_l2539_253973

theorem power_equation_solution : ∃ n : ℤ, (5 : ℝ) ^ (4 * n) = (1 / 5 : ℝ) ^ (n - 30) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2539_253973


namespace NUMINAMATH_CALUDE_twin_brothers_age_l2539_253929

/-- Theorem: Age of twin brothers
  Given that the product of their ages today is 13 less than the product of their ages a year from today,
  prove that the age of twin brothers today is 6 years old.
-/
theorem twin_brothers_age (x : ℕ) : x * x + 13 = (x + 1) * (x + 1) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_l2539_253929


namespace NUMINAMATH_CALUDE_honzik_payment_l2539_253943

theorem honzik_payment (lollipop_price ice_cream_price : ℕ) : 
  (3 * lollipop_price = 24) →
  (∃ n : ℕ, 2 ≤ n ∧ n ≤ 9 ∧ 4 * lollipop_price + n * ice_cream_price = 109) →
  lollipop_price + ice_cream_price = 19 :=
by sorry

end NUMINAMATH_CALUDE_honzik_payment_l2539_253943


namespace NUMINAMATH_CALUDE_product_prs_is_54_l2539_253941

theorem product_prs_is_54 (p r s : ℕ) : 
  3^p + 3^5 = 270 → 
  2^r + 58 = 122 → 
  7^2 + 5^s = 2504 → 
  p * r * s = 54 := by
sorry

end NUMINAMATH_CALUDE_product_prs_is_54_l2539_253941


namespace NUMINAMATH_CALUDE_system_solution_l2539_253961

theorem system_solution : 
  ∃ (x y z u : ℚ), 
    (x = 229 ∧ y = 149 ∧ z = 131 ∧ u = 121) ∧
    (x + y = 3/2 * (z + u)) ∧
    (x + z = -4/3 * (y + u)) ∧
    (x + u = 5/4 * (y + z)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2539_253961


namespace NUMINAMATH_CALUDE_video_game_pricing_l2539_253947

theorem video_game_pricing (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) :
  total_games = 16 →
  non_working_games = 8 →
  total_earnings = 56 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_video_game_pricing_l2539_253947


namespace NUMINAMATH_CALUDE_find_divisor_l2539_253994

theorem find_divisor : ∃ (D : ℕ), 
  (23 = 5 * D + 3) ∧ 
  (∃ (N : ℕ), N = 7 * D + 5) ∧ 
  D = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2539_253994


namespace NUMINAMATH_CALUDE_complex_simplification_l2539_253984

theorem complex_simplification :
  (4 - 3 * Complex.I) * 2 - (6 - 3 * Complex.I) = 2 - 3 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2539_253984


namespace NUMINAMATH_CALUDE_carmen_pets_difference_l2539_253919

/-- Proves that Carmen has 14 fewer cats than dogs after giving up some cats for adoption -/
theorem carmen_pets_difference (initial_cats initial_dogs : ℕ) 
  (cats_given_up_round1 cats_given_up_round2 cats_given_up_round3 : ℕ) : 
  initial_cats = 48 →
  initial_dogs = 36 →
  cats_given_up_round1 = 6 →
  cats_given_up_round2 = 12 →
  cats_given_up_round3 = 8 →
  initial_cats - (cats_given_up_round1 + cats_given_up_round2 + cats_given_up_round3) = initial_dogs - 14 :=
by
  sorry

end NUMINAMATH_CALUDE_carmen_pets_difference_l2539_253919


namespace NUMINAMATH_CALUDE_not_divisible_by_2019_l2539_253955

theorem not_divisible_by_2019 (n : ℕ) : ¬(2019 ∣ (n^2 + n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2019_l2539_253955


namespace NUMINAMATH_CALUDE_volleyball_team_starters_l2539_253960

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem volleyball_team_starters (total_players triplets twins starters : ℕ) 
  (h1 : total_players = 16)
  (h2 : triplets = 3)
  (h3 : twins = 2)
  (h4 : starters = 6) :
  (choose (total_players - triplets - twins) starters) + 
  (triplets * choose (total_players - triplets - twins) (starters - 1)) +
  (twins * choose (total_players - triplets - twins) (starters - 1)) = 2772 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_starters_l2539_253960


namespace NUMINAMATH_CALUDE_punch_mixture_difference_l2539_253921

/-- Proves that in a mixture with a 3:5 ratio of two components, 
    where the total volume is 72 cups, the difference between 
    the volumes of the two components is 18 cups. -/
theorem punch_mixture_difference (total_volume : ℕ) 
    (ratio_a : ℕ) (ratio_b : ℕ) (difference : ℕ) : 
    total_volume = 72 → 
    ratio_a = 3 → 
    ratio_b = 5 → 
    difference = ratio_b * (total_volume / (ratio_a + ratio_b)) - 
                 ratio_a * (total_volume / (ratio_a + ratio_b)) → 
    difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_punch_mixture_difference_l2539_253921


namespace NUMINAMATH_CALUDE_hadley_walk_back_home_l2539_253956

/-- The distance Hadley walked back home -/
def distance_back_home (distance_to_grocery : ℝ) (distance_to_pet : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance - (distance_to_grocery + distance_to_pet)

/-- Theorem: Hadley walked 3 miles back home -/
theorem hadley_walk_back_home :
  let distance_to_grocery : ℝ := 2
  let distance_to_pet : ℝ := 2 - 1
  let total_distance : ℝ := 6
  distance_back_home distance_to_grocery distance_to_pet total_distance = 3 := by
sorry

end NUMINAMATH_CALUDE_hadley_walk_back_home_l2539_253956


namespace NUMINAMATH_CALUDE_root_transformation_l2539_253999

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3 * r₁^2 + 13 = 0) ∧ 
  (r₂^3 - 3 * r₂^2 + 13 = 0) ∧ 
  (r₃^3 - 3 * r₃^2 + 13 = 0) →
  ((3 * r₁)^3 - 9 * (3 * r₁)^2 + 351 = 0) ∧
  ((3 * r₂)^3 - 9 * (3 * r₂)^2 + 351 = 0) ∧
  ((3 * r₃)^3 - 9 * (3 * r₃)^2 + 351 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l2539_253999


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l2539_253901

theorem largest_y_coordinate (x y : ℝ) : 
  (x - 3)^2 / 25 + (y - 2)^2 / 9 = 0 → y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l2539_253901


namespace NUMINAMATH_CALUDE_segment_construction_l2539_253972

theorem segment_construction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let x := a + b
  4 * (c * (a * c).sqrt).sqrt * x = 4 * (c * (a * c).sqrt).sqrt * a + 4 * (c * (a * c).sqrt).sqrt * b :=
by sorry

end NUMINAMATH_CALUDE_segment_construction_l2539_253972


namespace NUMINAMATH_CALUDE_polly_hungry_tweet_rate_l2539_253911

def happy_tweets_per_minute : ℕ := 18
def mirror_tweets_per_minute : ℕ := 45
def duration_per_state : ℕ := 20
def total_tweets : ℕ := 1340

def hungry_tweets_per_minute : ℕ := 4

theorem polly_hungry_tweet_rate :
  happy_tweets_per_minute * duration_per_state +
  hungry_tweets_per_minute * duration_per_state +
  mirror_tweets_per_minute * duration_per_state = total_tweets :=
by sorry

end NUMINAMATH_CALUDE_polly_hungry_tweet_rate_l2539_253911


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_3_and_9_l2539_253981

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem smallest_digit_divisible_by_3_and_9 : 
  ∃ (d : ℕ), d < 10 ∧ 
    is_divisible_by (528000 + d * 100 + 74) 3 ∧ 
    is_divisible_by (528000 + d * 100 + 74) 9 ∧
    ∀ (d' : ℕ), d' < d → 
      ¬(is_divisible_by (528000 + d' * 100 + 74) 3 ∧ 
        is_divisible_by (528000 + d' * 100 + 74) 9) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_3_and_9_l2539_253981


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2539_253918

theorem sqrt_sum_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2539_253918


namespace NUMINAMATH_CALUDE_cos_product_eighth_and_five_eighths_pi_l2539_253902

theorem cos_product_eighth_and_five_eighths_pi :
  Real.cos (π / 8) * Real.cos (5 * π / 8) = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_eighth_and_five_eighths_pi_l2539_253902


namespace NUMINAMATH_CALUDE_problem_statement_l2539_253913

-- Define the function f(x) = ax^2 + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1

-- Define what it means for a function to pass through a point
def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Define parallel relation for lines and planes
def parallel (α β : Set (ℝ × ℝ × ℝ)) : Prop := sorry

theorem problem_statement :
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ¬(passes_through (f a) (-1) 2)) ∧
  (∀ α β m : Set (ℝ × ℝ × ℝ), 
    parallel α β → (parallel m α ↔ parallel m β)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2539_253913


namespace NUMINAMATH_CALUDE_equiv_mod_seven_l2539_253953

theorem equiv_mod_seven (n : ℤ) : 0 ≤ n ∧ n ≤ 10 ∧ n ≡ -3137 [ZMOD 7] → n = 1 ∨ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_equiv_mod_seven_l2539_253953


namespace NUMINAMATH_CALUDE_solution_to_equation_l2539_253978

theorem solution_to_equation : ∃ x : ℝ, 12*x + 13*x + 16*x + 11 = 134 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2539_253978


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2539_253957

/-- An arithmetic sequence with positive terms where a_1 and a_3 are roots of x^2 - 8x + 7 = 0 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (a 1)^2 - 8*(a 1) + 7 = 0 ∧
  (a 3)^2 - 8*(a 3) + 7 = 0

/-- The general formula for the arithmetic sequence -/
def GeneralFormula (n : ℕ) : ℝ := 3 * n - 2

/-- Theorem stating that the general formula is correct for the given arithmetic sequence -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∀ n, a n = GeneralFormula n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2539_253957


namespace NUMINAMATH_CALUDE_dropped_student_score_l2539_253958

theorem dropped_student_score
  (total_students : ℕ)
  (remaining_students : ℕ)
  (initial_average : ℚ)
  (final_average : ℚ)
  (h1 : total_students = 16)
  (h2 : remaining_students = 15)
  (h3 : initial_average = 61.5)
  (h4 : final_average = 64)
  : (total_students : ℚ) * initial_average - (remaining_students : ℚ) * final_average = 24 := by
  sorry

end NUMINAMATH_CALUDE_dropped_student_score_l2539_253958


namespace NUMINAMATH_CALUDE_sequence_general_term_l2539_253917

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 3) :
  ∀ n : ℕ, a n = 3 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2539_253917


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2539_253975

theorem right_triangle_third_side 
  (a b c : ℝ) 
  (ha : a = 10) 
  (hb : b = 24) 
  (hright : a^2 + c^2 = b^2) : 
  c = 2 * Real.sqrt 119 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2539_253975


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l2539_253976

/-- The area of a rectangular plot with length thrice its breadth and breadth of 11 meters is 363 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 11 →
  length = 3 * breadth →
  area = length * breadth →
  area = 363 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l2539_253976


namespace NUMINAMATH_CALUDE_range_sum_l2539_253989

noncomputable def f (x : ℝ) : ℝ := 1 + (2^(x+1))/(2^x + 1) + Real.sin x

theorem range_sum (k : ℝ) (h : k > 0) :
  ∃ (m n : ℝ), (∀ x ∈ Set.Icc (-k) k, m ≤ f x ∧ f x ≤ n) ∧
                (∀ y, y ∈ Set.Icc m n ↔ ∃ x ∈ Set.Icc (-k) k, f x = y) ∧
                m + n = 4 :=
sorry

end NUMINAMATH_CALUDE_range_sum_l2539_253989


namespace NUMINAMATH_CALUDE_micah_typing_speed_l2539_253983

/-- The number of words Isaiah can type per minute. -/
def isaiah_words_per_minute : ℕ := 40

/-- The number of minutes in an hour. -/
def minutes_per_hour : ℕ := 60

/-- The difference in words typed per hour between Isaiah and Micah. -/
def word_difference_per_hour : ℕ := 1200

/-- The number of words Micah can type per minute. -/
def micah_words_per_minute : ℕ := 20

/-- Theorem stating that Micah can type 20 words per minute given the conditions. -/
theorem micah_typing_speed : micah_words_per_minute = 20 := by sorry

end NUMINAMATH_CALUDE_micah_typing_speed_l2539_253983


namespace NUMINAMATH_CALUDE_suit_price_calculation_suit_price_proof_l2539_253970

theorem suit_price_calculation (original_price : ℝ) 
  (increase_percentage : ℝ) (discount_percentage : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_percentage)
  let final_price := increased_price * (1 - discount_percentage)
  final_price

theorem suit_price_proof :
  suit_price_calculation 200 0.3 0.3 = 182 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_suit_price_proof_l2539_253970


namespace NUMINAMATH_CALUDE_systematic_sampling_method_l2539_253939

theorem systematic_sampling_method (population_size : ℕ) (sample_size : ℕ) 
  (h1 : population_size = 102) (h2 : sample_size = 9) : 
  ∃ (excluded : ℕ) (interval : ℕ), 
    excluded = 3 ∧ 
    interval = 11 ∧ 
    (population_size - excluded) % sample_size = 0 ∧
    (population_size - excluded) / sample_size = interval :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_method_l2539_253939


namespace NUMINAMATH_CALUDE_expression_factorization_l2539_253971

theorem expression_factorization (x : ℝ) : 
  (12 * x^5 + 33 * x^3 + 10) - (3 * x^5 - 4 * x^3 - 1) = x^3 * (9 * x^2 + 37) + 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2539_253971


namespace NUMINAMATH_CALUDE_chord_length_l2539_253979

/-- The circle passing through the intersection points of y = x, y = 2x, and y = 15 - 0.5x -/
def special_circle : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ (x - 5)^2 + (y - 5)^2 = 50}

/-- The line x + y = 16 -/
def line : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x + y = 16}

/-- The chord formed by the intersection of the special circle and the line -/
def chord : Set (ℝ × ℝ) :=
  special_circle ∩ line

theorem chord_length : 
  ∃ (p q : ℝ × ℝ), p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2539_253979


namespace NUMINAMATH_CALUDE_single_tool_users_count_l2539_253900

/-- The number of attendants who used a pencil -/
def pencil_users : ℕ := 25

/-- The number of attendants who used a pen -/
def pen_users : ℕ := 15

/-- The number of attendants who used both pencil and pen -/
def both_users : ℕ := 10

/-- The number of attendants who used only one type of writing tool -/
def single_tool_users : ℕ := (pencil_users - both_users) + (pen_users - both_users)

theorem single_tool_users_count : single_tool_users = 20 := by
  sorry

end NUMINAMATH_CALUDE_single_tool_users_count_l2539_253900


namespace NUMINAMATH_CALUDE_harold_catch_up_distance_l2539_253974

/-- The distance from X to Y in miles -/
def total_distance : ℝ := 60

/-- Adrienne's walking speed in miles per hour -/
def adrienne_speed : ℝ := 3

/-- Harold's walking speed in miles per hour -/
def harold_speed : ℝ := adrienne_speed + 1

/-- Time difference between Adrienne's and Harold's start in hours -/
def time_difference : ℝ := 1

/-- The distance Harold will have traveled when he catches up to Adrienne -/
def catch_up_distance : ℝ := 12

theorem harold_catch_up_distance :
  ∃ (t : ℝ), t > 0 ∧ 
  adrienne_speed * (t + time_difference) = harold_speed * t ∧
  catch_up_distance = harold_speed * t :=
by sorry

end NUMINAMATH_CALUDE_harold_catch_up_distance_l2539_253974


namespace NUMINAMATH_CALUDE_square_difference_1001_999_l2539_253915

theorem square_difference_1001_999 : 1001^2 - 999^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_1001_999_l2539_253915


namespace NUMINAMATH_CALUDE_race_result_l2539_253932

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ  -- Speed in meters per second
  time : ℝ   -- Time to complete the race in seconds

/-- The race scenario -/
def Race : Prop :=
  ∃ (A B : Runner),
    -- Total race distance is 200 meters
    A.speed * A.time = 200 ∧
    -- A's time is 33 seconds
    A.time = 33 ∧
    -- A is 35 meters ahead of B at the finish line
    A.speed * A.time - B.speed * A.time = 35 ∧
    -- B's total race time
    B.time * B.speed = 200 ∧
    -- A beats B by 7 seconds
    B.time - A.time = 7

/-- Theorem stating that given the race conditions, A beats B by 7 seconds -/
theorem race_result : Race := by sorry

end NUMINAMATH_CALUDE_race_result_l2539_253932


namespace NUMINAMATH_CALUDE_lisa_marbles_theorem_distribution_satisfies_conditions_l2539_253949

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 3)) / 2
  max (total_needed - initial_marbles) 0

/-- Proof that 40 additional marbles are needed for Lisa's scenario -/
theorem lisa_marbles_theorem :
  additional_marbles_needed 12 50 = 40 := by
  sorry

/-- Verify that the distribution satisfies the conditions -/
theorem distribution_satisfies_conditions 
  (num_friends : ℕ) 
  (initial_marbles : ℕ) 
  (h : num_friends > 0) :
  let additional := additional_marbles_needed num_friends initial_marbles
  let total := initial_marbles + additional
  (∀ i : ℕ, i > 0 ∧ i ≤ num_friends → i + 1 ≤ total / num_friends) ∧ 
  (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ i ≤ num_friends ∧ j ≤ num_friends ∧ i ≠ j → i + 1 ≠ j + 1) := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_theorem_distribution_satisfies_conditions_l2539_253949


namespace NUMINAMATH_CALUDE_max_value_condition_l2539_253954

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem max_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (a + 2), f x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 (a + 2), f x = 3) ↔ -2 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_condition_l2539_253954


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2539_253920

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {2,5,8}
def B : Set Nat := {1,3,5,7}

theorem complement_intersection_theorem : 
  (U \ A) ∩ B = {1,3,7} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2539_253920


namespace NUMINAMATH_CALUDE_nine_chapters_equal_distribution_l2539_253923

theorem nine_chapters_equal_distribution :
  ∀ (a : ℚ) (d : ℚ),
    (5 * a + 10 * d = 5) →  -- Sum of 5 terms is 5
    (2 * a + d = 3 * a + 9 * d) →  -- Sum of first two terms equals sum of last three terms
    a = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nine_chapters_equal_distribution_l2539_253923


namespace NUMINAMATH_CALUDE_winston_gas_refill_l2539_253938

/-- Calculates the amount of gas needed to refill a car's tank -/
def gas_needed_to_refill (initial_gas tank_capacity gas_used_store gas_used_doctor : ℚ) : ℚ :=
  tank_capacity - (initial_gas - gas_used_store - gas_used_doctor)

/-- Proves that given the initial conditions, the amount of gas needed to refill the tank is 10 gallons -/
theorem winston_gas_refill :
  let initial_gas : ℚ := 10
  let tank_capacity : ℚ := 12
  let gas_used_store : ℚ := 6
  let gas_used_doctor : ℚ := 2
  gas_needed_to_refill initial_gas tank_capacity gas_used_store gas_used_doctor = 10 := by
  sorry


end NUMINAMATH_CALUDE_winston_gas_refill_l2539_253938


namespace NUMINAMATH_CALUDE_s_bounds_l2539_253935

theorem s_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  let s := Real.sqrt (a * b / ((b + c) * (c + a))) +
           Real.sqrt (b * c / ((c + a) * (a + b))) +
           Real.sqrt (c * a / ((a + b) * (b + c)))
  1 ≤ s ∧ s ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_s_bounds_l2539_253935


namespace NUMINAMATH_CALUDE_equation_is_ellipse_l2539_253904

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0

-- Define what it means for the equation to represent an ellipse
def is_ellipse (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b h k : ℝ) (A B : ℝ), 
    A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), eq x y ↔ ((x - h)^2 / A + (y - k)^2 / B = 1)

-- Theorem statement
theorem equation_is_ellipse : is_ellipse equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_ellipse_l2539_253904


namespace NUMINAMATH_CALUDE_find_divisor_l2539_253934

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 163 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 163 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2539_253934


namespace NUMINAMATH_CALUDE_triangle_theorem_l2539_253959

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = π / 3 ∧ t.a = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2539_253959


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2539_253942

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + (3^a - 3) * y = 0 → 
    ∃ k : ℝ, y = (-3 / (3^a - 3)) * x + k) →
  (∀ x y : ℝ, 2 * x - y - 3 = 0 → 
    ∃ k : ℝ, y = 2 * x + k) →
  (∀ m₁ m₂ : ℝ, m₁ * m₂ = -1 → 
    m₁ = -3 / (3^a - 3) ∧ m₂ = 2) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2539_253942


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l2539_253944

/-- A regular pentadecagon is a 15-sided regular polygon -/
def regular_pentadecagon : ℕ := 15

/-- The number of vertices to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Proposition: The number of triangles formed by the vertices of a regular pentadecagon is 455 -/
theorem pentadecagon_triangles :
  (regular_pentadecagon.choose triangle_vertices) = 455 :=
sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l2539_253944


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2539_253910

theorem quadratic_equation_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + 4*x₁ - 4 = 0) ∧ (x₂^2 + 4*x₂ - 4 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2539_253910


namespace NUMINAMATH_CALUDE_finite_solutions_equation_l2539_253931

theorem finite_solutions_equation :
  ∃ (S : Finset (ℕ × ℕ)), ∀ m n : ℕ,
    m^2 + 2 * 3^n = m * (2^(n+1) - 1) → (m, n) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_equation_l2539_253931


namespace NUMINAMATH_CALUDE_complement_A_inter_B_wrt_U_l2539_253987

def U : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}
def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {x | -1 < x ∧ x < 2}

theorem complement_A_inter_B_wrt_U : (U \ (A ∩ B)) = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_wrt_U_l2539_253987


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l2539_253916

def f (m n : ℕ) : ℕ := m * n

theorem f_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := by
sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l2539_253916
