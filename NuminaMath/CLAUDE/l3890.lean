import Mathlib

namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l3890_389050

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ ∃ (n : ℕ), 4 * b + 5 = n^2 ∧ 
  ∀ (x : ℕ), x > 4 ∧ x < b → ¬∃ (m : ℕ), 4 * x + 5 = m^2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l3890_389050


namespace NUMINAMATH_CALUDE_existence_of_cube_sum_equal_100_power_100_l3890_389049

theorem existence_of_cube_sum_equal_100_power_100 : 
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = 100^100 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_cube_sum_equal_100_power_100_l3890_389049


namespace NUMINAMATH_CALUDE_triangle_area_lower_bound_no_triangle_large_altitudes_small_area_l3890_389087

/-- A triangle with two altitudes greater than 100 has an area greater than 1 -/
theorem triangle_area_lower_bound (a b c h1 h2 : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) (halt : h1 > 100 ∧ h2 > 100) : 
  (1/2) * a * h1 > 1 := by
  sorry

/-- There does not exist a triangle with two altitudes greater than 100 and area less than 1 -/
theorem no_triangle_large_altitudes_small_area : 
  ¬ ∃ (a b c h1 h2 : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧ 
  h1 > 100 ∧ h2 > 100 ∧ 
  (1/2) * a * h1 < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_lower_bound_no_triangle_large_altitudes_small_area_l3890_389087


namespace NUMINAMATH_CALUDE_johns_weight_bench_safety_percentage_l3890_389029

/-- Proves that the percentage under the maximum weight that John wants to stay is 20% -/
theorem johns_weight_bench_safety_percentage 
  (bench_max_capacity : ℝ) 
  (johns_weight : ℝ) 
  (bar_weight : ℝ) 
  (h1 : bench_max_capacity = 1000) 
  (h2 : johns_weight = 250) 
  (h3 : bar_weight = 550) : 
  100 - (johns_weight + bar_weight) / bench_max_capacity * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_johns_weight_bench_safety_percentage_l3890_389029


namespace NUMINAMATH_CALUDE_sum_greater_than_product_iff_one_l3890_389022

theorem sum_greater_than_product_iff_one (m n : ℕ+) :
  m + n > m * n ↔ m = 1 ∨ n = 1 := by sorry

end NUMINAMATH_CALUDE_sum_greater_than_product_iff_one_l3890_389022


namespace NUMINAMATH_CALUDE_inequality_solution_l3890_389085

theorem inequality_solution (x : ℝ) (hx : x ≠ 0) (hx2 : x^2 ≠ 6) :
  |4 * x^2 - 32 / x| + |x^2 + 5 / (x^2 - 6)| ≤ |3 * x^2 - 5 / (x^2 - 6) - 32 / x| ↔
  (x > -Real.sqrt 6 ∧ x ≤ -Real.sqrt 5) ∨
  (x ≥ -1 ∧ x < 0) ∨
  (x ≥ 1 ∧ x ≤ 2) ∨
  (x ≥ Real.sqrt 5 ∧ x < Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3890_389085


namespace NUMINAMATH_CALUDE_exactly_two_primes_probability_l3890_389093

-- Define a 12-sided die
def Die := Fin 12

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define the probability of rolling a prime number on a single die
def probPrime : ℚ := 5 / 12

-- Define the probability of not rolling a prime number on a single die
def probNotPrime : ℚ := 7 / 12

-- Define the number of dice
def numDice : ℕ := 3

-- Define the number of dice that should show prime numbers
def numPrimeDice : ℕ := 2

-- Theorem statement
theorem exactly_two_primes_probability :
  (numDice.choose numPrimeDice : ℚ) * probPrime ^ numPrimeDice * probNotPrime ^ (numDice - numPrimeDice) = 525 / 1728 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_primes_probability_l3890_389093


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_55_l3890_389067

theorem triangle_perimeter_not_55 (a b x : ℝ) : 
  a = 18 → b = 10 → 
  (a + b > x ∧ a + x > b ∧ b + x > a) → 
  a + b + x ≠ 55 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_55_l3890_389067


namespace NUMINAMATH_CALUDE_vector_projection_and_collinearity_l3890_389018

def a : Fin 3 → ℚ := ![2, 2, -1]
def b : Fin 3 → ℚ := ![-1, 4, 3]
def p : Fin 3 → ℚ := ![40/29, 64/29, 17/29]

theorem vector_projection_and_collinearity :
  (∀ i : Fin 3, (a i - p i) • (b - a) = 0) ∧
  (∀ i : Fin 3, (b i - p i) • (b - a) = 0) ∧
  ∃ t : ℚ, ∀ i : Fin 3, p i = a i + t * (b i - a i) := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_and_collinearity_l3890_389018


namespace NUMINAMATH_CALUDE_profit_increase_l3890_389072

theorem profit_increase (initial_profit : ℝ) (march_to_april : ℝ) : 
  (initial_profit * (1 + march_to_april / 100) * 0.8 * 1.5 = initial_profit * 1.5600000000000001) →
  march_to_april = 30 :=
by sorry

end NUMINAMATH_CALUDE_profit_increase_l3890_389072


namespace NUMINAMATH_CALUDE_no_thirty_degree_angle_l3890_389048

structure Cube where
  vertices : Finset (Fin 8)

def skew_lines (c : Cube) (p1 p2 p3 p4 : Fin 8) : Prop :=
  p1 ∈ c.vertices ∧ p2 ∈ c.vertices ∧ p3 ∈ c.vertices ∧ p4 ∈ c.vertices ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

def angle_between_lines (l1 l2 : Fin 8 × Fin 8) : ℝ :=
  sorry -- Definition of angle calculation between two lines in a cube

theorem no_thirty_degree_angle (c : Cube) :
  ∀ (p1 p2 p3 p4 : Fin 8),
    skew_lines c p1 p2 p3 p4 →
    angle_between_lines (p1, p2) (p3, p4) ≠ 30 :=
sorry

end NUMINAMATH_CALUDE_no_thirty_degree_angle_l3890_389048


namespace NUMINAMATH_CALUDE_fundraiser_proof_l3890_389057

def fundraiser (total_promised : ℕ) (amount_received : ℕ) (amy_owes : ℕ) : Prop :=
  let total_owed : ℕ := total_promised - amount_received
  let derek_owes : ℕ := amy_owes / 2
  let sally_carl_owe : ℕ := total_owed - (amy_owes + derek_owes)
  sally_carl_owe / 2 = 35

theorem fundraiser_proof : fundraiser 400 285 30 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_proof_l3890_389057


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_n_between_3_and_5_l3890_389041

/-- A simple polygon in 2D space -/
structure SimplePolygon where
  vertices : List (ℝ × ℝ)
  is_simple : Bool  -- Assume this is true for a simple polygon
  is_counterclockwise : Bool -- Assume this is true for counterclockwise orientation

/-- Represents a half-plane in 2D space -/
structure HalfPlane where
  normal : ℝ × ℝ
  offset : ℝ

/-- Function to get the positive half-planes of a simple polygon -/
def getPositiveHalfPlanes (p : SimplePolygon) : List HalfPlane :=
  sorry  -- Implementation details omitted

/-- Function to check if the intersection of half-planes is non-empty -/
def isIntersectionNonEmpty (planes : List HalfPlane) : Bool :=
  sorry  -- Implementation details omitted

/-- The main theorem -/
theorem intersection_nonempty_iff_n_between_3_and_5 (n : ℕ) :
  (∀ p : SimplePolygon, p.vertices.length = n →
    isIntersectionNonEmpty (getPositiveHalfPlanes p)) ↔ (3 ≤ n ∧ n ≤ 5) :=
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_n_between_3_and_5_l3890_389041


namespace NUMINAMATH_CALUDE_cubic_factorization_l3890_389083

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3890_389083


namespace NUMINAMATH_CALUDE_statue_cost_proof_l3890_389039

theorem statue_cost_proof (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) : 
  selling_price = 670 ∧ 
  profit_percentage = 0.25 ∧ 
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 536 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_proof_l3890_389039


namespace NUMINAMATH_CALUDE_ribbon_length_difference_l3890_389074

/-- The theorem about ribbon lengths difference after cutting and giving -/
theorem ribbon_length_difference 
  (initial_difference : Real) 
  (cut_length : Real) : 
  initial_difference = 8.8 → 
  cut_length = 4.3 → 
  initial_difference + 2 * cut_length = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_l3890_389074


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l3890_389058

theorem sales_tax_percentage 
  (total_bill : ℝ)
  (food_price : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_bill = 158.40)
  (h2 : food_price = 120)
  (h3 : tip_percentage = 0.20)
  : ∃ (tax_percentage : ℝ), 
    tax_percentage = 0.10 ∧ 
    total_bill = (food_price * (1 + tax_percentage) * (1 + tip_percentage)) :=
by sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l3890_389058


namespace NUMINAMATH_CALUDE_alan_bought_20_eggs_l3890_389019

/-- The number of eggs Alan bought at the market -/
def eggs : ℕ := sorry

/-- The price of each egg in dollars -/
def egg_price : ℕ := 2

/-- The number of chickens Alan bought -/
def chickens : ℕ := 6

/-- The price of each chicken in dollars -/
def chicken_price : ℕ := 8

/-- The total amount Alan spent at the market in dollars -/
def total_spent : ℕ := 88

/-- Theorem stating that Alan bought 20 eggs -/
theorem alan_bought_20_eggs : eggs = 20 := by
  sorry

end NUMINAMATH_CALUDE_alan_bought_20_eggs_l3890_389019


namespace NUMINAMATH_CALUDE_perpendicular_equivalence_l3890_389066

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_equivalence 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) 
  (h_n_perp_α : perp n α) 
  (h_n_perp_β : perp n β) :
  perp m α ↔ perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_equivalence_l3890_389066


namespace NUMINAMATH_CALUDE_intersection_length_l3890_389061

-- Define the line l passing through A(0,1) with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the circle C: (x-2)^2 + (y-3)^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the condition that line l intersects circle C at points M and N
def intersects (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 12

-- Main theorem
theorem intersection_length (k : ℝ) :
  intersects k →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    dot_product_condition x₁ y₁ x₂ y₂) →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_l3890_389061


namespace NUMINAMATH_CALUDE_rectangle_height_calculation_l3890_389033

/-- Represents a rectangle with a base and height in centimeters -/
structure Rectangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.base * r.height

theorem rectangle_height_calculation (r : Rectangle) 
  (h_base : r.base = 9)
  (h_area : area r = 33.3) :
  r.height = 3.7 := by
sorry


end NUMINAMATH_CALUDE_rectangle_height_calculation_l3890_389033


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_bound_l3890_389080

/-- Given a quadrilateral with sides a, b, c, and d, the surface area of a rectangular prism
    with edges a, b, and c meeting at a vertex is at most (a+b+c)^2 - d^2/3 -/
theorem rectangular_prism_surface_area_bound 
  (a b c d : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (quad : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c) :
  2 * (a * b + b * c + c * a) ≤ (a + b + c)^2 - d^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_bound_l3890_389080


namespace NUMINAMATH_CALUDE_vacation_cost_problem_l3890_389040

/-- The vacation cost problem -/
theorem vacation_cost_problem (sarah_paid derek_paid rita_paid : ℚ)
  (h_sarah : sarah_paid = 150)
  (h_derek : derek_paid = 210)
  (h_rita : rita_paid = 240)
  (s d : ℚ) :
  let total_paid := sarah_paid + derek_paid + rita_paid
  let equal_share := total_paid / 3
  let sarah_owes := equal_share - sarah_paid
  let derek_owes := equal_share - derek_paid
  s = sarah_owes ∧ d = derek_owes →
  s - d = 60 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_problem_l3890_389040


namespace NUMINAMATH_CALUDE_lunch_meeting_probability_l3890_389094

/-- The probability of Janet and Donald meeting for lunch -/
theorem lunch_meeting_probability :
  let arrival_interval : ℝ := 60
  let janet_wait_time : ℝ := 15
  let donald_wait_time : ℝ := 5
  let meeting_condition (x y : ℝ) : Prop := |x - y| ≤ min donald_wait_time janet_wait_time
  let total_area : ℝ := arrival_interval ^ 2
  let meeting_area : ℝ := arrival_interval * (2 * min donald_wait_time janet_wait_time)
  (meeting_area / total_area : ℝ) = 1/6 := by
sorry

end NUMINAMATH_CALUDE_lunch_meeting_probability_l3890_389094


namespace NUMINAMATH_CALUDE_allens_blocks_combinations_l3890_389081

/-- Given conditions for Allen's blocks problem -/
structure BlocksProblem where
  total_blocks : ℕ
  num_shapes : ℕ
  blocks_per_color : ℕ

/-- Calculate the number of color and shape combinations -/
def calculate_combinations (problem : BlocksProblem) : ℕ :=
  let num_colors := problem.total_blocks / problem.blocks_per_color
  problem.num_shapes * num_colors

/-- Theorem: The number of color and shape combinations is 80 -/
theorem allens_blocks_combinations (problem : BlocksProblem) 
  (h1 : problem.total_blocks = 100)
  (h2 : problem.num_shapes = 4)
  (h3 : problem.blocks_per_color = 5) :
  calculate_combinations problem = 80 := by
  sorry

#eval calculate_combinations ⟨100, 4, 5⟩

end NUMINAMATH_CALUDE_allens_blocks_combinations_l3890_389081


namespace NUMINAMATH_CALUDE_quiz_score_difference_l3890_389088

def quiz_scores : List (Float × Float) := [
  (0.05, 65),
  (0.25, 75),
  (0.40, 85),
  (0.20, 95),
  (0.10, 105)
]

def mean (scores : List (Float × Float)) : Float :=
  (scores.map (λ (p, s) => p * s)).sum

def median (scores : List (Float × Float)) : Float :=
  if (scores.map (λ (p, _) => p)).sum ≥ 0.5 then
    scores.filter (λ (_, s) => s ≥ 85)
      |> List.head!
      |> (λ (_, s) => s)
  else 85

theorem quiz_score_difference :
  median quiz_scores - mean quiz_scores = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_difference_l3890_389088


namespace NUMINAMATH_CALUDE_max_value_inequality_l3890_389097

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (M : ℝ), M = 1/2 ∧ 
  (∀ (N : ℝ), (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    x^3 + y^3 + z^3 - 3*x*y*z ≥ N*(|x-y|^3 + |x-z|^3 + |z-y|^3)) → N ≤ M) ∧
  (a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(|a-b|^3 + |a-c|^3 + |c-b|^3)) := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3890_389097


namespace NUMINAMATH_CALUDE_optimal_seating_arrangement_l3890_389020

/-- Represents the seating arrangement for children based on their heights. -/
structure SeatingArrangement where
  x : ℕ  -- Number of seats with three children
  y : ℕ  -- Number of seats with two children
  total_seats : ℕ
  group_a : ℕ  -- Children below 4 feet
  group_b : ℕ  -- Children between 4 and 4.5 feet
  group_c : ℕ  -- Children above 4.5 feet

/-- The seating arrangement satisfies all constraints. -/
def valid_arrangement (s : SeatingArrangement) : Prop :=
  s.x + s.y = s.total_seats ∧
  s.x ≤ s.group_a ∧
  2 * s.x + s.y ≤ s.group_b ∧
  s.y ≤ s.group_c

/-- The optimal seating arrangement exists and is unique. -/
theorem optimal_seating_arrangement :
  ∃! s : SeatingArrangement,
    s.total_seats = 7 ∧
    s.group_a = 5 ∧
    s.group_b = 8 ∧
    s.group_c = 6 ∧
    valid_arrangement s ∧
    s.x = 1 ∧
    s.y = 6 := by
  sorry


end NUMINAMATH_CALUDE_optimal_seating_arrangement_l3890_389020


namespace NUMINAMATH_CALUDE_ice_cream_volume_l3890_389042

/-- The volume of ice cream in a cone with hemisphere and cylindrical layer -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ) : 
  h_cone = 10 ∧ r = 3 ∧ h_cylinder = 2 →
  (1/3 * π * r^2 * h_cone) + (2/3 * π * r^3) + (π * r^2 * h_cylinder) = 66 * π := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l3890_389042


namespace NUMINAMATH_CALUDE_sugar_to_cream_cheese_ratio_l3890_389004

/-- Represents the ingredients and ratios in Betty's cheesecake recipe -/
structure CheesecakeRecipe where
  sugar : ℕ
  cream_cheese : ℕ
  vanilla : ℕ
  eggs : ℕ
  vanilla_to_cream_cheese_ratio : vanilla * 2 = cream_cheese
  eggs_to_vanilla_ratio : eggs = vanilla * 2
  sugar_used : sugar = 2
  eggs_used : eggs = 8

/-- The ratio of sugar to cream cheese in Betty's cheesecake is 1:4 -/
theorem sugar_to_cream_cheese_ratio (recipe : CheesecakeRecipe) : 
  recipe.sugar * 4 = recipe.cream_cheese := by
  sorry

#check sugar_to_cream_cheese_ratio

end NUMINAMATH_CALUDE_sugar_to_cream_cheese_ratio_l3890_389004


namespace NUMINAMATH_CALUDE_pyramid_face_area_l3890_389035

-- Define the pyramid
structure SquareBasedPyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

-- Define the problem
theorem pyramid_face_area (p : SquareBasedPyramid) 
  (h_base : p.baseEdge = 8)
  (h_lateral : p.lateralEdge = 7) : 
  Real.sqrt ((4 * p.baseEdge * Real.sqrt (p.lateralEdge ^ 2 - (p.baseEdge / 2) ^ 2)) ^ 2) = 16 * Real.sqrt 33 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_face_area_l3890_389035


namespace NUMINAMATH_CALUDE_sum_of_digits_of_1962_digit_number_div_by_9_l3890_389052

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 1962 digits -/
def has1962Digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1962_digit_number_div_by_9 (n : ℕ) 
  (h1 : has1962Digits n) 
  (h2 : n % 9 = 0) : 
  let a := sumOfDigits n
  let b := sumOfDigits a
  let c := sumOfDigits b
  c = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_1962_digit_number_div_by_9_l3890_389052


namespace NUMINAMATH_CALUDE_min_concerts_required_l3890_389045

/-- Represents a concert where some musicians play and others listen -/
structure Concert where
  players : Finset (Fin 6)

/-- Checks if a set of concerts satisfies the condition that 
    for every pair of musicians, each plays for the other in some concert -/
def satisfies_condition (concerts : List Concert) : Prop :=
  ∀ i j, i ≠ j → 
    (∃ c ∈ concerts, i ∈ c.players ∧ j ∉ c.players) ∧
    (∃ c ∈ concerts, j ∈ c.players ∧ i ∉ c.players)

/-- The main theorem: the minimum number of concerts required is 4 -/
theorem min_concerts_required : 
  (∃ concerts : List Concert, concerts.length = 4 ∧ satisfies_condition concerts) ∧
  (∀ concerts : List Concert, concerts.length < 4 → ¬satisfies_condition concerts) :=
sorry

end NUMINAMATH_CALUDE_min_concerts_required_l3890_389045


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3890_389095

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  Real.sqrt (a + 1) + Real.sqrt (b + 3) ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3890_389095


namespace NUMINAMATH_CALUDE_real_roots_quadratic_l3890_389028

theorem real_roots_quadratic (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ k ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_l3890_389028


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l3890_389079

theorem students_in_both_clubs 
  (total_students : ℕ) 
  (drama_club : ℕ) 
  (science_club : ℕ) 
  (either_club : ℕ) 
  (h1 : total_students = 300)
  (h2 : drama_club = 120)
  (h3 : science_club = 180)
  (h4 : either_club = 250) :
  drama_club + science_club - either_club = 50 := by
  sorry

#check students_in_both_clubs

end NUMINAMATH_CALUDE_students_in_both_clubs_l3890_389079


namespace NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3890_389065

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sum_odd_integers (start : ℕ) (n : ℕ) : ℕ :=
  let last := start + 2 * (n - 1)
  n * (start + last) / 2

/-- Theorem: The sum of the first 15 odd positive integers starting from 5 is 255 -/
theorem sum_first_15_odd_from_5 : sum_odd_integers 5 15 = 255 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3890_389065


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3890_389073

/-- Represents the price reduction in yuan -/
def price_reduction : ℕ := 10

/-- Cost to purchase each piece of clothing -/
def purchase_cost : ℕ := 45

/-- Original selling price of each piece of clothing -/
def original_price : ℕ := 65

/-- Original daily sales quantity -/
def original_sales : ℕ := 30

/-- Additional sales for each yuan of price reduction -/
def sales_increase_rate : ℕ := 5

/-- Target daily profit -/
def target_profit : ℕ := 800

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (original_price - price_reduction - purchase_cost) *
  (original_sales + sales_increase_rate * price_reduction) = target_profit :=
sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3890_389073


namespace NUMINAMATH_CALUDE_sanchez_rope_theorem_l3890_389063

def rope_problem (rope_last_week : ℕ) (rope_difference : ℕ) (inches_per_foot : ℕ) : Prop :=
  let rope_this_week : ℕ := rope_last_week - rope_difference
  let total_rope_feet : ℕ := rope_last_week + rope_this_week
  let total_rope_inches : ℕ := total_rope_feet * inches_per_foot
  total_rope_inches = 96

theorem sanchez_rope_theorem : rope_problem 6 4 12 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_rope_theorem_l3890_389063


namespace NUMINAMATH_CALUDE_count_cows_l3890_389012

def group_of_animals (ducks cows : ℕ) : Prop :=
  2 * ducks + 4 * cows = 22 + 2 * (ducks + cows)

theorem count_cows : ∃ ducks : ℕ, group_of_animals ducks 11 :=
sorry

end NUMINAMATH_CALUDE_count_cows_l3890_389012


namespace NUMINAMATH_CALUDE_linear_function_shift_l3890_389055

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shifts a linear function horizontally -/
def shiftHorizontal (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept + f.slope * units }

/-- Shifts a linear function vertically -/
def shiftVertical (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - units }

/-- The theorem to be proved -/
theorem linear_function_shift :
  let f := LinearFunction.mk 3 2
  let f_shifted_left := shiftHorizontal f 3
  let f_final := shiftVertical f_shifted_left 1
  f_final = LinearFunction.mk 3 10 := by sorry

end NUMINAMATH_CALUDE_linear_function_shift_l3890_389055


namespace NUMINAMATH_CALUDE_trig_identity_l3890_389026

theorem trig_identity (α β : ℝ) : 
  Real.sin (2 * α) ^ 2 + Real.sin β ^ 2 + Real.cos (2 * α + β) * Real.cos (2 * α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3890_389026


namespace NUMINAMATH_CALUDE_no_infinite_line_family_l3890_389015

theorem no_infinite_line_family :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k n ≠ 0) ∧ 
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧
    (∀ n, k n * k (n + 1) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_line_family_l3890_389015


namespace NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l3890_389007

theorem min_n_for_60n_divisible_by_4_and_8 : 
  ∃ (n : ℕ), n > 0 ∧ 
    (∀ (m : ℕ), m > 0 → (4 ∣ 60 * m) ∧ (8 ∣ 60 * m) → n ≤ m) ∧
    (4 ∣ 60 * n) ∧ (8 ∣ 60 * n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_min_n_for_60n_divisible_by_4_and_8_l3890_389007


namespace NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l3890_389009

theorem cube_root_of_four_sixth_powers (x : ℝ) :
  x = (4^6 + 4^6 + 4^6 + 4^6)^(1/3) → x = 16 * (4^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_four_sixth_powers_l3890_389009


namespace NUMINAMATH_CALUDE_total_birds_caught_l3890_389011

def birds_caught_day : ℕ := 8

def birds_caught_night (day : ℕ) : ℕ := 2 * day

theorem total_birds_caught :
  birds_caught_day + birds_caught_night birds_caught_day = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_birds_caught_l3890_389011


namespace NUMINAMATH_CALUDE_dot_product_properties_l3890_389008

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem dot_product_properties 
  (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 10)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 12)
  (h3 : angle_between a b = 2 * π / 3) : 
  (a.1 * b.1 + a.2 * b.2 = -60) ∧ 
  (3 * a.1 * (1/5 * b.1) + 3 * a.2 * (1/5 * b.2) = -36) ∧
  ((3 * b.1 - 2 * a.1) * (4 * a.1 + b.1) + (3 * b.2 - 2 * a.2) * (4 * a.2 + b.2) = -968) := by
  sorry

end NUMINAMATH_CALUDE_dot_product_properties_l3890_389008


namespace NUMINAMATH_CALUDE_sandwich_menu_count_l3890_389010

theorem sandwich_menu_count (initial_count sold_out remaining : ℕ) : 
  sold_out = 5 → remaining = 4 → initial_count = sold_out + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_menu_count_l3890_389010


namespace NUMINAMATH_CALUDE_jewelry_restock_cost_l3890_389030

/-- Represents the inventory and pricing information for a jewelry item -/
structure JewelryItem where
  name : String
  capacity : Nat
  current : Nat
  price : Nat
  discount1 : Nat
  discount1Threshold : Nat
  discount2 : Nat
  discount2Threshold : Nat

/-- Calculates the total cost for restocking jewelry items -/
def calculateTotalCost (items : List JewelryItem) : Rat :=
  let itemCosts := items.map (fun item =>
    let quantity := item.capacity - item.current
    let basePrice := quantity * item.price
    let discountedPrice :=
      if quantity >= item.discount2Threshold then
        basePrice * (1 - item.discount2 / 100)
      else if quantity >= item.discount1Threshold then
        basePrice * (1 - item.discount1 / 100)
      else
        basePrice
    discountedPrice)
  let subtotal := itemCosts.sum
  let shippingFee := subtotal * (2 / 100)
  subtotal + shippingFee

/-- Theorem stating that the total cost to restock the jewelry showroom is $257.04 -/
theorem jewelry_restock_cost :
  let necklaces : JewelryItem := ⟨"Necklace", 20, 8, 5, 10, 10, 15, 15⟩
  let rings : JewelryItem := ⟨"Ring", 40, 25, 8, 5, 20, 12, 30⟩
  let bangles : JewelryItem := ⟨"Bangle", 30, 17, 6, 8, 15, 18, 25⟩
  calculateTotalCost [necklaces, rings, bangles] = 257.04 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_restock_cost_l3890_389030


namespace NUMINAMATH_CALUDE_smallest_prime_after_eight_nonprimes_l3890_389075

def is_first_prime_after_eight_nonprimes (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, n ≤ k ∧ k < n + 8 → ¬Nat.Prime k) ∧
    (∀ q : ℕ, Nat.Prime q → q < p → q ≤ n - 1 ∨ q ≥ n + 8)

theorem smallest_prime_after_eight_nonprimes :
  is_first_prime_after_eight_nonprimes 59 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_eight_nonprimes_l3890_389075


namespace NUMINAMATH_CALUDE_multiple_z_values_l3890_389062

/-- Given two four-digit integers x and y where y is the reverse of x, 
    z = |x - y| can take multiple distinct values. -/
theorem multiple_z_values (x y z : ℕ) : 
  (1000 ≤ x ∧ x ≤ 9999) →
  (1000 ≤ y ∧ y ≤ 9999) →
  (y = (x % 10) * 1000 + ((x / 10) % 10) * 100 + ((x / 100) % 10) * 10 + (x / 1000)) →
  (z = Int.natAbs (x - y)) →
  ∃ (z₁ z₂ : ℕ), z₁ ≠ z₂ ∧ 
    ∃ (x₁ y₁ x₂ y₂ : ℕ), 
      (1000 ≤ x₁ ∧ x₁ ≤ 9999) ∧
      (1000 ≤ y₁ ∧ y₁ ≤ 9999) ∧
      (y₁ = (x₁ % 10) * 1000 + ((x₁ / 10) % 10) * 100 + ((x₁ / 100) % 10) * 10 + (x₁ / 1000)) ∧
      (z₁ = Int.natAbs (x₁ - y₁)) ∧
      (1000 ≤ x₂ ∧ x₂ ≤ 9999) ∧
      (1000 ≤ y₂ ∧ y₂ ≤ 9999) ∧
      (y₂ = (x₂ % 10) * 1000 + ((x₂ / 10) % 10) * 100 + ((x₂ / 100) % 10) * 10 + (x₂ / 1000)) ∧
      (z₂ = Int.natAbs (x₂ - y₂)) :=
by
  sorry


end NUMINAMATH_CALUDE_multiple_z_values_l3890_389062


namespace NUMINAMATH_CALUDE_cube_probability_l3890_389084

-- Define the type for cube faces
inductive CubeFace
| Face1 | Face2 | Face3 | Face4 | Face5 | Face6

-- Define the type for numbers
inductive Number
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

-- Define a function to check if two numbers are consecutive
def isConsecutive (n1 n2 : Number) : Prop := sorry

-- Define a function to check if two faces share an edge
def sharesEdge (f1 f2 : CubeFace) : Prop := sorry

-- Define the type for cube configuration
def CubeConfig := CubeFace → Option Number

-- Define a valid cube configuration
def isValidConfig (config : CubeConfig) : Prop :=
  (∀ f1 f2 : CubeFace, f1 ≠ f2 → config f1 ≠ config f2) ∧
  (∀ f1 f2 : CubeFace, sharesEdge f1 f2 →
    ∀ n1 n2 : Number, config f1 = some n1 → config f2 = some n2 →
      ¬isConsecutive n1 n2)

-- Define the total number of possible configurations
def totalConfigs : ℕ := sorry

-- Define the number of valid configurations
def validConfigs : ℕ := sorry

-- The main theorem
theorem cube_probability :
  (validConfigs : ℚ) / totalConfigs = 1 / 672 := by sorry

end NUMINAMATH_CALUDE_cube_probability_l3890_389084


namespace NUMINAMATH_CALUDE_stream_speed_l3890_389071

/-- The speed of the stream given rowing distances and times -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 78) 
  (h2 : upstream_distance = 50) (h3 : downstream_time = 2) (h4 : upstream_time = 2) : 
  ∃ (boat_speed stream_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧ 
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧ 
    stream_speed = 7 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3890_389071


namespace NUMINAMATH_CALUDE_value_range_of_f_l3890_389021

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem value_range_of_f :
  ∀ y ∈ Set.Icc (-3) 5, ∃ x ∈ Set.Icc 0 2, f x = y ∧
  ∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc (-3) 5 :=
by sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3890_389021


namespace NUMINAMATH_CALUDE_sum_of_integers_l3890_389078

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 14) 
  (h2 : x.val * y.val = 180) : 
  x.val + y.val = 2 * Real.sqrt 229 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3890_389078


namespace NUMINAMATH_CALUDE_set_T_is_hexagon_l3890_389090

/-- The set T of points (x, y) satisfying the given conditions forms a hexagon -/
theorem set_T_is_hexagon (b : ℝ) (hb : b > 0) :
  let T : Set (ℝ × ℝ) :=
    {p | b ≤ p.1 ∧ p.1 ≤ 3*b ∧
         b ≤ p.2 ∧ p.2 ≤ 3*b ∧
         p.1 + p.2 ≥ 2*b ∧
         p.1 + 2*b ≥ 2*p.2 ∧
         p.2 + 2*b ≥ 2*p.1}
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 6 ∧
    ∀ p ∈ T, p ∈ convexHull ℝ (vertices : Set (ℝ × ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_set_T_is_hexagon_l3890_389090


namespace NUMINAMATH_CALUDE_joan_balloon_count_l3890_389036

/-- The number of orange balloons Joan has after receiving more from a friend -/
def total_balloons (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Given Joan has 8 orange balloons initially and receives 2 more from a friend,
    she now has 10 orange balloons in total. -/
theorem joan_balloon_count : total_balloons 8 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloon_count_l3890_389036


namespace NUMINAMATH_CALUDE_parabola_vertex_above_x_axis_l3890_389032

/-- A parabola with equation y = x^2 - 3x + k has its vertex above the x-axis if and only if k > 9/4 -/
theorem parabola_vertex_above_x_axis (k : ℝ) : 
  (∃ (x y : ℝ), y = x^2 - 3*x + k ∧ y > 0 ∧ ∀ (x' : ℝ), x'^2 - 3*x' + k ≤ y) ↔ k > 9/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_above_x_axis_l3890_389032


namespace NUMINAMATH_CALUDE_doughnut_boxes_l3890_389000

theorem doughnut_boxes (total_doughnuts : ℕ) (doughnuts_per_box : ℕ) (h1 : total_doughnuts = 48) (h2 : doughnuts_per_box = 12) :
  total_doughnuts / doughnuts_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_boxes_l3890_389000


namespace NUMINAMATH_CALUDE_sum_xy_equals_three_l3890_389059

theorem sum_xy_equals_three (x y : ℝ) (h : Real.sqrt (1 - x) + abs (2 - y) = 0) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_equals_three_l3890_389059


namespace NUMINAMATH_CALUDE_tim_surprise_combinations_l3890_389001

/-- Represents the number of choices for each day of the week --/
structure WeekChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of combinations for Tim's surprise arrangements --/
def totalCombinations (choices : WeekChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Tim's specific choices for each day of the week --/
def timChoices : WeekChoices :=
  { monday := 1
  , tuesday := 2
  , wednesday := 6
  , thursday := 5
  , friday := 2 }

theorem tim_surprise_combinations :
  totalCombinations timChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_tim_surprise_combinations_l3890_389001


namespace NUMINAMATH_CALUDE_usual_price_equals_sale_price_l3890_389014

/-- Represents the laundry detergent scenario -/
structure DetergentScenario where
  loads_per_bottle : ℕ
  sale_price_per_bottle : ℚ
  cost_per_load : ℚ

/-- The usual price of a bottle of detergent is equal to the sale price -/
theorem usual_price_equals_sale_price (scenario : DetergentScenario)
  (h1 : scenario.loads_per_bottle = 80)
  (h2 : scenario.sale_price_per_bottle = 20)
  (h3 : scenario.cost_per_load = 1/4) :
  scenario.sale_price_per_bottle = scenario.loads_per_bottle * scenario.cost_per_load := by
  sorry

#check usual_price_equals_sale_price

end NUMINAMATH_CALUDE_usual_price_equals_sale_price_l3890_389014


namespace NUMINAMATH_CALUDE_quadratic_polynomial_sufficiency_necessity_l3890_389054

/-- A second-degree polynomial with distinct roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  x₁ : ℝ
  x₂ : ℝ
  distinct_roots : x₁ ≠ x₂
  is_root_x₁ : a * x₁^2 + b * x₁ + c = 0
  is_root_x₂ : a * x₂^2 + b * x₂ + c = 0

/-- The value of the polynomial at a given x -/
def QuadraticPolynomial.value (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem quadratic_polynomial_sufficiency_necessity 
    (p : QuadraticPolynomial) : 
    (p.a^2 + 3*p.a*p.c - p.b^2 = 0 → p.value (p.x₁^3) = p.value (p.x₂^3)) ∧
    (∃ p : QuadraticPolynomial, p.value (p.x₁^3) = p.value (p.x₂^3) ∧ p.a^2 + 3*p.a*p.c - p.b^2 ≠ 0) :=
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_sufficiency_necessity_l3890_389054


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3890_389047

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (n - 3) * (n + 5) < 0) ∧ Finset.card S = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3890_389047


namespace NUMINAMATH_CALUDE_min_value_interval_l3890_389005

def f (x : ℝ) := 3 * x - x^3

theorem min_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo (a^2 - 12) a, ∀ y ∈ Set.Ioo (a^2 - 12) a, f y ≥ f x) →
  a ∈ Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_interval_l3890_389005


namespace NUMINAMATH_CALUDE_martin_ice_cream_cost_l3890_389025

/-- Represents the cost of ice cream scoops in dollars -/
structure IceCreamPrices where
  kiddie : ℕ
  regular : ℕ
  double : ℕ

/-- Represents the Martin family's ice cream order -/
structure MartinOrder where
  regular : ℕ
  kiddie : ℕ
  double : ℕ

/-- Calculates the total cost of the Martin family's ice cream order -/
def calculateTotalCost (prices : IceCreamPrices) (order : MartinOrder) : ℕ :=
  prices.regular * order.regular +
  prices.kiddie * order.kiddie +
  prices.double * order.double

/-- Theorem stating that the total cost for the Martin family's ice cream order is $32 -/
theorem martin_ice_cream_cost :
  ∃ (prices : IceCreamPrices) (order : MartinOrder),
    prices.kiddie = 3 ∧
    prices.regular = 4 ∧
    prices.double = 6 ∧
    order.regular = 2 ∧
    order.kiddie = 2 ∧
    order.double = 3 ∧
    calculateTotalCost prices order = 32 :=
  sorry

end NUMINAMATH_CALUDE_martin_ice_cream_cost_l3890_389025


namespace NUMINAMATH_CALUDE_heather_starting_blocks_l3890_389056

/-- The number of blocks Heather shared with Jose -/
def shared_blocks : ℕ := 41

/-- The number of blocks Heather ended up with -/
def remaining_blocks : ℕ := 45

/-- The total number of blocks Heather started with -/
def starting_blocks : ℕ := shared_blocks + remaining_blocks

theorem heather_starting_blocks : starting_blocks = 86 := by
  sorry

end NUMINAMATH_CALUDE_heather_starting_blocks_l3890_389056


namespace NUMINAMATH_CALUDE_bus_purchase_problem_l3890_389016

/-- Represents the cost and capacity of a bus type -/
structure BusType where
  cost : ℕ
  capacity : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

def totalBuses : ℕ := 10

def scenario1Cost : ℕ := 380
def scenario2Cost : ℕ := 360

def maxTotalCost : ℕ := 880
def minTotalPassengers : ℕ := 5200000

theorem bus_purchase_problem 
  (typeA typeB : BusType)
  (plans : List PurchasePlan)
  (bestPlan : PurchasePlan)
  (minCost : ℕ) :
  (typeA.cost + 3 * typeB.cost = scenario1Cost) →
  (2 * typeA.cost + 2 * typeB.cost = scenario2Cost) →
  (typeA.capacity = 500000) →
  (typeB.capacity = 600000) →
  (∀ plan ∈ plans, 
    plan.typeA + plan.typeB = totalBuses ∧
    plan.typeA * typeA.cost + plan.typeB * typeB.cost ≤ maxTotalCost ∧
    plan.typeA * typeA.capacity + plan.typeB * typeB.capacity ≥ minTotalPassengers) →
  (bestPlan ∈ plans) →
  (∀ plan ∈ plans, 
    plan.typeA * typeA.cost + plan.typeB * typeB.cost ≥ 
    bestPlan.typeA * typeA.cost + bestPlan.typeB * typeB.cost) →
  (minCost = bestPlan.typeA * typeA.cost + bestPlan.typeB * typeB.cost) →
  typeA.cost = 80 ∧ 
  typeB.cost = 100 ∧
  plans = [⟨6, 4⟩, ⟨7, 3⟩, ⟨8, 2⟩] ∧
  bestPlan = ⟨8, 2⟩ ∧
  minCost = 840 := by
  sorry

end NUMINAMATH_CALUDE_bus_purchase_problem_l3890_389016


namespace NUMINAMATH_CALUDE_box_weight_sum_l3890_389002

theorem box_weight_sum (a b c : ℝ) 
  (hab : a + b = 132)
  (hbc : b + c = 135)
  (hca : c + a = 137)
  (ha : a > 40)
  (hb : b > 40)
  (hc : c > 40) :
  a + b + c = 202 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_sum_l3890_389002


namespace NUMINAMATH_CALUDE_difference_of_squares_l3890_389068

theorem difference_of_squares (m : ℝ) : m^2 - 9 = (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3890_389068


namespace NUMINAMATH_CALUDE_fraction_equality_l3890_389023

theorem fraction_equality (x y : ℚ) :
  (2/5)^2 + (1/7)^2 = 25*x * ((1/3)^2 + (1/8)^2) / (73*y) →
  Real.sqrt x / Real.sqrt y = 356 / 175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3890_389023


namespace NUMINAMATH_CALUDE_power_sum_of_i_l3890_389098

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i :
  i^15 + i^20 + i^25 + i^30 + i^35 = -i :=
by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l3890_389098


namespace NUMINAMATH_CALUDE_frances_towel_weight_l3890_389003

theorem frances_towel_weight (mary_towels frances_towels : ℕ) (total_weight : ℝ) :
  mary_towels = 24 →
  mary_towels = 4 * frances_towels →
  total_weight = 60 →
  (frances_towels * (total_weight / (mary_towels + frances_towels))) * 16 = 192 :=
by sorry

end NUMINAMATH_CALUDE_frances_towel_weight_l3890_389003


namespace NUMINAMATH_CALUDE_f_min_value_l3890_389037

/-- The function f(x) = |x + 3| + |x + 6| + |x + 8| + |x + 10| -/
def f (x : ℝ) : ℝ := |x + 3| + |x + 6| + |x + 8| + |x + 10|

/-- Theorem stating that f(x) has a minimum value of 9 at x = -8 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 9) ∧ f (-8) = 9 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l3890_389037


namespace NUMINAMATH_CALUDE_three_factor_numbers_product_l3890_389082

theorem three_factor_numbers_product (x y z : ℕ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  (∃ p₁ : ℕ, Prime p₁ ∧ x = p₁^2) →
  (∃ p₂ : ℕ, Prime p₂ ∧ y = p₂^2) →
  (∃ p₃ : ℕ, Prime p₃ ∧ z = p₃^2) →
  (Nat.card {d : ℕ | d ∣ x} = 3) →
  (Nat.card {d : ℕ | d ∣ y} = 3) →
  (Nat.card {d : ℕ | d ∣ z} = 3) →
  Nat.card {d : ℕ | d ∣ (x^2 * y^3 * z^4)} = 315 := by
sorry

end NUMINAMATH_CALUDE_three_factor_numbers_product_l3890_389082


namespace NUMINAMATH_CALUDE_not_always_congruent_l3890_389077

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)

-- Define the property of having two equal sides and three equal angles
def hasTwoEqualSidesThreeEqualAngles (t1 t2 : Triangle) : Prop :=
  ((t1.a = t2.a ∧ t1.b = t2.b) ∨ (t1.a = t2.a ∧ t1.c = t2.c) ∨ (t1.b = t2.b ∧ t1.c = t2.c)) ∧
  (t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ)

-- Define triangle congruence
def isCongruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem not_always_congruent :
  ∃ (t1 t2 : Triangle), hasTwoEqualSidesThreeEqualAngles t1 t2 ∧ ¬isCongruent t1 t2 :=
sorry

end NUMINAMATH_CALUDE_not_always_congruent_l3890_389077


namespace NUMINAMATH_CALUDE_evaluate_expression_l3890_389027

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  4 * x^y + 5 * y^x = 76 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3890_389027


namespace NUMINAMATH_CALUDE_calculation_result_l3890_389092

theorem calculation_result : (786 * 74) / 30 = 1938.8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3890_389092


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3890_389064

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 2*a

theorem quadratic_function_properties (a : ℝ) :
  (f a (-1) = -1 → a = 0) ∧
  (f a 3 = -1 → (∀ x ∈ Set.Icc (-2) 3, f a x ≤ 3) ∧ (∃ x ∈ Set.Icc (-2) 3, f a x = -6)) ∧
  (∃ x y : ℝ, x ≠ y ∧ f a x = -1 ∧ f a y = -1 ∧ |x - y| = |2*a + 2|) ∧
  (∃! x : ℝ, x ∈ Set.Icc (a - 1) (2*a + 3) ∧ f a x = -1 ↔ a ≥ 0 ∨ a = -1 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3890_389064


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3890_389060

theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.cos x
  let f' : ℝ → ℝ := λ x ↦ -Real.sin x
  let tangent_slope : ℝ := f' (π/6)
  tangent_slope * a = -1 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l3890_389060


namespace NUMINAMATH_CALUDE_yi_rong_ferry_distance_l3890_389017

/-- The Yi Rong ferry problem -/
theorem yi_rong_ferry_distance :
  let ferry_speed : ℝ := 40
  let water_speed : ℝ := 24
  let downstream_speed : ℝ := ferry_speed + water_speed
  let upstream_speed : ℝ := ferry_speed - water_speed
  let distance : ℝ := 192  -- The distance we want to prove

  -- Odd day condition
  (distance / downstream_speed * (43 / 18) = 
   distance / 2 / downstream_speed + distance / 2 / water_speed) ∧ 
  
  -- Even day condition
  (distance / upstream_speed = 
   distance / 2 / water_speed + 1 + distance / 2 / (2 * upstream_speed)) →
  
  distance = 192 := by sorry

end NUMINAMATH_CALUDE_yi_rong_ferry_distance_l3890_389017


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l3890_389031

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)

-- Define the subset relation for lines in planes
variable (subset : Line → Plane → Prop)

-- Define specific planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem parallel_planes_condition 
  (h1 : subset a α)
  (h2 : subset b α) :
  (∀ (α β : Plane), parallel α β → lineParallelToPlane a β ∧ lineParallelToPlane b β) ∧ 
  (∃ (α β : Plane) (a b : Line), 
    subset a α ∧ 
    subset b α ∧ 
    lineParallelToPlane a β ∧ 
    lineParallelToPlane b β ∧ 
    ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l3890_389031


namespace NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l3890_389006

/-- Represents a rectangle on a grid --/
structure Rectangle where
  m : ℕ
  n : ℕ

/-- Calculates the maximum number of quartets in a rectangle --/
def max_quartets (rect : Rectangle) : ℕ :=
  if rect.m % 2 = 0 ∧ rect.n % 2 = 1 then
    (rect.m * (rect.n - 1)) / 4
  else if rect.m % 2 = 1 ∧ rect.n % 2 = 0 then
    (rect.n * (rect.m - 1)) / 4
  else if rect.m % 2 = 1 ∧ rect.n % 2 = 1 then
    if (rect.n - 1) % 4 = 0 then
      (rect.m * (rect.n - 1)) / 4
    else
      (rect.m * (rect.n - 1) - 2) / 4
  else
    (rect.m * rect.n) / 4

theorem max_quartets_correct (rect : Rectangle) :
  max_quartets rect =
    if rect.m % 2 = 0 ∧ rect.n % 2 = 1 then
      (rect.m * (rect.n - 1)) / 4
    else if rect.m % 2 = 1 ∧ rect.n % 2 = 0 then
      (rect.n * (rect.m - 1)) / 4
    else if rect.m % 2 = 1 ∧ rect.n % 2 = 1 then
      if (rect.n - 1) % 4 = 0 then
        (rect.m * (rect.n - 1)) / 4
      else
        (rect.m * (rect.n - 1) - 2) / 4
    else
      (rect.m * rect.n) / 4 :=
by sorry

/-- Specific case for 5x5 square --/
def square_5x5 : Rectangle := { m := 5, n := 5 }

theorem max_quartets_5x5 :
  max_quartets square_5x5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l3890_389006


namespace NUMINAMATH_CALUDE_number_problem_l3890_389024

theorem number_problem (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 320) : 
  x * y = 64 ∧ x^3 + y^3 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3890_389024


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3890_389070

def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

theorem right_triangle_sets :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 7) 3 5 ∧
  is_right_triangle 6 8 10 ∧
  ¬ is_right_triangle 5 12 12 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3890_389070


namespace NUMINAMATH_CALUDE_equidifference_ratio_sequence_properties_l3890_389086

/-- Definition of an equidifference ratio sequence -/
def IsEquidifferenceRatioSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ+, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

theorem equidifference_ratio_sequence_properties
  (a : ℕ+ → ℝ) (h : IsEquidifferenceRatioSequence a) :
  (∃ k : ℝ, k ≠ 0 ∧ ∀ n : ℕ+, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k) ∧
  (∃ b : ℕ+ → ℝ, IsEquidifferenceRatioSequence b ∧ Set.Infinite {n : ℕ+ | b n = 0}) :=
by sorry

end NUMINAMATH_CALUDE_equidifference_ratio_sequence_properties_l3890_389086


namespace NUMINAMATH_CALUDE_max_pieces_8x8_grid_l3890_389099

/-- Represents a square grid -/
structure Grid :=
  (size : Nat)

/-- Represents the number of pieces after cutting -/
def num_pieces (g : Grid) (num_cuts : Nat) : Nat :=
  sorry

/-- The maximum number of pieces that can be obtained from an 8x8 grid -/
theorem max_pieces_8x8_grid :
  ∃ (max_pieces : Nat), 
    (∀ (g : Grid) (num_cuts : Nat), 
      g.size = 8 → num_pieces g num_cuts ≤ max_pieces) ∧ 
    (∃ (g : Grid) (num_cuts : Nat), 
      g.size = 8 ∧ num_pieces g num_cuts = max_pieces) ∧
    max_pieces = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_8x8_grid_l3890_389099


namespace NUMINAMATH_CALUDE_min_value_expression_l3890_389096

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y = 1/2) (horder : x ≤ y ∧ y ≤ z) :
  (x + z) / (x * y * z) ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3890_389096


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l3890_389038

theorem cos_2alpha_plus_pi_3 (α : Real) 
  (h : Real.sin (π / 6 - α) - Real.cos α = 1 / 3) : 
  Real.cos (2 * α + π / 3) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_3_l3890_389038


namespace NUMINAMATH_CALUDE_expression_equality_l3890_389044

theorem expression_equality : 
  |Real.sqrt 3 - 2| - (1 / 2)⁻¹ - 2 * Real.sin (π / 3) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3890_389044


namespace NUMINAMATH_CALUDE_pyramid_top_value_l3890_389051

/-- Represents a pyramid structure where each number is the product of the two numbers above it -/
structure Pyramid where
  bottom_row : Fin 3 → ℕ
  x : ℕ
  y : ℕ

/-- The conditions of the pyramid problem -/
def pyramid_conditions (p : Pyramid) : Prop :=
  p.bottom_row 0 = 240 ∧
  p.bottom_row 1 = 720 ∧
  p.bottom_row 2 = 1440 ∧
  p.x * 6 = 720

/-- The theorem stating that given the conditions, y must be 120 -/
theorem pyramid_top_value (p : Pyramid) (h : pyramid_conditions p) : p.y = 120 := by
  sorry

#check pyramid_top_value

end NUMINAMATH_CALUDE_pyramid_top_value_l3890_389051


namespace NUMINAMATH_CALUDE_product_nine_sum_zero_l3890_389091

theorem product_nine_sum_zero (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 9 →
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_nine_sum_zero_l3890_389091


namespace NUMINAMATH_CALUDE_cut_rectangle_decreases_area_and_perimeter_l3890_389043

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define the area and perimeter functions
def area (r : Rectangle) : ℝ := r.length * r.width
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

-- State the theorem
theorem cut_rectangle_decreases_area_and_perimeter 
  (R : Rectangle) 
  (S : Rectangle) 
  (h_cut : S.length ≤ R.length ∧ S.width ≤ R.width) 
  (h_proper_subset : S.length < R.length ∨ S.width < R.width) : 
  area S < area R ∧ perimeter S < perimeter R := by
  sorry

end NUMINAMATH_CALUDE_cut_rectangle_decreases_area_and_perimeter_l3890_389043


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3890_389013

theorem sum_of_roots_quadratic (x : ℝ) : x^2 = 16*x - 5 → ∃ y : ℝ, x^2 = 16*x - 5 ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3890_389013


namespace NUMINAMATH_CALUDE_infinitely_many_integers_l3890_389076

theorem infinitely_many_integers (k : ℕ) (hk : k > 1) :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_integers_l3890_389076


namespace NUMINAMATH_CALUDE_union_of_sets_l3890_389046

theorem union_of_sets (A B : Set ℕ) (m : ℕ) : 
  A = {1, 2, 4} → 
  B = {m, 4, 7} → 
  A ∩ B = {1, 4} → 
  A ∪ B = {1, 2, 4, 7} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l3890_389046


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l3890_389053

theorem circus_ticket_cost (total_spent : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℕ) : 
  total_spent = 308 → num_tickets = 7 → cost_per_ticket = total_spent / num_tickets → cost_per_ticket = 44 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l3890_389053


namespace NUMINAMATH_CALUDE_angle_sum_equals_arctangent_of_ratio_l3890_389089

theorem angle_sum_equals_arctangent_of_ratio
  (θ φ : ℝ)
  (θ_acute : 0 < θ ∧ θ < π / 2)
  (φ_acute : 0 < φ ∧ φ < π / 2)
  (tan_θ : Real.tan θ = 2 / 9)
  (sin_φ : Real.sin φ = 3 / 5) :
  θ + 2 * φ = Real.arctan (230 / 15) :=
sorry

end NUMINAMATH_CALUDE_angle_sum_equals_arctangent_of_ratio_l3890_389089


namespace NUMINAMATH_CALUDE_age_difference_is_two_l3890_389034

/-- The age difference between Jayson's dad and mom -/
def age_difference (jayson_age : ℕ) (mom_age_at_birth : ℕ) : ℕ :=
  (4 * jayson_age) - (mom_age_at_birth + jayson_age)

/-- Theorem stating the age difference between Jayson's dad and mom is 2 years -/
theorem age_difference_is_two :
  age_difference 10 28 = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_two_l3890_389034


namespace NUMINAMATH_CALUDE_inequality_proof_l3890_389069

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_3 : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b + c)^2) + (b^2 + 9) / (2*b^2 + (c + a)^2) + (c^2 + 9) / (2*c^2 + (a + b)^2) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3890_389069
