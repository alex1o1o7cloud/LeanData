import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l81_8198

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem solution_set_of_inequality 
  (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f a b x < f a b y) 
  (h_intersect : f a b 2 = 0) :
  {x : ℝ | b * x^2 - a * x > 0} = {x : ℝ | -1/2 < x ∧ x < 0} := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l81_8198


namespace NUMINAMATH_CALUDE_floor_equation_solution_l81_8112

theorem floor_equation_solution (x : ℚ) : 
  (⌊20 * x + 23⌋ = 20 + 23 * x) ↔ 
  (∃ k : ℕ, k ≤ 7 ∧ x = (23 - k : ℚ) / 23) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l81_8112


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l81_8154

theorem factorial_equation_solution (m k : ℕ) (hm : m = 7) (hk : k = 12) :
  ∃ P : ℕ, (Nat.factorial 7) * (Nat.factorial 14) = 18 * P * (Nat.factorial 11) ∧ P = 54080 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l81_8154


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l81_8189

-- Problem 1
theorem problem_1 (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^2 + x + 2023 = 2025 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h : a + b = 5) :
  2*(a + b) - 4*a - 4*b + 21 = 11 := by sorry

-- Problem 3
theorem problem_3 (a b : ℝ) (h1 : a^2 + 3*a*b = 20) (h2 : b^2 + 5*a*b = 8) :
  2*a^2 - b^2 + a*b = 32 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l81_8189


namespace NUMINAMATH_CALUDE_even_function_property_l81_8111

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_positive : ∀ x > 0, f x = x) :
  ∀ x < 0, f x = -x :=
by sorry

end NUMINAMATH_CALUDE_even_function_property_l81_8111


namespace NUMINAMATH_CALUDE_expression_evaluation_l81_8129

theorem expression_evaluation : (-1)^3 + 4 * (-2) - 3 / (-3) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l81_8129


namespace NUMINAMATH_CALUDE_ball_bird_intersection_time_l81_8118

/-- The time at which a ball thrown off a cliff and a bird flying upwards from the base of the cliff are at the same height -/
theorem ball_bird_intersection_time : 
  ∃ t : ℝ, t > 0 ∧ (60 - 9*t - 8*t^2 = 3*t^2 + 4*t) ∧ t = 20/11 := by
  sorry

#check ball_bird_intersection_time

end NUMINAMATH_CALUDE_ball_bird_intersection_time_l81_8118


namespace NUMINAMATH_CALUDE_complement_of_A_l81_8174

/-- Given that the universal set U is the set of real numbers and 
    A is the set of real numbers x such that 1 < x ≤ 3,
    prove that the complement of A with respect to U 
    is the set of real numbers x such that x ≤ 1 or x > 3 -/
theorem complement_of_A (U : Set ℝ) (A : Set ℝ) 
  (h_U : U = Set.univ)
  (h_A : A = {x : ℝ | 1 < x ∧ x ≤ 3}) :
  U \ A = {x : ℝ | x ≤ 1 ∨ x > 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l81_8174


namespace NUMINAMATH_CALUDE_exists_multiple_of_three_l81_8180

def CircleNumbers (n : ℕ) := Fin n → ℕ

def ValidCircle (nums : CircleNumbers 99) : Prop :=
  ∀ i : Fin 99, 
    (nums i - nums (i + 1) = 1) ∨ 
    (nums i - nums (i + 1) = 2) ∨ 
    (nums i / nums (i + 1) = 2)

theorem exists_multiple_of_three (nums : CircleNumbers 99) 
  (h : ValidCircle nums) : 
  ∃ i : Fin 99, 3 ∣ nums i :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_of_three_l81_8180


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cube_eq_x_l81_8126

theorem x_eq_one_sufficient_not_necessary_for_cube_eq_x (x : ℝ) :
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cube_eq_x_l81_8126


namespace NUMINAMATH_CALUDE_product_of_max_min_sum_l81_8142

theorem product_of_max_min_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * 2^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0 →
  ∃ (min_sum max_sum : ℝ),
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * 2^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 →
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 :=
by sorry

end NUMINAMATH_CALUDE_product_of_max_min_sum_l81_8142


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l81_8178

theorem factorial_ratio_equals_seven_and_half :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / 
  (Nat.factorial 9 * Nat.factorial 8 : ℚ) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l81_8178


namespace NUMINAMATH_CALUDE_complex_equation_solution_l81_8144

theorem complex_equation_solution (z : ℂ) : 
  (3 + Complex.I) * z = 2 - Complex.I → 
  z = (1 / 2 : ℂ) - (1 / 2 : ℂ) * Complex.I ∧ Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l81_8144


namespace NUMINAMATH_CALUDE_prism_volume_l81_8181

/-- The volume of a right rectangular prism with face areas 100, 200, and 300 square units -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 100)
  (h2 : b * c = 200)
  (h3 : c * a = 300) : 
  a * b * c = 1000 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l81_8181


namespace NUMINAMATH_CALUDE_min_bushes_cover_alley_l81_8175

/-- The length of the alley in meters -/
def alley_length : ℝ := 400

/-- The radius of scent spread for each lily of the valley bush in meters -/
def scent_radius : ℝ := 20

/-- The minimum number of bushes needed to cover the alley with scent -/
def min_bushes : ℕ := 10

/-- Theorem stating that the minimum number of bushes needed to cover the alley is correct -/
theorem min_bushes_cover_alley :
  ∀ (n : ℕ), n ≥ min_bushes → n * (2 * scent_radius) ≥ alley_length :=
by sorry

end NUMINAMATH_CALUDE_min_bushes_cover_alley_l81_8175


namespace NUMINAMATH_CALUDE_impossible_to_achieve_goal_state_l81_8184

/-- Represents a jar with a certain volume of tea and amount of sugar -/
structure Jar where
  volume : ℕ
  sugar : ℕ

/-- Represents the state of the system with three jars -/
structure SystemState where
  jar1 : Jar
  jar2 : Jar
  jar3 : Jar

/-- Represents a transfer of tea between jars -/
inductive Transfer where
  | from1to2 : Transfer
  | from1to3 : Transfer
  | from2to1 : Transfer
  | from2to3 : Transfer
  | from3to1 : Transfer
  | from3to2 : Transfer

def initial_state : SystemState :=
  { jar1 := { volume := 0, sugar := 0 },
    jar2 := { volume := 700, sugar := 50 },
    jar3 := { volume := 800, sugar := 60 } }

def transfer_amount : ℕ := 100

def is_valid_state (s : SystemState) : Prop :=
  s.jar1.volume + s.jar2.volume + s.jar3.volume = 1500 ∧
  s.jar1.volume % transfer_amount = 0 ∧
  s.jar2.volume % transfer_amount = 0 ∧
  s.jar3.volume % transfer_amount = 0

def apply_transfer (s : SystemState) (t : Transfer) : SystemState :=
  sorry

def is_goal_state (s : SystemState) : Prop :=
  s.jar1.volume = 0 ∧ s.jar2.sugar = s.jar3.sugar

theorem impossible_to_achieve_goal_state :
  ∀ (transfers : List Transfer),
    let final_state := transfers.foldl apply_transfer initial_state
    is_valid_state final_state → ¬is_goal_state final_state :=
  sorry

end NUMINAMATH_CALUDE_impossible_to_achieve_goal_state_l81_8184


namespace NUMINAMATH_CALUDE_shortest_path_bound_l81_8147

/-- Represents an equilateral tetrahedron -/
structure EquilateralTetrahedron where
  /-- The side length of the tetrahedron -/
  side_length : ℝ
  /-- Assertion that the side length is positive -/
  side_length_pos : side_length > 0

/-- Represents a point on the surface of an equilateral tetrahedron -/
structure SurfacePoint (T : EquilateralTetrahedron) where
  /-- Coordinates of the point on the surface -/
  coords : ℝ × ℝ × ℝ

/-- Calculates the shortest path between two points on the surface of an equilateral tetrahedron -/
def shortest_path (T : EquilateralTetrahedron) (p1 p2 : SurfacePoint T) : ℝ :=
  sorry

/-- Calculates the diameter of the circumscribed circle around a face of an equilateral tetrahedron -/
def face_circumcircle_diameter (T : EquilateralTetrahedron) : ℝ :=
  sorry

/-- Theorem: The shortest path between any two points on the surface of an equilateral tetrahedron
    is at most equal to the diameter of the circumscribed circle around a face of the tetrahedron -/
theorem shortest_path_bound (T : EquilateralTetrahedron) (p1 p2 : SurfacePoint T) :
  shortest_path T p1 p2 ≤ face_circumcircle_diameter T :=
  sorry

end NUMINAMATH_CALUDE_shortest_path_bound_l81_8147


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l81_8121

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 5 * x - 12 = 0 ↔ x = 3 ∨ x = -4/3) → k = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l81_8121


namespace NUMINAMATH_CALUDE_sin_plus_cos_eq_neg_one_solution_set_l81_8170

theorem sin_plus_cos_eq_neg_one_solution_set :
  {x : ℝ | Real.sin x + Real.cos x = -1} = 
  {x : ℝ | ∃ n : ℤ, x = (2*n - 1)*Real.pi ∨ x = 2*n*Real.pi - Real.pi/2} :=
by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_eq_neg_one_solution_set_l81_8170


namespace NUMINAMATH_CALUDE_car_truck_difference_l81_8146

theorem car_truck_difference (total_vehicles trucks : ℕ) 
  (h1 : total_vehicles = 69)
  (h2 : trucks = 21)
  (h3 : total_vehicles > 2 * trucks) : 
  total_vehicles - 2 * trucks = 27 := by
  sorry

end NUMINAMATH_CALUDE_car_truck_difference_l81_8146


namespace NUMINAMATH_CALUDE_police_text_percentage_l81_8191

theorem police_text_percentage : 
  ∀ (total_texts grocery_texts response_texts police_texts : ℕ),
    total_texts = 33 →
    grocery_texts = 5 →
    response_texts = 5 * grocery_texts →
    police_texts = total_texts - (grocery_texts + response_texts) →
    (police_texts : ℚ) / (grocery_texts + response_texts : ℚ) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_police_text_percentage_l81_8191


namespace NUMINAMATH_CALUDE_partitions_divisible_by_2_pow_n_l81_8115

/-- Represents a valid partition of a 1 × n strip -/
def StripPartition (n : ℕ) : Type := Unit

/-- The number of valid partitions for a 1 × n strip -/
def num_partitions (n : ℕ) : ℕ := sorry

/-- The main theorem: the number of valid partitions is divisible by 2^n -/
theorem partitions_divisible_by_2_pow_n (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, num_partitions n = 2^n * k :=
sorry

end NUMINAMATH_CALUDE_partitions_divisible_by_2_pow_n_l81_8115


namespace NUMINAMATH_CALUDE_common_tangents_exist_curves_intersect_at_angles_l81_8196

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with the equation 16x^2 + 25y^2 = 400 -/
def is_on_ellipse (p : Point) : Prop :=
  16 * p.x^2 + 25 * p.y^2 = 400

/-- Defines a circle with the equation x^2 + y^2 = 20 -/
def is_on_circle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 20

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to the ellipse -/
def is_tangent_to_ellipse (l : Line) : Prop :=
  ∃ p : Point, is_on_ellipse p ∧ l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line is tangent to the circle -/
def is_tangent_to_circle (l : Line) : Prop :=
  ∃ p : Point, is_on_circle p ∧ l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem stating that there exist common tangents to the ellipse and circle -/
theorem common_tangents_exist : 
  ∃ l : Line, is_tangent_to_ellipse l ∧ is_tangent_to_circle l :=
sorry

/-- Calculates the angle between two curves at an intersection point -/
noncomputable def angle_between_curves (p : Point) : ℝ :=
sorry

/-- Theorem stating that the ellipse and circle intersect at certain angles -/
theorem curves_intersect_at_angles : 
  ∃ p : Point, is_on_ellipse p ∧ is_on_circle p ∧ angle_between_curves p ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_exist_curves_intersect_at_angles_l81_8196


namespace NUMINAMATH_CALUDE_rich_book_pages_left_to_read_l81_8172

/-- Given a book with a total number of pages, the number of pages already read,
    and the number of pages to be skipped, calculate the number of pages left to read. -/
def pages_left_to_read (total_pages read_pages skipped_pages : ℕ) : ℕ :=
  total_pages - (read_pages + skipped_pages)

/-- Theorem stating that for a 372-page book with 125 pages read and 16 pages skipped,
    there are 231 pages left to read. -/
theorem rich_book_pages_left_to_read :
  pages_left_to_read 372 125 16 = 231 := by
  sorry

end NUMINAMATH_CALUDE_rich_book_pages_left_to_read_l81_8172


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_18n_integer_l81_8158

theorem smallest_n_for_sqrt_18n_integer (n : ℕ) : 
  (∀ k : ℕ, 0 < k → k < 2 → ¬ ∃ m : ℕ, m^2 = 18 * k) ∧ 
  (∃ m : ℕ, m^2 = 18 * 2) → 
  n = 2 → 
  (∃ m : ℕ, m^2 = 18 * n) ∧ 
  (∀ k : ℕ, 0 < k → k < n → ¬ ∃ m : ℕ, m^2 = 18 * k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_18n_integer_l81_8158


namespace NUMINAMATH_CALUDE_vacation_tents_l81_8134

/-- Represents the sleeping arrangements for a family vacation --/
structure SleepingArrangements where
  indoor_capacity : ℕ
  max_per_tent : ℕ
  total_people : ℕ
  teenagers : ℕ
  young_children : ℕ
  infant_families : ℕ
  single_adults : ℕ
  dogs : ℕ

/-- Calculates the number of tents needed given the sleeping arrangements --/
def calculate_tents (arrangements : SleepingArrangements) : ℕ :=
  let outdoor_people := arrangements.total_people - arrangements.indoor_capacity
  let teen_tents := (arrangements.teenagers + 1) / 2
  let child_tents := (arrangements.young_children + 1) / 2
  let adult_tents := (outdoor_people - arrangements.teenagers - arrangements.young_children - arrangements.infant_families + 1) / 2
  teen_tents + child_tents + adult_tents + arrangements.dogs

/-- Theorem stating that the given sleeping arrangements require 7 tents --/
theorem vacation_tents (arrangements : SleepingArrangements) 
  (h1 : arrangements.indoor_capacity = 6)
  (h2 : arrangements.max_per_tent = 2)
  (h3 : arrangements.total_people = 20)
  (h4 : arrangements.teenagers = 2)
  (h5 : arrangements.young_children = 5)
  (h6 : arrangements.infant_families = 3)
  (h7 : arrangements.single_adults = 1)
  (h8 : arrangements.dogs = 1) :
  calculate_tents arrangements = 7 := by
  sorry


end NUMINAMATH_CALUDE_vacation_tents_l81_8134


namespace NUMINAMATH_CALUDE_bobby_candy_count_l81_8153

/-- The total number of candy pieces Bobby ate -/
def total_candy (initial : ℕ) (more : ℕ) (chocolate : ℕ) : ℕ :=
  initial + more + chocolate

/-- Theorem stating that Bobby ate 133 pieces of candy in total -/
theorem bobby_candy_count :
  total_candy 28 42 63 = 133 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l81_8153


namespace NUMINAMATH_CALUDE_max_min_sum_on_interval_l81_8168

def f (x : ℝ) := 2 * x^2 - 6 * x + 1

theorem max_min_sum_on_interval :
  ∃ (m M : ℝ),
    (∀ x ∈ Set.Icc (-1) 1, m ≤ f x ∧ f x ≤ M) ∧
    (∃ x₁ ∈ Set.Icc (-1) 1, f x₁ = m) ∧
    (∃ x₂ ∈ Set.Icc (-1) 1, f x₂ = M) ∧
    M + m = 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_on_interval_l81_8168


namespace NUMINAMATH_CALUDE_two_week_riding_time_l81_8152

-- Define the riding schedule
def riding_schedule : List (String × Float) := [
  ("Monday", 1),
  ("Tuesday", 0.5),
  ("Wednesday", 1),
  ("Thursday", 0.5),
  ("Friday", 1),
  ("Saturday", 2),
  ("Sunday", 0)
]

-- Calculate the total riding time for one week
def weekly_riding_time : Float :=
  (riding_schedule.map (λ (_, time) => time)).sum

-- Theorem: The total riding time for a 2-week period is 12 hours
theorem two_week_riding_time :
  weekly_riding_time * 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_week_riding_time_l81_8152


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_interval_l81_8185

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = x^2}
def N : Set ℝ := {x | ∃ y, x^2 + y^2 = 2}

-- Define the interval [0, √2]
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ Real.sqrt 2}

-- State the theorem
theorem M_intersect_N_eq_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_interval_l81_8185


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l81_8119

/-- Given that real numbers 4, m, and 9 form a geometric sequence, 
    prove that the eccentricity of the conic section represented by 
    the equation x²/m + y² = 1 is either √(30)/6 or √7. -/
theorem conic_section_eccentricity (m : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ m = 4 * r ∧ 9 = m * r) →
  let e := if m > 0 
    then Real.sqrt (30) / 6 
    else Real.sqrt 7
  (∀ x y : ℝ, x^2 / m + y^2 = 1) →
  ∃ (a b c : ℝ), 
    (m > 0 → a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 - b^2 ∧ e = c / a) ∧
    (m < 0 → a^2 = 1 ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ e = c / a) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l81_8119


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l81_8123

/-- The sum of interior angles of a hexagon is 720 degrees -/
def hexagon_angle_sum : ℝ := 720

/-- The given angles of the hexagon -/
def given_angles : List ℝ := [135, 105, 87, 120, 78]

/-- Theorem: In a hexagon where five of the interior angles measure 135°, 105°, 87°, 120°, and 78°, the sixth angle measures 195°. -/
theorem hexagon_sixth_angle : 
  List.sum given_angles + 195 = hexagon_angle_sum := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l81_8123


namespace NUMINAMATH_CALUDE_incorrect_solution_set_proof_l81_8149

def Equation := ℝ → Prop

def SolutionSet (eq : Equation) := {x : ℝ | eq x}

theorem incorrect_solution_set_proof (eq : Equation) (S : Set ℝ) :
  (∀ x, ¬(eq x) → x ∉ S) ∧ (∀ x ∈ S, eq x) → S = SolutionSet eq → False :=
sorry

end NUMINAMATH_CALUDE_incorrect_solution_set_proof_l81_8149


namespace NUMINAMATH_CALUDE_fraction_relations_l81_8104

theorem fraction_relations (x y : ℚ) (h : x / y = 2 / 5) :
  (x + y) / y = 7 / 5 ∧ 
  y / (y - x) = 5 / 3 ∧ 
  x / (3 * y) = 2 / 15 ∧ 
  (x + 3 * y) / x ≠ 17 / 2 ∧ 
  (x - y) / y ≠ 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relations_l81_8104


namespace NUMINAMATH_CALUDE_percentage_increase_income_l81_8179

/-- Calculate the percentage increase in combined weekly income --/
theorem percentage_increase_income (initial_job_income initial_side_income final_job_income final_side_income : ℝ) :
  initial_job_income = 50 →
  initial_side_income = 20 →
  final_job_income = 90 →
  final_side_income = 30 →
  let initial_total := initial_job_income + initial_side_income
  let final_total := final_job_income + final_side_income
  let increase := final_total - initial_total
  let percentage_increase := (increase / initial_total) * 100
  ∀ ε > 0, |percentage_increase - 71.43| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_income_l81_8179


namespace NUMINAMATH_CALUDE_sin_power_sum_l81_8110

theorem sin_power_sum (φ : Real) (x : Real) (n : Nat) 
  (h1 : 0 < φ) (h2 : φ < π / 2) 
  (h3 : x + 1 / x = 2 * Real.sin φ) 
  (h4 : n > 0) : 
  x^n + 1 / x^n = 2 * Real.sin (n * φ) := by
  sorry

end NUMINAMATH_CALUDE_sin_power_sum_l81_8110


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l81_8103

theorem salary_savings_percentage (prev_salary : ℝ) (prev_savings_rate : ℝ) 
  (h1 : prev_savings_rate > 0) 
  (h2 : prev_savings_rate < 1) : 
  let new_salary : ℝ := prev_salary * 1.1
  let new_savings_rate : ℝ := 0.1
  let new_savings : ℝ := new_salary * new_savings_rate
  let prev_savings : ℝ := prev_salary * prev_savings_rate
  new_savings = prev_savings * 1.8333333333333331 → prev_savings_rate = 0.06 := by
sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l81_8103


namespace NUMINAMATH_CALUDE_food_distribution_problem_l81_8106

/-- The number of days that passed before additional men joined -/
def x : ℝ := 2

/-- The initial number of men -/
def initial_men : ℝ := 760

/-- The number of days the food was initially planned to last -/
def initial_days : ℝ := 22

/-- The number of additional men that joined -/
def additional_men : ℝ := 134.11764705882354

/-- The number of days the food lasted after additional men joined -/
def remaining_days : ℝ := 17

/-- The total food supply in man-days -/
def total_food_supply : ℝ := initial_men * initial_days

theorem food_distribution_problem :
  initial_men * x + (initial_men + additional_men) * remaining_days = total_food_supply := by
  sorry

end NUMINAMATH_CALUDE_food_distribution_problem_l81_8106


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l81_8188

/-- Two lines are parallel if their slopes are equal but not equal to the ratio of their constants -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ / n₁ = m₂ / n₂ ∧ m₁ / n₁ ≠ c₁ / c₂

theorem parallel_lines_a_value (a : ℝ) :
  parallel (3 + a) 4 (5 - 3*a) 2 (5 + a) 8 → a = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l81_8188


namespace NUMINAMATH_CALUDE_remainder_theorem_l81_8133

theorem remainder_theorem (x y u v : ℤ) (hx : 0 < x) (hy : 0 < y) (h_div : x = u * y + v) (h_rem : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l81_8133


namespace NUMINAMATH_CALUDE_at_least_one_equals_one_iff_sum_gt_product_l81_8131

theorem at_least_one_equals_one_iff_sum_gt_product (m n : ℕ+) :
  (m = 1 ∨ n = 1) ↔ (m + n : ℝ) > m * n := by sorry

end NUMINAMATH_CALUDE_at_least_one_equals_one_iff_sum_gt_product_l81_8131


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l81_8124

theorem range_of_m_for_quadratic_equation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ↔ 
  (m < -2 ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l81_8124


namespace NUMINAMATH_CALUDE_twin_prime_power_theorem_l81_8173

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q = p + 2

theorem twin_prime_power_theorem :
  ∀ n : ℕ, (∃ p q : ℕ, is_twin_prime p q ∧ is_twin_prime (2^n + p) (2^n + q)) ↔ n = 1 ∨ n = 3 :=
sorry

end NUMINAMATH_CALUDE_twin_prime_power_theorem_l81_8173


namespace NUMINAMATH_CALUDE_length_lost_per_knot_l81_8183

/-- Given a set of ropes and the total length after tying, calculate the length lost per knot -/
theorem length_lost_per_knot (rope_lengths : List ℝ) (total_length_after_tying : ℝ) : 
  rope_lengths = [8, 20, 2, 2, 2, 7] ∧ 
  total_length_after_tying = 35 → 
  (rope_lengths.sum - total_length_after_tying) / (rope_lengths.length - 1) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_length_lost_per_knot_l81_8183


namespace NUMINAMATH_CALUDE_BF_length_is_10_8_l81_8194

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral ABCD with specific properties -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  right_angle_A : True  -- Represents the right angle at A
  right_angle_C : True  -- Represents the right angle at C
  E_on_AC : True  -- Represents that E is on AC
  F_on_AC : True  -- Represents that F is on AC
  DE_perp_AC : True  -- Represents that DE is perpendicular to AC
  BF_perp_AC : True  -- Represents that BF is perpendicular to AC
  AE_length : Real
  DE_length : Real
  CE_length : Real
  h_AE : AE_length = 4
  h_DE : DE_length = 6
  h_CE : CE_length = 8

/-- Calculate the length of BF in the given quadrilateral -/
def calculate_BF_length (q : Quadrilateral) : ℝ := sorry

/-- Theorem stating that BF length is 10.8 -/
theorem BF_length_is_10_8 (q : Quadrilateral) : calculate_BF_length q = 10.8 := by sorry

end NUMINAMATH_CALUDE_BF_length_is_10_8_l81_8194


namespace NUMINAMATH_CALUDE_total_distance_calculation_l81_8190

/-- Calculates the total distance traveled given fuel efficiencies and fuel used for different driving conditions -/
theorem total_distance_calculation (city_efficiency highway_efficiency gravel_efficiency : ℝ)
  (city_fuel highway_fuel gravel_fuel : ℝ) : 
  city_efficiency = 15 →
  highway_efficiency = 25 →
  gravel_efficiency = 18 →
  city_fuel = 2.5 →
  highway_fuel = 3.8 →
  gravel_fuel = 1.7 →
  city_efficiency * city_fuel + highway_efficiency * highway_fuel + gravel_efficiency * gravel_fuel = 163.1 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_calculation_l81_8190


namespace NUMINAMATH_CALUDE_certain_number_proof_l81_8127

theorem certain_number_proof (N : ℚ) : 
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 200 → N = 384 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l81_8127


namespace NUMINAMATH_CALUDE_commission_for_398_machines_l81_8136

/-- Represents the commission structure and pricing model for machine sales -/
structure SalesModel where
  initialPrice : ℝ
  priceDecrease : ℝ
  commissionRate1 : ℝ
  commissionRate2 : ℝ
  commissionRate3 : ℝ
  threshold1 : ℕ
  threshold2 : ℕ

/-- Calculates the total commission for a given number of machines sold -/
def calculateCommission (model : SalesModel) (machinesSold : ℕ) : ℝ :=
  sorry

/-- The specific sales model for the problem -/
def problemModel : SalesModel :=
  { initialPrice := 10000
    priceDecrease := 500
    commissionRate1 := 0.03
    commissionRate2 := 0.04
    commissionRate3 := 0.05
    threshold1 := 150
    threshold2 := 250 }

theorem commission_for_398_machines :
  calculateCommission problemModel 398 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_commission_for_398_machines_l81_8136


namespace NUMINAMATH_CALUDE_sector_central_angle_l81_8176

/-- Given a sector with arc length 2π cm and radius 2 cm, its central angle is π radians. -/
theorem sector_central_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2 * Real.pi) (h2 : radius = 2) :
  arc_length / radius = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l81_8176


namespace NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l81_8102

/-- The minimum distance from a point on the ellipse x = 4cos(θ), y = 3sin(θ) to the line x - y - 6 = 0 is √2/2 -/
theorem min_distance_ellipse_to_line :
  let ellipse := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = 4 * Real.cos θ ∧ y = 3 * Real.sin θ}
  let line := {(x, y) : ℝ × ℝ | x - y - 6 = 0}
  ∀ p ∈ ellipse, (
    let dist := fun q : ℝ × ℝ => |q.1 - q.2 - 6| / Real.sqrt 2
    ∃ q ∈ line, dist q = Real.sqrt 2 / 2 ∧ ∀ r ∈ line, dist p ≥ Real.sqrt 2 / 2
  ) := by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_to_line_l81_8102


namespace NUMINAMATH_CALUDE_max_tank_volume_l81_8101

/-- A rectangular parallelepiped tank with the given properties -/
structure Tank where
  a : Real  -- length of the base
  b : Real  -- width of the base
  h : Real  -- height of the tank
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  side_area_condition : a * h ≥ a * b ∧ b * h ≥ a * b

/-- The theorem stating the maximum volume of the tank -/
theorem max_tank_volume (tank : Tank) (h_val : tank.h = 1.5) :
  (∀ t : Tank, t.h = 1.5 → t.a * t.b * t.h ≤ tank.a * tank.b * tank.h) →
  tank.a * tank.b * tank.h = 3.375 := by
  sorry

end NUMINAMATH_CALUDE_max_tank_volume_l81_8101


namespace NUMINAMATH_CALUDE_line_increase_theorem_l81_8164

/-- Represents a line in a Cartesian plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The increase in y for a given increase in x -/
def y_increase (l : Line) (x_increase : ℝ) : ℝ :=
  l.slope * x_increase

/-- Theorem: For a line with the given properties, an increase of 20 units in x
    from the point (1, 2) results in an increase of 41.8 units in y -/
theorem line_increase_theorem (l : Line) 
    (h1 : l.slope = 11 / 5)
    (h2 : 2 = l.slope * 1 + l.y_intercept) : 
    y_increase l 20 = 41.8 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_theorem_l81_8164


namespace NUMINAMATH_CALUDE_johns_annual_savings_l81_8187

/-- Calculates the annual savings for John's new apartment situation -/
theorem johns_annual_savings
  (former_rent_per_sqft : ℝ)
  (former_apartment_size : ℝ)
  (new_apartment_cost : ℝ)
  (h1 : former_rent_per_sqft = 2)
  (h2 : former_apartment_size = 750)
  (h3 : new_apartment_cost = 2800)
  : (former_rent_per_sqft * former_apartment_size - new_apartment_cost / 2) * 12 = 1200 := by
  sorry

#check johns_annual_savings

end NUMINAMATH_CALUDE_johns_annual_savings_l81_8187


namespace NUMINAMATH_CALUDE_cos_decreasing_interval_l81_8108

theorem cos_decreasing_interval (k : ℤ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 3)
  let a := k * π + π / 6
  let b := k * π + 2 * π / 3
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x :=
by sorry

end NUMINAMATH_CALUDE_cos_decreasing_interval_l81_8108


namespace NUMINAMATH_CALUDE_ellipse_properties_l81_8186

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The equation of the ellipse in the form x²/a² + y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- Checks if a point (x, y) lies on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  e.equation x y

/-- The focal distance of the ellipse -/
def Ellipse.focalDistance (e : Ellipse) : ℝ := 2

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
    (h1 : e.focalDistance = 2)
    (h2 : e.contains (-1) (3/2)) : 
  (∃ a b : ℝ, e.equation = fun x y ↦ x^2/a^2 + y^2/b^2 = 1 ∧ a = 2 ∧ b^2 = 3) ∧
  (∀ x y : ℝ, e.contains x y ↔ x^2/4 + y^2/3 = 1) ∧
  (e.contains 2 0 ∧ e.contains (-2) 0 ∧ e.contains 0 (Real.sqrt 3) ∧ e.contains 0 (-Real.sqrt 3)) ∧
  (∃ majorAxis : ℝ, majorAxis = 4) ∧
  (∃ minorAxis : ℝ, minorAxis = 2 * Real.sqrt 3) ∧
  (∃ eccentricity : ℝ, eccentricity = 1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l81_8186


namespace NUMINAMATH_CALUDE_gathering_handshakes_l81_8130

/-- Represents the number of handshakes in a gathering with specific group dynamics -/
def number_of_handshakes (total_people : ℕ) (group1_size : ℕ) (group2_size : ℕ) (group3_size : ℕ) (known_by_group3 : ℕ) : ℕ :=
  let group2_handshakes := group2_size * (total_people - group2_size)
  let group3_handshakes := group3_size * (group1_size - known_by_group3 + group2_size)
  (group2_handshakes + group3_handshakes) / 2

/-- The theorem states that for the given group sizes and dynamics, the number of handshakes is 210 -/
theorem gathering_handshakes :
  number_of_handshakes 35 25 5 5 18 = 210 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l81_8130


namespace NUMINAMATH_CALUDE_otimes_four_two_l81_8125

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem to prove
theorem otimes_four_two : otimes 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_two_l81_8125


namespace NUMINAMATH_CALUDE_plant_order_after_365_days_l81_8199

-- Define the plants
inductive Plant
| Cactus
| Dieffenbachia
| Orchid

-- Define the order of plants
def PlantOrder := List Plant

-- Define the initial order
def initialOrder : PlantOrder := [Plant.Cactus, Plant.Dieffenbachia, Plant.Orchid]

-- Define Luna's swap operation (left and center)
def lunaSwap (order : PlantOrder) : PlantOrder :=
  match order with
  | [a, b, c] => [b, a, c]
  | _ => order

-- Define Sam's swap operation (right and center)
def samSwap (order : PlantOrder) : PlantOrder :=
  match order with
  | [a, b, c] => [a, c, b]
  | _ => order

-- Define a single day's operation (Luna's swap followed by Sam's swap)
def dailyOperation (order : PlantOrder) : PlantOrder :=
  samSwap (lunaSwap order)

-- Define the operation for multiple days
def multiDayOperation (order : PlantOrder) (days : Nat) : PlantOrder :=
  match days with
  | 0 => order
  | n + 1 => multiDayOperation (dailyOperation order) n

-- Theorem to prove
theorem plant_order_after_365_days :
  multiDayOperation initialOrder 365 = [Plant.Orchid, Plant.Cactus, Plant.Dieffenbachia] :=
sorry

end NUMINAMATH_CALUDE_plant_order_after_365_days_l81_8199


namespace NUMINAMATH_CALUDE_max_area_is_35_l81_8162

/-- Represents the cost constraint for the rectangular frame -/
def cost_constraint (l w : ℕ) : Prop := 3 * l + 5 * w ≤ 50

/-- Represents the area of the rectangular frame -/
def area (l w : ℕ) : ℕ := l * w

/-- Theorem stating that the maximum area of the rectangular frame is 35 m² -/
theorem max_area_is_35 :
  ∃ (l w : ℕ), cost_constraint l w ∧ area l w = 35 ∧
  ∀ (l' w' : ℕ), cost_constraint l' w' → area l' w' ≤ 35 := by
  sorry

end NUMINAMATH_CALUDE_max_area_is_35_l81_8162


namespace NUMINAMATH_CALUDE_savings_after_purchase_l81_8140

/-- Calculates the amount left in savings after buying sweaters, scarves, and mittens for a family --/
theorem savings_after_purchase (sweater_price scarf_price mitten_price : ℕ) 
  (family_members total_savings : ℕ) : 
  sweater_price = 35 →
  scarf_price = 25 →
  mitten_price = 15 →
  family_members = 10 →
  total_savings = 800 →
  total_savings - (sweater_price + scarf_price + mitten_price) * family_members = 50 := by
  sorry

end NUMINAMATH_CALUDE_savings_after_purchase_l81_8140


namespace NUMINAMATH_CALUDE_short_answer_time_l81_8116

/-- Represents the time in minutes for various writing assignments --/
structure WritingTimes where
  essay : ℕ        -- Time for one essay in minutes
  paragraph : ℕ    -- Time for one paragraph in minutes
  shortAnswer : ℕ  -- Time for one short-answer question in minutes

/-- Represents the number of each type of assignment --/
structure AssignmentCounts where
  essays : ℕ
  paragraphs : ℕ
  shortAnswers : ℕ

/-- Calculates the total time in minutes for all assignments --/
def totalTime (times : WritingTimes) (counts : AssignmentCounts) : ℕ :=
  times.essay * counts.essays +
  times.paragraph * counts.paragraphs +
  times.shortAnswer * counts.shortAnswers

/-- The main theorem to prove --/
theorem short_answer_time 
  (times : WritingTimes) 
  (counts : AssignmentCounts) 
  (h1 : times.essay = 60)           -- Each essay takes 1 hour (60 minutes)
  (h2 : times.paragraph = 15)       -- Each paragraph takes 15 minutes
  (h3 : counts.essays = 2)          -- Karen assigns 2 essays
  (h4 : counts.paragraphs = 5)      -- Karen assigns 5 paragraphs
  (h5 : counts.shortAnswers = 15)   -- Karen assigns 15 short-answer questions
  (h6 : totalTime times counts = 240) -- Total homework time is 4 hours (240 minutes)
  : times.shortAnswer = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_short_answer_time_l81_8116


namespace NUMINAMATH_CALUDE_inequality_range_l81_8145

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) ↔ m ∈ Set.Iic (-3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l81_8145


namespace NUMINAMATH_CALUDE_grandmas_salad_l81_8171

/-- The number of mushrooms Grandma put on her salad -/
def mushrooms : ℕ := sorry

/-- The number of cherry tomatoes Grandma put on her salad -/
def cherry_tomatoes : ℕ := 2 * mushrooms

/-- The number of pickles Grandma put on her salad -/
def pickles : ℕ := 4 * cherry_tomatoes

/-- The total number of bacon bits Grandma put on her salad -/
def bacon_bits : ℕ := 4 * pickles

/-- The number of red bacon bits Grandma put on her salad -/
def red_bacon_bits : ℕ := 32

theorem grandmas_salad : mushrooms = 3 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_salad_l81_8171


namespace NUMINAMATH_CALUDE_linear_function_property_l81_8137

theorem linear_function_property (x y : ℝ) : 
  let f : ℝ → ℝ := fun x => 3 * x
  f ((x + y) / 2) = (1 / 2) * (f x + f y) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l81_8137


namespace NUMINAMATH_CALUDE_modular_inverse_27_mod_28_l81_8107

theorem modular_inverse_27_mod_28 : ∃ a : ℕ, 0 ≤ a ∧ a ≤ 27 ∧ (27 * a) % 28 = 1 :=
by
  use 27
  sorry

end NUMINAMATH_CALUDE_modular_inverse_27_mod_28_l81_8107


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l81_8163

/-- Represents the number of rooms that can be painted with the original amount of paint -/
def original_rooms : ℕ := 40

/-- Represents the number of rooms that can be painted after losing some paint -/
def remaining_rooms : ℕ := 31

/-- Represents the number of cans lost -/
def lost_cans : ℕ := 3

/-- Calculates the number of cans used to paint a given number of rooms -/
def cans_used (rooms : ℕ) : ℕ :=
  (rooms + 2) / 3

theorem paint_cans_theorem : cans_used remaining_rooms = 11 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l81_8163


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_difference_l81_8169

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- A predicate that checks if all digits in a number are different -/
def allDigitsDifferent (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10

theorem smallest_digit_sum_of_difference :
  ∀ a b : ℕ,
    100 ≤ a ∧ a < 1000 →
    100 ≤ b ∧ b < 1000 →
    a > b →
    allDigitsDifferent (1000000 * a + b) →
    100 ≤ a - b ∧ a - b < 1000 →
    (∀ D : ℕ, 100 ≤ D ∧ D < 1000 → D = a - b → sumOfDigits D ≥ 9) ∧
    (∃ D : ℕ, 100 ≤ D ∧ D < 1000 ∧ D = a - b ∧ sumOfDigits D = 9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_difference_l81_8169


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l81_8105

theorem greatest_integer_radius (r : ℕ) : (r : ℝ) ^ 2 * Real.pi < 90 * Real.pi → r ≤ 9 :=
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l81_8105


namespace NUMINAMATH_CALUDE_phone_plan_cost_difference_l81_8122

/-- Calculates the cost difference between Darnell's current phone plan and an alternative plan -/
theorem phone_plan_cost_difference :
  let current_plan_cost : ℚ := 12
  let texts_per_month : ℕ := 60
  let call_minutes_per_month : ℕ := 60
  let alt_plan_text_cost : ℚ := 1
  let alt_plan_text_limit : ℕ := 30
  let alt_plan_call_cost : ℚ := 3
  let alt_plan_call_limit : ℕ := 20
  let alt_plan_text_total : ℚ := (texts_per_month : ℚ) / alt_plan_text_limit * alt_plan_text_cost
  let alt_plan_call_total : ℚ := (call_minutes_per_month : ℚ) / alt_plan_call_limit * alt_plan_call_cost
  let alt_plan_total_cost : ℚ := alt_plan_text_total + alt_plan_call_total
  current_plan_cost - alt_plan_total_cost = 1 :=
by sorry

end NUMINAMATH_CALUDE_phone_plan_cost_difference_l81_8122


namespace NUMINAMATH_CALUDE_paths_in_7x6_grid_l81_8166

/-- The number of paths in a grid with specified horizontal and vertical steps --/
def numPaths (horizontal vertical : ℕ) : ℕ :=
  Nat.choose (horizontal + vertical) vertical

/-- Theorem stating that the number of paths in a 7x6 grid is 1716 --/
theorem paths_in_7x6_grid :
  numPaths 7 6 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x6_grid_l81_8166


namespace NUMINAMATH_CALUDE_can_display_properties_l81_8139

/-- Represents a triangular display of cans. -/
structure CanDisplay where
  totalCans : ℕ
  canWeight : ℕ

/-- Calculates the number of rows in the display. -/
def numberOfRows (d : CanDisplay) : ℕ :=
  Nat.sqrt d.totalCans

/-- Calculates the total weight of the display in kg. -/
def totalWeight (d : CanDisplay) : ℕ :=
  d.totalCans * d.canWeight

/-- Theorem stating the properties of the specific can display. -/
theorem can_display_properties (d : CanDisplay) 
  (h1 : d.totalCans = 225)
  (h2 : d.canWeight = 5) :
  numberOfRows d = 15 ∧ totalWeight d = 1125 := by
  sorry

end NUMINAMATH_CALUDE_can_display_properties_l81_8139


namespace NUMINAMATH_CALUDE_unique_number_with_conditions_l81_8150

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧ digit_sum n = 25 ∧ n % 5 = 0

theorem unique_number_with_conditions :
  ∃! n : ℕ, satisfies_conditions n :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_conditions_l81_8150


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_max_value_of_m_max_value_of_m_achievable_l81_8109

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + 2 * |x + b|

-- Theorem 1
theorem sum_of_a_and_b_is_one 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 1) 
  (hmin_exists : ∃ x, f x a b = 1) : 
  a + b = 1 := 
sorry

-- Theorem 2
theorem max_value_of_m 
  (a b m : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a + b = 1) 
  (hm : m ≤ 1/a + 2/b) : 
  m ≤ 3 + 2 * Real.sqrt 2 := 
sorry

-- The maximum value is achievable
theorem max_value_of_m_achievable :
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ 
  ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ m ≤ 1/a + 2/b :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_max_value_of_m_max_value_of_m_achievable_l81_8109


namespace NUMINAMATH_CALUDE_min_value_of_expression_l81_8159

theorem min_value_of_expression (x : ℝ) :
  (x^2 - 4*x + 3) * (x^2 + 4*x + 3) ≥ -16 ∧
  ∃ y : ℝ, (y^2 - 4*y + 3) * (y^2 + 4*y + 3) = -16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l81_8159


namespace NUMINAMATH_CALUDE_min_fraction_sum_l81_8100

theorem min_fraction_sum :
  ∃ (p q r s : ℕ),
    p ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    q ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    r ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    s ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    (p : ℚ) / q + (r : ℚ) / s = 5 / 6 ∧
    ∀ (a b c d : ℕ),
      a ∈ ({1, 2, 3, 4} : Set ℕ) →
      b ∈ ({1, 2, 3, 4} : Set ℕ) →
      c ∈ ({1, 2, 3, 4} : Set ℕ) →
      d ∈ ({1, 2, 3, 4} : Set ℕ) →
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
      (a : ℚ) / b + (c : ℚ) / d ≥ 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l81_8100


namespace NUMINAMATH_CALUDE_regression_change_l81_8148

/-- Represents a linear regression equation of the form ŷ = a + bx̂ -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the change in ŷ when x̂ increases by 1 unit -/
def change_in_y (eq : LinearRegression) : ℝ := -eq.b

/-- Theorem: For the regression equation ŷ = 2 - 3x̂, 
    when x̂ increases by 1 unit, ŷ decreases by 3 units -/
theorem regression_change : 
  let eq := LinearRegression.mk 2 (-3)
  change_in_y eq = -3 := by sorry

end NUMINAMATH_CALUDE_regression_change_l81_8148


namespace NUMINAMATH_CALUDE_no_y_intercepts_l81_8141

/-- A parabola defined by x = 2y^2 - 3y + 7 -/
def parabola (y : ℝ) : ℝ := 2 * y^2 - 3 * y + 7

/-- A y-intercept occurs when x = 0 -/
def is_y_intercept (y : ℝ) : Prop := parabola y = 0

/-- The parabola has no y-intercepts -/
theorem no_y_intercepts : ¬∃ y : ℝ, is_y_intercept y := by
  sorry

end NUMINAMATH_CALUDE_no_y_intercepts_l81_8141


namespace NUMINAMATH_CALUDE_unique_solution_condition_l81_8177

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l81_8177


namespace NUMINAMATH_CALUDE_soccer_balls_theorem_l81_8132

/-- The number of soccer balls originally purchased by the store -/
def original_balls : ℕ := 130

/-- The wholesale price of each soccer ball -/
def wholesale_price : ℕ := 30

/-- The retail price of each soccer ball -/
def retail_price : ℕ := 45

/-- The number of soccer balls remaining when the profit is calculated -/
def remaining_balls : ℕ := 30

/-- The profit made when there are 30 balls remaining -/
def profit : ℕ := 1500

/-- Theorem stating that the number of originally purchased soccer balls is 130 -/
theorem soccer_balls_theorem :
  (retail_price - wholesale_price) * (original_balls - remaining_balls) = profit :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_theorem_l81_8132


namespace NUMINAMATH_CALUDE_elderly_in_sample_l81_8197

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : ℕ
  middle : ℕ
  elderly : ℕ

/-- Represents the sampled employees -/
structure SampledEmployees where
  young : ℕ
  elderly : ℕ

/-- Theorem stating the number of elderly employees in the sample -/
theorem elderly_in_sample
  (total : ℕ)
  (employees : EmployeeCount)
  (sample : SampledEmployees)
  (h1 : total = employees.young + employees.middle + employees.elderly)
  (h2 : employees.young = 160)
  (h3 : employees.middle = 2 * employees.elderly)
  (h4 : total = 430)
  (h5 : sample.young = 32)
  : sample.elderly = 18 := by
  sorry

end NUMINAMATH_CALUDE_elderly_in_sample_l81_8197


namespace NUMINAMATH_CALUDE_village_population_equality_l81_8135

/-- The number of years it takes for two village populations to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_rate + x_rate)

/-- Theorem stating that the populations of Village X and Village Y will be equal after 16 years -/
theorem village_population_equality :
  years_to_equal_population 74000 1200 42000 800 = 16 := by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l81_8135


namespace NUMINAMATH_CALUDE_estimate_rabbit_population_l81_8151

/-- Estimate the number of rabbits in a forest using the capture-recapture method. -/
theorem estimate_rabbit_population (initial_marked : ℕ) (second_capture : ℕ) (marked_in_second : ℕ) :
  initial_marked = 50 →
  second_capture = 42 →
  marked_in_second = 5 →
  (initial_marked * second_capture) / marked_in_second = 420 :=
by
  sorry

#check estimate_rabbit_population

end NUMINAMATH_CALUDE_estimate_rabbit_population_l81_8151


namespace NUMINAMATH_CALUDE_rectangle_ratio_l81_8192

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) -- Outer square side length
  (h5 : x + y = 3*s) -- Outer square side length
  (h6 : (3*s)^2 = 9*s^2) -- Area of outer square is 9 times inner square
  : x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l81_8192


namespace NUMINAMATH_CALUDE_work_completion_theorem_l81_8114

/-- Given a work that could be finished in 12 days, and was actually finished in 9 days
    after 10 more men joined, prove that the original number of men employed was 30. -/
theorem work_completion_theorem (original_days : ℕ) (actual_days : ℕ) (additional_men : ℕ) :
  original_days = 12 →
  actual_days = 9 →
  additional_men = 10 →
  ∃ (original_men : ℕ), original_men * original_days = (original_men + additional_men) * actual_days ∧ original_men = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l81_8114


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l81_8128

def polynomial (x : ℝ) : ℝ := 4 * (2 * x^6 + 9 * x^3 - 6) + 8 * (x^4 - 6 * x^2 + 3)

theorem sum_of_coefficients : (polynomial 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l81_8128


namespace NUMINAMATH_CALUDE_non_congruent_squares_count_l81_8138

/-- Represents a square on a lattice grid -/
structure LatticeSquare where
  -- We'll represent a square by its side length and orientation
  side_length : ℕ
  is_rotated : Bool

/-- The size of the grid -/
def grid_size : ℕ := 6

/-- Counts the number of squares of a given side length on the grid -/
def count_squares (side_length : ℕ) : ℕ :=
  (grid_size - side_length) * (grid_size - side_length)

/-- Counts all non-congruent squares on the 6x6 grid -/
def count_all_squares : ℕ :=
  -- Count regular squares
  (count_squares 1) + (count_squares 2) + (count_squares 3) + 
  (count_squares 4) + (count_squares 5) +
  -- Count rotated squares (same formula as regular squares)
  (count_squares 1) + (count_squares 2) + (count_squares 3) + 
  (count_squares 4) + (count_squares 5)

/-- Theorem: The number of non-congruent squares on a 6x6 grid is 110 -/
theorem non_congruent_squares_count : count_all_squares = 110 := by
  sorry

end NUMINAMATH_CALUDE_non_congruent_squares_count_l81_8138


namespace NUMINAMATH_CALUDE_solve_for_z_l81_8157

theorem solve_for_z (x y z : ℚ) : x = 11 → y = -8 → 2 * x - 3 * z = 5 * y → z = 62 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l81_8157


namespace NUMINAMATH_CALUDE_least_valid_number_l81_8155

def is_valid (n : ℕ) : Prop :=
  n % 11 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 7 = 1

theorem least_valid_number : ∀ m : ℕ, m < 2521 → ¬(is_valid m) ∧ is_valid 2521 :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l81_8155


namespace NUMINAMATH_CALUDE_division_problem_l81_8195

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12) 
  (h2 : (x : ℝ) % (y : ℝ) = 5.76) : 
  y = 48 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l81_8195


namespace NUMINAMATH_CALUDE_intersection_slope_l81_8161

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y + 4 = 0

/-- Theorem stating that the slope of the line formed by the intersection points of the two circles is -1 -/
theorem intersection_slope : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle1 x1 y1 ∧ circle1 x2 y2 ∧ 
    circle2 x1 y1 ∧ circle2 x2 y2 ∧ 
    x1 ≠ x2 ∧
    (y2 - y1) / (x2 - x1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l81_8161


namespace NUMINAMATH_CALUDE_octal_subtraction_l81_8143

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- Theorem: 53₈ - 27₈ = 24₈ in base 8 --/
theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 53 - octal_to_decimal 27) = 24 := by sorry

end NUMINAMATH_CALUDE_octal_subtraction_l81_8143


namespace NUMINAMATH_CALUDE_accurate_to_thousands_l81_8160

/-- Represents a large number in millions with one decimal place -/
structure LargeNumber where
  whole : ℕ
  decimal : ℕ
  inv_ten : decimal < 10

/-- Converts a LargeNumber to its full integer representation -/
def LargeNumber.toInt (n : LargeNumber) : ℕ := n.whole * 1000000 + n.decimal * 100000

/-- Represents the place value in a number system -/
inductive PlaceValue
  | Thousands
  | Hundreds
  | Tens
  | Ones
  | Tenths
  | Hundredths

/-- Determines the smallest accurately represented place value for a given LargeNumber -/
def smallestAccuratePlaceValue (n : LargeNumber) : PlaceValue := 
  if n.decimal % 10 = 0 then PlaceValue.Hundreds else PlaceValue.Thousands

theorem accurate_to_thousands (n : LargeNumber) 
  (h : n.whole = 42 ∧ n.decimal = 3) : 
  smallestAccuratePlaceValue n = PlaceValue.Thousands := by
  sorry

end NUMINAMATH_CALUDE_accurate_to_thousands_l81_8160


namespace NUMINAMATH_CALUDE_f_max_at_two_l81_8120

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

-- State the theorem
theorem f_max_at_two :
  ∃ (max : ℝ), f 2 = max ∧ ∀ x, f x ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_f_max_at_two_l81_8120


namespace NUMINAMATH_CALUDE_eighth_grade_ratio_l81_8193

theorem eighth_grade_ratio (total_students : Nat) (girls : Nat) :
  total_students = 68 →
  girls = 28 →
  (total_students - girls : Nat) / girls = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_eighth_grade_ratio_l81_8193


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l81_8182

/-- The y-intercept of the line 4x + 7y = 28 is the point (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 4 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l81_8182


namespace NUMINAMATH_CALUDE_function_derivative_existence_l81_8167

open Set

theorem function_derivative_existence (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a) (h2 : a < b)
  (h3 : ContinuousOn f (Icc a b))
  (h4 : DifferentiableOn ℝ f (Ioo a b)) :
  ∃ c ∈ Ioo a b, deriv f c = 1 / (a - c) + 1 / (b - c) + 1 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_existence_l81_8167


namespace NUMINAMATH_CALUDE_complex_square_simplification_l81_8165

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 25 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l81_8165


namespace NUMINAMATH_CALUDE_game_ends_in_finite_steps_l81_8113

/-- Represents the state of the game at any point -/
structure GameState where
  m : ℕ+  -- Player A's number
  n : ℕ+  -- Player B's number
  t : ℤ   -- The other number written by the umpire
  k : ℕ   -- The current question number

/-- Represents whether a player knows the other's number -/
def knows (state : GameState) (player : Bool) : Prop :=
  if player 
  then state.t ≤ state.m + state.n - state.n / state.k
  else state.t ≥ state.m + state.n + state.n / state.k

/-- The main theorem stating that the game will end after a finite number of questions -/
theorem game_ends_in_finite_steps : 
  ∀ (initial_state : GameState), 
  ∃ (final_state : GameState), 
  (knows final_state true ∨ knows final_state false) ∧ 
  final_state.k ≥ initial_state.k :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_finite_steps_l81_8113


namespace NUMINAMATH_CALUDE_midnight_temperature_l81_8117

/-- 
Given an initial temperature, a temperature rise, and a temperature drop,
calculate the final temperature.
-/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/--
Theorem: Given the specific temperature changes in the problem,
the final temperature is -4°C.
-/
theorem midnight_temperature : final_temperature (-3) 6 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l81_8117


namespace NUMINAMATH_CALUDE_square_field_area_l81_8156

theorem square_field_area (wire_length : ℝ) (wire_turns : ℕ) (field_side : ℝ) : 
  wire_length = (4 * field_side * wire_turns) → 
  wire_length = 15840 → 
  wire_turns = 15 → 
  field_side * field_side = 69696 := by
sorry

end NUMINAMATH_CALUDE_square_field_area_l81_8156
