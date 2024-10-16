import Mathlib

namespace NUMINAMATH_CALUDE_midpoint_figure_area_l1522_152223

/-- The area of a figure in a 6x6 grid formed by connecting midpoints to the center -/
theorem midpoint_figure_area : 
  ∀ (grid_size : ℕ) (center_square_area corner_triangle_area : ℝ),
  grid_size = 6 →
  center_square_area = 4.5 →
  corner_triangle_area = 4.5 →
  center_square_area + 4 * corner_triangle_area = 22.5 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_figure_area_l1522_152223


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l1522_152233

/-- Represents the colors of balls in the urn -/
inductive Color
| Red
| Blue
| Green

/-- Represents the state of the urn -/
structure UrnState where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The operation of drawing a ball and adding a matching one -/
def draw_and_add (state : UrnState) : UrnState → Prop :=
  sorry

/-- Performs the draw_and_add operation n times -/
def perform_operations (n : ℕ) (initial : UrnState) : UrnState → Prop :=
  sorry

/-- The probability of a specific final state after n operations -/
noncomputable def probability_of_state (n : ℕ) (initial final : UrnState) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initial_state : UrnState := ⟨2, 1, 0⟩
  let final_state : UrnState := ⟨3, 3, 3⟩
  probability_of_state 6 initial_state final_state = 2/7 :=
by
  sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l1522_152233


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1522_152244

theorem largest_integer_satisfying_inequality :
  ∀ n : ℕ, (1 / 5 : ℚ) + (n : ℚ) / 8 < 9 / 5 ↔ n ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1522_152244


namespace NUMINAMATH_CALUDE_one_weighing_sufficient_l1522_152224

/-- Represents the types of balls -/
inductive BallType
| Aluminum
| Duralumin

/-- Represents a collection of balls -/
structure BallCollection where
  aluminum : ℕ
  duralumin : ℕ

/-- The mass of a ball collection -/
def mass (bc : BallCollection) : ℚ :=
  10 * bc.aluminum + 99/10 * bc.duralumin

theorem one_weighing_sufficient :
  ∃ (group1 group2 : BallCollection),
    group1.aluminum + group1.duralumin = 1000 ∧
    group2.aluminum + group2.duralumin = 1000 ∧
    group1.aluminum + group2.aluminum = 1000 ∧
    group1.duralumin + group2.duralumin = 1000 ∧
    mass group1 ≠ mass group2 :=
sorry

end NUMINAMATH_CALUDE_one_weighing_sufficient_l1522_152224


namespace NUMINAMATH_CALUDE_intersection_value_l1522_152248

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  equation : PolarPoint → Prop

def C₁ : PolarCurve :=
  { equation := fun p => p.ρ * (Real.cos p.θ + Real.sin p.θ) = 1 }

def C₂ (a : ℝ) : PolarCurve :=
  { equation := fun p => p.ρ = a }

def onPolarAxis (p : PolarPoint) : Prop :=
  p.θ = 0 ∨ p.θ = Real.pi

theorem intersection_value (a : ℝ) (h₁ : a > 0) :
  (∃ p : PolarPoint, C₁.equation p ∧ (C₂ a).equation p ∧ onPolarAxis p) →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l1522_152248


namespace NUMINAMATH_CALUDE_function_zero_in_interval_l1522_152222

/-- The function f(x) = 2ax^2 + 2x - 3 - a has a zero in the interval [-1, 1] 
    if and only if a ≤ (-3 - √7)/2 or a ≥ 1 -/
theorem function_zero_in_interval (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + 2 * x - 3 - a = 0) ↔ 
  (a ≤ (-3 - Real.sqrt 7) / 2 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_function_zero_in_interval_l1522_152222


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1522_152249

theorem unique_quadratic_solution (a c : ℤ) : 
  (∃! x : ℝ, a * x^2 + 36 * x + c = 0) →
  a + c = 37 →
  a < c →
  (a = 12 ∧ c = 25) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1522_152249


namespace NUMINAMATH_CALUDE_projection_theorem_l1522_152260

/-- A plane passing through the origin -/
structure Plane where
  normal : ℝ × ℝ × ℝ

/-- Projection of a vector onto a plane -/
def project (v : ℝ × ℝ × ℝ) (p : Plane) : ℝ × ℝ × ℝ := sorry

/-- The plane Q passing through the origin -/
def Q : Plane := sorry

theorem projection_theorem :
  project (6, 4, 6) Q = (4, 6, 2) →
  project (5, 2, 8) Q = (11/6, 31/6, 10/6) := by sorry

end NUMINAMATH_CALUDE_projection_theorem_l1522_152260


namespace NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l1522_152256

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem derivative_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f_derivative a x = f_derivative a (-x)) →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l1522_152256


namespace NUMINAMATH_CALUDE_point_on_line_l1522_152215

/-- Given a line passing through points (3, -5) and (5, 1), 
    prove that any point (7, y) on this line must have y = 7. -/
theorem point_on_line (y : ℝ) : 
  (∀ (x : ℝ), (x - 3) * (1 - (-5)) = (y - (-5)) * (5 - 3) → x = 7) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1522_152215


namespace NUMINAMATH_CALUDE_rectangular_room_tiles_l1522_152279

/-- Calculates the number of tiles touching the walls in a rectangular room -/
def tiles_touching_walls (length width : ℕ) : ℕ :=
  2 * length + 2 * width - 4

theorem rectangular_room_tiles (length width : ℕ) 
  (h_length : length = 10) (h_width : width = 5) : 
  tiles_touching_walls length width = 26 := by
  sorry

#eval tiles_touching_walls 10 5

end NUMINAMATH_CALUDE_rectangular_room_tiles_l1522_152279


namespace NUMINAMATH_CALUDE_smallest_impossible_score_l1522_152201

def dart_scores : Set ℕ := {0, 1, 3, 7, 8, 12}

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ dart_scores ∧ b ∈ dart_scores ∧ c ∈ dart_scores ∧ a + b + c = n

theorem smallest_impossible_score :
  (∀ m : ℕ, m < 22 → is_valid_sum m) ∧ ¬is_valid_sum 22 :=
sorry

end NUMINAMATH_CALUDE_smallest_impossible_score_l1522_152201


namespace NUMINAMATH_CALUDE_f_increasing_and_range_l1522_152211

-- Define the function f and its properties
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x > 0) ∧
  (f (-1) = -2)

-- Theorem statement
theorem f_increasing_and_range (f : ℝ → ℝ) (hf : f_properties f) :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range (fun x => f x) ∩ Set.Icc (-2) 1 = Set.Icc (-4) 2) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_and_range_l1522_152211


namespace NUMINAMATH_CALUDE_bbq_cooking_time_l1522_152278

/-- Calculates the time required to cook burgers for a BBQ --/
theorem bbq_cooking_time 
  (cooking_time_per_side : ℕ) 
  (grill_capacity : ℕ) 
  (total_guests : ℕ) 
  (guests_wanting_two : ℕ) 
  (guests_wanting_one : ℕ) 
  (h1 : cooking_time_per_side = 4)
  (h2 : grill_capacity = 5)
  (h3 : total_guests = 30)
  (h4 : guests_wanting_two = total_guests / 2)
  (h5 : guests_wanting_one = total_guests / 2)
  : (((guests_wanting_two * 2 + guests_wanting_one) / grill_capacity) * 
     (cooking_time_per_side * 2)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_bbq_cooking_time_l1522_152278


namespace NUMINAMATH_CALUDE_sum_abc_equals_33_l1522_152242

theorem sum_abc_equals_33 
  (a b c N : ℕ+) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : N = 5 * a + 3 * b + 5 * c)
  (h_eq2 : N = 4 * a + 5 * b + 4 * c)
  (h_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_33_l1522_152242


namespace NUMINAMATH_CALUDE_toffee_cost_l1522_152228

/-- The cost of 1 kg of toffees in rubles -/
def cost_per_kg : ℝ := 1.11

/-- The cost of 9 kg of toffees is less than 10 rubles -/
axiom nine_kg_cost : cost_per_kg * 9 < 10

/-- The cost of 10 kg of toffees is more than 11 rubles -/
axiom ten_kg_cost : cost_per_kg * 10 > 11

/-- Theorem: The cost of 1 kg of toffees is 1.11 rubles -/
theorem toffee_cost : cost_per_kg = 1.11 := by
  sorry

end NUMINAMATH_CALUDE_toffee_cost_l1522_152228


namespace NUMINAMATH_CALUDE_sallys_nickels_l1522_152259

theorem sallys_nickels (x : ℕ) : x + 9 + 2 = 18 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_l1522_152259


namespace NUMINAMATH_CALUDE_driving_distance_difference_l1522_152231

/-- Represents a driver's journey --/
structure Journey where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem statement --/
theorem driving_distance_difference 
  (liam : Journey) 
  (zoe : Journey) 
  (mia : Journey) 
  (h1 : zoe.time = liam.time + 2)
  (h2 : zoe.speed = liam.speed + 7)
  (h3 : zoe.distance = liam.distance + 80)
  (h4 : mia.time = liam.time + 3)
  (h5 : mia.speed = liam.speed + 15)
  (h6 : ∀ j : Journey, j.distance = j.speed * j.time) :
  mia.distance - liam.distance = 243 := by
  sorry

end NUMINAMATH_CALUDE_driving_distance_difference_l1522_152231


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1522_152295

theorem quadratic_roots_relation (a b : ℝ) (r₁ r₂ : ℂ) : 
  (∀ x : ℂ, x^2 + a*x + b = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ x : ℂ, x^2 + b*x + a = 0 ↔ x = 3*r₁ ∨ x = 3*r₂) →
  a/b = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1522_152295


namespace NUMINAMATH_CALUDE_rs_length_l1522_152257

/-- Triangle PQR with point S on PR -/
structure TrianglePQR where
  /-- Length of PQ -/
  PQ : ℝ
  /-- Length of QR -/
  QR : ℝ
  /-- Length of PS -/
  PS : ℝ
  /-- Length of QS -/
  QS : ℝ
  /-- PQ equals QR -/
  PQ_eq_QR : PQ = QR
  /-- PQ equals 8 -/
  PQ_eq_8 : PQ = 8
  /-- PS equals 10 -/
  PS_eq_10 : PS = 10
  /-- QS equals 5 -/
  QS_eq_5 : QS = 5

/-- The length of RS in the given triangle configuration is 3.5 -/
theorem rs_length (t : TrianglePQR) : ∃ RS : ℝ, RS = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_rs_length_l1522_152257


namespace NUMINAMATH_CALUDE_sum_of_disk_areas_l1522_152291

/-- The sum of areas of 16 congruent disks arranged on a unit circle --/
theorem sum_of_disk_areas (n : ℕ) (r : ℝ) : 
  n = 16 → 
  (2 * n * r : ℝ) = 2 * π → 
  (n : ℝ) * π * r^2 = 48 * π - 32 * π * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_disk_areas_l1522_152291


namespace NUMINAMATH_CALUDE_a_10_has_1000_nines_l1522_152285

/-- The sequence a(n) defined recursively -/
def a : ℕ → ℕ
  | 0 => 9
  | (n + 1) => 3 * (a n)^4 + 4 * (a n)^3

/-- Function to count the number of trailing nines in a natural number -/
def count_trailing_nines (n : ℕ) : ℕ := sorry

/-- Theorem stating that a(10) has at least 1000 trailing nines -/
theorem a_10_has_1000_nines : count_trailing_nines (a 10) ≥ 1000 := by sorry

end NUMINAMATH_CALUDE_a_10_has_1000_nines_l1522_152285


namespace NUMINAMATH_CALUDE_evaluate_expression_l1522_152299

theorem evaluate_expression (a x : ℝ) (h : x = a + 10) : (x - a + 3) * (x - a - 2) = 104 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1522_152299


namespace NUMINAMATH_CALUDE_jess_height_l1522_152213

/-- Given the heights of Jana, Kelly, and Jess, prove that Jess is 72 inches tall. -/
theorem jess_height (jana kelly jess : ℕ) 
  (h1 : jana = kelly + 5)
  (h2 : kelly = jess - 3)
  (h3 : jana = 74) : 
  jess = 72 := by
  sorry

end NUMINAMATH_CALUDE_jess_height_l1522_152213


namespace NUMINAMATH_CALUDE_full_price_revenue_l1522_152253

/-- Represents the fundraiser scenario -/
structure Fundraiser where
  total_tickets : ℕ
  total_revenue : ℚ
  full_price : ℚ
  full_price_tickets : ℕ

/-- The fundraiser satisfies the given conditions -/
def valid_fundraiser (f : Fundraiser) : Prop :=
  f.total_tickets = 180 ∧
  f.total_revenue = 2600 ∧
  f.full_price > 0 ∧
  f.full_price_tickets ≤ f.total_tickets ∧
  f.full_price_tickets * f.full_price + (f.total_tickets - f.full_price_tickets) * (f.full_price / 3) = f.total_revenue

/-- The theorem stating that the revenue from full-price tickets is $975 -/
theorem full_price_revenue (f : Fundraiser) (h : valid_fundraiser f) : 
  f.full_price_tickets * f.full_price = 975 := by
  sorry

end NUMINAMATH_CALUDE_full_price_revenue_l1522_152253


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_1260_l1522_152265

theorem sum_of_extreme_prime_factors_1260 : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  p ∣ 1260 ∧ q ∣ 1260 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 1260 → p ≤ r ∧ r ≤ q) ∧
  p + q = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_1260_l1522_152265


namespace NUMINAMATH_CALUDE_flow_rate_increase_l1522_152238

/-- Proves that the percentage increase in flow rate from the first to the second hour is 50% -/
theorem flow_rate_increase (r1 r2 r3 : ℝ) : 
  r2 = 36 →  -- Second hour flow rate
  r3 = 1.25 * r2 →  -- Third hour flow rate is 25% more than second
  r1 + r2 + r3 = 105 →  -- Total flow for all three hours
  r1 < r2 →  -- Second hour rate faster than first
  (r2 - r1) / r1 * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_flow_rate_increase_l1522_152238


namespace NUMINAMATH_CALUDE_angle4_value_l1522_152210

-- Define the angles
def angle1 : ℝ := sorry
def angle2 : ℝ := sorry
def angle3 : ℝ := sorry
def angle4 : ℝ := sorry
def angleA : ℝ := 80
def angleB : ℝ := 50

-- State the theorem
theorem angle4_value :
  (angle1 + angle2 = 180) →
  (angle3 = angle4) →
  (angle1 + angleA + angleB = 180) →
  (angle2 + angle3 + angle4 = 180) →
  angle4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_angle4_value_l1522_152210


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l1522_152266

/-- The probability of selecting three plates of the same color from 7 green and 5 yellow plates. -/
theorem same_color_plate_probability :
  let total_plates : ℕ := 7 + 5
  let green_plates : ℕ := 7
  let yellow_plates : ℕ := 5
  let total_combinations : ℕ := Nat.choose total_plates 3
  let green_combinations : ℕ := Nat.choose green_plates 3
  let yellow_combinations : ℕ := Nat.choose yellow_plates 3
  let same_color_combinations : ℕ := green_combinations + yellow_combinations
  (same_color_combinations : ℚ) / total_combinations = 9 / 44 := by
sorry


end NUMINAMATH_CALUDE_same_color_plate_probability_l1522_152266


namespace NUMINAMATH_CALUDE_first_sample_is_three_l1522_152239

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  totalPopulation : Nat
  sampleSize : Nat
  lastSample : Nat

/-- Calculates the first sample in a systematic sampling scenario -/
def firstSample (s : SystematicSampling) : Nat :=
  s.lastSample - (s.sampleSize - 1) * (s.totalPopulation / s.sampleSize)

/-- Theorem: In the given systematic sampling scenario, the first sample is 3 -/
theorem first_sample_is_three :
  let s : SystematicSampling := ⟨300, 60, 298⟩
  firstSample s = 3 := by sorry

end NUMINAMATH_CALUDE_first_sample_is_three_l1522_152239


namespace NUMINAMATH_CALUDE_james_fish_tanks_l1522_152282

def fish_tank_problem (num_tanks : ℕ) (fish_in_first_tank : ℕ) (total_fish : ℕ) : Prop :=
  ∃ (num_double_tanks : ℕ),
    num_tanks = 1 + num_double_tanks ∧
    fish_in_first_tank = 20 ∧
    total_fish = fish_in_first_tank + num_double_tanks * (2 * fish_in_first_tank) ∧
    total_fish = 100

theorem james_fish_tanks :
  ∃ (num_tanks : ℕ), fish_tank_problem num_tanks 20 100 ∧ num_tanks = 3 :=
sorry

end NUMINAMATH_CALUDE_james_fish_tanks_l1522_152282


namespace NUMINAMATH_CALUDE_fund_raising_ratio_l1522_152264

def fund_raising (goal : ℕ) (ken_collection : ℕ) (excess : ℕ) : Prop :=
  ∃ (mary_collection scott_collection : ℕ),
    mary_collection = 5 * ken_collection ∧
    ∃ (k : ℕ), mary_collection = k * scott_collection ∧
    mary_collection + scott_collection + ken_collection = goal + excess ∧
    mary_collection / scott_collection = 3

theorem fund_raising_ratio :
  fund_raising 4000 600 600 :=
sorry

end NUMINAMATH_CALUDE_fund_raising_ratio_l1522_152264


namespace NUMINAMATH_CALUDE_perpendicular_impossibility_l1522_152219

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Line → Line → Point → Prop)
variable (non_coincident : Line → Line → Prop)

-- State the theorem
theorem perpendicular_impossibility
  (a b : Line) (α : Plane) (P : Point)
  (h1 : non_coincident a b)
  (h2 : perpendicular a α)
  (h3 : intersect a b P) :
  ¬ (perpendicular b α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_impossibility_l1522_152219


namespace NUMINAMATH_CALUDE_power_gt_one_iff_product_gt_zero_l1522_152204

theorem power_gt_one_iff_product_gt_zero {a b : ℝ} (ha : a > 0) (ha' : a ≠ 1) :
  a^b > 1 ↔ (a - 1) * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_gt_one_iff_product_gt_zero_l1522_152204


namespace NUMINAMATH_CALUDE_expression_evaluation_l1522_152274

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (x - 1) / (x - 2) * ((x^2 - 4) / (x^2 - 2*x + 1)) - 2 / (x - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1522_152274


namespace NUMINAMATH_CALUDE_point_on_h_graph_and_coordinate_sum_l1522_152208

/-- Given a function g where g(4) = 8, and h defined as h(x) = 2(g(x))^3,
    prove that (4,1024) is on the graph of h and the sum of its coordinates is 1028 -/
theorem point_on_h_graph_and_coordinate_sum 
  (g : ℝ → ℝ) (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = 2 * (g x)^3)
  (g_value : g 4 = 8) :
  h 4 = 1024 ∧ 4 + 1024 = 1028 := by
  sorry

end NUMINAMATH_CALUDE_point_on_h_graph_and_coordinate_sum_l1522_152208


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l1522_152250

/-- Calculates the number of degrees for cherry pie in a pie chart given the class preferences. -/
theorem cherry_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ)
  (h1 : total_students = 48)
  (h2 : chocolate = 15)
  (h3 : apple = 10)
  (h4 : blueberry = 9)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  (((total_students - (chocolate + apple + blueberry)) / 2 : ℚ) / total_students) * 360 = 52.5 := by
  sorry

#eval ((7 : ℚ) / 48) * 360  -- Should output 52.5

end NUMINAMATH_CALUDE_cherry_pie_degrees_l1522_152250


namespace NUMINAMATH_CALUDE_remainder_problem_l1522_152294

theorem remainder_problem (x : ℤ) : 
  x % 82 = 5 → (x + 13) % 41 = 18 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1522_152294


namespace NUMINAMATH_CALUDE_number_division_l1522_152262

theorem number_division (x : ℝ) : x - 17 = 55 → x / 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l1522_152262


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_32pi_l1522_152272

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  PA : ℝ
  AB : ℝ
  BC : ℝ
  angleABC : ℝ

/-- The surface area of the circumscribed sphere of a triangular pyramid -/
def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating the surface area of the circumscribed sphere for the given pyramid -/
theorem circumscribed_sphere_surface_area_is_32pi :
  let pyramid : TriangularPyramid := {
    PA := 4,
    AB := 2,
    BC := 2,
    angleABC := 2 * Real.pi / 3  -- 120° in radians
  }
  circumscribedSphereSurfaceArea pyramid = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_32pi_l1522_152272


namespace NUMINAMATH_CALUDE_vikki_take_home_pay_l1522_152288

def weekly_pay_calculation (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : ℚ :=
  let total_earnings := hours_worked * hourly_rate
  let tax_deduction := total_earnings * tax_rate
  let insurance_deduction := total_earnings * insurance_rate
  let total_deductions := tax_deduction + insurance_deduction + union_dues
  total_earnings - total_deductions

theorem vikki_take_home_pay :
  weekly_pay_calculation 42 10 (20/100) (5/100) 5 = 310 := by
  sorry

end NUMINAMATH_CALUDE_vikki_take_home_pay_l1522_152288


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1522_152297

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (x, 2) (1, 6) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1522_152297


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1522_152230

def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1/4
  let n : ℕ := 5
  geometricSum a r n = 341/256 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1522_152230


namespace NUMINAMATH_CALUDE_minimum_square_side_l1522_152283

theorem minimum_square_side (area_min : ℝ) (side : ℝ) : 
  area_min = 625 → side^2 ≥ area_min → side ≥ 0 → side ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_square_side_l1522_152283


namespace NUMINAMATH_CALUDE_max_y_coordinate_difference_l1522_152209

-- Define the two functions
def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := 2 + x^2 + x^3

-- Define the intersection points
def intersection_points : Set ℝ := {x : ℝ | f x = g x}

-- Define the y-coordinates of the intersection points
def y_coordinates : Set ℝ := {y : ℝ | ∃ x ∈ intersection_points, f x = y}

-- Theorem statement
theorem max_y_coordinate_difference :
  ∃ (y1 y2 : ℝ), y1 ∈ y_coordinates ∧ y2 ∈ y_coordinates ∧
  ∀ (z1 z2 : ℝ), z1 ∈ y_coordinates → z2 ∈ y_coordinates →
  |y1 - y2| ≥ |z1 - z2| ∧ |y1 - y2| = 2 :=
sorry

end NUMINAMATH_CALUDE_max_y_coordinate_difference_l1522_152209


namespace NUMINAMATH_CALUDE_marcy_lip_gloss_tubs_l1522_152200

/-- The number of tubs of lip gloss Marcy needs to bring for a wedding -/
def tubs_of_lip_gloss (people : ℕ) (people_per_tube : ℕ) (tubes_per_tub : ℕ) : ℕ :=
  (people / people_per_tube) / tubes_per_tub

/-- Theorem: Marcy needs to bring 6 tubs of lip gloss for 36 people -/
theorem marcy_lip_gloss_tubs : tubs_of_lip_gloss 36 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_marcy_lip_gloss_tubs_l1522_152200


namespace NUMINAMATH_CALUDE_power_product_consecutive_integers_l1522_152286

theorem power_product_consecutive_integers (k : ℕ) : 
  (∃ (a b : ℕ), 2^a * 3^b = k * (k + 1)) ↔ k ∈ ({1, 2, 3, 8} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_product_consecutive_integers_l1522_152286


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1522_152255

/-- The average speed of a round trip where:
    - The total distance is 2m meters (m meters each way)
    - The northward journey is at 3 minutes per mile
    - The southward journey is at 3 miles per minute
    - 1 mile = 1609.34 meters
-/
theorem round_trip_average_speed (m : ℝ) :
  let meters_per_mile : ℝ := 1609.34
  let north_speed : ℝ := 1 / 3 -- miles per minute
  let south_speed : ℝ := 3 -- miles per minute
  let total_distance : ℝ := 2 * m / meters_per_mile -- in miles
  let north_time : ℝ := m / (meters_per_mile * north_speed) -- in minutes
  let south_time : ℝ := m / (meters_per_mile * south_speed) -- in minutes
  let total_time : ℝ := north_time + south_time -- in minutes
  let average_speed : ℝ := total_distance / (total_time / 60) -- in miles per hour
  average_speed = 60 := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1522_152255


namespace NUMINAMATH_CALUDE_workshop_sample_size_l1522_152235

/-- Calculates the sample size for a stratum in stratified sampling -/
def stratumSampleSize (totalPopulation : ℕ) (totalSampleSize : ℕ) (stratumSize : ℕ) : ℕ :=
  (totalSampleSize * stratumSize) / totalPopulation

theorem workshop_sample_size :
  let totalProducts : ℕ := 1024
  let sampleSize : ℕ := 64
  let workshopProduction : ℕ := 128
  stratumSampleSize totalProducts sampleSize workshopProduction = 8 := by
  sorry

end NUMINAMATH_CALUDE_workshop_sample_size_l1522_152235


namespace NUMINAMATH_CALUDE_exists_n_no_rational_solution_l1522_152229

-- Define a quadratic polynomial with real coefficients
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem exists_n_no_rational_solution (a b c : ℝ) :
  ∃ n : ℕ, ∀ x : ℚ, QuadraticPolynomial a b c x ≠ (1 : ℝ) / n := by
  sorry

end NUMINAMATH_CALUDE_exists_n_no_rational_solution_l1522_152229


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1522_152273

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l1522_152273


namespace NUMINAMATH_CALUDE_expression_equality_l1522_152284

theorem expression_equality (x : ℝ) (h1 : x^3 + 1 ≠ 0) (h2 : x^3 - 1 ≠ 0) : 
  ((x + 1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * 
  ((x - 1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1522_152284


namespace NUMINAMATH_CALUDE_compound_not_uniquely_determined_l1522_152234

/-- Represents a chemical compound -/
structure Compound where
  elements : List String
  mass_percentages : List Float
  mass_percentage_sum_eq_100 : mass_percentages.sum = 100

/-- A compound contains Cl with a mass percentage of 47.3% -/
def chlorine_compound : Compound := {
  elements := ["Cl", "Unknown"],
  mass_percentages := [47.3, 52.7],
  mass_percentage_sum_eq_100 := by sorry
}

/-- Predicate to check if a compound matches the given chlorine compound -/
def matches_chlorine_compound (c : Compound) : Prop :=
  "Cl" ∈ c.elements ∧ 47.3 ∈ c.mass_percentages

/-- Theorem stating that the compound cannot be uniquely determined -/
theorem compound_not_uniquely_determined :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ matches_chlorine_compound c1 ∧ matches_chlorine_compound c2 :=
by sorry

end NUMINAMATH_CALUDE_compound_not_uniquely_determined_l1522_152234


namespace NUMINAMATH_CALUDE_sum_of_solutions_equation_l1522_152246

theorem sum_of_solutions_equation : ∃ (x₁ x₂ : ℝ), 
  (4 * x₁ + 3) * (3 * x₁ - 7) = 0 ∧
  (4 * x₂ + 3) * (3 * x₂ - 7) = 0 ∧
  x₁ + x₂ = 19 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equation_l1522_152246


namespace NUMINAMATH_CALUDE_couscous_dishes_l1522_152218

/-- Calculates the number of dishes a restaurant can make from couscous shipments -/
theorem couscous_dishes (shipment1 shipment2 shipment3 pounds_per_dish : ℕ) :
  shipment1 = 7 →
  shipment2 = 13 →
  shipment3 = 45 →
  pounds_per_dish = 5 →
  (shipment1 + shipment2 + shipment3) / pounds_per_dish = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_couscous_dishes_l1522_152218


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1522_152281

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 256 * π →
    (4 / 3) * π * r^3 = (2048 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1522_152281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1522_152237

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 = 2 → a 4 = 6 → a 8 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1522_152237


namespace NUMINAMATH_CALUDE_factor_quadratic_l1522_152205

theorem factor_quadratic (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_quadratic_l1522_152205


namespace NUMINAMATH_CALUDE_complex_square_l1522_152203

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 5 - 3 * i) (h2 : i^2 = -1) :
  z^2 = 34 - 30 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l1522_152203


namespace NUMINAMATH_CALUDE_area_ratio_octagon_quadrilateral_l1522_152251

/-- Regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  area : ℝ

/-- Quadrilateral ACEG within the regular octagon -/
structure Quadrilateral where
  area : ℝ

/-- Theorem stating that the ratio of the quadrilateral area to the octagon area is √2/2 -/
theorem area_ratio_octagon_quadrilateral (octagon : RegularOctagon) (quad : Quadrilateral) :
  quad.area / octagon.area = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_octagon_quadrilateral_l1522_152251


namespace NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l1522_152293

theorem unique_m_satisfying_lcm_conditions : 
  ∃! m : ℕ+, (Nat.lcm 36 m.val = 180) ∧ (Nat.lcm m.val 45 = 225) ∧ (m.val = 25) :=
by sorry

end NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l1522_152293


namespace NUMINAMATH_CALUDE_janabel_sales_sum_l1522_152261

theorem janabel_sales_sum (n : ℕ) (a₁ d : ℤ) (h1 : n = 12) (h2 : a₁ = 1) (h3 : d = 4) :
  (n : ℤ) * (2 * a₁ + (n - 1) * d) / 2 = 276 :=
by sorry

end NUMINAMATH_CALUDE_janabel_sales_sum_l1522_152261


namespace NUMINAMATH_CALUDE_problem_solution_l1522_152289

theorem problem_solution : 
  let expr := (1 / (1 + 24 / 4) - 5 / 9) * (3 / (2 + 5 / 7)) / (2 / (3 + 3 / 4)) + 2.25
  ∀ A : ℝ, expr = 4 → (1 / (1 + 24 / A) - 5 / 9 = 1 / (1 + 24 / 4) - 5 / 9) → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1522_152289


namespace NUMINAMATH_CALUDE_special_function_range_l1522_152240

/-- A monotonically increasing function satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y) ∧
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (f 3 = 1)

/-- The theorem statement -/
theorem special_function_range (f : ℝ → ℝ) (hf : SpecialFunction f) :
  {x : ℝ | 0 < x ∧ f x + f (x - 8) ≤ 2} = Set.Ioo 8 9 := by
  sorry

end NUMINAMATH_CALUDE_special_function_range_l1522_152240


namespace NUMINAMATH_CALUDE_tommy_saw_13_cars_l1522_152247

/-- The number of cars Tommy saw -/
def num_cars : ℕ := 13

/-- The number of wheels per vehicle -/
def wheels_per_vehicle : ℕ := 4

/-- The number of trucks Tommy saw -/
def num_trucks : ℕ := 12

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := 100

theorem tommy_saw_13_cars :
  num_cars = (total_wheels - num_trucks * wheels_per_vehicle) / wheels_per_vehicle :=
by sorry

end NUMINAMATH_CALUDE_tommy_saw_13_cars_l1522_152247


namespace NUMINAMATH_CALUDE_remainder_98_power_50_mod_150_l1522_152276

theorem remainder_98_power_50_mod_150 : 98^50 ≡ 74 [ZMOD 150] := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_power_50_mod_150_l1522_152276


namespace NUMINAMATH_CALUDE_dan_limes_remaining_l1522_152298

theorem dan_limes_remaining (initial_limes given_limes : ℕ) : 
  initial_limes = 9 → given_limes = 4 → initial_limes - given_limes = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_limes_remaining_l1522_152298


namespace NUMINAMATH_CALUDE_xy_sum_problem_l1522_152202

theorem xy_sum_problem (x y : ℕ) 
  (pos_x : x > 0) (pos_y : y > 0)
  (bound_x : x < 30) (bound_y : y < 30)
  (eq_condition : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l1522_152202


namespace NUMINAMATH_CALUDE_solve_equation_l1522_152236

theorem solve_equation : ∃ x : ℝ, (12 : ℝ) ^ x * 6^2 / 432 = 144 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1522_152236


namespace NUMINAMATH_CALUDE_equation_solution_l1522_152267

theorem equation_solution (x : ℝ) (h : x ≠ 2/3) :
  (7*x + 2) / (3*x^2 + 7*x - 6) = 3*x / (3*x - 2) ↔ 
  x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1522_152267


namespace NUMINAMATH_CALUDE_income_distribution_l1522_152226

theorem income_distribution (income : ℝ) (h1 : income = 800000) : 
  let children_share := 0.2 * income * 3
  let wife_share := 0.3 * income
  let family_distribution := children_share + wife_share
  let remaining_after_family := income - family_distribution
  let orphan_donation := 0.05 * remaining_after_family
  let final_amount := remaining_after_family - orphan_donation
  final_amount = 76000 :=
by sorry

end NUMINAMATH_CALUDE_income_distribution_l1522_152226


namespace NUMINAMATH_CALUDE_meaningful_expression_l1522_152216

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1)^(0 : ℕ) / Real.sqrt (x + 2)) ↔ x > -2 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l1522_152216


namespace NUMINAMATH_CALUDE_wood_not_heavier_than_brick_l1522_152254

-- Define the mass of the block of wood in kg
def wood_mass_kg : ℝ := 8

-- Define the mass of the brick in g
def brick_mass_g : ℝ := 8000

-- Define the conversion factor from kg to g
def kg_to_g : ℝ := 1000

-- Theorem statement
theorem wood_not_heavier_than_brick : ¬(wood_mass_kg * kg_to_g > brick_mass_g) := by
  sorry

end NUMINAMATH_CALUDE_wood_not_heavier_than_brick_l1522_152254


namespace NUMINAMATH_CALUDE_alonzo_tomato_harvest_l1522_152290

/-- The amount of tomatoes (in kg) that Mr. Alonzo sold to Mrs. Maxwell -/
def sold_to_maxwell : ℝ := 125.5

/-- The amount of tomatoes (in kg) that Mr. Alonzo sold to Mr. Wilson -/
def sold_to_wilson : ℝ := 78

/-- The amount of tomatoes (in kg) that Mr. Alonzo has not sold -/
def not_sold : ℝ := 42

/-- The total amount of tomatoes (in kg) that Mr. Alonzo harvested -/
def total_harvested : ℝ := sold_to_maxwell + sold_to_wilson + not_sold

theorem alonzo_tomato_harvest : total_harvested = 245.5 := by
  sorry

end NUMINAMATH_CALUDE_alonzo_tomato_harvest_l1522_152290


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1522_152263

theorem product_remainder_mod_five :
  (1234 * 1987 * 2013 * 2021) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1522_152263


namespace NUMINAMATH_CALUDE_circle_constant_l1522_152206

/-- Theorem: For a circle with equation x^2 + 10x + y^2 + 8y + c = 0 and radius 5, the value of c is 16. -/
theorem circle_constant (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 8*y + c = 0 ↔ (x+5)^2 + (y+4)^2 = 25) → 
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_constant_l1522_152206


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1522_152252

theorem complex_modulus_problem (x y : ℝ) (h : (x + Complex.I) * x = 4 + 2 * y * Complex.I) :
  Complex.abs ((x + 4 * y * Complex.I) / (1 + Complex.I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1522_152252


namespace NUMINAMATH_CALUDE_school_children_count_l1522_152220

/-- Proves that the number of children in the school is 320 given the banana distribution conditions --/
theorem school_children_count : ∃ (C : ℕ) (B : ℕ), 
  B = 2 * C ∧                   -- Total bananas if each child gets 2
  B = 4 * (C - 160) ∧           -- Total bananas distributed to present children
  C = 320                       -- The number we want to prove
  := by sorry

end NUMINAMATH_CALUDE_school_children_count_l1522_152220


namespace NUMINAMATH_CALUDE_arithmetic_events_classification_l1522_152275

/-- Represents the sign of a number -/
inductive Sign
| Positive
| Negative

/-- Represents the result of an arithmetic operation -/
inductive Result
| Positive
| Negative

/-- Represents an arithmetic event -/
structure ArithmeticEvent :=
  (operation : String)
  (sign1 : Sign)
  (sign2 : Sign)
  (result : Result)

/-- Defines the four events described in the problem -/
def events : List ArithmeticEvent :=
  [ ⟨"Addition", Sign.Positive, Sign.Negative, Result.Negative⟩
  , ⟨"Subtraction", Sign.Positive, Sign.Negative, Result.Positive⟩
  , ⟨"Multiplication", Sign.Positive, Sign.Negative, Result.Positive⟩
  , ⟨"Division", Sign.Positive, Sign.Negative, Result.Negative⟩ ]

/-- Predicate to determine if an event is certain -/
def isCertain (e : ArithmeticEvent) : Prop :=
  e.operation = "Division" ∧ 
  e.sign1 ≠ e.sign2 ∧ 
  e.result = Result.Negative

/-- Predicate to determine if an event is random -/
def isRandom (e : ArithmeticEvent) : Prop :=
  (e.operation = "Addition" ∨ e.operation = "Subtraction") ∧
  e.sign1 ≠ e.sign2

theorem arithmetic_events_classification :
  ∃ (certain : ArithmeticEvent) (random1 random2 : ArithmeticEvent),
    certain ∈ events ∧
    random1 ∈ events ∧
    random2 ∈ events ∧
    isCertain certain ∧
    isRandom random1 ∧
    isRandom random2 ∧
    random1 ≠ random2 :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_events_classification_l1522_152275


namespace NUMINAMATH_CALUDE_place_value_ratio_l1522_152217

theorem place_value_ratio : 
  let number : ℝ := 37492.1053
  let ten_thousands_place_value : ℝ := 10000
  let ten_thousandths_place_value : ℝ := 0.0001
  ten_thousands_place_value / ten_thousandths_place_value = 100000000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l1522_152217


namespace NUMINAMATH_CALUDE_algebraic_identities_l1522_152241

variable (a b : ℝ)

theorem algebraic_identities :
  ((a - 2*b)^2 - (b - a)*(a + b) = 2*a^2 - 4*a*b + 3*b^2) ∧
  ((2*a - b)^2 * (2*a + b)^2 = 16*a^4 - 8*a^2*b^2 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l1522_152241


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1522_152212

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60. -/
theorem first_term_of_geometric_series : ∀ (a : ℝ),
  (∑' n, a * (1/4)^n) = 80 → a = 60 := by sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1522_152212


namespace NUMINAMATH_CALUDE_units_digit_problem_l1522_152225

def geometric_sum (a r : ℕ) (n : ℕ) : ℕ := 
  a * (r^(n+1) - 1) / (r - 1)

theorem units_digit_problem : 
  (2 * geometric_sum 1 3 9) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1522_152225


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1522_152292

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1522_152292


namespace NUMINAMATH_CALUDE_complex_product_real_implies_sum_modulus_l1522_152271

theorem complex_product_real_implies_sum_modulus (a : ℝ) : 
  let z₁ : ℂ := 2 + I
  let z₂ : ℂ := a + 3*I
  (z₁ * z₂).im = 0 → Complex.abs (z₁ + z₂) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_implies_sum_modulus_l1522_152271


namespace NUMINAMATH_CALUDE_teresas_colored_pencils_l1522_152207

/-- Given information about Teresa's pencils and her siblings, prove the number of colored pencils she has. -/
theorem teresas_colored_pencils 
  (black_pencils : ℕ) 
  (num_siblings : ℕ) 
  (pencils_per_sibling : ℕ) 
  (pencils_kept : ℕ) 
  (h1 : black_pencils = 35)
  (h2 : num_siblings = 3)
  (h3 : pencils_per_sibling = 13)
  (h4 : pencils_kept = 10) :
  black_pencils + (num_siblings * pencils_per_sibling + pencils_kept) - black_pencils = 14 :=
by sorry

end NUMINAMATH_CALUDE_teresas_colored_pencils_l1522_152207


namespace NUMINAMATH_CALUDE_unique_valid_x_l1522_152269

def is_valid_x (x : ℕ) : Prop :=
  x > 4 ∧ (x + 4) * (x - 4) * (x^3 + 25) < 1000

theorem unique_valid_x : ∃! x : ℕ, is_valid_x x :=
sorry

end NUMINAMATH_CALUDE_unique_valid_x_l1522_152269


namespace NUMINAMATH_CALUDE_rectangle_hexagon_apothem_comparison_l1522_152221

theorem rectangle_hexagon_apothem_comparison :
  ∀ (w l : ℝ) (s : ℝ),
    w > 0 ∧ l > 0 ∧ s > 0 →
    l = 3 * w →
    w * l = 2 * (w + l) →
    3 * Real.sqrt 3 / 2 * s^2 = 6 * s →
    w / 2 = 2/3 * (s * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_hexagon_apothem_comparison_l1522_152221


namespace NUMINAMATH_CALUDE_a_used_car_for_seven_hours_l1522_152227

/-- Represents the car hire scenario -/
structure CarHire where
  totalCost : ℕ
  bHours : ℕ
  bCost : ℕ
  cHours : ℕ

/-- Calculates the number of hours A used the car -/
def aHours (hire : CarHire) : ℕ :=
  (hire.totalCost - hire.bCost - (hire.cHours * hire.bCost / hire.bHours)) / (hire.bCost / hire.bHours)

/-- Theorem stating that A used the car for 7 hours given the conditions -/
theorem a_used_car_for_seven_hours :
  let hire := CarHire.mk 520 8 160 11
  aHours hire = 7 := by
  sorry


end NUMINAMATH_CALUDE_a_used_car_for_seven_hours_l1522_152227


namespace NUMINAMATH_CALUDE_gcf_of_lcms_equals_15_l1522_152277

theorem gcf_of_lcms_equals_15 : Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_equals_15_l1522_152277


namespace NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l1522_152232

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (x y : ℝ) : ℝ := |y|

/-- The theorem stating that the distance from (-2, -√5) to the x-axis is √5 -/
theorem distance_from_point_to_x_axis :
  distance_to_x_axis (-2 : ℝ) (-Real.sqrt 5) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_x_axis_l1522_152232


namespace NUMINAMATH_CALUDE_reservoir_water_amount_l1522_152287

/-- Represents the amount of water in million gallons -/
@[ext] structure WaterAmount where
  value : ℝ

/-- Represents a reservoir with its capacity and current amount -/
structure Reservoir where
  capacity : WaterAmount
  current_amount : WaterAmount
  normal_level : WaterAmount

/-- Conditions for the reservoir problem -/
def ReservoirConditions (r : Reservoir) : Prop :=
  r.current_amount.value = 2 * r.normal_level.value ∧
  r.current_amount.value = 0.7 * r.capacity.value ∧
  r.capacity.value = r.normal_level.value + 10

/-- The theorem stating the amount of water in the reservoir at the end of the month -/
theorem reservoir_water_amount (r : Reservoir) 
  (h : ReservoirConditions r) : 
  ∃ ε > 0, |r.current_amount.value - 10.766| < ε :=
sorry

end NUMINAMATH_CALUDE_reservoir_water_amount_l1522_152287


namespace NUMINAMATH_CALUDE_expression_simplification_l1522_152296

theorem expression_simplification (a b : ℝ) 
  (h1 : a ≠ b/2) (h2 : a ≠ -b/2) (h3 : a ≠ -b) (h4 : a ≠ 0) (h5 : b ≠ 0) :
  (((a - b) / (2*a - b) - (a^2 + b^2 + a) / (2*a^2 + a*b - b^2)) / 
   ((4*b^4 + 4*a*b^2 + a^2) / (2*b^2 + a))) * (b^2 + b + a*b + a) = 
  (b + 1) / (b - 2*a) := by
sorry


end NUMINAMATH_CALUDE_expression_simplification_l1522_152296


namespace NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l1522_152258

/-- Given that x varies as the fourth power of y and y varies as the cube root of z,
    prove that x varies as the (4/3)th power of z. -/
theorem x_varies_as_four_thirds_power_of_z 
  (k : ℝ) (j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_four_thirds_power_of_z_l1522_152258


namespace NUMINAMATH_CALUDE_sandy_money_left_l1522_152270

/-- The amount of money Sandy has left after buying a pie -/
def money_left (initial_amount pie_cost : ℕ) : ℕ :=
  initial_amount - pie_cost

/-- Theorem: Sandy has 57 dollars left after buying the pie -/
theorem sandy_money_left :
  money_left 63 6 = 57 :=
by sorry

end NUMINAMATH_CALUDE_sandy_money_left_l1522_152270


namespace NUMINAMATH_CALUDE_apartments_greater_than_scales_l1522_152243

theorem apartments_greater_than_scales (houses : ℕ) (K A P C : ℕ) :
  houses > 0 ∧ K > 0 ∧ A > 0 ∧ P > 0 ∧ C > 0 →  -- All quantities are positive
  K * A * P > A * P * C →                      -- Fish in house > scales in apartment
  K > C                                        -- Apartments in house > scales on fish
  := by sorry

end NUMINAMATH_CALUDE_apartments_greater_than_scales_l1522_152243


namespace NUMINAMATH_CALUDE_band_to_orchestra_ratio_l1522_152268

theorem band_to_orchestra_ratio : 
  ∀ (orchestra_students band_students choir_boys choir_girls total_students : ℕ),
    orchestra_students = 20 →
    choir_boys = 12 →
    choir_girls = 16 →
    total_students = 88 →
    total_students = orchestra_students + band_students + choir_boys + choir_girls →
    band_students = 2 * orchestra_students :=
by
  sorry

end NUMINAMATH_CALUDE_band_to_orchestra_ratio_l1522_152268


namespace NUMINAMATH_CALUDE_smallest_divisible_by_five_million_l1522_152245

def geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem smallest_divisible_by_five_million :
  let a₁ := 2
  let a₂ := 70
  let r := a₂ / a₁
  ∀ n : ℕ, n > 0 →
    (is_divisible_by (geometric_sequence a₁ r n) 5000000 ∧
     ∀ m : ℕ, 0 < m → m < n →
       ¬ is_divisible_by (geometric_sequence a₁ r m) 5000000) →
    n = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_five_million_l1522_152245


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1522_152280

/-- Given a cubic equation ax^3 + bx^2 + cx + d = 0 with a ≠ 0 and two roots 1 + i and 1 - i,
    prove that (b + c) / a = 2 -/
theorem cubic_root_sum (a b c d : ℂ) : 
  a ≠ 0 → 
  (∃ r : ℂ, a * (r^3) + b * (r^2) + c * r + d = 0) →
  (a * ((1 + Complex.I)^3) + b * ((1 + Complex.I)^2) + c * (1 + Complex.I) + d = 0) →
  (a * ((1 - Complex.I)^3) + b * ((1 - Complex.I)^2) + c * (1 - Complex.I) + d = 0) →
  (b + c) / a = 2 := by
  sorry

#check cubic_root_sum

end NUMINAMATH_CALUDE_cubic_root_sum_l1522_152280


namespace NUMINAMATH_CALUDE_symmetry_implies_congruence_l1522_152214

/-- Two shapes in a plane -/
structure Shape : Type :=
  -- Define necessary properties of a shape

/-- Line of symmetry between two shapes -/
structure SymmetryLine : Type :=
  -- Define necessary properties of a symmetry line

/-- Symmetry relation between two shapes about a line -/
def symmetrical (s1 s2 : Shape) (l : SymmetryLine) : Prop :=
  sorry

/-- Congruence relation between two shapes -/
def congruent (s1 s2 : Shape) : Prop :=
  sorry

/-- Theorem: If two shapes are symmetrical about a line, they are congruent -/
theorem symmetry_implies_congruence (s1 s2 : Shape) (l : SymmetryLine) :
  symmetrical s1 s2 l → congruent s1 s2 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_congruence_l1522_152214
