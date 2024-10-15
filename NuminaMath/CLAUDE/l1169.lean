import Mathlib

namespace NUMINAMATH_CALUDE_rain_probability_l1169_116934

theorem rain_probability (p : ℚ) (h : p = 3/4) :
  1 - (1 - p)^4 = 255/256 := by sorry

end NUMINAMATH_CALUDE_rain_probability_l1169_116934


namespace NUMINAMATH_CALUDE_max_pens_173_l1169_116964

/-- Represents a package of pens with its size and cost -/
structure PenPackage where
  size : Nat
  cost : Nat

/-- Finds the maximum number of pens that can be purchased with a given budget -/
def maxPens (budget : Nat) (packages : List PenPackage) : Nat :=
  sorry

/-- The specific problem setup -/
def problemSetup : List PenPackage := [
  ⟨12, 10⟩,
  ⟨20, 15⟩
]

/-- The theorem stating that the maximum number of pens purchasable with $173 is 224 -/
theorem max_pens_173 : maxPens 173 problemSetup = 224 := by
  sorry

end NUMINAMATH_CALUDE_max_pens_173_l1169_116964


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1169_116972

-- Define the fixed circle F
def F (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the fixed line L
def L (x : ℝ) : Prop := x = 1

-- Define the trajectory of the center M
def trajectory (x y : ℝ) : Prop := y^2 = -8*x

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y : ℝ),
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
      (∃ (x_f y_f : ℝ), F x_f y_f ∧ (x' - x_f)^2 + (y' - y_f)^2 = (r + 1)^2) ∧
      (∃ (x_l : ℝ), L x_l ∧ |x' - x_l| = r))) →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1169_116972


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1169_116917

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x - 3

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1169_116917


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1169_116946

/-- A linear function that does not pass through the third quadrant -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

/-- Predicate to check if a point is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating the range of k for which the linear function does not pass through the third quadrant -/
theorem linear_function_not_in_third_quadrant (k : ℝ) :
  (∀ x, ¬(in_third_quadrant x (linear_function k x))) ↔ (0 ≤ k ∧ k < 2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l1169_116946


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_properties_l1169_116979

theorem smallest_integer_with_divisibility_properties : 
  ∃ (n : ℕ), n > 1 ∧ 
  (∀ (m : ℕ), m > 1 → 
    ((m + 1) % 2 = 0 ∧ 
     (m + 2) % 3 = 0 ∧ 
     (m + 3) % 4 = 0 ∧ 
     (m + 4) % 5 = 0) → m ≥ n) ∧
  (n + 1) % 2 = 0 ∧ 
  (n + 2) % 3 = 0 ∧ 
  (n + 3) % 4 = 0 ∧ 
  (n + 4) % 5 = 0 ∧
  n = 61 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_properties_l1169_116979


namespace NUMINAMATH_CALUDE_largest_inexpressible_number_l1169_116950

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

def has_enough_coins (a b : ℕ) : Prop :=
  a > 10 ∧ b > 10

theorem largest_inexpressible_number :
  (∀ n : ℕ, n > 19 → n ≤ 50 → is_expressible n) ∧
  ¬(is_expressible 19) ∧
  (∀ a b : ℕ, has_enough_coins a b → ∀ n : ℕ, n ≤ 50 → is_expressible n → ∃ c d : ℕ, n = 5 * c + 6 * d ∧ c ≤ a ∧ d ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_largest_inexpressible_number_l1169_116950


namespace NUMINAMATH_CALUDE_monotone_function_a_bound_l1169_116949

/-- Given a function f(x) = x² + a/x that is monotonically increasing on [2, +∞),
    prove that a ≤ 16 -/
theorem monotone_function_a_bound (a : ℝ) :
  (∀ x ≥ 2, Monotone (fun x => x^2 + a/x)) →
  a ≤ 16 := by
  sorry

end NUMINAMATH_CALUDE_monotone_function_a_bound_l1169_116949


namespace NUMINAMATH_CALUDE_distance_equality_l1169_116918

theorem distance_equality : ∃ x : ℝ, |x - (-2)| = |x - 4| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_distance_equality_l1169_116918


namespace NUMINAMATH_CALUDE_square_binomial_constant_l1169_116939

/-- If x^2 + 50x + d is equal to the square of a binomial, then d = 625 -/
theorem square_binomial_constant (d : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 50*x + d = (x + b)^2) → d = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l1169_116939


namespace NUMINAMATH_CALUDE_line_equation_proof_l1169_116959

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (point : Point) (result_line : Line) : 
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = -2 ∧
  point.x = 1 ∧ point.y = 0 ∧
  result_line.a = 1 ∧ result_line.b = -2 ∧ result_line.c = -1 →
  result_line.contains point ∧ result_line.parallel given_line :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1169_116959


namespace NUMINAMATH_CALUDE_distance_to_midpoint_l1169_116930

/-- Right triangle with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Points where circle touches sides
  d : ℝ  -- Distance from B to D on AB
  e : ℝ  -- Distance from B to E on BC
  f : ℝ  -- Distance from C to F on AC
  -- Conditions
  ab_positive : ab > 0
  bc_positive : bc > 0
  d_in_range : 0 < d ∧ d < ab
  e_in_range : 0 < e ∧ e < bc
  f_in_range : 0 < f ∧ f < (ab^2 + bc^2).sqrt
  circle_tangent : d + e + f = (ab^2 + bc^2).sqrt

/-- The main theorem -/
theorem distance_to_midpoint
  (t : RightTriangleWithInscribedCircle)
  (h_ab : t.ab = 6)
  (h_bc : t.bc = 8) :
  t.ab / 2 - t.d = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_l1169_116930


namespace NUMINAMATH_CALUDE_carters_baseball_cards_l1169_116931

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carters_baseball_cards :
  ∀ (marcus_cards carter_cards : ℕ),
    marcus_cards = 210 →
    marcus_cards = carter_cards + 58 →
    carter_cards = 152 :=
by sorry

end NUMINAMATH_CALUDE_carters_baseball_cards_l1169_116931


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1169_116923

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 2 = 0 →
  3 * q^2 - 5 * q - 2 = 0 →
  p ≠ q →
  (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1169_116923


namespace NUMINAMATH_CALUDE_m_range_l1169_116980

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Ici (3/2), f (x/m) - 4*m^2 * f x ≤ f (x-1) + 4 * f m) →
  m ∈ Set.Iic (-Real.sqrt 3 / 2) ∪ Set.Ici (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1169_116980


namespace NUMINAMATH_CALUDE_line_slope_l1169_116910

/-- Given a line with equation y = 2x + 1, its slope is 2. -/
theorem line_slope (x y : ℝ) : y = 2 * x + 1 → (∃ m : ℝ, m = 2 ∧ y = m * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1169_116910


namespace NUMINAMATH_CALUDE_max_value_of_f_l1169_116937

def f (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

theorem max_value_of_f :
  ∃ (max : ℝ), max = 3 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1169_116937


namespace NUMINAMATH_CALUDE_tony_mileage_milestone_l1169_116900

/-- Represents the distances for Tony's errands -/
structure ErrandDistances where
  groceries : ℕ
  haircut : ℕ
  doctor : ℕ

/-- Calculates the point at which Tony has driven exactly 15 miles -/
def mileageMilestone (distances : ErrandDistances) : ℕ :=
  if distances.groceries ≥ 15 then 15
  else distances.groceries + min (15 - distances.groceries) distances.haircut

/-- Theorem stating that Tony will have driven exactly 15 miles after completing
    his grocery trip and driving partially towards his haircut destination -/
theorem tony_mileage_milestone (distances : ErrandDistances)
    (h1 : distances.groceries = 10)
    (h2 : distances.haircut = 15)
    (h3 : distances.doctor = 5) :
    mileageMilestone distances = 15 :=
  sorry

#eval mileageMilestone ⟨10, 15, 5⟩

end NUMINAMATH_CALUDE_tony_mileage_milestone_l1169_116900


namespace NUMINAMATH_CALUDE_ball_probability_l1169_116953

theorem ball_probability (m n : ℕ) : 
  (10 : ℝ) / (m + 10 + n : ℝ) = (m + n : ℝ) / (m + 10 + n : ℝ) → m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1169_116953


namespace NUMINAMATH_CALUDE_cube_surface_area_l1169_116971

theorem cube_surface_area (x d : ℝ) (h_volume : x^3 > 0) (h_diagonal : d > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s^3 = x^3 ∧ d^2 = 3 * s^2 ∧ 6 * s^2 = 2 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1169_116971


namespace NUMINAMATH_CALUDE_boat_rental_cost_sharing_l1169_116933

theorem boat_rental_cost_sharing (total_cost : ℝ) (initial_friends : ℕ) (additional_friends : ℕ) (cost_reduction : ℝ) :
  total_cost = 180 →
  initial_friends = 4 →
  additional_friends = 2 →
  cost_reduction = 15 →
  (total_cost / initial_friends) - cost_reduction = (total_cost / (initial_friends + additional_friends)) →
  total_cost / (initial_friends + additional_friends) = 30 :=
by sorry

end NUMINAMATH_CALUDE_boat_rental_cost_sharing_l1169_116933


namespace NUMINAMATH_CALUDE_even_triple_composition_l1169_116975

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem: if f is even, then f ∘ f ∘ f is even -/
theorem even_triple_composition {f : ℝ → ℝ} (hf : IsEven f) : IsEven (f ∘ f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_even_triple_composition_l1169_116975


namespace NUMINAMATH_CALUDE_friend_bikes_count_l1169_116938

/-- The number of bicycles Ignatius owns -/
def ignatius_bikes : ℕ := 4

/-- The number of tires on a bicycle -/
def tires_per_bike : ℕ := 2

/-- The number of tires on Ignatius's bikes -/
def ignatius_tires : ℕ := ignatius_bikes * tires_per_bike

/-- The total number of tires on the friend's cycles -/
def friend_total_tires : ℕ := 3 * ignatius_tires

/-- The number of tires on a unicycle -/
def unicycle_tires : ℕ := 1

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of tires on the friend's non-bicycle cycles -/
def friend_non_bike_tires : ℕ := unicycle_tires + tricycle_tires

/-- The number of tires on the friend's bicycles -/
def friend_bike_tires : ℕ := friend_total_tires - friend_non_bike_tires

theorem friend_bikes_count : (friend_bike_tires / tires_per_bike) = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_bikes_count_l1169_116938


namespace NUMINAMATH_CALUDE_shortest_side_length_l1169_116968

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Assumption that all lengths are positive -/
  r_pos : r > 0
  a_pos : a > 0
  b_pos : b > 0
  shortest_side_pos : shortest_side > 0

/-- Theorem stating the length of the shortest side in the specific triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
    (h1 : t.r = 5) 
    (h2 : t.a = 9) 
    (h3 : t.b = 15) : 
  t.shortest_side = 17 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l1169_116968


namespace NUMINAMATH_CALUDE_complex_quadrant_l1169_116982

theorem complex_quadrant (z : ℂ) : (z + 2*I) * (3 + I) = 7 - I →
  (z.re > 0 ∧ z.im < 0) :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1169_116982


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1169_116981

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- The property that the range of a function is [0, +∞) -/
def HasNonnegativeRange (f : ℝ → ℝ) : Prop :=
  ∀ y, (∃ x, f x = y) → y ≥ 0

/-- The property that the solution set of f(x) < c is (m, m+8) -/
def HasSolutionSet (f : ℝ → ℝ) (c m : ℝ) : Prop :=
  ∀ x, f x < c ↔ m < x ∧ x < m + 8

theorem quadratic_function_property (a b c m : ℝ) :
  HasNonnegativeRange (QuadraticFunction a b) →
  HasSolutionSet (QuadraticFunction a b) c m →
  c = 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1169_116981


namespace NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l1169_116956

/-- Calculates the cost of paving an L-shaped floor with two types of slabs -/
theorem l_shaped_floor_paving_cost
  (length1 width1 length2 width2 : ℝ)
  (cost_a cost_b : ℝ)
  (percent_a : ℝ)
  (h_length1 : length1 = 5.5)
  (h_width1 : width1 = 3.75)
  (h_length2 : length2 = 4.25)
  (h_width2 : width2 = 2.5)
  (h_cost_a : cost_a = 1000)
  (h_cost_b : cost_b = 1200)
  (h_percent_a : percent_a = 0.6)
  (h_nonneg : length1 ≥ 0 ∧ width1 ≥ 0 ∧ length2 ≥ 0 ∧ width2 ≥ 0 ∧ cost_a ≥ 0 ∧ cost_b ≥ 0 ∧ percent_a ≥ 0 ∧ percent_a ≤ 1) :
  let area1 := length1 * width1
  let area2 := length2 * width2
  let total_area := area1 + area2
  let area_a := total_area * percent_a
  let area_b := total_area * (1 - percent_a)
  let cost := area_a * cost_a + area_b * cost_b
  cost = 33750 :=
by sorry

end NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l1169_116956


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1169_116958

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≤ 2 ∧ y ≤ 3 → x + y ≤ 5) ∧
  ∃ x y : ℝ, x + y ≤ 5 ∧ (x > 2 ∨ y > 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1169_116958


namespace NUMINAMATH_CALUDE_fire_chief_hats_l1169_116902

theorem fire_chief_hats (o_brien_current : ℕ) (h1 : o_brien_current = 34) : ∃ (simpson : ℕ),
  simpson = 15 ∧ o_brien_current + 1 = 2 * simpson + 5 := by
  sorry

end NUMINAMATH_CALUDE_fire_chief_hats_l1169_116902


namespace NUMINAMATH_CALUDE_line_points_theorem_l1169_116926

-- Define the line L with slope 2 passing through (3, 5)
def L (x y : ℝ) : Prop := y - 5 = 2 * (x - 3)

-- Define the points
def P1 : ℝ × ℝ := (3, 5)
def P2 (x2 : ℝ) : ℝ × ℝ := (x2, 7)
def P3 (y3 : ℝ) : ℝ × ℝ := (-1, y3)

theorem line_points_theorem (x2 y3 : ℝ) :
  L P1.1 P1.2 ∧ L (P2 x2).1 (P2 x2).2 ∧ L (P3 y3).1 (P3 y3).2 →
  x2 = 4 ∧ y3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_points_theorem_l1169_116926


namespace NUMINAMATH_CALUDE_ten_liter_barrel_emptying_ways_l1169_116948

def emptyBarrel (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n + 2 => emptyBarrel (n + 1) + emptyBarrel n

theorem ten_liter_barrel_emptying_ways :
  emptyBarrel 10 = 89 := by sorry

end NUMINAMATH_CALUDE_ten_liter_barrel_emptying_ways_l1169_116948


namespace NUMINAMATH_CALUDE_tan_plus_cot_equals_three_l1169_116962

theorem tan_plus_cot_equals_three (α : Real) (h : Real.sin (2 * α) = 2/3) :
  Real.tan α + 1 / Real.tan α = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_cot_equals_three_l1169_116962


namespace NUMINAMATH_CALUDE_max_degree_difference_for_special_graph_l1169_116965

/-- A graph with specific properties -/
structure SpecialGraph where
  vertices : ℕ
  edges : ℕ
  disjoint_pairs : ℕ

/-- The maximal degree difference in a graph -/
def max_degree_difference (G : SpecialGraph) : ℕ :=
  sorry

/-- Theorem stating the maximal degree difference for a specific graph -/
theorem max_degree_difference_for_special_graph :
  ∃ (G : SpecialGraph),
    G.vertices = 30 ∧
    G.edges = 105 ∧
    G.disjoint_pairs = 4822 ∧
    max_degree_difference G = 22 :=
  sorry

end NUMINAMATH_CALUDE_max_degree_difference_for_special_graph_l1169_116965


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1169_116905

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence, if a₂ * a₈ = 16, then a₁ * a₉ = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 2 * a 8 = 16) : a 1 * a 9 = 16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l1169_116905


namespace NUMINAMATH_CALUDE_distance_between_locations_l1169_116999

theorem distance_between_locations (speed_A speed_B : ℝ) (time : ℝ) (remaining_fraction : ℝ) : 
  speed_A = 60 →
  speed_B = 45 →
  time = 2 →
  remaining_fraction = 2 / 5 →
  (speed_A + speed_B) * time / (1 - remaining_fraction) = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_locations_l1169_116999


namespace NUMINAMATH_CALUDE_power_of_power_product_simplification_expression_simplification_division_simplification_l1169_116955

-- Problem 1
theorem power_of_power : (3^3)^2 = 3^6 := by sorry

-- Problem 2
theorem product_simplification (x y : ℝ) : (-4*x*y^3)*(-2*x^2) = 8*x^3*y^3 := by sorry

-- Problem 3
theorem expression_simplification (x y : ℝ) : 2*x*(3*y-x^2)+2*x*x^2 = 6*x*y := by sorry

-- Problem 4
theorem division_simplification (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : 
  (20*x^3*y^5-10*x^4*y^4-20*x^3*y^2) / (-5*x^3*y^2) = -4*y^3 + 2*x*y^2 + 4 := by sorry

end NUMINAMATH_CALUDE_power_of_power_product_simplification_expression_simplification_division_simplification_l1169_116955


namespace NUMINAMATH_CALUDE_find_set_B_l1169_116941

-- Define the universal set U (we'll use ℤ for integers)
def U : Set ℤ := sorry

-- Define set A
def A : Set ℤ := {0, 2, 4}

-- Define the complement of A with respect to U
def C_UA : Set ℤ := {-1, 1}

-- Define the complement of B with respect to U
def C_UB : Set ℤ := {-1, 0, 2}

-- Define set B
def B : Set ℤ := {1, 4}

-- Theorem to prove
theorem find_set_B : B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_find_set_B_l1169_116941


namespace NUMINAMATH_CALUDE_sand_art_project_jason_sand_needed_l1169_116916

/-- The amount of sand needed for Jason's sand art project -/
theorem sand_art_project (rectangular_length : ℕ) (rectangular_width : ℕ) 
  (square_side : ℕ) (sand_per_inch : ℕ) : ℕ :=
  let rectangular_area := rectangular_length * rectangular_width
  let square_area := square_side * square_side
  let total_area := rectangular_area + square_area
  total_area * sand_per_inch

/-- Proof that Jason needs 201 grams of sand -/
theorem jason_sand_needed : sand_art_project 6 7 5 3 = 201 := by
  sorry

end NUMINAMATH_CALUDE_sand_art_project_jason_sand_needed_l1169_116916


namespace NUMINAMATH_CALUDE_new_students_count_l1169_116927

/-- The number of new students who joined Hendrix's class -/
def new_students : ℕ :=
  let initial_students : ℕ := 160
  let final_students : ℕ := 120
  let transfer_ratio : ℚ := 1/3
  let total_after_join : ℕ := final_students * 3 / 2
  total_after_join - initial_students

theorem new_students_count : new_students = 20 := by sorry

end NUMINAMATH_CALUDE_new_students_count_l1169_116927


namespace NUMINAMATH_CALUDE_fifteenth_row_seats_l1169_116945

/-- Represents the number of seats in a row of an auditorium -/
def seats (n : ℕ) : ℕ :=
  5 + 2 * (n - 1)

/-- Theorem: The fifteenth row of the auditorium has 33 seats -/
theorem fifteenth_row_seats : seats 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_row_seats_l1169_116945


namespace NUMINAMATH_CALUDE_min_sum_products_l1169_116994

theorem min_sum_products (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (h : x + y + z = 3 * x * y * z) : 
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 3 * a * b * c → 
    x * y + y * z + x * z ≤ a * b + b * c + a * c :=
by sorry

end NUMINAMATH_CALUDE_min_sum_products_l1169_116994


namespace NUMINAMATH_CALUDE_quadratic_vertex_range_l1169_116942

/-- A quadratic function of the form y = (a-1)x^2 + 3 -/
def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 3

/-- The condition for the quadratic function to open downwards -/
def opens_downwards (a : ℝ) : Prop := a - 1 < 0

theorem quadratic_vertex_range (a : ℝ) :
  (∃ x, ∃ y, quadratic_function a x = y ∧ 
    ∀ z, quadratic_function a z ≤ y) →
  opens_downwards a →
  a < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_range_l1169_116942


namespace NUMINAMATH_CALUDE_circle_condition_l1169_116919

/-- The equation of a potential circle with parameter a -/
def circle_equation (x y a : ℝ) : ℝ := x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1

/-- The set of a values for which the equation represents a circle -/
def circle_parameter_set : Set ℝ := {a | a < 2 ∨ a > 2}

/-- Theorem stating that the equation represents a circle if and only if a is in the specified set -/
theorem circle_condition (a : ℝ) :
  (∃ h k r : ℝ, r > 0 ∧ ∀ x y : ℝ, circle_equation x y a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔
  a ∈ circle_parameter_set :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1169_116919


namespace NUMINAMATH_CALUDE_stating_same_suit_selections_standard_deck_l1169_116944

/-- Represents a standard deck of cards. -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h1 : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards with 4 suits and 13 cards per suit. -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h1 := rfl }

/-- 
The number of ways to select two different cards from the same suit in a standard deck,
where order matters.
-/
def same_suit_selections (d : Deck) : Nat :=
  d.num_suits * (d.cards_per_suit * (d.cards_per_suit - 1))

/-- 
Theorem stating that the number of ways to select two different cards 
from the same suit in a standard deck, where order matters, is 624.
-/
theorem same_suit_selections_standard_deck : 
  same_suit_selections standard_deck = 624 := by
  sorry


end NUMINAMATH_CALUDE_stating_same_suit_selections_standard_deck_l1169_116944


namespace NUMINAMATH_CALUDE_point_c_value_l1169_116924

/-- Represents a point on a number line --/
structure Point where
  value : ℝ

/-- The distance between two points on a number line --/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_c_value (a b c : Point) :
  a.value = -1 →
  distance a b = 11 →
  b.value > a.value →
  distance b c = 5 →
  c.value = 5 ∨ c.value = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_c_value_l1169_116924


namespace NUMINAMATH_CALUDE_june_upload_total_l1169_116988

/-- Represents the upload schedule for a YouTuber in June --/
structure UploadSchedule where
  early_june : Nat  -- videos per day from June 1st to June 15th
  mid_june : Nat    -- videos per day from June 16th to June 23rd
  late_june : Nat   -- videos per day from June 24th to June 30th

/-- Calculates the total number of video hours uploaded in June --/
def total_video_hours (schedule : UploadSchedule) : Nat :=
  schedule.early_june * 15 + schedule.mid_june * 8 + schedule.late_june * 7

/-- Theorem stating that the given upload schedule results in 480 total video hours --/
theorem june_upload_total (schedule : UploadSchedule) 
  (h1 : schedule.early_june = 10)
  (h2 : schedule.mid_june = 15)
  (h3 : schedule.late_june = 30) : 
  total_video_hours schedule = 480 := by
  sorry

#eval total_video_hours { early_june := 10, mid_june := 15, late_june := 30 }

end NUMINAMATH_CALUDE_june_upload_total_l1169_116988


namespace NUMINAMATH_CALUDE_fourth_person_height_l1169_116992

theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ - h₁ = 2 →                 -- difference between 1st and 2nd
  h₃ - h₂ = 2 →                 -- difference between 2nd and 3rd
  h₄ - h₃ = 6 →                 -- difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76  -- average height
  → h₄ = 82 :=                  -- height of 4th person
by sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1169_116992


namespace NUMINAMATH_CALUDE_wire_cutting_l1169_116997

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 14 →
  ratio = 2 / 5 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 4 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1169_116997


namespace NUMINAMATH_CALUDE_power_of_two_start_with_any_digits_l1169_116969

theorem power_of_two_start_with_any_digits :
  ∀ A : ℕ, ∃ n m : ℕ+, (10 ^ m.val : ℝ) * A < (2 ^ n.val : ℝ) ∧ (2 ^ n.val : ℝ) < (10 ^ m.val : ℝ) * (A + 1) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_start_with_any_digits_l1169_116969


namespace NUMINAMATH_CALUDE_G_equals_3F_l1169_116977

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((3 * x + x^3) / (1 + 3 * x^2))

theorem G_equals_3F (x : ℝ) : G x = 3 * F x :=
  sorry

end NUMINAMATH_CALUDE_G_equals_3F_l1169_116977


namespace NUMINAMATH_CALUDE_smallest_y_in_arithmetic_sequence_l1169_116954

theorem smallest_y_in_arithmetic_sequence (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- x, y, z are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- x, y, z form an arithmetic sequence
  x * y * z = 216 →  -- product condition
  y ≥ 6 ∧ (∀ w : ℝ, w > 0 ∧ (∃ d' : ℝ, (w - d') * w * (w + d') = 216) → w ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_in_arithmetic_sequence_l1169_116954


namespace NUMINAMATH_CALUDE_product_of_complex_polars_l1169_116963

/-- Represents a complex number in polar form -/
structure ComplexPolar where
  magnitude : ℝ
  angle : ℝ

/-- Multiplication of complex numbers in polar form -/
def mul_complex_polar (z₁ z₂ : ComplexPolar) : ComplexPolar :=
  { magnitude := z₁.magnitude * z₂.magnitude,
    angle := z₁.angle + z₂.angle }

theorem product_of_complex_polars :
  let z₁ : ComplexPolar := { magnitude := 5, angle := 30 }
  let z₂ : ComplexPolar := { magnitude := 4, angle := 45 }
  let product := mul_complex_polar z₁ z₂
  product.magnitude = 20 ∧ product.angle = 75 := by sorry

end NUMINAMATH_CALUDE_product_of_complex_polars_l1169_116963


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l1169_116912

def bacteria_growth (initial_count : ℕ) (final_count : ℕ) (tripling_time : ℕ) : ℕ → Prop :=
  fun hours => initial_count * (3 ^ (hours / tripling_time)) = final_count

theorem bacteria_growth_time : 
  bacteria_growth 200 16200 6 24 := by sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l1169_116912


namespace NUMINAMATH_CALUDE_rectangle_equation_l1169_116978

/-- Given a rectangle with area 864 square steps and perimeter 120 steps,
    prove that the equation relating its length x to its area is x(60 - x) = 864 -/
theorem rectangle_equation (x : ℝ) 
  (area : ℝ) (perimeter : ℝ)
  (h_area : area = 864)
  (h_perimeter : perimeter = 120)
  (h_x : x > 0 ∧ x < 60) :
  x * (60 - x) = 864 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_equation_l1169_116978


namespace NUMINAMATH_CALUDE_visual_illusion_occurs_l1169_116996

/-- Represents the structure of the cardboard disc -/
structure Disc where
  inner_sectors : Nat
  outer_sectors : Nat
  inner_white : Nat
  outer_white : Nat

/-- Represents the properties of electric lighting -/
structure Lighting where
  flicker_frequency : Real
  flicker_interval : Real

/-- Defines the rotation speeds that create the visual illusion -/
def illusion_speeds (d : Disc) (l : Lighting) : Prop :=
  let inner_speed := 25
  let outer_speed := 20
  inner_speed * l.flicker_interval = 0.25 ∧
  outer_speed * l.flicker_interval = 0.2

theorem visual_illusion_occurs (d : Disc) (l : Lighting) :
  d.inner_sectors = 8 ∧
  d.outer_sectors = 10 ∧
  d.inner_white = 4 ∧
  d.outer_white = 5 ∧
  l.flicker_frequency = 100 ∧
  l.flicker_interval = 0.01 →
  illusion_speeds d l :=
by sorry


end NUMINAMATH_CALUDE_visual_illusion_occurs_l1169_116996


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_l1169_116922

-- Define the pyramid structure
structure RightPyramid :=
  (base_side : ℝ)
  (slant_height : ℝ)
  (lateral_face_area : ℝ)

-- Theorem statement
theorem right_pyramid_base_side 
  (p : RightPyramid) 
  (h1 : p.lateral_face_area = 120) 
  (h2 : p.slant_height = 40) : 
  p.base_side = 6 := by
  sorry


end NUMINAMATH_CALUDE_right_pyramid_base_side_l1169_116922


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l1169_116932

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem parallel_vectors_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + 2 • b = 0 → ∃ k : ℝ, a = k • b) ∧
  ¬(∃ k : ℝ, a = k • b → a + 2 • b = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l1169_116932


namespace NUMINAMATH_CALUDE_median_of_temperatures_l1169_116951

def temperatures : List ℝ := [19, 21, 25, 22, 19, 22, 21]

def median (l : List ℝ) : ℝ := sorry

theorem median_of_temperatures : median temperatures = 21 := by sorry

end NUMINAMATH_CALUDE_median_of_temperatures_l1169_116951


namespace NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l1169_116987

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop := ρ * (Real.cos θ)^2 = 4 * Real.sin θ

/-- The Cartesian equation of the curve -/
def cartesian_equation (x y : ℝ) : Prop := x^2 = 4 * y

/-- Theorem stating that the polar equation represents a parabola -/
theorem polar_to_cartesian_parabola :
  ∀ (x y ρ θ : ℝ), 
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  polar_equation ρ θ →
  cartesian_equation x y :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l1169_116987


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l1169_116957

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ 3^x < x^3) ↔ (∀ x : ℝ, x > 0 → 3^x ≥ x^3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l1169_116957


namespace NUMINAMATH_CALUDE_range_of_a_l1169_116943

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*|x - a| ≥ a^2) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1169_116943


namespace NUMINAMATH_CALUDE_samantha_born_1986_l1169_116993

/-- The year of the first Math Kangaroo contest -/
def first_math_kangaroo_year : ℕ := 1991

/-- The age of Samantha when she took the tenth Math Kangaroo -/
def samantha_age_tenth_kangaroo : ℕ := 14

/-- Function to calculate the year of the nth Math Kangaroo contest -/
def math_kangaroo_year (n : ℕ) : ℕ := first_math_kangaroo_year + n - 1

/-- Samantha's birth year -/
def samantha_birth_year : ℕ := math_kangaroo_year 10 - samantha_age_tenth_kangaroo

theorem samantha_born_1986 : samantha_birth_year = 1986 := by
  sorry

end NUMINAMATH_CALUDE_samantha_born_1986_l1169_116993


namespace NUMINAMATH_CALUDE_watch_gain_percentage_l1169_116991

/-- Calculates the gain percentage when a watch is sold at a higher price -/
theorem watch_gain_percentage (cost_price : ℝ) (loss_percentage : ℝ) (price_increase : ℝ) : 
  cost_price = 1400 →
  loss_percentage = 10 →
  price_increase = 196 →
  let initial_selling_price := cost_price * (1 - loss_percentage / 100)
  let new_selling_price := initial_selling_price + price_increase
  let gain_amount := new_selling_price - cost_price
  let gain_percentage := (gain_amount / cost_price) * 100
  gain_percentage = 4 := by
  sorry

end NUMINAMATH_CALUDE_watch_gain_percentage_l1169_116991


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l1169_116960

/-- An isosceles triangle with two angles of 60° and x° -/
structure IsoscelesTriangle60X where
  /-- The measure of angle x in degrees -/
  x : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : True
  /-- One angle of the triangle is 60° -/
  has60Angle : True
  /-- Another angle of the triangle is x° -/
  hasXAngle : True
  /-- The sum of angles in a triangle is 180° -/
  angleSum : True

/-- The sum of all possible values of x in an isosceles triangle with angles 60° and x° is 180° -/
theorem sum_of_possible_x_values (t : IsoscelesTriangle60X) : 
  ∃ (x₁ x₂ x₃ : ℝ), (x₁ + x₂ + x₃ = 180 ∧ 
    (t.x = x₁ ∨ t.x = x₂ ∨ t.x = x₃)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l1169_116960


namespace NUMINAMATH_CALUDE_exists_product_of_smallest_primes_l1169_116904

/-- The radical of a positive integer n is the product of its distinct prime factors -/
def rad (n : ℕ+) : ℕ+ :=
  sorry

/-- The sequence a_n defined by the recurrence relation a_{n+1} = a_n + rad(a_n) -/
def a : ℕ → ℕ+
  | 0 => sorry
  | n + 1 => a n + rad (a n)

/-- The s-th smallest prime number -/
def nthSmallestPrime (s : ℕ+) : ℕ+ :=
  sorry

/-- The product of the s smallest primes -/
def productOfSmallestPrimes (s : ℕ+) : ℕ+ :=
  sorry

theorem exists_product_of_smallest_primes :
  ∃ (t s : ℕ+), a t = productOfSmallestPrimes s := by
  sorry

end NUMINAMATH_CALUDE_exists_product_of_smallest_primes_l1169_116904


namespace NUMINAMATH_CALUDE_tree_planting_cost_l1169_116928

/-- The cost of planting trees around a circular park -/
theorem tree_planting_cost
  (park_circumference : ℕ) -- Park circumference in meters
  (planting_interval : ℕ) -- Interval between trees in meters
  (tree_cost : ℕ) -- Cost per tree in mill
  (h1 : park_circumference = 1500)
  (h2 : planting_interval = 30)
  (h3 : tree_cost = 5000) :
  (park_circumference / planting_interval) * tree_cost = 250000 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_cost_l1169_116928


namespace NUMINAMATH_CALUDE_exam_questions_l1169_116961

theorem exam_questions (correct_score : ℕ) (wrong_penalty : ℕ) (total_score : ℕ) (correct_answers : ℕ) : ℕ :=
  let total_questions := correct_answers + (correct_score * correct_answers - total_score)
  50

#check exam_questions 4 1 130 36

end NUMINAMATH_CALUDE_exam_questions_l1169_116961


namespace NUMINAMATH_CALUDE_shortest_distance_exp_to_line_l1169_116986

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the line g(x) = x
def g (x : ℝ) : ℝ := x

-- Statement: The shortest distance from any point on f to g is √2/2
theorem shortest_distance_exp_to_line :
  ∃ d : ℝ, d = Real.sqrt 2 / 2 ∧
  ∀ x y : ℝ, f x = y → 
  ∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
  d ≤ Real.sqrt ((p.1 - p.2)^2 + 1) / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_exp_to_line_l1169_116986


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1169_116947

theorem cube_root_equation_solution (x : ℝ) (h : (3 - 1 / x^2)^(1/3) = -4) : 
  x = 1 / Real.sqrt 67 ∨ x = -1 / Real.sqrt 67 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1169_116947


namespace NUMINAMATH_CALUDE_impossibleCubeLabeling_l1169_116901

-- Define a cube type
structure Cube where
  vertices : Fin 8 → ℕ

-- Define the property of being an odd number between 1 and 600
def isValidNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 600

-- Define adjacency in a cube
def isAdjacent (i j : Fin 8) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

-- Define the property of having a common divisor greater than 1
def hasCommonDivisor (a b : ℕ) : Prop :=
  ∃ (d : ℕ), d > 1 ∧ a % d = 0 ∧ b % d = 0

-- Main theorem
theorem impossibleCubeLabeling :
  ¬∃ (c : Cube),
    (∀ i : Fin 8, isValidNumber (c.vertices i)) ∧
    (∀ i j : Fin 8, i ≠ j → c.vertices i ≠ c.vertices j) ∧
    (∀ i j : Fin 8, isAdjacent i j → hasCommonDivisor (c.vertices i) (c.vertices j)) ∧
    (∀ i j : Fin 8, ¬isAdjacent i j → ¬hasCommonDivisor (c.vertices i) (c.vertices j)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleCubeLabeling_l1169_116901


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l1169_116970

def is_largest_prime_with_2005_digits (p : ℕ) : Prop :=
  Nat.Prime p ∧ 
  (10^2004 ≤ p) ∧ 
  (p < 10^2005) ∧ 
  ∀ q, Nat.Prime q → (10^2004 ≤ q) → (q < 10^2005) → q ≤ p

theorem smallest_k_for_divisibility_by_10 (p : ℕ) 
  (h : is_largest_prime_with_2005_digits p) : 
  (∃ k : ℕ, k > 0 ∧ (10 ∣ (p^2 - k))) ∧
  (∀ k : ℕ, k > 0 → (10 ∣ (p^2 - k)) → k ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l1169_116970


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1169_116952

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation
  (principal amount : ℚ)
  (time : ℕ)
  (h_principal : principal = 2500)
  (h_amount : amount = 3875)
  (h_time : time = 12)
  (h_positive : principal > 0 ∧ amount > principal ∧ time > 0) :
  (amount - principal) * 100 / (principal * time) = 55 / 12 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1169_116952


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1169_116990

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), ∀ x, (x + 3) * (x - 3) = 2 * x → a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -2 ∧ c = -9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1169_116990


namespace NUMINAMATH_CALUDE_sine_monotonicity_l1169_116935

open Real

theorem sine_monotonicity (k : ℤ) :
  let f : ℝ → ℝ := λ x => sin (2 * x + (5 * π) / 6)
  let interval := Set.Icc (k * π + π / 3) (k * π + 5 * π / 6)
  (∀ x, f x ≥ f (π / 3)) →
  StrictMono (interval.restrict f) :=
by sorry

end NUMINAMATH_CALUDE_sine_monotonicity_l1169_116935


namespace NUMINAMATH_CALUDE_solution_set_for_half_range_of_m_l1169_116906

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| - |2*x - 2*m|

-- Part 1
theorem solution_set_for_half (x : ℝ) :
  (f x (1/2) ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ m < 7/2) ↔
    (∀ x : ℝ, ∃ t : ℝ, f x m + |t - 3| < |t + 4|) :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_half_range_of_m_l1169_116906


namespace NUMINAMATH_CALUDE_animals_per_aquarium_l1169_116909

theorem animals_per_aquarium 
  (total_animals : ℕ) 
  (num_aquariums : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : num_aquariums = 20) 
  (h3 : total_animals % num_aquariums = 0) : 
  total_animals / num_aquariums = 2 := by
sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_l1169_116909


namespace NUMINAMATH_CALUDE_derived_figure_total_length_l1169_116966

/-- Represents a shape with perpendicular adjacent sides -/
structure NewShape where
  sides : ℕ

/-- Represents the derived figure created from the new shape -/
structure DerivedFigure where
  left_vertical : ℕ
  right_vertical : ℕ
  lower_horizontal : ℕ
  extra_top : ℕ

/-- Creates a derived figure from a new shape -/
def create_derived_figure (s : NewShape) : DerivedFigure :=
  { left_vertical := 12
  , right_vertical := 9
  , lower_horizontal := 7
  , extra_top := 2 }

/-- Calculates the total length of segments in the derived figure -/
def total_length (d : DerivedFigure) : ℕ :=
  d.left_vertical + d.right_vertical + d.lower_horizontal + d.extra_top

/-- Theorem stating that the total length of segments in the derived figure is 30 units -/
theorem derived_figure_total_length (s : NewShape) :
  total_length (create_derived_figure s) = 30 := by
  sorry

end NUMINAMATH_CALUDE_derived_figure_total_length_l1169_116966


namespace NUMINAMATH_CALUDE_monic_quartic_value_l1169_116995

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value (f : ℝ → ℝ) :
  is_monic_quartic f →
  f (-2) = -4 →
  f 1 = -1 →
  f (-3) = -9 →
  f 5 = -25 →
  f 2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_value_l1169_116995


namespace NUMINAMATH_CALUDE_right_triangle_third_side_length_l1169_116940

theorem right_triangle_third_side_length 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 13) 
  (hc : c * c = a * a + b * b) 
  (hright : a < b ∧ b > c) : c = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_length_l1169_116940


namespace NUMINAMATH_CALUDE_race_distance_l1169_116908

theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (distance : ℝ) : 
  time_A = 36 →
  time_B = 45 →
  lead = 26 →
  (distance / time_B) * time_A = distance - lead →
  distance = 130 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l1169_116908


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_ii_l1169_116915

/-- A linear function with slope k and y-intercept b -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- Quadrant II is the region where x < 0 and y > 0 -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem linear_function_not_in_quadrant_ii :
  ∀ x : ℝ, ¬InQuadrantII x (LinearFunction 3 (-2) x) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_ii_l1169_116915


namespace NUMINAMATH_CALUDE_log_inequality_l1169_116903

def number_of_distinct_prime_divisors (n : ℕ) : ℕ := sorry

theorem log_inequality (n : ℕ) (k : ℕ) (h : k = number_of_distinct_prime_divisors n) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1169_116903


namespace NUMINAMATH_CALUDE_group_size_proof_l1169_116921

/-- The number of people in a group where:
    1) Replacing a 60 kg person with a 110 kg person increases the total weight by 50 kg.
    2) The average weight increase is 5 kg.
-/
def group_size : ℕ :=
  10

theorem group_size_proof :
  (group_size : ℝ) * 5 = 110 - 60 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l1169_116921


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l1169_116914

/-- Two cyclists on a circular track problem -/
theorem cyclists_meeting_time
  (circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : circumference = 180)
  (h2 : speed1 = 7)
  (h3 : speed2 = 8) :
  circumference / (speed1 + speed2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l1169_116914


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l1169_116984

/-- Proves that the area of a rectangular garden with length three times its width and width of 12 meters is 432 square meters. -/
theorem rectangular_garden_area :
  ∀ (length width area : ℝ),
    width = 12 →
    length = 3 * width →
    area = length * width →
    area = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l1169_116984


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l1169_116925

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 1) * N * (N + 1)) / Nat.factorial (N + 2) = 1 / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l1169_116925


namespace NUMINAMATH_CALUDE_intersection_of_quadratic_equations_l1169_116929

theorem intersection_of_quadratic_equations (p q : ℝ) : 
  (∃ M N : Set ℝ, 
    (∀ x, x ∈ M ↔ x^2 - p*x + 8 = 0) ∧ 
    (∀ x, x ∈ N ↔ x^2 - q*x + p = 0) ∧ 
    (M ∩ N = {1})) → 
  p + q = 19 := by
sorry

end NUMINAMATH_CALUDE_intersection_of_quadratic_equations_l1169_116929


namespace NUMINAMATH_CALUDE_fermat_prime_condition_l1169_116985

theorem fermat_prime_condition (a n : ℕ) (ha : a > 1) (hn : n > 1) :
  Nat.Prime (a^n + 1) → (Even a ∧ ∃ k : ℕ, n = 2^k) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_condition_l1169_116985


namespace NUMINAMATH_CALUDE_parabola_vertex_l1169_116936

/-- The equation of a parabola in the form y^2 + 4y + 3x + 1 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4*y + 3*x + 1 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → y' ≥ y

/-- Theorem: The vertex of the parabola y^2 + 4y + 3x + 1 = 0 is (1, -2) -/
theorem parabola_vertex :
  is_vertex 1 (-2) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1169_116936


namespace NUMINAMATH_CALUDE_even_function_implies_b_zero_l1169_116998

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x(x+b) -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ x * (x + b)

/-- If f(x) = x(x+b) is an even function, then b = 0 -/
theorem even_function_implies_b_zero :
  ∀ b : ℝ, IsEven (f b) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_b_zero_l1169_116998


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1169_116983

/-- Represents the sampling methods --/
inductive SamplingMethod
  | StratifiedSampling
  | LotteryMethod
  | SystematicSampling
  | RandomNumberTableMethod

/-- Represents a school structure --/
structure School where
  num_classes : Nat
  students_per_class : Nat
  student_numbering : Nat → Nat → Nat  -- Class number → Student number → Assigned number

/-- Represents a selection method --/
structure SelectionMethod where
  selected_number : Nat

/-- Determines the sampling method based on school structure and selection method --/
def determineSamplingMethod (school : School) (selection : SelectionMethod) : SamplingMethod :=
  sorry

/-- Theorem stating that the given conditions result in Systematic Sampling --/
theorem systematic_sampling_proof (school : School) (selection : SelectionMethod) :
  school.num_classes = 18 ∧
  school.students_per_class = 56 ∧
  (∀ c s, school.student_numbering c s = s) ∧
  selection.selected_number = 14 →
  determineSamplingMethod school selection = SamplingMethod.SystematicSampling :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1169_116983


namespace NUMINAMATH_CALUDE_find_other_number_l1169_116976

theorem find_other_number (x y : ℤ) : 
  (3 * x + 2 * y = 130) → 
  ((x = 35 ∨ y = 35) → 
  ((x ≠ 35 → y = 35 ∧ x = 20) ∧ 
   (y ≠ 35 → x = 35 ∧ y = 20))) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l1169_116976


namespace NUMINAMATH_CALUDE_largest_solution_proof_l1169_116967

/-- The equation from the problem -/
def equation (x : ℝ) : Prop :=
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12*x - 5

/-- The largest real solution to the equation -/
def largest_solution : ℝ := 20

/-- The representation of the solution in the form d + √(e + √f) -/
def solution_form (d e f : ℕ) (x : ℝ) : Prop :=
  x = d + Real.sqrt (e + Real.sqrt f)

theorem largest_solution_proof :
  equation largest_solution ∧
  ∃ (d e f : ℕ), solution_form d e f largest_solution ∧
  ∀ (x : ℝ), equation x → x ≤ largest_solution :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_proof_l1169_116967


namespace NUMINAMATH_CALUDE_chemistry_books_count_l1169_116911

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chemistry_books_count :
  ∃ (c : ℕ),
    c > 0 ∧
    (choose2 10) * (choose2 c) = 1260 ∧
    ∀ (x : ℕ), x > 0 → (choose2 10) * (choose2 x) = 1260 → x = c :=
by sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l1169_116911


namespace NUMINAMATH_CALUDE_percentage_difference_l1169_116920

theorem percentage_difference (x : ℝ) : x = 30 → 0.9 * 40 = 0.8 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1169_116920


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l1169_116913

theorem sequence_sum_problem (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 8)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 81) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 425 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l1169_116913


namespace NUMINAMATH_CALUDE_school_students_count_l1169_116907

theorem school_students_count (total : ℕ) (difference : ℕ) (boys : ℕ) : 
  total = 650 →
  difference = 106 →
  boys + (boys + difference) = total →
  boys = 272 := by
sorry

end NUMINAMATH_CALUDE_school_students_count_l1169_116907


namespace NUMINAMATH_CALUDE_extremum_condition_l1169_116974

/-- A function f: ℝ → ℝ has an extremum at point a -/
def HasExtremumAt (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f x ≤ f a) ∨ (∀ x, f x ≥ f a)

theorem extremum_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ a : ℝ, HasExtremumAt f a → (deriv f) a = 0) ∧
  (∃ g : ℝ → ℝ, Differentiable ℝ g ∧ ∃ b : ℝ, (deriv g) b = 0 ∧ ¬HasExtremumAt g b) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l1169_116974


namespace NUMINAMATH_CALUDE_find_x_value_l1169_116973

theorem find_x_value (numbers : List ℕ) (x : ℕ) : 
  numbers = [54, 55, 57, 58, 59, 62, 62, 63, 65] →
  numbers.length = 9 →
  (numbers.sum + x) / 10 = 60 →
  x = 65 := by
sorry

end NUMINAMATH_CALUDE_find_x_value_l1169_116973


namespace NUMINAMATH_CALUDE_inverse_proportion_x_relationship_l1169_116989

/-- 
Given three points A(x₁, -2), B(x₂, 1), and C(x₃, 2) on the graph of the inverse proportion function y = -2/x,
prove that x₂ < x₃ < x₁.
-/
theorem inverse_proportion_x_relationship (x₁ x₂ x₃ : ℝ) : 
  (-2 = -2 / x₁) → (1 = -2 / x₂) → (2 = -2 / x₃) → x₂ < x₃ ∧ x₃ < x₁ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_x_relationship_l1169_116989
