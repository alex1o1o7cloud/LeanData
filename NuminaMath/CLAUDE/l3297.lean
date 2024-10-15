import Mathlib

namespace NUMINAMATH_CALUDE_square_plus_linear_plus_one_eq_square_l3297_329739

theorem square_plus_linear_plus_one_eq_square (x y : ℕ) :
  y^2 + y + 1 = x^2 ↔ x = 1 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_linear_plus_one_eq_square_l3297_329739


namespace NUMINAMATH_CALUDE_ratio_equivalence_l3297_329799

theorem ratio_equivalence (x : ℝ) : 
  (20 / 10 = 25 / x) → x = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l3297_329799


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_value_l3297_329751

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^3 - a

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * a * x^2

theorem tangent_parallel_implies_a_value (a : ℝ) :
  (f a 1 = a) →                           -- The point (1, a) is on the curve
  (f_derivative a 1 = 2) →                -- The slope of the tangent at (1, a) equals the slope of 2x - y + 1 = 0
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_value_l3297_329751


namespace NUMINAMATH_CALUDE_expression_evaluations_l3297_329713

theorem expression_evaluations :
  (3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3) ^ 0 + |Real.sqrt 3 - 2| = 3) ∧
  ((3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / Real.sqrt 3 = 28/3) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluations_l3297_329713


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3297_329765

theorem polynomial_division_theorem (x : ℝ) :
  x^6 + 5*x^4 + 3 = (x - 2) * (x^5 + 2*x^4 + 9*x^3 + 18*x^2 + 36*x + 72) + 147 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3297_329765


namespace NUMINAMATH_CALUDE_reflection_about_x_axis_l3297_329712

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  f : ℝ → ℝ

/-- Reflects a parabola about the x-axis -/
def reflect_x (p : Parabola) : Parabola :=
  { f := λ x => -(p.f x) }

/-- The original parabola y = x^2 + x - 2 -/
def original_parabola : Parabola :=
  { f := λ x => x^2 + x - 2 }

/-- The expected reflected parabola y = -x^2 - x + 2 -/
def expected_reflected_parabola : Parabola :=
  { f := λ x => -x^2 - x + 2 }

theorem reflection_about_x_axis :
  reflect_x original_parabola = expected_reflected_parabola :=
by sorry

end NUMINAMATH_CALUDE_reflection_about_x_axis_l3297_329712


namespace NUMINAMATH_CALUDE_smallest_integer_value_is_two_l3297_329719

/-- Represents a digit assignment for the kangaroo/game expression -/
structure DigitAssignment where
  k : Nat
  a : Nat
  n : Nat
  g : Nat
  r : Nat
  o : Nat
  m : Nat
  e : Nat
  k_nonzero : k ≠ 0
  a_nonzero : a ≠ 0
  n_nonzero : n ≠ 0
  g_nonzero : g ≠ 0
  r_nonzero : r ≠ 0
  o_nonzero : o ≠ 0
  m_nonzero : m ≠ 0
  e_nonzero : e ≠ 0
  all_different : k ≠ a ∧ k ≠ n ∧ k ≠ g ∧ k ≠ r ∧ k ≠ o ∧ k ≠ m ∧ k ≠ e ∧
                  a ≠ n ∧ a ≠ g ∧ a ≠ r ∧ a ≠ o ∧ a ≠ m ∧ a ≠ e ∧
                  n ≠ g ∧ n ≠ r ∧ n ≠ o ∧ n ≠ m ∧ n ≠ e ∧
                  g ≠ r ∧ g ≠ o ∧ g ≠ m ∧ g ≠ e ∧
                  r ≠ o ∧ r ≠ m ∧ r ≠ e ∧
                  o ≠ m ∧ o ≠ e ∧
                  m ≠ e
  all_digits : k < 10 ∧ a < 10 ∧ n < 10 ∧ g < 10 ∧ r < 10 ∧ o < 10 ∧ m < 10 ∧ e < 10

/-- Calculates the value of the kangaroo/game expression for a given digit assignment -/
def expressionValue (d : DigitAssignment) : Rat :=
  (d.k * d.a * d.n * d.g * d.a * d.r * d.o * d.o) / (d.g * d.a * d.m * d.e)

/-- States that the smallest integer value of the kangaroo/game expression is 2 -/
theorem smallest_integer_value_is_two :
  ∃ (d : DigitAssignment), expressionValue d = 2 ∧
  ∀ (d' : DigitAssignment), (expressionValue d').isInt → expressionValue d' ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_value_is_two_l3297_329719


namespace NUMINAMATH_CALUDE_rent_increase_problem_l3297_329766

theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800) (h2 : new_average = 850) 
  (h3 : num_friends = 4) (h4 : increase_percentage = 0.16) : 
  ∃ (original_rent : ℝ), 
    (num_friends * new_average - num_friends * initial_average) / increase_percentage = original_rent ∧ 
    original_rent = 1250 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l3297_329766


namespace NUMINAMATH_CALUDE_four_spheres_cover_point_source_l3297_329727

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray emanating from a point
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop := sorry

-- Theorem statement
theorem four_spheres_cover_point_source (source : Point3D) :
  ∃ (s1 s2 s3 s4 : Sphere),
    ∀ (r : Ray),
      r.origin = source →
      rayIntersectsSphere r s1 ∨
      rayIntersectsSphere r s2 ∨
      rayIntersectsSphere r s3 ∨
      rayIntersectsSphere r s4 := by
  sorry

end NUMINAMATH_CALUDE_four_spheres_cover_point_source_l3297_329727


namespace NUMINAMATH_CALUDE_f_plus_two_is_odd_l3297_329730

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- State the theorem
theorem f_plus_two_is_odd (h : satisfies_property f) : is_odd (λ x => f x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_plus_two_is_odd_l3297_329730


namespace NUMINAMATH_CALUDE_line_in_plane_equivalence_l3297_329754

-- Define a type for geometric objects
inductive GeometricObject
| Line : GeometricObject
| Plane : GeometricObject

-- Define a predicate for "is in"
def isIn (a b : GeometricObject) : Prop := sorry

-- Define the subset relation
def subset (a b : GeometricObject) : Prop := sorry

-- Theorem statement
theorem line_in_plane_equivalence (l : GeometricObject) (α : GeometricObject) :
  (l = GeometricObject.Line ∧ α = GeometricObject.Plane ∧ isIn l α) ↔ subset l α :=
sorry

end NUMINAMATH_CALUDE_line_in_plane_equivalence_l3297_329754


namespace NUMINAMATH_CALUDE_factory_production_quota_l3297_329705

theorem factory_production_quota (x : ℕ) : 
  ((x - 3) * 31 + 60 = (x + 3) * 25 - 60) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_quota_l3297_329705


namespace NUMINAMATH_CALUDE_james_passenger_count_l3297_329723

/-- Calculates the total number of passengers James has seen given the vehicle counts and passenger capacities. -/
def total_passengers (total_vehicles : ℕ) (trucks : ℕ) (buses : ℕ) (cars : ℕ) 
  (truck_capacity : ℕ) (bus_capacity : ℕ) (taxi_capacity : ℕ) (motorbike_capacity : ℕ) (car_capacity : ℕ) : ℕ :=
  let taxis := 2 * buses
  let motorbikes := total_vehicles - trucks - buses - taxis - cars
  trucks * truck_capacity + buses * bus_capacity + taxis * taxi_capacity + motorbikes * motorbike_capacity + cars * car_capacity

theorem james_passenger_count :
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end NUMINAMATH_CALUDE_james_passenger_count_l3297_329723


namespace NUMINAMATH_CALUDE_vegetable_sale_ratio_l3297_329724

theorem vegetable_sale_ratio : 
  let carrots : ℝ := 15
  let zucchini : ℝ := 13
  let broccoli : ℝ := 8
  let total_installed : ℝ := carrots + zucchini + broccoli
  let sold : ℝ := 18
  sold / total_installed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_sale_ratio_l3297_329724


namespace NUMINAMATH_CALUDE_sum_of_x_satisfying_condition_l3297_329771

def X : Finset ℕ := {0, 1, 2}

def g : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 1
| _ => 0

def f : ℕ → ℕ
| 0 => 2
| 1 => 1
| 2 => 0
| _ => 0

theorem sum_of_x_satisfying_condition : 
  (X.filter (fun x => f (g x) > g (f x))).sum id = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_satisfying_condition_l3297_329771


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l3297_329707

/-- Set M in the complex plane --/
def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = t / (1 + t) + Complex.I * (1 + t) / t}

/-- Set N in the complex plane --/
def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = Real.sqrt 2 * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

/-- The intersection of sets M and N is empty --/
theorem M_intersect_N_empty : M ∩ N = ∅ := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l3297_329707


namespace NUMINAMATH_CALUDE_sin_cos_equivalence_l3297_329734

/-- The function f(x) = sin(2x) + √3 * cos(2x) is equivalent to 2 * sin(2(x + π/6)) for all real x -/
theorem sin_cos_equivalence (x : ℝ) : 
  Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equivalence_l3297_329734


namespace NUMINAMATH_CALUDE_elevator_capacity_l3297_329732

theorem elevator_capacity (adult_avg_weight child_avg_weight next_person_max_weight : ℝ)
  (num_adults num_children : ℕ) :
  adult_avg_weight = 140 →
  child_avg_weight = 64 →
  next_person_max_weight = 52 →
  num_adults = 3 →
  num_children = 2 →
  (num_adults : ℝ) * adult_avg_weight + (num_children : ℝ) * child_avg_weight + next_person_max_weight = 600 :=
by sorry

end NUMINAMATH_CALUDE_elevator_capacity_l3297_329732


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3297_329776

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Line l with given properties -/
structure Line where
  m : ℝ

/-- Theorem stating the properties of the ellipse and line -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (e : ℝ)
  (min_dist : ℝ)
  (l : Line)
  (AB : ℝ)
  (h1 : e = Real.sqrt 3 / 3)
  (h2 : min_dist = Real.sqrt 3 - 1)
  (h3 : AB = 8 * Real.sqrt 3 / 5) :
  (∃ x y, x^2 / 3 + y^2 / 2 = 1) ∧
  (l.m = 1 ∨ l.m = -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3297_329776


namespace NUMINAMATH_CALUDE_quadratic_roots_l3297_329756

theorem quadratic_roots (x y : ℝ) : 
  x + y = 8 → 
  |x - y| = 10 → 
  x^2 - 8*x - 9 = 0 ∧ y^2 - 8*y - 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3297_329756


namespace NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l3297_329744

theorem min_value_xy_plus_two_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 1 → x * y + 2 / (x * y) ≤ z * w + 2 / (z * w)) ∧
  x * y + 2 / (x * y) = 33 / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l3297_329744


namespace NUMINAMATH_CALUDE_factor_expression_l3297_329795

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3297_329795


namespace NUMINAMATH_CALUDE_store_distribution_problem_l3297_329762

/-- Represents the number of ways to distribute stores among cities -/
def distributionCount (totalStores : ℕ) (totalCities : ℕ) (maxStoresPerCity : ℕ) : ℕ :=
  sorry

/-- The specific problem conditions -/
theorem store_distribution_problem :
  distributionCount 4 5 2 = 45 := by sorry

end NUMINAMATH_CALUDE_store_distribution_problem_l3297_329762


namespace NUMINAMATH_CALUDE_trig_identity_l3297_329767

theorem trig_identity (α φ : ℝ) : 
  4 * Real.cos α * Real.cos φ * Real.cos (α - φ) - 
  2 * (Real.cos (α - φ))^2 - Real.cos (2 * φ) = 
  Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3297_329767


namespace NUMINAMATH_CALUDE_race_distance_difference_l3297_329791

theorem race_distance_difference (race_distance : ℝ) (a_time b_time : ℝ) 
  (h1 : race_distance = 120)
  (h2 : a_time = 36)
  (h3 : b_time = 45) : 
  race_distance - (race_distance / b_time * a_time) = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_difference_l3297_329791


namespace NUMINAMATH_CALUDE_garage_to_other_rooms_ratio_l3297_329706

/-- Given the number of bulbs needed for other rooms and the total number of bulbs Sean has,
    prove that the ratio of garage bulbs to other room bulbs is 1:2. -/
theorem garage_to_other_rooms_ratio
  (other_rooms_bulbs : ℕ)
  (total_packs : ℕ)
  (bulbs_per_pack : ℕ)
  (h1 : other_rooms_bulbs = 8)
  (h2 : total_packs = 6)
  (h3 : bulbs_per_pack = 2) :
  (total_packs * bulbs_per_pack - other_rooms_bulbs) / other_rooms_bulbs = 1 / 2 := by
  sorry

#check garage_to_other_rooms_ratio

end NUMINAMATH_CALUDE_garage_to_other_rooms_ratio_l3297_329706


namespace NUMINAMATH_CALUDE_constant_distance_vector_l3297_329768

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem constant_distance_vector (a b p : V) :
  ‖p - b‖ = 3 * ‖p - a‖ →
  ∃ (c : ℝ), ∀ (q : V), ‖p - q‖ = c ↔ q = (9/8 : ℝ) • a - (1/8 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_constant_distance_vector_l3297_329768


namespace NUMINAMATH_CALUDE_value_of_x_l3297_329797

theorem value_of_x : ∃ X : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * X = (1/2 : ℚ) * (1/4 : ℚ) * 120 ∧ X = 160 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l3297_329797


namespace NUMINAMATH_CALUDE_complex_square_root_minus_100_plus_44i_l3297_329769

theorem complex_square_root_minus_100_plus_44i :
  {z : ℂ | z^2 = -100 + 44*I} = {2 + 11*I, -2 - 11*I} := by sorry

end NUMINAMATH_CALUDE_complex_square_root_minus_100_plus_44i_l3297_329769


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l3297_329790

/-- Given that y varies inversely as √x and y = 3 when x = 4, prove that y = √2 when x = 18 -/
theorem inverse_variation_sqrt (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, y x * Real.sqrt x = k) →  -- y varies inversely as √x
  y 4 = 3 →                      -- y = 3 when x = 4
  y 18 = Real.sqrt 2 :=          -- y = √2 when x = 18
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l3297_329790


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3297_329759

theorem reciprocal_of_negative_two :
  (1 : ℚ) / (-2 : ℚ) = -1/2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l3297_329759


namespace NUMINAMATH_CALUDE_ripe_orange_harvest_l3297_329778

/-- The number of days of harvest -/
def harvest_days : ℕ := 73

/-- The number of sacks of ripe oranges harvested per day -/
def daily_ripe_harvest : ℕ := 5

/-- The total number of sacks of ripe oranges harvested over the entire period -/
def total_ripe_harvest : ℕ := harvest_days * daily_ripe_harvest

theorem ripe_orange_harvest :
  total_ripe_harvest = 365 := by
  sorry

end NUMINAMATH_CALUDE_ripe_orange_harvest_l3297_329778


namespace NUMINAMATH_CALUDE_probability_red_ball_specific_l3297_329717

/-- The probability of drawing a red ball from a bag with specified ball counts. -/
def probability_red_ball (red_count black_count white_count : ℕ) : ℚ :=
  red_count / (red_count + black_count + white_count)

/-- Theorem: The probability of drawing a red ball from a bag with 3 red balls,
    5 black balls, and 4 white balls is 1/4. -/
theorem probability_red_ball_specific : probability_red_ball 3 5 4 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_specific_l3297_329717


namespace NUMINAMATH_CALUDE_minimum_total_cost_l3297_329742

-- Define the ticket prices
def price_cheap : ℕ := 60
def price_expensive : ℕ := 100

-- Define the total number of tickets
def total_tickets : ℕ := 140

-- Define the function to calculate the total cost
def total_cost (cheap_tickets expensive_tickets : ℕ) : ℕ :=
  cheap_tickets * price_cheap + expensive_tickets * price_expensive

-- State the theorem
theorem minimum_total_cost :
  ∃ (cheap_tickets expensive_tickets : ℕ),
    cheap_tickets + expensive_tickets = total_tickets ∧
    expensive_tickets ≥ 2 * cheap_tickets ∧
    ∀ (c e : ℕ),
      c + e = total_tickets →
      e ≥ 2 * c →
      total_cost cheap_tickets expensive_tickets ≤ total_cost c e ∧
      total_cost cheap_tickets expensive_tickets = 12160 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_total_cost_l3297_329742


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l3297_329700

theorem joshua_bottle_caps (initial bought given_away : ℕ) : 
  initial = 150 → bought = 23 → given_away = 37 → 
  initial + bought - given_away = 136 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l3297_329700


namespace NUMINAMATH_CALUDE_inequality_solution_min_value_theorem_equality_condition_l3297_329772

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 0 < x ∧ x < 2/3}

-- Theorem for the solution set of the inequality
theorem inequality_solution : 
  {x : ℝ | f x + |x + 1| < 2} = solution_set :=
sorry

-- Theorem for the minimum value of (4/m) + (1/n)
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (a : ℝ), (∀ x : ℝ, g x ≥ a) ∧ m + n = a) →
  (4/m + 1/n ≥ 9/2) :=
sorry

-- Theorem for the equality condition
theorem equality_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∃ (a : ℝ), (∀ x : ℝ, g x ≥ a) ∧ m + n = a) →
  (4/m + 1/n = 9/2 ↔ m = 4/3 ∧ n = 2/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_min_value_theorem_equality_condition_l3297_329772


namespace NUMINAMATH_CALUDE_range_of_a_l3297_329789

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 < 0) → a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3297_329789


namespace NUMINAMATH_CALUDE_star_example_l3297_329761

-- Define the * operation
def star (a b c d : ℚ) : ℚ := a * c * (d / (b + 1))

-- Theorem statement
theorem star_example : star 5 11 9 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l3297_329761


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3297_329793

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 - 4) + 4 / (2*x^2 + 7*x + 6) ≤ 1 / (2*x + 3) + 4 / (2*x^3 + 3*x^2 - 8*x - 12)) ↔ 
  (x ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ico 1 2 ∪ Set.Ici 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3297_329793


namespace NUMINAMATH_CALUDE_student_sample_total_prove_student_sample_size_l3297_329774

/-- Represents the composition of students in a high school sample -/
structure StudentSample where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The theorem stating the total number of students in the sample -/
theorem student_sample_total (s : StudentSample) : s.total = 800 :=
  by
  have h1 : s.juniors = (28 : ℕ) * s.total / 100 := sorry
  have h2 : s.sophomores = (25 : ℕ) * s.total / 100 := sorry
  have h3 : s.seniors = 160 := sorry
  have h4 : s.freshmen = s.sophomores + 16 := sorry
  have h5 : s.total = s.freshmen + s.sophomores + s.juniors + s.seniors := sorry
  sorry

/-- The main theorem proving the total number of students -/
theorem prove_student_sample_size : ∃ s : StudentSample, s.total = 800 :=
  by
  sorry

end NUMINAMATH_CALUDE_student_sample_total_prove_student_sample_size_l3297_329774


namespace NUMINAMATH_CALUDE_distance_between_red_lights_l3297_329779

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Calculates the position of the nth red light in the sequence -/
def redLightPosition (n : ℕ) : ℕ :=
  (n - 1) / 3 * 7 + (n - 1) % 3 + 1

/-- The distance between lights in inches -/
def light_spacing : ℕ := 8

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The main theorem stating the distance between the 4th and 19th red lights -/
theorem distance_between_red_lights :
  (redLightPosition 19 - redLightPosition 4) * light_spacing / inches_per_foot = 
    (22671 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_distance_between_red_lights_l3297_329779


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3297_329796

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields/axioms for a line in 3D space
  -- This is a simplified representation

/-- Represents perpendicularity between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be perpendicular
  sorry

/-- Represents parallelism between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be parallel
  sorry

/-- Theorem: If line a is parallel to line b, and line l is perpendicular to a,
    then l is also perpendicular to b -/
theorem perpendicular_parallel_transitive (a b l : Line3D) :
  parallel a b → perpendicular l a → perpendicular l b := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3297_329796


namespace NUMINAMATH_CALUDE_census_set_is_population_l3297_329714

/-- The term for the entire set of objects to be investigated in a census -/
def census_set : String := "Population"

/-- Theorem stating that the entire set of objects to be investigated in a census is called "Population" -/
theorem census_set_is_population : census_set = "Population" := by
  sorry

end NUMINAMATH_CALUDE_census_set_is_population_l3297_329714


namespace NUMINAMATH_CALUDE_max_sundays_in_56_days_l3297_329701

theorem max_sundays_in_56_days : ℕ := by
  -- Define the number of days
  let days : ℕ := 56
  
  -- Define the number of days in a week
  let days_per_week : ℕ := 7
  
  -- Define that each week has one Sunday
  let sundays_per_week : ℕ := 1
  
  -- The maximum number of Sundays is the number of complete weeks in 56 days
  have max_sundays : ℕ := days / days_per_week * sundays_per_week
  
  -- Assert that this equals 8
  have : max_sundays = 8 := by sorry
  
  -- Return the result
  exact max_sundays

end NUMINAMATH_CALUDE_max_sundays_in_56_days_l3297_329701


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3297_329770

/-- Given a geometric sequence {a_n} with sum of first n terms S_n,
    if a_1 + a_3 = 5/4 and a_2 + a_4 = 5/2, then S_6 / S_3 = 9 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5/4)
  (h2 : a 2 + a 4 = 5/2)
  (h_geom : ∀ n : ℕ, a (n+1) / a n = a 2 / a 1)
  (h_sum : ∀ n : ℕ, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) :
  S 6 / S 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3297_329770


namespace NUMINAMATH_CALUDE_smallest_d_inequality_l3297_329722

theorem smallest_d_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x * y) + (x^2 - y^2)^2 ≥ x + y ∧
  ∀ d : ℝ, d > 0 → d < 1 →
    ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + d * (x^2 - y^2)^2 < x + y :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_inequality_l3297_329722


namespace NUMINAMATH_CALUDE_white_to_red_black_ratio_l3297_329738

/-- Represents the number of socks James has -/
structure Socks :=
  (red : ℕ)
  (black : ℕ)
  (white : ℕ)

/-- The total number of socks James has -/
def total_socks (s : Socks) : ℕ := s.red + s.black + s.white

/-- The theorem stating the ratio of white socks to red and black socks -/
theorem white_to_red_black_ratio (s : Socks) :
  s.red = 40 →
  s.black = 20 →
  s.white = s.red + s.black →
  total_socks s = 90 →
  s.white * 2 = s.red + s.black :=
by
  sorry


end NUMINAMATH_CALUDE_white_to_red_black_ratio_l3297_329738


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l3297_329782

/-- Given an ellipse defined by 4x^2 + 9y^2 = 144 and a point P(3, 2) inside it,
    the slope of the line containing the chord with P as its midpoint is -2/3 -/
theorem ellipse_chord_slope (x y : ℝ) :
  4 * x^2 + 9 * y^2 = 144 →  -- Ellipse equation
  ∃ (x1 y1 x2 y2 : ℝ),       -- Endpoints of the chord
    4 * x1^2 + 9 * y1^2 = 144 ∧   -- First endpoint on ellipse
    4 * x2^2 + 9 * y2^2 = 144 ∧   -- Second endpoint on ellipse
    (x1 + x2) / 2 = 3 ∧           -- P is midpoint (x-coordinate)
    (y1 + y2) / 2 = 2 →           -- P is midpoint (y-coordinate)
    (y2 - y1) / (x2 - x1) = -2/3  -- Slope of the chord
:= by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l3297_329782


namespace NUMINAMATH_CALUDE_express_regular_speed_ratio_l3297_329740

/-- The speed ratio of express train to regular train -/
def speed_ratio : ℝ := 2.5

/-- Regular train travel time in hours -/
def regular_time : ℝ := 10

/-- Time difference between regular and express train arrival in hours -/
def time_difference : ℝ := 3

/-- Time after departure when both trains are at same distance from Moscow -/
def distance_equality_time : ℝ := 2

/-- Minimum waiting time for express train in hours -/
def min_wait_time : ℝ := 2.5

theorem express_regular_speed_ratio 
  (wait_time : ℝ) 
  (h_wait : wait_time > min_wait_time) 
  (h_express_time : regular_time - time_difference - wait_time > 0) 
  (h_distance_equality : 
    distance_equality_time * speed_ratio = (distance_equality_time + wait_time)) :
  speed_ratio = (wait_time + distance_equality_time) / distance_equality_time :=
sorry

end NUMINAMATH_CALUDE_express_regular_speed_ratio_l3297_329740


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3297_329783

def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3297_329783


namespace NUMINAMATH_CALUDE_sample_size_for_295_students_l3297_329745

/-- Calculates the sample size for systematic sampling --/
def calculateSampleSize (totalStudents : Nat) (samplingRatio : Nat) : Nat :=
  totalStudents / samplingRatio

/-- Theorem: The sample size for 295 students with a 1:5 sampling ratio is 59 --/
theorem sample_size_for_295_students :
  calculateSampleSize 295 5 = 59 := by
  sorry


end NUMINAMATH_CALUDE_sample_size_for_295_students_l3297_329745


namespace NUMINAMATH_CALUDE_bathroom_visits_time_l3297_329731

/-- Given that it takes 20 minutes for 8 bathroom visits, prove that 6 visits take 15 minutes. -/
theorem bathroom_visits_time (total_time : ℝ) (total_visits : ℕ) (target_visits : ℕ)
  (h1 : total_time = 20)
  (h2 : total_visits = 8)
  (h3 : target_visits = 6) :
  (total_time / total_visits) * target_visits = 15 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_visits_time_l3297_329731


namespace NUMINAMATH_CALUDE_total_cds_l3297_329798

def dawn_cds : ℕ := 10
def kristine_cds : ℕ := dawn_cds + 7

theorem total_cds : dawn_cds + kristine_cds = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_cds_l3297_329798


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l3297_329784

/-- The number of grandchildren --/
def n : ℕ := 12

/-- The probability of a child being male or female --/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters --/
def unequal_probability : ℚ := 793/1024

theorem unequal_gender_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = unequal_probability :=
sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l3297_329784


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l3297_329720

/-- Converts a base-7 number represented as a list of digits to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number -/
def base7Number : List Nat := [5, 4, 6]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 327 := by sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l3297_329720


namespace NUMINAMATH_CALUDE_product_repeating_decimal_three_and_eight_l3297_329757

/-- The product of 0.3̄ and 8 is equal to 8/3 -/
theorem product_repeating_decimal_three_and_eight :
  (∃ x : ℚ, x = 1/3 ∧ (∃ d : ℕ → ℕ, ∀ n, d n < 10 ∧ x = ∑' k, (d k : ℚ) / 10^(k+1)) ∧ x * 8 = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_three_and_eight_l3297_329757


namespace NUMINAMATH_CALUDE_range_of_a_l3297_329752

-- Define the inequality system
def inequality_system (x a : ℝ) : Prop :=
  (((x + 2) / 3 - x / 2) > 1) ∧ (2 * (x - a) ≤ 0)

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, inequality_system x a ↔ solution_set x) →
  a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3297_329752


namespace NUMINAMATH_CALUDE_solutions_for_twenty_initial_conditions_arithmetic_progression_l3297_329743

/-- The number of integer solutions for |x| + |y| = n -/
def numSolutions (n : ℕ) : ℕ := 4 * n

theorem solutions_for_twenty :
  numSolutions 20 = 80 :=
by sorry

/-- Verifies that the first three terms match the given conditions -/
theorem initial_conditions :
  numSolutions 1 = 4 ∧ numSolutions 2 = 8 ∧ numSolutions 3 = 12 :=
by sorry

/-- The sequence of solutions forms an arithmetic progression -/
theorem arithmetic_progression (n : ℕ) :
  numSolutions (n + 1) - numSolutions n = 4 :=
by sorry

end NUMINAMATH_CALUDE_solutions_for_twenty_initial_conditions_arithmetic_progression_l3297_329743


namespace NUMINAMATH_CALUDE_election_votes_calculation_l3297_329725

theorem election_votes_calculation (total_votes : ℕ) : 
  (4 : ℕ) ≤ total_votes ∧ 
  (total_votes : ℚ) * (1/2) - (total_votes : ℚ) * (1/4) = 174 →
  total_votes = 696 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l3297_329725


namespace NUMINAMATH_CALUDE_rotating_ngon_path_area_theorem_l3297_329715

/-- Represents a regular n-gon -/
structure RegularNGon where
  n : ℕ
  sideLength : ℝ

/-- The area enclosed by the path of a rotating n-gon vertex -/
def rotatingNGonPathArea (g : RegularNGon) : ℝ := sorry

/-- The area of a regular n-gon -/
def regularNGonArea (g : RegularNGon) : ℝ := sorry

/-- Theorem: The area enclosed by the rotating n-gon vertex path
    equals four times the area of the original n-gon -/
theorem rotating_ngon_path_area_theorem (g : RegularNGon) 
    (h1 : g.sideLength = 1) :
  rotatingNGonPathArea g = 4 * regularNGonArea g := by
  sorry

end NUMINAMATH_CALUDE_rotating_ngon_path_area_theorem_l3297_329715


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_cube_l3297_329703

theorem power_of_two_greater_than_cube (n : ℕ) (h : n ≥ 10) : 2^n > n^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_cube_l3297_329703


namespace NUMINAMATH_CALUDE_bird_round_trips_l3297_329733

/-- Given two birds collecting nest materials, this theorem proves the number of round trips each bird made. -/
theorem bird_round_trips (distance_to_materials : ℕ) (total_distance : ℕ) : 
  distance_to_materials = 200 →
  total_distance = 8000 →
  ∃ (trips_per_bird : ℕ), 
    trips_per_bird * 2 * (2 * distance_to_materials) = total_distance ∧
    trips_per_bird = 10 := by
  sorry

end NUMINAMATH_CALUDE_bird_round_trips_l3297_329733


namespace NUMINAMATH_CALUDE_problem_solution_l3297_329708

theorem problem_solution : ∃ x : ℝ, 
  (0.6 * x = 0.3 * (125 ^ (1/3 : ℝ)) + 27) ∧ x = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3297_329708


namespace NUMINAMATH_CALUDE_dawsons_b_students_l3297_329746

/-- Proves that given the conditions from the problem, the number of students
    receiving a 'B' in Mr. Dawson's class is 18. -/
theorem dawsons_b_students
  (carter_total : ℕ)
  (carter_b : ℕ)
  (dawson_total : ℕ)
  (h1 : carter_total = 20)
  (h2 : carter_b = 12)
  (h3 : dawson_total = 30)
  (h4 : (carter_b : ℚ) / carter_total = dawson_b / dawson_total) :
  dawson_b = 18 := by
  sorry

#check dawsons_b_students

end NUMINAMATH_CALUDE_dawsons_b_students_l3297_329746


namespace NUMINAMATH_CALUDE_solution_value_l3297_329704

theorem solution_value (a b : ℝ) : 
  (2 * 2 * a - 2 * b - 20 = 0) → (2023 - 2 * a + b = 2013) := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3297_329704


namespace NUMINAMATH_CALUDE_least_divisible_by_960_sixty_divisible_by_960_least_value_is_60_l3297_329749

theorem least_divisible_by_960 (a : ℕ) : a^5 % 960 = 0 → a ≥ 60 := by
  sorry

theorem sixty_divisible_by_960 : (60 : ℕ)^5 % 960 = 0 := by
  sorry

theorem least_value_is_60 : ∃ a : ℕ, a^5 % 960 = 0 ∧ ∀ b : ℕ, b^5 % 960 = 0 → b ≥ a := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_960_sixty_divisible_by_960_least_value_is_60_l3297_329749


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l3297_329788

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 878 / 1000)
  : (total_bananas - (total_oranges + total_bananas - 
     (good_fruits_percentage * (total_oranges + total_bananas)).floor - 
     (rotten_oranges_percentage * total_oranges).floor)) / total_bananas = 8 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l3297_329788


namespace NUMINAMATH_CALUDE_expand_product_l3297_329737

theorem expand_product (x : ℝ) : (x + 4) * (x^2 - 9) = x^3 + 4*x^2 - 9*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3297_329737


namespace NUMINAMATH_CALUDE_students_per_group_l3297_329781

/-- Given a total of 64 students, with 36 not picked, and divided into 4 groups,
    prove that there are 7 students in each group. -/
theorem students_per_group :
  ∀ (total : ℕ) (not_picked : ℕ) (groups : ℕ),
    total = 64 →
    not_picked = 36 →
    groups = 4 →
    (total - not_picked) / groups = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l3297_329781


namespace NUMINAMATH_CALUDE_money_distribution_l3297_329785

theorem money_distribution (m l n : ℚ) (h1 : m > 0) (h2 : l > 0) (h3 : n > 0) 
  (h4 : m / 5 = l / 3) (h5 : m / 5 = n / 2) : 
  (3 * (m / 5)) / (m + l + n) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3297_329785


namespace NUMINAMATH_CALUDE_mixture_composition_l3297_329747

/-- Represents a seed mixture --/
structure SeedMixture where
  ryegrass : ℝ
  other : ℝ
  sum_to_one : ryegrass + other = 1

/-- The final mixture of X and Y --/
def final_mixture (x y : SeedMixture) (p : ℝ) : SeedMixture :=
  { ryegrass := p * x.ryegrass + (1 - p) * y.ryegrass,
    other := p * x.other + (1 - p) * y.other,
    sum_to_one := by sorry }

theorem mixture_composition 
  (x : SeedMixture)
  (y : SeedMixture)
  (hx : x.ryegrass = 0.4)
  (hy : y.ryegrass = 0.25)
  : ∃ p : ℝ, 
    0 ≤ p ∧ p ≤ 1 ∧ 
    (final_mixture x y p).ryegrass = 0.38 ∧
    abs (p - 0.8667) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_mixture_composition_l3297_329747


namespace NUMINAMATH_CALUDE_lcm_of_20_45_28_l3297_329755

theorem lcm_of_20_45_28 : Nat.lcm (Nat.lcm 20 45) 28 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_28_l3297_329755


namespace NUMINAMATH_CALUDE_area_of_tangent_square_l3297_329741

/-- Given a 6 by 6 square with four semicircles on its sides, and another square EFGH
    with sides parallel and tangent to the semicircles, the area of EFGH is 144. -/
theorem area_of_tangent_square (original_side_length : ℝ) (EFGH_side_length : ℝ) : 
  original_side_length = 6 →
  EFGH_side_length = original_side_length + 2 * (original_side_length / 2) →
  EFGH_side_length ^ 2 = 144 := by
sorry

end NUMINAMATH_CALUDE_area_of_tangent_square_l3297_329741


namespace NUMINAMATH_CALUDE_expression_values_l3297_329718

theorem expression_values (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (gcd_ab : Nat.gcd a b = 1) (gcd_bc : Nat.gcd b c = 1) (gcd_ca : Nat.gcd c a = 1) :
  (a + b) / c + (b + c) / a + (c + a) / b = 7 ∨ (a + b) / c + (b + c) / a + (c + a) / b = 8 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l3297_329718


namespace NUMINAMATH_CALUDE_elmer_eats_more_l3297_329711

/-- The amount of food each animal eats per day in pounds -/
structure AnimalFood where
  penelope : ℝ
  greta : ℝ
  milton : ℝ
  elmer : ℝ
  rosie : ℝ
  carl : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (food : AnimalFood) : Prop :=
  food.penelope = 20 ∧
  food.penelope = 10 * food.greta ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.rosie = 3 * food.greta ∧
  food.carl = food.penelope / 2 ∧
  food.carl = 5 * food.greta

/-- The theorem to prove -/
theorem elmer_eats_more (food : AnimalFood) (h : satisfiesConditions food) :
    food.elmer - (food.penelope + food.greta + food.milton + food.rosie + food.carl) = 41.98 := by
  sorry

end NUMINAMATH_CALUDE_elmer_eats_more_l3297_329711


namespace NUMINAMATH_CALUDE_r_earnings_l3297_329792

def daily_earnings (p q r : ℝ) : Prop :=
  9 * (p + q + r) = 1980 ∧
  5 * (p + r) = 600 ∧
  7 * (q + r) = 910

theorem r_earnings (p q r : ℝ) (h : daily_earnings p q r) : r = 30 := by
  sorry

end NUMINAMATH_CALUDE_r_earnings_l3297_329792


namespace NUMINAMATH_CALUDE_inequality_problem_l3297_329758

theorem inequality_problem (a b : ℝ) (h : a ≠ b) :
  (a^2 + b^2 ≥ 2*(a - b - 1)) ∧
  ¬(∀ a b : ℝ, a + b > 2*b^2) ∧
  ¬(∀ a b : ℝ, a^5 + b^5 > a^3*b^2 + a^2*b^3) ∧
  ¬(∀ a b : ℝ, b/a + a/b > 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l3297_329758


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_l3297_329748

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  -- Asymptote equations
  asymptote1 : Real → Real
  asymptote2 : Real → Real
  -- Point the hyperbola passes through
  point : Real × Real
  -- Conditions on asymptotes
  asymptote1_eq : ∀ x, asymptote1 x = 2 * x + 5
  asymptote2_eq : ∀ x, asymptote2 x = -2 * x + 1
  -- Condition on the point
  point_eq : point = (0, 7)

/-- The standard form of a hyperbola -/
def standard_form (h k a b : Real) (x y : Real) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- Theorem stating the sum of a and h for the given hyperbola -/
theorem hyperbola_a_plus_h (H : Hyperbola) :
  ∃ (h k a b : Real), a > 0 ∧ b > 0 ∧
  (∀ x y, standard_form h k a b x y ↔ H.point = (x, y)) →
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_l3297_329748


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l3297_329726

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_yellow = 0.2 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_green = 0.3 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l3297_329726


namespace NUMINAMATH_CALUDE_y_derivative_l3297_329710

noncomputable def y (x : ℝ) : ℝ :=
  x - Real.log (1 + Real.exp x) - 2 * Real.exp (-x/2) * Real.arctan (Real.exp (x/2)) - (Real.arctan (Real.exp (x/2)))^2

theorem y_derivative (x : ℝ) :
  deriv y x = Real.arctan (Real.exp (x/2)) / (Real.exp (x/2) * (1 + Real.exp x)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3297_329710


namespace NUMINAMATH_CALUDE_market_fruit_count_l3297_329753

/-- Calculates the total number of apples and oranges in a market -/
def total_fruits (num_apples : ℕ) (apple_orange_diff : ℕ) : ℕ :=
  num_apples + (num_apples - apple_orange_diff)

/-- Theorem: Given a market with 164 apples and 27 more apples than oranges,
    the total number of apples and oranges is 301 -/
theorem market_fruit_count : total_fruits 164 27 = 301 := by
  sorry

end NUMINAMATH_CALUDE_market_fruit_count_l3297_329753


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3297_329780

theorem sum_of_fractions : 
  (2 / 10 : ℚ) + (3 / 10 : ℚ) + (5 / 10 : ℚ) + (6 / 10 : ℚ) + (7 / 10 : ℚ) + 
  (9 / 10 : ℚ) + (14 / 10 : ℚ) + (15 / 10 : ℚ) + (20 / 10 : ℚ) + (41 / 10 : ℚ) = 
  (122 / 10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3297_329780


namespace NUMINAMATH_CALUDE_sixth_employee_salary_l3297_329777

/-- Given the salaries of 5 employees and the mean salary of all 6 employees,
    prove that the salary of the sixth employee is equal to the difference between
    the total salary of all 6 employees and the sum of the known 5 salaries. -/
theorem sixth_employee_salary
  (salary1 salary2 salary3 salary4 salary5 : ℝ)
  (mean_salary : ℝ)
  (h1 : salary1 = 1000)
  (h2 : salary2 = 2500)
  (h3 : salary3 = 3100)
  (h4 : salary4 = 1500)
  (h5 : salary5 = 2000)
  (h_mean : mean_salary = 2291.67)
  : ∃ (salary6 : ℝ),
    salary6 = 6 * mean_salary - (salary1 + salary2 + salary3 + salary4 + salary5) :=
by sorry

end NUMINAMATH_CALUDE_sixth_employee_salary_l3297_329777


namespace NUMINAMATH_CALUDE_ellipse_properties_l3297_329764

/-- Given an ellipse with equation x²/m + y²/(m/(m+3)) = 1 where m > 0,
    and eccentricity e = √3/2, prove the following properties. -/
theorem ellipse_properties (m : ℝ) (h_m : m > 0) :
  let e := Real.sqrt 3 / 2
  let a := Real.sqrt m
  let b := Real.sqrt (m / (m + 3))
  let c := Real.sqrt ((m * (m + 2)) / (m + 3))
  (e = c / a) →
  (m = 1 ∧
   2 * a = 2 ∧ 2 * b = 1 ∧
   c = Real.sqrt 3 / 2 ∧
   a = 1 ∧ b = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3297_329764


namespace NUMINAMATH_CALUDE_chinese_chess_sets_l3297_329794

theorem chinese_chess_sets (go_cost : ℕ) (chinese_chess_cost : ℕ) (total_sets : ℕ) (total_cost : ℕ) :
  go_cost = 24 →
  chinese_chess_cost = 18 →
  total_sets = 14 →
  total_cost = 300 →
  ∃ (go_sets chinese_chess_sets : ℕ),
    go_sets + chinese_chess_sets = total_sets ∧
    go_cost * go_sets + chinese_chess_cost * chinese_chess_sets = total_cost ∧
    chinese_chess_sets = 6 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_sets_l3297_329794


namespace NUMINAMATH_CALUDE_odd_expressions_l3297_329763

theorem odd_expressions (m n p : ℕ) 
  (hm : m % 2 = 1) 
  (hn : n % 2 = 1) 
  (hp : p % 2 = 0) 
  (hm_pos : 0 < m) 
  (hn_pos : 0 < n) 
  (hp_pos : 0 < p) : 
  ((2 * m * n + 5)^2) % 2 = 1 ∧ (5 * m * n + p) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_expressions_l3297_329763


namespace NUMINAMATH_CALUDE_arithmetic_sequence_transformation_l3297_329786

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given arithmetic sequence and its transformation -/
theorem arithmetic_sequence_transformation
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : ∀ n : ℕ, b n = 3 * a n + 4) :
  ArithmeticSequence b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_transformation_l3297_329786


namespace NUMINAMATH_CALUDE_parabola_equation_l3297_329729

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  focus_x : ℝ
  focus_x_pos : focus_x > 0

/-- The line y = x -/
def line_y_eq_x (x : ℝ) : ℝ := x

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_equation (C : Parabola) 
  (A B : Point) 
  (P : Point)
  (h1 : line_y_eq_x A.x = A.y ∧ line_y_eq_x B.x = B.y)  -- A and B lie on y = x
  (h2 : P.x = 2 ∧ P.y = 2)  -- P is (2,2)
  (h3 : P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2)  -- P is midpoint of AB
  : ∀ (x y : ℝ), (y^2 = 4*x) ↔ (∃ (t : ℝ), x = t^2 * C.focus_x ∧ y = 2*t * C.focus_x) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3297_329729


namespace NUMINAMATH_CALUDE_max_a_value_l3297_329787

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x + 1)

noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 3 * x + 1)

theorem max_a_value :
  (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, f x₁ = g a x₂) →
  ∀ a' : ℝ, (∀ x₁ : ℝ, x₁ ≥ 0 → ∃ x₂ : ℝ, f x₁ = g a' x₂) →
  a' ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3297_329787


namespace NUMINAMATH_CALUDE_reachable_points_characterization_l3297_329750

-- Define the road as a line
def Road : Type := ℝ

-- Define a point in the 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the tourist's starting point A
def A : Point := ⟨0, 0⟩

-- Define the tourist's speed on the road
def roadSpeed : ℝ := 6

-- Define the tourist's speed on the field
def fieldSpeed : ℝ := 3

-- Define the time limit
def timeLimit : ℝ := 1

-- Define the set of reachable points
def ReachablePoints : Set Point :=
  {p : Point | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ timeLimit ∧
    (p.x^2 / roadSpeed^2 + p.y^2 / fieldSpeed^2 ≤ t^2)}

-- Define the line segment on the road
def RoadSegment : Set Point :=
  {p : Point | p.y = 0 ∧ |p.x| ≤ roadSpeed * timeLimit}

-- Define the semicircles
def Semicircles : Set Point :=
  {p : Point | ∃ (c : ℝ), 
    c = roadSpeed * timeLimit ∧
    ((p.x - c)^2 + p.y^2 ≤ (fieldSpeed * timeLimit)^2 ∨
     (p.x + c)^2 + p.y^2 ≤ (fieldSpeed * timeLimit)^2) ∧
    p.y ≥ 0}

-- Theorem statement
theorem reachable_points_characterization :
  ReachablePoints = RoadSegment ∪ Semicircles :=
sorry

end NUMINAMATH_CALUDE_reachable_points_characterization_l3297_329750


namespace NUMINAMATH_CALUDE_average_transformation_l3297_329716

theorem average_transformation (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) : 
  ((3*x₁ + 1) + (3*x₂ + 1) + (3*x₃ + 1) + (3*x₄ + 1) + (3*x₅ + 1)) / 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l3297_329716


namespace NUMINAMATH_CALUDE_machine_A_time_l3297_329773

/-- The time it takes for machines A, B, and C to finish a job together -/
def combined_time : ℝ := 2.181818181818182

/-- The time it takes for machine B to finish the job alone -/
def time_B : ℝ := 12

/-- The time it takes for machine C to finish the job alone -/
def time_C : ℝ := 8

/-- Theorem stating that if machines A, B, and C working together can finish a job in 
    2.181818181818182 hours, machine B alone takes 12 hours, and machine C alone takes 8 hours, 
    then machine A alone takes 4 hours to finish the job -/
theorem machine_A_time : 
  ∃ (time_A : ℝ), 
    1 / time_A + 1 / time_B + 1 / time_C = 1 / combined_time ∧ 
    time_A = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_A_time_l3297_329773


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_is_122_l3297_329775

/-- Represents a cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ := 1
  b : ℤ := 0
  c : ℤ := -707
  d : ℤ

/-- Predicate to check if a given integer is a root of the polynomial -/
def is_root (p : CubicPolynomial) (x : ℤ) : Prop :=
  p.a * x^3 + p.b * x^2 + p.c * x + p.d = 0

/-- Theorem stating the sum of absolute values of roots -/
theorem sum_of_abs_roots_is_122 (m : ℤ) (p q r : ℤ) :
  let poly : CubicPolynomial := { d := m }
  (is_root poly p) ∧ (is_root poly q) ∧ (is_root poly r) →
  |p| + |q| + |r| = 122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_is_122_l3297_329775


namespace NUMINAMATH_CALUDE_fabian_walnuts_amount_l3297_329735

/-- The amount of walnuts in grams that Fabian wants to buy -/
def walnuts_amount (apple_kg : ℕ) (sugar_packs : ℕ) (total_cost : ℕ) 
  (apple_price : ℕ) (walnut_price : ℕ) (sugar_discount : ℕ) : ℕ :=
  let apple_cost := apple_kg * apple_price
  let sugar_price := apple_price - sugar_discount
  let sugar_cost := sugar_packs * sugar_price
  let walnut_cost := total_cost - apple_cost - sugar_cost
  let walnut_grams_per_dollar := 1000 / walnut_price
  walnut_cost * walnut_grams_per_dollar

/-- Theorem stating that Fabian wants to buy 500 grams of walnuts -/
theorem fabian_walnuts_amount : 
  walnuts_amount 5 3 16 2 6 1 = 500 := by
  sorry

end NUMINAMATH_CALUDE_fabian_walnuts_amount_l3297_329735


namespace NUMINAMATH_CALUDE_option_b_neither_parallel_nor_perpendicular_l3297_329760

/-- Two vectors in R³ -/
structure VectorPair where
  μ : Fin 3 → ℝ
  v : Fin 3 → ℝ

/-- Check if two vectors are parallel -/
def isParallel (pair : VectorPair) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, pair.μ i = k * pair.v i)

/-- Check if two vectors are perpendicular -/
def isPerpendicular (pair : VectorPair) : Prop :=
  (pair.μ 0 * pair.v 0 + pair.μ 1 * pair.v 1 + pair.μ 2 * pair.v 2) = 0

/-- The specific vector pair for option B -/
def optionB : VectorPair where
  μ := ![3, 0, -1]
  v := ![0, 0, 2]

/-- Theorem stating that the vectors in option B are neither parallel nor perpendicular -/
theorem option_b_neither_parallel_nor_perpendicular :
  ¬(isParallel optionB) ∧ ¬(isPerpendicular optionB) := by
  sorry


end NUMINAMATH_CALUDE_option_b_neither_parallel_nor_perpendicular_l3297_329760


namespace NUMINAMATH_CALUDE_twelfth_term_of_ap_l3297_329728

-- Define the arithmetic progression
def arithmeticProgression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- State the theorem
theorem twelfth_term_of_ap : arithmeticProgression 2 8 12 = 90 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_ap_l3297_329728


namespace NUMINAMATH_CALUDE_function_symmetry_property_l3297_329702

open Real

/-- Given a function f(x) = a cos(x) + bx² + 2, prove that
    f(2016) - f(-2016) + f''(2017) + f''(-2017) = 0 for any real a and b -/
theorem function_symmetry_property (a b : ℝ) :
  let f := fun x => a * cos x + b * x^2 + 2
  let f'' := fun x => -a * cos x + 2 * b
  f 2016 - f (-2016) + f'' 2017 + f'' (-2017) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_property_l3297_329702


namespace NUMINAMATH_CALUDE_fourth_person_height_l3297_329736

/-- Proves that the height of the fourth person is 82 inches given the conditions -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧  -- Heights in increasing order
  h₂ = h₁ + 2 ∧                  -- Difference between 1st and 2nd
  h₃ = h₂ + 2 ∧                  -- Difference between 2nd and 3rd
  h₄ = h₃ + 6 ∧                  -- Difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76   -- Average height
  → h₄ = 82 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l3297_329736


namespace NUMINAMATH_CALUDE_lawyer_percentage_l3297_329709

theorem lawyer_percentage (total : ℝ) (h1 : total > 0) : 
  let women_ratio : ℝ := 0.9
  let women_lawyer_prob : ℝ := 0.54
  let women_count : ℝ := women_ratio * total
  let lawyer_ratio : ℝ := women_lawyer_prob / women_ratio
  lawyer_ratio = 0.6 := by sorry

end NUMINAMATH_CALUDE_lawyer_percentage_l3297_329709


namespace NUMINAMATH_CALUDE_f_properties_l3297_329721

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + k / x

theorem f_properties (k : ℝ) (h_k : k ≠ 0) (h_f3 : f k 3 = 6) :
  (∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ -3 → f k x₁ < f k x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3297_329721
