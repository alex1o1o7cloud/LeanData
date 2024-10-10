import Mathlib

namespace solutions_for_twenty_initial_conditions_arithmetic_progression_l2543_254384

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

end solutions_for_twenty_initial_conditions_arithmetic_progression_l2543_254384


namespace hyperbola_a_plus_h_l2543_254338

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

end hyperbola_a_plus_h_l2543_254338


namespace tangent_line_condition_minimum_value_condition_min_value_case1_min_value_case2_min_value_case3_l2543_254364

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a

-- Theorem for part 1
theorem tangent_line_condition (a : ℝ) :
  (∃ k, ∀ x, f a x = 2 * x + k - 2 * Real.exp 1) → a = Real.exp 1 :=
sorry

-- Theorem for part 2
theorem minimum_value_condition (a m : ℝ) (h : m > 0) :
  let min_value := min (f a (2 * m)) (min (f a (1 / Real.exp 1)) (f a m))
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ min_value :=
sorry

-- Additional theorems to specify the exact minimum value based on m
theorem min_value_case1 (a m : ℝ) (h1 : m > 0) (h2 : m ≤ 1 / (2 * Real.exp 1)) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a (2 * m) :=
sorry

theorem min_value_case2 (a m : ℝ) (h1 : m > 0) (h2 : 1 / (2 * Real.exp 1) < m) (h3 : m < 1 / Real.exp 1) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a (1 / Real.exp 1) :=
sorry

theorem min_value_case3 (a m : ℝ) (h1 : m > 0) (h2 : m ≥ 1 / Real.exp 1) :
  ∀ x ∈ Set.Icc m (2 * m), f a x ≥ f a m :=
sorry

end tangent_line_condition_minimum_value_condition_min_value_case1_min_value_case2_min_value_case3_l2543_254364


namespace platform_length_calculation_l2543_254305

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ platform_length : ℝ,
    (platform_length > 348) ∧ (platform_length < 349) ∧
    (train_length + platform_length) / time_platform = train_length / time_pole :=
by
  sorry

#check platform_length_calculation

end platform_length_calculation_l2543_254305


namespace blurred_pages_frequency_l2543_254350

theorem blurred_pages_frequency 
  (total_pages : ℕ) 
  (crumpled_frequency : ℕ) 
  (neither_crumpled_nor_blurred : ℕ) 
  (h1 : total_pages = 42)
  (h2 : crumpled_frequency = 7)
  (h3 : neither_crumpled_nor_blurred = 24) :
  (total_pages - neither_crumpled_nor_blurred - (total_pages / crumpled_frequency)) / total_pages = 2 / 7 := by
sorry

end blurred_pages_frequency_l2543_254350


namespace elmer_eats_more_l2543_254353

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

end elmer_eats_more_l2543_254353


namespace joshua_bottle_caps_l2543_254322

theorem joshua_bottle_caps (initial bought given_away : ℕ) : 
  initial = 150 → bought = 23 → given_away = 37 → 
  initial + bought - given_away = 136 := by
  sorry

end joshua_bottle_caps_l2543_254322


namespace bathroom_visits_time_l2543_254355

/-- Given that it takes 20 minutes for 8 bathroom visits, prove that 6 visits take 15 minutes. -/
theorem bathroom_visits_time (total_time : ℝ) (total_visits : ℕ) (target_visits : ℕ)
  (h1 : total_time = 20)
  (h2 : total_visits = 8)
  (h3 : target_visits = 6) :
  (total_time / total_visits) * target_visits = 15 := by
  sorry

end bathroom_visits_time_l2543_254355


namespace pie_point_returns_to_initial_position_l2543_254365

/-- Represents a point on a circular pie --/
structure PiePoint where
  angle : Real
  radius : Real

/-- Represents the operation of cutting, flipping, and rotating the pie --/
def pieOperation (α β : Real) (p : PiePoint) : PiePoint :=
  sorry

/-- The main theorem statement --/
theorem pie_point_returns_to_initial_position
  (α β : Real)
  (h1 : β < α)
  (h2 : α < 180)
  : ∃ N : ℕ, ∀ p : PiePoint,
    (pieOperation α β)^[N] p = p :=
  sorry

end pie_point_returns_to_initial_position_l2543_254365


namespace f_derivative_f_at_one_f_equality_l2543_254315

/-- A function f satisfying f'(x) = 4x^3 for all x and f(1) = -1 -/
def f : ℝ → ℝ :=
  sorry

theorem f_derivative (x : ℝ) : deriv f x = 4 * x^3 :=
  sorry

theorem f_at_one : f 1 = -1 :=
  sorry

theorem f_equality (x : ℝ) : f x = x^4 - 2 :=
  sorry

end f_derivative_f_at_one_f_equality_l2543_254315


namespace square_plus_linear_plus_one_eq_square_l2543_254341

theorem square_plus_linear_plus_one_eq_square (x y : ℕ) :
  y^2 + y + 1 = x^2 ↔ x = 1 ∧ y = 0 := by sorry

end square_plus_linear_plus_one_eq_square_l2543_254341


namespace sample_size_for_295_students_l2543_254357

/-- Calculates the sample size for systematic sampling --/
def calculateSampleSize (totalStudents : Nat) (samplingRatio : Nat) : Nat :=
  totalStudents / samplingRatio

/-- Theorem: The sample size for 295 students with a 1:5 sampling ratio is 59 --/
theorem sample_size_for_295_students :
  calculateSampleSize 295 5 = 59 := by
  sorry


end sample_size_for_295_students_l2543_254357


namespace f_properties_l2543_254389

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + k / x

theorem f_properties (k : ℝ) (h_k : k ≠ 0) (h_f3 : f k 3 = 6) :
  (∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x)) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ -3 → f k x₁ < f k x₂) := by
  sorry

end f_properties_l2543_254389


namespace average_transformation_l2543_254369

theorem average_transformation (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) : 
  ((3*x₁ + 1) + (3*x₂ + 1) + (3*x₃ + 1) + (3*x₄ + 1) + (3*x₅ + 1)) / 5 = 7 := by
  sorry

end average_transformation_l2543_254369


namespace M_intersect_N_empty_l2543_254379

/-- Set M in the complex plane --/
def M : Set ℂ :=
  {z | ∃ t : ℝ, t ≠ -1 ∧ t ≠ 0 ∧ z = t / (1 + t) + Complex.I * (1 + t) / t}

/-- Set N in the complex plane --/
def N : Set ℂ :=
  {z | ∃ t : ℝ, |t| ≤ 1 ∧ z = Real.sqrt 2 * (Complex.cos (Real.arcsin t) + Complex.I * Complex.cos (Real.arccos t))}

/-- The intersection of sets M and N is empty --/
theorem M_intersect_N_empty : M ∩ N = ∅ := by sorry

end M_intersect_N_empty_l2543_254379


namespace quadratic_roots_property_l2543_254320

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 - 5 * p - 8 = 0) → 
  (3 * q^2 - 5 * q - 8 = 0) → 
  p ≠ q →
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := by
  sorry

end quadratic_roots_property_l2543_254320


namespace minimum_total_cost_l2543_254383

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

end minimum_total_cost_l2543_254383


namespace count_integers_in_range_l2543_254311

theorem count_integers_in_range : 
  (Finset.range 1001).card = (Finset.Ico 1000 2001).card := by sorry

end count_integers_in_range_l2543_254311


namespace y_derivative_l2543_254352

noncomputable def y (x : ℝ) : ℝ :=
  x - Real.log (1 + Real.exp x) - 2 * Real.exp (-x/2) * Real.arctan (Real.exp (x/2)) - (Real.arctan (Real.exp (x/2)))^2

theorem y_derivative (x : ℝ) :
  deriv y x = Real.arctan (Real.exp (x/2)) / (Real.exp (x/2) * (1 + Real.exp x)) :=
by sorry

end y_derivative_l2543_254352


namespace plums_picked_total_l2543_254366

/-- The number of plums Melanie picked -/
def melanie_plums : ℕ := 4

/-- The number of plums Dan picked -/
def dan_plums : ℕ := 9

/-- The number of plums Sally picked -/
def sally_plums : ℕ := 3

/-- The total number of plums picked -/
def total_plums : ℕ := melanie_plums + dan_plums + sally_plums

theorem plums_picked_total : total_plums = 16 := by
  sorry

end plums_picked_total_l2543_254366


namespace magnitude_of_c_l2543_254306

def vector_a : Fin 2 → ℝ := ![1, -1]
def vector_b : Fin 2 → ℝ := ![2, 1]

def vector_c : Fin 2 → ℝ := λ i => 2 * vector_a i + vector_b i

theorem magnitude_of_c :
  Real.sqrt ((vector_c 0) ^ 2 + (vector_c 1) ^ 2) = Real.sqrt 17 := by
  sorry

end magnitude_of_c_l2543_254306


namespace bug_ends_on_two_l2543_254335

/-- Represents the points on the circle -/
inductive Point
| one
| two
| three
| four
| five
| six

/-- Defines the movement rules for the bug -/
def next_point (p : Point) : Point :=
  match p with
  | Point.one => Point.two
  | Point.two => Point.four
  | Point.three => Point.four
  | Point.four => Point.one
  | Point.five => Point.six
  | Point.six => Point.two

/-- Simulates the bug's movement for a given number of jumps -/
def bug_position (start : Point) (jumps : Nat) : Point :=
  match jumps with
  | 0 => start
  | n + 1 => next_point (bug_position start n)

/-- The main theorem to prove -/
theorem bug_ends_on_two :
  bug_position Point.six 2000 = Point.two := by
  sorry

end bug_ends_on_two_l2543_254335


namespace power_of_two_greater_than_cube_l2543_254371

theorem power_of_two_greater_than_cube (n : ℕ) (h : n ≥ 10) : 2^n > n^3 := by
  sorry

end power_of_two_greater_than_cube_l2543_254371


namespace garage_to_other_rooms_ratio_l2543_254378

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

end garage_to_other_rooms_ratio_l2543_254378


namespace min_value_xy_plus_two_over_xy_l2543_254356

theorem min_value_xy_plus_two_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 1 → x * y + 2 / (x * y) ≤ z * w + 2 / (z * w)) ∧
  x * y + 2 / (x * y) = 33 / 4 := by
sorry

end min_value_xy_plus_two_over_xy_l2543_254356


namespace rectangular_field_fence_l2543_254385

theorem rectangular_field_fence (area : ℝ) (fence_length : ℝ) (uncovered_side : ℝ) :
  area = 680 →
  fence_length = 146 →
  uncovered_side * (fence_length - uncovered_side) / 2 = area →
  uncovered_side = 136 :=
by
  sorry

end rectangular_field_fence_l2543_254385


namespace probability_total_more_than_seven_is_five_twelfths_l2543_254362

/-- The number of possible outcomes when throwing a pair of dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (total > 7) when throwing a pair of dice -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing a pair of dice -/
def probability_total_more_than_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_total_more_than_seven_is_five_twelfths :
  probability_total_more_than_seven = 5 / 12 := by sorry

end probability_total_more_than_seven_is_five_twelfths_l2543_254362


namespace four_spheres_cover_point_source_l2543_254380

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

end four_spheres_cover_point_source_l2543_254380


namespace polygon_20_vertices_has_170_diagonals_l2543_254321

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 vertices has 170 diagonals -/
theorem polygon_20_vertices_has_170_diagonals :
  num_diagonals 20 = 170 := by
  sorry

end polygon_20_vertices_has_170_diagonals_l2543_254321


namespace least_sum_exponents_3125_l2543_254308

theorem least_sum_exponents_3125 : 
  let n := 3125
  let is_valid_representation (rep : List ℕ) := 
    (rep.map (λ i => 2^i)).sum = n ∧ rep.Nodup
  ∃ (rep : List ℕ), is_valid_representation rep ∧
    ∀ (other_rep : List ℕ), is_valid_representation other_rep → 
      rep.sum ≤ other_rep.sum :=
by sorry

end least_sum_exponents_3125_l2543_254308


namespace problem_solution_l2543_254398

theorem problem_solution : ∃ x : ℝ, 
  (0.6 * x = 0.3 * (125 ^ (1/3 : ℝ)) + 27) ∧ x = 47.5 := by
  sorry

end problem_solution_l2543_254398


namespace cloth_cost_price_calculation_l2543_254376

/-- Given the selling price and loss per metre for a certain length of cloth,
    calculate the cost price per metre. -/
def cost_price_per_metre (selling_price total_length loss_per_metre : ℚ) : ℚ :=
  (selling_price + loss_per_metre * total_length) / total_length

/-- Theorem stating that under the given conditions, 
    the cost price per metre of cloth is 95. -/
theorem cloth_cost_price_calculation :
  let selling_price : ℚ := 18000
  let total_length : ℚ := 200
  let loss_per_metre : ℚ := 5
  cost_price_per_metre selling_price total_length loss_per_metre = 95 :=
by
  sorry


end cloth_cost_price_calculation_l2543_254376


namespace hollow_circles_count_l2543_254333

/-- Represents the pattern of circles, where each number is the position of a hollow circle in the repeating sequence -/
def hollow_circle_positions : List Nat := [2, 5, 9]

/-- The length of the repeating sequence -/
def sequence_length : Nat := 9

/-- The total number of circles in the sequence -/
def total_circles : Nat := 2001

/-- Calculates the number of hollow circles in a sequence of given length -/
def count_hollow_circles (n : Nat) : Nat :=
  (n / sequence_length) * hollow_circle_positions.length + 
  (hollow_circle_positions.filter (· ≤ n % sequence_length)).length

theorem hollow_circles_count :
  count_hollow_circles total_circles = 667 := by
  sorry

end hollow_circles_count_l2543_254333


namespace paint_weight_l2543_254375

theorem paint_weight (total_weight : ℝ) (half_empty_weight : ℝ) 
  (h1 : total_weight = 24)
  (h2 : half_empty_weight = 14) :
  total_weight - half_empty_weight = 10 ∧ 
  2 * (total_weight - half_empty_weight) = 20 := by
  sorry

#check paint_weight

end paint_weight_l2543_254375


namespace max_training_cost_l2543_254331

def training_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 1400 * x
  else 2000 * x - 20 * x * x

theorem max_training_cost :
  ∃ (x : ℕ), x ≤ 60 ∧ ∀ (y : ℕ), y ≤ 60 → training_cost y ≤ training_cost x ∧ training_cost x = 50000 := by
  sorry

end max_training_cost_l2543_254331


namespace jane_nail_polish_drying_time_l2543_254324

/-- The total drying time for Jane's nail polish -/
def total_drying_time : ℕ :=
  let base_coat := 4
  let first_color := 5
  let second_color := 6
  let third_color := 7
  let first_nail_art := 8
  let second_nail_art := 10
  let top_coat := 9
  base_coat + first_color + second_color + third_color + first_nail_art + second_nail_art + top_coat

theorem jane_nail_polish_drying_time :
  total_drying_time = 49 := by sorry

end jane_nail_polish_drying_time_l2543_254324


namespace journey_distance_l2543_254349

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 10)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) :
  (total_time * speed1 * speed2) / (speed1 + speed2) = 224 := by
  sorry

#check journey_distance

end journey_distance_l2543_254349


namespace sin_cos_equivalence_l2543_254395

/-- The function f(x) = sin(2x) + √3 * cos(2x) is equivalent to 2 * sin(2(x + π/6)) for all real x -/
theorem sin_cos_equivalence (x : ℝ) : 
  Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 6)) := by
  sorry

end sin_cos_equivalence_l2543_254395


namespace dodecahedron_edge_probability_l2543_254377

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  /-- The number of vertices in a regular dodecahedron -/
  num_vertices : ℕ
  /-- The number of edges connected to each vertex -/
  edges_per_vertex : ℕ
  /-- Properties of a regular dodecahedron -/
  vertex_count : num_vertices = 20
  edge_count : edges_per_vertex = 3

/-- The probability of randomly choosing two vertices that form an edge in a regular dodecahedron -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem stating the probability of randomly choosing two vertices that form an edge in a regular dodecahedron -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end dodecahedron_edge_probability_l2543_254377


namespace remainder_sum_mod_eight_l2543_254342

theorem remainder_sum_mod_eight (a b c : ℕ) : 
  a < 8 → b < 8 → c < 8 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 8 = 1 →
  (7 * c) % 8 = 3 →
  (5 * b) % 8 = (4 + b) % 8 →
  (a + b + c) % 8 = 3 := by
sorry

end remainder_sum_mod_eight_l2543_254342


namespace runners_meeting_time_l2543_254309

def anna_lap_time : ℕ := 5
def bob_lap_time : ℕ := 8
def carol_lap_time : ℕ := 10

def meeting_time : ℕ := 40

theorem runners_meeting_time :
  Nat.lcm (Nat.lcm anna_lap_time bob_lap_time) carol_lap_time = meeting_time :=
sorry

end runners_meeting_time_l2543_254309


namespace fabian_walnuts_amount_l2543_254396

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

end fabian_walnuts_amount_l2543_254396


namespace fourth_person_height_l2543_254345

/-- Proves that the height of the fourth person is 82 inches given the conditions -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℕ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧  -- Heights in increasing order
  h₂ = h₁ + 2 ∧                  -- Difference between 1st and 2nd
  h₃ = h₂ + 2 ∧                  -- Difference between 2nd and 3rd
  h₄ = h₃ + 6 ∧                  -- Difference between 3rd and 4th
  (h₁ + h₂ + h₃ + h₄) / 4 = 76   -- Average height
  → h₄ = 82 := by
sorry

end fourth_person_height_l2543_254345


namespace library_book_redistribution_l2543_254312

theorem library_book_redistribution (total_boxes : Nat) (books_per_box : Nat) (new_box_capacity : Nat) :
  total_boxes = 1421 →
  books_per_box = 27 →
  new_box_capacity = 35 →
  (total_boxes * books_per_box) % new_box_capacity = 7 := by
  sorry

end library_book_redistribution_l2543_254312


namespace vegetable_sale_ratio_l2543_254368

theorem vegetable_sale_ratio : 
  let carrots : ℝ := 15
  let zucchini : ℝ := 13
  let broccoli : ℝ := 8
  let total_installed : ℝ := carrots + zucchini + broccoli
  let sold : ℝ := 18
  sold / total_installed = 1 / 2 := by
  sorry

end vegetable_sale_ratio_l2543_254368


namespace base7_to_base10_conversion_l2543_254344

/-- Converts a base-7 number represented as a list of digits to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number -/
def base7Number : List Nat := [5, 4, 6]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 327 := by sorry

end base7_to_base10_conversion_l2543_254344


namespace zeta_sum_seventh_power_l2543_254363

theorem zeta_sum_seventh_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 14) :
  ζ₁^7 + ζ₂^7 + ζ₃^7 = 233 := by
  sorry

end zeta_sum_seventh_power_l2543_254363


namespace skateboard_distance_l2543_254307

/-- The sum of an arithmetic sequence with first term 8, common difference 10, and 40 terms -/
theorem skateboard_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) : 
  a₁ = 8 → d = 10 → n = 40 → 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = 8120 := by
  sorry

end skateboard_distance_l2543_254307


namespace dawsons_b_students_l2543_254358

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

end dawsons_b_students_l2543_254358


namespace sum_even_divisors_180_l2543_254319

/-- Sum of positive even divisors of a natural number n -/
def sumEvenDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive even divisors of 180 is 468 -/
theorem sum_even_divisors_180 : sumEvenDivisors 180 = 468 := by sorry

end sum_even_divisors_180_l2543_254319


namespace mixture_composition_l2543_254337

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

end mixture_composition_l2543_254337


namespace smallest_valid_configuration_l2543_254359

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ                   -- Total number of lines
  intersect_5 : ℕ         -- Index of line intersecting 5 others
  intersect_9 : ℕ         -- Index of line intersecting 9 others
  intersect_11 : ℕ        -- Index of line intersecting 11 others
  intersect_5_count : ℕ   -- Number of intersections for intersect_5
  intersect_9_count : ℕ   -- Number of intersections for intersect_9
  intersect_11_count : ℕ  -- Number of intersections for intersect_11

/-- Predicate to check if a line configuration is valid -/
def is_valid_configuration (config : LineConfiguration) : Prop :=
  config.n > 0 ∧
  config.intersect_5 < config.n ∧
  config.intersect_9 < config.n ∧
  config.intersect_11 < config.n ∧
  config.intersect_5_count = 5 ∧
  config.intersect_9_count = 9 ∧
  config.intersect_11_count = 11

/-- Theorem stating that 12 is the smallest number of lines satisfying the conditions -/
theorem smallest_valid_configuration :
  (∃ (config : LineConfiguration), is_valid_configuration config ∧ config.n = 12) ∧
  (∀ (config : LineConfiguration), is_valid_configuration config → config.n ≥ 12) :=
sorry

end smallest_valid_configuration_l2543_254359


namespace elevator_capacity_l2543_254325

theorem elevator_capacity (adult_avg_weight child_avg_weight next_person_max_weight : ℝ)
  (num_adults num_children : ℕ) :
  adult_avg_weight = 140 →
  child_avg_weight = 64 →
  next_person_max_weight = 52 →
  num_adults = 3 →
  num_children = 2 →
  (num_adults : ℝ) * adult_avg_weight + (num_children : ℝ) * child_avg_weight + next_person_max_weight = 600 :=
by sorry

end elevator_capacity_l2543_254325


namespace transform_graph_point_l2543_254332

/-- Given a function g : ℝ → ℝ such that g(8) = 5, prove that (8/3, 14/9) is on the graph of
    3y = g(3x)/3 + 3 and the sum of its coordinates is 38/9 -/
theorem transform_graph_point (g : ℝ → ℝ) (h : g 8 = 5) :
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
  sorry

end transform_graph_point_l2543_254332


namespace express_regular_speed_ratio_l2543_254374

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

end express_regular_speed_ratio_l2543_254374


namespace factory_production_quota_l2543_254394

theorem factory_production_quota (x : ℕ) : 
  ((x - 3) * 31 + 60 = (x + 3) * 25 - 60) → x = 8 := by
  sorry

end factory_production_quota_l2543_254394


namespace inscribed_cube_volume_l2543_254397

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

#check inscribed_cube_volume

end inscribed_cube_volume_l2543_254397


namespace average_sale_is_6900_l2543_254347

def sales : List ℕ := [6435, 6927, 6855, 7230, 6562, 7391]

theorem average_sale_is_6900 :
  (sales.sum : ℚ) / sales.length = 6900 := by
  sorry

end average_sale_is_6900_l2543_254347


namespace rectangle_perimeter_problem_l2543_254304

theorem rectangle_perimeter_problem :
  ∀ (a b : ℕ),
    a ≠ b →
    a * b = 2 * (2 * a + 2 * b) →
    2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_problem_l2543_254304


namespace area_of_tangent_square_l2543_254382

/-- Given a 6 by 6 square with four semicircles on its sides, and another square EFGH
    with sides parallel and tangent to the semicircles, the area of EFGH is 144. -/
theorem area_of_tangent_square (original_side_length : ℝ) (EFGH_side_length : ℝ) : 
  original_side_length = 6 →
  EFGH_side_length = original_side_length + 2 * (original_side_length / 2) →
  EFGH_side_length ^ 2 = 144 := by
sorry

end area_of_tangent_square_l2543_254382


namespace building_floor_height_l2543_254329

/-- Proves that the height of each of the first 18 floors is 3 meters -/
theorem building_floor_height
  (total_floors : ℕ)
  (last_two_extra_height : ℝ)
  (total_height : ℝ)
  (h : ℝ)
  (h_total_floors : total_floors = 20)
  (h_last_two_extra : last_two_extra_height = 0.5)
  (h_total_height : total_height = 61)
  (h_height_equation : 18 * h + 2 * (h + last_two_extra_height) = total_height) :
  h = 3 := by
sorry

end building_floor_height_l2543_254329


namespace charity_game_probability_l2543_254360

theorem charity_game_probability : 
  let p1 : ℝ := 0.9  -- Probability of correct answer for first picture
  let p2 : ℝ := 0.5  -- Probability of correct answer for second picture
  let p3 : ℝ := 0.4  -- Probability of correct answer for third picture
  let f1 : ℕ := 1000 -- Fund raised for first correct answer
  let f2 : ℕ := 2000 -- Fund raised for second correct answer
  let f3 : ℕ := 3000 -- Fund raised for third correct answer
  -- Probability of raising exactly 3000 yuan
  p1 * p2 * (1 - p3) = 0.27
  := by sorry

end charity_game_probability_l2543_254360


namespace f_plus_two_is_odd_l2543_254354

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

end f_plus_two_is_odd_l2543_254354


namespace smallest_d_inequality_l2543_254390

theorem smallest_d_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x * y) + (x^2 - y^2)^2 ≥ x + y ∧
  ∀ d : ℝ, d > 0 → d < 1 →
    ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ Real.sqrt (x * y) + d * (x^2 - y^2)^2 < x + y :=
by sorry

end smallest_d_inequality_l2543_254390


namespace different_grade_selections_l2543_254387

/-- The number of students in the first year -/
def first_year_students : ℕ := 4

/-- The number of students in the second year -/
def second_year_students : ℕ := 5

/-- The number of students in the third year -/
def third_year_students : ℕ := 4

/-- The total number of ways to select 2 students from different grades -/
def total_selections : ℕ := 56

theorem different_grade_selections :
  first_year_students * second_year_students +
  first_year_students * third_year_students +
  second_year_students * third_year_students = total_selections :=
by sorry

end different_grade_selections_l2543_254387


namespace twelfth_term_of_ap_l2543_254381

-- Define the arithmetic progression
def arithmeticProgression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- State the theorem
theorem twelfth_term_of_ap : arithmeticProgression 2 8 12 = 90 := by
  sorry

end twelfth_term_of_ap_l2543_254381


namespace paula_shirts_bought_l2543_254327

def shirts_bought (initial_amount : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  (initial_amount - pants_cost - remaining_amount) / shirt_cost

theorem paula_shirts_bought :
  shirts_bought 109 11 13 74 = 2 := by
  sorry

end paula_shirts_bought_l2543_254327


namespace max_sundays_in_56_days_l2543_254391

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

end max_sundays_in_56_days_l2543_254391


namespace beetles_consumed_1080_l2543_254313

/-- Represents the daily consumption and population changes in a tropical forest ecosystem --/
structure ForestEcosystem where
  beetles_per_bird : ℕ
  birds_per_snake : ℕ
  snakes_per_jaguar : ℕ
  jaguars_per_crocodile : ℕ
  bird_increase : ℕ
  snake_increase : ℕ
  jaguar_increase : ℕ
  initial_jaguars : ℕ
  initial_crocodiles : ℕ

/-- Calculates the number of beetles consumed in one day in the forest ecosystem --/
def beetles_consumed (eco : ForestEcosystem) : ℕ :=
  eco.initial_jaguars * eco.snakes_per_jaguar * eco.birds_per_snake * eco.beetles_per_bird

/-- Theorem stating that the number of beetles consumed in one day is 1080 --/
theorem beetles_consumed_1080 (eco : ForestEcosystem) 
  (h1 : eco.beetles_per_bird = 12)
  (h2 : eco.birds_per_snake = 3)
  (h3 : eco.snakes_per_jaguar = 5)
  (h4 : eco.jaguars_per_crocodile = 2)
  (h5 : eco.bird_increase = 4)
  (h6 : eco.snake_increase = 2)
  (h7 : eco.jaguar_increase = 1)
  (h8 : eco.initial_jaguars = 6)
  (h9 : eco.initial_crocodiles = 30) :
  beetles_consumed eco = 1080 := by
  sorry


end beetles_consumed_1080_l2543_254313


namespace bird_round_trips_l2543_254326

/-- Given two birds collecting nest materials, this theorem proves the number of round trips each bird made. -/
theorem bird_round_trips (distance_to_materials : ℕ) (total_distance : ℕ) : 
  distance_to_materials = 200 →
  total_distance = 8000 →
  ∃ (trips_per_bird : ℕ), 
    trips_per_bird * 2 * (2 * distance_to_materials) = total_distance ∧
    trips_per_bird = 10 := by
  sorry

end bird_round_trips_l2543_254326


namespace function_symmetry_property_l2543_254392

open Real

/-- Given a function f(x) = a cos(x) + bx² + 2, prove that
    f(2016) - f(-2016) + f''(2017) + f''(-2017) = 0 for any real a and b -/
theorem function_symmetry_property (a b : ℝ) :
  let f := fun x => a * cos x + b * x^2 + 2
  let f'' := fun x => -a * cos x + 2 * b
  f 2016 - f (-2016) + f'' 2017 + f'' (-2017) = 0 := by
  sorry

end function_symmetry_property_l2543_254392


namespace equal_fish_count_l2543_254302

def herring_fat : ℕ := 40
def eel_fat : ℕ := 20
def pike_fat : ℕ := eel_fat + 10
def total_fat : ℕ := 3600

theorem equal_fish_count (x : ℕ) 
  (h : x * herring_fat + x * eel_fat + x * pike_fat = total_fat) : 
  x = 40 := by
  sorry

end equal_fish_count_l2543_254302


namespace unique_number_l2543_254330

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def person_a_initially_unsure (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ 
    is_three_digit_number m ∧ 
    is_perfect_square m ∧ 
    has_distinct_digits m ∧ 
    m / 100 = n / 100

def person_b_knows_a_unsure (n : ℕ) : Prop :=
  ∀ m : ℕ, (m / 10) % 10 = (n / 10) % 10 → person_a_initially_unsure m

def person_c_knows_number (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      m % 10 = n % 10)

def person_a_knows_after_c (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      person_c_knows_number m ∧ 
      m / 100 = n / 100)

def person_b_knows_after_a (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      person_c_knows_number m ∧ 
      person_a_knows_after_c m ∧ 
      (m / 10) % 10 = (n / 10) % 10)

theorem unique_number : 
  ∃! n : ℕ, 
    is_three_digit_number n ∧ 
    is_perfect_square n ∧ 
    has_distinct_digits n ∧ 
    person_a_initially_unsure n ∧ 
    person_b_knows_a_unsure n ∧ 
    person_c_knows_number n ∧ 
    person_a_knows_after_c n ∧ 
    person_b_knows_after_a n ∧ 
    n = 289 := by sorry

end unique_number_l2543_254330


namespace reachable_points_characterization_l2543_254361

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

end reachable_points_characterization_l2543_254361


namespace smallest_integer_value_is_two_l2543_254343

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

end smallest_integer_value_is_two_l2543_254343


namespace quadratic_equation_roots_l2543_254386

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 4*x - 2*m + 5 = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (x₁ ≠ x₂) →
  (m ≥ 1/2) ∧ 
  (x₁ * x₂ + x₁ + x₂ = m^2 + 6 → m = 1) := by
sorry

end quadratic_equation_roots_l2543_254386


namespace parabola_chord_constant_sum_l2543_254316

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = 2x^2 -/
def parabola (p : Point) : Prop :=
  p.y = 2 * p.x^2

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: For the parabola y = 2x^2, if there exists a constant c such that
    for any chord AB passing through the point (0,c), the value
    t = 1/AC^2 + 1/BC^2 is constant, then c = 1/4 and t = 8 -/
theorem parabola_chord_constant_sum (c : ℝ) :
  (∃ t : ℝ, ∀ A B : Point,
    parabola A ∧ parabola B ∧
    (∃ m : ℝ, A.y = m * A.x + c ∧ B.y = m * B.x + c) →
    1 / distanceSquared A ⟨0, c⟩ + 1 / distanceSquared B ⟨0, c⟩ = t) →
  c = 1/4 ∧ t = 8 :=
sorry

end parabola_chord_constant_sum_l2543_254316


namespace combined_weight_l2543_254303

/-- The combined weight of Tracy, John, and Jake is 150 kg -/
theorem combined_weight (tracy_weight : ℕ) (jake_weight : ℕ) (john_weight : ℕ)
  (h1 : tracy_weight = 52)
  (h2 : jake_weight = tracy_weight + 8)
  (h3 : jake_weight - john_weight = 14 ∨ tracy_weight - john_weight = 14) :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

#check combined_weight

end combined_weight_l2543_254303


namespace choose_four_different_suits_standard_deck_l2543_254348

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The number of ways to choose 4 cards of different suits from a standard deck -/
def choose_four_different_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.suits

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem choose_four_different_suits_standard_deck :
  ∃ (d : Deck), d.cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ choose_four_different_suits d = 28561 := by
  sorry

end choose_four_different_suits_standard_deck_l2543_254348


namespace system_not_unique_solution_l2543_254328

/-- The system of equations does not have a unique solution when k = 9 -/
theorem system_not_unique_solution (x y z : ℝ) (k m : ℝ) :
  (3 * (3 * x^2 + 4 * y^2) = 36) →
  (k * x^2 + 12 * y^2 = 30) →
  (m * x^3 - 2 * y^3 + z^2 = 24) →
  (k = 9 → ∃ (c : ℝ), c ≠ 0 ∧ (3 * x^2 + 4 * y^2 = c * (k * x^2 + 12 * y^2))) :=
by sorry

end system_not_unique_solution_l2543_254328


namespace arithmetic_sequence_sum_base6_l2543_254323

/-- Represents a number in base 6 --/
structure Base6 :=
  (value : ℕ)
  (isValid : value < 6)

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List Base6 :=
  sorry

/-- Arithmetic sequence in base 6 --/
def arithmeticSequenceBase6 (a l d : Base6) : List Base6 :=
  sorry

/-- Sum of a list of Base6 numbers --/
def sumBase6 (lst : List Base6) : List Base6 :=
  sorry

theorem arithmetic_sequence_sum_base6 :
  let a := Base6.mk 1 (by norm_num)
  let l := Base6.mk 5 (by norm_num) -- 41 in base 6 is 5 * 6 + 5 = 35
  let d := Base6.mk 2 (by norm_num)
  let sequence := arithmeticSequenceBase6 a l d
  sumBase6 sequence = toBase6 441 :=
by
  sorry

end arithmetic_sequence_sum_base6_l2543_254323


namespace tangent_parallel_implies_a_value_l2543_254351

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^3 - a

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * a * x^2

theorem tangent_parallel_implies_a_value (a : ℝ) :
  (f a 1 = a) →                           -- The point (1, a) is on the curve
  (f_derivative a 1 = 2) →                -- The slope of the tangent at (1, a) equals the slope of 2x - y + 1 = 0
  a = 1/3 := by
sorry

end tangent_parallel_implies_a_value_l2543_254351


namespace log_N_between_consecutive_integers_l2543_254339

theorem log_N_between_consecutive_integers 
  (N : ℝ) 
  (h : Real.log 2500 < Real.log N ∧ Real.log N < Real.log 10000) : 
  ∃ (m : ℤ), m + (m + 1) = 7 ∧ 
    (↑m : ℝ) < Real.log N ∧ Real.log N < (↑m + 1 : ℝ) := by
sorry

end log_N_between_consecutive_integers_l2543_254339


namespace equivalent_expression_l2543_254388

theorem equivalent_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = (2 * x^3 + 2 * y^3) / (x * y) := by
  sorry

end equivalent_expression_l2543_254388


namespace james_passenger_count_l2543_254367

/-- Calculates the total number of passengers James has seen given the vehicle counts and passenger capacities. -/
def total_passengers (total_vehicles : ℕ) (trucks : ℕ) (buses : ℕ) (cars : ℕ) 
  (truck_capacity : ℕ) (bus_capacity : ℕ) (taxi_capacity : ℕ) (motorbike_capacity : ℕ) (car_capacity : ℕ) : ℕ :=
  let taxis := 2 * buses
  let motorbikes := total_vehicles - trucks - buses - taxis - cars
  trucks * truck_capacity + buses * bus_capacity + taxis * taxi_capacity + motorbikes * motorbike_capacity + cars * car_capacity

theorem james_passenger_count :
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end james_passenger_count_l2543_254367


namespace rotating_ngon_path_area_theorem_l2543_254370

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

end rotating_ngon_path_area_theorem_l2543_254370


namespace lawyer_percentage_l2543_254399

theorem lawyer_percentage (total : ℝ) (h1 : total > 0) : 
  let women_ratio : ℝ := 0.9
  let women_lawyer_prob : ℝ := 0.54
  let women_count : ℝ := women_ratio * total
  let lawyer_ratio : ℝ := women_lawyer_prob / women_ratio
  lawyer_ratio = 0.6 := by sorry

end lawyer_percentage_l2543_254399


namespace square_sequence_problem_l2543_254372

/-- The number of squares in the nth figure of the sequence -/
def g (n : ℕ) : ℕ :=
  2 * n^2 + 4 * n + 3

theorem square_sequence_problem :
  g 0 = 3 ∧ g 1 = 9 ∧ g 2 = 19 ∧ g 3 = 33 → g 100 = 20403 :=
by
  sorry

end square_sequence_problem_l2543_254372


namespace expand_product_l2543_254346

theorem expand_product (x : ℝ) : (x + 4) * (x^2 - 9) = x^3 + 4*x^2 - 9*x - 36 := by
  sorry

end expand_product_l2543_254346


namespace opposite_def_opposite_of_neg_two_l2543_254317

/-- The opposite of a real number -/
def opposite (a : ℝ) : ℝ := -a

/-- The property that defines the opposite of a number -/
theorem opposite_def (a : ℝ) : a + opposite a = 0 := by sorry

/-- Proof that the opposite of -2 is 2 -/
theorem opposite_of_neg_two : opposite (-2) = 2 := by sorry

end opposite_def_opposite_of_neg_two_l2543_254317


namespace cost_of_one_each_l2543_254310

/-- The cost of goods A, B, and C -/
structure GoodsCost where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions from the problem -/
def problem_conditions (cost : GoodsCost) : Prop :=
  3 * cost.A + 7 * cost.B + cost.C = 3.15 ∧
  4 * cost.A + 10 * cost.B + cost.C = 4.20

/-- The theorem to prove -/
theorem cost_of_one_each (cost : GoodsCost) :
  problem_conditions cost → cost.A + cost.B + cost.C = 1.05 := by
  sorry

end cost_of_one_each_l2543_254310


namespace claudia_water_amount_l2543_254314

/-- The amount of water in ounces that Claudia had initially -/
def initial_water : ℕ := 122

/-- The capacity of a 5-ounce glass in ounces -/
def five_ounce_glass : ℕ := 5

/-- The capacity of an 8-ounce glass in ounces -/
def eight_ounce_glass : ℕ := 8

/-- The capacity of a 4-ounce glass in ounces -/
def four_ounce_glass : ℕ := 4

/-- The number of 5-ounce glasses filled -/
def num_five_ounce : ℕ := 6

/-- The number of 8-ounce glasses filled -/
def num_eight_ounce : ℕ := 4

/-- The number of 4-ounce glasses that can be filled with the remaining water -/
def num_four_ounce : ℕ := 15

theorem claudia_water_amount :
  initial_water = 
    num_five_ounce * five_ounce_glass + 
    num_eight_ounce * eight_ounce_glass + 
    num_four_ounce * four_ounce_glass := by
  sorry

end claudia_water_amount_l2543_254314


namespace white_to_red_black_ratio_l2543_254340

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


end white_to_red_black_ratio_l2543_254340


namespace remainder_of_n_l2543_254300

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 1) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end remainder_of_n_l2543_254300


namespace solution_value_l2543_254393

theorem solution_value (a b : ℝ) : 
  (2 * 2 * a - 2 * b - 20 = 0) → (2023 - 2 * a + b = 2013) := by
  sorry

end solution_value_l2543_254393


namespace total_football_games_l2543_254373

theorem total_football_games (games_this_year games_last_year : ℕ) 
  (h1 : games_this_year = 14)
  (h2 : games_last_year = 29) :
  games_this_year + games_last_year = 43 := by
  sorry

end total_football_games_l2543_254373


namespace grade_ratio_l2543_254336

/-- Proves that the ratio of Bob's grade to Jason's grade is 1:2 -/
theorem grade_ratio (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = 35 →
  (bob_grade : ℚ) / jason_grade = 1 / 2 := by
  sorry

end grade_ratio_l2543_254336


namespace kids_ticket_price_l2543_254318

/-- Proves that the price of a kid's ticket is $12 given the specified conditions --/
theorem kids_ticket_price (total_people : ℕ) (adult_price : ℕ) (total_sales : ℕ) (num_kids : ℕ) :
  total_people = 254 →
  adult_price = 28 →
  total_sales = 3864 →
  num_kids = 203 →
  ∃ (kids_price : ℕ), kids_price = 12 ∧ 
    total_sales = (total_people - num_kids) * adult_price + num_kids * kids_price :=
by sorry


end kids_ticket_price_l2543_254318


namespace first_number_value_l2543_254301

theorem first_number_value (y x : ℚ) : 
  (y + 76 + x) / 3 = 5 → x = -63 → y = 2 := by
  sorry

end first_number_value_l2543_254301


namespace expression_values_l2543_254334

theorem expression_values (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (gcd_ab : Nat.gcd a b = 1) (gcd_bc : Nat.gcd b c = 1) (gcd_ca : Nat.gcd c a = 1) :
  (a + b) / c + (b + c) / a + (c + a) / b = 7 ∨ (a + b) / c + (b + c) / a + (c + a) / b = 8 :=
by sorry

end expression_values_l2543_254334
