import Mathlib

namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l3264_326406

def vector_a : Fin 2 → ℝ := ![2, -1]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![6, x]

theorem vector_difference_magnitude 
  (h_parallel : ∃ (k : ℝ), ∀ i, vector_a i = k * vector_b x i) :
  ∃ (x : ℝ), ‖vector_a - vector_b x‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l3264_326406


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3264_326405

theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
  deepak_age = 3 →
  rahul_age + 22 = 26 →
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3264_326405


namespace NUMINAMATH_CALUDE_intersection_slope_l3264_326467

/-- Given two circles in the xy-plane, this theorem states that the slope of the line
    passing through their intersection points is 1/7. -/
theorem intersection_slope (x y : ℝ) :
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) →
  (x^2 + y^2 - 8*x + 18*y + 40 = 0) →
  (∃ (m : ℝ), m = 1/7 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) →
    (x₁^2 + y₁^2 - 8*x₁ + 18*y₁ + 40 = 0) →
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) →
    (x₂^2 + y₂^2 - 8*x₂ + 18*y₂ + 40 = 0) →
    x₁ ≠ x₂ →
    m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry


end NUMINAMATH_CALUDE_intersection_slope_l3264_326467


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3264_326422

theorem arithmetic_evaluation : 
  -(18 / 3 * 11 - 48 / 4 + 5 * 9) = -99 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3264_326422


namespace NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l3264_326494

/-- Represents the cost of purchasing goldfish -/
def goldfish_cost (n : ℕ) : ℚ :=
  if n ≥ 3 then 20 * n else 0

/-- The set of points representing goldfish purchases from 3 to 15 -/
def goldfish_graph : Set (ℕ × ℚ) :=
  {p | ∃ n : ℕ, 3 ≤ n ∧ n ≤ 15 ∧ p = (n, goldfish_cost n)}

theorem goldfish_graph_is_finite_distinct_points :
  Finite goldfish_graph ∧ ∀ p q : (ℕ × ℚ), p ∈ goldfish_graph → q ∈ goldfish_graph → p ≠ q → p.1 ≠ q.1 :=
sorry

end NUMINAMATH_CALUDE_goldfish_graph_is_finite_distinct_points_l3264_326494


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_unique_a_for_nonnegative_f_l3264_326472

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (1 + x)

theorem tangent_line_at_x_1 (h : ℝ) :
  ∃ (m b : ℝ), ∀ x, (f 2 x - (f 2 1)) = m * (x - 1) + b ∧ 
  m * x + b = Real.log 2 - 1 := by sorry

theorem unique_a_for_nonnegative_f :
  ∃! a : ℝ, ∀ x : ℝ, x > -1 → f a x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_unique_a_for_nonnegative_f_l3264_326472


namespace NUMINAMATH_CALUDE_thirty_times_multiple_of_every_integer_l3264_326415

theorem thirty_times_multiple_of_every_integer (n : ℤ) :
  (∀ m : ℤ, ∃ k : ℤ, n = 30 * k * m) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_thirty_times_multiple_of_every_integer_l3264_326415


namespace NUMINAMATH_CALUDE_limit_cosine_fraction_l3264_326498

theorem limit_cosine_fraction :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |((1 - Real.cos (2*x)) / (Real.cos (7*x) - Real.cos (3*x))) + (1/10)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_cosine_fraction_l3264_326498


namespace NUMINAMATH_CALUDE_sally_seashell_earnings_l3264_326418

-- Define the number of seashells picked on Monday
def monday_seashells : ℕ := 30

-- Define the number of seashells picked on Tuesday
def tuesday_seashells : ℕ := monday_seashells / 2

-- Define the price of each seashell in cents
def seashell_price : ℕ := 120

-- Theorem statement
theorem sally_seashell_earnings :
  (monday_seashells + tuesday_seashells) * seashell_price = 5400 := by
  sorry

end NUMINAMATH_CALUDE_sally_seashell_earnings_l3264_326418


namespace NUMINAMATH_CALUDE_specificGrid_toothpicks_l3264_326435

/-- Represents a rectangular grid with diagonal supports -/
structure ToothpickGrid where
  length : ℕ
  width : ℕ
  diagonalInterval : ℕ

/-- Calculates the total number of toothpicks used in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let verticalToothpicks := (grid.length + 1) * grid.width
  let horizontalToothpicks := (grid.width + 1) * grid.length
  let diagonalLines := (grid.length + 1) / grid.diagonalInterval + (grid.width + 1) / grid.diagonalInterval
  let diagonalToothpicks := diagonalLines * 7  -- Approximation of √50
  verticalToothpicks + horizontalToothpicks + diagonalToothpicks

/-- The specific grid described in the problem -/
def specificGrid : ToothpickGrid :=
  { length := 45
    width := 25
    diagonalInterval := 5 }

theorem specificGrid_toothpicks :
  totalToothpicks specificGrid = 2446 := by
  sorry

end NUMINAMATH_CALUDE_specificGrid_toothpicks_l3264_326435


namespace NUMINAMATH_CALUDE_real_estate_investment_l3264_326483

def total_investment : ℝ := 200000
def real_estate_ratio : ℝ := 7

theorem real_estate_investment (mutual_funds : ℝ) 
  (h1 : mutual_funds + real_estate_ratio * mutual_funds = total_investment) :
  real_estate_ratio * mutual_funds = 175000 := by
  sorry

end NUMINAMATH_CALUDE_real_estate_investment_l3264_326483


namespace NUMINAMATH_CALUDE_problem_solution_l3264_326475

/-- Given a function f(x) = x² - 2x + 2a, where the solution set of f(x) ≤ 0 is {x | -2 ≤ x ≤ m},
    prove that a = -4 and m = 4, and find the range of c where (c+a)x² + 2(c+a)x - 1 < 0 always holds for x. -/
theorem problem_solution (a m : ℝ) (f : ℝ → ℝ) (c : ℝ) : 
  (f = fun x => x^2 - 2*x + 2*a) →
  (∀ x, f x ≤ 0 ↔ -2 ≤ x ∧ x ≤ m) →
  (a = -4 ∧ m = 4) ∧
  (∀ x, (c + a)*x^2 + 2*(c + a)*x - 1 < 0 ↔ 13/4 < c ∧ c < 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3264_326475


namespace NUMINAMATH_CALUDE_train_carriages_l3264_326409

theorem train_carriages (initial_seats : ℕ) (additional_capacity : ℕ) (total_passengers : ℕ) (num_trains : ℕ) :
  initial_seats = 25 →
  additional_capacity = 10 →
  total_passengers = 420 →
  num_trains = 3 →
  (total_passengers / (num_trains * (initial_seats + additional_capacity))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_train_carriages_l3264_326409


namespace NUMINAMATH_CALUDE_probability_drawing_white_ball_l3264_326479

theorem probability_drawing_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h1 : total_balls = 15)
  (h2 : red_balls = 9)
  (h3 : white_balls = 6)
  (h4 : total_balls = red_balls + white_balls) :
  (white_balls : ℚ) / (total_balls - 1 : ℚ) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_drawing_white_ball_l3264_326479


namespace NUMINAMATH_CALUDE_students_surveyed_l3264_326419

theorem students_surveyed : ℕ :=
  let total_students : ℕ := sorry
  let french_speakers : ℕ := sorry
  let french_english_speakers : ℕ := 10
  let french_only_speakers : ℕ := 40

  have h1 : french_speakers = french_english_speakers + french_only_speakers := by sorry
  have h2 : french_speakers = 50 := by sorry
  have h3 : french_speakers = total_students / 4 := by sorry

  200

/- Proof omitted -/

end NUMINAMATH_CALUDE_students_surveyed_l3264_326419


namespace NUMINAMATH_CALUDE_cosine_sum_equals_radius_ratio_l3264_326420

-- Define a triangle with its angles, circumradius, and inradius
structure Triangle where
  α : Real
  β : Real
  γ : Real
  R : Real
  r : Real
  angle_sum : α + β + γ = Real.pi
  positive_R : R > 0
  positive_r : r > 0

-- State the theorem
theorem cosine_sum_equals_radius_ratio (t : Triangle) :
  Real.cos t.α + Real.cos t.β + Real.cos t.γ = (t.R + t.r) / t.R :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_equals_radius_ratio_l3264_326420


namespace NUMINAMATH_CALUDE_correct_guess_probability_l3264_326445

-- Define a finite set with 4 elements
def GameOptions : Type := Fin 4

-- Define the power set of GameOptions
def PowerSet (α : Type) : Type := Set α

-- Define the number of elements in the power set of GameOptions
def NumPossibleAnswers : Nat := 2^4 - 1  -- Exclude the empty set

-- Define the probability of guessing correctly
def ProbCorrectGuess : ℚ := 1 / NumPossibleAnswers

-- Theorem statement
theorem correct_guess_probability :
  ProbCorrectGuess = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l3264_326445


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l3264_326444

/-- The number of ways to partition n indistinguishable objects into k or fewer non-empty, indistinguishable groups -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to partition 6 indistinguishable objects into 3 or fewer non-empty, indistinguishable groups -/
theorem six_balls_three_boxes : partition_count 6 3 = 6 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l3264_326444


namespace NUMINAMATH_CALUDE_bruce_son_age_l3264_326459

/-- Bruce's current age -/
def bruce_age : ℕ := 36

/-- Number of years in the future -/
def years_future : ℕ := 6

/-- Bruce's son's current age -/
def son_age : ℕ := 8

theorem bruce_son_age :
  (bruce_age + years_future) = 3 * (son_age + years_future) :=
sorry

end NUMINAMATH_CALUDE_bruce_son_age_l3264_326459


namespace NUMINAMATH_CALUDE_distance_solution_l3264_326487

/-- The distance from a dormitory to a city -/
def distance_problem (D : ℝ) : Prop :=
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 12 = D

theorem distance_solution : ∃ D : ℝ, distance_problem D ∧ D = 90 := by
  sorry

end NUMINAMATH_CALUDE_distance_solution_l3264_326487


namespace NUMINAMATH_CALUDE_tan_half_angle_l3264_326404

theorem tan_half_angle (α : Real) 
  (h1 : π < α ∧ α < 3*π/2) 
  (h2 : Real.sin (3*π/2 + α) = 4/5) : 
  Real.tan (α/2) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_half_angle_l3264_326404


namespace NUMINAMATH_CALUDE_pond_width_pond_width_is_10_l3264_326417

/-- The width of a rectangular pond, given its length, depth, and volume of soil extracted. -/
theorem pond_width (length depth volume : ℝ) (h1 : length = 20) (h2 : depth = 5) (h3 : volume = 1000) :
  volume = length * depth * (volume / (length * depth)) :=
by sorry

/-- The width of the pond is 10 meters. -/
theorem pond_width_is_10 (length depth volume : ℝ) (h1 : length = 20) (h2 : depth = 5) (h3 : volume = 1000) :
  volume / (length * depth) = 10 :=
by sorry

end NUMINAMATH_CALUDE_pond_width_pond_width_is_10_l3264_326417


namespace NUMINAMATH_CALUDE_min_value_cos_sin_l3264_326464

theorem min_value_cos_sin (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 3 * Real.pi / 2) :
  ∃ m : Real, m = -1/2 ∧ ∀ θ' : Real, 0 ≤ θ' ∧ θ' ≤ 3 * Real.pi / 2 →
    m ≤ Real.cos (θ' / 3) * (1 - Real.sin θ') :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_sin_l3264_326464


namespace NUMINAMATH_CALUDE_range_of_m_l3264_326477

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x m : ℝ) : Prop := x < m

-- Define the relationship between ¬p and ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, ¬(q x m) → ¬(p x) ∧ ∃ y, ¬(p y) ∧ q y m

-- Theorem statement
theorem range_of_m (m : ℝ) :
  not_p_necessary_not_sufficient_for_not_q m →
  m ∈ Set.Iic (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3264_326477


namespace NUMINAMATH_CALUDE_water_to_height_ratio_l3264_326454

def rons_height : ℝ := 12
def water_depth : ℝ := 60

theorem water_to_height_ratio : water_depth / rons_height = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_to_height_ratio_l3264_326454


namespace NUMINAMATH_CALUDE_tan_sum_product_l3264_326488

theorem tan_sum_product (α β : Real) (h : α + β = 3 * Real.pi / 4) :
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_l3264_326488


namespace NUMINAMATH_CALUDE_percentage_calculation_l3264_326425

theorem percentage_calculation (x y : ℝ) : 
  x = 0.8 * 350 → y = 0.6 * x → 1.2 * y = 201.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3264_326425


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3264_326449

theorem simplify_polynomial (r : ℝ) : (2*r^2 + 5*r - 7) - (r^2 + 4*r - 6) = r^2 + r - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3264_326449


namespace NUMINAMATH_CALUDE_rectangle_area_l3264_326436

theorem rectangle_area (b : ℝ) : 
  let square_area : ℝ := 2500
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_breadth : ℝ := b
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area = 20 * b := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3264_326436


namespace NUMINAMATH_CALUDE_arg_z_range_l3264_326452

theorem arg_z_range (z : ℂ) (h : |Complex.arg ((z + 1) / (z + 2))| = π / 6) :
  Complex.arg z ∈ Set.union
    (Set.Ioo (5 * π / 6 - Real.arcsin (Real.sqrt 3 / 3)) π)
    (Set.Ioo π (7 * π / 6 + Real.arcsin (Real.sqrt 3 / 3))) := by
  sorry

end NUMINAMATH_CALUDE_arg_z_range_l3264_326452


namespace NUMINAMATH_CALUDE_negation_equivalence_l3264_326465

/-- A triangle is a geometric shape with three sides and three angles. -/
structure Triangle where
  -- We don't need to define the specifics of a triangle for this problem

/-- An angle is obtuse if it is greater than 90 degrees. -/
def is_obtuse_angle (angle : ℝ) : Prop := angle > 90

/-- The original statement: Every triangle has at least two obtuse angles. -/
def original_statement : Prop :=
  ∀ t : Triangle, ∃ a b : ℝ, is_obtuse_angle a ∧ is_obtuse_angle b ∧ a ≠ b

/-- The negation: There exists a triangle that has at most one obtuse angle. -/
def negation : Prop :=
  ∃ t : Triangle, ∀ a b : ℝ, is_obtuse_angle a ∧ is_obtuse_angle b → a = b

/-- The negation of the original statement is equivalent to the given negation. -/
theorem negation_equivalence : ¬original_statement ↔ negation := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3264_326465


namespace NUMINAMATH_CALUDE_min_area_of_rectangle_with_perimeter_120_l3264_326499

-- Define the rectangle type
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the perimeter function
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Define the area function
def area (r : Rectangle) : ℕ := r.length * r.width

-- Theorem statement
theorem min_area_of_rectangle_with_perimeter_120 :
  ∃ (min_area : ℕ), 
    (∀ (r : Rectangle), perimeter r = 120 → area r ≥ min_area) ∧
    (∃ (r : Rectangle), perimeter r = 120 ∧ area r = min_area) ∧
    min_area = 59 := by
  sorry

end NUMINAMATH_CALUDE_min_area_of_rectangle_with_perimeter_120_l3264_326499


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3264_326438

/-- Given an angle of 60 degrees rotated 600 degrees clockwise, 
    the resulting acute angle measure is 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 → 
  rotation = 600 → 
  (initial_angle - (rotation % 360)) % 360 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3264_326438


namespace NUMINAMATH_CALUDE_compute_expression_l3264_326434

theorem compute_expression : 8 * (1 / 4)^4 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3264_326434


namespace NUMINAMATH_CALUDE_shoe_shirt_earnings_l3264_326421

theorem shoe_shirt_earnings : 
  let shoe_pairs : ℕ := 6
  let shoe_price : ℕ := 3
  let shirt_count : ℕ := 18
  let shirt_price : ℕ := 2
  let total_earnings := shoe_pairs * shoe_price + shirt_count * shirt_price
  let people_count : ℕ := 2
  (total_earnings / people_count : ℕ) = 27 := by sorry

end NUMINAMATH_CALUDE_shoe_shirt_earnings_l3264_326421


namespace NUMINAMATH_CALUDE_museum_artifacts_l3264_326469

theorem museum_artifacts (total_wings : ℕ) (painting_wings : ℕ) (large_painting : ℕ) 
  (small_paintings_per_wing : ℕ) (artifact_multiplier : ℕ) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  artifact_multiplier = 4 →
  let total_paintings := large_painting + 2 * small_paintings_per_wing
  let total_artifacts := artifact_multiplier * total_paintings
  let artifact_wings := total_wings - painting_wings
  total_artifacts / artifact_wings = 20 :=
by sorry

end NUMINAMATH_CALUDE_museum_artifacts_l3264_326469


namespace NUMINAMATH_CALUDE_return_speed_calculation_l3264_326491

/-- Proves that given a round trip with specified conditions, the return speed is 30 km/hr -/
theorem return_speed_calculation (distance : ℝ) (speed_going : ℝ) (average_speed : ℝ) 
  (h1 : distance = 150)
  (h2 : speed_going = 50)
  (h3 : average_speed = 37.5) : 
  (2 * distance) / ((distance / speed_going) + (distance / ((2 * distance) / average_speed - distance / speed_going))) = 30 :=
by sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l3264_326491


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l3264_326442

theorem parallel_angles_theorem (α β : Real) :
  (∃ k : ℤ, α + β = k * 180) →  -- Parallel sides condition
  (α = 3 * β - 36) →            -- Relationship between α and β
  (α = 18 ∨ α = 126) :=         -- Conclusion
by sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l3264_326442


namespace NUMINAMATH_CALUDE_student_ranking_l3264_326432

theorem student_ranking (total : Nat) (rank_right : Nat) (rank_left : Nat) : 
  total = 31 → rank_right = 21 → rank_left = total - rank_right + 1 → rank_left = 11 := by
  sorry

end NUMINAMATH_CALUDE_student_ranking_l3264_326432


namespace NUMINAMATH_CALUDE_zoo_trip_remaining_money_l3264_326457

/-- Calculates the amount of money left for lunch and snacks after a zoo trip -/
theorem zoo_trip_remaining_money 
  (ticket_price : ℚ)
  (bus_fare : ℚ)
  (total_money : ℚ)
  (num_people : ℕ)
  (h1 : ticket_price = 5)
  (h2 : bus_fare = 3/2)
  (h3 : total_money = 40)
  (h4 : num_people = 2)
  : total_money - (num_people * ticket_price + num_people * bus_fare * 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_remaining_money_l3264_326457


namespace NUMINAMATH_CALUDE_no_consecutive_heads_probability_l3264_326426

/-- The number of ways to toss n coins such that no two heads appear consecutively -/
def f : ℕ → ℕ
| 0 => 1  -- Convention for empty sequence
| 1 => 2  -- Base case
| 2 => 3  -- Base case
| (n + 3) => f (n + 2) + f (n + 1)

/-- The probability of no two heads appearing consecutively in 10 coin tosses -/
theorem no_consecutive_heads_probability :
  (f 10 : ℚ) / (2^10 : ℚ) = 9/64 := by sorry

end NUMINAMATH_CALUDE_no_consecutive_heads_probability_l3264_326426


namespace NUMINAMATH_CALUDE_polygon_sides_l3264_326489

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3264_326489


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l3264_326496

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D :=
  {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetricToXOyPlane (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetric_point_xoy_plane :
  let m : Point3D := ⟨2, 5, 8⟩
  let n : Point3D := ⟨2, 5, -8⟩
  symmetricToXOyPlane m n := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l3264_326496


namespace NUMINAMATH_CALUDE_solve_system_l3264_326453

theorem solve_system (a b : ℤ) 
  (eq1 : 2013 * a + 2015 * b = 2023)
  (eq2 : 2017 * a + 2019 * b = 2027) :
  a - b = -9 := by sorry

end NUMINAMATH_CALUDE_solve_system_l3264_326453


namespace NUMINAMATH_CALUDE_movie_of_the_year_criterion_l3264_326448

/-- The number of members in the Cinematic Academy -/
def academy_members : ℕ := 1500

/-- The fraction of top-10 lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1/2

/-- The smallest number of top-10 lists a film must appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 750

theorem movie_of_the_year_criterion :
  min_lists = (academy_members : ℚ) * required_fraction :=
by sorry

end NUMINAMATH_CALUDE_movie_of_the_year_criterion_l3264_326448


namespace NUMINAMATH_CALUDE_min_red_edges_six_red_edges_possible_l3264_326497

/-- Represents the color of an edge -/
inductive Color
| Red
| Green

/-- Represents a cube with colored edges -/
structure Cube :=
  (edges : Fin 12 → Color)

/-- Checks if a face has at least one red edge -/
def faceHasRedEdge (c : Cube) (face : Fin 6) : Prop := sorry

/-- The condition that every face of the cube has at least one red edge -/
def everyFaceHasRedEdge (c : Cube) : Prop :=
  ∀ face : Fin 6, faceHasRedEdge c face

/-- Counts the number of red edges in a cube -/
def countRedEdges (c : Cube) : Nat := sorry

/-- Theorem stating that the minimum number of red edges is 6 -/
theorem min_red_edges (c : Cube) (h : everyFaceHasRedEdge c) : 
  countRedEdges c ≥ 6 := sorry

/-- Theorem stating that 6 red edges is achievable -/
theorem six_red_edges_possible : 
  ∃ c : Cube, everyFaceHasRedEdge c ∧ countRedEdges c = 6 := sorry

end NUMINAMATH_CALUDE_min_red_edges_six_red_edges_possible_l3264_326497


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l3264_326451

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) (slower_speed : ℝ) :
  actual_distance = 24 →
  faster_speed = 5 →
  additional_distance = 6 →
  faster_speed * (actual_distance / slower_speed) = actual_distance + additional_distance →
  slower_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l3264_326451


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l3264_326407

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_x_values : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

theorem triangle_existence_condition (x : ℕ) :
  x > 0 → (triangle_exists 7 (x + 3) 10 ↔ x ∈ valid_x_values) := by sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l3264_326407


namespace NUMINAMATH_CALUDE_work_increase_per_person_l3264_326484

/-- Calculates the increase in work per person when 1/6 of the workforce is absent -/
theorem work_increase_per_person (p : ℕ) (W : ℝ) (h : p > 0) :
  let initial_work_per_person := W / p
  let remaining_workers := p - p / 6
  let new_work_per_person := W / remaining_workers
  new_work_per_person - initial_work_per_person = W / (5 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_per_person_l3264_326484


namespace NUMINAMATH_CALUDE_calculation_proof_l3264_326423

theorem calculation_proof : (-2)^0 - Real.sqrt 8 - abs (-5) + 4 * Real.sin (π/4) = -4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3264_326423


namespace NUMINAMATH_CALUDE_common_solution_of_linear_system_l3264_326416

theorem common_solution_of_linear_system :
  (∀ (a b : ℚ), ∃ (x y : ℚ), (a - b) * x - (a + b) * y = a + b) →
  (∃! (x y : ℚ), ∀ (a b : ℚ), (a - b) * x - (a + b) * y = a + b ∧ x = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_of_linear_system_l3264_326416


namespace NUMINAMATH_CALUDE_xy_value_l3264_326495

theorem xy_value (x y : ℝ) (h : Real.sqrt (x + 2) + (y - Real.sqrt 3) ^ 2 = 0) : 
  x * y = -2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3264_326495


namespace NUMINAMATH_CALUDE_A_eq_real_iff_m_in_range_l3264_326439

/-- The set A defined by the quadratic inequality -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + 2 * m * x + 1 > 0}

/-- Theorem stating the equivalence between A being equal to ℝ and m being in [0, 1) -/
theorem A_eq_real_iff_m_in_range (m : ℝ) : A m = Set.univ ↔ m ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_A_eq_real_iff_m_in_range_l3264_326439


namespace NUMINAMATH_CALUDE_ecommerce_sales_analysis_l3264_326412

/-- Represents the sales model for an e-commerce platform. -/
structure SalesModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the daily sales volume for a given price. -/
def daily_sales (model : SalesModel) (price : ℝ) : ℝ :=
  model.initial_sales + model.price_sensitivity * (model.initial_price - price)

/-- Calculates the daily profit for a given price. -/
def daily_profit (model : SalesModel) (price : ℝ) : ℝ :=
  (price - model.cost_price) * daily_sales model price

/-- The e-commerce platform's sales model. -/
def ecommerce_model : SalesModel := {
  cost_price := 40
  initial_price := 60
  initial_sales := 20
  price_sensitivity := 2
}

/-- Xiao Ming's store price. -/
def xiaoming_price : ℝ := 62.5

theorem ecommerce_sales_analysis 
  (h1 : daily_sales ecommerce_model 45 = 50)
  (h2 : ∃ x, x ≥ 40 ∧ x < 60 ∧ daily_profit ecommerce_model x = daily_profit ecommerce_model 60 ∧
             ∀ y, y ≥ 40 ∧ y < 60 ∧ daily_profit ecommerce_model y = daily_profit ecommerce_model 60 → x ≤ y)
  (h3 : ∃ d : ℝ, d ≥ 0 ∧ d ≤ 1 ∧ xiaoming_price * (1 - d) ≤ 50 ∧
             ∀ e, e ≥ 0 ∧ e < d ∧ xiaoming_price * (1 - e) ≤ 50 → False) :
  (daily_sales ecommerce_model 45 = 50) ∧
  (∃ x, x = 50 ∧ daily_profit ecommerce_model x = daily_profit ecommerce_model 60 ∧
        ∀ y, y ≥ 40 ∧ y < 60 ∧ daily_profit ecommerce_model y = daily_profit ecommerce_model 60 → x ≤ y) ∧
  (∃ d : ℝ, d = 0.2 ∧ xiaoming_price * (1 - d) ≤ 50 ∧
            ∀ e, e ≥ 0 ∧ e < d ∧ xiaoming_price * (1 - e) ≤ 50 → False) := by
  sorry

end NUMINAMATH_CALUDE_ecommerce_sales_analysis_l3264_326412


namespace NUMINAMATH_CALUDE_factorization_problems_l3264_326403

theorem factorization_problems (x y : ℝ) (m : ℝ) : 
  (x^2 - 4 = (x + 2) * (x - 2)) ∧ 
  (2*m*x^2 - 4*m*x + 2*m = 2*m*(x - 1)^2) ∧ 
  ((y^2 - 1)^2 - 6*(y^2 - 1) + 9 = (y + 2)^2 * (y - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l3264_326403


namespace NUMINAMATH_CALUDE_clothes_expenditure_fraction_l3264_326466

theorem clothes_expenditure_fraction (initial_amount : ℝ) (remaining_amount : ℝ) (F : ℝ) : 
  initial_amount = 499.9999999999999 →
  remaining_amount = 200 →
  remaining_amount = (3/5) * (1 - F) * initial_amount →
  F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_clothes_expenditure_fraction_l3264_326466


namespace NUMINAMATH_CALUDE_max_pages_copied_l3264_326476

-- Define the cost per 2 pages in cents
def cost_per_2_pages : ℕ := 7

-- Define the fixed fee in cents
def fixed_fee : ℕ := 500

-- Define the total budget in cents
def total_budget : ℕ := 3500

-- Define the function to calculate the number of pages
def pages_copied (budget : ℕ) : ℕ :=
  ((budget - fixed_fee) * 2) / cost_per_2_pages

-- Theorem statement
theorem max_pages_copied :
  pages_copied total_budget = 857 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l3264_326476


namespace NUMINAMATH_CALUDE_business_valuation_l3264_326493

def business_value (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) : ℕ :=
  (2 * sale_price : ℕ)

theorem business_valuation (total_ownership : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) 
  (h1 : total_ownership = 2/3)
  (h2 : sold_fraction = 3/4)
  (h3 : sale_price = 6500) :
  business_value total_ownership sold_fraction sale_price = 13000 := by
  sorry

end NUMINAMATH_CALUDE_business_valuation_l3264_326493


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l3264_326485

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l3264_326485


namespace NUMINAMATH_CALUDE_tan_simplification_l3264_326443

theorem tan_simplification (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + 3 * Real.sin α) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_simplification_l3264_326443


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_seven_l3264_326468

theorem factorial_fraction_equals_seven : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_seven_l3264_326468


namespace NUMINAMATH_CALUDE_certain_event_good_product_l3264_326430

/-- Represents the total number of products --/
def total_products : ℕ := 12

/-- Represents the number of good products --/
def good_products : ℕ := 10

/-- Represents the number of defective products --/
def defective_products : ℕ := 2

/-- Represents the number of products selected --/
def selected_products : ℕ := 3

/-- Represents a selection of products --/
def Selection := Fin selected_products → Fin total_products

/-- Predicate to check if a selection contains at least one good product --/
def contains_good_product (s : Selection) : Prop :=
  ∃ i, s i < good_products

/-- The main theorem stating that any selection contains at least one good product --/
theorem certain_event_good_product :
  ∀ s : Selection, contains_good_product s :=
sorry

end NUMINAMATH_CALUDE_certain_event_good_product_l3264_326430


namespace NUMINAMATH_CALUDE_draw_three_from_fifteen_l3264_326413

def box_numbers : List Nat := [1, 2, 3, 4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

def total_combinations (n k : Nat) : Nat :=
  Nat.choose n k

theorem draw_three_from_fifteen :
  total_combinations (List.length box_numbers) 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_draw_three_from_fifteen_l3264_326413


namespace NUMINAMATH_CALUDE_common_ratio_of_specific_geometric_sequence_l3264_326441

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_specific_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2/3 →
  a 4 = ∫ x in (1:ℝ)..(4:ℝ), (1 + 2*x) →
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_common_ratio_of_specific_geometric_sequence_l3264_326441


namespace NUMINAMATH_CALUDE_unique_permutations_four_letter_two_pairs_is_six_l3264_326408

/-- The number of unique permutations of a four-letter word with two pairs of identical letters -/
def unique_permutations_four_letter_two_pairs : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem: The number of unique permutations of a four-letter word with two pairs of identical letters is 6 -/
theorem unique_permutations_four_letter_two_pairs_is_six :
  unique_permutations_four_letter_two_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_permutations_four_letter_two_pairs_is_six_l3264_326408


namespace NUMINAMATH_CALUDE_N_subset_M_l3264_326431

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 ≥ x}
def N : Set ℝ := {x : ℝ | Real.log (x + 1) / Real.log (1/2) > 0}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l3264_326431


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l3264_326474

theorem consecutive_odd_squares_sum : ∃ x : ℤ, 
  (x - 2)^2 + x^2 + (x + 2)^2 = 5555 ∧ 
  Odd x ∧ Odd (x - 2) ∧ Odd (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_sum_l3264_326474


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3264_326401

/-- The sum of the first n terms of an arithmetic sequence -/
def T (a d : ℚ) (n : ℕ+) : ℚ := n * (2 * a + (n - 1) * d) / 2

/-- The theorem states that if T_{4n} / T_n is constant for an arithmetic sequence
    with common difference 5, then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_first_term
  (h : ∃ (k : ℚ), ∀ (n : ℕ+), T a 5 (4 * n) / T a 5 n = k) :
  a = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3264_326401


namespace NUMINAMATH_CALUDE_equilateral_triangle_reflection_parity_l3264_326447

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents a reflection of a triangle -/
def reflect (t : Triangle) : Triangle := sorry

/-- Predicate to check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Predicate to check if two triangles coincide -/
def coincide (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: If an equilateral triangle is reflected multiple times and 
    coincides with the original, the number of reflections is even -/
theorem equilateral_triangle_reflection_parity 
  (t : Triangle) (n : ℕ) (h1 : is_equilateral t) :
  (coincide ((reflect^[n]) t) t) → Even n := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_reflection_parity_l3264_326447


namespace NUMINAMATH_CALUDE_power_function_through_point_l3264_326400

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →
  f 27 = 3 →
  a = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3264_326400


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3264_326490

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 14) (h2 : x * y = 45) :
  1 / x + 1 / y = 14 / 45 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3264_326490


namespace NUMINAMATH_CALUDE_cookies_eaten_l3264_326470

/-- Given a package of cookies where some were eaten, this theorem proves
    the number of cookies eaten, given the initial count and remaining count. -/
theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (h : initial = 18) (h' : remaining = 9) :
  initial - remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l3264_326470


namespace NUMINAMATH_CALUDE_distribution_count_l3264_326492

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the Stirling number of the second kind function
noncomputable def stirling_second (n k : ℕ) : ℕ :=
  sorry  -- Implementation of Stirling number of the second kind

-- Theorem statement
theorem distribution_count :
  (stirling_second num_balls num_boxes) = 25 :=
sorry

end NUMINAMATH_CALUDE_distribution_count_l3264_326492


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3264_326414

theorem hyperbola_conjugate_axis_length :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (x^2 / a^2) - (y^2 / b^2) = 1 ∧
  2 * b = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l3264_326414


namespace NUMINAMATH_CALUDE_lower_right_is_three_l3264_326481

/-- Represents a 5x5 grid with digits from 1 to 5 -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a number is unique in its row -/
def unique_in_row (g : Grid) (row col : Fin 5) : Prop :=
  ∀ c : Fin 5, c ≠ col → g row c ≠ g row col

/-- Checks if a number is unique in its column -/
def unique_in_col (g : Grid) (row col : Fin 5) : Prop :=
  ∀ r : Fin 5, r ≠ row → g r col ≠ g row col

/-- Checks if the grid satisfies the uniqueness conditions -/
def valid_grid (g : Grid) : Prop :=
  ∀ r c : Fin 5, unique_in_row g r c ∧ unique_in_col g r c

/-- The theorem to be proved -/
theorem lower_right_is_three (g : Grid) 
  (h1 : valid_grid g)
  (h2 : g 0 0 = 1)
  (h3 : g 0 4 = 2)
  (h4 : g 1 1 = 4)
  (h5 : g 2 3 = 3)
  (h6 : g 3 2 = 5) :
  g 4 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lower_right_is_three_l3264_326481


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3264_326450

-- Define the sets A and B
def A : Set ℝ := {x | (2*x + 3)/(x - 2) > 0}
def B : Set ℝ := {x | |x - 1| < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3264_326450


namespace NUMINAMATH_CALUDE_calculation_proof_l3264_326455

theorem calculation_proof : Real.sqrt 4 - Real.sin (30 * π / 180) - (π - 1) ^ 0 + 2⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3264_326455


namespace NUMINAMATH_CALUDE_no_extreme_points_l3264_326460

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x

def f_derivative (x : ℝ) : ℝ := 3*(x - 1)^2

theorem no_extreme_points (h : ∀ x, f_derivative x ≥ 0) :
  ∀ x, ¬ (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_l3264_326460


namespace NUMINAMATH_CALUDE_village_population_equality_l3264_326458

theorem village_population_equality (x_initial : ℕ) (x_rate : ℕ) (y_initial : ℕ) (y_rate : ℕ) :
  x_initial = 72000 →
  x_rate = 1200 →
  y_initial = 42000 →
  y_rate = 800 →
  ∃ n : ℕ, (x_initial - n * x_rate = y_initial + n * y_rate) ∧ n = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l3264_326458


namespace NUMINAMATH_CALUDE_tennis_ball_difference_l3264_326427

/-- Given the number of tennis balls for Brian, Frodo, and Lily, prove that Frodo has 8 more tennis balls than Lily. -/
theorem tennis_ball_difference (brian frodo lily : ℕ) : 
  brian = 2 * frodo → 
  lily = 3 → 
  brian = 22 → 
  frodo - lily = 8 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_difference_l3264_326427


namespace NUMINAMATH_CALUDE_joan_apples_l3264_326433

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has now -/
def apples_left : ℕ := 16

/-- The total number of apples Joan picked from the orchard -/
def total_apples : ℕ := apples_given + apples_left

theorem joan_apples : total_apples = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l3264_326433


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3264_326402

/-- A line passing through (1,0) parallel to x-2y-2=0 has equation x-2y-1=0 -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y - 1 = 0) ↔ 
  (∃ (m b : ℝ), y = m*x + b ∧ 
                 m = (1 : ℝ)/(2 : ℝ) ∧ 
                 1 = m*1 + b ∧ 
                 0 = m*0 + b) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3264_326402


namespace NUMINAMATH_CALUDE_secret_spread_reaches_target_target_day_minimal_l3264_326478

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when the secret reaches 3280 students -/
def target_day : ℕ := 7

theorem secret_spread_reaches_target :
  secret_spread target_day = 3280 :=
sorry

theorem target_day_minimal :
  ∀ k < target_day, secret_spread k < 3280 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_reaches_target_target_day_minimal_l3264_326478


namespace NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_complex_inequality_equality_at_one_l3264_326424

theorem complex_inequality (z : ℂ) : Complex.abs z ^ 2 + 2 * Complex.abs (z - 1) ≥ 1 :=
by sorry

theorem complex_inequality_equality : ∃ z : ℂ, Complex.abs z ^ 2 + 2 * Complex.abs (z - 1) = 1 :=
by sorry

theorem complex_inequality_equality_at_one : Complex.abs (1 : ℂ) ^ 2 + 2 * Complex.abs (1 - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_inequality_complex_inequality_equality_complex_inequality_equality_at_one_l3264_326424


namespace NUMINAMATH_CALUDE_common_roots_product_l3264_326463

theorem common_roots_product (C : ℝ) : 
  ∃ (p q r t : ℝ), 
    (p^3 + 2*p^2 + 15 = 0) ∧ 
    (q^3 + 2*q^2 + 15 = 0) ∧ 
    (r^3 + 2*r^2 + 15 = 0) ∧
    (p^3 + C*p + 30 = 0) ∧ 
    (q^3 + C*q + 30 = 0) ∧ 
    (t^3 + C*t + 30 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
    (p ≠ t) ∧ (q ≠ t) →
    p * q = -5 * Real.rpow 2 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l3264_326463


namespace NUMINAMATH_CALUDE_sequence_divisibility_l3264_326456

def a (n : ℕ) : ℕ := 2 * 10^(n + 1) + 19

def divides_sequence (p : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 1 ∧ p ∣ a n

theorem sequence_divisibility :
  {p : ℕ | p.Prime ∧ p ≤ 19 ∧ divides_sequence p} = {3, 7, 13, 17} := by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l3264_326456


namespace NUMINAMATH_CALUDE_binomial_variance_example_l3264_326480

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(8, 0.7) is 1.68 -/
theorem binomial_variance_example :
  let X : BinomialRV := ⟨8, 0.7, by norm_num⟩
  variance X = 1.68 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l3264_326480


namespace NUMINAMATH_CALUDE_probability_second_math_given_first_math_l3264_326411

def total_questions : ℕ := 5
def math_questions : ℕ := 3
def physics_questions : ℕ := 2

theorem probability_second_math_given_first_math :
  let P : ℝ := (math_questions * (math_questions - 1)) / (total_questions * (total_questions - 1))
  let Q : ℝ := math_questions / total_questions
  P / Q = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_second_math_given_first_math_l3264_326411


namespace NUMINAMATH_CALUDE_only_B_is_random_event_l3264_326429

-- Define the events
inductive Event
| A : Event  -- Water boils at 100°C under standard atmospheric pressure
| B : Event  -- Buying a lottery ticket and winning a prize
| C : Event  -- A runner's speed is 30 meters per second
| D : Event  -- Drawing a red ball from a bag containing only white and black balls

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.A => false
  | Event.B => true
  | Event.C => false
  | Event.D => false

-- Theorem: Only Event B is a random event
theorem only_B_is_random_event :
  ∀ e : Event, isRandomEvent e ↔ e = Event.B :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_random_event_l3264_326429


namespace NUMINAMATH_CALUDE_total_baseball_cards_l3264_326410

def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards +
  john_cards + sarah_cards + emma_cards = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l3264_326410


namespace NUMINAMATH_CALUDE_problem_statement_l3264_326473

theorem problem_statement (a b : ℝ) (ha : a ≠ b) 
  (ha_eq : a^2 - 13*a + 1 = 0) (hb_eq : b^2 - 13*b + 1 = 0) :
  b / (1 + b) + (a^2 + a) / (a^2 + 2*a + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3264_326473


namespace NUMINAMATH_CALUDE_first_term_to_diff_ratio_l3264_326428

/-- An arithmetic sequence with a given property -/
structure ArithmeticSequence where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_property : (9 * a + 36 * d) = 3 * (6 * a + 15 * d)

/-- The ratio of the first term to the common difference is 1:-1 -/
theorem first_term_to_diff_ratio (seq : ArithmeticSequence) : seq.a / seq.d = -1 := by
  sorry

#check first_term_to_diff_ratio

end NUMINAMATH_CALUDE_first_term_to_diff_ratio_l3264_326428


namespace NUMINAMATH_CALUDE_total_votes_correct_l3264_326440

/-- The total number of votes in an election --/
def total_votes : ℕ := 560000

/-- The percentage of valid votes that Candidate A received --/
def candidate_A_percentage : ℚ := 55 / 100

/-- The percentage of invalid votes --/
def invalid_percentage : ℚ := 15 / 100

/-- The number of valid votes Candidate A received --/
def candidate_A_votes : ℕ := 261800

/-- Theorem stating that the total number of votes is correct given the conditions --/
theorem total_votes_correct :
  (↑candidate_A_votes : ℚ) = 
    (1 - invalid_percentage) * candidate_A_percentage * (↑total_votes : ℚ) :=
sorry

end NUMINAMATH_CALUDE_total_votes_correct_l3264_326440


namespace NUMINAMATH_CALUDE_school_vote_problem_l3264_326482

theorem school_vote_problem (U A B : Finset Nat) : 
  Finset.card U = 250 →
  Finset.card A = 175 →
  Finset.card B = 140 →
  Finset.card (U \ (A ∪ B)) = 45 →
  Finset.card (A ∩ B) = 110 := by
sorry

end NUMINAMATH_CALUDE_school_vote_problem_l3264_326482


namespace NUMINAMATH_CALUDE_regular_septagon_interior_angle_measure_l3264_326462

/-- The number of sides in a septagon -/
def n : ℕ := 7

/-- A regular septagon is a polygon with 7 sides and all interior angles equal -/
structure RegularSeptagon where
  sides : Fin n → ℝ
  angles : Fin n → ℝ
  all_sides_equal : ∀ i j : Fin n, sides i = sides j
  all_angles_equal : ∀ i j : Fin n, angles i = angles j

/-- Theorem: The measure of each interior angle in a regular septagon is 900/7 degrees -/
theorem regular_septagon_interior_angle_measure (s : RegularSeptagon) :
  ∀ i : Fin n, s.angles i = 900 / 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_septagon_interior_angle_measure_l3264_326462


namespace NUMINAMATH_CALUDE_solve_for_b_l3264_326446

/-- The piecewise function f(x) -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 3^x

/-- Theorem stating that if f(f(1/2)) = 9, then b = -1/2 -/
theorem solve_for_b :
  ∀ b : ℝ, f b (f b (1/2)) = 9 → b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3264_326446


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3264_326437

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3264_326437


namespace NUMINAMATH_CALUDE_total_wheels_l3264_326461

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels Dimitri saw at the park -/
theorem total_wheels : 
  bicycle_wheels * adults_on_bicycles + tricycle_wheels * children_on_tricycles = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l3264_326461


namespace NUMINAMATH_CALUDE_focus_of_symmetric_parabola_l3264_326471

/-- The focus of a parabola symmetric to x^2 = 4y with respect to x + y = 0 -/
def symmetric_parabola_focus : ℝ × ℝ :=
  (-1, 0)

/-- The original parabola equation -/
def original_parabola (x y : ℝ) : Prop :=
  x^2 = 4*y

/-- The line of symmetry equation -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y = 0

theorem focus_of_symmetric_parabola :
  symmetric_parabola_focus = (-1, 0) :=
sorry

end NUMINAMATH_CALUDE_focus_of_symmetric_parabola_l3264_326471


namespace NUMINAMATH_CALUDE_mary_flour_problem_l3264_326486

theorem mary_flour_problem (recipe_flour : ℕ) (flour_to_add : ℕ) 
  (h1 : recipe_flour = 7)
  (h2 : flour_to_add = 5) :
  recipe_flour - flour_to_add = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_problem_l3264_326486
