import Mathlib

namespace NUMINAMATH_CALUDE_taehyungs_mother_age_l553_55323

/-- Given the age differences and the younger brother's age, prove Taehyung's mother's age --/
theorem taehyungs_mother_age :
  ∀ (taehyung_age mother_age brother_age : ℕ),
    mother_age - taehyung_age = 31 →
    taehyung_age - brother_age = 5 →
    brother_age = 7 →
    mother_age = 43 := by
  sorry

end NUMINAMATH_CALUDE_taehyungs_mother_age_l553_55323


namespace NUMINAMATH_CALUDE_factorization_f_max_value_g_l553_55333

-- Define the polynomials
def f (x : ℝ) : ℝ := x^2 - 4*x - 5
def g (x : ℝ) : ℝ := -2*x^2 - 4*x + 3

-- Theorem for factorization of f
theorem factorization_f : ∀ x : ℝ, f x = (x + 1) * (x - 5) := by sorry

-- Theorem for maximum value of g
theorem max_value_g : 
  (∀ x : ℝ, g x ≤ 5) ∧ g (-1) = 5 := by sorry

end NUMINAMATH_CALUDE_factorization_f_max_value_g_l553_55333


namespace NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l553_55392

/-- The volume of a cube with total edge length of 72 feet is 216 cubic feet. -/
theorem cube_volume_from_total_edge_length :
  ∀ (s : ℝ), (12 * s = 72) → s^3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l553_55392


namespace NUMINAMATH_CALUDE_tetrahedron_sum_is_15_l553_55358

/-- Represents a tetrahedron -/
structure Tetrahedron where
  edges : ℕ
  vertices : ℕ
  faces : ℕ

/-- The properties of a tetrahedron -/
def is_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges = 6 ∧ t.vertices = 4 ∧ t.faces = 4

/-- The sum calculation with one vertex counted twice -/
def sum_with_extra_vertex (t : Tetrahedron) : ℕ :=
  t.edges + (t.vertices + 1) + t.faces

/-- Theorem: The sum of edges, faces, and vertices (with one counted twice) of a tetrahedron is 15 -/
theorem tetrahedron_sum_is_15 (t : Tetrahedron) (h : is_tetrahedron t) :
  sum_with_extra_vertex t = 15 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_is_15_l553_55358


namespace NUMINAMATH_CALUDE_prob_two_girls_l553_55361

/-- The probability of selecting two girls from a group of 15 members, where 6 are girls -/
theorem prob_two_girls (total : ℕ) (girls : ℕ) (h1 : total = 15) (h2 : girls = 6) :
  (girls.choose 2 : ℚ) / (total.choose 2) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_girls_l553_55361


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l553_55364

def A : Set ℝ := {x | x - 6 < 0}
def B : Set ℝ := {-3, 5, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {-3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l553_55364


namespace NUMINAMATH_CALUDE_route_length_l553_55312

/-- Proves that the length of a route is 125 miles given the conditions of two trains meeting. -/
theorem route_length (time_A time_B meeting_distance : ℝ) 
  (h1 : time_A = 12)
  (h2 : time_B = 8)
  (h3 : meeting_distance = 50)
  (h4 : time_A > 0)
  (h5 : time_B > 0)
  (h6 : meeting_distance > 0) :
  ∃ (route_length : ℝ),
    route_length = 125 ∧
    route_length / time_A * (meeting_distance * time_A / route_length) = meeting_distance ∧
    route_length / time_B * (meeting_distance * time_A / route_length) = route_length - meeting_distance :=
by
  sorry


end NUMINAMATH_CALUDE_route_length_l553_55312


namespace NUMINAMATH_CALUDE_horner_rule_v4_l553_55339

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_rule_v4 :
  horner_v4 2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_horner_rule_v4_l553_55339


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l553_55337

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l553_55337


namespace NUMINAMATH_CALUDE_range_of_a_l553_55363

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - a)^2 < 1}

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l553_55363


namespace NUMINAMATH_CALUDE_christinas_earnings_l553_55366

/-- The amount Christina earns for planting flowers and mowing the lawn -/
theorem christinas_earnings (flower_rate : ℚ) (mow_rate : ℚ) (flowers_planted : ℚ) (area_mowed : ℚ) :
  flower_rate = 8/3 →
  mow_rate = 5/2 →
  flowers_planted = 9/4 →
  area_mowed = 7/3 →
  flower_rate * flowers_planted + mow_rate * area_mowed = 71/6 := by
sorry

end NUMINAMATH_CALUDE_christinas_earnings_l553_55366


namespace NUMINAMATH_CALUDE_exists_common_point_l553_55351

/-- Represents a rectangular map with a scale factor -/
structure Map where
  scale : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point on a map -/
structure MapPoint where
  x : ℝ
  y : ℝ

/-- Theorem stating that there exists a common point on two maps of different scales -/
theorem exists_common_point (map1 map2 : Map) (h_scale : map2.scale = 5 * map1.scale) :
  ∃ (p1 : MapPoint) (p2 : MapPoint),
    p1.x / map1.width = p2.x / map2.width ∧
    p1.y / map1.height = p2.y / map2.height :=
sorry

end NUMINAMATH_CALUDE_exists_common_point_l553_55351


namespace NUMINAMATH_CALUDE_weight_loss_probability_is_0_241_l553_55368

/-- The probability of a person losing weight after taking a drug, given the total number of volunteers and the number of people who lost weight. -/
def probability_of_weight_loss (total_volunteers : ℕ) (weight_loss_count : ℕ) : ℚ :=
  weight_loss_count / total_volunteers

/-- Theorem stating that the probability of weight loss is 0.241 given the provided data. -/
theorem weight_loss_probability_is_0_241 
  (total_volunteers : ℕ) 
  (weight_loss_count : ℕ) 
  (h1 : total_volunteers = 1000) 
  (h2 : weight_loss_count = 241) : 
  probability_of_weight_loss total_volunteers weight_loss_count = 241 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_probability_is_0_241_l553_55368


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l553_55388

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ 
  (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l553_55388


namespace NUMINAMATH_CALUDE_expression_upper_bound_l553_55390

theorem expression_upper_bound (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_prod : a * c = b * d)
  (h_sum : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b) ≤ 4 ∧ 
  ∃ (a' b' c' d' : ℝ), a' / c' + c' / a' + b' / d' + d' / b' = 4 :=
sorry

end NUMINAMATH_CALUDE_expression_upper_bound_l553_55390


namespace NUMINAMATH_CALUDE_litter_patrol_collection_l553_55316

/-- The number of glass bottles picked up by the Litter Patrol -/
def glass_bottles : ℕ := 10

/-- The number of aluminum cans picked up by the Litter Patrol -/
def aluminum_cans : ℕ := 8

/-- The total number of pieces of litter is the sum of glass bottles and aluminum cans -/
def total_litter : ℕ := glass_bottles + aluminum_cans

/-- Theorem stating that the total number of pieces of litter is 18 -/
theorem litter_patrol_collection : total_litter = 18 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_collection_l553_55316


namespace NUMINAMATH_CALUDE_dilation_example_l553_55324

/-- Dilation of a complex number -/
def dilation (center scale : ℂ) (z : ℂ) : ℂ :=
  center + scale * (z - center)

theorem dilation_example : 
  dilation (2 - 3*I) (-2) (-1 + 2*I) = 8 - 13*I :=
by sorry

end NUMINAMATH_CALUDE_dilation_example_l553_55324


namespace NUMINAMATH_CALUDE_polynomial_product_equality_l553_55386

theorem polynomial_product_equality (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equality_l553_55386


namespace NUMINAMATH_CALUDE_complex_absolute_value_l553_55327

theorem complex_absolute_value (x y : ℝ) : 
  (Complex.I : ℂ) * (x + 3 * Complex.I) = y - Complex.I → 
  Complex.abs (x - y * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l553_55327


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l553_55367

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l553_55367


namespace NUMINAMATH_CALUDE_bottles_produced_l553_55370

-- Define the production rate of 4 machines
def production_rate_4 : ℕ := 16

-- Define the number of minutes
def minutes : ℕ := 3

-- Define the number of machines in the first scenario
def machines_1 : ℕ := 4

-- Define the number of machines in the second scenario
def machines_2 : ℕ := 8

-- Theorem to prove
theorem bottles_produced :
  (machines_2 * minutes * (production_rate_4 / machines_1)) = 96 := by
  sorry

end NUMINAMATH_CALUDE_bottles_produced_l553_55370


namespace NUMINAMATH_CALUDE_erased_numbers_l553_55353

def numbers_with_one : ℕ := 20
def numbers_with_two : ℕ := 19
def numbers_without_one_or_two : ℕ := 30
def total_numbers : ℕ := 100

theorem erased_numbers :
  numbers_with_one + numbers_with_two + numbers_without_one_or_two ≤ total_numbers ∧
  total_numbers - (numbers_with_one + numbers_with_two + numbers_without_one_or_two - 2) = 33 :=
sorry

end NUMINAMATH_CALUDE_erased_numbers_l553_55353


namespace NUMINAMATH_CALUDE_find_n_l553_55302

theorem find_n : ∃ n : ℝ, n + (n + 1) + (n + 2) + (n + 3) = 20 ∧ n = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l553_55302


namespace NUMINAMATH_CALUDE_certain_number_problem_l553_55318

theorem certain_number_problem (x : ℕ) (n : ℕ) : x = 4 → 3 * x + n = 48 → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l553_55318


namespace NUMINAMATH_CALUDE_vasya_petya_number_ambiguity_l553_55310

theorem vasya_petya_number_ambiguity (a : ℝ) (ha : a ≠ 0) :
  ∃ b : ℝ, b ≠ a ∧ a^4 + a^2 = b^4 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_vasya_petya_number_ambiguity_l553_55310


namespace NUMINAMATH_CALUDE_expected_value_of_coins_l553_55375

-- Define coin values in cents
def penny : ℚ := 1
def nickel : ℚ := 5
def dime : ℚ := 10
def quarter : ℚ := 25
def half_dollar : ℚ := 50

-- Define probability of heads
def prob_heads : ℚ := 1/2

-- Define function to calculate expected value for a single coin
def expected_value (coin_value : ℚ) : ℚ := prob_heads * coin_value

-- Theorem statement
theorem expected_value_of_coins : 
  expected_value penny + expected_value nickel + expected_value dime + 
  expected_value quarter + expected_value half_dollar = 45.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_coins_l553_55375


namespace NUMINAMATH_CALUDE_fraction_value_l553_55304

theorem fraction_value (a b c d : ℝ) 
  (ha : a = 4 * b) 
  (hb : b = 3 * c) 
  (hc : c = 5 * d) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l553_55304


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l553_55396

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = 2) :
  (1 / m + 1 / n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l553_55396


namespace NUMINAMATH_CALUDE_skew_lines_distance_and_angle_l553_55391

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (distance : Line → Line → ℝ)
variable (distancePointToLine : Plane → Point → Line → ℝ)
variable (orthogonalProjection : Line → Plane → Line)
variable (angle : Line → Line → ℝ)
variable (perpendicular : Plane → Line → Prop)
variable (intersect : Plane → Line → Point → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem skew_lines_distance_and_angle 
  (a b : Line) (α : Plane) (A : Point) :
  skew a b →
  perpendicular α a →
  intersect α a A →
  let b' := orthogonalProjection b α
  distance a b = distancePointToLine α A b' ∧
  angle b b' + angle a b = 90 := by
  sorry

end NUMINAMATH_CALUDE_skew_lines_distance_and_angle_l553_55391


namespace NUMINAMATH_CALUDE_time_for_A_alone_l553_55336

-- Define the work rates for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
def condition1 : Prop := 3 * (rA + rB) = 1
def condition2 : Prop := 6 * (rB + rC) = 1
def condition3 : Prop := (15/4) * (rA + rC) = 1

-- Theorem statement
theorem time_for_A_alone 
  (h1 : condition1 rA rB)
  (h2 : condition2 rB rC)
  (h3 : condition3 rA rC) :
  1 / rA = 60 / 13 :=
by sorry

end NUMINAMATH_CALUDE_time_for_A_alone_l553_55336


namespace NUMINAMATH_CALUDE_geometry_problem_l553_55377

/-- Two lines are different if they are not equal -/
def different_lines (a b : Line) : Prop := a ≠ b

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def planes_parallel (p1 p2 : Plane) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (l1 l2 : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perp (l1 l2 : Line) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perp (p1 p2 : Plane) : Prop := sorry

theorem geometry_problem (a b : Line) (α β : Plane) 
  (h1 : different_lines a b) (h2 : different_planes α β) : 
  (((line_perp_plane a α ∧ line_perp_plane b β ∧ planes_parallel α β) → lines_parallel a b) ∧
   ((line_perp_plane a α ∧ line_parallel_plane b β ∧ planes_parallel α β) → lines_perp a b) ∧
   (¬((planes_parallel α β ∧ line_in_plane a α ∧ line_in_plane b β) → lines_parallel a b)) ∧
   ((line_perp_plane a α ∧ line_perp_plane b β ∧ planes_perp α β) → lines_perp a b)) := by
  sorry

end NUMINAMATH_CALUDE_geometry_problem_l553_55377


namespace NUMINAMATH_CALUDE_inequality_proof_l553_55343

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = a^2 + b^2 + c^2) : 
  a^2 / (a^2 + b*c) + b^2 / (b^2 + c*a) + c^2 / (c^2 + a*b) ≥ (a + b + c) / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l553_55343


namespace NUMINAMATH_CALUDE_ship_length_in_emily_steps_l553_55350

/-- The length of the ship in terms of Emily's steps -/
def ship_length : ℕ := 70

/-- The number of steps Emily takes from back to front of the ship -/
def steps_back_to_front : ℕ := 210

/-- The number of steps Emily takes from front to back of the ship -/
def steps_front_to_back : ℕ := 42

/-- Emily's walking speed is faster than the ship's speed -/
axiom emily_faster : ∃ (e s : ℝ), e > s ∧ e > 0 ∧ s > 0

/-- Theorem stating the length of the ship in terms of Emily's steps -/
theorem ship_length_in_emily_steps :
  ∃ (e s : ℝ), e > s ∧ e > 0 ∧ s > 0 →
  (steps_back_to_front : ℝ) * e = ship_length + steps_back_to_front * s ∧
  (steps_front_to_back : ℝ) * e = ship_length - steps_front_to_back * s →
  ship_length = 70 := by
  sorry

end NUMINAMATH_CALUDE_ship_length_in_emily_steps_l553_55350


namespace NUMINAMATH_CALUDE_exists_between_elements_l553_55381

/-- A sequence of natural numbers where each natural number appears exactly once -/
def UniqueNatSequence : Type := ℕ → ℕ

/-- The property that each natural number appears exactly once in the sequence -/
def isUniqueNatSequence (a : UniqueNatSequence) : Prop :=
  (∀ n : ℕ, ∃ k : ℕ, a k = n) ∧ 
  (∀ m n : ℕ, a m = a n → m = n)

/-- The main theorem -/
theorem exists_between_elements (a : UniqueNatSequence) (h : isUniqueNatSequence a) :
  ∀ n : ℕ, ∃ k : ℕ, k < n ∧ (a (n - k) < a n ∧ a n < a (n + k)) :=
by sorry

end NUMINAMATH_CALUDE_exists_between_elements_l553_55381


namespace NUMINAMATH_CALUDE_shaded_area_proof_l553_55326

def circle_radius : ℝ := 3

def pi_value : ℝ := 3

theorem shaded_area_proof :
  let circle_area := pi_value * circle_radius^2
  let square_side := circle_radius * Real.sqrt 2
  let square_area := square_side^2
  let total_square_area := 2 * square_area
  circle_area - total_square_area = 9 := by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l553_55326


namespace NUMINAMATH_CALUDE_lines_are_parallel_l553_55331

-- Define the slope and y-intercept of a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the two lines
def l1 : Line := { slope := 2, intercept := 1 }
def l2 : Line := { slope := 2, intercept := 5 }

-- Define parallel lines
def parallel (a b : Line) : Prop := a.slope = b.slope

-- Theorem statement
theorem lines_are_parallel : parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l553_55331


namespace NUMINAMATH_CALUDE_sandy_carrots_l553_55321

theorem sandy_carrots (initial_carrots : ℕ) (taken_carrots : ℕ) :
  initial_carrots = 6 →
  taken_carrots = 3 →
  initial_carrots - taken_carrots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_carrots_l553_55321


namespace NUMINAMATH_CALUDE_bridgette_guest_count_l553_55328

/-- The number of guests Bridgette is inviting -/
def bridgette_guests : ℕ := 84

/-- The number of guests Alex is inviting -/
def alex_guests : ℕ := (2 * bridgette_guests) / 3

/-- The number of extra plates the caterer makes -/
def extra_plates : ℕ := 10

/-- The number of asparagus spears per plate -/
def spears_per_plate : ℕ := 8

/-- The total number of asparagus spears needed -/
def total_spears : ℕ := 1200

theorem bridgette_guest_count : 
  spears_per_plate * (bridgette_guests + alex_guests + extra_plates) = total_spears :=
by sorry

end NUMINAMATH_CALUDE_bridgette_guest_count_l553_55328


namespace NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l553_55320

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem third_term_of_specific_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a2 : a 2 = 2) : 
  a 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l553_55320


namespace NUMINAMATH_CALUDE_student_group_assignments_non_empty_coin_subsets_l553_55329

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of coins --/
def num_coins : ℕ := 7

/-- Theorem for the number of ways to assign students to groups --/
theorem student_group_assignments :
  (num_groups : ℕ) ^ num_students = 32 := by sorry

/-- Theorem for the number of non-empty subsets of coins --/
theorem non_empty_coin_subsets :
  2 ^ num_coins - 1 = 127 := by sorry

end NUMINAMATH_CALUDE_student_group_assignments_non_empty_coin_subsets_l553_55329


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l553_55338

theorem arithmetic_sequence_solution :
  ∀ (y : ℚ),
  let a₁ : ℚ := 3/4
  let a₂ : ℚ := y - 2
  let a₃ : ℚ := 4*y
  (a₂ - a₁ = a₃ - a₂) →  -- arithmetic sequence condition
  y = -19/8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l553_55338


namespace NUMINAMATH_CALUDE_line_representation_slope_nonexistence_x_intercept_angle_of_inclination_l553_55322

/-- Represents the equation ((m^2 - 2m - 3)x + (2m^2 + m - 1)y + 6 - 2m = 0) -/
def equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y + 6 - 2*m = 0

/-- The equation represents a line if and only if m ≠ -1 -/
theorem line_representation (m : ℝ) : 
  (∃ x y, equation m x y) ↔ m ≠ -1 := by sorry

/-- The slope of the line does not exist when m = 1/2 -/
theorem slope_nonexistence (m : ℝ) : 
  (∀ x y, equation m x y → x = 4/3) ↔ m = 1/2 := by sorry

/-- When the x-intercept is -3, m = -5/3 -/
theorem x_intercept (m : ℝ) : 
  (∃ y, equation m (-3) y) ↔ m = -5/3 := by sorry

/-- When the angle of inclination is 45°, m = 4/3 -/
theorem angle_of_inclination (m : ℝ) : 
  (∀ x₁ y₁ x₂ y₂, equation m x₁ y₁ ∧ equation m x₂ y₂ ∧ x₁ ≠ x₂ → 
    (y₂ - y₁) / (x₂ - x₁) = 1) ↔ m = 4/3 := by sorry

end NUMINAMATH_CALUDE_line_representation_slope_nonexistence_x_intercept_angle_of_inclination_l553_55322


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l553_55317

theorem sqrt_sum_simplification : Real.sqrt 3600 + Real.sqrt 1600 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l553_55317


namespace NUMINAMATH_CALUDE_jess_walks_to_store_l553_55300

/-- The number of blocks Jess walks to the store -/
def blocks_to_store : ℕ := sorry

/-- The total number of blocks Jess walks -/
def total_blocks : ℕ := 25

/-- Theorem stating that Jess walks 11 blocks to the store -/
theorem jess_walks_to_store : 
  blocks_to_store = 11 :=
by
  have h1 : blocks_to_store + 6 + 8 = total_blocks := sorry
  sorry


end NUMINAMATH_CALUDE_jess_walks_to_store_l553_55300


namespace NUMINAMATH_CALUDE_projected_revenue_increase_l553_55346

theorem projected_revenue_increase (last_year_revenue : ℝ) :
  let actual_revenue := 0.9 * last_year_revenue
  let projected_revenue := last_year_revenue * (1 + 0.2)
  actual_revenue = 0.75 * projected_revenue :=
by sorry

end NUMINAMATH_CALUDE_projected_revenue_increase_l553_55346


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_l553_55342

/-- The total number of bottle caps Sarah has after buying more -/
def total_bottle_caps (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Sarah has 29 bottle caps in total -/
theorem sarah_bottle_caps : total_bottle_caps 26 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bottle_caps_l553_55342


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l553_55354

/-- Represents a parabola with equation y² = px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line with equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Calculates the chord length of intersection between a parabola and a line -/
def chordLength (parabola : Parabola) (line : Line) : ℝ := 
  sorry

/-- Theorem stating that if a parabola with equation y² = px intersects 
    the line y = x - 1 with a chord length of √10, then p = 1 -/
theorem parabola_intersection_theorem (parabola : Parabola) (line : Line) :
  line.m = 1 ∧ line.b = -1 → chordLength parabola line = Real.sqrt 10 → parabola.p = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l553_55354


namespace NUMINAMATH_CALUDE_custom_mult_seven_three_l553_55307

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := 4*a + 5*b - a*b + 1

/-- Theorem stating that 7 * 3 = 23 under the custom multiplication -/
theorem custom_mult_seven_three : custom_mult 7 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_seven_three_l553_55307


namespace NUMINAMATH_CALUDE_not_all_positive_l553_55359

theorem not_all_positive (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_sq_eq : a^2 + b^2 + c^2 = 12)
  (prod_eq : a * b * c = 1) :
  ¬(a > 0 ∧ b > 0 ∧ c > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_positive_l553_55359


namespace NUMINAMATH_CALUDE_product_congruence_l553_55348

theorem product_congruence : 56 * 89 * 94 ≡ 21 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l553_55348


namespace NUMINAMATH_CALUDE_intersection_range_l553_55378

def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

theorem intersection_range (k : ℝ) (m : ℝ) :
  (∀ x y, line k x = y ∧ ellipse m x y → 
    ∃ x' y', x ≠ x' ∧ line k x' = y' ∧ ellipse m x' y') →
  m ∈ Set.Ioo 1 5 ∪ Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l553_55378


namespace NUMINAMATH_CALUDE_intersection_M_N_l553_55340

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l553_55340


namespace NUMINAMATH_CALUDE_inequality_theorem_l553_55382

theorem inequality_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l553_55382


namespace NUMINAMATH_CALUDE_farmers_market_spending_l553_55385

theorem farmers_market_spending (sandi_initial : ℕ) (gillian_total : ℕ) : 
  sandi_initial = 600 →
  gillian_total = 1050 →
  ∃ (multiple : ℕ), 
    gillian_total = (sandi_initial / 2) + (multiple * (sandi_initial / 2)) + 150 ∧
    multiple = 1 :=
by sorry

end NUMINAMATH_CALUDE_farmers_market_spending_l553_55385


namespace NUMINAMATH_CALUDE_apple_pricing_l553_55362

/-- The cost per kilogram for the first 30 kgs of apples -/
def l : ℚ := 200 / 10

/-- The cost per kilogram for each additional kilogram after the first 30 kgs -/
def m : ℚ := 21

theorem apple_pricing :
  (l * 30 + m * 3 = 663) ∧
  (l * 30 + m * 6 = 726) ∧
  (l * 10 = 200) →
  m = 21 := by sorry

end NUMINAMATH_CALUDE_apple_pricing_l553_55362


namespace NUMINAMATH_CALUDE_solution_to_system_l553_55360

theorem solution_to_system (x y : ℝ) : 
  (x^2*y - x*y^2 - 5*x + 5*y + 3 = 0 ∧ 
   x^3*y - x*y^3 - 5*x^2 + 5*y^2 + 15 = 0) ↔ 
  (x = 4 ∧ y = 1) := by sorry

end NUMINAMATH_CALUDE_solution_to_system_l553_55360


namespace NUMINAMATH_CALUDE_banana_price_is_five_l553_55344

/-- Represents the market problem with Peter's purchases --/
def market_problem (banana_price : ℝ) : Prop :=
  let initial_money : ℝ := 500
  let potato_kilos : ℝ := 6
  let potato_price : ℝ := 2
  let tomato_kilos : ℝ := 9
  let tomato_price : ℝ := 3
  let cucumber_kilos : ℝ := 5
  let cucumber_price : ℝ := 4
  let banana_kilos : ℝ := 3
  let remaining_money : ℝ := 426
  initial_money - (potato_kilos * potato_price + tomato_kilos * tomato_price + 
    cucumber_kilos * cucumber_price + banana_kilos * banana_price) = remaining_money

/-- Theorem stating that the price per kilo of bananas is $5 --/
theorem banana_price_is_five : 
  ∃ (banana_price : ℝ), market_problem banana_price ∧ banana_price = 5 :=
sorry

end NUMINAMATH_CALUDE_banana_price_is_five_l553_55344


namespace NUMINAMATH_CALUDE_runners_meet_closer_than_half_diagonal_l553_55389

/-- A point moving along a diagonal of a square -/
structure DiagonalRunner where
  position : ℝ  -- Position on the diagonal, normalized to [0, 1]
  direction : Bool  -- True if moving towards the endpoint, False if moving towards the start

/-- The state of two runners on diagonals of a square -/
structure SquareState where
  runner1 : DiagonalRunner
  runner2 : DiagonalRunner
  diagonal_length : ℝ

def distance (s : SquareState) : ℝ :=
  sorry

def update_state (s : SquareState) (t : ℝ) : SquareState :=
  sorry

theorem runners_meet_closer_than_half_diagonal
  (initial_state : SquareState)
  (h_positive_length : initial_state.diagonal_length > 0) :
  ∃ t : ℝ, distance (update_state initial_state t) < initial_state.diagonal_length / 2 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_closer_than_half_diagonal_l553_55389


namespace NUMINAMATH_CALUDE_paper_stack_height_l553_55311

/-- Given a ream of paper with 400 sheets that is 4 cm thick,
    prove that a stack of 6 cm will contain 600 sheets. -/
theorem paper_stack_height (sheets_per_ream : ℕ) (ream_thickness : ℝ) 
  (stack_height : ℝ) (h1 : sheets_per_ream = 400) (h2 : ream_thickness = 4) 
  (h3 : stack_height = 6) : 
  (stack_height / ream_thickness) * sheets_per_ream = 600 :=
sorry

end NUMINAMATH_CALUDE_paper_stack_height_l553_55311


namespace NUMINAMATH_CALUDE_f_always_positive_implies_m_greater_than_e_range_of_y_when_f_has_two_zeros_l553_55303

noncomputable section

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.exp x - x - 2

-- Theorem 1: If f(x) > 0 for all x in ℝ, then m > e
theorem f_always_positive_implies_m_greater_than_e (m : ℝ) :
  (∀ x : ℝ, f m x > 0) → m > Real.exp 1 := by sorry

-- Theorem 2: Range of y when f has two zeros
theorem range_of_y_when_f_has_two_zeros (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  f m x₁ = 0 →
  f m x₂ = 0 →
  let y := (Real.exp x₂ - Real.exp x₁) * (1 / (Real.exp x₂ + Real.exp x₁) - m)
  ∀ z : ℝ, z < 0 ∧ ∃ (t : ℝ), y = t := by sorry

end

end NUMINAMATH_CALUDE_f_always_positive_implies_m_greater_than_e_range_of_y_when_f_has_two_zeros_l553_55303


namespace NUMINAMATH_CALUDE_bridge_length_l553_55325

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 205 ∧
    bridge_length = train_speed_kmh * (1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l553_55325


namespace NUMINAMATH_CALUDE_probability_of_rolling_six_l553_55345

/-- The probability of rolling a total of 6 with two fair dice -/
theorem probability_of_rolling_six (dice : ℕ) (faces : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  dice = 2 →
  faces = 6 →
  total_outcomes = faces * faces →
  favorable_outcomes = 5 →
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_rolling_six_l553_55345


namespace NUMINAMATH_CALUDE_consecutive_square_differences_exist_l553_55387

theorem consecutive_square_differences_exist : 
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
    (a > 2022 ∨ b > 2022 ∨ c > 2022) ∧
    (∃ (k : ℤ), 
      (a^2 - b^2 = k) ∧ 
      (b^2 - c^2 = k + 1) ∧ 
      (c^2 - a^2 = k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_square_differences_exist_l553_55387


namespace NUMINAMATH_CALUDE_weight_ratio_proof_l553_55308

/-- Proves that the ratio of weight held in each hand to body weight is 1:1 --/
theorem weight_ratio_proof (body_weight hand_weight total_weight : ℝ) 
  (hw : body_weight = 150)
  (vest_weight : ℝ)
  (hv : vest_weight = body_weight / 2)
  (ht : total_weight = 525)
  (he : total_weight = body_weight + vest_weight + 2 * hand_weight) :
  hand_weight / body_weight = 1 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_proof_l553_55308


namespace NUMINAMATH_CALUDE_triangle_side_length_l553_55395

theorem triangle_side_length (a c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : c = 2) (h3 : A = π/6) :
  let b := Real.sqrt ((a^2 + c^2 - 2*a*c*(Real.cos A)) / (Real.sin A)^2)
  b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l553_55395


namespace NUMINAMATH_CALUDE_exponent_division_l553_55376

theorem exponent_division (a : ℝ) : a^7 / a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l553_55376


namespace NUMINAMATH_CALUDE_parabola_point_property_l553_55314

/-- Given a parabola y = a(x+3)^2 + c with two points (x₁, y₁) and (x₂, y₂),
    if |x₁+3| > |x₂+3|, then a(y₁-y₂) > 0 -/
theorem parabola_point_property (a c x₁ y₁ x₂ y₂ : ℝ) :
  y₁ = a * (x₁ + 3)^2 + c →
  y₂ = a * (x₂ + 3)^2 + c →
  |x₁ + 3| > |x₂ + 3| →
  a * (y₁ - y₂) > 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_property_l553_55314


namespace NUMINAMATH_CALUDE_matrix_power_100_l553_55399

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_100 : A^100 = !![1, 0; 200, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_100_l553_55399


namespace NUMINAMATH_CALUDE_bridge_length_l553_55309

/-- The length of a bridge given train parameters --/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_length + (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 215 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l553_55309


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l553_55305

/-- The function f(x) = x^2 + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Theorem stating that f(f(x)) has exactly 3 distinct real roots iff c = 1 - √13 -/
theorem f_comp_three_roots :
  ∀ c : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f_comp c r₁ = 0 ∧ f_comp c r₂ = 0 ∧ f_comp c r₃ = 0) ↔ 
  c = 1 - Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l553_55305


namespace NUMINAMATH_CALUDE_m_range_theorem_l553_55394

def prop_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

def m_range (m : ℝ) : Prop :=
  (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem m_range_theorem :
  ∀ m : ℝ, (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l553_55394


namespace NUMINAMATH_CALUDE_scientific_notation_of_80_million_l553_55349

theorem scientific_notation_of_80_million :
  ∃ (n : ℕ), 80000000 = 8 * (10 ^ n) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_80_million_l553_55349


namespace NUMINAMATH_CALUDE_c_completion_time_l553_55319

-- Define the work rates of A, B, and C
variable (A B C : ℝ)

-- Define the conditions
def condition1 : Prop := A + B = 1 / 15
def condition2 : Prop := A + B + C = 1 / 11

-- Theorem statement
theorem c_completion_time (h1 : condition1 A B) (h2 : condition2 A B C) :
  1 / C = 41.25 := by sorry

end NUMINAMATH_CALUDE_c_completion_time_l553_55319


namespace NUMINAMATH_CALUDE_blue_pill_cost_l553_55330

/-- Represents the cost of pills for Alice's medication --/
structure PillCosts where
  red : ℝ
  blue : ℝ
  yellow : ℝ

/-- The conditions of Alice's medication costs --/
def medication_conditions (costs : PillCosts) : Prop :=
  costs.blue = costs.red + 3 ∧
  costs.yellow = 2 * costs.red - 2 ∧
  21 * (costs.red + costs.blue + costs.yellow) = 924

/-- Theorem stating the cost of the blue pill --/
theorem blue_pill_cost (costs : PillCosts) :
  medication_conditions costs → costs.blue = 13.75 := by
  sorry


end NUMINAMATH_CALUDE_blue_pill_cost_l553_55330


namespace NUMINAMATH_CALUDE_cos420_plus_sin330_eq_zero_l553_55341

theorem cos420_plus_sin330_eq_zero :
  Real.cos (420 * π / 180) + Real.sin (330 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos420_plus_sin330_eq_zero_l553_55341


namespace NUMINAMATH_CALUDE_triangle_properties_l553_55371

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.B = Real.pi / 6 ∧
  t.b = Real.sqrt 7 ∧
  (1/2 * t.a * t.c * Real.sin t.B) = (15 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l553_55371


namespace NUMINAMATH_CALUDE_mistaken_divisor_l553_55315

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = 32 * correct_divisor →
  dividend = 56 * mistaken_divisor →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l553_55315


namespace NUMINAMATH_CALUDE_max_sum_reciprocals_l553_55335

theorem max_sum_reciprocals (k l m : ℕ+) (h : (k : ℝ)⁻¹ + (l : ℝ)⁻¹ + (m : ℝ)⁻¹ < 1) :
  ∃ (a b c : ℕ+), (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ = 41/42 ∧
    ∀ (x y z : ℕ+), (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ < 1 →
      (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ ≤ 41/42 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_reciprocals_l553_55335


namespace NUMINAMATH_CALUDE_dividend_rate_calculation_l553_55356

/-- Given a share worth 48 rupees, with a desired interest rate of 12%,
    and a market value of 36.00000000000001 rupees, the dividend rate is 16%. -/
theorem dividend_rate_calculation (share_value : ℝ) (interest_rate : ℝ) (market_value : ℝ)
    (h1 : share_value = 48)
    (h2 : interest_rate = 0.12)
    (h3 : market_value = 36.00000000000001) :
    (share_value * interest_rate / market_value) * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dividend_rate_calculation_l553_55356


namespace NUMINAMATH_CALUDE_chord_intersection_triangle_area_l553_55357

/-- Given two chords of a circle intersecting at a point, this theorem
    calculates the area of one triangle formed by the chords, given the
    area of the other triangle and the lengths of two segments. -/
theorem chord_intersection_triangle_area
  (PO SO : ℝ) (area_POR : ℝ) (h1 : PO = 3) (h2 : SO = 4) (h3 : area_POR = 7) :
  let area_QOS := (16 * area_POR) / (9 : ℝ)
  area_QOS = 112 / 9 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_triangle_area_l553_55357


namespace NUMINAMATH_CALUDE_line_equation_correct_l553_55306

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Check if a point (x, y) is on the line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y - l.point.2 = l.slope * (x - l.point.1)

/-- The specific line l with slope 2 passing through (2, -1) -/
def l : Line :=
  { slope := 2
  , point := (2, -1) }

/-- Theorem: The equation 2x - y - 5 = 0 represents the line l -/
theorem line_equation_correct :
  ∀ x y : ℝ, l.contains x y ↔ 2 * x - y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l553_55306


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l553_55334

theorem reciprocal_of_negative_2023 : 
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l553_55334


namespace NUMINAMATH_CALUDE_quadratic_points_theorem_l553_55347

/-- Quadratic function -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*m*x - 3

theorem quadratic_points_theorem (m n p q : ℝ) 
  (h_m : m > 0)
  (h_A : f m (n-2) = p)
  (h_B : f m 4 = q)
  (h_C : f m n = p)
  (h_q : -3 < q)
  (h_p : q < p) :
  (m = n - 1) ∧ ((3 < n ∧ n < 4) ∨ n > 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_points_theorem_l553_55347


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l553_55373

/-- The number of factors of 18000 that are perfect squares -/
def num_perfect_square_factors : ℕ := 8

/-- The prime factorization of 18000 -/
def factorization_18000 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 3)]

/-- Theorem: The number of factors of 18000 that are perfect squares is 8 -/
theorem count_perfect_square_factors :
  (List.prod (factorization_18000.map (fun (p, e) => e + 1)) / 8 : ℚ).num = num_perfect_square_factors := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l553_55373


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l553_55313

theorem real_part_of_complex_product : 
  let z : ℂ := (1 + Complex.I) * (1 + 2 * Complex.I)
  Complex.re z = -1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l553_55313


namespace NUMINAMATH_CALUDE_wire_length_problem_l553_55397

theorem wire_length_problem (total_wires : ℕ) (avg_length : ℝ) (third_avg_length : ℝ) :
  total_wires = 6 →
  avg_length = 80 →
  third_avg_length = 70 →
  let total_length := total_wires * avg_length
  let third_wires := total_wires / 3
  let third_total_length := third_wires * third_avg_length
  let remaining_wires := total_wires - third_wires
  let remaining_length := total_length - third_total_length
  remaining_length / remaining_wires = 85 := by
sorry

end NUMINAMATH_CALUDE_wire_length_problem_l553_55397


namespace NUMINAMATH_CALUDE_cars_meeting_time_l553_55398

theorem cars_meeting_time (distance : ℝ) (speed1 speed2 : ℝ) (h1 : distance = 60) 
  (h2 : speed1 = 13) (h3 : speed2 = 17) : 
  distance / (speed1 + speed2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l553_55398


namespace NUMINAMATH_CALUDE_final_result_l553_55384

def initial_value : ℕ := 10^8

def operation (n : ℕ) : ℕ := n * 3 / 2

def repeated_operation (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | k + 1 => repeated_operation (operation n) k

theorem final_result :
  repeated_operation initial_value 16 = 3^16 * 5^8 := by
  sorry

end NUMINAMATH_CALUDE_final_result_l553_55384


namespace NUMINAMATH_CALUDE_percentage_equality_l553_55379

theorem percentage_equality : (0.2 * 4 : ℝ) = (0.8 * 1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_percentage_equality_l553_55379


namespace NUMINAMATH_CALUDE_no_real_solutions_l553_55383

theorem no_real_solutions (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - 2| ≥ a) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l553_55383


namespace NUMINAMATH_CALUDE_cos_equality_implies_43_l553_55393

theorem cos_equality_implies_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_43_l553_55393


namespace NUMINAMATH_CALUDE_isosceles_right_triangles_are_similar_l553_55380

/-- An isosceles right triangle is a triangle with two equal sides and a right angle. -/
structure IsoscelesRightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right_angle : side1^2 + side2^2 = hypotenuse^2
  is_isosceles : side1 = side2

/-- Two triangles are similar if their corresponding angles are equal and the ratios of corresponding sides are equal. -/
def are_similar (t1 t2 : IsoscelesRightTriangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.hypotenuse = k * t2.hypotenuse

/-- Theorem: Any two isosceles right triangles are similar. -/
theorem isosceles_right_triangles_are_similar (t1 t2 : IsoscelesRightTriangle) : 
  are_similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangles_are_similar_l553_55380


namespace NUMINAMATH_CALUDE_uncertain_sum_l553_55332

theorem uncertain_sum (a b c : ℤ) (h : |a - b|^19 + |c - a|^95 = 1) :
  ∃ (x : ℤ), (x = 1 ∨ x = 2) ∧ |c - a| + |a - b| + |b - a| = x :=
sorry

end NUMINAMATH_CALUDE_uncertain_sum_l553_55332


namespace NUMINAMATH_CALUDE_hillary_climbing_rate_l553_55372

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  hillary_rate : ℝ
  eddy_rate : ℝ
  total_distance : ℝ
  hillary_stop_distance : ℝ
  hillary_descent_rate : ℝ
  total_time : ℝ

/-- The theorem stating that Hillary's climbing rate is 800 ft/hr -/
theorem hillary_climbing_rate 
  (scenario : ClimbingScenario)
  (h1 : scenario.total_distance = 5000)
  (h2 : scenario.eddy_rate = scenario.hillary_rate - 500)
  (h3 : scenario.hillary_stop_distance = 1000)
  (h4 : scenario.hillary_descent_rate = 1000)
  (h5 : scenario.total_time = 6)
  : scenario.hillary_rate = 800 := by
  sorry

end NUMINAMATH_CALUDE_hillary_climbing_rate_l553_55372


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l553_55369

theorem infinitely_many_non_representable : 
  Set.Infinite {n : ℤ | ∀ (a b c : ℕ), n ≠ 2^a + 3^b - 5^c} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l553_55369


namespace NUMINAMATH_CALUDE_matrix_subtraction_result_l553_55355

theorem matrix_subtraction_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 8]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 5; -3, 6]
  A - B = !![3, -8; 5, 2] := by sorry

end NUMINAMATH_CALUDE_matrix_subtraction_result_l553_55355


namespace NUMINAMATH_CALUDE_journey_distance_is_70_l553_55365

-- Define the journey parameters
def journey_time_at_40 : Real := 1.75
def journey_time_at_35 : Real := 2

-- Theorem statement
theorem journey_distance_is_70 :
  ∃ (distance : Real),
    distance = 40 * journey_time_at_40 ∧
    distance = 35 * journey_time_at_35 ∧
    distance = 70 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_is_70_l553_55365


namespace NUMINAMATH_CALUDE_inequality_not_true_l553_55352

theorem inequality_not_true (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ¬((2 * a * b) / (a + b) > Real.sqrt (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l553_55352


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l553_55301

theorem x_squared_less_than_abs_x_plus_two (x : ℝ) :
  x^2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_plus_two_l553_55301


namespace NUMINAMATH_CALUDE_aron_vacuuming_days_l553_55374

/-- The number of days Aron spends vacuuming each week -/
def vacuuming_days : ℕ := sorry

/-- The time spent vacuuming per day in minutes -/
def vacuuming_time_per_day : ℕ := 30

/-- The time spent dusting per day in minutes -/
def dusting_time_per_day : ℕ := 20

/-- The number of days Aron spends dusting each week -/
def dusting_days : ℕ := 2

/-- The total cleaning time per week in minutes -/
def total_cleaning_time : ℕ := 130

theorem aron_vacuuming_days :
  vacuuming_days * vacuuming_time_per_day +
  dusting_days * dusting_time_per_day =
  total_cleaning_time ∧
  vacuuming_days = 3 :=
by sorry

end NUMINAMATH_CALUDE_aron_vacuuming_days_l553_55374
