import Mathlib

namespace NUMINAMATH_CALUDE_caesars_meal_charge_is_30_l2160_216045

/-- Represents the charge per meal at Caesar's -/
def caesars_meal_charge : ℝ := sorry

/-- Caesar's room rental fee -/
def caesars_room_fee : ℝ := 800

/-- Venus Hall's room rental fee -/
def venus_room_fee : ℝ := 500

/-- Venus Hall's charge per meal -/
def venus_meal_charge : ℝ := 35

/-- Number of guests when the costs are equal -/
def num_guests : ℕ := 60

theorem caesars_meal_charge_is_30 :
  caesars_room_fee + num_guests * caesars_meal_charge =
  venus_room_fee + num_guests * venus_meal_charge →
  caesars_meal_charge = 30 := by sorry

end NUMINAMATH_CALUDE_caesars_meal_charge_is_30_l2160_216045


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2160_216073

theorem regular_polygon_exterior_angle (n : ℕ) (h : (n - 2) * 180 = 1800) :
  360 / n = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2160_216073


namespace NUMINAMATH_CALUDE_log_inequality_implies_greater_l2160_216069

theorem log_inequality_implies_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  Real.log a > Real.log b → a > b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_greater_l2160_216069


namespace NUMINAMATH_CALUDE_goods_train_passing_time_l2160_216029

/-- The time taken for a goods train to pass a man in an opposite moving train -/
theorem goods_train_passing_time
  (man_train_speed : ℝ)
  (goods_train_speed : ℝ)
  (goods_train_length : ℝ)
  (h1 : man_train_speed = 55)
  (h2 : goods_train_speed = 60.2)
  (h3 : goods_train_length = 320) :
  (goods_train_length / ((man_train_speed + goods_train_speed) * (1000 / 3600))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_goods_train_passing_time_l2160_216029


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l2160_216058

theorem quadratic_inequality_and_constraint (a b : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (a = 1 ∧ b = 2) ∧
  (∀ x y k, x > 0 → y > 0 → a / x + b / y = 1 → 
    (2 * x + y ≥ k^2 + k + 2) → 
    -3 ≤ k ∧ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l2160_216058


namespace NUMINAMATH_CALUDE_megan_seashell_count_l2160_216076

/-- The number of seashells Megan needs to add to her collection -/
def additional_shells : ℕ := 6

/-- The total number of seashells Megan wants in her collection -/
def target_shells : ℕ := 25

/-- Megan's current number of seashells -/
def current_shells : ℕ := target_shells - additional_shells

theorem megan_seashell_count : current_shells = 19 := by
  sorry

end NUMINAMATH_CALUDE_megan_seashell_count_l2160_216076


namespace NUMINAMATH_CALUDE_math_club_members_l2160_216082

/-- The number of female members in the Math club -/
def female_members : ℕ := 6

/-- The ratio of male to female members in the Math club -/
def male_to_female_ratio : ℕ := 2

/-- The total number of members in the Math club -/
def total_members : ℕ := female_members * (male_to_female_ratio + 1)

theorem math_club_members :
  total_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_math_club_members_l2160_216082


namespace NUMINAMATH_CALUDE_f_minimum_value_l2160_216057

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x^2 + 1/(x^2 + 1/x^2)

theorem f_minimum_value (x : ℝ) (h : x > 0) : 
  f x ≥ 2.5 ∧ f 1 = 2.5 := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2160_216057


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2160_216089

theorem quadratic_expression_value (x : ℝ) (h : x^2 + 3*x - 5 = 0) : 2*x^2 + 6*x - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2160_216089


namespace NUMINAMATH_CALUDE_preimage_of_neg_three_two_l2160_216037

def f (x y : ℝ) : ℝ × ℝ := (x * y, x + y)

theorem preimage_of_neg_three_two :
  {p : ℝ × ℝ | f p.1 p.2 = (-3, 2)} = {(3, -1), (-1, 3)} := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_neg_three_two_l2160_216037


namespace NUMINAMATH_CALUDE_treehouse_planks_l2160_216020

theorem treehouse_planks (total : ℕ) (storage_fraction : ℚ) (parents_fraction : ℚ) (friends : ℕ) 
  (h1 : total = 200)
  (h2 : storage_fraction = 1/4)
  (h3 : parents_fraction = 1/2)
  (h4 : friends = 20) :
  total - (↑total * storage_fraction).num - (↑total * parents_fraction).num - friends = 30 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_planks_l2160_216020


namespace NUMINAMATH_CALUDE_triangulation_count_l2160_216031

/-- A convex polygon with interior points and triangulation -/
structure ConvexPolygonWithTriangulation where
  n : ℕ  -- number of vertices in the polygon
  m : ℕ  -- number of interior points
  is_convex : Bool  -- the polygon is convex
  interior_points_are_vertices : Bool  -- each interior point is a vertex of at least one triangle
  vertices_among_given_points : Bool  -- vertices of triangles are among the n+m given points

/-- The number of triangles in the triangulation -/
def num_triangles (p : ConvexPolygonWithTriangulation) : ℕ := p.n + 2 * p.m - 2

/-- Theorem: The number of triangles in the triangulation is n + 2m - 2 -/
theorem triangulation_count (p : ConvexPolygonWithTriangulation) : 
  p.is_convex ∧ p.interior_points_are_vertices ∧ p.vertices_among_given_points →
  num_triangles p = p.n + 2 * p.m - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangulation_count_l2160_216031


namespace NUMINAMATH_CALUDE_volume_is_zero_l2160_216059

def S : Set (ℝ × ℝ) := {(x, y) | |6 - x| + y ≤ 8 ∧ 2*y - x ≥ 10}

def revolution_axis : Set (ℝ × ℝ) := {(x, y) | 2*y - x = 10}

def volume_of_revolution (region : Set (ℝ × ℝ)) (axis : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem volume_is_zero :
  volume_of_revolution S revolution_axis = 0 := by sorry

end NUMINAMATH_CALUDE_volume_is_zero_l2160_216059


namespace NUMINAMATH_CALUDE_atlanta_equals_boston_l2160_216010

/-- Two cyclists leave Cincinnati at the same time. One bikes to Boston, the other to Atlanta. -/
structure Cyclists where
  boston_distance : ℕ
  atlanta_distance : ℕ
  max_daily_distance : ℕ

/-- The conditions of the cycling problem -/
def cycling_problem (c : Cyclists) : Prop :=
  c.boston_distance = 840 ∧
  c.max_daily_distance = 40 ∧
  (c.boston_distance / c.max_daily_distance) * c.max_daily_distance = c.atlanta_distance

/-- The theorem stating that the distance to Atlanta is equal to the distance to Boston -/
theorem atlanta_equals_boston (c : Cyclists) (h : cycling_problem c) : 
  c.atlanta_distance = c.boston_distance :=
sorry

end NUMINAMATH_CALUDE_atlanta_equals_boston_l2160_216010


namespace NUMINAMATH_CALUDE_museum_artifacts_l2160_216060

theorem museum_artifacts (total_wings : Nat) 
  (painting_wings : Nat) (large_painting_wings : Nat) 
  (small_painting_wings : Nat) (paintings_per_small_wing : Nat) 
  (artifact_multiplier : Nat) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting_wings = 1 →
  small_painting_wings = 2 →
  paintings_per_small_wing = 12 →
  artifact_multiplier = 4 →
  let total_paintings := large_painting_wings + small_painting_wings * paintings_per_small_wing
  let total_artifacts := total_paintings * artifact_multiplier
  let artifact_wings := total_wings - painting_wings
  ∀ wing, wing ≤ artifact_wings → 
    (total_artifacts / artifact_wings : Nat) = 20 := by
  sorry

#check museum_artifacts

end NUMINAMATH_CALUDE_museum_artifacts_l2160_216060


namespace NUMINAMATH_CALUDE_highest_power_of_seven_in_square_of_factorial_l2160_216052

theorem highest_power_of_seven_in_square_of_factorial (n : ℕ) (h : n = 50) :
  (∃ k : ℕ, (7 : ℕ)^k ∣ (n! : ℕ)^2 ∧ ∀ m : ℕ, (7 : ℕ)^m ∣ (n! : ℕ)^2 → m ≤ k) →
  (∃ k : ℕ, k = 16 ∧ (7 : ℕ)^k ∣ (n! : ℕ)^2 ∧ ∀ m : ℕ, (7 : ℕ)^m ∣ (n! : ℕ)^2 → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_seven_in_square_of_factorial_l2160_216052


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2160_216011

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 - 13*x - 12
  (f 4 = 0) ∧ (f (-1) = 0) ∧ (f (-3) = 0) ∧
  (∀ x : ℝ, f x = 0 → (x = 4 ∨ x = -1 ∨ x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2160_216011


namespace NUMINAMATH_CALUDE_work_completion_time_l2160_216024

theorem work_completion_time (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (1 / x = 1 / 15) →
  (1 / x + 1 / y = 1 / 10) →
  y = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2160_216024


namespace NUMINAMATH_CALUDE_problem_solution_l2160_216005

def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : Set ℝ := {x | f x > -1}

theorem problem_solution :
  (M = Set.Ioo 0 2) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2160_216005


namespace NUMINAMATH_CALUDE_village_population_l2160_216077

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 80 / 100 →
  partial_population = 23040 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 28800 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2160_216077


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_find_value_l2160_216015

-- Question 1
theorem simplify_expression (a b : ℝ) :
  8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := by sorry

-- Question 2
theorem evaluate_expression (x y : ℝ) (h : x + y = 1/2) :
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := by sorry

-- Question 3
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  -3 * x^2 + 6 * y + 2 = -10 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_find_value_l2160_216015


namespace NUMINAMATH_CALUDE_weight_of_b_l2160_216032

theorem weight_of_b (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 60 →
  (a + b + c) / 3 = 55 →
  (b + c + d) / 3 = 58 →
  (c + d + e) / 3 = 62 →
  b = 114 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2160_216032


namespace NUMINAMATH_CALUDE_derivative_sin_2x_l2160_216083

theorem derivative_sin_2x (x : ℝ) : 
  deriv (fun x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_derivative_sin_2x_l2160_216083


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2160_216090

/-- Proves that a boat traveling 12 km downstream in 2 hours and 12 km upstream in 3 hours has a speed of 5 km/h in still water. -/
theorem boat_speed_in_still_water (downstream_distance : ℝ) (upstream_distance : ℝ)
  (downstream_time : ℝ) (upstream_time : ℝ) (h1 : downstream_distance = 12)
  (h2 : upstream_distance = 12) (h3 : downstream_time = 2) (h4 : upstream_time = 3) :
  ∃ (boat_speed : ℝ) (stream_speed : ℝ),
    boat_speed = 5 ∧
    downstream_distance / downstream_time = boat_speed + stream_speed ∧
    upstream_distance / upstream_time = boat_speed - stream_speed := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2160_216090


namespace NUMINAMATH_CALUDE_atomic_weight_Al_l2160_216075

/-- The atomic weight of oxygen -/
def atomic_weight_O : ℝ := 16

/-- The molecular weight of Al2O3 -/
def molecular_weight_Al2O3 : ℝ := 102

/-- The number of aluminum atoms in Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of oxygen atoms in Al2O3 -/
def num_O_atoms : ℕ := 3

/-- Theorem stating that the atomic weight of Al is 27 -/
theorem atomic_weight_Al :
  (molecular_weight_Al2O3 - num_O_atoms * atomic_weight_O) / num_Al_atoms = 27 := by
  sorry

end NUMINAMATH_CALUDE_atomic_weight_Al_l2160_216075


namespace NUMINAMATH_CALUDE_jayas_rank_from_bottom_l2160_216016

/-- Given a class of students, calculate the rank from the bottom based on the rank from the top -/
def rankFromBottom (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Theorem stating Jaya's rank from the bottom in a class of 53 students -/
theorem jayas_rank_from_bottom :
  let totalStudents : ℕ := 53
  let jayasRankFromTop : ℕ := 5
  rankFromBottom totalStudents jayasRankFromTop = 50 := by
  sorry


end NUMINAMATH_CALUDE_jayas_rank_from_bottom_l2160_216016


namespace NUMINAMATH_CALUDE_christina_weekly_distance_l2160_216078

/-- The total distance Christina covers in a week -/
def total_distance (school_distance : ℕ) (days : ℕ) (extra_distance : ℕ) : ℕ :=
  2 * school_distance * days + 2 * extra_distance

/-- Theorem stating the total distance Christina covered in a week -/
theorem christina_weekly_distance :
  total_distance 7 5 2 = 74 := by
  sorry

end NUMINAMATH_CALUDE_christina_weekly_distance_l2160_216078


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2160_216061

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: For an arithmetic sequence, if S_9 = 54 and S_8 - S_5 = 30, then S_11 = 88 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.S 9 = 54)
    (h2 : seq.S 8 - seq.S 5 = 30) :
    seq.S 11 = 88 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2160_216061


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l2160_216096

/-- Represents a triangular figure constructed with toothpicks -/
structure TriangularFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : TriangularFigure) : ℕ :=
  figure.upward_triangles

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_removal (figure : TriangularFigure) 
  (h1 : figure.total_toothpicks = 45)
  (h2 : figure.upward_triangles = 15)
  (h3 : figure.downward_triangles = 10) :
  min_toothpicks_to_remove figure = 15 := by
  sorry

#check min_toothpicks_removal

end NUMINAMATH_CALUDE_min_toothpicks_removal_l2160_216096


namespace NUMINAMATH_CALUDE_coefficient_x_10_in_expansion_l2160_216087

theorem coefficient_x_10_in_expansion : ∃ (c : ℤ), c = -11 ∧ 
  (∀ (x : ℝ), (x - 1)^11 = c * x^10 + (λ (y : ℝ) => (y - 1)^11 - c * y^10) x) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_10_in_expansion_l2160_216087


namespace NUMINAMATH_CALUDE_cube_root_of_four_l2160_216085

theorem cube_root_of_four (x : ℝ) : x^3 = 4 → x = 4^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_four_l2160_216085


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2160_216049

/-- Given a hyperbola with eccentricity √5 and one vertex at (1, 0), 
    prove that its equation is x^2 - y^2/4 = 1 -/
theorem hyperbola_equation (e : ℝ) (v : ℝ × ℝ) :
  e = Real.sqrt 5 →
  v = (1, 0) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 ↔ x^2 - y^2/4 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2160_216049


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2160_216099

theorem opposite_of_negative_2023 : 
  ∀ x : ℤ, x + (-2023) = 0 → x = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2160_216099


namespace NUMINAMATH_CALUDE_cos_4theta_l2160_216070

theorem cos_4theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 + Complex.I * Real.sqrt 7) / 4) : 
  Real.cos (4 * θ) = 1 / 32 := by
sorry

end NUMINAMATH_CALUDE_cos_4theta_l2160_216070


namespace NUMINAMATH_CALUDE_people_count_is_32_l2160_216013

/-- Given a room with chairs and people, calculate the number of people in the room. -/
def people_in_room (empty_chairs : ℕ) : ℕ :=
  let total_chairs := 3 * empty_chairs
  let seated_people := 2 * empty_chairs
  2 * seated_people

/-- Prove that the number of people in the room is 32, given the problem conditions. -/
theorem people_count_is_32 :
  let empty_chairs := 8
  let total_people := people_in_room empty_chairs
  let total_chairs := 3 * empty_chairs
  let seated_people := 2 * empty_chairs
  (2 * seated_people = total_people) ∧
  (seated_people = total_people / 2) ∧
  (seated_people = 2 * total_chairs / 3) ∧
  (total_people = 32) :=
by
  sorry

#eval people_in_room 8

end NUMINAMATH_CALUDE_people_count_is_32_l2160_216013


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2160_216039

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x^2 = 2*x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 2 ∧ x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2160_216039


namespace NUMINAMATH_CALUDE_paco_cookies_l2160_216034

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 19

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := 11

/-- The number of sweet cookies Paco ate in the first round -/
def sweet_cookies_eaten_first : ℕ := 5

/-- The number of salty cookies Paco ate in the first round -/
def salty_cookies_eaten_first : ℕ := 2

/-- The difference between sweet and salty cookies eaten in the second round -/
def sweet_salty_difference : ℕ := 3

theorem paco_cookies : 
  initial_sweet_cookies = 
    (initial_sweet_cookies - sweet_cookies_eaten_first - sweet_salty_difference) + 
    sweet_cookies_eaten_first + 
    (salty_cookies_eaten_first + sweet_salty_difference) :=
by sorry

end NUMINAMATH_CALUDE_paco_cookies_l2160_216034


namespace NUMINAMATH_CALUDE_tangent_secant_theorem_l2160_216043

-- Define the triangle ABC and point X
def Triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def RelativelyPrime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem tangent_secant_theorem
  (a b c : ℕ)
  (h_triangle : Triangle a b c)
  (h_coprime : RelativelyPrime b c) :
  ∃ (AX CX : ℚ),
    AX = (a * b * c : ℚ) / ((c * c - b * b) : ℚ) ∧
    CX = (a * b * b : ℚ) / ((c * c - b * b) : ℚ) ∧
    (¬ ∃ (n : ℤ), AX = n) ∧
    (¬ ∃ (n : ℤ), CX = n) :=
by sorry

end NUMINAMATH_CALUDE_tangent_secant_theorem_l2160_216043


namespace NUMINAMATH_CALUDE_det_scaled_columns_l2160_216056

variable {α : Type*} [LinearOrderedField α]

noncomputable def det (a b c : α × α × α) : α := sorry

theorem det_scaled_columns (a b c : α × α × α) :
  let D := det a b c
  det (3 • a) (2 • b) c = 6 * D :=
sorry

end NUMINAMATH_CALUDE_det_scaled_columns_l2160_216056


namespace NUMINAMATH_CALUDE_swim_club_members_l2160_216088

theorem swim_club_members :
  ∀ (total_members : ℕ) 
    (passed_test : ℕ) 
    (not_passed_with_course : ℕ) 
    (not_passed_without_course : ℕ),
  passed_test = (30 * total_members) / 100 →
  not_passed_with_course = 5 →
  not_passed_without_course = 30 →
  total_members = passed_test + not_passed_with_course + not_passed_without_course →
  total_members = 50 := by
sorry

end NUMINAMATH_CALUDE_swim_club_members_l2160_216088


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2160_216050

theorem cube_sum_theorem (x y k c : ℝ) (h1 : x^3 * y^3 = k) (h2 : 1 / x^3 + 1 / y^3 = c) :
  ∃ m : ℝ, m = x + y ∧ (x + y)^3 = c * k + 3 * (k^(1/3)) * m :=
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2160_216050


namespace NUMINAMATH_CALUDE_calculation_result_l2160_216026

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2160_216026


namespace NUMINAMATH_CALUDE_star_calculation_l2160_216017

def star (x y : ℝ) : ℝ := x^2 + y^2

theorem star_calculation : (star (star 3 5) 4) = 1172 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l2160_216017


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2160_216063

/-- Given a parabola x² = 2py with p > 0, if there exists a point on the parabola
    with ordinate l such that its distance to the focus is 3,
    then the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) (l : ℝ) (h1 : p > 0) :
  (∃ x : ℝ, x^2 = 2*p*l) →  -- point (x,l) is on the parabola
  (l + p/2)^2 + (p/2)^2 = 3^2 →  -- distance from (x,l) to focus (0,p/2) is 3
  p = 4 :=  -- distance from focus to directrix
by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2160_216063


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2160_216041

-- Define set A
def A : Set ℝ := {x | (x - 1) * (x - 4) < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = 2 - x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2160_216041


namespace NUMINAMATH_CALUDE_total_cards_l2160_216062

theorem total_cards (brenda janet mara : ℕ) : 
  janet = brenda + 9 →
  mara = 2 * janet →
  mara = 150 - 40 →
  brenda + janet + mara = 211 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l2160_216062


namespace NUMINAMATH_CALUDE_product_satisfies_X_l2160_216098

/-- Condition X: every positive integer less than m is a sum of distinct divisors of m -/
def condition_X (m : ℕ+) : Prop :=
  ∀ k < m, ∃ (S : Finset ℕ), (∀ d ∈ S, d ∣ m) ∧ (Finset.sum S id = k)

theorem product_satisfies_X (m n : ℕ+) (hm : condition_X m) (hn : condition_X n) :
  condition_X (m * n) :=
sorry

end NUMINAMATH_CALUDE_product_satisfies_X_l2160_216098


namespace NUMINAMATH_CALUDE_leaf_decrease_l2160_216086

theorem leaf_decrease (green_yesterday red_yesterday yellow_yesterday 
                       green_today yellow_today red_today : ℕ) :
  green_yesterday = red_yesterday →
  yellow_yesterday = 7 * red_yesterday →
  green_today = yellow_today →
  red_today = 7 * yellow_today →
  green_today + yellow_today + red_today ≤ (green_yesterday + red_yesterday + yellow_yesterday) / 4 :=
by sorry

end NUMINAMATH_CALUDE_leaf_decrease_l2160_216086


namespace NUMINAMATH_CALUDE_tim_weekly_reading_time_l2160_216091

/-- Tim's daily meditation time in hours -/
def daily_meditation_time : ℝ := 1

/-- Tim's daily reading time in hours -/
def daily_reading_time : ℝ := 2 * daily_meditation_time

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Tim's weekly reading time in hours -/
def weekly_reading_time : ℝ := daily_reading_time * days_in_week

theorem tim_weekly_reading_time :
  weekly_reading_time = 14 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_reading_time_l2160_216091


namespace NUMINAMATH_CALUDE_train_crossing_time_l2160_216021

/-- Time for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 150 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (5/18))) = 20 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2160_216021


namespace NUMINAMATH_CALUDE_distance_for_boy_problem_l2160_216095

/-- Calculates the distance covered given time in minutes and speed in meters per second -/
def distance_covered (time_minutes : ℕ) (speed_meters_per_second : ℕ) : ℕ :=
  time_minutes * 60 * speed_meters_per_second

/-- Proves that given 30 minutes and a speed of 1 meter per second, the distance covered is 1800 meters -/
theorem distance_for_boy_problem : distance_covered 30 1 = 1800 := by
  sorry

#eval distance_covered 30 1

end NUMINAMATH_CALUDE_distance_for_boy_problem_l2160_216095


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l2160_216074

theorem cubic_root_equation_solutions :
  ∀ x : ℝ, 
    (x^(1/3) - 4 / (x^(1/3) + 4) = 0) ↔ 
    (x = (-2 + 2 * Real.sqrt 2)^3 ∨ x = (-2 - 2 * Real.sqrt 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l2160_216074


namespace NUMINAMATH_CALUDE_linda_savings_l2160_216008

theorem linda_savings : ∃ S : ℚ,
  (5/8 : ℚ) * S + (1/4 : ℚ) * S + (1/8 : ℚ) * S = S ∧
  (1/4 : ℚ) * S = 400 ∧
  (1/8 : ℚ) * S = 600 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l2160_216008


namespace NUMINAMATH_CALUDE_bank_deposit_calculation_l2160_216036

/-- Calculates the total amount of principal and interest for a fixed deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate * years)

/-- Calculates the amount left after paying interest tax -/
def amountAfterTax (totalAmount : ℝ) (principal : ℝ) (taxRate : ℝ) : ℝ :=
  totalAmount - (totalAmount - principal) * taxRate

theorem bank_deposit_calculation :
  let principal : ℝ := 1000
  let rate : ℝ := 0.0225
  let years : ℝ := 1
  let taxRate : ℝ := 0.20
  let total := totalAmount principal rate years
  let afterTax := amountAfterTax total principal taxRate
  total = 1022.5 ∧ afterTax = 1018 := by sorry

end NUMINAMATH_CALUDE_bank_deposit_calculation_l2160_216036


namespace NUMINAMATH_CALUDE_f_composition_three_roots_l2160_216084

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

-- State the theorem
theorem f_composition_three_roots (c : ℝ) :
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    (∀ x : ℝ, f c (f c x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_composition_three_roots_l2160_216084


namespace NUMINAMATH_CALUDE_exponent_rule_problem_solution_l2160_216014

theorem exponent_rule (a : ℕ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

theorem problem_solution : 3000 * (3000^2999) = 3000^3000 := by
  have h1 : 3000 * (3000^2999) = 3000^1 * 3000^2999 := by sorry
  have h2 : 3000^1 * 3000^2999 = 3000^(1 + 2999) := by sorry
  have h3 : 1 + 2999 = 3000 := by sorry
  sorry

end NUMINAMATH_CALUDE_exponent_rule_problem_solution_l2160_216014


namespace NUMINAMATH_CALUDE_function_increasing_iff_a_in_range_range_of_a_for_increasing_function_l2160_216038

/-- The function f(x) = ax² - (a-1)x - 3 is increasing on [-1, +∞) if and only if a ∈ [0, 1/3] -/
theorem function_increasing_iff_a_in_range (a : ℝ) :
  (∀ x ≥ -1, ∀ y ≥ x, a * x^2 - (a - 1) * x - 3 ≤ a * y^2 - (a - 1) * y - 3) ↔
  0 ≤ a ∧ a ≤ 1/3 := by
  sorry

/-- The range of a for which f(x) = ax² - (a-1)x - 3 is increasing on [-1, +∞) is [0, 1/3] -/
theorem range_of_a_for_increasing_function :
  {a : ℝ | ∀ x ≥ -1, ∀ y ≥ x, a * x^2 - (a - 1) * x - 3 ≤ a * y^2 - (a - 1) * y - 3} =
  {a : ℝ | 0 ≤ a ∧ a ≤ 1/3} := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_iff_a_in_range_range_of_a_for_increasing_function_l2160_216038


namespace NUMINAMATH_CALUDE_paint_required_for_similar_statues_l2160_216068

/-- The amount of paint required for similar statues with different heights and thicknesses -/
theorem paint_required_for_similar_statues 
  (original_height : ℝ) 
  (original_paint : ℝ) 
  (new_height : ℝ) 
  (num_statues : ℕ) 
  (thickness_factor : ℝ)
  (h1 : original_height > 0)
  (h2 : original_paint > 0)
  (h3 : new_height > 0)
  (h4 : thickness_factor > 0) :
  let surface_area_ratio := (new_height / original_height) ^ 2
  let paint_per_new_statue := original_paint * surface_area_ratio * thickness_factor
  let total_paint := paint_per_new_statue * num_statues
  total_paint = 28.8 :=
by
  sorry

#check paint_required_for_similar_statues 10 1 2 360 2

end NUMINAMATH_CALUDE_paint_required_for_similar_statues_l2160_216068


namespace NUMINAMATH_CALUDE_parabola_sum_l2160_216000

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 7), vertical axis of symmetry, and containing the point (0, 4) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 7
  point_x : ℝ := 0
  point_y : ℝ := 4
  eq_at_point : p * point_x^2 + q * point_x + r = point_y
  vertex_form : ∀ x y, y = p * (x - vertex_x)^2 + vertex_y

theorem parabola_sum (par : Parabola) : par.p + par.q + par.r = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l2160_216000


namespace NUMINAMATH_CALUDE_barbaras_selling_price_l2160_216080

/-- Proves that Barbara's selling price for each stuffed animal is $2 --/
theorem barbaras_selling_price : 
  ∀ (barbara_price : ℚ),
  (9 : ℚ) * barbara_price + (2 * 9 : ℚ) * (3/2 : ℚ) = 45 →
  barbara_price = 2 := by
sorry

end NUMINAMATH_CALUDE_barbaras_selling_price_l2160_216080


namespace NUMINAMATH_CALUDE_problem_statement_l2160_216067

noncomputable section

def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

theorem problem_statement :
  (∀ m > -1 - Real.sqrt 2, ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
    x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ f x₁ + g x₂ < m) ∧
  (∀ x > -1, f x - g x > 0) := by
  sorry

end

end NUMINAMATH_CALUDE_problem_statement_l2160_216067


namespace NUMINAMATH_CALUDE_unbounded_function_l2160_216066

def IsUnbounded (f : ℝ → ℝ) : Prop :=
  ∀ M : ℝ, ∃ x : ℝ, f x > M

theorem unbounded_function (f : ℝ → ℝ) 
  (h_pos : ∀ x, 0 < f x) 
  (h_ineq : ∀ x y, 0 < x → 0 < y → (f (x + f y))^2 ≥ f x * (f (x + f y) + f y)) : 
  IsUnbounded f := by
  sorry

end NUMINAMATH_CALUDE_unbounded_function_l2160_216066


namespace NUMINAMATH_CALUDE_goldfish_equality_month_l2160_216033

theorem goldfish_equality_month : ∃ (n : ℕ), n > 0 ∧ 3 * 5^n = 243 * 3^n ∧ ∀ (m : ℕ), m > 0 → m < n → 3 * 5^m ≠ 243 * 3^m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_month_l2160_216033


namespace NUMINAMATH_CALUDE_adjacent_sums_odd_in_circular_arrangement_l2160_216019

/-- A circular arrangement of 2020 natural numbers -/
def CircularArrangement := Fin 2020 → ℕ

/-- The property that the sum of any two adjacent numbers in the arrangement is odd -/
def AdjacentSumsOdd (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2020, Odd ((arr i) + (arr (i + 1)))

/-- Theorem stating that for any circular arrangement of 2020 natural numbers,
    the sum of any two adjacent numbers is odd -/
theorem adjacent_sums_odd_in_circular_arrangement :
  ∀ arr : CircularArrangement, AdjacentSumsOdd arr :=
sorry

end NUMINAMATH_CALUDE_adjacent_sums_odd_in_circular_arrangement_l2160_216019


namespace NUMINAMATH_CALUDE_digit_37_is_1_l2160_216035

/-- The decimal representation of 1/7 has a repeating cycle of 6 digits: 142857 -/
def decimal_rep_of_one_seventh : List Nat := [1, 4, 2, 8, 5, 7]

/-- The length of the repeating cycle in the decimal representation of 1/7 -/
def cycle_length : Nat := decimal_rep_of_one_seventh.length

/-- The 37th digit after the decimal point in the decimal representation of 1/7 -/
def digit_37 : Nat := decimal_rep_of_one_seventh[(37 - 1) % cycle_length]

theorem digit_37_is_1 : digit_37 = 1 := by sorry

end NUMINAMATH_CALUDE_digit_37_is_1_l2160_216035


namespace NUMINAMATH_CALUDE_orange_shells_count_l2160_216006

theorem orange_shells_count 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (blue_shells : ℕ)
  (h1 : total_shells = 65)
  (h2 : purple_shells = 13)
  (h3 : pink_shells = 8)
  (h4 : yellow_shells = 18)
  (h5 : blue_shells = 12)
  (h6 : (total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)) * 100 = total_shells * 35) :
  total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells) = 14 := by
sorry

end NUMINAMATH_CALUDE_orange_shells_count_l2160_216006


namespace NUMINAMATH_CALUDE_maurice_earnings_l2160_216055

/-- Calculates the total earnings for a given number of tasks -/
def totalEarnings (tasksCompleted : ℕ) (earnPerTask : ℕ) (bonusPerTenTasks : ℕ) : ℕ :=
  let regularEarnings := tasksCompleted * earnPerTask
  let bonusEarnings := (tasksCompleted / 10) * bonusPerTenTasks
  regularEarnings + bonusEarnings

/-- Proves that Maurice's earnings for 30 tasks is $78 -/
theorem maurice_earnings : totalEarnings 30 2 6 = 78 := by
  sorry

end NUMINAMATH_CALUDE_maurice_earnings_l2160_216055


namespace NUMINAMATH_CALUDE_odd_function_domain_symmetry_l2160_216003

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_domain_symmetry
  (f : ℝ → ℝ) (t : ℝ)
  (h_odd : is_odd_function f)
  (h_domain : Set.Ioo t (2*t + 3) = {x | f x ≠ 0}) :
  t = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_domain_symmetry_l2160_216003


namespace NUMINAMATH_CALUDE_original_denominator_proof_l2160_216040

theorem original_denominator_proof (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 → (3 + 8 : ℚ) / (d + 8) = 1 / 3 → d = 25 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l2160_216040


namespace NUMINAMATH_CALUDE_domino_arrangements_4_5_l2160_216094

/-- The number of distinct arrangements for placing dominoes on a grid. -/
def dominoArrangements (m n k : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

/-- Theorem: The number of distinct arrangements for placing 4 dominoes on a 4 by 5 grid,
    moving only right or down from upper left to lower right corner, is 35. -/
theorem domino_arrangements_4_5 :
  dominoArrangements 4 5 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_domino_arrangements_4_5_l2160_216094


namespace NUMINAMATH_CALUDE_system_solution_l2160_216053

theorem system_solution (x y z t : ℝ) : 
  (x = (1/2) * (y + 1/y) ∧
   y = (1/2) * (z + 1/z) ∧
   z = (1/2) * (t + 1/t) ∧
   t = (1/2) * (x + 1/x)) →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1 ∧ t = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2160_216053


namespace NUMINAMATH_CALUDE_min_distance_PM_l2160_216071

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define a point P on l₁
structure Point_P where
  x : ℝ
  y : ℝ
  on_l₁ : l₁ x y

-- Define a line l₂ passing through P
structure Line_l₂ (P : Point_P) where
  slope : ℝ
  passes_through_P : True  -- This is a simplification, as we don't need the specific equation

-- Define the intersection point M
structure Point_M (P : Point_P) (l₂ : Line_l₂ P) where
  x : ℝ
  y : ℝ
  on_C : C x y
  on_l₂ : True  -- This is a simplification, as we don't need the specific condition

-- State the theorem
theorem min_distance_PM (P : Point_P) (l₂ : Line_l₂ P) (M : Point_M P l₂) :
  ∃ (d : ℝ), d = 4 ∧ ∀ (M' : Point_M P l₂), Real.sqrt ((M'.x - P.x)^2 + (M'.y - P.y)^2) ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_PM_l2160_216071


namespace NUMINAMATH_CALUDE_five_star_three_eq_four_l2160_216048

/-- The star operation for integers -/
def star (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

/-- Theorem: 5 star 3 equals 4 -/
theorem five_star_three_eq_four : star 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_star_three_eq_four_l2160_216048


namespace NUMINAMATH_CALUDE_equation_solution_l2160_216044

theorem equation_solution (a : ℝ) : (2 * a * 1 - 2 = a + 3) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2160_216044


namespace NUMINAMATH_CALUDE_no_odd_prime_sum_107_l2160_216001

theorem no_odd_prime_sum_107 : ¬∃ (p q k : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Odd p ∧ 
  Odd q ∧ 
  p + q = 107 ∧ 
  p * q = k :=
sorry

end NUMINAMATH_CALUDE_no_odd_prime_sum_107_l2160_216001


namespace NUMINAMATH_CALUDE_paulas_shopping_problem_l2160_216007

/-- Paula's shopping problem -/
theorem paulas_shopping_problem (initial_amount : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 109 →
  shirt_cost = 11 →
  pants_cost = 13 →
  remaining_amount = 74 →
  ∃ (num_shirts : ℕ), num_shirts * shirt_cost + pants_cost = initial_amount - remaining_amount ∧ num_shirts = 2 :=
by sorry

end NUMINAMATH_CALUDE_paulas_shopping_problem_l2160_216007


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_distances_existence_of_minimum_l2160_216042

theorem min_value_of_sum_of_distances (x : ℝ) :
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
sorry

theorem existence_of_minimum (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) < 2 * Real.sqrt 5 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_distances_existence_of_minimum_l2160_216042


namespace NUMINAMATH_CALUDE_circle_passes_fixed_point_circle_tangent_condition_l2160_216047

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*(a - 1) = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (4, -2)

-- Define the second circle
def second_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Theorem 1: The circle passes through the fixed point for all real a
theorem circle_passes_fixed_point :
  ∀ a : ℝ, circle_equation fixed_point.1 fixed_point.2 a :=
sorry

-- Theorem 2: The circle is tangent to the second circle iff a = 1 - √5 or a = 1 + √5
theorem circle_tangent_condition :
  ∀ a : ℝ, (∃ x y : ℝ, circle_equation x y a ∧ second_circle x y ∧
    (∀ x' y' : ℝ, circle_equation x' y' a ∧ second_circle x' y' → (x = x' ∧ y = y'))) ↔
    (a = 1 - Real.sqrt 5 ∨ a = 1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_circle_passes_fixed_point_circle_tangent_condition_l2160_216047


namespace NUMINAMATH_CALUDE_henry_lawn_mowing_earnings_l2160_216051

/-- Henry's lawn mowing earnings problem -/
theorem henry_lawn_mowing_earnings 
  (earnings_per_lawn : ℕ) 
  (total_lawns : ℕ) 
  (forgotten_lawns : ℕ) 
  (h1 : earnings_per_lawn = 5)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 7) :
  (total_lawns - forgotten_lawns) * earnings_per_lawn = 25 := by
  sorry

end NUMINAMATH_CALUDE_henry_lawn_mowing_earnings_l2160_216051


namespace NUMINAMATH_CALUDE_exists_non_prime_power_plus_a_l2160_216028

theorem exists_non_prime_power_plus_a (a : ℕ) (ha : a > 1) :
  ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_prime_power_plus_a_l2160_216028


namespace NUMINAMATH_CALUDE_smallest_three_digit_integer_l2160_216072

theorem smallest_three_digit_integer (n : ℕ) : 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (45 * n ≡ 135 [MOD 280]) ∧ 
  (n ≡ 3 [MOD 7]) →
  n ≥ 115 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_integer_l2160_216072


namespace NUMINAMATH_CALUDE_smallest_base_for_82_five_satisfies_condition_five_is_smallest_base_l2160_216065

theorem smallest_base_for_82 : 
  ∀ b : ℕ, b > 0 → (b^2 ≤ 82 ∧ 82 < b^3) → b ≥ 5 :=
by
  sorry

theorem five_satisfies_condition : 
  5^2 ≤ 82 ∧ 82 < 5^3 :=
by
  sorry

theorem five_is_smallest_base : 
  ∀ b : ℕ, b > 0 → b^2 ≤ 82 ∧ 82 < b^3 → b = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_82_five_satisfies_condition_five_is_smallest_base_l2160_216065


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l2160_216018

theorem quadratic_form_minimum (x y : ℝ) : 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10 ≥ 14/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l2160_216018


namespace NUMINAMATH_CALUDE_ellipse_max_angle_ratio_l2160_216079

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y + 10 = 0

-- Define the angle F₁PF₂
def angle_F₁PF₂ (P : ℝ × ℝ) : ℝ := sorry

-- Define the ratio PF₁/PF₂
def ratio_PF₁_PF₂ (P : ℝ × ℝ) : ℝ := sorry

theorem ellipse_max_angle_ratio :
  ∀ a b : ℝ, a > 0 → b > 0 →
  ∀ P : ℝ × ℝ,
  ellipse a b P.1 P.2 →
  line_l P.1 P.2 →
  (∀ Q : ℝ × ℝ, ellipse a b Q.1 Q.2 → line_l Q.1 Q.2 → angle_F₁PF₂ P ≥ angle_F₁PF₂ Q) →
  ratio_PF₁_PF₂ P = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_max_angle_ratio_l2160_216079


namespace NUMINAMATH_CALUDE_whiteboard_washing_time_l2160_216023

/-- If four kids can wash three whiteboards in 20 minutes, then one kid can wash six whiteboards in 160 minutes. -/
theorem whiteboard_washing_time 
  (time_four_kids : ℕ) 
  (num_whiteboards_four_kids : ℕ) 
  (num_kids : ℕ) 
  (num_whiteboards_one_kid : ℕ) :
  time_four_kids = 20 ∧ 
  num_whiteboards_four_kids = 3 ∧ 
  num_kids = 4 ∧ 
  num_whiteboards_one_kid = 6 →
  (time_four_kids * num_kids * num_whiteboards_one_kid) / num_whiteboards_four_kids = 160 := by
  sorry

#check whiteboard_washing_time

end NUMINAMATH_CALUDE_whiteboard_washing_time_l2160_216023


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l2160_216004

/-- Theorem: Surface Area of a Rectangular Box
Given a rectangular box with dimensions a, b, and c, if the sum of the lengths of its twelve edges
is 180 and the distance from one corner to the farthest corner is 25, then its total surface area
is 1400. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : 4 * a + 4 * b + 4 * c = 180)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25) :
  2 * (a * b + b * c + a * c) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l2160_216004


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2160_216093

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 22 ∧
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 1 → x + 2*y ≤ M) ∧
  (∃ x y : ℝ, 2 * x^2 + 3 * y^2 = 1 ∧ x + 2*y = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2160_216093


namespace NUMINAMATH_CALUDE_abs_opposite_of_one_l2160_216009

theorem abs_opposite_of_one (x : ℝ) (h : x = -1) : |x| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_opposite_of_one_l2160_216009


namespace NUMINAMATH_CALUDE_area_ratio_in_equally_divided_perimeter_l2160_216025

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- A point on the perimeter of a triangle -/
structure PerimeterPoint (t : Triangle) where
  point : ℝ × ℝ
  on_perimeter : sorry

/-- Theorem: Area ratio in a triangle with equally divided perimeter -/
theorem area_ratio_in_equally_divided_perimeter (ABC : Triangle) 
  (P Q R : PerimeterPoint ABC) : 
  perimeter ABC = 1 →
  P.point.1 < Q.point.1 →
  (P.point.1 - ABC.A.1) + (Q.point.1 - P.point.1) + 
    (perimeter ABC - (Q.point.1 - ABC.A.1)) = perimeter ABC →
  let PQR : Triangle := ⟨P.point, Q.point, R.point⟩
  area PQR / area ABC > 2/9 := by sorry

end NUMINAMATH_CALUDE_area_ratio_in_equally_divided_perimeter_l2160_216025


namespace NUMINAMATH_CALUDE_total_money_divided_l2160_216027

/-- Proves that the total amount of money divided among three persons is 116000,
    given their share ratios and the amount for one person. -/
theorem total_money_divided (share_a share_b share_c : ℝ) : 
  share_a = 29491.525423728814 →
  share_a / share_b = 3 / 4 →
  share_b / share_c = 5 / 6 →
  share_a + share_b + share_c = 116000 := by
sorry

end NUMINAMATH_CALUDE_total_money_divided_l2160_216027


namespace NUMINAMATH_CALUDE_probability_multiple_four_l2160_216054

-- Define the types for the dice
def DodecahedralDie := Fin 12
def SixSidedDie := Fin 6

-- Define the probability space
def Ω := DodecahedralDie × SixSidedDie

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define the event that the product is a multiple of 4
def MultipleFour : Set Ω := {ω | 4 ∣ (ω.1.val + 1) * (ω.2.val + 1)}

-- Theorem statement
theorem probability_multiple_four : P MultipleFour = 3/8 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_four_l2160_216054


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2160_216046

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * p^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p)^3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l2160_216046


namespace NUMINAMATH_CALUDE_line_equation_proof_l2160_216022

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The projection of one point onto a line -/
def Point.projection (p : Point) (l : Line) : Point :=
  sorry

theorem line_equation_proof (A : Point) (P : Point) (l : Line) :
  A.x = 1 ∧ A.y = 2 ∧ P.x = -1 ∧ P.y = 4 ∧ P = A.projection l →
  l.a = 1 ∧ l.b = -1 ∧ l.c = 5 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2160_216022


namespace NUMINAMATH_CALUDE_quartic_polynomial_value_l2160_216002

/-- A quartic polynomial with specific properties -/
def QuarticPolynomial (P : ℝ → ℝ) : Prop :=
  (∃ a b c d e : ℝ, ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e) ∧ 
  P 1 = 0 ∧
  (∀ x, P x ≤ 3) ∧
  P 2 = 3 ∧
  P 3 = 3

/-- The main theorem -/
theorem quartic_polynomial_value (P : ℝ → ℝ) (h : QuarticPolynomial P) : P 5 = -24 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_value_l2160_216002


namespace NUMINAMATH_CALUDE_vector_equation_l2160_216097

variable {V : Type*} [AddCommGroup V]

theorem vector_equation (O A B C : V) :
  (A - O) - (B - O) + (C - A) = C - B := by sorry

end NUMINAMATH_CALUDE_vector_equation_l2160_216097


namespace NUMINAMATH_CALUDE_rocking_chair_legs_count_l2160_216081

/-- Represents the number of legs on the rocking chair -/
def rocking_chair_legs : ℕ := 2

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 40

/-- Represents the number of four-legged tables -/
def four_leg_tables : ℕ := 4

/-- Represents the number of sofas -/
def sofas : ℕ := 1

/-- Represents the number of four-legged chairs -/
def four_leg_chairs : ℕ := 2

/-- Represents the number of three-legged tables -/
def three_leg_tables : ℕ := 3

/-- Represents the number of one-legged tables -/
def one_leg_tables : ℕ := 1

theorem rocking_chair_legs_count : 
  rocking_chair_legs = 
    total_legs - 
    (4 * four_leg_tables + 
     4 * sofas + 
     4 * four_leg_chairs + 
     3 * three_leg_tables + 
     1 * one_leg_tables) :=
by sorry

end NUMINAMATH_CALUDE_rocking_chair_legs_count_l2160_216081


namespace NUMINAMATH_CALUDE_pushup_progression_l2160_216012

/-- 
Given a person who does push-ups 3 times a week, increasing by 5 each time,
prove that if the total for the week is 45, then the number of push-ups on the first day is 10.
-/
theorem pushup_progression (first_day : ℕ) : 
  first_day + (first_day + 5) + (first_day + 10) = 45 → first_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_pushup_progression_l2160_216012


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l2160_216030

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 3 * x + 2

-- Define the solution set
def solution_set (a b : ℝ) : Set ℝ := {x | b < x ∧ x < 1}

-- Theorem statement
theorem quadratic_solution_set (a b : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a b) →
  a = -5 ∧ b = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l2160_216030


namespace NUMINAMATH_CALUDE_triple_addichiffrer_1998_power_l2160_216092

/-- The addichiffrer function adds all digits of a natural number. -/
def addichiffrer (n : ℕ) : ℕ := sorry

/-- Apply addichiffrer process three times to a given number. -/
def triple_addichiffrer (n : ℕ) : ℕ := 
  addichiffrer (addichiffrer (addichiffrer n))

/-- Theorem stating that applying addichiffrer three times to 1998^1998 results in 9. -/
theorem triple_addichiffrer_1998_power : triple_addichiffrer (1998^1998) = 9 := by sorry

end NUMINAMATH_CALUDE_triple_addichiffrer_1998_power_l2160_216092


namespace NUMINAMATH_CALUDE_range_of_f_l2160_216064

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x - 6

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -11 ≤ y ∧ y ≤ -2 } := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2160_216064
