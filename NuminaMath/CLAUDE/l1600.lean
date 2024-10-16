import Mathlib

namespace NUMINAMATH_CALUDE_savings_duration_l1600_160095

/-- Proves that saving $34 daily for a total of $12,410 results in 365 days of savings -/
theorem savings_duration (daily_savings : ℕ) (total_savings : ℕ) (days : ℕ) :
  daily_savings = 34 →
  total_savings = 12410 →
  total_savings = daily_savings * days →
  days = 365 := by
sorry

end NUMINAMATH_CALUDE_savings_duration_l1600_160095


namespace NUMINAMATH_CALUDE_circle_condition_tangent_lines_perpendicular_intersection_l1600_160020

-- Define the equation of circle C
def C (x y a : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + a = 0

-- Define the line l
def l (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem 1: C represents a circle iff a ∈ (-∞, 8)
theorem circle_condition (a : ℝ) : 
  (∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ), C x y a ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔ 
  a < 8 :=
sorry

-- Theorem 2: Tangent lines when a = -17
theorem tangent_lines : 
  (∀ (x y : ℝ), C x y (-17) → (39*x + 80*y - 207 = 0 ∨ x = 7)) ∧
  C 7 (-6) (-17) ∧
  (∃ (x y : ℝ), C x y (-17) ∧ 39*x + 80*y - 207 = 0) ∧
  (∃ (y : ℝ), C 7 y (-17)) :=
sorry

-- Theorem 3: Value of a when OA ⊥ OB
theorem perpendicular_intersection :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    C x₁ y₁ (-6/5) ∧ C x₂ y₂ (-6/5) ∧
    l x₁ y₁ ∧ l x₂ y₂ ∧
    x₁ * x₂ + y₁ * y₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_tangent_lines_perpendicular_intersection_l1600_160020


namespace NUMINAMATH_CALUDE_janet_waiting_time_l1600_160041

/-- Proves that Janet waits 3 hours for her sister to cross the lake -/
theorem janet_waiting_time (lake_width : ℝ) (janet_speed : ℝ) (sister_speed : ℝ) :
  lake_width = 60 →
  janet_speed = 30 →
  sister_speed = 12 →
  (lake_width / sister_speed) - (lake_width / janet_speed) = 3 := by
  sorry

end NUMINAMATH_CALUDE_janet_waiting_time_l1600_160041


namespace NUMINAMATH_CALUDE_equation_proof_l1600_160048

theorem equation_proof : 144 + 2 * 12 * 7 + 49 = 361 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1600_160048


namespace NUMINAMATH_CALUDE_gcd_459_357_l1600_160000

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1600_160000


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1600_160080

theorem trigonometric_identities :
  (Real.tan (25 * π / 180) + Real.tan (20 * π / 180) + Real.tan (25 * π / 180) * Real.tan (20 * π / 180) = 1) ∧
  (1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1600_160080


namespace NUMINAMATH_CALUDE_count_D_eq_2_is_30_l1600_160017

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 127 for which D(n) = 2 -/
def count_D_eq_2 : ℕ := sorry

theorem count_D_eq_2_is_30 : count_D_eq_2 = 30 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_2_is_30_l1600_160017


namespace NUMINAMATH_CALUDE_courtyard_width_l1600_160013

theorem courtyard_width (length : ℝ) (num_bricks : ℕ) (brick_length brick_width : ℝ) :
  length = 25 ∧ 
  num_bricks = 20000 ∧ 
  brick_length = 0.2 ∧ 
  brick_width = 0.1 →
  (num_bricks : ℝ) * brick_length * brick_width / length = 16 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_width_l1600_160013


namespace NUMINAMATH_CALUDE_lindas_mean_score_l1600_160077

def scores : List ℕ := [80, 86, 90, 92, 95, 97]

def jakes_mean : ℕ := 89

theorem lindas_mean_score (h1 : scores.length = 6)
  (h2 : ∃ (jake_scores linda_scores : List ℕ),
    jake_scores.length = 3 ∧
    linda_scores.length = 3 ∧
    jake_scores ++ linda_scores = scores)
  (h3 : ∃ (jake_scores : List ℕ),
    jake_scores.length = 3 ∧
    jake_scores.sum / jake_scores.length = jakes_mean) :
  ∃ (linda_scores : List ℕ),
    linda_scores.length = 3 ∧
    linda_scores.sum / linda_scores.length = 91 :=
by sorry

end NUMINAMATH_CALUDE_lindas_mean_score_l1600_160077


namespace NUMINAMATH_CALUDE_square_of_complex_number_l1600_160045

theorem square_of_complex_number :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l1600_160045


namespace NUMINAMATH_CALUDE_response_rate_percentage_l1600_160069

theorem response_rate_percentage : 
  ∀ (responses_needed : ℕ) (questionnaires_mailed : ℕ),
  responses_needed = 210 →
  questionnaires_mailed = 350 →
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_response_rate_percentage_l1600_160069


namespace NUMINAMATH_CALUDE_weeks_of_feed_l1600_160030

-- Define the given quantities
def boxes_bought : ℕ := 3
def boxes_in_pantry : ℕ := 5
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Calculate total boxes and total grams
def total_boxes : ℕ := boxes_bought + boxes_in_pantry
def total_grams : ℕ := total_boxes * grams_per_box

-- Calculate weekly consumption
def weekly_consumption : ℕ := parrot_consumption + cockatiel_consumption

-- Theorem to prove
theorem weeks_of_feed : (total_grams / weekly_consumption : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_weeks_of_feed_l1600_160030


namespace NUMINAMATH_CALUDE_shaded_area_in_squares_l1600_160062

/-- The area of the shaded region in a specific geometric configuration -/
theorem shaded_area_in_squares : 
  let small_square_side : ℝ := 4
  let large_square_side : ℝ := 12
  let rectangle_width : ℝ := 2
  let rectangle_height : ℝ := 4
  let total_width : ℝ := small_square_side + large_square_side
  let triangle_height : ℝ := (small_square_side * small_square_side) / total_width
  let triangle_area : ℝ := (1 / 2) * triangle_height * small_square_side
  let small_square_area : ℝ := small_square_side * small_square_side
  let shaded_area : ℝ := small_square_area - triangle_area
  shaded_area = 14 := by
    sorry

end NUMINAMATH_CALUDE_shaded_area_in_squares_l1600_160062


namespace NUMINAMATH_CALUDE_journey_distance_l1600_160053

/-- Given a constant speed, if a journey of 120 miles takes 3 hours, 
    then a journey of 5 hours at the same speed covers a distance of 200 miles. -/
theorem journey_distance (speed : ℝ) 
  (h1 : speed * 3 = 120) 
  (h2 : speed > 0) : 
  speed * 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1600_160053


namespace NUMINAMATH_CALUDE_annes_cleaning_time_l1600_160044

/-- Represents the time it takes Anne to clean the house individually -/
def annes_individual_time (bruce_rate anne_rate : ℚ) : ℚ :=
  1 / anne_rate

/-- The condition that Bruce and Anne can clean the house in 4 hours together -/
def condition1 (bruce_rate anne_rate : ℚ) : Prop :=
  bruce_rate + anne_rate = 1 / 4

/-- The condition that Bruce and Anne with Anne's doubled speed can clean the house in 3 hours -/
def condition2 (bruce_rate anne_rate : ℚ) : Prop :=
  bruce_rate + 2 * anne_rate = 1 / 3

theorem annes_cleaning_time 
  (bruce_rate anne_rate : ℚ) 
  (h1 : condition1 bruce_rate anne_rate) 
  (h2 : condition2 bruce_rate anne_rate) :
  annes_individual_time bruce_rate anne_rate = 12 := by
sorry

end NUMINAMATH_CALUDE_annes_cleaning_time_l1600_160044


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1600_160075

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1600_160075


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_7047_l1600_160060

theorem smallest_prime_factor_of_7047 : Nat.minFac 7047 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_7047_l1600_160060


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l1600_160006

-- Part 1: Non-existence of an infinite sequence of positive integers
theorem no_positive_integer_sequence :
  ¬ (∃ (a : ℕ → ℕ+), ∀ (n : ℕ), (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

-- Part 2: Existence of an infinite sequence of positive irrational numbers
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ (n : ℕ), Irrational (a n) ∧ a n > 0) ∧
    (∀ (n : ℕ), (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l1600_160006


namespace NUMINAMATH_CALUDE_log_problem_l1600_160099

theorem log_problem (x k : ℝ) : 
  (Real.log 3 / Real.log 4 = x) → 
  (Real.log 27 / Real.log 2 = k * x) → 
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l1600_160099


namespace NUMINAMATH_CALUDE_y_squared_value_l1600_160015

theorem y_squared_value (y : ℝ) (h : (y + 16) ^ (1/4) - (y - 16) ^ (1/4) = 2) : y^2 = 272 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_value_l1600_160015


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_equality_l1600_160018

theorem binomial_coefficient_sum_equality (n : ℕ) : 4^n = 2^10 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_equality_l1600_160018


namespace NUMINAMATH_CALUDE_line_perp_plane_iff_perp_all_lines_l1600_160089

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Predicate for a line being perpendicular to a plane -/
def perpendicular_to_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line being perpendicular to another line -/
def perpendicular_to_line (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line being inside a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Theorem stating the equivalence of a line being perpendicular to a plane
    and being perpendicular to all lines in that plane -/
theorem line_perp_plane_iff_perp_all_lines (l : Line3D) (α : Plane3D) :
  perpendicular_to_plane l α ↔ ∀ m : Line3D, line_in_plane m α → perpendicular_to_line l m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_iff_perp_all_lines_l1600_160089


namespace NUMINAMATH_CALUDE_convex_ngon_diagonal_intersections_l1600_160056

/-- The number of intersection points of diagonals in a convex n-gon -/
def diagonalIntersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: The number of intersection points for diagonals of a convex n-gon,
    where no three diagonals intersect at a single point, is equal to n(n-1)(n-2)(n-3)/24 -/
theorem convex_ngon_diagonal_intersections (n : ℕ) (h1 : n ≥ 4) :
  diagonalIntersections n = (n.choose 4) := by
  sorry

end NUMINAMATH_CALUDE_convex_ngon_diagonal_intersections_l1600_160056


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l1600_160066

/-- Represents the "graph number" of a quadratic function y = ax^2 + bx + c -/
structure GraphNumber where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- A quadratic function intersects the x-axis at only one point if and only if its discriminant is zero -/
def intersects_x_axis_once (g : GraphNumber) : Prop :=
  (g.b ^ 2) - (4 * g.a * g.c) = 0

theorem quadratic_intersection_theorem (m : ℝ) (hm : m ≠ 0) :
  let g := GraphNumber.mk m (2 * m + 4) (2 * m + 4) hm
  intersects_x_axis_once g → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l1600_160066


namespace NUMINAMATH_CALUDE_space_diagonals_of_Q_l1600_160068

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  hexagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := (Q.vertices.choose 2)
  let non_edge_segments := total_line_segments - Q.edges
  let face_diagonals := Q.hexagonal_faces * 9
  non_edge_segments - face_diagonals

/-- Theorem stating the number of space diagonals in the specific polyhedron Q -/
theorem space_diagonals_of_Q : 
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 32,
    hexagonal_faces := 12
  }
  space_diagonals Q = 255 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_Q_l1600_160068


namespace NUMINAMATH_CALUDE_square_over_fraction_equals_324_l1600_160024

theorem square_over_fraction_equals_324 : (45^2 : ℚ) / (7 - 3/4) = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_over_fraction_equals_324_l1600_160024


namespace NUMINAMATH_CALUDE_fred_initial_money_l1600_160031

/-- Calculates the initial amount of money Fred had given the number of books bought,
    the average cost per book, and the amount left after buying. -/
def initial_money (num_books : ℕ) (avg_cost : ℕ) (money_left : ℕ) : ℕ :=
  num_books * avg_cost + money_left

/-- Proves that Fred initially had 236 dollars given the problem conditions. -/
theorem fred_initial_money :
  let num_books : ℕ := 6
  let avg_cost : ℕ := 37
  let money_left : ℕ := 14
  initial_money num_books avg_cost money_left = 236 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_money_l1600_160031


namespace NUMINAMATH_CALUDE_multiples_count_l1600_160008

def count_multiples (n : ℕ) : ℕ := 
  (Finset.filter (λ x => (x % 3 = 0 ∨ x % 5 = 0) ∧ x % 6 ≠ 0) (Finset.range (n + 1))).card

theorem multiples_count : count_multiples 200 = 73 := by
  sorry

end NUMINAMATH_CALUDE_multiples_count_l1600_160008


namespace NUMINAMATH_CALUDE_angle_D_measure_l1600_160038

theorem angle_D_measure (A B C D : ℝ) :
  -- ABCD is a convex quadrilateral (implied by the angle sum condition)
  A + B + C + D = 360 →
  -- ∠C = 57°
  C = 57 →
  -- sin ∠A + sin ∠B = √2
  Real.sin A + Real.sin B = Real.sqrt 2 →
  -- cos ∠A + cos ∠B = 2 - √2
  Real.cos A + Real.cos B = 2 - Real.sqrt 2 →
  -- Then ∠D = 168°
  D = 168 := by
sorry

end NUMINAMATH_CALUDE_angle_D_measure_l1600_160038


namespace NUMINAMATH_CALUDE_x_varies_as_sixth_power_of_z_l1600_160064

/-- If x varies as the square of y, and y varies as the cube of z,
    then x varies as the 6th power of z. -/
theorem x_varies_as_sixth_power_of_z
  (x y z : ℝ)
  (k j : ℝ)
  (h1 : x = k * y^2)
  (h2 : y = j * z^3) :
  ∃ m : ℝ, x = m * z^6 := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_sixth_power_of_z_l1600_160064


namespace NUMINAMATH_CALUDE_initial_water_calculation_l1600_160093

/-- The amount of water initially poured into the pool -/
def initial_amount : ℝ := 1

/-- The amount of water added later -/
def added_amount : ℝ := 8.8

/-- The total amount of water in the pool -/
def total_amount : ℝ := 9.8

/-- Theorem stating that the initial amount plus the added amount equals the total amount -/
theorem initial_water_calculation :
  initial_amount + added_amount = total_amount := by
  sorry

end NUMINAMATH_CALUDE_initial_water_calculation_l1600_160093


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1600_160098

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1600_160098


namespace NUMINAMATH_CALUDE_jason_football_games_l1600_160022

/-- Given the number of football games Jason attended this month and last month,
    and the total number of games he plans to attend,
    prove that the number of games he plans to attend next month is 16. -/
theorem jason_football_games (this_month last_month total : ℕ) 
    (h1 : this_month = 11)
    (h2 : last_month = 17)
    (h3 : total = 44) :
    total - (this_month + last_month) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jason_football_games_l1600_160022


namespace NUMINAMATH_CALUDE_parabola_equations_l1600_160002

/-- Parabola with x-axis symmetry -/
def parabola_x_axis (m : ℝ) (x y : ℝ) : Prop :=
  y^2 = m * x

/-- Parabola with y-axis symmetry -/
def parabola_y_axis (p : ℝ) (x y : ℝ) : Prop :=
  x^2 = 4 * p * y

theorem parabola_equations :
  (∃ m : ℝ, m ≠ 0 ∧ parabola_x_axis m 6 (-3)) ∧
  (∃ p : ℝ, p > 0 ∧ parabola_y_axis p x y ↔ x^2 = 12 * y ∨ x^2 = -12 * y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equations_l1600_160002


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l1600_160084

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- Theorem statement
theorem complement_of_union_equals_set (U A B : Set Int) :
  U = {-2, -1, 0, 1, 2, 3} →
  A = {-1, 2} →
  B = {x : Int | x^2 - 4*x + 3 = 0} →
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

-- Note: We use \ for set difference (complement) in Lean

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l1600_160084


namespace NUMINAMATH_CALUDE_remaining_money_after_shopping_l1600_160097

/-- The amount of money remaining after spending 30% of $500 is $350. -/
theorem remaining_money_after_shopping (initial_amount : ℝ) (spent_percentage : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : spent_percentage = 0.30) :
  initial_amount - (spent_percentage * initial_amount) = 350 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_shopping_l1600_160097


namespace NUMINAMATH_CALUDE_pet_store_cages_l1600_160014

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 13 → sold_puppies = 7 → puppies_per_cage = 2 →
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1600_160014


namespace NUMINAMATH_CALUDE_lowest_unique_score_above_90_l1600_160057

/-- Represents the scoring system for the modified AHSME exam -/
def score (c w : ℕ) : ℕ := 35 + 4 * c - w

/-- The total number of questions in the exam -/
def total_questions : ℕ := 35

theorem lowest_unique_score_above_90 :
  ∀ s : ℕ,
  s > 90 →
  (∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s) →
  (∀ s' : ℕ, 90 < s' ∧ s' < s → ¬∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s') →
  s = 91 :=
sorry

end NUMINAMATH_CALUDE_lowest_unique_score_above_90_l1600_160057


namespace NUMINAMATH_CALUDE_positive_A_value_l1600_160036

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l1600_160036


namespace NUMINAMATH_CALUDE_vector_computation_l1600_160086

theorem vector_computation :
  let v1 : Fin 2 → ℝ := ![3, -5]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![2, -4]
  2 • v1 + 4 • v2 - 3 • v3 = ![(-4 : ℝ), 26] :=
by sorry

end NUMINAMATH_CALUDE_vector_computation_l1600_160086


namespace NUMINAMATH_CALUDE_evie_shell_collection_l1600_160096

/-- The number of shells Evie collects per day -/
def shells_per_day : ℕ := 10

/-- The number of shells Evie gives to her brother -/
def shells_given : ℕ := 2

/-- The number of shells Evie has left after giving some to her brother -/
def shells_left : ℕ := 58

/-- The number of days Evie collected shells -/
def collection_days : ℕ := 6

theorem evie_shell_collection :
  shells_per_day * collection_days - shells_given = shells_left :=
by sorry

end NUMINAMATH_CALUDE_evie_shell_collection_l1600_160096


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_odd_l1600_160061

theorem sum_of_multiples_is_odd (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) 
  (hd : ∃ n : ℤ, d = 9 * n) : 
  Odd (c + d) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_odd_l1600_160061


namespace NUMINAMATH_CALUDE_bluray_movies_returned_l1600_160028

/-- Represents the number of movies returned -/
def movies_returned (initial_dvd : ℕ) (initial_bluray : ℕ) (final_dvd : ℕ) (final_bluray : ℕ) : ℕ :=
  initial_bluray - final_bluray

theorem bluray_movies_returned :
  ∀ (initial_dvd initial_bluray final_dvd final_bluray : ℕ),
    initial_dvd + initial_bluray = 378 →
    initial_dvd * 4 = initial_bluray * 17 →
    final_dvd * 2 = final_bluray * 9 →
    final_dvd = initial_dvd →
    movies_returned initial_dvd initial_bluray final_dvd final_bluray = 4 := by
  sorry

end NUMINAMATH_CALUDE_bluray_movies_returned_l1600_160028


namespace NUMINAMATH_CALUDE_storks_vs_birds_l1600_160079

theorem storks_vs_birds (initial_birds : ℕ) (additional_storks : ℕ) (additional_birds : ℕ) :
  initial_birds = 3 →
  additional_storks = 6 →
  additional_birds = 2 →
  additional_storks - (initial_birds + additional_birds) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_vs_birds_l1600_160079


namespace NUMINAMATH_CALUDE_divide_into_eight_parts_l1600_160029

-- Define a type for geometric figures
inductive Figure
  | Cube
  | Rectangle

-- Define a function to check if a figure can be divided into 8 identical parts
def canDivideIntoEightParts (f : Figure) : Prop :=
  match f with
  | Figure.Cube => true
  | Figure.Rectangle => true

-- Theorem stating that any cube or rectangle can be divided into 8 identical parts
theorem divide_into_eight_parts (f : Figure) : canDivideIntoEightParts f := by
  sorry

#check divide_into_eight_parts

end NUMINAMATH_CALUDE_divide_into_eight_parts_l1600_160029


namespace NUMINAMATH_CALUDE_composition_equality_l1600_160090

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := 5 * x + b
def g (b : ℝ) (x : ℝ) : ℝ := b * x + 4

-- State the theorem
theorem composition_equality (b e : ℝ) : 
  (∀ x, f b (g b x) = 15 * x + e) → e = 23 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l1600_160090


namespace NUMINAMATH_CALUDE_quartet_songs_theorem_l1600_160004

theorem quartet_songs_theorem (a b c d e : ℕ) 
  (h1 : (a + b + c + d + e) % 4 = 0)
  (h2 : e = 8)
  (h3 : a = 5)
  (h4 : b > 5 ∧ b < 8)
  (h5 : c > 5 ∧ c < 8)
  (h6 : d > 5 ∧ d < 8) :
  (a + b + c + d + e) / 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_quartet_songs_theorem_l1600_160004


namespace NUMINAMATH_CALUDE_vincent_stickers_l1600_160040

theorem vincent_stickers (yesterday : ℕ) (today_extra : ℕ) : 
  yesterday = 15 → today_extra = 10 → yesterday + (yesterday + today_extra) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_vincent_stickers_l1600_160040


namespace NUMINAMATH_CALUDE_question_selection_probability_l1600_160058

/-- The probability of selecting an algebra question first and a geometry question second -/
def prob_AB (total_questions : ℕ) (algebra_questions : ℕ) (geometry_questions : ℕ) : ℚ :=
  (algebra_questions : ℚ) / total_questions * (geometry_questions : ℚ) / (total_questions - 1)

/-- The probability of selecting a geometry question second given an algebra question was selected first -/
def prob_B_given_A (total_questions : ℕ) (algebra_questions : ℕ) (geometry_questions : ℕ) : ℚ :=
  (geometry_questions : ℚ) / (total_questions - 1)

theorem question_selection_probability :
  let total_questions := 5
  let algebra_questions := 2
  let geometry_questions := 3
  prob_AB total_questions algebra_questions geometry_questions = 3 / 10 ∧
  prob_B_given_A total_questions algebra_questions geometry_questions = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_question_selection_probability_l1600_160058


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l1600_160019

theorem smallest_distance_between_circles (z w : ℂ) : 
  Complex.abs (z - (2 + 2 * Complex.I)) = 2 →
  Complex.abs (w - (5 + 6 * Complex.I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = 11 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 2 * Complex.I)) = 2 →
                   Complex.abs (w' - (5 + 6 * Complex.I)) = 4 →
                   Complex.abs (z' - w') ≥ min_dist :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l1600_160019


namespace NUMINAMATH_CALUDE_purple_sequin_rows_purple_sequin_rows_proof_l1600_160032

theorem purple_sequin_rows (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_per_row : Nat) (green_rows : Nat) (green_per_row : Nat) 
  (total_sequins : Nat) : Nat :=
  let blue_sequins := blue_rows * blue_per_row
  let green_sequins := green_rows * green_per_row
  let non_purple_sequins := blue_sequins + green_sequins
  let purple_sequins := total_sequins - non_purple_sequins
  purple_sequins / purple_per_row

#check purple_sequin_rows 6 8 12 9 6 162 = 5

theorem purple_sequin_rows_proof :
  purple_sequin_rows 6 8 12 9 6 162 = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_sequin_rows_purple_sequin_rows_proof_l1600_160032


namespace NUMINAMATH_CALUDE_computer_price_increase_l1600_160009

theorem computer_price_increase (d : ℝ) : 
  2 * d = 560 → (d * 1.3 : ℝ) = 364 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1600_160009


namespace NUMINAMATH_CALUDE_amanda_pay_l1600_160052

def hourly_rate : ℝ := 50
def hours_worked : ℝ := 10
def withholding_percentage : ℝ := 0.20

def daily_pay : ℝ := hourly_rate * hours_worked
def withheld_amount : ℝ := daily_pay * withholding_percentage
def final_pay : ℝ := daily_pay - withheld_amount

theorem amanda_pay : final_pay = 400 := by
  sorry

end NUMINAMATH_CALUDE_amanda_pay_l1600_160052


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l1600_160023

/-- Given a triangle ABC with heights ha, hb, hc corresponding to sides a, b, c respectively,
    prove that if ha = 6, hb = 4, and hc = 3, then a : b : c = 2 : 3 : 4 -/
theorem triangle_side_ratio (a b c ha hb hc : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_heights : ha = 6 ∧ hb = 4 ∧ hc = 3) 
  (h_area : a * ha = b * hb ∧ b * hb = c * hc) : 
  ∃ (k : ℝ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l1600_160023


namespace NUMINAMATH_CALUDE_work_rate_problem_l1600_160087

theorem work_rate_problem (a b c : ℝ) 
  (hab : a + b = 1/18)
  (hbc : b + c = 1/24)
  (hac : a + c = 1/36) :
  a + b + c = 1/16 := by
sorry

end NUMINAMATH_CALUDE_work_rate_problem_l1600_160087


namespace NUMINAMATH_CALUDE_smallest_divisor_of_720_two_divides_720_smallest_positive_divisor_of_720_is_two_l1600_160051

theorem smallest_divisor_of_720 : 
  ∀ n : ℕ, n > 0 → n ∣ 720 → n ≥ 2 :=
by
  sorry

theorem two_divides_720 : 2 ∣ 720 :=
by
  sorry

theorem smallest_positive_divisor_of_720_is_two : 
  ∃ (d : ℕ), d > 0 ∧ d ∣ 720 ∧ ∀ n : ℕ, n > 0 → n ∣ 720 → n ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_720_two_divides_720_smallest_positive_divisor_of_720_is_two_l1600_160051


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1600_160042

theorem cubic_equation_root (a b : ℚ) : 
  (1 + Real.sqrt 5)^3 + a * (1 + Real.sqrt 5)^2 + b * (1 + Real.sqrt 5) - 60 = 0 → 
  b = 26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1600_160042


namespace NUMINAMATH_CALUDE_square_area_is_17_l1600_160016

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by four vertices -/
structure Square where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Calculate the area of a square given its four vertices -/
def squareArea (s : Square) : ℝ :=
  squaredDistance s.P s.Q

/-- The specific square from the problem -/
def problemSquare : Square :=
  { P := { x := 1, y := 2 },
    Q := { x := -3, y := 3 },
    R := { x := -2, y := 8 },
    S := { x := 2, y := 7 } }

theorem square_area_is_17 :
  squareArea problemSquare = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_17_l1600_160016


namespace NUMINAMATH_CALUDE_smallest_mn_for_almost_shaded_square_l1600_160026

theorem smallest_mn_for_almost_shaded_square (m n : ℕ+) 
  (h_bound : 2 * n < m ∧ m < 3 * n) 
  (h_exists : ∃ (p q : ℕ) (k : ℤ), 
    p < m ∧ q < n ∧ 
    0 < (m * q - n * p) * (m * q - n * p) ∧ 
    (m * q - n * p) * (m * q - n * p) < 2 * m * n / 1000) :
  506 ≤ m * n ∧ m * n ≤ 510 := by
sorry

end NUMINAMATH_CALUDE_smallest_mn_for_almost_shaded_square_l1600_160026


namespace NUMINAMATH_CALUDE_red_peaches_per_basket_l1600_160083

theorem red_peaches_per_basket 
  (num_baskets : ℕ) 
  (green_per_basket : ℕ) 
  (total_peaches : ℕ) 
  (h1 : num_baskets = 15)
  (h2 : green_per_basket = 4)
  (h3 : total_peaches = 345) :
  (total_peaches - num_baskets * green_per_basket) / num_baskets = 19 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_per_basket_l1600_160083


namespace NUMINAMATH_CALUDE_work_completion_time_l1600_160005

theorem work_completion_time (a b c : ℝ) : 
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (a = 1/6) →
  (a + b + c = 1/3) →
  (c = 1/8 * (a + b)) →
  (b = 1/28) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1600_160005


namespace NUMINAMATH_CALUDE_original_selling_price_l1600_160073

theorem original_selling_price (CP : ℝ) : 
  CP * 0.85 = 544 → CP * 1.25 = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_selling_price_l1600_160073


namespace NUMINAMATH_CALUDE_last_digit_of_fraction_l1600_160065

def last_digit (n : ℚ) : ℕ := sorry

theorem last_digit_of_fraction :
  last_digit (1 / (2^10 * 3^10)) = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_fraction_l1600_160065


namespace NUMINAMATH_CALUDE_basic_computer_price_l1600_160007

/-- The price of a basic computer and printer totaling $2,500, 
    where an enhanced computer costing $500 more would make the printer 1/8 of the new total. -/
theorem basic_computer_price : 
  ∀ (basic_price printer_price enhanced_price : ℝ),
  basic_price + printer_price = 2500 →
  enhanced_price = basic_price + 500 →
  printer_price = (1/8) * (enhanced_price + printer_price) →
  basic_price = 2125 := by
sorry

end NUMINAMATH_CALUDE_basic_computer_price_l1600_160007


namespace NUMINAMATH_CALUDE_washing_machine_last_load_l1600_160046

theorem washing_machine_last_load (capacity : ℕ) (total_clothes : ℕ) : 
  capacity = 28 → total_clothes = 200 → 
  total_clothes % capacity = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_washing_machine_last_load_l1600_160046


namespace NUMINAMATH_CALUDE_max_sum_smallest_angles_l1600_160012

/-- A line in a plane --/
structure Line where
  -- We don't need to define the structure of a line for this statement

/-- Represents the configuration of lines in a plane --/
structure LineConfiguration where
  lines : Finset Line
  general_position : Bool

/-- Calculates the sum of smallest angles at intersections --/
def sum_smallest_angles (config : LineConfiguration) : ℝ :=
  sorry

/-- The theorem statement --/
theorem max_sum_smallest_angles :
  ∀ (config : LineConfiguration),
    config.lines.card = 10 ∧ config.general_position →
    ∃ (max_sum : ℝ), 
      (∀ (c : LineConfiguration), c.lines.card = 10 ∧ c.general_position → 
        sum_smallest_angles c ≤ max_sum) ∧
      max_sum = 2250 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_smallest_angles_l1600_160012


namespace NUMINAMATH_CALUDE_stock_discount_calculation_l1600_160049

/-- Calculates the discount on a stock given its original price, brokerage fee, and final cost price. -/
theorem stock_discount_calculation (original_price brokerage_rate final_cost_price : ℝ) : 
  original_price = 100 →
  brokerage_rate = 1 / 500 →
  final_cost_price = 95.2 →
  ∃ (discount : ℝ), 
    (original_price - discount) * (1 + brokerage_rate) = final_cost_price ∧
    abs (discount - 4.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_stock_discount_calculation_l1600_160049


namespace NUMINAMATH_CALUDE_M_inter_N_eq_N_l1600_160059

open Set Real

def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | log x > 0}

theorem M_inter_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_N_l1600_160059


namespace NUMINAMATH_CALUDE_meaningful_exponent_range_l1600_160074

theorem meaningful_exponent_range (x : ℝ) : 
  (∃ y : ℝ, (2*x - 3)^0 = y) ↔ x ≠ 3/2 := by sorry

end NUMINAMATH_CALUDE_meaningful_exponent_range_l1600_160074


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l1600_160011

theorem sqrt_sum_equals_2sqrt14 : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_2sqrt14_l1600_160011


namespace NUMINAMATH_CALUDE_mean_temperature_l1600_160021

theorem mean_temperature (temperatures : List ℤ) : 
  temperatures = [-10, -4, -6, -3, 0, 2, 5, 0] →
  (temperatures.sum / temperatures.length : ℚ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l1600_160021


namespace NUMINAMATH_CALUDE_zachary_pushups_l1600_160039

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (h1 : david_pushups = 62) (h2 : difference = 15) :
  david_pushups - difference = 47 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_l1600_160039


namespace NUMINAMATH_CALUDE_last_four_digits_of_perfect_square_l1600_160088

theorem last_four_digits_of_perfect_square (n : ℕ) : 
  (∃ d : ℕ, d < 10 ∧ n^2 % 1000 = d * 111) → 
  (n^2 % 10000 = 0 ∨ n^2 % 10000 = 1444) :=
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_perfect_square_l1600_160088


namespace NUMINAMATH_CALUDE_distance_from_origin_to_12_5_l1600_160067

/-- The distance from the origin to the point (12, 5) in a rectangular coordinate system is 13 units. -/
theorem distance_from_origin_to_12_5 : 
  Real.sqrt (12^2 + 5^2) = 13 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_12_5_l1600_160067


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l1600_160001

/-- The equation of the tangent line to y = x³ at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- The curve equation
  (x = 1 ∧ y = 1) → -- The point (1, 1) on the curve
  (3*x - y - 2 = 0) -- The equation of the tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l1600_160001


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l1600_160081

/-- The number of dots in each row or column of the grid -/
def grid_size : ℕ := 5

/-- The total number of dots in the grid -/
def total_dots : ℕ := grid_size * grid_size

/-- The number of dots to be selected -/
def selected_dots : ℕ := 4

/-- The number of sets of collinear dots -/
def collinear_sets : ℕ := 14

/-- The total number of ways to choose 4 dots out of 25 -/
def total_combinations : ℕ := Nat.choose total_dots selected_dots

/-- The probability of selecting 4 collinear dots -/
def collinear_probability : ℚ := collinear_sets / total_combinations

theorem collinear_dots_probability :
  collinear_probability = 7 / 6325 := by sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l1600_160081


namespace NUMINAMATH_CALUDE_min_side_length_l1600_160076

/-- An isosceles triangle with a perpendicular line from vertex to base -/
structure IsoscelesTriangleWithPerp where
  -- The length of two equal sides
  side : ℝ
  -- The length of CD
  cd : ℕ
  -- Assertion that BD^2 = 77
  h_bd_sq : side^2 - cd^2 = 77

/-- The theorem stating the minimal possible integer value for AC -/
theorem min_side_length (t : IsoscelesTriangleWithPerp) : 
  ∃ (min : ℕ), (∀ (t' : IsoscelesTriangleWithPerp), (t'.side : ℝ) ≥ min) ∧ min = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_side_length_l1600_160076


namespace NUMINAMATH_CALUDE_cube_volume_percentage_l1600_160037

theorem cube_volume_percentage (box_length box_width box_height cube_side : ℕ) 
  (h1 : box_length = 8)
  (h2 : box_width = 6)
  (h3 : box_height = 12)
  (h4 : cube_side = 4) :
  (((box_length / cube_side) * (box_width / cube_side) * (box_height / cube_side) * cube_side^3) : ℚ) /
  (box_length * box_width * box_height) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_percentage_l1600_160037


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1600_160003

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 18*x - 10*y + 22) :
  x + y = 4 + 2 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1600_160003


namespace NUMINAMATH_CALUDE_article_profit_percentage_l1600_160043

theorem article_profit_percentage (cost selling_price : ℚ) : 
  cost = 70 →
  (0.8 * cost) * 1.3 = selling_price - 14.70 →
  (selling_price - cost) / cost * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_article_profit_percentage_l1600_160043


namespace NUMINAMATH_CALUDE_stock_price_fluctuation_l1600_160055

theorem stock_price_fluctuation (original_price : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  increase_percent = 0.40 →
  decrease_percent = 2 / 7 →
  original_price * (1 + increase_percent) * (1 - decrease_percent) = original_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_fluctuation_l1600_160055


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_parallelepiped_l1600_160082

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  diagonal : ℝ
  angle_with_base : ℝ
  base_area : ℝ

-- Define the theorem
theorem lateral_surface_area_of_parallelepiped (p : RectangularParallelepiped) 
  (h1 : p.diagonal = 10)
  (h2 : p.angle_with_base = Real.pi / 3)  -- 60 degrees in radians
  (h3 : p.base_area = 12) :
  ∃ (lateral_area : ℝ), lateral_area = 70 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_parallelepiped_l1600_160082


namespace NUMINAMATH_CALUDE_triangle_properties_l1600_160063

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given equation holds for the triangle -/
def satisfies_equation (t : Triangle) : Prop :=
  (t.a^2 + t.c^2 - t.b^2) / (t.a^2 + t.b^2 - t.c^2) = t.c / (Real.sqrt 2 * t.a - t.c)

theorem triangle_properties (t : Triangle) (h : satisfies_equation t) :
  t.B = π/4 ∧ 
  (t.b = 1 → ∃ (max_area : ℝ), max_area = (Real.sqrt 2 + 1) / 4 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.c * Real.sin t.B → area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1600_160063


namespace NUMINAMATH_CALUDE_real_part_of_Z_l1600_160035

theorem real_part_of_Z (Z : ℂ) (h : (1 + Complex.I) * Z = Complex.abs (3 + 4 * Complex.I)) : 
  Z.re = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_Z_l1600_160035


namespace NUMINAMATH_CALUDE_area_scientific_notation_l1600_160078

-- Define the area in square kilometers
def area : ℝ := 6.4e6

-- Theorem to prove the scientific notation representation
theorem area_scientific_notation : area = 6.4 * (10 : ℝ)^6 := by
  sorry

end NUMINAMATH_CALUDE_area_scientific_notation_l1600_160078


namespace NUMINAMATH_CALUDE_smallest_nonnegative_a_l1600_160050

theorem smallest_nonnegative_a (b : ℝ) (a : ℝ) (h1 : b = π / 4) 
  (h2 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  a ≥ 0 ∧ a = 17 - π / 4 ∧ ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ a :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_a_l1600_160050


namespace NUMINAMATH_CALUDE_problem_solution_l1600_160070

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the conditions of the problem
def is_purely_imaginary (z : ℂ) : Prop := ∃ (a : ℝ), z = a * i

-- State the theorem
theorem problem_solution (x y : ℂ) 
  (h1 : is_purely_imaginary x) 
  (h2 : y.im = 0) 
  (h3 : 2 * x - 1 + i = y - (3 - y) * i) : 
  x + y = -1 - (5/2) * i := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1600_160070


namespace NUMINAMATH_CALUDE_class_size_l1600_160091

/-- Given a class with a hair color ratio of 3:6:7 (red:blonde:black) and 9 red-haired kids,
    the total number of kids in the class is 48. -/
theorem class_size (red blonde black : ℕ) (total : ℕ) : 
  red = 3 → blonde = 6 → black = 7 → -- ratio condition
  red + blonde + black = total → -- total parts in ratio
  9 * total = 48 * red → -- condition for 9 red-haired kids
  total = 48 := by sorry

end NUMINAMATH_CALUDE_class_size_l1600_160091


namespace NUMINAMATH_CALUDE_angle_difference_l1600_160092

theorem angle_difference (α β : Real) 
  (h1 : 3 * Real.sin α - Real.cos α = 0)
  (h2 : 7 * Real.sin β + Real.cos β = 0)
  (h3 : 0 < α) (h4 : α < Real.pi / 2)
  (h5 : Real.pi / 2 < β) (h6 : β < Real.pi) :
  2 * α - β = -3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_l1600_160092


namespace NUMINAMATH_CALUDE_expression_equality_l1600_160085

theorem expression_equality : 99^4 - 4 * 99^3 + 6 * 99^2 - 4 * 99 + 1 = 92199816 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1600_160085


namespace NUMINAMATH_CALUDE_no_integer_solution_l1600_160047

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_values : P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5) :
  ¬∃ k : ℤ, P.eval k = 8 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1600_160047


namespace NUMINAMATH_CALUDE_jimmy_shorts_count_l1600_160025

def senior_discount : ℚ := 10 / 100
def shorts_price : ℚ := 15
def shirt_price : ℚ := 17
def num_shirts : ℕ := 5
def total_paid : ℚ := 117

def num_shorts : ℕ := 2

theorem jimmy_shorts_count :
  let shirts_cost := shirt_price * num_shirts
  let discount := shirts_cost * senior_discount
  let irene_total := shirts_cost - discount
  let remaining := total_paid - irene_total
  (remaining / shorts_price).floor = num_shorts := by sorry

end NUMINAMATH_CALUDE_jimmy_shorts_count_l1600_160025


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1600_160010

theorem arithmetic_mean_problem : ∃ x : ℝ, (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 9 ∧ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1600_160010


namespace NUMINAMATH_CALUDE_money_left_over_l1600_160034

/-- Calculates the money left over after buying a bike given work parameters and bike cost -/
theorem money_left_over 
  (hourly_rate : ℝ) 
  (weekly_hours : ℝ) 
  (weeks_worked : ℝ) 
  (bike_cost : ℝ) 
  (h1 : hourly_rate = 8)
  (h2 : weekly_hours = 35)
  (h3 : weeks_worked = 4)
  (h4 : bike_cost = 400) :
  hourly_rate * weekly_hours * weeks_worked - bike_cost = 720 := by
  sorry


end NUMINAMATH_CALUDE_money_left_over_l1600_160034


namespace NUMINAMATH_CALUDE_second_number_is_22_l1600_160072

theorem second_number_is_22 (x y : ℝ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_22_l1600_160072


namespace NUMINAMATH_CALUDE_sand_weight_difference_l1600_160054

/-- Proves that the sand in the box is heavier than the sand in the barrel by 260 grams --/
theorem sand_weight_difference 
  (barrel_weight : ℕ) 
  (barrel_with_sand_weight : ℕ) 
  (box_weight : ℕ) 
  (box_with_sand_weight : ℕ) 
  (h1 : barrel_weight = 250)
  (h2 : barrel_with_sand_weight = 1780)
  (h3 : box_weight = 460)
  (h4 : box_with_sand_weight = 2250) :
  (box_with_sand_weight - box_weight) - (barrel_with_sand_weight - barrel_weight) = 260 := by
  sorry

#check sand_weight_difference

end NUMINAMATH_CALUDE_sand_weight_difference_l1600_160054


namespace NUMINAMATH_CALUDE_train_crossing_time_l1600_160071

/-- Calculates the time for a train to cross a signal pole given its length, platform length, and time to cross the platform. -/
theorem train_crossing_time (train_length platform_length time_cross_platform : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 600.0000000000001)
  (h3 : time_cross_platform = 54) : 
  ∃ (time_cross_pole : ℝ), 
    (time_cross_pole ≥ 17.99 ∧ time_cross_pole ≤ 18.01) ∧
    time_cross_pole = train_length / ((train_length + platform_length) / time_cross_platform) :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1600_160071


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1600_160027

theorem rationalize_denominator :
  let x := (Real.sqrt 12 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2)
  ∃ y, y = 4 - Real.sqrt 6 ∧ x = y :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1600_160027


namespace NUMINAMATH_CALUDE_doughnut_savings_l1600_160033

/-- The cost of one dozen doughnuts -/
def cost_one_dozen : ℕ := 8

/-- The cost of two dozens of doughnuts -/
def cost_two_dozens : ℕ := 14

/-- The number of sets when buying one dozen at a time -/
def sets_one_dozen : ℕ := 6

/-- The number of sets when buying two dozens at a time -/
def sets_two_dozens : ℕ := 3

/-- Theorem stating the savings when buying 3 sets of 2 dozens instead of 6 sets of 1 dozen -/
theorem doughnut_savings : 
  sets_one_dozen * cost_one_dozen - sets_two_dozens * cost_two_dozens = 6 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_savings_l1600_160033


namespace NUMINAMATH_CALUDE_least_repeating_digits_7_13_is_6_l1600_160094

/-- The least number of digits in a repeating block of the decimal expansion of 7/13 -/
def least_repeating_digits_7_13 : ℕ :=
  6

/-- Theorem stating that the least number of digits in a repeating block 
    of the decimal expansion of 7/13 is 6 -/
theorem least_repeating_digits_7_13_is_6 :
  least_repeating_digits_7_13 = 6 := by sorry

end NUMINAMATH_CALUDE_least_repeating_digits_7_13_is_6_l1600_160094
