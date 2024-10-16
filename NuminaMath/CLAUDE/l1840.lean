import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l1840_184032

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1}
def B (m : ℝ) : Set ℝ := {x | x ≤ m}

-- State the theorem
theorem range_of_m (m : ℝ) : B m ⊆ A → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1840_184032


namespace NUMINAMATH_CALUDE_rectangle_fold_ef_length_l1840_184072

-- Define the rectangle
structure Rectangle :=
  (AB : ℝ)
  (BC : ℝ)

-- Define the fold
structure Fold :=
  (distanceFromB : ℝ)
  (distanceFromC : ℝ)

-- Define the theorem
theorem rectangle_fold_ef_length 
  (rect : Rectangle)
  (fold : Fold)
  (h1 : rect.AB = 4)
  (h2 : rect.BC = 8)
  (h3 : fold.distanceFromB = 3)
  (h4 : fold.distanceFromC = 5)
  (h5 : fold.distanceFromB + fold.distanceFromC = rect.BC) :
  let EF := Real.sqrt ((rect.AB ^ 2) + (fold.distanceFromB ^ 2))
  EF = 5 := by sorry

end NUMINAMATH_CALUDE_rectangle_fold_ef_length_l1840_184072


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l1840_184019

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ((a + 1) * x > a + 1) ↔ (x < 1)) → 
  a < -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l1840_184019


namespace NUMINAMATH_CALUDE_perpendicular_vectors_coefficient_l1840_184062

/-- Given two vectors in the plane, if one is perpendicular to a linear combination of both,
    then the coefficient in the linear combination is -5. -/
theorem perpendicular_vectors_coefficient (a b : ℝ × ℝ) (t : ℝ) :
  a = (1, -1) →
  b = (6, -4) →
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) →
  t = -5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_coefficient_l1840_184062


namespace NUMINAMATH_CALUDE_lower_limit_of_b_l1840_184060

theorem lower_limit_of_b (a b : ℤ) (h1 : 10 ≤ a ∧ a ≤ 25) (h2 : b < 31) 
  (h3 : (a : ℚ) / b ≤ 4/3) : 19 ≤ b := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_of_b_l1840_184060


namespace NUMINAMATH_CALUDE_distance_maximized_at_neg_one_l1840_184034

/-- The point P -/
def P : ℝ × ℝ := (3, 2)

/-- The point Q -/
def Q : ℝ × ℝ := (2, 1)

/-- The line equation: mx - y + 1 - 2m = 0 -/
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - 2 * m = 0

/-- The line passes through point Q for all m -/
axiom line_through_Q (m : ℝ) : line_equation m Q.1 Q.2

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem distance_maximized_at_neg_one :
  ∃ (max_dist : ℝ), ∀ (m : ℝ),
    distance_to_line P m ≤ max_dist ∧
    distance_to_line P (-1) = max_dist :=
  sorry

end NUMINAMATH_CALUDE_distance_maximized_at_neg_one_l1840_184034


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1840_184066

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ x y : ℝ, x^2 = 4*y ∧ y = k*x - 2 ∧ k = (1/2)*x) → k^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1840_184066


namespace NUMINAMATH_CALUDE_michael_digging_time_l1840_184016

/-- Given the conditions of Michael and his father's digging, prove that Michael will take 700 hours to dig his hole. -/
theorem michael_digging_time (father_rate : ℝ) (father_time : ℝ) (depth_difference : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  depth_difference = 400 →
  let father_depth := father_rate * father_time
  let michael_depth := 2 * father_depth - depth_difference
  michael_depth / father_rate = 700 := by
  sorry

end NUMINAMATH_CALUDE_michael_digging_time_l1840_184016


namespace NUMINAMATH_CALUDE_union_complement_equality_l1840_184020

def I : Set Int := {-3, -2, -1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_complement_equality : A ∪ (I \ B) = {-3, -1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l1840_184020


namespace NUMINAMATH_CALUDE_f_properties_l1840_184005

def f (a x : ℝ) : ℝ := |x - 2*a| + |x + a|

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f 1 x ≥ 3) ∧ 
  (∃ x : ℝ, f 1 x = 3) ∧
  (∀ x : ℝ, a < 0 → f a x ≥ 5*a) ∧
  (∀ x : ℝ, a > 0 → (f a x ≥ 5*a ↔ (x ≤ -2*a ∨ x ≥ 3*a))) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1840_184005


namespace NUMINAMATH_CALUDE_percentage_problem_l1840_184047

theorem percentage_problem (x : ℝ) : 
  (0.12 * 160) - (x / 100 * 80) = 11.2 ↔ x = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1840_184047


namespace NUMINAMATH_CALUDE_range_of_a_l1840_184089

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≥ a}
def B : Set ℝ := {x | |x - 1| < 1}

-- Define the property of A being a necessary but not sufficient condition for B
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B)

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) :
  necessary_not_sufficient a → a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1840_184089


namespace NUMINAMATH_CALUDE_point_outside_circle_l1840_184039

theorem point_outside_circle (r d : ℝ) (hr : r = 2) (hd : d = 3) :
  d > r :=
by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1840_184039


namespace NUMINAMATH_CALUDE_unique_triangle_arrangement_l1840_184086

-- Define the structure of the triangle
structure Triangle :=
  (A B C D : ℕ)
  (side1 side2 side3 : ℕ)

-- Define the conditions of the problem
def validTriangle (t : Triangle) : Prop :=
  t.A ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.B ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.C ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.D ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧
  t.C ≠ t.D ∧
  t.side1 = 1 + t.B + 5 ∧
  t.side2 = 3 + 4 + t.D ∧
  t.side3 = 2 + t.A + 4 ∧
  t.side1 = t.side2 ∧ t.side2 = t.side3

-- Theorem statement
theorem unique_triangle_arrangement :
  ∃! t : Triangle, validTriangle t ∧ t.A = 6 ∧ t.B = 8 ∧ t.C = 7 ∧ t.D = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_arrangement_l1840_184086


namespace NUMINAMATH_CALUDE_probability_negative_product_l1840_184063

def dice_faces : Finset Int := {-3, -2, -1, 0, 1, 2}

def is_negative_product (x y : Int) : Bool :=
  x * y < 0

def count_negative_products : Nat :=
  (dice_faces.filter (λ x => x < 0)).card * (dice_faces.filter (λ x => x > 0)).card * 2

theorem probability_negative_product :
  (count_negative_products : ℚ) / (dice_faces.card * dice_faces.card) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_negative_product_l1840_184063


namespace NUMINAMATH_CALUDE_f_monotonicity_g_minimum_common_tangent_l1840_184050

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_monotonicity (x : ℝ) (hx : x > 0) :
  (x < Real.exp 1 → deriv f x > 0) ∧
  (x > Real.exp 1 → deriv f x < 0) := by sorry

def g (x : ℝ) : ℝ := Real.log x + 1 / x

theorem g_minimum :
  ∀ x > 0, g x ≥ 1 := by sorry

def h (x : ℝ) (m : ℝ) : ℝ := (1 / 6) * x^2 + (2 / 3) * x - m

theorem common_tangent (m : ℝ) :
  (∃ x, f x = h x m ∧ deriv f x = deriv (h · m) x) →
  m = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_f_monotonicity_g_minimum_common_tangent_l1840_184050


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l1840_184051

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l1840_184051


namespace NUMINAMATH_CALUDE_randy_lunch_cost_l1840_184017

theorem randy_lunch_cost (initial_amount : ℝ) (ice_cream_cost : ℝ) : 
  initial_amount = 30 →
  ice_cream_cost = 5 →
  ∃ (lunch_cost : ℝ),
    lunch_cost = 10 ∧
    (1/4) * (initial_amount - lunch_cost) = ice_cream_cost :=
by
  sorry

end NUMINAMATH_CALUDE_randy_lunch_cost_l1840_184017


namespace NUMINAMATH_CALUDE_sports_club_intersection_l1840_184061

theorem sports_club_intersection (N B T X : ℕ) : 
  N = 30 ∧ B = 18 ∧ T = 19 ∧ (N - (B + T - X) = 2) → X = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_intersection_l1840_184061


namespace NUMINAMATH_CALUDE_jack_classics_books_l1840_184002

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := 6

/-- The number of books per author -/
def books_per_author : ℕ := 33

/-- Theorem: The total number of books in Jack's classics section is 198 -/
theorem jack_classics_books : num_authors * books_per_author = 198 := by
  sorry

end NUMINAMATH_CALUDE_jack_classics_books_l1840_184002


namespace NUMINAMATH_CALUDE_shelbys_rainy_drive_time_l1840_184031

theorem shelbys_rainy_drive_time 
  (speed_no_rain : ℝ) 
  (speed_rain : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_no_rain = 30) 
  (h2 : speed_rain = 20) 
  (h3 : total_distance = 24) 
  (h4 : total_time = 50) : 
  ∃ (rain_time : ℝ), 
    rain_time = 3 ∧ 
    (speed_no_rain / 60) * (total_time - rain_time) + (speed_rain / 60) * rain_time = total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shelbys_rainy_drive_time_l1840_184031


namespace NUMINAMATH_CALUDE_signed_author_x_percentage_l1840_184018

-- Define the total number of books
def total_books : ℕ := 120

-- Define the percentage of novels
def novel_percentage : ℚ := 65 / 100

-- Define the number of graphic novels
def graphic_novels : ℕ := 18

-- Define the number of comic books by Author X
def author_x_comics : ℕ := 10

-- Define the number of signed comic books by Author X
def signed_author_x_comics : ℕ := 4

-- Theorem to prove
theorem signed_author_x_percentage :
  (signed_author_x_comics : ℚ) / total_books * 100 = 3.33 := by
  sorry

end NUMINAMATH_CALUDE_signed_author_x_percentage_l1840_184018


namespace NUMINAMATH_CALUDE_newspaper_collection_ratio_l1840_184038

def chris_newspapers : ℕ := 42
def lily_extra_newspapers : ℕ := 23

def lily_newspapers : ℕ := chris_newspapers + lily_extra_newspapers

theorem newspaper_collection_ratio :
  (chris_newspapers : ℚ) / (lily_newspapers : ℚ) = 42 / 65 :=
by sorry

end NUMINAMATH_CALUDE_newspaper_collection_ratio_l1840_184038


namespace NUMINAMATH_CALUDE_library_comic_books_l1840_184096

theorem library_comic_books (fairy_tale_books : ℕ) (science_tech_books : ℕ) (comic_books : ℕ) : 
  fairy_tale_books = 305 →
  science_tech_books = fairy_tale_books + 115 →
  comic_books = 4 * (fairy_tale_books + science_tech_books) →
  comic_books = 2900 := by
sorry

end NUMINAMATH_CALUDE_library_comic_books_l1840_184096


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1840_184055

theorem ceiling_sum_sqrt : ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 150⌉ + ⌈Real.sqrt 250⌉ = 37 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l1840_184055


namespace NUMINAMATH_CALUDE_min_value_theorem_l1840_184058

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  ∀ x, 2 * a + b + c ≥ x → x ≤ 2 * Real.sqrt 3 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1840_184058


namespace NUMINAMATH_CALUDE_equivalent_operations_l1840_184092

theorem equivalent_operations (x : ℝ) : 
  (x * (5/6)) / (2/3) = x * (15/12) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operations_l1840_184092


namespace NUMINAMATH_CALUDE_haley_garden_problem_l1840_184001

def seeds_in_big_garden (total_seeds small_gardens seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - small_gardens * seeds_per_small_garden

theorem haley_garden_problem (total_seeds small_gardens seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : small_gardens = 7)
  (h3 : seeds_per_small_garden = 3) :
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 := by
  sorry

end NUMINAMATH_CALUDE_haley_garden_problem_l1840_184001


namespace NUMINAMATH_CALUDE_machine_input_l1840_184015

/-- A machine that processes numbers -/
def Machine (x : ℤ) : ℤ := x + 15 - 6

/-- Theorem: If the machine outputs 35, the input must have been 26 -/
theorem machine_input (x : ℤ) : Machine x = 35 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_machine_input_l1840_184015


namespace NUMINAMATH_CALUDE_spinster_cat_difference_l1840_184068

theorem spinster_cat_difference (spinster_count : ℕ) (cat_count : ℕ) : 
  spinster_count = 14 →
  (2 : ℚ) / 7 = spinster_count / cat_count →
  cat_count > spinster_count →
  cat_count - spinster_count = 35 := by
sorry

end NUMINAMATH_CALUDE_spinster_cat_difference_l1840_184068


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1840_184075

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) 
  (h : 3 • a + (3/5 : ℝ) • (b - x) = b) : 
  x = 5 • a - (2/3 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1840_184075


namespace NUMINAMATH_CALUDE_find_divisor_l1840_184088

theorem find_divisor (N : ℕ) (D : ℕ) (h1 : N = 44 * 432) 
  (h2 : ∃ Q : ℕ, N = D * Q + 3) (h3 : D > 0) : D = 43 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1840_184088


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1840_184021

theorem sphere_surface_area_ratio (V₁ V₂ A₁ A₂ : ℝ) (h : V₁ / V₂ = 8 / 27) :
  A₁ / A₂ = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1840_184021


namespace NUMINAMATH_CALUDE_optimal_arrangement_l1840_184042

/-- Represents the capacity and cost of a truck type -/
structure TruckType where
  water_capacity : ℕ
  vegetable_capacity : ℕ
  cost : ℕ

/-- Represents the donation quantities and truck types -/
structure DonationProblem where
  total_donation : ℕ
  water_vegetable_diff : ℕ
  type_a : TruckType
  type_b : TruckType
  total_trucks : ℕ

def problem : DonationProblem :=
  { total_donation := 120
  , water_vegetable_diff := 12
  , type_a := { water_capacity := 5, vegetable_capacity := 8, cost := 400 }
  , type_b := { water_capacity := 6, vegetable_capacity := 6, cost := 360 }
  , total_trucks := 10
  }

def water_amount (p : DonationProblem) : ℕ :=
  (p.total_donation - p.water_vegetable_diff) / 2

def vegetable_amount (p : DonationProblem) : ℕ :=
  p.total_donation - water_amount p

def is_valid_arrangement (p : DonationProblem) (type_a_count : ℕ) : Prop :=
  let type_b_count := p.total_trucks - type_a_count
  type_a_count * p.type_a.water_capacity + type_b_count * p.type_b.water_capacity ≥ water_amount p ∧
  type_a_count * p.type_a.vegetable_capacity + type_b_count * p.type_b.vegetable_capacity ≥ vegetable_amount p

def transportation_cost (p : DonationProblem) (type_a_count : ℕ) : ℕ :=
  type_a_count * p.type_a.cost + (p.total_trucks - type_a_count) * p.type_b.cost

theorem optimal_arrangement (p : DonationProblem) :
  ∃ (type_a_count : ℕ),
    type_a_count = 3 ∧
    is_valid_arrangement p type_a_count ∧
    ∀ (other_count : ℕ),
      is_valid_arrangement p other_count →
      transportation_cost p type_a_count ≤ transportation_cost p other_count :=
sorry

#eval transportation_cost problem 3  -- Should evaluate to 3720

end NUMINAMATH_CALUDE_optimal_arrangement_l1840_184042


namespace NUMINAMATH_CALUDE_solution_value_l1840_184070

theorem solution_value (m : ℝ) : 
  (∃ x y : ℝ, m * x + 2 * y = 6 ∧ x = 1 ∧ y = 2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1840_184070


namespace NUMINAMATH_CALUDE_paint_bought_l1840_184084

theorem paint_bought (total_needed paint_existing paint_still_needed : ℕ) 
  (h1 : total_needed = 70)
  (h2 : paint_existing = 36)
  (h3 : paint_still_needed = 11) :
  total_needed - paint_existing - paint_still_needed = 23 :=
by sorry

end NUMINAMATH_CALUDE_paint_bought_l1840_184084


namespace NUMINAMATH_CALUDE_future_age_calculation_l1840_184010

theorem future_age_calculation (nora_current_age terry_current_age : ℕ) 
  (h1 : nora_current_age = 10)
  (h2 : terry_current_age = 30) :
  ∃ (years_future : ℕ), terry_current_age + years_future = 4 * nora_current_age ∧ years_future = 10 :=
by sorry

end NUMINAMATH_CALUDE_future_age_calculation_l1840_184010


namespace NUMINAMATH_CALUDE_regular_hexagon_cosine_product_l1840_184028

/-- A regular hexagon ABCDEF inscribed in a circle -/
structure RegularHexagon where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- Length of diagonal AC -/
  diagonal_length : ℝ
  /-- Side length is positive -/
  side_pos : side_length > 0
  /-- Diagonal length is positive -/
  diagonal_pos : diagonal_length > 0
  /-- Relationship between side length and diagonal length in a regular hexagon -/
  hexagon_property : diagonal_length^2 = side_length^2 + side_length^2 - 2 * side_length * side_length * (-1/2)

/-- Theorem about the product of cosines in a regular hexagon -/
theorem regular_hexagon_cosine_product (h : RegularHexagon) (h_side : h.side_length = 5) (h_diag : h.diagonal_length = 2) :
  (1 - Real.cos (2 * Real.pi / 3)) * (1 - Real.cos (2 * Real.pi / 3)) = 2.25 := by
  sorry


end NUMINAMATH_CALUDE_regular_hexagon_cosine_product_l1840_184028


namespace NUMINAMATH_CALUDE_paige_score_l1840_184065

/-- Given a dodgeball team with the following properties:
  * The team has 5 players
  * The team scored a total of 41 points
  * 4 players scored 6 points each
  Prove that the remaining player (Paige) scored 17 points. -/
theorem paige_score (team_size : ℕ) (total_score : ℕ) (other_player_score : ℕ) :
  team_size = 5 →
  total_score = 41 →
  other_player_score = 6 →
  total_score - (team_size - 1) * other_player_score = 17 := by
  sorry


end NUMINAMATH_CALUDE_paige_score_l1840_184065


namespace NUMINAMATH_CALUDE_r_equals_1464_when_n_is_1_l1840_184000

/-- Given the conditions for r and s, prove that r equals 1464 when n is 1 -/
theorem r_equals_1464_when_n_is_1 (n : ℕ) (s r : ℕ) 
  (h1 : s = 4^n + 2) 
  (h2 : r = 2 * 3^s + s) 
  (h3 : n = 1) : 
  r = 1464 := by
  sorry

end NUMINAMATH_CALUDE_r_equals_1464_when_n_is_1_l1840_184000


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l1840_184069

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l1840_184069


namespace NUMINAMATH_CALUDE_two_equidistant_points_l1840_184030

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Configuration of a circle and two parallel lines -/
structure CircleLineConfiguration where
  circle : Circle
  line1 : Line
  line2 : Line
  d : ℝ
  h : d > circle.radius

/-- A point is equidistant from a circle and two parallel lines -/
def isEquidistant (p : Point) (config : CircleLineConfiguration) : Prop :=
  sorry

/-- The number of equidistant points -/
def numEquidistantPoints (config : CircleLineConfiguration) : ℕ :=
  sorry

/-- Theorem: There are exactly 2 equidistant points -/
theorem two_equidistant_points (config : CircleLineConfiguration) :
  numEquidistantPoints config = 2 :=
sorry

end NUMINAMATH_CALUDE_two_equidistant_points_l1840_184030


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1840_184041

theorem sufficient_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d → a * c + b * d > b * c + a * d) ∧
  ∃ a b c d : ℝ, a * c + b * d > b * c + a * d ∧ ¬(a > b ∧ c > d) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1840_184041


namespace NUMINAMATH_CALUDE_equal_playing_time_l1840_184025

theorem equal_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) % total_players = 0 →
  (players_on_field * match_duration) / total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_equal_playing_time_l1840_184025


namespace NUMINAMATH_CALUDE_right_triangle_roots_l1840_184036

theorem right_triangle_roots (p : ℝ) : 
  (∃ a b c : ℝ, 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (a^3 - 2*p*(p+1)*a^2 + (p^4 + 4*p^3 - 1)*a - 3*p^3 = 0) ∧
    (b^3 - 2*p*(p+1)*b^2 + (p^4 + 4*p^3 - 1)*b - 3*p^3 = 0) ∧
    (c^3 - 2*p*(p+1)*c^2 + (p^4 + 4*p^3 - 1)*c - 3*p^3 = 0) ∧
    (a^2 + b^2 = c^2)) ↔ 
  p = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_roots_l1840_184036


namespace NUMINAMATH_CALUDE_tangent_line_equation_curve_passes_through_point_l1840_184011

/-- The equation of the tangent line to the curve y = x^3 at the point (1,1) -/
theorem tangent_line_equation :
  ∃ (a b c : ℝ), (a * 1 + b * 1 + c = 0) ∧
  (∀ (x y : ℝ), y = x^3 → (x - 1)^2 + (y - 1)^2 ≤ (a * x + b * y + c)^2) ∧
  ((a = 3 ∧ b = -1 ∧ c = -2) ∨ (a = 3 ∧ b = -4 ∧ c = 1)) :=
by sorry

/-- The curve y = x^3 passes through the point (1,1) -/
theorem curve_passes_through_point :
  (1 : ℝ)^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_curve_passes_through_point_l1840_184011


namespace NUMINAMATH_CALUDE_house_painting_cost_l1840_184045

/-- Calculates the total cost of painting a house given the areas and costs per square foot for different rooms. -/
def total_painting_cost (living_room_area : ℕ) (living_room_cost : ℕ)
                        (bedroom_area : ℕ) (bedroom_cost : ℕ)
                        (kitchen_area : ℕ) (kitchen_cost : ℕ)
                        (bathroom_area : ℕ) (bathroom_cost : ℕ) : ℕ :=
  living_room_area * living_room_cost +
  2 * bedroom_area * bedroom_cost +
  kitchen_area * kitchen_cost +
  2 * bathroom_area * bathroom_cost

/-- Theorem stating that the total cost of painting the house is 49500 Rs. -/
theorem house_painting_cost :
  total_painting_cost 600 30 450 25 300 20 100 15 = 49500 := by
  sorry

#eval total_painting_cost 600 30 450 25 300 20 100 15

end NUMINAMATH_CALUDE_house_painting_cost_l1840_184045


namespace NUMINAMATH_CALUDE_bananas_purchased_is_96_l1840_184090

/-- The number of pounds of bananas purchased by the grocer -/
def bananas_purchased : ℝ := 96

/-- The purchase price in dollars for 3 pounds of bananas -/
def purchase_price : ℝ := 0.50

/-- The selling price in dollars for 4 pounds of bananas -/
def selling_price : ℝ := 1.00

/-- The total profit in dollars -/
def total_profit : ℝ := 8.00

/-- Theorem stating that the number of pounds of bananas purchased is 96 -/
theorem bananas_purchased_is_96 :
  bananas_purchased = 96 ∧
  purchase_price = 0.50 ∧
  selling_price = 1.00 ∧
  total_profit = 8.00 ∧
  (selling_price / 4 - purchase_price / 3) * bananas_purchased = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bananas_purchased_is_96_l1840_184090


namespace NUMINAMATH_CALUDE_mixture_quantity_proof_l1840_184077

theorem mixture_quantity_proof (petrol kerosene diesel : ℝ) 
  (h1 : petrol / kerosene = 3 / 2)
  (h2 : petrol / diesel = 3 / 5)
  (h3 : (petrol - 6) / ((kerosene - 4) + 20) = 2 / 3)
  (h4 : (petrol - 6) / (diesel - 10) = 2 / 5)
  (h5 : petrol + kerosene + diesel > 0) :
  petrol + kerosene + diesel = 100 := by
sorry

end NUMINAMATH_CALUDE_mixture_quantity_proof_l1840_184077


namespace NUMINAMATH_CALUDE_article_cost_price_l1840_184081

theorem article_cost_price (original_selling_price original_cost_price new_selling_price new_cost_price : ℝ) :
  original_selling_price = 1.25 * original_cost_price →
  new_cost_price = 0.8 * original_cost_price →
  new_selling_price = original_selling_price - 12.60 →
  new_selling_price = 1.3 * new_cost_price →
  original_cost_price = 60 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l1840_184081


namespace NUMINAMATH_CALUDE_salt_solution_replacement_l1840_184048

theorem salt_solution_replacement (original_salt_percentage : Real) 
  (replaced_fraction : Real) (final_salt_percentage : Real) 
  (replacing_salt_percentage : Real) : 
  original_salt_percentage = 13 →
  replaced_fraction = 1/4 →
  final_salt_percentage = 16 →
  (1 - replaced_fraction) * original_salt_percentage + 
    replaced_fraction * replacing_salt_percentage = final_salt_percentage →
  replacing_salt_percentage = 25 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_replacement_l1840_184048


namespace NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l1840_184083

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 0, p x) ↔ (∀ x > 0, ¬ p x) := by sorry

theorem square_positive_negation :
  (¬ ∃ x > 0, x^2 > 0) ↔ (∀ x > 0, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_square_positive_negation_l1840_184083


namespace NUMINAMATH_CALUDE_max_sundays_in_84_days_l1840_184082

theorem max_sundays_in_84_days : ℕ :=
  let days_in_period : ℕ := 84
  let days_in_week : ℕ := 7
  let sundays_per_week : ℕ := 1

  have h1 : days_in_period % days_in_week = 0 := by sorry
  have h2 : days_in_period / days_in_week * sundays_per_week = 12 := by sorry

  12

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_max_sundays_in_84_days_l1840_184082


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1840_184074

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1840_184074


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1840_184023

theorem product_of_repeating_decimal_and_eight :
  let s : ℚ := 456 / 999
  8 * s = 1216 / 333 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1840_184023


namespace NUMINAMATH_CALUDE_inequality_solution_subset_l1840_184006

theorem inequality_solution_subset (a : ℝ) :
  (∀ x : ℝ, x^2 < |x - 1| + a → -3 < x ∧ x < 3) →
  a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_subset_l1840_184006


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l1840_184059

/-- A function f is even if f(x) = f(-x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The distance between intersections of a function with a horizontal line -/
def IntersectionDistance (f : ℝ → ℝ) (y : ℝ) : ℝ → ℝ → ℝ :=
  λ x₁ x₂ => |x₂ - x₁|

theorem monotone_increasing_interval
  (ω φ : ℝ)
  (f : ℝ → ℝ)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (hf : f = λ x => 2 * Real.sin (ω * x + φ))
  (heven : EvenFunction f)
  (hmin : ∃ x₁ x₂, f x₁ = 2 ∧ f x₂ = 2 ∧ 
    ∀ y₁ y₂, f y₁ = 2 → f y₂ = 2 → 
    IntersectionDistance f 2 y₁ y₂ ≥ IntersectionDistance f 2 x₁ x₂ ∧
    IntersectionDistance f 2 x₁ x₂ = π) :
  StrictMonoOn f (Set.Ioo (-π/2) (-π/4)) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l1840_184059


namespace NUMINAMATH_CALUDE_unique_four_digit_division_l1840_184022

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_valid : a > 0 ∧ a < 10
  b_valid : b < 10
  c_valid : c < 10
  d_valid : d > 0 ∧ d < 10
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a FourDigitNumber to its numeric value -/
def FourDigitNumber.toNat (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Converts a three-digit number DBA to its numeric value -/
def threeDigitToNat (d b a : Nat) : Nat :=
  100 * d + 10 * b + a

theorem unique_four_digit_division :
  ∃! (n : FourDigitNumber), n.toNat / n.d = threeDigitToNat n.d n.b n.a :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_division_l1840_184022


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1840_184071

theorem inequality_equivalence (x y : ℝ) :
  (2 * y - 3 * x < Real.sqrt (9 * x^2 + 16)) ↔
  ((y < 4 * x ∧ x ≥ 0) ∨ (y < -x ∧ x < 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1840_184071


namespace NUMINAMATH_CALUDE_equation_proof_l1840_184040

theorem equation_proof : (12 : ℕ)^3 * 6^2 / 432 = 144 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1840_184040


namespace NUMINAMATH_CALUDE_pole_length_l1840_184078

/-- The length of a pole that fits diagonally in a rectangular opening -/
theorem pole_length (w h : ℝ) (hw : w > 0) (hh : h > 0) : 
  (w + 4)^2 + (h + 2)^2 = 100 → w^2 + h^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_pole_length_l1840_184078


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1840_184004

theorem min_value_quadratic_form (a b : ℝ) (h : 4 ≤ a^2 + b^2 ∧ a^2 + b^2 ≤ 9) :
  2 ≤ a^2 - a*b + b^2 ∧ ∃ (x y : ℝ), 4 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 9 ∧ x^2 - x*y + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1840_184004


namespace NUMINAMATH_CALUDE_probability_diamond_then_ace_is_one_fiftytwo_l1840_184026

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of diamond cards in a standard deck -/
def DiamondCards : ℕ := 13

/-- Represents the number of ace cards in a standard deck -/
def AceCards : ℕ := 4

/-- The probability of drawing a diamond as the first card and an ace as the second card -/
def probability_diamond_then_ace : ℚ :=
  (DiamondCards : ℚ) / StandardDeck * AceCards / (StandardDeck - 1)

theorem probability_diamond_then_ace_is_one_fiftytwo :
  probability_diamond_then_ace = 1 / StandardDeck :=
sorry

end NUMINAMATH_CALUDE_probability_diamond_then_ace_is_one_fiftytwo_l1840_184026


namespace NUMINAMATH_CALUDE_age_difference_is_28_l1840_184033

/-- The age difference between a man and his son -/
def ageDifference (sonAge manAge : ℕ) : ℕ := manAge - sonAge

/-- Prove that the age difference between a man and his son is 28 years -/
theorem age_difference_is_28 :
  ∃ (sonAge manAge : ℕ),
    sonAge = 26 ∧
    manAge + 2 = 2 * (sonAge + 2) ∧
    ageDifference sonAge manAge = 28 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_is_28_l1840_184033


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1840_184095

theorem sandwich_combinations :
  let meat_types : ℕ := 12
  let cheese_types : ℕ := 10
  let bread_types : ℕ := 5
  let meat_choice := 1
  let cheese_choice := 2
  let bread_choice := 1
  Nat.choose meat_types meat_choice *
  Nat.choose cheese_types cheese_choice *
  Nat.choose bread_types bread_choice = 2700 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1840_184095


namespace NUMINAMATH_CALUDE_solve_for_y_l1840_184012

theorem solve_for_y (x z : ℝ) (h1 : x^2 * z - x * z^2 = 6) (h2 : x = -2) (h3 : z = 1) : 
  ∃ y : ℝ, x^2 * y * z - x * y * z^2 = 6 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l1840_184012


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1840_184067

theorem sufficient_not_necessary (a : ℝ) :
  (a < -1 → ∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) ∧
  (∃ a : ℝ, a ≥ -1 ∧ ∃ x₀ : ℝ, a * Real.cos x₀ + 1 < 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1840_184067


namespace NUMINAMATH_CALUDE_point_on_terminal_side_l1840_184013

/-- Given a point P (x, 3) on the terminal side of angle θ where cos θ = -4/5, prove that x = -4 -/
theorem point_on_terminal_side (x : ℝ) (θ : ℝ) : 
  (∃ P : ℝ × ℝ, P = (x, 3) ∧ P.1 = x * Real.cos θ ∧ P.2 = x * Real.sin θ) → 
  Real.cos θ = -4/5 → 
  x = -4 := by
sorry

end NUMINAMATH_CALUDE_point_on_terminal_side_l1840_184013


namespace NUMINAMATH_CALUDE_ellianna_fat_served_l1840_184076

/-- The amount of fat in ounces for a herring -/
def herring_fat : ℕ := 40

/-- The amount of fat in ounces for an eel -/
def eel_fat : ℕ := 20

/-- The amount of fat in ounces for a pike -/
def pike_fat : ℕ := eel_fat + 10

/-- The number of fish of each type that Ellianna cooked and served -/
def fish_count : ℕ := 40

/-- The total amount of fat in ounces served by Ellianna -/
def total_fat : ℕ := fish_count * (herring_fat + eel_fat + pike_fat)

theorem ellianna_fat_served : total_fat = 3600 := by
  sorry

end NUMINAMATH_CALUDE_ellianna_fat_served_l1840_184076


namespace NUMINAMATH_CALUDE_train_speed_l1840_184057

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 700) (h2 : time = 40) :
  length / time = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1840_184057


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l1840_184093

theorem floor_equation_solutions :
  let S := {x : ℤ | ⌊(x : ℚ) / 2⌋ + ⌊(x : ℚ) / 4⌋ = x}
  S = {0, -3, -2, -5} := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l1840_184093


namespace NUMINAMATH_CALUDE_tan_theta_value_l1840_184098

theorem tan_theta_value (θ : Real) 
  (h : 2 * Real.sin (θ + π/3) = 3 * Real.sin (π/3 - θ)) : 
  Real.tan θ = Real.sqrt 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1840_184098


namespace NUMINAMATH_CALUDE_flowers_left_l1840_184035

theorem flowers_left (total : ℕ) (min_young : ℕ) (yoo_jeong : ℕ) 
  (h1 : total = 18) 
  (h2 : min_young = 5) 
  (h3 : yoo_jeong = 6) : 
  total - (min_young + yoo_jeong) = 7 := by
sorry

end NUMINAMATH_CALUDE_flowers_left_l1840_184035


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l1840_184079

/-- Calculates the final amount after two years of compound interest with different rates each year -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated -/
theorem compound_interest_calculation 
  (initial_amount : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (h1 : initial_amount = 8736) 
  (h2 : rate1 = 0.04) 
  (h3 : rate2 = 0.05) : 
  final_amount initial_amount rate1 rate2 = 9539.712 := by
  sorry

#eval final_amount 8736 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l1840_184079


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1840_184091

/-- A quadratic equation that is divisible by (x - 1) and has a constant term of 2 -/
def quadratic_equation (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem quadratic_equation_properties : 
  (∃ (q : ℝ → ℝ), ∀ x, quadratic_equation x = (x - 1) * q x) ∧ 
  (quadratic_equation 0 = 2) := by
  sorry

#check quadratic_equation_properties

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1840_184091


namespace NUMINAMATH_CALUDE_angle_expression_equality_l1840_184044

theorem angle_expression_equality (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (3 * Real.pi / 2 + θ) + Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_equality_l1840_184044


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l1840_184049

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ+) 
  (h : 72 ∣ n^2) : 
  ∃ (d : ℕ), d ∣ n ∧ d = 12 ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l1840_184049


namespace NUMINAMATH_CALUDE_max_length_sum_l1840_184097

/-- The length of a positive integer is the number of prime factors (not necessarily distinct) in its prime factorization. -/
def length (n : ℕ) : ℕ := sorry

/-- The maximum sum of lengths of x and y given the constraints. -/
theorem max_length_sum : 
  ∃ (x y : ℕ), 
    x > 1 ∧ 
    y > 1 ∧ 
    x + 3*y < 940 ∧ 
    ∀ (a b : ℕ), a > 1 → b > 1 → a + 3*b < 940 → length x + length y ≥ length a + length b ∧
    length x + length y = 15 :=
sorry

end NUMINAMATH_CALUDE_max_length_sum_l1840_184097


namespace NUMINAMATH_CALUDE_vacuum_tube_alignment_l1840_184046

theorem vacuum_tube_alignment :
  ∃ (f g : Fin 7 → Fin 7), 
    ∀ (r : Fin 7), ∃ (k : Fin 7), f k = g ((r + k) % 7) := by
  sorry

end NUMINAMATH_CALUDE_vacuum_tube_alignment_l1840_184046


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1840_184085

-- Define an odd function f on ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the condition for f when x ≥ 0
def fPositive (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x * (1 + x)

-- Theorem statement
theorem odd_function_negative_domain
  (f : ℝ → ℝ) (odd : isOddFunction f) (pos : fPositive f) :
  ∀ x, x < 0 → f x = x * (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1840_184085


namespace NUMINAMATH_CALUDE_length_of_AB_l1840_184003

-- Define the curves (M) and (N)
def curve_M (x y : ℝ) : Prop := x - y = 1

def curve_N (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_M A.1 A.2 ∧ curve_N A.1 A.2 ∧
  curve_M B.1 B.2 ∧ curve_N B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l1840_184003


namespace NUMINAMATH_CALUDE_convex_number_probability_l1840_184073

-- Define the set of digits
def Digits : Finset Nat := {1, 2, 3, 4}

-- Define a three-digit number
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  digit_in_range : hundreds ∈ Digits ∧ tens ∈ Digits ∧ units ∈ Digits

-- Define a convex number
def isConvex (n : ThreeDigitNumber) : Prop :=
  n.hundreds < n.tens ∧ n.tens > n.units

-- Define the set of all possible three-digit numbers
def allNumbers : Finset ThreeDigitNumber := sorry

-- Define the set of convex numbers
def convexNumbers : Finset ThreeDigitNumber := sorry

-- Theorem to prove
theorem convex_number_probability :
  (Finset.card convexNumbers : Rat) / (Finset.card allNumbers : Rat) = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_convex_number_probability_l1840_184073


namespace NUMINAMATH_CALUDE_three_sequence_comparison_l1840_184027

theorem three_sequence_comparison 
  (a b c : ℕ → ℕ) : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
sorry

end NUMINAMATH_CALUDE_three_sequence_comparison_l1840_184027


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l1840_184014

/-- Rachel's homework problem -/
theorem rachel_homework_difference :
  ∀ (math_pages reading_pages biology_pages : ℕ),
    math_pages = 9 →
    reading_pages = 2 →
    biology_pages = 96 →
    math_pages - reading_pages = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l1840_184014


namespace NUMINAMATH_CALUDE_min_value_of_y_l1840_184007

theorem min_value_of_y (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end NUMINAMATH_CALUDE_min_value_of_y_l1840_184007


namespace NUMINAMATH_CALUDE_problem_solution_l1840_184080

def f (a x : ℝ) := |a*x - 1| - (a - 1) * |x|

theorem problem_solution :
  (∀ x : ℝ, f 2 x > 2 ↔ x < -1 ∨ x > 3) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 1 2, f a x < a + 1) → a ≥ 2/5) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1840_184080


namespace NUMINAMATH_CALUDE_saras_sister_notebooks_l1840_184043

theorem saras_sister_notebooks (initial final ordered lost : ℕ) : 
  final = 8 → ordered = 6 → lost = 2 → initial + ordered - lost = final → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_saras_sister_notebooks_l1840_184043


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l1840_184029

/-- Given a compound where 3 moles have a molecular weight of 222,
    prove that the molecular weight of 1 mole is 74 g/mol. -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 222)
  (h2 : num_moles = 3) :
  total_weight / num_moles = 74 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l1840_184029


namespace NUMINAMATH_CALUDE_a_is_zero_l1840_184024

/-- If a and b are natural numbers such that for every natural number n, 
    2^n * a + b is a perfect square, then a = 0. -/
theorem a_is_zero (a b : ℕ) 
    (h : ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2) : 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_is_zero_l1840_184024


namespace NUMINAMATH_CALUDE_peach_multiple_l1840_184099

theorem peach_multiple (martine_peaches benjy_peaches gabrielle_peaches m : ℕ) : 
  martine_peaches = m * benjy_peaches + 6 →
  benjy_peaches = gabrielle_peaches / 3 →
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  m = 2 := by sorry

end NUMINAMATH_CALUDE_peach_multiple_l1840_184099


namespace NUMINAMATH_CALUDE_amy_hourly_rate_l1840_184064

/-- Calculates the hourly rate given total earnings, hours worked, and tips received. -/
def hourly_rate (total_earnings hours_worked tips : ℚ) : ℚ :=
  (total_earnings - tips) / hours_worked

/-- Proves that Amy's hourly rate is $2, given the conditions from the problem. -/
theorem amy_hourly_rate :
  let total_earnings : ℚ := 23
  let hours_worked : ℚ := 7
  let tips : ℚ := 9
  hourly_rate total_earnings hours_worked tips = 2 := by
  sorry

end NUMINAMATH_CALUDE_amy_hourly_rate_l1840_184064


namespace NUMINAMATH_CALUDE_inequality_proof_l1840_184052

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1840_184052


namespace NUMINAMATH_CALUDE_circle_distance_problem_l1840_184054

theorem circle_distance_problem (r₁ r₂ d : ℝ) (A B C : ℝ × ℝ) :
  r₁ = 13 →
  r₂ = 30 →
  d = 41 →
  let O₁ : ℝ × ℝ := (0, 0)
  let O₂ : ℝ × ℝ := (d, 0)
  (A.1 - O₂.1)^2 + A.2^2 = r₁^2 →
  A.1 > r₂ →
  (B.1 - O₂.1)^2 + B.2^2 = r₁^2 →
  (C.1 - O₁.1)^2 + C.2^2 = r₂^2 →
  B = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 12^2 * 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_problem_l1840_184054


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1840_184094

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 8*x + 9 = 0 ↔ (x + 4)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1840_184094


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1840_184008

theorem quadratic_inequality_empty_solution_set (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) ↔ a ∈ Set.Icc (-4 : ℝ) 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1840_184008


namespace NUMINAMATH_CALUDE_acorn_problem_l1840_184087

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
def total_acorns (shawna sheila danny : ℕ) : ℕ := shawna + sheila + danny

/-- Theorem stating the total number of acorns given the problem conditions -/
theorem acorn_problem (shawna sheila danny : ℕ) 
  (h1 : shawna = 7)
  (h2 : sheila = 5 * shawna)
  (h3 : danny = sheila + 3) :
  total_acorns shawna sheila danny = 80 := by
  sorry

end NUMINAMATH_CALUDE_acorn_problem_l1840_184087


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1840_184056

/-- Given a circle with center (4, 6) and one endpoint of a diameter at (2, 1),
    prove that the other endpoint of the diameter is at (6, 11). -/
theorem circle_diameter_endpoint (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  P = (4, 6) →  -- Center of the circle
  A = (2, 1) →  -- One endpoint of the diameter
  (P.1 - A.1 = B.1 - P.1 ∧ P.2 - A.2 = B.2 - P.2) →  -- B is symmetric to A with respect to P
  B = (6, 11) :=  -- The other endpoint of the diameter
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1840_184056


namespace NUMINAMATH_CALUDE_girls_on_playground_l1840_184009

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 117)
  (h2 : boys = 40) :
  total_children - boys = 77 := by
sorry

end NUMINAMATH_CALUDE_girls_on_playground_l1840_184009


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1840_184053

/-- Given two vectors a and b in ℝ², prove that if |a| = 3, |b| = 4, and the angle between them is 120°, then |a - b| = √13 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 3) 
  (h2 : ‖b‖ = 4) 
  (h3 : a.1 * b.1 + a.2 * b.2 = -6) : ‖a - b‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1840_184053


namespace NUMINAMATH_CALUDE_cora_reading_schedule_l1840_184037

/-- The number of pages Cora needs to read on Thursday to finish her book -/
def pages_to_read_thursday (total_pages : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) (pages_wednesday : ℕ) : ℕ :=
  let pages_thursday := (total_pages - pages_monday - pages_tuesday - pages_wednesday) / 3
  pages_thursday

theorem cora_reading_schedule :
  pages_to_read_thursday 158 23 38 61 = 12 := by
  sorry

#eval pages_to_read_thursday 158 23 38 61

end NUMINAMATH_CALUDE_cora_reading_schedule_l1840_184037
