import Mathlib

namespace NUMINAMATH_CALUDE_triangulation_theorem_l3127_312796

/-- A triangulation of a convex polygon with interior points. -/
structure Triangulation where
  /-- The number of vertices in the original polygon. -/
  polygon_vertices : ℕ
  /-- The number of additional interior points. -/
  interior_points : ℕ
  /-- The property that no three interior points are collinear. -/
  no_collinear_interior : Prop

/-- The number of triangles in a triangulation. -/
def num_triangles (t : Triangulation) : ℕ :=
  2 * (t.polygon_vertices + t.interior_points) - 2

/-- The main theorem about the number of triangles in the specific triangulation. -/
theorem triangulation_theorem (t : Triangulation) 
  (h1 : t.polygon_vertices = 1000)
  (h2 : t.interior_points = 500)
  (h3 : t.no_collinear_interior) :
  num_triangles t = 2998 := by
  sorry

end NUMINAMATH_CALUDE_triangulation_theorem_l3127_312796


namespace NUMINAMATH_CALUDE_triangle_properties_l3127_312703

open Real

theorem triangle_properties (A B C : ℝ) (R : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  R = 1 →
  (sin A - sin B + sin C) / sin C = sin B / (sin A + sin B - sin C) →
  ∃ (S : ℝ),
    A = π / 3 ∧
    S ≤ 3 * sqrt 3 / 4 ∧
    (∀ (S' : ℝ), S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3127_312703


namespace NUMINAMATH_CALUDE_trapezoid_area_l3127_312724

/-- The area of a trapezoid given its median line and height -/
theorem trapezoid_area (median_line height : ℝ) (h1 : median_line = 8) (h2 : height = 12) :
  median_line * height = 96 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3127_312724


namespace NUMINAMATH_CALUDE_petyas_race_time_l3127_312706

/-- Proves that Petya's actual time is greater than the planned time -/
theorem petyas_race_time (a V : ℝ) (h1 : a > 0) (h2 : V > 0) :
  a / V < a / (2.5 * V) + a / (1.6 * V) :=
by sorry

end NUMINAMATH_CALUDE_petyas_race_time_l3127_312706


namespace NUMINAMATH_CALUDE_theresa_video_games_l3127_312775

theorem theresa_video_games (tory julia theresa : ℕ) : 
  tory = 6 → 
  julia = tory / 3 → 
  theresa = 3 * julia + 5 → 
  theresa = 11 := by
sorry

end NUMINAMATH_CALUDE_theresa_video_games_l3127_312775


namespace NUMINAMATH_CALUDE_route_down_length_is_15_l3127_312783

/-- Represents a hiking trip up and down a mountain -/
structure HikingTrip where
  rateUp : ℝ        -- Rate of hiking up the mountain in miles per day
  timeUp : ℝ        -- Time taken to hike up in days
  rateDownFactor : ℝ -- Factor by which the rate down is faster than the rate up

/-- Calculates the length of the route down the mountain -/
def routeDownLength (trip : HikingTrip) : ℝ :=
  trip.rateUp * trip.rateDownFactor * trip.timeUp

/-- Theorem stating that for the given conditions, the route down is 15 miles long -/
theorem route_down_length_is_15 : 
  ∀ (trip : HikingTrip), 
  trip.rateUp = 5 ∧ 
  trip.timeUp = 2 ∧ 
  trip.rateDownFactor = 1.5 → 
  routeDownLength trip = 15 := by
  sorry


end NUMINAMATH_CALUDE_route_down_length_is_15_l3127_312783


namespace NUMINAMATH_CALUDE_circle_tangents_and_chord_l3127_312739

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the tangent lines
def tangent1 (x y : ℝ) : Prop := 4*x + 3*y - 23 = 0
def tangent2 (x : ℝ) : Prop := x = 5

-- Define the chord line
def chord_line (x y : ℝ) : Prop := x + y - 4 = 0

-- Theorem statement
theorem circle_tangents_and_chord :
  -- Part 1: Tangent lines
  (∀ x y, C x y → (tangent1 x y → x^2 + y^2 = 25)) ∧
  (∀ x y, C x y → (tangent2 x → x^2 + y^2 = 25)) ∧
  tangent1 5 1 ∧ tangent2 5 ∧
  -- Part 2: Chord line
  (∀ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ (x₁ + x₂)/2 = 3 ∧ (y₁ + y₂)/2 = 1 →
    chord_line x₁ y₁ ∧ chord_line x₂ y₂) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_and_chord_l3127_312739


namespace NUMINAMATH_CALUDE_max_y_value_l3127_312765

theorem max_y_value (x y : ℝ) (h : (x + y)^4 = x - y) :
  y ≤ 3 * Real.rpow 2 (1/3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l3127_312765


namespace NUMINAMATH_CALUDE_amusement_park_distance_l3127_312722

-- Define the speeds and time
def speed_A : ℝ := 3
def speed_B : ℝ := 4
def total_time : ℝ := 4

-- Define the distance functions
def distance_A (t : ℝ) : ℝ := speed_A * t
def distance_B (t : ℝ) : ℝ := speed_B * t

-- Theorem statement
theorem amusement_park_distance :
  ∃ (t_A t_B : ℝ),
    t_A + t_B = total_time ∧
    distance_B t_B = distance_A t_A + 2 ∧
    distance_B t_B = 8 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_distance_l3127_312722


namespace NUMINAMATH_CALUDE_simplify_expression_l3127_312754

theorem simplify_expression : 
  1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3127_312754


namespace NUMINAMATH_CALUDE_division_result_l3127_312750

theorem division_result : ∃ (result : ℚ), 
  (40 / 2 = result) ∧ 
  (40 + result + 2 = 62) ∧ 
  (result = 20) := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3127_312750


namespace NUMINAMATH_CALUDE_pizza_and_burgers_theorem_l3127_312768

/-- The number of pupils who like both pizza and burgers -/
def both_pizza_and_burgers (total : ℕ) (pizza : ℕ) (burgers : ℕ) : ℕ :=
  pizza + burgers - total

/-- Theorem: Given 200 total pupils, 125 who like pizza, and 115 who like burgers,
    40 pupils like both pizza and burgers. -/
theorem pizza_and_burgers_theorem :
  both_pizza_and_burgers 200 125 115 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pizza_and_burgers_theorem_l3127_312768


namespace NUMINAMATH_CALUDE_triangle_problem_l3127_312770

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (Real.tan t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B)) →
  (Real.sin (t.B - t.A) = Real.cos t.C) →
  (t.A = π/4 ∧ t.C = π/3) ∧
  (((1/2) * t.a * t.c * Real.sin t.B = 3 + Real.sqrt 3) →
   (t.a = 2 * Real.sqrt 2 ∧ t.c = 2 * Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l3127_312770


namespace NUMINAMATH_CALUDE_tangent_line_to_exp_and_ln_l3127_312778

theorem tangent_line_to_exp_and_ln (a b : ℝ) : 
  (∃ x₁ : ℝ, (x₁ + b = Real.exp x₁) ∧ (1 = Real.exp x₁)) →
  (∃ x₂ : ℝ, (x₂ + b = Real.log (x₂ + a)) ∧ (1 = 1 / (x₂ + a))) →
  a = 2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_exp_and_ln_l3127_312778


namespace NUMINAMATH_CALUDE_group_size_problem_l3127_312740

theorem group_size_problem (n : ℕ) (h : ℝ) : 
  (n : ℝ) * ((n : ℝ) * h) = 362525 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l3127_312740


namespace NUMINAMATH_CALUDE_work_completion_time_l3127_312735

theorem work_completion_time 
  (a b c : ℝ) 
  (h1 : a + b + c = 1/4)  -- a, b, and c together finish in 4 days
  (h2 : b = 1/18)         -- b alone finishes in 18 days
  (h3 : c = 1/6)          -- c alone finishes in 6 days
  : a = 1/36 :=           -- a alone finishes in 36 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3127_312735


namespace NUMINAMATH_CALUDE_tetrahedron_edge_length_l3127_312711

structure Tetrahedron where
  edges : Finset ℝ
  pq : ℝ
  rs : ℝ

def valid_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges.card = 6 ∧
  t.edges = {9, 15, 22, 28, 34, 39} ∧
  t.pq ∈ t.edges ∧
  t.rs ∈ t.edges ∧
  t.pq = 39

theorem tetrahedron_edge_length (t : Tetrahedron) (h : valid_tetrahedron t) : t.rs = 9 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_length_l3127_312711


namespace NUMINAMATH_CALUDE_sum_of_monomials_l3127_312757

-- Define the monomials
def monomial1 (x y : ℝ) (m : ℕ) := x^2 * y^m
def monomial2 (x y : ℝ) (n : ℕ) := x^n * y^3

-- Define the condition that the sum is a monomial
def sum_is_monomial (x y : ℝ) (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), ∀ x y, monomial1 x y m + monomial2 x y n = x^a * y^b

-- State the theorem
theorem sum_of_monomials (m n : ℕ) :
  (∀ x y : ℝ, sum_is_monomial x y m n) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_monomials_l3127_312757


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l3127_312737

theorem complex_arithmetic_proof :
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5*I
  let R : ℂ := 1 + I
  let T : ℂ := 3 - 4*I
  B * R + Q + T = 4 + I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l3127_312737


namespace NUMINAMATH_CALUDE_remainder_1998_pow_10_mod_10000_l3127_312710

theorem remainder_1998_pow_10_mod_10000 : 1998^10 % 10000 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1998_pow_10_mod_10000_l3127_312710


namespace NUMINAMATH_CALUDE_no_article_before_word_l3127_312709

-- Define the sentence structure
def sentence_structure : String := "They sent us ______ word of the latest happenings."

-- Define the function to determine the correct article
def correct_article : String := ""

-- Theorem statement
theorem no_article_before_word :
  correct_article = "" := by sorry

end NUMINAMATH_CALUDE_no_article_before_word_l3127_312709


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3127_312795

-- Define the shop dimensions
def shop_length : ℝ := 20
def shop_width : ℝ := 15

-- Define the monthly rent
def monthly_rent : ℝ := 3600

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem annual_rent_per_square_foot :
  let shop_area := shop_length * shop_width
  let annual_rent := monthly_rent * months_per_year
  annual_rent / shop_area = 144 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3127_312795


namespace NUMINAMATH_CALUDE_function_highest_points_omega_range_l3127_312764

/-- Given a function f(x) = 2sin(ωx + π/4) with ω > 0, if the graph of f(x) has exactly 3 highest points
    in the interval [0,1], then ω is in the range [17π/4, 25π/4). -/
theorem function_highest_points_omega_range (ω : ℝ) (h1 : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + π / 4)
  (∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 1) ∧
    (∀ y ∈ Set.Icc 0 1, ∃ x ∈ s, f y ≤ f x) ∧
    (∀ z ∉ s, z ∈ Set.Icc 0 1 → ∃ x ∈ s, f z < f x)) →
  17 * π / 4 ≤ ω ∧ ω < 25 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_function_highest_points_omega_range_l3127_312764


namespace NUMINAMATH_CALUDE_helga_shopping_items_l3127_312766

def shopping_trip (store1_shoes store1_bags : ℕ) : Prop :=
  let store2_shoes := 2 * store1_shoes
  let store2_bags := store1_bags + 6
  let store3_shoes := 0
  let store3_bags := 0
  let store4_shoes := store1_bags + store2_bags
  let store4_bags := 0
  let store5_shoes := store4_shoes / 2
  let store5_bags := 8
  let store6_shoes := Int.floor (Real.sqrt (store2_shoes + store5_shoes))
  let store6_bags := store1_bags + store2_bags + store5_bags + 5
  let total_shoes := store1_shoes + store2_shoes + store3_shoes + store4_shoes + store5_shoes + store6_shoes
  let total_bags := store1_bags + store2_bags + store3_bags + store4_bags + store5_bags + store6_bags
  total_shoes + total_bags = 95

theorem helga_shopping_items :
  shopping_trip 7 4 := by
  sorry

end NUMINAMATH_CALUDE_helga_shopping_items_l3127_312766


namespace NUMINAMATH_CALUDE_book_reading_theorem_l3127_312742

def book_reading_problem (total_pages : ℕ) (reading_rate : ℕ) (monday_hours : ℕ) (tuesday_hours : ℚ) : ℚ :=
  let pages_read := monday_hours * reading_rate + tuesday_hours * reading_rate
  let pages_left := total_pages - pages_read
  pages_left / reading_rate

theorem book_reading_theorem :
  book_reading_problem 248 16 3 (13/2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_theorem_l3127_312742


namespace NUMINAMATH_CALUDE_range_of_a_when_not_p_range_of_m_when_p_necessary_not_sufficient_l3127_312702

-- Define the propositions
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + a + 3 = 0
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Theorem 1: When ¬p is true, a ∈ (-2, 6)
theorem range_of_a_when_not_p :
  ∀ a : ℝ, ¬(p a) → -2 < a ∧ a < 6 :=
sorry

-- Theorem 2: When p is necessary but not sufficient for q, m ∈ (-∞, -3] ∪ [7, +∞)
theorem range_of_m_when_p_necessary_not_sufficient :
  ∀ m : ℝ, (∀ a : ℝ, q m a → p a) ∧ (∃ a : ℝ, p a ∧ ¬(q m a)) →
  m ≤ -3 ∨ m ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_not_p_range_of_m_when_p_necessary_not_sufficient_l3127_312702


namespace NUMINAMATH_CALUDE_remainder_theorem_l3127_312734

theorem remainder_theorem (n : ℤ) : (7 - 2*n + (n + 5)) % 5 = (-n + 2) % 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3127_312734


namespace NUMINAMATH_CALUDE_periodicity_theorem_l3127_312721

/-- A polynomial with rational coefficients -/
def RationalPolynomial := Polynomial ℚ

/-- A sequence of rational numbers -/
def RationalSequence := ℕ → ℚ

/-- The statement of the periodicity theorem -/
theorem periodicity_theorem
  (p : RationalPolynomial)
  (q : RationalSequence)
  (h1 : p.degree ≥ 2)
  (h2 : ∀ n : ℕ, n ≥ 1 → q n = p.eval (q (n + 1))) :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n ≥ 1 → q (n + k) = q n :=
sorry

end NUMINAMATH_CALUDE_periodicity_theorem_l3127_312721


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l3127_312716

/-- Calculates the remaining bottle caps after sharing. -/
def remaining_bottle_caps (start : ℕ) (shared : ℕ) : ℕ :=
  start - shared

/-- Proves that Marilyn ends up with 15 bottle caps. -/
theorem marilyn_bottle_caps : remaining_bottle_caps 51 36 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l3127_312716


namespace NUMINAMATH_CALUDE_bags_filled_on_saturday_l3127_312755

theorem bags_filled_on_saturday (bags_sunday : ℕ) (cans_per_bag : ℕ) (total_cans : ℕ) : 
  bags_sunday = 4 →
  cans_per_bag = 9 →
  total_cans = 63 →
  ∃ (bags_saturday : ℕ), 
    bags_saturday * cans_per_bag + bags_sunday * cans_per_bag = total_cans ∧
    bags_saturday = 3 := by
  sorry

end NUMINAMATH_CALUDE_bags_filled_on_saturday_l3127_312755


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l3127_312749

theorem cylinder_volume_ratio : 
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 10
  let cylinder_a_height : ℝ := rectangle_height
  let cylinder_a_circumference : ℝ := rectangle_width
  let cylinder_b_height : ℝ := rectangle_width
  let cylinder_b_circumference : ℝ := rectangle_height
  let cylinder_volume (h : ℝ) (c : ℝ) : ℝ := h * (c / (2 * π))^2 * π
  let volume_a := cylinder_volume cylinder_a_height cylinder_a_circumference
  let volume_b := cylinder_volume cylinder_b_height cylinder_b_circumference
  max volume_a volume_b / min volume_a volume_b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l3127_312749


namespace NUMINAMATH_CALUDE_quadratic_polynomial_unique_l3127_312717

theorem quadratic_polynomial_unique (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →
  q (-4) = 17 →
  q 1 = 2 →
  q 3 = 10 →
  ∀ x, q x = x^2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_unique_l3127_312717


namespace NUMINAMATH_CALUDE_mike_earnings_l3127_312704

def mower_blade_cost : ℕ := 10
def game_cost : ℕ := 8
def number_of_games : ℕ := 4

def total_money_earned : ℕ :=
  mower_blade_cost + number_of_games * game_cost

theorem mike_earnings : total_money_earned = 42 := by
  sorry

end NUMINAMATH_CALUDE_mike_earnings_l3127_312704


namespace NUMINAMATH_CALUDE_jason_car_count_l3127_312763

/-- The number of red cars counted by Jason -/
def red_cars : ℕ := sorry

/-- The number of green cars counted by Jason -/
def green_cars : ℕ := sorry

/-- The number of purple cars counted by Jason -/
def purple_cars : ℕ := 47

theorem jason_car_count :
  (green_cars = 4 * red_cars) ∧
  (red_cars > purple_cars) ∧
  (green_cars + red_cars + purple_cars = 312) ∧
  (red_cars - purple_cars = 6) :=
by sorry

end NUMINAMATH_CALUDE_jason_car_count_l3127_312763


namespace NUMINAMATH_CALUDE_parabola_min_area_sum_l3127_312798

/-- A parabola in the Cartesian plane -/
structure Parabola where
  eqn : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- A point lies on a parabola -/
def lies_on (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangle_area (p q r : ℝ × ℝ) : ℝ := sorry

theorem parabola_min_area_sum (p : Parabola) (A B : ℝ × ℝ) :
  p.eqn = fun x y ↦ y^2 = 2*x →
  lies_on p A →
  lies_on p B →
  dot_product A B = -1 →
  let F := focus p
  let O := (0, 0)
  ∃ (min : ℝ), min = Real.sqrt 2 / 2 ∧
    ∀ (X Y : ℝ × ℝ), lies_on p X → lies_on p Y → dot_product X Y = -1 →
      triangle_area O F X + triangle_area O F Y ≥ min :=
sorry

end NUMINAMATH_CALUDE_parabola_min_area_sum_l3127_312798


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3127_312751

theorem cubic_equation_solutions : 
  ∃! (s : Finset Int), 
    (∀ x ∈ s, (x^3 - x - 1)^2015 = 1) ∧ 
    (∀ x : Int, (x^3 - x - 1)^2015 = 1 → x ∈ s) ∧ 
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3127_312751


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l3127_312715

/-- The Razorback t-shirt shop problem -/
theorem razorback_tshirt_sales (price : ℕ) (arkansas_sales : ℕ) (texas_tech_revenue : ℕ) :
  price = 78 →
  arkansas_sales = 172 →
  texas_tech_revenue = 1092 →
  arkansas_sales + (texas_tech_revenue / price) = 186 :=
by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l3127_312715


namespace NUMINAMATH_CALUDE_investment_growth_l3127_312784

/-- Calculates the final amount of an investment after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Proves that $1500 invested at 3% interest for 21 years results in approximately $2709.17 -/
theorem investment_growth :
  let principal : ℝ := 1500
  let rate : ℝ := 0.03
  let years : ℕ := 21
  let final_amount := compound_interest principal rate years
  ∃ ε > 0, |final_amount - 2709.17| < ε :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_l3127_312784


namespace NUMINAMATH_CALUDE_regular_polygon_27_diagonals_has_9_sides_l3127_312777

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular polygon with 27 diagonals has 9 sides -/
theorem regular_polygon_27_diagonals_has_9_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 27 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_27_diagonals_has_9_sides_l3127_312777


namespace NUMINAMATH_CALUDE_power_inequality_l3127_312707

theorem power_inequality : 0.1^0.8 < 0.2^0.8 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3127_312707


namespace NUMINAMATH_CALUDE_loan_period_is_three_l3127_312728

/-- The period of a loan (in years) where:
  - A lends Rs. 3500 to B at 10% per annum
  - B lends Rs. 3500 to C at 11.5% per annum
  - B gains Rs. 157.5 -/
def loanPeriod : ℝ → Prop := λ T =>
  let principal : ℝ := 3500
  let rateAtoB : ℝ := 10
  let rateBtoC : ℝ := 11.5
  let bGain : ℝ := 157.5
  let interestAtoB : ℝ := principal * rateAtoB * T / 100
  let interestBtoC : ℝ := principal * rateBtoC * T / 100
  interestBtoC - interestAtoB = bGain

theorem loan_period_is_three : loanPeriod 3 := by
  sorry

end NUMINAMATH_CALUDE_loan_period_is_three_l3127_312728


namespace NUMINAMATH_CALUDE_triangular_coin_array_l3127_312736

theorem triangular_coin_array (N : ℕ) : (N * (N + 1)) / 2 = 3003 → N = 77 := by
  sorry

end NUMINAMATH_CALUDE_triangular_coin_array_l3127_312736


namespace NUMINAMATH_CALUDE_function_properties_l3127_312787

noncomputable def f (A ω φ B x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + B

theorem function_properties :
  ∀ (A ω φ B : ℝ),
  A > 0 → ω > 0 → 0 < φ → φ < π →
  f A ω φ B (π / 3) = 1 →
  f A ω φ B (π / 2 / ω - φ / ω) = 3 →
  ∃ (x : ℝ), ω * x + φ = 0 ∧
  ∃ (y : ℝ), ω * y + φ = π ∧
  ω * (7 * π / 12) + φ = 2 * π →
  A = 1 ∧ B = 2 ∧ ω = 2 ∧ φ = 5 * π / 6 ∧
  (∀ (x : ℝ), f A ω φ B x = f A ω φ B (-4 * π / 3 - x)) ∧
  (∀ (x : ℝ), f A ω φ B (x - 5 * π / 12) = 4 - f A ω φ B (x + 5 * π / 12)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3127_312787


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3127_312758

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  first_term : a 1 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  sum : ℕ → ℝ
  sum_def : ∀ n : ℕ, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The theorem to be proved -/
theorem arithmetic_sequence_sum_6 (seq : ArithmeticSequence) (h : seq.sum 4 = 20) :
  seq.sum 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_6_l3127_312758


namespace NUMINAMATH_CALUDE_remainder_3456_div_23_l3127_312776

theorem remainder_3456_div_23 : 3456 % 23 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3456_div_23_l3127_312776


namespace NUMINAMATH_CALUDE_logarithm_square_sum_l3127_312797

theorem logarithm_square_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a * b * c = 10^11) 
  (h2 : Real.log a * Real.log (b * c) + Real.log b * Real.log (c * a) + Real.log c * Real.log (a * b) = 40 * Real.log 10) : 
  Real.sqrt ((Real.log a)^2 + (Real.log b)^2 + (Real.log c)^2) = 9 * Real.log 10 := by
sorry

end NUMINAMATH_CALUDE_logarithm_square_sum_l3127_312797


namespace NUMINAMATH_CALUDE_telescope_visual_range_increase_l3127_312743

theorem telescope_visual_range_increase (original_range new_range : ℝ) 
  (h1 : original_range = 80)
  (h2 : new_range = 150) :
  ((new_range - original_range) / original_range) * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_increase_l3127_312743


namespace NUMINAMATH_CALUDE_james_bought_three_dirt_bikes_l3127_312792

/-- Calculates the number of dirt bikes James bought given the costs and total spent -/
def number_of_dirt_bikes (dirt_bike_cost off_road_cost registration_cost total_cost : ℕ) 
  (num_off_road : ℕ) : ℕ :=
  let total_off_road_cost := num_off_road * (off_road_cost + registration_cost)
  let remaining_cost := total_cost - total_off_road_cost
  remaining_cost / (dirt_bike_cost + registration_cost)

/-- Proves that James bought 3 dirt bikes given the problem conditions -/
theorem james_bought_three_dirt_bikes : 
  number_of_dirt_bikes 150 300 25 1825 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_bought_three_dirt_bikes_l3127_312792


namespace NUMINAMATH_CALUDE_power_sum_zero_l3127_312791

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2^(3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l3127_312791


namespace NUMINAMATH_CALUDE_quotient_digits_l3127_312762

def dividend (n : ℕ) : ℕ := 100 * n + 38

theorem quotient_digits :
  (∀ n : ℕ, n ≤ 7 → (dividend n) / 8 < 100) ∧
  (dividend 7) / 8 ≥ 10 ∧
  (∀ n : ℕ, n ≥ 8 → (dividend n) / 8 ≥ 100) ∧
  (dividend 8) / 8 < 1000 :=
sorry

end NUMINAMATH_CALUDE_quotient_digits_l3127_312762


namespace NUMINAMATH_CALUDE_olivias_cookies_l3127_312729

/-- Proves the number of oatmeal cookies given the conditions of the problem -/
theorem olivias_cookies (cookies_per_baggie : ℕ) (total_baggies : ℕ) (chocolate_chip_cookies : ℕ)
  (h1 : cookies_per_baggie = 9)
  (h2 : total_baggies = 6)
  (h3 : chocolate_chip_cookies = 13) :
  cookies_per_baggie * total_baggies - chocolate_chip_cookies = 41 := by
  sorry

end NUMINAMATH_CALUDE_olivias_cookies_l3127_312729


namespace NUMINAMATH_CALUDE_incorrect_statement_l3127_312782

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem incorrect_statement 
  (α β : Plane) (m n : Line) :
  ¬(∀ (α β : Plane) (m n : Line),
    perp m n → perpPlane n α → parallel n β → perpPlanes α β) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l3127_312782


namespace NUMINAMATH_CALUDE_pet_store_cages_l3127_312730

def total_cages (initial_puppies initial_adult_dogs initial_kittens : ℕ)
                (sold_puppies sold_adult_dogs sold_kittens : ℕ)
                (puppies_per_cage adult_dogs_per_cage kittens_per_cage : ℕ) : ℕ :=
  let remaining_puppies := initial_puppies - sold_puppies
  let remaining_adult_dogs := initial_adult_dogs - sold_adult_dogs
  let remaining_kittens := initial_kittens - sold_kittens
  let puppy_cages := (remaining_puppies + puppies_per_cage - 1) / puppies_per_cage
  let adult_dog_cages := (remaining_adult_dogs + adult_dogs_per_cage - 1) / adult_dogs_per_cage
  let kitten_cages := (remaining_kittens + kittens_per_cage - 1) / kittens_per_cage
  puppy_cages + adult_dog_cages + kitten_cages

theorem pet_store_cages : 
  total_cages 45 30 25 39 15 10 3 2 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3127_312730


namespace NUMINAMATH_CALUDE_first_act_clown_mobiles_l3127_312759

/-- The number of clowns in each clown mobile -/
def clowns_per_mobile : ℕ := 28

/-- The total number of clowns in all clown mobiles -/
def total_clowns : ℕ := 140

/-- The number of clown mobiles -/
def num_clown_mobiles : ℕ := total_clowns / clowns_per_mobile

theorem first_act_clown_mobiles : num_clown_mobiles = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_act_clown_mobiles_l3127_312759


namespace NUMINAMATH_CALUDE_digit_150_is_3_l3127_312719

/-- The decimal expansion of 5/37 has a repeating block of length 3 -/
def repeating_block_length : ℕ := 3

/-- The repeating block in the decimal expansion of 5/37 is [1, 3, 5] -/
def repeating_block : List ℕ := [1, 3, 5]

/-- The 150th digit after the decimal point in the decimal expansion of 5/37 -/
def digit_150 : ℕ := repeating_block[(150 - 1) % repeating_block_length]

theorem digit_150_is_3 : digit_150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_is_3_l3127_312719


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l3127_312745

theorem smallest_integer_in_set (n : ℤ) : 
  (7 * n + 21 > 4 * n) → (∀ m : ℤ, m < n → ¬(7 * m + 21 > 4 * m)) → n = -6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l3127_312745


namespace NUMINAMATH_CALUDE_rental_fee_calculation_l3127_312746

/-- Rental fee calculation for comic books -/
theorem rental_fee_calculation 
  (rental_fee_per_30min : ℕ) 
  (num_students : ℕ) 
  (num_books : ℕ) 
  (rental_duration_hours : ℕ) 
  (h1 : rental_fee_per_30min = 4000)
  (h2 : num_students = 6)
  (h3 : num_books = 4)
  (h4 : rental_duration_hours = 3)
  : (rental_fee_per_30min * (rental_duration_hours * 2) * num_books) / num_students = 16000 := by
  sorry

#check rental_fee_calculation

end NUMINAMATH_CALUDE_rental_fee_calculation_l3127_312746


namespace NUMINAMATH_CALUDE_exactly_two_correct_l3127_312744

-- Define the propositions
def prop1 : Prop := ∃ n : ℤ, ∀ m : ℤ, m < 0 → m ≤ n
def prop2 : Prop := ∃ n : ℤ, ∀ m : ℤ, n ≤ m
def prop3 : Prop := ∀ n : ℤ, n < 0 → n ≤ -1
def prop4 : Prop := ∀ n : ℤ, n > 0 → 1 ≤ n

-- Theorem stating that exactly two propositions are correct
theorem exactly_two_correct : 
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_l3127_312744


namespace NUMINAMATH_CALUDE_python_eating_theorem_l3127_312779

/-- The number of days in the given time period -/
def total_days : ℕ := 616

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The rate at which the python eats alligators (alligators per week) -/
def eating_rate : ℕ := 1

/-- The maximum number of alligators the python can eat in the given time period -/
def max_alligators_eaten : ℕ := total_days / days_per_week

theorem python_eating_theorem :
  max_alligators_eaten = eating_rate * (total_days / days_per_week) :=
by sorry

end NUMINAMATH_CALUDE_python_eating_theorem_l3127_312779


namespace NUMINAMATH_CALUDE_sum_of_integers_l3127_312786

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 2)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3127_312786


namespace NUMINAMATH_CALUDE_flywheel_rotation_l3127_312771

-- Define the angular displacement function
def φ (t : ℝ) : ℝ := 8 * t - 0.5 * t^2

-- Define the angular velocity function
def ω (t : ℝ) : ℝ := 8 - t

theorem flywheel_rotation (t : ℝ) :
  -- 1. The angular velocity is the derivative of the angular displacement
  (deriv φ) t = ω t ∧
  -- 2. The angular velocity at t = 3 seconds is 5 rad/s
  ω 3 = 5 ∧
  -- 3. The flywheel stops rotating at t = 8 seconds
  ω 8 = 0 := by
  sorry


end NUMINAMATH_CALUDE_flywheel_rotation_l3127_312771


namespace NUMINAMATH_CALUDE_stones_combine_l3127_312705

/-- Two natural numbers are similar if the larger is at most twice the smaller -/
def similar (a b : ℕ) : Prop := max a b ≤ 2 * min a b

/-- A step in the combining process -/
inductive CombineStep (n : ℕ)
  | combine (a b : ℕ) (h : a + b ≤ n) (hsim : similar a b) : CombineStep n

/-- A sequence of combining steps -/
def CombineSeq (n : ℕ) := List (CombineStep n)

/-- The result of applying a sequence of combining steps -/
def applySeq (n : ℕ) (seq : CombineSeq n) : List ℕ :=
  sorry

/-- The theorem stating that any number of single-stone piles can be combined into one pile -/
theorem stones_combine (n : ℕ) : 
  ∃ (seq : CombineSeq n), applySeq n seq = [n] :=
sorry

end NUMINAMATH_CALUDE_stones_combine_l3127_312705


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3127_312769

theorem polynomial_factorization (m : ℤ) : 
  (∃ (a b c d e f : ℤ), ∀ (x y : ℤ), 
    x^2 + 2*x*y + 2*x + m*y + 2*m = (a*x + b*y + c) * (d*x + e*y + f)) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3127_312769


namespace NUMINAMATH_CALUDE_coin_difference_l3127_312725

def coin_values : List ℕ := [5, 10, 25, 50]

def target_amount : ℕ := 75

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.all (λ x => x ∈ coin_values) ∧
  combination.sum = target_amount

def num_coins (combination : List ℕ) : ℕ := combination.length

theorem coin_difference :
  ∃ (max_combination min_combination : List ℕ),
    is_valid_combination max_combination ∧
    is_valid_combination min_combination ∧
    (∀ c, is_valid_combination c →
      num_coins c ≤ num_coins max_combination ∧
      num_coins c ≥ num_coins min_combination) ∧
    num_coins max_combination - num_coins min_combination = 13 :=
  sorry

end NUMINAMATH_CALUDE_coin_difference_l3127_312725


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l3127_312718

/-- Represents the money redistribution game with three players -/
structure MoneyGame where
  amy_initial : ℝ
  bob_initial : ℝ
  cal_initial : ℝ
  cal_final : ℝ

/-- The rules of the money redistribution game -/
def redistribute (game : MoneyGame) : Prop :=
  ∃ (amy_mid bob_mid cal_mid : ℝ),
    -- Amy's redistribution
    amy_mid + bob_mid + cal_mid = game.amy_initial + game.bob_initial + game.cal_initial ∧
    bob_mid = 2 * game.bob_initial ∧
    cal_mid = 2 * game.cal_initial ∧
    -- Bob's redistribution
    ∃ (amy_mid2 bob_mid2 cal_mid2 : ℝ),
      amy_mid2 + bob_mid2 + cal_mid2 = amy_mid + bob_mid + cal_mid ∧
      amy_mid2 = 2 * amy_mid ∧
      cal_mid2 = 2 * cal_mid ∧
      -- Cal's redistribution
      ∃ (amy_final bob_final : ℝ),
        amy_final + bob_final + game.cal_final = amy_mid2 + bob_mid2 + cal_mid2 ∧
        amy_final = 2 * amy_mid2 ∧
        bob_final = 2 * bob_mid2

/-- The theorem stating the total money after redistribution -/
theorem total_money_after_redistribution (game : MoneyGame)
    (h1 : game.cal_initial = 50)
    (h2 : game.cal_final = 100)
    (h3 : redistribute game) :
    game.amy_initial + game.bob_initial + game.cal_initial = 300 :=
  sorry

end NUMINAMATH_CALUDE_total_money_after_redistribution_l3127_312718


namespace NUMINAMATH_CALUDE_toys_gained_example_l3127_312752

/-- Calculates the number of toys' cost price gained in a sale -/
def toys_cost_price_gained (num_toys : ℕ) (selling_price : ℕ) (cost_price_per_toy : ℕ) : ℕ :=
  (selling_price - num_toys * cost_price_per_toy) / cost_price_per_toy

/-- The number of toys' cost price gained when selling 18 toys for Rs. 21000 with a cost price of Rs. 1000 per toy is 3 -/
theorem toys_gained_example : toys_cost_price_gained 18 21000 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_toys_gained_example_l3127_312752


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3127_312712

theorem arithmetic_sequence_first_term 
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : (30 : ℚ) / 2 * (a + (a + 29 * d)) = 600) -- Sum of first 30 terms
  (h2 : (30 : ℚ) / 2 * ((a + 30 * d) + (a + 59 * d)) = 2100) -- Sum of next 30 terms
  : a = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3127_312712


namespace NUMINAMATH_CALUDE_marble_count_proof_l3127_312726

/-- The smallest positive integer greater than 1 that leaves a remainder of 1 when divided by 6, 7, and 8 -/
def smallest_marble_count : ℕ := 169

/-- Proves that the smallest_marble_count satisfies the given conditions -/
theorem marble_count_proof :
  smallest_marble_count > 1 ∧
  smallest_marble_count % 6 = 1 ∧
  smallest_marble_count % 7 = 1 ∧
  smallest_marble_count % 8 = 1 ∧
  ∀ n : ℕ, n > 1 →
    (n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1) →
    n ≥ smallest_marble_count :=
by sorry

end NUMINAMATH_CALUDE_marble_count_proof_l3127_312726


namespace NUMINAMATH_CALUDE_delta_curve_circumscribed_triangle_height_l3127_312794

/-- A Δ-curve is a curve with the property that all equilateral triangles circumscribing it have the same height -/
class DeltaCurve (α : Type*) [MetricSpace α] where
  is_delta_curve : α → Prop

variable {α : Type*} [MetricSpace α]

/-- An equilateral triangle -/
structure EquilateralTriangle (α : Type*) [MetricSpace α] where
  points : Fin 3 → α
  is_equilateral : ∀ i j : Fin 3, dist (points i) (points j) = dist (points 0) (points 1)

/-- A point lies on a line -/
def PointOnLine (p : α) (l : Set α) : Prop := p ∈ l

/-- A triangle circumscribes a curve if each side of the triangle touches the curve at exactly one point -/
def Circumscribes (t : EquilateralTriangle α) (k : Set α) : Prop :=
  ∃ a b c : α, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧
    PointOnLine a {x | dist x (t.points 0) = dist x (t.points 1)} ∧
    PointOnLine b {x | dist x (t.points 1) = dist x (t.points 2)} ∧
    PointOnLine c {x | dist x (t.points 2) = dist x (t.points 0)}

/-- The height of an equilateral triangle -/
def Height (t : EquilateralTriangle α) : ℝ := sorry

/-- The main theorem -/
theorem delta_curve_circumscribed_triangle_height 
  (k : Set α) [DeltaCurve α] (t : EquilateralTriangle α) 
  (h_circumscribes : Circumscribes t k) :
  ∀ (t₁ : EquilateralTriangle α),
    (∃ a b c : α, a ∈ k ∧ b ∈ k ∧ c ∈ k ∧
      PointOnLine a {x | dist x (t₁.points 0) = dist x (t₁.points 1)} ∧
      PointOnLine b {x | dist x (t₁.points 1) = dist x (t₁.points 2)} ∧
      PointOnLine c {x | dist x (t₁.points 2) = dist x (t₁.points 0)}) →
    Height t₁ ≤ Height t :=
sorry

end NUMINAMATH_CALUDE_delta_curve_circumscribed_triangle_height_l3127_312794


namespace NUMINAMATH_CALUDE_tim_reading_time_l3127_312780

/-- Given that Tim spends 1 hour a day meditating and twice as much time reading,
    prove that he spends 14 hours a week reading. -/
theorem tim_reading_time (meditation_time : ℝ) (reading_time : ℝ) (days_in_week : ℕ) :
  meditation_time = 1 →
  reading_time = 2 * meditation_time →
  days_in_week = 7 →
  reading_time * days_in_week = 14 := by
  sorry

end NUMINAMATH_CALUDE_tim_reading_time_l3127_312780


namespace NUMINAMATH_CALUDE_cuboid_inequality_l3127_312701

theorem cuboid_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_unit_diagonal : a^2 + b^2 + c^2 = 1) : 
  4*a + 4*b + 4*c + 4*a*b + 4*a*c + 4*b*c + 4*a*b*c < 12 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_inequality_l3127_312701


namespace NUMINAMATH_CALUDE_second_crate_granola_weight_l3127_312708

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a crate given its dimensions -/
def crateVolume (d : CrateDimensions) : ℝ := d.height * d.width * d.length

/-- Represents the properties of the first crate -/
def firstCrate : CrateDimensions := {
  height := 4,
  width := 3,
  length := 6
}

/-- The weight of coffee the first crate can hold -/
def firstCrateWeight : ℝ := 72

/-- Represents the properties of the second crate -/
def secondCrate : CrateDimensions := {
  height := firstCrate.height * 1.5,
  width := firstCrate.width * 1.5,
  length := firstCrate.length
}

/-- Theorem stating that the second crate can hold 162 grams of granola -/
theorem second_crate_granola_weight :
  (crateVolume secondCrate / crateVolume firstCrate) * firstCrateWeight = 162 := by sorry

end NUMINAMATH_CALUDE_second_crate_granola_weight_l3127_312708


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l3127_312790

theorem certain_fraction_proof (x : ℚ) : 
  (2 / 5) / x = (7 / 15) / (1 / 2) → x = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l3127_312790


namespace NUMINAMATH_CALUDE_unique_solution_implies_prime_l3127_312738

theorem unique_solution_implies_prime (n : ℕ) :
  (∃! (x y : ℕ), (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / n) →
  Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_prime_l3127_312738


namespace NUMINAMATH_CALUDE_sequence_existence_l3127_312700

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ a : ℕ → ℝ, 
    (a (n + 1) = a 1) ∧ 
    (a (n + 2) = a 2) ∧ 
    (∀ i ∈ Finset.range n, a i * a (i + 1) + 1 = a (i + 2)))
  ↔ 
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_l3127_312700


namespace NUMINAMATH_CALUDE_base_12_division_remainder_l3127_312720

def base_12_to_decimal (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem base_12_division_remainder :
  (base_12_to_decimal 1534) % 9 = 2 := by
sorry

end NUMINAMATH_CALUDE_base_12_division_remainder_l3127_312720


namespace NUMINAMATH_CALUDE_prob_even_and_greater_than_10_l3127_312772

/-- Represents a wheel with even and odd numbers -/
structure Wheel where
  evenCount : ℕ
  oddCount : ℕ

/-- Calculates the probability of selecting an even number from a wheel -/
def probEven (w : Wheel) : ℚ :=
  w.evenCount / (w.evenCount + w.oddCount)

/-- Calculates the probability of selecting an odd number from a wheel -/
def probOdd (w : Wheel) : ℚ :=
  w.oddCount / (w.evenCount + w.oddCount)

/-- The wheels used in the problem -/
def wheelA : Wheel := ⟨3, 5⟩
def wheelB : Wheel := ⟨2, 6⟩

/-- The probability that the sum of selected numbers is even -/
def probEvenSum : ℚ :=
  probEven wheelA * probEven wheelB + probOdd wheelA * probOdd wheelB

/-- The conditional probability that an even sum is greater than 10 -/
def probGreaterThan10GivenEven : ℚ := 1/3

/-- The main theorem to prove -/
theorem prob_even_and_greater_than_10 :
  probEvenSum * probGreaterThan10GivenEven = 3/16 := by
  sorry


end NUMINAMATH_CALUDE_prob_even_and_greater_than_10_l3127_312772


namespace NUMINAMATH_CALUDE_abc_ordering_l3127_312733

theorem abc_ordering (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (heq : a^2 + c^2 = 2*b*c) :
  (b > a ∧ a > c) ∨ (c > b ∧ b > a) :=
sorry

end NUMINAMATH_CALUDE_abc_ordering_l3127_312733


namespace NUMINAMATH_CALUDE_simplify_expression_l3127_312760

theorem simplify_expression (x y : ℝ) :
  3 * x + 7 * x^2 + 4 * y - (5 - 3 * x - 7 * x^2 + 2 * y) = 14 * x^2 + 6 * x + 2 * y - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3127_312760


namespace NUMINAMATH_CALUDE_xiaojun_pen_refills_l3127_312799

/-- The number of pen refills Xiaojun bought -/
def num_pen_refills : ℕ := 2

/-- The cost of each pen refill in yuan -/
def pen_refill_cost : ℕ := 2

/-- The cost of each eraser in yuan (positive integer) -/
def eraser_cost : ℕ := 2

/-- The total amount spent in yuan -/
def total_spent : ℕ := 6

/-- The number of erasers Xiaojun bought -/
def num_erasers : ℕ := 1

theorem xiaojun_pen_refills :
  num_pen_refills = 2 ∧
  pen_refill_cost = 2 ∧
  eraser_cost > 0 ∧
  total_spent = 6 ∧
  num_pen_refills = 2 * num_erasers ∧
  total_spent = num_pen_refills * pen_refill_cost + num_erasers * eraser_cost :=
by sorry

#check xiaojun_pen_refills

end NUMINAMATH_CALUDE_xiaojun_pen_refills_l3127_312799


namespace NUMINAMATH_CALUDE_f_passes_through_point_f_has_max_at_one_f_is_unique_l3127_312773

/-- A quadratic function that passes through (2, -6) and has a maximum of -4 at x = 1 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 6

/-- The function f passes through the point (2, -6) -/
theorem f_passes_through_point : f 2 = -6 := by sorry

/-- The function f has a maximum value of -4 when x = 1 -/
theorem f_has_max_at_one :
  (∀ x, f x ≤ f 1) ∧ f 1 = -4 := by sorry

/-- The function f is the unique quadratic function satisfying the given conditions -/
theorem f_is_unique (g : ℝ → ℝ) :
  (g 2 = -6) →
  ((∀ x, g x ≤ g 1) ∧ g 1 = -4) →
  (∃ a b c, ∀ x, g x = a * x^2 + b * x + c) →
  (∀ x, g x = f x) := by sorry

end NUMINAMATH_CALUDE_f_passes_through_point_f_has_max_at_one_f_is_unique_l3127_312773


namespace NUMINAMATH_CALUDE_price_reduction_equality_l3127_312789

theorem price_reduction_equality (z : ℝ) (h : z > 0) : 
  ∃ x : ℝ, x > 0 ∧ x < 100 ∧ 
  (z * (1 - 15/100) * (1 - 15/100) = z * (1 - x/100)) ∧
  x = 27.75 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_equality_l3127_312789


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l3127_312723

def is_in_third_quadrant (α : Real) : Prop :=
  180 < α % 360 ∧ α % 360 ≤ 270

theorem angle_in_third_quadrant (k : Int) (α : Real) 
  (h1 : (4*k + 1 : Real) * 180 < α) 
  (h2 : α < (4*k + 1 : Real) * 180 + 60) :
  is_in_third_quadrant α :=
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l3127_312723


namespace NUMINAMATH_CALUDE_sin_alpha_minus_2pi_over_3_l3127_312761

theorem sin_alpha_minus_2pi_over_3 (α : ℝ) (h : Real.cos (π / 6 - α) = 2 / 3) :
  Real.sin (α - 2 * π / 3) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_2pi_over_3_l3127_312761


namespace NUMINAMATH_CALUDE_B_max_at_181_l3127_312731

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sequence B_k -/
def B (k : ℕ) : ℝ := (binomial 2000 k : ℝ) * (0.1 ^ k)

/-- The theorem stating that B_k is maximum when k = 181 -/
theorem B_max_at_181 : 
  ∀ k : ℕ, k ≤ 2000 → B k ≤ B 181 :=
sorry

end NUMINAMATH_CALUDE_B_max_at_181_l3127_312731


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3127_312741

/-- Given an ellipse with specified center, focus, and semi-major axis endpoint,
    prove that its semi-minor axis has length √8. -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (1, -2))
  (h_focus : focus = (1, -3))
  (h_semi_major : semi_major_endpoint = (1, 1)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l3127_312741


namespace NUMINAMATH_CALUDE_lindsey_squat_weight_l3127_312713

/-- The weight Lindsey will squat given exercise bands and a dumbbell -/
theorem lindsey_squat_weight 
  (num_bands : ℕ) 
  (resistance_per_band : ℕ) 
  (dumbbell_weight : ℕ) 
  (h1 : num_bands = 2)
  (h2 : resistance_per_band = 5)
  (h3 : dumbbell_weight = 10) :
  num_bands * resistance_per_band + dumbbell_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_squat_weight_l3127_312713


namespace NUMINAMATH_CALUDE_maximize_annual_average_profit_l3127_312781

/-- Represents the problem of maximizing annual average profit for equipment purchase --/
theorem maximize_annual_average_profit :
  let initial_cost : ℕ := 90000
  let first_year_cost : ℕ := 20000
  let annual_cost_increase : ℕ := 20000
  let annual_revenue : ℕ := 110000
  let total_cost (n : ℕ) : ℕ := initial_cost + n * first_year_cost + (n * (n - 1) * annual_cost_increase) / 2
  let total_revenue (n : ℕ) : ℕ := n * annual_revenue
  let total_profit (n : ℕ) : ℤ := (total_revenue n : ℤ) - (total_cost n : ℤ)
  let annual_average_profit (n : ℕ) : ℚ := (total_profit n : ℚ) / n
  ∀ m : ℕ, m > 0 → annual_average_profit 3 ≥ annual_average_profit m :=
by
  sorry


end NUMINAMATH_CALUDE_maximize_annual_average_profit_l3127_312781


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l3127_312732

theorem tan_double_angle_special_case (α : Real) 
  (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) : 
  Real.tan (2 * α) = -8/15 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l3127_312732


namespace NUMINAMATH_CALUDE_divisor_and_totient_properties_l3127_312747

/-- Sum of divisors function -/
def τ (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ (n : ℕ) : ℕ := sorry

theorem divisor_and_totient_properties (n : ℕ) :
  (n > 1 → φ n * τ n < n^2) ∧
  (φ n * τ n + 1 = n^2 ↔ Nat.Prime n) ∧
  ¬∃ (m : ℕ), φ m * τ m + 2023 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_and_totient_properties_l3127_312747


namespace NUMINAMATH_CALUDE_flour_needed_for_doubled_recipe_l3127_312727

theorem flour_needed_for_doubled_recipe 
  (original_recipe : ℕ) 
  (already_added : ℕ) 
  (h1 : original_recipe = 7)
  (h2 : already_added = 3) : 
  (2 * original_recipe) - already_added = 11 :=
by sorry

end NUMINAMATH_CALUDE_flour_needed_for_doubled_recipe_l3127_312727


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3127_312788

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 9 → x ≠ -4 →
    (7 * x + 3) / (x^2 - 5*x - 36) = (66/13) / (x - 9) + (25/13) / (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3127_312788


namespace NUMINAMATH_CALUDE_dice_probability_theorem_l3127_312785

def num_dice : ℕ := 15
def num_sides : ℕ := 6
def target_count : ℕ := 4

theorem dice_probability_theorem :
  (Nat.choose num_dice target_count * 5^(num_dice - target_count)) / 6^num_dice =
  (1365 * 5^11) / 6^15 := by sorry

end NUMINAMATH_CALUDE_dice_probability_theorem_l3127_312785


namespace NUMINAMATH_CALUDE_alyssa_total_games_l3127_312756

/-- The total number of soccer games Alyssa attends over three years -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Theorem stating that Alyssa will attend 39 games in total -/
theorem alyssa_total_games :
  total_games 11 13 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_total_games_l3127_312756


namespace NUMINAMATH_CALUDE_root_sum_quotient_l3127_312753

theorem root_sum_quotient (p p₁ p₂ a b : ℝ) : 
  (a^2 - a) * p + 2 * a + 7 = 0 →
  (b^2 - b) * p + 2 * b + 7 = 0 →
  a / b + b / a = 7 / 10 →
  (p₁^2 - p₁) * a + 2 * a + 7 = 0 →
  (p₁^2 - p₁) * b + 2 * b + 7 = 0 →
  (p₂^2 - p₂) * a + 2 * a + 7 = 0 →
  (p₂^2 - p₂) * b + 2 * b + 7 = 0 →
  p₁ / p₂ + p₂ / p₁ = 9.2225 := by
sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l3127_312753


namespace NUMINAMATH_CALUDE_imaginary_real_sum_imaginary_real_sum_proof_l3127_312793

theorem imaginary_real_sum : ℂ → Prop :=
  fun z : ℂ => 
    let a : ℝ := Complex.im (z⁻¹)
    let b : ℝ := Complex.re ((1 + Complex.I) ^ 2)
    a + b = -1

theorem imaginary_real_sum_proof : imaginary_real_sum Complex.I := by
  sorry

end NUMINAMATH_CALUDE_imaginary_real_sum_imaginary_real_sum_proof_l3127_312793


namespace NUMINAMATH_CALUDE_problem_1_l3127_312714

theorem problem_1 (x : ℝ) : (3*x + 1)*(3*x - 1) - (3*x + 1)^2 = -6*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3127_312714


namespace NUMINAMATH_CALUDE_units_digit_of_p_l3127_312774

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  p > 0 → 
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  units_digit (p + 1) = 7 →
  units_digit p = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l3127_312774


namespace NUMINAMATH_CALUDE_oldest_turner_child_age_l3127_312748

theorem oldest_turner_child_age 
  (num_children : ℕ) 
  (average_age : ℕ) 
  (younger_children_ages : List ℕ) :
  num_children = 4 →
  average_age = 9 →
  younger_children_ages = [6, 8, 11] →
  (List.sum younger_children_ages + 11) / num_children = average_age :=
by sorry

end NUMINAMATH_CALUDE_oldest_turner_child_age_l3127_312748


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3127_312767

-- Problem 1
theorem problem_1 (m n : ℝ) (hm : m ≠ 0) :
  (2 * m * n) / (3 * m^2) * (6 * m * n) / (5 * n) = 4 * n / 5 :=
sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x ≠ 0) (hy : x ≠ y) :
  (5 * x - 5 * y) / (3 * x^2 * y) * (9 * x * y^2) / (x^2 - y^2) = 15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem problem_3 (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  ((x^3 * y^2) / z)^2 * ((y * z) / x^2)^3 = y^7 * z :=
sorry

-- Problem 4
theorem problem_4 (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 2*x + y ≠ 0) (hxy2 : 4*x^2 - y^2 ≠ 0) :
  (4 * x^2 * y^2) / (2*x + y) * (4*x^2 + 4*x*y + y^2) / (2*x + y) / ((2*x*y * (2*x - y)) / (4*x^2 - y^2)) = 4*x^2*y + 2*x*y^2 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3127_312767
