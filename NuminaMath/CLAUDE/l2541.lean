import Mathlib

namespace NUMINAMATH_CALUDE_M_is_line_segment_l2541_254146

-- Define the set of points M(x,y) satisfying the equation
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.sqrt ((p.1 - 1)^2 + p.2^2) + Real.sqrt ((p.1 + 1)^2 + p.2^2) = 2}

-- Define the line segment between (-1,0) and (1,0)
def lineSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (2*t - 1, 0)}

-- Theorem stating that M is equal to the line segment
theorem M_is_line_segment : M = lineSegment := by sorry

end NUMINAMATH_CALUDE_M_is_line_segment_l2541_254146


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2541_254179

/-- Given a hyperbola with eccentricity 5/4 and semi-major axis length 4,
    prove that its equation is x²/16 - y²/9 = 1 --/
theorem hyperbola_equation (x y : ℝ) :
  let a : ℝ := 4
  let e : ℝ := 5/4
  let c : ℝ := e * a
  let b : ℝ := Real.sqrt (c^2 - a^2)
  (x^2 / a^2) - (y^2 / b^2) = 1 → x^2 / 16 - y^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2541_254179


namespace NUMINAMATH_CALUDE_expansion_terms_imply_n_12_l2541_254118

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem expansion_terms_imply_n_12 (x a : ℝ) (n : ℕ) :
  (binomial_coefficient n 3 * x^(n-3) * a^3 = 120) →
  (binomial_coefficient n 4 * x^(n-4) * a^4 = 360) →
  (binomial_coefficient n 5 * x^(n-5) * a^5 = 720) →
  n = 12 :=
sorry

end NUMINAMATH_CALUDE_expansion_terms_imply_n_12_l2541_254118


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2541_254163

/-- A domino is a 1x2 rectangle on the board -/
structure Domino where
  x : Fin 3000
  y : Fin 3000
  horizontal : Bool

/-- A color is represented by a number from 0 to 2 -/
def Color := Fin 3

/-- A coloring assigns a color to each domino -/
def Coloring := Domino → Color

/-- Two dominoes are neighbors if they share an edge -/
def are_neighbors (d1 d2 : Domino) : Prop :=
  sorry

/-- The number of dominoes with a given color in a coloring -/
def count_color (c : Coloring) (color : Color) : Nat :=
  sorry

/-- The number of neighbors of a domino with the same color -/
def same_color_neighbors (c : Coloring) (d : Domino) : Nat :=
  sorry

/-- The main theorem: there exists a valid coloring -/
theorem exists_valid_coloring :
  ∃ (c : Coloring),
    (∀ color : Color, count_color c color = 1500000) ∧
    (∀ d : Domino, same_color_neighbors c d ≤ 2) :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2541_254163


namespace NUMINAMATH_CALUDE_smallest_square_partition_l2541_254149

/-- A partition of a square into smaller squares -/
structure SquarePartition (n : ℕ) :=
  (num_40 : ℕ)  -- number of 40x40 squares
  (num_49 : ℕ)  -- number of 49x49 squares
  (valid : num_40 * 40 * 40 + num_49 * 49 * 49 = n * n)
  (both_present : num_40 > 0 ∧ num_49 > 0)

/-- The main theorem stating that 2000 is the smallest n that satisfies the conditions -/
theorem smallest_square_partition :
  (∃ (p : SquarePartition 2000), True) ∧
  (∀ n < 2000, ¬ ∃ (p : SquarePartition n), True) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_partition_l2541_254149


namespace NUMINAMATH_CALUDE_f_properties_l2541_254183

/-- The function f(x) = x³ - 5x² + 3x -/
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 10*x + 3

theorem f_properties :
  (f' 3 = 0) ∧ 
  (f 1 = -1) ∧
  (∀ x ∈ Set.Icc 2 4, f x ≥ -9) ∧
  (f 3 = -9) ∧
  (∀ x ∈ Set.Icc 2 4, f x ≤ -4) ∧
  (f 4 = -4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2541_254183


namespace NUMINAMATH_CALUDE_max_distance_on_circle_l2541_254114

-- Define the circle Ω
def Ω : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + y^2 - 2*x - 4*y = 0}

-- Define the points that the circle passes through
def origin : ℝ × ℝ := (0, 0)
def point1 : ℝ × ℝ := (2, 4)
def point2 : ℝ × ℝ := (3, 3)

-- Theorem statement
theorem max_distance_on_circle :
  origin ∈ Ω ∧ point1 ∈ Ω ∧ point2 ∈ Ω →
  ∃ (max_dist : ℝ),
    (∀ p ∈ Ω, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ max_dist) ∧
    (∃ q ∈ Ω, Real.sqrt ((q.1 - 0)^2 + (q.2 - 0)^2) = max_dist) ∧
    max_dist = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_l2541_254114


namespace NUMINAMATH_CALUDE_point_on_line_l2541_254100

/-- Given three points in a 2D plane, this function checks if they are collinear --/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem point_on_line : are_collinear (0, 4) (-6, 1) (4, 6) := by sorry

end NUMINAMATH_CALUDE_point_on_line_l2541_254100


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l2541_254133

/-- Given a parabola y = x^2 + 1, prove that the equation of the tangent line
    passing through the point (0,0) is either 2x - y = 0 or 2x + y = 0. -/
theorem parabola_tangent_line (x y : ℝ) :
  y = x^2 + 1 →
  (∃ (m : ℝ), y = m*x ∧ 0 = 0^2 + 1) →
  (y = 2*x ∨ y = -2*x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l2541_254133


namespace NUMINAMATH_CALUDE_vanessa_album_pictures_l2541_254125

/-- Represents the number of pictures in an album -/
def pictures_per_album (phone_pics camera_pics total_albums : ℕ) : ℚ :=
  (phone_pics + camera_pics : ℚ) / total_albums

/-- Theorem stating the number of pictures per album given the conditions -/
theorem vanessa_album_pictures :
  pictures_per_album 56 28 8 = 21/2 := by sorry

end NUMINAMATH_CALUDE_vanessa_album_pictures_l2541_254125


namespace NUMINAMATH_CALUDE_min_max_abs_polynomial_l2541_254102

open Real

theorem min_max_abs_polynomial :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 - x^3 * z| ≤ |x^2 - x^3 * y|) ∧
    |x^2 - x^3 * y| ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_polynomial_l2541_254102


namespace NUMINAMATH_CALUDE_spongebob_fries_sold_l2541_254157

/-- Calculates the number of large fries sold given the total earnings, 
    number of burgers sold, price per burger, and price per large fries. -/
def large_fries_sold (total_earnings : ℚ) (num_burgers : ℕ) (price_burger : ℚ) (price_fries : ℚ) : ℚ :=
  (total_earnings - num_burgers * price_burger) / price_fries

/-- Proves that Spongebob sold 12 large fries given the conditions -/
theorem spongebob_fries_sold : 
  large_fries_sold 78 30 2 (3/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_spongebob_fries_sold_l2541_254157


namespace NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l2541_254197

theorem cubic_factorization_sum_of_squares (a b c d e f : ℤ) : 
  (∀ x : ℚ, 729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210 := by
sorry

end NUMINAMATH_CALUDE_cubic_factorization_sum_of_squares_l2541_254197


namespace NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2541_254122

theorem max_value_of_exponential_difference (x : ℝ) : 5^x - 25^x ≤ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_exponential_difference_l2541_254122


namespace NUMINAMATH_CALUDE_floor_ceil_problem_l2541_254127

theorem floor_ceil_problem : (⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉) * 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_problem_l2541_254127


namespace NUMINAMATH_CALUDE_inclination_angle_range_l2541_254164

/-- Given a line with slope k in [-1, √3] and inclination angle α in [0, π),
    prove that the range of α is [0, π/3] ∪ [3π/4, π) -/
theorem inclination_angle_range (k α : ℝ) :
  k ∈ Set.Icc (-1) (Real.sqrt 3) →
  α ∈ Set.Ico 0 π →
  k = Real.tan α →
  α ∈ Set.Icc 0 (π / 3) ∪ Set.Ico (3 * π / 4) π :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l2541_254164


namespace NUMINAMATH_CALUDE_geometric_series_problem_l2541_254185

theorem geometric_series_problem (n : ℝ) : 
  let a₁ := 15
  let r₁ := 5 / 15
  let S₁ := a₁ / (1 - r₁)
  let a₂ := 15
  let r₂ := (5 + n) / 15
  let S₂ := a₂ / (1 - r₂)
  S₂ = 3 * S₁ → n = 20/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l2541_254185


namespace NUMINAMATH_CALUDE_find_C_l2541_254192

theorem find_C (A B C : ℕ) : A = 680 → A = B + 157 → B = C + 185 → C = 338 := by
  sorry

end NUMINAMATH_CALUDE_find_C_l2541_254192


namespace NUMINAMATH_CALUDE_max_product_l2541_254139

theorem max_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 8) :
  a * b ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4 * b₀ = 8 ∧ a₀ * b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_product_l2541_254139


namespace NUMINAMATH_CALUDE_circle_polar_equation_l2541_254173

/-- A circle in the polar coordinate system with center at (1,0) and passing through the pole -/
structure PolarCircle where
  /-- The radius of the circle as a function of the angle θ -/
  ρ : ℝ → ℝ

/-- The polar coordinate equation of the circle -/
def polar_equation (c : PolarCircle) : Prop :=
  ∀ θ : ℝ, c.ρ θ = 2 * Real.cos θ

/-- Theorem stating that the polar coordinate equation of a circle with center at (1,0) 
    and passing through the pole is ρ = 2cos θ -/
theorem circle_polar_equation :
  ∀ c : PolarCircle, polar_equation c :=
sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l2541_254173


namespace NUMINAMATH_CALUDE_consecutive_coin_tosses_l2541_254119

theorem consecutive_coin_tosses (p : ℝ) (h : p = 1 / 2) :
  p ^ 5 = 1 / 32 := by
sorry

end NUMINAMATH_CALUDE_consecutive_coin_tosses_l2541_254119


namespace NUMINAMATH_CALUDE_girls_insects_count_l2541_254154

/-- The number of insects collected by boys -/
def boys_insects : ℕ := 200

/-- The number of groups the class was divided into -/
def num_groups : ℕ := 4

/-- The number of insects each group received -/
def insects_per_group : ℕ := 125

/-- The number of insects collected by girls -/
def girls_insects : ℕ := num_groups * insects_per_group - boys_insects

theorem girls_insects_count : girls_insects = 300 := by
  sorry

end NUMINAMATH_CALUDE_girls_insects_count_l2541_254154


namespace NUMINAMATH_CALUDE_students_just_passed_l2541_254190

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 25 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent ≤ 1) :
  total - (total * (first_div_percent + second_div_percent)).floor = 63 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l2541_254190


namespace NUMINAMATH_CALUDE_paco_cookie_difference_l2541_254170

/-- The number of more salty cookies than sweet cookies eaten by Paco -/
def cookies_difference (initial_sweet initial_salty eaten_sweet eaten_salty : ℕ) : ℕ :=
  eaten_salty - eaten_sweet

/-- Theorem stating that Paco ate 13 more salty cookies than sweet cookies -/
theorem paco_cookie_difference :
  cookies_difference 40 25 15 28 = 13 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookie_difference_l2541_254170


namespace NUMINAMATH_CALUDE_book_pages_count_l2541_254174

def days_in_week : ℕ := 7
def first_period : ℕ := 4
def second_period : ℕ := 2
def last_day : ℕ := 1
def pages_per_day_first_period : ℕ := 42
def pages_per_day_second_period : ℕ := 50
def pages_last_day : ℕ := 30

theorem book_pages_count :
  first_period * pages_per_day_first_period +
  second_period * pages_per_day_second_period +
  pages_last_day = 298 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l2541_254174


namespace NUMINAMATH_CALUDE_square_diagonal_l2541_254177

theorem square_diagonal (s : Real) (h : s > 0) (area_eq : s * s = 8) :
  Real.sqrt (2 * s * s) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_l2541_254177


namespace NUMINAMATH_CALUDE_select_blocks_count_l2541_254186

/-- The number of ways to select 4 blocks from a 6x6 grid such that no two blocks are in the same row or column -/
def select_blocks : ℕ := (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

/-- Theorem stating that the number of ways to select 4 blocks from a 6x6 grid
    such that no two blocks are in the same row or column is 5400 -/
theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end NUMINAMATH_CALUDE_select_blocks_count_l2541_254186


namespace NUMINAMATH_CALUDE_cars_without_features_l2541_254131

theorem cars_without_features (total : ℕ) (airbags : ℕ) (power_windows : ℕ) (both : ℕ)
  (h_total : total = 65)
  (h_airbags : airbags = 45)
  (h_power_windows : power_windows = 30)
  (h_both : both = 12) :
  total - (airbags + power_windows - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cars_without_features_l2541_254131


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2541_254167

theorem sphere_radius_ratio (V₁ V₂ : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : V₁ = (4 / 3) * π * r₁^3)
  (h₂ : V₂ = (4 / 3) * π * r₂^3)
  (h₃ : V₁ = 432 * π)
  (h₄ : V₂ = 0.25 * V₁) :
  r₂ / r₁ = 1 / Real.rpow 3 (1/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2541_254167


namespace NUMINAMATH_CALUDE_exactly_three_correct_delivery_l2541_254166

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- The probability of exactly k out of n packages being delivered correctly -/
def prob_correct_delivery (n k : ℕ) : ℚ :=
  (n.choose k * (n - k).factorial) / n.factorial

theorem exactly_three_correct_delivery :
  prob_correct_delivery n k = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_exactly_three_correct_delivery_l2541_254166


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l2541_254103

/-- The ratio of senior class size to junior class size -/
def class_ratio (senior_size junior_size : ℚ) : ℚ := senior_size / junior_size

theorem senior_junior_ratio 
  (senior_size junior_size : ℚ)
  (h1 : senior_size > 0)
  (h2 : junior_size > 0)
  (h3 : ∃ k : ℚ, k > 0 ∧ senior_size = k * junior_size)
  (h4 : (3/8) * senior_size + (1/4) * junior_size = (1/3) * (senior_size + junior_size)) :
  class_ratio senior_size junior_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l2541_254103


namespace NUMINAMATH_CALUDE_playground_girls_l2541_254156

theorem playground_girls (total_children : ℕ) (boys : ℕ) (girls : ℕ) :
  total_children = 63 → boys = 35 → girls = total_children - boys → girls = 28 := by
  sorry

end NUMINAMATH_CALUDE_playground_girls_l2541_254156


namespace NUMINAMATH_CALUDE_school_students_problem_l2541_254115

theorem school_students_problem (total : ℕ) (x : ℕ) : 
  total = 1150 →
  (total - x : ℚ) = (x : ℚ) * (total : ℚ) / 100 →
  x = 92 := by
sorry

end NUMINAMATH_CALUDE_school_students_problem_l2541_254115


namespace NUMINAMATH_CALUDE_exists_square_function_l2541_254123

theorem exists_square_function : ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_square_function_l2541_254123


namespace NUMINAMATH_CALUDE_complex_product_simplification_l2541_254136

theorem complex_product_simplification (x y : ℝ) :
  let i := Complex.I
  (x + i * y + 1) * (x - i * y + 1) = (x + 1)^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l2541_254136


namespace NUMINAMATH_CALUDE_impossible_exact_usage_l2541_254143

theorem impossible_exact_usage (p q r : ℕ) : 
  ¬∃ (x y z : ℤ), (2*x + 2*z = 2*p + 2*r + 2) ∧ 
                   (2*x + y = 2*p + q + 1) ∧ 
                   (y + z = q + r) :=
sorry

end NUMINAMATH_CALUDE_impossible_exact_usage_l2541_254143


namespace NUMINAMATH_CALUDE_steve_snack_shack_cost_l2541_254120

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem steve_snack_shack_cost : total_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_steve_snack_shack_cost_l2541_254120


namespace NUMINAMATH_CALUDE_complement_of_B_in_U_l2541_254188

open Set

theorem complement_of_B_in_U (U A B : Set ℕ) : 
  U = A ∪ B → 
  A = {1, 2, 3} → 
  A ∩ B = {1} → 
  U \ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_U_l2541_254188


namespace NUMINAMATH_CALUDE_negation_existential_square_positive_l2541_254117

theorem negation_existential_square_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existential_square_positive_l2541_254117


namespace NUMINAMATH_CALUDE_fixed_points_are_corresponding_l2541_254155

/-- A type representing a geometric figure -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Similarity transformation between two figures -/
def similarity (F1 F2 : Figure) : Prop :=
  sorry

/-- A point in a geometric figure -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Defines if a point is fixed under a similarity transformation -/
def is_fixed_point (p : Point) (F1 F2 : Figure) : Prop :=
  sorry

/-- Defines if two points are corresponding in similar figures -/
def are_corresponding (p1 p2 : Point) (F1 F2 : Figure) : Prop :=
  sorry

/-- Main theorem: Fixed points of three similar figures are their corresponding points -/
theorem fixed_points_are_corresponding 
  (F1 F2 F3 : Figure) 
  (h1 : similarity F1 F2) 
  (h2 : similarity F2 F3) 
  (h3 : similarity F3 F1) 
  (p1 : Point) 
  (p2 : Point) 
  (p3 : Point) 
  (hf1 : is_fixed_point p1 F1 F2) 
  (hf2 : is_fixed_point p2 F2 F3) 
  (hf3 : is_fixed_point p3 F3 F1) : 
  are_corresponding p1 p2 F1 F2 ∧ 
  are_corresponding p2 p3 F2 F3 ∧ 
  are_corresponding p3 p1 F3 F1 :=
sorry

end NUMINAMATH_CALUDE_fixed_points_are_corresponding_l2541_254155


namespace NUMINAMATH_CALUDE_alyssa_future_games_l2541_254129

/-- The number of soccer games Alyssa plans to attend next year -/
def games_next_year (games_this_year games_last_year total_games : ℕ) : ℕ :=
  total_games - (games_this_year + games_last_year)

/-- Theorem stating that Alyssa plans to attend 15 soccer games next year -/
theorem alyssa_future_games : 
  games_next_year 11 13 39 = 15 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_future_games_l2541_254129


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2541_254104

/-- Given that Paul gave away 213 crayons and lost 16 crayons,
    prove that the total number of crayons lost or given away is 229. -/
theorem crayons_lost_or_given_away :
  let crayons_given_away : ℕ := 213
  let crayons_lost : ℕ := 16
  crayons_given_away + crayons_lost = 229 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2541_254104


namespace NUMINAMATH_CALUDE_maddie_bought_two_white_packs_l2541_254144

/-- Represents the problem of determining the number of packs of white T-shirts Maddie bought. -/
def maddies_tshirt_problem (white_packs : ℕ) : Prop :=
  let blue_packs : ℕ := 4
  let white_per_pack : ℕ := 5
  let blue_per_pack : ℕ := 3
  let cost_per_shirt : ℕ := 3
  let total_spent : ℕ := 66
  
  (white_packs * white_per_pack + blue_packs * blue_per_pack) * cost_per_shirt = total_spent

/-- Theorem stating that Maddie bought 2 packs of white T-shirts. -/
theorem maddie_bought_two_white_packs : ∃ (white_packs : ℕ), white_packs = 2 ∧ maddies_tshirt_problem white_packs :=
sorry

end NUMINAMATH_CALUDE_maddie_bought_two_white_packs_l2541_254144


namespace NUMINAMATH_CALUDE_min_value_theorem_l2541_254140

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 25 / x ≥ 10 ∧ (x + 25 / x = 10 ↔ x = 5) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2541_254140


namespace NUMINAMATH_CALUDE_four_row_lattice_triangles_l2541_254162

/-- Represents a modified triangular lattice with n rows -/
structure ModifiedTriangularLattice (n : ℕ) where
  -- Each row i has i dots, with the base row having n dots
  rows : Fin n → ℕ
  rows_def : ∀ i : Fin n, rows i = i.val + 1

/-- Counts the number of triangles in a modified triangular lattice -/
def countTriangles (n : ℕ) : ℕ :=
  let lattice := ModifiedTriangularLattice n
  -- The actual counting logic would go here
  0 -- Placeholder

/-- The theorem stating that a 4-row modified triangular lattice contains 22 triangles -/
theorem four_row_lattice_triangles :
  countTriangles 4 = 22 := by
  sorry

#check four_row_lattice_triangles

end NUMINAMATH_CALUDE_four_row_lattice_triangles_l2541_254162


namespace NUMINAMATH_CALUDE_total_points_noa_and_phillip_l2541_254134

/-- 
Given that Noa scored 30 points and Phillip scored twice as many points as Noa,
prove that the total number of points scored by Noa and Phillip is 90.
-/
theorem total_points_noa_and_phillip : 
  let noa_points : ℕ := 30
  let phillip_points : ℕ := 2 * noa_points
  noa_points + phillip_points = 90 := by
sorry


end NUMINAMATH_CALUDE_total_points_noa_and_phillip_l2541_254134


namespace NUMINAMATH_CALUDE_bc_is_one_eighth_of_ad_l2541_254128

/-- Given a line segment AD with points B and C on it,
    prove that BC is 1/8 of AD if AB is 3 times BD and AC is 7 times CD -/
theorem bc_is_one_eighth_of_ad 
  (A B C D : EuclideanSpace ℝ (Fin 1))
  (h_B_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ B = (1 - t) • A + t • D)
  (h_C_on_AD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ C = (1 - s) • A + s • D)
  (h_AB_3BD : dist A B = 3 * dist B D)
  (h_AC_7CD : dist A C = 7 * dist C D) :
  dist B C = (1 / 8 : ℝ) * dist A D :=
sorry

end NUMINAMATH_CALUDE_bc_is_one_eighth_of_ad_l2541_254128


namespace NUMINAMATH_CALUDE_max_groups_is_two_l2541_254198

/-- Represents the number of boys in the class -/
def num_boys : ℕ := 20

/-- Represents the number of girls in the class -/
def num_girls : ℕ := 24

/-- Represents the total number of students in the class -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of age groups -/
def num_age_groups : ℕ := 3

/-- Represents the number of skill levels -/
def num_skill_levels : ℕ := 3

/-- Represents the maximum number of groups that can be formed -/
def max_groups : ℕ := 2

/-- Theorem stating that the maximum number of groups is 2 -/
theorem max_groups_is_two :
  (num_boys % max_groups = 0) ∧
  (num_girls % max_groups = 0) ∧
  (total_students % max_groups = 0) ∧
  (max_groups % num_age_groups = 0) ∧
  (max_groups % num_skill_levels = 0) ∧
  (∀ n : ℕ, n > max_groups →
    (num_boys % n ≠ 0) ∨
    (num_girls % n ≠ 0) ∨
    (total_students % n ≠ 0) ∨
    (n % num_age_groups ≠ 0) ∨
    (n % num_skill_levels ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_groups_is_two_l2541_254198


namespace NUMINAMATH_CALUDE_homework_questions_l2541_254151

theorem homework_questions (first_hour second_hour third_hour : ℕ) : 
  third_hour = 132 → 
  third_hour = 2 * second_hour → 
  third_hour = 3 * first_hour → 
  first_hour + second_hour + third_hour = 264 :=
by
  sorry

end NUMINAMATH_CALUDE_homework_questions_l2541_254151


namespace NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l2541_254101

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_3x_squared_minus_75_l2541_254101


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l2541_254195

/-- Converts a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ := sorry

/-- Converts a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String := sorry

theorem binary_multiplication_division :
  let a := binary_to_nat "1011010"
  let b := binary_to_nat "1010100"
  let c := binary_to_nat "100"
  let result := binary_to_nat "110001111100"
  (a / c) * b = result := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l2541_254195


namespace NUMINAMATH_CALUDE_second_grade_selection_theorem_l2541_254113

/-- Represents the school population --/
structure School :=
  (total_students : ℕ)
  (first_grade_male_prob : ℝ)

/-- Represents the sampling method --/
structure Sampling :=
  (total_volunteers : ℕ)
  (method : String)

/-- Calculates the number of students selected from the second grade --/
def second_grade_selection (s : School) (samp : Sampling) : ℕ :=
  sorry

theorem second_grade_selection_theorem (s : School) (samp : Sampling) :
  s.total_students = 4000 →
  s.first_grade_male_prob = 0.2 →
  samp.total_volunteers = 100 →
  samp.method = "stratified" →
  second_grade_selection s samp = 30 :=
sorry

end NUMINAMATH_CALUDE_second_grade_selection_theorem_l2541_254113


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l2541_254182

theorem sin_cos_sum_equals_quarter : 
  Real.sin (20 * π / 180) * Real.cos (70 * π / 180) + 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) = 1/4 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l2541_254182


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l2541_254193

/-- Craig's work cycle in days -/
def craig_cycle : ℕ := 6

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 6

/-- Number of days Craig works in his cycle -/
def craig_work_days : ℕ := 4

/-- Number of days Dana works in her cycle -/
def dana_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 1000

/-- The number of days both Craig and Dana have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / craig_cycle

theorem coinciding_rest_days_count :
  coinciding_rest_days = 166 := by sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l2541_254193


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2541_254145

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^4 → 
  (a : ℤ) + (b : ℤ) + (c : ℤ) = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2541_254145


namespace NUMINAMATH_CALUDE_solve_simultaneous_equations_l2541_254107

theorem solve_simultaneous_equations :
  ∀ x y : ℝ,
  (x / 5 + 7 = y / 4 - 7) →
  (x / 3 - 4 = y / 2 + 4) →
  (x = -660 ∧ y = -472) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_simultaneous_equations_l2541_254107


namespace NUMINAMATH_CALUDE_quadratic_roots_fourth_power_sum_l2541_254112

/-- For a quadratic equation x² - 2ax - 1/a² = 0 with roots x₁ and x₂,
    prove that x₁⁴ + x₂⁴ = 16 + 8√2 if and only if a = ± ∛∛(1/8) -/
theorem quadratic_roots_fourth_power_sum (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*a*x₁ - 1/a^2 = 0 → 
  x₂^2 - 2*a*x₂ - 1/a^2 = 0 → 
  (x₁^4 + x₂^4 = 16 + 8*Real.sqrt 2) ↔ 
  (a = Real.rpow (1/8) (1/8) ∨ a = -Real.rpow (1/8) (1/8)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_fourth_power_sum_l2541_254112


namespace NUMINAMATH_CALUDE_height_relation_l2541_254138

/-- Two right circular cylinders with equal volumes and related radii -/
structure TwoCylinders where
  r1 : ℝ  -- radius of the first cylinder
  h1 : ℝ  -- height of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h2 : ℝ  -- height of the second cylinder
  r1_pos : 0 < r1  -- r1 is positive
  h1_pos : 0 < h1  -- h1 is positive
  r2_pos : 0 < r2  -- r2 is positive
  h2_pos : 0 < h2  -- h2 is positive
  volume_eq : r1^2 * h1 = r2^2 * h2  -- volumes are equal
  radius_relation : r2 = 1.2 * r1  -- second radius is 20% more than the first

/-- The height of the first cylinder is 44% more than the height of the second cylinder -/
theorem height_relation (c : TwoCylinders) : c.h1 = 1.44 * c.h2 := by
  sorry


end NUMINAMATH_CALUDE_height_relation_l2541_254138


namespace NUMINAMATH_CALUDE_max_intersected_cells_8x8_l2541_254106

/-- Represents a chessboard with a given number of rows and columns. -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a straight line on a chessboard. -/
structure StraightLine

/-- The number of cells intersected by a straight line on a chessboard. -/
def intersectedCells (board : Chessboard) (line : StraightLine) : Nat :=
  sorry

/-- The maximum number of cells that can be intersected by any straight line on a given chessboard. -/
def maxIntersectedCells (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating that the maximum number of cells intersected by a straight line on an 8x8 chessboard is 15. -/
theorem max_intersected_cells_8x8 :
  maxIntersectedCells (Chessboard.mk 8 8) = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_intersected_cells_8x8_l2541_254106


namespace NUMINAMATH_CALUDE_art_students_count_l2541_254196

theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 40)
  (h3 : both = 10)
  (h4 : neither = 450) :
  ∃ art : ℕ, art = total - (music - both) - both - neither :=
by
  sorry

end NUMINAMATH_CALUDE_art_students_count_l2541_254196


namespace NUMINAMATH_CALUDE_sandy_marks_l2541_254116

/-- Sandy's marks calculation -/
theorem sandy_marks :
  ∀ (total_sums correct_sums : ℕ)
    (marks_per_correct marks_lost_per_incorrect : ℕ),
  total_sums = 30 →
  correct_sums = 23 →
  marks_per_correct = 3 →
  marks_lost_per_incorrect = 2 →
  (marks_per_correct * correct_sums) -
  (marks_lost_per_incorrect * (total_sums - correct_sums)) = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_l2541_254116


namespace NUMINAMATH_CALUDE_company_merger_profit_distribution_l2541_254165

theorem company_merger_profit_distribution (company_a_profit company_b_profit : ℝ) 
  (company_a_percentage : ℝ) :
  company_a_profit = 90000 ∧ 
  company_b_profit = 60000 ∧ 
  company_a_percentage = 60 →
  (company_b_profit / (company_a_profit + company_b_profit)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_company_merger_profit_distribution_l2541_254165


namespace NUMINAMATH_CALUDE_mans_rate_l2541_254171

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 6)
  (h2 : speed_against_stream = 3) :
  (speed_with_stream + speed_against_stream) / 2 = 4.5 := by
  sorry

#check mans_rate

end NUMINAMATH_CALUDE_mans_rate_l2541_254171


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2541_254126

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 7 * Real.pi / 4
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 2 * Real.sqrt 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2541_254126


namespace NUMINAMATH_CALUDE_max_partner_share_l2541_254132

def profit : ℕ := 36000
def ratio : List ℕ := [2, 4, 3, 5, 6]

theorem max_partner_share :
  let total_parts := ratio.sum
  let part_value := profit / total_parts
  let shares := ratio.map (· * part_value)
  shares.maximum? = some 10800 := by sorry

end NUMINAMATH_CALUDE_max_partner_share_l2541_254132


namespace NUMINAMATH_CALUDE_egg_distribution_l2541_254141

theorem egg_distribution (total_eggs : ℕ) (num_adults num_boys num_girls : ℕ) 
  (eggs_per_adult eggs_per_girl : ℕ) :
  total_eggs = 36 →
  num_adults = 3 →
  num_boys = 10 →
  num_girls = 7 →
  eggs_per_adult = 3 →
  eggs_per_girl = 1 →
  ∃ (eggs_per_boy : ℕ),
    total_eggs = num_adults * eggs_per_adult + num_boys * eggs_per_boy + num_girls * eggs_per_girl ∧
    eggs_per_boy = eggs_per_girl + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l2541_254141


namespace NUMINAMATH_CALUDE_parcel_weight_theorem_l2541_254158

theorem parcel_weight_theorem (x y z : ℕ) 
  (h1 : x + y = 132)
  (h2 : y + z = 135)
  (h3 : z + x = 140) :
  x + y + z = 204 := by
sorry

end NUMINAMATH_CALUDE_parcel_weight_theorem_l2541_254158


namespace NUMINAMATH_CALUDE_parabola_ellipse_tangency_l2541_254121

theorem parabola_ellipse_tangency (a b : ℝ) :
  (∀ x y : ℝ, y = x^2 - 5 → x^2/a + y^2/b = 1) →
  (∃! p : ℝ × ℝ, (p.2 = p.1^2 - 5) ∧ (p.1^2/a + p.2^2/b = 1)) →
  a = 1/10 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_tangency_l2541_254121


namespace NUMINAMATH_CALUDE_total_spent_by_pete_and_raymond_l2541_254178

def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

def initial_amount : ℕ := 250

def pete_nickels : ℕ := 4
def pete_dimes : ℕ := 3
def pete_quarters : ℕ := 2

def raymond_dimes_left : ℕ := 7
def raymond_quarters_left : ℕ := 4
def raymond_nickels_left : ℕ := 5

theorem total_spent_by_pete_and_raymond : 
  (initial_amount - (raymond_dimes_left * dime_value + raymond_quarters_left * quarter_value + raymond_nickels_left * nickel_value)) +
  (pete_nickels * nickel_value + pete_dimes * dime_value + pete_quarters * quarter_value) = 155 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_by_pete_and_raymond_l2541_254178


namespace NUMINAMATH_CALUDE_project_work_time_difference_l2541_254111

theorem project_work_time_difference (x : ℝ) 
  (h1 : x > 0)
  (h2 : 2*x + 3*x + 4*x = 90) : 4*x - 2*x = 20 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_difference_l2541_254111


namespace NUMINAMATH_CALUDE_initial_number_of_students_l2541_254130

theorem initial_number_of_students :
  ∀ (n : ℕ) (W : ℝ),
    W = n * 28 →
    W + 10 = (n + 1) * 27.4 →
    n = 29 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_students_l2541_254130


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2541_254137

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 3)
  (h_sum : a 3 + a 11 = 18) :
  (∀ n : ℕ, a n = 2 * n - 5) ∧
  (a 55 = 105) ∧
  (∃ n : ℕ, n * (2 * n - 8) / 2 = 32 ∧ n = 8) ∧
  (∀ n : ℕ, n * (2 * n - 8) / 2 ≥ 2 * (2 * 2 - 8) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2541_254137


namespace NUMINAMATH_CALUDE_bird_stork_difference_is_one_l2541_254124

/-- Given an initial number of birds on a fence, and additional birds and storks that join,
    calculate the difference between the final number of storks and birds. -/
def fence_bird_stork_difference (initial_birds : ℕ) (joined_birds : ℕ) (joined_storks : ℕ) : ℤ :=
  (joined_storks : ℤ) - ((initial_birds + joined_birds) : ℤ)

/-- Theorem stating that with 3 initial birds, 2 joined birds, and 6 joined storks,
    there is 1 more stork than birds on the fence. -/
theorem bird_stork_difference_is_one :
  fence_bird_stork_difference 3 2 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bird_stork_difference_is_one_l2541_254124


namespace NUMINAMATH_CALUDE_prob_three_blue_value_l2541_254147

/-- The number of red balls in the urn -/
def num_red : ℕ := 8

/-- The number of blue balls in the urn -/
def num_blue : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls : ℕ := num_red + num_blue

/-- The number of balls drawn -/
def num_drawn : ℕ := 3

/-- The probability of drawing 3 blue balls consecutively without replacement -/
def prob_three_blue : ℚ := (num_blue.choose num_drawn) / (total_balls.choose num_drawn)

theorem prob_three_blue_value : prob_three_blue = 5/91 := by sorry

end NUMINAMATH_CALUDE_prob_three_blue_value_l2541_254147


namespace NUMINAMATH_CALUDE_sum_f_positive_l2541_254161

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l2541_254161


namespace NUMINAMATH_CALUDE_xy_value_l2541_254172

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 + y^2 = 4) (h2 : x^4 + y^4 = 7) : 
  x * y = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2541_254172


namespace NUMINAMATH_CALUDE_all_grids_have_uniform_subgrid_l2541_254142

def Grid := Fin 5 → Fin 6 → Bool

def hasUniformSubgrid (g : Grid) : Prop :=
  ∃ (i j : Fin 4), 
    (g i j = g (i + 1) j ∧ 
     g i j = g i (j + 1) ∧ 
     g i j = g (i + 1) (j + 1))

theorem all_grids_have_uniform_subgrid :
  ∀ (g : Grid), hasUniformSubgrid g :=
sorry

end NUMINAMATH_CALUDE_all_grids_have_uniform_subgrid_l2541_254142


namespace NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_eight_l2541_254105

theorem odd_square_minus_one_divisible_by_eight (n : ℤ) (h : Odd n) : 
  ∃ k : ℤ, n^2 - 1 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_divisible_by_eight_l2541_254105


namespace NUMINAMATH_CALUDE_lcm_sum_implies_product_div_3_or_5_l2541_254199

theorem lcm_sum_implies_product_div_3_or_5 (a b c d : ℕ) :
  Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = a + b + c + d →
  3 ∣ (a * b * c * d) ∨ 5 ∣ (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_lcm_sum_implies_product_div_3_or_5_l2541_254199


namespace NUMINAMATH_CALUDE_coin_value_equality_l2541_254160

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Theorem stating that if the value of 25 quarters, 15 dimes, and 10 nickels 
    equals the value of 15 quarters, n dimes, and 20 nickels, then n = 35 -/
theorem coin_value_equality (n : ℕ) : 
  25 * quarter_value + 15 * dime_value + 10 * nickel_value = 
  15 * quarter_value + n * dime_value + 20 * nickel_value → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_equality_l2541_254160


namespace NUMINAMATH_CALUDE_min_value_theorem_l2541_254176

def f (a x : ℝ) : ℝ := x^2 + (a+8)*x + a^2 + a - 12

theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  (∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 37/4) ∧ 
  (∃ n : ℕ+, (f a n - 4*a) / (n + 1) = 37/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2541_254176


namespace NUMINAMATH_CALUDE_equation_solution_l2541_254109

theorem equation_solution : ∃ x : ℝ, 2 * x + 6 = 2 + 3 * x ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2541_254109


namespace NUMINAMATH_CALUDE_parking_lot_cars_l2541_254180

theorem parking_lot_cars (red_cars : ℕ) (black_cars : ℕ) : 
  red_cars = 33 → 
  red_cars * 8 = black_cars * 3 → 
  black_cars = 88 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l2541_254180


namespace NUMINAMATH_CALUDE_v₃_value_l2541_254108

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

/-- The value of x -/
def x : ℝ := 5

/-- The definition of v₃ -/
def v₃ : ℝ := (((5*x + 2)*x + 3.5)*x - 2.6)

/-- Theorem stating that v₃ equals 689.9 -/
theorem v₃_value : v₃ = 689.9 := by
  sorry

end NUMINAMATH_CALUDE_v₃_value_l2541_254108


namespace NUMINAMATH_CALUDE_intersection_when_a_2_b_subset_a_range_l2541_254175

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Theorem 1: Intersection of A and B when a = 2
theorem intersection_when_a_2 :
  A 2 ∩ B 2 = {x | 4 < x ∧ x < 5} :=
sorry

-- Theorem 2: Range of a for which B is a subset of A
theorem b_subset_a_range :
  ∀ a : ℝ, B a ⊆ A a ↔ a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_2_b_subset_a_range_l2541_254175


namespace NUMINAMATH_CALUDE_power_zero_plus_two_l2541_254169

theorem power_zero_plus_two : (-2010)^0 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_plus_two_l2541_254169


namespace NUMINAMATH_CALUDE_unicorn_rope_problem_l2541_254184

theorem unicorn_rope_problem (rope_length : ℝ) (tower_radius : ℝ) (rope_height : ℝ) (rope_distance : ℝ) 
  (h1 : rope_length = 30)
  (h2 : tower_radius = 10)
  (h3 : rope_height = 6)
  (h4 : rope_distance = 6) :
  ∃ (p q r : ℕ), 
    (p > 0 ∧ q > 0 ∧ r > 0) ∧ 
    Nat.Prime r ∧
    (p - Real.sqrt q) / r = 
      (rope_length * Real.sqrt ((tower_radius + rope_distance)^2 + rope_height^2)) / 
      (tower_radius + Real.sqrt ((tower_radius + rope_distance)^2 + rope_height^2)) ∧
    p + q + r = 1290 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_rope_problem_l2541_254184


namespace NUMINAMATH_CALUDE_simplify_expression_l2541_254189

theorem simplify_expression (y : ℝ) : 3*y + 5*y + 6*y + 10 = 14*y + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2541_254189


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2541_254191

theorem right_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≤ b) (hbc : b < c) (hright : a^2 + b^2 = c^2) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2541_254191


namespace NUMINAMATH_CALUDE_empty_union_l2541_254168

theorem empty_union (A : Set α) : A ∪ ∅ = A := by sorry

end NUMINAMATH_CALUDE_empty_union_l2541_254168


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2541_254181

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 15 ∧ (899830 - x) % 16 = 0 ∧ ∀ y : ℕ, y < x → (899830 - y) % 16 ≠ 0 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2541_254181


namespace NUMINAMATH_CALUDE_grace_pumpkin_pie_fraction_l2541_254110

theorem grace_pumpkin_pie_fraction :
  let total_pies : ℕ := 4
  let sold_pies : ℕ := 1
  let given_pies : ℕ := 1
  let slices_per_pie : ℕ := 6
  let remaining_slices : ℕ := 4
  
  let remaining_pies : ℕ := total_pies - sold_pies - given_pies
  let total_slices : ℕ := remaining_pies * slices_per_pie
  let eaten_slices : ℕ := total_slices - remaining_slices
  
  (eaten_slices : ℚ) / total_slices = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_grace_pumpkin_pie_fraction_l2541_254110


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2541_254152

/-- Given two inversely proportional quantities p and q, if p = 30 when q = 4,
    then p = 12 when q = 10. -/
theorem inverse_proportion_problem (p q : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, p = x ∧ q = y → x * y = k) :
  (p = 30 ∧ q = 4) → (q = 10 → p = 12) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2541_254152


namespace NUMINAMATH_CALUDE_surface_area_difference_l2541_254148

/-- The difference between the sum of surface areas of smaller cubes and the surface area of a larger cube -/
theorem surface_area_difference (larger_cube_volume : ℝ) (num_smaller_cubes : ℕ) (smaller_cube_volume : ℝ) : 
  larger_cube_volume = 343 →
  num_smaller_cubes = 343 →
  smaller_cube_volume = 1 →
  (num_smaller_cubes : ℝ) * (6 * smaller_cube_volume ^ (2/3)) - 6 * larger_cube_volume ^ (2/3) = 1764 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l2541_254148


namespace NUMINAMATH_CALUDE_pencil_cost_l2541_254150

/-- The cost of a pencil given the total cost with an eraser and the price difference -/
theorem pencil_cost (total : ℝ) (difference : ℝ) (h1 : total = 3.4) (h2 : difference = 3) :
  ∃ (pencil eraser : ℝ),
    pencil + eraser = total ∧
    pencil = eraser + difference ∧
    pencil = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l2541_254150


namespace NUMINAMATH_CALUDE_binomial_20_19_equals_20_l2541_254159

theorem binomial_20_19_equals_20 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_equals_20_l2541_254159


namespace NUMINAMATH_CALUDE_breakfast_rearrangements_count_l2541_254194

/-- The number of distinguishable rearrangements of "BREAKFAST" with vowels first -/
def breakfast_rearrangements : ℕ :=
  let vowels := 3  -- Number of vowels in "BREAKFAST"
  let repeated_vowels := 2  -- Number of times 'A' appears
  let consonants := 6  -- Number of consonants in "BREAKFAST"
  (vowels.factorial / repeated_vowels.factorial) * consonants.factorial

/-- Theorem stating that the number of rearrangements is 2160 -/
theorem breakfast_rearrangements_count :
  breakfast_rearrangements = 2160 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_rearrangements_count_l2541_254194


namespace NUMINAMATH_CALUDE_prob_more_surgeons_than_internists_mean_surgeons_selected_variance_surgeons_selected_l2541_254187

/-- Represents the selection of doctors for a medical outreach program. -/
structure DoctorSelection where
  total : Nat
  surgeons : Nat
  internists : Nat
  ophthalmologists : Nat
  selected : Nat

/-- The specific scenario of selecting 3 out of 6 doctors. -/
def scenario : DoctorSelection :=
  { total := 6
  , surgeons := 2
  , internists := 2
  , ophthalmologists := 2
  , selected := 3 }

/-- The probability of selecting more surgeons than internists. -/
def probMoreSurgeonsThanInternists (s : DoctorSelection) : ℚ :=
  3 / 10

/-- The mean number of surgeons selected. -/
def meanSurgeonsSelected (s : DoctorSelection) : ℚ :=
  1

/-- The variance of the number of surgeons selected. -/
def varianceSurgeonsSelected (s : DoctorSelection) : ℚ :=
  2 / 5

/-- Theorem stating the probability of selecting more surgeons than internists. -/
theorem prob_more_surgeons_than_internists :
  probMoreSurgeonsThanInternists scenario = 3 / 10 := by
  sorry

/-- Theorem stating the mean number of surgeons selected. -/
theorem mean_surgeons_selected :
  meanSurgeonsSelected scenario = 1 := by
  sorry

/-- Theorem stating the variance of the number of surgeons selected. -/
theorem variance_surgeons_selected :
  varianceSurgeonsSelected scenario = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_more_surgeons_than_internists_mean_surgeons_selected_variance_surgeons_selected_l2541_254187


namespace NUMINAMATH_CALUDE_field_length_proof_l2541_254153

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 8 →
  pond_side ^ 2 = (1 / 18) * (length * width) →
  length = 48 := by
  sorry

end NUMINAMATH_CALUDE_field_length_proof_l2541_254153


namespace NUMINAMATH_CALUDE_B_is_largest_l2541_254135

/-- A is defined as the sum of 2023/2022 and 2023/2024 -/
def A : ℚ := 2023/2022 + 2023/2024

/-- B is defined as the sum of 2024/2023 and 2026/2023 -/
def B : ℚ := 2024/2023 + 2026/2023

/-- C is defined as the sum of 2025/2024 and 2025/2026 -/
def C : ℚ := 2025/2024 + 2025/2026

/-- Theorem stating that B is the largest among A, B, and C -/
theorem B_is_largest : B > A ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_B_is_largest_l2541_254135
