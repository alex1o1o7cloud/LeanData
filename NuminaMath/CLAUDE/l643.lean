import Mathlib

namespace NUMINAMATH_CALUDE_fraction_multiplication_and_subtraction_l643_64391

theorem fraction_multiplication_and_subtraction :
  (5 : ℚ) / 6 * ((2 : ℚ) / 3 - (1 : ℚ) / 9) = (25 : ℚ) / 54 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_and_subtraction_l643_64391


namespace NUMINAMATH_CALUDE_no_prime_perfect_square_l643_64317

theorem no_prime_perfect_square : ¬∃ (p : ℕ), Prime p ∧ ∃ (a : ℕ), 7 * p + 3^p - 4 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_perfect_square_l643_64317


namespace NUMINAMATH_CALUDE_investment_problem_l643_64379

/-- Proves that given the conditions of the investment problem, the initial sum invested was $900 -/
theorem investment_problem (P : ℝ) : 
  P > 0 → 
  (P * (4.5 / 100) * 7) - (P * (4 / 100) * 7) = 31.5 → 
  P = 900 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l643_64379


namespace NUMINAMATH_CALUDE_amy_homework_rate_l643_64306

/-- Given a total number of problems and the time taken to complete them,
    calculate the number of problems completed per hour. -/
def problems_per_hour (total_problems : ℕ) (total_hours : ℕ) : ℚ :=
  total_problems / total_hours

/-- Theorem stating that with 24 problems completed in 6 hours,
    the number of problems completed per hour is 4. -/
theorem amy_homework_rate :
  problems_per_hour 24 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_amy_homework_rate_l643_64306


namespace NUMINAMATH_CALUDE_max_points_32_l643_64310

-- Define the total number of shots
def total_shots : ℕ := 40

-- Define the success rates for three-point and two-point shots
def three_point_rate : ℚ := 1/4
def two_point_rate : ℚ := 2/5

-- Define the function that calculates the total points based on the number of three-point attempts
def total_points (three_point_attempts : ℕ) : ℚ :=
  3 * three_point_rate * three_point_attempts + 
  2 * two_point_rate * (total_shots - three_point_attempts)

-- Theorem: The maximum number of points Jamal could score is 32
theorem max_points_32 : 
  ∀ x : ℕ, x ≤ total_shots → total_points x ≤ 32 :=
sorry

end NUMINAMATH_CALUDE_max_points_32_l643_64310


namespace NUMINAMATH_CALUDE_range_of_a_l643_64300

theorem range_of_a (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ (x : ℤ), (x + 6 < 2 + 3*x ∧ (a + x) / 4 > x) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  (15 < a ∧ a ≤ 18) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l643_64300


namespace NUMINAMATH_CALUDE_kevins_cards_l643_64363

/-- Kevin's card problem -/
theorem kevins_cards (initial_cards found_cards : ℕ) : 
  initial_cards = 7 → found_cards = 47 → initial_cards + found_cards = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevins_cards_l643_64363


namespace NUMINAMATH_CALUDE_f_properties_l643_64347

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.exp x

theorem f_properties :
  ∀ a : ℝ,
  (a = -1 →
    (∀ x y : ℝ, x < y → x < 0 → y < 0 → f a x < f a y) ∧
    (∀ x y : ℝ, x < y → x > 0 → y > 0 → f a x > f a y)) ∧
  (a ≥ 0 →
    ∀ x : ℝ, ¬∃ y : ℝ, (∀ z : ℝ, f a z ≤ f a y) ∨ (∀ z : ℝ, f a z ≥ f a y)) ∧
  (a < 0 →
    (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) ∧
    (¬∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
    (∃ x : ℝ, x = Real.log (-1/a) ∧ ∀ y : ℝ, f a y ≤ f a x) ∧
    (∃ x : ℝ, x = Real.log (-1/a) ∧ f a x = Real.log (-1/a) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l643_64347


namespace NUMINAMATH_CALUDE_intersection_with_complement_l643_64394

def U : Finset ℕ := {0, 1, 2, 3, 4}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 0}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l643_64394


namespace NUMINAMATH_CALUDE_unique_prime_fraction_l643_64387

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_fraction :
  ∀ a b : ℕ,
    a > 0 →
    b > 0 →
    a ≠ b →
    is_prime (a * b^2 / (a + b)) →
    a = 6 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_fraction_l643_64387


namespace NUMINAMATH_CALUDE_jerry_pills_clock_time_l643_64314

theorem jerry_pills_clock_time (total_pills : ℕ) (interval : ℕ) (start_time : ℕ) : 
  total_pills = 150 →
  interval = 5 →
  start_time = 12 →
  (start_time + (total_pills - 1) * interval) % 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_jerry_pills_clock_time_l643_64314


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l643_64309

theorem solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (x > a ∧ x > 1) ↔ x > 1) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l643_64309


namespace NUMINAMATH_CALUDE_triangles_are_similar_l643_64328

/-- Two triangles are similar if the ratios of their corresponding sides are equal -/
def are_similar (a b c d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ b = k * e ∧ c = k * f ∧ a = k * d

/-- Triangle ABC has sides of length 1, √2, and √5 -/
def triangle_ABC (a b c : ℝ) : Prop :=
  a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 5

/-- Triangle DEF has sides of length √3, √6, and √15 -/
def triangle_DEF (d e f : ℝ) : Prop :=
  d = Real.sqrt 3 ∧ e = Real.sqrt 6 ∧ f = Real.sqrt 15

theorem triangles_are_similar :
  ∀ (a b c d e f : ℝ),
    triangle_ABC a b c →
    triangle_DEF d e f →
    are_similar a b c d e f :=
by sorry

end NUMINAMATH_CALUDE_triangles_are_similar_l643_64328


namespace NUMINAMATH_CALUDE_soccer_team_win_percentage_l643_64373

/-- Given a soccer team that played 140 games and won 70 games, 
    prove that the percentage of games won is 50%. -/
theorem soccer_team_win_percentage 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (h1 : total_games = 140) 
  (h2 : games_won = 70) : 
  (games_won : ℚ) / total_games * 100 = 50 := by
  sorry

#check soccer_team_win_percentage

end NUMINAMATH_CALUDE_soccer_team_win_percentage_l643_64373


namespace NUMINAMATH_CALUDE_library_books_l643_64326

theorem library_books (original_books : ℕ) : 
  (original_books + 140 = (27 : ℚ) / 25 * original_books) → 
  original_books = 1750 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l643_64326


namespace NUMINAMATH_CALUDE_point_coordinates_l643_64320

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the Cartesian coordinate system -/
def third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (A : Point) 
  (h1 : third_quadrant A)
  (h2 : distance_to_x_axis A = 2)
  (h3 : distance_to_y_axis A = 3) :
  A.x = -3 ∧ A.y = -2 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l643_64320


namespace NUMINAMATH_CALUDE_cookie_pie_leftover_slices_l643_64385

theorem cookie_pie_leftover_slices (num_pies : ℕ) (slices_per_pie : ℕ) (num_classmates : ℕ) (num_teachers : ℕ) (slices_per_person : ℕ) :
  num_pies = 3 →
  slices_per_pie = 10 →
  num_classmates = 24 →
  num_teachers = 1 →
  slices_per_person = 1 →
  num_pies * slices_per_pie - (num_classmates + num_teachers + 1) * slices_per_person = 4 :=
by sorry

end NUMINAMATH_CALUDE_cookie_pie_leftover_slices_l643_64385


namespace NUMINAMATH_CALUDE_parabola_properties_l643_64318

/-- Given a parabola y = x^2 + 2bx + b^2 - 2 where b > 0 and passing through point (0, -1) -/
theorem parabola_properties (b : ℝ) (h1 : b > 0) :
  let f (x : ℝ) := x^2 + 2*b*x + b^2 - 2
  ∃ (vertex_x vertex_y : ℝ),
    -- 1. The vertex coordinates are (-b, -2)
    (vertex_x = -b ∧ vertex_y = -2) ∧
    -- Parabola passes through (0, -1)
    (f 0 = -1) ∧
    -- 2. When -2 < x < 3, the range of y is -2 ≤ y < 14
    (∀ x, -2 < x → x < 3 → -2 ≤ f x ∧ f x < 14) ∧
    -- 3. When k ≤ x ≤ 2 and -2 ≤ y ≤ 7, the range of k is -4 ≤ k ≤ -1
    (∀ k, (∀ x, k ≤ x → x ≤ 2 → -2 ≤ f x → f x ≤ 7) → -4 ≤ k ∧ k ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l643_64318


namespace NUMINAMATH_CALUDE_min_value_of_y_l643_64368

def y (x : ℝ) : ℝ :=
  |x - 1| + |x - 2| + |x - 3| + |x - 4| + |x - 5| + |x - 6| + |x - 7| + |x - 8| + |x - 9| + |x - 10|

theorem min_value_of_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y z ≥ y x ∧ y x = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_y_l643_64368


namespace NUMINAMATH_CALUDE_bird_count_problem_l643_64302

/-- Represents the number of birds in a group -/
structure BirdGroup where
  adults : ℕ
  offspring_per_adult : ℕ

/-- Calculates the total number of birds in a group -/
def total_birds (group : BirdGroup) : ℕ :=
  group.adults * (group.offspring_per_adult + 1)

/-- The problem statement -/
theorem bird_count_problem (duck_group1 duck_group2 duck_group3 geese_group swan_group : BirdGroup)
  (h1 : duck_group1 = { adults := 2, offspring_per_adult := 5 })
  (h2 : duck_group2 = { adults := 6, offspring_per_adult := 3 })
  (h3 : duck_group3 = { adults := 9, offspring_per_adult := 6 })
  (h4 : geese_group = { adults := 4, offspring_per_adult := 7 })
  (h5 : swan_group = { adults := 3, offspring_per_adult := 4 }) :
  (total_birds duck_group1 + total_birds duck_group2 + total_birds duck_group3 +
   total_birds geese_group + total_birds swan_group) * 3 = 438 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_problem_l643_64302


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l643_64327

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a)
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l643_64327


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l643_64399

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_derivative_at_2 : 
  deriv f 2 = (1 - Real.log 2) / 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l643_64399


namespace NUMINAMATH_CALUDE_marcos_dad_strawberries_weight_l643_64308

theorem marcos_dad_strawberries_weight (marco_weight dad_weight total_weight : ℕ) :
  marco_weight = 8 →
  total_weight = 40 →
  total_weight = marco_weight + dad_weight →
  dad_weight = 32 := by
sorry

end NUMINAMATH_CALUDE_marcos_dad_strawberries_weight_l643_64308


namespace NUMINAMATH_CALUDE_range_of_f_l643_64377

def f (x : ℝ) := -x^2 + 2*x + 3

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-5 : ℝ) 4, ∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = y ∧
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ∈ Set.Icc (-5 : ℝ) 4 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l643_64377


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l643_64335

theorem cubic_polynomial_property (n : ℕ+) : 
  ∃ k : ℤ, (n^3 : ℚ) + (3/2) * n^2 + (1/2) * n - 1 = k ∧ k % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l643_64335


namespace NUMINAMATH_CALUDE_inequality_proof_l643_64395

theorem inequality_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_ineq : ∀ x, f x > deriv f x) (a b : ℝ) (hab : a > b) :
  Real.exp a * f b > Real.exp b * f a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l643_64395


namespace NUMINAMATH_CALUDE_quadratic_transformation_l643_64330

/-- Given a quadratic equation x² + px + q = 0 with roots x₁ and x₂,
    this theorem proves the form of the quadratic equation whose roots are
    y₁ = (x₁ + x₁²) / (1 - x₂) and y₂ = (x₂ + x₂²) / (1 - x₁) -/
theorem quadratic_transformation (p q : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + p*x₁ + q = 0 →
  x₂^2 + p*x₂ + q = 0 →
  x₁ ≠ x₂ →
  x₁ ≠ 1 →
  x₂ ≠ 1 →
  let y₁ := (x₁ + x₁^2) / (1 - x₂)
  let y₂ := (x₂ + x₂^2) / (1 - x₁)
  ∃ (y : ℝ), y^2 + (p*(1 + 3*q - p^2) / (1 + p + q))*y + (q*(1 - p + q) / (1 + p + q)) = 0 ↔
             (y = y₁ ∨ y = y₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l643_64330


namespace NUMINAMATH_CALUDE_expand_expression_l643_64352

theorem expand_expression (x y : ℝ) : (2*x + 15) * (3*y + 5) = 6*x*y + 10*x + 45*y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l643_64352


namespace NUMINAMATH_CALUDE_min_overlap_blue_eyes_lunch_box_l643_64393

theorem min_overlap_blue_eyes_lunch_box 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 25) 
  (h2 : blue_eyes = 15) 
  (h3 : lunch_box = 18) :
  blue_eyes + lunch_box - total_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_blue_eyes_lunch_box_l643_64393


namespace NUMINAMATH_CALUDE_area_ratio_of_inscribed_squares_l643_64370

/-- A square inscribed in a circle -/
structure InscribedSquare where
  side : ℝ
  radius : ℝ
  inscribed : radius = side * Real.sqrt 2 / 2

/-- A square with two vertices on a line segment and two on a circle -/
structure PartiallyInscribedSquare where
  side : ℝ
  outer_square : InscribedSquare
  vertices_on_side : side ≤ outer_square.side
  vertices_on_circle : side = outer_square.side * Real.sqrt 2 / 5

/-- The theorem to be proved -/
theorem area_ratio_of_inscribed_squares (outer : InscribedSquare) 
    (inner : PartiallyInscribedSquare) (h : inner.outer_square = outer) :
    (inner.side ^ 2) / (outer.side ^ 2) = 2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_inscribed_squares_l643_64370


namespace NUMINAMATH_CALUDE_handshake_partition_handshake_same_neighbors_l643_64361

open Set

structure HandshakeGraph (α : Type*) [Fintype α] where
  edges : Set (α × α)
  symm : ∀ a b, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges
  handshake_property : ∀ a b c d, (a, b) ∈ edges → (b, c) ∈ edges → (c, d) ∈ edges →
    (a, c) ∈ edges ∨ (a, d) ∈ edges ∨ (b, d) ∈ edges

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem handshake_partition (n : ℕ) (h : n ≥ 4) (G : HandshakeGraph (Fin n)) :
  ∃ (X Y : Set (Fin n)), X.Nonempty ∧ Y.Nonempty ∧ X ∪ Y = univ ∧ X ∩ Y = ∅ ∧
  (∀ x y, x ∈ X → y ∈ Y → ((x, y) ∈ G.edges ↔ ∀ a ∈ X, ∀ b ∈ Y, (a, b) ∈ G.edges)) :=
sorry

theorem handshake_same_neighbors (n : ℕ) (h : n ≥ 4) (G : HandshakeGraph (Fin n)) :
  ∃ (A B : Fin n), A ≠ B ∧
  {x | x ≠ A ∧ x ≠ B ∧ (A, x) ∈ G.edges} = {x | x ≠ A ∧ x ≠ B ∧ (B, x) ∈ G.edges} :=
sorry

end NUMINAMATH_CALUDE_handshake_partition_handshake_same_neighbors_l643_64361


namespace NUMINAMATH_CALUDE_problem_solution_l643_64311

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem problem_solution (a : ℝ) (h_a : a > 0) (h_a_neq_1 : a ≠ 1) :
  -- Part 1
  (∀ x, f a 2 x = -f a 2 (-x)) →
  -- Part 2
  f a 2 1 < 0 →
  (∀ x t, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0 ↔ -3 < t ∧ t < 5) ∧
  (∀ x y, x < y → f a 2 y < f a 2 x) →
  -- Part 3
  f a 2 1 = 3/2 →
  (∃ m, ∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2) →
  (∃! m, ∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2 ∧
               (∃ y, y ≥ 1 ∧ a^(2*y) + a^(-2*y) - 2*m*(f a 2 y) = -2)) →
  ∃ m, m = 2 ∧
    (∀ x, x ≥ 1 → a^(2*x) + a^(-2*x) - 2*m*(f a 2 x) ≥ -2) ∧
    (∃ y, y ≥ 1 ∧ a^(2*y) + a^(-2*y) - 2*m*(f a 2 y) = -2) := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l643_64311


namespace NUMINAMATH_CALUDE_julia_tag_game_l643_64396

theorem julia_tag_game (monday_kids tuesday_kids : ℕ) 
  (h1 : monday_kids = 4) 
  (h2 : tuesday_kids = 14) : 
  monday_kids + tuesday_kids = 18 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l643_64396


namespace NUMINAMATH_CALUDE_angle_bisector_d_value_l643_64342

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (-4, -2)
def C : ℝ × ℝ := (7, -1)

-- Define the angle bisector equation
def angleBisectorEq (x y d : ℝ) : Prop := x - 3*y + d = 0

-- Theorem statement
theorem angle_bisector_d_value :
  ∃ d : ℝ, (∀ x y : ℝ, angleBisectorEq x y d ↔ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
      x = B.1 + t * (C.1 - B.1) ∧
      y = B.2 + t * (C.2 - B.2))) ∧
    angleBisectorEq B.1 B.2 d :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_d_value_l643_64342


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l643_64340

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence where the 4th term is 8 and the 10th term is 2, the 7th term is 1. -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_4th : a 4 = 8)
  (h_10th : a 10 = 2)
  : a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l643_64340


namespace NUMINAMATH_CALUDE_prob_all_red_when_n_3_n_value_when_prob_at_least_2_red_is_3_4_l643_64322

-- Define the contents of the bags
def bag_A : ℕ × ℕ := (2, 2)  -- (red balls, white balls)
def bag_B (n : ℕ) : ℕ × ℕ := (2, n)  -- (red balls, white balls)

-- Define the probability of drawing all red balls
def prob_all_red (n : ℕ) : ℚ :=
  (Nat.choose 2 2 * Nat.choose 2 2) / (Nat.choose 4 2 * Nat.choose (n + 2) 2)

-- Define the probability of drawing at least 2 red balls
def prob_at_least_2_red (n : ℕ) : ℚ :=
  1 - (Nat.choose 2 2 * Nat.choose n 2 + Nat.choose 2 1 * Nat.choose 2 1 * Nat.choose n 2 + Nat.choose 2 2 * Nat.choose 2 1 * Nat.choose n 1) / (Nat.choose 4 2 * Nat.choose (n + 2) 2)

theorem prob_all_red_when_n_3 :
  prob_all_red 3 = 1 / 60 := by sorry

theorem n_value_when_prob_at_least_2_red_is_3_4 :
  ∃ n : ℕ, prob_at_least_2_red n = 3 / 4 ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_prob_all_red_when_n_3_n_value_when_prob_at_least_2_red_is_3_4_l643_64322


namespace NUMINAMATH_CALUDE_eight_amp_two_l643_64339

/-- Custom binary operation & -/
def amp (a b : ℤ) : ℤ := (a + b) * (a - b) + a * b

/-- Theorem: 8 & 2 = 76 -/
theorem eight_amp_two : amp 8 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_eight_amp_two_l643_64339


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l643_64366

theorem quadratic_equation_properties (m : ℝ) :
  let f (x : ℝ) := x^2 - (2*m - 1)*x - 3*m^2 + m
  ∃ (x₁ x₂ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₂/x₁ + x₁/x₂ = -5/2 → (m = 1 ∨ m = 2/5)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l643_64366


namespace NUMINAMATH_CALUDE_max_clerks_results_l643_64344

theorem max_clerks_results (initial_count : ℕ) (operation_count : ℕ) 
  (h1 : initial_count = 100)
  (h2 : operation_count = initial_count - 1) :
  ∃ (max_results : ℕ), max_results = operation_count / 2 + 1 ∧ 
  max_results = 51 := by
  sorry

end NUMINAMATH_CALUDE_max_clerks_results_l643_64344


namespace NUMINAMATH_CALUDE_all_defective_impossible_l643_64362

structure ProductSet where
  total : ℕ
  defective : ℕ
  drawn : ℕ
  h_total : total = 10
  h_defective : defective = 2
  h_drawn : drawn = 3
  h_defective_lt_total : defective < total

def all_defective (s : ProductSet) : Prop :=
  ∀ (i : Fin s.drawn), i.val < s.defective

theorem all_defective_impossible (s : ProductSet) : ¬ (all_defective s) := by
  sorry

end NUMINAMATH_CALUDE_all_defective_impossible_l643_64362


namespace NUMINAMATH_CALUDE_college_student_count_l643_64381

theorem college_student_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 160) :
  boys + girls = 416 := by
sorry

end NUMINAMATH_CALUDE_college_student_count_l643_64381


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l643_64369

def A : Set ℝ := {0, 1, 2}

def B : Set ℝ := {x : ℝ | (x + 1) * (x + 2) ≤ 0}

theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l643_64369


namespace NUMINAMATH_CALUDE_limit_of_sequence_l643_64351

def a (n : ℕ) : ℚ := (3 * n - 2) / (2 * n - 1)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/2| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l643_64351


namespace NUMINAMATH_CALUDE_optimal_production_volume_l643_64376

-- Define the profit function
def W (x : ℝ) : ℝ := -2 * x^3 + 21 * x^2

-- State the theorem
theorem optimal_production_volume (x : ℝ) (h : x > 0) :
  ∃ (max_x : ℝ), max_x = 7 ∧ 
  ∀ y, y > 0 → W y ≤ W max_x :=
sorry

end NUMINAMATH_CALUDE_optimal_production_volume_l643_64376


namespace NUMINAMATH_CALUDE_parking_lot_problem_l643_64332

theorem parking_lot_problem (total_vehicles : ℕ) (total_wheels : ℕ) 
  (car_wheels : ℕ) (motorcycle_wheels : ℕ) :
  total_vehicles = 30 →
  total_wheels = 84 →
  car_wheels = 4 →
  motorcycle_wheels = 2 →
  ∃ (cars : ℕ) (motorcycles : ℕ),
    cars + motorcycles = total_vehicles ∧
    car_wheels * cars + motorcycle_wheels * motorcycles = total_wheels ∧
    motorcycles = 18 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_problem_l643_64332


namespace NUMINAMATH_CALUDE_classroom_composition_l643_64334

/-- In a class, each boy is friends with exactly two girls, and each girl is friends with exactly three boys. -/
structure Classroom where
  boys : ℕ
  girls : ℕ
  total_students : boys + girls = 31
  boy_girl_connections : 2 * boys = 3 * girls

/-- The number of boys and girls in the classroom satisfies the given conditions. -/
theorem classroom_composition : ∃ (c : Classroom), c.boys = 19 ∧ c.girls = 12 := by
  sorry

end NUMINAMATH_CALUDE_classroom_composition_l643_64334


namespace NUMINAMATH_CALUDE_combined_salaries_l643_64354

/-- The problem of calculating combined salaries -/
theorem combined_salaries 
  (salary_C : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) 
  (h1 : salary_C = 11000)
  (h2 : average_salary = 8200)
  (h3 : num_individuals = 5) :
  average_salary * num_individuals - salary_C = 30000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l643_64354


namespace NUMINAMATH_CALUDE_waiter_customers_l643_64367

/-- Calculates the total number of customers for a waiter given the number of tables and customers per table. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Proves that a waiter with 9 tables, each having 7 women and 3 men, has 90 customers in total. -/
theorem waiter_customers : total_customers 9 7 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l643_64367


namespace NUMINAMATH_CALUDE_divisibility_condition_l643_64331

theorem divisibility_condition (m n : ℤ) : 
  m > 1 → n > 1 → (m * n - 1 ∣ n^3 - 1) ↔ (m = n^2 ∨ n = m^2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l643_64331


namespace NUMINAMATH_CALUDE_sqrt_three_fourths_equals_sqrt_three_over_two_l643_64358

theorem sqrt_three_fourths_equals_sqrt_three_over_two :
  Real.sqrt (3 / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_fourths_equals_sqrt_three_over_two_l643_64358


namespace NUMINAMATH_CALUDE_fred_onions_l643_64350

/-- Proves that Fred grew 9 onions given the conditions of the problem -/
theorem fred_onions (sara : ℕ) (sally : ℕ) (fred : ℕ) (total : ℕ)
  (h1 : sara = 4)
  (h2 : sally = 5)
  (h3 : total = 18)
  (h4 : sara + sally + fred = total) :
  fred = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_onions_l643_64350


namespace NUMINAMATH_CALUDE_parabola_translation_l643_64380

/-- Represents a parabola in the form y = -(x - h)² + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { h := p.h + dx, k := p.k + dy }

theorem parabola_translation :
  let original := Parabola.mk 1 0
  let translated := translate original 1 2
  translated = Parabola.mk 2 2 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l643_64380


namespace NUMINAMATH_CALUDE_fuji_to_total_ratio_l643_64365

/-- Represents an apple orchard with Fuji and Gala trees -/
structure AppleOrchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The ratio of pure Fuji trees to all trees in the orchard is 39:52 -/
theorem fuji_to_total_ratio (orchard : AppleOrchard) :
  orchard.crossPollinated = (orchard.totalTrees : ℚ) * (1/10) ∧
  orchard.pureFuji + orchard.crossPollinated = 221 ∧
  orchard.pureGala = 39 →
  (orchard.pureFuji : ℚ) / orchard.totalTrees = 39 / 52 :=
by sorry

end NUMINAMATH_CALUDE_fuji_to_total_ratio_l643_64365


namespace NUMINAMATH_CALUDE_butter_calculation_l643_64325

/-- Calculates the required amount of butter given a change in sugar amount -/
def required_butter (original_butter original_sugar new_sugar : ℚ) : ℚ :=
  (new_sugar / original_sugar) * original_butter

theorem butter_calculation (original_butter original_sugar new_sugar : ℚ) 
  (h1 : original_butter = 25)
  (h2 : original_sugar = 125)
  (h3 : new_sugar = 1000) :
  required_butter original_butter original_sugar new_sugar = 200 := by
  sorry

#eval required_butter 25 125 1000

end NUMINAMATH_CALUDE_butter_calculation_l643_64325


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_l643_64374

/-- Proof of the first simplification -/
theorem simplify_expression_1 (a b : ℝ) : 2 * a^2 + 9 * b - 5 * a^2 - 4 * b = -3 * a^2 + 5 * b := by
  sorry

/-- Proof of the second simplification and evaluation -/
theorem simplify_and_evaluate_expression_2 : 3 * 1 * (-2)^2 + 1^2 * (-2) - 2 * (2 * 1 * (-2)^2 - 1^2 * (-2)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_and_evaluate_expression_2_l643_64374


namespace NUMINAMATH_CALUDE_percentage_subtraction_l643_64312

theorem percentage_subtraction (a : ℝ) : a - 0.02 * a = 0.98 * a := by
  sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l643_64312


namespace NUMINAMATH_CALUDE_total_remaining_sand_first_truck_percentage_lost_second_truck_percentage_lost_third_truck_percentage_lost_fourth_truck_percentage_lost_l643_64398

/- Define the trucks and their properties -/
structure Truck where
  initial_sand : Float
  sand_lost : Float
  miles_driven : Float

/- Define the four trucks -/
def truck1 : Truck := { initial_sand := 4.1, sand_lost := 2.4, miles_driven := 20 }
def truck2 : Truck := { initial_sand := 5.7, sand_lost := 3.6, miles_driven := 15 }
def truck3 : Truck := { initial_sand := 8.2, sand_lost := 1.9, miles_driven := 25 }
def truck4 : Truck := { initial_sand := 10.5, sand_lost := 2.1, miles_driven := 30 }

/- Calculate remaining sand for a truck -/
def remaining_sand (t : Truck) : Float :=
  t.initial_sand - t.sand_lost

/- Calculate percentage of sand lost for a truck -/
def percentage_lost (t : Truck) : Float :=
  (t.sand_lost / t.initial_sand) * 100

/- Theorem: Total remaining sand is 18.5 pounds -/
theorem total_remaining_sand :
  remaining_sand truck1 + remaining_sand truck2 + remaining_sand truck3 + remaining_sand truck4 = 18.5 := by
  sorry

/- Theorem: Percentage of sand lost by the first truck is 58.54% -/
theorem first_truck_percentage_lost :
  percentage_lost truck1 = 58.54 := by
  sorry

/- Theorem: Percentage of sand lost by the second truck is 63.16% -/
theorem second_truck_percentage_lost :
  percentage_lost truck2 = 63.16 := by
  sorry

/- Theorem: Percentage of sand lost by the third truck is 23.17% -/
theorem third_truck_percentage_lost :
  percentage_lost truck3 = 23.17 := by
  sorry

/- Theorem: Percentage of sand lost by the fourth truck is 20% -/
theorem fourth_truck_percentage_lost :
  percentage_lost truck4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_sand_first_truck_percentage_lost_second_truck_percentage_lost_third_truck_percentage_lost_fourth_truck_percentage_lost_l643_64398


namespace NUMINAMATH_CALUDE_race_result_l643_64386

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distanceTraveled (runner : Runner) (t : ℝ) : ℝ :=
  runner.speed * t

/-- The race problem setup -/
def raceProblem : Prop :=
  ∃ (A B : Runner),
    -- The race is 1000 meters long
    distanceTraveled A A.time = 1000 ∧
    -- A finishes the race in 115 seconds
    A.time = 115 ∧
    -- B finishes 10 seconds after A
    B.time = A.time + 10 ∧
    -- The distance by which A beats B is 80 meters
    1000 - distanceTraveled B A.time = 80

theorem race_result : raceProblem := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_race_result_l643_64386


namespace NUMINAMATH_CALUDE_solve_linear_system_l643_64389

theorem solve_linear_system (x y a : ℝ) 
  (eq1 : 4 * x + 3 * y = 1)
  (eq2 : a * x + (a - 1) * y = 3)
  (eq3 : x = y) :
  a = 11 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_system_l643_64389


namespace NUMINAMATH_CALUDE_cubes_fill_box_completely_l643_64392

def box_length : ℕ := 12
def box_width : ℕ := 6
def box_height : ℕ := 9
def cube_side : ℕ := 3

def cubes_per_length : ℕ := box_length / cube_side
def cubes_per_width : ℕ := box_width / cube_side
def cubes_per_height : ℕ := box_height / cube_side

def total_cubes : ℕ := cubes_per_length * cubes_per_width * cubes_per_height

def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side ^ 3
def total_cube_volume : ℕ := total_cubes * cube_volume

theorem cubes_fill_box_completely :
  total_cube_volume = box_volume := by sorry

end NUMINAMATH_CALUDE_cubes_fill_box_completely_l643_64392


namespace NUMINAMATH_CALUDE_exists_unrepresentable_group_l643_64336

/-- Represents a person in the group -/
structure Person :=
  (id : ℕ)

/-- Represents the acquaintance relationship between two people -/
def Acquainted (p1 p2 : Person) : Prop := sorry

/-- Represents a chord in a circle -/
structure Chord :=
  (person : Person)

/-- Represents the intersection of two chords -/
def Intersects (c1 c2 : Chord) : Prop := sorry

/-- The main theorem stating that there exists a group of people whose acquaintance relationships
    cannot be represented by intersecting chords in a circle -/
theorem exists_unrepresentable_group :
  ∃ (group : Set Person) (acquaintance : Person → Person → Prop),
    ¬∃ (chord_assignment : Person → Chord),
      ∀ (p1 p2 : Person),
        p1 ∈ group → p2 ∈ group → p1 ≠ p2 →
          (acquaintance p1 p2 ↔ Intersects (chord_assignment p1) (chord_assignment p2)) :=
sorry

end NUMINAMATH_CALUDE_exists_unrepresentable_group_l643_64336


namespace NUMINAMATH_CALUDE_sarah_reading_speed_l643_64316

/-- Calculates Sarah's reading speed in words per minute -/
def sarahReadingSpeed (wordsPerPage : ℕ) (pagesPerBook : ℕ) (readingHours : ℕ) (numberOfBooks : ℕ) : ℕ :=
  let totalWords := wordsPerPage * pagesPerBook * numberOfBooks
  let totalMinutes := readingHours * 60
  totalWords / totalMinutes

/-- Theorem stating Sarah's reading speed under given conditions -/
theorem sarah_reading_speed :
  sarahReadingSpeed 100 80 20 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sarah_reading_speed_l643_64316


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l643_64338

theorem triangle_max_perimeter :
  ∀ a b : ℕ,
  b = 4 * a →
  (a + b > 16 ∧ a + 16 > b ∧ b + 16 > a) →
  a + b + 16 ≤ 41 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l643_64338


namespace NUMINAMATH_CALUDE_odd_induction_l643_64341

theorem odd_induction (P : ℕ → Prop) 
  (base : P 1) 
  (step : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) : 
  ∀ n : ℕ, n > 0 ∧ Odd n → P n :=
sorry

end NUMINAMATH_CALUDE_odd_induction_l643_64341


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l643_64364

theorem simplify_sqrt_expression :
  Real.sqrt 768 / Real.sqrt 192 - Real.sqrt 98 / Real.sqrt 49 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l643_64364


namespace NUMINAMATH_CALUDE_range_of_k_when_q_range_of_a_when_p_necessary_not_sufficient_l643_64378

-- Define propositions p and q
def p (k a : ℝ) : Prop := ∃ x y : ℝ, x^2/(k-1) + y^2/(7-a) = 1 ∧ k ≠ 1 ∧ a ≠ 7

def q (k : ℝ) : Prop := ¬∃ x y : ℝ, (4-k)*x^2 + (k-2)*y^2 = 1 ∧ (4-k)*(k-2) < 0

-- Theorem 1: Range of k when q is true
theorem range_of_k_when_q (k : ℝ) : q k → 2 ≤ k ∧ k ≤ 4 :=
sorry

-- Theorem 2: Range of a when p is a necessary but not sufficient condition for q
theorem range_of_a_when_p_necessary_not_sufficient (a : ℝ) :
  (∀ k : ℝ, q k → (∃ k', p k' a)) ∧ (∃ k : ℝ, p k a ∧ ¬q k) → a < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_when_q_range_of_a_when_p_necessary_not_sufficient_l643_64378


namespace NUMINAMATH_CALUDE_hat_scarf_game_theorem_l643_64333

/-- Represents the maximum guaranteed points in the hat-scarf game -/
def max_guaranteed_points (n k : ℕ) : ℕ :=
  n / k

theorem hat_scarf_game_theorem :
  (∀ n k : ℕ, max_guaranteed_points n k = n / k) ∧
  (max_guaranteed_points 2 2 = 1) := by
  sorry

#check hat_scarf_game_theorem

end NUMINAMATH_CALUDE_hat_scarf_game_theorem_l643_64333


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l643_64345

/-- For a quadratic equation x^2 + 2x + 4c = 0 to have two distinct real roots, c must be less than 1/4 -/
theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) →
  c < (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l643_64345


namespace NUMINAMATH_CALUDE_triangle_inequality_l643_64321

/-- Triangle inequality theorem -/
theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_triangle : A + B + C = π)
  (h_sine_law : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)) :
  A * a + B * b + C * c ≥ (1/2) * (A * b + B * a + A * c + C * a + B * c + C * b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l643_64321


namespace NUMINAMATH_CALUDE_special_sequence_sixth_term_l643_64346

/-- A sequence of positive integers where each term after the first is 1/3 of the sum of the term that precedes it and the term that follows it. -/
def SpecialSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > 0 ∧ a (n + 1) = (a n + a (n + 2)) / 3

theorem special_sequence_sixth_term
  (a : ℕ → ℚ)
  (h_seq : SpecialSequence a)
  (h_first : a 1 = 3)
  (h_fifth : a 5 = 54) :
  a 6 = 1133 / 7 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sixth_term_l643_64346


namespace NUMINAMATH_CALUDE_line_equation_through_parabola_intersection_l643_64356

/-- The equation of a line passing through (0, 2) and intersecting the parabola y² = 2x
    at two points M and N, where OM · ON = 0, is x + y - 2 = 0 -/
theorem line_equation_through_parabola_intersection (x y : ℝ) :
  let parabola := (fun (x y : ℝ) ↦ y^2 = 2*x)
  let line := (fun (x y : ℝ) ↦ ∃ (k : ℝ), y = k*x + 2)
  let O := (0, 0)
  let M := (x, y)
  let N := (2/y, y)  -- Using the parabola equation to express N
  parabola x y ∧
  line x y ∧
  (M.1 * N.1 + M.2 * N.2 = 0)  -- OM · ON = 0
  →
  x + y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_parabola_intersection_l643_64356


namespace NUMINAMATH_CALUDE_CaCO3_molecular_weight_l643_64359

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Calcium atoms in CaCO3 -/
def num_Ca : ℕ := 1

/-- The number of Carbon atoms in CaCO3 -/
def num_C : ℕ := 1

/-- The number of Oxygen atoms in CaCO3 -/
def num_O : ℕ := 3

/-- The molecular weight of CaCO3 in g/mol -/
def molecular_weight_CaCO3 : ℝ :=
  num_Ca * atomic_weight_Ca + num_C * atomic_weight_C + num_O * atomic_weight_O

theorem CaCO3_molecular_weight :
  molecular_weight_CaCO3 = 100.09 := by sorry

end NUMINAMATH_CALUDE_CaCO3_molecular_weight_l643_64359


namespace NUMINAMATH_CALUDE_volunteer_distribution_l643_64337

/-- The number of ways to distribute n distinguishable volunteers among k distinguishable places,
    such that each place has at least one volunteer. -/
def distribute_volunteers (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k * (k-1) / 2) * (k-2)^n

/-- The problem statement -/
theorem volunteer_distribution :
  distribute_volunteers 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l643_64337


namespace NUMINAMATH_CALUDE_two_distinct_integer_roots_l643_64357

theorem two_distinct_integer_roots (r : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ r^2 * x^2 + 2*r*x + 4 = 28*r^2 ∧ r^2 * y^2 + 2*r*y + 4 = 28*r^2) ↔ 
  (r = 1 ∨ r = -1 ∨ r = 1/2 ∨ r = -1/2 ∨ r = 1/3 ∨ r = -1/3) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_integer_roots_l643_64357


namespace NUMINAMATH_CALUDE_correct_arrangement_l643_64353

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 6 ∧
  (∀ n, n ∈ arr → n ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ i, i < 3 → is_perfect_square (arr[2*i]! * arr[2*i+1]!))

theorem correct_arrangement :
  valid_arrangement [4, 2, 5, 3, 6, 1] :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_l643_64353


namespace NUMINAMATH_CALUDE_increase_by_percentage_l643_64303

/-- Theorem: Increasing 350 by 50% results in 525. -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 350 → percentage = 50 → result = initial * (1 + percentage / 100) → result = 525 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l643_64303


namespace NUMINAMATH_CALUDE_optimal_boat_combinations_l643_64397

/-- Represents a combination of large and small boats -/
structure BoatCombination where
  large_boats : Nat
  small_boats : Nat

/-- Checks if a boat combination is valid for the given number of people -/
def is_valid_combination (total_people : Nat) (large_capacity : Nat) (small_capacity : Nat) (combo : BoatCombination) : Prop :=
  combo.large_boats * large_capacity + combo.small_boats * small_capacity = total_people

theorem optimal_boat_combinations : 
  ∃ (combo1 combo2 : BoatCombination),
    combo1 ≠ combo2 ∧
    is_valid_combination 43 7 4 combo1 ∧
    is_valid_combination 43 7 4 combo2 :=
by sorry

end NUMINAMATH_CALUDE_optimal_boat_combinations_l643_64397


namespace NUMINAMATH_CALUDE_quadratic_factorization_l643_64371

theorem quadratic_factorization (x : ℝ) : 12 * x^2 + 8 * x - 4 = 4 * (3 * x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l643_64371


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l643_64384

theorem cube_root_equation_solution : 
  {x : ℝ | ∃ y : ℝ, y^3 = 4*x - 1 ∧ ∃ z : ℝ, z^3 = 4*x + 1 ∧ ∃ w : ℝ, w^3 = 8*x ∧ y + z = w} = 
  {0, 1/4, -1/4} := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l643_64384


namespace NUMINAMATH_CALUDE_shrinking_cities_proportion_comparison_l643_64315

/-- Represents a circle in Hubei province -/
structure Circle where
  total_cities : ℕ
  shrinking_cities : ℕ
  shrinking_cities_le_total : shrinking_cities ≤ total_cities

/-- Calculates the proportion of shrinking cities in a circle -/
def shrinking_proportion (c : Circle) : ℚ :=
  c.shrinking_cities / c.total_cities

theorem shrinking_cities_proportion_comparison 
  (west : Circle)
  (middle : Circle)
  (east : Circle)
  (hw : west.total_cities = 5 ∧ west.shrinking_cities = 5)
  (hm : middle.total_cities = 13 ∧ middle.shrinking_cities = 9)
  (he : east.total_cities = 18 ∧ east.shrinking_cities = 13) :
  shrinking_proportion middle < shrinking_proportion west ∧ 
  shrinking_proportion middle < shrinking_proportion east :=
sorry

end NUMINAMATH_CALUDE_shrinking_cities_proportion_comparison_l643_64315


namespace NUMINAMATH_CALUDE_geraldine_doll_count_l643_64313

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℝ := 1209.0

/-- The number of additional dolls Geraldine has compared to Jazmin -/
def additional_dolls : ℕ := 977

/-- The total number of dolls Geraldine has -/
def geraldine_dolls : ℝ := jazmin_dolls + additional_dolls

theorem geraldine_doll_count : geraldine_dolls = 2186 := by
  sorry

end NUMINAMATH_CALUDE_geraldine_doll_count_l643_64313


namespace NUMINAMATH_CALUDE_product_sum_zero_l643_64372

theorem product_sum_zero (a b c d : ℚ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(9 - 3*x) = a*x^3 + b*x^2 + c*x + d) → 
  27*a + 9*b + 3*c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_zero_l643_64372


namespace NUMINAMATH_CALUDE_young_bonnets_theorem_l643_64304

/-- Calculates the number of bonnets Mrs. Young sends to each orphanage --/
def bonnets_per_orphanage : ℕ :=
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let saturday := friday - 8
  let sunday := 3 * saturday
  let total := monday + tuesday_wednesday + thursday + friday + saturday + sunday
  total / 10

/-- Theorem stating that Mrs. Young sends 6 bonnets to each orphanage --/
theorem young_bonnets_theorem : bonnets_per_orphanage = 6 := by
  sorry

end NUMINAMATH_CALUDE_young_bonnets_theorem_l643_64304


namespace NUMINAMATH_CALUDE_jordan_running_time_l643_64388

theorem jordan_running_time (steve_time steve_distance jordan_distance_1 jordan_distance_2 : ℚ)
  (h1 : steve_time = 32)
  (h2 : steve_distance = 4)
  (h3 : jordan_distance_1 = 3)
  (h4 : jordan_distance_2 = 7)
  (h5 : jordan_distance_1 / (steve_time / 2) = steve_distance / steve_time) :
  jordan_distance_2 / (jordan_distance_1 / (steve_time / 2)) = 112 / 3 := by sorry

end NUMINAMATH_CALUDE_jordan_running_time_l643_64388


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l643_64329

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 240) (h2 : P + 6 = N / 4 - 6) : 
  (P + 6) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l643_64329


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l643_64349

theorem ratio_sum_problem (a b : ℝ) : 
  a / b = 3 / 8 → b - a = 20 → a + b = 44 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l643_64349


namespace NUMINAMATH_CALUDE_michael_pizza_portion_l643_64355

theorem michael_pizza_portion
  (total_pizza : ℚ)
  (treshawn_portion : ℚ)
  (lamar_portion : ℚ)
  (h1 : total_pizza = 1)
  (h2 : treshawn_portion = 1 / 2)
  (h3 : lamar_portion = 1 / 6)
  : total_pizza - (treshawn_portion + lamar_portion) = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_michael_pizza_portion_l643_64355


namespace NUMINAMATH_CALUDE_expression_evaluation_l643_64301

theorem expression_evaluation :
  let w : ℤ := 3
  let x : ℤ := -2
  let y : ℤ := 1
  let z : ℤ := 4
  w^2 * x^2 * y * z - w * x^2 * y * z^2 + w * y^3 * z^2 - w * y^2 * x * z^4 = 1536 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l643_64301


namespace NUMINAMATH_CALUDE_lindas_painting_area_l643_64390

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular opening in a wall -/
structure Opening where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ := width * height

/-- Calculates the total wall area of a room -/
def totalWallArea (room : RoomDimensions) : ℝ :=
  2 * (room.width * room.height + room.length * room.height)

/-- Calculates the area of an opening -/
def openingArea (opening : Opening) : ℝ :=
  rectangleArea opening.width opening.height

/-- Represents Linda's bedroom -/
def lindasBedroom : RoomDimensions := {
  width := 20,
  length := 20,
  height := 8
}

/-- Represents the doorway in Linda's bedroom -/
def doorway : Opening := {
  width := 3,
  height := 7
}

/-- Represents the window in Linda's bedroom -/
def window : Opening := {
  width := 6,
  height := 4
}

/-- Represents the closet doorway in Linda's bedroom -/
def closetDoorway : Opening := {
  width := 5,
  height := 7
}

/-- Theorem stating the total area of wall space Linda will have to paint -/
theorem lindas_painting_area :
  totalWallArea lindasBedroom -
  (openingArea doorway + openingArea window + openingArea closetDoorway) = 560 := by
  sorry

end NUMINAMATH_CALUDE_lindas_painting_area_l643_64390


namespace NUMINAMATH_CALUDE_f_vertex_f_at_zero_f_expression_f_monotonic_interval_l643_64360

/-- A quadratic function with vertex at (1, 1) and f(0) = 3 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The vertex of f is at (1, 1) -/
theorem f_vertex : ∀ x : ℝ, f x ≥ f 1 := sorry

/-- f(0) = 3 -/
theorem f_at_zero : f 0 = 3 := sorry

/-- f(x) = 2x^2 - 4x + 3 -/
theorem f_expression : ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 3 := sorry

/-- f(x) is monotonic in [a, a+1] iff a ≤ 0 or a ≥ 1 -/
theorem f_monotonic_interval (a : ℝ) :
  (∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ a + 1 → f x ≤ f y) ↔ (a ≤ 0 ∨ a ≥ 1) := sorry

end NUMINAMATH_CALUDE_f_vertex_f_at_zero_f_expression_f_monotonic_interval_l643_64360


namespace NUMINAMATH_CALUDE_quadratic_factorization_l643_64375

theorem quadratic_factorization (a b : ℤ) :
  (∀ x, 24 * x^2 - 98 * x - 168 = (6 * x + a) * (4 * x + b)) →
  a + 2 * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l643_64375


namespace NUMINAMATH_CALUDE_four_propositions_l643_64305

-- Definition of opposite numbers
def are_opposite (x y : ℝ) : Prop := x = -y

-- Definition of congruent triangles
def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of triangle area
def triangle_area (t : Set ℝ × Set ℝ) : ℝ := sorry

-- Statement of the theorem
theorem four_propositions :
  (∀ x y : ℝ, are_opposite x y → x + y = 0) ∧
  (∀ q : ℝ, (∃ x : ℝ, x^2 + 2*x + q = 0) → q ≤ 1) ∧
  (∃ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) ∧ triangle_area t1 = triangle_area t2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) :=
by sorry

end NUMINAMATH_CALUDE_four_propositions_l643_64305


namespace NUMINAMATH_CALUDE_gcd_21_and_number_between_50_60_l643_64383

theorem gcd_21_and_number_between_50_60 :
  ∃! n : ℕ, 50 ≤ n ∧ n ≤ 60 ∧ Nat.gcd 21 n = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_gcd_21_and_number_between_50_60_l643_64383


namespace NUMINAMATH_CALUDE_division_reduction_l643_64382

theorem division_reduction (x : ℝ) (h : x > 0) : 54 / x = 54 - 36 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l643_64382


namespace NUMINAMATH_CALUDE_line_properties_l643_64323

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given the equation of a line y + 7 = -x - 3, prove it passes through (-3, -7) with slope -1 -/
theorem line_properties :
  let l : Line := { slope := -1, yIntercept := -10 }
  let p : Point := { x := -3, y := -7 }
  (p.y + 7 = -p.x - 3) ∧ 
  (l.slope = -1) ∧
  (p.y = l.slope * p.x + l.yIntercept) := by
  sorry


end NUMINAMATH_CALUDE_line_properties_l643_64323


namespace NUMINAMATH_CALUDE_polynomial_expansion_l643_64348

theorem polynomial_expansion (x : ℝ) : 
  (x^3 - 3*x^2 + 3*x - 1) * (x^2 + 3*x + 3) = x^5 - 3*x^3 - x^2 + 3*x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l643_64348


namespace NUMINAMATH_CALUDE_height_range_is_75cm_l643_64319

/-- The range of a set of values is the difference between the maximum and minimum values. -/
def range (max min : ℝ) : ℝ := max - min

/-- The heights of five students at Gleeson Middle School. -/
structure StudentHeights where
  num_students : ℕ
  max_height : ℝ
  min_height : ℝ

/-- The range of heights of the students is 75 cm. -/
theorem height_range_is_75cm (heights : StudentHeights) 
  (h1 : heights.num_students = 5)
  (h2 : heights.max_height = 175)
  (h3 : heights.min_height = 100) : 
  range heights.max_height heights.min_height = 75 := by
sorry

end NUMINAMATH_CALUDE_height_range_is_75cm_l643_64319


namespace NUMINAMATH_CALUDE_vector_magnitude_cosine_sine_l643_64324

theorem vector_magnitude_cosine_sine (α : Real) : 
  let a : Fin 2 → Real := ![Real.cos α, Real.sin α]
  ‖a‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_cosine_sine_l643_64324


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l643_64307

theorem complex_power_magnitude : Complex.abs ((1 + Complex.I) ^ 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l643_64307


namespace NUMINAMATH_CALUDE_optimal_path_to_island_l643_64343

/-- Represents the optimal path problem for Hagrid to reach Harry Potter --/
theorem optimal_path_to_island (island_distance : ℝ) (shore_distance : ℝ) 
  (shore_speed : ℝ) (sea_speed : ℝ) :
  island_distance = 9 →
  shore_distance = 15 →
  shore_speed = 50 →
  sea_speed = 40 →
  ∃ (x : ℝ), x = 3 ∧ 
    ∀ (y : ℝ), y ≥ 0 → 
      (x / shore_speed + (Real.sqrt ((island_distance^2) + (shore_distance - x)^2)) / sea_speed) ≤
      (y / shore_speed + (Real.sqrt ((island_distance^2) + (shore_distance - y)^2)) / sea_speed) :=
by sorry


end NUMINAMATH_CALUDE_optimal_path_to_island_l643_64343
