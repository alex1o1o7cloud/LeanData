import Mathlib

namespace NUMINAMATH_CALUDE_lending_time_combined_l35_3529

-- Define the lending time for chocolate bars and bonbons
def lending_time_chocolate (bars : ℚ) : ℚ := (3 / 2) * bars

def lending_time_bonbons (bonbons : ℚ) : ℚ := (1 / 6) * bonbons

-- Theorem to prove
theorem lending_time_combined : 
  lending_time_chocolate 1 + lending_time_bonbons 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lending_time_combined_l35_3529


namespace NUMINAMATH_CALUDE_kelly_baking_powder_l35_3513

def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1

theorem kelly_baking_powder :
  let current_amount := yesterday_amount - difference
  current_amount = 0.3 := by
sorry

end NUMINAMATH_CALUDE_kelly_baking_powder_l35_3513


namespace NUMINAMATH_CALUDE_perfect_square_digits_l35_3512

theorem perfect_square_digits (a b x y : ℕ) : 
  (∃ n : ℕ, a = n^2) →  -- a is a perfect square
  (∃ m : ℕ, b = m^2) →  -- b is a perfect square
  a % 10 = 1 →         -- unit digit of a is 1
  (a / 10) % 10 = x →  -- tens digit of a is x
  b % 10 = 6 →         -- unit digit of b is 6
  (b / 10) % 10 = y →  -- tens digit of b is y
  Even x ∧ Odd y :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_digits_l35_3512


namespace NUMINAMATH_CALUDE_regular_tetradecagon_side_length_l35_3583

/-- A regular tetradecagon with perimeter 154 cm has sides of length 11 cm. -/
theorem regular_tetradecagon_side_length :
  ∀ (perimeter : ℝ) (num_sides : ℕ) (side_length : ℝ),
    perimeter = 154 →
    num_sides = 14 →
    side_length * num_sides = perimeter →
    side_length = 11 :=
by sorry

end NUMINAMATH_CALUDE_regular_tetradecagon_side_length_l35_3583


namespace NUMINAMATH_CALUDE_unique_solution_for_n_equals_one_l35_3530

theorem unique_solution_for_n_equals_one (n : ℕ+) :
  (∃ x : ℤ, x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_n_equals_one_l35_3530


namespace NUMINAMATH_CALUDE_geometric_series_sum_l35_3539

theorem geometric_series_sum : 
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 15
  let series_sum := (a * (1 - r^n)) / (1 - r)
  series_sum = 3216929751 / 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l35_3539


namespace NUMINAMATH_CALUDE_polyhedron_volume_l35_3565

/-- Represents a polygon in the figure -/
inductive Polygon
| ScaleneRightTriangle : Polygon
| Rectangle : Polygon
| EquilateralTriangle : Polygon

/-- The figure consisting of multiple polygons -/
structure Figure where
  scaleneTriangles : Fin 3 → Polygon
  rectangles : Fin 3 → Polygon
  equilateralTriangle : Polygon
  scaleneTriangleLegs : ℝ × ℝ
  rectangleDimensions : ℝ × ℝ

/-- The polyhedron formed by folding the figure -/
def Polyhedron (f : Figure) : Type := Unit

/-- The volume of the polyhedron -/
noncomputable def volume (p : Polyhedron f) : ℝ := sorry

/-- The main theorem stating the volume of the polyhedron is 4 -/
theorem polyhedron_volume (f : Figure)
  (h1 : ∀ i, f.scaleneTriangles i = Polygon.ScaleneRightTriangle)
  (h2 : ∀ i, f.rectangles i = Polygon.Rectangle)
  (h3 : f.equilateralTriangle = Polygon.EquilateralTriangle)
  (h4 : f.scaleneTriangleLegs = (1, 2))
  (h5 : f.rectangleDimensions = (1, 2))
  (p : Polyhedron f) :
  volume p = 4 := by sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l35_3565


namespace NUMINAMATH_CALUDE_fourth_circle_radius_l35_3509

def circle_configuration (radii : Fin 7 → ℝ) : Prop :=
  ∀ i : Fin 6, radii i < radii (i + 1) ∧ 
  ∃ (r : ℝ), ∀ i : Fin 6, radii (i + 1) = radii i * r

theorem fourth_circle_radius 
  (radii : Fin 7 → ℝ) 
  (h_config : circle_configuration radii) 
  (h_smallest : radii 0 = 6) 
  (h_largest : radii 6 = 24) : 
  radii 3 = 12 :=
sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_l35_3509


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l35_3576

theorem divisibility_by_nine : ∃ k : ℤ, 8 * 10^18 + 1^18 = 9 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l35_3576


namespace NUMINAMATH_CALUDE_sum_of_distances_l35_3504

/-- A circle touches the sides of an angle at points A and B. 
    C is a point on the circle. -/
structure CircleConfig where
  A : Point
  B : Point
  C : Point

/-- The distance from C to line AB is 6 -/
def distance_to_AB (config : CircleConfig) : ℝ := 6

/-- The distances from C to the sides of the angle -/
def distance_to_sides (config : CircleConfig) : ℝ × ℝ := sorry

/-- One distance is 9 times less than the other -/
axiom distance_ratio (config : CircleConfig) : 
  let (d₁, d₂) := distance_to_sides config
  d₁ = (1/9) * d₂ ∨ d₂ = (1/9) * d₁

theorem sum_of_distances (config : CircleConfig) : 
  let (d₁, d₂) := distance_to_sides config
  distance_to_AB config + d₁ + d₂ = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_l35_3504


namespace NUMINAMATH_CALUDE_maximize_x_3_minus_3x_l35_3594

theorem maximize_x_3_minus_3x :
  ∀ x : ℝ, 0 < x → x < 1 → x * (3 - 3 * x) ≤ 3 / 4 ∧
  (x * (3 - 3 * x) = 3 / 4 ↔ x = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_maximize_x_3_minus_3x_l35_3594


namespace NUMINAMATH_CALUDE_cookie_problem_l35_3549

theorem cookie_problem (total_cookies : ℕ) (nuts_per_cookie : ℕ) (fraction_with_nuts : ℚ) (total_nuts_used : ℕ) :
  nuts_per_cookie = 2 →
  fraction_with_nuts = 1/4 →
  total_nuts_used = 72 →
  (fraction_with_nuts * total_cookies : ℚ).num * nuts_per_cookie = total_nuts_used →
  total_cookies = 144 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l35_3549


namespace NUMINAMATH_CALUDE_equality_of_expressions_l35_3561

theorem equality_of_expressions (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_expressions_l35_3561


namespace NUMINAMATH_CALUDE_age_pencil_ratio_l35_3535

/-- Given the ages and pencil counts of Asaf and Alexander, prove the ratio of their age difference to Asaf's pencil count -/
theorem age_pencil_ratio (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age + alexander_age = 140 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 →
  (alexander_age - asaf_age : ℚ) / asaf_pencils = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_age_pencil_ratio_l35_3535


namespace NUMINAMATH_CALUDE_house_wall_planks_l35_3556

/-- The number of large planks needed for the house wall. -/
def large_planks : ℕ := 12

/-- The number of small planks needed for the house wall. -/
def small_planks : ℕ := 17

/-- The total number of planks needed for the house wall. -/
def total_planks : ℕ := large_planks + small_planks

theorem house_wall_planks : total_planks = 29 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_planks_l35_3556


namespace NUMINAMATH_CALUDE_ashley_champagne_toast_l35_3501

/-- The number of bottles of champagne needed for a wedding toast --/
def bottlesNeeded (guests : ℕ) (glassesPerGuest : ℕ) (servingsPerBottle : ℕ) : ℕ :=
  (guests * glassesPerGuest + servingsPerBottle - 1) / servingsPerBottle

/-- Theorem: Ashley needs 40 bottles of champagne for her wedding toast --/
theorem ashley_champagne_toast :
  bottlesNeeded 120 2 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashley_champagne_toast_l35_3501


namespace NUMINAMATH_CALUDE_sugar_for_recipe_l35_3540

/-- The amount of sugar needed for a cake recipe, given the amounts for frosting and cake. -/
theorem sugar_for_recipe (frosting_sugar cake_sugar : ℚ) 
  (h1 : frosting_sugar = 0.6)
  (h2 : cake_sugar = 0.2) : 
  frosting_sugar + cake_sugar = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_recipe_l35_3540


namespace NUMINAMATH_CALUDE_spencer_jumps_l35_3581

/-- Calculates the total number of jumps Spencer will do in 5 days -/
def total_jumps (jumps_per_minute : ℕ) (minutes_per_session : ℕ) (sessions_per_day : ℕ) (days : ℕ) : ℕ :=
  jumps_per_minute * minutes_per_session * sessions_per_day * days

/-- Theorem stating that Spencer will do 400 jumps in 5 days -/
theorem spencer_jumps :
  total_jumps 4 10 2 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_spencer_jumps_l35_3581


namespace NUMINAMATH_CALUDE_k_is_even_if_adjacent_to_odds_l35_3550

/-- A circular arrangement of numbers from 1 to 1000 -/
def CircularArrangement := Fin 1000 → ℕ

/-- Property that each number is a divisor of the sum of its neighbors -/
def IsDivisorOfNeighborsSum (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 1000, arr i ∣ (arr (i - 1) + arr (i + 1))

/-- Theorem: If k is adjacent to two odd numbers in a valid circular arrangement, then k is even -/
theorem k_is_even_if_adjacent_to_odds
  (arr : CircularArrangement)
  (h_valid : IsDivisorOfNeighborsSum arr)
  (k : Fin 1000)
  (h_k_adj_odd : Odd (arr (k - 1)) ∧ Odd (arr (k + 1))) :
  Even (arr k) := by
  sorry

end NUMINAMATH_CALUDE_k_is_even_if_adjacent_to_odds_l35_3550


namespace NUMINAMATH_CALUDE_inequality_holds_l35_3510

theorem inequality_holds (a b c : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) : a / c^2 > b / c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l35_3510


namespace NUMINAMATH_CALUDE_sum_quadratic_distinct_roots_l35_3544

/-- A quadratic function f(x) = x^2 + ax + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- The discriminant of a quadratic function -/
def discriminant (f : QuadraticFunction) : ℝ := f.a^2 - 4*f.b

/-- The sum of two quadratic functions -/
def sum_quadratic (f g : QuadraticFunction) : QuadraticFunction :=
  ⟨f.a + g.a, f.b + g.b⟩

/-- The sum of a list of quadratic functions -/
def sum_quadratic_list (fs : List QuadraticFunction) : QuadraticFunction :=
  fs.foldl sum_quadratic ⟨0, 0⟩

/-- Theorem: Given conditions on quadratic functions, their sum has distinct real roots -/
theorem sum_quadratic_distinct_roots
  (n : ℕ)
  (hn : n ≥ 2)
  (fs : List QuadraticFunction)
  (hfs : fs.length = n)
  (h_same_discriminant : ∀ (f g : QuadraticFunction), f ∈ fs → g ∈ fs → discriminant f = discriminant g)
  (h_distinct_roots : ∀ (f g : QuadraticFunction), f ∈ fs → g ∈ fs → f ≠ g →
    (discriminant (sum_quadratic f g) > 0)) :
  discriminant (sum_quadratic_list fs) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_quadratic_distinct_roots_l35_3544


namespace NUMINAMATH_CALUDE_cylinder_height_after_forging_l35_3526

theorem cylinder_height_after_forging (initial_diameter initial_height new_diameter : ℝ) 
  (h_initial_diameter : initial_diameter = 6)
  (h_initial_height : initial_height = 24)
  (h_new_diameter : new_diameter = 16) :
  let new_height := (initial_diameter^2 * initial_height) / new_diameter^2
  new_height = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_cylinder_height_after_forging_l35_3526


namespace NUMINAMATH_CALUDE_expression_simplification_l35_3578

theorem expression_simplification 
  (a b c k x : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_k_nonzero : k ≠ 0) :
  k * ((x + a)^2 / ((a - b)*(a - c)) + 
       (x + b)^2 / ((b - a)*(b - c)) + 
       (x + c)^2 / ((c - a)*(c - b))) = k :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l35_3578


namespace NUMINAMATH_CALUDE_building_height_calculation_l35_3507

-- Define the given constants
def box_height : ℝ := 3
def box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- Define the theorem
theorem building_height_calculation :
  ∃ (building_height : ℝ),
    (box_height / box_shadow = building_height / building_shadow) ∧
    building_height = 9 := by
  sorry

end NUMINAMATH_CALUDE_building_height_calculation_l35_3507


namespace NUMINAMATH_CALUDE_inequality_max_a_l35_3566

theorem inequality_max_a : 
  (∀ x : ℝ, x ∈ Set.Icc 1 12 → x^2 + 25 + |x^3 - 5*x^2| ≥ (5/2)*x) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x ∈ Set.Icc 1 12 ∧ x^2 + 25 + |x^3 - 5*x^2| < (5/2 + ε)*x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_max_a_l35_3566


namespace NUMINAMATH_CALUDE_billy_watches_95_videos_billy_within_time_constraint_l35_3536

/-- The number of videos Billy watches in total -/
def total_videos (suggestions_per_trial : ℕ) (num_trials : ℕ) (suggestions_per_category : ℕ) (num_categories : ℕ) : ℕ :=
  suggestions_per_trial * num_trials + suggestions_per_category * num_categories

/-- Theorem stating that Billy watches 95 videos in total -/
theorem billy_watches_95_videos :
  total_videos 15 5 10 2 = 95 := by
  sorry

/-- Billy's time constraint in minutes -/
def time_constraint : ℕ := 60

/-- Time taken to watch each video in minutes -/
def time_per_video : ℕ := 4

/-- Theorem stating that Billy's total watching time does not exceed the time constraint -/
theorem billy_within_time_constraint :
  total_videos 15 5 10 2 * time_per_video ≤ time_constraint := by
  sorry

end NUMINAMATH_CALUDE_billy_watches_95_videos_billy_within_time_constraint_l35_3536


namespace NUMINAMATH_CALUDE_solve_email_problem_l35_3548

def email_problem (initial_delete : ℕ) (first_receive : ℕ) (second_delete : ℕ) (final_receive : ℕ) (final_count : ℕ) : Prop :=
  ∃ (x : ℕ), 
    initial_delete = 50 ∧
    first_receive = 15 ∧
    second_delete = 20 ∧
    final_receive = 10 ∧
    final_count = 30 ∧
    first_receive + x + final_receive = final_count ∧
    x = 5

theorem solve_email_problem :
  ∃ (initial_delete first_receive second_delete final_receive final_count : ℕ),
    email_problem initial_delete first_receive second_delete final_receive final_count :=
by
  sorry

end NUMINAMATH_CALUDE_solve_email_problem_l35_3548


namespace NUMINAMATH_CALUDE_remainder_theorem_l35_3577

theorem remainder_theorem (x : ℤ) : 
  (2*x + 3)^504 ≡ 16*x + 5 [ZMOD (x^2 - x + 1)] :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l35_3577


namespace NUMINAMATH_CALUDE_cookies_per_pack_l35_3585

theorem cookies_per_pack (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ) 
  (h1 : trays = 8) 
  (h2 : cookies_per_tray = 36) 
  (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 := by
  sorry

#check cookies_per_pack

end NUMINAMATH_CALUDE_cookies_per_pack_l35_3585


namespace NUMINAMATH_CALUDE_power_of_power_l35_3588

theorem power_of_power : (3^4)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l35_3588


namespace NUMINAMATH_CALUDE_quadratic_minimum_l35_3541

/-- The function f(x) = 5x^2 - 20x + 7 has a minimum value when x = 2 -/
theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ (y : ℝ), 5 * x^2 - 20 * x + 7 ≥ 5 * min_x^2 - 20 * min_x + 7 ∧ min_x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l35_3541


namespace NUMINAMATH_CALUDE_paperclip_growth_l35_3521

theorem paperclip_growth (n : ℕ) : (8 * 3^n > 1000) ↔ n ≥ 5 := by sorry

end NUMINAMATH_CALUDE_paperclip_growth_l35_3521


namespace NUMINAMATH_CALUDE_parabola_complementary_lines_l35_3546

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

theorem parabola_complementary_lines (para : Parabola) (P A B : Point)
  (h_P_on_para : P.y^2 = 2 * para.p * P.x)
  (h_P_y_pos : P.y > 0)
  (h_A_on_para : A.y^2 = 2 * para.p * A.x)
  (h_B_on_para : B.y^2 = 2 * para.p * B.x)
  (h_PA_slope_exists : A.x ≠ P.x)
  (h_PB_slope_exists : B.x ≠ P.x)
  (h_complementary : 
    (A.y - P.y) / (A.x - P.x) * (B.y - P.y) / (B.x - P.x) = -1) :
  (A.y + B.y) / P.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_complementary_lines_l35_3546


namespace NUMINAMATH_CALUDE_maryann_rescue_time_l35_3511

/-- The time (in minutes) it takes Maryann to pick the lock on a cheap pair of handcuffs -/
def cheap_handcuff_time : ℕ := 6

/-- The time (in minutes) it takes Maryann to pick the lock on an expensive pair of handcuffs -/
def expensive_handcuff_time : ℕ := 8

/-- The number of friends Maryann needs to rescue -/
def number_of_friends : ℕ := 3

/-- The time it takes to free one friend -/
def time_per_friend : ℕ := cheap_handcuff_time + expensive_handcuff_time

/-- The total time it takes to free all friends -/
def total_rescue_time : ℕ := time_per_friend * number_of_friends

theorem maryann_rescue_time : total_rescue_time = 42 := by
  sorry

end NUMINAMATH_CALUDE_maryann_rescue_time_l35_3511


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l35_3533

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 →                             -- One side is 12 cm
  (1/2) * a * b = 54 →                 -- Area of the triangle is 54 square centimeters
  a^2 + b^2 = c^2 →                    -- Pythagorean theorem (right-angled triangle)
  c = 15 :=                            -- Hypotenuse length is 15 cm
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l35_3533


namespace NUMINAMATH_CALUDE_factor_polynomial_l35_3575

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l35_3575


namespace NUMINAMATH_CALUDE_problem_1999_squared_minus_1998_times_2002_l35_3514

theorem problem_1999_squared_minus_1998_times_2002 : 1999^2 - 1998 * 2002 = -3991 := by
  sorry

end NUMINAMATH_CALUDE_problem_1999_squared_minus_1998_times_2002_l35_3514


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l35_3563

theorem triangle_cosine_problem (A B C a b c : ℝ) : 
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  -- a, b, c are sides opposite to A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Given condition
  ((Real.sqrt 3 * b - c) * Real.cos A = a * Real.cos C) →
  -- Conclusion
  Real.cos A = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l35_3563


namespace NUMINAMATH_CALUDE_negation_of_proposition_l35_3531

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l35_3531


namespace NUMINAMATH_CALUDE_range_of_m_l35_3519

def p (x : ℝ) : Prop := (x - 1) / x ≤ 0

def q (x m : ℝ) : Prop := (x - m) * (x - m + 2) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) → 1 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l35_3519


namespace NUMINAMATH_CALUDE_domain_shift_l35_3517

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 4

-- State the theorem
theorem domain_shift :
  (∀ x, f x ≠ 0 → x ∈ domain_f) →
  (∀ x, f (x - 1) ≠ 0 → x ∈ Set.Icc 2 5) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l35_3517


namespace NUMINAMATH_CALUDE_class_size_l35_3545

/-- Given a class of children where:
  * 19 play tennis
  * 21 play squash
  * 10 play neither sport
  * 12 play both sports
  This theorem proves that there are 38 children in the class. -/
theorem class_size (T S N B : ℕ) (h1 : T = 19) (h2 : S = 21) (h3 : N = 10) (h4 : B = 12) :
  T + S - B + N = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l35_3545


namespace NUMINAMATH_CALUDE_polygon_construction_possible_l35_3573

/-- Represents a line segment with a fixed length -/
structure Segment where
  length : ℝ

/-- Represents a polygon constructed from line segments -/
structure Polygon where
  segments : List Segment

/-- Calculates the area of a polygon -/
def calculateArea (p : Polygon) : ℝ := sorry

/-- Checks if all segments in a polygon are used -/
def allSegmentsUsed (p : Polygon) (segments : List Segment) : Prop := sorry

theorem polygon_construction_possible (segments : List Segment) :
  segments.length = 12 ∧ 
  ∀ s ∈ segments, s.length = 2 →
  ∃ p : Polygon, calculateArea p = 16 ∧ allSegmentsUsed p segments :=
sorry

end NUMINAMATH_CALUDE_polygon_construction_possible_l35_3573


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l35_3516

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 4/5, then the ratio of their areas is 16:25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) :
  (a * b) / (c * d) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l35_3516


namespace NUMINAMATH_CALUDE_dropped_student_score_l35_3572

theorem dropped_student_score 
  (initial_students : ℕ) 
  (remaining_students : ℕ) 
  (initial_average : ℚ) 
  (new_average : ℚ) 
  (h1 : initial_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 61.5) 
  (h4 : new_average = 64) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 24 := by
  sorry

#check dropped_student_score

end NUMINAMATH_CALUDE_dropped_student_score_l35_3572


namespace NUMINAMATH_CALUDE_fruit_basket_total_l35_3534

/-- Represents the number of fruit pieces in a basket -/
structure FruitBasket where
  redApples : Nat
  greenApples : Nat
  purpleGrapes : Nat
  yellowBananas : Nat
  orangeOranges : Nat

/-- Calculates the total number of fruit pieces in the basket -/
def totalFruits (basket : FruitBasket) : Nat :=
  basket.redApples + basket.greenApples + basket.purpleGrapes + basket.yellowBananas + basket.orangeOranges

/-- Theorem stating that the total number of fruit pieces in the given basket is 24 -/
theorem fruit_basket_total :
  let basket : FruitBasket := {
    redApples := 9,
    greenApples := 4,
    purpleGrapes := 3,
    yellowBananas := 6,
    orangeOranges := 2
  }
  totalFruits basket = 24 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_l35_3534


namespace NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l35_3506

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem quadratic_function_passes_through_points :
  f (-1) = 0 ∧ f 3 = 0 ∧ f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_passes_through_points_l35_3506


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l35_3520

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 7 = 45 →                     -- first given condition
  a 2 + a 5 + a 8 = 29 →                     -- second given condition
  a 3 + a 6 + a 9 = 13 :=                    -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l35_3520


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l35_3557

def initial_amount : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def found_money : ℚ := 7.43

theorem jacket_cost_calculation : 
  let remaining_after_shirt := initial_amount - shirt_cost
  let total_remaining := remaining_after_shirt + found_money
  total_remaining = 9.28 := by sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l35_3557


namespace NUMINAMATH_CALUDE_problem_solution_l35_3500

theorem problem_solution (x y z : ℝ) : 
  (3 * x = 0.75 * y) →
  (x = 24) →
  (z = 0.5 * y) →
  (z = 48) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l35_3500


namespace NUMINAMATH_CALUDE_f_upper_bound_and_g_monotonicity_l35_3551

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.log x + 1

def g (a x : ℝ) : ℝ := (f x - f a) / (x - a)

theorem f_upper_bound_and_g_monotonicity :
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) ∧
  (∀ c : ℝ, c < -1 → ∃ x : ℝ, x > 0 ∧ f x > 2 * x + c) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → g a x₁ > g a x₂) ∧
    (∀ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ → g a x₁ > g a x₂)) :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_and_g_monotonicity_l35_3551


namespace NUMINAMATH_CALUDE_solution_set_f_min_value_sum_equality_condition_l35_3538

-- Define the function f
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

-- Theorem 1: Solution set of f(x + 3/2) ≥ 0
theorem solution_set_f (x : ℝ) : 
  f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) := by sorry

-- Theorem 2: Minimum value of 3p + 2q + r
theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 := by sorry

-- Theorem 3: Condition for equality in Theorem 2
theorem equality_condition (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r = 9/4 ↔ p = 1/4 ∧ q = 3/8 ∧ r = 3/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_min_value_sum_equality_condition_l35_3538


namespace NUMINAMATH_CALUDE_area_of_four_squares_l35_3598

/-- The area of a shape composed of four identical squares with side length 3 cm -/
theorem area_of_four_squares : 
  ∀ (side_length : ℝ) (num_squares : ℕ),
    side_length = 3 →
    num_squares = 4 →
    (num_squares : ℝ) * (side_length^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_four_squares_l35_3598


namespace NUMINAMATH_CALUDE_bikini_fraction_correct_l35_3595

/-- The fraction of garments that are bikinis at Lindsey's Vacation Wear -/
def bikini_fraction : ℝ := 0.38

/-- The fraction of garments that are trunks at Lindsey's Vacation Wear -/
def trunk_fraction : ℝ := 0.25

/-- The fraction of garments that are either bikinis or trunks at Lindsey's Vacation Wear -/
def bikini_or_trunk_fraction : ℝ := 0.63

/-- Theorem stating that the fraction of garments that are bikinis is correct -/
theorem bikini_fraction_correct :
  bikini_fraction + trunk_fraction = bikini_or_trunk_fraction :=
by sorry

end NUMINAMATH_CALUDE_bikini_fraction_correct_l35_3595


namespace NUMINAMATH_CALUDE_power_of_four_l35_3570

theorem power_of_four (x : ℕ) 
  (h1 : 2 * x + 5 + 2 = 29) : x = 11 := by
  sorry

#check power_of_four

end NUMINAMATH_CALUDE_power_of_four_l35_3570


namespace NUMINAMATH_CALUDE_problem_statement_l35_3555

theorem problem_statement (m : ℝ) (h : |m| = m + 1) : (4 * m + 1)^2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l35_3555


namespace NUMINAMATH_CALUDE_five_arithmetic_operations_l35_3518

theorem five_arithmetic_operations :
  -- 1. 5555 = 7
  (5 + 5 + 5) / 5 = 3 ∧
  (5 + 5) / 5 + 5 = 7 ∧
  -- 2. 5555 = 55
  (5 + 5) * 5 + 5 = 55 ∧
  -- 3. 5,5,5,5 = 4
  (5 * 5 - 5) / 5 = 4 ∧
  -- 4. 5,5,5,5 = 26
  5 * 5 + (5 / 5) = 26 ∧
  -- 5. 5,5,5,5 = 120
  5 * 5 * 5 - 5 = 120 ∧
  -- 6. 5,5,5,5 = 5
  (5 - 5) * 5 + 5 = 5 ∧
  -- 7. 5555 = 30
  (5 / 5 + 5) * 5 = 30 ∧
  -- 8. 5,5,5,5 = 130
  5 * 5 * 5 + 5 = 130 ∧
  -- 9. 5555 = 6
  (5 * 5 + 5) / 5 = 6 ∧
  -- 10. 5555 = 50
  5 * 5 + 5 * 5 = 50 ∧
  -- 11. 5555 = 625
  5 * 5 * 5 * 5 = 625 := by
  sorry

#check five_arithmetic_operations

end NUMINAMATH_CALUDE_five_arithmetic_operations_l35_3518


namespace NUMINAMATH_CALUDE_alonzo_unsold_tomatoes_l35_3502

/-- Calculates the amount of unsold tomatoes given the total harvest and amounts sold to two buyers. -/
def unsold_tomatoes (total_harvest : ℝ) (sold_to_maxwell : ℝ) (sold_to_wilson : ℝ) : ℝ :=
  total_harvest - (sold_to_maxwell + sold_to_wilson)

/-- Proves that given the specific amounts in Mr. Alonzo's tomato sales, the unsold amount is 42 kg. -/
theorem alonzo_unsold_tomatoes :
  unsold_tomatoes 245.5 125.5 78 = 42 := by
  sorry

end NUMINAMATH_CALUDE_alonzo_unsold_tomatoes_l35_3502


namespace NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_between_sqrt_l35_3527

-- Define the square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Theorem 1
theorem sqrt_190_44 : sqrt 190.44 = 13.8 ∨ sqrt 190.44 = -13.8 := by sorry

-- Theorem 2
theorem sqrt_176_9_and_18769 :
  (13.3 < sqrt 176.9 ∧ sqrt 176.9 < 13.4) ∧ sqrt 18769 = 137 := by sorry

-- Theorem 3
theorem integer_between_sqrt :
  ∀ n : ℤ, (13.5 < sqrt (n : ℝ) ∧ sqrt (n : ℝ) < 13.6) → (n = 183 ∨ n = 184) := by sorry

end NUMINAMATH_CALUDE_sqrt_190_44_sqrt_176_9_and_18769_integer_between_sqrt_l35_3527


namespace NUMINAMATH_CALUDE_gerald_toy_cars_l35_3553

theorem gerald_toy_cars (initial_cars : ℕ) (donation_fraction : ℚ) (remaining_cars : ℕ) :
  initial_cars = 20 →
  donation_fraction = 1 / 4 →
  remaining_cars = initial_cars - (initial_cars * donation_fraction).floor →
  remaining_cars = 15 := by
  sorry

end NUMINAMATH_CALUDE_gerald_toy_cars_l35_3553


namespace NUMINAMATH_CALUDE_work_completion_time_l35_3558

theorem work_completion_time (x : ℝ) : 
  x > 0 → 
  (8 * (1 / x + 1 / 20) = 14 / 15) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l35_3558


namespace NUMINAMATH_CALUDE_shelf_filling_l35_3596

/-- Given a shelf that can be filled with books, this theorem relates the number of
    physics and chemistry books needed to fill it. -/
theorem shelf_filling (P C R B G : ℕ) : 
  (P > 0) → (C > 0) → (R > 0) → (B > 0) → (G > 0) →  -- Positive integers
  (P ≠ C) → (P ≠ R) → (P ≠ B) → (P ≠ G) →  -- Distinct values
  (C ≠ R) → (C ≠ B) → (C ≠ G) →
  (R ≠ B) → (R ≠ G) →
  (B ≠ G) →
  (∃ (x : ℚ), x > 0 ∧ P * x + 2 * C * x = G * x) →  -- Shelf filling condition
  (∃ (x : ℚ), x > 0 ∧ R * x + 2 * B * x = G * x) →  -- Alternative filling
  G = P + 2 * C :=
by sorry

end NUMINAMATH_CALUDE_shelf_filling_l35_3596


namespace NUMINAMATH_CALUDE_digit_1983_is_7_l35_3564

/-- Represents the decimal number x as described in the problem -/
def x : ℝ :=
  sorry

/-- Returns the nth digit after the decimal point in x -/
def nthDigit (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the 1983rd digit of x is 7 -/
theorem digit_1983_is_7 : nthDigit 1983 = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_1983_is_7_l35_3564


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l35_3523

theorem xyz_sum_sqrt (x y z : ℝ) 
  (eq1 : y + z = 15)
  (eq2 : z + x = 17)
  (eq3 : x + y = 16) :
  Real.sqrt (x * y * z * (x + y + z)) = 72 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l35_3523


namespace NUMINAMATH_CALUDE_bernoullis_inequality_l35_3560

theorem bernoullis_inequality (x : ℝ) (n : ℕ) (h1 : x ≥ -1) (h2 : n ≥ 1) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_bernoullis_inequality_l35_3560


namespace NUMINAMATH_CALUDE_true_discount_calculation_l35_3503

/-- Given a bill with face value 540 and banker's discount 108, prove the true discount is 90 -/
theorem true_discount_calculation (face_value banker_discount : ℚ) 
  (h1 : face_value = 540)
  (h2 : banker_discount = 108)
  (h3 : ∀ (td : ℚ), banker_discount = td + (td * banker_discount / face_value)) :
  ∃ (true_discount : ℚ), true_discount = 90 ∧ 
    banker_discount = true_discount + (true_discount * banker_discount / face_value) := by
  sorry

#check true_discount_calculation

end NUMINAMATH_CALUDE_true_discount_calculation_l35_3503


namespace NUMINAMATH_CALUDE_total_trees_formula_l35_3586

/-- The total number of trees planted by three teams under specific conditions -/
def total_trees (a : ℕ) : ℕ :=
  let team1 := a
  let team2 := 2 * a + 8
  let team3 := (team2 / 2) - 6
  team1 + team2 + team3

/-- Theorem stating the total number of trees planted by the three teams -/
theorem total_trees_formula (a : ℕ) : total_trees a = 4 * a + 6 := by
  sorry

#eval total_trees 100  -- Should output 406

end NUMINAMATH_CALUDE_total_trees_formula_l35_3586


namespace NUMINAMATH_CALUDE_son_age_theorem_l35_3593

/-- Represents the ages of three generations in a family -/
structure FamilyAges where
  grandson_days : ℕ
  son_months : ℕ
  grandfather_years : ℕ

/-- Calculates the son's age in weeks given the family ages -/
def son_age_weeks (ages : FamilyAges) : ℕ :=
  ages.son_months * 4 -- Approximate weeks in a month

/-- The main theorem stating the son's age in weeks -/
theorem son_age_theorem (ages : FamilyAges) : 
  ages.grandson_days = ages.son_months ∧ 
  ages.grandson_days / 30 = ages.grandfather_years ∧ 
  ages.grandson_days / 360 + ages.son_months / 12 + ages.grandfather_years = 140 ∧ 
  ages.grandfather_years = 84 →
  son_age_weeks ages = 2548 := by
  sorry

#eval son_age_weeks { grandson_days := 2520, son_months := 588, grandfather_years := 84 }

end NUMINAMATH_CALUDE_son_age_theorem_l35_3593


namespace NUMINAMATH_CALUDE_transversal_exists_l35_3567

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are in general position (not parallel or coincident) -/
def general_position (l1 l2 : Line3D) : Prop := 
  ¬ parallel l1 l2 ∧ ¬ (l1 = l2)

/-- Theorem: Existence of a transversal parallel to a given line -/
theorem transversal_exists (l1 l2 lp : Line3D) 
  (h : general_position l1 l2) : 
  ∃ lt : Line3D, intersect lt l1 ∧ intersect lt l2 ∧ parallel lt lp := by
  sorry

end NUMINAMATH_CALUDE_transversal_exists_l35_3567


namespace NUMINAMATH_CALUDE_vector_operations_l35_3543

/-- Given vectors a and b, prove their sum and dot product -/
theorem vector_operations (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, 2, 2)) (hb : b = (6, -3, 2)) : 
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2) = (7, -1, 4) ∧ 
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l35_3543


namespace NUMINAMATH_CALUDE_max_digits_product_4digit_3digit_l35_3568

theorem max_digits_product_4digit_3digit : 
  ∀ (a b : ℕ), 
    1000 ≤ a ∧ a < 10000 → 
    100 ≤ b ∧ b < 1000 → 
    a * b < 10000000 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_4digit_3digit_l35_3568


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l35_3522

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ Real.sqrt 637 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = Real.sqrt 637 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l35_3522


namespace NUMINAMATH_CALUDE_min_value_of_a_min_value_is_one_l35_3525

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ a) → a ≥ 1 := by
  sorry

theorem min_value_is_one :
  ∃ a : ℝ, (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ a) ∧ 
    (∀ b : ℝ, (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ b) → a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_min_value_is_one_l35_3525


namespace NUMINAMATH_CALUDE_dessert_preference_theorem_l35_3582

/-- Represents the dessert preferences of a group of students -/
structure DessertPreferences where
  total : ℕ
  apple : ℕ
  chocolate : ℕ
  carrot : ℕ
  none : ℕ
  apple_chocolate_not_carrot : ℕ

/-- The theorem stating the number of students who like both apple pie and chocolate cake but not carrot cake -/
theorem dessert_preference_theorem (prefs : DessertPreferences) : 
  prefs.total = 50 ∧ 
  prefs.apple = 23 ∧ 
  prefs.chocolate = 20 ∧ 
  prefs.carrot = 10 ∧ 
  prefs.none = 15 → 
  prefs.apple_chocolate_not_carrot = 7 := by
  sorry

end NUMINAMATH_CALUDE_dessert_preference_theorem_l35_3582


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l35_3554

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, a ≥ 1 ∧ b ≥ 1 → a + b ≥ 2) ∧ 
  (∃ a b : ℝ, a + b ≥ 2 ∧ ¬(a ≥ 1 ∧ b ≥ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l35_3554


namespace NUMINAMATH_CALUDE_tan_negative_thirteen_fourths_pi_l35_3508

theorem tan_negative_thirteen_fourths_pi : Real.tan (-13/4 * π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_thirteen_fourths_pi_l35_3508


namespace NUMINAMATH_CALUDE_three_digit_divisibility_by_37_l35_3559

theorem three_digit_divisibility_by_37 (A B C : ℕ) (h_three_digit : 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000) (h_divisible : (100 * A + 10 * B + C) % 37 = 0) :
  ∃ M : ℕ, M = 100 * B + 10 * C + A ∧ 100 ≤ M ∧ M < 1000 ∧ M % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_by_37_l35_3559


namespace NUMINAMATH_CALUDE_sine_cosine_acute_less_than_one_l35_3515

-- Define an acute angle
def is_acute_angle (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define sine and cosine for an acute angle in a right-angled triangle
def sine_acute (α : Real) (h : is_acute_angle α) : Real := sorry
def cosine_acute (α : Real) (h : is_acute_angle α) : Real := sorry

-- Theorem statement
theorem sine_cosine_acute_less_than_one (α : Real) (h : is_acute_angle α) :
  sine_acute α h < 1 ∧ cosine_acute α h < 1 := by sorry

end NUMINAMATH_CALUDE_sine_cosine_acute_less_than_one_l35_3515


namespace NUMINAMATH_CALUDE_combined_average_marks_specific_average_marks_l35_3532

/-- Given two classes of students with their respective sizes and average marks,
    calculate the combined average mark of all students. -/
theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℝ) :
  n1 > 0 → n2 > 0 →
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (n1 * avg1 + n2 * avg2) / (n1 + n2) := by
  sorry

/-- The average mark of all students given the specific class sizes and averages. -/
theorem specific_average_marks :
  let n1 := 24
  let n2 := 50
  let avg1 := 40
  let avg2 := 60
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  total_marks / total_students = (24 * 40 + 50 * 60) / (24 + 50) := by
  sorry

end NUMINAMATH_CALUDE_combined_average_marks_specific_average_marks_l35_3532


namespace NUMINAMATH_CALUDE_pet_store_combinations_l35_3587

def num_puppies : ℕ := 15
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 8
def num_people : ℕ := 3

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * Nat.factorial num_people = 7200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l35_3587


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l35_3584

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l35_3584


namespace NUMINAMATH_CALUDE_parabola_equation_l35_3569

/-- The equation of a parabola given its parametric form -/
theorem parabola_equation (t : ℝ) :
  let x : ℝ := 3 * t + 6
  let y : ℝ := 5 * t^2 - 7
  y = (5/9) * x^2 - (20/3) * x + 13 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l35_3569


namespace NUMINAMATH_CALUDE_max_value_quadratic_l35_3562

theorem max_value_quadratic (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, x^2 - a*x - a ≤ 1) ∧ 
  (∃ x ∈ Set.Icc 0 2, x^2 - a*x - a = 1) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l35_3562


namespace NUMINAMATH_CALUDE_even_function_m_value_l35_3597

-- Define a function f
def f (m : ℝ) (x : ℝ) : ℝ := (x - 2) * (x - m)

-- State the theorem
theorem even_function_m_value :
  (∀ x : ℝ, f m x = f m (-x)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_m_value_l35_3597


namespace NUMINAMATH_CALUDE_time_difference_l35_3590

theorem time_difference (brian_time todd_time : ℕ) 
  (h1 : brian_time = 96) 
  (h2 : todd_time = 88) : 
  brian_time - todd_time = 8 := by
sorry

end NUMINAMATH_CALUDE_time_difference_l35_3590


namespace NUMINAMATH_CALUDE_adjusted_equilateral_triangle_l35_3591

/-- Given a triangle XYZ that was originally equilateral, prove that if angle X is decreased by 5 degrees, 
    then angles Y and Z will each measure 62.5 degrees. -/
theorem adjusted_equilateral_triangle (X Y Z : ℝ) : 
  X + Y + Z = 180 →  -- Sum of angles in a triangle is 180°
  X = 55 →           -- Angle X after decrease
  Y = Z →            -- Angles Y and Z remain equal
  Y = 62.5 ∧ Z = 62.5 := by
sorry

end NUMINAMATH_CALUDE_adjusted_equilateral_triangle_l35_3591


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l35_3592

def initial_trees : ℕ := 237
def planting_factor : ℕ := 5

theorem oak_trees_after_planting :
  initial_trees + planting_factor * initial_trees = 1422 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l35_3592


namespace NUMINAMATH_CALUDE_hannah_age_proof_l35_3571

def siblings_ages : List ℕ := [103, 124, 146, 81, 114, 195, 183]

def average_age (ages : List ℕ) : ℚ :=
  (ages.sum : ℚ) / ages.length

theorem hannah_age_proof :
  let avg_sibling_age := average_age siblings_ages
  let hannah_age := 3.2 * avg_sibling_age
  hannah_age = 432 := by sorry

end NUMINAMATH_CALUDE_hannah_age_proof_l35_3571


namespace NUMINAMATH_CALUDE_inequality_solution_set_l35_3589

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 - x - 3 > 0) ↔ (x > 3/2 ∨ x < -1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l35_3589


namespace NUMINAMATH_CALUDE_cube_face_sum_l35_3505

/-- Represents the numbers on the faces of a cube -/
structure CubeNumbers where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+
  h : ℕ+
  vertex_sum_eq : (a + e) * (b + f) * (c + g) * h = 2002

/-- The sum of numbers on the faces of the cube is 39 -/
theorem cube_face_sum (cube : CubeNumbers) : 
  cube.a + cube.b + cube.c + cube.e + cube.f + cube.g + cube.h = 39 := by
  sorry


end NUMINAMATH_CALUDE_cube_face_sum_l35_3505


namespace NUMINAMATH_CALUDE_relationship_holds_l35_3537

/-- The function describing the relationship between x and y -/
def f (x : ℕ) : ℕ := x^2 - 3*x + 2

/-- The set of x values given in the table -/
def X : Set ℕ := {2, 3, 4, 5, 6}

/-- The proposition that the function f correctly describes the relationship for all x in X -/
theorem relationship_holds (x : ℕ) (h : x ∈ X) : 
  (x = 2 → f x = 0) ∧ 
  (x = 3 → f x = 2) ∧ 
  (x = 4 → f x = 6) ∧ 
  (x = 5 → f x = 12) ∧ 
  (x = 6 → f x = 20) :=
by sorry

end NUMINAMATH_CALUDE_relationship_holds_l35_3537


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l35_3552

theorem square_perimeters_sum (x : ℝ) : 
  let area1 := x^2 + 8*x + 16
  let area2 := 4*x^2 - 12*x + 9
  let area3 := 9*x^2 - 6*x + 1
  let perimeter1 := 4 * Real.sqrt area1
  let perimeter2 := 4 * Real.sqrt area2
  let perimeter3 := 4 * Real.sqrt area3
  perimeter1 + perimeter2 + perimeter3 = 48 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l35_3552


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_six_l35_3580

theorem sqrt_expression_equals_six :
  (Real.sqrt 27 - 3 * Real.sqrt (1/3)) / (1 / Real.sqrt 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_six_l35_3580


namespace NUMINAMATH_CALUDE_product_of_integers_l35_3547

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 
  2 * (a * b) + 14 * a = 5 * b + 290 → 
  a * b = 104 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l35_3547


namespace NUMINAMATH_CALUDE_larger_number_problem_l35_3542

theorem larger_number_problem (x y : ℝ) 
  (sum_eq : x + y = 55) 
  (diff_eq : x - y = 15) : 
  x = 35 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l35_3542


namespace NUMINAMATH_CALUDE_equation_solutions_l35_3599

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 + 2*x*(x - 3)
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l35_3599


namespace NUMINAMATH_CALUDE_math_team_combinations_l35_3574

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the math club. -/
def num_girls : ℕ := 4

/-- The number of boys in the math club. -/
def num_boys : ℕ := 6

/-- The number of girls required in the team. -/
def girls_in_team : ℕ := 3

/-- The number of boys required in the team. -/
def boys_in_team : ℕ := 4

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l35_3574


namespace NUMINAMATH_CALUDE_B_largest_at_45_l35_3524

/-- B_k is defined as the binomial coefficient (500 choose k) multiplied by 0.1^k -/
def B (k : ℕ) : ℝ := (Nat.choose 500 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem B_largest_at_45 : ∀ k : ℕ, k ≤ 500 → B 45 ≥ B k := by
  sorry

end NUMINAMATH_CALUDE_B_largest_at_45_l35_3524


namespace NUMINAMATH_CALUDE_wool_production_l35_3579

variables (x y z w v : ℝ)
variable (breed_A_production : ℝ → ℝ → ℝ → ℝ)
variable (breed_B_production : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ)

-- Breed A production rate
axiom breed_A_rate : breed_A_production x y z = y / (x * z)

-- Breed B produces twice as much as Breed A
axiom breed_B_rate : breed_B_production x y z w v = 2 * breed_A_production x y z * w * v

-- Theorem to prove
theorem wool_production : breed_B_production x y z w v = (2 * y * w * v) / (x * z) := by
  sorry

end NUMINAMATH_CALUDE_wool_production_l35_3579


namespace NUMINAMATH_CALUDE_square_difference_l35_3528

theorem square_difference (x y : ℝ) 
  (h1 : (x + y) / 2 = 5)
  (h2 : (x - y) / 2 = 2) : 
  x^2 - y^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l35_3528
