import Mathlib

namespace NUMINAMATH_CALUDE_total_broadcasting_period_l3929_392906

/-- Given a music station that played commercials for a certain duration and maintained a specific ratio of music to commercials, this theorem proves the total broadcasting period. -/
theorem total_broadcasting_period 
  (commercial_duration : ℕ) 
  (music_ratio : ℕ) 
  (commercial_ratio : ℕ) 
  (h1 : commercial_duration = 40)
  (h2 : music_ratio = 9)
  (h3 : commercial_ratio = 5) :
  commercial_duration + (commercial_duration * music_ratio) / commercial_ratio = 112 :=
by sorry

end NUMINAMATH_CALUDE_total_broadcasting_period_l3929_392906


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_on_y_axis_l3929_392979

-- Define the circle
def circle_center : ℝ × ℝ := (-3, 5)
def circle_radius : ℝ := 15

-- Define a function to represent the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem sum_of_y_coordinates_on_y_axis :
  ∃ y₁ y₂ : ℝ, 
    circle_equation 0 y₁ ∧ 
    circle_equation 0 y₂ ∧ 
    y₁ ≠ y₂ ∧ 
    y₁ + y₂ = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_on_y_axis_l3929_392979


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l3929_392915

theorem polynomial_root_implies_coefficients : 
  ∀ (a b : ℝ), 
  (Complex.I : ℂ) ^ 4 + a * (Complex.I : ℂ) ^ 3 - (Complex.I : ℂ) ^ 2 + b * (Complex.I : ℂ) - 6 = 0 →
  (2 - Complex.I : ℂ) ^ 4 + a * (2 - Complex.I : ℂ) ^ 3 - (2 - Complex.I : ℂ) ^ 2 + b * (2 - Complex.I : ℂ) - 6 = 0 →
  a = -4 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l3929_392915


namespace NUMINAMATH_CALUDE_sixteenth_row_seats_l3929_392993

/-- 
Represents the number of seats in a row of an auditorium where:
- The first row has 5 seats
- Each subsequent row increases by 2 seats
-/
def seats_in_row (n : ℕ) : ℕ := 2 * n + 3

/-- 
Theorem: The 16th row of the auditorium has 35 seats
-/
theorem sixteenth_row_seats : seats_in_row 16 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sixteenth_row_seats_l3929_392993


namespace NUMINAMATH_CALUDE_lovers_watches_prime_sum_squares_l3929_392938

theorem lovers_watches_prime_sum_squares :
  ∃ (x y : Fin 12 → ℕ) (m : Fin 12 → ℕ),
    (∀ i : Fin 12, Nat.Prime (x i)) ∧
    (∀ i : Fin 12, Nat.Prime (y i)) ∧
    (∀ i j : Fin 12, i ≠ j → x i ≠ x j) ∧
    (∀ i j : Fin 12, i ≠ j → y i ≠ y j) ∧
    (∀ i : Fin 12, x i ≠ y i) ∧
    (∀ k : Fin 12, x k + x (k.succ) = y k + y (k.succ)) ∧
    (∀ k : Fin 12, ∃ (m_k : ℕ), x k + x (k.succ) = m_k ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_lovers_watches_prime_sum_squares_l3929_392938


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3929_392931

theorem square_floor_tiles (black_tiles : ℕ) (total_tiles : ℕ) : 
  black_tiles = 101 → 
  (∃ (side_length : ℕ), 
    side_length * side_length = total_tiles ∧ 
    2 * side_length - 1 = black_tiles) → 
  total_tiles = 2601 :=
by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3929_392931


namespace NUMINAMATH_CALUDE_puzzle_piece_ratio_l3929_392912

theorem puzzle_piece_ratio (total pieces : ℕ) (border : ℕ) (trevor : ℕ) (missing : ℕ) :
  total = 500 →
  border = 75 →
  trevor = 105 →
  missing = 5 →
  ∃ (joe : ℕ), joe = total - border - trevor - missing ∧ joe = 3 * trevor :=
by sorry

end NUMINAMATH_CALUDE_puzzle_piece_ratio_l3929_392912


namespace NUMINAMATH_CALUDE_stratified_sample_probability_l3929_392972

theorem stratified_sample_probability (grade10 grade11 grade12 : ℕ) (sample_size : ℕ) :
  grade10 = 300 →
  grade11 = 300 →
  grade12 = 400 →
  sample_size = 40 →
  (grade12 : ℚ) / (grade10 + grade11 + grade12 : ℚ) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_probability_l3929_392972


namespace NUMINAMATH_CALUDE_isosceles_triangle_m_values_l3929_392916

/-- An isosceles triangle with side lengths satisfying a quadratic equation -/
structure IsoscelesTriangle where
  -- The length of side BC
  bc : ℝ
  -- The parameter m in the quadratic equation
  m : ℝ
  -- The roots of the quadratic equation x^2 - 10x + m = 0 represent the lengths of AB and AC
  root1 : ℝ
  root2 : ℝ
  -- Ensure that root1 and root2 are indeed roots of the equation
  eq1 : root1^2 - 10*root1 + m = 0
  eq2 : root2^2 - 10*root2 + m = 0
  -- Ensure that the triangle is isosceles (two sides are equal)
  isosceles : root1 = root2 ∨ (root1 = bc ∧ root2 = 10 - bc) ∨ (root2 = bc ∧ root1 = 10 - bc)
  -- Given condition that BC = 8
  bc_eq_8 : bc = 8

/-- The theorem stating that m is either 16 or 25 -/
theorem isosceles_triangle_m_values (t : IsoscelesTriangle) : t.m = 16 ∨ t.m = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_m_values_l3929_392916


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l3929_392985

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (intersectPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_same_line
  (m n : Line) (α β : Plane)
  (h1 : ¬ intersect m n)
  (h2 : ¬ intersectPlanes α β)
  (h3 : perpendicular m α)
  (h4 : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l3929_392985


namespace NUMINAMATH_CALUDE_eventually_point_difference_exceeds_50_l3929_392980

/-- Represents a player in the tournament -/
structure Player where
  id : Nat
  points : Int

/-- Represents the state of the tournament on a given day -/
structure TournamentDay where
  players : Vector Player 200
  day : Nat

/-- Function to sort players by their points -/
def sortPlayers (players : Vector Player 200) : Vector Player 200 := sorry

/-- Function to play matches for a day and update points -/
def playMatches (t : TournamentDay) : TournamentDay := sorry

/-- Predicate to check if the point difference exceeds 50 -/
def pointDifferenceExceeds50 (t : TournamentDay) : Prop :=
  ∃ i j, i < 200 ∧ j < 200 ∧ (t.players.get i).points - (t.players.get j).points > 50

/-- The main theorem to be proved -/
theorem eventually_point_difference_exceeds_50 :
  ∃ n : Nat, ∃ t : TournamentDay, t.day = n ∧ pointDifferenceExceeds50 t :=
sorry

end NUMINAMATH_CALUDE_eventually_point_difference_exceeds_50_l3929_392980


namespace NUMINAMATH_CALUDE_equation_solution_l3929_392930

theorem equation_solution :
  ∃ x : ℝ, (1 / x + (2 / x) / (4 / x) = 3 / 4) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3929_392930


namespace NUMINAMATH_CALUDE_triangle_side_length_l3929_392920

/-- Given a triangle ABC with the following properties:
  - f(x) = 2sin(2x + π/6) + 1
  - f(A) = 2
  - b = 1
  - Area of triangle ABC is √3/2
  Prove that a = √3 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6) + 1) →
  f A = 2 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a = Real.sqrt 3 := by 
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3929_392920


namespace NUMINAMATH_CALUDE_grass_weeds_count_l3929_392970

/-- Represents the number of weeds in different areas of the garden -/
structure GardenWeeds where
  flower_bed : ℕ
  vegetable_patch : ℕ
  grass : ℕ

/-- Represents Lucille's earnings and expenses -/
structure LucilleFinances where
  cents_per_weed : ℕ
  soda_cost : ℕ
  remaining_cents : ℕ

def calculate_grass_weeds (garden : GardenWeeds) (finances : LucilleFinances) : ℕ :=
  garden.grass

theorem grass_weeds_count 
  (garden : GardenWeeds) 
  (finances : LucilleFinances) 
  (h1 : garden.flower_bed = 11)
  (h2 : garden.vegetable_patch = 14)
  (h3 : finances.cents_per_weed = 6)
  (h4 : finances.soda_cost = 99)
  (h5 : finances.remaining_cents = 147)
  : calculate_grass_weeds garden finances = 32 := by
  sorry

#eval calculate_grass_weeds 
  { flower_bed := 11, vegetable_patch := 14, grass := 32 } 
  { cents_per_weed := 6, soda_cost := 99, remaining_cents := 147 }

end NUMINAMATH_CALUDE_grass_weeds_count_l3929_392970


namespace NUMINAMATH_CALUDE_sum_cis_angle_sequence_l3929_392978

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ := Complex.exp (θ * Complex.I)

-- Define the arithmetic sequence of angles
def angleSequence : List ℝ := List.range 12 |>.map (λ n => 70 + 8 * n)

-- State the theorem
theorem sum_cis_angle_sequence (r : ℝ) (θ : ℝ) 
  (h_r : r > 0) (h_θ : 0 ≤ θ ∧ θ < 360) :
  (angleSequence.map (λ α => cis α)).sum = r * cis θ → θ = 114 := by
  sorry

end NUMINAMATH_CALUDE_sum_cis_angle_sequence_l3929_392978


namespace NUMINAMATH_CALUDE_share_ratio_l3929_392998

theorem share_ratio (total amount : ℕ) (a_share b_share c_share : ℕ) 
  (h1 : total = 595)
  (h2 : a_share = 420)
  (h3 : b_share = 105)
  (h4 : c_share = 70)
  (h5 : total = a_share + b_share + c_share)
  (h6 : b_share = c_share / 4) :
  a_share / b_share = 4 := by
sorry

end NUMINAMATH_CALUDE_share_ratio_l3929_392998


namespace NUMINAMATH_CALUDE_medium_lights_count_l3929_392988

/-- Represents the number of medium ceiling lights -/
def M : ℕ := sorry

/-- The number of small ceiling lights -/
def small_lights : ℕ := M + 10

/-- The number of large ceiling lights -/
def large_lights : ℕ := 2 * M

/-- The total number of bulbs needed -/
def total_bulbs : ℕ := 118

/-- Theorem stating that the number of medium ceiling lights is 12 -/
theorem medium_lights_count : M = 12 := by
  have bulb_equation : small_lights * 1 + M * 2 + large_lights * 3 = total_bulbs := by sorry
  sorry

end NUMINAMATH_CALUDE_medium_lights_count_l3929_392988


namespace NUMINAMATH_CALUDE_f_monotonicity_and_roots_l3929_392961

/-- The function f(x) = x³ - 2x² + x + t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + x + t

theorem f_monotonicity_and_roots (t : ℝ) :
  (∀ x y, x < y → ((x < 1/3 ∧ y < 1/3) ∨ (x > 1 ∧ y > 1)) → f t x < f t y) ∧ 
  (∀ x y, 1/3 < x → x < y → y < 1 → f t x > f t y) ∧
  (∃ x y z, x < y ∧ y < z ∧ f t x = 0 ∧ f t y = 0 ∧ f t z = 0 → -4/27 < t ∧ t < 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_roots_l3929_392961


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_equals_sqrt_six_l3929_392910

theorem sqrt_two_times_sqrt_three_equals_sqrt_six :
  Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_three_equals_sqrt_six_l3929_392910


namespace NUMINAMATH_CALUDE_hyperbola_tangent_equation_l3929_392958

-- Define the ellipse
def is_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the hyperbola
def is_hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the tangent line for the ellipse
def is_ellipse_tangent (x y x₀ y₀ a b : ℝ) : Prop := (x₀ * x / a^2) + (y₀ * y / b^2) = 1

-- Define the tangent line for the hyperbola
def is_hyperbola_tangent (x y x₀ y₀ a b : ℝ) : Prop := (x₀ * x / a^2) - (y₀ * y / b^2) = 1

-- State the theorem
theorem hyperbola_tangent_equation (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : is_hyperbola x₀ y₀ a b) :
  is_hyperbola_tangent x y x₀ y₀ a b :=
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_equation_l3929_392958


namespace NUMINAMATH_CALUDE_train_length_calculation_l3929_392955

/-- The length of the train in meters -/
def train_length : ℝ := 1200

/-- The time (in seconds) it takes for the train to cross a tree -/
def tree_crossing_time : ℝ := 120

/-- The time (in seconds) it takes for the train to pass a platform -/
def platform_crossing_time : ℝ := 160

/-- The length of the platform in meters -/
def platform_length : ℝ := 400

theorem train_length_calculation :
  train_length = 1200 ∧
  (train_length / tree_crossing_time = (train_length + platform_length) / platform_crossing_time) :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3929_392955


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3929_392974

/-- Represents a quadratic equation ax^2 + (a+2)x + 9a = 0 -/
def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 1 ∧ 1 < x₂ ∧
    quadratic_equation a x₁ = 0 ∧ quadratic_equation a x₂ = 0) →
  -2/11 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3929_392974


namespace NUMINAMATH_CALUDE_small_font_words_per_page_l3929_392997

/-- Calculates the number of words per page in the small font given the article constraints -/
theorem small_font_words_per_page 
  (total_words : ℕ) 
  (total_pages : ℕ) 
  (large_font_pages : ℕ) 
  (large_font_words_per_page : ℕ) 
  (h1 : total_words = 48000)
  (h2 : total_pages = 21)
  (h3 : large_font_pages = 4)
  (h4 : large_font_words_per_page = 1800) :
  (total_words - large_font_pages * large_font_words_per_page) / (total_pages - large_font_pages) = 2400 :=
by
  sorry

#check small_font_words_per_page

end NUMINAMATH_CALUDE_small_font_words_per_page_l3929_392997


namespace NUMINAMATH_CALUDE_problem_statement_l3929_392945

theorem problem_statement (x y : ℝ) (h1 : x = 2 * y) (h2 : y ≠ 0) :
  (x + 2 * y) - (2 * x + y) = -y := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3929_392945


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3929_392948

/-- Proves that for a parabola y = ax^2 + bx + c with vertex at (q,q) and y-intercept at (0, -2q), 
    where q ≠ 0, the coefficient b equals 6/q. -/
theorem parabola_coefficient (a b c q : ℝ) (h1 : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (q, q) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) →
  -2 * q = c →
  b = 6 / q :=
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3929_392948


namespace NUMINAMATH_CALUDE_only_set_A_forms_triangle_l3929_392937

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def set_A : List ℝ := [5, 6, 10]
def set_B : List ℝ := [5, 2, 9]
def set_C : List ℝ := [5, 7, 12]
def set_D : List ℝ := [3, 4, 8]

-- Theorem statement
theorem only_set_A_forms_triangle :
  (can_form_triangle 5 6 10) ∧
  ¬(can_form_triangle 5 2 9) ∧
  ¬(can_form_triangle 5 7 12) ∧
  ¬(can_form_triangle 3 4 8) :=
sorry

end NUMINAMATH_CALUDE_only_set_A_forms_triangle_l3929_392937


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3929_392934

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3929_392934


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3929_392929

/-- Definition of the binary operation ◇ -/
noncomputable def diamond (k : ℝ) (a b : ℝ) : ℝ :=
  k / b

/-- Theorem stating the solution to the equation -/
theorem diamond_equation_solution (k : ℝ) (h1 : k = 2) :
  ∃ x : ℝ, diamond k 2023 (diamond k 7 x) = 150 ∧ x = 150 / 2023 := by
  sorry

/-- Properties of the binary operation ◇ -/
axiom diamond_assoc (k a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond k a (diamond k b c) = k * (diamond k a b) * c

axiom diamond_self (k a : ℝ) (ha : a ≠ 0) :
  diamond k a a = k

end NUMINAMATH_CALUDE_diamond_equation_solution_l3929_392929


namespace NUMINAMATH_CALUDE_ladder_length_l3929_392992

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) :
  angle = Real.pi / 3 →  -- 60 degrees in radians
  adjacent = 9.493063650744542 →
  hypotenuse = adjacent / Real.cos angle →
  hypotenuse = 18.986127301489084 := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l3929_392992


namespace NUMINAMATH_CALUDE_survey_result_l3929_392946

/-- Represents the survey results of teachers' health conditions -/
structure TeacherSurvey where
  total : ℕ
  highBP : ℕ
  heartTrouble : ℕ
  bothConditions : ℕ

/-- Calculates the percentage of teachers with neither high blood pressure nor heart trouble -/
def percentWithNeitherCondition (survey : TeacherSurvey) : ℚ :=
  let withoutConditions := survey.total - (survey.highBP + survey.heartTrouble - survey.bothConditions)
  (withoutConditions : ℚ) / survey.total * 100

/-- The main theorem stating the result of the survey -/
theorem survey_result : 
  let survey : TeacherSurvey := { 
    total := 150,
    highBP := 90,
    heartTrouble := 50,
    bothConditions := 30
  }
  percentWithNeitherCondition survey = 26.67 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l3929_392946


namespace NUMINAMATH_CALUDE_sine_function_properties_l3929_392983

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_properties :
  ∃ (A ω φ : ℝ),
    (f A ω φ 0 = 0) ∧
    (f A ω φ (π/2) = 2) ∧
    (f A ω φ π = 0) ∧
    (f A ω φ (3*π/2) = -2) ∧
    (f A ω φ (2*π) = 0) ∧
    (5*π/3 + π/3 = 2*π) →
    (A = 2) ∧
    (ω = 1/2) ∧
    (φ = 2*π/3) ∧
    (∀ x : ℝ, f A ω φ (x - π/3) = f A ω φ (-x - π/3)) :=
by sorry

end NUMINAMATH_CALUDE_sine_function_properties_l3929_392983


namespace NUMINAMATH_CALUDE_trapezoid_sides_l3929_392911

-- Define the right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  is_345 : a = 3 ∧ b = 4 ∧ c = 5

-- Define the perpendicular line
def perpendicular_line (t : RightTriangle) (d : ℝ) : Prop :=
  d = 1 ∨ d = t.c - 1

-- Define the trapezoid formed by the perpendicular line
structure Trapezoid where
  s1 : ℝ
  s2 : ℝ
  s3 : ℝ
  s4 : ℝ

-- Theorem statement
theorem trapezoid_sides (t : RightTriangle) (d : ℝ) (trap : Trapezoid) 
  (h1 : perpendicular_line t d) :
  (trap.s1 = trap.s4 ∧ trap.s2 = trap.s3) ∧
  ((trap.s1 = 3 ∧ trap.s2 = 3/2) ∨ (trap.s1 = 4 ∧ trap.s2 = 4/3)) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l3929_392911


namespace NUMINAMATH_CALUDE_store_coloring_books_l3929_392939

/-- The number of coloring books sold during the sale -/
def books_sold : ℕ := 39

/-- The number of shelves used after the sale -/
def shelves : ℕ := 9

/-- The number of books on each shelf after the sale -/
def books_per_shelf : ℕ := 9

/-- The initial number of coloring books in stock -/
def initial_stock : ℕ := books_sold + shelves * books_per_shelf

theorem store_coloring_books : initial_stock = 120 := by
  sorry

end NUMINAMATH_CALUDE_store_coloring_books_l3929_392939


namespace NUMINAMATH_CALUDE_quadratic_properties_l3929_392957

/-- A quadratic equation with roots 1 and -1 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  root_one : a + b + c = 0
  root_neg_one : a - b + c = 0

theorem quadratic_properties (eq : QuadraticEquation) :
  eq.a + eq.b + eq.c = 0 ∧ eq.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3929_392957


namespace NUMINAMATH_CALUDE_sum_of_products_l3929_392901

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = 12)
  (h2 : y^2 + y*z + z^2 = 25)
  (h3 : z^2 + x*z + x^2 = 37) :
  x*y + y*z + x*z = 20 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_products_l3929_392901


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3929_392902

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  6 * x + 1 / (x^2) ≥ 7 * (6^(1/3)) := by
  sorry

theorem equality_condition : 
  6 * ((1/6)^(1/3)) + 1 / (((1/6)^(1/3))^2) = 7 * (6^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3929_392902


namespace NUMINAMATH_CALUDE_symmetry_condition_1_symmetry_condition_2_symmetry_condition_3_l3929_392969

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Theorem 1
theorem symmetry_condition_1 (h : ∀ x : ℝ, f (x + 2) = -f (-x)) :
  ∀ x : ℝ, f (x + 1) = -f (1 - x) := by sorry

-- Theorem 2
theorem symmetry_condition_2 :
  ∀ x : ℝ, f x = -f (2 - x) → 
  ∀ y : ℝ, f (1 + y) = -f (1 - y) := by sorry

-- Theorem 3
theorem symmetry_condition_3 :
  ∀ x : ℝ, f (-1 + (2 - x)) - f (1 - (2 - x)) = 
           f (-1 + x) - f (1 - x) := by sorry

end NUMINAMATH_CALUDE_symmetry_condition_1_symmetry_condition_2_symmetry_condition_3_l3929_392969


namespace NUMINAMATH_CALUDE_sandwich_contest_difference_l3929_392925

theorem sandwich_contest_difference : (5 : ℚ) / 6 - (2 : ℚ) / 3 = (1 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_sandwich_contest_difference_l3929_392925


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3929_392936

-- Define the polynomial
def f (x : ℝ) := 40 * x^3 - 70 * x^2 + 32 * x - 2

-- State the theorem
theorem root_sum_reciprocal (a b c : ℝ) :
  f a = 0 → f b = 0 → f c = 0 →  -- a, b, c are roots of f
  a ≠ b → b ≠ c → a ≠ c →        -- a, b, c are distinct
  0 < a → a < 1 →                -- 0 < a < 1
  0 < b → b < 1 →                -- 0 < b < 1
  0 < c → c < 1 →                -- 0 < c < 1
  1/(1-a) + 1/(1-b) + 1/(1-c) = 11/20 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3929_392936


namespace NUMINAMATH_CALUDE_product_equality_l3929_392995

theorem product_equality : 1500 * 451 * 0.0451 * 25 = 7627537500 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3929_392995


namespace NUMINAMATH_CALUDE_nina_widget_purchase_l3929_392949

/-- Calculates the number of widgets Nina can purchase given her budget and widget price information. -/
def widgets_nina_can_buy (budget : ℕ) (reduced_price_widgets : ℕ) (price_reduction : ℕ) : ℕ :=
  let original_price := (budget + reduced_price_widgets * price_reduction) / reduced_price_widgets
  budget / original_price

/-- Proves that Nina can buy 6 widgets given the problem conditions. -/
theorem nina_widget_purchase :
  widgets_nina_can_buy 48 8 2 = 6 := by
  sorry

#eval widgets_nina_can_buy 48 8 2

end NUMINAMATH_CALUDE_nina_widget_purchase_l3929_392949


namespace NUMINAMATH_CALUDE_smallest_delicious_integer_l3929_392989

/-- An integer is delicious if there exist several consecutive integers, starting from it, that add up to 2020. -/
def Delicious (n : ℤ) : Prop :=
  ∃ k : ℕ+, (Finset.range k).sum (fun i => n + i) = 2020

/-- The smallest delicious integer less than -2020 is -2021. -/
theorem smallest_delicious_integer :
  (∀ n < -2020, Delicious n → n ≥ -2021) ∧ Delicious (-2021) :=
sorry

end NUMINAMATH_CALUDE_smallest_delicious_integer_l3929_392989


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3929_392926

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (2 * x + y) + 1 / (y + 1) = 1) :
  x + 2 * y ≥ 1 / 2 + Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (2 * x₀ + y₀) + 1 / (y₀ + 1) = 1 ∧
    x₀ + 2 * y₀ = 1 / 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3929_392926


namespace NUMINAMATH_CALUDE_f_has_one_zero_when_a_gt_3_l3929_392913

-- Define the function f(x) = x^2 - ax + 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Define the property of having exactly one zero in the interval (0, 2)
def has_one_zero_in_interval (f : ℝ → ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 2 ∧ f x = 0

-- State the theorem
theorem f_has_one_zero_when_a_gt_3 (a : ℝ) (h : a > 3) :
  has_one_zero_in_interval (f a) := by
  sorry


end NUMINAMATH_CALUDE_f_has_one_zero_when_a_gt_3_l3929_392913


namespace NUMINAMATH_CALUDE_root_power_floor_l3929_392940

theorem root_power_floor (a : ℝ) : 
  a^5 - a^3 + a - 2 = 0 → ⌊a^6⌋ = 3 := by
sorry

end NUMINAMATH_CALUDE_root_power_floor_l3929_392940


namespace NUMINAMATH_CALUDE_intersection_points_count_l3929_392908

/-- The number of intersection points between y = |3x + 4| and y = -|4x + 3| -/
theorem intersection_points_count : ∃! p : ℝ × ℝ, 
  (p.2 = |3 * p.1 + 4|) ∧ (p.2 = -|4 * p.1 + 3|) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3929_392908


namespace NUMINAMATH_CALUDE_two_roots_condition_l3929_392973

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/4) * x + 1 else Real.log x

-- State the theorem
theorem two_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x = a * x ∧ f y = a * y ∧
   ∀ z : ℝ, z ≠ x ∧ z ≠ y → f z ≠ a * z) ↔
  a > 1/4 ∧ a < 1/Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l3929_392973


namespace NUMINAMATH_CALUDE_value_of_a_l3929_392982

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

-- State the theorem
theorem value_of_a : 
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3929_392982


namespace NUMINAMATH_CALUDE_complex_subtraction_l3929_392921

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 + I) (h2 : b = 2 - 3 * I) :
  a - 3 * b = 11 - 8 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3929_392921


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l3929_392976

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l3929_392976


namespace NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l3929_392990

theorem quadratic_equality_implies_coefficient (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 9 = (x - 3)^2) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equality_implies_coefficient_l3929_392990


namespace NUMINAMATH_CALUDE_count_divisors_eq_twelve_l3929_392924

/-- The number of natural numbers m such that 2023 ≡ 23 (mod m) -/
def count_divisors : ℕ :=
  (Finset.filter (fun m => m > 23 ∧ 2023 % m = 23) (Finset.range 2024)).card

theorem count_divisors_eq_twelve : count_divisors = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_eq_twelve_l3929_392924


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3929_392919

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3929_392919


namespace NUMINAMATH_CALUDE_gear_angular_speed_relationship_l3929_392954

theorem gear_angular_speed_relationship 
  (x y z : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (ω₁ ω₂ ω₃ : ℝ) 
  (h₁ : 2 * x * ω₁ = 3 * y * ω₂) 
  (h₂ : 3 * y * ω₂ = 4 * z * ω₃) :
  ∃ (k : ℝ), k > 0 ∧ 
    ω₁ = k * (2 * z / x) ∧
    ω₂ = k * (4 * z / (3 * y)) ∧
    ω₃ = k :=
by sorry

end NUMINAMATH_CALUDE_gear_angular_speed_relationship_l3929_392954


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3929_392971

-- Problem 1
theorem factorization_problem_1 (x : ℝ) :
  2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3929_392971


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3929_392900

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x^2 + 7 * x = 3.5 * (x - 4)^2 + 1.5 * (x - 2) * (x - 4) + 18 * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3929_392900


namespace NUMINAMATH_CALUDE_product_of_roots_plus_two_l3929_392965

theorem product_of_roots_plus_two (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_two_l3929_392965


namespace NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_l3929_392999

/-- Represents a discount as a fraction between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- Apply a discount to a price -/
def applyDiscount (price : ℝ) (discount : Discount) : ℝ :=
  price * (1 - discount.val)

/-- Apply two successive discounts -/
def applySuccessiveDiscounts (price : ℝ) (d1 d2 : Discount) : ℝ :=
  applyDiscount (applyDiscount price d1) d2

theorem successive_discounts_equivalent_to_single (price : ℝ) :
  let d1 : Discount := ⟨0.1, by norm_num⟩
  let d2 : Discount := ⟨0.2, by norm_num⟩
  let singleDiscount : Discount := ⟨0.28, by norm_num⟩
  applySuccessiveDiscounts price d1 d2 = applyDiscount price singleDiscount := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_l3929_392999


namespace NUMINAMATH_CALUDE_max_sum_constraint_l3929_392960

theorem max_sum_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
  16 * x' * y' * z' = (x' + y')^2 * (x' + z')^2 ∧ x' + y' + z' = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constraint_l3929_392960


namespace NUMINAMATH_CALUDE_jerry_candy_problem_l3929_392984

/-- Given a total number of candy pieces, number of bags, and the distribution of chocolate types,
    calculate the number of non-chocolate candy pieces. -/
def non_chocolate_candy (total_candy : ℕ) (total_bags : ℕ) (heart_bags : ℕ) (kiss_bags : ℕ) : ℕ :=
  total_candy - (heart_bags + kiss_bags) * (total_candy / total_bags)

/-- Theorem stating that given 63 pieces of candy divided into 9 bags,
    with 2 bags of chocolate hearts and 3 bags of chocolate kisses,
    the number of non-chocolate candies is 28. -/
theorem jerry_candy_problem :
  non_chocolate_candy 63 9 2 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_jerry_candy_problem_l3929_392984


namespace NUMINAMATH_CALUDE_max_xy_min_ratio_l3929_392927

theorem max_xy_min_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 4) : 
  (∀ a b, a > 0 → b > 0 → a + 2*b = 4 → a*b ≤ x*y) ∧ 
  (∀ a b, a > 0 → b > 0 → a + 2*b = 4 → y/x + 4/y ≤ a/b + 4/a) :=
sorry

end NUMINAMATH_CALUDE_max_xy_min_ratio_l3929_392927


namespace NUMINAMATH_CALUDE_total_miles_driven_is_2225_l3929_392991

/-- A structure representing a car's weekly fuel consumption and mileage. -/
structure Car where
  gallons_consumed : ℝ
  average_mpg : ℝ

/-- Calculates the total miles driven by a car given its fuel consumption and average mpg. -/
def miles_driven (car : Car) : ℝ :=
  car.gallons_consumed * car.average_mpg

/-- Represents the family's two cars and their combined mileage. -/
structure FamilyCars where
  car1 : Car
  car2 : Car
  total_average_mpg : ℝ

/-- Theorem stating that under the given conditions, the total miles driven by both cars is 2225. -/
theorem total_miles_driven_is_2225 (family_cars : FamilyCars)
    (h1 : family_cars.car1.gallons_consumed = 25)
    (h2 : family_cars.car2.gallons_consumed = 35)
    (h3 : family_cars.car1.average_mpg = 40)
    (h4 : family_cars.total_average_mpg = 75) :
    miles_driven family_cars.car1 + miles_driven family_cars.car2 = 2225 := by
  sorry


end NUMINAMATH_CALUDE_total_miles_driven_is_2225_l3929_392991


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3929_392967

theorem arithmetic_mean_of_fractions :
  (5 : ℚ) / 6 = ((3 : ℚ) / 4 + (7 : ℚ) / 8) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3929_392967


namespace NUMINAMATH_CALUDE_problem_statement_l3929_392981

theorem problem_statement (x : ℝ) (h : x = 4) : 5 * x + 7 = 27 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3929_392981


namespace NUMINAMATH_CALUDE_abc_fraction_equals_twelve_l3929_392994

theorem abc_fraction_equals_twelve
  (a b c m : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hsum : a + b + c = m)
  (hsquare_sum : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2*a)^2 + b * (m - 2*b)^2 + c * (m - 2*c)^2) / (a * b * c) = 12 :=
by sorry

end NUMINAMATH_CALUDE_abc_fraction_equals_twelve_l3929_392994


namespace NUMINAMATH_CALUDE_trivia_team_schools_l3929_392914

theorem trivia_team_schools (total_tryouts : ℝ) (not_picked : ℝ) (total_picked : ℕ) :
  total_tryouts = 65.0 →
  not_picked = 17.0 →
  total_picked = 384 →
  (total_picked : ℝ) / (total_tryouts - not_picked) = 8 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_schools_l3929_392914


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_products_l3929_392987

theorem sum_of_reciprocal_products (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2022*a^2 + 1011 = 0 →
  b^3 - 2022*b^2 + 1011 = 0 →
  c^3 - 2022*c^2 + 1011 = 0 →
  1/(a*b) + 1/(b*c) + 1/(a*c) = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_products_l3929_392987


namespace NUMINAMATH_CALUDE_school_students_count_l3929_392951

theorem school_students_count : ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ 
  n % 6 = 1 ∧ n % 8 = 2 ∧ n % 9 = 3 ∧ n = 265 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l3929_392951


namespace NUMINAMATH_CALUDE_executive_committee_formation_l3929_392909

theorem executive_committee_formation (total_members : ℕ) (committee_size : ℕ) (president : ℕ) :
  total_members = 30 →
  committee_size = 5 →
  president = 1 →
  Nat.choose (total_members - president) (committee_size - president) = 25839 :=
by sorry

end NUMINAMATH_CALUDE_executive_committee_formation_l3929_392909


namespace NUMINAMATH_CALUDE_stimulus_check_distribution_l3929_392966

theorem stimulus_check_distribution (total amount_to_wife amount_to_first_son amount_to_second_son savings : ℚ) :
  total = 2000 ∧
  amount_to_wife = (2 / 5) * total ∧
  amount_to_first_son = (2 / 5) * (total - amount_to_wife) ∧
  savings = 432 ∧
  amount_to_second_son = total - amount_to_wife - amount_to_first_son - savings →
  amount_to_second_son / (total - amount_to_wife - amount_to_first_son) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_stimulus_check_distribution_l3929_392966


namespace NUMINAMATH_CALUDE_arithmetic_sum_equals_twenty_l3929_392922

/-- The sum of an arithmetic sequence with 5 terms, where the first term is 2 and the last term is 6 -/
def arithmetic_sum : ℕ :=
  let n := 5  -- number of days
  let a₁ := 2 -- first day's distance
  let aₙ := 6 -- last day's distance
  n * (a₁ + aₙ) / 2

/-- The theorem states that the arithmetic sum defined above equals 20 -/
theorem arithmetic_sum_equals_twenty : arithmetic_sum = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_equals_twenty_l3929_392922


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3929_392959

theorem purely_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (b : ℝ), (1 - 2*Complex.I)*(a + Complex.I) = b*Complex.I) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3929_392959


namespace NUMINAMATH_CALUDE_triangle_equation_l3929_392953

theorem triangle_equation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let angleA : ℝ := 60 * π / 180
  (a^2 = b^2 + c^2 - 2*b*c*(angleA.cos)) →
  (3 / (a + b + c) = 1 / (a + b) + 1 / (a + c)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_equation_l3929_392953


namespace NUMINAMATH_CALUDE_bus_ride_difference_l3929_392950

theorem bus_ride_difference (vince_ride : ℝ) (zachary_ride : ℝ)
  (h1 : vince_ride = 0.62)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.12 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l3929_392950


namespace NUMINAMATH_CALUDE_dartboard_central_angles_l3929_392903

/-- Represents a region on a circular dartboard -/
structure DartboardRegion where
  probability : ℚ
  centralAngle : ℚ

/-- Theorem: Given the probabilities of hitting regions A and B on a circular dartboard,
    prove that their central angles are 45° and 30° respectively -/
theorem dartboard_central_angles 
  (regionA regionB : DartboardRegion)
  (hA : regionA.probability = 1/8)
  (hB : regionB.probability = 1/12)
  (h_total : regionA.centralAngle + regionB.centralAngle ≤ 360) :
  regionA.centralAngle = 45 ∧ regionB.centralAngle = 30 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angles_l3929_392903


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_800_l3929_392928

def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def isDistinctPowerSum (n : ℕ) (powers : List ℕ) : Prop :=
  (powers.length ≥ 2) ∧
  (powers.sum = n) ∧
  (∀ p ∈ powers, isPowerOfTwo p) ∧
  powers.Nodup

theorem least_exponent_sum_for_800 :
  ∃ (powers : List ℕ),
    isDistinctPowerSum 800 powers ∧
    (∀ (other_powers : List ℕ),
      isDistinctPowerSum 800 other_powers →
      (powers.map (fun p => (Nat.log p 2))).sum ≤ (other_powers.map (fun p => (Nat.log p 2))).sum) ∧
    (powers.map (fun p => (Nat.log p 2))).sum = 22 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_800_l3929_392928


namespace NUMINAMATH_CALUDE_sin_odd_function_phi_l3929_392941

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sin_odd_function_phi (φ : ℝ) :
  is_odd_function (λ x => Real.sin (x + φ)) → φ = π :=
sorry

end NUMINAMATH_CALUDE_sin_odd_function_phi_l3929_392941


namespace NUMINAMATH_CALUDE_square_diagonal_l3929_392935

theorem square_diagonal (perimeter : ℝ) (h : perimeter = 40) :
  let side := perimeter / 4
  let diagonal := side * Real.sqrt 2
  diagonal = 10 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_l3929_392935


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_at_least_two_zeros_l3929_392986

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 9 * 10^5

/-- The number of 6-digit numbers with no zeros -/
def numbers_with_no_zeros : ℕ := 9^6

/-- The number of 6-digit numbers with exactly one zero -/
def numbers_with_one_zero : ℕ := 5 * 9^5

/-- The number of 6-digit numbers with at least two zeros -/
def numbers_with_at_least_two_zeros : ℕ := total_six_digit_numbers - (numbers_with_no_zeros + numbers_with_one_zero)

theorem six_digit_numbers_with_at_least_two_zeros :
  numbers_with_at_least_two_zeros = 73314 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_at_least_two_zeros_l3929_392986


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3929_392964

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3929_392964


namespace NUMINAMATH_CALUDE_johns_journey_cost_l3929_392918

/-- Calculates the total cost of John's journey given the specified conditions. -/
theorem johns_journey_cost : 
  let rental_cost : ℚ := 150
  let rental_discount : ℚ := 0.15
  let gas_cost_per_gallon : ℚ := 3.5
  let gas_gallons : ℚ := 8
  let driving_cost_per_mile : ℚ := 0.5
  let initial_distance : ℚ := 320
  let additional_distance : ℚ := 50
  let toll_fees : ℚ := 15
  let parking_cost_per_day : ℚ := 20
  let parking_days : ℚ := 3
  let meals_lodging_cost_per_day : ℚ := 70
  let meals_lodging_days : ℚ := 2

  let discounted_rental := rental_cost * (1 - rental_discount)
  let total_gas_cost := gas_cost_per_gallon * gas_gallons
  let total_distance := initial_distance + additional_distance
  let total_driving_cost := driving_cost_per_mile * total_distance
  let total_parking_cost := parking_cost_per_day * parking_days
  let total_meals_lodging := meals_lodging_cost_per_day * meals_lodging_days

  let total_cost := discounted_rental + total_gas_cost + total_driving_cost + 
                    toll_fees + total_parking_cost + total_meals_lodging

  total_cost = 555.5 := by sorry

end NUMINAMATH_CALUDE_johns_journey_cost_l3929_392918


namespace NUMINAMATH_CALUDE_sum_diagonal_blocks_420_eq_2517_l3929_392933

/-- Given a 420 × 420 square grid tiled with 1 × 2 blocks, this function calculates
    the sum of all possible values for the total number of blocks
    that the two diagonals pass through. -/
def sum_diagonal_blocks_420 : ℕ :=
  let grid_size : ℕ := 420
  let diagonal_squares : ℕ := 2 * grid_size
  let non_center_squares : ℕ := diagonal_squares - 4
  let non_center_blocks : ℕ := non_center_squares
  let min_center_blocks : ℕ := 2
  let max_center_blocks : ℕ := 4
  (non_center_blocks + min_center_blocks) +
  (non_center_blocks + min_center_blocks + 1) +
  (non_center_blocks + max_center_blocks)

theorem sum_diagonal_blocks_420_eq_2517 :
  sum_diagonal_blocks_420 = 2517 := by
  sorry

end NUMINAMATH_CALUDE_sum_diagonal_blocks_420_eq_2517_l3929_392933


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3929_392943

theorem rectangle_area_diagonal (length width diagonal k : ℝ) : 
  length > 0 →
  width > 0 →
  diagonal > 0 →
  length / width = 5 / 2 →
  diagonal^2 = length^2 + width^2 →
  k = 10 / 29 →
  length * width = k * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3929_392943


namespace NUMINAMATH_CALUDE_tangerine_consumption_change_l3929_392942

/-- Demand function before embargo -/
def initial_demand (p : ℝ) : ℝ := 50 - p

/-- Demand function after embargo -/
def new_demand (p : ℝ) : ℝ := 2.5 * (50 - p)

/-- Marginal cost (constant) -/
def marginal_cost : ℝ := 5

/-- Initial equilibrium quantity under perfect competition -/
def initial_equilibrium_quantity : ℝ := initial_demand marginal_cost

/-- New equilibrium quantity under monopoly -/
noncomputable def new_equilibrium_quantity : ℝ := 56.25

theorem tangerine_consumption_change :
  new_equilibrium_quantity / initial_equilibrium_quantity = 1.25 := by sorry

end NUMINAMATH_CALUDE_tangerine_consumption_change_l3929_392942


namespace NUMINAMATH_CALUDE_divisor_degree_l3929_392923

-- Define the degrees of the polynomials
def deg_dividend : ℕ := 15
def deg_quotient : ℕ := 9
def deg_remainder : ℕ := 4

-- Theorem statement
theorem divisor_degree :
  ∀ (deg_divisor : ℕ),
    deg_dividend = deg_divisor + deg_quotient ∧
    deg_remainder < deg_divisor →
    deg_divisor = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_degree_l3929_392923


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3929_392947

-- Define probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define number of shots
def num_shots : ℕ := 4

-- Define the probability of A missing at least once in 4 shots
def prob_A_miss_at_least_once : ℚ := 1 - prob_A_hit^num_shots

-- Define the probability of A hitting exactly 2 times in 4 shots
def prob_A_hit_exactly_two : ℚ := 
  (num_shots.choose 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)^(num_shots - 2)

-- Define the probability of B hitting exactly 3 times in 4 shots
def prob_B_hit_exactly_three : ℚ :=
  (num_shots.choose 3 : ℚ) * prob_B_hit^3 * (1 - prob_B_hit)^(num_shots - 3)

-- Define the probability of B stopping after exactly 5 shots
def prob_B_stop_after_five : ℚ := 
  prob_B_hit^2 * (1 - prob_B_hit) * (1 - prob_B_hit^2)

theorem shooting_probabilities :
  prob_A_miss_at_least_once = 65/81 ∧
  prob_A_hit_exactly_two * prob_B_hit_exactly_three = 1/8 ∧
  prob_B_stop_after_five = 45/1024 := by sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l3929_392947


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l3929_392975

theorem rectangle_area_perimeter_sum (w : ℕ) (h : w > 0) : 
  let l := 2 * w
  let A := l * w
  let P := 2 * (l + w)
  A + P ≠ 110 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l3929_392975


namespace NUMINAMATH_CALUDE_expression_simplification_l3929_392962

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x * (y^2 + 1) / y) - ((x^2 - 1) / y * (y^3 - 1) / x) =
  (x^3 * y^2 - x^2 * y^3 + x^3 + x^2 + y^2 + y^3) / (x * y) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3929_392962


namespace NUMINAMATH_CALUDE_total_weight_is_5040_l3929_392963

/-- The weight of all settings for a catering event. -/
def total_weight_of_settings : ℕ :=
  let silverware_weight_per_piece : ℕ := 4
  let silverware_pieces_per_setting : ℕ := 3
  let plate_weight : ℕ := 12
  let plates_per_setting : ℕ := 2
  let tables : ℕ := 15
  let settings_per_table : ℕ := 8
  let backup_settings : ℕ := 20
  
  let total_settings : ℕ := tables * settings_per_table + backup_settings
  let weight_per_setting : ℕ := silverware_weight_per_piece * silverware_pieces_per_setting + 
                                 plate_weight * plates_per_setting
  
  total_settings * weight_per_setting

theorem total_weight_is_5040 : total_weight_of_settings = 5040 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_5040_l3929_392963


namespace NUMINAMATH_CALUDE_new_person_weight_l3929_392968

theorem new_person_weight (initial_count : Nat) (replaced_weight : Real) (avg_increase : Real) :
  initial_count = 8 →
  replaced_weight = 35 →
  avg_increase = 2.5 →
  (initial_count * avg_increase + replaced_weight : Real) = 55 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3929_392968


namespace NUMINAMATH_CALUDE_no_odd_pieces_all_diagonals_black_squares_count_equivalence_l3929_392932

/-- Represents a chess piece on a chessboard --/
structure ChessPiece where
  position : Nat × Nat
  color : Bool

/-- Represents a chessboard with pieces --/
def Chessboard := List ChessPiece

/-- Represents a diagonal on a chessboard --/
inductive Diagonal
| A1H8 : Nat → Diagonal  -- Diagonals parallel to a1-h8
| A8H1 : Nat → Diagonal  -- Diagonals parallel to a8-h1

/-- Returns the number of pieces on a given diagonal --/
def piecesOnDiagonal (board : Chessboard) (diag : Diagonal) : Nat :=
  sorry

/-- Checks if a number is odd --/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- Main theorem: It's impossible to have an odd number of pieces on all 30 diagonals --/
theorem no_odd_pieces_all_diagonals (board : Chessboard) : 
  ¬(∀ (d : Diagonal), isOdd (piecesOnDiagonal board d)) :=
by
  sorry

/-- Helper function to count pieces on black squares along a1-h8 diagonals --/
def countBlackSquaresA1H8 (board : Chessboard) : Nat :=
  sorry

/-- Helper function to count pieces on black squares along a8-h1 diagonals --/
def countBlackSquaresA8H1 (board : Chessboard) : Nat :=
  sorry

/-- Theorem: The two ways of counting pieces on black squares are equivalent --/
theorem black_squares_count_equivalence (board : Chessboard) :
  countBlackSquaresA1H8 board = countBlackSquaresA8H1 board :=
by
  sorry

end NUMINAMATH_CALUDE_no_odd_pieces_all_diagonals_black_squares_count_equivalence_l3929_392932


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3929_392905

def total_cars : ℕ := 300

def audi_percent : ℚ := 10 / 100
def toyota_percent : ℚ := 15 / 100
def acura_percent : ℚ := 20 / 100
def honda_percent : ℚ := 18 / 100

def other_brands_percent : ℚ := audi_percent + toyota_percent + acura_percent + honda_percent

def bmw_percent : ℚ := 1 - other_brands_percent

theorem bmw_sales_count : ⌊(bmw_percent * total_cars : ℚ)⌋ = 111 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3929_392905


namespace NUMINAMATH_CALUDE_p_true_q_false_l3929_392944

theorem p_true_q_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_CALUDE_p_true_q_false_l3929_392944


namespace NUMINAMATH_CALUDE_boxer_weight_theorem_l3929_392952

/-- Represents a diet with a specific weight loss per month -/
structure Diet where
  weightLossPerMonth : ℝ
  
/-- Calculates the weight after a given number of months on a diet -/
def weightAfterMonths (initialWeight : ℝ) (diet : Diet) (months : ℝ) : ℝ :=
  initialWeight - diet.weightLossPerMonth * months

/-- Theorem about boxer's weight and diets -/
theorem boxer_weight_theorem (x : ℝ) :
  let dietA : Diet := ⟨2⟩
  let dietB : Diet := ⟨3⟩
  let dietC : Diet := ⟨4⟩
  let monthsToFight : ℝ := 4
  
  (weightAfterMonths x dietB monthsToFight = 97) →
  (x = 109) ∧
  (weightAfterMonths x dietA monthsToFight = 101) ∧
  (weightAfterMonths x dietB monthsToFight = 97) ∧
  (weightAfterMonths x dietC monthsToFight = 93) := by
  sorry


end NUMINAMATH_CALUDE_boxer_weight_theorem_l3929_392952


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l3929_392917

-- Define the hyperbola equation
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := (Real.sqrt 3 * x + y = 0) ∧ (Real.sqrt 3 * x - y = 0)

-- Theorem statement
theorem hyperbola_b_value (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∀ x y, hyperbola x y b → asymptotes x y) : 
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l3929_392917


namespace NUMINAMATH_CALUDE_gcd_7_factorial_5_factorial_squared_l3929_392996

theorem gcd_7_factorial_5_factorial_squared : Nat.gcd (Nat.factorial 7) ((Nat.factorial 5)^2) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7_factorial_5_factorial_squared_l3929_392996


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3929_392956

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 1| > 1} = Set.Ioi (-2) ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3929_392956


namespace NUMINAMATH_CALUDE_expression_simplification_l3929_392904

theorem expression_simplification (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b*c)*(b^2 - a*c) + (a^2 - b*c)*(c^2 - a*b) + (b^2 - a*c)*(c^2 - a*b)) = k/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3929_392904


namespace NUMINAMATH_CALUDE_f_range_l3929_392907

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 2 * Real.sin x ^ 2 - 4 * Real.sin x + 3 * Real.cos x + 3 * Real.cos x ^ 2 - 2) / (Real.sin x - 1)

theorem f_range :
  Set.range (fun (x : ℝ) => f x) = Set.Icc 1 (1 + 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_range_l3929_392907


namespace NUMINAMATH_CALUDE_ball_box_arrangements_count_l3929_392977

/-- The number of ways to put 4 distinguishable balls into 4 distinguishable boxes,
    where one particular ball cannot be placed in one specific box. -/
def ball_box_arrangements : ℕ :=
  let num_balls : ℕ := 4
  let num_boxes : ℕ := 4
  let restricted_ball_choices : ℕ := num_boxes - 1
  let unrestricted_ball_choices : ℕ := num_boxes
  restricted_ball_choices * (unrestricted_ball_choices ^ (num_balls - 1))

/-- Theorem stating that the number of arrangements is 192. -/
theorem ball_box_arrangements_count : ball_box_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_arrangements_count_l3929_392977
