import Mathlib

namespace NUMINAMATH_CALUDE_ceiling_product_equation_solution_l476_47698

theorem ceiling_product_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ (⌈x⌉ : ℝ) * x = 210 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_solution_l476_47698


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_40_l476_47670

theorem smallest_four_digit_divisible_by_40 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 40 = 0 → n ≥ 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_40_l476_47670


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l476_47646

/-- Given a > 0 and f(x) = x³ + ax² - 9x - 1, if the tangent line with the smallest slope
    on the curve y = f(x) is perpendicular to the line x - 12y = 0, then a = 3 -/
theorem tangent_line_perpendicular (a : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 - 9*x - 1
  (∃ x₀ : ℝ, ∀ x : ℝ, (deriv f x₀ ≤ deriv f x) ∧ 
    (deriv f x₀ * (1 / 12) = -1)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l476_47646


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l476_47624

theorem cousins_ages_sum : 
  ∀ (a b c d : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10 ∧ 0 < c ∧ c < 10 ∧ 0 < d ∧ d < 10 →
  (a * b = 24 ∧ c * d = 30) ∨ (a * c = 24 ∧ b * d = 30) ∨ (a * d = 24 ∧ b * c = 30) →
  a + b + c + d = 22 :=
by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l476_47624


namespace NUMINAMATH_CALUDE_min_wins_for_playoffs_l476_47656

/-- 
Given a basketball league with the following conditions:
- Each game must have a winner
- A team earns 3 points for a win and loses 1 point for a loss
- The season consists of 32 games
- A team needs at least 48 points to have a chance at the playoffs

This theorem proves that a team must win at least 20 games to have a chance of advancing to the playoffs.
-/
theorem min_wins_for_playoffs (total_games : ℕ) (win_points loss_points : ℤ) (min_points : ℕ) : 
  total_games = 32 → win_points = 3 → loss_points = -1 → min_points = 48 → 
  ∃ (min_wins : ℕ), min_wins = 20 ∧ 
    ∀ (wins : ℕ), wins ≥ min_wins → 
      wins * win_points + (total_games - wins) * loss_points ≥ min_points :=
by sorry

end NUMINAMATH_CALUDE_min_wins_for_playoffs_l476_47656


namespace NUMINAMATH_CALUDE_largest_minimum_uniform_output_l476_47650

def black_box (n : ℕ) : ℕ :=
  if n % 2 = 1 then 4 * n + 1 else n / 2

def series_black_box (n : ℕ) : ℕ :=
  black_box (black_box (black_box n))

def is_valid_input (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < n ∧ b < n ∧ c < n ∧
  series_black_box a = series_black_box b ∧
  series_black_box b = series_black_box c ∧
  series_black_box c = series_black_box n

theorem largest_minimum_uniform_output :
  ∃ (n : ℕ), is_valid_input n ∧
  (∀ m, is_valid_input m → m ≤ n) ∧
  (∀ k, k < n → ¬is_valid_input k) ∧
  n = 680 :=
sorry

end NUMINAMATH_CALUDE_largest_minimum_uniform_output_l476_47650


namespace NUMINAMATH_CALUDE_quadratic_inequality_l476_47694

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 14 < 0 ↔ 2 < x ∧ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l476_47694


namespace NUMINAMATH_CALUDE_roses_apple_sharing_l476_47630

/-- Given that Rose has 9 apples and each friend receives 3 apples,
    prove that the number of friends Rose shares her apples with is 3. -/
theorem roses_apple_sharing :
  let total_apples : ℕ := 9
  let apples_per_friend : ℕ := 3
  total_apples / apples_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_roses_apple_sharing_l476_47630


namespace NUMINAMATH_CALUDE_f_pi_plus_3_l476_47619

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.tan x + 1

theorem f_pi_plus_3 (a b : ℝ) :
  f a b (-3) = 5 → f a b (Real.pi + 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_plus_3_l476_47619


namespace NUMINAMATH_CALUDE_trees_died_in_typhoon_l476_47680

theorem trees_died_in_typhoon (initial_trees left_trees : ℕ) : 
  initial_trees = 20 → left_trees = 4 → initial_trees - left_trees = 16 := by
  sorry

end NUMINAMATH_CALUDE_trees_died_in_typhoon_l476_47680


namespace NUMINAMATH_CALUDE_price_decrease_l476_47611

theorem price_decrease (original_price reduced_price : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.24))
  (h2 : reduced_price = 532) : original_price = 700 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_l476_47611


namespace NUMINAMATH_CALUDE_arjun_initial_investment_l476_47681

/-- Represents the investment details of a partner in the business --/
structure Investment where
  amount : ℝ
  duration : ℝ

/-- Calculates the share of a partner based on their investment and duration --/
def calculateShare (inv : Investment) : ℝ :=
  inv.amount * inv.duration

/-- Proves that Arjun's initial investment was 2000 given the problem conditions --/
theorem arjun_initial_investment 
  (arjun : Investment)
  (anoop : Investment)
  (h1 : arjun.duration = 12)
  (h2 : anoop.amount = 4000)
  (h3 : anoop.duration = 6)
  (h4 : calculateShare arjun = calculateShare anoop) : 
  arjun.amount = 2000 := by
  sorry

#check arjun_initial_investment

end NUMINAMATH_CALUDE_arjun_initial_investment_l476_47681


namespace NUMINAMATH_CALUDE_solution_to_equation_l476_47651

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^4 = (14 * x)^3 ∧ x = 8/7 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l476_47651


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l476_47640

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define symmetry about the line x + y = m
def symmetric_about_line (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  (x₁ + y₁ + x₂ + y₂) / 2 = m

-- Main theorem
theorem symmetric_points_on_parabola (x₁ y₁ x₂ y₂ m : ℝ) :
  is_on_parabola x₁ y₁ →
  is_on_parabola x₂ y₂ →
  symmetric_about_line x₁ y₁ x₂ y₂ m →
  y₁ * y₂ = -1/2 →
  m = 9/4 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l476_47640


namespace NUMINAMATH_CALUDE_alex_martin_games_l476_47655

/-- The number of players in the four-square league --/
def total_players : ℕ := 12

/-- The number of players in each game --/
def players_per_game : ℕ := 6

/-- The number of players to be chosen after Alex and Martin are included --/
def players_to_choose : ℕ := players_per_game - 2

/-- The number of remaining players after Alex and Martin are excluded --/
def remaining_players : ℕ := total_players - 2

/-- The number of times Alex plays in the same game as Martin --/
def games_together : ℕ := Nat.choose remaining_players players_to_choose

theorem alex_martin_games :
  games_together = 210 :=
sorry

end NUMINAMATH_CALUDE_alex_martin_games_l476_47655


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l476_47695

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  height : ℝ
  is_isosceles : True

/-- Represents a geometric solid -/
inductive Solid
  | Cylinder
  | Cone
  | Frustum

/-- The result of rotating an isosceles trapezoid -/
def rotate_isosceles_trapezoid (t : IsoscelesTrapezoid) : List Solid :=
  sorry

/-- Theorem stating that rotating an isosceles trapezoid around its longer base
    results in one cylinder and two cones -/
theorem isosceles_trapezoid_rotation 
  (t : IsoscelesTrapezoid) : 
  rotate_isosceles_trapezoid t = [Solid.Cylinder, Solid.Cone, Solid.Cone] :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_rotation_l476_47695


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l476_47660

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (Complex.mk (m + 2) (-1) = Complex.mk 0 (Complex.im (Complex.mk (m + 2) (-1)))) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l476_47660


namespace NUMINAMATH_CALUDE_pages_per_day_l476_47654

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 612) (h2 : days = 6) :
  total_pages / days = 102 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l476_47654


namespace NUMINAMATH_CALUDE_min_distance_point_triangle_l476_47633

/-- Given a triangle ABC with vertices (x₁, y₁), (x₂, y₂), and (x₃, y₃), 
    this theorem states that the point P which minimizes the sum of squared distances 
    to the vertices of triangle ABC has coordinates ((x₁ + x₂ + x₃)/3, (y₁ + y₂ + y₃)/3). -/
theorem min_distance_point_triangle (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) :
  let vertices := [(x₁, y₁), (x₂, y₂), (x₃, y₃)]
  let sum_squared_distances (px py : ℝ) := 
    (vertices.map (fun (x, y) => (px - x)^2 + (py - y)^2)).sum
  let p := ((x₁ + x₂ + x₃)/3, (y₁ + y₂ + y₃)/3)
  ∀ q : ℝ × ℝ, sum_squared_distances p.1 p.2 ≤ sum_squared_distances q.1 q.2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_triangle_l476_47633


namespace NUMINAMATH_CALUDE_nabla_calculation_l476_47689

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l476_47689


namespace NUMINAMATH_CALUDE_brick_surface_area_l476_47667

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm² -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l476_47667


namespace NUMINAMATH_CALUDE_cube_edge_length_l476_47648

-- Define the volume of the cube in milliliters
def cube_volume : ℝ := 729

-- Define the edge length of the cube in centimeters
def edge_length : ℝ := 9

-- Theorem: The edge length of a cube with volume 729 ml is 9 cm
theorem cube_edge_length : 
  edge_length ^ 3 * 1000 = cube_volume ∧ edge_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l476_47648


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l476_47621

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  z₁ = 2 + I → 
  z₂.re = -z₁.re ∧ z₂.im = z₁.im → 
  z₁ * z₂ = -5 := by sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l476_47621


namespace NUMINAMATH_CALUDE_isosceles_triangles_105_similar_l476_47665

-- Define an isosceles triangle with a specific angle
structure IsoscelesTriangle :=
  (base_angle : ℝ)
  (vertex_angle : ℝ)
  (is_isosceles : base_angle * 2 + vertex_angle = 180)

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.base_angle = t2.base_angle ∧ t1.vertex_angle = t2.vertex_angle

-- Theorem statement
theorem isosceles_triangles_105_similar :
  ∀ (t1 t2 : IsoscelesTriangle),
  t1.vertex_angle = 105 → t2.vertex_angle = 105 →
  are_similar t1 t2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_105_similar_l476_47665


namespace NUMINAMATH_CALUDE_tournament_games_played_l476_47639

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_teams : ℕ
  no_ties : Bool

/-- The number of games played in a single-elimination tournament -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 24 teams and no ties,
    the number of games played to declare a winner is 23 -/
theorem tournament_games_played :
  ∀ (t : SingleEliminationTournament),
    t.num_teams = 24 → t.no_ties = true →
    games_played t = 23 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_played_l476_47639


namespace NUMINAMATH_CALUDE_sum_of_digits_of_f_l476_47684

/-- The number of digits in (10^2020 + 2020)^2 when written out in full -/
def num_digits : ℕ := 4041

/-- The function that calculates (10^2020 + 2020)^2 -/
def f : ℕ := (10^2020 + 2020)^2

/-- The sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_f : sum_of_digits f = 25 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_f_l476_47684


namespace NUMINAMATH_CALUDE_larger_number_proof_l476_47645

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 2500) (h3 : L = 6 * S + 15) : L = 2997 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l476_47645


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l476_47677

theorem sqrt_eight_and_nine_sixteenths :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l476_47677


namespace NUMINAMATH_CALUDE_average_of_quadratic_solutions_l476_47699

/-- Given a quadratic equation ax² - 4ax + b = 0 with two real solutions,
    prove that the average of these solutions is 2. -/
theorem average_of_quadratic_solutions (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 - 4 * a * x + b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_quadratic_solutions_l476_47699


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l476_47617

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l476_47617


namespace NUMINAMATH_CALUDE_b_is_largest_l476_47601

/-- Represents a number with a finite or repeating decimal expansion -/
structure DecimalNumber where
  whole : ℕ
  finite : List ℕ
  repeating : List ℕ

/-- Converts a DecimalNumber to a real number -/
noncomputable def toReal (d : DecimalNumber) : ℝ :=
  sorry

/-- The five numbers we're comparing -/
def a : DecimalNumber := { whole := 8, finite := [1, 2, 3, 6, 6], repeating := [] }
def b : DecimalNumber := { whole := 8, finite := [1, 2, 3], repeating := [6] }
def c : DecimalNumber := { whole := 8, finite := [1, 2], repeating := [3, 6] }
def d : DecimalNumber := { whole := 8, finite := [1], repeating := [2, 3, 6] }
def e : DecimalNumber := { whole := 8, finite := [], repeating := [1, 2, 3, 6] }

/-- Theorem stating that b is the largest among the given numbers -/
theorem b_is_largest :
  (toReal b > toReal a) ∧
  (toReal b > toReal c) ∧
  (toReal b > toReal d) ∧
  (toReal b > toReal e) :=
by
  sorry

end NUMINAMATH_CALUDE_b_is_largest_l476_47601


namespace NUMINAMATH_CALUDE_complex_number_problem_l476_47622

theorem complex_number_problem (z ω : ℂ) :
  (((1 : ℂ) + 3*Complex.I) * z).re = 0 →
  ω = z / ((2 : ℂ) + Complex.I) →
  Complex.abs ω = 5 * Real.sqrt 2 →
  ω = 7 - Complex.I ∨ ω = -7 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l476_47622


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l476_47671

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l476_47671


namespace NUMINAMATH_CALUDE_flagpole_break_height_l476_47609

theorem flagpole_break_height (h : ℝ) (b : ℝ) (break_height : ℝ) :
  h = 8 →
  b = 3 →
  break_height = (Real.sqrt (h^2 + b^2)) / 2 →
  break_height = Real.sqrt 73 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l476_47609


namespace NUMINAMATH_CALUDE_point_on_line_l476_47649

/-- Given two points (m, n) and (m + p, n + 21) on the line x = (y / 7) - (2 / 5),
    prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 7 - 2 / 5) ∧ (m + p = (n + 21) / 7 - 2 / 5) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l476_47649


namespace NUMINAMATH_CALUDE_taxi_fare_equality_l476_47620

/-- Taxi fare calculation problem -/
theorem taxi_fare_equality (mike_start_fee annie_start_fee annie_toll_fee : ℚ)
  (per_mile_rate : ℚ) (annie_miles : ℚ) :
  mike_start_fee = 2.5 ∧
  annie_start_fee = 2.5 ∧
  annie_toll_fee = 5 ∧
  per_mile_rate = 0.25 ∧
  annie_miles = 22 →
  ∃ (mike_miles : ℚ),
    mike_start_fee + per_mile_rate * mike_miles =
    annie_start_fee + annie_toll_fee + per_mile_rate * annie_miles ∧
    mike_miles = 42 :=
by sorry

end NUMINAMATH_CALUDE_taxi_fare_equality_l476_47620


namespace NUMINAMATH_CALUDE_melanie_marbles_l476_47658

def sandy_marbles : ℕ := 56 * 12

theorem melanie_marbles : ∃ m : ℕ, m * 8 = sandy_marbles ∧ m = 84 := by
  sorry

end NUMINAMATH_CALUDE_melanie_marbles_l476_47658


namespace NUMINAMATH_CALUDE_probability_square_factor_l476_47661

/-- A standard 6-sided die -/
def StandardDie : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- The number of dice rolled -/
def NumDice : Nat := 6

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, n = m * m

/-- The probability of rolling a product containing a square factor -/
def probabilitySquareFactor : ℚ := 665 / 729

/-- Theorem stating the probability of rolling a product containing a square factor -/
theorem probability_square_factor :
  (1 : ℚ) - (2 / 3) ^ NumDice = probabilitySquareFactor := by sorry

end NUMINAMATH_CALUDE_probability_square_factor_l476_47661


namespace NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l476_47607

/-- Given that the binomial coefficients of the third and fourth terms 
    in the expansion of (x - 2/x)^n are equal, prove that the area enclosed 
    by the line y = nx and the curve y = x^2 is 125/6 -/
theorem area_enclosed_by_line_and_curve (n : ℕ) : 
  (Nat.choose n 2 = Nat.choose n 3) → 
  (∫ (x : ℝ) in (0)..(5), n * x - x^2) = 125 / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l476_47607


namespace NUMINAMATH_CALUDE_factorization_equality_l476_47603

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l476_47603


namespace NUMINAMATH_CALUDE_three_pipes_fill_time_l476_47643

/-- Represents the time taken to fill a tank given a number of pipes -/
def fill_time (num_pipes : ℕ) (time : ℝ) : Prop :=
  num_pipes > 0 ∧ time > 0 ∧ num_pipes * time = 36

theorem three_pipes_fill_time :
  fill_time 2 18 → fill_time 3 12 := by
  sorry

end NUMINAMATH_CALUDE_three_pipes_fill_time_l476_47643


namespace NUMINAMATH_CALUDE_minjoo_walked_distance_l476_47686

-- Define the distances walked by Yongchan and Min-joo
def yongchan_distance : ℝ := 1.05
def difference : ℝ := 0.46

-- Define Min-joo's distance as a function of Yongchan's distance and the difference
def minjoo_distance : ℝ := yongchan_distance - difference

-- Theorem statement
theorem minjoo_walked_distance : minjoo_distance = 0.59 := by
  sorry

end NUMINAMATH_CALUDE_minjoo_walked_distance_l476_47686


namespace NUMINAMATH_CALUDE_total_jumps_l476_47636

def hattie_first_round : ℕ := 180

def lorelei_first_round : ℕ := (3 * hattie_first_round) / 4

def hattie_second_round : ℕ := (2 * hattie_first_round) / 3

def lorelei_second_round : ℕ := hattie_second_round + 50

def hattie_third_round : ℕ := hattie_second_round + hattie_second_round / 3

def lorelei_third_round : ℕ := (4 * lorelei_first_round) / 5

theorem total_jumps :
  hattie_first_round + lorelei_first_round +
  hattie_second_round + lorelei_second_round +
  hattie_third_round + lorelei_third_round = 873 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_l476_47636


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l476_47682

theorem smallest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + 2 * z)) + Real.sqrt (y / (2 * x + z)) + Real.sqrt (z / (x + 2 * y)) > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l476_47682


namespace NUMINAMATH_CALUDE_sphere_in_cube_intersection_l476_47627

-- Define the cube and sphere
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the intersection of the sphere with a face
def intersectionRadius (s : Sphere) (face : List (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem sphere_in_cube_intersection (c : Cube) (s : Sphere) :
  s.radius = 10 →
  intersectionRadius s [c.vertices 0, c.vertices 1, c.vertices 4, c.vertices 5] = 1 →
  intersectionRadius s [c.vertices 4, c.vertices 5, c.vertices 6, c.vertices 7] = 1 →
  intersectionRadius s [c.vertices 2, c.vertices 3, c.vertices 6, c.vertices 7] = 3 →
  distance s.center (c.vertices 7) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cube_intersection_l476_47627


namespace NUMINAMATH_CALUDE_reciprocal_of_four_l476_47605

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_four : reciprocal 4 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_four_l476_47605


namespace NUMINAMATH_CALUDE_ben_sandwich_options_l476_47673

/-- Represents the number of different types for each sandwich component -/
structure SandwichOptions where
  bread : Nat
  meat : Nat
  cheese : Nat

/-- Represents specific sandwich combinations that are not allowed -/
structure ForbiddenCombinations where
  beef_swiss : Nat
  rye_turkey : Nat
  turkey_swiss : Nat

/-- Calculates the number of sandwich options given the available choices and forbidden combinations -/
def calculate_sandwich_options (options : SandwichOptions) (forbidden : ForbiddenCombinations) : Nat :=
  options.bread * options.meat * options.cheese - (forbidden.beef_swiss + forbidden.rye_turkey + forbidden.turkey_swiss)

/-- The main theorem stating the number of different sandwiches Ben could order -/
theorem ben_sandwich_options :
  let options : SandwichOptions := { bread := 5, meat := 7, cheese := 6 }
  let forbidden : ForbiddenCombinations := { beef_swiss := 5, rye_turkey := 6, turkey_swiss := 5 }
  calculate_sandwich_options options forbidden = 194 := by
  sorry

end NUMINAMATH_CALUDE_ben_sandwich_options_l476_47673


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l476_47666

/-- Converts a base-3 number represented as a list of digits to its base-10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number we're considering -/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l476_47666


namespace NUMINAMATH_CALUDE_calculate_boxes_l476_47659

/-- Given the number of blocks and blocks per box, calculate the number of boxes -/
theorem calculate_boxes (total_blocks : ℕ) (blocks_per_box : ℕ) (h : blocks_per_box > 0) :
  total_blocks / blocks_per_box = total_blocks / blocks_per_box :=
by sorry

/-- George's specific case -/
def george_boxes : ℕ :=
  let total_blocks : ℕ := 12
  let blocks_per_box : ℕ := 6
  total_blocks / blocks_per_box

#eval george_boxes

end NUMINAMATH_CALUDE_calculate_boxes_l476_47659


namespace NUMINAMATH_CALUDE_quadratic_roots_l476_47626

theorem quadratic_roots (k : ℝ) (C D : ℝ) : 
  (k * C^2 + 2 * C + 5 = 0) →
  (k * D^2 + 2 * D + 5 = 0) →
  (C = 10) →
  (D = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l476_47626


namespace NUMINAMATH_CALUDE_sum_equals_1529_l476_47657

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The value of C in base 14 -/
def C : Nat := 12

/-- The value of D in base 14 -/
def D : Nat := 13

/-- Theorem stating that 345₁₃ + 4CD₁₄ = 1529 in base 10 -/
theorem sum_equals_1529 : 
  toBase10 [5, 4, 3] 13 + toBase10 [D, C, 4] 14 = 1529 := by sorry

end NUMINAMATH_CALUDE_sum_equals_1529_l476_47657


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l476_47637

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_one_plus_i : (1 + i)^2 = 2*i := by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l476_47637


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l476_47614

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 6 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -32/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l476_47614


namespace NUMINAMATH_CALUDE_negation_equivalence_l476_47613

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l476_47613


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_zero_implies_zero_l476_47602

theorem sqrt_sum_squares_zero_implies_zero (a b : ℂ) : 
  Real.sqrt (Complex.abs a ^ 2 + Complex.abs b ^ 2) = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_zero_implies_zero_l476_47602


namespace NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l476_47608

theorem tan_value_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * π / 2) (2 * π)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = 1 / 5) →  -- given condition
  Real.tan α = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l476_47608


namespace NUMINAMATH_CALUDE_quadratic_monotone_decreasing_m_range_l476_47676

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

-- State the theorem
theorem quadratic_monotone_decreasing_m_range :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f m x₁ > f m x₂) →
  m ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotone_decreasing_m_range_l476_47676


namespace NUMINAMATH_CALUDE_silverware_probability_l476_47644

def forks : ℕ := 8
def spoons : ℕ := 10
def knives : ℕ := 6

def total_silverware : ℕ := forks + spoons + knives

def favorable_outcomes : ℕ := forks * spoons * knives

def total_outcomes : ℕ := Nat.choose total_silverware 3

theorem silverware_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 120 / 506 := by
  sorry

end NUMINAMATH_CALUDE_silverware_probability_l476_47644


namespace NUMINAMATH_CALUDE_train_crossing_time_l476_47669

/-- The time taken for a train to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length > 0 → train_speed_kmh > 0 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 
  (train_length / (train_speed_kmh * (5 / 18))) :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l476_47669


namespace NUMINAMATH_CALUDE_number_problem_l476_47693

theorem number_problem (x : ℤ) : x + 12 - 27 = 24 → x = 39 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l476_47693


namespace NUMINAMATH_CALUDE_three_cones_apex_angle_l476_47625

/-- Represents a cone with vertex at point A -/
structure Cone where
  apexAngle : ℝ

/-- Represents the configuration of three cones -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  touchingPlane : Bool
  sameSide : Bool

/-- The theorem statement -/
theorem three_cones_apex_angle 
  (config : ConeConfiguration)
  (h1 : config.cone1 = config.cone2)
  (h2 : config.cone3.apexAngle = π / 2)
  (h3 : config.touchingPlane)
  (h4 : config.sameSide) :
  config.cone1.apexAngle = 2 * Real.arctan (4 / 5) := by
  sorry


end NUMINAMATH_CALUDE_three_cones_apex_angle_l476_47625


namespace NUMINAMATH_CALUDE_five_people_round_table_l476_47696

/-- The number of unique seating arrangements for n people around a round table,
    where rotations are considered identical -/
def roundTableArrangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.factorial n / n

theorem five_people_round_table :
  roundTableArrangements 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_five_people_round_table_l476_47696


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l476_47610

theorem sum_of_fractions_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l476_47610


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l476_47612

theorem polynomial_division_theorem :
  let f (x : ℝ) := x^4 - 8*x^3 + 18*x^2 - 22*x + 8
  let g (x : ℝ) := x^2 - 3*x + k
  let r (x : ℝ) := x + a
  ∀ (k a : ℝ),
  (∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x) →
  (k = 8/3 ∧ a = 64/9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l476_47612


namespace NUMINAMATH_CALUDE_messages_cleared_in_29_days_l476_47606

/-- The number of days required to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  (initial_messages + (read_per_day - new_per_day) - 1) / (read_per_day - new_per_day)

/-- Proof that it takes 29 days to clear all unread messages -/
theorem messages_cleared_in_29_days :
  days_to_clear_messages 198 15 8 = 29 := by
  sorry

#eval days_to_clear_messages 198 15 8

end NUMINAMATH_CALUDE_messages_cleared_in_29_days_l476_47606


namespace NUMINAMATH_CALUDE_obtuse_angle_in_second_quadrant_l476_47678

/-- An angle is obtuse if it's greater than 90 degrees and less than 180 degrees -/
def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- An angle is in the second quadrant if it's greater than 90 degrees and less than or equal to 180 degrees -/
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α ≤ 180

/-- Theorem: An obtuse angle is an angle in the second quadrant -/
theorem obtuse_angle_in_second_quadrant (α : ℝ) :
  is_obtuse_angle α → is_in_second_quadrant α :=
by sorry

end NUMINAMATH_CALUDE_obtuse_angle_in_second_quadrant_l476_47678


namespace NUMINAMATH_CALUDE_range_of_omega_l476_47642

/-- Given a function f and its shifted version g, prove the range of ω -/
theorem range_of_omega (f g : ℝ → ℝ) (ω : ℝ) : 
  (ω > 0) →
  (∀ x, f x = Real.sin (π / 3 - ω * x)) →
  (∀ x, g x = Real.sin (ω * x - π / 3)) →
  (∀ x ∈ Set.Icc 0 π, -Real.sqrt 3 / 2 ≤ g x ∧ g x ≤ 1) →
  (5 / 6 : ℝ) ≤ ω ∧ ω ≤ (5 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_omega_l476_47642


namespace NUMINAMATH_CALUDE_min_triangles_is_eighteen_l476_47618

/-- Represents a non-convex hexagon formed by removing one corner square from an 8x8 chessboard -/
structure ChessboardHexagon where
  area : ℝ
  side_length : ℝ

/-- Calculates the minimum number of congruent triangles needed to partition the ChessboardHexagon -/
def min_congruent_triangles (h : ChessboardHexagon) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of congruent triangles is 18 -/
theorem min_triangles_is_eighteen (h : ChessboardHexagon) 
  (h_area : h.area = 63)
  (h_side : h.side_length = 8) : 
  min_congruent_triangles h = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_is_eighteen_l476_47618


namespace NUMINAMATH_CALUDE_yule_log_surface_area_increase_l476_47632

/-- Proves that cutting a cylindrical Yule log into 9 slices increases its surface area by 100π -/
theorem yule_log_surface_area_increase :
  let h : ℝ := 10  -- height of the log
  let d : ℝ := 5   -- diameter of the log
  let n : ℕ := 9   -- number of slices
  let r : ℝ := d / 2  -- radius of the log
  let original_surface_area : ℝ := 2 * π * r * h + 2 * π * r^2
  let slice_height : ℝ := h / n
  let slice_surface_area : ℝ := 2 * π * r * slice_height + 2 * π * r^2
  let total_sliced_surface_area : ℝ := n * slice_surface_area
  let surface_area_increase : ℝ := total_sliced_surface_area - original_surface_area
  surface_area_increase = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_yule_log_surface_area_increase_l476_47632


namespace NUMINAMATH_CALUDE_stamps_per_book_is_15_l476_47634

/-- The number of stamps in each book of the second type -/
def stamps_per_book : ℕ := sorry

/-- The total number of stamps Ruel has -/
def total_stamps : ℕ := 130

/-- The number of books of the first type (10 stamps each) -/
def books_type1 : ℕ := 4

/-- The number of stamps in each book of the first type -/
def stamps_per_book_type1 : ℕ := 10

/-- The number of books of the second type -/
def books_type2 : ℕ := 6

theorem stamps_per_book_is_15 : 
  stamps_per_book = 15 ∧ 
  total_stamps = books_type1 * stamps_per_book_type1 + books_type2 * stamps_per_book :=
by sorry

end NUMINAMATH_CALUDE_stamps_per_book_is_15_l476_47634


namespace NUMINAMATH_CALUDE_function_positive_implies_a_bound_l476_47629

/-- Given a function f(x) = x^2 - ax + 2 that is positive for all x > 2,
    prove that a ≤ 3. -/
theorem function_positive_implies_a_bound (a : ℝ) :
  (∀ x > 2, x^2 - a*x + 2 > 0) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_implies_a_bound_l476_47629


namespace NUMINAMATH_CALUDE_first_number_calculation_l476_47653

theorem first_number_calculation (average : ℝ) (num1 num2 added_num : ℝ) :
  average = 13 ∧ num1 = 16 ∧ num2 = 8 ∧ added_num = 22 →
  ∃ x : ℝ, (x + num1 + num2 + added_num) / 4 = average ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_first_number_calculation_l476_47653


namespace NUMINAMATH_CALUDE_star_one_two_l476_47635

-- Define the * operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- State the theorem
theorem star_one_two (a : ℝ) : star (star a 1) 2 = 6 * a + 5 := by
  sorry

end NUMINAMATH_CALUDE_star_one_two_l476_47635


namespace NUMINAMATH_CALUDE_total_distance_is_1734_l476_47668

/-- The number of trees in the row -/
def num_trees : ℕ := 18

/-- The interval between adjacent trees in meters -/
def tree_interval : ℕ := 3

/-- Calculate the total distance walked to water all trees -/
def total_distance : ℕ :=
  -- Sum of distances for each tree
  (Finset.range num_trees).sum (fun i => 2 * i * tree_interval)

/-- Theorem stating the total distance walked -/
theorem total_distance_is_1734 : total_distance = 1734 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_1734_l476_47668


namespace NUMINAMATH_CALUDE_song_length_proof_l476_47672

/-- Proves that given the conditions, each song on the album is 3.5 minutes long -/
theorem song_length_proof 
  (jumps_per_second : ℕ) 
  (total_songs : ℕ) 
  (total_jumps : ℕ) 
  (h1 : jumps_per_second = 1)
  (h2 : total_songs = 10)
  (h3 : total_jumps = 2100) :
  (total_jumps : ℚ) / (jumps_per_second * 60 * total_songs) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_song_length_proof_l476_47672


namespace NUMINAMATH_CALUDE_prob_less_than_5_eq_half_l476_47688

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The probability of an event occurring when rolling a fair 8-sided die -/
def prob (event : Finset ℕ) : ℚ := (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

/-- The event of rolling a number less than 5 -/
def less_than_5 : Finset ℕ := Finset.filter (λ x => x < 5) fair_8_sided_die

theorem prob_less_than_5_eq_half : 
  prob less_than_5 = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_less_than_5_eq_half_l476_47688


namespace NUMINAMATH_CALUDE_tan_periodic_equality_l476_47691

theorem tan_periodic_equality (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (1500 * π / 180) → m = 60 := by
  sorry

end NUMINAMATH_CALUDE_tan_periodic_equality_l476_47691


namespace NUMINAMATH_CALUDE_pentagon_percentage_is_fifty_percent_l476_47652

/-- Represents a tiling of the plane with squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each large square tile -/
  smallSquaresPerTile : ℕ
  /-- The number of smaller squares that form parts of pentagons -/
  smallSquaresInPentagons : ℕ

/-- Calculates the percentage of the plane enclosed by pentagons -/
def pentagonPercentage (tiling : PlaneTiling) : ℚ :=
  (tiling.smallSquaresInPentagons : ℚ) / (tiling.smallSquaresPerTile : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 50% -/
theorem pentagon_percentage_is_fifty_percent (tiling : PlaneTiling) 
  (h1 : tiling.smallSquaresPerTile = 16)
  (h2 : tiling.smallSquaresInPentagons = 8) : 
  pentagonPercentage tiling = 50 := by
  sorry

#eval pentagonPercentage { smallSquaresPerTile := 16, smallSquaresInPentagons := 8 }

end NUMINAMATH_CALUDE_pentagon_percentage_is_fifty_percent_l476_47652


namespace NUMINAMATH_CALUDE_equation_solutions_l476_47623

theorem equation_solutions : 
  let f (x : ℝ) := 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 
                   1 / ((x - 4) * (x - 5)) + 1 / ((x - 5) * (x - 6))
  ∀ x : ℝ, f x = 1 / 12 ↔ x = 12 ∨ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l476_47623


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l476_47600

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 40) :
  (perimeter / 4) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l476_47600


namespace NUMINAMATH_CALUDE_remaining_payment_l476_47616

def deposit_percentage : ℝ := 0.1
def deposit_amount : ℝ := 120

theorem remaining_payment (total : ℝ) (h1 : total * deposit_percentage = deposit_amount) :
  total - deposit_amount = 1080 := by sorry

end NUMINAMATH_CALUDE_remaining_payment_l476_47616


namespace NUMINAMATH_CALUDE_opposite_of_2023_l476_47687

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l476_47687


namespace NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l476_47631

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l476_47631


namespace NUMINAMATH_CALUDE_relationship_between_variables_l476_47663

theorem relationship_between_variables (a b c d : ℝ) 
  (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end NUMINAMATH_CALUDE_relationship_between_variables_l476_47663


namespace NUMINAMATH_CALUDE_simplify_expression_l476_47692

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 - 2*b + 4) - 2*b^2 = 9*b^3 - 8*b^2 + 12*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l476_47692


namespace NUMINAMATH_CALUDE_population_decrease_percentage_l476_47638

/-- Calculates the percentage of population that moved away after a growth spurt -/
def percentage_moved_away (initial_population : ℕ) (growth_rate : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_growth := initial_population * (1 + growth_rate)
  let people_moved_away := population_after_growth - final_population
  people_moved_away / population_after_growth

theorem population_decrease_percentage 
  (initial_population : ℕ) 
  (growth_rate : ℚ) 
  (final_population : ℕ) 
  (h1 : initial_population = 684) 
  (h2 : growth_rate = 1/4) 
  (h3 : final_population = 513) : 
  percentage_moved_away initial_population growth_rate final_population = 2/5 := by
  sorry

#eval percentage_moved_away 684 (1/4) 513

end NUMINAMATH_CALUDE_population_decrease_percentage_l476_47638


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l476_47664

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s > 0 ∧ s^3 = 7*x ∧ 6*s^2 = x) → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l476_47664


namespace NUMINAMATH_CALUDE_complex_sum_parts_l476_47683

theorem complex_sum_parts (z : ℂ) (h : z / (1 + 2*I) = 2 + I) : 
  (z + 5).re + (z + 5).im = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_parts_l476_47683


namespace NUMINAMATH_CALUDE_min_value_product_l476_47679

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a/b + b/c + c/a + b/a + c/b + a/c = 10)
  (h2 : a^2 + b^2 + c^2 = 9) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 91/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l476_47679


namespace NUMINAMATH_CALUDE_nori_crayons_left_l476_47662

def crayons_problem (boxes : ℕ) (crayons_per_box : ℕ) (given_to_mae : ℕ) (extra_to_lea : ℕ) : ℕ :=
  let total := boxes * crayons_per_box
  let after_mae := total - given_to_mae
  let given_to_lea := given_to_mae + extra_to_lea
  after_mae - given_to_lea

theorem nori_crayons_left :
  crayons_problem 4 8 5 7 = 15 := by
  sorry

end NUMINAMATH_CALUDE_nori_crayons_left_l476_47662


namespace NUMINAMATH_CALUDE_distance_traveled_l476_47641

theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 57 → time = 30 / 3600 → speed * time = 0.475 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l476_47641


namespace NUMINAMATH_CALUDE_non_zero_vector_positive_norm_l476_47674

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem non_zero_vector_positive_norm (a b : V) 
  (h_a : a ≠ 0) (h_b : ‖b‖ = 1) : 
  ‖a‖ > 0 := by sorry

end NUMINAMATH_CALUDE_non_zero_vector_positive_norm_l476_47674


namespace NUMINAMATH_CALUDE_no_month_with_five_mondays_and_thursdays_l476_47615

/-- Represents the possible number of days in a month -/
inductive MonthDays : Type where
  | days28 : MonthDays
  | days29 : MonthDays
  | days30 : MonthDays
  | days31 : MonthDays

/-- Converts MonthDays to a natural number -/
def monthDaysToNat (md : MonthDays) : Nat :=
  match md with
  | MonthDays.days28 => 28
  | MonthDays.days29 => 29
  | MonthDays.days30 => 30
  | MonthDays.days31 => 31

/-- Represents a day of the week -/
inductive Weekday : Type where
  | monday : Weekday
  | tuesday : Weekday
  | wednesday : Weekday
  | thursday : Weekday
  | friday : Weekday
  | saturday : Weekday
  | sunday : Weekday

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- Counts the number of occurrences of a specific weekday in a month -/
def countWeekday (startDay : Weekday) (monthLength : MonthDays) (day : Weekday) : Nat :=
  sorry  -- Implementation details omitted

theorem no_month_with_five_mondays_and_thursdays :
  ∀ (md : MonthDays) (start : Weekday),
    ¬(countWeekday start md Weekday.monday = 5 ∧ countWeekday start md Weekday.thursday = 5) :=
by sorry


end NUMINAMATH_CALUDE_no_month_with_five_mondays_and_thursdays_l476_47615


namespace NUMINAMATH_CALUDE_kylies_daisies_l476_47628

/-- Proves that Kylie's initial number of daisies is 5 given the problem conditions -/
theorem kylies_daisies (initial : ℕ) (sister_gift : ℕ) (remaining : ℕ) : 
  sister_gift = 9 → 
  remaining = 7 → 
  (initial + sister_gift) / 2 = remaining → 
  initial = 5 := by
sorry

end NUMINAMATH_CALUDE_kylies_daisies_l476_47628


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l476_47685

theorem smallest_divisor_with_remainder (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  x % 9 = 2 →
  x % 7 = 4 →
  y % 13 = 12 →
  y - x = 14 →
  y % z = 3 →
  (∀ w : ℕ, w > 0 ∧ w < z ∧ y % w = 3 → False) →
  z = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l476_47685


namespace NUMINAMATH_CALUDE_solution_set_equality_l476_47697

-- Define the set S
def S : Set ℝ := {x | |x + 2| + |x - 1| ≤ 4}

-- State the theorem
theorem solution_set_equality : S = Set.Icc (-5/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l476_47697


namespace NUMINAMATH_CALUDE_trisomicCrossRatio_l476_47647

-- Define the basic types
inductive Genotype
| BB
| Bb
| bb
| Bbb
| bbb

inductive Gamete
| B
| b
| Bb
| bb

-- Define the meiosis process for trisomic cells
def trisomicMeiosis (g : Genotype) : List Gamete := sorry

-- Define the fertilization process
def fertilize (female : Gamete) (male : Gamete) : Option Genotype := sorry

-- Define the phenotype (disease resistance) based on genotype
def isResistant (g : Genotype) : Bool := sorry

-- Define the cross between two plants
def cross (female : Genotype) (male : Genotype) : List Genotype := sorry

-- Define the ratio calculation function
def ratioResistantToSusceptible (offspring : List Genotype) : Rat := sorry

-- Theorem statement
theorem trisomicCrossRatio :
  let femaleParent : Genotype := Genotype.bbb
  let maleParent : Genotype := Genotype.BB
  let f1 : List Genotype := cross femaleParent maleParent
  let f1Trisomic : Genotype := Genotype.Bbb
  let susceptibleNormal : Genotype := Genotype.bb
  let f2 : List Genotype := cross f1Trisomic susceptibleNormal
  ratioResistantToSusceptible f2 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_trisomicCrossRatio_l476_47647


namespace NUMINAMATH_CALUDE_range_of_a_l476_47604

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | (x - a) / (x + a) < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 1 ∉ A a ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l476_47604


namespace NUMINAMATH_CALUDE_curve_k_values_l476_47675

-- Define the curve equation
def curve_equation (x y k : ℝ) : Prop :=
  5 * x^2 - k * y^2 = 5

-- Define the focal length
def focal_length : ℝ := 4

-- Theorem statement
theorem curve_k_values :
  ∃ k : ℝ, (k = 5/3 ∨ k = -1) ∧
  ∀ x y : ℝ, curve_equation x y k ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ curve_equation x y k) ∧
    (max a b - min a b) / 2 = focal_length) :=
sorry

end NUMINAMATH_CALUDE_curve_k_values_l476_47675


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l476_47690

theorem perfect_square_trinomial_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ y : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) → 
  (m = 20 ∨ m = -20) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l476_47690
