import Mathlib

namespace NUMINAMATH_CALUDE_computer_price_ratio_l1405_140517

theorem computer_price_ratio (c : ℝ) (h1 : c > 0) (h2 : c * 1.3 = 351) :
  (c + 351) / c = 2.3 := by
sorry

end NUMINAMATH_CALUDE_computer_price_ratio_l1405_140517


namespace NUMINAMATH_CALUDE_math_city_intersections_l1405_140551

/-- Represents a city with straight streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  (c.num_streets.choose 2)

/-- Theorem: A city with 10 streets meeting the given conditions has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 → c.no_parallel → c.no_triple_intersections →
  num_intersections c = 45 := by
  sorry

#check math_city_intersections

end NUMINAMATH_CALUDE_math_city_intersections_l1405_140551


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l1405_140581

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The problem statement -/
theorem bouncing_ball_distance :
  totalDistance 200 (2/3) 4 = 4200 :=
sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l1405_140581


namespace NUMINAMATH_CALUDE_find_k_l1405_140577

def vector_a (k : ℝ) : ℝ × ℝ := (k, 3)
def vector_b : ℝ × ℝ := (1, 4)
def vector_c : ℝ × ℝ := (2, 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_k : ∃ k : ℝ, 
  perpendicular ((2 * (vector_a k).1 - 3 * vector_b.1, 2 * (vector_a k).2 - 3 * vector_b.2)) vector_c ∧ 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1405_140577


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l1405_140513

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  a + 4 * b ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l1405_140513


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l1405_140586

/-- An ellipse with given properties -/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse with the given properties -/
def ellipse_area (e : Ellipse) : ℝ :=
  sorry

/-- The theorem stating that the area of the specific ellipse is 42π -/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-5, -1),
    major_axis_endpoint2 := (15, -1),
    point_on_ellipse := (12, 2)
  }
  ellipse_area e = 42 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l1405_140586


namespace NUMINAMATH_CALUDE_find_y_value_l1405_140585

theorem find_y_value (y : ℝ) (h : (15^2 * 8^3) / y = 450) : y = 256 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1405_140585


namespace NUMINAMATH_CALUDE_stacy_paper_completion_time_l1405_140582

/-- The number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 
  63 / 9

/-- The total number of pages in Stacy's history paper -/
def total_pages : ℕ := 63

/-- The number of pages Stacy has to write per day -/
def pages_per_day : ℕ := 9

/-- Theorem stating that Stacy has 7 days to complete her paper -/
theorem stacy_paper_completion_time : days_to_complete = 7 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_completion_time_l1405_140582


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l1405_140546

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  h : ℕ
  c : ℕ
  o : ℕ

/-- Atomic weights of elements in atomic mass units (amu) -/
def atomic_weight (element : String) : ℕ :=
  match element with
  | "H" => 1
  | "C" => 12
  | "O" => 16
  | _ => 0

/-- Calculate the molecular weight of a compound -/
def molecular_weight (compound : Compound) : ℕ :=
  compound.h * atomic_weight "H" +
  compound.c * atomic_weight "C" +
  compound.o * atomic_weight "O"

/-- Theorem: A compound with 2 H atoms, 1 C atom, and molecular weight 62 amu has 3 O atoms -/
theorem compound_oxygen_atoms (compound : Compound) :
  compound.h = 2 ∧ compound.c = 1 ∧ molecular_weight compound = 62 →
  compound.o = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l1405_140546


namespace NUMINAMATH_CALUDE_lineup_combinations_l1405_140514

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_players : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_players - 1) * (total_players - 2) * (total_players - 3)

/-- Theorem stating the number of ways to choose a starting lineup for the given team composition -/
theorem lineup_combinations : choose_lineup 12 4 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l1405_140514


namespace NUMINAMATH_CALUDE_max_value_3x_4y_l1405_140529

theorem max_value_3x_4y (x y : ℝ) : 
  x^2 + y^2 = 14*x + 6*y + 6 → (3*x + 4*y ≤ 73) := by
  sorry

end NUMINAMATH_CALUDE_max_value_3x_4y_l1405_140529


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_l1405_140545

/-- The line x + y - 2 = 0 -/
def line (x y : ℝ) : Prop := x + y - 2 = 0

/-- The curve x^2 + y^2 - 12x - 12y + 54 = 0 -/
def curve (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 12*y + 54 = 0

/-- The circle with center (6, 6) and radius 3√2 -/
def small_circle (x y : ℝ) : Prop := (x - 6)^2 + (y - 6)^2 = (3 * Real.sqrt 2)^2

/-- A circle is tangent to the line and curve if it touches them at exactly one point each -/
def is_tangent (circle line curve : ℝ → ℝ → Prop) : Prop := sorry

/-- A circle has the smallest radius if no other circle with a smaller radius is tangent to both the line and curve -/
def has_smallest_radius (circle line curve : ℝ → ℝ → Prop) : Prop := sorry

theorem smallest_tangent_circle :
  is_tangent small_circle line curve ∧ has_smallest_radius small_circle line curve := by sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_l1405_140545


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1405_140501

theorem trigonometric_equality (α : ℝ) :
  1 + Real.sin (3 * (α + π / 2)) * Real.cos (2 * α) +
  2 * Real.sin (3 * α) * Real.cos (3 * π - α) * Real.sin (α - π) =
  2 * (Real.sin (5 * α / 2))^2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1405_140501


namespace NUMINAMATH_CALUDE_lawn_width_calculation_l1405_140560

/-- Calculates the width of a rectangular lawn given specific conditions. -/
theorem lawn_width_calculation (length width road_width cost_per_sqm total_cost : ℝ) 
  (h1 : length = 80)
  (h2 : road_width = 10)
  (h3 : cost_per_sqm = 3)
  (h4 : total_cost = 3900)
  (h5 : (road_width * width + road_width * length - road_width * road_width) * cost_per_sqm = total_cost) :
  width = 60 := by
sorry

end NUMINAMATH_CALUDE_lawn_width_calculation_l1405_140560


namespace NUMINAMATH_CALUDE_m_values_l1405_140558

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem m_values : 
  {m : ℝ | B m ⊆ A} = {1/3, -1/2} := by sorry

end NUMINAMATH_CALUDE_m_values_l1405_140558


namespace NUMINAMATH_CALUDE_negation_of_all_positive_square_plus_one_l1405_140518

theorem negation_of_all_positive_square_plus_one (q : Prop) : 
  (q ↔ ∀ x : ℝ, x^2 + 1 > 0) →
  (¬q ↔ ∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_positive_square_plus_one_l1405_140518


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1405_140580

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1405_140580


namespace NUMINAMATH_CALUDE_triangle_problem_l1405_140563

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  let vec_a : ℝ × ℝ := (Real.cos A, Real.cos B)
  let vec_b : ℝ × ℝ := (a, 2*c - b)
  (∃ k : ℝ, vec_a = k • vec_b) →  -- vectors are parallel
  b = 3 →
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 →  -- area condition
  A = π/3 ∧ a = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1405_140563


namespace NUMINAMATH_CALUDE_paris_visits_l1405_140575

/-- Represents the attractions in Paris --/
inductive Attraction
  | EiffelTower
  | ArcDeTriomphe
  | Montparnasse
  | Playground

/-- Represents a nephew's statement about visiting an attraction --/
structure Statement where
  attraction : Attraction
  visited : Bool

/-- Represents a nephew's set of statements --/
structure NephewStatements where
  statements : List Statement

/-- The statements made by the three nephews --/
def nephewsStatements : List NephewStatements := [
  { statements := [
    { attraction := Attraction.EiffelTower, visited := true },
    { attraction := Attraction.ArcDeTriomphe, visited := true },
    { attraction := Attraction.Montparnasse, visited := false }
  ] },
  { statements := [
    { attraction := Attraction.EiffelTower, visited := true },
    { attraction := Attraction.Montparnasse, visited := true },
    { attraction := Attraction.ArcDeTriomphe, visited := false },
    { attraction := Attraction.Playground, visited := false }
  ] },
  { statements := [
    { attraction := Attraction.EiffelTower, visited := false },
    { attraction := Attraction.ArcDeTriomphe, visited := true }
  ] }
]

/-- The theorem to prove --/
theorem paris_visits (statements : List NephewStatements) 
  (h : statements = nephewsStatements) : 
  ∃ (visits : List Attraction),
    visits = [Attraction.EiffelTower, Attraction.ArcDeTriomphe, Attraction.Montparnasse] ∧
    Attraction.Playground ∉ visits :=
sorry

end NUMINAMATH_CALUDE_paris_visits_l1405_140575


namespace NUMINAMATH_CALUDE_y_derivative_l1405_140528

noncomputable def y (x : ℝ) : ℝ := Real.cos (Real.log 13) - (1 / 44) * (Real.cos (22 * x))^2 / Real.sin (44 * x)

theorem y_derivative (x : ℝ) (h : Real.sin (22 * x) ≠ 0) : 
  deriv y x = 1 / (4 * (Real.sin (22 * x))^2) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l1405_140528


namespace NUMINAMATH_CALUDE_max_value_product_sum_l1405_140511

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l1405_140511


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l1405_140572

/-- The Ferris wheel problem -/
theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 9) (h2 : total_people = 18) :
  total_people / people_per_seat = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l1405_140572


namespace NUMINAMATH_CALUDE_stock_value_change_l1405_140531

/-- Calculates the net percentage change in stock value over three years --/
def netPercentageChange (year1Change year2Change year3Change dividend : ℝ) : ℝ :=
  let value1 := (1 + year1Change) * (1 + dividend)
  let value2 := value1 * (1 + year2Change) * (1 + dividend)
  let value3 := value2 * (1 + year3Change) * (1 + dividend)
  (value3 - 1) * 100

/-- The net percentage change in stock value is approximately 17.52% --/
theorem stock_value_change :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧
  |netPercentageChange (-0.08) 0.10 0.06 0.03 - 17.52| < ε :=
sorry

end NUMINAMATH_CALUDE_stock_value_change_l1405_140531


namespace NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l1405_140571

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that the sum of digits of N is 14, where N is the number of rows in a triangular array containing 3003 coins -/
theorem sum_of_digits_of_triangular_array_rows :
  ∃ N : ℕ, triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l1405_140571


namespace NUMINAMATH_CALUDE_number_of_complementary_sets_l1405_140506

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 3
  color : Fin 3
  shade : Fin 3
  size : Fin 3

/-- The deck of all possible cards -/
def deck : Finset Card := sorry

/-- Checks if a set of three cards is complementary -/
def isComplementary (c1 c2 c3 : Card) : Prop := sorry

/-- The set of all complementary three-card sets -/
def complementarySets : Finset (Finset Card) := sorry

theorem number_of_complementary_sets :
  Finset.card complementarySets = 4536 := by sorry

end NUMINAMATH_CALUDE_number_of_complementary_sets_l1405_140506


namespace NUMINAMATH_CALUDE_unique_solution_equation_l1405_140541

theorem unique_solution_equation :
  ∃! y : ℝ, (3 * y^2 - 12 * y) / (y^2 - 4 * y) = y - 2 ∧
             y ≠ 2 ∧
             y^2 - 4 * y ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l1405_140541


namespace NUMINAMATH_CALUDE_fixed_monthly_costs_l1405_140564

/-- A problem about calculating fixed monthly costs for a computer manufacturer. -/
theorem fixed_monthly_costs (production_cost shipping_cost monthly_units lowest_price : ℕ) :
  production_cost = 80 →
  shipping_cost = 2 →
  monthly_units = 150 →
  lowest_price = 190 →
  (production_cost + shipping_cost) * monthly_units + 16200 = lowest_price * monthly_units :=
by sorry

end NUMINAMATH_CALUDE_fixed_monthly_costs_l1405_140564


namespace NUMINAMATH_CALUDE_min_additional_wins_correct_l1405_140592

/-- The minimum number of additional wins required to achieve a 90% winning percentage -/
def min_additional_wins : ℕ := 26

/-- The initial number of games played -/
def initial_games : ℕ := 4

/-- The initial number of games won -/
def initial_wins : ℕ := 1

/-- The target winning percentage -/
def target_percentage : ℚ := 9/10

theorem min_additional_wins_correct :
  ∀ n : ℕ, 
    (n ≥ min_additional_wins) ↔ 
    ((initial_wins + n : ℚ) / (initial_games + n)) ≥ target_percentage :=
sorry

end NUMINAMATH_CALUDE_min_additional_wins_correct_l1405_140592


namespace NUMINAMATH_CALUDE_equilateral_triangle_from_polynomial_roots_l1405_140548

theorem equilateral_triangle_from_polynomial_roots (a b c : ℂ) :
  (∀ z : ℂ, z^3 + 5*z + 7 = 0 ↔ z = a ∨ z = b ∨ z = c) →
  Complex.abs a ^ 2 + Complex.abs b ^ 2 + Complex.abs c ^ 2 = 300 →
  Complex.abs (a - b) = Complex.abs (b - c) ∧ 
  Complex.abs (b - c) = Complex.abs (c - a) →
  (Complex.abs (a - b))^2 = 225 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_from_polynomial_roots_l1405_140548


namespace NUMINAMATH_CALUDE_equidistant_points_l1405_140587

/-- Two points are equidistant if the larger of their distances to the x and y axes are equal -/
def equidistant (p q : ℝ × ℝ) : Prop :=
  max (|p.1|) (|p.2|) = max (|q.1|) (|q.2|)

theorem equidistant_points :
  (equidistant (-3, 7) (3, -7) ∧ equidistant (-3, 7) (7, 4)) ∧
  (equidistant (-4, 2) (-4, -3) ∧ equidistant (-4, 2) (3, 4)) ∧
  (equidistant (3, 4 + 2) (2 * 2 - 5, 6) ∧ equidistant (3, 4 + 9) (2 * 9 - 5, 6)) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_l1405_140587


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_two_plus_i_l1405_140527

theorem imaginary_part_of_i_times_two_plus_i (i : ℂ) : 
  (i * i = -1) → Complex.im (i * (2 + i)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_two_plus_i_l1405_140527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1405_140508

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 3 = 2) →
  (a 3 + a 5 = 4) →
  (a 5 + a 7 = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1405_140508


namespace NUMINAMATH_CALUDE_calzone_ratio_l1405_140567

def calzone_problem (onion_time garlic_time knead_time rest_time assemble_time total_time : ℕ) : Prop :=
  let pepper_time := garlic_time
  onion_time = 20 ∧
  garlic_time = onion_time / 4 ∧
  knead_time = 30 ∧
  rest_time = 2 * knead_time ∧
  total_time = 124 ∧
  total_time = onion_time + garlic_time + pepper_time + knead_time + rest_time + assemble_time

theorem calzone_ratio (onion_time garlic_time knead_time rest_time assemble_time total_time : ℕ) :
  calzone_problem onion_time garlic_time knead_time rest_time assemble_time total_time →
  (assemble_time : ℚ) / (knead_time + rest_time : ℚ) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_calzone_ratio_l1405_140567


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1405_140593

theorem negation_of_universal_proposition (f : ℝ → ℝ) :
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x₀ : ℝ, f x₀ ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1405_140593


namespace NUMINAMATH_CALUDE_triangle_side_length_l1405_140542

theorem triangle_side_length (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2) 
  (h_sin : Real.sin A = 3/5) (h_AB : AB = 5) (h_AC : AC = 6) : BC = Real.sqrt 13 := by
  sorry

#check triangle_side_length

end NUMINAMATH_CALUDE_triangle_side_length_l1405_140542


namespace NUMINAMATH_CALUDE_abs_product_zero_implies_one_equal_one_l1405_140569

theorem abs_product_zero_implies_one_equal_one (a b : ℝ) :
  |a - 1| * |b - 1| = 0 → a = 1 ∨ b = 1 := by
sorry

end NUMINAMATH_CALUDE_abs_product_zero_implies_one_equal_one_l1405_140569


namespace NUMINAMATH_CALUDE_kellys_games_l1405_140512

/-- Kelly's Nintendo games problem -/
theorem kellys_games (initial_games given_away_games : ℕ) : 
  initial_games = 121 → given_away_games = 99 → 
  initial_games - given_away_games = 22 := by
  sorry

end NUMINAMATH_CALUDE_kellys_games_l1405_140512


namespace NUMINAMATH_CALUDE_max_sum_of_digits_12hour_clock_l1405_140568

/-- Represents a time in 12-hour format -/
structure Time12Hour where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours ≥ 1 ∧ hours ≤ 12
  minutes_valid : minutes ≤ 59
  seconds_valid : seconds ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given Time12Hour -/
def sumOfTimeDigits (t : Time12Hour) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The maximum sum of digits on a 12-hour format digital clock -/
theorem max_sum_of_digits_12hour_clock :
  ∃ (t : Time12Hour), ∀ (t' : Time12Hour), sumOfTimeDigits t ≥ sumOfTimeDigits t' ∧ sumOfTimeDigits t = 37 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_12hour_clock_l1405_140568


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_l1405_140566

/-- Represents the driving scenario for Daniel's commute --/
structure DrivingScenario where
  x : ℝ  -- Speed on Sunday in miles per hour
  y : ℝ  -- Speed for first 32 miles on Monday in miles per hour
  total_distance : ℝ  -- Total distance in miles
  first_part_distance : ℝ  -- Distance of first part on Monday in miles

/-- The theorem stating the ratio of speeds --/
theorem speed_ratio_is_two (scenario : DrivingScenario) : 
  scenario.x > 0 → 
  scenario.y > 0 → 
  scenario.total_distance = 60 → 
  scenario.first_part_distance = 32 → 
  (scenario.first_part_distance / scenario.y + (scenario.total_distance - scenario.first_part_distance) / (scenario.x / 2)) = 
    1.2 * (scenario.total_distance / scenario.x) → 
  scenario.y / scenario.x = 2 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_two_l1405_140566


namespace NUMINAMATH_CALUDE_constant_distance_l1405_140519

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line (x y m : ℝ) : Prop := y = (1/2) * x + m

-- Define the constraint on m
def m_constraint (m : ℝ) : Prop := -Real.sqrt 2 < m ∧ m < Real.sqrt 2

-- Define the intersection points A and C
def intersection_points (xa ya xc yc m : ℝ) : Prop :=
  ellipse xa ya ∧ ellipse xc yc ∧ line xa ya m ∧ line xc yc m

-- Define the square ABCD
def square_ABCD (xa ya xb yb xc yc xd yd : ℝ) : Prop :=
  (xc - xa)^2 + (yc - ya)^2 = (xd - xb)^2 + (yd - yb)^2 ∧
  (xb - xa)^2 + (yb - ya)^2 = (xd - xc)^2 + (yd - yc)^2

-- Define point N
def point_N (xn m : ℝ) : Prop := xn = -2 * m

-- Main theorem
theorem constant_distance
  (m xa ya xb yb xc yc xd yd xn : ℝ)
  (h_m : m_constraint m)
  (h_int : intersection_points xa ya xc yc m)
  (h_square : square_ABCD xa ya xb yb xc yc xd yd)
  (h_N : point_N xn m) :
  (xb - xn)^2 + yb^2 = 5/2 :=
sorry

end NUMINAMATH_CALUDE_constant_distance_l1405_140519


namespace NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l1405_140578

theorem ratio_sum_squares_to_sum (a b c : ℝ) : 
  (b = 2 * a) → 
  (c = 4 * a) → 
  (a^2 + b^2 + c^2 = 1701) → 
  (a + b + c = 63) := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_to_sum_l1405_140578


namespace NUMINAMATH_CALUDE_solve_for_y_l1405_140579

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 64) (h2 : x = 8) : y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1405_140579


namespace NUMINAMATH_CALUDE_worker_a_time_l1405_140554

theorem worker_a_time (worker_b_time worker_ab_time : ℝ) 
  (hb : worker_b_time = 15)
  (hab : worker_ab_time = 20 / 3) : 
  ∃ worker_a_time : ℝ, 
    worker_a_time = 12 ∧ 
    1 / worker_a_time + 1 / worker_b_time = 1 / worker_ab_time :=
by sorry

end NUMINAMATH_CALUDE_worker_a_time_l1405_140554


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l1405_140584

theorem continued_fraction_sum (x y z : ℕ+) : 
  (151 : ℚ) / 44 = 3 + 1 / (x.val + 1 / (y.val + 1 / z.val)) → 
  x.val + y.val + z.val = 11 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l1405_140584


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l1405_140516

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem sum_of_max_min_f : 
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, m ≤ f x) ∧ (M + m = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l1405_140516


namespace NUMINAMATH_CALUDE_lila_tulips_l1405_140589

/-- Calculates the number of tulips after maintaining the ratio --/
def final_tulips (initial_orchids : ℕ) (added_orchids : ℕ) (tulip_ratio : ℕ) (orchid_ratio : ℕ) : ℕ :=
  let final_orchids := initial_orchids + added_orchids
  let groups := final_orchids / orchid_ratio
  tulip_ratio * groups

/-- Proves that Lila will have 21 tulips after maintaining the ratio --/
theorem lila_tulips : 
  final_tulips 16 12 3 4 = 21 := by
  sorry

#eval final_tulips 16 12 3 4

end NUMINAMATH_CALUDE_lila_tulips_l1405_140589


namespace NUMINAMATH_CALUDE_line_segment_intersection_range_l1405_140562

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x - y - 2 * a - 1 = 0

-- Define the endpoints of the line segment
def point_A : ℝ × ℝ := (-2, 3)
def point_B : ℝ × ℝ := (5, 2)

-- Define the intersection condition
def intersects_segment (a : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation a x y ∧
    ((x - point_A.1) / (point_B.1 - point_A.1) = (y - point_A.2) / (point_B.2 - point_A.2)) ∧
    0 ≤ (x - point_A.1) / (point_B.1 - point_A.1) ∧ (x - point_A.1) / (point_B.1 - point_A.1) ≤ 1

-- The theorem statement
theorem line_segment_intersection_range :
  ∀ a : ℝ, intersects_segment a ↔ (a ≤ -1 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_line_segment_intersection_range_l1405_140562


namespace NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_1300_l1405_140522

/-- The cost price of a toy given the selling conditions -/
theorem cost_price_of_toy (num_toys : ℕ) (selling_price : ℕ) (gain_toys : ℕ) : ℕ :=
  let cost_price := selling_price / (num_toys + gain_toys)
  cost_price

/-- Proof that the cost price of a toy is 1300 under given conditions -/
theorem cost_price_is_1300 : cost_price_of_toy 18 27300 3 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_toy_cost_price_is_1300_l1405_140522


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_l1405_140595

/-- Calculates the final price of a bicycle after two consecutive discounts. -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that a $200 bicycle, after a 40% discount followed by a 25% discount, costs $90. -/
theorem bicycle_price_after_discounts :
  final_price 200 0.4 0.25 = 90 := by
  sorry

#eval final_price 200 0.4 0.25

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_l1405_140595


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l1405_140570

/-- Represents the money redistribution problem among boys and girls -/
theorem money_redistribution_theorem 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (boy_initial : ℕ) 
  (girl_initial : ℕ) : 
  num_boys = 9 → 
  num_girls = 3 → 
  boy_initial = 12 → 
  girl_initial = 36 → 
  ∃ (boy_gives girl_gives final_amount : ℕ), 
    (∀ (b : ℕ), b < num_boys → 
      boy_initial - num_girls * boy_gives + num_girls * girl_gives = final_amount) ∧
    (∀ (g : ℕ), g < num_girls → 
      girl_initial - num_boys * girl_gives + num_boys * boy_gives = final_amount) := by
  sorry

end NUMINAMATH_CALUDE_money_redistribution_theorem_l1405_140570


namespace NUMINAMATH_CALUDE_mountain_height_l1405_140515

/-- The relative height of a mountain given temperature conditions -/
theorem mountain_height (temp_decrease_rate : ℝ) (summit_temp : ℝ) (base_temp : ℝ) :
  temp_decrease_rate = 0.7 →
  summit_temp = 14.1 →
  base_temp = 26 →
  (base_temp - summit_temp) / temp_decrease_rate * 100 = 1700 := by
  sorry

#check mountain_height

end NUMINAMATH_CALUDE_mountain_height_l1405_140515


namespace NUMINAMATH_CALUDE_cde_value_l1405_140524

/-- Represents the digits in the coding system -/
inductive Digit
| A | B | C | D | E | F

/-- Converts a Digit to its corresponding integer value -/
def digit_to_int : Digit → Nat
| Digit.A => 0
| Digit.B => 5
| Digit.C => 0
| Digit.D => 1
| Digit.E => 0
| Digit.F => 5

/-- Represents a number in the coding system -/
structure EncodedNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (ones : Digit)

/-- Converts an EncodedNumber to its base-10 value -/
def to_base_10 (n : EncodedNumber) : Nat :=
  6^2 * (digit_to_int n.hundreds) + 6 * (digit_to_int n.tens) + (digit_to_int n.ones)

/-- States that BCF, BCE, CAA are consecutive integers -/
axiom consecutive_encoding :
  ∃ (n : Nat),
    to_base_10 (EncodedNumber.mk Digit.B Digit.C Digit.F) = n ∧
    to_base_10 (EncodedNumber.mk Digit.B Digit.C Digit.E) = n + 1 ∧
    to_base_10 (EncodedNumber.mk Digit.C Digit.A Digit.A) = n + 2

theorem cde_value :
  to_base_10 (EncodedNumber.mk Digit.C Digit.D Digit.E) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cde_value_l1405_140524


namespace NUMINAMATH_CALUDE_stratified_sampling_young_employees_l1405_140557

theorem stratified_sampling_young_employees 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 200) 
  (h2 : young_employees = 120) 
  (h3 : sample_size = 25) :
  ↑sample_size * (↑young_employees / ↑total_employees) = 15 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_young_employees_l1405_140557


namespace NUMINAMATH_CALUDE_square_roots_and_cube_root_l1405_140547

theorem square_roots_and_cube_root (x a : ℝ) (hx : x > 0) : 
  ((2*a - 1)^2 = x ∧ (-a + 2)^2 = x ∧ 2*a - 1 ≠ -a + 2) →
  (a = -1 ∧ x = 9 ∧ (4*x + 9*a)^(1/3) = 3) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_and_cube_root_l1405_140547


namespace NUMINAMATH_CALUDE_intersection_point_l1405_140543

theorem intersection_point (x y : ℝ) : 
  y = 4 * x - 32 ∧ y = -6 * x + 8 → (x, y) = (4, -16) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l1405_140543


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1405_140561

theorem quadratic_inequality_always_positive (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4) * x - m + 7 > 0) ↔ m ∈ Set.Ioo (-2 : ℝ) 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1405_140561


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_four_l1405_140530

theorem sum_of_fractions_equals_four (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_eq : a / (b + c + d) = b / (a + c + d) ∧ 
          b / (a + c + d) = c / (a + b + d) ∧ 
          c / (a + b + d) = d / (a + b + c)) : 
  (a + b) / (c + d) + (b + c) / (a + d) + 
  (c + d) / (a + b) + (d + a) / (b + c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_four_l1405_140530


namespace NUMINAMATH_CALUDE_prob_first_class_correct_l1405_140555

/-- Represents the two types of items -/
inductive ItemClass
| First
| Second

/-- Represents the two trucks -/
inductive Truck
| A
| B

/-- The total number of items -/
def totalItems : Nat := 10

/-- The number of items in each truck -/
def truckItems : Truck → ItemClass → Nat
| Truck.A, ItemClass.First => 2
| Truck.A, ItemClass.Second => 2
| Truck.B, ItemClass.First => 4
| Truck.B, ItemClass.Second => 2

/-- The number of broken items per truck -/
def brokenItemsPerTruck : Nat := 1

/-- The number of remaining items after breakage -/
def remainingItems : Nat := totalItems - 2 * brokenItemsPerTruck

/-- The probability of selecting a first-class item from the remaining items -/
def probFirstClass : Rat := 29 / 48

theorem prob_first_class_correct :
  probFirstClass = 29 / 48 := by sorry

end NUMINAMATH_CALUDE_prob_first_class_correct_l1405_140555


namespace NUMINAMATH_CALUDE_three_digit_powers_of_two_l1405_140544

theorem three_digit_powers_of_two (n : ℕ) : 
  (∃ m : ℕ, 100 ≤ 2^m ∧ 2^m ≤ 999) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_two_l1405_140544


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l1405_140505

theorem shoe_price_calculation (initial_price : ℝ) (friday_increase : ℝ) (monday_decrease : ℝ) : 
  initial_price = 50 → 
  friday_increase = 0.20 → 
  monday_decrease = 0.15 → 
  initial_price * (1 + friday_increase) * (1 - monday_decrease) = 51 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l1405_140505


namespace NUMINAMATH_CALUDE_max_correct_answers_for_given_test_l1405_140500

/-- Represents a multiple choice test with scoring system -/
structure MCTest where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (test : MCTest) : ℕ :=
  sorry

/-- Theorem stating the maximum number of correct answers for the given test -/
theorem max_correct_answers_for_given_test :
  let test : MCTest := {
    total_questions := 60,
    correct_points := 3,
    incorrect_points := -2,
    total_score := 126
  }
  max_correct_answers test = 49 := by sorry

end NUMINAMATH_CALUDE_max_correct_answers_for_given_test_l1405_140500


namespace NUMINAMATH_CALUDE_b_initial_investment_l1405_140537

/-- Represents the business investment scenario --/
structure BusinessInvestment where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment
  initial_duration : ℕ  -- Initial duration in months
  a_withdrawal : ℕ  -- Amount A withdraws after initial duration
  b_addition : ℕ  -- Amount B adds after initial duration
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit_share : ℕ  -- A's share of the profit

/-- Calculates the total investment of A --/
def total_investment_a (bi : BusinessInvestment) : ℕ :=
  bi.a_initial * bi.initial_duration + (bi.a_initial - bi.a_withdrawal) * (12 - bi.initial_duration)

/-- Calculates the total investment of B --/
def total_investment_b (bi : BusinessInvestment) : ℕ :=
  bi.b_initial * bi.initial_duration + (bi.b_initial + bi.b_addition) * (12 - bi.initial_duration)

/-- Theorem stating that given the conditions, B's initial investment was 4000 Rs --/
theorem b_initial_investment (bi : BusinessInvestment) 
  (h1 : bi.a_initial = 3000)
  (h2 : bi.initial_duration = 8)
  (h3 : bi.a_withdrawal = 1000)
  (h4 : bi.b_addition = 1000)
  (h5 : bi.total_profit = 840)
  (h6 : bi.a_profit_share = 320)
  : bi.b_initial = 4000 := by
  sorry

end NUMINAMATH_CALUDE_b_initial_investment_l1405_140537


namespace NUMINAMATH_CALUDE_f_two_plus_f_five_l1405_140550

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_one : f 1 = 4

axiom f_z (z : ℝ) : z ≠ 1 → f z = 3 * z + 6

axiom f_sum (x y : ℝ) : ∃ (a b : ℝ), f (x + y) = f x + f y + a * x * y + b

theorem f_two_plus_f_five : f 2 + f 5 = 33 := by sorry

end NUMINAMATH_CALUDE_f_two_plus_f_five_l1405_140550


namespace NUMINAMATH_CALUDE_student_count_is_35_l1405_140588

/-- The number of different Roman numerals -/
def num_roman_numerals : ℕ := 7

/-- The number of sketches for each Roman numeral -/
def sketches_per_numeral : ℕ := 5

/-- The total number of students in the class -/
def num_students : ℕ := num_roman_numerals * sketches_per_numeral

theorem student_count_is_35 : num_students = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_count_is_35_l1405_140588


namespace NUMINAMATH_CALUDE_teacher_books_l1405_140565

theorem teacher_books (num_children : ℕ) (books_per_child : ℕ) (total_books : ℕ) : 
  num_children = 10 → books_per_child = 7 → total_books = 78 →
  total_books - (num_children * books_per_child) = 8 := by
sorry

end NUMINAMATH_CALUDE_teacher_books_l1405_140565


namespace NUMINAMATH_CALUDE_quadratic_sets_solution_l1405_140573

/-- Given sets A and B defined by quadratic equations, prove the values of a, b, and c -/
theorem quadratic_sets_solution :
  ∀ (a b c : ℝ),
  let A := {x : ℝ | x^2 + a*x + b = 0}
  let B := {x : ℝ | x^2 + c*x + 15 = 0}
  (A ∪ B = {3, 5} ∧ A ∩ B = {3}) →
  (a = -6 ∧ b = 9 ∧ c = -8) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_sets_solution_l1405_140573


namespace NUMINAMATH_CALUDE_pencils_per_child_l1405_140599

theorem pencils_per_child (total_children : ℕ) (total_pencils : ℕ) 
  (h1 : total_children = 9) 
  (h2 : total_pencils = 18) : 
  total_pencils / total_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_child_l1405_140599


namespace NUMINAMATH_CALUDE_book_cost_l1405_140533

theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) : 
  (7 * (cost_of_three / 3) : ℝ) = 105 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_l1405_140533


namespace NUMINAMATH_CALUDE_journey_time_reduction_l1405_140535

/-- Given a person's journey where increasing speed by 10% reduces time by x minutes, 
    prove the original journey time was 11x minutes. -/
theorem journey_time_reduction (d : ℝ) (s : ℝ) (x : ℝ) 
  (h1 : d > 0) (h2 : s > 0) (h3 : x > 0) 
  (h4 : d / s - d / (1.1 * s) = x) : 
  d / s = 11 * x := by
sorry

end NUMINAMATH_CALUDE_journey_time_reduction_l1405_140535


namespace NUMINAMATH_CALUDE_min_value_theorem_l1405_140559

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1 / x^2 + 1 / y + 1 / z = 6) :
  x^3 * y^2 * z^2 ≥ 1 / (8 * Real.sqrt 2) ∧
  ∃ x₀ y₀ z₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    1 / x₀^2 + 1 / y₀ + 1 / z₀ = 6 ∧
    x₀^3 * y₀^2 * z₀^2 = 1 / (8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1405_140559


namespace NUMINAMATH_CALUDE_sandwich_cost_l1405_140509

theorem sandwich_cost (num_sandwiches num_drinks drink_cost total_cost : ℕ) 
  (h1 : num_sandwiches = 3)
  (h2 : num_drinks = 2)
  (h3 : drink_cost = 4)
  (h4 : total_cost = 26) :
  ∃ (sandwich_cost : ℕ), 
    num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_cost ∧ 
    sandwich_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l1405_140509


namespace NUMINAMATH_CALUDE_problem_solution_l1405_140503

noncomputable def problem (a b c d m : ℝ) : Prop :=
  (a = -b) ∧ 
  (c * d = 1) ∧ 
  (m = 3 ∨ m = -3) → 
  (3 * c * d + (a + b) / (c * d) - m = 0 ∨ 3 * c * d + (a + b) / (c * d) - m = 6)

theorem problem_solution :
  ∀ a b c d m : ℝ, problem a b c d m := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1405_140503


namespace NUMINAMATH_CALUDE_amount_of_c_l1405_140540

/-- Given four people a, b, c, and d with monetary amounts, prove that c has 500 units of currency. -/
theorem amount_of_c (a b c d : ℕ) : 
  a + b + c + d = 1800 →
  a + c = 500 →
  b + c = 900 →
  a + d = 700 →
  a + b + d = 1300 →
  c = 500 := by
  sorry

end NUMINAMATH_CALUDE_amount_of_c_l1405_140540


namespace NUMINAMATH_CALUDE_integral_value_l1405_140576

theorem integral_value : ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - (x - 1)^2) - x^2) = π/4 - 1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_value_l1405_140576


namespace NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l1405_140539

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 1/2)
  (h2 : Real.cos x + Real.cos y = Real.sqrt 3 / 2) :
  Real.tan x + Real.tan y = - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l1405_140539


namespace NUMINAMATH_CALUDE_square_diagonal_perimeter_l1405_140549

theorem square_diagonal_perimeter (d : ℝ) (h : d = 20) :
  let side := d / Real.sqrt 2
  4 * side = 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_perimeter_l1405_140549


namespace NUMINAMATH_CALUDE_opposite_sign_square_root_l1405_140536

theorem opposite_sign_square_root (a b : ℝ) : 
  (|2*a - 4| + Real.sqrt (3*b + 12) = 0) → 
  Real.sqrt (2*a - 3*b) = 4 ∨ Real.sqrt (2*a - 3*b) = -4 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_square_root_l1405_140536


namespace NUMINAMATH_CALUDE_sandys_marks_per_correct_sum_l1405_140538

/-- Given Sandy's quiz results, calculate the marks for each correct sum -/
theorem sandys_marks_per_correct_sum 
  (total_sums : ℕ) 
  (correct_sums : ℕ) 
  (total_marks : ℤ) 
  (penalty_per_incorrect : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : correct_sums = 23) 
  (h3 : total_marks = 55) 
  (h4 : penalty_per_incorrect = 2) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_sums - 
    penalty_per_incorrect * (total_sums - correct_sums) = total_marks ∧ 
    marks_per_correct = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_marks_per_correct_sum_l1405_140538


namespace NUMINAMATH_CALUDE_estate_division_percentage_l1405_140594

/-- Represents the estate division problem --/
structure EstateDivision where
  amount₁ : ℝ  -- Amount received by the first person
  range : ℝ    -- Smallest possible range between highest and lowest amounts
  percentage : ℝ -- Percentage stipulation

/-- The estate division problem satisfies the given conditions --/
def valid_division (e : EstateDivision) : Prop :=
  e.amount₁ = 20000 ∧ 
  e.range = 10000 ∧ 
  0 < e.percentage ∧ 
  e.percentage < 100

/-- The theorem stating that the percentage stipulation is 25% --/
theorem estate_division_percentage (e : EstateDivision) 
  (h : valid_division e) : e.percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_estate_division_percentage_l1405_140594


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_l1405_140523

def dozen : ℕ := 12

def red_roses : ℕ := 2 * dozen
def white_roses : ℕ := 1 * dozen
def yellow_roses : ℕ := 2 * dozen

def red_price : ℚ := 6
def white_price : ℚ := 7
def yellow_price : ℚ := 5

def total_roses : ℕ := red_roses + white_roses + yellow_roses

def initial_cost : ℚ := 
  red_roses * red_price + white_roses * white_price + yellow_roses * yellow_price

def first_discount_rate : ℚ := 15 / 100
def second_discount_rate : ℚ := 10 / 100

theorem total_cost_after_discounts :
  total_roses > 30 ∧ total_roses > 50 →
  let cost_after_first_discount := initial_cost * (1 - first_discount_rate)
  let final_cost := cost_after_first_discount * (1 - second_discount_rate)
  final_cost = 266.22 := by sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_l1405_140523


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1405_140596

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : is_geometric a) 
  (h_product : a 2 * a 10 = 4)
  (h_sum_positive : a 2 + a 10 > 0) :
  a 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1405_140596


namespace NUMINAMATH_CALUDE_sum_areas_eighteen_disks_l1405_140507

/-- The sum of areas of 18 congruent disks arranged on a unit circle --/
theorem sum_areas_eighteen_disks : ℝ := by
  -- Define the number of disks
  let n : ℕ := 18

  -- Define the radius of the large circle
  let R : ℝ := 1

  -- Define the central angle for each disk
  let central_angle : ℝ := 2 * Real.pi / n

  -- Define the radius of each small disk
  let r : ℝ := Real.tan (central_angle / 2)

  -- Define the area of a single disk
  let single_disk_area : ℝ := Real.pi * r^2

  -- Define the sum of areas of all disks
  let total_area : ℝ := n * single_disk_area

  -- The theorem statement
  have : total_area = 18 * Real.pi * (Real.tan (Real.pi / 18))^2 := by sorry

  -- Return the result
  exact total_area


end NUMINAMATH_CALUDE_sum_areas_eighteen_disks_l1405_140507


namespace NUMINAMATH_CALUDE_equal_bisecting_diagonals_implies_rectangle_bisecting_diagonals_implies_parallelogram_rhombus_equal_diagonals_implies_square_l1405_140521

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def diagonals_perpendicular (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem equal_bisecting_diagonals_implies_rectangle 
  (q : Quadrilateral) 
  (h1 : has_equal_diagonals q) 
  (h2 : diagonals_bisect_each_other q) : 
  is_rectangle q := by sorry

-- Theorem 2
theorem bisecting_diagonals_implies_parallelogram 
  (q : Quadrilateral) 
  (h : diagonals_bisect_each_other q) : 
  is_parallelogram q := by sorry

-- Theorem 3
theorem rhombus_equal_diagonals_implies_square 
  (q : Quadrilateral) 
  (h1 : is_rhombus q) 
  (h2 : has_equal_diagonals q) : 
  is_square q := by sorry

end NUMINAMATH_CALUDE_equal_bisecting_diagonals_implies_rectangle_bisecting_diagonals_implies_parallelogram_rhombus_equal_diagonals_implies_square_l1405_140521


namespace NUMINAMATH_CALUDE_relay_team_permutations_l1405_140510

def team_size : ℕ := 4
def fixed_runner : String := "Lisa"
def fixed_lap : ℕ := 2

theorem relay_team_permutations :
  let remaining_runners := team_size - 1
  let free_laps := team_size - 1
  (remaining_runners.factorial : ℕ) = 6 := by sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l1405_140510


namespace NUMINAMATH_CALUDE_numerical_puzzle_solution_l1405_140598

theorem numerical_puzzle_solution :
  ∃! (A B C D E F : ℕ),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ F ≤ 9 ∧
    (10 * A + A) * (10 * A + B) = 1000 * C + 100 * D + 10 * E + F ∧
    (10 * C + C) * (100 * C + 10 * E + F) = 1000 * C + 100 * D + 10 * E + F ∧
    A = 4 ∧ B = 5 ∧ C = 1 ∧ D = 9 ∧ E = 8 ∧ F = 0 :=
by sorry

end NUMINAMATH_CALUDE_numerical_puzzle_solution_l1405_140598


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1405_140504

/-- 
Given an arithmetic sequence {a_n} with sum of first n terms S_n = n^2 - 3n,
prove that the general term formula is a_n = 2n - 4.
-/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = n^2 - 3*n) : 
  ∀ n, a n = 2*n - 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1405_140504


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_mean_of_fifteen_numbers_l1405_140526

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) 
                                  (set2_count : ℕ) (set2_mean : ℚ) : ℚ :=
  let total_count := set1_count + set2_count
  let combined_sum := set1_count * set1_mean + set2_count * set2_mean
  combined_sum / total_count

theorem mean_of_fifteen_numbers : 
  combined_mean_of_two_sets 7 15 8 22 = 281 / 15 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_mean_of_fifteen_numbers_l1405_140526


namespace NUMINAMATH_CALUDE_median_unchanged_after_removing_extremes_l1405_140553

theorem median_unchanged_after_removing_extremes 
  (x : Fin 10 → ℝ) 
  (h_ordered : ∀ i j : Fin 10, i ≤ j → x i ≤ x j) :
  (x 4 + x 5) / 2 = (x 5 + x 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_median_unchanged_after_removing_extremes_l1405_140553


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1405_140552

theorem quadratic_real_roots (k m : ℝ) (hm : m ≠ 0) :
  (∃ x : ℝ, x^2 + k*x + m = 0) ↔ m ≤ k^2/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1405_140552


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1405_140590

/-- Given a line with equation 5x - 3y = 9, prove that the slope of any parallel line is 5/3 -/
theorem parallel_line_slope (x y : ℝ) (h : 5 * x - 3 * y = 9) :
  ∃ (m : ℝ), m = 5 / 3 ∧ ∀ (x₁ y₁ : ℝ), (5 * x₁ - 3 * y₁ = 9) → (y₁ - y) = m * (x₁ - x) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1405_140590


namespace NUMINAMATH_CALUDE_gcd_problems_l1405_140532

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 153 119 = 17) := by
  sorry

end NUMINAMATH_CALUDE_gcd_problems_l1405_140532


namespace NUMINAMATH_CALUDE_square_count_theorem_l1405_140583

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Represents the configuration of two perpendicular families of lines -/
structure LineConfiguration :=
  (family1 : LineFamily)
  (family2 : LineFamily)

/-- Represents the set of intersection points -/
def IntersectionPoints (config : LineConfiguration) : ℕ :=
  config.family1.count * config.family2.count

/-- Counts the number of squares with sides parallel to the coordinate axes -/
def countParallelSquares (config : LineConfiguration) : ℕ :=
  sorry

/-- Counts the number of slanted squares -/
def countSlantedSquares (config : LineConfiguration) : ℕ :=
  sorry

/-- The main theorem -/
theorem square_count_theorem (config : LineConfiguration) 
  (h1 : config.family1.count = 15)
  (h2 : config.family2.count = 11)
  (h3 : IntersectionPoints config = 165) :
  countParallelSquares config + countSlantedSquares config ≥ 1986 :=
sorry

end NUMINAMATH_CALUDE_square_count_theorem_l1405_140583


namespace NUMINAMATH_CALUDE_fh_length_squared_value_l1405_140520

/-- Represents a parallelogram EFGH with specific properties -/
structure Parallelogram where
  /-- Area of the parallelogram -/
  area : ℝ
  /-- Length of JK, where J and K are projections of E and G onto FH -/
  jk_length : ℝ
  /-- Length of LM, where L and M are projections of F and H onto EG -/
  lm_length : ℝ
  /-- Assertion that EG is √2 times shorter than FH -/
  eg_fh_ratio : ℝ

/-- The square of the length of FH in the parallelogram -/
def fh_length_squared (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the square of FH's length given specific conditions -/
theorem fh_length_squared_value (p : Parallelogram) 
  (h_area : p.area = 20)
  (h_jk : p.jk_length = 7)
  (h_lm : p.lm_length = 9)
  (h_ratio : p.eg_fh_ratio = Real.sqrt 2) :
  fh_length_squared p = 27.625 := by sorry

end NUMINAMATH_CALUDE_fh_length_squared_value_l1405_140520


namespace NUMINAMATH_CALUDE_existence_of_n_for_k_l1405_140502

/-- f₂(n) is the number of divisors of n which are perfect squares -/
def f₂ (n : ℕ+) : ℕ := sorry

/-- f₃(n) is the number of divisors of n which are perfect cubes -/
def f₃ (n : ℕ+) : ℕ := sorry

/-- For all positive integers k, there exists a positive integer n such that f₂(n)/f₃(n) = k -/
theorem existence_of_n_for_k (k : ℕ+) : ∃ n : ℕ+, (f₂ n : ℚ) / (f₃ n : ℚ) = k := by sorry

end NUMINAMATH_CALUDE_existence_of_n_for_k_l1405_140502


namespace NUMINAMATH_CALUDE_vegetable_cost_l1405_140597

def initial_amount : ℤ := 100
def roast_cost : ℤ := 17
def remaining_amount : ℤ := 72

theorem vegetable_cost :
  initial_amount - roast_cost - remaining_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_cost_l1405_140597


namespace NUMINAMATH_CALUDE_constant_path_mapping_l1405_140525

/-- Given two segments AB and A'B' with their respective midpoints D and D', 
    prove that for any point P on AB with distance x from D, 
    and its associated point P' on A'B' with distance y from D', x + y = 6.5 -/
theorem constant_path_mapping (AB A'B' : ℝ) (D D' x y : ℝ) : 
  AB = 5 →
  A'B' = 8 →
  D = AB / 2 →
  D' = A'B' / 2 →
  x + y + D + D' = AB + A'B' →
  x + y = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_constant_path_mapping_l1405_140525


namespace NUMINAMATH_CALUDE_original_number_problem_l1405_140591

theorem original_number_problem (x : ℝ) :
  1 - 1/x = 5/2 → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l1405_140591


namespace NUMINAMATH_CALUDE_A_intersect_B_l1405_140574

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < 2 - x ∧ 2 - x < 3}

theorem A_intersect_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1405_140574


namespace NUMINAMATH_CALUDE_squares_in_3x3_lattice_l1405_140534

/-- A point in a 2D lattice -/
structure LatticePoint where
  x : ℕ
  y : ℕ

/-- A square lattice -/
structure SquareLattice where
  size : ℕ
  points : List LatticePoint

/-- A square formed by four lattice points -/
structure LatticeSquare where
  vertices : List LatticePoint

/-- Function to check if four points form a valid square in the lattice -/
def is_valid_square (l : SquareLattice) (s : LatticeSquare) : Prop :=
  sorry

/-- Function to count the number of valid squares in a lattice -/
def count_squares (l : SquareLattice) : ℕ :=
  sorry

/-- Theorem: The number of squares in a 3x3 square lattice is 5 -/
theorem squares_in_3x3_lattice :
  ∀ (l : SquareLattice), l.size = 3 → count_squares l = 5 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_3x3_lattice_l1405_140534


namespace NUMINAMATH_CALUDE_apples_remaining_l1405_140556

/-- The number of apples left after picking and eating -/
def applesLeft (mikeApples nancyApples keithApples : Float) : Float :=
  mikeApples + nancyApples - keithApples

theorem apples_remaining :
  applesLeft 7.0 3.0 6.0 = 4.0 := by
  sorry

#eval applesLeft 7.0 3.0 6.0

end NUMINAMATH_CALUDE_apples_remaining_l1405_140556
