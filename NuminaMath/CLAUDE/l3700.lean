import Mathlib

namespace NUMINAMATH_CALUDE_car_pricing_problem_l3700_370033

theorem car_pricing_problem (X : ℝ) (A : ℝ) : 
  X > 0 →
  0.8 * X * (1 + A / 100) = 1.2 * X →
  A = 50 := by
sorry

end NUMINAMATH_CALUDE_car_pricing_problem_l3700_370033


namespace NUMINAMATH_CALUDE_problem_statement_l3700_370025

theorem problem_statement (θ : ℝ) : 
  ((∀ x : ℝ, x^2 - 2*x*Real.sin θ + 1 ≥ 0) ∨ 
   (∀ α β : ℝ, Real.sin (α + β) ≤ Real.sin α + Real.sin β)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3700_370025


namespace NUMINAMATH_CALUDE_square_of_sqrt_plus_two_l3700_370053

theorem square_of_sqrt_plus_two (n : ℕ) (h : ∃ k : ℤ, k^2 = 1 + 12*n^2) :
  ∃ m : ℕ, (2 + 2*(Int.sqrt (1 + 12*n^2)))^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sqrt_plus_two_l3700_370053


namespace NUMINAMATH_CALUDE_circle_equation_l3700_370031

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point lies on a given line -/
def pointOnLine (p : Point) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y + c = 0

/-- Checks if a point lies on a given circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem to prove -/
theorem circle_equation : ∃ (c : Circle),
  (c.center.x + c.center.y - 2 = 0) ∧
  pointOnCircle ⟨1, -1⟩ c ∧
  pointOnCircle ⟨-1, 1⟩ c ∧
  c.center = ⟨1, 1⟩ ∧
  c.radius = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3700_370031


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3700_370002

def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3700_370002


namespace NUMINAMATH_CALUDE_simplify_nested_radicals_l3700_370095

theorem simplify_nested_radicals : 
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt 48))) = Real.sqrt 6 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_nested_radicals_l3700_370095


namespace NUMINAMATH_CALUDE_wood_stove_burn_rate_l3700_370091

/-- Wood stove burning rate problem -/
theorem wood_stove_burn_rate 
  (morning_duration : ℝ) 
  (afternoon_duration : ℝ)
  (morning_rate : ℝ) 
  (starting_wood : ℝ) 
  (ending_wood : ℝ) : 
  morning_duration = 4 →
  afternoon_duration = 4 →
  morning_rate = 2 →
  starting_wood = 30 →
  ending_wood = 3 →
  ∃ (afternoon_rate : ℝ), 
    afternoon_rate = (starting_wood - ending_wood - morning_duration * morning_rate) / afternoon_duration ∧ 
    afternoon_rate = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_wood_stove_burn_rate_l3700_370091


namespace NUMINAMATH_CALUDE_range_of_special_set_l3700_370059

def three_number_set (a b c : ℝ) : Prop :=
  a ≤ b ∧ b ≤ c

theorem range_of_special_set (a b c : ℝ) 
  (h_set : three_number_set a b c)
  (h_mean : (a + b + c) / 3 = 6)
  (h_median : b = 6)
  (h_min : a = 2) :
  c - a = 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_special_set_l3700_370059


namespace NUMINAMATH_CALUDE_dress_final_price_l3700_370099

/-- The final price of a dress after discounts and taxes -/
def final_price (d : ℝ) : ℝ :=
  let discount_price := d * (1 - 0.45)
  let staff_discount_price := discount_price * (1 - 0.40)
  let employee_month_price := staff_discount_price * (1 - 0.10)
  let local_tax_price := employee_month_price * (1 + 0.08)
  local_tax_price * (1 + 0.05)

/-- Theorem stating the final price of the dress -/
theorem dress_final_price (d : ℝ) :
  final_price d = 0.3549 * d := by
  sorry

end NUMINAMATH_CALUDE_dress_final_price_l3700_370099


namespace NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l3700_370007

/-- Represents the scoring system and conditions of the AMC 12 problem -/
structure AMC12Scoring where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int
  target_score : Int

/-- Calculates the score based on the number of correct answers -/
def calculate_score (s : AMC12Scoring) (correct_answers : Nat) : Int :=
  let incorrect_answers := s.attempted_problems - correct_answers
  let unanswered := s.total_problems - s.attempted_problems
  correct_answers * s.correct_points + 
  incorrect_answers * s.incorrect_points + 
  unanswered * s.unanswered_points

/-- Theorem stating the minimum number of correct answers needed to reach the target score -/
theorem min_correct_answers_for_target_score (s : AMC12Scoring) 
  (h1 : s.total_problems = 30)
  (h2 : s.attempted_problems = 26)
  (h3 : s.correct_points = 7)
  (h4 : s.incorrect_points = -1)
  (h5 : s.unanswered_points = 1)
  (h6 : s.target_score = 150) :
  ∃ n : Nat, (∀ m : Nat, m < n → calculate_score s m < s.target_score) ∧ 
             calculate_score s n ≥ s.target_score ∧
             n = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_target_score_l3700_370007


namespace NUMINAMATH_CALUDE_x_value_proof_l3700_370072

theorem x_value_proof (x : ℝ) (h : 9 / x^3 = x / 81) : x = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3700_370072


namespace NUMINAMATH_CALUDE_leah_daily_earnings_l3700_370070

/-- Represents Leah's earnings over a period of time -/
structure Earnings where
  total : ℕ  -- Total earnings in dollars
  weeks : ℕ  -- Number of weeks worked
  daily : ℕ  -- Daily earnings in dollars

/-- Calculates the number of days in a given number of weeks -/
def daysInWeeks (weeks : ℕ) : ℕ :=
  7 * weeks

/-- Theorem: Leah's daily earnings are 60 dollars -/
theorem leah_daily_earnings (e : Earnings) (h1 : e.total = 1680) (h2 : e.weeks = 4) :
  e.daily = 60 := by
  sorry

end NUMINAMATH_CALUDE_leah_daily_earnings_l3700_370070


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3700_370083

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b c : ℝ) (y : ℝ) : 
  -- The ellipse equation
  (∀ x y, x^2 / a^2 + y^2 / b^2 = 1) →
  -- F₁ and F₂ are foci of the ellipse
  (∃ F₁ F₂ : ℝ × ℝ, F₁.1 = -c ∧ F₁.2 = 0 ∧ F₂.1 = c ∧ F₂.2 = 0) →
  -- Point P is on the line x = -a
  (∃ P : ℝ × ℝ, P.1 = -a ∧ P.2 = y) →
  -- |PF₁| = |F₁F₂|
  ((a - c)^2 + y^2 = (2*c)^2) →
  -- ∠PF₁F₂ = 120°
  (y / (a - c) = Real.sqrt 3) →
  -- The eccentricity is 1/2
  c / a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3700_370083


namespace NUMINAMATH_CALUDE_arithmetic_mean_midpoint_l3700_370023

/-- Given two points on a number line, their arithmetic mean is located halfway between them -/
theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m - a = b - m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_midpoint_l3700_370023


namespace NUMINAMATH_CALUDE_female_average_score_l3700_370035

theorem female_average_score (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : male_average = 84)
  (h3 : male_count = 8)
  (h4 : female_count = 24) :
  let total_count := male_count + female_count
  let total_sum := total_average * total_count
  let male_sum := male_average * male_count
  let female_sum := total_sum - male_sum
  female_sum / female_count = 92 := by sorry

end NUMINAMATH_CALUDE_female_average_score_l3700_370035


namespace NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l3700_370088

theorem circle_radius_from_area_circumference_ratio 
  (M N : ℝ) (h : M / N = 25) : 
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 50 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_circumference_ratio_l3700_370088


namespace NUMINAMATH_CALUDE_ratio_seconds_minutes_l3700_370044

theorem ratio_seconds_minutes : ∃ x : ℝ, (12 / x = 6 / (4 * 60)) ∧ x = 480 := by
  sorry

end NUMINAMATH_CALUDE_ratio_seconds_minutes_l3700_370044


namespace NUMINAMATH_CALUDE_claps_per_second_is_seventeen_l3700_370060

/-- The number of claps achieved in one minute -/
def claps_per_minute : ℕ := 1020

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of claps per second -/
def claps_per_second : ℚ := claps_per_minute / seconds_per_minute

theorem claps_per_second_is_seventeen : 
  claps_per_second = 17 := by sorry

end NUMINAMATH_CALUDE_claps_per_second_is_seventeen_l3700_370060


namespace NUMINAMATH_CALUDE_textbook_cost_calculation_l3700_370047

theorem textbook_cost_calculation : 
  let sale_price : ℝ := 15 * (1 - 0.2)
  let sale_books : ℝ := 5
  let friend_books_cost : ℝ := 12 + 2 * 15
  let online_books_cost : ℝ := 45 * (1 - 0.1)
  let bookstore_books_cost : ℝ := 3 * 45
  let tax_rate : ℝ := 0.08
  
  sale_price * sale_books + friend_books_cost + online_books_cost + bookstore_books_cost + 
  ((sale_price * sale_books + friend_books_cost + online_books_cost + bookstore_books_cost) * tax_rate) = 299.70 := by
sorry


end NUMINAMATH_CALUDE_textbook_cost_calculation_l3700_370047


namespace NUMINAMATH_CALUDE_sum_x_y_value_l3700_370082

theorem sum_x_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : x + 3 * y = -1) : 
  x + y = 29 / 13 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_value_l3700_370082


namespace NUMINAMATH_CALUDE_manuscript_has_100_pages_l3700_370092

/-- Represents the pricing and revision structure of a typing service --/
structure TypingService where
  initial_cost : ℕ  -- Cost per page for initial typing
  revision_cost : ℕ  -- Cost per page for each revision

/-- Represents the manuscript details --/
structure Manuscript where
  total_pages : ℕ
  once_revised : ℕ
  twice_revised : ℕ

/-- Calculates the total cost for typing and revising a manuscript --/
def total_cost (service : TypingService) (manuscript : Manuscript) : ℕ :=
  service.initial_cost * manuscript.total_pages +
  service.revision_cost * manuscript.once_revised +
  2 * service.revision_cost * manuscript.twice_revised

/-- Theorem stating that given the conditions, the manuscript has 100 pages --/
theorem manuscript_has_100_pages (service : TypingService) (manuscript : Manuscript) :
  service.initial_cost = 5 →
  service.revision_cost = 4 →
  manuscript.once_revised = 30 →
  manuscript.twice_revised = 20 →
  total_cost service manuscript = 780 →
  manuscript.total_pages = 100 :=
by
  sorry


end NUMINAMATH_CALUDE_manuscript_has_100_pages_l3700_370092


namespace NUMINAMATH_CALUDE_sixty_percent_of_three_fifths_of_hundred_l3700_370005

theorem sixty_percent_of_three_fifths_of_hundred (n : ℝ) : n = 100 → (0.6 * (3/5 * n)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sixty_percent_of_three_fifths_of_hundred_l3700_370005


namespace NUMINAMATH_CALUDE_tommy_bike_ride_l3700_370090

theorem tommy_bike_ride (E : ℕ) : 
  (4 * (E + 2) : ℕ) * 4 = 80 → E = 3 := by sorry

end NUMINAMATH_CALUDE_tommy_bike_ride_l3700_370090


namespace NUMINAMATH_CALUDE_M_equals_N_l3700_370081

def M : Set ℝ := {x | ∃ k : ℤ, x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 5 * Real.pi / 6 + 2 * k * Real.pi}

def N : Set ℝ := {x | ∃ k : ℤ, x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = -7 * Real.pi / 6 + 2 * k * Real.pi}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l3700_370081


namespace NUMINAMATH_CALUDE_noras_oranges_l3700_370097

/-- The total number of oranges Nora picked from three trees -/
def total_oranges (tree1 tree2 tree3 : ℕ) : ℕ := tree1 + tree2 + tree3

/-- Theorem stating that the total number of oranges Nora picked is 260 -/
theorem noras_oranges : total_oranges 80 60 120 = 260 := by
  sorry

end NUMINAMATH_CALUDE_noras_oranges_l3700_370097


namespace NUMINAMATH_CALUDE_bus_variance_proof_l3700_370037

def bus_durations : List ℝ := [10, 11, 9, 9, 11]

theorem bus_variance_proof :
  let n : ℕ := bus_durations.length
  let mean : ℝ := (bus_durations.sum) / n
  let variance : ℝ := (bus_durations.map (fun x => (x - mean)^2)).sum / n
  (mean = 10 ∧ n = 5) → variance = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_bus_variance_proof_l3700_370037


namespace NUMINAMATH_CALUDE_conference_hall_tables_l3700_370062

/-- Represents the number of tables in the conference hall -/
def num_tables : ℕ := 16

/-- Represents the number of stools per table -/
def stools_per_table : ℕ := 8

/-- Represents the number of chairs per table -/
def chairs_per_table : ℕ := 4

/-- Represents the number of legs per stool -/
def legs_per_stool : ℕ := 3

/-- Represents the number of legs per chair -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs per table -/
def legs_per_table : ℕ := 4

/-- Represents the total number of legs for all furniture -/
def total_legs : ℕ := 704

theorem conference_hall_tables :
  num_tables * (stools_per_table * legs_per_stool + 
                chairs_per_table * legs_per_chair + 
                legs_per_table) = total_legs :=
by sorry

end NUMINAMATH_CALUDE_conference_hall_tables_l3700_370062


namespace NUMINAMATH_CALUDE_football_players_count_l3700_370043

theorem football_players_count (total_players cricket_players hockey_players softball_players : ℕ) 
  (h1 : total_players = 59)
  (h2 : cricket_players = 16)
  (h3 : hockey_players = 12)
  (h4 : softball_players = 13) :
  total_players - (cricket_players + hockey_players + softball_players) = 18 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l3700_370043


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_4_is_acute_l3700_370063

theorem triangle_with_angle_ratio_2_3_4_is_acute (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  b = (3/2) * a →          -- ratio between second and first angle
  c = 2 * a →              -- ratio between third and first angle
  a < 90 ∧ b < 90 ∧ c < 90 -- all angles are less than 90 degrees (acute triangle)
  := by sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_2_3_4_is_acute_l3700_370063


namespace NUMINAMATH_CALUDE_function_always_positive_l3700_370069

theorem function_always_positive
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (h : ∀ x : ℝ, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x : ℝ, f x > 0 :=
sorry

end NUMINAMATH_CALUDE_function_always_positive_l3700_370069


namespace NUMINAMATH_CALUDE_function_inequality_implies_lower_bound_on_a_l3700_370006

open Real

theorem function_inequality_implies_lower_bound_on_a :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → (log x - a ≤ x * exp x - x)) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_lower_bound_on_a_l3700_370006


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3700_370084

theorem quadratic_inequality_solution_range (m : ℝ) : 
  m > 0 ∧ 
  (∃ a b : ℤ, a ≠ b ∧ 
    (∀ x : ℝ, (2*x^2 - 2*m*x + m < 0) ↔ (a < x ∧ x < b)) ∧
    (∀ c : ℤ, (2*c^2 - 2*m*c + m < 0) → (c = a ∨ c = b)))
  → 
  8/3 < m ∧ m ≤ 18/5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3700_370084


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l3700_370027

theorem min_draws_for_even_product (cards : Finset ℕ) : 
  cards = Finset.range 14 →
  ∃ (n : ℕ), n = 8 ∧ 
    ∀ (subset : Finset ℕ), subset ⊆ cards → subset.card < n → 
      ∃ (x : ℕ), x ∈ subset ∧ Even x :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l3700_370027


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l3700_370038

theorem largest_lcm_with_18 (n : Fin 6 → ℕ) (h : n = ![3, 6, 9, 12, 15, 18]) :
  (Finset.range 6).sup (λ i => Nat.lcm 18 (n i)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l3700_370038


namespace NUMINAMATH_CALUDE_sphere_radius_increase_l3700_370016

/-- Theorem: If the surface area of a sphere increases by 21.00000000000002%,
    then the radius of the sphere increases by approximately 10%. -/
theorem sphere_radius_increase (r : ℝ) (h : r > 0) :
  let new_surface_area := 4 * Real.pi * r^2 * 1.2100000000000002
  let new_radius := r * (1 + 10/100)
  abs (new_surface_area - 4 * Real.pi * new_radius^2) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_increase_l3700_370016


namespace NUMINAMATH_CALUDE_preimage_of_three_one_l3700_370004

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2, 1) is the preimage of (3, 1) under f -/
theorem preimage_of_three_one :
  ∃! p : ℝ × ℝ, f p = (3, 1) ∧ p = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_three_one_l3700_370004


namespace NUMINAMATH_CALUDE_barbara_paper_count_l3700_370050

/-- The number of sheets in a bundle of colored paper -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a bunch of white paper -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a heap of scrap paper -/
def sheets_per_heap : ℕ := 20

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The total number of sheets Barbara removed from the chest of drawers -/
def total_sheets : ℕ := colored_bundles * sheets_per_bundle + 
                         white_bunches * sheets_per_bunch + 
                         scrap_heaps * sheets_per_heap

theorem barbara_paper_count : total_sheets = 114 := by
  sorry

end NUMINAMATH_CALUDE_barbara_paper_count_l3700_370050


namespace NUMINAMATH_CALUDE_amusing_numbers_l3700_370009

def is_amusing (x : Nat) : Prop :=
  (1000 ≤ x ∧ x < 10000) ∧
  ∃ y : Nat, (1000 ≤ y ∧ y < 10000) ∧
  (y % x = 0) ∧
  (∀ i : Fin 4,
    let x_digit := (x / (10 ^ i.val)) % 10
    let y_digit := (y / (10 ^ i.val)) % 10
    (x_digit = 0 ∧ y_digit = 1) ∨
    (x_digit = 9 ∧ y_digit = 8) ∨
    (x_digit ≠ 0 ∧ x_digit ≠ 9 ∧ (y_digit = x_digit - 1 ∨ y_digit = x_digit + 1)))

theorem amusing_numbers :
  is_amusing 1111 ∧ is_amusing 1091 ∧ is_amusing 1109 ∧ is_amusing 1089 :=
sorry

end NUMINAMATH_CALUDE_amusing_numbers_l3700_370009


namespace NUMINAMATH_CALUDE_expression_simplification_l3700_370046

theorem expression_simplification (x : ℝ) : 
  ((7*x + 3) - 3*x*2)*5 + (5 - 2/2)*(8*x - 5) = 37*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3700_370046


namespace NUMINAMATH_CALUDE_line_segment_lengths_l3700_370096

/-- Given points A, B, and C on a line, prove that if AB = 5 and AC = BC + 1, then AC = 3 and BC = 2 -/
theorem line_segment_lengths (A B C : ℝ) (h1 : |B - A| = 5) (h2 : |C - A| = |C - B| + 1) :
  |C - A| = 3 ∧ |C - B| = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_lengths_l3700_370096


namespace NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3700_370048

/-- Given a function f(x) = x^2 - ax - a with maximum value 1 on [0, 2], prove a = 1 -/
theorem max_value_implies_a_equals_one (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - a*x - a) ∧
   (∀ x ∈ Set.Icc 0 2, f x ≤ 1) ∧
   (∃ x ∈ Set.Icc 0 2, f x = 1)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_equals_one_l3700_370048


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3700_370074

/-- Given a quadratic equation x^2 - 6x + 4 = 0, 
    its equivalent form using the completing the square method is (x - 3)^2 = 5 -/
theorem quadratic_completing_square : 
  ∀ x : ℝ, x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3700_370074


namespace NUMINAMATH_CALUDE_largest_possible_a_l3700_370065

theorem largest_possible_a :
  ∀ (a b c d e : ℕ),
    a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    a < 3 * b →
    b < 4 * c →
    c < 5 * d →
    e = d - 10 →
    e < 105 →
    a ≤ 6824 ∧ ∃ (a' b' c' d' e' : ℕ),
      a' = 6824 ∧
      b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
      a' < 3 * b' ∧
      b' < 4 * c' ∧
      c' < 5 * d' ∧
      e' = d' - 10 ∧
      e' < 105 :=
by
  sorry


end NUMINAMATH_CALUDE_largest_possible_a_l3700_370065


namespace NUMINAMATH_CALUDE_farm_animals_l3700_370012

theorem farm_animals (cows chickens : ℕ) : 
  cows + chickens = 12 →
  4 * cows + 2 * chickens = 20 + 2 * (cows + chickens) →
  cows = 10 := by sorry

end NUMINAMATH_CALUDE_farm_animals_l3700_370012


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l3700_370058

/-- Two cyclists moving in opposite directions on a circular track meet at the starting point -/
theorem cyclists_meeting_time
  (circumference : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : circumference = 675)
  (h2 : speed1 = 7)
  (h3 : speed2 = 8) :
  circumference / (speed1 + speed2) = 45 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l3700_370058


namespace NUMINAMATH_CALUDE_legs_code_is_6189_l3700_370076

-- Define the type for our code
def Code := String

-- Define the mapping function
def digit_map (code : Code) (c : Char) : Nat :=
  match c with
  | 'N' => 0
  | 'E' => 1
  | 'W' => 2
  | 'C' => 3
  | 'H' => 4
  | 'A' => 5
  | 'L' => 6
  | 'G' => 8
  | 'S' => 9
  | _ => 0  -- Default case, should not occur in our problem

-- Define the function to convert a code word to a number
def code_to_number (code : Code) : Nat :=
  code.foldl (fun acc c => 10 * acc + digit_map code c) 0

-- The main theorem
theorem legs_code_is_6189 (code : Code) (h1 : code = "NEW CHALLENGES") :
  code_to_number "LEGS" = 6189 := by
  sorry


end NUMINAMATH_CALUDE_legs_code_is_6189_l3700_370076


namespace NUMINAMATH_CALUDE_base_k_conversion_l3700_370079

theorem base_k_conversion (k : ℕ) : k > 0 ∧ 1 * k^2 + 3 * k + 2 = 42 ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l3700_370079


namespace NUMINAMATH_CALUDE_worker_payment_problem_l3700_370028

/-- Proves that the total number of days is 60 given the conditions of the worker payment problem. -/
theorem worker_payment_problem (daily_pay : ℕ) (daily_deduction : ℕ) (total_payment : ℕ) (idle_days : ℕ) :
  daily_pay = 20 →
  daily_deduction = 3 →
  total_payment = 280 →
  idle_days = 40 →
  ∃ (work_days : ℕ), daily_pay * work_days - daily_deduction * idle_days = total_payment ∧
                      work_days + idle_days = 60 :=
by sorry

end NUMINAMATH_CALUDE_worker_payment_problem_l3700_370028


namespace NUMINAMATH_CALUDE_correct_equation_l3700_370030

theorem correct_equation (a b : ℝ) : -2 * b * a^2 + a^2 * b = -a^2 * b := by sorry

end NUMINAMATH_CALUDE_correct_equation_l3700_370030


namespace NUMINAMATH_CALUDE_sequence_properties_l3700_370042

theorem sequence_properties (a : Fin 4 → ℝ) 
  (h_decreasing : ∀ i j : Fin 4, i < j → a i > a j)
  (h_nonneg : a 3 ≥ 0)
  (h_closed : ∀ i j : Fin 4, i ≤ j → ∃ k : Fin 4, a i - a j = a k) :
  (∃ d : ℝ, ∀ i : Fin 4, i.val < 3 → a i.succ = a i - d) ∧ 
  (∃ i j : Fin 4, i < j ∧ (i.val + 1) * a i = (j.val + 1) * a j) ∧
  (∃ i : Fin 4, a i = 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3700_370042


namespace NUMINAMATH_CALUDE_evening_customers_is_40_l3700_370068

/-- Represents the revenue and customer data for a movie theater on a Friday night. -/
structure TheaterData where
  matineePrice : ℕ
  eveningPrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  openingNightCustomers : ℕ
  totalRevenue : ℕ

/-- Calculates the number of evening customers based on the theater data. -/
def eveningCustomers (data : TheaterData) : ℕ :=
  let totalCustomers := data.matineeCustomers + data.openingNightCustomers + (data.totalRevenue - 
    (data.matineePrice * data.matineeCustomers + 
     data.openingNightPrice * data.openingNightCustomers + 
     data.popcornPrice * (data.matineeCustomers + data.openingNightCustomers) / 2)) / data.eveningPrice
  (totalCustomers - data.matineeCustomers - data.openingNightCustomers)

/-- Theorem stating that the number of evening customers is 40 given the specific theater data. -/
theorem evening_customers_is_40 (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.openingNightPrice = 10)
  (h4 : data.popcornPrice = 10)
  (h5 : data.matineeCustomers = 32)
  (h6 : data.openingNightCustomers = 58)
  (h7 : data.totalRevenue = 1670) :
  eveningCustomers data = 40 := by
  sorry

end NUMINAMATH_CALUDE_evening_customers_is_40_l3700_370068


namespace NUMINAMATH_CALUDE_grid_coverage_possible_specific_case_101x101_l3700_370098

/-- Represents a square stamp with black cells -/
structure Stamp :=
  (size : ℕ)
  (black_cells : ℕ)

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Predicate to check if a grid can be covered by a stamp, leaving one corner uncovered -/
def can_cover (s : Stamp) (g : Grid) (num_stamps : ℕ) : Prop :=
  ∃ (N : ℕ), 
    g.size = 2*N + 1 ∧ 
    s.size = 2*N ∧ 
    s.black_cells = 4*N + 2 ∧ 
    num_stamps = 4*N

/-- Theorem stating that it's possible to cover a (2N+1) x (2N+1) grid with a 2N x 2N stamp -/
theorem grid_coverage_possible :
  ∀ (N : ℕ), N > 0 → 
    let s : Stamp := ⟨2*N, 4*N + 2⟩
    let g : Grid := ⟨2*N + 1⟩
    can_cover s g (4*N) :=
by
  sorry

/-- The specific case for the 101 x 101 grid with 102 black cells on the stamp -/
theorem specific_case_101x101 :
  let s : Stamp := ⟨100, 102⟩
  let g : Grid := ⟨101⟩
  can_cover s g 100 :=
by
  sorry

end NUMINAMATH_CALUDE_grid_coverage_possible_specific_case_101x101_l3700_370098


namespace NUMINAMATH_CALUDE_non_union_women_percentage_is_75_l3700_370021

/-- Represents the composition of employees in a company -/
structure CompanyEmployees where
  total : ℕ
  men : ℕ
  unionized : ℕ
  unionized_men : ℕ

/-- The percentage of non-union employees who are women -/
def non_union_women_percentage (c : CompanyEmployees) : ℚ :=
  let non_union := c.total - c.unionized
  let non_union_men := c.men - c.unionized_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union * 100

/-- Theorem stating the percentage of non-union women employees -/
theorem non_union_women_percentage_is_75 (c : CompanyEmployees) : 
  c.total > 0 →
  c.men = (52 * c.total) / 100 →
  c.unionized = (60 * c.total) / 100 →
  c.unionized_men = (70 * c.unionized) / 100 →
  non_union_women_percentage c = 75 := by
sorry

end NUMINAMATH_CALUDE_non_union_women_percentage_is_75_l3700_370021


namespace NUMINAMATH_CALUDE_range_of_m_l3700_370008

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -7 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 7}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 3*m - 2}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3700_370008


namespace NUMINAMATH_CALUDE_range_of_m_and_n_l3700_370094

-- Define sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n ≤ 0}

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem range_of_m_and_n (m n : ℝ) 
  (h1 : P ∈ A m) 
  (h2 : P ∉ B n) : 
  m > -1 ∧ n < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_and_n_l3700_370094


namespace NUMINAMATH_CALUDE_sod_square_size_l3700_370029

/-- Given a total area and number of squares, prove the side length of each square -/
theorem sod_square_size (total_area : ℝ) (num_squares : ℕ) 
  (h1 : total_area = 6000) 
  (h2 : num_squares = 1500) : 
  Real.sqrt (total_area / num_squares) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sod_square_size_l3700_370029


namespace NUMINAMATH_CALUDE_decimal_multiplication_l3700_370036

theorem decimal_multiplication (a b c : ℕ) (h : a * b = c) :
  (a : ℚ) / 100 * ((b : ℚ) / 100) = (c : ℚ) / 10000 :=
by
  sorry

-- Example usage
example : (268 : ℚ) / 100 * ((74 : ℚ) / 100) = (19832 : ℚ) / 10000 :=
decimal_multiplication 268 74 19832 rfl

end NUMINAMATH_CALUDE_decimal_multiplication_l3700_370036


namespace NUMINAMATH_CALUDE_james_new_friends_l3700_370040

def number_of_new_friends (initial_friends lost_friends final_friends : ℕ) : ℕ :=
  final_friends - (initial_friends - lost_friends)

theorem james_new_friends :
  number_of_new_friends 20 2 19 = 1 := by
  sorry

end NUMINAMATH_CALUDE_james_new_friends_l3700_370040


namespace NUMINAMATH_CALUDE_min_toothpicks_removal_l3700_370049

/-- Represents a triangular lattice structure made of toothpicks -/
structure TriangularLattice :=
  (toothpicks : ℕ)
  (triangles : ℕ)
  (horizontal_toothpicks : ℕ)

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (lattice : TriangularLattice) : ℕ :=
  lattice.horizontal_toothpicks

theorem min_toothpicks_removal (lattice : TriangularLattice) 
  (h1 : lattice.toothpicks = 40)
  (h2 : lattice.triangles > 40)
  (h3 : lattice.horizontal_toothpicks = 15) :
  min_toothpicks_to_remove lattice = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_removal_l3700_370049


namespace NUMINAMATH_CALUDE_composite_divides_factorial_l3700_370011

theorem composite_divides_factorial (k n : ℕ) (P_k : ℕ) : 
  k ≥ 14 →
  P_k < k →
  (∀ p, p < k ∧ Nat.Prime p → p ≤ P_k) →
  Nat.Prime P_k →
  P_k ≥ 3 * k / 4 →
  ¬Nat.Prime n →
  n > 2 * P_k →
  n ∣ Nat.factorial (n - k) :=
by sorry

end NUMINAMATH_CALUDE_composite_divides_factorial_l3700_370011


namespace NUMINAMATH_CALUDE_max_xy_value_l3700_370087

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2*x + 3*y = 6) :
  ∃ (max_val : ℝ), max_val = 3/2 ∧ ∀ (z : ℝ), x*y ≤ z → z ≤ max_val :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3700_370087


namespace NUMINAMATH_CALUDE_product_of_a_values_l3700_370015

theorem product_of_a_values (a : ℂ) (α β γ : ℂ) : 
  (∀ x : ℂ, x^3 - x^2 + a*x - 1 = 0 ↔ (x = α ∨ x = β ∨ x = γ)) →
  (α^3 + 1) * (β^3 + 1) * (γ^3 + 1) = 2018 →
  ∃ (a₁ a₂ a₃ : ℂ), (∀ x : ℂ, x^3 - 6*x + 2009 = 0 ↔ (x = a₁ ∨ x = a₂ ∨ x = a₃)) ∧ a₁ * a₂ * a₃ = 2009 :=
by sorry

end NUMINAMATH_CALUDE_product_of_a_values_l3700_370015


namespace NUMINAMATH_CALUDE_price_reduction_problem_l3700_370010

/-- The price reduction problem -/
theorem price_reduction_problem (reduced_price : ℝ) (extra_oil : ℝ) (total_money : ℝ) 
  (h1 : reduced_price = 15)
  (h2 : extra_oil = 6)
  (h3 : total_money = 900) :
  let original_price := total_money / (total_money / reduced_price - extra_oil)
  let percentage_reduction := (original_price - reduced_price) / original_price * 100
  ∃ (ε : ℝ), ε > 0 ∧ abs (percentage_reduction - 10) < ε :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_problem_l3700_370010


namespace NUMINAMATH_CALUDE_cos_F_value_l3700_370019

-- Define the triangle
def Triangle (DE DF : ℝ) : Prop :=
  DE > 0 ∧ DF > 0 ∧ DE < DF

-- Define right triangle
def RightTriangle (DE DF : ℝ) : Prop :=
  Triangle DE DF ∧ DE^2 + (DF^2 - DE^2) = DF^2

-- Theorem statement
theorem cos_F_value (DE DF : ℝ) :
  RightTriangle DE DF → DE = 8 → DF = 17 → Real.cos (Real.arccos (DE / DF)) = 8 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_F_value_l3700_370019


namespace NUMINAMATH_CALUDE_mr_grey_purchases_l3700_370013

/-- The cost of Mr. Grey's purchases -/
theorem mr_grey_purchases (polo_price : ℝ) : polo_price = 26 :=
  let necklace_price := 83
  let game_price := 90
  let rebate := 12
  let total_cost := 322
  let num_polos := 3
  let num_necklaces := 2
  have h : num_polos * polo_price + num_necklaces * necklace_price + game_price - rebate = total_cost :=
    by sorry
  sorry

end NUMINAMATH_CALUDE_mr_grey_purchases_l3700_370013


namespace NUMINAMATH_CALUDE_geometric_sequence_S3_lower_bound_l3700_370039

/-- A geometric sequence with positive terms where the second term is 1 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 2 = 1) ∧ (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)

/-- The sum of the first three terms of a sequence -/
def S3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

theorem geometric_sequence_S3_lower_bound
  (a : ℕ → ℝ) (h : GeometricSequence a) : S3 a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_S3_lower_bound_l3700_370039


namespace NUMINAMATH_CALUDE_cuboid_volume_problem_l3700_370045

theorem cuboid_volume_problem (x y : ℕ) : 
  (x > 0) → 
  (y > 0) → 
  (x < 4) → 
  (y < 15) → 
  (15 * 5 * 4 - x * 5 * y = 120) → 
  (x + y = 15) := by
sorry

end NUMINAMATH_CALUDE_cuboid_volume_problem_l3700_370045


namespace NUMINAMATH_CALUDE_min_a6_geometric_sequence_l3700_370078

theorem min_a6_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → a n > 0) →
  (∀ n : ℕ, 1 ≤ n ∧ n < 6 → a (n + 1) = (a n : ℚ) * q) →
  1 < q ∧ q < 2 →
  243 ≤ a 6 :=
by sorry

end NUMINAMATH_CALUDE_min_a6_geometric_sequence_l3700_370078


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3700_370080

def first_n_even_sum (n : ℕ) : ℕ := n * (n + 1)

def first_n_odd_sum (n : ℕ) : ℕ := n^2

theorem even_odd_sum_difference : 
  first_n_even_sum 1500 - first_n_odd_sum 1500 = 1500 :=
by sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3700_370080


namespace NUMINAMATH_CALUDE_family_structure_l3700_370054

/-- Represents a family with siblings -/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- Calculates the number of sisters a girl has in the family, excluding herself -/
def sisters_of_girl (f : Family) : ℕ := f.sisters - 1

/-- Calculates the number of brothers a girl has in the family -/
def brothers_of_girl (f : Family) : ℕ := f.brothers

/-- Calculates the ratio of sisters to total siblings for a girl in the family -/
def sister_ratio (f : Family) : ℚ :=
  (sisters_of_girl f : ℚ) / (f.sisters + f.brothers - 1 : ℚ)

/-- Theorem about the family structure and sibling relationships -/
theorem family_structure (f : Family) 
  (h1 : f.sisters = 5)
  (h2 : f.brothers = 5) :
  sisters_of_girl f = 4 ∧
  brothers_of_girl f = 4 ∧
  sisters_of_girl f + brothers_of_girl f = 8 ∧
  sister_ratio f = 1/2 := by
  sorry

#check family_structure

end NUMINAMATH_CALUDE_family_structure_l3700_370054


namespace NUMINAMATH_CALUDE_samuel_bought_two_dozen_l3700_370089

/-- The number of dozens of doughnuts Samuel bought -/
def samuel_dozens : ℕ := sorry

/-- The number of dozens of doughnuts Cathy bought -/
def cathy_dozens : ℕ := 3

/-- The total number of people sharing the doughnuts -/
def total_people : ℕ := 10

/-- The number of doughnuts each person received -/
def doughnuts_per_person : ℕ := 6

/-- Theorem stating that Samuel bought 2 dozen doughnuts -/
theorem samuel_bought_two_dozen : samuel_dozens = 2 := by
  sorry

end NUMINAMATH_CALUDE_samuel_bought_two_dozen_l3700_370089


namespace NUMINAMATH_CALUDE_geometric_increasing_iff_second_greater_first_l3700_370067

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: For a geometric sequence with positive first term,
    the second term being greater than the first is equivalent to
    the sequence being increasing -/
theorem geometric_increasing_iff_second_greater_first
    (a : ℕ → ℝ) (h_geom : GeometricSequence a) (h_pos : a 1 > 0) :
    a 1 < a 2 ↔ IncreasingSequence a :=
  sorry

end NUMINAMATH_CALUDE_geometric_increasing_iff_second_greater_first_l3700_370067


namespace NUMINAMATH_CALUDE_basketball_players_l3700_370034

theorem basketball_players (cricket : ℕ) (both : ℕ) (total : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 3)
  (h3 : total = 12) :
  ∃ basketball : ℕ, basketball = total - cricket + both :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_players_l3700_370034


namespace NUMINAMATH_CALUDE_exam_students_count_l3700_370066

theorem exam_students_count : 
  ∀ N : ℕ,
  (N : ℝ) * 80 = 160 + (N - 8 : ℝ) * 90 →
  N = 56 :=
by
  sorry

#check exam_students_count

end NUMINAMATH_CALUDE_exam_students_count_l3700_370066


namespace NUMINAMATH_CALUDE_total_throw_distance_l3700_370026

/-- Proves the total distance thrown over two days is 1600 yards. -/
theorem total_throw_distance (T : ℝ) : 
  let throw_distance_T := 20
  let throw_distance_80 := 2 * throw_distance_T
  let saturday_throws := 20
  let sunday_throws := 30
  let saturday_distance := saturday_throws * throw_distance_T
  let sunday_distance := sunday_throws * throw_distance_80
  saturday_distance + sunday_distance = 1600 := by sorry

end NUMINAMATH_CALUDE_total_throw_distance_l3700_370026


namespace NUMINAMATH_CALUDE_largest_n_binomial_equality_l3700_370003

theorem largest_n_binomial_equality : ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_equality_l3700_370003


namespace NUMINAMATH_CALUDE_ball_probability_relationship_l3700_370057

/-- Given a pocket with 7 balls, including 3 white and 4 black balls, 
    if x white balls and y black balls are added, and the probability 
    of drawing a white ball becomes 1/4, then y = 3x + 5 -/
theorem ball_probability_relationship (x y : ℤ) : 
  (((3 : ℚ) + x) / ((7 : ℚ) + x + y) = (1 : ℚ) / 4) → y = 3 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_relationship_l3700_370057


namespace NUMINAMATH_CALUDE_committee_combinations_l3700_370064

theorem committee_combinations : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_committee_combinations_l3700_370064


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3700_370075

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 18) :
  (Nat.gcd (12 * a) (20 * b) ≥ 72) ∧ ∃ (a₀ b₀ : ℕ+), Nat.gcd a₀ b₀ = 18 ∧ Nat.gcd (12 * a₀) (20 * b₀) = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3700_370075


namespace NUMINAMATH_CALUDE_num_different_results_is_1024_l3700_370001

/-- The expression as a list of integers -/
def expression : List Int := [1, -2, -4, -8, -16, -32, -64, -128, -256, -512, -1024]

/-- The number of terms in the expression that can have their sign changed -/
def num_changeable_terms : Nat := expression.length - 1

/-- The number of different results obtainable by placing parentheses in the expression -/
def num_different_results : Nat := 2^num_changeable_terms

theorem num_different_results_is_1024 : num_different_results = 1024 := by
  sorry

end NUMINAMATH_CALUDE_num_different_results_is_1024_l3700_370001


namespace NUMINAMATH_CALUDE_not_diff_of_squares_2022_l3700_370052

theorem not_diff_of_squares_2022 : ∀ a b : ℤ, a^2 - b^2 ≠ 2022 := by
  sorry

end NUMINAMATH_CALUDE_not_diff_of_squares_2022_l3700_370052


namespace NUMINAMATH_CALUDE_total_apples_in_pile_l3700_370061

def initial_apples : ℕ := 8
def added_apples : ℕ := 5
def package_size : ℕ := 11

theorem total_apples_in_pile :
  initial_apples + added_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_in_pile_l3700_370061


namespace NUMINAMATH_CALUDE_seashells_remaining_l3700_370018

theorem seashells_remaining (initial_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : given_seashells = 43) : 
  initial_seashells - given_seashells = 27 := by
  sorry

end NUMINAMATH_CALUDE_seashells_remaining_l3700_370018


namespace NUMINAMATH_CALUDE_prob_less_than_3_l3700_370020

/-- A fair cubic die with faces labeled 1 to 6 -/
structure FairDie :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The event of rolling a number less than 3 -/
def LessThan3 (d : FairDie) : Finset Nat :=
  d.faces.filter (λ x => x < 3)

/-- The probability of an event for a fair die -/
def Probability (d : FairDie) (event : Finset Nat) : Rat :=
  (event.card : Rat) / (d.faces.card : Rat)

theorem prob_less_than_3 (d : FairDie) :
  Probability d (LessThan3 d) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_3_l3700_370020


namespace NUMINAMATH_CALUDE_child_b_share_l3700_370071

theorem child_b_share (total_money : ℕ) (ratio_a ratio_b ratio_c : ℕ) : 
  total_money = 900 → 
  ratio_a = 2 → 
  ratio_b = 3 → 
  ratio_c = 4 → 
  (ratio_b * total_money) / (ratio_a + ratio_b + ratio_c) = 300 := by
  sorry

end NUMINAMATH_CALUDE_child_b_share_l3700_370071


namespace NUMINAMATH_CALUDE_third_sample_is_43_l3700_370041

/-- Systematic sampling function -/
def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * (total / sample_size)

/-- Theorem for the specific problem -/
theorem third_sample_is_43 
  (total : ℕ) (sample_size : ℕ) (start : ℕ) 
  (h1 : total = 900) 
  (h2 : sample_size = 50) 
  (h3 : start = 7) :
  systematic_sample total sample_size start 3 = 43 := by
  sorry

#eval systematic_sample 900 50 7 3

end NUMINAMATH_CALUDE_third_sample_is_43_l3700_370041


namespace NUMINAMATH_CALUDE_perfect_apples_l3700_370032

/-- Given a batch of apples, calculate the number of perfect apples -/
theorem perfect_apples (total : ℕ) (too_small : ℚ) (not_ripe : ℚ) 
  (h1 : total = 30) 
  (h2 : too_small = 1/6) 
  (h3 : not_ripe = 1/3) : 
  ↑total * (1 - (too_small + not_ripe)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_perfect_apples_l3700_370032


namespace NUMINAMATH_CALUDE_model_x_completion_time_l3700_370014

/-- The time (in minutes) it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 30

/-- The number of Model X computers used -/
def num_model_x : ℕ := 20

/-- The time (in minutes) it takes to complete the task when using equal numbers of both models -/
def combined_time : ℝ := 1

/-- The time (in minutes) it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := 60

theorem model_x_completion_time :
  (num_model_x : ℝ) * (1 / model_x_time + 1 / model_y_time) = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_model_x_completion_time_l3700_370014


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3700_370086

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α - π / 3) = 2 / 3) : 
  Real.sin α = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3700_370086


namespace NUMINAMATH_CALUDE_probability_white_or_red_l3700_370073

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def red_balls : ℕ := 5

def total_balls : ℕ := white_balls + black_balls + red_balls

def favorable_outcomes : ℕ := white_balls + red_balls

theorem probability_white_or_red :
  (favorable_outcomes : ℚ) / total_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_or_red_l3700_370073


namespace NUMINAMATH_CALUDE_remaining_three_average_l3700_370024

theorem remaining_three_average (total : ℕ) (all_avg first_four_avg next_three_avg following_two_avg : ℚ) :
  total = 12 →
  all_avg = 6.30 →
  first_four_avg = 5.60 →
  next_three_avg = 4.90 →
  following_two_avg = 7.25 →
  (total * all_avg - (4 * first_four_avg + 3 * next_three_avg + 2 * following_two_avg)) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remaining_three_average_l3700_370024


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l3700_370056

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem 1: When a = 0, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_zero :
  A 0 ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ⊆ B if and only if 1 ≤ a ≤ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ 1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l3700_370056


namespace NUMINAMATH_CALUDE_lemon_sequences_l3700_370022

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of times the class meets in a week -/
def meetings_per_week : ℕ := 5

/-- The number of possible sequences of lemon recipients in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating the number of possible sequences of lemon recipients -/
theorem lemon_sequences :
  num_sequences = 759375 :=
by sorry

end NUMINAMATH_CALUDE_lemon_sequences_l3700_370022


namespace NUMINAMATH_CALUDE_exam_score_proof_l3700_370093

theorem exam_score_proof (mean : ℝ) (low_score : ℝ) (std_dev_below : ℝ) (std_dev_above : ℝ) :
  mean = 88.8 →
  low_score = 86 →
  std_dev_below = 7 →
  std_dev_above = 3 →
  low_score = mean - std_dev_below * ((mean - low_score) / std_dev_below) →
  mean + std_dev_above * ((mean - low_score) / std_dev_below) = 90 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_proof_l3700_370093


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l3700_370000

theorem long_furred_brown_dogs (total : ℕ) (long_furred : ℕ) (brown : ℕ) (neither : ℕ) :
  total = 45 →
  long_furred = 29 →
  brown = 17 →
  neither = 8 →
  long_furred + brown - (total - neither) = 9 :=
by sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l3700_370000


namespace NUMINAMATH_CALUDE_tan_ratio_problem_l3700_370085

theorem tan_ratio_problem (x : ℝ) (h : Real.tan (x + π/4) = 2) : 
  Real.tan x / Real.tan (2*x) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_problem_l3700_370085


namespace NUMINAMATH_CALUDE_half_and_neg_third_are_like_terms_l3700_370017

/-- Definition of like terms -/
def are_like_terms (a b : ℚ) : Prop :=
  (∀ x, a.num * x = 0 ↔ b.num * x = 0) ∧ (a ≠ 0 ∨ b ≠ 0)

/-- Theorem: 1/2 and -1/3 are like terms -/
theorem half_and_neg_third_are_like_terms :
  are_like_terms (1/2 : ℚ) (-1/3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_half_and_neg_third_are_like_terms_l3700_370017


namespace NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l3700_370077

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (d : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 0.75̅ -/
def zeroPointSevenFive : RepeatingDecimal :=
  { integerPart := 0, repeatingPart := 75 }

/-- The repeating decimal 2.25̅ -/
def twoPointTwoFive : RepeatingDecimal :=
  { integerPart := 2, repeatingPart := 25 }

/-- Theorem stating that the ratio of 0.75̅ to 2.25̅ is equal to 2475/7329 -/
theorem ratio_of_repeating_decimals :
  (toRational zeroPointSevenFive) / (toRational twoPointTwoFive) = 2475 / 7329 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_repeating_decimals_l3700_370077


namespace NUMINAMATH_CALUDE_rectangle_z_value_l3700_370055

-- Define the rectangle
def rectangle (z : ℝ) : Set (ℝ × ℝ) :=
  {(-2, z), (6, z), (-2, 4), (6, 4)}

-- Define the area of the rectangle
def area (z : ℝ) : ℝ :=
  (6 - (-2)) * (z - 4)

-- Theorem statement
theorem rectangle_z_value (z : ℝ) :
  z > 0 ∧ area z = 64 → z = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_z_value_l3700_370055


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_l3700_370051

/-- Given points A, B, and C in a 2D plane, prove that if AB is perpendicular to BC,
    then the x-coordinate of C is 8/3. -/
theorem perpendicular_vectors_imply_m (A B C : ℝ × ℝ) :
  A = (-1, 3) →
  B = (2, 1) →
  C.2 = 2 →
  (B.1 - A.1, B.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 0 →
  C.1 = 8/3 := by
  sorry

#check perpendicular_vectors_imply_m

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_l3700_370051
