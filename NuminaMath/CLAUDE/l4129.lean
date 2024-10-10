import Mathlib

namespace concentric_squares_ratio_l4129_412900

/-- Given two concentric squares ABCD (outer) and EFGH (inner) with side lengths a and b
    respectively, if the area of the shaded region between them is p% of the area of ABCD,
    then a/b = 1/sqrt(1-p/100). -/
theorem concentric_squares_ratio (a b p : ℝ) (ha : a > 0) (hb : b > 0) (hp : 0 < p ∧ p < 100) :
  (a^2 - b^2) / a^2 = p / 100 → a / b = 1 / Real.sqrt (1 - p / 100) := by
  sorry

end concentric_squares_ratio_l4129_412900


namespace haunted_mansion_scenarios_l4129_412983

theorem haunted_mansion_scenarios (windows : ℕ) (rooms : ℕ) : windows = 8 → rooms = 3 → windows * (windows - 1) * rooms = 168 := by
  sorry

end haunted_mansion_scenarios_l4129_412983


namespace equation_solutions_l4129_412965

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (-3 + Real.sqrt 13) / 2 ∧ x2 = (-3 - Real.sqrt 13) / 2 ∧
    x1^2 + 3*x1 - 1 = 0 ∧ x2^2 + 3*x2 - 1 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -2 ∧ x2 = -1 ∧
    (x1 + 2)^2 = x1 + 2 ∧ (x2 + 2)^2 = x2 + 2) :=
by sorry

end equation_solutions_l4129_412965


namespace roi_difference_emma_briana_l4129_412979

/-- Calculates the difference in return-on-investment between two investors after a given time period. -/
def roi_difference (emma_investment briana_investment : ℝ) 
                   (emma_yield_rate briana_yield_rate : ℝ) 
                   (years : ℕ) : ℝ :=
  (briana_investment * briana_yield_rate * years) - (emma_investment * emma_yield_rate * years)

/-- Theorem stating the difference in return-on-investment between Briana and Emma after 2 years. -/
theorem roi_difference_emma_briana : 
  roi_difference 300 500 0.15 0.10 2 = 10 := by
  sorry

end roi_difference_emma_briana_l4129_412979


namespace x_less_than_two_necessary_not_sufficient_l4129_412930

theorem x_less_than_two_necessary_not_sufficient :
  (∀ x : ℝ, |x - 1| < 1 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ |x - 1| ≥ 1) :=
by sorry

end x_less_than_two_necessary_not_sufficient_l4129_412930


namespace minimize_y_l4129_412907

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

/-- The theorem stating that (2a + 3b) / 5 minimizes y -/
theorem minimize_y (a b : ℝ) :
  let x_min := (2 * a + 3 * b) / 5
  ∀ x, y x_min a b ≤ y x a b :=
by sorry

end minimize_y_l4129_412907


namespace arithmetic_sequence_common_difference_l4129_412911

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_30 : a 30 = 100)
  (h_100 : a 100 = 30) :
  ∃ d : ℝ, d = -1 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end arithmetic_sequence_common_difference_l4129_412911


namespace julio_twice_james_age_l4129_412912

/-- The age difference between Julio and James -/
def age_difference : ℕ := 36 - 11

/-- The number of years until Julio's age is twice James' age -/
def years_until_double : ℕ := 14

theorem julio_twice_james_age :
  36 + years_until_double = 2 * (11 + years_until_double) :=
sorry

end julio_twice_james_age_l4129_412912


namespace basketball_team_points_l4129_412928

theorem basketball_team_points (x : ℚ) (z : ℕ) : 
  (1 / 3 : ℚ) * x + (3 / 8 : ℚ) * x + 18 + z = x →
  z ≤ 27 →
  z = 21 := by
  sorry

end basketball_team_points_l4129_412928


namespace car_capacities_and_rental_plans_l4129_412944

/-- The capacity of a type A car in tons -/
def capacity_A : ℕ := 3

/-- The capacity of a type B car in tons -/
def capacity_B : ℕ := 4

/-- The total weight of goods to be transported -/
def total_weight : ℕ := 31

/-- A rental plan is a pair of natural numbers (a, b) where a is the number of type A cars and b is the number of type B cars -/
def RentalPlan := ℕ × ℕ

/-- The set of all valid rental plans -/
def valid_rental_plans : Set RentalPlan :=
  {plan | plan.1 * capacity_A + plan.2 * capacity_B = total_weight}

theorem car_capacities_and_rental_plans :
  (2 * capacity_A + capacity_B = 10) ∧
  (capacity_A + 2 * capacity_B = 11) ∧
  (valid_rental_plans = {(1, 7), (5, 4), (9, 1)}) := by
  sorry


end car_capacities_and_rental_plans_l4129_412944


namespace four_weighings_sufficient_l4129_412963

/-- Represents the weight of a coin in grams -/
inductive CoinWeight
  | One
  | Two
  | Three
  | Four

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing action -/
def Weighing := (List CoinWeight) → (List CoinWeight) → WeighingResult

/-- The set of four coins with weights 1, 2, 3, and 4 grams -/
def CoinSet : Set CoinWeight := {CoinWeight.One, CoinWeight.Two, CoinWeight.Three, CoinWeight.Four}

/-- A strategy is a sequence of weighings -/
def Strategy := List Weighing

/-- Checks if a strategy can identify all coins uniquely -/
def canIdentifyAllCoins (s : Strategy) (coins : Set CoinWeight) : Prop := sorry

/-- Main theorem: There exists a strategy with at most 4 weighings that can identify all coins -/
theorem four_weighings_sufficient :
  ∃ (s : Strategy), s.length ≤ 4 ∧ canIdentifyAllCoins s CoinSet := by sorry

end four_weighings_sufficient_l4129_412963


namespace opposite_gender_selections_l4129_412903

def society_size : ℕ := 24
def male_count : ℕ := 14
def female_count : ℕ := 10

theorem opposite_gender_selections :
  (male_count * female_count) + (female_count * male_count) = 280 := by
  sorry

end opposite_gender_selections_l4129_412903


namespace happy_point_properties_l4129_412975

/-- A point (m, n+2) is a "happy point" if 2m = 8 + n --/
def is_happy_point (m n : ℝ) : Prop := 2 * m = 8 + n

/-- The point B(4,5) --/
def B : ℝ × ℝ := (4, 5)

/-- The point M(a, a-1) --/
def M (a : ℝ) : ℝ × ℝ := (a, a - 1)

theorem happy_point_properties :
  (¬ is_happy_point B.1 (B.2 - 2)) ∧
  (∀ a : ℝ, is_happy_point (M a).1 ((M a).2 - 2) → a > 0 ∧ a - 1 > 0) :=
by sorry

end happy_point_properties_l4129_412975


namespace triangle_CSE_is_equilateral_l4129_412969

-- Define the circle k
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the chord AB
def Chord (k : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ k ∧ B ∈ k

-- Define the perpendicular bisector
def PerpendicularBisector (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.1 - P.1)^2 + (X.2 - P.2)^2 = (X.1 - Q.1)^2 + (X.2 - Q.2)^2}

-- Define the line through two points
def LineThroughPoints (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.2 - P.2) * (Q.1 - P.1) = (X.1 - P.1) * (Q.2 - P.2)}

theorem triangle_CSE_is_equilateral
  (k : Set (ℝ × ℝ))
  (r : ℝ)
  (A B S C D E : ℝ × ℝ)
  (h1 : k = Circle (0, 0) r)
  (h2 : Chord k A B)
  (h3 : S ∈ LineThroughPoints A B)
  (h4 : (S.1 - A.1)^2 + (S.2 - A.2)^2 = r^2)
  (h5 : (B.1 - A.1)^2 + (B.2 - A.2)^2 > r^2)
  (h6 : C ∈ k ∧ C ∈ PerpendicularBisector B S)
  (h7 : D ∈ k ∧ D ∈ PerpendicularBisector B S)
  (h8 : E ∈ k ∧ E ∈ LineThroughPoints D S) :
  (C.1 - S.1)^2 + (C.2 - S.2)^2 = (C.1 - E.1)^2 + (C.2 - E.2)^2 ∧
  (C.1 - S.1)^2 + (C.2 - S.2)^2 = (E.1 - S.1)^2 + (E.2 - S.2)^2 :=
sorry

end triangle_CSE_is_equilateral_l4129_412969


namespace morse_code_symbols_l4129_412958

/-- The number of possible symbols (dot, dash, space) -/
def num_symbols : ℕ := 3

/-- The maximum length of a sequence -/
def max_length : ℕ := 3

/-- Calculates the number of distinct sequences for a given length -/
def sequences_of_length (n : ℕ) : ℕ := num_symbols ^ n

/-- The total number of distinct symbols that can be represented -/
def total_distinct_symbols : ℕ :=
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3)

/-- Theorem: The total number of distinct symbols that can be represented is 39 -/
theorem morse_code_symbols : total_distinct_symbols = 39 := by
  sorry

end morse_code_symbols_l4129_412958


namespace equation_solution_l4129_412934

theorem equation_solution (a b : ℝ) :
  (∀ x y : ℝ, y = a + b / x) →
  (3 = a + b / 2) →
  (-1 = a + b / (-4)) →
  a + b = 4 := by
sorry

end equation_solution_l4129_412934


namespace quadratic_function_unique_l4129_412929

/-- A quadratic function is a function of the form f(x) = ax² + bx + c, where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (h_quad : QuadraticFunction f)
  (h_f_2 : f 2 = -1)
  (h_f_neg1 : f (-1) = -1)
  (h_max : ∃ x_max, ∀ x, f x ≤ f x_max ∧ f x_max = 8) :
  ∀ x, f x = -4 * x^2 + 4 * x + 7 := by
  sorry

end quadratic_function_unique_l4129_412929


namespace speed_conversion_l4129_412923

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The initial speed in meters per second -/
def initial_speed : ℝ := 5

/-- Theorem: 5 mps is equal to 18 kmph -/
theorem speed_conversion : initial_speed * mps_to_kmph = 18 := by
  sorry

end speed_conversion_l4129_412923


namespace parallel_perpendicular_implication_parallel_contained_implication_l4129_412908

structure GeometrySpace where
  Line : Type
  Plane : Type
  parallel_lines : Line → Line → Prop
  parallel_plane_line : Plane → Line → Prop
  parallel_planes : Plane → Plane → Prop
  perpendicular_plane_line : Plane → Line → Prop
  line_in_plane : Line → Plane → Prop
  line_not_in_plane : Line → Plane → Prop

variable (G : GeometrySpace)

theorem parallel_perpendicular_implication
  (m n : G.Line) (α β : G.Plane)
  (h1 : G.parallel_lines m n)
  (h2 : G.perpendicular_plane_line α m)
  (h3 : G.perpendicular_plane_line β n) :
  G.parallel_planes α β :=
sorry

theorem parallel_contained_implication
  (m n : G.Line) (α β : G.Plane)
  (h1 : G.parallel_lines m n)
  (h2 : G.line_in_plane n α)
  (h3 : G.parallel_planes α β)
  (h4 : G.line_not_in_plane m β) :
  G.parallel_plane_line β m :=
sorry

end parallel_perpendicular_implication_parallel_contained_implication_l4129_412908


namespace equation_holds_iff_specific_pairs_l4129_412984

def S (r : ℕ) (x y z : ℝ) : ℝ := x^r + y^r + z^r

theorem equation_holds_iff_specific_pairs (m n : ℕ) (x y z : ℝ) 
  (h : x + y + z = 0) :
  (∀ (x y z : ℝ), x + y + z = 0 → 
    S (m + n) x y z / (m + n : ℝ) = (S m x y z / m) * (S n x y z / n)) ↔ 
  ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2)) :=
sorry

end equation_holds_iff_specific_pairs_l4129_412984


namespace inequality_theorem_l4129_412949

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hk₁ : x₁ * y₁ - z₁^2 > 0) (hk₂ : x₂ * y₂ - z₂^2 > 0) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
    x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂ ∧ x₁ * y₁ - z₁^2 = x₂ * y₂ - z₂^2) :=
by sorry

end inequality_theorem_l4129_412949


namespace william_final_napkins_l4129_412971

def napkin_problem (initial_napkins : ℕ) (olivia_napkins : ℕ) : ℕ :=
  let amelia_napkins := 2 * olivia_napkins
  let charlie_napkins := amelia_napkins / 2
  let georgia_napkins := 3 * charlie_napkins
  initial_napkins + olivia_napkins + amelia_napkins + charlie_napkins + georgia_napkins

theorem william_final_napkins :
  napkin_problem 15 10 = 85 := by
  sorry

end william_final_napkins_l4129_412971


namespace candy_sampling_percentage_l4129_412925

theorem candy_sampling_percentage (caught_percentage : ℝ) (total_percentage : ℝ)
  (h1 : caught_percentage = 22)
  (h2 : total_percentage = 23.157894736842106) :
  total_percentage - caught_percentage = 1.157894736842106 := by
  sorry

end candy_sampling_percentage_l4129_412925


namespace hyperbola_equation_l4129_412914

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if point P(3, 5/2) lies on the hyperbola and the radius of the incircle of
    triangle PF₁F₂ (where F₁ and F₂ are the left and right foci) is 1,
    then a = 2 and b = √5. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (F₁ F₂ : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the hyperbola
    (F₁.1 < 0 ∧ F₂.1 > 0) ∧
    -- P(3, 5/2) lies on the hyperbola
    3^2 / a^2 - (5/2)^2 / b^2 = 1 ∧
    -- The radius of the incircle of triangle PF₁F₂ is 1
    (∃ (r : ℝ), r = 1 ∧
      r = (dist F₁ (3, 5/2) + dist F₂ (3, 5/2) + dist F₁ F₂) /
          (dist F₁ (3, 5/2) / r + dist F₂ (3, 5/2) / r + dist F₁ F₂ / r))) →
  a = 2 ∧ b = Real.sqrt 5 := by
sorry


end hyperbola_equation_l4129_412914


namespace exam_score_proof_l4129_412904

/-- Proves that the average score of students who took the exam on the assigned day was 65% -/
theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℚ) 
  (makeup_score : ℚ) (class_average : ℚ) : 
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_score = 95 / 100 →
  class_average = 74 / 100 →
  (assigned_day_percentage * total_students * assigned_day_score + 
   (1 - assigned_day_percentage) * total_students * makeup_score) / total_students = class_average →
  assigned_day_score = 65 / 100 :=
by
  sorry

#check exam_score_proof

end exam_score_proof_l4129_412904


namespace two_digit_numbers_problem_l4129_412994

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def share_digit (a b : ℕ) : Prop :=
  (a / 10 = b / 10) ∨ (a % 10 = b % 10)

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem two_digit_numbers_problem (a b : ℕ) :
  is_two_digit a ∧ is_two_digit b ∧
  a = b + 14 ∧
  share_digit a b ∧
  sum_of_digits a = 2 * sum_of_digits b →
  ((a = 37 ∧ b = 23) ∨ (a = 31 ∧ b = 17)) := by
  sorry

end two_digit_numbers_problem_l4129_412994


namespace common_prime_root_quadratics_l4129_412920

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * p + b = 0 ∧ 
    (p : ℤ)^2 + b * p + 1100 = 0) → 
  a = 274 ∨ a = 40 := by
sorry

end common_prime_root_quadratics_l4129_412920


namespace decimal_expansion_four_seventeenths_l4129_412996

/-- The decimal expansion of 4/17 has a repeating block of 235. -/
theorem decimal_expansion_four_seventeenths :
  ∃ (a b : ℕ), (4 : ℚ) / 17 = (a : ℚ) / 999 + (b : ℚ) / (999 * 1000) ∧ a = 235 ∧ b < 999 := by
  sorry

end decimal_expansion_four_seventeenths_l4129_412996


namespace smaller_root_of_quadratic_l4129_412921

theorem smaller_root_of_quadratic (x : ℝ) :
  x^2 + 10*x - 24 = 0 → (x = -12 ∨ x = 2) ∧ -12 < 2 := by
  sorry

end smaller_root_of_quadratic_l4129_412921


namespace quadratic_equation_roots_quadratic_equation_distinct_roots_l4129_412948

/-- The quadratic equation (a-3)x^2 - 4x - 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop :=
  (a - 3) * x^2 - 4 * x - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ :=
  16 - 4 * (a - 3) * (-1)

theorem quadratic_equation_roots (a : ℝ) :
  (∃ x : ℝ, quadratic_equation a x ∧
    (∀ y : ℝ, quadratic_equation a y → y = x)) →
  a = -1 ∧ (∀ x : ℝ, quadratic_equation a x → x = -1/2) :=
sorry

theorem quadratic_equation_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x ∧ quadratic_equation a y) →
  (a > -1 ∧ a ≠ 3) :=
sorry

end quadratic_equation_roots_quadratic_equation_distinct_roots_l4129_412948


namespace all_crop_to_diagonal_l4129_412999

/-- A symmetric kite-shaped field -/
structure KiteField where
  long_side : ℝ
  short_side : ℝ
  angle : ℝ
  long_side_positive : 0 < long_side
  short_side_positive : 0 < short_side
  angle_range : 0 < angle ∧ angle < π

/-- The fraction of the field area closer to the longer diagonal -/
def fraction_closer_to_diagonal (k : KiteField) : ℝ :=
  1 -- Definition, not proof

/-- The theorem statement -/
theorem all_crop_to_diagonal (k : KiteField) 
  (h1 : k.long_side = 100)
  (h2 : k.short_side = 70)
  (h3 : k.angle = 2 * π / 3) :
  fraction_closer_to_diagonal k = 1 := by
  sorry

end all_crop_to_diagonal_l4129_412999


namespace acorn_problem_l4129_412951

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
def total_acorns (shawna sheila danny : ℕ) : ℕ := shawna + sheila + danny

/-- Theorem stating the total number of acorns given the problem conditions -/
theorem acorn_problem (shawna sheila danny : ℕ) 
  (h1 : shawna = 7)
  (h2 : sheila = 5 * shawna)
  (h3 : danny = sheila + 3) :
  total_acorns shawna sheila danny = 80 := by
  sorry

end acorn_problem_l4129_412951


namespace fraction_ordering_l4129_412913

theorem fraction_ordering : (8 : ℚ) / 25 < 1 / 3 ∧ 1 / 3 < 10 / 31 ∧ 10 / 31 < 6 / 17 := by
  sorry

end fraction_ordering_l4129_412913


namespace surface_area_unchanged_l4129_412973

/-- Represents a cube with given side length -/
structure Cube where
  side : ℝ
  side_pos : side > 0

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def original_cube : Cube := ⟨4, by norm_num⟩

/-- Represents the corner cube to be removed -/
def corner_cube : Cube := ⟨2, by norm_num⟩

/-- Number of corners in a cube -/
def num_corners : ℕ := 8

/-- Theorem stating that the surface area remains unchanged after removing corner cubes -/
theorem surface_area_unchanged : 
  surface_area original_cube = surface_area original_cube := by sorry

end surface_area_unchanged_l4129_412973


namespace symmetric_point_x_axis_l4129_412968

/-- Given two points P and Q in a 2D plane, where Q is symmetric to P with respect to the x-axis,
    this theorem proves that the x-coordinate of Q is the same as P, and the y-coordinate of Q
    is the negative of P's y-coordinate. -/
theorem symmetric_point_x_axis 
  (P Q : ℝ × ℝ) 
  (h_symmetric : Q.1 = P.1 ∧ Q.2 = -P.2) 
  (h_P : P = (-3, 1)) : 
  Q = (-3, -1) := by
  sorry

end symmetric_point_x_axis_l4129_412968


namespace circle_equation_equivalence_l4129_412940

theorem circle_equation_equivalence :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 := by
sorry

end circle_equation_equivalence_l4129_412940


namespace james_beats_record_l4129_412947

/-- The number of points James beat the old record by -/
def points_above_record (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (old_record : ℕ) : ℕ :=
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + 
                      two_point_conversions * 2
  total_points - old_record

/-- Theorem stating that James beat the old record by 72 points -/
theorem james_beats_record : points_above_record 4 6 15 6 300 = 72 := by
  sorry

end james_beats_record_l4129_412947


namespace cheryl_pesto_production_l4129_412906

/-- Represents the pesto production scenario -/
structure PestoProduction where
  basil_per_pesto : ℕ  -- cups of basil needed for 1 cup of pesto
  basil_per_week : ℕ   -- cups of basil harvested per week
  harvest_weeks : ℕ    -- number of weeks of harvest

/-- Calculates the total cups of pesto that can be produced -/
def total_pesto (p : PestoProduction) : ℕ :=
  (p.basil_per_week * p.harvest_weeks) / p.basil_per_pesto

/-- Theorem: Given the conditions, Cheryl can make 32 cups of pesto -/
theorem cheryl_pesto_production :
  let p := PestoProduction.mk 4 16 8
  total_pesto p = 32 := by
  sorry

end cheryl_pesto_production_l4129_412906


namespace people_who_left_line_l4129_412916

theorem people_who_left_line (initial_people : ℕ) (joined : ℕ) (final_people : ℕ) 
  (h1 : initial_people = 9)
  (h2 : joined = 3)
  (h3 : final_people = 6)
  : initial_people - (initial_people - joined + final_people) = 6 := by
  sorry

end people_who_left_line_l4129_412916


namespace line_equation_coordinate_form_l4129_412992

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line passing through the origin -/
structure Line where
  direction : Vector3D

/-- The direction vector of a line is a unit vector -/
def Line.isUnitVector (l : Line) : Prop :=
  l.direction.x^2 + l.direction.y^2 + l.direction.z^2 = 1

/-- The components of the direction vector are cosines of angles with coordinate axes -/
def Line.directionCosines (l : Line) (α β γ : ℝ) : Prop :=
  l.direction.x = Real.cos α ∧
  l.direction.y = Real.cos β ∧
  l.direction.z = Real.cos γ

/-- A point on the line -/
def Line.pointOnLine (l : Line) (t : ℝ) : Vector3D :=
  { x := t * l.direction.x,
    y := t * l.direction.y,
    z := t * l.direction.z }

/-- The coordinate form of the line equation -/
def Line.coordinateForm (l : Line) (α β γ : ℝ) : Prop :=
  ∀ (p : Vector3D), p ∈ Set.range (l.pointOnLine) →
    p.x / Real.cos α = p.y / Real.cos β ∧
    p.y / Real.cos β = p.z / Real.cos γ

/-- The main theorem: proving the coordinate form of the line equation -/
theorem line_equation_coordinate_form (l : Line) (α β γ : ℝ) :
  l.isUnitVector →
  l.directionCosines α β γ →
  l.coordinateForm α β γ := by
  sorry


end line_equation_coordinate_form_l4129_412992


namespace function_is_periodic_l4129_412986

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the parameters a and b
variable (a b : ℝ)

-- State the conditions
axiom cond1 : ∀ x, f x = f (2 * b - x)
axiom cond2 : ∀ x, f (a + x) = -f (a - x)
axiom cond3 : a ≠ b

-- State the theorem
theorem function_is_periodic : ∀ x, f x = f (x + 4 * (a - b)) := by sorry

end function_is_periodic_l4129_412986


namespace no_x3_term_condition_l4129_412967

/-- The coefficient of x^3 in the expansion of ((x+a)^2(2x-1/x)^5) -/
def coeff_x3 (a : ℝ) : ℝ := 80 - 80 * a^2

/-- The theorem stating that the value of 'a' for which the expansion of 
    ((x+a)^2(2x-1/x)^5) does not contain the x^3 term is ±1 -/
theorem no_x3_term_condition (a : ℝ) : 
  coeff_x3 a = 0 ↔ a = 1 ∨ a = -1 := by
  sorry

#check no_x3_term_condition

end no_x3_term_condition_l4129_412967


namespace product_of_repeating_decimals_l4129_412935

/-- Represents the repeating decimal 0.090909... -/
def a : ℚ := 1 / 11

/-- Represents the repeating decimal 0.777777... -/
def b : ℚ := 7 / 9

/-- The product of the repeating decimals 0.090909... and 0.777777... equals 7/99 -/
theorem product_of_repeating_decimals : a * b = 7 / 99 := by
  sorry

end product_of_repeating_decimals_l4129_412935


namespace class_size_l4129_412910

theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 10) :
  football + tennis - both + neither = 39 := by
  sorry

end class_size_l4129_412910


namespace isosceles_triangle_30_angle_diff_l4129_412939

-- Define an isosceles triangle with one angle of 30 degrees
structure IsoscelesTriangle30 where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  isosceles : (angles 0 = angles 1) ∨ (angles 1 = angles 2) ∨ (angles 0 = angles 2)
  has_30 : angles 0 = 30 ∨ angles 1 = 30 ∨ angles 2 = 30

-- State the theorem
theorem isosceles_triangle_30_angle_diff 
  (t : IsoscelesTriangle30) : 
  ∃ (i j : Fin 3), i ≠ j ∧ (t.angles i - t.angles j = 90 ∨ t.angles i - t.angles j = 0) :=
sorry

end isosceles_triangle_30_angle_diff_l4129_412939


namespace set_operations_l4129_412962

-- Define the universal set U
def U : Set ℕ := {n : ℕ | n % 2 = 0 ∧ n ≤ 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B
def B : Set ℕ := {x : ℕ | x ∈ A ∧ x < 4}

theorem set_operations :
  (Set.compl A) = {8, 10} ∧
  (A ∩ (Set.compl B)) = {4, 6} := by
  sorry

#check set_operations

end set_operations_l4129_412962


namespace journey_equation_correct_l4129_412945

/-- Represents a journey with a stop -/
structure Journey where
  rate_before : ℝ  -- rate before stop in km/h
  rate_after : ℝ   -- rate after stop in km/h
  stop_time : ℝ    -- stop time in hours
  total_time : ℝ   -- total journey time in hours
  total_distance : ℝ -- total distance traveled in km

/-- The equation for the journey is correct -/
theorem journey_equation_correct (j : Journey) 
  (h1 : j.rate_before = 80)
  (h2 : j.rate_after = 100)
  (h3 : j.stop_time = 1/3)
  (h4 : j.total_time = 3)
  (h5 : j.total_distance = 250) :
  ∃ t : ℝ, t ≥ 0 ∧ t ≤ j.total_time - j.stop_time ∧ 
  j.rate_before * t + j.rate_after * (j.total_time - j.stop_time - t) = j.total_distance :=
sorry

end journey_equation_correct_l4129_412945


namespace tangent_line_to_parabola_l4129_412954

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ x y : ℝ, x^2 = 4*y ∧ y = k*x - 2 ∧ k = (1/2)*x) → k^2 = 2 :=
sorry

end tangent_line_to_parabola_l4129_412954


namespace angle_expression_equality_l4129_412931

theorem angle_expression_equality (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin (3 * Real.pi / 2 + θ) + Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 2 := by
  sorry

end angle_expression_equality_l4129_412931


namespace book_selling_loss_l4129_412995

/-- Calculates the loss from buying and selling books -/
theorem book_selling_loss 
  (books_per_month : ℕ) 
  (book_cost : ℕ) 
  (months : ℕ) 
  (selling_price : ℕ) : 
  books_per_month * months * book_cost - selling_price = 220 :=
by
  sorry

#check book_selling_loss 3 20 12 500

end book_selling_loss_l4129_412995


namespace gcd_problem_l4129_412974

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 360 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 72) b = 72 := by
  sorry

end gcd_problem_l4129_412974


namespace inequality_proof_l4129_412901

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end inequality_proof_l4129_412901


namespace a_range_for_two_positive_zeros_l4129_412932

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

/-- The condition for f to have two positive zeros -/
def has_two_positive_zeros (a : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- Theorem stating the range of a for f to have two positive zeros -/
theorem a_range_for_two_positive_zeros :
  ∀ a : ℝ, has_two_positive_zeros a ↔ a > 3 :=
sorry

end a_range_for_two_positive_zeros_l4129_412932


namespace triangle_side_calculation_l4129_412933

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  a = 2 →
  c = 3 →
  B = 2 * π / 3 →  -- 120° in radians
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →  -- Law of Cosines
  b = Real.sqrt 19 := by
sorry

end triangle_side_calculation_l4129_412933


namespace race_head_start_l4129_412998

theorem race_head_start (L : ℝ) (va vb : ℝ) (h : va = 20/13 * vb) :
  let H := (L - H) / vb + 0.6 * L / vb - L / va
  H = 19/20 * L := by
sorry

end race_head_start_l4129_412998


namespace largest_solution_value_l4129_412919

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log 10 / Real.log (x^2) + Real.log 10 / Real.log (x^4) + Real.log 10 / Real.log (9*x^5) = 0

-- Define the set of solutions
def solution_set := { x : ℝ | equation x ∧ x > 0 }

-- State the theorem
theorem largest_solution_value :
  ∃ (x : ℝ), x ∈ solution_set ∧ 
  (∀ (y : ℝ), y ∈ solution_set → y ≤ x) ∧
  (1 / x^18 = 9^93) := by
  sorry

end largest_solution_value_l4129_412919


namespace smallest_integer_with_remainder_l4129_412987

theorem smallest_integer_with_remainder (k : ℕ) : k = 275 ↔ 
  k > 1 ∧ 
  k % 13 = 2 ∧ 
  k % 7 = 2 ∧ 
  k % 3 = 2 ∧ 
  ∀ m : ℕ, m > 1 → m % 13 = 2 → m % 7 = 2 → m % 3 = 2 → k ≤ m :=
by sorry

end smallest_integer_with_remainder_l4129_412987


namespace asymptote_sum_l4129_412922

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) with integer coefficients A, B, C,
    if it has vertical asymptotes at x = -3, 0, and 4, then A + B + C = -13 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 4 → 
    x / (x^3 + A*x^2 + B*x + C) ≠ 0) →
  A + B + C = -13 := by
  sorry

end asymptote_sum_l4129_412922


namespace A_value_l4129_412926

noncomputable def A (x y : ℝ) : ℝ :=
  (Real.sqrt (x^3 + 2*x^2*y) + Real.sqrt (x^4 + 2 - x^3) - (x^(3/2) + x^2)) /
  (Real.sqrt (2*(x + y - Real.sqrt (x^2 + 2*x*y))) * (x^(2/3) - x^(5/6) + x))

theorem A_value (x y : ℝ) (hx : x > 0) :
  (y > 0 → A x y = x^(1/3) + x^(1/2)) ∧
  (-x/2 ≤ y ∧ y < 0 → A x y = -(x^(1/3) + x^(1/2))) :=
by sorry

end A_value_l4129_412926


namespace binary_to_base5_l4129_412959

-- Define the binary number
def binary_num : List Bool := [true, true, false, true, false, true, true]

-- Function to convert binary to decimal
def binary_to_decimal (bin : List Bool) : ℕ :=
  bin.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert decimal to base 5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_to_base5 :
  decimal_to_base5 (binary_to_decimal binary_num) = [4, 1, 2] :=
sorry

end binary_to_base5_l4129_412959


namespace factoring_expression_l4129_412989

theorem factoring_expression (y : ℝ) : 
  5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end factoring_expression_l4129_412989


namespace johns_butterfly_jars_l4129_412961

/-- The number of caterpillars in each jar -/
def caterpillars_per_jar : ℕ := 10

/-- The percentage of caterpillars that fail to become butterflies -/
def failure_rate : ℚ := 40 / 100

/-- The price of each butterfly in dollars -/
def price_per_butterfly : ℕ := 3

/-- The total amount made from selling butterflies in dollars -/
def total_amount : ℕ := 72

/-- The number of jars John has -/
def number_of_jars : ℕ := 4

theorem johns_butterfly_jars :
  let butterflies_per_jar := caterpillars_per_jar * (1 - failure_rate)
  let revenue_per_jar := butterflies_per_jar * price_per_butterfly
  total_amount / revenue_per_jar = number_of_jars := by sorry

end johns_butterfly_jars_l4129_412961


namespace subset_intersection_theorem_l4129_412997

theorem subset_intersection_theorem (α : ℝ) 
  (h_pos : α > 0) (h_bound : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ+) (S T : Finset (Finset (Fin n))),
    p > α * 2^(n : ℝ) ∧
    S.card = p ∧
    T.card = p ∧
    (∀ s ∈ S, ∀ t ∈ T, (s ∩ t).Nonempty) :=
sorry

end subset_intersection_theorem_l4129_412997


namespace range_of_a_l4129_412902

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 :=
sorry

end range_of_a_l4129_412902


namespace quadratic_roots_range_l4129_412924

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 2 ∧ 2 < x₂ ∧ 
   x₁^2 + 2*a*x₁ - 9 = 0 ∧ 
   x₂^2 + 2*a*x₂ - 9 = 0) → 
  a < 5/4 := by
sorry

end quadratic_roots_range_l4129_412924


namespace quadratic_solution_difference_squared_l4129_412941

theorem quadratic_solution_difference_squared : 
  ∀ a b : ℝ, (5 * a^2 - 6 * a - 55 = 0) → 
             (5 * b^2 - 6 * b - 55 = 0) → 
             (a ≠ b) →
             (a - b)^2 = 1296 / 25 := by
  sorry

end quadratic_solution_difference_squared_l4129_412941


namespace ferris_wheel_capacity_l4129_412972

theorem ferris_wheel_capacity 
  (total_seats : ℕ) 
  (people_per_seat : ℕ) 
  (broken_seats : ℕ) 
  (h1 : total_seats = 18) 
  (h2 : people_per_seat = 15) 
  (h3 : broken_seats = 10) :
  (total_seats - broken_seats) * people_per_seat = 120 := by
  sorry

end ferris_wheel_capacity_l4129_412972


namespace janes_number_l4129_412982

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of positive divisors of a natural number -/
def sum_divisors (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of prime divisors of a natural number -/
def sum_prime_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is uniquely determined by its sum of divisors -/
def is_unique_by_sum_divisors (n : ℕ) : Prop := 
  ∀ m : ℕ, sum_divisors m = sum_divisors n → m = n

/-- A function that checks if a number is uniquely determined by its sum of prime divisors -/
def is_unique_by_sum_prime_divisors (n : ℕ) : Prop := 
  ∀ m : ℕ, sum_prime_divisors m = sum_prime_divisors n → m = n

theorem janes_number : 
  ∃! n : ℕ, 
    500 < n ∧ 
    n < 1000 ∧ 
    num_divisors n = 20 ∧ 
    ¬ is_unique_by_sum_divisors n ∧ 
    ¬ is_unique_by_sum_prime_divisors n ∧ 
    n = 880 := by sorry

end janes_number_l4129_412982


namespace opposite_definition_opposite_of_eight_l4129_412980

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The opposite of a number added to the original number equals zero -/
theorem opposite_definition (x : ℝ) : x + opposite x = 0 := by sorry

/-- The opposite of 8 is -8 -/
theorem opposite_of_eight : opposite 8 = -8 := by sorry

end opposite_definition_opposite_of_eight_l4129_412980


namespace watch_sale_loss_percentage_l4129_412952

/-- Proves that the loss percentage is 10% for a watch sale scenario --/
theorem watch_sale_loss_percentage 
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1200)
  (h2 : selling_price < cost_price)
  (h3 : selling_price + 180 = cost_price * 1.05) :
  (cost_price - selling_price) / cost_price * 100 = 10 := by
  sorry

end watch_sale_loss_percentage_l4129_412952


namespace hcf_problem_l4129_412905

/-- Given two positive integers with specific properties, prove their HCF is 24 -/
theorem hcf_problem (a b : ℕ) : 
  (a > 0) → 
  (b > 0) → 
  (a ≤ b) → 
  (b = 744) → 
  (Nat.lcm a b = Nat.gcd a b * 11 * 12) → 
  Nat.gcd a b = 24 := by
sorry

end hcf_problem_l4129_412905


namespace square_orientation_after_1011_transformations_l4129_412938

/-- Represents the possible orientations of the square -/
inductive SquareOrientation
  | ABCD
  | DABC
  | BADC
  | DCBA

/-- Applies the 90-degree clockwise rotation -/
def rotate90 (s : SquareOrientation) : SquareOrientation :=
  match s with
  | SquareOrientation.ABCD => SquareOrientation.DABC
  | SquareOrientation.DABC => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.DCBA
  | SquareOrientation.DCBA => SquareOrientation.ABCD

/-- Applies the 180-degree rotation -/
def rotate180 (s : SquareOrientation) : SquareOrientation :=
  match s with
  | SquareOrientation.ABCD => SquareOrientation.BADC
  | SquareOrientation.DABC => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.ABCD
  | SquareOrientation.DCBA => SquareOrientation.ABCD

/-- Applies both rotations in sequence -/
def applyTransformations (s : SquareOrientation) : SquareOrientation :=
  rotate180 (rotate90 s)

/-- Applies the transformations n times -/
def applyNTimes (s : SquareOrientation) (n : Nat) : SquareOrientation :=
  match n with
  | 0 => s
  | n + 1 => applyTransformations (applyNTimes s n)

theorem square_orientation_after_1011_transformations :
  applyNTimes SquareOrientation.ABCD 1011 = SquareOrientation.DCBA := by
  sorry


end square_orientation_after_1011_transformations_l4129_412938


namespace andrews_mangoes_l4129_412915

/-- Given Andrew's purchase of grapes and mangoes, prove the amount of mangoes bought -/
theorem andrews_mangoes :
  ∀ (mango_kg : ℝ),
  let grape_kg : ℝ := 8
  let grape_price : ℝ := 70
  let mango_price : ℝ := 55
  let total_cost : ℝ := 1055
  (grape_kg * grape_price + mango_kg * mango_price = total_cost) →
  mango_kg = 9 := by
sorry

end andrews_mangoes_l4129_412915


namespace vasya_fraction_simplification_l4129_412964

theorem vasya_fraction_simplification (n : ℕ) : 
  (n = (2 * (10^1990 - 1)) / 3) → 
  (10^1990 + n) / (10 * n + 4) = 1/4 := by
sorry

end vasya_fraction_simplification_l4129_412964


namespace balance_theorem_l4129_412993

/-- The number of blue balls that balance one green ball -/
def green_to_blue : ℚ := 2

/-- The number of blue balls that balance one yellow ball -/
def yellow_to_blue : ℚ := 2.5

/-- The number of blue balls that balance one white ball -/
def white_to_blue : ℚ := 10/7

/-- The number of blue balls that balance 5 green, 4 yellow, and 3 white balls -/
def total_blue_balls : ℚ := 170/7

theorem balance_theorem :
  5 * green_to_blue + 4 * yellow_to_blue + 3 * white_to_blue = total_blue_balls :=
by sorry

end balance_theorem_l4129_412993


namespace sum_difference_with_triangular_problem_solution_l4129_412917

def even_sum (n : ℕ) : ℕ := n * (n + 1)

def odd_sum (n : ℕ) : ℕ := n * n

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sum_difference_with_triangular (n : ℕ) :
  even_sum n - odd_sum n + triangular_sum n = n * (n * n + 3) / 3 :=
by sorry

theorem problem_solution : 
  even_sum 1500 - odd_sum 1500 + triangular_sum 1500 = 563628000 :=
by sorry

end sum_difference_with_triangular_problem_solution_l4129_412917


namespace impossible_parking_space_l4129_412991

theorem impossible_parking_space (L W : ℝ) : 
  L = 99 ∧ L + 2 * W = 37 → False :=
by sorry

end impossible_parking_space_l4129_412991


namespace geometric_sequence_bounded_ratio_counterexample_l4129_412970

theorem geometric_sequence_bounded_ratio_counterexample :
  ¬ (∀ (a₁ : ℝ) (q : ℝ) (a : ℝ),
    (a₁ > 0 ∧ q > 0) →
    (∀ n : ℕ, a₁ * q^(n - 1) < a) →
    (q > 0 ∧ q < 1)) :=
by sorry

end geometric_sequence_bounded_ratio_counterexample_l4129_412970


namespace slope_angles_of_line_l_l4129_412990

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

/-- Line l in parametric form -/
def line_l (x y t α : ℝ) : Prop := x = 1 + t * Real.cos α ∧ y = t * Real.sin α

/-- Intersection condition -/
def intersection_condition (t α : ℝ) : Prop := t^2 - 2*t*Real.cos α - 3 = 0

/-- Main theorem -/
theorem slope_angles_of_line_l (α : ℝ) :
  (∃ ρ θ x y t, curve_C ρ θ ∧ line_l x y t α ∧ intersection_condition t α) →
  α = π/4 ∨ α = 3*π/4 :=
sorry

end slope_angles_of_line_l_l4129_412990


namespace paige_score_l4129_412953

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


end paige_score_l4129_412953


namespace equation_system_result_l4129_412976

theorem equation_system_result (x y z : ℝ) 
  (eq1 : 2*x + y + z = 6) 
  (eq2 : x + 2*y + z = 7) : 
  5*x^2 + 8*x*y + 5*y^2 = 41 := by
sorry

end equation_system_result_l4129_412976


namespace cos_product_equation_l4129_412927

theorem cos_product_equation (α : Real) (h : Real.tan α = 2) :
  Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 + α) = 2 / 5 := by
  sorry

end cos_product_equation_l4129_412927


namespace john_necklaces_l4129_412909

/-- Given the number of wire spools, length of each spool, and wire required per necklace,
    calculate the number of necklaces that can be made. -/
def necklaces_from_wire (num_spools : ℕ) (spool_length : ℕ) (wire_per_necklace : ℕ) : ℕ :=
  (num_spools * spool_length) / wire_per_necklace

/-- Prove that John can make 15 necklaces with the given conditions. -/
theorem john_necklaces : necklaces_from_wire 3 20 4 = 15 := by
  sorry

end john_necklaces_l4129_412909


namespace rectangle_width_l4129_412937

/-- Given a rectangle with area 300 square meters and perimeter 70 meters, prove its width is 15 meters. -/
theorem rectangle_width (length width : ℝ) : 
  length * width = 300 ∧ 
  2 * (length + width) = 70 → 
  width = 15 := by
sorry

end rectangle_width_l4129_412937


namespace percentage_problem_l4129_412985

theorem percentage_problem (x : ℝ) : 
  (0.12 * 160) - (x / 100 * 80) = 11.2 ↔ x = 10 := by sorry

end percentage_problem_l4129_412985


namespace debt_amount_l4129_412960

/-- Represents the savings of the three girls and the debt amount -/
structure Savings where
  lulu : ℕ
  nora : ℕ
  tamara : ℕ
  debt : ℕ

/-- Theorem stating the debt amount given the conditions -/
theorem debt_amount (s : Savings) :
  s.lulu = 6 ∧
  s.nora = 5 * s.lulu ∧
  s.nora = 3 * s.tamara ∧
  s.lulu + s.nora + s.tamara = s.debt + 6 →
  s.debt = 40 := by
  sorry


end debt_amount_l4129_412960


namespace tiling_problem_l4129_412942

/-- Number of ways to tile a 3 × n rectangle -/
def tiling_ways (n : ℕ) : ℚ :=
  (2^(n+2) + (-1)^(n+1)) / 3

/-- Proof of the tiling problem -/
theorem tiling_problem (n : ℕ) (h : n > 3) :
  tiling_ways n = (2^(n+2) + (-1)^(n+1)) / 3 :=
by sorry

end tiling_problem_l4129_412942


namespace negative_rational_and_fraction_l4129_412981

-- Define the number -0.3
def num : ℚ := -3/10

-- Theorem statement
theorem negative_rational_and_fraction (n : ℚ) (h : n = -3/10) :
  n < 0 ∧ ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b :=
sorry

end negative_rational_and_fraction_l4129_412981


namespace quadratic_one_root_l4129_412936

theorem quadratic_one_root (k : ℝ) : k > 0 ∧ (∃! x : ℝ, x^2 + 6*k*x + 9*k = 0) ↔ k = 1 := by
  sorry

end quadratic_one_root_l4129_412936


namespace area_of_circle_with_diameter_6_l4129_412955

-- Define the circle
def circle_diameter : ℝ := 6

-- Theorem statement
theorem area_of_circle_with_diameter_6 :
  (π * (circle_diameter / 2)^2) = 9 * π := by sorry

end area_of_circle_with_diameter_6_l4129_412955


namespace person_A_age_l4129_412950

theorem person_A_age (current_age_A current_age_B past_age_A past_age_B years_ago : ℕ) : 
  current_age_A + current_age_B = 70 →
  current_age_A - years_ago = current_age_B →
  past_age_B = past_age_A / 2 →
  past_age_A = current_age_A →
  past_age_B = current_age_B - years_ago →
  current_age_A = 42 := by
  sorry

end person_A_age_l4129_412950


namespace inscribed_circle_diameter_l4129_412966

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 13) (h2 : DF = 14) (h3 : EF = 15) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let radius := area / s
  2 * radius = 8 := by sorry

end inscribed_circle_diameter_l4129_412966


namespace inequality_solution_set_l4129_412918

theorem inequality_solution_set (x : ℝ) :
  x ≠ -2 ∧ x ≠ 9/2 →
  ((x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9)) ↔
  (-9/2 ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2) :=
by sorry

end inequality_solution_set_l4129_412918


namespace geometric_sequence_problem_l4129_412956

theorem geometric_sequence_problem (a : ℝ) :
  a > 0 ∧
  (∃ (r : ℝ), 210 * r = a ∧ a * r = 63 / 40) →
  a = 18.1875 := by
sorry

end geometric_sequence_problem_l4129_412956


namespace common_tangent_sum_l4129_412978

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 52/5

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 25/10

/-- Common tangent line L -/
def L (a b c x y : ℝ) : Prop := a*x + b*y = c

/-- Theorem stating the sum of a, b, and c for the common tangent line -/
theorem common_tangent_sum (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧ L a b c x₁ y₁ ∧ L a b c x₂ y₂) →
  (∃ (k : ℚ), a = k * b) →
  Nat.gcd a (Nat.gcd b c) = 1 →
  a + b + c = 17 := by
  sorry

end common_tangent_sum_l4129_412978


namespace sandwiches_theorem_l4129_412977

/-- The number of sandwiches Ruth prepared -/
def total_sandwiches : ℕ := 10

/-- The number of sandwiches Ruth ate -/
def ruth_ate : ℕ := 1

/-- The number of sandwiches Ruth's brother ate -/
def brother_ate : ℕ := 2

/-- The number of sandwiches Ruth's first cousin ate -/
def first_cousin_ate : ℕ := 2

/-- The number of sandwiches each of Ruth's other two cousins ate -/
def other_cousins_ate_each : ℕ := 1

/-- The number of sandwiches left -/
def sandwiches_left : ℕ := 3

/-- Theorem stating that the total number of sandwiches Ruth prepared
    is equal to the sum of sandwiches eaten by everyone and those left -/
theorem sandwiches_theorem :
  total_sandwiches = ruth_ate + brother_ate + first_cousin_ate +
    (2 * other_cousins_ate_each) + sandwiches_left :=
by
  sorry

end sandwiches_theorem_l4129_412977


namespace quadrilateral_area_l4129_412946

/-- The area of a quadrilateral formed by four squares arranged in a specific manner -/
theorem quadrilateral_area (s₁ s₂ s₃ s₄ : ℝ) (h₁ : s₁ = 1) (h₂ : s₂ = 3) (h₃ : s₃ = 5) (h₄ : s₄ = 7) :
  let total_length := s₁ + s₂ + s₃ + s₄
  let height_ratio := s₄ / total_length
  let height₂ := s₂ * height_ratio
  let height₃ := s₃ * height_ratio
  let quadrilateral_height := s₃ - s₂
  (height₂ + height₃) * quadrilateral_height / 2 = 3.5 := by
  sorry

end quadrilateral_area_l4129_412946


namespace function_inequality_l4129_412957

open Real

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x : ℝ, f x > deriv f x) : 
  (ℯ^2016 * f (-2016) > f 0) ∧ (f 2016 < ℯ^2016 * f 0) := by
  sorry

end function_inequality_l4129_412957


namespace second_parallel_line_length_l4129_412988

/-- Given a triangle with base length 18 and three parallel lines dividing it into four equal areas,
    the length of the second parallel line from the base is 9√2. -/
theorem second_parallel_line_length (base : ℝ) (l₁ l₂ l₃ : ℝ) :
  base = 18 →
  l₁ < l₂ ∧ l₂ < l₃ →
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ base → (x * l₁) = (x * l₂) ∧ (x * l₂) = (x * l₃) ∧ (x * l₃) = (x * base / 4)) →
  l₂ = 9 * Real.sqrt 2 := by
  sorry

end second_parallel_line_length_l4129_412988


namespace greatest_b_value_l4129_412943

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 14*x + 45 ≤ 0 → x ≤ 9) ∧ 
  (9^2 - 14*9 + 45 ≤ 0) := by
  sorry

end greatest_b_value_l4129_412943
