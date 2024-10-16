import Mathlib

namespace NUMINAMATH_CALUDE_bus_trip_speed_l1169_116935

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 450 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (original_speed : ℝ),
    distance / original_speed - time_decrease = distance / (original_speed + speed_increase) ∧
    original_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l1169_116935


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_min_value_of_f_l1169_116945

-- Part a
theorem polynomial_nonnegative (x : ℝ) (h : x ≥ 1) :
  x^3 - 5*x^2 + 8*x - 4 ≥ 0 := by sorry

-- Part b
def f (a b : ℝ) := a*b*(a+b-10) + 8*(a+b)

theorem min_value_of_f :
  ∃ (min : ℝ), min = 8 ∧ 
  ∀ (a b : ℝ), a ≥ 1 → b ≥ 1 → f a b ≥ min := by sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_min_value_of_f_l1169_116945


namespace NUMINAMATH_CALUDE_dad_caught_more_trouts_l1169_116960

-- Define the number of trouts Caleb caught
def caleb_trouts : ℕ := 2

-- Define the number of trouts Caleb's dad caught
def dad_trouts : ℕ := 3 * caleb_trouts

-- Theorem to prove
theorem dad_caught_more_trouts : dad_trouts - caleb_trouts = 4 := by
  sorry

end NUMINAMATH_CALUDE_dad_caught_more_trouts_l1169_116960


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1169_116914

/-- Calculates the total distance covered by a car given its uphill and downhill speeds and times. -/
def total_distance (uphill_speed downhill_speed uphill_time downhill_time : ℝ) : ℝ :=
  uphill_speed * uphill_time + downhill_speed * downhill_time

/-- Theorem stating that under the given conditions, the total distance covered by the car is 400 km. -/
theorem car_distance_theorem :
  let uphill_speed : ℝ := 30
  let downhill_speed : ℝ := 50
  let uphill_time : ℝ := 5
  let downhill_time : ℝ := 5
  total_distance uphill_speed downhill_speed uphill_time downhill_time = 400 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1169_116914


namespace NUMINAMATH_CALUDE_calendar_reuse_2052_l1169_116997

def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

def calendar_repeats (year1 year2 : ℕ) : Prop :=
  is_leap_year year1 ∧ is_leap_year year2 ∧ (year2 - year1) % 28 = 0

theorem calendar_reuse_2052 :
  ∀ y : ℕ, y > 1912 → y < 2052 → ¬(calendar_repeats y 2052) →
  calendar_repeats 1912 2052 ∧ is_leap_year 1912 ∧ is_leap_year 2052 :=
sorry

end NUMINAMATH_CALUDE_calendar_reuse_2052_l1169_116997


namespace NUMINAMATH_CALUDE_f_properties_l1169_116916

noncomputable def f (b c x : ℝ) : ℝ := |x| * x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x y : ℝ, x < y → b > 0 → f b c x < f b c y) ∧
  (∀ x : ℝ, f b c x = f b c (-x)) ∧
  (∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1169_116916


namespace NUMINAMATH_CALUDE_magnitude_of_sum_equals_five_l1169_116927

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![2, 2]

theorem magnitude_of_sum_equals_five :
  ‖vector_a + vector_b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_equals_five_l1169_116927


namespace NUMINAMATH_CALUDE_vector_relation_in_right_triangular_prism_l1169_116956

/-- A right triangular prism with vertices A, B, C, A₁, B₁, C₁ -/
structure RightTriangularPrism (V : Type*) [AddCommGroup V] :=
  (A B C A₁ B₁ C₁ : V)

/-- The theorem stating the relation between vectors in a right triangular prism -/
theorem vector_relation_in_right_triangular_prism
  {V : Type*} [AddCommGroup V] (prism : RightTriangularPrism V)
  (a b c : V)
  (h1 : prism.C - prism.A = a)
  (h2 : prism.C - prism.B = b)
  (h3 : prism.C - prism.C₁ = c) :
  prism.A₁ - prism.B = -a - c + b := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_in_right_triangular_prism_l1169_116956


namespace NUMINAMATH_CALUDE_johns_total_hours_l1169_116912

/-- Represents John's volunteering schedule for the year -/
structure VolunteerSchedule where
  jan_to_mar : Nat  -- Hours for January to March
  apr_to_jun : Nat  -- Hours for April to June
  jul_to_aug : Nat  -- Hours for July and August
  sep_to_oct : Nat  -- Hours for September and October
  november : Nat    -- Hours for November
  december : Nat    -- Hours for December
  bonus_days : Nat  -- Hours for bonus days (third Saturday of every month except May and June)
  charity_run : Nat -- Hours for annual charity run in June

/-- Calculates the total volunteering hours for the year -/
def total_hours (schedule : VolunteerSchedule) : Nat :=
  schedule.jan_to_mar +
  schedule.apr_to_jun +
  schedule.jul_to_aug +
  schedule.sep_to_oct +
  schedule.november +
  schedule.december +
  schedule.bonus_days +
  schedule.charity_run

/-- John's actual volunteering schedule for the year -/
def johns_schedule : VolunteerSchedule :=
  { jan_to_mar := 18
  , apr_to_jun := 24
  , jul_to_aug := 64
  , sep_to_oct := 24
  , november := 6
  , december := 6
  , bonus_days := 40
  , charity_run := 8
  }

/-- Theorem stating that John's total volunteering hours for the year is 190 -/
theorem johns_total_hours : total_hours johns_schedule = 190 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_hours_l1169_116912


namespace NUMINAMATH_CALUDE_sequence_problem_l1169_116913

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, a n ≠ 0) →
  a 3 - (a 7)^2 / 2 + a 11 = 0 →
  geometric_sequence b →
  b 7 = a 7 →
  b 1 * b 13 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1169_116913


namespace NUMINAMATH_CALUDE_crushing_load_calculation_l1169_116957

theorem crushing_load_calculation (T H K : ℝ) (hT : T = 3) (hH : H = 9) (hK : K = 2) :
  (50 * T^5) / (K * H^3) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_crushing_load_calculation_l1169_116957


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_exists_l1169_116995

theorem diophantine_equation_solution_exists :
  ∃ (x y z : ℕ+), 
    (z = Nat.gcd x y) ∧ 
    (x + y^2 + z^3 = x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_exists_l1169_116995


namespace NUMINAMATH_CALUDE_product_112_54_l1169_116937

theorem product_112_54 : 112 * 54 = 6048 := by
  sorry

end NUMINAMATH_CALUDE_product_112_54_l1169_116937


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1169_116911

/-- Given a geometric sequence {aₙ} with positive terms where a₁a₅ + 2a₃a₅ + a₃a₇ = 25, prove that a₃ + a₅ = 5. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) 
    (h2 : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n) 
    (h3 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) : 
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1169_116911


namespace NUMINAMATH_CALUDE_polygon_sides_l1169_116936

theorem polygon_sides (n : ℕ) : 
  (((n - 2) * 180) / 360 : ℚ) = 3 / 1 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1169_116936


namespace NUMINAMATH_CALUDE_max_distinct_values_l1169_116939

-- Define a 4x4 grid of non-negative integers
def Grid := Matrix (Fin 4) (Fin 4) ℕ

-- Define a function to check if a set of 5 cells sums to 5
def SumToFive (g : Grid) (cells : Finset (Fin 4 × Fin 4)) : Prop :=
  cells.card = 5 ∧ (cells.sum (fun c => g c.1 c.2) = 5)

-- Define the property that all valid 5-cell configurations sum to 5
def AllConfigsSumToFive (g : Grid) : Prop :=
  ∀ cells : Finset (Fin 4 × Fin 4), SumToFive g cells

-- Define the number of distinct values in the grid
def DistinctValues (g : Grid) : ℕ :=
  (Finset.univ.image (fun i => Finset.univ.image (g i))).card

-- State the theorem
theorem max_distinct_values (g : Grid) (h : AllConfigsSumToFive g) :
  DistinctValues g ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_distinct_values_l1169_116939


namespace NUMINAMATH_CALUDE_total_marbles_is_72_marble_ratio_is_2_4_6_l1169_116989

/-- Represents the number of marbles of each color in a bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Defines the properties of the marble bag based on the given conditions -/
def special_marble_bag : MarbleBag :=
  { red := 12,
    blue := 24,
    yellow := 36 }

/-- Theorem stating that the total number of marbles in the special bag is 72 -/
theorem total_marbles_is_72 :
  special_marble_bag.red + special_marble_bag.blue + special_marble_bag.yellow = 72 := by
  sorry

/-- Theorem stating that the ratio of marbles in the special bag is 2:4:6 -/
theorem marble_ratio_is_2_4_6 :
  2 * special_marble_bag.red = special_marble_bag.blue ∧
  3 * special_marble_bag.red = special_marble_bag.yellow := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_72_marble_ratio_is_2_4_6_l1169_116989


namespace NUMINAMATH_CALUDE_paco_cookies_l1169_116906

theorem paco_cookies (initial_cookies : ℕ) : initial_cookies = 7 :=
by
  -- Define the number of cookies eaten initially
  let initial_eaten : ℕ := 5
  -- Define the number of cookies bought
  let bought : ℕ := 3
  -- Define the number of cookies eaten after buying
  let later_eaten : ℕ := bought + 2
  -- Assert that all cookies were eaten
  have h : initial_cookies - initial_eaten + bought - later_eaten = 0 := by sorry
  -- Prove that initial_cookies = 7
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l1169_116906


namespace NUMINAMATH_CALUDE_total_area_of_triangles_l1169_116944

/-- Given two right triangles ABC and ABD with shared side AB, prove their total area -/
theorem total_area_of_triangles (AB AC BD : ℝ) : 
  AB = 15 →
  AC = 10 →
  BD = 8 →
  (1/2 * AB * AC) + (1/2 * AB * BD) = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_triangles_l1169_116944


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1169_116932

theorem sum_of_three_numbers (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1169_116932


namespace NUMINAMATH_CALUDE_water_percentage_in_first_liquid_l1169_116994

/-- Given two liquids in a glass, prove the percentage of water in the first liquid --/
theorem water_percentage_in_first_liquid 
  (water_percent_second : ℝ) 
  (parts_first : ℝ) 
  (parts_second : ℝ) 
  (water_percent_mixture : ℝ) : 
  water_percent_second = 0.35 →
  parts_first = 10 →
  parts_second = 4 →
  water_percent_mixture = 0.24285714285714285 →
  ∃ (water_percent_first : ℝ), 
    water_percent_first * parts_first + water_percent_second * parts_second = 
    water_percent_mixture * (parts_first + parts_second) ∧
    water_percent_first = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_first_liquid_l1169_116994


namespace NUMINAMATH_CALUDE_unique_k_for_rational_solutions_l1169_116985

def has_rational_solutions (a b c : ℤ) : Prop :=
  ∃ (x : ℚ), a * x^2 + b * x + c = 0

theorem unique_k_for_rational_solutions :
  ∀ k : ℕ+, (has_rational_solutions k 16 k ↔ k = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_rational_solutions_l1169_116985


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1169_116920

-- Inequality 1
theorem inequality_one (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) ↔ x ≤ -3/2 := by sorry

-- Inequality 2
theorem inequality_two (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1169_116920


namespace NUMINAMATH_CALUDE_range_of_f_l1169_116984

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The range of f is [2, +∞) -/
theorem range_of_f : Set.range f = { y | 2 ≤ y } := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1169_116984


namespace NUMINAMATH_CALUDE_black_ribbon_count_l1169_116938

theorem black_ribbon_count (total : ℕ) (silver : ℕ) : 
  silver = 40 →
  (1 : ℚ) / 4 + 1 / 3 + 1 / 6 + 1 / 12 + (silver : ℚ) / total = 1 →
  (total : ℚ) / 12 = 20 :=
by sorry

end NUMINAMATH_CALUDE_black_ribbon_count_l1169_116938


namespace NUMINAMATH_CALUDE_equation_equality_l1169_116950

theorem equation_equality (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  a^7 / (1 - a)^2 - a^7 / (1 + a)^2 = b^7 / (1 - b)^2 - b^7 / (1 + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l1169_116950


namespace NUMINAMATH_CALUDE_equation_roots_l1169_116922

theorem equation_roots : 
  ∀ x : ℝ, (x - 2) * (x - 3) = x - 2 ↔ x = 2 ∨ x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l1169_116922


namespace NUMINAMATH_CALUDE_g_negative_implies_a_range_l1169_116929

/-- The function g(x) defined on (0, +∞) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

/-- Theorem stating the range of a when g(x) is negative on [1, +∞) -/
theorem g_negative_implies_a_range (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, x ≥ 1 → g a x < 0) → -2 < a ∧ a < 0 := by
  sorry

/-- Lemma for the domain of g(x) -/
lemma g_domain (a : ℝ) (x : ℝ) (h : x > 0) : 
  g a x = (1/2) * a * x^2 - (a + 1) * x + Real.log x := by
  sorry

end NUMINAMATH_CALUDE_g_negative_implies_a_range_l1169_116929


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1169_116902

def f (x : ℝ) : ℝ := x^4 - 3*x^3 + 10*x^2 - 16*x + 5

def g (x k : ℝ) : ℝ := x^2 - x + k

theorem polynomial_division_remainder (k a : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, f x = g x k * q x + (2*x + a)) ↔ k = 8.5 ∧ a = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1169_116902


namespace NUMINAMATH_CALUDE_sum_of_squares_l1169_116931

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_seven_eq : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1169_116931


namespace NUMINAMATH_CALUDE_polygon_existence_l1169_116966

/-- A polygon on a unit grid --/
structure GridPolygon where
  sides : ℕ
  area : ℕ
  vertices : List (ℕ × ℕ)

/-- Predicate to check if a GridPolygon is valid --/
def isValidGridPolygon (p : GridPolygon) : Prop :=
  p.sides = p.vertices.length ∧
  p.area ≤ (List.maximum (p.vertices.map Prod.fst)).getD 0 * 
           (List.maximum (p.vertices.map Prod.snd)).getD 0

theorem polygon_existence : 
  (∃ p : GridPolygon, p.sides = 20 ∧ p.area = 9 ∧ isValidGridPolygon p) ∧
  (∃ p : GridPolygon, p.sides = 100 ∧ p.area = 49 ∧ isValidGridPolygon p) := by
  sorry

end NUMINAMATH_CALUDE_polygon_existence_l1169_116966


namespace NUMINAMATH_CALUDE_group_size_proof_l1169_116962

theorem group_size_proof (W : ℝ) (n : ℕ) 
  (h1 : (W + 20) / n = W / n + 4) 
  (h2 : W > 0) 
  (h3 : n > 0) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l1169_116962


namespace NUMINAMATH_CALUDE_journey_speed_proof_l1169_116979

/-- Proves that given a journey of 336 km completed in 15 hours, where the second half is traveled at 24 km/hr, the speed for the first half of the journey is 21 km/hr. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 336 →
  total_time = 15 →
  second_half_speed = 24 →
  let first_half_distance : ℝ := total_distance / 2
  let second_half_distance : ℝ := total_distance / 2
  let second_half_time : ℝ := second_half_distance / second_half_speed
  let first_half_time : ℝ := total_time - second_half_time
  let first_half_speed : ℝ := first_half_distance / first_half_time
  first_half_speed = 21 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l1169_116979


namespace NUMINAMATH_CALUDE_fletcher_well_diggers_l1169_116953

/-- The number of men hired by Mr Fletcher to dig a well -/
def num_men : ℕ :=
  let hours_day1 : ℕ := 10
  let hours_day2 : ℕ := 8
  let hours_day3 : ℕ := 15
  let total_hours : ℕ := hours_day1 + hours_day2 + hours_day3
  let pay_per_hour : ℕ := 10
  let total_payment : ℕ := 660
  total_payment / (total_hours * pay_per_hour)

theorem fletcher_well_diggers :
  num_men = 2 := by sorry

end NUMINAMATH_CALUDE_fletcher_well_diggers_l1169_116953


namespace NUMINAMATH_CALUDE_symmetric_circle_l1169_116969

/-- Given a circle and a line of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, (x + 2)^2 + y^2 = 2016) →  -- Original circle equation
  (∀ x y, x - y + 1 = 0) →           -- Line of symmetry
  (∀ x y, (x + 1)^2 + (y + 1)^2 = 2016) -- Symmetric circle equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l1169_116969


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1169_116996

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1169_116996


namespace NUMINAMATH_CALUDE_decomposition_theorem_l1169_116908

theorem decomposition_theorem (d n : ℕ) (hd : d > 0) (hn : n > 0) :
  ∃ (A B : Set ℕ), 
    (∀ k : ℕ, k > 0 → (k ∈ A ∨ k ∈ B)) ∧
    (A ∩ B = ∅) ∧
    (∀ x ∈ A, ∃ y ∈ B, d * x = n * d * y) ∧
    (∀ y ∈ B, ∃ x ∈ A, d * x = n * d * y) :=
sorry

end NUMINAMATH_CALUDE_decomposition_theorem_l1169_116908


namespace NUMINAMATH_CALUDE_power_equality_l1169_116974

theorem power_equality (n b : ℝ) : n = 2^(1/4) → n^b = 16 → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1169_116974


namespace NUMINAMATH_CALUDE_games_within_division_l1169_116955

/-- Represents a baseball league with specific game scheduling rules. -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  N : ℕ
  /-- Number of games played against each team in the other division -/
  M : ℕ
  /-- N is greater than twice M -/
  h1 : N > 2 * M
  /-- M is greater than 6 -/
  h2 : M > 6
  /-- Total number of games each team plays is 92 -/
  h3 : 3 * N + 4 * M = 92

/-- The number of games a team plays within its own division in the given baseball league is 60. -/
theorem games_within_division (league : BaseballLeague) : 3 * league.N = 60 := by
  sorry

end NUMINAMATH_CALUDE_games_within_division_l1169_116955


namespace NUMINAMATH_CALUDE_extremum_sum_l1169_116934

/-- A function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_sum (a b : ℝ) :
  f' a b 1 = 0 ∧ f a b 1 = 10 → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_extremum_sum_l1169_116934


namespace NUMINAMATH_CALUDE_frogs_on_logs_count_l1169_116954

/-- The number of frogs that climbed onto logs in the pond -/
def frogs_on_logs (total_frogs lily_pad_frogs rock_frogs : ℕ) : ℕ :=
  total_frogs - (lily_pad_frogs + rock_frogs)

/-- Theorem stating that the number of frogs on logs is 3 -/
theorem frogs_on_logs_count :
  frogs_on_logs 32 5 24 = 3 := by
  sorry

end NUMINAMATH_CALUDE_frogs_on_logs_count_l1169_116954


namespace NUMINAMATH_CALUDE_unique_intersection_bounded_difference_l1169_116907

-- Define the set U of functions satisfying the conditions
def U : Set (ℝ → ℝ) :=
  {f | ∃ x, f x = 2 * x ∧ ∀ x, 0 < (deriv f) x ∧ (deriv f) x < 2}

-- Statement 1: For any f in U, f(x) = 2x has exactly one solution
theorem unique_intersection (f : ℝ → ℝ) (hf : f ∈ U) :
  ∃! x, f x = 2 * x :=
sorry

-- Statement 2: For any h in U and x₁, x₂ close to 2023, |h(x₁) - h(x₂)| < 4
theorem bounded_difference (h : ℝ → ℝ) (hh : h ∈ U) :
  ∀ x₁ x₂, |x₁ - 2023| < 1 → |x₂ - 2023| < 1 → |h x₁ - h x₂| < 4 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_bounded_difference_l1169_116907


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_roots_existence_of_integral_roots_smallest_m_is_195_l1169_116901

theorem smallest_m_for_integral_roots (m : ℤ) : m > 0 → 
  (∃ x y : ℤ, 15 * x^2 - m * x + 630 = 0 ∧ 
              15 * y^2 - m * y + 630 = 0 ∧ 
              abs (x - y) ≤ 10) →
  m ≥ 195 :=
by sorry

theorem existence_of_integral_roots : 
  ∃ x y : ℤ, 15 * x^2 - 195 * x + 630 = 0 ∧ 
            15 * y^2 - 195 * y + 630 = 0 ∧ 
            abs (x - y) ≤ 10 :=
by sorry

theorem smallest_m_is_195 : 
  ∀ m : ℤ, m > 0 → 
  (∃ x y : ℤ, 15 * x^2 - m * x + 630 = 0 ∧ 
            15 * y^2 - m * y + 630 = 0 ∧ 
            abs (x - y) ≤ 10) → 
  m ≥ 195 ∧ 
  (∃ x y : ℤ, 15 * x^2 - 195 * x + 630 = 0 ∧ 
            15 * y^2 - 195 * y + 630 = 0 ∧ 
            abs (x - y) ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_roots_existence_of_integral_roots_smallest_m_is_195_l1169_116901


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l1169_116987

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l1169_116987


namespace NUMINAMATH_CALUDE_uber_lyft_cost_difference_uber_lyft_cost_difference_proof_l1169_116992

/-- The cost difference between Uber and Lyft rides --/
theorem uber_lyft_cost_difference : ℝ :=
  let taxi_cost : ℝ := 15  -- Derived from the 20% tip condition
  let lyft_cost : ℝ := taxi_cost + 4
  let uber_cost : ℝ := 22
  uber_cost - lyft_cost

/-- Proof of the cost difference between Uber and Lyft rides --/
theorem uber_lyft_cost_difference_proof :
  uber_lyft_cost_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_uber_lyft_cost_difference_uber_lyft_cost_difference_proof_l1169_116992


namespace NUMINAMATH_CALUDE_sampling_most_appropriate_for_qingming_l1169_116921

-- Define the survey methods
inductive SurveyMethod
| Census
| Sampling

-- Define the survey scenarios
inductive SurveyScenario
| MilkHygiene
| SubwaySecurity
| StudentSleep
| QingmingCommemoration

-- Define a function to determine the appropriateness of a survey method for a given scenario
def is_appropriate (scenario : SurveyScenario) (method : SurveyMethod) : Prop :=
  match scenario, method with
  | SurveyScenario.MilkHygiene, SurveyMethod.Sampling => True
  | SurveyScenario.SubwaySecurity, SurveyMethod.Census => True
  | SurveyScenario.StudentSleep, SurveyMethod.Sampling => True
  | SurveyScenario.QingmingCommemoration, SurveyMethod.Sampling => True
  | _, _ => False

-- Theorem stating that sampling is the most appropriate method for the Qingming commemoration scenario
theorem sampling_most_appropriate_for_qingming :
  ∀ (scenario : SurveyScenario) (method : SurveyMethod),
    is_appropriate scenario method →
    (scenario = SurveyScenario.QingmingCommemoration ∧ method = SurveyMethod.Sampling) ∨
    (scenario ≠ SurveyScenario.QingmingCommemoration) :=
by sorry

end NUMINAMATH_CALUDE_sampling_most_appropriate_for_qingming_l1169_116921


namespace NUMINAMATH_CALUDE_fraction_sum_power_six_l1169_116917

theorem fraction_sum_power_six : (5 / 3 : ℚ)^6 + (2 / 3 : ℚ)^6 = 15689 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_power_six_l1169_116917


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1169_116990

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_sum1 : a 1 + a 3 = 5/2)
  (h_sum2 : a 2 + a 4 = 5/4) :
  ∀ n : ℕ, a n = 2^(2-n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1169_116990


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1169_116904

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define permutation
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem problem_1 : (binomial 100 2 + binomial 100 97) / (permutation 101 3) = 1 / 6 := by
  sorry

theorem problem_2 : (Finset.sum (Finset.range 8) (λ i => binomial (i + 3) 3)) = 330 := by
  sorry

theorem problem_3 (n m : ℕ) (h : m ≤ n) : 
  (binomial (n + 1) m / binomial n m) - (binomial n (n - m + 1) / binomial n (n - m)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1169_116904


namespace NUMINAMATH_CALUDE_additional_lawn_needed_l1169_116983

/-- The amount LaKeisha charges per square foot of lawn mowed -/
def lawn_rate : ℚ := 1 / 10

/-- The amount LaKeisha charges per linear foot of hedge trimmed -/
def hedge_rate : ℚ := 1 / 20

/-- The amount LaKeisha charges per square foot of leaves raked -/
def rake_rate : ℚ := 1 / 50

/-- The cost of the book set -/
def book_cost : ℚ := 375

/-- The number of lawns LaKeisha has mowed -/
def lawns_mowed : ℕ := 5

/-- The length of each lawn -/
def lawn_length : ℕ := 30

/-- The width of each lawn -/
def lawn_width : ℕ := 20

/-- The number of linear feet of hedges LaKeisha has trimmed -/
def hedges_trimmed : ℕ := 100

/-- The number of square feet of leaves LaKeisha has raked -/
def leaves_raked : ℕ := 500

/-- The additional square feet of lawn LaKeisha needs to mow -/
def additional_lawn : ℕ := 600

theorem additional_lawn_needed :
  (book_cost - (lawn_rate * (lawns_mowed * lawn_length * lawn_width : ℚ) +
    hedge_rate * hedges_trimmed +
    rake_rate * leaves_raked)) / lawn_rate = additional_lawn := by sorry

end NUMINAMATH_CALUDE_additional_lawn_needed_l1169_116983


namespace NUMINAMATH_CALUDE_area_of_figure_l1169_116998

/-- Given a figure F in the plane of one face of a dihedral angle,
    S is the area of its orthogonal projection onto the other face,
    Q is the area of its orthogonal projection onto the bisector plane.
    This theorem proves that the area T of figure F is equal to (1/2)(√(S² + 8Q²) - S). -/
theorem area_of_figure (S Q : ℝ) (hS : S > 0) (hQ : Q > 0) :
  ∃ T : ℝ, T = (1/2) * (Real.sqrt (S^2 + 8*Q^2) - S) ∧ T > 0 := by
sorry

end NUMINAMATH_CALUDE_area_of_figure_l1169_116998


namespace NUMINAMATH_CALUDE_parabola_translation_l1169_116941

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ

/-- Represents a translation in the xy-plane -/
structure Translation where
  horizontal : ℝ
  vertical : ℝ

/-- Applies a translation to a parabola -/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { equation := fun x => p.equation (x - t.horizontal) + t.vertical }

theorem parabola_translation :
  let original : Parabola := { equation := fun x => x^2 }
  let translation : Translation := { horizontal := 3, vertical := -4 }
  let transformed := apply_translation original translation
  ∀ x, transformed.equation x = (x + 3)^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1169_116941


namespace NUMINAMATH_CALUDE_square_difference_equals_648_l1169_116977

theorem square_difference_equals_648 : (36 + 9)^2 - (9^2 + 36^2) = 648 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_648_l1169_116977


namespace NUMINAMATH_CALUDE_art_interest_group_end_time_l1169_116947

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.minutes + m
  let newHours := (t.hours + totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

theorem art_interest_group_end_time :
  let start_time : Time := ⟨15, 20, by sorry, by sorry⟩
  let duration : Nat := 50
  addMinutes start_time duration = ⟨16, 10, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_art_interest_group_end_time_l1169_116947


namespace NUMINAMATH_CALUDE_max_oranges_removal_l1169_116968

/-- A triangular grid of length n -/
structure TriangularGrid (n : ℕ) where
  (n_pos : 0 < n)
  (n_not_div_3 : ¬ 3 ∣ n)

/-- The total number of oranges in a triangular grid -/
def totalOranges (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- A good triple of oranges -/
structure GoodTriple (n : ℕ) where
  (isValid : Bool)

/-- The maximum number of oranges that can be removed -/
def maxRemovableOranges (n : ℕ) : ℕ := totalOranges n - 3

theorem max_oranges_removal (n : ℕ) (grid : TriangularGrid n) :
  maxRemovableOranges n = totalOranges n - 3 := by sorry

end NUMINAMATH_CALUDE_max_oranges_removal_l1169_116968


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1169_116986

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (sum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1169_116986


namespace NUMINAMATH_CALUDE_jack_coffee_batch_size_l1169_116993

/-- Proves that Jack makes 1.5 gallons of cold brew coffee in each batch given the conditions --/
theorem jack_coffee_batch_size :
  let coffee_per_2days : ℝ := 96  -- ounces
  let days : ℝ := 24
  let hours_per_batch : ℝ := 20
  let total_hours : ℝ := 120
  let ounces_per_gallon : ℝ := 128
  
  let total_coffee := (days / 2) * coffee_per_2days
  let total_gallons := total_coffee / ounces_per_gallon
  let num_batches := total_hours / hours_per_batch
  let gallons_per_batch := total_gallons / num_batches
  
  gallons_per_batch = 1.5 := by sorry

end NUMINAMATH_CALUDE_jack_coffee_batch_size_l1169_116993


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_zero_l1169_116959

theorem sum_mod_thirteen_zero : (9023 + 9024 + 9025 + 9026) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_zero_l1169_116959


namespace NUMINAMATH_CALUDE_sector_angle_l1169_116958

/-- Given a circular sector with perimeter 8 and area 4, its central angle is 2 radians. -/
theorem sector_angle (R : ℝ) (α : ℝ) (h1 : 2 * R + R * α = 8) (h2 : 1/2 * α * R^2 = 4) : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1169_116958


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1169_116900

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 3 * y - 2 = 0) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (a b : ℝ), a + 3 * b - 2 = 0 → 2^a + 8^b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1169_116900


namespace NUMINAMATH_CALUDE_notebook_cost_l1169_116910

theorem notebook_cost (total_students : Nat) (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat) :
  total_students = 42 →
  buyers > total_students / 2 →
  notebooks_per_student.Prime →
  cost_per_notebook > notebooks_per_student →
  buyers * notebooks_per_student * cost_per_notebook = 2310 →
  cost_per_notebook = 22 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1169_116910


namespace NUMINAMATH_CALUDE_max_daily_profit_l1169_116915

/-- Represents the daily profit function for a store selling football souvenir books. -/
def daily_profit (x : ℝ) : ℝ :=
  (x - 40) * (-10 * x + 740)

/-- Theorem stating the maximum daily profit and the corresponding selling price. -/
theorem max_daily_profit :
  let cost_price : ℝ := 40
  let initial_price : ℝ := 44
  let initial_sales : ℝ := 300
  let price_range : Set ℝ := {x | 44 ≤ x ∧ x ≤ 52}
  let sales_decrease_rate : ℝ := 10
  ∃ (max_price : ℝ), max_price ∈ price_range ∧
    ∀ (x : ℝ), x ∈ price_range →
      daily_profit x ≤ daily_profit max_price ∧
      daily_profit max_price = 2640 ∧
      max_price = 52 :=
by sorry


end NUMINAMATH_CALUDE_max_daily_profit_l1169_116915


namespace NUMINAMATH_CALUDE_rotation_result_l1169_116967

/-- Rotation of a vector about the origin -/
def rotate90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Check if a vector passes through the y-axis -/
def passesYAxis (v : ℝ × ℝ × ℝ) : Prop := sorry

theorem rotation_result :
  let v₀ : ℝ × ℝ × ℝ := (2, 1, 1)
  let v₁ := rotate90 v₀
  passesYAxis v₁ →
  v₁ = (Real.sqrt (6/11), -3 * Real.sqrt (6/11), Real.sqrt (6/11)) :=
by sorry

end NUMINAMATH_CALUDE_rotation_result_l1169_116967


namespace NUMINAMATH_CALUDE_power_equation_l1169_116976

theorem power_equation (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l1169_116976


namespace NUMINAMATH_CALUDE_smallest_three_digit_twice_in_pascal_l1169_116928

/-- Represents a position in Pascal's triangle by row and column -/
structure PascalPosition where
  row : Nat
  col : Nat
  h : col ≤ row

/-- Returns the value at a given position in Pascal's triangle -/
def pascal_value (pos : PascalPosition) : Nat :=
  sorry

/-- Predicate to check if a number appears at least twice in Pascal's triangle -/
def appears_twice (n : Nat) : Prop :=
  ∃ (pos1 pos2 : PascalPosition), pos1 ≠ pos2 ∧ pascal_value pos1 = n ∧ pascal_value pos2 = n

/-- The smallest three-digit number is 100 -/
def smallest_three_digit : Nat := 100

theorem smallest_three_digit_twice_in_pascal :
  (appears_twice smallest_three_digit) ∧
  (∀ n : Nat, n < smallest_three_digit → ¬(appears_twice n ∧ n ≥ 100)) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_twice_in_pascal_l1169_116928


namespace NUMINAMATH_CALUDE_tank_capacity_l1169_116905

theorem tank_capacity : 
  ∀ (initial_fraction final_fraction added_amount total_capacity : ℚ),
  initial_fraction = 1/4 →
  final_fraction = 2/3 →
  added_amount = 160 →
  (final_fraction - initial_fraction) * total_capacity = added_amount →
  total_capacity = 384 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l1169_116905


namespace NUMINAMATH_CALUDE_video_recorder_price_l1169_116975

/-- Given a wholesale cost, markup percentage, and discount percentage,
    calculate the final price after markup and discount. -/
def finalPrice (wholesaleCost markup discount : ℝ) : ℝ :=
  wholesaleCost * (1 + markup) * (1 - discount)

/-- Theorem stating that for a video recorder with a $200 wholesale cost,
    20% markup, and 25% employee discount, the final price is $180. -/
theorem video_recorder_price :
  finalPrice 200 0.20 0.25 = 180 := by
  sorry

end NUMINAMATH_CALUDE_video_recorder_price_l1169_116975


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l1169_116971

/-- Represents a rectangular yard with flower beds -/
structure FlowerYard where
  /-- Length of the yard -/
  length : ℝ
  /-- Width of the yard -/
  width : ℝ
  /-- Radius of the circular flower bed -/
  circle_radius : ℝ
  /-- Length of the shorter parallel side of the trapezoidal remainder -/
  trapezoid_short_side : ℝ
  /-- Length of the longer parallel side of the trapezoidal remainder -/
  trapezoid_long_side : ℝ

/-- Theorem stating the fraction of the yard occupied by flower beds -/
theorem flower_bed_fraction (yard : FlowerYard) 
  (h1 : yard.trapezoid_short_side = 20)
  (h2 : yard.trapezoid_long_side = 40)
  (h3 : yard.circle_radius = 2)
  (h4 : yard.length = yard.trapezoid_long_side)
  (h5 : yard.width = (yard.trapezoid_long_side - yard.trapezoid_short_side) / 2) :
  (100 + 4 * Real.pi) / 400 = 
    ((yard.trapezoid_long_side - yard.trapezoid_short_side)^2 / 4 + yard.circle_radius^2 * Real.pi) / 
    (yard.length * yard.width) :=
by sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l1169_116971


namespace NUMINAMATH_CALUDE_least_k_divisible_by_480_l1169_116903

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem least_k_divisible_by_480 :
  ∃ k : ℕ+, (k : ℕ) = 101250 ∧
    is_divisible (k^4) 480 ∧
    ∀ m : ℕ+, m < k → ¬is_divisible (m^4) 480 := by
  sorry

end NUMINAMATH_CALUDE_least_k_divisible_by_480_l1169_116903


namespace NUMINAMATH_CALUDE_initial_typists_count_l1169_116923

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 60

/-- The number of typists in the second group -/
def second_group_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 270

/-- The time ratio between 1 hour and 20 minutes -/
def time_ratio : ℚ := 3

theorem initial_typists_count :
  initial_typists * second_group_typists * letters_20min * time_ratio = letters_1hour * initial_typists * time_ratio :=
sorry

end NUMINAMATH_CALUDE_initial_typists_count_l1169_116923


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_conditions_l1169_116991

/-- A function that replaces a digit at position k with zero in a natural number n -/
def replace_digit_with_zero (n : ℕ) (k : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with zero -/
def ends_with_zero (n : ℕ) : Prop := sorry

/-- A function that counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ := sorry

/-- The main theorem stating that there are exactly 7 numbers satisfying the conditions -/
theorem exactly_seven_numbers_satisfy_conditions : 
  ∃! (s : Finset ℕ), 
    (s.card = 7) ∧ 
    (∀ n ∈ s, 
      ¬ends_with_zero n ∧ 
      ∃ k, k < digit_count n ∧ 
        9 * replace_digit_with_zero n k = n) :=
by sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_conditions_l1169_116991


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l1169_116933

-- Define the propositions p and q
def p (m : ℝ) : Prop := 1/4 < m ∧ m < 1

def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_q :
  (∀ m : ℝ, p m → q m) ∧ 
  (∃ m : ℝ, q m ∧ ¬p m) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l1169_116933


namespace NUMINAMATH_CALUDE_quadratic_solution_values_second_quadratic_solution_set_l1169_116961

-- Definition for the quadratic inequality
def quadratic_inequality (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + 3 * x - 2 > 0

-- Definition for the solution set
def solution_set (n : ℝ) (x : ℝ) : Prop :=
  n < x ∧ x < 2

-- Theorem for the first part of the problem
theorem quadratic_solution_values :
  (∀ x, quadratic_inequality m x ↔ solution_set n x) →
  m = -1 ∧ n = 1 :=
sorry

-- Definition for the second quadratic inequality
def second_quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 + (a - 1) * x - a > 0

-- Theorem for the second part of the problem
theorem second_quadratic_solution_set (a : ℝ) :
  (a < -1 → ∀ x, second_quadratic_inequality a x ↔ (x > 1 ∨ x < -a)) ∧
  (a = -1 → ∀ x, second_quadratic_inequality a x ↔ x ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_values_second_quadratic_solution_set_l1169_116961


namespace NUMINAMATH_CALUDE_equal_interest_principal_second_amount_calculation_l1169_116964

/-- Given two investments with equal interest, calculate the principal of the second investment -/
theorem equal_interest_principal (p₁ r₁ t₁ r₂ t₂ : ℚ) (hp₁ : p₁ > 0) (hr₁ : r₁ > 0) (ht₁ : t₁ > 0) (hr₂ : r₂ > 0) (ht₂ : t₂ > 0) :
  p₁ * r₁ * t₁ = (p₁ * r₁ * t₁ / (r₂ * t₂)) * r₂ * t₂ :=
by sorry

/-- The second amount that produces the same interest as Rs 200 at 10% for 12 years, when invested at 12% for 5 years, is Rs 400 -/
theorem second_amount_calculation :
  let p₁ : ℚ := 200
  let r₁ : ℚ := 10 / 100
  let t₁ : ℚ := 12
  let r₂ : ℚ := 12 / 100
  let t₂ : ℚ := 5
  (p₁ * r₁ * t₁ / (r₂ * t₂)) = 400 :=
by sorry

end NUMINAMATH_CALUDE_equal_interest_principal_second_amount_calculation_l1169_116964


namespace NUMINAMATH_CALUDE_expected_rainfall_is_50_4_l1169_116978

/-- Weather forecast probabilities and rainfall amounts -/
structure WeatherForecast where
  days : ℕ
  prob_sun : ℝ
  prob_rain_3 : ℝ
  prob_rain_8 : ℝ
  amount_rain_3 : ℝ
  amount_rain_8 : ℝ

/-- Expected total rainfall over the forecast period -/
def expectedTotalRainfall (forecast : WeatherForecast) : ℝ :=
  forecast.days * (forecast.prob_rain_3 * forecast.amount_rain_3 + 
                   forecast.prob_rain_8 * forecast.amount_rain_8)

/-- Theorem: The expected total rainfall for the given forecast is 50.4 inches -/
theorem expected_rainfall_is_50_4 (forecast : WeatherForecast) 
  (h1 : forecast.days = 14)
  (h2 : forecast.prob_sun = 0.3)
  (h3 : forecast.prob_rain_3 = 0.4)
  (h4 : forecast.prob_rain_8 = 0.3)
  (h5 : forecast.amount_rain_3 = 3)
  (h6 : forecast.amount_rain_8 = 8)
  (h7 : forecast.prob_sun + forecast.prob_rain_3 + forecast.prob_rain_8 = 1) :
  expectedTotalRainfall forecast = 50.4 := by
  sorry

#eval expectedTotalRainfall { 
  days := 14, 
  prob_sun := 0.3, 
  prob_rain_3 := 0.4, 
  prob_rain_8 := 0.3, 
  amount_rain_3 := 3, 
  amount_rain_8 := 8 
}

end NUMINAMATH_CALUDE_expected_rainfall_is_50_4_l1169_116978


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l1169_116965

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * triangle_base →
  triangle_base = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l1169_116965


namespace NUMINAMATH_CALUDE_election_loss_calculation_l1169_116948

theorem election_loss_calculation (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8000 →
  candidate_percentage = 1/4 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 4000 := by
  sorry

end NUMINAMATH_CALUDE_election_loss_calculation_l1169_116948


namespace NUMINAMATH_CALUDE_daily_reading_goal_l1169_116909

def september_days : ℕ := 30
def total_pages : ℕ := 600
def unavailable_days : ℕ := 10
def flight_day_pages : ℕ := 100

def available_days : ℕ := september_days - unavailable_days - 1
def remaining_pages : ℕ := total_pages - flight_day_pages

theorem daily_reading_goal :
  ∃ (pages_per_day : ℕ),
    pages_per_day * available_days ≥ remaining_pages ∧
    pages_per_day = 27 := by
  sorry

end NUMINAMATH_CALUDE_daily_reading_goal_l1169_116909


namespace NUMINAMATH_CALUDE_weaver_productivity_l1169_116999

/-- Given that 16 weavers can weave 64 mats in 16 days at a constant rate,
    prove that 4 weavers can weave 16 mats in 4 days at the same rate. -/
theorem weaver_productivity 
  (rate : ℝ) -- The constant rate of weaving (mats per weaver per day)
  (h1 : 16 * rate * 16 = 64) -- 16 weavers can weave 64 mats in 16 days
  : 4 * rate * 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_weaver_productivity_l1169_116999


namespace NUMINAMATH_CALUDE_opposite_of_four_l1169_116982

-- Define the concept of opposite (additive inverse)
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_four : opposite 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_four_l1169_116982


namespace NUMINAMATH_CALUDE_certain_number_proof_l1169_116980

theorem certain_number_proof : ∃ n : ℕ, n = 213 * 16 ∧ n = 3408 := by
  -- Given condition: 0.016 * 2.13 = 0.03408
  have h : (0.016 : ℝ) * 2.13 = 0.03408 := by sorry
  
  -- Proof that 213 * 16 = 3408
  sorry


end NUMINAMATH_CALUDE_certain_number_proof_l1169_116980


namespace NUMINAMATH_CALUDE_wire_cutting_l1169_116970

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 →
  ratio = 2 / 5 →
  shorter_piece + shorter_piece / ratio = total_length →
  shorter_piece = 120 / 7 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1169_116970


namespace NUMINAMATH_CALUDE_inequality_transformation_l1169_116988

theorem inequality_transformation (a b c : ℝ) :
  (b / (a^2 + 1) > c / (a^2 + 1)) → b > c := by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l1169_116988


namespace NUMINAMATH_CALUDE_master_craftsman_production_l1169_116930

/-- The number of parts manufactured by the master craftsman during the shift -/
def total_parts : ℕ := 210

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 3/2

theorem master_craftsman_production :
  ∃ (remaining_parts : ℕ),
    remaining_parts / first_hour_parts - remaining_parts / (first_hour_parts + rate_increase) = time_saved ∧
    total_parts = first_hour_parts + remaining_parts :=
by sorry

end NUMINAMATH_CALUDE_master_craftsman_production_l1169_116930


namespace NUMINAMATH_CALUDE_benny_savings_theorem_l1169_116946

/-- The amount of money Benny adds to his piggy bank in January -/
def january_savings : ℕ := 19

/-- The amount of money Benny adds to his piggy bank in February -/
def february_savings : ℕ := january_savings

/-- The amount of money Benny adds to his piggy bank in March -/
def march_savings : ℕ := 8

/-- The total amount of money in Benny's piggy bank by the end of March -/
def total_savings : ℕ := january_savings + february_savings + march_savings

/-- Theorem stating that the total amount in Benny's piggy bank by the end of March is $46 -/
theorem benny_savings_theorem : total_savings = 46 := by
  sorry

end NUMINAMATH_CALUDE_benny_savings_theorem_l1169_116946


namespace NUMINAMATH_CALUDE_dot_product_v_w_l1169_116972

def v : Fin 3 → ℝ := ![(-5 : ℝ), 2, -3]
def w : Fin 3 → ℝ := ![7, -4, 6]

theorem dot_product_v_w :
  (Finset.univ.sum fun i => v i * w i) = -61 := by sorry

end NUMINAMATH_CALUDE_dot_product_v_w_l1169_116972


namespace NUMINAMATH_CALUDE_share_multiple_l1169_116925

theorem share_multiple (a b c k : ℚ) : 
  a + b + c = 585 →
  4 * a = 6 * b →
  4 * a = k * c →
  c = 260 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_share_multiple_l1169_116925


namespace NUMINAMATH_CALUDE_safe_journey_possible_l1169_116981

-- Define the duration of various segments of the journey
def road_duration : ℕ := 4
def trail_duration : ℕ := 4

-- Define the eruption patterns of the craters
def crater1_eruption : ℕ := 1
def crater1_silence : ℕ := 17
def crater2_eruption : ℕ := 1
def crater2_silence : ℕ := 9

-- Define the total journey time
def total_journey_time : ℕ := 2 * (road_duration + trail_duration)

-- Define a function to check if a given time is safe for travel
def is_safe_time (t : ℕ) : Prop :=
  let crater1_cycle := crater1_eruption + crater1_silence
  let crater2_cycle := crater2_eruption + crater2_silence
  ∀ i : ℕ, i < total_journey_time →
    (((t + i) % crater1_cycle ≥ crater1_eruption) ∨ (i < road_duration ∨ i ≥ road_duration + 2 * trail_duration)) ∧
    (((t + i) % crater2_cycle ≥ crater2_eruption) ∨ (i < road_duration ∨ i ≥ road_duration + trail_duration))

-- Theorem stating that a safe journey is possible
theorem safe_journey_possible : ∃ t : ℕ, is_safe_time t :=
sorry

end NUMINAMATH_CALUDE_safe_journey_possible_l1169_116981


namespace NUMINAMATH_CALUDE_no_common_root_for_rational_coefficients_l1169_116926

theorem no_common_root_for_rational_coefficients :
  ∀ (a b : ℚ), ¬∃ (x : ℂ), (x^5 - x - 1 = 0) ∧ (x^2 + a*x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_for_rational_coefficients_l1169_116926


namespace NUMINAMATH_CALUDE_zinc_copper_ratio_in_mixture_l1169_116943

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the composition of a metal mixture -/
structure MetalMixture where
  totalWeight : ℝ
  zincWeight : ℝ

/-- Calculates the ratio of zinc to copper in a metal mixture -/
def zincCopperRatio (mixture : MetalMixture) : Ratio :=
  sorry

theorem zinc_copper_ratio_in_mixture :
  let mixture : MetalMixture := { totalWeight := 70, zincWeight := 31.5 }
  (zincCopperRatio mixture).numerator = 9 ∧
  (zincCopperRatio mixture).denominator = 11 :=
by sorry

end NUMINAMATH_CALUDE_zinc_copper_ratio_in_mixture_l1169_116943


namespace NUMINAMATH_CALUDE_parabola_equation_l1169_116949

/-- A parabola with focus F and a line passing through it -/
structure ParabolaWithLine where
  /-- Parameter of the parabola (p > 0) -/
  p : ℝ
  /-- The line passes through the focus F -/
  passesThroughFocus : Bool
  /-- The line is parallel to one of the asymptotes of x^2 - y^2/8 = 1 -/
  parallelToAsymptote : Bool
  /-- The line intersects the parabola at points A and B -/
  intersectsParabola : Bool
  /-- |AF| > |BF| -/
  afLongerThanBf : Bool
  /-- |AF| = 3 -/
  afLength : ℝ
  hP : p > 0
  hAf : afLength = 3

/-- The theorem stating that under given conditions, the parabola equation is y^2 = 4x -/
theorem parabola_equation (pwl : ParabolaWithLine) : pwl.p = 2 := by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l1169_116949


namespace NUMINAMATH_CALUDE_greatest_k_for_100_power_dividing_50_factorial_l1169_116942

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def highest_power_of_2 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n / 2) + highest_power_of_2 (n / 2)

def highest_power_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n / 5) + highest_power_of_5 (n / 5)

theorem greatest_k_for_100_power_dividing_50_factorial :
  (∃ k : ℕ, k = 6 ∧
    ∀ m : ℕ, (100 ^ m : ℕ) ∣ factorial 50 → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_100_power_dividing_50_factorial_l1169_116942


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1169_116918

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) :
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1169_116918


namespace NUMINAMATH_CALUDE_train_length_l1169_116952

theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 →
  crossing_time = 45 →
  train_speed = 47.99999999999999 →
  (train_speed * crossing_time) - bridge_length = 1860 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1169_116952


namespace NUMINAMATH_CALUDE_carolines_socks_l1169_116963

/-- Given Caroline's sock inventory changes, calculate how many pairs she received as a gift. -/
theorem carolines_socks (initial : ℕ) (lost : ℕ) (donation_fraction : ℚ) (purchased : ℕ) (final : ℕ) : 
  initial = 40 →
  lost = 4 →
  donation_fraction = 2/3 →
  purchased = 10 →
  final = 25 →
  final = initial - lost - (initial - lost) * donation_fraction + purchased + (final - (initial - lost - (initial - lost) * donation_fraction + purchased)) :=
by sorry

end NUMINAMATH_CALUDE_carolines_socks_l1169_116963


namespace NUMINAMATH_CALUDE_a_is_increasing_l1169_116924

def a (n : ℕ) : ℚ := (n - 1) / (n + 1)

theorem a_is_increasing : ∀ k j : ℕ, k > j → k ≥ 1 → j ≥ 1 → a k > a j := by
  sorry

end NUMINAMATH_CALUDE_a_is_increasing_l1169_116924


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1169_116919

/-- The number of ways three people can sit in a row of six chairs -/
def seating_arrangements : ℕ := 6 * 5 * 4

/-- Theorem stating that the number of seating arrangements is 120 -/
theorem seating_arrangements_count : seating_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1169_116919


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l1169_116973

open Set

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | (x - 2) / (x + 1) ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the interval (-1, 2]
def interval : Set ℝ := Ioc (-1) 2

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l1169_116973


namespace NUMINAMATH_CALUDE_jacob_breakfast_calories_l1169_116940

theorem jacob_breakfast_calories 
  (daily_limit : ℕ) 
  (lunch_calories : ℕ) 
  (dinner_calories : ℕ) 
  (exceeded_calories : ℕ) 
  (h1 : daily_limit = 1800)
  (h2 : lunch_calories = 900)
  (h3 : dinner_calories = 1100)
  (h4 : exceeded_calories = 600) :
  daily_limit + exceeded_calories - (lunch_calories + dinner_calories) = 400 := by
sorry

end NUMINAMATH_CALUDE_jacob_breakfast_calories_l1169_116940


namespace NUMINAMATH_CALUDE_iAmALiar_is_false_iAmALiar_identifies_spy_l1169_116951

/-- Represents a person who can be either a knight or a knave -/
inductive Person
| Knight
| Knave

/-- Represents a statement that can be made by a person -/
def Statement := Person → Prop

/-- A function that determines if a statement is true for a given person -/
def isTruthful (s : Statement) (p : Person) : Prop :=
  match p with
  | Person.Knight => s p
  | Person.Knave => ¬(s p)

/-- The statement "I am a liar" -/
def iAmALiar : Statement :=
  fun p => p = Person.Knave

/-- Theorem: The statement "I am a liar" is false for both knights and knaves -/
theorem iAmALiar_is_false (p : Person) : ¬(isTruthful iAmALiar p) := by
  sorry

/-- Theorem: The statement "I am a liar" immediately identifies the speaker as a spy -/
theorem iAmALiar_identifies_spy (p : Person) : 
  isTruthful iAmALiar p → False := by
  sorry

end NUMINAMATH_CALUDE_iAmALiar_is_false_iAmALiar_identifies_spy_l1169_116951
