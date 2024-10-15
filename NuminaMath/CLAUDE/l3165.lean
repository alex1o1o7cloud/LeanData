import Mathlib

namespace NUMINAMATH_CALUDE_floor_equation_solution_l3165_316508

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - (1/2 : ℝ)⌋ = ⌊x + 3⌋ ↔ 3.5 ≤ x ∧ x < 4.5 := by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3165_316508


namespace NUMINAMATH_CALUDE_lesser_number_l3165_316545

theorem lesser_number (x y : ℝ) (sum_eq : x + y = 60) (diff_eq : x - y = 10) : 
  min x y = 25 := by sorry

end NUMINAMATH_CALUDE_lesser_number_l3165_316545


namespace NUMINAMATH_CALUDE_common_point_exists_l3165_316572

-- Define the basic structures
structure Ray where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the given conditions
def intersect_point : ℝ × ℝ := sorry

def ray1 : Ray := sorry
def ray2 : Ray := sorry

def a : ℝ := sorry
axiom a_positive : 0 < a

-- Define the circle properties
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop := sorry

def circle_intersects_ray (c : Circle) (r : Ray) : ℝ × ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem common_point_exists :
  ∀ (c : Circle),
    circle_passes_through c intersect_point ∧
    ∃ (B C : ℝ × ℝ),
      B = circle_intersects_ray c ray1 ∧
      C = circle_intersects_ray c ray2 ∧
      distance intersect_point B + distance intersect_point C = a
    →
    ∃ (Z : ℝ × ℝ), Z ≠ intersect_point ∧
      ∀ (c' : Circle),
        circle_passes_through c' intersect_point ∧
        ∃ (B' C' : ℝ × ℝ),
          B' = circle_intersects_ray c' ray1 ∧
          C' = circle_intersects_ray c' ray2 ∧
          distance intersect_point B' + distance intersect_point C' = a
        →
        circle_passes_through c' Z :=
sorry

end NUMINAMATH_CALUDE_common_point_exists_l3165_316572


namespace NUMINAMATH_CALUDE_arrangements_eq_18_l3165_316598

/-- Represents the number of people in the lineup --/
def n : ℕ := 5

/-- Represents the possible positions for Person A --/
def A_positions : Set ℕ := {1, 2}

/-- Represents the possible positions for Person B --/
def B_positions : Set ℕ := {2, 3}

/-- The number of ways to arrange n people with the given constraints --/
def num_arrangements (n : ℕ) (A_pos : Set ℕ) (B_pos : Set ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of arrangements is 18 --/
theorem arrangements_eq_18 :
  num_arrangements n A_positions B_positions = 18 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_eq_18_l3165_316598


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3165_316558

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 + (y + 1/(2*x))^2 ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 + (y + 1/(2*x))^2 = 3 + 2 * Real.sqrt 2 ↔ 
  x = Real.sqrt 2 / 2 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3165_316558


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3165_316571

/-- Given a quadratic expression 8k^2 - 12k + 20, when rewritten in the form d(k + r)^2 + s
    where d, r, and s are constants, prove that r + s = 14.75 -/
theorem quadratic_rewrite_sum (k : ℝ) : 
  ∃ (d r s : ℝ), (∀ k, 8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ r + s = 14.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3165_316571


namespace NUMINAMATH_CALUDE_room_freezer_temp_difference_l3165_316525

-- Define the temperatures
def freezer_temp : Int := -4
def room_temp : Int := 18

-- Define the temperature difference function
def temp_difference (room : Int) (freezer : Int) : Int :=
  room - freezer

-- Theorem to prove
theorem room_freezer_temp_difference :
  temp_difference room_temp freezer_temp = 22 := by
  sorry

end NUMINAMATH_CALUDE_room_freezer_temp_difference_l3165_316525


namespace NUMINAMATH_CALUDE_drew_marbles_difference_l3165_316576

theorem drew_marbles_difference (drew_initial : ℕ) (marcus_initial : ℕ) (john_initial : ℕ) 
  (h1 : marcus_initial = 45)
  (h2 : john_initial = 70)
  (h3 : ∃ x : ℕ, drew_initial / 4 + marcus_initial = x ∧ drew_initial / 8 + john_initial = x) :
  drew_initial - marcus_initial = 155 :=
by sorry

end NUMINAMATH_CALUDE_drew_marbles_difference_l3165_316576


namespace NUMINAMATH_CALUDE_wind_velocity_problem_l3165_316569

/-- Represents the relationship between pressure, area, and velocity -/
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

theorem wind_velocity_problem (k : ℝ) :
  pressure_relation k 2 8 = 4 →
  pressure_relation k 4.5 (40/3) = 25 :=
by sorry

end NUMINAMATH_CALUDE_wind_velocity_problem_l3165_316569


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3165_316563

theorem inequality_equivalence (x : ℝ) : 
  Real.sqrt ((1 / (2 - x) + 1) ^ 2) ≥ 2 ↔ 
  (x ≥ 1 ∧ x < 2) ∨ (x > 2 ∧ x ≤ 7/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3165_316563


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3165_316541

/-- A polynomial of degree 3 with a parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + k * x - 12

/-- The divisor x - 3 -/
def g (x : ℝ) : ℝ := x - 3

/-- The potential divisor 3x^2 + 4 -/
def h (x : ℝ) : ℝ := 3 * x^2 + 4

theorem polynomial_divisibility (k : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, f k x = g x * q x) →
  (∃ r : ℝ → ℝ, ∀ x, f k x = h x * r x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3165_316541


namespace NUMINAMATH_CALUDE_ceiling_minus_y_l3165_316537

theorem ceiling_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - (y - ⌊y⌋) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_y_l3165_316537


namespace NUMINAMATH_CALUDE_min_value_problem_l3165_316546

theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3165_316546


namespace NUMINAMATH_CALUDE_erik_pie_amount_l3165_316543

theorem erik_pie_amount (frank_pie : ℝ) (erik_extra : ℝ) 
  (h1 : frank_pie = 0.3333333333333333)
  (h2 : erik_extra = 0.3333333333333333) :
  frank_pie + erik_extra = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_erik_pie_amount_l3165_316543


namespace NUMINAMATH_CALUDE_sum_47_58_base5_l3165_316522

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_47_58_base5 : toBase5 (47 + 58) = [4, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_47_58_base5_l3165_316522


namespace NUMINAMATH_CALUDE_production_equation_proof_l3165_316538

/-- Represents a furniture production scenario -/
structure ProductionScenario where
  total : ℕ              -- Total sets to produce
  increase : ℕ           -- Daily production increase
  days_saved : ℕ         -- Days saved due to increase
  original_rate : ℕ      -- Original daily production rate

/-- Theorem stating the correct equation for the production scenario -/
theorem production_equation_proof (s : ProductionScenario) 
  (h1 : s.total = 540)
  (h2 : s.increase = 2)
  (h3 : s.days_saved = 3) :
  (s.total : ℝ) / s.original_rate - (s.total : ℝ) / (s.original_rate + s.increase) = s.days_saved := by
  sorry

#check production_equation_proof

end NUMINAMATH_CALUDE_production_equation_proof_l3165_316538


namespace NUMINAMATH_CALUDE_third_root_unity_sum_l3165_316519

theorem third_root_unity_sum (z : ℂ) (h1 : z^3 - 1 = 0) (h2 : z ≠ 1) :
  z^100 + z^101 + z^102 + z^103 + z^104 = 0 := by
  sorry

end NUMINAMATH_CALUDE_third_root_unity_sum_l3165_316519


namespace NUMINAMATH_CALUDE_range_of_a_l3165_316529

theorem range_of_a (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 / x + 1 / y = 1) :
  (∀ a : ℝ, x + y + a > 0) ↔ ∀ a : ℝ, a > -3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3165_316529


namespace NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l3165_316584

theorem consecutive_product_plus_one_is_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l3165_316584


namespace NUMINAMATH_CALUDE_existence_of_solution_l3165_316591

theorem existence_of_solution (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3 * n^2 + 4 * n) 
  (hb : b ≤ 3 * n^2 + 4 * n) 
  (hc : c ≤ 3 * n^2 + 4 * n) : 
  ∃ (x y z : ℤ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    (abs x ≤ 2 * n) ∧ 
    (abs y ≤ 2 * n) ∧ 
    (abs z ≤ 2 * n) ∧ 
    (a * x + b * y + c * z = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_solution_l3165_316591


namespace NUMINAMATH_CALUDE_function_bounds_l3165_316515

theorem function_bounds (x : ℝ) : 
  0.95 ≤ (x^4 + x^2 + 5) / ((x^2 + 1)^2) ∧ (x^4 + x^2 + 5) / ((x^2 + 1)^2) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l3165_316515


namespace NUMINAMATH_CALUDE_log_inequality_l3165_316513

theorem log_inequality : ∀ x : ℝ, x > 0 → x + 1/x > 2 := by sorry

end NUMINAMATH_CALUDE_log_inequality_l3165_316513


namespace NUMINAMATH_CALUDE_min_packages_required_l3165_316549

/-- Represents a floor in the apartment building -/
inductive Floor
| First
| Second
| Third

/-- Calculates the number of times a specific digit appears on a floor -/
def digit_count (floor : Floor) (digit : ℕ) : ℕ :=
  match floor with
  | Floor.First => if digit = 1 then 52 else 0
  | Floor.Second => if digit = 2 then 52 else 0
  | Floor.Third => if digit = 3 then 52 else 0

/-- Theorem stating the minimum number of packages required -/
theorem min_packages_required : 
  (∀ (floor : Floor) (digit : ℕ), digit_count floor digit ≤ 52) ∧ 
  (∃ (floor : Floor) (digit : ℕ), digit_count floor digit = 52) → 
  (∀ n : ℕ, n < 52 → ¬(∀ (floor : Floor) (digit : ℕ), digit_count floor digit ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_min_packages_required_l3165_316549


namespace NUMINAMATH_CALUDE_calculation_proof_l3165_316510

theorem calculation_proof : 70 + 5 * 12 / (180 / 3) = 71 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3165_316510


namespace NUMINAMATH_CALUDE_min_sum_m_n_min_sum_is_three_l3165_316553

theorem min_sum_m_n (m n : ℕ+) (h : 32 * m = n ^ 5) : 
  ∀ (m' n' : ℕ+), 32 * m' = n' ^ 5 → m + n ≤ m' + n' :=
by
  sorry

theorem min_sum_is_three : 
  ∃ (m n : ℕ+), 32 * m = n ^ 5 ∧ m + n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_m_n_min_sum_is_three_l3165_316553


namespace NUMINAMATH_CALUDE_equation_solution_l3165_316595

theorem equation_solution : ∃ x : ℚ, (x - 7) / 2 - (1 + x) / 3 = 1 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3165_316595


namespace NUMINAMATH_CALUDE_sqrt_squared_2a_minus_1_l3165_316568

theorem sqrt_squared_2a_minus_1 (a : ℝ) (h : a ≥ (1/2 : ℝ)) :
  Real.sqrt ((2*a - 1)^2) = 2*a - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_squared_2a_minus_1_l3165_316568


namespace NUMINAMATH_CALUDE_max_y_rectangular_prism_l3165_316594

/-- The maximum value of y for a rectangular prism with volume 360 and integer dimensions x, y, z satisfying 1 < z < y < x -/
theorem max_y_rectangular_prism : 
  ∀ x y z : ℕ, 
  x * y * z = 360 → 
  1 < z → z < y → y < x → 
  y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_y_rectangular_prism_l3165_316594


namespace NUMINAMATH_CALUDE_athlete_arrangement_and_allocation_l3165_316507

/-- The number of male athletes -/
def num_male_athletes : ℕ := 4

/-- The number of female athletes -/
def num_female_athletes : ℕ := 3

/-- The total number of athletes -/
def total_athletes : ℕ := num_male_athletes + num_female_athletes

/-- The number of ways to arrange the athletes with all female athletes together -/
def arrangement_count : ℕ := (Nat.factorial (num_male_athletes + 1)) * (Nat.factorial num_female_athletes)

/-- The number of ways to allocate male athletes to two venues -/
def allocation_count : ℕ := Nat.choose num_male_athletes 1 + Nat.choose num_male_athletes 2

theorem athlete_arrangement_and_allocation :
  arrangement_count = 720 ∧ allocation_count = 10 := by sorry


end NUMINAMATH_CALUDE_athlete_arrangement_and_allocation_l3165_316507


namespace NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_independent_l3165_316555

-- Define the sample space
def S : Set (ℕ × ℕ) := {p | p.1 ∈ Finset.range 6 ∧ p.2 ∈ Finset.range 6}

-- Define events A, B, and C
def A : Set (ℕ × ℕ) := {p ∈ S | p.1 + p.2 = 7}
def B : Set (ℕ × ℕ) := {p ∈ S | Odd (p.1 * p.2)}
def C : Set (ℕ × ℕ) := {p ∈ S | p.1 > 3}

-- Define probability measure
noncomputable def P : Set (ℕ × ℕ) → ℝ := sorry

-- Theorem statements
theorem A_B_mutually_exclusive : A ∩ B = ∅ := by sorry

theorem A_C_independent : P (A ∩ C) = P A * P C := by sorry

end NUMINAMATH_CALUDE_A_B_mutually_exclusive_A_C_independent_l3165_316555


namespace NUMINAMATH_CALUDE_complex_modulus_l3165_316550

theorem complex_modulus (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = (i + 1) / (1 - i)^2 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3165_316550


namespace NUMINAMATH_CALUDE_total_cost_usd_l3165_316577

/-- The cost of items in British pounds and US dollars -/
def cost_in_usd (tea_gbp : ℝ) (scone_gbp : ℝ) (exchange_rate : ℝ) : ℝ :=
  (tea_gbp + scone_gbp) * exchange_rate

/-- Theorem: The total cost in USD for a tea and a scone is $10.80 -/
theorem total_cost_usd :
  cost_in_usd 5 3 1.35 = 10.80 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_usd_l3165_316577


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l3165_316532

-- Inequality 1: (x+1)^2 + 3(x+1) - 4 > 0
theorem inequality_one (x : ℝ) : 
  (x + 1)^2 + 3*(x + 1) - 4 > 0 ↔ x < -5 ∨ x > 0 := by sorry

-- Inequality 2: x^4 - 2x^2 + 1 > x^2 - 1
theorem inequality_two (x : ℝ) : 
  x^4 - 2*x^2 + 1 > x^2 - 1 ↔ x < -Real.sqrt 2 ∨ (-1 < x ∧ x < 1) ∨ x > Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l3165_316532


namespace NUMINAMATH_CALUDE_poles_for_given_plot_l3165_316523

/-- Calculates the number of poles needed for a side of a plot -/
def polesForSide (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing) + 1

/-- Represents a trapezoidal plot with given side lengths and pole spacings -/
structure TrapezoidalPlot where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  side4 : ℕ
  spacing1 : ℕ
  spacing2 : ℕ

/-- Calculates the total number of poles needed for a trapezoidal plot -/
def totalPoles (plot : TrapezoidalPlot) : ℕ :=
  polesForSide plot.side1 plot.spacing1 +
  polesForSide plot.side2 plot.spacing2 +
  polesForSide plot.side3 plot.spacing1 +
  polesForSide plot.side4 plot.spacing2

/-- The main theorem stating that the number of poles for the given plot is 40 -/
theorem poles_for_given_plot :
  let plot := TrapezoidalPlot.mk 60 30 50 40 5 4
  totalPoles plot = 40 := by
  sorry


end NUMINAMATH_CALUDE_poles_for_given_plot_l3165_316523


namespace NUMINAMATH_CALUDE_cubic_parabola_x_intercepts_l3165_316500

theorem cubic_parabola_x_intercepts :
  ∃! x : ℝ, x = -3 * 0^3 + 2 * 0^2 - 0 + 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_parabola_x_intercepts_l3165_316500


namespace NUMINAMATH_CALUDE_min_movie_audience_l3165_316512

/-- Represents the number of people in the movie theater -/
structure MovieTheater where
  adults : ℕ
  children : ℕ

/-- Conditions for the movie theater audience -/
class MovieTheaterConditions (t : MovieTheater) where
  adult_men : t.adults * 4 = t.adults * 5
  male_children : t.children * 2 = t.adults * 2
  boy_children : t.children * 1 = t.children * 5

/-- The theorem stating the minimum number of people in the movie theater -/
theorem min_movie_audience (t : MovieTheater) [MovieTheaterConditions t] :
  t.adults + t.children ≥ 55 := by
  sorry

#check min_movie_audience

end NUMINAMATH_CALUDE_min_movie_audience_l3165_316512


namespace NUMINAMATH_CALUDE_total_passengers_l3165_316520

theorem total_passengers (on_time : ℕ) (late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) :
  on_time + late = 14720 := by
  sorry

end NUMINAMATH_CALUDE_total_passengers_l3165_316520


namespace NUMINAMATH_CALUDE_test_score_calculation_l3165_316552

theorem test_score_calculation (total_marks : ℕ) (percentage : ℚ) : 
  total_marks = 50 → percentage = 80 / 100 → (percentage * total_marks : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_test_score_calculation_l3165_316552


namespace NUMINAMATH_CALUDE_probability_both_primary_l3165_316575

/-- Represents the types of schools in the area -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → Nat
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the number of schools selected in stratified sampling -/
def selectedSchools : SchoolType → Nat
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- The total number of schools selected -/
def totalSelected : Nat := 6

/-- The number of ways to choose 2 schools from the selected primary schools -/
def waysToChoosePrimary : Nat := 3

/-- The total number of ways to choose 2 schools from all selected schools -/
def totalWaysToChoose : Nat := 15

theorem probability_both_primary :
  (waysToChoosePrimary : Rat) / totalWaysToChoose = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_both_primary_l3165_316575


namespace NUMINAMATH_CALUDE_five_books_three_bins_l3165_316511

-- Define the Stirling number of the second kind
def stirling2 (n k : ℕ) : ℕ := sorry

-- State the theorem
theorem five_books_three_bins : stirling2 5 3 = 25 := by sorry

end NUMINAMATH_CALUDE_five_books_three_bins_l3165_316511


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3165_316501

/-- A color represented as a natural number -/
def Color := ℕ

/-- A point in the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x ≤ 12
  h_y : y ≤ 12

/-- A coloring of the grid -/
def GridColoring := GridPoint → Color

/-- A rectangle in the grid -/
structure Rectangle where
  x1 : ℕ
  y1 : ℕ
  x2 : ℕ
  y2 : ℕ
  h_x1 : x1 ≤ 12
  h_y1 : y1 ≤ 12
  h_x2 : x2 ≤ 12
  h_y2 : y2 ≤ 12
  h_distinct : (x1 ≠ x2 ∧ y1 ≠ y2) ∨ (x1 ≠ x2 ∧ y1 = y2) ∨ (x1 = x2 ∧ y1 ≠ y2)

/-- The theorem stating that there exists a monochromatic rectangle -/
theorem monochromatic_rectangle_exists (coloring : GridColoring) :
  ∃ (r : Rectangle) (c : Color),
    coloring ⟨r.x1, r.y1, r.h_x1, r.h_y1⟩ = c ∧
    coloring ⟨r.x1, r.y2, r.h_x1, r.h_y2⟩ = c ∧
    coloring ⟨r.x2, r.y1, r.h_x2, r.h_y1⟩ = c ∧
    coloring ⟨r.x2, r.y2, r.h_x2, r.h_y2⟩ = c :=
  sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3165_316501


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l3165_316533

/-- The parabolas y = (x - 2)^2 and x + 6 = (y + 1)^2 intersect at points that lie on a circle with radius squared 5/2 -/
theorem intersection_points_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ),
      (p.2 = (p.1 - 2)^2 ∧ p.1 + 6 = (p.2 + 1)^2) →
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l3165_316533


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3165_316521

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : ∃ R : ℝ,
  R > 0 ∧ R < 100 ∧ (P * R * 10) / 100 = P / 5 ∧ R = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3165_316521


namespace NUMINAMATH_CALUDE_factorial_ratio_42_40_l3165_316505

theorem factorial_ratio_42_40 : Nat.factorial 42 / Nat.factorial 40 = 1722 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_42_40_l3165_316505


namespace NUMINAMATH_CALUDE_edward_spent_five_on_supplies_l3165_316582

/-- Edward's lawn mowing business finances -/
def lawn_mowing_problem (spring_earnings summer_earnings final_amount : ℤ) : Prop :=
  let total_earnings := spring_earnings + summer_earnings
  let supplies_cost := total_earnings - final_amount
  supplies_cost = 5

/-- Theorem: Edward spent $5 on supplies -/
theorem edward_spent_five_on_supplies :
  lawn_mowing_problem 2 27 24 := by sorry

end NUMINAMATH_CALUDE_edward_spent_five_on_supplies_l3165_316582


namespace NUMINAMATH_CALUDE_intersection_q_complement_p_l3165_316531

-- Define the sets
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≥ 9}
def Q : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_q_complement_p :
  Q ∩ (U \ P) = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_q_complement_p_l3165_316531


namespace NUMINAMATH_CALUDE_restaurant_group_cost_l3165_316559

/-- Calculates the total cost for a group to eat at a restaurant where kids eat free. -/
def group_meal_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Proves that the total cost for a group of 11 people with 2 kids is $72,
    given that adult meals cost $8 and kids eat free. -/
theorem restaurant_group_cost :
  group_meal_cost 11 2 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_cost_l3165_316559


namespace NUMINAMATH_CALUDE_box_volume_count_l3165_316565

theorem box_volume_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun x => x > 2 ∧ (x + 3) * (x - 2) * (x^2 + 10) < 500) 
    (Finset.range 100)).card :=
by
  sorry

end NUMINAMATH_CALUDE_box_volume_count_l3165_316565


namespace NUMINAMATH_CALUDE_inverse_f_93_l3165_316589

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_93 : f⁻¹ 93 = (28 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_inverse_f_93_l3165_316589


namespace NUMINAMATH_CALUDE_equation_one_solution_l3165_316556

theorem equation_one_solution (x : ℝ) : 3 * x * (x - 1) = 1 - x → x = 1 ∨ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solution_l3165_316556


namespace NUMINAMATH_CALUDE_determinant_evaluation_l3165_316585

-- Define the matrix
def matrix (x y z : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
λ i j => match i, j with
  | 0, 0 => 1
  | 0, 1 => x
  | 0, 2 => y
  | 0, 3 => z
  | 1, 0 => 1
  | 1, 1 => x + y
  | 1, 2 => y
  | 1, 3 => z
  | 2, 0 => 1
  | 2, 1 => x
  | 2, 2 => x + y
  | 2, 3 => z
  | 3, 0 => 1
  | 3, 1 => x
  | 3, 2 => y
  | 3, 3 => x + y + z

theorem determinant_evaluation (x y z : ℝ) :
  Matrix.det (matrix x y z) = y * x^2 + y^2 * x := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l3165_316585


namespace NUMINAMATH_CALUDE_manicure_total_cost_l3165_316526

-- Define the cost of the manicure
def manicure_cost : ℝ := 30

-- Define the tip percentage
def tip_percentage : ℝ := 0.30

-- Define the function to calculate the total amount paid
def total_amount_paid (cost tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- Theorem to prove
theorem manicure_total_cost :
  total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end NUMINAMATH_CALUDE_manicure_total_cost_l3165_316526


namespace NUMINAMATH_CALUDE_parabola_vertex_l3165_316551

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 2 is (1, 2) -/
theorem parabola_vertex : 
  ∀ x : ℝ, parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3165_316551


namespace NUMINAMATH_CALUDE_polynomial_efficient_evaluation_l3165_316527

/-- The polynomial 6x^5+5x^4+4x^3+3x^2+2x+2002 can be evaluated using 5 multiplications and 5 additions -/
theorem polynomial_efficient_evaluation :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = 6*x^5 + 5*x^4 + 4*x^3 + 3*x^2 + 2*x + 2002) ∧
    (∃ (g : ℝ → ℝ) (a b c d e : ℝ → ℝ),
      (∀ x, f x = g x + 2002) ∧
      (∀ x, g x = (((a x * x + b x) * x + c x) * x + d x) * x + e x) ∧
      (∀ x, a x = 6*x + 5) ∧
      (∀ x, b x = 4) ∧
      (∀ x, c x = 3) ∧
      (∀ x, d x = 2) ∧
      (∀ x, e x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_efficient_evaluation_l3165_316527


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3165_316560

/-- Given a triangle with sides of lengths 3, 6, and x, where x is a solution to x^2 - 7x + 12 = 0
    and satisfies the triangle inequality, prove that the perimeter of the triangle is 13. -/
theorem triangle_perimeter (x : ℝ) : 
  x^2 - 7*x + 12 = 0 →
  x + 3 > 6 →
  x + 6 > 3 →
  3 + 6 + x = 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3165_316560


namespace NUMINAMATH_CALUDE_f_properties_l3165_316566

def f (x : ℝ) := x^3 - 12*x

theorem f_properties :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧ 
  (∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y) ∧ 
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ f (-2)) ∧
  (∀ x, f 2 ≤ f x) ∧
  (f (-2) = 16) ∧
  (f 2 = -16) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3165_316566


namespace NUMINAMATH_CALUDE_ellipse_param_sum_l3165_316528

/-- An ellipse with given foci and distance sum -/
structure Ellipse where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  distance_sum : ℝ

/-- Standard form parameters of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Calculate the parameters of the ellipse given its foci and distance sum -/
def calculate_ellipse_params (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem to be proved -/
theorem ellipse_param_sum (e : Ellipse) (p : EllipseParams) :
  e.f1 = (0, 1) →
  e.f2 = (6, 1) →
  e.distance_sum = 10 →
  p = calculate_ellipse_params e →
  p.h + p.k + p.a + p.b = 13 :=
sorry

end NUMINAMATH_CALUDE_ellipse_param_sum_l3165_316528


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3165_316504

theorem least_addition_for_divisibility :
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), (1056 + m) % 28 = 0 → m ≥ n) ∧
  (1056 + n) % 28 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3165_316504


namespace NUMINAMATH_CALUDE_students_present_l3165_316516

theorem students_present (total : ℕ) (absent_percent : ℚ) : 
  total = 100 → absent_percent = 14/100 → 
  (total : ℚ) * (1 - absent_percent) = 86 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l3165_316516


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3165_316506

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram in 2D space -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

theorem parallelogram_vertex_sum (ABCD : Parallelogram) : 
  ABCD.A = Point.mk (-1) 2 →
  ABCD.B = Point.mk 3 (-1) →
  ABCD.D = Point.mk 5 7 →
  isPerpendicular ABCD.A ABCD.B ABCD.B ABCD.C →
  ABCD.C.x + ABCD.C.y = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3165_316506


namespace NUMINAMATH_CALUDE_power_function_is_odd_l3165_316530

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ) (α : ℝ), ∀ x, f x = c * x^α

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem power_function_is_odd (α : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (α - 2) * x^α
  isPowerFunction f → isOddFunction f := by
  sorry


end NUMINAMATH_CALUDE_power_function_is_odd_l3165_316530


namespace NUMINAMATH_CALUDE_mcdonald_farm_eggs_l3165_316562

/-- Calculates the total number of eggs needed per month for a community -/
def total_eggs_per_month (saly_weekly : ℕ) (ben_weekly : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let ked_weekly := ben_weekly / 2
  let total_weekly := saly_weekly + ben_weekly + ked_weekly
  total_weekly * weeks_per_month

/-- Proves that the total eggs needed per month is 124 given the specific requirements -/
theorem mcdonald_farm_eggs : total_eggs_per_month 10 14 4 = 124 := by
  sorry

end NUMINAMATH_CALUDE_mcdonald_farm_eggs_l3165_316562


namespace NUMINAMATH_CALUDE_smallest_six_consecutive_number_max_six_consecutive_with_perfect_square_F_l3165_316542

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a FourDigitNumber has distinct non-zero digits -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  0 < a ∧ a ≤ 9 ∧
  0 < b ∧ b ≤ 9 ∧
  0 < c ∧ c ≤ 9 ∧
  0 < d ∧ d ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Checks if a FourDigitNumber is a "six-consecutive number" -/
def isSixConsecutive (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  (a + b) * (c + d) = 60

/-- Calculates F(M) for a FourDigitNumber -/
def F (n : FourDigitNumber) : Int :=
  let (a, b, c, d) := n
  (a * 10 + d) - (b * 10 + c) - ((a * 10 + c) - (b * 10 + d))

/-- Converts a FourDigitNumber to its integer representation -/
def toInt (n : FourDigitNumber) : Nat :=
  let (a, b, c, d) := n
  a * 1000 + b * 100 + c * 10 + d

theorem smallest_six_consecutive_number :
  ∃ (M : FourDigitNumber),
    isValidFourDigitNumber M ∧
    isSixConsecutive M ∧
    (∀ (N : FourDigitNumber),
      isValidFourDigitNumber N → isSixConsecutive N →
      toInt M ≤ toInt N) ∧
    toInt M = 1369 := by sorry

theorem max_six_consecutive_with_perfect_square_F :
  ∃ (N : FourDigitNumber),
    isValidFourDigitNumber N ∧
    isSixConsecutive N ∧
    (∃ (k : Nat), F N = k * k) ∧
    (∀ (M : FourDigitNumber),
      isValidFourDigitNumber M → isSixConsecutive M →
      (∃ (j : Nat), F M = j * j) →
      toInt M ≤ toInt N) ∧
    toInt N = 9613 := by sorry

end NUMINAMATH_CALUDE_smallest_six_consecutive_number_max_six_consecutive_with_perfect_square_F_l3165_316542


namespace NUMINAMATH_CALUDE_middle_position_theorem_l3165_316599

/-- Represents the color of a stone -/
inductive Color
  | Black
  | White

/-- Represents the state of the stone line -/
def StoneLine := Fin 2021 → Color

/-- Checks if a position is valid for the operation -/
def validPosition (n : Fin 2021) : Prop :=
  1 < n.val ∧ n.val < 2021

/-- Represents a single operation on the stone line -/
def operation (line : StoneLine) (n : Fin 2021) : StoneLine :=
  fun i => if i = n - 1 ∨ i = n + 1 then
    match line i with
    | Color.Black => Color.White
    | Color.White => Color.Black
    else line i

/-- Checks if all stones in the line are black -/
def allBlack (line : StoneLine) : Prop :=
  ∀ i, line i = Color.Black

/-- Initial configuration with one black stone at position n -/
def initialConfig (n : Fin 2021) : StoneLine :=
  fun i => if i = n then Color.Black else Color.White

/-- Represents the ability to make all stones black through operations -/
def canMakeAllBlack (line : StoneLine) : Prop :=
  ∃ (seq : List (Fin 2021)), 
    (∀ n ∈ seq, validPosition n) ∧
    allBlack (seq.foldl operation line)

/-- The main theorem to be proved -/
theorem middle_position_theorem :
  ∀ n : Fin 2021, canMakeAllBlack (initialConfig n) ↔ n = ⟨1011, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_middle_position_theorem_l3165_316599


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3165_316524

-- Define the universal set I
def I : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x * (x - 1) ≥ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3165_316524


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3165_316535

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3165_316535


namespace NUMINAMATH_CALUDE_abc_inequality_l3165_316597

theorem abc_inequality (a b c : ℝ) (h : a^2*b*c + a*b^2*c + a*b*c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3165_316597


namespace NUMINAMATH_CALUDE_total_paths_is_fifteen_l3165_316518

/-- A graph representing paths between points A, B, C, and D. -/
structure PathGraph where
  paths_AB : Nat
  paths_BC : Nat
  paths_CD : Nat
  direct_AC : Nat

/-- Calculates the total number of paths from A to D in the given graph. -/
def total_paths (g : PathGraph) : Nat :=
  g.paths_AB * g.paths_BC * g.paths_CD + g.direct_AC * g.paths_CD

/-- Theorem stating that the total number of paths from A to D is 15. -/
theorem total_paths_is_fifteen (g : PathGraph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BC = 2)
  (h3 : g.paths_CD = 3)
  (h4 : g.direct_AC = 1) : 
  total_paths g = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_paths_is_fifteen_l3165_316518


namespace NUMINAMATH_CALUDE_group_message_problem_l3165_316578

theorem group_message_problem (n : ℕ) (k : ℕ) : 
  n > 1 → 
  k > 0 → 
  k * n * (n - 1) = 440 → 
  n = 2 ∨ n = 5 ∨ n = 11 :=
by sorry

end NUMINAMATH_CALUDE_group_message_problem_l3165_316578


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l3165_316557

/-- Given an ellipse x²/m² + y²/n² = 1 passing through point P(a, b),
    prove that the minimum value of m + n is (a²/³ + b²/³)¹/³ -/
theorem ellipse_min_sum (a b m n : ℝ) (hm : m > 0) (hn : n > 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : abs a ≠ abs b)
  (h_ellipse : a^2 / m^2 + b^2 / n^2 = 1) :
  ∃ (min_sum : ℝ), min_sum = (a^(2/3) + b^(2/3))^(1/3) ∧
    ∀ (m' n' : ℝ), m' > 0 → n' > 0 → a^2 / m'^2 + b^2 / n'^2 = 1 →
      m' + n' ≥ min_sum :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l3165_316557


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3165_316564

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop :=
  2 * a * x^2 + 4 * x + b ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | x = -1/a}

-- State the theorem
theorem quadratic_inequality_theorem (a b : ℝ) 
  (h1 : ∀ x, x ∈ solution_set a ↔ quadratic_inequality a b x)
  (h2 : a > b) :
  (ab = 2) ∧ 
  (∀ a b, (2*a + b^3) / (2 - b^2) ≥ 4) ∧
  (∃ a b, (2*a + b^3) / (2 - b^2) = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3165_316564


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3165_316596

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_a3 : a 3 = 2)
  (h_a4 : a 4 = 4) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3165_316596


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3165_316503

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 ∧ 4 * y + b * x - 6 = 0 → 
   (2 / 3) * (b / 4) = 1) → 
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3165_316503


namespace NUMINAMATH_CALUDE_water_added_to_fill_tank_l3165_316554

/-- Proves that the amount of water added to fill a tank is 16 gallons, given the initial state and capacity. -/
theorem water_added_to_fill_tank (initial_fraction : ℚ) (full_capacity : ℕ) : 
  initial_fraction = 1/3 → full_capacity = 24 → (1 - initial_fraction) * full_capacity = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_fill_tank_l3165_316554


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3165_316580

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 9) :
  w / y = 8 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3165_316580


namespace NUMINAMATH_CALUDE_satisfying_polynomial_form_l3165_316570

/-- A polynomial with real coefficients satisfying the given equality for all real a, b, c -/
def SatisfyingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, p (a + b - 2*c) + p (b + c - 2*a) + p (c + a - 2*b) = 
               3 * (p (a - b) + p (b - c) + p (c - a))

/-- The theorem stating that any satisfying polynomial has the form a₂x² + a₁x -/
theorem satisfying_polynomial_form (p : ℝ → ℝ) (h : SatisfyingPolynomial p) :
  ∃ a₂ a₁ : ℝ, ∀ x : ℝ, p x = a₂ * x^2 + a₁ * x :=
sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_form_l3165_316570


namespace NUMINAMATH_CALUDE_no_solution_for_equation_expression_simplifies_to_half_l3165_316586

-- Define the domain for x
def X := {x : ℤ | -3 < x ∧ x ≤ 0}

-- Problem 1
theorem no_solution_for_equation :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → (2 / (x - 2) - 4 / (x^2 - 4) ≠ 1 / (x + 2)) :=
sorry

-- Problem 2
theorem expression_simplifies_to_half :
  ∀ x ∈ X, x = 0 →
  (x^2 / (x + 1) - x + 1) / ((x + 2) / (x^2 + 2*x + 1)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_expression_simplifies_to_half_l3165_316586


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l3165_316514

theorem largest_n_with_conditions : 
  ∃ (n : ℕ), n = 25 ∧ 
  (∃ (k : ℕ), n^2 = (k+1)^4 - k^4) ∧ 
  (∃ (b : ℕ), 3*n + 100 = b^2) ∧
  (∀ (m : ℕ), m > n → 
    (∀ (j : ℕ), m^2 ≠ (j+1)^4 - j^4) ∨ 
    (∀ (c : ℕ), 3*m + 100 ≠ c^2)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_conditions_l3165_316514


namespace NUMINAMATH_CALUDE_octadecagon_relation_l3165_316588

/-- Given a regular octadecagon inscribed in a circle with side length a and radius r,
    prove that a³ + r³ = 3r²a. -/
theorem octadecagon_relation (a r : ℝ) (h : a > 0) (k : r > 0) :
  a^3 + r^3 = 3 * r^2 * a := by
  sorry

end NUMINAMATH_CALUDE_octadecagon_relation_l3165_316588


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_3_million_l3165_316536

theorem scientific_notation_of_1_3_million :
  1300000 = 1.3 * (10 : ℝ)^6 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_3_million_l3165_316536


namespace NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l3165_316539

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum_odd (n : ℕ) : ℕ :=
  (n / 10000000) + ((n / 100000) % 10) + ((n / 1000) % 10) + ((n / 10) % 10)

def digit_sum_even (n : ℕ) : ℕ :=
  ((n / 1000000) % 10) + ((n / 10000) % 10) + ((n / 100) % 10) + (n % 10)

theorem smallest_digit_divisible_by_11 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_11 (85210000 + d * 1000 + 784) ↔ d = 1) ∧
    (∀ d' : ℕ, d' < d → ¬is_divisible_by_11 (85210000 + d' * 1000 + 784)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_divisible_by_11_l3165_316539


namespace NUMINAMATH_CALUDE_red_peaches_count_l3165_316517

theorem red_peaches_count (green_peaches : ℕ) (red_peaches : ℕ) : 
  green_peaches = 16 → red_peaches = green_peaches + 1 → red_peaches = 17 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l3165_316517


namespace NUMINAMATH_CALUDE_locus_of_M_constant_ratio_l3165_316544

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)
axiom P_on_ellipse : ellipse P.1 P.2

-- Define point M
def M (P : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define point N
def N (P : ℝ × ℝ) : ℝ × ℝ := sorry

theorem locus_of_M (P : ℝ × ℝ) (h : ellipse P.1 P.2) : 
  (M P).1 = -8 := by sorry

theorem constant_ratio (P : ℝ × ℝ) (h : ellipse P.1 P.2) :
  ‖N P - F₁‖ / ‖M P - F₁‖ = 1/2 := by sorry

end NUMINAMATH_CALUDE_locus_of_M_constant_ratio_l3165_316544


namespace NUMINAMATH_CALUDE_prob_red_then_black_value_l3165_316573

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hTotal : total_cards = 52)
  (hRed : red_cards = 26)
  (hBlack : black_cards = 26)
  (hSum : red_cards + black_cards = total_cards)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.black_cards : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a red card first and a black card second -/
theorem prob_red_then_black_value (d : Deck) : prob_red_then_black d = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_value_l3165_316573


namespace NUMINAMATH_CALUDE_parabola_perpendicular_range_l3165_316567

/-- Given a parabola y = x^2 with a fixed point A(-1, 1) and two moving points P and Q on the parabola,
    if PA ⊥ PQ, then the x-coordinate of Q is in (-∞, -3] ∪ [1, +∞) -/
theorem parabola_perpendicular_range (a x : ℝ) :
  let P : ℝ × ℝ := (a, a^2)
  let Q : ℝ × ℝ := (x, x^2)
  let A : ℝ × ℝ := (-1, 1)
  (a + 1) * (x - a) + (a^2 - 1) * (x^2 - a^2) = 0 →
  x ≤ -3 ∨ x ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_range_l3165_316567


namespace NUMINAMATH_CALUDE_apples_processed_equals_stems_l3165_316574

/-- A machine that processes apples and cuts stems -/
structure AppleProcessor where
  stems_after_2_hours : ℕ
  apples_processed : ℕ

/-- The number of stems after 2 hours is equal to the number of apples processed -/
axiom stems_equal_apples (m : AppleProcessor) : m.stems_after_2_hours = m.apples_processed

/-- Theorem: The number of apples processed is equal to the number of stems observed after 2 hours -/
theorem apples_processed_equals_stems (m : AppleProcessor) :
  m.apples_processed = m.stems_after_2_hours := by sorry

end NUMINAMATH_CALUDE_apples_processed_equals_stems_l3165_316574


namespace NUMINAMATH_CALUDE_frictional_force_is_10N_l3165_316509

/-- The acceleration due to gravity (m/s²) -/
def g : ℝ := 9.8

/-- Mass of the tank (kg) -/
def m₁ : ℝ := 2

/-- Mass of the cart (kg) -/
def m₂ : ℝ := 10

/-- Acceleration of the cart (m/s²) -/
def a : ℝ := 5

/-- Coefficient of friction between the tank and cart -/
def μ : ℝ := 0.6

/-- The frictional force acting on the tank from the cart (N) -/
def frictional_force : ℝ := m₁ * a

theorem frictional_force_is_10N : frictional_force = 10 := by
  sorry

#check frictional_force_is_10N

end NUMINAMATH_CALUDE_frictional_force_is_10N_l3165_316509


namespace NUMINAMATH_CALUDE_grandpas_initial_tomatoes_l3165_316561

-- Define the number of tomatoes that grew during vacation
def tomatoes_grown : ℕ := 3564

-- Define the multiplication factor for tomato growth
def growth_factor : ℕ := 100

-- Define the function to calculate the initial number of tomatoes
def initial_tomatoes : ℕ := (tomatoes_grown + growth_factor - 1) / growth_factor

-- Theorem statement
theorem grandpas_initial_tomatoes :
  initial_tomatoes = 36 :=
sorry

end NUMINAMATH_CALUDE_grandpas_initial_tomatoes_l3165_316561


namespace NUMINAMATH_CALUDE_counterexample_exists_l3165_316548

theorem counterexample_exists : ∃ (a b : ℕ), 
  (∃ (k : ℕ), a^7 = b^3 * k) ∧ 
  ¬(∃ (m : ℕ), a^2 = b * m) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3165_316548


namespace NUMINAMATH_CALUDE_no_rational_solution_l3165_316593

theorem no_rational_solution (n : ℕ+) : ¬ ∃ (x y : ℚ), 0 < x ∧ 0 < y ∧ x + y + 1/x + 1/y = 3*n := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l3165_316593


namespace NUMINAMATH_CALUDE_roots_cubic_equation_l3165_316502

theorem roots_cubic_equation (x₁ x₂ : ℝ) (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_cubic_equation_l3165_316502


namespace NUMINAMATH_CALUDE_fish_tank_leak_bucket_size_l3165_316592

/-- 
Given a fish tank leaking at a rate of 1.5 ounces per hour and a maximum time away of 12 hours,
prove that a bucket with twice the capacity of the total leakage will hold 36 ounces.
-/
theorem fish_tank_leak_bucket_size 
  (leak_rate : ℝ) 
  (max_time : ℝ) 
  (h1 : leak_rate = 1.5)
  (h2 : max_time = 12) : 
  2 * (leak_rate * max_time) = 36 := by
  sorry

#check fish_tank_leak_bucket_size

end NUMINAMATH_CALUDE_fish_tank_leak_bucket_size_l3165_316592


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3165_316547

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the non-coincident property for lines
variable (non_coincident_lines : Line → Line → Prop)

-- Define the non-coincident property for planes
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (a b : Line) (α β : Plane)
  (h_non_coincident_lines : non_coincident_lines a b)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h_a_perp_α : perp_line_plane a α)
  (h_b_perp_β : perp_line_plane b β)
  (h_α_perp_β : perp_plane_plane α β) :
  perp_line_line a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3165_316547


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3165_316590

/-- A two-digit number satisfying specific conditions -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n % 10 = n / 10 + 2) ∧
  (n * (n / 10 + n % 10) = 144)

/-- Theorem stating that 24 is the only two-digit number satisfying the given conditions -/
theorem unique_two_digit_number : ∃! n : ℕ, TwoDigitNumber n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3165_316590


namespace NUMINAMATH_CALUDE_smallest_common_factor_l3165_316540

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 3 → gcd (12*m - 3) (8*m + 9) = 1) ∧ 
  gcd (12*3 - 3) (8*3 + 9) > 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l3165_316540


namespace NUMINAMATH_CALUDE_c_rent_share_l3165_316587

/-- Represents the rent share calculation for a pasture -/
def RentShare (total_rent : ℕ) (a_oxen b_oxen c_oxen : ℕ) (a_months b_months c_months : ℕ) : ℕ :=
  let total_ox_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months
  let c_ox_months := c_oxen * c_months
  (total_rent * c_ox_months) / total_ox_months

/-- Theorem stating that c's share of the rent is 45 Rs -/
theorem c_rent_share :
  RentShare 175 10 12 15 7 5 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_c_rent_share_l3165_316587


namespace NUMINAMATH_CALUDE_problem_statement_l3165_316581

theorem problem_statement (a b : ℝ) : 
  (Real.sqrt (a + 2) + |b - 1| = 0) → ((a + b)^2007 = -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3165_316581


namespace NUMINAMATH_CALUDE_power_of_power_l3165_316579

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3165_316579


namespace NUMINAMATH_CALUDE_limit_at_one_l3165_316583

def f (x : ℝ) : ℝ := x^2

theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 2| < ε :=
  sorry

end NUMINAMATH_CALUDE_limit_at_one_l3165_316583


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3165_316534

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3165_316534
