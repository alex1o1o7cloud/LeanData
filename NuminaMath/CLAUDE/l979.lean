import Mathlib

namespace NUMINAMATH_CALUDE_total_expense_calculation_l979_97945

/-- Sandy's current age -/
def sandy_age : ℕ := 34

/-- Kim's current age -/
def kim_age : ℕ := 10

/-- Alex's current age -/
def alex_age : ℕ := sandy_age / 2

/-- Sandy's monthly phone bill expense -/
def sandy_expense : ℕ := 10 * sandy_age

/-- Alex's monthly expense next month -/
def alex_expense : ℕ := 2 * sandy_expense

theorem total_expense_calculation :
  sandy_age = 34 ∧
  kim_age = 10 ∧
  alex_age = sandy_age / 2 ∧
  sandy_expense = 10 * sandy_age ∧
  alex_expense = 2 * sandy_expense ∧
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_expense + alex_expense = 1020 := by
  sorry

end NUMINAMATH_CALUDE_total_expense_calculation_l979_97945


namespace NUMINAMATH_CALUDE_roots_geometric_sequence_range_l979_97974

theorem roots_geometric_sequence_range (a b : ℝ) (q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (∀ x : ℝ, (x^2 - a*x + 1)*(x^2 - b*x + 1) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) ∧ 
    (∃ r : ℝ, x₂ = x₁ * r ∧ x₃ = x₂ * r ∧ x₄ = x₃ * r) ∧
    q = r ∧ 
    1/3 ≤ q ∧ q ≤ 2) →
  4 ≤ a*b ∧ a*b ≤ 112/9 := by
sorry

end NUMINAMATH_CALUDE_roots_geometric_sequence_range_l979_97974


namespace NUMINAMATH_CALUDE_power_sum_equality_l979_97978

theorem power_sum_equality : (-1 : ℤ) ^ 47 + 2 ^ (3^3 + 4^2 - 6^2) = 127 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l979_97978


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l979_97913

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 8 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, m > 1 → m % 3 = 1 → m % 5 = 1 → m % 8 = 1 → m % 7 = 2 → m ≥ n) ∧
  n = 481 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l979_97913


namespace NUMINAMATH_CALUDE_tigrasha_first_snezhok_last_l979_97927

-- Define the kittens
inductive Kitten : Type
| Chernysh : Kitten
| Tigrasha : Kitten
| Snezhok : Kitten
| Pushok : Kitten

-- Define the eating speed for each kitten
def eating_speed (k : Kitten) : ℕ :=
  match k with
  | Kitten.Chernysh => 2
  | Kitten.Tigrasha => 5
  | Kitten.Snezhok => 3
  | Kitten.Pushok => 4

-- Define the initial number of sausages (same for all kittens)
def initial_sausages : ℕ := 7

-- Define the time to finish eating for each kitten
def time_to_finish (k : Kitten) : ℚ :=
  (initial_sausages : ℚ) / (eating_speed k : ℚ)

-- Theorem statement
theorem tigrasha_first_snezhok_last :
  (∀ k : Kitten, k ≠ Kitten.Tigrasha → time_to_finish Kitten.Tigrasha ≤ time_to_finish k) ∧
  (∀ k : Kitten, k ≠ Kitten.Snezhok → time_to_finish k ≤ time_to_finish Kitten.Snezhok) :=
sorry

end NUMINAMATH_CALUDE_tigrasha_first_snezhok_last_l979_97927


namespace NUMINAMATH_CALUDE_points_collinear_implies_a_equals_4_l979_97928

-- Define the points
def A : ℝ × ℝ := (4, 3)
def B (a : ℝ) : ℝ × ℝ := (5, a)
def C : ℝ × ℝ := (6, 5)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

-- Theorem statement
theorem points_collinear_implies_a_equals_4 (a : ℝ) :
  collinear A (B a) C → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_implies_a_equals_4_l979_97928


namespace NUMINAMATH_CALUDE_cyclist_pedestrian_speed_ratio_l979_97908

/-- Represents the speed of a person -/
structure Speed :=
  (value : ℝ)

/-- Represents a point in time -/
structure Time :=
  (hours : ℝ)

/-- Represents a distance between two points -/
structure Distance :=
  (value : ℝ)

/-- The problem setup -/
structure ProblemSetup :=
  (pedestrian_start : Time)
  (cyclist_start : Time)
  (meetup_time : Time)
  (cyclist_return : Time)
  (final_meetup : Time)
  (distance_AB : Distance)

/-- The theorem to be proved -/
theorem cyclist_pedestrian_speed_ratio 
  (setup : ProblemSetup)
  (pedestrian_speed : Speed)
  (cyclist_speed : Speed)
  (h1 : setup.pedestrian_start.hours = 12)
  (h2 : setup.meetup_time.hours = 13)
  (h3 : setup.final_meetup.hours = 16)
  (h4 : setup.pedestrian_start.hours < setup.cyclist_start.hours)
  (h5 : setup.cyclist_start.hours < setup.meetup_time.hours) :
  cyclist_speed.value / pedestrian_speed.value = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_pedestrian_speed_ratio_l979_97908


namespace NUMINAMATH_CALUDE_gary_money_after_sale_l979_97950

theorem gary_money_after_sale (initial_amount selling_price : ℝ) 
  (h1 : initial_amount = 73.0) 
  (h2 : selling_price = 55.0) : 
  initial_amount + selling_price = 128.0 :=
by sorry

end NUMINAMATH_CALUDE_gary_money_after_sale_l979_97950


namespace NUMINAMATH_CALUDE_xyz_sum_l979_97905

theorem xyz_sum (x y z : ℝ) (eq1 : 2*x + 3*y + 4*z = 10) (eq2 : y + 2*z = 2) : 
  x + y + z = 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l979_97905


namespace NUMINAMATH_CALUDE_stating_probability_of_target_sequence_l979_97917

/-- The number of balls in the box -/
def total_balls : ℕ := 500

/-- The number of balls selected -/
def selections : ℕ := 5

/-- The probability of selecting an odd-numbered ball -/
def prob_odd : ℚ := 1 / 2

/-- The probability of selecting an even-numbered ball -/
def prob_even : ℚ := 1 / 2

/-- The sequence of selections we're interested in (odd, even, odd, even, odd) -/
def target_sequence : List Bool := [true, false, true, false, true]

/-- 
Theorem stating that the probability of selecting the target sequence 
(odd, even, odd, even, odd) from a box of 500 balls numbered 1 to 500, 
with 5 selections and replacement, is 1/32.
-/
theorem probability_of_target_sequence : 
  (List.prod (target_sequence.map (fun b => if b then prob_odd else prob_even))) = (1 : ℚ) / 32 := by
  sorry

end NUMINAMATH_CALUDE_stating_probability_of_target_sequence_l979_97917


namespace NUMINAMATH_CALUDE_prime_between_30_and_40_with_specific_remainder_l979_97901

theorem prime_between_30_and_40_with_specific_remainder : 
  {n : ℕ | 30 ≤ n ∧ n ≤ 40 ∧ Prime n ∧ 1 ≤ n % 7 ∧ n % 7 ≤ 6} = {31, 37} := by
  sorry

end NUMINAMATH_CALUDE_prime_between_30_and_40_with_specific_remainder_l979_97901


namespace NUMINAMATH_CALUDE_girls_combined_average_l979_97918

/-- Represents a high school with exam scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- The average score calculation problem -/
theorem girls_combined_average 
  (cedar : School)
  (drake : School)
  (boys_combined_avg : ℝ)
  (h_cedar : cedar.combined_avg = 78)
  (h_drake : drake.combined_avg = 88)
  (h_cedar_boys : cedar.boys_avg = 75)
  (h_cedar_girls : cedar.girls_avg = 80)
  (h_drake_boys : drake.boys_avg = 85)
  (h_drake_girls : drake.girls_avg = 92)
  (h_boys_combined : boys_combined_avg = 83) :
  ∃ (girls_combined_avg : ℝ), girls_combined_avg = 88 := by
  sorry


end NUMINAMATH_CALUDE_girls_combined_average_l979_97918


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l979_97943

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 2 →
  a 5 = 8 →
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l979_97943


namespace NUMINAMATH_CALUDE_average_speed_calculation_l979_97926

/-- Given a distance of 100 kilometers traveled in 1.25 hours,
    prove that the average speed is 80 kilometers per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 100 →
  time = 1.25 →
  speed = distance / time →
  speed = 80 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l979_97926


namespace NUMINAMATH_CALUDE_sum_of_roots_l979_97968

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  c^2 + a*c + b = 0 →
  d^2 + a*d + b = 0 →
  a^2 + c*a + d = 0 →
  b^2 + c*b + d = 0 →
  a + b + c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l979_97968


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l979_97984

def A : Set ℕ := {1, 2}
def B : Set ℕ := {x | 2^x = 8}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l979_97984


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l979_97996

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
    2 * x₁^2 + (m + 1) * x₁ + m = 0 ∧
    2 * x₂^2 + (m + 1) * x₂ + m = 0) →
  m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l979_97996


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l979_97900

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l979_97900


namespace NUMINAMATH_CALUDE_line_slope_theorem_l979_97929

/-- Given a line with equation x = 5y + 5 passing through points (m, n) and (m + a, n + p),
    where p = 0.4, prove that a = 2. -/
theorem line_slope_theorem (m n a p : ℝ) : 
  p = 0.4 →
  m = 5 * n + 5 →
  (m + a) = 5 * (n + p) + 5 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_theorem_l979_97929


namespace NUMINAMATH_CALUDE_equation_solution_l979_97993

theorem equation_solution :
  ∃ x : ℝ, (4 * x + 6 * x = 360 - 9 * (x - 4)) ∧ (x = 396 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l979_97993


namespace NUMINAMATH_CALUDE_periodic_sine_condition_l979_97949

/-- Given a function f(x) = 2sin(ωx - π/3), prove that
    "∀x∈ℝ, f(x+π)=f(x)" is a necessary but not sufficient condition for ω = 2 -/
theorem periodic_sine_condition (ω : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x - π / 3)
  (∀ x, f (x + π) = f x) → ω = 2 ∧
  ∃ ω', ω' ≠ 2 ∧ (∀ x, 2 * Real.sin (ω' * x - π / 3) = 2 * Real.sin (ω' * (x + π) - π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_periodic_sine_condition_l979_97949


namespace NUMINAMATH_CALUDE_probability_rain_given_strong_winds_l979_97902

theorem probability_rain_given_strong_winds 
  (p_strong_winds : ℝ) 
  (p_rain : ℝ) 
  (p_both : ℝ) 
  (h1 : p_strong_winds = 0.4) 
  (h2 : p_rain = 0.5) 
  (h3 : p_both = 0.3) : 
  p_both / p_strong_winds = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_given_strong_winds_l979_97902


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l979_97904

/-- Parabola function -/
def f (x : ℝ) : ℝ := -(x + 1)^2 + 5

/-- Point A on the parabola -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the parabola -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the parabola -/
def C : ℝ × ℝ := (2, f 2)

/-- Theorem stating the relationship between y-coordinates of A, B, and C -/
theorem parabola_point_relationship : A.2 > B.2 ∧ B.2 > C.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l979_97904


namespace NUMINAMATH_CALUDE_fraction_equality_l979_97938

theorem fraction_equality (p q r u v w : ℝ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0 ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_sum_squares1 : p^2 + q^2 + r^2 = 49)
  (h_sum_squares2 : u^2 + v^2 + w^2 = 64)
  (h_dot_product : p*u + q*v + r*w = 56)
  (h_p_2q : p = 2*q) :
  (p + q + r) / (u + v + w) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l979_97938


namespace NUMINAMATH_CALUDE_min_abs_z_complex_l979_97961

/-- Given a complex number z satisfying |z - 5i| + |z - 3| = 7, 
    the minimum value of |z| is 15/7 -/
theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 5*Complex.I) + Complex.abs (z - 3) = 7) :
  ∃ (w : ℂ), Complex.abs w = 15/7 ∧ ∀ (v : ℂ), Complex.abs (v - 5*Complex.I) + Complex.abs (v - 3) = 7 → Complex.abs w ≤ Complex.abs v :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_complex_l979_97961


namespace NUMINAMATH_CALUDE_karen_wall_paint_area_l979_97976

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℝ := d.height * d.width

/-- Represents Karen's living room wall with its components -/
structure Wall where
  dimensions : Dimensions
  window : Dimensions
  door : Dimensions

/-- Calculates the area to be painted on the wall -/
def areaToPaint (w : Wall) : ℝ :=
  area w.dimensions - area w.window - area w.door

theorem karen_wall_paint_area :
  let wall : Wall := {
    dimensions := { height := 10, width := 15 },
    window := { height := 3, width := 5 },
    door := { height := 2, width := 6 }
  }
  areaToPaint wall = 123 := by sorry

end NUMINAMATH_CALUDE_karen_wall_paint_area_l979_97976


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l979_97909

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9) :
  ∀ n, a n = 2 * n - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l979_97909


namespace NUMINAMATH_CALUDE_cottage_build_time_l979_97956

/-- Represents the time (in days) it takes to build a cottage given the number of builders -/
def build_time (num_builders : ℕ) : ℚ := sorry

theorem cottage_build_time :
  build_time 3 = 8 →
  build_time 6 = 4 :=
by sorry

end NUMINAMATH_CALUDE_cottage_build_time_l979_97956


namespace NUMINAMATH_CALUDE_characterize_function_l979_97915

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m < n → f m < f n

/-- The main theorem about the structure of functions satisfying f(f(n)) = f(n)^2 -/
theorem characterize_function (f : ℕ+ → ℕ+) 
  (h_incr : StrictlyIncreasing f) 
  (h_eq : ∀ n : ℕ+, f (f n) = (f n)^2) :
  ∃ c : ℕ+, 
    (∀ n : ℕ+, n ≥ 2 → f n = c * n) ∧
    (f 1 = 1 ∨ f 1 = c) :=
sorry

end NUMINAMATH_CALUDE_characterize_function_l979_97915


namespace NUMINAMATH_CALUDE_no_positive_reals_satisfy_inequalities_l979_97962

theorem no_positive_reals_satisfy_inequalities :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2) ∧
    (a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3)) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_reals_satisfy_inequalities_l979_97962


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l979_97972

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_6 = 6 and a_9 = 9, prove that a_3 = 4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_6 : a 6 = 6) 
    (h_9 : a 9 = 9) : 
  a 3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l979_97972


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l979_97988

theorem simplify_trig_expression :
  Real.sqrt (2 + Real.cos (20 * π / 180) - Real.sin (10 * π / 180)^2) = Real.sqrt 3 * Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l979_97988


namespace NUMINAMATH_CALUDE_expression_value_l979_97995

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l979_97995


namespace NUMINAMATH_CALUDE_additional_machines_needed_l979_97959

/-- Given that 15 machines can finish a job in 36 days, prove that 5 additional machines
    are needed to finish the job in one-fourth less time. -/
theorem additional_machines_needed (machines : ℕ) (days : ℕ) (job : ℕ) :
  machines = 15 →
  days = 36 →
  job = machines * days →
  (machines + 5) * (days - days / 4) = job :=
by sorry

end NUMINAMATH_CALUDE_additional_machines_needed_l979_97959


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l979_97977

-- Define set M
def M : Set ℝ := {y | ∃ x, y = x^2 + 2*x - 3}

-- Define set N
def N : Set ℝ := {x | |x - 2| ≤ 3}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {y | -1 ≤ y ∧ y ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l979_97977


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l979_97935

-- Define the cryptarithm equation
def cryptarithm (A B C : ℕ) : Prop :=
  A < 10 ∧ B < 10 ∧ C < 10 ∧ 
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  100 * C + 10 * B + A + 100 * A + 10 * A + A = 10 * B + A

-- Theorem statement
theorem cryptarithm_solution :
  ∃! (A B C : ℕ), cryptarithm A B C ∧ A = 5 ∧ B = 9 ∧ C = 3 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l979_97935


namespace NUMINAMATH_CALUDE_mistaken_subtraction_l979_97992

theorem mistaken_subtraction (x : ℤ) : x - 64 = 122 → x - 46 = 140 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_subtraction_l979_97992


namespace NUMINAMATH_CALUDE_large_circle_diameter_l979_97930

theorem large_circle_diameter (r : ℝ) (h : r = 4) :
  let small_circles_radius := r
  let small_circles_count := 8
  let inner_octagon_side := 2 * small_circles_radius
  let inner_octagon_radius := inner_octagon_side / Real.sqrt 2
  let large_circle_radius := inner_octagon_radius + small_circles_radius
  large_circle_radius * 2 = 8 * Real.sqrt 2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_large_circle_diameter_l979_97930


namespace NUMINAMATH_CALUDE_cost_of_paints_paint_set_cost_l979_97960

theorem cost_of_paints (total_spent : ℕ) (num_classes : ℕ) (folders_per_class : ℕ) 
  (pencils_per_class : ℕ) (pencils_per_eraser : ℕ) (folder_cost : ℕ) (pencil_cost : ℕ) 
  (eraser_cost : ℕ) : ℕ :=
  let num_folders := num_classes * folders_per_class
  let num_pencils := num_classes * pencils_per_class
  let num_erasers := num_pencils / pencils_per_eraser
  let folders_total_cost := num_folders * folder_cost
  let pencils_total_cost := num_pencils * pencil_cost
  let erasers_total_cost := num_erasers * eraser_cost
  let supplies_cost := folders_total_cost + pencils_total_cost + erasers_total_cost
  total_spent - supplies_cost

theorem paint_set_cost : cost_of_paints 80 6 1 3 6 6 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paints_paint_set_cost_l979_97960


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l979_97916

theorem smallest_number_with_conditions : ∃! n : ℕ, 
  (n % 11 = 0) ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → n % k = 1) ∧
  (∀ m : ℕ, 
    (m % 11 = 0) ∧ 
    (∀ k : ℕ, 2 ≤ k ∧ k ≤ 8 → m % k = 1) → 
    n ≤ m) ∧
  n = 6721 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l979_97916


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l979_97925

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a⋅cos B = b⋅cos A, then the triangle is isosceles with A = B -/
theorem isosceles_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  a * Real.cos B = b * Real.cos A →
  A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l979_97925


namespace NUMINAMATH_CALUDE_bedroom_wall_area_l979_97946

/-- Calculates the total paintable wall area for multiple identical bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - non_paintable_area
  num_bedrooms * paintable_area

/-- Proves that the total paintable wall area of 4 bedrooms with given dimensions is 1860 square feet -/
theorem bedroom_wall_area : total_paintable_area 4 15 12 10 75 = 1860 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_wall_area_l979_97946


namespace NUMINAMATH_CALUDE_triangle_intersection_ratio_l979_97985

/-- Given a triangle XYZ, this theorem proves that if point P is on XY with XP:PY = 4:1,
    point Q is on YZ with YQ:QZ = 4:1, and lines PQ and XZ intersect at R,
    then PQ:QR = 4:1. -/
theorem triangle_intersection_ratio (X Y Z P Q R : ℝ × ℝ) : 
  (∃ t : ℝ, P = (1 - t) • X + t • Y ∧ t = 1/5) →  -- P is on XY with XP:PY = 4:1
  (∃ s : ℝ, Q = (1 - s) • Y + s • Z ∧ s = 4/5) →  -- Q is on YZ with YQ:QZ = 4:1
  (∃ u v : ℝ, R = (1 - u) • X + u • Z ∧ R = (1 - v) • P + v • Q) →  -- R is intersection of XZ and PQ
  ∃ k : ℝ, k • (Q - P) = R - Q ∧ k = 1/4 :=  -- PQ:QR = 4:1
by sorry

end NUMINAMATH_CALUDE_triangle_intersection_ratio_l979_97985


namespace NUMINAMATH_CALUDE_sara_movie_rental_cost_l979_97948

def movie_spending (theater_ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) : Prop :=
  let theater_total : ℚ := theater_ticket_price * num_tickets
  let known_expenses : ℚ := theater_total + bought_movie_price
  let rental_cost : ℚ := total_spent - known_expenses
  rental_cost = 159/100

theorem sara_movie_rental_cost :
  movie_spending (1062/100) 2 (1395/100) (3678/100) :=
sorry

end NUMINAMATH_CALUDE_sara_movie_rental_cost_l979_97948


namespace NUMINAMATH_CALUDE_polynomial_simplification_l979_97947

/-- Simplification of a polynomial expression -/
theorem polynomial_simplification (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) =
  -2 * y^3 + y^2 + 10 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l979_97947


namespace NUMINAMATH_CALUDE_small_triangle_perimeter_l979_97922

/-- Given a triangle and three trapezoids formed by cuts parallel to its sides,
    this theorem proves the perimeter of the resulting small triangle. -/
theorem small_triangle_perimeter
  (original_perimeter : ℝ)
  (trapezoid1_perimeter trapezoid2_perimeter trapezoid3_perimeter : ℝ)
  (h1 : original_perimeter = 11)
  (h2 : trapezoid1_perimeter = 5)
  (h3 : trapezoid2_perimeter = 7)
  (h4 : trapezoid3_perimeter = 9) :
  trapezoid1_perimeter + trapezoid2_perimeter + trapezoid3_perimeter - original_perimeter = 10 :=
by sorry

end NUMINAMATH_CALUDE_small_triangle_perimeter_l979_97922


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l979_97931

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) : 
  Nat.lcm X Y = 180 → 
  (X : ℚ) / (Y : ℚ) = 2 / 5 → 
  Nat.gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l979_97931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l979_97986

theorem arithmetic_sequence_middle_term 
  (a : Fin 5 → ℝ)  -- a is a function from Fin 5 to ℝ, representing the 5 terms
  (h1 : a 0 = -8)  -- first term is -8
  (h2 : a 4 = 10)  -- last term is 10
  (h3 : ∀ i : Fin 4, a (i + 1) - a i = a 1 - a 0)  -- arithmetic sequence condition
  : a 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l979_97986


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l979_97973

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l979_97973


namespace NUMINAMATH_CALUDE_solve_eggs_problem_l979_97966

def eggs_problem (breakfast_eggs lunch_eggs total_eggs : ℕ) : Prop :=
  let dinner_eggs := total_eggs - (breakfast_eggs + lunch_eggs)
  dinner_eggs = 1

theorem solve_eggs_problem :
  eggs_problem 2 3 6 :=
sorry

end NUMINAMATH_CALUDE_solve_eggs_problem_l979_97966


namespace NUMINAMATH_CALUDE_three_x_plus_five_y_equals_six_l979_97906

theorem three_x_plus_five_y_equals_six 
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 
  3 * x + 5 * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_x_plus_five_y_equals_six_l979_97906


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l979_97932

theorem systematic_sampling_proof (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 883) (h2 : sample_size = 80) :
  ∃ (sampling_interval : ℕ) (n : ℕ),
    sampling_interval = 11 ∧ 
    n = 3 ∧ 
    total_students = sample_size * sampling_interval + n :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l979_97932


namespace NUMINAMATH_CALUDE_fraction_equality_l979_97952

theorem fraction_equality (P Q : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) →
  Q / P = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l979_97952


namespace NUMINAMATH_CALUDE_article_sale_price_l979_97994

/-- Given an article with cost price CP, prove that the selling price SP
    that yields the same percentage profit as the percentage loss when
    sold for 1280 is 1820, given that selling it for 1937.5 gives a 25% profit. -/
theorem article_sale_price (CP : ℝ) 
    (h1 : 1937.5 = CP * 1.25)  -- 25% profit condition
    (h2 : ∃ SP, (SP - CP) / CP = (CP - 1280) / CP)  -- Equal percentage condition
    : ∃ SP, SP = 1820 ∧ (SP - CP) / CP = (CP - 1280) / CP := by
  sorry

end NUMINAMATH_CALUDE_article_sale_price_l979_97994


namespace NUMINAMATH_CALUDE_projection_theorem_l979_97954

def v : Fin 2 → ℝ := ![6, -3]
def u : Fin 2 → ℝ := ![3, 0]

theorem projection_theorem :
  (((v • u) / (u • u)) • u) = ![6, 0] := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l979_97954


namespace NUMINAMATH_CALUDE_exists_prime_not_cube_root_l979_97981

theorem exists_prime_not_cube_root (p q : ℕ) : 
  ∃ q : ℕ, Prime q ∧ ∀ p : ℕ, Prime p → ¬∃ n : ℕ, n^3 = p^2 + q :=
sorry

end NUMINAMATH_CALUDE_exists_prime_not_cube_root_l979_97981


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l979_97936

-- Define variables
variable (x y a b : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l979_97936


namespace NUMINAMATH_CALUDE_function_symmetry_implies_a_value_l979_97921

theorem function_symmetry_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, 2*(1-x)^2 - a*(1-x) + 3 = 2*(1+x)^2 - a*(1+x) + 3) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_implies_a_value_l979_97921


namespace NUMINAMATH_CALUDE_arithmetic_square_root_is_function_l979_97987

theorem arithmetic_square_root_is_function : 
  ∀ (x : ℝ), x > 0 → ∃! (y : ℝ), y > 0 ∧ y^2 = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_is_function_l979_97987


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l979_97957

theorem average_of_five_numbers 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : (x₁ + x₂) / 2 = 12) 
  (h₂ : (x₃ + x₄ + x₅) / 3 = 7) : 
  (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l979_97957


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l979_97989

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 24 km/hr in still water, traveling downstream
    in a stream with a speed of 4 km/hr for 6 hours, covers a distance of 168 km. -/
theorem boat_downstream_distance :
  distance_downstream 24 4 6 = 168 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l979_97989


namespace NUMINAMATH_CALUDE_two_color_no_monochromatic_ap_l979_97934

theorem two_color_no_monochromatic_ap :
  ∃ f : ℕ+ → Bool, ∀ q r : ℕ+, ∃ n1 n2 : ℕ+, f (q * n1 + r) ≠ f (q * n2 + r) :=
by sorry

end NUMINAMATH_CALUDE_two_color_no_monochromatic_ap_l979_97934


namespace NUMINAMATH_CALUDE_relay_race_time_l979_97997

/-- Represents the time taken by each runner in the relay race -/
structure RelayTimes where
  mary : ℕ
  susan : ℕ
  jen : ℕ
  tiffany : ℕ

/-- Calculates the total time of the relay race -/
def total_time (times : RelayTimes) : ℕ :=
  times.mary + times.susan + times.jen + times.tiffany

/-- Theorem stating the total time of the relay race -/
theorem relay_race_time : ∃ (times : RelayTimes),
  times.mary = 2 * times.susan ∧
  times.susan = times.jen + 10 ∧
  times.jen = 30 ∧
  times.tiffany = times.mary - 7 ∧
  total_time times = 223 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_l979_97997


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l979_97991

theorem mean_equality_implies_z_value :
  let mean1 := (5 + 8 + 17) / 3
  let mean2 := (15 + z) / 2
  mean1 = mean2 → z = 5 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l979_97991


namespace NUMINAMATH_CALUDE_test_scores_order_l979_97941

-- Define the scores as natural numbers
variable (J E N L : ℕ)

-- Define the theorem
theorem test_scores_order :
  -- Conditions
  (E = J) →  -- Elina's score is the same as Jasper's
  (N ≤ J) →  -- Norah's score is not higher than Jasper's
  (L > J) →  -- Liam's score is higher than Jasper's
  -- Conclusion: The order of scores from lowest to highest is N, E, L
  (N ≤ E ∧ E < L) := by
sorry

end NUMINAMATH_CALUDE_test_scores_order_l979_97941


namespace NUMINAMATH_CALUDE_polynomial_equality_l979_97912

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x - 2) * (x + 3) = x^2 + a*x + b) → (a = 1 ∧ b = -6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l979_97912


namespace NUMINAMATH_CALUDE_train_speed_and_length_l979_97942

/-- Proves that given a bridge of length 1000m, if a train takes 60s to pass from the beginning
    to the end of the bridge and spends 40s on the bridge, then the speed of the train is 20 m/s
    and its length is 200m. -/
theorem train_speed_and_length
  (bridge_length : ℝ)
  (time_to_pass : ℝ)
  (time_on_bridge : ℝ)
  (h1 : bridge_length = 1000)
  (h2 : time_to_pass = 60)
  (h3 : time_on_bridge = 40)
  : ∃ (speed length : ℝ),
    speed = 20 ∧
    length = 200 ∧
    time_to_pass * speed = bridge_length + length ∧
    time_on_bridge * speed = bridge_length - length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l979_97942


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l979_97970

/-- Calculate the overall loss percentage on three articles with given purchase prices, exchange rates, shipping fees, selling prices, and sales tax. -/
theorem overall_loss_percentage
  (purchase_a purchase_b purchase_c : ℝ)
  (exchange_eur exchange_gbp : ℝ)
  (shipping_fee : ℝ)
  (sell_a sell_b sell_c : ℝ)
  (sales_tax_rate : ℝ)
  (h_purchase_a : purchase_a = 100)
  (h_purchase_b : purchase_b = 200)
  (h_purchase_c : purchase_c = 300)
  (h_exchange_eur : exchange_eur = 1.1)
  (h_exchange_gbp : exchange_gbp = 1.3)
  (h_shipping_fee : shipping_fee = 10)
  (h_sell_a : sell_a = 110)
  (h_sell_b : sell_b = 250)
  (h_sell_c : sell_c = 330)
  (h_sales_tax_rate : sales_tax_rate = 0.05) :
  ∃ (loss_percentage : ℝ), 
    abs (loss_percentage - 0.0209) < 0.0001 ∧
    loss_percentage = 
      (((sell_a + sell_b + sell_c) * (1 + sales_tax_rate) - 
        (purchase_a + purchase_b * exchange_eur + purchase_c * exchange_gbp + 3 * shipping_fee)) / 
       (purchase_a + purchase_b * exchange_eur + purchase_c * exchange_gbp + 3 * shipping_fee)) * (-100) :=
by sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l979_97970


namespace NUMINAMATH_CALUDE_function_lower_bound_l979_97955

theorem function_lower_bound (a b : ℝ) (h : a + b = 4) :
  ∀ x : ℝ, |x + a^2| + |x - b^2| ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l979_97955


namespace NUMINAMATH_CALUDE_factorization_problem_1_l979_97953

theorem factorization_problem_1 (a : ℝ) : -2*a^2 + 4*a = -2*a*(a - 2) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l979_97953


namespace NUMINAMATH_CALUDE_multiply_98_squared_l979_97951

theorem multiply_98_squared : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_multiply_98_squared_l979_97951


namespace NUMINAMATH_CALUDE_unique_abc_solution_l979_97999

/-- Represents a base-6 number with two digits -/
def Base6TwoDigit (a b : Nat) : Nat := 6 * a + b

/-- Represents a base-6 number with one digit -/
def Base6OneDigit (c : Nat) : Nat := c

theorem unique_abc_solution :
  ∀ A B C : Nat,
    A ≠ 0 → B ≠ 0 → C ≠ 0 →
    A < 6 → B < 6 → C < 6 →
    A ≠ B → B ≠ C → A ≠ C →
    Base6TwoDigit A B + Base6OneDigit C = Base6TwoDigit C 0 →
    Base6TwoDigit A B + Base6TwoDigit B A = Base6TwoDigit C C →
    A = 4 ∧ B = 1 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_abc_solution_l979_97999


namespace NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l979_97920

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 20)^2 / 15^2 = 1

-- Define the focus with smaller x-coordinate
def focus_smaller_x : ℝ × ℝ := (-11.55, 20)

-- Theorem statement
theorem hyperbola_focus_smaller_x :
  ∃ (f : ℝ × ℝ), 
    (∀ x y, hyperbola x y → (x - 5)^2 + (y - 20)^2 ≥ (f.1 - 5)^2 + (f.2 - 20)^2) ∧
    (∀ x y, hyperbola x y → x ≤ 5 → x ≥ f.1) ∧
    f = focus_smaller_x :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_smaller_x_l979_97920


namespace NUMINAMATH_CALUDE_divisor_sum_representation_l979_97964

theorem divisor_sum_representation (n : ℕ) :
  ∀ k : ℕ, k ≤ n! → ∃ (S : Finset ℕ), 
    (∀ x ∈ S, x ∣ n!) ∧ 
    S.card ≤ n ∧ 
    k = S.sum id :=
by sorry

end NUMINAMATH_CALUDE_divisor_sum_representation_l979_97964


namespace NUMINAMATH_CALUDE_total_people_is_803_l979_97944

/-- The number of parents in the program -/
def num_parents : ℕ := 105

/-- The number of pupils in the program -/
def num_pupils : ℕ := 698

/-- The total number of people in the program -/
def total_people : ℕ := num_parents + num_pupils

/-- Theorem stating that the total number of people in the program is 803 -/
theorem total_people_is_803 : total_people = 803 := by
  sorry

end NUMINAMATH_CALUDE_total_people_is_803_l979_97944


namespace NUMINAMATH_CALUDE_second_class_fare_collection_l979_97979

/-- Proves that the amount collected from II class passengers is 1250 given the specified conditions --/
theorem second_class_fare_collection
  (passenger_ratio : ℚ) -- Ratio of I class to II class passengers
  (fare_ratio : ℚ) -- Ratio of I class to II class fares
  (total_amount : ℕ) -- Total amount collected
  (h1 : passenger_ratio = 1 / 50)
  (h2 : fare_ratio = 3 / 1)
  (h3 : total_amount = 1325) :
  (50 : ℚ) * (total_amount : ℚ) / (53 : ℚ) = 1250 := by
  sorry


end NUMINAMATH_CALUDE_second_class_fare_collection_l979_97979


namespace NUMINAMATH_CALUDE_base_9_to_base_10_l979_97980

theorem base_9_to_base_10 : 
  (8 * 9^1 + 5 * 9^0 : ℕ) = 77 := by
  sorry

end NUMINAMATH_CALUDE_base_9_to_base_10_l979_97980


namespace NUMINAMATH_CALUDE_length_ae_is_21_l979_97924

/-- Given 5 consecutive points on a straight line, prove that under certain conditions, the length of ae is 21 -/
theorem length_ae_is_21
  (a b c d e : ℝ) -- Representing points as real numbers on a line
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points
  (h_bc_cd : c - b = 3 * (d - c)) -- bc = 3 cd
  (h_de : e - d = 8) -- de = 8
  (h_ab : b - a = 5) -- ab = 5
  (h_ac : c - a = 11) -- ac = 11
  : e - a = 21 := by
  sorry

end NUMINAMATH_CALUDE_length_ae_is_21_l979_97924


namespace NUMINAMATH_CALUDE_number_is_nine_l979_97998

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def xiao_qian_statement (n : ℕ) : Prop := is_perfect_square n ∧ n < 5

def xiao_lu_statement (n : ℕ) : Prop := n < 7 ∧ n ≥ 10

def xiao_dai_statement (n : ℕ) : Prop := is_perfect_square n ∧ n ≥ 5

def one_all_true (n : ℕ) : Prop :=
  (xiao_qian_statement n) ∨ (xiao_lu_statement n) ∨ (xiao_dai_statement n)

def one_all_false (n : ℕ) : Prop :=
  (¬xiao_qian_statement n) ∨ (¬xiao_lu_statement n) ∨ (¬xiao_dai_statement n)

def one_true_one_false (n : ℕ) : Prop :=
  (is_perfect_square n ∧ ¬(n < 5)) ∨
  ((n < 7) ∧ ¬(n ≥ 10)) ∨
  ((is_perfect_square n) ∧ ¬(n ≥ 5))

theorem number_is_nine :
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 99 ∧ one_all_true n ∧ one_all_false n ∧ one_true_one_false n ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_number_is_nine_l979_97998


namespace NUMINAMATH_CALUDE_inverse_sum_property_l979_97967

-- Define the function f and its properties
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + f (-x) = 2

-- Define the inverse function property
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

-- State the theorem
theorem inverse_sum_property
  (f : ℝ → ℝ)
  (h_inv : has_inverse f)
  (h_prop : f_property f) :
  ∀ x : ℝ, f⁻¹ (2008 - x) + f⁻¹ (x - 2006) = 0 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_property_l979_97967


namespace NUMINAMATH_CALUDE_wenzhou_population_scientific_notation_l979_97910

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) (h : x > 0) : ScientificNotation :=
  sorry

theorem wenzhou_population_scientific_notation :
  toScientificNotation 9570000 (by norm_num) =
    ScientificNotation.mk 9.57 6 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_wenzhou_population_scientific_notation_l979_97910


namespace NUMINAMATH_CALUDE_hash_example_l979_97983

def hash (a b c d : ℝ) : ℝ := d * b^2 - 5 * a * c

theorem hash_example : hash 2 3 1 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_hash_example_l979_97983


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l979_97933

theorem division_multiplication_equality : (144 / 6) * 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l979_97933


namespace NUMINAMATH_CALUDE_parabola_vertex_l979_97969

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -3 * (x - 1)^2 + 4

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 4)

/-- Theorem: The vertex of the parabola y = -3(x-1)^2 + 4 is at the point (1,4) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l979_97969


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l979_97963

/-- Calculate the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmh = 60 →
  crossing_time = 45 →
  ∃ bridge_length : ℝ,
    (bridge_length ≥ 550) ∧ 
    (bridge_length ≤ 551) ∧
    (train_speed_kmh * 1000 / 3600 * crossing_time = train_length + bridge_length) :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l979_97963


namespace NUMINAMATH_CALUDE_outfits_count_l979_97975

/-- The number of outfits that can be made with given numbers of shirts, pants, and hats -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (hats : ℕ) : ℕ :=
  shirts * pants * hats

/-- Theorem stating that the number of outfits with 4 shirts, 5 pants, and 3 hats is 60 -/
theorem outfits_count :
  number_of_outfits 4 5 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l979_97975


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l979_97907

def I : Set ℕ := {x | 0 < x ∧ x ≤ 6}
def P : Set ℕ := {x | 6 % x = 0}
def Q : Set ℕ := {1, 3, 4, 5}

theorem complement_intersection_theorem :
  (I \ P) ∩ Q = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l979_97907


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematics_l979_97971

def alphabet_size : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8

theorem probability_letter_in_mathematics :
  (unique_letters_in_mathematics : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematics_l979_97971


namespace NUMINAMATH_CALUDE_prob_last_roll_is_15th_l979_97958

/-- The number of sides on the die -/
def n : ℕ := 20

/-- The total number of rolls -/
def total_rolls : ℕ := 15

/-- The number of non-repeating rolls -/
def non_repeating_rolls : ℕ := 13

/-- Probability of getting a specific sequence of rolls on a n-sided die,
    where the first 'non_repeating_rolls' are different from their predecessors,
    and the last roll is the same as its predecessor -/
def prob_sequence (n : ℕ) (total_rolls : ℕ) (non_repeating_rolls : ℕ) : ℚ :=
  (n - 1 : ℚ)^non_repeating_rolls / n^(total_rolls - 1)

theorem prob_last_roll_is_15th :
  prob_sequence n total_rolls non_repeating_rolls = 19^13 / 20^14 := by
  sorry

end NUMINAMATH_CALUDE_prob_last_roll_is_15th_l979_97958


namespace NUMINAMATH_CALUDE_jerry_books_count_l979_97982

/-- The number of books each shelf can hold -/
def books_per_shelf : ℕ := 4

/-- The number of shelves Jerry needs -/
def shelves_needed : ℕ := 3

/-- The total number of books Jerry has to put away -/
def total_books : ℕ := books_per_shelf * shelves_needed

theorem jerry_books_count : total_books = 12 := by
  sorry

end NUMINAMATH_CALUDE_jerry_books_count_l979_97982


namespace NUMINAMATH_CALUDE_three_correct_propositions_l979_97990

theorem three_correct_propositions (a b c d : ℝ) : 
  (∃! n : ℕ, n = 3 ∧ 
    (((a * b > 0 ∧ b * c - a * d > 0) → (c / a - d / b > 0)) ∧
     ((a * b > 0 ∧ c / a - d / b > 0) → (b * c - a * d > 0)) ∧
     ((b * c - a * d > 0 ∧ c / a - d / b > 0) → (a * b > 0)))) := by
  sorry

end NUMINAMATH_CALUDE_three_correct_propositions_l979_97990


namespace NUMINAMATH_CALUDE_annual_car_insurance_cost_l979_97911

/-- Theorem: If a person spends 40000 dollars on car insurance over a decade,
    then their annual car insurance cost is 4000 dollars. -/
theorem annual_car_insurance_cost (total_cost : ℕ) (years : ℕ) (annual_cost : ℕ) :
  total_cost = 40000 →
  years = 10 →
  annual_cost = total_cost / years →
  annual_cost = 4000 := by
  sorry

end NUMINAMATH_CALUDE_annual_car_insurance_cost_l979_97911


namespace NUMINAMATH_CALUDE_min_value_of_expression_l979_97965

theorem min_value_of_expression (x y : ℝ) :
  2 * x^2 + 2 * x * y + y^2 - 2 * x - 1 ≥ -2 ∧
  ∃ (a b : ℝ), 2 * a^2 + 2 * a * b + b^2 - 2 * a - 1 = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l979_97965


namespace NUMINAMATH_CALUDE_double_factorial_sum_denominator_l979_97919

/-- Double factorial for odd numbers -/
def odd_double_factorial (n : ℕ) : ℕ := sorry

/-- Double factorial for even numbers -/
def even_double_factorial (n : ℕ) : ℕ := sorry

/-- The sum of the ratios of double factorials -/
def double_factorial_sum : ℚ :=
  (Finset.range 2009).sum (fun i => (odd_double_factorial (2*i+1)) / (even_double_factorial (2*i+2)))

/-- The denominator of the sum when expressed in lowest terms -/
def denominator_of_sum : ℕ := sorry

/-- The power of 2 in the denominator -/
def a : ℕ := sorry

/-- The odd factor in the denominator -/
def b : ℕ := sorry

theorem double_factorial_sum_denominator :
  denominator_of_sum = 2^a * b ∧ Odd b ∧ a*b/10 = 401 := by sorry

end NUMINAMATH_CALUDE_double_factorial_sum_denominator_l979_97919


namespace NUMINAMATH_CALUDE_inequality_proof_l979_97940

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ 
  ((a^2 + b^2 + c^2) * (a*b + b*c + c*a)) / (a*b*c * (a + b + c)) + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l979_97940


namespace NUMINAMATH_CALUDE_remainder_problem_l979_97923

theorem remainder_problem (N : ℤ) : 
  ∃ k : ℤ, N = 761 * k + 173 → N % 29 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l979_97923


namespace NUMINAMATH_CALUDE_student_arrangement_count_l979_97903

/-- The number of ways to arrange n elements -/
def factorial (n : ℕ) : ℕ := (List.range n).foldr (· * ·) 1

/-- The number of ways to arrange 5 students in a line -/
def totalArrangements : ℕ := factorial 5

/-- The number of ways to arrange 3 students together and 2 separately -/
def restrictedArrangements : ℕ := factorial 3 * factorial 3

/-- The number of valid arrangements where 3 students are not next to each other -/
def validArrangements : ℕ := totalArrangements - restrictedArrangements

theorem student_arrangement_count :
  validArrangements = 84 :=
sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l979_97903


namespace NUMINAMATH_CALUDE_expression_evaluation_l979_97914

theorem expression_evaluation (x y k : ℤ) 
  (hx : x = 7) (hy : y = 3) (hk : k = 10) : 
  (x - y) * (x + y) + k = 50 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l979_97914


namespace NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approx_7_9916_l979_97939

/-- Calculates the speed of a man given the parameters of a passing train -/
theorem man_speed_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let man_speed_mps := train_speed_mps - relative_speed
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  man_speed_kmph

/-- The speed of the man is approximately 7.9916 kmph -/
theorem man_speed_approx_7_9916 : 
  ∃ ε > 0, |man_speed_calculation 350 68 20.99832013438925 - 7.9916| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_calculation_man_speed_approx_7_9916_l979_97939


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l979_97937

theorem solution_satisfies_equation :
  ∃ (x y : ℝ), x ≥ 0 ∧ y > 0 ∧
  Real.sqrt (9 + x) + Real.sqrt (9 - x) + Real.sqrt y = 5 * Real.sqrt 3 ∧
  x = 0 ∧ y = 111 - 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l979_97937
