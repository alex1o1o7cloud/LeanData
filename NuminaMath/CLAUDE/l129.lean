import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l129_12943

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / Real.sqrt (a^2 + a + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l129_12943


namespace NUMINAMATH_CALUDE_square_area_error_l129_12934

theorem square_area_error (S : ℝ) (h : S > 0) : 
  let measured_side := S * (1 + 0.06)
  let actual_area := S^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 12.36 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l129_12934


namespace NUMINAMATH_CALUDE_game_size_proof_l129_12930

/-- Given a game download scenario where:
  * 310 MB has already been downloaded
  * The remaining download speed is 3 MB/minute
  * It takes 190 more minutes to finish the download
  Prove that the total size of the game is 880 MB -/
theorem game_size_proof (already_downloaded : ℕ) (download_speed : ℕ) (remaining_time : ℕ) :
  already_downloaded = 310 →
  download_speed = 3 →
  remaining_time = 190 →
  already_downloaded + download_speed * remaining_time = 880 :=
by sorry

end NUMINAMATH_CALUDE_game_size_proof_l129_12930


namespace NUMINAMATH_CALUDE_problem_solution_l129_12907

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| - |x + b|

-- Define the function g
def g (a b x : ℝ) : ℝ := -x^2 - a*x - b

theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hmax_achieved : ∃ x, f a b x = 3) 
  (hg_less_f : ∀ x ≥ a, g a b x < f a b x) :
  (a + b = 3) ∧ (1/2 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l129_12907


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l129_12951

theorem square_area_from_diagonal (d : ℝ) (h : d = 16 * Real.sqrt 2) : 
  (d / Real.sqrt 2) ^ 2 = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l129_12951


namespace NUMINAMATH_CALUDE_completing_square_transformation_l129_12979

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 8*x + 2 = 0) ↔ ((x - 4)^2 = 14) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l129_12979


namespace NUMINAMATH_CALUDE_spinner_probability_l129_12997

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 →
  pB = 1/3 →
  pA + pB + pC + pD = 1 →
  pD = 1/4 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l129_12997


namespace NUMINAMATH_CALUDE_equation_solution_l129_12936

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (5*x^2 + 70*x + 2) / (3*x + 28) - (4*x + 2)
  let sol1 : ℝ := (-48 + 28*Real.sqrt 22) / 14
  let sol2 : ℝ := (-48 - 28*Real.sqrt 22) / 14
  f sol1 = 0 ∧ f sol2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l129_12936


namespace NUMINAMATH_CALUDE_convention_handshakes_l129_12959

/-- The number of handshakes at the Annual Mischief Convention --/
def annual_mischief_convention_handshakes (num_gremlins : ℕ) (num_imps : ℕ) : ℕ :=
  let gremlin_handshakes := num_gremlins.choose 2
  let imp_gremlin_handshakes := num_imps * num_gremlins
  gremlin_handshakes + imp_gremlin_handshakes

/-- Theorem stating the number of handshakes at the Annual Mischief Convention --/
theorem convention_handshakes :
  annual_mischief_convention_handshakes 25 20 = 800 := by
  sorry

#eval annual_mischief_convention_handshakes 25 20

end NUMINAMATH_CALUDE_convention_handshakes_l129_12959


namespace NUMINAMATH_CALUDE_teresa_spent_forty_l129_12919

/-- The total amount spent by Teresa at the local shop -/
def total_spent (sandwich_price : ℚ) (sandwich_quantity : ℕ)
  (salami_price : ℚ) (olive_price_per_pound : ℚ) (olive_quantity : ℚ)
  (feta_price_per_pound : ℚ) (feta_quantity : ℚ) (bread_price : ℚ) : ℚ :=
  sandwich_price * sandwich_quantity +
  salami_price +
  3 * salami_price +
  olive_price_per_pound * olive_quantity +
  feta_price_per_pound * feta_quantity +
  bread_price

/-- Theorem: Teresa spends $40.00 at the local shop -/
theorem teresa_spent_forty : 
  total_spent 7.75 2 4 10 (1/4) 8 (1/2) 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_teresa_spent_forty_l129_12919


namespace NUMINAMATH_CALUDE_nidas_chocolates_l129_12922

theorem nidas_chocolates (x : ℕ) 
  (h1 : 3 * x + 5 + 25 = 5 * x) : 3 * x + 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_nidas_chocolates_l129_12922


namespace NUMINAMATH_CALUDE_minimize_distance_to_point_l129_12915

/-- Given points P(-2, -2) and R(2, m), prove that the value of m that minimizes 
    the distance PR is -2. -/
theorem minimize_distance_to_point (m : ℝ) : 
  let P : ℝ × ℝ := (-2, -2)
  let R : ℝ × ℝ := (2, m)
  let distance := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  (∀ k : ℝ, distance ≤ Real.sqrt ((P.1 - 2)^2 + (P.2 - k)^2)) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_minimize_distance_to_point_l129_12915


namespace NUMINAMATH_CALUDE_quadratic_property_l129_12905

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_property (a b c : ℝ) :
  (∀ x, f a b c x ≥ 10) ∧  -- minimum value is 10
  (f a b c (-2) = 10) ∧    -- minimum occurs at x = -2
  (f a b c 0 = 6) →        -- passes through (0, 6)
  f a b c 5 = -39 :=       -- f(5) = -39
by sorry

end NUMINAMATH_CALUDE_quadratic_property_l129_12905


namespace NUMINAMATH_CALUDE_sequences_properties_l129_12978

def sequence_a (n : ℕ) : ℤ := (-2) ^ n
def sequence_b (n : ℕ) : ℤ := (-2) ^ (n - 1)
def sequence_c (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)

theorem sequences_properties :
  (sequence_a 6 = 64) ∧
  (sequence_b 7 = 64) ∧
  (sequence_c 7 = 192) ∧
  (sequence_c 11 = 3072) := by
sorry

end NUMINAMATH_CALUDE_sequences_properties_l129_12978


namespace NUMINAMATH_CALUDE_exactly_one_and_two_white_mutually_exclusive_not_contradictory_l129_12935

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls from the bag -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Red) ∨
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.White)

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (event1 event2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(event1 outcome ∧ event2 outcome)

/-- Two events are contradictory if one of them must occur -/
def contradictory (event1 event2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, event1 outcome ∨ event2 outcome

theorem exactly_one_and_two_white_mutually_exclusive_not_contradictory :
  mutuallyExclusive exactlyOneWhite exactlyTwoWhite ∧
  ¬contradictory exactlyOneWhite exactlyTwoWhite :=
sorry

end NUMINAMATH_CALUDE_exactly_one_and_two_white_mutually_exclusive_not_contradictory_l129_12935


namespace NUMINAMATH_CALUDE_chorus_group_size_l129_12954

theorem chorus_group_size :
  let S := {n : ℕ | 100 < n ∧ n < 200 ∧
                    n % 3 = 1 ∧
                    n % 4 = 2 ∧
                    n % 6 = 4 ∧
                    n % 8 = 6}
  S = {118, 142, 166, 190} := by
  sorry

end NUMINAMATH_CALUDE_chorus_group_size_l129_12954


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l129_12911

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 12 → x ≥ 7 ∧ 7 < 3*7 - 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l129_12911


namespace NUMINAMATH_CALUDE_solutions_count_2x_3y_763_l129_12944

theorem solutions_count_2x_3y_763 : 
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 763 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 764) (Finset.range 764))).card = 127 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_2x_3y_763_l129_12944


namespace NUMINAMATH_CALUDE_initial_fuel_calculation_l129_12903

/-- Calculates the initial amount of fuel in a car's tank given its fuel consumption rate,
    journey distance, and remaining fuel after the journey. -/
theorem initial_fuel_calculation (consumption_rate : ℝ) (journey_distance : ℝ) (fuel_left : ℝ) :
  consumption_rate = 12 →
  journey_distance = 275 →
  fuel_left = 14 →
  (consumption_rate / 100) * journey_distance + fuel_left = 47 := by
  sorry

#check initial_fuel_calculation

end NUMINAMATH_CALUDE_initial_fuel_calculation_l129_12903


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l129_12984

theorem coin_and_die_probability :
  let coin_prob : ℚ := 2/3  -- Probability of heads for the biased coin
  let die_prob : ℚ := 1/6   -- Probability of rolling a 5 on a fair six-sided die
  coin_prob * die_prob = 1/9 := by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l129_12984


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l129_12946

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l129_12946


namespace NUMINAMATH_CALUDE_three_integer_sum_l129_12977

theorem three_integer_sum (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 →
  a * b * c = 343000 →
  Nat.gcd a b = 1 → Nat.gcd b c = 1 → Nat.gcd a c = 1 →
  a + b + c = 476 := by
sorry

end NUMINAMATH_CALUDE_three_integer_sum_l129_12977


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l129_12983

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_equation_solutions 
  (a b c : ℝ) 
  (h1 : f a b c (-2) = 3)
  (h2 : f a b c (-1) = 4)
  (h3 : f a b c 0 = 3)
  (h4 : f a b c 1 = 0)
  (h5 : f a b c 2 = -5) :
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = -4 ∧ f a b c x₁ = -5 ∧ f a b c x₂ = -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l129_12983


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l129_12913

/-- The complex number z -/
def z : ℂ := (-8 + Complex.I) * Complex.I

/-- A complex number is in the third quadrant if its real part is negative and its imaginary part is negative -/
def is_in_third_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im < 0

/-- Theorem: z is located in the third quadrant of the complex plane -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l129_12913


namespace NUMINAMATH_CALUDE_f_iter_formula_l129_12968

def f (x : ℝ) := 3 * x + 2

def f_iter : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ f_iter n

theorem f_iter_formula (n : ℕ) (x : ℝ) : 
  f_iter n x = 3^n * x + 3^n - 1 := by sorry

end NUMINAMATH_CALUDE_f_iter_formula_l129_12968


namespace NUMINAMATH_CALUDE_student_count_l129_12909

theorem student_count (n : ℕ) (right_rank left_rank : ℕ) 
  (h1 : right_rank = 6) 
  (h2 : left_rank = 5) 
  (h3 : n = right_rank + left_rank - 1) : n = 10 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l129_12909


namespace NUMINAMATH_CALUDE_bianca_not_recycled_bags_l129_12965

/-- The number of bags Bianca did not recycle -/
def bags_not_recycled (total_bags : ℕ) (points_per_bag : ℕ) (total_points : ℕ) : ℕ :=
  total_bags - (total_points / points_per_bag)

/-- Theorem stating that Bianca did not recycle 8 bags -/
theorem bianca_not_recycled_bags : bags_not_recycled 17 5 45 = 8 := by
  sorry

end NUMINAMATH_CALUDE_bianca_not_recycled_bags_l129_12965


namespace NUMINAMATH_CALUDE_max_value_a_plus_sqrt3b_l129_12975

theorem max_value_a_plus_sqrt3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.sqrt 3 * b = Real.sqrt ((1 - a) * (1 + a))) :
  ∃ (x : ℝ), ∀ (y : ℝ), (∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧
    Real.sqrt 3 * b' = Real.sqrt ((1 - a') * (1 + a')) ∧
    y = a' + Real.sqrt 3 * b') →
  y ≤ x ∧ x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_plus_sqrt3b_l129_12975


namespace NUMINAMATH_CALUDE_solve_linear_equation_l129_12970

theorem solve_linear_equation :
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 :=
by
  use -30
  constructor
  · -- Prove that x = -30 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check solve_linear_equation

end NUMINAMATH_CALUDE_solve_linear_equation_l129_12970


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l129_12961

theorem sum_of_roots_equals_one :
  ∀ x y : ℝ, (x + 3) * (x - 4) = 18 ∧ (y + 3) * (y - 4) = 18 → x + y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l129_12961


namespace NUMINAMATH_CALUDE_max_distance_point_to_circle_l129_12992

/-- The maximum distance between a point and a circle -/
theorem max_distance_point_to_circle :
  let P : ℝ × ℝ := (-1, -1)
  let center : ℝ × ℝ := (3, 0)
  let radius : ℝ := 2
  let circle := {(x, y) : ℝ × ℝ | (x - 3)^2 + y^2 = 4}
  (∀ Q ∈ circle, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt 17 + 2) ∧
  (∃ Q ∈ circle, Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 17 + 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_point_to_circle_l129_12992


namespace NUMINAMATH_CALUDE_cover_rectangles_l129_12949

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with a center point and radius -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Returns the number of circles needed to cover a rectangle -/
def circles_to_cover (r : Rectangle) (circle_radius : ℝ) : ℕ :=
  sorry

theorem cover_rectangles :
  let r1 := Rectangle.mk 6 3
  let r2 := Rectangle.mk 5 3
  let circle_radius := Real.sqrt 2
  (circles_to_cover r1 circle_radius = 6) ∧
  (circles_to_cover r2 circle_radius = 5) := by
  sorry

end NUMINAMATH_CALUDE_cover_rectangles_l129_12949


namespace NUMINAMATH_CALUDE_num_quadrilaterals_equals_choose_12_4_l129_12929

/-- The number of ways to choose 4 items from 12 items -/
def choose_12_4 : ℕ := 495

/-- The number of distinct points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

/-- Theorem: The number of different convex quadrilaterals formed by selecting 4 vertices 
    from 12 distinct points on the circumference of a circle is equal to choose_12_4 -/
theorem num_quadrilaterals_equals_choose_12_4 : 
  choose_12_4 = Nat.choose num_points vertices_per_quadrilateral := by
  sorry

#eval choose_12_4  -- This should output 495
#eval Nat.choose num_points vertices_per_quadrilateral  -- This should also output 495

end NUMINAMATH_CALUDE_num_quadrilaterals_equals_choose_12_4_l129_12929


namespace NUMINAMATH_CALUDE_hike_remaining_distance_l129_12927

/-- Calculates the remaining distance of a hike given the total distance and distance already hiked. -/
def remaining_distance (total : ℕ) (hiked : ℕ) : ℕ :=
  total - hiked

/-- Proves that for a 36-mile hike with 9 miles already hiked, 27 miles remain. -/
theorem hike_remaining_distance :
  remaining_distance 36 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_hike_remaining_distance_l129_12927


namespace NUMINAMATH_CALUDE_number_of_parents_at_park_parents_at_park_l129_12942

/-- Given a group of people at a park, prove the number of parents. -/
theorem number_of_parents_at_park (num_girls : ℕ) (num_boys : ℕ) (num_groups : ℕ) (group_size : ℕ) : ℕ :=
  let total_people := num_groups * group_size
  let total_children := num_girls + num_boys
  total_people - total_children

/-- Prove that there are 50 parents at the park given the specified conditions. -/
theorem parents_at_park : number_of_parents_at_park 14 11 3 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_parents_at_park_parents_at_park_l129_12942


namespace NUMINAMATH_CALUDE_prime_fraction_equation_l129_12981

theorem prime_fraction_equation (p q : ℕ) (hp : Prime p) (hq : Prime q) (n : ℕ+) 
  (h : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / (p * q) = (1 : ℚ) / n) :
  (p = 2 ∧ q = 3 ∧ n = 1) ∨ (p = 3 ∧ q = 2 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_equation_l129_12981


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l129_12916

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon) : ℕ := p.n

/-- The smallest positive angle of rotational symmetry for a regular polygon (in degrees) -/
def rotationalSymmetryAngle (p : RegularPolygon) : ℚ := 360 / p.n

/-- The theorem to be proved -/
theorem regular_18gon_symmetry_sum :
  let p : RegularPolygon := ⟨18, by norm_num⟩
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l129_12916


namespace NUMINAMATH_CALUDE_steps_climbed_fraction_l129_12901

/-- Proves that climbing 25 steps in a 6-floor building with 12 steps between floors
    is equivalent to climbing 5/12 of the total steps. -/
theorem steps_climbed_fraction (total_floors : Nat) (steps_per_floor : Nat) (steps_climbed : Nat) :
  total_floors = 6 →
  steps_per_floor = 12 →
  steps_climbed = 25 →
  (steps_climbed : Rat) / ((total_floors - 1) * steps_per_floor) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_steps_climbed_fraction_l129_12901


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l129_12986

theorem fourth_root_equation_solution :
  ∃ x : ℝ, (x^(1/4) * (x^5)^(1/8) = 4) ∧ (x = 4^(8/7)) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l129_12986


namespace NUMINAMATH_CALUDE_isosceles_triangle_l129_12994

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.cos t.B * Real.sin t.A = Real.sin t.C) : 
  t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l129_12994


namespace NUMINAMATH_CALUDE_black_area_after_three_cycles_l129_12948

/-- Represents the fraction of black area remaining after a number of cycles. -/
def blackAreaFraction (cycles : ℕ) : ℚ :=
  (2 / 3) ^ cycles

/-- The number of cycles in the problem. -/
def numCycles : ℕ := 3

/-- Theorem stating that after three cycles, 8/27 of the original area remains black. -/
theorem black_area_after_three_cycles :
  blackAreaFraction numCycles = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_three_cycles_l129_12948


namespace NUMINAMATH_CALUDE_bagel_store_expenditure_l129_12912

theorem bagel_store_expenditure (B D : ℝ) : 
  D = B / 2 →
  B = D + 15 →
  B + D = 45 := by sorry

end NUMINAMATH_CALUDE_bagel_store_expenditure_l129_12912


namespace NUMINAMATH_CALUDE_trig_identity_l129_12904

theorem trig_identity (α β : ℝ) :
  1 - Real.cos (β - α) + Real.cos α - Real.cos β =
  4 * Real.cos (α / 2) * Real.sin (β / 2) * Real.sin ((β - α) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l129_12904


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l129_12995

theorem ordering_of_expressions : 3^(1/5) > 0.2^3 ∧ 0.2^3 > Real.log 0.1 / Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l129_12995


namespace NUMINAMATH_CALUDE_congruence_and_range_implies_value_l129_12947

theorem congruence_and_range_implies_value :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -1234 [ZMOD 8] → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_congruence_and_range_implies_value_l129_12947


namespace NUMINAMATH_CALUDE_number_of_pigs_l129_12920

theorem number_of_pigs (total_cost : ℕ) (num_hens : ℕ) (avg_price_hen : ℕ) (avg_price_pig : ℕ) :
  total_cost = 1200 →
  num_hens = 10 →
  avg_price_hen = 30 →
  avg_price_pig = 300 →
  ∃ (num_pigs : ℕ), num_pigs = 3 ∧ total_cost = num_pigs * avg_price_pig + num_hens * avg_price_hen :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_pigs_l129_12920


namespace NUMINAMATH_CALUDE_john_weekly_earnings_l129_12940

/-- Calculates John's weekly earnings from crab fishing -/
def weekly_earnings (small_baskets medium_baskets large_baskets jumbo_baskets : ℕ)
  (small_per_basket medium_per_basket large_per_basket jumbo_per_basket : ℕ)
  (small_price medium_price large_price jumbo_price : ℕ) : ℕ :=
  (small_baskets * small_per_basket * small_price) +
  (medium_baskets * medium_per_basket * medium_price) +
  (large_baskets * large_per_basket * large_price) +
  (jumbo_baskets * jumbo_per_basket * jumbo_price)

theorem john_weekly_earnings :
  weekly_earnings 3 2 4 1 4 3 5 2 3 4 5 7 = 174 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_earnings_l129_12940


namespace NUMINAMATH_CALUDE_select_with_boys_l129_12960

theorem select_with_boys (num_boys num_girls : ℕ) : 
  num_boys = 6 → num_girls = 4 → 
  (2^(num_boys + num_girls) - 2^num_girls) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_select_with_boys_l129_12960


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l129_12971

-- Problem 1
theorem problem_1 (a b : ℚ) (h1 : a = 2) (h2 : b = 1/3) :
  3 * (a^2 - a*b + 7) - 2 * (3*a*b - a^2 + 1) + 3 = 36 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : (x + 2)^2 + |y - 1/2| = 0) :
  5*x^2 - (2*x*y - 3*(1/3*x*y + 2) + 4*x^2) = 11 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l129_12971


namespace NUMINAMATH_CALUDE_work_completion_l129_12939

/-- The number of men in the first group -/
def first_group : ℕ := 18

/-- The number of days for the first group to complete the work -/
def first_days : ℕ := 30

/-- The number of days for the second group to complete the work -/
def second_days : ℕ := 36

/-- The number of men in the second group -/
def second_group : ℕ := (first_group * first_days) / second_days

theorem work_completion :
  second_group = 15 := by sorry

end NUMINAMATH_CALUDE_work_completion_l129_12939


namespace NUMINAMATH_CALUDE_smallest_multiple_with_factors_l129_12932

theorem smallest_multiple_with_factors : 
  ∀ n : ℕ+, 
    (936 * n : ℕ) % 2^5 = 0 ∧ 
    (936 * n : ℕ) % 3^3 = 0 ∧ 
    (936 * n : ℕ) % 11^2 = 0 → 
    n ≥ 4356 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_factors_l129_12932


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l129_12991

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l129_12991


namespace NUMINAMATH_CALUDE_coin_flip_probability_l129_12937

/-- Represents the outcome of a coin flip -/
inductive CoinOutcome
  | Heads
  | Tails

/-- Represents the set of 5 coins -/
structure CoinSet :=
  (penny : CoinOutcome)
  (nickel : CoinOutcome)
  (dime : CoinOutcome)
  (quarter : CoinOutcome)
  (halfDollar : CoinOutcome)

/-- The total number of possible outcomes when flipping 5 coins -/
def totalOutcomes : Nat := 32

/-- Predicate for successful outcomes (penny, dime, and half-dollar are heads) -/
def isSuccessfulOutcome (cs : CoinSet) : Prop :=
  cs.penny = CoinOutcome.Heads ∧ cs.dime = CoinOutcome.Heads ∧ cs.halfDollar = CoinOutcome.Heads

/-- The number of successful outcomes -/
def successfulOutcomes : Nat := 4

/-- The probability of getting heads on penny, dime, and half-dollar -/
def probability : Rat := 1 / 8

theorem coin_flip_probability :
  (successfulOutcomes : Rat) / totalOutcomes = probability :=
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l129_12937


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l129_12982

theorem meaningful_fraction_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l129_12982


namespace NUMINAMATH_CALUDE_intersection_area_is_525_l129_12931

/-- A cube with edge length 30 units -/
def Cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 30}

/-- Point A of the cube -/
def A : Fin 3 → ℝ := λ _ ↦ 0

/-- Point B of the cube -/
def B : Fin 3 → ℝ := λ i ↦ if i = 0 then 30 else 0

/-- Point C of the cube -/
def C : Fin 3 → ℝ := λ i ↦ if i = 2 then 30 else B i

/-- Point D of the cube -/
def D : Fin 3 → ℝ := λ _ ↦ 30

/-- Point P on edge AB -/
def P : Fin 3 → ℝ := λ i ↦ if i = 0 then 10 else 0

/-- Point Q on edge BC -/
def Q : Fin 3 → ℝ := λ i ↦ if i = 0 then 30 else if i = 2 then 20 else 0

/-- Point R on edge CD -/
def R : Fin 3 → ℝ := λ i ↦ if i = 1 then 15 else 30

/-- The plane PQR -/
def PlanePQR : Set (Fin 3 → ℝ) :=
  {x | 3 * x 0 + 2 * x 1 - 3 * x 2 = 30}

/-- The intersection of the cube and the plane PQR -/
def Intersection : Set (Fin 3 → ℝ) :=
  Cube ∩ PlanePQR

/-- The area of the intersection -/
noncomputable def IntersectionArea : ℝ := sorry

theorem intersection_area_is_525 :
  IntersectionArea = 525 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_525_l129_12931


namespace NUMINAMATH_CALUDE_system_solution_l129_12953

theorem system_solution (x y : ℝ) : 
  (x = 5 ∧ y = -1) → (2 * x + 3 * y = 7 ∧ x = -2 * y + 3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l129_12953


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l129_12955

/-- Given a square with perimeter 40 meters, its area is 100 square meters. -/
theorem square_area_from_perimeter : 
  ∀ s : Real, 
  (4 * s = 40) → -- perimeter is 40 meters
  (s * s = 100)  -- area is 100 square meters
:= by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l129_12955


namespace NUMINAMATH_CALUDE_inequality_proof_l129_12938

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a^3 / (b^2 - 1)) + (b^3 / (c^2 - 1)) + (c^3 / (a^2 - 1)) ≥ (9 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l129_12938


namespace NUMINAMATH_CALUDE_devonshire_cows_cost_l129_12926

/-- The number of hearts in a standard deck of 52 playing cards -/
def hearts_in_deck : ℕ := 13

/-- The number of cows in Devonshire -/
def cows_in_devonshire : ℕ := 2 * hearts_in_deck

/-- The price of each cow in dollars -/
def price_per_cow : ℕ := 200

/-- The total cost of all cows in Devonshire when sold -/
def total_cost : ℕ := cows_in_devonshire * price_per_cow

theorem devonshire_cows_cost : total_cost = 5200 := by
  sorry

end NUMINAMATH_CALUDE_devonshire_cows_cost_l129_12926


namespace NUMINAMATH_CALUDE_tank_emptying_time_difference_l129_12972

/-- Proves the time difference for emptying a tank with and without an inlet pipe. -/
theorem tank_emptying_time_difference 
  (tank_capacity : ℝ) 
  (outlet_rate : ℝ) 
  (inlet_rate : ℝ) 
  (h1 : tank_capacity = 21600) 
  (h2 : outlet_rate = 2160) 
  (h3 : inlet_rate = 960) : 
  (tank_capacity / outlet_rate) - (tank_capacity / (outlet_rate - inlet_rate)) = 8 := by
  sorry

#check tank_emptying_time_difference

end NUMINAMATH_CALUDE_tank_emptying_time_difference_l129_12972


namespace NUMINAMATH_CALUDE_qz_length_l129_12998

/-- A quadrilateral ABZY with a point Q on the intersection of AZ and BY -/
structure Quadrilateral :=
  (A B Y Z Q : ℝ × ℝ)
  (AB_parallel_YZ : (A.2 - B.2) / (A.1 - B.1) = (Y.2 - Z.2) / (Y.1 - Z.1))
  (Q_on_AZ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • A + t • Z)
  (Q_on_BY : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • B + s • Y)
  (AZ_length : Real.sqrt ((A.1 - Z.1)^2 + (A.2 - Z.2)^2) = 42)
  (BQ_length : Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) = 12)
  (QY_length : Real.sqrt ((Q.1 - Y.1)^2 + (Q.2 - Y.2)^2) = 24)

/-- The length of QZ in the given quadrilateral is 28 -/
theorem qz_length (quad : Quadrilateral) :
  Real.sqrt ((quad.Q.1 - quad.Z.1)^2 + (quad.Q.2 - quad.Z.2)^2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_qz_length_l129_12998


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l129_12924

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  let c := 4
  F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ellipse x y

-- Theorem statement
theorem ellipse_triangle_perimeter
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_ellipse : point_on_ellipse P)
  (h_foci : foci F₁ F₂) :
  let perimeter := dist P F₁ + dist P F₂ + dist F₁ F₂
  perimeter = 18 :=
sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l129_12924


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l129_12921

theorem triangle_abc_properties (b c : ℝ) (A B : ℝ) :
  A = π / 3 →
  3 * b = 2 * c →
  (1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →
  b = 2 ∧ Real.sin B = Real.sqrt 21 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l129_12921


namespace NUMINAMATH_CALUDE_rhombus_side_length_l129_12969

-- Define a rhombus with area K and diagonals d and 3d
structure Rhombus where
  K : ℝ  -- Area of the rhombus
  d : ℝ  -- Length of the shorter diagonal
  h : K = (3/2) * d^2  -- Area formula for rhombus

-- Theorem: The side length of the rhombus is sqrt(5K/3)
theorem rhombus_side_length (r : Rhombus) : 
  ∃ s : ℝ, s^2 = (5/3) * r.K ∧ s > 0 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l129_12969


namespace NUMINAMATH_CALUDE_percent_democrat_voters_l129_12958

theorem percent_democrat_voters (D R : ℝ) : 
  D + R = 100 →
  0.85 * D + 0.20 * R = 59 →
  D = 60 :=
by sorry

end NUMINAMATH_CALUDE_percent_democrat_voters_l129_12958


namespace NUMINAMATH_CALUDE_gcd_a_b_is_one_or_three_l129_12950

def a (n : ℤ) : ℤ := n^5 + 6*n^3 + 8*n
def b (n : ℤ) : ℤ := n^4 + 4*n^2 + 3

theorem gcd_a_b_is_one_or_three (n : ℤ) : Nat.gcd (Int.natAbs (a n)) (Int.natAbs (b n)) = 1 ∨ Nat.gcd (Int.natAbs (a n)) (Int.natAbs (b n)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_a_b_is_one_or_three_l129_12950


namespace NUMINAMATH_CALUDE_find_n_l129_12923

theorem find_n : ∃ n : ℕ, 2^3 * 8 = 4^n ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l129_12923


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l129_12918

theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l129_12918


namespace NUMINAMATH_CALUDE_turnip_solution_l129_12967

/-- The number of turnips grown by Melanie, Benny, and Caroline, and the difference between
    the combined turnips of Melanie and Benny versus Caroline's turnips. -/
def turnip_problem (melanie_turnips benny_turnips caroline_turnips : ℕ) : Prop :=
  let combined_turnips := melanie_turnips + benny_turnips
  combined_turnips - caroline_turnips = 80

/-- The theorem stating the solution to the turnip problem -/
theorem turnip_solution : turnip_problem 139 113 172 := by
  sorry

end NUMINAMATH_CALUDE_turnip_solution_l129_12967


namespace NUMINAMATH_CALUDE_power_multiplication_l129_12987

theorem power_multiplication (x : ℝ) : x^4 * x^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l129_12987


namespace NUMINAMATH_CALUDE_polygon_sides_count_l129_12917

theorem polygon_sides_count : ∀ n : ℕ,
  (n ≥ 3) →
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 5 :=
by
  sorry

#check polygon_sides_count

end NUMINAMATH_CALUDE_polygon_sides_count_l129_12917


namespace NUMINAMATH_CALUDE_pear_juice_percentage_l129_12974

def pears_for_juice : ℕ := 4
def oranges_for_juice : ℕ := 3
def pear_juice_yield : ℚ := 12
def orange_juice_yield : ℚ := 6
def pears_in_blend : ℕ := 8
def oranges_in_blend : ℕ := 6

theorem pear_juice_percentage :
  let pear_juice_per_fruit : ℚ := pear_juice_yield / pears_for_juice
  let orange_juice_per_fruit : ℚ := orange_juice_yield / oranges_for_juice
  let total_pear_juice : ℚ := pear_juice_per_fruit * pears_in_blend
  let total_orange_juice : ℚ := orange_juice_per_fruit * oranges_in_blend
  let total_juice : ℚ := total_pear_juice + total_orange_juice
  (total_pear_juice / total_juice) * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_l129_12974


namespace NUMINAMATH_CALUDE_find_k_l129_12990

theorem find_k (a b c k : ℚ) : 
  (∀ x : ℚ, (a*x^2 + b*x + c + b*x^2 + a*x - 7 + k*x^2 + c*x + 3) / (x^2 - 2*x - 5) = 1) → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_find_k_l129_12990


namespace NUMINAMATH_CALUDE_museum_trip_ratio_l129_12963

theorem museum_trip_ratio : 
  let total_people : ℕ := 123
  let num_boys : ℕ := 50
  let num_staff : ℕ := 3  -- driver, assistant, and teacher
  let num_girls : ℕ := total_people - num_boys - num_staff
  (num_girls > num_boys) →
  (num_girls - num_boys : ℚ) / num_boys = 21 / 50 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_trip_ratio_l129_12963


namespace NUMINAMATH_CALUDE_factor_expression_l129_12962

theorem factor_expression (x : ℝ) : 2*x*(x+3) + (x+3) = (2*x+1)*(x+3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l129_12962


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l129_12985

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 11) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l129_12985


namespace NUMINAMATH_CALUDE_largest_positive_integer_theorem_l129_12908

/-- Binary operation @ defined as n @ n = n - (n * 5) -/
def binary_op (n : ℤ) : ℤ := n - (n * 5)

/-- Proposition: 1 is the largest positive integer n such that n @ n < 10 -/
theorem largest_positive_integer_theorem :
  ∀ n : ℕ, n > 1 → binary_op n ≥ 10 ∧ binary_op 1 < 10 := by
  sorry

end NUMINAMATH_CALUDE_largest_positive_integer_theorem_l129_12908


namespace NUMINAMATH_CALUDE_beth_initial_coins_l129_12933

theorem beth_initial_coins (initial_coins : ℕ) : 
  (initial_coins + 35) / 2 = 80 → initial_coins = 125 := by
  sorry

end NUMINAMATH_CALUDE_beth_initial_coins_l129_12933


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l129_12973

theorem fixed_point_of_exponential_function (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-2) + 3
  f 2 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l129_12973


namespace NUMINAMATH_CALUDE_mixed_grains_calculation_l129_12952

/-- Calculates the amount of mixed grains in a batch of rice -/
theorem mixed_grains_calculation (total_rice : ℝ) (sample_size : ℝ) (mixed_in_sample : ℝ) :
  total_rice * (mixed_in_sample / sample_size) = 150 :=
by
  -- Assuming total_rice = 1500, sample_size = 200, and mixed_in_sample = 20
  have h1 : total_rice = 1500 := by sorry
  have h2 : sample_size = 200 := by sorry
  have h3 : mixed_in_sample = 20 := by sorry
  
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mixed_grains_calculation_l129_12952


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l129_12956

def g (x : ℝ) : ℝ := -3 * x^4 + 5

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) := by
  sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l129_12956


namespace NUMINAMATH_CALUDE_function_equation_solution_l129_12966

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) = f x * y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l129_12966


namespace NUMINAMATH_CALUDE_combined_salaries_l129_12900

theorem combined_salaries (salary_A : ℕ) (num_people : ℕ) (avg_salary : ℕ) : 
  salary_A = 8000 → 
  num_people = 5 → 
  avg_salary = 9000 → 
  (avg_salary * num_people - salary_A = 37000) := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l129_12900


namespace NUMINAMATH_CALUDE_triangle_properties_l129_12906

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.a + t.c) / (t.a + t.b) = (t.b - t.a) / t.c ∧
  t.b = Real.sqrt 14 ∧
  Real.sin t.A = 2 * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.B = 2 * Real.pi / 3 ∧ min t.a (min t.b t.c) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l129_12906


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_reciprocals_l129_12925

/-- Given real numbers x and y satisfying x + y = 6 and x * y = 5,
    prove that x + (x^3 / y^2) + (y^3 / x^2) + y = 137.04 -/
theorem sum_of_cubes_and_reciprocals (x y : ℝ) 
  (h1 : x + y = 6) (h2 : x * y = 5) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 137.04 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_reciprocals_l129_12925


namespace NUMINAMATH_CALUDE_calculate_expression_l129_12999

theorem calculate_expression : 
  Real.sqrt 4 - abs (-1/4 : ℝ) + (π - 2)^0 + 2^(-2 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l129_12999


namespace NUMINAMATH_CALUDE_tan_30_plus_4sin_30_l129_12996

/-- The tangent of 30 degrees plus 4 times the sine of 30 degrees equals (√3)/3 + 2 -/
theorem tan_30_plus_4sin_30 : Real.tan (30 * π / 180) + 4 * Real.sin (30 * π / 180) = (Real.sqrt 3) / 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_4sin_30_l129_12996


namespace NUMINAMATH_CALUDE_position_number_difference_l129_12910

structure Student where
  initial_i : ℤ
  initial_j : ℤ
  new_m : ℤ
  new_n : ℤ

def movement (s : Student) : ℤ × ℤ :=
  (s.initial_i - s.new_m, s.initial_j - s.new_n)

def position_number (s : Student) : ℤ :=
  let (a, b) := movement s
  a + b

def sum_position_numbers (students : List Student) : ℤ :=
  students.map position_number |>.sum

theorem position_number_difference (students : List Student) :
  ∃ (S_max S_min : ℤ),
    (∀ s, sum_position_numbers s ≤ S_max ∧ sum_position_numbers s ≥ S_min) ∧
    S_max - S_min = 12 :=
sorry

end NUMINAMATH_CALUDE_position_number_difference_l129_12910


namespace NUMINAMATH_CALUDE_ellipse_equivalence_l129_12914

/-- Given an ellipse with equation 9x^2 + 4y^2 = 36, prove that the ellipse with equation
    x^2/20 + y^2/25 = 1 has the same foci and a minor axis length of 4√5 -/
theorem ellipse_equivalence (x y : ℝ) : 
  (∃ (a b c : ℝ), 9 * x^2 + 4 * y^2 = 36 ∧ 
   c^2 = a^2 - b^2 ∧
   x^2 / 20 + y^2 / 25 = 1 ∧
   b = 2 * (5 : ℝ).sqrt) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equivalence_l129_12914


namespace NUMINAMATH_CALUDE_club_membership_theorem_l129_12941

theorem club_membership_theorem :
  ∃ n : ℕ, n ≥ 300 ∧ n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m ≥ 300 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m % 11 = 0 → m ≥ n :=
by
  use 792
  sorry

end NUMINAMATH_CALUDE_club_membership_theorem_l129_12941


namespace NUMINAMATH_CALUDE_at_least_eight_composite_l129_12902

theorem at_least_eight_composite (n : ℕ) (h : n > 1000) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ 
  (∀ x ∈ s, x ≥ n ∧ x < n + 12) ∧
  (∃ (t : Finset ℕ), t ⊆ s ∧ t.card ≥ 8 ∧ ∀ y ∈ t, ¬ Nat.Prime y) := by
  sorry

end NUMINAMATH_CALUDE_at_least_eight_composite_l129_12902


namespace NUMINAMATH_CALUDE_bike_ride_time_l129_12993

/-- The time taken to ride a bike along semicircular paths on a highway -/
theorem bike_ride_time (highway_length : Real) (highway_width : Real) (speed : Real) : 
  highway_length = 2 → 
  highway_width = 60 / 5280 → 
  speed = 6 → 
  (π * highway_length / highway_width) / speed = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_time_l129_12993


namespace NUMINAMATH_CALUDE_lcm_36_100_l129_12989

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l129_12989


namespace NUMINAMATH_CALUDE_equivalent_statements_l129_12988

variable (P Q : Prop)

theorem equivalent_statements : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by sorry

end NUMINAMATH_CALUDE_equivalent_statements_l129_12988


namespace NUMINAMATH_CALUDE_sequence_property_l129_12980

def sequence_a (n : ℕ) : ℕ := sorry

theorem sequence_property : 
  ∃ (b c d : ℤ), 
    (∀ n : ℕ, n > 0 → sequence_a n = b * Int.floor (Real.sqrt (n + c)) + d) ∧ 
    sequence_a 1 = 1 ∧ 
    b + c + d = 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l129_12980


namespace NUMINAMATH_CALUDE_floor_abs_plus_const_l129_12928

theorem floor_abs_plus_const : 
  ⌊|(-47.3 : ℝ)| + 0.7⌋ = 48 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_plus_const_l129_12928


namespace NUMINAMATH_CALUDE_max_difference_with_broken_calculator_l129_12964

def is_valid_digit (d : ℕ) (valid_digits : List ℕ) : Prop :=
  d ∈ valid_digits

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem max_difference_with_broken_calculator :
  ∀ (a b c d e f : ℕ),
    is_valid_digit a [3, 5, 9] →
    is_valid_digit b [2, 3, 7] →
    is_valid_digit c [3, 4, 8, 9] →
    is_valid_digit d [2, 3, 7] →
    is_valid_digit e [3, 5, 9] →
    is_valid_digit f [1, 4, 7] →
    is_three_digit_number (100 * a + 10 * b + c) →
    is_three_digit_number (100 * d + 10 * e + f) →
    is_three_digit_number ((100 * a + 10 * b + c) - (100 * d + 10 * e + f)) →
    (100 * a + 10 * b + c) - (100 * d + 10 * e + f) ≤ 529 ∧
    (a = 9 ∧ b = 2 ∧ c = 3 ∧ d = 3 ∧ e = 9 ∧ f = 4 →
      ∀ (x y z u v w : ℕ),
        is_valid_digit x [3, 5, 9] →
        is_valid_digit y [2, 3, 7] →
        is_valid_digit z [3, 4, 8, 9] →
        is_valid_digit u [2, 3, 7] →
        is_valid_digit v [3, 5, 9] →
        is_valid_digit w [1, 4, 7] →
        is_three_digit_number (100 * x + 10 * y + z) →
        is_three_digit_number (100 * u + 10 * v + w) →
        is_three_digit_number ((100 * x + 10 * y + z) - (100 * u + 10 * v + w)) →
        (100 * x + 10 * y + z) - (100 * u + 10 * v + w) ≤ (100 * a + 10 * b + c) - (100 * d + 10 * e + f)) :=
by sorry

end NUMINAMATH_CALUDE_max_difference_with_broken_calculator_l129_12964


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_union_equals_A_l129_12945

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 5*x - 6 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 ≤ x ∧ x < a + 5}

-- Theorem 1: A ∩ B when a = 0
theorem intersection_when_a_zero : A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} := by sorry

-- Theorem 2: Range of values for a when A ∪ B = A
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a ∈ Set.Ioo 0 1 ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_union_equals_A_l129_12945


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l129_12976

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l129_12976


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l129_12957

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_6th : a 6 = 9)
  (h_3rd : a 3 = 3 * a 2) :
  a 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l129_12957
