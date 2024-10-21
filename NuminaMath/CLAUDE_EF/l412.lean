import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowerbed_area_approx_l412_41245

noncomputable def rectangle_length : ℝ := 45
noncomputable def rectangle_width : ℝ := 15
noncomputable def path_intersection : ℝ := rectangle_length / 3

noncomputable def triangular_area : ℝ := 
  (1 / 2) * path_intersection * (rectangle_width * path_intersection / rectangle_length)

theorem flowerbed_area_approx : 
  Int.floor (triangular_area + 0.5) = 38 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowerbed_area_approx_l412_41245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l412_41298

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 - (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

-- State the theorem
theorem f_properties :
  -- The smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  -- The maximum value in the interval [-π/4, π/3] is 2
  (∀ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/3 → f x ≤ 2) ∧
  (∃ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/3 ∧ f x = 2) ∧
  -- The minimum value in the interval [-π/4, π/3] is -√3
  (∀ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/3 → -Real.sqrt 3 ≤ f x) ∧
  (∃ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/3 ∧ f x = -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l412_41298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l412_41241

def a : ℕ → ℤ
| 0 => 0
| 1 => 1
| n + 2 => 2 * a (n + 1) + a n

theorem sequence_divisibility (k n : ℕ) :
  (2^k : ℤ) ∣ a n ↔ 2^k ∣ n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_divisibility_l412_41241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l412_41294

-- Define the triangle ABC and point P
variable (A B C P : EuclideanSpace ℝ (Fin 2))

-- Define the condition
variable (h : (P - A) + (P - B) + (P - C) = C - A)

-- Theorem to prove
theorem point_on_line_segment :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_segment_l412_41294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l412_41218

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h : ∀ n, a (n + 1) = q * a n

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then n * seq.a 0
  else seq.a 0 * (1 - seq.q ^ n) / (1 - seq.q)

theorem geometric_sequence_sum_ratio (seq : GeometricSequence) :
  2 * sumGeometric seq 3 = 7 * seq.a 1 →
  sumGeometric seq 5 / seq.a 1 = 31/2 ∨ sumGeometric seq 5 / seq.a 1 = 31/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l412_41218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l412_41222

/-- The y-intercept of a line is the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ × ℝ :=
  (0, c / b)

/-- The equation of the line is 4x + 7y = 28. -/
theorem y_intercept_of_line : y_intercept 4 7 28 = (0, 4) := by
  -- Unfold the definition of y_intercept
  unfold y_intercept
  -- Simplify the expression
  simp
  -- Check that 28 / 7 = 4
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l412_41222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l412_41253

/-- Two lines ax + by + c = 0 and dx + ey + f = 0 are parallel if and only if ae - bd = 0 and af - cd ≠ 0 -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  a * e - b * d = 0 ∧ a * f - c * d ≠ 0

/-- The given line x - 2y + 1 = 0 -/
def given_line : ℝ → ℝ → Prop :=
  fun x y ↦ x - 2 * y + 1 = 0

/-- The candidate line 2x - 4y + 1 = 0 -/
def candidate_line : ℝ → ℝ → Prop :=
  fun x y ↦ 2 * x - 4 * y + 1 = 0

theorem parallel_lines :
  are_parallel 1 (-2) 1 2 (-4) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l412_41253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_mult_two_equals_72x_l412_41281

variable (x : ℝ)

-- Define the denominators
def denom1 (x : ℝ) : ℝ := 4 * x
def denom2 (x : ℝ) : ℝ := 6 * x
def denom3 (x : ℝ) : ℝ := 9 * x

-- Define the LCM operation
def lcm_result (x : ℝ) : ℝ := (Nat.lcm (Nat.lcm 4 6) 9 : ℝ) * x

-- The theorem to prove
theorem lcm_mult_two_equals_72x (x : ℝ) : 2 * lcm_result x = 72 * x := by
  -- Expand the definition of lcm_result
  unfold lcm_result
  -- Simplify the LCM calculation
  have h1 : Nat.lcm (Nat.lcm 4 6) 9 = 36 := by norm_num
  rw [h1]
  -- Algebraic simplification
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_mult_two_equals_72x_l412_41281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_power_of_two_degree_one_l412_41220

/-- A monic polynomial that maps some positive integer to every power of 2 must be of degree 1 -/
theorem monic_polynomial_power_of_two_degree_one 
  (p : Polynomial ℝ) 
  (monic_p : Polynomial.Monic p) 
  (h : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ p.eval (m : ℝ) = 2^n) : 
  p.degree = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_polynomial_power_of_two_degree_one_l412_41220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_one_two_three_l412_41210

-- Define the nabla operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_one_two_three :
  nabla (nabla 1 2) 3 = 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_one_two_three_l412_41210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_length_squared_l412_41284

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1
noncomputable def g (x : ℝ) : ℝ := -3 * x
noncomputable def h (x : ℝ) : ℝ := x + 2
noncomputable def i (x : ℝ) : ℝ := -2 * x + 3
noncomputable def j (x : ℝ) : ℝ := max (f x) (max (g x) (max (h x) (i x)))

def interval : Set ℝ := Set.Icc (-2) 2

noncomputable def graphLength : ℝ := sorry

theorem graph_length_squared :
  graphLength ^ 2 = (21 * Real.sqrt 10 / 5 + 7 * Real.sqrt 5 / 5 + 3 * Real.sqrt 5 / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_length_squared_l412_41284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l412_41244

def lottery_size : ℕ := 45
def combination_size : ℕ := 6

theorem lottery_probability : 
  (6 * Nat.choose (lottery_size - combination_size) (combination_size - 1) : ℚ) / 
  Nat.choose lottery_size combination_size = 
  (6 * Nat.choose 39 5 : ℚ) / Nat.choose 45 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l412_41244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l412_41250

theorem cube_root_equation (x : ℝ) (h : x ≠ 0) :
  (3 + 2 / x) ^ (1/3 : ℝ) = 2 → x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l412_41250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_formula_l412_41261

noncomputable def geometric_sequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => -a * geometric_sequence a n

noncomputable def b (a : ℝ) (n : ℕ) : ℝ :=
  let a_n := geometric_sequence a n
  a_n * Real.log (abs a_n)

noncomputable def S (a : ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) (λ i => b a (i + 1))

theorem geometric_sum_formula (a : ℝ) (n : ℕ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  S a n = (a * Real.log (abs a) / (1 + a)^2) * (a + (-1)^(n+1) * (1 + n + n*a) * a^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_formula_l412_41261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l412_41243

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The point (2, 2√2) -/
noncomputable def point : ℝ × ℝ := (2, 2*Real.sqrt 2)

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_to_focus :
  parabola point.1 point.2 →
  distance point focus = 3 := by
  sorry

#check distance_to_focus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l412_41243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_inequality_l412_41211

/-- Given points A, B, C, D in a plane, and points M and N such that M is the midpoint of AB 
    and N is the midpoint of CD, prove that the length of MN is less than or equal to half 
    the sum of the lengths of AC and BD, and also less than or equal to half the sum of 
    the lengths of BC and AD. -/
theorem midpoint_inequality {n : Type*} [NormedAddCommGroup ℝ] [InnerProductSpace ℝ ℝ] 
  (A B C D M N : ℝ) : 
  M = (A + B) / 2 → N = (C + D) / 2 → 
  ‖M - N‖ ≤ (‖A - C‖ + ‖B - D‖) / 2 ∧ ‖M - N‖ ≤ (‖B - C‖ + ‖A - D‖) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_inequality_l412_41211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lowest_score_l412_41212

/-- Represents a contestant in the mathematical olympiad. -/
structure Contestant where
  id : Nat
  score : Nat

/-- Represents a problem in the mathematical olympiad. -/
structure Problem where
  id : Nat
  solvedBy : Finset Contestant
  points : Nat

/-- The mathematical olympiad setup. -/
structure MathOlympiad where
  contestants : Finset Contestant
  problems : Finset Problem
  ivanId : Nat
  scoreSystem : Problem → Nat

/-- The conditions of the mathematical olympiad. -/
def OlympiadConditions (mo : MathOlympiad) : Prop :=
  mo.contestants.card = 30
  ∧ mo.problems.card = 8
  ∧ ∀ p ∈ mo.problems, p.points = mo.scoreSystem p
  ∧ ∀ p ∈ mo.problems, mo.scoreSystem p = 30 - p.solvedBy.card
  ∧ ∃ ivan ∈ mo.contestants, ivan.id = mo.ivanId
      ∧ ∀ c ∈ mo.contestants, c.id ≠ mo.ivanId → ivan.score < c.score

/-- The theorem to be proved. -/
theorem max_lowest_score (mo : MathOlympiad) 
  (h : OlympiadConditions mo) : 
  ∃ ivan ∈ mo.contestants, ivan.id = mo.ivanId ∧ ivan.score = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lowest_score_l412_41212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_division_l412_41285

theorem cookie_division (total : ℝ) (total_pos : total > 0) : 
  (total - (4 / 8) * total - (3 / 8) * ((1 / 2) * total) - (1 / 8) * ((1 / 2) * total - (3 / 16) * total)) / total = 75 / 128 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_division_l412_41285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_pairs_in_arithmetic_progression_l412_41270

def ArithmeticProgression (α : Type*) [LinearOrderedField α] (f : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, f (n + 1) - f n = d

theorem no_real_pairs_in_arithmetic_progression :
  ¬ ∃ (a b : ℝ), ArithmeticProgression ℝ (λ n ↦ match n with
    | 0 => 15
    | 1 => a
    | 2 => b
    | 3 => a * b
    | _ => 0
  ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_pairs_in_arithmetic_progression_l412_41270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_people_round_table_l412_41217

/-- The number of distinct seating arrangements for n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of people to be seated around the round table. -/
def numberOfPeople : ℕ := 10

/-- Theorem stating that the number of distinct seating arrangements for 10 people
    around a round table is 362880. -/
theorem ten_people_round_table :
  roundTableArrangements numberOfPeople = 362880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_people_round_table_l412_41217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_final_result_l412_41293

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 10*y - 7 = -y^2 - 8*x + 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-4, 5)

-- Define the radius of the circle
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 13

-- Theorem stating the properties of the circle
theorem circle_properties :
  ∀ (x y : ℝ), circle_equation x y →
  ∃ (a b : ℝ), 
    circle_center = (a, b) ∧
    (x - a)^2 + (y - b)^2 = circle_radius^2 ∧
    a + b + circle_radius = 1 + 2 * Real.sqrt 13 :=
by sorry

-- Theorem for the final result
theorem final_result :
  let (a, b) := circle_center
  a + b + circle_radius = 1 + 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_final_result_l412_41293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_with_eccentricity_2_l412_41249

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and eccentricity e = 2, prove that its asymptotes have the equation y = ±√3 x -/
theorem hyperbola_asymptotes_with_eccentricity_2 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let c := e * a
  let hyperbola := λ (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := λ (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  c^2 = a^2 + b^2 → ∀ x y, asymptotes x y ↔ hyperbola x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_with_eccentricity_2_l412_41249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_speed_l412_41227

/-- Amanda's journey from her house to Kimberly's house -/
structure Journey where
  distance : ℚ  -- Distance in miles (using rational numbers)
  time : ℚ      -- Time in hours (using rational numbers)

/-- Calculate the speed of a journey in miles per hour -/
def speed (j : Journey) : ℚ := j.distance / j.time

theorem amanda_speed :
  let amanda_journey : Journey := { distance := 6, time := 3 }
  speed amanda_journey = 2 := by
  -- Unfold the definitions
  unfold speed
  unfold Journey.distance
  unfold Journey.time
  -- Simplify the rational number division
  simp
  -- The proof is complete
  rfl

#eval speed { distance := 6, time := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amanda_speed_l412_41227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_36_l412_41260

noncomputable def average_after_13_innings (initial_average : ℝ) : ℝ :=
  (12 * initial_average + 96) / 13

theorem new_average_is_36 (initial_average : ℝ) :
  average_after_13_innings initial_average = initial_average + 5 →
  average_after_13_innings initial_average = 36 :=
by
  intro h
  have : initial_average = 31 := by
    -- Proof that initial_average = 31
    sorry
  rw [this] at h ⊢
  simp [average_after_13_innings] at h ⊢
  -- Proof that (12 * 31 + 96) / 13 = 36
  sorry

#check new_average_is_36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_is_36_l412_41260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lion_in_room_3_l412_41264

-- Define the possible locations for the lion
inductive Room
  | Room1
  | Room2
  | Room3

-- Define the lion's location
variable (lion_location : Room)

-- Define the statements on each door
def statement (r : Room) : Prop :=
  match r with
  | Room.Room1 => (lion_location = Room.Room1)
  | Room.Room2 => (lion_location ≠ Room.Room2)
  | Room.Room3 => (2 + 3 = 2 * 3)

-- Theorem statement
theorem lion_in_room_3 :
  (∃! r : Room, statement lion_location r) →
  (lion_location = Room.Room3) := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lion_in_room_3_l412_41264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lynnelle_purple_cars_l412_41208

/-- The number of red cars Lynnelle has -/
def r : ℕ := sorry

/-- The number of purple cars Lynnelle has -/
def p : ℕ := sorry

/-- The number of green cars Lynnelle has -/
def g : ℕ := sorry

/-- The total number of cars Lynnelle and Moor have of each color -/
def total_cars : ℕ := 27

/-- The difference in total number of cars between Lynnelle and Moor -/
def car_difference : ℕ := 17

theorem lynnelle_purple_cars :
  (r + (total_cars - r) = total_cars) →  -- Total red cars is 27
  (p + (total_cars - p) = total_cars) →  -- Total purple cars is 27
  (g + (total_cars - g) = total_cars) →  -- Total green cars is 27
  (r = total_cars - g) →                 -- Lynnelle's red cars = Moor's green cars
  ((r + p + g) - ((total_cars - r) + (total_cars - p) + (total_cars - g)) = car_difference) →  -- Lynnelle has 17 more cars
  p = 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lynnelle_purple_cars_l412_41208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_interpretations_l412_41296

-- Define a generic function f
variable (f : ℝ → ℝ)

-- Define a point x₀
variable (x₀ : ℝ)

-- Define the slope of the tangent line
noncomputable def tangent_slope (f : ℝ → ℝ) (x₀ : ℝ) : ℝ := 
  deriv f x₀

-- Define a displacement function s
variable (s : ℝ → ℝ)

-- Define a time t₀
variable (t₀ : ℝ)

-- Define instantaneous velocity
noncomputable def inst_velocity (s : ℝ → ℝ) (t₀ : ℝ) : ℝ := 
  deriv s t₀

-- Define a velocity function v
variable (v : ℝ → ℝ)

-- Define instantaneous acceleration
noncomputable def inst_acceleration (v : ℝ → ℝ) (t₀ : ℝ) : ℝ := 
  deriv v t₀

-- Theorem stating the correctness of statements A, C, and D
theorem derivative_interpretations 
  (f : ℝ → ℝ) (x₀ : ℝ) (s : ℝ → ℝ) (v : ℝ → ℝ) (t₀ : ℝ) :
  (tangent_slope f x₀ = deriv f x₀) ∧ 
  (inst_velocity s t₀ = deriv s t₀) ∧ 
  (inst_acceleration v t₀ = deriv v t₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_interpretations_l412_41296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_nested_sqrt_l412_41201

/-- Nested square root function -/
noncomputable def nestedSqrt : ℕ → ℝ
  | 0 => 0
  | n + 1 => Real.sqrt (2 + nestedSqrt n)

/-- Main theorem -/
theorem cosine_nested_sqrt (n : ℕ) :
  2 * Real.cos (π / (2 ^ (n + 1))) = nestedSqrt n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_nested_sqrt_l412_41201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l412_41272

open Real

theorem theta_range (θ : ℝ) (h1 : θ ∈ Set.Icc 0 (π/2)) 
  (h2 : sin θ ^ 3 - cos θ ^ 3 ≥ log (cos θ / sin θ)) : 
  θ ∈ Set.Icc (π/4) (π/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_l412_41272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temp_is_43_l412_41255

/-- Represents the temperature on a given day -/
structure DayTemp where
  temp : ℝ

/-- Represents the temperatures for Tuesday to Friday -/
structure WeekTemp where
  tuesday : DayTemp
  wednesday : DayTemp
  thursday : DayTemp
  friday : DayTemp

noncomputable def WeekTemp.avgTueWedThu (w : WeekTemp) : ℝ :=
  (w.tuesday.temp + w.wednesday.temp + w.thursday.temp) / 3

noncomputable def WeekTemp.avgWedThuFri (w : WeekTemp) : ℝ :=
  (w.wednesday.temp + w.thursday.temp + w.friday.temp) / 3

theorem friday_temp_is_43 (w : WeekTemp) 
  (h1 : w.avgTueWedThu = 42)
  (h2 : w.avgWedThuFri = 44)
  (h3 : w.tuesday.temp = 37)
  (h4 : w.tuesday.temp = 43 ∨ w.wednesday.temp = 43 ∨ w.thursday.temp = 43 ∨ w.friday.temp = 43) :
  w.friday.temp = 43 := by
  sorry

#check friday_temp_is_43

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friday_temp_is_43_l412_41255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l412_41202

/-- Given a loan with principal and total repayment after one year, calculate the annual interest rate -/
noncomputable def annual_interest_rate (principal : ℝ) (total_repayment : ℝ) : ℝ :=
  ((total_repayment - principal) / principal) * 100

theorem interest_rate_calculation (principal total_repayment : ℝ) 
  (h1 : principal = 150)
  (h2 : total_repayment = 162) :
  annual_interest_rate principal total_repayment = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l412_41202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_parameters_l412_41290

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (x : ℝ) : ℝ := 
  (1 / Real.sqrt (8 * Real.pi)) * Real.exp (-(x^2) / 8)

/-- The mean of the distribution -/
def mean : ℝ := 0

/-- The standard deviation of the distribution -/
def std_dev : ℝ := 2

/-- Theorem stating that the given probability density function 
    corresponds to a normal distribution with mean 0 and standard deviation 2 -/
theorem normal_distribution_parameters : 
  (∀ x : ℝ, normal_pdf x = (1 / Real.sqrt (8 * Real.pi)) * Real.exp (-(x^2) / 8)) →
  mean = 0 ∧ std_dev = 2 := by
  intro h
  apply And.intro
  · -- Proof for mean = 0
    sorry
  · -- Proof for std_dev = 2
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_parameters_l412_41290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l412_41228

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define what it means for a point to be in a plane
def Point3D.in_plane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

-- Define a membership relation for Point3D and Plane3D
instance : Membership Point3D Plane3D where
  mem := Point3D.in_plane

-- Theorem: A triangle uniquely determines a plane in 3D space
theorem triangle_determines_plane (t : Triangle3D) : ∃! p : Plane3D, t.a ∈ p ∧ t.b ∈ p ∧ t.c ∈ p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l412_41228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sequence_recurrence_sequence_initial_conditions_l412_41246

/-- A sequence satisfying the given recurrence relation and initial conditions -/
def mySequence (n : ℕ) : ℝ :=
  2 * n

/-- The theorem stating the general formula for the sequence -/
theorem sequence_formula (n : ℕ) :
  mySequence n = 2 * n :=
by
  rfl  -- reflexivity proves this trivial equality

/-- The theorem stating that the sequence satisfies the recurrence relation -/
theorem sequence_recurrence (n : ℕ) :
  mySequence n - 2 * mySequence (n + 1) + mySequence (n + 2) = 0 :=
by
  simp [mySequence]
  ring

/-- The theorem stating that the sequence satisfies the initial conditions -/
theorem sequence_initial_conditions :
  mySequence 1 = 2 ∧ mySequence 2 = 4 :=
by
  simp [mySequence]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_sequence_recurrence_sequence_initial_conditions_l412_41246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_time_ratio_l412_41200

/-- Given the total time spent with a cat and the time spent petting,
    calculate the ratio of time spent combing to time spent petting. -/
theorem cat_time_ratio 
  (total_time petting_time : ℚ) 
  (h1 : total_time = 16)
  (h2 : petting_time = 12)
  (h3 : total_time = petting_time + (total_time - petting_time)) :
  (total_time - petting_time) / petting_time = 1 / 3 := by
  sorry

#check cat_time_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_time_ratio_l412_41200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_approx_3_62_l412_41291

/-- The length of the shortest side in an oblique projection of an equilateral triangle --/
noncomputable def shortest_side_length (side_length : ℝ) : ℝ :=
  let height := side_length * Real.sqrt 3 / 2
  Real.sqrt ((height / 2) ^ 2 + (side_length / 2) ^ 2 - height * (side_length / 2) * Real.sqrt 2 / 2)

/-- Theorem stating that the shortest side length in the oblique projection of an equilateral triangle
    with side length 10 is approximately 3.62 --/
theorem shortest_side_approx_3_62 :
  ∃ ε > 0, abs (shortest_side_length 10 - 3.62) < ε ∧ ε < 0.005 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_approx_3_62_l412_41291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_fraction_l412_41216

/-- Simple interest calculation -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem simple_interest_fraction :
  ∀ (principal : ℝ), principal > 0 →
  simple_interest principal 4 5 = principal / 5 := by
  intro principal h_pos
  unfold simple_interest
  field_simp
  ring
  -- The proof is completed, so we don't need 'sorry' here

#check simple_interest_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_fraction_l412_41216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l412_41219

noncomputable def f (x : ℝ) : ℝ := (x^3 + 5*x^2 + 8*x + 4) / (x + 1)
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem function_simplification_and_sum :
  (∀ x : ℝ, x ≠ -1 → f x = g x) ∧
  (1 + 4 + 4 + (-1) = 8) := by
  constructor
  · intro x hx
    sorry -- Proof of function equality
  · norm_num -- Proves the sum equality


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_simplification_and_sum_l412_41219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_q_fill_time_l412_41257

noncomputable def cistern_volume : ℝ := 1

noncomputable def pipe_p_rate : ℝ := 1 / 10

noncomputable def both_pipes_time : ℝ := 4

noncomputable def remaining_time : ℝ := 4.999999999999999

noncomputable def pipe_q_time : ℝ := 15

theorem pipe_q_fill_time :
  ∃ (pipe_q_rate : ℝ),
    pipe_q_rate > 0 ∧
    pipe_p_rate > 0 ∧
    (both_pipes_time * (pipe_p_rate + pipe_q_rate) + remaining_time * pipe_q_rate = cistern_volume) ∧
    pipe_q_time * pipe_q_rate = cistern_volume :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_q_fill_time_l412_41257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_approx_42_65_l412_41289

/-- Represents the car's journey on a hilly road -/
structure HillyJourney where
  uphill_min_speed : ℝ
  uphill_max_speed : ℝ
  downhill_min_speed : ℝ
  downhill_max_speed : ℝ
  uphill_distances : List ℝ
  downhill_distances : List ℝ

/-- Calculates the average speed of the car during the entire journey -/
noncomputable def average_speed (journey : HillyJourney) : ℝ :=
  let total_distance := (journey.uphill_distances.sum + journey.downhill_distances.sum)
  let avg_uphill_speed := (journey.uphill_min_speed + journey.uphill_max_speed) / 2
  let avg_downhill_speed := (journey.downhill_min_speed + journey.downhill_max_speed) / 2
  let uphill_time := journey.uphill_distances.sum / avg_uphill_speed
  let downhill_time := journey.downhill_distances.sum / avg_downhill_speed
  let total_time := uphill_time + downhill_time
  total_distance / total_time

/-- The main theorem stating the average speed of the car -/
theorem car_average_speed_approx_42_65 (journey : HillyJourney) 
  (h1 : journey.uphill_min_speed = 25)
  (h2 : journey.uphill_max_speed = 35)
  (h3 : journey.downhill_min_speed = 70)
  (h4 : journey.downhill_max_speed = 90)
  (h5 : journey.uphill_distances = [20, 25, 30, 25])
  (h6 : journey.downhill_distances = [40, 30, 20]) :
  ∃ ε > 0, |average_speed journey - 42.65| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_approx_42_65_l412_41289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_white_is_one_512_l412_41287

/-- Represents a 3x3 grid with each cell being either white or black -/
def Grid := Fin 3 → Fin 3 → Bool

/-- Probability of a single cell being white initially -/
noncomputable def initial_white_prob : ℝ := 1 / 2

/-- Rotates the grid 90 degrees clockwise -/
def rotate (g : Grid) : Grid :=
  fun i j => g (2 - j) i

/-- Applies the repainting rule after rotation -/
def repaint (g : Grid) (g_orig : Grid) : Grid :=
  fun i j => g i j || (¬ g_orig i j)

/-- Calculates the probability of the grid being all white after rotation and repainting -/
noncomputable def prob_all_white_after_rotation (g : Grid) : ℝ :=
  sorry

theorem prob_all_white_is_one_512 :
  ∃ g : Grid, prob_all_white_after_rotation g = 1 / 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_white_is_one_512_l412_41287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l412_41234

/-- 
A right parallelepiped with base sides a and b, acute angle 60° between them,
and longer base diagonal equal to shorter parallelepiped diagonal,
has volume (1/2) * a * b * √(6ab).
-/
theorem parallelepiped_volume 
  (a b : ℝ) 
  (h_positive : 0 < a ∧ 0 < b) 
  (h_angle : Real.cos (60 * π / 180) = 1/2) 
  (h_diagonal : a^2 + b^2 + a*b = a^2 + b^2 - a*b + (Real.sqrt (2*a*b))^2) :
  (1/2) * a * b * Real.sqrt (6*a*b) = 
    a * b * Real.sin (60 * π / 180) * Real.sqrt (2*a*b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l412_41234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l412_41204

-- Define the circle and ellipse
def circleEq (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def ellipseEq (x y : ℝ) : Prop := ((x - 3)^2 / 9) + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_circle_ellipse :
  ∃ (x1 y1 x2 y2 : ℝ),
    circleEq x1 y1 ∧ ellipseEq x2 y2 ∧
    ∀ (x3 y3 x4 y4 : ℝ),
      circleEq x3 y3 → ellipseEq x4 y4 →
      distance x1 y1 x2 y2 ≤ distance x3 y3 x4 y4 ∧
      distance x1 y1 x2 y2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_ellipse_l412_41204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotational_homothety_commutativity_l412_41221

-- Define a rotational homothety
structure RotationalHomothety (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] where
  center : α
  scale : ℝ
  angle : ℝ

-- Define the composition of rotational homotheties
noncomputable def compose {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (H₁ H₂ : RotationalHomothety α) : RotationalHomothety α :=
sorry

-- Define equality of rotational homotheties
def equal {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (H₁ H₂ : RotationalHomothety α) : Prop :=
sorry

-- State the theorem
theorem rotational_homothety_commutativity 
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] 
  (H₁ H₂ : RotationalHomothety α) :
  equal (compose H₁ H₂) (compose H₂ H₁) ↔ H₁.center = H₂.center :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotational_homothety_commutativity_l412_41221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_dinner_cost_l412_41230

/-- Calculate the total cost after discount for Pauline's taco dinner --/
theorem taco_dinner_cost : 
  (let taco_shells : ℚ := 5;
   let bell_peppers : ℚ := 4 * 1.5;
   let meat : ℚ := 2 * 3;
   let tomatoes : ℚ := 3 * 0.75;
   let shredded_cheese : ℚ := 4;
   let tortillas : ℚ := 2.5;
   let salsa : ℚ := 3.25;
   let total_before_discount : ℚ := taco_shells + bell_peppers + meat + tomatoes + shredded_cheese + tortillas + salsa;
   let discount_rate : ℚ := 0.15;
   let discount_amount : ℚ := discount_rate * total_before_discount;
   let total_after_discount : ℚ := total_before_discount - discount_amount;
   total_after_discount) = 24.65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_dinner_cost_l412_41230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l412_41297

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_property (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) :
  f x₁ + f x₂ = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l412_41297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l412_41235

theorem sequence_convergence (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ m n : ℕ, 0 < m → 0 < n → |a n - a m| ≤ (2 * (m : ℝ) * (n : ℝ)) / ((m^2 : ℝ) + (n^2 : ℝ))) :
  ∀ n : ℕ, a n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_convergence_l412_41235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_lambda_sum_constant_l412_41225

noncomputable section

/-- Ellipse C with given properties -/
def Ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

/-- Right focus of the ellipse -/
def RightFocus : ℝ × ℝ := (2, 0)

/-- Line l passing through right focus -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2)

/-- Point on the ellipse -/
def PointOnEllipse (x y : ℝ) : Prop := Ellipse x y ∧ ∃ k, Line k x y

/-- Intersection of line with y-axis -/
def IntersectionWithYAxis (k : ℝ) : ℝ × ℝ := (0, -2 * k)

/-- λ for a point on the ellipse -/
noncomputable def Lambda (x : ℝ) : ℝ := x / (2 - x)

theorem ellipse_lambda_sum_constant 
  (A B : ℝ × ℝ) 
  (hA : PointOnEllipse A.1 A.2) 
  (hB : PointOnEllipse B.1 B.2) 
  (k : ℝ) 
  (hk : Line k A.1 A.2 ∧ Line k B.1 B.2) :
  Lambda A.1 + Lambda B.1 = -10 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_lambda_sum_constant_l412_41225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_inequalities_identification_l412_41263

-- Define a predicate for linear inequalities with one variable
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ (∀ x, f x ↔ a * x > b ∨ a * x < b)

-- Define the given expressions
def expr1 : ℝ → Prop := λ x ↦ x > 0
def expr2 : ℝ → Prop := λ x ↦ 2 * x < -2 + x
def expr3 : ℝ → ℝ → Prop := λ x y ↦ x - y > -3
def expr4 : ℝ → Prop := λ x ↦ 4 * x = -1
noncomputable def expr5 : ℝ → Prop := λ a ↦ Real.sqrt (a + 1) ≥ 0
def expr6 : ℝ → Prop := λ x ↦ x^2 > 2

-- State the theorem
theorem linear_inequalities_identification :
  is_linear_inequality_one_var expr1 ∧
  is_linear_inequality_one_var expr2 ∧
  ¬is_linear_inequality_one_var (λ x ↦ ∃ y, expr3 x y) ∧
  ¬is_linear_inequality_one_var expr4 ∧
  ¬is_linear_inequality_one_var expr5 ∧
  ¬is_linear_inequality_one_var expr6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_inequalities_identification_l412_41263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_constructible_l412_41214

/-- A right-angled triangle with given altitude and angle bisector foot distance -/
structure RightTriangle where
  -- The length of the altitude from B to AC
  BB₁ : ℝ
  -- The distance from B to the foot of angle bisector AA₁
  BA₁ : ℝ
  -- Assumption that both lengths are positive
  BB₁_pos : BB₁ > 0
  BA₁_pos : BA₁ > 0

/-- The condition for the existence of a right-angled triangle with given parameters -/
def constructible (t : RightTriangle) : Prop :=
  t.BA₁ > t.BB₁ / 2

/-- Theorem stating the necessary and sufficient condition for triangle construction -/
theorem right_triangle_constructible (t : RightTriangle) :
  constructible t ↔ ∃ (A B C : ℝ × ℝ),
    -- B is at right angle
    (B.2 - A.2) * (C.1 - A.1) = (B.1 - A.1) * (C.2 - A.2) ∧
    -- BB₁ is the altitude
    t.BB₁ = abs ((C.2 - A.2) * B.1 + (A.1 - C.1) * B.2 + (C.1 * A.2 - A.1 * C.2)) /
            Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) ∧
    -- BA₁ is the distance to angle bisector foot
    t.BA₁ = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) /
            (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) + Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_constructible_l412_41214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_perpendicular_vectors_sum_l412_41224

/-- Hyperbola type -/
structure Hyperbola (a b : ℝ) where
  eq : (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1

/-- Point on a hyperbola -/
structure PointOnHyperbola (a b : ℝ) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_perpendicular_vectors_sum (a b : ℝ) (ha : 0 < a) (hb : a < b)
  (A B : PointOnHyperbola a b) (h_perp : A.x * B.x + A.y * B.y = 0) :
  1 / (A.x^2 + A.y^2) + 1 / (B.x^2 + B.y^2) = 1 / a^2 - 1 / b^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_perpendicular_vectors_sum_l412_41224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_and_perpendicular_vector_l412_41280

def A : Fin 3 → ℝ := ![-1, 2, 1]
def B : Fin 3 → ℝ := ![1, 2, 1]
def C : Fin 3 → ℝ := ![-1, 6, 4]

def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
def AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

noncomputable def parallelogram_area (v w : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (((v 1 * w 2 - v 2 * w 1)^2 + (v 2 * w 0 - v 0 * w 2)^2 + (v 0 * w 1 - v 1 * w 0)^2))

def is_perpendicular (v w : Fin 3 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 + v 2 * w 2 = 0

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (v 0^2 + v 1^2 + v 2^2)

theorem parallelogram_area_and_perpendicular_vector :
  (parallelogram_area AB AC = 10) ∧
  (∀ a : Fin 3 → ℝ, is_perpendicular a AB ∧ is_perpendicular a AC ∧ magnitude a = 10 →
    a = ![0, -6, 8] ∨ a = ![0, 6, -8]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_and_perpendicular_vector_l412_41280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_2_root_2_l412_41209

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def givenTriangle : Triangle where
  a := 3
  b := 3
  c := 2
  A := Real.arccos (7/9)
  B := Real.arccos (7/9)
  C := Real.arccos (7/9)

-- Theorem statement
theorem triangle_area_is_2_root_2 (t : Triangle) 
  (h1 : t.a + t.b = 6)
  (h2 : t.c = 2)
  (h3 : Real.cos t.C = 7/9) :
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 2 := by
  sorry

#check triangle_area_is_2_root_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_2_root_2_l412_41209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_when_tan_is_one_l412_41266

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) / 
  (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))

theorem f_equals_one_when_tan_is_one (α : ℝ) (h : Real.tan α = 1) : f α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_when_tan_is_one_l412_41266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l412_41223

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  sine_rule_ab : a / Real.sin A = b / Real.sin B
  sine_rule_bc : b / Real.sin B = c / Real.sin C
  cosine_rule_c : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.a - t.a * Real.cos t.B = t.b * Real.cos t.A) 
  (h2 : t.b = 4) 
  (h3 : Real.cos t.C = 1/4) : 
  t.c / t.a = 2 ∧ 
  (let S := (1/2) * t.a * t.b * Real.sin t.C; S^2 = 15) := by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l412_41223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolates_sold_l412_41203

/-- The number of chocolates sold at the selling price -/
def n : ℕ := 50

/-- The cost price of one chocolate -/
def C : ℝ := sorry

/-- The selling price of one chocolate -/
def S : ℝ := sorry

/-- The cost price of 65 chocolates equals the selling price of n chocolates -/
axiom cost_price_equals_selling_price : 65 * C = n * S

/-- The gain percent is 30% -/
axiom gain_percent : (S - C) / C = 0.3

theorem chocolates_sold : n = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolates_sold_l412_41203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_side_length_l412_41237

theorem min_square_side_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let min_side := if a < (Real.sqrt 2 + 1) * b
                   then a
                   else (Real.sqrt 2 / 2) * (a + b)
  ∀ s : ℝ, s ≥ min_side → ∃ (x y : ℝ), x^2 + y^2 ≤ s^2 ∧ (x = a ∨ y = b) :=
by
  sorry

#check min_square_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_square_side_length_l412_41237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_percentage_approx_l412_41276

noncomputable def cost_prices : List ℝ := [15000, 8000, 12000, 10000, 25000, 5000]
noncomputable def profit_loss_percentages : List ℝ := [-3, 10, -5, 8, 6, -4]

noncomputable def calculate_selling_price (cp : ℝ) (plp : ℝ) : ℝ :=
  cp * (1 + plp / 100)

noncomputable def total_cost_price : ℝ := cost_prices.sum
noncomputable def total_selling_price : ℝ := List.sum (List.zipWith calculate_selling_price cost_prices profit_loss_percentages)

noncomputable def overall_loss_percentage : ℝ :=
  (total_cost_price - total_selling_price) / total_cost_price * 100

theorem overall_loss_percentage_approx :
  ∃ ε > 0, |overall_loss_percentage - 2.87| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_loss_percentage_approx_l412_41276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l412_41295

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b ∧
  Real.cos t.B = 1/4 ∧
  t.a + t.b + t.c = 5

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  Real.sin t.C / Real.sin t.A = 2 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l412_41295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_enclosed_l412_41242

/-- The length of the rope in meters -/
def rope_length : ℚ := 100

/-- The maximum area that can be enclosed by a rectangular shape given a rope of fixed length -/
def max_enclosed_area (rope_length : ℚ) : ℚ := (rope_length / 4) ^ 2

/-- Theorem stating that the maximum area enclosed by a 100-meter rope in a rectangular shape is 625 square meters -/
theorem max_area_enclosed : max_enclosed_area rope_length = 625 := by
  -- Unfold the definition of max_enclosed_area
  unfold max_enclosed_area
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_enclosed_l412_41242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l412_41277

theorem rationalize_denominator : 
  ∃ (a b : ℝ) (h : b ≠ 0), (7 / Real.sqrt 98) * (a / b) = Real.sqrt 2 / 2 ∧ 
  (∀ x : ℝ, x^2 = 2 → x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_l412_41277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_between_parallel_lines_l412_41269

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A line on a 2D grid, represented by its slope and y-intercept -/
structure GridLine where
  slope : ℚ
  intercept : ℚ

/-- Checks if a point lies between two parallel lines -/
def isBetweenLines (p : GridPoint) (l1 l2 : GridLine) : Prop :=
  let y1 := l1.slope * p.x + l1.intercept
  let y2 := l2.slope * p.x + l2.intercept
  (p.y - y1) * (p.y - y2) ≤ 0

theorem infinite_points_between_parallel_lines 
  (l1 l2 : GridLine) 
  (h_parallel : l1.slope = l2.slope) 
  (h_distinct : l1.intercept ≠ l2.intercept) :
  ∃ (S : Set GridPoint), Set.Infinite S ∧ ∀ p ∈ S, isBetweenLines p l1 l2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_points_between_parallel_lines_l412_41269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_arc_division_l412_41259

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point on a circle -/
def PointOnCircle (p : Point) (c : Circle) : Prop := 
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The ratio in which a point divides an arc on a circle -/
noncomputable def DivideArcRatio (p : Point) (arc1 : Set Point) (arc2 : Set Point) (c : Circle) : ℝ → Prop :=
  λ r => ∃ (l1 l2 : ℝ), l1 / l2 = r ∧ 
    (∀ q ∈ arc1, PointOnCircle q c) ∧
    (∀ q ∈ arc2, PointOnCircle q c) ∧
    (Set.ncard arc1 / Set.ncard arc2 : ℝ) = l1 / l2

theorem equilateral_triangle_arc_division 
  (ABC : Triangle) 
  (M : Point) 
  (c1 c2 : Circle) 
  (n : ℝ) :
  ABC.A ≠ ABC.B ∧ ABC.B ≠ ABC.C ∧ ABC.C ≠ ABC.A →
  ABC.A ≠ M ∧ ABC.C ≠ M →
  PointOnCircle ABC.A c1 ∧ PointOnCircle ABC.B c1 ∧ PointOnCircle M c1 →
  PointOnCircle ABC.B c2 ∧ PointOnCircle ABC.C c2 ∧ PointOnCircle M c2 →
  DivideArcRatio ABC.A {M, ABC.A} {ABC.A, ABC.B} c1 n →
  DivideArcRatio ABC.C {M, ABC.C} {ABC.C, ABC.B} c2 (2 * n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_arc_division_l412_41259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_masha_waiting_time_l412_41205

/-- Represents the rate at which a clock runs compared to real time -/
structure ClockRate where
  numerator : ℕ
  denominator : ℕ
  is_positive : 0 < denominator

/-- Calculates the real time elapsed given a duration on a clock with a specific rate -/
noncomputable def realTimeElapsed (duration : ℝ) (rate : ClockRate) : ℝ :=
  duration * (rate.denominator : ℝ) / (rate.numerator : ℝ)

theorem petya_masha_waiting_time :
  let petya_rate : ClockRate := ⟨13, 12, by norm_num⟩
  let masha_rate : ClockRate := ⟨13, 15, by norm_num⟩
  let agreed_time : ℝ := 18.5  -- 6:30 PM in hours since midnight
  let petya_real_arrival := realTimeElapsed agreed_time petya_rate
  let masha_real_arrival := realTimeElapsed agreed_time masha_rate
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |masha_real_arrival - petya_real_arrival - (4 + 16/60)| < ε := by
  sorry

#check petya_masha_waiting_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petya_masha_waiting_time_l412_41205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_first_four_terms_mySequence_general_term_l412_41240

def mySequence (n : ℕ+) : ℚ := 1 / n.val

theorem mySequence_first_four_terms :
  (mySequence 1 = 1) ∧
  (mySequence 2 = 1/2) ∧
  (mySequence 3 = 1/3) ∧
  (mySequence 4 = 1/4) :=
by
  constructor
  · rfl
  · constructor
    · rfl
    · constructor
      · rfl
      · rfl

-- General term formula theorem
theorem mySequence_general_term (n : ℕ+) :
  mySequence n = 1 / n.val :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_first_four_terms_mySequence_general_term_l412_41240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_squares_8x9_l412_41232

/-- The number of squares of size k × k in an m × n rectangular board -/
def count_squares (m n k : ℕ) : ℕ := (m - k + 1) * (n - k + 1)

/-- The total number of squares in an m × n rectangular board -/
def total_squares (m n : ℕ) : ℕ :=
  Finset.sum (Finset.range (min m n + 1)) (λ k => count_squares m n k)

/-- Theorem: The total number of squares in an 8 × 9 rectangular board is 240 -/
theorem total_squares_8x9 :
  total_squares 8 9 = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_squares_8x9_l412_41232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l412_41282

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | n + 2 => 7 * sequence_a (n + 1) - sequence_a n

theorem perfect_square_property (n : ℕ) : ∃ b : ℤ, b ^ 2 = sequence_a n + 2 + sequence_a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_property_l412_41282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_probability_l412_41278

def M : Finset Int := {-2, -1, 0, 1, 2, 3}
def N : Finset Int := {-3, -2, -1, 0, 1, 2}

def is_meaningful (m n : Int) : Prop :=
  m ≠ 0 ∧ n ≠ 0

def represents_hyperbola (m n : Int) : Prop :=
  (m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0)

def total_meaningful_pairs : ℕ :=
  (M.filter (λ m => m ≠ 0)).card * (N.filter (λ n => n ≠ 0)).card

def hyperbola_pairs : ℕ :=
  (M.filter (λ m => m > 0)).card * (N.filter (λ n => n < 0)).card +
  (M.filter (λ m => m < 0)).card * (N.filter (λ n => n > 0)).card

theorem hyperbola_probability :
  (hyperbola_pairs : ℚ) / total_meaningful_pairs = 13 / 25 := by
  -- Evaluate the sets
  have h1 : M.filter (λ m => m ≠ 0) = {-2, -1, 1, 2, 3} := by rfl
  have h2 : N.filter (λ n => n ≠ 0) = {-3, -2, -1, 1, 2} := by rfl
  have h3 : M.filter (λ m => m > 0) = {1, 2, 3} := by rfl
  have h4 : N.filter (λ n => n < 0) = {-3, -2, -1} := by rfl
  have h5 : M.filter (λ m => m < 0) = {-2, -1} := by rfl
  have h6 : N.filter (λ n => n > 0) = {1, 2} := by rfl

  -- Calculate the cardinalities
  have c1 : (M.filter (λ m => m ≠ 0)).card = 5 := by rw [h1]; rfl
  have c2 : (N.filter (λ n => n ≠ 0)).card = 5 := by rw [h2]; rfl
  have c3 : (M.filter (λ m => m > 0)).card = 3 := by rw [h3]; rfl
  have c4 : (N.filter (λ n => n < 0)).card = 3 := by rw [h4]; rfl
  have c5 : (M.filter (λ m => m < 0)).card = 2 := by rw [h5]; rfl
  have c6 : (N.filter (λ n => n > 0)).card = 2 := by rw [h6]; rfl

  -- Calculate the final result
  calc
    (hyperbola_pairs : ℚ) / total_meaningful_pairs
    = ((3 * 3 + 2 * 2) : ℚ) / (5 * 5) := by
      rw [hyperbola_pairs, total_meaningful_pairs]
      rw [c1, c2, c3, c4, c5, c6]
      norm_num
    _ = 13 / 25 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_probability_l412_41278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_focus_coordinate_l412_41275

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2*y^2 = 1

-- Define the x-coordinate of the right focus
noncomputable def right_focus_x : ℝ := Real.sqrt 6 / 2

-- Theorem statement
theorem right_focus_coordinate (x y : ℝ) :
  hyperbola_equation x y → x = right_focus_x ∧ y = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_focus_coordinate_l412_41275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l412_41229

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - 1/x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l412_41229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_toss_losers_specific_l412_41231

/-- Given a ratio of winners to losers and the number of winners,
    calculate the number of losers in the ring toss game. -/
def ring_toss_losers (ratio_winners : ℕ) (ratio_losers : ℕ) (winners : ℕ) : ℕ :=
  winners * ratio_losers / ratio_winners

/-- Theorem stating that the number of losers is 7 given the specific conditions -/
theorem ring_toss_losers_specific : ring_toss_losers 4 1 28 = 7 := by
  -- Unfold the definition of ring_toss_losers
  unfold ring_toss_losers
  -- Perform the calculation
  norm_num

-- Evaluate the function with the given values
#eval ring_toss_losers 4 1 28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_toss_losers_specific_l412_41231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l412_41247

theorem negation_of_forall_sin_leq_one :
  (¬ ∀ x : ℝ, x ≥ 0 → Real.sin x ≤ 1) ↔ (∃ x : ℝ, x ≥ 0 ∧ Real.sin x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_forall_sin_leq_one_l412_41247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_implies_a_equals_negative_three_l412_41248

-- Define the slopes of the lines
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def slope_l2 (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop := slope_l1 a * slope_l2 a = -1

-- Theorem statement
theorem lines_perpendicular_implies_a_equals_negative_three :
  ∀ a : ℝ, perpendicular a → a = -3 :=
by
  intro a h
  sorry

#check lines_perpendicular_implies_a_equals_negative_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_implies_a_equals_negative_three_l412_41248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l412_41215

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log x}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (U \ B) = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l412_41215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l412_41262

/-- The lateral surface area of a frustum of a right circular cone -/
noncomputable def lateralSurfaceAreaFrustum (R r h : ℝ) : ℝ :=
  Real.pi * (R + r) * Real.sqrt ((R - r)^2 + h^2)

/-- Theorem: The lateral surface area of a frustum of a right circular cone
    with lower base radius 8 inches, upper base radius 2 inches, and
    vertical height 6 inches is equal to 60√2π square inches. -/
theorem frustum_lateral_surface_area :
  lateralSurfaceAreaFrustum 8 2 6 = 60 * Real.sqrt 2 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_surface_area_l412_41262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l412_41239

noncomputable def x : ℝ := Real.arccos (-1/3)

theorem tan_2x_value :
  Real.cos x = -1/3 ∧ 
  π < x ∧ x < 3*π/2 →
  Real.tan (2*x) = -4*Real.sqrt 2/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l412_41239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_two_zeros_implies_a_range_l412_41283

/-- The function f(x) = ax^2 - x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

/-- The function g(x) = ln x -/
noncomputable def g (x : ℝ) : ℝ := Real.log x

/-- The function h(x) = f(x) - g(x) -/
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g x

/-- Theorem: If h(x) has two distinct zeros, then 0 < a < 1 -/
theorem h_two_zeros_implies_a_range (a : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) →
  0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_two_zeros_implies_a_range_l412_41283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_l412_41268

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem derivative_of_exp (x : ℝ) : 
  deriv f x = f x := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_l412_41268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_half_l412_41236

open BigOperators Real

/-- The sum of the infinite series ∑(n=1 to ∞) (n^2 + n - 1) / ((n + 2)!) is equal to 1/2. -/
theorem infinite_series_sum_equals_half :
  ∑' n : ℕ, (n^2 + n - 1 : ℝ) / (Nat.factorial (n + 2)) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_half_l412_41236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disprove_hypotheses_l412_41226

theorem disprove_hypotheses :
  (∃ k : ℕ, k ≥ 2 ∧
    (∃ start : ℕ, ∀ i ∈ Finset.range k,
      ∃ p : ℕ, p < k ∧ Nat.Prime p ∧ (p ∣ start + i))) ∧
  (∃ k : ℕ, k ≥ 2 ∧
    (∃ start : ℕ, ∀ i ∈ Finset.range k,
      ∃ j ∈ Finset.range k, i ≠ j ∧ ¬(Nat.gcd (start + i) (start + j) = 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disprove_hypotheses_l412_41226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_inequality_l412_41279

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 2 / 2
  | n + 1 => Real.sqrt 2 / 2 * Real.sqrt (1 - Real.sqrt (1 - a n ^ 2))

noncomputable def b : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (1 + b n ^ 2) - 1) / b n

theorem a_b_inequality : ∀ n : ℕ, 2^(n+2) * a n < Real.pi ∧ Real.pi < 2^(n+2) * b n := by
  sorry

#check a_b_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_b_inequality_l412_41279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yura_finish_day_l412_41292

/-- Represents the number of problems solved on a given day -/
def problems_solved (day : Nat) : Nat := 
  if day = 1 then 16
  else if day ≥ 2 then problems_solved (day - 1) - 1
  else 0

/-- The total number of problems in the textbook -/
def total_problems : Nat := 91

/-- The number of problems left after the third day -/
def problems_left_after_day_3 : Nat := 46

/-- The day when Yura finishes solving all problems -/
def finish_day : Nat := 7

theorem yura_finish_day :
  (problems_solved 1 + problems_solved 2 + problems_solved 3 = total_problems - problems_left_after_day_3) ∧
  (∀ d : Nat, d ≥ 2 → problems_solved d = problems_solved (d - 1) - 1) ∧
  (Finset.sum (Finset.range finish_day) (fun d => problems_solved (d + 1)) = total_problems) := by
  sorry

#eval Finset.sum (Finset.range finish_day) (fun d => problems_solved (d + 1))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yura_finish_day_l412_41292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_divisible_by_24_l412_41256

theorem divisor_sum_divisible_by_24 (n : ℕ) (h : 24 ∣ (n + 1)) :
  24 ∣ (Finset.sum (Nat.divisors n) id) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_divisible_by_24_l412_41256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_are_good_exists_infinite_good_set_disjoint_from_squares_l412_41254

-- Define the set of natural numbers
def ℕSet : Set ℕ := { n | n ≥ 0 }

-- Define a "good" subset of natural numbers
def IsGood (A : Set ℕ) : Prop :=
  ∀ n > 0, ∀ p q : ℕ, Nat.Prime p → Nat.Prime q → p ≠ q →
    (n - p ∈ A ∧ n - q ∈ A) → False

-- Define the set of perfect squares
def PerfectSquares : Set ℕ := { n | ∃ m : ℕ, n = m * m }

-- Define the set of odd powers of a prime
def OddPowersOfPrime (q : ℕ) : Set ℕ := { n | ∃ k : ℕ, n = q^(2*k+1) }

theorem perfect_squares_are_good :
  IsGood PerfectSquares := by
  sorry

theorem exists_infinite_good_set_disjoint_from_squares :
  ∃ q : ℕ, Nat.Prime q ∧ IsGood (OddPowersOfPrime q) ∧ Disjoint (OddPowersOfPrime q) PerfectSquares := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_are_good_exists_infinite_good_set_disjoint_from_squares_l412_41254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_streak_matches_l412_41213

theorem winning_streak_matches (M P Q : ℝ) (hP : 0 ≤ P ∧ P < 100) (hQ : P < Q ∧ Q < 100) :
  ∃ N : ℝ, N = (P * M - Q * M) / (Q - 100) ∧ 
     Q * (M + N) / 100 = P * M / 100 + N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_streak_matches_l412_41213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_part_of_proportional_division_l412_41267

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  let x := total / (a + b + c)
  min (min (a * x) (b * x)) (c * x) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_part_of_proportional_division_l412_41267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_calculation_l412_41258

/-- Calculates the original price of a book given its selling price and profit rate. -/
noncomputable def original_price (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

/-- Theorem: If a book is sold for Rs 80 with a 60% profit, then the original purchase price was Rs 50. -/
theorem book_price_calculation :
  let selling_price : ℝ := 80
  let profit_rate : ℝ := 0.6
  original_price selling_price profit_rate = 50 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval original_price 80 0.6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_calculation_l412_41258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_cube_l412_41288

theorem sphere_volume_circumscribing_cube (edge_length : ℝ) (h : edge_length = 2) :
  (4 / 3) * Real.pi * ((edge_length * Real.sqrt 3) / 2) ^ 3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_circumscribing_cube_l412_41288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_disconnected_regions_l412_41273

/-- Represents a chessboard square --/
structure Square where
  x : Nat
  y : Nat

/-- Represents a strip on the chessboard --/
structure Strip where
  start : Square
  length : Nat
  horizontal : Bool

/-- Represents a chessboard --/
structure Chessboard where
  size : Nat
  strips : List Strip

/-- Checks if two squares are adjacent --/
def adjacent (a b : Square) : Bool :=
  (a.x = b.x && (a.y + 1 = b.y || a.y = b.y + 1)) ||
  (a.y = b.y && (a.x + 1 = b.x || a.x = b.x + 1))

/-- Checks if a square is covered by a strip --/
def covered (s : Square) (strip : Strip) : Bool :=
  if strip.horizontal then
    s.y = strip.start.y && s.x >= strip.start.x && s.x < strip.start.x + strip.length
  else
    s.x = strip.start.x && s.y >= strip.start.y && s.y < strip.start.y + strip.length

/-- Checks if a square is empty (not covered by any strip) --/
def empty (s : Square) (board : Chessboard) : Bool :=
  !board.strips.any (covered s)

/-- Represents an empty region on the chessboard --/
structure EmptyRegion where
  squares : List Square

/-- Checks if two empty regions are disconnected --/
def disconnected (r1 r2 : EmptyRegion) : Bool :=
  !r1.squares.any (fun s1 => r2.squares.any (fun s2 => adjacent s1 s2))

/-- The main theorem --/
theorem max_disconnected_regions (m n : Nat) :
  ∀ (board : Chessboard),
    board.size = m →
    board.strips.length = n →
    (∀ (s1 s2 : Strip), s1 ∈ board.strips → s2 ∈ board.strips → s1 ≠ s2 →
      ∀ (sq : Square), ¬(covered sq s1 ∧ covered sq s2)) →
    (∀ (regions : List EmptyRegion),
      (∀ r, r ∈ regions → ∀ s, s ∈ r.squares → empty s board) →
      (∀ r1 r2, r1 ∈ regions → r2 ∈ regions → r1 ≠ r2 → disconnected r1 r2) →
      regions.length ≤ n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_disconnected_regions_l412_41273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_in_subdivided_triangle_l412_41238

/-- The set of vertices in a subdivision of a triangle. -/
def VerticesOfSubdivision (subdivision : Type) : Set Point :=
  sorry

/-- The cardinality of a subdivision of a triangle. -/
def CardinalityOfSubdivision (subdivision : Type) : ℕ :=
  sorry

/-- Given a triangle divided into 1000 smaller triangles, the maximum number of distinct points
    that can be vertices of these triangles is 1002. -/
theorem max_vertices_in_subdivided_triangle : ∃ (n : ℕ), n = 1002 ∧ 
  ∀ (m : ℕ), (∃ (subdivision : Type) (vertices : Finset Point), 
    (CardinalityOfSubdivision subdivision = 1000) ∧
    (VerticesOfSubdivision subdivision = vertices.toSet) →
    vertices.card ≤ n) ∧
  (∃ (optimal_subdivision : Type) (optimal_vertices : Finset Point),
    (CardinalityOfSubdivision optimal_subdivision = 1000) ∧
    (VerticesOfSubdivision optimal_subdivision = optimal_vertices.toSet) ∧
    optimal_vertices.card = n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vertices_in_subdivided_triangle_l412_41238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_l412_41252

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 1) / (x^2 + 1)

-- Define M as the maximum value of f(x)
noncomputable def M : ℝ := sSup (Set.range f)

-- Define N as the minimum value of f(x)
noncomputable def N : ℝ := sInf (Set.range f)

-- Theorem statement
theorem sum_of_max_and_min : M + N = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_l412_41252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_puzzle_l412_41274

/-- Represents the area of a house in the figure -/
noncomputable def house_area (base_area : ℝ) : ℝ := base_area + (base_area / 2)

/-- The total area of the figure consisting of three stacked houses -/
noncomputable def total_area (base_area : ℝ) : ℝ :=
  house_area base_area + house_area (base_area / 4) + house_area (base_area / 16)

/-- Theorem stating that if the total area is 35, the base area of the largest square is 16 -/
theorem house_puzzle (base_area : ℝ) :
  total_area base_area = 35 → base_area = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_puzzle_l412_41274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l412_41271

-- Define the function f(x) = x + 2^x
noncomputable def f (x : ℝ) : ℝ := x + (2 : ℝ)^x

-- State the theorem
theorem zero_in_interval :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 0, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l412_41271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_parallelism_relation_l412_41206

/-- A line in 3D space -/
structure Line3D where
  -- Define a line (placeholder definition)
  point : ℝ → ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane (placeholder definition)
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Predicate for a line being parallel to another line -/
def parallel_lines (l m : Line3D) : Prop :=
  -- Define line parallelism (placeholder)
  sorry

/-- Predicate for a line being parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  -- Define line-plane parallelism (placeholder)
  sorry

/-- Predicate for a line being contained in a plane -/
def line_in_plane (m : Line3D) (α : Plane3D) : Prop :=
  -- Define line containment in a plane (placeholder)
  sorry

theorem line_plane_parallelism_relation (l m : Line3D) (α : Plane3D) 
    (h : line_in_plane m α) : 
  ¬(∀ (l m : Line3D) (α : Plane3D), parallel_lines l m → parallel_line_plane l α) ∧
  ¬(∀ (l m : Line3D) (α : Plane3D), parallel_line_plane l α → parallel_lines l m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_parallelism_relation_l412_41206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_comparison_l412_41299

theorem function_comparison (a x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_x : x₁ < x₂) 
  (h_sum : x₁ + x₂ = 0) : 
  a * x₁^2 + 2 * a * x₁ + 4 < a * x₂^2 + 2 * a * x₂ + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_comparison_l412_41299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_speed_is_36_l412_41207

noncomputable section

/-- Jack's walking speed function -/
def jack_speed (x : ℝ) : ℝ := x^3 - 7*x^2 - 14*x

/-- Jill's distance function -/
def jill_distance (x : ℝ) : ℝ := x^3 + 3*x^2 - 90*x

/-- Jill's time function -/
def jill_time (x : ℝ) : ℝ := x + 10

/-- Jill's speed function -/
noncomputable def jill_speed (x : ℝ) : ℝ := jill_distance x / jill_time x

theorem same_speed_is_36 (x : ℝ) (h : x ≠ -10) :
  jack_speed x = jill_speed x → jack_speed x = 36 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_speed_is_36_l412_41207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_for_f_of_f_equals_2_l412_41251

-- Define the function f based on the graph
noncomputable def f : ℝ → ℝ 
| x => if x < -1 then -2*x
       else if x < 3 then 2*x + 3
       else -0.5*x + 6.5

-- State the theorem
theorem unique_x_for_f_of_f_equals_2 : ∃! x : ℝ, f (f x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_for_f_of_f_equals_2_l412_41251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_AEBF_l412_41233

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line passing through the origin with slope k
def line_through_origin (k : ℝ) (x y : ℝ) : Prop := y = k * x

-- Define points A and B
def point_A : ℝ × ℝ := (2, 0)
def point_B : ℝ × ℝ := (0, 1)

-- Define the area of quadrilateral AEBF as a function of k
noncomputable def area_AEBF (k : ℝ) : ℝ := 2 * Real.sqrt (1 + 4 / ((1/k) + 4*k))

theorem max_area_AEBF :
  ∀ k > 0, area_AEBF k ≤ 2 * Real.sqrt 2 ∧
  ∃ k > 0, area_AEBF k = 2 * Real.sqrt 2 :=
by
  sorry

#check max_area_AEBF

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_AEBF_l412_41233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_count_lower_bound_l412_41286

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- A predicate to check if four points are concyclic -/
def concyclic (p q r s : Point) : Prop := sorry

/-- A predicate to check if a point is inside a circle -/
def inside_circle (center : Point) (radius : ℝ) (p : Point) : Prop := sorry

/-- A predicate to check if a point is outside a circle -/
def outside_circle (center : Point) (radius : ℝ) (p : Point) : Prop := sorry

/-- The number of circles satisfying the condition -/
noncomputable def k (points : Finset Point) : ℕ := sorry

/-- The theorem to be proved -/
theorem circle_count_lower_bound (n : ℕ) (points : Finset Point) 
  (h1 : points.card = 2*n + 3)
  (h2 : ∀ (p q r : Point), p ∈ points → q ∈ points → r ∈ points → 
        p ≠ q → q ≠ r → p ≠ r → ¬collinear p q r)
  (h3 : ∀ (p q r s : Point), p ∈ points → q ∈ points → r ∈ points → s ∈ points → 
        p ≠ q → q ≠ r → r ≠ s → p ≠ r → p ≠ s → q ≠ s → ¬concyclic p q r s) :
  k points > (1 / Real.pi) * (Nat.choose (2*n + 3) 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_count_lower_bound_l412_41286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l412_41265

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a - t.c) * Real.cos t.B)
  (h2 : t.b = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l412_41265
