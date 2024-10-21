import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l155_15507

/-- Given a triangle with sides a, b, c, area A, and inscribed circle radius r,
    prove that r = A / s, where s is the semiperimeter of the triangle. -/
theorem inscribed_circle_radius (a b c A r : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sides : a = 30 ∧ b = 21 ∧ c = 15)
  (h_area : A = 77)
  (h_inscribed : A = r * ((a + b + c) / 2)) :
  r = A / ((a + b + c) / 2) := by
  sorry

#check inscribed_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l155_15507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_distance_to_tangent_point_l155_15502

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the line that point A is on
def line_A (x y : ℝ) : Prop := x - y - 2 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, -1)

-- Theorem for part (I)
theorem tangent_line_equation : 
  ∃ (x1 y1 x2 y2 : ℝ),
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧
    (∀ (x y : ℝ), x - 3*y + 2 = 0 ↔ (y - y1)*(x1 - 1) = (x - x1)*(y1 + 1) ∧ 
                                    (y - y2)*(x2 - 1) = (x - x2)*(y2 + 1)) :=
by sorry

-- Theorem for part (II)
theorem min_distance_to_tangent_point :
  ∃ (d : ℝ),
    d = 2 ∧
    (∀ (x y xt yt : ℝ), 
      line_A x y → circle_eq xt yt → 
      (x - xt)^2 + (y - yt)^2 ≥ d^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_distance_to_tangent_point_l155_15502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l155_15596

theorem angle_on_line (θ : ℝ) : 
  (∃ (x y : ℝ), y = 3 * x ∧ x > 0 ∧ (Real.sin θ = y / Real.sqrt (x^2 + y^2)) ∧ (Real.cos θ = x / Real.sqrt (x^2 + y^2))) →
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_on_line_l155_15596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_l155_15541

/-- Represents a point on the circle --/
structure CirclePoint where
  index : Fin 2020

/-- Represents a chord on the circle --/
structure Chord where
  start : CirclePoint
  end' : CirclePoint

/-- Determines if two chords intersect --/
def chords_intersect (c1 c2 : Chord) : Prop := sorry

/-- The set of all possible quintuples of points --/
def all_quintuples : Set (Fin 5 → CirclePoint) := sorry

/-- The set of quintuples where AB intersects CD but neither intersect AE --/
def valid_quintuples : Set (Fin 5 → CirclePoint) := sorry

/-- The probability of selecting a valid quintuple --/
noncomputable def probability : ℚ := sorry

/-- The main theorem stating the probability is 1/3 --/
theorem intersection_probability : probability = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_probability_l155_15541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_B_to_C_is_10_percent_l155_15578

/-- The interest rate at which B lent to C, given the conditions of the problem -/
noncomputable def interest_rate_B_to_C (principal : ℝ) (rate_A_to_B : ℝ) (years : ℝ) (gain_B : ℝ) : ℝ :=
  let interest_B_to_A := principal * rate_A_to_B * years / 100
  let interest_C_to_B := interest_B_to_A + gain_B
  (interest_C_to_B * 100) / (principal * years)

/-- Theorem stating that the interest rate at which B lent to C is 10% per annum -/
theorem interest_rate_B_to_C_is_10_percent :
  interest_rate_B_to_C 3500 10 3 157.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_B_to_C_is_10_percent_l155_15578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_relationship_x_values_l155_15539

/-- Represents the distance traveled west -/
def x : ℝ := sorry

/-- Represents the angle of turn to the left -/
def turn_angle : ℝ := 120

/-- Represents the distance traveled in the new direction -/
def new_distance : ℝ := 4

/-- Represents the final distance from the starting point -/
def final_distance : ℝ := 2

/-- Theorem stating the relationship between the distances and angles -/
theorem distance_relationship : 
  (x - new_distance * Real.cos ((π / 180) * (180 - turn_angle)))^2 + 
  (new_distance * Real.sin ((π / 180) * (180 - turn_angle)))^2 = 
  final_distance^2 := by
  sorry

/-- Theorem proving the possible values of x -/
theorem x_values : x = 2 * Real.sqrt 3 + 2 ∨ x = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_relationship_x_values_l155_15539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_negative_quarter_x_intercept_l155_15567

/-- A line in the coordinate plane. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The x-intercept of a line. -/
noncomputable def x_intercept (l : Line) : ℝ := -l.y_intercept / l.slope

/-- Theorem: For a line with y-intercept 0.25, its slope is -1/4 times its x-intercept. -/
theorem slope_is_negative_quarter_x_intercept (k : Line) (h : k.y_intercept = 0.25) :
  k.slope = -(1/4) * x_intercept k := by
  sorry

#check slope_is_negative_quarter_x_intercept

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_is_negative_quarter_x_intercept_l155_15567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_b_measure_l155_15561

/-- A parallelogram with vertices A, B, C, and D -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The measure of an angle in degrees -/
noncomputable def angle_measure (p : Parallelogram) (v : Fin 4) : ℝ := sorry

/-- The sum of adjacent angles in a parallelogram is 180° -/
axiom adjacent_angles_sum (p : Parallelogram) :
  angle_measure p 0 + angle_measure p 1 = 180

/-- Theorem: In a parallelogram, if angle A is 50°, then angle B is 130° -/
theorem angle_b_measure (p : Parallelogram) :
  angle_measure p 0 = 50 → angle_measure p 1 = 130 := by
  intro h
  have : angle_measure p 0 + angle_measure p 1 = 180 := adjacent_angles_sum p
  rw [h] at this
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_b_measure_l155_15561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_mult_count_for_specific_polynomial_l155_15518

/-- Represents a polynomial as a list of coefficients -/
def MyPolynomial (α : Type*) := List α

/-- Horner's method for polynomial evaluation -/
def horner_eval {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Number of multiplication operations in Horner's method -/
def horner_mult_count {α : Type*} (p : MyPolynomial α) : Nat :=
  p.length - 1

theorem horner_mult_count_for_specific_polynomial :
  let p : MyPolynomial ℤ := [5, 4, 3, 2, 1, 1]
  horner_mult_count p = 5 := by
  rfl

#eval horner_mult_count ([5, 4, 3, 2, 1, 1] : MyPolynomial ℤ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_mult_count_for_specific_polynomial_l155_15518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_l155_15584

-- Define the rowing speeds and times
noncomputable def ethan_speed : ℝ := 3
noncomputable def ethan_time : ℝ := 25 / 60
noncomputable def frank_speed : ℝ := 4
noncomputable def lucy_speed : ℝ := 2
noncomputable def lucy_time : ℝ := 45 / 60

-- Define the total distance function
noncomputable def total_distance : ℝ :=
  ethan_speed * ethan_time +
  frank_speed * (2 * ethan_time) +
  lucy_speed * lucy_time

-- Theorem statement
theorem total_distance_approx :
  ∃ ε > 0, |total_distance - 6.08| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_approx_l155_15584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_ordering_l155_15527

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function representing an inverse proportion -/
noncomputable def inverseProportionFunction (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_point_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hk : k < 0)
  (hA : y₁ = inverseProportionFunction k (-4))
  (hB : y₂ = inverseProportionFunction k (-2))
  (hC : y₃ = inverseProportionFunction k 3) :
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_ordering_l155_15527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_one_wins_l155_15503

/-- Represents a position on the chessboard -/
structure Position where
  x : Nat
  y : Nat

/-- Represents a player in the game -/
inductive Player
| One
| Two

/-- Represents the game state -/
structure GameState where
  board_size : Nat
  current_position : Position
  current_player : Player

/-- Checks if a position is on the main diagonal of the board -/
def is_on_diagonal (p : Position) : Bool :=
  p.x = p.y

/-- Checks if a position is in the opposite corner from the starting position -/
def is_opposite_corner (n : Nat) (p : Position) : Bool :=
  p.x = n - 1 && p.y = n - 1

/-- Applies the strategy to the game state -/
def apply_strategy (strategy : GameState → Position) (game : GameState) : GameState :=
  { game with current_position := strategy game }

/-- The main theorem stating that Player 1 always wins -/
theorem player_one_wins (n : Nat) (h : n > 3) :
  ∃ (strategy : GameState → Position),
    ∀ (game : GameState),
      game.board_size = n →
      game.current_position = ⟨0, 0⟩ →
      game.current_player = Player.One →
      ∃ (k : Nat), is_opposite_corner n (Nat.iterate (apply_strategy strategy) k game).current_position := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_one_wins_l155_15503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_l155_15528

variable {R : Type*} [Field R]

theorem matrix_equality (A B : Matrix (Fin 3) (Fin 3) R) 
  (hA : IsUnit A) (hB : IsUnit B) (hBA : IsUnit (B⁻¹ - A)) 
  (hABA : IsUnit (A⁻¹ + (B⁻¹ - A)⁻¹)) : 
  A - (A⁻¹ + (B⁻¹ - A)⁻¹)⁻¹ = A * B * A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_l155_15528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reach_area_l155_15597

/-- The area accessible by a dog tethered to a vertex of a regular hexagon --/
noncomputable def dog_accessible_area (side_length : ℝ) (rope_length : ℝ) : ℝ :=
  sorry  -- Implementation details omitted

/-- The area outside a regular hexagon that a dog can reach when tethered to a vertex --/
theorem dog_reach_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  ∃ (area : ℝ), area = 12 * Real.pi ∧ 
    area = dog_accessible_area side_length rope_length :=
by
  sorry  -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_reach_area_l155_15597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_exists_in_interval_l155_15565

-- Define the function f(x) = ln x - 3/x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 / x

-- State the theorem
theorem zero_exists_in_interval :
  (∀ x > 0, f x < f (x + 1)) →  -- f is increasing over its domain
  f 2 < 0 →                     -- f(2) < 0
  f 3 > 0 →                     -- f(3) > 0
  ∃ x ∈ Set.Ioo 2 3, f x = 0 :=  -- There exists a zero of f in the open interval (2, 3)
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_exists_in_interval_l155_15565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_sequence_equals_product_of_primes_l155_15572

def P (n : ℕ+) : ℕ := n * (n + 1) * (2 * n + 1) * (3 * n + 1) * (4 * n + 1) * (5 * n + 1) * (6 * n + 1) *
                      (7 * n + 1) * (8 * n + 1) * (9 * n + 1) * (10 * n + 1) * (11 * n + 1) * (12 * n + 1) *
                      (13 * n + 1) * (14 * n + 1) * (15 * n + 1) * (16 * n + 1)

def P_sequence : List ℕ := List.map (fun n => P ⟨n + 1, Nat.succ_pos n⟩) (List.range 2016)

theorem gcd_of_sequence_equals_product_of_primes :
  Nat.gcd (List.foldl Nat.gcd 0 P_sequence) (2 * 3 * 5 * 7 * 11 * 13 * 17) = 2 * 3 * 5 * 7 * 11 * 13 * 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_sequence_equals_product_of_primes_l155_15572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_price_l155_15551

/-- Calculates the discount amount based on the marked price --/
noncomputable def discount (price : ℝ) : ℝ :=
  if price ≤ 200 then 0
  else if price ≤ 500 then (price - 200) * 0.1
  else 300 * 0.1 + (price - 500) * 0.2

/-- The marked price of the appliance --/
def markedPrice : ℝ := 2000

/-- The amount saved through discounts --/
def savedAmount : ℝ := 330

theorem appliance_price :
  discount markedPrice = savedAmount :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_price_l155_15551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_calculation_l155_15521

/-- Represents the scale of the map in miles per inch -/
noncomputable def scale : ℝ := 300

/-- Represents the length of the short diagonal on the map in inches -/
noncomputable def shortDiagonalOnMap : ℝ := 10

/-- Calculates the real length of the short diagonal in miles -/
noncomputable def realShortDiagonal : ℝ := shortDiagonalOnMap * scale

/-- Calculates the area of the rhombus-shaped park in square miles -/
noncomputable def parkArea : ℝ := (1 / 2) * realShortDiagonal * realShortDiagonal

theorem park_area_calculation :
  parkArea = 4500000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_calculation_l155_15521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_usage_life_l155_15550

/-- The total cost function for the equipment usage -/
def total_cost (x : ℕ+) : ℝ := (x : ℝ)^2 + 2*(x : ℝ) + 100

/-- The annual average cost function -/
noncomputable def annual_avg_cost (x : ℕ+) : ℝ := total_cost x / (x : ℝ)

/-- Theorem stating that the optimal usage life is 10 years -/
theorem optimal_usage_life :
  ∀ x : ℕ+, x ≠ 10 → annual_avg_cost x > annual_avg_cost 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_usage_life_l155_15550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_parabola_points_distance_l155_15577

/-- Two points on a parabola symmetric about x + y = 0 -/
structure SymmetricParabolaPoints where
  a : ℝ
  b : ℝ
  on_parabola : b = 3 - a^2
  symmetric : -a = 3 - b^2

/-- The distance between two points -/
noncomputable def distance (p : SymmetricParabolaPoints) : ℝ :=
  Real.sqrt ((p.a + p.b)^2 + (p.b + p.a)^2)

/-- Theorem: The distance between symmetric points on the parabola is 3√2 -/
theorem symmetric_parabola_points_distance (p : SymmetricParabolaPoints) :
  distance p = 3 * Real.sqrt 2 := by
  sorry

#check symmetric_parabola_points_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_parabola_points_distance_l155_15577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l155_15549

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (Real.pi + ω * x) * Real.sin (3 * Real.pi / 2 - ω * x) - Real.cos (ω * x) ^ 2

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x ∧ ∀ t ∈ Set.Ioo 0 T, ∃ y, f (y + t) ≠ f y

theorem f_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_period : smallest_positive_period (f ω) Real.pi) :
  f ω (2 * Real.pi / 3) = -1 ∧
  ∀ (A B C a b c : ℝ),
    0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    a * Real.sin C = b * Real.sin A →
    b * Real.sin C = c * Real.sin B →
    c * Real.sin A = a * Real.sin B →
    (2 * a - c) * Real.cos B = b * Real.cos C →
    B = Real.pi / 3 ∧ -1 < f ω A ∧ f ω A ≤ 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l155_15549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_speed_to_gym_l155_15552

-- Define the constants from the problem
def distance_home_to_grocery : ℝ := 250
def distance_grocery_to_gym : ℝ := 360
def time_difference : ℝ := 70

-- Define Angelina's speed from home to grocery
noncomputable def speed_home_to_grocery : ℝ → ℝ := λ v => v

-- Define Angelina's speed from grocery to gym
noncomputable def speed_grocery_to_gym : ℝ → ℝ := λ v => 2 * v

-- Define the time taken from home to grocery
noncomputable def time_home_to_grocery : ℝ → ℝ := λ v => distance_home_to_grocery / (speed_home_to_grocery v)

-- Define the time taken from grocery to gym
noncomputable def time_grocery_to_gym : ℝ → ℝ := λ v => distance_grocery_to_gym / (speed_grocery_to_gym v)

-- State the theorem
theorem angelina_speed_to_gym :
  ∃ v : ℝ, v > 0 ∧ 
    time_home_to_grocery v - time_grocery_to_gym v = time_difference ∧
    speed_grocery_to_gym v = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angelina_speed_to_gym_l155_15552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l155_15511

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
noncomputable def distance_between_planes (p1 p2 : Plane) : ℝ :=
  abs (p1.d / Real.sqrt (p1.a^2 + p1.b^2 + p1.c^2) - p2.d / Real.sqrt (p2.a^2 + p2.b^2 + p2.c^2))

theorem distance_between_specific_planes :
  let p1 : Plane := { a := 3, b := -4, c := 12, d := 12 }
  let p2 : Plane := { a := 6, b := -8, c := 24, d := 48 }
  distance_between_planes p1 p2 = 12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l155_15511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_quadratic_trinomial_implies_unique_abc_l155_15532

theorem prime_quadratic_trinomial_implies_unique_abc (p : ℕ) 
  (hp : p > 1)
  (h_prime : ∀ x : ℕ, x < p → Nat.Prime (x^2 - x + p)) :
  ∃! (a b c : ℤ), b^2 - 4*a*c = 1 - 4*(p : ℤ) ∧ 0 < a ∧ a ≤ c ∧ -a ≤ b ∧ b < a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_quadratic_trinomial_implies_unique_abc_l155_15532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_calculation_l155_15589

/-- Calculates the length of a train given the speeds of two trains, time to clear, and length of the other train --/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (time_to_clear : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * time_to_clear
  total_distance - other_train_length

/-- The length of the first train is approximately 151.019 meters --/
theorem first_train_length_calculation :
  let speed1 := (80 : ℝ)
  let speed2 := (65 : ℝ)
  let time_to_clear := (7.844889650207294 : ℝ)
  let second_train_length := (165 : ℝ)
  let first_train_length := calculate_train_length speed1 speed2 time_to_clear second_train_length
  ∃ ε > 0, abs (first_train_length - 151.019) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_calculation_l155_15589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_equals_two_l155_15540

theorem cube_root_of_eight_equals_two : (2 : ℝ) ^ 3 = 8 := by
  norm_num

#eval (2 : ℝ) ^ 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_equals_two_l155_15540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_conical_tank_l155_15570

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Represents the water tank problem -/
def waterTankProblem (fullTank : Cone) (waterPercentage : ℝ) : Prop :=
  let waterHeight := 50 * Real.rpow 3.2 (1/3)
  fullTank.baseRadius = 20 ∧
  fullTank.height = 100 ∧
  waterPercentage = 0.4 ∧
  let waterTank : Cone := ⟨fullTank.baseRadius * (waterHeight / fullTank.height), waterHeight⟩
  coneVolume waterTank = waterPercentage * coneVolume fullTank

theorem water_height_in_conical_tank :
  ∀ (fullTank : Cone) (waterPercentage : ℝ),
  waterTankProblem fullTank waterPercentage →
  ∃ (waterHeight : ℝ), waterHeight = 50 * Real.rpow 3.2 (1/3) := by
  sorry

#check water_height_in_conical_tank

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_conical_tank_l155_15570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artificial_scarcity_effect_l155_15533

/-- Represents the number of watches consumers are willing to buy -/
def consumer_demand : ℕ := 3000

/-- Represents the price of each watch in currency units -/
def watch_price : ℕ := 15000

/-- Represents the number of watches actually produced by the firm -/
def actual_production : ℕ := 200

/-- Represents whether a firm implements an interview requirement -/
def has_interview_requirement : Prop := sorry

/-- Represents whether a firm can deny purchase based on history -/
def can_deny_purchase : Prop := sorry

/-- Represents the perceived value of the product -/
def perceived_value : ℕ → ℕ := sorry

/-- Represents the level of market control -/
def market_control : ℕ → ℕ := sorry

/-- Theorem stating that artificial scarcity leads to increased perceived value and market control -/
theorem artificial_scarcity_effect 
  (h1 : actual_production < consumer_demand)
  (h2 : has_interview_requirement)
  (h3 : can_deny_purchase) :
  perceived_value actual_production > perceived_value consumer_demand ∧ 
  market_control actual_production > market_control consumer_demand :=
by
  sorry

#check artificial_scarcity_effect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_artificial_scarcity_effect_l155_15533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_theorem_l155_15566

def checkerboard_size : ℕ := 10

def is_valid_square (n : ℕ) : Prop :=
  n ≥ 4 ∧ n ≤ checkerboard_size

def count_squares (n : ℕ) : ℕ :=
  if n ≥ 4 ∧ n ≤ checkerboard_size then (checkerboard_size + 1 - n) * (checkerboard_size + 1 - n) else 0

def total_valid_squares : ℕ := 
  Finset.sum (Finset.range (checkerboard_size - 3)) (λ i => count_squares (i + 4))

theorem checkerboard_theorem : total_valid_squares = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_theorem_l155_15566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_unit_square_distance_l155_15555

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a square with side length 1
def UnitSquare : Set Point := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem five_points_unit_square_distance (points : Finset Point) :
  points.card = 5 → (∀ p ∈ points, p ∈ UnitSquare) →
  ∃ p1 p2 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_points_unit_square_distance_l155_15555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_sin_x_l155_15562

theorem integral_sqrt_one_minus_x_squared_plus_sin_x : 
  ∫ x in (-1 : ℝ)..1, (Real.sqrt (1 - x^2) + Real.sin x) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_sin_x_l155_15562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l155_15504

/-- Represents a clock with 12 hours and 360 degrees -/
structure Clock :=
  (hours : Nat := 12)
  (degrees : Nat := 360)

/-- Calculates the position of the hour hand at a given time -/
noncomputable def hourHandPosition (c : Clock) (hour : Nat) (minute : Nat) : ℝ :=
  (hour % c.hours : ℝ) * (c.degrees : ℝ) / (c.hours : ℝ) + 
  (minute : ℝ) * (c.degrees : ℝ) / (c.hours : ℝ) / 60

/-- Calculates the position of the minute hand at a given time -/
noncomputable def minuteHandPosition (c : Clock) (minute : Nat) : ℝ :=
  (minute : ℝ) * (c.degrees : ℝ) / 60

/-- Calculates the smaller angle between two positions on a clock -/
noncomputable def smallerAngle (c : Clock) (pos1 : ℝ) (pos2 : ℝ) : ℝ :=
  min (abs (pos1 - pos2)) ((c.degrees : ℝ) - abs (pos1 - pos2))

/-- Theorem: The smaller angle between the hour and minute hands at 3:30 is 75 degrees -/
theorem clock_angle_at_3_30 (c : Clock) : 
  smallerAngle c (hourHandPosition c 3 30) (minuteHandPosition c 30) = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_30_l155_15504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_of_P_l155_15524

/-- Define the complex number i as the square root of -1 -/
def i : ℂ := Complex.I

/-- Define the function P(x) -/
noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (Complex.I * 2 * x) + 
  Complex.exp (Complex.I * 3 * x) - Complex.exp (Complex.I * 4 * x)

/-- Theorem stating that there are exactly three roots of P(x) in [0, 2π) -/
theorem three_roots_of_P :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x < 2 * Real.pi ∧ P x = 0) ∧
    (∀ x, 0 ≤ x → x < 2 * Real.pi → P x = 0 → x ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_of_P_l155_15524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_area_theorem_l155_15535

-- Define a function g
def g : ℝ → ℝ := sorry

-- Define the area between a function and the x-axis
def area_between_curve_and_axis (f : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_area_theorem :
  area_between_curve_and_axis g = 8 →
  area_between_curve_and_axis (fun x ↦ 4 * g (x + 3)) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_area_theorem_l155_15535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l155_15543

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem angle_between_vectors (a b c : E) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hnorm : ‖a‖ = ‖b‖ ∧ ‖b‖ = ‖c‖) (hsum : a + b = c) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = (2 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l155_15543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_is_ellipse_l155_15544

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 36

-- Define the center of the circle
def M : ℝ × ℝ := (-2, 0)

-- Define point N
def N : ℝ × ℝ := (2, 0)

-- Define point A on the circle
noncomputable def A : ℝ × ℝ := sorry

-- Assume A is on the circle
axiom A_on_circle : circle_equation A.1 A.2

-- Define point P
noncomputable def P : ℝ × ℝ := sorry

-- P is on the perpendicular bisector of AN
axiom P_on_perp_bisector : 
  let midpoint := ((A.1 + N.1) / 2, (A.2 + N.2) / 2)
  (P.1 - midpoint.1) * (A.1 - N.1) + (P.2 - midpoint.2) * (A.2 - N.2) = 0

-- P is on MA
axiom P_on_MA : 
  ∃ t : ℝ, P = (t * A.1 + (1 - t) * M.1, t * A.2 + (1 - t) * M.2)

-- Define an ellipse (simplified definition)
def is_ellipse (P : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ) (h k : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1

-- Theorem statement
theorem locus_of_P_is_ellipse : is_ellipse P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_is_ellipse_l155_15544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_sum_l155_15548

theorem max_value_and_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + 9 * c^2 = 1) :
  (∃ (max : ℝ), max = Real.sqrt 21 / 3 ∧
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + 9 * z^2 = 1 →
      Real.sqrt x + Real.sqrt y + Real.sqrt 3 * z ≤ max)) ∧
  (a + b + c = (18 + Real.sqrt 7) / 21 ↔
    Real.sqrt a + Real.sqrt b + Real.sqrt 3 * c = Real.sqrt 21 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_and_sum_l155_15548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l155_15581

theorem shaded_area_proof (large_square_area small_square_area : ℝ) 
  (h1 : large_square_area = 16)
  (h2 : small_square_area = 1) :
  large_square_area - (4 * small_square_area) - 4 * ((large_square_area ^ (1/2)) / 2)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_proof_l155_15581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l155_15579

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x - 2

-- Part 1
theorem part_one (k : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 k ∧ line_l B.1 B.2 k ∧
    (A.1 * B.1 + A.2 * B.2 = 0)) →
  k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

-- Part 2
theorem part_two :
  ∃ F : ℝ × ℝ, ∀ P : ℝ × ℝ,
    line_l P.1 P.2 (1/2) →
    (∃ C D : ℝ × ℝ,
      circle_O C.1 C.2 ∧ circle_O D.1 D.2 ∧
      ((C.1 - P.1) * (C.1 - 0) + (C.2 - P.2) * (C.2 - 0) = 0) ∧
      ((D.1 - P.1) * (D.1 - 0) + (D.2 - P.2) * (D.2 - 0) = 0) ∧
      (F.1 = 1/2 ∧ F.2 = -1) ∧
      ((F.2 - C.2) * (D.1 - C.1) = (F.1 - C.1) * (D.2 - C.2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l155_15579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_a_to_neg_half_power_l155_15558

theorem coefficient_of_a_to_neg_half_power (a : ℝ) (h : a > 0) : 
  (Finset.range 8).sum (fun k => (-1)^k * (Nat.choose 7 k) * a^(7 - 3*k/2 : ℝ)) = -21 * a^(-1/2 : ℝ) + 
  (Finset.range 8).sum (fun k => if k ≠ 5 then (-1)^k * (Nat.choose 7 k) * a^(7 - 3*k/2 : ℝ) else 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_a_to_neg_half_power_l155_15558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_of_positive_six_l155_15510

theorem negative_of_positive_six : -(6) = -6 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_of_positive_six_l155_15510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_max_height_and_threat_l155_15563

/-- Represents the motion of a rocket launched vertically upwards -/
noncomputable def rocket_motion (a : ℝ) (τ : ℝ) (g : ℝ) : ℝ → ℝ := fun t =>
  if t ≤ τ then
    (1/2) * a * t^2
  else
    (1/2) * a * τ^2 + a * τ * (t - τ) - (1/2) * g * (t - τ)^2

/-- The time at which the rocket reaches its maximum height -/
noncomputable def max_height_time (a : ℝ) (τ : ℝ) (g : ℝ) : ℝ := τ + a * τ / g

/-- Theorem stating the maximum height reached by the rocket and whether it poses a threat -/
theorem rocket_max_height_and_threat (a g : ℝ) (h_a : a = 20) (h_g : g = 10) :
  let τ : ℝ := 50
  let max_height := rocket_motion a τ g (max_height_time a τ g)
  (max_height = 75000) ∧ (max_height > 70000) := by
  sorry

#check rocket_max_height_and_threat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocket_max_height_and_threat_l155_15563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l155_15560

theorem calculation_proof : 
  -Real.sqrt 9 - 4 * (-2) + 2 * Real.cos (π / 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l155_15560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_duration_min_n_value_l155_15591

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 4 then 10 / (4 + x)
  else if 4 ≤ x ∧ x ≤ 6 then 4 - x / 2
  else 0

-- Define the concentration function y(m, x)
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := m * f x

-- Theorem 1: Duration of effective treatment
theorem effective_duration (h1 : 1 ≤ 2 ∧ 2 ≤ 4) :
  ∀ x, 0 ≤ x ∧ x ≤ 6 → y 2 x ≥ 2 := by
  sorry

-- Theorem 2: Minimum value of n
theorem min_n_value (h2 : ∀ x, 4 ≤ x ∧ x ≤ 6 → 8 - x + 10 * 0 / x ≥ 2) :
  ∀ n, n ≥ 0 → ∀ x, 4 ≤ x ∧ x ≤ 6 → 8 - x + 10 * n / x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_duration_min_n_value_l155_15591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_3_and_5_plus_5i_l155_15599

/-- The angle between two vectors in the complex plane -/
noncomputable def angle_between_vectors (z₁ z₂ : ℂ) : ℝ :=
  Real.arccos (((z₁.re * z₂.re + z₁.im * z₂.im) : ℝ) / (Complex.abs z₁ * Complex.abs z₂))

/-- Theorem: The angle between vectors corresponding to complex numbers 3 and 5+5i is π/4 -/
theorem angle_between_3_and_5_plus_5i :
  angle_between_vectors 3 (5 + 5*I) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_3_and_5_plus_5i_l155_15599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_area_is_60_l155_15501

/-- The area of a square with side length equal to the radius of a circle -/
def square_area : ℝ := 1296

/-- The breadth of the rectangle -/
def rectangle_breadth : ℝ := 10

/-- The angle between adjacent sides of the parallelogram -/
def parallelogram_angle : ℝ := 120

/-- The side of the parallelogram not connected to the rectangle -/
def parallelogram_side : ℝ := 16

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt square_area

/-- The length of the rectangle -/
noncomputable def rectangle_length : ℝ := circle_radius / 6

/-- The area of the rectangular region -/
noncomputable def rectangular_area : ℝ := rectangle_length * rectangle_breadth

theorem rectangular_area_is_60 : rectangular_area = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_area_is_60_l155_15501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_exp_squared_over_exp_plus_one_squared_l155_15590

theorem integral_of_exp_squared_over_exp_plus_one_squared (x : ℝ) :
  deriv (λ x ↦ Real.log (Real.exp x + 1) + 1 / (Real.exp x + 1)) x = 
    Real.exp (2*x) / (Real.exp x + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_exp_squared_over_exp_plus_one_squared_l155_15590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_geq_two_is_zero_l155_15509

noncomputable def numbers : List ℝ := [0.8, 1/2, 0.9]

theorem count_geq_two_is_zero : 
  (numbers.filter (λ x => x ≥ 2)).length = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_geq_two_is_zero_l155_15509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l155_15515

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 → y = (4/3)*x ∨ y = -(4/3)*x) ∧
  (∀ x y : ℝ, y^2/25 - x^2/M = 1 → y = (5/Real.sqrt M)*x ∨ y = -(5/Real.sqrt M)*x) →
  M = 225/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolas_same_asymptotes_l155_15515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robi_contribution_is_4000_l155_15505

noncomputable section

-- Define Robi's contribution
def robi_contribution : ℝ := 4000

-- Define Rudy's contribution
def rudy_contribution : ℝ := robi_contribution + (1/4) * robi_contribution

-- Define total contribution
def total_contribution : ℝ := robi_contribution + rudy_contribution

-- Define profit percentage
def profit_percentage : ℝ := 20/100

-- Define total profit
def total_profit : ℝ := profit_percentage * total_contribution

-- Define individual profit
def individual_profit : ℝ := 900

theorem robi_contribution_is_4000 :
  robi_contribution = 4000 ∧
  rudy_contribution = robi_contribution + (1/4) * robi_contribution ∧
  total_contribution = robi_contribution + rudy_contribution ∧
  profit_percentage = 20/100 ∧
  total_profit = profit_percentage * total_contribution ∧
  individual_profit = 900 ∧
  total_profit = 2 * individual_profit :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_robi_contribution_is_4000_l155_15505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_300_l155_15500

/-- Represents the properties of a boat journey -/
structure BoatJourney where
  still_speed : ℝ  -- Speed of the boat in still water
  upstream_time : ℝ  -- Time taken to travel upstream
  downstream_time : ℝ  -- Time taken to travel downstream

/-- Calculates the distance traveled upstream given a boat journey -/
noncomputable def distance_upstream (j : BoatJourney) : ℝ :=
  let current_speed := (j.still_speed * (j.upstream_time - j.downstream_time)) / (j.upstream_time + j.downstream_time)
  (j.still_speed - current_speed) * j.upstream_time

/-- Theorem stating that for the given conditions, the upstream distance is 300 miles -/
theorem upstream_distance_is_300 (j : BoatJourney) 
    (h1 : j.still_speed = 105)
    (h2 : j.upstream_time = 5)
    (h3 : j.downstream_time = 2) : 
  distance_upstream j = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_300_l155_15500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_three_four_five_l155_15586

def M : Set ℕ := {x | x < 6}
def N : Set ℝ := {x | x^2 - 11*x + 18 < 0}

theorem intersection_equals_three_four_five :
  (M.image (↑) ∩ N) = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_three_four_five_l155_15586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_covering_exists_l155_15545

/-- A square on a plane with sides parallel to the coordinate axes -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- The set of 1000 squares -/
def squares : Finset Square :=
  sorry

/-- The set of centers of the squares -/
def M : Set (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a point is inside a square -/
def is_inside (p : ℝ × ℝ) (s : Square) : Prop :=
  sorry

/-- Helper function to make is_inside decidable -/
def is_inside_decidable (p : ℝ × ℝ) (s : Square) : Decidable (is_inside p s) :=
  sorry

instance (p : ℝ × ℝ) : DecidablePred (is_inside p) :=
  is_inside_decidable p

theorem square_covering_exists :
  ∃ (subset : Finset Square),
    subset ⊆ squares ∧
    (∀ m ∈ M, ∃ s ∈ subset, is_inside m s) ∧
    (∀ m ∈ M, (Finset.filter (is_inside m) subset).card ≤ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_covering_exists_l155_15545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_angle_l155_15520

theorem min_value_angle (A : ℝ) : 
  (∀ θ : ℝ, Real.cos (A / 2) + Real.sqrt 3 * Real.sin (A / 2) ≤ Real.cos (θ / 2) + Real.sqrt 3 * Real.sin (θ / 2)) → 
  A = 5 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_angle_l155_15520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l155_15592

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ
  c : ℝ

def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.c

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def triangle_area (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem ellipse_problem (b : ℝ) (h_b : b > 0) :
  let M : Ellipse := ⟨3, b, ⟨by norm_num, h_b⟩⟩
  let focus : Point := ⟨2, 0⟩
  let N : Ellipse := ⟨1, Real.sqrt 6, ⟨by norm_num, by norm_num⟩⟩
  let P : Point := ⟨Real.sqrt 2 / 2, Real.sqrt 3⟩
  let l : Line := ⟨1, -2⟩
  ∃ (A B : Point),
    (M.contains focus) ∧
    (N.contains P) ∧
    (l.contains A) ∧ (l.contains B) ∧ (N.contains A) ∧ (N.contains B) ∧
    (distance A B = 12/7) ∧
    (triangle_area A B ⟨0, 0⟩ = 6 * Real.sqrt 2 / 7) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l155_15592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l155_15585

/-- Represents the speed of a boat in a river with current -/
structure RiverBoat where
  boatSpeed : ℝ  -- Speed of the boat in still water
  currentSpeed : ℝ  -- Speed of the current

/-- Calculates the time taken for a journey given distance and effective speed -/
noncomputable def travelTime (distance : ℝ) (effectiveSpeed : ℝ) : ℝ :=
  distance / effectiveSpeed

theorem river_current_speed 
  (rb : RiverBoat) 
  (downstreamTime : ℝ) 
  (upstreamTime : ℝ) 
  (distance : ℝ) 
  (h1 : downstreamTime = travelTime distance (rb.boatSpeed + rb.currentSpeed))
  (h2 : upstreamTime = travelTime distance (rb.boatSpeed - rb.currentSpeed))
  (h3 : downstreamTime = 4)
  (h4 : upstreamTime = 6)
  (h5 : distance = 24)
  : rb.currentSpeed = 1 := by
  sorry

#check river_current_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_current_speed_l155_15585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_four_minutes_l155_15594

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  area : ℝ
  ratio_length_breadth : length / breadth = 1 / 3
  area_eq : area = length * breadth

/-- Represents a cyclist -/
structure Cyclist where
  speed : ℝ
  speed_eq : speed = 12 -- in km/hr

noncomputable def time_for_one_round (park : RectangularPark) (cyclist : Cyclist) : ℝ :=
  2 * (park.length + park.breadth) / (cyclist.speed * 1000 / 3600)

theorem time_is_four_minutes 
  (park : RectangularPark)
  (cyclist : Cyclist)
  (h_area : park.area = 30000) :
  time_for_one_round park cyclist = 4 / 60 := by
  sorry

#check time_is_four_minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_is_four_minutes_l155_15594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_january_first_is_friday_l155_15506

/-- Represents the days of the week -/
inductive Weekday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday
deriving Repr, DecidableEq

/-- Returns the next day of the week -/
def nextDay (d : Weekday) : Weekday :=
  match d with
  | Weekday.monday => Weekday.tuesday
  | Weekday.tuesday => Weekday.wednesday
  | Weekday.wednesday => Weekday.thursday
  | Weekday.thursday => Weekday.friday
  | Weekday.friday => Weekday.saturday
  | Weekday.saturday => Weekday.sunday
  | Weekday.sunday => Weekday.monday

/-- Counts the number of occurrences of a specific day in a month -/
def countDayInMonth (startDay : Weekday) (numDays : Nat) (targetDay : Weekday) : Nat :=
  let rec count (currentDay : Weekday) (daysLeft : Nat) (acc : Nat) : Nat :=
    if daysLeft = 0 then acc
    else count (nextDay currentDay) (daysLeft - 1) (if currentDay = targetDay then acc + 1 else acc)
  count startDay numDays 0

theorem january_first_is_friday 
  (h1 : countDayInMonth Weekday.friday 31 Weekday.monday = 4)
  (h2 : countDayInMonth Weekday.friday 31 Weekday.thursday = 4) :
  Weekday.friday = Weekday.friday := by
  sorry

#check january_first_is_friday

end NUMINAMATH_CALUDE_ERRORFEEDBACK_january_first_is_friday_l155_15506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_123_l155_15573

-- Define a function to convert a natural number to its binary representation
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n |>.reverse

-- Define a function to convert a list of booleans to a natural number
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

-- Theorem statement
theorem decimal_to_binary_123 :
  toBinary 123 = [true, true, true, true, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_binary_123_l155_15573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_b_greater_a_is_one_fifth_l155_15534

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {1, 2, 3}

def prob_b_greater_a : ℚ :=
  (Finset.filter (λ (p : ℕ × ℕ) => p.2 > p.1) (A.product B)).card /
  ((A.card * B.card : ℕ) : ℚ)

theorem prob_b_greater_a_is_one_fifth :
  prob_b_greater_a = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_b_greater_a_is_one_fifth_l155_15534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l155_15547

theorem problem_statement : 
  (∀ x : ℝ, x ≥ 0 → (2 : ℝ)^x ≥ 1) ∧ 
  ¬(∀ x y : ℝ, x > y → x^2 > y^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l155_15547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l155_15582

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l155_15582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l155_15522

-- Define ConvexPolygon as a structure
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here

-- Define InteriorAngle as a function
def InteriorAngle (p : ConvexPolygon n) (i : Fin n) : ℝ :=
  -- Define the interior angle calculation here
  sorry

theorem polygon_sides (n : ℕ) (h_convex : ConvexPolygon n) : 
  (∀ i : Fin n, InteriorAngle h_convex i = 140 * π / 180) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_l155_15522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_negative_two_equals_twenty_l155_15546

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 9 * ((x - 1) / 3)^2 - 6 * ((x - 1) / 3) + 5

-- Theorem statement
theorem f_of_negative_two_equals_twenty : f (-2) = 20 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Perform numerical calculations
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_negative_two_equals_twenty_l155_15546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l155_15593

theorem percentage_difference : (0.6 * 50) - (0.45 * 30) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_difference_l155_15593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_two_implies_e_l155_15575

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem
theorem derivative_equals_two_implies_e (x₀ : ℝ) (h : x₀ > 0) :
  deriv f x₀ = 2 → x₀ = Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_equals_two_implies_e_l155_15575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_sum_is_twelve_l155_15576

/-- Represents a card with a color and a number. -/
inductive Card
  | red (n : Nat)
  | blue (n : Nat)
deriving Inhabited

/-- Checks if two numbers satisfy the divisibility condition. -/
def divisible (a b : Nat) : Bool :=
  a % b = 0 || b % a = 0

/-- Checks if a sequence of cards is valid according to the rules. -/
def validSequence (cards : List Card) : Prop :=
  cards.length = 13 ∧
  (∀ i, i < cards.length - 1 → match cards[i]?, cards[i+1]? with
    | some (Card.red n), some (Card.blue m) | some (Card.blue m), some (Card.red n) => divisible n m
    | _, _ => false)

/-- The set of red cards. -/
def redCards : List Card := [1, 2, 3, 4, 5, 6, 7].map Card.red

/-- The set of blue cards. -/
def blueCards : List Card := [4, 5, 6, 7, 8, 9].map Card.blue

/-- Gets the number from a card. -/
def cardNumber (c : Card) : Nat :=
  match c with
  | Card.red n => n
  | Card.blue n => n

theorem middle_three_sum_is_twelve :
  ∀ (seq : List Card),
    seq.length = 13 ∧
    validSequence seq ∧
    (∀ c, c ∈ seq → c ∈ redCards ∨ c ∈ blueCards) →
    (cardNumber (seq.get! 5) + cardNumber (seq.get! 6) + cardNumber (seq.get! 7)) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_three_sum_is_twelve_l155_15576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_iff_z_is_imaginary_iff_z_is_pure_imaginary_iff_l155_15574

-- Define the complex number z as a function of m
noncomputable def z (m : ℝ) : ℂ := (m^2 + m - 6) / m + (m^2 - 2*m) * Complex.I

-- Theorem for the real number case
theorem z_is_real_iff (m : ℝ) : m ≠ 0 → (z m).im = 0 ↔ m = 2 := by sorry

-- Theorem for the imaginary number case
theorem z_is_imaginary_iff (m : ℝ) : m ≠ 0 → (z m).im ≠ 0 ↔ m ≠ 2 := by sorry

-- Theorem for the pure imaginary number case
theorem z_is_pure_imaginary_iff (m : ℝ) : m ≠ 0 → ((z m).re = 0 ∧ (z m).im ≠ 0) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_iff_z_is_imaginary_iff_z_is_pure_imaginary_iff_l155_15574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_prime_value_l155_15595

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def f (x : ℤ) : ℕ := Int.natAbs (8 * x^2 - 62 * x + 21)

theorem largest_integer_prime_value :
  ∀ x : ℤ, (x > 4 ∨ x ≤ 4) →
  (x > 4 → ¬(is_prime (f x))) ∧
  (is_prime (f 4)) ∧
  (∀ y : ℤ, y < 4 → y > 0 → ¬(is_prime (f y))) :=
by
  sorry

#check largest_integer_prime_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_prime_value_l155_15595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_less_than_2_necessary_not_sufficient_l155_15556

theorem x_less_than_2_necessary_not_sufficient :
  (∀ x : ℝ, (2 : ℝ)^x < 1 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ (2 : ℝ)^x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_less_than_2_necessary_not_sufficient_l155_15556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_equidistant_from_B_and_C_l155_15514

noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem point_A_equidistant_from_B_and_C :
  let A : ℝ × ℝ × ℝ := (0, 0, 7.5)
  let B : ℝ × ℝ × ℝ := (-13, 4, 6)
  let C : ℝ × ℝ × ℝ := (10, -9, 5)
  distance A.1 A.2.1 A.2.2 B.1 B.2.1 B.2.2 = distance A.1 A.2.1 A.2.2 C.1 C.2.1 C.2.2 :=
by
  sorry

#check point_A_equidistant_from_B_and_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_equidistant_from_B_and_C_l155_15514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_inequality_l155_15564

theorem largest_constant_inequality (l : ℝ) (h : l > 0) :
  (∃ C : ℝ, ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x^2 + y^2 + l*x*y ≥ C*(x+y)^2) ∧
  (∀ C : ℝ, (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x^2 + y^2 + l*x*y ≥ C*(x+y)^2) →
    C ≤ (if l ≥ 2 then 1 else (2 + l)/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_inequality_l155_15564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_integer_part_l155_15588

theorem average_integer_part (N : ℤ) (h1 : 7 < N) (h2 : N < 15) :
  let avg : ℚ := (N + 8 + 12) / 3
  (⌊avg⌋ : ℤ) ∈ ({9, 10, 11} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_integer_part_l155_15588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_100_terms_l155_15512

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle Nat.zero
  | 1 => 2
  | (n + 1) => 1 - 1 / sequence_a n

def S (n : ℕ) : ℚ := (List.range n).map sequence_a |>.sum

theorem sum_of_100_terms : S 100 = 103 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_100_terms_l155_15512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_floor_l155_15569

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

-- State the theorem
theorem zero_point_floor :
  ∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = 0 ∧ floor x₀ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_floor_l155_15569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_order_golden_ratio_l155_15508

/-- Defines the n-order golden section ratio for a line segment AB with point C on it -/
noncomputable def golden_section_ratio (n : ℕ) (AB AC BC : ℝ) : ℝ :=
  BC / (Real.sqrt n * AC)

/-- Theorem: The 4-order golden section ratio is (-1 + √17) / 4 -/
theorem fourth_order_golden_ratio :
  ∀ (AB AC BC : ℝ),
    AB > 0 → AC > 0 → BC > 0 →
    AB = AC + BC →
    golden_section_ratio 4 AB AC BC = BC / (2 * AC) →
    golden_section_ratio 4 AB AC BC = 2 * AC / AB →
    golden_section_ratio 4 AB AC BC = (-1 + Real.sqrt 17) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_order_golden_ratio_l155_15508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_B_l155_15583

def sequenceList : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

def nthLetter (n : Nat) : Char :=
  sequenceList[n % sequenceList.length]'sorry

theorem letter_2023_is_B : nthLetter 2022 = 'B' := by
  -- Proof steps would go here
  sorry

#eval nthLetter 2022

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_2023_is_B_l155_15583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_cell_phone_bill_l155_15559

/-- Calculates the total cost of a cell phone plan based on given parameters -/
def calculate_total_cost (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
  (international_cost : ℚ) (texts_sent : ℕ) (minutes_talked : ℕ) (international_minutes : ℕ) : ℚ :=
  let extra_minutes := max (minutes_talked - 40 * 60) 0
  base_cost + 
  (text_cost * (texts_sent : ℚ) / 100) + 
  (extra_minute_cost * (extra_minutes : ℚ) / 100) + 
  (international_cost * (international_minutes : ℚ))

/-- Theorem stating that John's cell phone bill is $69.00 -/
theorem johns_cell_phone_bill : 
  calculate_total_cost 25 8 15 1 200 (42 * 60) 10 = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_cell_phone_bill_l155_15559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l155_15516

/-- The angle between two 2D vectors -/
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

/-- Theorem: The angle between vectors (3,0) and (-5,5) is 3π/4 -/
theorem angle_between_specific_vectors :
  angle_between (3, 0) (-5, 5) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l155_15516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l155_15557

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  center : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the centroid of a triangle -/
noncomputable def centroid (p1 p2 p3 : Point) : Point :=
  { x := (p1.x + p2.x + p3.x) / 3,
    y := (p1.y + p2.y + p3.y) / 3 }

/-- The main theorem -/
theorem centroid_quadrilateral_area 
  (s : Square) 
  (q : Point) 
  (h1 : s.sideLength = 40) 
  (h2 : distance { x := s.center.x - s.sideLength/2, y := s.center.y + s.sideLength/2 } q = 16)
  (h3 : distance { x := s.center.x - s.sideLength/2, y := s.center.y - s.sideLength/2 } q = 34)
  : 
  let e := { x := s.center.x - s.sideLength/2, y := s.center.y + s.sideLength/2 }
  let f := { x := s.center.x - s.sideLength/2, y := s.center.y - s.sideLength/2 }
  let g := { x := s.center.x + s.sideLength/2, y := s.center.y - s.sideLength/2 }
  let h := { x := s.center.x + s.sideLength/2, y := s.center.y + s.sideLength/2 }
  let c1 := centroid e f q
  let c2 := centroid f g q
  let c3 := centroid g h q
  let c4 := centroid h e q
  let area := (distance c1 c3 * distance c2 c4) / 2
  area = 800/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_quadrilateral_area_l155_15557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_is_six_l155_15523

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := fun n ↦ a₁ + (n - 1 : ℝ) * d

def geometric_sequence (b₁ : ℝ) (r : ℝ) : ℕ → ℝ := fun n ↦ b₁ * r^(n - 1 : ℕ)

def is_valid_n (n : ℕ) (a₂₆ : ℝ) (b : ℕ → ℝ) : Prop := b n * a₂₆ < 1

theorem smallest_valid_n_is_six 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h₁ : a = arithmetic_sequence 1 (1/2)) 
  (h₂ : b = geometric_sequence 6 (1/3)) 
  (h₃ : b 2 = a 3) :
  (∀ k < 6, ¬(is_valid_n k (a 26) b)) ∧ (is_valid_n 6 (a 26) b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_is_six_l155_15523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_opinion_change_is_40_percent_l155_15519

/-- Represents the distribution of opinions towards math in a semester -/
structure OpinionDistribution :=
  (loved : ℚ)
  (neutral : ℚ)
  (notLoved : ℚ)
  (sum_to_one : loved + neutral + notLoved = 1)

/-- Calculates the maximum percentage of students who changed their opinion -/
def maxOpinionChange (first third : OpinionDistribution) : ℚ :=
  let stableLoving := min first.loved third.loved
  let stableNeutral := min first.neutral third.neutral
  let stableNotLoving := min first.notLoved third.notLoved
  1 - (stableLoving + stableNeutral + stableNotLoving)

/-- The main theorem stating that the maximum opinion change is 40% -/
theorem max_opinion_change_is_40_percent 
  (first : OpinionDistribution)
  (third : OpinionDistribution)
  (h1 : first.loved = 3/10)
  (h2 : first.neutral = 4/10)
  (h3 : first.notLoved = 3/10)
  (h4 : third.loved = 1/2)
  (h5 : third.neutral = 1/5)
  (h6 : third.notLoved = 3/10) :
  maxOpinionChange first third = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_opinion_change_is_40_percent_l155_15519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_cosine_l155_15525

theorem triangle_special_cosine (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  -- Sides form arithmetic sequence
  a + c = 2 * b ∧
  -- Angle A is three times Angle C
  A = 3 * C →
  -- Conclusion: cos C equals the given expression
  Real.cos C = (1 + Real.sqrt 33) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_cosine_l155_15525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_to_AB_ratio_l155_15587

/-- Right triangle ABC with hypotenuse AB, where AC = 15, BC = 20, and CD is the altitude to AB.
    ω is the circle with CD as diameter.
    I is a point outside triangle ABC such that AI and BI are tangent to circle ω. -/
structure TriangleConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  I : ℝ × ℝ
  ω : Set (ℝ × ℝ)
  is_right_triangle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 15
  BC_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 20
  CD_is_altitude : (A.1 - B.1) * (D.1 - C.1) + (A.2 - B.2) * (D.2 - C.2) = 0
  ω_is_circle_with_CD_diameter : ∀ p ∈ ω, (p.1 - D.1)^2 + (p.2 - D.2)^2 = ((C.1 - D.1)^2 + (C.2 - D.2)^2) / 4
  I_outside_triangle : (I.1 - A.1) * (B.2 - A.2) - (I.2 - A.2) * (B.1 - A.1) ≠ 0
  AI_tangent_to_ω : ∃ p ∈ ω, (I.1 - A.1) * (p.1 - A.1) + (I.2 - A.2) * (p.2 - A.2) = 0
  BI_tangent_to_ω : ∃ p ∈ ω, (I.1 - B.1) * (p.1 - B.1) + (I.2 - B.2) * (p.2 - B.2) = 0

/-- The ratio of the perimeter of triangle ABI to AB is 177/100. -/
theorem perimeter_to_AB_ratio (config : TriangleConfig) :
  let AB := Real.sqrt ((config.A.1 - config.B.1)^2 + (config.A.2 - config.B.2)^2)
  let AI := Real.sqrt ((config.A.1 - config.I.1)^2 + (config.A.2 - config.I.2)^2)
  let BI := Real.sqrt ((config.B.1 - config.I.1)^2 + (config.B.2 - config.I.2)^2)
  (AI + BI + AB) / AB = 177 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_to_AB_ratio_l155_15587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l155_15580

-- Define the complex number z
noncomputable def z : ℂ := (2 * Complex.I) / (1 - Complex.I)

-- Theorem statement
theorem z_in_second_quadrant : 
  Real.sign z.re = -1 ∧ Real.sign z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l155_15580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_half_trajectory_equation_l155_15568

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := (a - 1) * x + a * y + 1/2 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop := 
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem 1: If l₁ ∥ l₂, then a = 1/2
theorem parallel_lines_imply_a_half : 
  parallel l₁ (l₂ (1/2)) := by
  sorry

-- Theorem 2: The trajectory equation
theorem trajectory_equation (x y : ℝ) :
  (∃ (A B : ℝ), l₁ A 0 ∧ l₂ (1/2) B 0 ∧ 
    distance x y A 0 = Real.sqrt 2 * distance x y B 0) →
  (x - 3)^2 + y^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_half_trajectory_equation_l155_15568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l155_15571

/-- Represents the time it takes for a train to cross an electric pole. -/
noncomputable def train_crossing_time (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

/-- Theorem: A train with length 2500 meters and speed 180 km/h takes 50 seconds to cross an electric pole. -/
theorem train_crossing_theorem :
  train_crossing_time 2500 180 = 50 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l155_15571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_nonnegative_integer_solution_l155_15513

theorem one_nonnegative_integer_solution :
  ∃! (x : ℕ), x^2 = 16 - 4*x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_nonnegative_integer_solution_l155_15513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_two_l155_15553

/-- Given n = x - y^(x+y), x = 3, and y = -1, prove that n = 2 -/
theorem n_equals_two (x y n : ℤ) : x = 3 → y = -1 → n = x - (y : ℚ)^(x+y) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_equals_two_l155_15553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l155_15529

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  majorAxis : Point × Point
  minorAxis : Point × Point

/-- Calculates the center of an ellipse -/
noncomputable def centerOfEllipse (e : Ellipse) : Point :=
  { x := (e.majorAxis.1.x + e.majorAxis.2.x) / 2,
    y := (e.majorAxis.1.y + e.majorAxis.2.y) / 2 }

/-- Calculates the semi-major axis length -/
noncomputable def semiMajorAxis (e : Ellipse) : ℝ :=
  (e.majorAxis.2.x - e.majorAxis.1.x) / 2

/-- Calculates the semi-minor axis length -/
noncomputable def semiMinorAxis (e : Ellipse) : ℝ :=
  (e.minorAxis.1.y - e.minorAxis.2.y) / 2

/-- Calculates the focal distance from the center -/
noncomputable def focalDistance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

theorem ellipse_focus_coordinates (e : Ellipse) 
  (h1 : e.majorAxis = ({ x := 0, y := -1 }, { x := 10, y := -1 }))
  (h2 : e.minorAxis = ({ x := 5, y := 2 }, { x := 5, y := -4 })) :
  let center := centerOfEllipse e
  let a := semiMajorAxis e
  let b := semiMinorAxis e
  let c := focalDistance a b
  (Point.mk (center.x + c) center.y) = (Point.mk 9 (-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_coordinates_l155_15529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l155_15542

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.sqrt 2 * Real.cos x

theorem problem_solution (m : ℝ) (A B : ℝ) (a b : ℝ) 
  (h1 : m > 0)
  (h2 : ∀ x, f m x ≤ 2)
  (h3 : ∃ x, f m x = 2)
  (h4 : Real.sqrt 3 = 2 * Real.sin A * Real.sin B / Real.sin (A + B))
  (h5 : f m (A - π/4) + f m (B - π/4) = 4 * Real.sqrt 6 * Real.sin A * Real.sin B)
  (h6 : a = 2 * Real.sqrt 3 * Real.sin B / Real.sin (A + B))
  (h7 : b = 2 * Real.sqrt 3 * Real.sin A / Real.sin (A + B)) :
  (∀ x ∈ Set.union (Set.Icc (-2) (-1)) (Set.Ioo 2 6), -Real.sqrt 2 ≤ f m x ∧ f m x ≤ Real.sqrt 2) ∧
  1/a + 1/b = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l155_15542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l155_15526

theorem rectangle_area (p : ℝ) (p_pos : p > 0) :
  let l := p / 6
  let w := 2 * l
  let area := l * w
  area = p^2 / 18 := by
  -- Unfold the let bindings
  simp
  -- Algebraic manipulation
  field_simp
  -- Prove equality
  ring

#check rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l155_15526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoemakers_knife_l155_15531

-- Define the points and lengths
variable (A B C D E F O O₁ O₂ : ℝ × ℝ)
variable (a b : ℝ)

-- Define the conditions
axiom C_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

axiom semicircle_O : ∃ center : ℝ × ℝ, ‖center - A‖ = ‖center - B‖ ∧ O = center

axiom semicircle_O₁ : ∃ center : ℝ × ℝ, ‖center - A‖ = ‖center - C‖ ∧ O₁ = center

axiom semicircle_O₂ : ∃ center : ℝ × ℝ, ‖center - C‖ = ‖center - B‖ ∧ O₂ = center

axiom D_on_perpendicular : (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0

axiom D_on_semicircle_O : ‖D - O‖^2 = ‖A - O‖^2

axiom EF_common_tangent : ∃ E F : ℝ × ℝ, 
  ‖E - O₁‖ = a ∧ ‖F - O₂‖ = b ∧
  (E.1 - F.1) * (O₁.1 - O₂.1) + (E.2 - F.2) * (O₁.2 - O₂.2) = 0

-- State the theorem
theorem shoemakers_knife : ‖D - C‖ = ‖E - F‖ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoemakers_knife_l155_15531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subset_l155_15537

def P : Set ℝ := {1, 2, 3, 4, 5}
def Q : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 5}

theorem intersection_subset : P ∩ Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_subset_l155_15537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l155_15538

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^2

theorem s_range :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, x ≠ 2 ∧ s x = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l155_15538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l155_15517

/-- Triangle with angle bisector -/
structure AngleBisectorTriangle where
  A : ℝ × ℝ  -- Point A of the triangle
  B : ℝ × ℝ  -- Point B of the triangle
  C : ℝ × ℝ  -- Point C of the triangle
  D : ℝ × ℝ  -- Point where angle bisector intersects opposite side
  E : ℝ × ℝ  -- Point where angle bisector intersects circumcircle
  b : ℝ      -- Length of side adjacent to bisected angle
  c : ℝ      -- Length of other side adjacent to bisected angle
  m : ℝ      -- Length of one segment of opposite side
  n : ℝ      -- Length of other segment of opposite side
  ℓ : ℝ      -- Length of angle bisector

/-- The square of the angle bisector length equals the product of adjacent sides 
    minus the product of opposite side segments -/
theorem angle_bisector_theorem (t : AngleBisectorTriangle) : t.ℓ^2 = t.b * t.c - t.m * t.n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l155_15517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l155_15598

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: The eccentricity of a specific hyperbola configuration is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) (c : Circle) 
  (F A B : Point) (O : Point := ⟨0, 0⟩) :
  -- The circle passes through the right focus of the hyperbola
  c.r^2 = F.x^2 + F.y^2 →
  -- The circle intersects the asymptotes at A and B
  (A.y = (h.b / h.a) * A.x ∧ B.y = -(h.b / h.a) * B.x) →
  -- OAFB is a rhombus
  (O.x - A.x)^2 + (O.y - A.y)^2 = 
  (O.x - F.x)^2 + (O.y - F.y)^2 →
  -- The eccentricity of the hyperbola is 2
  eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l155_15598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_coordinate_vertex_x_coordinate_specific_l155_15554

/-- The x-coordinate of the vertex of a quadratic function ax² + bx + c is given by -b/(2a) -/
theorem vertex_x_coordinate (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let x_vertex : ℝ := -b / (2 * a)
  ∀ x, f x ≥ f x_vertex := by sorry

/-- The x-coordinate of the vertex of the quadratic function x² - 8x + 15 is 4 -/
theorem vertex_x_coordinate_specific :
  let f : ℝ → ℝ := λ x => x^2 - 8*x + 15
  let x_vertex : ℝ := 4
  ∀ x, f x ≥ f x_vertex := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_x_coordinate_vertex_x_coordinate_specific_l155_15554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l155_15536

-- Define the solid T
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p;
    x + 2*y ≤ 1 ∧
    x + 2*z ≤ 1 ∧
    2*y + 2*z ≤ 1 ∧
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_T : MeasureTheory.volume T = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l155_15536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l155_15530

/-- The slope of a line given its equation ax + by + c = 0 -/
noncomputable def slopeOfLine (a b : ℝ) : ℝ := -a / b

/-- The slope angle of a line given its slope -/
noncomputable def slopeAngle (m : ℝ) : ℝ := Real.arctan m

/-- Check if a point (x, y) lies on a line ax + by + c = 0 -/
def pointOnLine (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

theorem line_equation_proof (x y : ℝ) : 
  pointOnLine 3 (-1) 9 x y ↔ 
  (pointOnLine 3 (-1) 9 (-2) 3 ∧ 
   slopeAngle (slopeOfLine 3 (-1)) = (1/2 : ℝ) * slopeAngle (slopeOfLine 3 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l155_15530
