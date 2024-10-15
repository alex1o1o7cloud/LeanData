import Mathlib

namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3010_301060

theorem partial_fraction_decomposition (x A B C : ℚ) :
  x ≠ 2 → x ≠ 4 → x ≠ 5 →
  ((x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) ↔
  (A = 5/3 ∧ B = -7/2 ∧ C = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3010_301060


namespace NUMINAMATH_CALUDE_tan_600_degrees_equals_sqrt_3_l3010_301014

theorem tan_600_degrees_equals_sqrt_3 : Real.tan (600 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_600_degrees_equals_sqrt_3_l3010_301014


namespace NUMINAMATH_CALUDE_table_wobbles_l3010_301028

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a table with four legs -/
structure Table where
  leg1 : Point3D
  leg2 : Point3D
  leg3 : Point3D
  leg4 : Point3D

/-- Checks if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ 
    a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
    a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
    a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
    a * p4.x + b * p4.y + c * p4.z + d = 0

/-- Defines a square table with given leg lengths -/
def squareTable : Table :=
  { leg1 := ⟨0, 0, 70⟩
  , leg2 := ⟨1, 0, 71⟩
  , leg3 := ⟨1, 1, 72.5⟩
  , leg4 := ⟨0, 1, 72⟩ }

/-- Theorem: The square table with given leg lengths wobbles -/
theorem table_wobbles : ¬areCoplanar squareTable.leg1 squareTable.leg2 squareTable.leg3 squareTable.leg4 := by
  sorry

end NUMINAMATH_CALUDE_table_wobbles_l3010_301028


namespace NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l3010_301043

theorem tan_eleven_pi_sixths : Real.tan (11 * Real.pi / 6) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_sixths_l3010_301043


namespace NUMINAMATH_CALUDE_min_value_theorem_l3010_301059

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f (x : ℝ) := x^2 - 2*x + 2
  let g (x : ℝ) := -x^2 + a*x + b
  let f' (x : ℝ) := 2*x - 2
  let g' (x : ℝ) := -2*x + a
  ∃ x₀ : ℝ, f x₀ = g x₀ ∧ f' x₀ * g' x₀ = -1 →
  (1/a + 4/b) ≥ 18/5 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = 18/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3010_301059


namespace NUMINAMATH_CALUDE_symmetric_function_product_l3010_301012

/-- A function f(x) that is symmetric about the line x = 2 -/
def f (a b : ℝ) (x : ℝ) : ℝ := (x^2 - 1) * (-x^2 + a*x - b)

/-- The symmetry condition for f(x) about x = 2 -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x : ℝ, f a b (2 - x) = f a b (2 + x)

/-- Theorem: If f(x) is symmetric about x = 2, then ab = 120 -/
theorem symmetric_function_product (a b : ℝ) :
  is_symmetric a b → a * b = 120 := by sorry

end NUMINAMATH_CALUDE_symmetric_function_product_l3010_301012


namespace NUMINAMATH_CALUDE_expression_simplification_l3010_301074

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x / (x - 1) - 1) / ((x^2 + 2*x + 1) / (x^2 - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3010_301074


namespace NUMINAMATH_CALUDE_factorial_division_l3010_301054

theorem factorial_division : Nat.factorial 6 / Nat.factorial (6 - 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3010_301054


namespace NUMINAMATH_CALUDE_number_of_balls_correct_l3010_301066

/-- The number of balls in a box, which is as much greater than 40 as it is less than 60. -/
def number_of_balls : ℕ := 50

/-- The condition that the number of balls is as much greater than 40 as it is less than 60. -/
def ball_condition (x : ℕ) : Prop := x - 40 = 60 - x

theorem number_of_balls_correct : ball_condition number_of_balls := by
  sorry

end NUMINAMATH_CALUDE_number_of_balls_correct_l3010_301066


namespace NUMINAMATH_CALUDE_magic_square_difference_l3010_301044

/-- Represents a 3x3 magic square with some given values -/
structure MagicSquare where
  x : ℝ
  y : ℝ
  isValid : x - 2 = 2*y + y ∧ x - 2 = -2 + y + 6

/-- Proves that in the given magic square, y - x = -6 -/
theorem magic_square_difference (ms : MagicSquare) : ms.y - ms.x = -6 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_difference_l3010_301044


namespace NUMINAMATH_CALUDE_painters_work_days_l3010_301003

/-- Given that 5 painters take 1.8 work-days to finish a job, prove that 4 painters
    working at the same rate will take 2.25 work-days to finish the same job. -/
theorem painters_work_days (initial_painters : ℕ) (initial_days : ℝ) 
  (new_painters : ℕ) (new_days : ℝ) :
  initial_painters = 5 →
  initial_days = 1.8 →
  new_painters = 4 →
  (initial_painters : ℝ) * initial_days = (new_painters : ℝ) * new_days →
  new_days = 2.25 := by
sorry

end NUMINAMATH_CALUDE_painters_work_days_l3010_301003


namespace NUMINAMATH_CALUDE_second_train_speed_l3010_301031

/-- The speed of the first train in km/h -/
def speed_first : ℝ := 40

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 1

/-- The distance at which the two trains meet, in km -/
def meeting_distance : ℝ := 200

/-- The speed of the second train in km/h -/
def speed_second : ℝ := 50

theorem second_train_speed :
  speed_second = meeting_distance / (meeting_distance / speed_first - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_second_train_speed_l3010_301031


namespace NUMINAMATH_CALUDE_unique_fraction_representation_l3010_301088

theorem unique_fraction_representation (p : ℕ) (hp : p > 2) (hprime : Nat.Prime p) :
  ∃! (x y : ℕ), x ≠ y ∧ (2 : ℚ) / p = 1 / x + 1 / y :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_representation_l3010_301088


namespace NUMINAMATH_CALUDE_x_bound_y_bound_l3010_301016

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Calculates the position of a particle after n minutes -/
def particlePosition (n : ℕ) : Position :=
  sorry

/-- The initial rightward movement is 2 units -/
axiom initial_rightward : (particlePosition 1).x = 2

/-- The y-coordinate doesn't change in the first minute -/
axiom initial_upward : (particlePosition 1).y = 0

/-- The x-coordinate never decreases -/
axiom x_nondecreasing (n : ℕ) : (particlePosition n).x ≤ (particlePosition (n + 1)).x

/-- The y-coordinate never decreases -/
axiom y_nondecreasing (n : ℕ) : (particlePosition n).y ≤ (particlePosition (n + 1)).y

/-- The x-coordinate is bounded by the initial movement plus subsequent rightward movements -/
theorem x_bound (n : ℕ) : 
  (particlePosition n).x ≤ 2 + 2 * (n / 4) * ((n / 4) + 1) :=
  sorry

/-- The y-coordinate is bounded by the sum of upward movements -/
theorem y_bound (n : ℕ) : 
  (particlePosition n).y ≤ (n - 1) * (n / 4) :=
  sorry

end NUMINAMATH_CALUDE_x_bound_y_bound_l3010_301016


namespace NUMINAMATH_CALUDE_peggy_doll_ratio_l3010_301029

/-- Represents the number of dolls in various situations --/
structure DollCount where
  initial : Nat
  fromGrandmother : Nat
  final : Nat

/-- Calculates the ratio of birthday/Christmas dolls to grandmother's dolls --/
def dollRatio (d : DollCount) : Rat :=
  let birthdayChristmas := d.final - d.initial - d.fromGrandmother
  birthdayChristmas / d.fromGrandmother

/-- Theorem stating the ratio of dolls Peggy received --/
theorem peggy_doll_ratio (d : DollCount) 
  (h1 : d.initial = 6)
  (h2 : d.fromGrandmother = 30)
  (h3 : d.final = 51) :
  dollRatio d = 1/2 := by
  sorry

#eval dollRatio ⟨6, 30, 51⟩

end NUMINAMATH_CALUDE_peggy_doll_ratio_l3010_301029


namespace NUMINAMATH_CALUDE_power_of_product_l3010_301037

theorem power_of_product (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3010_301037


namespace NUMINAMATH_CALUDE_bob_remaining_corn_l3010_301034

/-- Represents the amount of corn, either in bushels or individual ears. -/
inductive CornAmount
| bushels (n : ℕ)
| ears (n : ℕ)

/-- Converts CornAmount to total number of ears. -/
def to_ears (amount : CornAmount) (ears_per_bushel : ℕ) : ℕ :=
  match amount with
  | CornAmount.bushels n => n * ears_per_bushel
  | CornAmount.ears n => n

/-- Calculates the remaining ears of corn after giving some away. -/
def remaining_corn (initial : CornAmount) (given_away : List CornAmount) (ears_per_bushel : ℕ) : ℕ :=
  to_ears initial ears_per_bushel - (given_away.map (λ a => to_ears a ears_per_bushel)).sum

theorem bob_remaining_corn :
  let initial := CornAmount.bushels 120
  let given_away := [
    CornAmount.bushels 15,  -- Terry
    CornAmount.bushels 8,   -- Jerry
    CornAmount.bushels 25,  -- Linda
    CornAmount.ears 42,     -- Stacy
    CornAmount.bushels 9,   -- Susan
    CornAmount.bushels 4,   -- Tim (bushels)
    CornAmount.ears 18      -- Tim (ears)
  ]
  let ears_per_bushel := 15
  remaining_corn initial given_away ears_per_bushel = 825 := by
  sorry

#eval remaining_corn (CornAmount.bushels 120) [
  CornAmount.bushels 15,
  CornAmount.bushels 8,
  CornAmount.bushels 25,
  CornAmount.ears 42,
  CornAmount.bushels 9,
  CornAmount.bushels 4,
  CornAmount.ears 18
] 15

end NUMINAMATH_CALUDE_bob_remaining_corn_l3010_301034


namespace NUMINAMATH_CALUDE_william_car_wash_time_l3010_301076

/-- Represents the time in minutes for each car washing task -/
structure CarWashTime where
  windows : ℕ
  body : ℕ
  tires : ℕ
  waxing : ℕ

/-- Calculates the total time for washing a normal car -/
def normalCarTime (t : CarWashTime) : ℕ :=
  t.windows + t.body + t.tires + t.waxing

/-- Theorem: William's total car washing time is 96 minutes -/
theorem william_car_wash_time :
  ∀ (t : CarWashTime),
  t.windows = 4 →
  t.body = 7 →
  t.tires = 4 →
  t.waxing = 9 →
  2 * normalCarTime t + 2 * normalCarTime t = 96 := by
  sorry

#check william_car_wash_time

end NUMINAMATH_CALUDE_william_car_wash_time_l3010_301076


namespace NUMINAMATH_CALUDE_fraction_reciprocal_sum_ge_two_l3010_301069

theorem fraction_reciprocal_sum_ge_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / b + b / a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_sum_ge_two_l3010_301069


namespace NUMINAMATH_CALUDE_total_cost_proof_l3010_301048

def hand_mitts_cost : ℝ := 14
def apron_cost : ℝ := 16
def utensils_cost : ℝ := 10
def knife_cost : ℝ := 2 * utensils_cost
def discount_rate : ℝ := 0.25
def num_sets : ℕ := 3

def total_cost : ℝ := num_sets * ((hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * (1 - discount_rate))

theorem total_cost_proof : total_cost = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l3010_301048


namespace NUMINAMATH_CALUDE_wax_needed_l3010_301033

theorem wax_needed (total_wax : ℕ) (available_wax : ℕ) (h1 : total_wax = 288) (h2 : available_wax = 28) :
  total_wax - available_wax = 260 := by
  sorry

end NUMINAMATH_CALUDE_wax_needed_l3010_301033


namespace NUMINAMATH_CALUDE_ellipse_equation_and_max_area_l3010_301067

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of an ellipse -/
structure Foci where
  left : Point
  right : Point

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

theorem ellipse_equation_and_max_area 
  (C : Ellipse) 
  (P : Point)
  (F : Foci)
  (h_P_on_C : P.x^2 / C.a^2 + P.y^2 / C.b^2 = 1)
  (h_P_coords : P.x = 1 ∧ P.y = Real.sqrt 2 / 2)
  (h_PF_sum : distance P F.left + distance P F.right = 2 * Real.sqrt 2) :
  (∃ (a b : ℝ), C.a = a ∧ C.b = b ∧ a^2 = 2 ∧ b^2 = 1) ∧
  (∃ (max_area : ℝ), 
    (∀ (Q : Point) (h_Q_on_C : Q.x^2 / C.a^2 + Q.y^2 / C.b^2 = 1),
      abs (P.x * Q.y - P.y * Q.x) / 2 ≤ max_area) ∧
    max_area = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_max_area_l3010_301067


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3010_301030

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def inscribed_in_parabola (q : Quadrilateral) : Prop :=
  parabola q.A.1 = q.A.2 ∧ parabola q.B.1 = q.B.2 ∧
  parabola q.C.1 = q.C.2 ∧ parabola q.D.1 = q.D.2

def angle_BAD_is_right (q : Quadrilateral) : Prop :=
  (q.B.1 - q.A.1) * (q.D.1 - q.A.1) + (q.B.2 - q.A.2) * (q.D.2 - q.A.2) = 0

def AC_parallel_to_x_axis (q : Quadrilateral) : Prop :=
  q.A.2 = q.C.2

def AC_bisects_BAD (q : Quadrilateral) : Prop :=
  (q.C.1 - q.A.1)^2 + (q.C.2 - q.A.2)^2 =
  (q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2

def diagonal_BD_length (q : Quadrilateral) (p : ℝ) : Prop :=
  (q.B.1 - q.D.1)^2 + (q.B.2 - q.D.2)^2 = p^2

-- The theorem
theorem quadrilateral_area (q : Quadrilateral) (p : ℝ) :
  inscribed_in_parabola q →
  angle_BAD_is_right q →
  AC_parallel_to_x_axis q →
  AC_bisects_BAD q →
  diagonal_BD_length q p →
  (q.A.1 - q.C.1) * (q.B.2 - q.D.2) / 2 = (p^2 - 4) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3010_301030


namespace NUMINAMATH_CALUDE_kim_water_consumption_l3010_301063

/-- Proves that the total amount of water Kim drinks is 60 ounces -/
theorem kim_water_consumption (quart_to_ounce : ℚ) (bottle_volume : ℚ) (can_volume : ℚ) :
  quart_to_ounce = 32 →
  bottle_volume = 3/2 →
  can_volume = 12 →
  bottle_volume * quart_to_ounce + can_volume = 60 := by
  sorry

end NUMINAMATH_CALUDE_kim_water_consumption_l3010_301063


namespace NUMINAMATH_CALUDE_prob_sum_seven_or_eleven_l3010_301085

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of ways to roll a sum of 7 -/
def waysToRollSeven : ℕ := 6

/-- The number of ways to roll a sum of 11 -/
def waysToRollEleven : ℕ := 2

/-- The total number of favorable outcomes (sum of 7 or 11) -/
def favorableOutcomes : ℕ := waysToRollSeven + waysToRollEleven

/-- The probability of rolling a sum of 7 or 11 with two fair six-sided dice -/
theorem prob_sum_seven_or_eleven : 
  (favorableOutcomes : ℚ) / totalOutcomes = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_seven_or_eleven_l3010_301085


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3010_301083

/-- An isosceles triangle with sides 12cm and 24cm has a perimeter of 60cm -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 12 →
  b = 24 →
  c = 24 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c = 60 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3010_301083


namespace NUMINAMATH_CALUDE_line_parameterization_l3010_301027

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 30

/-- The parameterization of the line -/
def parameterization (g : ℝ → ℝ) (t : ℝ) : ℝ × ℝ := (g t, 18 * t - 10)

/-- The theorem stating that g(t) = 9t + 10 satisfies the line equation and parameterization -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ t, line_equation (g t) (18 * t - 10)) ↔ (∀ t, g t = 9 * t + 10) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3010_301027


namespace NUMINAMATH_CALUDE_sand_pile_removal_l3010_301047

theorem sand_pile_removal (initial_weight : ℚ) (first_removal : ℚ) (second_removal : ℚ)
  (h1 : initial_weight = 8 / 3)
  (h2 : first_removal = 1 / 4)
  (h3 : second_removal = 5 / 6) :
  first_removal + second_removal = 13 / 12 := by
sorry

end NUMINAMATH_CALUDE_sand_pile_removal_l3010_301047


namespace NUMINAMATH_CALUDE_mri_to_xray_ratio_l3010_301087

/-- The cost of an x-ray in dollars -/
def x_ray_cost : ℝ := 250

/-- The cost of an MRI as a multiple of the x-ray cost -/
def mri_cost (k : ℝ) : ℝ := k * x_ray_cost

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.8

/-- The amount Mike paid in dollars -/
def mike_payment : ℝ := 200

/-- The theorem stating the ratio of MRI cost to x-ray cost -/
theorem mri_to_xray_ratio :
  ∃ k : ℝ,
    (1 - insurance_coverage) * (x_ray_cost + mri_cost k) = mike_payment ∧
    k = 3 :=
sorry

end NUMINAMATH_CALUDE_mri_to_xray_ratio_l3010_301087


namespace NUMINAMATH_CALUDE_not_all_angles_exceed_90_l3010_301040

/-- A plane quadrilateral is a geometric figure with four sides and four angles in a plane. -/
structure PlaneQuadrilateral where
  angles : Fin 4 → ℝ
  sum_of_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360

/-- Theorem: In a plane quadrilateral, it is impossible for all four internal angles to exceed 90°. -/
theorem not_all_angles_exceed_90 (q : PlaneQuadrilateral) : 
  ¬(∀ i : Fin 4, q.angles i > 90) := by
  sorry

end NUMINAMATH_CALUDE_not_all_angles_exceed_90_l3010_301040


namespace NUMINAMATH_CALUDE_power_congruence_l3010_301015

theorem power_congruence (h : 5^500 ≡ 1 [ZMOD 2000]) : 5^15000 ≡ 1 [ZMOD 2000] := by
  sorry

end NUMINAMATH_CALUDE_power_congruence_l3010_301015


namespace NUMINAMATH_CALUDE_laptop_cost_proof_l3010_301089

theorem laptop_cost_proof (x y : ℝ) (h1 : y = 3 * x) (h2 : x + y = 2000) : x = 500 := by
  sorry

end NUMINAMATH_CALUDE_laptop_cost_proof_l3010_301089


namespace NUMINAMATH_CALUDE_f_nonnegative_implies_a_eq_four_l3010_301041

/-- The function f(x) = ax^3 - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3*x + 1

/-- Theorem: If f(x) ≥ 0 for all x in [-1, 1], then a = 4 -/
theorem f_nonnegative_implies_a_eq_four (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_nonnegative_implies_a_eq_four_l3010_301041


namespace NUMINAMATH_CALUDE_equal_distances_l3010_301058

/-- The number of people seated at the round table. -/
def n : ℕ := 41

/-- The distance between two positions in a circular arrangement. -/
def circularDistance (a b : ℕ) : ℕ :=
  min ((a - b + n) % n) ((b - a + n) % n)

/-- Theorem stating that the distance from 31 to 7 equals the distance from 31 to 14
    in a circular arrangement of 41 people. -/
theorem equal_distances : circularDistance 31 7 = circularDistance 31 14 := by
  sorry


end NUMINAMATH_CALUDE_equal_distances_l3010_301058


namespace NUMINAMATH_CALUDE_highest_temperature_correct_l3010_301009

/-- The highest temperature reached during candy making --/
def highest_temperature (initial_temp final_temp : ℝ) (heating_rate cooling_rate : ℝ) (total_time : ℝ) : ℝ :=
  let T : ℝ := 240
  T

/-- Theorem stating that the highest temperature is correct --/
theorem highest_temperature_correct 
  (initial_temp : ℝ) (final_temp : ℝ) (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ)
  (h1 : initial_temp = 60)
  (h2 : final_temp = 170)
  (h3 : heating_rate = 5)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46) :
  let T := highest_temperature initial_temp final_temp heating_rate cooling_rate total_time
  (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time :=
by
  sorry

#check highest_temperature_correct

end NUMINAMATH_CALUDE_highest_temperature_correct_l3010_301009


namespace NUMINAMATH_CALUDE_second_outlet_rate_calculation_l3010_301086

/-- Represents the rate of the second outlet pipe in cubic inches per minute -/
def second_outlet_rate : ℝ := 9

/-- Tank volume in cubic feet -/
def tank_volume : ℝ := 30

/-- Inlet pipe rate in cubic inches per minute -/
def inlet_rate : ℝ := 3

/-- First outlet pipe rate in cubic inches per minute -/
def first_outlet_rate : ℝ := 6

/-- Time to empty the tank when all pipes are open, in minutes -/
def emptying_time : ℝ := 4320

/-- Conversion factor from cubic feet to cubic inches -/
def cubic_feet_to_inches : ℝ := 12 ^ 3

theorem second_outlet_rate_calculation :
  second_outlet_rate = 
    (tank_volume * cubic_feet_to_inches - emptying_time * (inlet_rate - first_outlet_rate)) / 
    emptying_time := by
  sorry

end NUMINAMATH_CALUDE_second_outlet_rate_calculation_l3010_301086


namespace NUMINAMATH_CALUDE_magician_hourly_rate_l3010_301064

/-- Proves that the hourly rate for a magician who works 3 hours per day for 2 weeks
    and receives a total payment of $2520 is $60 per hour. -/
theorem magician_hourly_rate :
  let hours_per_day : ℕ := 3
  let days : ℕ := 14
  let total_payment : ℕ := 2520
  let total_hours : ℕ := hours_per_day * days
  let hourly_rate : ℚ := total_payment / total_hours
  hourly_rate = 60 := by
  sorry

#check magician_hourly_rate

end NUMINAMATH_CALUDE_magician_hourly_rate_l3010_301064


namespace NUMINAMATH_CALUDE_car_speed_proof_l3010_301084

/-- The speed of the first car in miles per hour -/
def speed1 : ℝ := 52

/-- The time traveled in hours -/
def time : ℝ := 3.5

/-- The total distance between the cars after the given time in miles -/
def total_distance : ℝ := 385

/-- The speed of the second car in miles per hour -/
def speed2 : ℝ := 58

theorem car_speed_proof :
  speed1 * time + speed2 * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3010_301084


namespace NUMINAMATH_CALUDE_cos_negative_ninety_degrees_l3010_301018

theorem cos_negative_ninety_degrees : Real.cos (-(π / 2)) = 0 := by sorry

end NUMINAMATH_CALUDE_cos_negative_ninety_degrees_l3010_301018


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l3010_301092

theorem fixed_point_parabola :
  ∀ (t : ℝ), 36 = 4 * (3 : ℝ)^2 + t * 3 - 3 * t := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l3010_301092


namespace NUMINAMATH_CALUDE_tiles_difference_7_6_l3010_301042

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n ^ 2

/-- The theorem stating the difference in tiles between the 7th and 6th squares -/
theorem tiles_difference_7_6 : tiles_in_square 7 - tiles_in_square 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_7_6_l3010_301042


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3010_301025

theorem quadratic_coefficient_sum : ∃ (coeff_sum : ℤ),
  (∀ a : ℤ, 
    (∃ r s : ℤ, r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24 ∧ r + s = a) →
    (∃ x y : ℤ, x^2 + a*x + 24 = 0 ∧ y^2 + a*y + 24 = 0 ∧ x ≠ y ∧ x < 0 ∧ y < 0)) ∧
  (∀ a : ℤ,
    (∃ x y : ℤ, x^2 + a*x + 24 = 0 ∧ y^2 + a*y + 24 = 0 ∧ x ≠ y ∧ x < 0 ∧ y < 0) →
    (∃ r s : ℤ, r < 0 ∧ s < 0 ∧ r ≠ s ∧ r * s = 24 ∧ r + s = a)) ∧
  coeff_sum = -60 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l3010_301025


namespace NUMINAMATH_CALUDE_grade_swap_possible_l3010_301082

/-- Represents a grade scaling system -/
structure GradeScale where
  upper_limit : ℕ
  round_up_half : Set ℕ

/-- Represents a grade within a scaling system -/
def Grade := { g : ℚ // 0 < g ∧ g < 1 }

/-- Function to rescale a grade -/
def rescale (g : Grade) (old_scale new_scale : GradeScale) : Grade :=
  sorry

/-- Theorem stating that any two grades can be swapped through a series of rescalings -/
theorem grade_swap_possible (a b : Grade) :
  ∃ (scales : List GradeScale), 
    let final_scale := scales.foldl (λ acc s => s) { upper_limit := 100, round_up_half := ∅ }
    let new_a := scales.foldl (λ acc s => rescale acc s final_scale) a
    let new_b := scales.foldl (λ acc s => rescale acc s final_scale) b
    new_a = b ∧ new_b = a :=
  sorry

end NUMINAMATH_CALUDE_grade_swap_possible_l3010_301082


namespace NUMINAMATH_CALUDE_cookout_2006_l3010_301008

/-- The number of kids at the cookout in 2004 -/
def kids_2004 : ℕ := 60

/-- The number of kids at the cookout in 2005 -/
def kids_2005 : ℕ := kids_2004 / 2

/-- The number of kids at the cookout in 2006 -/
def kids_2006 : ℕ := kids_2005 * 2 / 3

/-- Theorem stating that the number of kids at the cookout in 2006 is 20 -/
theorem cookout_2006 : kids_2006 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookout_2006_l3010_301008


namespace NUMINAMATH_CALUDE_forum_member_count_l3010_301081

/-- The number of members in an online forum. -/
def forum_members : ℕ := 200

/-- The average number of questions posted per hour by each member. -/
def questions_per_hour : ℕ := 3

/-- The ratio of answers to questions posted by each member. -/
def answer_to_question_ratio : ℕ := 3

/-- The total number of posts (questions and answers) in a day. -/
def total_daily_posts : ℕ := 57600

/-- The number of hours in a day. -/
def hours_per_day : ℕ := 24

theorem forum_member_count :
  forum_members * (questions_per_hour * hours_per_day * (1 + answer_to_question_ratio)) = total_daily_posts :=
by sorry

end NUMINAMATH_CALUDE_forum_member_count_l3010_301081


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3010_301002

theorem prime_sum_squares (p q : ℕ) : 
  Prime p ∧ Prime q ∧ Prime (2^2 + p^2 + q^2) ↔ (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3) :=
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3010_301002


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l3010_301045

theorem at_least_one_not_less_than_six (a b : ℝ) (h : a + b = 12) : max a b ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l3010_301045


namespace NUMINAMATH_CALUDE_even_sine_function_l3010_301024

theorem even_sine_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (∀ x, f (-x) = f x) →
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_even_sine_function_l3010_301024


namespace NUMINAMATH_CALUDE_perp_to_countless_lines_necessary_not_sufficient_l3010_301004

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Perpendicularity between a line and a plane -/
def perp_line_plane (l : Line3D) (α : Plane3D) : Prop := sorry

/-- A line is perpendicular to countless lines within a plane -/
def perp_to_countless_lines (l : Line3D) (α : Plane3D) : Prop := sorry

/-- Main theorem: The statement "Line l is perpendicular to countless lines within plane α" 
    is a necessary but not sufficient condition for "l ⊥ α" -/
theorem perp_to_countless_lines_necessary_not_sufficient (l : Line3D) (α : Plane3D) :
  (perp_line_plane l α → perp_to_countless_lines l α) ∧
  ∃ l' α', perp_to_countless_lines l' α' ∧ ¬perp_line_plane l' α' := by
  sorry

end NUMINAMATH_CALUDE_perp_to_countless_lines_necessary_not_sufficient_l3010_301004


namespace NUMINAMATH_CALUDE_sin_330_degrees_l3010_301097

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l3010_301097


namespace NUMINAMATH_CALUDE_linear_equation_implies_mn_one_l3010_301011

/-- If (m+2)x^(|m|-1) + y^(2n) = 5 is a linear equation in x and y, where m and n are real numbers, then mn = 1 -/
theorem linear_equation_implies_mn_one (m n : ℝ) : 
  (∃ a b c : ℝ, ∀ x y : ℝ, (m + 2) * x^(|m| - 1) + y^(2*n) = 5 ↔ a*x + b*y = c) → 
  m * n = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_implies_mn_one_l3010_301011


namespace NUMINAMATH_CALUDE_system_solutions_l3010_301055

-- Define the system of equations
def system (t x y z : ℝ) : Prop :=
  t * (x + y + z) = 0 ∧ t * (x + y) + z = 1 ∧ t * x + y + z = 2

-- State the theorem
theorem system_solutions :
  ∀ t x y z : ℝ,
    (t = 0 → system t x y z ↔ y = 1 ∧ z = 1) ∧
    (t ≠ 0 ∧ t ≠ 1 → system t x y z ↔ x = 2 / (t - 1) ∧ y = -1 / (t - 1) ∧ z = -1 / (t - 1)) ∧
    (t = 1 → ¬∃ x y z, system t x y z) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l3010_301055


namespace NUMINAMATH_CALUDE_sector_central_angle_l3010_301090

/-- Given a circular sector with radius 10 and area 100, prove that the central angle is 2 radians. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (h1 : radius = 10) (h2 : area = 100) :
  (2 * area) / (radius ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3010_301090


namespace NUMINAMATH_CALUDE_prime_divisor_bound_l3010_301095

theorem prime_divisor_bound (p : ℕ) : 
  Prime p → 
  (Finset.card (Nat.divisors (p^2 + 71)) ≤ 10) → 
  p = 2 ∨ p = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_bound_l3010_301095


namespace NUMINAMATH_CALUDE_cosine_sum_inequality_l3010_301036

theorem cosine_sum_inequality (n : ℕ) (x : ℝ) :
  (Finset.range (n + 1)).sum (fun i => |Real.cos (2^i * x)|) ≥ n / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_inequality_l3010_301036


namespace NUMINAMATH_CALUDE_complex_calculation_l3010_301071

theorem complex_calculation : 550 - (104 / (Real.sqrt 20.8)^2)^3 = 425 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3010_301071


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_l3010_301096

theorem sqrt_a_sqrt_a (a : ℝ) (ha : 0 < a) : Real.sqrt (a * Real.sqrt a) = a^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_l3010_301096


namespace NUMINAMATH_CALUDE_third_place_prize_l3010_301049

def prize_distribution (total_people : ℕ) (contribution : ℕ) (first_place_percentage : ℚ) : ℚ :=
  let total_pot : ℚ := (total_people * contribution : ℚ)
  let first_place_prize : ℚ := total_pot * first_place_percentage
  let remaining : ℚ := total_pot - first_place_prize
  remaining / 2

theorem third_place_prize :
  prize_distribution 8 5 (4/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_place_prize_l3010_301049


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3010_301098

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {x | (x - 3) * (x - 1) < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3010_301098


namespace NUMINAMATH_CALUDE_asha_remaining_money_l3010_301019

/-- Calculates the remaining money for Asha after spending 3/4 of her total money --/
def remaining_money (brother_loan sister_loan father_loan mother_loan granny_gift savings : ℚ) : ℚ :=
  let total := brother_loan + sister_loan + father_loan + mother_loan + granny_gift + savings
  total - (3/4 * total)

/-- Theorem stating that Asha remains with $65 after spending --/
theorem asha_remaining_money :
  remaining_money 20 0 40 30 70 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_asha_remaining_money_l3010_301019


namespace NUMINAMATH_CALUDE_negative_number_identification_l3010_301080

theorem negative_number_identification :
  let a := -3^2
  let b := (-3)^2
  let c := |-3|
  let d := -(-3)
  (a < 0) ∧ (b ≥ 0) ∧ (c ≥ 0) ∧ (d ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l3010_301080


namespace NUMINAMATH_CALUDE_work_increase_percentage_l3010_301013

/-- Proves that when 1/5 of the members in an office are absent, 
    the percentage increase in work for each remaining person is 25% -/
theorem work_increase_percentage (p : ℝ) (W : ℝ) (h1 : p > 0) (h2 : W > 0) : 
  let original_work_per_person := W / p
  let remaining_persons := p * (4/5)
  let new_work_per_person := W / remaining_persons
  let increase_percentage := (new_work_per_person - original_work_per_person) / original_work_per_person * 100
  increase_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_work_increase_percentage_l3010_301013


namespace NUMINAMATH_CALUDE_division_scaling_l3010_301061

theorem division_scaling (a b c : ℝ) (h : a / b = c) :
  (a / 10) / (b / 10) = c := by
  sorry

end NUMINAMATH_CALUDE_division_scaling_l3010_301061


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3010_301017

theorem inequality_solution_range :
  ∀ (a : ℝ), (∃ x ∈ Set.Icc 0 3, x^2 - a*x - a + 1 ≥ 0) ↔ a ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3010_301017


namespace NUMINAMATH_CALUDE_s_square_minus_product_abs_eq_eight_l3010_301005

/-- The sequence s_n defined for three real numbers a, b, c -/
def s (a b c : ℝ) : ℕ → ℝ
  | 0 => 3  -- s_0 = a^0 + b^0 + c^0 = 3
  | n + 1 => a^(n + 1) + b^(n + 1) + c^(n + 1)

/-- The theorem statement -/
theorem s_square_minus_product_abs_eq_eight
  (a b c : ℝ)
  (h1 : s a b c 1 = 2)
  (h2 : s a b c 2 = 6)
  (h3 : s a b c 3 = 14) :
  ∀ n : ℕ, n > 1 → |(s a b c n)^2 - (s a b c (n-1)) * (s a b c (n+1))| = 8 := by
  sorry

end NUMINAMATH_CALUDE_s_square_minus_product_abs_eq_eight_l3010_301005


namespace NUMINAMATH_CALUDE_possible_student_counts_l3010_301039

def is_valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ (120 - 2) % (n - 1) = 0

theorem possible_student_counts :
  ∀ n : ℕ, is_valid_student_count n ↔ n = 2 ∨ n = 3 ∨ n = 60 ∨ n = 119 := by
  sorry

end NUMINAMATH_CALUDE_possible_student_counts_l3010_301039


namespace NUMINAMATH_CALUDE_system_solution_l3010_301075

/-- Given a system of equations and the condition that a ≠ bc, 
    prove that x = 1, y = 0, and z = 0 are the solutions. -/
theorem system_solution (a b c : ℝ) (h : a ≠ b * c) :
  ∃! (x y z : ℝ), 
    a = (a * x + c * y) / (b * z + 1) ∧
    b = (b * x + y) / (b * z + 1) ∧
    c = (a * z + c) / (b * z + 1) ∧
    x = 1 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3010_301075


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l3010_301079

theorem same_terminal_side (α β : Real) : 
  ∃ k : Int, α = β + 2 * π * (k : Real) → 
  α.cos = β.cos ∧ α.sin = β.sin :=
by sorry

theorem angle_with_same_terminal_side : 
  ∃ k : Int, (11 * π / 8 : Real) = (-5 * π / 8 : Real) + 2 * π * (k : Real) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l3010_301079


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_l3010_301000

theorem shaded_area_rectangle (total_width total_height : ℝ)
  (small_rect_width small_rect_height : ℝ)
  (triangle1_base triangle1_height : ℝ)
  (triangle2_base triangle2_height : ℝ) :
  total_width = 8 ∧ total_height = 5 ∧
  small_rect_width = 4 ∧ small_rect_height = 2 ∧
  triangle1_base = 5 ∧ triangle1_height = 2 ∧
  triangle2_base = 3 ∧ triangle2_height = 2 →
  total_width * total_height -
  (2 * small_rect_width * small_rect_height +
   2 * (1/2 * triangle1_base * triangle1_height) +
   2 * (1/2 * triangle2_base * triangle2_height)) = 6.5 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_l3010_301000


namespace NUMINAMATH_CALUDE_line_symmetry_l3010_301001

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Returns true if two lines are symmetric about the y-axis -/
def symmetricAboutYAxis (l1 l2 : Line) : Prop :=
  l1.slope = -l2.slope ∧ l1.intercept = l2.intercept

theorem line_symmetry (l1 l2 : Line) :
  l1.slope = 2 ∧ l1.intercept = 3 →
  symmetricAboutYAxis l1 l2 →
  l2.slope = -2 ∧ l2.intercept = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3010_301001


namespace NUMINAMATH_CALUDE_boat_current_speed_ratio_l3010_301032

/-- Proves that the ratio of boat speed to current speed is 4:1 given upstream and downstream travel times -/
theorem boat_current_speed_ratio 
  (distance : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h_upstream : upstream_time = 6) 
  (h_downstream : downstream_time = 10) 
  (h_positive_distance : distance > 0) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧ 
    current_speed > 0 ∧
    distance = upstream_time * (boat_speed - current_speed) ∧
    distance = downstream_time * (boat_speed + current_speed) ∧
    boat_speed = 4 * current_speed :=
sorry

end NUMINAMATH_CALUDE_boat_current_speed_ratio_l3010_301032


namespace NUMINAMATH_CALUDE_stating_downstream_speed_l3010_301020

/-- Represents the rowing speeds of a man in different conditions. -/
structure RowingSpeeds where
  upstream : ℝ
  still_water : ℝ
  downstream : ℝ

/-- 
Theorem stating that given a man's upstream rowing speed and still water speed,
we can determine his downstream speed.
-/
theorem downstream_speed (speeds : RowingSpeeds) 
  (h_upstream : speeds.upstream = 7)
  (h_still_water : speeds.still_water = 20)
  (h_average : speeds.still_water = (speeds.upstream + speeds.downstream) / 2) :
  speeds.downstream = 33 := by
  sorry

#check downstream_speed

end NUMINAMATH_CALUDE_stating_downstream_speed_l3010_301020


namespace NUMINAMATH_CALUDE_rectangular_to_cylindrical_l3010_301023

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := 6
  let θ : ℝ := 5 * Real.pi / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ ∧
  z = z :=
by
  sorry

#check rectangular_to_cylindrical

end NUMINAMATH_CALUDE_rectangular_to_cylindrical_l3010_301023


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3010_301077

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(23,24,30), (12,30,31), (9,30,32), (4,30,33), (15,22,36), (9,18,40), (4,15,42)}

theorem diophantine_equation_solution :
  {t : ℕ × ℕ × ℕ | is_valid_triple t.1 t.2.1 t.2.2} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3010_301077


namespace NUMINAMATH_CALUDE_find_a_value_l3010_301057

theorem find_a_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3010_301057


namespace NUMINAMATH_CALUDE_average_age_of_population_l3010_301070

/-- The average age of a population given the ratio of men to women and their respective average ages -/
theorem average_age_of_population 
  (ratio_men_to_women : ℚ) 
  (avg_age_men : ℝ) 
  (avg_age_women : ℝ) :
  ratio_men_to_women = 2/3 →
  avg_age_men = 37 →
  avg_age_women = 42 →
  let total_population := ratio_men_to_women + 1
  let weighted_age_men := ratio_men_to_women * avg_age_men
  let weighted_age_women := 1 * avg_age_women
  (weighted_age_men + weighted_age_women) / total_population = 40 :=
by sorry


end NUMINAMATH_CALUDE_average_age_of_population_l3010_301070


namespace NUMINAMATH_CALUDE_work_completion_time_l3010_301026

theorem work_completion_time (b a_and_b : ℝ) (hb : b = 8) (hab : a_and_b = 4.8) :
  let a := (1 / a_and_b - 1 / b)⁻¹
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3010_301026


namespace NUMINAMATH_CALUDE_prop_2_prop_3_l3010_301050

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State that m and n are distinct
variable (h_distinct_lines : m ≠ n)

-- State that α and β are different
variable (h_different_planes : α ≠ β)

-- Proposition ②
theorem prop_2 : 
  (parallel_planes α β ∧ subset m α) → parallel_lines m β :=
sorry

-- Proposition ③
theorem prop_3 : 
  (perpendicular n α ∧ perpendicular n β ∧ perpendicular m α) → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_prop_2_prop_3_l3010_301050


namespace NUMINAMATH_CALUDE_circle_product_arrangement_l3010_301091

theorem circle_product_arrangement : ∃ (a b c d e f : ℚ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a = b * f ∧
  b = a * c ∧
  c = b * d ∧
  d = c * e ∧
  e = d * f ∧
  f = e * a := by
  sorry


end NUMINAMATH_CALUDE_circle_product_arrangement_l3010_301091


namespace NUMINAMATH_CALUDE_pencil_length_l3010_301046

/-- The total length of a pencil with given colored sections -/
theorem pencil_length (purple_length black_length blue_length : ℝ) 
  (h_purple : purple_length = 1.5)
  (h_black : black_length = 0.5)
  (h_blue : blue_length = 2) :
  purple_length + black_length + blue_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l3010_301046


namespace NUMINAMATH_CALUDE_house_area_l3010_301072

theorem house_area (living_dining_kitchen_area master_bedroom_area : ℝ)
  (h1 : living_dining_kitchen_area = 1000)
  (h2 : master_bedroom_area = 1040)
  (guest_bedroom_area : ℝ)
  (h3 : guest_bedroom_area = (1 / 4) * master_bedroom_area) :
  living_dining_kitchen_area + master_bedroom_area + guest_bedroom_area = 2300 :=
by sorry

end NUMINAMATH_CALUDE_house_area_l3010_301072


namespace NUMINAMATH_CALUDE_hall_width_is_15_l3010_301093

/-- Represents the dimensions and cost information of a rectangular hall -/
structure Hall where
  length : ℝ
  height : ℝ
  width : ℝ
  cost_per_sqm : ℝ
  total_expenditure : ℝ

/-- Calculates the total area to be covered with mat in the hall -/
def total_area (h : Hall) : ℝ :=
  2 * (h.length * h.width) + 2 * (h.length * h.height) + 2 * (h.width * h.height)

/-- Theorem stating that given the hall's dimensions and cost information, the width is 15 meters -/
theorem hall_width_is_15 (h : Hall) 
  (h_length : h.length = 20)
  (h_height : h.height = 5)
  (h_cost : h.cost_per_sqm = 50)
  (h_expenditure : h.total_expenditure = 47500)
  (h_area_eq : h.total_expenditure = (total_area h) * h.cost_per_sqm) :
  h.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_hall_width_is_15_l3010_301093


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3010_301006

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3010_301006


namespace NUMINAMATH_CALUDE_specific_gold_cube_profit_l3010_301021

/-- Calculates the profit from selling a gold cube -/
def gold_cube_profit (side_length : ℝ) (density : ℝ) (purchase_price : ℝ) (markup : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * purchase_price
  let selling_price := cost * markup
  selling_price - cost

/-- Theorem stating the profit for a specific gold cube -/
theorem specific_gold_cube_profit :
  gold_cube_profit 6 19 60 1.5 = 123120 := by
  sorry

end NUMINAMATH_CALUDE_specific_gold_cube_profit_l3010_301021


namespace NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l3010_301038

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_red : red = 4)
  (h_blue : blue = 2) :
  (green + yellow : ℚ) / (green + yellow + red + blue) = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l3010_301038


namespace NUMINAMATH_CALUDE_class_average_problem_l3010_301052

theorem class_average_problem (N : ℝ) (h : N > 0) :
  let total_average : ℝ := 80
  let three_fourths_average : ℝ := 76
  let one_fourth_average : ℝ := (4 * total_average * N - 3 * three_fourths_average * N) / N
  one_fourth_average = 92 := by sorry

end NUMINAMATH_CALUDE_class_average_problem_l3010_301052


namespace NUMINAMATH_CALUDE_eight_b_equals_sixteen_l3010_301007

theorem eight_b_equals_sixteen
  (h1 : 6 * a + 3 * b = 0)
  (h2 : b - 3 = a)
  (h3 : b + c = 5)
  : 8 * b = 16 := by
  sorry

end NUMINAMATH_CALUDE_eight_b_equals_sixteen_l3010_301007


namespace NUMINAMATH_CALUDE_expression_evaluation_l3010_301056

theorem expression_evaluation :
  (2 ^ 2010 * 3 ^ 2012 * 5 ^ 2) / 6 ^ 2011 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3010_301056


namespace NUMINAMATH_CALUDE_eggs_distribution_l3010_301094

/-- Given a total number of eggs and a number of groups, 
    calculate the number of eggs per group -/
def eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) : ℕ :=
  total_eggs / num_groups

/-- Theorem stating that with 18 eggs split into 3 groups, 
    each group should have 6 eggs -/
theorem eggs_distribution :
  eggs_per_group 18 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_distribution_l3010_301094


namespace NUMINAMATH_CALUDE_symmetric_difference_equality_l3010_301035

open Set

theorem symmetric_difference_equality (A B K : Set α) : 
  symmDiff A K = symmDiff B K → A = B :=
by sorry

end NUMINAMATH_CALUDE_symmetric_difference_equality_l3010_301035


namespace NUMINAMATH_CALUDE_civil_service_exam_probability_l3010_301053

theorem civil_service_exam_probability 
  (pass_rate_written : ℝ) 
  (pass_rate_overall : ℝ) 
  (h1 : pass_rate_written = 0.2) 
  (h2 : pass_rate_overall = 0.04) :
  pass_rate_overall / pass_rate_written = 0.2 :=
sorry

end NUMINAMATH_CALUDE_civil_service_exam_probability_l3010_301053


namespace NUMINAMATH_CALUDE_special_linear_function_unique_l3010_301073

/-- A linear function f such that f(f(x)) = x + 2 -/
def special_linear_function (f : ℝ → ℝ) : Prop :=
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧ 
  (∀ x, f (f x) = x + 2)

/-- The unique linear function satisfying f(f(x)) = x + 2 is f(x) = x + 1 -/
theorem special_linear_function_unique (f : ℝ → ℝ) :
  special_linear_function f → (∀ x, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_special_linear_function_unique_l3010_301073


namespace NUMINAMATH_CALUDE_distinct_c_values_l3010_301065

theorem distinct_c_values (r s t u : ℂ) (h_distinct : r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ s ≠ t ∧ s ≠ u ∧ t ≠ u) 
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = 
    (z - r * c) * (z - s * c) * (z - t * c) * (z - u * c)) : 
  ∃! (values : Finset ℂ), values.card = 4 ∧ ∀ c : ℂ, c ∈ values ↔ 
    (∀ z : ℂ, (z - r) * (z - s) * (z - t) * (z - u) = 
      (z - r * c) * (z - s * c) * (z - t * c) * (z - u * c)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_c_values_l3010_301065


namespace NUMINAMATH_CALUDE_smallest_valid_N_exists_l3010_301022

def is_valid_configuration (N : ℕ) (c₁ c₂ c₃ c₄ c₅ c₆ : ℕ) : Prop :=
  c₁ ≤ N ∧ c₂ ≤ N ∧ c₃ ≤ N ∧ c₄ ≤ N ∧ c₅ ≤ N ∧ c₆ ≤ N ∧
  c₁ = 6 * c₂ - 1 ∧
  N + c₂ = 6 * c₃ - 2 ∧
  2 * N + c₃ = 6 * c₄ - 3 ∧
  3 * N + c₄ = 6 * c₅ - 4 ∧
  4 * N + c₅ = 6 * c₆ - 5 ∧
  5 * N + c₆ = 6 * c₁

theorem smallest_valid_N_exists :
  ∃ N : ℕ, N > 0 ∧ 
  (∃ c₁ c₂ c₃ c₄ c₅ c₆ : ℕ, is_valid_configuration N c₁ c₂ c₃ c₄ c₅ c₆) ∧
  (∀ M : ℕ, M < N → ¬∃ c₁ c₂ c₃ c₄ c₅ c₆ : ℕ, is_valid_configuration M c₁ c₂ c₃ c₄ c₅ c₆) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_N_exists_l3010_301022


namespace NUMINAMATH_CALUDE_intersection_when_a_is_four_subset_iff_a_geq_four_l3010_301051

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem 1: When a = 4, A ∩ B = A
theorem intersection_when_a_is_four :
  A ∩ B 4 = A := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 4
theorem subset_iff_a_geq_four (a : ℝ) :
  A ⊆ B a ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_four_subset_iff_a_geq_four_l3010_301051


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3010_301078

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem simple_interest_problem :
  let principal : ℝ := 10000
  let rate : ℝ := 0.08
  let time : ℝ := 1
  simple_interest principal rate time = 800 := by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3010_301078


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3010_301099

theorem solve_system_of_equations (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 210)
  (eq2 : t = 3 * s) : 
  s = 35 / 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3010_301099


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3010_301068

theorem arithmetic_sequence_common_difference (d : ℕ+) : 
  (∃ n : ℕ, 1 + (n - 1) * d.val = 81) → d ≠ 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3010_301068


namespace NUMINAMATH_CALUDE_ellie_wide_reflections_count_l3010_301010

/-- The number of times Sarah sees her reflection in tall mirror rooms -/
def sarah_tall_reflections : ℕ := 10

/-- The number of times Sarah sees her reflection in wide mirror rooms -/
def sarah_wide_reflections : ℕ := 5

/-- The number of times Ellie sees her reflection in tall mirror rooms -/
def ellie_tall_reflections : ℕ := 6

/-- The number of times both Sarah and Ellie passed through tall mirror rooms -/
def tall_room_visits : ℕ := 3

/-- The number of times both Sarah and Ellie passed through wide mirror rooms -/
def wide_room_visits : ℕ := 5

/-- The total number of reflections for both Sarah and Ellie -/
def total_reflections : ℕ := 88

/-- The number of times Ellie sees her reflection in wide mirror rooms -/
def ellie_wide_reflections : ℕ := 3

theorem ellie_wide_reflections_count :
  sarah_tall_reflections * tall_room_visits +
  sarah_wide_reflections * wide_room_visits +
  ellie_tall_reflections * tall_room_visits +
  ellie_wide_reflections * wide_room_visits = total_reflections :=
by sorry

end NUMINAMATH_CALUDE_ellie_wide_reflections_count_l3010_301010


namespace NUMINAMATH_CALUDE_system_solution_l3010_301062

theorem system_solution (x y z : ℝ) : 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  (x^2 + y^2 = -x + 3*y + z ∧
   y^2 + z^2 = x + 3*y - z ∧
   x^2 + z^2 = 2*x + 2*y - z) →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ 
   (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3010_301062
