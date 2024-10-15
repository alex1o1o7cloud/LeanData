import Mathlib

namespace NUMINAMATH_CALUDE_min_values_and_corresponding_points_l1535_153541

theorem min_values_and_corresponding_points (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 2) : 
  (∃ (min_ab : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → x*y ≥ min_ab ∧ a*b = min_ab) ∧
  (∃ (min_sum : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 2 → x + 2*y ≥ min_sum ∧ a + 2*b = min_sum) ∧
  a = (1 + Real.sqrt 2) / 2 ∧ b = (2 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_values_and_corresponding_points_l1535_153541


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l1535_153553

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_of_union_A_B : (U \ (A ∪ B)) = {-2, 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l1535_153553


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1535_153565

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1535_153565


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1535_153563

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + a*x - 2 = 0 ↔ x = x₁ ∨ x = x₂) →  -- equation has exactly two roots
  x₁ ≠ x₂ →  -- roots are distinct
  x₁ < -1 →
  x₂ > 1 →
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1535_153563


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_negative_l1535_153566

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing_negative : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_negative_l1535_153566


namespace NUMINAMATH_CALUDE_escalator_length_l1535_153559

/-- The length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
theorem escalator_length 
  (escalator_speed : ℝ) 
  (walking_speed : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15) 
  (h2 : walking_speed = 3) 
  (h3 : time_taken = 10) : 
  escalator_speed * time_taken + walking_speed * time_taken = 180 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_l1535_153559


namespace NUMINAMATH_CALUDE_parallelogram_side_ge_altitude_l1535_153586

/-- A parallelogram with side lengths and altitudes. -/
structure Parallelogram where
  side_a : ℝ
  side_b : ℝ
  altitude_a : ℝ
  altitude_b : ℝ
  side_a_pos : 0 < side_a
  side_b_pos : 0 < side_b
  altitude_a_pos : 0 < altitude_a
  altitude_b_pos : 0 < altitude_b

/-- 
Theorem: For any parallelogram, there exists a side length that is 
greater than or equal to the altitude perpendicular to that side.
-/
theorem parallelogram_side_ge_altitude (p : Parallelogram) :
  (p.side_a ≥ p.altitude_a) ∨ (p.side_b ≥ p.altitude_b) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_side_ge_altitude_l1535_153586


namespace NUMINAMATH_CALUDE_teacher_weight_l1535_153527

theorem teacher_weight (num_students : ℕ) (student_avg_weight : ℝ) (avg_increase : ℝ) :
  num_students = 24 →
  student_avg_weight = 35 →
  avg_increase = 0.4 →
  let total_student_weight := num_students * student_avg_weight
  let new_avg := student_avg_weight + avg_increase
  let total_weight_with_teacher := new_avg * (num_students + 1)
  total_weight_with_teacher - total_student_weight = 45 := by
  sorry

end NUMINAMATH_CALUDE_teacher_weight_l1535_153527


namespace NUMINAMATH_CALUDE_molecular_properties_l1535_153567

structure MolecularSystem where
  surface_distance : ℝ
  internal_distance : ℝ
  surface_attraction : Bool

structure IdealGas where
  temperature : ℝ
  collision_frequency : ℝ

structure OleicAcid where
  diameter : ℝ
  molar_volume : ℝ

def surface_tension (ms : MolecularSystem) : Prop :=
  ms.surface_distance > ms.internal_distance ∧ ms.surface_attraction

def gas_collision_frequency (ig : IdealGas) : Prop :=
  ig.collision_frequency = ig.temperature

def avogadro_estimation (oa : OleicAcid) : Prop :=
  oa.diameter > 0 ∧ oa.molar_volume > 0

theorem molecular_properties 
  (ms : MolecularSystem) 
  (ig : IdealGas) 
  (oa : OleicAcid) : 
  surface_tension ms ∧ 
  gas_collision_frequency ig ∧ 
  avogadro_estimation oa :=
sorry

end NUMINAMATH_CALUDE_molecular_properties_l1535_153567


namespace NUMINAMATH_CALUDE_angie_coffee_amount_l1535_153551

/-- Represents the number of cups of coffee brewed per pound of coffee. -/
def cupsPerPound : ℕ := 40

/-- Represents the number of cups of coffee Angie drinks per day. -/
def cupsPerDay : ℕ := 3

/-- Represents the number of days the coffee lasts. -/
def daysLasting : ℕ := 40

/-- Calculates the number of pounds of coffee Angie bought. -/
def coffeeAmount : ℕ := (cupsPerDay * daysLasting) / cupsPerPound

theorem angie_coffee_amount : coffeeAmount = 3 := by
  sorry

end NUMINAMATH_CALUDE_angie_coffee_amount_l1535_153551


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1535_153515

theorem rectangle_perimeter (x y : ℝ) 
  (h1 : 2*x + 2*y = x/2 + 2*y + 18) 
  (h2 : x*y = x*y/4 + 18) : 
  2*x + 2*y = 28 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1535_153515


namespace NUMINAMATH_CALUDE_exists_function_satisfying_conditions_l1535_153595

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_derivative_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, deriv f (-x) - deriv f x = 2 * Real.sqrt 2 * Real.sin x

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x, x > -3 * Real.pi / 2 → f x ≤ Real.exp (x + Real.pi / 4) - Real.pi / 4

theorem exists_function_satisfying_conditions :
  ∃ f : ℝ → ℝ,
    is_even f ∧
    satisfies_derivative_condition f ∧
    satisfies_inequality f ∧
    f = fun x ↦ Real.sqrt 2 * Real.cos x - 10 := by sorry

end NUMINAMATH_CALUDE_exists_function_satisfying_conditions_l1535_153595


namespace NUMINAMATH_CALUDE_remainder_problem_l1535_153516

theorem remainder_problem : (1989 * 1990 * 1991 + 1992^2) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1535_153516


namespace NUMINAMATH_CALUDE_platform_length_l1535_153581

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 10 seconds, prove that the platform length is 870 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 10) :
  let train_speed := train_length / pole_time
  let platform_length := train_speed * platform_time - train_length
  platform_length = 870 := by sorry

end NUMINAMATH_CALUDE_platform_length_l1535_153581


namespace NUMINAMATH_CALUDE_sum_divisible_by_3_probability_l1535_153575

/-- Represents the outcome of rolling a fair 6-sided die -/
def DieRoll : Type := Fin 6

/-- The sample space of rolling a fair die three times -/
def SampleSpace : Type := DieRoll × DieRoll × DieRoll

/-- The number of possible outcomes in the sample space -/
def totalOutcomes : Nat := 216

/-- Predicate for outcomes where the sum is divisible by 3 -/
def sumDivisibleBy3 (outcome : SampleSpace) : Prop :=
  (outcome.1.val + outcome.2.1.val + outcome.2.2.val + 3) % 3 = 0

/-- The number of favorable outcomes (sum divisible by 3) -/
def favorableOutcomes : Nat := 72

/-- The probability of the sum being divisible by 3 -/
def probability : ℚ := favorableOutcomes / totalOutcomes

theorem sum_divisible_by_3_probability :
  probability = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_3_probability_l1535_153575


namespace NUMINAMATH_CALUDE_find_x_l1535_153508

theorem find_x : ∃ x : ℕ, 
  (∃ k : ℕ, x = 9 * k) ∧ 
  x^2 > 120 ∧ 
  x < 25 ∧ 
  x % 2 = 1 ∧
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_find_x_l1535_153508


namespace NUMINAMATH_CALUDE_dolphins_score_l1535_153548

theorem dolphins_score (total_points sharks_points dolphins_points : ℕ) : 
  total_points = 36 →
  sharks_points = dolphins_points + 12 →
  sharks_points + dolphins_points = total_points →
  dolphins_points = 12 := by
sorry

end NUMINAMATH_CALUDE_dolphins_score_l1535_153548


namespace NUMINAMATH_CALUDE_quartic_root_product_l1535_153578

theorem quartic_root_product (a : ℝ) : 
  (∃ x y : ℝ, x * y = -32 ∧ 
   x^4 - 18*x^3 + a*x^2 + 200*x - 1984 = 0 ∧
   y^4 - 18*y^3 + a*y^2 + 200*y - 1984 = 0) →
  a = 86 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_product_l1535_153578


namespace NUMINAMATH_CALUDE_prob_A_and_B_l1535_153582

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.85

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring simultaneously
    is equal to the product of their individual probabilities -/
theorem prob_A_and_B : prob_A * prob_B = 0.51 := by sorry

end NUMINAMATH_CALUDE_prob_A_and_B_l1535_153582


namespace NUMINAMATH_CALUDE_x_power_2188_minus_reciprocal_l1535_153502

theorem x_power_2188_minus_reciprocal (x : ℂ) :
  x - (1 / x) = Complex.I * Real.sqrt 3 →
  x^2188 - (1 / x^2188) = -1 := by sorry

end NUMINAMATH_CALUDE_x_power_2188_minus_reciprocal_l1535_153502


namespace NUMINAMATH_CALUDE_parabola_properties_l1535_153517

/-- Represents a quadratic function of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

def Parabola.opensDownward (p : Parabola) : Prop := p.a < 0

def Parabola.axisOfSymmetry (p : Parabola) : ℝ := p.h

def Parabola.vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

def Parabola.increasingOnInterval (p : Parabola) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → p.a * (x - p.h)^2 + p.k < p.a * (y - p.h)^2 + p.k

theorem parabola_properties (p : Parabola) (h1 : p.a = -1) (h2 : p.h = -1) (h3 : p.k = 3) :
  (p.opensDownward ∧ 
   p.vertex = (-1, 3) ∧ 
   ¬(p.axisOfSymmetry = 1) ∧ 
   ¬(p.increasingOnInterval 0 (-p.h))) := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1535_153517


namespace NUMINAMATH_CALUDE_remainder_problem_l1535_153519

theorem remainder_problem (d r : ℤ) : 
  d > 1 ∧ 
  1012 % d = r ∧ 
  1548 % d = r ∧ 
  2860 % d = r → 
  d - r = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1535_153519


namespace NUMINAMATH_CALUDE_equation_solution_l1535_153524

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ -3 ∧ (2 / x + x / (x + 3) = 1) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1535_153524


namespace NUMINAMATH_CALUDE_inverse_variation_example_l1535_153526

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_variation_example :
  ∀ x y : ℝ → ℝ,
  VaryInversely x y →
  y 1500 = 0.4 →
  y 3000 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l1535_153526


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_theorem_l1535_153555

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by its six vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- The intersection point of the diagonals -/
def G (h : Hexagon) : Point := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a hexagon is convex -/
def isConvex (h : Hexagon) : Prop := sorry

/-- Checks if a hexagon is inscribed in a circle -/
def isInscribed (h : Hexagon) : Prop := sorry

/-- Checks if three lines intersect at a point forming 60° angles -/
def intersectAt60Degrees (p1 p2 p3 p4 p5 p6 : Point) : Prop := sorry

/-- The main theorem -/
theorem hexagon_diagonal_intersection_theorem (h : Hexagon) 
  (convex : isConvex h)
  (inscribed : isInscribed h)
  (intersect : intersectAt60Degrees h.A h.D h.B h.E h.C h.F) :
  distance (G h) h.A + distance (G h) h.C + distance (G h) h.E =
  distance (G h) h.B + distance (G h) h.D + distance (G h) h.F := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_theorem_l1535_153555


namespace NUMINAMATH_CALUDE_shaded_area_equals_circle_area_l1535_153554

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- Configuration of the diagram -/
structure DiagramConfig where
  x : ℝ
  A : Point
  B : Point
  C : Point
  D : Point
  semicircleAB : Semicircle
  semicircleAC : Semicircle
  semicircleCB : Semicircle

/-- The main theorem -/
theorem shaded_area_equals_circle_area (config : DiagramConfig) : 
  (config.A.x - config.B.x = 8 * config.x) →
  (config.A.x - config.C.x = 6 * config.x) →
  (config.C.x - config.B.x = 2 * config.x) →
  (config.D.y - config.C.y = Real.sqrt 3 * config.x) →
  (config.semicircleAB.radius = 4 * config.x) →
  (config.semicircleAC.radius = 3 * config.x) →
  (config.semicircleCB.radius = config.x) →
  (config.semicircleAB.center.x = (config.A.x + config.B.x) / 2) →
  (config.semicircleAC.center.x = (config.A.x + config.C.x) / 2) →
  (config.semicircleCB.center.x = (config.C.x + config.B.x) / 2) →
  (config.A.y = config.B.y) →
  (config.C.y = config.A.y) →
  (config.D.x = config.C.x) →
  (π * (4 * config.x)^2 / 2 - π * (3 * config.x)^2 / 2 - π * config.x^2 / 2 = π * (Real.sqrt 3 * config.x)^2) :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_equals_circle_area_l1535_153554


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_123_l1535_153503

theorem last_three_digits_of_7_to_123 : 7^123 % 1000 = 773 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_123_l1535_153503


namespace NUMINAMATH_CALUDE_exists_k_good_iff_k_ge_two_l1535_153537

/-- A function is k-good if the GCD of f(m) + n and f(n) + m is at most k for all m ≠ n -/
def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

/-- There exists a k-good function if and only if k ≥ 2 -/
theorem exists_k_good_iff_k_ge_two :
  ∀ k : ℕ, (∃ f : ℕ+ → ℕ+, IsKGood k f) ↔ k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_exists_k_good_iff_k_ge_two_l1535_153537


namespace NUMINAMATH_CALUDE_kathryn_remaining_money_l1535_153500

/-- Calculates the remaining money after expenses for Kathryn -/
def remaining_money (initial_rent : ℕ) (salary : ℕ) : ℕ :=
  let food_travel : ℕ := 2 * initial_rent
  let new_rent : ℕ := initial_rent / 2
  let total_expenses : ℕ := new_rent + food_travel
  salary - total_expenses

/-- Proves that Kathryn's remaining money is $2000 -/
theorem kathryn_remaining_money :
  remaining_money 1200 5000 = 2000 := by
  sorry

#eval remaining_money 1200 5000

end NUMINAMATH_CALUDE_kathryn_remaining_money_l1535_153500


namespace NUMINAMATH_CALUDE_even_odd_handshakers_l1535_153579

theorem even_odd_handshakers (population : ℕ) : ∃ (even_shakers odd_shakers : ℕ),
  even_shakers + odd_shakers = population ∧ 
  Even odd_shakers := by
  sorry

end NUMINAMATH_CALUDE_even_odd_handshakers_l1535_153579


namespace NUMINAMATH_CALUDE_train_distance_theorem_l1535_153532

/-- Represents the train journey with given conditions -/
structure TrainJourney where
  speed : ℝ
  stop_interval : ℝ
  regular_stop_duration : ℝ
  fifth_stop_duration : ℝ
  total_travel_time : ℝ

/-- Calculates the total distance traveled by the train -/
def total_distance (journey : TrainJourney) : ℝ :=
  sorry

/-- Theorem stating the total distance traveled by the train -/
theorem train_distance_theorem (journey : TrainJourney) 
  (h1 : journey.speed = 60)
  (h2 : journey.stop_interval = 48)
  (h3 : journey.regular_stop_duration = 1/6)
  (h4 : journey.fifth_stop_duration = 1/2)
  (h5 : journey.total_travel_time = 58) :
  total_distance journey = 2870 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l1535_153532


namespace NUMINAMATH_CALUDE_jimmy_wins_l1535_153509

/-- Represents a fan with four blades -/
structure Fan :=
  (rotation_speed : ℝ)
  (blade_count : ℕ)

/-- Represents a bullet trajectory -/
structure Trajectory :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Checks if a trajectory intersects a blade at a given position and time -/
def intersects_blade (f : Fan) (t : Trajectory) (position : ℕ) (time : ℝ) : Prop :=
  sorry

/-- The main theorem stating that there exists a trajectory that intersects all blades -/
theorem jimmy_wins (f : Fan) : 
  f.rotation_speed = 50 ∧ f.blade_count = 4 → 
  ∃ t : Trajectory, ∀ p : ℕ, p < f.blade_count → 
    ∃ time : ℝ, intersects_blade f t p time :=
sorry

end NUMINAMATH_CALUDE_jimmy_wins_l1535_153509


namespace NUMINAMATH_CALUDE_max_value_of_f_l1535_153589

noncomputable def f (a b x : ℝ) : ℝ := (4 - x^2) * (a * x^2 + b * x + 5)

theorem max_value_of_f (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-3 - x)) →
  (∃ x : ℝ, ∀ y : ℝ, f a b y ≤ f a b x) ∧
  (∃ x : ℝ, f a b x = 36) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1535_153589


namespace NUMINAMATH_CALUDE_complex_locus_is_ellipse_l1535_153533

theorem complex_locus_is_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (w : ℂ), w = 2 * z + 1 / z →
  (w.re / a) ^ 2 + (w.im / b) ^ 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_locus_is_ellipse_l1535_153533


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1535_153504

theorem min_value_sum_squares (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 3*y₁ + 2*y₂ + y₃ = 90) : 
  y₁^2 + 4*y₂^2 + 9*y₃^2 ≥ 4050/7 ∧ 
  ∃ y₁' y₂' y₃', y₁'^2 + 4*y₂'^2 + 9*y₃'^2 = 4050/7 ∧ 
                 y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 
                 3*y₁' + 2*y₂' + y₃' = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1535_153504


namespace NUMINAMATH_CALUDE_double_inequality_solution_l1535_153596

theorem double_inequality_solution (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9 * x + 3) ↔ 
  (x > 3 + 2 * Real.sqrt 2 ∧ x < 5.5 + Real.sqrt 32.25) :=
sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l1535_153596


namespace NUMINAMATH_CALUDE_drama_club_subject_distribution_l1535_153587

theorem drama_club_subject_distribution (total : ℕ) (math physics chem : ℕ) 
  (math_physics math_chem physics_chem : ℕ) (all_three : ℕ) :
  total = 100 ∧ 
  math = 50 ∧ 
  physics = 40 ∧ 
  chem = 30 ∧ 
  math_physics = 20 ∧ 
  physics_chem = 10 ∧ 
  all_three = 5 →
  total - (math + physics + chem - math_physics - physics_chem - math_chem + all_three) = 20 :=
by sorry

end NUMINAMATH_CALUDE_drama_club_subject_distribution_l1535_153587


namespace NUMINAMATH_CALUDE_quadratic_max_value_quadratic_max_value_achieved_l1535_153547

theorem quadratic_max_value (s : ℝ) : -3 * s^2 + 24 * s - 7 ≤ 41 := by sorry

theorem quadratic_max_value_achieved : ∃ s : ℝ, -3 * s^2 + 24 * s - 7 = 41 := by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_quadratic_max_value_achieved_l1535_153547


namespace NUMINAMATH_CALUDE_cube_edge_length_l1535_153562

theorem cube_edge_length (box_edge : ℝ) (num_cubes : ℕ) (h1 : box_edge = 1) (h2 : num_cubes = 1000) :
  ∃ (cube_edge : ℝ), cube_edge^3 * num_cubes = box_edge^3 ∧ cube_edge = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1535_153562


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1535_153570

theorem point_in_fourth_quadrant (a b : ℝ) (z z₁ z₂ : ℂ) :
  z = a + b * Complex.I ∧
  z₁ = 1 + Complex.I ∧
  z₂ = 3 - Complex.I ∧
  z = z₁ * z₂ →
  a > 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1535_153570


namespace NUMINAMATH_CALUDE_house_rent_fraction_l1535_153576

theorem house_rent_fraction (salary : ℚ) (food_fraction : ℚ) (clothes_fraction : ℚ) (remaining : ℚ) :
  salary = 180000 →
  food_fraction = 1/5 →
  clothes_fraction = 3/5 →
  remaining = 18000 →
  ∃ (house_rent_fraction : ℚ),
    house_rent_fraction * salary + food_fraction * salary + clothes_fraction * salary + remaining = salary ∧
    house_rent_fraction = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l1535_153576


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1535_153529

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1535_153529


namespace NUMINAMATH_CALUDE_club_members_count_l1535_153531

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 8

/-- The total cost of apparel for all members in dollars -/
def total_cost : ℕ := 4440

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 1

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + tshirt_additional_cost

/-- The cost of apparel for one member in dollars -/
def cost_per_member : ℕ := sock_cost * socks_per_member + tshirt_cost * tshirts_per_member

/-- The number of members in the club -/
def club_members : ℕ := total_cost / cost_per_member

theorem club_members_count : club_members = 130 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l1535_153531


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1535_153525

theorem consecutive_integers_sum (n : ℤ) : 
  (∀ k : ℤ, n - 4 ≤ k ∧ k ≤ n + 4 → k > 0) →
  (n - 4) + (n - 3) + (n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 99 →
  n + 4 = 15 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1535_153525


namespace NUMINAMATH_CALUDE_solution_of_system_l1535_153591

theorem solution_of_system (α β : ℝ) : 
  (∃ (n k : ℤ), (α = π/6 ∨ α = -π/6) ∧ α = α + 2*π*n ∧ 
                 (β = π/4 ∨ β = -π/4) ∧ β = β + 2*π*k) ∨
  (∃ (n k : ℤ), (α = π/4 ∨ α = -π/4) ∧ α = α + 2*π*n ∧ 
                 (β = π/6 ∨ β = -π/6) ∧ β = β + 2*π*k) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l1535_153591


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1535_153593

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x : ℝ, x^3 - 18*x^2 + 77*x - 120 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 18*s^2 + 77*s - 120) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 196 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1535_153593


namespace NUMINAMATH_CALUDE_masking_tape_length_l1535_153539

/-- The total length of masking tape needed for four walls -/
def total_tape_length (wall_width1 : ℝ) (wall_width2 : ℝ) : ℝ :=
  2 * wall_width1 + 2 * wall_width2

/-- Theorem: The total length of masking tape needed is 20 meters -/
theorem masking_tape_length :
  total_tape_length 4 6 = 20 :=
by sorry

end NUMINAMATH_CALUDE_masking_tape_length_l1535_153539


namespace NUMINAMATH_CALUDE_minimum_additional_marbles_lisa_additional_marbles_l1535_153599

theorem minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let required_marbles := (num_friends * (num_friends + 1)) / 2
  if required_marbles > initial_marbles then
    required_marbles - initial_marbles
  else
    0

theorem lisa_additional_marbles :
  minimum_additional_marbles 12 40 = 38 := by
  sorry

end NUMINAMATH_CALUDE_minimum_additional_marbles_lisa_additional_marbles_l1535_153599


namespace NUMINAMATH_CALUDE_peters_expression_exists_l1535_153588

/-- An expression type that can represent sums and products of ones -/
inductive Expr
  | one : Expr
  | add : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Evaluate an expression -/
def eval : Expr → Nat
  | Expr.one => 1
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

/-- Swap addition and multiplication in an expression -/
def swap : Expr → Expr
  | Expr.one => Expr.one
  | Expr.add e1 e2 => Expr.mul (swap e1) (swap e2)
  | Expr.mul e1 e2 => Expr.add (swap e1) (swap e2)

/-- There exists an expression that evaluates to 2014 and still evaluates to 2014 after swapping + and × -/
theorem peters_expression_exists : ∃ e : Expr, eval e = 2014 ∧ eval (swap e) = 2014 := by
  sorry

end NUMINAMATH_CALUDE_peters_expression_exists_l1535_153588


namespace NUMINAMATH_CALUDE_milk_storage_theorem_l1535_153556

def initial_milk : ℕ := 30000
def pump_out_rate : ℕ := 2880
def pump_out_hours : ℕ := 4
def add_milk_hours : ℕ := 7
def initial_add_rate : ℕ := 1200
def add_rate_increase : ℕ := 200

def final_milk_amount : ℕ := 31080

theorem milk_storage_theorem :
  let milk_after_pump_out := initial_milk - pump_out_rate * pump_out_hours
  let milk_added := add_milk_hours * (initial_add_rate + (initial_add_rate + (add_milk_hours - 1) * add_rate_increase)) / 2
  milk_after_pump_out + milk_added = final_milk_amount := by sorry

end NUMINAMATH_CALUDE_milk_storage_theorem_l1535_153556


namespace NUMINAMATH_CALUDE_power_equation_solution_l1535_153514

theorem power_equation_solution (n b : ℝ) : n = 2^(1/4) → n^b = 16 → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1535_153514


namespace NUMINAMATH_CALUDE_stanley_tire_cost_l1535_153540

/-- The total cost of tires purchased by Stanley -/
def total_cost (num_tires : ℕ) (cost_per_tire : ℚ) : ℚ :=
  num_tires * cost_per_tire

/-- Proof that Stanley's total cost for tires is $240.00 -/
theorem stanley_tire_cost :
  let num_tires : ℕ := 4
  let cost_per_tire : ℚ := 60
  total_cost num_tires cost_per_tire = 240 := by
  sorry

#eval total_cost 4 60

end NUMINAMATH_CALUDE_stanley_tire_cost_l1535_153540


namespace NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l1535_153501

theorem gold_copper_alloy_ratio (gold_density copper_density alloy_density : ℝ) 
  (hg : gold_density = 10)
  (hc : copper_density = 6)
  (ha : alloy_density = 8) :
  ∃ (g c : ℝ), g > 0 ∧ c > 0 ∧ 
    (gold_density * g + copper_density * c) / (g + c) = alloy_density ∧
    g = c := by
  sorry

end NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l1535_153501


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1535_153530

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function

/-- Conditions for the geometric sequence -/
def GeometricSequenceConditions (seq : GeometricSequence) : Prop :=
  seq.a 3 = 3/2 ∧ seq.S 3 = 9/2

/-- The value m forms a geometric sequence with a₃ and S₃ -/
def FormsGeometricSequence (seq : GeometricSequence) (m : ℝ) : Prop :=
  ∃ q : ℝ, seq.a 3 * q = m ∧ m * q = seq.S 3

theorem geometric_sequence_problem (seq : GeometricSequence) 
  (h : GeometricSequenceConditions seq) :
  (∀ m : ℝ, FormsGeometricSequence seq m → m = 3*Real.sqrt 3/2 ∨ m = -3*Real.sqrt 3/2) ∧
  (seq.a 1 = 3/2 ∨ seq.a 1 = 6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1535_153530


namespace NUMINAMATH_CALUDE_ratio_sum_difference_l1535_153511

theorem ratio_sum_difference (a b : ℝ) (h1 : a / b = 3 / 8) (h2 : a + b = 44) : b - a = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_difference_l1535_153511


namespace NUMINAMATH_CALUDE_unique_A_exists_l1535_153546

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def satisfies_conditions (A : ℕ) : Prop :=
  is_single_digit A ∧
  72 % A = 0 ∧
  (354100 + 10 * A + 6) % 4 = 0 ∧
  (354100 + 10 * A + 6) % 9 = 0

theorem unique_A_exists :
  ∃! A, satisfies_conditions A :=
sorry

end NUMINAMATH_CALUDE_unique_A_exists_l1535_153546


namespace NUMINAMATH_CALUDE_visit_neither_country_l1535_153597

theorem visit_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 60 → iceland = 35 → norway = 23 → both = 31 →
  total - (iceland + norway - both) = 33 := by
sorry

end NUMINAMATH_CALUDE_visit_neither_country_l1535_153597


namespace NUMINAMATH_CALUDE_june_net_income_l1535_153545

def daily_milk_production : ℕ := 200
def milk_price : ℚ := 355/100
def monthly_expenses : ℕ := 3000
def days_in_june : ℕ := 30

def daily_income : ℚ := daily_milk_production * milk_price

def total_income : ℚ := daily_income * days_in_june

def net_income : ℚ := total_income - monthly_expenses

theorem june_net_income : net_income = 18300 := by
  sorry

end NUMINAMATH_CALUDE_june_net_income_l1535_153545


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l1535_153528

def f (x : ℝ) : ℝ := |x - 2| + |x - 4| - |2*x - 6|

theorem sum_of_max_min_f : 
  ∃ (max min : ℝ), 
    (∀ x, 2 ≤ x ∧ x ≤ 8 → f x ≤ max) ∧ 
    (∃ x, 2 ≤ x ∧ x ≤ 8 ∧ f x = max) ∧
    (∀ x, 2 ≤ x ∧ x ≤ 8 → min ≤ f x) ∧ 
    (∃ x, 2 ≤ x ∧ x ≤ 8 ∧ f x = min) ∧
    max + min = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l1535_153528


namespace NUMINAMATH_CALUDE_least_positive_y_l1535_153534

theorem least_positive_y (x y : ℤ) : 
  (∃ (k : ℤ), 0 < 24 * x + k * y ∧ ∀ (m : ℤ), 0 < 24 * x + m * y → 24 * x + k * y ≤ 24 * x + m * y) ∧
  (∀ (n : ℤ), 0 < 24 * x + n * y → 4 ≤ 24 * x + n * y) →
  y = 4 ∨ y = -4 :=
sorry

end NUMINAMATH_CALUDE_least_positive_y_l1535_153534


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l1535_153571

def a : ℝ × ℝ := (-2, 1)
def b (x : ℝ) : ℝ × ℝ := (x, x^2 + 1)

theorem perpendicular_vectors_x_equals_one :
  ∀ x : ℝ, (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_equals_one_l1535_153571


namespace NUMINAMATH_CALUDE_simplest_radical_among_options_l1535_153561

def is_simplest_radical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ 
  (∀ m : ℕ, m ^ 2 ≤ n → m ^ 2 = n ∨ m ^ 2 < n) ∧
  (∀ a b : ℕ, n = a * b → (a = 1 ∨ b = 1 ∨ ¬ ∃ k : ℕ, k ^ 2 = a))

theorem simplest_radical_among_options :
  is_simplest_radical (Real.sqrt 7) ∧
  ¬ is_simplest_radical (Real.sqrt 9) ∧
  ¬ is_simplest_radical (Real.sqrt 20) ∧
  ¬ is_simplest_radical (Real.sqrt (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_simplest_radical_among_options_l1535_153561


namespace NUMINAMATH_CALUDE_pete_backward_speed_calculation_l1535_153572

/-- Pete's backward walking speed in miles per hour -/
def pete_backward_speed : ℝ := 12

/-- Susan's forward walking speed in miles per hour -/
def susan_forward_speed : ℝ := 4

/-- Tracy's cartwheel speed in miles per hour -/
def tracy_cartwheel_speed : ℝ := 8

/-- Pete's hand-walking speed in miles per hour -/
def pete_hand_speed : ℝ := 2

theorem pete_backward_speed_calculation :
  (pete_backward_speed = 3 * susan_forward_speed) ∧
  (tracy_cartwheel_speed = 2 * susan_forward_speed) ∧
  (pete_hand_speed = (1/4) * tracy_cartwheel_speed) ∧
  (pete_hand_speed = 2) →
  pete_backward_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_pete_backward_speed_calculation_l1535_153572


namespace NUMINAMATH_CALUDE_current_rate_calculation_l1535_153536

/-- Given a boat with speed in still water and distance travelled downstream in a specific time,
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (time_minutes : ℝ) :
  boat_speed = 24 ∧ downstream_distance = 6.75 ∧ time_minutes = 15 →
  ∃ (current_rate : ℝ), current_rate = 3 ∧
    boat_speed + current_rate = downstream_distance / (time_minutes / 60) :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l1535_153536


namespace NUMINAMATH_CALUDE_sector_arc_length_l1535_153594

theorem sector_arc_length (θ : Real) (A : Real) (l : Real) : 
  θ = 120 * π / 180 →  -- Convert 120° to radians
  A = π →              -- Area of the sector
  l = 2 * Real.sqrt 3 * π / 3 → 
  l = θ * Real.sqrt (2 * A / θ) := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1535_153594


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_l1535_153564

theorem negative_integer_square_plus_self (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_l1535_153564


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1535_153583

theorem simplify_sqrt_expression : 
  (Real.sqrt 448 / Real.sqrt 32) - (Real.sqrt 245 / Real.sqrt 49) = Real.sqrt 2 * Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1535_153583


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1535_153590

/-- The line on which the reflection occurs -/
def reflection_line (x y : ℝ) : Prop := 8 * x + 6 * y = 25

/-- The point through which the reflected ray passes -/
def reflection_point : ℝ × ℝ := (-4, 3)

/-- The origin point from which the incident ray starts -/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating that the reflected ray has the equation y = 3 -/
theorem reflected_ray_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (∃ (t : ℝ), reflection_line ((1 - t) * origin.1 + t * x) ((1 - t) * origin.2 + t * y)) →
    (∃ (s : ℝ), x = (1 - s) * reflection_point.1 + s * m ∧ 
                y = (1 - s) * reflection_point.2 + s * 3) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1535_153590


namespace NUMINAMATH_CALUDE_contractor_absence_l1535_153585

/-- A contractor's work problem -/
theorem contractor_absence (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) :
  total_days = 30 ∧ 
  daily_wage = 25 ∧ 
  daily_fine = (15/2) ∧ 
  total_amount = 425 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_wage * days_worked - daily_fine * days_absent = total_amount ∧
    days_absent = 10 := by
  sorry

end NUMINAMATH_CALUDE_contractor_absence_l1535_153585


namespace NUMINAMATH_CALUDE_extremum_sum_l1535_153520

theorem extremum_sum (a b : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x^3 + a*x^2 + b*x + a^2) ∧ 
   (f 1 = 10) ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ 10)) →
  a + b = -7 := by
sorry

end NUMINAMATH_CALUDE_extremum_sum_l1535_153520


namespace NUMINAMATH_CALUDE_range_of_m_for_increasing_function_l1535_153549

-- Define an increasing function on an open interval
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → f x < f y

-- State the theorem
theorem range_of_m_for_increasing_function 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : IncreasingFunction f) 
  (h_inequality : f (m - 1) < f (1 - 2*m)) :
  m ∈ Set.Ioo (-1/2) (2/3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_increasing_function_l1535_153549


namespace NUMINAMATH_CALUDE_ellipse_condition_l1535_153505

/-- An equation represents an ellipse if it's of the form (x^2)/a + (y^2)/b = 1,
    where a and b are positive real numbers and a ≠ b. -/
def IsEllipse (m : ℝ) : Prop :=
  m > 0 ∧ 2*m - 1 > 0 ∧ m ≠ 2*m - 1

/-- If the equation (x^2)/m + (y^2)/(2m-1) = 1 represents an ellipse,
    then m > 1/2 and m ≠ 1. -/
theorem ellipse_condition (m : ℝ) :
  IsEllipse m → m > 1/2 ∧ m ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_condition_l1535_153505


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_relation_l1535_153512

/-- A hyperbola with equation x^2/a + y^2/9 = 1 and asymptotes 3x ± 2y = 0 has a = -4 -/
theorem hyperbola_asymptote_relation (a : ℝ) :
  (∀ x y : ℝ, x^2/a + y^2/9 = 1 ↔ (3*x - 2*y = 0 ∨ 3*x + 2*y = 0)) →
  a = -4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_relation_l1535_153512


namespace NUMINAMATH_CALUDE_smallest_N_with_g_geq_10_N_mod_1000_l1535_153560

/-- Sum of digits in base b representation of n -/
def digitSum (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-5 representation of n -/
def f (n : ℕ) : ℕ := digitSum n 5

/-- g(n) is the sum of digits in base-7 representation of f(n) -/
def g (n : ℕ) : ℕ := digitSum (f n) 7

theorem smallest_N_with_g_geq_10 :
  ∃ N : ℕ, (∀ k < N, g k < 10) ∧ g N ≥ 10 ∧ N = 610 := by sorry

theorem N_mod_1000 :
  ∃ N : ℕ, (∀ k < N, g k < 10) ∧ g N ≥ 10 ∧ N ≡ 610 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_smallest_N_with_g_geq_10_N_mod_1000_l1535_153560


namespace NUMINAMATH_CALUDE_car_speed_l1535_153513

/-- Calculates the speed of a car given distance and time -/
theorem car_speed (distance : ℝ) (time : ℝ) (h1 : distance = 624) (h2 : time = 2 + 2/5) :
  distance / time = 260 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_l1535_153513


namespace NUMINAMATH_CALUDE_banana_arrangements_l1535_153522

def word_length : ℕ := 6
def occurrences : List ℕ := [1, 2, 3]

theorem banana_arrangements :
  (word_length.factorial) / (occurrences.prod) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1535_153522


namespace NUMINAMATH_CALUDE_symmetry_point_xoz_l1535_153538

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D := {p : Point3D | p.y = 0}

/-- Symmetry with respect to the xoz plane -/
def symmetry_xoz (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetry_point_xoz :
  let p : Point3D := ⟨1, 2, 3⟩
  symmetry_xoz p = ⟨1, -2, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_xoz_l1535_153538


namespace NUMINAMATH_CALUDE_max_value_of_one_minus_cos_l1535_153569

open Real

theorem max_value_of_one_minus_cos (x : ℝ) :
  ∃ (k : ℤ), (∀ y : ℝ, 1 - cos y ≤ 1 - cos (π + 2 * π * ↑k)) ∧
              (1 - cos x = 1 - cos (π + 2 * π * ↑k) ↔ ∃ m : ℤ, x = π + 2 * π * ↑m) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_one_minus_cos_l1535_153569


namespace NUMINAMATH_CALUDE_min_committee_size_l1535_153557

/-- Represents a committee with the given properties -/
structure Committee where
  meetings : Nat
  attendees_per_meeting : Nat
  total_members : Nat
  attendance : Fin meetings → Finset (Fin total_members)
  ten_per_meeting : ∀ m, (attendance m).card = attendees_per_meeting
  at_most_once : ∀ i j m₁ m₂, i ≠ j → m₁ ≠ m₂ → 
    (i ∈ attendance m₁ ∧ i ∈ attendance m₂) → 
    (j ∉ attendance m₁ ∨ j ∉ attendance m₂)

/-- The main theorem stating the minimum number of members -/
theorem min_committee_size :
  ∀ c : Committee, c.meetings = 12 → c.attendees_per_meeting = 10 → c.total_members ≥ 58 :=
by sorry

end NUMINAMATH_CALUDE_min_committee_size_l1535_153557


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1535_153507

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1535_153507


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1535_153543

theorem correct_average_after_error_correction (n : ℕ) (incorrect_sum correct_sum : ℝ) :
  n = 10 →
  incorrect_sum = 46 * n →
  correct_sum = incorrect_sum + 50 →
  correct_sum / n = 51 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1535_153543


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1535_153568

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (2/3 : ℂ) + (5/8 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (2/3 : ℂ) - (5/8 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1535_153568


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l1535_153592

def total_balls : ℕ := 9
def white_balls : ℕ := 5
def black_balls : ℕ := 4

def alternating_sequence_probability : ℚ :=
  1 / (total_balls.choose black_balls)

theorem alternating_draw_probability :
  alternating_sequence_probability = 1 / 126 :=
by sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l1535_153592


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l1535_153598

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_cost : ℚ := 12
  let mushroom_cost : ℚ := 3
  let pepperoni_cost : ℚ := 5
  let bob_slices : ℕ := 8
  let anne_slices : ℕ := 3
  let total_cost : ℚ := plain_cost + mushroom_cost + pepperoni_cost
  let bob_cost : ℚ := total_cost - (anne_slices : ℚ) * (plain_cost / total_slices)
  let anne_cost : ℚ := (anne_slices : ℚ) * (plain_cost / total_slices)
  bob_cost - anne_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l1535_153598


namespace NUMINAMATH_CALUDE_parabola_through_point_l1535_153521

theorem parabola_through_point (x y : ℝ) :
  (x = 1 ∧ y = 2) →
  (y^2 = 4*x ∨ x^2 = (1/2)*y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_point_l1535_153521


namespace NUMINAMATH_CALUDE_external_diagonals_inequality_five_seven_ten_not_valid_l1535_153574

/-- External diagonals of a right regular prism -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_b : a ≤ b
  b_le_c : b ≤ c
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- Theorem: For valid external diagonals of a right regular prism, a² + b² > c² -/
theorem external_diagonals_inequality (d : ExternalDiagonals) : d.a^2 + d.b^2 > d.c^2 := by
  sorry

/-- The set {5, 7, 10} cannot be the lengths of external diagonals of a right regular prism -/
theorem five_seven_ten_not_valid : ¬∃ (d : ExternalDiagonals), d.a = 5 ∧ d.b = 7 ∧ d.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_external_diagonals_inequality_five_seven_ten_not_valid_l1535_153574


namespace NUMINAMATH_CALUDE_age_puzzle_solution_l1535_153580

theorem age_puzzle_solution :
  ∃! x : ℕ, 6 * (x + 6) - 6 * (x - 6) = x ∧ x = 72 :=
by sorry

end NUMINAMATH_CALUDE_age_puzzle_solution_l1535_153580


namespace NUMINAMATH_CALUDE_small_prism_surface_area_l1535_153544

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (d : Dimensions) : ℕ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Theorem: Surface area of small prism in arrangement of 9 identical prisms -/
theorem small_prism_surface_area 
  (small : Dimensions) 
  (large_surface_area : ℕ) 
  (h1 : large_surface_area = 360) 
  (h2 : 3 * small.width = 2 * small.length) 
  (h3 : small.length = 3 * small.height) 
  (h4 : surfaceArea { length := 3 * small.width, 
                      width := 3 * small.width, 
                      height := small.length + small.height } = large_surface_area) : 
  surfaceArea small = 88 := by
sorry

end NUMINAMATH_CALUDE_small_prism_surface_area_l1535_153544


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1535_153518

/-- 
Given a geometric sequence with:
- First term a = 5
- Common ratio r = 2
- Number of terms n = 5

Prove that the sum of this sequence is 155.
-/
theorem geometric_sequence_sum : 
  let a : ℕ := 5  -- first term
  let r : ℕ := 2  -- common ratio
  let n : ℕ := 5  -- number of terms
  (a * (r^n - 1)) / (r - 1) = 155 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1535_153518


namespace NUMINAMATH_CALUDE_solution_difference_l1535_153542

theorem solution_difference (p q : ℝ) : 
  ((p - 5) * (p + 5) = 17 * p - 85) →
  ((q - 5) * (q + 5) = 17 * q - 85) →
  p ≠ q →
  p > q →
  p - q = 7 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1535_153542


namespace NUMINAMATH_CALUDE_smallest_square_area_l1535_153573

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The smallest square that can contain two rectangles without overlap -/
def smallest_containing_square (r1 r2 : Rectangle) : ℕ :=
  (min r1.width r1.height + min r2.width r2.height) ^ 2

/-- Theorem stating the smallest possible area of the square -/
theorem smallest_square_area (r1 r2 : Rectangle) 
  (h1 : r1 = ⟨2, 3⟩) 
  (h2 : r2 = ⟨3, 4⟩) : 
  smallest_containing_square r1 r2 = 25 := by
  sorry

#eval smallest_containing_square ⟨2, 3⟩ ⟨3, 4⟩

end NUMINAMATH_CALUDE_smallest_square_area_l1535_153573


namespace NUMINAMATH_CALUDE_ice_skating_given_skiing_l1535_153558

-- Define the probability space
variable (Ω : Type) [MeasurableSpace Ω] [Fintype Ω] (P : Measure Ω)

-- Define events
variable (A B : Set Ω)

-- Define the probabilities
variable (hA : P A = 0.6)
variable (hB : P B = 0.5)
variable (hAorB : P (A ∪ B) = 0.7)

-- Define the theorem
theorem ice_skating_given_skiing :
  P (A ∩ B) / P B = 0.8 :=
sorry

end NUMINAMATH_CALUDE_ice_skating_given_skiing_l1535_153558


namespace NUMINAMATH_CALUDE_prob_both_selected_l1535_153584

/-- The probability of both Ram and Ravi being selected in an exam -/
theorem prob_both_selected (prob_ram prob_ravi : ℚ) 
  (h_ram : prob_ram = 3/7)
  (h_ravi : prob_ravi = 1/5) :
  prob_ram * prob_ravi = 3/35 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l1535_153584


namespace NUMINAMATH_CALUDE_bug_meeting_point_l1535_153506

/-- Triangle with side lengths 7, 8, and 9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (RP : ℝ)
  (h_PQ : PQ = 7)
  (h_QR : QR = 8)
  (h_RP : RP = 9)

/-- The meeting point of two bugs crawling from P in opposite directions -/
def meetingPoint (t : Triangle) : ℝ := sorry

/-- Theorem stating that QS = 5 in the given triangle -/
theorem bug_meeting_point (t : Triangle) : meetingPoint t = 5 := by sorry

end NUMINAMATH_CALUDE_bug_meeting_point_l1535_153506


namespace NUMINAMATH_CALUDE_exam_score_standard_deviations_l1535_153535

/-- Given an exam with mean score and standard deviation, prove the number of standard deviations above the mean for a specific score -/
theorem exam_score_standard_deviations (mean sd : ℝ) (x : ℝ) 
  (h1 : mean - 2 * sd = 58)
  (h2 : mean = 74)
  (h3 : mean + x * sd = 98) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_exam_score_standard_deviations_l1535_153535


namespace NUMINAMATH_CALUDE_exact_two_fours_probability_l1535_153577

-- Define the number of dice
def num_dice : ℕ := 15

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the target number we're looking for
def target_number : ℕ := 4

-- Define the number of dice we want to show the target number
def target_count : ℕ := 2

-- Define the probability of rolling the target number on a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of not rolling the target number on a single die
def single_prob_complement : ℚ := 1 - single_prob

-- Theorem statement
theorem exact_two_fours_probability :
  (Nat.choose num_dice target_count : ℚ) * single_prob ^ target_count * single_prob_complement ^ (num_dice - target_count) =
  (105 : ℚ) * 5^13 / 6^15 := by
  sorry

end NUMINAMATH_CALUDE_exact_two_fours_probability_l1535_153577


namespace NUMINAMATH_CALUDE_function_properties_l1535_153510

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * log x

theorem function_properties (a : ℝ) :
  (∀ x > 0, Monotone (f a)) → a = 1 ∧
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x₀ = -2 ∧ f a x ≥ f a x₀) → a = Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1535_153510


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l1535_153552

theorem largest_number_in_ratio (a b c d : ℚ) : 
  a / b = -3/2 →
  b / c = 4/5 →
  c / d = -2/3 →
  a + b + c + d = 1344 →
  max a (max b (max c d)) = 40320 := by
sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l1535_153552


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1535_153550

theorem complex_equation_solution (z : ℂ) :
  (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I →
  z = 3/5 + 12/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1535_153550


namespace NUMINAMATH_CALUDE_grid_separation_impossible_l1535_153523

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 8
  y : Fin 8

/-- Represents a line on the grid -/
structure GridLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line passes through a point -/
def line_passes_through (l : GridLine) (p : GridPoint) : Prop :=
  l.a * p.x.val + l.b * p.y.val + l.c = 0

/-- Checks if two points are separated by a line -/
def points_separated_by_line (l : GridLine) (p1 p2 : GridPoint) : Prop :=
  (l.a * p1.x.val + l.b * p1.y.val + l.c) * (l.a * p2.x.val + l.b * p2.y.val + l.c) < 0

/-- The main theorem stating the impossibility of the grid separation -/
theorem grid_separation_impossible :
  ¬ ∃ (lines : Fin 13 → GridLine),
    (∀ (l : Fin 13) (p : GridPoint), ¬ line_passes_through (lines l) p) ∧
    (∀ (p1 p2 : GridPoint), p1 ≠ p2 → ∃ (l : Fin 13), points_separated_by_line (lines l) p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_grid_separation_impossible_l1535_153523
