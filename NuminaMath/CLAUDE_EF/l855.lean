import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_power_sum_l855_85551

theorem last_digit_power_sum (x : ℝ) (h1 : x ≠ 0) (h2 : x + x⁻¹ = 3) :
  ∀ n : ℕ, n > 0 → (x^(2^n) + (x⁻¹)^(2^n)) % 10 = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_power_sum_l855_85551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_l855_85505

-- Define the slopes of two lines
noncomputable def slope1 : ℝ := -1
noncomputable def slope2 (a : ℝ) : ℝ := 1 / a

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines_condition (a : ℝ) :
  (a = 1) ↔ are_perpendicular slope1 (slope2 a) := by
  sorry

#check perpendicular_lines_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_condition_l855_85505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_m_factors_imply_a_and_b_factorization_result_l855_85527

-- Define the polynomials
def p (m : ℤ) (x : ℤ) : ℤ := x^2 + m*x + 8
def q (a b : ℤ) (x : ℤ) : ℤ := x^3 + a*x^2 - 5*x + b

-- Theorem 1
theorem factor_implies_m (m : ℤ) : 
  (∀ x, p m x = (x - 4) * (x + m + 2)) → m = -6 := by sorry

-- Theorem 2
theorem factors_imply_a_and_b (a b : ℤ) :
  (∀ x, q a b x = (x - 1) * (x + 2) * (x + a + 3)) → a = -2 ∧ b = 6 := by sorry

-- Theorem 3
theorem factorization_result (a b : ℤ) :
  (∀ x, q a b x = (x - 1) * (x + 2) * (x + a + 3)) →
  (a = -2 ∧ b = 6) →
  (∀ x, q a b x = (x - 1) * (x + 2) * (x - 3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_implies_m_factors_imply_a_and_b_factorization_result_l855_85527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_can_escape_l855_85514

/-- Represents the square arena --/
structure Square where
  side_length : ℝ
  deriving Inhabited

/-- Represents the rabbit --/
structure Rabbit where
  position : ℝ × ℝ
  max_speed : ℝ
  deriving Inhabited

/-- Represents a wolf --/
structure Wolf where
  position : ℝ × ℝ
  max_speed : ℝ
  deriving Inhabited

/-- The setup of the game --/
structure GameSetup where
  square : Square
  rabbit : Rabbit
  wolves : Fin 4 → Wolf

/-- Predicate to check if a position is on the edge of the square --/
def isOnEdge (s : Square) (pos : ℝ × ℝ) : Prop :=
  pos.1 = 0 ∨ pos.1 = s.side_length ∨ pos.2 = 0 ∨ pos.2 = s.side_length

/-- Initial game setup --/
def initialSetup : GameSetup where
  square := { side_length := 1 }
  rabbit := { position := (0.5, 0.5), max_speed := 1 }
  wolves := fun i => { 
    position := match i with
      | 0 => (0, 0)
      | 1 => (1, 0)
      | 2 => (1, 1)
      | 3 => (0, 1)
    max_speed := 1.4
  }

/-- Predicate to check if the rabbit can escape --/
def canEscape (setup : GameSetup) : Prop :=
  ∃ (strategy : ℕ → ℝ × ℝ),
    (∀ n, ‖strategy n - strategy (n+1)‖ ≤ setup.rabbit.max_speed) ∧
    (∃ m, isOnEdge setup.square (strategy m)) ∧
    (∀ n (wolf_strategy : Fin 4 → ℕ → ℝ × ℝ),
      (∀ i k, k ≤ n → isOnEdge setup.square (wolf_strategy i k)) →
      (∀ i k, ‖wolf_strategy i k - wolf_strategy i (k+1)‖ ≤ (setup.wolves i).max_speed) →
      ∀ i, ‖strategy n - wolf_strategy i n‖ > 0)

/-- Theorem stating that the rabbit can escape --/
theorem rabbit_can_escape : canEscape initialSetup := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_can_escape_l855_85514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_C_l855_85571

/-- Curve C in Cartesian coordinates -/
def C (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Curve C' after transformation -/
def C' (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The function to be minimized on C' -/
noncomputable def f (x y : ℝ) : ℝ := x + 2 * Real.sqrt 3 * y

theorem min_value_on_C' :
  ∃ (min : ℝ), min = -4 ∧
  (∀ x y : ℝ, C' x y → f x y ≥ min) ∧
  (∃ x y : ℝ, C' x y ∧ f x y = min) := by
  sorry

#check min_value_on_C'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_C_l855_85571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_l855_85579

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions of the problem
noncomputable def problemTriangle : Triangle where
  a := 1
  angleB := Real.pi / 4  -- 45° in radians
  b := sorry       -- to be proven
  c := sorry       -- to be proven
  angleA := sorry  -- not given
  angleC := sorry  -- not given

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := 
  (1 / 2) * t.a * t.c * Real.sin t.angleB

-- Theorem statement
theorem solution (t : Triangle) (h1 : t = problemTriangle) (h2 : area t = 2) :
  t.c = 4 * Real.sqrt 2 ∧ t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_l855_85579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_l855_85500

-- Define the curve C
def curve_C (a : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = -2 * x + 4

-- Define the distance from a point to the origin
noncomputable def distance_to_origin (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem curve_C_equation (a : ℝ) (M N : ℝ × ℝ) :
  a ≠ 0 →
  curve_C a M.1 M.2 →
  curve_C a N.1 N.2 →
  line_l M.1 M.2 →
  line_l N.1 N.2 →
  M ≠ N →
  distance_to_origin M.1 M.2 = distance_to_origin N.1 N.2 →
  ∀ (x y : ℝ), curve_C a x y ↔ x^2 + y^2 - 4*x - 2*y = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_equation_l855_85500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_bound_a_1_pos_b_1_pos_l855_85581

noncomputable section

/-- The function f(x) = (16x + 7) / (4x + 4) -/
def f (x : ℝ) : ℝ := (16 * x + 7) / (4 * x + 4)

/-- Sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Arbitrary positive value for a₀
  | n + 1 => f (a n)

/-- Sequence b_n defined recursively -/
noncomputable def b : ℕ → ℝ
  | 0 => 2  -- Arbitrary positive value for b₀
  | n + 1 => f (b n)

/-- Main theorem: |b_n - a_n| ≤ (1/8^(n-3)) * |b₁ - a₁| for n ≥ 1 -/
theorem sequence_difference_bound (n : ℕ) (h : n ≥ 1) :
  |b n - a n| ≤ (1 / 8^(n - 3)) * |b 1 - a 1| := by
  sorry

/-- Initial conditions: a₁ > 0 and b₁ > 0 -/
theorem a_1_pos : a 1 > 0 := by
  sorry

theorem b_1_pos : b 1 > 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_difference_bound_a_1_pos_b_1_pos_l855_85581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_side_c_value_l855_85578

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.C + t.b * Real.cos t.C + t.c * Real.cos t.B = 0

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : Real :=
  (1/2) * t.a * t.b * Real.sin t.C

-- Theorem 1
theorem angle_C_value (t : Triangle) (h : triangle_condition t) : 
  t.C = 2 * Real.pi / 3 := by sorry

-- Theorem 2
theorem side_c_value (t : Triangle) (h1 : t.a = 2) (h2 : triangle_area t = Real.sqrt 3 / 2) 
  (h3 : t.C = 2 * Real.pi / 3) : t.c = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_value_side_c_value_l855_85578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l855_85589

theorem solution_difference : 
  ∃ (r₁ r₂ : ℝ), 
    ((r₁^2 - 3*r₁ - 17) / (r₁ + 4) = 2*r₁ + 7) ∧
    ((r₂^2 - 3*r₂ - 17) / (r₂ + 4) = 2*r₂ + 7) ∧
    |r₁ - r₂| = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l855_85589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l855_85595

theorem least_number_with_remainder (n : ℕ) : 
  (∀ d ∈ ({5, 6, 9, 12} : Set ℕ), n % d = 4) →
  (∀ m < n, ∃ d ∈ ({5, 6, 9, 12} : Set ℕ), m % d ≠ 4) →
  n = 184 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l855_85595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_is_280_l855_85501

/-- Represents the details of a floor in the apartment building -/
structure Floor :=
  (rooms : ℕ)
  (rent : ℕ)
  (occupancy : ℚ)

/-- Calculates the monthly rent for a floor -/
def floorRent (f : Floor) : ℕ :=
  (((f.rooms : ℚ) * f.occupancy).floor.toNat) * f.rent

/-- Represents Krystiana's apartment building -/
def apartmentBuilding : List Floor :=
  [{ rooms := 5, rent := 15, occupancy := 4/5 },
   { rooms := 6, rent := 25, occupancy := 3/4 },
   { rooms := 9, rent := 30, occupancy := 1/2 }]

/-- The total monthly rent from all occupied rooms in the apartment building -/
def totalMonthlyRent : ℕ :=
  (apartmentBuilding.map floorRent).sum

theorem total_rent_is_280 : totalMonthlyRent = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_is_280_l855_85501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_hydroxide_formation_l855_85508

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction NaH + H₂O → NaOH + H₂ -/
structure Reaction where
  naH : Moles
  h2O : Moles
  naOH : Moles
  h2 : Moles

/-- The reaction is balanced when the number of moles of reactants equals the number of moles of products -/
def is_balanced (r : Reaction) : Prop :=
  r.naH = r.naOH ∧ r.h2O = r.naOH

/-- The theorem states that when 1 mole of Water is used and 1 mole of Sodium hydroxide is produced,
    the number of moles of Sodium hydride combined equals the number of moles of Sodium hydroxide formed -/
theorem sodium_hydroxide_formation (r : Reaction) 
  (h1 : r.h2O = (1 : ℝ))
  (h2 : r.naOH = (1 : ℝ))
  (h3 : is_balanced r) :
  r.naH = r.naOH := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_hydroxide_formation_l855_85508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_space_shuttle_speed_l855_85540

/-- The speed of a space shuttle orbiting the Earth in kilometers per hour. -/
noncomputable def speed_km_per_hour : ℚ := 21600

/-- The number of seconds in an hour. -/
noncomputable def seconds_per_hour : ℚ := 3600

/-- The speed of the space shuttle in kilometers per second. -/
noncomputable def speed_km_per_second : ℚ := speed_km_per_hour / seconds_per_hour

/-- Theorem stating that the speed of the space shuttle is 6 kilometers per second. -/
theorem space_shuttle_speed : speed_km_per_second = 6 := by
  -- Unfold the definitions
  unfold speed_km_per_second speed_km_per_hour seconds_per_hour
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_space_shuttle_speed_l855_85540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_surface_area_l855_85584

/-- The surface area of a rectangular parallelepiped given its face diagonals -/
noncomputable def surface_area (a b c : ℝ) : ℝ :=
  (a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) ^ (1/2 : ℝ) +
  (a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) ^ (1/2 : ℝ) +
  (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2) ^ (1/2 : ℝ)

/-- Theorem: The surface area of a rectangular parallelepiped with face diagonals a, b, and c
    is equal to the sum of the square roots of the products of pairs of expressions
    involving a, b, and c. -/
theorem rectangular_parallelepiped_surface_area (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    a^2 = x^2 + y^2 ∧
    b^2 = x^2 + z^2 ∧
    c^2 = y^2 + z^2 ∧
    2 * (x*y + x*z + y*z) = surface_area a b c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_surface_area_l855_85584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_endpoint_l855_85556

/-- Predicate to check if a line segment is a diameter of a circle -/
def is_diameter (center : ℝ × ℝ) (point1 : ℝ × ℝ) (point2 : ℝ × ℝ) : Prop :=
  center.1 = (point1.1 + point2.1) / 2 ∧ 
  center.2 = (point1.2 + point2.2) / 2

/-- Given a circle with center (2,3) and one endpoint of its diameter at (-1,-1),
    the other endpoint of the diameter is at (5,7). -/
theorem circle_diameter_endpoint (O A B : ℝ × ℝ) : 
  O = (2, 3) → A = (-1, -1) → is_diameter O A B → B = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_endpoint_l855_85556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_plane_l855_85567

-- Define the types for planes and lines
variable (α β : Set (Fin 3 → ℝ)) -- Planes are represented as sets in ℝ³
variable (m : Set (Fin 3 → ℝ)) -- Line is represented as a set in ℝ³

-- Define the relations
def parallel (s t : Set (Fin 3 → ℝ)) : Prop := sorry
def perpendicular (s t : Set (Fin 3 → ℝ)) : Prop := sorry

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (h1 : m ⊆ α) (h2 : parallel α β) : parallel m β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_plane_l855_85567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_problem_l855_85519

/-- Calculates the banker's gain given the banker's discount, interest rate, and time period. -/
noncomputable def bankers_gain (bankers_discount : ℝ) (interest_rate : ℝ) (time_period : ℝ) : ℝ :=
  (bankers_discount * interest_rate * time_period) / (100 + (interest_rate * time_period))

/-- Theorem stating that under the given conditions, the banker's gain is 900. -/
theorem bankers_gain_problem : 
  let bankers_discount : ℝ := 2150
  let interest_rate : ℝ := 12
  let time_period : ℝ := 6
  bankers_gain bankers_discount interest_rate time_period = 900 := by
  -- Unfold the definition of bankers_gain
  unfold bankers_gain
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_problem_l855_85519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_magnitude_magnitude_amplitude_ratio_l855_85568

-- Define the Richter magnitude formula
noncomputable def richter_magnitude (A : ℝ) (A_0 : ℝ) : ℝ := Real.log A / Real.log 10 - Real.log A_0 / Real.log 10

-- Theorem for the first part of the problem
theorem earthquake_magnitude :
  richter_magnitude 1000 0.001 = 6 := by sorry

-- Theorem for the second part of the problem
theorem magnitude_amplitude_ratio :
  ∃ (A_9 A_5 : ℝ), 
    richter_magnitude A_9 1 = 9 ∧ 
    richter_magnitude A_5 1 = 5 ∧ 
    A_9 / A_5 = 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_earthquake_magnitude_magnitude_amplitude_ratio_l855_85568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l855_85593

/-- The number of positive integer divisors of n, including 1 and n -/
def τ (n : ℕ+) : ℕ := sorry

/-- The sum of τ(k) for k from 1 to n -/
def S (n : ℕ+) : ℕ := sorry

/-- The number of positive integers n ≤ 3000 with S(n) odd -/
def c : ℕ := sorry

/-- The number of positive integers n ≤ 3000 with S(n) even -/
def d : ℕ := sorry

theorem divisor_sum_parity_difference :
  |Int.ofNat c - Int.ofNat d| = 138 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sum_parity_difference_l855_85593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_theorem_l855_85550

noncomputable def family_average_age (num_grandparents num_parents num_grandchildren : ℕ)
                       (avg_age_grandparents avg_age_parents avg_age_grandchildren : ℝ) : ℝ :=
  let total_age := (num_grandparents : ℝ) * avg_age_grandparents +
                   (num_parents : ℝ) * avg_age_parents +
                   (num_grandchildren : ℝ) * avg_age_grandchildren
  let total_members := (num_grandparents + num_parents + num_grandchildren : ℝ)
  total_age / total_members

theorem family_age_theorem :
  family_average_age 2 2 3 64 39 6 = 32 := by
  -- Unfold the definition of family_average_age
  unfold family_average_age
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_age_theorem_l855_85550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_exponential_increasing_one_third_exp_decreasing_l855_85574

/-- Definition of an exponential function -/
noncomputable def exponential_function (a : ℝ) (x : ℝ) : ℝ := a^x

/-- Definition of an increasing function -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem stating that not all exponential functions are increasing -/
theorem not_all_exponential_increasing :
  ¬(∀ a : ℝ, a > 0 → a ≠ 1 → is_increasing (exponential_function a)) := by
  sorry

/-- Theorem stating that the specific exponential function (1/3)^x is decreasing -/
theorem one_third_exp_decreasing :
  ¬(is_increasing (exponential_function (1/3))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_exponential_increasing_one_third_exp_decreasing_l855_85574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_difference_l855_85548

theorem sum_divisible_by_difference (n : ℕ) : 
  ∃ (S : Finset ℕ), Finset.card S = n ∧ 
  ∀ a b, a ∈ S → b ∈ S → a ≠ b → (a + b) % (Int.natAbs (a - b)) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_difference_l855_85548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equation_l855_85510

def vector : Type := Fin 2 → ℝ

noncomputable def dot_product (v w : vector) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def norm_squared (v : vector) : ℝ := dot_product v v

noncomputable def projection (v w : vector) : vector :=
  λ i => (dot_product v w / norm_squared w) * w i

theorem projection_equation (c : ℝ) : 
  let v : vector := λ i => if i.val = 0 then 5 else c
  let w : vector := λ i => if i.val = 0 then 3 else 2
  (∀ i, projection v w i = (1 / 13) * w i) → c = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_equation_l855_85510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_consumption_l855_85573

/-- The absorption rate of fiber for koalas -/
noncomputable def koala_absorption_rate : ℝ := 0.3

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
noncomputable def absorbed_fiber : ℝ := 12

/-- The total amount of fiber eaten by the koala in one day (in ounces) -/
noncomputable def total_fiber : ℝ := absorbed_fiber / koala_absorption_rate

theorem koala_fiber_consumption :
  total_fiber = 40 :=
by
  -- Unfold the definitions
  unfold total_fiber absorbed_fiber koala_absorption_rate
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_consumption_l855_85573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l855_85504

/-- Distance between two points in polar coordinates -/
noncomputable def polarDistance (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : ℝ :=
  Real.sqrt ((r1 * Real.cos θ1 - r2 * Real.cos θ2)^2 + (r1 * Real.sin θ1 - r2 * Real.sin θ2)^2)

theorem distance_between_polar_points :
  polarDistance 3 (π/3) 1 (4*π/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l855_85504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_is_45_l855_85566

def remove_digit (n : ℕ) (i : Fin 5) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  (10000 ≤ n ∧ n ≤ 99999) ∧ 
  ∃ (i : Fin 5), remove_digit n i = 7777

def count_valid_numbers : ℕ := sorry

theorem count_valid_numbers_is_45 : 
  count_valid_numbers = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_is_45_l855_85566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l855_85545

/-- Given a power function f(x) = kx^α passing through (1/2, √2/2), prove k + α = 3/2 -/
theorem power_function_sum (k α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = k * x ^ α)
  (h2 : f (1/2) = Real.sqrt 2 / 2) : 
  k + α = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_sum_l855_85545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ABC_l855_85596

noncomputable def partial_fraction_decomposition (x : ℝ) : ℝ :=
  (x^2 - 23) / (x^4 - 3*x^3 - 7*x^2 + 15*x - 10)

noncomputable def decomposed_form (A B C : ℝ) (x : ℝ) : ℝ :=
  A / (x - 1) + B / (x + 2) + C / (x - 2)

theorem product_ABC :
  ∃ A B C : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ -2 ∧ x ≠ 2 →
    partial_fraction_decomposition x = decomposed_form A B C x) →
  A * B * C = 275 / 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ABC_l855_85596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_l855_85546

theorem cube_root_difference : (8 : ℝ) ^ (1/3) - (343 : ℝ) ^ (1/3) = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_difference_l855_85546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_in_pool_b_l855_85525

/-- Represents a rectangular pool with given dimensions -/
structure Pool where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents a valve with a flow rate -/
structure Valve where
  flow_rate : ℝ

noncomputable def pool_volume (p : Pool) : ℝ := p.length * p.width * p.depth

noncomputable def time_to_fill (p : Pool) (v : Valve) : ℝ := pool_volume p / v.flow_rate

theorem water_in_pool_b (pool_a pool_b : Pool) (valve_1 valve_2 : Valve) 
  (h1 : pool_a = pool_b)
  (h2 : pool_a.length = 3 ∧ pool_a.width = 2 ∧ pool_a.depth = 1.2)
  (h3 : time_to_fill pool_a valve_1 = 18)
  (h4 : time_to_fill pool_a valve_2 = 24)
  (h5 : pool_a.depth * (1/3) = 0.4) :
  pool_volume pool_b = 7.2 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_in_pool_b_l855_85525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l855_85529

theorem count_integer_solutions : 
  let S := {(a, b) : ℕ × ℕ | a > 0 ∧ b > 0 ∧ a + b = 40}
  Finset.card (Finset.filter (fun (a, b) => a > 0 ∧ b > 0 ∧ a + b = 40) (Finset.range 41 ×ˢ Finset.range 41)) = 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l855_85529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l855_85538

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ x ∈ S, |7 * x - 5| ≤ 3 * x + 4) ∧ S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l855_85538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l855_85536

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1 - 3 / a

/-- The theorem stating the condition for f to have exactly 3 zeros -/
theorem f_has_three_zeros (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧
    ∀ w : ℝ, f a w = 0 → w = x ∨ w = y ∨ w = z) ↔ 
  (a > -1 ∧ a < 0) ∨ (a > 3 ∧ a < 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l855_85536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_exchange_rate_change_l855_85524

/-- The change in dollar exchange rate from January 1, 2014, to December 31, 2014, 
    rounded to the nearest whole number. -/
theorem dollar_exchange_rate_change : ℤ := by
  let initial_rate : ℚ := 32.6587
  let final_rate : ℚ := 56.2584
  let rate_change : ℚ := final_rate - initial_rate
  have h : (round rate_change : ℤ) = 24 := by sorry
  exact 24

#check dollar_exchange_rate_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_exchange_rate_change_l855_85524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dreamy_vacation_probability_l855_85513

/-- The probability of success in a single trial -/
def p : ℝ := 0.4

/-- The number of trials -/
def n : ℕ := 5

/-- The number of successes we're interested in -/
def k : ℕ := 3

/-- The binomial probability formula -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The theorem stating that the probability of exactly 3 successes in 5 trials
    with a success probability of 0.4 is approximately 0.2304 -/
theorem dreamy_vacation_probability :
  |binomialProbability n k p - 0.2304| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dreamy_vacation_probability_l855_85513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_sinusoid_l855_85565

theorem max_omega_sinusoid (ω : ℝ) (h_pos : ω > 0) : 
  (∀ x ∈ Set.Icc 0 (π / 3), HasDerivAt (λ t => Real.sin (ω * t)) (ω * Real.cos (ω * x)) x) →
  (∀ x, Real.sin (ω * (x + 3 * π)) = Real.sin (ω * (-x + 3 * π))) →
  ω ≤ 4 / 3 ∧ ∃ k : ℤ, ω = k / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_sinusoid_l855_85565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l855_85542

def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

def N : Set ℝ := {x : ℝ | x^2 < 2 ∧ ∃ n : ℤ, x = n}

theorem intersection_M_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l855_85542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l855_85590

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : 0 < φ ∧ φ < Real.pi / 2) 
  (h4 : ∀ (x1 x2 : ℝ), f A ω φ x1 = 0 → f A ω φ x2 = 0 → x1 ≠ x2 → |x1 - x2| = Real.pi / 2) 
  (h5 : f A ω φ (2 * Real.pi / 3) = -2) :
  ∃ (g : ℝ → ℝ), 
    (g = f 2 2 (Real.pi / 6)) ∧ 
    (∀ (x : ℝ), g (x + Real.pi) = g x) ∧
    (∀ (x : ℝ), Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1 ≤ g x ∧ g x ≤ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l855_85590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_rate_proof_l855_85537

/-- Calculates the commission rate given the total sales, savings percentage, and amount saved. -/
noncomputable def calculate_commission_rate (total_sales : ℝ) (savings_percentage : ℝ) (amount_saved : ℝ) : ℝ :=
  (amount_saved / (savings_percentage * total_sales)) * 100

theorem commission_rate_proof (total_sales savings_percentage amount_saved : ℝ) 
  (h1 : total_sales = 24000)
  (h2 : savings_percentage = 0.4)
  (h3 : amount_saved = 1152) :
  calculate_commission_rate total_sales savings_percentage amount_saved = 12 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_rate_proof_l855_85537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l855_85503

/-- Given a triangle ABC with side lengths a, b, and c, if a^2 + c^2 = b^2 + √3*a*c, 
    then the measure of angle B is 30°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (h : a^2 + c^2 = b^2 + Real.sqrt 3 * a * c) :
  Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_in_special_triangle_l855_85503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l855_85552

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V)
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = 1)
  (h3 : ‖a + b‖ = Real.sqrt 3) :
  ‖a - b‖ = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l855_85552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_one_increasing_h_implies_c_range_l855_85507

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2) / Real.exp x

noncomputable def tangent_line (x : ℝ) : ℝ := (1 / Real.exp 1) * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := min (f a x) (x - 1/x)

noncomputable def h (a c : ℝ) (x : ℝ) : ℝ := g a x - c * x^2

theorem tangent_line_implies_a_equals_one (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = tangent_line x₀ ∧ 
   (deriv (f a)) x₀ = (deriv tangent_line) x₀) →
  a = 1 :=
by sorry

theorem increasing_h_implies_c_range (a c : ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → h a c x < h a c y) →
  c ≤ -1 / (2 * Real.exp 1 ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_one_increasing_h_implies_c_range_l855_85507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l855_85506

/-- The side length of the square region D -/
def side_length : ℝ := 3

/-- The radius of the circle centered at the origin -/
def circle_radius : ℝ := 2

/-- The area of the square region D -/
def square_area : ℝ := side_length ^ 2

/-- The area of the circle sector within the square -/
noncomputable def sector_area : ℝ := Real.pi * circle_radius ^ 2 / 4

/-- The probability that a randomly selected point in the square region D
    has a distance greater than 2 from the origin -/
noncomputable def probability : ℝ := (square_area - sector_area) / square_area

theorem probability_calculation :
  probability = (9 - Real.pi) / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_calculation_l855_85506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rocky_first_round_knockouts_middleweight_l855_85509

/-- Represents Rocky's boxing career statistics -/
structure BoxingStats where
  total_fights : ℕ
  lightweight_fights : ℕ
  middleweight_fights : ℕ
  heavyweight_fights : ℕ
  lightweight_win_rate : ℚ
  lightweight_ko_rate : ℚ
  middleweight_win_rate : ℚ
  middleweight_ko_rate : ℚ
  heavyweight_win_rate : ℚ
  heavyweight_ko_rate : ℚ
  first_round_ko_rate : ℚ

/-- Calculates the number of first-round knockouts in the Middleweight category -/
def first_round_knockouts_middleweight (stats : BoxingStats) : ℕ :=
  Int.toNat <| Int.floor ((stats.middleweight_fights : ℚ) * stats.middleweight_win_rate * stats.middleweight_ko_rate * stats.first_round_ko_rate)

/-- Theorem stating that Rocky had 8 first-round knockouts in the Middleweight category -/
theorem rocky_first_round_knockouts_middleweight :
  let rocky_stats : BoxingStats := {
    total_fights := 250,
    lightweight_fights := 100,
    middleweight_fights := 100,
    heavyweight_fights := 50,
    lightweight_win_rate := 3/5,
    lightweight_ko_rate := 1/2,
    middleweight_win_rate := 7/10,
    middleweight_ko_rate := 3/5,
    heavyweight_win_rate := 4/5,
    heavyweight_ko_rate := 2/5,
    first_round_ko_rate := 1/5
  }
  first_round_knockouts_middleweight rocky_stats = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rocky_first_round_knockouts_middleweight_l855_85509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l855_85528

def positional_value (place : Nat) : Nat :=
  match place with
  | 0 => 1
  | 1 => 10
  | 2 => 100
  | 3 => 1000
  | _ => 0

def digit_contribution (digit : Nat) (place : Nat) : Nat :=
  digit * digit * positional_value place

def is_valid_number (n : Nat) : Bool :=
  (n ≥ 1000 ∧ n ≤ 999999) ∧ (Nat.digits 10 n).eraseDups = Nat.digits 10 n

noncomputable def S : Nat :=
  (List.range 1000000).filter is_valid_number
    |>.map (λ n => (Nat.digits 10 n).enum.map (λ (i, d) => digit_contribution d i) |>.sum)
    |>.sum

theorem sum_remainder_theorem : S % 1000 = 220 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_theorem_l855_85528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l855_85569

noncomputable section

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the intersection points
def intersectionPoints (A B : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 ∧ circle2 B.1 B.2

-- Define the length of a line segment
noncomputable def length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem common_chord_length (A B : ℝ × ℝ) :
  intersectionPoints A B → length A B = 2 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l855_85569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_2_sqrt_3_l855_85541

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 29 = 0

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 2 * Real.sqrt 3

/-- Theorem stating that the radius of the circle is 2√3 -/
theorem circle_radius_is_2_sqrt_3 :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_2_sqrt_3_l855_85541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_stats_l855_85515

/-- Represents a class of students with their math scores -/
structure MathClass where
  totalStudents : Nat
  group1Average : ℝ
  group2Average : ℝ
  group1StdDev : ℝ
  group2StdDev : ℝ

/-- Calculates the average score of the entire class -/
noncomputable def classAverage (c : MathClass) : ℝ :=
  (c.group1Average + c.group2Average) / 2

/-- Calculates the variance of the entire class -/
noncomputable def classVariance (c : MathClass) : ℝ :=
  (c.group1StdDev ^ 2 + c.group2StdDev ^ 2) / 2 + 
  ((c.group1Average - classAverage c) ^ 2 + (c.group2Average - classAverage c) ^ 2) / 2

/-- Theorem stating the average score and variance of the given class -/
theorem math_class_stats (c : MathClass) 
  (h1 : c.totalStudents = 40)
  (h2 : c.group1Average = 80)
  (h3 : c.group2Average = 90)
  (h4 : c.group1StdDev = 4)
  (h5 : c.group2StdDev = 6) :
  classAverage c = 85 ∧ classVariance c = 51 := by
  sorry

-- Remove the #eval statements as they are not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_stats_l855_85515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_seven_same_denomination_l855_85539

/-- The number of coins the boy has -/
def total_coins : ℕ := 25

/-- The number of different coin denominations -/
def num_denominations : ℕ := 4

/-- The minimum number of coins of the same denomination we want to prove exists -/
def target_count : ℕ := 7

theorem at_least_seven_same_denomination :
  ∃ (d : Fin num_denominations), 
    target_count ≤ (total_coins / num_denominations) + (total_coins % num_denominations) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_seven_same_denomination_l855_85539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l855_85543

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) : ℝ → ℝ → Prop := λ x y ↦
  (m + 3) * x + 2 * y = 5 - 3 * m

def l₂ (m : ℝ) : ℝ → ℝ → Prop := λ x y ↦
  4 * x + (5 + m) * y = 16

-- Define the conditions
def intersect (m : ℝ) : Prop :=
  ∃ x y, l₁ m x y ∧ l₂ m x y

def parallel (m : ℝ) : Prop :=
  m ≠ -5 ∧ (m + 3) / 4 = 2 / (5 + m)

def coincide (m : ℝ) : Prop :=
  ∀ x y, l₁ m x y ↔ l₂ m x y

def perpendicular (m : ℝ) : Prop :=
  m ≠ -5 ∧ (m + 3) / (-2) * (-4) / (m + 5) = -1

-- State the theorem
theorem line_relations :
  (∀ m : ℝ, intersect m ↔ m ≠ -1 ∧ m ≠ -7) ∧
  (∀ m : ℝ, parallel m ↔ m = -7) ∧
  (∀ m : ℝ, coincide m ↔ m = -1) ∧
  (∀ m : ℝ, perpendicular m ↔ m = -11/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_relations_l855_85543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l855_85587

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  b^2 = a*c ∧ a^2 - c^2 = a*c - b*c

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.cos (2*x - Real.pi/6) + Real.sin (2*x)

theorem triangle_and_function_properties
  (a b c : ℝ)
  (h : Triangle a b c) :
  -- Part 1: cos A = 1/2
  Real.cos (Real.arccos (1/2)) = 1/2 ∧
  -- Part 2: Maximum value of f(x) on [0, π/2] is √3
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi/2 ∧ f x = Real.sqrt 3 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.pi/2 → f y ≤ Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l855_85587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_c_is_linear_l855_85531

-- Define the general form of a linear function
def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

-- Define the given functions
noncomputable def f_a : ℝ → ℝ := λ x ↦ 1 / x
noncomputable def f_b : ℝ → ℝ := λ x ↦ 2 * x^2 + 1
noncomputable def f_c : ℝ → ℝ := λ x ↦ 3 - (1/2) * x
noncomputable def f_d : ℝ → ℝ := λ x ↦ Real.sqrt x

-- Theorem stating that only f_c is a linear function
theorem only_f_c_is_linear :
  ¬(is_linear_function f_a) ∧
  ¬(is_linear_function f_b) ∧
  (is_linear_function f_c) ∧
  ¬(is_linear_function f_d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_c_is_linear_l855_85531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_valley_theorem_l855_85549

/-- The number of ways to arrange animals in the Happy Valley Kennel --/
def happy_valley_arrangements : ℕ :=
  let num_chickens : ℕ := 5
  let num_dogs : ℕ := 2
  let num_cats : ℕ := 4
  let total_animals : ℕ := num_chickens + num_dogs + num_cats
  let num_species : ℕ := 3
  (Nat.factorial num_species) * (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

/-- Theorem stating that the number of arrangements is 34560 --/
theorem happy_valley_theorem :
  happy_valley_arrangements = 34560 :=
by
  -- Unfold the definition of happy_valley_arrangements
  unfold happy_valley_arrangements
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_valley_theorem_l855_85549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_equation_certain_number_approx_l855_85530

/-- The certain number that satisfies the equation (228% of 1265) ÷ x = 480.7 -/
noncomputable def certain_number : ℝ := 2884.2 / 480.7

/-- The equation that defines the certain number -/
theorem certain_number_equation : (2.28 * 1265) / certain_number = 480.7 := by sorry

/-- The certain number is approximately 6 -/
theorem certain_number_approx : ∃ (ε : ℝ), ε > 0 ∧ |certain_number - 6| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_equation_certain_number_approx_l855_85530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_friction_coefficient_l855_85591

/-- The angle at which the rod begins to slide (in radians) -/
noncomputable def α : Real := 85 * Real.pi / 180

/-- The ratio of normal force to weight when the rod is vertical -/
def normal_to_weight_ratio : Real := 6

/-- The coefficient of friction between the surface and the rod -/
def μ : Real := 0.08

theorem rod_friction_coefficient :
  ∀ (m g : Real),
  m > 0 → g > 0 →
  ∃ (F N : Real),
  F > 0 ∧ N > 0 ∧
  F * Real.sin α - m * g + N * Real.cos α = 0 ∧
  F * Real.cos α - μ * N - N * Real.sin α = 0 ∧
  N = normal_to_weight_ratio * m * g :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_friction_coefficient_l855_85591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_freddy_speed_ratio_l855_85533

/-- Represents a journey between two cities -/
structure Journey where
  distance : ℝ
  time : ℝ

/-- Calculate the average speed of a journey -/
noncomputable def averageSpeed (j : Journey) : ℝ :=
  j.distance / j.time

theorem eddy_freddy_speed_ratio :
  let eddy : Journey := { distance := 600, time := 3 }
  let freddy : Journey := { distance := 300, time := 3 }
  (averageSpeed eddy) / (averageSpeed freddy) = 2 := by
  -- Unfold the definitions
  unfold averageSpeed
  -- Simplify the expressions
  simp
  -- The proof is now trivial
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eddy_freddy_speed_ratio_l855_85533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l855_85564

-- Define the function f
noncomputable def f (p q : ℝ) (x : ℝ) : ℝ := (p * x^2 + 2) / (q - 3 * x)

-- State the theorem
theorem odd_function_properties (p q : ℝ) :
  (∀ x, x ≠ 0 → f p q x = -f p q (-x)) →  -- f is an odd function
  (f p q 2 = -5/3) →                       -- f(2) = -5/3
  (∃ p' q', ∀ x, x ≠ 0 → f p' q' x = -(2/3) * (x + 1/x)) ∧  -- Explicit formula exists
  (∀ x y, 0 < x → x < y → y < 1 → f p q x < f p q y) :=  -- Monotonically increasing in (0, 1)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l855_85564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_sequence_624_l855_85572

def trapezoidal_sequence : ℕ → ℕ
  | 0 => 5
  | n + 1 => trapezoidal_sequence n + (n + 3)

theorem trapezoidal_sequence_624 : trapezoidal_sequence 624 = 196250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoidal_sequence_624_l855_85572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l855_85532

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.sqrt (5 - x) + Real.log (x + 1)
noncomputable def g (a : ℝ) (x : ℝ) := Real.log (x^2 - 2*x + a)

-- Define the domains A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x + a > 0}

-- Part I
theorem part_I : 
  A ∩ B (-8) = {x | 4 < x ∧ x ≤ 5} := by sorry

-- Part II
theorem part_II : 
  (A ∩ (B (-3))ᶜ = {x | -1 < x ∧ x ≤ 3}) → -3 = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_I_part_II_l855_85532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l855_85580

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5) + (x + 4) ^ (1/3 : ℝ)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l855_85580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l855_85560

/-- Represents a sequence of 2018 natural numbers -/
def Sequence := Fin 2018 → ℕ

/-- The operation of counting occurrences in a sequence -/
def count_occurrences (s : Sequence) : Sequence :=
  λ i => Finset.card (Finset.filter (λ j => s j = s i) Finset.univ)

/-- A sequence is stable if applying count_occurrences doesn't change it -/
def is_stable (s : Sequence) : Prop :=
  count_occurrences s = s

/-- The main theorem stating that repeated application of count_occurrences 
    will eventually result in a stable sequence -/
theorem eventual_stability (initial : Sequence) : 
  ∃ n : ℕ, is_stable ((count_occurrences^[n]) initial) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventual_stability_l855_85560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drilled_cube_surface_area_l855_85597

-- Define the cube
structure Cube where
  sideLength : ℝ
  vertexA : ℝ × ℝ × ℝ

-- Define points on the cube
structure CubePoints (c : Cube) where
  L : ℝ × ℝ × ℝ
  M : ℝ × ℝ × ℝ
  N : ℝ × ℝ × ℝ

-- Define the drilled solid
structure DrilledSolid (c : Cube) (p : CubePoints c) where
  surfaceArea : ℝ

-- Theorem statement
theorem drilled_cube_surface_area 
  (c : Cube) 
  (p : CubePoints c)
  (t : DrilledSolid c p)
  (h1 : c.sideLength = 10)
  (h2 : p.L = (3, 0, 0))
  (h3 : p.M = (3, 0, 3))
  (h4 : p.N = (3, 3, 0))
  : ∃ (x y q : ℕ), 
    t.surfaceArea = x + y * Real.sqrt q ∧
    q > 0 ∧
    (∀ (prime : ℕ), prime.Prime → ¬(q % (prime * prime) = 0)) ∧
    x + y + q = 660 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drilled_cube_surface_area_l855_85597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_sum_between_multiples_l855_85598

theorem smallest_divisible_sum_between_multiples (a b : ℕ) : 
  (0 < a ∧ 0 < b) →
  (∀ x y : ℕ, 0 < x ∧ 0 < y → 21 ≤ lcm x y) →
  21 = lcm a b →
  (Finset.range 5).sum (λ k => lcm a b * (k + 8)) = 630 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_sum_between_multiples_l855_85598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_rate_solution_l855_85521

/-- Represents the hotel stay problem --/
def hotel_stay_problem (initial_amount : ℚ) (night_hours : ℕ) (morning_hours : ℕ) 
  (morning_rate : ℚ) (remaining_amount : ℚ) : Prop :=
  ∃ (night_rate : ℚ),
    night_rate * night_hours + morning_rate * morning_hours = initial_amount - remaining_amount

/-- Theorem stating the solution to the hotel stay problem --/
theorem hotel_rate_solution :
  hotel_stay_problem 80 6 4 2 63 →
  ∃ (night_rate : ℚ), night_rate = 3/2 := by
  sorry

#check hotel_rate_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_rate_solution_l855_85521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l855_85577

/-- The length of the major axis of an ellipse with given properties -/
noncomputable def major_axis_length : ℝ := 2 * Real.sqrt 7

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop :=
  x^2 / (major_axis_length/2)^2 + y^2 / ((major_axis_length/2)^2 - 4) = 1

/-- Definition of the line -/
def is_on_line (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y + 4 = 0

/-- The foci of the ellipse -/
def focus1 : ℝ × ℝ := (-2, 0)
def focus2 : ℝ × ℝ := (2, 0)

/-- Theorem stating the length of the major axis -/
theorem ellipse_major_axis_length :
  ∃! (x y : ℝ), is_ellipse x y ∧ is_on_line x y →
  major_axis_length = 2 * Real.sqrt 7 := by
  sorry

#check ellipse_major_axis_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l855_85577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_special_case_l855_85547

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculates the distance between two points -/
noncomputable def Point.distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of an equilateral triangle given its side length -/
noncomputable def equilateralTriangleArea (sideLength : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * sideLength^2

/-- Main theorem -/
theorem equilateral_triangle_area_special_case 
  (A B C D E : Point) (ℓ : Line) : 
  (∀ X Y, X ≠ Y → Point.distance X Y = Point.distance A B) →  -- ABC is equilateral
  A.onLine ℓ →  -- ℓ passes through A
  (D.onLine ℓ ∧ E.onLine ℓ) →  -- D and E are on ℓ
  (Point.distance B D)^2 + (Point.distance D A)^2 = (Point.distance B A)^2 →  -- D is orthogonal projection of B
  (Point.distance C E)^2 + (Point.distance E A)^2 = (Point.distance C A)^2 →  -- E is orthogonal projection of C
  Point.distance D E = 1 →  -- DE = 1
  2 * Point.distance B D = Point.distance C E →  -- 2BD = CE
  equilateralTriangleArea (Point.distance A B) = 7 * Real.sqrt 3 := by
  sorry  -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_special_case_l855_85547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pricing_strategy_l855_85520

/-- Represents the pricing strategy of a store --/
structure PricingStrategy where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  sale_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount --/
def purchase_price (ps : PricingStrategy) : ℝ :=
  ps.list_price * (1 - ps.purchase_discount)

/-- Calculates the selling price based on the marked price and sale discount --/
def selling_price (ps : PricingStrategy) : ℝ :=
  ps.marked_price * (1 - ps.sale_discount)

/-- Calculates the profit based on the selling price and purchase price --/
def profit (ps : PricingStrategy) : ℝ :=
  selling_price ps - purchase_price ps

/-- Theorem stating the optimal pricing strategy --/
theorem optimal_pricing_strategy (ps : PricingStrategy) 
  (h1 : ps.list_price > 0)
  (h2 : ps.purchase_discount = 0.15)
  (h3 : ps.sale_discount = 0.1)
  (h4 : ps.profit_margin = 0.2)
  (h5 : profit ps = ps.profit_margin * selling_price ps) :
  ∃ (ε : ℝ), abs (ps.marked_price - 1.18 * ps.list_price) < ε := by
  sorry

#check optimal_pricing_strategy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_pricing_strategy_l855_85520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_l855_85523

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y + Real.sqrt 3 * x = Real.sqrt 3 * m

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

-- Define the center of C
def center_C : ℝ × ℝ := (1, 0)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y m : ℝ) : ℝ :=
  |Real.sqrt 3 - m * Real.sqrt 3| / 2

-- Theorem 1: When m = 3, the line is tangent to the curve
theorem line_tangent_to_curve :
  distance_point_to_line (center_C.1) (center_C.2) 3 = Real.sqrt 3 := by
  sorry

-- Theorem 2: Range of m for which there's a point on C at distance √3/2 from l
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, curve_C x y ∧ distance_point_to_line x y m = Real.sqrt 3 / 2) ↔
  -2 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_l855_85523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_chemistry_marks_l855_85570

/-- Given David's marks in different subjects and his average, prove his Chemistry marks --/
theorem davids_chemistry_marks
  (english : ℕ) (mathematics : ℕ) (physics : ℕ) (biology : ℕ) (average : ℚ)
  (h_english : english = 90)
  (h_mathematics : mathematics = 92)
  (h_physics : physics = 85)
  (h_biology : biology = 85)
  (h_average : average = 87.8) :
  ∃ chemistry : ℕ,
    (english + mathematics + physics + chemistry + biology : ℚ) / 5 = average ∧
    chemistry = 87 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_chemistry_marks_l855_85570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_probability_l855_85588

/-- The probability of finding at least one counterfeit coin when selecting one coin from each of 10 boxes -/
noncomputable def P1 : ℝ := 1 - (99/100)^10

/-- The probability of finding at least one counterfeit coin when selecting two coins from each of 5 boxes -/
noncomputable def P2 : ℝ := 1 - (49/50)^5

/-- Theorem stating that P1 is less than P2 -/
theorem counterfeit_coin_probability : P1 < P2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_probability_l855_85588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l855_85558

open Set

theorem complement_of_union (A B : Set ℝ) :
  A = {x : ℝ | 3 ≤ x ∧ x < 7} →
  B = {x : ℝ | 2 < x ∧ x < 10} →
  (Aᶜ ∩ Bᶜ) = Iic 2 ∪ Ioi 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_l855_85558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_commuting_time_proof_l855_85553

noncomputable def regular_walking_time : ℝ := 2
noncomputable def regular_biking_time : ℝ := 1
noncomputable def faster_pace_reduction : ℝ := 0.25
noncomputable def park_route_addition : ℝ := 0.5
noncomputable def busy_road_addition : ℝ := 0.25
noncomputable def rain_increase_factor : ℝ := 1.2
noncomputable def walking_rest_stop : ℝ := 0.1667
noncomputable def biking_rest_stops : ℝ := 0.1667

noncomputable def monday_time : ℝ := 2 * ((regular_walking_time - faster_pace_reduction) * rain_increase_factor + walking_rest_stop)
noncomputable def tuesday_time : ℝ := 2 * (regular_biking_time + biking_rest_stops)
noncomputable def wednesday_time : ℝ := 2 * (regular_walking_time + park_route_addition + walking_rest_stop)
noncomputable def thursday_time : ℝ := 2 * ((regular_walking_time - faster_pace_reduction) * rain_increase_factor + walking_rest_stop)
noncomputable def friday_time : ℝ := 2 * (regular_biking_time + busy_road_addition + biking_rest_stops)

noncomputable def total_commuting_time : ℝ := monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem total_commuting_time_proof : 
  abs (total_commuting_time - 19.566) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_commuting_time_proof_l855_85553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_not_equal_l855_85512

noncomputable section

-- Define the function pairs
def f1_1 (x : ℝ) : ℝ := (x + 3) * (x - 5) / (x + 3)
def f1_2 (x : ℝ) : ℝ := x - 5

noncomputable def f2_1 (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)
noncomputable def f2_2 (x : ℝ) : ℝ := Real.sqrt ((x + 1) * (x - 1))

def f3_1 (x : ℝ) : ℝ := x
noncomputable def f3_2 (x : ℝ) : ℝ := Real.sqrt (x^2)

def f4_1 (x : ℝ) : ℝ := x
def f4_2 (x : ℝ) : ℝ := 3 * x^3

noncomputable def f5_1 (x : ℝ) : ℝ := (Real.sqrt (2 * x - 5))^2
def f5_2 (x : ℝ) : ℝ := 2 * x - 5

end noncomputable section

-- Theorem stating that all function pairs are not equal
theorem all_pairs_not_equal :
  (∃ x, f1_1 x ≠ f1_2 x) ∧
  (∃ x, f2_1 x ≠ f2_2 x) ∧
  (∃ x, f3_1 x ≠ f3_2 x) ∧
  (∃ x, f4_1 x ≠ f4_2 x) ∧
  (∃ x, f5_1 x ≠ f5_2 x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_not_equal_l855_85512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l855_85594

noncomputable def a (x : Real) : Real × Real := (2 * Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def b (x : Real) : Real × Real := (Real.cos x, 2 * Real.cos x)
noncomputable def f (x m : Real) : Real := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 + 2 * m - 1

theorem vector_function_properties :
  ∀ x m : Real,
  x ∈ Set.Icc 0 (Real.pi / 2) →
  (f x m = 4 * Real.sin (2 * x + Real.pi / 6) + 2 * m + 1) ∧
  (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y m ≥ 3 → m = 2) ∧
  ((∀ y ∈ Set.Icc 0 (Real.pi / 2), f y m ≤ 6) → m ≤ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_function_properties_l855_85594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_revolution_count_l855_85518

theorem circle_revolution_count (n : Nat) (hn : n = 90) :
  (Finset.filter (fun s => s > 0 ∧ n % s = 0) (Finset.range n)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_revolution_count_l855_85518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l855_85563

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def isDecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : isEven f)
  (h_decreasing : isDecreasingOn f { x | 0 ≤ x })
  (h_condition : f a ≥ f 3) :
  -3 ≤ a ∧ a ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l855_85563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_l855_85555

/-- The circle C defined by x^2 + (y-1)^2 = 1 -/
def circleC (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

/-- Point A with coordinates (2, 1) -/
def point_A : ℝ × ℝ := (2, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The maximum distance from point A to any point on the circle C is 3 -/
theorem max_distance_to_circle :
  ∃ (max_dist : ℝ),
    (∀ (x y : ℝ), circleC x y → distance point_A (x, y) ≤ max_dist) ∧
    (∃ (x y : ℝ), circleC x y ∧ distance point_A (x, y) = max_dist) ∧
    max_dist = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_l855_85555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l855_85559

noncomputable def f (x : ℝ) := Real.cos x

theorem cosine_function_properties :
  ∃ (α : ℝ),
    α ∈ Set.Icc 0 Real.pi ∧
    f α = 1/3 ∧
    f (α - Real.pi/3) = 1/6 + Real.sqrt 6/3 ∧
    (∀ x, f (2*x) - 2*f x ≥ -3/2) ∧
    ∃ y, f (2*y) - 2*f y = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_properties_l855_85559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_logarithmic_equation_l855_85517

/-- Proves that there are no real solutions to the logarithmic equation -/
theorem no_solution_logarithmic_equation :
  ¬ ∃ x : ℝ, (8 - 10*x - 12*x^2 > 0 ∧ 2*x - 1 > 0) ∧ 
  Real.log (8 - 10*x - 12*x^2) = 3 * Real.log (2*x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_logarithmic_equation_l855_85517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l855_85561

theorem inclination_angle_range (a b : ℝ) (h : a * b < 0) :
  let P : ℝ × ℝ := (0, -1/b)
  let Q : ℝ × ℝ := (1/a, 0)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  let α : ℝ := Real.arctan slope
  π/2 < α ∧ α < π :=
by
  -- Introduce the local definitions
  intro P Q slope α
  
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l855_85561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_expression_l855_85599

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (5 / (1 - 2*Complex.I)) * Complex.I
  (z.im : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_expression_l855_85599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_2s_l855_85592

-- Define the motion equation
def S (t : ℝ) : ℝ := t^2 + 3

-- Define instantaneous velocity as the derivative of S
noncomputable def instantaneous_velocity (t : ℝ) : ℝ := deriv S t

-- Theorem statement
theorem velocity_at_2s :
  instantaneous_velocity 2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_2s_l855_85592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_triangle_area_l855_85585

/-- A parabola with vertex at the origin, focus on the x-axis, and passing through (1,2) -/
structure Parabola where
  -- The parabola passes through (1,2)
  passes_through : ∀ x y, y^2 = 4*x → (x = 1 ∧ y = 2)
  -- The vertex is at the origin
  vertex_at_origin : ∀ x y, y^2 = 4*x → (x = 0 → y = 0)
  -- The focus is on the x-axis
  focus_on_x_axis : ∃ p > 0, ∀ x y, y^2 = 4*p*x

/-- The line y = x - 4 -/
def line (x y : ℝ) : Prop := y = x - 4

/-- Area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem parabola_equation_and_triangle_area (p : Parabola) :
  (∀ x y, y^2 = 4*x) ∧ 
  (∃ A B : ℝ × ℝ, 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A.2^2 = 4*A.1 ∧ 
    B.2^2 = 4*B.1 ∧ 
    area_triangle (0, 0) A B = 16 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_triangle_area_l855_85585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_100_l855_85516

def b : ℕ → ℕ
  | 0 => 15  -- Base case for 0
  | 1 => 15  -- Base case for 1
  | n+2 => if n+2 ≤ 15 then 15 else 50 * b (n+1) + (n+2)^2

theorem least_multiple_of_100 : ∀ n > 15, b n % 100 ≠ 0 → b 16 % 100 = 0 := by
  sorry

#eval b 16 % 100  -- This will evaluate b 16 mod 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_100_l855_85516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biography_increase_percentage_l855_85534

/-- Represents a library branch with its total books and biography percentage --/
structure Branch where
  total_books : ℕ
  biography_percent : ℚ

/-- Calculates the number of biographies in a branch --/
def biography_count (b : Branch) : ℕ :=
  (b.total_books : ℚ) * b.biography_percent |>.floor.toNat

/-- Represents the library system before changes --/
def initial_system : List Branch := [
  { total_books := 8000, biography_percent := 20/100 },
  { total_books := 10000, biography_percent := 25/100 },
  { total_books := 12000, biography_percent := 28/100 }
]

/-- Represents the library system after changes --/
def final_system : List Branch := [
  { total_books := 8000, biography_percent := 32/100 },
  { total_books := 10000, biography_percent := 35/100 },
  { total_books := 12000, biography_percent := 40/100 }
]

/-- Calculates the total number of biographies in a system --/
def total_biographies (system : List Branch) : ℕ :=
  system.map biography_count |>.sum

/-- Theorem stating the percentage increase in biographies --/
theorem biography_increase_percentage :
  let initial_total := total_biographies initial_system
  let final_total := total_biographies final_system
  let increase := final_total - initial_total
  ∃ ε > 0, abs ((increase : ℚ) / initial_total * 100 - 45.58) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biography_increase_percentage_l855_85534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_ratio_l855_85502

/-- An arithmetic sequence with a non-zero common difference where a_1, a_3, and a_4 form a geometric sequence. -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : d ≠ 0 -- Non-zero common difference
  h2 : ∀ n : ℕ, a (n + 1) = a n + d  -- Arithmetic sequence property
  h3 : (a 3) ^ 2 = a 1 * a 4  -- Geometric sequence property for a_1, a_3, a_4

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  n / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

/-- The main theorem -/
theorem special_arithmetic_sequence_ratio (seq : SpecialArithmeticSequence) :
  (S seq 3 - S seq 2) / (S seq 5 - S seq 3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_arithmetic_sequence_ratio_l855_85502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario_uses_systematic_sampling_l855_85554

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | Lottery
  | RandomNumberTable
  | Systematic

/-- Represents a class of students --/
structure StudentClass where
  size : Nat
  numbering : Fin size → Nat

/-- Represents the grade with multiple classes --/
structure Grade where
  classCount : Nat
  classes : Fin classCount → StudentClass

/-- The sampling method used in the scenario --/
def scenarioSamplingMethod : SamplingMethod := SamplingMethod.Systematic

/-- The grade in the scenario --/
def scenarioGrade : Grade where
  classCount := 12
  classes := fun _ => { size := 50, numbering := fun _ => sorry }

/-- The selected student number in each class --/
def selectedStudentNumber : Nat := 40

theorem scenario_uses_systematic_sampling :
  (∀ i : Fin scenarioGrade.classCount,
    ∃ j : Fin (scenarioGrade.classes i).size,
      (scenarioGrade.classes i).numbering j = selectedStudentNumber) →
  scenarioSamplingMethod = SamplingMethod.Systematic :=
by
  sorry

#check scenario_uses_systematic_sampling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scenario_uses_systematic_sampling_l855_85554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l855_85544

/-- The curve C in the Cartesian plane -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2 * p.1}

/-- The line L in the Cartesian plane -/
noncomputable def L (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 = Real.sqrt 3 * p.2 + m}

/-- The point P -/
def P (m : ℝ) : ℝ × ℝ := (m, 0)

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem statement -/
theorem intersection_property (m : ℝ) :
  ∃ (A B : ℝ × ℝ), A ∈ C ∩ L m ∧ B ∈ C ∩ L m ∧ A ≠ B ∧
  distance (P m) A * distance (P m) B = 1 →
  m = 1 + Real.sqrt 2 ∨ m = 1 - Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l855_85544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approx_l855_85557

/-- Represents the dimensions and costs of paths on a rectangular lawn. -/
structure LawnPaths where
  length : ℝ
  breadth : ℝ
  path1_width : ℝ
  path1_cost : ℝ
  path2_width : ℝ
  path2_cost : ℝ
  path3_width : ℝ
  path3_cost : ℝ
  path4_diameter : ℝ
  path4_cost : ℝ

/-- Calculates the total cost of traveling all paths on the lawn. -/
noncomputable def totalPathCost (lawn : LawnPaths) : ℝ :=
  let path1_area := lawn.path1_width * lawn.length
  let path2_area := lawn.path2_width * lawn.breadth
  let diagonal := Real.sqrt (lawn.length ^ 2 + lawn.breadth ^ 2)
  let path3_area := lawn.path3_width * diagonal
  let path4_area := Real.pi * (lawn.path4_diameter / 2) ^ 2
  path1_area * lawn.path1_cost +
  path2_area * lawn.path2_cost +
  path3_area * lawn.path3_cost +
  path4_area * lawn.path4_cost

/-- Theorem stating that the total cost of traveling all paths is approximately 5040.64 rs. -/
theorem total_cost_approx (lawn : LawnPaths)
  (h1 : lawn.length = 100)
  (h2 : lawn.breadth = 80)
  (h3 : lawn.path1_width = 5)
  (h4 : lawn.path1_cost = 2)
  (h5 : lawn.path2_width = 4)
  (h6 : lawn.path2_cost = 1.5)
  (h7 : lawn.path3_width = 6)
  (h8 : lawn.path3_cost = 3)
  (h9 : lawn.path4_diameter = 20)
  (h10 : lawn.path4_cost = 4) :
  ∃ ε > 0, |totalPathCost lawn - 5040.64| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_approx_l855_85557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_altitude_feet_relation_l855_85535

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def isAcuteAngled (t : Triangle) : Prop := sorry

noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

noncomputable def altitudeFeet (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_altitude_feet_relation (t : Triangle) 
  (h_acute : isAcuteAngled t) 
  (H : ℝ × ℝ) 
  (H₁ H₂ H₃ : ℝ × ℝ) :
  H = orthocenter t →
  (H₁, H₂, H₃) = altitudeFeet t →
  distance t.A H * distance t.A H₁ + 
  distance t.B H * distance t.B H₂ + 
  distance t.C H * distance t.C H₃ = 
  1/2 * (distance t.A t.B ^ 2 + distance t.B t.C ^ 2 + distance t.C t.A ^ 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_altitude_feet_relation_l855_85535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_neighborhood_range_l855_85576

def B_neighborhood (A B : ℝ) := {x : ℝ | |x - A| < B}

theorem symmetric_neighborhood_range (a b : ℝ) :
  (∀ x, x ∈ B_neighborhood (a + b - 2) (a + b) ↔ -x ∈ B_neighborhood (a + b - 2) (a + b)) →
  Set.range (λ (x : ℝ) => 1 / (2 - x) + 4 / x) = Set.Iic (1 / 2) ∪ Set.Ici (9 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_neighborhood_range_l855_85576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_50_value_l855_85562

def G : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | (n + 1) => (3 * G n + 1) / 3

theorem G_50_value : G 50 = 152 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_50_value_l855_85562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_240_degree_angle_l855_85522

/-- Given a point P(-4, a) on the terminal side of 240°, prove that a = -4√3 -/
theorem point_on_240_degree_angle (a : ℝ) : 
  (∃ P : ℝ × ℝ, P = (-4, a) ∧ P.1 = -4 * Real.cos (240 * π / 180) ∧ P.2 = -4 * Real.sin (240 * π / 180)) → 
  a = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_240_degree_angle_l855_85522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_concentration_formula_l855_85582

/-- The concentration of a saline solution mixture -/
noncomputable def mixedConcentration (a b : ℝ) : ℝ :=
  (0.15 * a + 0.20 * b) / (a + b)

/-- Theorem stating that the concentration of the mixed solution
    is equal to the total amount of salt divided by the total mass -/
theorem mixed_concentration_formula (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  mixedConcentration a b = (0.15 * a + 0.20 * b) / (a + b) := by
  -- Unfold the definition of mixedConcentration
  unfold mixedConcentration
  -- The equation is true by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_concentration_formula_l855_85582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_hcf_24_195_l855_85575

def hcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem reciprocal_of_hcf_24_195 : (1 : ℚ) / (hcf 24 195 : ℚ) = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_hcf_24_195_l855_85575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_intersect_at_one_l855_85583

-- Define the common log function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := lg (x^2)
noncomputable def g (x : ℝ) : ℝ := (lg x)^2

-- Theorem statement
theorem functions_intersect_at_one :
  ∀ x : ℝ, x > 0 → (f 1 = g 1 ∧ f 1 = 0 ∧ g 1 = 0) := by
  sorry

#check functions_intersect_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_intersect_at_one_l855_85583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_radius_correct_l855_85586

/-- A symmetrical figure composed of 3 unit squares arranged in a right-angle triangle formation -/
structure TriSquareFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- Assertion that the side length is 1 -/
  is_unit : side_length = 1

/-- The radius of the smallest circle containing the TriSquareFigure -/
noncomputable def smallest_circle_radius (figure : TriSquareFigure) : ℝ :=
  (5 * Real.sqrt 17) / 16

/-- Theorem stating that the smallest_circle_radius function correctly calculates the radius -/
theorem smallest_circle_radius_correct (figure : TriSquareFigure) :
  smallest_circle_radius figure = (5 * Real.sqrt 17) / 16 :=
by
  -- Unfold the definition of smallest_circle_radius
  unfold smallest_circle_radius
  -- The equality holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_circle_radius_correct_l855_85586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_is_natural_l855_85511

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 => (1/2) * sequence_a (n + 1) + 1 / (4 * sequence_a (n + 1))

noncomputable def sequence_b (n : ℕ) : ℝ := Real.sqrt (2 / (2 * (sequence_a n)^2 - 1))

theorem sequence_b_is_natural (n : ℕ) (h : n > 1) : ∃ k : ℕ, sequence_b n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_b_is_natural_l855_85511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_coefficients_l855_85526

-- Define the vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

-- State the theorem
theorem vector_sum_coefficients (l m : ℝ) :
  c = l • a + m • b → l + m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_coefficients_l855_85526
