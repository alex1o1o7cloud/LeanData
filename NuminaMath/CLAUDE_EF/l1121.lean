import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_asymptotically_stable_l1121_112176

-- Define the system of differential equations
def dx_dt (x y : ℝ) : ℝ := -x + 4*y - 4*x*y^3
def dy_dt (x y : ℝ) : ℝ := -2*y - x^2*y^2

-- Define the Lyapunov function
def v (a b x y : ℝ) : ℝ := a*x^2 + b*y^2

-- Define partial derivatives of v
def dv_dx (a b x _y : ℝ) : ℝ := 2*a*x
def dv_dy (a b _x y : ℝ) : ℝ := 2*b*y

-- State the theorem
theorem origin_asymptotically_stable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), v a b x y > 0 ∨ (x = 0 ∧ y = 0)) ∧
  (∀ (x y : ℝ), x ≠ 0 ∨ y ≠ 0 →
    (dv_dx a b x y) * (dx_dt x y) + (dv_dy a b x y) * (dy_dt x y) < 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_asymptotically_stable_l1121_112176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_real_implies_zero_l1121_112194

theorem complex_square_real_implies_zero (x : ℝ) :
  (x + Complex.I : ℂ) ^ 2 ∈ Set.range (Complex.ofReal) →
  x = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_real_implies_zero_l1121_112194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_is_two_thirds_l1121_112150

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  greenTime : ℝ
  redTime : ℝ
  totalTime : ℝ
  green_red_sum : greenTime + redTime = totalTime

/-- Expected waiting time for a pedestrian at a traffic light -/
noncomputable def expectedWaitingTime (cycle : TrafficLightCycle) : ℝ :=
  (cycle.redTime / cycle.totalTime) * (cycle.redTime / 2)

/-- Theorem: Expected waiting time for the given traffic light cycle is 2/3 minutes -/
theorem expected_waiting_time_is_two_thirds
  (cycle : TrafficLightCycle)
  (h_green : cycle.greenTime = 1)
  (h_red : cycle.redTime = 2)
  (h_total : cycle.totalTime = 3) :
  expectedWaitingTime cycle = 2/3 := by
  sorry

#check expected_waiting_time_is_two_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_waiting_time_is_two_thirds_l1121_112150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_range_of_a_l1121_112104

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

noncomputable def g (x : ℝ) : ℝ := -Real.log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := min (f a x) (g x)

theorem three_zeros_range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    h a x₁ = 0 ∧ h a x₂ = 0 ∧ h a x₃ = 0) →
  -5/4 < a ∧ a < -3/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_zeros_range_of_a_l1121_112104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_inequality_l1121_112125

theorem subset_inequality (n k : ℕ) (A : Type*) [Fintype A] (S : Finset (Finset A)) :
  (Fintype.card A = n) →
  (Finset.card S = k) →
  (∀ (x y : A), x ≠ y → ∃ (Ai : Finset A), Ai ∈ S ∧ ((x ∈ Ai ∧ y ∉ Ai) ∨ (x ∉ Ai ∧ y ∈ Ai))) →
  2^k ≥ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_inequality_l1121_112125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_gasoline_consumption_l1121_112173

/-- Proves that a car with given fuel consumption, speed, and travel time uses a specific amount of gasoline -/
theorem car_gasoline_consumption 
  (fuel_consumption : ℝ) 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : fuel_consumption = 0.14) 
  (h2 : speed = 93.6) 
  (h3 : time = 2.5) : 
  fuel_consumption * speed * time = 32.76 := by
  -- Replace all occurrences with their actual values
  rw [h1, h2, h3]
  -- Simplify the expression
  norm_num
  -- The proof is complete
  done

#check car_gasoline_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_gasoline_consumption_l1121_112173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l1121_112135

-- Define the points
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem y_coordinate_of_P (P : ℝ × ℝ) 
  (h1 : distance P A + distance P D = 10)
  (h2 : distance P B + distance P C = 10) : 
  P.2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l1121_112135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_increase_l1121_112172

theorem cricket_average_increase (innings : ℕ) (current_average : ℕ) (increase : ℕ) : 
  innings = 10 → current_average = 32 → increase = 4 →
  (innings * current_average + (current_average + increase) * (innings + 1) - innings * current_average) / (innings + 1) = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_increase_l1121_112172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_parallelogram_l1121_112137

noncomputable section

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the trajectory Γ
def trajectory_Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define line l
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the condition for OARB to be a parallelogram
def is_parallelogram (k m : ℝ) : Prop :=
  k = 3/8 ∧ m = 5/8 ∨ k = 0 ∧ m = 1

theorem trajectory_and_parallelogram :
  ∀ (x y k m : ℝ),
    (∃ (P : ℝ × ℝ), circle_E P.1 P.2) →
    (∃ (Q : ℝ × ℝ), trajectory_Γ Q.1 Q.2) →
    line_l k m 1 1 →
    (trajectory_Γ x y ↔ x^2 / 4 + y^2 = 1) ∧
    (is_parallelogram k m ↔ line_l k m x y ∧ trajectory_Γ x y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_parallelogram_l1121_112137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_film_radius_l1121_112180

noncomputable section

/-- The radius of a circular film formed by pouring a volume of liquid onto water -/
def film_radius (volume : ℝ) (thickness : ℝ) : ℝ :=
  Real.sqrt (volume / (Real.pi * thickness))

theorem liquid_film_radius :
  let volume : ℝ := 1000  -- cm³
  let thickness : ℝ := 0.2  -- cm
  film_radius volume thickness = Real.sqrt (5000 / Real.pi) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_film_radius_l1121_112180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_l1121_112169

theorem log_equation (x : ℝ) (h1 : x < 1) (h2 : ((Real.log x) / (Real.log 10))^2 - (Real.log (x^2)) / (Real.log 10) = 48) :
  ((Real.log x) / (Real.log 10))^3 - (Real.log (x^3)) / (Real.log 10) = -198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_l1121_112169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_ratio_l1121_112121

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- The ratio of the sum of squares of a parallelepiped's space diagonal and three face diagonals
    to the sum of squares of its three edges is 4. -/
theorem parallelepiped_diagonal_ratio (a b c : V) : 
  (‖a‖^2 + ‖(a + b) - b‖^2 + ‖(a + b + c) - (b + c)‖^2 + ‖(a + c) - c‖^2) / (‖a‖^2 + ‖b‖^2 + ‖c‖^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_ratio_l1121_112121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_is_sqrt3_div_2_l1121_112100

/-- A regular octahedron with edge length 1 -/
structure RegularOctahedron where
  edge_length : ℝ
  is_regular : edge_length = 1

/-- The shortest path on the surface of a regular octahedron from the midpoint of an edge to a vertex -/
noncomputable def shortest_path (o : RegularOctahedron) : ℝ :=
  Real.sqrt 3 / 2

/-- Theorem stating that the shortest path on the surface of a regular octahedron 
    from the midpoint of an edge to a vertex is √3/2 -/
theorem shortest_path_is_sqrt3_div_2 (o : RegularOctahedron) :
  shortest_path o = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_is_sqrt3_div_2_l1121_112100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_percentage_difference_l1121_112186

-- Define the angles of the sectors
noncomputable def manufacturing_angle : ℝ := 144
noncomputable def rd_angle : ℝ := 108
noncomputable def marketing_angle : ℝ := 108

-- Define the total angle of a circle
noncomputable def total_angle : ℝ := 360

-- Define the percentage calculation function
noncomputable def percentage (angle : ℝ) : ℝ := (angle / total_angle) * 100

-- Define the theorem
theorem employee_percentage_difference :
  let manufacturing_percent := percentage manufacturing_angle
  let rd_marketing_percent := percentage (rd_angle + marketing_angle)
  abs (manufacturing_percent - rd_marketing_percent) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_percentage_difference_l1121_112186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_150_l1121_112141

structure Rhombus where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

def circumradius (A B C : ℝ × ℝ) : ℝ := sorry

def diagonal_length (A B : ℝ × ℝ) : ℝ := sorry

def rhombus_area (r : Rhombus) : ℝ := sorry

theorem rhombus_area_is_150 (EFGH : Rhombus) :
  circumradius EFGH.E EFGH.F EFGH.G = 15 →
  circumradius EFGH.E EFGH.G EFGH.H = 30 →
  diagonal_length EFGH.E EFGH.G = 3 * diagonal_length EFGH.F EFGH.H →
  rhombus_area EFGH = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_is_150_l1121_112141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_interest_rate_l1121_112191

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem higher_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (base_rate : ℝ) 
  (interest_difference : ℝ) 
  (higher_rate : ℝ) :
  principal = 8400 →
  time = 2 →
  base_rate = 0.1 →
  interest_difference = 840 →
  simple_interest principal higher_rate time = 
    simple_interest principal base_rate time + interest_difference →
  higher_rate = 0.15 := by
  sorry

#check higher_interest_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_higher_interest_rate_l1121_112191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_probability_theorem_problem_solution_l1121_112166

/-- Represents a production line with its production volume and defect rate -/
structure ProductionLine where
  volume : ℝ
  defectRate : ℝ

/-- Calculates the probability of a defective product given multiple production lines -/
def defectiveProbability (lines : List ProductionLine) : ℝ :=
  (lines.map fun line => line.volume * line.defectRate).sum

/-- Theorem: The probability of a defective product is the sum of defect probabilities from each line -/
theorem defective_probability_theorem (lines : List ProductionLine) 
  (h1 : lines.length = 3)
  (h2 : (lines.map ProductionLine.volume).sum = 1)
  (h3 : ∀ l ∈ lines, 0 ≤ l.volume ∧ l.volume ≤ 1)
  (h4 : ∀ l ∈ lines, 0 ≤ l.defectRate ∧ l.defectRate ≤ 1) :
  defectiveProbability lines = 
    (lines.map fun l => l.volume * l.defectRate).sum := by
  sorry

/-- The specific problem instance -/
def problemInstance : List ProductionLine := [
  ⟨0.30, 0.03⟩,
  ⟨0.25, 0.02⟩,
  ⟨0.45, 0.04⟩
]

/-- Theorem: The probability of a defective product in the given problem is 0.032 -/
theorem problem_solution :
  defectiveProbability problemInstance = 0.032 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_probability_theorem_problem_solution_l1121_112166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l1121_112129

/-- The shaded area of a square with six inscribed circles -/
theorem shaded_area_square_with_circles (square_side : ℝ) (num_circles : ℕ) 
  (h1 : square_side = 24)
  (h2 : num_circles = 6) : 
  ℝ := by
  -- Define the shaded area
  let shaded_area := square_side^2 - num_circles * Real.pi * (square_side / (2 * num_circles))^2
  -- Prove that shaded_area = 576 - 96π
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l1121_112129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ijk_eq_neg_one_l1121_112117

/-- Quaternion type -/
structure MyQuaternion where
  re : ℝ
  i : ℝ
  j : ℝ
  k : ℝ

/-- Multiplication of quaternions -/
def quat_mul (q1 q2 : MyQuaternion) : MyQuaternion :=
  { re := q1.re * q2.re - q1.i * q2.i - q1.j * q2.j - q1.k * q2.k,
    i := q1.re * q2.i + q1.i * q2.re + q1.j * q2.k - q1.k * q2.j,
    j := q1.re * q2.j + q1.j * q2.re + q1.k * q2.i - q1.i * q2.k,
    k := q1.re * q2.k + q1.k * q2.re + q1.i * q2.j - q1.j * q2.i }

/-- Unit quaternions i, j, k -/
def i : MyQuaternion := { re := 0, i := 1, j := 0, k := 0 }
def j : MyQuaternion := { re := 0, i := 0, j := 1, k := 0 }
def k : MyQuaternion := { re := 0, i := 0, j := 0, k := 1 }

/-- Properties of quaternions -/
axiom i_squared : quat_mul i i = { re := -1, i := 0, j := 0, k := 0 }
axiom j_squared : quat_mul j j = { re := -1, i := 0, j := 0, k := 0 }
axiom k_squared : quat_mul k k = { re := -1, i := 0, j := 0, k := 0 }
axiom ij_eq_k : quat_mul i j = k
axiom ji_eq_neg_k : quat_mul j i = { re := 0, i := 0, j := 0, k := -1 }
axiom jk_eq_i : quat_mul j k = i
axiom kj_eq_neg_i : quat_mul k j = { re := 0, i := -1, j := 0, k := 0 }
axiom ki_eq_j : quat_mul k i = j
axiom ik_eq_neg_j : quat_mul i k = { re := 0, i := 0, j := -1, k := 0 }

/-- Theorem: ijk = -1 -/
theorem ijk_eq_neg_one : quat_mul (quat_mul i j) k = { re := -1, i := 0, j := 0, k := 0 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ijk_eq_neg_one_l1121_112117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_distance_sum_l1121_112182

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (2, 0)

-- Define the points on the parabola
def PointOnParabola (P : ℝ × ℝ) : Prop :=
  Parabola P.1 P.2

-- Define the distance between two points
noncomputable def Distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem parabola_points_distance_sum 
  (P₁ P₂ P₃ : ℝ × ℝ) 
  (h₁ : PointOnParabola P₁)
  (h₂ : PointOnParabola P₂)
  (h₃ : PointOnParabola P₃)
  (h_sum : P₁.1 + P₂.1 + P₃.1 = 10) :
  Distance P₁ Focus + Distance P₂ Focus + Distance P₃ Focus = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_points_distance_sum_l1121_112182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truth_count_l1121_112153

/-- Represents the three types of ice cream -/
inductive IceCream
  | vanilla
  | chocolate
  | fruit
deriving DecidableEq

/-- Represents a dwarf -/
structure Dwarf where
  truthful : Bool
  favorite : IceCream
deriving DecidableEq

/-- The problem statement -/
theorem dwarf_truth_count (dwarfs : Finset Dwarf) : 
  (dwarfs.card = 10) →
  (∀ d ∈ dwarfs, d.truthful ∨ ¬d.truthful) →
  (∀ d ∈ dwarfs, ∃! ic : IceCream, d.favorite = ic) →
  (∀ d ∈ dwarfs, d.truthful = (d.favorite = IceCream.vanilla) ∨ ¬d.truthful) →
  ((dwarfs.filter (λ d => (d.truthful ∧ d.favorite = IceCream.chocolate) ∨ 
                          (¬d.truthful ∧ d.favorite ≠ IceCream.chocolate))).card = 5) →
  ((dwarfs.filter (λ d => (d.truthful ∧ d.favorite = IceCream.fruit) ∨ 
                          (¬d.truthful ∧ d.favorite ≠ IceCream.fruit))).card = 1) →
  (dwarfs.filter (λ d => d.truthful)).card = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dwarf_truth_count_l1121_112153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_l1121_112188

theorem constant_function (f : ℝ → ℝ) (a : ℝ) (h1 : a > 0) (h2 : f a = 1)
  (h3 : ∀ x y : ℝ, x > 0 → y > 0 → f x * f y + f (a / x) * f (a / y) = 2 * f (x * y)) :
  ∀ x : ℝ, x > 0 → f x = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_l1121_112188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_doubles_hourly_rate_bears_per_week_increase_is_80_percent_l1121_112114

/-- Represents Jane's toy bear production --/
structure BearProduction where
  bears_per_week : ℝ
  hours_per_week : ℝ

/-- Represents Jane's toy bear production with an assistant --/
noncomputable def with_assistant (prod : BearProduction) (percent_increase : ℝ) : BearProduction :=
  { bears_per_week := prod.bears_per_week * (1 + percent_increase / 100),
    hours_per_week := prod.hours_per_week * 0.9 }

/-- The percentage increase in bears per week when working with an assistant --/
noncomputable def assistant_increase (prod : BearProduction) : ℝ :=
  let new_prod := with_assistant prod 80
  (new_prod.bears_per_week - prod.bears_per_week) / prod.bears_per_week * 100

theorem assistant_doubles_hourly_rate (prod : BearProduction) :
  (with_assistant prod 80).bears_per_week / (with_assistant prod 80).hours_per_week =
  2 * (prod.bears_per_week / prod.hours_per_week) := by
  sorry

theorem bears_per_week_increase_is_80_percent (prod : BearProduction) :
  assistant_increase prod = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assistant_doubles_hourly_rate_bears_per_week_increase_is_80_percent_l1121_112114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l1121_112164

/-- A parabola is defined by its coefficients a, b, and c in the form ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ :=
  (- p.b / (2 * p.a), - (p.b^2 - 4*p.a*p.c) / (4 * p.a))

/-- Check if a point lies on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop :=
  p.a * x^2 + p.b * x + p.c = y

/-- Theorem stating that the given parabola satisfies all conditions -/
theorem parabola_satisfies_conditions : ∃ p : Parabola,
  p.a = -3 ∧ p.b = 18 ∧ p.c = -22 ∧
  vertex p = (3, 5) ∧
  contains_point p 2 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_satisfies_conditions_l1121_112164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l1121_112149

/-- The curve function f(x) = (2/3)x^3 - x^2 + ax - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2/3) * x^3 - x^2 + a*x - 1

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2*x^2 - 2*x + a

theorem tangent_slope_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    f' a x₁ = 3 ∧ f' a x₂ = 3 ∧
    (∀ x : ℝ, x > 0 ∧ f' a x = 3 → x = x₁ ∨ x = x₂)) →
  3 < a ∧ a < 7/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l1121_112149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_per_review_is_twenty_l1121_112131

/-- Represents the payment details for a limousine driver --/
structure DriverPayment where
  hourlyWage : ℚ
  rideBonus : ℚ
  numRides : ℕ
  hoursWorked : ℚ
  gasGallons : ℚ
  gasPrice : ℚ
  numReviews : ℕ
  totalOwed : ℚ

/-- Calculates the bonus per positive review --/
def bonusPerReview (p : DriverPayment) : ℚ :=
  (p.totalOwed - (p.hourlyWage * p.hoursWorked + p.rideBonus * ↑p.numRides + p.gasGallons * p.gasPrice)) / ↑p.numReviews

/-- Theorem stating that the bonus per positive review is $20 --/
theorem bonus_per_review_is_twenty :
  let p : DriverPayment := {
    hourlyWage := 15,
    rideBonus := 5,
    numRides := 3,
    hoursWorked := 8,
    gasGallons := 17,
    gasPrice := 3,
    numReviews := 2,
    totalOwed := 226
  }
  bonusPerReview p = 20 := by sorry

#eval bonusPerReview {
  hourlyWage := 15,
  rideBonus := 5,
  numRides := 3,
  hoursWorked := 8,
  gasGallons := 17,
  gasPrice := 3,
  numReviews := 2,
  totalOwed := 226
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bonus_per_review_is_twenty_l1121_112131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_var_prob_l1121_112152

-- Define a random variable ξ taking values 1 to n with equal probability
noncomputable def equal_prob (n : ℕ) (k : ℕ) : ℝ := 1 / n

-- Define the probability of ξ < 4
noncomputable def prob_less_than_4 (n : ℕ) : ℝ := 3 / n

-- Theorem statement
theorem random_var_prob (n : ℕ) :
  (∀ k : ℕ, k ≤ n → equal_prob n k = 1 / n) →
  prob_less_than_4 n = 0.3 →
  n = 10 := by
  intro h1 h2
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_var_prob_l1121_112152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vertex_l1121_112187

-- Define the complex numbers corresponding to the vertices
noncomputable def O : ℂ := 0
noncomputable def A : ℂ := 3 + 2*Complex.I
noncomputable def C : ℂ := -2 + 4*Complex.I

-- Define the parallelogram property
def is_parallelogram (O A B C : ℂ) : Prop :=
  B - O = A - O + C - O

-- Theorem statement
theorem parallelogram_vertex : 
  ∀ B : ℂ, is_parallelogram O A B C → B = 1 + 6*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_vertex_l1121_112187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1121_112108

/-- The curve function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 3)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' point.1

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_equation :
  tangent_line point.1 point.2 ∧
  ∀ x y : ℝ, y = f x → (x = point.1 ∨ (y - point.2) / (x - point.1) ≠ m) ∨ tangent_line x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1121_112108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1121_112178

theorem negation_equivalence :
  (¬ (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 3*x + 2 ≤ 0)) ↔
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 - 3*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l1121_112178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_with_sale_l1121_112162

theorem max_books_with_sale (regular_price : ℝ) (initial_budget : ℝ) : 
  regular_price > 0 →
  initial_budget = 40 * regular_price →
  (let pair_price := regular_price + regular_price / 2
   let num_pairs := ⌊initial_budget / pair_price⌋
   2 * num_pairs) = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_with_sale_l1121_112162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_inequalities_theta_l1121_112192

noncomputable section

def are_dual (f g : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧
    {x : ℝ | f x < 0} = Set.Ioo a b ∧
    {x : ℝ | g x < 0} = Set.Ioo (1/b) (1/a)

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := x^2 - 4 * Real.sqrt 3 * x * Real.cos (2*θ) + 2

noncomputable def g (θ : ℝ) (x : ℝ) : ℝ := 2 * x^2 + 4 * x * Real.sin (2*θ) + 1

theorem dual_inequalities_theta (θ : ℝ) 
  (h1 : are_dual (f θ) (g θ))
  (h2 : θ > 0)
  (h3 : θ < Real.pi) :
  θ = Real.pi / 3 ∨ θ = 5 * Real.pi / 6 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dual_inequalities_theta_l1121_112192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equal_radius_arc_measure_l1121_112130

/-- A circle with a chord equal in length to its radius has an opposite arc of either 60° or 300°. -/
theorem chord_equal_radius_arc_measure (c : Real) (chord : Real) (arc : Real) :
  chord = c →
  (arc = 60 ∨ arc = 300) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equal_radius_arc_measure_l1121_112130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_triangle_le_quarter_l1121_112165

/-- Given a figure with area S and three lines that each divide the figure's area in half,
    the area of the region enclosed by the triangle formed by these three lines
    is less than or equal to S/4 -/
theorem area_enclosed_by_triangle_le_quarter (S : ℝ) (h : S > 0) : 
  ∃ (S₁ S₂ S₃ S₄ S₅ S₆ S₇ : ℝ),
    (S₁ + S₂ + S₃ + S₄ + S₅ + S₆ + S₇ = S) ∧ 
    (S₃ + S₂ + S₇ = S / 2) ∧
    (S₁ + S₆ + S₂ + S₇ = S / 2) ∧
    (S₅ + S₄ + S₂ + S₇ = S / 2) ∧
    (∀ i : Fin 7, 0 ≤ (match i with
      | ⟨0, _⟩ => S₁ | ⟨1, _⟩ => S₂ | ⟨2, _⟩ => S₃ | ⟨3, _⟩ => S₄ 
      | ⟨4, _⟩ => S₅ | ⟨5, _⟩ => S₆ | ⟨6, _⟩ => S₇
    )) →
    S₁ ≤ S / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_triangle_le_quarter_l1121_112165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1121_112170

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := Real.log (x + Real.sqrt (2 + x^2))

-- Theorem statement
theorem g_neither_even_nor_odd :
  (∃ x : ℝ, g (-x) ≠ g x) ∧ (∃ x : ℝ, g (-x) ≠ -g x) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l1121_112170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_l1121_112175

theorem sum_reciprocals_bound (N : ℕ) (A : Finset ℕ) (h_N : N ≥ 4) 
  (h_A : ∀ a ∈ A, a < N) 
  (h_lcm : ∀ a b, a ∈ A → b ∈ A → a ≠ b → Nat.lcm a b > N) :
  (A.sum fun a => (1 : ℚ) / a) < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_l1121_112175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l1121_112106

theorem quadratic_root_difference :
  let a : ℝ := 5
  let b : ℝ := -2
  let c : ℝ := -15
  let discriminant := b^2 - 4*a*c
  let root_difference := (Real.sqrt discriminant) / a
  (5 : ℝ) * root_difference = Real.sqrt 304 ∧ 
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ 304) :=
by sorry

#check quadratic_root_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_difference_l1121_112106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1121_112105

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (2 * a^2) / x + x

theorem function_properties (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x > 0 → HasDerivAt (f a) ((a / x) - (2 * a^2) / x^2 + 1) x) →
  (HasDerivAt (f a) (-2) 1) →
  a = 3/2 ∧
  (∀ x : ℝ, 0 < x → x < 3/2 → HasDerivAt (f (3/2)) (((3/2) / x) - (2 * (3/2)^2) / x^2 + 1) x ∧ ((3/2) / x) - (2 * (3/2)^2) / x^2 + 1 < 0) ∧
  (∀ x : ℝ, x > 3/2 → HasDerivAt (f (3/2)) (((3/2) / x) - (2 * (3/2)^2) / x^2 + 1) x ∧ ((3/2) / x) - (2 * (3/2)^2) / x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1121_112105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_part1_line_equation_part2_l1121_112183

noncomputable section

-- Define the line l
def line_l (k : ℝ) : ℝ → ℝ := λ x ↦ k * (x - 1) + 2

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (1 - 2/k, 0)

-- Define point B
def point_B (k : ℝ) : ℝ × ℝ := (0, 2 - k)

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define vector AP
def vector_AP (k : ℝ) : ℝ × ℝ := (2/k, 2)

-- Define vector PB
def vector_PB (k : ℝ) : ℝ × ℝ := (-1, -k)

-- Define the area of triangle AOB
def area_AOB (k : ℝ) : ℝ := (1/2) * (1 - 2/k) * (2 - k)

theorem line_equation_part1 :
  ∃ k : ℝ, k < 0 ∧ vector_AP k = 3 • vector_PB k →
  line_l k = λ x ↦ (-2/3) * x + 14/3 :=
by sorry

theorem line_equation_part2 :
  ∃ k : ℝ, k < 0 ∧ area_AOB k = 4 →
  line_l k = λ x ↦ -2 * x + 6 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_part1_line_equation_part2_l1121_112183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_function_satisfies_equation_l1121_112143

theorem no_integer_function_satisfies_equation :
  ¬ ∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_function_satisfies_equation_l1121_112143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_MQ_l1121_112113

-- Define the points
def A : ℝ × ℝ := (1, 1)
def P : ℝ × ℝ := (-1, 3)
def Q : ℝ × ℝ := (2, 4)

-- Define the center of the circle (midpoint of AP)
def C : ℝ × ℝ := (0, 2)

-- Define the radius of the circle
noncomputable def r : ℝ := Real.sqrt 2

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_range_MQ :
  ∀ M : ℝ × ℝ, distance C M = r →
    r ≤ distance M Q ∧ distance M Q ≤ 3 * r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_MQ_l1121_112113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_one_zero_l1121_112155

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x

-- Part I: Minimum value of f(x) on (0, π)
theorem f_min_value (x : ℝ) (hx : 0 < x ∧ x < Real.pi) : f x ≥ -1 := by
  sorry

-- Part II: Exactly one zero of f(x) in (2, 3)
theorem f_one_zero : ∃! x, 2 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_f_one_zero_l1121_112155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_triangle_line_theorem_l1121_112124

/-- A line that forms a triangle with the positive halves of the coordinate axes -/
structure AxisTriangleLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0

/-- The area of the triangle formed by the line and the positive halves of the coordinate axes -/
noncomputable def triangleArea (l : AxisTriangleLine) : ℝ := (1/2) * l.a * l.b

/-- The difference between the x-intercept and y-intercept -/
noncomputable def interceptDifference (l : AxisTriangleLine) : ℝ := |l.a - l.b|

/-- The equation of the line in the form ax + by = ab -/
def lineEquation (l : AxisTriangleLine) : ℝ × ℝ × ℝ := (l.b, l.a, l.a * l.b)

theorem axis_triangle_line_theorem :
  ∀ l : AxisTriangleLine,
  triangleArea l = 2 ∧ interceptDifference l = 3 →
  lineEquation l = (1, 4, 4) ∨ lineEquation l = (4, 1, 4) := by
  sorry

#check axis_triangle_line_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_triangle_line_theorem_l1121_112124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1121_112177

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function -/
def g (x : ℝ) : ℝ := x - 2

/-- The distance function from a point (x, f(x)) on the curve to the line y = x - 2 -/
noncomputable def distance (x : ℝ) : ℝ := |f x - g x| / Real.sqrt 2

theorem min_distance_curve_to_line :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), x > 0 → distance x₀ ≤ distance x ∧ distance x₀ = Real.sqrt 2 := by
  sorry

#check min_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1121_112177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleToFillTable_l1121_112193

/-- Represents a 6x7 table of digits -/
def Table := Matrix (Fin 6) (Fin 7) Nat

/-- Checks if a sequence of numbers forms an arithmetic progression -/
def isArithmeticProgression (seq : List Nat) : Prop :=
  ∃ d, ∀ i, i + 1 < seq.length → seq[i + 1]! - seq[i]! = d

/-- Represents the condition that the fourth column contains specific digits -/
def fourthColumnCondition (t : Table) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 6)), ∀ i : Fin 6, t i 3 = [1, 2, 3, 4, 5, 7][perm i]!

/-- Represents the condition that all rows form arithmetic progressions -/
def rowsAreAP (t : Table) : Prop :=
  ∀ i : Fin 6, isArithmeticProgression (List.ofFn (λ j => t i j))

/-- Represents the condition that all columns form arithmetic progressions -/
def columnsAreAP (t : Table) : Prop :=
  ∀ j : Fin 7, isArithmeticProgression (List.ofFn (λ i => t i j))

/-- The main theorem stating that it's impossible to fill the table satisfying all conditions -/
theorem impossibleToFillTable : ¬∃ (t : Table), 
  fourthColumnCondition t ∧ rowsAreAP t ∧ columnsAreAP t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleToFillTable_l1121_112193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1121_112139

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x^2 + 2*x)

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, 0 ≤ x ∧ x < 3 ∧ f x = y) ↔ Real.exp (-3) < y ∧ y ≤ Real.exp 1 :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1121_112139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l1121_112138

/-- The volume of a regular triangular pyramid -/
noncomputable def pyramid_volume (a b : ℝ) : ℝ :=
  (a^3 * b) / (12 * Real.sqrt (3 * a^2 - 4 * b^2))

/-- Theorem: The volume of a regular triangular pyramid with base side length a and 
    height b dropped from a vertex of the base to the opposite lateral face is 
    (a³b) / (12√(3a² - 4b²)) -/
theorem regular_triangular_pyramid_volume (a b : ℝ) 
    (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a^2 > 4 * b^2) : 
  pyramid_volume a b = (a^3 * b) / (12 * Real.sqrt (3 * a^2 - 4 * b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_volume_l1121_112138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l1121_112118

/-- The cost of one pen in rubles -/
def pen_cost : ℕ := sorry

/-- The number of pens Masha bought -/
def masha_pens : ℕ := sorry

/-- The number of pens Olya bought -/
def olya_pens : ℕ := sorry

theorem total_pens_bought :
  pen_cost > 10 ∧
  pen_cost * masha_pens = 357 ∧
  pen_cost * olya_pens = 441 →
  masha_pens + olya_pens = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pens_bought_l1121_112118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_l1121_112148

noncomputable def odot (a b : ℝ) : ℝ := if a - b ≤ 1 then a else b

noncomputable def f (x : ℝ) : ℝ := odot (2^(x+1)) (1-x)

def g (x : ℝ) : ℝ := x^2 - 6*x

def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

theorem m_value (m : ℝ) :
  m ∈ ({-1, 0, 1, 3} : Set ℝ) →
  is_decreasing f m (m+1) →
  is_decreasing g m (m+1) →
  m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_l1121_112148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1121_112146

def a : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) + (Finset.range (n + 1)).sum (λ k => (n + 1).choose k)

theorem sequence_properties :
  (a 1 = 1 ∧ a 2 = 4 ∧ a 3 = 15 ∧ a 4 = 64 ∧ a 5 = 325) ∧
  (∀ n : ℕ, n ≥ 2 → a n = n + n * a (n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → (Finset.range n).prod (λ k => (1 + 1 / (a k))) < 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1121_112146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_ratio_l1121_112123

-- Define the mixtures and their ratios
def mixture_p_milk_ratio : ℚ := 5
def mixture_p_water_ratio : ℚ := 4
def mixture_q_milk_ratio : ℚ := 2
def mixture_q_water_ratio : ℚ := 7

-- Define the desired water percentage and total volume
def desired_water_percentage : ℚ := 3/5 -- 0.6 as a rational number
def total_volume : ℚ := 50

-- Define the quantities of mixtures p and q as variables
variable (x y : ℚ)

-- State the theorem
theorem mixture_ratio :
  ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ 
  x / y = 8 / 7 ∧
  (mixture_p_water_ratio * x + mixture_q_water_ratio * y) / 
  (mixture_p_milk_ratio * x + mixture_p_water_ratio * x + 
   mixture_q_milk_ratio * y + mixture_q_water_ratio * y) = desired_water_percentage ∧
  mixture_p_milk_ratio * x + mixture_p_water_ratio * x + 
  mixture_q_milk_ratio * y + mixture_q_water_ratio * y = total_volume :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_ratio_l1121_112123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_problem_l1121_112159

-- Define lg as the logarithm in base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_problem : lg 25 - 2 * lg (1/2) = 2 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lg_problem_l1121_112159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l1121_112154

noncomputable def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (-1)^n * a n + (1 / 2^n) + n - 3

theorem sequence_range (a : ℕ → ℝ) (t : ℝ) :
  (∀ n : ℕ+, sequence_sum a n = (-1)^n.val * a n.val + (1 / 2^n.val) + n.val - 3) →
  (∀ n : ℕ, (t - a (n + 1)) * (t - a n) < 0) →
  -3/4 < t ∧ t < 11/4 := by
  sorry

#check sequence_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l1121_112154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1121_112199

theorem right_triangle_area (hypotenuse base : ℝ) (h1 : hypotenuse = 15) (h2 : base = 9) :
  (base * Real.sqrt (hypotenuse^2 - base^2)) / 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l1121_112199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nikifor_minimal_expense_max_info_cost_l1121_112184

/-- Represents the total number of voters -/
def total_voters : ℕ := 25

/-- Represents the number of voters willing to sell their vote -/
def sellable_voters : ℕ := 15

/-- Represents the number of votes each candidate starts with -/
def base_votes : ℕ := 5

/-- Represents the minimum number of votes needed to win the election -/
def votes_to_win : ℕ := 13

/-- Function that calculates the number of votes a candidate receives based on the price offered per vote -/
def votes_received (price : ℕ) : ℕ :=
  if price = 0 then base_votes
  else if price ≤ sellable_voters then base_votes + price
  else total_voters - base_votes

/-- Theorem stating that Nikifor's minimal expense to win the election is 117 monetary units -/
theorem nikifor_minimal_expense :
  ∃ (price : ℕ), votes_received price ≥ votes_to_win ∧
  price * votes_received price = 117 := by
  sorry

/-- Theorem stating the maximum amount Nikifor would pay for individual voter information -/
theorem max_info_cost :
  ∃ (F : ℕ), F = 3 ∧
  (Finset.sum (Finset.range 9) id) + total_voters * F ≤ 117 := by
  sorry

#eval total_voters
#eval votes_received 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nikifor_minimal_expense_max_info_cost_l1121_112184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_sale_earnings_l1121_112156

/-- Represents the total earnings from Toby's garage sale -/
noncomputable def total_earnings : ℝ := 2408.54

/-- Represents the price of the treadmill -/
noncomputable def treadmill_price : ℝ := 300

/-- Represents the price of the chest of drawers -/
noncomputable def chest_price : ℝ := treadmill_price / 2

/-- Represents the price of the television -/
noncomputable def tv_price : ℝ := treadmill_price * 3

/-- Represents the price of the bicycle -/
noncomputable def bicycle_price : ℝ := chest_price * 2 - 25

/-- Represents the price of the antique vase -/
noncomputable def vase_price : ℝ := bicycle_price + 75

/-- Represents the sum of prices for the five items excluding the coffee table -/
noncomputable def sum_five_items : ℝ := treadmill_price + chest_price + tv_price + bicycle_price + vase_price

/-- Represents the price of the coffee table as a percentage of total earnings -/
def coffee_table_percent : ℝ := 0.08

/-- Represents the percentage of total earnings from all six items -/
def six_items_percent : ℝ := 0.90

theorem garage_sale_earnings :
  sum_five_items + coffee_table_percent * total_earnings = six_items_percent * total_earnings := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_sale_earnings_l1121_112156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_groups_l1121_112158

def range : ℝ := 35
def class_width : ℝ := 4

theorem number_of_groups : Nat.ceil (range / class_width) = 9 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_groups_l1121_112158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_gradients_zero_l1121_112185

noncomputable section

/-- The angle between the gradients of two functions at a given point -/
def angle_between_gradients (u v : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
  let grad_u := (deriv (fun x => u x y) x, deriv (fun y => u x y) y)
  let grad_v := (deriv (fun x => v x y) x, deriv (fun y => v x y) y)
  let dot_product := grad_u.1 * grad_v.1 + grad_u.2 * grad_v.2
  let magnitude_u := Real.sqrt (grad_u.1^2 + grad_u.2^2)
  let magnitude_v := Real.sqrt (grad_v.1^2 + grad_v.2^2)
  Real.arccos (dot_product / (magnitude_u * magnitude_v))

/-- The function u(x, y) = √(x² + y²) -/
def u (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The function v(x, y) = x + y + 2√(xy) -/
def v (x y : ℝ) : ℝ := x + y + 2 * Real.sqrt (x * y)

theorem angle_between_gradients_zero :
  angle_between_gradients u v 1 1 = 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_gradients_zero_l1121_112185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polyhedron_spheres_l1121_112111

-- Define the Point type
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure RegularPolyhedron where
  vertices : Set Point
  faces : Set (Set Point)
  edges : Set (Set Point)

structure Sphere where
  center : Point
  radius : ℝ

def onSphere (p : Point) (s : Sphere) : Prop :=
  (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 = s.radius^2

def tangentToSphere (points : Set Point) (s : Sphere) : Prop :=
  ∃ p ∈ points, onSphere p s ∧ ∀ q ∈ points, q ≠ p → ¬onSphere q s

def CircumscribedSphere (p : RegularPolyhedron) (s : Sphere) : Prop :=
  ∀ v ∈ p.vertices, onSphere v s

def InscribedSphere (p : RegularPolyhedron) (s : Sphere) : Prop :=
  ∀ f ∈ p.faces, tangentToSphere f s

def MidSphere (p : RegularPolyhedron) (s : Sphere) : Prop :=
  ∀ e ∈ p.edges, tangentToSphere e s

theorem regular_polyhedron_spheres (p : RegularPolyhedron) :
  ∃ (center : Point) (r₁ r₂ r₃ : ℝ),
    let s₁ := Sphere.mk center r₁
    let s₂ := Sphere.mk center r₂
    let s₃ := Sphere.mk center r₃
    CircumscribedSphere p s₁ ∧
    InscribedSphere p s₂ ∧
    MidSphere p s₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polyhedron_spheres_l1121_112111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_sum_exists_l1121_112107

theorem divisible_sum_exists (S : Finset ℕ) (h1 : S ⊆ Finset.range 2011) (h2 : S.card = 673) :
  ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a + b) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_sum_exists_l1121_112107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l1121_112144

/-- Given that u, v, and w are the three distinct solutions of the equation 
    (x - ∛20)(x - ∛70)(x - ∛120) = 1/2, prove that u³ + v³ + w³ = 211.5 -/
theorem sum_of_cubes_of_solutions (u v w : ℝ) : 
  (u - 20^(1/3 : ℝ)) * (u - 70^(1/3 : ℝ)) * (u - 120^(1/3 : ℝ)) = 1/2 →
  (v - 20^(1/3 : ℝ)) * (v - 70^(1/3 : ℝ)) * (v - 120^(1/3 : ℝ)) = 1/2 →
  (w - 20^(1/3 : ℝ)) * (w - 70^(1/3 : ℝ)) * (w - 120^(1/3 : ℝ)) = 1/2 →
  u ≠ v → u ≠ w → v ≠ w →
  u^3 + v^3 + w^3 = 211.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l1121_112144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_gas_carbon_atoms_l1121_112174

/-- Avogadro's constant -/
noncomputable def N_A : ℝ := Real.exp 1 -- placeholder value, replace with actual definition if needed

/-- Molar mass of ethylene in g/mol -/
def molar_mass_ethylene : ℝ := 28

/-- Molar mass of cyclopropane in g/mol -/
def molar_mass_cyclopropane : ℝ := 42

/-- Number of carbon atoms in an ethylene molecule -/
def carbon_atoms_ethylene : ℕ := 2

/-- Number of carbon atoms in a cyclopropane molecule -/
def carbon_atoms_cyclopropane : ℕ := 3

/-- Mass of the mixed gas in grams -/
def mixed_gas_mass : ℝ := 5.6

/-- Theorem stating that the mixed gas contains 0.4 N_A carbon atoms -/
theorem mixed_gas_carbon_atoms :
  ∃ (mass_ethylene mass_cyclopropane : ℝ),
    mass_ethylene + mass_cyclopropane = mixed_gas_mass ∧
    (mass_ethylene / molar_mass_ethylene * carbon_atoms_ethylene +
     mass_cyclopropane / molar_mass_cyclopropane * carbon_atoms_cyclopropane) * N_A = 0.4 * N_A :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_gas_carbon_atoms_l1121_112174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1121_112168

/-- The function f(x) = (-2x^2 + x - 3) / x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := (-2 * x^2 + x - 3) / x

/-- Theorem stating that f(x) has a maximum value of 1 - 2√6 for x > 0 -/
theorem f_max_value :
  ∀ x : ℝ, x > 0 → f x ≤ 1 - 2 * Real.sqrt 6 ∧
  ∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = 1 - 2 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1121_112168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_intersection_l1121_112167

/-- Two lines in a 2D plane -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Definition of intersection point -/
def intersect_at (l1 l2 : Line) (x y : ℚ) : Prop :=
  l1.a * x + l1.b * y = l1.c ∧ l2.a * x + l2.b * y = l2.c

/-- The main theorem -/
theorem perpendicular_lines_intersection :
  let line1 : Line := ⟨4, -3, 16⟩
  let line2 : Line := ⟨3, 4, 15⟩
  perpendicular line1 line2 ∧ intersect_at line1 line2 (12/25) (109/25) := by
  sorry

#check perpendicular_lines_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_intersection_l1121_112167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1121_112132

theorem diophantine_equation_solutions :
  ∀ (x y : ℕ), (3 : ℤ)^x - (2 : ℤ)^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l1121_112132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1121_112145

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 4) / (abs x - 5)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ (x ≥ 4 ∧ x ≠ 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1121_112145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_pythagorean_theorem_l1121_112128

-- Define a right spherical triangle
structure RightSphericalTriangle where
  a : ℝ  -- leg
  b : ℝ  -- leg
  c : ℝ  -- hypotenuse
  right_angle : c > a ∧ c > b  -- condition for right angle in spherical triangle

-- Define the tangent function
noncomputable def tg (x : ℝ) : ℝ := Real.tan x

-- State the theorem
theorem spherical_pythagorean_theorem (t : RightSphericalTriangle) : 
  (tg t.c)^2 = (tg t.a)^2 + (tg t.b)^2 + (tg t.a)^2 * (tg t.b)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_pythagorean_theorem_l1121_112128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_extrema_l1121_112142

/-- The function f satisfying the given differential equation and initial condition -/
noncomputable def f : ℝ → ℝ := sorry

/-- The domain of f is (0, +∞) -/
axiom f_domain : ∀ x : ℝ, x > 0 → f x ≠ 0

/-- f satisfies the differential equation xf'(x) - f(x) = x ln x -/
axiom f_diff_eq : ∀ x : ℝ, x > 0 → x * (deriv f x) - f x = x * Real.log x

/-- Initial condition: f(1/e) = 1/e -/
axiom f_initial : f (1 / Real.exp 1) = 1 / Real.exp 1

/-- Theorem: f has neither a maximum nor a minimum value -/
theorem f_no_extrema : ¬(∃ x : ℝ, x > 0 ∧ (IsLocalMax f x ∨ IsLocalMin f x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_extrema_l1121_112142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l1121_112190

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Defines the given polar curve -/
noncomputable def polar_curve (p : PolarPoint) : Prop :=
  p.ρ = 4 * Real.sin (p.θ - Real.pi/3)

/-- Defines the line of symmetry -/
noncomputable def symmetry_line : ℝ := 5*Real.pi/6

/-- States that the curve is symmetric with respect to the given line -/
theorem curve_symmetry :
  ∀ (p : PolarPoint), polar_curve p →
    ∃ (q : PolarPoint), polar_curve q ∧
      q.ρ = p.ρ ∧ q.θ = 2*symmetry_line - p.θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_l1121_112190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1121_112103

noncomputable section

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the area of a triangle given its base and height
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Define the theorem
theorem ellipse_problem (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_C 0 1 a b) (h4 : eccentricity a b = Real.sqrt 2 / 2) :
  -- Part I: The equation of C is x^2/2 + y^2 = 1
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  -- Part II: The maximum area of triangle OAB is √2/2
  (∃ M : ℝ, M = Real.sqrt 2 / 2 ∧
    ∀ k : ℝ, ∃ A B : ℝ × ℝ,
      -- A and B are on the ellipse C
      ellipse_C A.1 A.2 a b ∧ ellipse_C B.1 B.2 a b ∧
      -- A and B are on the line y = kx + 1
      A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧
      -- The area of triangle OAB is less than or equal to M
      triangle_area ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt (1 / (k^2 + 1).sqrt) ≤ M) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1121_112103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1121_112109

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem f_max_value :
  ∃ (max_val : ℝ) (max_x : ℝ), 
    (∀ x > 1, f x ≤ max_val) ∧
    (max_x > 1) ∧
    (f max_x = max_val) ∧
    max_val = 1 ∧
    max_x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1121_112109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_131_decay_l1121_112181

/-- The half-life of iodine-131 in days -/
noncomputable def half_life : ℝ := 8

/-- The number of days between October 1st and October 25th -/
noncomputable def time_span : ℝ := 24

/-- The amount of iodine-131 remaining after the time span in milligrams -/
noncomputable def remaining_amount : ℝ := 2

/-- The number of half-lives that occur during the time span -/
noncomputable def num_half_lives : ℝ := time_span / half_life

/-- The initial amount of iodine-131 in milligrams -/
noncomputable def initial_amount : ℝ := 16

/-- Theorem stating the relationship between initial amount, decay rate, and remaining amount -/
theorem iodine_131_decay :
  initial_amount * (1/2)^num_half_lives = remaining_amount := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_131_decay_l1121_112181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_52_50_l1121_112195

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $60 -/
axiom total_price : chair_price + table_price = 60

/-- The theorem stating that the price of 1 table is $52.50 -/
theorem table_price_is_52_50 : table_price = 52.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_52_50_l1121_112195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_specific_polynomial_l1121_112126

/-- Represents a polynomial in one variable -/
structure MyPolynomial (α : Type*) where
  coeffs : List α

/-- Horner's method for polynomial evaluation -/
def horner_eval {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.coeffs.foldl (fun acc a => acc * x + a) 0

/-- Counts the number of operations in Horner's method -/
def horner_operations (degree : Nat) : Nat × Nat :=
  (degree, degree)

theorem horner_operations_for_specific_polynomial :
  let f : MyPolynomial ℤ := ⟨[2, 0, -1, 0, 0, 4]⟩  -- 4x^5 - x^2 + 2
  let x : ℤ := 3
  let (mults, adds) := horner_operations 5
  mults = 5 ∧ adds = 2 := by
  sorry

#eval horner_operations 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_specific_polynomial_l1121_112126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1121_112116

/-- Given a parabola C defined by y = x^2 and a point (a, a^2) in the first quadrant,
    prove that the tangent line to C at (a, a^2) intersects the y-axis at (0, -a^2) -/
theorem tangent_line_intersection (a : ℝ) (h : a > 0) :
  let C := fun x : ℝ => x^2
  let tangent_line := fun x : ℝ => 2*a*x - a^2
  tangent_line 0 = -a^2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_l1121_112116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_15_factorial_base_25_is_one_l1121_112115

/-- The number of trailing zeros in the base 25 representation of 15! -/
def trailing_zeros_15_factorial_base_25 : ℕ :=
  let factorial_15 := (Nat.factorial 15)
  let base := 25
  Nat.log base (factorial_15 % (base ^ (Nat.log base factorial_15 + 1)))

/-- Theorem stating that the number of trailing zeros in the base 25 representation of 15! is 1 -/
theorem trailing_zeros_15_factorial_base_25_is_one :
  trailing_zeros_15_factorial_base_25 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trailing_zeros_15_factorial_base_25_is_one_l1121_112115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_good_regions_formula_max_good_regions_formula_find_k_and_n_l1121_112198

/-- The number of parallel lines -/
def k : ℕ := sorry

/-- The number of intersecting lines -/
def n : ℕ := sorry

/-- The total number of lines -/
def total_lines : ℕ := k + n

/-- The minimum number of good regions -/
def min_good_regions : ℕ := 176

/-- The maximum number of good regions -/
def max_good_regions : ℕ := 221

/-- Condition: k > 1 -/
axiom k_gt_one : k > 1

/-- Condition: Any three lines do not pass through the same point -/
axiom no_triple_intersection : True

/-- Condition: Exactly k lines are parallel -/
axiom k_parallel_lines : True

/-- Condition: All other n lines intersect each other -/
axiom n_intersecting_lines : True

/-- Formula for minimum number of good regions -/
theorem min_good_regions_formula : min_good_regions = (n + 1) * (k - 1) := by sorry

/-- Formula for maximum number of good regions -/
theorem max_good_regions_formula : max_good_regions = 2 * (k - 1) + (n - 1) * k + (n - 2) * (n - 1) / 2 := by sorry

/-- The main theorem to prove -/
theorem find_k_and_n : k = 17 ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_good_regions_formula_max_good_regions_formula_find_k_and_n_l1121_112198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1121_112119

/-- The complex number resulting from the multiplication of (1+3i) and (3-i) -/
def z : ℂ := (1 + 3 * Complex.I) * (3 - Complex.I)

/-- A point is in the first quadrant if its real and imaginary parts are both positive -/
def is_in_first_quadrant (c : ℂ) : Prop := 0 < c.re ∧ 0 < c.im

/-- Theorem stating that z is in the first quadrant -/
theorem z_in_first_quadrant : is_in_first_quadrant z := by
  -- Expand the definition of z
  have h1 : z = 6 + 8 * Complex.I := by
    -- Calculation steps
    sorry
  
  -- Show that the real part is positive
  have h_re : 0 < z.re := by
    -- Proof that 6 > 0
    sorry
  
  -- Show that the imaginary part is positive
  have h_im : 0 < z.im := by
    -- Proof that 8 > 0
    sorry
  
  -- Combine the results
  exact ⟨h_re, h_im⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_first_quadrant_l1121_112119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_root_difference_max_value_range_l1121_112127

-- Define the function f(x) = mx^2 + (m+4)x + 3
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m + 4) * x + 3

-- Part 1
theorem min_root_difference (m : ℝ) :
  (∃ x₁ x₂, f m x₁ = 0 ∧ f m x₂ = 0 ∧ x₁ ≠ x₂) →
  (∀ m' : ℝ, (∃ x₁' x₂', f m' x₁' = 0 ∧ f m' x₂' = 0 ∧ x₁' ≠ x₂') →
    |x₁ - x₂| ≤ |x₁' - x₂'|) →
  m = 8 ∧ |x₁ - x₂| = Real.sqrt 3 / 2 :=
sorry

-- Part 2
theorem max_value_range (l : ℝ) (h : l > 0) :
  (l < 3/2 → ∀ x ∈ Set.Icc 0 l, f (-1) x ≤ -l^2 + 3*l + 3) ∧
  (l ≥ 3/2 → ∀ x ∈ Set.Icc 0 l, f (-1) x ≤ 21/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_root_difference_max_value_range_l1121_112127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1121_112112

theorem equation_solutions (x : ℝ) : 
  (8:ℝ)^x + (27:ℝ)^x = (7:ℝ)/6 * ((12:ℝ)^x + (18:ℝ)^x) ↔ x = -1 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1121_112112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_uniqueness_l1121_112163

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define membership for Point in Circle
instance : Membership Point Circle where
  mem p c := (p.x - c.center.1)^2 + (p.y - c.center.2)^2 = c.radius^2

-- Define the properties of the given points
def is_altitude_intersection (c : Circle) (t : Triangle) (p : Point) : Prop :=
  sorry

def is_angle_bisector_intersection (c : Circle) (t : Triangle) (p : Point) : Prop :=
  sorry

def is_median_intersection (c : Circle) (t : Triangle) (p : Point) : Prop :=
  sorry

-- Main theorem
theorem triangle_construction_uniqueness 
  (c : Circle) (M N P : Point) : 
  (M ∈ c) → (N ∈ c) → (P ∈ c) →
  ∃! (t : Triangle),
    (t.A ∈ c) ∧ (t.B ∈ c) ∧ (t.C ∈ c) ∧
    (is_altitude_intersection c t M ∨ 
     is_angle_bisector_intersection c t N ∨ 
     is_median_intersection c t P) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_uniqueness_l1121_112163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1121_112140

/-- The curve function f(x) = x³ - 2x -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

theorem tangent_line_equation (x₀ : ℝ) :
  (x₀ = 1 ∨ x₀ = -1/2) →
  ((1 : ℝ) - (-1 : ℝ) - 2 = 0) ∨ 
  (5*(1 : ℝ) + 4*(-1 : ℝ) - 1 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1121_112140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_and_fourth_meet_l1121_112110

-- Define a type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for lines in a plane
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a type for moving points
structure MovingPoint where
  line : Line
  speed : ℝ

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if three lines are concurrent
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  ∃ (p : Point), (p.y = l1.slope * p.x + l1.intercept) ∧
                 (p.y = l2.slope * p.x + l2.intercept) ∧
                 (p.y = l3.slope * p.x + l3.intercept)

-- Define a function to check if two moving points intersect
def intersect (p1 p2 : MovingPoint) : Prop :=
  ∃ (t : ℝ), ∃ (x y : ℝ),
    x = p1.line.slope * t + p1.line.intercept ∧
    y = p2.line.slope * t + p2.line.intercept

-- Main theorem
theorem third_and_fourth_meet
  (l1 l2 l3 l4 : Line)
  (p1 p2 p3 p4 : MovingPoint)
  (h1 : p1.line = l1 ∧ p2.line = l2 ∧ p3.line = l3 ∧ p4.line = l4)
  (h2 : ∀ (i j : Fin 4), i ≠ j → ¬are_parallel ([l1, l2, l3, l4].get i) ([l1, l2, l3, l4].get j))
  (h3 : ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k →
        ¬are_concurrent ([l1, l2, l3, l4].get i) ([l1, l2, l3, l4].get j) ([l1, l2, l3, l4].get k))
  (h4 : intersect p1 p2 ∧ intersect p1 p3 ∧ intersect p1 p4)
  (h5 : intersect p2 p3 ∧ intersect p2 p4)
  : intersect p3 p4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_and_fourth_meet_l1121_112110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l1121_112133

theorem sum_of_powers (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) :
  (Finset.range 17).sum (λ k => ω^(17 + 3*k)) = ω :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_l1121_112133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_combined_set_l1121_112171

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def is_odd_multiple_of_7_less_than_100 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * (2 * k + 1) ∧ n < 100

def combined_set : Set ℕ :=
  {n | is_two_digit_prime n ∨ is_odd_multiple_of_7_less_than_100 n}

theorem range_of_combined_set :
  ∃ max min : ℕ, max ∈ combined_set ∧ min ∈ combined_set ∧
    (∀ n ∈ combined_set, min ≤ n ∧ n ≤ max) ∧
    max - min = 86 :=
by
  -- We know that 97 is the maximum and 11 is the minimum
  use 97, 11
  sorry -- The detailed proof is omitted

#eval 97 - 11 -- This should output 86

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_combined_set_l1121_112171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_smart_person_l1121_112120

/-- The maximum number of fools in a group of 30 people that allows identifying at least one smart person -/
def max_fools : ℕ := 8

theorem round_table_smart_person (n : ℕ) (hn : n = 30) (f : ℕ) (hf : f ≤ max_fools) :
  ∃ (smart : Fin n → Bool), ∃ (i : Fin n), smart i = true :=
by
  -- We assume the existence of a function 'smart' that maps each person to their intelligence status
  -- The proof would involve showing that given f ≤ 8, we can always find a smart person
  sorry

#check round_table_smart_person

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_smart_person_l1121_112120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_cases_1995_l1121_112157

/-- Represents the number of disease cases over time -/
structure DiseaseCases where
  year : ℕ
  cases : ℕ

/-- Represents the rate of decrease in cases per year -/
abbrev DecreaseRate := ℕ

/-- The disease progression model -/
structure DiseaseModel where
  initial : DiseaseCases
  final : DiseaseCases
  vaccineYear : ℕ
  initialRate : DecreaseRate
  reducedRate : DecreaseRate

/-- Calculate the number of cases for a given year -/
def calculateCases (model : DiseaseModel) (year : ℕ) : ℕ :=
  sorry

theorem disease_cases_1995 (model : DiseaseModel) :
  model.initial.year = 1970 ∧
  model.initial.cases = 600000 ∧
  model.final.year = 2000 ∧
  model.final.cases = 600 ∧
  model.vaccineYear = 1990 ∧
  model.reducedRate = model.initialRate / 2 →
  calculateCases model 1995 = 60580 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_cases_1995_l1121_112157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_is_ellipse_AC_passes_through_fixed_point_l1121_112147

-- Define the points and constants
noncomputable def F₁ : ℝ × ℝ := (-1, 0)
noncomputable def F₂ : ℝ × ℝ := (1, 0)
noncomputable def d : ℝ := 2 * Real.sqrt 2

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the condition for point P
def is_valid_P (P : ℝ × ℝ) : Prop :=
  ∃ M : ℝ × ℝ, 
    dist M F₂ = d ∧ 
    dist P M = dist P F₁ ∧ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₂ + t • M)

-- Theorem 1: The locus of P is the ellipse
theorem locus_of_P_is_ellipse :
  ∀ P : ℝ × ℝ, is_valid_P P ↔ ellipse P.1 P.2 :=
sorry

-- Define a line through F₂
def line_through_F₂ (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

-- Define the intersection points A and B
noncomputable def intersect_ellipse_line (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
sorry

-- Define point C
def point_C (B : ℝ × ℝ) : ℝ × ℝ := (2, B.2)

-- Define line AC
noncomputable def line_AC (A C : ℝ × ℝ) (x : ℝ) : ℝ :=
sorry

-- Theorem 2: AC always passes through (3/2, 0)
theorem AC_passes_through_fixed_point :
  ∀ k : ℝ, k ≠ 0 →
    let (A, B) := intersect_ellipse_line k
    let C := point_C B
    line_AC A C (3/2) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_P_is_ellipse_AC_passes_through_fixed_point_l1121_112147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_probability_l1121_112179

/-- The probability that a randomly selected point inside the circumscribed sphere of a regular tetrahedron
    lies within one of the five smaller spheres (one inscribed and four tangent to faces) is 5/27. -/
theorem tetrahedron_sphere_probability (s : ℝ) (s_pos : s > 0) : 
  let R := s * Real.sqrt 6 / 4  -- Radius of circumscribed sphere
  let r := s * Real.sqrt 6 / 12 -- Radius of inscribed sphere and tangent spheres
  let V_large := (4 / 3) * Real.pi * R^3
  let V_small := (4 / 3) * Real.pi * r^3
  5 * V_small / V_large = 5 / 27 := by
  sorry

#check tetrahedron_sphere_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_sphere_probability_l1121_112179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1121_112102

/-- The area of the common region between two overlapping squares -/
noncomputable def common_area (side_length : ℝ) (β : ℝ) : ℝ :=
  2 / 5 * side_length ^ 2

theorem overlapping_squares_area (side_length β : ℝ) 
  (h1 : side_length = 2)
  (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.cos β = 3/5) :
  common_area side_length β = 2/5 := by
  sorry

#check overlapping_squares_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l1121_112102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_alpha_l1121_112151

/-- Given that the terminal side of angle α passes through point P(4, -3), prove that cos(-α) = 4/5 -/
theorem cos_negative_alpha (α : ℝ) (h : ∃ (t : ℝ), t > 0 ∧ t * (Real.cos α) = 4 ∧ t * (Real.sin α) = -3) :
  Real.cos (-α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_alpha_l1121_112151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_five_sunday_months_l1121_112122

def is_leap_year (y : ℕ) : Bool := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)

theorem max_five_sunday_months : ∃ (year : ℕ), 
  let days_in_year := if is_leap_year year then 366 else 365
  let total_sundays := (days_in_year / 7 : ℕ) + (if days_in_year % 7 > 0 then 1 else 0)
  let months_with_five_sundays := total_sundays - 48
  months_with_five_sundays = 5 ∧ 
  ∀ (y : ℕ), 
    let days := if is_leap_year y then 366 else 365
    let sundays := (days / 7 : ℕ) + (if days % 7 > 0 then 1 else 0)
    sundays - 48 ≤ 5 :=
by
  sorry

#check max_five_sunday_months

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_five_sunday_months_l1121_112122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_20s_l1121_112196

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

/-- Theorem: A train of length 250 m traveling at 72 kmph takes approximately 20 seconds to cross a platform of length 150.03 m -/
theorem train_crossing_approx_20s :
  ∃ ε > 0, |train_crossing_time 250 72 150.03 - 20| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_approx_20s_l1121_112196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_range_of_a_l1121_112189

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_zero (x : ℝ) : 
  ∃ (m b : ℝ), (∀ y, y = m * x + b) → 
  (∀ ε > 0, ∃ δ > 0, ∀ h, |h| < δ → |f 1 (0 + h) - (f 1 0 + m * h)| ≤ ε * |h|) →
  m = 0 ∧ b = 0 := by sorry

-- Part 2: Range of a for non-negative f(x) on [0, 1)
theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Set.Ico 0 1 → f a x ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_range_of_a_l1121_112189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1121_112161

/-- The function g(x) defined as 4 / (5x^8 - 7) -/
noncomputable def g (x : ℝ) : ℝ := 4 / (5 * x^8 - 7)

/-- Theorem stating that g(x) is an even function -/
theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  simp [g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l1121_112161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_l1121_112134

theorem sum_reciprocals (a b c d : ℂ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  a ≠ -Complex.I → b ≠ -Complex.I → c ≠ -Complex.I → d ≠ -Complex.I →
  ω^3 = 1 → ω ≠ 1 →
  let ψ := ω^2
  (a + ψ)⁻¹ + (b + ψ)⁻¹ + (c + ψ)⁻¹ + (d + ψ)⁻¹ = 4 / ω →
  (a + 1)⁻¹ + (b + 1)⁻¹ + (c + 1)⁻¹ + (d + 1)⁻¹ = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_l1121_112134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_regular_triangular_pyramid_l1121_112136

/-- A regular triangular pyramid with a circumscribed sphere of radius 1 -/
structure RegularTriangularPyramid where
  -- The radius of the circumscribed sphere is 1
  circumscribed_sphere_radius : ℝ := 1
  -- The height of the pyramid
  height : ℝ
  -- The radius of the base circle
  base_radius : ℝ
  -- Relation between height and base radius
  height_base_relation : base_radius^2 = 2 * height - height^2
  -- Height is between 0 and 2
  height_bounds : 0 < height ∧ height < 2

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ :=
  (Real.sqrt 3 / 4) * p.base_radius^2 * p.height

/-- The maximum volume of a regular triangular pyramid with circumscribed sphere radius 1 -/
theorem max_volume_regular_triangular_pyramid :
  ∃ (p : RegularTriangularPyramid), ∀ (q : RegularTriangularPyramid), volume p ≥ volume q ∧ volume p = 8 * Real.sqrt 3 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_regular_triangular_pyramid_l1121_112136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_theorem_l1121_112101

noncomputable def area1 : ℝ := ∫ x in (0:ℝ)..(3:ℝ), (x^2 + 1)

noncomputable def area2 : ℝ := (1/2) * ∫ y in (-2:ℝ)..(3:ℝ), y^2

noncomputable def area3 : ℝ := ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt x - x^2)

noncomputable def area4 : ℝ := 2 * (∫ y in (0:ℝ)..(5:ℝ), Real.sqrt (2*y) - ∫ y in (1:ℝ)..(5:ℝ), Real.sqrt (y-1))

theorem areas_theorem :
  area1 = 12 ∧
  area2 = 35/6 ∧
  area3 = 1/3 ∧
  area4 = (4/3) * (5 * Real.sqrt 10 - 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_areas_theorem_l1121_112101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_correct_answers_l1121_112160

/-- Represents a mathematics contest with the given rules and outcomes. -/
structure MathContest where
  total_problems : Nat
  correct_points : Int
  incorrect_points : Int
  final_score : Int

/-- Calculates the number of correct answers in a math contest. -/
def correct_answers (contest : MathContest) : Int :=
  let total_points := contest.correct_points - contest.incorrect_points
  (contest.final_score + contest.incorrect_points * contest.total_problems) / total_points

/-- Theorem stating that given the specific contest conditions, the number of correct answers is 8. -/
theorem contest_correct_answers :
  let contest := MathContest.mk 12 6 (-3) 36
  correct_answers contest = 8 := by
  sorry

#eval correct_answers (MathContest.mk 12 6 (-3) 36)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_correct_answers_l1121_112160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_square_side_length_of_square_area_of_square_is_two_l1121_112197

/-- Regular 4000-gon with specific properties -/
structure RegularPolygon4000 where
  -- Vertices of the 4000-gon
  A : Fin 4000 → ℝ × ℝ
  -- X is the foot of the altitude from A₁₉₈₆ onto diagonal A₁₀₀₀A₃₀₀₀
  X : ℝ × ℝ
  -- Y is the foot of the altitude from A₂₀₁₄ onto A₂₀₀₀A₄₀₀₀
  Y : ℝ × ℝ
  -- The distance between X and Y is 1
  xy_distance : dist X Y = 1
  -- Ensure the polygon is regular
  is_regular : ∀ i j : Fin 4000, dist (A i) (A j) = dist (A 0) (A 1)

/-- The area of square A₅₀₀A₁₅₀₀A₂₅₀₀A₃₅₀₀ in the given regular 4000-gon is 2 -/
theorem area_of_square (p : RegularPolygon4000) : 
  Real.sqrt ((p.A 500).1 - (p.A 2500).1)^2 + ((p.A 500).2 - (p.A 2500).2)^2 = Real.sqrt 2 := by
  sorry

/-- The side length of the square is √2 -/
theorem side_length_of_square (p : RegularPolygon4000) :
  dist (p.A 500) (p.A 1500) = Real.sqrt 2 := by
  sorry

/-- The area of square A₅₀₀A₁₅₀₀A₂₅₀₀A₃₅₀₀ in the given regular 4000-gon is 2 -/
theorem area_of_square_is_two (p : RegularPolygon4000) :
  (dist (p.A 500) (p.A 1500))^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_square_side_length_of_square_area_of_square_is_two_l1121_112197
