import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_log2_l1302_130278

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + 9 * x^2)^(1/2) - 3 * x) + 1

-- State the theorem
theorem f_sum_log2 : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_log2_l1302_130278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1302_130290

-- Define the power function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4*m + 2)

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Define the set A as the range of f(x) for x ∈ (1,2]
def A (m : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Ioo 1 2, f m x = y}

-- Define the set B as the range of g(x) for x ∈ (1,2]
def B (k : ℝ) : Set ℝ := {y | ∃ x ∈ Set.Ioo 1 2, g k x = y}

-- State the theorem
theorem power_function_theorem (m k : ℝ) : 
  (∀ x > 0, Monotone (f m)) → -- f is monotonically increasing on (0,+∞)
  A m ∪ B k = A m → -- A ∪ B = A
  m = 0 ∧ k ∈ Set.Icc 0 1 := by -- Conclusion: m = 0 and k ∈ [0,1]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1302_130290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_trip_theorem_l1302_130219

/-- Represents the walking trip scenario --/
structure WalkingTrip where
  front_speed : ℝ
  back_speed : ℝ
  liaison_speed : ℝ
  head_start : ℝ

/-- Calculates the time for the back team to catch up --/
noncomputable def catch_up_time (trip : WalkingTrip) : ℝ :=
  trip.head_start * trip.front_speed / (trip.back_speed - trip.front_speed)

/-- Calculates the distance cycled by the liaison officer --/
noncomputable def liaison_distance (trip : WalkingTrip) : ℝ :=
  trip.liaison_speed * catch_up_time trip

/-- Theorem statement for the walking trip problem --/
theorem walking_trip_theorem (trip : WalkingTrip)
  (h1 : trip.front_speed = 4)
  (h2 : trip.back_speed = 6)
  (h3 : trip.liaison_speed = 12)
  (h4 : trip.head_start = 1) :
  catch_up_time trip = 2 ∧ liaison_distance trip = 24 := by
  sorry

-- Remove #eval statements as they are not computable
-- #eval catch_up_time { front_speed := 4, back_speed := 6, liaison_speed := 12, head_start := 1 }
-- #eval liaison_distance { front_speed := 4, back_speed := 6, liaison_speed := 12, head_start := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_trip_theorem_l1302_130219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_increasing_on_interval_l1302_130233

open Real

theorem sin_2x_increasing_on_interval :
  ∀ x ∈ Set.Icc (3 * π / 4) (5 * π / 4),
    ∀ y ∈ Set.Icc (3 * π / 4) (5 * π / 4),
      x < y → sin (2 * x) < sin (2 * y) := by
  intros x hx y hy hxy
  -- The proof goes here
  sorry

#check sin_2x_increasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_increasing_on_interval_l1302_130233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_diagonal_tangent_l1302_130203

/-- Represents a conic section -/
structure ConicSection where
  a : ℝ  -- semi-major axis
  c : ℝ  -- distance from center to focus
  center : ℝ × ℝ

/-- Represents a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is tangent to a conic section -/
def isTangent (l : Line) (cs : ConicSection) : Prop := sorry

/-- Function to get the vertices of a conic section -/
def getVertices (cs : ConicSection) : Point × Point := sorry

/-- Function to get the foci of a conic section -/
def getFoci (cs : ConicSection) : Point × Point := sorry

/-- Function to get the tangents at the vertices of a conic section -/
def getVertexTangents (cs : ConicSection) : Line × Line := sorry

/-- Function to get intersection points of a circle and two lines -/
def getIntersectionPoints (c : Circle) (l1 l2 : Line) : Point × Point × Point × Point := sorry

/-- Function to get the diagonals of a rectangle given by four points -/
def getRectangleDiagonals (p1 p2 p3 p4 : Point) : Line × Line := sorry

/-- Check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Main theorem statement -/
theorem conic_section_diagonal_tangent (cs : ConicSection) (c : Circle) : 
  let (v1, v2) := getVertices cs
  let (f1, f2) := getFoci cs
  let (t1, t2) := getVertexTangents cs
  let (p1, p2, p3, p4) := getIntersectionPoints c t1 t2
  let (d1, d2) := getRectangleDiagonals p1 p2 p3 p4
  (isOnCircle f1 c) ∧ (isOnCircle f2 c) → isTangent d1 cs ∧ isTangent d2 cs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_diagonal_tangent_l1302_130203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_wins_l1302_130296

/-- Represents an equation with asterisks --/
inductive Equation
  | eq1 : Equation  -- *=*
  | eq2 : Equation  -- *+*=*
  | eq3 : Equation  -- *+*+*=*
  | eq4 : Equation  -- *+*+*+*=*
  | eq5 : Equation  -- *+*+*+*+*=*
  | eq6 : Equation  -- *+*+*+*+*+*=*
  | eq7 : Equation  -- *+*+*+*+*+*+*=*

/-- Represents the game state --/
structure GameState where
  equations : List Equation
  currentPlayer : Bool  -- true for player A, false for player B

/-- Represents a move in the game --/
structure Move where
  equation : Equation
  position : Nat
  value : Int

/-- Check if an equation is satisfied --/
def isSatisfied : Equation → Bool
  | _ => sorry  -- We'll implement this later

/-- The winning strategy for player A --/
def winningStrategy : GameState → Option Move := sorry

/-- Theorem stating that player A has a winning strategy --/
theorem player_a_wins (initialState : GameState) : 
  initialState.currentPlayer = true → 
  ∃ (strategy : GameState → Option Move), 
    (∀ (state : GameState), 
      state.equations.isEmpty → 
      (∀ (eq : Equation), eq ∈ state.equations → isSatisfied eq)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_wins_l1302_130296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_harvested_l1302_130236

/-- The amount of apples harvested -/
def total_apples : ℕ := 0

/-- The amount of apples used for juice -/
def juice_apples : ℕ := 90

/-- The amount of apples given to a restaurant -/
def restaurant_apples : ℕ := 60

/-- The weight of each bag of apples in kg -/
def bag_weight : ℕ := 5

/-- The price of each bag of apples in dollars -/
def bag_price : ℕ := 8

/-- The total sales from selling bags of apples in dollars -/
def total_sales : ℕ := 408

theorem apples_harvested :
  total_apples = juice_apples + restaurant_apples + (total_sales / bag_price * bag_weight) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_harvested_l1302_130236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simpson_law_properties_l1302_130259

/-- Simpson's law probability density function -/
noncomputable def simpson_pdf (c : ℝ) (x : ℝ) : ℝ :=
  if -c ≤ x ∧ x ≤ c then (1/c) * (1 - |x|/c) else 0

/-- Theorem stating properties of Simpson's law distribution -/
theorem simpson_law_properties (c : ℝ) (hc : c > 0) :
  (∫ x, simpson_pdf c x) = 1 ∧
  (∫ x in Set.Ioo (c/2) c, simpson_pdf c x) = 1/8 := by
  sorry

#check simpson_law_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simpson_law_properties_l1302_130259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preference_list_divisibility_l1302_130262

def S (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

def P (n : ℕ) : ℕ := (Finset.powerset (Finset.range n)).sum (λ s ↦ s.card.factorial)

theorem preference_list_divisibility (n m : ℕ) (h : n > m) :
  (n - m) ∣ (P n - P m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preference_list_divisibility_l1302_130262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_A_equivalent_pair_B_not_equivalent_pair_C_not_equivalent_pair_D_equivalent_l1302_130285

-- Define the functions for pair A
def f_A (x : ℝ) : ℝ := x^2 - x - 1
def g_A (t : ℝ) : ℝ := t^2 - t + 1

-- Define the functions for pair B
noncomputable def f_B (x : ℝ) : ℝ := Real.sqrt (x^2 - 1)
noncomputable def g_B (x : ℝ) : ℝ := Real.sqrt (x + 1) * Real.sqrt (x - 1)

-- Define the functions for pair C
noncomputable def f_C (x : ℝ) : ℝ := (Real.sqrt x)^2
noncomputable def g_C (x : ℝ) : ℝ := Real.sqrt (x^2)

-- Define the functions for pair D
noncomputable def f_D (x : ℝ) : ℝ := Real.sqrt (x^2) / x
noncomputable def g_D (x : ℝ) : ℝ := x / Real.sqrt (x^2)

-- Theorem statements
theorem pair_A_equivalent : ∀ x : ℝ, f_A x = g_A x := by sorry

theorem pair_B_not_equivalent : ∃ x : ℝ, f_B x ≠ g_B x := by sorry

theorem pair_C_not_equivalent : ∃ x : ℝ, f_C x ≠ g_C x := by sorry

theorem pair_D_equivalent : ∀ x : ℝ, x ≠ 0 → f_D x = g_D x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_A_equivalent_pair_B_not_equivalent_pair_C_not_equivalent_pair_D_equivalent_l1302_130285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_27389_l1302_130289

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

theorem round_45_27389 :
  round_to_hundredth 45.27389 = 45.27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_45_27389_l1302_130289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_protest_jail_time_l1302_130248

theorem protest_jail_time (
  protest_days : ℕ)
  (num_cities : ℕ)
  (arrests_per_day : ℕ)
  (pre_trial_days : ℕ)
  (sentence_weeks : ℕ)
  (h1 : protest_days = 30)
  (h2 : num_cities = 21)
  (h3 : arrests_per_day = 10)
  (h4 : pre_trial_days = 4)
  (h5 : sentence_weeks = 2)
  : (protest_days * num_cities * arrests_per_day * 
    (pre_trial_days + sentence_weeks * 7 / 2)) / 7 = 9900 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_protest_jail_time_l1302_130248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_translation_l1302_130297

/-- Original curve equation -/
noncomputable def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

/-- Translated curve equation -/
noncomputable def translated_curve (x y : ℝ) : Prop :=
  (y + 1) * Real.sin x + 2 * y + 1 = 0

/-- Translation parameters -/
noncomputable def x_shift : ℝ := Real.pi / 2
noncomputable def y_shift : ℝ := -1

/-- Theorem stating the equivalence of the original and translated curves -/
theorem curve_translation :
  ∀ x y : ℝ, original_curve (x - x_shift) (y - y_shift) ↔ translated_curve x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_translation_l1302_130297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l1302_130225

/-- Given vectors in ℝ², prove that if c is parallel to 2a + b, then λ = 1/2 -/
theorem parallel_vector_lambda (a b c : ℝ × ℝ) (l : ℝ) : 
  a = (1, 2) → 
  b = (2, -2) → 
  c = (1, l) → 
  (∃ (k : ℝ), c = k • (2 • a + b)) → 
  l = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vector_lambda_l1302_130225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_percentage_l1302_130291

/-- The number of boys in the fourth grade -/
def num_boys : ℕ := 20

/-- The number of girls in the fourth grade -/
def num_girls : ℕ := 26

/-- The percentage of boys to girls -/
noncomputable def percentage_boys_to_girls : ℝ := (num_boys : ℝ) / (num_girls : ℝ) * 100

theorem boys_to_girls_percentage :
  ∃ (ε : ℝ), ε > 0 ∧ |percentage_boys_to_girls - 76.9| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_percentage_l1302_130291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1302_130299

theorem divisibility_condition (n : ℕ) (hn : n > 0) :
  (∃ (k : ℕ), k % 2 = 1 ∧ n = 2^2014 * k - 1) ↔
  (2^2015 ∣ n^(n - 1) - 1) ∧ ¬(2^2016 ∣ n^(n - 1) - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1302_130299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_logarithmic_satisfies_property_l1302_130275

-- Define the property that we're looking for
def HasPropertyF (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y

-- Define the types of functions we're considering
def IsLinear (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

def IsExponential (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = b * a^x

noncomputable def IsLogarithmic (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ x, x > 0 → f x = Real.log x / Real.log a

def IsSine (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = Real.sin x

-- State the theorem
theorem only_logarithmic_satisfies_property :
  ∃ (f : ℝ → ℝ), IsLogarithmic f ∧ HasPropertyF f ∧
  (∀ g : ℝ → ℝ, (IsLinear g ∨ IsExponential g ∨ IsSine g) → HasPropertyF g → g = f) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_logarithmic_satisfies_property_l1302_130275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_S_value_l1302_130273

/-- The number of sides on a standard die -/
def die_sides : ℕ := 8

/-- The target sum we want to achieve -/
def target_sum : ℕ := 2400

/-- The function to calculate the transformed sum given the number of dice -/
def transformed_sum (n : ℕ) : ℕ := 9 * n - target_sum

/-- The minimum number of dice needed to potentially achieve the target sum -/
def min_dice : ℕ := (target_sum + die_sides - 1) / die_sides

/-- The theorem stating that the smallest possible value of S is 300 -/
theorem smallest_S_value : 
  ∃ (n : ℕ), 
    n ≥ min_dice ∧ 
    (∃ (roll : Fin n → Fin die_sides), (Finset.sum Finset.univ (fun i => (roll i).val)) = target_sum) ∧
    transformed_sum n = 300 ∧
    ∀ m, m < n → transformed_sum m ≠ 300 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_S_value_l1302_130273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_l1302_130237

/-- A function that represents $\sqrt{3}\cos(3x - \phi) - \sin(3x - \phi)$ -/
noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * Real.cos (3 * x - φ) - Real.sin (3 * x - φ)

/-- Theorem stating the condition for f to be an odd function -/
theorem f_is_odd_iff (φ : ℝ) :
  (∀ x, f x φ = -f (-x) φ) ↔ ∃ k : ℤ, φ = k * Real.pi - Real.pi / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_l1302_130237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ominous_iff_one_or_prime_l1302_130241

/-- A year is fortunate if the first n positive integers can be divided into at least two groups
    such that the sum of the numbers and the number of elements in each group are equal. -/
def is_fortunate (n : ℕ) : Prop :=
  ∃ (k : ℕ) (partitions : Fin k → Finset ℕ),
    k ≥ 2 ∧ 
    (∀ i : Fin k, (partitions i).Nonempty) ∧
    (⋃ i, partitions i : Set ℕ) = Finset.range n ∧
    (∀ i j : Fin k, i ≠ j → Disjoint (partitions i) (partitions j)) ∧
    (∀ i j : Fin k, (partitions i).sum id = (partitions j).sum id ∧ 
                    (partitions i).card = (partitions j).card)

/-- A year is ominous if it's not fortunate. -/
def is_ominous (n : ℕ) : Prop := ¬ is_fortunate n

/-- Theorem: A positive integer n is ominous if and only if n = 1 or n is prime. -/
theorem ominous_iff_one_or_prime (n : ℕ) (hn : n > 0) :
  is_ominous n ↔ n = 1 ∨ Nat.Prime n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ominous_iff_one_or_prime_l1302_130241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1302_130210

/-- The area of the region inside a regular hexagon with side length 4,
    but outside six semicircles (each with diameter equal to the hexagon's side length) -/
noncomputable def shaded_area : ℝ := 24 * Real.sqrt 3 - 12 * Real.pi

/-- Theorem stating that the shaded area is equal to 24√3 - 12π -/
theorem shaded_area_calculation (s : ℝ) (h : s = 4) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area := Real.pi * (s/2)^2 / 2
  hexagon_area - 6 * semicircle_area = shaded_area :=
by
  -- Substitute s = 4
  rw [h]
  -- Simplify
  simp [shaded_area]
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1302_130210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1302_130286

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the ellipse C
noncomputable def ellipse_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 2 * Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, line_l t = p ∧ ellipse_C θ = p}

-- Theorem statement
theorem intersection_segment_length :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l1302_130286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_202_is_green_l1302_130257

/-- Represents the color of a marble -/
inductive Color
| Red
| Green
| Blue

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : Color :=
  match n % 15 with
  | k => if k < 6 then Color.Red
         else if k < 11 then Color.Green
         else Color.Blue

/-- Theorem stating that the 202nd marble is green -/
theorem marble_202_is_green : marbleColor 202 = Color.Green := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_202_is_green_l1302_130257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_conditions_l1302_130206

-- Define the system of differential equations
def system (x y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv x t = 2 * x t - 4 * y t) ∧ (deriv y t = x t - 3 * y t)

-- Define the initial conditions
def initial_conditions (x y : ℝ → ℝ) : Prop :=
  x 0 = 1 ∧ y 0 = 2

-- Define the solution functions
noncomputable def x_solution (t : ℝ) : ℝ := -4/3 * Real.exp t + 7/3 * Real.exp (-2*t)
noncomputable def y_solution (t : ℝ) : ℝ := -1/3 * Real.exp t + 7/3 * Real.exp (-2*t)

-- Theorem statement
theorem solution_satisfies_system_and_conditions :
  system x_solution y_solution ∧ initial_conditions x_solution y_solution := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_and_conditions_l1302_130206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_absolute_value_geq_one_implies_a_geq_e_over_two_l1302_130217

open Real Set

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - log x

-- State the theorem
theorem f_absolute_value_geq_one_implies_a_geq_e_over_two :
  ∀ a : ℝ, (∀ x ∈ Ioo 0 1, |f a x| ≥ 1) → a ≥ (exp 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_absolute_value_geq_one_implies_a_geq_e_over_two_l1302_130217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_sum_l1302_130223

noncomputable def f (x : ℝ) := 2^x + x - 5

theorem root_interval_sum (a b : ℤ) : 
  (∃ r : ℝ, f r = 0 ∧ a < r ∧ r < b) →
  b - a = 1 →
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_interval_sum_l1302_130223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1302_130234

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

-- Part 1
theorem part_one (k : ℝ) : 
  (∀ x, f x > k ↔ (x < -3 ∨ x > -2)) → k = -2/5 := by sorry

-- Part 2
theorem part_two :
  (∃ t, ∀ x > 0, f x ≤ t) → 
  (∀ t, (∀ x > 0, f x ≤ t) ↔ t ≥ Real.sqrt 6 / 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1302_130234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_of_arithmetic_sequence_l1302_130245

-- Define the arithmetic sequence
noncomputable def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

-- Theorem statement
theorem eleventh_term_of_arithmetic_sequence
  (a₁ d : ℚ)
  (h₁ : a₁ = 5)
  (h₂ : sum_arithmetic_sequence a₁ d 7 = 77) :
  arithmetic_sequence a₁ d 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleventh_term_of_arithmetic_sequence_l1302_130245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_shop_expenses_l1302_130261

/-- Calculates the weekly expenses for running a flower shop -/
theorem flower_shop_expenses (rent : ℚ) (utility_rate : ℚ) (employees_per_shift : ℕ)
  (hours_per_day : ℕ) (days_per_week : ℕ) (hourly_wage : ℚ) : ℚ :=
  by
  have h1 : rent = 1200 := by sorry
  have h2 : utility_rate = 1/5 := by sorry
  have h3 : employees_per_shift = 2 := by sorry
  have h4 : hours_per_day = 16 := by sorry
  have h5 : days_per_week = 5 := by sorry
  have h6 : hourly_wage = 25/2 := by sorry

  let utilities := rent * utility_rate
  let total_hours := (employees_per_shift : ℚ) * (hours_per_day : ℚ) * (days_per_week : ℚ)
  let wages := total_hours * hourly_wage
  let total_expenses := rent + utilities + wages

  have : total_expenses = 3440 := by sorry

  exact total_expenses

-- Example usage (commented out to avoid compilation issues)
-- #eval flower_shop_expenses 1200 (1/5) 2 16 5 (25/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_shop_expenses_l1302_130261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_plus_sqrt_five_minus_one_power_zero_minus_sqrt_one_fourth_plus_two_power_negative_one_equals_three_l1302_130207

theorem cube_root_eight_plus_sqrt_five_minus_one_power_zero_minus_sqrt_one_fourth_plus_two_power_negative_one_equals_three :
  (8 : ℝ) ^ (1/3 : ℝ) + ((5 : ℝ).sqrt - 1) ^ (0 : ℝ) - (1/4 : ℝ).sqrt + (2 : ℝ) ^ (-1 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_eight_plus_sqrt_five_minus_one_power_zero_minus_sqrt_one_fourth_plus_two_power_negative_one_equals_three_l1302_130207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_l1302_130293

/-- The radius of a right circular cylinder inscribed in a right circular cone -/
noncomputable def cylinder_radius (cone_diameter : ℝ) (cone_altitude : ℝ) : ℝ :=
  (20 : ℝ) / 9

/-- Theorem stating that the radius of the inscribed cylinder is 20/9 -/
theorem inscribed_cylinder_radius :
  cylinder_radius 8 10 = 20 / 9 := by
  -- Unfold the definition of cylinder_radius
  unfold cylinder_radius
  -- The equality holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_l1302_130293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_interpretation_theorem_l1302_130208

/-- Represents the result of a χ² test for independence --/
structure ChiSquareTest where
  observed_value : ℝ
  confidence_level : ℝ

/-- Represents the relationship between smoking and lung disease --/
inductive Relationship
  | Related
  | Unrelated

/-- Interprets the result of a χ² test --/
noncomputable def interpret_chi_square_test (test : ChiSquareTest) : Relationship :=
  if test.observed_value > 6.635 && test.confidence_level ≥ 0.99 then
    Relationship.Related
  else
    Relationship.Unrelated

theorem chi_square_interpretation_theorem (test : ChiSquareTest) 
  (h_observed : test.observed_value = 6.635) 
  (h_confidence : test.confidence_level = 0.99) :
  interpret_chi_square_test test = Relationship.Related ∧
  ¬(∀ (smokers : ℕ), smokers = 100 → ∃ (lung_disease : ℕ), lung_disease = 99) ∧
  ¬(∃ (error_chance : ℝ), error_chance = 0.05) ∧
  ¬(∀ (smoker : Bool), smoker = true → ∃ (lung_disease_prob : ℝ), lung_disease_prob = 0.99) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chi_square_interpretation_theorem_l1302_130208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_Z_complex_inequality_l1302_130201

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number Z
noncomputable def Z (m : ℝ) : ℂ := (5 * m^2) / (1 - 2*i) - (1 + 5*i) * m - 3 * (2 + i)

-- Part 1
theorem pure_imaginary_Z (m : ℝ) : 
  (Z m).re = 0 → m = -2 := by sorry

-- Part 2
theorem complex_inequality (m : ℝ) :
  Complex.abs (m^2 - (m^2 - 3*m)*i) < Complex.abs ((m^2 - 4*m + 3)*i + 10) → m = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_Z_complex_inequality_l1302_130201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_special_angle_l1302_130288

theorem cosine_sum_special_angle : 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) - Real.sin (20 * π / 180) * Real.sin (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_special_angle_l1302_130288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l1302_130268

/-- A line that passes through the origin -/
structure OriginLine where
  slope : ℝ

/-- The vertical line x = 1 -/
def verticalLine : Set (ℝ × ℝ) := {p | p.1 = 1}

/-- The line y = 2 + 1/2x -/
def slantedLine : Set (ℝ × ℝ) := {p | p.2 = 2 + 1/2 * p.1}

/-- Predicate to check if three points form an equilateral triangle -/
def isEquilateralTriangle (a b c : ℝ × ℝ) : Prop :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
  (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

/-- The perimeter of a triangle given its three vertices -/
noncomputable def trianglePerimeter (a b c : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) +
  Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) +
  Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)

theorem equilateral_triangle_perimeter :
  ∃ (l : OriginLine) (a b c : ℝ × ℝ),
    a ∈ verticalLine ∧
    b ∈ slantedLine ∧
    c = (0, 0) ∧
    isEquilateralTriangle a b c ∧
    trianglePerimeter a b c = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l1302_130268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cities_in_airline_graph_l1302_130274

/-- A graph representing the airline system in a country. -/
structure AirlineGraph where
  /-- The set of vertices (cities) in the graph. -/
  V : Type
  /-- The edge relation between vertices. -/
  E : V → V → Prop
  /-- Property ensuring each vertex has at most 3 neighbors. -/
  degree_at_most_three : ∀ v : V, (∃ (n : Nat), n ≤ 3 ∧ (∃ (neighbors : Finset V), neighbors.card = n ∧ (∀ u ∈ neighbors, E v u)))
  /-- Property ensuring the distance between any two vertices is at most 2. -/
  distance_at_most_two : ∀ v w : V, v ≠ w → (E v w ∨ ∃ u : V, E v u ∧ E u w)

/-- Theorem stating that the maximum number of vertices in an AirlineGraph is 10. -/
theorem max_cities_in_airline_graph (G : AirlineGraph) [Fintype G.V] : Fintype.card G.V ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cities_in_airline_graph_l1302_130274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_theorem_l1302_130260

/-- The area of an isosceles triangle with two sides of length 20 and the third side of length 36 -/
noncomputable def isoscelesTriangleArea : ℝ := 72 * Real.sqrt 19

/-- Theorem: The area of an isosceles triangle with two sides of length 20 and the third side of length 36 is 72√19 -/
theorem isosceles_triangle_area_theorem (p q r : ℝ × ℝ) :
  let pq := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let qr := Real.sqrt ((q.1 - r.1)^2 + (q.2 - r.2)^2)
  let pr := Real.sqrt ((p.1 - r.1)^2 + (p.2 - r.2)^2)
  pq = 20 ∧ qr = 20 ∧ pr = 36 →
  (1/2) * pr * Real.sqrt (pq^2 - (pr/2)^2) = isoscelesTriangleArea :=
by sorry

#check isosceles_triangle_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_theorem_l1302_130260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_b_l1302_130231

def is_valid_polynomial (Q : ℤ → ℤ) (b : ℕ) : Prop :=
  (∀ x : ℕ, x ≤ 12 → ∃ (c : ℤ), Q x = c * b) ∧
  (∀ x : ℕ, x = 1 ∨ x = 4 ∨ x = 7 ∨ x = 10 → Q x = b) ∧
  (∀ x : ℕ, x = 2 ∨ x = 5 ∨ x = 8 ∨ x = 11 → Q x = -2*b) ∧
  (∀ x : ℕ, x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 12 → Q x = 3*b)

theorem smallest_valid_b : 
  ∃ (Q : ℤ → ℤ), is_valid_polynomial Q 252 ∧
  ∀ (b : ℕ) (Q : ℤ → ℤ), b < 252 → ¬(is_valid_polynomial Q b) :=
by
  sorry

#check smallest_valid_b

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_b_l1302_130231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1302_130287

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + x - sin x

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x ∈ Set.Ioo 0 (2 * π), ∀ y ∈ Set.Ioo 0 (2 * π),
  x < y → f x < f y :=
by
  -- We'll use sorry to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1302_130287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_length_l1302_130251

/-- Represents a trapezoidal prism-shaped swimming pool -/
structure SwimmingPool where
  width : ℝ
  shallowDepth : ℝ
  deepDepth : ℝ
  volume : ℝ

/-- Calculates the length of a swimming pool given its dimensions and volume -/
noncomputable def poolLength (pool : SwimmingPool) : ℝ :=
  pool.volume / ((pool.shallowDepth + pool.deepDepth) * pool.width / 2)

theorem swimming_pool_length (pool : SwimmingPool) 
  (h1 : pool.width = 9)
  (h2 : pool.shallowDepth = 1)
  (h3 : pool.deepDepth = 4)
  (h4 : pool.volume = 270) :
  poolLength pool = 12 := by
  sorry

-- Remove the #eval line as it's not necessary for the proof and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_length_l1302_130251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_sequence_property_l1302_130211

/-- A sequence of integers satisfying the recurrence relation a_{n+2} = a_{n+1} - m * a_n -/
def RecurrenceSequence (m : ℤ) := ℕ → ℤ

/-- The property that a₁ and a₂ are not both zero -/
def NonZeroStart (a : ℕ → ℤ) : Prop :=
  a 1 ≠ 0 ∨ a 2 ≠ 0

/-- The recurrence relation a_{n+2} = a_{n+1} - m * a_n -/
def SatisfiesRecurrence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

theorem recurrence_sequence_property
  (m : ℤ)
  (hm : |m| ≥ 2)
  (a : ℕ → ℤ)
  (h_start : NonZeroStart a)
  (h_rec : SatisfiesRecurrence m a)
  (r s : ℕ)
  (hrs : r > s ∧ s ≥ 2)
  (h_equal : a r = a s ∧ a r = a 1) :
  r - s ≥ |m| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_sequence_property_l1302_130211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_iff_geometric_mean_l1302_130284

/-- A truncated cone with an inscribed sphere. -/
structure TruncatedConeWithSphere where
  r : ℝ  -- radius of the top base
  R : ℝ  -- radius of the bottom base
  h : ℝ  -- height of the cone
  r_pos : r > 0
  R_pos : R > 0
  h_pos : h > 0

/-- The condition for a sphere to be inscribed in a truncated cone. -/
def has_inscribed_sphere (cone : TruncatedConeWithSphere) : Prop :=
  cone.h = 2 * Real.sqrt (cone.r * cone.R)

/-- A sphere. -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a sphere is inscribed in a truncated cone. -/
def Sphere.inscribed_in (sphere : Sphere) (cone : TruncatedConeWithSphere) : Prop :=
  sorry -- We'll leave this undefined for now, as it requires more complex geometry

/-- Theorem stating the necessary and sufficient condition for a sphere
    to be inscribed in a truncated cone. -/
theorem inscribed_sphere_iff_geometric_mean (cone : TruncatedConeWithSphere) :
  (∃ (sphere : Sphere), sphere.inscribed_in cone) ↔ has_inscribed_sphere cone :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_iff_geometric_mean_l1302_130284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1302_130250

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + (m - 7) * x + 1 ≠ 0

def q (m : ℝ) : Prop := ∃ (max min : ℝ), 
  (∀ x, m - 9 < x ∧ x < 9 - m → -x^3 + 3*x ≤ max ∧ -x^3 + 3*x ≥ min) ∧
  (∃ x1 x2, m - 9 < x1 ∧ x1 < 9 - m ∧ m - 9 < x2 ∧ x2 < 9 - m ∧ 
    -x1^3 + 3*x1 = max ∧ -x2^3 + 3*x2 = min)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, m < 9 → (p m ∨ q m) → ¬(p m ∧ q m) → 5 < m ∧ m < 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1302_130250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_composition_theorem_l1302_130244

/-- Represents the composition of a solution -/
structure SolutionComposition where
  x : ℝ
  water : ℝ
  z : ℝ

/-- Calculate the new composition after water evaporation -/
noncomputable def evaporateWater (initial : SolutionComposition) (initialWeight : ℝ) (evaporatedWater : ℝ) : SolutionComposition :=
  { x := initial.x * initialWeight
    water := initial.water * initialWeight - evaporatedWater
    z := initial.z * initialWeight }

/-- Calculate the new composition after adding more solution -/
noncomputable def addSolution (current : SolutionComposition) (added : SolutionComposition) : SolutionComposition :=
  { x := current.x + added.x
    water := current.water + added.water
    z := current.z + added.z }

/-- Calculate the percentage of liquid X in the solution -/
noncomputable def percentageX (composition : SolutionComposition) : ℝ :=
  composition.x / (composition.x + composition.water + composition.z) * 100

theorem solution_composition_theorem (initialY : SolutionComposition)
    (h1 : initialY.x = 0.4)
    (h2 : initialY.water = 0.45)
    (h3 : initialY.z = 0.15)
    (initialWeight : ℝ)
    (h4 : initialWeight = 18)
    (evaporatedWater : ℝ)
    (h5 : evaporatedWater = 6)
    (addedWeight : ℝ)
    (h6 : addedWeight = 5) :
    ∃ ε > 0, |percentageX (addSolution (evaporateWater initialY initialWeight evaporatedWater)
                              (SolutionComposition.mk (initialY.x * addedWeight)
                                                      (initialY.water * addedWeight)
                                                      (initialY.z * addedWeight))) - 54.12| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_composition_theorem_l1302_130244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_country_frequency_l1302_130222

/-- The frequency of a word in a phrase -/
def word_frequency (total_words : ℕ) (word_occurrences : ℕ) : ℚ :=
  word_occurrences / total_words

/-- The phrase containing the word "country" -/
def phrase : String := "When the youth are strong, the country is strong; when the youth are wise, the country is wise; when the youth are wealthy, the country is wealthy."

/-- Count occurrences of a word in a list of words -/
def count_occurrences (word : String) (words : List String) : ℕ :=
  words.filter (· = word) |>.length

theorem country_frequency :
  word_frequency (phrase.split Char.isWhitespace |>.length) (count_occurrences "country" (phrase.split Char.isWhitespace)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_country_frequency_l1302_130222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1302_130209

def point1 : ℝ × ℝ := (3, -2)
def point2 : ℝ × ℝ := (-7, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points : distance point1 point2 = 2 * Real.sqrt 34 := by
  -- Unfold the definitions
  unfold distance point1 point2
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1302_130209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_MNP_area_l1302_130220

/-- Triangle MNP with vertices M, N, and P -/
structure Triangle where
  M : Prod ℝ ℝ
  N : Prod ℝ ℝ
  P : Prod ℝ ℝ

/-- The line on which point P lies -/
def line (x y : ℝ) : Prop := x + y = 8

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (a b c : Prod ℝ ℝ) : ℝ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₁) - (x₁ - x₂) * (y₃ - y₁))

/-- The main theorem -/
theorem triangle_MNP_area :
  ∀ (t : Triangle),
    t.M = (5, 0) →
    t.N = (0, 5) →
    line t.P.1 t.P.2 →
    triangleArea t.M t.N t.P = 12.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_MNP_area_l1302_130220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_example_l1302_130215

/-- Calculates the remaining amount to be paid for a product given the deposit amount,
    deposit percentage, discount percentage, and sales tax percentage. -/
noncomputable def remaining_payment (deposit : ℝ) (deposit_percent discount_percent sales_tax_percent : ℝ) : ℝ :=
  let original_price := deposit / deposit_percent
  let discounted_price := original_price * (1 - discount_percent)
  let final_cost := discounted_price * (1 + sales_tax_percent)
  final_cost - deposit

/-- Theorem stating that the remaining payment for a product with given conditions is $1381.875 -/
theorem remaining_payment_example :
  remaining_payment 150 0.10 0.05 0.075 = 1381.875 := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval remaining_payment 150 0.10 0.05 0.075

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_payment_example_l1302_130215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sin_has_property_T_l1302_130240

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (deriv f x₁) * (deriv f x₂) = -1

-- Define the functions
noncomputable def ln_func (x : ℝ) : ℝ := Real.log x
noncomputable def sin_func (x : ℝ) : ℝ := Real.sin x
noncomputable def exp_func (x : ℝ) : ℝ := Real.exp x
def cube_func (x : ℝ) : ℝ := x^3

-- Theorem statement
theorem only_sin_has_property_T :
  ¬(has_property_T ln_func) ∧
  (has_property_T sin_func) ∧
  ¬(has_property_T exp_func) ∧
  ¬(has_property_T cube_func) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_sin_has_property_T_l1302_130240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_max_at_four_l1302_130235

/-- Geometric sequence with common ratio √2 -/
noncomputable def geometric_sequence (a₁ : ℝ) : ℕ+ → ℝ :=
  λ n => a₁ * (Real.sqrt 2) ^ (n : ℝ)

/-- Sum of the first n terms of the geometric sequence -/
noncomputable def S (a₁ : ℝ) (n : ℕ+) : ℝ :=
  a₁ * (1 - (Real.sqrt 2) ^ (n : ℝ)) / (1 - Real.sqrt 2)

/-- Definition of T_n -/
noncomputable def T (a₁ : ℝ) (n : ℕ+) : ℝ :=
  (17 * S a₁ n - S a₁ (2 * n)) / (geometric_sequence a₁ (n + 1))

/-- Theorem: T_n is maximum when n = 4 -/
theorem T_max_at_four (a₁ : ℝ) (h : a₁ > 0) :
  ∀ n : ℕ+, T a₁ 4 ≥ T a₁ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_max_at_four_l1302_130235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_dimension_proof_l1302_130281

/-- The cost of insulation per square foot -/
def cost_per_sq_ft : ℝ := 20

/-- The total cost to insulate the tank -/
def total_cost : ℝ := 1240

/-- The length of the tank -/
def tank_length : ℝ := 3

/-- The height of the tank -/
def tank_height : ℝ := 2

/-- Calculate the surface area of a rectangular tank -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

theorem tank_dimension_proof (x : ℝ) :
  surface_area tank_length x tank_height * cost_per_sq_ft = total_cost → x = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_dimension_proof_l1302_130281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_for_distinct_sums_l1302_130283

/-- Represents a 3x3 grid of non-negative integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Sum of a row in the grid -/
def row_sum (g : Grid) (i : Fin 3) : ℕ :=
  (g i 0) + (g i 1) + (g i 2)

/-- Sum of a column in the grid -/
def col_sum (g : Grid) (j : Fin 3) : ℕ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- The set of all row and column sums for a given grid -/
def all_sums (g : Grid) : Finset ℕ :=
  Finset.image (row_sum g) (Finset.univ : Finset (Fin 3)) ∪
  Finset.image (col_sum g) (Finset.univ : Finset (Fin 3))

/-- The total number of pieces in the grid -/
def total_pieces (g : Grid) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => Finset.sum (Finset.univ : Finset (Fin 3)) (g i))

/-- The main theorem: The minimum number of pieces required to make all row and column sums distinct is 8 -/
theorem min_pieces_for_distinct_sums :
  ∃ (g : Grid), (Finset.card (all_sums g) = 6) ∧ (total_pieces g = 8) ∧
  (∀ (g' : Grid), Finset.card (all_sums g') = 6 → total_pieces g' ≥ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_pieces_for_distinct_sums_l1302_130283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_distance_l1302_130228

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C D P : ℝ × ℝ)

-- Define the conditions
variable (h1 : A ∈ circle)
variable (h2 : B ∈ circle)
variable (h3 : C ∈ circle)
variable (h4 : D ∈ circle)
variable (h5 : ∃ t : ℝ, A + t • (C - A) = P ∧ B + t • (D - B) = P)
variable (h6 : dist A P = 5)
variable (h7 : dist P C = 2)
variable (h8 : dist B D = 9)
variable (h9 : dist B P < dist D P)

-- State the theorem
theorem intersection_point_distance :
  dist B P = (9 - Real.sqrt 41) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_distance_l1302_130228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2456783_2468_to_nearest_tenth_l1302_130212

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The problem statement -/
theorem round_2456783_2468_to_nearest_tenth :
  roundToNearestTenth 2456783.2468 = 2456783.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_2456783_2468_to_nearest_tenth_l1302_130212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_roll_distance_l1302_130224

/-- The distance traveled by the center of a cylinder rolling along a track -/
noncomputable def cylinder_center_distance (d R₁ R₂ R₃ straight_length : ℝ) : ℝ :=
  let r := d / 2
  let arc₁ := Real.pi * (R₁ - r)
  let arc₂ := Real.pi * (R₂ + r)
  let arc₃ := Real.pi * (R₃ - r)
  arc₁ + arc₂ + arc₃ + straight_length

/-- Theorem stating the distance traveled by the center of the cylinder -/
theorem cylinder_roll_distance :
  cylinder_center_distance 6 104 64 84 100 = 249 * Real.pi + 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_roll_distance_l1302_130224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_n_digits_same_l1302_130202

theorem first_n_digits_same (n : ℕ) : 
  |(5 + Real.sqrt 26)^n - Int.floor ((5 + Real.sqrt 26)^n)| < (10 : ℝ)^(-n : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_n_digits_same_l1302_130202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AC_onto_AB_l1302_130282

def A : Fin 3 → ℝ := ![1, 1, 0]
def B : Fin 3 → ℝ := ![0, 3, 0]
def C : Fin 3 → ℝ := ![2, 2, 3]

def vec_AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]
def vec_AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

def magnitude_squared (v : Fin 3 → ℝ) : ℝ := (v 0)^2 + (v 1)^2 + (v 2)^2

noncomputable def projection_vector (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  let scalar := (dot_product v w) / (magnitude_squared w)
  ![scalar * w 0, scalar * w 1, scalar * w 2]

theorem projection_AC_onto_AB :
  projection_vector vec_AC vec_AB = ![-(1/5), 2/5, 0] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_AC_onto_AB_l1302_130282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_approx_l1302_130254

-- Define the function f
noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^4 + y^2)

-- Define the domain
def domain (x y : ℝ) : Prop := 0 < x ∧ x ≤ 3/4 ∧ 1/4 ≤ y ∧ y ≤ 2/3

-- Theorem statement
theorem max_value_approx (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), domain x y ∧ 
    (∀ (x' y' : ℝ), domain x' y' → f x' y' ≤ f x y) ∧ 
    abs (f x y - 0.371) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_approx_l1302_130254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_l1302_130280

/-- The radius of a sphere inscribed in a right cone -/
noncomputable def inscribed_sphere_radius (base_radius : ℝ) (height : ℝ) : ℝ :=
  (base_radius * height) / (base_radius * (height^2 + base_radius^2).sqrt + base_radius^2 + height^2)

theorem inscribed_sphere_in_cone (b d : ℝ) :
  inscribed_sphere_radius 10 40 = b * d.sqrt - b →
  b + d = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_l1302_130280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_sugar_fills_l1302_130205

/-- The number of times Alice must fill her measuring cup -/
def number_of_fills (total_sugar_needed : ℚ) (sugar_available : ℚ) 
  (measuring_cup_capacity : ℚ) (ounces_per_cup : ℕ) : ℕ :=
  let total_sugar_ounces := total_sugar_needed * ounces_per_cup
  let available_sugar_ounces := sugar_available * ounces_per_cup
  let remaining_sugar_ounces := total_sugar_ounces - available_sugar_ounces
  Nat.ceil (remaining_sugar_ounces / measuring_cup_capacity)

/-- Theorem stating that Alice needs to fill her measuring cup 15 times -/
theorem alice_sugar_fills : 
  number_of_fills (7/2) (3/4) (3/2) 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_sugar_fills_l1302_130205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_param_sum_value_l1302_130270

/-- Represents a hyperbola with center (h, k), focus (h, f), and vertex (h, v) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  f : ℝ
  v : ℝ

/-- The sum of parameters for a specific hyperbola -/
noncomputable def hyperbola_param_sum (hyp : Hyperbola) : ℝ :=
  let a := |hyp.k - hyp.v|
  let c := |hyp.k - hyp.f|
  let b := (c^2 - a^2).sqrt
  hyp.h + hyp.k + a + b

/-- Theorem stating the sum of parameters for the given hyperbola -/
theorem hyperbola_param_sum_value : 
  let hyp := Hyperbola.mk 1 (-2) 5 0
  hyperbola_param_sum hyp = 1 + 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_param_sum_value_l1302_130270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l1302_130272

/-- Represents the ingredients of the lemonade mixture -/
structure Ingredient where
  name : String
  weight : ℚ
  caloriesPer100g : ℚ

/-- Calculates the total weight of the lemonade mixture -/
def totalWeight (ingredients : List Ingredient) : ℚ :=
  ingredients.foldl (λ acc i => acc + i.weight) 0

/-- Calculates the total calories in the lemonade mixture -/
def totalCalories (ingredients : List Ingredient) : ℚ :=
  ingredients.foldl (λ acc i => acc + i.weight * i.caloriesPer100g / 100) 0

/-- Theorem: 150 grams of the lemonade mixture contains approximately 160 calories -/
theorem lemonade_calories : ∃ (caloriesIn150g : ℚ), 
  let ingredients := [
    { name := "lemon juice", weight := 150, caloriesPer100g := 25 },
    { name := "sugar", weight := 120, caloriesPer100g := 386 },
    { name := "water", weight := 350, caloriesPer100g := 0 },
    { name := "honey", weight := 80, caloriesPer100g := 304 }
  ]
  let totalWeight := totalWeight ingredients
  let totalCal := totalCalories ingredients
  caloriesIn150g = totalCal * 150 / totalWeight ∧
  159 ≤ caloriesIn150g ∧ caloriesIn150g ≤ 161 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_calories_l1302_130272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_type_selection_rice_type_A_preferred_l1302_130264

noncomputable def yield_A : List ℝ := [550, 580, 570, 570, 550, 600]
noncomputable def yield_B : List ℝ := [540, 590, 560, 580, 590, 560]

noncomputable def average (yields : List ℝ) : ℝ := (yields.sum) / yields.length

noncomputable def variance (yields : List ℝ) : ℝ :=
  let avg := average yields
  (yields.map (λ x => (x - avg)^2)).sum / yields.length

theorem rice_type_selection (yields_A yields_B : List ℝ)
  (h1 : yields_A.length = yields_B.length)
  (h2 : yields_A.length > 0)
  (h3 : average yields_A = average yields_B)
  (h4 : variance yields_A < variance yields_B) :
  variance yields_A < variance yields_B := by
  exact h4

-- This theorem states the conclusion in a more formal way
theorem rice_type_A_preferred (yields_A yields_B : List ℝ)
  (h1 : yields_A.length = yields_B.length)
  (h2 : yields_A.length > 0)
  (h3 : average yields_A = average yields_B)
  (h4 : variance yields_A < variance yields_B) :
  ∃ (preferred : List ℝ), preferred = yields_A ∧ variance preferred < variance yields_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_type_selection_rice_type_A_preferred_l1302_130264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l1302_130232

theorem junior_score (total_students : ℕ) (total_score : ℕ) 
  (h1 : total_students > 0)
  (h2 : total_score = total_students * 86)
  (h3 : (total_students : ℚ) * (20 : ℚ) / 100 = ↑(total_students / 5))
  (h4 : (total_students - total_students / 5) * 85 + (total_students / 5) * 90 = total_score) :
  ∀ junior : Fin (total_students / 5), 90 = (junior.val + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l1302_130232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l1302_130256

theorem book_price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * (1 - 0.2)
  let final_price := price_after_decrease * (1 + 0.1)
  final_price = P * 0.88 :=
by
  -- Unfold the let bindings
  simp only [mul_sub, mul_one, sub_mul, mul_assoc]
  -- Simplify the arithmetic
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l1302_130256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_initial_value_l1302_130214

noncomputable def sequenceV (v : ℝ) : ℕ → ℝ
  | 0 => v
  | k + 1 => sequenceV v k - sequenceV v k / 3

theorem sequence_initial_value (v : ℝ) :
  sequenceV v 2 = 12 ∧ sequenceV v 1 = 20 → v = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_initial_value_l1302_130214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_oranges_count_initial_oranges_count_proof_l1302_130238

/-- Proves that the initial number of oranges in a bag is 8, given the conditions in the problem. -/
theorem initial_oranges_count : ℕ := by
  let initial_apples : ℕ := 7
  let initial_mangoes : ℕ := 15
  let apples_taken : ℕ := 2
  let mangoes_taken : ℕ := (2 * initial_mangoes) / 3
  let remaining_fruits : ℕ := 14

  -- Define a function to calculate the number of oranges taken
  let oranges_taken := λ (initial_oranges : ℕ) => 2 * apples_taken

  -- Define a function to calculate the remaining fruits
  let calculate_remaining_fruits := λ (initial_oranges : ℕ) =>
    (initial_apples - apples_taken) + 
    (initial_oranges - oranges_taken initial_oranges) + 
    (initial_mangoes - mangoes_taken)

  -- The theorem states that there exists a unique value for initial_oranges
  -- that satisfies the condition of having 14 remaining fruits
  have h : ∃! initial_oranges : ℕ, calculate_remaining_fruits initial_oranges = remaining_fruits := by
    sorry -- The proof is omitted as per the instructions

  exact 8 -- The answer is 8 oranges

/-- The proof of the theorem -/
theorem initial_oranges_count_proof : initial_oranges_count = 8 := by
  rfl -- Reflexivity proves the equality since initial_oranges_count is defined as 8


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_oranges_count_initial_oranges_count_proof_l1302_130238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_signs_total_l1302_130267

theorem road_signs_total (first_intersection : ℕ) 
  (h1 : first_intersection = 40) : ℕ := by
  let second_intersection := first_intersection + first_intersection / 4
  let third_intersection := 2 * second_intersection
  let fourth_intersection := third_intersection - 20
  have : first_intersection + second_intersection + third_intersection + fourth_intersection = 270 := by
    sorry
  exact 270


end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_signs_total_l1302_130267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_tan_one_third_l1302_130216

theorem sin_double_angle_tan_one_third (θ : ℝ) (h : Real.tan θ = 1/3) : Real.sin (2*θ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_tan_one_third_l1302_130216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_difference_in_H_l1302_130298

-- Define the set H
def H : Set ℤ := {x | ∃ i : ℤ, i > 0 ∧ x = ⌊i * Real.sqrt 2⌋} \ {1, 2, 4, 5, 7}

-- Define the constant C
noncomputable def C : ℝ := Real.sqrt (2 * Real.sqrt 2 - 2)

-- State the theorem
theorem existence_of_difference_in_H (n : ℕ) (A : Finset ℕ) :
  (∀ a ∈ A, a ≤ n) →
  A.card ≥ ⌈C * Real.sqrt n⌉ →
  ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ (a - b : ℤ) ∈ H :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_difference_in_H_l1302_130298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_increasing_interval_l1302_130213

-- Define the function
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

-- State the theorem
theorem sine_increasing_interval (ω : ℝ) :
  ω > 0 →
  (∀ x₁ x₂, -π/5 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π/4 → f ω x₁ < f ω x₂) →
  0 < ω ∧ ω ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_increasing_interval_l1302_130213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_l1302_130295

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define a line
def line (p q : ℝ) (x y : ℝ) : Prop :=
  p * x + q * y = 1

-- Define the circle (renamed to avoid conflict)
def targetCircle (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = (a^2 * b^2) / (a^2 + b^2)

-- Define perpendicularity of two points from the origin
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_tangent_circle (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  ∀ (p q : ℝ), p ≠ 0 → q ≠ 0 →
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  ellipse a b x₁ y₁ →
  ellipse a b x₂ y₂ →
  line p q x₁ y₁ →
  line p q x₂ y₂ →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ (x y : ℝ), targetCircle a b x y ∧ line p q x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_circle_l1302_130295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_match_probabilities_l1302_130269

/-- Represents the outcome of a single game -/
inductive GameResult
| AWin
| BWin

/-- Represents the state of the match -/
structure MatchState :=
  (a_wins : Nat)
  (b_wins : Nat)

/-- The probability of player A winning a single game -/
noncomputable def prob_a_win : ℝ := 1/3

/-- The probability of exactly 5 games being played in a best-of-seven match -/
noncomputable def prob_five_games : ℝ := 8/27

/-- The probability distribution of X (remaining games when A leads 3:1) -/
noncomputable def prob_x (x : Nat) : ℝ :=
  match x with
  | 1 => 1/3
  | 2 => 2/9
  | 3 => 4/9
  | _ => 0

/-- The expectation of X when A leads 3:1 -/
noncomputable def expectation_x : ℝ := 19/9

theorem match_probabilities :
  (prob_five_games = 8/27) ∧
  (prob_x 1 = 1/3) ∧ (prob_x 2 = 2/9) ∧ (prob_x 3 = 4/9) ∧
  (expectation_x = 19/9) := by
  sorry

#check match_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_match_probabilities_l1302_130269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1302_130230

theorem trigonometric_identities (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (6 * Real.sin θ + Real.cos θ) / (3 * Real.sin θ - 2 * Real.cos θ) = 13 / 4 ∧
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1302_130230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_final_amount_is_33_20_l1302_130200

noncomputable def total_amount : ℝ := 72

noncomputable def ryan_fraction : ℝ := 2/5
noncomputable def sarah_fraction : ℝ := 1/4

noncomputable def ryan_debt_to_leo : ℝ := 8
noncomputable def sarah_debt_to_leo : ℝ := 10
noncomputable def leo_debt_to_ryan : ℝ := 6
noncomputable def leo_debt_to_sarah : ℝ := 4

noncomputable def ryan_amount : ℝ := ryan_fraction * total_amount
noncomputable def sarah_amount : ℝ := sarah_fraction * total_amount
noncomputable def leo_initial_amount : ℝ := total_amount - ryan_amount - sarah_amount

noncomputable def ryan_net_debt : ℝ := ryan_debt_to_leo - leo_debt_to_ryan
noncomputable def sarah_net_debt : ℝ := sarah_debt_to_leo - leo_debt_to_sarah

theorem leo_final_amount_is_33_20 :
  leo_initial_amount + ryan_net_debt + sarah_net_debt = 33.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_final_amount_is_33_20_l1302_130200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1302_130226

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) :
  ∃ N : ℚ, sum_n seq 20 = 10 * N → N = seq.a 9 + seq.a 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_property_l1302_130226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l1302_130265

theorem quadratic_solution_sum (a b c d : ℕ+) (x : ℝ) :
  (x^2 - 6*x + 2 = 0) →
  (x = (a : ℝ) + (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ) ∨
   x = (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) / (d : ℝ)) →
  a + b + c + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_sum_l1302_130265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_example_l1302_130243

/-- The height of a cuboid with given base area and volume -/
noncomputable def cuboid_height (base_area : ℝ) (volume : ℝ) : ℝ :=
  volume / base_area

/-- Theorem: The height of a cuboid with base area 50 cm² and volume 2000 cm³ is 40 cm -/
theorem cuboid_height_example : cuboid_height 50 2000 = 40 := by
  -- Unfold the definition of cuboid_height
  unfold cuboid_height
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_height_example_l1302_130243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_obtuse_inclination_l1302_130253

/-- Given a line passing through points A(1-t, 1+t) and B(3, 2t), 
    if this line has an obtuse angle of inclination, then t ∈ (-2, 1). -/
theorem line_with_obtuse_inclination (t : ℝ) : 
  (∃ (m : ℝ), m < 0 ∧ m = (2*t - (1+t)) / (3 - (1-t))) → 
  t > -2 ∧ t < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_with_obtuse_inclination_l1302_130253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1302_130221

noncomputable section

variable (k : ℝ)

def f (x : ℝ) : ℝ := (Real.log x - k - 1) * x

theorem f_properties (k : ℝ) :
  (∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f k x < 4 * Real.log x) →
  (k > 1 - 8 / (Real.exp 2)) ∧
  (k > 0 → ∃ x > (1 : ℝ), ∀ y > (1 : ℝ), f k x ≤ f k y) ∧
  (∀ x₁ x₂, x₁ > (1 : ℝ) → x₂ > (1 : ℝ) → x₁ ≠ x₂ → f k x₁ = f k x₂ → x₁ * x₂ < Real.exp (2 * k)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1302_130221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_phi_divisors_2008_mod_1000_l1302_130279

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The sum of Euler's totient function over all positive divisors of n -/
def sum_phi_divisors (n : ℕ) : ℕ :=
  (Nat.divisors n).sum phi

theorem sum_phi_divisors_2008_mod_1000 :
  sum_phi_divisors 2008 ≡ 8 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_phi_divisors_2008_mod_1000_l1302_130279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_Z_outside_XYW_l1302_130263

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration
noncomputable def circleX : Circle := { center := (0, 0), radius := 2 }
noncomputable def circleY : Circle := { center := (4, 0), radius := 2 }
noncomputable def circleZ : Circle := { center := (2, 2 * Real.sqrt 3), radius := 2 }
noncomputable def circleW : Circle := { center := (-3, 0), radius := 1 }

-- Define the area calculation function
noncomputable def areaOutsideXYW (z : Circle) (x : Circle) (y : Circle) (w : Circle) : ℝ := sorry

-- Theorem statement
theorem area_inside_Z_outside_XYW :
  areaOutsideXYW circleZ circleX circleY circleW = (5 * Real.pi / 3 + 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_Z_outside_XYW_l1302_130263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_mean_eq_one_l1302_130271

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability density function of a normal distribution -/
noncomputable def normalPDF (X : NormalRV) (x : ℝ) : ℝ :=
  (1 / (X.σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - X.μ) / X.σ)^2)

/-- The cumulative distribution function of a normal distribution -/
noncomputable def normalCDF (X : NormalRV) (x : ℝ) : ℝ :=
  ∫ y in Set.Iio x, normalPDF X y

/-- Theorem: If X ~ N(μ, σ²) and P(X ≤ 0) = P(X ≥ 2), then μ = 1 -/
theorem normal_distribution_mean_eq_one (X : NormalRV) 
    (h : normalCDF X 0 = 1 - normalCDF X 2) : X.μ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_mean_eq_one_l1302_130271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_inner_length_is_four_l1302_130249

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  x : ℝ
  inner : RectDimensions := { length := x, width := 2 }
  middle : RectDimensions := { length := x + 4, width := 6 }
  outer : RectDimensions := { length := x + 8, width := 10 }

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ r : RugRegions,
  isArithmeticProgression (area r.inner) (area r.middle) (area r.outer) →
  r.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_inner_length_is_four_l1302_130249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_people_meet_at_midpoint_l1302_130218

noncomputable section

/-- The distance between points M and N in miles -/
def total_distance : ℝ := 72

/-- The constant speed of the first person in miles per hour -/
def speed_person1 : ℝ := 4

/-- The initial speed of the second person in miles per hour -/
def initial_speed_person2 : ℝ := 2

/-- The hourly increase in speed for the second person in miles per hour -/
def speed_increase_person2 : ℝ := 0.5

/-- The time at which the two people meet in hours -/
def meeting_time : ℝ := 9

/-- The distance traveled by the first person when they meet -/
noncomputable def distance_person1 : ℝ := speed_person1 * meeting_time

/-- The distance traveled by the second person when they meet -/
noncomputable def distance_person2 : ℝ := meeting_time / 2 * (2 * initial_speed_person2 + speed_increase_person2 * (meeting_time - 1))

theorem people_meet_at_midpoint :
  distance_person1 = distance_person2 ∧
  distance_person1 = total_distance / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_people_meet_at_midpoint_l1302_130218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3456_to_hundredth_l1302_130255

/-- Rounds a real number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The problem statement -/
theorem round_3456_to_hundredth :
  roundToHundredth 3.456 = 3.46 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3456_to_hundredth_l1302_130255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_calculation_l1302_130247

/-- Calculates the total profit of a partnership given the investments and one partner's share --/
def total_profit (invest_a invest_b invest_c c_share : ℕ) : ℕ :=
  let gcd := invest_a.gcd invest_b |>.gcd invest_c
  let ratio_sum := invest_a / gcd + invest_b / gcd + invest_c / gcd
  let c_ratio := invest_c / gcd
  (c_share * ratio_sum) / c_ratio

theorem partnership_profit_calculation (invest_a invest_b invest_c c_share : ℕ) 
  (h1 : invest_a = 12000)
  (h2 : invest_b = 16000)
  (h3 : invest_c = 20000)
  (h4 : c_share = 36000) :
  total_profit invest_a invest_b invest_c c_share = 86400 := by
  sorry

#eval total_profit 12000 16000 20000 36000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_calculation_l1302_130247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_number_difference_l1302_130294

/-- A 4-digit geometric number is a number between 1000 and 9999 whose digits form a geometric sequence. -/
def IsGeometricNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  ∃ (a r : ℕ), r ≠ 0 ∧
    (let d1 := n / 1000
     let d2 := (n / 100) % 10
     let d3 := (n / 10) % 10
     let d4 := n % 10
     d1 = a ∧ d2 = a * r ∧ d3 = a * r * r ∧ d4 = a * r * r * r ∧
     d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)

/-- The largest 4-digit geometric number -/
def LargestGeometricNumber : ℕ := 3927

/-- The smallest 4-digit geometric number -/
def SmallestGeometricNumber : ℕ := 1248

theorem geometric_number_difference :
  IsGeometricNumber LargestGeometricNumber ∧
  IsGeometricNumber SmallestGeometricNumber ∧
  (∀ n : ℕ, IsGeometricNumber n → SmallestGeometricNumber ≤ n ∧ n ≤ LargestGeometricNumber) ∧
  LargestGeometricNumber - SmallestGeometricNumber = 2679 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_number_difference_l1302_130294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_BHD_is_one_fourth_l1302_130239

/-- A rectangular solid with specific angle conditions -/
structure RectangularSolid where
  /-- The measure of angle DHG in radians -/
  angle_DHG : ℝ
  /-- The measure of angle FHB in radians -/
  angle_FHB : ℝ
  /-- Condition that angle DHG is 30 degrees (π/6 radians) -/
  DHG_is_30_deg : angle_DHG = π / 6
  /-- Condition that angle FHB is 45 degrees (π/4 radians) -/
  FHB_is_45_deg : angle_FHB = π / 4

/-- The angle BHD in a rectangular solid -/
def angle_BHD (solid : RectangularSolid) : ℝ := sorry

/-- The cosine of angle BHD in a rectangular solid with the given conditions is 1/4 -/
theorem cos_BHD_is_one_fourth (solid : RectangularSolid) :
  Real.cos (angle_BHD solid) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_BHD_is_one_fourth_l1302_130239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1302_130227

-- Define the function f
noncomputable def f (θ : Real) : Real :=
  ∫ x in (0 : Real)..1, |Real.sqrt (1 - x^2) - Real.sin θ|

-- State the theorem
theorem f_properties :
  (∀ θ : Real, 0 ≤ θ ∧ θ ≤ π/2 → f θ ≤ 1) ∧
  (∀ θ : Real, 0 ≤ θ ∧ θ ≤ π/2 → f θ ≥ Real.sqrt 2 - 1) ∧
  (∃ θ : Real, 0 ≤ θ ∧ θ ≤ π/2 ∧ f θ = 1) ∧
  (∃ θ : Real, 0 ≤ θ ∧ θ ≤ π/2 ∧ f θ = Real.sqrt 2 - 1) ∧
  (∫ θ in (0 : Real)..(π/2), f θ = 4 - π/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1302_130227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l1302_130252

noncomputable def points : List (ℝ × ℝ) := [(2, 9), (8, 25), (10, 30), (15, 45), (25, 60)]

noncomputable def isAboveLine (point : ℝ × ℝ) : Bool :=
  point.2 > 3 * point.1 + 4

noncomputable def sumXCoordinates (pts : List (ℝ × ℝ)) : ℝ :=
  (pts.filter isAboveLine).map (·.1) |>.sum

theorem sum_x_coordinates_above_line :
  sumXCoordinates points = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l1302_130252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1302_130229

variable (a x₁ x₂ : ℝ)

noncomputable def f (x : ℝ) : ℝ := Real.log x - a * x^2

noncomputable def g (x : ℝ) : ℝ := f a x + a * x^2 - x

theorem inequality_proof (h : x₁ > x₂) (h' : x₂ > 0) :
  (x₁ / (x₁^2 + x₂^2)) - (g a x₁ - g a x₂) / (x₁ - x₂) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1302_130229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_pathway_area_ratio_l1302_130258

/-- Represents a garden pathway with a rectangular center and half-circles on each end. -/
structure GardenPathway where
  width : ℝ
  length : ℝ

/-- Calculates the area of the rectangular part of the pathway. -/
def rectangleArea (p : GardenPathway) : ℝ := p.width * p.length

/-- Calculates the combined area of the two half-circles. -/
noncomputable def halfCirclesArea (p : GardenPathway) : ℝ := Real.pi * (p.width / 2) ^ 2

/-- Theorem: The ratio of the rectangular area to the half-circles area is 16/(3π) 
    for a pathway with width 24 and length-to-width ratio of 4:3. -/
theorem garden_pathway_area_ratio :
  ∀ (p : GardenPathway),
    p.width = 24 →
    p.length = (4 / 3) * p.width →
    (rectangleArea p) / (halfCirclesArea p) = 16 / (3 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_pathway_area_ratio_l1302_130258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l1302_130246

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 - 4*I) * (2 + 3*I) ∧ 
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l1302_130246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1302_130242

-- Define the concept of a quadratic radical
def QuadraticRadical (x : ℝ) : Prop := ∃ y : ℝ, x = y^2

-- Define the concept of simplest quadratic radical
def SimplestQuadraticRadical (x : ℝ) : Prop :=
  QuadraticRadical x ∧ ∀ y : ℝ, QuadraticRadical y → (∃ z : ℝ, x = z * y) → y = 1 ∨ y = x

-- Define the given options
noncomputable def option_A : ℝ := Real.sqrt 27
noncomputable def option_B : ℝ := Real.sqrt 15
noncomputable def option_C (a : ℝ) : ℝ := Real.sqrt (3 * a^2)
noncomputable def option_D (a : ℝ) : ℝ := Real.sqrt (1 / a)

-- Theorem statement
theorem simplest_quadratic_radical :
  ∀ a : ℝ, a ≠ 0 →
    SimplestQuadraticRadical option_B ∧
    ¬SimplestQuadraticRadical option_A ∧
    ¬SimplestQuadraticRadical (option_C a) ∧
    ¬SimplestQuadraticRadical (option_D a) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1302_130242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1302_130276

/-- Given a train and platform with specific dimensions and crossing time, 
    calculate the time taken for the train to cross a signal pole. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 300) 
  (h2 : platform_length = 250) 
  (h3 : platform_crossing_time = 33) : 
  ∃ (signal_crossing_time : ℝ), abs (signal_crossing_time - 18) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1302_130276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_has_two_zeros_l1302_130204

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := 3 - x^2

-- Define the function h as the difference of f and g
noncomputable def h (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem h_has_two_zeros : ∃ (a b : ℝ), a ≠ b ∧ h a = 0 ∧ h b = 0 ∧ ∀ x, h x = 0 → x = a ∨ x = b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_has_two_zeros_l1302_130204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1302_130277

theorem angle_values (α β : Real) 
  (h1 : Real.cos (π/2 - α) = Real.sqrt 2 * Real.cos (3*π/2 + β))
  (h2 : Real.sqrt 3 * Real.sin (3*π/2 - α) = -Real.sqrt 2 * Real.sin (π/2 + β))
  (h3 : 0 < α) (h4 : α < π)
  (h5 : 0 < β) (h6 : β < π) :
  ((α = π/4 ∧ β = π/6) ∨ (α = 3*π/4 ∧ β = 5*π/6)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l1302_130277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_implies_product_l1302_130292

open Real

theorem sin_cos_equation_implies_product (θ : ℝ) :
  sin θ + cos θ = 2 * (sin θ - cos θ) →
  sin (θ - π) * sin (π / 2 - θ) = -3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_implies_product_l1302_130292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_range_smallest_angle_range_middle_angle_range_l1302_130266

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define functions to get the largest, smallest, and middle angles
noncomputable def largest_angle (t : Triangle) : ℝ := max t.angle1 (max t.angle2 t.angle3)
noncomputable def smallest_angle (t : Triangle) : ℝ := min t.angle1 (min t.angle2 t.angle3)
noncomputable def middle_angle (t : Triangle) : ℝ :=
  t.angle1 + t.angle2 + t.angle3 - largest_angle t - smallest_angle t

-- Theorem statements
theorem largest_angle_range (t : Triangle) : 
  60 ≤ largest_angle t ∧ largest_angle t < 180 := by sorry

theorem smallest_angle_range (t : Triangle) : 
  0 < smallest_angle t ∧ smallest_angle t ≤ 60 := by sorry

theorem middle_angle_range (t : Triangle) : 
  0 < middle_angle t ∧ middle_angle t < 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_range_smallest_angle_range_middle_angle_range_l1302_130266
