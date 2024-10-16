import Mathlib

namespace NUMINAMATH_CALUDE_modulus_of_i_times_one_plus_i_l2024_202412

theorem modulus_of_i_times_one_plus_i : Complex.abs (Complex.I * (1 + Complex.I)) = 1 := by sorry

end NUMINAMATH_CALUDE_modulus_of_i_times_one_plus_i_l2024_202412


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2024_202478

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + r^2 + 5 * r - 4) - (r^3 + 3 * r^2 + 7 * r - 2) = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2024_202478


namespace NUMINAMATH_CALUDE_machine_completion_time_l2024_202400

/-- Given two machines where one takes T hours and the other takes 8 hours to complete an order,
    if they complete the order together in 4.235294117647059 hours, then T = 9. -/
theorem machine_completion_time (T : ℝ) : 
  (1 / T + 1 / 8 = 1 / 4.235294117647059) → T = 9 := by
sorry

end NUMINAMATH_CALUDE_machine_completion_time_l2024_202400


namespace NUMINAMATH_CALUDE_cubic_tangent_max_l2024_202446

/-- A cubic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_tangent_max (a b m : ℝ) (hm : m ≠ 0) :
  f' a b m = 0 ∧                   -- Tangent condition (derivative = 0 at x = m)
  f a b m = 0 ∧                    -- Tangent condition (f(m) = 0)
  (∃ x, f a b x = (1/2 : ℝ)) ∧     -- Maximum value condition
  (∀ x, f a b x ≤ (1/2 : ℝ)) →     -- Maximum value condition
  m = (3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cubic_tangent_max_l2024_202446


namespace NUMINAMATH_CALUDE_road_trip_time_calculation_l2024_202424

/-- Calculates the total time for a road trip given the specified conditions -/
theorem road_trip_time_calculation (distance : ℝ) (speed : ℝ) (break_interval : ℝ) (break_duration : ℝ) (hotel_search_time : ℝ) : 
  distance = 2790 →
  speed = 62 →
  break_interval = 5 →
  break_duration = 0.5 →
  hotel_search_time = 0.5 →
  (distance / speed + 
   (⌊distance / speed / break_interval⌋ - 1) * break_duration + 
   hotel_search_time) = 49.5 := by
  sorry

#check road_trip_time_calculation

end NUMINAMATH_CALUDE_road_trip_time_calculation_l2024_202424


namespace NUMINAMATH_CALUDE_max_handshakes_networking_event_l2024_202405

/-- Calculate the number of handshakes in a group -/
def handshakesInGroup (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculate the total number of handshakes -/
def totalHandshakes (total : ℕ) (groupA : ℕ) (groupB : ℕ) (groupC : ℕ) : ℕ :=
  handshakesInGroup total - (handshakesInGroup groupA + handshakesInGroup groupB + handshakesInGroup groupC)

/-- Theorem stating the maximum number of handshakes under given conditions -/
theorem max_handshakes_networking_event :
  let total := 100
  let groupA := 30
  let groupB := 35
  let groupC := 35
  totalHandshakes total groupA groupB groupC = 3325 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_networking_event_l2024_202405


namespace NUMINAMATH_CALUDE_expected_faces_six_rolls_l2024_202435

/-- The number of sides on a fair die -/
def n : ℕ := 6

/-- The number of times the die is rolled -/
def k : ℕ := 6

/-- The probability that a specific face does not appear in a single roll -/
def p : ℚ := (n - 1) / n

/-- The expected number of different faces that appear when rolling a fair n-sided die k times -/
def expected_different_faces : ℚ := n * (1 - p^k)

/-- Theorem stating that the expected number of different faces that appear when 
    rolling a fair 6-sided die 6 times is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_faces_six_rolls : 
  expected_different_faces = (n^k - (n-1)^k) / n^(k-1) :=
sorry

end NUMINAMATH_CALUDE_expected_faces_six_rolls_l2024_202435


namespace NUMINAMATH_CALUDE_train_crossing_time_l2024_202430

/-- Proves that a train of given length crossing a bridge of given length in a given time will take 40 seconds to cross a signal post. -/
theorem train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (bridge_crossing_time : ℝ) :
  train_length = 600 →
  bridge_length = 9000 →
  bridge_crossing_time = 600 →
  (train_length / (bridge_length / bridge_crossing_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2024_202430


namespace NUMINAMATH_CALUDE_cosine_two_local_minima_l2024_202432

/-- A function f(x) = cos(ωx) has exactly two local minimum points in [0, π/2] iff 6 ≤ ω < 10 -/
theorem cosine_two_local_minima (ω : ℝ) (h : ω > 0) :
  (∃! (n : ℕ), n = 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (π / 2) →
    (∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), y ∈ Set.Ioo (x - ε) (x + ε) →
      Real.cos (ω * y) ≥ Real.cos (ω * x))) ↔
  6 ≤ ω ∧ ω < 10 :=
sorry

end NUMINAMATH_CALUDE_cosine_two_local_minima_l2024_202432


namespace NUMINAMATH_CALUDE_max_areas_theorem_l2024_202454

/-- Represents the number of non-overlapping areas in a circular disk -/
def max_areas (n : ℕ) : ℕ := 3 * n + 1

/-- 
Theorem: Given a circular disk divided by 2n equally spaced radii (n > 0) and one secant line, 
the maximum number of non-overlapping areas is 3n + 1.
-/
theorem max_areas_theorem (n : ℕ) (h : n > 0) : 
  max_areas n = 3 * n + 1 := by
  sorry

#check max_areas_theorem

end NUMINAMATH_CALUDE_max_areas_theorem_l2024_202454


namespace NUMINAMATH_CALUDE_increasing_decreasing_functions_exist_l2024_202470

-- Define a function that is increasing on one interval and decreasing on another
def has_increasing_decreasing_intervals (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∧
    (∀ x y, c < x ∧ x < y ∧ y < d → f y < f x)

-- Theorem stating that such functions exist
theorem increasing_decreasing_functions_exist :
  ∃ f : ℝ → ℝ, has_increasing_decreasing_intervals f :=
sorry

end NUMINAMATH_CALUDE_increasing_decreasing_functions_exist_l2024_202470


namespace NUMINAMATH_CALUDE_intersection_range_l2024_202489

/-- The range of k for which the line y = kx + 2 intersects the right branch of 
    the hyperbola x^2 - y^2 = 6 at two distinct points -/
theorem intersection_range :
  ∀ k : ℝ, 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 0 ∧ x₂ > 0 ∧
   x₁^2 - (k * x₁ + 2)^2 = 6 ∧
   x₂^2 - (k * x₂ + 2)^2 = 6) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l2024_202489


namespace NUMINAMATH_CALUDE_goat_roaming_area_specific_case_l2024_202473

/-- Represents the dimensions of a rectangular shed -/
structure ShedDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area a goat can roam when tied to the corner of a rectangular shed -/
def goatRoamingArea (shed : ShedDimensions) (leashLength : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area a goat can roam under specific conditions -/
theorem goat_roaming_area_specific_case :
  let shed : ShedDimensions := { length := 5, width := 4 }
  let leashLength : ℝ := 4
  goatRoamingArea shed leashLength = 12.25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_goat_roaming_area_specific_case_l2024_202473


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2024_202456

/-- Given a quadratic equation ax^2 + bx + c = 0 with two real roots,
    s1 is the sum of the roots,
    s2 is the sum of the squares of the roots,
    s3 is the sum of the cubes of the roots.
    This theorem proves that as3 + bs2 + cs1 = 0. -/
theorem quadratic_roots_sum (a b c : ℝ) (s1 s2 s3 : ℝ) 
    (h1 : a ≠ 0)
    (h2 : b^2 - 4*a*c > 0)
    (h3 : s1 = -b/a)
    (h4 : s2 = b^2/a^2 - 2*c/a)
    (h5 : s3 = -b/a * (b^2/a^2 - 3*c/a)) :
  a * s3 + b * s2 + c * s1 = 0 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_sum_l2024_202456


namespace NUMINAMATH_CALUDE_parabola_C₃_expression_l2024_202403

/-- The parabola C₁ -/
def C₁ (x y : ℝ) : Prop := y = x^2 - 2*x + 3

/-- The parabola C₂, shifted 1 unit to the left from C₁ -/
def C₂ (x y : ℝ) : Prop := C₁ (x + 1) y

/-- The parabola C₃, symmetric to C₂ with respect to the y-axis -/
def C₃ (x y : ℝ) : Prop := C₂ (-x) y

/-- The theorem stating the analytical expression of C₃ -/
theorem parabola_C₃_expression : ∀ x y : ℝ, C₃ x y ↔ y = x^2 + 2 := by sorry

end NUMINAMATH_CALUDE_parabola_C₃_expression_l2024_202403


namespace NUMINAMATH_CALUDE_sin_48_greater_cos_48_l2024_202423

theorem sin_48_greater_cos_48 : Real.sin (48 * π / 180) > Real.cos (48 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_48_greater_cos_48_l2024_202423


namespace NUMINAMATH_CALUDE_bridge_length_l2024_202499

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 240 :=
by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l2024_202499


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l2024_202401

/-- Properties of a parallelepiped -/
structure Parallelepiped where
  projection : ℝ
  height : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculate the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : Parallelepiped) : ℝ := sorry

/-- Calculate the volume of the parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

/-- Theorem stating the lateral surface area and volume of the given parallelepiped -/
theorem parallelepiped_properties (p : Parallelepiped) 
  (h1 : p.projection = 5)
  (h2 : p.height = 12)
  (h3 : p.rhombus_area = 24)
  (h4 : p.rhombus_diagonal = 8) :
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l2024_202401


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2024_202480

-- Problem 1
theorem equation_one_solution (x : ℝ) : 
  (9 / x = 8 / (x - 1)) ↔ x = 9 :=
sorry

-- Problem 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℝ), ((x - 8) / (x - 7) - 8 = 1 / (7 - x)) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2024_202480


namespace NUMINAMATH_CALUDE_equation_solution_l2024_202486

theorem equation_solution :
  ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 1) = (128 : ℝ) ^ (x + 1) ∧ x = -14/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2024_202486


namespace NUMINAMATH_CALUDE_tracy_candies_problem_l2024_202471

theorem tracy_candies_problem (x : ℕ) : 
  (x % 4 = 0) →  -- x is divisible by 4
  (x % 2 = 0) →  -- x is divisible by 2
  (∃ y : ℕ, 2 ≤ y ∧ y ≤ 6 ∧ x / 2 - 20 - y = 5) →  -- sister took between 2 to 6 candies, leaving 5
  x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_tracy_candies_problem_l2024_202471


namespace NUMINAMATH_CALUDE_escalator_speed_l2024_202444

/-- Given an escalator and a person walking on it, calculate the escalator's speed. -/
theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) : 
  escalator_length = 112 →
  person_speed = 4 →
  time_taken = 8 →
  (person_speed + (escalator_length / time_taken - person_speed)) * time_taken = escalator_length →
  escalator_length / time_taken - person_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_escalator_speed_l2024_202444


namespace NUMINAMATH_CALUDE_spending_ratio_theorem_l2024_202494

/-- Represents David's wages from last week -/
def last_week_wages : ℝ := 1

/-- Percentage spent on recreation last week -/
def last_week_recreation_percent : ℝ := 0.20

/-- Percentage spent on transportation last week -/
def last_week_transportation_percent : ℝ := 0.10

/-- Percentage reduction in wages this week -/
def wage_reduction_percent : ℝ := 0.30

/-- Percentage spent on recreation this week -/
def this_week_recreation_percent : ℝ := 0.25

/-- Percentage spent on transportation this week -/
def this_week_transportation_percent : ℝ := 0.15

/-- The ratio of this week's combined spending to last week's is approximately 0.9333 -/
theorem spending_ratio_theorem : 
  let last_week_total := (last_week_recreation_percent + last_week_transportation_percent) * last_week_wages
  let this_week_wages := (1 - wage_reduction_percent) * last_week_wages
  let this_week_total := (this_week_recreation_percent + this_week_transportation_percent) * this_week_wages
  abs ((this_week_total / last_week_total) - 0.9333) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_spending_ratio_theorem_l2024_202494


namespace NUMINAMATH_CALUDE_negation_equivalence_l2024_202495

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2024_202495


namespace NUMINAMATH_CALUDE_no_solution_in_interval_l2024_202408

theorem no_solution_in_interval (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), (2 - a) * (x - 1) - 2 * Real.log x ≠ 0) ↔ 
  a ∈ Set.Ici (2 - 4 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_interval_l2024_202408


namespace NUMINAMATH_CALUDE_extreme_value_at_three_increasing_on_negative_l2024_202455

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

theorem extreme_value_at_three (h : ∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 3 → |x - 3| < ε → f a x ≤ f a 3) :
  a = 3 := by sorry

theorem increasing_on_negative (h : ∀ (x y : ℝ), x < y → y < 0 → f a x < f a y) :
  0 ≤ a := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_three_increasing_on_negative_l2024_202455


namespace NUMINAMATH_CALUDE_inequality_not_holding_l2024_202421

theorem inequality_not_holding (x y : ℝ) (h : x > y) : ¬(-2*x > -2*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_holding_l2024_202421


namespace NUMINAMATH_CALUDE_money_sharing_l2024_202404

theorem money_sharing (jessica kevin laura total : ℕ) : 
  jessica + kevin + laura = total →
  jessica = 45 →
  3 * kevin = 4 * jessica →
  3 * laura = 9 * jessica →
  total = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l2024_202404


namespace NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l2024_202409

theorem modular_inverse_15_mod_17 :
  ∃ a : ℕ, a ≤ 16 ∧ (15 * a) % 17 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l2024_202409


namespace NUMINAMATH_CALUDE_remaining_distance_proof_l2024_202467

def total_distance : ℝ := 369
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2

theorem remaining_distance_proof :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time) = 121 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_proof_l2024_202467


namespace NUMINAMATH_CALUDE_kevin_cards_l2024_202418

theorem kevin_cards (initial : ℕ) (found : ℕ) (total : ℕ) : 
  initial = 7 → found = 47 → total = initial + found → total = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l2024_202418


namespace NUMINAMATH_CALUDE_function_properties_l2024_202448

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x + 1

theorem function_properties :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧
  (f 1 1 = 2) ∧
  (∀ (a : ℝ), (∃! (x : ℝ), x > Real.exp (-3) ∧ f a x = 0) ↔ 
    (a ≤ 2 / Real.exp 3 ∨ a = 1 / Real.exp 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2024_202448


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2024_202461

/-- Given an arithmetic sequence {a_n} with common difference 3,
    where a_1, a_2, a_5 form a geometric sequence, prove that a_10 = 57/2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = 3) →  -- arithmetic sequence with common difference 3
  (a 2)^2 = a 1 * a 5 →         -- a_1, a_2, a_5 form a geometric sequence
  a 10 = 57/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2024_202461


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l2024_202415

def ice_cream_sales (monday tuesday : ℕ) : Prop :=
  ∃ (wednesday thursday total : ℕ),
    wednesday = 2 * tuesday ∧
    thursday = (3 * wednesday) / 2 ∧
    total = monday + tuesday + wednesday + thursday ∧
    total = 82000

theorem ice_cream_sales_theorem :
  ice_cream_sales 10000 12000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l2024_202415


namespace NUMINAMATH_CALUDE_square_area_of_fourth_side_l2024_202451

theorem square_area_of_fourth_side (EF FG GH : ℝ) (h1 : EF^2 = 25) (h2 : FG^2 = 49) (h3 : GH^2 = 64) : 
  ∃ EG EH : ℝ, EG^2 = EF^2 + FG^2 ∧ EH^2 = EG^2 + GH^2 ∧ EH^2 = 138 := by
  sorry

end NUMINAMATH_CALUDE_square_area_of_fourth_side_l2024_202451


namespace NUMINAMATH_CALUDE_vector_trig_relations_l2024_202406

/-- Given vectors a and b with specific properties, prove cosine and sine relations -/
theorem vector_trig_relations (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.sin β = -4/5)
  (h4 : ‖(Real.cos α, Real.sin α) - (Real.cos β, Real.sin β)‖ = 4 * Real.sqrt 13 / 13) :
  Real.cos (α - β) = 5/13 ∧ Real.sin α = 16/65 := by
  sorry


end NUMINAMATH_CALUDE_vector_trig_relations_l2024_202406


namespace NUMINAMATH_CALUDE_dreams_driving_distance_l2024_202458

/-- Represents the problem of calculating Dream's driving distance --/
theorem dreams_driving_distance :
  let gas_consumption_rate : ℝ := 4  -- gallons per mile
  let additional_miles_tomorrow : ℝ := 200
  let total_gas_consumption : ℝ := 4000
  ∃ (miles_today : ℝ),
    gas_consumption_rate * miles_today + 
    gas_consumption_rate * (miles_today + additional_miles_tomorrow) = 
    total_gas_consumption ∧
    miles_today = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_dreams_driving_distance_l2024_202458


namespace NUMINAMATH_CALUDE_f_is_cone_bottomed_g_is_not_cone_bottomed_h_max_cone_bottomed_constant_l2024_202413

-- Definition of a "cone-bottomed" function
def is_cone_bottomed (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≥ M * |x|

-- Specific functions
def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := x^2 + 1

-- Theorems to prove
theorem f_is_cone_bottomed : is_cone_bottomed f := sorry

theorem g_is_not_cone_bottomed : ¬ is_cone_bottomed g := sorry

theorem h_max_cone_bottomed_constant :
  ∀ M : ℝ, (is_cone_bottomed h ∧ ∀ N : ℝ, is_cone_bottomed h → N ≤ M) → M = 2 := sorry

end NUMINAMATH_CALUDE_f_is_cone_bottomed_g_is_not_cone_bottomed_h_max_cone_bottomed_constant_l2024_202413


namespace NUMINAMATH_CALUDE_different_remainders_l2024_202497

theorem different_remainders (a b c p : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (hp : Nat.Prime p) (hsum : p = a * b + b * c + a * c) : 
  (a ^ 2 % p ≠ b ^ 2 % p ∧ a ^ 2 % p ≠ c ^ 2 % p ∧ b ^ 2 % p ≠ c ^ 2 % p) ∧
  (a ^ 3 % p ≠ b ^ 3 % p ∧ a ^ 3 % p ≠ c ^ 3 % p ∧ b ^ 3 % p ≠ c ^ 3 % p) := by
  sorry

end NUMINAMATH_CALUDE_different_remainders_l2024_202497


namespace NUMINAMATH_CALUDE_max_distance_point_to_line_l2024_202466

/-- The maximum distance from a point to a line --/
theorem max_distance_point_to_line : 
  let P : ℝ × ℝ := (-1, 3)
  let line_equation (k x : ℝ) := k * (x - 2)
  ∀ k : ℝ, 
  (∃ x : ℝ, abs (P.2 - line_equation k P.1) / Real.sqrt (k^2 + 1) ≤ 3 * Real.sqrt 2) ∧ 
  (∃ k₀ : ℝ, abs (P.2 - line_equation k₀ P.1) / Real.sqrt (k₀^2 + 1) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_point_to_line_l2024_202466


namespace NUMINAMATH_CALUDE_problem_solution_l2024_202498

theorem problem_solution : 
  |Real.sqrt 3 - 2| + (27 : ℝ) ^ (1/3 : ℝ) - Real.sqrt 16 + (-1) ^ 2023 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2024_202498


namespace NUMINAMATH_CALUDE_f_properties_l2024_202491

noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

theorem f_properties :
  (∀ x, f (x + π) = f x) ∧ 
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < π / 2 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2024_202491


namespace NUMINAMATH_CALUDE_even_not_div_four_not_sum_consec_odd_l2024_202422

theorem even_not_div_four_not_sum_consec_odd (n : ℤ) : 
  ¬(∃ k : ℤ, 2 * (n + 1) = 4 * k + 2) :=
sorry

end NUMINAMATH_CALUDE_even_not_div_four_not_sum_consec_odd_l2024_202422


namespace NUMINAMATH_CALUDE_vector_sum_proof_l2024_202462

theorem vector_sum_proof :
  let v1 : Fin 3 → ℝ := ![(-3), 2, (-1)]
  let v2 : Fin 3 → ℝ := ![1, 5, (-3)]
  v1 + v2 = ![(-2), 7, (-4)] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l2024_202462


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2024_202481

/-- Given a triangle with one side of length 10 and the opposite angle of 45°,
    the diameter of its circumscribed circle is 10√2. -/
theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h_side : side = 10) (h_angle : angle = Real.pi / 4) :
  (side / Real.sin angle) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2024_202481


namespace NUMINAMATH_CALUDE_first_rope_longer_l2024_202453

-- Define the initial length of the ropes
variable (initial_length : ℝ)

-- Define the lengths cut from each rope
def cut_length_1 : ℝ := 0.3
def cut_length_2 : ℝ := 3

-- Define the remaining lengths of each rope
def remaining_length_1 : ℝ := initial_length - cut_length_1
def remaining_length_2 : ℝ := initial_length - cut_length_2

-- Theorem statement
theorem first_rope_longer :
  remaining_length_1 initial_length > remaining_length_2 initial_length :=
by sorry

end NUMINAMATH_CALUDE_first_rope_longer_l2024_202453


namespace NUMINAMATH_CALUDE_area_of_EFGH_l2024_202439

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle EFGH formed by three identical smaller rectangles -/
def area_EFGH (small : Rectangle) : ℝ :=
  (2 * small.width) * small.length

theorem area_of_EFGH : 
  ∀ (small : Rectangle),
  small.width = 7 →
  small.length = 3 * small.width →
  area_EFGH small = 294 := by
sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l2024_202439


namespace NUMINAMATH_CALUDE_kim_payroll_time_l2024_202496

/-- Represents the time Kim spends on her morning routine -/
structure MorningRoutine where
  coffee_time : ℕ
  status_update_time_per_employee : ℕ
  num_employees : ℕ
  total_routine_time : ℕ

/-- Calculates the time spent per employee on payroll records -/
def payroll_time_per_employee (routine : MorningRoutine) : ℕ :=
  let total_status_update_time := routine.status_update_time_per_employee * routine.num_employees
  let remaining_time := routine.total_routine_time - (routine.coffee_time + total_status_update_time)
  remaining_time / routine.num_employees

/-- Theorem stating that Kim spends 3 minutes per employee updating payroll records -/
theorem kim_payroll_time (kim_routine : MorningRoutine) 
  (h1 : kim_routine.coffee_time = 5)
  (h2 : kim_routine.status_update_time_per_employee = 2)
  (h3 : kim_routine.num_employees = 9)
  (h4 : kim_routine.total_routine_time = 50) :
  payroll_time_per_employee kim_routine = 3 := by
  sorry

#eval payroll_time_per_employee { coffee_time := 5, status_update_time_per_employee := 2, num_employees := 9, total_routine_time := 50 }

end NUMINAMATH_CALUDE_kim_payroll_time_l2024_202496


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2024_202465

/-- Represents the investment and profit information for a partnership business --/
structure PartnershipBusiness where
  a_initial : ℕ
  a_additional : ℕ
  a_additional_time : ℕ
  b_initial : ℕ
  b_withdrawal : ℕ
  b_withdrawal_time : ℕ
  c_initial : ℕ
  c_additional : ℕ
  c_additional_time : ℕ
  total_time : ℕ
  c_profit : ℕ

/-- Calculates the total profit of the partnership business --/
def calculate_total_profit (pb : PartnershipBusiness) : ℕ :=
  sorry

/-- Theorem stating that given the specific investment conditions, 
    if C's profit is 45000, then the total profit is 103571 --/
theorem partnership_profit_calculation 
  (pb : PartnershipBusiness)
  (h1 : pb.a_initial = 5000)
  (h2 : pb.a_additional = 2000)
  (h3 : pb.a_additional_time = 4)
  (h4 : pb.b_initial = 8000)
  (h5 : pb.b_withdrawal = 1000)
  (h6 : pb.b_withdrawal_time = 4)
  (h7 : pb.c_initial = 9000)
  (h8 : pb.c_additional = 3000)
  (h9 : pb.c_additional_time = 6)
  (h10 : pb.total_time = 12)
  (h11 : pb.c_profit = 45000) :
  calculate_total_profit pb = 103571 :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2024_202465


namespace NUMINAMATH_CALUDE_max_product_2015_l2024_202441

/-- Given the digits 2, 0, 1, and 5, the maximum product obtained by rearranging
    these digits into two numbers and multiplying them is 1050. -/
theorem max_product_2015 : ∃ (a b : ℕ),
  (a ≤ 99 ∧ b ≤ 99) ∧
  (∀ (d : ℕ), d ∈ [a.div 10, a % 10, b.div 10, b % 10] → d ∈ [2, 0, 1, 5]) ∧
  (a * b = 1050) ∧
  (∀ (c d : ℕ), c ≤ 99 → d ≤ 99 →
    (∀ (e : ℕ), e ∈ [c.div 10, c % 10, d.div 10, d % 10] → e ∈ [2, 0, 1, 5]) →
    c * d ≤ 1050) :=
by sorry

end NUMINAMATH_CALUDE_max_product_2015_l2024_202441


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_5_mod_8_l2024_202419

theorem largest_integer_less_than_150_with_remainder_5_mod_8 :
  ∃ n : ℕ, n < 150 ∧ n % 8 = 5 ∧ ∀ m : ℕ, m < 150 ∧ m % 8 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_5_mod_8_l2024_202419


namespace NUMINAMATH_CALUDE_lucy_aquarium_cleaning_l2024_202475

/-- The number of aquariums Lucy can clean in a given time period. -/
def aquariums_cleaned (aquariums_per_period : ℚ) (hours : ℚ) : ℚ :=
  aquariums_per_period * hours

/-- Theorem stating how many aquariums Lucy can clean in 24 hours. -/
theorem lucy_aquarium_cleaning :
  let aquariums_per_3hours : ℚ := 2
  let cleaning_period : ℚ := 3
  let working_hours : ℚ := 24
  aquariums_cleaned (aquariums_per_3hours / cleaning_period) working_hours = 16 := by
  sorry

end NUMINAMATH_CALUDE_lucy_aquarium_cleaning_l2024_202475


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l2024_202477

theorem compare_negative_fractions : -3/4 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l2024_202477


namespace NUMINAMATH_CALUDE_revenue_growth_equation_l2024_202426

theorem revenue_growth_equation (x : ℝ) : 
  let january_revenue : ℝ := 900000
  let total_revenue : ℝ := 1440000
  90000 * (1 + x) + 90000 * (1 + x)^2 = total_revenue - january_revenue :=
by sorry

end NUMINAMATH_CALUDE_revenue_growth_equation_l2024_202426


namespace NUMINAMATH_CALUDE_triangle_area_bound_l2024_202490

-- Define a triangle with integer coordinates
structure IntTriangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ

-- Define a function to count integer points inside a triangle
def countInteriorPoints (t : IntTriangle) : ℕ := sorry

-- Define a function to count integer points on the edges of a triangle
def countBoundaryPoints (t : IntTriangle) : ℕ := sorry

-- Define a function to calculate the area of a triangle
def triangleArea (t : IntTriangle) : ℚ := sorry

-- Theorem statement
theorem triangle_area_bound (t : IntTriangle) :
  countInteriorPoints t = 1 → triangleArea t ≤ 9/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_bound_l2024_202490


namespace NUMINAMATH_CALUDE_eggs_per_year_is_3380_l2024_202433

/-- The number of eggs Lisa cooks for her family for breakfast in a year -/
def eggs_per_year : ℕ :=
  let days_per_week : ℕ := 5
  let num_children : ℕ := 4
  let eggs_per_child : ℕ := 2
  let eggs_for_husband : ℕ := 3
  let eggs_for_self : ℕ := 2
  let weeks_per_year : ℕ := 52
  
  let eggs_per_day : ℕ := num_children * eggs_per_child + eggs_for_husband + eggs_for_self
  let eggs_per_week : ℕ := eggs_per_day * days_per_week
  
  eggs_per_week * weeks_per_year

theorem eggs_per_year_is_3380 : eggs_per_year = 3380 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_year_is_3380_l2024_202433


namespace NUMINAMATH_CALUDE_max_a_value_l2024_202469

-- Define the line equation
def line_equation (m : ℚ) (x : ℚ) : ℚ := m * x + 3

-- Define the condition for not passing through lattice points
def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℕ, 0 < x → x ≤ 50 → ¬ ∃ y : ℤ, line_equation m x = y

-- Define the theorem
theorem max_a_value : 
  (∀ m : ℚ, 1/2 < m → m < 26/51 → no_lattice_points m) ∧
  ¬(∀ m : ℚ, 1/2 < m → m < 26/51 + 1/10000 → no_lattice_points m) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2024_202469


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2024_202402

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l2024_202402


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2024_202438

theorem sqrt_equation_solution :
  ∃ y : ℝ, (Real.sqrt 27 + Real.sqrt y) / Real.sqrt 75 = 2.4 → y = 243 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2024_202438


namespace NUMINAMATH_CALUDE_olivias_remaining_money_l2024_202417

-- Define the given values
def initial_amount : ℝ := 500
def groceries_cost : ℝ := 125
def shoe_original_price : ℝ := 150
def shoe_discount : ℝ := 0.2
def belt_price : ℝ := 35
def jacket_price : ℝ := 85
def exchange_rate : ℝ := 1.2

-- Define the calculation steps
def discounted_shoe_price : ℝ := shoe_original_price * (1 - shoe_discount)
def total_clothing_cost : ℝ := (discounted_shoe_price + belt_price + jacket_price) * exchange_rate
def total_spent : ℝ := groceries_cost + total_clothing_cost
def remaining_amount : ℝ := initial_amount - total_spent

-- Theorem statement
theorem olivias_remaining_money :
  remaining_amount = 87 :=
by sorry

end NUMINAMATH_CALUDE_olivias_remaining_money_l2024_202417


namespace NUMINAMATH_CALUDE_ajax_final_weight_l2024_202447

/-- Calculates the final weight in pounds after a weight loss program -/
def final_weight (initial_weight_kg : ℝ) (weight_loss_per_hour : ℝ) (hours_per_day : ℝ) (days : ℝ) : ℝ :=
  let kg_to_pounds : ℝ := 2.2
  let initial_weight_pounds : ℝ := initial_weight_kg * kg_to_pounds
  let total_weight_loss : ℝ := weight_loss_per_hour * hours_per_day * days
  initial_weight_pounds - total_weight_loss

/-- Theorem: Ajax's weight after the exercise program -/
theorem ajax_final_weight :
  final_weight 80 1.5 2 14 = 134 := by
sorry


end NUMINAMATH_CALUDE_ajax_final_weight_l2024_202447


namespace NUMINAMATH_CALUDE_triangle_side_length_l2024_202414

/-- In a triangle ABC, given that tan B = √3, AB = 3, and the area is (3√3)/2, prove that AC = √7 -/
theorem triangle_side_length (B : Real) (C : Real) (tanB : Real.tan B = Real.sqrt 3) 
  (AB : Real) (hAB : AB = 3) (area : Real) (harea : area = (3 * Real.sqrt 3) / 2) : 
  Real.sqrt ((AB^2) + (2^2) - 2 * AB * 2 * Real.cos B) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2024_202414


namespace NUMINAMATH_CALUDE_exterior_angle_square_octagon_l2024_202437

-- Define the necessary structures
structure Polygon :=
  (sides : ℕ)

-- Define the square and octagon
def square : Polygon := ⟨4⟩
def octagon : Polygon := ⟨8⟩

-- Define the function to calculate interior angle of a regular polygon
def interior_angle (p : Polygon) : ℚ :=
  180 * (p.sides - 2) / p.sides

-- Define the theorem
theorem exterior_angle_square_octagon :
  let octagon_interior_angle := interior_angle octagon
  let square_interior_angle := 90
  360 - octagon_interior_angle - square_interior_angle = 135 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_square_octagon_l2024_202437


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l2024_202445

theorem largest_number_with_equal_quotient_and_remainder :
  ∀ (A B C : ℕ),
    A = 8 * B + C →
    B = C →
    0 ≤ C ∧ C < 8 →
    A ≤ 63 ∧ (∃ (A' : ℕ), A' = 63 ∧ ∃ (B' C' : ℕ), A' = 8 * B' + C' ∧ B' = C' ∧ 0 ≤ C' ∧ C' < 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l2024_202445


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2024_202428

theorem absolute_value_inequality (x : ℝ) : 
  |((7 - x) / 5)| < 3 ↔ -8 < x ∧ x < 22 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2024_202428


namespace NUMINAMATH_CALUDE_waiter_tips_l2024_202425

/-- Calculates the total tips earned by a waiter --/
def calculate_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $27 in tips --/
theorem waiter_tips : calculate_tips 7 4 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l2024_202425


namespace NUMINAMATH_CALUDE_exists_quadratic_without_cyclic_solution_l2024_202483

/-- A quadratic polynomial function -/
def QuadraticPolynomial := ℝ → ℝ

/-- Property that checks if a function satisfies the cyclic condition for given a, b, c, d -/
def SatisfiesCyclicCondition (f : QuadraticPolynomial) (a b c d : ℝ) : Prop :=
  f a = b ∧ f b = c ∧ f c = d ∧ f d = a

/-- Theorem stating that there exists a quadratic polynomial for which no distinct a, b, c, d satisfy the cyclic condition -/
theorem exists_quadratic_without_cyclic_solution :
  ∃ f : QuadraticPolynomial, ∀ a b c d : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    ¬(SatisfiesCyclicCondition f a b c d) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_without_cyclic_solution_l2024_202483


namespace NUMINAMATH_CALUDE_fraction_value_l2024_202463

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2024_202463


namespace NUMINAMATH_CALUDE_tea_trader_profit_percentage_l2024_202436

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage
  (tea1_weight : ℝ) (tea1_cost : ℝ)
  (tea2_weight : ℝ) (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 19.2) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let cost_per_kg := total_cost / total_weight
  let profit_per_kg := sale_price - cost_per_kg
  let profit_percentage := (profit_per_kg / cost_per_kg) * 100
  profit_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_tea_trader_profit_percentage_l2024_202436


namespace NUMINAMATH_CALUDE_system_solvability_l2024_202485

/-- The first equation of the system -/
def equation1 (x y : ℝ) : Prop :=
  (x - 2)^2 + (|y - 1| - 1)^2 = 4

/-- The second equation of the system -/
def equation2 (x y a b : ℝ) : Prop :=
  y = b * |x - 1| + a

/-- The system has a solution for given a and b -/
def has_solution (a b : ℝ) : Prop :=
  ∃ x y, equation1 x y ∧ equation2 x y a b

theorem system_solvability (a : ℝ) :
  (∀ b, has_solution a b) ↔ -Real.sqrt 3 ≤ a ∧ a ≤ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_system_solvability_l2024_202485


namespace NUMINAMATH_CALUDE_permutation_combination_sum_l2024_202482

/-- Given that A_n^m = 272 and C_n^m = 136, prove that m + n = 19 -/
theorem permutation_combination_sum (m n : ℕ) 
  (h1 : m.factorial * (n - m).factorial * 272 = n.factorial)
  (h2 : m.factorial * (n - m).factorial * 136 = n.factorial) : 
  m + n = 19 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_sum_l2024_202482


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2024_202443

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 169 + y^2 / 144 = 1

-- Define a point on the ellipse
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : is_on_ellipse x y

-- Define the foci of the ellipse
structure Foci where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  are_foci : ∀ (p : PointOnEllipse), 
    Real.sqrt ((p.x - f1.1)^2 + (p.y - f1.2)^2) + 
    Real.sqrt ((p.x - f2.1)^2 + (p.y - f2.2)^2) = 26

-- The theorem to prove
theorem ellipse_triangle_perimeter 
  (p : PointOnEllipse) (f : Foci) : 
  Real.sqrt ((p.x - f.f1.1)^2 + (p.y - f.f1.2)^2) +
  Real.sqrt ((p.x - f.f2.1)^2 + (p.y - f.f2.2)^2) +
  Real.sqrt ((f.f1.1 - f.f2.1)^2 + (f.f1.2 - f.f2.2)^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2024_202443


namespace NUMINAMATH_CALUDE_max_value_of_complex_sum_l2024_202427

theorem max_value_of_complex_sum (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z + 1 + Complex.I * Real.sqrt 3) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_complex_sum_l2024_202427


namespace NUMINAMATH_CALUDE_inequality_proof_l2024_202484

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2024_202484


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2024_202420

/-- Pie-eating contest theorem -/
theorem pie_eating_contest 
  (adam bill sierra taylor : ℕ) -- Number of pies eaten by each participant
  (h1 : adam = bill + 3) -- Adam eats three more pies than Bill
  (h2 : sierra = 2 * bill) -- Sierra eats twice as many pies as Bill
  (h3 : taylor = (adam + bill + sierra) / 3) -- Taylor eats the average of Adam, Bill, and Sierra
  (h4 : sierra = 12) -- Sierra ate 12 pies
  : adam + bill + sierra + taylor = 36 ∧ adam + bill + sierra + taylor ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2024_202420


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2024_202449

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) →
  (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) →
  (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
  p^2 + q^2 + r^2 = 34/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2024_202449


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l2024_202407

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l2024_202407


namespace NUMINAMATH_CALUDE_cost_price_is_40_l2024_202450

/-- Calculates the cost price per metre of cloth given the total length, 
    total selling price, and loss per metre. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Proves that the cost price per metre is 40 given the specified conditions. -/
theorem cost_price_is_40 :
  cost_price_per_metre 500 15000 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_40_l2024_202450


namespace NUMINAMATH_CALUDE_fans_with_all_items_l2024_202411

/-- The number of fans in the stadium -/
def total_fans : ℕ := 5000

/-- The interval for t-shirt vouchers -/
def t_shirt_interval : ℕ := 60

/-- The interval for cap vouchers -/
def cap_interval : ℕ := 45

/-- The interval for water bottle vouchers -/
def water_bottle_interval : ℕ := 40

/-- Theorem: The number of fans receiving all three items is equal to the floor of total_fans divided by the LCM of the three intervals -/
theorem fans_with_all_items (total_fans t_shirt_interval cap_interval water_bottle_interval : ℕ) :
  (total_fans / Nat.lcm (Nat.lcm t_shirt_interval cap_interval) water_bottle_interval : ℕ) = 13 :=
sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l2024_202411


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2024_202472

/-- An isosceles triangle with perimeter 13 and one side length 3 -/
structure IsoscelesTriangle where
  -- The length of two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- The triangle is isosceles
  isIsosceles : side ≠ base
  -- The perimeter is 13
  perimeterIs13 : side + side + base = 13
  -- One side length is 3
  oneSideIs3 : side = 3 ∨ base = 3

/-- The base of an isosceles triangle with perimeter 13 and one side 3 must be 3 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.base = 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2024_202472


namespace NUMINAMATH_CALUDE_root_sum_squares_l2024_202488

theorem root_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → 
  (p * q + q * r + r * p = 22) →
  (p * q * r = 8) →
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 406 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2024_202488


namespace NUMINAMATH_CALUDE_sum_of_An_and_Bn_l2024_202492

/-- The sum of numbers in the n-th group of positive integers -/
def A (n : ℕ) : ℕ :=
  (2 * n - 1) * (n^2 - n + 1)

/-- The difference between the second and first number in the n-th group of cubes of natural numbers -/
def B (n : ℕ) : ℕ :=
  n^3 - (n - 1)^3

/-- Theorem stating that A_n + B_n = 2n³ for any positive integer n -/
theorem sum_of_An_and_Bn (n : ℕ) : A n + B n = 2 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_An_and_Bn_l2024_202492


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l2024_202464

theorem halloween_candy_distribution (initial_candies given_away remaining_candies : ℕ) : 
  initial_candies = 60 → given_away = 40 → remaining_candies = initial_candies - given_away → remaining_candies = 20 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l2024_202464


namespace NUMINAMATH_CALUDE_last_segment_speed_is_90_l2024_202457

-- Define the problem parameters
def total_distance : ℝ := 150
def total_time : ℝ := 135
def first_segment_time : ℝ := 45
def second_segment_time : ℝ := 45
def last_segment_time : ℝ := 45
def first_segment_speed : ℝ := 50
def second_segment_speed : ℝ := 60

-- Define the theorem
theorem last_segment_speed_is_90 :
  let last_segment_speed := 
    (total_distance - (first_segment_speed * first_segment_time / 60 + 
                       second_segment_speed * second_segment_time / 60)) / 
    (last_segment_time / 60)
  last_segment_speed = 90 := by sorry

end NUMINAMATH_CALUDE_last_segment_speed_is_90_l2024_202457


namespace NUMINAMATH_CALUDE_stamp_exchange_problem_l2024_202460

theorem stamp_exchange_problem (petya_stamps : ℕ) (kolya_stamps : ℕ) : 
  kolya_stamps = petya_stamps + 5 →
  (0.76 * kolya_stamps + 0.2 * petya_stamps : ℝ) = ((0.8 * petya_stamps + 0.24 * kolya_stamps : ℝ) - 1) →
  petya_stamps = 45 ∧ kolya_stamps = 50 := by
sorry

end NUMINAMATH_CALUDE_stamp_exchange_problem_l2024_202460


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l2024_202476

def point : ℝ × ℝ × ℝ := (2, 3, 4)
def line_point : ℝ × ℝ × ℝ := (4, 6, 5)
def line_direction : ℝ × ℝ × ℝ := (1, 3, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  distance_to_line point line_point line_direction = Real.sqrt 62 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l2024_202476


namespace NUMINAMATH_CALUDE_triangle_inequality_max_l2024_202468

theorem triangle_inequality_max (a b c x y z : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  a * y * z + b * z * x + c * x * y ≤ 
    (a * b * c) / (-a^2 - b^2 - c^2 + 2 * (a * b + b * c + c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_max_l2024_202468


namespace NUMINAMATH_CALUDE_min_absolute_sum_l2024_202410

theorem min_absolute_sum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_sum_l2024_202410


namespace NUMINAMATH_CALUDE_smallest_number_with_divisor_conditions_l2024_202487

/-- The number of positive odd integer divisors of n -/
def num_odd_divisors (n : ℕ) : ℕ := sorry

/-- The number of positive even integer divisors of n -/
def num_even_divisors (n : ℕ) : ℕ := sorry

/-- Predicate for a number satisfying the divisor conditions -/
def satisfies_divisor_conditions (n : ℕ) : Prop :=
  num_odd_divisors n = 8 ∧ num_even_divisors n = 16

theorem smallest_number_with_divisor_conditions :
  satisfies_divisor_conditions 420 ∧
  ∀ m : ℕ, m < 420 → ¬(satisfies_divisor_conditions m) := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_divisor_conditions_l2024_202487


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2024_202479

theorem arithmetic_sequence_difference (n : ℕ) (sum : ℝ) (min max : ℝ) : 
  n = 150 →
  sum = 9000 →
  min = 20 →
  max = 90 →
  let avg := sum / n
  let d := (max - min) / (2 * (n - 1))
  let L' := avg - (79 * d)
  let G' := avg + (79 * d)
  G' - L' = 6660 / 149 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2024_202479


namespace NUMINAMATH_CALUDE_domino_covering_implies_divisibility_by_three_l2024_202474

/-- Represents a domino covering of a square grid -/
structure Covering (n : ℕ) where
  red : Fin (2*n) → Fin (2*n) → Bool
  blue : Fin (2*n) → Fin (2*n) → Bool

/-- Checks if a covering is valid -/
def is_valid_covering (n : ℕ) (c : Covering n) : Prop :=
  ∀ i j, ∃! k l, (c.red i j ∧ c.red k l) ∨ (c.blue i j ∧ c.blue k l)

/-- Represents an integer assignment to each square -/
def Assignment (n : ℕ) := Fin (2*n) → Fin (2*n) → ℤ

/-- Checks if an assignment satisfies the neighbor difference condition -/
def satisfies_difference_condition (n : ℕ) (c : Covering n) (a : Assignment n) : Prop :=
  ∀ i j, ∃ k₁ l₁ k₂ l₂, 
    (c.red i j ∧ c.red k₁ l₁ ∧ c.blue i j ∧ c.blue k₂ l₂) →
    (a i j ≠ 0 ∧ a i j = a k₁ l₁ - a k₂ l₂)

theorem domino_covering_implies_divisibility_by_three (n : ℕ) 
  (h₁ : n > 0)
  (c : Covering n)
  (h₂ : is_valid_covering n c)
  (a : Assignment n)
  (h₃ : satisfies_difference_condition n c a) :
  3 ∣ n :=
sorry

end NUMINAMATH_CALUDE_domino_covering_implies_divisibility_by_three_l2024_202474


namespace NUMINAMATH_CALUDE_lindas_age_l2024_202429

theorem lindas_age (jane : ℕ) (linda : ℕ) : 
  linda = 2 * jane + 3 →
  (jane + 5) + (linda + 5) = 28 →
  linda = 13 := by
sorry

end NUMINAMATH_CALUDE_lindas_age_l2024_202429


namespace NUMINAMATH_CALUDE_fraction_3x_3x_minus_2_simplest_form_l2024_202452

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1 -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- The fraction 3x/(3x-2) -/
def f (x : ℤ) : ℚ := (3 * x) / (3 * x - 2)

/-- Theorem: The fraction 3x/(3x-2) is in its simplest form -/
theorem fraction_3x_3x_minus_2_simplest_form (x : ℤ) :
  IsSimplestForm (3 * x) (3 * x - 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_3x_3x_minus_2_simplest_form_l2024_202452


namespace NUMINAMATH_CALUDE_count_satisfying_integers_l2024_202431

def satisfies_conditions (n : ℤ) : Prop :=
  (n + 5) * (n - 5) * (n - 15) < 0 ∧ n > 7

theorem count_satisfying_integers :
  ∃ (S : Finset ℤ), (∀ n ∈ S, satisfies_conditions n) ∧ 
                    (∀ n, satisfies_conditions n → n ∈ S) ∧
                    S.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_l2024_202431


namespace NUMINAMATH_CALUDE_figure_100_squares_l2024_202442

def figure_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

theorem figure_100_squares :
  figure_squares 0 = 1 ∧
  figure_squares 1 = 7 ∧
  figure_squares 2 = 19 ∧
  figure_squares 3 = 37 →
  figure_squares 100 = 30301 := by
sorry

end NUMINAMATH_CALUDE_figure_100_squares_l2024_202442


namespace NUMINAMATH_CALUDE_tourist_cyclist_speed_l2024_202416

/-- Given the conditions of a tourist and cyclist problem, prove their speeds -/
theorem tourist_cyclist_speed :
  -- Distance from A to B
  let distance : ℝ := 24

  -- Time difference between tourist and cyclist start
  let time_diff : ℝ := 4/3

  -- Time for cyclist to overtake tourist
  let overtake_time : ℝ := 1/2

  -- Time between first and second encounter
  let encounter_interval : ℝ := 3/2

  -- Speed of cyclist
  let v_cyclist : ℝ := 16.5

  -- Speed of tourist
  let v_tourist : ℝ := 4.5

  -- Equations based on the problem conditions
  (v_cyclist * overtake_time = v_tourist * (time_diff + overtake_time)) ∧
  (v_cyclist * 2 + v_tourist * (time_diff + overtake_time + encounter_interval) = 2 * distance)

  -- Conclusion: The speeds satisfy the equations
  → v_cyclist = 16.5 ∧ v_tourist = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_tourist_cyclist_speed_l2024_202416


namespace NUMINAMATH_CALUDE_negation_equivalence_l2024_202440

theorem negation_equivalence : 
  (¬∃ x : ℝ, x^2 - 2*x + 4 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2024_202440


namespace NUMINAMATH_CALUDE_quadratic_sum_l2024_202459

/-- A quadratic function passing through (1, 3) and (2, 12) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := λ x ↦ p * x^2 + q * x + r

/-- The theorem stating that p + q + 3r = -5 for the given quadratic function -/
theorem quadratic_sum (p q r : ℝ) : 
  (QuadraticFunction p q r 1 = 3) → 
  (QuadraticFunction p q r 2 = 12) → 
  p + q + 3 * r = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2024_202459


namespace NUMINAMATH_CALUDE_pascal_triangle_specific_element_l2024_202434

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in the row of Pascal's triangle -/
def row_elements : ℕ := 56

/-- The row number (0-indexed) in Pascal's triangle -/
def row_number : ℕ := row_elements - 1

/-- The position (0-indexed) of the number we're looking for in the row -/
def position : ℕ := 23

theorem pascal_triangle_specific_element : 
  binomial row_number position = 29248649430 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_specific_element_l2024_202434


namespace NUMINAMATH_CALUDE_sin_negative_three_pi_fourths_l2024_202493

theorem sin_negative_three_pi_fourths :
  Real.sin (-3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_three_pi_fourths_l2024_202493
