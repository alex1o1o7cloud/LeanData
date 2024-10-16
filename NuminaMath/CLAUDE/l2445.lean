import Mathlib

namespace NUMINAMATH_CALUDE_median_is_5_probability_l2445_244567

def classCount : ℕ := 9
def selectedClassCount : ℕ := 5
def medianClassNumber : ℕ := 5

def probabilityMedianIs5 : ℚ :=
  (Nat.choose 4 2 * Nat.choose 4 2) / Nat.choose classCount selectedClassCount

theorem median_is_5_probability :
  probabilityMedianIs5 = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_median_is_5_probability_l2445_244567


namespace NUMINAMATH_CALUDE_function_properties_l2445_244593

def StrictlyDecreasing (f : ℝ → ℝ) :=
  ∀ x y, x < y → f x > f y

def StrictlyConvex (f : ℝ → ℝ) :=
  ∀ x y t, 0 < t → t < 1 → f (t * x + (1 - t) * y) < t * f x + (1 - t) * f y

theorem function_properties (f : ℝ → ℝ) (h1 : StrictlyDecreasing f) (h2 : StrictlyConvex f) :
  ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → x₂ < 1 →
    (x₂ * f x₁ > x₁ * f x₂) ∧
    ((f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2445_244593


namespace NUMINAMATH_CALUDE_mouse_cheese_distance_l2445_244555

/-- The point where the mouse starts getting farther from the cheese -/
def mouse_turn_point : ℚ × ℚ := (-33/17, 285/17)

/-- The location of the cheese -/
def cheese_location : ℚ × ℚ := (9, 15)

/-- The equation of the line the mouse is running on: y = -4x + 9 -/
def mouse_path (x : ℚ) : ℚ := -4 * x + 9

theorem mouse_cheese_distance :
  let (a, b) := mouse_turn_point
  -- The point is on the mouse's path
  (mouse_path a = b) ∧
  -- The line from the cheese to the point is perpendicular to the mouse's path
  ((b - 15) / (a - 9) = 1 / 4) ∧
  -- The sum of the coordinates is 252/17
  (a + b = 252 / 17) := by sorry

end NUMINAMATH_CALUDE_mouse_cheese_distance_l2445_244555


namespace NUMINAMATH_CALUDE_bounded_region_area_l2445_244526

-- Define the equation
def equation (x y : ℝ) : Prop :=
  y^2 + 3*x*y + 60*|x| = 600

-- Define the bounded region
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation p.1 p.2}

-- Define the area function
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem bounded_region_area : area bounded_region = 800 := by sorry

end NUMINAMATH_CALUDE_bounded_region_area_l2445_244526


namespace NUMINAMATH_CALUDE_probability_red_second_draw_three_five_l2445_244564

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue

/-- Represents a box of balls -/
structure Box where
  total : Nat
  red : Nat
  blue : Nat
  h_total : total = red + blue

/-- Calculates the probability of drawing a red ball on the second draw -/
def probability_red_second_draw (box : Box) : Rat :=
  (box.red * (box.total - 1) + box.blue * box.red) / (box.total * (box.total - 1))

/-- Theorem stating the probability of drawing a red ball on the second draw -/
theorem probability_red_second_draw_three_five :
  let box : Box := ⟨5, 3, 2, rfl⟩
  probability_red_second_draw box = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_second_draw_three_five_l2445_244564


namespace NUMINAMATH_CALUDE_sues_trail_mix_composition_sues_dried_fruit_percentage_proof_l2445_244584

/-- The percentage of dried fruit in Sue's trail mix -/
def sues_dried_fruit_percentage : ℝ := 70

theorem sues_trail_mix_composition :
  sues_dried_fruit_percentage = 70 :=
by
  -- Proof goes here
  sorry

/-- Sue's trail mix nuts percentage -/
def sues_nuts_percentage : ℝ := 30

/-- Jane's trail mix nuts percentage -/
def janes_nuts_percentage : ℝ := 60

/-- Jane's trail mix chocolate chips percentage -/
def janes_chocolate_percentage : ℝ := 40

/-- Combined mixture nuts percentage -/
def combined_nuts_percentage : ℝ := 45

/-- Combined mixture dried fruit percentage -/
def combined_dried_fruit_percentage : ℝ := 35

/-- Sue's trail mix consists of only nuts and dried fruit -/
axiom sues_mix_composition :
  sues_nuts_percentage + sues_dried_fruit_percentage = 100

/-- The combined mixture percentages are consistent with individual mixes -/
axiom combined_mix_consistency (s j : ℝ) :
  s > 0 ∧ j > 0 →
  sues_nuts_percentage * s + janes_nuts_percentage * j = combined_nuts_percentage * (s + j) ∧
  sues_dried_fruit_percentage * s = combined_dried_fruit_percentage * (s + j)

theorem sues_dried_fruit_percentage_proof :
  sues_dried_fruit_percentage = 70 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sues_trail_mix_composition_sues_dried_fruit_percentage_proof_l2445_244584


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_empty_solution_l2445_244594

def quadratic_inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 - 2 * x + 3 * k < 0

def solution_set_case1 (x : ℝ) : Prop :=
  x < -3 ∨ x > -1

def solution_set_case2 : Set ℝ :=
  ∅

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x, quadratic_inequality k x ↔ solution_set_case1 x) → k = -1/2 :=
sorry

theorem quadratic_inequality_empty_solution (k : ℝ) :
  (∀ x, ¬ quadratic_inequality k x) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_empty_solution_l2445_244594


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l2445_244513

-- Define the quadrilateral AMOL
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the lengths of sides
def AM (q : Quadrilateral) : ℝ := 10
def MO (q : Quadrilateral) : ℝ := 11
def OL (q : Quadrilateral) : ℝ := 12

-- Define the condition for perpendicular bisectors
def perpendicular_bisectors_condition (q : Quadrilateral) : Prop :=
  ∃ E : ℝ × ℝ, 
    E = ((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2) ∧
    (E.1 - q.A.1) * (q.B.1 - q.A.1) + (E.2 - q.A.2) * (q.B.2 - q.A.2) = 0 ∧
    (E.1 - q.C.1) * (q.D.1 - q.C.1) + (E.2 - q.C.2) * (q.D.2 - q.C.2) = 0

-- State the theorem
theorem quadrilateral_side_length (q : Quadrilateral) :
  AM q = 10 ∧ MO q = 11 ∧ OL q = 12 ∧ perpendicular_bisectors_condition q →
  Real.sqrt ((q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2) = Real.sqrt 77 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_length_l2445_244513


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2445_244597

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 > 4}

-- Define set N
def N : Set ℝ := {x | (3 - x) / (x + 1) > 0}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.compl N) = {x : ℝ | x < -2 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2445_244597


namespace NUMINAMATH_CALUDE_cone_no_rectangular_front_view_l2445_244587

-- Define the types of solids
inductive Solid
  | Cube
  | RegularTriangularPrism
  | Cylinder
  | Cone

-- Define a property for having a rectangular front view
def has_rectangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.Cube => True
  | Solid.RegularTriangularPrism => True
  | Solid.Cylinder => True
  | Solid.Cone => False

-- Theorem statement
theorem cone_no_rectangular_front_view :
  ∀ s : Solid, ¬(has_rectangular_front_view s) ↔ s = Solid.Cone :=
sorry

end NUMINAMATH_CALUDE_cone_no_rectangular_front_view_l2445_244587


namespace NUMINAMATH_CALUDE_sara_added_hundred_pencils_l2445_244577

/-- The number of pencils Sara placed in the drawer -/
def pencils_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Sara added 100 pencils to the drawer -/
theorem sara_added_hundred_pencils :
  pencils_added 115 215 = 100 := by sorry

end NUMINAMATH_CALUDE_sara_added_hundred_pencils_l2445_244577


namespace NUMINAMATH_CALUDE_mixed_number_sum_l2445_244551

theorem mixed_number_sum : (2 + 1/10) + (3 + 11/100) = 5.21 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_sum_l2445_244551


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2445_244517

theorem cubic_equation_solution (p q : ℝ) :
  ∃ x : ℝ, x^3 + p*x + q = 0 ∧
  x = -(Real.rpow ((q/2) + Real.sqrt ((q^2/4) + (p^3/27))) (1/3)) -
      (Real.rpow ((q/2) - Real.sqrt ((q^2/4) + (p^3/27))) (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2445_244517


namespace NUMINAMATH_CALUDE_min_races_for_top_three_l2445_244519

/-- Represents a race track with a maximum capacity and a set of horses -/
structure RaceTrack where
  maxCapacity : Nat
  totalHorses : Nat

/-- Represents the minimum number of races needed to find the top n fastest horses -/
def minRacesForTopN (track : RaceTrack) (n : Nat) : Nat :=
  sorry

/-- Theorem stating that for a race track with 5 horse capacity and 25 total horses,
    the minimum number of races to find the top 3 fastest horses is 7 -/
theorem min_races_for_top_three (track : RaceTrack) :
  track.maxCapacity = 5 → track.totalHorses = 25 → minRacesForTopN track 3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_races_for_top_three_l2445_244519


namespace NUMINAMATH_CALUDE_one_hundred_fiftieth_term_l2445_244527

/-- An arithmetic sequence with first term 2 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 2 + (n - 1) * 5

theorem one_hundred_fiftieth_term :
  arithmeticSequence 150 = 747 := by
  sorry

end NUMINAMATH_CALUDE_one_hundred_fiftieth_term_l2445_244527


namespace NUMINAMATH_CALUDE_at_least_one_red_not_basic_event_l2445_244583

structure Ball := (color : String)

def bag : Multiset Ball := 
  2 • {Ball.mk "red"} + 2 • {Ball.mk "white"} + 2 • {Ball.mk "black"}

def is_basic_event (event : Set (Ball × Ball)) : Prop :=
  ∃ (b1 b2 : Ball), event = {(b1, b2)}

def at_least_one_red (pair : Ball × Ball) : Prop :=
  (pair.1.color = "red") ∨ (pair.2.color = "red")

theorem at_least_one_red_not_basic_event :
  ¬ (is_basic_event {pair | at_least_one_red pair}) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_red_not_basic_event_l2445_244583


namespace NUMINAMATH_CALUDE_wombat_count_l2445_244595

/-- The number of wombats in the enclosure -/
def num_wombats : ℕ := 9

/-- The number of rheas in the enclosure -/
def num_rheas : ℕ := 3

/-- The number of times each wombat claws -/
def wombat_claws : ℕ := 4

/-- The number of times each rhea claws -/
def rhea_claws : ℕ := 1

/-- The total number of claws -/
def total_claws : ℕ := 39

theorem wombat_count : 
  num_wombats * wombat_claws + num_rheas * rhea_claws = total_claws :=
by sorry

end NUMINAMATH_CALUDE_wombat_count_l2445_244595


namespace NUMINAMATH_CALUDE_sqrt_18_equals_ab_squared_l2445_244502

theorem sqrt_18_equals_ab_squared (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) :
  Real.sqrt 18 = a * b^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_18_equals_ab_squared_l2445_244502


namespace NUMINAMATH_CALUDE_sum_of_coordinates_l2445_244571

/-- Given a point A with coordinates (m, n) that are (-3, 2) with respect to the origin, 
    prove that m + n = -1 -/
theorem sum_of_coordinates (m n : ℝ) (h : (m, n) = (-3, 2)) : m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_l2445_244571


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2445_244544

theorem fraction_equivalence : (9 : ℚ) / (7 * 53) = 0.9 / (0.7 * 53) := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2445_244544


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2445_244528

theorem consecutive_integers_sum (a b c : ℤ) : 
  b = 19 ∧ c = b + 1 ∧ a = b - 1 → a + b + c = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2445_244528


namespace NUMINAMATH_CALUDE_conversion_factor_feet_to_miles_l2445_244566

/-- Conversion factor from feet to miles -/
def feet_per_mile : ℝ := 5280

/-- Speed of the object in miles per hour -/
def speed_mph : ℝ := 68.18181818181819

/-- Distance traveled by the object in feet -/
def distance_feet : ℝ := 400

/-- Time taken by the object in seconds -/
def time_seconds : ℝ := 4

/-- Theorem stating that the conversion factor from feet to miles is 5280 -/
theorem conversion_factor_feet_to_miles :
  feet_per_mile = (distance_feet / time_seconds) / (speed_mph / 3600) := by
  sorry

#check conversion_factor_feet_to_miles

end NUMINAMATH_CALUDE_conversion_factor_feet_to_miles_l2445_244566


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2445_244569

theorem quadratic_roots_sum_and_product :
  let a : ℝ := 9
  let b : ℝ := -45
  let c : ℝ := 50
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots = 5 ∧ product_of_roots = 50 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2445_244569


namespace NUMINAMATH_CALUDE_fraction_sum_l2445_244514

theorem fraction_sum (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2445_244514


namespace NUMINAMATH_CALUDE_tricycle_wheel_revolutions_l2445_244500

/-- Calculates the number of revolutions of the back wheel of a tricycle -/
theorem tricycle_wheel_revolutions (front_radius back_radius : ℝ) (front_revolutions : ℕ) : 
  front_radius = 3 →
  back_radius = 1/2 →
  front_revolutions = 50 →
  (2 * π * front_radius * front_revolutions) / (2 * π * back_radius) = 300 := by
  sorry

#check tricycle_wheel_revolutions

end NUMINAMATH_CALUDE_tricycle_wheel_revolutions_l2445_244500


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2445_244554

noncomputable section

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -Real.sqrt 3

-- Define the asymptote of the hyperbola
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define points A and B
def A : ℝ × ℝ := (-Real.sqrt 3, 2)
def B : ℝ × ℝ := (-Real.sqrt 3, -2)

-- Define the property of equilateral triangle
def is_equilateral_triangle (F A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, hyperbola a b x y ↔ parabola x y) →
  (∀ x, directrix x → ∃ y, hyperbola a b x y) →
  (∀ x y, asymptote x y → hyperbola a b x y) →
  is_equilateral_triangle focus A B →
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2 / 2 = 1) :=
sorry

end

end NUMINAMATH_CALUDE_hyperbola_equation_l2445_244554


namespace NUMINAMATH_CALUDE_combination_value_l2445_244501

theorem combination_value (n : ℕ) (h : n * (n - 1) = 90) :
  Nat.choose (n + 2) n = 66 := by
  sorry

end NUMINAMATH_CALUDE_combination_value_l2445_244501


namespace NUMINAMATH_CALUDE_bus_stop_speed_fraction_l2445_244582

theorem bus_stop_speed_fraction (usual_time normal_delay : ℕ) (fraction : ℚ) : 
  usual_time = 28 →
  normal_delay = 7 →
  fraction * (usual_time + normal_delay) = usual_time →
  fraction = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_bus_stop_speed_fraction_l2445_244582


namespace NUMINAMATH_CALUDE_data_center_connections_l2445_244547

theorem data_center_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_data_center_connections_l2445_244547


namespace NUMINAMATH_CALUDE_hash_3_8_l2445_244599

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b ^ 2

-- Theorem statement
theorem hash_3_8 : hash 3 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_hash_3_8_l2445_244599


namespace NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2445_244529

def original_price : ℝ := 200
def tuesday_discount : ℝ := 0.40
def thursday_discount : ℝ := 0.25

theorem bicycle_price_after_discounts :
  original_price * (1 - tuesday_discount) * (1 - thursday_discount) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_after_discounts_l2445_244529


namespace NUMINAMATH_CALUDE_fraction_equality_l2445_244586

theorem fraction_equality (a b c d e : ℚ) 
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / 5 = 1 / 2)
  (h4 : b + d + 5 ≠ 0) :
  (a + c + e) / (b + d + 5) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2445_244586


namespace NUMINAMATH_CALUDE_limit_x_to_x_as_x_approaches_zero_l2445_244541

theorem limit_x_to_x_as_x_approaches_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |x^x - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_to_x_as_x_approaches_zero_l2445_244541


namespace NUMINAMATH_CALUDE_donald_drinks_nine_l2445_244536

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℕ := 2 * paul_bottles + 3

/-- Theorem stating that Donald drinks 9 bottles of juice per day -/
theorem donald_drinks_nine : donald_bottles = 9 := by
  sorry

end NUMINAMATH_CALUDE_donald_drinks_nine_l2445_244536


namespace NUMINAMATH_CALUDE_maple_trees_planted_l2445_244532

/-- The number of maple trees planted in a park --/
theorem maple_trees_planted 
  (initial_maples : ℕ) 
  (final_maples : ℕ) 
  (h : final_maples ≥ initial_maples) : 
  final_maples - initial_maples = final_maples - initial_maples :=
by sorry

end NUMINAMATH_CALUDE_maple_trees_planted_l2445_244532


namespace NUMINAMATH_CALUDE_stream_speed_l2445_244574

/-- Proves that the speed of a stream is 19 kmph given the conditions of the rowing problem -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) : 
  boat_speed = 57 →
  upstream_time = 2 * downstream_time →
  (boat_speed - 19) * (boat_speed + 19) = boat_speed^2 :=
by
  sorry

#eval (57 : ℝ) - 19 -- Expected output: 38
#eval (57 : ℝ) + 19 -- Expected output: 76
#eval (57 : ℝ)^2    -- Expected output: 3249
#eval 38 * 76       -- Expected output: 2888

end NUMINAMATH_CALUDE_stream_speed_l2445_244574


namespace NUMINAMATH_CALUDE_order_of_zeros_and_roots_l2445_244561

def f (x m n : ℝ) : ℝ := 2 * (x - m) * (x - n) - 7

theorem order_of_zeros_and_roots (m n α β : ℝ) 
  (h1 : m < n) 
  (h2 : α < β) 
  (h3 : f α m n = 0)
  (h4 : f β m n = 0) :
  α < m ∧ m < n ∧ n < β := by sorry

end NUMINAMATH_CALUDE_order_of_zeros_and_roots_l2445_244561


namespace NUMINAMATH_CALUDE_problem_solution_l2445_244538

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + 8*x

/-- The function g(x) as defined in the problem -/
def g (a : ℝ) (x : ℝ) : ℝ := f a x - 7*x - a^2 + 3

theorem problem_solution :
  (∀ x > -2, ∀ a > 0,
    (a = 1 → {x | f a x ≥ 2*x + 1} = {x | x ≥ 0}) ∧
    ({a | ∀ x > -2, g a x ≥ 0} = Set.Ioo 0 2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2445_244538


namespace NUMINAMATH_CALUDE_pinterest_group_initial_pins_l2445_244523

/-- Calculates the initial number of pins in a Pinterest group --/
def initial_pins (
  daily_contribution : ℕ)  -- Average daily contribution per person
  (weekly_deletion : ℕ)    -- Weekly deletion rate per person
  (group_size : ℕ)         -- Number of people in the group
  (days : ℕ)               -- Number of days
  (final_pins : ℕ)         -- Total pins after the given period
  : ℕ :=
  final_pins - (daily_contribution * group_size * days) + (weekly_deletion * group_size * (days / 7))

theorem pinterest_group_initial_pins :
  initial_pins 10 5 20 30 6600 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pinterest_group_initial_pins_l2445_244523


namespace NUMINAMATH_CALUDE_initial_men_correct_l2445_244504

/-- The number of men initially working in a garment industry -/
def initial_men : ℕ := 12

/-- The number of hours worked per day in the initial scenario -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked in the initial scenario -/
def initial_days : ℕ := 10

/-- The number of men in the second scenario -/
def second_men : ℕ := 24

/-- The number of hours worked per day in the second scenario -/
def second_hours_per_day : ℕ := 5

/-- The number of days worked in the second scenario -/
def second_days : ℕ := 8

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct : 
  initial_men * initial_hours_per_day * initial_days = 
  second_men * second_hours_per_day * second_days :=
by
  sorry

#check initial_men_correct

end NUMINAMATH_CALUDE_initial_men_correct_l2445_244504


namespace NUMINAMATH_CALUDE_problem_solution_l2445_244573

theorem problem_solution (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t)
  (h2 : y = 5 * t + 6)
  (h3 : x = -2) :
  y = 37 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2445_244573


namespace NUMINAMATH_CALUDE_final_number_is_81_l2445_244559

/-- Represents the elimination process on a list of numbers -/
def elimination_process (n : ℕ) : ℕ :=
  if n ≤ 3 then n else
  let m := elimination_process ((2 * n + 3) / 3)
  if m * 3 > n then m else m + 1

/-- The theorem stating that 81 is the final remaining number -/
theorem final_number_is_81 : elimination_process 200 = 81 := by
  sorry

end NUMINAMATH_CALUDE_final_number_is_81_l2445_244559


namespace NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l2445_244591

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.5

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 68

/-- The number of oxygen atoms in the compound -/
def n : ℕ := 2

theorem oxygen_atoms_in_compound :
  ∃ (n : ℕ), n = 2 ∧
  total_weight = hydrogen_weight + chlorine_weight + n * oxygen_weight :=
sorry

end NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l2445_244591


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l2445_244524

/-- A sequence of natural numbers -/
def NatSequence := ℕ → ℕ

/-- The sum of the first k terms of a sequence -/
def PartialSum (a : NatSequence) (k : ℕ) : ℕ :=
  (Finset.range k).sum (fun i => a i)

/-- Predicate for a sequence containing each natural number exactly once -/
def ContainsEachNatOnce (a : NatSequence) : Prop :=
  ∀ n : ℕ, ∃! k : ℕ, a k = n

/-- Predicate for the divisibility condition -/
def DivisibilityCondition (a : NatSequence) : Prop :=
  ∀ k : ℕ, k ∣ PartialSum a k

theorem existence_of_special_sequence :
  ∃ a : NatSequence, ContainsEachNatOnce a ∧ DivisibilityCondition a := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l2445_244524


namespace NUMINAMATH_CALUDE_boat_meeting_times_l2445_244505

/-- Represents the meeting time of two boats given their speeds and the river current. -/
def meeting_time (speed_A speed_C current distance : ℝ) : Set ℝ :=
  let effective_speed_A := speed_A + current
  let effective_speed_C_against := speed_C - current
  let effective_speed_C_with := speed_C + current
  let time_opposite := distance / (effective_speed_A + effective_speed_C_against)
  let time_same_direction := distance / (effective_speed_A - effective_speed_C_with)
  {time_opposite, time_same_direction}

/-- The theorem stating the meeting times of the boats under given conditions. -/
theorem boat_meeting_times :
  meeting_time 7 3 2 20 = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_boat_meeting_times_l2445_244505


namespace NUMINAMATH_CALUDE_symmetric_difference_eq_zero_three_l2445_244563

-- Define the function f
def f (n : ℕ) : ℕ := 2 * n + 1

-- Define the sets P and Q
def P : Set ℕ := {1, 2, 3, 4, 5}
def Q : Set ℕ := {3, 4, 5, 6, 7}

-- Define sets A and B
def A : Set ℕ := {n : ℕ | f n ∈ P}
def B : Set ℕ := {n : ℕ | f n ∈ Q}

-- State the theorem
theorem symmetric_difference_eq_zero_three :
  (A ∩ (Set.univ \ B)) ∪ (B ∩ (Set.univ \ A)) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_eq_zero_three_l2445_244563


namespace NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l2445_244510

theorem function_inequality_implies_upper_bound (a : ℝ) :
  (∀ x1 ∈ Set.Icc (1/2 : ℝ) 3, ∃ x2 ∈ Set.Icc 2 3, x1 + 4/x1 ≥ 2^x2 + a) →
  a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_upper_bound_l2445_244510


namespace NUMINAMATH_CALUDE_max_visibility_score_l2445_244512

/-- Represents a configuration of towers --/
structure TowerConfig where
  height1 : ℕ  -- Number of towers with height 1
  height2 : ℕ  -- Number of towers with height 2

/-- The total height of all towers is 30 --/
def validConfig (config : TowerConfig) : Prop :=
  config.height1 + 2 * config.height2 = 30

/-- Calculate the visibility score for a given configuration --/
def visibilityScore (config : TowerConfig) : ℕ :=
  config.height1 * config.height2

/-- Theorem: The maximum visibility score is 112 and is achieved
    when all towers are either height 1 or 2 --/
theorem max_visibility_score :
  ∃ (config : TowerConfig), validConfig config ∧
    visibilityScore config = 112 ∧
    (∀ (other : TowerConfig), validConfig other →
      visibilityScore other ≤ visibilityScore config) :=
by sorry

end NUMINAMATH_CALUDE_max_visibility_score_l2445_244512


namespace NUMINAMATH_CALUDE_no_polynomial_iteration_fixed_points_l2445_244576

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- A function from integers to integers -/
def IntFunction := ℤ → ℤ

/-- n-fold application of a function -/
def iterate (f : IntFunction) (n : ℕ) : IntFunction := sorry

/-- The number of fixed points of a function -/
def fixedPointCount (f : IntFunction) : ℕ := sorry

/-- Main theorem -/
theorem no_polynomial_iteration_fixed_points :
  ¬ ∃ (P : IntPolynomial) (T : IntFunction),
    degree P ≥ 1 ∧
    (∀ n : ℕ, n ≥ 1 → fixedPointCount (iterate T n) = P n) :=
sorry

end NUMINAMATH_CALUDE_no_polynomial_iteration_fixed_points_l2445_244576


namespace NUMINAMATH_CALUDE_cubic_integer_root_l2445_244550

theorem cubic_integer_root (p q : ℤ) : 
  (∃ (x : ℝ), x^3 - p*x - q = 0 ∧ x = 4 - Real.sqrt 10) →
  ((-8 : ℝ)^3 - p*(-8) - q = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l2445_244550


namespace NUMINAMATH_CALUDE_student_arrangement_problem_l2445_244590

/-- The number of ways to arrange n elements --/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k elements from n elements --/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem student_arrangement_problem :
  let total_students : ℕ := 7
  let special_positions : ℕ := 3  -- left, right, middle for A in part (I)
  let remaining_students : ℕ := total_students - 1
  
  -- Part (I)
  (combinations special_positions 1 * permutations remaining_students = 2160) ∧
  
  -- Part (II)
  (permutations total_students - 
   (combinations (total_students - 1) 1 * permutations (total_students - 2)) = 3720)
  := by sorry

end NUMINAMATH_CALUDE_student_arrangement_problem_l2445_244590


namespace NUMINAMATH_CALUDE_point_coordinates_l2445_244553

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem stating the coordinates of point P given the conditions -/
theorem point_coordinates (P : Point) 
  (h1 : SecondQuadrant P) 
  (h2 : DistanceToXAxis P = 4) 
  (h3 : DistanceToYAxis P = 3) : 
  P.x = -3 ∧ P.y = 4 := by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l2445_244553


namespace NUMINAMATH_CALUDE_five_n_plus_three_composite_l2445_244537

theorem five_n_plus_three_composite (n : ℕ+) 
  (h1 : ∃ k : ℕ+, 2 * n + 1 = k^2) 
  (h2 : ∃ m : ℕ+, 3 * n + 1 = m^2) : 
  ¬(Nat.Prime (5 * n + 3)) :=
by sorry

end NUMINAMATH_CALUDE_five_n_plus_three_composite_l2445_244537


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2445_244596

theorem arithmetic_geometric_sequence (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →  -- non-zero common difference
  a 1 = 1 →  -- a_1 = 1
  (a 2) * (a 5) = (a 4)^2 →  -- a_2, a_4, and a_5 form a geometric sequence
  d = 1/5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2445_244596


namespace NUMINAMATH_CALUDE_inequality_proof_l2445_244543

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (a*b + 1) / (a + b)^2 + (b*c + 1) / (b + c)^2 + (c*a + 1) / (c + a)^2 ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2445_244543


namespace NUMINAMATH_CALUDE_average_of_dataset_l2445_244507

def dataset : List ℝ := [5, 9, 9, 3, 4]

theorem average_of_dataset : 
  (dataset.sum / dataset.length : ℝ) = 6 := by sorry

end NUMINAMATH_CALUDE_average_of_dataset_l2445_244507


namespace NUMINAMATH_CALUDE_suhwan_milk_consumption_l2445_244570

/-- Amount of milk Suhwan drinks per time in liters -/
def milk_per_time : ℝ := 0.2

/-- Number of times Suhwan drinks milk per day -/
def times_per_day : ℕ := 3

/-- Number of days in a week -/
def days_in_week : ℕ := 7

/-- Suhwan's weekly milk consumption in liters -/
def weekly_milk_consumption : ℝ :=
  milk_per_time * (times_per_day : ℝ) * (days_in_week : ℝ)

theorem suhwan_milk_consumption :
  weekly_milk_consumption = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_suhwan_milk_consumption_l2445_244570


namespace NUMINAMATH_CALUDE_reduction_equivalence_l2445_244516

def operation (seq : Vector ℤ 8) : Vector ℤ 8 :=
  Vector.ofFn (λ i => |seq.get i - seq.get ((i + 1) % 8)|)

def all_equal (seq : Vector ℤ 8) : Prop :=
  ∀ i j, seq.get i = seq.get j

def all_zero (seq : Vector ℤ 8) : Prop :=
  ∀ i, seq.get i = 0

def reduces_to_equal (init : Vector ℤ 8) : Prop :=
  ∃ n : ℕ, all_equal (n.iterate operation init)

def reduces_to_zero (init : Vector ℤ 8) : Prop :=
  ∃ n : ℕ, all_zero (n.iterate operation init)

theorem reduction_equivalence (init : Vector ℤ 8) :
  reduces_to_equal init ↔ reduces_to_zero init :=
sorry

end NUMINAMATH_CALUDE_reduction_equivalence_l2445_244516


namespace NUMINAMATH_CALUDE_marks_interest_earned_l2445_244572

/-- Calculates the interest earned on an investment with annual compound interest -/
def interestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- The interest earned on Mark's investment -/
theorem marks_interest_earned :
  let principal : ℝ := 1500
  let rate : ℝ := 0.02
  let years : ℕ := 8
  abs (interestEarned principal rate years - 257.49) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_marks_interest_earned_l2445_244572


namespace NUMINAMATH_CALUDE_sandy_has_24_red_balloons_l2445_244585

/-- The number of red balloons Sandy has -/
def sandys_red_balloons (saras_red_balloons total_red_balloons : ℕ) : ℕ :=
  total_red_balloons - saras_red_balloons

/-- Theorem stating that Sandy has 24 red balloons -/
theorem sandy_has_24_red_balloons :
  sandys_red_balloons 31 55 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sandy_has_24_red_balloons_l2445_244585


namespace NUMINAMATH_CALUDE_probability_two_students_together_l2445_244521

/-- The probability of two specific students standing together in a row of 4 students -/
theorem probability_two_students_together (n : ℕ) (h : n = 4) : 
  (2 * 3 * 2 * 1 : ℚ) / (n * (n - 1) * (n - 2) * (n - 3)) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_students_together_l2445_244521


namespace NUMINAMATH_CALUDE_kat_strength_training_frequency_l2445_244540

/-- Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_duration : ℝ  -- Duration of each strength training session in hours
  strength_frequency : ℝ  -- Number of strength training sessions per week
  boxing_duration : ℝ     -- Duration of each boxing session in hours
  boxing_frequency : ℝ    -- Number of boxing sessions per week
  total_hours : ℝ         -- Total training hours per week

/-- Theorem stating that Kat does strength training 3 times a week -/
theorem kat_strength_training_frequency 
  (schedule : TrainingSchedule) 
  (h1 : schedule.strength_duration = 1)
  (h2 : schedule.boxing_duration = 1.5)
  (h3 : schedule.boxing_frequency = 4)
  (h4 : schedule.total_hours = 9)
  (h5 : schedule.total_hours = schedule.strength_duration * schedule.strength_frequency + 
                               schedule.boxing_duration * schedule.boxing_frequency) :
  schedule.strength_frequency = 3 := by
  sorry

#check kat_strength_training_frequency

end NUMINAMATH_CALUDE_kat_strength_training_frequency_l2445_244540


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_l2445_244568

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  side : ℝ

/-- The diagonal of an isosceles trapezoid -/
def diagonal (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal of the specified isosceles trapezoid is 13 units -/
theorem isosceles_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { base1 := 24, base2 := 12, side := 13 }
  diagonal t = 13 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_diagonal_l2445_244568


namespace NUMINAMATH_CALUDE_postman_pete_mileage_l2445_244552

def pedometer_max : ℕ := 99999
def flips_in_year : ℕ := 50
def last_day_steps : ℕ := 25000
def steps_per_mile : ℕ := 1500

def total_steps : ℕ := flips_in_year * (pedometer_max + 1) + last_day_steps

def miles_walked : ℚ := total_steps / steps_per_mile

theorem postman_pete_mileage :
  ∃ (m : ℕ), m ≥ 3000 ∧ m ≤ 4000 ∧ 
  ∀ (n : ℕ), (n ≥ 3000 ∧ n ≤ 4000) → |miles_walked - m| ≤ |miles_walked - n| :=
sorry

end NUMINAMATH_CALUDE_postman_pete_mileage_l2445_244552


namespace NUMINAMATH_CALUDE_two_player_three_point_probability_l2445_244560

/-- The probability that at least one of two players makes both of their two three-point shots -/
theorem two_player_three_point_probability (p_a p_b : ℝ) 
  (h_a : p_a = 0.4) (h_b : p_b = 0.5) : 
  1 - (1 - p_a^2) * (1 - p_b^2) = 0.37 := by
  sorry

end NUMINAMATH_CALUDE_two_player_three_point_probability_l2445_244560


namespace NUMINAMATH_CALUDE_ginos_popsicle_sticks_l2445_244581

def my_popsicle_sticks : ℕ := 50
def total_popsicle_sticks : ℕ := 113

theorem ginos_popsicle_sticks :
  total_popsicle_sticks - my_popsicle_sticks = 63 := by sorry

end NUMINAMATH_CALUDE_ginos_popsicle_sticks_l2445_244581


namespace NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l2445_244530

theorem x_cube_plus_reciprocal (x : ℝ) (h : 11 = x^6 + 1/x^6) : x^3 + 1/x^3 = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l2445_244530


namespace NUMINAMATH_CALUDE_ellipse_angle_bisector_l2445_244545

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- Definition of a point being on a chord through F -/
def is_on_chord_through_F (x y : ℝ) : Prop := 
  ∃ (m : ℝ), y = m * (x - 2)

/-- Definition of the angle equality condition -/
def angle_equality (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - p)) = -(y₂ / (x₂ - p))

/-- The main theorem -/
theorem ellipse_angle_bisector :
  ∃! (p : ℝ), p > 0 ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
    is_on_chord_through_F x₁ y₁ ∧ is_on_chord_through_F x₂ y₂ ∧
    x₁ ≠ x₂ →
    angle_equality p x₁ y₁ x₂ y₂) ∧
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_angle_bisector_l2445_244545


namespace NUMINAMATH_CALUDE_cats_in_meow_and_paw_l2445_244592

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := meow_cats + paw_cats

/-- Theorem stating that the total number of cats in Cat Cafe Meow and Cat Cafe Paw is 40 -/
theorem cats_in_meow_and_paw : total_cats = 40 := by
  sorry

end NUMINAMATH_CALUDE_cats_in_meow_and_paw_l2445_244592


namespace NUMINAMATH_CALUDE_dormitory_students_l2445_244557

theorem dormitory_students (F S : ℝ) (h1 : F + S = 1) 
  (h2 : 4/5 * F = F - F/5) 
  (h3 : S - 4 * (F/5) = 4/5 * F) 
  (h4 : S - (S - 4 * (F/5)) = 0.2) : 
  S = 2/3 := by sorry

end NUMINAMATH_CALUDE_dormitory_students_l2445_244557


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2445_244534

theorem fraction_equation_solution (a b : ℝ) (h1 : a ≠ b) (h2 : b = 1) 
  (h3 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 4) : 
  a / b = (17 + Real.sqrt 269) / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2445_244534


namespace NUMINAMATH_CALUDE_quadratic_function_range_quadratic_function_range_restricted_l2445_244511

theorem quadratic_function_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + 2 ≥ a) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

theorem quadratic_function_range_restricted (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_quadratic_function_range_restricted_l2445_244511


namespace NUMINAMATH_CALUDE_middle_card_is_five_l2445_244503

def is_valid_trio (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c = 16

def leftmost_uncertain (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ is_valid_trio a b₁ c₁ ∧ is_valid_trio a b₂ c₂

def rightmost_uncertain (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ is_valid_trio a₁ b₁ c ∧ is_valid_trio a₂ b₂ c

def middle_uncertain (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, (a₁ ≠ a₂ ∨ c₁ ≠ c₂) ∧ is_valid_trio a₁ b c₁ ∧ is_valid_trio a₂ b c₂

theorem middle_card_is_five :
  ∀ a b c : ℕ,
    is_valid_trio a b c →
    leftmost_uncertain a →
    rightmost_uncertain c →
    middle_uncertain b →
    b = 5 := by sorry

end NUMINAMATH_CALUDE_middle_card_is_five_l2445_244503


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2445_244542

/-- Proves that the first discount percentage is 10% for an article with a given price and two successive discounts -/
theorem first_discount_percentage
  (normal_price : ℝ)
  (first_discount : ℝ)
  (second_discount : ℝ)
  (h1 : normal_price = 174.99999999999997)
  (h2 : first_discount = 0.1)
  (h3 : second_discount = 0.2)
  : first_discount = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2445_244542


namespace NUMINAMATH_CALUDE_cubic_eq_given_quadratic_l2445_244509

theorem cubic_eq_given_quadratic (x : ℝ) :
  x^2 + 5*x - 990 = 0 → x^3 + 6*x^2 - 985*x + 1012 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_cubic_eq_given_quadratic_l2445_244509


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l2445_244578

theorem delta_max_success_ratio 
  (charlie_day1_score charlie_day1_total : ℕ)
  (charlie_day2_score charlie_day2_total : ℕ)
  (delta_day1_score delta_day1_total : ℕ)
  (delta_day2_score delta_day2_total : ℕ)
  (h1 : charlie_day1_score = 200)
  (h2 : charlie_day1_total = 360)
  (h3 : charlie_day2_score = 160)
  (h4 : charlie_day2_total = 240)
  (h5 : delta_day1_score > 0)
  (h6 : delta_day2_score > 0)
  (h7 : delta_day1_total + delta_day2_total = 600)
  (h8 : delta_day1_total ≠ 360)
  (h9 : (delta_day1_score : ℚ) / delta_day1_total < (charlie_day1_score : ℚ) / charlie_day1_total)
  (h10 : (delta_day2_score : ℚ) / delta_day2_total < (charlie_day2_score : ℚ) / charlie_day2_total)
  (h11 : (charlie_day1_score + charlie_day2_score : ℚ) / (charlie_day1_total + charlie_day2_total) = 3/5) :
  (delta_day1_score + delta_day2_score : ℚ) / (delta_day1_total + delta_day2_total) ≤ 166/600 :=
by sorry


end NUMINAMATH_CALUDE_delta_max_success_ratio_l2445_244578


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2445_244565

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * (x + 2 * y) - 5 * y = -1) ∧ (3 * (x - y) + y = 2) ∧ (x = -4) ∧ (y = -7) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2445_244565


namespace NUMINAMATH_CALUDE_trapezoid_long_side_length_l2445_244508

/-- Represents a square divided into two trapezoids and a quadrilateral -/
structure DividedSquare where
  side_length : ℝ
  segment_length : ℝ
  trapezoid_long_side : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : DividedSquare) : Prop :=
  s.side_length = 2 ∧
  s.segment_length = 1 ∧
  (s.trapezoid_long_side + s.segment_length) * s.segment_length / 2 = s.side_length^2 / 3

/-- The theorem to be proved -/
theorem trapezoid_long_side_length (s : DividedSquare) :
  problem_conditions s → s.trapezoid_long_side = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_long_side_length_l2445_244508


namespace NUMINAMATH_CALUDE_function_domain_range_equality_l2445_244556

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The theorem stating that b = 2 for the given conditions -/
theorem function_domain_range_equality (b : ℝ) (h1 : b > 1) 
  (h2 : Set.Icc 1 b = Set.range f)
  (h3 : ∀ x, x ∈ Set.Icc 1 b → f x ∈ Set.Icc 1 b) : b = 2 := by
  sorry

#check function_domain_range_equality

end NUMINAMATH_CALUDE_function_domain_range_equality_l2445_244556


namespace NUMINAMATH_CALUDE_parallel_transitive_l2445_244533

-- Define the type for lines
def Line : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation between lines
def parallel (a b : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l2445_244533


namespace NUMINAMATH_CALUDE_max_edges_l2445_244520

/-- A square partitioned into convex polygons -/
structure PartitionedSquare where
  n : ℕ  -- number of polygons
  v : ℕ  -- number of vertices
  e : ℕ  -- number of edges

/-- Euler's theorem for partitioned square -/
axiom euler_theorem (ps : PartitionedSquare) : ps.v - ps.e + ps.n = 1

/-- The degree of each vertex is at least 2, except for at most 4 corner vertices -/
axiom vertex_degree (ps : PartitionedSquare) : 2 * ps.e ≥ 3 * ps.v - 4

/-- Theorem: Maximum number of edges in a partitioned square -/
theorem max_edges (ps : PartitionedSquare) : ps.e ≤ 3 * ps.n + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_edges_l2445_244520


namespace NUMINAMATH_CALUDE_davids_daughter_age_l2445_244549

/-- David's current age -/
def david_age : ℕ := 40

/-- Number of years in the future when David's age will be twice his daughter's -/
def years_until_double : ℕ := 16

/-- David's daughter's current age -/
def daughter_age : ℕ := 12

/-- Theorem stating that David's daughter is 12 years old today -/
theorem davids_daughter_age :
  daughter_age = 12 ∧
  david_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_davids_daughter_age_l2445_244549


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_coverage_l2445_244589

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tilesNeeded (region : Dimensions) (tile : Dimensions) : ℕ :=
  (area region + area tile - 1) / area tile

theorem min_tiles_for_floor_coverage :
  let tile := Dimensions.mk 2 6
  let region := Dimensions.mk (feetToInches 3) (feetToInches 4)
  tilesNeeded region tile = 144 := by
    sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_coverage_l2445_244589


namespace NUMINAMATH_CALUDE_range_of_x2_plus_y2_l2445_244575

theorem range_of_x2_plus_y2 (x y : ℝ) (h : x^2 - 2*x*y + 5*y^2 = 4) :
  ∃ (min max : ℝ), min = 3 - Real.sqrt 5 ∧ max = 3 + Real.sqrt 5 ∧
  (min ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ max) ∧
  ∃ (x1 y1 x2 y2 : ℝ), x1^2 - 2*x1*y1 + 5*y1^2 = 4 ∧
                       x2^2 - 2*x2*y2 + 5*y2^2 = 4 ∧
                       x1^2 + y1^2 = min ∧
                       x2^2 + y2^2 = max :=
by sorry

end NUMINAMATH_CALUDE_range_of_x2_plus_y2_l2445_244575


namespace NUMINAMATH_CALUDE_glass_volume_l2445_244531

/-- The volume of a glass given pessimist and optimist perspectives --/
theorem glass_volume (V : ℝ) (h1 : V > 0) : 
  let pessimist_empty_percent : ℝ := 0.6
  let optimist_full_percent : ℝ := 0.6
  let water_difference : ℝ := 46
  (optimist_full_percent * V) - ((1 - pessimist_empty_percent) * V) = water_difference →
  V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_l2445_244531


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l2445_244539

/-- A parabola with equation y = x^2 - 4x + c has its vertex on the x-axis if and only if c = 4 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + c = 0 ∧ ∀ y : ℝ, y^2 - 4*y + c ≥ x^2 - 4*x + c) ↔ c = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l2445_244539


namespace NUMINAMATH_CALUDE_derivative_sqrt_derivative_log2_l2445_244588

-- Define the derivative of square root
theorem derivative_sqrt (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by sorry

-- Define the derivative of log base 2
theorem derivative_log2 (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_derivative_log2_l2445_244588


namespace NUMINAMATH_CALUDE_tom_vegetable_ratio_l2445_244548

/-- The ratio of broccoli to carrots eaten by Tom -/
def broccoli_to_carrots_ratio : ℚ := by sorry

theorem tom_vegetable_ratio :
  let carrot_calories_per_pound : ℚ := 51
  let carrot_amount : ℚ := 1
  let broccoli_calories_per_pound : ℚ := carrot_calories_per_pound / 3
  let total_calories : ℚ := 85
  let broccoli_amount : ℚ := (total_calories - carrot_calories_per_pound * carrot_amount) / broccoli_calories_per_pound
  broccoli_to_carrots_ratio = broccoli_amount / carrot_amount := by sorry

end NUMINAMATH_CALUDE_tom_vegetable_ratio_l2445_244548


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2445_244546

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 28) ∧ 
  (0 ≤ N) ∧ (N ≤ 999) ∧
  (36 * N < 2000) ∧
  (72 * N ≥ 2000) ∧
  (∀ M : ℕ, M < N → 
    (M = 0) ∨ (M > 999) ∨ 
    (36 * M ≥ 2000) ∨ 
    (72 * M < 2000)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2445_244546


namespace NUMINAMATH_CALUDE_function_upper_bound_l2445_244525

theorem function_upper_bound (x : ℝ) (h : x ≥ 1) : (1 + Real.log x) / x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l2445_244525


namespace NUMINAMATH_CALUDE_angle_sine_relation_l2445_244558

theorem angle_sine_relation (A B : Real) (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2) :
  A > B ↔ Real.sin A > Real.sin B :=
sorry

end NUMINAMATH_CALUDE_angle_sine_relation_l2445_244558


namespace NUMINAMATH_CALUDE_boris_books_l2445_244506

theorem boris_books (boris_initial : ℕ) (cameron_initial : ℕ) : 
  cameron_initial = 30 →
  (3 * boris_initial / 4 : ℚ) + (2 * cameron_initial / 3 : ℚ) = 38 →
  boris_initial = 24 :=
by sorry

end NUMINAMATH_CALUDE_boris_books_l2445_244506


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2445_244598

theorem fraction_to_decimal : 
  (52 : ℚ) / 180 = 0.1444444444444444 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2445_244598


namespace NUMINAMATH_CALUDE_particle_movement_probability_reach_origin_l2445_244579

/-- Probability of reaching (0,0) from (x,y) before hitting any other point on the axes -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The particle's movement rules and starting position -/
theorem particle_movement (x y : ℕ) (h : x > 0 ∧ y > 0) :
  P x y = (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3 :=
sorry

/-- The probability of reaching (0,0) from (5,5) -/
theorem probability_reach_origin : P 5 5 = 381 / 2187 :=
sorry

end NUMINAMATH_CALUDE_particle_movement_probability_reach_origin_l2445_244579


namespace NUMINAMATH_CALUDE_sin_65pi_over_6_l2445_244518

theorem sin_65pi_over_6 : Real.sin (65 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_65pi_over_6_l2445_244518


namespace NUMINAMATH_CALUDE_rent_increase_group_size_l2445_244522

theorem rent_increase_group_size :
  ∀ (n : ℕ) (initial_average rent_increase new_average original_rent : ℚ),
    initial_average = 800 →
    new_average = 880 →
    original_rent = 1600 →
    rent_increase = 0.2 * original_rent →
    n * new_average = n * initial_average + rent_increase →
    n = 4 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_group_size_l2445_244522


namespace NUMINAMATH_CALUDE_dance_off_combined_time_l2445_244580

/-- Given John and James' dancing schedules, prove their combined dancing time is 20 hours --/
theorem dance_off_combined_time (john_first_session : ℝ) (john_break : ℝ) (john_second_session : ℝ) 
  (james_extra_fraction : ℝ) : 
  john_first_session = 3 ∧ 
  john_break = 1 ∧ 
  john_second_session = 5 ∧ 
  james_extra_fraction = 1/3 → 
  (john_first_session + john_second_session) + 
  ((john_first_session + john_break + john_second_session) + 
   (john_first_session + john_break + john_second_session) * james_extra_fraction) = 20 := by
sorry

end NUMINAMATH_CALUDE_dance_off_combined_time_l2445_244580


namespace NUMINAMATH_CALUDE_not_proportional_D_l2445_244535

-- Define the equations
def equation_A (x y : ℝ) : Prop := x + y = 5
def equation_B (x y : ℝ) : Prop := 4 * x * y = 12
def equation_C (x y : ℝ) : Prop := x = 3 * y
def equation_D (x y : ℝ) : Prop := 4 * x + 2 * y = 8
def equation_E (x y : ℝ) : Prop := x / y = 4

-- Define direct and inverse proportionality
def directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

-- Theorem statement
theorem not_proportional_D :
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_A x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_B x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_C x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  (∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_E x y ↔ y = f x) ∧
                (directly_proportional f ∨ inversely_proportional f)) ∧
  ¬(∃ f : ℝ → ℝ, (∀ x y : ℝ, equation_D x y ↔ y = f x) ∧
                 (directly_proportional f ∨ inversely_proportional f)) :=
by sorry

end NUMINAMATH_CALUDE_not_proportional_D_l2445_244535


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l2445_244562

theorem remainder_6n_mod_4 (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l2445_244562


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2445_244515

/-- A quadratic function f(x) = x^2 + ax + b with specific properties -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

theorem quadratic_function_properties (a b : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), QuadraticFunction a b y ≥ QuadraticFunction a b x ∧ QuadraticFunction a b x = 2) →
  (∀ (x : ℝ), QuadraticFunction a b (2 - x) = QuadraticFunction a b x) →
  (∃ (m n : ℝ), m < n ∧
    (∀ (x : ℝ), m ≤ x ∧ x ≤ n → QuadraticFunction a b x ≤ 6) ∧
    (∃ (x : ℝ), m ≤ x ∧ x ≤ n ∧ QuadraticFunction a b x = 6)) →
  (∃ (m n : ℝ), n - m = 4 ∧
    ∀ (m' n' : ℝ), (∀ (x : ℝ), m' ≤ x ∧ x ≤ n' → QuadraticFunction a b x ≤ 6) →
    (∃ (x : ℝ), m' ≤ x ∧ x ≤ n' ∧ QuadraticFunction a b x = 6) →
    n' - m' ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2445_244515
