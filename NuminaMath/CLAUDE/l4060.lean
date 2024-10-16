import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_l4060_406072

/-- Triangle inequality proof -/
theorem triangle_inequality (a b c s R r : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 → r > 0 →
  s = (a + b + c) / 2 →
  a + b > c → b + c > a → c + a > b →
  (a / (s - a)) + (b / (s - b)) + (c / (s - c)) ≥ 3 * R / r := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4060_406072


namespace NUMINAMATH_CALUDE_triangle_construction_l4060_406071

-- Define the necessary structures and properties
structure Point where
  x : ℝ
  y : ℝ

def nonCollinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def isOrthocenter (P A B C : Point) : Prop :=
  ((B.y - A.y) * (P.x - A.x) + (A.x - B.x) * (P.y - A.y) = 0) ∧
  ((C.y - B.y) * (P.x - B.x) + (B.x - C.x) * (P.y - B.y) = 0) ∧
  ((A.y - C.y) * (P.x - C.x) + (C.x - A.x) * (P.y - C.y) = 0)

-- State the theorem
theorem triangle_construction (M N P : Point) (h : nonCollinear M N P) :
  ∃ (A B C : Point),
    (isMidpoint M A B ∨ isMidpoint M B C ∨ isMidpoint M A C) ∧
    (isMidpoint N A B ∨ isMidpoint N B C ∨ isMidpoint N A C) ∧
    (isMidpoint M A B → isMidpoint N A C ∨ isMidpoint N B C) ∧
    (isMidpoint M B C → isMidpoint N A B ∨ isMidpoint N A C) ∧
    (isMidpoint M A C → isMidpoint N A B ∨ isMidpoint N B C) ∧
    isOrthocenter P A B C :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_l4060_406071


namespace NUMINAMATH_CALUDE_triangle_sine_relation_l4060_406037

theorem triangle_sine_relation (A B C : ℝ) (h : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) :
  Real.sin (A + π / 4) = Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_relation_l4060_406037


namespace NUMINAMATH_CALUDE_total_profit_is_54000_l4060_406043

/-- Calculates the total profit given the investments and Jose's profit share -/
def calculate_total_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  let tom_ratio : ℕ := tom_investment * tom_months
  let jose_ratio : ℕ := jose_investment * jose_months
  let total_ratio : ℕ := tom_ratio + jose_ratio
  (jose_profit * total_ratio) / jose_ratio

/-- The total profit for Tom and Jose's business venture -/
theorem total_profit_is_54000 :
  calculate_total_profit 30000 12 45000 10 30000 = 54000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_54000_l4060_406043


namespace NUMINAMATH_CALUDE_ellipse_properties_l4060_406021

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y k m : ℝ) : Prop := y = k * x + m

-- Define the conditions
def conditions (a b k m : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * b = 2 ∧ (a^2 - b^2) / a^2 = 1/2

-- Define the perpendicular bisector condition
def perp_bisector_condition (k m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ (Real.sqrt 2) 1 ∧
    ellipse x₂ y₂ (Real.sqrt 2) 1 ∧
    line x₁ y₁ k m ∧
    line x₂ y₂ k m ∧
    (y₁ + y₂) / 2 + 1/2 = -1/k * ((x₁ + x₂) / 2)

-- Define the theorem
theorem ellipse_properties (a b k m : ℝ) :
  conditions a b k m →
  (∀ x y, ellipse x y a b ↔ ellipse x y (Real.sqrt 2) 1) ∧
  (perp_bisector_condition k m → 2 * k^2 + 1 = 2 * m) ∧
  (∃ (S : ℝ → ℝ), (∀ k m, perp_bisector_condition k m → S m ≤ Real.sqrt 2 / 2) ∧
                  (∃ k₀ m₀, perp_bisector_condition k₀ m₀ ∧ S m₀ = Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4060_406021


namespace NUMINAMATH_CALUDE_problem_1_l4060_406005

theorem problem_1 : 2 * Real.tan (π / 3) - |Real.sqrt 3 - 2| - 3 * Real.sqrt 3 + (1 / 3)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l4060_406005


namespace NUMINAMATH_CALUDE_soccer_league_games_times_each_team_plays_l4060_406076

/-- 
Proves that in a soccer league with 12 teams, where a total of 66 games are played, 
each team plays every other team exactly 2 times.
-/
theorem soccer_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 12) (h2 : total_games = 66) :
  (n * (n - 1) * 2) / 2 = total_games :=
by sorry

/-- 
Proves that the number of times each team plays others is 2.
-/
theorem times_each_team_plays (n : ℕ) (total_games : ℕ) (h1 : n = 12) (h2 : total_games = 66) :
  ∃ x : ℕ, (n * (n - 1) * x) / 2 = total_games ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_games_times_each_team_plays_l4060_406076


namespace NUMINAMATH_CALUDE_partial_pressure_of_compound_l4060_406028

/-- Represents the partial pressure of a compound in a gas mixture. -/
def partial_pressure (mole_fraction : ℝ) (total_pressure : ℝ) : ℝ :=
  mole_fraction * total_pressure

/-- Theorem stating that the partial pressure of a compound in a gas mixture
    is 0.375 atm, given specific conditions. -/
theorem partial_pressure_of_compound (mole_fraction : ℝ) (total_pressure : ℝ) 
  (h1 : mole_fraction = 0.15)
  (h2 : total_pressure = 2.5) :
  partial_pressure mole_fraction total_pressure = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_partial_pressure_of_compound_l4060_406028


namespace NUMINAMATH_CALUDE_largest_divisible_n_l4060_406053

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 15) ∣ (n^3 + 150) ∧ ∀ (m : ℕ), m > n → ¬((m + 15) ∣ (m^3 + 150)) :=
by
  -- The largest such n is 2385
  use 2385
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l4060_406053


namespace NUMINAMATH_CALUDE_final_state_of_B_l4060_406050

/-- Represents a memory unit with a number of data pieces -/
structure MemoryUnit where
  data : ℕ

/-- Represents the state of all three memory units -/
structure MemoryState where
  A : MemoryUnit
  B : MemoryUnit
  C : MemoryUnit

/-- Performs the first operation: storing N data pieces in each unit -/
def firstOperation (N : ℕ) (state : MemoryState) : MemoryState :=
  { A := ⟨N⟩, B := ⟨N⟩, C := ⟨N⟩ }

/-- Performs the second operation: moving 2 data pieces from A to B -/
def secondOperation (state : MemoryState) : MemoryState :=
  { state with
    A := ⟨state.A.data - 2⟩
    B := ⟨state.B.data + 2⟩ }

/-- Performs the third operation: moving 2 data pieces from C to B -/
def thirdOperation (state : MemoryState) : MemoryState :=
  { state with
    B := ⟨state.B.data + 2⟩
    C := ⟨state.C.data - 2⟩ }

/-- Performs the fourth operation: moving N-2 data pieces from B to A -/
def fourthOperation (N : ℕ) (state : MemoryState) : MemoryState :=
  { state with
    A := ⟨state.A.data + (N - 2)⟩
    B := ⟨state.B.data - (N - 2)⟩ }

/-- The main theorem stating that after all operations, B has 6 data pieces -/
theorem final_state_of_B (N : ℕ) (h : N ≥ 3) :
  let initialState : MemoryState := ⟨⟨0⟩, ⟨0⟩, ⟨0⟩⟩
  let finalState := fourthOperation N (thirdOperation (secondOperation (firstOperation N initialState)))
  finalState.B.data = 6 := by sorry

end NUMINAMATH_CALUDE_final_state_of_B_l4060_406050


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4060_406075

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, (a < x ∧ x < a + 2) → x > 3) ∧
  (∃ x, x > 3 ∧ ¬(a < x ∧ x < a + 2)) →
  a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4060_406075


namespace NUMINAMATH_CALUDE_infinite_linear_combinations_l4060_406097

/-- An infinite sequence of strictly positive integers with a_k < a_{k+1} for all k -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that a_m can be written as x * a_p + y * a_q -/
def CanBeWrittenAs (a : ℕ → ℕ) (m p q x y : ℕ) : Prop :=
  a m = x * a p + y * a q ∧ 0 < x ∧ 0 < y ∧ p ≠ q

theorem infinite_linear_combinations (a : ℕ → ℕ) 
  (h : StrictlyIncreasingSequence a) :
  ∀ n : ℕ, ∃ m p q x y, m > n ∧ CanBeWrittenAs a m p q x y :=
sorry

end NUMINAMATH_CALUDE_infinite_linear_combinations_l4060_406097


namespace NUMINAMATH_CALUDE_probability_not_losing_l4060_406034

theorem probability_not_losing (p_draw p_win : ℝ) :
  p_draw = 1/2 →
  p_win = 1/3 →
  p_draw + p_win = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_losing_l4060_406034


namespace NUMINAMATH_CALUDE_morse_alphabet_size_l4060_406067

/-- The number of signals in each letter -/
def signal_length : Nat := 7

/-- The number of possible signals (dot and dash) -/
def signal_types : Nat := 2

/-- The number of possible alterations for each sequence (including the original) -/
def alterations_per_sequence : Nat := signal_length + 1

/-- The total number of possible sequences -/
def total_sequences : Nat := signal_types ^ signal_length

/-- The maximum number of unique letters in the alphabet -/
def max_letters : Nat := total_sequences / alterations_per_sequence

theorem morse_alphabet_size :
  max_letters = 16 := by sorry

end NUMINAMATH_CALUDE_morse_alphabet_size_l4060_406067


namespace NUMINAMATH_CALUDE_product_sum_l4060_406085

theorem product_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_l4060_406085


namespace NUMINAMATH_CALUDE_oil_leak_during_repairs_l4060_406060

theorem oil_leak_during_repairs 
  (total_leaked : ℕ) 
  (leaked_before_repairs : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before_repairs = 2475) :
  total_leaked - leaked_before_repairs = 3731 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_during_repairs_l4060_406060


namespace NUMINAMATH_CALUDE_unique_positive_solution_l4060_406062

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x ≤ 1 ∧ Real.cos (Real.arctan (Real.sin (Real.arccos x))) = x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l4060_406062


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l4060_406048

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l4060_406048


namespace NUMINAMATH_CALUDE_digit_A_value_l4060_406049

theorem digit_A_value : ∃ (A : ℕ), A < 10 ∧ 2 * 1000000 * A + 299561 = (3 * (523 + A))^2 → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_A_value_l4060_406049


namespace NUMINAMATH_CALUDE_remainder_of_four_to_eleven_mod_five_l4060_406041

theorem remainder_of_four_to_eleven_mod_five : 4^11 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_four_to_eleven_mod_five_l4060_406041


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l4060_406014

theorem discount_percentage_calculation 
  (num_people : ℕ) 
  (discount_per_person : ℝ) 
  (final_price : ℝ) : 
  num_people = 3 →
  discount_per_person = 4 →
  final_price = 48 →
  (((num_people : ℝ) * discount_per_person) / 
   (final_price + (num_people : ℝ) * discount_per_person)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l4060_406014


namespace NUMINAMATH_CALUDE_fence_cost_l4060_406002

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3944 := by sorry

end NUMINAMATH_CALUDE_fence_cost_l4060_406002


namespace NUMINAMATH_CALUDE_solve_euro_equation_l4060_406091

-- Define the € operation
def euro (x y : ℝ) := 3 * x * y

-- State the theorem
theorem solve_euro_equation (y : ℝ) (h1 : euro y (euro x 5) = 540) (h2 : y = 3) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_euro_equation_l4060_406091


namespace NUMINAMATH_CALUDE_defeat_dragon_l4060_406007

/-- Represents the three heroes --/
inductive Hero
| Ilya
| Dobrynya
| Alyosha

/-- Calculates the number of heads removed by a hero's strike --/
def headsRemoved (hero : Hero) (h : ℕ) : ℕ :=
  match hero with
  | Hero.Ilya => (h / 2) + 1
  | Hero.Dobrynya => (h / 3) + 2
  | Hero.Alyosha => (h / 4) + 3

/-- Represents a sequence of strikes by the heroes --/
def Strike := List Hero

/-- Applies a sequence of strikes to the initial number of heads --/
def applyStrikes (initialHeads : ℕ) (strikes : Strike) : ℕ :=
  strikes.foldl (fun remaining hero => remaining - (headsRemoved hero remaining)) initialHeads

/-- Theorem: For any initial number of heads, there exists a sequence of strikes that reduces it to zero --/
theorem defeat_dragon (initialHeads : ℕ) : ∃ (strikes : Strike), applyStrikes initialHeads strikes = 0 :=
sorry


end NUMINAMATH_CALUDE_defeat_dragon_l4060_406007


namespace NUMINAMATH_CALUDE_counterexample_exists_l4060_406017

theorem counterexample_exists : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^3 + b^3 < 2 * a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4060_406017


namespace NUMINAMATH_CALUDE_inequality_solution_l4060_406074

theorem inequality_solution (x : ℝ) : 
  let x₁ : ℝ := (-9 - Real.sqrt 21) / 2
  let x₂ : ℝ := (-9 + Real.sqrt 21) / 2
  (x - 1) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ 
    (x > -3 ∧ x < x₁) ∨ (x > x₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4060_406074


namespace NUMINAMATH_CALUDE_medicine_dose_per_kg_l4060_406008

theorem medicine_dose_per_kg (child_weight : ℝ) (dose_parts : ℕ) (dose_per_part : ℝ) :
  child_weight = 30 →
  dose_parts = 3 →
  dose_per_part = 50 →
  (dose_parts * dose_per_part) / child_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_medicine_dose_per_kg_l4060_406008


namespace NUMINAMATH_CALUDE_dan_picked_nine_limes_l4060_406098

/-- The number of limes Dan has now -/
def total_limes : ℕ := 13

/-- The number of limes Sara gave to Dan -/
def sara_limes : ℕ := 4

/-- The number of limes Dan picked -/
def dan_picked_limes : ℕ := total_limes - sara_limes

theorem dan_picked_nine_limes : dan_picked_limes = 9 := by sorry

end NUMINAMATH_CALUDE_dan_picked_nine_limes_l4060_406098


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l4060_406092

theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l4060_406092


namespace NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l4060_406016

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |x - b|

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f x (-2) 1 < 6 ↔ -1 < x ∧ x < 3 := by sorry

-- Part 2
theorem min_value_part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x : ℝ, f x a b = 1 ∧ ∀ y : ℝ, f y a b ≥ 1) →
  (∀ c d : ℝ, c > 0 → d > 0 → 2/c + 1/d ≥ 4) ∧
  (∃ e g : ℝ, e > 0 ∧ g > 0 ∧ 2/e + 1/g = 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_min_value_part2_l4060_406016


namespace NUMINAMATH_CALUDE_margo_walk_distance_l4060_406089

/-- Calculates the total distance walked given the time and speed for each direction -/
def totalDistanceWalked (timeToFriend timeFromFriend : ℚ) (speedToFriend speedFromFriend : ℚ) : ℚ :=
  timeToFriend * speedToFriend + timeFromFriend * speedFromFriend

theorem margo_walk_distance :
  let timeToFriend : ℚ := 15 / 60
  let timeFromFriend : ℚ := 25 / 60
  let speedToFriend : ℚ := 5
  let speedFromFriend : ℚ := 3
  totalDistanceWalked timeToFriend timeFromFriend speedToFriend speedFromFriend = 5 / 2 := by
  sorry

#eval totalDistanceWalked (15/60) (25/60) 5 3

end NUMINAMATH_CALUDE_margo_walk_distance_l4060_406089


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l4060_406059

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Define the open interval (1, 2)
def open_interval : Set ℝ := {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l4060_406059


namespace NUMINAMATH_CALUDE_rhombus_area_l4060_406033

/-- The area of a rhombus with side length 13 and diagonals differing by 10 units is 208 square units. -/
theorem rhombus_area (s d₁ d₂ : ℝ) (h₁ : s = 13) (h₂ : d₂ - d₁ = 10) 
    (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : d₁ * d₂ / 2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4060_406033


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l4060_406090

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l4060_406090


namespace NUMINAMATH_CALUDE_church_attendance_female_adults_l4060_406044

theorem church_attendance_female_adults
  (total : ℕ) (children : ℕ) (male_adults : ℕ)
  (h1 : total = 200)
  (h2 : children = 80)
  (h3 : male_adults = 60) :
  total - children - male_adults = 60 :=
by sorry

end NUMINAMATH_CALUDE_church_attendance_female_adults_l4060_406044


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l4060_406029

theorem angle_measure_in_triangle (D E F : ℝ) : 
  D = 75 →
  E = 4 * F - 15 →
  D + E + F = 180 →
  F = 24 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l4060_406029


namespace NUMINAMATH_CALUDE_equal_incircle_radii_of_original_triangles_l4060_406046

/-- A structure representing a triangle with an inscribed circle -/
structure TriangleWithIncircle where
  vertices : Fin 3 → ℝ × ℝ
  incircle_center : ℝ × ℝ
  incircle_radius : ℝ

/-- A structure representing the configuration of two intersecting triangles -/
structure IntersectingTriangles where
  triangle1 : TriangleWithIncircle
  triangle2 : TriangleWithIncircle
  hexagon_vertices : Fin 6 → ℝ × ℝ
  small_triangles : Fin 6 → TriangleWithIncircle

/-- The theorem statement -/
theorem equal_incircle_radii_of_original_triangles 
  (config : IntersectingTriangles)
  (h_equal_small_radii : ∀ i j : Fin 6, (config.small_triangles i).incircle_radius = (config.small_triangles j).incircle_radius) :
  config.triangle1.incircle_radius = config.triangle2.incircle_radius :=
sorry

end NUMINAMATH_CALUDE_equal_incircle_radii_of_original_triangles_l4060_406046


namespace NUMINAMATH_CALUDE_triangle_side_length_l4060_406040

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  a + b + c = 20 →  -- Perimeter condition
  (1/2) * a * b * Real.sin A = 10 →  -- Area condition
  A = π / 3 →  -- Angle A is 60°
  c = 7 :=  -- Length of side BC
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4060_406040


namespace NUMINAMATH_CALUDE_special_ten_digit_count_l4060_406013

/-- A natural number is special if all its digits are different or if changing one digit results in all digits being different. -/
def IsSpecial (n : ℕ) : Prop := sorry

/-- The count of 10-digit numbers. -/
def TenDigitCount : ℕ := 9000000000

/-- The count of special 10-digit numbers. -/
def SpecialTenDigitCount : ℕ := sorry

theorem special_ten_digit_count :
  SpecialTenDigitCount = 414 * Nat.factorial 9 := by sorry

end NUMINAMATH_CALUDE_special_ten_digit_count_l4060_406013


namespace NUMINAMATH_CALUDE_medal_distribution_theorem_l4060_406082

/-- The number of ways to distribute medals to students -/
def distribute_medals (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of students -/
def num_students : ℕ := 12

/-- The number of medal types -/
def num_medal_types : ℕ := 3

/-- The number of ways to distribute medals -/
def num_distributions : ℕ := distribute_medals (num_students - num_medal_types) num_medal_types

theorem medal_distribution_theorem : num_distributions = 55 := by
  sorry

end NUMINAMATH_CALUDE_medal_distribution_theorem_l4060_406082


namespace NUMINAMATH_CALUDE_horner_method_f_2_l4060_406003

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_method_f_2 :
  f 2 = horner_eval [1, 0, 2, 3, 1, 1] 2 ∧ horner_eval [1, 0, 2, 3, 1, 1] 2 = 41 := by
  sorry

#eval f 2
#eval horner_eval [1, 0, 2, 3, 1, 1] 2

end NUMINAMATH_CALUDE_horner_method_f_2_l4060_406003


namespace NUMINAMATH_CALUDE_correlation_function_is_even_l4060_406065

/-- Represents a stationary random process -/
class StationaryRandomProcess (X : ℝ → ℝ) : Prop where
  is_stationary : ∀ t₁ t₂ τ : ℝ, X (t₁ + τ) = X (t₂ + τ)

/-- Correlation function for a stationary random process -/
def correlationFunction (X : ℝ → ℝ) [StationaryRandomProcess X] (τ : ℝ) : ℝ :=
  sorry -- Definition of correlation function

/-- Theorem: The correlation function of a stationary random process is an even function -/
theorem correlation_function_is_even
  (X : ℝ → ℝ) [StationaryRandomProcess X] :
  ∀ τ : ℝ, correlationFunction X τ = correlationFunction X (-τ) := by
  sorry


end NUMINAMATH_CALUDE_correlation_function_is_even_l4060_406065


namespace NUMINAMATH_CALUDE_selection_ways_l4060_406009

def club_size : ℕ := 20
def co_presidents : ℕ := 2
def treasurers : ℕ := 1

theorem selection_ways : 
  (club_size.choose co_presidents * (club_size - co_presidents).choose treasurers) = 3420 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l4060_406009


namespace NUMINAMATH_CALUDE_grants_apartment_rooms_l4060_406086

-- Define the number of rooms in Danielle's apartment
def danielles_rooms : ℕ := 6

-- Define the number of rooms in Heidi's apartment
def heidis_rooms : ℕ := 3 * danielles_rooms

-- Define the number of rooms in Grant's apartment
def grants_rooms : ℕ := heidis_rooms / 9

-- Theorem stating that Grant's apartment has 2 rooms
theorem grants_apartment_rooms : grants_rooms = 2 := by
  sorry

end NUMINAMATH_CALUDE_grants_apartment_rooms_l4060_406086


namespace NUMINAMATH_CALUDE_clock_angle_at_3_45_l4060_406096

/-- The smaller angle between the hour hand and minute hand on a 12-hour analog clock at 3:45 --/
theorem clock_angle_at_3_45 :
  let full_rotation : ℝ := 360
  let hour_marks : ℕ := 12
  let degrees_per_hour : ℝ := full_rotation / hour_marks
  let minute_hand_angle : ℝ := 270
  let hour_hand_angle : ℝ := 3 * degrees_per_hour + 3/4 * degrees_per_hour
  let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
  min angle_diff (full_rotation - angle_diff) = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_45_l4060_406096


namespace NUMINAMATH_CALUDE_courtyard_width_l4060_406057

/-- Proves that the width of a courtyard is 25 feet given specific conditions --/
theorem courtyard_width : ∀ (width : ℝ),
  (width > 0) →  -- Ensure width is positive
  (4 * 10 * width * (0.4 * 3 + 0.6 * 1.5) = 2100) →
  width = 25 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_width_l4060_406057


namespace NUMINAMATH_CALUDE_university_applications_l4060_406027

theorem university_applications (n : ℕ) (s : Fin 5 → ℕ) : 
  (∀ i, s i ≥ n / 2) → 
  ∃ i j, i ≠ j ∧ (s i).min (s j) ≥ n / 5 := by
  sorry


end NUMINAMATH_CALUDE_university_applications_l4060_406027


namespace NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l4060_406066

theorem eight_fourth_equals_sixteen_n (n : ℕ) : 8^4 = 16^n → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_eight_fourth_equals_sixteen_n_l4060_406066


namespace NUMINAMATH_CALUDE_sandy_marks_loss_l4060_406039

theorem sandy_marks_loss (total_sums : ℕ) (correct_sums : ℕ) (total_marks : ℕ) 
  (marks_per_correct : ℕ) (h1 : total_sums = 30) (h2 : correct_sums = 25) 
  (h3 : total_marks = 65) (h4 : marks_per_correct = 3) : 
  (marks_per_correct * correct_sums - total_marks) / (total_sums - correct_sums) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_loss_l4060_406039


namespace NUMINAMATH_CALUDE_constant_functions_from_functional_equation_l4060_406080

theorem constant_functions_from_functional_equation 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → f (x^2 + y^2) = g (x * y)) :
  ∃ (c : ℝ), (∀ (x : ℝ), x > 0 → f x = c) ∧ (∀ (x : ℝ), x > 0 → g x = c) :=
sorry

end NUMINAMATH_CALUDE_constant_functions_from_functional_equation_l4060_406080


namespace NUMINAMATH_CALUDE_crease_lines_set_l4060_406081

/-- Given a circle with center O, radius R, and a point A inside the circle such that OA = a,
    the set of points P that satisfy |PO| + |PA| ≥ R is equivalent to the set
    {(x, y): ((x - a/2)^2) / ((R/2)^2) + (y^2) / ((R/2)^2 - (a/2)^2) ≥ 1} -/
theorem crease_lines_set (O A : ℝ × ℝ) (R a : ℝ) (h1 : R > 0) (h2 : 0 < a ∧ a < R) :
  let d (P : ℝ × ℝ) (Q : ℝ × ℝ) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  {P : ℝ × ℝ | d P O + d P A ≥ R} =
  {P : ℝ × ℝ | ((P.1 - a/2)^2) / ((R/2)^2) + (P.2^2) / ((R/2)^2 - (a/2)^2) ≥ 1} :=
by sorry

end NUMINAMATH_CALUDE_crease_lines_set_l4060_406081


namespace NUMINAMATH_CALUDE_total_population_l4060_406026

def population_problem (springfield_population greenville_population : ℕ) : Prop :=
  springfield_population = 482653 ∧
  greenville_population = springfield_population - 119666 ∧
  springfield_population + greenville_population = 845640

theorem total_population :
  ∃ (springfield_population greenville_population : ℕ),
    population_problem springfield_population greenville_population :=
by
  sorry

end NUMINAMATH_CALUDE_total_population_l4060_406026


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4060_406055

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (7 + 3 * z) = 15 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4060_406055


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l4060_406077

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℤ)
  (second : ℤ)

/-- Generates the nth pair in the sequence -/
def generateNthPair (n : ℕ) : IntPair :=
  sorry

/-- The sequence of integer pairs as described in the problem -/
def sequencePairs : ℕ → IntPair :=
  generateNthPair

theorem sixtieth_pair_is_five_seven :
  sequencePairs 60 = IntPair.mk 5 7 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_seven_l4060_406077


namespace NUMINAMATH_CALUDE_function_inequality_and_range_l4060_406020

/-- Given functions f and g, prove that if |f(x)| ≤ |g(x)| for all x, then f = g/2 - 4.
    Also prove that if f(x) ≥ (m + 2)x - m - 15 for all x > 2, then m ≤ 2. -/
theorem function_inequality_and_range (a b m : ℝ) : 
  let f := fun (x : ℝ) => x^2 + a*x + b
  let g := fun (x : ℝ) => 2*x^2 - 4*x - 16
  (∀ x, |f x| ≤ |g x|) →
  (a = -2 ∧ b = -8) ∧
  ((∀ x > 2, f x ≥ (m + 2)*x - m - 15) → m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_and_range_l4060_406020


namespace NUMINAMATH_CALUDE_smallest_two_base_representation_l4060_406061

/-- Represents a number in a given base with two identical digits --/
def twoDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a number is valid in a given base --/
def isValidInBase (n : Nat) (base : Nat) : Prop :=
  n < base

theorem smallest_two_base_representation : 
  ∀ n : Nat, n < 24 → 
  ¬(∃ (a b : Nat), 
    isValidInBase a 5 ∧ 
    isValidInBase b 7 ∧ 
    n = twoDigitNumber a 5 ∧ 
    n = twoDigitNumber b 7) ∧
  (∃ (a b : Nat),
    isValidInBase a 5 ∧
    isValidInBase b 7 ∧
    24 = twoDigitNumber a 5 ∧
    24 = twoDigitNumber b 7) :=
by sorry

#check smallest_two_base_representation

end NUMINAMATH_CALUDE_smallest_two_base_representation_l4060_406061


namespace NUMINAMATH_CALUDE_smallest_even_with_repeated_seven_l4060_406093

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_repeated_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  ∃ k : ℕ, k > 1 ∧ p ^ k ∣ n

theorem smallest_even_with_repeated_seven :
  ∀ n : ℕ, 
    is_even n ∧ 
    has_repeated_prime_factor n 7 → 
    n ≥ 98 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_even_with_repeated_seven_l4060_406093


namespace NUMINAMATH_CALUDE_nonzero_terms_count_l4060_406099

/-- The number of nonzero terms in the expansion of (x^2+2)(3x^3+2x^2+4)-4(x^4+x^3-3x) -/
theorem nonzero_terms_count : ∃ (p : Polynomial ℝ), 
  p = (X^2 + 2) * (3*X^3 + 2*X^2 + 4) - 4*(X^4 + X^3 - 3*X) ∧ 
  p.support.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_terms_count_l4060_406099


namespace NUMINAMATH_CALUDE_tangent_slope_at_4_l4060_406069

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5*x - 8

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x + 5

-- Theorem statement
theorem tangent_slope_at_4 : f_derivative 4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_4_l4060_406069


namespace NUMINAMATH_CALUDE_max_value_constraint_l4060_406036

theorem max_value_constraint (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (z : ℝ), 4 * x + y ≤ z → z ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_constraint_l4060_406036


namespace NUMINAMATH_CALUDE_remainder_theorem_l4060_406006

theorem remainder_theorem (x : ℝ) : 
  ∃ (P : ℝ → ℝ) (S : ℝ → ℝ), 
    (∀ x, x^105 = (x^2 - 4*x + 3) * P x + S x) ∧ 
    (∀ x, S x = (3^105 * (x - 1) - (x - 2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4060_406006


namespace NUMINAMATH_CALUDE_recurrence_is_geometric_iff_first_two_equal_l4060_406094

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℝ) : Prop :=
  (∀ n, b n > 0) ∧ (∀ n, b (n + 2) = 3 * b n * b (n + 1))

/-- A geometric progression -/
def IsGeometricProgression (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n, b (n + 1) = r * b n

/-- The main theorem -/
theorem recurrence_is_geometric_iff_first_two_equal
    (b : ℕ → ℝ) (h : RecurrenceSequence b) :
    IsGeometricProgression b ↔ b 1 = b 2 := by
  sorry

end NUMINAMATH_CALUDE_recurrence_is_geometric_iff_first_two_equal_l4060_406094


namespace NUMINAMATH_CALUDE_fourth_circle_radius_is_p_l4060_406038

-- Define the right triangle
structure RightTriangle :=
  (a b c : ℝ)
  (right_angle : a^2 + b^2 = c^2)
  (perimeter : a + b + c = 2 * p)

-- Define the circles
structure Circles (t : RightTriangle) :=
  (r1 r2 r3 : ℝ)
  (externally_tangent : t.a = r2 + r3 ∧ t.b = r1 + r3 ∧ t.c = r1 + r2)
  (fourth_circle_radius : ℝ)
  (internally_tangent : 
    t.a = fourth_circle_radius - r3 + (fourth_circle_radius - r2) ∧
    t.b = fourth_circle_radius - r1 + (fourth_circle_radius - r3) ∧
    t.c = fourth_circle_radius - r1 + (fourth_circle_radius - r2))

-- The theorem to prove
theorem fourth_circle_radius_is_p (t : RightTriangle) (c : Circles t) : 
  c.fourth_circle_radius = p :=
sorry

end NUMINAMATH_CALUDE_fourth_circle_radius_is_p_l4060_406038


namespace NUMINAMATH_CALUDE_four_distinct_roots_l4060_406047

theorem four_distinct_roots (m : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ (x : ℝ), x^2 - 4*|x| + 5 - m = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)))
  ↔ (1 < m ∧ m < 5) :=
sorry

end NUMINAMATH_CALUDE_four_distinct_roots_l4060_406047


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l4060_406004

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l4060_406004


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4060_406064

theorem polynomial_factorization (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = (x - 3) * (x - 2)) →
  (a = 1 ∧ b = -5 ∧ c = 6) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4060_406064


namespace NUMINAMATH_CALUDE_billy_crayons_l4060_406001

theorem billy_crayons (initial_crayons eaten_crayons remaining_crayons : ℕ) :
  eaten_crayons = 52 →
  remaining_crayons = 10 →
  initial_crayons = eaten_crayons + remaining_crayons →
  initial_crayons = 62 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l4060_406001


namespace NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l4060_406022

/-- Represents a 3x3 tic-tac-toe board --/
def TicTacToeBoard := Fin 3 → Fin 3 → Bool

/-- The total number of cells on the board --/
def totalCells : Nat := 9

/-- The number of noughts on the board --/
def numNoughts : Nat := 3

/-- The number of crosses on the board --/
def numCrosses : Nat := 6

/-- The number of ways to arrange noughts on the board --/
def totalArrangements : Nat := Nat.choose totalCells numNoughts

/-- The number of winning positions for noughts --/
def winningPositions : Nat := 8

/-- The probability of noughts being in a winning position --/
def winningProbability : ℚ := winningPositions / totalArrangements

theorem tic_tac_toe_winning_probability :
  winningProbability = 2 / 21 := by sorry

end NUMINAMATH_CALUDE_tic_tac_toe_winning_probability_l4060_406022


namespace NUMINAMATH_CALUDE_sin_210_degrees_l4060_406095

theorem sin_210_degrees : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l4060_406095


namespace NUMINAMATH_CALUDE_sequence_properties_l4060_406056

def sequence_a (n : ℕ) : ℚ := 2 * n + 1

def S (n : ℕ) : ℚ := n * sequence_a n - n * (n - 1)

def sequence_b (n : ℕ) : ℚ := 1 / (sequence_a n * sequence_a (n + 1))

def T (n : ℕ) : ℚ := n / (6 * n + 9)

theorem sequence_properties :
  (sequence_a 1 = 3) ∧
  (∀ n : ℕ, S n = n * sequence_a n - n * (n - 1)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = 2 * n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → (Finset.sum (Finset.range n) (λ i => sequence_b (i + 1))) = T n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l4060_406056


namespace NUMINAMATH_CALUDE_min_distinct_values_l4060_406019

/-- Given a list of 2023 positive integers with a unique mode occurring 15 times,
    the minimum number of distinct values is 146 -/
theorem min_distinct_values (l : List ℕ+) : 
  l.length = 2023 →
  ∃! m : ℕ+, (l.count m = 15 ∧ ∀ n : ℕ+, l.count n ≤ 15) →
  (∀ k : ℕ+, l.count k = 15 → k = m) →
  (Finset.card l.toFinset : ℕ) ≥ 146 ∧ 
  ∃ l' : List ℕ+, l'.length = 2023 ∧ 
    (∃! m' : ℕ+, (l'.count m' = 15 ∧ ∀ n : ℕ+, l'.count n ≤ 15)) ∧
    (Finset.card l'.toFinset : ℕ) = 146 :=
by
  sorry

end NUMINAMATH_CALUDE_min_distinct_values_l4060_406019


namespace NUMINAMATH_CALUDE_apple_banana_equivalence_l4060_406083

theorem apple_banana_equivalence (apple_value banana_value : ℚ) : 
  (3 / 4 * 12 : ℚ) * apple_value = 10 * banana_value →
  (2 / 3 * 9 : ℚ) * apple_value = (20 / 3 : ℚ) * banana_value := by
  sorry

end NUMINAMATH_CALUDE_apple_banana_equivalence_l4060_406083


namespace NUMINAMATH_CALUDE_age_difference_l4060_406030

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : a = c + 10 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l4060_406030


namespace NUMINAMATH_CALUDE_max_value_quadratic_l4060_406084

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (z : ℝ), z = x^2 + 3*x*y + 2*y^2 ∧ z ≤ 120 - 30*Real.sqrt 3 ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x'^2 - 2*x'*y' + 3*y'^2 = 10 ∧
  x'^2 + 3*x'*y' + 2*y'^2 = 120 - 30*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l4060_406084


namespace NUMINAMATH_CALUDE_expression_one_equality_l4060_406054

theorem expression_one_equality : 
  4 * Real.sqrt 54 * 3 * Real.sqrt 2 / (-(3/2) * Real.sqrt (1/3)) = -144 := by
sorry

end NUMINAMATH_CALUDE_expression_one_equality_l4060_406054


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l4060_406012

theorem line_passes_through_quadrants (a b c p : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a + b) / c = p) (h5 : (b + c) / a = p) (h6 : (c + a) / b = p) :
  ∃ (x y : ℝ), x < 0 ∧ y = p * x + p ∧ y < 0 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l4060_406012


namespace NUMINAMATH_CALUDE_min_squares_exceeding_300_l4060_406079

/-- The function that represents repeated squaring of a number -/
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

/-- The theorem stating that 3 is the smallest positive integer n for which
    repeated squaring of 3, n times, exceeds 300 -/
theorem min_squares_exceeding_300 :
  ∀ n : ℕ, n > 0 → (
    (repeated_square 3 n > 300 ∧ ∀ m : ℕ, m > 0 → m < n → repeated_square 3 m ≤ 300) ↔ n = 3
  ) :=
sorry

end NUMINAMATH_CALUDE_min_squares_exceeding_300_l4060_406079


namespace NUMINAMATH_CALUDE_third_segment_length_l4060_406042

/-- Represents the lengths of interview segments in a radio show. -/
structure InterviewSegments where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Checks if the given segment lengths satisfy the radio show conditions. -/
def validSegments (s : InterviewSegments) : Prop :=
  s.first = 2 * (s.second + s.third) ∧
  s.third = s.second / 2 ∧
  s.first + s.second + s.third = 90

theorem third_segment_length :
  ∀ s : InterviewSegments, validSegments s → s.third = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_segment_length_l4060_406042


namespace NUMINAMATH_CALUDE_triangle_area_after_transformation_l4060_406000

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 5]
def T : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

theorem triangle_area_after_transformation :
  let Ta := T.mulVec a
  let Tb := T.mulVec b
  (1/2) * abs (Ta 0 * Tb 1 - Ta 1 * Tb 0) = 8.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_after_transformation_l4060_406000


namespace NUMINAMATH_CALUDE_opposite_sides_of_line_l4060_406010

theorem opposite_sides_of_line (m : ℝ) : 
  (∀ (x y : ℝ), 2*x + y - 2 = 0 → (2*(-2) + m - 2) * (2*m + 4 - 2) < 0) → 
  -1 < m ∧ m < 6 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_of_line_l4060_406010


namespace NUMINAMATH_CALUDE_probability_product_216_l4060_406087

def standard_die := Finset.range 6

def roll_product (x y z : ℕ) : ℕ := x * y * z

theorem probability_product_216 :
  (Finset.filter (λ (t : ℕ × ℕ × ℕ) => roll_product t.1 t.2.1 t.2.2 = 216) 
    (standard_die.product (standard_die.product standard_die))).card / 
  (standard_die.card ^ 3 : ℚ) = 1 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_product_216_l4060_406087


namespace NUMINAMATH_CALUDE_base6_45_equals_29_l4060_406088

/-- Converts a base-6 number to decimal --/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The decimal representation of 45 in base 6 --/
def base6_45 : Nat := base6ToDecimal [5, 4]

theorem base6_45_equals_29 : base6_45 = 29 := by sorry

end NUMINAMATH_CALUDE_base6_45_equals_29_l4060_406088


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l4060_406032

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 42 + 2 * Real.sqrt 261 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l4060_406032


namespace NUMINAMATH_CALUDE_valid_parameterization_l4060_406035

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line parameterization -/
structure LineParam where
  point : Vector2D
  direction : Vector2D

/-- Checks if a point lies on the line y = -2x + 7 -/
def liesOnLine (v : Vector2D) : Prop :=
  v.y = -2 * v.x + 7

/-- Checks if a vector is a scalar multiple of (1, -2) -/
def isValidDirection (v : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * 1 ∧ v.y = k * (-2)

/-- Main theorem: A parameterization is valid iff it satisfies both conditions -/
theorem valid_parameterization (p : LineParam) :
  (liesOnLine p.point ∧ isValidDirection p.direction) ↔
  (∀ (t : ℝ), liesOnLine ⟨p.point.x + t * p.direction.x, p.point.y + t * p.direction.y⟩) :=
by sorry

end NUMINAMATH_CALUDE_valid_parameterization_l4060_406035


namespace NUMINAMATH_CALUDE_batsman_average_l4060_406078

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  (previous_total = 16 * previous_average) →
  (previous_total + 85 = 17 * (previous_average + 3)) →
  (previous_average + 3 = 37) := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l4060_406078


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l4060_406068

def jeff_scores : List ℚ := [86, 94, 87, 96, 92, 89]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℚ) = 544 / 6 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l4060_406068


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l4060_406073

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 6 = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 + m*y + 6 = 0 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l4060_406073


namespace NUMINAMATH_CALUDE_coat_price_reduction_percentage_l4060_406058

/-- The percentage reduction when a coat's price is reduced from $500 to $150 is 70% -/
theorem coat_price_reduction_percentage : 
  let original_price : ℚ := 500
  let reduced_price : ℚ := 150
  let reduction : ℚ := original_price - reduced_price
  let percentage_reduction : ℚ := (reduction / original_price) * 100
  percentage_reduction = 70 := by sorry

end NUMINAMATH_CALUDE_coat_price_reduction_percentage_l4060_406058


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l4060_406070

theorem largest_coefficient_binomial_expansion :
  ∃ (k : ℕ) (c : ℚ), 
    (k = 3 ∧ c = 160) ∧
    ∀ (j : ℕ) (d : ℚ), 
      (Nat.choose 6 j * (2 ^ j)) ≤ (Nat.choose 6 k * (2 ^ k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l4060_406070


namespace NUMINAMATH_CALUDE_find_a_l4060_406063

def U (a : ℝ) : Set ℝ := {3, a, a^2 + 2*a - 3}
def A : Set ℝ := {2, 3}

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, x ∈ U a → x ∈ A ∨ x = 5) ∧
  (∀ x : ℝ, x ∈ A → x ∈ U a) ∧
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_find_a_l4060_406063


namespace NUMINAMATH_CALUDE_root_product_theorem_l4060_406018

theorem root_product_theorem (m p : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  (r = 16/3) := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l4060_406018


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l4060_406023

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x - 3) * (x + 1) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 12)) →
  p = -8 ∧ q = -10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l4060_406023


namespace NUMINAMATH_CALUDE_sixty_one_invalid_l4060_406025

/-- Represents the seat numbers of selected students -/
def selected_seats : List Nat := [5, 16, 27, 38, 49]

/-- The number of selected students -/
def num_selected : Nat := 5

/-- Checks if the given number can be the total number of students in the class -/
def is_valid_class_size (x : Nat) : Prop :=
  ∃ k, x = k * (num_selected - 1) + selected_seats.head!

/-- Theorem stating that 61 cannot be the number of students in the class -/
theorem sixty_one_invalid : ¬ is_valid_class_size 61 := by
  sorry


end NUMINAMATH_CALUDE_sixty_one_invalid_l4060_406025


namespace NUMINAMATH_CALUDE_equation_solution_l4060_406015

theorem equation_solution (x : ℝ) (h : 1/x + 1/(2*x) + 1/(3*x) = 1/12) : x = 22 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4060_406015


namespace NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l4060_406051

theorem at_least_one_less_than_or_equal_to_one
  (x y z : ℝ)
  (positive_x : 0 < x)
  (positive_y : 0 < y)
  (positive_z : 0 < z)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l4060_406051


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4060_406011

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 7}

-- Define the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4060_406011


namespace NUMINAMATH_CALUDE_cd_length_sum_l4060_406052

theorem cd_length_sum : 
  let num_cds : ℕ := 3
  let regular_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * regular_cd_length
  let total_length : ℝ := 2 * regular_cd_length + long_cd_length
  total_length = 6 := by sorry

end NUMINAMATH_CALUDE_cd_length_sum_l4060_406052


namespace NUMINAMATH_CALUDE_largest_prime_divisor_for_primality_test_l4060_406045

theorem largest_prime_divisor_for_primality_test :
  ∀ n : ℕ, 950 ≤ n → n ≤ 1000 →
  (∀ p : ℕ, p ≤ 31 → Nat.Prime p → ¬(p ∣ n)) →
  Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_for_primality_test_l4060_406045


namespace NUMINAMATH_CALUDE_inner_segments_sum_l4060_406024

theorem inner_segments_sum (perimeter_quadrilaterals perimeter_triangles perimeter_ABC : ℝ) 
  (h1 : perimeter_quadrilaterals = 25)
  (h2 : perimeter_triangles = 20)
  (h3 : perimeter_ABC = 19) :
  let total_perimeter := perimeter_quadrilaterals + perimeter_triangles
  let inner_segments := total_perimeter - perimeter_ABC
  inner_segments / 2 = 13 := by sorry

end NUMINAMATH_CALUDE_inner_segments_sum_l4060_406024


namespace NUMINAMATH_CALUDE_best_play_wins_probability_best_play_always_wins_more_than_two_plays_l4060_406031

/-- The probability that the best play wins in a competition with two plays -/
def probability_best_play_wins (n : ℕ) : ℚ :=
  1 - (n.factorial * n.factorial : ℚ) / ((2 * n).factorial : ℚ)

/-- The setup of the competition -/
structure Competition :=
  (n : ℕ)  -- number of students in each play
  (honest_mothers : ℕ)  -- number of mothers voting honestly
  (biased_mothers : ℕ)  -- number of mothers voting for their child's play

/-- The conditions of the competition -/
def competition_conditions (c : Competition) : Prop :=
  c.honest_mothers = c.n ∧ c.biased_mothers = c.n

/-- The theorem stating the probability of the best play winning -/
theorem best_play_wins_probability (c : Competition) 
  (h : competition_conditions c) : 
  probability_best_play_wins c.n = 1 - (c.n.factorial * c.n.factorial : ℚ) / ((2 * c.n).factorial : ℚ) :=
sorry

/-- For more than two plays, the best play always wins -/
theorem best_play_always_wins_more_than_two_plays (c : Competition) (s : ℕ) 
  (h1 : competition_conditions c) (h2 : s > 2) : 
  probability_best_play_wins c.n = 1 :=
sorry

end NUMINAMATH_CALUDE_best_play_wins_probability_best_play_always_wins_more_than_two_plays_l4060_406031
