import Mathlib

namespace NUMINAMATH_CALUDE_point_on_x_axis_l3803_380375

theorem point_on_x_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 3, m + 1)
  (P.2 = 0) → P = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3803_380375


namespace NUMINAMATH_CALUDE_odd_scripts_in_final_state_l3803_380382

/-- Represents the state of the box of scripts -/
structure ScriptBox where
  total : Nat
  odd : Nat
  even : Nat

/-- The procedure of selecting and manipulating scripts -/
def select_and_manipulate (box : ScriptBox) : ScriptBox :=
  sorry

/-- Represents the final state of the box -/
def final_state (initial : ScriptBox) : ScriptBox :=
  sorry

theorem odd_scripts_in_final_state :
  ∀ (initial : ScriptBox),
    initial.total = 4032 →
    initial.odd = initial.total / 2 →
    initial.even = initial.total / 2 →
    let final := final_state initial
    final.total = 3 →
    final.odd > 0 →
    final.even > 0 →
    final.odd = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_scripts_in_final_state_l3803_380382


namespace NUMINAMATH_CALUDE_uniform_color_probability_l3803_380380

theorem uniform_color_probability : 
  let sock_colors : Nat := 3
  let short_colors : Nat := 4
  let total_combinations : Nat := sock_colors * short_colors
  let matching_combinations : Nat := min sock_colors short_colors
  (total_combinations - matching_combinations : ℚ) / total_combinations = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_uniform_color_probability_l3803_380380


namespace NUMINAMATH_CALUDE_total_grapes_is_83_l3803_380306

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_grapes_is_83_l3803_380306


namespace NUMINAMATH_CALUDE_fish_value_in_rice_l3803_380347

/-- Represents the value of items in a barter system -/
structure BarterValue where
  fish : ℚ
  bread : ℚ
  rice : ℚ

/-- The barter system with given exchange rates -/
def barterSystem : BarterValue where
  fish := 1
  bread := 3/5
  rice := 1/10

theorem fish_value_in_rice (b : BarterValue) 
  (h1 : 5 * b.fish = 3 * b.bread) 
  (h2 : b.bread = 6 * b.rice) : 
  b.fish = 18/5 * b.rice := by
  sorry

#check fish_value_in_rice barterSystem

end NUMINAMATH_CALUDE_fish_value_in_rice_l3803_380347


namespace NUMINAMATH_CALUDE_units_digit_of_nine_to_eight_to_seven_l3803_380356

theorem units_digit_of_nine_to_eight_to_seven (n : Nat) : n = 9^(8^7) → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_nine_to_eight_to_seven_l3803_380356


namespace NUMINAMATH_CALUDE_ted_banana_purchase_l3803_380377

/-- The number of oranges Ted needs to purchase -/
def num_oranges : ℕ := 10

/-- The cost of one banana in dollars -/
def banana_cost : ℚ := 2

/-- The cost of one orange in dollars -/
def orange_cost : ℚ := 3/2

/-- The total cost of the fruits in dollars -/
def total_cost : ℚ := 25

/-- The number of bananas Ted needs to purchase -/
def num_bananas : ℕ := 5

theorem ted_banana_purchase :
  num_bananas * banana_cost + num_oranges * orange_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_ted_banana_purchase_l3803_380377


namespace NUMINAMATH_CALUDE_P_inter_Q_equiv_l3803_380325

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem P_inter_Q_equiv : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_P_inter_Q_equiv_l3803_380325


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3803_380383

theorem sum_remainder_mod_seven
  (a b c : ℕ)
  (ha : 0 < a ∧ a < 7)
  (hb : 0 < b ∧ b < 7)
  (hc : 0 < c ∧ c < 7)
  (h1 : a * b * c % 7 = 1)
  (h2 : 4 * c % 7 = 3)
  (h3 : 5 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3803_380383


namespace NUMINAMATH_CALUDE_double_iced_cubes_count_l3803_380374

/-- Represents a 3D coordinate in the cake --/
structure Coordinate where
  x : Nat
  y : Nat
  z : Nat

/-- The size of the cake --/
def cakeSize : Nat := 5

/-- Checks if a coordinate is on an edge with exactly two iced sides --/
def isDoubleIcedEdge (c : Coordinate) : Bool :=
  -- Top edge (front)
  (c.z = cakeSize - 1 && c.y = 0 && c.x > 0 && c.x < cakeSize - 1) ||
  -- Top edge (left)
  (c.z = cakeSize - 1 && c.x = 0 && c.y > 0 && c.y < cakeSize - 1) ||
  -- Front-left edge
  (c.x = 0 && c.y = 0 && c.z > 0 && c.z < cakeSize - 1)

/-- Counts the number of cubes with icing on exactly two sides --/
def countDoubleIcedCubes : Nat :=
  let coords := List.range cakeSize >>= fun x =>
                List.range cakeSize >>= fun y =>
                List.range cakeSize >>= fun z =>
                [{x := x, y := y, z := z}]
  (coords.filter isDoubleIcedEdge).length

/-- The main theorem to prove --/
theorem double_iced_cubes_count :
  countDoubleIcedCubes = 31 := by
  sorry


end NUMINAMATH_CALUDE_double_iced_cubes_count_l3803_380374


namespace NUMINAMATH_CALUDE_alloy_density_proof_l3803_380386

/-- The specific gravity of gold relative to water -/
def gold_specific_gravity : ℝ := 19

/-- The specific gravity of copper relative to water -/
def copper_specific_gravity : ℝ := 9

/-- The ratio of gold to copper in the alloy -/
def gold_copper_ratio : ℝ := 4

/-- The specific gravity of the resulting alloy -/
def alloy_specific_gravity : ℝ := 17

/-- Theorem stating that mixing gold and copper in the given ratio results in the specified alloy density -/
theorem alloy_density_proof :
  (gold_copper_ratio * gold_specific_gravity + copper_specific_gravity) / (gold_copper_ratio + 1) = alloy_specific_gravity :=
by sorry

end NUMINAMATH_CALUDE_alloy_density_proof_l3803_380386


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3803_380372

/-- If 9x^2 + mxy + 16y^2 is a perfect square trinomial, then m = ±24 -/
theorem perfect_square_trinomial_condition (m : ℝ) :
  (∃ (a b : ℝ), ∀ (x y : ℝ), 9*x^2 + m*x*y + 16*y^2 = (a*x + b*y)^2) →
  (m = 24 ∨ m = -24) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3803_380372


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l3803_380346

-- Define set A
def A : Set ℝ := {x | |x - 1| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1)}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l3803_380346


namespace NUMINAMATH_CALUDE_ryan_sandwiches_l3803_380327

/-- The number of sandwiches Ryan can make given the total number of bread slices and slices per sandwich -/
def number_of_sandwiches (total_slices : ℕ) (slices_per_sandwich : ℕ) : ℕ :=
  total_slices / slices_per_sandwich

/-- Theorem: Ryan can make 5 sandwiches with 15 total slices and 3 slices per sandwich -/
theorem ryan_sandwiches :
  number_of_sandwiches 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ryan_sandwiches_l3803_380327


namespace NUMINAMATH_CALUDE_students_per_school_is_247_l3803_380370

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := 6175

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := total_students / num_schools

/-- Theorem stating that the number of students in each elementary school is 247 -/
theorem students_per_school_is_247 : students_per_school = 247 := by sorry

end NUMINAMATH_CALUDE_students_per_school_is_247_l3803_380370


namespace NUMINAMATH_CALUDE_machine_work_time_l3803_380395

/-- The number of shirts made by the machine today -/
def shirts_today : ℕ := 8

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 2

/-- The number of minutes the machine worked today -/
def minutes_worked : ℕ := shirts_today / shirts_per_minute

theorem machine_work_time : minutes_worked = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l3803_380395


namespace NUMINAMATH_CALUDE_pyramid_volume_l3803_380371

-- Define the rectangular parallelepiped
structure Parallelepiped where
  AB : ℝ
  BC : ℝ
  CG : ℝ

-- Define the rectangular pyramid
structure Pyramid where
  base : ℝ -- Area of the base BDFE
  height : ℝ -- Height of the pyramid (XM)

-- Define the problem
theorem pyramid_volume (p : Parallelepiped) (pyr : Pyramid) : 
  p.AB = 4 → 
  p.BC = 2 → 
  p.CG = 5 → 
  pyr.base = p.AB * p.BC → 
  pyr.height = p.CG → 
  (1/3 : ℝ) * pyr.base * pyr.height = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3803_380371


namespace NUMINAMATH_CALUDE_f_monotonicity_and_negativity_l3803_380334

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem f_monotonicity_and_negativity (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) ∨
  (∃ c, c > 0 ∧ 
    (∀ x y, 0 < x ∧ x < y ∧ y < c → f a x < f a y) ∧
    (∀ x y, c < x ∧ x < y → f a y < f a x)) ∧
  (∀ x, x > 0 → f a x < 0) ↔ a > (Real.exp 1)⁻¹ :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_negativity_l3803_380334


namespace NUMINAMATH_CALUDE_positive_number_equation_l3803_380384

theorem positive_number_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 3*a) : a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equation_l3803_380384


namespace NUMINAMATH_CALUDE_deduced_conclusions_l3803_380354

-- Define our sets
variable (U : Type) -- Universe set
variable (Cem Ben Den : Set U)

-- Define our hypotheses
variable (h1 : Cem ∩ Ben = ∅)
variable (h2 : Den ⊆ Ben)
variable (h3 : ∃ x, x ∈ Cem ∧ x ∉ Den)

-- Define our conclusions
def conclusion_B : Prop := ∃ x, x ∈ Den ∧ x ∉ Cem
def conclusion_C : Prop := Den ∩ Cem = ∅

-- Theorem statement
theorem deduced_conclusions : conclusion_B U Cem Den ∧ conclusion_C U Cem Den :=
sorry

end NUMINAMATH_CALUDE_deduced_conclusions_l3803_380354


namespace NUMINAMATH_CALUDE_rectangle_area_minus_hole_l3803_380392

def large_rect_length (x : ℝ) : ℝ := x^2 + 7
def large_rect_width (x : ℝ) : ℝ := x^2 + 5
def hole_rect_length (x : ℝ) : ℝ := 2*x^2 - 3
def hole_rect_width (x : ℝ) : ℝ := x^2 - 2

theorem rectangle_area_minus_hole (x : ℝ) :
  large_rect_length x * large_rect_width x - hole_rect_length x * hole_rect_width x
  = -x^4 + 19*x^2 + 29 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_minus_hole_l3803_380392


namespace NUMINAMATH_CALUDE_dollar_three_neg_two_l3803_380373

-- Define the operation $
def dollar (a b : ℤ) : ℤ := a * (b - 1) + a * b

-- Theorem statement
theorem dollar_three_neg_two : dollar 3 (-2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_neg_two_l3803_380373


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3803_380365

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂ ∧ a₁ / b₁ ≠ c₁ / c₂

/-- The value of m for which the lines x + my + 6 = 0 and (m-2)x + 3y + 2m = 0 are parallel -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel 1 m 6 (m-2) 3 (2*m) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3803_380365


namespace NUMINAMATH_CALUDE_square_sum_factorial_solutions_l3803_380393

theorem square_sum_factorial_solutions :
  ∀ (a b n : ℕ+),
    n < 14 →
    a ≤ b →
    a ^ 2 + b ^ 2 = n! →
    ((n = 2 ∧ a = 1 ∧ b = 1) ∨ (n = 6 ∧ a = 12 ∧ b = 24)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_factorial_solutions_l3803_380393


namespace NUMINAMATH_CALUDE_wire_length_proof_l3803_380369

theorem wire_length_proof (part1 part2 total : ℕ) : 
  part1 = 106 →
  part2 = 74 →
  part1 = part2 + 32 →
  total = part1 + part2 →
  total = 180 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l3803_380369


namespace NUMINAMATH_CALUDE_crackers_distribution_l3803_380358

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) : 
  total_crackers = 36 →
  num_friends = 18 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 2 := by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l3803_380358


namespace NUMINAMATH_CALUDE_salary_increase_l3803_380337

/-- Given a salary increase of 100% resulting in a new salary of $80,
    prove that the original salary was $40. -/
theorem salary_increase (new_salary : ℝ) (increase_percentage : ℝ) : 
  new_salary = 80 ∧ increase_percentage = 100 → 
  new_salary / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3803_380337


namespace NUMINAMATH_CALUDE_legacy_cleaning_time_l3803_380385

/-- The number of floors in the building -/
def num_floors : ℕ := 4

/-- The number of rooms per floor -/
def rooms_per_floor : ℕ := 10

/-- Legacy's hourly rate in dollars -/
def hourly_rate : ℕ := 15

/-- Total earnings from cleaning all floors in dollars -/
def total_earnings : ℕ := 3600

/-- Time to clean one room in hours -/
def time_per_room : ℚ := 6

theorem legacy_cleaning_time :
  time_per_room = (total_earnings : ℚ) / (hourly_rate * num_floors * rooms_per_floor : ℚ) :=
sorry

end NUMINAMATH_CALUDE_legacy_cleaning_time_l3803_380385


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3803_380315

-- Define a line passing through (2,5) with equal intercepts
structure EqualInterceptLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- Condition: Line passes through (2,5)
  point_condition : 5 = slope * 2 + y_intercept
  -- Condition: Equal intercepts on both axes
  equal_intercepts : y_intercept = slope * y_intercept

-- Theorem statement
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 5/2 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = 7) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3803_380315


namespace NUMINAMATH_CALUDE_max_value_of_f_l3803_380307

-- Define the function
def f (x : ℝ) : ℝ := (x + 1)^2 - 4

-- State the theorem
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 5 ∧ ∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3803_380307


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3803_380399

def i : ℂ := Complex.I

theorem complex_number_in_second_quadrant :
  let z : ℂ := (1 + 2*i) / (1 + 2*i^3)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3803_380399


namespace NUMINAMATH_CALUDE_function_composition_equality_l3803_380328

theorem function_composition_equality (c : ℝ) : 
  let p (x : ℝ) := 4 * x - 9
  let q (x : ℝ) := 5 * x - c
  p (q 3) = 14 → c = 9.25 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3803_380328


namespace NUMINAMATH_CALUDE_race_outcomes_l3803_380390

/-- The number of permutations of n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of positions to be filled -/
def positions_to_fill : ℕ := 4

theorem race_outcomes : permutations num_participants positions_to_fill = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l3803_380390


namespace NUMINAMATH_CALUDE_discriminant_5x2_minus_8x_plus_1_l3803_380339

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 8x + 1 is 44 -/
theorem discriminant_5x2_minus_8x_plus_1 : discriminant 5 (-8) 1 = 44 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_5x2_minus_8x_plus_1_l3803_380339


namespace NUMINAMATH_CALUDE_not_integer_fraction_l3803_380364

theorem not_integer_fraction (a b : ℕ) (ha : a > b) (hb : b > 2) :
  ¬ (∃ k : ℤ, (2^a + 1 : ℤ) = k * (2^b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_integer_fraction_l3803_380364


namespace NUMINAMATH_CALUDE_not_perfect_square_different_parity_l3803_380376

theorem not_perfect_square_different_parity (a b : ℤ) 
  (h : a % 2 ≠ b % 2) : 
  ¬∃ (k : ℤ), (a + 3*b) * (5*a + 7*b) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_different_parity_l3803_380376


namespace NUMINAMATH_CALUDE_remove_six_maximizes_probability_l3803_380314

def original_list : List Int := [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def remove_number (list : List Int) (n : Int) : List Int :=
  list.filter (· ≠ n)

def count_pairs_sum_12 (list : List Int) : Nat :=
  (list.filterMap (λ x => 
    if x < 12 ∧ list.contains (12 - x) ∧ x ≠ 12 - x
    then some (min x (12 - x))
    else none
  )).dedup.length

theorem remove_six_maximizes_probability : 
  ∀ n ∈ original_list, n ≠ 6 → 
    count_pairs_sum_12 (remove_number original_list 6) ≥ 
    count_pairs_sum_12 (remove_number original_list n) :=
by sorry

end NUMINAMATH_CALUDE_remove_six_maximizes_probability_l3803_380314


namespace NUMINAMATH_CALUDE_binomial_product_nine_two_seven_two_l3803_380391

theorem binomial_product_nine_two_seven_two :
  Nat.choose 9 2 * Nat.choose 7 2 = 756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_nine_two_seven_two_l3803_380391


namespace NUMINAMATH_CALUDE_probability_one_blue_one_white_l3803_380350

def total_marbles : ℕ := 8
def blue_marbles : ℕ := 3
def white_marbles : ℕ := 5
def marbles_left : ℕ := 2

def favorable_outcomes : ℕ := blue_marbles * white_marbles
def total_outcomes : ℕ := Nat.choose total_marbles marbles_left

theorem probability_one_blue_one_white :
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 28 := by sorry

end NUMINAMATH_CALUDE_probability_one_blue_one_white_l3803_380350


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3803_380302

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(- Real.sqrt b < a ∧ a < Real.sqrt b) → ¬(a^2 < b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ - Real.sqrt b) → a^2 ≥ b) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3803_380302


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3803_380344

/-- The perimeter of a semicircle with radius r is equal to 2r + πr -/
theorem semicircle_perimeter (r : ℝ) (h : r = 35) :
  ∃ P : ℝ, P = 2 * r + π * r := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3803_380344


namespace NUMINAMATH_CALUDE_candy_bar_cost_is_25_l3803_380308

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := sorry

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The number of quarters needed to buy the items -/
def quarters_needed : ℕ := 11

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of candy bars purchased -/
def candy_bars_bought : ℕ := 3

/-- The number of chocolate pieces purchased -/
def chocolates_bought : ℕ := 2

/-- The number of juice packs purchased -/
def juices_bought : ℕ := 1

theorem candy_bar_cost_is_25 : 
  candy_bar_cost = 25 :=
by
  have h1 : quarters_needed * quarter_value = 
    candy_bars_bought * candy_bar_cost + 
    chocolates_bought * chocolate_cost + 
    juices_bought * juice_cost := by sorry
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_is_25_l3803_380308


namespace NUMINAMATH_CALUDE_traffic_light_color_change_probability_l3803_380324

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle time of the traffic light -/
def totalCycleTime (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def colorChangeInterval (cycle : TrafficLightCycle) (observationTime : ℕ) : ℕ :=
  3 * observationTime

/-- Theorem: The probability of observing a color change in a randomly selected 
    4-second interval of a traffic light cycle is 12/85 -/
theorem traffic_light_color_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 40)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 40)
  (observationTime : ℕ)
  (h4 : observationTime = 4) :
  (colorChangeInterval cycle observationTime : ℚ) / (totalCycleTime cycle) = 12 / 85 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_color_change_probability_l3803_380324


namespace NUMINAMATH_CALUDE_complex_addition_multiplication_l3803_380316

theorem complex_addition_multiplication : 
  let z₁ : ℂ := 2 + 6 * I
  let z₂ : ℂ := 5 - 3 * I
  3 * (z₁ + z₂) = 21 + 9 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_multiplication_l3803_380316


namespace NUMINAMATH_CALUDE_initial_weight_calculation_l3803_380330

/-- 
Given a person who:
1. Loses 10% of their initial weight
2. Then gains 2 pounds
3. Ends up weighing 200 pounds

Their initial weight was 220 pounds.
-/
theorem initial_weight_calculation (initial_weight : ℝ) : 
  (initial_weight * 0.9 + 2 = 200) → initial_weight = 220 := by
  sorry

end NUMINAMATH_CALUDE_initial_weight_calculation_l3803_380330


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3803_380305

theorem complex_sum_theorem (a c d e f : ℝ) : 
  e = -a - c → (a + 2*I) + (c + d*I) + (e + f*I) = 2*I → d + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3803_380305


namespace NUMINAMATH_CALUDE_tangent_line_property_l3803_380317

/-- Given a line x + y = b tangent to the curve y = ax + 2/x at the point P(1, m), 
    prove that a + b - m = 2 -/
theorem tangent_line_property (a b m : ℝ) : 
  (∀ x, x + (a * x + 2 / x) = b) →  -- Line is tangent to the curve
  (1 + m = b) →                     -- Point P(1, m) is on the line
  (m = a + 2) →                     -- Point P(1, m) is on the curve
  (a + b - m = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_property_l3803_380317


namespace NUMINAMATH_CALUDE_sqrt_two_over_two_gt_sqrt_three_over_three_l3803_380329

theorem sqrt_two_over_two_gt_sqrt_three_over_three :
  (Real.sqrt 2) / 2 > (Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_over_two_gt_sqrt_three_over_three_l3803_380329


namespace NUMINAMATH_CALUDE_S_infinite_l3803_380341

/-- The number of distinct odd prime divisors of a natural number -/
def num_odd_prime_divisors (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which the number of distinct odd prime divisors of n(n+3) is divisible by 3 -/
def S : Set ℕ := {n : ℕ | 3 ∣ num_odd_prime_divisors (n * (n + 3))}

/-- The main theorem stating that S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l3803_380341


namespace NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l3803_380360

/-- The average waiting time for the first bite in a fishing scenario --/
theorem average_waiting_time_for_first_bite 
  (time_interval : ℝ) 
  (first_rod_bites : ℝ) 
  (second_rod_bites : ℝ) 
  (total_bites : ℝ) 
  (h1 : time_interval = 6)
  (h2 : first_rod_bites = 3)
  (h3 : second_rod_bites = 2)
  (h4 : total_bites = first_rod_bites + second_rod_bites) :
  (time_interval / total_bites) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l3803_380360


namespace NUMINAMATH_CALUDE_cheese_grating_time_is_five_l3803_380303

/-- The time in minutes it takes to grate cheese for one omelet --/
def cheese_grating_time (
  total_time : ℕ)
  (num_omelets : ℕ)
  (pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (omelet_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ) : ℕ :=
  total_time - 
  (num_peppers * pepper_chop_time + 
   num_onions * onion_chop_time + 
   num_omelets * omelet_cook_time)

theorem cheese_grating_time_is_five :
  cheese_grating_time 50 5 3 4 5 4 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cheese_grating_time_is_five_l3803_380303


namespace NUMINAMATH_CALUDE_set_A_characterization_intersection_A_B_complement_A_union_B_l3803_380322

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def B : Set ℝ := {x | x ≤ 4}

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Theorem statements
theorem set_A_characterization : A = {x | x > 3 ∨ x < -1} := by sorry

theorem intersection_A_B : A ∩ B = {x | 3 < x ∧ x ≤ 4} := by sorry

theorem complement_A_union_B : (Set.compl A) ∪ B = {x | x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_set_A_characterization_intersection_A_B_complement_A_union_B_l3803_380322


namespace NUMINAMATH_CALUDE_percent_problem_l3803_380301

theorem percent_problem (x : ℝ) : (0.0001 * x = 1.2356) → x = 12356 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l3803_380301


namespace NUMINAMATH_CALUDE_nested_root_evaluation_l3803_380320

theorem nested_root_evaluation (N : ℝ) (h : N > 1) :
  (N * (N * (N ^ (1/3)) ^ (1/4))) ^ (1/3) = N ^ (4/9) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_evaluation_l3803_380320


namespace NUMINAMATH_CALUDE_first_price_increase_l3803_380342

theorem first_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.15 = 1.38 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_price_increase_l3803_380342


namespace NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l3803_380379

theorem abs_x_minus_y_equals_four (x y : ℝ) 
  (h1 : x^3 + y^3 = 26) 
  (h2 : x*y*(x+y) = -6) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l3803_380379


namespace NUMINAMATH_CALUDE_factoring_expression_l3803_380313

theorem factoring_expression (x y : ℝ) : 3*x*(x+3) + y*(x+3) = (x+3)*(3*x+y) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3803_380313


namespace NUMINAMATH_CALUDE_card_addition_l3803_380318

theorem card_addition (initial_cards : ℕ) (added_cards : ℕ) : 
  initial_cards = 9 → added_cards = 4 → initial_cards + added_cards = 13 := by
sorry

end NUMINAMATH_CALUDE_card_addition_l3803_380318


namespace NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l3803_380357

theorem sum_five_consecutive_squares_not_perfect_square (n : ℤ) :
  ¬∃ (m : ℤ), 5 * n^2 + 10 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_squares_not_perfect_square_l3803_380357


namespace NUMINAMATH_CALUDE_connie_marbles_l3803_380363

/-- Given that Juan has 25 more marbles than Connie and Juan has 64 marbles, 
    prove that Connie has 39 marbles. -/
theorem connie_marbles (connie juan : ℕ) 
  (h1 : juan = connie + 25) 
  (h2 : juan = 64) : 
  connie = 39 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l3803_380363


namespace NUMINAMATH_CALUDE_colinear_vectors_x_value_l3803_380353

/-- Two vectors are colinear if one is a scalar multiple of the other -/
def colinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem colinear_vectors_x_value :
  let a : ℝ × ℝ := (1, Real.sqrt (1 + Real.sin (20 * π / 180)))
  let b : ℝ × ℝ := (1 / Real.sin (55 * π / 180), x)
  colinear a b → x = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_colinear_vectors_x_value_l3803_380353


namespace NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l3803_380368

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the absolute value of the coefficient for a given r
def coeff (r : ℕ) : ℕ := binomial 7 r

-- State the theorem
theorem largest_coefficient_in_expansion :
  ∀ r : ℕ, r ≤ 7 → coeff r ≤ coeff 4 :=
sorry

end NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l3803_380368


namespace NUMINAMATH_CALUDE_problem_statement_l3803_380378

theorem problem_statement :
  ∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 →
    (2 * x^3 - 3 * x^2 + 1 ≥ 0) ∧
    ((2 / (1 + x^3) + 2 / (1 + y^3) + 2 / (1 + z^3) = 3) →
      ((1 - x) / (1 - x + x^2) + (1 - y) / (1 - y + y^2) + (1 - z) / (1 - z + z^2) ≥ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3803_380378


namespace NUMINAMATH_CALUDE_arithmetic_sequence_exists_geometric_sequence_not_exists_l3803_380398

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (a b c d : ℝ)
  (sum_opposite : a + c = 180 ∧ b + d = 180)
  (angle_bounds : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)

-- Theorem for arithmetic sequence
theorem arithmetic_sequence_exists (q : CyclicQuadrilateral) :
  ∃ (α d : ℝ), d ≠ 0 ∧
    q.a = α ∧ q.b = α + d ∧ q.c = α + 2*d ∧ q.d = α + 3*d :=
sorry

-- Theorem for geometric sequence
theorem geometric_sequence_not_exists (q : CyclicQuadrilateral) :
  ¬∃ (α r : ℝ), r ≠ 1 ∧ r > 0 ∧
    q.a = α ∧ q.b = α * r ∧ q.c = α * r^2 ∧ q.d = α * r^3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_exists_geometric_sequence_not_exists_l3803_380398


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l3803_380345

theorem range_of_a_for_false_proposition :
  ∀ (a : ℝ),
    (¬ ∃ (x₀ : ℝ), x₀^2 + a*x₀ - 4*a < 0) ↔
    (a ∈ Set.Icc (-16 : ℝ) 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l3803_380345


namespace NUMINAMATH_CALUDE_equation_solution_l3803_380333

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ↔ x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3803_380333


namespace NUMINAMATH_CALUDE_dinosaur_book_cost_l3803_380311

/-- The cost of a dinosaur book, given the total cost of three books and the costs of two of them. -/
theorem dinosaur_book_cost (total_cost dictionary_cost cookbook_cost : ℕ) 
  (h_total : total_cost = 37)
  (h_dict : dictionary_cost = 11)
  (h_cook : cookbook_cost = 7) :
  total_cost - dictionary_cost - cookbook_cost = 19 := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_book_cost_l3803_380311


namespace NUMINAMATH_CALUDE_absolute_value_equation_implies_fourth_power_l3803_380355

theorem absolute_value_equation_implies_fourth_power (m : ℝ) : 
  (abs m = m + 1) → (4*m - 1)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_implies_fourth_power_l3803_380355


namespace NUMINAMATH_CALUDE_peter_erasers_l3803_380381

theorem peter_erasers (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 3 → total = initial + received → total = 11 := by
  sorry

end NUMINAMATH_CALUDE_peter_erasers_l3803_380381


namespace NUMINAMATH_CALUDE_min_values_l3803_380338

-- Define the equation
def equation (x y : ℝ) : Prop := Real.log (3 * x) + Real.log y = Real.log (x + y + 1)

-- Theorem statement
theorem min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : equation x y) :
  (∀ a b, equation a b → x * y ≤ a * b) ∧
  (∀ a b, equation a b → x + y ≤ a + b) ∧
  (∀ a b, equation a b → 1 / x + 1 / y ≤ 1 / a + 1 / b) ∧
  x * y = 1 ∧ x + y = 2 ∧ 1 / x + 1 / y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_values_l3803_380338


namespace NUMINAMATH_CALUDE_complement_of_A_l3803_380336

def U : Set ℕ := {x | 0 < x ∧ x < 8}
def A : Set ℕ := {2, 4, 5}

theorem complement_of_A : (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3803_380336


namespace NUMINAMATH_CALUDE_other_side_formula_l3803_380387

/-- Represents a rectangle with perimeter 30 and one side x -/
structure Rectangle30 where
  x : ℝ
  other : ℝ
  perimeter_eq : x + other = 15

theorem other_side_formula (rect : Rectangle30) : rect.other = 15 - rect.x := by
  sorry

end NUMINAMATH_CALUDE_other_side_formula_l3803_380387


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l3803_380310

/-- Given a simple interest scenario, prove that the rate percent is 10% -/
theorem simple_interest_rate_percent (P A T : ℝ) (h1 : P = 750) (h2 : A = 1125) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 10 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l3803_380310


namespace NUMINAMATH_CALUDE_domain_of_composed_function_inequality_proof_l3803_380396

-- Definition of the function f
def f : Set ℝ := Set.Icc (1/2) 2

-- Theorem 1: Domain of y = f(2^x)
theorem domain_of_composed_function :
  {x : ℝ | 2^x ∈ f} = Set.Icc (-1) 1 := by sorry

-- Theorem 2: Inequality proof
theorem inequality_proof (x y : ℝ) (h1 : -2 < x) (h2 : x < y) (h3 : y < 1) :
  -3 < x - y ∧ x - y < 0 := by sorry

end NUMINAMATH_CALUDE_domain_of_composed_function_inequality_proof_l3803_380396


namespace NUMINAMATH_CALUDE_max_value_of_complex_number_l3803_380335

theorem max_value_of_complex_number (z : ℂ) : 
  Complex.abs (z - (3 - I)) = 2 → 
  (∀ w : ℂ, Complex.abs (w - (3 - I)) = 2 → Complex.abs (w + (1 + I)) ≤ Complex.abs (z + (1 + I))) → 
  Complex.abs (z + (1 + I)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_number_l3803_380335


namespace NUMINAMATH_CALUDE_olives_price_per_pound_l3803_380394

/-- Calculates the price per pound of olives given Teresa's shopping list and total spent --/
theorem olives_price_per_pound (sandwich_price : ℝ) (salami_price : ℝ) (olive_weight : ℝ) 
  (feta_weight : ℝ) (feta_price_per_pound : ℝ) (bread_price : ℝ) (total_spent : ℝ) 
  (h1 : sandwich_price = 7.75)
  (h2 : salami_price = 4)
  (h3 : olive_weight = 1/4)
  (h4 : feta_weight = 1/2)
  (h5 : feta_price_per_pound = 8)
  (h6 : bread_price = 2)
  (h7 : total_spent = 40) :
  (total_spent - (2 * sandwich_price + salami_price + 3 * salami_price + 
  feta_weight * feta_price_per_pound + bread_price)) / olive_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_olives_price_per_pound_l3803_380394


namespace NUMINAMATH_CALUDE_transport_probabilities_theorem_l3803_380340

structure TransportProbabilities where
  plane : ℝ
  ship : ℝ
  train : ℝ
  car : ℝ
  sum_to_one : plane + ship + train + car = 1
  all_nonnegative : plane ≥ 0 ∧ ship ≥ 0 ∧ train ≥ 0 ∧ car ≥ 0

def prob_train_or_plane (p : TransportProbabilities) : ℝ :=
  p.train + p.plane

def prob_not_ship (p : TransportProbabilities) : ℝ :=
  1 - p.ship

theorem transport_probabilities_theorem (p : TransportProbabilities)
    (h1 : p.plane = 0.2)
    (h2 : p.ship = 0.3)
    (h3 : p.train = 0.4)
    (h4 : p.car = 0.1) :
    prob_train_or_plane p = 0.6 ∧ prob_not_ship p = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_transport_probabilities_theorem_l3803_380340


namespace NUMINAMATH_CALUDE_john_memory_card_cost_l3803_380362

/-- The number of pictures John takes per day -/
def pictures_per_day : ℕ := 10

/-- The number of years John has been taking pictures -/
def years : ℕ := 3

/-- The number of images a memory card can store -/
def images_per_card : ℕ := 50

/-- The cost of each memory card in dollars -/
def cost_per_card : ℕ := 60

/-- The number of days in a year (assuming no leap years) -/
def days_per_year : ℕ := 365

theorem john_memory_card_cost :
  (years * days_per_year * pictures_per_day / images_per_card) * cost_per_card = 13140 :=
sorry

end NUMINAMATH_CALUDE_john_memory_card_cost_l3803_380362


namespace NUMINAMATH_CALUDE_binomial_10_2_l3803_380321

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l3803_380321


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3803_380352

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  a 2 + a 3 = 4 →               -- given condition
  a 1 + a 4 = 6 :=              -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3803_380352


namespace NUMINAMATH_CALUDE_new_student_weight_l3803_380367

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_avg = 14.6 →
  (initial_count : ℝ) * initial_avg + (initial_count + 1 : ℝ) * new_avg - (initial_count : ℝ) * initial_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l3803_380367


namespace NUMINAMATH_CALUDE_minuend_value_l3803_380309

theorem minuend_value (minuend subtrahend difference : ℕ) 
  (h : minuend + subtrahend + difference = 600) : minuend = 300 := by
  sorry

end NUMINAMATH_CALUDE_minuend_value_l3803_380309


namespace NUMINAMATH_CALUDE_num_possible_lists_eq_1728_l3803_380343

/-- The number of balls in the bin -/
def num_balls : ℕ := 12

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The number of possible lists when drawing with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem stating that the number of possible lists is 1728 -/
theorem num_possible_lists_eq_1728 : num_possible_lists = 1728 := by
  sorry

end NUMINAMATH_CALUDE_num_possible_lists_eq_1728_l3803_380343


namespace NUMINAMATH_CALUDE_a_squared_lt_one_sufficient_not_necessary_for_a_lt_two_l3803_380300

theorem a_squared_lt_one_sufficient_not_necessary_for_a_lt_two :
  (∀ a : ℝ, a^2 < 1 → a < 2) ∧
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_a_squared_lt_one_sufficient_not_necessary_for_a_lt_two_l3803_380300


namespace NUMINAMATH_CALUDE_pentagon_h_coordinate_l3803_380326

structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

def has_vertical_symmetry (p : Pentagon) : Prop := sorry

def area (p : Pentagon) : ℝ := sorry

theorem pentagon_h_coordinate (p : Pentagon) 
  (sym : has_vertical_symmetry p)
  (coords : p.F = (0, 0) ∧ p.G = (0, 6) ∧ p.H.1 = 3 ∧ p.J = (6, 0))
  (total_area : area p = 60) :
  p.H.2 = 14 := by sorry

end NUMINAMATH_CALUDE_pentagon_h_coordinate_l3803_380326


namespace NUMINAMATH_CALUDE_min_value_theorem_l3803_380319

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 →
  a^4 * b^3 * c^2 ≤ x^4 * y^3 * z^2 ∧
  a^4 * b^3 * c^2 = 1/1152 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3803_380319


namespace NUMINAMATH_CALUDE_train_speed_l3803_380332

/-- Given a train of length 180 meters that crosses a stationary point in 6 seconds,
    prove that its speed is 30 meters per second. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 6) :
  length / time = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3803_380332


namespace NUMINAMATH_CALUDE_competition_earnings_difference_l3803_380349

/-- Represents the earnings of a seller for a single day -/
structure DayEarnings where
  regular_sales : ℝ
  discounted_sales : ℝ
  tax_rate : ℝ
  exchange_rate : ℝ

/-- Calculates the total earnings for a day after tax and currency conversion -/
def calculate_day_earnings (e : DayEarnings) : ℝ :=
  let total_sales := e.regular_sales + e.discounted_sales
  let after_tax := total_sales * (1 - e.tax_rate)
  after_tax * e.exchange_rate

/-- Represents the earnings of a seller for the two-day competition -/
structure CompetitionEarnings where
  day1 : DayEarnings
  day2 : DayEarnings

/-- Calculates the total earnings for the two-day competition -/
def calculate_total_earnings (e : CompetitionEarnings) : ℝ :=
  calculate_day_earnings e.day1 + calculate_day_earnings e.day2

/-- Theorem statement for the competition earnings -/
theorem competition_earnings_difference
  (bert_earnings tory_earnings : CompetitionEarnings)
  (h_bert_day1 : bert_earnings.day1 = {
    regular_sales := 9 * 18,
    discounted_sales := 3 * (18 * 0.85),
    tax_rate := 0.05,
    exchange_rate := 1
  })
  (h_bert_day2 : bert_earnings.day2 = {
    regular_sales := 10 * 15,
    discounted_sales := 0,
    tax_rate := 0.05,
    exchange_rate := 1.4
  })
  (h_tory_day1 : tory_earnings.day1 = {
    regular_sales := 10 * 20,
    discounted_sales := 5 * (20 * 0.9),
    tax_rate := 0.05,
    exchange_rate := 1
  })
  (h_tory_day2 : tory_earnings.day2 = {
    regular_sales := 8 * 18,
    discounted_sales := 0,
    tax_rate := 0.05,
    exchange_rate := 1.4
  }) :
  calculate_total_earnings tory_earnings - calculate_total_earnings bert_earnings = 71.82 := by
  sorry


end NUMINAMATH_CALUDE_competition_earnings_difference_l3803_380349


namespace NUMINAMATH_CALUDE_integer_solution_proof_l3803_380361

theorem integer_solution_proof (a b c : ℤ) :
  a + b + c = 24 →
  a^2 + b^2 + c^2 = 210 →
  a * b * c = 440 →
  ({a, b, c} : Set ℤ) = {5, 8, 11} := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_proof_l3803_380361


namespace NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l3803_380366

/-- Given a set of four numbers with six pairwise sums, find the maximum value of x + y -/
theorem max_sum_of_pairwise_sums (a b c d : ℝ) : 
  let sums := [a + b, a + c, a + d, b + c, b + d, c + d]
  ∃ (perm : List ℝ), perm.Perm sums ∧ 
    perm.take 4 = [210, 336, 294, 252] →
  (∃ (x y : ℝ), x ∈ perm.drop 4 ∧ y ∈ perm.drop 4 ∧ x + y ≤ 798) ∧
  (∃ (a' b' c' d' : ℝ), 
    let sums' := [a' + b', a' + c', a' + d', b' + c', b' + d', c' + d']
    ∃ (perm' : List ℝ), perm'.Perm sums' ∧ 
      perm'.take 4 = [210, 336, 294, 252] ∧
      ∃ (x' y' : ℝ), x' ∈ perm'.drop 4 ∧ y' ∈ perm'.drop 4 ∧ x' + y' = 798) := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l3803_380366


namespace NUMINAMATH_CALUDE_different_color_chip_probability_l3803_380312

theorem different_color_chip_probability : 
  let total_chips : ℕ := 6 + 5 + 4
  let green_chips : ℕ := 6
  let blue_chips : ℕ := 5
  let red_chips : ℕ := 4
  let prob_green : ℚ := green_chips / total_chips
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_red : ℚ := red_chips / total_chips
  let prob_not_green : ℚ := (blue_chips + red_chips) / total_chips
  let prob_not_blue : ℚ := (green_chips + red_chips) / total_chips
  let prob_not_red : ℚ := (green_chips + blue_chips) / total_chips
  prob_green * prob_not_green + prob_blue * prob_not_blue + prob_red * prob_not_red = 148 / 225 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chip_probability_l3803_380312


namespace NUMINAMATH_CALUDE_distance_between_centers_l3803_380304

/-- Given an isosceles triangle with circumradius R and inradius r,
    the distance d between the centers of the circumcircle and incircle
    is given by d = √(R(R-2r)). -/
theorem distance_between_centers (R r : ℝ) (h : R > 0 ∧ r > 0) :
  ∃ (d : ℝ), d = Real.sqrt (R * (R - 2 * r)) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3803_380304


namespace NUMINAMATH_CALUDE_smallest_twin_egg_number_l3803_380388

/-- Definition of a "twin egg number" -/
def is_twin_egg (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧
  ((n / 100) % 10 = (n / 10) % 10)

/-- Function to swap digits as described -/
def swap_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  b * 1000 + a * 100 + d * 10 + c

/-- The F function as defined in the problem -/
def F (m : ℕ) : ℤ := (m - swap_digits m) / 11

/-- Main theorem statement -/
theorem smallest_twin_egg_number :
  ∀ m : ℕ,
  is_twin_egg m →
  (m / 1000 ≠ (m / 100) % 10) →
  ∃ k : ℕ, F m / 27 = k * k →
  4114 ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_twin_egg_number_l3803_380388


namespace NUMINAMATH_CALUDE_sin_negative_390_degrees_l3803_380359

theorem sin_negative_390_degrees : 
  Real.sin ((-390 : ℝ) * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_390_degrees_l3803_380359


namespace NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l3803_380323

/-- Represents the number of students in a school -/
structure School where
  boarders : ℕ
  dayScholars : ℕ

/-- Represents a ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def School.ratio (s : School) : Ratio :=
  { numerator := s.boarders, denominator := s.dayScholars }

def School.addBoarders (s : School) (n : ℕ) : School :=
  { boarders := s.boarders + n, dayScholars := s.dayScholars }

theorem new_ratio_after_boarders_join
  (initialSchool : School)
  (initialRatio : Ratio)
  (newBoarders : ℕ) :
  initialSchool.ratio = initialRatio →
  initialSchool.boarders = 560 →
  initialRatio.numerator = 7 →
  initialRatio.denominator = 16 →
  newBoarders = 80 →
  (initialSchool.addBoarders newBoarders).ratio =
    { numerator := 1, denominator := 2 } :=
by
  sorry

end NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l3803_380323


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3803_380348

theorem rhombus_diagonal (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) :
  area = 135 →
  ratio_long = 5 →
  ratio_short = 3 →
  (ratio_long * ratio_short * (longer_diagonal ^ 2)) / (2 * (ratio_long ^ 2)) = area →
  longer_diagonal = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3803_380348


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3803_380389

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3803_380389


namespace NUMINAMATH_CALUDE_pencils_misplaced_l3803_380351

theorem pencils_misplaced (initial : ℕ) (broken found bought final : ℕ) : 
  initial = 20 →
  broken = 3 →
  found = 4 →
  bought = 2 →
  final = 16 →
  initial - broken + found + bought - final = 7 := by
  sorry

end NUMINAMATH_CALUDE_pencils_misplaced_l3803_380351


namespace NUMINAMATH_CALUDE_total_pay_for_given_scenario_l3803_380331

/-- The total amount paid to two employees, where one is paid 120% of the other's pay -/
def total_pay (y_pay : ℝ) : ℝ :=
  y_pay + 1.2 * y_pay

theorem total_pay_for_given_scenario :
  total_pay 260 = 572 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_for_given_scenario_l3803_380331


namespace NUMINAMATH_CALUDE_curve_arc_length_l3803_380397

noncomputable def arcLength (t₁ t₂ : Real) : Real :=
  ∫ t in t₁..t₂, Real.sqrt ((12 * Real.cos t ^ 2 * Real.sin t) ^ 2 + (12 * Real.sin t ^ 2 * Real.cos t) ^ 2)

theorem curve_arc_length :
  arcLength (π / 6) (π / 4) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_arc_length_l3803_380397
