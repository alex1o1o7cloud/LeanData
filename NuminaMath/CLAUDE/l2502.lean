import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2502_250299

theorem problem_1 : 2 * Real.cos (45 * π / 180) + (π - Real.sqrt 3) ^ 0 - Real.sqrt 8 = 1 - Real.sqrt 2 := by
  sorry

theorem problem_2 (m : ℝ) (h : m ≠ 1) : 
  ((2 / (m - 1) + 1) / ((2 * m + 2) / (m^2 - 2 * m + 1))) = (m - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2502_250299


namespace NUMINAMATH_CALUDE_min_value_theorem_l2502_250237

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (2^x) + Real.log (8^y) = Real.log 2) : 
  (x + y) / (x * y) ≥ 2 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2502_250237


namespace NUMINAMATH_CALUDE_orange_boxes_needed_l2502_250239

/-- Calculates the number of boxes needed for oranges given the initial conditions --/
theorem orange_boxes_needed (baskets : ℕ) (oranges_per_basket : ℕ) (oranges_eaten : ℕ) (oranges_per_box : ℕ)
  (h1 : baskets = 7)
  (h2 : oranges_per_basket = 31)
  (h3 : oranges_eaten = 3)
  (h4 : oranges_per_box = 17) :
  (baskets * oranges_per_basket - oranges_eaten + oranges_per_box - 1) / oranges_per_box = 13 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_needed_l2502_250239


namespace NUMINAMATH_CALUDE_third_part_value_l2502_250265

theorem third_part_value (total : ℚ) (ratio1 ratio2 ratio3 : ℚ) 
  (h_total : total = 782)
  (h_ratio1 : ratio1 = 1/2)
  (h_ratio2 : ratio2 = 2/3)
  (h_ratio3 : ratio3 = 3/4) :
  (ratio3 / (ratio1 + ratio2 + ratio3)) * total = 306 :=
by sorry

end NUMINAMATH_CALUDE_third_part_value_l2502_250265


namespace NUMINAMATH_CALUDE_parallelogram_area_18_10_l2502_250227

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 18 cm and height 10 cm is 180 cm² -/
theorem parallelogram_area_18_10 : 
  parallelogram_area 18 10 = 180 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_18_10_l2502_250227


namespace NUMINAMATH_CALUDE_cargo_passenger_relationship_l2502_250284

/-- Represents a train with passenger cars and cargo cars. -/
structure Train where
  total_cars : ℕ
  passenger_cars : ℕ
  cargo_cars : ℕ

/-- Defines the properties of our specific train. -/
def our_train : Train where
  total_cars := 71
  passenger_cars := 44
  cargo_cars := 25

/-- Theorem stating the relationship between cargo cars and passenger cars. -/
theorem cargo_passenger_relationship (t : Train) 
  (h1 : t.total_cars = t.passenger_cars + t.cargo_cars + 2) 
  (h2 : t.cargo_cars = t.passenger_cars / 2 + (t.cargo_cars - t.passenger_cars / 2)) : 
  t.cargo_cars - t.passenger_cars / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_cargo_passenger_relationship_l2502_250284


namespace NUMINAMATH_CALUDE_small_circles_radius_l2502_250275

theorem small_circles_radius (R : ℝ) (r : ℝ) : 
  R = 10 → 3 * (2 * r) = 2 * R → r = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_small_circles_radius_l2502_250275


namespace NUMINAMATH_CALUDE_coin_game_probability_l2502_250234

def coin_game (n : ℕ) : ℝ :=
  sorry

theorem coin_game_probability : coin_game 5 = 1521 / 2^15 := by
  sorry

end NUMINAMATH_CALUDE_coin_game_probability_l2502_250234


namespace NUMINAMATH_CALUDE_beatrix_height_relative_to_georgia_l2502_250282

theorem beatrix_height_relative_to_georgia (B V G : ℝ) 
  (h1 : B = 2 * V) 
  (h2 : V = 2/3 * G) : 
  B = 4/3 * G := by
sorry

end NUMINAMATH_CALUDE_beatrix_height_relative_to_georgia_l2502_250282


namespace NUMINAMATH_CALUDE_unique_intersection_iff_k_eq_22_div_3_l2502_250292

/-- The parabola function -/
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 7

/-- The line function -/
def line (k : ℝ) : ℝ := k

/-- The condition for exactly one intersection point -/
def has_unique_intersection (k : ℝ) : Prop :=
  ∃! y, parabola y = line k

theorem unique_intersection_iff_k_eq_22_div_3 :
  ∀ k : ℝ, has_unique_intersection k ↔ k = 22 / 3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_iff_k_eq_22_div_3_l2502_250292


namespace NUMINAMATH_CALUDE_arithmetic_progression_theorem_l2502_250259

/-- An arithmetic progression with n terms -/
structure ArithmeticProgression where
  n : ℕ
  a : ℕ → ℕ
  d : ℕ
  progression : ∀ i, i < n → a (i + 1) = a i + d

/-- The sum of an arithmetic progression -/
def sum (ap : ArithmeticProgression) : ℕ :=
  (ap.n * (2 * ap.a 0 + (ap.n - 1) * ap.d)) / 2

theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum ap = 112 ∧
  ap.a 1 * ap.d = 30 ∧
  ap.a 2 + ap.a 4 = 32 →
  ap.n = 7 ∧
  ((ap.a 0 = 7 ∧ ap.a 1 = 10 ∧ ap.a 2 = 13) ∨
   (ap.a 0 = 1 ∧ ap.a 1 = 6 ∧ ap.a 2 = 11)) :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_progression_theorem_l2502_250259


namespace NUMINAMATH_CALUDE_f_at_2_eq_neg_22_l2502_250205

/-- Given a function f(x) = x^5 - ax^3 + bx - 6 where f(-2) = 10, prove that f(2) = -22 -/
theorem f_at_2_eq_neg_22 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 - a*x^3 + b*x - 6)
    (h2 : f (-2) = 10) : 
  f 2 = -22 := by sorry

end NUMINAMATH_CALUDE_f_at_2_eq_neg_22_l2502_250205


namespace NUMINAMATH_CALUDE_abs_negative_seventeen_l2502_250249

theorem abs_negative_seventeen : |(-17 : ℤ)| = 17 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_seventeen_l2502_250249


namespace NUMINAMATH_CALUDE_albert_needs_more_money_l2502_250204

-- Define the costs of items and Albert's current money
def paintbrush_cost : ℚ := 1.50
def paints_cost : ℚ := 4.35
def easel_cost : ℚ := 12.65
def canvas_cost : ℚ := 7.95
def palette_cost : ℚ := 3.75
def albert_current_money : ℚ := 10.60

-- Define the total cost of items
def total_cost : ℚ := paintbrush_cost + paints_cost + easel_cost + canvas_cost + palette_cost

-- Theorem: Albert needs $19.60 more
theorem albert_needs_more_money : total_cost - albert_current_money = 19.60 := by
  sorry

end NUMINAMATH_CALUDE_albert_needs_more_money_l2502_250204


namespace NUMINAMATH_CALUDE_selection_problem_l2502_250256

theorem selection_problem (n_sergeants m_soldiers : ℕ) 
  (k_sergeants k_soldiers : ℕ) (factor : ℕ) :
  n_sergeants = 6 →
  m_soldiers = 60 →
  k_sergeants = 2 →
  k_soldiers = 20 →
  factor = 3 →
  (factor * Nat.choose n_sergeants k_sergeants * Nat.choose m_soldiers k_soldiers) = 
  (3 * Nat.choose 6 2 * Nat.choose 60 20) :=
by sorry

end NUMINAMATH_CALUDE_selection_problem_l2502_250256


namespace NUMINAMATH_CALUDE_bike_speed_l2502_250213

/-- Given a bike moving at a constant speed that covers 5400 meters in 9 minutes,
    prove that its speed is 10 meters per second. -/
theorem bike_speed (distance : ℝ) (time_minutes : ℝ) (speed : ℝ) 
    (h1 : distance = 5400)
    (h2 : time_minutes = 9)
    (h3 : speed = distance / (time_minutes * 60)) : 
    speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_bike_speed_l2502_250213


namespace NUMINAMATH_CALUDE_birds_left_after_week_l2502_250285

/-- Calculates the number of birds left in a poultry farm after a week of disease -/
def birdsLeftAfterWeek (initialChickens initialTurkeys initialGuineaFowls : ℕ)
                       (dailyLossChickens dailyLossTurkeys dailyLossGuineaFowls : ℕ) : ℕ :=
  let daysInWeek : ℕ := 7
  let chickensLeft := initialChickens - daysInWeek * dailyLossChickens
  let turkeysLeft := initialTurkeys - daysInWeek * dailyLossTurkeys
  let guineaFowlsLeft := initialGuineaFowls - daysInWeek * dailyLossGuineaFowls
  chickensLeft + turkeysLeft + guineaFowlsLeft

theorem birds_left_after_week :
  birdsLeftAfterWeek 300 200 80 20 8 5 = 349 := by
  sorry

end NUMINAMATH_CALUDE_birds_left_after_week_l2502_250285


namespace NUMINAMATH_CALUDE_school_population_theorem_l2502_250289

theorem school_population_theorem (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 400 →
  boys + girls = total →
  girls = (boys * 100) / total →
  boys = 320 := by
sorry

end NUMINAMATH_CALUDE_school_population_theorem_l2502_250289


namespace NUMINAMATH_CALUDE_y_value_at_x_2_l2502_250209

theorem y_value_at_x_2 :
  let y₁ := λ x : ℝ => x^2 - 7*x + 6
  let y₂ := λ x : ℝ => 7*x - 3
  let y := λ x : ℝ => y₁ x + x * y₂ x
  y 2 = 18 := by sorry

end NUMINAMATH_CALUDE_y_value_at_x_2_l2502_250209


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2502_250266

theorem complex_number_quadrant : 
  let z : ℂ := (Complex.I : ℂ) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2502_250266


namespace NUMINAMATH_CALUDE_problem_solution_l2502_250261

theorem problem_solution (x : ℚ) : 
  (1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 6 : ℚ) = 4 / x → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2502_250261


namespace NUMINAMATH_CALUDE_solution_set_of_trig_equation_l2502_250260

theorem solution_set_of_trig_equation :
  let S : Set ℝ := {x | 5 * Real.sin x = 4 + 2 * Real.cos (2 * x)}
  S = {x | ∃ k : ℤ, x = Real.arcsin (3/4) + 2 * k * Real.pi ∨ 
                    x = Real.pi - Real.arcsin (3/4) + 2 * k * Real.pi} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_trig_equation_l2502_250260


namespace NUMINAMATH_CALUDE_largest_base5_is_124_l2502_250216

/-- Represents a three-digit base-5 number -/
structure Base5Number where
  hundreds : Fin 5
  tens : Fin 5
  ones : Fin 5

/-- Converts a Base5Number to its decimal (base 10) representation -/
def toDecimal (n : Base5Number) : ℕ :=
  n.hundreds * 25 + n.tens * 5 + n.ones

/-- The largest three-digit base-5 number -/
def largestBase5 : Base5Number :=
  { hundreds := 4, tens := 4, ones := 4 }

theorem largest_base5_is_124 : toDecimal largestBase5 = 124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_is_124_l2502_250216


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_equality_l2502_250268

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.repeatingPart / (99 : ℚ)

/-- The main theorem stating that the given fraction of repeating decimals equals the specified rational number -/
theorem repeating_decimal_fraction_equality : 
  let a := RepeatingDecimal.mk 0 75
  let b := RepeatingDecimal.mk 2 25
  (toRational a) / (toRational b) = 2475 / 7339 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_equality_l2502_250268


namespace NUMINAMATH_CALUDE_percent_of_number_l2502_250217

theorem percent_of_number (percent : ℝ) (number : ℝ) (result : ℝ) :
  percent = 37.5 ∧ number = 725 ∧ result = 271.875 →
  (percent / 100) * number = result :=
by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l2502_250217


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2502_250228

/-- Given four points A, B, C, and D on a line in that order, with AB = 4, BC = 3, and AD = 20,
    prove that the ratio of AC to BD is 7/16. -/
theorem ratio_of_segments (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D → -- Points lie on a line in order
  B - A = 4 →             -- AB = 4
  C - B = 3 →             -- BC = 3
  D - A = 20 →            -- AD = 20
  (C - A) / (D - B) = 7 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2502_250228


namespace NUMINAMATH_CALUDE_art_supplies_theorem_l2502_250200

def art_supplies_problem (total_spent canvas_cost paint_cost_ratio easel_cost : ℚ) : Prop :=
  let paint_cost := canvas_cost * paint_cost_ratio
  let other_items_cost := canvas_cost + paint_cost + easel_cost
  let paintbrush_cost := total_spent - other_items_cost
  paintbrush_cost = 15

theorem art_supplies_theorem :
  art_supplies_problem 90 40 (1/2) 15 := by
  sorry

end NUMINAMATH_CALUDE_art_supplies_theorem_l2502_250200


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2502_250287

/-- The perimeter of an equilateral triangle whose area is numerically equal to twice its side length is 8√3. -/
theorem equilateral_triangle_perimeter : 
  ∀ s : ℝ, s > 0 → (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2502_250287


namespace NUMINAMATH_CALUDE_ten_player_tournament_rounds_l2502_250276

/-- The number of rounds needed for a round-robin tennis tournament -/
def roundsNeeded (players : ℕ) (courts : ℕ) : ℕ :=
  (players * (players - 1) / 2 + courts - 1) / courts

/-- Theorem: A 10-player round-robin tournament on 5 courts needs 9 rounds -/
theorem ten_player_tournament_rounds :
  roundsNeeded 10 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_rounds_l2502_250276


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_closed_interval_l2502_250203

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 2*x ≤ 0}
def Q : Set ℝ := {y | ∃ x, y = x^2 - 2*x}

-- State the theorem
theorem P_intersect_Q_equals_closed_interval :
  P ∩ Q = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_closed_interval_l2502_250203


namespace NUMINAMATH_CALUDE_garden_area_l2502_250233

/-- A rectangular garden with perimeter 36 feet and one side 10 feet has an area of 80 square feet. -/
theorem garden_area (perimeter : ℝ) (side : ℝ) (h1 : perimeter = 36) (h2 : side = 10) :
  let other_side := (perimeter - 2 * side) / 2
  side * other_side = 80 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l2502_250233


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2502_250269

/-- The quadratic function f(x) = (x-2)^2 - 1 -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 1

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (2, -1)

/-- Theorem: The vertex of the quadratic function f(x) = (x-2)^2 - 1 is at the point (2, -1) -/
theorem quadratic_vertex : 
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ vertex.2 = f (vertex.1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2502_250269


namespace NUMINAMATH_CALUDE_mixture_replacement_solution_l2502_250201

/-- Represents the mixture replacement problem -/
def MixtureReplacement (initial_A : ℝ) (initial_ratio_A : ℝ) (initial_ratio_B : ℝ) 
                       (final_ratio_A : ℝ) (final_ratio_B : ℝ) : Prop :=
  let initial_B := initial_A * initial_ratio_B / initial_ratio_A
  let replaced_amount := 
    (final_ratio_B * initial_A - final_ratio_A * initial_B) / 
    (final_ratio_A + final_ratio_B)
  replaced_amount = 40

/-- Theorem stating the solution to the mixture replacement problem -/
theorem mixture_replacement_solution :
  MixtureReplacement 32 4 1 2 3 := by
  sorry

end NUMINAMATH_CALUDE_mixture_replacement_solution_l2502_250201


namespace NUMINAMATH_CALUDE_div_exp_eq_pow_reciprocal_l2502_250202

/-- Division exponentiation for rational numbers -/
def div_exp (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (div_exp a (n - 1))

/-- Theorem: Division exponentiation equals power of reciprocal -/
theorem div_exp_eq_pow_reciprocal (a : ℚ) (n : ℕ) (h1 : a ≠ 0) (h2 : n ≥ 3) :
  div_exp a n = (1 / a) ^ (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_div_exp_eq_pow_reciprocal_l2502_250202


namespace NUMINAMATH_CALUDE_three_distinct_prime_factors_l2502_250221

theorem three_distinct_prime_factors (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_order : q > p ∧ p > 2) :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c ∣ 2^(p*q) - 1) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_prime_factors_l2502_250221


namespace NUMINAMATH_CALUDE_cube_root_inequality_l2502_250291

theorem cube_root_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.rpow (a * b) (1/3) + Real.rpow (c * d) (1/3) ≤ Real.rpow ((a + b + c) * (b + c + d)) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l2502_250291


namespace NUMINAMATH_CALUDE_perfect_square_conversion_l2502_250277

theorem perfect_square_conversion (a b : ℝ) : 9 * a^4 * b^2 - 42 * a^2 * b = (3 * a^2 * b - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_conversion_l2502_250277


namespace NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l2502_250206

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → |x| < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l2502_250206


namespace NUMINAMATH_CALUDE_race_distance_l2502_250253

/-- A race between two runners p and q, where p is faster but q gets a head start -/
structure Race where
  /-- The speed of runner q (in meters per second) -/
  q_speed : ℝ
  /-- The speed of runner p (in meters per second) -/
  p_speed : ℝ
  /-- The head start given to runner q (in meters) -/
  head_start : ℝ
  /-- The condition that p is 25% faster than q -/
  speed_ratio : p_speed = 1.25 * q_speed
  /-- The head start is 60 meters -/
  head_start_value : head_start = 60

/-- The theorem stating that if the race ends in a tie, p ran 300 meters -/
theorem race_distance (race : Race) : 
  (∃ t : ℝ, race.q_speed * t = race.p_speed * t - race.head_start) → 
  race.p_speed * (300 / race.p_speed) = 300 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l2502_250253


namespace NUMINAMATH_CALUDE_cheesecake_calories_per_slice_quarter_of_slices_is_two_l2502_250270

/-- Represents a cheesecake with its total calories and number of slices -/
structure Cheesecake where
  totalCalories : ℕ
  numSlices : ℕ

/-- Calculates the number of calories per slice in a cheesecake -/
def caloriesPerSlice (cake : Cheesecake) : ℕ :=
  cake.totalCalories / cake.numSlices

theorem cheesecake_calories_per_slice :
  ∀ (cake : Cheesecake),
    cake.totalCalories = 2800 →
    cake.numSlices = 8 →
    caloriesPerSlice cake = 350 := by
  sorry

/-- Verifies that 25% of the total slices is equal to 2 slices -/
theorem quarter_of_slices_is_two (cake : Cheesecake) :
  cake.numSlices = 8 →
  cake.numSlices / 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cheesecake_calories_per_slice_quarter_of_slices_is_two_l2502_250270


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l2502_250281

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 4 ∧ 
  (∀ n : ℕ, n > 0 → ∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ k' : ℕ, k' < k → ∃ n : ℕ, n > 0 ∧ ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l2502_250281


namespace NUMINAMATH_CALUDE_area_of_polygon_AIHFGD_l2502_250278

-- Define the points
variable (A B C D E F G H I : ℝ × ℝ)

-- Define the squares
def is_square (P Q R S : ℝ × ℝ) : Prop := sorry

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Define midpoint
def is_midpoint (M P Q : ℝ × ℝ) : Prop := sorry

theorem area_of_polygon_AIHFGD :
  is_square A B C D →
  is_square E F G D →
  area [A, B, C, D] = 25 →
  area [E, F, G, D] = 25 →
  is_midpoint H B C →
  is_midpoint H E F →
  is_midpoint I A B →
  area [A, I, H, F, G, D] = 25 := by
  sorry

end NUMINAMATH_CALUDE_area_of_polygon_AIHFGD_l2502_250278


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2502_250262

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2502_250262


namespace NUMINAMATH_CALUDE_fescue_percentage_in_Y_l2502_250250

/-- Represents a seed mixture --/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y --/
def CombinedMixture (X Y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := xWeight * X.ryegrass + (1 - xWeight) * Y.ryegrass,
    bluegrass := xWeight * X.bluegrass + (1 - xWeight) * Y.bluegrass,
    fescue := xWeight * X.fescue + (1 - xWeight) * Y.fescue }

theorem fescue_percentage_in_Y
  (X : SeedMixture)
  (Y : SeedMixture)
  (h1 : X.ryegrass = 0.4)
  (h2 : X.bluegrass = 0.6)
  (h3 : Y.ryegrass = 0.25)
  (h4 : (CombinedMixture X Y (1/3)).ryegrass = 0.3)
  : Y.fescue = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fescue_percentage_in_Y_l2502_250250


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l2502_250235

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_oranges : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_oranges = 30)
  (h4 : afternoon_apples = 50)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ (morning_apples : ℕ), 
    apple_price * (morning_apples + afternoon_apples) + 
    orange_price * (morning_oranges + afternoon_oranges) = total_sales ∧
    morning_apples = 40 := by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l2502_250235


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l2502_250240

theorem inscribed_circle_square_area : 
  ∀ (r : ℝ) (s : ℝ),
  r > 0 →
  s > 0 →
  π * r^2 = 9 * π →
  2 * r = s →
  s^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l2502_250240


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l2502_250248

/-- The probability of drawing a red ball exactly on the fourth draw with replacement -/
def prob_red_fourth_with_replacement (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  (1 - red_balls / total_balls) ^ 3 * (red_balls / total_balls)

/-- The probability of drawing a red ball exactly on the fourth draw without replacement -/
def prob_red_fourth_without_replacement (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) *
  ((white_balls - 2) / (total_balls - 2)) * (red_balls / (total_balls - 3))

theorem ball_drawing_probabilities :
  let total_balls := 10
  let red_balls := 6
  let white_balls := 4
  prob_red_fourth_with_replacement total_balls red_balls = 24 / 625 ∧
  prob_red_fourth_without_replacement total_balls red_balls white_balls = 1 / 70 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l2502_250248


namespace NUMINAMATH_CALUDE_parallelogram_area_is_36_l2502_250208

-- Define the vectors v and w
def v : ℝ × ℝ := (4, -6)
def w : ℝ × ℝ := (8, -3)

-- Define the area of the parallelogram
def parallelogramArea (a b : ℝ × ℝ) : ℝ :=
  |a.1 * b.2 - a.2 * b.1|

-- Theorem statement
theorem parallelogram_area_is_36 :
  parallelogramArea v w = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_36_l2502_250208


namespace NUMINAMATH_CALUDE_a_share_in_profit_l2502_250254

/-- Given the investments of A, B, and C, and the total profit, prove A's share in the profit --/
theorem a_share_in_profit 
  (a_investment b_investment c_investment total_profit : ℕ) 
  (h1 : a_investment = 2400)
  (h2 : b_investment = 7200)
  (h3 : c_investment = 9600)
  (h4 : total_profit = 9000) :
  a_investment * total_profit / (a_investment + b_investment + c_investment) = 1125 := by
  sorry

end NUMINAMATH_CALUDE_a_share_in_profit_l2502_250254


namespace NUMINAMATH_CALUDE_car_journey_speed_l2502_250224

/-- Proves that given specific conditions about a car's journey, 
    the speed for the remaining part of the trip is 60 mph. -/
theorem car_journey_speed (D : ℝ) (h1 : D > 0) : 
  let first_part_distance := 0.4 * D
  let first_part_speed := 40
  let total_average_speed := 50
  let remaining_part_distance := 0.6 * D
  let remaining_part_speed := 
    remaining_part_distance / 
    (D / total_average_speed - first_part_distance / first_part_speed)
  remaining_part_speed = 60 := by
  sorry


end NUMINAMATH_CALUDE_car_journey_speed_l2502_250224


namespace NUMINAMATH_CALUDE_first_worker_load_time_l2502_250236

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℝ := 3.0769230769230766

/-- The time it takes for the second worker to load a truck alone -/
def second_worker_time : ℝ := 8

/-- The time it takes for the first worker to load a truck alone -/
def first_worker_time : ℝ := 5

/-- Theorem stating that given the combined time and the second worker's time, 
    the first worker's time to load the truck alone is 5 hours -/
theorem first_worker_load_time : 
  1 / first_worker_time + 1 / second_worker_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_first_worker_load_time_l2502_250236


namespace NUMINAMATH_CALUDE_missing_figure_proof_l2502_250231

theorem missing_figure_proof (x : ℝ) : (0.1 / 100) * x = 0.24 → x = 240 := by
  sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l2502_250231


namespace NUMINAMATH_CALUDE_inequality_solution_l2502_250223

theorem inequality_solution (x : ℝ) (h : x ≠ 4) :
  (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2502_250223


namespace NUMINAMATH_CALUDE_ceiling_times_self_equals_156_l2502_250247

theorem ceiling_times_self_equals_156 :
  ∃! (x : ℝ), ⌈x⌉ * x = 156 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_equals_156_l2502_250247


namespace NUMINAMATH_CALUDE_no_two_roots_exist_l2502_250229

-- Define the equation as a function of x, y, and a
def equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x = |x - a| - 1

-- Theorem statement
theorem no_two_roots_exist :
  ¬ ∃ (a : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧ 
    equation x₁ y₁ a ∧ 
    equation x₂ y₂ a ∧ 
    (∀ (x y : ℝ), equation x y a → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

end NUMINAMATH_CALUDE_no_two_roots_exist_l2502_250229


namespace NUMINAMATH_CALUDE_A_C_mutually_exclusive_l2502_250283

/-- Represents the sample space of three products -/
structure ThreeProducts where
  product1 : Bool  -- True if defective, False if not defective
  product2 : Bool
  product3 : Bool

/-- Event A: All three products are not defective -/
def A (s : ThreeProducts) : Prop :=
  ¬s.product1 ∧ ¬s.product2 ∧ ¬s.product3

/-- Event B: All three products are defective -/
def B (s : ThreeProducts) : Prop :=
  s.product1 ∧ s.product2 ∧ s.product3

/-- Event C: At least one of the three products is defective -/
def C (s : ThreeProducts) : Prop :=
  s.product1 ∨ s.product2 ∨ s.product3

/-- Theorem: A and C are mutually exclusive -/
theorem A_C_mutually_exclusive :
  ∀ s : ThreeProducts, ¬(A s ∧ C s) :=
by sorry

end NUMINAMATH_CALUDE_A_C_mutually_exclusive_l2502_250283


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l2502_250219

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1)

theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l2502_250219


namespace NUMINAMATH_CALUDE_one_third_of_cake_flour_one_third_of_cake_flour_mixed_number_l2502_250222

theorem one_third_of_cake_flour :
  let full_recipe : ℚ := 19 / 3
  let one_third_recipe : ℚ := full_recipe / 3
  one_third_recipe = 19 / 9 :=
by sorry

-- Convert to mixed number
theorem one_third_of_cake_flour_mixed_number :
  let full_recipe : ℚ := 19 / 3
  let one_third_recipe : ℚ := full_recipe / 3
  ∃ (whole : ℕ) (numerator : ℕ) (denominator : ℕ),
    one_third_recipe = whole + (numerator : ℚ) / denominator ∧
    whole = 2 ∧ numerator = 1 ∧ denominator = 9 :=
by sorry

end NUMINAMATH_CALUDE_one_third_of_cake_flour_one_third_of_cake_flour_mixed_number_l2502_250222


namespace NUMINAMATH_CALUDE_sqrt2_minus_1_power_form_l2502_250212

theorem sqrt2_minus_1_power_form (n : ℕ) :
  ∃ k : ℤ, (Real.sqrt 2 - 1) ^ n = Real.sqrt (k + 1) - Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_minus_1_power_form_l2502_250212


namespace NUMINAMATH_CALUDE_hawks_score_l2502_250230

theorem hawks_score (total_points margin eagles_three_pointers : ℕ) 
  (h1 : total_points = 82)
  (h2 : margin = 18)
  (h3 : eagles_three_pointers = 12) : 
  total_points - (total_points + margin) / 2 = 32 :=
sorry

end NUMINAMATH_CALUDE_hawks_score_l2502_250230


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l2502_250257

theorem smallest_k_for_64_power_gt_4_16 :
  ∀ k : ℕ, (64 : ℝ) ^ k > (4 : ℝ) ^ 16 ↔ k ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_16_l2502_250257


namespace NUMINAMATH_CALUDE_double_sum_equals_one_point_five_l2502_250267

/-- The double sum of 1/(mn(m+n+2)) from m=1 to ∞ and n=1 to ∞ equals 1.5 -/
theorem double_sum_equals_one_point_five :
  (∑' m : ℕ+, ∑' n : ℕ+, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_double_sum_equals_one_point_five_l2502_250267


namespace NUMINAMATH_CALUDE_quotient_problem_l2502_250252

theorem quotient_problem (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l2502_250252


namespace NUMINAMATH_CALUDE_max_log_sum_min_reciprocal_sum_l2502_250280

-- Define the conditions
variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h_eq : 2 * x + 5 * y = 20)

-- Theorem for the maximum value of log x + log y
theorem max_log_sum :
  ∃ (max : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + 5 * b = 20 → 
    Real.log a + Real.log b ≤ max ∧ 
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 2 * c + 5 * d = 20 ∧ Real.log c + Real.log d = max) ∧
    max = 1 :=
sorry

-- Theorem for the minimum value of 1/x + 1/y
theorem min_reciprocal_sum :
  ∃ (min : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + 5 * b = 20 → 
    1 / a + 1 / b ≥ min ∧ 
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 2 * c + 5 * d = 20 ∧ 1 / c + 1 / d = min) ∧
    min = (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_min_reciprocal_sum_l2502_250280


namespace NUMINAMATH_CALUDE_inequality_proof_l2502_250215

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2502_250215


namespace NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l2502_250298

/-- A polyhedron circumscribed around a sphere. -/
structure CircumscribedPolyhedron where
  -- The volume of the polyhedron
  volume : ℝ
  -- The radius of the inscribed sphere
  sphereRadius : ℝ
  -- The total surface area of the polyhedron
  surfaceArea : ℝ

/-- 
The volume of a polyhedron circumscribed around a sphere is equal to 
one-third of the product of the sphere's radius and the polyhedron's total surface area.
-/
theorem volume_of_circumscribed_polyhedron (p : CircumscribedPolyhedron) : 
  p.volume = (1 / 3) * p.sphereRadius * p.surfaceArea := by
  sorry

end NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l2502_250298


namespace NUMINAMATH_CALUDE_couscous_first_shipment_l2502_250271

theorem couscous_first_shipment (total_shipments : ℕ) 
  (shipment_a shipment_b first_shipment : ℝ) 
  (num_dishes : ℕ) (couscous_per_dish : ℝ) : 
  total_shipments = 3 →
  shipment_a = 13 →
  shipment_b = 45 →
  num_dishes = 13 →
  couscous_per_dish = 5 →
  first_shipment ≠ shipment_b →
  first_shipment = num_dishes * couscous_per_dish :=
by sorry

end NUMINAMATH_CALUDE_couscous_first_shipment_l2502_250271


namespace NUMINAMATH_CALUDE_gcf_of_60_and_75_l2502_250258

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_and_75_l2502_250258


namespace NUMINAMATH_CALUDE_product_325_7_4_7_l2502_250238

/-- Converts a base 7 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 --/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Theorem: The product of 325₇ and 4₇ is equal to 1656₇ in base 7 --/
theorem product_325_7_4_7 : 
  to_base_7 (to_base_10 [5, 2, 3] * to_base_10 [4]) = [6, 5, 6, 1] := by
  sorry

end NUMINAMATH_CALUDE_product_325_7_4_7_l2502_250238


namespace NUMINAMATH_CALUDE_ratio_composition_l2502_250232

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_composition_l2502_250232


namespace NUMINAMATH_CALUDE_complex_real_condition_l2502_250244

theorem complex_real_condition (a : ℝ) : 
  (((a : ℂ) + Complex.I) / (3 + 4 * Complex.I)).im = 0 → a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2502_250244


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2502_250286

/-- The custom operation ⊗ -/
def tensor (a b c d : ℂ) : ℂ := a * c - b * d

/-- The complex number z satisfying the given equation -/
noncomputable def z : ℂ := sorry

/-- The statement to prove -/
theorem z_in_second_quadrant :
  tensor z (1 - 2*I) (-1) (1 + I) = 0 →
  z.re < 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2502_250286


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l2502_250245

/-- Proves that the loss percentage is 10% given the selling prices with loss and with 10% gain --/
theorem book_sale_loss_percentage 
  (sp_loss : ℝ) 
  (sp_gain : ℝ) 
  (h_sp_loss : sp_loss = 450)
  (h_sp_gain : sp_gain = 550)
  (h_gain_percentage : sp_gain = 1.1 * (sp_gain / 1.1)) : 
  (((sp_gain / 1.1) - sp_loss) / (sp_gain / 1.1)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l2502_250245


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2502_250214

theorem solve_linear_equation (x y : ℝ) : 2 * x + y = 3 → y = 3 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2502_250214


namespace NUMINAMATH_CALUDE_function_and_composition_l2502_250210

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 - b*x + c

theorem function_and_composition 
  (h1 : f 1 b c = 0) 
  (h2 : f 2 b c = -3) :
  (∀ x, f x b c = x^2 - 6*x + 5) ∧ 
  (∀ x > -1, f (1 / Real.sqrt (x + 1)) b c = 1 / (x + 1) - 6 / Real.sqrt (x + 1) + 5) := by
  sorry

end NUMINAMATH_CALUDE_function_and_composition_l2502_250210


namespace NUMINAMATH_CALUDE_afternoon_morning_difference_l2502_250297

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 52

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 61

/-- The theorem states that the difference between the number of campers
    who went rowing in the afternoon and the number of campers who went
    rowing in the morning is 9 -/
theorem afternoon_morning_difference :
  afternoon_campers - morning_campers = 9 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_morning_difference_l2502_250297


namespace NUMINAMATH_CALUDE_negative_rational_power_equality_l2502_250263

theorem negative_rational_power_equality : 
  Real.rpow (-3 * (3/8)) (-(2/3)) = 4/9 := by sorry

end NUMINAMATH_CALUDE_negative_rational_power_equality_l2502_250263


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2502_250272

/-- The quadratic function f(x) = x^2 - 2x - 3 has exactly two real roots -/
theorem quadratic_two_roots : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
  (∀ x, x^2 - 2*x - 3 = 0 ↔ x = r₁ ∨ x = r₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2502_250272


namespace NUMINAMATH_CALUDE_symmetry_implies_difference_l2502_250241

/-- Two points are symmetric with respect to the origin if the sum of their coordinates is (0,0) -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_origin a (-2) 4 b → a - b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_difference_l2502_250241


namespace NUMINAMATH_CALUDE_negative_half_times_negative_two_l2502_250255

theorem negative_half_times_negative_two : (-1/2 : ℚ) * (-2 : ℚ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_times_negative_two_l2502_250255


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2502_250218

theorem arithmetic_calculation : 2535 + 240 / 30 - 435 = 2108 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2502_250218


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2502_250293

theorem trigonometric_equation_solution :
  ∀ x : ℝ, ((7/2 * Real.cos (2*x) + 2) * abs (2 * Real.cos (2*x) - 1) = 
            Real.cos x * (Real.cos x + Real.cos (5*x))) ↔
           (∃ k : ℤ, x = π/6 + k*π/2 ∨ x = -π/6 + k*π/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2502_250293


namespace NUMINAMATH_CALUDE_coordinates_of_point_B_l2502_250274

def point := ℝ × ℝ

theorem coordinates_of_point_B 
  (A B : point) 
  (length_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5)
  (parallel_to_x : A.2 = B.2)
  (coord_A : A = (-1, 3)) :
  B = (-6, 3) ∨ B = (4, 3) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_B_l2502_250274


namespace NUMINAMATH_CALUDE_art_probability_correct_l2502_250251

def art_arrangement_probability (total : ℕ) (escher : ℕ) (picasso : ℕ) : ℚ :=
  let other := total - escher - picasso
  let grouped_items := other + 2  -- other items + Escher block + Picasso block
  (grouped_items.factorial * escher.factorial * picasso.factorial : ℚ) / total.factorial

theorem art_probability_correct :
  art_arrangement_probability 12 4 3 = 1 / 660 := by
  sorry

end NUMINAMATH_CALUDE_art_probability_correct_l2502_250251


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2502_250211

theorem line_segment_endpoint (x : ℝ) : 
  (∃ (y : ℝ), (x - 3)^2 + (y - (-1))^2 = 15^2 ∧ 
               y = 7 ∧ 
               (y - (-1)) / (x - 3) = 1) →
  (x - 3)^2 + 64 = 225 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2502_250211


namespace NUMINAMATH_CALUDE_popsicle_sticks_given_away_l2502_250288

/-- Given that Gino initially had 63.0 popsicle sticks and now has 13 left,
    prove that he gave away 50 popsicle sticks. -/
theorem popsicle_sticks_given_away 
  (initial_sticks : ℝ) 
  (remaining_sticks : ℕ) 
  (h1 : initial_sticks = 63.0)
  (h2 : remaining_sticks = 13) :
  initial_sticks - remaining_sticks = 50 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_given_away_l2502_250288


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l2502_250273

/-- A rectangular prism that is not a cube -/
structure RectangularPrism where
  /-- The number of faces of the rectangular prism -/
  faces : ℕ
  /-- Each face is a rectangle -/
  faces_are_rectangles : True
  /-- The number of diagonals in each rectangular face -/
  diagonals_per_face : ℕ
  /-- The number of space diagonals in the rectangular prism -/
  space_diagonals : ℕ
  /-- The rectangular prism has exactly 6 faces -/
  face_count : faces = 6
  /-- Each rectangular face has exactly 2 diagonals -/
  face_diagonal_count : diagonals_per_face = 2
  /-- The rectangular prism has exactly 4 space diagonals -/
  space_diagonal_count : space_diagonals = 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (rp : RectangularPrism) : ℕ :=
  rp.faces * rp.diagonals_per_face + rp.space_diagonals

/-- Theorem: A rectangular prism (not a cube) has 16 diagonals -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) : total_diagonals rp = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l2502_250273


namespace NUMINAMATH_CALUDE_time_for_600_parts_l2502_250242

/-- Linear regression equation relating parts processed to time spent -/
def linear_regression (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem time_for_600_parts : linear_regression 600 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_time_for_600_parts_l2502_250242


namespace NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l2502_250246

theorem sqrt_seven_less_than_three : Real.sqrt 7 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_less_than_three_l2502_250246


namespace NUMINAMATH_CALUDE_number_of_lists_18_4_l2502_250279

/-- The number of elements in the set of balls -/
def n : ℕ := 18

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from a set of n elements -/
def number_of_lists (n k : ℕ) : ℕ := n^k

/-- Theorem: The number of possible lists when drawing 4 times with replacement from a set of 18 elements is 104,976 -/
theorem number_of_lists_18_4 : number_of_lists n k = 104976 := by
  sorry

end NUMINAMATH_CALUDE_number_of_lists_18_4_l2502_250279


namespace NUMINAMATH_CALUDE_quadrant_crossing_linear_function_y_intercept_positive_l2502_250295

/-- A linear function passing through the first, second, and third quadrants -/
structure QuadrantCrossingLinearFunction where
  b : ℝ
  passes_first_quadrant : ∃ x y, x > 0 ∧ y > 0 ∧ y = x + b
  passes_second_quadrant : ∃ x y, x < 0 ∧ y > 0 ∧ y = x + b
  passes_third_quadrant : ∃ x y, x < 0 ∧ y < 0 ∧ y = x + b

/-- The y-intercept of a quadrant crossing linear function is positive -/
theorem quadrant_crossing_linear_function_y_intercept_positive
  (f : QuadrantCrossingLinearFunction) : f.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_crossing_linear_function_y_intercept_positive_l2502_250295


namespace NUMINAMATH_CALUDE_area_of_union_rectangle_circle_l2502_250226

def rectangle_width : ℝ := 8
def rectangle_height : ℝ := 12
def circle_radius : ℝ := 10

theorem area_of_union_rectangle_circle :
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius^2
  let overlap_area := (π * circle_radius^2) / 4
  rectangle_area + circle_area - overlap_area = 96 + 75 * π := by
sorry

end NUMINAMATH_CALUDE_area_of_union_rectangle_circle_l2502_250226


namespace NUMINAMATH_CALUDE_g_properties_l2502_250290

-- Define g as a function from real numbers to real numbers
variable (g : ℝ → ℝ)

-- Define the properties of g
axiom g_positive : ∀ x, g x > 0
axiom g_sum_property : ∀ a b, g a + g b = g (a + b + 1)

-- State the theorem
theorem g_properties :
  (∃ k : ℝ, k > 0 ∧ g 0 = k) ∧
  (∃ a : ℝ, g (-a) ≠ 1 - g a) :=
sorry

end NUMINAMATH_CALUDE_g_properties_l2502_250290


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l2502_250264

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (A B C : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = 180 →
  B = 2 * A →
  C = 3 * A →
  C = 90 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l2502_250264


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_AE_squared_l2502_250294

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E on AC -/
  E : ℝ × ℝ
  /-- AB is parallel to CD -/
  parallel_AB_CD : (B.1 - A.1) * (D.2 - C.2) = (B.2 - A.2) * (D.1 - C.1)
  /-- Length of AB is 6 -/
  AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6
  /-- Length of CD is 14 -/
  CD_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 14
  /-- ∠AEC is a right angle -/
  AEC_right_angle : (E.1 - A.1) * (E.1 - C.1) + (E.2 - A.2) * (E.2 - C.2) = 0
  /-- CE = CB -/
  CE_eq_CB : (E.1 - C.1)^2 + (E.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- The theorem to be proved -/
theorem isosceles_trapezoid_AE_squared (t : IsoscelesTrapezoid) :
  (t.E.1 - t.A.1)^2 + (t.E.2 - t.A.2)^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_AE_squared_l2502_250294


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l2502_250225

theorem sum_of_solutions_quadratic (a b c : ℚ) (h : a ≠ 0) :
  let eq := fun x => a * x^2 + b * x + c
  (∀ x, eq x = 0 → x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ 
                    x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) + (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a) = -b / a :=
by sorry

theorem sum_of_solutions_specific_quadratic :
  let eq := fun x : ℚ => -48 * x^2 + 100 * x + 200
  (∀ x, eq x = 0 → x = (-100 + Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48)) ∨ 
                    x = (-100 - Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48))) →
  (-100 + Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48)) + 
  (-100 - Real.sqrt (100^2 - 4*(-48)*200)) / (2*(-48)) = 25 / 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l2502_250225


namespace NUMINAMATH_CALUDE_max_colors_theorem_l2502_250243

/-- Represents a color configuration of an n × n × n cube -/
structure ColorConfig (n : ℕ) where
  colors : Fin n → Fin n → Fin n → ℕ

/-- Represents a set of colors in an n × n × 1 box -/
def ColorSet (n : ℕ) := Set ℕ

/-- Returns the set of colors in an n × n × 1 box for a given configuration and orientation -/
def getColorSet (n : ℕ) (config : ColorConfig n) (orientation : Fin 3) (i : Fin n) : ColorSet n :=
  sorry

/-- Checks if the color configuration satisfies the problem conditions -/
def validConfig (n : ℕ) (config : ColorConfig n) : Prop :=
  ∀ (o1 o2 o3 : Fin 3) (i j : Fin n),
    o1 ≠ o2 ∧ o2 ≠ o3 ∧ o1 ≠ o3 →
    ∃ (k l : Fin n), 
      getColorSet n config o1 i = getColorSet n config o2 k ∧
      getColorSet n config o1 i = getColorSet n config o3 l

/-- The maximal number of colors in a valid configuration -/
def maxColors (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem max_colors_theorem (n : ℕ) (h : n > 1) :
  ∃ (config : ColorConfig n),
    validConfig n config ∧
    (∀ (config' : ColorConfig n), validConfig n config' →
      Finset.card (Finset.image (config.colors) Finset.univ) ≥
      Finset.card (Finset.image (config'.colors) Finset.univ)) ∧
    Finset.card (Finset.image (config.colors) Finset.univ) = maxColors n :=
  sorry

end NUMINAMATH_CALUDE_max_colors_theorem_l2502_250243


namespace NUMINAMATH_CALUDE_ab_max_and_4a2_b2_min_l2502_250220

theorem ab_max_and_4a2_b2_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 4 * a^2 + b^2 ≤ 4 * x^2 + y^2) ∧
  a * b = 1/8 ∧
  4 * a^2 + b^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ab_max_and_4a2_b2_min_l2502_250220


namespace NUMINAMATH_CALUDE_raisins_sum_l2502_250296

-- Define the amounts of yellow and black raisins
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- Define the total amount of raisins
def total_raisins : ℝ := yellow_raisins + black_raisins

-- Theorem statement
theorem raisins_sum : total_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_raisins_sum_l2502_250296


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l2502_250207

/-- Given a function f and its derivative f', g is defined as their sum -/
def g (f : ℝ → ℝ) (f' : ℝ → ℝ) : ℝ → ℝ := λ x => f x + f' x

/-- f is a cubic function with parameters a and b -/
def f (a b : ℝ) : ℝ → ℝ := λ x => a * x^3 + x^2 + b * x

/-- f' is the derivative of f -/
def f' (a b : ℝ) : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * x + b

theorem cubic_function_theorem (a b : ℝ) :
  (∀ x, g (f a b) (f' a b) (-x) = -(g (f a b) (f' a b) x)) →
  (f a b = λ x => -1/3 * x^3 + x^2) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, g (f a b) (f' a b) y ≤ g (f a b) (f' a b) x) ∧
  (g (f a b) (f' a b) x = 5/3) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, g (f a b) (f' a b) x ≤ g (f a b) (f' a b) y) ∧
  (g (f a b) (f' a b) x = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l2502_250207
