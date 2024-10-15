import Mathlib

namespace NUMINAMATH_CALUDE_average_hours_worked_l2532_253293

def april_hours : ℕ := 6
def june_hours : ℕ := 5
def september_hours : ℕ := 8
def days_per_month : ℕ := 30
def num_months : ℕ := 3

def total_hours : ℕ := april_hours * days_per_month + june_hours * days_per_month + september_hours * days_per_month

theorem average_hours_worked (h : total_hours = april_hours * days_per_month + june_hours * days_per_month + september_hours * days_per_month) : 
  total_hours / num_months = 190 := by
  sorry

end NUMINAMATH_CALUDE_average_hours_worked_l2532_253293


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l2532_253270

theorem divisibility_by_twelve (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l2532_253270


namespace NUMINAMATH_CALUDE_product_equivalence_l2532_253267

theorem product_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 := by
  sorry

end NUMINAMATH_CALUDE_product_equivalence_l2532_253267


namespace NUMINAMATH_CALUDE_square_side_length_l2532_253235

theorem square_side_length (x : ℝ) (triangle_side : ℝ) (square_side : ℝ) : 
  triangle_side = 2 * x →
  4 * square_side = 3 * triangle_side →
  x = 4 →
  square_side = 6 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2532_253235


namespace NUMINAMATH_CALUDE_phd_time_ratio_l2532_253256

/-- Represents the time spent in years for each phase of John's PhD journey -/
structure PhDTime where
  total : ℝ
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- Theorem stating the ratio of dissertation writing time to acclimation time -/
theorem phd_time_ratio (t : PhDTime) : 
  t.total = 7 ∧ 
  t.acclimation = 1 ∧ 
  t.basics = 2 ∧ 
  t.research = t.basics * 1.75 ∧
  t.total = t.acclimation + t.basics + t.research + t.dissertation →
  t.dissertation / t.acclimation = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_phd_time_ratio_l2532_253256


namespace NUMINAMATH_CALUDE_joshua_borrowed_amount_l2532_253241

/-- The cost of the pen in dollars -/
def pen_cost : ℚ := 6

/-- The amount Joshua has in dollars -/
def joshua_has : ℚ := 5

/-- The additional amount Joshua needs in cents -/
def additional_cents : ℚ := 32 / 100

/-- The amount Joshua borrowed in cents -/
def borrowed_amount : ℚ := 132 / 100

theorem joshua_borrowed_amount :
  borrowed_amount = (pen_cost - joshua_has) * 100 + additional_cents := by
  sorry

end NUMINAMATH_CALUDE_joshua_borrowed_amount_l2532_253241


namespace NUMINAMATH_CALUDE_function_always_positive_l2532_253201

theorem function_always_positive (x : ℝ) : 
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ 
  (x < 1 ∨ x > 3) := by
sorry

end NUMINAMATH_CALUDE_function_always_positive_l2532_253201


namespace NUMINAMATH_CALUDE_book_original_price_l2532_253226

theorem book_original_price (discounted_price original_price : ℝ) : 
  discounted_price = 5 → 
  discounted_price = (1 / 10) * original_price → 
  original_price = 50 := by
sorry

end NUMINAMATH_CALUDE_book_original_price_l2532_253226


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2532_253297

/-- Proves that the interest rate for the first part of an investment is 3% given the specified conditions --/
theorem investment_interest_rate : 
  ∀ (total_amount first_part second_part first_rate second_rate total_interest : ℚ),
  total_amount = 4000 →
  first_part = 2800 →
  second_part = total_amount - first_part →
  second_rate = 5 →
  (first_part * first_rate / 100 + second_part * second_rate / 100) = total_interest →
  total_interest = 144 →
  first_rate = 3 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2532_253297


namespace NUMINAMATH_CALUDE_contrapositive_example_l2532_253249

theorem contrapositive_example (a b : ℝ) :
  (∀ a b, a > b → a - 5 > b - 5) ↔ (∀ a b, a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l2532_253249


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2532_253237

/-- Given vectors a and b, function f, and triangle ABC, prove the perimeter range -/
theorem triangle_perimeter_range (x : ℝ) :
  let a : ℝ × ℝ := (1, Real.sin x)
  let b : ℝ × ℝ := (Real.cos (2*x + π/3), Real.sin x)
  let f : ℝ → ℝ := λ x => a.1 * b.1 + a.2 * b.2 - (1/2) * Real.cos (2*x)
  let c : ℝ := Real.sqrt 3
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = π ∧
    f C = 0 ∧
    2 * Real.sqrt 3 < A + B + c ∧ A + B + c ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2532_253237


namespace NUMINAMATH_CALUDE_yuna_and_friends_count_l2532_253277

/-- Given a line of people where Yuna is 4th from the front and 6th from the back,
    the total number of people in the line is 9. -/
theorem yuna_and_friends_count (people : ℕ) (yuna_position_front yuna_position_back : ℕ) :
  yuna_position_front = 4 →
  yuna_position_back = 6 →
  people = 9 :=
by sorry

end NUMINAMATH_CALUDE_yuna_and_friends_count_l2532_253277


namespace NUMINAMATH_CALUDE_diving_class_capacity_is_270_l2532_253254

/-- The number of people who can take diving classes in 3 weeks -/
def diving_class_capacity : ℕ :=
  let weekday_classes_per_day : ℕ := 2
  let weekend_classes_per_day : ℕ := 4
  let weekdays_per_week : ℕ := 5
  let weekend_days_per_week : ℕ := 2
  let people_per_class : ℕ := 5
  let weeks : ℕ := 3

  let weekday_classes_per_week : ℕ := weekday_classes_per_day * weekdays_per_week
  let weekend_classes_per_week : ℕ := weekend_classes_per_day * weekend_days_per_week
  let total_classes_per_week : ℕ := weekday_classes_per_week + weekend_classes_per_week
  let people_per_week : ℕ := total_classes_per_week * people_per_class
  
  people_per_week * weeks

/-- Theorem stating that the diving class capacity for 3 weeks is 270 people -/
theorem diving_class_capacity_is_270 : diving_class_capacity = 270 := by
  sorry

end NUMINAMATH_CALUDE_diving_class_capacity_is_270_l2532_253254


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l2532_253214

/-- A sequence of complex numbers -/
def ComplexSequence := ℕ → ℂ

/-- Predicate to check if a natural number is prime -/
def IsPrime (p : ℕ) : Prop := sorry

/-- Predicate to check if a series converges -/
def Converges (s : ℕ → ℂ) : Prop := sorry

/-- The main theorem -/
theorem existence_of_special_sequence :
  ∃ (a : ComplexSequence), ∀ (p : ℕ), p > 0 →
    (Converges (fun n => (a n)^p) ↔ ¬(IsPrime p)) := by sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l2532_253214


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2532_253228

theorem unique_solution_quadratic :
  ∃! (q : ℝ), q ≠ 0 ∧ (∃! x, q * x^2 - 8 * x + 16 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2532_253228


namespace NUMINAMATH_CALUDE_right_pyramid_surface_area_l2532_253279

/-- Represents a right pyramid with a parallelogram base -/
structure RightPyramid where
  base_side1 : ℝ
  base_side2 : ℝ
  base_angle : ℝ
  height : ℝ

/-- Calculates the total surface area of a right pyramid -/
def total_surface_area (p : RightPyramid) : ℝ :=
  sorry

theorem right_pyramid_surface_area :
  let p := RightPyramid.mk 12 14 (π / 3) 15
  total_surface_area p = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_right_pyramid_surface_area_l2532_253279


namespace NUMINAMATH_CALUDE_multiply_ones_seven_l2532_253285

theorem multiply_ones_seven : 1111111 * 1111111 = 1234567654321 := by
  sorry

end NUMINAMATH_CALUDE_multiply_ones_seven_l2532_253285


namespace NUMINAMATH_CALUDE_probability_of_six_on_fifth_roll_l2532_253218

def fair_die_prob : ℚ := 1 / 6
def biased_die_6_prob : ℚ := 2 / 3
def biased_die_6_other_prob : ℚ := 1 / 15
def biased_die_3_prob : ℚ := 1 / 2
def biased_die_3_other_prob : ℚ := 1 / 10

def initial_pick_prob : ℚ := 1 / 3

def observed_rolls : ℕ := 4
def observed_sixes : ℕ := 3
def observed_threes : ℕ := 1

theorem probability_of_six_on_fifth_roll :
  let fair_prob := initial_pick_prob * (fair_die_prob ^ observed_sixes * fair_die_prob ^ observed_threes)
  let biased_6_prob := initial_pick_prob * (biased_die_6_prob ^ observed_sixes * biased_die_6_other_prob ^ observed_threes)
  let biased_3_prob := initial_pick_prob * (biased_die_3_other_prob ^ observed_sixes * biased_die_3_prob ^ observed_threes)
  let total_prob := fair_prob + biased_6_prob + biased_3_prob
  (biased_6_prob / total_prob) * biased_die_6_prob = 8 / 135 / (3457.65 / 3888) * (2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_probability_of_six_on_fifth_roll_l2532_253218


namespace NUMINAMATH_CALUDE_linear_function_composition_l2532_253248

/-- Given f(x) = x^2 - 2x + 1 and g(x) is a linear function such that f[g(x)] = 4x^2,
    prove that g(x) = 2x + 1 or g(x) = -2x + 1 -/
theorem linear_function_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2 - 2*x + 1) →
  (∃ a b : ℝ, ∀ x, g x = a*x + b) →
  (∀ x, f (g x) = 4 * x^2) →
  (∀ x, g x = 2*x + 1 ∨ g x = -2*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2532_253248


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2532_253210

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I)^2 = Complex.abs (1 + Complex.I)^2 → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2532_253210


namespace NUMINAMATH_CALUDE_circle_and_line_equations_l2532_253255

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 1 ∨ y = -x + 1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + y + 2 * Real.sqrt 2 = 0

theorem circle_and_line_equations :
  ∃ (a : ℝ),
    a ≤ 0 ∧
    circle_M 0 (-2) ∧
    (∃ (x y : ℝ), circle_M x y ∧ tangent_line x y) ∧
    line_l 0 1 ∧
    (∃ (A B : ℝ × ℝ),
      circle_M A.1 A.2 ∧
      circle_M B.1 B.2 ∧
      line_l A.1 A.2 ∧
      line_l B.1 B.2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_and_line_equations_l2532_253255


namespace NUMINAMATH_CALUDE_largest_common_divisor_540_315_l2532_253280

theorem largest_common_divisor_540_315 : Nat.gcd 540 315 = 45 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_540_315_l2532_253280


namespace NUMINAMATH_CALUDE_min_h25_for_tenuous_min_sum_l2532_253204

/-- A function h : ℕ → ℤ is tenuous if h(x) + h(y) > 2 * y^2 for all positive integers x and y. -/
def Tenuous (h : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → h x + h y > 2 * y^2

/-- The sum of h(1) to h(30) for a function h : ℕ → ℤ. -/
def SumH30 (h : ℕ → ℤ) : ℤ :=
  (Finset.range 30).sum (λ i => h (i + 1))

theorem min_h25_for_tenuous_min_sum (h : ℕ → ℤ) :
  Tenuous h → (∀ g : ℕ → ℤ, Tenuous g → SumH30 h ≤ SumH30 g) → h 25 ≥ 1189 := by
  sorry

end NUMINAMATH_CALUDE_min_h25_for_tenuous_min_sum_l2532_253204


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2532_253299

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y - z = 0 →
  4 * x + 2 * k * y + 3 * z = 0 →
  3 * x + 6 * y + 2 * z = 0 →
  x * z / (y^2) = 1368 / 25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2532_253299


namespace NUMINAMATH_CALUDE_combination_ratio_l2532_253274

def combination (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem combination_ratio :
  (combination 5 2) / (combination 7 3) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_combination_ratio_l2532_253274


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l2532_253276

/-- The total surface area of a pyramid formed from a cube --/
theorem pyramid_surface_area (a : ℝ) (h : a > 0) : 
  let cube_edge := a
  let base_side := a * Real.sqrt 2 / 2
  let slant_height := 3 * a * Real.sqrt 2 / 4
  let lateral_area := 4 * (base_side * slant_height / 2)
  let base_area := base_side ^ 2
  lateral_area + base_area = 2 * a ^ 2 := by
  sorry

#check pyramid_surface_area

end NUMINAMATH_CALUDE_pyramid_surface_area_l2532_253276


namespace NUMINAMATH_CALUDE_second_number_equality_l2532_253219

theorem second_number_equality : ∃ x : ℤ, (9548 + x = 3362 + 13500) ∧ (x = 7314) := by
  sorry

end NUMINAMATH_CALUDE_second_number_equality_l2532_253219


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l2532_253298

/-- Represents a chess tournament between two schools --/
structure ChessTournament where
  /-- Number of players in each school --/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent from the other school --/
  games_per_opponent : Nat
  /-- Number of games played simultaneously in each round --/
  games_per_round : Nat

/-- Calculates the total number of games in the tournament --/
def totalGames (t : ChessTournament) : Nat :=
  t.players_per_school * t.players_per_school * t.games_per_opponent

/-- Calculates the number of rounds in the tournament --/
def numberOfRounds (t : ChessTournament) : Nat :=
  totalGames t / t.games_per_round

/-- Theorem stating the number of ways to schedule the tournament --/
theorem chess_tournament_schedules (t : ChessTournament) 
  (h1 : t.players_per_school = 4)
  (h2 : t.games_per_opponent = 2)
  (h3 : t.games_per_round = 4) :
  Nat.factorial (numberOfRounds t) = 40320 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_schedules_l2532_253298


namespace NUMINAMATH_CALUDE_divisor_problem_l2532_253240

theorem divisor_problem (original : ℕ) (added : ℕ) (divisor : ℕ) : 
  original = 821562 →
  added = 6 →
  (original + added) % divisor = 0 →
  ∀ d : ℕ, d < added → (original + d) % divisor ≠ 0 →
  divisor = 6 :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l2532_253240


namespace NUMINAMATH_CALUDE_trapezoid_area_l2532_253268

/-- Given four identical trapezoids that form a square, prove the area of each trapezoid --/
theorem trapezoid_area (base_small : ℝ) (base_large : ℝ) (square_area : ℝ) :
  base_small = 30 →
  base_large = 50 →
  square_area = 2500 →
  (∃ (trapezoid_area : ℝ), 
    trapezoid_area = (square_area - base_small ^ 2) / 4 ∧ 
    trapezoid_area = 400) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2532_253268


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l2532_253252

theorem cube_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (10 * x - 2) ^ (1/3) + (20 * x + 3) ^ (1/3) - 5 * x ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -1/25 ∨ x = 1/375 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l2532_253252


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2532_253245

/-- Given that ax^2 + bx + c can be expressed as 4(x - 5)^2 + 16, prove that when 5ax^2 + 5bx + 5c 
    is expressed in the form n(x - h)^2 + k, the value of h is 5. -/
theorem quadratic_transformation (a b c : ℝ) 
    (h : ∀ x, a * x^2 + b * x + c = 4 * (x - 5)^2 + 16) :
    ∃ n k, ∀ x, 5 * a * x^2 + 5 * b * x + 5 * c = n * (x - 5)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2532_253245


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l2532_253247

def quadratic_inequality_A (x : ℝ) := x^2 - 12*x + 20 > 0

def quadratic_inequality_B (x : ℝ) := x^2 - 5*x + 6 < 0

def quadratic_inequality_C (x : ℝ) := 9*x^2 - 6*x + 1 > 0

def quadratic_inequality_D (x : ℝ) := -2*x^2 + 2*x - 3 > 0

theorem quadratic_inequalities :
  (∀ x, quadratic_inequality_A x ↔ (x < 2 ∨ x > 10)) ∧
  (∀ x, quadratic_inequality_B x ↔ (2 < x ∧ x < 3)) ∧
  (∃ x, ¬quadratic_inequality_C x) ∧
  (∀ x, ¬quadratic_inequality_D x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l2532_253247


namespace NUMINAMATH_CALUDE_athlete_distance_l2532_253221

/-- Proves that an athlete running at 28.8 km/h for 25 seconds covers a distance of 200 meters. -/
theorem athlete_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 28.8 → time = 25 → distance = speed * time * 1000 / 3600 → distance = 200 := by
  sorry

end NUMINAMATH_CALUDE_athlete_distance_l2532_253221


namespace NUMINAMATH_CALUDE_mask_production_rates_l2532_253260

/-- Represents the daily production rate of masks in millions before equipment change -/
def initial_rate : ℝ := 40

/-- Represents the daily production rate of masks in millions after equipment change -/
def final_rate : ℝ := 56

/-- Represents the number of masks left to produce in millions -/
def remaining_masks : ℝ := 280

/-- Represents the increase in production efficiency as a decimal -/
def efficiency_increase : ℝ := 0.4

/-- Represents the number of days saved due to equipment change -/
def days_saved : ℝ := 2

theorem mask_production_rates :
  (remaining_masks / initial_rate - remaining_masks / (initial_rate * (1 + efficiency_increase)) = days_saved) ∧
  (final_rate = initial_rate * (1 + efficiency_increase)) := by
  sorry

end NUMINAMATH_CALUDE_mask_production_rates_l2532_253260


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l2532_253243

structure Friends where
  Andrew : Prop
  Boris : Prop
  Vasya : Prop
  Gena : Prop
  Denis : Prop

def consistent (f : Friends) : Prop :=
  (f.Andrew → (f.Boris ∧ ¬f.Vasya)) ∧
  (f.Boris → (f.Gena ∨ f.Denis)) ∧
  (¬f.Vasya → (¬f.Boris ∧ ¬f.Denis)) ∧
  (¬f.Andrew → (f.Boris ∧ ¬f.Gena))

theorem hide_and_seek_players :
  ∀ f : Friends, consistent f → (f.Boris ∧ f.Vasya ∧ f.Denis ∧ ¬f.Andrew ∧ ¬f.Gena) :=
by sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l2532_253243


namespace NUMINAMATH_CALUDE_percentage_of_men_in_class_l2532_253239

theorem percentage_of_men_in_class (
  women_science_major_percentage : Real)
  (non_science_major_percentage : Real)
  (men_science_major_percentage : Real)
  (h1 : women_science_major_percentage = 0.1)
  (h2 : non_science_major_percentage = 0.6)
  (h3 : men_science_major_percentage = 0.8500000000000001)
  : Real :=
by
  sorry

#check percentage_of_men_in_class

end NUMINAMATH_CALUDE_percentage_of_men_in_class_l2532_253239


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2532_253281

theorem absolute_value_equality (x : ℝ) : |x + 6| = -(x + 6) ↔ x ≤ -6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2532_253281


namespace NUMINAMATH_CALUDE_angle_sum_in_special_pentagon_l2532_253275

theorem angle_sum_in_special_pentagon (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x < 180) 
  (h2 : 0 ≤ y ∧ y < 180) : 
  x + y = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_special_pentagon_l2532_253275


namespace NUMINAMATH_CALUDE_trig_identity_l2532_253242

theorem trig_identity : (Real.cos (10 * π / 180) - 2 * Real.sin (20 * π / 180)) / Real.sin (10 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2532_253242


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2532_253233

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_of_geometric_sequence (a : ℕ → ℝ) (y : ℝ) :
  geometric_sequence a (3 * y) →
  a 0 = 3 →
  a 1 = 9 * y →
  a 2 = 27 * y^2 →
  a 3 = 81 * y^3 →
  a 4 = 243 * y^4 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2532_253233


namespace NUMINAMATH_CALUDE_min_value_expression_l2532_253238

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ y₀ / x₀ + 4 / y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2532_253238


namespace NUMINAMATH_CALUDE_sequence_limit_zero_l2532_253203

/-- Given an infinite sequence {a_n} where the limit of (a_{n+1} - a_n/2) as n approaches infinity is 0,
    prove that the limit of a_n as n approaches infinity is 0. -/
theorem sequence_limit_zero
  (a : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |a (n + 1) - a n / 2| < ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n| < ε :=
sorry

end NUMINAMATH_CALUDE_sequence_limit_zero_l2532_253203


namespace NUMINAMATH_CALUDE_pool_surface_area_l2532_253200

/-- A rectangular swimming pool with given dimensions. -/
structure RectangularPool where
  length : ℝ
  width : ℝ

/-- Calculate the surface area of a rectangular pool. -/
def surfaceArea (pool : RectangularPool) : ℝ :=
  pool.length * pool.width

/-- Theorem: The surface area of a rectangular pool with length 20 meters and width 15 meters is 300 square meters. -/
theorem pool_surface_area :
  let pool : RectangularPool := { length := 20, width := 15 }
  surfaceArea pool = 300 := by
  sorry

end NUMINAMATH_CALUDE_pool_surface_area_l2532_253200


namespace NUMINAMATH_CALUDE_marble_capacity_l2532_253271

/-- Given a container of volume 24 cm³ holding 75 marbles, 
    prove that a container of volume 72 cm³ will hold 225 marbles, 
    assuming a linear relationship between volume and marble capacity. -/
theorem marble_capacity 
  (volume_small : ℝ) 
  (marbles_small : ℕ) 
  (volume_large : ℝ) 
  (h1 : volume_small = 24) 
  (h2 : marbles_small = 75) 
  (h3 : volume_large = 72) : 
  (volume_large / volume_small) * marbles_small = 225 := by
sorry

end NUMINAMATH_CALUDE_marble_capacity_l2532_253271


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l2532_253216

theorem min_value_quadratic_sum :
  ∀ x y : ℝ, (2*x - y + 3)^2 + (x + 2*y - 1)^2 ≥ 295/72 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l2532_253216


namespace NUMINAMATH_CALUDE_remainder_comparison_l2532_253207

theorem remainder_comparison (P P' : ℕ) (h1 : P = P' + 10) (h2 : P % 10 = 0) (h3 : P' % 10 = 0) :
  (P^2 - P'^2) % 10 = 0 ∧ 0 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_comparison_l2532_253207


namespace NUMINAMATH_CALUDE_work_left_fraction_l2532_253296

/-- The fraction of work left after two workers work together for a given number of days -/
def fraction_left (days_a : ℕ) (days_b : ℕ) (days_together : ℕ) : ℚ :=
  1 - (days_together : ℚ) * ((1 : ℚ) / days_a + (1 : ℚ) / days_b)

/-- Theorem: Given A can do a job in 15 days and B in 20 days, if they work together for 3 days, 
    the fraction of work left is 13/20 -/
theorem work_left_fraction : fraction_left 15 20 3 = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l2532_253296


namespace NUMINAMATH_CALUDE_positive_solution_quadratic_equation_l2532_253257

theorem positive_solution_quadratic_equation :
  ∃ x : ℝ, x > 0 ∧ 
  (1/3) * (4 * x^2 - 2) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10) ∧
  x = (75 + Real.sqrt 5693) / 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_quadratic_equation_l2532_253257


namespace NUMINAMATH_CALUDE_lcm_problem_l2532_253289

theorem lcm_problem (e n : ℕ) : 
  e > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm e n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ e) →
  n = 230 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l2532_253289


namespace NUMINAMATH_CALUDE_jade_transactions_l2532_253224

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel / 10) →
  cal = (2 * anthony) / 3 →
  jade = cal + 15 →
  jade = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l2532_253224


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2532_253290

theorem inequality_equivalence (x : ℝ) :
  abs (2 * x - 1) + abs (x + 1) ≥ x + 2 ↔ x ≤ 0 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2532_253290


namespace NUMINAMATH_CALUDE_same_problem_different_algorithms_l2532_253213

-- Define the characteristics of algorithms
structure AlgorithmCharacteristics where
  finiteness : Bool
  determinacy : Bool
  sequentiality : Bool
  correctness : Bool
  nonUniqueness : Bool
  universality : Bool

-- Define the possible representations of algorithms
inductive AlgorithmRepresentation
  | NaturalLanguage
  | GraphicalLanguage
  | ProgrammingLanguage

-- Define a problem that can be solved by algorithms
structure Problem where
  description : String

-- Define an algorithm
structure Algorithm where
  steps : List String
  representation : AlgorithmRepresentation

-- Theorem: The same problem can have different algorithms
theorem same_problem_different_algorithms 
  (p : Problem) 
  (chars : AlgorithmCharacteristics) 
  (reprs : List AlgorithmRepresentation) :
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ 
  (∀ (input : String), 
    (a1.steps.foldl (λ acc step => step ++ acc) input) = 
    (a2.steps.foldl (λ acc step => step ++ acc) input)) :=
sorry

end NUMINAMATH_CALUDE_same_problem_different_algorithms_l2532_253213


namespace NUMINAMATH_CALUDE_pink_tulips_count_l2532_253209

theorem pink_tulips_count (total : ℕ) (red_fraction blue_fraction : ℚ) : 
  total = 56 →
  red_fraction = 3 / 7 →
  blue_fraction = 3 / 8 →
  ↑total - (↑total * red_fraction + ↑total * blue_fraction) = 11 := by
  sorry

end NUMINAMATH_CALUDE_pink_tulips_count_l2532_253209


namespace NUMINAMATH_CALUDE_smaller_cube_edge_length_l2532_253208

/-- Given a cube with volume 1000 cm³ divided into 8 equal smaller cubes,
    prove that the edge length of each smaller cube is 5 cm. -/
theorem smaller_cube_edge_length :
  ∀ (original_volume smaller_volume : ℝ) (original_edge smaller_edge : ℝ),
  original_volume = 1000 →
  smaller_volume = original_volume / 8 →
  original_volume = original_edge ^ 3 →
  smaller_volume = smaller_edge ^ 3 →
  smaller_edge = 5 := by
sorry

end NUMINAMATH_CALUDE_smaller_cube_edge_length_l2532_253208


namespace NUMINAMATH_CALUDE_sector_central_angle_l2532_253246

/-- Given a circular sector with arc length 4 and area 2, 
    prove that its central angle is 4 radians. -/
theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (θ : ℝ) :
  arc_length = 4 →
  area = 2 →
  θ = arc_length / (2 * area / arc_length) →
  θ = 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2532_253246


namespace NUMINAMATH_CALUDE_geometric_sequence_a9_l2532_253265

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a9 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = 3 → a 6 = 9 → a 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a9_l2532_253265


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_factor_l2532_253220

/-- A number is a police emergency number if it ends with 133 in decimal system -/
def is_police_emergency_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1000 * k + 133

/-- Every police emergency number has a prime factor greater than 7 -/
theorem police_emergency_number_has_large_prime_factor (n : ℕ) 
  (h : is_police_emergency_number n) : 
  ∃ p : ℕ, p > 7 ∧ Nat.Prime p ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_factor_l2532_253220


namespace NUMINAMATH_CALUDE_original_paper_sheets_l2532_253264

-- Define the number of sheets per book
def sheets_per_book : ℕ := sorry

-- Define the total number of sheets
def total_sheets : ℕ := 18000

-- Theorem statement
theorem original_paper_sheets :
  (120 * sheets_per_book = (60 : ℕ) * total_sheets / 100) ∧
  (185 * sheets_per_book + 1350 = total_sheets) :=
by sorry

end NUMINAMATH_CALUDE_original_paper_sheets_l2532_253264


namespace NUMINAMATH_CALUDE_year_spans_53_or_54_weeks_l2532_253287

/-- A year is either common (365 days) or leap (366 days) -/
inductive Year
  | Common
  | Leap

/-- Definition of how many days are in a year -/
def daysInYear (y : Year) : ℕ :=
  match y with
  | Year.Common => 365
  | Year.Leap => 366

/-- Definition of when a year covers a week -/
def yearCoversWeek (daysInYear : ℕ) (weekStartDay : ℕ) : Prop :=
  daysInYear - weekStartDay ≥ 6

/-- Theorem stating that a year can span either 53 or 54 weeks -/
theorem year_spans_53_or_54_weeks (y : Year) :
  ∃ (n : ℕ), (n = 53 ∨ n = 54) ∧
    (∀ (w : ℕ), w ≤ n → yearCoversWeek (daysInYear y) ((w - 1) * 7)) ∧
    (∀ (w : ℕ), w > n → ¬yearCoversWeek (daysInYear y) ((w - 1) * 7)) :=
  sorry

end NUMINAMATH_CALUDE_year_spans_53_or_54_weeks_l2532_253287


namespace NUMINAMATH_CALUDE_parabola_line_intersection_area_l2532_253211

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

/-- Line structure -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Triangle structure -/
structure Triangle where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- Function to calculate distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem -/
theorem parabola_line_intersection_area 
  (p : Parabola) 
  (l : Line) 
  (A B : ℝ × ℝ) 
  (h1 : p.equation = fun (x, y) ↦ y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.point1 = p.focus)
  (h4 : p.equation A ∧ p.equation B)
  (h5 : distance A p.focus = 3) :
  triangleArea { point1 := (0, 0), point2 := A, point3 := B } = 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_area_l2532_253211


namespace NUMINAMATH_CALUDE_misread_signs_count_l2532_253269

def f (x : ℝ) : ℝ := 10*x^9 + 9*x^8 + 8*x^7 + 7*x^6 + 6*x^5 + 5*x^4 + 4*x^3 + 3*x^2 + 2*x + 1

theorem misread_signs_count :
  let correct_result := f (-1)
  let incorrect_result := 7
  let difference := incorrect_result - correct_result
  difference / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_misread_signs_count_l2532_253269


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2532_253234

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) : 
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2532_253234


namespace NUMINAMATH_CALUDE_green_ball_count_l2532_253278

/-- Represents the number of balls of each color --/
structure BallCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The conditions of the problem --/
def validBallCount (bc : BallCount) : Prop :=
  bc.red + bc.blue + bc.green = 50 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 34 → bc.red > 0 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 35 → bc.blue > 0 ∧
  ∀ (subset : ℕ), subset ≤ 50 → subset ≥ 36 → bc.green > 0

/-- The theorem to be proved --/
theorem green_ball_count (bc : BallCount) (h : validBallCount bc) :
  15 ≤ bc.green ∧ bc.green ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_count_l2532_253278


namespace NUMINAMATH_CALUDE_robot_wear_combinations_l2532_253261

/-- Represents the number of ways to wear items on one arm -/
def waysPerArm : ℕ := 1

/-- Represents the number of arms -/
def numArms : ℕ := 2

/-- Represents the number of ways to order items between arms -/
def waysBetweenArms : ℕ := 1

/-- Calculates the total number of ways to wear all items -/
def totalWays : ℕ := waysPerArm ^ numArms * waysBetweenArms

theorem robot_wear_combinations : totalWays = 4 := by
  sorry

end NUMINAMATH_CALUDE_robot_wear_combinations_l2532_253261


namespace NUMINAMATH_CALUDE_cube_sum_zero_l2532_253288

theorem cube_sum_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = a^4 + b^4 + c^4) :
  a^3 + b^3 + c^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_zero_l2532_253288


namespace NUMINAMATH_CALUDE_octagon_coloring_count_l2532_253205

/-- Represents a coloring of 8 disks arranged in an octagon. -/
structure OctagonColoring where
  blue : Finset (Fin 8)
  red : Finset (Fin 8)
  yellow : Finset (Fin 8)
  partition : Disjoint blue red ∧ Disjoint blue yellow ∧ Disjoint red yellow
  cover : blue ∪ red ∪ yellow = Finset.univ
  blue_count : blue.card = 4
  red_count : red.card = 3
  yellow_count : yellow.card = 1

/-- The group of symmetries of an octagon. -/
def OctagonSymmetry : Type := Unit -- Placeholder, actual implementation would be more complex

/-- Two colorings are equivalent if one can be obtained from the other by a symmetry. -/
def equivalent (c₁ c₂ : OctagonColoring) (sym : OctagonSymmetry) : Prop := sorry

/-- The number of distinct colorings under symmetry. -/
def distinctColorings : ℕ := sorry

/-- The main theorem: There are exactly 26 distinct colorings. -/
theorem octagon_coloring_count : distinctColorings = 26 := by sorry

end NUMINAMATH_CALUDE_octagon_coloring_count_l2532_253205


namespace NUMINAMATH_CALUDE_total_oil_needed_l2532_253286

def oil_for_wheels : ℕ := 2 * 15
def oil_for_chain : ℕ := 10
def oil_for_pedals : ℕ := 5
def oil_for_brakes : ℕ := 8

theorem total_oil_needed : 
  oil_for_wheels + oil_for_chain + oil_for_pedals + oil_for_brakes = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_needed_l2532_253286


namespace NUMINAMATH_CALUDE_proportion_problem_l2532_253225

theorem proportion_problem (x : ℝ) : (18 / 12 = x / (6 * 60)) → x = 540 := by sorry

end NUMINAMATH_CALUDE_proportion_problem_l2532_253225


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2532_253244

theorem division_remainder_proof (dividend : Nat) (divisor : Nat) (quotient : Nat) (h1 : dividend = 729) (h2 : divisor = 38) (h3 : quotient = 19) :
  dividend - divisor * quotient = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2532_253244


namespace NUMINAMATH_CALUDE_question_one_question_two_l2532_253294

-- Define the sets A, B, and M
def A (a : ℝ) : Set ℝ := {x | x^2 + (a - 1) * x - a > 0}
def B (a b : ℝ) : Set ℝ := {x | (x + a) * (x + b) > 0}
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Define the complement of B in ℝ
def C_I_B (a b : ℝ) : Set ℝ := {x | ¬((x + a) * (x + b) > 0)}

-- Theorem for question 1
theorem question_one (a b : ℝ) (h1 : a < b) (h2 : C_I_B a b = M) : 
  a = -1 ∧ b = 3 := by sorry

-- Theorem for question 2
theorem question_two (a b : ℝ) (h : a > b ∧ b > -1) : 
  A a ∩ B a b = {x | x < -a ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_question_one_question_two_l2532_253294


namespace NUMINAMATH_CALUDE_cubic_equation_with_double_root_l2532_253222

/-- Given a cubic equation 2x^3 + 9x^2 - 117x + k = 0 where two roots are equal and k is positive,
    prove that k = 47050/216 -/
theorem cubic_equation_with_double_root (k : ℝ) : 
  (∃ x y : ℝ, (2 * x^3 + 9 * x^2 - 117 * x + k = 0) ∧ 
               (2 * y^3 + 9 * y^2 - 117 * y + k = 0) ∧
               (x ≠ y)) ∧
  (∃ z : ℝ, (2 * z^3 + 9 * z^2 - 117 * z + k = 0) ∧
            (∃ w : ℝ, w ≠ z ∧ 2 * w^3 + 9 * w^2 - 117 * w + k = 0)) ∧
  (k > 0) →
  k = 47050 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_with_double_root_l2532_253222


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2532_253202

/-- The number of rectangles in a 4x4 grid -/
def num_rectangles_4x4 : ℕ := 36

/-- The number of horizontal or vertical lines in a 4x4 grid -/
def grid_lines : ℕ := 4

/-- Theorem: The number of rectangles in a 4x4 grid is 36 -/
theorem rectangles_in_4x4_grid :
  (grid_lines.choose 2) * (grid_lines.choose 2) = num_rectangles_4x4 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2532_253202


namespace NUMINAMATH_CALUDE_min_value_xyz_l2532_253212

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x*y/z + y*z/x + z*x/y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_l2532_253212


namespace NUMINAMATH_CALUDE_bianca_received_30_dollars_l2532_253236

/-- The amount of money Bianca received for her birthday -/
def biancas_birthday_money (num_friends : ℕ) (dollars_per_friend : ℕ) : ℕ :=
  num_friends * dollars_per_friend

/-- Theorem stating that Bianca received 30 dollars for her birthday -/
theorem bianca_received_30_dollars :
  biancas_birthday_money 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bianca_received_30_dollars_l2532_253236


namespace NUMINAMATH_CALUDE_range_of_g_l2532_253291

def f (x : ℝ) : ℝ := 4 * x + 1

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → 85 ≤ g x ∧ g x ≤ 853 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l2532_253291


namespace NUMINAMATH_CALUDE_not_A_necessary_not_sufficient_for_not_B_l2532_253250

-- Define propositions A and B
variable (A B : Prop)

-- Define what it means for A to be sufficient but not necessary for B
def sufficient_not_necessary (A B : Prop) : Prop :=
  (A → B) ∧ ¬(B → A)

-- Theorem statement
theorem not_A_necessary_not_sufficient_for_not_B
  (h : sufficient_not_necessary A B) :
  (¬B → ¬A) ∧ ¬(¬A → ¬B) := by
  sorry

end NUMINAMATH_CALUDE_not_A_necessary_not_sufficient_for_not_B_l2532_253250


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2532_253272

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) : 
  Real.cos (a 2 + a 8) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l2532_253272


namespace NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l2532_253253

theorem four_x_plus_t_is_odd (x t : ℤ) (h : 2 * x - t = 11) : Odd (4 * x + t) := by
  sorry

end NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l2532_253253


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l2532_253215

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) :=
by sorry

theorem equality_condition (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (h₁ : x₁ * y₁ - z₁^2 > 0) (h₂ : x₂ * y₂ - z₂^2 > 0) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
  x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂ :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l2532_253215


namespace NUMINAMATH_CALUDE_minimal_polynomial_with_roots_l2532_253263

/-- The polynomial we're proving is correct -/
def f (x : ℝ) : ℝ := x^4 - 8*x^3 + 14*x^2 + 8*x - 3

/-- A root of a polynomial -/
def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

/-- A polynomial with rational coefficients -/
def has_rational_coeffs (p : ℝ → ℝ) : Prop := 
  ∃ (a b c d e : ℚ), ∀ x, p x = a*x^4 + b*x^3 + c*x^2 + d*x + e

theorem minimal_polynomial_with_roots : 
  (is_root f (2 + Real.sqrt 3)) ∧ 
  (is_root f (2 + Real.sqrt 5)) ∧ 
  (has_rational_coeffs f) ∧
  (∀ g : ℝ → ℝ, has_rational_coeffs g → is_root g (2 + Real.sqrt 3) → 
    is_root g (2 + Real.sqrt 5) → (∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * g x) → 
    (∃ n : ℕ, ∀ x, g x = (f x) * x^n)) := 
sorry

end NUMINAMATH_CALUDE_minimal_polynomial_with_roots_l2532_253263


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2532_253273

theorem arithmetic_sequence_length (a₁ l d : ℕ) (h : l = a₁ + (n - 1) * d) :
  a₁ = 4 → l = 205 → d = 3 → n = 68 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2532_253273


namespace NUMINAMATH_CALUDE_stratified_sampling_second_grade_l2532_253295

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of students -/
def total_students (g : GradeDistribution) : ℕ :=
  g.first + g.second + g.third

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total sample size -/
def total_sample (s : SampleDistribution) : ℕ :=
  s.first + s.second + s.third

/-- Checks if the sample distribution is proportional to the grade distribution -/
def is_proportional_sample (g : GradeDistribution) (s : SampleDistribution) : Prop :=
  g.first * s.second = g.second * s.first ∧
  g.second * s.third = g.third * s.second

theorem stratified_sampling_second_grade
  (g : GradeDistribution)
  (s : SampleDistribution)
  (h1 : total_students g = 2000)
  (h2 : g.first = 5 * g.third)
  (h3 : g.second = 3 * g.third)
  (h4 : total_sample s = 20)
  (h5 : is_proportional_sample g s) :
  s.second = 6 := by
  sorry

#check stratified_sampling_second_grade

end NUMINAMATH_CALUDE_stratified_sampling_second_grade_l2532_253295


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_30_l2532_253251

theorem sum_of_fifth_powers_divisible_by_30 (a b c : ℤ) (h : 30 ∣ (a + b + c)) :
  30 ∣ (a^5 + b^5 + c^5) := by sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_30_l2532_253251


namespace NUMINAMATH_CALUDE_larry_jogging_time_l2532_253217

/-- Represents the number of days Larry jogs in the first week -/
def days_first_week : ℕ := 3

/-- Represents the number of days Larry jogs in the second week -/
def days_second_week : ℕ := 5

/-- Represents the total number of hours Larry jogs in two weeks -/
def total_hours : ℕ := 4

/-- Calculates the total number of days Larry jogs in two weeks -/
def total_days : ℕ := days_first_week + days_second_week

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

/-- Represents the total jogging time in minutes -/
def total_minutes : ℕ := hours_to_minutes total_hours

/-- Theorem: Larry jogs for 30 minutes each day -/
theorem larry_jogging_time :
  total_minutes / total_days = 30 := by sorry

end NUMINAMATH_CALUDE_larry_jogging_time_l2532_253217


namespace NUMINAMATH_CALUDE_number_division_l2532_253230

theorem number_division (x : ℤ) : (x - 39 = 54) → (x / 3 = 31) := by
  sorry

end NUMINAMATH_CALUDE_number_division_l2532_253230


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2532_253292

/-- A circle is tangent to the coordinate axes and the hypotenuse of a 45-45-90 triangle -/
structure TangentCircle where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The circle is tangent to the x-axis -/
  tangent_x : center.2 = radius
  /-- The circle is tangent to the y-axis -/
  tangent_y : center.1 = radius
  /-- The circle is tangent to the hypotenuse of the 45-45-90 triangle -/
  tangent_hypotenuse : center.1 + center.2 + radius = 2 * Real.sqrt 2

/-- The side length of the 45-45-90 triangle -/
def triangleSide : ℝ := 2

/-- The theorem stating that the radius of the tangent circle is √2 -/
theorem tangent_circle_radius :
  ∀ (c : TangentCircle), c.radius = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2532_253292


namespace NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l2532_253229

/-- Calculates the profit per meter of cloth given the total meters sold,
    total selling price, and cost price per meter. -/
def profit_per_meter (total_meters : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℚ :=
  (total_selling_price - total_meters * cost_price_per_meter : ℚ) / total_meters

/-- Proves that for a specific cloth sale, the profit per meter is 7 -/
theorem cloth_sale_profit_per_meter :
  profit_per_meter 80 10000 118 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l2532_253229


namespace NUMINAMATH_CALUDE_boat_journey_distance_l2532_253232

-- Define the constants
def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the theorem
theorem boat_journey_distance :
  ∃ (x : ℝ), 
    (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time ∧ x = 20) ∨
    (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time ∧ x = 20/3) :=
by sorry


end NUMINAMATH_CALUDE_boat_journey_distance_l2532_253232


namespace NUMINAMATH_CALUDE_A_3_2_equals_5_l2532_253259

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 2
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_5 : A 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_A_3_2_equals_5_l2532_253259


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2532_253223

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_twelfths_l2532_253223


namespace NUMINAMATH_CALUDE_sweets_remaining_problem_l2532_253283

/-- The number of sweets remaining in a packet after some are eaten and given away -/
def sweets_remaining (cherry strawberry pineapple : ℕ) : ℕ :=
  let total := cherry + strawberry + pineapple
  let eaten := (cherry / 2) + (strawberry / 2) + (pineapple / 2)
  let given_away := 5
  total - eaten - given_away

/-- Theorem stating the number of sweets remaining in the packet -/
theorem sweets_remaining_problem :
  sweets_remaining 30 40 50 = 55 := by
  sorry

end NUMINAMATH_CALUDE_sweets_remaining_problem_l2532_253283


namespace NUMINAMATH_CALUDE_nested_expression_value_l2532_253282

theorem nested_expression_value : (3*(3*(3*(3*(3*(3*(3+2)+2)+2)+2)+2)+2)+2) = 4373 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l2532_253282


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l2532_253266

/-- Part 1: Non-existence of positive integer sequence --/
theorem no_positive_integer_sequence :
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, (f (n + 1))^2 ≥ 2 * (f n) * (f (n + 2)) :=
sorry

/-- Part 2: Existence of positive irrational sequence --/
theorem exists_positive_irrational_sequence :
  ∃ f : ℕ+ → ℝ, (∀ n : ℕ+, Irrational (f n)) ∧
    (∀ n : ℕ+, f n > 0) ∧
    (∀ n : ℕ+, (f (n + 1))^2 ≥ 2 * (f n) * (f (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l2532_253266


namespace NUMINAMATH_CALUDE_cos_neg_570_deg_l2532_253231

-- Define the cosine function for degrees
noncomputable def cos_deg (x : ℝ) : ℝ := Real.cos (x * Real.pi / 180)

-- State the theorem
theorem cos_neg_570_deg : cos_deg (-570) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_570_deg_l2532_253231


namespace NUMINAMATH_CALUDE_base_number_proof_l2532_253258

theorem base_number_proof (x : ℝ) : 9^7 = x^14 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2532_253258


namespace NUMINAMATH_CALUDE_estate_distribution_l2532_253262

/-- Represents the estate distribution problem --/
theorem estate_distribution (E : ℕ) 
  (h1 : ∃ (d s w c : ℕ), d + s + w + c = E) 
  (h2 : ∃ (d s : ℕ), d + s = E / 2) 
  (h3 : ∃ (d s : ℕ), 3 * s = 2 * d) 
  (h4 : ∃ (d w : ℕ), w = 3 * d) 
  (h5 : ∃ (c : ℕ), c = 800) :
  E = 2000 := by
  sorry

end NUMINAMATH_CALUDE_estate_distribution_l2532_253262


namespace NUMINAMATH_CALUDE_pen_cost_l2532_253227

-- Define the number of pens bought by Robert
def robert_pens : ℕ := 4

-- Define the number of pens bought by Julia in terms of Robert's
def julia_pens : ℕ := 3 * robert_pens

-- Define the number of pens bought by Dorothy in terms of Julia's
def dorothy_pens : ℕ := julia_pens / 2

-- Define the total amount spent
def total_spent : ℚ := 33

-- Define the total number of pens bought
def total_pens : ℕ := dorothy_pens + julia_pens + robert_pens

-- Theorem: The cost of one pen is $1.50
theorem pen_cost : total_spent / total_pens = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l2532_253227


namespace NUMINAMATH_CALUDE_equation_solutions_l2532_253206

-- Define the equation
def equation (a : ℝ) (t : ℝ) : Prop :=
  (4*a*(Real.sin t)^2 + 4*a*(1 + 2*Real.sqrt 2)*Real.cos t - 4*(a - 1)*Real.sin t - 5*a + 2) / 
  (2*Real.sqrt 2*Real.cos t - Real.sin t) = 4*a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {t : ℝ | equation a t ∧ 0 < t ∧ t < Real.pi/2}

-- Define the condition for exactly two distinct solutions
def has_two_distinct_solutions (a : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), t₁ ∈ solution_set a ∧ t₂ ∈ solution_set a ∧ t₁ ≠ t₂ ∧
  ∀ (t : ℝ), t ∈ solution_set a → t = t₁ ∨ t = t₂

-- The main theorem
theorem equation_solutions (a : ℝ) :
  has_two_distinct_solutions a ↔ (a > 6 ∧ a < 18 + 24*Real.sqrt 2) ∨ a > 18 + 24*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2532_253206


namespace NUMINAMATH_CALUDE_max_students_planting_trees_l2532_253284

theorem max_students_planting_trees (a b : ℕ) : 
  3 * a + 5 * b = 115 → a + b ≤ 37 := by
  sorry

end NUMINAMATH_CALUDE_max_students_planting_trees_l2532_253284
