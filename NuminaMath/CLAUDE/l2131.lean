import Mathlib

namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l2131_213111

/-- Given a flagpole and a building under similar conditions, prove the length of the flagpole's shadow. -/
theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 26)
  (h3 : building_shadow = 65)
  : ∃ (flagpole_shadow : ℝ), flagpole_shadow = 45 ∧ 
    flagpole_height / flagpole_shadow = building_height / building_shadow :=
by sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l2131_213111


namespace NUMINAMATH_CALUDE_jackie_exercise_hours_l2131_213198

/-- Represents Jackie's daily schedule --/
structure DailySchedule where
  total_hours : ℕ
  work_hours : ℕ
  sleep_hours : ℕ
  free_hours : ℕ

/-- Calculates the number of hours Jackie spends exercising --/
def exercise_hours (schedule : DailySchedule) : ℕ :=
  schedule.total_hours - (schedule.work_hours + schedule.sleep_hours + schedule.free_hours)

/-- Theorem stating that Jackie spends 3 hours exercising --/
theorem jackie_exercise_hours :
  let schedule : DailySchedule := {
    total_hours := 24,
    work_hours := 8,
    sleep_hours := 8,
    free_hours := 5
  }
  exercise_hours schedule = 3 := by sorry

end NUMINAMATH_CALUDE_jackie_exercise_hours_l2131_213198


namespace NUMINAMATH_CALUDE_square_side_length_l2131_213174

theorem square_side_length (circle_area : ℝ) (square_perimeter : ℝ) : 
  circle_area = 100 → circle_area = square_perimeter → square_perimeter / 4 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2131_213174


namespace NUMINAMATH_CALUDE_total_games_in_season_l2131_213107

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The number of months in the hockey season -/
def months_in_season : ℕ := 14

/-- The total number of hockey games in the season -/
def total_games : ℕ := games_per_month * months_in_season

/-- Theorem stating that the total number of hockey games in the season is 182 -/
theorem total_games_in_season : total_games = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l2131_213107


namespace NUMINAMATH_CALUDE_wheel_probability_l2131_213143

theorem wheel_probability (P_A P_B P_C P_D : ℚ) : 
  P_A = 1/4 → P_B = 1/3 → P_C = 1/6 → P_A + P_B + P_C + P_D = 1 → P_D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l2131_213143


namespace NUMINAMATH_CALUDE_system_solution_l2131_213164

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y + z = 15) ∧ 
  (x^2 + y^2 + z^2 = 81) ∧ 
  (x*y + x*z = 3*y*z) ∧ 
  ((x = 6 ∧ y = 3 ∧ z = 6) ∨ (x = 6 ∧ y = 6 ∧ z = 3)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2131_213164


namespace NUMINAMATH_CALUDE_largest_sum_is_185_l2131_213131

/-- Represents a digit (1-9) -/
def Digit := Fin 9

/-- The sum of two two-digit numbers formed by three digits -/
def sum_XYZ (X Y Z : Digit) : ℕ := 10 * X.val + 11 * Y.val + Z.val

/-- The largest possible sum given the constraints -/
def largest_sum : ℕ := 185

/-- Theorem stating that 185 is the largest possible sum -/
theorem largest_sum_is_185 :
  ∀ X Y Z : Digit,
    X.val > Y.val →
    Y.val > Z.val →
    X ≠ Y →
    Y ≠ Z →
    X ≠ Z →
    sum_XYZ X Y Z ≤ largest_sum :=
sorry

end NUMINAMATH_CALUDE_largest_sum_is_185_l2131_213131


namespace NUMINAMATH_CALUDE_cubic_factorization_l2131_213117

theorem cubic_factorization (x : ℝ) :
  (189 * x^3 + 129 * x^2 + 183 * x + 19 = (4*x - 2)^3 + (5*x + 3)^3) ∧
  (x^3 + 69 * x^2 + 87 * x + 167 = 5*(x + 3)^3 - 4*(x - 2)^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2131_213117


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2131_213139

theorem arithmetic_calculations :
  (-4 - 4 = -8) ∧ ((-32) / 4 = -8) ∧ (-(-2)^3 = 8) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2131_213139


namespace NUMINAMATH_CALUDE_triangle_min_value_l2131_213130

/-- In a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if acosB - bcosA = c/3, then the minimum value of (acosA + bcosB) / (acosB) is √2. -/
theorem triangle_min_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * Real.sin C = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin B →
  a * Real.cos B - b * Real.cos A = c / 3 →
  ∃ (x : ℝ), x = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B) ∧
    x ≥ Real.sqrt 2 ∧
    ∀ (y : ℝ), y = (a * Real.cos A + b * Real.cos B) / (a * Real.cos B) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_triangle_min_value_l2131_213130


namespace NUMINAMATH_CALUDE_min_moves_to_monochrome_l2131_213180

/-- A move on a checkerboard that inverts colors in a rectangle -/
structure Move where
  top_left : Nat × Nat
  bottom_right : Nat × Nat

/-- A checkerboard with m rows and n columns -/
structure Checkerboard (m n : Nat) where
  board : Matrix (Fin m) (Fin n) Bool

/-- The result of applying a move to a checkerboard -/
def apply_move (board : Checkerboard m n) (move : Move) : Checkerboard m n :=
  sorry

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Check if a checkerboard is monochrome -/
def is_monochrome (board : Checkerboard m n) : Prop :=
  sorry

/-- The theorem stating the minimum number of moves required -/
theorem min_moves_to_monochrome (m n : Nat) :
  ∃ (moves : MoveSequence),
    (∀ (board : Checkerboard m n),
      is_monochrome (moves.foldl apply_move board)) ∧
    moves.length = Nat.floor (n / 2) + Nat.floor (m / 2) ∧
    (∀ (other_moves : MoveSequence),
      (∀ (board : Checkerboard m n),
        is_monochrome (other_moves.foldl apply_move board)) →
      other_moves.length ≥ moves.length) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_monochrome_l2131_213180


namespace NUMINAMATH_CALUDE_amusement_park_visitors_l2131_213171

/-- Amusement park visitor count problem -/
theorem amusement_park_visitors :
  let morning_visitors : ℕ := 473
  let noon_departures : ℕ := 179
  let afternoon_visitors : ℕ := 268
  let total_visitors : ℕ := morning_visitors + afternoon_visitors
  let current_visitors : ℕ := morning_visitors - noon_departures + afternoon_visitors
  (total_visitors = 741) ∧ (current_visitors = 562) := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_visitors_l2131_213171


namespace NUMINAMATH_CALUDE_kite_area_in_regular_hexagon_l2131_213132

/-- The area of a kite-shaped region in a regular hexagon -/
theorem kite_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 8) :
  let radius := side_length
  let angle := 120 * π / 180
  let kite_area := (1 / 2) * radius * radius * Real.sin angle
  kite_area = 16 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_kite_area_in_regular_hexagon_l2131_213132


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_f_unique_l2131_213112

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then 0 else 2 / (2 - x)

theorem f_satisfies_conditions :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  f 2 = 0 ∧
  (∀ x : ℝ, 0 ≤ x → x < 2 → f x ≠ 0) ∧
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) :=
sorry

theorem f_unique :
  ∀ g : ℝ → ℝ,
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → g (x * g y) * g y = g (x + y)) →
  g 2 = 0 →
  (∀ x : ℝ, 0 ≤ x → x < 2 → g x ≠ 0) →
  (∀ x : ℝ, x ≥ 0 → g x ≥ 0) →
  g = f :=
sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_f_unique_l2131_213112


namespace NUMINAMATH_CALUDE_range_of_m_l2131_213140

def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2131_213140


namespace NUMINAMATH_CALUDE_area_FJGH_area_FJGH_proof_l2131_213166

/-- Represents a parallelogram EFGH with point J on side EH -/
structure Parallelogram where
  /-- Length of side EH -/
  eh : ℝ
  /-- Length of JH -/
  jh : ℝ
  /-- Height of the parallelogram from FG to EH -/
  height : ℝ
  /-- Condition that EH = 12 -/
  eh_eq : eh = 12
  /-- Condition that JH = 8 -/
  jh_eq : jh = 8
  /-- Condition that the height is 10 -/
  height_eq : height = 10

/-- The area of region FJGH in the parallelogram is 100 -/
theorem area_FJGH (p : Parallelogram) : ℝ :=
  100

#check area_FJGH

/-- Proof of the theorem -/
theorem area_FJGH_proof (p : Parallelogram) : area_FJGH p = 100 := by
  sorry

end NUMINAMATH_CALUDE_area_FJGH_area_FJGH_proof_l2131_213166


namespace NUMINAMATH_CALUDE_jill_jack_distance_difference_l2131_213110

/-- The side length of the inner square (Jack's path) in feet -/
def inner_side_length : ℕ := 300

/-- The width of the street in feet -/
def street_width : ℕ := 15

/-- The side length of the outer square (Jill's path) in feet -/
def outer_side_length : ℕ := inner_side_length + 2 * street_width

/-- The difference in distance run by Jill and Jack -/
def distance_difference : ℕ := 4 * outer_side_length - 4 * inner_side_length

theorem jill_jack_distance_difference : distance_difference = 120 := by
  sorry

end NUMINAMATH_CALUDE_jill_jack_distance_difference_l2131_213110


namespace NUMINAMATH_CALUDE_valid_sequences_count_l2131_213161

/-- The number of colors available at each station -/
def num_colors : ℕ := 4

/-- The number of stations (including start and end) -/
def num_stations : ℕ := 4

/-- A function that calculates the number of valid color sequences -/
def count_valid_sequences : ℕ :=
  num_colors * (num_colors - 1)^(num_stations - 1)

/-- Theorem stating that the number of valid color sequences is 108 -/
theorem valid_sequences_count :
  count_valid_sequences = 108 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l2131_213161


namespace NUMINAMATH_CALUDE_time_to_run_square_field_l2131_213181

/-- The time taken for a boy to run around a square field -/
theorem time_to_run_square_field (side_length : ℝ) (speed_kmh : ℝ) : 
  side_length = 35 → speed_kmh = 9 → 
  (4 * side_length) / (speed_kmh * 1000 / 3600) = 56 := by
  sorry

end NUMINAMATH_CALUDE_time_to_run_square_field_l2131_213181


namespace NUMINAMATH_CALUDE_thabo_hardcover_books_l2131_213195

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 500 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 30 ∧
  bc.paperback_fiction = 3 * bc.paperback_nonfiction

theorem thabo_hardcover_books (bc : BookCollection) 
  (h : is_valid_collection bc) : bc.hardcover_nonfiction = 76 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_books_l2131_213195


namespace NUMINAMATH_CALUDE_phone_problem_solution_l2131_213184

/-- Represents a phone model with purchase and selling prices -/
structure PhoneModel where
  purchase_price : ℝ
  selling_price : ℝ

/-- The problem setup -/
def phone_problem : Prop :=
  let a : PhoneModel := ⟨3000, 3400⟩
  let b : PhoneModel := ⟨3500, 4000⟩
  ∃ (x y : ℕ),
    (x * a.purchase_price + y * b.purchase_price = 32000) ∧
    (x * (a.selling_price - a.purchase_price) + y * (b.selling_price - b.purchase_price) = 4400) ∧
    x = 6 ∧ y = 4

/-- The profit maximization problem -/
def profit_maximization : Prop :=
  let a : PhoneModel := ⟨3000, 3400⟩
  let b : PhoneModel := ⟨3500, 4000⟩
  ∃ (x : ℕ),
    x ≥ 10 ∧
    (30 - x) ≤ 2 * x ∧
    400 * x + 500 * (30 - x) = 14000 ∧
    ∀ (y : ℕ), y ≥ 10 → (30 - y) ≤ 2 * y → 400 * y + 500 * (30 - y) ≤ 14000

theorem phone_problem_solution : 
  phone_problem ∧ profit_maximization :=
sorry

end NUMINAMATH_CALUDE_phone_problem_solution_l2131_213184


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l2131_213157

theorem factor_difference_of_squares (y : ℝ) : 100 - 25 * y^2 = 25 * (2 - y) * (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l2131_213157


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2131_213178

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_M_and_N : 
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2131_213178


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l2131_213123

theorem multiple_solutions_exist : ∃ p₁ p₂ : ℝ, 
  p₁ ≠ p₂ ∧ 
  p₁ ∈ Set.Ioo 0 1 ∧ 
  p₂ ∈ Set.Ioo 0 1 ∧
  10 * p₁^3 * (1 - p₁)^2 = 144/625 ∧
  10 * p₂^3 * (1 - p₂)^2 = 144/625 :=
sorry

end NUMINAMATH_CALUDE_multiple_solutions_exist_l2131_213123


namespace NUMINAMATH_CALUDE_initial_hay_bales_l2131_213126

theorem initial_hay_bales (better_quality_cost previous_cost : ℚ) 
  (cost_difference : ℚ) : 
  better_quality_cost = 18 →
  previous_cost = 15 →
  cost_difference = 210 →
  ∃ x : ℚ, x = 10 ∧ 2 * better_quality_cost * x - previous_cost * x = cost_difference :=
by sorry

end NUMINAMATH_CALUDE_initial_hay_bales_l2131_213126


namespace NUMINAMATH_CALUDE_cubic_derivative_equality_l2131_213150

theorem cubic_derivative_equality (f : ℝ → ℝ) (x : ℝ) :
  (f = fun x ↦ x^3) →
  (deriv f x = 3) →
  (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_derivative_equality_l2131_213150


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2131_213187

theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = 4 * a n) 
  (h_sum : a 1 + a 2 + a 3 = 21) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2131_213187


namespace NUMINAMATH_CALUDE_share_distribution_l2131_213138

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 595 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  a = 70 := by sorry

end NUMINAMATH_CALUDE_share_distribution_l2131_213138


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2131_213153

theorem polynomial_multiplication_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^9 + 5*y^7 + 2*y^5) = 
  15*y^13 - 10*y^12 + 9*y^10 - 6*y^9 + 15*y^8 - 10*y^7 + 6*y^6 - 4*y^5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l2131_213153


namespace NUMINAMATH_CALUDE_josh_marbles_l2131_213121

/-- The number of marbles Josh has -/
def total_marbles (blue red yellow : ℕ) : ℕ := blue + red + yellow

/-- The problem statement -/
theorem josh_marbles : 
  ∀ (blue red yellow : ℕ),
  blue = 3 * red →
  red = 14 →
  yellow = 29 →
  total_marbles blue red yellow = 85 := by
sorry

end NUMINAMATH_CALUDE_josh_marbles_l2131_213121


namespace NUMINAMATH_CALUDE_sum_fractions_l2131_213100

theorem sum_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_l2131_213100


namespace NUMINAMATH_CALUDE_range_of_m_l2131_213186

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) 
  (h_ineq : ∀ m : ℝ, x + y > m^2 + 8*m) : 
  -9 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2131_213186


namespace NUMINAMATH_CALUDE_inequality_relation_l2131_213124

theorem inequality_relation (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, (1 / x < a ∧ x ≤ 1 / a)) ∧
  (∀ x : ℝ, x > 1 / a → 1 / x < a) :=
sorry

end NUMINAMATH_CALUDE_inequality_relation_l2131_213124


namespace NUMINAMATH_CALUDE_min_value_theorem_l2131_213134

theorem min_value_theorem (x : ℝ) (h : x > 0) : 
  x + 81 / x ≥ 18 ∧ (x + 81 / x = 18 ↔ x = 9) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2131_213134


namespace NUMINAMATH_CALUDE_classmate_height_most_suitable_l2131_213119

-- Define the characteristics of a survey
structure Survey where
  destructive : Bool
  large_scale : Bool

-- Define the four survey options
def light_bulb_survey : Survey := { destructive := true, large_scale := false }
def classmate_height_survey : Survey := { destructive := false, large_scale := false }
def nationwide_student_survey : Survey := { destructive := false, large_scale := true }
def missile_accuracy_survey : Survey := { destructive := true, large_scale := false }

-- Define what makes a survey suitable for a census
def suitable_for_census (s : Survey) : Prop := ¬s.destructive ∧ ¬s.large_scale

-- Theorem stating that the classmate height survey is most suitable for a census
theorem classmate_height_most_suitable :
  suitable_for_census classmate_height_survey ∧
  ¬suitable_for_census light_bulb_survey ∧
  ¬suitable_for_census nationwide_student_survey ∧
  ¬suitable_for_census missile_accuracy_survey :=
sorry

end NUMINAMATH_CALUDE_classmate_height_most_suitable_l2131_213119


namespace NUMINAMATH_CALUDE_pet_food_ratio_l2131_213136

/-- Represents the amounts of pet food in kilograms -/
structure PetFood where
  dog : ℕ
  cat : ℕ
  bird : ℕ

/-- The total amount of pet food -/
def total_food (pf : PetFood) : ℕ := pf.dog + pf.cat + pf.bird

/-- The ratio of pet food types -/
def food_ratio (pf : PetFood) : (ℕ × ℕ × ℕ) :=
  let gcd := Nat.gcd pf.dog (Nat.gcd pf.cat pf.bird)
  (pf.dog / gcd, pf.cat / gcd, pf.bird / gcd)

theorem pet_food_ratio : 
  let bought := PetFood.mk 15 10 5
  let final := PetFood.mk 40 15 5
  let initial := PetFood.mk (final.dog - bought.dog) (final.cat - bought.cat) (final.bird - bought.bird)
  total_food final = 60 →
  food_ratio final = (8, 3, 1) := by
  sorry

end NUMINAMATH_CALUDE_pet_food_ratio_l2131_213136


namespace NUMINAMATH_CALUDE_percentage_problem_l2131_213133

theorem percentage_problem (P : ℝ) : 
  (0.1 * 0.3 * (P / 100) * 6000 = 90) → P = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2131_213133


namespace NUMINAMATH_CALUDE_roots_of_equation_l2131_213101

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 4*x + 3)*(x - 5)*(x + 1)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 3 ∨ x = 5 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2131_213101


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l2131_213113

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_two_digit_prime_factor_of_binomial_200_100 :
  ∃ (p : ℕ), Prime p ∧ p < 100 ∧ p ∣ binomial 200 100 ∧
  ∀ (q : ℕ), Prime q → q < 100 → q ∣ binomial 200 100 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_200_100_l2131_213113


namespace NUMINAMATH_CALUDE_plant_growth_theorem_l2131_213129

structure PlantType where
  seedsPerPacket : ℕ
  growthRate : ℕ
  initialPackets : ℕ

def totalPlants (p : PlantType) : ℕ :=
  p.seedsPerPacket * p.initialPackets

def additionalPacketsNeeded (p : PlantType) (targetPlants : ℕ) : ℕ :=
  max 0 ((targetPlants - totalPlants p + p.seedsPerPacket - 1) / p.seedsPerPacket)

def growthTime (p : PlantType) (targetPlants : ℕ) : ℕ :=
  p.growthRate * max 0 (targetPlants - totalPlants p)

theorem plant_growth_theorem (targetPlants : ℕ) 
  (typeA typeB typeC : PlantType)
  (h1 : typeA = { seedsPerPacket := 3, growthRate := 5, initialPackets := 2 })
  (h2 : typeB = { seedsPerPacket := 6, growthRate := 7, initialPackets := 3 })
  (h3 : typeC = { seedsPerPacket := 9, growthRate := 4, initialPackets := 3 })
  (h4 : targetPlants = 12) : 
  additionalPacketsNeeded typeA targetPlants = 2 ∧ 
  growthTime typeA targetPlants = 5 ∧
  additionalPacketsNeeded typeB targetPlants = 0 ∧
  additionalPacketsNeeded typeC targetPlants = 0 := by
  sorry

end NUMINAMATH_CALUDE_plant_growth_theorem_l2131_213129


namespace NUMINAMATH_CALUDE_linear_function_monotonicity_linear_function_parity_linear_function_y_intercept_linear_function_x_intercept_l2131_213151

-- Define a linear function
def linearFunction (a b x : ℝ) : ℝ := a * x + b

-- Theorem about monotonicity of linear functions
theorem linear_function_monotonicity (a b : ℝ) :
  (∀ x y : ℝ, x < y → linearFunction a b x < linearFunction a b y) ↔ a > 0 :=
sorry

-- Theorem about parity of linear functions
theorem linear_function_parity (a b : ℝ) :
  (∀ x : ℝ, linearFunction a b (-x) = -linearFunction a b x + 2*b) ↔ b = 0 :=
sorry

-- Theorem about y-intercept of linear functions
theorem linear_function_y_intercept (a b : ℝ) :
  linearFunction a b 0 = b :=
sorry

-- Theorem about x-intercept of linear functions (when it exists)
theorem linear_function_x_intercept (a b : ℝ) (h : a ≠ 0) :
  linearFunction a b (-b/a) = 0 :=
sorry

end NUMINAMATH_CALUDE_linear_function_monotonicity_linear_function_parity_linear_function_y_intercept_linear_function_x_intercept_l2131_213151


namespace NUMINAMATH_CALUDE_unique_prime_tower_l2131_213146

def tower_of_twos (p : ℕ) : ℕ :=
  match p with
  | 0 => 1
  | n + 1 => 2^(tower_of_twos n)

def is_prime_tower (p : ℕ) : Prop :=
  Nat.Prime (tower_of_twos p + 9)

theorem unique_prime_tower : ∀ p : ℕ, is_prime_tower p ↔ p = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_tower_l2131_213146


namespace NUMINAMATH_CALUDE_subtract_fractions_l2131_213147

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l2131_213147


namespace NUMINAMATH_CALUDE_exponent_equality_l2131_213122

theorem exponent_equality (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2131_213122


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2131_213167

/-- Given a point A and a line L, this theorem proves that the equation
    4x + y - 14 = 0 represents the line passing through A and parallel to L. -/
theorem parallel_line_equation (A : ℝ × ℝ) (L : Set (ℝ × ℝ)) : 
  A.1 = 3 ∧ A.2 = 2 →
  L = {(x, y) | 4 * x + y - 2 = 0} →
  {(x, y) | 4 * x + y - 14 = 0} = 
    {(x, y) | ∃ (t : ℝ), x = A.1 + t ∧ y = A.2 - 4 * t} :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2131_213167


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2131_213118

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : 
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2131_213118


namespace NUMINAMATH_CALUDE_cube_shadow_problem_l2131_213179

/-- The shadow area function calculates the area of the shadow cast by a cube,
    excluding the area beneath the cube. -/
def shadow_area (cube_edge : ℝ) (light_height : ℝ) : ℝ := sorry

/-- The problem statement -/
theorem cube_shadow_problem (y : ℝ) : 
  shadow_area 2 y = 200 → 
  ⌊1000 * y⌋ = 6140 := by sorry

end NUMINAMATH_CALUDE_cube_shadow_problem_l2131_213179


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2131_213137

/-- A symmetric trapezoid EFGH with given properties -/
structure SymmetricTrapezoid :=
  (EF : ℝ)  -- Length of top base EF
  (GH : ℝ)  -- Length of bottom base GH
  (height : ℝ)  -- Height from EF to GH
  (isSymmetric : Bool)  -- Is the trapezoid symmetric?
  (EFGHEqual : Bool)  -- Are EF and GH equal in length?

/-- Properties of the specific trapezoid in the problem -/
def problemTrapezoid : SymmetricTrapezoid :=
  { EF := 10,
    GH := 22,
    height := 6,
    isSymmetric := true,
    EFGHEqual := true }

/-- Theorem stating the perimeter of the trapezoid -/
theorem trapezoid_perimeter (t : SymmetricTrapezoid) 
  (h1 : t.EF = 10)
  (h2 : t.GH = t.EF + 12)
  (h3 : t.height = 6)
  (h4 : t.isSymmetric = true)
  (h5 : t.EFGHEqual = true) :
  (t.EF + t.GH + 2 * Real.sqrt (t.height^2 + ((t.GH - t.EF) / 2)^2)) = 32 + 12 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2131_213137


namespace NUMINAMATH_CALUDE_power_difference_divisibility_l2131_213106

theorem power_difference_divisibility (N : ℕ+) :
  ∃ (r s : ℕ), r ≠ s ∧ ∀ (A : ℤ), (N : ℤ) ∣ (A^r - A^s) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_divisibility_l2131_213106


namespace NUMINAMATH_CALUDE_difference_x_y_l2131_213103

theorem difference_x_y (x y : ℤ) 
  (sum_eq : x + y = 20)
  (diff_eq : x - y = 10)
  (x_val : x = 15) :
  x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_x_y_l2131_213103


namespace NUMINAMATH_CALUDE_triangle_side_length_l2131_213152

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2131_213152


namespace NUMINAMATH_CALUDE_painting_price_l2131_213165

theorem painting_price (num_paintings : ℕ) (num_toys : ℕ) (toy_price : ℝ) 
  (painting_discount : ℝ) (toy_discount : ℝ) (total_loss : ℝ) :
  num_paintings = 10 →
  num_toys = 8 →
  toy_price = 20 →
  painting_discount = 0.1 →
  toy_discount = 0.15 →
  total_loss = 64 →
  ∃ (painting_price : ℝ),
    painting_price * num_paintings + toy_price * num_toys -
    (painting_price * (1 - painting_discount) * num_paintings + 
     toy_price * (1 - toy_discount) * num_toys) = total_loss ∧
    painting_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_l2131_213165


namespace NUMINAMATH_CALUDE_intersection_A_B_l2131_213173

def A : Set ℝ := {x | Real.sqrt (x^2 - 1) / Real.sqrt x = 0}
def B : Set ℝ := {y | -2 ≤ y ∧ y ≤ 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2131_213173


namespace NUMINAMATH_CALUDE_overlap_length_l2131_213183

/-- Given information about overlapping segments, prove the length of each overlap --/
theorem overlap_length (total_length : ℝ) (measured_length : ℝ) (num_overlaps : ℕ) 
  (h1 : total_length = 98)
  (h2 : measured_length = 83)
  (h3 : num_overlaps = 6) :
  ∃ x : ℝ, x = 2.5 ∧ total_length = measured_length + num_overlaps * x :=
by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l2131_213183


namespace NUMINAMATH_CALUDE_minimum_value_implies_b_equals_one_l2131_213158

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + x + b

-- State the theorem
theorem minimum_value_implies_b_equals_one (a : ℝ) :
  (∃ b : ℝ, (f a b 1 = 1) ∧ 
    (∀ x : ℝ, f a b x ≥ 1) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a b x < f a b 1 + ε)) →
  (∃ b : ℝ, b = 1 ∧ (f a b 1 = 1) ∧ 
    (∀ x : ℝ, f a b x ≥ 1) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → f a b x < f a b 1 + ε)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_b_equals_one_l2131_213158


namespace NUMINAMATH_CALUDE_train_length_calculation_l2131_213115

/-- Given a train that crosses a platform and a post, calculate its length. -/
theorem train_length_calculation (platform_length : ℝ) (platform_time : ℝ) (post_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : post_time = 18) :
  ∃ (train_length : ℝ), train_length = 300 ∧ 
    (train_length + platform_length) / platform_time = train_length / post_time := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l2131_213115


namespace NUMINAMATH_CALUDE_factorization_proof_l2131_213128

theorem factorization_proof (x y : ℝ) : x * (y - 1) + 4 * (1 - y) = (y - 1) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2131_213128


namespace NUMINAMATH_CALUDE_f_greater_than_one_max_a_for_derivative_inequality_l2131_213163

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - (1/2) * a * x^2

theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f 2 x > 1 := by sorry

theorem max_a_for_derivative_inequality :
  ∃ (a : ℕ), a = 2 ∧
  (∀ (x : ℝ), x > 0 → deriv (f a) x ≥ x^2 * log x) ∧
  (∀ (b : ℕ), b > 2 → ∃ (x : ℝ), x > 0 ∧ deriv (f b) x < x^2 * log x) := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_max_a_for_derivative_inequality_l2131_213163


namespace NUMINAMATH_CALUDE_no_threefold_decreasing_number_l2131_213155

theorem no_threefold_decreasing_number : ¬∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c = 3 * (100 * b + 10 * c + a)) := by
  sorry

end NUMINAMATH_CALUDE_no_threefold_decreasing_number_l2131_213155


namespace NUMINAMATH_CALUDE_product_95_105_l2131_213102

theorem product_95_105 : 95 * 105 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_product_95_105_l2131_213102


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2131_213177

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 2) : 
  x + y ≥ 9/2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 2 ∧ x + y = 9/2 :=
by
  sorry

#check min_value_of_sum

end NUMINAMATH_CALUDE_min_value_of_sum_l2131_213177


namespace NUMINAMATH_CALUDE_inverse_not_correct_l2131_213189

-- Define the set T as non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation ⊕
def oplus (a b : ℝ) : ℝ := 3 * a * b + a * b

-- Theorem statement
theorem inverse_not_correct (a : ℝ) (ha : a ∈ T) : 
  ¬(∀ (x : ℝ), x ∈ T → oplus a (1 / (3 * a + a)) = 1 / 4 ∧ oplus (1 / (3 * a + a)) a = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_not_correct_l2131_213189


namespace NUMINAMATH_CALUDE_set_intersection_union_l2131_213199

theorem set_intersection_union (M N P : Set ℕ) : 
  M = {1} → N = {1, 2} → P = {1, 2, 3} → (M ∪ N) ∩ P = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_l2131_213199


namespace NUMINAMATH_CALUDE_square_division_exists_l2131_213114

/-- Represents a trapezoid with a given height -/
structure Trapezoid where
  height : ℝ

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ

/-- Represents a division of a square into trapezoids -/
structure SquareDivision where
  square : Square
  trapezoids : List Trapezoid

/-- Checks if a list of trapezoids has the required heights -/
def has_required_heights (trapezoids : List Trapezoid) : Prop :=
  trapezoids.length = 4 ∧
  (∃ (h₁ h₂ h₃ h₄ : Trapezoid),
    trapezoids = [h₁, h₂, h₃, h₄] ∧
    h₁.height = 1 ∧ h₂.height = 2 ∧ h₃.height = 3 ∧ h₄.height = 4)

/-- Checks if a square division is valid -/
def is_valid_division (div : SquareDivision) : Prop :=
  div.square.side_length = 4 ∧
  has_required_heights div.trapezoids

/-- Theorem: A square with side length 4 can be divided into four trapezoids with heights 1, 2, 3, and 4 -/
theorem square_division_exists : ∃ (div : SquareDivision), is_valid_division div := by
  sorry

end NUMINAMATH_CALUDE_square_division_exists_l2131_213114


namespace NUMINAMATH_CALUDE_square_construction_l2131_213188

noncomputable section

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the line
def Line (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the square
structure Square (P Q V U : ℝ × ℝ) : Prop where
  side_equal : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (Q.1 - V.1)^2 + (Q.2 - V.2)^2
             ∧ (Q.1 - V.1)^2 + (Q.2 - V.2)^2 = (V.1 - U.1)^2 + (V.2 - U.2)^2
             ∧ (V.1 - U.1)^2 + (V.2 - U.2)^2 = (U.1 - P.1)^2 + (U.2 - P.2)^2
  right_angles : (P.1 - Q.1) * (Q.1 - V.1) + (P.2 - Q.2) * (Q.2 - V.2) = 0
                ∧ (Q.1 - V.1) * (V.1 - U.1) + (Q.2 - V.2) * (V.2 - U.2) = 0
                ∧ (V.1 - U.1) * (U.1 - P.1) + (V.2 - U.2) * (U.2 - P.2) = 0
                ∧ (U.1 - P.1) * (P.1 - Q.1) + (U.2 - P.2) * (P.2 - Q.2) = 0

theorem square_construction (O : ℝ × ℝ) (r : ℝ) (a b c : ℝ) 
  (h : ∀ p ∈ Line a b c, p ∉ Circle O r) :
  ∃ P Q V U : ℝ × ℝ, Square P Q V U ∧ 
    P ∈ Line a b c ∧ Q ∈ Line a b c ∧
    V ∈ Circle O r ∧ U ∈ Circle O r :=
sorry

end NUMINAMATH_CALUDE_square_construction_l2131_213188


namespace NUMINAMATH_CALUDE_negation_equivalence_l2131_213116

theorem negation_equivalence :
  (¬ ∀ m : ℝ, m > 0 → m^2 > 0) ↔ (∃ m : ℝ, m ≤ 0 ∧ m^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2131_213116


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2131_213182

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * a 2^2 - 10 * a 2 + 3 = 0) →
  (3 * a 6^2 - 10 * a 6 + 3 = 0) →
  (1 / a 2 + 1 / a 6 + a 4^2 = 13/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2131_213182


namespace NUMINAMATH_CALUDE_symmetric_function_properties_l2131_213135

def f (x m : ℝ) : ℝ := 2 * |x| + |2*x - m|

theorem symmetric_function_properties (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x : ℝ, f x m = f (2 - x) m) :
  (m = 4) ∧ 
  (∀ x : ℝ, f x m ≥ 4) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = m → 1/a + 4/b ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_properties_l2131_213135


namespace NUMINAMATH_CALUDE_negative_square_to_fourth_power_l2131_213120

theorem negative_square_to_fourth_power (a : ℝ) : (-a^2)^4 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_to_fourth_power_l2131_213120


namespace NUMINAMATH_CALUDE_smallest_square_area_l2131_213149

/-- Given three squares arranged as described in the problem, 
    this theorem relates the area of the smallest square to that of the middle square. -/
theorem smallest_square_area 
  (largest_square_area : ℝ) 
  (middle_square_area : ℝ) 
  (h1 : largest_square_area = 1) 
  (h2 : 0 < middle_square_area) 
  (h3 : middle_square_area < 1) :
  ∃ (smallest_square_area : ℝ), 
    smallest_square_area = ((1 - middle_square_area) / 2)^2 ∧ 
    0 < smallest_square_area ∧
    smallest_square_area < middle_square_area := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l2131_213149


namespace NUMINAMATH_CALUDE_C_grazed_for_4_months_l2131_213176

/-- The number of milkmen who rented the pasture -/
def num_milkmen : ℕ := 4

/-- The number of cows grazed by milkman A -/
def cows_A : ℕ := 24

/-- The number of months milkman A grazed his cows -/
def months_A : ℕ := 3

/-- The number of cows grazed by milkman B -/
def cows_B : ℕ := 10

/-- The number of months milkman B grazed his cows -/
def months_B : ℕ := 5

/-- The number of cows grazed by milkman C -/
def cows_C : ℕ := 35

/-- The number of cows grazed by milkman D -/
def cows_D : ℕ := 21

/-- The number of months milkman D grazed his cows -/
def months_D : ℕ := 3

/-- A's share of the rent in rupees -/
def share_A : ℕ := 720

/-- The total rent of the field in rupees -/
def total_rent : ℕ := 3250

/-- The theorem stating that C grazed his cows for 4 months -/
theorem C_grazed_for_4_months :
  ∃ (months_C : ℕ),
    months_C = 4 ∧
    total_rent = share_A +
      (cows_B * months_B * share_A / (cows_A * months_A)) +
      (cows_C * months_C * share_A / (cows_A * months_A)) +
      (cows_D * months_D * share_A / (cows_A * months_A)) :=
by sorry

end NUMINAMATH_CALUDE_C_grazed_for_4_months_l2131_213176


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2131_213144

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (a + 2*i)/i = b + i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2131_213144


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2131_213185

theorem inequality_solution_set : 
  {x : ℝ | 2 ≥ (1 / (x - 1))} = Set.Iic 1 ∪ Set.Ici (3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2131_213185


namespace NUMINAMATH_CALUDE_basketball_points_third_game_l2131_213175

theorem basketball_points_third_game 
  (total_points : ℕ) 
  (first_game_fraction : ℚ) 
  (second_game_fraction : ℚ) 
  (h1 : total_points = 20) 
  (h2 : first_game_fraction = 1/2) 
  (h3 : second_game_fraction = 1/10) : 
  total_points - (first_game_fraction * total_points + second_game_fraction * total_points) = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_points_third_game_l2131_213175


namespace NUMINAMATH_CALUDE_teacher_student_arrangement_l2131_213105

theorem teacher_student_arrangement (n : ℕ) (m : ℕ) :
  n = 1 ∧ m = 6 →
  (n + m - 2) * (m.factorial) = 3600 :=
by sorry

end NUMINAMATH_CALUDE_teacher_student_arrangement_l2131_213105


namespace NUMINAMATH_CALUDE_fuel_savings_l2131_213148

theorem fuel_savings (old_efficiency : ℝ) (old_cost : ℝ) 
  (h_old_positive : old_efficiency > 0) (h_cost_positive : old_cost > 0) : 
  let new_efficiency := old_efficiency * (1 + 0.6)
  let new_cost := old_cost * 1.25
  let old_trip_cost := old_cost
  let new_trip_cost := (old_efficiency / new_efficiency) * new_cost
  let savings_percent := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percent = 21.875 := by
sorry


end NUMINAMATH_CALUDE_fuel_savings_l2131_213148


namespace NUMINAMATH_CALUDE_pq_length_l2131_213172

/-- Triangle DEF with given side lengths and a parallel segment PQ on DE --/
structure TriangleWithParallelSegment where
  /-- Length of side DE --/
  de : ℝ
  /-- Length of side EF --/
  ef : ℝ
  /-- Length of side FD --/
  fd : ℝ
  /-- Length of segment PQ --/
  pq : ℝ
  /-- PQ is parallel to EF --/
  pq_parallel_ef : True
  /-- PQ is on DE --/
  pq_on_de : True
  /-- PQ is one-third of DE --/
  pq_is_third_of_de : pq = de / 3

/-- The length of PQ in the given triangle is 25/3 --/
theorem pq_length (t : TriangleWithParallelSegment) (h1 : t.de = 25) (h2 : t.ef = 29) (h3 : t.fd = 32) : 
  t.pq = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_l2131_213172


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l2131_213104

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l2131_213104


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2131_213196

theorem hyperbola_equation (a b p x₀ : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (b / a = 2) →
  (p / 2 = 4 / 3) →
  (x₀ = 3) →
  (16 = 2 * p * x₀) →
  (9 / a^2 - 16 / b^2 = 1) →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 5 - y^2 / 20 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2131_213196


namespace NUMINAMATH_CALUDE_circle_C_properties_l2131_213190

/-- Circle C defined by the equation x^2 + y^2 - 2x + 4y - 4 = 0 --/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- The center of circle C --/
def center : ℝ × ℝ := (1, -2)

/-- The radius of circle C --/
def radius : ℝ := 3

/-- A line with slope 1 --/
def line_with_slope_1 (a b : ℝ) (x y : ℝ) : Prop := y - b = x - a

/-- Theorem stating the properties of circle C and the existence of special lines --/
theorem circle_C_properties :
  (∀ x y : ℝ, circle_C x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
  (∃ a b : ℝ, (line_with_slope_1 a b (0) (0) ∧ 
              (line_with_slope_1 a b (-4) (-4) ∨ line_with_slope_1 a b (1) (1)) ∧
              (∃ x₁ y₁ x₂ y₂ : ℝ, 
                circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                line_with_slope_1 a b x₁ y₁ ∧ line_with_slope_1 a b x₂ y₂ ∧
                (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * ((x₁ + x₂)/2)^2 + 4 * ((y₁ + y₂)/2)^2))) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l2131_213190


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2131_213156

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > p.1 + 1 ∧ p.2 > 3 - 2*p.1}

-- Theorem stating that all points in S are in Quadrants I or II
theorem points_in_quadrants_I_and_II : 
  ∀ p ∈ S, (p.1 > 0 ∧ p.2 > 0) ∨ (p.1 < 0 ∧ p.2 > 0) := by
  sorry

-- Helper lemma: All points in S have positive y-coordinate
lemma points_have_positive_y : 
  ∀ p ∈ S, p.2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2131_213156


namespace NUMINAMATH_CALUDE_parabola_vertex_vertex_coordinates_l2131_213145

/-- The vertex of a parabola y = a(x - h)^2 + k is the point (h, k) --/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * (x - h)^2 + k
  (h, k) = (h, f h) ∧ ∀ x, f x ≥ f h := by sorry

/-- The coordinates of the vertex of the parabola y = 2(x-1)^2 + 8 are (1, 8) --/
theorem vertex_coordinates :
  let f : ℝ → ℝ := fun x ↦ 2 * (x - 1)^2 + 8
  (1, 8) = (1, f 1) ∧ ∀ x, f x ≥ f 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_vertex_coordinates_l2131_213145


namespace NUMINAMATH_CALUDE_max_candy_pieces_l2131_213108

theorem max_candy_pieces (n : ℕ) (μ : ℚ) (min_pieces : ℕ) : 
  n = 35 → 
  μ = 6 → 
  min_pieces = 2 →
  ∃ (max_pieces : ℕ), 
    max_pieces = 142 ∧ 
    (∀ (student_pieces : List ℕ), 
      student_pieces.length = n ∧ 
      (∀ x ∈ student_pieces, x ≥ min_pieces) ∧ 
      (student_pieces.sum : ℚ) / n = μ →
      ∀ x ∈ student_pieces, x ≤ max_pieces) :=
by sorry

end NUMINAMATH_CALUDE_max_candy_pieces_l2131_213108


namespace NUMINAMATH_CALUDE_mistaken_subtraction_l2131_213194

/-- Given a two-digit number where the units digit is 9, 
    if subtracting 57 from the number with the units digit mistaken as 6 results in 39,
    then the original number is 99. -/
theorem mistaken_subtraction (x : ℕ) : 
  x < 10 →  -- Ensure x is a single digit (tens place)
  (10 * x + 6) - 57 = 39 → 
  10 * x + 9 = 99 :=
by sorry

end NUMINAMATH_CALUDE_mistaken_subtraction_l2131_213194


namespace NUMINAMATH_CALUDE_same_side_of_line_l2131_213154

/-- Given a line x + y = a, if the origin (0, 0) and the point (1, 1) are on the same side of this line, then a < 0 or a > 2. -/
theorem same_side_of_line (a : ℝ) : 
  (0 + 0 - a) * (1 + 1 - a) > 0 → a < 0 ∨ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_same_side_of_line_l2131_213154


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l2131_213162

theorem inequality_system_solution_range (m : ℝ) : 
  (∀ x : ℝ, ((x - 1) / 2 ≥ (x - 2) / 3 ∧ 2 * x - m ≥ x) ↔ x ≥ m) → 
  m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l2131_213162


namespace NUMINAMATH_CALUDE_bus_meeting_problem_l2131_213197

theorem bus_meeting_problem (n k : ℕ) : n > 3 → 
  (n * (n - 1) * (2 * k - 1) = 600) → 
  ((n = 4 ∧ k = 13) ∨ (n = 5 ∧ k = 8)) := by
  sorry

end NUMINAMATH_CALUDE_bus_meeting_problem_l2131_213197


namespace NUMINAMATH_CALUDE_domain_v_correct_l2131_213127

/-- The domain of v(x, y) = 1/√(x + y) where x and y are real numbers -/
def domain_v : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 > -p.1}

/-- The function v(x, y) = 1/√(x + y) -/
noncomputable def v (p : ℝ × ℝ) : ℝ :=
  1 / Real.sqrt (p.1 + p.2)

theorem domain_v_correct :
  ∀ p : ℝ × ℝ, p ∈ domain_v ↔ ∃ z : ℝ, v p = z :=
by sorry

end NUMINAMATH_CALUDE_domain_v_correct_l2131_213127


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2131_213159

/-- The volume of a cube with surface area 24 cm² is 8 cm³. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 24 → s^3 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2131_213159


namespace NUMINAMATH_CALUDE_desmond_bought_240_toys_l2131_213169

/-- The number of toys Mr. Desmond bought for his elder son -/
def elder_son_toys : ℕ := 60

/-- The number of toys Mr. Desmond bought for his younger son -/
def younger_son_toys : ℕ := 3 * elder_son_toys

/-- The total number of toys Mr. Desmond bought -/
def total_toys : ℕ := elder_son_toys + younger_son_toys

theorem desmond_bought_240_toys : total_toys = 240 := by
  sorry

end NUMINAMATH_CALUDE_desmond_bought_240_toys_l2131_213169


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2131_213141

/-- Given a line segment with one endpoint (5, -2) and midpoint (3, 4),
    the sum of coordinates of the other endpoint is 11. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (5 + x) / 2 = 3 → 
    (-2 + y) / 2 = 4 → 
    x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2131_213141


namespace NUMINAMATH_CALUDE_max_y_value_l2131_213193

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) : 
  y ≤ 27 ∧ ∃ (x' y' : ℤ), x' * y' + 7 * x' + 6 * y' = -8 ∧ y' = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2131_213193


namespace NUMINAMATH_CALUDE_cyclists_speed_l2131_213125

theorem cyclists_speed (initial_distance : ℝ) (fly_speed : ℝ) (fly_distance : ℝ) :
  initial_distance = 50 ∧ 
  fly_speed = 15 ∧ 
  fly_distance = 37.5 →
  ∃ (cyclist_speed : ℝ),
    cyclist_speed = 10 ∧ 
    initial_distance = 2 * cyclist_speed * (fly_distance / fly_speed) :=
by sorry

end NUMINAMATH_CALUDE_cyclists_speed_l2131_213125


namespace NUMINAMATH_CALUDE_inequality_solution_l2131_213191

theorem inequality_solution (a b : ℝ) (h1 : ∀ x, (2*a - b)*x + a - 5*b > 0 ↔ x < 10/7) :
  ∀ x, a*x + b > 0 ↔ x < -3/5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2131_213191


namespace NUMINAMATH_CALUDE_cos_double_angle_specific_l2131_213168

theorem cos_double_angle_specific (α : Real) :
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cos_double_angle_specific_l2131_213168


namespace NUMINAMATH_CALUDE_factory_output_equation_l2131_213109

/-- Represents the factory's output model -/
def factory_output (initial_output : ℝ) (growth_rate : ℝ) (months : ℕ) : ℝ :=
  initial_output * (1 + growth_rate) ^ months

/-- Theorem stating that the equation 500(1+x)^2 = 720 correctly represents the factory's output in March -/
theorem factory_output_equation (x : ℝ) : 
  factory_output 500 x 2 = 720 ↔ 500 * (1 + x)^2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_equation_l2131_213109


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2131_213160

theorem decimal_to_fraction :
  (3.375 : ℚ) = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2131_213160


namespace NUMINAMATH_CALUDE_sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2_l2131_213170

theorem sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2 :
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt3_plus_sqrt2_inverse_of_sqrt3_minus_sqrt2_l2131_213170


namespace NUMINAMATH_CALUDE_inequality_implication_condition_l2131_213142

theorem inequality_implication_condition :
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) ∧
  ¬(∀ x : ℝ, |x - 1| < 2 → x * (x - 3) < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implication_condition_l2131_213142


namespace NUMINAMATH_CALUDE_university_packaging_volume_l2131_213192

/-- The minimum volume needed to package the university's collection given the box dimensions, cost per box, and minimum amount spent. -/
theorem university_packaging_volume
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (cost_per_box : ℝ)
  (min_amount_spent : ℝ)
  (h_box_length : box_length = 20)
  (h_box_width : box_width = 20)
  (h_box_height : box_height = 12)
  (h_cost_per_box : cost_per_box = 0.5)
  (h_min_amount_spent : min_amount_spent = 200) :
  (min_amount_spent / cost_per_box) * (box_length * box_width * box_height) = 1920000 :=
by sorry

end NUMINAMATH_CALUDE_university_packaging_volume_l2131_213192
