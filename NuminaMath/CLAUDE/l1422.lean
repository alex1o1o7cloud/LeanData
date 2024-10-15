import Mathlib

namespace NUMINAMATH_CALUDE_downstream_speed_l1422_142270

/-- The speed of a man rowing downstream, given his upstream speed and still water speed -/
theorem downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 20 →
  still_water_speed = 40 →
  still_water_speed + (still_water_speed - upstream_speed) = 60 :=
by sorry

end NUMINAMATH_CALUDE_downstream_speed_l1422_142270


namespace NUMINAMATH_CALUDE_probability_three_face_cards_different_suits_value_l1422_142211

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of face cards in a standard deck. -/
def FaceCards : ℕ := 12

/-- The number of suits in a standard deck. -/
def Suits : ℕ := 4

/-- The number of face cards per suit. -/
def FaceCardsPerSuit : ℕ := FaceCards / Suits

/-- The probability of selecting three face cards of different suits from a standard deck without replacement. -/
def probability_three_face_cards_different_suits : ℚ :=
  (FaceCards : ℚ) / StandardDeck *
  (FaceCards - FaceCardsPerSuit : ℚ) / (StandardDeck - 1) *
  (FaceCards - 2 * FaceCardsPerSuit : ℚ) / (StandardDeck - 2)

theorem probability_three_face_cards_different_suits_value :
  probability_three_face_cards_different_suits = 4 / 915 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_face_cards_different_suits_value_l1422_142211


namespace NUMINAMATH_CALUDE_future_cup_analysis_l1422_142241

/-- Represents a class's defensive performance in the "Future Cup" football match --/
structure DefensivePerformance where
  average_goals_conceded : ℝ
  standard_deviation : ℝ

/-- The defensive performance of Class A --/
def class_a : DefensivePerformance :=
  { average_goals_conceded := 1.9,
    standard_deviation := 0.3 }

/-- The defensive performance of Class B --/
def class_b : DefensivePerformance :=
  { average_goals_conceded := 1.3,
    standard_deviation := 1.2 }

theorem future_cup_analysis :
  (class_b.average_goals_conceded < class_a.average_goals_conceded) ∧
  (class_b.standard_deviation > class_a.standard_deviation) ∧
  (class_a.average_goals_conceded + class_a.standard_deviation < 
   class_b.average_goals_conceded + class_b.standard_deviation) :=
by sorry

end NUMINAMATH_CALUDE_future_cup_analysis_l1422_142241


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l1422_142252

theorem quadratic_always_nonnegative (x y : ℝ) : x^2 + x*y + y^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l1422_142252


namespace NUMINAMATH_CALUDE_average_problem_l1422_142215

theorem average_problem (y : ℝ) : (15 + 26 + y) / 3 = 23 → y = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1422_142215


namespace NUMINAMATH_CALUDE_remaining_students_l1422_142246

def number_of_groups : ℕ := 3
def students_per_group : ℕ := 8
def students_who_left : ℕ := 2

theorem remaining_students :
  (number_of_groups * students_per_group) - students_who_left = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_l1422_142246


namespace NUMINAMATH_CALUDE_exam_theorem_l1422_142293

def exam_problem (total_boys : ℕ) (overall_average : ℚ) (passed_boys : ℕ) (failed_average : ℚ) : Prop :=
  let passed_average : ℚ := (total_boys * overall_average - (total_boys - passed_boys) * failed_average) / passed_boys
  passed_average = 39

theorem exam_theorem : exam_problem 120 36 105 15 := by
  sorry

end NUMINAMATH_CALUDE_exam_theorem_l1422_142293


namespace NUMINAMATH_CALUDE_equal_distance_travel_l1422_142284

theorem equal_distance_travel (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 3) (h2 : v2 = 4) (h3 : v3 = 5) (h4 : t = 47/60) :
  let d := t / (1/v1 + 1/v2 + 1/v3)
  3 * d = 3 :=
by sorry

end NUMINAMATH_CALUDE_equal_distance_travel_l1422_142284


namespace NUMINAMATH_CALUDE_exactly_one_true_l1422_142250

-- Define what it means for three numbers to be in geometric progression
def in_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

-- Define the original proposition
def original_proposition : Prop :=
  ∀ a b c : ℝ, in_geometric_progression a b c → b^2 = a * c

-- Define the converse
def converse : Prop :=
  ∀ a b c : ℝ, b^2 = a * c → in_geometric_progression a b c

-- Define the inverse
def inverse : Prop :=
  ∀ a b c : ℝ, ¬(in_geometric_progression a b c) → b^2 ≠ a * c

-- Define the contrapositive
def contrapositive : Prop :=
  ∀ a b c : ℝ, b^2 ≠ a * c → ¬(in_geometric_progression a b c)

-- Theorem to prove
theorem exactly_one_true :
  (original_proposition ∧
   (converse ∨ inverse ∨ contrapositive) ∧
   ¬(converse ∧ inverse) ∧
   ¬(converse ∧ contrapositive) ∧
   ¬(inverse ∧ contrapositive)) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l1422_142250


namespace NUMINAMATH_CALUDE_function_composition_constant_l1422_142200

theorem function_composition_constant (b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 5 * x + b
  let g : ℝ → ℝ := λ x ↦ b * x + 3
  (∀ x, f (g x) = 15 * x + 18) :=
by
  sorry

end NUMINAMATH_CALUDE_function_composition_constant_l1422_142200


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1422_142234

theorem consecutive_integers_average (c d : ℝ) : 
  (c + (c+1) + (c+2) + (c+3) + (c+4) + (c+5)) / 6 = d →
  ((d-1) + d + (d+1) + (d+2) + (d+3) + (d+4)) / 6 = c + 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1422_142234


namespace NUMINAMATH_CALUDE_square_field_area_l1422_142217

/-- Represents the properties of a square field with barbed wire fencing --/
structure SquareField where
  side : ℝ
  wireRate : ℝ
  gateWidth : ℝ
  gateCount : ℕ
  totalCost : ℝ

/-- Calculates the area of the square field --/
def fieldArea (field : SquareField) : ℝ :=
  field.side * field.side

/-- Calculates the length of barbed wire needed --/
def wireLength (field : SquareField) : ℝ :=
  4 * field.side - field.gateWidth * field.gateCount

/-- Theorem stating the area of the square field given the conditions --/
theorem square_field_area (field : SquareField)
  (h1 : field.wireRate = 1)
  (h2 : field.gateWidth = 1)
  (h3 : field.gateCount = 2)
  (h4 : field.totalCost = 666)
  (h5 : wireLength field * field.wireRate = field.totalCost) :
  fieldArea field = 27889 := by
  sorry

#eval 167 * 167  -- To verify the result

end NUMINAMATH_CALUDE_square_field_area_l1422_142217


namespace NUMINAMATH_CALUDE_fair_attendance_difference_l1422_142229

theorem fair_attendance_difference : 
  ∀ (last_year : ℕ) (this_year : ℕ) (next_year : ℕ),
    this_year = 600 →
    next_year = 2 * this_year →
    last_year + this_year + next_year = 2800 →
    last_year < next_year →
    next_year - last_year = 200 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_difference_l1422_142229


namespace NUMINAMATH_CALUDE_factorization_equality_l1422_142240

theorem factorization_equality (a b : ℝ) : a * b^2 + 10 * a * b + 25 * a = a * (b + 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1422_142240


namespace NUMINAMATH_CALUDE_liam_picked_40_oranges_l1422_142210

/-- The number of oranges Liam picked -/
def liam_oranges : ℕ := sorry

/-- The price of 2 of Liam's oranges in cents -/
def liam_price : ℕ := 250

/-- The number of oranges Claire picked -/
def claire_oranges : ℕ := 30

/-- The price of each of Claire's oranges in cents -/
def claire_price : ℕ := 120

/-- The total amount saved in cents -/
def total_saved : ℕ := 8600

theorem liam_picked_40_oranges :
  liam_oranges = 40 ∧
  liam_price = 250 ∧
  claire_oranges = 30 ∧
  claire_price = 120 ∧
  total_saved = 8600 ∧
  (liam_oranges * liam_price / 2 + claire_oranges * claire_price = total_saved) :=
by sorry

end NUMINAMATH_CALUDE_liam_picked_40_oranges_l1422_142210


namespace NUMINAMATH_CALUDE_max_value_of_f_l1422_142263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

theorem max_value_of_f (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  (∃ x : ℝ, f a x = 5 * Real.exp (-3) ∧ ∀ y : ℝ, f a y ≤ 5 * Real.exp (-3)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1422_142263


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1422_142278

theorem quadratic_inequality (x : ℝ) : 9 * x^2 + 6 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1422_142278


namespace NUMINAMATH_CALUDE_triangle_max_sum_l1422_142212

theorem triangle_max_sum (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a = 3 →
  1 + (Real.tan A) / (Real.tan B) = 2 * c / b →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b > 0 ∧ c > 0 →
  (∀ b' c' : ℝ, b' > 0 ∧ c' > 0 →
    a^2 = b'^2 + c'^2 - 2 * b' * c' * Real.cos A →
    b' + c' ≤ b + c) →
  b + c = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_sum_l1422_142212


namespace NUMINAMATH_CALUDE_angle_value_in_triangle_l1422_142279

/-- Given a triangle ABC where ∠ABC = 120°, and two angles are 3y° and y°, prove that y = 30 -/
theorem angle_value_in_triangle (y : ℝ) : 
  (3 * y + y = 120) → y = 30 := by sorry

end NUMINAMATH_CALUDE_angle_value_in_triangle_l1422_142279


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l1422_142220

def i : ℂ := Complex.I

theorem complex_difference_magnitude : Complex.abs ((1 + i)^13 - (1 - i)^13) = 128 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l1422_142220


namespace NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l1422_142276

theorem odd_prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y m : ℤ), 1 + x^2 + y^2 = m * p ∧ 0 < m ∧ m < p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l1422_142276


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_for_60_bottles_6_samples_l1422_142209

/-- Systematic sampling interval for a given population and sample size -/
def systematicSamplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- The problem statement -/
theorem systematic_sampling_interval_for_60_bottles_6_samples :
  systematicSamplingInterval 60 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_for_60_bottles_6_samples_l1422_142209


namespace NUMINAMATH_CALUDE_area_difference_is_quarter_l1422_142265

/-- Represents a regular octagon with side length 1 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 1)

/-- Represents the cutting operation on the octagon -/
def cut (o : RegularOctagon) : ℝ × ℝ := sorry

/-- The difference in area between the larger and smaller parts after cutting -/
def area_difference (o : RegularOctagon) : ℝ :=
  let (larger, smaller) := cut o
  larger - smaller

/-- Theorem stating that the area difference is 1/4 -/
theorem area_difference_is_quarter (o : RegularOctagon) :
  area_difference o = 1/4 := by sorry

end NUMINAMATH_CALUDE_area_difference_is_quarter_l1422_142265


namespace NUMINAMATH_CALUDE_tire_circumference_l1422_142295

/-- The circumference of a tire given its rotation speed and the car's velocity -/
theorem tire_circumference
  (revolutions_per_minute : ℝ)
  (car_speed_kmh : ℝ)
  (h1 : revolutions_per_minute = 400)
  (h2 : car_speed_kmh = 48)
  : ∃ (circumference : ℝ), circumference = 2 := by
  sorry

end NUMINAMATH_CALUDE_tire_circumference_l1422_142295


namespace NUMINAMATH_CALUDE_polynomial_value_symmetry_l1422_142236

theorem polynomial_value_symmetry (a b c : ℝ) :
  ((-3)^5 * a + (-3)^3 * b + (-3) * c - 5 = 7) →
  (3^5 * a + 3^3 * b + 3 * c - 5 = -17) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_symmetry_l1422_142236


namespace NUMINAMATH_CALUDE_johns_age_l1422_142204

theorem johns_age (john : ℕ) (matt : ℕ) : 
  matt = 4 * john - 3 → 
  john + matt = 52 → 
  john = 11 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l1422_142204


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l1422_142286

theorem quadrilateral_diagonal_length 
  (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) (diagonal : ℝ) :
  offset1 = 10 →
  offset2 = 6 →
  area = 240 →
  area = (1 / 2) * diagonal * (offset1 + offset2) →
  diagonal = 30 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l1422_142286


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1422_142201

def A : Set ℝ := {x | x - 1 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ C → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ C) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1422_142201


namespace NUMINAMATH_CALUDE_fifth_term_value_l1422_142206

/-- Given a sequence {aₙ} where Sₙ denotes the sum of its first n terms and Sₙ = n² + 1,
    prove that a₅ = 9. -/
theorem fifth_term_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = n^2 + 1) : a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_value_l1422_142206


namespace NUMINAMATH_CALUDE_toms_profit_l1422_142292

def flour_needed : ℕ := 500
def flour_bag_size : ℕ := 50
def flour_bag_price : ℕ := 20
def salt_needed : ℕ := 10
def salt_price : ℚ := 1/5
def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 500

def total_cost : ℚ := 
  (flour_needed / flour_bag_size * flour_bag_price : ℚ) + 
  (salt_needed * salt_price) + 
  promotion_cost

def total_revenue : ℕ := ticket_price * tickets_sold

theorem toms_profit : 
  total_revenue - total_cost = 8798 := by sorry

end NUMINAMATH_CALUDE_toms_profit_l1422_142292


namespace NUMINAMATH_CALUDE_prob_three_students_same_group_l1422_142254

/-- The total number of students -/
def total_students : ℕ := 800

/-- The number of lunch groups -/
def num_groups : ℕ := 4

/-- The size of each lunch group -/
def group_size : ℕ := total_students / num_groups

/-- The probability of a student being assigned to a specific group -/
def prob_one_group : ℚ := 1 / num_groups

/-- The probability that three specific students are assigned to the same lunch group -/
theorem prob_three_students_same_group :
  (prob_one_group * prob_one_group : ℚ) = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_prob_three_students_same_group_l1422_142254


namespace NUMINAMATH_CALUDE_impossible_to_equalize_l1422_142225

/-- Represents the circular arrangement of six numbers -/
def CircularArrangement := Fin 6 → ℕ

/-- The initial arrangement of numbers from 1 to 6 -/
def initial_arrangement : CircularArrangement :=
  fun i => i.val + 1

/-- Adds 1 to three consecutive numbers in the arrangement -/
def add_to_consecutive (a : CircularArrangement) (start : Fin 6) : CircularArrangement :=
  fun i => if i = start ∨ i = start.succ ∨ i = start.succ.succ then a i + 1 else a i

/-- Subtracts 1 from three alternating numbers in the arrangement -/
def subtract_from_alternating (a : CircularArrangement) (start : Fin 6) : CircularArrangement :=
  fun i => if i = start ∨ i = start.succ.succ ∨ i = start.succ.succ.succ.succ then a i - 1 else a i

/-- Checks if all numbers in the arrangement are equal -/
def all_equal (a : CircularArrangement) : Prop :=
  ∀ i j : Fin 6, a i = a j

/-- Main theorem: It's impossible to equalize all numbers using the given operations -/
theorem impossible_to_equalize :
  ¬ ∃ (ops : List (CircularArrangement → CircularArrangement)),
    all_equal (ops.foldl (fun acc op => op acc) initial_arrangement) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_equalize_l1422_142225


namespace NUMINAMATH_CALUDE_jeans_extra_trips_l1422_142285

theorem jeans_extra_trips (total_trips : ℕ) (jeans_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : jeans_trips = 23) : 
  jeans_trips - (total_trips - jeans_trips) = 6 := by
  sorry

end NUMINAMATH_CALUDE_jeans_extra_trips_l1422_142285


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l1422_142275

/-- Given two internally tangent circles C₁ and C₂ with equations x² + y² + 2ax + a² - 4 = 0 and 
    x² + y² - 2by + b² - 1 = 0 respectively, where a, b ∈ ℝ and ab ≠ 0, 
    the minimum value of 1/a² + 1/b² is 9 -/
theorem min_value_sum_reciprocal_squares (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0) →
  a ≠ 0 →
  b ≠ 0 →
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 ≠ 0 ∨ x^2 + y^2 - 2*b*y + b^2 - 1 ≠ 0 ∨ 
    (x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ x^2 + y^2 - 2*b*y + b^2 - 1 = 0)) →
  (1 / a^2 + 1 / b^2) ≥ 9 :=
by sorry

#check min_value_sum_reciprocal_squares

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l1422_142275


namespace NUMINAMATH_CALUDE_unique_coin_combination_l1422_142205

/-- Represents a coin with a value in kopecks -/
structure Coin where
  value : ℕ

/-- Represents a wallet containing two coins -/
structure Wallet where
  coin1 : Coin
  coin2 : Coin

/-- The total value of coins in a wallet -/
def walletValue (w : Wallet) : ℕ := w.coin1.value + w.coin2.value

/-- Predicate to check if a coin is not a five-kopeck coin -/
def isNotFiveKopecks (c : Coin) : Prop := c.value ≠ 5

/-- Theorem stating the only possible combination of coins -/
theorem unique_coin_combination (w : Wallet) 
  (h1 : walletValue w = 15)
  (h2 : isNotFiveKopecks w.coin1 ∨ isNotFiveKopecks w.coin2) :
  (w.coin1.value = 5 ∧ w.coin2.value = 10) ∨ (w.coin1.value = 10 ∧ w.coin2.value = 5) :=
sorry

end NUMINAMATH_CALUDE_unique_coin_combination_l1422_142205


namespace NUMINAMATH_CALUDE_budget_allocation_l1422_142256

theorem budget_allocation (salaries utilities equipment supplies transportation : ℝ) 
  (h1 : salaries = 60)
  (h2 : utilities = 5)
  (h3 : equipment = 4)
  (h4 : supplies = 2)
  (h5 : transportation = 72 / 360 * 100)
  (h6 : salaries + utilities + equipment + supplies + transportation < 100) :
  100 - (salaries + utilities + equipment + supplies + transportation) = 9 := by
sorry

end NUMINAMATH_CALUDE_budget_allocation_l1422_142256


namespace NUMINAMATH_CALUDE_problem_solution_l1422_142249

theorem problem_solution (N : ℚ) : 
  (4 / 5 : ℚ) * (3 / 8 : ℚ) * N = 24 → (5 / 2 : ℚ) * N = 200 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1422_142249


namespace NUMINAMATH_CALUDE_common_solution_y_values_l1422_142222

theorem common_solution_y_values : 
  ∃ y₁ y₂ : ℝ, 
    (∀ x y : ℝ, x^2 + y^2 - 9 = 0 ∧ x^2 - 4*y + 8 = 0 → y = y₁ ∨ y = y₂) ∧
    y₁ = -2 + Real.sqrt 21 ∧
    y₂ = -2 - Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_common_solution_y_values_l1422_142222


namespace NUMINAMATH_CALUDE_expression_equality_l1422_142271

theorem expression_equality (y : ℝ) (Q : ℝ) (h : 5 * (3 * y - 7 * Real.pi) = Q) :
  10 * (6 * y - 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1422_142271


namespace NUMINAMATH_CALUDE_mothers_day_discount_percentage_l1422_142298

/-- Calculates the discount percentage for a Mother's day special at a salon -/
theorem mothers_day_discount_percentage 
  (regular_price : ℝ) 
  (num_services : ℕ) 
  (discounted_total : ℝ) 
  (h1 : regular_price = 40)
  (h2 : num_services = 5)
  (h3 : discounted_total = 150) : 
  (1 - discounted_total / (regular_price * num_services)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_discount_percentage_l1422_142298


namespace NUMINAMATH_CALUDE_average_of_sample_l1422_142216

def sample_average (x : Fin 10 → ℝ) (a b : ℝ) : Prop :=
  (x 0 + x 1 + x 2) / 3 = a ∧
  (x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 7 = b

theorem average_of_sample (x : Fin 10 → ℝ) (a b : ℝ) 
  (h : sample_average x a b) : 
  (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 10 = (3 * a + 7 * b) / 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_sample_l1422_142216


namespace NUMINAMATH_CALUDE_division_scaling_certain_number_proof_l1422_142291

theorem division_scaling (a b c : ℝ) (h : a / b = c) : (100 * a) / (100 * b) = c := by
  sorry

theorem certain_number_proof :
  29.94 / 1.45 = 17.7 → 2994 / 14.5 = 17.7 := by
  sorry

end NUMINAMATH_CALUDE_division_scaling_certain_number_proof_l1422_142291


namespace NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l1422_142299

-- Problem 1
theorem max_value_problem (x : ℝ) (h : 0 < x ∧ x < 2) : x * (4 - 2*x) ≤ 2 := by
  sorry

-- Problem 2
theorem min_value_problem (x : ℝ) (h : x > 3/2) : x + 8 / (2*x - 3) ≥ 11/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l1422_142299


namespace NUMINAMATH_CALUDE_cross_section_area_less_than_half_face_l1422_142296

/-- A cube with an inscribed sphere and a triangular cross-section touching the sphere -/
structure CubeWithSphereAndCrossSection where
  /-- Side length of the cube -/
  a : ℝ
  /-- Assumption that the cube has positive side length -/
  a_pos : 0 < a
  /-- The triangular cross-section touches the inscribed sphere -/
  touches_sphere : Bool

/-- The area of the triangular cross-section is less than half the area of the cube face -/
theorem cross_section_area_less_than_half_face (cube : CubeWithSphereAndCrossSection) :
  ∃ (area : ℝ), area < (1/2) * cube.a^2 ∧ 
  (∀ (cross_section_area : ℝ), cross_section_area ≤ area) :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_less_than_half_face_l1422_142296


namespace NUMINAMATH_CALUDE_selected_is_sample_size_l1422_142245

/-- Represents a statistical study -/
structure StatisticalStudy where
  population_size : ℕ
  selected_size : ℕ
  selected_size_le_population : selected_size ≤ population_size

/-- Definition of sample size -/
def sample_size (study : StatisticalStudy) : ℕ := study.selected_size

theorem selected_is_sample_size (study : StatisticalStudy) 
  (h1 : study.population_size = 3000) 
  (h2 : study.selected_size = 100) : 
  sample_size study = study.selected_size :=
by
  sorry

#check selected_is_sample_size

end NUMINAMATH_CALUDE_selected_is_sample_size_l1422_142245


namespace NUMINAMATH_CALUDE_unique_a_value_l1422_142266

theorem unique_a_value (a : ℕ) : 
  (∃ k : ℕ, 88 * a = 2 * k + 1) →  -- 88a is odd
  (∃ m : ℕ, 88 * a = 3 * m) →      -- 88a is a multiple of 3
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_a_value_l1422_142266


namespace NUMINAMATH_CALUDE_smallest_base_for_124_l1422_142235

theorem smallest_base_for_124 (b : ℕ) : b ≥ 5 ↔ b ^ 2 ≤ 124 ∧ 124 < b ^ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_124_l1422_142235


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l1422_142219

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 ∧ 4 ∣ x ∧ x^3 < 1728 → x ≤ 8 ∧ ∃ y : ℕ, y > 0 ∧ 4 ∣ y ∧ y^3 < 1728 ∧ y = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l1422_142219


namespace NUMINAMATH_CALUDE_complex_number_problem_l1422_142238

def z (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_problem (b : ℝ) 
  (h : ∃ (y : ℝ), (1 + 3 * Complex.I) * z b = y * Complex.I) :
  z b = 3 + Complex.I ∧ Complex.abs (z b / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1422_142238


namespace NUMINAMATH_CALUDE_loaves_sold_l1422_142227

/-- The number of loaves sold in a supermarket given initial, delivered, and final counts. -/
theorem loaves_sold (initial : ℕ) (delivered : ℕ) (final : ℕ) :
  initial = 2355 →
  delivered = 489 →
  final = 2215 →
  initial + delivered - final = 629 := by
  sorry

#check loaves_sold

end NUMINAMATH_CALUDE_loaves_sold_l1422_142227


namespace NUMINAMATH_CALUDE_game_value_conversion_l1422_142244

/-- Calculates the final value of sold games in USD after multiple currency conversions and fees --/
theorem game_value_conversion (initial_value : ℝ) (usd_to_eur_rate : ℝ) (eur_to_usd_fee : ℝ)
  (value_increase : ℝ) (eur_to_jpy_rate : ℝ) (eur_to_jpy_fee : ℝ) (sell_percentage : ℝ)
  (japan_tax_rate : ℝ) (jpy_to_usd_rate : ℝ) (jpy_to_usd_fee : ℝ) :
  initial_value = 200 →
  usd_to_eur_rate = 0.85 →
  eur_to_usd_fee = 0.03 →
  value_increase = 3 →
  eur_to_jpy_rate = 130 →
  eur_to_jpy_fee = 0.02 →
  sell_percentage = 0.4 →
  japan_tax_rate = 0.1 →
  jpy_to_usd_rate = 0.0085 →
  jpy_to_usd_fee = 0.01 →
  ∃ final_value : ℝ, abs (final_value - 190.93) < 0.01 ∧
  final_value = initial_value * usd_to_eur_rate * (1 - eur_to_usd_fee) * value_increase *
                eur_to_jpy_rate * (1 - eur_to_jpy_fee) * sell_percentage *
                (1 - japan_tax_rate) * jpy_to_usd_rate * (1 - jpy_to_usd_fee) := by
  sorry

end NUMINAMATH_CALUDE_game_value_conversion_l1422_142244


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l1422_142294

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = (1 / 2) * a n

theorem geometric_sequence_a1 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : a 4 = 8) : 
  a 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l1422_142294


namespace NUMINAMATH_CALUDE_triple_coverage_theorem_l1422_142253

/-- Represents a rectangular rug with given dimensions -/
structure Rug where
  width : ℝ
  length : ℝ

/-- Represents the arrangement of rugs in the auditorium -/
structure AuditoriumArrangement where
  auditorium_size : ℝ
  rug1 : Rug
  rug2 : Rug
  rug3 : Rug

/-- Calculates the area covered by all three rugs simultaneously -/
def triple_coverage_area (arrangement : AuditoriumArrangement) : ℝ :=
  sorry

/-- The specific arrangement in the problem -/
def problem_arrangement : AuditoriumArrangement :=
  { auditorium_size := 10
  , rug1 := { width := 6, length := 8 }
  , rug2 := { width := 6, length := 6 }
  , rug3 := { width := 5, length := 7 }
  }

theorem triple_coverage_theorem :
  triple_coverage_area problem_arrangement = 6 := by sorry

end NUMINAMATH_CALUDE_triple_coverage_theorem_l1422_142253


namespace NUMINAMATH_CALUDE_news_spread_theorem_l1422_142290

/-- Represents the spread of news in a village -/
structure NewsSpread where
  residents : ℕ
  start_date : ℕ
  current_date : ℕ
  informed_residents : Finset ℕ

/-- The number of days since the news started spreading -/
def days_passed (ns : NewsSpread) : ℕ :=
  ns.current_date - ns.start_date

/-- Predicate to check if all residents are informed -/
def all_informed (ns : NewsSpread) : Prop :=
  ns.informed_residents.card = ns.residents

theorem news_spread_theorem (ns : NewsSpread) 
  (h_residents : ns.residents = 20)
  (h_start : ns.start_date = 1) :
  (∃ d₁ d₂, d₁ ≤ 15 ∧ d₂ ≥ 18 ∧ days_passed {ns with current_date := ns.start_date + d₁} < ns.residents ∧
            all_informed {ns with current_date := ns.start_date + d₂}) ∧
  (∀ d, d > 20 → all_informed {ns with current_date := ns.start_date + d}) :=
by sorry

end NUMINAMATH_CALUDE_news_spread_theorem_l1422_142290


namespace NUMINAMATH_CALUDE_multiply_powers_of_a_l1422_142289

theorem multiply_powers_of_a (a : ℝ) : 5 * a^3 * (3 * a^3) = 15 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_a_l1422_142289


namespace NUMINAMATH_CALUDE_star_sum_larger_than_emilio_sum_l1422_142248

def star_numbers : List ℕ := List.range 50

def emilio_numbers : List ℕ :=
  star_numbers.map (fun n => 
    let tens := n / 10
    let ones := n % 10
    if tens = 2 ∨ tens = 3 then
      (if tens = 2 then 5 else 5) * 10 + ones
    else if ones = 2 ∨ ones = 3 then
      tens * 10 + 5
    else
      n
  )

theorem star_sum_larger_than_emilio_sum :
  (star_numbers.sum - emilio_numbers.sum) = 550 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_larger_than_emilio_sum_l1422_142248


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1422_142297

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1422_142297


namespace NUMINAMATH_CALUDE_chess_competition_games_l1422_142214

theorem chess_competition_games (W M : ℕ) 
  (h1 : W * (W - 1) / 2 = 45)
  (h2 : M * (M - 1) / 2 = 190) :
  W * M = 200 := by
  sorry

end NUMINAMATH_CALUDE_chess_competition_games_l1422_142214


namespace NUMINAMATH_CALUDE_matrix_product_result_l1422_142243

def matrix_product (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  (List.range n).foldl
    (λ acc i => acc * !![1, 2*(i+1); 0, 1])
    (!![1, 0; 0, 1])

theorem matrix_product_result :
  matrix_product 50 = !![1, 2550; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_product_result_l1422_142243


namespace NUMINAMATH_CALUDE_average_of_abc_l1422_142267

theorem average_of_abc (a b c : ℝ) : 
  (4 + 6 + 9 + a + b + c) / 6 = 18 → (a + b + c) / 3 = 29 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_l1422_142267


namespace NUMINAMATH_CALUDE_marlon_lollipops_l1422_142232

/-- The number of lollipops Marlon had initially -/
def initial_lollipops : ℕ := 42

/-- The fraction of lollipops Marlon gave to Emily -/
def emily_fraction : ℚ := 2/3

/-- The number of lollipops Marlon kept for himself -/
def marlon_kept : ℕ := 4

/-- The number of lollipops Lou received -/
def lou_received : ℕ := 10

theorem marlon_lollipops :
  (initial_lollipops : ℚ) * (1 - emily_fraction) = (marlon_kept + lou_received : ℚ) := by
  sorry

#check marlon_lollipops

end NUMINAMATH_CALUDE_marlon_lollipops_l1422_142232


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1422_142274

/-- Given that (m+1)x^(m^2+1) - 2x - 5 = 0 is a quadratic equation in x 
    and m + 1 ≠ 0, prove that m = 1 -/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∃ a b c : ℝ, ∀ x : ℝ, (m + 1) * x^(m^2 + 1) - 2*x - 5 = a*x^2 + b*x + c) ∧ 
  (m + 1 ≠ 0) → 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1422_142274


namespace NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_l1422_142280

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_divisible_by_225 :
  ∃ (n : ℕ), is_binary_number n ∧ 225 ∣ n ∧
  ∀ (m : ℕ), is_binary_number m → 225 ∣ m → n ≤ m :=
by
  -- The proof would go here
  sorry

#eval (11111111100 : ℕ).digits 10  -- To verify the number in base 10
#eval 11111111100 % 225  -- To verify divisibility by 225

end NUMINAMATH_CALUDE_smallest_binary_divisible_by_225_l1422_142280


namespace NUMINAMATH_CALUDE_ap_terms_count_l1422_142257

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Even n → 
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 36 →
  (n / 2 : ℚ) * (2 * a + (n - 1) * d) = 44 →
  a + (n - 1) * d - a = 12 →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_ap_terms_count_l1422_142257


namespace NUMINAMATH_CALUDE_segment_ratio_l1422_142203

/-- Given four points A, B, C, D on a line segment, 
    if AB : BC = 1 : 2 and BC : CD = 8 : 5, 
    then AB : BD = 4 : 13 -/
theorem segment_ratio (A B C D : ℝ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (ratio1 : (B - A) / (C - B) = 1 / 2)
  (ratio2 : (C - B) / (D - C) = 8 / 5) :
  (B - A) / (D - B) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l1422_142203


namespace NUMINAMATH_CALUDE_composite_product_quotient_l1422_142251

/-- The first ten positive composite integers -/
def first_ten_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]

/-- The product of the first five positive composite integers -/
def product_first_five : ℕ := (first_ten_composites.take 5).prod

/-- The product of the next five positive composite integers -/
def product_next_five : ℕ := (first_ten_composites.drop 5).prod

/-- Theorem stating that the quotient of the product of the first five positive composite integers
    divided by the product of the next five composite integers equals 1/42 -/
theorem composite_product_quotient :
  (product_first_five : ℚ) / (product_next_five : ℚ) = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_quotient_l1422_142251


namespace NUMINAMATH_CALUDE_lucille_earnings_lucille_earnings_proof_l1422_142224

/-- Calculates the amount of money Lucille has left after weeding and buying a soda -/
theorem lucille_earnings (cents_per_weed : ℕ) (flower_bed_weeds : ℕ) (vegetable_patch_weeds : ℕ) 
  (grass_weeds : ℕ) (soda_cost : ℕ) : ℕ :=
  let total_weeds := flower_bed_weeds + vegetable_patch_weeds + grass_weeds / 2
  let total_earnings := total_weeds * cents_per_weed
  total_earnings - soda_cost

/-- Proves that Lucille has 147 cents left after weeding and buying a soda -/
theorem lucille_earnings_proof :
  lucille_earnings 6 11 14 32 99 = 147 := by
  sorry

end NUMINAMATH_CALUDE_lucille_earnings_lucille_earnings_proof_l1422_142224


namespace NUMINAMATH_CALUDE_system_solution_l1422_142288

theorem system_solution (a b : ℝ) : 
  (a * 1 - b * 2 = -1) → 
  (a * 1 + b * 2 = 7) → 
  3 * a - 4 * b = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1422_142288


namespace NUMINAMATH_CALUDE_solve_for_c_l1422_142230

theorem solve_for_c (a b c : ℤ) 
  (sum_eq : a + b + c = 60)
  (a_eq : a = (b + c) / 3)
  (b_eq : b = (a + c) / 5) :
  c = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l1422_142230


namespace NUMINAMATH_CALUDE_square_perimeter_l1422_142277

theorem square_perimeter (s : ℝ) (h : s > 0) :
  (5 * s / 2 = 40) → (4 * s = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1422_142277


namespace NUMINAMATH_CALUDE_logan_tower_height_l1422_142242

/-- The height of the city's water tower in meters -/
def city_tower_height : ℝ := 60

/-- The volume of water the city's water tower can hold in liters -/
def city_tower_volume : ℝ := 150000

/-- The volume of water Logan's miniature water tower can hold in liters -/
def miniature_tower_volume : ℝ := 0.15

/-- The height of Logan's miniature water tower in meters -/
def miniature_tower_height : ℝ := 0.6

/-- Theorem stating that the height of Logan's miniature tower should be 0.6 meters -/
theorem logan_tower_height : miniature_tower_height = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_logan_tower_height_l1422_142242


namespace NUMINAMATH_CALUDE_at_least_two_first_class_products_l1422_142255

def total_products : ℕ := 9
def first_class : ℕ := 4
def second_class : ℕ := 3
def third_class : ℕ := 2
def products_to_draw : ℕ := 4

theorem at_least_two_first_class_products :
  (Nat.choose first_class 2 * Nat.choose (total_products - first_class) 2 +
   Nat.choose first_class 3 * Nat.choose (total_products - first_class) 1 +
   Nat.choose first_class 4 * Nat.choose (total_products - first_class) 0) =
  (Nat.choose total_products products_to_draw -
   Nat.choose (second_class + third_class) products_to_draw -
   (Nat.choose first_class 1 * Nat.choose (second_class + third_class) 3)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_first_class_products_l1422_142255


namespace NUMINAMATH_CALUDE_total_retail_price_calculation_l1422_142264

def calculate_retail_price (wholesale_price : ℝ) (profit_margin : ℝ) (discount : ℝ) : ℝ :=
  let retail_before_discount := wholesale_price * (1 + profit_margin)
  retail_before_discount * (1 - discount)

theorem total_retail_price_calculation (P Q R : ℝ) 
  (h1 : P = 90) (h2 : Q = 120) (h3 : R = 150) : 
  calculate_retail_price P 0.2 0.1 + 
  calculate_retail_price Q 0.25 0.15 + 
  calculate_retail_price R 0.3 0.2 = 380.7 := by
  sorry

#eval calculate_retail_price 90 0.2 0.1 + 
      calculate_retail_price 120 0.25 0.15 + 
      calculate_retail_price 150 0.3 0.2

end NUMINAMATH_CALUDE_total_retail_price_calculation_l1422_142264


namespace NUMINAMATH_CALUDE_division_evaluation_l1422_142202

theorem division_evaluation : 250 / (5 + 12 * 3^2) = 250 / 113 := by sorry

end NUMINAMATH_CALUDE_division_evaluation_l1422_142202


namespace NUMINAMATH_CALUDE_percentage_of_4_to_50_percentage_of_4_to_50_proof_l1422_142281

theorem percentage_of_4_to_50 : ℝ → Prop :=
  fun x => (4 / 50 * 100 = x) → x = 8

-- The proof goes here
theorem percentage_of_4_to_50_proof : percentage_of_4_to_50 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_4_to_50_percentage_of_4_to_50_proof_l1422_142281


namespace NUMINAMATH_CALUDE_housewife_spending_l1422_142231

theorem housewife_spending (initial_amount : ℚ) (spent_fraction : ℚ) :
  initial_amount = 150 →
  spent_fraction = 2/3 →
  initial_amount * (1 - spent_fraction) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_housewife_spending_l1422_142231


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_l1422_142259

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4444^4444 -/
def A : ℕ := sum_of_digits (4444^4444)

/-- B is the sum of digits of A -/
def B : ℕ := sum_of_digits A

/-- Theorem: The sum of digits of B is 7 -/
theorem sum_of_digits_of_B : sum_of_digits B = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_l1422_142259


namespace NUMINAMATH_CALUDE_inequality_proof_l1422_142269

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1422_142269


namespace NUMINAMATH_CALUDE_lcm_36_100_l1422_142226

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l1422_142226


namespace NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l1422_142247

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem solution_to_logarithmic_equation :
  ∃ x : ℝ, lg (3 * x + 4) = 1 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l1422_142247


namespace NUMINAMATH_CALUDE_even_integers_in_pascal_triangle_l1422_142237

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- Counts the number of even integers in the first n rows of Pascal's Triangle -/
def countEvenIntegers (pt : PascalTriangle n) : ℕ := sorry

theorem even_integers_in_pascal_triangle :
  ∀ (pt10 : PascalTriangle 10) (pt15 : PascalTriangle 15),
    countEvenIntegers pt10 = 22 →
    countEvenIntegers pt15 = 53 := by sorry

end NUMINAMATH_CALUDE_even_integers_in_pascal_triangle_l1422_142237


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l1422_142283

def circle_equation (x y : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 36

theorem max_sum_on_circle :
  ∃ (max : ℤ),
    (∀ x y : ℤ, circle_equation x y → x + y ≤ max) ∧
    (∃ x y : ℤ, circle_equation x y ∧ x + y = max) ∧
    max = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l1422_142283


namespace NUMINAMATH_CALUDE_special_square_area_special_square_area_is_64_l1422_142287

/-- A square in the coordinate plane with specific properties -/
structure SpecialSquare where
  verticesOnY2 : ℝ × ℝ → Prop
  verticesOnY10 : ℝ × ℝ → Prop
  sidesParallelOrPerpendicular : Prop

/-- The area of the special square is 64 -/
theorem special_square_area (s : SpecialSquare) : ℝ :=
  64

/-- The main theorem stating that the area of the special square is 64 -/
theorem special_square_area_is_64 (s : SpecialSquare) : special_square_area s = 64 := by
  sorry

end NUMINAMATH_CALUDE_special_square_area_special_square_area_is_64_l1422_142287


namespace NUMINAMATH_CALUDE_triangle_inequality_l1422_142207

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  t.a^2 * (t.b + t.c - t.a) + t.b^2 * (t.c + t.a - t.b) + t.c^2 * (t.a + t.b - t.c) ≤ 3 * t.a * t.b * t.c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1422_142207


namespace NUMINAMATH_CALUDE_octagon_proof_l1422_142261

theorem octagon_proof (n : ℕ) (h : n > 2) : 
  (n * (n - 3)) / 2 = n + 2 * (n - 2) → n = 8 := by
sorry

end NUMINAMATH_CALUDE_octagon_proof_l1422_142261


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1422_142208

theorem arithmetic_sequence_count :
  let first : ℤ := 162
  let last : ℤ := 42
  let diff : ℤ := -3
  let count := (last - first) / diff + 1
  count = 41 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1422_142208


namespace NUMINAMATH_CALUDE_acorns_given_calculation_l1422_142213

/-- The number of acorns Megan gave to her sister -/
def acorns_given : ℕ := sorry

/-- The initial number of acorns Megan had -/
def initial_acorns : ℕ := 16

/-- The number of acorns Megan has left -/
def acorns_left : ℕ := 9

/-- Theorem stating that the number of acorns given is the difference between
    the initial number and the number left -/
theorem acorns_given_calculation : acorns_given = initial_acorns - acorns_left := by
  sorry

end NUMINAMATH_CALUDE_acorns_given_calculation_l1422_142213


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l1422_142272

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101_equals_13 :
  binary_to_decimal [true, false, true, true] = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l1422_142272


namespace NUMINAMATH_CALUDE_find_a_min_value_of_sum_l1422_142260

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (I)
theorem find_a :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) →
  (∃ a : ℝ, a = 2) :=
sorry

-- Theorem for part (II)
theorem min_value_of_sum (x : ℝ) :
  ∃ m : ℝ, m = 5/3 ∧ ∀ x : ℝ, f 2 (3*x) + f 2 (x+3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_find_a_min_value_of_sum_l1422_142260


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_zero_l1422_142268

theorem min_sum_squares (x y s : ℝ) (h : x + y + s = 0) : 
  ∀ a b c : ℝ, a + b + c = 0 → x^2 + y^2 + s^2 ≤ a^2 + b^2 + c^2 :=
by
  sorry

theorem min_sum_squares_zero (x y s : ℝ) (h : x + y + s = 0) : 
  ∃ a b c : ℝ, a + b + c = 0 ∧ a^2 + b^2 + c^2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_zero_l1422_142268


namespace NUMINAMATH_CALUDE_division_equality_l1422_142273

theorem division_equality : 204 / 12.75 = 16 := by
  -- Given condition
  have h1 : 2.04 / 1.275 = 1.6 := by sorry
  
  -- Define the scaling factor
  let scale : ℝ := 100 / 10
  
  -- Prove that 204 / 12.75 = 16
  sorry

end NUMINAMATH_CALUDE_division_equality_l1422_142273


namespace NUMINAMATH_CALUDE_max_area_quadrilateral_l1422_142258

/-- Given a rectangle ABCD with AB = c and AD = d, and points E on AB and F on AD
    such that AE = AF = x, the maximum area of quadrilateral CDFE is (c + d)^2 / 8. -/
theorem max_area_quadrilateral (c d : ℝ) (h_c : c > 0) (h_d : d > 0) :
  ∃ x : ℝ, 0 < x ∧ x < min c d ∧
    ∀ y : ℝ, 0 < y ∧ y < min c d →
      x * (c + d - 2*x) / 2 ≥ y * (c + d - 2*y) / 2 ∧
      x * (c + d - 2*x) / 2 = (c + d)^2 / 8 :=
by sorry


end NUMINAMATH_CALUDE_max_area_quadrilateral_l1422_142258


namespace NUMINAMATH_CALUDE_complex_division_result_l1422_142282

theorem complex_division_result (z : ℂ) (h : z = 1 + I) : z / (1 - I) = I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1422_142282


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1422_142262

/-- The number of apples handed out to students -/
def apples_to_students : ℕ := 30

/-- The number of pies made -/
def number_of_pies : ℕ := 7

/-- The number of apples required for each pie -/
def apples_per_pie : ℕ := 8

/-- The total number of apples in the cafeteria initially -/
def total_apples : ℕ := apples_to_students + number_of_pies * apples_per_pie

theorem cafeteria_apples : total_apples = 86 := by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1422_142262


namespace NUMINAMATH_CALUDE_difference_of_squares_multiplication_l1422_142228

theorem difference_of_squares_multiplication (a b : ℕ) :
  58 * 42 = 2352 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_multiplication_l1422_142228


namespace NUMINAMATH_CALUDE_function_nature_l1422_142218

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem function_nature (f : ℝ → ℝ) 
  (h1 : f 0 ≠ 0)
  (h2 : ∀ x₁ x₂ : ℝ, f x₁ + f x₂ = 2 * f ((x₁ + x₂) / 2) * f ((x₁ - x₂) / 2)) :
  is_even f ∧ ¬ is_odd f := by
sorry

end NUMINAMATH_CALUDE_function_nature_l1422_142218


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equation_l1422_142221

/-- Given a curve C with polar coordinate equation ρ sin (θ - π/4) = √2,
    where the origin is at the pole and the polar axis lies on the x-axis
    in a rectangular coordinate system, prove that the rectangular
    coordinate equation of C is x - y + 2 = 0. -/
theorem polar_to_rectangular_equation :
  ∀ (ρ θ x y : ℝ),
  (ρ * Real.sin (θ - π/4) = Real.sqrt 2) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x - y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equation_l1422_142221


namespace NUMINAMATH_CALUDE_logarithm_sum_equals_two_l1422_142223

theorem logarithm_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equals_two_l1422_142223


namespace NUMINAMATH_CALUDE_sallys_quarters_l1422_142239

/-- Given that Sally had 760 quarters initially and spent 418 quarters,
    prove that she now has 342 quarters. -/
theorem sallys_quarters (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
    (h1 : initial = 760)
    (h2 : spent = 418)
    (h3 : remaining = initial - spent) :
  remaining = 342 := by
  sorry

end NUMINAMATH_CALUDE_sallys_quarters_l1422_142239


namespace NUMINAMATH_CALUDE_f_properties_l1422_142233

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * abs (x - a)

theorem f_properties :
  (∀ x : ℝ, (f (-1) x = 1 ↔ x ≤ -1 ∨ x = 1)) ∧
  (∀ a : ℝ, (StrictMono (f a) ↔ a ≥ 1/3)) ∧
  (∀ a : ℝ, a < 1 → (∀ x : ℝ, f a x ≥ 2*x - 3) ↔ a ∈ Set.Icc (-3) 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1422_142233
