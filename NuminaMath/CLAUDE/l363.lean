import Mathlib

namespace NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l363_36330

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_seven_fourths_squared_l363_36330


namespace NUMINAMATH_CALUDE_transport_tax_calculation_l363_36326

/-- Calculates the transport tax for a vehicle -/
def transportTax (horsepower : ℕ) (taxRate : ℕ) (ownershipMonths : ℕ) : ℕ :=
  horsepower * taxRate * ownershipMonths / 12

/-- Proves that the transport tax for the given conditions is 2000 rubles -/
theorem transport_tax_calculation :
  transportTax 150 20 8 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_transport_tax_calculation_l363_36326


namespace NUMINAMATH_CALUDE_solve_apple_problem_l363_36321

def apple_problem (initial_apples : ℕ) (pears_difference : ℕ) (pears_bought : ℕ) (final_total : ℕ) : Prop :=
  let initial_pears : ℕ := initial_apples + pears_difference
  let new_pears : ℕ := initial_pears + pears_bought
  let apples_sold : ℕ := initial_apples + new_pears - final_total
  apples_sold = 599

theorem solve_apple_problem :
  apple_problem 1238 374 276 2527 :=
by sorry

end NUMINAMATH_CALUDE_solve_apple_problem_l363_36321


namespace NUMINAMATH_CALUDE_kelly_games_l363_36310

theorem kelly_games (initial_games given_away left : ℕ) : 
  given_away = 99 → left = 22 → initial_games = given_away + left :=
by sorry

end NUMINAMATH_CALUDE_kelly_games_l363_36310


namespace NUMINAMATH_CALUDE_class_average_weight_l363_36332

/-- Given two sections A and B in a class, calculate the average weight of the whole class. -/
theorem class_average_weight 
  (students_A : ℕ) 
  (students_B : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_B : ℝ) : 
  students_A = 50 → 
  students_B = 40 → 
  avg_weight_A = 50 → 
  avg_weight_B = 70 → 
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 
    (50 * 50 + 70 * 40) / (50 + 40) := by
  sorry

#eval (50 * 50 + 70 * 40) / (50 + 40)  -- This will evaluate to approximately 58.89

end NUMINAMATH_CALUDE_class_average_weight_l363_36332


namespace NUMINAMATH_CALUDE_train_length_calculation_l363_36316

theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 27) : ∃ L : ℝ,
  L = 37.5 ∧ 
  2 * L = (v_fast - v_slow) * (5 / 18) * t :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l363_36316


namespace NUMINAMATH_CALUDE_hotel_cost_calculation_l363_36354

def trip_expenses (savings flight_cost food_cost remaining : ℕ) : Prop :=
  ∃ hotel_cost : ℕ, 
    savings = flight_cost + food_cost + hotel_cost + remaining

theorem hotel_cost_calculation (savings flight_cost food_cost remaining : ℕ) 
  (h : trip_expenses savings flight_cost food_cost remaining) : 
  ∃ hotel_cost : ℕ, hotel_cost = 800 ∧ trip_expenses 6000 1200 3000 1000 := by
  sorry

end NUMINAMATH_CALUDE_hotel_cost_calculation_l363_36354


namespace NUMINAMATH_CALUDE_factorization_equality_l363_36369

theorem factorization_equality (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4*x*y = (x*y - 1 + x + y) * (x*y - 1 - x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l363_36369


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l363_36361

/-- Given a cone with slant height 13 cm and height 12 cm, its lateral surface area is 65π cm². -/
theorem cone_lateral_surface_area (s h r : ℝ) : 
  s = 13 → h = 12 → s^2 = h^2 + r^2 → (π * r * s : ℝ) = 65 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l363_36361


namespace NUMINAMATH_CALUDE_min_vertical_distance_l363_36335

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

theorem min_vertical_distance : 
  ∃ (x₀ : ℝ), ∀ (x : ℝ), |f x - g x| ≥ |f x₀ - g x₀| ∧ |f x₀ - g x₀| = 10 :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l363_36335


namespace NUMINAMATH_CALUDE_right_triangle_division_l363_36307

theorem right_triangle_division (n : ℝ) (h : n > 0) :
  ∀ (rect_area rect_short rect_long small_triangle1_area : ℝ),
    rect_short > 0 →
    rect_long > 0 →
    rect_area > 0 →
    small_triangle1_area > 0 →
    rect_long = 3 * rect_short →
    rect_area = rect_short * rect_long →
    small_triangle1_area = n * rect_area →
    ∃ (small_triangle2_area : ℝ),
      small_triangle2_area > 0 ∧
      small_triangle2_area / rect_area = 1 / (4 * n) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_division_l363_36307


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l363_36387

def x : List ℝ := [1, 2, 3, 4, 5]
def y : List ℝ := [3, 5.3, 6.9, 9.1, 10.8]
def U : List ℝ := [1, 2, 3, 4, 5]
def V : List ℝ := [12.7, 10.2, 7, 3.6, 1]

def r1 : ℝ := sorry
def r2 : ℝ := sorry

theorem correlation_coefficient_comparison : r2 < 0 ∧ 0 < r1 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l363_36387


namespace NUMINAMATH_CALUDE_parallelogram_area_l363_36362

/-- The area of a parallelogram with base 22 cm and height 21 cm is 462 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 22
  let height : ℝ := 21
  let area := base * height
  area = 462 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l363_36362


namespace NUMINAMATH_CALUDE_median_of_special_list_l363_36309

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list length -/
def list_length : ℕ := triangular_number 150

/-- The median position -/
def median_position : ℕ := (list_length + 1) / 2

/-- The cumulative count up to n -/
def cumulative_count (n : ℕ) : ℕ := triangular_number n

theorem median_of_special_list : ∃ (n : ℕ), n = 106 ∧ 
  cumulative_count (n - 1) < median_position ∧ 
  cumulative_count n ≥ median_position := by sorry

end NUMINAMATH_CALUDE_median_of_special_list_l363_36309


namespace NUMINAMATH_CALUDE_chocolate_division_theorem_l363_36337

/-- Represents a piece of chocolate -/
structure ChocolatePiece where
  area : ℝ

/-- Represents the chocolate bar -/
structure ChocolateBar where
  length : ℝ
  width : ℝ
  pieces : Fin 4 → ChocolatePiece

/-- The chocolate bar is divided as described in the problem -/
def is_divided_as_described (bar : ChocolateBar) : Prop :=
  bar.length = 6 ∧ bar.width = 4 ∧
  ∃ (p1 p2 p3 p4 : ChocolatePiece),
    bar.pieces 0 = p1 ∧ bar.pieces 1 = p2 ∧ bar.pieces 2 = p3 ∧ bar.pieces 3 = p4 ∧
    p1.area + p2.area + p3.area + p4.area = bar.length * bar.width

/-- All pieces have equal area -/
def all_pieces_equal (bar : ChocolateBar) : Prop :=
  ∀ i j : Fin 4, (bar.pieces i).area = (bar.pieces j).area

/-- Main theorem: If the chocolate bar is divided as described, all pieces have equal area -/
theorem chocolate_division_theorem (bar : ChocolateBar) :
  is_divided_as_described bar → all_pieces_equal bar := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_theorem_l363_36337


namespace NUMINAMATH_CALUDE_parking_lot_valid_tickets_percentage_l363_36322

theorem parking_lot_valid_tickets_percentage 
  (total_cars : ℕ) 
  (unpaid_cars : ℕ) 
  (valid_ticket_percentage : ℝ) :
  total_cars = 300 →
  unpaid_cars = 30 →
  (valid_ticket_percentage / 5 + valid_ticket_percentage) * total_cars / 100 = total_cars - unpaid_cars →
  valid_ticket_percentage = 75 := by
sorry

end NUMINAMATH_CALUDE_parking_lot_valid_tickets_percentage_l363_36322


namespace NUMINAMATH_CALUDE_lives_lost_l363_36385

theorem lives_lost (initial_lives : ℕ) (lives_gained : ℕ) (final_lives : ℕ) 
  (h1 : initial_lives = 14)
  (h2 : lives_gained = 36)
  (h3 : final_lives = 46) :
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 4 := by
  sorry

end NUMINAMATH_CALUDE_lives_lost_l363_36385


namespace NUMINAMATH_CALUDE_gymnastics_performance_participants_l363_36301

/-- The number of grades participating in the gymnastics performance -/
def num_grades : ℕ := 3

/-- The number of classes in each grade -/
def classes_per_grade : ℕ := 4

/-- The number of participants selected from each class -/
def participants_per_class : ℕ := 15

/-- The total number of participants in the gymnastics performance -/
def total_participants : ℕ := num_grades * classes_per_grade * participants_per_class

theorem gymnastics_performance_participants : total_participants = 180 := by
  sorry

end NUMINAMATH_CALUDE_gymnastics_performance_participants_l363_36301


namespace NUMINAMATH_CALUDE_binomial_10_1_l363_36318

theorem binomial_10_1 : Nat.choose 10 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_1_l363_36318


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l363_36360

theorem complete_square_with_integer (y : ℝ) : ∃ k : ℤ, y^2 + 10*y + 33 = (y + 5)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l363_36360


namespace NUMINAMATH_CALUDE_range_of_a_given_three_integer_solutions_l363_36364

/-- The inequality (2x-1)^2 < ax^2 has exactly three integer solutions -/
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), (2*w - 1)^2 < a*w^2 ↔ w = x ∨ w = y ∨ w = z)

/-- The theorem stating the range of a given the condition -/
theorem range_of_a_given_three_integer_solutions :
  ∀ a : ℝ, has_three_integer_solutions a ↔ 25/9 < a ∧ a ≤ 49/16 := by sorry

end NUMINAMATH_CALUDE_range_of_a_given_three_integer_solutions_l363_36364


namespace NUMINAMATH_CALUDE_simplify_and_sum_l363_36349

theorem simplify_and_sum : ∃ (a b : ℕ), 
  (a > 0) ∧ (b > 0) ∧ 
  ((2^10 * 5^2)^(1/4) : ℝ) = a * (b^(1/4) : ℝ) ∧ 
  a + b = 104 := by sorry

end NUMINAMATH_CALUDE_simplify_and_sum_l363_36349


namespace NUMINAMATH_CALUDE_linda_borrowed_amount_l363_36302

-- Define the pay pattern
def payPattern : List Nat := [2, 4, 6, 8, 10]

-- Function to calculate pay for a given number of hours
def calculatePay (hours : Nat) : Nat :=
  let fullCycles := hours / payPattern.length
  let remainingHours := hours % payPattern.length
  fullCycles * payPattern.sum + (payPattern.take remainingHours).sum

-- Theorem statement
theorem linda_borrowed_amount :
  calculatePay 22 = 126 := by
  sorry

end NUMINAMATH_CALUDE_linda_borrowed_amount_l363_36302


namespace NUMINAMATH_CALUDE_hoopit_hands_l363_36373

/-- Represents the number of toes on each hand of a Hoopit -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of toes on each hand of a Neglart -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands each Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that each Hoopit has 4 hands -/
theorem hoopit_hands : 
  ∃ (h : ℕ), h = 4 ∧ 
  hoopit_students * h * hoopit_toes_per_hand + 
  neglart_students * neglart_hands * neglart_toes_per_hand = total_toes :=
sorry

end NUMINAMATH_CALUDE_hoopit_hands_l363_36373


namespace NUMINAMATH_CALUDE_fraction_denominator_l363_36371

theorem fraction_denominator (x y : ℝ) (h : x / y = 7 / 3) :
  ∃ z : ℝ, (x + y) / z = 2.5 ∧ z = 4 * y / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_denominator_l363_36371


namespace NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l363_36381

theorem largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5 :
  ∃ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 24 < Real.sqrt n ∧ Real.sqrt n < 24.5 ∧
  ∀ m : ℕ, m > 0 → 18 ∣ m → 24 < Real.sqrt m → Real.sqrt m < 24.5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l363_36381


namespace NUMINAMATH_CALUDE_weekly_coffee_cost_household_weekly_coffee_cost_l363_36313

/-- Calculates the weekly cost of coffee for a household -/
theorem weekly_coffee_cost 
  (people : ℕ) 
  (cups_per_person : ℕ) 
  (ounces_per_cup : ℚ) 
  (cost_per_ounce : ℚ) : ℚ :=
  let daily_cups := people * cups_per_person
  let daily_ounces := daily_cups * ounces_per_cup
  let weekly_ounces := daily_ounces * 7
  weekly_ounces * cost_per_ounce

/-- Proves that the weekly coffee cost for the given household is $35 -/
theorem household_weekly_coffee_cost : 
  weekly_coffee_cost 4 2 (1/2) (5/4) = 35 := by
  sorry

end NUMINAMATH_CALUDE_weekly_coffee_cost_household_weekly_coffee_cost_l363_36313


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l363_36379

theorem circle_radius_theorem (r : ℝ) (h : r > 0) :
  3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l363_36379


namespace NUMINAMATH_CALUDE_greatest_abcba_div_11_is_greatest_greatest_abcba_div_11_is_divisible_by_11_l363_36374

/-- Represents a five-digit number in the form AB,CBA -/
structure ABCBA where
  a : Nat
  b : Nat
  c : Nat
  value : Nat
  h1 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9
  h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c
  h3 : value = a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- The greatest ABCBA number divisible by 11 -/
def greatest_abcba_div_11 : ABCBA :=
  { a := 9
  , b := 6
  , c := 5
  , value := 96569
  , h1 := by simp
  , h2 := by simp
  , h3 := by simp
  }

theorem greatest_abcba_div_11_is_greatest :
  ∀ n : ABCBA, n.value % 11 = 0 → n.value ≤ greatest_abcba_div_11.value :=
sorry

theorem greatest_abcba_div_11_is_divisible_by_11 :
  greatest_abcba_div_11.value % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_div_11_is_greatest_greatest_abcba_div_11_is_divisible_by_11_l363_36374


namespace NUMINAMATH_CALUDE_lamp_height_difference_l363_36338

theorem lamp_height_difference (old_height new_height : Real) 
  (h1 : old_height = 1)
  (h2 : new_height = 2.33) :
  new_height - old_height = 1.33 := by
sorry

end NUMINAMATH_CALUDE_lamp_height_difference_l363_36338


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l363_36324

/-- An isosceles triangle with side lengths 5 and 10 has a perimeter of 25 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 10 → b = 10 → c = 5 →
  a + b > c → b + c > a → c + a > b →
  a + b + c = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l363_36324


namespace NUMINAMATH_CALUDE_sequence_double_plus_one_greater_l363_36331

/-- Definition of the property $\{a_n\} > M$ -/
def sequence_greater_than (a : ℕ → ℝ) (M : ℝ) : Prop :=
  ∀ n : ℕ, a n ≥ M ∨ a (n + 1) ≥ M

/-- Main theorem -/
theorem sequence_double_plus_one_greater (a : ℕ → ℝ) (M : ℝ) :
  sequence_greater_than a M → sequence_greater_than (fun n ↦ 2 * a n + 1) (2 * M + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_double_plus_one_greater_l363_36331


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l363_36395

theorem circle_tangency_problem :
  let max_radius : ℕ := 36
  let valid_radius (s : ℕ) : Prop := 1 ≤ s ∧ s < max_radius ∧ max_radius % s = 0
  (Finset.filter valid_radius (Finset.range max_radius)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l363_36395


namespace NUMINAMATH_CALUDE_no_primes_in_range_l363_36320

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ p, Prime p → ¬(n.factorial + 2 < p ∧ p < n.factorial + n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l363_36320


namespace NUMINAMATH_CALUDE_bottle_caps_wrappers_difference_l363_36344

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- The number of bottle caps Danny now has in his collection -/
def bottle_caps_in_collection : ℕ := 21

/-- The number of wrappers Danny now has in his collection -/
def wrappers_in_collection : ℕ := 52

/-- Theorem stating the difference between bottle caps and wrappers found at the park -/
theorem bottle_caps_wrappers_difference : 
  bottle_caps_found - wrappers_found = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_wrappers_difference_l363_36344


namespace NUMINAMATH_CALUDE_inequality_proof_l363_36342

def f (x a : ℝ) : ℝ := |x + a| + 2 * |x - 1|

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ x ∈ Set.Icc 1 2, f x a > x^2 - b + 1) :
  (a + 1/2)^2 + (b + 1/2)^2 > 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l363_36342


namespace NUMINAMATH_CALUDE_susan_walk_distance_l363_36392

theorem susan_walk_distance (total_distance : ℝ) (difference : ℝ) :
  total_distance = 15 ∧ difference = 3 →
  ∃ susan_distance erin_distance : ℝ,
    susan_distance + erin_distance = total_distance ∧
    erin_distance = susan_distance - difference ∧
    susan_distance = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_susan_walk_distance_l363_36392


namespace NUMINAMATH_CALUDE_eunice_pots_l363_36319

/-- Given a total number of seeds and a number of seeds per pot (except for the last pot),
    calculate the number of pots needed. -/
def calculate_pots (total_seeds : ℕ) (seeds_per_pot : ℕ) : ℕ :=
  (total_seeds - 1) / seeds_per_pot + 1

/-- Theorem stating that with 10 seeds and 3 seeds per pot (except for the last pot),
    the number of pots needed is 4. -/
theorem eunice_pots : calculate_pots 10 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eunice_pots_l363_36319


namespace NUMINAMATH_CALUDE_product_of_factorials_plus_one_l363_36317

theorem product_of_factorials_plus_one : 
  (1 + 1 / 1) * 
  (1 + 1 / 2) * 
  (1 + 1 / 6) * 
  (1 + 1 / 24) * 
  (1 + 1 / 120) * 
  (1 + 1 / 720) * 
  (1 + 1 / 5040) = 5041 / 5040 := by sorry

end NUMINAMATH_CALUDE_product_of_factorials_plus_one_l363_36317


namespace NUMINAMATH_CALUDE_school2_selection_l363_36382

/-- Represents the number of students selected from a school in a system sampling. -/
def studentsSelected (schoolSize totalStudents selectedStudents : ℕ) : ℚ :=
  (schoolSize : ℚ) * (selectedStudents : ℚ) / (totalStudents : ℚ)

/-- The main theorem about the number of students selected from School 2. -/
theorem school2_selection :
  let totalStudents : ℕ := 360
  let school1Size : ℕ := 123
  let school2Size : ℕ := 123
  let school3Size : ℕ := 114
  let totalSelected : ℕ := 60
  let remainingSelected : ℕ := totalSelected - 1
  let remainingStudents : ℕ := totalStudents - 1
  Int.ceil (studentsSelected school2Size remainingStudents remainingSelected) = 20 := by
  sorry

#check school2_selection

end NUMINAMATH_CALUDE_school2_selection_l363_36382


namespace NUMINAMATH_CALUDE_counterexample_exists_l363_36346

theorem counterexample_exists : ∃ (a b c d : ℝ), a < b ∧ c < d ∧ a * c ≥ b * d := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l363_36346


namespace NUMINAMATH_CALUDE_right_angled_triangle_check_l363_36333

theorem right_angled_triangle_check : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt 5
  (a * a + b * b = c * c) ∧ 
  ¬(1 * 1 + 1 * 1 = Real.sqrt 3 * Real.sqrt 3) ∧
  ¬(0.2 * 0.2 + 0.3 * 0.3 = 0.5 * 0.5) ∧
  ¬((1/3) * (1/3) + (1/4) * (1/4) = (1/5) * (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_check_l363_36333


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l363_36341

theorem reciprocal_of_negative_2022 : (1 : ℚ) / (-2022 : ℚ) = -1 / 2022 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2022_l363_36341


namespace NUMINAMATH_CALUDE_largest_divisible_n_l363_36343

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 12) ∣ (n^3 + 105) ∧ 
  ∀ (m : ℕ), m > n → m > 0 → ¬((m + 12) ∣ (m^3 + 105)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l363_36343


namespace NUMINAMATH_CALUDE_point_C_coordinates_main_theorem_l363_36396

-- Define points A, B, and C in ℝ²
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (13, 9)
def C : ℝ × ℝ := (19, 12)

-- Define the vector from A to B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the vector from B to C
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Theorem stating that C is the correct point
theorem point_C_coordinates : 
  BC = (1/2 : ℝ) • AB := by sorry

-- Main theorem to prove
theorem main_theorem : C = (19, 12) := by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_main_theorem_l363_36396


namespace NUMINAMATH_CALUDE_new_profit_percentage_l363_36327

/-- Calculate the new profit percentage given the original selling price, profit percentage, and additional profit --/
theorem new_profit_percentage
  (original_selling_price : ℝ)
  (original_profit_percentage : ℝ)
  (additional_profit : ℝ)
  (h1 : original_selling_price = 550)
  (h2 : original_profit_percentage = 0.1)
  (h3 : additional_profit = 35) :
  let original_cost_price := original_selling_price / (1 + original_profit_percentage)
  let new_cost_price := original_cost_price * 0.9
  let new_selling_price := original_selling_price + additional_profit
  let new_profit := new_selling_price - new_cost_price
  let new_profit_percentage := new_profit / new_cost_price
  new_profit_percentage = 0.3 := by
sorry


end NUMINAMATH_CALUDE_new_profit_percentage_l363_36327


namespace NUMINAMATH_CALUDE_smallest_delicious_integer_l363_36370

/-- A delicious integer is an integer A for which there exist consecutive integers starting from A that sum to 2024. -/
def IsDelicious (A : ℤ) : Prop :=
  ∃ n : ℕ+, (n : ℤ) * (2 * A + n - 1) / 2 = 2024

/-- The smallest delicious integer is -2023. -/
theorem smallest_delicious_integer : 
  (IsDelicious (-2023) ∧ ∀ A : ℤ, A < -2023 → ¬IsDelicious A) := by
  sorry

end NUMINAMATH_CALUDE_smallest_delicious_integer_l363_36370


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l363_36368

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  100 ≤ n ∧ n ≤ 999 ∧ (75 * n) % 300 = 225 →
  n ≤ 999 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (75 * m) % 300 = 225 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l363_36368


namespace NUMINAMATH_CALUDE_cab_journey_delay_l363_36378

theorem cab_journey_delay (S : ℝ) (h : S > 0) : 
  let usual_time := 40
  let reduced_speed := (5/6) * S
  let new_time := usual_time * S / reduced_speed
  new_time - usual_time = 8 := by
sorry

end NUMINAMATH_CALUDE_cab_journey_delay_l363_36378


namespace NUMINAMATH_CALUDE_stating_prob_served_last_independent_of_position_prob_served_last_2014_l363_36303

/-- 
Represents a round table with n people, where food is passed randomly.
n is the number of people at the table.
-/
structure RoundTable where
  n : ℕ
  hn : n > 1

/-- 
The probability of a specific person (other than the head) being served last.
table: The round table setup
person: The index of the person we're interested in (2 ≤ person ≤ n)
-/
def probabilityServedLast (table : RoundTable) (person : ℕ) : ℚ :=
  1 / (table.n - 1)

/-- 
Theorem stating that the probability of any specific person (other than the head) 
being served last is 1/(n-1), regardless of their position.
-/
theorem prob_served_last_independent_of_position (table : RoundTable) 
    (person : ℕ) (h : 2 ≤ person ∧ person ≤ table.n) : 
    probabilityServedLast table person = 1 / (table.n - 1) := by
  sorry

/-- 
The specific case for the problem with 2014 people and the person of interest
seated 2 seats away from the head.
-/
def table2014 : RoundTable := ⟨2014, by norm_num⟩

theorem prob_served_last_2014 : 
    probabilityServedLast table2014 2 = 1 / 2013 := by
  sorry

end NUMINAMATH_CALUDE_stating_prob_served_last_independent_of_position_prob_served_last_2014_l363_36303


namespace NUMINAMATH_CALUDE_polynomial_equality_l363_36356

theorem polynomial_equality (a b c : ℝ) :
  (∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) →
  4 * a + 2 * b + c = 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l363_36356


namespace NUMINAMATH_CALUDE_cookie_boxes_l363_36350

theorem cookie_boxes (n : Nat) (h : n = 392) : 
  (Finset.filter (fun p => 1 < p ∧ p < n ∧ n / p > 3) (Finset.range (n + 1))).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_boxes_l363_36350


namespace NUMINAMATH_CALUDE_problem_solution_l363_36336

theorem problem_solution (a b c : ℝ) 
  (h1 : 2 * |a + 3| + 4 - b = 0)
  (h2 : c^2 + 4*b - 4*c - 12 = 0) :
  a + b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l363_36336


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l363_36323

theorem average_marks_math_chem (M P C : ℕ) : 
  M + P = 20 →
  C = P + 20 →
  (M + C) / 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l363_36323


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l363_36348

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l363_36348


namespace NUMINAMATH_CALUDE_owen_final_turtles_l363_36334

/-- Represents the number of turtles each person has at different times --/
structure TurtleCount where
  owen_initial : ℕ
  johanna_initial : ℕ
  owen_after_month : ℕ
  johanna_after_month : ℕ
  owen_final : ℕ

/-- Calculates the final number of turtles Owen has --/
def calculate_final_turtles (t : TurtleCount) : Prop :=
  t.owen_initial = 21 ∧
  t.johanna_initial = t.owen_initial - 5 ∧
  t.owen_after_month = 2 * t.owen_initial ∧
  t.johanna_after_month = t.johanna_initial / 2 ∧
  t.owen_final = t.owen_after_month + t.johanna_after_month ∧
  t.owen_final = 50

theorem owen_final_turtles :
  ∃ t : TurtleCount, calculate_final_turtles t :=
sorry

end NUMINAMATH_CALUDE_owen_final_turtles_l363_36334


namespace NUMINAMATH_CALUDE_segment_length_product_l363_36353

theorem segment_length_product (a : ℝ) :
  (∃ a₁ a₂ : ℝ, 
    (∀ a : ℝ, (3*a - 5)^2 + (a - 3)^2 = 117 ↔ (a = a₁ ∨ a = a₂)) ∧
    a₁ * a₂ = -8.32) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_product_l363_36353


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l363_36345

/-- A figure on a grid --/
structure GridFigure where
  -- Add necessary properties of the figure
  area : ℕ

/-- A triangle on a grid --/
structure GridTriangle where
  -- Add necessary properties of a triangle
  area : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to cut a figure into triangles --/
def cut_into_triangles (figure : GridFigure) : List GridTriangle :=
  sorry

/-- Function to check if triangles can form a square --/
def can_form_square (triangles : List GridTriangle) : Bool :=
  sorry

/-- The main theorem --/
theorem figure_to_square_possible (figure : GridFigure) :
  ∃ (triangles : List GridTriangle),
    (triangles.length = 5) ∧
    (cut_into_triangles figure = triangles) ∧
    (can_form_square triangles = true) :=
  sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l363_36345


namespace NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l363_36357

theorem last_two_digits_of_seven_power : 7^30105 ≡ 7 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l363_36357


namespace NUMINAMATH_CALUDE_abs_equation_roots_l363_36389

def abs_equation (x : ℝ) : Prop :=
  |x|^2 + |x| - 6 = 0

theorem abs_equation_roots :
  ∃ (r₁ r₂ : ℝ),
    (abs_equation r₁ ∧ abs_equation r₂) ∧
    (∀ x, abs_equation x → (x = r₁ ∨ x = r₂)) ∧
    (r₁ + r₂ = 0) ∧
    (r₁ * r₂ = -4) :=
by sorry

end NUMINAMATH_CALUDE_abs_equation_roots_l363_36389


namespace NUMINAMATH_CALUDE_student_incorrect_answer_l363_36308

theorem student_incorrect_answer 
  (D : ℕ) -- Dividend
  (h1 : D / 36 = 42) -- Correct division
  (h2 : 63 ≠ 36) -- Student used wrong divisor
  : D / 63 = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_incorrect_answer_l363_36308


namespace NUMINAMATH_CALUDE_apartment_211_location_l363_36388

/-- Represents a building with apartments -/
structure Building where
  total_floors : ℕ
  shop_floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the floor and entrance of an apartment -/
def apartment_location (b : Building) (apartment_number : ℕ) : ℕ × ℕ :=
  sorry

/-- The specific building in the problem -/
def problem_building : Building :=
  { total_floors := 9
  , shop_floors := 1
  , apartments_per_floor := 6 }

theorem apartment_211_location :
  apartment_location problem_building 211 = (5, 5) :=
sorry

end NUMINAMATH_CALUDE_apartment_211_location_l363_36388


namespace NUMINAMATH_CALUDE_f_increasing_l363_36384

def f (x : ℝ) := 3 * x + 2

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l363_36384


namespace NUMINAMATH_CALUDE_die_probabilities_order_l363_36328

def is_less_than_2 (n : ℕ) : Bool := n < 2

def is_greater_than_2 (n : ℕ) : Bool := n > 2

def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

def prob_less_than_2 : ℚ := 1 / 6

def prob_greater_than_2 : ℚ := 2 / 3

def prob_odd : ℚ := 1 / 2

theorem die_probabilities_order :
  prob_less_than_2 < prob_odd ∧ prob_odd < prob_greater_than_2 :=
sorry

end NUMINAMATH_CALUDE_die_probabilities_order_l363_36328


namespace NUMINAMATH_CALUDE_intersection_sum_l363_36376

theorem intersection_sum (a b : ℚ) : 
  (3 = (1/3) * 4 + a) → 
  (4 = (1/2) * 3 + b) → 
  (a + b = 25/6) := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l363_36376


namespace NUMINAMATH_CALUDE_sum_divisible_by_ten_l363_36355

theorem sum_divisible_by_ten (n : ℕ) : 
  10 ∣ (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) ↔ n % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_ten_l363_36355


namespace NUMINAMATH_CALUDE_kerrys_age_l363_36312

/-- Given the conditions of Kerry's birthday candles, prove his age --/
theorem kerrys_age (num_cakes : ℕ) (candles_per_box : ℕ) (cost_per_box : ℚ) (total_cost : ℚ) :
  num_cakes = 5 →
  candles_per_box = 22 →
  cost_per_box = 9/2 →
  total_cost = 27 →
  ∃ (age : ℕ), age = 26 ∧ (num_cakes * age : ℚ) ≤ (total_cost / cost_per_box * candles_per_box) :=
by sorry

end NUMINAMATH_CALUDE_kerrys_age_l363_36312


namespace NUMINAMATH_CALUDE_race_problem_l363_36390

/-- The race problem -/
theorem race_problem (total_distance : ℝ) (time_A : ℝ) (time_B : ℝ) 
  (h1 : total_distance = 70)
  (h2 : time_A = 20)
  (h3 : time_B = 25) :
  total_distance - (total_distance / time_B * time_A) = 14 := by
  sorry

end NUMINAMATH_CALUDE_race_problem_l363_36390


namespace NUMINAMATH_CALUDE_min_touches_theorem_l363_36329

/-- Represents the minimal number of touches required to turn on all lamps in an n×n grid -/
def minTouches (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else n^2

/-- Theorem stating the minimal number of touches required for an n×n grid of lamps -/
theorem min_touches_theorem (n : ℕ) :
  (∀ (grid : Fin n → Fin n → Bool), ∃ (touches : Fin n → Fin n → Bool),
    (∀ i j, touches i j → (∀ k, grid i k = !grid i k ∧ grid k j = !grid k j)) →
    (∀ i j, grid i j = true)) →
  minTouches n = if n % 2 = 1 then n else n^2 :=
sorry

end NUMINAMATH_CALUDE_min_touches_theorem_l363_36329


namespace NUMINAMATH_CALUDE_unique_divisibility_function_l363_36383

/-- A function from positive integers to positive integers -/
def NatFunction := ℕ+ → ℕ+

/-- The property that f(m) + f(n) divides m + n for all m and n -/
def HasDivisibilityProperty (f : NatFunction) : Prop :=
  ∀ m n : ℕ+, (f m + f n) ∣ (m + n)

/-- The identity function on positive integers -/
def identityFunction : NatFunction := fun x => x

/-- Theorem stating that the identity function is the only function satisfying the divisibility property -/
theorem unique_divisibility_function :
  ∀ f : NatFunction, HasDivisibilityProperty f ↔ f = identityFunction := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_function_l363_36383


namespace NUMINAMATH_CALUDE_gcf_of_abc_l363_36347

def a : ℕ := 90
def b : ℕ := 126
def c : ℕ := 180

-- The condition that c is the product of a and b divided by some integer
axiom exists_divisor : ∃ k : ℕ, k ≠ 0 ∧ c * k = a * b

-- Define the greatest common factor function
def gcf (x y z : ℕ) : ℕ := Nat.gcd x (Nat.gcd y z)

theorem gcf_of_abc : gcf a b c = 18 := by sorry

end NUMINAMATH_CALUDE_gcf_of_abc_l363_36347


namespace NUMINAMATH_CALUDE_power_product_reciprocals_l363_36339

theorem power_product_reciprocals (n : ℕ) : (1 / 4 : ℝ) ^ n * 4 ^ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_product_reciprocals_l363_36339


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l363_36314

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- Theorem: The radius of the circle with the given equation is 2 -/
theorem circle_radius_is_two :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l363_36314


namespace NUMINAMATH_CALUDE_inequality_solution_implies_k_range_l363_36358

theorem inequality_solution_implies_k_range :
  ∀ k : ℝ,
  (∀ x : ℝ, x > 1/2 ↔ (k^2 - 2*k + 3/2)^x < (k^2 - 2*k + 3/2)^(1-x)) →
  (1 - Real.sqrt 2 / 2 < k ∧ k < 1 + Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_k_range_l363_36358


namespace NUMINAMATH_CALUDE_a_is_negative_l363_36391

theorem a_is_negative (a b : ℤ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, 3 + a + b^2 = 6*a*k) : a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_is_negative_l363_36391


namespace NUMINAMATH_CALUDE_division_relation_l363_36340

theorem division_relation (a b c : ℚ) 
  (h1 : a / b = 2) 
  (h2 : b / c = 3/4) : 
  c / a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_division_relation_l363_36340


namespace NUMINAMATH_CALUDE_closest_to_target_l363_36305

def target : ℕ := 100000

def numbers : List ℕ := [100260, 99830, 98900, 100320]

def distance (x : ℕ) : ℕ := Int.natAbs (x - target)

theorem closest_to_target : 
  ∀ n ∈ numbers, distance 99830 ≤ distance n :=
by sorry

end NUMINAMATH_CALUDE_closest_to_target_l363_36305


namespace NUMINAMATH_CALUDE_jason_has_21_toys_l363_36380

-- Define the number of toys for each person
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- Theorem statement
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_21_toys_l363_36380


namespace NUMINAMATH_CALUDE_expression_evaluation_l363_36300

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  let A : ℤ := 2*x + y
  let B : ℤ := 2*x - y
  (A^2 - B^2) * (x - 2*y) = 80 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l363_36300


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l363_36386

def f (x : ℝ) := 2 * abs (x + 1) + abs (x + 2)

theorem min_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    (a^2 + b^2) / c + (c^2 + a^2) / b + (b^2 + c^2) / a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l363_36386


namespace NUMINAMATH_CALUDE_max_value_theorem_l363_36393

-- Define the line l
def line_l (y : ℝ) : Prop := y = 8

-- Define the circle C
def circle_C (x y : ℝ) : Prop := ∃ φ, x = 2 * Real.cos φ ∧ y = 2 + 2 * Real.sin φ

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := ∃ α, θ = α ∧ 0 < α ∧ α < Real.pi / 2

-- Define the ray ON
def ray_ON (θ : ℝ) : Prop := ∃ α, θ = α + Real.pi / 2 ∧ 0 < α ∧ α < Real.pi / 2

-- Define the theorem
theorem max_value_theorem :
  ∃ (OP OM OQ ON : ℝ),
    (∀ y, line_l y → ∃ x, circle_C x y) →
    (∀ θ, ray_OM θ → ∃ x y, circle_C x y) →
    (∀ θ, ray_ON θ → ∃ x y, circle_C x y) →
    (∀ α, 0 < α → α < Real.pi / 2 →
      ∃ (OP OM OQ ON : ℝ),
        (OP / OM) * (OQ / ON) ≤ 1 / 16) ∧
    (∃ α, 0 < α ∧ α < Real.pi / 2 ∧
      (OP / OM) * (OQ / ON) = 1 / 16) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l363_36393


namespace NUMINAMATH_CALUDE_pascal_triangle_48th_number_l363_36397

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of elements in the row of Pascal's triangle -/
def row_size : ℕ := 51

/-- The index of the number we're looking for in the row -/
def target_index : ℕ := 48

/-- The theorem stating that the 48th number in the row with 51 numbers 
    of Pascal's triangle is 19600 -/
theorem pascal_triangle_48th_number : 
  binomial (row_size - 1) (target_index - 1) = 19600 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_48th_number_l363_36397


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_2n_gt_sqrt_n_l363_36377

theorem negation_of_proposition (p : ℕ → Prop) : 
  (¬∀ n, p n) ↔ (∃ n, ¬p n) :=
by sorry

theorem negation_of_2n_gt_sqrt_n :
  (¬∀ n : ℕ, 2^n > Real.sqrt n) ↔ (∃ n : ℕ, 2^n ≤ Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_2n_gt_sqrt_n_l363_36377


namespace NUMINAMATH_CALUDE_bread_for_double_meat_sandwiches_l363_36398

/-- Given the following conditions:
  - Two pieces of bread are needed for one regular sandwich.
  - Three pieces of bread are needed for a double meat sandwich.
  - There are 14 regular sandwiches.
  - A total of 64 pieces of bread are used.
Prove that the number of bread pieces used for double meat sandwiches is 36. -/
theorem bread_for_double_meat_sandwiches :
  let regular_sandwich_bread : ℕ := 2
  let double_meat_sandwich_bread : ℕ := 3
  let regular_sandwiches : ℕ := 14
  let total_bread : ℕ := 64
  let double_meat_bread := total_bread - regular_sandwich_bread * regular_sandwiches
  double_meat_bread = 36 := by
  sorry

end NUMINAMATH_CALUDE_bread_for_double_meat_sandwiches_l363_36398


namespace NUMINAMATH_CALUDE_least_number_added_for_divisibility_l363_36311

theorem least_number_added_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(23 ∣ (1054 + y))) ∧ (23 ∣ (1054 + x)) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_least_number_added_for_divisibility_l363_36311


namespace NUMINAMATH_CALUDE_min_draws_for_even_product_l363_36304

theorem min_draws_for_even_product (cards : Finset ℕ) : 
  cards = Finset.range 14 →
  ∃ (n : ℕ), n = 8 ∧ 
    ∀ (subset : Finset ℕ), subset ⊆ cards → subset.card < n → 
      ∃ (x : ℕ), x ∈ subset ∧ Even x :=
by sorry

end NUMINAMATH_CALUDE_min_draws_for_even_product_l363_36304


namespace NUMINAMATH_CALUDE_complement_union_A_B_l363_36372

-- Define the sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l363_36372


namespace NUMINAMATH_CALUDE_article_profit_l363_36306

/-- If selling an article at 2/3 of its original price results in a 20% loss,
    then selling it at the original price results in a 20% profit. -/
theorem article_profit (original_price : ℝ) (cost_price : ℝ) 
    (h1 : original_price > 0) 
    (h2 : cost_price > 0)
    (h3 : (2/3) * original_price = 0.8 * cost_price) : 
  (original_price - cost_price) / cost_price = 0.2 := by
  sorry

#check article_profit

end NUMINAMATH_CALUDE_article_profit_l363_36306


namespace NUMINAMATH_CALUDE_binomial_20_10_l363_36375

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 43758) 
                       (h2 : Nat.choose 18 9 = 48620) 
                       (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l363_36375


namespace NUMINAMATH_CALUDE_carol_peanuts_count_l363_36315

def initial_peanuts : ℕ := 2
def additional_peanuts : ℕ := 5

theorem carol_peanuts_count : initial_peanuts + additional_peanuts = 7 := by
  sorry

end NUMINAMATH_CALUDE_carol_peanuts_count_l363_36315


namespace NUMINAMATH_CALUDE_f_properties_l363_36363

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x - a| + 1

theorem f_properties :
  (∀ x ∈ Set.Icc 0 2, f 0 x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 2, f 0 x = 3) ∧
  (∀ x ∈ Set.Icc 0 2, f 0 x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc 0 2, f 0 x = 3/4) ∧
  (∀ a < 0, ∀ x, f a x ≥ 3/4 + a) ∧
  (∀ a < 0, ∃ x, f a x = 3/4 + a) ∧
  (∀ a ≥ 0, ∀ x, f a x ≥ 3/4 - a) ∧
  (∀ a ≥ 0, ∃ x, f a x = 3/4 - a) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l363_36363


namespace NUMINAMATH_CALUDE_subtract_inequality_negative_l363_36399

theorem subtract_inequality_negative (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_negative_l363_36399


namespace NUMINAMATH_CALUDE_remainder_of_198_digit_sequence_l363_36325

/-- The sum of digits function for a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The function that generates the sequence of digits up to the nth digit -/
def sequenceUpTo (n : ℕ) : List ℕ := sorry

/-- Sum of all digits in the sequence up to the nth digit -/
def sumOfSequenceDigits (n : ℕ) : ℕ := 
  (sequenceUpTo n).map sumOfDigits |>.sum

theorem remainder_of_198_digit_sequence : 
  sumOfSequenceDigits 198 % 9 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_of_198_digit_sequence_l363_36325


namespace NUMINAMATH_CALUDE_preimage_of_three_one_l363_36352

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem stating that (1, 1) is the pre-image of (3, 1) under f -/
theorem preimage_of_three_one :
  f (1, 1) = (3, 1) ∧ ∀ p : ℝ × ℝ, f p = (3, 1) → p = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_three_one_l363_36352


namespace NUMINAMATH_CALUDE_winnie_balloons_l363_36365

/-- The number of balloons Winnie keeps for herself -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

theorem winnie_balloons :
  let red_balloons : ℕ := 15
  let blue_balloons : ℕ := 42
  let yellow_balloons : ℕ := 54
  let purple_balloons : ℕ := 92
  let total_balloons : ℕ := red_balloons + blue_balloons + yellow_balloons + purple_balloons
  let num_friends : ℕ := 11
  balloons_kept total_balloons num_friends = 5 :=
by sorry

end NUMINAMATH_CALUDE_winnie_balloons_l363_36365


namespace NUMINAMATH_CALUDE_reciprocal_power_l363_36394

theorem reciprocal_power (a : ℝ) (h : a⁻¹ = -1) : a^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_l363_36394


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l363_36359

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) (ha : a ≠ 0) :
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a := by
  sorry

-- Problem 2
theorem simplify_expression_2 :
  (25^(1/3) - 125^(1/2)) / 25^(1/4) = 5^(1/6) - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l363_36359


namespace NUMINAMATH_CALUDE_equation_solution_l363_36351

/-- Custom multiplication operation for real numbers -/
def custom_mul (a b : ℝ) : ℝ := a^2 - b^2

/-- Theorem stating the solution to the equation -/
theorem equation_solution :
  ∃ x : ℝ, custom_mul (x + 2) 5 = (x - 5) * (5 + x) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l363_36351


namespace NUMINAMATH_CALUDE_one_dime_in_collection_l363_36367

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat

/-- Calculate the total value of a coin collection in cents --/
def totalValue (coins : CoinCollection) : Nat :=
  coins.pennies * coinValue CoinType.Penny +
  coins.nickels * coinValue CoinType.Nickel +
  coins.dimes * coinValue CoinType.Dime +
  coins.quarters * coinValue CoinType.Quarter

/-- The main theorem --/
theorem one_dime_in_collection :
  ∀ (coins : CoinCollection),
    totalValue coins = 102 ∧
    coins.pennies + coins.nickels + coins.dimes + coins.quarters = 9 ∧
    coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧ coins.quarters ≥ 1
    → coins.dimes = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_dime_in_collection_l363_36367


namespace NUMINAMATH_CALUDE_jakes_weight_l363_36366

/-- Proves Jake's current weight given the conditions of the problem -/
theorem jakes_weight (jake sister brother : ℝ) 
  (h1 : jake - 20 = 2 * sister)
  (h2 : brother = 0.5 * jake)
  (h3 : jake + sister + brother = 330) :
  jake = 170 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l363_36366
