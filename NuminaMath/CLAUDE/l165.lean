import Mathlib

namespace NUMINAMATH_CALUDE_total_weekly_prayers_l165_16589

/-- The number of prayers Pastor Paul makes on a regular day -/
def paul_regular_prayers : ℕ := 20

/-- The number of prayers Pastor Caroline makes on a regular day -/
def caroline_regular_prayers : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weekdays (non-Sunday days) in a week -/
def weekdays : ℕ := 6

/-- Calculate Pastor Paul's total prayers for a week -/
def paul_weekly_prayers : ℕ :=
  paul_regular_prayers * weekdays + 2 * paul_regular_prayers

/-- Calculate Pastor Bruce's total prayers for a week -/
def bruce_weekly_prayers : ℕ :=
  (paul_regular_prayers / 2) * weekdays + 2 * (2 * paul_regular_prayers)

/-- Calculate Pastor Caroline's total prayers for a week -/
def caroline_weekly_prayers : ℕ :=
  caroline_regular_prayers * weekdays + 3 * caroline_regular_prayers

/-- The main theorem: total prayers of all pastors in a week -/
theorem total_weekly_prayers :
  paul_weekly_prayers + bruce_weekly_prayers + caroline_weekly_prayers = 390 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_prayers_l165_16589


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l165_16540

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l165_16540


namespace NUMINAMATH_CALUDE_candies_to_remove_for_even_distribution_l165_16545

def total_candies : ℕ := 24
def num_sisters : ℕ := 4

theorem candies_to_remove_for_even_distribution :
  (total_candies % num_sisters = 0) ∧
  (total_candies / num_sisters * num_sisters = total_candies) :=
by sorry

end NUMINAMATH_CALUDE_candies_to_remove_for_even_distribution_l165_16545


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l165_16597

/-- Given a right triangle ABC with angle C = 90°, side a = 12, and side b = 16, prove that the length of side c is 20. -/
theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 → b = 16 → c^2 = a^2 + b^2 → c = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l165_16597


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l165_16506

/-- A sequence of 10 natural numbers where each number from the third onwards
    is the sum of the two preceding numbers. -/
def FibonacciLikeSequence (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, i.val ≥ 2 → a i = a (i - 1) + a (i - 2)

theorem fourth_number_in_sequence
  (a : Fin 10 → ℕ)
  (h_seq : FibonacciLikeSequence a)
  (h_seventh : a 6 = 42)
  (h_ninth : a 8 = 110) :
  a 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l165_16506


namespace NUMINAMATH_CALUDE_tank_capacity_is_33_l165_16507

/-- Represents the capacity of a water tank with specific filling conditions. -/
def tank_capacity (initial_fraction : ℚ) (added_water : ℕ) (leak_rate : ℕ) (fill_time : ℕ) : ℚ :=
  let total_leak := (leak_rate * fill_time : ℚ)
  let total_added := (added_water : ℚ) + total_leak
  total_added / (1 - initial_fraction)

/-- Theorem stating that under given conditions, the tank capacity is 33 gallons. -/
theorem tank_capacity_is_33 :
  tank_capacity (1/3) 16 2 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_33_l165_16507


namespace NUMINAMATH_CALUDE_cindy_travel_time_l165_16531

/-- Calculates the total time for Cindy to travel 1 mile -/
theorem cindy_travel_time (run_speed walk_speed run_distance walk_distance : ℝ) :
  run_speed = 3 →
  walk_speed = 1 →
  run_distance = 0.5 →
  walk_distance = 0.5 →
  run_distance + walk_distance = 1 →
  (run_distance / run_speed + walk_distance / walk_speed) * 60 = 40 :=
by sorry

end NUMINAMATH_CALUDE_cindy_travel_time_l165_16531


namespace NUMINAMATH_CALUDE_max_soccer_balls_buyable_l165_16528

/-- The cost of 6 soccer balls in yuan -/
def cost_of_six_balls : ℕ := 168

/-- The number of balls in a set -/
def balls_in_set : ℕ := 6

/-- The amount of money available to spend in yuan -/
def available_money : ℕ := 500

/-- The maximum number of soccer balls that can be bought -/
def max_balls_bought : ℕ := 17

theorem max_soccer_balls_buyable :
  (cost_of_six_balls * max_balls_bought) / balls_in_set ≤ available_money ∧
  (cost_of_six_balls * (max_balls_bought + 1)) / balls_in_set > available_money :=
by sorry

end NUMINAMATH_CALUDE_max_soccer_balls_buyable_l165_16528


namespace NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l165_16587

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a ≤ 0

-- Define the theorem
theorem range_of_a_when_p_is_false :
  (∀ a : ℝ, ¬(p a) ↔ a > 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_is_false_l165_16587


namespace NUMINAMATH_CALUDE_num_tough_weeks_is_three_l165_16556

def tough_week_sales : ℕ := 800
def good_week_sales : ℕ := 2 * tough_week_sales
def num_good_weeks : ℕ := 5
def total_sales : ℕ := 10400

theorem num_tough_weeks_is_three :
  ∃ (num_tough_weeks : ℕ),
    num_tough_weeks * tough_week_sales + num_good_weeks * good_week_sales = total_sales ∧
    num_tough_weeks = 3 :=
by sorry

end NUMINAMATH_CALUDE_num_tough_weeks_is_three_l165_16556


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l165_16572

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l165_16572


namespace NUMINAMATH_CALUDE_proportional_function_graph_l165_16576

/-- A proportional function with coefficient 2 -/
def f (x : ℝ) : ℝ := 2 * x

theorem proportional_function_graph (x y : ℝ) :
  y = f x → (∃ k : ℝ, k > 0 ∧ y = k * x) ∧ f 0 = 0 := by
  sorry

#check proportional_function_graph

end NUMINAMATH_CALUDE_proportional_function_graph_l165_16576


namespace NUMINAMATH_CALUDE_workshop_workers_count_l165_16594

/-- Proves that the total number of workers in a workshop is 24 given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  (W : ℚ) * 8000 = (8 : ℚ) * 12000 + (N : ℚ) * 6000 →  -- total salary equation
  W = 8 + N →                                          -- total workers equation
  W = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l165_16594


namespace NUMINAMATH_CALUDE_gcd_count_for_product_504_l165_16570

theorem gcd_count_for_product_504 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 504) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 504) ∧ s.card = 9 :=
sorry

end NUMINAMATH_CALUDE_gcd_count_for_product_504_l165_16570


namespace NUMINAMATH_CALUDE_cars_meeting_time_l165_16563

/-- The time when two cars meet on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) :
  highway_length = 600 →
  speed1 = 65 →
  speed2 = 75 →
  (highway_length / (speed1 + speed2) : ℝ) = 30 / 7 :=
by sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l165_16563


namespace NUMINAMATH_CALUDE_special_gp_common_ratio_l165_16508

/-- A geometric progression with positive terms where any term minus the next term 
    equals half the sum of the next two terms. -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : 0 < a
  r_pos : 0 < r
  special_property : ∀ n : ℕ, a * r^n - a * r^(n+1) = (1/2) * (a * r^(n+1) + a * r^(n+2))

/-- The common ratio of a special geometric progression is (√17 - 3) / 2. -/
theorem special_gp_common_ratio (gp : SpecialGeometricProgression) : 
  gp.r = (Real.sqrt 17 - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_gp_common_ratio_l165_16508


namespace NUMINAMATH_CALUDE_seven_valid_methods_l165_16530

/-- The number of valid purchasing methods for software and tapes -/
def validPurchaseMethods : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    60 * p.1 + 70 * p.2 ≤ 500 ∧
    p.1 ≥ 3 ∧
    p.2 ≥ 2)
    (Finset.product (Finset.range 9) (Finset.range 8))).card

/-- Theorem stating that there are exactly 7 valid purchasing methods -/
theorem seven_valid_methods : validPurchaseMethods = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_valid_methods_l165_16530


namespace NUMINAMATH_CALUDE_problem_statement_l165_16543

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem problem_statement : 8^(2/3) + lg 25 - lg (1/4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l165_16543


namespace NUMINAMATH_CALUDE_inequality_proof_l165_16544

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l165_16544


namespace NUMINAMATH_CALUDE_christina_rearrangements_l165_16524

theorem christina_rearrangements (n : ℕ) (rate1 rate2 : ℕ) (h1 : n = 9) (h2 : rate1 = 12) (h3 : rate2 = 18) :
  (n.factorial / 2 / rate1 + n.factorial / 2 / rate2) / 60 = 420 := by
  sorry

end NUMINAMATH_CALUDE_christina_rearrangements_l165_16524


namespace NUMINAMATH_CALUDE_pokemon_cards_distribution_l165_16580

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 56) (h2 : num_friends = 4) (h3 : num_friends > 0) :
  total_cards / num_friends = 14 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_distribution_l165_16580


namespace NUMINAMATH_CALUDE_max_area_square_with_perimeter_32_l165_16592

/-- The maximum area of a square with a perimeter of 32 meters is 64 square meters. -/
theorem max_area_square_with_perimeter_32 :
  let perimeter : ℝ := 32
  let side_length : ℝ := perimeter / 4
  let area : ℝ := side_length ^ 2
  area = 64 := by sorry

end NUMINAMATH_CALUDE_max_area_square_with_perimeter_32_l165_16592


namespace NUMINAMATH_CALUDE_midpoint_path_difference_l165_16560

/-- Given a rectangle with sides a and b, and a segment AB of length 4 inside it,
    the path traced by the midpoint C of AB as A completes one revolution around 
    the perimeter is shorter than the perimeter by 16 - 4π. -/
theorem midpoint_path_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hl : 4 < min a b) :
  2 * (a + b) - (2 * (a + b) - 16 + 4 * Real.pi) = 16 - 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_midpoint_path_difference_l165_16560


namespace NUMINAMATH_CALUDE_mahogany_count_l165_16523

/-- The number of initially planted Mahogany trees -/
def initial_mahogany : ℕ := sorry

/-- The number of initially planted Narra trees -/
def initial_narra : ℕ := 30

/-- The total number of trees that fell -/
def total_fallen : ℕ := 5

/-- The number of Mahogany trees that fell -/
def mahogany_fallen : ℕ := sorry

/-- The number of Narra trees that fell -/
def narra_fallen : ℕ := sorry

/-- The number of new Mahogany trees planted after the typhoon -/
def new_mahogany : ℕ := sorry

/-- The number of new Narra trees planted after the typhoon -/
def new_narra : ℕ := sorry

/-- The total number of trees after replanting -/
def total_trees : ℕ := 88

theorem mahogany_count : initial_mahogany = 50 :=
  by sorry

end NUMINAMATH_CALUDE_mahogany_count_l165_16523


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l165_16504

/-- If the quadratic equation x^2 - 3x + 2k = 0 has a root of 1, then k = 1 -/
theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*k = 0) ∧ (1^2 - 3*1 + 2*k = 0) → k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l165_16504


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l165_16573

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) : 
  2 * (x - 2)^2 + 2 * (y - 3)^2 + 2 * (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l165_16573


namespace NUMINAMATH_CALUDE_value_of_a_l165_16577

/-- Proves that if 0.5% of a equals 95 paise, then a equals 190 rupees -/
theorem value_of_a (a : ℚ) : (0.5 / 100) * a = 95 / 100 → a = 190 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l165_16577


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l165_16536

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def f (a : ℕ → ℝ) : ℝ := 1  -- Definition of f, which always returns 1

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 5 / a 3 = 5 / 9) : 
  f a = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l165_16536


namespace NUMINAMATH_CALUDE_cubic_expression_value_l165_16520

theorem cubic_expression_value : 
  let x : ℤ := -2
  (-2)^3 + (-2)^2 + 3*(-2) - 6 = -16 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l165_16520


namespace NUMINAMATH_CALUDE_positive_solution_equation_l165_16510

theorem positive_solution_equation : ∃ x : ℝ, 
  x > 0 ∧ 
  x = 21 + Real.sqrt 449 ∧ 
  (1 / 2) * (4 * x^2 - 2) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_equation_l165_16510


namespace NUMINAMATH_CALUDE_thabo_hardcover_count_l165_16519

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 160 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 20 ∧
  bc.paperback_fiction = 2 * bc.paperback_nonfiction

theorem thabo_hardcover_count (bc : BookCollection) 
  (h : is_valid_collection bc) : bc.hardcover_nonfiction = 25 := by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_count_l165_16519


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l165_16553

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (count1 : ℕ)
  (avg1 : ℚ)
  (count2 : ℕ)
  (avg2 : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_count1 : count1 = 2)
  (h_avg1 : avg1 = 4.4)
  (h_count2 : count2 = 2)
  (h_avg2 : avg2 = 3.85) :
  let sum_all := total * avg_all
  let sum1 := count1 * avg1
  let sum2 := count2 * avg2
  let remaining := total - count1 - count2
  let sum_remaining := sum_all - sum1 - sum2
  (sum_remaining / remaining : ℚ) = 3.6 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l165_16553


namespace NUMINAMATH_CALUDE_median_length_in_right_triangle_l165_16526

theorem median_length_in_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let c := Real.sqrt (a^2 + b^2)  -- hypotenuse
  let m := Real.sqrt (b^2 + (a/2)^2)  -- median
  (∃ k : ℝ, k = 0.51 ∧ m = k * c) ∧ ¬(∃ k : ℝ, k = 0.49 ∧ m = k * c) :=
by sorry

end NUMINAMATH_CALUDE_median_length_in_right_triangle_l165_16526


namespace NUMINAMATH_CALUDE_second_bakery_sacks_per_week_l165_16547

/-- Proves that the second bakery needs 4 sacks per week given the conditions of Antoine's strawberry supply -/
theorem second_bakery_sacks_per_week 
  (total_sacks : ℕ) 
  (num_weeks : ℕ) 
  (first_bakery_sacks_per_week : ℕ) 
  (third_bakery_sacks_per_week : ℕ) 
  (h1 : total_sacks = 72) 
  (h2 : num_weeks = 4) 
  (h3 : first_bakery_sacks_per_week = 2) 
  (h4 : third_bakery_sacks_per_week = 12) : 
  (total_sacks - (first_bakery_sacks_per_week * num_weeks) - (third_bakery_sacks_per_week * num_weeks)) / num_weeks = 4 := by
sorry

end NUMINAMATH_CALUDE_second_bakery_sacks_per_week_l165_16547


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l165_16558

theorem function_satisfies_conditions (m n : ℕ) :
  let f : ℕ → ℕ → ℕ := λ m n => m * n
  (m ≥ 1 ∧ n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ k : ℕ, f k 0 = 0 ∧ f 0 k = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l165_16558


namespace NUMINAMATH_CALUDE_square_difference_sum_l165_16578

theorem square_difference_sum : 
  27^2 - 25^2 + 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 392 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_sum_l165_16578


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l165_16513

theorem rectangle_area_proof (square_area : ℝ) (rectangle_width rectangle_length : ℝ) : 
  square_area = 25 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 2 * rectangle_width →
  rectangle_width * rectangle_length = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l165_16513


namespace NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l165_16548

/-- The number of coughs per minute for Georgia -/
def georgia_coughs_per_minute : ℕ := 5

/-- The number of coughs per minute for Robert -/
def robert_coughs_per_minute : ℕ := 2 * georgia_coughs_per_minute

/-- The duration in minutes -/
def duration : ℕ := 20

/-- The total number of coughs after the given duration -/
def total_coughs : ℕ := (georgia_coughs_per_minute + robert_coughs_per_minute) * duration

theorem total_coughs_after_20_minutes :
  total_coughs = 300 := by sorry

end NUMINAMATH_CALUDE_total_coughs_after_20_minutes_l165_16548


namespace NUMINAMATH_CALUDE_mixed_gender_selections_l165_16564

-- Define the number of male and female students
def num_male_students : Nat := 5
def num_female_students : Nat := 3

-- Define the total number of students
def total_students : Nat := num_male_students + num_female_students

-- Define the number of students to be selected
def students_to_select : Nat := 3

-- Function to calculate combinations
def combination (n : Nat) (r : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem mixed_gender_selections :
  combination total_students students_to_select -
  combination num_male_students students_to_select -
  combination num_female_students students_to_select = 45 := by
  sorry


end NUMINAMATH_CALUDE_mixed_gender_selections_l165_16564


namespace NUMINAMATH_CALUDE_parabola_equation_l165_16514

/-- Represents a parabola with integer coefficients -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  a_pos : 0 < a
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, -2)

/-- The directrix of the parabola -/
def directrix (x y : ℝ) : Prop := 5 * x + 2 * y = 10

/-- Checks if a point is on the parabola -/
def isOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  p.a * x^2 + p.b * x * y + p.c * y^2 + p.d * x + p.e * y + p.f = 0

/-- The theorem stating the equation of the parabola -/
theorem parabola_equation : ∃ (p : Parabola), 
  ∀ (x y : ℝ), isOnParabola p x y ↔ 
    (x - focus.1)^2 + (y - focus.2)^2 = (5 * x + 2 * y - 10)^2 / 29 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l165_16514


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l165_16509

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 - 5 * x > 22) → x ≤ -4 ∧ (3 - 5 * (-4) > 22) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l165_16509


namespace NUMINAMATH_CALUDE_triangle_n_values_l165_16502

theorem triangle_n_values :
  let valid_n (n : ℕ) : Prop :=
    3*n + 15 > 3*n + 10 ∧ 
    3*n + 10 > 4*n ∧ 
    4*n + (3*n + 10) > 3*n + 15 ∧ 
    4*n + (3*n + 15) > 3*n + 10 ∧ 
    (3*n + 10) + (3*n + 15) > 4*n
  ∃! (s : Finset ℕ), (∀ n ∈ s, valid_n n) ∧ s.card = 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_n_values_l165_16502


namespace NUMINAMATH_CALUDE_item_list_price_l165_16565

theorem item_list_price : ∃ (list_price : ℝ), 
  list_price > 0 ∧
  0.15 * (list_price - 15) = 0.25 * (list_price - 25) ∧
  list_price = 40 := by
sorry

end NUMINAMATH_CALUDE_item_list_price_l165_16565


namespace NUMINAMATH_CALUDE_carlotta_time_theorem_l165_16532

def singing_time : ℕ := 6

def practice_time (n : ℕ) : ℕ := 2 * n

def tantrum_time (n : ℕ) : ℕ := 3 * n + 1

def total_time (singing : ℕ) : ℕ :=
  singing +
  singing * practice_time singing +
  singing * tantrum_time singing

theorem carlotta_time_theorem :
  total_time singing_time = 192 := by sorry

end NUMINAMATH_CALUDE_carlotta_time_theorem_l165_16532


namespace NUMINAMATH_CALUDE_inverse_proportion_through_point_l165_16550

/-- An inverse proportion function passing through the point (2,5) has k = 10 -/
theorem inverse_proportion_through_point (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (k / x = 5 ↔ x = 2)) → k = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_through_point_l165_16550


namespace NUMINAMATH_CALUDE_water_tower_theorem_l165_16588

def water_tower_problem (total_capacity : ℕ) (first_neighborhood : ℕ) : Prop :=
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let used_water := first_neighborhood + second_neighborhood + third_neighborhood
  total_capacity - used_water = 350

theorem water_tower_theorem : water_tower_problem 1200 150 := by
  sorry

end NUMINAMATH_CALUDE_water_tower_theorem_l165_16588


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l165_16516

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h1 : pens = 1204) (h2 : pencils = 840) :
  Nat.gcd pens pencils = 28 := by
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l165_16516


namespace NUMINAMATH_CALUDE_board_ratio_l165_16533

theorem board_ratio (total_length shorter_length : ℝ) 
  (h1 : total_length = 6)
  (h2 : shorter_length = 2)
  (h3 : shorter_length < total_length) :
  (total_length - shorter_length) / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_board_ratio_l165_16533


namespace NUMINAMATH_CALUDE_percentage_problem_l165_16500

theorem percentage_problem (x y : ℝ) (P : ℝ) :
  (0.1 * x = P / 100 * y) →  -- 10% of x equals P% of y
  (x / y = 2) →              -- The ratio of x to y is 2
  P = 20 :=                  -- The percentage of y is 20%
by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l165_16500


namespace NUMINAMATH_CALUDE_function_inequality_l165_16542

theorem function_inequality (a : ℝ) : 
  let f (x : ℝ) := (1/3) * x^3 - Real.log (x + 1)
  let g (x : ℝ) := x^2 - 2 * a * x
  (∃ (x₁ : ℝ) (x₂ : ℝ), x₁ ∈ Set.Icc 0 1 ∧ x₂ ∈ Set.Icc 1 2 ∧ 
    (deriv f x₁) ≥ g x₂) →
  a ≥ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l165_16542


namespace NUMINAMATH_CALUDE_probability_at_least_one_head_and_three_l165_16593

def coin_flip : Nat := 2
def die_sides : Nat := 8

def coin_success : Nat := 3  -- number of successful coin flip outcomes (HH, HT, TH)
def die_success : Nat := 1   -- number of successful die roll outcomes (3)

def total_outcomes : Nat := coin_flip^2 * die_sides
def successful_outcomes : Nat := coin_success * die_success

theorem probability_at_least_one_head_and_three :
  (successful_outcomes : ℚ) / total_outcomes = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_head_and_three_l165_16593


namespace NUMINAMATH_CALUDE_garden_walkway_area_l165_16541

/-- Represents the configuration of a garden with flower beds and walkways -/
structure Garden where
  bed_width : ℕ
  bed_height : ℕ
  walkway_width : ℕ
  rows : ℕ
  beds_in_first_row : ℕ
  beds_in_other_rows : ℕ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℕ :=
  let total_width := g.bed_width * g.beds_in_first_row + (g.beds_in_first_row + 1) * g.walkway_width
  let total_height := g.bed_height * g.rows + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := (g.bed_width * g.bed_height) * (g.beds_in_first_row + (g.rows - 1) * g.beds_in_other_rows)
  total_area - bed_area

/-- The theorem stating that for the given garden configuration, the walkway area is 488 square feet -/
theorem garden_walkway_area :
  let g : Garden := {
    bed_width := 8,
    bed_height := 3,
    walkway_width := 2,
    rows := 4,
    beds_in_first_row := 3,
    beds_in_other_rows := 2
  }
  walkway_area g = 488 := by sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l165_16541


namespace NUMINAMATH_CALUDE_max_sum_of_coefficients_l165_16501

/-- Given a temperature function T(t) = a * sin(t) + b * cos(t) where t ∈ (0, +∞),
    a and b are positive real numbers, and the maximum temperature difference is 10°C,
    prove that the maximum value of a + b is 5√2. -/
theorem max_sum_of_coefficients (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ t, t > 0 → t < Real.pi → ∃ T, T = a * Real.sin t + b * Real.cos t) →
  (∃ t₁ t₂, t₁ > 0 ∧ t₂ > 0 ∧ t₁ < Real.pi ∧ t₂ < Real.pi ∧
    a * Real.sin t₁ + b * Real.cos t₁ - (a * Real.sin t₂ + b * Real.cos t₂) = 10) →
  a + b ≤ 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_coefficients_l165_16501


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l165_16579

def f (m : ℕ) : ℕ :=
  if m % 2 = 0 ∧ m > 0 then
    (List.range (m / 2)).foldl (λ acc i => acc * (2 * i + 2)) 1
  else
    0

theorem greatest_prime_factor_f_28 :
  (Nat.factors (f 28)).maximum? = some 13 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_28_l165_16579


namespace NUMINAMATH_CALUDE_largest_room_width_l165_16561

theorem largest_room_width (width smallest_width smallest_length largest_length area_difference : ℝ) :
  smallest_width = 15 →
  smallest_length = 8 →
  largest_length = 30 →
  area_difference = 1230 →
  width * largest_length - smallest_width * smallest_length = area_difference →
  width = 45 := by
sorry

end NUMINAMATH_CALUDE_largest_room_width_l165_16561


namespace NUMINAMATH_CALUDE_sixth_term_term_1994_l165_16581

-- Define the sequence
def a (n : ℕ) : ℕ := n * (n + 1)

-- Theorem for the 6th term
theorem sixth_term : a 6 = 42 := by sorry

-- Theorem for the 1994th term
theorem term_1994 : a 1994 = 3978030 := by sorry

end NUMINAMATH_CALUDE_sixth_term_term_1994_l165_16581


namespace NUMINAMATH_CALUDE_system_solution_arithmetic_progression_l165_16562

/-- 
Given a system of equations:
  x + y + m*z = a
  x + m*y + z = b
  m*x + y + z = c
This theorem states that for m ≠ 1 and m ≠ -2, the system has a unique solution (x, y, z) 
in arithmetic progression if and only if a, b, c are in arithmetic progression.
-/
theorem system_solution_arithmetic_progression 
  (m a b c : ℝ) (hm1 : m ≠ 1) (hm2 : m ≠ -2) :
  (∃! x y z : ℝ, x + y + m*z = a ∧ x + m*y + z = b ∧ m*x + y + z = c ∧ 
   2*y = x + z) ↔ 2*b = a + c :=
sorry

end NUMINAMATH_CALUDE_system_solution_arithmetic_progression_l165_16562


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l165_16538

theorem gcd_power_minus_one (m n : ℕ+) :
  Nat.gcd ((2 ^ m.val) - 1) ((2 ^ n.val) - 1) = (2 ^ Nat.gcd m.val n.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l165_16538


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l165_16595

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {0, 3, 4, 5}

theorem intersection_of_P_and_Q : P ∩ Q = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l165_16595


namespace NUMINAMATH_CALUDE_fraction_equality_l165_16599

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = -3/7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l165_16599


namespace NUMINAMATH_CALUDE_inequality_proof_l165_16596

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l165_16596


namespace NUMINAMATH_CALUDE_chessboard_zero_condition_l165_16554

/-- Represents a chessboard with natural numbers -/
def Chessboard (m n : ℕ) := Fin m → Fin n → ℕ

/-- Sums the numbers on black squares of a chessboard -/
def sumBlack (board : Chessboard m n) : ℕ := sorry

/-- Sums the numbers on white squares of a chessboard -/
def sumWhite (board : Chessboard m n) : ℕ := sorry

/-- Represents an allowed move on the chessboard -/
def allowedMove (board : Chessboard m n) (i j : Fin m) (k l : Fin n) (value : ℤ) : Chessboard m n := sorry

/-- Predicate to check if all numbers on the board are zero -/
def allZero (board : Chessboard m n) : Prop := ∀ i j, board i j = 0

/-- Predicate to check if a board can be reduced to all zeros using allowed moves -/
def canReduceToZero (board : Chessboard m n) : Prop := sorry

theorem chessboard_zero_condition {m n : ℕ} (board : Chessboard m n) :
  canReduceToZero board ↔ sumBlack board = sumWhite board := by sorry

end NUMINAMATH_CALUDE_chessboard_zero_condition_l165_16554


namespace NUMINAMATH_CALUDE_easter_egg_decoration_l165_16534

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- Mia's egg decoration rate per hour -/
def mia_rate : ℕ := 2 * dozen

/-- Billy's egg decoration rate per hour -/
def billy_rate : ℕ := 10

/-- The number of hours Mia and Billy work together -/
def work_hours : ℕ := 5

/-- The total number of eggs Mia and Billy need to decorate -/
def total_eggs : ℕ := mia_rate * work_hours + billy_rate * work_hours

theorem easter_egg_decoration :
  total_eggs = 170 :=
by sorry

end NUMINAMATH_CALUDE_easter_egg_decoration_l165_16534


namespace NUMINAMATH_CALUDE_remainder_theorem_l165_16503

/-- Given a polynomial q(x) satisfying specific conditions, 
    prove properties about its remainder when divided by (x - 3)(x + 2)(x - 4) -/
theorem remainder_theorem (q : ℝ → ℝ) (h1 : q 3 = 2) (h2 : q (-2) = -3) (h3 : q 4 = 6) :
  ∃ (s : ℝ → ℝ), 
    (∀ x, q x = (x - 3) * (x + 2) * (x - 4) * (q x / ((x - 3) * (x + 2) * (x - 4))) + s x) ∧
    (∀ x, s x = 1/2 * x^2 + 1/2 * x - 4) ∧
    (s 5 = 11) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l165_16503


namespace NUMINAMATH_CALUDE_hyperbola_properties_l165_16529

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the length of the real axis
def real_axis_length : ℝ := 4

-- Define the distance from foci to asymptote
def foci_to_asymptote_distance : ℝ := 1

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → 
    real_axis_length = 4 ∧ 
    foci_to_asymptote_distance = 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l165_16529


namespace NUMINAMATH_CALUDE_alex_sandwiches_l165_16575

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) : ℕ :=
  num_meats * (num_cheeses.choose 3)

/-- Theorem: Alex can make 1760 different sandwiches -/
theorem alex_sandwiches :
  num_sandwiches 8 12 = 1760 := by
  sorry

end NUMINAMATH_CALUDE_alex_sandwiches_l165_16575


namespace NUMINAMATH_CALUDE_trig_identity_l165_16546

theorem trig_identity : 
  (Real.cos (12 * π / 180) - Real.cos (18 * π / 180) * Real.sin (60 * π / 180)) / 
  Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l165_16546


namespace NUMINAMATH_CALUDE_x_value_l165_16505

theorem x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l165_16505


namespace NUMINAMATH_CALUDE_group_size_proof_l165_16586

/-- The number of people in the group -/
def n : ℕ := 10

/-- The weight increase of the group when the new person joins -/
def weight_increase : ℕ := 40

/-- The weight of the person being replaced -/
def old_weight : ℕ := 70

/-- The weight of the new person joining the group -/
def new_weight : ℕ := 110

/-- The average weight increase per person -/
def avg_increase : ℕ := 4

theorem group_size_proof :
  n * old_weight + weight_increase = n * (old_weight + avg_increase) :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l165_16586


namespace NUMINAMATH_CALUDE_value_of_a_l165_16522

def A (a : ℝ) : Set ℝ := {0, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem value_of_a : ∀ a : ℝ, A a ∪ B a = {0, 1, 2, 4} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l165_16522


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l165_16569

/-- A line passing through two points is parallel to another line -/
def is_parallel_line (x1 y1 x2 y2 k : ℚ) : Prop :=
  (20 - (-8)) / (k - 3) = -(3 : ℚ) / 4

/-- The theorem stating that k equals -103/3 for the given conditions -/
theorem parallel_line_k_value :
  ∀ k : ℚ, is_parallel_line 3 (-8) k 20 k → k = -103/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l165_16569


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l165_16582

/-- The inequality holds for all real x and θ ∈ [0, π/2] if and only if a is in the specified range -/
theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l165_16582


namespace NUMINAMATH_CALUDE_sixth_employee_salary_l165_16585

def employee_salaries : List ℝ := [1000, 2500, 3650, 1500, 2000]
def mean_salary : ℝ := 2291.67
def num_employees : ℕ := 6

theorem sixth_employee_salary :
  let total_salary := (mean_salary * num_employees)
  let known_salaries_sum := employee_salaries.sum
  total_salary - known_salaries_sum = 2100 := by
  sorry

end NUMINAMATH_CALUDE_sixth_employee_salary_l165_16585


namespace NUMINAMATH_CALUDE_minimum_score_to_increase_average_l165_16598

def scores : List ℕ := [84, 76, 89, 94, 67, 90]

def current_average : ℚ := (scores.sum : ℚ) / scores.length

def target_average : ℚ := current_average + 5

def required_score : ℕ := 118

theorem minimum_score_to_increase_average : 
  (((scores.sum + required_score : ℚ) / (scores.length + 1)) = target_average) ∧
  (∀ (s : ℕ), s < required_score → 
    ((scores.sum + s : ℚ) / (scores.length + 1)) < target_average) := by
  sorry

end NUMINAMATH_CALUDE_minimum_score_to_increase_average_l165_16598


namespace NUMINAMATH_CALUDE_emily_beads_count_l165_16527

/-- The number of necklaces Emily made -/
def necklaces : ℕ := 26

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 2

/-- The total number of beads Emily had -/
def total_beads : ℕ := necklaces * beads_per_necklace

/-- Theorem stating that the total number of beads Emily had is 52 -/
theorem emily_beads_count : total_beads = 52 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l165_16527


namespace NUMINAMATH_CALUDE_symmetric_center_of_translated_cosine_l165_16555

theorem symmetric_center_of_translated_cosine : 
  let f (x : ℝ) := Real.cos (2 * x + π / 4)
  let g (x : ℝ) := f (x - π / 4)
  ∃ (k : ℤ), g ((k : ℝ) * π / 2 + 3 * π / 8) = g (-(k : ℝ) * π / 2 - 3 * π / 8) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_center_of_translated_cosine_l165_16555


namespace NUMINAMATH_CALUDE_surface_generates_solid_l165_16525

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- A cone formed by rotating a right-angled triangle -/
structure Cone where
  base_radius : ℝ
  height : ℝ

/-- Rotation of a right-angled triangle around one of its right-angle sides -/
def rotate_triangle (t : RightTriangle) : Cone :=
  { base_radius := t.base, height := t.height }

/-- Theorem: Rotating a right-angled triangle generates a solid (cone) -/
theorem surface_generates_solid (t : RightTriangle) :
  ∃ (c : Cone), c = rotate_triangle t :=
sorry

end NUMINAMATH_CALUDE_surface_generates_solid_l165_16525


namespace NUMINAMATH_CALUDE_white_squares_20th_row_l165_16591

/-- The total number of squares in the nth row of the modified "stair-step" figure -/
def totalSquares (n : ℕ) : ℕ := 3 * n

/-- The number of white squares in the nth row of the modified "stair-step" figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n) / 2

theorem white_squares_20th_row :
  whiteSquares 20 = 30 := by sorry

end NUMINAMATH_CALUDE_white_squares_20th_row_l165_16591


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l165_16574

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l165_16574


namespace NUMINAMATH_CALUDE_no_trapezoid_solution_l165_16512

theorem no_trapezoid_solution : ¬∃ (b₁ b₂ : ℕ), 
  b₁ > 0 ∧ b₂ > 0 ∧
  b₁ % 12 = 0 ∧ b₂ % 12 = 0 ∧
  80 * (b₁ + b₂) / 2 = 2800 :=
sorry

end NUMINAMATH_CALUDE_no_trapezoid_solution_l165_16512


namespace NUMINAMATH_CALUDE_triple_composition_even_l165_16537

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Given an even function f, prove that f(f(f(x))) is also even -/
theorem triple_composition_even (f : ℝ → ℝ) (hf : IsEven f) : IsEven (fun x ↦ f (f (f x))) := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l165_16537


namespace NUMINAMATH_CALUDE_vincent_laundry_theorem_l165_16517

/-- Represents the types of laundry loads --/
inductive LoadType
  | Regular
  | Delicate
  | Heavy

/-- Represents a day's laundry schedule --/
structure DaySchedule where
  regular : Nat
  delicate : Nat
  heavy : Nat

/-- Calculate total loads for a day --/
def totalLoads (schedule : DaySchedule) : Nat :=
  schedule.regular + schedule.delicate + schedule.heavy

/-- Vincent's laundry week --/
def laundryWeek : List DaySchedule :=
  [
    { regular := 2, delicate := 1, heavy := 3 },  -- Wednesday
    { regular := 4, delicate := 2, heavy := 4 },  -- Thursday
    { regular := 2, delicate := 1, heavy := 0 },  -- Friday
    { regular := 0, delicate := 0, heavy := 1 }   -- Saturday
  ]

theorem vincent_laundry_theorem :
  (laundryWeek.map totalLoads).sum = 20 := by
  sorry

#eval (laundryWeek.map totalLoads).sum

end NUMINAMATH_CALUDE_vincent_laundry_theorem_l165_16517


namespace NUMINAMATH_CALUDE_arcs_not_exceeding_120_degrees_l165_16567

/-- Given 21 points on a circle, the number of arcs with these points as endpoints
    that have a measure of no more than 120° is equal to 100. -/
theorem arcs_not_exceeding_120_degrees (n : ℕ) (h : n = 21) : 
  (n.choose 2) - (n - 1) * (n / 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_arcs_not_exceeding_120_degrees_l165_16567


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l165_16511

theorem subtraction_of_decimals : 7.25 - 3.1 - 1.05 = 3.10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l165_16511


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l165_16571

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The problem statement -/
theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ → ℝ × ℝ := fun y ↦ (1, -2*y)
  ∃ y : ℝ, are_parallel a (b y) ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l165_16571


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l165_16568

/-- Given a circle and a line intersecting at two points, prove the value of m and the equation of the circle with the intersection points as diameter. -/
theorem circle_intersection_theorem (x y : ℝ) (m : ℝ) : 
  let circle := x^2 + y^2 - 2*x - 4*y + m
  let line := x + 2*y - 4
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (circle = 0 ∧ line = 0 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) ∧
    (x₁ * x₂ + y₁ * y₂ = 0) →
    (m = 8/5 ∧ 
     ∀ (x y : ℝ), x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
       ((x - x₁)*(x - x₂) + (y - y₁)*(y - y₂) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l165_16568


namespace NUMINAMATH_CALUDE_prime_factors_of_2_pow_8_minus_1_l165_16557

theorem prime_factors_of_2_pow_8_minus_1 :
  ∃ (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 2^8 - 1 ∧
    p + q + r = 25 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_2_pow_8_minus_1_l165_16557


namespace NUMINAMATH_CALUDE_four_row_grid_has_sixteen_triangles_l165_16590

/-- Represents a triangular grid with a given number of rows at the base -/
structure TriangularGrid where
  baseRows : Nat

/-- Calculates the number of small triangles in a triangular grid -/
def smallTriangles (grid : TriangularGrid) : Nat :=
  (grid.baseRows * (grid.baseRows + 1)) / 2

/-- Calculates the number of medium triangles in a triangular grid -/
def mediumTriangles (grid : TriangularGrid) : Nat :=
  ((grid.baseRows - 1) * grid.baseRows) / 2

/-- Calculates the number of large triangles in a triangular grid -/
def largeTriangles (grid : TriangularGrid) : Nat :=
  if grid.baseRows ≥ 3 then 1 else 0

/-- Calculates the total number of triangles in a triangular grid -/
def totalTriangles (grid : TriangularGrid) : Nat :=
  smallTriangles grid + mediumTriangles grid + largeTriangles grid

/-- Theorem: A triangular grid with 4 rows at the base has 16 total triangles -/
theorem four_row_grid_has_sixteen_triangles :
  totalTriangles { baseRows := 4 } = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_row_grid_has_sixteen_triangles_l165_16590


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_three_l165_16521

theorem smallest_two_digit_multiple_of_three : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 99) ∧ 
  n % 3 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 10 ∧ m ≤ 99) ∧ m % 3 = 0 → n ≤ m) ∧
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_three_l165_16521


namespace NUMINAMATH_CALUDE_fruit_count_l165_16551

theorem fruit_count (total fruits apples oranges bananas : ℕ) : 
  total = 12 → 
  apples = 3 → 
  oranges = 5 → 
  total = apples + oranges + bananas → 
  bananas = 4 :=
by sorry

end NUMINAMATH_CALUDE_fruit_count_l165_16551


namespace NUMINAMATH_CALUDE_money_problem_l165_16539

theorem money_problem (a b : ℝ) : 
  (4 * a - b = 40) ∧ (6 * a + b = 110) → a = 15 ∧ b = 20 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l165_16539


namespace NUMINAMATH_CALUDE_new_routine_exceeds_usual_l165_16535

/-- Represents Terrell's workout routine -/
structure WorkoutRoutine where
  usual_heavy_weight : ℕ
  usual_light_weight : ℕ
  usual_heavy_reps : ℕ
  usual_light_reps : ℕ
  new_heavy_weight : ℕ
  new_light_weight : ℕ

/-- Calculates the total weight lifted in the usual routine -/
def usual_total_weight (w : WorkoutRoutine) : ℕ :=
  2 * w.usual_heavy_weight * w.usual_heavy_reps + 2 * w.usual_light_weight * w.usual_light_reps

/-- Calculates the total weight lifted in the new routine -/
def new_total_weight (w : WorkoutRoutine) (reps : ℕ) : ℕ :=
  (w.new_heavy_weight + w.new_light_weight) * reps

/-- Theorem stating that 19 reps of the new routine exceeds the usual total weight -/
theorem new_routine_exceeds_usual (w : WorkoutRoutine) :
  w.usual_heavy_weight = 20 →
  w.usual_light_weight = 10 →
  w.usual_heavy_reps = 12 →
  w.usual_light_reps = 8 →
  w.new_heavy_weight = 20 →
  w.new_light_weight = 15 →
  new_total_weight w 19 > usual_total_weight w := by
  sorry

end NUMINAMATH_CALUDE_new_routine_exceeds_usual_l165_16535


namespace NUMINAMATH_CALUDE_mens_wages_l165_16559

/-- Proves that the wage of one man is 24 Rs given the problem conditions -/
theorem mens_wages (men women boys : ℕ) (total_earnings : ℚ) : 
  men = 5 → 
  boys = 8 → 
  total_earnings = 120 → 
  ∃ (w : ℕ), (5 : ℚ) * (total_earnings / (men + w + boys)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_mens_wages_l165_16559


namespace NUMINAMATH_CALUDE_composite_power_sum_l165_16518

theorem composite_power_sum (n : ℕ) (h : n > 1) :
  ∃ (k : ℕ), k > 1 ∧ k ∣ ((2^(2^(n+1)) + 2^(2^n) + 1) / 3) := by
  sorry

end NUMINAMATH_CALUDE_composite_power_sum_l165_16518


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l165_16584

/-- A random variable following a normal distribution -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for the normal distribution -/
noncomputable def probability (X : NormalDistribution) (a b : ℝ) : ℝ :=
  sorry

theorem normal_distribution_probability (X : NormalDistribution) 
  (h1 : X.μ = 4)
  (h2 : X.σ = 1)
  (h3 : probability X (X.μ - 2 * X.σ) (X.μ + 2 * X.σ) = 0.9544)
  (h4 : probability X (X.μ - X.σ) (X.μ + X.σ) = 0.6826) :
  probability X 5 6 = 0.1359 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l165_16584


namespace NUMINAMATH_CALUDE_polynomial_remainder_l165_16552

def polynomial (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 2*x^6 - 8*x^4 + 6*x^3 - 9

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 26207 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l165_16552


namespace NUMINAMATH_CALUDE_cat_weight_ratio_l165_16566

def megs_cat_weight : ℕ := 20
def weight_difference : ℕ := 8

def annes_cat_weight : ℕ := megs_cat_weight + weight_difference

theorem cat_weight_ratio :
  (megs_cat_weight : ℚ) / annes_cat_weight = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_cat_weight_ratio_l165_16566


namespace NUMINAMATH_CALUDE_dogwood_trees_in_other_part_l165_16549

/-- The number of dogwood trees in the first part of the park -/
def trees_in_first_part : ℝ := 5.0

/-- The number of trees park workers plan to cut down -/
def planned_trees_to_cut : ℝ := 7.0

/-- The number of park workers on the job -/
def park_workers : ℝ := 8.0

/-- The number of dogwood trees left in the park after the work is done -/
def trees_left_after_work : ℝ := 2.0

/-- The number of dogwood trees in the other part of the park -/
def trees_in_other_part : ℝ := trees_left_after_work

theorem dogwood_trees_in_other_part : 
  trees_in_other_part = 2.0 :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_other_part_l165_16549


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l165_16583

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The number of bananas that are worth as much as 9 oranges -/
def bananas_equal_to_9_oranges : ℚ := (3/4) * 12

/-- The number of bananas we want to find the orange equivalent for -/
def bananas_to_convert : ℚ := (1/3) * 6

theorem banana_orange_equivalence : 
  banana_value * bananas_to_convert = 2 := by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l165_16583


namespace NUMINAMATH_CALUDE_cloth_selling_price_l165_16515

/-- Calculates the total selling price of cloth given the quantity, loss per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (loss_per_meter : ℚ) (cost_price_per_meter : ℚ) : ℚ :=
  quantity * (cost_price_per_meter - loss_per_meter)

/-- Proves that the total selling price for 400 meters of cloth is $18,000 given the specified conditions. -/
theorem cloth_selling_price :
  total_selling_price 400 5 50 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l165_16515
