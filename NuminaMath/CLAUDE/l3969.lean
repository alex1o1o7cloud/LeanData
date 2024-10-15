import Mathlib

namespace NUMINAMATH_CALUDE_floor_equation_solution_l3969_396955

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 3/2⌋ = ⌊x + 3⌋ ↔ 7/3 ≤ x ∧ x < 8/3 :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3969_396955


namespace NUMINAMATH_CALUDE_product_sum_in_base_l3969_396909

/-- Given a base b, convert a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, convert a number from base 10 to base b -/
def fromBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Check if a number is valid in a given base -/
def isValidInBase (n : ℕ) (b : ℕ) : Prop := sorry

theorem product_sum_in_base (b : ℕ) : 
  (b > 1) →
  (isValidInBase 14 b) →
  (isValidInBase 17 b) →
  (isValidInBase 18 b) →
  (isValidInBase 4356 b) →
  (toBase10 14 b * toBase10 17 b * toBase10 18 b = toBase10 4356 b) →
  (fromBase10 (toBase10 14 b + toBase10 17 b + toBase10 18 b) b = 39) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_in_base_l3969_396909


namespace NUMINAMATH_CALUDE_expression_simplification_l3969_396904

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 4) (h2 : x ≠ 2) (h3 : x ≠ 3) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 9) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2) * (x - 3)) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3969_396904


namespace NUMINAMATH_CALUDE_family_age_problem_l3969_396938

/-- Represents the ages of family members at different points in time -/
structure FamilyAges where
  grandpa : ℕ
  dad : ℕ
  xiaoming : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages1 ages2 : FamilyAges) : Prop :=
  ages1.grandpa = 2 * ages1.dad ∧
  ages1.xiaoming = 1 ∧
  ages2.dad = 8 * ages2.xiaoming ∧
  ages2.grandpa = 61

/-- The theorem to be proved -/
theorem family_age_problem (ages1 ages2 : FamilyAges) 
  (h : problem_conditions ages1 ages2) : 
  (∃ (ages3 : FamilyAges), 
    ages3.grandpa - ages3.xiaoming = 57 ∧ 
    ages3.grandpa = 20 * ages3.xiaoming ∧ 
    ages3.dad = 31) :=
sorry

end NUMINAMATH_CALUDE_family_age_problem_l3969_396938


namespace NUMINAMATH_CALUDE_peanuts_in_box_l3969_396932

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) :
  initial_peanuts = 4 → added_peanuts = 12 → initial_peanuts + added_peanuts = 16 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l3969_396932


namespace NUMINAMATH_CALUDE_candy_game_solution_l3969_396914

/-- The number of questions Vanya answered correctly in the candy game -/
def correct_answers : ℕ := 15

/-- The total number of questions asked in the game -/
def total_questions : ℕ := 50

/-- The number of candies gained for a correct answer -/
def correct_reward : ℕ := 7

/-- The number of candies lost for an incorrect answer -/
def incorrect_penalty : ℕ := 3

theorem candy_game_solution :
  correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty :=
by sorry

end NUMINAMATH_CALUDE_candy_game_solution_l3969_396914


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l3969_396961

def is_palindrome (n : ℕ) : Prop := sorry

def initial_reading : ℕ := 12321
def drive_duration : ℝ := 4
def speed_limit : ℝ := 80
def min_average_speed : ℝ := 60

theorem greatest_possible_average_speed :
  ∀ (final_reading : ℕ),
    is_palindrome initial_reading →
    is_palindrome final_reading →
    final_reading > initial_reading →
    (final_reading - initial_reading : ℝ) / drive_duration > min_average_speed →
    (final_reading - initial_reading : ℝ) / drive_duration ≤ speed_limit →
    (∀ (other_reading : ℕ),
      is_palindrome other_reading →
      other_reading > initial_reading →
      (other_reading - initial_reading : ℝ) / drive_duration > min_average_speed →
      (other_reading - initial_reading : ℝ) / drive_duration ≤ speed_limit →
      (other_reading - initial_reading : ℝ) / drive_duration ≤ (final_reading - initial_reading : ℝ) / drive_duration) →
    (final_reading - initial_reading : ℝ) / drive_duration = 75 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l3969_396961


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3969_396989

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 8 * x + c = 0) →  -- Exactly one solution
  (a + c = 10) →                     -- Sum condition
  (a < c) →                          -- Inequality condition
  (a = 2 ∧ c = 8) :=                 -- Conclusion
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3969_396989


namespace NUMINAMATH_CALUDE_darnel_jogging_distance_l3969_396963

theorem darnel_jogging_distance (sprint_distance : Real) (extra_sprint : Real) : 
  sprint_distance = 0.875 →
  extra_sprint = 0.125 →
  sprint_distance = (sprint_distance - extra_sprint) + extra_sprint →
  (sprint_distance - extra_sprint) = 0.750 := by
sorry

end NUMINAMATH_CALUDE_darnel_jogging_distance_l3969_396963


namespace NUMINAMATH_CALUDE_diamond_two_neg_five_l3969_396908

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_two_neg_five : diamond 2 (-5) = 56 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_neg_five_l3969_396908


namespace NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l3969_396984

theorem least_prime_angle_in_right_triangle : ∀ a b : ℕ,
  (a > b) →
  (a + b = 90) →
  (Nat.Prime a) →
  (Nat.Prime b) →
  (∀ c : ℕ, (c < b) → (c + a ≠ 90 ∨ ¬(Nat.Prime c))) →
  b = 7 :=
by sorry

end NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l3969_396984


namespace NUMINAMATH_CALUDE_family_ages_solution_l3969_396913

/-- Represents the ages of a family with two parents and two children -/
structure FamilyAges where
  father : ℕ
  mother : ℕ
  older_son : ℕ
  younger_son : ℕ

/-- The conditions of the family ages problem -/
def family_ages_conditions (ages : FamilyAges) : Prop :=
  ages.father = ages.mother + 3 ∧
  ages.older_son = ages.younger_son + 4 ∧
  ages.father + ages.mother + ages.older_son + ages.younger_son = 81 ∧
  ages.father + ages.mother + ages.older_son + max (ages.younger_son - 5) 0 = 62

/-- The theorem stating the solution to the family ages problem -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), family_ages_conditions ages ∧
    ages.father = 36 ∧ ages.mother = 33 ∧ ages.older_son = 8 ∧ ages.younger_son = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l3969_396913


namespace NUMINAMATH_CALUDE_complex_numbers_with_extreme_arguments_l3969_396928

open Complex

/-- The complex numbers with smallest and largest arguments satisfying |z - 5 - 5i| = 5 -/
theorem complex_numbers_with_extreme_arguments :
  ∃ (z₁ z₂ : ℂ),
    (∀ z : ℂ, abs (z - (5 + 5*I)) = 5 →
      arg z₁ ≤ arg z ∧ arg z ≤ arg z₂) ∧
    z₁ = 5 ∧
    z₂ = 5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_numbers_with_extreme_arguments_l3969_396928


namespace NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l3969_396935

theorem existence_of_n_with_k_prime_factors 
  (k : Nat) 
  (m : Nat) 
  (hk : k ≠ 0) 
  (hm : Odd m) :
  ∃ n : Nat, (Nat.factors (m^n + n^m)).card ≥ k :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l3969_396935


namespace NUMINAMATH_CALUDE_expression_value_for_a_one_third_l3969_396995

theorem expression_value_for_a_one_third :
  let a : ℚ := 1/3
  (4 * a⁻¹ - (2 * a⁻¹) / 3) / (a^2) = 90 := by sorry

end NUMINAMATH_CALUDE_expression_value_for_a_one_third_l3969_396995


namespace NUMINAMATH_CALUDE_multiply_and_add_l3969_396919

theorem multiply_and_add : 45 * 28 + 45 * 72 + 45 = 4545 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l3969_396919


namespace NUMINAMATH_CALUDE_bus_stop_time_l3969_396917

/-- Proves that a bus with given speeds stops for 4 minutes per hour -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 90) 
  (h2 : speed_with_stops = 84) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l3969_396917


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l3969_396977

/-- The perimeter of a semi-circle with radius 31.50774690151576 cm is 162.12300409103152 cm. -/
theorem semicircle_perimeter : 
  let r : ℝ := 31.50774690151576
  let π : ℝ := Real.pi
  let semicircle_perimeter : ℝ := π * r + 2 * r
  semicircle_perimeter = 162.12300409103152 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l3969_396977


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3969_396937

theorem quadratic_solution_sum (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (∃ x : ℝ, x^2 + 14*x = 65 ∧ x > 0 ∧ x = Real.sqrt a - b) →
  a + b = 121 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3969_396937


namespace NUMINAMATH_CALUDE_haley_music_files_l3969_396940

theorem haley_music_files :
  ∀ (initial_music_files : ℕ),
    initial_music_files + 42 - 11 = 58 →
    initial_music_files = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_haley_music_files_l3969_396940


namespace NUMINAMATH_CALUDE_age_difference_l3969_396906

/-- Proves that the age difference between a man and his son is 30 years -/
theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 28 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 30 := by
  sorry

#check age_difference

end NUMINAMATH_CALUDE_age_difference_l3969_396906


namespace NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l3969_396924

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line L
def line (x y : ℝ) : Prop :=
  y = 3/4 * (x - 3)

theorem ellipse_intersection_midpoint :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  ellipse a b 0 4 →
  (a^2 - b^2) / a^2 = (3/5)^2 →
  ∃ x1 x2 y1 y2 : ℝ,
    ellipse a b x1 y1 ∧
    ellipse a b x2 y2 ∧
    line x1 y1 ∧
    line x2 y2 ∧
    (x1 + x2) / 2 = 1 ∧
    (y1 + y2) / 2 = -9/4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_midpoint_l3969_396924


namespace NUMINAMATH_CALUDE_complete_square_sum_l3969_396986

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x, 64 * x^2 + 48 * x - 36 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 56 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l3969_396986


namespace NUMINAMATH_CALUDE_farm_problem_l3969_396918

/-- Represents the number of animals on a farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Represents the conditions of the farm problem -/
def farm_conditions (f : FarmAnimals) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows > f.goats ∧
  f.cows + f.pigs + f.goats = 56 ∧
  f.goats = 11

/-- Theorem stating that under the given conditions, the farmer has 4 more cows than goats -/
theorem farm_problem (f : FarmAnimals) (h : farm_conditions f) : f.cows - f.goats = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_problem_l3969_396918


namespace NUMINAMATH_CALUDE_boat_trips_theorem_l3969_396972

/-- The number of boat trips in one day -/
def boat_trips_per_day (boat_capacity : ℕ) (people_per_two_days : ℕ) : ℕ :=
  (people_per_two_days / 2) / boat_capacity

/-- Theorem: The number of boat trips in one day is 4 -/
theorem boat_trips_theorem :
  boat_trips_per_day 12 96 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_trips_theorem_l3969_396972


namespace NUMINAMATH_CALUDE_min_value_of_function_l3969_396950

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  ∃ (y_min : ℝ), y_min = 4 * Real.sqrt 2 + 1 ∧
  ∀ (y : ℝ), y = 2 * x + 4 / (x - 1) - 1 → y ≥ y_min := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3969_396950


namespace NUMINAMATH_CALUDE_smallest_unpayable_amount_l3969_396988

/-- Represents the number of coins of each denomination -/
structure CoinCollection where
  fiveP : Nat
  fourP : Nat
  threeP : Nat
  twoP : Nat
  oneP : Nat

/-- Calculates the total value of a coin collection in pence -/
def totalValue (coins : CoinCollection) : Nat :=
  5 * coins.fiveP + 4 * coins.fourP + 3 * coins.threeP + 2 * coins.twoP + coins.oneP

/-- Checks if a given amount can be paid using the coin collection -/
def canPay (coins : CoinCollection) (amount : Nat) : Prop :=
  ∃ (a b c d e : Nat),
    a ≤ coins.fiveP ∧
    b ≤ coins.fourP ∧
    c ≤ coins.threeP ∧
    d ≤ coins.twoP ∧
    e ≤ coins.oneP ∧
    5 * a + 4 * b + 3 * c + 2 * d + e = amount

/-- Edward's coin collection -/
def edwardCoins : CoinCollection :=
  { fiveP := 5, fourP := 4, threeP := 3, twoP := 2, oneP := 1 }

theorem smallest_unpayable_amount :
  (∀ n < 56, canPay edwardCoins n) ∧ ¬(canPay edwardCoins 56) := by
  sorry

end NUMINAMATH_CALUDE_smallest_unpayable_amount_l3969_396988


namespace NUMINAMATH_CALUDE_zero_meetings_on_circular_track_l3969_396943

/-- Represents the number of meetings between two people moving on a circular track. -/
def number_of_meetings (circumference : ℝ) (speed_forward : ℝ) (speed_backward : ℝ) : ℕ :=
  -- The actual calculation is not implemented, as we only need the statement
  sorry

/-- Theorem stating that the number of meetings is 0 under the given conditions. -/
theorem zero_meetings_on_circular_track :
  let circumference : ℝ := 270
  let speed_forward : ℝ := 6
  let speed_backward : ℝ := 3
  number_of_meetings circumference speed_forward speed_backward = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_meetings_on_circular_track_l3969_396943


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3969_396954

theorem inequality_equivalence (x : ℝ) : 
  (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 6) ↔ (2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3969_396954


namespace NUMINAMATH_CALUDE_novel_reading_ratio_l3969_396925

theorem novel_reading_ratio :
  ∀ (jordan alexandre : ℕ),
    jordan = 120 →
    jordan = alexandre + 108 →
    (alexandre : ℚ) / jordan = 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_novel_reading_ratio_l3969_396925


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l3969_396979

theorem quadratic_roots_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
   x₁^2 + a*x₁ + a^2 - 1 = 0 ∧ x₂^2 + a*x₂ + a^2 - 1 = 0) →
  -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l3969_396979


namespace NUMINAMATH_CALUDE_binomial_coeff_congruence_l3969_396931

-- Define the binomial coefficient
def binomial_coeff (n p : ℕ) : ℕ := Nat.choose n p

-- State the theorem
theorem binomial_coeff_congruence (p n : ℕ) 
  (hp : Nat.Prime p) 
  (hodd : Odd p) 
  (hn : n ≥ p) :
  binomial_coeff n p ≡ (n / p) [MOD p] :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_congruence_l3969_396931


namespace NUMINAMATH_CALUDE_ball_probabilities_l3969_396905

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The probability of drawing a red ball -/
def prob_red : ℚ := red_balls / total_balls

/-- The probability of drawing two black balls without replacement -/
def prob_two_black : ℚ := (black_balls * (black_balls - 1)) / (total_balls * (total_balls - 1))

theorem ball_probabilities :
  prob_red = 2/5 ∧ prob_two_black = 3/10 :=
sorry

end NUMINAMATH_CALUDE_ball_probabilities_l3969_396905


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l3969_396916

universe u

def U : Set (Fin 7) := {1, 2, 3, 4, 5, 6, 7}
def P : Set (Fin 7) := {1, 2, 3, 4, 5}
def Q : Set (Fin 7) := {3, 4, 5, 6, 7}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l3969_396916


namespace NUMINAMATH_CALUDE_miles_collection_height_l3969_396998

/-- Represents the height of a book collection in inches and pages -/
structure BookCollection where
  height_inches : ℝ
  total_pages : ℝ

/-- Calculates the total pages in a book collection given the height in inches and pages per inch -/
def total_pages (height : ℝ) (pages_per_inch : ℝ) : ℝ :=
  height * pages_per_inch

theorem miles_collection_height 
  (miles_ratio : ℝ) 
  (daphne_ratio : ℝ) 
  (daphne_height : ℝ) 
  (longest_collection_pages : ℝ)
  (h1 : miles_ratio = 5)
  (h2 : daphne_ratio = 50)
  (h3 : daphne_height = 25)
  (h4 : longest_collection_pages = 1250)
  (h5 : total_pages daphne_height daphne_ratio = longest_collection_pages) :
  ∃ (miles_collection : BookCollection), 
    miles_collection.height_inches = 250 ∧ 
    miles_collection.total_pages = longest_collection_pages :=
sorry

end NUMINAMATH_CALUDE_miles_collection_height_l3969_396998


namespace NUMINAMATH_CALUDE_opposite_of_pi_l3969_396978

theorem opposite_of_pi : -(Real.pi) = -Real.pi := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_pi_l3969_396978


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3969_396969

/-- Given a line L1 with equation mx - m^2y = 1 and a point P(2,1) on L1,
    prove that the line L2 perpendicular to L1 at P has equation x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ m * x - m^2 * y = 1
  let P : ℝ × ℝ := (2, 1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + y - 3 = 0
  L1 P.1 P.2 → L2 = (λ x y ↦ x + y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3969_396969


namespace NUMINAMATH_CALUDE_positive_solution_x_l3969_396987

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 12 - 2 * x - 3 * y)
  (eq2 : y * z = 8 - 4 * y - 2 * z)
  (eq3 : x * z = 24 - 4 * x - 3 * z)
  (x_pos : x > 0) :
  x = 3 := by sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3969_396987


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3969_396927

/-- A quadratic function f(x) = (x + 1)² has a symmetry axis of x = -1 -/
theorem quadratic_symmetry_axis (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x + 1)^2
  ∀ y : ℝ, f (-1 - y) = f (-1 + y) := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3969_396927


namespace NUMINAMATH_CALUDE_andrew_work_hours_l3969_396900

/-- The number of days Andrew worked on his Science report -/
def days_worked : ℝ := 3

/-- The number of hours Andrew worked each day -/
def hours_per_day : ℝ := 2.5

/-- The total number of hours Andrew worked -/
def total_hours : ℝ := days_worked * hours_per_day

theorem andrew_work_hours : total_hours = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l3969_396900


namespace NUMINAMATH_CALUDE_inequality_proof_l3969_396947

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3969_396947


namespace NUMINAMATH_CALUDE_bridge_extension_l3969_396959

theorem bridge_extension (river_width bridge_length : ℕ) 
  (hw : river_width = 487) 
  (hb : bridge_length = 295) : 
  river_width - bridge_length = 192 := by
sorry

end NUMINAMATH_CALUDE_bridge_extension_l3969_396959


namespace NUMINAMATH_CALUDE_middle_number_is_eight_l3969_396996

/-- A sequence of 11 numbers satisfying the given conditions -/
def Sequence := Fin 11 → ℝ

/-- The property that the sum of any three consecutive numbers is 18 -/
def ConsecutiveSum (s : Sequence) : Prop :=
  ∀ i : Fin 9, s i + s (i + 1) + s (i + 2) = 18

/-- The property that the sum of all numbers is 64 -/
def TotalSum (s : Sequence) : Prop :=
  (Finset.univ.sum s) = 64

/-- The theorem stating that the middle number is 8 -/
theorem middle_number_is_eight (s : Sequence) 
  (h1 : ConsecutiveSum s) (h2 : TotalSum s) : s 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_eight_l3969_396996


namespace NUMINAMATH_CALUDE_equipment_production_theorem_l3969_396902

theorem equipment_production_theorem
  (total_production : ℕ)
  (sample_size : ℕ)
  (sample_from_A : ℕ)
  (h1 : total_production = 4800)
  (h2 : sample_size = 80)
  (h3 : sample_from_A = 50)
  : (total_production - (sample_from_A * (total_production / sample_size))) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_equipment_production_theorem_l3969_396902


namespace NUMINAMATH_CALUDE_probability_even_rolls_l3969_396960

def is_even (n : ℕ) : Bool := n % 2 = 0

def count_even (n : ℕ) : ℕ := (List.range n).filter is_even |>.length

theorem probability_even_rolls (die1 : ℕ) (die2 : ℕ) 
  (h1 : die1 = 6) (h2 : die2 = 7) : 
  (count_even die1 : ℚ) / die1 * (count_even die2 : ℚ) / die2 = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_rolls_l3969_396960


namespace NUMINAMATH_CALUDE_reflection_coordinate_sum_l3969_396965

/-- Given a point A with coordinates (x, 7), prove that the sum of its coordinates
    and the coordinates of its reflection B over the x-axis is 2x. -/
theorem reflection_coordinate_sum (x : ℝ) : 
  let A : ℝ × ℝ := (x, 7)
  let B : ℝ × ℝ := (x, -7)  -- Reflection of A over x-axis
  (A.1 + A.2 + B.1 + B.2) = 2 * x := by
sorry

end NUMINAMATH_CALUDE_reflection_coordinate_sum_l3969_396965


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l3969_396991

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_12_less_than_square : 
  (∃ n : ℕ, is_perfect_square n ∧ is_prime (n - 12)) ∧ 
  (∀ m : ℕ, is_perfect_square m ∧ is_prime (m - 12) → m - 12 ≥ 13) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l3969_396991


namespace NUMINAMATH_CALUDE_swim_team_ratio_l3969_396948

theorem swim_team_ratio (total : ℕ) (girls : ℕ) (h1 : total = 96) (h2 : girls = 80) :
  (girls : ℚ) / (total - girls) = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_ratio_l3969_396948


namespace NUMINAMATH_CALUDE_bee_count_l3969_396967

/-- The number of bees initially in the hive -/
def initial_bees : ℕ := 16

/-- The number of bees that flew in -/
def new_bees : ℕ := 10

/-- The total number of bees in the hive -/
def total_bees : ℕ := initial_bees + new_bees

theorem bee_count : total_bees = 26 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l3969_396967


namespace NUMINAMATH_CALUDE_triangle_theorem_l3969_396964

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * sin (t.A - t.C) = t.b * (sin t.A - sin t.B)

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.c = 4) : 
  t.C = π/3 ∧ 
  (∀ (t' : Triangle), satisfiesCondition t' → t'.c = 4 → 
    t.a + t.b + t.c ≤ 12) :=
sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l3969_396964


namespace NUMINAMATH_CALUDE_two_sin_plus_cos_value_cos_and_tan_values_l3969_396901

-- Define the angle α
variable (α : Real)

-- Define the point P
structure Point where
  x : Real
  y : Real

-- Theorem 1
theorem two_sin_plus_cos_value (P : Point) (h1 : P.x = 4) (h2 : P.y = -3) :
  2 * Real.sin α + Real.cos α = -2/5 := by sorry

-- Theorem 2
theorem cos_and_tan_values (P : Point) (m : Real) 
  (h1 : P.x = -Real.sqrt 3) (h2 : P.y = m) (h3 : m ≠ 0)
  (h4 : Real.sin α = (Real.sqrt 2 * m) / 4) :
  Real.cos α = -Real.sqrt 3 / 5 ∧ 
  (Real.tan α = -Real.sqrt 10 / 3 ∨ Real.tan α = Real.sqrt 10 / 3) := by sorry

end NUMINAMATH_CALUDE_two_sin_plus_cos_value_cos_and_tan_values_l3969_396901


namespace NUMINAMATH_CALUDE_grass_seed_cost_l3969_396921

structure GrassSeed where
  bag5_price : ℝ
  bag10_price : ℝ
  bag25_price : ℝ
  min_purchase : ℝ
  max_purchase : ℝ
  least_cost : ℝ

def is_valid_purchase (gs : GrassSeed) (x : ℕ) (y : ℕ) (z : ℕ) : Prop :=
  5 * x + 10 * y + 25 * z ≥ gs.min_purchase ∧
  5 * x + 10 * y + 25 * z ≤ gs.max_purchase

def total_cost (gs : GrassSeed) (x : ℕ) (y : ℕ) (z : ℕ) : ℝ :=
  gs.bag5_price * x + gs.bag10_price * y + gs.bag25_price * z

theorem grass_seed_cost (gs : GrassSeed) 
  (h1 : gs.bag5_price = 13.80)
  (h2 : gs.bag10_price = 20.43)
  (h3 : gs.min_purchase = 65)
  (h4 : gs.max_purchase = 80)
  (h5 : gs.least_cost = 98.73) :
  gs.bag25_price = 17.01 :=
by sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l3969_396921


namespace NUMINAMATH_CALUDE_comparison_of_exponents_l3969_396990

theorem comparison_of_exponents :
  (1.7 ^ 2.5 < 1.7 ^ 3) ∧
  (0.8 ^ (-0.1) < 0.8 ^ (-0.2)) ∧
  (1.7 ^ 0.3 > 0.9 ^ 3.1) ∧
  ((1/3) ^ (1/3) < (1/4) ^ (1/4)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_exponents_l3969_396990


namespace NUMINAMATH_CALUDE_square_side_increase_l3969_396968

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let side_b := 2 * a
  let area_a := a^2
  let area_b := side_b^2
  let area_c := (area_a + area_b) * 2.45
  let side_c := Real.sqrt area_c
  (side_c - side_b) / side_b = 0.75 := by sorry

end NUMINAMATH_CALUDE_square_side_increase_l3969_396968


namespace NUMINAMATH_CALUDE_outfits_count_l3969_396992

/-- The number of outfits with different colored shirts and hats -/
def num_outfits : ℕ :=
  let red_shirts := 5
  let green_shirts := 5
  let pants := 6
  let green_hats := 8
  let red_hats := 8
  (red_shirts * pants * green_hats) + (green_shirts * pants * red_hats)

theorem outfits_count : num_outfits = 480 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3969_396992


namespace NUMINAMATH_CALUDE_decreasing_then_increasing_possible_increasing_then_decreasing_impossible_l3969_396958

/-- Definition of our sequence based on original positive numbers -/
def sequence_a (original : List ℝ) : ℕ → ℝ :=
  λ n => (original.map (λ x => x ^ n)).sum

/-- Theorem stating the possibility of decreasing then increasing sequence -/
theorem decreasing_then_increasing_possible :
  ∃ original : List ℝ, 
    (∀ x ∈ original, x > 0) ∧
    (sequence_a original 1 > sequence_a original 2) ∧
    (sequence_a original 2 > sequence_a original 3) ∧
    (sequence_a original 3 > sequence_a original 4) ∧
    (sequence_a original 4 > sequence_a original 5) ∧
    (∀ n ≥ 5, sequence_a original n < sequence_a original (n + 1)) :=
sorry

/-- Theorem stating the impossibility of increasing then decreasing sequence -/
theorem increasing_then_decreasing_impossible :
  ¬∃ original : List ℝ, 
    (∀ x ∈ original, x > 0) ∧
    (sequence_a original 1 < sequence_a original 2) ∧
    (sequence_a original 2 < sequence_a original 3) ∧
    (sequence_a original 3 < sequence_a original 4) ∧
    (sequence_a original 4 < sequence_a original 5) ∧
    (∀ n ≥ 5, sequence_a original n > sequence_a original (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_decreasing_then_increasing_possible_increasing_then_decreasing_impossible_l3969_396958


namespace NUMINAMATH_CALUDE_points_4_units_from_neg5_are_neg9_and_neg1_l3969_396930

-- Define the distance between two points on a number line
def distance (x y : ℝ) : ℝ := |x - y|

-- Define the set of points that are 4 units away from -5
def points_4_units_from_neg5 : Set ℝ := {x : ℝ | distance x (-5) = 4}

-- Theorem statement
theorem points_4_units_from_neg5_are_neg9_and_neg1 :
  points_4_units_from_neg5 = {-9, -1} := by sorry

end NUMINAMATH_CALUDE_points_4_units_from_neg5_are_neg9_and_neg1_l3969_396930


namespace NUMINAMATH_CALUDE_gcd_228_1995_l3969_396966

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l3969_396966


namespace NUMINAMATH_CALUDE_f_max_value_l3969_396997

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

/-- The maximum value of f(x) is 22 -/
theorem f_max_value : ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

end NUMINAMATH_CALUDE_f_max_value_l3969_396997


namespace NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l3969_396951

/-- Represents the five points on the circle -/
inductive CirclePoint
  | one
  | two
  | three
  | four
  | five

/-- Determines if a point is odd-numbered -/
def isOdd (p : CirclePoint) : Bool :=
  match p with
  | CirclePoint.one => true
  | CirclePoint.two => false
  | CirclePoint.three => true
  | CirclePoint.four => false
  | CirclePoint.five => true

/-- Determines the next point based on the current point -/
def nextPoint (p : CirclePoint) : CirclePoint :=
  match p with
  | CirclePoint.one => CirclePoint.two
  | CirclePoint.two => CirclePoint.four
  | CirclePoint.three => CirclePoint.four
  | CirclePoint.four => CirclePoint.one
  | CirclePoint.five => CirclePoint.one

/-- Calculates the position after a given number of jumps -/
def positionAfterJumps (start : CirclePoint) (jumps : Nat) : CirclePoint :=
  match jumps with
  | 0 => start
  | n + 1 => nextPoint (positionAfterJumps start n)

theorem bug_position_after_1995_jumps :
  positionAfterJumps CirclePoint.five 1995 = CirclePoint.four := by
  sorry


end NUMINAMATH_CALUDE_bug_position_after_1995_jumps_l3969_396951


namespace NUMINAMATH_CALUDE_steven_peach_apple_difference_l3969_396933

theorem steven_peach_apple_difference :
  ∀ (steven_peaches steven_apples jake_peaches jake_apples : ℕ),
    steven_peaches = 18 →
    steven_apples = 11 →
    jake_peaches = steven_peaches - 8 →
    jake_apples = steven_apples + 10 →
    steven_peaches - steven_apples = 7 := by
  sorry

end NUMINAMATH_CALUDE_steven_peach_apple_difference_l3969_396933


namespace NUMINAMATH_CALUDE_election_votes_l3969_396912

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) 
  (h1 : total_votes = 7000)
  (h2 : invalid_percent = 1/5)
  (h3 : winner_percent = 11/20) :
  ↑total_votes * (1 - invalid_percent) * (1 - winner_percent) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3969_396912


namespace NUMINAMATH_CALUDE_decimal_point_movement_l3969_396993

theorem decimal_point_movement (x : ℝ) : x / 100 = x - 1.485 ↔ x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_movement_l3969_396993


namespace NUMINAMATH_CALUDE_kho_kho_only_count_l3969_396981

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of people who play both games -/
def both_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 30

/-- The number of people who play kho kho only -/
def kho_kho_only_players : ℕ := total_players - (kabadi_players + both_players)

theorem kho_kho_only_count : kho_kho_only_players = 20 := by
  sorry

end NUMINAMATH_CALUDE_kho_kho_only_count_l3969_396981


namespace NUMINAMATH_CALUDE_password_theorem_triangle_password_theorem_factorization_theorem_l3969_396953

-- Part 1
def password_generator (x y : ℤ) : List ℤ :=
  [x * 10000 + (x - y) * 100 + (x + y),
   x * 10000 + (x + y) * 100 + (x - y),
   (x - y) * 10000 + x * 100 + (x + y)]

theorem password_theorem :
  password_generator 21 7 = [211428, 212814, 142128] :=
sorry

-- Part 2
def right_triangle_password (x y : ℝ) : ℝ :=
  x * y * (x^2 + y^2)

theorem triangle_password_theorem :
  ∀ x y : ℝ,
  x + y + 13 = 30 →
  x^2 + y^2 = 13^2 →
  right_triangle_password x y = 10140 :=
sorry

-- Part 3
def polynomial_factorization (m n : ℤ) : Prop :=
  ∀ x : ℤ,
  x^3 + (m - 3*n)*x^2 - n*x - 21 = (x - 3)*(x + 1)*(x + 7)

theorem factorization_theorem :
  polynomial_factorization 56 17 :=
sorry

end NUMINAMATH_CALUDE_password_theorem_triangle_password_theorem_factorization_theorem_l3969_396953


namespace NUMINAMATH_CALUDE_probability_all_scissors_l3969_396983

-- Define the possible choices in the game
inductive Choice
  | Rock
  | Paper
  | Scissors

-- Define a function to calculate the probability of a specific outcome
def probability_of_outcome (num_players : ℕ) (num_choices : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (num_choices ^ num_players : ℚ)

-- Theorem statement
theorem probability_all_scissors :
  let num_players : ℕ := 3
  let num_choices : ℕ := 3
  let favorable_outcomes : ℕ := 1
  probability_of_outcome num_players num_choices favorable_outcomes = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_probability_all_scissors_l3969_396983


namespace NUMINAMATH_CALUDE_matrix_and_transformation_problem_l3969_396945

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -2, -3]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, -2]
def C (x y : ℝ) : Prop := x^2 - 4*x*y + y^2 = 1

theorem matrix_and_transformation_problem :
  ∃ (A_inv X : Matrix (Fin 2) (Fin 2) ℝ) (C' : ℝ → ℝ → Prop),
    (A_inv = !![(-3), (-2); 2, 1]) ∧
    (A * A_inv = 1) ∧ (A_inv * A = 1) ∧
    (X = !![(-2), 1; 1, 0]) ∧
    (A * X = B) ∧
    (∀ x y x' y', C x y ∧ x' = y ∧ y' = x - 2*y → C' x' y') ∧
    (∀ x' y', C' x' y' ↔ 3*x'^2 - y'^2 = -1) := by
  sorry

end NUMINAMATH_CALUDE_matrix_and_transformation_problem_l3969_396945


namespace NUMINAMATH_CALUDE_range_of_p_range_of_p_xor_q_l3969_396929

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def q (a : ℝ) : Prop := |2 * a - 1| < 3

-- Define the set of a for which p is true
def S₁ : Set ℝ := {a : ℝ | p a}

-- Define the set of a for which p∨q is true and p∧q is false
def S₂ : Set ℝ := {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)}

-- Theorem 1
theorem range_of_p : S₁ = Set.Ici 0 ∩ Set.Iio 4 := by sorry

-- Theorem 2
theorem range_of_p_xor_q : S₂ = (Set.Ioi (-1) ∩ Set.Iio 0) ∪ (Set.Ici 2 ∩ Set.Iio 4) := by sorry

end NUMINAMATH_CALUDE_range_of_p_range_of_p_xor_q_l3969_396929


namespace NUMINAMATH_CALUDE_solution_sum_equals_23_l3969_396907

theorem solution_sum_equals_23 (x y a b c d : ℝ) : 
  (x + y = 5) →
  (2 * x * y = 5) →
  (∃ (sign : Bool), x = (a + if sign then b * Real.sqrt c else -b * Real.sqrt c) / d) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (d > 0) →
  (∀ k : ℕ, k > 1 → ¬(∃ (m n : ℤ), a * k = m * d ∧ b * k = n * d)) →
  (a + b + c + d = 23) := by
sorry

end NUMINAMATH_CALUDE_solution_sum_equals_23_l3969_396907


namespace NUMINAMATH_CALUDE_positive_sum_l3969_396973

theorem positive_sum (x y z : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l3969_396973


namespace NUMINAMATH_CALUDE_investment_triple_period_l3969_396985

/-- The annual interest rate as a decimal -/
def r : ℝ := 0.341

/-- The condition for the investment to more than triple -/
def triple_condition (t : ℝ) : Prop := (1 + r) ^ t > 3

/-- The smallest investment period in years for the investment to more than triple -/
def smallest_period : ℕ := 4

theorem investment_triple_period :
  (∀ t : ℝ, t < smallest_period → ¬ triple_condition t) ∧
  triple_condition (smallest_period : ℝ) :=
sorry

end NUMINAMATH_CALUDE_investment_triple_period_l3969_396985


namespace NUMINAMATH_CALUDE_curve_identification_l3969_396982

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := (ρ - 1) * (θ - Real.pi) = 0 ∧ ρ ≥ 0

-- Define the parametric equations
def parametric_equations (x y θ : ℝ) : Prop := x = Real.tan θ ∧ y = 2 / Real.cos θ

-- Theorem statement
theorem curve_identification :
  (∃ (x y : ℝ), x^2 + y^2 = 1) ∧  -- Circle
  (∃ (x y : ℝ), x < 0 ∧ y = 0) ∧  -- Ray
  (∃ (x y : ℝ), y^2 - 4*x^2 = 4)  -- Hyperbola
  :=
sorry

end NUMINAMATH_CALUDE_curve_identification_l3969_396982


namespace NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3969_396903

theorem largest_n_satisfying_conditions : 
  ∃ (m : ℤ), 365^2 = (m+1)^3 - m^3 ∧
  ∃ (a : ℤ), 2*365 + 111 = a^2 ∧
  ∀ (n : ℤ), n > 365 → 
    (∀ (m : ℤ), n^2 ≠ (m+1)^3 - m^3 ∨ 
    ∀ (a : ℤ), 2*n + 111 ≠ a^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_conditions_l3969_396903


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3969_396956

/-- The function f(x) = x(1+ax)^2 --/
def f (a : ℝ) (x : ℝ) : ℝ := x * (1 + a * x)^2

/-- Proposition stating that a = 2/3 is sufficient but not necessary for f(3) = 27 --/
theorem sufficient_not_necessary (a : ℝ) : 
  (f a 3 = 27 ↔ a = 2/3) ↔ False :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3969_396956


namespace NUMINAMATH_CALUDE_expression_simplification_l3969_396952

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) = (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3969_396952


namespace NUMINAMATH_CALUDE_area_less_than_one_third_l3969_396910

theorem area_less_than_one_third (a : ℝ) (h : 1 < a ∧ a < 2) : 
  let f (x : ℝ) := 1 - |x - 1|
  let g (x : ℝ) := |2*x - a|
  let area := (1/6) * |(a - 1)*(a - 2)|
  area < (1/3) :=
by sorry

end NUMINAMATH_CALUDE_area_less_than_one_third_l3969_396910


namespace NUMINAMATH_CALUDE_cube_sum_sqrt_l3969_396923

theorem cube_sum_sqrt : Real.sqrt (4^3 + 4^3 + 4^3) = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cube_sum_sqrt_l3969_396923


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3969_396949

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - 4*x + 3 < 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 < 0 ∧ (x ≤ 1 ∨ x ≥ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3969_396949


namespace NUMINAMATH_CALUDE_min_sum_squares_l3969_396970

theorem min_sum_squares (a b c d : ℝ) 
  (h1 : a + b = 9 / (c - d)) 
  (h2 : c + d = 25 / (a - b)) : 
  ∀ x y z w : ℝ, x^2 + y^2 + z^2 + w^2 ≥ 34 ∧ 
  (∃ a b c d : ℝ, a^2 + b^2 + c^2 + d^2 = 34 ∧ 
   a + b = 9 / (c - d) ∧ c + d = 25 / (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3969_396970


namespace NUMINAMATH_CALUDE_train_length_l3969_396976

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, 
  (abs (length - 129.96) < 0.01) ∧ (length = speed * 1000 / 3600 * time) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3969_396976


namespace NUMINAMATH_CALUDE_sum_of_first_and_third_l3969_396962

theorem sum_of_first_and_third (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B / C = 5 / 8 →
  B = 30 →
  A + C = 68 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_and_third_l3969_396962


namespace NUMINAMATH_CALUDE_quadratic_coefficient_positive_l3969_396939

theorem quadratic_coefficient_positive
  (a b c n : ℤ)
  (h_a_nonzero : a ≠ 0)
  (p : ℤ → ℤ)
  (h_p : ∀ x, p x = a * x^2 + b * x + c)
  (h_ineq : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_positive_l3969_396939


namespace NUMINAMATH_CALUDE_contradiction_assumption_l3969_396975

theorem contradiction_assumption (x y : ℝ) (h : x < y) :
  (¬ (x^3 < y^3)) ↔ (x^3 = y^3 ∨ x^3 > y^3) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l3969_396975


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3969_396999

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 9}

theorem complement_of_A_in_U : 
  (U \ A) = {3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3969_396999


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3969_396911

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water : 
  let current_speed : ℝ := 6
  let downstream_distance : ℝ := 10.67
  let downstream_time : ℝ := 1/3
  let boat_speed : ℝ := 26.01
  (boat_speed + current_speed) * downstream_time = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3969_396911


namespace NUMINAMATH_CALUDE_solve_record_problem_l3969_396980

def record_problem (initial_records : ℕ) (bought_records : ℕ) (days_per_record : ℕ) (total_days : ℕ) : Prop :=
  let total_records := total_days / days_per_record
  let friends_records := total_records - (initial_records + bought_records)
  friends_records = 12

theorem solve_record_problem :
  record_problem 8 30 2 100 := by
  sorry

end NUMINAMATH_CALUDE_solve_record_problem_l3969_396980


namespace NUMINAMATH_CALUDE_full_price_revenue_l3969_396915

/-- Represents the ticket sales data for a charity event -/
structure TicketSales where
  fullPrice : ℕ  -- Price of a full-price ticket in dollars
  fullCount : ℕ  -- Number of full-price tickets sold
  halfCount : ℕ  -- Number of half-price tickets sold
  premiumCount : ℕ := 12  -- Number of premium tickets sold (fixed at 12)

/-- Calculates the total number of tickets sold -/
def TicketSales.totalTickets (ts : TicketSales) : ℕ :=
  ts.fullCount + ts.halfCount + ts.premiumCount

/-- Calculates the total revenue from all ticket sales -/
def TicketSales.totalRevenue (ts : TicketSales) : ℕ :=
  ts.fullPrice * ts.fullCount + 
  (ts.fullPrice / 2) * ts.halfCount + 
  (2 * ts.fullPrice) * ts.premiumCount

/-- Theorem stating the revenue from full-price tickets -/
theorem full_price_revenue (ts : TicketSales) : 
  ts.totalTickets = 160 ∧ 
  ts.totalRevenue = 2514 ∧ 
  ts.fullPrice > 0 →
  ts.fullPrice * ts.fullCount = 770 := by
  sorry

#check full_price_revenue

end NUMINAMATH_CALUDE_full_price_revenue_l3969_396915


namespace NUMINAMATH_CALUDE_two_part_problem_solution_count_l3969_396942

theorem two_part_problem_solution_count 
  (part1_methods : ℕ) 
  (part2_methods : ℕ) 
  (h1 : part1_methods = 2) 
  (h2 : part2_methods = 3) : 
  part1_methods * part2_methods = 6 := by
sorry

end NUMINAMATH_CALUDE_two_part_problem_solution_count_l3969_396942


namespace NUMINAMATH_CALUDE_equation_solution_l3969_396926

theorem equation_solution : 
  ∃ x : ℚ, (x - 1) / 2 = 1 - (x + 2) / 3 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3969_396926


namespace NUMINAMATH_CALUDE_problem_equality_l3969_396957

/-- The function g as defined in the problem -/
def g (n : ℤ) : ℚ := (1 / 4) * n * (n + 1) * (n + 3)

/-- Theorem stating the equality to be proved -/
theorem problem_equality (s : ℤ) : g s - g (s - 1) + s * (s + 1) = 2 * s^2 + 2 * s := by
  sorry

end NUMINAMATH_CALUDE_problem_equality_l3969_396957


namespace NUMINAMATH_CALUDE_total_amount_shared_l3969_396922

/-- Represents the amount of money received by each person -/
structure ShareDistribution where
  john : ℕ
  jose : ℕ
  binoy : ℕ

/-- Defines the ratio of money distribution -/
def ratio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 4
  | 2 => 6

/-- Proves that the total amount shared is 12000 given the conditions -/
theorem total_amount_shared (d : ShareDistribution) 
  (h1 : d.john = 2000)
  (h2 : d.jose = 2 * d.john)
  (h3 : d.binoy = 3 * d.john) : 
  d.john + d.jose + d.binoy = 12000 := by
  sorry

#check total_amount_shared

end NUMINAMATH_CALUDE_total_amount_shared_l3969_396922


namespace NUMINAMATH_CALUDE_leap_stride_difference_l3969_396974

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 56

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 16

/-- The total number of poles -/
def total_poles : ℕ := 51

/-- The distance in feet between the first and last pole -/
def total_distance : ℕ := 8000

/-- Elmer's stride length in feet -/
def elmer_stride_length : ℚ := total_distance / (elmer_strides * (total_poles - 1))

/-- Oscar's leap length in feet -/
def oscar_leap_length : ℚ := total_distance / (oscar_leaps * (total_poles - 1))

theorem leap_stride_difference : 
  ⌊oscar_leap_length - elmer_stride_length⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_leap_stride_difference_l3969_396974


namespace NUMINAMATH_CALUDE_f_lower_bound_l3969_396971

/-- Given f(x) = e^x - x^2 - 1 for all x ∈ ℝ, prove that f(x) ≥ x^2 + x for all x ∈ ℝ. -/
theorem f_lower_bound (x : ℝ) : Real.exp x - x^2 - 1 ≥ x^2 + x := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_l3969_396971


namespace NUMINAMATH_CALUDE_range_of_a_l3969_396936

def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - (2 * a + 5)) > 0}

def B (a : ℝ) : Set ℝ := {x | ((a^2 + 2) - x) * (2 * a - x) < 0}

theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 →
    (B a ⊆ A a) →
    (B a ≠ A a) →
    a ∈ Set.Ioo (1/2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3969_396936


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3969_396934

theorem trigonometric_simplification (α : ℝ) (h : 2 * Real.sin α ^ 2 * (2 * α) - Real.sin (4 * α) ≠ 0) :
  (1 - Real.cos (2 * α)) * Real.cos (π / 4 + α) / (2 * Real.sin α ^ 2 * (2 * α) - Real.sin (4 * α)) =
  -Real.sqrt 2 / 4 * Real.tan α :=
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3969_396934


namespace NUMINAMATH_CALUDE_kids_at_camp_l3969_396941

theorem kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) (h1 : total_kids = 1059955) (h2 : kids_at_home = 495718) :
  total_kids - kids_at_home = 564237 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_l3969_396941


namespace NUMINAMATH_CALUDE_composite_has_small_divisor_l3969_396920

/-- A number is composite if it's a natural number greater than 1 with a divisor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

/-- For any composite number, there exists a divisor greater than 1 but not greater than its square root. -/
theorem composite_has_small_divisor (n : ℕ) (h : IsComposite n) :
    ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d ≤ Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_composite_has_small_divisor_l3969_396920


namespace NUMINAMATH_CALUDE_cheetah_gazelle_distance_l3969_396944

/-- Proves that the initial distance between a cheetah and a gazelle is 210 feet
    given their speeds and the time it takes for the cheetah to catch up. -/
theorem cheetah_gazelle_distance (cheetah_speed : ℝ) (gazelle_speed : ℝ) 
  (mph_to_fps : ℝ) (catch_up_time : ℝ) :
  cheetah_speed = 60 →
  gazelle_speed = 40 →
  mph_to_fps = 1.5 →
  catch_up_time = 7 →
  (cheetah_speed * mph_to_fps - gazelle_speed * mph_to_fps) * catch_up_time = 210 := by
  sorry

#check cheetah_gazelle_distance

end NUMINAMATH_CALUDE_cheetah_gazelle_distance_l3969_396944


namespace NUMINAMATH_CALUDE_circle_intersection_properties_l3969_396946

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 + 8 = 0}

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Define point Q
def Q : ℝ × ℝ := (6, 4)

-- Define the center of the circle
def center : ℝ × ℝ := (3, 0)

-- Define a line passing through P with slope k
def line_through_P (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

-- Define the function for the equation of line PC
def line_PC (p : ℝ × ℝ) : ℝ := p.1 + 3 * p.2 - 3

-- State the theorem
theorem circle_intersection_properties :
  -- 1) The equation of line PC
  (∀ p, p ∈ C → line_PC p = 0) ∧
  -- 2) The range of slope k
  (∀ k, (∃ A B, A ≠ B ∧ A ∈ C ∧ B ∈ C ∧ A ∈ line_through_P k ∧ B ∈ line_through_P k) ↔ -3/4 < k ∧ k < 0) ∧
  -- 3) Non-existence of perpendicular bisector through Q
  (¬∃ k₁, ∃ A B, A ≠ B ∧ A ∈ C ∧ B ∈ C ∧
    (∃ k, A ∈ line_through_P k ∧ B ∈ line_through_P k) ∧
    Q ∈ {p | p.2 - 4 = k₁ * (p.1 - 6)} ∧
    (A.1 + B.1) / 2 = (Q.1 + center.1) / 2 ∧
    (A.2 + B.2) / 2 = (Q.2 + center.2) / 2 ∧
    k₁ * k = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_properties_l3969_396946


namespace NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l3969_396994

theorem square_sum_equals_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l3969_396994
