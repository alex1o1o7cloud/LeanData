import Mathlib

namespace NUMINAMATH_CALUDE_dime_difference_l2617_261729

/-- Represents the content of a piggy bank --/
structure PiggyBank where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins in the piggy bank --/
def totalCoins (pb : PiggyBank) : ℕ :=
  pb.pennies + pb.nickels + pb.dimes

/-- Calculates the total value in cents of the coins in the piggy bank --/
def totalValue (pb : PiggyBank) : ℕ :=
  pb.pennies + 5 * pb.nickels + 10 * pb.dimes

/-- Checks if a piggy bank configuration is valid --/
def isValidPiggyBank (pb : PiggyBank) : Prop :=
  totalCoins pb = 150 ∧ totalValue pb = 500

/-- The set of all valid piggy bank configurations --/
def validPiggyBanks : Set PiggyBank :=
  {pb | isValidPiggyBank pb}

/-- The theorem to be proven --/
theorem dime_difference : 
  (⨆ (pb : PiggyBank) (h : pb ∈ validPiggyBanks), pb.dimes) -
  (⨅ (pb : PiggyBank) (h : pb ∈ validPiggyBanks), pb.dimes) = 39 := by
  sorry

end NUMINAMATH_CALUDE_dime_difference_l2617_261729


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_equality_holds_largest_n_is_six_l2617_261732

theorem largest_binomial_coefficient (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem equality_holds : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), n = 6 ∧ 
    Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧
    ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_equality_holds_largest_n_is_six_l2617_261732


namespace NUMINAMATH_CALUDE_model_height_is_correct_l2617_261706

/-- The height of the actual observatory tower in meters -/
def actual_height : ℝ := 60

/-- The volume of water the actual observatory tower can hold in liters -/
def actual_volume : ℝ := 200000

/-- The volume of water Carson's miniature model can hold in liters -/
def model_volume : ℝ := 0.2

/-- The height of Carson's miniature tower in meters -/
def model_height : ℝ := 0.6

/-- Theorem stating that the calculated model height is correct -/
theorem model_height_is_correct :
  model_height = actual_height * (model_volume / actual_volume)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_model_height_is_correct_l2617_261706


namespace NUMINAMATH_CALUDE_total_fertilizer_needed_l2617_261738

def petunia_flats : ℕ := 4
def petunias_per_flat : ℕ := 8
def rose_flats : ℕ := 3
def roses_per_flat : ℕ := 6
def venus_flytraps : ℕ := 2
def fertilizer_per_petunia : ℕ := 8
def fertilizer_per_rose : ℕ := 3
def fertilizer_per_venus_flytrap : ℕ := 2

theorem total_fertilizer_needed : 
  petunia_flats * petunias_per_flat * fertilizer_per_petunia + 
  rose_flats * roses_per_flat * fertilizer_per_rose + 
  venus_flytraps * fertilizer_per_venus_flytrap = 314 := by
  sorry

end NUMINAMATH_CALUDE_total_fertilizer_needed_l2617_261738


namespace NUMINAMATH_CALUDE_cube_painting_probability_l2617_261792

/-- The number of colors used to paint the cube -/
def num_colors : ℕ := 3

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The probability of each color for a single face -/
def color_probability : ℚ := 1 / 3

/-- The total number of possible color arrangements for the cube -/
def total_arrangements : ℕ := num_colors ^ num_faces

/-- The number of favorable arrangements where the cube can be placed with four vertical faces of the same color -/
def favorable_arrangements : ℕ := 75

/-- The probability of painting the cube such that it can be placed with four vertical faces of the same color -/
def probability_four_same : ℚ := favorable_arrangements / total_arrangements

theorem cube_painting_probability :
  probability_four_same = 25 / 243 := by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l2617_261792


namespace NUMINAMATH_CALUDE_power_product_rule_l2617_261784

theorem power_product_rule (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l2617_261784


namespace NUMINAMATH_CALUDE_smallest_positive_integer_e_l2617_261776

theorem smallest_positive_integer_e (a b c d e : ℤ) :
  (∃ (x : ℚ), a * x^4 + b * x^3 + c * x^2 + d * x + e = 0) →
  (a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0) →
  (a * 6^4 + b * 6^3 + c * 6^2 + d * 6 + e = 0) →
  (a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e = 0) →
  (a * (-2/5)^4 + b * (-2/5)^3 + c * (-2/5)^2 + d * (-2/5) + e = 0) →
  e > 0 →
  e ≥ 360 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_e_l2617_261776


namespace NUMINAMATH_CALUDE_night_games_count_l2617_261761

theorem night_games_count (total_games : ℕ) (h1 : total_games = 864) 
  (h2 : ∃ (night_games day_games : ℕ), night_games + day_games = total_games ∧ night_games = day_games) : 
  ∃ (night_games : ℕ), night_games = 432 := by
sorry

end NUMINAMATH_CALUDE_night_games_count_l2617_261761


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2617_261754

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 3) = 5 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2617_261754


namespace NUMINAMATH_CALUDE_bus_problem_l2617_261715

/-- Proof of the number of people who got off at the first bus stop -/
theorem bus_problem (total_rows : Nat) (seats_per_row : Nat) 
  (initial_boarding : Nat) (first_stop_boarding : Nat) 
  (second_stop_boarding : Nat) (second_stop_departing : Nat) 
  (empty_seats_after_second : Nat) : 
  total_rows = 23 → 
  seats_per_row = 4 → 
  initial_boarding = 16 → 
  first_stop_boarding = 15 → 
  second_stop_boarding = 17 → 
  second_stop_departing = 10 → 
  empty_seats_after_second = 57 → 
  ∃ (first_stop_departing : Nat), 
    first_stop_departing = 3 ∧
    (total_rows * seats_per_row) - 
    (initial_boarding + first_stop_boarding + second_stop_boarding - 
     first_stop_departing - second_stop_departing) = 
    empty_seats_after_second :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l2617_261715


namespace NUMINAMATH_CALUDE_kims_morning_routine_l2617_261723

/-- Kim's morning routine calculation -/
theorem kims_morning_routine (coffee_time : ℕ) (status_update_time : ℕ) (payroll_update_time : ℕ) (num_employees : ℕ) :
  coffee_time = 5 →
  status_update_time = 2 →
  payroll_update_time = 3 →
  num_employees = 9 →
  coffee_time + num_employees * (status_update_time + payroll_update_time) = 50 := by
  sorry

#check kims_morning_routine

end NUMINAMATH_CALUDE_kims_morning_routine_l2617_261723


namespace NUMINAMATH_CALUDE_count_numbers_with_at_most_two_digits_is_2151_l2617_261751

/-- The count of positive integers less than 100,000 with at most two different digits -/
def count_numbers_with_at_most_two_digits : ℕ :=
  let max_number := 100000
  let single_digit_count := 9 * 5
  let two_digits_without_zero := 36 * (2^2 - 2 + 2^3 - 2 + 2^4 - 2 + 2^5 - 2)
  let two_digits_with_zero := 9 * (2^1 - 1 + 2^2 - 1 + 2^3 - 1 + 2^4 - 1)
  single_digit_count + two_digits_without_zero + two_digits_with_zero

theorem count_numbers_with_at_most_two_digits_is_2151 :
  count_numbers_with_at_most_two_digits = 2151 :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_with_at_most_two_digits_is_2151_l2617_261751


namespace NUMINAMATH_CALUDE_position_of_2010_l2617_261744

/-- The row number for a given positive integer in the arrangement -/
def row (n : ℕ) : ℕ := 
  (n.sqrt : ℕ) + (if n > (n.sqrt : ℕ)^2 then 1 else 0)

/-- The column number for a given positive integer in the arrangement -/
def column (n : ℕ) : ℕ := 
  n - (row n - 1)^2

/-- The theorem stating that 2010 appears in row 45 and column 74 -/
theorem position_of_2010 : row 2010 = 45 ∧ column 2010 = 74 := by
  sorry

end NUMINAMATH_CALUDE_position_of_2010_l2617_261744


namespace NUMINAMATH_CALUDE_solution_count_l2617_261745

/-- The number of pairs of positive integers (x, y) that satisfy x^2 - y^2 = 45 -/
def count_solutions : Nat :=
  (Finset.filter (fun p : Nat × Nat =>
    let (x, y) := p
    x > 0 ∧ y > 0 ∧ x^2 - y^2 = 45
  ) (Finset.product (Finset.range 46) (Finset.range 46))).card

theorem solution_count : count_solutions = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l2617_261745


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2617_261711

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : geometric_sequence a) :
  (a 3) ^ 2 - 6 * (a 3) + 8 = 0 ∧
  (a 15) ^ 2 - 6 * (a 15) + 8 = 0 →
  (a 1 * a 17) / a 9 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2617_261711


namespace NUMINAMATH_CALUDE_race_distance_proof_l2617_261783

/-- The total distance of a race where:
    - A covers the distance in 45 seconds
    - B covers the distance in 60 seconds
    - A beats B by 50 meters
-/
def race_distance : ℝ := 150

theorem race_distance_proof :
  ∀ (a_time b_time : ℝ) (lead : ℝ),
  a_time = 45 ∧ 
  b_time = 60 ∧ 
  lead = 50 →
  race_distance = (lead * b_time) / (b_time / a_time - 1) :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l2617_261783


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_l2617_261780

/-- Given 9 parallel lines intersected by n parallel lines forming 1008 parallelograms, n must equal 127 -/
theorem parallel_lines_intersection (n : ℕ) : 
  (9 - 1) * (n - 1) = 1008 → n = 127 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_l2617_261780


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l2617_261733

/-- Represents a cross-country meet between two teams -/
structure CrossCountryMeet where
  /-- Total number of runners -/
  total_runners : Nat
  /-- Number of runners per team -/
  runners_per_team : Nat
  /-- Minimum possible team score -/
  min_score : Nat
  /-- Maximum possible team score -/
  max_score : Nat

/-- Calculates the number of different winning scores possible in a cross-country meet -/
def count_winning_scores (meet : CrossCountryMeet) : Nat :=
  sorry

/-- Theorem stating the number of different winning scores in the given cross-country meet -/
theorem cross_country_winning_scores :
  ∃ (meet : CrossCountryMeet),
    meet.total_runners = 10 ∧
    meet.runners_per_team = 5 ∧
    meet.min_score = 15 ∧
    meet.max_score = 40 ∧
    count_winning_scores meet = 13 :=
  sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l2617_261733


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2617_261759

/-- The center of a circle that is tangent to two parallel lines and lies on a third line -/
theorem circle_center_coordinates (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    ((x - 20)^2 + (y - 10)^2 = r^2) ∧ 
    ((x - 40/3)^2 + y^2 = r^2) ∧ 
    x^2 + y^2 = r^2) → 
  (3*x - 4*y = 20 ∧ x - 2*y = 0) → 
  x = 20 ∧ y = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2617_261759


namespace NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_two_plus_sqrt_six_simplify_sqrt_expression_l2617_261703

-- Problem 1
theorem sqrt_three_minus_sqrt_two_plus_sqrt_six : 
  Real.sqrt 3 * (Real.sqrt 3 - Real.sqrt 2) + Real.sqrt 6 = 3 := by sorry

-- Problem 2
theorem simplify_sqrt_expression (a : ℝ) (ha : a > 0) : 
  2 * Real.sqrt (12 * a) + Real.sqrt (6 * a^2) + Real.sqrt (2 * a) = 
  4 * Real.sqrt (3 * a) + Real.sqrt 6 * a + Real.sqrt (2 * a) := by sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_two_plus_sqrt_six_simplify_sqrt_expression_l2617_261703


namespace NUMINAMATH_CALUDE_correct_guess_probability_l2617_261724

/-- The number of possible digits in a phone number -/
def num_digits : ℕ := 10

/-- The length of a phone number -/
def phone_number_length : ℕ := 7

/-- The probability of correctly guessing a single unknown digit in a phone number -/
def probability_correct_guess : ℚ := 1 / num_digits

theorem correct_guess_probability : 
  probability_correct_guess = 1 / num_digits :=
by sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l2617_261724


namespace NUMINAMATH_CALUDE_square_roots_equality_l2617_261773

theorem square_roots_equality (m : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (m + 1)^2 = x ∧ (3*m - 1)^2 = x) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_equality_l2617_261773


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l2617_261764

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def num_history_books : ℕ := 4
def num_science_books : ℕ := 6

theorem book_arrangement_theorem :
  factorial 2 * factorial num_history_books * factorial num_science_books = 34560 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l2617_261764


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l2617_261749

theorem complex_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (b : ℝ), (a + 3 * Complex.I) / (1 + 2 * Complex.I) = b * Complex.I) →
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l2617_261749


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2617_261710

theorem smaller_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : 
  min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2617_261710


namespace NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_prob_l2617_261793

-- Define the probabilities of success for each person
def xavier_prob : ℚ := 1/4
def yvonne_prob : ℚ := 2/3
def zelda_prob : ℚ := 5/8

-- Define the probability of the desired outcome
def desired_outcome_prob : ℚ := xavier_prob * yvonne_prob * (1 - zelda_prob)

-- Theorem statement
theorem xavier_yvonne_not_zelda_prob : desired_outcome_prob = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_prob_l2617_261793


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l2617_261735

theorem largest_stamps_per_page (book1_stamps : ℕ) (book2_stamps : ℕ) :
  book1_stamps = 924 →
  book2_stamps = 1200 →
  ∃ (stamps_per_page : ℕ),
    stamps_per_page = Nat.gcd book1_stamps book2_stamps ∧
    stamps_per_page ≤ book1_stamps ∧
    stamps_per_page ≤ book2_stamps ∧
    ∀ (n : ℕ), n ∣ book1_stamps ∧ n ∣ book2_stamps → n ≤ stamps_per_page :=
by sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l2617_261735


namespace NUMINAMATH_CALUDE_correct_set_representations_l2617_261762

-- Define the sets
def RealNumbers : Type := Real
def NaturalNumbers : Type := Nat
def Integers : Type := Int
def RationalNumbers : Type := Rat

-- State the theorem
theorem correct_set_representations :
  (RealNumbers = ℝ) ∧
  (NaturalNumbers = ℕ) ∧
  (Integers = ℤ) ∧
  (RationalNumbers = ℚ) := by
  sorry

end NUMINAMATH_CALUDE_correct_set_representations_l2617_261762


namespace NUMINAMATH_CALUDE_rice_cost_difference_l2617_261788

/-- Represents the rice purchase and distribution scenario -/
structure RiceScenario where
  total_rice : ℝ
  price1 : ℝ
  price2 : ℝ
  price3 : ℝ
  quantity1 : ℝ
  quantity2 : ℝ
  quantity3 : ℝ
  kept_ratio : ℝ

/-- Calculates the cost difference between kept and given rice -/
def cost_difference (scenario : RiceScenario) : ℝ :=
  let total_cost := scenario.price1 * scenario.quantity1 + 
                    scenario.price2 * scenario.quantity2 + 
                    scenario.price3 * scenario.quantity3
  let kept_quantity := scenario.kept_ratio * scenario.total_rice
  let given_quantity := scenario.total_rice - kept_quantity
  let kept_cost := scenario.price1 * scenario.quantity1 + 
                   scenario.price2 * (kept_quantity - scenario.quantity1) + 
                   scenario.price3 * (kept_quantity - scenario.quantity1 - scenario.quantity2)
  let given_cost := total_cost - kept_cost
  kept_cost - given_cost

/-- The main theorem stating the cost difference for the given scenario -/
theorem rice_cost_difference : 
  let scenario : RiceScenario := {
    total_rice := 50,
    price1 := 1.2,
    price2 := 1.5,
    price3 := 2,
    quantity1 := 20,
    quantity2 := 25,
    quantity3 := 5,
    kept_ratio := 0.7
  }
  cost_difference scenario = 41.5 := by sorry


end NUMINAMATH_CALUDE_rice_cost_difference_l2617_261788


namespace NUMINAMATH_CALUDE_cosine_sum_constant_l2617_261778

theorem cosine_sum_constant (A B C : ℝ) 
  (h1 : Real.cos A + Real.cos B + Real.cos C = 0)
  (h2 : Real.sin A + Real.sin B + Real.sin C = 0) : 
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_constant_l2617_261778


namespace NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l2617_261799

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry about the xOy plane -/
def symmetricAboutXOy (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetry_about_xOy_plane :
  let p := Point3D.mk 1 3 (-5)
  let q := Point3D.mk 1 3 5
  symmetricAboutXOy p q :=
by
  sorry

#check symmetry_about_xOy_plane

end NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l2617_261799


namespace NUMINAMATH_CALUDE_value_of_N_l2617_261714

theorem value_of_N : ∃ N : ℝ, (0.25 * N = 0.55 * 3010) ∧ (N = 6622) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l2617_261714


namespace NUMINAMATH_CALUDE_jake_sister_weight_ratio_l2617_261747

theorem jake_sister_weight_ratio (J S : ℝ) (hJ : J > 0) (hS : S > 0) 
  (h1 : J + S = 132) (h2 : J - 15 = 2 * S) : (J - 15) / S = 2 := by
  sorry

end NUMINAMATH_CALUDE_jake_sister_weight_ratio_l2617_261747


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l2617_261725

def min_sugar_purchase (s : ℕ) : Prop :=
  ∃ (f : ℝ),
    f ≥ 4 + s / 3 ∧
    f ≤ 3 * s ∧
    2 * s + 3 * f ≤ 36 ∧
    ∀ (s' : ℕ), s' < s → ¬∃ (f' : ℝ),
      f' ≥ 4 + s' / 3 ∧
      f' ≤ 3 * s' ∧
      2 * s' + 3 * f' ≤ 36

theorem betty_sugar_purchase : min_sugar_purchase 4 :=
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l2617_261725


namespace NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_1000_l2617_261719

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem exists_fibonacci_divisible_by_1000 :
  ∃ n : ℕ, n ≤ 1000001 ∧ fibonacci n % 1000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_1000_l2617_261719


namespace NUMINAMATH_CALUDE_train_speed_l2617_261798

/-- The speed of a train given the time to pass a pole and a stationary train -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) :
  t_pole = 8 →
  t_stationary = 18 →
  l_stationary = 400 →
  ∃ (speed : ℝ), speed = 144 ∧ speed * 1000 / 3600 * t_pole = speed * 1000 / 3600 * t_stationary - l_stationary :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2617_261798


namespace NUMINAMATH_CALUDE_det_A_eq_33_l2617_261782

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2],
    ![1, 3,  4],
    ![0, -1, 1]]

theorem det_A_eq_33 : A.det = 33 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_33_l2617_261782


namespace NUMINAMATH_CALUDE_five_nine_difference_l2617_261720

/-- Count of a specific digit in page numbers from 1 to n -/
def digitCount (digit : Nat) (n : Nat) : Nat :=
  sorry

/-- The difference between the count of 5's and 9's in page numbers from 1 to 600 -/
theorem five_nine_difference : digitCount 5 600 - digitCount 9 600 = 100 := by
  sorry

end NUMINAMATH_CALUDE_five_nine_difference_l2617_261720


namespace NUMINAMATH_CALUDE_square_figure_division_l2617_261791

/-- Represents a rectangular figure composed of squares -/
structure SquareFigure where
  width : ℕ
  height : ℕ
  pattern : List (List Bool)

/-- Represents a cut in the figure -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut

/-- Checks if a cut follows the sides of the squares -/
def isValidCut (figure : SquareFigure) (cut : Cut) : Prop :=
  match cut with
  | Cut.Vertical n => n > 0 ∧ n < figure.width
  | Cut.Horizontal n => n > 0 ∧ n < figure.height

/-- Checks if two cuts divide the figure into four parts -/
def dividesFourParts (figure : SquareFigure) (cut1 cut2 : Cut) : Prop :=
  isValidCut figure cut1 ∧ isValidCut figure cut2 ∧
  ((∃ n m, cut1 = Cut.Vertical n ∧ cut2 = Cut.Horizontal m) ∨
   (∃ n m, cut1 = Cut.Horizontal n ∧ cut2 = Cut.Vertical m))

/-- Checks if all parts are identical after cuts -/
def partsAreIdentical (figure : SquareFigure) (cut1 cut2 : Cut) : Prop :=
  sorry  -- Definition of identical parts

/-- Main theorem: The figure can be divided into four identical parts -/
theorem square_figure_division (figure : SquareFigure) :
  ∃ cut1 cut2, dividesFourParts figure cut1 cut2 ∧ partsAreIdentical figure cut1 cut2 :=
sorry


end NUMINAMATH_CALUDE_square_figure_division_l2617_261791


namespace NUMINAMATH_CALUDE_base_conversion_1729_l2617_261768

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_1729 :
  toBase5 1729 = [2, 3, 4, 0, 4] ∧ fromBase5 [2, 3, 4, 0, 4] = 1729 :=
sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l2617_261768


namespace NUMINAMATH_CALUDE_derivative_at_one_l2617_261789

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

theorem derivative_at_one :
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2617_261789


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_15_25_40_75_l2617_261712

theorem greatest_four_digit_divisible_by_15_25_40_75 : ∃ n : ℕ,
  n ≤ 9999 ∧
  n ≥ 1000 ∧
  n % 15 = 0 ∧
  n % 25 = 0 ∧
  n % 40 = 0 ∧
  n % 75 = 0 ∧
  ∀ m : ℕ, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 15 = 0 ∧ m % 25 = 0 ∧ m % 40 = 0 ∧ m % 75 = 0 → m ≤ n :=
by
  -- Proof goes here
  sorry

#eval 9600 -- Expected output: 9600

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_15_25_40_75_l2617_261712


namespace NUMINAMATH_CALUDE_subtract_three_numbers_l2617_261702

theorem subtract_three_numbers : 15 - 3 - 15 = -3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_three_numbers_l2617_261702


namespace NUMINAMATH_CALUDE_factor_4t_squared_minus_100_l2617_261742

theorem factor_4t_squared_minus_100 (t : ℝ) : 4 * t^2 - 100 = (2*t - 10) * (2*t + 10) := by
  sorry

end NUMINAMATH_CALUDE_factor_4t_squared_minus_100_l2617_261742


namespace NUMINAMATH_CALUDE_hockey_pad_cost_calculation_l2617_261737

def hockey_pad_cost (initial_amount : ℝ) (skate_fraction : ℝ) (remaining : ℝ) : ℝ :=
  initial_amount - initial_amount * skate_fraction - remaining

theorem hockey_pad_cost_calculation :
  hockey_pad_cost 150 (1/2) 25 = 50 := by
  sorry

end NUMINAMATH_CALUDE_hockey_pad_cost_calculation_l2617_261737


namespace NUMINAMATH_CALUDE_author_average_earnings_l2617_261760

theorem author_average_earnings 
  (months_per_book : ℕ) 
  (years_writing : ℕ) 
  (total_earnings : ℕ) : 
  months_per_book = 2 → 
  years_writing = 20 → 
  total_earnings = 3600000 → 
  (total_earnings : ℚ) / ((12 / months_per_book) * years_writing) = 30000 :=
by sorry

end NUMINAMATH_CALUDE_author_average_earnings_l2617_261760


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2617_261750

open Set

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_not_sufficient :
  (∀ x, x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧
  (∃ x, (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2617_261750


namespace NUMINAMATH_CALUDE_badminton_tournament_matches_l2617_261709

/-- Represents a single elimination tournament -/
structure Tournament :=
  (total_participants : ℕ)
  (auto_progressed : ℕ)
  (first_round_players : ℕ)
  (h_participants : total_participants = auto_progressed + first_round_players)

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : ℕ := t.total_participants - 1

theorem badminton_tournament_matches :
  ∀ t : Tournament,
  t.total_participants = 120 →
  t.auto_progressed = 16 →
  t.first_round_players = 104 →
  total_matches t = 119 :=
by sorry

end NUMINAMATH_CALUDE_badminton_tournament_matches_l2617_261709


namespace NUMINAMATH_CALUDE_not_in_third_quadrant_l2617_261704

def linear_function (x : ℝ) : ℝ := -x + 2

theorem not_in_third_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_not_in_third_quadrant_l2617_261704


namespace NUMINAMATH_CALUDE_petes_marbles_l2617_261700

theorem petes_marbles (total_initial : ℕ) (blue_percent : ℚ) (trade_ratio : ℕ) (kept_red : ℕ) :
  total_initial = 10 ∧
  blue_percent = 2/5 ∧
  trade_ratio = 2 ∧
  kept_red = 1 →
  (total_initial * blue_percent).floor +
  kept_red +
  trade_ratio * ((total_initial * (1 - blue_percent)).floor - kept_red) = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_marbles_l2617_261700


namespace NUMINAMATH_CALUDE_time_to_return_home_l2617_261794

/-- The time it takes Eric to go to the park -/
def time_to_park : ℕ := 20 + 10

/-- The factor by which the return trip is longer than the trip to the park -/
def return_factor : ℕ := 3

/-- Theorem: The time it takes Eric to return home is 90 minutes -/
theorem time_to_return_home : time_to_park * return_factor = 90 := by
  sorry

end NUMINAMATH_CALUDE_time_to_return_home_l2617_261794


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l2617_261790

theorem sqrt_sum_equals_sqrt_of_sum_sqrt (a b : ℚ) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) ↔
  ({a, b} : Set ℚ) = {1/2, 3/2} :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l2617_261790


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2617_261758

theorem p_necessary_not_sufficient_for_q :
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) ∧
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2617_261758


namespace NUMINAMATH_CALUDE_inequality_proof_l2617_261748

theorem inequality_proof (a b c x y z : Real) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a^x = b*c) (eq2 : b^y = c*a) (eq3 : c^z = a*b) :
  (1 / (2 + x)) + (1 / (2 + y)) + (1 / (2 + z)) ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2617_261748


namespace NUMINAMATH_CALUDE_custom_operation_solution_l2617_261757

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the theorem
theorem custom_operation_solution :
  ∃ x : ℝ, star 3 (star 4 x) = 10 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_solution_l2617_261757


namespace NUMINAMATH_CALUDE_yard_length_ratio_l2617_261771

theorem yard_length_ratio : 
  ∀ (alex_length brianne_length derrick_length : ℝ),
  brianne_length = 6 * alex_length →
  brianne_length = 30 →
  derrick_length = 10 →
  alex_length / derrick_length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_ratio_l2617_261771


namespace NUMINAMATH_CALUDE_calculation_proof_l2617_261707

theorem calculation_proof :
  (1) * (Real.pi - 3.14) ^ 0 - |2 - Real.sqrt 3| + (-1/2)^2 = Real.sqrt 3 - 3/4 ∧
  Real.sqrt (1/3) + Real.sqrt 6 * (1/Real.sqrt 2 + Real.sqrt 8) = 16 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2617_261707


namespace NUMINAMATH_CALUDE_total_clothing_items_l2617_261770

theorem total_clothing_items (short_sleeve : ℕ) (long_sleeve : ℕ) (pants : ℕ) (jackets : ℕ) 
  (h1 : short_sleeve = 7)
  (h2 : long_sleeve = 9)
  (h3 : pants = 4)
  (h4 : jackets = 2) :
  short_sleeve + long_sleeve + pants + jackets = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_clothing_items_l2617_261770


namespace NUMINAMATH_CALUDE_harriets_siblings_product_l2617_261786

/-- Given a family where Harry has 4 sisters and 6 brothers, and Harriet is one of Harry's sisters,
    this theorem proves that the product of the number of Harriet's sisters and brothers is 24. -/
theorem harriets_siblings_product (harry_sisters : ℕ) (harry_brothers : ℕ) 
  (harriet_sisters : ℕ) (harriet_brothers : ℕ) :
  harry_sisters = 4 →
  harry_brothers = 6 →
  harriet_sisters = harry_sisters - 1 →
  harriet_brothers = harry_brothers →
  harriet_sisters * harriet_brothers = 24 :=
by sorry

end NUMINAMATH_CALUDE_harriets_siblings_product_l2617_261786


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_one_range_of_a_when_complement_A_subset_B_l2617_261716

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_one :
  (A ∩ B 1 = {x | x < -2}) ∧ (A ∪ B 1 = {x | x > 2 ∨ x ≤ 1}) := by sorry

-- Theorem for part (2)
theorem range_of_a_when_complement_A_subset_B :
  ∀ a : ℝ, (Set.univ \ A : Set ℝ) ⊆ B a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_one_range_of_a_when_complement_A_subset_B_l2617_261716


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l2617_261763

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^4 + 4*y^4) = (f (x^2))^2 + 4*y^3 * f y) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l2617_261763


namespace NUMINAMATH_CALUDE_smallest_m_is_13_l2617_261769

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

def has_nth_root_of_unity (n : ℕ) : Prop :=
  ∃ z ∈ T, z^n = 1

theorem smallest_m_is_13 :
  (∃ m : ℕ, m > 0 ∧ ∀ n ≥ m, has_nth_root_of_unity n) ∧
  (∀ m < 13, ∃ n ≥ m, ¬has_nth_root_of_unity n) ∧
  (∀ n ≥ 13, has_nth_root_of_unity n) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_13_l2617_261769


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l2617_261765

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l2617_261765


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l2617_261781

/-- The set of factors of 48 -/
def factors_of_48 : Set ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

/-- Proposition: The smallest product of two distinct factors of 48 that is not a factor of 48 is 18 -/
theorem smallest_non_factor_product :
  ∃ (x y : ℕ), x ∈ factors_of_48 ∧ y ∈ factors_of_48 ∧ x ≠ y ∧ x * y ∉ factors_of_48 ∧
  x * y = 18 ∧ ∀ (a b : ℕ), a ∈ factors_of_48 → b ∈ factors_of_48 → a ≠ b →
  a * b ∉ factors_of_48 → a * b ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l2617_261781


namespace NUMINAMATH_CALUDE_diamond_equal_forms_intersecting_lines_l2617_261795

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ⋄ y = y ⋄ x -/
def diamond_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The set of points on the lines y = x and y = -x -/
def intersecting_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 ∨ p.2 = -p.1}

theorem diamond_equal_forms_intersecting_lines :
  diamond_equal_set = intersecting_lines :=
sorry

end NUMINAMATH_CALUDE_diamond_equal_forms_intersecting_lines_l2617_261795


namespace NUMINAMATH_CALUDE_divisible_by_three_l2617_261701

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (∃ k : ℕ, n = 6*k + 1 ∨ n = 6*k + 2) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_three_l2617_261701


namespace NUMINAMATH_CALUDE_expression_evaluation_l2617_261779

theorem expression_evaluation :
  let a : ℤ := -3
  let b : ℤ := -2
  (a + b) * (b - a) + (2 * a^2 * b - a^3) / (-a) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2617_261779


namespace NUMINAMATH_CALUDE_fencing_cost_rectangle_l2617_261785

/-- Proves that for a rectangular field with sides in the ratio 3:4 and an area of 8748 sq. m, 
    the cost of fencing at 25 paise per metre is 94.5 rupees. -/
theorem fencing_cost_rectangle (length width : ℝ) (area perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 4 / 3 →
  area = 8748 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.25 →
  total_cost = perimeter * cost_per_meter →
  total_cost = 94.5 := by
sorry

end NUMINAMATH_CALUDE_fencing_cost_rectangle_l2617_261785


namespace NUMINAMATH_CALUDE_third_range_is_56_prove_third_range_l2617_261718

/-- The minimum possible range of scores -/
def min_range : ℕ := 30

/-- The first given range -/
def range1 : ℕ := 18

/-- The second given range -/
def range2 : ℕ := 26

/-- The theorem stating that the third range is 56 -/
theorem third_range_is_56 : ℕ :=
  min_range + range2

/-- The main theorem to prove -/
theorem prove_third_range :
  third_range_is_56 = 56 :=
by sorry

end NUMINAMATH_CALUDE_third_range_is_56_prove_third_range_l2617_261718


namespace NUMINAMATH_CALUDE_weight_at_170cm_l2617_261734

/-- Represents the weight of a student in kg -/
def weight : ℝ → ℝ := λ x => 0.75 * x - 68.2

/-- Theorem stating that for a height of 170 cm, the weight is 59.3 kg -/
theorem weight_at_170cm : weight 170 = 59.3 := by
  sorry

end NUMINAMATH_CALUDE_weight_at_170cm_l2617_261734


namespace NUMINAMATH_CALUDE_ferry_travel_time_l2617_261741

/-- Represents the travel time of Ferry P in hours -/
def t : ℝ := 2

/-- The speed of Ferry P in kilometers per hour -/
def speed_p : ℝ := 8

/-- The speed of Ferry Q in kilometers per hour -/
def speed_q : ℝ := speed_p + 4

/-- The distance traveled by Ferry P in kilometers -/
def distance_p : ℝ := speed_p * t

/-- The distance traveled by Ferry Q in kilometers -/
def distance_q : ℝ := 3 * distance_p

/-- The travel time of Ferry Q in hours -/
def time_q : ℝ := t + 2

theorem ferry_travel_time :
  speed_q * time_q = distance_q ∧
  t = 2 := by sorry

end NUMINAMATH_CALUDE_ferry_travel_time_l2617_261741


namespace NUMINAMATH_CALUDE_octagon_area_in_square_l2617_261767

-- Define the square's perimeter
def square_perimeter : ℝ := 72

-- Define the number of parts each side is divided into
def parts_per_side : ℕ := 3

-- Theorem statement
theorem octagon_area_in_square (square_perimeter : ℝ) (parts_per_side : ℕ) :
  square_perimeter = 72 ∧ parts_per_side = 3 →
  let side_length := square_perimeter / 4
  let segment_length := side_length / parts_per_side
  let triangle_area := 1/2 * segment_length * segment_length
  let total_removed_area := 4 * triangle_area
  let square_area := side_length * side_length
  square_area - total_removed_area = 252 :=
by sorry

end NUMINAMATH_CALUDE_octagon_area_in_square_l2617_261767


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2617_261772

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2617_261772


namespace NUMINAMATH_CALUDE_rectangle_diagonal_distance_sum_equal_l2617_261722

-- Define a rectangle in 2D space
structure Rectangle where
  a : ℝ  -- half-width
  b : ℝ  -- half-height

-- Define points A, B, C, D of the rectangle
def cornerA (r : Rectangle) : ℝ × ℝ := (-r.a, -r.b)
def cornerB (r : Rectangle) : ℝ × ℝ := (r.a, -r.b)
def cornerC (r : Rectangle) : ℝ × ℝ := (r.a, r.b)
def cornerD (r : Rectangle) : ℝ × ℝ := (-r.a, r.b)

-- Define the distance squared between two points
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the theorem
theorem rectangle_diagonal_distance_sum_equal (r : Rectangle) (p : ℝ × ℝ) :
  distanceSquared p (cornerA r) + distanceSquared p (cornerC r) =
  distanceSquared p (cornerB r) + distanceSquared p (cornerD r) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_diagonal_distance_sum_equal_l2617_261722


namespace NUMINAMATH_CALUDE_max_value_theorem_l2617_261728

theorem max_value_theorem (x : ℝ) (h : x > 0) :
  (x^2 + 3 - Real.sqrt (x^4 + 6)) / x ≤ 36 / (2 * Real.sqrt 3 + Real.sqrt (2 * Real.sqrt 6)) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2617_261728


namespace NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_two_l2617_261731

theorem sqrt_x_plus_y_equals_two (x y : ℝ) : 
  Real.sqrt (3 - x) + Real.sqrt (x - 3) + 1 = y → Real.sqrt (x + y) = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_y_equals_two_l2617_261731


namespace NUMINAMATH_CALUDE_house_painting_cost_l2617_261755

/-- The total cost of painting a house -/
def total_cost (area : ℝ) (price_per_sqft : ℝ) : ℝ :=
  area * price_per_sqft

/-- Theorem: The total cost of painting a house with an area of 484 sq ft
    at a price of Rs. 20 per sq ft is Rs. 9680 -/
theorem house_painting_cost :
  total_cost 484 20 = 9680 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_l2617_261755


namespace NUMINAMATH_CALUDE_fraction_equality_l2617_261787

theorem fraction_equality : 
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) / 
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2617_261787


namespace NUMINAMATH_CALUDE_inspector_meter_count_l2617_261730

theorem inspector_meter_count : 
  ∀ (total_meters : ℕ) (defective_meters : ℕ) (rejection_rate : ℚ),
    rejection_rate = 1/10 →
    defective_meters = 15 →
    (rejection_rate * total_meters : ℚ) = defective_meters →
    total_meters = 150 := by
  sorry

end NUMINAMATH_CALUDE_inspector_meter_count_l2617_261730


namespace NUMINAMATH_CALUDE_rectangular_garden_area_l2617_261775

/-- The area of a rectangular garden with length 2.5 meters and width 0.48 meters is 1.2 square meters. -/
theorem rectangular_garden_area : 
  let length : ℝ := 2.5
  let width : ℝ := 0.48
  length * width = 1.2 := by sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_l2617_261775


namespace NUMINAMATH_CALUDE_min_x_plus_y_l2617_261739

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 8*b - a*b = 0 ∧ a + b = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l2617_261739


namespace NUMINAMATH_CALUDE_volunteer_event_arrangements_l2617_261713

/-- The number of ways to arrange volunteers for a 5-day event -/
def volunteer_arrangements (total_days : ℕ) (consecutive_days : ℕ) (total_people : ℕ) : ℕ :=
  (total_days - consecutive_days + 1) * (Nat.factorial (total_people - 1))

/-- Theorem: The number of arrangements for the volunteer event is 24 -/
theorem volunteer_event_arrangements :
  volunteer_arrangements 5 2 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_event_arrangements_l2617_261713


namespace NUMINAMATH_CALUDE_percentage_problem_l2617_261774

theorem percentage_problem (x : ℝ) (h1 : 0.2 * x = 400) : 
  (2400 / x) * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2617_261774


namespace NUMINAMATH_CALUDE_lucas_150_mod_9_l2617_261753

def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by sorry

end NUMINAMATH_CALUDE_lucas_150_mod_9_l2617_261753


namespace NUMINAMATH_CALUDE_cosine_function_minimum_l2617_261727

theorem cosine_function_minimum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≥ a * Real.cos c) ∧ 
  (∀ ε > 0, ∃ x, a * Real.cos (b * x + (c - ε)) > a * Real.cos (c - ε)) →
  c = π :=
sorry

end NUMINAMATH_CALUDE_cosine_function_minimum_l2617_261727


namespace NUMINAMATH_CALUDE_square_starts_with_123456789_l2617_261705

theorem square_starts_with_123456789 : ∃ (n : ℕ) (k : ℕ), 
  (123456789 : ℕ) * 10^k ≤ n^2 ∧ n^2 < (123456790 : ℕ) * 10^k :=
sorry

end NUMINAMATH_CALUDE_square_starts_with_123456789_l2617_261705


namespace NUMINAMATH_CALUDE_only_proposition3_is_true_l2617_261796

-- Define the propositions
def proposition1 : Prop := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0

def proposition2 : Prop := ∀ x y : ℝ, x + y > 2 → (x > 1 ∧ y > 1)

def proposition3 : Prop := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3

def proposition4 : Prop := ∀ a b c : ℝ, a ≠ 0 →
  (b^2 - 4*a*c > 0 ↔ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)

-- The main theorem
theorem only_proposition3_is_true :
  ¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4 :=
sorry

end NUMINAMATH_CALUDE_only_proposition3_is_true_l2617_261796


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l2617_261752

theorem smallest_integer_bound (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧  -- Largest is 90
  (a + b + c + d) / 4 = 70  -- Average is 70
  → a ≥ 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l2617_261752


namespace NUMINAMATH_CALUDE_ap_num_terms_l2617_261717

/-- The number of terms in an arithmetic progression -/
def num_terms_ap (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) : ℤ :=
  (aₙ - a₁) / d + 1

/-- Theorem: In an arithmetic progression with first term 2, last term 62,
    and common difference 2, the number of terms is 31 -/
theorem ap_num_terms :
  num_terms_ap 2 62 2 = 31 := by
  sorry

#eval num_terms_ap 2 62 2

end NUMINAMATH_CALUDE_ap_num_terms_l2617_261717


namespace NUMINAMATH_CALUDE_sum_of_ages_l2617_261746

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71. -/
theorem sum_of_ages (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age = 2 * shannen_age + 5 →
  beckett_age + olaf_age + shannen_age + jack_age = 71 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_ages_l2617_261746


namespace NUMINAMATH_CALUDE_stationery_cost_l2617_261721

/-- The cost of items at a stationery store -/
theorem stationery_cost (x y z : ℝ) 
  (h1 : 4 * x + y + 10 * z = 11) 
  (h2 : 3 * x + y + 7 * z = 8.9) : 
  x + y + z = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l2617_261721


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2617_261740

theorem smallest_lcm_with_gcd_five (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  Nat.gcd a b = 5 →
  201000 ≤ Nat.lcm a b ∧ 
  ∃ (x y : ℕ), 1000 ≤ x ∧ x < 10000 ∧ 
               1000 ≤ y ∧ y < 10000 ∧ 
               Nat.gcd x y = 5 ∧ 
               Nat.lcm x y = 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_five_l2617_261740


namespace NUMINAMATH_CALUDE_extra_fruit_calculation_l2617_261708

theorem extra_fruit_calculation (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 42)
  (h2 : green_apples = 7)
  (h3 : students = 9) : 
  red_apples + green_apples - students = 40 := by
  sorry

end NUMINAMATH_CALUDE_extra_fruit_calculation_l2617_261708


namespace NUMINAMATH_CALUDE_meaningful_expression_l2617_261797

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 2)) ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2617_261797


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l2617_261756

/-- 
If the sum of the monomials x^2 * y^(m+2) and x^n * y is still a monomial, 
then m + n = 1.
-/
theorem monomial_sum_condition (m n : ℤ) : 
  (∃ (x y : ℚ), x ≠ 0 ∧ y ≠ 0 ∧ ∃ (k : ℚ), x^2 * y^(m+2) + x^n * y = k * (x^2 * y^(m+2))) → 
  m + n = 1 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l2617_261756


namespace NUMINAMATH_CALUDE_h_range_l2617_261726

-- Define the function h(x)
def h (x : ℝ) : ℝ := 2 * (x - 3)

-- Define the domain of h(x)
def dom_h : Set ℝ := {x : ℝ | x ≠ -7}

-- Define the range of h(x)
def range_h : Set ℝ := {y : ℝ | y ≠ -20}

-- Theorem statement
theorem h_range : 
  {y : ℝ | ∃ x ∈ dom_h, h x = y} = range_h :=
sorry

end NUMINAMATH_CALUDE_h_range_l2617_261726


namespace NUMINAMATH_CALUDE_student_ability_theorem_l2617_261736

-- Define the function
def f (x : ℝ) : ℝ := -0.1 * x^2 + 2.6 * x + 43

-- Define the domain
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 30 }

theorem student_ability_theorem :
  (∀ x ∈ domain, x ≤ 13 → ∀ y ∈ domain, x ≤ y → f x ≤ f y) ∧
  (∀ x ∈ domain, x ≥ 13 → ∀ y ∈ domain, x ≤ y → f x ≥ f y) ∧
  f 10 = 59 ∧
  (∀ x ∈ domain, f x ≤ f 13) := by
  sorry

end NUMINAMATH_CALUDE_student_ability_theorem_l2617_261736


namespace NUMINAMATH_CALUDE_work_completion_men_count_l2617_261777

/-- Proves that the number of men in the second group is 15, given the conditions of the problem -/
theorem work_completion_men_count : 
  ∀ (work : ℕ) (men1 men2 days1 days2 : ℕ),
    men1 = 18 →
    days1 = 20 →
    days2 = 24 →
    work = men1 * days1 →
    work = men2 * days2 →
    men2 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l2617_261777


namespace NUMINAMATH_CALUDE_large_number_arithmetic_l2617_261743

theorem large_number_arithmetic : 
  999999999999 - 888888888888 + 111111111111 = 222222222222 := by
  sorry

end NUMINAMATH_CALUDE_large_number_arithmetic_l2617_261743


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2617_261766

/-- The diagonal of a rectangle with length 6 and width 8 is 10. -/
theorem rectangle_diagonal : ∀ (l w d : ℝ), 
  l = 6 → w = 8 → d^2 = l^2 + w^2 → d = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2617_261766
