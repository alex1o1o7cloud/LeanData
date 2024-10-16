import Mathlib

namespace NUMINAMATH_CALUDE_revolver_game_probability_l3404_340451

/-- Represents a six-shot revolver with one bullet -/
def Revolver : Type := Unit

/-- Represents a player in the game -/
inductive Player : Type
| A : Player
| B : Player

/-- The probability of firing the bullet on a single shot -/
def singleShotProbability : ℚ := 1 / 6

/-- The probability of not firing the bullet on a single shot -/
def singleShotMissProbability : ℚ := 1 - singleShotProbability

/-- The starting player of the game -/
def startingPlayer : Player := Player.A

/-- The probability that the gun will fire while player A is holding it -/
noncomputable def probabilityAFires : ℚ := 6 / 11

/-- Theorem stating that the probability of A firing the gun is 6/11 -/
theorem revolver_game_probability :
  probabilityAFires = 6 / 11 :=
sorry

end NUMINAMATH_CALUDE_revolver_game_probability_l3404_340451


namespace NUMINAMATH_CALUDE_macaroon_weight_l3404_340441

theorem macaroon_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  num_bags = 4 →
  total_macaroons % num_bags = 0 →
  (total_macaroons - total_macaroons / num_bags) * weight_per_macaroon = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_weight_l3404_340441


namespace NUMINAMATH_CALUDE_max_divisor_of_f_l3404_340499

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), m = 36 ∧ 
  (∀ (n : ℕ), n > 0 → m ∣ f n) ∧
  (∀ (k : ℕ), k > 36 → ∃ (n : ℕ), n > 0 ∧ ¬(k ∣ f n)) :=
sorry

end NUMINAMATH_CALUDE_max_divisor_of_f_l3404_340499


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3404_340446

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle -/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius : ℝ := r / 2
  let height : ℝ := r * Real.sqrt 3 / 2
  (1 / 3) * Real.pi * base_radius^2 * height = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3404_340446


namespace NUMINAMATH_CALUDE_intersection_point_unique_intersection_point_correct_l3404_340439

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 7) / 3 = (y - 3) / 1 ∧ (y - 3) / 1 = (z + 1) / (-2)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  2 * x + y + 7 * z - 3 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (10, 4, -3)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 :=
by
  sorry

theorem intersection_point_correct :
  line intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  plane intersection_point.1 intersection_point.2.1 intersection_point.2.2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_unique_intersection_point_correct_l3404_340439


namespace NUMINAMATH_CALUDE_length_of_cd_l3404_340478

/-- Given points R and S on line segment CD, where R divides CD in the ratio 3:5,
    S divides CD in the ratio 4:7, and RS = 3, the length of CD is 264. -/
theorem length_of_cd (C D R S : Real) : 
  (∃ m n : Real, C + m = R ∧ R + n = D ∧ m / n = 3 / 5) →  -- R divides CD in ratio 3:5
  (∃ p q : Real, C + p = S ∧ S + q = D ∧ p / q = 4 / 7) →  -- S divides CD in ratio 4:7
  (S - R = 3) →                                            -- RS = 3
  (D - C = 264) :=                                         -- Length of CD is 264
by sorry

end NUMINAMATH_CALUDE_length_of_cd_l3404_340478


namespace NUMINAMATH_CALUDE_sum_of_ages_l3404_340431

/-- 
Given that Tom is 15 years old now and in 3 years he will be twice Tim's age,
prove that the sum of their current ages is 21 years.
-/
theorem sum_of_ages : 
  ∀ (tim_age : ℕ), 
  (15 + 3 = 2 * (tim_age + 3)) →
  (15 + tim_age = 21) := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3404_340431


namespace NUMINAMATH_CALUDE_cherry_tart_fraction_l3404_340480

theorem cherry_tart_fraction (total : ℝ) (blueberry : ℝ) (peach : ℝ) 
  (h1 : total = 0.91)
  (h2 : blueberry = 0.75)
  (h3 : peach = 0.08)
  (h4 : ∃ cherry : ℝ, cherry + blueberry + peach = total) :
  ∃ cherry : ℝ, cherry = 0.08 ∧ cherry + blueberry + peach = total := by
sorry

end NUMINAMATH_CALUDE_cherry_tart_fraction_l3404_340480


namespace NUMINAMATH_CALUDE_equation_solution_l3404_340447

theorem equation_solution : ∃! y : ℚ, (4 * y + 2) / (5 * y - 5) = 3 / 4 ∧ 5 * y - 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3404_340447


namespace NUMINAMATH_CALUDE_two_cookies_eaten_l3404_340466

/-- Given an initial number of cookies and the number of cookies left,
    calculate the number of cookies eaten. -/
def cookies_eaten (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Theorem: Given 7 initial cookies and 5 cookies left,
    prove that 2 cookies were eaten. -/
theorem two_cookies_eaten :
  cookies_eaten 7 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_cookies_eaten_l3404_340466


namespace NUMINAMATH_CALUDE_longest_chord_length_l3404_340454

theorem longest_chord_length (r : ℝ) (h : r = 11) : 
  2 * r = 22 := by sorry

end NUMINAMATH_CALUDE_longest_chord_length_l3404_340454


namespace NUMINAMATH_CALUDE_element_in_M_l3404_340425

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}

theorem element_in_M : 2.5 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_M_l3404_340425


namespace NUMINAMATH_CALUDE_sqrt_10_greater_than_3_l3404_340498

theorem sqrt_10_greater_than_3 : Real.sqrt 10 > 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_greater_than_3_l3404_340498


namespace NUMINAMATH_CALUDE_man_age_difference_l3404_340474

/-- Proves that a man is 37 years older than his son given the conditions. -/
theorem man_age_difference (son_age man_age : ℕ) : 
  son_age = 35 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 37 := by
  sorry

end NUMINAMATH_CALUDE_man_age_difference_l3404_340474


namespace NUMINAMATH_CALUDE_blocks_standing_final_value_l3404_340422

/-- The number of blocks left standing in the final tower -/
def blocks_standing_final (first_stack : ℕ) (second_stack_diff : ℕ) (final_stack_diff : ℕ) 
  (blocks_standing_second : ℕ) (total_fallen : ℕ) : ℕ :=
  let second_stack := first_stack + second_stack_diff
  let final_stack := second_stack + final_stack_diff
  let fallen_first := first_stack
  let fallen_second := second_stack - blocks_standing_second
  let fallen_final := total_fallen - fallen_first - fallen_second
  final_stack - fallen_final

theorem blocks_standing_final_value :
  blocks_standing_final 7 5 7 2 33 = 3 := by sorry

end NUMINAMATH_CALUDE_blocks_standing_final_value_l3404_340422


namespace NUMINAMATH_CALUDE_min_value_function_inequality_abc_l3404_340410

-- Part 1
theorem min_value_function (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := by sorry

-- Part 2
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := by sorry

end NUMINAMATH_CALUDE_min_value_function_inequality_abc_l3404_340410


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_4_6_l3404_340494

theorem gcd_lcm_sum_4_6 : Nat.gcd 4 6 + Nat.lcm 4 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_4_6_l3404_340494


namespace NUMINAMATH_CALUDE_jen_shooting_game_times_l3404_340489

theorem jen_shooting_game_times (shooting_cost carousel_cost russel_rides total_tickets : ℕ) 
  (h1 : shooting_cost = 5)
  (h2 : carousel_cost = 3)
  (h3 : russel_rides = 3)
  (h4 : total_tickets = 19) :
  ∃ (jen_times : ℕ), jen_times * shooting_cost + russel_rides * carousel_cost = total_tickets ∧ jen_times = 2 := by
  sorry

end NUMINAMATH_CALUDE_jen_shooting_game_times_l3404_340489


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3404_340467

theorem point_in_second_quadrant (A B : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2) :
  let P : ℝ × ℝ := (Real.cos B - Real.sin A, Real.sin B - Real.cos A)
  P.1 < 0 ∧ P.2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3404_340467


namespace NUMINAMATH_CALUDE_equation_solutions_l3404_340438

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 5/3 ∧
    3*(x₁-1)^2 = 2*(x₁-1) ∧ 3*(x₂-1)^2 = 2*(x₂-1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3404_340438


namespace NUMINAMATH_CALUDE_keychain_cost_decrease_l3404_340434

theorem keychain_cost_decrease (P : ℝ) : 
  P - P * 0.35 - (P - P * 0.50) = 15 ∧ P - P * 0.50 = 50 → 
  P - P * 0.35 = 65 := by
sorry

end NUMINAMATH_CALUDE_keychain_cost_decrease_l3404_340434


namespace NUMINAMATH_CALUDE_max_nondegenerate_triangles_l3404_340485

/-- Represents a triangle with colored sides -/
structure ColoredTriangle where
  blue : ℝ
  red : ℝ
  white : ℝ
  is_nondegenerate : blue + red > white ∧ blue + white > red ∧ red + white > blue

/-- The number of triangles -/
def num_triangles : ℕ := 2009

/-- A collection of 2009 non-degenerated triangles with colored sides -/
def triangle_collection : Fin num_triangles → ColoredTriangle := sorry

/-- Sorted blue sides -/
def sorted_blue : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).blue

/-- Sorted red sides -/
def sorted_red : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).red

/-- Sorted white sides -/
def sorted_white : Fin num_triangles → ℝ := 
  λ i => (triangle_collection i).white

/-- Sides are sorted in non-decreasing order -/
axiom sides_sorted : 
  (∀ i j, i ≤ j → sorted_blue i ≤ sorted_blue j) ∧
  (∀ i j, i ≤ j → sorted_red i ≤ sorted_red j) ∧
  (∀ i j, i ≤ j → sorted_white i ≤ sorted_white j)

/-- The main theorem: The maximum number of indices for which we can form non-degenerated triangles is 2009 -/
theorem max_nondegenerate_triangles : 
  (∃ f : Fin num_triangles → Fin num_triangles, 
    Function.Injective f ∧
    ∀ i, (sorted_blue (f i) + sorted_red (f i) > sorted_white (f i)) ∧
         (sorted_blue (f i) + sorted_white (f i) > sorted_red (f i)) ∧
         (sorted_red (f i) + sorted_white (f i) > sorted_blue (f i))) ∧
  (∀ k > num_triangles, ¬∃ f : Fin k → Fin num_triangles, 
    Function.Injective f ∧
    ∀ i, (sorted_blue (f i) + sorted_red (f i) > sorted_white (f i)) ∧
         (sorted_blue (f i) + sorted_white (f i) > sorted_red (f i)) ∧
         (sorted_red (f i) + sorted_white (f i) > sorted_blue (f i))) :=
by sorry

end NUMINAMATH_CALUDE_max_nondegenerate_triangles_l3404_340485


namespace NUMINAMATH_CALUDE_hexagon_angle_A_l3404_340458

/-- A hexagon is a polygon with 6 sides -/
def Hexagon (A B C D E F : ℝ) : Prop :=
  A + B + C + D + E + F = 720

/-- The theorem states that in a hexagon ABCDEF where B = 134°, C = 98°, D = 120°, E = 139°, and F = 109°, the measure of angle A is 120° -/
theorem hexagon_angle_A (A B C D E F : ℝ) 
  (h : Hexagon A B C D E F) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) : 
  A = 120 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_A_l3404_340458


namespace NUMINAMATH_CALUDE_investment_solution_l3404_340402

def investment_problem (x y r1 r2 total_investment desired_interest : ℝ) : Prop :=
  x + y = total_investment ∧
  r1 * x + r2 * y = desired_interest

theorem investment_solution :
  investment_problem 6000 4000 0.09 0.11 10000 980 := by
  sorry

end NUMINAMATH_CALUDE_investment_solution_l3404_340402


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l3404_340436

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l3404_340436


namespace NUMINAMATH_CALUDE_digit2List_998_999_1000_l3404_340462

/-- A list of increasing positive integers starting with 2 and containing all numbers with a first digit of 2 -/
def digit2List : List ℕ := sorry

/-- The function that extracts the nth digit from the digit2List -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 998th, 999th, and 1000th digits in digit2List form the number 216 -/
theorem digit2List_998_999_1000 : 
  nthDigit 998 = 2 ∧ nthDigit 999 = 1 ∧ nthDigit 1000 = 6 := by sorry

end NUMINAMATH_CALUDE_digit2List_998_999_1000_l3404_340462


namespace NUMINAMATH_CALUDE_time_difference_problem_l3404_340468

theorem time_difference_problem (speed_ratio : ℚ) (time_A : ℚ) :
  speed_ratio = 3 / 4 →
  time_A = 2 →
  ∃ (time_B : ℚ), time_A - time_B = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_time_difference_problem_l3404_340468


namespace NUMINAMATH_CALUDE_jeffs_remaining_laps_l3404_340490

/-- Given Jeff's swimming requirements and progress, calculate the remaining laps before his break. -/
theorem jeffs_remaining_laps (total_laps : ℕ) (saturday_laps : ℕ) (sunday_morning_laps : ℕ) 
  (h1 : total_laps = 98)
  (h2 : saturday_laps = 27)
  (h3 : sunday_morning_laps = 15) :
  total_laps - saturday_laps - sunday_morning_laps = 56 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_remaining_laps_l3404_340490


namespace NUMINAMATH_CALUDE_log_inequality_implies_sum_nonnegative_l3404_340493

theorem log_inequality_implies_sum_nonnegative (x y : ℝ) :
  (Real.log 3 / Real.log 2)^x + (Real.log 5 / Real.log 3)^y ≥ 
  (Real.log 2 / Real.log 3)^y + (Real.log 3 / Real.log 5)^x →
  x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_sum_nonnegative_l3404_340493


namespace NUMINAMATH_CALUDE_purple_greater_than_green_less_than_triple_l3404_340450

-- Define the probability space
def prob_space : Type := Unit

-- Define the random variables
def X : prob_space → ℝ := sorry
def Y : prob_space → ℝ := sorry

-- Define the probability measure
def P : Set prob_space → ℝ := sorry

-- State the theorem
theorem purple_greater_than_green_less_than_triple (ω : prob_space) : 
  P {ω | X ω < Y ω ∧ Y ω < min (3 * X ω) 1} = 1/3 := by sorry

end NUMINAMATH_CALUDE_purple_greater_than_green_less_than_triple_l3404_340450


namespace NUMINAMATH_CALUDE_min_value_of_x_l3404_340487

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 + (1/3) * Real.log x) : x ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l3404_340487


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3404_340449

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) :
  a^2 + b^2 + c^2 + d^2 ≥ 24/5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3404_340449


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3404_340444

/-- The average speed of a round trip given different speeds in each direction -/
theorem round_trip_average_speed 
  (speed_there : ℝ) 
  (speed_back : ℝ) 
  (h1 : speed_there = 6) 
  (h2 : speed_back = 4) : 
  (2 * speed_there * speed_back) / (speed_there + speed_back) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3404_340444


namespace NUMINAMATH_CALUDE_regression_change_l3404_340435

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 7 - 3 * x

-- Theorem statement
theorem regression_change (x₁ x₂ : ℝ) (h : x₂ = x₁ + 2) :
  regression_equation x₁ - regression_equation x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_regression_change_l3404_340435


namespace NUMINAMATH_CALUDE_shelves_needed_l3404_340486

theorem shelves_needed (initial_books : ℝ) (added_books : ℝ) (books_per_shelf : ℝ) :
  initial_books = 46.0 →
  added_books = 10.0 →
  books_per_shelf = 4.0 →
  ((initial_books + added_books) / books_per_shelf) = 14.0 := by
  sorry

end NUMINAMATH_CALUDE_shelves_needed_l3404_340486


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_subset_condition_l3404_340492

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |2*x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 2/3} :=
sorry

-- Part 2
theorem range_of_a_for_subset_condition :
  (∀ x ∈ Set.Icc (1/2) 1, f a x ≤ 2*x) → a ∈ Set.Icc (-3/2) 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_subset_condition_l3404_340492


namespace NUMINAMATH_CALUDE_robotics_club_non_participants_l3404_340428

theorem robotics_club_non_participants (total students_in_electronics students_in_programming students_in_both : ℕ) 
  (h1 : total = 80)
  (h2 : students_in_electronics = 45)
  (h3 : students_in_programming = 50)
  (h4 : students_in_both = 30) :
  total - (students_in_electronics + students_in_programming - students_in_both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_non_participants_l3404_340428


namespace NUMINAMATH_CALUDE_negative_two_classification_l3404_340420

theorem negative_two_classification :
  (∃ (n : ℤ), n = -2) →  -- -2 is an integer
  (∃ (q : ℚ), q = -2 ∧ q < 0)  -- -2 is a negative rational number
:= by sorry

end NUMINAMATH_CALUDE_negative_two_classification_l3404_340420


namespace NUMINAMATH_CALUDE_proportional_sum_l3404_340424

theorem proportional_sum (M N : ℚ) : 
  (3 / 5 : ℚ) = M / 45 ∧ (3 / 5 : ℚ) = 60 / N → M + N = 127 := by
  sorry

end NUMINAMATH_CALUDE_proportional_sum_l3404_340424


namespace NUMINAMATH_CALUDE_frogs_eat_pests_l3404_340475

/-- The number of pests a single frog eats per day -/
def pests_per_frog_per_day : ℕ := 80

/-- The number of frogs -/
def num_frogs : ℕ := 5

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: 5 frogs eat 2800 pests in a week -/
theorem frogs_eat_pests : 
  pests_per_frog_per_day * num_frogs * days_in_week = 2800 := by
  sorry

end NUMINAMATH_CALUDE_frogs_eat_pests_l3404_340475


namespace NUMINAMATH_CALUDE_fraction_equality_l3404_340460

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c^2 / d = 16) :
  d / a = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3404_340460


namespace NUMINAMATH_CALUDE_function_range_l3404_340479

def f (x : ℝ) : ℝ := -x^2 + 3*x + 1

theorem function_range :
  ∃ (a b : ℝ), a = -3 ∧ b = 13/4 ∧
  (∀ x, x ∈ Set.Icc (-1) 2 → f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc (-1) 2, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l3404_340479


namespace NUMINAMATH_CALUDE_inequality_implication_l3404_340426

theorem inequality_implication (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3404_340426


namespace NUMINAMATH_CALUDE_inverse_function_value_l3404_340412

theorem inverse_function_value (a : ℝ) (h : a > 1) :
  let f (x : ℝ) := a^(x + 1) - 2
  let f_inv := Function.invFun f
  f_inv (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_value_l3404_340412


namespace NUMINAMATH_CALUDE_distinct_primes_dividing_sequence_l3404_340433

theorem distinct_primes_dividing_sequence (n M : ℕ) (h : M > n^(n-1)) :
  ∃ (p : Fin n → ℕ), (∀ i : Fin n, Nat.Prime (p i)) ∧ 
  (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
  (∀ i : Fin n, (p i) ∣ (M + i.val + 1)) :=
sorry

end NUMINAMATH_CALUDE_distinct_primes_dividing_sequence_l3404_340433


namespace NUMINAMATH_CALUDE_dog_to_hamster_lifespan_ratio_l3404_340448

/-- The average lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a fish in years -/
def fish_lifespan : ℝ := 12

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := fish_lifespan - 2

theorem dog_to_hamster_lifespan_ratio :
  dog_lifespan / hamster_lifespan = 4 := by sorry

end NUMINAMATH_CALUDE_dog_to_hamster_lifespan_ratio_l3404_340448


namespace NUMINAMATH_CALUDE_regression_line_mean_y_l3404_340429

theorem regression_line_mean_y (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 5) (h₃ : x₃ = 7) (h₄ : x₄ = 13) (h₅ : x₅ = 19)
  (regression_eq : ℝ → ℝ) (h_reg : ∀ x, regression_eq x = 1.5 * x + 45) : 
  let x_mean := (x₁ + x₂ + x₃ + x₄ + x₅) / 5
  regression_eq x_mean = 58.5 := by
sorry

end NUMINAMATH_CALUDE_regression_line_mean_y_l3404_340429


namespace NUMINAMATH_CALUDE_monkey_climb_time_25m_l3404_340482

/-- Represents the time taken for a monkey to climb a greased pole -/
def monkey_climb_time (pole_height : ℕ) (ascend_rate : ℕ) (slip_rate : ℕ) : ℕ :=
  let full_cycles := (pole_height - ascend_rate) / (ascend_rate - slip_rate)
  full_cycles * 2 + 1

/-- Theorem stating that it takes 45 minutes for the monkey to climb the pole -/
theorem monkey_climb_time_25m :
  monkey_climb_time 25 3 2 = 45 := by sorry

end NUMINAMATH_CALUDE_monkey_climb_time_25m_l3404_340482


namespace NUMINAMATH_CALUDE_equation_solution_l3404_340432

theorem equation_solution :
  ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 8 ∧ x = 112 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3404_340432


namespace NUMINAMATH_CALUDE_circle_equation_l3404_340457

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define tangency
def IsTangent (c : Circle) (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation (c : Circle) 
  (h1 : c.radius = 2)
  (h2 : c.center.2 = 0 ∧ c.center.1 > 0)
  (h3 : IsTangent c (Line 3 4 4)) :
  ∀ (x y : ℝ), (x - c.center.1)^2 + y^2 = 4 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3404_340457


namespace NUMINAMATH_CALUDE_square_root_expression_evaluation_l3404_340455

theorem square_root_expression_evaluation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.000000001 ∧ 
  |22 + Real.sqrt (-4 + 6 * 4 * 3) - 30.246211251| < ε := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_evaluation_l3404_340455


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l3404_340430

/-- A function f is increasing on an interval (a, +∞) if for all x₁, x₂ in the interval,
    x₁ < x₂ implies f x₁ < f x₂ -/
def IncreasingOn (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The quadratic function we're considering -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1 - a)*x + 2

theorem quadratic_increasing_condition (a : ℝ) :
  IncreasingOn (f a) 4 → a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l3404_340430


namespace NUMINAMATH_CALUDE_min_value_and_max_t_l3404_340491

/-- Given a > 0, b > 0, and f(x) = |x + a| + |2x - b| with a minimum value of 1 -/
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

theorem min_value_and_max_t (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) (hmin_exists : ∃ x, f a b x = 1) :
  (2*a + b = 2) ∧ 
  (∀ t, (∀ a b, a > 0 → b > 0 → a + 2*b ≥ t*a*b) → t ≤ 9/2) ∧
  (∃ t, t = 9/2 ∧ ∀ a b, a > 0 → b > 0 → a + 2*b ≥ t*a*b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_max_t_l3404_340491


namespace NUMINAMATH_CALUDE_virginia_egg_problem_l3404_340442

/-- Virginia's egg problem -/
theorem virginia_egg_problem (initial_eggs final_eggs taken_eggs : ℕ) :
  final_eggs = 93 →
  taken_eggs = 3 →
  initial_eggs = final_eggs + taken_eggs →
  initial_eggs = 96 := by
  sorry

end NUMINAMATH_CALUDE_virginia_egg_problem_l3404_340442


namespace NUMINAMATH_CALUDE_ice_cream_scoops_prove_ice_cream_scoops_l3404_340417

-- Define the given conditions
def aaron_savings : ℚ := 40
def carson_savings : ℚ := 40
def total_savings : ℚ := aaron_savings + carson_savings
def dinner_bill_ratio : ℚ := 3 / 4
def scoop_cost : ℚ := 3 / 2
def change_per_person : ℚ := 1

-- Define the theorem
theorem ice_cream_scoops : ℚ :=
  let dinner_bill := dinner_bill_ratio * total_savings
  let remaining_after_dinner := total_savings - dinner_bill
  let ice_cream_spending := remaining_after_dinner - 2 * change_per_person
  let total_scoops := ice_cream_spending / scoop_cost
  total_scoops / 2

-- The theorem to prove
theorem prove_ice_cream_scoops : ice_cream_scoops = 6 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_prove_ice_cream_scoops_l3404_340417


namespace NUMINAMATH_CALUDE_claire_schedule_l3404_340409

/-- Claire's daily schedule problem -/
theorem claire_schedule (total_hours cleaning_hours cooking_hours crafting_hours : ℕ) 
  (h1 : total_hours = 24)
  (h2 : cleaning_hours = 4)
  (h3 : cooking_hours = 2)
  (h4 : crafting_hours = 5)
  (h5 : ∃ tailoring_hours : ℕ, tailoring_hours = crafting_hours) :
  total_hours - (cleaning_hours + cooking_hours + crafting_hours + crafting_hours) = 8 := by
  sorry

end NUMINAMATH_CALUDE_claire_schedule_l3404_340409


namespace NUMINAMATH_CALUDE_emily_age_l3404_340400

theorem emily_age :
  ∀ (e m : ℕ),
  e = m - 18 →
  e + m = 54 →
  e = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_age_l3404_340400


namespace NUMINAMATH_CALUDE_antonias_supplements_l3404_340469

theorem antonias_supplements :
  let total_pills : ℕ := 3 * 120 + 2 * 30
  let days : ℕ := 14
  let remaining_pills : ℕ := 350
  let supplements : ℕ := (total_pills - remaining_pills) / days
  supplements = 5 :=
by sorry

end NUMINAMATH_CALUDE_antonias_supplements_l3404_340469


namespace NUMINAMATH_CALUDE_problem_solution_l3404_340484

theorem problem_solution : 
  ((-1/2 - 1/3 + 3/4) * (-60) = 5) ∧ 
  ((-1)^4 - 1/6 * (3 - (-3)^2) = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3404_340484


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_500_l3404_340472

theorem largest_common_value_less_than_500 
  (ap1 : ℕ → ℕ) 
  (ap2 : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, ap1 n = 5 + 4 * n) 
  (h2 : ∀ n : ℕ, ap2 n = 7 + 8 * n) : 
  (∃ k : ℕ, k < 500 ∧ (∃ n m : ℕ, ap1 n = k ∧ ap2 m = k)) ∧
  (∀ l : ℕ, l < 500 → (∃ n m : ℕ, ap1 n = l ∧ ap2 m = l) → l ≤ 497) ∧
  (∃ n m : ℕ, ap1 n = 497 ∧ ap2 m = 497) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_less_than_500_l3404_340472


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_threshold_l3404_340465

theorem sum_of_numbers_greater_than_threshold : 
  let numbers : List ℝ := [0.8, 1/2, 0.9]
  let threshold : ℝ := 0.3
  (numbers.filter (λ x => x > threshold)).sum = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_threshold_l3404_340465


namespace NUMINAMATH_CALUDE_binomial_12_11_l3404_340414

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l3404_340414


namespace NUMINAMATH_CALUDE_max_loquat_wholesale_l3404_340461

-- Define the fruit types
inductive Fruit
| Loquat
| Cherries
| Apples

-- Define the wholesale and retail prices
def wholesale_price (f : Fruit) : ℝ :=
  match f with
  | Fruit.Loquat => 8
  | Fruit.Cherries => 36
  | Fruit.Apples => 12

def retail_price (f : Fruit) : ℝ :=
  match f with
  | Fruit.Loquat => 10
  | Fruit.Cherries => 42
  | Fruit.Apples => 16

-- Define the theorem
theorem max_loquat_wholesale (x : ℝ) :
  -- Conditions
  (wholesale_price Fruit.Cherries = wholesale_price Fruit.Loquat + 28) →
  (80 * wholesale_price Fruit.Loquat + 120 * wholesale_price Fruit.Cherries = 4960) →
  (∃ y : ℝ, x * wholesale_price Fruit.Loquat + 
            (160 - x) * wholesale_price Fruit.Apples + 
            y * wholesale_price Fruit.Cherries = 5280) →
  (x * (retail_price Fruit.Loquat - wholesale_price Fruit.Loquat) +
   (160 - x) * (retail_price Fruit.Apples - wholesale_price Fruit.Apples) +
   ((5280 - x * wholesale_price Fruit.Loquat - (160 - x) * wholesale_price Fruit.Apples) / 
    wholesale_price Fruit.Cherries) * 
   (retail_price Fruit.Cherries - wholesale_price Fruit.Cherries) ≥ 1120) →
  -- Conclusion
  x ≤ 60 :=
by sorry


end NUMINAMATH_CALUDE_max_loquat_wholesale_l3404_340461


namespace NUMINAMATH_CALUDE_youngest_child_age_l3404_340405

/-- Given a family where:
    1. 10 years ago, the average age of 4 members was 24 years
    2. Two children were born with an age difference of 2 years
    3. The present average age of the family (now 6 members) is still 24 years
    Prove that the present age of the youngest child is 3 years -/
theorem youngest_child_age
  (past_average_age : ℕ)
  (past_family_size : ℕ)
  (years_passed : ℕ)
  (present_average_age : ℕ)
  (present_family_size : ℕ)
  (age_difference : ℕ)
  (h1 : past_average_age = 24)
  (h2 : past_family_size = 4)
  (h3 : years_passed = 10)
  (h4 : present_average_age = 24)
  (h5 : present_family_size = 6)
  (h6 : age_difference = 2) :
  ∃ (youngest_age : ℕ), youngest_age = 3 ∧
    present_average_age * present_family_size =
    (past_average_age * past_family_size + years_passed * past_family_size + youngest_age + (youngest_age + age_difference)) :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3404_340405


namespace NUMINAMATH_CALUDE_fruit_shop_results_l3404_340404

/-- Represents the fruit inventory and pricing information for a shopkeeper --/
structure FruitShop where
  totalFruits : Nat
  oranges : Nat
  bananas : Nat
  apples : Nat
  rottenOrangesPercent : Rat
  rottenBananasPercent : Rat
  rottenApplesPercent : Rat
  orangePurchasePrice : Rat
  bananaPurchasePrice : Rat
  applePurchasePrice : Rat
  orangeSellingPrice : Rat
  bananaSellingPrice : Rat
  appleSellingPrice : Rat

/-- Calculates the percentage of fruits in good condition and the overall profit --/
def calculateResults (shop : FruitShop) : (Rat × Rat) :=
  sorry

/-- Theorem stating the correct percentage of good fruits and overall profit --/
theorem fruit_shop_results (shop : FruitShop) 
  (h1 : shop.totalFruits = 1000)
  (h2 : shop.oranges = 600)
  (h3 : shop.bananas = 300)
  (h4 : shop.apples = 100)
  (h5 : shop.rottenOrangesPercent = 15/100)
  (h6 : shop.rottenBananasPercent = 8/100)
  (h7 : shop.rottenApplesPercent = 20/100)
  (h8 : shop.orangePurchasePrice = 60/100)
  (h9 : shop.bananaPurchasePrice = 30/100)
  (h10 : shop.applePurchasePrice = 1)
  (h11 : shop.orangeSellingPrice = 120/100)
  (h12 : shop.bananaSellingPrice = 60/100)
  (h13 : shop.appleSellingPrice = 150/100) :
  calculateResults shop = (866/1000, 3476/10) := by
  sorry


end NUMINAMATH_CALUDE_fruit_shop_results_l3404_340404


namespace NUMINAMATH_CALUDE_exponential_function_exists_l3404_340415

theorem exponential_function_exists : ∃ (f : ℝ → ℝ), 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧ 
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂) ∧
  (∃ a : ℝ, a > 1 ∧ ∀ x : ℝ, f x = a^x) :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_exists_l3404_340415


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3404_340403

theorem max_sum_of_squares (a b c d : ℕ+) (h : a^2 + b^2 + c^2 + d^2 = 70) :
  a + b + c + d ≤ 16 ∧ ∃ (a' b' c' d' : ℕ+), a'^2 + b'^2 + c'^2 + d'^2 = 70 ∧ a' + b' + c' + d' = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3404_340403


namespace NUMINAMATH_CALUDE_gnuff_tutoring_time_l3404_340463

/-- Calculates the number of minutes tutored given the flat rate, per-minute rate, and total amount paid. -/
def minutes_tutored (flat_rate : ℚ) (per_minute_rate : ℚ) (total_paid : ℚ) : ℚ :=
  (total_paid - flat_rate) / per_minute_rate

/-- Proves that Gnuff tutored for 18 minutes given the specified rates and total amount paid. -/
theorem gnuff_tutoring_time :
  let flat_rate : ℚ := 20
  let per_minute_rate : ℚ := 7
  let total_paid : ℚ := 146
  minutes_tutored flat_rate per_minute_rate total_paid = 18 := by
  sorry

end NUMINAMATH_CALUDE_gnuff_tutoring_time_l3404_340463


namespace NUMINAMATH_CALUDE_ratio_six_three_percent_l3404_340408

/-- Expresses a ratio as a percentage -/
def ratioToPercent (a b : ℕ) : ℚ :=
  (a : ℚ) / (b : ℚ) * 100

/-- The ratio 6:3 expressed as a percent is 200% -/
theorem ratio_six_three_percent : ratioToPercent 6 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_six_three_percent_l3404_340408


namespace NUMINAMATH_CALUDE_cylinder_dihedral_angle_l3404_340483

-- Define the cylinder and its properties
structure Cylinder where
  A : Point
  A₁ : Point
  B : Point
  B₁ : Point
  C : Point
  α : Real  -- dihedral angle
  β : Real  -- ∠CAB
  γ : Real  -- ∠CA₁B

-- Define the theorem
theorem cylinder_dihedral_angle (cyl : Cylinder) :
  cyl.α = Real.arcsin (Real.cos cyl.β / Real.cos cyl.γ) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_dihedral_angle_l3404_340483


namespace NUMINAMATH_CALUDE_sum_of_six_l3404_340419

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Properties of the specific arithmetic sequence -/
def special_sequence (seq : ArithmeticSequence) : Prop :=
  seq.a 1 = 2 ∧ seq.S 4 = 20

theorem sum_of_six (seq : ArithmeticSequence) (h : special_sequence seq) : 
  seq.S 6 = 42 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_six_l3404_340419


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3404_340481

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3404_340481


namespace NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_l3404_340459

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def divisible_by_five_units (n : ℕ) : Prop := n % 10 = 0 ∨ n % 10 = 5

theorem smallest_non_five_divisible_unit : 
  (∀ d, is_digit d → (∀ n, divisible_by_five_units n → n % 10 ≠ d) → d ≥ 1) ∧
  (∃ n, divisible_by_five_units n ∧ n % 10 ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_five_divisible_unit_l3404_340459


namespace NUMINAMATH_CALUDE_distance_not_unique_l3404_340411

/-- Given two segments AB and BC with lengths 4 and 3 respectively, 
    prove that the length of AC cannot be uniquely determined. -/
theorem distance_not_unique (A B C : ℝ × ℝ) 
  (hAB : dist A B = 4) 
  (hBC : dist B C = 3) : 
  ¬ ∃! d, dist A C = d :=
sorry

end NUMINAMATH_CALUDE_distance_not_unique_l3404_340411


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3404_340488

/-- Represents a configuration of unit cubes -/
structure CubeConfiguration where
  num_cubes : ℕ
  num_outlying : ℕ
  num_exposed_faces : ℕ

/-- Calculates the volume of a cube configuration -/
def volume (config : CubeConfiguration) : ℕ := config.num_cubes

/-- Calculates the surface area of a cube configuration -/
def surface_area (config : CubeConfiguration) : ℕ := config.num_exposed_faces

/-- The specific configuration described in the problem -/
def problem_config : CubeConfiguration :=
  { num_cubes := 8
  , num_outlying := 7
  , num_exposed_faces := 33 }

/-- Theorem stating the ratio of volume to surface area for the given configuration -/
theorem volume_to_surface_area_ratio (config : CubeConfiguration) :
  config = problem_config →
  (volume config : ℚ) / (surface_area config : ℚ) = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l3404_340488


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l3404_340473

theorem no_solution_to_equation : ¬∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l3404_340473


namespace NUMINAMATH_CALUDE_bus_seats_l3404_340406

theorem bus_seats (west_lake : Nat) (east_lake : Nat)
  (h1 : west_lake = 138)
  (h2 : east_lake = 115)
  (h3 : ∀ x : Nat, x > 1 ∧ x ∣ west_lake ∧ x ∣ east_lake → x ≤ 23) :
  23 > 1 ∧ 23 ∣ west_lake ∧ 23 ∣ east_lake :=
by sorry

#check bus_seats

end NUMINAMATH_CALUDE_bus_seats_l3404_340406


namespace NUMINAMATH_CALUDE_total_amount_paid_prove_total_amount_l3404_340477

/-- Calculate the total amount paid for grapes and mangoes -/
theorem total_amount_paid 
  (grape_quantity : ℕ) (grape_price : ℕ) 
  (mango_quantity : ℕ) (mango_price : ℕ) : ℕ :=
  grape_quantity * grape_price + mango_quantity * mango_price

/-- Prove that the total amount paid for the given quantities and prices is 1135 -/
theorem prove_total_amount : 
  total_amount_paid 8 80 9 55 = 1135 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_prove_total_amount_l3404_340477


namespace NUMINAMATH_CALUDE_problem_solution_l3404_340471

theorem problem_solution : 
  (9^100 : ℕ) % 8 = 1 ∧ (2012^2012 : ℕ) % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3404_340471


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l3404_340443

/-- Proves that 6,390,000 is equal to 6.39 × 10^6 -/
theorem scientific_notation_equality : (6390000 : ℝ) = 6.39 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l3404_340443


namespace NUMINAMATH_CALUDE_line_circle_distance_range_l3404_340407

/-- The range of k for which a line y = k(x+2) has at least three points on the circle x^2 + y^2 = 4 at distance 1 from it -/
theorem line_circle_distance_range :
  ∀ k : ℝ,
  (∃ (A B C : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ (C.1^2 + C.2^2 = 4) ∧
    (|k * (A.1 + 2) - A.2| / Real.sqrt (k^2 + 1) = 1) ∧
    (|k * (B.1 + 2) - B.2| / Real.sqrt (k^2 + 1) = 1) ∧
    (|k * (C.1 + 2) - C.2| / Real.sqrt (k^2 + 1) = 1))
  ↔
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_distance_range_l3404_340407


namespace NUMINAMATH_CALUDE_max_cards_saved_is_34_l3404_340401

/-- The set of digits that remain valid when flipped upside down -/
def valid_digits : Finset Nat := {1, 6, 8, 9}

/-- The set of digits that can be used in the tens place -/
def tens_digits : Finset Nat := {0, 1, 6, 8, 9}

/-- The total number of three-digit numbers -/
def total_numbers : Nat := 900

/-- The number of valid reversible three-digit numbers -/
def reversible_numbers : Nat := valid_digits.card * tens_digits.card * valid_digits.card

/-- The number of palindromic reversible numbers -/
def palindromic_numbers : Nat := valid_digits.card * 3

/-- The maximum number of cards that can be saved -/
def max_cards_saved : Nat := (reversible_numbers - palindromic_numbers) / 2

theorem max_cards_saved_is_34 : max_cards_saved = 34 := by
  sorry

#eval max_cards_saved

end NUMINAMATH_CALUDE_max_cards_saved_is_34_l3404_340401


namespace NUMINAMATH_CALUDE_triangle_altitude_slopes_l3404_340423

/-- Given a triangle ABC with vertices A(-1,0), B(1,1), and C(0,2),
    prove that the slopes of the altitudes on sides AB, AC, and BC
    are -2, -1/2, and 1 respectively. -/
theorem triangle_altitude_slopes :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (0, 2)
  let slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  let perpendicular_slope (m : ℝ) : ℝ := -1 / m
  (perpendicular_slope (slope A B) = -2) ∧
  (perpendicular_slope (slope A C) = -1/2) ∧
  (perpendicular_slope (slope B C) = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_slopes_l3404_340423


namespace NUMINAMATH_CALUDE_point_coordinates_l3404_340470

/-- A point in the first quadrant with given distances to axes -/
structure FirstQuadrantPoint where
  m : ℝ
  n : ℝ
  first_quadrant : m > 0 ∧ n > 0
  x_axis_distance : n = 5
  y_axis_distance : m = 3

/-- Theorem: The coordinates of the point are (3,5) -/
theorem point_coordinates (P : FirstQuadrantPoint) : P.m = 3 ∧ P.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l3404_340470


namespace NUMINAMATH_CALUDE_teachers_not_picking_square_l3404_340416

theorem teachers_not_picking_square (total_teachers : ℕ) (square_teachers : ℕ) 
  (h1 : total_teachers = 20) 
  (h2 : square_teachers = 7) : 
  total_teachers - square_teachers = 13 := by
  sorry

end NUMINAMATH_CALUDE_teachers_not_picking_square_l3404_340416


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l3404_340418

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The doubling period of bacteria in seconds -/
def doubling_period : ℕ := 30

/-- The duration of the experiment in minutes -/
def experiment_duration : ℕ := 4

/-- The number of bacteria after the experiment -/
def final_bacteria_count : ℕ := 65536

/-- The number of doubling periods in the experiment -/
def doubling_periods : ℕ := (experiment_duration * seconds_per_minute) / doubling_period

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := final_bacteria_count / (2 ^ doubling_periods)

theorem initial_bacteria_count : initial_bacteria = 256 := by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l3404_340418


namespace NUMINAMATH_CALUDE_combined_shape_perimeter_l3404_340496

/-- Given a shape consisting of a rectangle and a right triangle sharing one side,
    where the rectangle has sides of length 6 and x, and the triangle has legs of length x and 6,
    the perimeter of the combined shape is 18 + 2x + √(x^2 + 36). -/
theorem combined_shape_perimeter (x : ℝ) :
  let rectangle_perimeter := 2 * (6 + x)
  let triangle_hypotenuse := Real.sqrt (x^2 + 36)
  let shared_side := x
  rectangle_perimeter + x + 6 + triangle_hypotenuse - shared_side = 18 + 2*x + Real.sqrt (x^2 + 36) := by
  sorry

end NUMINAMATH_CALUDE_combined_shape_perimeter_l3404_340496


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_ratio_l3404_340427

theorem equilateral_triangle_area_ratio :
  ∀ s : ℝ,
  s > 0 →
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let large_triangle_side := 3 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  (3 * small_triangle_area) / large_triangle_area = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_ratio_l3404_340427


namespace NUMINAMATH_CALUDE_remaining_for_coffee_l3404_340440

def initial_amount : ℝ := 60
def celery_cost : ℝ := 5
def cereal_original_price : ℝ := 12
def cereal_discount : ℝ := 0.5
def bread_cost : ℝ := 8
def milk_original_price : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6

def total_spent : ℝ :=
  celery_cost +
  (cereal_original_price * (1 - cereal_discount)) +
  bread_cost +
  (milk_original_price * (1 - milk_discount)) +
  (potato_cost * potato_quantity)

theorem remaining_for_coffee :
  initial_amount - total_spent = 26 :=
sorry

end NUMINAMATH_CALUDE_remaining_for_coffee_l3404_340440


namespace NUMINAMATH_CALUDE_triangle_base_length_l3404_340456

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 24 →
  height = 8 →
  area = (base * height) / 2 →
  base = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3404_340456


namespace NUMINAMATH_CALUDE_min_abs_z_l3404_340464

open Complex

theorem min_abs_z (z : ℂ) (h : abs (z - 1) + abs (z - (3 + 2*I)) = 2 * Real.sqrt 2) :
  ∃ (w : ℂ), abs w ≤ abs z ∧ abs w = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l3404_340464


namespace NUMINAMATH_CALUDE_ternary_121_eq_decimal_16_l3404_340421

/-- Converts a ternary (base 3) number to decimal (base 10) --/
def ternary_to_decimal (t₂ t₁ t₀ : ℕ) : ℕ :=
  t₂ * 3^2 + t₁ * 3^1 + t₀ * 3^0

/-- Proves that the ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_eq_decimal_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_eq_decimal_16_l3404_340421


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l3404_340453

theorem jacket_price_calculation (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (sales_tax : ℝ) :
  initial_price = 120 ∧
  discount1 = 0.20 ∧
  discount2 = 0.25 ∧
  sales_tax = 0.05 →
  initial_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax) = 75.60 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l3404_340453


namespace NUMINAMATH_CALUDE_function_monotonicity_l3404_340495

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2) * x + 2

theorem function_monotonicity (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  4 ≤ a ∧ a < 8 :=
by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l3404_340495


namespace NUMINAMATH_CALUDE_integral_equals_eighteen_implies_a_equals_three_l3404_340476

theorem integral_equals_eighteen_implies_a_equals_three (a : ℝ) :
  (∫ (x : ℝ) in -a..a, x^2 + Real.sin x) = 18 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_eighteen_implies_a_equals_three_l3404_340476


namespace NUMINAMATH_CALUDE_average_difference_l3404_340413

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35)
  (h2 : (b + c) / 2 = 80) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l3404_340413


namespace NUMINAMATH_CALUDE_math_competition_solution_l3404_340437

/-- Represents the number of contestants from each school -/
structure ContestantCounts where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ

/-- The conditions of the math competition -/
def ValidContestantCounts (counts : ContestantCounts) : Prop :=
  counts.A + counts.B = 16 ∧
  counts.B + counts.C = 20 ∧
  counts.C + counts.D = 34 ∧
  counts.A < counts.B ∧
  counts.B < counts.C ∧
  counts.C < counts.D

/-- The theorem to prove -/
theorem math_competition_solution :
  ∃ (counts : ContestantCounts), ValidContestantCounts counts ∧
    counts.A = 7 ∧ counts.B = 9 ∧ counts.C = 11 ∧ counts.D = 23 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_solution_l3404_340437


namespace NUMINAMATH_CALUDE_gabby_fruit_problem_l3404_340452

theorem gabby_fruit_problem (watermelons peaches plums : ℕ) : 
  peaches = watermelons + 12 →
  plums = 3 * peaches →
  watermelons + peaches + plums = 53 →
  watermelons = 1 := by
sorry

end NUMINAMATH_CALUDE_gabby_fruit_problem_l3404_340452


namespace NUMINAMATH_CALUDE_quadratic_ratio_l3404_340445

/-- Given a quadratic polynomial x^2 + 1560x + 2400, prove that when written in the form (x + b)^2 + c, the ratio c/b equals -300 -/
theorem quadratic_ratio (x : ℝ) : 
  ∃ (b c : ℝ), (∀ x, x^2 + 1560*x + 2400 = (x + b)^2 + c) ∧ c/b = -300 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l3404_340445


namespace NUMINAMATH_CALUDE_parabola_symmetric_points_a_range_l3404_340497

/-- A parabola with equation y = ax^2 - 1 where a ≠ 0 -/
structure Parabola where
  a : ℝ
  a_nonzero : a ≠ 0

/-- A point on the parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2 - 1

/-- Two points are symmetric about the line y + x = 0 -/
def symmetric_about_line (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 + p1.2 + p2.1 + p2.2 = 0

/-- The main theorem -/
theorem parabola_symmetric_points_a_range (p : Parabola) 
  (p1 p2 : ParabolaPoint p) (h_distinct : p1 ≠ p2) 
  (h_symmetric : symmetric_about_line (p1.x, p1.y) (p2.x, p2.y)) : 
  p.a > 3/4 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_points_a_range_l3404_340497
