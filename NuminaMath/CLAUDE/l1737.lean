import Mathlib

namespace NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l1737_173702

/-- A function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_implies_a_minus_b (a b : ℝ) :
  (f a b (-1) = 0) →  -- f(x) has value 0 at x = -1
  (f_deriv a b (-1) = 0) →  -- f'(x) = 0 at x = -1 (condition for extreme value)
  (a - b = -7) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_minus_b_l1737_173702


namespace NUMINAMATH_CALUDE_kitchen_size_is_400_l1737_173773

/-- Represents a modular home configuration --/
structure ModularHome where
  totalSize : ℕ
  kitchenCost : ℕ
  bathroomSize : ℕ
  bathroomCost : ℕ
  otherCost : ℕ
  numKitchens : ℕ
  numBathrooms : ℕ
  totalCost : ℕ

/-- Calculates the size of the kitchen module --/
def kitchenSize (home : ModularHome) : ℕ :=
  home.totalSize - home.numBathrooms * home.bathroomSize -
    (home.totalCost - home.kitchenCost * home.numKitchens -
     home.bathroomCost * home.numBathrooms) / home.otherCost

/-- Theorem stating that the kitchen size is 400 square feet --/
theorem kitchen_size_is_400 (home : ModularHome)
    (h1 : home.totalSize = 2000)
    (h2 : home.kitchenCost = 20000)
    (h3 : home.bathroomSize = 150)
    (h4 : home.bathroomCost = 12000)
    (h5 : home.otherCost = 100)
    (h6 : home.numKitchens = 1)
    (h7 : home.numBathrooms = 2)
    (h8 : home.totalCost = 174000) :
    kitchenSize home = 400 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_size_is_400_l1737_173773


namespace NUMINAMATH_CALUDE_phi_value_l1737_173775

theorem phi_value (φ : Real) (h1 : 0 < φ ∧ φ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos φ - Real.sin φ) : 
  φ = 30 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l1737_173775


namespace NUMINAMATH_CALUDE_loan_amount_proof_l1737_173743

/-- Calculates the total loan amount given the loan terms -/
def total_loan_amount (down_payment : ℕ) (monthly_payment : ℕ) (years : ℕ) : ℕ :=
  down_payment + monthly_payment * years * 12

/-- Proves that the total loan amount is correct given the specified conditions -/
theorem loan_amount_proof (down_payment monthly_payment years : ℕ) 
  (h1 : down_payment = 10000)
  (h2 : monthly_payment = 600)
  (h3 : years = 5) :
  total_loan_amount down_payment monthly_payment years = 46000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_proof_l1737_173743


namespace NUMINAMATH_CALUDE_douglas_vote_county_y_l1737_173728

theorem douglas_vote_county_y (total_vote_percent : ℝ) (county_x_percent : ℝ) (ratio_x_to_y : ℝ) :
  total_vote_percent = 60 ∧ 
  county_x_percent = 72 ∧ 
  ratio_x_to_y = 2 →
  let county_y_percent := (3 * total_vote_percent - 2 * county_x_percent) / 1
  county_y_percent = 36 := by
sorry

end NUMINAMATH_CALUDE_douglas_vote_county_y_l1737_173728


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l1737_173792

theorem gcd_n_cube_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l1737_173792


namespace NUMINAMATH_CALUDE_solve_abc_values_l1737_173763

theorem solve_abc_values (A B : Set ℝ) (a b c : ℝ) :
  A = {x : ℝ | x^2 - a*x - 2 = 0} →
  B = {x : ℝ | x^3 + b*x + c = 0} →
  -2 ∈ A ∩ B →
  A ∩ B = A →
  a = -1 ∧ b = -3 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_abc_values_l1737_173763


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l1737_173759

theorem hyperbola_m_range (m : ℝ) :
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m ≠ 0 ∧ m + 1 ≠ 0) ∧
   ((2 + m > 0 ∧ m + 1 < 0) ∨ (2 + m < 0 ∧ m + 1 > 0))) →
  m < -2 ∨ m > -1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l1737_173759


namespace NUMINAMATH_CALUDE_reflected_ray_deviation_l1737_173790

/-- The angle of deviation for a light ray reflected from a rotated mirror -/
theorem reflected_ray_deviation (α β : Real) :
  let deviation_angle := 2 * Real.arcsin (Real.sin α * Real.sin β)
  (0 ≤ α) → (α ≤ π / 2) → (0 ≤ β) → (β ≤ π / 2) →
  deviation_angle = 2 * Real.arcsin (Real.sin α * Real.sin β) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_deviation_l1737_173790


namespace NUMINAMATH_CALUDE_exam_comparison_l1737_173781

theorem exam_comparison (total_items : ℕ) (liza_percentage : ℚ) (rose_incorrect : ℕ) : 
  total_items = 60 →
  liza_percentage = 90 / 100 →
  rose_incorrect = 4 →
  (rose_incorrect : ℚ) < total_items →
  ∃ (liza_correct rose_correct : ℕ),
    (liza_correct : ℚ) = liza_percentage * total_items ∧
    rose_correct = total_items - rose_incorrect ∧
    rose_correct - liza_correct = 2 := by
sorry

end NUMINAMATH_CALUDE_exam_comparison_l1737_173781


namespace NUMINAMATH_CALUDE_complement_of_intersection_l1737_173789

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {3, 4, 5}

-- Theorem statement
theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 2, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l1737_173789


namespace NUMINAMATH_CALUDE_largest_whole_number_inequality_l1737_173777

theorem largest_whole_number_inequality (x : ℕ) : x ≤ 3 ↔ (1 / 4 : ℚ) + (x : ℚ) / 5 < 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_inequality_l1737_173777


namespace NUMINAMATH_CALUDE_triangle_max_area_l1737_173779

/-- Given a triangle ABC with area S and sides a, b, c, 
    if 4S = a² - (b - c)² and b + c = 4, 
    then the maximum value of S is 2 -/
theorem triangle_max_area (a b c S : ℝ) : 
  4 * S = a^2 - (b - c)^2 → b + c = 4 → S ≤ 2 ∧ ∃ b c, b + c = 4 ∧ S = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1737_173779


namespace NUMINAMATH_CALUDE_part1_calculation_part2_equation_solution_l1737_173757

-- Part 1
theorem part1_calculation : Real.sqrt 2 * (Real.sqrt 2 + 1) + |Real.sqrt 2 - Real.sqrt 3| = 2 + Real.sqrt 3 := by
  sorry

-- Part 2
theorem part2_equation_solution : 
  {x : ℝ | 4 * x^2 = 25} = {5/2, -5/2} := by
  sorry

end NUMINAMATH_CALUDE_part1_calculation_part2_equation_solution_l1737_173757


namespace NUMINAMATH_CALUDE_root_existence_iff_a_ge_three_l1737_173704

/-- The function f(x) = ln x + x + 2/x - a has a root for some x > 0 if and only if a ≥ 3 -/
theorem root_existence_iff_a_ge_three (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Real.log x + x + 2 / x - a = 0) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_iff_a_ge_three_l1737_173704


namespace NUMINAMATH_CALUDE_problem_solution_l1737_173760

theorem problem_solution (a b : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b - Real.sqrt (1 - x^2)| ≤ (Real.sqrt 2 - 1) / 2) : 
  a = -1 ∧ b = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1737_173760


namespace NUMINAMATH_CALUDE_line_not_parallel_when_planes_not_perpendicular_l1737_173755

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_not_parallel_when_planes_not_perpendicular
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : ¬ plane_perpendicular α β) :
  ¬ parallel l m :=
sorry

end NUMINAMATH_CALUDE_line_not_parallel_when_planes_not_perpendicular_l1737_173755


namespace NUMINAMATH_CALUDE_quadratic_real_solution_l1737_173794

theorem quadratic_real_solution (m : ℝ) : 
  (∃ z : ℝ, z^2 + Complex.I * z + m = 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solution_l1737_173794


namespace NUMINAMATH_CALUDE_probability_mean_greater_than_median_l1737_173745

/-- A fair six-sided die --/
def Die : Type := Fin 6

/-- The result of rolling three dice --/
structure ThreeDiceRoll :=
  (d1 d2 d3 : Die)

/-- The sample space of all possible outcomes when rolling three dice --/
def sampleSpace : Finset ThreeDiceRoll := sorry

/-- The mean of a three dice roll --/
def mean (roll : ThreeDiceRoll) : ℚ := sorry

/-- The median of a three dice roll --/
def median (roll : ThreeDiceRoll) : ℚ := sorry

/-- The event where the mean is greater than the median --/
def meanGreaterThanMedian : Finset ThreeDiceRoll := sorry

theorem probability_mean_greater_than_median :
  (meanGreaterThanMedian.card : ℚ) / sampleSpace.card = 29 / 72 := by sorry

end NUMINAMATH_CALUDE_probability_mean_greater_than_median_l1737_173745


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1737_173799

theorem square_sum_theorem (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = ((a*b)^2 + (a*c)^2 + (b*c)^2) / (a*b*c) := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1737_173799


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1737_173796

theorem simplify_trig_expression :
  1 / Real.sqrt (1 + Real.tan (160 * π / 180) ^ 2) = -Real.cos (160 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1737_173796


namespace NUMINAMATH_CALUDE_three_percent_difference_l1737_173786

theorem three_percent_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.10 * y) : 
  x - y = -10 := by
sorry

end NUMINAMATH_CALUDE_three_percent_difference_l1737_173786


namespace NUMINAMATH_CALUDE_badminton_players_count_l1737_173701

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  badminton_players : ℕ
  tennis_players : ℕ
  neither_players : ℕ
  both_players : ℕ

/-- Theorem stating the number of badminton players in the specific sports club scenario -/
theorem badminton_players_count (club : SportsClub) 
  (h1 : club.total_members = 30)
  (h2 : club.badminton_players = club.tennis_players)
  (h3 : club.neither_players = 2)
  (h4 : club.both_players = 6)
  (h5 : club.total_members = 
        club.badminton_players + club.tennis_players - club.both_players + club.neither_players) :
  club.badminton_players = 17 := by
  sorry


end NUMINAMATH_CALUDE_badminton_players_count_l1737_173701


namespace NUMINAMATH_CALUDE_point_on_line_l1737_173708

/-- Given a line defined by x = (y / 2) - (2 / 5), if (m, n) and (m + p, n + 4) both lie on this line, then p = 2 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 2 - 2 / 5) →
  (m + p = (n + 4) / 2 - 2 / 5) →
  p = 2 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1737_173708


namespace NUMINAMATH_CALUDE_wendy_running_distance_l1737_173776

/-- The distance Wendy walked in miles -/
def distance_walked : ℝ := 9.17

/-- The additional distance Wendy ran compared to what she walked in miles -/
def additional_distance_ran : ℝ := 10.67

/-- The total distance Wendy ran in miles -/
def distance_ran : ℝ := distance_walked + additional_distance_ran

theorem wendy_running_distance : distance_ran = 19.84 := by
  sorry

end NUMINAMATH_CALUDE_wendy_running_distance_l1737_173776


namespace NUMINAMATH_CALUDE_circle_area_tripled_l1737_173739

theorem circle_area_tripled (r n : ℝ) (h : r > 0) (h_n : n > 0) :
  π * (r + n)^2 = 3 * π * r^2 → r = n * (1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l1737_173739


namespace NUMINAMATH_CALUDE_maria_cookies_left_l1737_173764

/-- Calculates the number of cookies Maria has left after distributing them -/
def cookiesLeft (initialCookies : ℕ) : ℕ :=
  let afterFriend := initialCookies - (initialCookies * 20 / 100)
  let afterFamily := afterFriend - (afterFriend / 3)
  let afterEating := afterFamily - 4
  let toNeighbor := afterEating / 6
  afterEating - toNeighbor

/-- Theorem stating that Maria will have 24 cookies left -/
theorem maria_cookies_left : cookiesLeft 60 = 24 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_left_l1737_173764


namespace NUMINAMATH_CALUDE_difference_of_place_values_l1737_173774

def place_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

def sum_place_values_27242 : ℕ := place_value 2 0 + place_value 2 2

def sum_place_values_7232062 : ℕ := place_value 2 1 + place_value 2 6

theorem difference_of_place_values : 
  sum_place_values_7232062 - sum_place_values_27242 = 1999818 := by sorry

end NUMINAMATH_CALUDE_difference_of_place_values_l1737_173774


namespace NUMINAMATH_CALUDE_find_f_2_l1737_173707

theorem find_f_2 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 5 * x^2 - 4 * x + 1) : 
  f 2 = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_find_f_2_l1737_173707


namespace NUMINAMATH_CALUDE_max_value_fraction_l1737_173793

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (x + y) / x ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1737_173793


namespace NUMINAMATH_CALUDE_hash_four_two_l1737_173767

-- Define the # operation
def hash (a b : ℝ) : ℝ := (a^2 + b^2) * (a - b)

-- Theorem statement
theorem hash_four_two : hash 4 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_two_l1737_173767


namespace NUMINAMATH_CALUDE_five_fridays_in_july_l1737_173754

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- June of year N -/
def june : Month := {
  days := 30,
  firstDay := DayOfWeek.Tuesday  -- Assuming the first Tuesday is on the 2nd
}

/-- July of year N -/
def july : Month := {
  days := 31,
  firstDay := DayOfWeek.Wednesday  -- Based on June's last day being Tuesday
}

/-- Count occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Main theorem -/
theorem five_fridays_in_july (h : countDayOccurrences june DayOfWeek.Tuesday = 5) :
  countDayOccurrences july DayOfWeek.Friday = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_fridays_in_july_l1737_173754


namespace NUMINAMATH_CALUDE_f_value_at_log_half_24_l1737_173726

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_log_half_24 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x < 1 → f x = 2^x - 1) :
  f (Real.log 24 / Real.log (1/2)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_at_log_half_24_l1737_173726


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l1737_173746

/-- Represents the money redistribution problem with Ben, Tom, and Max -/
theorem money_redistribution_theorem 
  (ben_start : ℕ) 
  (max_start_end : ℕ) 
  (ben_end : ℕ) 
  (tom_end : ℕ) 
  (max_end : ℕ) 
  (h1 : ben_start = 48)
  (h2 : max_start_end = 48)
  (h3 : max_end = max_start_end)
  (h4 : ben_end = ben_start)
  : ben_end + tom_end + max_end = 144 := by
  sorry

#check money_redistribution_theorem

end NUMINAMATH_CALUDE_money_redistribution_theorem_l1737_173746


namespace NUMINAMATH_CALUDE_product_after_digit_reversal_mistake_l1737_173751

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, 1 < m → m < p → ¬(p % m = 0)

theorem product_after_digit_reversal_mistake (a b : ℕ) :
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  isPrime b →           -- b is prime
  reverseDigits a * b = 280 →  -- product after mistake is 280
  a * b = 28 :=         -- correct product is 28
by sorry

end NUMINAMATH_CALUDE_product_after_digit_reversal_mistake_l1737_173751


namespace NUMINAMATH_CALUDE_zeros_after_one_in_factorial_power_is_2400_l1737_173788

/-- The number of zeros following the digit '1' in the decimal expansion of (100!)^100 -/
def zeros_after_one_in_factorial_power : ℕ :=
  let factors_of_five : ℕ := (100 / 5) + (100 / 25)
  let zeros_in_factorial : ℕ := factors_of_five
  zeros_in_factorial * 100

/-- Theorem stating that the number of zeros after '1' in (100!)^100 is 2400 -/
theorem zeros_after_one_in_factorial_power_is_2400 :
  zeros_after_one_in_factorial_power = 2400 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_factorial_power_is_2400_l1737_173788


namespace NUMINAMATH_CALUDE_no_perfect_squares_with_conditions_l1737_173713

theorem no_perfect_squares_with_conditions : 
  ¬∃ (n : ℕ), 
    n^2 < 20000 ∧ 
    4 ∣ n^2 ∧ 
    ∃ (k : ℕ), n^2 = (k + 1)^2 - k^2 :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_with_conditions_l1737_173713


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1737_173723

theorem quadratic_complete_square (c : ℝ) (h1 : c > 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + c*x + 20 = (x + n)^2 + 12) →
  c = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1737_173723


namespace NUMINAMATH_CALUDE_line_param_solution_l1737_173705

/-- Represents a 2D vector -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Represents the parameterization of a line -/
def lineParam (s h : ℝ) (t : ℝ) : Vec2 :=
  { x := s + 5 * t
    y := -2 + h * t }

/-- The equation of the line y = 3x - 11 -/
def lineEq (v : Vec2) : Prop :=
  v.y = 3 * v.x - 11

theorem line_param_solution :
  ∃ (s h : ℝ), ∀ (t : ℝ), lineEq (lineParam s h t) ∧ s = 3 ∧ h = 15 := by
  sorry

end NUMINAMATH_CALUDE_line_param_solution_l1737_173705


namespace NUMINAMATH_CALUDE_triangle_problem_l1737_173768

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A) →
  (Real.cos A = 1 / 3) →
  (B = π / 6) ∧
  (Real.sin C = (2 * Real.sqrt 6 + 1) / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1737_173768


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l1737_173748

def M (a : ℤ) : Set ℤ := {a, 0}

def N : Set ℤ := {x : ℤ | 2 * x^2 - 5 * x < 0}

theorem intersection_nonempty_implies_a_value (a : ℤ) :
  (M a ∩ N).Nonempty → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l1737_173748


namespace NUMINAMATH_CALUDE_total_earnings_l1737_173710

def wednesday_amount : ℚ := 1832
def sunday_amount : ℚ := 3162.5

theorem total_earnings : wednesday_amount + sunday_amount = 4994.5 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l1737_173710


namespace NUMINAMATH_CALUDE_matthew_baking_time_l1737_173758

/-- The time it takes Matthew to make caramel-apple coffee cakes when his oven malfunctions -/
def baking_time (assembly_time bake_time_normal decorate_time : ℝ) : ℝ :=
  assembly_time + 2 * bake_time_normal + decorate_time

/-- Theorem stating that Matthew's total baking time is 5 hours when his oven malfunctions -/
theorem matthew_baking_time :
  baking_time 1 1.5 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_matthew_baking_time_l1737_173758


namespace NUMINAMATH_CALUDE_cone_base_radius_l1737_173722

/-- Given a cone with slant height 6 cm and central angle of unfolded lateral surface 120°,
    prove that the radius of its base is 2 cm. -/
theorem cone_base_radius (slant_height : ℝ) (central_angle : ℝ) :
  slant_height = 6 →
  central_angle = 120 * π / 180 →
  2 * π * slant_height * (central_angle / (2 * π)) = 2 * π * 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1737_173722


namespace NUMINAMATH_CALUDE_card_row_theorem_l1737_173729

/-- Represents a row of nine cards --/
def CardRow := Fin 9 → ℕ

/-- Checks if three consecutive cards are in increasing order --/
def increasing_three (row : CardRow) (i : Fin 7) : Prop :=
  row i < row (i + 1) ∧ row (i + 1) < row (i + 2)

/-- Checks if three consecutive cards are in decreasing order --/
def decreasing_three (row : CardRow) (i : Fin 7) : Prop :=
  row i > row (i + 1) ∧ row (i + 1) > row (i + 2)

/-- The main theorem --/
theorem card_row_theorem (row : CardRow) : 
  (∀ i : Fin 9, row i ∈ Finset.range 10) →  -- Cards are numbered 1 to 9
  (∀ i j : Fin 9, i ≠ j → row i ≠ row j) →  -- All numbers are different
  (∀ i : Fin 7, ¬increasing_three row i) →  -- No three consecutive increasing
  (∀ i : Fin 7, ¬decreasing_three row i) →  -- No three consecutive decreasing
  row 0 = 1 →                               -- Given visible cards
  row 1 = 6 →
  row 2 = 3 →
  row 3 = 4 →
  row 6 = 8 →
  row 7 = 7 →
  row 4 = 5 ∧ row 5 = 2 ∧ row 8 = 9         -- Conclusion: A = 5, B = 2, C = 9
:= by sorry


end NUMINAMATH_CALUDE_card_row_theorem_l1737_173729


namespace NUMINAMATH_CALUDE_function_expression_l1737_173770

theorem function_expression (f : ℝ → ℝ) (h : ∀ x, f (x + 2) = 2 * x + 3) :
  ∀ x, f x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_expression_l1737_173770


namespace NUMINAMATH_CALUDE_smallest_m_correct_l1737_173749

/-- The smallest positive value of m for which the equation 10x^2 - mx + 600 = 0 has consecutive integer solutions -/
def smallest_m : ℕ := 170

/-- Predicate to check if two integers are consecutive -/
def consecutive (a b : ℤ) : Prop := b = a + 1 ∨ a = b + 1

theorem smallest_m_correct :
  ∀ m : ℕ,
  (∃ x y : ℤ, consecutive x y ∧ 10 * x^2 - m * x + 600 = 0 ∧ 10 * y^2 - m * y + 600 = 0) →
  m ≥ smallest_m :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_correct_l1737_173749


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l1737_173709

/-- Represents a block in the wall --/
structure Block where
  height : ℕ
  length : ℕ

/-- Represents a row in the wall --/
structure Row where
  blocks : List Block
  length : ℕ

/-- Represents the entire wall --/
structure Wall where
  rows : List Row
  height : ℕ
  length : ℕ

/-- Checks if the vertical joins are properly staggered --/
def isProperlyStaggered (wall : Wall) : Prop := sorry

/-- Checks if the wall is even on both ends --/
def isEvenOnEnds (wall : Wall) : Prop := sorry

/-- Counts the total number of blocks in the wall --/
def countBlocks (wall : Wall) : ℕ := sorry

/-- Theorem stating the minimum number of blocks required --/
theorem min_blocks_for_wall :
  ∀ (wall : Wall),
    wall.length = 120 ∧ 
    wall.height = 10 ∧ 
    (∀ b : Block, b ∈ (wall.rows.bind Row.blocks) → b.height = 1 ∧ (b.length = 2 ∨ b.length = 3)) ∧
    isProperlyStaggered wall ∧
    isEvenOnEnds wall →
    countBlocks wall ≥ 466 := by sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l1737_173709


namespace NUMINAMATH_CALUDE_radius_of_circle_B_l1737_173765

/-- Given two circles A and B, prove that the radius of B is 10 cm -/
theorem radius_of_circle_B (diameter_A radius_A radius_B : ℝ) : 
  diameter_A = 80 → radius_A = diameter_A / 2 → radius_A = 4 * radius_B → radius_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_B_l1737_173765


namespace NUMINAMATH_CALUDE_min_power_of_two_greater_than_10_factorial_l1737_173717

/-- Given logarithm values -/
def log10_2 : ℝ := 0.301
def log10_3 : ℝ := 0.477
def log10_7 : ℝ := 0.845

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

/-- Theorem: The minimum integer n such that 10! < 2ⁿ is 22 -/
theorem min_power_of_two_greater_than_10_factorial :
  ∃ (n : ℕ), n = 22 ∧ (factorial 10 < 2^n) ∧ ∀ m < n, ¬(factorial 10 < 2^m) :=
sorry

end NUMINAMATH_CALUDE_min_power_of_two_greater_than_10_factorial_l1737_173717


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l1737_173716

theorem quadratic_inequality_and_constraint (a b : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (a = 1 ∧ b = 2) ∧
  (∀ x y k, x > 0 → y > 0 → a / x + b / y = 1 → 
    (2 * x + y ≥ k^2 + k + 2) → 
    -3 ≤ k ∧ k ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l1737_173716


namespace NUMINAMATH_CALUDE_douglas_county_z_votes_l1737_173721

theorem douglas_county_z_votes
  (total_percent : ℝ)
  (county_x_percent : ℝ)
  (county_y_percent : ℝ)
  (county_x_voters : ℝ)
  (county_y_voters : ℝ)
  (county_z_voters : ℝ)
  (h1 : total_percent = 63)
  (h2 : county_x_percent = 74)
  (h3 : county_y_percent = 67)
  (h4 : county_x_voters = 3 * county_z_voters)
  (h5 : county_y_voters = 2 * county_z_voters)
  : ∃ county_z_percent : ℝ,
    county_z_percent = 22 ∧
    total_percent / 100 * (county_x_voters + county_y_voters + county_z_voters) =
    county_x_percent / 100 * county_x_voters +
    county_y_percent / 100 * county_y_voters +
    county_z_percent / 100 * county_z_voters :=
by sorry

end NUMINAMATH_CALUDE_douglas_county_z_votes_l1737_173721


namespace NUMINAMATH_CALUDE_sum_35_25_base6_l1737_173756

/-- Represents a number in base 6 --/
def Base6 := Nat

/-- Converts a base 6 number to a natural number --/
def to_nat (b : Base6) : Nat := sorry

/-- Converts a natural number to a base 6 number --/
def from_nat (n : Nat) : Base6 := sorry

/-- Adds two base 6 numbers --/
def add_base6 (a b : Base6) : Base6 := from_nat (to_nat a + to_nat b)

theorem sum_35_25_base6 :
  add_base6 (from_nat 35) (from_nat 25) = from_nat 104 := by sorry

end NUMINAMATH_CALUDE_sum_35_25_base6_l1737_173756


namespace NUMINAMATH_CALUDE_number_calculation_l1737_173737

theorem number_calculation (x : Float) (h : x = 0.08999999999999998) :
  let number := x * 0.1
  number = 0.008999999999999999 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l1737_173737


namespace NUMINAMATH_CALUDE_star_specific_value_l1737_173741

/-- Custom binary operation star -/
def star (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b + 2 * b^2

/-- Theorem: Given the custom operation star and specific values for a and b,
    prove that the result equals 113 -/
theorem star_specific_value : star 3 5 = 113 := by
  sorry

end NUMINAMATH_CALUDE_star_specific_value_l1737_173741


namespace NUMINAMATH_CALUDE_production_equation_l1737_173711

/-- Represents the equation for a production scenario where increasing the daily rate by 20%
    completes the task 4 days earlier. -/
theorem production_equation (x : ℝ) (h : x > 0) :
  (3000 : ℝ) / x = 4 + (3000 : ℝ) / (x * (1 + 20 / 100)) :=
sorry

end NUMINAMATH_CALUDE_production_equation_l1737_173711


namespace NUMINAMATH_CALUDE_balloon_altitude_l1737_173762

/-- Calculates the altitude of a balloon given temperature conditions -/
theorem balloon_altitude 
  (temp_decrease_rate : ℝ) -- Temperature decrease rate per 1000 meters
  (ground_temp : ℝ)        -- Ground temperature in °C
  (balloon_temp : ℝ)       -- Balloon temperature in °C
  (h : temp_decrease_rate = 6)
  (i : ground_temp = 5)
  (j : balloon_temp = -2) :
  (ground_temp - balloon_temp) / temp_decrease_rate = 7 / 6 := by
sorry

end NUMINAMATH_CALUDE_balloon_altitude_l1737_173762


namespace NUMINAMATH_CALUDE_john_sublet_count_l1737_173798

/-- The number of people John sublets his apartment to -/
def num_subletters : ℕ := by sorry

/-- Monthly payment per subletter in dollars -/
def subletter_payment : ℕ := 400

/-- John's monthly rent in dollars -/
def john_rent : ℕ := 900

/-- John's annual profit in dollars -/
def annual_profit : ℕ := 3600

/-- Number of months in a year -/
def months_per_year : ℕ := 12

theorem john_sublet_count : 
  num_subletters * subletter_payment * months_per_year - john_rent * months_per_year = annual_profit → 
  num_subletters = 3 := by sorry

end NUMINAMATH_CALUDE_john_sublet_count_l1737_173798


namespace NUMINAMATH_CALUDE_shoe_probability_l1737_173791

def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def red_pairs : ℕ := 4
def white_pairs : ℕ := 3

def total_shoes : ℕ := total_pairs * 2

def favorable_outcomes : ℕ := black_pairs * black_pairs + red_pairs * red_pairs + white_pairs * white_pairs

theorem shoe_probability : 
  (favorable_outcomes : ℚ) / (total_shoes.choose 2) = 89 / 435 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_l1737_173791


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_inequality_three_solution_inequality_four_solution_system_five_solution_l1737_173718

-- System 1
theorem system_one_solution (x y : ℝ) : 
  x + y = 10 ∧ 2*x + y = 16 → x = 6 ∧ y = 4 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  4*(x - y - 1) = 3*(1 - y) - 2 ∧ x/2 + y/3 = 2 → x = 2 ∧ y = 3 := by sorry

-- Inequality 3
theorem inequality_three_solution (x : ℝ) : 
  10 - 4*(x - 4) ≤ 2*(x + 1) ↔ x ≥ 4 := by sorry

-- Inequality 4
theorem inequality_four_solution (y : ℝ) : 
  (y + 1)/6 - (2*y - 5)/4 ≥ 1 ↔ y ≤ 5/4 := by sorry

-- System 5
theorem system_five_solution (x : ℝ) : 
  x - 3*(x - 2) ≥ 4 ∧ (2*x - 1)/5 ≥ (x + 1)/2 → x ≤ -7 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_inequality_three_solution_inequality_four_solution_system_five_solution_l1737_173718


namespace NUMINAMATH_CALUDE_parking_arrangements_l1737_173732

def parking_spaces : ℕ := 7
def car_models : ℕ := 3
def consecutive_empty : ℕ := 3

theorem parking_arrangements :
  (car_models.factorial) *
  (parking_spaces - car_models).choose 2 *
  ((parking_spaces - car_models - consecutive_empty + 1).factorial) = 72 := by
  sorry

end NUMINAMATH_CALUDE_parking_arrangements_l1737_173732


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l1737_173771

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem eighth_odd_multiple_of_five : 
  ∀ (a d : ℕ),
    a = 5 → 
    d = 10 → 
    (∀ n : ℕ, n > 0 → arithmetic_sequence a d n % 2 = 1) →
    (∀ n : ℕ, n > 0 → arithmetic_sequence a d n % 5 = 0) →
    arithmetic_sequence a d 8 = 75 :=
by sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_five_l1737_173771


namespace NUMINAMATH_CALUDE_chessboard_border_covering_l1737_173766

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Number of ways to cover a 2xn rectangle with 1x2 dominos -/
def cover_2xn (n : ℕ) : ℕ := fib (n + 1)

/-- Number of ways to cover the 2-unit wide border of an 8x8 chessboard with 1x2 dominos -/
def cover_chessboard_border : ℕ :=
  let f9 := cover_2xn 8
  let f10 := cover_2xn 9
  let f11 := cover_2xn 10
  2 + 2 * f11^2 * f9^2 + 12 * f11 * f10^2 * f9 + 2 * f10^4

theorem chessboard_border_covering :
  cover_chessboard_border = 146458404 := by sorry

end NUMINAMATH_CALUDE_chessboard_border_covering_l1737_173766


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_bus_stop_walking_time_proof_l1737_173734

/-- The time to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 7 minutes later than normal, is 28 minutes. -/
theorem bus_stop_walking_time : ℝ → Prop :=
  fun T : ℝ =>
    (4 / 5 * T + 7 = T) → T = 28

/-- Proof of the bus_stop_walking_time theorem -/
theorem bus_stop_walking_time_proof : ∃ T : ℝ, bus_stop_walking_time T :=
  sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_bus_stop_walking_time_proof_l1737_173734


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1737_173700

theorem sum_of_coefficients (a b c : ℤ) : 
  (∀ x, x^2 + 10*x + 21 = (x + a) * (x + b)) →
  (∀ x, x^2 + 3*x - 88 = (x + b) * (x - c)) →
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1737_173700


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2005_l1737_173795

/-- Rainfall data for Mathborough from 2003 to 2005 -/
structure RainfallData where
  rainfall_2003 : ℝ
  increase_2004 : ℝ
  increase_2005 : ℝ

/-- Calculate the total rainfall in Mathborough for 2005 -/
def totalRainfall2005 (data : RainfallData) : ℝ :=
  12 * (data.rainfall_2003 + data.increase_2004 + data.increase_2005)

/-- Theorem stating the total rainfall in Mathborough for 2005 -/
theorem mathborough_rainfall_2005 (data : RainfallData)
  (h1 : data.rainfall_2003 = 50)
  (h2 : data.increase_2004 = 5)
  (h3 : data.increase_2005 = 3) :
  totalRainfall2005 data = 696 := by
  sorry

#eval totalRainfall2005 ⟨50, 5, 3⟩

end NUMINAMATH_CALUDE_mathborough_rainfall_2005_l1737_173795


namespace NUMINAMATH_CALUDE_limit_equivalence_l1737_173787

def has_limit (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∃ N : ℕ, ∀ n : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

def alt_def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, |L - u n| ≤ ε ∨ n < N)

def alt_def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∀ n : ℕ, ∃ N : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

def alt_def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∃ N : ℕ, ∀ n : ℕ, (ε > 0 ∧ n > N) → |L - u n| < ε

def alt_def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε : ℝ, ∀ n : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

theorem limit_equivalence (u : ℕ → ℝ) (L : ℝ) :
  (has_limit u L ↔ alt_def1 u L) ∧
  (has_limit u L ↔ alt_def3 u L) ∧
  ¬(has_limit u L ↔ alt_def2 u L) ∧
  ¬(has_limit u L ↔ alt_def4 u L) := by
  sorry

end NUMINAMATH_CALUDE_limit_equivalence_l1737_173787


namespace NUMINAMATH_CALUDE_division_with_remainder_l1737_173724

theorem division_with_remainder (A : ℕ) : 
  (A / 9 = 6) ∧ (A % 9 = 5) → A = 59 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l1737_173724


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1737_173712

theorem complex_fraction_evaluation : (1 - I) / (2 + I) = 1/5 - 3/5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1737_173712


namespace NUMINAMATH_CALUDE_base_eight_solution_l1737_173769

/-- Converts a list of digits in base h to its decimal representation -/
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldl (fun acc d => acc * h + d) 0

/-- Checks if the equation holds for a given base h -/
def equation_holds (h : Nat) : Prop :=
  to_decimal [9, 8, 7, 6, 5, 4] h + to_decimal [6, 9, 8, 5, 5, 5] h = to_decimal [1, 7, 9, 6, 2, 2, 9] h

theorem base_eight_solution :
  ∃ (h : Nat), h > 0 ∧ equation_holds h ∧ ∀ (k : Nat), k > 0 ∧ equation_holds k → k = h :=
by
  sorry

end NUMINAMATH_CALUDE_base_eight_solution_l1737_173769


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l1737_173752

/-- Given a segment with endpoints A(x₁, y₁) and B(x₂, y₂), and its midpoint M(x₀, y₀),
    prove that the coordinates of the midpoint are the averages of the endpoints' coordinates. -/
theorem midpoint_coordinates (x₀ x₁ x₂ y₀ y₁ y₂ : ℝ) :
  (∀ t : ℝ, t ∈ (Set.Icc 0 1) → 
    (x₀ = (1 - t) * x₁ + t * x₂ ∧ 
     y₀ = (1 - t) * y₁ + t * y₂)) →
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 := by
  sorry


end NUMINAMATH_CALUDE_midpoint_coordinates_l1737_173752


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1737_173780

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) :
  x^3 + y^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1737_173780


namespace NUMINAMATH_CALUDE_roger_trays_second_table_l1737_173735

/-- Represents the number of trays Roger can carry in one trip -/
def trays_per_trip : ℕ := 4

/-- Represents the number of trips Roger made -/
def num_trips : ℕ := 3

/-- Represents the number of trays Roger picked up from the first table -/
def trays_first_table : ℕ := 10

/-- Calculates the number of trays Roger picked up from the second table -/
def trays_second_table : ℕ := trays_per_trip * num_trips - trays_first_table

theorem roger_trays_second_table : trays_second_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_roger_trays_second_table_l1737_173735


namespace NUMINAMATH_CALUDE_complement_of_angle_l1737_173785

theorem complement_of_angle (A : ℝ) (h : A = 35) : 90 - A = 55 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_l1737_173785


namespace NUMINAMATH_CALUDE_no_prime_fraction_equality_l1737_173783

theorem no_prime_fraction_equality : ¬∃ (a b c d : ℕ), 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
  a < b ∧ b < c ∧ c < d ∧
  (1 : ℚ) / a + (1 : ℚ) / d = (1 : ℚ) / b + (1 : ℚ) / c := by
  sorry

end NUMINAMATH_CALUDE_no_prime_fraction_equality_l1737_173783


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l1737_173744

/-- Given three angles around a point B, prove that ∠ABC = 60° -/
theorem angle_ABC_measure (ABC ABD CBD : ℝ) : 
  CBD = 90 → ABD = 30 → ABC + ABD + CBD = 180 → ABC = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l1737_173744


namespace NUMINAMATH_CALUDE_female_fraction_is_19_52_l1737_173703

/-- Represents the chess club membership --/
structure ChessClub where
  males_last_year : ℕ
  total_increase_rate : ℚ
  male_increase_rate : ℚ
  female_increase_rate : ℚ

/-- Calculates the fraction of female participants in the chess club this year --/
def female_fraction (club : ChessClub) : ℚ :=
  sorry

/-- Theorem stating that the fraction of female participants is 19/52 --/
theorem female_fraction_is_19_52 (club : ChessClub) 
  (h1 : club.males_last_year = 30)
  (h2 : club.total_increase_rate = 15/100)
  (h3 : club.male_increase_rate = 10/100)
  (h4 : club.female_increase_rate = 25/100) :
  female_fraction club = 19/52 := by
  sorry

end NUMINAMATH_CALUDE_female_fraction_is_19_52_l1737_173703


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1737_173753

theorem absolute_value_inequality (x : ℝ) :
  |x - 2| + |x + 3| < 6 ↔ -7/2 < x ∧ x < 5/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1737_173753


namespace NUMINAMATH_CALUDE_jennifers_cans_count_l1737_173720

/-- The number of cans Jennifer brought home from the store -/
def jennifers_total_cans (initial_cans : ℕ) (marks_cans : ℕ) : ℕ :=
  initial_cans + (6 * marks_cans) / 5

/-- Theorem stating the total number of cans Jennifer brought home -/
theorem jennifers_cans_count : jennifers_total_cans 40 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jennifers_cans_count_l1737_173720


namespace NUMINAMATH_CALUDE_contrapositive_even_sum_l1737_173747

theorem contrapositive_even_sum (a b : ℤ) : 
  (¬(Even (a + b)) → ¬(Even a ∧ Even b)) ↔ 
  (∀ (a b : ℤ), (Even a ∧ Even b) → Even (a + b))ᶜ :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_even_sum_l1737_173747


namespace NUMINAMATH_CALUDE_discount_calculation_l1737_173730

theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) :
  list_price = 65 →
  final_price = 57.33 →
  first_discount = 10 →
  ∃ second_discount : ℝ,
    second_discount = 2 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1737_173730


namespace NUMINAMATH_CALUDE_math_homework_pages_l1737_173727

theorem math_homework_pages (reading : ℕ) (math : ℕ) : 
  math = reading + 3 →
  reading + math = 13 →
  math = 8 := by
sorry

end NUMINAMATH_CALUDE_math_homework_pages_l1737_173727


namespace NUMINAMATH_CALUDE_total_taco_combinations_l1737_173742

/-- The number of optional toppings available for tacos. -/
def num_toppings : ℕ := 8

/-- The number of meat options available for tacos. -/
def meat_options : ℕ := 3

/-- The number of shell options available for tacos. -/
def shell_options : ℕ := 2

/-- Calculates the total number of different taco combinations. -/
def taco_combinations : ℕ := 2^num_toppings * meat_options * shell_options

/-- Theorem stating that the total number of taco combinations is 1536. -/
theorem total_taco_combinations : taco_combinations = 1536 := by
  sorry

end NUMINAMATH_CALUDE_total_taco_combinations_l1737_173742


namespace NUMINAMATH_CALUDE_no_solution_exists_l1737_173733

theorem no_solution_exists : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (∃ (n : ℕ), 
    ((b + c - a) / a = n) ∧
    ((a + c - b) / b = n) ∧
    ((a + b - c) / c = n)) ∧
  ((a + b) * (b + c) * (a + c)) / (a * b * c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1737_173733


namespace NUMINAMATH_CALUDE_river_speed_l1737_173797

/-- Proves that the speed of the river is 1.2 kmph given the conditions of the rowing problem -/
theorem river_speed (still_water_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) :
  still_water_speed = 10 →
  total_time = 1 →
  total_distance = 9.856 →
  ∃ (river_speed : ℝ),
    river_speed = 1.2 ∧
    total_distance = (still_water_speed - river_speed) * (total_time / 2) +
                     (still_water_speed + river_speed) * (total_time / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_river_speed_l1737_173797


namespace NUMINAMATH_CALUDE_min_value_of_z_l1737_173736

-- Define the variables and the objective function
variables (x y : ℝ)
def z (x y : ℝ) : ℝ := 2 * x + y

-- State the theorem
theorem min_value_of_z (hx : x ≥ 1) (hxy : x + y ≤ 3) (hxy2 : x - 2 * y - 3 ≤ 0) :
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 1 ∧ x₀ + y₀ ≤ 3 ∧ x₀ - 2 * y₀ - 3 ≤ 0 ∧
  ∀ (x y : ℝ), x ≥ 1 → x + y ≤ 3 → x - 2 * y - 3 ≤ 0 → z x₀ y₀ ≤ z x y ∧ z x₀ y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1737_173736


namespace NUMINAMATH_CALUDE_linear_function_theorem_l1737_173719

/-- A linear function satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ := sorry

theorem linear_function_theorem :
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) → -- f is linear
  (∀ x : ℝ, f x = 3 * f_inv x + 9) → -- f(x) = 3f^(-1)(x) + 9
  f 0 = 3 → -- f(0) = 3
  f_inv 3 = 0 → -- f^(-1)(3) = 0
  f 3 = 6 * Real.sqrt 3 := by -- f(3) = 6√3
sorry

end NUMINAMATH_CALUDE_linear_function_theorem_l1737_173719


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1737_173738

theorem complex_fraction_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (2 + i) / (1 + i) = a + b * i → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1737_173738


namespace NUMINAMATH_CALUDE_list_average_problem_l1737_173725

theorem list_average_problem (n : ℕ) (original_avg : ℚ) (new_avg : ℚ) (added_num : ℤ) : 
  original_avg = 7 →
  new_avg = 6 →
  added_num = -11 →
  (n : ℚ) * original_avg + added_num = (n + 1 : ℚ) * new_avg →
  n = 17 := by sorry

end NUMINAMATH_CALUDE_list_average_problem_l1737_173725


namespace NUMINAMATH_CALUDE_larry_lunch_cost_l1737_173784

/-- Calculates the amount spent on lunch given initial amount, final amount, and amount given to brother -/
def lunch_cost (initial : ℕ) (final : ℕ) (given_to_brother : ℕ) : ℕ :=
  initial - final - given_to_brother

/-- Proves that Larry's lunch cost is $5 given the problem conditions -/
theorem larry_lunch_cost :
  lunch_cost 22 15 2 = 5 := by sorry

end NUMINAMATH_CALUDE_larry_lunch_cost_l1737_173784


namespace NUMINAMATH_CALUDE_seating_arrangements_l1737_173715

/-- The number of ways to arrange n elements --/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := n.choose k

/-- The number of seating arrangements for 5 people in 5 seats --/
def totalArrangements : ℕ := arrangements 5

/-- The number of arrangements where 3 people are in their numbered seats --/
def threeInPlace : ℕ := choose 5 3 * arrangements 2

/-- The number of arrangements where all 5 people are in their numbered seats --/
def allInPlace : ℕ := 1

theorem seating_arrangements :
  totalArrangements - threeInPlace - allInPlace = 109 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1737_173715


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1737_173706

theorem inequality_solution_set (x : ℝ) :
  (4 * x^4 + x^2 + 4*x - 5 * x^2 * |x + 2| + 4 ≥ 0) ↔ 
  (x ≤ -1 ∨ ((1 - Real.sqrt 33) / 8 ≤ x ∧ x ≤ (1 + Real.sqrt 33) / 8) ∨ x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1737_173706


namespace NUMINAMATH_CALUDE_special_function_at_one_l1737_173782

/-- A monotonic function on positive real numbers satisfying f(f(x) - ln x) = 1 + e -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, ∀ y > 0, x < y → f x < f y) ∧
  (∀ x > 0, f (f x - Real.log x) = 1 + Real.exp 1)

/-- The value of f(1) for a special function f is e -/
theorem special_function_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_one_l1737_173782


namespace NUMINAMATH_CALUDE_new_person_weight_l1737_173772

theorem new_person_weight (n : ℕ) (old_weight average_increase : ℝ) :
  n = 10 →
  old_weight = 65 →
  average_increase = 3.2 →
  ∃ (new_weight : ℝ),
    new_weight = old_weight + n * average_increase ∧
    new_weight = 97 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1737_173772


namespace NUMINAMATH_CALUDE_f_value_at_7_l1737_173714

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^8 + b * x^7 + c * x^3 + d * x - 6

-- State the theorem
theorem f_value_at_7 (a b c d : ℝ) :
  f a b c d (-7) = 10 → f a b c d 7 = 11529580 * a - 22 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_l1737_173714


namespace NUMINAMATH_CALUDE_compound_propositions_truth_count_l1737_173761

theorem compound_propositions_truth_count
  (p q : Prop)
  (hp : p)
  (hq : ¬q) :
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) ∧ (¬q) :=
by sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_count_l1737_173761


namespace NUMINAMATH_CALUDE_apples_left_l1737_173740

def apples_bought : ℕ := 15
def apples_given : ℕ := 7

theorem apples_left : apples_bought - apples_given = 8 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l1737_173740


namespace NUMINAMATH_CALUDE_exam_average_problem_l1737_173731

theorem exam_average_problem (total_students : ℕ) (high_score_students : ℕ) (high_score : ℝ) (total_average : ℝ) :
  total_students = 25 →
  high_score_students = 10 →
  high_score = 90 →
  total_average = 84 →
  ∃ (low_score_students : ℕ) (low_score : ℝ),
    low_score_students + high_score_students = total_students ∧
    low_score = 80 ∧
    (low_score_students * low_score + high_score_students * high_score) / total_students = total_average ∧
    low_score_students = 15 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_problem_l1737_173731


namespace NUMINAMATH_CALUDE_correct_average_weight_l1737_173750

theorem correct_average_weight 
  (num_boys : ℕ) 
  (initial_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) : 
  num_boys = 20 → 
  initial_avg = 58.4 → 
  misread_weight = 56 → 
  correct_weight = 65 → 
  (num_boys * initial_avg + correct_weight - misread_weight) / num_boys = 58.85 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l1737_173750


namespace NUMINAMATH_CALUDE_intercept_sum_l1737_173778

/-- A line is described by the equation y + 3 = -3(x + 2) -/
def line_equation (x y : ℝ) : Prop := y + 3 = -3 * (x + 2)

/-- The x-intercept of the line -/
def x_intercept : ℝ := -3

/-- The y-intercept of the line -/
def y_intercept : ℝ := -9

theorem intercept_sum :
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept ∧ x_intercept + y_intercept = -12 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l1737_173778
