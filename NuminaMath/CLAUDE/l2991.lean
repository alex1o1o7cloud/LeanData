import Mathlib

namespace NUMINAMATH_CALUDE_flowers_died_in_danes_garden_l2991_299196

/-- The number of flowers that died in Dane's daughters' garden -/
def flowers_died (initial_flowers : ℕ) (new_flowers : ℕ) (baskets : ℕ) (flowers_per_basket : ℕ) : ℕ :=
  initial_flowers + new_flowers - (baskets * flowers_per_basket)

/-- Theorem stating the number of flowers that died in the specific scenario -/
theorem flowers_died_in_danes_garden : flowers_died 10 20 5 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_flowers_died_in_danes_garden_l2991_299196


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2991_299104

/-- Given a triangle ABC with points E on BC and G on AB, and Q the intersection of AE and CG,
    if AQ:QE = 3:2 and GQ:QC = 2:3, then AG:GB = 1:2 -/
theorem triangle_ratio_theorem (A B C E G Q : ℝ × ℝ) : 
  (E.1 - B.1) / (C.1 - B.1) = (E.2 - B.2) / (C.2 - B.2) →  -- E is on BC
  (G.1 - A.1) / (B.1 - A.1) = (G.2 - A.2) / (B.2 - A.2) →  -- G is on AB
  ∃ (t : ℝ), Q = (1 - t) • A + t • E ∧                     -- Q is on AE
             Q = (1 - t) • C + t • G →                     -- Q is on CG
  (Q.1 - A.1) / (E.1 - Q.1) = 3 / 2 →                      -- AQ:QE = 3:2
  (G.1 - Q.1) / (Q.1 - C.1) = 2 / 3 →                      -- GQ:QC = 2:3
  (G.1 - A.1) / (B.1 - G.1) = 1 / 2 :=                     -- AG:GB = 1:2
by sorry


end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2991_299104


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2991_299157

theorem smallest_solution_of_equation : 
  ∃ x : ℝ, x^4 - 50*x^2 + 625 = 0 ∧ 
  (∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → x ≤ y) ∧ 
  x = -5 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2991_299157


namespace NUMINAMATH_CALUDE_walking_distance_l2991_299115

/-- The distance traveled given a constant speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that walking at 4 miles per hour for 2 hours results in 8 miles traveled -/
theorem walking_distance : distance 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l2991_299115


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2991_299149

theorem complex_equation_solution (z : ℂ) : (2 - Complex.I) * z = 4 + 3 * Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2991_299149


namespace NUMINAMATH_CALUDE_staff_assignment_arrangements_l2991_299153

/-- The number of staff members --/
def n : ℕ := 7

/-- The number of days --/
def d : ℕ := 7

/-- Staff member A cannot be assigned on the first day --/
def a_constraint : Prop := true

/-- Staff member B cannot be assigned on the last day --/
def b_constraint : Prop := true

/-- The number of different arrangements --/
def num_arrangements : ℕ := 3720

/-- Theorem stating the number of different arrangements --/
theorem staff_assignment_arrangements :
  a_constraint → b_constraint → num_arrangements = 3720 := by
  sorry

end NUMINAMATH_CALUDE_staff_assignment_arrangements_l2991_299153


namespace NUMINAMATH_CALUDE_rectangle_to_square_l2991_299102

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents the area of a shape -/
def area : Rectangle → ℕ
  | ⟨l, w⟩ => l * w

/-- Theorem stating that a 9x4 rectangle can be cut and rearranged into a 6x6 square -/
theorem rectangle_to_square : 
  ∃ (r : Rectangle) (s : Square), 
    r.length = 9 ∧ 
    r.width = 4 ∧ 
    s.side = 6 ∧ 
    area r = s.side * s.side := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l2991_299102


namespace NUMINAMATH_CALUDE_max_distance_complex_l2991_299163

theorem max_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (w : ℂ), Complex.abs (w + 2 - 2*I) = 1 ∧
             ∀ (v : ℂ), Complex.abs (v + 2 - 2*I) = 1 →
                        Complex.abs (v - 2 - 2*I) ≤ Complex.abs (w - 2 - 2*I) ∧
             Complex.abs (w - 2 - 2*I) = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2991_299163


namespace NUMINAMATH_CALUDE_fixed_monthly_charge_l2991_299132

-- Define the fixed monthly charge for internet service
def F : ℝ := sorry

-- Define the charge for calls in January
def C : ℝ := sorry

-- Define the total bill for January
def january_bill : ℝ := 50

-- Define the total bill for February
def february_bill : ℝ := 76

-- Theorem to prove the fixed monthly charge for internet service
theorem fixed_monthly_charge :
  (F + C = january_bill) →
  (F + 2 * C = february_bill) →
  F = 24 := by sorry

end NUMINAMATH_CALUDE_fixed_monthly_charge_l2991_299132


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_negative_one_l2991_299161

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_negative_one :
  log10 (5/2) + 2 * log10 2 - (1/2)⁻¹ = -1 :=
by
  -- Assume the given condition
  have h : log10 2 + log10 5 = 1 := by sorry
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_negative_one_l2991_299161


namespace NUMINAMATH_CALUDE_contest_permutations_l2991_299100

/-- The number of letters in "CONTEST" -/
def total_letters : ℕ := 7

/-- The number of vowels in "CONTEST" -/
def num_vowels : ℕ := 2

/-- The number of consonants in "CONTEST" -/
def num_consonants : ℕ := 5

/-- The number of repeated consonants in "CONTEST" -/
def num_repeated_consonants : ℕ := 1

theorem contest_permutations : 
  (num_vowels.factorial) * (num_consonants.factorial / num_repeated_consonants.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_contest_permutations_l2991_299100


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l2991_299107

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 6) 7 = 84 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l2991_299107


namespace NUMINAMATH_CALUDE_tangent_length_l2991_299111

/-- The circle C with equation x^2 + y^2 - 2x - 6y + 9 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

/-- The point P on the x-axis -/
def point_P : ℝ × ℝ := (1, 0)

/-- The length of the tangent from P to circle C is 2√2 -/
theorem tangent_length : 
  ∃ (t : ℝ × ℝ), 
    circle_C t.1 t.2 ∧ 
    ((t.1 - point_P.1)^2 + (t.2 - point_P.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_length_l2991_299111


namespace NUMINAMATH_CALUDE_victors_hourly_rate_l2991_299198

theorem victors_hourly_rate (hours_worked : ℕ) (total_earned : ℕ) 
  (h1 : hours_worked = 10) 
  (h2 : total_earned = 60) : 
  total_earned / hours_worked = 6 := by
  sorry

end NUMINAMATH_CALUDE_victors_hourly_rate_l2991_299198


namespace NUMINAMATH_CALUDE_lily_hydrangea_plants_l2991_299184

/-- Prove that Lily buys 1 hydrangea plant per year -/
theorem lily_hydrangea_plants (start_year end_year : ℕ) (plant_cost total_spent : ℚ) : 
  start_year = 1989 →
  end_year = 2021 →
  plant_cost = 20 →
  total_spent = 640 →
  (total_spent / (end_year - start_year : ℚ)) / plant_cost = 1 := by
  sorry

end NUMINAMATH_CALUDE_lily_hydrangea_plants_l2991_299184


namespace NUMINAMATH_CALUDE_magical_stack_size_l2991_299199

/-- Represents a stack of cards -/
structure CardStack :=
  (n : ℕ)  -- Half the total number of cards

/-- Checks if a card number is in its original position after restacking -/
def retains_position (stack : CardStack) (card : ℕ) : Prop :=
  card ≤ stack.n → card = 2 * card - 1
  ∧ card > stack.n → card = 2 * (card - stack.n)

/-- Defines a magical stack -/
def is_magical (stack : CardStack) : Prop :=
  ∃ (a b : ℕ), a ≤ stack.n ∧ b > stack.n ∧ retains_position stack a ∧ retains_position stack b

/-- Main theorem: A magical stack where card 161 retains its position has 482 cards -/
theorem magical_stack_size :
  ∀ (stack : CardStack),
    is_magical stack →
    retains_position stack 161 →
    2 * stack.n = 482 :=
by sorry

end NUMINAMATH_CALUDE_magical_stack_size_l2991_299199


namespace NUMINAMATH_CALUDE_firm_ratio_proof_l2991_299164

/-- Represents the number of partners in the firm -/
def partners : ℕ := 20

/-- Represents the additional associates to be hired -/
def additional_associates : ℕ := 50

/-- Represents the ratio of partners to associates after hiring additional associates -/
def new_ratio : ℚ := 1 / 34

/-- Calculates the initial number of associates in the firm -/
def initial_associates : ℕ := partners * 34 - additional_associates

/-- Represents the initial ratio of partners to associates -/
def initial_ratio : ℚ := partners / initial_associates

theorem firm_ratio_proof :
  initial_ratio = 2 / 63 := by
  sorry

end NUMINAMATH_CALUDE_firm_ratio_proof_l2991_299164


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2991_299172

def number_of_ways_to_form_subcommittee (total_republicans : ℕ) (total_democrats : ℕ) (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_count :
  number_of_ways_to_form_subcommittee 10 7 4 3 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2991_299172


namespace NUMINAMATH_CALUDE_class_size_l2991_299101

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 6) :
  french + german - both + neither = 60 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2991_299101


namespace NUMINAMATH_CALUDE_sum_reciprocal_F_powers_of_two_converges_to_one_l2991_299162

/-- Definition of the sequence F -/
def F : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n+2) => (3/2) * F (n+1) - (1/2) * F n

/-- The sum of the reciprocals of F(2^n) converges to 1 -/
theorem sum_reciprocal_F_powers_of_two_converges_to_one :
  ∑' n, (1 : ℝ) / F (2^n) = 1 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_F_powers_of_two_converges_to_one_l2991_299162


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2991_299120

/-- A function that returns the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

/-- A predicate that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- A predicate that checks if a natural number is even -/
def is_even (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

/-- A predicate that checks if a natural number is a multiple of 9 -/
def is_multiple_of_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

/-- A predicate that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem unique_two_digit_number : 
  ∀ n : ℕ, 
    is_two_digit n ∧ 
    is_even n ∧ 
    is_multiple_of_9 n ∧ 
    is_perfect_square (tens_digit n * ones_digit n) → 
    n = 90 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2991_299120


namespace NUMINAMATH_CALUDE_largest_integer_divisibility_l2991_299173

theorem largest_integer_divisibility : ∃ (n : ℕ), n = 1956 ∧ 
  (∀ m : ℕ, m > n → ¬(∃ k : ℤ, (m^2 - 2012 : ℤ) = k * (m + 7))) ∧
  (∃ k : ℤ, (n^2 - 2012 : ℤ) = k * (n + 7)) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_divisibility_l2991_299173


namespace NUMINAMATH_CALUDE_triangle_problem_l2991_299190

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B →
  a + c = 6 →
  (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 →
  B = π / 3 ∧ b = 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2991_299190


namespace NUMINAMATH_CALUDE_logo_area_difference_l2991_299171

/-- The logo problem -/
theorem logo_area_difference :
  let triangle_side : ℝ := 12
  let square_side : ℝ := 2 * (9 - 3 * Real.sqrt 3)
  let overlapped_area : ℝ := square_side^2 - (square_side / 2) * (triangle_side - square_side / 2)
  let non_overlapping_area : ℝ := 2 * (square_side / 2) * (triangle_side - square_side / 2) / Real.sqrt 3
  overlapped_area - non_overlapping_area = 102.6 - 57.6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_logo_area_difference_l2991_299171


namespace NUMINAMATH_CALUDE_trailing_zeroes_1500_factorial_l2991_299186

def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem trailing_zeroes_1500_factorial :
  trailingZeroes 1500 = 374 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_1500_factorial_l2991_299186


namespace NUMINAMATH_CALUDE_inequalities_count_l2991_299135

theorem inequalities_count (a c : ℝ) (h : a * c < 0) :
  ∃! n : ℕ, n = (Bool.toNat (a / c < 0) +
                 Bool.toNat (a * c^2 < 0) +
                 Bool.toNat (a^2 * c < 0) +
                 Bool.toNat (c^3 * a < 0) +
                 Bool.toNat (c * a^3 < 0)) ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_count_l2991_299135


namespace NUMINAMATH_CALUDE_exists_monotone_increasing_symmetric_about_point_exists_three_roots_l2991_299192

-- Define the function f
def f (b c x : ℝ) : ℝ := |x| * x + b * x + c

-- Statement 1
theorem exists_monotone_increasing :
  ∃ b : ℝ, b > 0 ∧ ∀ x y : ℝ, x < y → f b 0 x < f b 0 y :=
sorry

-- Statement 2
theorem symmetric_about_point :
  ∀ b c : ℝ, ∀ x : ℝ, f b c x = f b c (-x) :=
sorry

-- Statement 3
theorem exists_three_roots :
  ∃ b c : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_monotone_increasing_symmetric_about_point_exists_three_roots_l2991_299192


namespace NUMINAMATH_CALUDE_cut_length_of_divided_square_l2991_299151

/-- Represents a square cake divided into four equal pieces by two straight cuts -/
structure DividedSquare where
  side_length : ℝ
  cut_length : ℝ

/-- The perimeter of the original square cake -/
def square_perimeter (s : DividedSquare) : ℝ := 4 * s.side_length

/-- The perimeter of each smaller piece after division -/
def piece_perimeter (s : DividedSquare) : ℝ := 2 * s.side_length + 2 * s.cut_length

/-- Theorem stating the length of the cut in a divided square cake -/
theorem cut_length_of_divided_square (s : DividedSquare) 
  (h1 : square_perimeter s = 100)
  (h2 : piece_perimeter s = 56) : 
  s.cut_length = 3 := by sorry

end NUMINAMATH_CALUDE_cut_length_of_divided_square_l2991_299151


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2991_299128

/-- Given two points A(-3, y₁) and B(2, y₂) on the graph of y = 6/x, prove that y₁ < y₂ -/
theorem inverse_proportion_y_relationship (y₁ y₂ : ℝ) : 
  y₁ = 6 / (-3) → y₂ = 6 / 2 → y₁ < y₂ := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l2991_299128


namespace NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_initial_speed_is_unique_l2991_299106

/-- The child's initial walking speed in meters per minute -/
def initial_speed : ℝ := 5

/-- The time it takes for the child to walk to school at the initial speed -/
def initial_time : ℝ := 126

/-- The distance from home to school in meters -/
def distance : ℝ := 630

/-- Theorem stating that the initial speed satisfies the given conditions -/
theorem initial_speed_satisfies_conditions :
  distance = initial_speed * initial_time ∧
  distance = 7 * (initial_time - 36) :=
sorry

/-- Theorem proving that the initial speed is unique -/
theorem initial_speed_is_unique (v : ℝ) :
  (∃ t : ℝ, distance = v * t ∧ distance = 7 * (t - 36)) →
  v = initial_speed :=
sorry

end NUMINAMATH_CALUDE_initial_speed_satisfies_conditions_initial_speed_is_unique_l2991_299106


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l2991_299169

-- Define the initial amount in the piggy bank
def initial_amount : ℕ := 200

-- Define the cost per store trip
def cost_per_trip : ℕ := 2

-- Define the number of trips per month
def trips_per_month : ℕ := 4

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Define the function to calculate the remaining amount
def remaining_amount : ℕ :=
  initial_amount - (cost_per_trip * trips_per_month * months_in_year)

-- Theorem to prove
theorem piggy_bank_savings : remaining_amount = 104 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l2991_299169


namespace NUMINAMATH_CALUDE_prob_tails_at_least_twice_eq_half_l2991_299159

/-- Probability of getting tails k times in n flips of a fair coin -/
def binomialProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The number of coin flips -/
def numFlips : ℕ := 3

/-- Probability of getting tails at least twice but not more than 3 times in 3 flips -/
def probTailsAtLeastTwice : ℚ :=
  binomialProbability numFlips 2 + binomialProbability numFlips 3

theorem prob_tails_at_least_twice_eq_half :
  probTailsAtLeastTwice = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_tails_at_least_twice_eq_half_l2991_299159


namespace NUMINAMATH_CALUDE_wendy_polished_glasses_l2991_299166

def small_glasses : ℕ := 50
def large_glasses : ℕ := small_glasses + 10

theorem wendy_polished_glasses : small_glasses + large_glasses = 110 := by
  sorry

end NUMINAMATH_CALUDE_wendy_polished_glasses_l2991_299166


namespace NUMINAMATH_CALUDE_video_votes_l2991_299180

theorem video_votes (net_score : ℚ) (like_percentage : ℚ) (dislike_percentage : ℚ) :
  net_score = 75 →
  like_percentage = 55 / 100 →
  dislike_percentage = 45 / 100 →
  like_percentage + dislike_percentage = 1 →
  ∃ (total_votes : ℚ),
    total_votes * (like_percentage - dislike_percentage) = net_score ∧
    total_votes = 750 :=
by sorry

end NUMINAMATH_CALUDE_video_votes_l2991_299180


namespace NUMINAMATH_CALUDE_sams_morning_run_l2991_299154

theorem sams_morning_run (morning_run : ℝ) 
  (store_walk : ℝ)
  (bike_ride : ℝ)
  (total_distance : ℝ)
  (h1 : store_walk = 2 * morning_run)
  (h2 : bike_ride = 12)
  (h3 : total_distance = 18)
  (h4 : morning_run + store_walk + bike_ride = total_distance) :
  morning_run = 2 := by
sorry

end NUMINAMATH_CALUDE_sams_morning_run_l2991_299154


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2991_299114

/-- The number of integer solutions to the equation 2x^2 + 5xy + 3y^2 = 30 -/
def num_solutions : ℕ := 16

/-- The quadratic equation -/
def quadratic_equation (x y : ℤ) : Prop :=
  2 * x^2 + 5 * x * y + 3 * y^2 = 30

/-- Known solution to the equation -/
def known_solution : ℤ × ℤ := (9, -4)

theorem quadratic_equation_solutions :
  (quadratic_equation known_solution.1 known_solution.2) ∧
  (∃ (solutions : Finset (ℤ × ℤ)), 
    solutions.card = num_solutions ∧
    ∀ (sol : ℤ × ℤ), sol ∈ solutions ↔ quadratic_equation sol.1 sol.2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2991_299114


namespace NUMINAMATH_CALUDE_sum_base3_equals_10200_l2991_299176

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun digit acc => acc * 3 + digit) 0

/-- Represents a number in base 3 -/
structure Base3 where
  digits : List Nat
  valid : ∀ d ∈ digits, d < 3

/-- Addition of Base3 numbers -/
def addBase3 (a b : Base3) : Base3 :=
  sorry

theorem sum_base3_equals_10200 :
  let a := Base3.mk [1] (by simp)
  let b := Base3.mk [2, 1] (by simp)
  let c := Base3.mk [2, 1, 2] (by simp)
  let d := Base3.mk [1, 2, 1, 2] (by simp)
  let result := Base3.mk [0, 0, 2, 0, 1] (by simp)
  addBase3 (addBase3 (addBase3 a b) c) d = result :=
sorry

end NUMINAMATH_CALUDE_sum_base3_equals_10200_l2991_299176


namespace NUMINAMATH_CALUDE_extremum_and_tangent_imply_max_min_difference_l2991_299119

def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

theorem extremum_and_tangent_imply_max_min_difference
  (a b c : ℝ) :
  (∃ x, deriv (f a b c) x = 0 ∧ x = 2) →
  (deriv (f a b c) 1 = -3) →
  ∃ max min : ℝ, 
    (∀ x, f a b c x ≤ max) ∧
    (∀ x, f a b c x ≥ min) ∧
    (max - min = 4) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_tangent_imply_max_min_difference_l2991_299119


namespace NUMINAMATH_CALUDE_joggers_problem_l2991_299197

theorem joggers_problem (tyson alexander christopher : ℕ) : 
  alexander = tyson + 22 →
  christopher = 20 * tyson →
  christopher = alexander + 54 →
  christopher = 80 :=
by sorry

end NUMINAMATH_CALUDE_joggers_problem_l2991_299197


namespace NUMINAMATH_CALUDE_boys_count_in_class_l2991_299158

theorem boys_count_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h1 : total = 49) (h2 : boy_ratio = 4) (h3 : girl_ratio = 3) :
  (total * boy_ratio) / (boy_ratio + girl_ratio) = 28 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_in_class_l2991_299158


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2991_299124

theorem inequality_solution_set (x : ℝ) :
  (x - 3) / (x^2 - 2*x + 11) ≥ 0 ↔ x ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2991_299124


namespace NUMINAMATH_CALUDE_max_toys_buyable_l2991_299168

def initial_amount : ℕ := 57
def game_cost : ℕ := 27
def toy_cost : ℕ := 6

theorem max_toys_buyable : 
  (initial_amount - game_cost) / toy_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_toys_buyable_l2991_299168


namespace NUMINAMATH_CALUDE_complex_number_properties_l2991_299188

theorem complex_number_properties (z : ℂ) (h : z - 2*I = z*I + 4) : 
  Complex.abs z = Real.sqrt 10 ∧ ((z - 1) / 3) ^ 2023 = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2991_299188


namespace NUMINAMATH_CALUDE_ellipse_ratio_squared_l2991_299142

/-- For an ellipse with semi-major axis a, semi-minor axis b, and distance from center to focus c,
    if b/a = a/c and c^2 = a^2 - b^2, then (b/a)^2 = 1/2 -/
theorem ellipse_ratio_squared (a b c : ℝ) (h1 : b / a = a / c) (h2 : c^2 = a^2 - b^2) :
  (b / a)^2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_squared_l2991_299142


namespace NUMINAMATH_CALUDE_union_of_sets_l2991_299127

def A (a : ℕ) : Set ℕ := {3, 2^a}
def B (a b : ℕ) : Set ℕ := {a, b}

theorem union_of_sets (a b : ℕ) (h : A a ∩ B a b = {2}) : A a ∪ B a b = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2991_299127


namespace NUMINAMATH_CALUDE_train_speed_l2991_299177

/-- Calculates the speed of a train passing through a tunnel -/
theorem train_speed (train_length : ℝ) (tunnel_length : ℝ) (time_minutes : ℝ) :
  train_length = 1 →
  tunnel_length = 70 →
  time_minutes = 6 →
  (train_length + tunnel_length) / (time_minutes / 60) = 710 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2991_299177


namespace NUMINAMATH_CALUDE_vectors_collinear_l2991_299126

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b in ℝ², prove they are collinear -/
theorem vectors_collinear :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (1, -2)
  collinear a b := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l2991_299126


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2991_299118

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 1 ↔ -2 < x ∧ x < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2991_299118


namespace NUMINAMATH_CALUDE_at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one_l2991_299170

theorem at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one (n : ℕ) (hn : n > 1) :
  ∃! p : ℕ, Prime p ∧ n! < p ∧ p < n! + n + 1 :=
by sorry

end NUMINAMATH_CALUDE_at_most_one_prime_between_factorial_and_factorial_plus_n_plus_one_l2991_299170


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2991_299160

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (loss_margin : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 2000 →
  loss_margin = 640 →
  candidate_percentage * total_votes + (candidate_percentage * total_votes + loss_margin) = total_votes →
  candidate_percentage = 34 / 100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2991_299160


namespace NUMINAMATH_CALUDE_ones_divisible_by_power_of_three_l2991_299116

/-- Given a natural number n ≥ 1, the function returns the number formed by 3^n consecutive ones. -/
def number_of_ones (n : ℕ) : ℕ :=
  (10^(3^n) - 1) / 9

/-- Theorem stating that for any natural number n ≥ 1, the number formed by 3^n consecutive ones
    is divisible by 3^n. -/
theorem ones_divisible_by_power_of_three (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, number_of_ones n = 3^n * k :=
sorry

end NUMINAMATH_CALUDE_ones_divisible_by_power_of_three_l2991_299116


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2991_299185

theorem min_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 12) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 12 → 1/x + 1/y ≤ 1/a + 1/b) → 1/x + 1/y = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2991_299185


namespace NUMINAMATH_CALUDE_friday_dinner_customers_l2991_299145

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The predicted number of customers for Saturday -/
def saturday_prediction : ℕ := 574

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

theorem friday_dinner_customers : 
  dinner_customers = saturday_prediction / 2 - breakfast_customers - lunch_customers :=
by sorry

end NUMINAMATH_CALUDE_friday_dinner_customers_l2991_299145


namespace NUMINAMATH_CALUDE_student_count_last_year_l2991_299140

theorem student_count_last_year 
  (increase_rate : ℝ) 
  (current_count : ℕ) 
  (h1 : increase_rate = 0.2) 
  (h2 : current_count = 960) : 
  ℕ :=
  by
    -- Proof goes here
    sorry

#check student_count_last_year

end NUMINAMATH_CALUDE_student_count_last_year_l2991_299140


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l2991_299121

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (feet_per_step : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  let length_feet := length_steps * feet_per_step
  let width_feet := width_steps * feet_per_step
  let area_sqft := length_feet * width_feet
  (area_sqft : ℚ) * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden --/
theorem mr_green_potato_yield :
  expected_potato_yield 15 20 2 (1/2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l2991_299121


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l2991_299133

theorem quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l2991_299133


namespace NUMINAMATH_CALUDE_shape_has_four_sides_l2991_299146

/-- The shape being fenced -/
structure Shape where
  sides : ℕ
  cost_per_side : ℕ
  total_cost : ℕ

/-- The shape satisfies the given conditions -/
def satisfies_conditions (s : Shape) : Prop :=
  s.cost_per_side = 69 ∧ s.total_cost = 276 ∧ s.total_cost = s.cost_per_side * s.sides

theorem shape_has_four_sides (s : Shape) (h : satisfies_conditions s) : s.sides = 4 := by
  sorry

end NUMINAMATH_CALUDE_shape_has_four_sides_l2991_299146


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l2991_299117

theorem no_solution_iff_n_eq_neg_two (n : ℝ) :
  (∀ x y z : ℝ, (n * x + y + z = 2 ∧ x + n * y + z = 2 ∧ x + y + n * z = 2) → False) ↔ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_two_l2991_299117


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2991_299147

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (m n : ℝ) (h : m < 0 ∧ n > 0) :
  (∀ x y : ℝ, x^2 / m + y^2 / n = 1) →  -- Hyperbola equation
  (2 * Real.sqrt 3 / 3 : ℝ) = 2 / Real.sqrt n →  -- Eccentricity condition
  (∃ c : ℝ, c = 2 ∧ ∀ x y : ℝ, x^2 = 8*y → y = c/2) →  -- Shared focus with parabola
  (∀ x y : ℝ, y^2 / 3 - x^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2991_299147


namespace NUMINAMATH_CALUDE_x_value_l2991_299143

theorem x_value : ∃ x : ℝ, x = 12 * (1 + 0.2) ∧ x = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2991_299143


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2991_299141

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 23210 →
  b % 5 = 0 →
  a = 2 * (b / 10) →
  b - a = 15480 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2991_299141


namespace NUMINAMATH_CALUDE_distance_in_A_l2991_299195

/-- A positive three-digit integer -/
def ThreeDigitInt := { n : ℕ // 100 ≤ n ∧ n ≤ 999 }

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Set of positive three-digit integers with digit sum 16 -/
def A : Set ThreeDigitInt := { a | digit_sum a.val = 16 }

/-- The greatest distance between two numbers in set A -/
def max_distance : ℕ := sorry

/-- The smallest distance between two numbers in set A -/
def min_distance : ℕ := sorry

/-- Theorem stating the greatest and smallest distances in set A -/
theorem distance_in_A : max_distance = 801 ∧ min_distance = 9 := by sorry

end NUMINAMATH_CALUDE_distance_in_A_l2991_299195


namespace NUMINAMATH_CALUDE_sugar_profit_theorem_l2991_299182

/-- Represents the profit calculation for a sugar trader --/
def sugar_profit (total_quantity : ℝ) (quantity_at_unknown_profit : ℝ) (known_profit : ℝ) (overall_profit : ℝ) : Prop :=
  let quantity_at_known_profit := total_quantity - quantity_at_unknown_profit
  let unknown_profit := (overall_profit * total_quantity - known_profit * quantity_at_known_profit) / quantity_at_unknown_profit
  unknown_profit = 12

/-- Theorem stating the profit percentage on the rest of the sugar --/
theorem sugar_profit_theorem :
  sugar_profit 1600 1200 8 11 := by
  sorry

end NUMINAMATH_CALUDE_sugar_profit_theorem_l2991_299182


namespace NUMINAMATH_CALUDE_school_selection_probability_l2991_299144

theorem school_selection_probability :
  let total_schools : ℕ := 4
  let schools_to_select : ℕ := 2
  let total_combinations : ℕ := (total_schools.choose schools_to_select)
  let favorable_outcomes : ℕ := ((total_schools - 1).choose (schools_to_select - 1))
  favorable_outcomes / total_combinations = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_school_selection_probability_l2991_299144


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2991_299131

/-- Proves that mixing 175 gallons of 15% alcohol solution with 75 gallons of 35% alcohol solution 
    results in 250 gallons of 21% alcohol solution. -/
theorem alcohol_mixture_proof :
  let solution_1_volume : ℝ := 175
  let solution_1_concentration : ℝ := 0.15
  let solution_2_volume : ℝ := 75
  let solution_2_concentration : ℝ := 0.35
  let total_volume : ℝ := 250
  let final_concentration : ℝ := 0.21
  (solution_1_volume + solution_2_volume = total_volume) ∧
  (solution_1_volume * solution_1_concentration + solution_2_volume * solution_2_concentration = 
   total_volume * final_concentration) :=
by sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2991_299131


namespace NUMINAMATH_CALUDE_total_books_l2991_299109

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
sorry

end NUMINAMATH_CALUDE_total_books_l2991_299109


namespace NUMINAMATH_CALUDE_average_reading_time_emery_serena_l2991_299148

/-- The average reading time for two people, given one person's reading speed and time -/
def averageReadingTime (fasterReaderTime : ℕ) (speedRatio : ℕ) : ℚ :=
  (fasterReaderTime + fasterReaderTime * speedRatio) / 2

/-- Theorem: The average reading time for Emery and Serena is 60 days -/
theorem average_reading_time_emery_serena :
  averageReadingTime 20 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_reading_time_emery_serena_l2991_299148


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2991_299125

theorem solution_set_inequality (x : ℝ) :
  x * (x - 1) < 0 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2991_299125


namespace NUMINAMATH_CALUDE_stratified_sampling_senior_high_l2991_299167

theorem stratified_sampling_senior_high (total_students : ℕ) (senior_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1800)
  (h2 : senior_students = 600)
  (h3 : sample_size = 180) :
  (senior_students * sample_size) / total_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_senior_high_l2991_299167


namespace NUMINAMATH_CALUDE_tomato_land_area_l2991_299193

/-- Represents the farm land allocation -/
structure FarmLand where
  total : ℝ
  cleared_percentage : ℝ
  barley_percentage : ℝ
  potato_percentage : ℝ

/-- Calculates the area of land planted with tomato -/
def tomato_area (farm : FarmLand) : ℝ :=
  let cleared_land := farm.total * farm.cleared_percentage
  let barley_land := cleared_land * farm.barley_percentage
  let potato_land := cleared_land * farm.potato_percentage
  cleared_land - (barley_land + potato_land)

/-- Theorem stating the area of land planted with tomato -/
theorem tomato_land_area : 
  let farm := FarmLand.mk 1000 0.9 0.8 0.1
  tomato_area farm = 90 := by
  sorry


end NUMINAMATH_CALUDE_tomato_land_area_l2991_299193


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l2991_299175

theorem orange_cost_calculation (family_size : ℕ) (planned_spending : ℚ) (savings_percentage : ℚ) (oranges_received : ℕ) : 
  family_size = 4 → 
  planned_spending = 15 → 
  savings_percentage = 40 / 100 → 
  oranges_received = family_size →
  (planned_spending * savings_percentage) / oranges_received = 3/2 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_calculation_l2991_299175


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2991_299187

theorem quadratic_equation_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 6*m*x + 9*m^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 2*x₂ → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2991_299187


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l2991_299191

theorem smallest_prime_factor_of_2023 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2023 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2023 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l2991_299191


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_twenty_fourth_l2991_299178

theorem reciprocal_of_negative_one_twenty_fourth :
  ((-1 / 24)⁻¹ : ℚ) = -24 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_twenty_fourth_l2991_299178


namespace NUMINAMATH_CALUDE_square_of_105_l2991_299136

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l2991_299136


namespace NUMINAMATH_CALUDE_digit_79_is_2_l2991_299129

/-- The sequence of digits formed by writing consecutive integers from 65 to 1 in descending order -/
def descending_sequence : List Nat := sorry

/-- The 79th digit in the descending sequence -/
def digit_79 : Nat := sorry

/-- Theorem stating that the 79th digit in the descending sequence is 2 -/
theorem digit_79_is_2 : digit_79 = 2 := by sorry

end NUMINAMATH_CALUDE_digit_79_is_2_l2991_299129


namespace NUMINAMATH_CALUDE_cone_cube_distance_l2991_299137

/-- The distance between the vertex of a cone and the closest vertex of a cube placed inside it. -/
theorem cone_cube_distance (cube_edge : ℝ) (cone_diameter cone_height : ℝ) 
  (h_cube_edge : cube_edge = 3)
  (h_cone_diameter : cone_diameter = 8)
  (h_cone_height : cone_height = 24)
  (h_diagonal_coincide : ∃ (diagonal : ℝ), diagonal = cube_edge * Real.sqrt 3 ∧ 
    diagonal = cone_height * (cone_diameter / 8)) :
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 6 - Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_cone_cube_distance_l2991_299137


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2991_299105

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y, x < y ∧ y < 0 → x^2 > y^2) ∧ 
  (∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2991_299105


namespace NUMINAMATH_CALUDE_smallest_a_with_single_digit_sum_l2991_299123

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is single-digit -/
def is_single_digit (n : ℕ) : Prop := n < 10

/-- The property we want to prove -/
def has_single_digit_sum (a : ℕ) : Prop :=
  is_single_digit (sum_of_digits (10^a - 74))

theorem smallest_a_with_single_digit_sum :
  (∀ k < 2, ¬ has_single_digit_sum k) ∧ has_single_digit_sum 2 := by sorry

end NUMINAMATH_CALUDE_smallest_a_with_single_digit_sum_l2991_299123


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2991_299113

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2991_299113


namespace NUMINAMATH_CALUDE_pastries_sold_l2991_299155

theorem pastries_sold (cupcakes cookies left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : cookies = 5) 
  (h3 : left = 8) : 
  cupcakes + cookies - left = 4 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sold_l2991_299155


namespace NUMINAMATH_CALUDE_not_necessarily_true_squared_l2991_299122

theorem not_necessarily_true_squared (a b : ℝ) (h : a < b) : 
  ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_true_squared_l2991_299122


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l2991_299189

/-- Sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem :
  let a : ℚ := 1/3
  let r : ℚ := 1/4
  let n : ℕ := 7
  geometric_sum a r n = 16383/12288 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l2991_299189


namespace NUMINAMATH_CALUDE_two_red_or_blue_marbles_probability_l2991_299181

/-- The probability of drawing two marbles consecutively where both are either red or blue
    from a bag containing 5 red, 3 blue, and 7 yellow marbles, with replacement. -/
theorem two_red_or_blue_marbles_probability :
  let red_marbles : ℕ := 5
  let blue_marbles : ℕ := 3
  let yellow_marbles : ℕ := 7
  let total_marbles : ℕ := red_marbles + blue_marbles + yellow_marbles
  let prob_red_or_blue : ℚ := (red_marbles + blue_marbles : ℚ) / total_marbles
  (prob_red_or_blue * prob_red_or_blue) = 64 / 225 := by
  sorry

end NUMINAMATH_CALUDE_two_red_or_blue_marbles_probability_l2991_299181


namespace NUMINAMATH_CALUDE_profit_percentage_is_twenty_percent_l2991_299152

def robi_contribution : ℝ := 4000
def rudy_contribution : ℝ := 1.25 * robi_contribution
def total_contribution : ℝ := robi_contribution + rudy_contribution
def individual_profit : ℝ := 900
def total_profit : ℝ := 2 * individual_profit

theorem profit_percentage_is_twenty_percent :
  (total_profit / total_contribution) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_twenty_percent_l2991_299152


namespace NUMINAMATH_CALUDE_area_of_EFGH_l2991_299139

-- Define the rectangle and squares
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Square :=
  (side : ℝ)

-- Define the problem setup
def smallest_square : Square :=
  { side := 1 }

def rectangle_EFGH : Rectangle :=
  { width := 4, height := 3 }

-- Define the theorem
theorem area_of_EFGH :
  (rectangle_EFGH.width * rectangle_EFGH.height : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l2991_299139


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_equation_l2991_299134

/-- A triangle with sides a, b, and c is isosceles if two of its sides are equal. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- 
  Theorem: If the three sides a, b, c of a triangle ABC satisfy a²-ac-b²+bc=0, 
  then the triangle is isosceles.
-/
theorem triangle_isosceles_from_equation 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_eq : a^2 - a*c - b^2 + b*c = 0) : 
  IsIsosceles a b c := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_from_equation_l2991_299134


namespace NUMINAMATH_CALUDE_product_divisible_by_5184_l2991_299194

theorem product_divisible_by_5184 (k m : ℕ) : 
  5184 ∣ ((k^3 - 1) * k^3 * (k^3 + 1) * (m^3 - 1) * m^3 * (m^3 + 1)) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_5184_l2991_299194


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2991_299179

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_property
  (f : ℝ → ℝ)
  (h_quad : is_quadratic f)
  (h_pos : ∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c)
  (h_sym : ∀ x : ℝ, f x = f (4 - x))
  (h_ineq : ∀ a : ℝ, f (2 - a^2) < f (1 + a - a^2)) :
  ∀ a : ℝ, a < 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2991_299179


namespace NUMINAMATH_CALUDE_profit_sharing_l2991_299108

/-- The profit sharing problem -/
theorem profit_sharing
  (mary_investment mike_investment : ℚ)
  (equal_share_ratio investment_share_ratio : ℚ)
  (mary_extra : ℚ)
  (h1 : mary_investment = 650)
  (h2 : mike_investment = 350)
  (h3 : equal_share_ratio = 1/3)
  (h4 : investment_share_ratio = 2/3)
  (h5 : mary_extra = 600)
  : ∃ P : ℚ,
    P / 6 + (mary_investment / (mary_investment + mike_investment)) * (2 * P / 3) -
    (P / 6 + (mike_investment / (mary_investment + mike_investment)) * (2 * P / 3)) = mary_extra ∧
    P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_l2991_299108


namespace NUMINAMATH_CALUDE_expression_equality_l2991_299150

theorem expression_equality : 3 * Real.sqrt 2 + |1 - Real.sqrt 2| + (8 : ℝ) ^ (1/3) = 4 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2991_299150


namespace NUMINAMATH_CALUDE_z_max_min_difference_l2991_299183

theorem z_max_min_difference (x y z : ℝ) 
  (sum_eq : x + y + z = 5)
  (sum_squares_eq : x^2 + y^2 + z^2 = 20)
  (xy_eq : x * y = 2) : 
  ∃ (z_max z_min : ℝ), 
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 20 ∧ x' * y' = 2) → z' ≤ z_max) ∧
    (∀ z', (∃ x' y', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 20 ∧ x' * y' = 2) → z' ≥ z_min) ∧
    z_max - z_min = 6 :=
by sorry

end NUMINAMATH_CALUDE_z_max_min_difference_l2991_299183


namespace NUMINAMATH_CALUDE_club_membership_l2991_299138

theorem club_membership (total_members men : ℕ) (h1 : total_members = 52) (h2 : men = 37) :
  total_members - men = 15 := by
  sorry

end NUMINAMATH_CALUDE_club_membership_l2991_299138


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l2991_299165

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def M : ℚ := (factorial 2 * factorial 19) * (
  1 / (factorial 3 * factorial 18) +
  1 / (factorial 4 * factorial 17) +
  1 / (factorial 5 * factorial 16) +
  1 / (factorial 6 * factorial 15) +
  1 / (factorial 7 * factorial 14) +
  1 / (factorial 8 * factorial 13) +
  1 / (factorial 9 * factorial 12) +
  1 / (factorial 10 * factorial 11)
)

theorem greatest_integer_less_than_M_over_100 : 
  ⌊M / 100⌋ = 49 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l2991_299165


namespace NUMINAMATH_CALUDE_division_by_three_remainder_l2991_299112

theorem division_by_three_remainder (n : ℤ) : 
  (n % 3 ≠ 0) → (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_division_by_three_remainder_l2991_299112


namespace NUMINAMATH_CALUDE_specific_weekly_profit_l2991_299130

/-- Represents a business owner's financial situation --/
structure BusinessOwner where
  daily_earnings : ℕ
  weekly_rent : ℕ

/-- Calculates the weekly profit for a business owner --/
def weekly_profit (owner : BusinessOwner) : ℕ :=
  owner.daily_earnings * 7 - owner.weekly_rent

/-- Theorem stating that a business owner with specific earnings and rent has a weekly profit of $36 --/
theorem specific_weekly_profit :
  ∀ (owner : BusinessOwner),
    owner.daily_earnings = 8 →
    owner.weekly_rent = 20 →
    weekly_profit owner = 36 := by
  sorry

#eval weekly_profit { daily_earnings := 8, weekly_rent := 20 }

end NUMINAMATH_CALUDE_specific_weekly_profit_l2991_299130


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2991_299103

/-- Given a rectangle formed by 2 rows and 3 columns of identical squares with a total area of 150 cm²,
    prove that its perimeter is 50 cm. -/
theorem rectangle_perimeter (total_area : ℝ) (num_squares : ℕ) (rows cols : ℕ) :
  total_area = 150 ∧ 
  num_squares = 6 ∧ 
  rows = 2 ∧ 
  cols = 3 →
  (2 * rows + 2 * cols) * Real.sqrt (total_area / num_squares) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2991_299103


namespace NUMINAMATH_CALUDE_chord_diameter_relationship_l2991_299110

/-- Represents a sphere with a chord and diameter -/
structure SphereWithChord where
  /-- The radius of the sphere -/
  radius : ℝ
  /-- The length of the chord AB -/
  chord_length : ℝ
  /-- The angle between the chord AB and the diameter CD -/
  angle : ℝ
  /-- The distance from C to A -/
  distance_CA : ℝ

/-- Theorem stating the relationship between the given conditions and BD -/
theorem chord_diameter_relationship (s : SphereWithChord) 
  (h1 : s.radius = 1)
  (h2 : s.chord_length = 1)
  (h3 : s.angle = Real.pi / 3)  -- 60 degrees in radians
  (h4 : s.distance_CA = Real.sqrt 2) :
  ∃ (BD : ℝ), BD = 1 := by
  sorry


end NUMINAMATH_CALUDE_chord_diameter_relationship_l2991_299110


namespace NUMINAMATH_CALUDE_sum_of_roots_l2991_299174

theorem sum_of_roots (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α - 17 = 0)
  (h2 : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2991_299174


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2991_299156

theorem cos_sum_of_complex_exponentials (α β : ℝ) 
  (h1 : Complex.exp (α * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I)
  (h2 : Complex.exp (β * Complex.I) = -(5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I) :
  Real.cos (α + β) = -(7 / 13) := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l2991_299156
