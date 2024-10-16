import Mathlib

namespace NUMINAMATH_CALUDE_prime_digits_imply_prime_count_l457_45743

theorem prime_digits_imply_prime_count (n : ℕ) (x : ℕ) : 
  (x = (10^n - 1) / 9) →  -- x is an integer with n digits, all equal to 1
  Nat.Prime x →           -- x is prime
  Nat.Prime n :=          -- n is prime
by sorry

end NUMINAMATH_CALUDE_prime_digits_imply_prime_count_l457_45743


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l457_45783

theorem two_digit_number_puzzle (a b : ℕ) : 
  a < 10 → b < 10 → a ≠ 0 →
  (10 * a + b) - (10 * b + a) = 36 →
  2 * a = b →
  (a + b) - (a - b) = 16 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l457_45783


namespace NUMINAMATH_CALUDE_polynomial_factor_l457_45782

/-- The polynomial with parameters a and b -/
def P (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + 48 * x^2 - 24 * x + 4

/-- The factor of the polynomial -/
def F (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 1

/-- Theorem stating that the polynomial P has the factor F when a = -16 and b = -36 -/
theorem polynomial_factor (x : ℝ) : ∃ (Q : ℝ → ℝ), P (-16) (-36) x = F x * Q x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l457_45782


namespace NUMINAMATH_CALUDE_area_rectangle_circumscribing_right_triangle_l457_45720

/-- The area of a rectangle circumscribing a right triangle with legs of length 5 and 6 is 30. -/
theorem area_rectangle_circumscribing_right_triangle : 
  ∀ (A B C D E : ℝ × ℝ),
    -- Right triangle ABC
    (B.1 - A.1) * (C.2 - B.2) = (C.1 - B.1) * (B.2 - A.2) →
    -- AB = 5
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 →
    -- BC = 6
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 →
    -- Rectangle ADEC circumscribes triangle ABC
    A.1 = D.1 ∧ A.2 = D.2 ∧
    C.1 = E.1 ∧ C.2 = E.2 ∧
    D.2 = E.2 ∧ A.1 = C.1 →
    -- Area of rectangle ADEC is 30
    (E.1 - D.1) * (E.2 - D.2) = 30 := by
  sorry


end NUMINAMATH_CALUDE_area_rectangle_circumscribing_right_triangle_l457_45720


namespace NUMINAMATH_CALUDE_gravel_path_cost_l457_45707

/-- Calculate the cost of gravelling a path around a rectangular plot -/
theorem gravel_path_cost 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm_paise : ℝ) : 
  plot_length = 150 ∧ 
  plot_width = 95 ∧ 
  path_width = 4.5 ∧ 
  cost_per_sqm_paise = 90 → 
  (((plot_length + 2 * path_width) * (plot_width + 2 * path_width) - 
    plot_length * plot_width) * 
   (cost_per_sqm_paise / 100)) = 2057.40 :=
by sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l457_45707


namespace NUMINAMATH_CALUDE_composite_and_infinite_x_l457_45708

theorem composite_and_infinite_x (a : ℕ) :
  (∃ x : ℕ, ∃ y z : ℕ, y > 1 ∧ z > 1 ∧ a * x + 1 = y * z) ∧
  (∀ n : ℕ, ∃ x : ℕ, x > n ∧ ∃ y z : ℕ, y > 1 ∧ z > 1 ∧ a * x + 1 = y * z) :=
by sorry

end NUMINAMATH_CALUDE_composite_and_infinite_x_l457_45708


namespace NUMINAMATH_CALUDE_circle_polar_equation_l457_45769

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  a : ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a given polar equation represents the specified circle -/
def is_correct_polar_equation (circle : PolarCircle) (equation : ℝ → ℝ → Prop) : Prop :=
  circle.center = (circle.a / 2, Real.pi / 2) ∧
  circle.radius = circle.a / 2 ∧
  ∀ θ ρ, equation ρ θ ↔ ρ = circle.a * Real.sin θ

theorem circle_polar_equation (a : ℝ) (h : a > 0) :
  let circle : PolarCircle := ⟨a, (a / 2, Real.pi / 2), a / 2⟩
  is_correct_polar_equation circle (fun ρ θ ↦ ρ = a * Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l457_45769


namespace NUMINAMATH_CALUDE_prime_sum_divisible_by_six_l457_45701

theorem prime_sum_divisible_by_six (p q r : Nat) : 
  Prime p → Prime q → Prime r → p > 3 → q > 3 → r > 3 → Prime (p + q + r) → 
  (6 ∣ p + q) ∨ (6 ∣ p + r) ∨ (6 ∣ q + r) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divisible_by_six_l457_45701


namespace NUMINAMATH_CALUDE_solution_to_equation_l457_45788

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^5 = (14 * x)^4 ↔ x = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l457_45788


namespace NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpos_l457_45760

theorem abs_eq_neg_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpos_l457_45760


namespace NUMINAMATH_CALUDE_task_completion_probability_l457_45762

theorem task_completion_probability (p1 p2 p3 : ℚ) 
  (h1 : p1 = 2/3) (h2 : p2 = 3/5) (h3 : p3 = 4/7) :
  p1 * (1 - p2) * p3 = 16/105 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l457_45762


namespace NUMINAMATH_CALUDE_equation_solutions_l457_45727

def equation (x : ℝ) : Prop :=
  1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
  1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 10

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 10 ∨ x = -3.5 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l457_45727


namespace NUMINAMATH_CALUDE_tuesday_occurs_five_times_in_august_l457_45790

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- July of year N -/
def july : Month := sorry

/-- August of year N -/
def august : Month := sorry

/-- Counts the occurrences of a specific day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat := sorry

theorem tuesday_occurs_five_times_in_august 
  (h1 : july.days = 31)
  (h2 : countDayOccurrences july DayOfWeek.Monday = 5)
  (h3 : august.days = 30) :
  countDayOccurrences august DayOfWeek.Tuesday = 5 := by sorry

end NUMINAMATH_CALUDE_tuesday_occurs_five_times_in_august_l457_45790


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l457_45744

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_fifth : a 5 = 8)
  (h_sum : a 1 + a 2 + a 3 = 6) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l457_45744


namespace NUMINAMATH_CALUDE_problem_statement_l457_45729

theorem problem_statement (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*z/(y+z) + x*y/(z+x) = -10)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 15) :
  y/(x+y) + z/(y+z) + x/(z+x) = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l457_45729


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l457_45715

-- Define Pascal's triangle
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

-- Theorem statement
theorem fifth_element_row_20 : pascal 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l457_45715


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l457_45791

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I) / z = I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l457_45791


namespace NUMINAMATH_CALUDE_election_winner_votes_l457_45728

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  (winner_percentage = 62 / 100) →
  (winner_percentage * total_votes - (1 - winner_percentage) * total_votes = vote_difference) →
  (vote_difference = 300) →
  (winner_percentage * total_votes = 775) := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l457_45728


namespace NUMINAMATH_CALUDE_sum_of_fractions_geq_six_l457_45710

theorem sum_of_fractions_geq_six (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z + z / y + y / x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_geq_six_l457_45710


namespace NUMINAMATH_CALUDE_current_speed_l457_45709

/-- Proves that the speed of the current is approximately 3 km/hr given the conditions -/
theorem current_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) :
  rowing_speed = 6 →
  distance = 80 →
  time = 31.99744020478362 →
  ∃ (current_speed : ℝ), 
    (abs (current_speed - 3) < 0.001) ∧ 
    (distance / time = rowing_speed / 3.6 + current_speed / 3.6) := by
  sorry


end NUMINAMATH_CALUDE_current_speed_l457_45709


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l457_45779

theorem modular_inverse_17_mod_800 : ∃ x : ℕ, x < 800 ∧ (17 * x) % 800 = 1 :=
by
  use 753
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_800_l457_45779


namespace NUMINAMATH_CALUDE_initial_cats_is_28_l457_45767

/-- Represents the animal shelter scenario --/
structure AnimalShelter where
  initialDogs : ℕ
  initialLizards : ℕ
  dogAdoptionRate : ℚ
  catAdoptionRate : ℚ
  lizardAdoptionRate : ℚ
  newPetsPerMonth : ℕ
  totalPetsAfterOneMonth : ℕ

/-- Calculates the initial number of cats in the shelter --/
def calculateInitialCats (shelter : AnimalShelter) : ℚ :=
  let remainingDogs : ℚ := shelter.initialDogs * (1 - shelter.dogAdoptionRate)
  let remainingLizards : ℚ := shelter.initialLizards * (1 - shelter.lizardAdoptionRate)
  let nonCatPets : ℚ := remainingDogs + remainingLizards + shelter.newPetsPerMonth
  let remainingCats : ℚ := shelter.totalPetsAfterOneMonth - nonCatPets
  remainingCats / (1 - shelter.catAdoptionRate)

/-- Theorem stating that the initial number of cats is 28 --/
theorem initial_cats_is_28 (shelter : AnimalShelter) 
  (h1 : shelter.initialDogs = 30)
  (h2 : shelter.initialLizards = 20)
  (h3 : shelter.dogAdoptionRate = 1/2)
  (h4 : shelter.catAdoptionRate = 1/4)
  (h5 : shelter.lizardAdoptionRate = 1/5)
  (h6 : shelter.newPetsPerMonth = 13)
  (h7 : shelter.totalPetsAfterOneMonth = 65) :
  calculateInitialCats shelter = 28 := by
  sorry

#eval calculateInitialCats {
  initialDogs := 30,
  initialLizards := 20,
  dogAdoptionRate := 1/2,
  catAdoptionRate := 1/4,
  lizardAdoptionRate := 1/5,
  newPetsPerMonth := 13,
  totalPetsAfterOneMonth := 65
}

end NUMINAMATH_CALUDE_initial_cats_is_28_l457_45767


namespace NUMINAMATH_CALUDE_coffee_fraction_is_37_84_l457_45765

-- Define the initial conditions
def initial_coffee : ℚ := 5
def initial_cream : ℚ := 7
def cup_size : ℚ := 10

-- Define the transfers
def first_transfer : ℚ := 2
def second_transfer : ℚ := 3
def third_transfer : ℚ := 1

-- Define the function to calculate the final fraction of coffee in cup 1
def final_coffee_fraction (ic : ℚ) (icr : ℚ) (cs : ℚ) (ft : ℚ) (st : ℚ) (tt : ℚ) : ℚ :=
  let coffee_after_first := ic - ft
  let total_after_first := coffee_after_first + icr + ft
  let coffee_ratio_second := ft / total_after_first
  let coffee_returned := st * coffee_ratio_second
  let total_after_second := coffee_after_first + coffee_returned + st * (1 - coffee_ratio_second)
  let coffee_after_second := coffee_after_first + coffee_returned
  let coffee_ratio_third := coffee_after_second / total_after_second
  let coffee_final := coffee_after_second - tt * coffee_ratio_third
  let total_final := total_after_second - tt
  coffee_final / total_final

-- Theorem statement
theorem coffee_fraction_is_37_84 :
  final_coffee_fraction initial_coffee initial_cream cup_size first_transfer second_transfer third_transfer = 37 / 84 := by
  sorry

end NUMINAMATH_CALUDE_coffee_fraction_is_37_84_l457_45765


namespace NUMINAMATH_CALUDE_floor_width_is_twenty_l457_45786

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem floor_width_is_twenty
  (floor : FloorWithRug)
  (h1 : floor.length = 25)
  (h2 : floor.strip_width = 4)
  (h3 : floor.rug_area = 204)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.width = 20 := by
  sorry

#check floor_width_is_twenty

end NUMINAMATH_CALUDE_floor_width_is_twenty_l457_45786


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l457_45712

/-- Triangle XYZ with side lengths -/
structure Triangle where
  xy : ℝ
  yz : ℝ
  xz : ℝ

/-- Rectangle MNPQ inscribed in Triangle XYZ -/
structure InscribedRectangle where
  triangle : Triangle
  ω : ℝ  -- side length MN

/-- Area of rectangle MNPQ as a function of ω -/
def rectangleArea (rect : InscribedRectangle) : ℝ → ℝ :=
  fun ω => a * ω - b * ω^2
  where
    a : ℝ := sorry
    b : ℝ := sorry

/-- Theorem statement -/
theorem inscribed_rectangle_area_coefficient
  (t : Triangle)
  (h1 : t.xy = 15)
  (h2 : t.yz = 20)
  (h3 : t.xz = 13) :
  ∃ (rect : InscribedRectangle),
    rect.triangle = t ∧
    ∃ (a b : ℝ),
      (∀ ω, rectangleArea rect ω = a * ω - b * ω^2) ∧
      b = 9 / 25 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l457_45712


namespace NUMINAMATH_CALUDE_katie_cookies_l457_45706

def pastry_sale (cupcakes sold leftover : ℕ) : Prop :=
  ∃ (cookies total : ℕ),
    total = sold + leftover ∧
    total = cupcakes + cookies ∧
    cupcakes = 7 ∧
    sold = 4 ∧
    leftover = 8 ∧
    cookies = 5

theorem katie_cookies : pastry_sale 7 4 8 := by
  sorry

end NUMINAMATH_CALUDE_katie_cookies_l457_45706


namespace NUMINAMATH_CALUDE_area_inequality_special_quadrilateral_l457_45758

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- Check if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculate the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Check if a point is on a line segment between two other points -/
def isOnSegment (p : Point) (a b : Point) : Prop := sorry

/-- Check if four points form a parallelogram -/
def isParallelogram (a b c d : Point) : Prop := sorry

/-- Theorem: Area inequality for quadrilaterals with special interior point -/
theorem area_inequality_special_quadrilateral 
  (ABCD : Quadrilateral) 
  (O K L M N : Point) 
  (h_convex : isConvex ABCD)
  (h_inside : isInside O ABCD)
  (h_K : isOnSegment K ABCD.A ABCD.B)
  (h_L : isOnSegment L ABCD.B ABCD.C)
  (h_M : isOnSegment M ABCD.C ABCD.D)
  (h_N : isOnSegment N ABCD.D ABCD.A)
  (h_OKBL : isParallelogram O K ABCD.B L)
  (h_OMDN : isParallelogram O M ABCD.D N)
  (S := area ABCD)
  (S1 := area (Quadrilateral.mk O N ABCD.A K))
  (S2 := area (Quadrilateral.mk O L ABCD.C M)) :
  Real.sqrt S ≥ Real.sqrt S1 + Real.sqrt S2 := by
  sorry

end NUMINAMATH_CALUDE_area_inequality_special_quadrilateral_l457_45758


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l457_45794

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 8*x + 6 = 0 ↔ (x - 4)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l457_45794


namespace NUMINAMATH_CALUDE_prob_B_wins_third_round_correct_A_has_lower_expected_additional_time_l457_45754

-- Define the probabilities of answering correctly for each participant
def prob_correct_A : ℚ := 3/5
def prob_correct_B : ℚ := 2/3

-- Define the number of rounds and questions per round
def num_rounds : ℕ := 3
def questions_per_round : ℕ := 3

-- Define the time penalty for an incorrect answer
def time_penalty : ℕ := 20

-- Define the time difference in recitation per round
def recitation_time_diff : ℕ := 10

-- Define the function to calculate the probability of B winning in the third round
def prob_B_wins_third_round : ℚ := 448/3375

-- Define the expected number of incorrect answers for each participant
def expected_incorrect_A : ℚ := (1 - prob_correct_A) * (num_rounds * questions_per_round : ℚ)
def expected_incorrect_B : ℚ := (1 - prob_correct_B) * (num_rounds * questions_per_round : ℚ)

-- Theorem: The probability of B winning in the third round is 448/3375
theorem prob_B_wins_third_round_correct :
  prob_B_wins_third_round = 448/3375 := by sorry

-- Theorem: A has a lower expected additional time due to incorrect answers
theorem A_has_lower_expected_additional_time :
  expected_incorrect_A * time_penalty < expected_incorrect_B * time_penalty + (num_rounds * recitation_time_diff : ℚ) := by sorry

end NUMINAMATH_CALUDE_prob_B_wins_third_round_correct_A_has_lower_expected_additional_time_l457_45754


namespace NUMINAMATH_CALUDE_power_function_odd_l457_45799

def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ n : ℤ, ∀ x : ℝ, f x = x ^ n

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem power_function_odd (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 1 = 3) :
  isOddFunction f := by
  sorry

end NUMINAMATH_CALUDE_power_function_odd_l457_45799


namespace NUMINAMATH_CALUDE_keyboard_warrior_estimate_l457_45770

theorem keyboard_warrior_estimate (total_population : ℕ) (sample_size : ℕ) (favorable_count : ℕ) 
  (h1 : total_population = 9600)
  (h2 : sample_size = 50)
  (h3 : favorable_count = 15) :
  (total_population : ℚ) * (1 - favorable_count / sample_size) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_keyboard_warrior_estimate_l457_45770


namespace NUMINAMATH_CALUDE_nabla_square_l457_45787

theorem nabla_square (odot nabla : ℕ) : 
  odot ≠ nabla → 
  0 < odot → odot < 20 → 
  0 < nabla → nabla < 20 → 
  nabla * nabla * nabla = nabla →
  nabla * nabla = 64 := by
sorry

end NUMINAMATH_CALUDE_nabla_square_l457_45787


namespace NUMINAMATH_CALUDE_some_base_value_l457_45740

theorem some_base_value (k : ℕ) (some_base : ℝ) 
  (h1 : (1/2)^16 * (1/some_base)^k = 1/(18^16))
  (h2 : k = 8) : 
  some_base = 81 := by
sorry

end NUMINAMATH_CALUDE_some_base_value_l457_45740


namespace NUMINAMATH_CALUDE_fuel_capacity_ratio_l457_45702

theorem fuel_capacity_ratio (original_cost : ℝ) (price_increase : ℝ) (new_cost : ℝ) :
  original_cost = 200 →
  price_increase = 0.2 →
  new_cost = 480 →
  (new_cost / (original_cost * (1 + price_increase))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_fuel_capacity_ratio_l457_45702


namespace NUMINAMATH_CALUDE_thirteenth_term_of_arithmetic_sequence_l457_45766

/-- An arithmetic sequence is defined by its third and twenty-third terms -/
def arithmetic_sequence (a₃ a₂₃ : ℚ) :=
  ∃ (a : ℕ → ℚ), (∀ n m, a (n + 1) - a n = a (m + 1) - a m) ∧ a 3 = a₃ ∧ a 23 = a₂₃

/-- The thirteenth term of the sequence is the average of the third and twenty-third terms -/
theorem thirteenth_term_of_arithmetic_sequence 
  (h : arithmetic_sequence (2/11) (3/7)) : 
  ∃ (a : ℕ → ℚ), a 13 = 47/154 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_term_of_arithmetic_sequence_l457_45766


namespace NUMINAMATH_CALUDE_longest_non_decreasing_subsequence_12022_l457_45755

/-- Represents a natural number as a list of its digits. -/
def digits_of (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 10) :: aux (m / 10)
  (aux n).reverse

/-- Computes the length of the longest non-decreasing subsequence in a list. -/
def longest_non_decreasing_subsequence_length (l : List ℕ) : ℕ :=
  let rec aux (prev : ℕ) (current : List ℕ) (acc : ℕ) : ℕ :=
    match current with
    | [] => acc
    | x :: xs => if x ≥ prev then aux x xs (acc + 1) else aux prev xs acc
  aux 0 l 0

/-- The theorem stating that the longest non-decreasing subsequence of digits in 12022 has length 3. -/
theorem longest_non_decreasing_subsequence_12022 :
  longest_non_decreasing_subsequence_length (digits_of 12022) = 3 := by
  sorry

end NUMINAMATH_CALUDE_longest_non_decreasing_subsequence_12022_l457_45755


namespace NUMINAMATH_CALUDE_circle_equation_l457_45795

/-- The equation of a circle with center (-2, 1) passing through the point (2, -2) -/
theorem circle_equation :
  let center : ℝ × ℝ := (-2, 1)
  let point : ℝ × ℝ := (2, -2)
  ∀ x y : ℝ,
  (x - center.1)^2 + (y - center.2)^2 = (point.1 - center.1)^2 + (point.2 - center.2)^2 ↔
  (x + 2)^2 + (y - 1)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l457_45795


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l457_45713

/-- A circle with two parallel tangents and a third connecting tangent -/
structure TangentCircle where
  -- Radius of the circle
  r : ℝ
  -- Length of the first parallel tangent
  ab : ℝ
  -- Length of the second parallel tangent
  cd : ℝ
  -- Length of the connecting tangent
  ef : ℝ
  -- Condition that ab and cd are parallel tangents
  h_parallel : ab < cd
  -- Condition that ef is a tangent connecting ab and cd
  h_connecting : ef > ab ∧ ef < cd

/-- The theorem stating that for the given configuration, the radius is 2.5 -/
theorem tangent_circle_radius (c : TangentCircle)
    (h_ab : c.ab = 5)
    (h_cd : c.cd = 11)
    (h_ef : c.ef = 15) :
    c.r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l457_45713


namespace NUMINAMATH_CALUDE_proportion_fourth_term_l457_45759

theorem proportion_fourth_term (x y : ℝ) : 
  (0.6 : ℝ) / x = 5 / y → x = 0.96 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_proportion_fourth_term_l457_45759


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l457_45781

/-- The number of perfect square factors of 345600 -/
def perfectSquareFactors : ℕ := 16

/-- The prime factorization of 345600 -/
def n : ℕ := 2^6 * 3^3 * 5^2

/-- A function that counts the number of perfect square factors of n -/
def countPerfectSquareFactors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors :
  countPerfectSquareFactors n = perfectSquareFactors := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l457_45781


namespace NUMINAMATH_CALUDE_first_group_size_l457_45742

/-- The number of men in the first group -/
def first_group_men : ℕ := 12

/-- The number of acres the first group can reap -/
def first_group_acres : ℝ := 120

/-- The number of days the first group works -/
def first_group_days : ℕ := 36

/-- The number of men in the second group -/
def second_group_men : ℕ := 24

/-- The number of acres the second group can reap -/
def second_group_acres : ℝ := 413.33333333333337

/-- The number of days the second group works -/
def second_group_days : ℕ := 62

theorem first_group_size :
  (first_group_men : ℝ) * first_group_days * second_group_acres =
  second_group_men * second_group_days * first_group_acres :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l457_45742


namespace NUMINAMATH_CALUDE_lcm_18_35_l457_45745

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l457_45745


namespace NUMINAMATH_CALUDE_selling_price_is_200_l457_45717

/-- Calculates the selling price per acre given the initial purchase details and profit --/
def selling_price_per_acre (total_acres : ℕ) (purchase_price_per_acre : ℕ) (profit : ℕ) : ℕ :=
  let total_cost := total_acres * purchase_price_per_acre
  let acres_sold := total_acres / 2
  let total_revenue := total_cost + profit
  total_revenue / acres_sold

/-- Proves that the selling price per acre is $200 given the problem conditions --/
theorem selling_price_is_200 :
  selling_price_per_acre 200 70 6000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_200_l457_45717


namespace NUMINAMATH_CALUDE_sqrt_four_plus_abs_sqrt_three_minus_two_l457_45722

theorem sqrt_four_plus_abs_sqrt_three_minus_two :
  Real.sqrt 4 + |Real.sqrt 3 - 2| = 4 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_plus_abs_sqrt_three_minus_two_l457_45722


namespace NUMINAMATH_CALUDE_inverse_of_f_l457_45776

def f (x : ℝ) : ℝ := 3 - 4 * x

theorem inverse_of_f :
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x) ∧ (∀ x, g x = (3 - x) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_of_f_l457_45776


namespace NUMINAMATH_CALUDE_total_votes_l457_45747

/-- Given that Ben and Matt received votes in the ratio 2:3 and Ben got 24 votes,
    prove that the total number of votes cast is 60. -/
theorem total_votes (ben_votes : ℕ) (matt_votes : ℕ) : 
  ben_votes = 24 → 
  ben_votes * 3 = matt_votes * 2 → 
  ben_votes + matt_votes = 60 := by
sorry

end NUMINAMATH_CALUDE_total_votes_l457_45747


namespace NUMINAMATH_CALUDE_factorial_340_trailing_zeros_l457_45775

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 340! ends with 83 zeros -/
theorem factorial_340_trailing_zeros :
  trailingZeros 340 = 83 := by
  sorry

end NUMINAMATH_CALUDE_factorial_340_trailing_zeros_l457_45775


namespace NUMINAMATH_CALUDE_race_head_start_l457_45778

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (21 / 19) * Vb) :
  ∃ H : ℝ, H = (2 / 21) * L ∧ L / Va = (L - H) / Vb :=
sorry

end NUMINAMATH_CALUDE_race_head_start_l457_45778


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l457_45705

theorem solve_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 2 * y = 11 ∧ x + 3 * y = 12 ∧ x = 57 / 11 ∧ y = 25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l457_45705


namespace NUMINAMATH_CALUDE_system_solution_l457_45716

theorem system_solution :
  let S := {(x, y, z, t) : ℕ × ℕ × ℕ × ℕ | 
    x + y + z + t = 5 ∧ 
    x + 2*y + 5*z + 10*t = 17}
  S = {(1, 3, 0, 1), (2, 0, 3, 0)} := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l457_45716


namespace NUMINAMATH_CALUDE_inverse_relation_l457_45789

theorem inverse_relation (x : ℝ) (h : 1 / x = 40) : x = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_inverse_relation_l457_45789


namespace NUMINAMATH_CALUDE_center_coordinate_sum_l457_45718

/-- Given two points that are endpoints of a diameter of a circle,
    prove that the sum of the coordinates of the center is -3. -/
theorem center_coordinate_sum (p1 p2 : ℝ × ℝ) : 
  p1 = (5, -7) → p2 = (-7, 3) → 
  (∃ (c : ℝ × ℝ), c.1 + c.2 = -3 ∧ 
    c = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_center_coordinate_sum_l457_45718


namespace NUMINAMATH_CALUDE_jerry_cases_jerry_cases_proof_l457_45731

/-- The number of cases Jerry has, given the following conditions:
  - Each case has 3 shelves
  - Each shelf can hold 20 records
  - Each vinyl record has 60 ridges
  - The shelves are 60% full
  - There are 8640 ridges on all records
-/
theorem jerry_cases : ℕ :=
  let shelves_per_case : ℕ := 3
  let records_per_shelf : ℕ := 20
  let ridges_per_record : ℕ := 60
  let shelf_fullness : ℚ := 3/5
  let total_ridges : ℕ := 8640
  
  4

/-- Proof that Jerry has 4 cases -/
theorem jerry_cases_proof : jerry_cases = 4 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cases_jerry_cases_proof_l457_45731


namespace NUMINAMATH_CALUDE_binomial_12_6_l457_45711

theorem binomial_12_6 : Nat.choose 12 6 = 924 := by sorry

end NUMINAMATH_CALUDE_binomial_12_6_l457_45711


namespace NUMINAMATH_CALUDE_intersection_sum_l457_45764

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 4*x + 3
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Prop := ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
  f x₁ = y₁ ∧ g x₁ y₁ ∧
  f x₂ = y₂ ∧ g x₂ y₂ ∧
  f x₃ = y₃ ∧ g x₃ y₃

-- Theorem statement
theorem intersection_sum : intersection_points →
  ∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ,
    f x₁ = y₁ ∧ g x₁ y₁ ∧
    f x₂ = y₂ ∧ g x₂ y₂ ∧
    f x₃ = y₃ ∧ g x₃ y₃ ∧
    x₁ + x₂ + x₃ = 0 ∧
    y₁ + y₂ + y₃ = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l457_45764


namespace NUMINAMATH_CALUDE_tank_capacities_l457_45738

theorem tank_capacities (x y z : ℚ) : 
  x + y + z = 1620 →
  z = x + (1/5) * y →
  z = y + (1/3) * x →
  x = 540 ∧ y = 450 ∧ z = 630 := by
sorry

end NUMINAMATH_CALUDE_tank_capacities_l457_45738


namespace NUMINAMATH_CALUDE_lemonade_price_ratio_l457_45746

theorem lemonade_price_ratio :
  -- Define the ratio of small cups sold
  let small_ratio : ℚ := 3/5
  -- Define the ratio of large cups sold
  let large_ratio : ℚ := 1 - small_ratio
  -- Define the fraction of revenue from large cups
  let large_revenue_fraction : ℚ := 357142857142857150 / 1000000000000000000
  -- Define the price ratio of large to small cups
  let price_ratio : ℚ := large_revenue_fraction * (1 / large_ratio)
  -- The theorem
  price_ratio = 892857142857143 / 1000000000000000 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_price_ratio_l457_45746


namespace NUMINAMATH_CALUDE_third_term_is_64_l457_45733

/-- A geometric sequence with positive integer terms -/
structure GeometricSequence where
  terms : ℕ → ℕ
  first_term : terms 1 = 4
  is_geometric : ∀ n : ℕ, n > 0 → ∃ r : ℚ, terms (n + 1) = (terms n : ℚ) * r

/-- The theorem stating that for a geometric sequence with first term 4 and fourth term 256, the third term is 64 -/
theorem third_term_is_64 (seq : GeometricSequence) (h : seq.terms 4 = 256) : seq.terms 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_64_l457_45733


namespace NUMINAMATH_CALUDE_floor_abs_negative_l457_45703

theorem floor_abs_negative : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l457_45703


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_l457_45771

theorem smallest_k_with_remainder (k : ℕ) : k = 135 ↔ 
  (k > 1) ∧ 
  (∃ a : ℕ, k = 11 * a + 3) ∧ 
  (∃ b : ℕ, k = 4 * b + 3) ∧ 
  (∃ c : ℕ, k = 3 * c + 3) ∧ 
  (∀ m : ℕ, m > 1 → 
    ((∃ x : ℕ, m = 11 * x + 3) ∧ 
     (∃ y : ℕ, m = 4 * y + 3) ∧ 
     (∃ z : ℕ, m = 3 * z + 3)) → 
    m ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_l457_45771


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l457_45724

theorem sqrt_sum_squares_geq_sqrt2_sum (a b : ℝ) : 
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_geq_sqrt2_sum_l457_45724


namespace NUMINAMATH_CALUDE_find_b_l457_45785

def is_valid_set (x b : ℕ) : Prop :=
  x > 0 ∧ x + 2 > 0 ∧ x + b > 0 ∧ x + 7 > 0 ∧ x + 32 > 0

def median (x b : ℕ) : ℚ := x + b

def mean (x b : ℕ) : ℚ := (x + (x + 2) + (x + b) + (x + 7) + (x + 32)) / 5

theorem find_b (x : ℕ) :
  ∃ b : ℕ, is_valid_set x b ∧ mean x b = median x b + 5 → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l457_45785


namespace NUMINAMATH_CALUDE_remainder_9387_div_11_l457_45719

theorem remainder_9387_div_11 : 9387 % 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9387_div_11_l457_45719


namespace NUMINAMATH_CALUDE_problem_solution_l457_45714

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : ((x * y) / 7)^3 = x^2) (h2 : ((x * y) / 7)^3 = y^3) : 
  x = 7 ∧ y = 7^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l457_45714


namespace NUMINAMATH_CALUDE_triangle_side_range_l457_45748

-- Define an acute-angled triangle with side lengths 2, 4, and x
def is_acute_triangle (x : ℝ) : Prop :=
  0 < x ∧ x < 2 + 4 ∧ 2 < 4 + x ∧ 4 < 2 + x ∧
  (2^2 + 4^2 > x^2) ∧ (2^2 + x^2 > 4^2) ∧ (4^2 + x^2 > 2^2)

-- Theorem statement
theorem triangle_side_range :
  ∀ x : ℝ, is_acute_triangle x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l457_45748


namespace NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l457_45750

/-- A hyperbola with given properties -/
structure Hyperbola where
  -- Point A lies on the hyperbola
  point_A : ℝ × ℝ
  point_A_on_hyperbola : point_A.1^2 / 4 - point_A.2^2 / 12 = 1
  -- Eccentricity is 2
  eccentricity : ℝ
  eccentricity_eq : eccentricity = 2

/-- The angle bisector of ∠F₁AF₂ -/
def angle_bisector (h : Hyperbola) : ℝ → ℝ := 
  fun x ↦ 2 * x - 2

theorem hyperbola_and_angle_bisector (h : Hyperbola) 
  (h_point_A : h.point_A = (4, 6)) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 12 = 1}) ∧
  (∀ x : ℝ, angle_bisector h x = 2 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_angle_bisector_l457_45750


namespace NUMINAMATH_CALUDE_calvins_weight_after_training_l457_45736

/-- Calculates the final weight after a period of constant weight loss -/
def final_weight (initial_weight : ℕ) (weight_loss_per_month : ℕ) (months : ℕ) : ℕ :=
  initial_weight - weight_loss_per_month * months

/-- Theorem stating that Calvin's weight after one year of training is 154 pounds -/
theorem calvins_weight_after_training :
  final_weight 250 8 12 = 154 := by
  sorry

end NUMINAMATH_CALUDE_calvins_weight_after_training_l457_45736


namespace NUMINAMATH_CALUDE_log_identity_l457_45784

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_identity : log10 2 ^ 2 + log10 2 * log10 5 + log10 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l457_45784


namespace NUMINAMATH_CALUDE_mikis_sandcastle_height_l457_45735

/-- The height of Miki's sandcastle given the height of her sister's sandcastle and the difference in height -/
theorem mikis_sandcastle_height 
  (sisters_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : sisters_height = 0.5)
  (h2 : height_difference = 0.3333333333333333) : 
  sisters_height + height_difference = 0.8333333333333333 :=
by sorry

end NUMINAMATH_CALUDE_mikis_sandcastle_height_l457_45735


namespace NUMINAMATH_CALUDE_function_inequality_implies_t_bound_l457_45726

theorem function_inequality_implies_t_bound (t : ℝ) : 
  (∀ x : ℝ, (Real.exp (2 * x) - t) ≥ (t * Real.exp x - 1)) → 
  t ≤ 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_t_bound_l457_45726


namespace NUMINAMATH_CALUDE_smallest_AC_l457_45739

/-- Triangle ABC with point D on AC --/
structure Triangle :=
  (AC : ℕ)
  (CD : ℕ)
  (BD : ℝ)

/-- Conditions for the triangle --/
def ValidTriangle (t : Triangle) : Prop :=
  t.AC = t.AC  -- AB = AC (isosceles)
  ∧ t.CD ≤ t.AC  -- D is on AC
  ∧ t.BD ^ 2 = 85  -- BD² = 85
  ∧ t.AC ^ 2 = (t.AC - t.CD) ^ 2 + 85  -- Pythagorean theorem

/-- The smallest possible AC value is 11 --/
theorem smallest_AC : 
  ∀ t : Triangle, ValidTriangle t → t.AC ≥ 11 ∧ ∃ t' : Triangle, ValidTriangle t' ∧ t'.AC = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_AC_l457_45739


namespace NUMINAMATH_CALUDE_initial_group_size_l457_45725

/-- The number of initial persons in a group, given specific average age conditions. -/
theorem initial_group_size (initial_avg : ℝ) (new_persons : ℕ) (new_avg : ℝ) (final_avg : ℝ) :
  initial_avg = 16 →
  new_persons = 12 →
  new_avg = 15 →
  final_avg = 15.5 →
  ∃ n : ℕ, n * initial_avg + new_persons * new_avg = (n + new_persons) * final_avg ∧ n = 12 :=
by sorry

end NUMINAMATH_CALUDE_initial_group_size_l457_45725


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l457_45768

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_sum : a 0 + a 1 + a 2 = 3 * a 0) 
  (h_nonzero : a 0 ≠ 0) : 
  q = -2 ∨ q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l457_45768


namespace NUMINAMATH_CALUDE_cloth_cost_price_l457_45700

/-- Proves that the cost price of one meter of cloth is 140 Rs. given the conditions -/
theorem cloth_cost_price
  (total_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 30)
  (h2 : selling_price = 4500)
  (h3 : profit_per_meter = 10) :
  (selling_price - total_length * profit_per_meter) / total_length = 140 :=
by
  sorry

#check cloth_cost_price

end NUMINAMATH_CALUDE_cloth_cost_price_l457_45700


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l457_45780

/-- A triangle with sides that are consecutive natural numbers and largest angle twice the smallest -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n - 1
  side2 : ℕ := n
  side3 : ℕ := n + 1
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angle_sum : angleA + angleB + angleC = π
  angle_relation : angleC = 2 * angleA
  law_of_sines : (n - 1) / Real.sin angleA = n / Real.sin angleB
  law_of_cosines : (n - 1)^2 = (n + 1)^2 + n^2 - 2 * (n + 1) * n * Real.cos angleC

/-- The perimeter of the special triangle is 15 -/
theorem special_triangle_perimeter (t : SpecialTriangle) : t.side1 + t.side2 + t.side3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_perimeter_l457_45780


namespace NUMINAMATH_CALUDE_g_derivative_l457_45721

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem g_derivative (x : ℝ) : 
  deriv g x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_g_derivative_l457_45721


namespace NUMINAMATH_CALUDE_pure_imaginary_solution_real_sum_solution_l457_45772

def is_pure_imaginary (z : ℂ) : Prop := ∃ a : ℝ, z = Complex.I * a

theorem pure_imaginary_solution (z : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : Complex.abs (z - 1) = Complex.abs (z - 1 + Complex.I)) : 
  z = Complex.I ∨ z = -Complex.I :=
sorry

theorem real_sum_solution (z : ℂ) 
  (h1 : ∃ r : ℝ, z + 10 / z = r) 
  (h2 : 1 ≤ (z + 10 / z).re ∧ (z + 10 / z).re ≤ 6) :
  z = 1 + 3 * Complex.I ∨ z = 3 + Complex.I ∨ z = 3 - Complex.I :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solution_real_sum_solution_l457_45772


namespace NUMINAMATH_CALUDE_head_start_time_l457_45734

/-- Proves that given a runner completes a 1000-meter race in 190 seconds,
    the time equivalent to a 50-meter head start is 9.5 seconds. -/
theorem head_start_time (race_distance : ℝ) (race_time : ℝ) (head_start_distance : ℝ) : 
  race_distance = 1000 →
  race_time = 190 →
  head_start_distance = 50 →
  (head_start_distance / (race_distance / race_time)) = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_head_start_time_l457_45734


namespace NUMINAMATH_CALUDE_ian_lottery_winnings_l457_45774

theorem ian_lottery_winnings :
  ∀ (lottery_winnings : ℕ) (colin_payment helen_payment benedict_payment remaining : ℕ),
  colin_payment = 20 →
  helen_payment = 2 * colin_payment →
  benedict_payment = helen_payment / 2 →
  remaining = 20 →
  lottery_winnings = colin_payment + helen_payment + benedict_payment + remaining →
  lottery_winnings = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ian_lottery_winnings_l457_45774


namespace NUMINAMATH_CALUDE_positive_A_value_l457_45798

theorem positive_A_value : ∃ A : ℕ+, A^2 - 1 = 3577 * 3579 ∧ A = 3578 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l457_45798


namespace NUMINAMATH_CALUDE_exist_integers_product_minus_third_l457_45773

theorem exist_integers_product_minus_third : ∃ (a b c : ℤ), 
  (a * b - c = 2018) ∧ (b * c - a = 2018) ∧ (c * a - b = 2018) := by
sorry

end NUMINAMATH_CALUDE_exist_integers_product_minus_third_l457_45773


namespace NUMINAMATH_CALUDE_equal_area_triangles_l457_45777

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 20 20 24 = triangleArea 20 20 32 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l457_45777


namespace NUMINAMATH_CALUDE_gotham_street_homes_l457_45752

theorem gotham_street_homes (total_homes : ℚ) : 
  let termite_ridden := (1 / 3 : ℚ) * total_homes
  let collapsing := (1 / 4 : ℚ) * termite_ridden
  termite_ridden - collapsing = (1 / 4 : ℚ) * total_homes :=
by sorry

end NUMINAMATH_CALUDE_gotham_street_homes_l457_45752


namespace NUMINAMATH_CALUDE_min_scabs_per_day_l457_45753

def total_scabs : ℕ := 220
def days_in_week : ℕ := 7

theorem min_scabs_per_day :
  ∃ (n : ℕ), n * days_in_week ≥ total_scabs ∧
  ∀ (m : ℕ), m * days_in_week ≥ total_scabs → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_scabs_per_day_l457_45753


namespace NUMINAMATH_CALUDE_work_completion_time_l457_45741

theorem work_completion_time (a b c : ℝ) : 
  b = 12 →
  c = 24 →
  1 / a + 1 / b + 1 / c = 1 / 4 →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l457_45741


namespace NUMINAMATH_CALUDE_expand_expression_l457_45704

theorem expand_expression (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l457_45704


namespace NUMINAMATH_CALUDE_z_less_than_y_percentage_l457_45763

/-- Given w, x, y, z are real numbers satisfying certain conditions,
    prove that z is 46% less than y. -/
theorem z_less_than_y_percentage (w x y z : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 1.5 * w) :
  z = 0.54 * y := by
  sorry

end NUMINAMATH_CALUDE_z_less_than_y_percentage_l457_45763


namespace NUMINAMATH_CALUDE_salary_increase_l457_45730

theorem salary_increase
  (num_employees : ℕ)
  (avg_salary : ℝ)
  (manager_salary : ℝ)
  (h1 : num_employees = 20)
  (h2 : avg_salary = 1500)
  (h3 : manager_salary = 3600) :
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / (num_employees + 1)
  new_avg_salary - avg_salary = 100 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_l457_45730


namespace NUMINAMATH_CALUDE_prob_two_even_dice_l457_45756

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The set of even numbers on an 8-sided die -/
def even_numbers : Finset ℕ := {2, 4, 6, 8}

/-- The probability of rolling an even number on a single 8-sided die -/
def prob_even : ℚ := (even_numbers.card : ℚ) / sides

/-- The probability of rolling two even numbers on two 8-sided dice -/
theorem prob_two_even_dice : prob_even * prob_even = 1/4 := by sorry

end NUMINAMATH_CALUDE_prob_two_even_dice_l457_45756


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l457_45797

theorem min_value_theorem (x : ℝ) (h : x > 0) : 2*x + 1/(2*x) + 1 ≥ 3 := by
  sorry

theorem equality_exists : ∃ x : ℝ, x > 0 ∧ 2*x + 1/(2*x) + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l457_45797


namespace NUMINAMATH_CALUDE_fraction_simplification_l457_45761

theorem fraction_simplification (x : ℝ) : (2*x - 3)/4 + (5 - 4*x)/3 = (-10*x + 11)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l457_45761


namespace NUMINAMATH_CALUDE_F_max_value_l457_45796

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def f_derivative (x : ℝ) : ℝ := Real.cos x - Real.sin x

noncomputable def F (x : ℝ) : ℝ := f x * f_derivative x + f x ^ 2

theorem F_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), F x ≤ M) ∧ (∃ (x : ℝ), F x = M) ∧ M = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_F_max_value_l457_45796


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l457_45792

theorem complex_fraction_sum (A B : ℝ) : 
  (Complex.I : ℂ) * (3 + Complex.I) = (1 + 2 * Complex.I) * (A + B * Complex.I) → 
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l457_45792


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l457_45732

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 48 → area = (perimeter / 4)^2 → area = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l457_45732


namespace NUMINAMATH_CALUDE_three_letter_sets_count_l457_45749

/-- The number of permutations of k elements chosen from a set of n distinct elements -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def set_size : ℕ := 3

theorem three_letter_sets_count : permutations num_letters set_size = 720 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_sets_count_l457_45749


namespace NUMINAMATH_CALUDE_joanne_earnings_l457_45737

/-- Joanne's work schedule and earnings calculation -/
theorem joanne_earnings :
  let main_job_hours : ℕ := 8
  let main_job_rate : ℚ := 16
  let part_time_hours : ℕ := 2
  let part_time_rate : ℚ := 27/2  -- $13.50 represented as a fraction
  let days_worked : ℕ := 5
  
  let main_job_daily := main_job_hours * main_job_rate
  let part_time_daily := part_time_hours * part_time_rate
  let total_daily := main_job_daily + part_time_daily
  let total_weekly := total_daily * days_worked
  
  total_weekly = 775
:= by sorry


end NUMINAMATH_CALUDE_joanne_earnings_l457_45737


namespace NUMINAMATH_CALUDE_square_equality_l457_45723

theorem square_equality : (2023 + (-1011.5))^2 = (-1011.5)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l457_45723


namespace NUMINAMATH_CALUDE_f_properties_l457_45757

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

/-- The theorem stating the properties of function f -/
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → (f x ≥ 1 ↔ x ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Icc (11 * Real.pi / 12) Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l457_45757


namespace NUMINAMATH_CALUDE_coefficients_of_equation_l457_45751

-- Define the coefficients of a quadratic equation
def QuadraticCoefficients := ℝ × ℝ × ℝ

-- Function to get coefficients from a quadratic equation
def getCoefficients (a b c : ℝ) : QuadraticCoefficients := (a, b, c)

-- Theorem stating that the coefficients of 2x^2 - 6x = 9 are (2, -6, -9)
theorem coefficients_of_equation : 
  let eq := fun x : ℝ => 2 * x^2 - 6 * x - 9
  getCoefficients 2 (-6) (-9) = (2, -6, -9) := by sorry

end NUMINAMATH_CALUDE_coefficients_of_equation_l457_45751


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l457_45793

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (4^11 + 6^13)) → 
  2 ∣ (4^11 + 6^13) ∧ 
  ∀ p : ℕ, Nat.Prime p → p ∣ (4^11 + 6^13) → p ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l457_45793
