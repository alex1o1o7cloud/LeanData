import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_and_argument_l176_17621

theorem complex_magnitude_and_argument :
  ∃ (t : ℝ), t > 0 ∧ 
  (Complex.abs (9 + t * Complex.I) = 13 ↔ t = Real.sqrt 88) ∧
  Complex.arg (9 + t * Complex.I) ≠ π / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_and_argument_l176_17621


namespace NUMINAMATH_CALUDE_max_diff_slightly_unlucky_l176_17665

/-- A natural number is slightly unlucky if the sum of its digits in decimal system is divisible by 13. -/
def SlightlyUnlucky (n : ℕ) : Prop :=
  (n.digits 10).sum % 13 = 0

/-- For any non-negative integer k, the intervals [100(k+1), 100(k+1)+39], [100k+60, 100k+99], and [100k+20, 100k+59] each contain at least one slightly unlucky number. -/
axiom slightly_unlucky_intervals (k : ℕ) :
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*(k+1) ≤ n ∧ n ≤ 100*(k+1)+39) ∧
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*k+60 ≤ n ∧ n ≤ 100*k+99) ∧
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*k+20 ≤ n ∧ n ≤ 100*k+59)

/-- The maximum difference between consecutive slightly unlucky numbers is 79. -/
theorem max_diff_slightly_unlucky :
  ∀ m n : ℕ, SlightlyUnlucky m → SlightlyUnlucky n → m < n →
  (∀ k : ℕ, SlightlyUnlucky k → m < k → k < n → False) →
  n - m ≤ 79 :=
sorry

end NUMINAMATH_CALUDE_max_diff_slightly_unlucky_l176_17665


namespace NUMINAMATH_CALUDE_total_mass_of_water_l176_17602

/-- The total mass of water in two glasses on an unequal-arm scale -/
theorem total_mass_of_water (L m l : ℝ) (hL : L > 0) (hm : m > 0) (hl : l ≠ 0) : ∃ total_mass : ℝ,
  (∃ m₁ m₂ l₁ : ℝ, 
    -- Initial balance condition
    m₁ * l₁ = m₂ * (L - l₁) ∧
    -- Balance condition after transfer
    (m₁ - m) * (l₁ + l) = (m₂ + m) * (L - l₁ - l) ∧
    -- Total mass definition
    total_mass = m₁ + m₂) ∧
  total_mass = m * L / l :=
sorry

end NUMINAMATH_CALUDE_total_mass_of_water_l176_17602


namespace NUMINAMATH_CALUDE_average_annual_growth_rate_l176_17683

theorem average_annual_growth_rate 
  (p q : ℝ) 
  (hp : p > -1) 
  (hq : q > -1) :
  ∃ x : ℝ, x > -1 ∧ (1 + x)^2 = (1 + p) * (1 + q) ∧ 
  x = Real.sqrt ((1 + p) * (1 + q)) - 1 :=
sorry

end NUMINAMATH_CALUDE_average_annual_growth_rate_l176_17683


namespace NUMINAMATH_CALUDE_unique_square_number_l176_17647

theorem unique_square_number : ∃! x : ℕ, 
  x > 39 ∧ x < 80 ∧ 
  ∃ y : ℕ, x = y * y ∧ 
  ∃ z : ℕ, x = 4 * z :=
by
  sorry

end NUMINAMATH_CALUDE_unique_square_number_l176_17647


namespace NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l176_17616

theorem min_value_sin_2x_minus_pi_4 :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (2 * x - Real.pi / 4) ≥ -Real.sqrt 2 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (2 * x - Real.pi / 4) = -Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_sin_2x_minus_pi_4_l176_17616


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l176_17680

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l176_17680


namespace NUMINAMATH_CALUDE_inscribed_cube_edge_length_l176_17675

theorem inscribed_cube_edge_length (S : Real) (r : Real) (x : Real) :
  S = 4 * Real.pi →  -- Surface area of the sphere
  S = 4 * Real.pi * r^2 →  -- Formula for surface area of a sphere
  x * Real.sqrt 3 = 2 * r →  -- Relationship between cube diagonal and sphere diameter
  x = 2 * Real.sqrt 3 / 3 :=  -- Edge length of the inscribed cube
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_edge_length_l176_17675


namespace NUMINAMATH_CALUDE_car_subsidy_theorem_l176_17605

/-- Represents the sales and pricing data for a car dealership --/
structure CarSalesData where
  manual_nov : ℕ
  auto_nov : ℕ
  manual_dec : ℕ
  auto_dec : ℕ
  manual_price : ℕ
  auto_price : ℕ
  subsidy_rate : ℚ

/-- Calculates the total government subsidy based on car sales data --/
def total_subsidy (data : CarSalesData) : ℚ :=
  (data.manual_dec * data.manual_price + data.auto_dec * data.auto_price) * data.subsidy_rate

/-- Theorem stating the total government subsidy for the given scenario --/
theorem car_subsidy_theorem (data : CarSalesData) :
  data.manual_nov + data.auto_nov = 960 →
  data.manual_dec + data.auto_dec = 1228 →
  data.manual_dec = (13 * data.manual_nov) / 10 →
  data.auto_dec = (5 * data.auto_nov) / 4 →
  data.manual_price = 80000 →
  data.auto_price = 90000 →
  data.subsidy_rate = 1 / 20 →
  total_subsidy data = 516200000 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_car_subsidy_theorem_l176_17605


namespace NUMINAMATH_CALUDE_relay_team_permutations_l176_17655

theorem relay_team_permutations :
  (Finset.range 4).card.factorial = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l176_17655


namespace NUMINAMATH_CALUDE_runners_simultaneous_time_l176_17699

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℚ  -- Speed in circles per second
  direction : Bool  -- true for clockwise, false for counterclockwise

/-- Calculates the time two runners spend simultaneously in a quarter arc section -/
def simultaneous_time_in_section (runner1 runner2 : Runner) : ℚ :=
  sorry

/-- Theorem stating that the total time two specific runners spend simultaneously
    in a quarter arc section is 41 seconds -/
theorem runners_simultaneous_time :
  let xiaohua : Runner := ⟨1/72, true⟩
  let xiaozhang : Runner := ⟨1/80, false⟩
  simultaneous_time_in_section xiaohua xiaozhang = 41 := by sorry

end NUMINAMATH_CALUDE_runners_simultaneous_time_l176_17699


namespace NUMINAMATH_CALUDE_extreme_values_and_inequality_l176_17623

def f (a b c x : ℝ) : ℝ := x^3 - a*x^2 + b*x + c

theorem extreme_values_and_inequality 
  (a b c : ℝ) 
  (h1 : ∃ y, (deriv (f a b c)) (-1) = y ∧ y = 0)
  (h2 : ∃ y, (deriv (f a b c)) 3 = y ∧ y = 0)
  (h3 : ∀ x ∈ Set.Icc (-2) 6, f a b c x < c^2 + 4*c) :
  a = 3 ∧ b = -9 ∧ (c > 6 ∨ c < -9) := by sorry

end NUMINAMATH_CALUDE_extreme_values_and_inequality_l176_17623


namespace NUMINAMATH_CALUDE_quadratic_transformation_l176_17644

theorem quadratic_transformation (x : ℝ) :
  (x^2 - 4*x + 3 = 0) → (∃ h k : ℝ, (x + h)^2 = k ∧ k = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l176_17644


namespace NUMINAMATH_CALUDE_ones_digit_of_8_power_50_l176_17670

theorem ones_digit_of_8_power_50 : 8^50 % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_power_50_l176_17670


namespace NUMINAMATH_CALUDE_equation_solutions_l176_17689

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, (1/4) * (2*x + 3)^3 = 16 ↔ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l176_17689


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l176_17625

theorem cos_five_pi_thirds_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π/6 - α) = 2/3) : 
  Real.cos (5*π/3 + 2*α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l176_17625


namespace NUMINAMATH_CALUDE_digit_2023_is_7_l176_17695

/-- The sequence of digits obtained by writing integers 1 through 9999 in ascending order -/
def digit_sequence : ℕ → ℕ := sorry

/-- The real number x defined as .123456789101112...99989999 -/
noncomputable def x : ℝ := sorry

/-- The nth digit to the right of the decimal point in x -/
def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_2023_is_7 : nth_digit 2023 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_2023_is_7_l176_17695


namespace NUMINAMATH_CALUDE_village_population_l176_17627

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.2) = 3553 → P = 4678 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l176_17627


namespace NUMINAMATH_CALUDE_factors_of_42_l176_17698

/-- The number of positive factors of 42 -/
def number_of_factors_42 : ℕ :=
  (Finset.filter (· ∣ 42) (Finset.range 43)).card

/-- Theorem stating that the number of positive factors of 42 is 8 -/
theorem factors_of_42 : number_of_factors_42 = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_42_l176_17698


namespace NUMINAMATH_CALUDE_smaller_bedroom_size_l176_17657

/-- Given two bedrooms with a total area of 300 square feet, where one bedroom
    is 60 square feet larger than the other, prove that the smaller bedroom
    is 120 square feet. -/
theorem smaller_bedroom_size (total_area : ℝ) (difference : ℝ) (smaller : ℝ) :
  total_area = 300 →
  difference = 60 →
  total_area = smaller + (smaller + difference) →
  smaller = 120 := by
  sorry

end NUMINAMATH_CALUDE_smaller_bedroom_size_l176_17657


namespace NUMINAMATH_CALUDE_cryptarithm_solution_is_unique_l176_17648

/-- Represents a cryptarithm solution -/
structure CryptarithmSolution where
  F : Nat
  R : Nat
  Y : Nat
  H : Nat
  A : Nat
  M : Nat
  digit_constraint : F < 10 ∧ R < 10 ∧ Y < 10 ∧ H < 10 ∧ A < 10 ∧ M < 10
  unique_digits : F ≠ R ∧ F ≠ Y ∧ F ≠ H ∧ F ≠ A ∧ F ≠ M ∧
                  R ≠ Y ∧ R ≠ H ∧ R ≠ A ∧ R ≠ M ∧
                  Y ≠ H ∧ Y ≠ A ∧ Y ≠ M ∧
                  H ≠ A ∧ H ≠ M ∧
                  A ≠ M
  equation_holds : 7 * (100000 * F + 10000 * R + 1000 * Y + 100 * H + 10 * A + M) =
                   6 * (100000 * H + 10000 * A + 1000 * M + 100 * F + 10 * R + Y)

theorem cryptarithm_solution_is_unique : 
  ∀ (sol : CryptarithmSolution), 
    100 * sol.F + 10 * sol.R + sol.Y = 461 ∧ 
    100 * sol.H + 10 * sol.A + sol.M = 538 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_is_unique_l176_17648


namespace NUMINAMATH_CALUDE_tuesday_sales_fifty_l176_17672

/-- Calculates the number of books sold on Tuesday given the initial stock,
    sales on other days, and the percentage of unsold books. -/
def books_sold_tuesday (initial_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ)
    (unsold_percentage : ℚ) : ℕ :=
  let unsold_books := (initial_stock : ℚ) * unsold_percentage / 100
  let other_days_sales := monday_sales + wednesday_sales + thursday_sales + friday_sales
  initial_stock - (other_days_sales + unsold_books.ceil.toNat)

/-- Theorem stating that the number of books sold on Tuesday is 50. -/
theorem tuesday_sales_fifty :
  books_sold_tuesday 1100 75 64 78 135 (63945/1000) = 50 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_sales_fifty_l176_17672


namespace NUMINAMATH_CALUDE_four_correct_propositions_l176_17650

theorem four_correct_propositions : 
  (2 ≤ 3) ∧ 
  (∀ m : ℝ, m ≥ 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧ 
  (∀ x y : ℝ, x^2 = y^2 → |x| = |y|) ∧ 
  (∀ a b c : ℝ, a > b ↔ a + c > b + c) := by
  sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l176_17650


namespace NUMINAMATH_CALUDE_total_questions_submitted_l176_17686

/-- Given the ratio of questions submitted by Rajat, Vikas, and Abhishek,
    and the number of questions submitted by Vikas, calculate the total
    number of questions submitted. -/
theorem total_questions_submitted
  (ratio_rajat : ℕ)
  (ratio_vikas : ℕ)
  (ratio_abhishek : ℕ)
  (vikas_questions : ℕ)
  (h_ratio : ratio_rajat = 7 ∧ ratio_vikas = 3 ∧ ratio_abhishek = 2)
  (h_vikas : vikas_questions = 6) :
  ratio_rajat * vikas_questions / ratio_vikas +
  vikas_questions +
  ratio_abhishek * vikas_questions / ratio_vikas = 24 :=
by sorry

end NUMINAMATH_CALUDE_total_questions_submitted_l176_17686


namespace NUMINAMATH_CALUDE_dave_total_earnings_l176_17652

def dave_earnings (hourly_wage : ℝ) (monday_hours : ℝ) (tuesday_hours : ℝ) : ℝ :=
  hourly_wage * (monday_hours + tuesday_hours)

theorem dave_total_earnings :
  dave_earnings 6 6 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_dave_total_earnings_l176_17652


namespace NUMINAMATH_CALUDE_calculation_part1_sum_first_25_odd_numbers_l176_17613

-- Part 1
theorem calculation_part1 : 0.45 * 2.5 + 4.5 * 0.65 + 0.45 = 4.5 := by
  sorry

-- Part 2
def first_n_odd_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * i + 1)

theorem sum_first_25_odd_numbers :
  (first_n_odd_numbers 25).sum = 625 := by
  sorry

end NUMINAMATH_CALUDE_calculation_part1_sum_first_25_odd_numbers_l176_17613


namespace NUMINAMATH_CALUDE_thirty_percent_more_than_hundred_l176_17690

theorem thirty_percent_more_than_hundred (x : ℝ) : x + (1/4) * x = 130 → x = 104 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_more_than_hundred_l176_17690


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l176_17639

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- Returns the tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- Returns the units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- The product of digits of a two-digit number -/
def product_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n * units_digit n

theorem unique_two_digit_number : 
  ∃! (n : TwoDigitNumber), 
    n.val = 4 * sum_of_digits n ∧ 
    n.val = 3 * product_of_digits n ∧
    n.val = 24 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l176_17639


namespace NUMINAMATH_CALUDE_unique_campers_difference_l176_17673

def rowing_problem (morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only : ℕ) : Prop :=
  let total_afternoon := morning_and_afternoon + afternoon_and_evening + (afternoon - morning_and_afternoon - afternoon_and_evening)
  let total_evening := afternoon_and_evening + evening_only
  morning = 33 ∧
  morning_and_afternoon = 11 ∧
  afternoon = 34 ∧
  afternoon_and_evening = 20 ∧
  evening_only = 10 ∧
  total_afternoon - total_evening = 4

theorem unique_campers_difference :
  ∃ (morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only : ℕ),
    rowing_problem morning afternoon evening morning_and_afternoon afternoon_and_evening evening_only :=
by
  sorry

end NUMINAMATH_CALUDE_unique_campers_difference_l176_17673


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l176_17617

def fruit_survey (apples bananas cherries oranges grapes : ℕ) : Prop :=
  let total := apples + bananas + cherries + oranges + grapes
  let apple_percentage := (apples : ℚ) / (total : ℚ) * 100
  apple_percentage = 26.67

theorem apple_preference_percentage :
  fruit_survey 80 90 50 40 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l176_17617


namespace NUMINAMATH_CALUDE_isabels_candy_l176_17632

/-- Given that Isabel initially had 68 pieces of candy and ended up with 93 pieces,
    prove that her friend gave her 25 pieces. -/
theorem isabels_candy (initial : ℕ) (final : ℕ) (h1 : initial = 68) (h2 : final = 93) :
  final - initial = 25 := by
  sorry

end NUMINAMATH_CALUDE_isabels_candy_l176_17632


namespace NUMINAMATH_CALUDE_birds_flew_up_l176_17638

theorem birds_flew_up (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 231)
  (h2 : final_birds = 312)
  : final_birds - initial_birds = 81 := by
  sorry

end NUMINAMATH_CALUDE_birds_flew_up_l176_17638


namespace NUMINAMATH_CALUDE_set_B_representation_l176_17626

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x - x = 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b x - a*x = 0}

-- State the theorem
theorem set_B_representation (a b : ℝ) : 
  A a b = {1, -3} → B a b = {-2 - Real.sqrt 7, -2 + Real.sqrt 7} := by
  sorry

end NUMINAMATH_CALUDE_set_B_representation_l176_17626


namespace NUMINAMATH_CALUDE_boxes_with_neither_l176_17630

theorem boxes_with_neither (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : pencils = 8)
  (h3 : pens = 5)
  (h4 : both = 4) :
  total - (pencils + pens - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l176_17630


namespace NUMINAMATH_CALUDE_triangle_sine_sum_bound_l176_17601

/-- Given a triangle with angles A, B, and C (in radians), 
    the sum of the sines of its angles is at most 3√3/2, 
    with equality if and only if the triangle is equilateral. -/
theorem triangle_sine_sum_bound (A B C : ℝ) 
    (h_angles : A + B + C = π) 
    (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 ∧ 
  (Real.sin A + Real.sin B + Real.sin C = 3 * Real.sqrt 3 / 2 ↔ A = B ∧ B = C) := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_bound_l176_17601


namespace NUMINAMATH_CALUDE_positive_numbers_inequalities_l176_17642

theorem positive_numbers_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c = 1 → (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c) ∧
  (∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r → a^2 + b^2 + c^2 > (a - b + c)^2) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequalities_l176_17642


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l176_17612

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (2 * a * (-1) - b * 2 + 2 = 0) → -- Line passes through circle center (-1, 2)
  (∀ x y : ℝ, 2 * a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (2 * a' * (-1) - b' * 2 + 2 = 0) → 
    (1/a + 1/b) ≤ (1/a' + 1/b')) →
  1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l176_17612


namespace NUMINAMATH_CALUDE_bob_show_dogs_count_l176_17643

/-- The number of show dogs Bob bought -/
def num_show_dogs : ℕ := 2

/-- The cost of each show dog in dollars -/
def cost_per_show_dog : ℕ := 250

/-- The number of puppies -/
def num_puppies : ℕ := 6

/-- The selling price of each puppy in dollars -/
def price_per_puppy : ℕ := 350

/-- The total profit in dollars -/
def total_profit : ℕ := 1600

theorem bob_show_dogs_count :
  num_puppies * price_per_puppy - num_show_dogs * cost_per_show_dog = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bob_show_dogs_count_l176_17643


namespace NUMINAMATH_CALUDE_least_three_digit_with_product_12_l176_17606

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_product_12_l176_17606


namespace NUMINAMATH_CALUDE_triangle_side_length_l176_17663

theorem triangle_side_length (y : ℝ) :
  y > 0 →  -- y is positive
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧  -- a and b are positive
    a = 10 ∧  -- shorter leg is 10
    a^2 + b^2 = y^2 ∧  -- Pythagorean theorem
    b = a * Real.sqrt 3 →  -- ratio of sides in a 30-60-90 triangle
  y = 10 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l176_17663


namespace NUMINAMATH_CALUDE_cake_portion_theorem_l176_17620

theorem cake_portion_theorem (tom_ate jenny_took : ℚ) : 
  tom_ate = 60 / 100 →
  jenny_took = 1 / 4 →
  (1 - tom_ate) * (1 - jenny_took) = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_portion_theorem_l176_17620


namespace NUMINAMATH_CALUDE_linear_function_derivative_l176_17674

/-- Given a linear function f(x) = ax + 3 where f'(1) = 3, prove that a = 3 -/
theorem linear_function_derivative (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x + 3) ∧ (deriv f 1 = 3)) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_derivative_l176_17674


namespace NUMINAMATH_CALUDE_ticket_probabilities_l176_17658

/-- Represents a group of tickets -/
structure TicketGroup where
  football : ℕ
  volleyball : ℕ

/-- The probability of drawing a football ticket from a group -/
def football_prob (group : TicketGroup) : ℚ :=
  group.football / (group.football + group.volleyball)

/-- The setup of the ticket drawing scenario -/
def ticket_scenario : Prop :=
  ∃ (group1 group2 : TicketGroup),
    group1.football = 6 ∧ group1.volleyball = 4 ∧
    group2.football = 4 ∧ group2.volleyball = 6

theorem ticket_probabilities (h : ticket_scenario) :
  ∃ (group1 group2 : TicketGroup),
    (football_prob group1 * football_prob group2 = 6/25) ∧
    (1 - (1 - football_prob group1) * (1 - football_prob group2) = 19/25) :=
by sorry

end NUMINAMATH_CALUDE_ticket_probabilities_l176_17658


namespace NUMINAMATH_CALUDE_m_range_theorem_l176_17615

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 1| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem m_range_theorem (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(q x m) → ¬(p x)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  (0 < m ∧ m ≤ 2) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_m_range_theorem_l176_17615


namespace NUMINAMATH_CALUDE_age_problem_l176_17671

/-- Theorem: Given the age relationships and total age, prove b's age --/
theorem age_problem (a b c d e : ℝ) : 
  a = b + 2 →
  b = 2 * c →
  d = a - 3 →
  e = d / 2 + 3 →
  a + b + c + d + e = 70 →
  b = 16.625 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l176_17671


namespace NUMINAMATH_CALUDE_rectangle_division_theorem_l176_17669

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

theorem rectangle_division_theorem :
  ∃ (original : Rectangle) (largest smallest : Rectangle),
    (∃ (other1 other2 : Rectangle),
      area original = area largest + area smallest + area other1 + area other2) ∧
    perimeter largest = 28 ∧
    perimeter smallest = 12 ∧
    area original = 96 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_theorem_l176_17669


namespace NUMINAMATH_CALUDE_equation_solution_l176_17685

theorem equation_solution : ∃ x : ℚ, (-2*x + 3 - 2*x + 3 = 3*x - 6) ∧ (x = 12/7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l176_17685


namespace NUMINAMATH_CALUDE_smallest_consecutive_sum_l176_17624

theorem smallest_consecutive_sum (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : n * a + n * (n - 1) / 2 = 2016) : 
  ∃ (m : ℕ), m ≥ 1 ∧ m * a + m * (m - 1) / 2 = 2016 ∧ m > 1 → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_consecutive_sum_l176_17624


namespace NUMINAMATH_CALUDE_smallest_z_value_l176_17678

theorem smallest_z_value (w x y z : ℤ) : 
  (∀ n : ℤ, n ≥ 0 → (w + 2*n)^3 + (x + 2*n)^3 + (y + 2*n)^3 = (z + 2*n)^3) →
  (x = w + 2) →
  (y = x + 2) →
  (z = y + 2) →
  (w > 0) →
  (2 : ℤ) ≤ z :=
sorry

end NUMINAMATH_CALUDE_smallest_z_value_l176_17678


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l176_17619

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def linePassesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.intercept / l.slope = -l.intercept

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∀ (l : Line2D),
    linePassesThroughPoint l { x := 1, y := 2 } →
    hasEqualIntercepts l →
    (l.slope = -1 ∧ l.intercept = 3) ∨ (l.slope = 2 ∧ l.intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l176_17619


namespace NUMINAMATH_CALUDE_fraction_product_equality_l176_17600

theorem fraction_product_equality : (1 / 3 : ℚ)^3 * (1 / 7 : ℚ)^2 = 1 / 1323 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l176_17600


namespace NUMINAMATH_CALUDE_remainder_equality_l176_17618

theorem remainder_equality (P P' K D R R' : ℕ) (r r' : ℕ) 
  (h1 : P > P') 
  (h2 : K ∣ P) 
  (h3 : K ∣ P') 
  (h4 : P % D = R) 
  (h5 : P' % D = R') 
  (h6 : (P * K - P') % D = r) 
  (h7 : (R * K - R') % D = r') : 
  r = r' := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l176_17618


namespace NUMINAMATH_CALUDE_large_ball_uses_300_rubber_bands_l176_17637

/-- Calculates the number of rubber bands used in a large ball -/
def large_ball_rubber_bands (total_rubber_bands : ℕ) (small_balls : ℕ) (rubber_bands_per_small : ℕ) (large_balls : ℕ) : ℕ :=
  (total_rubber_bands - small_balls * rubber_bands_per_small) / large_balls

/-- Proves that a large ball uses 300 rubber bands given the problem conditions -/
theorem large_ball_uses_300_rubber_bands :
  large_ball_rubber_bands 5000 22 50 13 = 300 := by
  sorry

end NUMINAMATH_CALUDE_large_ball_uses_300_rubber_bands_l176_17637


namespace NUMINAMATH_CALUDE_f_max_value_l176_17656

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + Real.sin x * Real.cos x + Real.cos x ^ 4

theorem f_max_value : ∀ x : ℝ, f x ≤ 9/8 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l176_17656


namespace NUMINAMATH_CALUDE_eliminate_denominators_l176_17634

theorem eliminate_denominators (x : ℝ) : 
  (x / 2 - 1 = (x - 1) / 3) ↔ (3 * x - 6 = 2 * (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l176_17634


namespace NUMINAMATH_CALUDE_cube_root_function_l176_17691

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 125, 
    prove that y = 8√3/5 when x = 8 -/
theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → (k * x^(1/3) = 4 * Real.sqrt 3 ↔ x = 125)) → 
  k * 8^(1/3) = 8 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l176_17691


namespace NUMINAMATH_CALUDE_mario_salary_increase_l176_17608

/-- Proves that Mario's salary increase is 0% given the conditions of the problem -/
theorem mario_salary_increase (mario_salary_this_year : ℝ) 
  (bob_salary_last_year : ℝ) (bob_salary_increase : ℝ) :
  mario_salary_this_year = 4000 →
  bob_salary_last_year = 3 * mario_salary_this_year →
  bob_salary_increase = 0.2 →
  (mario_salary_this_year / bob_salary_last_year * 3 - 1) * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mario_salary_increase_l176_17608


namespace NUMINAMATH_CALUDE_rocks_theorem_l176_17654

def rocks_problem (initial_rocks : ℕ) (eaten_fraction : ℚ) (retrieved_rocks : ℕ) : Prop :=
  let remaining_after_eating := initial_rocks - (initial_rocks * eaten_fraction).floor
  let final_rocks := remaining_after_eating + retrieved_rocks
  initial_rocks = 10 ∧ eaten_fraction = 1/2 ∧ retrieved_rocks = 2 → final_rocks = 7

theorem rocks_theorem : rocks_problem 10 (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_rocks_theorem_l176_17654


namespace NUMINAMATH_CALUDE_quadratic_sum_l176_17696

/-- Given a quadratic expression 4x^2 - 8x - 3, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals -2. -/
theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 4*x^2 - 8*x - 3 = a*(x - h)^2 + k) → a + h + k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l176_17696


namespace NUMINAMATH_CALUDE_combination_equation_solution_l176_17636

theorem combination_equation_solution (n : ℕ) : n ≥ 2 → (Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l176_17636


namespace NUMINAMATH_CALUDE_special_arrangements_count_l176_17604

/-- The number of ways to arrange guests in a special circular formation. -/
def specialArrangements (n : ℕ) : ℕ :=
  (3 * n).factorial

/-- The main theorem stating that the number of special arrangements is (3n)! -/
theorem special_arrangements_count (n : ℕ) :
  specialArrangements n = (3 * n).factorial := by
  sorry

end NUMINAMATH_CALUDE_special_arrangements_count_l176_17604


namespace NUMINAMATH_CALUDE_no_indefinite_cutting_l176_17666

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

/-- Defines the cutting procedure for rectangles -/
def cut_rectangle (T : Rectangle) : Option (Rectangle × Rectangle) :=
  sorry

/-- Checks if two rectangles are similar -/
def are_similar (T1 T2 : Rectangle) : Prop :=
  T1.length / T1.width = T2.length / T2.width

/-- Checks if two rectangles are congruent -/
def are_congruent (T1 T2 : Rectangle) : Prop :=
  T1.length = T2.length ∧ T1.width = T2.width

/-- Defines the property of indefinite cutting -/
def can_cut_indefinitely (T : Rectangle) : Prop :=
  ∀ n : ℕ, ∃ (T_seq : ℕ → Rectangle), 
    T_seq 0 = T ∧
    (∀ i < n, 
      ∃ T1 T2 : Rectangle, 
        cut_rectangle (T_seq i) = some (T1, T2) ∧
        are_similar T1 T2 ∧
        ¬are_congruent T1 T2 ∧
        T_seq (i + 1) = T1)

theorem no_indefinite_cutting : ¬∃ T : Rectangle, can_cut_indefinitely T := by
  sorry

end NUMINAMATH_CALUDE_no_indefinite_cutting_l176_17666


namespace NUMINAMATH_CALUDE_lg_100_equals_2_l176_17645

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_100_equals_2 : lg 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_100_equals_2_l176_17645


namespace NUMINAMATH_CALUDE_real_y_condition_l176_17610

theorem real_y_condition (x y : ℝ) : 
  (4 * y^2 + 2 * x * y + |x| + 8 = 0) → 
  (∃ (y : ℝ), 4 * y^2 + 2 * x * y + |x| + 8 = 0) ↔ (x ≤ -10 ∨ x ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_real_y_condition_l176_17610


namespace NUMINAMATH_CALUDE_problem_statement_l176_17694

theorem problem_statement (A B C : ℚ) 
  (h1 : 1 / A = -3)
  (h2 : 2 / B = 4)
  (h3 : 3 / C = 1 / 2) :
  6 * A - 8 * B + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l176_17694


namespace NUMINAMATH_CALUDE_sum_of_digits_7_pow_1050_l176_17693

theorem sum_of_digits_7_pow_1050 : ∃ (a b : ℕ), 
  7^1050 % 100 = 10 * a + b ∧ a + b = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_pow_1050_l176_17693


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l176_17682

theorem unique_two_digit_number (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (13 * t ≡ 42 [ZMOD 100]) ↔ t = 34 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l176_17682


namespace NUMINAMATH_CALUDE_value_of_x_l176_17614

theorem value_of_x (x y z : ℤ) 
  (eq1 : 4*x + y + z = 80) 
  (eq2 : 2*x - y - z = 40) 
  (eq3 : 3*x + y - z = 20) : 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l176_17614


namespace NUMINAMATH_CALUDE_infinite_representations_l176_17609

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 10*x^2 + 29*x - 25

-- Define the property for a number to be a root of f
def is_root (x : ℝ) : Prop := f x = 0

-- Define the property for two numbers to be distinct
def are_distinct (x y : ℝ) : Prop := x ≠ y

-- Define the property for a positive integer to have the required representation
def has_representation (n : ℕ) (α β : ℝ) : Prop :=
  ∃ (r s : ℤ), n = ⌊r * α⌋ ∧ n = ⌊s * β⌋

-- State the theorem
theorem infinite_representations :
  ∃ (α β : ℝ), is_root α ∧ is_root β ∧ are_distinct α β ∧
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → has_representation n α β :=
sorry

end NUMINAMATH_CALUDE_infinite_representations_l176_17609


namespace NUMINAMATH_CALUDE_carrie_farm_earnings_l176_17631

def total_money (num_tomatoes : ℕ) (num_carrots : ℕ) (price_tomato : ℚ) (price_carrot : ℚ) : ℚ :=
  num_tomatoes * price_tomato + num_carrots * price_carrot

theorem carrie_farm_earnings :
  total_money 200 350 1 (3/2) = 725 := by
  sorry

end NUMINAMATH_CALUDE_carrie_farm_earnings_l176_17631


namespace NUMINAMATH_CALUDE_load_exceeds_five_trip_capacity_l176_17688

/-- Proves that the total weight of the load exceeds the truck's capacity in five trips -/
theorem load_exceeds_five_trip_capacity
  (truck_capacity : ℕ)
  (rice_sacks : ℕ)
  (rice_weight : ℕ)
  (corn_sacks : ℕ)
  (corn_weight : ℕ)
  (h1 : truck_capacity = 2000)
  (h2 : rice_sacks = 150)
  (h3 : rice_weight = 60)
  (h4 : corn_sacks = 100)
  (h5 : corn_weight = 25)
  : rice_sacks * rice_weight + corn_sacks * corn_weight > 5 * truck_capacity :=
by
  sorry

#check load_exceeds_five_trip_capacity

end NUMINAMATH_CALUDE_load_exceeds_five_trip_capacity_l176_17688


namespace NUMINAMATH_CALUDE_root_conditions_l176_17664

theorem root_conditions (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c ↔ x = -1 ∨ x = 1) ↔ 
  (a = -6 ∧ b = -1 ∧ c = -4) :=
sorry

end NUMINAMATH_CALUDE_root_conditions_l176_17664


namespace NUMINAMATH_CALUDE_max_value_x_minus_2y_l176_17633

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 - 8*x + y^2 - 6*y + 24 = 0) :
  ∃ (max : ℝ), max = Real.sqrt 5 - 2 ∧ ∀ (x' y' : ℝ), x'^2 - 8*x' + y'^2 - 6*y' + 24 = 0 → x' - 2*y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2y_l176_17633


namespace NUMINAMATH_CALUDE_min_value_a_l176_17640

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a * x * Real.exp x - x - Real.log x ≥ 0) → 
  a ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l176_17640


namespace NUMINAMATH_CALUDE_nested_radical_value_l176_17676

/-- The value of the infinite nested radical √(6 + √(6 + √(6 + ...))) -/
noncomputable def nested_radical : ℝ :=
  Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 6)))

/-- The nested radical equals 3 -/
theorem nested_radical_value : nested_radical = 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l176_17676


namespace NUMINAMATH_CALUDE_average_brown_mms_l176_17684

def brown_mms : List Nat := [9, 12, 8, 8, 3]

theorem average_brown_mms :
  (brown_mms.sum / brown_mms.length : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_brown_mms_l176_17684


namespace NUMINAMATH_CALUDE_gold_coins_count_verify_conditions_l176_17697

/-- The number of gold coins -/
def n : ℕ := 109

/-- The number of treasure chests -/
def c : ℕ := 13

/-- Theorem stating that the number of gold coins is 109 -/
theorem gold_coins_count : n = 109 :=
  by
  -- Condition 1: When putting 12 gold coins in each chest, 4 chests were left empty
  have h1 : n = 12 * (c - 4) := by sorry
  
  -- Condition 2: When putting 8 gold coins in each chest, 5 gold coins were left over
  have h2 : n = 8 * c + 5 := by sorry
  
  -- Prove that n equals 109
  sorry

/-- Theorem verifying the conditions -/
theorem verify_conditions :
  (n = 12 * (c - 4)) ∧ (n = 8 * c + 5) :=
  by sorry

end NUMINAMATH_CALUDE_gold_coins_count_verify_conditions_l176_17697


namespace NUMINAMATH_CALUDE_bakery_chairs_count_l176_17692

theorem bakery_chairs_count :
  let indoor_tables : ℕ := 8
  let outdoor_tables : ℕ := 12
  let chairs_per_indoor_table : ℕ := 3
  let chairs_per_outdoor_table : ℕ := 3
  let total_chairs := indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table
  total_chairs = 60 := by
  sorry

end NUMINAMATH_CALUDE_bakery_chairs_count_l176_17692


namespace NUMINAMATH_CALUDE_alphabet_theorem_l176_17629

theorem alphabet_theorem (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h1 : total = 76) 
  (h2 : both = 20) 
  (h3 : line_only = 46) 
  (h4 : total = both + line_only + (total - (both + line_only))) :
  total - (both + line_only) = 30 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_theorem_l176_17629


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l176_17661

theorem hexagon_angle_measure (N U M B E R S : ℝ) : 
  -- Hexagon condition
  N + U + M + B + E + R + S = 720 →
  -- Congruent angles
  N = M →
  B = R →
  -- Supplementary angles
  U + S = 180 →
  -- Conclusion
  E = 180 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l176_17661


namespace NUMINAMATH_CALUDE_stating_largest_valid_m_l176_17653

/-- 
Given a positive integer m, checks if m! can be expressed as the product 
of m - 4 consecutive positive integers.
-/
def is_valid (m : ℕ+) : Prop :=
  ∃ a : ℕ, m.val.factorial = (Finset.range (m - 4)).prod (λ i => i + a + 1)

/-- 
Theorem stating that 1 is the largest positive integer m such that m! 
can be expressed as the product of m - 4 consecutive positive integers.
-/
theorem largest_valid_m : 
  is_valid 1 ∧ ∀ m : ℕ+, m > 1 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_stating_largest_valid_m_l176_17653


namespace NUMINAMATH_CALUDE_largest_selected_is_57_l176_17677

/-- Represents a systematic sampling of a numbered set of elements. -/
structure SystematicSampling where
  total_elements : ℕ
  smallest_selected : ℕ
  second_smallest_selected : ℕ

/-- Calculates the largest selected number in a systematic sampling. -/
def largest_selected (s : SystematicSampling) : ℕ :=
  let interval := s.second_smallest_selected - s.smallest_selected
  let sample_size := s.total_elements / interval
  s.smallest_selected + interval * (sample_size - 1)

/-- Theorem stating that for the given systematic sampling, the largest selected number is 57. -/
theorem largest_selected_is_57 (s : SystematicSampling) 
  (h1 : s.total_elements = 60)
  (h2 : s.smallest_selected = 3)
  (h3 : s.second_smallest_selected = 9) : 
  largest_selected s = 57 := by
  sorry

end NUMINAMATH_CALUDE_largest_selected_is_57_l176_17677


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complements_A_B_l176_17641

-- Define the universal set U
def U : Set Nat := {x | 1 ≤ x ∧ x ≤ 10}

-- Define sets A and B
def A : Set Nat := {1, 2, 3, 5, 8}
def B : Set Nat := {1, 3, 5, 7, 9}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3, 5} := by sorry

-- Theorem for the intersection of complements of A and B
theorem intersection_complements_A_B : (U \ A) ∩ (U \ B) = {4, 6, 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complements_A_B_l176_17641


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l176_17611

theorem largest_divisor_of_n (n : ℕ+) 
  (h1 : (n : ℕ)^4 % 850 = 0)
  (h2 : ∀ p : ℕ, p > 20 → Nat.Prime p → (n : ℕ) % p ≠ 0) :
  ∃ k : ℕ, k ∣ (n : ℕ) ∧ k = 10 ∧ ∀ m : ℕ, m ∣ (n : ℕ) → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l176_17611


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l176_17662

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -(x + 2)^2 + 6

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- Theorem: The intersection of the parabola and y-axis is at (0, 2) -/
theorem parabola_y_axis_intersection :
  ∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ y_axis p.1 ∧ p = (0, 2) := by
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l176_17662


namespace NUMINAMATH_CALUDE_at_least_100_triangles_l176_17668

/-- Represents a configuration of lines on a plane -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_triple_intersection : Bool

/-- Calculates the number of triangular regions formed by the lines -/
def num_triangular_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem stating that for 300 lines with given conditions, there are at least 100 triangular regions -/
theorem at_least_100_triangles (config : LineConfiguration) 
  (h1 : config.num_lines = 300)
  (h2 : config.no_parallel = true)
  (h3 : config.no_triple_intersection = true) :
  num_triangular_regions config ≥ 100 := by
  sorry

end NUMINAMATH_CALUDE_at_least_100_triangles_l176_17668


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l176_17651

/-- Given a student's scores in mathematics, physics, and chemistry, prove that the difference between chemistry and physics scores is 20 marks. -/
theorem chemistry_physics_difference
  (M P C : ℕ)  -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 60)  -- Total marks in mathematics and physics is 60
  (h2 : ∃ X : ℕ, C = P + X)  -- Chemistry score is some marks more than physics
  (h3 : (M + C) / 2 = 40)  -- Average marks in mathematics and chemistry is 40
  : ∃ X : ℕ, C = P + X ∧ X = 20 := by
  sorry

#check chemistry_physics_difference

end NUMINAMATH_CALUDE_chemistry_physics_difference_l176_17651


namespace NUMINAMATH_CALUDE_first_angle_is_55_l176_17646

-- Define the triangle with the given conditions
def triangle (x : ℝ) : Prop :=
  let angle1 := x
  let angle2 := 2 * x
  let angle3 := x - 40
  (angle1 + angle2 + angle3 = 180) ∧ (angle1 > 0) ∧ (angle2 > 0) ∧ (angle3 > 0)

-- Theorem stating that the first angle is 55 degrees
theorem first_angle_is_55 : ∃ x, triangle x ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_first_angle_is_55_l176_17646


namespace NUMINAMATH_CALUDE_soccer_goals_product_l176_17622

def first_ten_games : List Nat := [2, 5, 3, 6, 2, 4, 2, 5, 1, 3]

def total_first_ten : Nat := first_ten_games.sum

theorem soccer_goals_product (g11 g12 : Nat) : 
  g11 < 8 → 
  g12 < 8 → 
  (total_first_ten + g11) % 11 = 0 → 
  (total_first_ten + g11 + g12) % 12 = 0 → 
  g11 * g12 = 49 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goals_product_l176_17622


namespace NUMINAMATH_CALUDE_cubic_root_equation_l176_17628

theorem cubic_root_equation (y : ℝ) : 
  (1 + Real.sqrt (2 * y - 3)) ^ (1/3 : ℝ) = 3 → y = 339.5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_l176_17628


namespace NUMINAMATH_CALUDE_sequence_sum_l176_17679

theorem sequence_sum (A B C D E F G H I : ℤ) : 
  E = 7 →
  A + B + C + D = 40 →
  B + C + D + E = 40 →
  C + D + E + F = 40 →
  D + E + F + G = 40 →
  E + F + G + H = 40 →
  F + G + H + I = 40 →
  A + I = 40 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l176_17679


namespace NUMINAMATH_CALUDE_gcd_of_sum_and_lcm_l176_17681

theorem gcd_of_sum_and_lcm (a b : ℕ+) (h1 : a + b = 33) (h2 : Nat.lcm a b = 90) : 
  Nat.gcd a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sum_and_lcm_l176_17681


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l176_17667

theorem simplify_trig_expression (x : ℝ) (h : 5 * π / 4 < x ∧ x < 3 * π / 2) :
  Real.sqrt (1 - 2 * Real.sin x * Real.cos x) = Real.cos x - Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l176_17667


namespace NUMINAMATH_CALUDE_cube_split_with_2023_l176_17603

theorem cube_split_with_2023 (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ), 2 * k + 1 = 2023 ∧ 
   k ≥ (m + 2) * (m - 1) / 2 - m + 1 ∧ 
   k < (m + 2) * (m - 1) / 2 + 1) → 
  m = 45 := by
sorry

end NUMINAMATH_CALUDE_cube_split_with_2023_l176_17603


namespace NUMINAMATH_CALUDE_rachel_pool_fill_time_l176_17635

/-- Represents the time (in hours) required to fill a pool -/
def fill_time (pool_capacity : ℕ) (num_hoses : ℕ) (flow_rate : ℕ) : ℕ :=
  let total_flow_per_hour := num_hoses * flow_rate * 60
  (pool_capacity + total_flow_per_hour - 1) / total_flow_per_hour

/-- Proves that it takes 33 hours to fill Rachel's pool -/
theorem rachel_pool_fill_time :
  fill_time 30000 5 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_rachel_pool_fill_time_l176_17635


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l176_17607

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a ≥ 2 * Real.sqrt 5 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ 
    a₀^2 + b₀^2 + 1/a₀^2 + 1/b₀^2 + b₀/a₀ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l176_17607


namespace NUMINAMATH_CALUDE_perimeter_semicircular_bounded_rectangle_l176_17660

/-- The perimeter of a region bounded by semicircular arcs constructed on each side of a rectangle --/
theorem perimeter_semicircular_bounded_rectangle (l w : ℝ) (hl : l = 4 / π) (hw : w = 2 / π) :
  let semicircle_length := π * l / 2
  let semicircle_width := π * w / 2
  semicircle_length + semicircle_length + semicircle_width + semicircle_width = 6 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_semicircular_bounded_rectangle_l176_17660


namespace NUMINAMATH_CALUDE_job_completion_time_l176_17659

/-- Given that:
  * A can do a job in 15 days
  * A and B working together for 4 days complete 0.4666666666666667 of the job
  Prove that B can do the job alone in 20 days -/
theorem job_completion_time (a_time : ℝ) (together_time : ℝ) (together_completion : ℝ) (b_time : ℝ) :
  a_time = 15 →
  together_time = 4 →
  together_completion = 0.4666666666666667 →
  together_completion = together_time * (1 / a_time + 1 / b_time) →
  b_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l176_17659


namespace NUMINAMATH_CALUDE_jackson_souvenir_collection_l176_17687

/-- Proves that given the conditions in Jackson's souvenir collection, 
    the number of starfish per spiral shell is 2. -/
theorem jackson_souvenir_collection 
  (hermit_crabs : ℕ) 
  (shells_per_crab : ℕ) 
  (total_souvenirs : ℕ) 
  (h1 : hermit_crabs = 45)
  (h2 : shells_per_crab = 3)
  (h3 : total_souvenirs = 450) :
  (total_souvenirs - hermit_crabs - hermit_crabs * shells_per_crab) / (hermit_crabs * shells_per_crab) = 2 :=
by sorry

end NUMINAMATH_CALUDE_jackson_souvenir_collection_l176_17687


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l176_17649

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ (n : ℕ), a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_sum : a 1 * a 5 + 2 * a 3 * a 7 + a 5 * a 9 = 16)
  (h_mean : (a 5 + a 9) / 2 = 4) :
  ∃ (q : ℝ), q > 0 ∧ (∀ (n : ℕ), a (n + 1) = q * a n) ∧ q = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l176_17649
