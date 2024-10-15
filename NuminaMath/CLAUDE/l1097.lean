import Mathlib

namespace NUMINAMATH_CALUDE_cost_of_treat_l1097_109747

/-- The cost of dog treats given daily treats, days, and total cost -/
def treat_cost (treats_per_day : ℕ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (treats_per_day * days)

/-- Theorem: The cost of each treat is $0.10 given the problem conditions -/
theorem cost_of_treat :
  let treats_per_day : ℕ := 2
  let days : ℕ := 30
  let total_cost : ℚ := 6
  treat_cost treats_per_day days total_cost = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_treat_l1097_109747


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1097_109765

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a * b > 4) ∧
  (∃ a b, a * b > 4 ∧ ¬(a > 2 ∧ b > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1097_109765


namespace NUMINAMATH_CALUDE_min_a_for_log_equation_solution_l1097_109760

theorem min_a_for_log_equation_solution : 
  ∃ (a : ℝ), a > 0 ∧ (∀ a' : ℝ, a' < a → ¬∃ x : ℝ, (Real.log (a' - 2^x) / Real.log (1/2) = 2 + x)) ∧
  (∃ x : ℝ, (Real.log (a - 2^x) / Real.log (1/2) = 2 + x)) :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_log_equation_solution_l1097_109760


namespace NUMINAMATH_CALUDE_proposition_implication_l1097_109793

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 8) : 
  ¬ P 7 := by sorry

end NUMINAMATH_CALUDE_proposition_implication_l1097_109793


namespace NUMINAMATH_CALUDE_probability_divisible_by_three_l1097_109751

/-- The set of positive integers from 1 to 2007 -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 2007}

/-- The probability that a number in S is divisible by 3 -/
def prob_div_3 : ℚ := 669 / 2007

/-- The probability that a number in S is not divisible by 3 -/
def prob_not_div_3 : ℚ := 1338 / 2007

/-- The probability that b and c satisfy the condition when a is not divisible by 3 -/
def prob_bc_condition : ℚ := 2 / 9

theorem probability_divisible_by_three :
  (prob_div_3 + prob_not_div_3 * prob_bc_condition : ℚ) = 1265 / 2007 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_three_l1097_109751


namespace NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_x_in_open_zero_one_l1097_109745

theorem arccos_lt_arcsin_iff_x_in_open_zero_one (x : ℝ) :
  x ∈ Set.Icc (-1) 1 →
  (Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo 0 1) :=
by sorry

end NUMINAMATH_CALUDE_arccos_lt_arcsin_iff_x_in_open_zero_one_l1097_109745


namespace NUMINAMATH_CALUDE_sample_size_is_six_l1097_109791

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 18 + 12 + 6

/-- Represents the sample size -/
def n : ℕ := 6

/-- Checks if a number divides the total number of teachers -/
def divides_total (k : ℕ) : Prop := k ∣ total_teachers

/-- Checks if stratified sampling works for a given sample size -/
def stratified_sampling_works (k : ℕ) : Prop :=
  k ∣ 18 ∧ k ∣ 12 ∧ k ∣ 6

/-- Checks if systematic sampling works for a given sample size -/
def systematic_sampling_works (k : ℕ) : Prop :=
  divides_total k

/-- Checks if increasing the sample size by 1 requires excluding 1 person for systematic sampling -/
def exclusion_condition (k : ℕ) : Prop :=
  ¬(divides_total (k + 1)) ∧ ((k + 1) ∣ (total_teachers - 1))

theorem sample_size_is_six :
  stratified_sampling_works n ∧
  systematic_sampling_works n ∧
  exclusion_condition n ∧
  ∀ m : ℕ, m ≠ n →
    ¬(stratified_sampling_works m ∧
      systematic_sampling_works m ∧
      exclusion_condition m) :=
sorry

end NUMINAMATH_CALUDE_sample_size_is_six_l1097_109791


namespace NUMINAMATH_CALUDE_bobby_free_throws_l1097_109716

theorem bobby_free_throws (initial_throws : ℕ) (initial_success_rate : ℚ)
  (additional_throws : ℕ) (new_success_rate : ℚ) :
  initial_throws = 30 →
  initial_success_rate = 3/5 →
  additional_throws = 10 →
  new_success_rate = 16/25 →
  ∃ (last_successful_throws : ℕ),
    last_successful_throws = 8 ∧
    (initial_success_rate * initial_throws + last_successful_throws) / 
    (initial_throws + additional_throws) = new_success_rate :=
by
  sorry

end NUMINAMATH_CALUDE_bobby_free_throws_l1097_109716


namespace NUMINAMATH_CALUDE_derivative_sin_minus_x_cos_l1097_109701

theorem derivative_sin_minus_x_cos (x : ℝ) :
  deriv (λ x => Real.sin x - x * Real.cos x) x = x * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_minus_x_cos_l1097_109701


namespace NUMINAMATH_CALUDE_number_difference_l1097_109729

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : |x - y| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1097_109729


namespace NUMINAMATH_CALUDE_unique_rational_solution_l1097_109741

theorem unique_rational_solution (x y z : ℚ) : 
  x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_rational_solution_l1097_109741


namespace NUMINAMATH_CALUDE_inverse_proportion_point_order_l1097_109756

theorem inverse_proportion_point_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-2) → y₂ = -2 / 2 → y₃ = -2 / 3 → y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_order_l1097_109756


namespace NUMINAMATH_CALUDE_complex_point_location_l1097_109721

theorem complex_point_location (x y : ℝ) 
  (h : (x + y) + (y - 1) * Complex.I = (2 * x + 3 * y) + (2 * y + 1) * Complex.I) : 
  x > 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_location_l1097_109721


namespace NUMINAMATH_CALUDE_savings_account_growth_l1097_109715

/-- Represents the total amount in a savings account after a given number of months -/
def total_amount (initial_deposit : ℝ) (monthly_rate : ℝ) (months : ℝ) : ℝ :=
  initial_deposit * (1 + monthly_rate * months)

theorem savings_account_growth (x : ℝ) :
  let initial_deposit : ℝ := 100
  let monthly_rate : ℝ := 0.006
  let y : ℝ := total_amount initial_deposit monthly_rate x
  y = 100 * (1 + 0.006 * x) ∧
  total_amount initial_deposit monthly_rate 4 = 102.4 := by
  sorry

#check savings_account_growth

end NUMINAMATH_CALUDE_savings_account_growth_l1097_109715


namespace NUMINAMATH_CALUDE_log_sum_simplification_l1097_109707

theorem log_sum_simplification : 
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
  (1 / (Real.log x / Real.log 12 + 1)) + 
  (1 / (Real.log y / Real.log 20 + 1)) + 
  (1 / (Real.log z / Real.log 8 + 1)) = 1.75 :=
by sorry

end NUMINAMATH_CALUDE_log_sum_simplification_l1097_109707


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1097_109736

theorem smallest_solution_absolute_value_equation :
  let f := fun x : ℝ => x * |x| - 3 * x + 2
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x = 0 → x₀ ≤ x ∧ x₀ = (-3 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l1097_109736


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1097_109700

theorem sum_of_solutions : ∃ (S : Finset (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ S ↔ (p.1 * p.2 = 6 * (p.1 + p.2) ∧ p.1 > 0 ∧ p.2 > 0)) ∧ 
  (S.sum (λ p => p.1 + p.2) = 290) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1097_109700


namespace NUMINAMATH_CALUDE_intersecting_chords_area_theorem_l1097_109754

/-- Represents a circle with two intersecting chords -/
structure IntersectingChordsCircle where
  radius : ℝ
  chord_length : ℝ
  intersection_distance : ℝ

/-- Represents the area of a region in the form m*π - n*√d -/
structure RegionArea where
  m : ℕ
  n : ℕ
  d : ℕ

/-- Checks if a number is square-free (not divisible by the square of any prime) -/
def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p) ∣ n → p = 1

/-- The main theorem statement -/
theorem intersecting_chords_area_theorem (circle : IntersectingChordsCircle)
  (h1 : circle.radius = 50)
  (h2 : circle.chord_length = 90)
  (h3 : circle.intersection_distance = 24) :
  ∃ (area : RegionArea), 
    (area.m > 0 ∧ area.n > 0 ∧ area.d > 0) ∧
    is_square_free area.d ∧
    ∃ (region_area : ℝ), region_area = area.m * Real.pi - area.n * Real.sqrt area.d :=
by sorry

end NUMINAMATH_CALUDE_intersecting_chords_area_theorem_l1097_109754


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l1097_109749

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) 
  (h1 : total_players = 25) 
  (h2 : goalies = 4) 
  (h3 : goalies ≤ total_players) : 
  goalies * (total_players - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l1097_109749


namespace NUMINAMATH_CALUDE_square_root_condition_l1097_109710

theorem square_root_condition (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3*x - 5) ↔ x ≥ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_condition_l1097_109710


namespace NUMINAMATH_CALUDE_parallelogram_sides_sum_l1097_109723

theorem parallelogram_sides_sum (x y : ℚ) : 
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -8/5 := by sorry

end NUMINAMATH_CALUDE_parallelogram_sides_sum_l1097_109723


namespace NUMINAMATH_CALUDE_train_length_l1097_109789

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 6 → ∃ length : ℝ, abs (length - 100.02) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1097_109789


namespace NUMINAMATH_CALUDE_all_numbers_equal_l1097_109768

/-- Represents a 10x10 table of real numbers -/
def Table := Fin 10 → Fin 10 → ℝ

/-- Predicate to check if a number is underlined in its row -/
def is_underlined_in_row (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, t i j ≥ t i k

/-- Predicate to check if a number is underlined in its column -/
def is_underlined_in_col (t : Table) (i j : Fin 10) : Prop :=
  ∀ k : Fin 10, t i j ≤ t k j

/-- Predicate to check if a number is underlined exactly twice -/
def is_underlined_twice (t : Table) (i j : Fin 10) : Prop :=
  is_underlined_in_row t i j ∧ is_underlined_in_col t i j

theorem all_numbers_equal (t : Table) 
  (h : ∀ i j : Fin 10, is_underlined_in_row t i j ∨ is_underlined_in_col t i j → is_underlined_twice t i j) :
  ∀ i j k l : Fin 10, t i j = t k l :=
sorry

end NUMINAMATH_CALUDE_all_numbers_equal_l1097_109768


namespace NUMINAMATH_CALUDE_quadratic_roots_l1097_109784

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 4*x - 21 = 0) ↔ (x = 3 ∨ x = -7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1097_109784


namespace NUMINAMATH_CALUDE_lynn_in_fourth_car_l1097_109740

-- Define the set of people
inductive Person : Type
| Trent : Person
| Jamie : Person
| Eden : Person
| Lynn : Person
| Mira : Person
| Cory : Person

-- Define the seating arrangement
def SeatingArrangement := Fin 6 → Person

-- Define the conditions of the seating arrangement
def ValidArrangement (s : SeatingArrangement) : Prop :=
  -- Trent is in the lead car
  s 0 = Person.Trent ∧
  -- Eden is directly behind Jamie
  (∃ i : Fin 5, s i = Person.Jamie ∧ s (i + 1) = Person.Eden) ∧
  -- Lynn sits ahead of Mira
  (∃ i j : Fin 6, i < j ∧ s i = Person.Lynn ∧ s j = Person.Mira) ∧
  -- Mira is not in the last car
  s 5 ≠ Person.Mira ∧
  -- At least two people sit between Cory and Lynn
  (∃ i j : Fin 6, |i - j| > 2 ∧ s i = Person.Cory ∧ s j = Person.Lynn)

-- The theorem to prove
theorem lynn_in_fourth_car (s : SeatingArrangement) :
  ValidArrangement s → s 3 = Person.Lynn :=
by sorry

end NUMINAMATH_CALUDE_lynn_in_fourth_car_l1097_109740


namespace NUMINAMATH_CALUDE_zachary_pushup_count_l1097_109720

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The difference between Zachary's and David's push-ups -/
def pushup_difference : ℕ := 7

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := david_pushups + pushup_difference

theorem zachary_pushup_count : zachary_pushups = 51 := by
  sorry

end NUMINAMATH_CALUDE_zachary_pushup_count_l1097_109720


namespace NUMINAMATH_CALUDE_at_least_one_composite_l1097_109774

theorem at_least_one_composite (a b c k : ℕ) 
  (ha : a ≥ 3) (hb : b ≥ 3) (hc : c ≥ 3) 
  (heq : a * b * c = k^2 + 1) : 
  ¬(Nat.Prime (a - 1) ∧ Nat.Prime (b - 1) ∧ Nat.Prime (c - 1)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_composite_l1097_109774


namespace NUMINAMATH_CALUDE_sum_abcd_is_negative_six_l1097_109705

theorem sum_abcd_is_negative_six 
  (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_is_negative_six_l1097_109705


namespace NUMINAMATH_CALUDE_product_positive_not_imply_both_positive_l1097_109785

theorem product_positive_not_imply_both_positive : ∃ (a b : ℝ), a * b > 0 ∧ ¬(a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_product_positive_not_imply_both_positive_l1097_109785


namespace NUMINAMATH_CALUDE_negative_a_exponent_division_l1097_109718

theorem negative_a_exponent_division (a : ℝ) : (-a)^10 / (-a)^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_exponent_division_l1097_109718


namespace NUMINAMATH_CALUDE_hat_problem_l1097_109739

/-- Proves that given the conditions of the hat problem, the number of green hats is 30 -/
theorem hat_problem (total_hats : ℕ) (blue_price green_price : ℕ) (total_price : ℕ)
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 540) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 30 :=
by sorry

end NUMINAMATH_CALUDE_hat_problem_l1097_109739


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l1097_109797

/-- The number of lamp posts along the alley -/
def total_posts : ℕ := 400

/-- The lamp post number where Alla is observed -/
def alla_observed : ℕ := 55

/-- The lamp post number where Boris is observed -/
def boris_observed : ℕ := 321

/-- The function to calculate the meeting point of Alla and Boris -/
def meeting_point : ℕ :=
  let alla_traveled := alla_observed - 1
  let boris_traveled := total_posts - boris_observed
  let total_traveled := alla_traveled + boris_traveled
  let alla_to_meeting := 3 * alla_traveled
  1 + alla_to_meeting

/-- Theorem stating that Alla and Boris will meet at lamp post 163 -/
theorem alla_boris_meeting :
  meeting_point = 163 := by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l1097_109797


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l1097_109799

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 3 * x - 1

-- Define the solution set
def solution_set : Set ℝ := {x | f x > 0}

-- Theorem statement
theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l1097_109799


namespace NUMINAMATH_CALUDE_evaluate_f_l1097_109762

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_f : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l1097_109762


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1097_109782

theorem min_value_sum_reciprocals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 5) : 
  (1/x + 4/y + 9/z) ≥ 36/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1097_109782


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1097_109778

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1097_109778


namespace NUMINAMATH_CALUDE_digit_difference_after_reversal_l1097_109702

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the digits of a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  h_hundreds := n.h_units
  h_tens := n.h_tens
  h_units := n.h_hundreds

theorem digit_difference_after_reversal
  (numbers : Finset ThreeDigitNumber)
  (reversed : ThreeDigitNumber)
  (h_count : numbers.card = 10)
  (h_reversed_in : reversed ∈ numbers)
  (h_average_increase : (numbers.sum value + value (reverse reversed) - value reversed) / 10 - numbers.sum value / 10 = 198 / 10) :
  (reverse reversed).units - (reverse reversed).hundreds = 2 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_after_reversal_l1097_109702


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1097_109796

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 2 / (1 + Complex.I)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1097_109796


namespace NUMINAMATH_CALUDE_box_surface_area_l1097_109722

/-- Calculates the interior surface area of a box formed by removing square corners from a rectangular sheet -/
def interior_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- The interior surface area of the box is 731 square units -/
theorem box_surface_area : 
  interior_surface_area 25 35 6 = 731 := by sorry

end NUMINAMATH_CALUDE_box_surface_area_l1097_109722


namespace NUMINAMATH_CALUDE_painting_cost_is_84_l1097_109780

/-- Calculates the cost of painting house numbers on a street --/
def cost_of_painting (houses_per_side : ℕ) (south_start : ℕ) (north_start : ℕ) (increment : ℕ) : ℕ :=
  let south_end := south_start + increment * (houses_per_side - 1)
  let north_end := north_start + increment * (houses_per_side - 1)
  let south_cost := (houses_per_side - (south_end / 100)) + (south_end / 100)
  let north_cost := (houses_per_side - (north_end / 100)) + (north_end / 100)
  south_cost + north_cost

/-- The total cost of painting house numbers on the street is 84 dollars --/
theorem painting_cost_is_84 :
  cost_of_painting 30 5 6 6 = 84 :=
by sorry

end NUMINAMATH_CALUDE_painting_cost_is_84_l1097_109780


namespace NUMINAMATH_CALUDE_ball_probability_l1097_109770

theorem ball_probability (m : ℕ) : 
  (8 : ℝ) / (8 + m) > (m : ℝ) / (8 + m) → m < 8 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1097_109770


namespace NUMINAMATH_CALUDE_morning_milk_calculation_l1097_109703

/-- The number of gallons of milk Aunt May got this morning -/
def morning_milk : ℕ := 365

/-- The number of gallons of milk Aunt May got in the evening -/
def evening_milk : ℕ := 380

/-- The number of gallons of milk Aunt May sold -/
def sold_milk : ℕ := 612

/-- The number of gallons of milk left over from yesterday -/
def leftover_milk : ℕ := 15

/-- The number of gallons of milk remaining -/
def remaining_milk : ℕ := 148

/-- Theorem stating that the morning milk calculation is correct -/
theorem morning_milk_calculation :
  morning_milk + evening_milk + leftover_milk - sold_milk = remaining_milk :=
by sorry

end NUMINAMATH_CALUDE_morning_milk_calculation_l1097_109703


namespace NUMINAMATH_CALUDE_power_function_decreasing_m_l1097_109727

/-- A function f: ℝ → ℝ is a power function if it has the form f(x) = ax^b for some constants a and b, where a ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, x > 0 → f x = a * x ^ b

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if for any x₁, x₂ ∈ (0, +∞) with x₁ < x₂, we have f(x₁) > f(x₂) -/
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂

/-- The main theorem -/
theorem power_function_decreasing_m (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * x^m
  IsPowerFunction f ∧ IsDecreasingOn f → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_m_l1097_109727


namespace NUMINAMATH_CALUDE_fraction_addition_l1097_109731

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1097_109731


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l1097_109744

theorem x_squared_less_than_abs_x (x : ℝ) :
  x^2 < |x| ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l1097_109744


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1097_109767

theorem largest_solution_of_equation (x : ℝ) :
  (x / 3 + 1 / (3 * x) = 1 / 2) → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1097_109767


namespace NUMINAMATH_CALUDE_computer_price_reduction_l1097_109735

/-- Given a computer price reduction of 40% resulting in a final price of 'a' yuan,
    prove that the original price was (5/3)a yuan. -/
theorem computer_price_reduction (a : ℝ) : 
  (∃ (original_price : ℝ), 
    original_price * (1 - 0.4) = a ∧ 
    original_price = (5/3) * a) :=
by sorry

end NUMINAMATH_CALUDE_computer_price_reduction_l1097_109735


namespace NUMINAMATH_CALUDE_tim_needs_72_keys_l1097_109795

/-- The number of keys Tim needs to make for his rental properties -/
def total_keys (num_complexes : ℕ) (apartments_per_complex : ℕ) (keys_per_apartment : ℕ) : ℕ :=
  num_complexes * apartments_per_complex * keys_per_apartment

/-- Theorem stating that Tim needs 72 keys for his rental properties -/
theorem tim_needs_72_keys :
  total_keys 2 12 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tim_needs_72_keys_l1097_109795


namespace NUMINAMATH_CALUDE_x_squared_minus_x_greater_cube_sum_greater_l1097_109711

-- Part 1
theorem x_squared_minus_x_greater (x : ℝ) : x^2 - x > x - 2 := by sorry

-- Part 2
theorem cube_sum_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by sorry

end NUMINAMATH_CALUDE_x_squared_minus_x_greater_cube_sum_greater_l1097_109711


namespace NUMINAMATH_CALUDE_height_of_cylinder_A_l1097_109750

/-- Theorem: Height of Cylinder A given volume ratio with Cylinder B -/
theorem height_of_cylinder_A (r_A r_B h_B : ℝ) 
  (h_circum_A : 2 * Real.pi * r_A = 8)
  (h_circum_B : 2 * Real.pi * r_B = 10)
  (h_height_B : h_B = 8)
  (h_volume_ratio : Real.pi * r_A^2 * (7 : ℝ) = 0.5600000000000001 * Real.pi * r_B^2 * h_B) :
  ∃ h_A : ℝ, h_A = 7 ∧ Real.pi * r_A^2 * h_A = 0.5600000000000001 * Real.pi * r_B^2 * h_B := by
  sorry

end NUMINAMATH_CALUDE_height_of_cylinder_A_l1097_109750


namespace NUMINAMATH_CALUDE_carpooling_distance_ratio_l1097_109753

def distance_to_first_friend : ℝ := 8

def distance_to_second_friend : ℝ := 4

def distance_to_work (d1 d2 : ℝ) : ℝ := 3 * (d1 + d2)

theorem carpooling_distance_ratio :
  distance_to_work distance_to_first_friend distance_to_second_friend = 36 →
  distance_to_second_friend / distance_to_first_friend = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carpooling_distance_ratio_l1097_109753


namespace NUMINAMATH_CALUDE_pythagorean_number_existence_l1097_109738

theorem pythagorean_number_existence (n : ℕ) (hn : n > 12) :
  ∃ (a b c P : ℕ), a > b ∧ b > 0 ∧ c > 0 ∧
  P = a * b * (a^2 - b^2) * c^2 ∧
  n < P ∧ P < 2 * n :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_number_existence_l1097_109738


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1097_109758

theorem solution_set_reciprocal_inequality (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1097_109758


namespace NUMINAMATH_CALUDE_b_minus_d_squared_l1097_109759

theorem b_minus_d_squared (a b c d : ℤ) 
  (eq1 : a - b - c + d = 12)
  (eq2 : a + b - c - d = 6) : 
  (b - d)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_b_minus_d_squared_l1097_109759


namespace NUMINAMATH_CALUDE_adults_count_is_21_l1097_109788

/-- Represents the trekking group and meal information -/
structure TrekkingGroup where
  childrenCount : ℕ
  adultMealCapacity : ℕ
  childrenMealCapacity : ℕ
  remainingChildrenCapacity : ℕ
  adultsMealCount : ℕ

/-- Theorem stating that the number of adults in the trekking group is 21 -/
theorem adults_count_is_21 (group : TrekkingGroup)
  (h1 : group.childrenCount = 70)
  (h2 : group.adultMealCapacity = 70)
  (h3 : group.childrenMealCapacity = 90)
  (h4 : group.remainingChildrenCapacity = 63)
  (h5 : group.adultsMealCount = 21) :
  group.adultsMealCount = 21 := by
  sorry

#check adults_count_is_21

end NUMINAMATH_CALUDE_adults_count_is_21_l1097_109788


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l1097_109794

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def intersects_x_axis_at (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  (p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2 = c.radius^2 ∧
  (p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2 = c.radius^2 ∧
  p1.2 = 0 ∧ p2.2 = 0

def tangent_to_line (c : Circle) : Prop :=
  ∃ (x y : ℝ), (x - y + 1 = 0) ∧
  ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (|x - y + 1| / Real.sqrt 2 = c.radius)

-- State the theorem
theorem circle_radius_is_sqrt_2 (c : Circle) :
  is_in_first_quadrant c.center →
  intersects_x_axis_at c (1, 0) (3, 0) →
  tangent_to_line c →
  c.radius = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_2_l1097_109794


namespace NUMINAMATH_CALUDE_sravans_journey_l1097_109766

/-- Calculates the total distance traveled given the conditions of Sravan's journey -/
theorem sravans_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 15 ∧ speed1 = 45 ∧ speed2 = 30 → 
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧
    distance = 540 := by
  sorry


end NUMINAMATH_CALUDE_sravans_journey_l1097_109766


namespace NUMINAMATH_CALUDE_third_divisor_is_seventeen_l1097_109708

theorem third_divisor_is_seventeen : ∃ (d : ℕ), d = 17 ∧ d > 11 ∧ 
  (3374 % 9 = 8) ∧ (3374 % 11 = 8) ∧ (3374 % d = 8) ∧
  (∀ (x : ℕ), x > 11 ∧ x < d → (3374 % x ≠ 8)) :=
by sorry

end NUMINAMATH_CALUDE_third_divisor_is_seventeen_l1097_109708


namespace NUMINAMATH_CALUDE_orange_profit_problem_l1097_109783

/-- Represents the fruit vendor's orange selling problem -/
theorem orange_profit_problem 
  (buy_quantity : ℕ) 
  (buy_price : ℚ) 
  (sell_quantity : ℕ) 
  (sell_price : ℚ) 
  (target_profit : ℚ) :
  buy_quantity = 8 →
  buy_price = 15 →
  sell_quantity = 6 →
  sell_price = 18 →
  target_profit = 150 →
  ∃ (n : ℕ), 
    n * (sell_price / sell_quantity - buy_price / buy_quantity) ≥ target_profit ∧
    ∀ (m : ℕ), m * (sell_price / sell_quantity - buy_price / buy_quantity) ≥ target_profit → m ≥ n ∧
    n = 134 :=
sorry

end NUMINAMATH_CALUDE_orange_profit_problem_l1097_109783


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1097_109712

/-- The quadratic equation x^2 + 6mx + m has exactly one real root if and only if m = 1/9 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 + 6*m*x + m = 0) ↔ m = 1/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1097_109712


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1097_109742

theorem two_digit_number_sum (x : ℕ) : 
  x < 10 →                             -- units digit is less than 10
  (11 * x + 30) % (2 * x + 3) = 3 →    -- remainder is 3
  (11 * x + 30) / (2 * x + 3) = 7 →    -- quotient is 7
  2 * x + 3 = 7 :=                     -- sum of digits is 7
by sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1097_109742


namespace NUMINAMATH_CALUDE_concert_ticket_problem_l1097_109773

def ticket_price_possibilities (seventh_grade_total eighth_grade_total : ℕ) : ℕ :=
  (Finset.filter (fun x => seventh_grade_total % x = 0 ∧ eighth_grade_total % x = 0)
    (Finset.range (min seventh_grade_total eighth_grade_total + 1))).card

theorem concert_ticket_problem : ticket_price_possibilities 36 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_problem_l1097_109773


namespace NUMINAMATH_CALUDE_decreasing_product_function_properties_l1097_109714

/-- A decreasing function f defined on (0, +∞) satisfying f(x) + f(y) = f(xy) -/
def DecreasingProductFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 0 ∧ y > 0 → f x + f y = f (x * y)) ∧
  (∀ x y, x > 0 ∧ y > 0 ∧ x < y → f x > f y)

theorem decreasing_product_function_properties
  (f : ℝ → ℝ)
  (h : DecreasingProductFunction f)
  (h_f4 : f 4 = -4) :
  (∀ x y, x > 0 ∧ y > 0 → f x - f y = f (x / y)) ∧
  (Set.Ioo 12 16 : Set ℝ) = {x | f x - f (1 / (x - 12)) ≥ -12} := by
  sorry

end NUMINAMATH_CALUDE_decreasing_product_function_properties_l1097_109714


namespace NUMINAMATH_CALUDE_unique_a_value_l1097_109786

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 4^x else 2^(a - x)

-- State the theorem
theorem unique_a_value (a : ℝ) (h1 : a ≠ 1) :
  f a (1 - a) = f a (a - 1) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1097_109786


namespace NUMINAMATH_CALUDE_coin_problem_l1097_109781

theorem coin_problem (x y : ℕ) : 
  x + y = 40 →
  2 * x + 5 * y = 125 →
  y = 15 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l1097_109781


namespace NUMINAMATH_CALUDE_quadratic_one_solution_negative_k_value_l1097_109790

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x, 9 * x^2 + k * x + 36 = 0) ↔ k = 36 ∨ k = -36 :=
by sorry

theorem negative_k_value (k : ℝ) : 
  (∃! x, 9 * x^2 + k * x + 36 = 0) → k = -36 ∨ k = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_negative_k_value_l1097_109790


namespace NUMINAMATH_CALUDE_all_nat_gt2_as_fib_sum_l1097_109775

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define a function to check if a number is in the Fibonacci sequence
def isFib (n : ℕ) : Prop :=
  ∃ k, fib k = n

-- Define a function to represent a number as a sum of distinct Fibonacci numbers
def representAsFibSum (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ), (∀ x ∈ S, isFib x) ∧ (S.sum id = n)

-- The main theorem
theorem all_nat_gt2_as_fib_sum :
  ∀ n : ℕ, n > 2 → representAsFibSum n :=
by
  sorry


end NUMINAMATH_CALUDE_all_nat_gt2_as_fib_sum_l1097_109775


namespace NUMINAMATH_CALUDE_sector_arc_length_l1097_109779

theorem sector_arc_length (θ : Real) (r : Real) (l : Real) : 
  θ = 2 * Real.pi / 3 → r = 2 → l = θ * r → l = 4 * Real.pi / 3 := by
  sorry

#check sector_arc_length

end NUMINAMATH_CALUDE_sector_arc_length_l1097_109779


namespace NUMINAMATH_CALUDE_sticker_distribution_theorem_l1097_109737

def distribute_stickers (total_stickers : ℕ) (num_sheets : ℕ) : ℕ :=
  sorry

theorem sticker_distribution_theorem :
  distribute_stickers 10 5 = 126 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_theorem_l1097_109737


namespace NUMINAMATH_CALUDE_greatest_seven_digit_divisible_by_lcm_l1097_109726

def is_seven_digit (n : ℕ) : Prop := n ≥ 1000000 ∧ n ≤ 9999999

def lcm_primes : ℕ := 41 * 43 * 47 * 53

theorem greatest_seven_digit_divisible_by_lcm :
  ∀ n : ℕ, is_seven_digit n → n % lcm_primes = 0 → n ≤ 8833702 := by sorry

end NUMINAMATH_CALUDE_greatest_seven_digit_divisible_by_lcm_l1097_109726


namespace NUMINAMATH_CALUDE_adjacent_nonadjacent_probability_l1097_109704

def num_students : ℕ := 5

def total_arrangements : ℕ := num_students.factorial

def valid_arrangements : ℕ := 24

theorem adjacent_nonadjacent_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_adjacent_nonadjacent_probability_l1097_109704


namespace NUMINAMATH_CALUDE_john_mean_score_l1097_109761

def john_scores : List ℝ := [88, 92, 94, 86, 90, 85]

theorem john_mean_score :
  (john_scores.sum / john_scores.length : ℝ) = 535 / 6 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l1097_109761


namespace NUMINAMATH_CALUDE_gcd_102_238_l1097_109719

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l1097_109719


namespace NUMINAMATH_CALUDE_hat_wearers_count_l1097_109769

theorem hat_wearers_count (total_people adults children : ℕ)
  (adult_women adult_men : ℕ)
  (women_hat_percentage men_hat_percentage children_hat_percentage : ℚ) :
  total_people = adults + children →
  adults = adult_women + adult_men →
  adult_women = adult_men →
  women_hat_percentage = 25 / 100 →
  men_hat_percentage = 12 / 100 →
  children_hat_percentage = 10 / 100 →
  adults = 1800 →
  children = 200 →
  (adult_women * women_hat_percentage).floor +
  (adult_men * men_hat_percentage).floor +
  (children * children_hat_percentage).floor = 353 := by
sorry

end NUMINAMATH_CALUDE_hat_wearers_count_l1097_109769


namespace NUMINAMATH_CALUDE_limes_picked_equals_total_l1097_109717

/-- The number of limes picked by Alyssa -/
def alyssas_limes : ℕ := 25

/-- The number of limes picked by Mike -/
def mikes_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := 57

/-- Theorem: The sum of limes picked by Alyssa and Mike equals the total number of limes picked -/
theorem limes_picked_equals_total : alyssas_limes + mikes_limes = total_limes := by
  sorry

end NUMINAMATH_CALUDE_limes_picked_equals_total_l1097_109717


namespace NUMINAMATH_CALUDE_nut_count_theorem_l1097_109713

def total_pistachios : ℕ := 80
def total_almonds : ℕ := 60
def total_cashews : ℕ := 40

def pistachio_shell_ratio : ℚ := 95 / 100
def pistachio_opened_ratio : ℚ := 75 / 100

def almond_shell_ratio : ℚ := 90 / 100
def almond_cracked_ratio : ℚ := 80 / 100

def cashew_shell_ratio : ℚ := 85 / 100
def cashew_salted_ratio : ℚ := 70 / 100
def cashew_opened_ratio : ℚ := 60 / 100

theorem nut_count_theorem :
  let pistachios_opened := ⌊(total_pistachios : ℚ) * pistachio_shell_ratio * pistachio_opened_ratio⌋
  let almonds_cracked := ⌊(total_almonds : ℚ) * almond_shell_ratio * almond_cracked_ratio⌋
  let cashews_opened := ⌊(total_cashews : ℚ) * cashew_shell_ratio * cashew_opened_ratio⌋
  let total_opened_cracked := pistachios_opened + almonds_cracked + cashews_opened
  let shelled_salted_cashews := ⌊(total_cashews : ℚ) * cashew_shell_ratio * cashew_salted_ratio⌋
  total_opened_cracked = 120 ∧ shelled_salted_cashews = 23 := by
  sorry

end NUMINAMATH_CALUDE_nut_count_theorem_l1097_109713


namespace NUMINAMATH_CALUDE_height_difference_l1097_109743

/-- Given the heights of three siblings, prove the height difference between two of them. -/
theorem height_difference (cary_height bill_height jan_height : ℕ) :
  cary_height = 72 →
  bill_height = cary_height / 2 →
  jan_height = 42 →
  jan_height - bill_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l1097_109743


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1097_109763

def alice_number : ℕ := 45

def is_valid_bob_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ alice_number → p^2 ∣ n) ∧ (p ∣ n → p ∣ alice_number)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), is_valid_bob_number bob_number ∧
    ∀ (m : ℕ), is_valid_bob_number m → bob_number ≤ m ∧ bob_number = 2025 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1097_109763


namespace NUMINAMATH_CALUDE_complex_sum_difference_l1097_109772

theorem complex_sum_difference (A M S : ℂ) (P : ℝ) 
  (hA : A = 3 - 2*I) 
  (hM : M = -5 + 3*I) 
  (hS : S = -2*I) 
  (hP : P = 3) : 
  A + M + S - P = -5 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_difference_l1097_109772


namespace NUMINAMATH_CALUDE_rose_ratio_l1097_109776

theorem rose_ratio (total : ℕ) (tulips : ℕ) (carnations : ℕ) :
  total = 40 ∧ tulips = 10 ∧ carnations = 14 →
  (total - tulips - carnations : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rose_ratio_l1097_109776


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1097_109792

/-- An arithmetic sequence with first term 3 and common difference 5 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + (n - 1) * 5

/-- The 150th term of the arithmetic sequence is 748 -/
theorem arithmetic_sequence_150th_term : arithmeticSequence 150 = 748 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1097_109792


namespace NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1097_109733

theorem complex_cube_sum_ratio (x y z : ℂ) 
  (hnonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (hsum : x + y + z = 30)
  (hdiff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1097_109733


namespace NUMINAMATH_CALUDE_minimum_discount_for_profit_margin_l1097_109706

theorem minimum_discount_for_profit_margin 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (min_profit_margin : ℝ) 
  (discount : ℝ) :
  cost_price = 800 →
  marked_price = 1200 →
  min_profit_margin = 0.2 →
  discount = 0.08 →
  marked_price * (1 - discount) ≥ cost_price * (1 + min_profit_margin) ∧
  ∀ d : ℝ, d < discount → marked_price * (1 - d) < cost_price * (1 + min_profit_margin) :=
by sorry

end NUMINAMATH_CALUDE_minimum_discount_for_profit_margin_l1097_109706


namespace NUMINAMATH_CALUDE_sector_area_l1097_109732

/-- Given a circular sector with a central angle of 2 radians and an arc length of 4 cm,
    the area of the sector is 4 cm². -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (h1 : θ = 2) (h2 : arc_length = 4) :
  (1/2) * arc_length * (arc_length / θ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1097_109732


namespace NUMINAMATH_CALUDE_area_midpoint_triangle_is_sqrt3_l1097_109755

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- The triangle formed by connecting midpoints of three adjacent regular hexagons -/
structure MidpointTriangle :=
  (hexagon1 : RegularHexagon)
  (hexagon2 : RegularHexagon)
  (hexagon3 : RegularHexagon)
  (are_adjacent : hexagon1 ≠ hexagon2 ∧ hexagon2 ≠ hexagon3 ∧ hexagon3 ≠ hexagon1)

/-- The area of the triangle formed by connecting midpoints of three adjacent regular hexagons -/
def area_midpoint_triangle (t : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the midpoint triangle is √3 -/
theorem area_midpoint_triangle_is_sqrt3 (t : MidpointTriangle) : 
  area_midpoint_triangle t = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_midpoint_triangle_is_sqrt3_l1097_109755


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l1097_109757

theorem max_value_of_sum_of_squares (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -1/2)
  (y_ge : y ≥ -3/2)
  (z_ge : z ≥ -1) :
  Real.sqrt (3 * x + 1.5) + Real.sqrt (3 * y + 4.5) + Real.sqrt (3 * z + 3) ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l1097_109757


namespace NUMINAMATH_CALUDE_y_fourth_power_zero_l1097_109752

theorem y_fourth_power_zero (y : ℝ) (hy : y > 0) 
  (h : Real.sqrt (1 - y^2) + Real.sqrt (1 + y^2) = 2) : y^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_fourth_power_zero_l1097_109752


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1097_109724

theorem largest_integer_in_interval : ∃ x : ℤ, 
  (1 / 4 : ℚ) < (x : ℚ) / 9 ∧ 
  (x : ℚ) / 9 < (7 / 9 : ℚ) ∧ 
  ∀ y : ℤ, ((1 / 4 : ℚ) < (y : ℚ) / 9 ∧ (y : ℚ) / 9 < (7 / 9 : ℚ)) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1097_109724


namespace NUMINAMATH_CALUDE_sin_cube_identity_l1097_109746

theorem sin_cube_identity (θ : ℝ) :
  ∃! (c d : ℝ), ∀ θ, Real.sin θ ^ 3 = c * Real.sin (3 * θ) + d * Real.sin θ :=
by
  -- The unique pair (c, d) is (-1/4, 3/4)
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l1097_109746


namespace NUMINAMATH_CALUDE_candy_calculation_l1097_109734

/-- Calculates the number of candy pieces Faye's sister gave her --/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

theorem candy_calculation (initial eaten final : ℕ) 
  (h1 : initial ≥ eaten) 
  (h2 : final ≥ initial - eaten) : 
  candy_from_sister initial eaten final = final - (initial - eaten) :=
by
  sorry

#eval candy_from_sister 47 25 62  -- Should output 40

end NUMINAMATH_CALUDE_candy_calculation_l1097_109734


namespace NUMINAMATH_CALUDE_Φ_is_connected_Φ_single_part_l1097_109728

/-- The set of points (x, y) in R^2 satisfying the given system of inequalities -/
def Φ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt (x^2 - 3*y^2 + 4*x + 4) ≤ 2*x + 1 ∧
               x^2 + y^2 ≤ 4}

/-- Theorem stating that Φ is a connected set -/
theorem Φ_is_connected : IsConnected Φ := by
  sorry

/-- Corollary stating that Φ consists of a single part -/
theorem Φ_single_part : ∃! (S : Set (ℝ × ℝ)), S = Φ ∧ IsConnected S := by
  sorry

end NUMINAMATH_CALUDE_Φ_is_connected_Φ_single_part_l1097_109728


namespace NUMINAMATH_CALUDE_equation_solution_l1097_109771

theorem equation_solution : (25 - 7 = 3 + x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1097_109771


namespace NUMINAMATH_CALUDE_circumcenter_property_implies_isosceles_l1097_109798

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector addition and scalar multiplication
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = (t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2 ∨
  (t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2 = (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 ∨
  (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2

-- The main theorem
theorem circumcenter_property_implies_isosceles (t : Triangle) :
  let O := circumcenter t
  vec_add (vec_add (vec_add O (vec_scale (-1) t.A)) (vec_add O (vec_scale (-1) t.B))) (vec_scale (Real.sqrt 2) (vec_add O (vec_scale (-1) t.C))) = (0, 0)
  → is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_circumcenter_property_implies_isosceles_l1097_109798


namespace NUMINAMATH_CALUDE_total_turtles_is_30_l1097_109709

/-- The number of turtles Kristen has -/
def kristens_turtles : ℕ := 12

/-- The number of turtles Kris has -/
def kris_turtles : ℕ := kristens_turtles / 4

/-- The number of turtles Trey has -/
def treys_turtles : ℕ := 5 * kris_turtles

/-- The total number of turtles -/
def total_turtles : ℕ := kristens_turtles + kris_turtles + treys_turtles

theorem total_turtles_is_30 : total_turtles = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_is_30_l1097_109709


namespace NUMINAMATH_CALUDE_modulus_of_x_minus_yi_l1097_109730

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def complex_equation (x y : ℝ) : Prop :=
  (x + Real.sqrt 2 * i) / i = y + i

-- Theorem statement
theorem modulus_of_x_minus_yi (x y : ℝ) 
  (h : complex_equation x y) : Complex.abs (x - y * i) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_x_minus_yi_l1097_109730


namespace NUMINAMATH_CALUDE_free_square_positions_l1097_109777

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the rectangle size
def RectangleSize := (3, 1)

-- Define the number of rectangles
def NumRectangles := 21

-- Define the possible free square positions
def FreePosns : List (Fin 8 × Fin 8) := [(3, 3), (3, 6), (6, 3), (6, 6)]

-- Theorem statement
theorem free_square_positions (board : Chessboard) (rectangles : Fin NumRectangles → Chessboard) :
  (∃! pos : Chessboard, pos ∉ (rectangles '' univ)) →
  (∃ pos ∈ FreePosns, pos ∉ (rectangles '' univ)) :=
sorry

end NUMINAMATH_CALUDE_free_square_positions_l1097_109777


namespace NUMINAMATH_CALUDE_a_squared_b_irrational_l1097_109748

def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem a_squared_b_irrational 
  (a b : ℝ) 
  (h_a_rational : is_rational a) 
  (h_b_irrational : ¬ is_rational b) 
  (h_ab_rational : is_rational (a * b)) : 
  ¬ is_rational (a^2 * b) :=
sorry

end NUMINAMATH_CALUDE_a_squared_b_irrational_l1097_109748


namespace NUMINAMATH_CALUDE_book_discount_percentage_l1097_109787

def original_price : ℝ := 60
def discounted_price : ℝ := 45
def discount : ℝ := 15
def tax_rate : ℝ := 0.1

theorem book_discount_percentage :
  (discount / original_price) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_discount_percentage_l1097_109787


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1097_109764

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (2*x + y) + y / (2*y + z) + z / (2*z + x) ≤ 1 := by
  sorry

#check max_value_of_expression

end NUMINAMATH_CALUDE_max_value_of_expression_l1097_109764


namespace NUMINAMATH_CALUDE_spade_operation_result_l1097_109725

-- Define the spade operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_operation_result : spade 2 (spade 4 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_operation_result_l1097_109725
