import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sample_size_l638_63810

/-- Represents the structure of a company's workforce -/
structure Company where
  staff : ℕ
  middle_managers : ℕ
  senior_managers : ℕ

/-- Calculates the total number of employees in the company -/
def Company.total (c : Company) : ℕ := c.staff + c.middle_managers + c.senior_managers

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  company : Company
  sample_size : ℕ
  selected_senior_managers : ℕ

/-- Theorem stating the correct sample size for the given conditions -/
theorem stratified_sample_size
  (c : Company)
  (sample : StratifiedSample)
  (h1 : c.staff = 160)
  (h2 : c.middle_managers = 30)
  (h3 : c.senior_managers = 10)
  (h4 : sample.company = c)
  (h5 : sample.selected_senior_managers = 1) :
  sample.sample_size = 20 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_size_l638_63810


namespace NUMINAMATH_CALUDE_common_roots_product_l638_63844

-- Define the two polynomial functions
def f (x : ℝ) : ℝ := x^3 + 3*x + 20
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 80

-- Define the property of having common roots
def has_common_roots (p q : ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ p x = 0 ∧ p y = 0 ∧ q x = 0 ∧ q y = 0

-- Theorem statement
theorem common_roots_product :
  has_common_roots f g →
  ∃ (x y : ℝ), x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ g x = 0 ∧ g y = 0 ∧ x * y = 20 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l638_63844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l638_63872

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = 16 and a₉ = 80,
    prove that a₆ = 48. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_3 : a 3 = 16)
    (h_9 : a 9 = 80) : 
  a 6 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l638_63872


namespace NUMINAMATH_CALUDE_first_discount_is_twenty_percent_l638_63887

/-- Proves that the first discount is 20% given the conditions of the problem -/
theorem first_discount_is_twenty_percent
  (list_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : list_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5)
  : ∃ (first_discount : ℝ),
    first_discount = 20 ∧
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_is_twenty_percent_l638_63887


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l638_63801

theorem unique_root_of_equation (a b c d : ℝ) 
  (h1 : a + d = 2016)
  (h2 : b + c = 2016)
  (h3 : a ≠ c) :
  ∃! x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) ∧ x = 1008 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l638_63801


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l638_63886

-- Define the square
def square_area : ℝ := 24

-- Define the rectangle's side ratio
def rectangle_ratio : ℝ := 3

-- Theorem statement
theorem inscribed_rectangle_area :
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    y = rectangle_ratio * x ∧
    x * y = 18 ∧
    x^2 + y^2 = square_area := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l638_63886


namespace NUMINAMATH_CALUDE_total_bird_wings_l638_63800

/-- The number of birds in the sky -/
def num_birds : ℕ := 13

/-- The number of wings each bird has -/
def wings_per_bird : ℕ := 2

/-- Theorem: The total number of bird wings in the sky is 26 -/
theorem total_bird_wings : num_birds * wings_per_bird = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_bird_wings_l638_63800


namespace NUMINAMATH_CALUDE_father_son_age_difference_l638_63868

theorem father_son_age_difference :
  ∀ (f s : ℕ+),
  f * s = 2015 →
  f > s →
  f - s = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l638_63868


namespace NUMINAMATH_CALUDE_father_current_age_l638_63842

/-- The age of the daughter now -/
def daughter_age : ℕ := 10

/-- The age of the father now -/
def father_age : ℕ := 4 * daughter_age

/-- In 20 years, the father will be twice as old as the daughter -/
axiom future_relation : father_age + 20 = 2 * (daughter_age + 20)

theorem father_current_age : father_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_father_current_age_l638_63842


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l638_63835

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 4*x + 4 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l638_63835


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l638_63893

theorem angle_sum_around_point (y : ℝ) : 
  y + y + 140 = 360 → y = 110 := by sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l638_63893


namespace NUMINAMATH_CALUDE_probability_specific_arrangement_l638_63820

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem probability_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles) = 1 / 35 := by sorry

end NUMINAMATH_CALUDE_probability_specific_arrangement_l638_63820


namespace NUMINAMATH_CALUDE_opposite_number_l638_63883

theorem opposite_number (a : ℝ) : -a = -2023 → a = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_l638_63883


namespace NUMINAMATH_CALUDE_smallest_winning_points_l638_63848

/-- Represents the possible placings in a race -/
inductive Placing
| First
| Second
| Third
| Other

/-- Calculates the points for a given placing -/
def points_for_placing (p : Placing) : ℕ :=
  match p with
  | Placing.First => 7
  | Placing.Second => 4
  | Placing.Third => 2
  | Placing.Other => 0

/-- Calculates the total points for a list of placings -/
def total_points (placings : List Placing) : ℕ :=
  placings.map points_for_placing |>.sum

/-- Represents the results of four races -/
def RaceResults := List Placing

/-- Checks if a given point total guarantees winning -/
def guarantees_win (points : ℕ) : Prop :=
  ∀ (other_results : RaceResults), 
    other_results.length = 4 → total_points other_results < points

theorem smallest_winning_points : 
  (guarantees_win 25) ∧ (∀ p : ℕ, p < 25 → ¬guarantees_win p) := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_points_l638_63848


namespace NUMINAMATH_CALUDE_line_and_circle_tangent_l638_63869

-- Define the lines and circle
def l₁ (x y : ℝ) : Prop := 2 * x - y = 1
def l₂ (x y : ℝ) : Prop := x + 2 * y = 3
def l₃ (x y : ℝ) : Prop := x - y + 1 = 0
def C (x y a : ℝ) : Prop := (x - a)^2 + y^2 = 8

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Main theorem
theorem line_and_circle_tangent :
  (∀ x y : ℝ, l₁ x y ∧ l₂ x y → (x, y) = P) →  -- P is the intersection of l₁ and l₂
  (∀ x y : ℝ, l x y → l₃ (x + 1) (y + 1)) →  -- l is perpendicular to l₃
  ∃ a : ℝ, a > 0 ∧
    (∀ x y : ℝ, l x y → 
      (∃ t : ℝ, C x y a ∧ 
        (∀ x' y', C x' y' a → (x - x')^2 + (y - y')^2 ≥ t^2) ∧
        (∃ x' y', C x' y' a ∧ (x - x')^2 + (y - y')^2 = t^2))) →
  (∀ x y : ℝ, l x y ↔ x + y - 2 = 0) ∧ a = 6 :=
sorry

end NUMINAMATH_CALUDE_line_and_circle_tangent_l638_63869


namespace NUMINAMATH_CALUDE_postage_cost_for_625_ounces_l638_63861

/-- Calculates the postage cost for a letter -/
def postage_cost (weight : ℚ) (base_rate : ℚ) (additional_rate : ℚ) : ℚ :=
  let additional_weight := (weight - 1).ceil
  base_rate + additional_weight * additional_rate

theorem postage_cost_for_625_ounces :
  postage_cost (6.25 : ℚ) (0.50 : ℚ) (0.30 : ℚ) = (2.30 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_postage_cost_for_625_ounces_l638_63861


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_main_theorem_l638_63897

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + x - 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 1

-- Theorem statement
theorem tangent_line_perpendicular (a : ℝ) : 
  (f' a 1 = 2) → a = 1 := by
  sorry

-- Main theorem
theorem main_theorem (a : ℝ) : 
  (∃ (k : ℝ), f' a 1 = k ∧ k * (-1/2) = -1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_main_theorem_l638_63897


namespace NUMINAMATH_CALUDE_adelaide_ducks_main_theorem_l638_63858

/-- Proves that Adelaide bought 30 ducks given the conditions of the problem -/
theorem adelaide_ducks : ℕ → ℕ → ℕ → Prop :=
  fun adelaide ephraim kolton =>
    adelaide = 2 * ephraim ∧
    ephraim = kolton - 45 ∧
    (adelaide + ephraim + kolton) / 3 = 35 →
    adelaide = 30

/-- Main theorem statement -/
theorem main_theorem : ∃ (a e k : ℕ), adelaide_ducks a e k :=
  sorry

end NUMINAMATH_CALUDE_adelaide_ducks_main_theorem_l638_63858


namespace NUMINAMATH_CALUDE_cube_difference_factorization_l638_63808

theorem cube_difference_factorization (a b : ℝ) :
  a^3 - 8*b^3 = (a - 2*b)*(a^2 + 2*a*b + 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_factorization_l638_63808


namespace NUMINAMATH_CALUDE_additional_pots_in_warm_hour_l638_63806

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time in minutes to produce a pot when the machine is cold -/
def cold_production_time : ℕ := 6

/-- The time in minutes to produce a pot when the machine is warm -/
def warm_production_time : ℕ := 5

/-- Theorem stating the difference in pot production between warm and cold hours -/
theorem additional_pots_in_warm_hour :
  (minutes_per_hour / warm_production_time) - (minutes_per_hour / cold_production_time) = 2 :=
by sorry

end NUMINAMATH_CALUDE_additional_pots_in_warm_hour_l638_63806


namespace NUMINAMATH_CALUDE_peggy_final_doll_count_l638_63812

/-- Calculates the final number of dolls Peggy has after a series of events --/
def finalDollCount (initialDolls : ℕ) (grandmotherGift : ℕ) : ℕ :=
  let birthdayGift := grandmotherGift / 2
  let afterBirthday := initialDolls + grandmotherGift + birthdayGift
  let afterSpringCleaning := afterBirthday - (afterBirthday / 10)
  let easterGift := birthdayGift / 3
  let afterEaster := afterSpringCleaning + easterGift
  let afterExchange := afterEaster - 1
  let christmasGift := easterGift + (easterGift / 5)
  let afterChristmas := afterExchange + christmasGift
  afterChristmas - 3

/-- Theorem stating that Peggy ends up with 50 dolls --/
theorem peggy_final_doll_count :
  finalDollCount 6 28 = 50 := by
  sorry

end NUMINAMATH_CALUDE_peggy_final_doll_count_l638_63812


namespace NUMINAMATH_CALUDE_equation_one_solution_l638_63853

theorem equation_one_solution :
  ∀ x : ℝ, x^4 - x^2 - 6 = 0 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l638_63853


namespace NUMINAMATH_CALUDE_range_of_a_l638_63804

theorem range_of_a (a : ℝ) : 
  (a + 1)^(-1/2 : ℝ) < (3 - 2*a)^(-1/2 : ℝ) → 
  2/3 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l638_63804


namespace NUMINAMATH_CALUDE_field_trip_bus_occupancy_l638_63809

/-- Proves that given the conditions from the field trip problem, 
    the number of people in each bus is 18.0 --/
theorem field_trip_bus_occupancy 
  (num_vans : ℝ) 
  (num_buses : ℝ) 
  (people_per_van : ℝ) 
  (additional_people_in_buses : ℝ) 
  (h1 : num_vans = 6.0)
  (h2 : num_buses = 8.0)
  (h3 : people_per_van = 6.0)
  (h4 : additional_people_in_buses = 108.0) :
  (num_vans * people_per_van + additional_people_in_buses) / num_buses = 18.0 := by
  sorry

#eval (6.0 * 6.0 + 108.0) / 8.0  -- This should evaluate to 18.0

end NUMINAMATH_CALUDE_field_trip_bus_occupancy_l638_63809


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l638_63898

theorem factorial_equation_solution :
  ∃! N : ℕ, (6 : ℕ).factorial * (11 : ℕ).factorial = 20 * N.factorial :=
by sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l638_63898


namespace NUMINAMATH_CALUDE_custom_mul_one_neg_three_l638_63892

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + 2*a*b - b^2

-- Theorem statement
theorem custom_mul_one_neg_three :
  custom_mul 1 (-3) = -14 :=
by
  sorry

end NUMINAMATH_CALUDE_custom_mul_one_neg_three_l638_63892


namespace NUMINAMATH_CALUDE_extremum_values_l638_63850

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

theorem extremum_values (a b : ℝ) : 
  (∀ x, f a b x ≤ f a b 1) ∨ (∀ x, f a b x ≥ f a b 1) →
  f a b 1 = 10 →
  a = -4 ∧ b = 11 := by
sorry

end NUMINAMATH_CALUDE_extremum_values_l638_63850


namespace NUMINAMATH_CALUDE_system_solution_l638_63888

theorem system_solution :
  let x₁ : ℝ := (35 + Real.sqrt 1321) / 24
  let y₁ : ℝ := (-125 - 7 * Real.sqrt 1321) / 72
  let x₂ : ℝ := (35 - Real.sqrt 1321) / 24
  let y₂ : ℝ := (-125 + 7 * Real.sqrt 1321) / 72
  (7 * x₁ + 3 * y₁ = 5 ∧ 4 * x₁^2 + 5 * y₁ = 9) ∧
  (7 * x₂ + 3 * y₂ = 5 ∧ 4 * x₂^2 + 5 * y₂ = 9) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l638_63888


namespace NUMINAMATH_CALUDE_sector_central_angle_l638_63825

/-- Given a circular sector with arc length 4 and area 4, prove that its central angle in radians is 2. -/
theorem sector_central_angle (arc_length area : ℝ) (h1 : arc_length = 4) (h2 : area = 4) :
  let r := 2 * area / arc_length
  2 * area / (r ^ 2) = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l638_63825


namespace NUMINAMATH_CALUDE_mcpherson_rent_contribution_l638_63874

/-- Calculates the amount Mr. McPherson needs to raise for rent -/
theorem mcpherson_rent_contribution 
  (total_rent : ℕ) 
  (mrs_mcpherson_percentage : ℚ) 
  (h1 : total_rent = 1200)
  (h2 : mrs_mcpherson_percentage = 30 / 100) : 
  total_rent - (mrs_mcpherson_percentage * total_rent).floor = 840 := by
sorry

end NUMINAMATH_CALUDE_mcpherson_rent_contribution_l638_63874


namespace NUMINAMATH_CALUDE_correct_average_l638_63803

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 16 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 35 → 
  (n * incorrect_avg + (correct_num - incorrect_num)) / n = 17 := by
sorry

end NUMINAMATH_CALUDE_correct_average_l638_63803


namespace NUMINAMATH_CALUDE_inequality_proof_l638_63863

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (a + 2*b + c)^2 / (2*b^2 + (c + a)^2) +
  (a + b + 2*c)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l638_63863


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l638_63838

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The probability of a binomial random variable being greater than or equal to k -/
noncomputable def prob_ge (X : BinomialRV) (k : ℕ) : ℝ := sorry

/-- The theorem statement -/
theorem binomial_probability_theorem (ξ η : BinomialRV) 
  (h_ξ : ξ.n = 2) (h_η : η.n = 4) (h_p : ξ.p = η.p) 
  (h_prob : prob_ge ξ 1 = 5/9) : 
  prob_ge η 2 = 11/27 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l638_63838


namespace NUMINAMATH_CALUDE_smallest_lucky_number_l638_63873

theorem smallest_lucky_number : 
  ∃ (a b c d : ℕ+), 
    (545 = a^2 + b^2 ∧ 545 = c^2 + d^2) ∧
    (a - c = 7 ∧ d - b = 13) ∧
    (∀ (N : ℕ) (a' b' c' d' : ℕ+), 
      (N < 545 → ¬(N = a'^2 + b'^2 ∧ N = c'^2 + d'^2 ∧ a' - c' = 7 ∧ d' - b' = 13))) := by
  sorry

#check smallest_lucky_number

end NUMINAMATH_CALUDE_smallest_lucky_number_l638_63873


namespace NUMINAMATH_CALUDE_function_range_theorem_l638_63878

open Real

theorem function_range_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x > 0, 9 * f x < x * (deriv f x) ∧ x * (deriv f x) < 10 * f x)
  (h2 : ∀ x > 0, f x > 0) :
  2^9 < f 2 / f 1 ∧ f 2 / f 1 < 2^10 := by
sorry

end NUMINAMATH_CALUDE_function_range_theorem_l638_63878


namespace NUMINAMATH_CALUDE_inequality_proof_l638_63884

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_leq_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l638_63884


namespace NUMINAMATH_CALUDE_cows_ran_away_after_10_days_l638_63867

/-- The number of days that passed before cows ran away -/
def days_before_cows_ran_away (initial_cows : ℕ) (initial_duration : ℕ) (cows_ran_away : ℕ) : ℕ :=
  (initial_cows * initial_duration - (initial_cows - cows_ran_away) * initial_duration) / initial_cows

theorem cows_ran_away_after_10_days :
  days_before_cows_ran_away 1000 50 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cows_ran_away_after_10_days_l638_63867


namespace NUMINAMATH_CALUDE_concert_ticket_sales_l638_63816

/-- Represents the number of non-student tickets sold at an annual concert --/
def non_student_tickets : ℕ := 60

/-- Represents the number of student tickets sold at an annual concert --/
def student_tickets : ℕ := 150 - non_student_tickets

/-- The price of a student ticket in dollars --/
def student_price : ℕ := 5

/-- The price of a non-student ticket in dollars --/
def non_student_price : ℕ := 8

/-- The total revenue from ticket sales in dollars --/
def total_revenue : ℕ := 930

/-- The total number of tickets sold --/
def total_tickets : ℕ := 150

theorem concert_ticket_sales :
  (student_tickets * student_price + non_student_tickets * non_student_price = total_revenue) ∧
  (student_tickets + non_student_tickets = total_tickets) :=
by sorry

end NUMINAMATH_CALUDE_concert_ticket_sales_l638_63816


namespace NUMINAMATH_CALUDE_equal_distribution_l638_63802

/-- Proves that when Rs 42,900 is distributed equally among 22 persons, each person receives Rs 1,950. -/
theorem equal_distribution (total_amount : ℕ) (num_persons : ℕ) (amount_per_person : ℕ) : 
  total_amount = 42900 → 
  num_persons = 22 → 
  amount_per_person = total_amount / num_persons → 
  amount_per_person = 1950 := by
sorry

end NUMINAMATH_CALUDE_equal_distribution_l638_63802


namespace NUMINAMATH_CALUDE_max_product_sum_300_l638_63823

theorem max_product_sum_300 : 
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l638_63823


namespace NUMINAMATH_CALUDE_aisha_head_fraction_l638_63821

/-- Miss Aisha's height measurements -/
structure AishaHeight where
  total : ℝ
  legs : ℝ
  rest : ℝ
  head : ℝ

/-- Properties of Miss Aisha's height -/
def aisha_properties (h : AishaHeight) : Prop :=
  h.total = 60 ∧
  h.legs = (1/3) * h.total ∧
  h.rest = 25 ∧
  h.head = h.total - (h.legs + h.rest)

/-- Theorem: Miss Aisha's head is 1/4 of her total height -/
theorem aisha_head_fraction (h : AishaHeight) 
  (hprops : aisha_properties h) : h.head / h.total = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_aisha_head_fraction_l638_63821


namespace NUMINAMATH_CALUDE_expression_simplification_l638_63843

theorem expression_simplification (x : ℝ) (h : x = 2 + Real.sqrt 3) :
  (x + 1) / (x^2 - 4) * ((1 / (x + 1)) + 1) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l638_63843


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l638_63864

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

theorem downstream_speed_calculation (s : RowingSpeed)
  (h1 : s.upstream = 25)
  (h2 : s.stillWater = 31) :
  downstreamSpeed s = 37 := by sorry

end NUMINAMATH_CALUDE_downstream_speed_calculation_l638_63864


namespace NUMINAMATH_CALUDE_parallel_vectors_l638_63817

/-- Given two vectors a and b in R², prove that ka + b is parallel to a - 3b iff k = -1/3 -/
theorem parallel_vectors (a b : Fin 2 → ℝ) (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : b 0 = -3) (h4 : b 1 = 2) :
  (∃ k : ℝ, ∀ i : Fin 2, k * (a i) + (b i) = c * ((a i) - 3 * (b i)) ∧ c ≠ 0) ↔ k = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l638_63817


namespace NUMINAMATH_CALUDE_liam_nickels_problem_l638_63819

theorem liam_nickels_problem :
  ∃! n : ℕ, 120 < n ∧ n < 400 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 6 = 4 ∧
    n = 374 := by
  sorry

end NUMINAMATH_CALUDE_liam_nickels_problem_l638_63819


namespace NUMINAMATH_CALUDE_relationship_abc_l638_63851

theorem relationship_abc (a b c : ℝ) (ha : a = Real.exp 0.3) (hb : b = 0.9^2) (hc : c = Real.log 0.9) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l638_63851


namespace NUMINAMATH_CALUDE_smallest_bench_arrangement_l638_63862

theorem smallest_bench_arrangement (n : ℕ) : 
  (∃ k : ℕ, 8 * n = 10 * k) ∧ 
  (n % 8 = 0) ∧ (n % 10 = 0) ∧
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, 8 * m = 10 * k) ∧ (m % 8 = 0) ∧ (m % 10 = 0))) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_bench_arrangement_l638_63862


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l638_63822

theorem polynomial_root_problem (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ t : ℝ, t^3 + a*t^2 + b*t + 20 = 0 ↔ t = x ∨ t = y ∨ t = z)) →
  (∀ t : ℝ, t^3 + a*t^2 + b*t + 20 = 0 → t^4 + t^3 + b*t^2 + c*t + 200 = 0) →
  (1 : ℝ)^4 + (1 : ℝ)^3 + b*(1 : ℝ)^2 + c*(1 : ℝ) + 200 = 132 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l638_63822


namespace NUMINAMATH_CALUDE_even_four_digit_count_is_336_l638_63857

/-- A function that counts the number of even integers between 4000 and 8000 with four different digits -/
def count_even_four_digit_numbers : ℕ :=
  336

/-- Theorem stating that the count of even integers between 4000 and 8000 with four different digits is 336 -/
theorem even_four_digit_count_is_336 : count_even_four_digit_numbers = 336 := by
  sorry

end NUMINAMATH_CALUDE_even_four_digit_count_is_336_l638_63857


namespace NUMINAMATH_CALUDE_friend_payment_amount_l638_63830

/-- The cost per item for each food item --/
def hamburger_cost : ℚ := 3
def fries_cost : ℚ := 6/5  -- 1.20 as a rational number
def soda_cost : ℚ := 1/2
def spaghetti_cost : ℚ := 27/10
def milkshake_cost : ℚ := 5/2
def nuggets_cost : ℚ := 7/2

/-- The number of each item ordered --/
def hamburger_count : ℕ := 5
def fries_count : ℕ := 4
def soda_count : ℕ := 5
def spaghetti_count : ℕ := 1
def milkshake_count : ℕ := 3
def nuggets_count : ℕ := 2

/-- The discount percentage as a rational number --/
def discount_percent : ℚ := 1/10

/-- The percentage of the bill paid by the birthday friend --/
def birthday_friend_percent : ℚ := 3/10

/-- The number of friends splitting the remaining bill --/
def remaining_friends : ℕ := 4

/-- The theorem stating that each remaining friend will pay $6.22 --/
theorem friend_payment_amount : 
  let total_bill := hamburger_cost * hamburger_count + 
                    fries_cost * fries_count +
                    soda_cost * soda_count +
                    spaghetti_cost * spaghetti_count +
                    milkshake_cost * milkshake_count +
                    nuggets_cost * nuggets_count
  let discounted_bill := total_bill * (1 - discount_percent)
  let remaining_bill := discounted_bill * (1 - birthday_friend_percent)
  remaining_bill / remaining_friends = 311/50  -- 6.22 as a rational number
  := by sorry

end NUMINAMATH_CALUDE_friend_payment_amount_l638_63830


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l638_63855

/-- A point (x, y) lies on the hyperbola y = -6/x if and only if xy = -6 -/
def lies_on_hyperbola (x y : ℝ) : Prop := x * y = -6

/-- The point (3, -2) lies on the hyperbola y = -6/x -/
theorem point_on_hyperbola : lies_on_hyperbola 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l638_63855


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l638_63879

theorem sum_of_specific_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l638_63879


namespace NUMINAMATH_CALUDE_parallel_sides_equal_or_complementary_l638_63815

/-- Two angles in space -/
structure AngleInSpace where
  -- Define the necessary components of an angle in space
  -- This is a simplified representation
  measure : ℝ

/-- Predicate to check if two angles have parallel sides -/
def has_parallel_sides (a b : AngleInSpace) : Prop :=
  -- This is a placeholder for the actual condition of parallel sides
  True

/-- Predicate to check if two angles are equal -/
def are_equal (a b : AngleInSpace) : Prop :=
  a.measure = b.measure

/-- Predicate to check if two angles are complementary -/
def are_complementary (a b : AngleInSpace) : Prop :=
  a.measure + b.measure = 90

/-- Theorem: If two angles in space have parallel sides, 
    then they are either equal or complementary -/
theorem parallel_sides_equal_or_complementary (a b : AngleInSpace) :
  has_parallel_sides a b → (are_equal a b ∨ are_complementary a b) := by
  sorry

end NUMINAMATH_CALUDE_parallel_sides_equal_or_complementary_l638_63815


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l638_63846

/-- Proposition p: x = 1 and y = 1 -/
def p (x y : ℝ) : Prop := x = 1 ∧ y = 1

/-- Proposition q: x + y = 2 -/
def q (x y : ℝ) : Prop := x + y = 2

/-- p is a sufficient but not necessary condition for q -/
theorem p_sufficient_not_necessary :
  (∀ x y : ℝ, p x y → q x y) ∧
  (∃ x y : ℝ, q x y ∧ ¬p x y) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l638_63846


namespace NUMINAMATH_CALUDE_matrix_power_2018_l638_63837

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 1, 1]

theorem matrix_power_2018 :
  A ^ 2018 = !![1, 0; 2018, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2018_l638_63837


namespace NUMINAMATH_CALUDE_cookie_difference_l638_63866

/-- Proves that the difference between the number of cookies in 8 boxes and 9 bags is 33,
    given that each bag contains 7 cookies and each box contains 12 cookies. -/
theorem cookie_difference :
  let cookies_per_bag : ℕ := 7
  let cookies_per_box : ℕ := 12
  let num_boxes : ℕ := 8
  let num_bags : ℕ := 9
  (num_boxes * cookies_per_box) - (num_bags * cookies_per_bag) = 33 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l638_63866


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l638_63828

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_75th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_15 : a 15 = 8)
  (h_60 : a 60 = 20) :
  a 75 = 24 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l638_63828


namespace NUMINAMATH_CALUDE_houses_painted_in_three_hours_l638_63839

/-- The number of houses that can be painted in a given time -/
def houses_painted (minutes_per_house : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours * 60) / minutes_per_house

/-- Theorem: Given it takes 20 minutes to paint a house, 
    the number of houses that can be painted in 3 hours is 9 -/
theorem houses_painted_in_three_hours :
  houses_painted 20 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_houses_painted_in_three_hours_l638_63839


namespace NUMINAMATH_CALUDE_largest_expression_l638_63811

theorem largest_expression : 
  let a := 3 + 1 + 4
  let b := 3 * 1 + 4
  let c := 3 + 1 * 4
  let d := 3 * 1 * 4
  let e := 3 + 0 * 1 + 4
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l638_63811


namespace NUMINAMATH_CALUDE_deepak_age_l638_63813

theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (future_years : ℕ) (rahul_future_age : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 3 →
  future_years = 6 →
  rahul_future_age = 18 →
  ∃ (x : ℚ), rahul_ratio * x + future_years = rahul_future_age ∧ deepak_ratio * x = 9 :=
by sorry

end NUMINAMATH_CALUDE_deepak_age_l638_63813


namespace NUMINAMATH_CALUDE_average_of_c_and_d_l638_63891

theorem average_of_c_and_d (c d : ℝ) : 
  (4 + 6 + 9 + c + d) / 5 = 18 → (c + d) / 2 = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_c_and_d_l638_63891


namespace NUMINAMATH_CALUDE_not_valid_base_5_l638_63859

/-- Given a base k and a sequence of digits, determines if it's a valid representation in that base -/
def is_valid_base_k_number (k : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < k

/-- The theorem states that 32501 is not a valid base-5 number -/
theorem not_valid_base_5 :
  ¬ (is_valid_base_k_number 5 [3, 2, 5, 0, 1]) :=
sorry

end NUMINAMATH_CALUDE_not_valid_base_5_l638_63859


namespace NUMINAMATH_CALUDE_quadrilateral_area_l638_63890

/-- The area of a quadrilateral given its four sides and the angle between diagonals -/
theorem quadrilateral_area (a b c d ω : ℝ) (h_pos : 0 < ω ∧ ω < π) :
  ∃ t : ℝ, t = (1/4) * (b^2 + d^2 - a^2 - c^2) * Real.tan ω :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l638_63890


namespace NUMINAMATH_CALUDE_rectangle_area_l638_63807

/-- Rectangle ABCD with specific properties -/
structure Rectangle where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side AD -/
  AD : ℝ
  /-- AD is 9 units longer than AB -/
  length_diff : AD = AB + 9
  /-- Area of trapezoid ABCE is 5 times the area of triangle ADE -/
  area_ratio : AB * AD = 6 * ((1/2) * AB * (1/3 * AD))
  /-- Perimeter difference between trapezoid ABCE and triangle ADE -/
  perimeter_diff : AB + (2/3 * AB) - (1/3 * AB) = 68

/-- The area of the rectangle ABCD is 3060 square units -/
theorem rectangle_area (rect : Rectangle) : rect.AB * rect.AD = 3060 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l638_63807


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l638_63860

-- Define the parabola E: y^2 = 4x
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the directrix l: x = -1
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the focus F(1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define a function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Main theorem
theorem fixed_point_theorem (A B : ℝ × ℝ) (h_A : A ∈ E) (h_B : B ∈ E) 
  (h_line : ∃ k : ℝ, A.2 - F.2 = k * (A.1 - F.1) ∧ B.2 - F.2 = k * (B.1 - F.1)) :
  ∃ t : ℝ, reflect_x A + t • (B - reflect_x A) = (-1, 0) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l638_63860


namespace NUMINAMATH_CALUDE_sets_equality_l638_63881

-- Define the sets M, N, and P
def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1/2}

-- Theorem statement
theorem sets_equality : N = M ∪ P := by sorry

end NUMINAMATH_CALUDE_sets_equality_l638_63881


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l638_63847

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the intersection points
def intersection_points (m : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define the condition |MD| = 2|NF|
def length_condition (M N : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := M
  let (x₂, y₂) := N
  x₁ = 2*x₂ + 2

-- Main theorem
theorem parabola_intersection_theorem (m : ℝ) : 
  let (x₁, y₁, x₂, y₂) := intersection_points m
  let M := (x₁, y₁)
  let N := (x₂, y₂)
  parabola x₁ y₁ ∧ 
  parabola x₂ y₂ ∧
  line_through_focus m x₁ y₁ ∧
  line_through_focus m x₂ y₂ ∧
  length_condition M N →
  Real.sqrt ((x₁ - 1)^2 + y₁^2) = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l638_63847


namespace NUMINAMATH_CALUDE_james_run_calories_l638_63877

/-- Calculates the calories burned per minute during James' run -/
def caloriesBurnedPerMinute (bagsEaten : ℕ) (ouncesPerBag : ℕ) (caloriesPerOunce : ℕ) 
  (runDuration : ℕ) (excessCalories : ℕ) : ℕ :=
  let totalOunces := bagsEaten * ouncesPerBag
  let totalCaloriesConsumed := totalOunces * caloriesPerOunce
  let caloriesBurned := totalCaloriesConsumed - excessCalories
  caloriesBurned / runDuration

theorem james_run_calories : 
  caloriesBurnedPerMinute 3 2 150 40 420 = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_run_calories_l638_63877


namespace NUMINAMATH_CALUDE_hexagon_problem_l638_63827

-- Define the regular hexagon
structure RegularHexagon :=
  (side_length : ℝ)
  (A B C D E F : ℝ × ℝ)

-- Define the intersection point L
def L (hex : RegularHexagon) : ℝ × ℝ := sorry

-- Define point K
def K (hex : RegularHexagon) : ℝ × ℝ := sorry

-- Function to check if a point is outside the hexagon
def is_outside (hex : RegularHexagon) (point : ℝ × ℝ) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem hexagon_problem (hex : RegularHexagon) 
  (h1 : hex.side_length = 2) :
  is_outside hex (K hex) ∧ 
  distance (K hex) hex.B = (2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_problem_l638_63827


namespace NUMINAMATH_CALUDE_cube_root_not_always_two_l638_63880

theorem cube_root_not_always_two (x : ℝ) (h : x^2 = 64) : 
  ∃ y, y^3 = x ∧ y ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_not_always_two_l638_63880


namespace NUMINAMATH_CALUDE_sinusoidal_amplitude_l638_63865

/-- Given a sinusoidal function f(x) = a * sin(bx + c) + d with positive constants a, b, c, d,
    if the maximum value of f is 3 and the minimum value is -1, then a = 2 -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (b * x + c) + d
  (∀ x, f x ≤ 3) ∧ (∀ x, f x ≥ -1) ∧ (∃ x y, f x = 3 ∧ f y = -1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_amplitude_l638_63865


namespace NUMINAMATH_CALUDE_min_area_special_square_l638_63829

/-- A square with one side on y = 2x - 17 and two vertices on y = x^2 -/
structure SpecialSquare where
  -- Coordinates of the two vertices on the parabola
  x₁ : ℝ
  x₂ : ℝ
  -- Conditions
  vertex_on_parabola : x₁ < x₂ ∧ (x₁, x₁^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2} ∧ (x₂, x₂^2) ∈ {p : ℝ × ℝ | p.2 = p.1^2}
  side_on_line : ∃ (a b : ℝ), (a, 2*a - 17) ∈ {p : ℝ × ℝ | p.2 = 2*p.1 - 17} ∧ 
                               (b, 2*b - 17) ∈ {p : ℝ × ℝ | p.2 = 2*p.1 - 17} ∧
                               (b - a)^2 + (2*b - 17 - (2*a - 17))^2 = (x₂ - x₁)^2 + (x₂^2 - x₁^2)^2

/-- The area of a SpecialSquare -/
def area (s : SpecialSquare) : ℝ := (s.x₂ - s.x₁)^2 + (s.x₂^2 - s.x₁^2)^2

/-- Theorem stating the minimum area of a SpecialSquare is 80 -/
theorem min_area_special_square : 
  ∀ s : SpecialSquare, area s ≥ 80 ∧ ∃ s' : SpecialSquare, area s' = 80 := by
  sorry

end NUMINAMATH_CALUDE_min_area_special_square_l638_63829


namespace NUMINAMATH_CALUDE_fathers_remaining_chocolates_fathers_remaining_chocolates_eq_five_l638_63871

theorem fathers_remaining_chocolates 
  (initial_chocolates : ℕ) 
  (num_sisters : ℕ) 
  (chocolates_to_mother : ℕ) 
  (chocolates_eaten : ℕ) : ℕ :=
  let total_people := num_sisters + 1
  let chocolates_per_person := initial_chocolates / total_people
  let chocolates_given_to_father := total_people * (chocolates_per_person / 2)
  let remaining_chocolates := chocolates_given_to_father - chocolates_to_mother - chocolates_eaten
  remaining_chocolates

theorem fathers_remaining_chocolates_eq_five :
  fathers_remaining_chocolates 20 4 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fathers_remaining_chocolates_fathers_remaining_chocolates_eq_five_l638_63871


namespace NUMINAMATH_CALUDE_problem_statement_l638_63841

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 4) :
  x^2 * y^3 + y^2 * x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l638_63841


namespace NUMINAMATH_CALUDE_remainder_seven_fourth_mod_hundred_l638_63814

theorem remainder_seven_fourth_mod_hundred : 7^4 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_fourth_mod_hundred_l638_63814


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_45_l638_63818

theorem no_primes_divisible_by_45 : ∀ p : ℕ, Nat.Prime p → ¬(45 ∣ p) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_45_l638_63818


namespace NUMINAMATH_CALUDE_football_field_fertilizer_l638_63899

/-- Proves that the total amount of fertilizer spread across a football field is 800 pounds,
    given the field's area and a known fertilizer distribution over a portion of the field. -/
theorem football_field_fertilizer (total_area : ℝ) (partial_area : ℝ) (partial_fertilizer : ℝ) :
  total_area = 9600 →
  partial_area = 3600 →
  partial_fertilizer = 300 →
  (partial_fertilizer / partial_area) * total_area = 800 := by
  sorry

end NUMINAMATH_CALUDE_football_field_fertilizer_l638_63899


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l638_63824

/-- The line l with equation (a-2)y = (3a-1)x - 1 does not pass through the second quadrant
    if and only if a ∈ [2, +∞) -/
theorem line_not_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 2) * y = (3 * a - 1) * x - 1 → ¬(x < 0 ∧ y > 0)) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l638_63824


namespace NUMINAMATH_CALUDE_inequality_preservation_l638_63889

theorem inequality_preservation (x y : ℝ) : x < y → 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l638_63889


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l638_63895

/-- Proves that for a square plot with an area of 36 sq ft and a total fencing cost of Rs. 1392, the price per foot of fencing is Rs. 58. -/
theorem fence_cost_per_foot (plot_area : ℝ) (total_cost : ℝ) (h1 : plot_area = 36) (h2 : total_cost = 1392) :
  let side_length : ℝ := Real.sqrt plot_area
  let perimeter : ℝ := 4 * side_length
  let cost_per_foot : ℝ := total_cost / perimeter
  cost_per_foot = 58 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l638_63895


namespace NUMINAMATH_CALUDE_return_flight_theorem_l638_63870

/-- Represents a direction in degrees relative to a cardinal direction -/
structure Direction where
  angle : ℝ
  cardinal : String
  relative : String

/-- Represents a flight path -/
structure FlightPath where
  distance : ℝ
  direction : Direction

/-- Returns the opposite direction for a given flight path -/
def oppositeDirection (fp : FlightPath) : Direction :=
  { angle := fp.direction.angle,
    cardinal := if fp.direction.cardinal = "east" then "west" else "east",
    relative := if fp.direction.relative = "south" then "north" else "south" }

theorem return_flight_theorem (outbound : FlightPath) 
  (h1 : outbound.distance = 1200)
  (h2 : outbound.direction.angle = 30)
  (h3 : outbound.direction.cardinal = "east")
  (h4 : outbound.direction.relative = "south") :
  ∃ (inbound : FlightPath),
    inbound.distance = outbound.distance ∧
    inbound.direction = oppositeDirection outbound :=
  sorry

end NUMINAMATH_CALUDE_return_flight_theorem_l638_63870


namespace NUMINAMATH_CALUDE_fifth_term_geometric_sequence_l638_63833

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifth_term_geometric_sequence :
  let a₁ : ℚ := 2
  let a₂ : ℚ := 1/4
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 5 = 1/2048 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_sequence_l638_63833


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_six_l638_63885

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → x = 2 ∧ y = 3

/-- The first monomial 5a^m * b^3 -/
def term1 (m : ℕ) (x y : ℕ) : ℚ :=
  if x = m ∧ y = 3 then 5 else 0

/-- The second monomial -4a^2 * b^(n-1) -/
def term2 (n : ℕ) (x y : ℕ) : ℚ :=
  if x = 2 ∧ y = n - 1 then -4 else 0

theorem like_terms_imply_sum_six (m n : ℕ) :
  like_terms (term1 m) (term2 n) → m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_six_l638_63885


namespace NUMINAMATH_CALUDE_total_monthly_wages_after_new_hires_l638_63894

/-- Calculate total monthly wages after new hires -/
theorem total_monthly_wages_after_new_hires 
  (initial_employees : ℕ) 
  (hourly_wage : ℕ) 
  (hours_per_day : ℕ) 
  (days_per_week : ℕ) 
  (weeks_per_month : ℕ) 
  (new_employees : ℕ) 
  (h1 : initial_employees = 500)
  (h2 : hourly_wage = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : new_employees = 200) : 
  (initial_employees + new_employees) * 
  (hourly_wage * hours_per_day * days_per_week * weeks_per_month) = 1680000 := by
  sorry

#eval 700 * (12 * 10 * 5 * 4)  -- Should output 1680000

end NUMINAMATH_CALUDE_total_monthly_wages_after_new_hires_l638_63894


namespace NUMINAMATH_CALUDE_division_remainder_seventeen_by_two_l638_63875

theorem division_remainder_seventeen_by_two :
  ∃ (q : ℕ), 17 = 2 * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_seventeen_by_two_l638_63875


namespace NUMINAMATH_CALUDE_original_alcohol_percentage_l638_63856

/-- Proves that given a 15-liter mixture of alcohol and water, if adding 3 liters of water
    results in a new mixture with 20.833333333333336% alcohol, then the original mixture
    contained 25% alcohol. -/
theorem original_alcohol_percentage
  (original_volume : ℝ)
  (added_water : ℝ)
  (new_alcohol_percentage : ℝ)
  (h1 : original_volume = 15)
  (h2 : added_water = 3)
  (h3 : new_alcohol_percentage = 20.833333333333336)
  : ∃ (original_alcohol_percentage : ℝ),
    original_alcohol_percentage = 25 ∧
    (original_alcohol_percentage / 100) * original_volume =
    (new_alcohol_percentage / 100) * (original_volume + added_water) :=
by sorry

end NUMINAMATH_CALUDE_original_alcohol_percentage_l638_63856


namespace NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_zero_l638_63882

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + Real.log x

theorem tangent_line_implies_b_minus_a_zero (a b : ℝ) :
  (∀ x, f a b x = a * x^2 + b * x + Real.log x) →
  (∃ m c, ∀ x, m * x + c = 4 * x - 2 ∧ f a b 1 = m * 1 + c) →
  b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_zero_l638_63882


namespace NUMINAMATH_CALUDE_fill_time_with_leak_l638_63836

/-- Time to fill the cistern without a leak (in hours) -/
def fill_time : ℝ := 8

/-- Time to empty the full cistern through the leak (in hours) -/
def empty_time : ℝ := 24

/-- Theorem: The time to fill the cistern with a leak is 12 hours -/
theorem fill_time_with_leak : 
  (1 / fill_time - 1 / empty_time)⁻¹ = 12 := by sorry

end NUMINAMATH_CALUDE_fill_time_with_leak_l638_63836


namespace NUMINAMATH_CALUDE_jerry_money_duration_l638_63876

/-- The number of weeks Jerry's money will last -/
def weeks_money_lasts (lawn_mowing_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_mowing_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem: Given Jerry's earnings and weekly spending, his money will last 9 weeks -/
theorem jerry_money_duration :
  weeks_money_lasts 14 31 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_duration_l638_63876


namespace NUMINAMATH_CALUDE_imaginary_power_2011_l638_63896

theorem imaginary_power_2011 (i : ℂ) (h : i^2 = -1) : i^2011 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_2011_l638_63896


namespace NUMINAMATH_CALUDE_residue_negative_999_mod_25_l638_63834

theorem residue_negative_999_mod_25 : Int.mod (-999) 25 = 1 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_999_mod_25_l638_63834


namespace NUMINAMATH_CALUDE_laptop_repairs_count_l638_63840

def phone_repair_cost : ℕ := 11
def laptop_repair_cost : ℕ := 15
def computer_repair_cost : ℕ := 18
def phone_repairs : ℕ := 5
def computer_repairs : ℕ := 2
def total_earnings : ℕ := 121

theorem laptop_repairs_count :
  ∃ (laptop_repairs : ℕ),
    phone_repair_cost * phone_repairs +
    laptop_repair_cost * laptop_repairs +
    computer_repair_cost * computer_repairs = total_earnings ∧
    laptop_repairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_laptop_repairs_count_l638_63840


namespace NUMINAMATH_CALUDE_joined_hexagon_triangle_edges_l638_63849

/-- A regular polygon with n sides and side length 1 -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  regularHexagon : sides = 6 → sideLength = 1
  regularTriangle : sides = 3 → sideLength = 1

/-- The number of edges in a shape formed by joining a regular hexagon and a regular triangle -/
def joinedEdges (hexagon triangle : RegularPolygon) : ℕ :=
  hexagon.sides + triangle.sides - 3

theorem joined_hexagon_triangle_edges :
  ∀ (hexagon triangle : RegularPolygon),
  hexagon.sides = 6 ∧ 
  triangle.sides = 3 ∧ 
  hexagon.sideLength = 1 ∧ 
  triangle.sideLength = 1 →
  joinedEdges hexagon triangle = 5 := by
  sorry

end NUMINAMATH_CALUDE_joined_hexagon_triangle_edges_l638_63849


namespace NUMINAMATH_CALUDE_sun_energy_china_equivalence_l638_63805

/-- The energy received from the sun in one year on 1 square kilometer of land,
    measured in kilograms of coal equivalent -/
def energy_per_sq_km : ℝ := 1.3 * 10^8

/-- The approximate land area of China in square kilometers -/
def china_area : ℝ := 9.6 * 10^6

/-- The total energy received from the sun on China's land area,
    measured in kilograms of coal equivalent -/
def total_energy : ℝ := energy_per_sq_km * china_area

theorem sun_energy_china_equivalence :
  total_energy = 1.248 * 10^15 := by
  sorry

end NUMINAMATH_CALUDE_sun_energy_china_equivalence_l638_63805


namespace NUMINAMATH_CALUDE_derived_function_coefficients_target_point_coords_two_base_points_and_distance_range_l638_63852

/-- Definition of a derived function -/
def is_derived_function (a b c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ 0 ∧ x₂ ≠ 0 ∧
  (a * x₁ + b = c / x₂) ∧
  (x₁ = -x₂)

/-- Part 1: Derived function coefficients -/
theorem derived_function_coefficients :
  is_derived_function 2 4 5 := by sorry

/-- Part 2: Target point coordinates -/
theorem target_point_coords (b c : ℝ) :
  is_derived_function 1 b c →
  (∃ (x : ℝ), x^2 + b*x + c = 0) →
  (1 + b = -c) →
  (∃ (x y : ℝ), x = -1 ∧ y = -1 ∧ y = c / x) := by sorry

/-- Part 3: Existence of two base points and their distance range -/
theorem two_base_points_and_distance_range (a b : ℝ) :
  a > b ∧ b > 0 →
  is_derived_function a (2*b) (-2) →
  (∃ (x : ℝ), a*x^2 + 2*b*x - 2 = 6) →
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a*x₁ + 2*b = a*x₁^2 + 2*b*x₁ - 2 ∧ a*x₂ + 2*b = a*x₂^2 + 2*b*x₂ - 2) ∧
  (∃ (x₁ x₂ : ℝ), 2 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_derived_function_coefficients_target_point_coords_two_base_points_and_distance_range_l638_63852


namespace NUMINAMATH_CALUDE_population_growth_l638_63831

theorem population_growth (P : ℝ) : 
  P * 1.1 * 1.2 = 1320 → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_l638_63831


namespace NUMINAMATH_CALUDE_italian_sausage_length_l638_63854

/-- The length of an Italian sausage in inches -/
def sausage_length : ℚ := 12 * (2 / 3)

/-- Theorem: The length of the Italian sausage is 8 inches -/
theorem italian_sausage_length : sausage_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_italian_sausage_length_l638_63854


namespace NUMINAMATH_CALUDE_labourer_savings_labourer_savings_specific_l638_63826

theorem labourer_savings (monthly_income : ℕ) (initial_expense : ℕ) (reduced_expense : ℕ) 
  (initial_months : ℕ) (reduced_months : ℕ) : ℕ :=
  let initial_total_expense := initial_months * initial_expense
  let initial_total_income := initial_months * monthly_income
  let debt := if initial_total_expense > initial_total_income 
    then initial_total_expense - initial_total_income 
    else 0
  let reduced_total_expense := reduced_months * reduced_expense
  let reduced_total_income := reduced_months * monthly_income
  let savings := reduced_total_income - (reduced_total_expense + debt)
  savings

theorem labourer_savings_specific : 
  labourer_savings 72 75 60 6 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_labourer_savings_labourer_savings_specific_l638_63826


namespace NUMINAMATH_CALUDE_sum_of_integers_l638_63845

theorem sum_of_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 45 →
  a + b + c + d + e = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l638_63845


namespace NUMINAMATH_CALUDE_count_specific_divisors_l638_63832

/-- The number of positive integer divisors of 2016^2016 that are divisible by exactly 2016 positive integers -/
def divisors_with_2016_divisors : ℕ :=
  let base := 2016
  let exponent := 2016
  let target_divisors := 2016
  -- Definition of the function to count the divisors
  sorry

/-- The main theorem stating that the number of such divisors is 126 -/
theorem count_specific_divisors :
  divisors_with_2016_divisors = 126 := by sorry

end NUMINAMATH_CALUDE_count_specific_divisors_l638_63832
