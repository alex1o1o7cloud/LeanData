import Mathlib

namespace NUMINAMATH_CALUDE_a_share_calculation_l373_37393

/-- Calculates the share of profit for a partner in a business partnership. -/
def calculateShare (investment totalInvestment totalProfit : ℚ) : ℚ :=
  (investment / totalInvestment) * totalProfit

theorem a_share_calculation (investmentA investmentB investmentC shareB : ℚ) 
  (h1 : investmentA = 15000)
  (h2 : investmentB = 21000)
  (h3 : investmentC = 27000)
  (h4 : shareB = 1540) : 
  calculateShare investmentA (investmentA + investmentB + investmentC) 
    ((investmentA + investmentB + investmentC) * shareB / investmentB) = 1100 := by
  sorry

end NUMINAMATH_CALUDE_a_share_calculation_l373_37393


namespace NUMINAMATH_CALUDE_correct_number_of_selection_plans_l373_37394

def number_of_people : ℕ := 6
def number_of_cities : ℕ := 4
def number_of_restricted_people : ℕ := 2

def selection_plans : ℕ := 240

theorem correct_number_of_selection_plans :
  (number_of_people.factorial / (number_of_people - number_of_cities).factorial) -
  (number_of_restricted_people * ((number_of_people - 1).factorial / (number_of_people - number_of_cities).factorial)) =
  selection_plans := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_selection_plans_l373_37394


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l373_37323

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
   p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
   n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m > 0 → m < n → 
   ¬(∃ q₁ q₂ q₃ q₄ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
     q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
     m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l373_37323


namespace NUMINAMATH_CALUDE_three_lines_exist_l373_37307

-- Define the line segment AB
def AB : ℝ := 10

-- Define the distances from points A and B to line l
def distance_A_to_l : ℝ := 6
def distance_B_to_l : ℝ := 4

-- Define a function that counts the number of lines satisfying the conditions
def count_lines : ℕ := sorry

-- Theorem statement
theorem three_lines_exist : count_lines = 3 := by sorry

end NUMINAMATH_CALUDE_three_lines_exist_l373_37307


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l373_37344

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 4) : x^2 + 1/x^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l373_37344


namespace NUMINAMATH_CALUDE_tickets_left_l373_37349

def tickets_bought : ℕ := 11
def tickets_spent : ℕ := 3

theorem tickets_left : tickets_bought - tickets_spent = 8 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l373_37349


namespace NUMINAMATH_CALUDE_first_robber_guarantee_l373_37361

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : ℕ
  maxBags : ℕ

/-- Represents the outcome of the game for the first robber --/
def firstRobberOutcome (game : CoinGame) (coinsPerBag : ℕ) : ℕ :=
  min (game.totalCoins - game.maxBags * coinsPerBag) (game.maxBags * coinsPerBag)

/-- The theorem stating the maximum guaranteed coins for the first robber --/
theorem first_robber_guarantee (game : CoinGame) : 
  game.totalCoins = 300 → game.maxBags = 11 → 
  ∃ (coinsPerBag : ℕ), firstRobberOutcome game coinsPerBag ≥ 146 := by
  sorry

#eval firstRobberOutcome { totalCoins := 300, maxBags := 11 } 14

end NUMINAMATH_CALUDE_first_robber_guarantee_l373_37361


namespace NUMINAMATH_CALUDE_hiking_trail_length_l373_37338

/-- Represents the hiking trail problem -/
def HikingTrail :=
  {length : ℝ // length > 0}

/-- The total time for the round trip in hours -/
def totalTime : ℝ := 3

/-- The uphill speed in km/h -/
def uphillSpeed : ℝ := 2

/-- The downhill speed in km/h -/
def downhillSpeed : ℝ := 4

/-- Theorem stating that the length of the hiking trail is 4 km -/
theorem hiking_trail_length :
  ∃ (trail : HikingTrail),
    (trail.val / uphillSpeed + trail.val / downhillSpeed = totalTime) ∧
    trail.val = 4 := by sorry

end NUMINAMATH_CALUDE_hiking_trail_length_l373_37338


namespace NUMINAMATH_CALUDE_solution_difference_l373_37368

theorem solution_difference (p q : ℝ) : 
  (p - 4) * (p + 4) = 28 * p - 84 →
  (q - 4) * (q + 4) = 28 * q - 84 →
  p ≠ q →
  p > q →
  p - q = 16 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l373_37368


namespace NUMINAMATH_CALUDE_final_height_in_feet_l373_37333

def initial_height : ℕ := 66
def growth_rate : ℕ := 2
def growth_duration : ℕ := 3
def inches_per_foot : ℕ := 12

theorem final_height_in_feet :
  (initial_height + growth_rate * growth_duration) / inches_per_foot = 6 :=
by sorry

end NUMINAMATH_CALUDE_final_height_in_feet_l373_37333


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l373_37355

theorem fraction_to_decimal : (49 : ℚ) / 160 = 0.30625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l373_37355


namespace NUMINAMATH_CALUDE_typist_margin_l373_37366

theorem typist_margin (sheet_width sheet_length side_margin : ℝ)
  (percentage_used : ℝ) (h1 : sheet_width = 20)
  (h2 : sheet_length = 30) (h3 : side_margin = 2)
  (h4 : percentage_used = 0.64) :
  let total_area := sheet_width * sheet_length
  let typing_width := sheet_width - 2 * side_margin
  let top_bottom_margin := (total_area * percentage_used / typing_width - sheet_length) / (-2)
  top_bottom_margin = 3 := by
  sorry

end NUMINAMATH_CALUDE_typist_margin_l373_37366


namespace NUMINAMATH_CALUDE_Ann_age_is_6_l373_37318

/-- Ann's current age -/
def Ann_age : ℕ := sorry

/-- Tom's current age -/
def Tom_age : ℕ := 2 * Ann_age

/-- The sum of their ages 10 years later -/
def sum_ages_later : ℕ := Ann_age + 10 + Tom_age + 10

theorem Ann_age_is_6 : Ann_age = 6 := by
  have h1 : sum_ages_later = 38 := sorry
  sorry

end NUMINAMATH_CALUDE_Ann_age_is_6_l373_37318


namespace NUMINAMATH_CALUDE_f_divides_characterization_l373_37330

def f (x : ℕ) : ℕ := x^2 + x + 1

def is_valid (n : ℕ) : Prop :=
  n = 1 ∨ 
  (Nat.Prime n ∧ n % 3 = 1) ∨ 
  (∃ p, Nat.Prime p ∧ p ≠ 3 ∧ n = p^2)

theorem f_divides_characterization (n : ℕ) :
  (∀ k : ℕ, k > 0 → k ∣ n → f k ∣ f n) ↔ is_valid n :=
sorry

end NUMINAMATH_CALUDE_f_divides_characterization_l373_37330


namespace NUMINAMATH_CALUDE_great_wall_scientific_notation_l373_37303

theorem great_wall_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 21200000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.12 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_great_wall_scientific_notation_l373_37303


namespace NUMINAMATH_CALUDE_soccer_goals_proof_l373_37311

theorem soccer_goals_proof (total_goals : ℕ) : 
  (total_goals / 3 : ℚ) + (total_goals / 5 : ℚ) + 8 + 20 = total_goals →
  20 ≤ 27 →
  ∃ (individual_goals : List ℕ), 
    individual_goals.length = 9 ∧ 
    individual_goals.sum = 20 ∧
    ∀ g ∈ individual_goals, g ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_soccer_goals_proof_l373_37311


namespace NUMINAMATH_CALUDE_rocket_max_height_l373_37387

/-- The height of a rocket as a function of time -/
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

/-- Theorem stating that the maximum height of the rocket is 175 meters -/
theorem rocket_max_height :
  ∃ (max_height : ℝ), max_height = 175 ∧ ∀ (t : ℝ), rocket_height t ≤ max_height :=
by sorry

end NUMINAMATH_CALUDE_rocket_max_height_l373_37387


namespace NUMINAMATH_CALUDE_agatha_bike_purchase_l373_37378

/-- Given Agatha's bike purchase scenario, prove the remaining amount for seat and handlebar tape. -/
theorem agatha_bike_purchase (total_budget : ℕ) (frame_cost : ℕ) (front_wheel_cost : ℕ) :
  total_budget = 60 →
  frame_cost = 15 →
  front_wheel_cost = 25 →
  total_budget - (frame_cost + front_wheel_cost) = 20 :=
by sorry

end NUMINAMATH_CALUDE_agatha_bike_purchase_l373_37378


namespace NUMINAMATH_CALUDE_beaumont_high_school_science_classes_beaumont_high_school_main_theorem_l373_37365

/-- The number of players taking at least two sciences at Beaumont High School -/
theorem beaumont_high_school_science_classes (total_players : ℕ) 
  (biology_players : ℕ) (chemistry_players : ℕ) (physics_players : ℕ) 
  (all_three_players : ℕ) : ℕ :=
by
  sorry

/-- The main theorem about Beaumont High School science classes -/
theorem beaumont_high_school_main_theorem : 
  beaumont_high_school_science_classes 30 15 10 5 3 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_beaumont_high_school_science_classes_beaumont_high_school_main_theorem_l373_37365


namespace NUMINAMATH_CALUDE_kendra_evening_minivans_l373_37345

/-- The number of minivans Kendra saw in the afternoon -/
def afternoon_minivans : ℕ := 4

/-- The total number of minivans Kendra saw -/
def total_minivans : ℕ := 5

/-- The number of minivans Kendra saw in the evening -/
def evening_minivans : ℕ := total_minivans - afternoon_minivans

theorem kendra_evening_minivans : evening_minivans = 1 := by
  sorry

end NUMINAMATH_CALUDE_kendra_evening_minivans_l373_37345


namespace NUMINAMATH_CALUDE_total_monthly_pay_is_12708_l373_37354

-- Define the structure for an employee
structure Employee where
  name : String
  hours_per_week : ℕ
  hourly_rate : ℕ

-- Define the list of employees
def employees : List Employee := [
  { name := "Fiona", hours_per_week := 40, hourly_rate := 20 },
  { name := "John", hours_per_week := 30, hourly_rate := 22 },
  { name := "Jeremy", hours_per_week := 25, hourly_rate := 18 },
  { name := "Katie", hours_per_week := 35, hourly_rate := 21 },
  { name := "Matt", hours_per_week := 28, hourly_rate := 19 }
]

-- Define the number of weeks in a month
def weeks_in_month : ℕ := 4

-- Calculate the monthly pay for all employees
def total_monthly_pay : ℕ :=
  employees.foldl (fun acc e => acc + e.hours_per_week * e.hourly_rate * weeks_in_month) 0

-- Theorem stating that the total monthly pay is $12,708
theorem total_monthly_pay_is_12708 : total_monthly_pay = 12708 := by
  sorry

end NUMINAMATH_CALUDE_total_monthly_pay_is_12708_l373_37354


namespace NUMINAMATH_CALUDE_janet_song_time_l373_37325

theorem janet_song_time (original_time : ℝ) (speed_increase : ℝ) (new_time : ℝ) : 
  original_time = 200 →
  speed_increase = 0.25 →
  new_time = original_time / (1 + speed_increase) →
  new_time = 160 := by
sorry

end NUMINAMATH_CALUDE_janet_song_time_l373_37325


namespace NUMINAMATH_CALUDE_min_employees_for_tech_company_l373_37359

/-- Calculates the minimum number of employees needed given the number of employees
    for hardware, software, and those working on both. -/
def min_employees (hardware : ℕ) (software : ℕ) (both : ℕ) : ℕ :=
  hardware + software - both

/-- Theorem stating that given 150 employees for hardware, 130 for software,
    and 50 for both, the minimum number of employees needed is 230. -/
theorem min_employees_for_tech_company :
  min_employees 150 130 50 = 230 := by
  sorry

#eval min_employees 150 130 50

end NUMINAMATH_CALUDE_min_employees_for_tech_company_l373_37359


namespace NUMINAMATH_CALUDE_cube_diagonal_l373_37356

theorem cube_diagonal (surface_area : ℝ) (h : surface_area = 294) :
  let side_length := Real.sqrt (surface_area / 6)
  let diagonal_length := side_length * Real.sqrt 3
  diagonal_length = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_l373_37356


namespace NUMINAMATH_CALUDE_bear_population_l373_37376

/-- The number of black bears in the park -/
def black_bears : ℕ := 60

/-- The number of white bears in the park -/
def white_bears : ℕ := black_bears / 2

/-- The number of brown bears in the park -/
def brown_bears : ℕ := black_bears + 40

/-- The total population of bears in the park -/
def total_bears : ℕ := white_bears + black_bears + brown_bears

theorem bear_population : total_bears = 190 := by
  sorry

end NUMINAMATH_CALUDE_bear_population_l373_37376


namespace NUMINAMATH_CALUDE_smaller_cup_radius_l373_37396

/-- The radius of smaller hemisphere-shaped cups when water from a large hemisphere
    is evenly distributed. -/
theorem smaller_cup_radius (R : ℝ) (n : ℕ) (h1 : R = 2) (h2 : n = 64) :
  ∃ r : ℝ, r > 0 ∧ n * ((2/3) * Real.pi * r^3) = (2/3) * Real.pi * R^3 ∧ r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cup_radius_l373_37396


namespace NUMINAMATH_CALUDE_last_four_digits_of_3_24000_l373_37370

theorem last_four_digits_of_3_24000 (h : 3^800 ≡ 1 [ZMOD 2000]) :
  3^24000 ≡ 1 [ZMOD 2000] := by sorry

end NUMINAMATH_CALUDE_last_four_digits_of_3_24000_l373_37370


namespace NUMINAMATH_CALUDE_total_pencils_l373_37391

def mitchell_pencils : ℕ := 30

def antonio_pencils : ℕ := mitchell_pencils - mitchell_pencils * 20 / 100

theorem total_pencils : mitchell_pencils + antonio_pencils = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l373_37391


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l373_37371

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l373_37371


namespace NUMINAMATH_CALUDE_books_about_trains_l373_37375

def books_about_animals : ℕ := 10
def books_about_space : ℕ := 1
def cost_per_book : ℕ := 16
def total_spent : ℕ := 224

theorem books_about_trains : ℕ := by
  sorry

end NUMINAMATH_CALUDE_books_about_trains_l373_37375


namespace NUMINAMATH_CALUDE_grandmas_salad_ratio_l373_37346

/-- Prove that the ratio of bacon bits to pickles is 4:1 given the conditions in Grandma's salad --/
theorem grandmas_salad_ratio : 
  ∀ (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ),
    mushrooms = 3 →
    cherry_tomatoes = 2 * mushrooms →
    pickles = 4 * cherry_tomatoes →
    red_bacon_bits = 32 →
    3 * red_bacon_bits = bacon_bits →
    (bacon_bits : ℚ) / pickles = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_salad_ratio_l373_37346


namespace NUMINAMATH_CALUDE_sine_function_parameters_l373_37398

theorem sine_function_parameters
  (A ω m : ℝ)
  (h_A_pos : A > 0)
  (h_ω_pos : ω > 0)
  (h_max : ∀ x, A * Real.sin (ω * x + π / 6) + m ≤ 3)
  (h_min : ∀ x, A * Real.sin (ω * x + π / 6) + m ≥ -5)
  (h_max_achieved : ∃ x, A * Real.sin (ω * x + π / 6) + m = 3)
  (h_min_achieved : ∃ x, A * Real.sin (ω * x + π / 6) + m = -5)
  (h_symmetry : ω * (π / 2) = π) :
  A = 4 ∧ ω = 2 ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_parameters_l373_37398


namespace NUMINAMATH_CALUDE_contractor_payment_example_l373_37329

/-- Calculates the total amount a contractor receives given the contract terms and absent days. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  (total_days - absent_days : ℚ) * payment_per_day - (absent_days : ℚ) * fine_per_day

/-- Theorem stating that under the given conditions, the contractor receives Rs. 425. -/
theorem contractor_payment_example : 
  contractor_payment 30 25 7.5 10 = 425 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_example_l373_37329


namespace NUMINAMATH_CALUDE_customers_who_tipped_l373_37377

/-- The number of customers who left a tip at 'The Greasy Spoon' restaurant -/
theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ) :
  initial_customers = 39 →
  additional_customers = 12 →
  non_tipping_customers = 49 →
  initial_customers + additional_customers - non_tipping_customers = 2 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tipped_l373_37377


namespace NUMINAMATH_CALUDE_snow_on_monday_l373_37324

theorem snow_on_monday (total_snow : ℝ) (tuesday_snow : ℝ) 
  (h1 : total_snow = 0.53)
  (h2 : tuesday_snow = 0.21) :
  total_snow - tuesday_snow = 0.53 - 0.21 := by
sorry

end NUMINAMATH_CALUDE_snow_on_monday_l373_37324


namespace NUMINAMATH_CALUDE_semicircle_radius_l373_37343

/-- The radius of a semicircle with perimeter 180 cm is equal to 180 / (π + 2) cm. -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 180) :
  ∃ r : ℝ, r = perimeter / (Real.pi + 2) ∧ r * (Real.pi + 2) = perimeter := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l373_37343


namespace NUMINAMATH_CALUDE_count_arithmetic_mean_subsets_l373_37385

/-- The number of three-element subsets of {1, 2, ..., n} where one element
    is the arithmetic mean of the other two. -/
def arithmeticMeanSubsets (n : ℕ) : ℕ :=
  (n / 2) * ((n - 1) / 2)

/-- Theorem stating that for any natural number n ≥ 3, the number of three-element
    subsets of {1, 2, ..., n} where one element is the arithmetic mean of the
    other two is equal to ⌊n/2⌋ * ⌊(n-1)/2⌋. -/
theorem count_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
  arithmeticMeanSubsets n = (n / 2) * ((n - 1) / 2) := by
  sorry

#check count_arithmetic_mean_subsets

end NUMINAMATH_CALUDE_count_arithmetic_mean_subsets_l373_37385


namespace NUMINAMATH_CALUDE_sons_age_l373_37367

theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 22 →
  (man_age + 2) = 2 * (son_age + 2) →
  son_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_sons_age_l373_37367


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l373_37353

theorem amusement_park_tickets 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_paid : ℕ) 
  (child_tickets : ℕ) : 
  adult_price = 8 → 
  child_price = 5 → 
  total_paid = 201 → 
  child_tickets = 21 → 
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + child_price * child_tickets = total_paid ∧ 
    adult_tickets + child_tickets = 33 :=
by
  sorry

#check amusement_park_tickets

end NUMINAMATH_CALUDE_amusement_park_tickets_l373_37353


namespace NUMINAMATH_CALUDE_different_choices_four_two_l373_37363

/-- The number of ways two people can choose courses differently from a set of courses -/
def differentChoices (totalCourses : ℕ) (coursesPerPerson : ℕ) : ℕ :=
  (totalCourses.choose coursesPerPerson)^2 - totalCourses.choose coursesPerPerson

/-- Theorem: Given 4 courses and 2 courses per person, there are 30 ways to choose differently -/
theorem different_choices_four_two : differentChoices 4 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_choices_four_two_l373_37363


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l373_37328

/-- The total bill for a group of friends dining at a restaurant -/
def total_bill : ℕ := 270

/-- The number of friends dining at the restaurant -/
def num_friends : ℕ := 10

/-- The extra amount each paying friend contributes to cover the non-paying friend -/
def extra_contribution : ℕ := 3

/-- The number of friends who pay the bill -/
def num_paying_friends : ℕ := num_friends - 1

theorem restaurant_bill_proof :
  total_bill = num_paying_friends * (total_bill / num_friends + extra_contribution) :=
sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l373_37328


namespace NUMINAMATH_CALUDE_runners_photo_probability_l373_37384

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the track and photo setup -/
structure TrackSetup where
  photo_fraction : ℝ
  photo_time : ℝ

/-- Calculates the probability of both runners being in the photo -/
def probability_both_in_photo (ellie sam : Runner) (setup : TrackSetup) : ℝ :=
  sorry

/-- The main theorem statement -/
theorem runners_photo_probability :
  let ellie : Runner := { lap_time := 120, direction := true }
  let sam : Runner := { lap_time := 75, direction := false }
  let setup : TrackSetup := { photo_fraction := 1/3, photo_time := 600 }
  probability_both_in_photo ellie sam setup = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_runners_photo_probability_l373_37384


namespace NUMINAMATH_CALUDE_bottles_left_after_purchase_l373_37350

/-- Given a store shelf with bottles of milk, prove the number of bottles left after purchases. -/
theorem bottles_left_after_purchase (initial : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) 
  (h1 : initial = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6) :
  initial - (jason_buys + harry_buys) = 24 := by
  sorry

#check bottles_left_after_purchase

end NUMINAMATH_CALUDE_bottles_left_after_purchase_l373_37350


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l373_37302

-- Define the parabola function
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- Theorem: The parabola has exactly one x-intercept
theorem parabola_one_x_intercept : 
  ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l373_37302


namespace NUMINAMATH_CALUDE_complex_sum_real_l373_37372

theorem complex_sum_real (a : ℝ) : 
  let z₁ : ℂ := (16 / (a + 5)) - (10 - a^2) * I
  let z₂ : ℂ := (2 / (1 - a)) + (2*a - 5) * I
  (z₁ + z₂).im = 0 → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_real_l373_37372


namespace NUMINAMATH_CALUDE_student_count_l373_37320

theorem student_count (total : ℕ) 
  (h1 : total / 5 + total / 4 + total / 2 + 30 = total) : total = 600 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l373_37320


namespace NUMINAMATH_CALUDE_photographer_choices_l373_37380

theorem photographer_choices (n : ℕ) (k₁ k₂ : ℕ) (h₁ : n = 7) (h₂ : k₁ = 4) (h₃ : k₂ = 5) :
  Nat.choose n k₁ + Nat.choose n k₂ = 56 :=
by sorry

end NUMINAMATH_CALUDE_photographer_choices_l373_37380


namespace NUMINAMATH_CALUDE_hiking_route_length_l373_37386

/-- The total length of the hiking route in kilometers. -/
def total_length : ℝ := 150

/-- The initial distance walked on foot in kilometers. -/
def initial_walk : ℝ := 30

/-- The fraction of the remaining route traveled by raft. -/
def raft_fraction : ℝ := 0.2

/-- The multiplier for the second walking distance compared to the raft distance. -/
def second_walk_multiplier : ℝ := 1.5

/-- The speed of the truck in km/h. -/
def truck_speed : ℝ := 40

/-- The time spent on the truck in hours. -/
def truck_time : ℝ := 1.5

theorem hiking_route_length :
  initial_walk +
  raft_fraction * (total_length - initial_walk) +
  second_walk_multiplier * (raft_fraction * (total_length - initial_walk)) +
  truck_speed * truck_time = total_length := by sorry

end NUMINAMATH_CALUDE_hiking_route_length_l373_37386


namespace NUMINAMATH_CALUDE_hcf_of_specific_numbers_l373_37362

/-- Given two positive integers with a product of 363 and the greater number being 33,
    prove that their highest common factor (HCF) is 11. -/
theorem hcf_of_specific_numbers :
  ∀ A B : ℕ+,
  A * B = 363 →
  A = 33 →
  A > B →
  Nat.gcd A.val B.val = 11 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_specific_numbers_l373_37362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l373_37339

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l373_37339


namespace NUMINAMATH_CALUDE_smallest_purple_balls_l373_37312

theorem smallest_purple_balls (x : ℕ) (y : ℕ) : 
  x > 0 ∧ 
  x % 10 = 0 ∧ 
  x % 8 = 0 ∧ 
  x % 3 = 0 ∧
  x / 8 + 10 = 8 ∧
  y = x / 10 + x / 8 + x / 3 + (x / 10 + 9) + 8 + x / 8 - x ∧
  y > 0 →
  y ≥ 25 :=
sorry

end NUMINAMATH_CALUDE_smallest_purple_balls_l373_37312


namespace NUMINAMATH_CALUDE_exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary_l373_37374

/-- Represents the outcome of drawing balls from a bag -/
inductive BallDraw
  | oneWhite
  | twoWhite
  | threeWhite
  | allBlack

/-- The set of all possible outcomes when drawing 3 balls from a bag with 3 white and 4 black balls -/
def allOutcomes : Set BallDraw := {BallDraw.oneWhite, BallDraw.twoWhite, BallDraw.threeWhite, BallDraw.allBlack}

/-- The event of drawing exactly one white ball -/
def exactlyOneWhite : Set BallDraw := {BallDraw.oneWhite}

/-- The event of drawing all white balls -/
def allWhite : Set BallDraw := {BallDraw.threeWhite}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set BallDraw) : Prop := A ∩ B = ∅

/-- Two events are complementary if their union is the set of all outcomes -/
def complementary (A B : Set BallDraw) : Prop := A ∪ B = allOutcomes

theorem exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneWhite allWhite ∧ ¬complementary exactlyOneWhite allWhite :=
sorry

end NUMINAMATH_CALUDE_exactly_one_white_and_all_white_mutually_exclusive_but_not_complementary_l373_37374


namespace NUMINAMATH_CALUDE_inequality_proof_l373_37321

theorem inequality_proof (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4*a*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l373_37321


namespace NUMINAMATH_CALUDE_sum_reciprocals_l373_37322

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / (1 + ω)) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l373_37322


namespace NUMINAMATH_CALUDE_milk_fraction_in_second_cup_l373_37308

theorem milk_fraction_in_second_cup 
  (V : ℝ) -- Volume of each cup
  (x : ℝ) -- Fraction of milk in the second cup
  (h1 : V > 0) -- Volume is positive
  (h2 : 0 ≤ x ∧ x ≤ 1) -- x is a valid fraction
  : ((2/5 * V + (1 - x) * V) / ((3/5 * V + x * V))) = 3/7 → x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_milk_fraction_in_second_cup_l373_37308


namespace NUMINAMATH_CALUDE_triangular_floor_area_l373_37301

theorem triangular_floor_area (length_feet : ℝ) (width_feet : ℝ) (feet_per_yard : ℝ) : 
  length_feet = 15 → width_feet = 12 → feet_per_yard = 3 →
  (1 / 2) * (length_feet / feet_per_yard) * (width_feet / feet_per_yard) = 10 := by
sorry

end NUMINAMATH_CALUDE_triangular_floor_area_l373_37301


namespace NUMINAMATH_CALUDE_twelve_students_pairs_l373_37388

/-- The number of unique pairs that can be formed from a group of n elements -/
def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of unique pairs from a group of 12 students is 66 -/
theorem twelve_students_pairs : number_of_pairs 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_students_pairs_l373_37388


namespace NUMINAMATH_CALUDE_equal_division_of_money_l373_37336

/-- Proves that when $3.75 is equally divided among 3 people, each person receives $1.25. -/
theorem equal_division_of_money (total_amount : ℚ) (num_people : ℕ) :
  total_amount = 3.75 ∧ num_people = 3 →
  total_amount / (num_people : ℚ) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_money_l373_37336


namespace NUMINAMATH_CALUDE_hexagon_y_coordinate_l373_37305

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon with six vertices -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

/-- Calculates the area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- Checks if a hexagon has a vertical line of symmetry -/
def hasVerticalSymmetry (h : Hexagon) : Prop := sorry

theorem hexagon_y_coordinate 
  (h : Hexagon)
  (symm : hasVerticalSymmetry h)
  (aCoord : h.A = ⟨0, 0⟩)
  (bCoord : h.B = ⟨0, 6⟩)
  (eCoord : h.E = ⟨4, 0⟩)
  (dCoord : h.D.x = 4)
  (area : hexagonArea h = 58) :
  h.D.y = 14.5 := by sorry

end NUMINAMATH_CALUDE_hexagon_y_coordinate_l373_37305


namespace NUMINAMATH_CALUDE_tangent_circle_rectangle_existence_l373_37334

/-- Given a circle of radius r tangent to the legs of a right angle,
    there exists a point M on the circumference forming a rectangle MPOQ
    with perimeter 2p if and only if r(2 - √2) ≤ p ≤ r(2 + √2) -/
theorem tangent_circle_rectangle_existence (r p : ℝ) (hr : r > 0) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = p ∧ x^2 + y^2 = 2*r*p) ↔
  r*(2 - Real.sqrt 2) ≤ p ∧ p ≤ r*(2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_rectangle_existence_l373_37334


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l373_37319

def average_salary_jan_to_apr : ℝ := 8000
def average_salary_some_months : ℝ := 8800
def salary_may : ℝ := 6500
def salary_jan : ℝ := 3300

theorem average_salary_feb_to_may :
  let total_salary_jan_to_apr := average_salary_jan_to_apr * 4
  let total_salary_feb_to_apr := total_salary_jan_to_apr - salary_jan
  let total_salary_feb_to_may := total_salary_feb_to_apr + salary_may
  total_salary_feb_to_may / 4 = average_salary_some_months :=
by sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l373_37319


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l373_37379

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x, -x^2 + b*x - 7 < 0 ↔ x < 2 ∨ x > 6) → b = 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l373_37379


namespace NUMINAMATH_CALUDE_a_minus_c_equals_three_l373_37369

theorem a_minus_c_equals_three (a b c d : ℝ) 
  (h1 : a - b = c + d + 9)
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_three_l373_37369


namespace NUMINAMATH_CALUDE_rhombus_area_l373_37309

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 100 square units. -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 145 →
  d₂ - d₁ = 10 →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  (1/2) * d₁ * d₂ = 100 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l373_37309


namespace NUMINAMATH_CALUDE_max_value_x_y3_z4_l373_37351

theorem max_value_x_y3_z4 (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 1) :
  ∃ (max : ℝ), max = 1 ∧ ∀ (a b c : ℝ), 
    a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 1 → 
    a + b^3 + c^4 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_y3_z4_l373_37351


namespace NUMINAMATH_CALUDE_coefficient_x4_in_polynomial_product_l373_37392

theorem coefficient_x4_in_polynomial_product : 
  let p1 : Polynomial ℤ := X^5 - 4*X^4 + 3*X^3 - 2*X^2 + X - 1
  let p2 : Polynomial ℤ := 3*X^2 - X + 5
  (p1 * p2).coeff 4 = -13 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_polynomial_product_l373_37392


namespace NUMINAMATH_CALUDE_no_fogh_prime_l373_37397

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem no_fogh_prime :
  ¬∃ (f o g h : ℕ),
    is_digit f ∧ is_digit o ∧ is_digit g ∧ is_digit h ∧
    f ≠ o ∧ f ≠ g ∧ f ≠ h ∧ o ≠ g ∧ o ≠ h ∧ g ≠ h ∧
    (1000 * f + 100 * o + 10 * g + h ≥ 1000) ∧
    (1000 * f + 100 * o + 10 * g + h < 10000) ∧
    is_prime (1000 * f + 100 * o + 10 * g + h) ∧
    (1000 * f + 100 * o + 10 * g + h) * (f * o * g * h) = (1000 * f + 100 * o + 10 * g + h) :=
sorry

end NUMINAMATH_CALUDE_no_fogh_prime_l373_37397


namespace NUMINAMATH_CALUDE_watch_cost_price_l373_37373

-- Define the cost price of the watch
def cost_price : ℝ := 1166.67

-- Define the selling price at 10% loss
def selling_price_loss : ℝ := 0.90 * cost_price

-- Define the selling price at 2% gain
def selling_price_gain : ℝ := 1.02 * cost_price

-- Theorem statement
theorem watch_cost_price :
  (selling_price_loss = 0.90 * cost_price) ∧
  (selling_price_gain = 1.02 * cost_price) ∧
  (selling_price_gain = selling_price_loss + 140) →
  cost_price = 1166.67 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l373_37373


namespace NUMINAMATH_CALUDE_hawkeye_battery_charges_l373_37348

def battery_problem (cost_per_charge : ℚ) (initial_budget : ℚ) (remaining_money : ℚ) : Prop :=
  let total_spent : ℚ := initial_budget - remaining_money
  let number_of_charges : ℚ := total_spent / cost_per_charge
  number_of_charges = 4

theorem hawkeye_battery_charges : 
  battery_problem (35/10) 20 6 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_battery_charges_l373_37348


namespace NUMINAMATH_CALUDE_robin_chocolate_chip_cookies_l373_37317

/-- Given information about Robin's cookies --/
structure CookieInfo where
  cookies_per_bag : ℕ
  oatmeal_cookies : ℕ
  baggies : ℕ

/-- Calculate the number of chocolate chip cookies --/
def chocolate_chip_cookies (info : CookieInfo) : ℕ :=
  info.baggies * info.cookies_per_bag - info.oatmeal_cookies

/-- Theorem: Robin has 23 chocolate chip cookies --/
theorem robin_chocolate_chip_cookies :
  let info : CookieInfo := {
    cookies_per_bag := 6,
    oatmeal_cookies := 25,
    baggies := 8
  }
  chocolate_chip_cookies info = 23 := by
  sorry

end NUMINAMATH_CALUDE_robin_chocolate_chip_cookies_l373_37317


namespace NUMINAMATH_CALUDE_student_hamster_difference_l373_37315

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 20

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 1

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of hamsters in all classrooms -/
def total_hamsters : ℕ := num_classrooms * hamsters_per_classroom

theorem student_hamster_difference :
  total_students - total_hamsters = 95 := by
  sorry

end NUMINAMATH_CALUDE_student_hamster_difference_l373_37315


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l373_37335

/-- Theorem: Given a cubic function f(x) = -2x^3 - x + 1 defined on the interval [m, n],
    where f(m)f(n) < 0, the equation f(x) = 0 has exactly one real root in the interval [m, n]. -/
theorem unique_root_in_interval (m n : ℝ) (h : m ≤ n) :
  let f : ℝ → ℝ := λ x ↦ -2 * x^3 - x + 1
  (f m) * (f n) < 0 →
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l373_37335


namespace NUMINAMATH_CALUDE_not_p_or_not_q_must_be_true_l373_37306

theorem not_p_or_not_q_must_be_true (h1 : ¬(p ∧ q)) (h2 : p ∨ q) : ¬p ∨ ¬q :=
by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_must_be_true_l373_37306


namespace NUMINAMATH_CALUDE_cheddar_package_size_l373_37332

/-- The number of slices in a package of Swiss cheese -/
def swiss_slices_per_package : ℕ := 28

/-- The total number of slices bought for each type of cheese -/
def total_slices_per_type : ℕ := 84

/-- The number of slices in a package of cheddar cheese -/
def cheddar_slices_per_package : ℕ := sorry

theorem cheddar_package_size :
  cheddar_slices_per_package = swiss_slices_per_package :=
by
  sorry

#check cheddar_package_size

end NUMINAMATH_CALUDE_cheddar_package_size_l373_37332


namespace NUMINAMATH_CALUDE_simplify_expression_l373_37300

theorem simplify_expression : (((3 + 4 + 6 + 7) / 3) + ((3 * 6 + 9) / 4)) = 161 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l373_37300


namespace NUMINAMATH_CALUDE_flagpole_break_height_l373_37383

theorem flagpole_break_height :
  let initial_height : ℝ := 10
  let horizontal_distance : ℝ := 3
  let break_height : ℝ := Real.sqrt 109 / 2
  (break_height^2 + horizontal_distance^2 = (initial_height - break_height)^2) ∧
  (2 * break_height = Real.sqrt (horizontal_distance^2 + initial_height^2)) :=
by sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l373_37383


namespace NUMINAMATH_CALUDE_fence_cost_l373_37304

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 →
  price_per_foot = 60 →
  cost = 4 * Real.sqrt area * price_per_foot →
  cost = 4080 := by
  sorry


end NUMINAMATH_CALUDE_fence_cost_l373_37304


namespace NUMINAMATH_CALUDE_root_implies_m_value_l373_37327

theorem root_implies_m_value (m : ℝ) : (3^2 - 4*3 + m = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l373_37327


namespace NUMINAMATH_CALUDE_product_of_xy_l373_37337

theorem product_of_xy (x y z w : ℕ+) 
  (h1 : x = w)
  (h2 : y = z)
  (h3 : w + w = w * w)
  (h4 : y = w)
  (h5 : z = 3) :
  x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_xy_l373_37337


namespace NUMINAMATH_CALUDE_students_per_class_l373_37360

/-- Prove the number of students per class in a school's reading program -/
theorem students_per_class (c : ℕ) (h1 : c > 0) : 
  let books_per_student_per_year := 5 * 12
  let total_books_read := 60
  let s := total_books_read / (c * books_per_student_per_year)
  s = 1 / c := by
  sorry

end NUMINAMATH_CALUDE_students_per_class_l373_37360


namespace NUMINAMATH_CALUDE_unique_five_digit_sum_l373_37313

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (λ acc d => acc * 10 + d) 0

theorem unique_five_digit_sum (n : ℕ) : 
  is_five_digit n ∧ 
  (∃ (pos : Fin 5), n + remove_digit n pos = 54321) ↔ 
  n = 49383 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_sum_l373_37313


namespace NUMINAMATH_CALUDE_probability_two_same_color_l373_37347

def total_balls : ℕ := 6
def balls_per_color : ℕ := 2
def num_colors : ℕ := 3
def balls_drawn : ℕ := 3

def total_ways : ℕ := Nat.choose total_balls balls_drawn

def ways_two_same_color : ℕ := num_colors * (Nat.choose balls_per_color 2) * (total_balls - balls_per_color)

theorem probability_two_same_color :
  (ways_two_same_color : ℚ) / total_ways = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_two_same_color_l373_37347


namespace NUMINAMATH_CALUDE_function_transformation_l373_37331

/-- Given a function f such that f(x-1) = 2x^2 + 3x for all x,
    prove that f(x) = 2x^2 + 7x + 5 for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = 2 * x^2 + 3 * x) : 
    ∀ x, f x = 2 * x^2 + 7 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l373_37331


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l373_37364

/-- The probability that a randomly chosen point in a circular region with radius 3
    is closer to the center than to the boundary is 1/4. -/
theorem probability_closer_to_center (r : ℝ) (h : r = 3) : 
  (π * (r/2)^2) / (π * r^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l373_37364


namespace NUMINAMATH_CALUDE_is_fractional_expression_example_l373_37340

/-- A fractional expression is an expression where the denominator contains a variable. -/
def IsFractionalExpression (n d : ℝ → ℝ) : Prop :=
  ∃ x, d x ≠ 0 ∧ (∀ y, d y ≠ d x)

/-- The expression (x + 3) / x is a fractional expression. -/
theorem is_fractional_expression_example :
  IsFractionalExpression (λ x => x + 3) (λ x => x) := by
  sorry

end NUMINAMATH_CALUDE_is_fractional_expression_example_l373_37340


namespace NUMINAMATH_CALUDE_sqrt_sum_equivalence_l373_37395

theorem sqrt_sum_equivalence (n : ℝ) (h : Real.sqrt 15 = n) :
  Real.sqrt 0.15 + Real.sqrt 1500 = (101 / 10) * n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equivalence_l373_37395


namespace NUMINAMATH_CALUDE_problem_statement_l373_37326

theorem problem_statement (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l373_37326


namespace NUMINAMATH_CALUDE_sequence_property_l373_37381

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sequence_property (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) + 1 = a n) : 
  is_arithmetic_sequence a (-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l373_37381


namespace NUMINAMATH_CALUDE_left_square_side_length_l373_37382

theorem left_square_side_length :
  ∀ (x : ℝ),
  (∃ (y z : ℝ),
    y = x + 17 ∧
    z = y - 6 ∧
    x + y + z = 52) →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l373_37382


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_attained_l373_37389

/-- The function f(x) = (x^2 + 2) / √(x^2 + 1) has a minimum value of 2 for all real x -/
theorem min_value_of_f (x : ℝ) : (x^2 + 2) / Real.sqrt (x^2 + 1) ≥ 2 := by
  sorry

/-- The minimum value 2 is attained when x = 0 -/
theorem min_value_attained : ∃ x : ℝ, (x^2 + 2) / Real.sqrt (x^2 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_attained_l373_37389


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l373_37357

theorem complex_magnitude_proof : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / i
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l373_37357


namespace NUMINAMATH_CALUDE_optimal_base_side_l373_37341

/-- A lidless water tank with square base -/
structure WaterTank where
  volume : ℝ
  baseSide : ℝ
  height : ℝ

/-- The surface area of a lidless water tank -/
def surfaceArea (tank : WaterTank) : ℝ :=
  tank.baseSide ^ 2 + 4 * tank.baseSide * tank.height

/-- The volume constraint for the water tank -/
def volumeConstraint (tank : WaterTank) : Prop :=
  tank.volume = tank.baseSide ^ 2 * tank.height

/-- Theorem: The base side length that minimizes the surface area of a lidless water tank
    with volume 256 cubic units and a square base is 8 units -/
theorem optimal_base_side :
  ∃ (tank : WaterTank),
    tank.volume = 256 ∧
    volumeConstraint tank ∧
    (∀ (other : WaterTank),
      other.volume = 256 →
      volumeConstraint other →
      surfaceArea tank ≤ surfaceArea other) ∧
    tank.baseSide = 8 :=
  sorry

end NUMINAMATH_CALUDE_optimal_base_side_l373_37341


namespace NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l373_37358

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the center and left focus
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the dot product of OP and FP
def dot_product (x y : ℝ) : ℝ := x^2 + x + y^2

theorem min_dot_product_on_ellipse :
  ∀ x y : ℝ, is_on_ellipse x y →
  dot_product x y ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_ellipse_l373_37358


namespace NUMINAMATH_CALUDE_find_n_l373_37352

theorem find_n : ∀ n : ℤ, (∀ x : ℝ, (x - 2) * (x + 1) = x^2 + n*x - 2) → n = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l373_37352


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l373_37342

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l373_37342


namespace NUMINAMATH_CALUDE_min_surface_area_circumscribed_sphere_l373_37310

/-- The minimum surface area of a sphere circumscribed around a right rectangular prism --/
theorem min_surface_area_circumscribed_sphere (h : ℝ) (a : ℝ) :
  h = 3 →
  a * a = 7 / 2 →
  ∃ (S : ℝ), S = 16 * Real.pi ∧ ∀ (R : ℝ), R ≥ 2 → 4 * Real.pi * R^2 ≥ S :=
by sorry

end NUMINAMATH_CALUDE_min_surface_area_circumscribed_sphere_l373_37310


namespace NUMINAMATH_CALUDE_derivative_tan_cot_l373_37399

open Real

theorem derivative_tan_cot (x : ℝ) (k : ℤ) : 
  (∀ k, x ≠ (2 * k + 1) * π / 2 → deriv tan x = 1 / (cos x)^2) ∧
  (∀ k, x ≠ k * π → deriv cot x = -(1 / (sin x)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_tan_cot_l373_37399


namespace NUMINAMATH_CALUDE_inequality_solution_and_absolute_value_bound_l373_37316

-- Define the solution set
def solution_set : Set ℝ := Set.Icc (-1) 2

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |2 * x - a| ≤ 3

-- Theorem statement
theorem inequality_solution_and_absolute_value_bound (a : ℝ) :
  (∀ x, x ∈ solution_set ↔ inequality a x) →
  (a = 1 ∧ ∀ x m, |x - m| < a → |x| < |m| + 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_and_absolute_value_bound_l373_37316


namespace NUMINAMATH_CALUDE_line_properties_l373_37314

-- Define a type for lines in 2D Cartesian plane
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a function to check if two lines are perpendicular
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

-- Define a function to represent a line passing through (1,1) with slope 1
def line_through_1_1_slope_1 (x : ℝ) : Prop :=
  x ≠ 1 → ∃ y : ℝ, y - 1 = x - 1

-- Theorem stating the properties we want to prove
theorem line_properties :
  ∀ (l1 l2 : Line),
    (are_perpendicular l1 l2 → l1.slope * l2.slope = -1) ∧
    line_through_1_1_slope_1 2 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l373_37314


namespace NUMINAMATH_CALUDE_yahs_to_bahs_1500_l373_37390

/-- Conversion rates between bahs, rahs, and yahs -/
structure ConversionRates where
  bah_to_rah : ℚ
  rah_to_yah : ℚ

/-- Given conversion rates, calculate the number of bahs equivalent to a given number of yahs -/
def yahs_to_bahs (rates : ConversionRates) (yahs : ℚ) : ℚ :=
  yahs * rates.rah_to_yah⁻¹ * rates.bah_to_rah⁻¹

/-- Theorem stating the equivalence of 1500 yahs to 562.5 bahs given the specified conversion rates -/
theorem yahs_to_bahs_1500 :
  let rates : ConversionRates := ⟨16/10, 20/12⟩
  yahs_to_bahs rates 1500 = 562.5 := by
  sorry

end NUMINAMATH_CALUDE_yahs_to_bahs_1500_l373_37390
