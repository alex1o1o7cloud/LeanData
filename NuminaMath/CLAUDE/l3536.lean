import Mathlib

namespace NUMINAMATH_CALUDE_no_nonneg_int_solutions_l3536_353683

theorem no_nonneg_int_solutions : 
  ¬∃ (x : ℕ), 4 * (x - 2) > 2 * (3 * x + 5) := by
sorry

end NUMINAMATH_CALUDE_no_nonneg_int_solutions_l3536_353683


namespace NUMINAMATH_CALUDE_no_wobbly_multiple_iff_div_10_or_25_l3536_353693

/-- A wobbly number is a positive integer whose digits in base 10 are alternatively non-zero and zero, with the units digit being non-zero. -/
def IsWobbly (n : ℕ) : Prop := sorry

/-- Theorem: A positive integer n does not divide any wobbly number if and only if n is divisible by 10 or 25. -/
theorem no_wobbly_multiple_iff_div_10_or_25 (n : ℕ) (hn : n > 0) :
  (∀ w : ℕ, IsWobbly w → ¬(w % n = 0)) ↔ (n % 10 = 0 ∨ n % 25 = 0) := by sorry

end NUMINAMATH_CALUDE_no_wobbly_multiple_iff_div_10_or_25_l3536_353693


namespace NUMINAMATH_CALUDE_contact_box_price_l3536_353647

/-- The price of a box of contacts given the number of contacts and cost per contact -/
def box_price (num_contacts : ℕ) (cost_per_contact : ℚ) : ℚ :=
  num_contacts * cost_per_contact

/-- The cost per contact for a box given its total price and number of contacts -/
def cost_per_contact (total_price : ℚ) (num_contacts : ℕ) : ℚ :=
  total_price / num_contacts

theorem contact_box_price :
  let box1_contacts : ℕ := 50
  let box2_contacts : ℕ := 99
  let box2_price : ℚ := 33

  let box2_cost_per_contact := cost_per_contact box2_price box2_contacts
  let chosen_cost_per_contact : ℚ := 1 / 3

  box_price box1_contacts chosen_cost_per_contact = 50 * (1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_contact_box_price_l3536_353647


namespace NUMINAMATH_CALUDE_group_size_from_average_age_change_l3536_353698

theorem group_size_from_average_age_change (N : ℕ) (T : ℕ) : 
  N > 0 → 
  (T : ℚ) / N - 3 = (T - 42 + 12 : ℚ) / N → 
  N = 10 := by
sorry

end NUMINAMATH_CALUDE_group_size_from_average_age_change_l3536_353698


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_monotone_increasing_sufficiency_l3536_353679

/-- A function f is monotonically increasing on an interval (a, +∞) if for any x₁, x₂ in the interval
    where x₁ < x₂, we have f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ → f x₁ < f x₂

/-- The function f(x) = x^2 + mx - 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

/-- Theorem: If f(x) = x^2 + mx - 2 is monotonically increasing on (2, +∞), then m ≥ -4 -/
theorem monotone_increasing_condition (m : ℝ) :
  MonotonicallyIncreasing (f m) 2 → m ≥ -4 := by
  sorry

/-- Theorem: If m ≥ -4, then f(x) = x^2 + mx - 2 is monotonically increasing on (2, +∞) -/
theorem monotone_increasing_sufficiency (m : ℝ) :
  m ≥ -4 → MonotonicallyIncreasing (f m) 2 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_monotone_increasing_sufficiency_l3536_353679


namespace NUMINAMATH_CALUDE_equation_solution_l3536_353676

theorem equation_solution : ∃ x : ℝ, (3034 - (1002 / x) = 2984) ∧ x = 20.04 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3536_353676


namespace NUMINAMATH_CALUDE_product_of_roots_eq_one_l3536_353617

theorem product_of_roots_eq_one : 
  ∃ (r₁ r₂ : ℝ), r₁ * r₂ = 1 ∧ r₁^(2*Real.log r₁) = ℯ ∧ r₂^(2*Real.log r₂) = ℯ ∧
  ∀ (x : ℝ), x^(2*Real.log x) = ℯ → x = r₁ ∨ x = r₂ :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_eq_one_l3536_353617


namespace NUMINAMATH_CALUDE_price_restoration_l3536_353630

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) : 
  reduced_price = 0.8 * original_price → 
  reduced_price * 1.25 = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_restoration_l3536_353630


namespace NUMINAMATH_CALUDE_songs_downloaded_l3536_353661

theorem songs_downloaded (internet_speed : ℕ) (song_size : ℕ) (download_time : ℕ) : 
  internet_speed = 20 → 
  song_size = 5 → 
  download_time = 1800 → 
  (internet_speed * download_time) / song_size = 7200 :=
by
  sorry

end NUMINAMATH_CALUDE_songs_downloaded_l3536_353661


namespace NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_optimal_l3536_353696

theorem right_triangle_inequality (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (side_order : a ≤ b ∧ b < c) :
  a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ (2 + 3 * Real.sqrt 2) * a * b * c :=
by sorry

theorem right_triangle_inequality_optimal (k : ℝ) 
  (h : ∀ (a b c : ℝ), a^2 + b^2 = c^2 → a ≤ b → b < c → 
    a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b) ≥ k * a * b * c) :
  k ≤ 2 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_right_triangle_inequality_optimal_l3536_353696


namespace NUMINAMATH_CALUDE_jane_inspection_fraction_l3536_353678

theorem jane_inspection_fraction :
  ∀ (P : ℝ) (J : ℝ),
    P > 0 →
    J > 0 →
    J < 1 →
    0.005 * (1 - J) * P + 0.008 * J * P = 0.0075 * P →
    J = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_jane_inspection_fraction_l3536_353678


namespace NUMINAMATH_CALUDE_sequence_equality_l3536_353638

theorem sequence_equality (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, (a (n + 1))^2 + (a n)^2 + 1 = 2 * ((a (n + 1)) * (a n) + (a (n + 1)) - (a n))) :
  ∀ n : ℕ, a n = n := by
sorry

end NUMINAMATH_CALUDE_sequence_equality_l3536_353638


namespace NUMINAMATH_CALUDE_paulas_friends_l3536_353682

/-- Given the initial number of candies, additional candies bought, and candies per friend,
    prove that the number of friends is equal to the total number of candies divided by the number of candies per friend. -/
theorem paulas_friends (initial_candies additional_candies candies_per_friend : ℕ) 
  (h1 : initial_candies = 20)
  (h2 : additional_candies = 4)
  (h3 : candies_per_friend = 4)
  : (initial_candies + additional_candies) / candies_per_friend = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_paulas_friends_l3536_353682


namespace NUMINAMATH_CALUDE_cube_root_27_minus_2_l3536_353612

theorem cube_root_27_minus_2 : (27 : ℝ) ^ (1/3) - 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_minus_2_l3536_353612


namespace NUMINAMATH_CALUDE_value_of_a_l3536_353628

theorem value_of_a (a b c : ℤ) 
  (eq1 : a + b = 12) 
  (eq2 : b + c = 16) 
  (eq3 : c = 7) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3536_353628


namespace NUMINAMATH_CALUDE_inverse_89_mod_91_l3536_353673

theorem inverse_89_mod_91 : ∃ x : ℕ, x < 91 ∧ (89 * x) % 91 = 1 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_89_mod_91_l3536_353673


namespace NUMINAMATH_CALUDE_trail_mix_nuts_l3536_353621

theorem trail_mix_nuts (walnuts almonds : ℚ) 
  (h1 : walnuts = 0.25)
  (h2 : almonds = 0.25) : 
  walnuts + almonds = 0.50 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_nuts_l3536_353621


namespace NUMINAMATH_CALUDE_sin_105_degrees_l3536_353631

theorem sin_105_degrees : 
  Real.sin (105 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_sin_105_degrees_l3536_353631


namespace NUMINAMATH_CALUDE_popsicle_sticks_per_boy_l3536_353660

theorem popsicle_sticks_per_boy (num_boys num_girls : ℕ) (sticks_per_girl : ℕ) (diff : ℕ) :
  num_boys = 10 →
  num_girls = 12 →
  sticks_per_girl = 12 →
  num_girls * sticks_per_girl + diff = num_boys * (num_girls * sticks_per_girl + diff) / num_boys →
  diff = 6 →
  (num_girls * sticks_per_girl + diff) / num_boys = 15 :=
by sorry

end NUMINAMATH_CALUDE_popsicle_sticks_per_boy_l3536_353660


namespace NUMINAMATH_CALUDE_power_fraction_equality_l3536_353655

theorem power_fraction_equality : (16^6 * 8^3) / 4^10 = 2^13 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l3536_353655


namespace NUMINAMATH_CALUDE_f_and_g_properties_l3536_353667

-- Define the functions
def f (x : ℝ) : ℝ := 1 + x^2
def g (x : ℝ) : ℝ := |x| + 1

-- Define evenness
def is_even (h : ℝ → ℝ) : Prop := ∀ x, h x = h (-x)

-- Define monotonically decreasing on (-∞, 0)
def is_decreasing_neg (h : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → h x > h y

theorem f_and_g_properties :
  (is_even f ∧ is_decreasing_neg f) ∧
  (is_even g ∧ is_decreasing_neg g) := by sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l3536_353667


namespace NUMINAMATH_CALUDE_janessa_baseball_cards_l3536_353601

/-- Janessa's baseball card collection problem -/
theorem janessa_baseball_cards
  (initial_cards : ℕ)
  (father_cards : ℕ)
  (ebay_cards : ℕ)
  (bad_cards : ℕ)
  (cards_given_to_dexter : ℕ)
  (h1 : initial_cards = 4)
  (h2 : father_cards = 13)
  (h3 : ebay_cards = 36)
  (h4 : bad_cards = 4)
  (h5 : cards_given_to_dexter = 29) :
  initial_cards + father_cards + ebay_cards - bad_cards - cards_given_to_dexter = 20 := by
  sorry

#check janessa_baseball_cards

end NUMINAMATH_CALUDE_janessa_baseball_cards_l3536_353601


namespace NUMINAMATH_CALUDE_spade_calculation_l3536_353692

def spade (k : ℕ) (x y : ℝ) : ℝ := (x + y + k) * (x - y + k)

theorem spade_calculation : 
  let k : ℕ := 2
  spade k 5 (spade k 3 2) = -392 := by
sorry

end NUMINAMATH_CALUDE_spade_calculation_l3536_353692


namespace NUMINAMATH_CALUDE_average_increase_l3536_353652

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  avgRuns : ℚ

/-- Calculate the new average after scoring additional runs -/
def newAverage (player : CricketPlayer) (additionalRuns : ℕ) : ℚ :=
  (player.totalRuns + additionalRuns) / (player.innings + 1)

/-- The main theorem about the increase in average -/
theorem average_increase (player : CricketPlayer) (additionalRuns : ℕ) :
  player.innings = 10 ∧ 
  player.avgRuns = 35 ∧ 
  additionalRuns = 79 →
  newAverage player additionalRuns - player.avgRuns = 4 := by
sorry


end NUMINAMATH_CALUDE_average_increase_l3536_353652


namespace NUMINAMATH_CALUDE_joseph_baseball_cards_l3536_353694

theorem joseph_baseball_cards (X : ℚ) : 
  X - (3/8) * X - 2 = (1/2) * X → X = 16 := by
  sorry

end NUMINAMATH_CALUDE_joseph_baseball_cards_l3536_353694


namespace NUMINAMATH_CALUDE_car_speed_proof_l3536_353609

/-- The speed of a car in km/h -/
def car_speed : ℝ := 30

/-- The reference speed in km/h -/
def reference_speed : ℝ := 36

/-- The additional time taken in seconds -/
def additional_time : ℝ := 20

/-- The distance traveled in km -/
def distance : ℝ := 1

theorem car_speed_proof :
  car_speed = 30 ∧
  (distance / car_speed) * 3600 = (distance / reference_speed) * 3600 + additional_time :=
sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3536_353609


namespace NUMINAMATH_CALUDE_hot_dog_cost_l3536_353608

/-- Given that 6 hot dogs cost 300 cents in total, and each hot dog costs the same amount,
    prove that each hot dog costs 50 cents. -/
theorem hot_dog_cost (total_cost : ℕ) (num_hot_dogs : ℕ) (cost_per_hot_dog : ℕ) 
    (h1 : total_cost = 300)
    (h2 : num_hot_dogs = 6)
    (h3 : total_cost = num_hot_dogs * cost_per_hot_dog) : 
  cost_per_hot_dog = 50 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_cost_l3536_353608


namespace NUMINAMATH_CALUDE_simplify_expression_l3536_353695

theorem simplify_expression (y : ℝ) : 7 * y + 8 - 3 * y + 16 = 4 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3536_353695


namespace NUMINAMATH_CALUDE_sum_of_non_solutions_l3536_353689

/-- Given an equation with infinitely many solutions, prove the sum of non-solution x values -/
theorem sum_of_non_solutions (A B C : ℝ) : 
  (∀ x : ℝ, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x + B) * (A * x + 36) ≠ 3 * (x + C) * (x + 9) ↔ (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = -21 :=
sorry

end NUMINAMATH_CALUDE_sum_of_non_solutions_l3536_353689


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3536_353688

/-- The area of a square with a diagonal of 28 meters is 392 square meters. -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 28) : 
  (d ^ 2 / 2) = 392 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3536_353688


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3536_353625

theorem smallest_x_absolute_value_equation :
  ∀ x : ℝ, |x - 3| = 8 → x ≥ -5 ∧ |-5 - 3| = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l3536_353625


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l3536_353633

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_hours : ℝ)
  (overtime_rate_increase : ℝ) :
  regular_rate = 16 →
  regular_hours = 40 →
  overtime_hours = 8 →
  overtime_rate_increase = 0.75 →
  regular_rate * regular_hours +
  (regular_rate * (1 + overtime_rate_increase)) * overtime_hours = 864 :=
by sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l3536_353633


namespace NUMINAMATH_CALUDE_correct_regression_sequence_l3536_353691

/-- A step in the linear regression analysis process -/
inductive RegressionStep
  | predict : RegressionStep
  | collectData : RegressionStep
  | deriveEquation : RegressionStep
  | plotScatter : RegressionStep

/-- The correct sequence of steps in linear regression analysis -/
def correctSequence : List RegressionStep :=
  [RegressionStep.collectData, RegressionStep.plotScatter, 
   RegressionStep.deriveEquation, RegressionStep.predict]

/-- Theorem stating that the given sequence is the correct order of steps -/
theorem correct_regression_sequence :
  correctSequence = [RegressionStep.collectData, RegressionStep.plotScatter, 
                     RegressionStep.deriveEquation, RegressionStep.predict] := by
  sorry

end NUMINAMATH_CALUDE_correct_regression_sequence_l3536_353691


namespace NUMINAMATH_CALUDE_eddy_rate_is_correct_l3536_353665

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  summit_distance : ℝ  -- Distance from base camp to summit in feet
  hillary_rate : ℝ     -- Hillary's climbing rate in ft/hr
  hillary_stop : ℝ     -- Distance from summit where Hillary stops
  hillary_descent : ℝ  -- Hillary's descent rate in ft/hr
  start_time : ℝ       -- Start time in hours (0 represents 06:00)
  meet_time : ℝ        -- Time when Hillary and Eddy meet in hours

/-- Calculates Eddy's climbing rate given a climbing scenario -/
def eddy_rate (scenario : ClimbingScenario) : ℝ :=
  -- The actual calculation of Eddy's rate
  sorry

/-- Theorem stating that Eddy's climbing rate is 5000/6 ft/hr given the specific scenario -/
theorem eddy_rate_is_correct (scenario : ClimbingScenario) 
  (h1 : scenario.summit_distance = 5000)
  (h2 : scenario.hillary_rate = 800)
  (h3 : scenario.hillary_stop = 1000)
  (h4 : scenario.hillary_descent = 1000)
  (h5 : scenario.start_time = 0)
  (h6 : scenario.meet_time = 6) : 
  eddy_rate scenario = 5000 / 6 := by
  sorry

end NUMINAMATH_CALUDE_eddy_rate_is_correct_l3536_353665


namespace NUMINAMATH_CALUDE_investment_interest_l3536_353615

/-- Calculates the interest earned on an investment with compound interest -/
def interestEarned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Proves that the interest earned on a $5000 investment at 3% annual interest
    compounded annually for 10 years is $1720 (rounded to the nearest dollar) -/
theorem investment_interest : 
  Int.floor (interestEarned 5000 0.03 10) = 1720 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_l3536_353615


namespace NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_two_times_sqrt_three_minus_sqrt_two_l3536_353697

theorem sqrt_three_plus_sqrt_two_times_sqrt_three_minus_sqrt_two : 
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_two_times_sqrt_three_minus_sqrt_two_l3536_353697


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3536_353648

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  ∃ (k : ℤ), (15 * x + 3) * (15 * x + 9) * (5 * x + 10) = 90 * k ∧
  ∀ (m : ℤ), m > 90 → ¬(∀ (y : ℤ), Even y →
    ∃ (l : ℤ), (15 * y + 3) * (15 * y + 9) * (5 * y + 10) = m * l) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3536_353648


namespace NUMINAMATH_CALUDE_greg_and_sarah_apples_l3536_353685

/-- Represents the number of apples each person has -/
structure AppleDistribution where
  greg : ℕ
  sarah : ℕ
  susan : ℕ
  mark : ℕ
  mom : ℕ

/-- Checks if the apple distribution satisfies the given conditions -/
def is_valid_distribution (d : AppleDistribution) : Prop :=
  d.greg = d.sarah ∧
  d.susan = 2 * d.greg ∧
  d.mark = d.susan - 5 ∧
  d.mom = 49

/-- Theorem stating that Greg and Sarah have 18 apples in total -/
theorem greg_and_sarah_apples (d : AppleDistribution) 
  (h : is_valid_distribution d) : d.greg + d.sarah = 18 := by
  sorry

end NUMINAMATH_CALUDE_greg_and_sarah_apples_l3536_353685


namespace NUMINAMATH_CALUDE_product_of_roots_l3536_353610

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  (∃ r₁ r₂ r₃ : ℝ, x^3 - 15*x^2 + 75*x - 50 = (x - r₁) * (x - r₂) * (x - r₃) ∧ r₁ * r₂ * r₃ = 50) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3536_353610


namespace NUMINAMATH_CALUDE_coefficient_x_8_in_expansion_l3536_353670

/-- The coefficient of x^8 in the expansion of (x^3 + 1/(2√x))^5 is 5/2 -/
theorem coefficient_x_8_in_expansion : 
  let expansion := (fun x => (x^3 + 1/(2*Real.sqrt x))^5)
  ∃ c : ℝ, c = 5/2 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ x > 0, 
      |x - 1| < δ → |expansion x / x^8 - c| < ε :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_8_in_expansion_l3536_353670


namespace NUMINAMATH_CALUDE_total_distance_walked_l3536_353622

/-- The distance Spencer walked from his house to the library -/
def distance_house_to_library : ℝ := 0.3

/-- The distance Spencer walked from the library to the post office -/
def distance_library_to_post_office : ℝ := 0.1

/-- The distance Spencer walked from the post office back home -/
def distance_post_office_to_house : ℝ := 0.4

/-- The theorem stating that the total distance Spencer walked is 0.8 miles -/
theorem total_distance_walked :
  distance_house_to_library + distance_library_to_post_office + distance_post_office_to_house = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l3536_353622


namespace NUMINAMATH_CALUDE_area_of_region_is_10_625_l3536_353603

/-- The lower boundary function of the region -/
def lower_boundary (x : ℝ) : ℝ := |x - 4|

/-- The upper boundary function of the region -/
def upper_boundary (x : ℝ) : ℝ := 5 - |x - 2|

/-- The region in the xy-plane -/
def region : Set (ℝ × ℝ) :=
  {p | lower_boundary p.1 ≤ p.2 ∧ p.2 ≤ upper_boundary p.1}

/-- The area of the region -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_region_is_10_625 : area_of_region = 10.625 := by sorry

end NUMINAMATH_CALUDE_area_of_region_is_10_625_l3536_353603


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3536_353620

/-- Given two vectors a and b in ℝ³, where a = (-2, 1, 5) and b = (6, m, -15),
    if a and b are parallel, then m = -3. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 3 → ℝ := ![(-2 : ℝ), 1, 5]
  let b : Fin 3 → ℝ := ![6, m, -15]
  (∃ (t : ℝ), b = fun i => t * a i) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3536_353620


namespace NUMINAMATH_CALUDE_a_composition_zero_l3536_353646

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem a_composition_zero : a (a (a 0)) = 343 := by sorry

end NUMINAMATH_CALUDE_a_composition_zero_l3536_353646


namespace NUMINAMATH_CALUDE_like_terms_exponent_relation_l3536_353641

/-- Given that -32a^(2m)b and b^(3-n)a^4 are like terms, prove that m^n = n^m -/
theorem like_terms_exponent_relation (a b m n : ℕ) : 
  (2 * m = 4 ∧ 3 - n = 1) → m^n = n^m := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_relation_l3536_353641


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l3536_353611

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The y-coordinate of a point on a line given its x-coordinate -/
def Line.y_at (l : Line) (x : ℝ) : ℝ :=
  l.slope * x + l.y_intercept

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line_through_point 
  (l : Line) (x₀ y₀ : ℝ) : 
  parallel l { slope := -3, y_intercept := 6 } →
  l.y_at x₀ = y₀ →
  x₀ = 3 →
  y₀ = -2 →
  l.y_intercept = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_through_point_l3536_353611


namespace NUMINAMATH_CALUDE_inverse_cube_squared_l3536_353635

theorem inverse_cube_squared : (3⁻¹)^2 = (1 : ℚ) / 9 := by sorry

end NUMINAMATH_CALUDE_inverse_cube_squared_l3536_353635


namespace NUMINAMATH_CALUDE_phd_basics_time_l3536_353699

/-- Represents the time John spent on his PhD journey -/
structure PhDTime where
  total : ℝ
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- The conditions of John's PhD journey -/
def phd_conditions (t : PhDTime) : Prop :=
  t.total = 7 ∧
  t.acclimation = 1 ∧
  t.research = t.basics + 0.75 * t.basics ∧
  t.dissertation = 0.5 * t.acclimation ∧
  t.total = t.acclimation + t.basics + t.research + t.dissertation

/-- Theorem stating that given the PhD conditions, the time spent learning basics is 2 years -/
theorem phd_basics_time (t : PhDTime) (h : phd_conditions t) : t.basics = 2 := by
  sorry

end NUMINAMATH_CALUDE_phd_basics_time_l3536_353699


namespace NUMINAMATH_CALUDE_project_completion_time_l3536_353616

/-- The number of days it takes for two workers to complete a job -/
structure WorkerPair :=
  (worker1 : ℕ)
  (worker2 : ℕ)
  (days : ℕ)

/-- The rate at which a worker completes the job per day -/
def workerRate (days : ℕ) : ℚ :=
  1 / days

theorem project_completion_time 
  (ab : WorkerPair) 
  (bc : WorkerPair) 
  (c_alone : ℕ) 
  (a_days : ℕ) 
  (b_days : ℕ) :
  ab.days = 10 →
  bc.days = 18 →
  c_alone = 45 →
  a_days = 5 →
  b_days = 10 →
  ∃ (c_days : ℕ), c_days = 15 ∧ 
    (workerRate ab.days * a_days + 
     workerRate ab.days * b_days + 
     workerRate c_alone * c_days = 1) :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l3536_353616


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l3536_353677

theorem mean_of_combined_sets :
  ∀ (set1 set2 : List ℝ),
    set1.length = 7 →
    set2.length = 8 →
    (set1.sum / set1.length : ℝ) = 15 →
    (set2.sum / set2.length : ℝ) = 20 →
    ((set1 ++ set2).sum / (set1 ++ set2).length : ℝ) = 17.67 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l3536_353677


namespace NUMINAMATH_CALUDE_train_length_l3536_353681

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 16 → speed * time * (1000 / 3600) = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3536_353681


namespace NUMINAMATH_CALUDE_simplify_cube_root_l3536_353634

theorem simplify_cube_root (a b : ℝ) (h : a < 0) : 
  Real.sqrt (a^3 * b) = -a * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l3536_353634


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3536_353637

/-- Given a curve y = x³ + ax + b and a line y = kx + 1 tangent to the curve at point (l, 3),
    prove that k = 2. -/
theorem tangent_line_slope (a b l : ℝ) : 
  (∃ k : ℝ, (3 = l^3 + a*l + b) ∧ (3 = k*l + 1) ∧ 
   (∀ x : ℝ, k*x + 1 ≤ x^3 + a*x + b) ∧
   (∃ x : ℝ, x ≠ l ∧ k*x + 1 < x^3 + a*x + b)) →
  (∃ k : ℝ, k = 2 ∧ (3 = k*l + 1)) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3536_353637


namespace NUMINAMATH_CALUDE_convergence_implies_cluster_sets_l3536_353623

open Set
open Filter
open Topology

/-- A sequence converges to a limit -/
def SequenceConvergesTo (x : ℕ → ℝ) (a : ℝ) :=
  Tendsto x atTop (𝓝 a)

/-- An interval is a cluster set for a sequence if it contains infinitely many terms of the sequence -/
def IsClusterSet (x : ℕ → ℝ) (s : Set ℝ) :=
  ∀ n : ℕ, ∃ m ≥ n, x m ∈ s

theorem convergence_implies_cluster_sets (x : ℕ → ℝ) (a : ℝ) :
  SequenceConvergesTo x a →
  (∀ ε > 0, IsClusterSet x (Ioo (a - ε) (a + ε))) ∧
  (∀ s : Set ℝ, IsOpen s → a ∉ s → ¬IsClusterSet x s) :=
sorry

end NUMINAMATH_CALUDE_convergence_implies_cluster_sets_l3536_353623


namespace NUMINAMATH_CALUDE_initial_machines_l3536_353639

/-- The number of machines working initially -/
def N : ℕ := sorry

/-- The number of units produced by N machines in 5 days -/
def x : ℝ := sorry

/-- Machines work at a constant rate -/
axiom constant_rate : ∀ (m : ℕ) (u t : ℝ), m ≠ 0 → t ≠ 0 → u / (m * t) = x / (N * 5)

theorem initial_machines :
  N * (x / 5) = 12 * (x / 30) → N = 2 :=
sorry

end NUMINAMATH_CALUDE_initial_machines_l3536_353639


namespace NUMINAMATH_CALUDE_two_part_trip_first_part_length_l3536_353602

/-- Proves that in a two-part trip with given conditions, the first part is 30 km long -/
theorem two_part_trip_first_part_length 
  (total_distance : ℝ)
  (speed_first_part : ℝ)
  (speed_second_part : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : speed_first_part = 60)
  (h3 : speed_second_part = 30)
  (h4 : average_speed = 40) :
  ∃ (first_part_distance : ℝ),
    first_part_distance = 30 ∧
    first_part_distance / speed_first_part + (total_distance - first_part_distance) / speed_second_part = total_distance / average_speed :=
by sorry

end NUMINAMATH_CALUDE_two_part_trip_first_part_length_l3536_353602


namespace NUMINAMATH_CALUDE_lindas_savings_l3536_353649

theorem lindas_savings (savings : ℝ) : (1 / 4 : ℝ) * savings = 230 → savings = 920 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l3536_353649


namespace NUMINAMATH_CALUDE_divisibility_problem_l3536_353619

theorem divisibility_problem (x y : ℤ) 
  (hx : x ≠ -1) (hy : y ≠ -1) 
  (h : ∃ k : ℤ, (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) = k) : 
  ∃ m : ℤ, x^4 * y^44 - 1 = m * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3536_353619


namespace NUMINAMATH_CALUDE_vegetable_ghee_weight_l3536_353600

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 700

/-- The ratio of brand 'a' to brand 'b' in the mixture -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3280

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

theorem vegetable_ghee_weight : 
  (weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume) + 
  (weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume) = total_weight :=
by sorry

end NUMINAMATH_CALUDE_vegetable_ghee_weight_l3536_353600


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3536_353629

theorem quadratic_one_root (n : ℝ) : 
  (∀ x : ℝ, x^2 + 6*n*x + 2*n = 0 → (∀ y : ℝ, y^2 + 6*n*y + 2*n = 0 → y = x)) →
  n = 2/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3536_353629


namespace NUMINAMATH_CALUDE_coin_sum_theorem_l3536_353632

def coin_values : List Nat := [5, 10, 25, 50]

def is_valid_sum (n : Nat) : Prop :=
  ∃ (a b c d e : Nat), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ d ∈ coin_values ∧ e ∈ coin_values ∧
    a + b + c + d + e = n

theorem coin_sum_theorem :
  ¬(is_valid_sum 40) ∧ 
  (is_valid_sum 65) ∧ 
  (is_valid_sum 85) ∧ 
  (is_valid_sum 105) ∧ 
  (is_valid_sum 130) := by
  sorry

end NUMINAMATH_CALUDE_coin_sum_theorem_l3536_353632


namespace NUMINAMATH_CALUDE_symmetry_conditions_l3536_353658

/-- A function is symmetric about a point (a, b) if f(x) + f(2a - x) = 2b for all x in its domain -/
def SymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f x + f (2 * a - x) = 2 * b

theorem symmetry_conditions (m a : ℝ) :
  let f := fun x : ℝ => (x^2 + m*x + m) / x
  let g := fun x : ℝ => if x > 0 then x^2 + a*x + 1 else -x^2 + a*x + 1
  (SymmetricAbout f 0 1) ∧
  (∀ x ≠ 0, SymmetricAbout g 0 1) ∧
  (∀ x t, x < 0 → t > 0 → g x < f t) →
  (m = 1) ∧
  (∀ x < 0, g x = -x^2 + a*x + 1) ∧
  (-2 * Real.sqrt 2 < a) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_conditions_l3536_353658


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3536_353686

/-- Given two adjacent points of a square at (1,2) and (5,5), the area of the square is 25. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (5, 5)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_points_l3536_353686


namespace NUMINAMATH_CALUDE_satellite_upgraded_fraction_l3536_353668

/-- Represents a satellite with modular units and sensors. -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on a satellite. -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

theorem satellite_upgraded_fraction :
  ∀ s : Satellite,
    s.units = 24 →
    s.non_upgraded_per_unit * 6 = s.total_upgraded →
    upgraded_fraction s = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_satellite_upgraded_fraction_l3536_353668


namespace NUMINAMATH_CALUDE_equation_equivalence_l3536_353680

theorem equation_equivalence (a c x y : ℤ) (m n p : ℕ) : 
  (a^9*x*y - a^8*y - a^7*x = a^6*(c^3 - 1)) →
  ((a^m*x - a^n)*(a^p*y - a^3) = a^6*c^3) →
  m*n*p = 90 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3536_353680


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3536_353606

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (a^2 - a - 2) = (a^2 - 2*a) + Complex.I * (a^2 - a - 2)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3536_353606


namespace NUMINAMATH_CALUDE_building_area_scientific_notation_l3536_353675

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem building_area_scientific_notation :
  toScientificNotation 258000 = ScientificNotation.mk 2.58 5 sorry := by sorry

end NUMINAMATH_CALUDE_building_area_scientific_notation_l3536_353675


namespace NUMINAMATH_CALUDE_tangent_equality_l3536_353666

-- Define the types for circles and points
variable (Circle Point : Type)

-- Define the predicates and functions
variable (outside : Circle → Circle → Prop)
variable (touches : Circle → Circle → Point → Point → Prop)
variable (passes_through : Circle → Point → Point → Prop)
variable (intersects_at : Circle → Circle → Point → Prop)
variable (tangent_at : Circle → Point → Point → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem tangent_equality 
  (S₁ S₂ S₃ : Circle) 
  (A B C D K : Point) :
  outside S₁ S₂ →
  touches S₁ S₂ A B →
  passes_through S₃ A B →
  intersects_at S₃ S₁ C →
  intersects_at S₃ S₂ D →
  tangent_at S₁ C K →
  tangent_at S₂ D K →
  distance K C = distance K D :=
sorry

end NUMINAMATH_CALUDE_tangent_equality_l3536_353666


namespace NUMINAMATH_CALUDE_birth_year_problem_l3536_353674

theorem birth_year_problem : ∃! x : ℕ, x ∈ Finset.range 50 ∧ x^2 - x = 1892 := by
  sorry

end NUMINAMATH_CALUDE_birth_year_problem_l3536_353674


namespace NUMINAMATH_CALUDE_equivalent_inequalities_l3536_353687

theorem equivalent_inequalities :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ ((1 / x > 1) ∧ (Real.log x < 0)) :=
sorry

end NUMINAMATH_CALUDE_equivalent_inequalities_l3536_353687


namespace NUMINAMATH_CALUDE_no_solutions_prime_equation_l3536_353651

theorem no_solutions_prime_equation (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p^a - 1 ≠ 2^n * (p - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_prime_equation_l3536_353651


namespace NUMINAMATH_CALUDE_cycle_original_price_l3536_353669

/-- Given a cycle sold at a 20% loss for Rs. 1120, prove that the original price was Rs. 1400 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1120)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    original_price = 1400 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cycle_original_price_l3536_353669


namespace NUMINAMATH_CALUDE_x_plus_y_values_l3536_353618

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) :
  (x + y = -3) ∨ (x + y = -9) :=
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l3536_353618


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l3536_353659

/-- Proves that arctan(tan 75° - 3 tan 30°) is approximately 124.1°. -/
theorem arctan_tan_difference (ε : ℝ) (h : ε > 0) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 180 ∧ |θ - 124.1| < ε ∧ θ = Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (30 * π / 180)) * 180 / π :=
sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l3536_353659


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l3536_353684

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 675 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l3536_353684


namespace NUMINAMATH_CALUDE_root_in_interval_l3536_353645

noncomputable def f (x : ℝ) := Real.exp x - x - 2

theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3536_353645


namespace NUMINAMATH_CALUDE_readers_both_sf_and_lit_l3536_353690

/-- Represents the number of readers who read both science fiction and literary works. -/
def readers_both (total readers_sf readers_lit : ℕ) : ℕ :=
  readers_sf + readers_lit - total

/-- 
Given a group of 400 readers, where 250 read science fiction and 230 read literary works,
proves that 80 readers read both science fiction and literary works.
-/
theorem readers_both_sf_and_lit : 
  readers_both 400 250 230 = 80 := by
  sorry

end NUMINAMATH_CALUDE_readers_both_sf_and_lit_l3536_353690


namespace NUMINAMATH_CALUDE_six_people_non_adjacent_seating_l3536_353664

/-- The number of ways to seat n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat n people around a round table
    where two specific individuals are adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := (n - 1) * Nat.factorial (n - 2)

/-- The number of ways to seat 6 people around a round table
    where two specific individuals are not adjacent. -/
theorem six_people_non_adjacent_seating :
  roundTableArrangements 6 - adjacentArrangements 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_people_non_adjacent_seating_l3536_353664


namespace NUMINAMATH_CALUDE_max_distance_Z₁Z₂_l3536_353672

-- Define complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition_z₁ : Prop := Complex.abs z₁ ≤ 2
def condition_z₂ : Prop := z₂ = Complex.mk 3 (-4)

-- Define the vector from Z₁ to Z₂
def vector_Z₁Z₂ : ℂ := z₂ - z₁

-- Theorem statement
theorem max_distance_Z₁Z₂ (hz₁ : condition_z₁ z₁) (hz₂ : condition_z₂ z₂) :
  ∃ (max_dist : ℝ), max_dist = 7 ∧ ∀ (z₁' : ℂ), condition_z₁ z₁' → Complex.abs (vector_Z₁Z₂ z₁' z₂) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_Z₁Z₂_l3536_353672


namespace NUMINAMATH_CALUDE_ln_abs_properties_l3536_353671

-- Define the function f(x) = ln|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem ln_abs_properties :
  (∀ x ≠ 0, f (-x) = f x) ∧  -- f is even
  (∀ x y, 0 < x → x < y → f x < f y) :=  -- f is increasing on (0, +∞)
by sorry

end NUMINAMATH_CALUDE_ln_abs_properties_l3536_353671


namespace NUMINAMATH_CALUDE_remainder_theorem_l3536_353604

theorem remainder_theorem (s : ℤ) : 
  (s^15 - 2) % (s - 3) = 14348905 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3536_353604


namespace NUMINAMATH_CALUDE_part1_part2_l3536_353607

-- Define the sequences
def a : ℕ → ℝ := λ n => 2^n
def b : ℕ → ℝ := λ n => 3^n
def c : ℕ → ℝ := λ n => a n + b n

-- Part 1
theorem part1 (p : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, c (n + 2) - p * c (n + 1) = r * (c (n + 1) - p * c n)) →
  p = 2 ∨ p = 3 := by sorry

-- Part 2
theorem part2 {q1 q2 : ℝ} (hq : q1 ≠ q2) 
  (ha : ∀ n : ℕ, a (n + 1) = q1 * a n) 
  (hb : ∀ n : ℕ, b (n + 1) = q2 * b n) :
  ¬ (∃ r : ℝ, ∀ n : ℕ, c (n + 1) = r * c n) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3536_353607


namespace NUMINAMATH_CALUDE_average_weight_proof_l3536_353605

theorem average_weight_proof (rachel_weight jimmy_weight adam_weight : ℝ) : 
  rachel_weight = 75 ∧ 
  rachel_weight = jimmy_weight - 6 ∧ 
  rachel_weight = adam_weight + 15 → 
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 := by
sorry

end NUMINAMATH_CALUDE_average_weight_proof_l3536_353605


namespace NUMINAMATH_CALUDE_mitchs_weekly_earnings_l3536_353656

/-- Mitch's weekly earnings calculation --/
theorem mitchs_weekly_earnings : 
  let weekday_hours : ℕ := 5
  let weekend_hours : ℕ := 3
  let weekday_rate : ℕ := 3
  let weekend_rate : ℕ := 2 * weekday_rate
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2

  weekdays * weekday_hours * weekday_rate + 
  weekend_days * weekend_hours * weekend_rate = 111 := by
  sorry

end NUMINAMATH_CALUDE_mitchs_weekly_earnings_l3536_353656


namespace NUMINAMATH_CALUDE_triangle_properties_l3536_353662

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = 3 →
  C = π / 3 →  -- 60° in radians
  c = Real.sqrt 7 ∧
  (1 / 2 * a * b * Real.sin C) = (3 * Real.sqrt 3) / 2 ∧
  Real.sin (2 * A) = (4 * Real.sqrt 3) / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3536_353662


namespace NUMINAMATH_CALUDE_original_number_proof_l3536_353644

theorem original_number_proof (x : ℝ) (h : 1 + 1/x = 7/3) : x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3536_353644


namespace NUMINAMATH_CALUDE_product_base_8_units_digit_l3536_353624

def base_10_to_8_units_digit (n : ℕ) : ℕ :=
  n % 8

theorem product_base_8_units_digit :
  base_10_to_8_units_digit (348 * 27) = 4 := by
sorry

end NUMINAMATH_CALUDE_product_base_8_units_digit_l3536_353624


namespace NUMINAMATH_CALUDE_sandy_fish_purchase_l3536_353642

/-- Given that Sandy initially had 26 fish and now has 32 fish, 
    prove that she bought 6 fish. -/
theorem sandy_fish_purchase :
  ∀ (initial_fish current_fish purchased_fish : ℕ),
  initial_fish = 26 →
  current_fish = 32 →
  purchased_fish = current_fish - initial_fish →
  purchased_fish = 6 := by
sorry

end NUMINAMATH_CALUDE_sandy_fish_purchase_l3536_353642


namespace NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l3536_353657

theorem reservoir_fullness_after_storm 
  (original_content : ℝ) 
  (initial_percentage : ℝ) 
  (added_water : ℝ) 
  (h1 : original_content = 220)
  (h2 : initial_percentage = 55.00000000000001)
  (h3 : added_water = 120) : 
  (original_content + added_water) / (original_content / (initial_percentage / 100)) * 100 = 85 := by
sorry

end NUMINAMATH_CALUDE_reservoir_fullness_after_storm_l3536_353657


namespace NUMINAMATH_CALUDE_percentage_relation_l3536_353627

theorem percentage_relation (x a b : ℝ) (h1 : a = 0.07 * x) (h2 : b = 0.14 * x) :
  a = 0.5 * b := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l3536_353627


namespace NUMINAMATH_CALUDE_six_digit_numbers_count_l3536_353614

/-- The number of ways to choose 2 items from 4 items -/
def choose_4_2 : ℕ := 6

/-- The number of ways to choose 1 item from 2 items -/
def choose_2_1 : ℕ := 2

/-- The number of ways to arrange 3 items -/
def arrange_3_3 : ℕ := 6

/-- The number of ways to choose 2 positions from 4 positions -/
def insert_2_in_4 : ℕ := 6

/-- The total number of valid six-digit numbers -/
def total_numbers : ℕ := choose_4_2 * choose_2_1 * arrange_3_3 * insert_2_in_4

theorem six_digit_numbers_count : total_numbers = 432 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_count_l3536_353614


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3536_353640

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (n = 3) ∧ 
  (∀ m : ℕ, m < n → ¬(
    (∃ p : ℤ, m^2 = (p+2)^5 - p^5) ∧ 
    (∃ k : ℕ, 3*m + 100 = k^2) ∧ 
    Odd m
  )) ∧
  (∃ p : ℤ, n^2 = (p+2)^5 - p^5) ∧ 
  (∃ k : ℕ, 3*n + 100 = k^2) ∧ 
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3536_353640


namespace NUMINAMATH_CALUDE_max_value_a_inequality_l3536_353650

theorem max_value_a_inequality (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ π/2 → 
    (x₂ * Real.sin x₁ - x₁ * Real.sin x₂) / (x₁ - x₂) > a) →
  a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_inequality_l3536_353650


namespace NUMINAMATH_CALUDE_eric_ben_difference_l3536_353636

theorem eric_ben_difference (jack ben eric : ℕ) : 
  jack = 26 → 
  ben = jack - 9 → 
  eric + ben + jack = 50 → 
  ben - eric = 10 := by
sorry

end NUMINAMATH_CALUDE_eric_ben_difference_l3536_353636


namespace NUMINAMATH_CALUDE_triangle_with_sides_4_6_5_l3536_353654

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_with_sides_4_6_5 :
  can_form_triangle 4 6 5 ∧
  ¬can_form_triangle 4 6 2 ∧
  ¬can_form_triangle 4 6 10 ∧
  ¬can_form_triangle 4 6 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_sides_4_6_5_l3536_353654


namespace NUMINAMATH_CALUDE_number_relationship_l3536_353643

theorem number_relationship : 
  let a : ℝ := -0.3
  let b : ℝ := (0.3:ℝ)^2
  let c : ℝ := 2^(0.3:ℝ)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_number_relationship_l3536_353643


namespace NUMINAMATH_CALUDE_tissues_left_l3536_353653

/-- The number of tissues in one box -/
def tissues_per_box : ℕ := 160

/-- The number of boxes bought -/
def boxes_bought : ℕ := 3

/-- The number of tissues used -/
def tissues_used : ℕ := 210

/-- Theorem: Given the conditions, prove that the number of tissues left is 270 -/
theorem tissues_left : 
  tissues_per_box * boxes_bought - tissues_used = 270 := by
  sorry

end NUMINAMATH_CALUDE_tissues_left_l3536_353653


namespace NUMINAMATH_CALUDE_mother_twice_lisa_age_l3536_353626

/-- Represents a person with their birth year -/
structure Person where
  birth_year : ℕ

/-- The year of Lisa's 6th birthday -/
def lisa_sixth_birthday : ℕ := 2010

/-- Lisa's birth year -/
def lisa : Person :=
  ⟨lisa_sixth_birthday - 6⟩

/-- Lisa's mother's birth year -/
def lisa_mother : Person :=
  ⟨lisa_sixth_birthday - 30⟩

/-- The year when Lisa's mother's age will be twice Lisa's age -/
def target_year : ℕ := 2028

/-- Theorem stating that the target year is correct -/
theorem mother_twice_lisa_age :
  (target_year - lisa_mother.birth_year) = 2 * (target_year - lisa.birth_year) :=
by sorry

end NUMINAMATH_CALUDE_mother_twice_lisa_age_l3536_353626


namespace NUMINAMATH_CALUDE_nh4cl_molecular_weight_l3536_353613

/-- The molecular weight of NH4Cl in grams per mole -/
def molecular_weight_NH4Cl : ℝ := 53

/-- The number of moles given in the problem -/
def moles : ℝ := 8

/-- The total weight of the given moles of NH4Cl in grams -/
def total_weight : ℝ := 424

/-- Theorem: The molecular weight of NH4Cl is 53 grams/mole -/
theorem nh4cl_molecular_weight :
  molecular_weight_NH4Cl = total_weight / moles :=
by sorry

end NUMINAMATH_CALUDE_nh4cl_molecular_weight_l3536_353613


namespace NUMINAMATH_CALUDE_integer_relation_l3536_353663

theorem integer_relation (x y z w : ℕ+) 
  (h1 : x^3 = y^2) 
  (h2 : z^4 = w^3) 
  (h3 : z - x = 22) : 
  ∃ (p q : ℕ+), 
    x = p^2 ∧ 
    y = p^3 ∧ 
    z = q^3 ∧ 
    w = q^4 ∧ 
    q^3 - p^2 = 22 ∧ 
    w - y = (q^(4/3))^3 - p^3 :=
sorry

end NUMINAMATH_CALUDE_integer_relation_l3536_353663
