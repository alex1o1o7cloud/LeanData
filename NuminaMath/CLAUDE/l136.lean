import Mathlib

namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l136_13639

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l136_13639


namespace NUMINAMATH_CALUDE_recycling_money_calculation_l136_13697

/-- Calculates the total money received from recycling cans and newspapers. -/
def recycling_money (can_rate : ℚ) (newspaper_rate : ℚ) (cans : ℕ) (newspapers : ℕ) : ℚ :=
  (can_rate * (cans / 12 : ℚ)) + (newspaper_rate * (newspapers / 5 : ℚ))

/-- Theorem: Given the recycling rates and the family's collection, the total money received is $12. -/
theorem recycling_money_calculation :
  recycling_money (1/2) (3/2) 144 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_recycling_money_calculation_l136_13697


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l136_13655

/-- The count of three-digit numbers. -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two identical non-adjacent digits. -/
def excluded_numbers : ℕ := 81

/-- The count of valid three-digit numbers according to the problem conditions. -/
def valid_numbers : ℕ := total_three_digit_numbers - excluded_numbers

/-- Theorem stating that the count of valid three-digit numbers is 819. -/
theorem valid_three_digit_numbers_count :
  valid_numbers = 819 := by sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l136_13655


namespace NUMINAMATH_CALUDE_function_passes_through_point_l136_13636

theorem function_passes_through_point (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a^(x - 1)
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l136_13636


namespace NUMINAMATH_CALUDE_log_equation_solution_l136_13642

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) = 3 →
  x = Real.exp ((9 + Real.sqrt 77) * Real.log b / 2) ∨
  x = Real.exp ((9 - Real.sqrt 77) * Real.log b / 2) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l136_13642


namespace NUMINAMATH_CALUDE_min_sum_squares_l136_13652

theorem min_sum_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  (∃ m : ℝ, m = a^2 + b^2 ∧ ∀ c d : ℝ, (∃ y : ℝ, y^4 + c*y^3 + d*y^2 + c*y + 1 = 0) → m ≤ c^2 + d^2) ∧ 
  (∃ n : ℝ, n = 4/5 ∧ n = a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l136_13652


namespace NUMINAMATH_CALUDE_interest_difference_approx_l136_13643

/-- Calculates the balance of an account with compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Calculates the balance of an account with simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The difference between compound and simple interest balances -/
def interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (time : ℕ) : ℝ :=
  compound_interest principal compound_rate time - simple_interest principal simple_rate time

theorem interest_difference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |interest_difference 15000 0.06 0.08 20 - 9107| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l136_13643


namespace NUMINAMATH_CALUDE_product_equals_10000_l136_13651

theorem product_equals_10000 : ∃ x : ℕ, 469160 * x = 4691130840 ∧ x = 10000 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_10000_l136_13651


namespace NUMINAMATH_CALUDE_job_completion_time_l136_13658

/-- Given a job that can be completed by a man in 10 days and his son in 20/3 days,
    prove that they can complete the job together in 4 days. -/
theorem job_completion_time (man_time son_time combined_time : ℚ) : 
  man_time = 10 → son_time = 20 / 3 → 
  combined_time = 1 / (1 / man_time + 1 / son_time) → 
  combined_time = 4 := by sorry

end NUMINAMATH_CALUDE_job_completion_time_l136_13658


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l136_13614

/-- Given a cubic equation with two known roots, find the value of k and the third root -/
theorem cubic_equation_roots (k : ℝ) : 
  (∀ x : ℝ, x^3 + 5*x^2 + k*x - 12 = 0 ↔ x = 3 ∨ x = -2 ∨ x = -6) →
  k = -12 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l136_13614


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l136_13663

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r ^ (n - 1)

theorem ninth_term_of_sequence :
  let a₁ : ℚ := 5
  let r : ℚ := 3/2
  geometric_sequence a₁ r 9 = 32805/256 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l136_13663


namespace NUMINAMATH_CALUDE_factors_of_20160_l136_13606

theorem factors_of_20160 : (Finset.filter (· ∣ 20160) (Finset.range 20161)).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_20160_l136_13606


namespace NUMINAMATH_CALUDE_pony_discount_rate_l136_13685

/-- Represents the discount rates for Fox and Pony jeans -/
structure DiscountRates where
  fox : ℝ
  pony : ℝ

/-- The problem setup -/
def jeans_problem (d : DiscountRates) : Prop :=
  -- Regular prices
  let fox_price : ℝ := 15
  let pony_price : ℝ := 18
  -- Total savings condition
  3 * fox_price * (d.fox / 100) + 2 * pony_price * (d.pony / 100) = 9 ∧
  -- Sum of discount rates condition
  d.fox + d.pony = 25

/-- The theorem to prove -/
theorem pony_discount_rate : 
  ∃ (d : DiscountRates), jeans_problem d ∧ d.pony = 25 := by
  sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l136_13685


namespace NUMINAMATH_CALUDE_j_type_sequence_properties_l136_13664

/-- Definition of a J_k type sequence -/
def is_J_k_type (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, (a (n + k))^2 = a n * a (n + 2*k)

theorem j_type_sequence_properties 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) :
  (is_J_k_type a 2 ∧ a 2 = 8 ∧ a 8 = 1 → 
    ∀ n : ℕ, a (2*n) = 2^(4-n)) ∧
  (is_J_k_type a 3 ∧ is_J_k_type a 4 → 
    ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n) :=
sorry

end NUMINAMATH_CALUDE_j_type_sequence_properties_l136_13664


namespace NUMINAMATH_CALUDE_distance_condition_l136_13626

theorem distance_condition (a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → 
      (x - a)^2 + (1/x - a)^2 ≤ (y - a)^2 + (1/y - a)^2) ∧
    (x - a)^2 + (1/x - a)^2 = 8) ↔ 
  (a = -1 ∨ a = Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_distance_condition_l136_13626


namespace NUMINAMATH_CALUDE_x_lt_neg_one_necessary_not_sufficient_l136_13676

theorem x_lt_neg_one_necessary_not_sufficient :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_neg_one_necessary_not_sufficient_l136_13676


namespace NUMINAMATH_CALUDE_x_values_l136_13615

def A (x : ℝ) : Set ℝ := {x, x^2}

theorem x_values (x : ℝ) (h : 1 ∈ A x) : x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l136_13615


namespace NUMINAMATH_CALUDE_garden_width_calculation_l136_13611

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  perimeter : ℝ

/-- Theorem: Given a rectangular garden with length 1.2 meters and perimeter 8.4 meters, the width is 3.0 meters -/
theorem garden_width_calculation (garden : RectangularGarden) 
  (h1 : garden.length = 1.2)
  (h2 : garden.perimeter = 8.4) : 
  garden.width = 3.0 := by
  sorry

#check garden_width_calculation

end NUMINAMATH_CALUDE_garden_width_calculation_l136_13611


namespace NUMINAMATH_CALUDE_log_half_inequality_condition_l136_13607

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

theorem log_half_inequality_condition (x : ℝ) (hx : x ∈ Set.Ioo 0 (1/2)) :
  (∀ a : ℝ, a < 0 → log_half x > x + a) ∧
  ∃ a : ℝ, a ≥ 0 ∧ log_half x > x + a :=
by
  sorry

#check log_half_inequality_condition

end NUMINAMATH_CALUDE_log_half_inequality_condition_l136_13607


namespace NUMINAMATH_CALUDE_sine_shift_l136_13653

theorem sine_shift (x : ℝ) : 3 * Real.sin (2 * x + π / 5) = 3 * Real.sin (2 * (x + π / 10)) := by
  sorry

end NUMINAMATH_CALUDE_sine_shift_l136_13653


namespace NUMINAMATH_CALUDE_max_value_of_product_l136_13666

theorem max_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 2) :
  a^2 * b^3 * c^4 ≤ 143327232 / 386989855 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_l136_13666


namespace NUMINAMATH_CALUDE_average_of_r_s_t_l136_13674

theorem average_of_r_s_t (r s t : ℝ) (h : (5 / 4) * (r + s + t - 2) = 15) : 
  (r + s + t) / 3 = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_r_s_t_l136_13674


namespace NUMINAMATH_CALUDE_expression_value_l136_13603

theorem expression_value : ((2^2 - 3*2 - 10) / (2 - 5)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l136_13603


namespace NUMINAMATH_CALUDE_james_browser_tabs_l136_13661

/-- Calculates the number of tabs per window given the total number of browsers,
    windows per browser, and total tabs. -/
def tabsPerWindow (browsers : ℕ) (windowsPerBrowser : ℕ) (totalTabs : ℕ) : ℕ :=
  totalTabs / (browsers * windowsPerBrowser)

/-- Proves that with 2 browsers, 3 windows per browser, and 60 total tabs,
    the number of tabs per window is 10. -/
theorem james_browser_tabs :
  tabsPerWindow 2 3 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_browser_tabs_l136_13661


namespace NUMINAMATH_CALUDE_chocolate_theorem_l136_13631

def chocolate_problem (total_boxes : ℕ) (pieces_per_box : ℕ) (remaining_pieces : ℕ) : ℕ :=
  (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box

theorem chocolate_theorem : chocolate_problem 12 6 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l136_13631


namespace NUMINAMATH_CALUDE_third_circle_radius_l136_13673

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle -/
theorem third_circle_radius (center_P center_Q center_R : ℝ × ℝ) 
  (radius_P radius_Q radius_R : ℝ) : 
  radius_P = 2 →
  radius_Q = 6 →
  (center_P.1 - center_Q.1)^2 + (center_P.2 - center_Q.2)^2 = (radius_P + radius_Q)^2 →
  (center_P.1 - center_R.1)^2 + (center_P.2 - center_R.2)^2 = (radius_P + radius_R)^2 →
  (center_Q.1 - center_R.1)^2 + (center_Q.2 - center_R.2)^2 = (radius_Q + radius_R)^2 →
  (∃ (t : ℝ), center_R.2 = t * center_P.2 + (1 - t) * center_Q.2 ∧ 
              center_R.1 = t * center_P.1 + (1 - t) * center_Q.1 ∧ 
              0 ≤ t ∧ t ≤ 1) →
  radius_R = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l136_13673


namespace NUMINAMATH_CALUDE_functional_equation_solution_l136_13602

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (f x - y) + 4 * f x * y

/-- The main theorem stating that any function satisfying the equation
    must be of the form f(x) = x² + C for some constant C -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ C : ℝ, ∀ x : ℝ, f x = x^2 + C := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l136_13602


namespace NUMINAMATH_CALUDE_theta_range_l136_13678

theorem theta_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) 
  (h2 : Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) :
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_theta_range_l136_13678


namespace NUMINAMATH_CALUDE_probability_masters_degree_expected_value_bachelors_or_higher_male_education_greater_than_female_l136_13632

/-- Represents the education levels in the census data -/
inductive EducationLevel
  | NoSchooling
  | PrimarySchool
  | JuniorHighSchool
  | HighSchool
  | CollegeAssociate
  | CollegeBachelor
  | MastersDegree
  | DoctoralDegree

/-- Represents the gender in the census data -/
inductive Gender
  | Male
  | Female

/-- Census data for City Z -/
def censusData : Gender → EducationLevel → Float
  | Gender.Male, EducationLevel.NoSchooling => 0.00
  | Gender.Male, EducationLevel.PrimarySchool => 0.03
  | Gender.Male, EducationLevel.JuniorHighSchool => 0.14
  | Gender.Male, EducationLevel.HighSchool => 0.11
  | Gender.Male, EducationLevel.CollegeAssociate => 0.07
  | Gender.Male, EducationLevel.CollegeBachelor => 0.11
  | Gender.Male, EducationLevel.MastersDegree => 0.03
  | Gender.Male, EducationLevel.DoctoralDegree => 0.01
  | Gender.Female, EducationLevel.NoSchooling => 0.01
  | Gender.Female, EducationLevel.PrimarySchool => 0.04
  | Gender.Female, EducationLevel.JuniorHighSchool => 0.11
  | Gender.Female, EducationLevel.HighSchool => 0.11
  | Gender.Female, EducationLevel.CollegeAssociate => 0.08
  | Gender.Female, EducationLevel.CollegeBachelor => 0.12
  | Gender.Female, EducationLevel.MastersDegree => 0.03
  | Gender.Female, EducationLevel.DoctoralDegree => 0.00

/-- Proportion of residents aged 15 and above in City Z -/
def proportionAged15AndAbove : Float := 0.85

/-- Theorem 1: Probability of selecting a person aged 15 and above with a Master's degree -/
theorem probability_masters_degree : 
  proportionAged15AndAbove * (censusData Gender.Male EducationLevel.MastersDegree + 
  censusData Gender.Female EducationLevel.MastersDegree) = 0.051 := by sorry

/-- Theorem 2: Expected value of X (number of people with Bachelor's degree or higher among two randomly selected residents aged 15 and above) -/
theorem expected_value_bachelors_or_higher : 
  let p := censusData Gender.Male EducationLevel.CollegeBachelor + 
           censusData Gender.Female EducationLevel.CollegeBachelor +
           censusData Gender.Male EducationLevel.MastersDegree + 
           censusData Gender.Female EducationLevel.MastersDegree +
           censusData Gender.Male EducationLevel.DoctoralDegree + 
           censusData Gender.Female EducationLevel.DoctoralDegree
  2 * p * (1 - p) + 2 * p * p = 0.6 := by sorry

/-- Theorem 3: Relationship between average years of education for male and female residents -/
theorem male_education_greater_than_female :
  let male_avg := 0 * censusData Gender.Male EducationLevel.NoSchooling +
                  6 * censusData Gender.Male EducationLevel.PrimarySchool +
                  9 * censusData Gender.Male EducationLevel.JuniorHighSchool +
                  12 * censusData Gender.Male EducationLevel.HighSchool +
                  16 * (censusData Gender.Male EducationLevel.CollegeAssociate +
                        censusData Gender.Male EducationLevel.CollegeBachelor +
                        censusData Gender.Male EducationLevel.MastersDegree +
                        censusData Gender.Male EducationLevel.DoctoralDegree)
  let female_avg := 0 * censusData Gender.Female EducationLevel.NoSchooling +
                    6 * censusData Gender.Female EducationLevel.PrimarySchool +
                    9 * censusData Gender.Female EducationLevel.JuniorHighSchool +
                    12 * censusData Gender.Female EducationLevel.HighSchool +
                    16 * (censusData Gender.Female EducationLevel.CollegeAssociate +
                          censusData Gender.Female EducationLevel.CollegeBachelor +
                          censusData Gender.Female EducationLevel.MastersDegree +
                          censusData Gender.Female EducationLevel.DoctoralDegree)
  male_avg > female_avg := by sorry

end NUMINAMATH_CALUDE_probability_masters_degree_expected_value_bachelors_or_higher_male_education_greater_than_female_l136_13632


namespace NUMINAMATH_CALUDE_bag_weight_problem_l136_13623

theorem bag_weight_problem (w1 w2 w3 : ℝ) : 
  w1 / w2 = 4 / 5 ∧ w2 / w3 = 5 / 6 ∧ w1 + w3 = w2 + 45 → w1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_problem_l136_13623


namespace NUMINAMATH_CALUDE_joint_purchase_savings_l136_13679

/-- Represents the store's tile offer structure -/
structure TileOffer where
  regularPrice : ℕ  -- Regular price per tile
  buyQuantity : ℕ   -- Number of tiles to buy
  freeQuantity : ℕ  -- Number of free tiles given

/-- Calculates the cost of purchasing a given number of tiles under the offer -/
def calculateCost (offer : TileOffer) (tilesNeeded : ℕ) : ℕ :=
  let fullSets := tilesNeeded / (offer.buyQuantity + offer.freeQuantity)
  let remainingTiles := tilesNeeded % (offer.buyQuantity + offer.freeQuantity)
  fullSets * offer.buyQuantity * offer.regularPrice + remainingTiles * offer.regularPrice

/-- Theorem stating the savings when Dave and Doug purchase together -/
theorem joint_purchase_savings (offer : TileOffer) (daveTiles dougTiles : ℕ) :
  offer.regularPrice = 150 ∧ 
  offer.buyQuantity = 9 ∧ 
  offer.freeQuantity = 2 ∧
  daveTiles = 11 ∧
  dougTiles = 13 →
  calculateCost offer daveTiles + calculateCost offer dougTiles - 
  calculateCost offer (daveTiles + dougTiles) = 600 := by
  sorry

end NUMINAMATH_CALUDE_joint_purchase_savings_l136_13679


namespace NUMINAMATH_CALUDE_log_product_equals_ten_l136_13640

theorem log_product_equals_ten (n : ℕ) (h : n = 2) : 
  7.63 * (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_ten_l136_13640


namespace NUMINAMATH_CALUDE_commute_time_difference_l136_13695

theorem commute_time_difference (distance : ℝ) (speed_actual : ℝ) (speed_suggested : ℝ) :
  distance = 10 ∧ speed_actual = 30 ∧ speed_suggested = 25 →
  (distance / speed_suggested - distance / speed_actual) * 60 = 4 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_l136_13695


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l136_13670

/-- A function that returns true if n can be expressed as the sum of k consecutive positive integers starting from a -/
def is_sum_of_consecutive_integers (n k a : ℕ) : Prop :=
  n = k * a + k * (k - 1) / 2

/-- A function that checks if n satisfies all the conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ 2000 ∧ ∃ k a : ℕ, k ≥ 60 ∧ is_sum_of_consecutive_integers n k a

/-- The main theorem stating that there are exactly 6 numbers satisfying the conditions -/
theorem count_numbers_satisfying_conditions :
  ∃! (S : Finset ℕ), (∀ n ∈ S, satisfies_conditions n) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l136_13670


namespace NUMINAMATH_CALUDE_probability_prime_product_l136_13672

/-- A standard 6-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of prime numbers on a 6-sided die -/
def PrimesOnDie : Finset ℕ := {2, 3, 5}

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 9

/-- The total number of possible outcomes when rolling three dice -/
def TotalOutcomes : ℕ := 216

/-- The probability of rolling three 6-sided dice and getting a prime number as the product of their face values -/
theorem probability_prime_product (d : Finset ℕ) (p : Finset ℕ) (f : ℕ) (t : ℕ) 
  (h1 : d = Die) 
  (h2 : p = PrimesOnDie) 
  (h3 : f = FavorableOutcomes) 
  (h4 : t = TotalOutcomes) :
  (f : ℚ) / t = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_prime_product_l136_13672


namespace NUMINAMATH_CALUDE_total_cars_is_seventeen_l136_13617

/-- The number of cars Tommy has -/
def tommy_cars : ℕ := 3

/-- The number of cars Jessie has -/
def jessie_cars : ℕ := 3

/-- The number of additional cars Jessie's older brother has compared to Tommy and Jessie combined -/
def brother_additional_cars : ℕ := 5

/-- The total number of cars for all three of them -/
def total_cars : ℕ := tommy_cars + jessie_cars + (tommy_cars + jessie_cars + brother_additional_cars)

theorem total_cars_is_seventeen : total_cars = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_seventeen_l136_13617


namespace NUMINAMATH_CALUDE_area_under_curve_l136_13669

open Real MeasureTheory

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2

-- State the theorem
theorem area_under_curve : 
  ∫ x in (1)..(2), f x = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_under_curve_l136_13669


namespace NUMINAMATH_CALUDE_intersection_height_l136_13668

/-- The height of the intersection point of lines drawn between two poles -/
theorem intersection_height (h1 h2 d : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (d_pos : 0 < d) :
  let x := (h1 * h2 * d) / (h1 * d + h2 * d)
  h1 = 20 → h2 = 80 → d = 100 → x = 16 := by
  sorry


end NUMINAMATH_CALUDE_intersection_height_l136_13668


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l136_13680

theorem sqrt_product_sqrt : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l136_13680


namespace NUMINAMATH_CALUDE_point_on_line_l136_13634

/-- The point A lies on the line y = 2x - 4 -/
theorem point_on_line : ∃ (A : ℝ × ℝ), A.1 = 3 ∧ A.2 = 2 ∧ A.2 = 2 * A.1 - 4 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l136_13634


namespace NUMINAMATH_CALUDE_point_on_graph_l136_13691

/-- The linear function f(x) = 3x + 1 -/
def f (x : ℝ) : ℝ := 3 * x + 1

/-- The point (2, 7) -/
def point : ℝ × ℝ := (2, 7)

/-- Theorem: The point (2, 7) lies on the graph of f(x) = 3x + 1 -/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_graph_l136_13691


namespace NUMINAMATH_CALUDE_part_one_part_two_l136_13682

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}

-- Define set C with parameter a
def C (a : ℝ) : Set ℝ := {x | 2*x + a > 0}

-- Theorem for part (1)
theorem part_one :
  B = {x : ℝ | x ≥ 2} ∧
  (A ∩ B)ᶜ = {x : ℝ | x < 2 ∨ x ≥ 3} :=
sorry

-- Theorem for part (2)
theorem part_two :
  {a : ℝ | ∀ x, x ∈ B → x ∈ C a} = {a : ℝ | a > -4} :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l136_13682


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l136_13671

theorem binomial_probability_problem (p : ℝ) (X : ℕ → ℝ) :
  (∀ k, X k = Nat.choose 4 k * p^k * (1 - p)^(4 - k)) →
  X 2 = 8/27 →
  p = 1/3 ∨ p = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l136_13671


namespace NUMINAMATH_CALUDE_salem_poem_words_per_line_l136_13608

/-- A poem with a given structure and word count -/
structure Poem where
  stanzas : ℕ
  lines_per_stanza : ℕ
  total_words : ℕ

/-- Calculate the number of words per line in a poem -/
def words_per_line (p : Poem) : ℕ :=
  p.total_words / (p.stanzas * p.lines_per_stanza)

/-- Theorem: Given a poem with 20 stanzas, 10 lines per stanza, and 1600 total words,
    the number of words per line is 8 -/
theorem salem_poem_words_per_line :
  let p : Poem := { stanzas := 20, lines_per_stanza := 10, total_words := 1600 }
  words_per_line p = 8 := by
  sorry

#check salem_poem_words_per_line

end NUMINAMATH_CALUDE_salem_poem_words_per_line_l136_13608


namespace NUMINAMATH_CALUDE_fractional_linear_transformation_cross_ratio_l136_13612

theorem fractional_linear_transformation_cross_ratio 
  (a b c d : ℝ) (h : a * d - b * c ≠ 0)
  (x₁ x₂ x₃ x₄ : ℝ) :
  let y : ℝ → ℝ := λ x => (a * x + b) / (c * x + d)
  let y₁ := y x₁
  let y₂ := y x₂
  let y₃ := y x₃
  let y₄ := y x₄
  (y₃ - y₁) / (y₃ - y₂) / ((y₄ - y₁) / (y₄ - y₂)) = 
  (x₃ - x₁) / (x₃ - x₂) / ((x₄ - x₁) / (x₄ - x₂)) :=
by sorry

end NUMINAMATH_CALUDE_fractional_linear_transformation_cross_ratio_l136_13612


namespace NUMINAMATH_CALUDE_factorization_problem_value_problem_l136_13692

-- Problem 1
theorem factorization_problem (a : ℝ) : 
  a^3 - 3*a^2 - 4*a + 12 = (a - 3) * (a - 2) * (a + 2) := by sorry

-- Problem 2
theorem value_problem (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) : 
  m^2 - n^2 + 2*m - 2*n = 7 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_value_problem_l136_13692


namespace NUMINAMATH_CALUDE_zoo_total_animals_l136_13662

def zoo_animals (num_penguins : ℕ) (num_polar_bears : ℕ) : ℕ :=
  num_penguins + num_polar_bears

theorem zoo_total_animals :
  let num_penguins : ℕ := 21
  let num_polar_bears : ℕ := 2 * num_penguins
  zoo_animals num_penguins num_polar_bears = 63 := by
  sorry

end NUMINAMATH_CALUDE_zoo_total_animals_l136_13662


namespace NUMINAMATH_CALUDE_big_sixteen_game_count_l136_13616

/-- Represents a basketball league with the given structure -/
structure BasketballLeague where
  totalTeams : Nat
  divisionsCount : Nat
  intraGameCount : Nat
  interGameCount : Nat

/-- Calculates the total number of scheduled games in the league -/
def totalGames (league : BasketballLeague) : Nat :=
  let teamsPerDivision := league.totalTeams / league.divisionsCount
  let intraGamesPerDivision := teamsPerDivision * (teamsPerDivision - 1) / 2 * league.intraGameCount
  let totalIntraGames := intraGamesPerDivision * league.divisionsCount
  let totalInterGames := league.totalTeams * teamsPerDivision * league.interGameCount / 2
  totalIntraGames + totalInterGames

/-- Theorem stating that the Big Sixteen Basketball League schedules 296 games -/
theorem big_sixteen_game_count :
  let bigSixteen : BasketballLeague := {
    totalTeams := 16
    divisionsCount := 2
    intraGameCount := 3
    interGameCount := 2
  }
  totalGames bigSixteen = 296 := by
  sorry

end NUMINAMATH_CALUDE_big_sixteen_game_count_l136_13616


namespace NUMINAMATH_CALUDE_number_wall_solution_l136_13699

/-- Represents a simplified Number Wall structure -/
structure NumberWall where
  bottom_left : ℕ
  bottom_second : ℕ
  bottom_third : ℕ
  bottom_right : ℕ
  second_row_right : ℕ
  top : ℕ

/-- Checks if a NumberWall is valid according to the sum rule -/
def is_valid_wall (w : NumberWall) : Prop :=
  w.bottom_left + w.bottom_second = w.bottom_left + w.bottom_second ∧
  w.bottom_second + w.bottom_third = w.bottom_second + w.bottom_third ∧
  w.bottom_third + w.bottom_right = w.second_row_right ∧
  (w.bottom_left + w.bottom_second) + (w.bottom_second + w.bottom_third) = w.top

theorem number_wall_solution (w : NumberWall) 
  (h1 : w.bottom_second = 5)
  (h2 : w.bottom_third = 9)
  (h3 : w.bottom_right = 6)
  (h4 : w.second_row_right = 14)
  (h5 : w.top = 57)
  (h6 : is_valid_wall w) :
  w.bottom_left = 8 := by
  sorry

#check number_wall_solution

end NUMINAMATH_CALUDE_number_wall_solution_l136_13699


namespace NUMINAMATH_CALUDE_solution_of_system_l136_13698

theorem solution_of_system (x y : ℚ) 
  (eq1 : 3 * y - 4 * x = 8)
  (eq2 : 2 * y + x = -1) : 
  x = -19/11 ∧ y = 4/11 := by
sorry

end NUMINAMATH_CALUDE_solution_of_system_l136_13698


namespace NUMINAMATH_CALUDE_prob_at_least_eight_sixes_l136_13681

/-- The probability of rolling a six on a fair die -/
def prob_six : ℚ := 1/6

/-- The number of rolls -/
def num_rolls : ℕ := 10

/-- The minimum number of sixes required -/
def min_sixes : ℕ := 8

/-- Calculates the probability of rolling exactly k sixes in n rolls -/
def prob_exact_sixes (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (prob_six ^ k) * ((1 - prob_six) ^ (n - k))

/-- The probability of rolling at least 8 sixes in 10 rolls of a fair die -/
theorem prob_at_least_eight_sixes : 
  (prob_exact_sixes num_rolls min_sixes + 
   prob_exact_sixes num_rolls (min_sixes + 1) + 
   prob_exact_sixes num_rolls (min_sixes + 2)) = 3/15504 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_eight_sixes_l136_13681


namespace NUMINAMATH_CALUDE_alternative_bases_l136_13613

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem alternative_bases
  (a b c : V)
  (h : LinearIndependent ℝ ![a, b, c])
  (hspan : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, a + c, a] ∧
  Submodule.span ℝ {a + b, a + c, a} = ⊤ ∧
  LinearIndependent ℝ ![a - b + c, a - b, a + c] ∧
  Submodule.span ℝ {a - b + c, a - b, a + c} = ⊤ := by
sorry

end NUMINAMATH_CALUDE_alternative_bases_l136_13613


namespace NUMINAMATH_CALUDE_addition_commutative_example_l136_13622

theorem addition_commutative_example : 73 + 93 + 27 = 73 + 27 + 93 := by
  sorry

end NUMINAMATH_CALUDE_addition_commutative_example_l136_13622


namespace NUMINAMATH_CALUDE_mike_office_visits_l136_13644

/-- The number of pull-ups Mike does each time he enters his office -/
def pull_ups_per_visit : ℕ := 2

/-- The number of pull-ups Mike does in a week -/
def pull_ups_per_week : ℕ := 70

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of times Mike goes into his office each day -/
def office_visits_per_day : ℕ := 5

theorem mike_office_visits :
  office_visits_per_day * days_per_week * pull_ups_per_visit = pull_ups_per_week :=
by sorry

end NUMINAMATH_CALUDE_mike_office_visits_l136_13644


namespace NUMINAMATH_CALUDE_arccos_gt_twice_arcsin_l136_13641

theorem arccos_gt_twice_arcsin (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 1 → (Real.arccos x > 2 * Real.arcsin x ↔ -1 < x ∧ x ≤ (1 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_arccos_gt_twice_arcsin_l136_13641


namespace NUMINAMATH_CALUDE_term1_and_term2_are_like_terms_l136_13619

-- Define a structure for terms
structure Term where
  coefficient : ℚ
  x_power : ℕ
  y_power : ℕ

-- Define what it means for two terms to be like terms
def are_like_terms (t1 t2 : Term) : Prop :=
  t1.x_power = t2.x_power ∧ t1.y_power = t2.y_power

-- Define the two terms we're comparing
def term1 : Term := { coefficient := 4, x_power := 2, y_power := 1 }
def term2 : Term := { coefficient := -1, x_power := 2, y_power := 1 }

-- Theorem stating that term1 and term2 are like terms
theorem term1_and_term2_are_like_terms : are_like_terms term1 term2 := by
  sorry


end NUMINAMATH_CALUDE_term1_and_term2_are_like_terms_l136_13619


namespace NUMINAMATH_CALUDE_positive_number_problem_l136_13627

theorem positive_number_problem (n : ℝ) : n > 0 ∧ 5 * (n^2 + n) = 780 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_problem_l136_13627


namespace NUMINAMATH_CALUDE_exists_x0_exp_greater_than_square_sum_of_roots_equals_five_l136_13689

-- Proposition ③
theorem exists_x0_exp_greater_than_square :
  ∃ x₀ : ℝ, ∀ x > x₀, (2 : ℝ) ^ x > x ^ 2 := by sorry

-- Proposition ⑤
theorem sum_of_roots_equals_five :
  let f₁ := fun x : ℝ => x + Real.log 2 * Real.log x / Real.log 10 - 5
  let f₂ := fun x : ℝ => x + (10 : ℝ) ^ x - 5
  ∀ x₁ x₂ : ℝ, f₁ x₁ = 0 → f₂ x₂ = 0 → x₁ + x₂ = 5 := by sorry

end NUMINAMATH_CALUDE_exists_x0_exp_greater_than_square_sum_of_roots_equals_five_l136_13689


namespace NUMINAMATH_CALUDE_copper_zinc_mass_ranges_l136_13659

/-- Represents the properties of a copper-zinc mixture -/
structure CopperZincMixture where
  total_mass : ℝ
  total_volume : ℝ
  copper_density_min : ℝ
  copper_density_max : ℝ
  zinc_density_min : ℝ
  zinc_density_max : ℝ

/-- Theorem stating the mass ranges of copper and zinc in the mixture -/
theorem copper_zinc_mass_ranges (mixture : CopperZincMixture)
  (h_total_mass : mixture.total_mass = 400)
  (h_total_volume : mixture.total_volume = 50)
  (h_copper_density : mixture.copper_density_min = 8.8 ∧ mixture.copper_density_max = 9)
  (h_zinc_density : mixture.zinc_density_min = 7.1 ∧ mixture.zinc_density_max = 7.2) :
  ∃ (copper_mass zinc_mass : ℝ),
    200 ≤ copper_mass ∧ copper_mass ≤ 233 ∧
    167 ≤ zinc_mass ∧ zinc_mass ≤ 200 ∧
    copper_mass + zinc_mass = mixture.total_mass ∧
    copper_mass / mixture.copper_density_max + zinc_mass / mixture.zinc_density_min = mixture.total_volume :=
by sorry

end NUMINAMATH_CALUDE_copper_zinc_mass_ranges_l136_13659


namespace NUMINAMATH_CALUDE_questions_to_write_l136_13665

theorem questions_to_write 
  (total_mc : ℕ) (total_ps : ℕ) (total_tf : ℕ)
  (frac_mc : ℚ) (frac_ps : ℚ) (frac_tf : ℚ)
  (h1 : total_mc = 50)
  (h2 : total_ps = 30)
  (h3 : total_tf = 40)
  (h4 : frac_mc = 5/8)
  (h5 : frac_ps = 7/12)
  (h6 : frac_tf = 2/5) :
  ↑total_mc - ⌊frac_mc * total_mc⌋ + 
  ↑total_ps - ⌊frac_ps * total_ps⌋ + 
  ↑total_tf - ⌊frac_tf * total_tf⌋ = 56 := by
sorry

end NUMINAMATH_CALUDE_questions_to_write_l136_13665


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l136_13620

def line (x y : ℝ) : Prop := y = 2 * x + 1

theorem distance_to_x_axis (k : ℝ) (h : line (-2) k) : 
  |k| = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l136_13620


namespace NUMINAMATH_CALUDE_record_collection_problem_l136_13625

theorem record_collection_problem (shared_records : ℕ) (emily_total : ℕ) (mark_unique : ℕ) : 
  shared_records = 15 → emily_total = 25 → mark_unique = 10 →
  emily_total - shared_records + mark_unique = 20 := by
sorry

end NUMINAMATH_CALUDE_record_collection_problem_l136_13625


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l136_13667

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

theorem words_with_vowels_count :
  (alphabet.card ^ word_length) - (consonants.card ^ word_length) = 6752 := by
  sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l136_13667


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_triangle_shape_l136_13696

/-- Factorization of 2a^2 - 8a + 8 --/
theorem factorization_1 (a : ℝ) : 2*a^2 - 8*a + 8 = 2*(a-2)^2 := by sorry

/-- Factorization of x^2 - y^2 + 3x - 3y --/
theorem factorization_2 (x y : ℝ) : x^2 - y^2 + 3*x - 3*y = (x-y)*(x+y+3) := by sorry

/-- Shape of triangle ABC given a^2 - ab - ac + bc = 0 --/
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (eq : a^2 - a*b - a*c + b*c = 0) :
  (a = b ∨ a = c ∨ b = c) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_triangle_shape_l136_13696


namespace NUMINAMATH_CALUDE_secretary_project_time_l136_13675

theorem secretary_project_time (t1 t2 t3 : ℕ) : 
  t1 + t2 + t3 = 120 →
  t2 = 2 * t1 →
  t3 = 5 * t1 →
  t3 = 75 := by
sorry

end NUMINAMATH_CALUDE_secretary_project_time_l136_13675


namespace NUMINAMATH_CALUDE_log_system_solutions_l136_13693

noncomputable def solve_log_system (x y : ℝ) : Prop :=
  x > y ∧ y > 0 ∧
  Real.log (x - y) + Real.log 2 = (1 / 2) * (Real.log x - Real.log y) ∧
  Real.log (x + y) - Real.log 3 = (1 / 2) * (Real.log y - Real.log x)

theorem log_system_solutions :
  (∃ (x y : ℝ), solve_log_system x y ∧ 
    ((x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) ∨ 
     (x = 3 * Real.sqrt 3 / 4 ∧ y = Real.sqrt 3 / 4))) ∧
  (∀ (x y : ℝ), solve_log_system x y → 
    ((x = Real.sqrt 2 ∧ y = 1 / Real.sqrt 2) ∨ 
     (x = 3 * Real.sqrt 3 / 4 ∧ y = Real.sqrt 3 / 4))) :=
by sorry

end NUMINAMATH_CALUDE_log_system_solutions_l136_13693


namespace NUMINAMATH_CALUDE_weight_difference_after_one_year_l136_13677

/-- Calculates the final weight of the labrador puppy after one year -/
def labrador_final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 1.1
  let weight2 := weight1 * 1.2
  let weight3 := weight2 * 1.25
  weight3 + 5

/-- Calculates the final weight of the dachshund puppy after one year -/
def dachshund_final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 1.05
  let weight2 := weight1 * 1.15
  let weight3 := weight2 - 1
  let weight4 := weight3 * 1.2
  weight4 + 3

/-- The difference in weight between the labrador and dachshund puppies after one year -/
theorem weight_difference_after_one_year :
  labrador_final_weight 40 - dachshund_final_weight 12 = 51.812 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_after_one_year_l136_13677


namespace NUMINAMATH_CALUDE_simple_interest_problem_l136_13601

/-- Given a sum put at simple interest for 3 years, if increasing the interest
    rate by 1% results in Rs. 69 more interest, then the sum is Rs. 2300. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 69) → P = 2300 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l136_13601


namespace NUMINAMATH_CALUDE_remainder_problem_l136_13660

theorem remainder_problem (k : ℕ) (h1 : k > 0) (h2 : 90 % (k^2) = 10) : 150 % k = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l136_13660


namespace NUMINAMATH_CALUDE_work_completion_time_l136_13635

/-- Given a man who can do a piece of work in 5 days and his son who can do the same work in 20 days,
    they can complete the work together in 4 days. -/
theorem work_completion_time (man_time son_time combined_time : ℝ) 
    (h1 : man_time = 5)
    (h2 : son_time = 20)
    (h3 : combined_time = 1 / (1 / man_time + 1 / son_time)) :
    combined_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l136_13635


namespace NUMINAMATH_CALUDE_parents_age_when_mark_born_l136_13645

/-- Given the ages of Mark and John, and their parents' current age relative to John's,
    prove the age of the parents when Mark was born. -/
theorem parents_age_when_mark_born
  (mark_age : ℕ)
  (john_age_diff : ℕ)
  (parents_age_factor : ℕ)
  (h1 : mark_age = 18)
  (h2 : john_age_diff = 10)
  (h3 : parents_age_factor = 5) :
  mark_age - (parents_age_factor * (mark_age - john_age_diff)) = 22 :=
by sorry

end NUMINAMATH_CALUDE_parents_age_when_mark_born_l136_13645


namespace NUMINAMATH_CALUDE_tiger_enclosure_optimizations_l136_13687

/-- Represents a rectangular tiger enclosure -/
structure TigerEnclosure where
  length : ℝ
  width : ℝ

/-- Calculates the area of a tiger enclosure -/
def area (e : TigerEnclosure) : ℝ := e.length * e.width

/-- Calculates the wire mesh length needed for a tiger enclosure -/
def wireMeshLength (e : TigerEnclosure) : ℝ := e.length + 2 * e.width

/-- The total available wire mesh length -/
def totalWireMesh : ℝ := 36

/-- The fixed area for part 2 of the problem -/
def fixedArea : ℝ := 32

theorem tiger_enclosure_optimizations :
  (∃ (e : TigerEnclosure),
    wireMeshLength e = totalWireMesh ∧
    area e = 162 ∧
    e.length = 18 ∧
    e.width = 9 ∧
    (∀ (e' : TigerEnclosure), wireMeshLength e' ≤ totalWireMesh → area e' ≤ area e)) ∧
  (∃ (e : TigerEnclosure),
    area e = fixedArea ∧
    wireMeshLength e = 16 ∧
    e.length = 8 ∧
    e.width = 4 ∧
    (∀ (e' : TigerEnclosure), area e' = fixedArea → wireMeshLength e' ≥ wireMeshLength e)) :=
by sorry

end NUMINAMATH_CALUDE_tiger_enclosure_optimizations_l136_13687


namespace NUMINAMATH_CALUDE_evaluate_expression_l136_13690

theorem evaluate_expression (x y z : ℤ) (hx : x = -1) (hy : y = 4) (hz : z = 2) :
  z * (2 * y - 3 * x) = 22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l136_13690


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l136_13633

open Complex

theorem modulus_of_complex_number : ∃ z : ℂ, z = (2 - I)^2 / I ∧ Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l136_13633


namespace NUMINAMATH_CALUDE_chime_2500_date_l136_13610

/-- Represents a date with year, month, and day. -/
structure Date :=
  (year : ℕ)
  (month : ℕ)
  (day : ℕ)

/-- Represents a time with hour and minute. -/
structure Time :=
  (hour : ℕ)
  (minute : ℕ)

/-- Represents the chiming pattern of the clock. -/
def chime_pattern (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute == 30 then 1
  else if hour ≤ 12 then hour
  else hour - 12

/-- Calculates the total chimes from a given start date and time to a target chime count. -/
def chimes_to_date (start_date : Date) (start_time : Time) (target_chimes : ℕ) : Date :=
  sorry

/-- The theorem to be proved. -/
theorem chime_2500_date :
  let start_date := Date.mk 2003 3 15
  let start_time := Time.mk 14 30
  let target_chimes := 2500
  let result_date := chimes_to_date start_date start_time target_chimes
  result_date = Date.mk 2003 4 8 :=
sorry

end NUMINAMATH_CALUDE_chime_2500_date_l136_13610


namespace NUMINAMATH_CALUDE_eliana_steps_theorem_l136_13618

/-- The number of steps Eliana walked on the first day -/
def first_day_steps : ℕ := 200 + 300

/-- The number of steps Eliana walked on the second day -/
def second_day_steps : ℕ := 2 * first_day_steps

/-- The additional steps Eliana walked on the third day -/
def third_day_additional_steps : ℕ := 100

/-- The total number of steps Eliana walked during the three days -/
def total_steps : ℕ := first_day_steps + second_day_steps + third_day_additional_steps

theorem eliana_steps_theorem : total_steps = 1600 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_theorem_l136_13618


namespace NUMINAMATH_CALUDE_johns_cloth_cost_l136_13605

/-- The total cost of cloth purchased by John -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem stating the total cost of John's cloth purchase -/
theorem johns_cloth_cost :
  let length : ℝ := 9.25
  let price_per_metre : ℝ := 46
  total_cost length price_per_metre = 425.50 := by
  sorry

end NUMINAMATH_CALUDE_johns_cloth_cost_l136_13605


namespace NUMINAMATH_CALUDE_new_light_wattage_l136_13647

theorem new_light_wattage (original_wattage : ℝ) (increase_percentage : ℝ) : 
  original_wattage = 110 → 
  increase_percentage = 30 → 
  original_wattage * (1 + increase_percentage / 100) = 143 := by
sorry

end NUMINAMATH_CALUDE_new_light_wattage_l136_13647


namespace NUMINAMATH_CALUDE_eraser_cost_l136_13628

theorem eraser_cost (total_cartons : ℕ) (total_cost : ℕ) (pencil_cost : ℕ) (pencil_cartons : ℕ) :
  total_cartons = 100 →
  total_cost = 360 →
  pencil_cost = 6 →
  pencil_cartons = 20 →
  (total_cost - pencil_cost * pencil_cartons) / (total_cartons - pencil_cartons) = 3 := by
sorry

end NUMINAMATH_CALUDE_eraser_cost_l136_13628


namespace NUMINAMATH_CALUDE_lines_are_parallel_l136_13650

-- Define the lines
def line1 (a : ℝ) (θ : ℝ) : Prop := θ = a
def line2 (p a θ : ℝ) : Prop := p * Real.sin (θ - a) = 1

-- Theorem statement
theorem lines_are_parallel (a p : ℝ) : 
  ∀ θ, ¬(line1 a θ ∧ line2 p a θ) :=
sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l136_13650


namespace NUMINAMATH_CALUDE_vector_combination_vectors_parallel_l136_13649

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- The theorem states that a = (5/9)b + (8/9)c -/
theorem vector_combination : a = (5/9 • b) + (8/9 • c) := by sorry

/-- Helper function to check if two vectors are parallel -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), v = t • w ∨ w = t • v

/-- The theorem states that (a + kc) is parallel to (2b - a) when k = -16/13 -/
theorem vectors_parallel : are_parallel (a + (-16/13 • c)) (2 • b - a) := by sorry

end NUMINAMATH_CALUDE_vector_combination_vectors_parallel_l136_13649


namespace NUMINAMATH_CALUDE_edward_games_boxes_l136_13684

def number_of_boxes (initial_games : ℕ) (sold_games : ℕ) (games_per_box : ℕ) : ℕ :=
  (initial_games - sold_games) / games_per_box

theorem edward_games_boxes :
  number_of_boxes 35 19 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_games_boxes_l136_13684


namespace NUMINAMATH_CALUDE_non_sibling_probability_l136_13646

/-- Represents a person in the room -/
structure Person where
  siblings : Nat

/-- Represents the room with people -/
def Room : Type := List Person

/-- The number of people in the room -/
def room_size : Nat := 7

/-- The condition that 4 people have exactly 1 sibling -/
def one_sibling_count (room : Room) : Prop :=
  (room.filter (fun p => p.siblings = 1)).length = 4

/-- The condition that 3 people have exactly 2 siblings -/
def two_siblings_count (room : Room) : Prop :=
  (room.filter (fun p => p.siblings = 2)).length = 3

/-- The probability of selecting two non-siblings -/
def prob_non_siblings (room : Room) : ℚ :=
  16 / 21

/-- The main theorem -/
theorem non_sibling_probability (room : Room) :
  room.length = room_size →
  one_sibling_count room →
  two_siblings_count room →
  prob_non_siblings room = 16 / 21 := by
  sorry


end NUMINAMATH_CALUDE_non_sibling_probability_l136_13646


namespace NUMINAMATH_CALUDE_xyz_inequality_l136_13683

theorem xyz_inequality : ∃ c : ℝ, ∀ x y z : ℝ, -|x*y*z| > c * (|x| + |y| + |z|) := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l136_13683


namespace NUMINAMATH_CALUDE_adjusted_ratio_equals_three_halves_l136_13604

theorem adjusted_ratio_equals_three_halves :
  (2^2003 * 3^2005) / 6^2004 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_adjusted_ratio_equals_three_halves_l136_13604


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l136_13657

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - m = 0 ∧ x₂^2 - 2*x₂ - m = 0) → m ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l136_13657


namespace NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l136_13621

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be monochromatic
def isMonochromatic (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- The main theorem
theorem monochromatic_unit_area_triangle_exists (coloring : Coloring) :
  ∃ t : Triangle, triangleArea t = 1 ∧ isMonochromatic t coloring := by sorry

end NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l136_13621


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l136_13609

-- Define the prices and quantities
def pencil_price : ℚ := 0.5
def folder_price : ℚ := 0.9
def notebook_price : ℚ := 1.2
def stapler_price : ℚ := 2.5

def pencil_quantity : ℕ := 24
def folder_quantity : ℕ := 20
def notebook_quantity : ℕ := 15
def stapler_quantity : ℕ := 10

-- Define discount rates
def pencil_discount_rate : ℚ := 0.1
def folder_discount_rate : ℚ := 0.15

-- Define discount conditions
def pencil_discount_threshold : ℕ := 15
def folder_discount_threshold : ℕ := 10

-- Define notebook offer
def notebook_offer : ℕ := 3  -- buy 2 get 1 free

-- Define the total cost function
def total_cost : ℚ :=
  let pencil_cost := pencil_price * pencil_quantity * (1 - pencil_discount_rate)
  let folder_cost := folder_price * folder_quantity * (1 - folder_discount_rate)
  let notebook_cost := notebook_price * (notebook_quantity - notebook_quantity / notebook_offer)
  let stapler_cost := stapler_price * stapler_quantity
  pencil_cost + folder_cost + notebook_cost + stapler_cost

-- Theorem to prove
theorem total_cost_is_correct : total_cost = 63.1 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l136_13609


namespace NUMINAMATH_CALUDE_ab_value_l136_13648

theorem ab_value (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l136_13648


namespace NUMINAMATH_CALUDE_vanessa_score_in_game_l136_13694

/-- Calculates Vanessa's score in a basketball game -/
def vanessaScore (totalScore : ℕ) (otherPlayersCount : ℕ) (otherPlayersAverage : ℕ) : ℕ :=
  totalScore - (otherPlayersCount * otherPlayersAverage)

/-- Theorem stating Vanessa's score given the game conditions -/
theorem vanessa_score_in_game : 
  vanessaScore 60 7 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_score_in_game_l136_13694


namespace NUMINAMATH_CALUDE_exists_term_with_four_zero_digits_l136_13637

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem exists_term_with_four_zero_digits : 
  ∃ n : ℕ, n < 100000001 ∧ last_four_digits (fibonacci n) = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_term_with_four_zero_digits_l136_13637


namespace NUMINAMATH_CALUDE_ellipse_properties_max_radius_l136_13688

/-- The ellipse C with foci F₁(-c, 0) and F₂(c, 0), and upper vertex M satisfying F₁M ⋅ F₂M = 0 -/
structure Ellipse (a b c : ℝ) :=
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (foci_condition : c^2 = a^2 - b^2)
  (vertex_condition : -c^2 + b^2 = 0)

/-- The point N(0, 2) is the center of a circle intersecting the ellipse C -/
def N : ℝ × ℝ := (0, 2)

/-- The theorem stating properties of the ellipse C -/
theorem ellipse_properties (a b c : ℝ) (C : Ellipse a b c) :
  -- The eccentricity of C is √2/2
  (c / a = Real.sqrt 2 / 2) ∧
  -- The equation of C is x²/18 + y²/9 = 1
  (∀ x y : ℝ, x^2 / 18 + y^2 / 9 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  -- The range of k for symmetric points A and B on C w.r.t. y = kx - 1
  (∀ k : ℝ, (k < -1/2 ∨ k > 1/2) ↔
    ∃ A B : ℝ × ℝ,
      (A.1^2 / 18 + A.2^2 / 9 = 1) ∧
      (B.1^2 / 18 + B.2^2 / 9 = 1) ∧
      (A.2 = k * A.1 - 1) ∧
      (B.2 = k * B.1 - 1) ∧
      (A ≠ B)) :=
sorry

/-- The maximum radius of the circle centered at N intersecting C is √26 -/
theorem max_radius (a b c : ℝ) (C : Ellipse a b c) :
  ∀ P : ℝ × ℝ, P.1^2 / a^2 + P.2^2 / b^2 = 1 →
    (P.1 - N.1)^2 + (P.2 - N.2)^2 ≤ 26 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_max_radius_l136_13688


namespace NUMINAMATH_CALUDE_complex_equation_solution_l136_13629

theorem complex_equation_solution (z : ℂ) :
  (1 + Complex.I) * z = -2 * Complex.I →
  z = -1 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l136_13629


namespace NUMINAMATH_CALUDE_division_problem_l136_13624

theorem division_problem (n : ℕ) : 
  n / 18 = 11 ∧ n % 18 = 1 → n = 199 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l136_13624


namespace NUMINAMATH_CALUDE_derek_remaining_money_l136_13654

theorem derek_remaining_money (initial_amount : ℕ) : 
  initial_amount = 960 →
  let textbook_expense := initial_amount / 2
  let remaining_after_textbooks := initial_amount - textbook_expense
  let supply_expense := remaining_after_textbooks / 4
  let final_remaining := remaining_after_textbooks - supply_expense
  final_remaining = 360 := by
sorry

end NUMINAMATH_CALUDE_derek_remaining_money_l136_13654


namespace NUMINAMATH_CALUDE_davis_popsicle_sticks_l136_13656

def popsicle_sticks_left (initial_sticks : ℕ) (num_groups : ℕ) (sticks_per_group : ℕ) : ℕ :=
  initial_sticks - num_groups * sticks_per_group

theorem davis_popsicle_sticks :
  popsicle_sticks_left 170 10 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_davis_popsicle_sticks_l136_13656


namespace NUMINAMATH_CALUDE_bricks_to_paint_theorem_l136_13630

/-- Represents a stack of bricks -/
structure BrickStack :=
  (height : ℕ)
  (width : ℕ)
  (depth : ℕ)
  (total_bricks : ℕ)
  (sides_against_wall : ℕ)

/-- Calculates the number of bricks that need to be painted on their exposed surfaces -/
def bricks_to_paint (stack : BrickStack) : ℕ :=
  let front_face := stack.height * stack.width + stack.depth
  let top_face := stack.width * stack.depth
  front_face * stack.height + top_face * (4 - stack.sides_against_wall)

theorem bricks_to_paint_theorem (stack : BrickStack) :
  stack.height = 4 ∧ 
  stack.width = 3 ∧ 
  stack.depth = 15 ∧ 
  stack.total_bricks = 180 ∧ 
  stack.sides_against_wall = 2 →
  bricks_to_paint stack = 96 :=
by sorry

end NUMINAMATH_CALUDE_bricks_to_paint_theorem_l136_13630


namespace NUMINAMATH_CALUDE_cosine_of_angle_l136_13600

/-- Given two vectors a and b in ℝ², prove that the cosine of the angle between them is -63/65,
    when a + b = (2, -8) and a - b = (-8, 16). -/
theorem cosine_of_angle (a b : ℝ × ℝ) 
    (sum_eq : a + b = (2, -8)) 
    (diff_eq : a - b = (-8, 16)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -63/65 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_l136_13600


namespace NUMINAMATH_CALUDE_books_to_decorations_ratio_l136_13686

theorem books_to_decorations_ratio 
  (total_books : ℕ) 
  (books_per_shelf : ℕ) 
  (decorations_per_shelf : ℕ) 
  (initial_shelves : ℕ) 
  (h1 : total_books = 42)
  (h2 : books_per_shelf = 2)
  (h3 : decorations_per_shelf = 1)
  (h4 : initial_shelves = 3) :
  (total_books : ℚ) / ((total_books / (books_per_shelf * initial_shelves)) * decorations_per_shelf) = 6 / 1 := by
sorry

end NUMINAMATH_CALUDE_books_to_decorations_ratio_l136_13686


namespace NUMINAMATH_CALUDE_remainder_sum_mod_21_l136_13638

theorem remainder_sum_mod_21 (c d : ℤ) 
  (hc : c % 60 = 47) 
  (hd : d % 42 = 17) : 
  (c + d) % 21 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_21_l136_13638
