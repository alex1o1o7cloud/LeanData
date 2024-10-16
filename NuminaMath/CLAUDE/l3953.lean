import Mathlib

namespace NUMINAMATH_CALUDE_vector_equation_l3953_395388

theorem vector_equation (a b c : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → c = (-1, -2) → 
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_l3953_395388


namespace NUMINAMATH_CALUDE_negative_sum_of_squares_l3953_395344

theorem negative_sum_of_squares (a : ℝ) : -3 * a^2 - 5 * a^2 = -8 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_of_squares_l3953_395344


namespace NUMINAMATH_CALUDE_parabola_max_ratio_l3953_395392

theorem parabola_max_ratio (p : ℝ) (h : p > 0) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 / 4 ∧
  ∀ (x y : ℝ), y^2 = 2*p*x →
    Real.sqrt (x^2 + y^2) / Real.sqrt ((x - p/6)^2 + y^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_ratio_l3953_395392


namespace NUMINAMATH_CALUDE_stating_public_foundation_share_l3953_395384

/-- Represents the charity donation problem -/
structure CharityDonation where
  X : ℝ  -- Total amount raised in dollars
  Y : ℝ  -- Percentage donated to public foundation
  Z : ℕ+  -- Number of organizations in public foundation
  W : ℕ+  -- Number of local non-profit groups
  A : ℝ  -- Amount received by each local non-profit group in dollars
  h1 : X > 0  -- Total amount raised is positive
  h2 : 0 < Y ∧ Y < 100  -- Percentage is between 0 and 100
  h3 : W * A = X * (100 - Y) / 100  -- Equation for local non-profit groups

/-- 
Theorem stating that each organization in the public foundation 
receives YX / (100Z) dollars
-/
theorem public_foundation_share (c : CharityDonation) :
  (c.Y * c.X) / (100 * c.Z) = 
  (c.X * c.Y / 100) / c.Z :=
sorry

end NUMINAMATH_CALUDE_stating_public_foundation_share_l3953_395384


namespace NUMINAMATH_CALUDE_slow_train_speed_l3953_395373

-- Define the problem parameters
def total_distance : ℝ := 901
def fast_train_speed : ℝ := 58
def slow_train_departure_time : ℝ := 5.5  -- 5:30 AM in decimal hours
def fast_train_departure_time : ℝ := 9.5  -- 9:30 AM in decimal hours
def meeting_time : ℝ := 16.5  -- 4:30 PM in decimal hours

-- Define the theorem
theorem slow_train_speed :
  let slow_train_travel_time : ℝ := meeting_time - slow_train_departure_time
  let fast_train_travel_time : ℝ := meeting_time - fast_train_departure_time
  let fast_train_distance : ℝ := fast_train_speed * fast_train_travel_time
  let slow_train_distance : ℝ := total_distance - fast_train_distance
  slow_train_distance / slow_train_travel_time = 45 := by
  sorry


end NUMINAMATH_CALUDE_slow_train_speed_l3953_395373


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3953_395318

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 - y^2 = a^2

-- Define the semi-latus rectum of a parabola
def semi_latus_rectum (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

theorem hyperbola_real_axis_length 
  (a p x1 y1 x2 y2 : ℝ) 
  (h1 : hyperbola a x1 y1)
  (h2 : hyperbola a x2 y2)
  (h3 : semi_latus_rectum p x1)
  (h4 : semi_latus_rectum p x2)
  (h5 : distance x1 y1 x2 y2 = 4 * (3^(1/2))) :
  2 * a = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3953_395318


namespace NUMINAMATH_CALUDE_first_part_speed_l3953_395338

/-- Proves that given a total trip distance of 255 miles, with the second part being 3 hours at 55 mph,
    the speed S for the first 2 hours must be 45 mph. -/
theorem first_part_speed (total_distance : ℝ) (first_duration : ℝ) (second_duration : ℝ) (second_speed : ℝ) :
  total_distance = 255 →
  first_duration = 2 →
  second_duration = 3 →
  second_speed = 55 →
  ∃ S : ℝ, S = 45 ∧ total_distance = first_duration * S + second_duration * second_speed :=
by sorry

end NUMINAMATH_CALUDE_first_part_speed_l3953_395338


namespace NUMINAMATH_CALUDE_product_of_integers_l3953_395301

theorem product_of_integers (p q r : ℕ+) : 
  p + q + r = 30 → 
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 420 / (p * q * r) = 1 → 
  p * q * r = 1800 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l3953_395301


namespace NUMINAMATH_CALUDE_triangle_inequality_l3953_395326

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a / (b + c)) + Real.sqrt (b / (a + c)) + Real.sqrt (c / (a + b)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3953_395326


namespace NUMINAMATH_CALUDE_triangle_inequality_l3953_395353

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 1) : a^2 * c + b^2 * a + c^2 * b < 1/8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3953_395353


namespace NUMINAMATH_CALUDE_vegetarian_eaters_l3953_395354

theorem vegetarian_eaters (only_veg : ℕ) (only_non_veg : ℕ) (both : ℕ) 
  (h1 : only_veg = 15) 
  (h2 : only_non_veg = 8) 
  (h3 : both = 11) : 
  only_veg + both = 26 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_l3953_395354


namespace NUMINAMATH_CALUDE_land_profit_calculation_l3953_395391

/-- Represents the profit calculation for land distribution among sons -/
theorem land_profit_calculation (total_land : ℝ) (num_sons : ℕ) 
  (profit_per_unit : ℝ) (unit_area : ℝ) (hectare_to_sqm : ℝ) : 
  total_land = 3 ∧ 
  num_sons = 8 ∧ 
  profit_per_unit = 500 ∧ 
  unit_area = 750 ∧ 
  hectare_to_sqm = 10000 → 
  (total_land * hectare_to_sqm / num_sons / unit_area * profit_per_unit * 4 : ℝ) = 10000 := by
  sorry

#check land_profit_calculation

end NUMINAMATH_CALUDE_land_profit_calculation_l3953_395391


namespace NUMINAMATH_CALUDE_polynomial_divisibility_and_factor_l3953_395328

theorem polynomial_divisibility_and_factor :
  let p (x : ℝ) := 6 * x^3 - 18 * x^2 + 24 * x - 24
  let q (x : ℝ) := x - 2
  let r (x : ℝ) := 6 * x^2 + 4
  (∃ (s : ℝ → ℝ), p = q * s) ∧ (∃ (t : ℝ → ℝ), p = r * t) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_and_factor_l3953_395328


namespace NUMINAMATH_CALUDE_count_two_digit_numbers_unit_gte_tens_is_45_l3953_395386

/-- The count of two-digit numbers where the unit digit is not less than the tens digit -/
def count_two_digit_numbers_unit_gte_tens : ℕ := 45

/-- Proof that the count of two-digit numbers where the unit digit is not less than the tens digit is 45 -/
theorem count_two_digit_numbers_unit_gte_tens_is_45 :
  count_two_digit_numbers_unit_gte_tens = 45 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_numbers_unit_gte_tens_is_45_l3953_395386


namespace NUMINAMATH_CALUDE_max_prob_two_unqualified_expected_cost_min_compensation_fee_l3953_395348

-- Define the probability of a fruit being unqualified
variable (p : ℝ) (hp : 0 < p ∧ p < 1)

-- Define the number of fruits in a box and sample size
def box_size : ℕ := 80
def sample_size : ℕ := 10

-- Define the inspection cost per fruit
def inspection_cost : ℝ := 1.5

-- Define the compensation fee per unqualified fruit
variable (a : ℕ) (ha : a > 0)

-- Function to calculate the probability of exactly k unqualified fruits in a sample of n
def binomial_prob (n k : ℕ) : ℝ → ℝ :=
  λ p => (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

-- Statement 1: Probability that maximizes likelihood of 2 unqualified fruits in 10
theorem max_prob_two_unqualified :
  ∃ p₀, 0 < p₀ ∧ p₀ < 1 ∧
  ∀ p, 0 < p ∧ p < 1 → binomial_prob sample_size 2 p ≤ binomial_prob sample_size 2 p₀ ∧
  p₀ = 0.2 := sorry

-- Statement 2: Expected cost given p = 0.2
theorem expected_cost (p₀ : ℝ) (hp₀ : p₀ = 0.2) :
  (sample_size : ℝ) * inspection_cost + a * (box_size - sample_size : ℝ) * p₀ = 15 + 14 * a := sorry

-- Statement 3: Minimum compensation fee for full inspection
theorem min_compensation_fee :
  ∃ a_min : ℕ, a_min > 0 ∧
  ∀ a : ℕ, a ≥ a_min →
    (box_size : ℝ) * inspection_cost < (sample_size : ℝ) * inspection_cost + a * (box_size - sample_size : ℝ) * 0.2 ∧
  a_min = 8 := sorry

end NUMINAMATH_CALUDE_max_prob_two_unqualified_expected_cost_min_compensation_fee_l3953_395348


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l3953_395337

theorem unique_quadratic_root (a : ℝ) : 
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ (a = 1 ∨ a = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l3953_395337


namespace NUMINAMATH_CALUDE_quadratic_expansion_l3953_395387

theorem quadratic_expansion (m n : ℝ) :
  (∀ x : ℝ, (x + 4) * (x - 2) = x^2 + m*x + n) →
  m = 2 ∧ n = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expansion_l3953_395387


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l3953_395350

/-- Theorem: Theater Ticket Pricing
  Given:
  - Total tickets sold is 340
  - Total revenue is $3,320
  - Orchestra seat price is $12
  - Number of balcony seats sold is 40 more than orchestra seats
  Prove that the cost of a balcony seat is $8
-/
theorem theater_ticket_pricing 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (orchestra_price : ℕ) 
  (balcony_excess : ℕ) 
  (h1 : total_tickets = 340)
  (h2 : total_revenue = 3320)
  (h3 : orchestra_price = 12)
  (h4 : balcony_excess = 40) :
  let orchestra_seats := (total_tickets - balcony_excess) / 2
  let balcony_seats := orchestra_seats + balcony_excess
  let balcony_revenue := total_revenue - orchestra_price * orchestra_seats
  balcony_revenue / balcony_seats = 8 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l3953_395350


namespace NUMINAMATH_CALUDE_problem_1_l3953_395382

theorem problem_1 : 2^2 - 2023^0 + |3 - Real.pi| = Real.pi := by sorry

end NUMINAMATH_CALUDE_problem_1_l3953_395382


namespace NUMINAMATH_CALUDE_translation_problem_l3953_395334

/-- A translation in the complex plane. -/
def Translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

/-- The theorem statement -/
theorem translation_problem (T : ℂ → ℂ) (h : T (1 + 3*I) = 4 + 6*I) :
  T (2 - I) = 5 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l3953_395334


namespace NUMINAMATH_CALUDE_beta_value_l3953_395383

/-- Given α = 2023°, if β has the same terminal side as α and β ∈ (0, 2π), then β = 223π/180 -/
theorem beta_value (α β : Real) : 
  α = 2023 * (π / 180) →
  (∃ k : ℤ, β = α + k * 2 * π) →
  β ∈ Set.Ioo 0 (2 * π) →
  β = 223 * (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_beta_value_l3953_395383


namespace NUMINAMATH_CALUDE_new_average_age_l3953_395359

-- Define the initial conditions
def initial_people : ℕ := 8
def initial_average_age : ℚ := 28
def leaving_person_age : ℕ := 20
def entering_person_age : ℕ := 25

-- Define the theorem
theorem new_average_age :
  let initial_total_age : ℚ := initial_people * initial_average_age
  let after_leaving_age : ℚ := initial_total_age - leaving_person_age
  let final_total_age : ℚ := after_leaving_age + entering_person_age
  final_total_age / initial_people = 229 / 8 := by sorry

end NUMINAMATH_CALUDE_new_average_age_l3953_395359


namespace NUMINAMATH_CALUDE_raspberry_pies_count_l3953_395379

/-- The total number of pies -/
def total_pies : ℕ := 36

/-- The ratio of apple pies -/
def apple_ratio : ℕ := 1

/-- The ratio of blueberry pies -/
def blueberry_ratio : ℕ := 3

/-- The ratio of cherry pies -/
def cherry_ratio : ℕ := 2

/-- The ratio of raspberry pies -/
def raspberry_ratio : ℕ := 4

/-- The sum of all ratios -/
def total_ratio : ℕ := apple_ratio + blueberry_ratio + cherry_ratio + raspberry_ratio

/-- Theorem: The number of raspberry pies is 14.4 -/
theorem raspberry_pies_count : 
  (total_pies : ℚ) * raspberry_ratio / total_ratio = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_raspberry_pies_count_l3953_395379


namespace NUMINAMATH_CALUDE_expression_values_l3953_395352

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  expr = 5 ∨ expr = 1 ∨ expr = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l3953_395352


namespace NUMINAMATH_CALUDE_chef_wage_percentage_increase_l3953_395355

/-- Proves that the percentage increase in the hourly wage of a chef compared to a dishwasher is 20% -/
theorem chef_wage_percentage_increase (manager_wage : ℝ) (chef_wage : ℝ) (dishwasher_wage : ℝ) :
  manager_wage = 7.5 →
  chef_wage = manager_wage - 3 →
  dishwasher_wage = manager_wage / 2 →
  (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chef_wage_percentage_increase_l3953_395355


namespace NUMINAMATH_CALUDE_not_perfect_square_l3953_395323

theorem not_perfect_square : 
  (∃ x : ℕ, 6^2040 = x^2) ∧ 
  (∀ y : ℕ, 7^2041 ≠ y^2) ∧ 
  (∃ z : ℕ, 8^2042 = z^2) ∧ 
  (∃ w : ℕ, 9^2043 = w^2) ∧ 
  (∃ v : ℕ, 10^2044 = v^2) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3953_395323


namespace NUMINAMATH_CALUDE_inequality_proof_l3953_395319

theorem inequality_proof (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3953_395319


namespace NUMINAMATH_CALUDE_tissue_used_count_l3953_395306

def initial_tissue_count : ℕ := 97
def remaining_tissue_count : ℕ := 93

theorem tissue_used_count : initial_tissue_count - remaining_tissue_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_tissue_used_count_l3953_395306


namespace NUMINAMATH_CALUDE_find_a_l3953_395320

theorem find_a : ∃ a : ℚ, 3 * a - 2 = 2 / 2 + 3 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3953_395320


namespace NUMINAMATH_CALUDE_runners_meet_time_l3953_395307

/-- Two runners on a circular track meet after approximately 15 seconds --/
theorem runners_meet_time (track_length : ℝ) (speed1 speed2 : ℝ) : 
  track_length = 250 →
  speed1 = 20 * (1000 / 3600) →
  speed2 = 40 * (1000 / 3600) →
  abs (15 - track_length / (speed1 + speed2)) < 0.1 := by
  sorry

#check runners_meet_time

end NUMINAMATH_CALUDE_runners_meet_time_l3953_395307


namespace NUMINAMATH_CALUDE_binomial_coefficient_23_5_l3953_395374

theorem binomial_coefficient_23_5 (h1 : Nat.choose 21 3 = 1330)
                                  (h2 : Nat.choose 21 4 = 5985)
                                  (h3 : Nat.choose 21 5 = 20349) :
  Nat.choose 23 5 = 33649 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_23_5_l3953_395374


namespace NUMINAMATH_CALUDE_train_crossing_time_l3953_395356

theorem train_crossing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  platform_length = 320 →
  platform_crossing_time = 34 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * platform_crossing_time - platform_length
  let man_crossing_time := train_length / train_speed_mps
  man_crossing_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3953_395356


namespace NUMINAMATH_CALUDE_triangle_inequality_variant_l3953_395390

theorem triangle_inequality_variant (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x^2 + 3*y^2) + Real.sqrt (x^2 + z^2 + x*z) > Real.sqrt (z^2 + 3*y^2 + 3*y*z) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_variant_l3953_395390


namespace NUMINAMATH_CALUDE_square_difference_l3953_395321

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 1) : a^2 - b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3953_395321


namespace NUMINAMATH_CALUDE_yasmin_children_count_l3953_395367

def john_children (yasmin_children : ℕ) : ℕ := 2 * yasmin_children

theorem yasmin_children_count :
  ∃ (yasmin_children : ℕ),
    yasmin_children = 2 ∧
    john_children yasmin_children + yasmin_children = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_yasmin_children_count_l3953_395367


namespace NUMINAMATH_CALUDE_equation_one_solution_l3953_395315

theorem equation_one_solution (a : ℝ) : 
  (∃! x : ℝ, (Real.log (x + 1) + Real.log (3 - x) = Real.log (1 - a * x)) ∧ 
   (-1 < x ∧ x < 3)) ↔ 
  (-1 ≤ a ∧ a ≤ 1/3) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_l3953_395315


namespace NUMINAMATH_CALUDE_first_five_valid_codes_l3953_395396

def is_valid_code (n : ℕ) : Bool := n < 800

def extract_codes (seq : List ℕ) : List ℕ :=
  seq.filter is_valid_code

theorem first_five_valid_codes 
  (random_sequence : List ℕ := [785, 916, 955, 567, 199, 981, 050, 717, 512]) :
  (extract_codes random_sequence).take 5 = [785, 567, 199, 507, 175] := by
  sorry

end NUMINAMATH_CALUDE_first_five_valid_codes_l3953_395396


namespace NUMINAMATH_CALUDE_budget_allocation_circle_graph_l3953_395395

theorem budget_allocation_circle_graph (microphotonics : ℝ) (home_electronics : ℝ) 
  (food_additives : ℝ) (genetically_modified_microorganisms : ℝ) (industrial_lubricants : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 10 →
  genetically_modified_microorganisms = 29 →
  industrial_lubricants = 8 →
  (360 : ℝ) * (100 - (microphotonics + home_electronics + food_additives + 
    genetically_modified_microorganisms + industrial_lubricants)) / 100 = 54 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_circle_graph_l3953_395395


namespace NUMINAMATH_CALUDE_abs_sum_gt_abs_diff_when_product_positive_l3953_395333

theorem abs_sum_gt_abs_diff_when_product_positive (a b : ℝ) (h : a * b > 0) :
  |a + b| > |a - b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_gt_abs_diff_when_product_positive_l3953_395333


namespace NUMINAMATH_CALUDE_sine_function_vertical_shift_l3953_395385

/-- Given a sine function y = a * sin(b * x + c) + d that oscillates between 4 and -2,
    prove that the vertical shift d equals 1. -/
theorem sine_function_vertical_shift
  (a b c d : ℝ)
  (positive_constants : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (oscillation : ∀ x : ℝ, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_vertical_shift_l3953_395385


namespace NUMINAMATH_CALUDE_remaining_speed_calculation_l3953_395381

/-- Given a trip with the following characteristics:
  * Total distance of 80 miles
  * First 30 miles traveled at 30 mph
  * Average speed for the entire trip is 40 mph
  Prove that the speed for the remaining part of the trip is 50 mph -/
theorem remaining_speed_calculation (total_distance : ℝ) (first_part_distance : ℝ) 
  (first_part_speed : ℝ) (average_speed : ℝ) :
  total_distance = 80 ∧ 
  first_part_distance = 30 ∧ 
  first_part_speed = 30 ∧ 
  average_speed = 40 →
  (total_distance - first_part_distance) / 
    (total_distance / average_speed - first_part_distance / first_part_speed) = 50 :=
by sorry

end NUMINAMATH_CALUDE_remaining_speed_calculation_l3953_395381


namespace NUMINAMATH_CALUDE_cats_weight_l3953_395313

/-- The weight of two cats, where one cat weighs 2 kilograms and the other is twice as heavy, is 6 kilograms. -/
theorem cats_weight (weight_cat1 weight_cat2 : ℝ) : 
  weight_cat1 = 2 → weight_cat2 = 2 * weight_cat1 → weight_cat1 + weight_cat2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cats_weight_l3953_395313


namespace NUMINAMATH_CALUDE_mean_of_combined_sets_l3953_395349

theorem mean_of_combined_sets (set1_count set1_mean set2_count set2_mean : ℚ) 
  (h1 : set1_count = 4)
  (h2 : set1_mean = 15)
  (h3 : set2_count = 8)
  (h4 : set2_mean = 20) :
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 55 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_combined_sets_l3953_395349


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3953_395325

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (a < -4) ↔ 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3953_395325


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l3953_395309

theorem sum_of_specific_numbers : 12534 + 25341 + 53412 + 34125 = 125412 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l3953_395309


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3953_395302

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3953_395302


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3953_395327

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h1 : ∀ n ≥ 1, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h2 : a 4 = 16) 
  (h3 : a 5 = 32) 
  (h4 : a 6 = 64) : 
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3953_395327


namespace NUMINAMATH_CALUDE_f_range_of_a_l3953_395339

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * 2^(x-1) - 1/a else (a-2)*x + 5/3

theorem f_range_of_a (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (x₁ - x₂) * (f a x₁ - f a x₂) > 0) →
  a ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_f_range_of_a_l3953_395339


namespace NUMINAMATH_CALUDE_digits_of_expression_l3953_395300

theorem digits_of_expression : ∃ n : ℕ, n = 12 ∧ n = (Nat.digits 10 (2^15 * 5^12 - 10^5)).length := by
  sorry

end NUMINAMATH_CALUDE_digits_of_expression_l3953_395300


namespace NUMINAMATH_CALUDE_room_length_calculation_l3953_395363

/-- Proves that given a room with specified width, cost per square meter for paving,
    and total paving cost, the length of the room is as calculated. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  width = 3.75 ∧ cost_per_sqm = 600 ∧ total_cost = 12375 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3953_395363


namespace NUMINAMATH_CALUDE_f_properties_l3953_395311

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 * Real.log x

-- State the theorem
theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- Minimum value of f(x) is -1/(2e)
  (∃ x : ℝ, x > 0 ∧ f a x = -(1/(2*Real.exp 1)) ∧ ∀ y : ℝ, y > 0 → f a y ≥ -(1/(2*Real.exp 1))) →
  -- f(x) is decreasing on (0, e^(-1/2))
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < Real.exp (-1/2) → f a x > f a y) ∧
  -- f(x) is increasing on (e^(-1/2), +∞)
  (∀ x y : ℝ, Real.exp (-1/2) < x ∧ x < y → f a x < f a y) ∧
  -- For all x > 0, f(x) > x^2/e^x - 3/4
  (∀ x : ℝ, x > 0 → f a x > x^2 / Real.exp x - 3/4) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3953_395311


namespace NUMINAMATH_CALUDE_track_circumference_is_720_l3953_395399

/-- Represents the circumference of a circular track given specific meeting conditions of two travelers -/
def track_circumference (first_meeting_distance : ℝ) (second_meeting_remaining : ℝ) : ℝ :=
  let half_circumference := 360
  2 * half_circumference

/-- Theorem stating that under the given conditions, the track circumference is 720 yards -/
theorem track_circumference_is_720 :
  track_circumference 150 90 = 720 :=
by
  -- The proof would go here
  sorry

#eval track_circumference 150 90

end NUMINAMATH_CALUDE_track_circumference_is_720_l3953_395399


namespace NUMINAMATH_CALUDE_share_ratio_l3953_395308

/-- Proves that given a total amount of 527 and three shares A = 372, B = 93, and C = 62, 
    the ratio of A's share to B's share is 4:1. -/
theorem share_ratio (total : ℕ) (A B C : ℕ) 
  (h_total : total = 527)
  (h_A : A = 372)
  (h_B : B = 93)
  (h_C : C = 62)
  (h_sum : A + B + C = total) : 
  A / B = 4 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l3953_395308


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3953_395324

def A : Set ℤ := {x | x^2 + x - 6 ≤ 0}
def B : Set ℤ := {x | x ≥ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3953_395324


namespace NUMINAMATH_CALUDE_min_sum_squares_min_value_is_two_l3953_395345

theorem min_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a^2 + b^2 ≤ x^2 + y^2 :=
by sorry

theorem min_value_is_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∃ m : ℝ, m = 2 ∧ a^2 + b^2 ≥ m ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ x^2 + y^2 = m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_value_is_two_l3953_395345


namespace NUMINAMATH_CALUDE_max_value_theorem_l3953_395357

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  x^2 + 3 * x * y + y^2 ≤ (1144 + 204 * Real.sqrt 15) / 91 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3953_395357


namespace NUMINAMATH_CALUDE_fence_poles_count_l3953_395389

def side_length : ℝ := 150
def pole_spacing : ℝ := 30

theorem fence_poles_count :
  let perimeter := 4 * side_length
  let poles_count := perimeter / pole_spacing
  poles_count = 20 := by sorry

end NUMINAMATH_CALUDE_fence_poles_count_l3953_395389


namespace NUMINAMATH_CALUDE_opposite_numbers_l3953_395368

theorem opposite_numbers : -3 = -(Real.sqrt ((-3)^2)) := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l3953_395368


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3953_395394

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (x : ℚ)⁻¹ + (y : ℚ)⁻¹ = 1/12) :
  (∀ a b : ℕ+, a ≠ b → (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 1/12 → (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ)) ∧
  (x : ℕ) + (y : ℕ) = 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3953_395394


namespace NUMINAMATH_CALUDE_max_product_2017_l3953_395369

def sumToN (n : ℕ) := {l : List ℕ | l.sum = n}

def productOfList (l : List ℕ) := l.prod

def optimalSumProduct (n : ℕ) : List ℕ := 
  List.replicate 671 3 ++ List.replicate 2 2

theorem max_product_2017 :
  ∀ l ∈ sumToN 2017, 
    productOfList l ≤ productOfList (optimalSumProduct 2017) :=
sorry

end NUMINAMATH_CALUDE_max_product_2017_l3953_395369


namespace NUMINAMATH_CALUDE_grade_distribution_l3953_395378

theorem grade_distribution (n : ℕ) : 
  ∃ (a b c m : ℕ),
    (2 * m + 3 = n) ∧  -- Total students
    (b = a + 2) ∧      -- B grades
    (c = 2 * b) ∧      -- C grades
    (4 * a + 6 ≠ n)    -- Total A, B, C grades ≠ Total students
  := by sorry

end NUMINAMATH_CALUDE_grade_distribution_l3953_395378


namespace NUMINAMATH_CALUDE_percentage_problem_l3953_395366

theorem percentage_problem : ∃ x : ℝ, 
  (x / 100) * 150 - (20 / 100) * 250 = 43 ∧ 
  x = 62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3953_395366


namespace NUMINAMATH_CALUDE_brads_running_speed_l3953_395398

theorem brads_running_speed
  (distance_between_homes : ℝ)
  (maxwells_speed : ℝ)
  (time_until_meeting : ℝ)
  (brads_delay : ℝ)
  (h1 : distance_between_homes = 54)
  (h2 : maxwells_speed = 4)
  (h3 : time_until_meeting = 6)
  (h4 : brads_delay = 1)
  : ∃ (brads_speed : ℝ), brads_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_brads_running_speed_l3953_395398


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3953_395347

theorem quadratic_root_implies_m_value (m n : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  ((-3 : ℂ) + 2 * Complex.I) ^ 2 + m * ((-3 : ℂ) + 2 * Complex.I) + n = 0 →
  m = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l3953_395347


namespace NUMINAMATH_CALUDE_floor_times_self_eq_54_l3953_395380

theorem floor_times_self_eq_54 (x : ℝ) :
  x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 → x = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_54_l3953_395380


namespace NUMINAMATH_CALUDE_fraction_equality_l3953_395340

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3953_395340


namespace NUMINAMATH_CALUDE_second_divisor_problem_l3953_395317

theorem second_divisor_problem (n : Nat) (h1 : n > 13) (h2 : n ∣ 192) : 
  (197 % n = 5 ∧ ∀ m : Nat, m > 13 → m < n → m ∣ 192 → 197 % m ≠ 5) → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l3953_395317


namespace NUMINAMATH_CALUDE_equal_products_l3953_395314

def numbers : List Nat := [12, 15, 33, 44, 51, 85]
def group1 : List Nat := [12, 33, 85]
def group2 : List Nat := [44, 51, 15]

theorem equal_products :
  (List.prod group1 = List.prod group2) ∧
  (group1.toFinset ∪ group2.toFinset = numbers.toFinset) ∧
  (group1.toFinset ∩ group2.toFinset = ∅) :=
sorry

end NUMINAMATH_CALUDE_equal_products_l3953_395314


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3953_395335

theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3953_395335


namespace NUMINAMATH_CALUDE_arithmetic_combinations_of_threes_l3953_395358

theorem arithmetic_combinations_of_threes : ∃ (op1 op2 op3 op4 op5 op6 op7 op8 op9 op10 op11 op12 : ℝ → ℝ → ℝ),
  (∀ x y, (op1 x y = x / y ∨ op1 x y = x * y ∨ op1 x y = x + y ∨ op1 x y = x - y)) ∧
  (∀ x y, (op2 x y = x / y ∨ op2 x y = x * y ∨ op2 x y = x + y ∨ op2 x y = x - y)) ∧
  (∀ x y, (op3 x y = x / y ∨ op3 x y = x * y ∨ op3 x y = x + y ∨ op3 x y = x - y)) ∧
  (∀ x y, (op4 x y = x / y ∨ op4 x y = x * y ∨ op4 x y = x + y ∨ op4 x y = x - y)) ∧
  (∀ x y, (op5 x y = x / y ∨ op5 x y = x * y ∨ op5 x y = x + y ∨ op5 x y = x - y)) ∧
  (∀ x y, (op6 x y = x / y ∨ op6 x y = x * y ∨ op6 x y = x + y ∨ op6 x y = x - y)) ∧
  (∀ x y, (op7 x y = x / y ∨ op7 x y = x * y ∨ op7 x y = x + y ∨ op7 x y = x - y)) ∧
  (∀ x y, (op8 x y = x / y ∨ op8 x y = x * y ∨ op8 x y = x + y ∨ op8 x y = x - y)) ∧
  (∀ x y, (op9 x y = x / y ∨ op9 x y = x * y ∨ op9 x y = x + y ∨ op9 x y = x - y)) ∧
  (∀ x y, (op10 x y = x / y ∨ op10 x y = x * y ∨ op10 x y = x + y ∨ op10 x y = x - y)) ∧
  (∀ x y, (op11 x y = x / y ∨ op11 x y = x * y ∨ op11 x y = x + y ∨ op11 x y = x - y)) ∧
  (∀ x y, (op12 x y = x / y ∨ op12 x y = x * y ∨ op12 x y = x + y ∨ op12 x y = x - y)) ∧
  op1 (op2 (op3 3 3) 3) 3 = 1 ∧
  op4 (op5 (op6 3 3) 3) 3 = 3 ∧
  op7 (op8 (op9 3 3) 3) 3 = 7 ∧
  op10 (op11 (op12 3 3) 3) 3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_combinations_of_threes_l3953_395358


namespace NUMINAMATH_CALUDE_annie_passes_bonnie_at_six_laps_l3953_395377

/-- Represents the track and runners' properties -/
structure RaceSetup where
  trackLength : ℝ
  annieSpeedFactor : ℝ
  bonnieAcceleration : ℝ

/-- Calculates the number of laps Annie runs when she first passes Bonnie -/
def lapsWhenAnniePasses (setup : RaceSetup) (bonnieInitialSpeed : ℝ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem annie_passes_bonnie_at_six_laps (setup : RaceSetup) (bonnieInitialSpeed : ℝ) 
    (h1 : setup.trackLength = 300)
    (h2 : setup.annieSpeedFactor = 1.2)
    (h3 : setup.bonnieAcceleration = 0.1) :
  lapsWhenAnniePasses setup bonnieInitialSpeed = 6 := by
  sorry

end NUMINAMATH_CALUDE_annie_passes_bonnie_at_six_laps_l3953_395377


namespace NUMINAMATH_CALUDE_quadratic_roots_are_integers_l3953_395361

theorem quadratic_roots_are_integers
  (a b : ℤ)
  (h : ∃ (p q : ℤ), q ≠ 0 ∧ a^2 - 4*b = (p / q : ℚ)^2) :
  ∃ (x : ℤ), x^2 - a*x + b = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_are_integers_l3953_395361


namespace NUMINAMATH_CALUDE_streetlight_problem_l3953_395310

/-- The number of streetlights -/
def total_streetlights : ℕ := 12

/-- The number of streetlights that need to be turned off -/
def lights_to_turn_off : ℕ := 4

/-- The number of available positions to turn off lights, considering the constraints -/
def available_positions : ℕ := total_streetlights - 5

/-- The number of ways to choose 4 non-adjacent positions from 7 available positions -/
def ways_to_turn_off_lights : ℕ := Nat.choose available_positions lights_to_turn_off

theorem streetlight_problem :
  ways_to_turn_off_lights = 35 :=
sorry

end NUMINAMATH_CALUDE_streetlight_problem_l3953_395310


namespace NUMINAMATH_CALUDE_root_sum_property_l3953_395375

theorem root_sum_property (x₁ x₂ : ℝ) (n : ℕ) (hn : n ≥ 1) :
  (x₁^2 - 6*x₁ + 1 = 0) → (x₂^2 - 6*x₂ + 1 = 0) →
  (∃ (m : ℤ), x₁^n + x₂^n = m) ∧ ¬(∃ (k : ℤ), x₁^n + x₂^n = 5*k) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_property_l3953_395375


namespace NUMINAMATH_CALUDE_sin_continuous_l3953_395336

theorem sin_continuous : ContinuousOn Real.sin Set.univ := by
  sorry

end NUMINAMATH_CALUDE_sin_continuous_l3953_395336


namespace NUMINAMATH_CALUDE_min_value_of_f_l3953_395346

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 6*x + 9

-- Theorem stating that the minimum value of f is 0
theorem min_value_of_f :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ f x₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3953_395346


namespace NUMINAMATH_CALUDE_unfixable_percentage_l3953_395397

def total_computers : ℕ := 20
def waiting_percentage : ℚ := 40 / 100
def fixed_right_away : ℕ := 8

theorem unfixable_percentage :
  (total_computers - (waiting_percentage * total_computers).num - fixed_right_away) / total_computers * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_unfixable_percentage_l3953_395397


namespace NUMINAMATH_CALUDE_triple_lcm_equation_l3953_395305

theorem triple_lcm_equation (a b c n : ℕ+) :
  (a.val^2 + b.val^2 = n.val * Nat.lcm a.val b.val + n.val^2) ∧
  (b.val^2 + c.val^2 = n.val * Nat.lcm b.val c.val + n.val^2) ∧
  (c.val^2 + a.val^2 = n.val * Nat.lcm c.val a.val + n.val^2) →
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triple_lcm_equation_l3953_395305


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l3953_395370

theorem max_value_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  (a * b + 2 * b * c + 2 * c * d + d * e) / (a^2 + 3 * b^2 + 3 * c^2 + 5 * d^2 + e^2) ≤ Real.sqrt 2 :=
by sorry

theorem max_value_achievable :
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    (a * b + 2 * b * c + 2 * c * d + d * e) / (a^2 + 3 * b^2 + 3 * c^2 + 5 * d^2 + e^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achievable_l3953_395370


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_distinct_l3953_395365

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 20) ∧
  (n / 1000 ≠ n / 100 % 10) ∧ (n / 1000 ≠ n / 10 % 10) ∧ (n / 1000 ≠ n % 10) ∧
  (n / 100 % 10 ≠ n / 10 % 10) ∧ (n / 100 % 10 ≠ n % 10) ∧
  (n / 10 % 10 ≠ n % 10)

theorem largest_four_digit_sum_20_distinct : 
  ∀ n : ℕ, is_valid_number n → n ≤ 9821 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_distinct_l3953_395365


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l3953_395312

theorem smallest_five_digit_multiple_of_18 : ∀ n : ℕ, 
  n ≥ 10000 ∧ n ≤ 99999 ∧ n % 18 = 0 → n ≥ 10008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l3953_395312


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l3953_395393

def total_republicans : Nat := 10
def total_democrats : Nat := 8
def subcommittee_republicans : Nat := 4
def subcommittee_democrats : Nat := 3
def senior_democrat : Nat := 1

def ways_to_form_subcommittee : Nat :=
  Nat.choose total_republicans subcommittee_republicans *
  Nat.choose (total_democrats - senior_democrat) (subcommittee_democrats - senior_democrat)

theorem subcommittee_formation_count :
  ways_to_form_subcommittee = 4410 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l3953_395393


namespace NUMINAMATH_CALUDE_simplify_expression_l3953_395329

theorem simplify_expression : (2^8 + 5^3) * (2^2 - (-1)^5)^7 = 29765625 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3953_395329


namespace NUMINAMATH_CALUDE_quadratic_discriminant_with_specific_roots_l3953_395364

/-- The discriminant of a quadratic polynomial with specific root conditions -/
theorem quadratic_discriminant_with_specific_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) ∧
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4 * a * c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_with_specific_roots_l3953_395364


namespace NUMINAMATH_CALUDE_trig_inequality_l3953_395341

theorem trig_inequality (α β γ : Real) 
  (h1 : 0 ≤ α ∧ α < Real.pi / 2)
  (h2 : 0 ≤ β ∧ β < Real.pi / 2)
  (h3 : 0 ≤ γ ∧ γ < Real.pi / 2)
  (h4 : Real.sin α + Real.sin β + Real.sin γ = 1) :
  Real.tan α ^ 2 + Real.tan β ^ 2 + Real.tan γ ^ 2 ≥ 3/8 := by
sorry

end NUMINAMATH_CALUDE_trig_inequality_l3953_395341


namespace NUMINAMATH_CALUDE_point_division_l3953_395331

/-- Given two points A and B in a vector space, and a point P on the line segment AB
    such that AP:PB = 4:1, prove that P = (4/5)*A + (1/5)*B -/
theorem point_division (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (A B P : V) (h : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ P - A = (4 * k) • (B - A) ∧ B - P = k • (B - A)) :
  P = (4/5) • A + (1/5) • B := by
  sorry

end NUMINAMATH_CALUDE_point_division_l3953_395331


namespace NUMINAMATH_CALUDE_max_min_x2_xy_y2_l3953_395303

theorem max_min_x2_xy_y2 (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (A_min A_max : ℝ), A_min = 2 ∧ A_max = 6 ∧
  ∀ A, A = x^2 + x*y + y^2 → A_min ≤ A ∧ A ≤ A_max :=
by sorry

end NUMINAMATH_CALUDE_max_min_x2_xy_y2_l3953_395303


namespace NUMINAMATH_CALUDE_preimage_of_one_is_zero_one_neg_one_l3953_395316

-- Define the sets A and B as subsets of ℝ
variable (A B : Set ℝ)

-- Define the function f: A → B
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the set of elements in A that map to 1 under f
def preimage_of_one (A : Set ℝ) : Set ℝ := {x ∈ A | f x = 1}

-- Theorem statement
theorem preimage_of_one_is_zero_one_neg_one (A B : Set ℝ) :
  preimage_of_one A = {0, 1, -1} := by sorry

end NUMINAMATH_CALUDE_preimage_of_one_is_zero_one_neg_one_l3953_395316


namespace NUMINAMATH_CALUDE_bananas_in_jar_l3953_395362

/-- The number of bananas originally in the jar -/
def original_bananas : ℕ := 46

/-- The number of bananas removed from the jar -/
def removed_bananas : ℕ := 5

/-- The number of bananas left in the jar after removal -/
def remaining_bananas : ℕ := 41

/-- Theorem stating that the original number of bananas is equal to the sum of removed and remaining bananas -/
theorem bananas_in_jar : original_bananas = removed_bananas + remaining_bananas := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_jar_l3953_395362


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3953_395360

theorem inequality_solution_set (x : ℝ) :
  (x + 5) / (x - 1) ≥ 2 ↔ x ∈ Set.Ioo 1 7 ∪ {7} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3953_395360


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l3953_395372

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_sin : Real.sin (θ / 2) = Real.sqrt ((x - 1) / (2 * x)))
  (h_x_pos : x > 0) : 
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l3953_395372


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3953_395332

-- Define the quadratic function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_function_properties (b c : ℝ) :
  (∀ α : ℝ, f b c (Real.sin α) ≥ 0) →
  (∀ β : ℝ, f b c (2 + Real.cos β) ≤ 0) →
  (∃ M : ℝ, M = 8 ∧ ∀ α : ℝ, f b c (Real.sin α) ≤ M) →
  b = -4 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3953_395332


namespace NUMINAMATH_CALUDE_expand_product_l3953_395342

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3953_395342


namespace NUMINAMATH_CALUDE_people_counting_ratio_l3953_395376

theorem people_counting_ratio :
  ∀ (day1 day2 : ℕ),
  day2 = 500 →
  day1 + day2 = 1500 →
  ∃ (k : ℕ), day1 = k * day2 →
  day1 / day2 = 2 := by
sorry

end NUMINAMATH_CALUDE_people_counting_ratio_l3953_395376


namespace NUMINAMATH_CALUDE_sqrt_trig_identity_l3953_395330

theorem sqrt_trig_identity : 
  Real.sqrt (1 - 2 * Real.cos (π / 2 + 3) * Real.sin (π / 2 - 3)) = -Real.sin 3 - Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_trig_identity_l3953_395330


namespace NUMINAMATH_CALUDE_cab_journey_time_l3953_395343

/-- 
If a cab traveling at 5/6 of its usual speed is 12 minutes late, 
then its usual time to cover the journey is 1 hour.
-/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) :
  (usual_time + 1/5) / usual_time = 6/5 → usual_time = 1 := by
  sorry

#check cab_journey_time

end NUMINAMATH_CALUDE_cab_journey_time_l3953_395343


namespace NUMINAMATH_CALUDE_intersection_point_l3953_395322

-- Define the line
def line (t : ℝ) : ℝ × ℝ × ℝ := (1 + t, -1, 1 - t)

-- Define the plane
def plane (x y z : ℝ) : Prop := 3*x - 2*y - 4*z - 8 = 0

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ × ℝ, 
    (∃ t : ℝ, line t = p) ∧ 
    plane p.1 p.2.1 p.2.2 ∧
    p = (-6, -1, 8) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l3953_395322


namespace NUMINAMATH_CALUDE_tea_price_calculation_l3953_395371

/-- The price of the first variety of tea in rupees per kg -/
def first_tea_price : ℝ := 126

/-- The price of the second variety of tea in rupees per kg -/
def second_tea_price : ℝ := 135

/-- The price of the third variety of tea in rupees per kg -/
def third_tea_price : ℝ := 175.5

/-- The price of the mixture in rupees per kg -/
def mixture_price : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def first_ratio : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def second_ratio : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def third_ratio : ℝ := 2

theorem tea_price_calculation :
  first_tea_price * first_ratio + 
  second_tea_price * second_ratio + 
  third_tea_price * third_ratio = 
  mixture_price * (first_ratio + second_ratio + third_ratio) := by
  sorry

#check tea_price_calculation

end NUMINAMATH_CALUDE_tea_price_calculation_l3953_395371


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l3953_395351

-- Define the triangle and circle
structure Triangle :=
  (A B C : ℝ × ℝ)

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the inscribed circle property
def isInscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define the point of tangency
def pointOfTangency (t : Triangle) (c : Circle) : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two vectors
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem inscribed_circle_theorem (t : Triangle) (c : Circle) (M : ℝ × ℝ) :
  isInscribed t c →
  M = pointOfTangency t c →
  distance t.A M = 1 →
  distance t.B M = 4 →
  angle (t.B - t.A) (t.C - t.A) = 2 * π / 3 →
  distance t.C M = Real.sqrt 273 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l3953_395351


namespace NUMINAMATH_CALUDE_inequality_proof_l3953_395304

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3953_395304
