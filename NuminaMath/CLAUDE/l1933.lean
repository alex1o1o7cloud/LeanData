import Mathlib

namespace NUMINAMATH_CALUDE_fraction_subtraction_proof_l1933_193357

theorem fraction_subtraction_proof :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_proof_l1933_193357


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1933_193384

/-- Given a quadratic of the form x^2 + bx + 50 where b is positive,
    if it can be written as (x+n)^2 + 8, then b = 2√42. -/
theorem quadratic_coefficient (b n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 50 = (x+n)^2 + 8) → 
  b = 2 * Real.sqrt 42 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1933_193384


namespace NUMINAMATH_CALUDE_power_of_two_subset_bound_l1933_193340

/-- The set of powers of 2 from 2^7 to 2^n -/
def S (n : ℕ) : Set ℕ := {x | ∃ k, 7 ≤ k ∧ k ≤ n ∧ x = 2^k}

/-- The subset of S where the sum of the last three digits is 8 -/
def A (n : ℕ) : Set ℕ := {x ∈ S n | (x % 1000 / 100 + x % 100 / 10 + x % 10) = 8}

/-- The number of elements in a finite set -/
def card {α : Type*} (s : Set α) : ℕ := sorry

theorem power_of_two_subset_bound (n : ℕ) (h : n ≥ 2009) :
  (28 : ℚ) / 2009 < (card (A n) : ℚ) / (card (S n)) ∧
  (card (A n) : ℚ) / (card (S n)) < 82 / 2009 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_subset_bound_l1933_193340


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_1729728_l1933_193382

theorem sum_of_prime_factors_1729728 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (1729728 + 1))) id) = 36 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_1729728_l1933_193382


namespace NUMINAMATH_CALUDE_supplementary_angles_problem_l1933_193348

theorem supplementary_angles_problem (A B : Real) : 
  A + B = 180 →  -- angles A and B are supplementary
  A = 7 * B →    -- measure of angle A is 7 times angle B
  A = 157.5 :=   -- prove that measure of angle A is 157.5°
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_problem_l1933_193348


namespace NUMINAMATH_CALUDE_initial_cookies_l1933_193356

/-- The number of cookies remaining after the first day -/
def remaining_after_first_day (C : ℚ) : ℚ :=
  C * (1/4) * (4/5)

/-- The number of cookies remaining after the second day -/
def remaining_after_second_day (C : ℚ) : ℚ :=
  remaining_after_first_day C * (1/2)

/-- Theorem stating the initial number of cookies -/
theorem initial_cookies : ∃ C : ℚ, C > 0 ∧ remaining_after_second_day C = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_cookies_l1933_193356


namespace NUMINAMATH_CALUDE_linda_savings_l1933_193329

theorem linda_savings (savings : ℝ) : (1 / 4 : ℝ) * savings = 220 → savings = 880 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l1933_193329


namespace NUMINAMATH_CALUDE_prob_not_both_white_l1933_193351

theorem prob_not_both_white (prob_white_A prob_white_B : ℚ) 
  (h1 : prob_white_A = 1/3)
  (h2 : prob_white_B = 1/2) :
  1 - prob_white_A * prob_white_B = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_both_white_l1933_193351


namespace NUMINAMATH_CALUDE_water_requirement_l1933_193397

/-- Represents the amount of a substance in moles -/
def Moles : Type := ℝ

/-- Represents a chemical reaction between ammonium chloride and water -/
structure Reaction where
  nh4cl : Moles  -- Amount of ammonium chloride
  h2o : Moles    -- Amount of water
  hcl : Moles    -- Amount of hydrochloric acid produced
  nh4oh : Moles  -- Amount of ammonium hydroxide produced

/-- The reaction is balanced when the amounts of reactants and products are in the correct proportion -/
def is_balanced (r : Reaction) : Prop :=
  r.nh4cl = r.h2o ∧ r.nh4cl = r.hcl ∧ r.nh4cl = r.nh4oh

/-- The amount of water required is equal to the amount of ammonium chloride when the reaction is balanced -/
theorem water_requirement (r : Reaction) (h : is_balanced r) : r.h2o = r.nh4cl := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_l1933_193397


namespace NUMINAMATH_CALUDE_square_root_49_squared_l1933_193374

theorem square_root_49_squared : Real.sqrt 49 ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_49_squared_l1933_193374


namespace NUMINAMATH_CALUDE_child_ticket_cost_l1933_193385

theorem child_ticket_cost (adult_price : ℕ) (total_people : ℕ) (total_revenue : ℕ) (num_children : ℕ) :
  adult_price = 11 →
  total_people = 23 →
  total_revenue = 246 →
  num_children = 7 →
  ∃ (child_price : ℕ), child_price = 10 ∧ 
    adult_price * (total_people - num_children) + child_price * num_children = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l1933_193385


namespace NUMINAMATH_CALUDE_min_value_of_sum_roots_l1933_193315

theorem min_value_of_sum_roots (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hsum : a + b + c + d = 1) : 
  Real.sqrt (a^2 + 1/(8*a)) + Real.sqrt (b^2 + 1/(8*b)) + 
  Real.sqrt (c^2 + 1/(8*c)) + Real.sqrt (d^2 + 1/(8*d)) ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_roots_l1933_193315


namespace NUMINAMATH_CALUDE_librarians_work_schedule_l1933_193314

theorem librarians_work_schedule : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 14)) = 280 := by
  sorry

end NUMINAMATH_CALUDE_librarians_work_schedule_l1933_193314


namespace NUMINAMATH_CALUDE_star_properties_l1933_193366

/-- The star operation -/
def star (a b : ℝ) : ℝ := a + b + a * b

/-- The prime operation -/
noncomputable def prime (a : ℝ) : ℝ := -a / (a + 1)

theorem star_properties :
  ∀ (a b c : ℝ),
  (a ≠ -1) →
  (b ≠ -1) →
  (c ≠ -1) →
  (∀ (x : ℝ), star a x = c ↔ x = (c - a) / (a + 1)) ∧
  (star a (prime a) = 0) ∧
  (prime (prime a) = a) ∧
  (prime (star a b) = star (prime a) (prime b)) ∧
  (prime (star (prime a) b) = star (prime a) (prime b)) ∧
  (prime (star (prime a) (prime b)) = star a b) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l1933_193366


namespace NUMINAMATH_CALUDE_jesses_room_carpet_area_l1933_193333

theorem jesses_room_carpet_area :
  let room_length : ℝ := 12
  let room_width : ℝ := 8
  let room_area := room_length * room_width
  room_area = 96 := by sorry

end NUMINAMATH_CALUDE_jesses_room_carpet_area_l1933_193333


namespace NUMINAMATH_CALUDE_midpoint_property_l1933_193344

/-- Given two points A and B in a 2D plane, if C is their midpoint,
    then 3 times the x-coordinate of C minus 2 times the y-coordinate of C equals -3. -/
theorem midpoint_property (A B C : ℝ × ℝ) :
  A = (-6, 9) →
  B = (8, -3) →
  C.1 = (A.1 + B.1) / 2 →
  C.2 = (A.2 + B.2) / 2 →
  3 * C.1 - 2 * C.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_property_l1933_193344


namespace NUMINAMATH_CALUDE_certain_number_problem_l1933_193395

theorem certain_number_problem : ∃! x : ℝ, x / 9 + x + 9 = 69 := by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1933_193395


namespace NUMINAMATH_CALUDE_tims_trip_duration_l1933_193310

/-- Calculates the total duration of Tim's trip given the specified conditions -/
theorem tims_trip_duration :
  let total_driving_time : ℝ := 5
  let num_traffic_jams : ℕ := 3
  let first_jam_multiplier : ℝ := 1.5
  let second_jam_multiplier : ℝ := 2
  let third_jam_multiplier : ℝ := 3
  let num_pit_stops : ℕ := 2
  let pit_stop_duration : ℝ := 0.5
  let time_before_first_jam : ℝ := 1
  let time_between_first_and_second_jam : ℝ := 1.5

  let first_jam_duration : ℝ := first_jam_multiplier * time_before_first_jam
  let second_jam_duration : ℝ := second_jam_multiplier * time_between_first_and_second_jam
  let time_before_third_jam : ℝ := total_driving_time - time_before_first_jam - time_between_first_and_second_jam
  let third_jam_duration : ℝ := third_jam_multiplier * time_before_third_jam
  let total_pit_stop_time : ℝ := num_pit_stops * pit_stop_duration

  let total_duration : ℝ := total_driving_time + first_jam_duration + second_jam_duration + third_jam_duration + total_pit_stop_time

  total_duration = 18 := by sorry

end NUMINAMATH_CALUDE_tims_trip_duration_l1933_193310


namespace NUMINAMATH_CALUDE_terrys_breakfast_spending_l1933_193362

theorem terrys_breakfast_spending (x : ℝ) : 
  x > 0 ∧ x + 2*x + 6*x = 54 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_terrys_breakfast_spending_l1933_193362


namespace NUMINAMATH_CALUDE_bc_distances_l1933_193338

/-- Represents a circular road with four gas stations -/
structure CircularRoad where
  circumference : ℝ
  distAB : ℝ
  distAC : ℝ
  distCD : ℝ
  distDA : ℝ

/-- Theorem stating the possible distances between B and C -/
theorem bc_distances (road : CircularRoad)
  (h_circ : road.circumference = 100)
  (h_ab : road.distAB = 50)
  (h_ac : road.distAC = 40)
  (h_cd : road.distCD = 25)
  (h_da : road.distDA = 35) :
  ∃ (d1 d2 : ℝ), d1 = 10 ∧ d2 = 90 ∧
  (∀ (d : ℝ), (d = d1 ∨ d = d2) ↔ 
    (d = road.distAB - road.distAC ∨ 
     d = road.circumference - (road.distAB + road.distAC))) :=
by sorry

end NUMINAMATH_CALUDE_bc_distances_l1933_193338


namespace NUMINAMATH_CALUDE_at_least_one_red_ball_probability_l1933_193332

theorem at_least_one_red_ball_probability 
  (prob_red_A : ℝ) 
  (prob_red_B : ℝ) 
  (h1 : prob_red_A = 1/3) 
  (h2 : prob_red_B = 1/2) :
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_red_ball_probability_l1933_193332


namespace NUMINAMATH_CALUDE_fraction_to_decimal_conversion_l1933_193335

theorem fraction_to_decimal_conversion :
  ∃ (d : ℚ), (d.num / d.den = 7 / 12) ∧ (abs (d - 0.58333) < 0.000005) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_conversion_l1933_193335


namespace NUMINAMATH_CALUDE_cyclic_power_inequality_l1933_193301

theorem cyclic_power_inequality (a b c r s : ℝ) 
  (hr : r > s) (hs : s > 0) (hab : a > b) (hbc : b > c) :
  a^r * b^s + b^r * c^s + c^r * a^s ≥ a^s * b^r + b^s * c^r + c^s * a^r :=
by sorry

end NUMINAMATH_CALUDE_cyclic_power_inequality_l1933_193301


namespace NUMINAMATH_CALUDE_simple_random_sampling_probability_l1933_193359

theorem simple_random_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 8)
  (h2 : sample_size = 4) :
  (sample_size : ℚ) / population_size = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_random_sampling_probability_l1933_193359


namespace NUMINAMATH_CALUDE_min_value_inequality_l1933_193393

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  Real.sqrt ((x^2 + y^2) * (5 * x^2 + y^2)) / (x * y) ≥ Real.sqrt 5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1933_193393


namespace NUMINAMATH_CALUDE_fossil_age_count_is_40_l1933_193303

/-- The number of possible 6-digit numbers formed using the digits 1 (three times), 4, 8, and 9,
    where the number must end with an even digit. -/
def fossil_age_count : ℕ :=
  let digits : List ℕ := [1, 1, 1, 4, 8, 9]
  let even_endings : List ℕ := [4, 8]
  2 * (Nat.factorial 5 / Nat.factorial 3)

theorem fossil_age_count_is_40 : fossil_age_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_fossil_age_count_is_40_l1933_193303


namespace NUMINAMATH_CALUDE_inequality_proof_l1933_193326

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 3) :
  (a / (1 + b^2 * c)) + (b / (1 + c^2 * a)) + (c / (1 + a^2 * b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1933_193326


namespace NUMINAMATH_CALUDE_odd_square_plus_four_odd_l1933_193370

theorem odd_square_plus_four_odd (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  Odd (p^2 + 4*q) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_plus_four_odd_l1933_193370


namespace NUMINAMATH_CALUDE_gnomes_taken_is_40_percent_l1933_193349

/-- The percentage of gnomes taken by the forest owner in Ravenswood forest --/
def gnomes_taken_percentage (westerville_gnomes : ℕ) (ravenswood_multiplier : ℕ) (remaining_gnomes : ℕ) : ℚ :=
  100 - (remaining_gnomes : ℚ) / ((westerville_gnomes * ravenswood_multiplier) : ℚ) * 100

/-- Theorem stating that the percentage of gnomes taken is 40% --/
theorem gnomes_taken_is_40_percent :
  gnomes_taken_percentage 20 4 48 = 40 := by
  sorry

end NUMINAMATH_CALUDE_gnomes_taken_is_40_percent_l1933_193349


namespace NUMINAMATH_CALUDE_worksheet_grading_problem_l1933_193307

theorem worksheet_grading_problem 
  (problems_per_worksheet : ℕ)
  (worksheets_graded : ℕ)
  (remaining_problems : ℕ)
  (h1 : problems_per_worksheet = 4)
  (h2 : worksheets_graded = 5)
  (h3 : remaining_problems = 16) :
  worksheets_graded + (remaining_problems / problems_per_worksheet) = 9 := by
  sorry

end NUMINAMATH_CALUDE_worksheet_grading_problem_l1933_193307


namespace NUMINAMATH_CALUDE_midnight_temperature_l1933_193367

/-- Given an initial temperature, a temperature rise, and a temperature drop,
    calculate the final temperature. -/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/-- Theorem stating that given the specific temperature changes in the problem,
    the final temperature is -8°C. -/
theorem midnight_temperature :
  final_temperature (-5) 5 8 = -8 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l1933_193367


namespace NUMINAMATH_CALUDE_daughters_work_time_l1933_193308

/-- Given a man can do a piece of work in 4 days, and together with his daughter
    they can do it in 3 days, prove that the daughter can do the work alone in 12 days. -/
theorem daughters_work_time (man_time : ℕ) (combined_time : ℕ) (daughter_time : ℕ) :
  man_time = 4 →
  combined_time = 3 →
  (1 : ℚ) / man_time + (1 : ℚ) / daughter_time = (1 : ℚ) / combined_time →
  daughter_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_daughters_work_time_l1933_193308


namespace NUMINAMATH_CALUDE_fraction_simplification_l1933_193364

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 12) / ((x^2 - 4*x + 4) / (x^2 - 6*x + 9)) = 
  ((x - 1) * (x - 3)^2) / ((x - 4) * (x - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1933_193364


namespace NUMINAMATH_CALUDE_people_in_room_l1933_193390

theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs ∧ 
  chairs - (4 : ℚ) / 5 * chairs = 10 →
  people = 67 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l1933_193390


namespace NUMINAMATH_CALUDE_smallest_number_l1933_193337

def number_set : Set ℤ := {-3, 2, -2, 0}

theorem smallest_number : ∀ x ∈ number_set, -3 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_number_l1933_193337


namespace NUMINAMATH_CALUDE_experimental_plans_count_l1933_193377

/-- The number of ways to select an even number of elements from a set of 6 elements -/
def evenSelectionA : ℕ := Finset.sum (Finset.filter (λ k => k % 2 = 0) (Finset.range 7)) (λ k => Nat.choose 6 k)

/-- The number of ways to select at least 2 elements from a set of 4 elements -/
def atLeastTwoSelectionB : ℕ := Finset.sum (Finset.range 3) (λ k => Nat.choose 4 (k + 2))

/-- The total number of experimental plans -/
def totalExperimentalPlans : ℕ := evenSelectionA * atLeastTwoSelectionB

theorem experimental_plans_count : totalExperimentalPlans = 352 := by
  sorry

end NUMINAMATH_CALUDE_experimental_plans_count_l1933_193377


namespace NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l1933_193398

theorem positive_sum_from_positive_difference (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_from_positive_difference_l1933_193398


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1933_193334

/-- For a parallelogram with given height and area, prove that its base length is as calculated. -/
theorem parallelogram_base_length (height area : ℝ) (h_height : height = 11) (h_area : area = 44) :
  area / height = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1933_193334


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1933_193363

theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ - 1 = 0 ∧ k * x₂^2 - 2 * x₂ - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1933_193363


namespace NUMINAMATH_CALUDE_candy_problem_l1933_193304

theorem candy_problem (r g b : ℕ+) (n : ℕ) : 
  (10 * r = 16 * g) → 
  (16 * g = 18 * b) → 
  (18 * b = 25 * n) → 
  (∀ m : ℕ, m < n → ¬(∃ r' g' b' : ℕ+, 10 * r' = 16 * g' ∧ 16 * g' = 18 * b' ∧ 18 * b' = 25 * m)) →
  n = 29 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l1933_193304


namespace NUMINAMATH_CALUDE_zookeeper_excess_fish_l1933_193358

/-- The number of penguins in the zoo -/
def total_penguins : ℕ := 48

/-- The ratio of Emperor to Adelie penguins -/
def emperor_ratio : ℕ := 3
def adelie_ratio : ℕ := 5

/-- The amount of fish needed for each type of penguin -/
def emperor_fish_need : ℚ := 3/2
def adelie_fish_need : ℕ := 2

/-- The percentage of additional fish the zookeeper has -/
def additional_fish_percentage : ℕ := 150

theorem zookeeper_excess_fish :
  let emperor_count : ℕ := (emperor_ratio * total_penguins) / (emperor_ratio + adelie_ratio)
  let adelie_count : ℕ := (adelie_ratio * total_penguins) / (emperor_ratio + adelie_ratio)
  let total_fish_needed : ℚ := emperor_count * emperor_fish_need + adelie_count * adelie_fish_need
  let zookeeper_fish : ℕ := total_penguins + (additional_fish_percentage * total_penguins) / 100
  (zookeeper_fish : ℚ) - total_fish_needed = 33 := by
  sorry

end NUMINAMATH_CALUDE_zookeeper_excess_fish_l1933_193358


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l1933_193346

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem problem_solution : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 180 = 5 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l1933_193346


namespace NUMINAMATH_CALUDE_symmetrical_circles_sum_l1933_193391

/-- Two circles are symmetrical with respect to the line y = x + 1 -/
def symmetrical_circles (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = 1 ∧ 
                (x - 1)^2 + (y - 3)^2 = 1 ∧
                y = x + 1

/-- If two circles are symmetrical with respect to the line y = x + 1,
    then the sum of their center coordinates is 2 -/
theorem symmetrical_circles_sum (a b : ℝ) :
  symmetrical_circles a b → a + b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetrical_circles_sum_l1933_193391


namespace NUMINAMATH_CALUDE_expression_evaluation_l1933_193388

theorem expression_evaluation : (20 + 16 * 20) / (20 * 16) = 17 / 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1933_193388


namespace NUMINAMATH_CALUDE_zip_code_sum_l1933_193309

theorem zip_code_sum (a b c d e : ℕ) : 
  a + b + c + d + e = 10 →
  a = b →
  c = 0 →
  d = 2 * a →
  d + e = 8 := by
sorry

end NUMINAMATH_CALUDE_zip_code_sum_l1933_193309


namespace NUMINAMATH_CALUDE_correct_travel_times_l1933_193372

/-- Represents the travel times of Winnie-the-Pooh and Piglet -/
structure TravelTimes where
  pooh : ℝ
  piglet : ℝ

/-- Calculates the travel times based on the given conditions -/
def calculate_travel_times (time_after_meeting_pooh : ℝ) (time_after_meeting_piglet : ℝ) : TravelTimes :=
  let speed_ratio := time_after_meeting_pooh / time_after_meeting_piglet
  let time_before_meeting := time_after_meeting_piglet
  { pooh := time_before_meeting + time_after_meeting_piglet
  , piglet := time_before_meeting + time_after_meeting_pooh }

/-- Theorem stating that the calculated travel times are correct -/
theorem correct_travel_times :
  let result := calculate_travel_times 4 1
  result.pooh = 2 ∧ result.piglet = 6 := by sorry

end NUMINAMATH_CALUDE_correct_travel_times_l1933_193372


namespace NUMINAMATH_CALUDE_distance_squared_is_53_l1933_193399

/-- A notched circle with specific measurements -/
structure NotchedCircle where
  radius : ℝ
  AB : ℝ
  BC : ℝ
  right_angle : Bool

/-- The square of the distance from point B to the center of the circle -/
def distance_squared (nc : NotchedCircle) : ℝ :=
  sorry

/-- Theorem stating the square of the distance from B to the center is 53 -/
theorem distance_squared_is_53 (nc : NotchedCircle) 
  (h1 : nc.radius = Real.sqrt 72)
  (h2 : nc.AB = 8)
  (h3 : nc.BC = 3)
  (h4 : nc.right_angle = true) :
  distance_squared nc = 53 :=
sorry

end NUMINAMATH_CALUDE_distance_squared_is_53_l1933_193399


namespace NUMINAMATH_CALUDE_garden_perimeter_is_700_l1933_193336

/-- The perimeter of a rectangular garden with given length and breadth -/
def garden_perimeter (length : ℝ) (breadth : ℝ) : ℝ :=
  2 * (length + breadth)

theorem garden_perimeter_is_700 :
  garden_perimeter 250 100 = 700 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_700_l1933_193336


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1933_193347

theorem solve_system_of_equations (a x y : ℚ) 
  (eq1 : a * x + y = 8)
  (eq2 : 3 * x - 4 * y = 5)
  (eq3 : 7 * x - 3 * y = 23) :
  a = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1933_193347


namespace NUMINAMATH_CALUDE_A_power_50_l1933_193319

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem A_power_50 : A^50 = !![(-301 : ℤ), -100; 800, 299] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l1933_193319


namespace NUMINAMATH_CALUDE_reflection_of_C_l1933_193381

/-- Reflects a point over the line y = x -/
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem reflection_of_C : 
  let C : ℝ × ℝ := (2, 2)
  reflect_over_y_eq_x C = C :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_C_l1933_193381


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1933_193371

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 3x + 5 and y = (5k)x + 7 are parallel -/
theorem parallel_lines_k_value : 
  (∀ x y : ℝ, y = 3 * x + 5 ↔ y = (5 * k) * x + 7) → k = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1933_193371


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1933_193355

theorem min_value_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : 
  x + y ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 9 ∧ x + y = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1933_193355


namespace NUMINAMATH_CALUDE_total_vehicles_is_2800_l1933_193387

/-- Calculates the total number of vehicles on a road with given conditions -/
def totalVehicles (lanes : ℕ) (trucksPerLane : ℕ) (busesPerLane : ℕ) : ℕ :=
  let totalTrucks := lanes * trucksPerLane
  let carsPerLane := 2 * totalTrucks
  let totalCars := lanes * carsPerLane
  let totalBuses := lanes * busesPerLane
  let motorcyclesPerLane := 3 * busesPerLane
  let totalMotorcycles := lanes * motorcyclesPerLane
  totalTrucks + totalCars + totalBuses + totalMotorcycles

/-- Theorem stating that under the given conditions, the total number of vehicles is 2800 -/
theorem total_vehicles_is_2800 : totalVehicles 4 60 40 = 2800 := by
  sorry

#eval totalVehicles 4 60 40

end NUMINAMATH_CALUDE_total_vehicles_is_2800_l1933_193387


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_and_a_div_b_l1933_193341

theorem range_of_a_minus_b_and_a_div_b (a b : ℝ) 
  (ha : 12 < a ∧ a < 60) (hb : 15 < b ∧ b < 36) : 
  (-24 < a - b ∧ a - b < 45) ∧ (1/3 < a/b ∧ a/b < 4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_and_a_div_b_l1933_193341


namespace NUMINAMATH_CALUDE_star_five_three_l1933_193389

-- Define the ※ operation
def star (a b : ℝ) : ℝ := b^2 + 1

-- Theorem statement
theorem star_five_three : star 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l1933_193389


namespace NUMINAMATH_CALUDE_subscription_difference_is_5000_l1933_193365

/-- Represents the subscription amounts and profit distribution in a business venture -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  a_profit : ℕ
  a_extra : ℕ

/-- Calculates the difference between B's and C's subscriptions -/
def subscription_difference (bv : BusinessVenture) : ℕ :=
  let b_subscription := (bv.total_subscription * bv.a_profit * 2) / (bv.total_profit * 3) - bv.a_extra
  let c_subscription := bv.total_subscription - b_subscription - (b_subscription + bv.a_extra)
  b_subscription - c_subscription

/-- Theorem stating that the difference between B's and C's subscriptions is 5000 -/
theorem subscription_difference_is_5000 (bv : BusinessVenture) 
    (h1 : bv.total_subscription = 50000)
    (h2 : bv.total_profit = 35000)
    (h3 : bv.a_profit = 14700)
    (h4 : bv.a_extra = 4000) :
  subscription_difference bv = 5000 := by
  sorry

#eval subscription_difference ⟨50000, 35000, 14700, 4000⟩

end NUMINAMATH_CALUDE_subscription_difference_is_5000_l1933_193365


namespace NUMINAMATH_CALUDE_nice_function_property_l1933_193379

def nice (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, f^[a] b = f (a + b - 1)

theorem nice_function_property (g : ℕ → ℕ) (A : ℕ) 
  (hg : nice g)
  (hA : g (A + 2018) = g A + 1)
  (hA1 : g (A + 1) ≠ g (A + 1 + 2017^2017)) :
  ∀ n < A, g n = n + 1 := by sorry

end NUMINAMATH_CALUDE_nice_function_property_l1933_193379


namespace NUMINAMATH_CALUDE_power_sum_zero_l1933_193345

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l1933_193345


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l1933_193373

/-- The number of ways to distribute n distinct objects into m distinct containers,
    where each container can hold multiple objects. -/
def distribute (n m : ℕ) : ℕ := m^n

/-- The number of distinct arrangements for wearing 5 different rings
    on the 5 fingers of the right hand. -/
def ringArrangements : ℕ := distribute 5 5

theorem ring_arrangements_count :
  ringArrangements = 5^5 := by sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l1933_193373


namespace NUMINAMATH_CALUDE_difference_of_products_equals_one_l1933_193325

theorem difference_of_products_equals_one : (1011 : ℕ) * 1011 - 1010 * 1012 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_products_equals_one_l1933_193325


namespace NUMINAMATH_CALUDE_reciprocal_pairs_l1933_193330

def are_reciprocals (a b : ℚ) : Prop := a * b = 1

theorem reciprocal_pairs :
  ¬(are_reciprocals 1 (-1)) ∧
  ¬(are_reciprocals (-1/3) 3) ∧
  are_reciprocals (-5) (-1/5) ∧
  ¬(are_reciprocals (-3) (|(-3)|)) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_pairs_l1933_193330


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l1933_193383

theorem rectangular_field_perimeter (length width : ℝ) (area perimeter : ℝ) : 
  width = 0.6 * length →
  area = length * width →
  area = 37500 →
  perimeter = 2 * (length + width) →
  perimeter = 800 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l1933_193383


namespace NUMINAMATH_CALUDE_circle_radius_l1933_193375

theorem circle_radius (r : ℝ) (h : r > 0) :
  π * r^2 + 2 * π * r = 100 * π → r = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l1933_193375


namespace NUMINAMATH_CALUDE_restaurant_hamburgers_l1933_193322

/-- The number of hamburgers served by the restaurant -/
def hamburgers_served : ℕ := 3

/-- The number of hamburgers left over -/
def hamburgers_leftover : ℕ := 6

/-- The total number of hamburgers made by the restaurant -/
def total_hamburgers : ℕ := hamburgers_served + hamburgers_leftover

/-- Theorem stating that the total number of hamburgers is 9 -/
theorem restaurant_hamburgers : total_hamburgers = 9 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_hamburgers_l1933_193322


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1933_193378

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    5 * x - 3 = (x^2 - 3*x - 18) * (C / (x - 6) + D / (x + 3))) →
  C = 3 ∧ D = 2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1933_193378


namespace NUMINAMATH_CALUDE_cosine_inequality_l1933_193354

theorem cosine_inequality (y : Real) :
  y ∈ Set.Icc 0 Real.pi →
  (∀ x : Real, Real.cos (x + y) ≤ Real.cos x * Real.cos y) ↔
  (y = 0 ∨ y = Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l1933_193354


namespace NUMINAMATH_CALUDE_emma_age_theorem_l1933_193352

def emma_age_problem (emma_current_age : ℕ) (age_difference : ℕ) (sister_future_age : ℕ) : Prop :=
  let sister_current_age := emma_current_age + age_difference
  let years_passed := sister_future_age - sister_current_age
  emma_current_age + years_passed = 47

theorem emma_age_theorem :
  emma_age_problem 7 9 56 :=
by
  sorry

end NUMINAMATH_CALUDE_emma_age_theorem_l1933_193352


namespace NUMINAMATH_CALUDE_john_tax_difference_l1933_193320

/-- Calculates the difference in taxes paid given old and new tax rates and incomes -/
def tax_difference (old_rate new_rate old_income new_income : ℚ) : ℚ :=
  new_rate * new_income - old_rate * old_income

/-- Proves that the difference in taxes paid by John is $250,000 -/
theorem john_tax_difference :
  tax_difference (20 / 100) (30 / 100) 1000000 1500000 = 250000 := by
  sorry

end NUMINAMATH_CALUDE_john_tax_difference_l1933_193320


namespace NUMINAMATH_CALUDE_tom_trade_amount_l1933_193321

/-- The amount Tom initially gave to trade his Super Nintendo for an NES -/
def trade_amount (super_nintendo_value : ℝ) (credit_percentage : ℝ) 
  (nes_price : ℝ) (game_value : ℝ) (change : ℝ) : ℝ :=
  let credit := super_nintendo_value * credit_percentage
  let remaining := nes_price - credit
  let total_needed := remaining + game_value
  total_needed + change

/-- Theorem stating the amount Tom initially gave -/
theorem tom_trade_amount : 
  trade_amount 150 0.8 160 30 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tom_trade_amount_l1933_193321


namespace NUMINAMATH_CALUDE_solution_to_inequality_l1933_193312

theorem solution_to_inequality : 1 - 1 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_solution_to_inequality_l1933_193312


namespace NUMINAMATH_CALUDE_dvd_discount_amount_l1933_193353

/-- The discount on a pack of DVDs -/
def discount (original_price discounted_price : ℝ) : ℝ :=
  original_price - discounted_price

/-- Theorem: The discount on each pack of DVDs is 25 dollars -/
theorem dvd_discount_amount : discount 76 51 = 25 := by
  sorry

end NUMINAMATH_CALUDE_dvd_discount_amount_l1933_193353


namespace NUMINAMATH_CALUDE_root_product_sum_l1933_193392

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 2021 * x^3 - 4043 * x^2 + x + 2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 2 * Real.sqrt 2021 :=
by sorry

end NUMINAMATH_CALUDE_root_product_sum_l1933_193392


namespace NUMINAMATH_CALUDE_toms_lifting_capacity_l1933_193343

/-- Calculates the total weight Tom can lift after training -/
def totalWeightAfterTraining (initialCapacity : ℝ) : ℝ :=
  let afterIntensiveTraining := initialCapacity * (1 + 1.5)
  let afterSpecialization := afterIntensiveTraining * (1 + 0.25)
  let afterNewGripTechnique := afterSpecialization * (1 + 0.1)
  2 * afterNewGripTechnique

/-- Theorem stating that Tom's final lifting capacity is 687.5 kg -/
theorem toms_lifting_capacity :
  totalWeightAfterTraining 100 = 687.5 := by
  sorry

#eval totalWeightAfterTraining 100

end NUMINAMATH_CALUDE_toms_lifting_capacity_l1933_193343


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1933_193300

/-- A regular polygon with an exterior angle of 18° has 20 sides -/
theorem regular_polygon_sides (n : ℕ) (ext_angle : ℝ) : 
  ext_angle = 18 → n * ext_angle = 360 → n = 20 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1933_193300


namespace NUMINAMATH_CALUDE_greater_eighteen_league_games_l1933_193305

/-- Represents a hockey league with the given specifications -/
structure HockeyLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games in the hockey league -/
def total_games (league : HockeyLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games + 
                        (total_teams - league.teams_per_division) * league.inter_division_games
  total_teams * games_per_team / 2

/-- Theorem stating that the total number of games in the specified league is 351 -/
theorem greater_eighteen_league_games : 
  total_games { divisions := 3
              , teams_per_division := 6
              , intra_division_games := 3
              , inter_division_games := 2 } = 351 := by
  sorry

end NUMINAMATH_CALUDE_greater_eighteen_league_games_l1933_193305


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_glb_l1933_193361

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := by sorry

theorem min_value_is_glb : ∃ (seq : ℕ → ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  ((seq n)^2 + 9) / Real.sqrt ((seq n)^2 + 5) < 4 + ε := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_glb_l1933_193361


namespace NUMINAMATH_CALUDE_last_locker_opened_l1933_193328

/-- Represents the process of opening lockers in a hall -/
def openLockers (n : ℕ) : ℕ :=
  n - 2

/-- Theorem stating that the last locker opened is number 727 -/
theorem last_locker_opened (total_lockers : ℕ) (h : total_lockers = 729) :
  openLockers total_lockers = 727 :=
by sorry

end NUMINAMATH_CALUDE_last_locker_opened_l1933_193328


namespace NUMINAMATH_CALUDE_major_selection_ways_l1933_193331

-- Define the total number of majors
def total_majors : ℕ := 10

-- Define the number of majors to be chosen
def chosen_majors : ℕ := 3

-- Define the number of incompatible majors
def incompatible_majors : ℕ := 2

-- Theorem statement
theorem major_selection_ways :
  (total_majors.choose chosen_majors) - 
  ((total_majors - incompatible_majors).choose (chosen_majors - 1)) = 672 := by
  sorry

end NUMINAMATH_CALUDE_major_selection_ways_l1933_193331


namespace NUMINAMATH_CALUDE_quotient_less_than_dividend_l1933_193316

theorem quotient_less_than_dividend : 
  let a := (5 : ℚ) / 7
  let b := (5 : ℚ) / 4
  a / b < a :=
by sorry

end NUMINAMATH_CALUDE_quotient_less_than_dividend_l1933_193316


namespace NUMINAMATH_CALUDE_pen_retailer_profit_percentage_pen_retailer_profit_is_ten_percent_l1933_193342

/-- Calculates the profit percentage for a pen retailer. -/
theorem pen_retailer_profit_percentage 
  (num_pens_bought : ℕ) 
  (num_pens_price : ℕ) 
  (discount_percent : ℚ) : ℚ :=
  let cost_price := num_pens_price
  let market_price_per_pen := 1
  let selling_price_per_pen := market_price_per_pen * (1 - discount_percent / 100)
  let total_selling_price := num_pens_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- Proves that the profit percentage is 10% for the given scenario. -/
theorem pen_retailer_profit_is_ten_percent :
  pen_retailer_profit_percentage 40 36 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pen_retailer_profit_percentage_pen_retailer_profit_is_ten_percent_l1933_193342


namespace NUMINAMATH_CALUDE_obtain_11_from_1_l1933_193311

/-- Represents the allowed operations on the calculator -/
inductive Operation
  | Multiply3
  | Add3
  | Divide3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Multiply3 => n * 3
  | Operation.Add3 => n + 3
  | Operation.Divide3 => if n % 3 = 0 then n / 3 else n

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem: It's possible to obtain 11 from 1 using the given calculator operations -/
theorem obtain_11_from_1 : ∃ (ops : List Operation), applyOperations 1 ops = 11 := by
  sorry

end NUMINAMATH_CALUDE_obtain_11_from_1_l1933_193311


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_max_sum_achieved_l1933_193376

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7/3 := by
sorry

theorem max_sum_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ (x y : ℝ), 4 * x + 3 * y ≤ 9 ∧ 2 * x + 4 * y ≤ 8 ∧ x + y > 7/3 - ε := by
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_max_sum_achieved_l1933_193376


namespace NUMINAMATH_CALUDE_bobs_number_l1933_193360

theorem bobs_number (alice bob : ℂ) : 
  alice * bob = 48 - 16 * I ∧ alice = 7 + 4 * I → 
  bob = 272/65 - 304/65 * I := by
sorry

end NUMINAMATH_CALUDE_bobs_number_l1933_193360


namespace NUMINAMATH_CALUDE_total_vitamins_in_box_vitamins_in_half_bag_l1933_193324

-- Define the number of bags in a box
def bags_per_box : ℕ := 9

-- Define the grams of vitamins per bag
def vitamins_per_bag : ℝ := 0.2

-- Theorem for total vitamins in a box
theorem total_vitamins_in_box :
  bags_per_box * vitamins_per_bag = 1.8 := by sorry

-- Theorem for vitamins in half a bag
theorem vitamins_in_half_bag :
  vitamins_per_bag / 2 = 0.1 := by sorry

end NUMINAMATH_CALUDE_total_vitamins_in_box_vitamins_in_half_bag_l1933_193324


namespace NUMINAMATH_CALUDE_remainder_of_division_l1933_193318

theorem remainder_of_division (n : ℕ) : 
  (2^224 + 104) % (2^112 + 2^56 + 1) = 103 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_division_l1933_193318


namespace NUMINAMATH_CALUDE_car_journey_metrics_l1933_193302

/-- Represents the car's journey with given conditions -/
structure CarJourney where
  total_distance : ℝ
  acc_dec_distance : ℝ
  constant_speed_distance : ℝ
  acc_dec_time : ℝ
  reference_speed : ℝ
  reference_time : ℝ

/-- Calculates the acceleration rate, deceleration rate, and highest speed of the car -/
def calculate_car_metrics (journey : CarJourney) : 
  (ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct acceleration, deceleration, and highest speed -/
theorem car_journey_metrics :
  let journey : CarJourney := {
    total_distance := 100,
    acc_dec_distance := 1,
    constant_speed_distance := 98,
    acc_dec_time := 100,
    reference_speed := 40 / 3600,
    reference_time := 90
  }
  let (acc_rate, dec_rate, highest_speed) := calculate_car_metrics journey
  acc_rate = 0.0002 ∧ dec_rate = 0.0002 ∧ highest_speed = 72 / 3600 :=
by sorry

end NUMINAMATH_CALUDE_car_journey_metrics_l1933_193302


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l1933_193394

/-- Represents the composition of a chemical solution -/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions -/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

/-- The problem statement -/
theorem chemical_mixture_problem (x y : Solution) (m : Mixture) :
  x.a = 0.1 ∧                 -- Solution x is 10% chemical a
  x.b = 0.9 ∧                 -- Solution x is 90% chemical b
  y.b = 0.8 ∧                 -- Solution y is 80% chemical b
  m.x = x ∧                   -- Mixture contains solution x
  m.y = y ∧                   -- Mixture contains solution y
  m.x_ratio = 0.8 ∧           -- 80% of the mixture is solution x
  m.x_ratio * x.a + (1 - m.x_ratio) * y.a = 0.12  -- Mixture is 12% chemical a
  →
  y.a = 0.2                   -- Percentage of chemical a in solution y is 20%
  := by sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l1933_193394


namespace NUMINAMATH_CALUDE_sum_product_bounds_l1933_193306

theorem sum_product_bounds (a b c d : ℝ) (h : a + b + c + d = 1) :
  0 ≤ a * b + a * c + a * d + b * c + b * d + c * d ∧
  a * b + a * c + a * d + b * c + b * d + c * d ≤ 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l1933_193306


namespace NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l1933_193313

theorem complex_function_minimum_on_unit_circle
  (a : ℝ) (ha : 0 < a ∧ a < 1)
  (f : ℂ → ℂ) (hf : ∀ z, f z = z^2 - z + a) :
  ∀ z : ℂ, 1 ≤ Complex.abs z →
  ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs (f z₀) ≤ Complex.abs (f z) :=
by sorry

end NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l1933_193313


namespace NUMINAMATH_CALUDE_yoo_jeong_borrowed_nine_notebooks_l1933_193369

/-- The number of notebooks Min-young originally had -/
def original_notebooks : ℕ := 17

/-- The number of notebooks Min-young had left after lending -/
def remaining_notebooks : ℕ := 8

/-- The number of notebooks Yoo-jeong borrowed -/
def borrowed_notebooks : ℕ := original_notebooks - remaining_notebooks

theorem yoo_jeong_borrowed_nine_notebooks : borrowed_notebooks = 9 := by
  sorry

end NUMINAMATH_CALUDE_yoo_jeong_borrowed_nine_notebooks_l1933_193369


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l1933_193317

/-- Given the prices and quantities of blankets, proves that the unknown rate is 285 when the average price is 162. -/
theorem unknown_blanket_rate (price1 price2 avg_price : ℚ) (qty1 qty2 qty_unknown : ℕ) : 
  price1 = 100 →
  price2 = 150 →
  qty1 = 3 →
  qty2 = 5 →
  qty_unknown = 2 →
  avg_price = 162 →
  (qty1 * price1 + qty2 * price2 + qty_unknown * (avg_price * (qty1 + qty2 + qty_unknown) - qty1 * price1 - qty2 * price2) / qty_unknown) / (qty1 + qty2 + qty_unknown) = avg_price →
  (avg_price * (qty1 + qty2 + qty_unknown) - qty1 * price1 - qty2 * price2) / qty_unknown = 285 := by
  sorry

#check unknown_blanket_rate

end NUMINAMATH_CALUDE_unknown_blanket_rate_l1933_193317


namespace NUMINAMATH_CALUDE_justin_reading_problem_l1933_193380

/-- Justin's reading problem -/
theorem justin_reading_problem (pages_first_day : ℕ) (remaining_days : ℕ) :
  pages_first_day = 10 →
  remaining_days = 6 →
  pages_first_day + remaining_days * (2 * pages_first_day) = 130 :=
by
  sorry

end NUMINAMATH_CALUDE_justin_reading_problem_l1933_193380


namespace NUMINAMATH_CALUDE_square_of_negative_two_i_l1933_193396

theorem square_of_negative_two_i (i : ℂ) (hi : i^2 = -1) : (-2 * i)^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_i_l1933_193396


namespace NUMINAMATH_CALUDE_convex_pentagon_exists_l1933_193323

-- Define the points
variable (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ : ℝ × ℝ)

-- Define the square
def is_square (A₁ A₂ A₃ A₄ : ℝ × ℝ) : Prop := sorry

-- Define the convex quadrilateral
def is_convex_quadrilateral (A₅ A₆ A₇ A₈ : ℝ × ℝ) : Prop := sorry

-- Define the point inside the quadrilateral
def point_inside_quadrilateral (A₉ A₅ A₆ A₇ A₈ : ℝ × ℝ) : Prop := sorry

-- Define the non-collinearity condition
def no_three_collinear (points : List (ℝ × ℝ)) : Prop := sorry

-- Define a convex pentagon
def is_convex_pentagon (points : List (ℝ × ℝ)) : Prop := sorry

theorem convex_pentagon_exists 
  (h1 : is_square A₁ A₂ A₃ A₄)
  (h2 : is_convex_quadrilateral A₅ A₆ A₇ A₈)
  (h3 : point_inside_quadrilateral A₉ A₅ A₆ A₇ A₈)
  (h4 : no_three_collinear [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉]) :
  ∃ (points : List (ℝ × ℝ)), points.length = 5 ∧ 
    (∀ p ∈ points, p ∈ [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉]) ∧ 
    is_convex_pentagon points :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_exists_l1933_193323


namespace NUMINAMATH_CALUDE_max_principals_is_five_l1933_193368

/-- Represents the duration of a principal's term in years -/
def term_length : ℕ := 3

/-- Represents the total period in years we're considering -/
def total_period : ℕ := 10

/-- Calculates the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 
  (total_period / term_length) + 
  (if total_period % term_length > 0 then 2 else 1)

/-- Theorem stating the maximum number of principals during the given period -/
theorem max_principals_is_five : max_principals = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_five_l1933_193368


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l1933_193339

theorem multiplication_addition_equality : 21 * 47 + 21 * 53 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l1933_193339


namespace NUMINAMATH_CALUDE_function_positive_l1933_193327

/-- Given a function f: ℝ → ℝ with derivative f', 
    if 2f(x) + xf'(x) > x² for all x ∈ ℝ, 
    then f(x) > 0 for all x ∈ ℝ. -/
theorem function_positive 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * deriv f x > x^2) : 
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_l1933_193327


namespace NUMINAMATH_CALUDE_swap_result_l1933_193350

def swap_values (a b : ℕ) : ℕ × ℕ :=
  let t := a
  let a := b
  let b := t
  (a, b)

theorem swap_result : swap_values 3 2 = (2, 3) := by sorry

end NUMINAMATH_CALUDE_swap_result_l1933_193350


namespace NUMINAMATH_CALUDE_triangle_inequality_l1933_193386

theorem triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1933_193386
