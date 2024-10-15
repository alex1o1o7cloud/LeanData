import Mathlib

namespace NUMINAMATH_CALUDE_color_film_fraction_l1813_181348

/-- Given a film festival selection process, prove that the fraction of selected films that are in color is 30/31. -/
theorem color_film_fraction (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw : ℚ := 20 * x
  let total_color : ℚ := 6 * y
  let selected_bw : ℚ := (y / x) * total_bw / 100
  let selected_color : ℚ := total_color
  (selected_color) / (selected_bw + selected_color) = 30 / 31 := by
sorry


end NUMINAMATH_CALUDE_color_film_fraction_l1813_181348


namespace NUMINAMATH_CALUDE_repunit_243_divisible_by_243_l1813_181375

/-- The number formed by n consecutive ones -/
def repunit (n : ℕ) : ℕ := (10^n - 1) / 9

/-- Theorem: The repunit of 243 is divisible by 243 -/
theorem repunit_243_divisible_by_243 : 243 ∣ repunit 243 := by
  sorry

end NUMINAMATH_CALUDE_repunit_243_divisible_by_243_l1813_181375


namespace NUMINAMATH_CALUDE_sum_of_roots_is_18_l1813_181322

/-- A function f that satisfies f(3 + x) = f(3 - x) for all real x -/
def symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

/-- The proposition that f has exactly 6 distinct real roots -/
def has_6_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
     r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
     r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
     r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
     r₅ ≠ r₆)

theorem sum_of_roots_is_18 (f : ℝ → ℝ) 
  (h₁ : symmetric_about_3 f) (h₂ : has_6_distinct_roots f) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), 
    (∀ x : ℝ, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_18_l1813_181322


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l1813_181354

/-- Represents the number of samples to be drawn from a stratum in stratified sampling -/
def stratumSample (totalSample : ℕ) (stratumRatio : ℕ) (totalRatio : ℕ) : ℕ :=
  totalSample * stratumRatio / totalRatio

/-- Theorem: In a stratified sampling with a total sample size of 200 and a population ratio of 5:2:3
    for three strata, the number of samples to be drawn from the third stratum is 60 -/
theorem third_grade_sample_size :
  let totalSample : ℕ := 200
  let firstRatio : ℕ := 5
  let secondRatio : ℕ := 2
  let thirdRatio : ℕ := 3
  let totalRatio : ℕ := firstRatio + secondRatio + thirdRatio
  stratumSample totalSample thirdRatio totalRatio = 60 := by
  sorry


end NUMINAMATH_CALUDE_third_grade_sample_size_l1813_181354


namespace NUMINAMATH_CALUDE_suv_max_distance_l1813_181306

-- Define the parameters
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def total_gallons : ℝ := 20
def highway_city_split : ℝ := 0.5  -- Equal split between highway and city

-- Theorem statement
theorem suv_max_distance :
  let highway_distance := highway_mpg * (highway_city_split * total_gallons)
  let city_distance := city_mpg * (highway_city_split * total_gallons)
  let max_distance := highway_distance + city_distance
  max_distance = 198 := by
  sorry

end NUMINAMATH_CALUDE_suv_max_distance_l1813_181306


namespace NUMINAMATH_CALUDE_square_area_error_l1813_181382

theorem square_area_error (side_error : Real) (area_error : Real) : 
  side_error = 0.17 → area_error = 36.89 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1813_181382


namespace NUMINAMATH_CALUDE_football_league_selection_l1813_181389

def division_A : ℕ := 12
def division_B : ℕ := 8
def teams_to_select : ℕ := 5

theorem football_league_selection :
  -- Part 1
  (Nat.choose (division_A + division_B - 2) (teams_to_select - 1) = 3060) ∧
  -- Part 2
  (Nat.choose (division_A + division_B) teams_to_select -
   Nat.choose division_A teams_to_select -
   Nat.choose division_B teams_to_select = 14656) := by
  sorry

end NUMINAMATH_CALUDE_football_league_selection_l1813_181389


namespace NUMINAMATH_CALUDE_cake_muffin_mix_buyers_l1813_181370

theorem cake_muffin_mix_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) : 
  total = 100 → cake = 50 → muffin = 40 → neither_prob = 29/100 → 
  ∃ both : ℕ, both = 19 ∧ 
    (total - (cake + muffin - both)) / total = neither_prob :=
by sorry

end NUMINAMATH_CALUDE_cake_muffin_mix_buyers_l1813_181370


namespace NUMINAMATH_CALUDE_minimum_score_condition_l1813_181369

/-- Represents a knowledge competition with specific scoring rules. -/
structure KnowledgeCompetition where
  totalQuestions : Nat
  correctPoints : Int
  incorrectDeduction : Int
  minimumScore : Int

/-- Calculates the score based on the number of correct answers. -/
def calculateScore (comp : KnowledgeCompetition) (correctAnswers : Nat) : Int :=
  comp.correctPoints * correctAnswers - comp.incorrectDeduction * (comp.totalQuestions - correctAnswers)

/-- Theorem stating the condition to achieve the minimum score. -/
theorem minimum_score_condition (comp : KnowledgeCompetition) 
  (h1 : comp.totalQuestions = 20)
  (h2 : comp.correctPoints = 5)
  (h3 : comp.incorrectDeduction = 1)
  (h4 : comp.minimumScore = 88) :
  ∃ x : Nat, calculateScore comp x ≥ comp.minimumScore ∧ 
    ∀ y : Nat, y < x → calculateScore comp y < comp.minimumScore :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_score_condition_l1813_181369


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_169_l1813_181319

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_169_l1813_181319


namespace NUMINAMATH_CALUDE_modulo_seven_residue_l1813_181331

theorem modulo_seven_residue : (312 + 6 * 51 + 8 * 175 + 3 * 28) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_seven_residue_l1813_181331


namespace NUMINAMATH_CALUDE_inner_circle_radius_is_one_l1813_181347

/-- A square with side length 4 containing 16 tangent semicircles along its perimeter --/
structure TangentCirclesSquare where
  side_length : ℝ
  num_semicircles : ℕ
  semicircle_radius : ℝ
  h_side_length : side_length = 4
  h_num_semicircles : num_semicircles = 16
  h_semicircle_radius : semicircle_radius = 1

/-- The radius of a circle tangent to all semicircles in a TangentCirclesSquare --/
def inner_circle_radius (s : TangentCirclesSquare) : ℝ := 1

/-- Theorem stating that the radius of the inner tangent circle is 1 --/
theorem inner_circle_radius_is_one (s : TangentCirclesSquare) :
  inner_circle_radius s = 1 := by sorry

end NUMINAMATH_CALUDE_inner_circle_radius_is_one_l1813_181347


namespace NUMINAMATH_CALUDE_g_of_3_equals_209_l1813_181399

-- Define the function g
def g (x : ℝ) : ℝ := 9 * x^3 - 4 * x^2 + 3 * x - 7

-- Theorem statement
theorem g_of_3_equals_209 : g 3 = 209 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_209_l1813_181399


namespace NUMINAMATH_CALUDE_third_iteration_interval_l1813_181350

def bisection_interval (a b : ℝ) (n : ℕ) : Set (ℝ × ℝ) :=
  match n with
  | 0 => {(a, b)}
  | n+1 => let m := (a + b) / 2
           (bisection_interval a m n) ∪ (bisection_interval m b n)

theorem third_iteration_interval (a b : ℝ) (h : (a, b) = (-2, 4)) :
  (-1/2, 1) ∈ bisection_interval a b 3 :=
sorry

end NUMINAMATH_CALUDE_third_iteration_interval_l1813_181350


namespace NUMINAMATH_CALUDE_random_selection_properties_l1813_181384

/-- Represents the number of female students chosen -/
inductive FemaleCount : Type
  | zero : FemaleCount
  | one : FemaleCount
  | two : FemaleCount

/-- The probability distribution of choosing female students -/
def prob_dist : FemaleCount → ℚ
  | FemaleCount.zero => 1/5
  | FemaleCount.one => 3/5
  | FemaleCount.two => 1/5

/-- The expected value of the number of female students chosen -/
def expected_value : ℚ := 1

/-- The probability of choosing at most one female student -/
def prob_at_most_one : ℚ := 4/5

/-- Theorem stating the properties of the random selection -/
theorem random_selection_properties 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (chosen_count : ℕ) 
  (h1 : male_count = 4) 
  (h2 : female_count = 2) 
  (h3 : chosen_count = 3) :
  (∀ x : FemaleCount, prob_dist x = prob_dist x) ∧
  expected_value = 1 ∧
  prob_at_most_one = 4/5 := by
  sorry

#check random_selection_properties

end NUMINAMATH_CALUDE_random_selection_properties_l1813_181384


namespace NUMINAMATH_CALUDE_probability_seven_heads_in_ten_flips_l1813_181367

-- Define the number of coins and the number of heads
def n : ℕ := 10
def k : ℕ := 7

-- Define the probability of getting heads on a single flip
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- State the theorem
theorem probability_seven_heads_in_ten_flips :
  (binomial_coeff n k : ℚ) * p^n = 15/128 := by sorry

end NUMINAMATH_CALUDE_probability_seven_heads_in_ten_flips_l1813_181367


namespace NUMINAMATH_CALUDE_c_share_is_54_l1813_181342

/-- Represents the rental arrangement for a pasture --/
structure PastureRental where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℕ

/-- Calculates the share of rent for person c given a PastureRental arrangement --/
def calculate_c_share (rental : PastureRental) : ℚ :=
  let total_ox_months := rental.a_oxen * rental.a_months + 
                         rental.b_oxen * rental.b_months + 
                         rental.c_oxen * rental.c_months
  let rent_per_ox_month : ℚ := rental.total_rent / total_ox_months
  (rental.c_oxen * rental.c_months : ℚ) * rent_per_ox_month

/-- The main theorem stating that c's share of the rent is 54 --/
theorem c_share_is_54 (rental : PastureRental) 
  (h1 : rental.a_oxen = 10) (h2 : rental.a_months = 7)
  (h3 : rental.b_oxen = 12) (h4 : rental.b_months = 5)
  (h5 : rental.c_oxen = 15) (h6 : rental.c_months = 3)
  (h7 : rental.total_rent = 210) : 
  calculate_c_share rental = 54 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_54_l1813_181342


namespace NUMINAMATH_CALUDE_find_y_value_l1813_181305

theorem find_y_value (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : x = 3) : y = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l1813_181305


namespace NUMINAMATH_CALUDE_bethany_portraits_l1813_181335

theorem bethany_portraits (total_paintings : ℕ) (still_life_ratio : ℕ) : 
  total_paintings = 200 → still_life_ratio = 6 →
  ∃ (portraits : ℕ), portraits = 28 ∧ 
    portraits * (still_life_ratio + 1) = total_paintings :=
by sorry

end NUMINAMATH_CALUDE_bethany_portraits_l1813_181335


namespace NUMINAMATH_CALUDE_largest_two_digit_number_with_conditions_l1813_181364

theorem largest_two_digit_number_with_conditions : ∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 3 = 0 ∧         -- divisible by 3
  n % 4 = 0 ∧         -- divisible by 4
  n % 5 = 4 ∧         -- remainder 4 when divided by 5
  ∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 4) → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_with_conditions_l1813_181364


namespace NUMINAMATH_CALUDE_tank_capacity_is_24_gallons_l1813_181314

/-- Represents the capacity of a tank and its contents over time -/
structure TankState where
  capacity : ℝ
  initialMixture : ℝ
  initialSodiumChloride : ℝ
  initialWater : ℝ
  evaporationRate : ℝ
  time : ℝ

/-- Calculates the final water volume after evaporation -/
def finalWaterVolume (state : TankState) : ℝ :=
  state.initialWater - state.evaporationRate * state.time

/-- Theorem stating the tank capacity given the conditions -/
theorem tank_capacity_is_24_gallons :
  ∀ (state : TankState),
    state.initialMixture = state.capacity / 4 →
    state.initialSodiumChloride = 0.3 * state.initialMixture →
    state.initialWater = 0.7 * state.initialMixture →
    state.evaporationRate = 0.4 →
    state.time = 6 →
    finalWaterVolume state = state.initialSodiumChloride →
    state.capacity = 24 := by
  sorry

#check tank_capacity_is_24_gallons

end NUMINAMATH_CALUDE_tank_capacity_is_24_gallons_l1813_181314


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l1813_181361

theorem pythagorean_triple_divisibility (x y z : ℤ) : 
  x^2 + y^2 = z^2 → ∃ k : ℤ, x * y = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l1813_181361


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1813_181321

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ z : ℂ, z = m * (m - 1) + (m - 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1813_181321


namespace NUMINAMATH_CALUDE_appropriate_citizen_actions_l1813_181366

/-- Represents the current state of the cultural market -/
structure CulturalMarket where
  entertainment_trend : Bool
  vulgarity_trend : Bool

/-- Represents possible actions citizens can take -/
inductive CitizenAction
  | choose_personality_trends
  | improve_distinction_ability
  | enhance_aesthetic_taste
  | pursue_high_end_culture

/-- Determines if an action is appropriate given the cultural market state -/
def is_appropriate_action (market : CulturalMarket) (action : CitizenAction) : Prop :=
  match action with
  | CitizenAction.improve_distinction_ability => true
  | CitizenAction.enhance_aesthetic_taste => true
  | _ => false

/-- Theorem stating the most appropriate actions for citizens -/
theorem appropriate_citizen_actions (market : CulturalMarket) 
  (h_entertainment : market.entertainment_trend = true)
  (h_vulgarity : market.vulgarity_trend = true) :
  (∀ action : CitizenAction, is_appropriate_action market action ↔ 
    (action = CitizenAction.improve_distinction_ability ∨ 
     action = CitizenAction.enhance_aesthetic_taste)) :=
by sorry

end NUMINAMATH_CALUDE_appropriate_citizen_actions_l1813_181366


namespace NUMINAMATH_CALUDE_tangent_line_intersection_f_increasing_inequality_proof_l1813_181301

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 / x + 1 - a

theorem tangent_line_intersection (a : ℝ) :
  (f' a (Real.exp 1)) * (Real.exp 1) = f a (Real.exp 1) - (2 - Real.exp 1) →
  a = 2 := by sorry

theorem f_increasing (a : ℝ) :
  a ≤ 2 →
  ∀ x > 0, f' a x ≥ 0 := by sorry

theorem inequality_proof (x : ℝ) :
  1 < x → x < 2 →
  (2 / (x - 1)) > (1 / Real.log x - 1 / Real.log (2 - x)) := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_intersection_f_increasing_inequality_proof_l1813_181301


namespace NUMINAMATH_CALUDE_investment_percentage_l1813_181390

theorem investment_percentage (initial_investment : ℝ) (initial_rate : ℝ) 
  (additional_investment : ℝ) (additional_rate : ℝ) :
  initial_investment = 2000 →
  initial_rate = 0.05 →
  additional_investment = 999.9999999999998 →
  additional_rate = 0.08 →
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_investment_percentage_l1813_181390


namespace NUMINAMATH_CALUDE_inverse_g_90_l1813_181346

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_g_90 : g⁻¹ 90 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_90_l1813_181346


namespace NUMINAMATH_CALUDE_pencil_count_l1813_181333

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 5 →
  pencils = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1813_181333


namespace NUMINAMATH_CALUDE_people_per_column_l1813_181338

theorem people_per_column (P : ℕ) (X : ℕ) : 
  P = 30 * 16 → P = X * 40 → X = 12 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l1813_181338


namespace NUMINAMATH_CALUDE_sum_squares_inequality_l1813_181381

theorem sum_squares_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 5) : 
  a^2 + 2*b^2 + c^2 ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_l1813_181381


namespace NUMINAMATH_CALUDE_solve_ice_problem_l1813_181358

def ice_problem (ice_in_glass : ℕ) (num_trays : ℕ) : Prop :=
  let ice_in_pitcher : ℕ := 2 * ice_in_glass
  let total_ice : ℕ := ice_in_glass + ice_in_pitcher
  let spaces_per_tray : ℕ := total_ice / num_trays
  (ice_in_glass = 8) ∧ (num_trays = 2) → (spaces_per_tray = 12)

theorem solve_ice_problem : ice_problem 8 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_ice_problem_l1813_181358


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1813_181377

theorem unique_integer_solution : 
  ∃! (x : ℤ), (abs x : ℝ) < 5 * Real.pi ∧ x^2 - 4*x + 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1813_181377


namespace NUMINAMATH_CALUDE_constant_function_derivative_l1813_181311

-- Define the constant function f(x) = 0
def f : ℝ → ℝ := λ x ↦ 0

-- State the theorem
theorem constant_function_derivative :
  ∀ x : ℝ, deriv f x = 0 := by sorry

end NUMINAMATH_CALUDE_constant_function_derivative_l1813_181311


namespace NUMINAMATH_CALUDE_books_on_shelf_after_changes_l1813_181388

/-- The total number of books on the shelf after Marta's changes -/
def total_books_after_changes (initial_fiction : ℕ) (initial_nonfiction : ℕ) 
  (added_fiction : ℕ) (removed_nonfiction : ℕ) (added_sets : ℕ) (books_per_set : ℕ) : ℕ :=
  (initial_fiction + added_fiction) + 
  (initial_nonfiction - removed_nonfiction) + 
  (added_sets * books_per_set)

/-- Theorem stating that the total number of books after changes is 70 -/
theorem books_on_shelf_after_changes : 
  total_books_after_changes 38 15 10 5 3 4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_after_changes_l1813_181388


namespace NUMINAMATH_CALUDE_rhombus_area_l1813_181365

/-- Given a rhombus with side length √113 and diagonals differing by 10 units, 
    its area is 72 square units -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 113 →
  d₁ - d₂ = 10 →
  d₁ * d₂ = 4 * s^2 →
  (d₁ * d₂) / 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1813_181365


namespace NUMINAMATH_CALUDE_expression_equality_l1813_181334

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1/y^2) :
  (x^2 - 4/x^2) * (y^2 + 4/y^2) = x^4 - 16/x^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1813_181334


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l1813_181339

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℕ, x ≤ 16 ↔ 9 * x + 5 < 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l1813_181339


namespace NUMINAMATH_CALUDE_pattern_equality_l1813_181380

theorem pattern_equality (n : ℕ) (h : n ≥ 2) :
  Real.sqrt (n + n / (n^2 - 1)) = n * Real.sqrt (n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_pattern_equality_l1813_181380


namespace NUMINAMATH_CALUDE_gumball_theorem_l1813_181363

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  purple : Nat
  orange : Nat
  green : Nat
  yellow : Nat

/-- The minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  3 * 4 + 1

theorem gumball_theorem (machine : GumballMachine) 
  (h : machine = { purple := 12, orange := 6, green := 8, yellow := 5 }) : 
  minGumballsForFour machine = 13 := by
  sorry

#eval minGumballsForFour { purple := 12, orange := 6, green := 8, yellow := 5 }

end NUMINAMATH_CALUDE_gumball_theorem_l1813_181363


namespace NUMINAMATH_CALUDE_bobs_spending_l1813_181325

def notebook_price : ℝ := 2
def magazine_price : ℝ := 5
def book_price : ℝ := 15
def notebook_quantity : ℕ := 4
def magazine_quantity : ℕ := 3
def book_quantity : ℕ := 2
def book_discount : ℝ := 0.2
def coupon_value : ℝ := 10
def coupon_threshold : ℝ := 50

def total_spending : ℝ := 
  notebook_price * notebook_quantity +
  magazine_price * magazine_quantity +
  book_price * (1 - book_discount) * book_quantity

theorem bobs_spending (spending : ℝ) :
  spending = total_spending ∧ 
  spending < coupon_threshold →
  spending = 47 :=
by sorry

end NUMINAMATH_CALUDE_bobs_spending_l1813_181325


namespace NUMINAMATH_CALUDE_second_smallest_hotdog_pack_l1813_181397

def is_valid_hotdog_pack (n : ℕ) : Prop :=
  ∃ b : ℕ, 12 * n - 10 * b = 6 ∧ n % 5 = 3

theorem second_smallest_hotdog_pack :
  ∃ n : ℕ, is_valid_hotdog_pack n ∧
  (∀ m : ℕ, m < n → ¬is_valid_hotdog_pack m ∨ 
   (∃ k : ℕ, k < m ∧ is_valid_hotdog_pack k)) ∧
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_hotdog_pack_l1813_181397


namespace NUMINAMATH_CALUDE_months_passed_l1813_181309

/-- Represents the number of bones Barkley receives each month -/
def bones_per_month : ℕ := 10

/-- Represents the number of bones Barkley currently has available -/
def available_bones : ℕ := 8

/-- Represents the number of bones Barkley has buried -/
def buried_bones : ℕ := 42

/-- Calculates the total number of bones Barkley has received -/
def total_bones (months : ℕ) : ℕ := bones_per_month * months

/-- Theorem stating that 5 months have passed based on the given conditions -/
theorem months_passed :
  ∃ (months : ℕ), months = 5 ∧ total_bones months = available_bones + buried_bones :=
sorry

end NUMINAMATH_CALUDE_months_passed_l1813_181309


namespace NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l1813_181308

theorem power_of_seven_mod_thousand : 7^2023 % 1000 = 343 := by sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l1813_181308


namespace NUMINAMATH_CALUDE_cement_mixture_water_fraction_l1813_181330

theorem cement_mixture_water_fraction 
  (total_weight : ℝ) 
  (sand_fraction : ℝ) 
  (gravel_weight : ℝ) 
  (h1 : total_weight = 49.99999999999999)
  (h2 : sand_fraction = 1/2)
  (h3 : gravel_weight = 15) :
  (total_weight - sand_fraction * total_weight - gravel_weight) / total_weight = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_water_fraction_l1813_181330


namespace NUMINAMATH_CALUDE_outfits_count_l1813_181307

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of ties available -/
def num_ties : ℕ := 7

/-- Calculate the number of outfits possible -/
def num_outfits : ℕ := num_shirts * (num_ties + 1)

/-- Theorem stating that the number of outfits is 64 -/
theorem outfits_count : num_outfits = 64 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1813_181307


namespace NUMINAMATH_CALUDE_reading_time_proof_l1813_181385

/-- The number of days it takes to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Theorem: It takes 12 days to read a 240-page book at 20 pages per day -/
theorem reading_time_proof : days_to_read 240 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_proof_l1813_181385


namespace NUMINAMATH_CALUDE_expression_equals_one_half_l1813_181313

theorem expression_equals_one_half :
  (4 * 6) / (12 * 8) * (5 * 12 * 8) / (4 * 5 * 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_half_l1813_181313


namespace NUMINAMATH_CALUDE_number_equation_l1813_181303

theorem number_equation (x : ℝ) : ((x - 8) - 12) / 5 = 7 ↔ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1813_181303


namespace NUMINAMATH_CALUDE_relationship_abc_l1813_181327

theorem relationship_abc : 
  let a := (1/2) * Real.cos (2 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (2 * Real.pi / 180)
  let b := (2 * Real.tan (14 * Real.pi / 180)) / (1 - Real.tan (14 * Real.pi / 180)^2)
  let c := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)
  c < a ∧ a < b :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l1813_181327


namespace NUMINAMATH_CALUDE_postcard_cost_l1813_181310

theorem postcard_cost (cost : ℕ) : cost = 111 :=
  by
  have h1 : 9 * cost < 1000 := sorry
  have h2 : 10 * cost > 1100 := sorry
  have h3 : cost > 0 := sorry
  sorry

end NUMINAMATH_CALUDE_postcard_cost_l1813_181310


namespace NUMINAMATH_CALUDE_expected_monthly_profit_l1813_181337

/-- Represents the color of a ball -/
inductive BallColor
| Yellow
| White

/-- Represents a ball with its color and label -/
structure Ball :=
  (color : BallColor)
  (label : Char)

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of people drawing per day -/
def daily_draws : ℕ := 100

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The reward for drawing 3 balls of the same color -/
def same_color_reward : ℚ := 5

/-- The cost for drawing 3 balls of different colors -/
def diff_color_cost : ℚ := 1

/-- The probability of drawing 3 balls of the same color -/
def prob_same_color : ℚ := 2 / 20

/-- The probability of drawing 3 balls of different colors -/
def prob_diff_color : ℚ := 18 / 20

/-- The expected daily profit for the stall owner -/
def expected_daily_profit : ℚ :=
  daily_draws * (prob_diff_color * diff_color_cost - prob_same_color * same_color_reward)

/-- Theorem: The expected monthly profit for the stall owner is $1200 -/
theorem expected_monthly_profit :
  expected_daily_profit * days_in_month = 1200 := by sorry

end NUMINAMATH_CALUDE_expected_monthly_profit_l1813_181337


namespace NUMINAMATH_CALUDE_three_number_problem_l1813_181323

theorem three_number_problem (a b c : ℝ) :
  a + b + c = 114 →
  b^2 = a * c →
  ∃ d : ℝ, b = a + 3*d ∧ c = a + 24*d →
  ((a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98)) :=
by sorry

end NUMINAMATH_CALUDE_three_number_problem_l1813_181323


namespace NUMINAMATH_CALUDE_triangle_ABC_area_l1813_181357

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (8, 2)
def C : ℝ × ℝ := (6, -1)

-- Define the function to calculate the area of a triangle given its vertices
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_ABC_area :
  triangleArea A B C = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_area_l1813_181357


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1813_181362

theorem unique_solution_floor_equation :
  ∃! b : ℝ, b + ⌊b⌋ = 14.3 ∧ b = 7.3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1813_181362


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1813_181315

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (π - 2) * Real.cos (π - 2)) = Real.sin 2 + Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1813_181315


namespace NUMINAMATH_CALUDE_lizette_stamp_count_l1813_181392

/-- The number of stamps Minerva has -/
def minerva_stamps : ℕ := 688

/-- The number of additional stamps Lizette has compared to Minerva -/
def additional_stamps : ℕ := 125

/-- The total number of stamps Lizette has -/
def lizette_stamps : ℕ := minerva_stamps + additional_stamps

theorem lizette_stamp_count : lizette_stamps = 813 := by
  sorry

end NUMINAMATH_CALUDE_lizette_stamp_count_l1813_181392


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1813_181326

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 3) (h2 : b = 8) (h3 : Odd c) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  a + b + c = 18 ∨ a + b + c = 20 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1813_181326


namespace NUMINAMATH_CALUDE_tourist_groups_speed_l1813_181393

theorem tourist_groups_speed : ∀ (x y : ℝ),
  (x > 0 ∧ y > 0) →  -- Speeds are positive
  (4.5 * x + 2.5 * y = 30) →  -- First scenario equation
  (3 * x + 5 * y = 30) →  -- Second scenario equation
  (x = 5 ∧ y = 3) :=  -- Speeds of the two groups
by sorry

end NUMINAMATH_CALUDE_tourist_groups_speed_l1813_181393


namespace NUMINAMATH_CALUDE_total_coins_count_l1813_181395

theorem total_coins_count (dimes nickels quarters : ℕ) : 
  dimes = 2 → nickels = 2 → quarters = 7 → dimes + nickels + quarters = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_count_l1813_181395


namespace NUMINAMATH_CALUDE_constant_zero_is_arithmetic_not_geometric_l1813_181312

def constant_zero_sequence : ℕ → ℝ := fun _ ↦ 0

theorem constant_zero_is_arithmetic_not_geometric :
  (∃ d : ℝ, ∀ n : ℕ, constant_zero_sequence (n + 1) = constant_zero_sequence n + d) ∧
  (¬ ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, constant_zero_sequence (n + 1) = constant_zero_sequence n * r) :=
by sorry

end NUMINAMATH_CALUDE_constant_zero_is_arithmetic_not_geometric_l1813_181312


namespace NUMINAMATH_CALUDE_parallel_lines_x_value_l1813_181345

/-- A line passing through two points -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- Check if a line is vertical -/
def Line.isVertical (l : Line) : Prop :=
  l.x₁ = l.x₂

/-- Two lines are parallel if they are both vertical or have the same slope -/
def parallelLines (l₁ l₂ : Line) : Prop :=
  (l₁.isVertical ∧ l₂.isVertical) ∨
  (¬l₁.isVertical ∧ ¬l₂.isVertical ∧ 
   (l₁.y₂ - l₁.y₁) / (l₁.x₂ - l₁.x₁) = (l₂.y₂ - l₂.y₁) / (l₂.x₂ - l₂.x₁))

theorem parallel_lines_x_value :
  ∀ (x : ℝ),
  let l₁ : Line := ⟨-1, -2, -1, 4⟩
  let l₂ : Line := ⟨2, 1, x, 6⟩
  parallelLines l₁ l₂ → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_x_value_l1813_181345


namespace NUMINAMATH_CALUDE_choir_third_group_members_l1813_181304

/-- Represents a choir split into three groups -/
structure Choir :=
  (total_members : ℕ)
  (group1_members : ℕ)
  (group2_members : ℕ)
  (group3_members : ℕ)

/-- Theorem stating the number of members in the third group of the choir -/
theorem choir_third_group_members (c : Choir) 
  (h1 : c.total_members = 70)
  (h2 : c.group1_members = 25)
  (h3 : c.group2_members = 30)
  (h4 : c.group3_members = c.total_members - c.group1_members - c.group2_members) :
  c.group3_members = 15 := by
  sorry

end NUMINAMATH_CALUDE_choir_third_group_members_l1813_181304


namespace NUMINAMATH_CALUDE_private_teacher_cost_l1813_181341

/-- Calculates the amount each parent must pay for a private teacher --/
theorem private_teacher_cost 
  (former_salary : ℕ) 
  (raise_percentage : ℚ) 
  (num_kids : ℕ) 
  (h1 : former_salary = 45000)
  (h2 : raise_percentage = 1/5)
  (h3 : num_kids = 9) : 
  (former_salary * (1 + raise_percentage)) / num_kids = 6000 := by
  sorry

end NUMINAMATH_CALUDE_private_teacher_cost_l1813_181341


namespace NUMINAMATH_CALUDE_parabola_vertex_l1813_181387

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 3 * (x - 1)^2 + 8

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 8)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 8 is (1, 8) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1813_181387


namespace NUMINAMATH_CALUDE_find_divisor_l1813_181320

theorem find_divisor (n m d : ℕ) (h1 : n = 2304) (h2 : m = 2319) 
  (h3 : m > n) (h4 : m % d = 0) 
  (h5 : ∀ k, n < k ∧ k < m → k % d ≠ 0) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1813_181320


namespace NUMINAMATH_CALUDE_total_teaching_time_is_5160_l1813_181318

-- Define the number of classes and durations for Eduardo
def eduardo_math_classes : ℕ := 3
def eduardo_science_classes : ℕ := 4
def eduardo_history_classes : ℕ := 2
def eduardo_math_duration : ℕ := 60
def eduardo_science_duration : ℕ := 90
def eduardo_history_duration : ℕ := 120

-- Define Frankie's multiplier
def frankie_multiplier : ℕ := 2

-- Define Georgina's multiplier and class durations
def georgina_multiplier : ℕ := 3
def georgina_math_duration : ℕ := 80
def georgina_science_duration : ℕ := 100
def georgina_history_duration : ℕ := 150

-- Calculate total teaching time
def total_teaching_time : ℕ :=
  -- Eduardo's teaching time
  (eduardo_math_classes * eduardo_math_duration +
   eduardo_science_classes * eduardo_science_duration +
   eduardo_history_classes * eduardo_history_duration) +
  -- Frankie's teaching time
  (frankie_multiplier * eduardo_math_classes * eduardo_math_duration +
   frankie_multiplier * eduardo_science_classes * eduardo_science_duration +
   frankie_multiplier * eduardo_history_classes * eduardo_history_duration) +
  -- Georgina's teaching time
  (georgina_multiplier * eduardo_math_classes * georgina_math_duration +
   georgina_multiplier * eduardo_science_classes * georgina_science_duration +
   georgina_multiplier * eduardo_history_classes * georgina_history_duration)

-- Theorem statement
theorem total_teaching_time_is_5160 : total_teaching_time = 5160 := by
  sorry

end NUMINAMATH_CALUDE_total_teaching_time_is_5160_l1813_181318


namespace NUMINAMATH_CALUDE_ceiling_plus_self_eq_150_l1813_181336

theorem ceiling_plus_self_eq_150 :
  ∃ y : ℝ, ⌈y⌉ + y = 150 ∧ y = 75 := by sorry

end NUMINAMATH_CALUDE_ceiling_plus_self_eq_150_l1813_181336


namespace NUMINAMATH_CALUDE_tangent_line_property_l1813_181386

theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₁ ≠ 1) :
  (((1 : ℝ) / x₁ = Real.exp x₂) ∧
   (Real.log x₁ - 1 = Real.exp x₂ * (1 - x₂))) →
  2 / (x₁ - 1) + x₂ = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_property_l1813_181386


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1813_181372

/-- Given a hyperbola with equation (x²/a²) - (y²/b²) = 1, where a > 0 and b > 0,
    with eccentricity 2 and distance from focus to asymptote √3,
    prove that its focal length is 4. -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let d := Real.sqrt 3  -- distance from focus to asymptote
  let c := e * a  -- distance from center to focus
  let focal_length := 2 * c
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → 
    e = c / a ∧
    d = (b * c) / Real.sqrt (a^2 + b^2)) →
  focal_length = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1813_181372


namespace NUMINAMATH_CALUDE_cross_to_square_l1813_181351

/-- Represents a cross made of five equal squares -/
structure Cross where
  side_length : ℝ
  num_squares : Nat
  h_num_squares : num_squares = 5

/-- Represents the square formed by reassembling the cross parts -/
structure ReassembledSquare where
  side_length : ℝ

/-- States that a cross can be cut into parts that form a square -/
def can_form_square (c : Cross) (s : ReassembledSquare) : Prop :=
  s.side_length = c.side_length * Real.sqrt 5

/-- Theorem stating that a cross of five equal squares can be cut to form a square -/
theorem cross_to_square (c : Cross) :
  ∃ s : ReassembledSquare, can_form_square c s :=
sorry

end NUMINAMATH_CALUDE_cross_to_square_l1813_181351


namespace NUMINAMATH_CALUDE_lattice_paths_avoiding_point_l1813_181352

/-- Represents a point on the lattice -/
structure Point where
  x : Nat
  y : Nat

/-- Calculates the number of paths from (0,0) to a given point -/
def numPaths (p : Point) : Nat :=
  Nat.choose (p.x + p.y) p.x

/-- The theorem to be proved -/
theorem lattice_paths_avoiding_point :
  numPaths ⟨4, 4⟩ - numPaths ⟨2, 2⟩ * numPaths ⟨2, 2⟩ = 34 := by
  sorry

#eval numPaths ⟨4, 4⟩ - numPaths ⟨2, 2⟩ * numPaths ⟨2, 2⟩

end NUMINAMATH_CALUDE_lattice_paths_avoiding_point_l1813_181352


namespace NUMINAMATH_CALUDE_specific_tunnel_length_l1813_181368

/-- Calculates the length of a tunnel given train and travel parameters. -/
def tunnel_length (train_length : ℚ) (train_speed : ℚ) (exit_time : ℚ) : ℚ :=
  train_speed * exit_time / 60 - train_length

/-- Theorem stating the length of the tunnel given specific parameters. -/
theorem specific_tunnel_length :
  let train_length : ℚ := 2
  let train_speed : ℚ := 40
  let exit_time : ℚ := 5
  tunnel_length train_length train_speed exit_time = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_specific_tunnel_length_l1813_181368


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1813_181360

/-- Given a group of 8 people, prove that if replacing one person with a new person
    weighing 105 kg increases the average weight by 2.5 kg, then the weight of the
    replaced person is 85 kg. -/
theorem weight_of_replaced_person
  (initial_count : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h_initial_count : initial_count = 8)
  (h_weight_increase : weight_increase = 2.5)
  (h_new_person_weight : new_person_weight = 105)
  : ∃ (replaced_weight : ℝ),
    replaced_weight = 85 ∧
    (initial_count : ℝ) * weight_increase = new_person_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1813_181360


namespace NUMINAMATH_CALUDE_expression_equality_l1813_181398

theorem expression_equality : 
  |Real.sqrt 8 - 2| + (π - 2023)^(0 : ℝ) + (-1/2)^(-2 : ℝ) - 2 * Real.cos (60 * π / 180) = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1813_181398


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l1813_181329

theorem complex_magnitude_one (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l1813_181329


namespace NUMINAMATH_CALUDE_six_color_theorem_l1813_181349

-- Define a Map as a structure with countries and their adjacencies
structure Map where
  countries : Set (Nat)
  adjacent : countries → countries → Prop

-- Define a Coloring as a function from countries to colors
def Coloring (m : Map) := m.countries → Fin 6

-- Define what it means for a coloring to be proper
def IsProperColoring (m : Map) (c : Coloring m) : Prop :=
  ∀ x y : m.countries, m.adjacent x y → c x ≠ c y

-- State the theorem
theorem six_color_theorem (m : Map) : 
  ∃ c : Coloring m, IsProperColoring m c :=
sorry

end NUMINAMATH_CALUDE_six_color_theorem_l1813_181349


namespace NUMINAMATH_CALUDE_gcd_m_l1813_181302

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_m_l1813_181302


namespace NUMINAMATH_CALUDE_grandfathers_age_l1813_181353

theorem grandfathers_age : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 : ℕ) + (n % 10)^2 = n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_l1813_181353


namespace NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l1813_181373

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number satisfying the conditions -/
theorem exists_number_with_digit_sum_decrease : 
  ∃ (N : ℕ), (∃ (M : ℕ), M = (11 * N) / 10) ∧ 
  (sum_of_digits ((11 * N) / 10) = (9 * sum_of_digits N) / 10) := by sorry

end NUMINAMATH_CALUDE_exists_number_with_digit_sum_decrease_l1813_181373


namespace NUMINAMATH_CALUDE_valid_sequence_count_l1813_181328

/-- Represents the number of valid sequences of length n -/
def S (n : ℕ) : ℕ :=
  sorry

/-- Represents the number of valid sequences of length n ending with A -/
def A (n : ℕ) : ℕ :=
  sorry

theorem valid_sequence_count : S 2015 % 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_valid_sequence_count_l1813_181328


namespace NUMINAMATH_CALUDE_remainder_invariant_can_reach_43_from_3_cannot_reach_43_from_5_l1813_181324

/-- The set of allowed operations in Daniel's game -/
inductive GameOperation
  | AddFour    : GameOperation
  | MultiplyFour : GameOperation
  | Square     : GameOperation

/-- Apply a single game operation to a number -/
def applyOperation (n : ℤ) (op : GameOperation) : ℤ :=
  match op with
  | GameOperation.AddFour    => n + 4
  | GameOperation.MultiplyFour => n * 4
  | GameOperation.Square     => n * n

/-- Apply a sequence of game operations to a number -/
def applyOperations (n : ℤ) (ops : List GameOperation) : ℤ :=
  ops.foldl applyOperation n

/-- Proposition: Starting from a number with remainder 1 when divided by 4,
    any resulting number will have remainder 0 or 1 -/
theorem remainder_invariant (n : ℤ) (ops : List GameOperation) :
  n % 4 = 1 → (applyOperations n ops) % 4 = 0 ∨ (applyOperations n ops) % 4 = 1 :=
sorry

/-- Proposition: It's possible to obtain 43 from 3 using allowed operations -/
theorem can_reach_43_from_3 : ∃ (ops : List GameOperation), applyOperations 3 ops = 43 :=
sorry

/-- Proposition: It's impossible to obtain 43 from 5 using allowed operations -/
theorem cannot_reach_43_from_5 : ¬ ∃ (ops : List GameOperation), applyOperations 5 ops = 43 :=
sorry

end NUMINAMATH_CALUDE_remainder_invariant_can_reach_43_from_3_cannot_reach_43_from_5_l1813_181324


namespace NUMINAMATH_CALUDE_solutions_bounded_above_not_below_l1813_181343

/-- The differential equation y'' = (x^3 + kx)y with initial conditions y(0) = 1 and y'(0) = 0 -/
noncomputable def DiffEq (k : ℝ) (y : ℝ → ℝ) : Prop :=
  (∀ x, (deriv^[2] y) x = (x^3 + k*x) * y x) ∧ y 0 = 1 ∧ (deriv y) 0 = 0

/-- The theorem stating that solutions of y = 0 for the given differential equation
    are bounded above but not below -/
theorem solutions_bounded_above_not_below (k : ℝ) (y : ℝ → ℝ) 
  (h : DiffEq k y) : 
  (∃ M : ℝ, ∀ x : ℝ, y x = 0 → x ≤ M) ∧ 
  (∀ M : ℝ, ∃ x : ℝ, x < M ∧ y x = 0) :=
sorry

end NUMINAMATH_CALUDE_solutions_bounded_above_not_below_l1813_181343


namespace NUMINAMATH_CALUDE_percent_excess_l1813_181383

theorem percent_excess (M N k : ℝ) (hM : M > 0) (hN : N > 0) (hk : k > 0) :
  (M - k * N) / (k * N) * 100 = 100 * (M - k * N) / (k * N) := by
  sorry

end NUMINAMATH_CALUDE_percent_excess_l1813_181383


namespace NUMINAMATH_CALUDE_acute_triangle_perpendicular_pyramid_l1813_181394

theorem acute_triangle_perpendicular_pyramid (a b c : ℝ) 
  (h_acute : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ (x y z : ℝ), 
    x^2 + y^2 = c^2 ∧
    y^2 + z^2 = a^2 ∧
    x^2 + z^2 = b^2 ∧
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_perpendicular_pyramid_l1813_181394


namespace NUMINAMATH_CALUDE_cube_not_square_in_progression_l1813_181371

/-- An arithmetic progression is represented by its first term and common difference -/
structure ArithmeticProgression (α : Type*) [Add α] where
  first : α
  diff : α

/-- Predicate to check if a number is in an arithmetic progression -/
def inProgression (a : ArithmeticProgression ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = a.first + k * a.diff

/-- Predicate to check if a number is a perfect cube -/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

theorem cube_not_square_in_progression (a : ArithmeticProgression ℕ) :
  (∃ n : ℕ, inProgression a n ∧ isPerfectCube n) →
  (∃ m : ℕ, inProgression a m ∧ isPerfectCube m ∧ ¬isPerfectSquare m) :=
by sorry

end NUMINAMATH_CALUDE_cube_not_square_in_progression_l1813_181371


namespace NUMINAMATH_CALUDE_root_between_alpha_beta_l1813_181332

theorem root_between_alpha_beta (p q α β : ℝ) 
  (h_alpha : α^2 + p*α + q = 0)
  (h_beta : -β^2 + p*β + q = 0) :
  ∃ γ : ℝ, (min α β < γ ∧ γ < max α β) ∧ (1/2 * γ^2 + p*γ + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_between_alpha_beta_l1813_181332


namespace NUMINAMATH_CALUDE_two_angles_not_unique_l1813_181344

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ -- length of one leg
  b : ℝ -- length of the other leg
  c : ℝ -- length of the hypotenuse
  angle_A : ℝ -- one acute angle in radians
  angle_B : ℝ -- the other acute angle in radians
  right_angle : c^2 = a^2 + b^2 -- Pythagorean theorem
  acute_angles : angle_A > 0 ∧ angle_A < π/2 ∧ angle_B > 0 ∧ angle_B < π/2
  angle_sum : angle_A + angle_B = π/2 -- sum of acute angles in a right triangle

-- Theorem stating that two acute angles do not uniquely determine a right-angled triangle
theorem two_angles_not_unique (angle1 angle2 : ℝ) 
  (h1 : angle1 > 0 ∧ angle1 < π/2) 
  (h2 : angle2 > 0 ∧ angle2 < π/2) 
  (h3 : angle1 + angle2 = π/2) : 
  ∃ t1 t2 : RightTriangle, t1 ≠ t2 ∧ t1.angle_A = angle1 ∧ t1.angle_B = angle2 ∧
                           t2.angle_A = angle1 ∧ t2.angle_B = angle2 :=
sorry

end NUMINAMATH_CALUDE_two_angles_not_unique_l1813_181344


namespace NUMINAMATH_CALUDE_min_exercise_books_l1813_181376

def total_books : ℕ := 20
def max_cost : ℕ := 60
def exercise_book_cost : ℕ := 2
def notebook_cost : ℕ := 5

theorem min_exercise_books : 
  ∃ (x : ℕ), 
    (x ≤ total_books) ∧ 
    (exercise_book_cost * x + notebook_cost * (total_books - x) ≤ max_cost) ∧
    (∀ (y : ℕ), y < x → 
      exercise_book_cost * y + notebook_cost * (total_books - y) > max_cost) ∧
    x = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_exercise_books_l1813_181376


namespace NUMINAMATH_CALUDE_equation_solution_l1813_181391

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1813_181391


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l1813_181355

theorem logarithm_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l1813_181355


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l1813_181300

-- Define the point (-4, -3)
def point_A : ℝ × ℝ := (-4, -3)

-- Define the x-coordinate of point Q
def x_Q : ℝ := 1

-- Define the distance between Q and point_A
def distance : ℝ := 8

-- Theorem statement
theorem product_of_y_coordinates :
  ∃ (y₁ y₂ : ℝ), 
    (x_Q - point_A.1)^2 + (y₁ - point_A.2)^2 = distance^2 ∧
    (x_Q - point_A.1)^2 + (y₂ - point_A.2)^2 = distance^2 ∧
    y₁ * y₂ = -30 :=
by sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l1813_181300


namespace NUMINAMATH_CALUDE_triangle_count_l1813_181316

/-- The number of distinct triangles that can be formed from 10 points -/
def num_triangles : ℕ := 120

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of vertices in a triangle -/
def triangle_vertices : ℕ := 3

theorem triangle_count :
  Nat.choose num_points triangle_vertices = num_triangles :=
sorry

end NUMINAMATH_CALUDE_triangle_count_l1813_181316


namespace NUMINAMATH_CALUDE_solution_set_l1813_181374

/-- A linear function passing through first, second, and third quadrants -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  intersect_x_axis : 0 = a * (-2) + b

/-- The solution set of ax > b for the given linear function -/
theorem solution_set (f : LinearFunction) : 
  ∀ x : ℝ, f.a * x > f.b ↔ x > -2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_l1813_181374


namespace NUMINAMATH_CALUDE_divisible_by_236_sum_of_middle_digits_l1813_181317

theorem divisible_by_236_sum_of_middle_digits :
  ∀ (a b : ℕ),
  (a < 10 ∧ b < 10) →
  (6000 + 100 * a + 10 * b + 8) % 236 = 0 →
  a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_236_sum_of_middle_digits_l1813_181317


namespace NUMINAMATH_CALUDE_solve_diamond_equation_l1813_181356

-- Define the binary operation ⋄
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Axioms for the binary operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) * c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) : diamond a a = 1

-- Theorem statement
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 504 (diamond 12 x) = 50 → x = 25 / 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_diamond_equation_l1813_181356


namespace NUMINAMATH_CALUDE_geometric_sequences_exist_l1813_181359

/-- Represents a geometric sequence --/
structure GeometricSequence where
  firstTerm : ℝ
  ratio : ℝ

/-- Represents three geometric sequences --/
structure ThreeGeometricSequences where
  seq1 : GeometricSequence
  seq2 : GeometricSequence
  seq3 : GeometricSequence

/-- Checks if the first terms of three geometric sequences form a geometric sequence with ratio 2 --/
def firstTermsFormGeometricSequence (s : ThreeGeometricSequences) : Prop :=
  s.seq2.firstTerm = 2 * s.seq1.firstTerm ∧ s.seq3.firstTerm = 2 * s.seq2.firstTerm

/-- Checks if the ratios of three geometric sequences form an arithmetic sequence with difference 1 --/
def ratiosFormArithmeticSequence (s : ThreeGeometricSequences) : Prop :=
  s.seq2.ratio = s.seq1.ratio + 1 ∧ s.seq3.ratio = s.seq2.ratio + 1

/-- Calculates the sum of the second terms of three geometric sequences --/
def sumOfSecondTerms (s : ThreeGeometricSequences) : ℝ :=
  s.seq1.firstTerm * s.seq1.ratio + s.seq2.firstTerm * s.seq2.ratio + s.seq3.firstTerm * s.seq3.ratio

/-- Calculates the sum of the first three terms of a geometric sequence --/
def sumOfFirstThreeTerms (s : GeometricSequence) : ℝ :=
  s.firstTerm + s.firstTerm * s.ratio + s.firstTerm * s.ratio^2

/-- The main theorem stating the existence of two sets of three geometric sequences satisfying the given conditions --/
theorem geometric_sequences_exist : 
  ∃ (s1 s2 : ThreeGeometricSequences), 
    firstTermsFormGeometricSequence s1 ∧
    firstTermsFormGeometricSequence s2 ∧
    ratiosFormArithmeticSequence s1 ∧
    ratiosFormArithmeticSequence s2 ∧
    sumOfSecondTerms s1 = 24 ∧
    sumOfSecondTerms s2 = 24 ∧
    sumOfFirstThreeTerms s1.seq3 = 84 ∧
    sumOfFirstThreeTerms s2.seq3 = 84 ∧
    s1 ≠ s2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequences_exist_l1813_181359


namespace NUMINAMATH_CALUDE_sisters_age_ratio_l1813_181379

/-- Given John's current age and the future ages of John and his sister,
    prove that the ratio of his sister's age to his age is 2:1 -/
theorem sisters_age_ratio (johns_current_age : ℕ) (johns_future_age : ℕ) (sisters_future_age : ℕ)
  (h1 : johns_current_age = 10)
  (h2 : johns_future_age = 50)
  (h3 : sisters_future_age = 60) :
  (sisters_future_age - (johns_future_age - johns_current_age)) / johns_current_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_sisters_age_ratio_l1813_181379


namespace NUMINAMATH_CALUDE_parallelogram_area_l1813_181396

/-- The area of a parallelogram with sides a and b and angle γ between them is ab sin γ -/
theorem parallelogram_area (a b γ : ℝ) (ha : a > 0) (hb : b > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ S : ℝ, S = a * b * Real.sin γ ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1813_181396


namespace NUMINAMATH_CALUDE_square_triangle_area_l1813_181340

theorem square_triangle_area (x : ℝ) : 
  x > 0 →
  (3 * x) ^ 2 + (4 * x) ^ 2 + (1 / 2) * (3 * x) * (4 * x) = 962 →
  x = Real.sqrt 31 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_area_l1813_181340


namespace NUMINAMATH_CALUDE_f_composition_value_l1813_181378

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else Real.exp (x + 1) - 2

theorem f_composition_value : f (f (1 / Real.exp 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l1813_181378
